#include "omnivla/policy_client.hpp"

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#include <httplib.h>

namespace omnivla {

namespace {

constexpr size_t kClientKnown = 48;  // サーバが覚えているとみなす枚数 (サーバは 64 枚: remote.py の IMAGE_CACHE)

struct Url {
  std::string host = "127.0.0.1";
  int port = 8765;
};

Url parse_url(std::string url) {
  Url u;
  const auto scheme = url.find("://");
  if (scheme != std::string::npos) url = url.substr(scheme + 3);
  const auto slash = url.find('/');
  if (slash != std::string::npos) url = url.substr(0, slash);
  const auto colon = url.rfind(':');
  if (colon != std::string::npos) {
    u.host = url.substr(0, colon);
    u.port = std::stoi(url.substr(colon + 1));
  } else if (!url.empty()) {
    u.host = url;
  }
  return u;
}

// [4 byte big endian: JSON の長さ][JSON][配列のバイト列]
std::string pack(json header, const std::vector<const std::vector<uint8_t>*>& arrays) {
  json shapes = json::array();
  for (const auto* a : arrays) shapes.push_back(json::array({json::array({a->size()}), "|u1"}));
  header["arrays"] = shapes;
  const std::string h = header.dump();
  std::string out;
  size_t total = 4 + h.size();
  for (const auto* a : arrays) total += a->size();
  out.reserve(total);
  const uint32_t n = static_cast<uint32_t>(h.size());
  out.push_back(static_cast<char>((n >> 24) & 0xff));
  out.push_back(static_cast<char>((n >> 16) & 0xff));
  out.push_back(static_cast<char>((n >> 8) & 0xff));
  out.push_back(static_cast<char>(n & 0xff));
  out += h;
  for (const auto* a : arrays) out.append(reinterpret_cast<const char*>(a->data()), a->size());
  return out;
}

size_t itemsize(const std::string& dtype) {
  // "<f8" "<f4" "|u1" "<i8" ...
  return static_cast<size_t>(std::stoi(dtype.substr(2)));
}

Waypoints to_waypoints(const json& shape, const std::string& bytes) {
  const size_t rows = shape.at(0).get<size_t>(), cols = shape.size() > 1 ? shape.at(1).get<size_t>() : 1;
  if (bytes.size() != rows * cols * sizeof(double)) throw std::runtime_error("bad waypoint array");
  Waypoints w(rows);
  const double* d = reinterpret_cast<const double*>(bytes.data());
  for (size_t r = 0; r < rows; ++r)
    for (size_t c = 0; c < 4 && c < cols; ++c) w[r][c] = d[r * cols + c];
  return w;
}

}  // namespace

PolicyClient::PolicyClient(const std::string& url, double timeout, double wait,
                           const std::function<void(const std::string&)>& log) {
  const Url u = parse_url(url.empty() ? "http://127.0.0.1:8765" : url);
  cli_ = std::make_unique<httplib::Client>(u.host, u.port);
  cli_->set_keep_alive(true);
  cli_->set_tcp_nodelay(true);  // 小さな要求を待たせない
  cli_->set_connection_timeout(2, 0);
  const auto to = std::chrono::duration<double>(timeout);
  cli_->set_read_timeout(std::chrono::duration_cast<std::chrono::microseconds>(to));
  cli_->set_write_timeout(std::chrono::duration_cast<std::chrono::microseconds>(to));
  const auto t_end = std::chrono::steady_clock::now() + std::chrono::duration<double>(wait);
  auto last_log = std::chrono::steady_clock::now() - std::chrono::seconds(60);
  while (true) {
    auto res = cli_->Get("/info");
    if (res && res->status == 200) {
      const json j = json::parse(res->body);
      info_.model = j.value("model", "?");
      info_.history = j.value("history", false);
      info_.context_size = j.value("context_size", 0);
      info_.context_stride = std::max(1, j.value("context_stride", 1));
      info_.metric_spacing = j.value("metric_spacing", 0.0);
      if (j.contains("meta") && j["meta"].is_object()) info_.meta = j["meta"];
      if (j.value("protocol", 1) < 2)
        throw std::runtime_error("policy server is too old (protocol < 2). update omnivla_real on the server side");
      break;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now > t_end) throw std::runtime_error("policy server " + u.host + ":" + std::to_string(u.port) + " is not reachable");
    if (now - last_log > std::chrono::seconds(10)) {
      if (log) log("waiting for policy server " + u.host + ":" + std::to_string(u.port) + " ...");
      last_log = now;
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  if (log)
    log("policy server: " + info_.model + " at " + u.host + ":" + std::to_string(u.port) +
        " (history=" + (info_.history ? "yes" : "no") + ")");
}

PolicyClient::~PolicyClient() = default;

PolicyClient::Response PolicyClient::post(const std::string& path, const std::string& body, int* status) {
  auto res = cli_->Post(path, body, "application/octet-stream");
  if (!res) throw std::runtime_error("policy server: " + httplib::to_string(res.error()));
  *status = res->status;
  Response out;
  const std::string& b = res->body;
  if (b.size() < 4) throw std::runtime_error("policy server: short response (" + std::to_string(res->status) + ")");
  const uint32_t n = (static_cast<uint8_t>(b[0]) << 24) | (static_cast<uint8_t>(b[1]) << 16) |
                     (static_cast<uint8_t>(b[2]) << 8) | static_cast<uint8_t>(b[3]);
  out.header = json::parse(b.substr(4, n));
  size_t off = 4 + n;
  if (out.header.contains("arrays")) {
    for (const auto& a : out.header["arrays"]) {
      const json shape = a.at(0);
      size_t count = 1;
      for (const auto& d : shape) count *= d.get<size_t>();
      const size_t size = count * itemsize(a.at(1).get<std::string>());
      if (off + size > b.size()) throw std::runtime_error("policy server: truncated response");
      out.arrays.emplace_back(shape, b.substr(off, size));
      off += size;
    }
  }
  return out;
}

PolicyClient::Response PolicyClient::call(const std::string& path, json header, const std::vector<FramePtr>& images) {
  std::lock_guard<std::mutex> lk(mu_);
  for (int attempt = 0; attempt < 2; ++attempt) {
    json spec = json::array();
    std::vector<const std::vector<uint8_t>*> arrays;
    for (const auto& im : images) {
      if (known_.count(im->key)) {
        spec.push_back({{"key", im->key}});
        continue;
      }
      json e = {{"key", im->key}, {"array", arrays.size()}, {"kind", im->kind}, {"preprocess", im->preprocess}};
      if (im->kind == "encoded") {
        e["format"] = im->format;
      } else {
        e["encoding"] = im->encoding;
        e["width"] = im->width;
        e["height"] = im->height;
        e["step"] = im->step;
        e["is_bigendian"] = im->is_bigendian ? 1 : 0;
      }
      spec.push_back(e);
      arrays.push_back(im->data.get());
    }
    header["images"] = spec;
    const std::string body = pack(header, arrays);
    int status = 0;
    Response r = post(path, body, &status);
    if (status == 409 && attempt == 0) {  // サーバが覚えていない (再起動など): 全部送り直す
      known_.clear();
      known_order_.clear();
      continue;
    }
    if (status != 200) throw std::runtime_error("policy server error (" + std::to_string(status) + "): " +
                                                r.header.value("error", std::string("?")));
    bytes_sent_ += body.size();
    for (const auto& im : images) {
      if (known_.insert(im->key).second) known_order_.push_back(im->key);
    }
    while (known_order_.size() > kClientKnown) {
      known_.erase(known_order_.front());
      known_order_.pop_front();
    }
    return r;
  }
  throw std::runtime_error("policy server: images missing after resend");
}

PredictResult PolicyClient::predict(const FramePtr& current, const FramePtr& goal,
                                    const std::vector<FramePtr>& observations, const std::optional<Pose>& goal_pose,
                                    const std::string& modality, bool want_preview) {
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<FramePtr> images;
  std::unordered_map<const Frame*, int> index;
  auto add = [&](const FramePtr& f) {
    auto it = index.find(f.get());
    if (it != index.end()) return it->second;
    const int i = static_cast<int>(images.size());
    index[f.get()] = i;
    images.push_back(f);
    return i;
  };
  json h = {{"current", add(current)}, {"modality", modality}, {"instruction", nullptr}};
  h["goal"] = goal ? json(add(goal)) : json(nullptr);
  h["goal_pose"] = goal_pose ? json::array({goal_pose->x, goal_pose->y, goal_pose->yaw}) : json(nullptr);
  if (info_.history) {
    json obs = json::array();
    for (const auto& o : observations) obs.push_back(add(o));
    h["observations"] = obs;
  }
  if (want_preview) h["want_preview"] = true;
  Response r = call("/predict", h, images);
  PredictResult out;
  out.waypoints = to_waypoints(r.arrays.at(0).first, r.arrays.at(0).second);
  out.normalized = to_waypoints(r.arrays.at(1).first, r.arrays.at(1).second);
  out.modality = r.header.value("modality", 0);
  out.server_latency = r.header.value("latency", 0.0);
  if (r.header.contains("distance") && r.header["distance"].is_number()) out.distance = r.header["distance"].get<double>();
  if (r.header.contains("preview") && r.header["preview"].is_number()) {
    const auto& p = r.arrays.at(r.header["preview"].get<size_t>()).second;
    out.preview = std::make_shared<std::vector<uint8_t>>(p.begin(), p.end());
    if (r.header.contains("preview_size")) {
      out.preview_width = r.header["preview_size"][0].get<int>();
      out.preview_height = r.header["preview_size"][1].get<int>();
    }
  }
  out.latency = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return out;
}

std::vector<float> PolicyClient::embed(const FramePtr& image) {
  Response r = call("/embed", {{"image", 0}}, {image});
  const auto& b = r.arrays.at(0).second;
  std::vector<float> v(b.size() / sizeof(float));
  std::memcpy(v.data(), b.data(), v.size() * sizeof(float));
  return v;
}

}  // namespace omnivla
