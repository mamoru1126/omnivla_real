#include "omnivla/web_ui.hpp"

#include <algorithm>
#include <chrono>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <sstream>

#include <httplib.h>

namespace omnivla {

namespace {

std::string index_html() {
  // 開発用: OMNIVLA_WEB_DIR を指定すると、ビルドし直さずに web/index.html を読み直す
  const char* dir = std::getenv("OMNIVLA_WEB_DIR");
  if (dir && *dir) {
    std::ifstream f(std::string(dir) + "/index.html");
    if (f) {
      std::stringstream ss;
      ss << f.rdbuf();
      return ss.str();
    }
  }
  return kIndexHtml;
}

}  // namespace

WebUi::WebUi(const std::string& host, int port, Log log) : srv_(std::make_unique<httplib::Server>()), log_(std::move(log)) {
  auto& s = *srv_;
  s.Get("/", [](const httplib::Request&, httplib::Response& res) {
    res.set_header("Cache-Control", "no-cache");
    res.set_content(index_html(), "text/html; charset=utf-8");
  });
  s.Get("/api/config", [this](const httplib::Request&, httplib::Response& res) {
    std::lock_guard<std::mutex> lk(mu_);
    res.set_header("Cache-Control", "no-cache");
    res.set_content(config_.dump(), "application/json");
  });
  s.Get("/api/state", [this](const httplib::Request&, httplib::Response& res) {
    std::lock_guard<std::mutex> lk(mu_);
    nlohmann::json m = step_;
    for (auto it = status_.begin(); it != status_.end(); ++it) m[it.key()] = it.value();
    if (!m.is_null() && !m.empty()) m["kind"] = step_.empty() ? "status" : "step";
    res.set_header("Cache-Control", "no-cache");
    res.set_content(m.dump(), "application/json");
  });
  s.Get("/api/events", [this](const httplib::Request&, httplib::Response& res) {
    ++viewers_;
    auto last = std::make_shared<std::pair<uint64_t, uint64_t>>(0, 0);  // 送った step / status の番号
    res.set_header("Cache-Control", "no-cache");
    res.set_header("X-Accel-Buffering", "no");
    res.set_chunked_content_provider(
        "text/event-stream",
        [this, last](size_t, httplib::DataSink& sink) {
          std::string msg;
          {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait_for(lk, std::chrono::seconds(1), [&] {
              return step_seq_ != last->first || status_seq_ != last->second || stopping_;
            });
            if (stopping_) return false;
            // 推論の結果は取りこぼさない (状態の更新に上書きされない). 番号の小さい順に送る
            std::vector<std::pair<uint64_t, const nlohmann::json*>> out;
            if (step_seq_ != last->first) out.emplace_back(step_seq_, &step_);
            if (status_seq_ != last->second) out.emplace_back(status_seq_, &status_);
            std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            for (const auto& o : out) msg += "data: " + o.second->dump() + "\n\n";
            last->first = step_seq_;
            last->second = status_seq_;
            if (msg.empty()) msg = ": ping\n\n";  // 接続が切れたことに気付くため
          }
          return sink.write(msg.data(), msg.size());
        },
        [this](bool) { --viewers_; });
  });
  s.Get("/img/current.jpg", [this](const httplib::Request&, httplib::Response& res) {
    Bytes b;
    {
      std::lock_guard<std::mutex> lk(mu_);
      b = preview_;
    }
    if (!b) {
      res.status = 404;
      return;
    }
    res.set_header("Cache-Control", "no-store");
    res.set_content(reinterpret_cast<const char*>(b->data()), b->size(), "image/jpeg");
  });
  s.Get(R"(/img/goal/(\d+)\.jpg)", [this](const httplib::Request& req, httplib::Response& res) {
    FramePtr f;
    {
      std::lock_guard<std::mutex> lk(mu_);
      const size_t k = std::stoul(req.matches[1]);
      if (k < goals_.size()) f = goals_[k];
    }
    if (!f || !f->data) {
      res.status = 404;
      return;
    }
    res.set_header("Cache-Control", "max-age=3600");
    res.set_content(reinterpret_cast<const char*>(f->data->data()), f->data->size(),
                    f->format == "png" ? "image/png" : "image/jpeg");
  });
  s.Post("/api/enable", [this](const httplib::Request& req, httplib::Response& res) {
    bool en = false;
    try {
      const auto j = nlohmann::json::parse(req.body);
      en = j.value("enable", false);
    } catch (...) {
      res.status = 400;
      return;
    }
    const bool ok = on_enable ? on_enable(en) : false;
    res.set_content(nlohmann::json({{"ok", ok}, {"enable", en}}).dump(), "application/json");
  });
  if (!s.bind_to_port(host, port)) throw std::runtime_error("web ui: cannot listen on " + host + ":" + std::to_string(port));
  th_ = std::thread([this] { srv_->listen_after_bind(); });
  if (log_) log_("debug page: http://" + (host == "0.0.0.0" ? std::string("<this machine>") : host) + ":" + std::to_string(port));
}

WebUi::~WebUi() {
  stopping_ = true;
  cv_.notify_all();
  srv_->stop();
  if (th_.joinable()) th_.join();
}

void WebUi::set_config(const nlohmann::json& config, const std::vector<FramePtr>& goals) {
  std::lock_guard<std::mutex> lk(mu_);
  config_ = config;
  goals_ = goals;
}

void WebUi::publish_step(nlohmann::json state, const Bytes& preview) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (preview) preview_ = preview;
    state["kind"] = "step";
    state["seq"] = ++seq_;
    step_ = std::move(state);
    step_seq_ = seq_;
  }
  cv_.notify_all();
}

void WebUi::publish_status(nlohmann::json status) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    status["kind"] = "status";
    status["seq"] = ++seq_;
    status_ = std::move(status);
    status_seq_ = seq_;
  }
  cv_.notify_all();
}

}  // namespace omnivla
