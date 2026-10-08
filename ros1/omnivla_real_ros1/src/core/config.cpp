#include "omnivla/config.hpp"

#include <sys/wait.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace omnivla {

namespace {

template <typename T>
void get(const json& j, const char* key, T& out) {
  auto it = j.find(key);
  if (it != j.end() && !it->is_null()) out = it->get<T>();
}

std::optional<Pose> pose_of(const json& j) {
  if (j.is_null()) return std::nullopt;
  if (j.is_array() && j.size() >= 2) {
    return Pose{j[0].get<double>(), j[1].get<double>(), j.size() > 2 ? j[2].get<double>() : 0.0};
  }
  if (j.is_object() && j.contains("x")) {
    return Pose{j["x"].get<double>(), j["y"].get<double>(), j.value("yaw", 0.0)};
  }
  return std::nullopt;
}

std::string shell_quote(const std::string& s) {
  std::string out = "'";
  for (char c : s) {
    if (c == '\'')
      out += "'\\''";
    else
      out += c;
  }
  return out + "'";
}

}  // namespace

TopomapIndex topomap_index_from_json(const json& j) {
  TopomapIndex t;
  get(j, "directory", t.directory);
  get(j, "frame", t.frame);
  if (j.contains("start")) t.start = pose_of(j["start"]);
  if (j.contains("meta") && j["meta"].is_object()) t.meta = j["meta"];
  for (const auto& n : j.at("nodes")) {
    TopomapNode node;
    node.path = n.at("path").get<std::string>();
    if (n.contains("pose")) node.pose = pose_of(n["pose"]);
    if (n.contains("s") && !n["s"].is_null()) node.s = n["s"].get<double>();
    t.nodes.push_back(node);
  }
  return t;
}

NavConfig nav_config_from_json(const json& j) {
  NavConfig c;
  c.raw = j;
  if (j.contains("robot")) {
    const auto& r = j["robot"];
    get(r, "name", c.robot.name);
    get(r, "time_source", c.robot.time_source);
    if (r.contains("topics")) {
      const auto& t = r["topics"];
      get(t, "image", c.robot.topics.image);
      get(t, "odom", c.robot.topics.odom);
      get(t, "cmd", c.robot.topics.cmd);
      get(t, "localization", c.robot.topics.localization);
      get(t, "localization_frame", c.robot.topics.localization_frame);
      get(t, "localization_child_frame", c.robot.topics.localization_child_frame);
    }
    if (r.contains("image")) {
      const auto& im = r["image"];
      if (im.contains("crop")) {
        auto v = im["crop"].get<std::vector<double>>();
        if (v.size() != 4) throw std::runtime_error("image.crop must be [top, bottom, left, right]");
        for (int i = 0; i < 4; ++i) c.robot.image.crop[i] = v[i];
      }
      get(im, "width", c.robot.image.width);
      get(im, "rotate180", c.robot.image.rotate180);
      get(im, "jpeg_quality", c.robot.image.jpeg_quality);
    }
    if (r.contains("camera")) {
      const auto& cam = r["camera"];
      get(cam, "hfov_deg", c.robot.camera.hfov_deg);
      get(cam, "height", c.robot.camera.height);
      get(cam, "x_offset", c.robot.camera.x_offset);
      get(cam, "pitch_deg", c.robot.camera.pitch_deg);
    }
  }
  if (j.contains("model")) {
    const auto& m = j["model"];
    get(m, "model", c.model.model);
    get(m, "url", c.model.url);
    get(m, "timeout", c.model.timeout);
  }
  if (j.contains("engine")) {
    const auto& e = j["engine"];
    get(e, "modality", c.engine.modality);
    get(e, "sample_rate", c.engine.sample_rate);
    get(e, "max_image_age", c.engine.max_image_age);
    get(e, "stuck_timeout", c.engine.stuck_timeout);
    get(e, "stop_at_goal", c.engine.stop_at_goal);
    get(e, "debug_image", c.engine.debug_image);
    if (e.contains("controller")) {
      const auto& k = e["controller"];
      auto& o = c.engine.controller;
      get(k, "mode", o.mode);
      get(k, "track_horizon", o.track_horizon);
      get(k, "track_max_v", o.track_max_v);
      get(k, "track_max_w", o.track_max_w);
      get(k, "track_phi_scale", o.track_phi_scale);
      get(k, "waypoint_index", o.waypoint_index);
      get(k, "dt", o.dt);
      get(k, "max_linear_raw", o.max_linear_raw);
      get(k, "max_angular_raw", o.max_angular_raw);
      get(k, "max_v", o.max_v);
      get(k, "max_w", o.max_w);
      get(k, "respect_predicted_speed", o.respect_predicted_speed);
      get(k, "lookahead", o.lookahead);
      get(k, "pp_speed", o.pp_speed);
      get(k, "pp_max_w", o.pp_max_w);
      get(k, "rotate_in_place_angle", o.rotate_in_place_angle);
    }
    if (e.contains("tracker")) {
      const auto& k = e["tracker"];
      auto& o = c.engine.tracker;
      get(k, "reach_check", o.reach_check);
      get(k, "subgoal_radius", o.subgoal_radius);
      get(k, "goal_radius", o.goal_radius);
      get(k, "reach_angle_deg", o.reach_angle_deg);
      get(k, "pass_radius", o.pass_radius);
      get(k, "pass_angle_deg", o.pass_angle_deg);
      get(k, "image_threshold", o.image_threshold);
      get(k, "goal_image_threshold", o.goal_image_threshold);
      get(k, "search_window", o.search_window);
      get(k, "confirm", o.confirm);
      get(k, "odom_gate_m", o.odom_gate_m);
      get(k, "odom_pass_m", o.odom_pass_m);
      get(k, "goal_odom_gate_m", o.goal_odom_gate_m);
      get(k, "min_travel_m", o.min_travel_m);
    }
  }
  if (j.contains("io")) {
    const auto& k = j["io"];
    auto& o = c.io;
    get(k, "cmd_vel", o.cmd_vel);
    get(k, "cmd_stamped", o.cmd_stamped);
    get(k, "base_frame", o.base_frame);
    get(k, "path", o.path);
    get(k, "debug_image", o.debug_image);
    get(k, "status", o.status);
    get(k, "enable", o.enable);
    get(k, "topomap", o.topomap);
    get(k, "localization_type", o.localization_type);
    get(k, "control_rate", o.control_rate);
    get(k, "cmd_timeout", o.cmd_timeout);
    get(k, "max_inference_rate", o.max_inference_rate);
    get(k, "log_dir", o.log_dir);
    get(k, "log_images", o.log_images);
    get(k, "web_port", o.web_port);
    get(k, "web_host", o.web_host);
  }
  if (j.contains("topomap") && !j["topomap"].is_null()) c.topomap = topomap_index_from_json(j["topomap"]);
  return c;
}

json to_json(const ControllerConfig& o) {
  return {{"mode", o.mode},
          {"track_horizon", o.track_horizon},
          {"track_max_v", o.track_max_v},
          {"track_max_w", o.track_max_w},
          {"track_phi_scale", o.track_phi_scale},
          {"waypoint_index", o.waypoint_index},
          {"dt", o.dt},
          {"max_v", o.max_v},
          {"max_w", o.max_w},
          {"respect_predicted_speed", o.respect_predicted_speed}};
}

json to_json(const TrackerConfig& o) {
  return {{"reach_check", o.reach_check},       {"subgoal_radius", o.subgoal_radius},
          {"goal_radius", o.goal_radius},       {"reach_angle_deg", o.reach_angle_deg},
          {"pass_radius", o.pass_radius},       {"pass_angle_deg", o.pass_angle_deg},
          {"image_threshold", o.image_threshold}, {"goal_image_threshold", o.goal_image_threshold},
          {"search_window", o.search_window},   {"confirm", o.confirm},
          {"odom_gate_m", o.odom_gate_m},       {"odom_pass_m", o.odom_pass_m},
          {"goal_odom_gate_m", o.goal_odom_gate_m}, {"min_travel_m", o.min_travel_m}};
}

json run_config_tool(const std::string& repo_root, const std::vector<std::string>& args) {
  const char* py = std::getenv("OMNIVLA_PYTHON");
  std::string cmd = shell_quote(py && *py ? py : "python3") + " " + shell_quote(repo_root + "/tools/nav_config_json.py");
  for (const auto& a : args) cmd += " " + shell_quote(a);
  FILE* f = popen(cmd.c_str(), "r");
  if (!f) throw std::runtime_error("cannot run: " + cmd);
  std::string out;
  char buf[65536];
  size_t n;
  while ((n = fread(buf, 1, sizeof(buf), f)) > 0) out.append(buf, n);
  int status = pclose(f);
  if (status != 0 || out.empty()) {
    throw std::runtime_error("config tool failed (exit " + std::to_string(WEXITSTATUS(status)) + "): " + cmd);
  }
  return json::parse(out);
}

Bytes read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + path);
  auto v = std::make_shared<std::vector<uint8_t>>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  return v;
}

std::shared_ptr<Topomap> load_topomap(const TopomapIndex& index) {
  static int next_id = 1;
  auto t = std::make_shared<Topomap>();
  t->index = index;
  t->id = next_id++;
  for (size_t k = 0; k < index.nodes.size(); ++k) {
    const auto& path = index.nodes[k].path;
    auto f = std::make_shared<Frame>();
    f->key = "g" + std::to_string(t->id) + "_" + std::to_string(k);
    f->kind = "encoded";
    const auto dot = path.find_last_of('.');
    std::string ext = dot == std::string::npos ? "" : path.substr(dot + 1);
    for (auto& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    f->format = ext == "png" ? "png" : "jpeg";
    f->preprocess = false;  // topomap の画像は作るときに前処理済み
    f->data = read_file(path);
    t->images.push_back(f);
  }
  return t;
}

}  // namespace omnivla
