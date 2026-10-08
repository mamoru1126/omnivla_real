#include "omnivla/app.hpp"

#include <cmath>

namespace omnivla {

NavConfig load_nav_config(const AppOptions& opt) {
  std::vector<std::string> args;
  if (!opt.robot_config.empty()) args.insert(args.end(), {"--robot", opt.robot_config});
  if (!opt.nav_config.empty()) args.insert(args.end(), {"--nav", opt.nav_config});
  if (!opt.topomap.empty()) args.insert(args.end(), {"--topomap", opt.topomap});
  for (const auto& s : opt.sets) args.insert(args.end(), {"--set", s});
  return nav_config_from_json(run_config_tool(opt.repo_root, args));
}

std::shared_ptr<Topomap> load_topomap_dir(const std::string& repo_root, const std::string& path) {
  return load_topomap(topomap_index_from_json(run_config_tool(repo_root, {"--topomap_only", path})));
}

NavigatorApp::NavigatorApp(const AppOptions& opt, NavRunner::Clock clock, NavRunner::Log log, bool threaded)
    : opt_(opt), log_(log) {
  cfg_ = load_nav_config(opt);
  if (cfg_.model.model != "remote")
    log_("note: the C++ node always uses the policy server (model: " + cfg_.model.model + " runs in the server)");
  auto tm = cfg_.topomap ? load_topomap(*cfg_.topomap) : std::make_shared<Topomap>();
  log_("connecting to the policy server " + (cfg_.model.url.empty() ? std::string("http://127.0.0.1:8765") : cfg_.model.url));
  auto policy = std::make_unique<PolicyClient>(cfg_.model.url, cfg_.model.timeout, 600.0, log_);
  const std::string repo = opt.repo_root;
  runner_ = std::make_unique<NavRunner>(cfg_, std::move(policy), tm, opt.topomap, clock, log_,
                                        [repo](const std::string& p) { return load_topomap_dir(repo, p); }, threaded);
  if (cfg_.io.web_port > 0) {
    try {
      web_ = std::make_unique<WebUi>(cfg_.io.web_host, cfg_.io.web_port, log_);
      web_->set_config(web_config(), runner_->topomap()->images);
      web_topomap_id_ = runner_->topomap()->id;
      web_->on_enable = [this](bool en) {
        if (en) return runner_->start();
        runner_->stop("stopped from the debug page");
        return true;
      };
      runner_->want_preview = [this] { return web_ && web_->has_viewers(); };
    } catch (const std::exception& e) {
      log_(std::string("debug page disabled: ") + e.what());
      web_.reset();
    }
  }
  runner_->on_result = [this](const StepResult& r) {
    if (web_) {
      if (r.state == "running" || r.state == "reached" || r.state == "stopped") {
        const int id = runner_->topomap()->id;
        if (id != web_topomap_id_) {  // topomap が差し替わったら画面の設定も
          web_->set_config(web_config(), runner_->topomap()->images);
          web_topomap_id_ = id;
        }
        web_->publish_step(step_state(r), r.preview);
      }
    }
    if (on_result) on_result(r);
  };
}

NavigatorApp::~NavigatorApp() {
  if (runner_) runner_->shutdown();
  web_.reset();
  runner_.reset();
}

nlohmann::json NavigatorApp::web_config() const {
  const auto& r = cfg_.robot;
  auto tm = runner_->topomap();
  nlohmann::json nodes = nlohmann::json::array();
  for (const auto& n : tm->index.nodes)
    nodes.push_back(n.pose ? nlohmann::json::array({n.pose->x, n.pose->y, n.pose->yaw}) : nlohmann::json(nullptr));
  const auto& info = runner_->policy().info();
  return {{"robot",
           {{"name", r.name},
            {"camera",
             {{"hfov_deg", r.camera.hfov_deg},
              {"height", r.camera.height},
              {"x_offset", r.camera.x_offset},
              {"pitch_deg", r.camera.pitch_deg}}},
            {"image", {{"crop", r.image.crop}, {"width", r.image.width}, {"rotate180", r.image.rotate180}}},
            {"topics", {{"image", r.topics.image}, {"odom", r.topics.odom}, {"localization", r.topics.localization}}}}},
          {"model", {{"server", info.model}, {"history", info.history}, {"url", cfg_.model.url}}},
          {"engine",
           {{"sample_rate", runner_->engine().sample_rate()},
            {"modality", cfg_.engine.modality},
            {"controller", to_json(runner_->engine().config().controller)},
            {"tracker", to_json(cfg_.engine.tracker)}}},
          {"io", {{"control_rate", cfg_.io.control_rate}, {"cmd_timeout", cfg_.io.cmd_timeout}}},
          {"topomap",
           {{"path", runner_->topomap_path()}, {"id", tm->id}, {"num_nodes", tm->size()}, {"frame", tm->index.frame},
            {"nodes", nodes}}}};
}

nlohmann::json NavigatorApp::step_state(const StepResult& r) {
  auto opt = [](const std::optional<double>& v) { return v ? nlohmann::json(*v) : nlohmann::json(nullptr); };
  nlohmann::json wps = nlohmann::json::array();
  if (r.waypoints)
    for (const auto& p : *r.waypoints) wps.push_back({p[0], p[1], std::atan2(p[3], p[2])});
  auto st = runner_->status_json();
  st["t"] = r.t;
  st["step"] = r.step;
  st["state"] = r.state;
  st["subgoal"] = r.subgoal;
  st["num_nodes"] = r.num_nodes;
  st["similarity"] = opt(r.similarity);
  st["distance"] = opt(r.distance);
  st["latency"] = r.latency;
  st["v"] = r.v;
  st["w"] = r.w;
  st["waypoints"] = wps;
  for (const auto& e : r.events) {
    event_log_.emplace_back(++event_id_, e);
    while (event_log_.size() > 30) event_log_.pop_front();
  }
  nlohmann::json log = nlohmann::json::array();
  for (const auto& e : event_log_) log.push_back({e.first, e.second});
  st["event_log"] = log;
  st["reason"] = r.reason;
  st["preview"] = static_cast<bool>(r.preview);
  st["preview_size"] = {r.preview_width, r.preview_height};
  st["frame_stamp"] = r.frame ? r.frame->stamp : 0.0;
  st["topomap_id"] = runner_->topomap()->id;
  return st;
}

void NavigatorApp::on_command(double t, const std::optional<std::pair<double, double>>& cmd) {
  if (!web_ || !web_->has_viewers()) return;
  if (t - last_web_status_ < 0.2 && t >= last_web_status_) return;  // 5Hz
  last_web_status_ = t;
  auto st = runner_->status_json();
  st["cmd_v"] = cmd ? nlohmann::json(cmd->first) : nlohmann::json(nullptr);
  st["cmd_w"] = cmd ? nlohmann::json(cmd->second) : nlohmann::json(nullptr);
  st["cmd_t"] = t;
  // status_json の v, w は最後の推論の値なので、画面の「指示値」には cmd_v / cmd_w を使う
  st.erase("v");
  st.erase("w");
  st.erase("latency");
  web_->publish_status(st);
}

}  // namespace omnivla
