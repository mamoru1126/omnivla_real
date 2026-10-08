#include "omnivla/runner.hpp"

#include <chrono>

namespace omnivla {

NavRunner::NavRunner(const NavConfig& cfg, std::unique_ptr<Policy> policy, std::shared_ptr<Topomap> topomap,
                     const std::string& topomap_path, Clock clock, Log log, TopomapLoader loader, bool threaded)
    : cfg_(cfg),
      policy_(std::move(policy)),
      topomap_path_(topomap_path),
      clock_(std::move(clock)),
      log_(std::move(log)),
      loader_(std::move(loader)),
      threaded_(threaded) {
  engine_ = std::make_unique<NavEngine>(policy_.get(), std::move(topomap), cfg_.engine);
  log_("ready: model=" + policy_->info().model + ", sample_rate=" + std::to_string(engine_->sample_rate()) +
       " Hz, topomap=" + (topomap_path_.empty() ? "(none)" : topomap_path_) + " (" +
       std::to_string(engine_->topomap()->size()) + " subgoals)");
}

NavRunner::~NavRunner() {
  shutdown();
  if (thread_.joinable()) thread_.join();
}

std::unique_ptr<NavLogger> NavRunner::make_logger() {
  if (cfg_.io.log_dir.empty()) return nullptr;
  auto tm = engine_->topomap();
  nlohmann::json final_goal = nullptr;
  if (tm->size() && tm->index.nodes.back().pose) {
    const auto& p = *tm->index.nodes.back().pose;
    final_goal = {p.x, p.y, p.yaw};
  }
  nlohmann::json model = cfg_.raw.value("model", nlohmann::json::object());
  model["server_model"] = policy_->info().model;
  nlohmann::json meta = {
      {"topomap", topomap_path_},
      {"topomap_meta", tm->index.meta},
      {"num_nodes", tm->size()},
      {"final_goal_pose", final_goal},
      {"model", model},
      {"robot", cfg_.raw.value("robot", nlohmann::json::object())},
      {"engine",
       {{"modality", cfg_.engine.modality},
        {"sample_rate", engine_->sample_rate()},
        {"controller", to_json(engine_->config().controller)},
        {"tracker", to_json(cfg_.engine.tracker)}}},
      {"node", "cpp"}};
  try {
    return std::make_unique<NavLogger>(cfg_.io.log_dir, meta, tm->images, cfg_.io.log_images);
  } catch (const std::exception& e) {
    log_(std::string("cannot write logs to ") + cfg_.io.log_dir + ": " + e.what());
    return nullptr;
  }
}

bool NavRunner::start(const std::string& topomap_path) {
  std::string mode;
  {
    std::lock_guard<std::recursive_mutex> lk(mu_);
    std::shared_ptr<Topomap> tm;
    if (!topomap_path.empty()) {
      if (!loader_) {
        log_("cannot load a topomap at runtime (no loader)");
        return false;
      }
      try {
        tm = loader_(topomap_path);
      } catch (const std::exception& e) {
        log_(std::string("cannot load topomap ") + topomap_path + ": " + e.what());
        return false;
      }
      topomap_path_ = topomap_path;
    }
    const size_t n = tm ? tm->size() : engine_->topomap()->size();
    if (n == 0) {
      log_("no topomap. publish a directory to the topomap topic");
      return false;
    }
    if (logger_) logger_->close("restarted");
    if (tm) engine_->set_topomap(tm);  // 先に差し替えてからログを作る (goals/ に新しい画像を保存するため)
    logger_ = make_logger();
    engine_->logger = logger_.get();
    mode = engine_->start(nullptr);
    std::lock_guard<std::mutex> lc(cmd_mu_);
    cmd_ = {0.0, 0.0};
    cmd_time_ = -1e9;
  }
  log_("start: " + std::to_string(engine_->topomap()->size()) + " subgoals, reach_check=" + mode +
       ", log=" + (logger_ ? logger_->dir() : std::string("-")));
  ensure_thread();
  return true;
}

void NavRunner::stop(const std::string& reason) {
  {  // 推論中でもすぐに 0 を出す
    std::lock_guard<std::mutex> lc(cmd_mu_);
    stopping_ = true;
    cmd_ = {0.0, 0.0};
    stopped_at_ = clock_();
  }
  {
    std::lock_guard<std::recursive_mutex> lk(mu_);
    engine_->stop(reason);
    if (logger_) logger_->close(reason);
  }
  {
    std::lock_guard<std::mutex> lc(cmd_mu_);
    stopping_ = false;
    stopped_at_ = clock_();
  }
  log_("stop: " + reason);
}

void NavRunner::shutdown() {
  if (!alive_.exchange(false)) return;
  if (engine_ && engine_->state() == "running") stop("shutdown");
}

void NavRunner::ensure_thread() {
  if (threaded_ && !thread_.joinable()) thread_ = std::thread([this] { loop(); });
}

std::optional<StepResult> NavRunner::step_once() {
  if (engine_->state() != "running" || !engine_->has_new_image()) return std::nullopt;
  StepResult res;
  {
    std::lock_guard<std::recursive_mutex> lk(mu_);
    const bool preview = want_preview && want_preview();
    res = engine_->step(clock_(), preview);
    std::lock_guard<std::mutex> lc(cmd_mu_);
    if (res.state == "running" && !stopping_) {
      cmd_ = {res.v, res.w};
      cmd_time_ = clock_();
    } else if (res.state == "reached" || res.state == "stopped") {
      cmd_ = {0.0, 0.0};
      stopped_at_ = clock_();
    }
  }
  for (const auto& e : res.events) log_(e);
  if (res.state == "reached" || res.state == "stopped") log_("finished: " + res.state + " " + res.reason);
  if (on_result) on_result(res);
  return res;
}

void NavRunner::loop() {
  using namespace std::chrono;
  const double rate = cfg_.io.max_inference_rate;
  const auto period = rate > 0 ? duration<double>(1.0 / rate) : duration<double>(0.0);
  while (alive_) {
    if (engine_->state() != "running") {
      std::this_thread::sleep_for(milliseconds(50));
      continue;
    }
    if (!engine_->has_new_image()) {  // 新しい画像が来るまで待つ (同じ画像で推論し直さない)
      std::this_thread::sleep_for(milliseconds(5));
      continue;
    }
    const auto t0 = steady_clock::now();
    std::optional<StepResult> res;
    try {
      res = step_once();
    } catch (const std::exception& e) {  // 推論の失敗でノードごと落ちないようにする
      log_(std::string("ERROR in step: ") + e.what());
      {
        std::lock_guard<std::mutex> lc(cmd_mu_);
        cmd_ = {0.0, 0.0};
      }
      std::this_thread::sleep_for(milliseconds(500));
      continue;
    }
    const auto wait = period - (steady_clock::now() - t0);
    if (wait.count() > 0) std::this_thread::sleep_for(wait);
    if (res && res->state == "waiting_image") std::this_thread::sleep_for(milliseconds(50));
  }
}

std::optional<std::pair<double, double>> NavRunner::command() {
  const std::string state = engine_->state();
  std::lock_guard<std::mutex> lc(cmd_mu_);
  const double now = clock_();
  if (state == "running") {
    if (stopping_) return std::make_pair(0.0, 0.0);
    if (now - cmd_time_ <= cfg_.io.cmd_timeout) return cmd_;
    return std::make_pair(0.0, 0.0);
  }
  if (now - stopped_at_ < 1.0) return std::make_pair(0.0, 0.0);
  return std::nullopt;
}

nlohmann::json NavRunner::status_json() {
  auto st = engine_->status();
  st["topomap"] = topomap_path_;
  return st;
}

}  // namespace omnivla
