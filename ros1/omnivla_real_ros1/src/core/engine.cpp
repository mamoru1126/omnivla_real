#include "omnivla/engine.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>

#include "omnivla/geometry.hpp"
#include "omnivla/navlog.hpp"

namespace omnivla {

namespace {

std::string fmt_double(double v) {
  // Python の str(float) に近い表示 (3.0 -> "3.0")
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.15g", v);
  std::string s = buf;
  if (s.find_first_of(".eE") == std::string::npos) s += ".0";
  return s;
}

double cosine(const std::vector<float>& a, const std::vector<float>& b) {
  double ab = 0.0, aa = 0.0, bb = 0.0;
  for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
    ab += static_cast<double>(a[i]) * b[i];
    aa += static_cast<double>(a[i]) * a[i];
    bb += static_cast<double>(b[i]) * b[i];
  }
  return ab / (std::sqrt(aa) * std::sqrt(bb) + 1e-9);
}

nlohmann::json opt(const std::optional<double>& v) { return v ? nlohmann::json(*v) : nlohmann::json(nullptr); }

double wall_time() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

}  // namespace

NavEngine::NavEngine(Policy* policy, std::shared_ptr<Topomap> topomap, const EngineConfig& cfg)
    : policy_(policy), map_(std::move(topomap)), cfg_(cfg) {
  double meta_rate = 0.0;
  const auto& meta = policy_->info().meta;
  if (meta.contains("sample_rate") && meta["sample_rate"].is_number()) meta_rate = meta["sample_rate"].get<double>();
  sample_rate_ = cfg_.sample_rate > 0 ? cfg_.sample_rate : (meta_rate > 0 ? meta_rate : 3.0);
  cfg_.controller.dt = 1.0 / sample_rate_;  // waypoint k は (k+1)/sample_rate 秒後
  if (cfg_.stuck_timeout > 0) stuck_ = std::make_unique<StuckDetector>(cfg_.stuck_timeout);
  if (policy_->info().history)
    history_len_ = static_cast<size_t>(policy_->info().context_size * policy_->info().context_stride + 1);
  if (!map_) map_ = std::make_shared<Topomap>();
}

void NavEngine::on_image(double t, FramePtr frame) {
  std::lock_guard<std::mutex> lk(in_mu_);
  latest_ = std::move(frame);
  img_t_ = t;
  ++img_seq_;
  // 観測履歴 (edge) は sample_rate ごとに積む (engine.py と同じ間隔の決め方)
  if (!next_hist_ || t + 1e-6 >= *next_hist_) {
    const double period = 1.0 / sample_rate_;
    if (next_hist_ && *next_hist_ != 0.0 && t - *next_hist_ < period)
      next_hist_ = *next_hist_ + period;
    else
      next_hist_ = t + period;
    if (history_len_ > 0) {
      history_.push_back(latest_);
      while (history_.size() > history_len_) history_.pop_front();
    }
  }
}

void NavEngine::on_odom(double, const Pose& pose) {
  std::lock_guard<std::mutex> lk(in_mu_);
  if (odom_) travel_ += std::hypot(pose.x - odom_->x, pose.y - odom_->y);
  odom_ = pose;
}

void NavEngine::on_localization(double, const Pose& pose) {
  std::lock_guard<std::mutex> lk(in_mu_);
  loc_ = pose;
}

std::string NavEngine::start(std::shared_ptr<Topomap> topomap, const std::string& reason) {
  if (topomap) {
    map_ = std::move(topomap);
    node_emb_.clear();
  }
  double img_t;
  {
    std::lock_guard<std::mutex> lk(in_mu_);
    mode_ = resolve_mode(cfg_.tracker.reach_check, loc_.has_value(), odom_.has_value(), map_->index);
    odom_start_ = odom_;
    travel_ = 0.0;
    history_.clear();
    img_t = img_t_;
  }
  tracker_ = std::make_unique<SubgoalTracker>(map_->index, cfg_.tracker, mode_);
  set_state("running");
  if (stuck_) stuck_->reset();
  event(reason + ": " + std::to_string(map_->size()) + " subgoals, reach_check=" + mode_ +
            ", sample_rate=" + fmt_double(sample_rate_),
        img_t);
  return mode_;
}

void NavEngine::stop(const std::string& reason) {
  if (state() == "running") event("stop: " + reason, latest_stamp());
  set_state(reason != "reached" ? "stopped" : "reached");
}

std::optional<Pose> NavEngine::course_pose() const {
  std::optional<Pose> odom, loc;
  {
    std::lock_guard<std::mutex> lk(in_mu_);
    odom = odom_;
    loc = loc_;
  }
  return course_pose_of(odom, loc);
}

std::optional<Pose> NavEngine::course_pose_of(const std::optional<Pose>& odom, const std::optional<Pose>& loc) const {
  if (mode_ == "pose") return loc;
  if (odom && odom_start_ && map_->index.start) return align_to_start(*odom, *odom_start_, *map_->index.start);
  return std::nullopt;
}

std::vector<FramePtr> NavEngine::observation_window(const std::vector<FramePtr>& history,
                                                    const FramePtr& current) const {
  std::vector<FramePtr> hist(history);
  if (hist.empty() || hist.back() != current) hist.push_back(current);
  const int s = policy_->info().context_stride, n = policy_->info().context_size;
  std::vector<FramePtr> out;
  for (int k = n; k >= 0; --k) {
    const int i = static_cast<int>(hist.size()) - 1 - s * k;
    out.push_back(hist[std::max(0, i)]);
  }
  return out;
}

double NavEngine::similarity(int j, const FramePtr& cur) {
  if (node_emb_map_id_ != map_->id) {
    node_emb_.clear();
    node_emb_map_id_ = map_->id;
  }
  auto it = node_emb_.find(j);
  if (it == node_emb_.end()) it = node_emb_.emplace(j, policy_->embed(map_->images.at(j))).first;
  if (!emb_now_) emb_now_ = policy_->embed(cur);
  return cosine(*emb_now_, it->second);
}

void NavEngine::event(const std::string& text, double t) {
  if (logger) logger->event(t, text);
}

StepResult NavEngine::result(double v, double w, const std::optional<Waypoints>& wps, const std::string& state,
                             double latency, const std::string& reason, std::vector<std::string> events) {
  StepResult r;
  r.v = v;
  r.w = w;
  r.waypoints = wps;
  r.state = state;
  r.subgoal = tracker_ ? tracker_->index : 0;
  r.num_nodes = static_cast<int>(map_->size());
  r.latency = latency;
  if (tracker_) {
    r.similarity = tracker_->last_similarity;
    r.distance = tracker_->last_distance;
  }
  r.reason = reason;
  r.events = std::move(events);
  r.step = n_step_;
  last_ = r;
  return r;
}

StepResult NavEngine::step(double t, bool want_preview) {
  const int n = static_cast<int>(map_->size());
  std::vector<std::string> events;
  const std::string st = state();
  if (st != "running" || !tracker_) return result(0.0, 0.0, std::nullopt, st, 0.0, "", events);
  Inputs in;
  {
    std::lock_guard<std::mutex> lk(in_mu_);
    if (!latest_ || t - img_t_ > cfg_.max_image_age)
      return result(0.0, 0.0, std::nullopt, "waiting_image", 0.0, "no recent image", events);
    used_seq_ = img_seq_;
    in.frame = latest_;
    in.img_t = img_t_;
    in.odom = odom_;
    in.loc = loc_;
    in.travel = travel_;
    in.history.assign(history_.begin(), history_.end());
  }
  const FramePtr cur = in.frame;
  const auto pose = course_pose_of(in.odom, in.loc);
  // --- サブゴールの切り替え ---
  emb_now_.reset();
  const int before = tracker_->index;
  std::function<double(int)> sim;
  if (mode_ == "image" || mode_ == "image_odom") sim = [this, &cur](int j) { return similarity(j, cur); };
  if (tracker_->update(pose, sim, in.odom ? std::optional<double>(in.travel) : std::nullopt)) {
    if (tracker_->done)
      events.push_back("final goal reached [" + tracker_->last_reason + "]");
    else
      events.push_back("subgoal " + std::to_string(before) + " -> " + std::to_string(tracker_->index) + "/" +
                       std::to_string(n - 1) + " [" + tracker_->last_reason + "]");
  }
  for (const auto& e : events) event(e, t);
  if (tracker_->done && cfg_.stop_at_goal) {
    set_state("reached");
    auto r = result(0.0, 0.0, std::nullopt, "reached", 0.0, tracker_->last_reason, events);
    if (logger) logger->close("reached", true);
    return r;
  }
  // --- 推論 ---
  const int k = tracker_->index;
  const auto& node = map_->index.nodes.at(k);
  std::optional<Pose> goal_pose;
  if (cfg_.modality.find("pose") != std::string::npos && pose && node.pose) goal_pose = relative_pose(*pose, *node.pose);
  const auto obs = history_len_ > 0 ? observation_window(in.history, cur) : std::vector<FramePtr>{};
  PredictResult out = policy_->predict(cur, map_->images.at(k), obs, goal_pose, cfg_.modality, want_preview);
  auto [v, w] = compute_command(out.waypoints, cfg_.controller);
  // --- 動けない (押し付け) ---
  if (stuck_ && in.odom && stuck_->update(t, in.odom, v)) {
    stop("stuck");
    char buf[96];
    std::snprintf(buf, sizeof(buf), "stuck: no movement for %ss", fmt_double(cfg_.stuck_timeout).c_str());
    event(buf, t);
    if (logger) logger->close("stuck");
    auto r = result(0.0, 0.0, out.waypoints, "stopped", out.latency, "stuck", events);
    r.frame = cur;
    return r;
  }
  ++n_step_;
  auto r = result(v, w, out.waypoints, "running", out.latency, "", events);
  r.frame = cur;
  r.preview = out.preview;
  r.preview_width = out.preview_width;
  r.preview_height = out.preview_height;
  r.t = t;
  last_ = r;
  if (logger) {
    std::optional<std::pair<double, double>> gl;
    if (pose && node.pose) {
      const Pose rel = relative_pose(*pose, *node.pose);
      gl = std::make_pair(rel.x, rel.y);
    }
    logger->step(n_step_, t, pose, k, n, node.pose, gl, out.modality, cfg_.controller.mode, v, w, out.latency,
                 out.waypoints, tracker_->last_distance, tracker_->last_similarity, "running", cur);
  }
  return r;
}

nlohmann::json NavEngine::status() const {
  return {{"state", state()},
          {"subgoal", tracker_ ? nlohmann::json(tracker_->index) : nlohmann::json(nullptr)},
          {"num_nodes", map_->size()},
          {"reach_check", mode_},
          {"similarity", tracker_ ? opt(tracker_->last_similarity) : nlohmann::json(nullptr)},
          {"distance", tracker_ ? opt(tracker_->last_distance) : nlohmann::json(nullptr)},
          {"travel_m", travel()},
          {"v", last_.v},
          {"w", last_.w},
          {"latency", last_.state.empty() ? nlohmann::json(nullptr) : nlohmann::json(last_.latency)},
          {"time", wall_time()}};
}

}  // namespace omnivla
