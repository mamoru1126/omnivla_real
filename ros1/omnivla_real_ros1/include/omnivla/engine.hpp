// 走行の中身 (omnivla_real/engine.py の NavEngine と同じ処理). ROS 非依存.
//   1. サブゴールに着いたか判定して切り替える (SubgoalTracker)
//   2. 推論サーバに「今の画像 + 観測履歴 + サブゴール画像」を送り、8 点の軌跡を受け取る
//   3. 軌跡を再現する (v, w) を計算する
#pragma once

#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "omnivla/config.hpp"
#include "omnivla/controller.hpp"
#include "omnivla/policy_client.hpp"
#include "omnivla/tracker.hpp"

namespace omnivla {

class NavLogger;

struct StepResult {
  double v = 0.0, w = 0.0;
  std::optional<Waypoints> waypoints;
  std::string state;             // running | reached | idle | waiting_image | stopped
  int subgoal = 0;
  int num_nodes = 0;
  double latency = 0.0;
  std::optional<double> similarity;
  std::optional<double> distance;
  std::string reason;
  std::vector<std::string> events;
  FramePtr frame;                // 推論に使った画像
  Bytes preview;                 // want_preview: モデルに入れた画像 (前処理後, JPEG)
  int preview_width = 0, preview_height = 0;
  double t = 0.0;
  int step = 0;
};

class NavEngine {
 public:
  NavEngine(Policy* policy, std::shared_ptr<Topomap> topomap, const EngineConfig& cfg);

  void on_image(double t, FramePtr frame);
  void on_odom(double t, const Pose& pose);
  void on_localization(double t, const Pose& pose);
  // 走行開始 (topomap を渡すと差し替え). 戻り値はサブゴールの判定方法
  std::string start(std::shared_ptr<Topomap> topomap = nullptr, const std::string& reason = "start");
  void set_topomap(std::shared_ptr<Topomap> topomap) {
    map_ = std::move(topomap);
    node_emb_.clear();
  }
  void stop(const std::string& reason = "stopped");
  StepResult step(double t, bool want_preview = false);
  bool has_new_image() const {
    std::lock_guard<std::mutex> lk(in_mu_);
    return latest_ && img_seq_ != used_seq_;
  }
  std::optional<Pose> course_pose() const;
  nlohmann::json status() const;

  std::string state() const {
    std::lock_guard<std::mutex> lk(in_mu_);
    return state_;
  }
  const std::string& mode() const { return mode_; }
  double sample_rate() const { return sample_rate_; }
  std::shared_ptr<Topomap> topomap() const { return map_; }
  std::optional<Pose> odom() const {
    std::lock_guard<std::mutex> lk(in_mu_);
    return odom_;
  }
  bool has_localization() const {
    std::lock_guard<std::mutex> lk(in_mu_);
    return loc_.has_value();
  }
  FramePtr latest() const {
    std::lock_guard<std::mutex> lk(in_mu_);
    return latest_;
  }
  const EngineConfig& config() const { return cfg_; }
  double travel() const {
    std::lock_guard<std::mutex> lk(in_mu_);
    return travel_;
  }
  double latest_stamp() const {
    std::lock_guard<std::mutex> lk(in_mu_);
    return img_t_;
  }
  NavLogger* logger = nullptr;

 private:
  struct Inputs {
    FramePtr frame;
    double img_t = 0.0;
    std::optional<Pose> odom, loc;
    double travel = 0.0;
    std::vector<FramePtr> history;
  };
  void set_state(const std::string& s) {
    std::lock_guard<std::mutex> lk(in_mu_);
    state_ = s;
  }
  std::optional<Pose> course_pose_of(const std::optional<Pose>& odom, const std::optional<Pose>& loc) const;
  double similarity(int j, const FramePtr& cur);
  StepResult result(double v, double w, const std::optional<Waypoints>& wps, const std::string& state, double latency,
                    const std::string& reason, std::vector<std::string> events);
  void event(const std::string& text, double t);
  std::vector<FramePtr> observation_window(const std::vector<FramePtr>& history, const FramePtr& current) const;

  Policy* policy_;
  std::shared_ptr<Topomap> map_;
  EngineConfig cfg_;
  double sample_rate_ = 3.0;
  std::string state_ = "idle";
  std::string mode_ = "image";
  std::unique_ptr<SubgoalTracker> tracker_;
  std::unique_ptr<StuckDetector> stuck_;
  // 入力 (ROS のコールバックから書かれる. in_mu_ で守る)
  mutable std::mutex in_mu_;
  FramePtr latest_;
  double img_t_ = 0.0;
  uint64_t img_seq_ = 0, used_seq_ = 0;
  std::optional<double> next_hist_;
  std::deque<FramePtr> history_;
  size_t history_len_ = 0;
  std::optional<Pose> odom_, odom_start_, loc_;
  double travel_ = 0.0;
  int n_step_ = 0;
  std::map<int, std::vector<float>> node_emb_;
  int node_emb_map_id_ = -1;
  std::optional<std::vector<float>> emb_now_;
  StepResult last_;
};

}  // namespace omnivla
