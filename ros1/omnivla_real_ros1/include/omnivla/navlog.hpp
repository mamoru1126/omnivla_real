// 走行ログ (omnivla_real/navlog.py の NavRunLogger と同じ形式. tools/plot_nav_log.py で解析できる)
//   <log_dir>/<時刻>/meta.json, steps.csv, events.log, summary.json, goals/<k>.jpg, raw/<step>.jpg
#pragma once

#include <cstdio>
#include <mutex>
#include <optional>
#include <string>

#include <nlohmann/json.hpp>

#include "omnivla/types.hpp"

namespace omnivla {

class NavLogger {
 public:
  NavLogger(const std::string& root, nlohmann::json meta, const std::vector<FramePtr>& goals, bool save_raw);
  ~NavLogger();

  void event(std::optional<double> t, const std::string& text);
  void track_pose(double t, const std::optional<Pose>& pose);
  void step(int step, double t, const std::optional<Pose>& pose, int subgoal, int num_nodes,
            const std::optional<Pose>& subgoal_pose, const std::optional<std::pair<double, double>>& goal_local,
            int modality, const std::string& controller, double v, double w, double latency, const Waypoints& wps,
            std::optional<double> dist, std::optional<double> similarity, const std::string& state,
            const FramePtr& raw);
  nlohmann::json close(const std::string& reason, bool reached = false);
  const std::string& dir() const { return dir_; }
  bool closed() const { return closed_; }

 private:
  std::mutex mu_;
  std::string dir_;
  bool save_raw_;
  FILE* csv_ = nullptr;
  bool closed_ = false;
  int n_steps_ = 0;
  std::optional<Pose> first_pose_, last_pose_, final_goal_;
  std::optional<double> first_time_, last_time_;
  double path_len_ = 0.0;
  double min_final_dist_ = 1e18;
};

}  // namespace omnivla
