// 予測軌跡 -> (v, w) (omnivla_real/controller.py と同じ計算. tests/test_cpp.py で突き合わせ)
#pragma once

#include <optional>
#include <utility>

#include "omnivla/config.hpp"
#include "omnivla/types.hpp"

namespace omnivla {

// (v [m/s], w [rad/s])
std::pair<double, double> compute_command(const Waypoints& wps, const ControllerConfig& cfg);
std::pair<double, double> trajectory_command(const Waypoints& wps, const ControllerConfig& cfg);
std::pair<double, double> upstream_command(const Waypoint& wp, const ControllerConfig& cfg);
std::pair<double, double> pure_pursuit_command(const Waypoints& wps, const ControllerConfig& cfg);

// 前進指令を出しているのに timeout 秒ほとんど動かなければ true (壁に押し付けている)
class StuckDetector {
 public:
  explicit StuckDetector(double timeout = 4.0, double min_move = 0.05, double min_turn = 0.1, double min_cmd_v = 0.05)
      : timeout_(timeout), min_move_(min_move), min_turn_(min_turn), min_cmd_v_(min_cmd_v) {}
  void reset() { ref_.reset(); }
  bool update(double t, const std::optional<Pose>& pose, double cmd_v);

 private:
  struct Ref {
    double t;
    Pose pose;
  };
  double timeout_, min_move_, min_turn_, min_cmd_v_;
  std::optional<Ref> ref_;
};

}  // namespace omnivla
