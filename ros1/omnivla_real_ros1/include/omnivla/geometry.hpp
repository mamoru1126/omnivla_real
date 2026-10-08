// 2D 幾何 (omnivla_real/geometry.py, topomap.py の compose / align_to_start と同じ計算)
#pragma once

#include <cmath>

#include "omnivla/types.hpp"

namespace omnivla {

constexpr double kPi = 3.14159265358979323846;

// [-pi, pi) に正規化 (Python の (a + pi) % (2 pi) - pi と同じ)
inline double wrap_angle(double a) {
  double m = std::fmod(a + kPi, 2.0 * kPi);
  if (m < 0.0) m += 2.0 * kPi;
  return m - kPi;
}

// current から見た target (x 前, y 左, dyaw)
inline Pose relative_pose(const Pose& current, const Pose& target) {
  const double c = std::cos(current.yaw), s = std::sin(current.yaw);
  const double dx = target.x - current.x, dy = target.y - current.y;
  return {dx * c + dy * s, dx * -s + dy * c, wrap_angle(target.yaw - current.yaw)};
}

// a の座標系で表した b をワールドへ (a ∘ b)
inline Pose compose(const Pose& a, const Pose& b) {
  const double c = std::cos(a.yaw), s = std::sin(a.yaw);
  return {a.x + c * b.x - s * b.y, a.y + s * b.x + c * b.y, wrap_angle(a.yaw + b.yaw)};
}

// 走行開始時のロボット位置 = topomap の start とみなして、今のオドメトリ位置を topomap の座標へ
inline Pose align_to_start(const Pose& odom, const Pose& odom_at_start, const Pose& topomap_start) {
  return compose(topomap_start, relative_pose(odom_at_start, odom));
}

inline double yaw_from_quaternion(double x, double y, double z, double w) {
  return std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
}

}  // namespace omnivla
