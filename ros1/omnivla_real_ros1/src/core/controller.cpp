#include "omnivla/controller.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "omnivla/geometry.hpp"

namespace omnivla {

namespace {

double sign(double x) { return x > 0 ? 1.0 : (x < 0 ? -1.0 : 0.0); }

// visualnav-transformer の clip_angle
double clip_angle(double theta) {
  theta = std::fmod(theta, 2.0 * kPi);
  if (theta < 0.0) theta += 2.0 * kPi;
  if (theta > kPi) theta -= 2.0 * kPi;
  return theta;
}

// 公式 run_omnivla.py の速度制限 (曲率半径を保つ)
std::pair<double, double> limit_velocity(double v, double w, double maxv, double maxw) {
  if (std::abs(v) <= maxv) {
    if (std::abs(w) <= maxw) return {v, w};
    const double rd = v / w;
    return {maxw * sign(v) * std::abs(rd), maxw * sign(w)};
  }
  if (std::abs(w) <= 0.001) return {maxv * sign(v), 0.0};
  const double rd = v / w;
  if (std::abs(rd) >= maxv / maxw) return {maxv * sign(v), maxv * sign(w) / std::abs(rd)};
  return {maxw * sign(v) * std::abs(rd), maxw * sign(w)};
}

// numpy.unwrap と同じ
std::vector<double> unwrap(const std::vector<double>& p) {
  std::vector<double> out(p);
  double correction = 0.0;
  for (size_t i = 1; i < p.size(); ++i) {
    const double dd = p[i] - p[i - 1];
    double ddmod = std::fmod(dd + kPi, 2.0 * kPi);
    if (ddmod < 0) ddmod += 2.0 * kPi;
    ddmod -= kPi;
    if (ddmod == -kPi && dd > 0) ddmod = kPi;
    double ph = ddmod - dd;
    if (std::abs(dd) < kPi) ph = 0.0;
    correction += ph;
    out[i] = p[i] + correction;
  }
  return out;
}

}  // namespace

std::pair<double, double> upstream_command(const Waypoint& wp, const ControllerConfig& cfg) {
  const double dx = wp[0], dy = wp[1], hx = wp[2], hy = wp[3];
  const double eps = 1e-8, dt = cfg.dt;
  double v, w;
  if (std::abs(dx) < eps && std::abs(dy) < eps) {
    v = 0.0;
    w = clip_angle(std::atan2(hy, hx)) / dt;
  } else if (std::abs(dx) < eps) {
    v = 0.0;
    w = sign(dy) * kPi / (2.0 * dt);
  } else {
    v = dx / dt;
    w = std::atan2(dy, dx) / dt;
  }
  v = std::clamp(v, 0.0, cfg.max_linear_raw);
  w = std::clamp(w, -cfg.max_angular_raw, cfg.max_angular_raw);
  return limit_velocity(v, w, cfg.max_v, cfg.max_w);
}

std::pair<double, double> pure_pursuit_command(const Waypoints& wps, const ControllerConfig& cfg) {
  double dmax = 0.0;
  for (const auto& p : wps) dmax = std::max(dmax, std::hypot(p[0], p[1]));
  if (dmax < 0.05) return {0.0, 0.0};
  size_t idx = wps.size() - 1;
  for (size_t i = 0; i < wps.size(); ++i) {
    if (std::hypot(wps[i][0], wps[i][1]) >= cfg.lookahead) {
      idx = i;
      break;
    }
  }
  const double tx = wps[idx][0], ty = wps[idx][1];
  const double ld = std::max(std::hypot(tx, ty), 1e-3);
  const double alpha = std::atan2(ty, tx);
  if (std::abs(alpha) > cfg.rotate_in_place_angle) return {0.0, sign(alpha) * cfg.pp_max_w};
  double v = cfg.pp_speed * std::min(1.0, ld / std::max(cfg.lookahead, 1e-3));
  const double curvature = 2.0 * std::sin(alpha) / ld;
  double w = curvature * v;
  if (std::abs(w) > cfg.pp_max_w) {
    v = v * cfg.pp_max_w / std::abs(w);
    w = sign(w) * cfg.pp_max_w;
  }
  return {v, w};
}

std::pair<double, double> trajectory_command(const Waypoints& wps, const ControllerConfig& cfg) {
  if (wps.empty()) return {0.0, 0.0};
  const int K = std::clamp(cfg.track_horizon, 0, static_cast<int>(wps.size()) - 1);
  std::vector<double> yaw_raw(K + 1);
  for (int k = 0; k <= K; ++k) yaw_raw[k] = std::atan2(wps[k][3], wps[k][2]);
  const auto yaw = unwrap(yaw_raw);
  double tt = 0.0, ta = 0.0, tw = 0.0;
  for (int k = 0; k <= K; ++k) {
    const double t = (k + 1.0) * cfg.dt;
    const double x = wps[k][0], y = wps[k][1];
    const double chord = std::hypot(x, y);
    const double phi = 2.0 * std::atan2(y, x);
    const double half = std::abs(phi) / 2.0;
    const double arc = half > 1e-6 ? chord * half / std::max(std::sin(half), 1e-6) : chord;
    const double a = 0.5 * chord * chord / (chord * chord + cfg.track_phi_scale * cfg.track_phi_scale);
    const double turn = a * phi + (1.0 - a) * yaw[k];
    tt += t * t;
    ta += t * arc;
    tw += t * turn;
  }
  double v = std::max(ta / tt, 0.0);
  double w = tw / tt;
  double s = 1.0;
  if (v > cfg.track_max_v) s = std::min(s, cfg.track_max_v / v);
  if (std::abs(w) > cfg.track_max_w) s = std::min(s, cfg.track_max_w / std::abs(w));
  return {v * s, w * s};
}

std::pair<double, double> compute_command(const Waypoints& wps, const ControllerConfig& cfg) {
  if (cfg.mode == "upstream") {
    const int idx = std::clamp(cfg.waypoint_index, 0, static_cast<int>(wps.size()) - 1);
    auto [v, w] = upstream_command(wps[idx], cfg);
    if (cfg.respect_predicted_speed) {
      const double v_pred = std::hypot(wps[idx][0], wps[idx][1]) / ((idx + 1) * cfg.dt);
      if (!(v <= 1e-6 || v <= v_pred)) {
        const double s = std::max(v_pred, 0.0) / v;
        v *= s;
        w *= s;
      }
    }
    return {v, w};
  }
  if (cfg.mode == "trajectory") return trajectory_command(wps, cfg);
  if (cfg.mode == "pure_pursuit") return pure_pursuit_command(wps, cfg);
  throw std::runtime_error("unknown controller mode: " + cfg.mode);
}

bool StuckDetector::update(double t, const std::optional<Pose>& pose, double cmd_v) {
  if (!pose || timeout_ <= 0) return false;
  if (cmd_v < min_cmd_v_) {
    ref_.reset();
    return false;
  }
  if (!ref_) {
    ref_ = Ref{t, *pose};
    return false;
  }
  const double moved = std::hypot(pose->x - ref_->pose.x, pose->y - ref_->pose.y);
  const double turned = std::abs(wrap_angle(pose->yaw - ref_->pose.yaw));
  if (moved > min_move_ || turned > min_turn_) {
    ref_ = Ref{t, *pose};
    return false;
  }
  return t - ref_->t > timeout_;
}

}  // namespace omnivla
