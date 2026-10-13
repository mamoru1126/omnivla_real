#include "omnivla/tracker.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "omnivla/geometry.hpp"

namespace omnivla {

namespace {

bool has_poses(const TopomapIndex& t) {
  if (t.nodes.empty()) return false;
  for (const auto& n : t.nodes)
    if (!n.pose) return false;
  return true;
}

std::string fmt(const char* f, double a, double b = 0.0) {
  char buf[128];
  std::snprintf(buf, sizeof(buf), f, a, b);
  return buf;
}

double rad(double deg) { return deg * kPi / 180.0; }

}  // namespace

std::string resolve_mode(const std::string& mode, bool has_localization, bool has_odom, const TopomapIndex& t) {
  static const char* kModes[] = {"auto", "pose", "odom", "image", "image_odom", "none"};
  if (std::find_if(std::begin(kModes), std::end(kModes), [&](const char* m) { return mode == m; }) == std::end(kModes))
    throw std::runtime_error("reach_check must be one of auto, pose, odom, image, image_odom, none");
  if (mode != "auto") return mode;
  if (has_localization && t.frame == "map" && has_poses(t)) return "pose";
  if (has_odom && has_poses(t) && t.start) return "image_odom";
  return "image";
}

std::vector<double> course_distances(const TopomapIndex& t) {
  const auto& nodes = t.nodes;
  std::vector<double> out;
  if (std::all_of(nodes.begin(), nodes.end(), [](const TopomapNode& n) { return n.s.has_value(); })) {
    for (const auto& n : nodes) out.push_back(*n.s);
    return out;
  }
  if (has_poses(t)) {
    out.push_back(0.0);
    for (size_t k = 1; k < nodes.size(); ++k)
      out.push_back(out.back() + std::hypot(nodes[k].pose->x - nodes[k - 1].pose->x,
                                            nodes[k].pose->y - nodes[k - 1].pose->y));
    return out;
  }
  double sp = 1.0;
  if (t.meta.is_object() && t.meta.contains("spacing_m") && t.meta["spacing_m"].is_number() &&
      t.meta["spacing_m"].get<double>() != 0.0)
    sp = t.meta["spacing_m"].get<double>();
  for (size_t k = 0; k < nodes.size(); ++k) out.push_back(static_cast<double>(k) * sp);
  return out;
}

SubgoalTracker::SubgoalTracker(const TopomapIndex& topomap, const TrackerConfig& cfg, const std::string& mode)
    : map_(topomap), cfg_(cfg), mode_(mode) {
  if (map_.nodes.empty()) throw std::runtime_error("empty topomap");
  if ((mode == "pose" || mode == "odom" || mode == "image_odom") && !has_poses(map_))
    throw std::runtime_error("reach_check=" + mode + " needs node poses in poses.yaml");
  dist = course_distances(map_);
  anchor_s_ = dist[0];
}

double SubgoalTracker::course_s_of(double travel) {
  if (!anchor_travel_) anchor_travel_ = travel;
  return anchor_s_ + (travel - *anchor_travel_);
}

bool SubgoalTracker::advance(int to, const std::string& reason, const std::optional<double>& travel) {
  last_reason = reason;
  pending_.reset();
  pending_count_ = 0;
  if (travel) travel_at_switch_ = *travel;
  if (to >= num_nodes()) {
    index = num_nodes() - 1;
    done = true;
  } else {
    index = to;
  }
  return true;
}

bool SubgoalTracker::pose_check(const Pose& pose, const std::optional<double>& travel) {
  const int last = num_nodes() - 1;
  for (int j = std::min(last, index + 1); j > index - 1; --j) {  // 1 つ先まで (飛ばし対応)
    const Pose node = *map_.nodes[j].pose;
    const Pose rel = relative_pose(pose, node);
    const double d = std::hypot(rel.x, rel.y);
    if (j == index) last_distance = d;
    const bool final = j == last;
    if (d < (final ? cfg_.goal_radius : cfg_.subgoal_radius)) {
      const double dyaw = std::abs(wrap_angle(pose.yaw - node.yaw));
      if (final || cfg_.reach_angle_deg <= 0 || dyaw <= rad(cfg_.reach_angle_deg)) {
        return advance(j + 1, fmt("within radius (d=%.2fm, dyaw=%.0fdeg)", d, dyaw * 180.0 / kPi), travel);
      }
    }
    if (!final && d < cfg_.pass_radius && std::abs(std::atan2(rel.y, rel.x)) > rad(cfg_.pass_angle_deg)) {
      return advance(j + 1, fmt("passed (d=%.2fm)", d), travel);
    }
  }
  return false;
}

bool SubgoalTracker::image_check(const std::function<double(int)>& sim, const std::optional<double>& travel) {
  const int last = num_nodes() - 1;
  const int w = std::max(0, cfg_.search_window);
  const int hi = std::min(last, index + w);
  const bool use_odom = mode_ == "image_odom" && travel.has_value();
  // image_odom は今のノードより手前も比べる (道のりの推定を後ろへ直すため. 戻りはしない)
  const int lo = use_odom ? std::max(0, index - w) : index;
  similarities.clear();
  for (int j = lo; j <= hi; ++j) similarities[j] = sim(j);
  last_similarity = similarities[index];
  std::optional<double> est;
  if (use_odom) est = course_s_of(*travel);
  course_s = est;
  if (est) last_distance = dist[index] - *est;  // 道のりで今のサブゴールまで (負なら通り過ぎ)
  // 道のりで明らかに先のノードは候補から外す
  std::map<int, double> cand = similarities;
  if (est) {
    for (auto it = cand.begin(); it != cand.end();) {
      const double gate = it->first == last ? std::min(cfg_.odom_gate_m, cfg_.goal_odom_gate_m) : cfg_.odom_gate_m;
      if (dist[it->first] - *est > gate)
        it = cand.erase(it);
      else
        ++it;
    }
  }
  // 最大 (同点なら番号が小さい方: Python の max(dict, key=...) と同じ)
  std::optional<int> best;
  for (const auto& [j, s] : cand)
    if (!best || s > cand[*best]) best = j;
  auto thr = [&](int j) { return j == last ? cfg_.goal_image_threshold : cfg_.image_threshold; };
  if (best && cand[*best] >= thr(*best)) {
    const bool moved = !travel || *travel - travel_at_switch_ >= cfg_.min_travel_m || *best > index;
    if (moved) {
      pending_count_ = (pending_ && *pending_ == *best) ? pending_count_ + 1 : 1;
      pending_ = *best;
      if (pending_count_ >= std::max(1, cfg_.confirm)) {
        if (est) {  // 画像で見えたノードに道のりを寄せる (前回の補正から走った距離の forward_correction 倍まで)
          const double lim = std::max(0.0, cfg_.forward_correction) * (*travel - *anchor_travel_);
          anchor_s_ = std::min(std::max(dist[*best], *est - lim), *est + lim);
          anchor_travel_ = *travel;
        }
        if (*best < index) {  // 手前のノードが一番似ている: 道のりを直すだけ (サブゴールは戻さない)
          pending_.reset();
          pending_count_ = 0;
          travel_at_switch_ = *travel;
          return false;
        }
        return advance(*best + 1, fmt("image similarity %.3f (node %.0f)", cand[*best], *best), travel);
      }
      return false;
    }
  } else {
    pending_.reset();
    pending_count_ = 0;
  }
  // 画像で判定できないまま、道のり上で今のノードを通り過ぎた (最終ゴールなら止まる)
  if (est && *est - dist[index] > cfg_.odom_pass_m)
    return advance(index + 1, fmt("passed by odometry (%.2fm behind)", *est - dist[index]), travel);
  return false;
}

bool SubgoalTracker::update(const std::optional<Pose>& pose, const std::function<double(int)>& sim,
                            const std::optional<double>& travel) {
  if (done || mode_ == "none") return false;
  if (mode_ == "pose" || mode_ == "odom") {
    if (!pose) return false;
    return pose_check(*pose, travel);
  }
  if (!sim) return false;
  return image_check(sim, travel);
}

}  // namespace omnivla
