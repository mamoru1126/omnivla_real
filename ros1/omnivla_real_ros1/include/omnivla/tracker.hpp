// サブゴールの切り替え (omnivla_real/topomap.py の SubgoalTracker と同じ判定. tests/test_cpp.py で突き合わせ)
#pragma once

#include <functional>
#include <map>
#include <optional>
#include <string>

#include "omnivla/config.hpp"
#include "omnivla/types.hpp"

namespace omnivla {

// auto -> pose (自己位置あり) / image_odom (オドメトリあり) / image
std::string resolve_mode(const std::string& mode, bool has_localization, bool has_odom, const TopomapIndex& topomap);

class SubgoalTracker {
 public:
  SubgoalTracker(const TopomapIndex& topomap, const TrackerConfig& cfg, const std::string& mode);

  // pose: topomap と同じ座標の今の位置 (pose / odom / image_odom). sim(j): 今の画像とノード j の類似度.
  // travel: オドメトリ上の累積走行距離. 進んだら true
  bool update(const std::optional<Pose>& pose, const std::function<double(int)>& sim,
              const std::optional<double>& travel);

  int index = 0;
  bool done = false;
  std::string last_reason;
  std::optional<double> last_similarity;
  std::optional<double> last_distance;
  std::map<int, double> similarities;
  const std::string& mode() const { return mode_; }
  int num_nodes() const { return static_cast<int>(map_.nodes.size()); }
  bool is_final() const { return index == num_nodes() - 1; }

 private:
  bool advance(int to, const std::string& reason, const std::optional<double>& travel);
  bool pose_check(const Pose& pose, const std::optional<double>& travel);
  bool image_check(const std::function<double(int)>& sim, const std::optional<Pose>& course_pose,
                   const std::optional<double>& travel);

  TopomapIndex map_;
  TrackerConfig cfg_;
  std::string mode_;
  std::optional<int> pending_;
  int pending_count_ = 0;
  double travel_at_switch_ = 0.0;
};

}  // namespace omnivla
