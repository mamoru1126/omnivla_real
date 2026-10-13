// サブゴールの切り替え (omnivla_real/topomap.py の SubgoalTracker と同じ判定. tests/test_cpp.py で突き合わせ)
#pragma once

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "omnivla/config.hpp"
#include "omnivla/types.hpp"

namespace omnivla {

// auto -> pose (自己位置あり) / image_odom (オドメトリあり) / image
std::string resolve_mode(const std::string& mode, bool has_localization, bool has_odom, const TopomapIndex& topomap);

// 各ノードのスタートからの道のり [m] (poses.yaml の s > ノード位置の間の距離の累積 > 番号 x spacing_m)
std::vector<double> course_distances(const TopomapIndex& topomap);

class SubgoalTracker {
 public:
  SubgoalTracker(const TopomapIndex& topomap, const TrackerConfig& cfg, const std::string& mode);

  // pose: topomap と同じ座標の今の位置 (pose / odom). sim(j): 今の画像とノード j の類似度.
  // travel: オドメトリ上の累積走行距離 (image_odom はこの道のりで候補を絞る). 進んだら true
  bool update(const std::optional<Pose>& pose, const std::function<double(int)>& sim,
              const std::optional<double>& travel);

  int index = 0;
  bool done = false;
  std::string last_reason;
  std::optional<double> last_similarity;
  std::optional<double> last_distance;
  std::map<int, double> similarities;
  std::vector<double> dist;          // 各ノードの道のり [m]
  std::optional<double> course_s;    // 今の道のりの推定 [m] (image_odom)
  const std::string& mode() const { return mode_; }
  int num_nodes() const { return static_cast<int>(map_.nodes.size()); }
  bool is_final() const { return index == num_nodes() - 1; }

 private:
  bool advance(int to, const std::string& reason, const std::optional<double>& travel);
  bool pose_check(const Pose& pose, const std::optional<double>& travel);
  bool image_check(const std::function<double(int)>& sim, const std::optional<double>& travel);
  double course_s_of(double travel);

  TopomapIndex map_;
  TrackerConfig cfg_;
  std::string mode_;
  std::optional<int> pending_;
  int pending_count_ = 0;
  double travel_at_switch_ = 0.0;
  double anchor_s_ = 0.0;                  // 道のりの推定 = anchor_s_ + (travel - anchor_travel_)
  std::optional<double> anchor_travel_;
};

}  // namespace omnivla
