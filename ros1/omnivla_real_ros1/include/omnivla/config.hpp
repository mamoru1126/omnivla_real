// 設定. 値は tools/nav_config_json.py が robot.yaml / navigator.yaml / topomap から作った JSON を読む
// (YAML の読み方と既定値は Python 側の 1 か所にまとめてある. ここの既定値は JSON に無い場合の保険)
#pragma once

#include <array>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "omnivla/types.hpp"

namespace omnivla {

using json = nlohmann::json;

struct TopicConfig {
  std::string image = "/camera/image_raw/compressed";
  std::string odom = "/odom";
  std::string cmd = "/cmd_vel";
  std::string localization;
  std::string localization_frame = "map";
  std::string localization_child_frame = "base_link";
};

struct ImageConfig {
  std::array<double, 4> crop{0.0, 0.0, 0.0, 0.0};  // 上, 下, 左, 右 (割合)
  int width = 320;
  bool rotate180 = false;
  int jpeg_quality = 95;
};

struct CameraConfig {
  double hfov_deg = 90.0;
  double height = 0.5;
  double x_offset = 0.2;
  double pitch_deg = 0.0;
};

struct RobotConfig {
  std::string name = "my_robot";
  TopicConfig topics;
  ImageConfig image;
  CameraConfig camera;
  std::string time_source = "auto";
};

struct ControllerConfig {
  std::string mode = "trajectory";  // trajectory | upstream | pure_pursuit
  int track_horizon = 4;
  double track_max_v = 0.4;
  double track_max_w = 1.0;
  double track_phi_scale = 0.15;
  int waypoint_index = 4;
  double dt = 1.0 / 3.0;
  double max_linear_raw = 0.5;
  double max_angular_raw = 1.0;
  double max_v = 0.3;
  double max_w = 0.3;
  bool respect_predicted_speed = true;
  double lookahead = 0.5;
  double pp_speed = 0.3;
  double pp_max_w = 0.8;
  double rotate_in_place_angle = 1.2;
};

struct TrackerConfig {
  std::string reach_check = "auto";
  double subgoal_radius = 0.5;
  double goal_radius = 0.5;
  double reach_angle_deg = 30.0;
  double pass_radius = 1.5;
  double pass_angle_deg = 90.0;
  double image_threshold = 0.80;
  double goal_image_threshold = 0.85;
  int search_window = 2;
  int confirm = 2;
  double odom_gate_m = 2.5;
  double odom_pass_m = 1.0;
  double goal_odom_gate_m = 1.5;
  double min_travel_m = 0.2;
};

struct EngineConfig {
  std::string modality = "image";
  double sample_rate = 0.0;
  ControllerConfig controller;
  TrackerConfig tracker;
  double max_image_age = 0.5;
  double stuck_timeout = 0.0;
  bool stop_at_goal = true;
  bool debug_image = true;
};

struct IOConfig {
  std::string cmd_vel = "/cmd_vel";
  bool cmd_stamped = false;
  std::string base_frame = "base_link";
  std::string path = "/omnivla/path";
  std::string debug_image = "/omnivla/debug_image";
  std::string status = "/omnivla/status";
  std::string enable = "/omnivla/enable";
  std::string topomap = "/omnivla/topomap";
  std::string localization_type = "PoseWithCovarianceStamped";
  double control_rate = 10.0;
  double cmd_timeout = 1.0;
  double max_inference_rate = 0.0;
  std::string log_dir = "/workspace/log/nav";
  bool log_images = true;
  int web_port = 8080;
  std::string web_host = "0.0.0.0";
};

struct ModelConfig {
  std::string model = "remote";
  std::string url;
  double timeout = 10.0;
};

struct TopomapNode {
  std::string path;
  std::optional<Pose> pose;
  std::optional<double> s;
};

struct TopomapIndex {
  std::string directory;
  std::string frame = "odom";
  std::optional<Pose> start;
  json meta = json::object();
  std::vector<TopomapNode> nodes;
};

// サブゴール画像列 (画像ファイルの中身も読み込んだもの. 画像は前処理済みの JPEG)
struct Topomap {
  TopomapIndex index;
  std::vector<FramePtr> images;
  int id = 0;  // 読み込むたびに変わる番号 (推論サーバに送る画像の key "g<id>_<k>" に使う)

  size_t size() const { return index.nodes.size(); }
  bool has_poses() const {
    if (index.nodes.empty()) return false;
    for (const auto& n : index.nodes)
      if (!n.pose) return false;
    return true;
  }
};

struct NavConfig {
  RobotConfig robot;
  ModelConfig model;
  EngineConfig engine;
  IOConfig io;
  std::optional<TopomapIndex> topomap;
  json raw = json::object();  // 元の JSON (ログの meta 用)
};

NavConfig nav_config_from_json(const json& j);
TopomapIndex topomap_index_from_json(const json& j);
json to_json(const ControllerConfig& c);
json to_json(const TrackerConfig& c);

// tools/nav_config_json.py を実行して JSON を得る (python は OMNIVLA_PYTHON か python3)
json run_config_tool(const std::string& repo_root, const std::vector<std::string>& args);

// 画像ファイルも読み込む
std::shared_ptr<Topomap> load_topomap(const TopomapIndex& index);

// ファイルの中身
Bytes read_file(const std::string& path);

}  // namespace omnivla
