// 共通の型 (ROS 非依存)
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace omnivla {

// 2D 姿勢. ワールド/odom 座標は ROS と同じ (yaw は x 軸から反時計回り), ロボット座標は x 前, y 左
struct Pose {
  double x = 0.0, y = 0.0, yaw = 0.0;
};

using Bytes = std::shared_ptr<const std::vector<uint8_t>>;

// カメラ画像 1 枚. デコードはしない (推論サーバが前処理する. ブラウザには JPEG のまま渡す)
struct Frame {
  std::string key;          // 推論サーバでのキャッシュの名前 ("f123")
  double stamp = 0.0;       // [s]
  std::string kind;         // "encoded" (JPEG/PNG: CompressedImage) | "raw" (sensor_msgs/Image)
  std::string format;       // encoded: "jpeg" | "png"
  std::string encoding;     // raw: rgb8, bgr8, mono8, ...
  int width = 0, height = 0, step = 0;
  bool is_bigendian = false;
  bool preprocess = true;   // 推論サーバで robot.yaml の前処理をかけるか (topomap の画像は前処理済みなので false)
  Bytes data;

  bool is_jpeg() const { return kind == "encoded" && (format.empty() || format.find("jpeg") != std::string::npos ||
                                                        format.find("jpg") != std::string::npos); }
};
using FramePtr = std::shared_ptr<const Frame>;

// 予測軌跡の 1 点: [x[m], y[m], cos(yaw), sin(yaw)] (ロボット座標). k 点目は (k+1)/sample_rate 秒後
using Waypoint = std::array<double, 4>;
using Waypoints = std::vector<Waypoint>;

}  // namespace omnivla
