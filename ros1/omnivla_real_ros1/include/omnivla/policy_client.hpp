// 推論サーバ (tools/policy_server.py) のクライアント. 通信の形式は omnivla_real/remote.py を参照.
// カメラ画像は JPEG / raw のまま送り、前処理 (切り抜き・縮小) はサーバ側 (学習時と同じ Python の関数) で行う.
// 送った画像はサーバが key で覚えているので、観測履歴・サブゴール画像は 2 回目から key だけを送る.
#pragma once

#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#include "omnivla/types.hpp"

namespace httplib {
class Client;
}

namespace omnivla {

using json = nlohmann::json;

struct PolicyInfo {
  std::string model;
  bool history = false;      // 観測履歴を使う (edge)
  int context_size = 0;
  int context_stride = 1;
  double metric_spacing = 0.0;
  json meta = json::object();
};

struct PredictResult {
  Waypoints waypoints;       // (8, 4) [x, y, cos, sin] ロボット座標 [m]
  Waypoints normalized;
  int modality = 0;
  double latency = 0.0;      // 往復 [s]
  double server_latency = 0.0;
  std::optional<double> distance;
  Bytes preview;             // want_preview: モデルに入れた現在画像 (前処理後) の JPEG
  int preview_width = 0, preview_height = 0;
};

// NavEngine から見たポリシー (テストでは偽物に差し替える)
class Policy {
 public:
  virtual ~Policy() = default;
  virtual const PolicyInfo& info() const = 0;
  virtual PredictResult predict(const FramePtr& current, const FramePtr& goal, const std::vector<FramePtr>& observations,
                                const std::optional<Pose>& goal_pose, const std::string& modality,
                                bool want_preview) = 0;
  virtual std::vector<float> embed(const FramePtr& image) = 0;
};

class PolicyClient : public Policy {
 public:
  // url 例 http://127.0.0.1:8765. wait 秒まで推論サーバの起動を待つ
  PolicyClient(const std::string& url, double timeout, double wait, const std::function<void(const std::string&)>& log);
  ~PolicyClient() override;

  const PolicyInfo& info() const override { return info_; }
  PredictResult predict(const FramePtr& current, const FramePtr& goal, const std::vector<FramePtr>& observations,
                        const std::optional<Pose>& goal_pose, const std::string& modality, bool want_preview) override;
  std::vector<float> embed(const FramePtr& image) override;
  size_t bytes_sent() const { return bytes_sent_; }

 private:
  struct Response {
    json header;
    std::vector<std::pair<json, std::string>> arrays;  // (shape, バイト列)
  };
  Response call(const std::string& path, json header, const std::vector<FramePtr>& images);
  Response post(const std::string& path, const std::string& body, int* status);

  std::unique_ptr<httplib::Client> cli_;
  std::mutex mu_;
  PolicyInfo info_;
  std::deque<std::string> known_order_;
  std::unordered_set<std::string> known_;
  size_t bytes_sent_ = 0;
};

}  // namespace omnivla
