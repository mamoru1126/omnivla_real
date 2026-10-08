// ブラウザのデバッグ画面 (http://<ロボットの IP>:8080). ノードに内蔵した小さな HTTP サーバ.
//   GET  /                 画面 (web/index.html を埋め込み)
//   GET  /api/config       カメラの値・前処理・topomap など (画面で軌跡を画像に重ねるのに使う)
//   GET  /api/state        最新の状態 (JSON)
//   GET  /api/events       推論のたびに状態を送る (Server-Sent Events)
//   GET  /img/current.jpg  モデルに入れた現在画像 (前処理後. 推論サーバが返したもの)
//   GET  /img/goal/<k>.jpg サブゴール画像
//   POST /api/enable       {"enable": true|false} 開始 / 停止
// 画面を開いている人がいないときは何もしない (推論サーバにプレビュー画像も頼まない).
#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <nlohmann/json.hpp>

#include "omnivla/types.hpp"

namespace httplib {
class Server;
}

namespace omnivla {

class WebUi {
 public:
  using Log = std::function<void(const std::string&)>;
  WebUi(const std::string& host, int port, Log log);
  ~WebUi();

  // 画面の設定 (topomap が変わったら呼び直す)
  void set_config(const nlohmann::json& config, const std::vector<FramePtr>& goals);
  // 推論 1 回分 (state は JSON. preview はモデルに入れた画像の JPEG)
  void publish_step(nlohmann::json state, const Bytes& preview);
  // 指示値・状態の定期更新 (推論が止まっているときも画面を動かす)
  void publish_status(nlohmann::json status);
  bool has_viewers() const { return viewers_.load() > 0; }

  std::function<bool(bool)> on_enable;  // 開始 / 停止ボタン

 private:

  std::unique_ptr<httplib::Server> srv_;
  std::thread th_;
  Log log_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  uint64_t seq_ = 0;                                     // step と status で共通の通し番号
  nlohmann::json step_ = nlohmann::json::object();       // 最後の推論 (軌跡・画像など)
  nlohmann::json status_ = nlohmann::json::object();     // 最後の状態・指示値
  uint64_t step_seq_ = 0, status_seq_ = 0;
  nlohmann::json config_ = nlohmann::json::object();
  std::vector<FramePtr> goals_;
  Bytes preview_;
  std::atomic<int> viewers_{0};
  std::atomic<bool> stopping_{false};
};

extern const char* const kIndexHtml;  // web/index.html (CMake が埋め込む)

}  // namespace omnivla
