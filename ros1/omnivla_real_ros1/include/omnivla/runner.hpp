// ノードの中身 (omnivla_real/ros_common.py の NavRunner と同じ): エンジン + 推論スレッド + 指示値の保持.
// 推論は新しいカメラ画像が来たときだけ行い、指示値は control_rate で最新の結果を出し続ける (古くなったら 0).
#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>

#include <nlohmann/json.hpp>

#include "omnivla/config.hpp"
#include "omnivla/engine.hpp"
#include "omnivla/navlog.hpp"
#include "omnivla/policy_client.hpp"

namespace omnivla {

class NavRunner {
 public:
  using Clock = std::function<double()>;
  using Log = std::function<void(const std::string&)>;
  using TopomapLoader = std::function<std::shared_ptr<Topomap>(const std::string&)>;

  // threaded=false: 推論スレッドを作らない (試験用. step_once() を自分で呼ぶ)
  NavRunner(const NavConfig& cfg, std::unique_ptr<Policy> policy, std::shared_ptr<Topomap> topomap,
            const std::string& topomap_path, Clock clock, Log log, TopomapLoader loader = nullptr,
            bool threaded = true);
  ~NavRunner();

  bool start(const std::string& topomap_path = "");
  void stop(const std::string& reason = "stopped by user");
  void shutdown();
  // 今出すべき指示値. nullopt なら何も出さない (止めてから 1 秒以上経った時)
  std::optional<std::pair<double, double>> command();
  nlohmann::json status_json();
  // 推論 1 回 (新しい画像が無ければ nullopt)
  std::optional<StepResult> step_once();

  NavEngine& engine() { return *engine_; }
  const NavConfig& config() const { return cfg_; }
  const Policy& policy() const { return *policy_; }
  std::shared_ptr<Topomap> topomap() const { return engine_->topomap(); }
  const std::string& topomap_path() const { return topomap_path_; }

  std::function<void(const StepResult&)> on_result;   // 推論のたびに (別スレッドから) 呼ぶ
  std::function<bool()> want_preview;                 // デバッグ画面を見ている人がいるか

 private:
  void loop();
  void ensure_thread();
  std::unique_ptr<NavLogger> make_logger();

  NavConfig cfg_;
  std::unique_ptr<Policy> policy_;
  std::unique_ptr<NavEngine> engine_;
  std::unique_ptr<NavLogger> logger_;
  std::string topomap_path_;
  Clock clock_;
  Log log_;
  TopomapLoader loader_;
  std::recursive_mutex mu_;      // エンジン (推論中は長く持つ)
  std::mutex cmd_mu_;            // 指示値 (すぐ返す)
  bool stopping_ = false;
  std::pair<double, double> cmd_{0.0, 0.0};
  double cmd_time_ = -1e9;
  double stopped_at_ = -1e9;
  bool threaded_ = true;
  std::atomic<bool> alive_{true};
  std::thread thread_;
};

}  // namespace omnivla
