// ROS ノードと試験用プログラム (core_check replay) の共通部分:
// 設定 (tools/nav_config_json.py) -> 推論サーバへの接続 -> NavRunner -> デバッグ画面
#pragma once

#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "omnivla/config.hpp"
#include "omnivla/runner.hpp"
#include "omnivla/web_ui.hpp"

namespace omnivla {

struct AppOptions {
  std::string repo_root;              // omnivla_real (tools/nav_config_json.py を呼ぶ)
  std::string robot_config, nav_config, topomap;
  std::vector<std::string> sets;      // "section.key=value" (navigator.yaml の上書き)
};

// 設定を読む (python3 tools/nav_config_json.py)
NavConfig load_nav_config(const AppOptions& opt);
// topomap を読む (poses.yaml の解釈は Python 側と同じ)
std::shared_ptr<Topomap> load_topomap_dir(const std::string& repo_root, const std::string& path);

class NavigatorApp {
 public:
  NavigatorApp(const AppOptions& opt, NavRunner::Clock clock, NavRunner::Log log, bool threaded = true);
  ~NavigatorApp();

  NavRunner& runner() { return *runner_; }
  const NavConfig& config() const { return cfg_; }
  WebUi* web() { return web_.get(); }
  // 指示値を出した時に呼ぶ (画面の更新. 5Hz 程度に間引く)
  void on_command(double t, const std::optional<std::pair<double, double>>& cmd);
  // 推論結果を受け取る (ROS の Path を出すなど). runner の on_result の代わりにこちらを使う
  std::function<void(const StepResult&)> on_result;

 private:
  nlohmann::json web_config() const;
  nlohmann::json step_state(const StepResult& r);

  AppOptions opt_;
  NavConfig cfg_;
  NavRunner::Log log_;
  std::unique_ptr<NavRunner> runner_;
  std::unique_ptr<WebUi> web_;
  double last_web_status_ = -1e9;
  int web_topomap_id_ = -1;
  std::optional<std::pair<double, double>> last_cmd_;
  std::mutex cmd_mu_;
  // 画面に出す出来事 (番号付き. 画面は前に受け取った番号より後のものだけを足す)
  std::deque<std::pair<int, std::string>> event_log_;
  int event_id_ = 0;
};

}  // namespace omnivla
