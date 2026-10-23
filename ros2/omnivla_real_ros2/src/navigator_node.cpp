// OmniVLA 実機ナビゲーション (ROS 2, C++). 推論は推論サーバ (tools/policy_server.py) に任せる.
// 中身 (サブゴールの切り替え・制御・走行ログ・デバッグ画面) は ROS 1 の C++ ノードと同じ (ros1/omnivla_real_ros1/src/core).
//
//   ros2 run omnivla_real_ros2 navigator --ros-args -p topomap:=/data/topomaps/course_a
//   ros2 launch omnivla_real_ros2 navigator.launch.py topomap:=/data/topomaps/course_a
//
// subscribe : robot.yaml の topics.image (CompressedImage / Image), topics.odom (任意), topics.localization (任意),
//             /omnivla/enable (std_msgs/Bool), /omnivla/topomap (std_msgs/String)
// publish   : navigator.yaml の io.cmd_vel (Twist / TwistStamped), /omnivla/path (nav_msgs/Path, base_link 座標の予測軌跡),
//             /omnivla/status (std_msgs/String, JSON)
// ブラウザ  : http://<この PC>:8080 (navigator.yaml の io.web_port)
// カメラ画像はデコードせずに (JPEG のまま) 推論サーバとブラウザに渡す.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>

#include "omnivla/app.hpp"
#include "omnivla/geometry.hpp"

using omnivla::Bytes;
using omnivla::Frame;
using omnivla::Pose;

namespace {

std::string env_or(const char* name, const std::string& def) {
  const char* v = std::getenv(name);
  return v && *v ? v : def;
}

Pose pose_of(const geometry_msgs::msg::Pose& p) {
  return {p.position.x, p.position.y,
          omnivla::yaw_from_quaternion(p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)};
}

std::string random_session() {
  std::random_device rd;
  std::ostringstream os;
  os << "n" << std::hex << (rd() & 0xffffff) << "_";
  return os.str();
}

}  // namespace

class NavigatorNode : public rclcpp::Node {
 public:
  NavigatorNode() : rclcpp::Node("omnivla_navigator"), session_(random_session()) {
    omnivla::AppOptions opt;
    opt.repo_root = declare_parameter<std::string>("repo_root", env_or("OMNIVLA_REAL_ROOT", "/workspace"));
    opt.robot_config = declare_parameter<std::string>("robot_config", opt.repo_root + "/configs/robot.yaml");
    opt.nav_config = declare_parameter<std::string>("nav_config", opt.repo_root + "/configs/navigator.yaml");
    opt.topomap = declare_parameter<std::string>("topomap", "");
    const bool autostart = declare_parameter<bool>("autostart", true);
    const std::string url = declare_parameter<std::string>("policy_url", "");
    if (!url.empty()) opt.sets.push_back("model.url=" + url);
    const std::string log_dir = declare_parameter<std::string>("log_dir", "");
    if (!log_dir.empty()) opt.sets.push_back("io.log_dir=" + log_dir);
    // web_port: -p web_port:=8080 (数値) でも launch の文字列でもよい. 空なら navigator.yaml の io.web_port
    rcl_interfaces::msg::ParameterDescriptor any_type;
    any_type.dynamic_typing = true;
    const auto web_port = declare_parameter("web_port", rclcpp::ParameterValue(std::string("")), any_type);
    if (web_port.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER)
      opt.sets.push_back("io.web_port=" + std::to_string(web_port.get<int64_t>()));
    else if (web_port.get_type() == rclcpp::ParameterType::PARAMETER_STRING && !web_port.get<std::string>().empty())
      opt.sets.push_back("io.web_port=" + web_port.get<std::string>());
    // その他の上書き: "engine.tracker.reach_check=pose io.cmd_vel=/omnivla/cmd_vel/raw"
    std::istringstream extra(declare_parameter<std::string>("overrides", ""));
    for (std::string kv; extra >> kv;) opt.sets.push_back(kv);

    app_ = std::make_unique<omnivla::NavigatorApp>(
        opt, [this] { return now().seconds(); },
        [this](const std::string& s) { RCLCPP_INFO(get_logger(), "%s", s.c_str()); });
    const auto& cfg = app_->config();
    const auto& tp = cfg.robot.topics;
    const auto& io = cfg.io;
    autostart_ = autostart && !opt.topomap.empty();
    has_odom_ = !tp.odom.empty();
    has_loc_ = !tp.localization.empty();

    // 画像の受信・タイマー・推論結果を並行して扱う (ROS 1 の AsyncSpinner と同じ. 中身はスレッドセーフ)
    group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    rclcpp::SubscriptionOptions so;
    so.callback_group = group_;

    // --- publishers ---
    if (io.cmd_stamped)
      cmd_stamped_pub_ = create_publisher<geometry_msgs::msg::TwistStamped>(io.cmd_vel, 10);
    else
      cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>(io.cmd_vel, 10);
    path_pub_ = create_publisher<nav_msgs::msg::Path>(io.path, 10);
    status_pub_ = create_publisher<std_msgs::msg::String>(io.status, 10);
    app_->on_result = [this](const omnivla::StepResult& r) { publish_path(r); };

    // --- subscribers --- (カメラ・オドメトリは sensor data: 古いものは捨てる. reliable の送り手ともつながる)
    const std::string& img = tp.image;
    const bool compressed = img.size() >= 10 && img.compare(img.size() - 10, 10, "compressed") == 0;
    const auto sensor = rclcpp::SensorDataQoS().keep_last(1);
    if (compressed)
      compressed_sub_ = create_subscription<sensor_msgs::msg::CompressedImage>(
          img, sensor, [this](sensor_msgs::msg::CompressedImage::ConstSharedPtr m) { on_compressed(m); }, so);
    else
      raw_sub_ = create_subscription<sensor_msgs::msg::Image>(
          img, sensor, [this](sensor_msgs::msg::Image::ConstSharedPtr m) { on_raw(m); }, so);
    if (has_odom_)
      odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
          tp.odom, rclcpp::SensorDataQoS().keep_last(10),
          [this](nav_msgs::msg::Odometry::ConstSharedPtr m) {
            app_->runner().engine().on_odom(stamp_of(m->header), pose_of(m->pose.pose));
            got_odom_ = true;
          },
          so);
    if (has_loc_) {
      if (io.localization_type == "PoseStamped")
        loc_pose_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
            tp.localization, 10,
            [this](geometry_msgs::msg::PoseStamped::ConstSharedPtr m) { loc(stamp_of(m->header), pose_of(m->pose)); }, so);
      else if (io.localization_type == "Odometry")
        loc_odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            tp.localization, 10,
            [this](nav_msgs::msg::Odometry::ConstSharedPtr m) { loc(stamp_of(m->header), pose_of(m->pose.pose)); }, so);
      else
        loc_cov_sub_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            tp.localization, 10,
            [this](geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr m) {
              loc(stamp_of(m->header), pose_of(m->pose.pose));
            },
            so);
    }
    enable_sub_ = create_subscription<std_msgs::msg::Bool>(
        io.enable, 10,
        [this](std_msgs::msg::Bool::ConstSharedPtr m) {
          if (m->data)
            app_->runner().start();
          else
            app_->runner().stop();
        },
        so);
    topomap_sub_ = create_subscription<std_msgs::msg::String>(
        io.topomap, 10, [this](std_msgs::msg::String::ConstSharedPtr m) { app_->runner().start(m->data); }, so);

    // --- timers (ROS の時刻. bag をシミュレーション時刻で流すときも合う) ---
    control_timer_ = rclcpp::create_timer(this, get_clock(),
                                          rclcpp::Duration::from_seconds(1.0 / std::max(1.0, io.control_rate)),
                                          [this] { on_control(); }, group_);
    status_timer_ = rclcpp::create_timer(this, get_clock(), rclcpp::Duration::from_seconds(1.0),
                                         [this] { on_status(); }, group_);
    RCLCPP_INFO(get_logger(), "waiting for %s%s%s", img.c_str(), has_odom_ ? (" and " + tp.odom).c_str() : "",
                has_loc_ ? (" and " + tp.localization).c_str() : "");
  }

  ~NavigatorNode() override {
    if (app_) app_->runner().shutdown();
  }

 private:
  double stamp_of(const std_msgs::msg::Header& h) {
    const double t = rclcpp::Time(h.stamp).seconds();
    return t > 0 ? t : now().seconds();
  }

  void on_compressed(const sensor_msgs::msg::CompressedImage::ConstSharedPtr& msg) {
    auto f = std::make_shared<Frame>();
    f->key = session_ + std::to_string(seq_++);
    f->stamp = stamp_of(msg->header);
    f->kind = "encoded";
    f->format = msg->format;
    f->data = Bytes(msg, &msg->data);  // メッセージをそのまま使う (コピーしない)
    app_->runner().engine().on_image(f->stamp, f);
    got_image_ = true;
    maybe_autostart();
  }

  void on_raw(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    auto f = std::make_shared<Frame>();
    f->key = session_ + std::to_string(seq_++);
    f->stamp = stamp_of(msg->header);
    f->kind = "raw";
    f->encoding = msg->encoding;
    f->width = static_cast<int>(msg->width);
    f->height = static_cast<int>(msg->height);
    f->step = static_cast<int>(msg->step);
    f->is_bigendian = msg->is_bigendian != 0;
    f->data = Bytes(msg, &msg->data);
    app_->runner().engine().on_image(f->stamp, f);
    got_image_ = true;
    maybe_autostart();
  }

  void loc(double t, const Pose& p) {
    app_->runner().engine().on_localization(t, p);
    got_loc_ = true;
  }

  void maybe_autostart() {
    bool expected = true;
    if (got_image_ && (got_odom_ || !has_odom_) && (got_loc_ || !has_loc_) &&
        autostart_.compare_exchange_strong(expected, false))
      app_->runner().start();
  }

  void on_control() {
    const auto cmd = app_->runner().command();
    app_->on_command(now().seconds(), cmd);
    if (!cmd) return;
    geometry_msgs::msg::Twist tw;
    tw.linear.x = cmd->first;
    tw.angular.z = cmd->second;
    if (cmd_stamped_pub_) {
      geometry_msgs::msg::TwistStamped m;
      m.header.stamp = now();
      m.header.frame_id = app_->config().io.base_frame;
      m.twist = tw;
      cmd_stamped_pub_->publish(m);
    } else {
      cmd_pub_->publish(tw);
    }
  }

  void on_status() {
    std_msgs::msg::String m;
    m.data = app_->runner().status_json().dump();
    status_pub_->publish(m);
  }

  void publish_path(const omnivla::StepResult& r) {
    if (!r.waypoints) return;
    nav_msgs::msg::Path path;
    path.header.stamp = now();
    path.header.frame_id = app_->config().io.base_frame;
    auto add = [&](double x, double y, double yaw) {
      geometry_msgs::msg::PoseStamped ps;
      ps.header = path.header;
      ps.pose.position.x = x;
      ps.pose.position.y = y;
      ps.pose.orientation.z = std::sin(yaw / 2.0);
      ps.pose.orientation.w = std::cos(yaw / 2.0);
      path.poses.push_back(ps);
    };
    add(0.0, 0.0, 0.0);
    for (const auto& p : *r.waypoints) add(p[0], p[1], std::atan2(p[3], p[2]));
    path_pub_->publish(path);
  }

  std::string session_;
  std::unique_ptr<omnivla::NavigatorApp> app_;
  rclcpp::CallbackGroup::SharedPtr group_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr cmd_stamped_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr compressed_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr raw_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_, loc_odom_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr loc_pose_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr loc_cov_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr enable_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr topomap_sub_;
  rclcpp::TimerBase::SharedPtr control_timer_, status_timer_;
  std::atomic<uint64_t> seq_{0};
  std::atomic<bool> got_image_{false}, got_odom_{false}, got_loc_{false}, autostart_{false};
  bool has_odom_ = false, has_loc_ = false;
};

int main(int argc, char** argv) {
  // ログをファイルに流しても溜め込まない
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  rclcpp::init(argc, argv);
  int rc = 0;
  try {
    auto node = std::make_shared<NavigatorNode>();
    rclcpp::executors::MultiThreadedExecutor exec(rclcpp::ExecutorOptions(), 3);
    exec.add_node(node);
    exec.spin();
  } catch (const std::exception& e) {
    RCLCPP_FATAL(rclcpp::get_logger("omnivla_navigator"), "%s", e.what());
    rc = 1;
  }
  rclcpp::shutdown();
  return rc;
}
