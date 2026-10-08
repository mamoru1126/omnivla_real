// OmniVLA 実機ナビゲーション (ROS 1, C++). 推論は推論サーバ (tools/policy_server.py) に任せる.
//
//   roslaunch omnivla_real_ros1 navigator.launch topomap:=/data/topomaps/course_a
//
// subscribe : robot.yaml の topics.image (CompressedImage / Image), topics.odom (任意), topics.localization (任意),
//             /omnivla/enable (std_msgs/Bool), /omnivla/topomap (std_msgs/String)
// publish   : /cmd_vel (Twist / TwistStamped), /omnivla/path (nav_msgs/Path, base_link 座標の予測軌跡),
//             /omnivla/status (std_msgs/String, JSON)
// ブラウザ  : http://<この PC>:8080 (navigator.yaml の io.web_port)
// カメラ画像はデコードせずに (JPEG のまま) 推論サーバとブラウザに渡す.
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <random>
#include <sstream>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/TwistStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Bool.h>
#include <std_msgs/String.h>

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

Pose pose_of(const geometry_msgs::Pose& p) {
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

class NavigatorNode {
 public:
  NavigatorNode() : pnh_("~"), session_(random_session()) {
    omnivla::AppOptions opt;
    opt.repo_root = pnh_.param<std::string>("repo_root", env_or("OMNIVLA_REAL_ROOT", "/workspace"));
    opt.robot_config = pnh_.param<std::string>("robot_config", opt.repo_root + "/configs/robot.yaml");
    opt.nav_config = pnh_.param<std::string>("nav_config", opt.repo_root + "/configs/navigator.yaml");
    opt.topomap = pnh_.param<std::string>("topomap", "");
    const std::string url = pnh_.param<std::string>("policy_url", "");
    if (!url.empty()) opt.sets.push_back("model.url=" + url);
    const std::string log_dir = pnh_.param<std::string>("log_dir", "");
    if (!log_dir.empty()) opt.sets.push_back("io.log_dir=" + log_dir);
    int web_port_i = 0;
    std::string web_port_s;
    if (pnh_.getParam("web_port", web_port_i))  // rosrun ... _web_port:=8080 (数値)
      opt.sets.push_back("io.web_port=" + std::to_string(web_port_i));
    else if (pnh_.getParam("web_port", web_port_s) && !web_port_s.empty())  // launch (文字列)
      opt.sets.push_back("io.web_port=" + web_port_s);
    // その他の上書き: "engine.tracker.reach_check=pose io.control_rate=20"
    std::istringstream extra(pnh_.param<std::string>("overrides", ""));
    for (std::string kv; extra >> kv;) opt.sets.push_back(kv);

    app_ = std::make_unique<omnivla::NavigatorApp>(
        opt, [] { return ros::Time::now().toSec(); }, [](const std::string& s) { ROS_INFO("%s", s.c_str()); });
    const auto& cfg = app_->config();
    const auto& tp = cfg.robot.topics;
    const auto& io = cfg.io;
    autostart_ = pnh_.param<bool>("autostart", true) && !opt.topomap.empty();
    has_odom_ = !tp.odom.empty();
    has_loc_ = !tp.localization.empty();

    // --- publishers ---
    if (io.cmd_stamped)
      cmd_pub_ = nh_.advertise<geometry_msgs::TwistStamped>(io.cmd_vel, 1);
    else
      cmd_pub_ = nh_.advertise<geometry_msgs::Twist>(io.cmd_vel, 1);
    path_pub_ = nh_.advertise<nav_msgs::Path>(io.path, 1);
    status_pub_ = nh_.advertise<std_msgs::String>(io.status, 1);
    app_->on_result = [this](const omnivla::StepResult& r) { publish_path(r); };

    // --- subscribers ---
    const std::string& img = tp.image;
    const bool compressed = img.size() >= 10 && img.compare(img.size() - 10, 10, "compressed") == 0;
    if (compressed)
      subs_.push_back(nh_.subscribe(img, 1, &NavigatorNode::on_compressed, this, ros::TransportHints().tcpNoDelay()));
    else
      subs_.push_back(nh_.subscribe(img, 1, &NavigatorNode::on_raw, this, ros::TransportHints().tcpNoDelay()));
    if (has_odom_) subs_.push_back(nh_.subscribe(tp.odom, 10, &NavigatorNode::on_odom, this));
    if (has_loc_) {
      if (io.localization_type == "PoseStamped")
        subs_.push_back(nh_.subscribe(tp.localization, 10, &NavigatorNode::on_loc_pose, this));
      else if (io.localization_type == "Odometry")
        subs_.push_back(nh_.subscribe(tp.localization, 10, &NavigatorNode::on_loc_odom, this));
      else
        subs_.push_back(nh_.subscribe(tp.localization, 10, &NavigatorNode::on_loc_cov, this));
    }
    subs_.push_back(nh_.subscribe(io.enable, 1, &NavigatorNode::on_enable, this));
    subs_.push_back(nh_.subscribe(io.topomap, 1, &NavigatorNode::on_topomap, this));

    // --- timers ---
    control_timer_ = nh_.createTimer(ros::Duration(1.0 / std::max(1.0, io.control_rate)), &NavigatorNode::on_control, this);
    status_timer_ = nh_.createTimer(ros::Duration(1.0), &NavigatorNode::on_status, this);
    ROS_INFO("waiting for %s%s%s", img.c_str(), has_odom_ ? (" and " + tp.odom).c_str() : "",
             has_loc_ ? (" and " + tp.localization).c_str() : "");
  }

  ~NavigatorNode() {
    if (app_) app_->runner().shutdown();
  }

 private:
  double stamp_of(const std_msgs::Header& h) const {
    const double t = h.stamp.toSec();
    return t > 0 ? t : ros::Time::now().toSec();
  }

  void on_compressed(const sensor_msgs::CompressedImageConstPtr& msg) {
    auto f = std::make_shared<Frame>();
    f->key = session_ + std::to_string(seq_++);
    f->stamp = stamp_of(msg->header);
    f->kind = "encoded";
    f->format = msg->format;
    f->data = Bytes(&msg->data, [msg](const std::vector<uint8_t>*) {});  // メッセージをそのまま使う (コピーしない)
    app_->runner().engine().on_image(f->stamp, f);
    got_image_ = true;
    maybe_autostart();
  }

  void on_raw(const sensor_msgs::ImageConstPtr& msg) {
    auto f = std::make_shared<Frame>();
    f->key = session_ + std::to_string(seq_++);
    f->stamp = stamp_of(msg->header);
    f->kind = "raw";
    f->encoding = msg->encoding;
    f->width = static_cast<int>(msg->width);
    f->height = static_cast<int>(msg->height);
    f->step = static_cast<int>(msg->step);
    f->is_bigendian = msg->is_bigendian != 0;
    f->data = Bytes(&msg->data, [msg](const std::vector<uint8_t>*) {});
    app_->runner().engine().on_image(f->stamp, f);
    got_image_ = true;
    maybe_autostart();
  }

  void on_odom(const nav_msgs::OdometryConstPtr& msg) {
    app_->runner().engine().on_odom(stamp_of(msg->header), pose_of(msg->pose.pose));
    got_odom_ = true;
  }
  void on_loc_pose(const geometry_msgs::PoseStampedConstPtr& msg) { loc(stamp_of(msg->header), pose_of(msg->pose)); }
  void on_loc_cov(const geometry_msgs::PoseWithCovarianceStampedConstPtr& msg) {
    loc(stamp_of(msg->header), pose_of(msg->pose.pose));
  }
  void on_loc_odom(const nav_msgs::OdometryConstPtr& msg) { loc(stamp_of(msg->header), pose_of(msg->pose.pose)); }
  void loc(double t, const Pose& p) {
    app_->runner().engine().on_localization(t, p);
    got_loc_ = true;
  }

  void on_enable(const std_msgs::BoolConstPtr& msg) {
    if (msg->data)
      app_->runner().start();
    else
      app_->runner().stop();
  }
  void on_topomap(const std_msgs::StringConstPtr& msg) { app_->runner().start(msg->data); }

  void maybe_autostart() {
    if (autostart_ && got_image_ && (got_odom_ || !has_odom_) && (got_loc_ || !has_loc_)) {
      autostart_ = false;
      app_->runner().start();
    }
  }

  void on_control(const ros::TimerEvent&) {
    const auto cmd = app_->runner().command();
    const double now = ros::Time::now().toSec();
    app_->on_command(now, cmd);
    if (!cmd) return;
    geometry_msgs::Twist tw;
    tw.linear.x = cmd->first;
    tw.angular.z = cmd->second;
    const auto& io = app_->config().io;
    if (io.cmd_stamped) {
      geometry_msgs::TwistStamped m;
      m.header.stamp = ros::Time::now();
      m.header.frame_id = io.base_frame;
      m.twist = tw;
      cmd_pub_.publish(m);
    } else {
      cmd_pub_.publish(tw);
    }
  }

  void on_status(const ros::TimerEvent&) {
    std_msgs::String m;
    m.data = app_->runner().status_json().dump();
    status_pub_.publish(m);
  }

  void publish_path(const omnivla::StepResult& r) {
    if (!r.waypoints) return;
    nav_msgs::Path path;
    path.header.stamp = ros::Time::now();
    path.header.frame_id = app_->config().io.base_frame;
    auto add = [&](double x, double y, double yaw) {
      geometry_msgs::PoseStamped ps;
      ps.header = path.header;
      ps.pose.position.x = x;
      ps.pose.position.y = y;
      ps.pose.orientation.z = std::sin(yaw / 2.0);
      ps.pose.orientation.w = std::cos(yaw / 2.0);
      path.poses.push_back(ps);
    };
    add(0.0, 0.0, 0.0);
    for (const auto& p : *r.waypoints) add(p[0], p[1], std::atan2(p[3], p[2]));
    path_pub_.publish(path);
  }

  ros::NodeHandle nh_, pnh_;
  std::string session_;
  std::unique_ptr<omnivla::NavigatorApp> app_;
  ros::Publisher cmd_pub_, path_pub_, status_pub_;
  std::vector<ros::Subscriber> subs_;
  ros::Timer control_timer_, status_timer_;
  std::atomic<uint64_t> seq_{0};
  std::atomic<bool> got_image_{false}, got_odom_{false}, got_loc_{false};
  bool autostart_ = false, has_odom_ = false, has_loc_ = false;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "omnivla_navigator");
  try {
    NavigatorNode node;
    ros::AsyncSpinner spinner(3);
    spinner.start();
    ros::waitForShutdown();
  } catch (const std::exception& e) {
    ROS_FATAL("%s", e.what());
    return 1;
  }
  return 0;
}
