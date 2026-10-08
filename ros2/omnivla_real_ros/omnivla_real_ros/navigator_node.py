"""OmniVLA 実機ナビゲーション (ROS 2 ノード).

  ros2 launch omnivla_real_ros navigator.launch.py topomap:=/data/topomaps/course_a

subscribe : robot.yaml の topics.image (Image / CompressedImage), topics.odom (任意), topics.localization (任意)
            /omnivla/enable (std_msgs/Bool), /omnivla/topomap (std_msgs/String)
publish   : /cmd_vel (Twist / TwistStamped), /omnivla/path (nav_msgs/Path, base_link 座標の予測軌跡),
            /omnivla/debug_image (sensor_msgs/Image), /omnivla/status (std_msgs/String, JSON)
中身は omnivla_real.ros_common.NavRunner (ROS1 版と共通).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist, TwistStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, String

REPO = os.environ.get("OMNIVLA_REAL_ROOT", "/workspace")
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from omnivla_real.bag.messages import header_stamp, image_to_rgb, pose_values  # noqa: E402
from omnivla_real.ros_common import NavRunner, path_points, quaternion_from_yaw  # noqa: E402

LOC_TYPES = {"PoseStamped": PoseStamped, "PoseWithCovarianceStamped": PoseWithCovarianceStamped,
             "Odometry": Odometry}


class NavigatorNode(Node):
    def __init__(self):
        super().__init__("omnivla_navigator")
        p = self.declare_parameter
        p("robot_config", os.path.join(REPO, "configs/robot.yaml"))
        p("nav_config", os.path.join(REPO, "configs/navigator.yaml"))
        p("topomap", "")
        p("autostart", True)
        p("model", "")            # navigator.yaml の model.* を上書き (空なら上書きしない)
        p("weights", "")
        p("finetuned_dir", "")
        p("device", "")           # 例 cuda:0 / cpu
        p("log_dir", "")          # 走行ログの保存先 (navigator.yaml の io.log_dir)
        p("policy_url", "")       # 推論サーバ (tools/policy_server.py) を使う場合の URL
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        overrides = {"model.model": g("model"), "model.weights": g("weights"), "model.finetuned_dir": g("finetuned_dir"),
                     "model.device": g("device"), "io.log_dir": g("log_dir")}
        if g("policy_url"):
            overrides.update({"model.model": "remote", "model.url": g("policy_url")})
        self.runner = NavRunner(g("robot_config"), g("nav_config"), g("topomap"), overrides,
                                clock=self._now, log=lambda s: self.get_logger().info(s))
        self.runner.on_result = self._on_result
        io = self.runner.cfg.io
        tp = self.runner.robot.topics
        self.autostart = bool(g("autostart")) and bool(g("topomap"))
        # --- subscribers ---
        img_type = CompressedImage if tp.image.rstrip("/").endswith("compressed") else Image
        self.create_subscription(img_type, tp.image, self._on_image, qos_profile_sensor_data)
        self.has_odom = bool(tp.odom)
        self.has_loc = bool(tp.localization)
        self.got_loc = False
        if tp.odom:
            self.create_subscription(Odometry, tp.odom, self._on_odom, qos_profile_sensor_data)
        if tp.localization:
            self.create_subscription(LOC_TYPES[io.localization_type], tp.localization, self._on_loc, 10)
        self.create_subscription(Bool, io.enable, self._on_enable, 10)
        self.create_subscription(String, io.topomap, self._on_topomap, 10)
        # --- publishers ---
        self.cmd_pub = self.create_publisher(TwistStamped if io.cmd_stamped else Twist, io.cmd_vel, 10)
        self.path_pub = self.create_publisher(Path, io.path, 10)
        self.dbg_pub = self.create_publisher(Image, io.debug_image, QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.status_pub = self.create_publisher(String, io.status, 10)
        self.create_timer(1.0 / io.control_rate, self._publish_cmd)
        self.create_timer(1.0, self._publish_status)
        self.got_image = False
        self.got_odom = False
        self.get_logger().info(f"waiting for {tp.image}" + (f" and {tp.odom}" if tp.odom else ""))

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------------ callbacks
    def _on_image(self, msg) -> None:
        t = header_stamp(msg) or self._now()
        self.runner.engine.on_image(t, lambda m=msg: image_to_rgb(m))
        self.got_image = True
        self._maybe_autostart()

    def _on_odom(self, msg: Odometry) -> None:
        self.runner.engine.on_odom(header_stamp(msg) or self._now(), pose_values(msg))
        self.got_odom = True

    def _on_loc(self, msg) -> None:
        pose = pose_values(msg)
        if pose is not None:
            self.runner.engine.on_localization(header_stamp(msg) or self._now(), pose)
            self.got_loc = True

    def _on_enable(self, msg: Bool) -> None:
        if msg.data:
            self.runner.start()
        else:
            self.runner.stop()

    def _on_topomap(self, msg: String) -> None:
        self.runner.start(msg.data)

    def _maybe_autostart(self) -> None:
        if self.autostart and self.got_image and (self.got_odom or not self.has_odom) \
                and (self.got_loc or not self.has_loc):
            self.autostart = False
            self.runner.start()

    # ------------------------------------------------------------------ outputs
    def _publish_cmd(self) -> None:
        cmd = self.runner.command()
        if cmd is None:
            return
        tw = Twist()
        tw.linear.x, tw.angular.z = float(cmd[0]), float(cmd[1])
        if self.runner.cfg.io.cmd_stamped:
            m = TwistStamped()
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id = self.runner.cfg.io.base_frame
            m.twist = tw
            self.cmd_pub.publish(m)
        else:
            self.cmd_pub.publish(tw)

    def _publish_status(self) -> None:
        self.status_pub.publish(String(data=self.runner.status_json()))

    def _on_result(self, res) -> None:
        stamp = self.get_clock().now().to_msg()
        if res.waypoints is not None:
            path = Path()
            path.header.stamp = stamp
            path.header.frame_id = self.runner.cfg.io.base_frame
            for x, y, yaw in path_points(res.waypoints):
                ps = PoseStamped()
                ps.header = path.header
                ps.pose.position.x, ps.pose.position.y = x, y
                q = quaternion_from_yaw(yaw)
                ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = q
                path.poses.append(ps)
            self.path_pub.publish(path)
        if res.debug is not None and self.dbg_pub.get_subscription_count() > 0:
            arr = np.asarray(res.debug.convert("RGB"))
            m = Image()
            m.header.stamp = stamp
            m.height, m.width = arr.shape[:2]
            m.encoding = "rgb8"
            m.step = arr.shape[1] * 3
            m.data = arr.tobytes()
            self.dbg_pub.publish(m)

    def destroy_node(self):
        self.runner.shutdown()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = NavigatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
