#!/usr/bin/env python3
"""OmniVLA 実機ナビゲーション (ROS 1 ノード). 中身は ROS 2 版と同じ omnivla_real.ros_common.NavRunner.

  roslaunch omnivla_real_ros1 navigator.launch topomap:=/data/topomaps/course_a
  rosrun omnivla_real_ros1 navigator_node.py _topomap:=/data/topomaps/course_a

subscribe : robot.yaml の topics.image / topics.odom (任意) / topics.localization (任意),
            /omnivla/enable (std_msgs/Bool), /omnivla/topomap (std_msgs/String)
publish   : /cmd_vel, /omnivla/path (nav_msgs/Path), /omnivla/debug_image, /omnivla/status
推論は 2 通り:
  * policy_url:=http://127.0.0.1:8765  推論サーバ (tools/policy_server.py, 別コンテナの PyTorch) を呼ぶ (推奨).
                                       このノードは rospy + numpy + PIL + PyYAML だけで動く (docker/Dockerfile.ros1)
  * policy_url 無し                    このノードの中で推論する (rospy と PyTorch が同じ Python に必要)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist, TwistStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, String

REPO = os.environ.get("OMNIVLA_REAL_ROOT", "/workspace")
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from omnivla_real.bag.messages import header_stamp, image_to_rgb, pose_values  # noqa: E402
from omnivla_real.ros_common import NavRunner, path_points, quaternion_from_yaw  # noqa: E402

LOC_TYPES = {"PoseStamped": PoseStamped, "PoseWithCovarianceStamped": PoseWithCovarianceStamped,
             "Odometry": Odometry}


class NavigatorNode:
    def __init__(self):
        g = lambda n, d: rospy.get_param("~" + n, d)  # noqa: E731
        overrides = {"model.model": g("model", ""), "model.weights": g("weights", ""),
                     "model.finetuned_dir": g("finetuned_dir", ""), "model.device": g("device", ""),
                     "io.log_dir": g("log_dir", "")}
        if g("policy_url", ""):   # 推論サーバ (tools/policy_server.py) を使う. このノードは torch 不要
            overrides.update({"model.model": "remote", "model.url": g("policy_url", "")})
        topomap = g("topomap", "")
        self.runner = NavRunner(g("robot_config", os.path.join(REPO, "configs/robot.yaml")),
                                g("nav_config", os.path.join(REPO, "configs/navigator.yaml")), topomap, overrides,
                                clock=lambda: rospy.Time.now().to_sec(), log=rospy.loginfo)
        self.runner.on_result = self._on_result
        io = self.runner.cfg.io
        tp = self.runner.robot.topics
        self.autostart = bool(g("autostart", True)) and bool(topomap)
        self.has_odom = bool(tp.odom)
        self.has_loc = bool(tp.localization)
        self.got_loc = False
        self.got_image = self.got_odom = False
        img_type = CompressedImage if tp.image.rstrip("/").endswith("compressed") else Image
        rospy.Subscriber(tp.image, img_type, self._on_image, queue_size=1, buff_size=2 ** 24)
        if tp.odom:
            rospy.Subscriber(tp.odom, Odometry, self._on_odom, queue_size=10)
        if tp.localization:
            rospy.Subscriber(tp.localization, LOC_TYPES[io.localization_type], self._on_loc, queue_size=10)
        rospy.Subscriber(io.enable, Bool, self._on_enable, queue_size=1)
        rospy.Subscriber(io.topomap, String, self._on_topomap, queue_size=1)
        self.cmd_pub = rospy.Publisher(io.cmd_vel, TwistStamped if io.cmd_stamped else Twist, queue_size=1)
        self.path_pub = rospy.Publisher(io.path, Path, queue_size=1)
        self.dbg_pub = rospy.Publisher(io.debug_image, Image, queue_size=1)
        self.status_pub = rospy.Publisher(io.status, String, queue_size=1)
        rospy.Timer(rospy.Duration(1.0 / io.control_rate), self._publish_cmd)
        rospy.Timer(rospy.Duration(1.0), lambda _: self.status_pub.publish(String(data=self.runner.status_json())))
        rospy.on_shutdown(self.runner.shutdown)
        rospy.loginfo(f"waiting for {tp.image}" + (f" and {tp.odom}" if tp.odom else ""))

    def _now(self) -> float:
        return rospy.Time.now().to_sec()

    def _on_image(self, msg) -> None:
        self.runner.engine.on_image(header_stamp(msg) or self._now(), lambda m=msg: image_to_rgb(m))
        self.got_image = True
        self._maybe_autostart()

    def _on_odom(self, msg) -> None:
        self.runner.engine.on_odom(header_stamp(msg) or self._now(), pose_values(msg))
        self.got_odom = True

    def _on_loc(self, msg) -> None:
        pose = pose_values(msg)
        if pose is not None:
            self.runner.engine.on_localization(header_stamp(msg) or self._now(), pose)
            self.got_loc = True

    def _on_enable(self, msg) -> None:
        self.runner.start() if msg.data else self.runner.stop()

    def _on_topomap(self, msg) -> None:
        self.runner.start(msg.data)

    def _maybe_autostart(self) -> None:
        if self.autostart and self.got_image and (self.got_odom or not self.has_odom) \
                and (self.got_loc or not self.has_loc):
            self.autostart = False
            self.runner.start()

    def _publish_cmd(self, _event) -> None:
        cmd = self.runner.command()
        if cmd is None:
            return
        tw = Twist()
        tw.linear.x, tw.angular.z = float(cmd[0]), float(cmd[1])
        if self.runner.cfg.io.cmd_stamped:
            m = TwistStamped()
            m.header.stamp = rospy.Time.now()
            m.header.frame_id = self.runner.cfg.io.base_frame
            m.twist = tw
            self.cmd_pub.publish(m)
        else:
            self.cmd_pub.publish(tw)

    def _on_result(self, res) -> None:
        stamp = rospy.Time.now()
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
        if res.debug is not None and self.dbg_pub.get_num_connections() > 0:
            arr = np.asarray(res.debug.convert("RGB"))
            m = Image()
            m.header.stamp = stamp
            m.height, m.width = arr.shape[:2]
            m.encoding = "rgb8"
            m.step = arr.shape[1] * 3
            m.data = arr.tobytes()
            self.dbg_pub.publish(m)


def main():
    rospy.init_node("omnivla_navigator")
    NavigatorNode()
    rospy.spin()


if __name__ == "__main__":
    main()
