#!/usr/bin/env python3
"""ROS ノードの出力を一定時間受けて確認する (tests/ros_node_test.sh から使う).

  python3 tests/ros_check.py ros2 60      # ROS 2: 60 秒間 /cmd_vel, /omnivla/path, /omnivla/status を受ける
  python3 tests/ros_check.py ros1 60      # ROS 1
"""
from __future__ import annotations

import json
import sys
import time


def check(got: dict) -> int:
    cmds, paths, status = got["cmd"], got["path"], got["status"]
    nz = sum(abs(v) > 1e-3 or abs(w) > 1e-3 for v, w in cmds)
    print(f"/cmd_vel: {len(cmds)} msgs ({nz} non-zero)")
    if cmds:
        vs = [c[0] for c in cmds]
        ws = [c[1] for c in cmds]
        print(f"   v in [{min(vs):.3f}, {max(vs):.3f}]  w in [{min(ws):.3f}, {max(ws):.3f}]")
    print(f"/omnivla/path: {len(paths)} msgs, poses per path {sorted(set(paths))}")
    states = [s.get("state") for s in status]
    print(f"/omnivla/status: {len(status)} msgs, states {sorted(set(map(str, states)))}")
    if status:
        print("   last:", json.dumps(status[-1]))
    ok = True
    if len(paths) < 3 or any(n != 9 for n in paths):
        print("NG: /omnivla/path should have 9 poses (origin + 8 waypoints)")
        ok = False
    if len(cmds) < 10:
        print("NG: too few /cmd_vel")
        ok = False
    if nz == 0:
        print("NG: /cmd_vel is always zero")
        ok = False
    if "running" not in states:
        print("NG: never running")
        ok = False
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


def run_ros2(duration: float) -> dict:
    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Path
    from std_msgs.msg import String

    rclpy.init()
    node = rclpy.create_node("omnivla_check")
    got = {"cmd": [], "path": [], "status": []}
    node.create_subscription(Twist, "/cmd_vel", lambda m: got["cmd"].append((m.linear.x, m.angular.z)), 10)
    node.create_subscription(Path, "/omnivla/path", lambda m: got["path"].append(len(m.poses)), 10)
    node.create_subscription(String, "/omnivla/status", lambda m: got["status"].append(json.loads(m.data)), 10)
    t_end = time.time() + duration
    while time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()
    return got


def run_ros1(duration: float) -> dict:
    import rospy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Path
    from std_msgs.msg import String

    rospy.init_node("omnivla_check", anonymous=True)
    got = {"cmd": [], "path": [], "status": []}
    rospy.Subscriber("/cmd_vel", Twist, lambda m: got["cmd"].append((m.linear.x, m.angular.z)))
    rospy.Subscriber("/omnivla/path", Path, lambda m: got["path"].append(len(m.poses)))
    rospy.Subscriber("/omnivla/status", String, lambda m: got["status"].append(json.loads(m.data)))
    t_end = time.time() + duration
    while time.time() < t_end and not rospy.is_shutdown():
        time.sleep(0.1)
    return got


if __name__ == "__main__":
    which, dur = sys.argv[1], float(sys.argv[2])
    sys.exit(check(run_ros2(dur) if which == "ros2" else run_ros1(dur)))
