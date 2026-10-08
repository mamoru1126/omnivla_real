#!/usr/bin/env python3
"""bag の中身 (トピック・型・数・周期) を表示し、robot.yaml のトピック設定の候補を出す.

  python3 tools/bag_info.py /data/bags/course_run1            # ROS2 bag (ディレクトリ)
  python3 tools/bag_info.py /data/bags/course_run1.bag        # ROS1 bag
"""
from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from omnivla_real.bag.messages import short_type  # noqa: E402
from omnivla_real.runio import open_run  # noqa: E402

CANDIDATES = {
    "image": ("sensor_msgs/CompressedImage", "sensor_msgs/Image"),
    "odom": ("nav_msgs/Odometry",),
    "cmd": ("geometry_msgs/Twist", "geometry_msgs/TwistStamped"),
    "localization": ("geometry_msgs/PoseWithCovarianceStamped", "geometry_msgs/PoseStamped", "nav_msgs/Odometry",
                     "tf2_msgs/TFMessage"),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", help="bag (分割 bag は a_0.bag,a_1.bag のようにカンマでつなぐ)")
    ap.add_argument("--typestore", default="ROS2_HUMBLE")
    args = ap.parse_args(argv)
    with open_run(args.bag, args.typestore) as bag:
        dur = bag.end_time - bag.start_time
        topics = bag.topics()
        print(f"bag: {args.bag}\nduration: {dur:.1f} s\n")
        print(f"{'topic':45s} {'type':42s} {'count':>8s} {'Hz':>7s}")
        for t, (mt, n) in sorted(topics.items()):
            print(f"{t:45s} {short_type(mt):42s} {n:8d} {n / dur if dur > 0 else 0:7.1f}")
    print("\nrobot.yaml の topics の候補:")
    for key, types in CANDIDATES.items():
        found = [t for t, (mt, _) in sorted(topics.items()) if short_type(mt) in types]
        if key == "localization":
            found = [t for t in found if "odom" not in t.lower() or "loc" in t.lower()]
        print(f"  {key:13s}: {', '.join(found) if found else '(なし)'}")


if __name__ == "__main__":
    main()
