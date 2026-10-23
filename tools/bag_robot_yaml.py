#!/usr/bin/env python3
"""その場で録った bag 用の robot.yaml を作る (topomap 作成・机上評価で使う. bin/field.sh が呼ぶ).

configs/robot.yaml を元に、topics を bag の中にあるものに合わせる (画像の前処理などはそのまま):
  image        : robot.yaml の名前が無ければ、bag の中の画像トピック (CompressedImage を優先)
  odom         : 無ければ bag の中の Odometry (それも無ければ空 = 使わない)
  cmd          : robot.yaml の名前 → /cmd_vel → CMD_VEL_TOPIC (/omnivla/cmd_vel/raw) の順で bag にあるもの (無ければ空)
  localization : bag に無ければ空

  python3 tools/bag_robot_yaml.py /workspace/log/bags/course_20261024_100000   # -> <bag>.robot.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

import yaml

ROOT = os.environ.get("OMNIVLA_REAL_ROOT", "/workspace")
sys.path.insert(0, ROOT)
from omnivla_real.bag.messages import short_type  # noqa: E402
from omnivla_real.runio import open_run  # noqa: E402


def pick(topics: dict, want: str, types: tuple, extra: tuple = ()) -> str:
    """want が bag にあればそれ. 無ければ extra の名前, それも無ければ型が合う最初のトピック."""
    for name in (want, *extra):
        if name and name in topics:
            return name
    for name, (mt, _n) in sorted(topics.items()):
        if short_type(mt) in types:
            return name
    return ""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag")
    ap.add_argument("--robot", default=os.path.join(ROOT, "configs/robot.yaml"))
    ap.add_argument("--out", default="", help="既定: <bag>.robot.yaml")
    ap.add_argument("--typestore", default="ROS2_HUMBLE")
    args = ap.parse_args(argv)
    with open(args.robot) as f:
        cfg = yaml.safe_load(f) or {}
    tp = cfg.setdefault("topics", {})
    with open_run(args.bag, args.typestore) as bag:
        topics = bag.topics()
    print("bag のトピック:")
    for k, (mt, n) in sorted(topics.items()):
        print(f"  {k}  [{mt}]  {n} 件")
    before = {k: tp.get(k, "") for k in ("image", "odom", "cmd", "localization")}
    tp["image"] = pick(topics, tp.get("image", ""), ("sensor_msgs/CompressedImage",)) or \
        pick(topics, "", ("sensor_msgs/Image",))
    tp["odom"] = pick(topics, tp.get("odom", ""), ("nav_msgs/Odometry",))
    cmd_env = os.environ.get("CMD_VEL_TOPIC", "") or "/omnivla/cmd_vel/raw"
    tp["cmd"] = pick(topics, tp.get("cmd", ""), (), ("/cmd_vel", cmd_env))
    loc = tp.get("localization", "")
    tp["localization"] = loc if loc in topics else ""
    if not tp["image"]:
        sys.exit("bag に画像のトピックがありません")
    out = args.out or args.bag.rstrip("/") + ".robot.yaml"
    with open(out, "w") as f:
        f.write(f"# {args.bag} 用 (tools/bag_robot_yaml.py が {args.robot} から作成)\n")
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    for k in ("image", "odom", "cmd", "localization"):
        mark = "" if before[k] == tp[k] else f"   (robot.yaml: {before[k] or '空'})"
        print(f"  topics.{k}: {tp[k] or '(使わない)'}{mark}")
    if not tp["odom"]:
        print("注意: オドメトリが無いので、topomap は指示値の積算 (--pose_source cmd) か時間で作ることになります")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
