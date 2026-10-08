#!/usr/bin/env python3
"""走行の設定 (robot.yaml + navigator.yaml + 上書き + topomap) を、既定値まで埋めた 1 つの JSON にして出力する.

C++ の ROS 1 ノード (ros1/omnivla_real_ros1) は起動時にこれを呼んで設定を読む。
YAML の読み方と既定値を Python 側 (omnivla_real.nav_config / robot_config / topomap) の 1 か所にまとめるため。

  python3 tools/nav_config_json.py --robot configs/robot.yaml --nav configs/navigator.yaml \\
      --topomap /data/topomaps/course_a --set model.url=http://127.0.0.1:8765 --set io.log_dir=/tmp/nav
  python3 tools/nav_config_json.py --topomap_only /data/topomaps/course_b      # topomap だけ
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from omnivla_real.nav_config import load_nav_config  # noqa: E402
from omnivla_real.robot_config import RobotConfig  # noqa: E402
from omnivla_real.topomap import topomap_index  # noqa: E402


def _value(text: str):
    """--set の値: JSON として読めればその型 (数値・真偽値), だめなら文字列."""
    try:
        return json.loads(text)
    except ValueError:
        return text


def resolved(robot: str, nav: str, overrides: dict, topomap: str = "") -> dict:
    rc = RobotConfig.load(robot)
    nc = load_nav_config(nav if nav and os.path.exists(nav) else None, overrides)
    eng = dataclasses.asdict(nc.engine)
    out = {"robot": rc.to_dict(), "model": dataclasses.asdict(nc.model), "engine": eng, "io": dataclasses.asdict(nc.io)}
    if topomap:
        out["topomap"] = topomap_index(topomap)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", default=os.path.join(REPO, "configs/robot.yaml"))
    ap.add_argument("--nav", default=os.path.join(REPO, "configs/navigator.yaml"))
    ap.add_argument("--topomap", default="")
    ap.add_argument("--topomap_only", default="", help="topomap の一覧だけを出す")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="navigator.yaml の上書き (例 model.url=http://127.0.0.1:8765, engine.tracker.reach_check=pose)")
    args = ap.parse_args(argv)
    if args.topomap_only:
        print(json.dumps(topomap_index(args.topomap_only)))
        return
    overrides = {}
    for item in args.set:
        key, _, val = item.partition("=")
        if not key or not _:
            raise SystemExit(f"--set needs SECTION.KEY=VALUE, got '{item}'")
        overrides[key] = _value(val)
    # engine.tracker.x / engine.controller.x は load_nav_config の上書き (section.key) では書けないので直接入れる
    nested = {k: v for k, v in overrides.items() if k.count(".") >= 2}
    flat = {k: v for k, v in overrides.items() if k.count(".") < 2}
    out = resolved(args.robot, args.nav, flat, args.topomap)
    for k, v in nested.items():
        d = out
        parts = k.split(".")
        for p in parts[:-1]:
            if p not in d:
                raise SystemExit(f"unknown setting '{k}'")
            d = d[p]
        if parts[-1] not in d:
            raise SystemExit(f"unknown setting '{k}'")
        d[parts[-1]] = v
    print(json.dumps(out))


if __name__ == "__main__":
    main()
