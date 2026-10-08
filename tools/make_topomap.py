#!/usr/bin/env python3
"""コースを走った bag からサブゴール画像列 (topomap) を作る.

コースを 1 回走った bag から、一定距離 (--spacing m) ごとの画像を取り出す。ナビゲーション時は
この画像を順にたどって最後の画像 (ゴール) まで走る。

  python3 tools/make_topomap.py --robot configs/robot.yaml --out /data/topomaps/course_a /data/bags/run1
  # bag の一部だけ使う: --start_sec 12 --end_sec 340

出力: <out>/0.jpg ... N.jpg, poses.yaml (各画像を撮った位置. 自己位置があれば map, 無ければ odom 座標),
      topomap.json, overview.png
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from omnivla_real.extract import read_streams  # noqa: E402
from omnivla_real.robot_config import RobotConfig  # noqa: E402
from omnivla_real.runio import add_common_args, open_run, run_name  # noqa: E402
from omnivla_real.topomap_build import build_topomap  # noqa: E402


def plot(out_dir: str, x, y, sel, title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 7), dpi=100)
    ax.plot(x, y, color="#999", lw=1)
    ax.plot(x[sel], y[sel], "o", color="#b7791f", ms=5)
    for k, i in enumerate(sel):
        if k % max(1, len(sel) // 20) == 0 or k == len(sel) - 1:
            ax.annotate(str(k), (x[i], y[i]), fontsize=8)
    ax.plot(x[sel[0]], y[sel[0]], "o", color="green", ms=10, label="start")
    ax.plot(x[sel[-1]], y[sel[-1]], "*", color="red", ms=14, label="goal")
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "overview.png"))
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("bag")
    ap.add_argument("--out", required=True)
    ap.add_argument("--spacing", type=float, default=1.0, help="サブゴールの間隔 [m]")
    ap.add_argument("--start_sec", type=float, help="bag 先頭からの秒数 (コースの開始)")
    ap.add_argument("--end_sec", type=float, help="bag 先頭からの秒数 (コースの終了)")
    ap.add_argument("--pose_source", default="auto", choices=["auto", "odom", "odom_twist", "cmd", "localization"],
                    help="距離を測る位置 (auto: 自己位置があればそれ, 無ければ odom)")
    ap.add_argument("--rate", type=float, default=10.0, help="候補フレームを取り出す周期 [Hz]")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    robot = RobotConfig.load(args.robot)
    tmp = tempfile.mkdtemp(prefix="topomap_")
    try:
        with open_run(args.bag, args.typestore) as bag:
            streams = read_streams(bag, robot, tmp, args.rate, args.start_sec, args.end_sec, run_name(args.bag))
        res = build_topomap(streams, args.out, args.spacing, args.pose_source,
                            meta={"bag": args.bag, "robot": robot.to_dict()}, overwrite=args.overwrite,
                            jpeg_quality=robot.image.jpeg_quality)
        xy = res["path_xy"]
        plot(res["out"], xy[:, 0], xy[:, 1], res["selected"],
             f"{res['num_nodes']} subgoals every {args.spacing} m ({res['pose_source']})")
        m = res["meta"]
        print(f"topomap: {os.path.abspath(args.out)}\n  {res['num_nodes']} subgoals, course {m['course_length_m']:.1f} m "
              f"({m['t_start']:.1f}s - {m['t_end']:.1f}s of the bag), poses in '{res['frame']}' frame")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
