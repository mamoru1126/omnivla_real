#!/usr/bin/env python3
"""コースを走った rosbag (ROS1/ROS2) を OmniVLA の学習データに変換する.

  python3 tools/bag_to_dataset.py --robot configs/robot.yaml --config configs/convert.yaml \
      --out /data/dataset /data/bags/run1 /data/bags/run2.bag
  # 分割 bag は 1 走行としてカンマでつなぐ: /data/bags/run3_0.bag,/data/bags/run3_1.bag

出力:
  <out>/<bag名>_<区間>_<塊>/{0.jpg, ..., traj_data.pkl, meta.json}   学習用の軌跡
  <out>/dataset_info.json                                             全体の統計 (metric_waypoint_spacing など)
  <out>/_reports/<bag名>.json, .png                                   bag ごとの変換結果 (どこを使ったか)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from omnivla_real.convert import build_frames, convert_streams, segment_frames, update_dataset_info  # noqa: E402
from omnivla_real.extract import check_clock, frame_period_stats, read_streams  # noqa: E402
from omnivla_real.robot_config import RobotConfig  # noqa: E402
from omnivla_real.runio import add_common_args, load_convert_config, open_run, overrides_from_args, run_name  # noqa: E402

CONVERT_KEYS = ("sample_rate", "pose_source", "cmd_delay", "start_sec", "end_sec", "chunk_sec")


def plot_run(streams, cfg, segs_time, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ft = build_frames(streams, cfg)
    segs = segment_frames(ft, cfg)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), dpi=100)
    axs[0].plot(ft.x, ft.y, color="#bbb", lw=1, label="all frames")
    for k, (a, b) in enumerate(segs):
        axs[0].plot(ft.x[a:b], ft.y[a:b], lw=2, label="used" if k == 0 else None)
    axs[0].set_aspect("equal")
    axs[0].set_title(f"{streams.name}: path ({cfg.pose_source})")
    axs[0].legend(fontsize=8)
    t = ft.t - streams.bag_start
    axs[1].plot(t, ft.cmd_v, label="cmd v [m/s]", lw=1)
    axs[1].plot(t, ft.cmd_w, label="cmd w [rad/s]", lw=1)
    if len(streams.odom_t):
        axs[1].plot(t, ft.odom_v, label="odom v", lw=1, ls=":")
        axs[1].plot(t, ft.odom_w, label="odom w", lw=1, ls=":")
    for a, b in segs:
        axs[1].axvspan(t[a], t[b - 1], color="#2e8b57", alpha=0.12)
    axs[1].set_xlabel("time [s]  (green = used for training)")
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--config", default="configs/convert.yaml", help="変換の設定 (convert.yaml)")
    ap.add_argument("--out", required=True, help="出力ディレクトリ (学習の data_dirs に指定する)")
    ap.add_argument("bags", nargs="+")
    ap.add_argument("--sample_rate", type=float)
    ap.add_argument("--pose_source", choices=["odom", "odom_twist", "cmd", "localization"])
    ap.add_argument("--cmd_delay", type=float)
    ap.add_argument("--start_sec", type=float, help="bag 先頭からの秒数 (使う範囲の開始)")
    ap.add_argument("--end_sec", type=float)
    ap.add_argument("--chunk_sec", type=float)
    ap.add_argument("--name", default="", help="出力名 (bag が 1 つのとき. 既定は bag 名)")
    ap.add_argument("--overwrite", action="store_true", help="同じ名前の軌跡があれば作り直す")
    ap.add_argument("--keep_frames", action="store_true", help="間引いた画像のキャッシュを消さない")
    args = ap.parse_args(argv)

    robot = RobotConfig.load(args.robot)
    cfg = load_convert_config(args.config if os.path.exists(args.config) else None,
                              overrides_from_args(args, CONVERT_KEYS))
    out = os.path.abspath(os.path.expanduser(args.out))
    os.makedirs(os.path.join(out, "_reports"), exist_ok=True)
    total = 0
    for spec in args.bags:
        name = args.name if args.name and len(args.bags) == 1 else run_name(spec)
        cache = os.path.join(out, "_frames", name)
        shutil.rmtree(cache, ignore_errors=True)
        print(f"== {name}: reading {spec}")
        with open_run(spec, args.typestore) as bag:
            streams = read_streams(bag, robot, cache, cfg.sample_rate, cfg.start_sec, cfg.end_sec, name,
                                   progress=lambda s: print("   ", s, flush=True))
        info = streams.summary()
        for w in check_clock(streams):
            print("  WARNING:", w)
        print(f"   images {info['images_total']} ({info['image_rate_hz']:.1f} Hz) -> saved {info['images_saved']} "
              f"at {cfg.sample_rate} Hz, cmd {info['cmd_rate_hz']:.1f} Hz, odom {info['odom_rate_hz']:.1f} Hz")
        res = convert_streams(streams, cfg, out, name,
                              extra_meta={"bag_path": spec, "robot": robot.to_dict(),
                                          "image_size": info["image_size"]}, overwrite=args.overwrite)
        total += len(res["trajectories"])
        print(f"   frames {res['frames']} (valid {res['valid_frames']}, moving {res['moving_frames']}) -> "
              f"used {res['used_frames']} in {res['segments']} segments -> {len(res['trajectories'])} trajectories")
        report = {"streams": info, "frame_period": frame_period_stats(streams.image_t),
                  "clock_warnings": check_clock(streams), **{k: v for k, v in res.items() if k != "trajectories"},
                  "trajectories": [os.path.basename(t) for t in res["trajectories"]]}
        with open(os.path.join(out, "_reports", f"{name}.json"), "w") as f:
            json.dump(report, f, indent=1)
        plot_run(streams, cfg, res["segments_time"], os.path.join(out, "_reports", f"{name}.png"))
        if not args.keep_frames:
            shutil.rmtree(cache, ignore_errors=True)
    if not args.keep_frames:
        shutil.rmtree(os.path.join(out, "_frames"), ignore_errors=True)
    info = update_dataset_info(out)
    print(f"\n{total} trajectories written. dataset: {info['trajectories']} trajectories, {info['frames']} frames, "
          f"{info['path_length_m']:.0f} m, metric_waypoint_spacing ~ {info['metric_waypoint_spacing']}")
    if info.get("warning"):
        print("WARNING:", info["warning"])


if __name__ == "__main__":
    main()
