#!/usr/bin/env python3
"""学習ラベルに使う「位置」の候補 (ホイールオドメトリ / 指示値の積算 / 自己位置) を比べる.

OmniVLA のラベルは「この先 約 2.7 秒 (8 フレーム) の動き」なので、その長さの動きが候補どうしで
どれくらい一致するかを見る。あわせて、指示値から実際に動くまでの遅れと、速度の比 (スリップ) も推定する。

  python3 tools/check_odometry.py --robot configs/robot.yaml /data/bags/run1

結果の読み方:
  odom vs cmd の位置の差が小さい (中央値 数 cm)  -> どちらを使っても良い. 既定の odom で OK
  odom vs cmd で向きの差だけ大きい               -> 旋回時にスリップしている可能性 (自己位置があれば比較)
  distance_ratio が 1 から離れる                  -> 指示値どおりに進んでいない (cmd を使うなら注意)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from omnivla_real.extract import read_streams  # noqa: E402
from omnivla_real.odometry import compare_tracks, estimate_delay, integrate_twist  # noqa: E402
from omnivla_real.robot_config import RobotConfig  # noqa: E402
from omnivla_real.runio import add_common_args, open_run, run_name  # noqa: E402


def analyze(streams, horizon: float, sample_rate: float = 3.0) -> dict:
    tracks = {}
    if len(streams.odom_t) > 1:
        tracks["odom"] = streams.odom_track()
        tracks["odom_twist"] = integrate_twist(streams.odom_t, streams.odom_v, streams.odom_w, timeout=0.5)
    res = {"horizon_s": horizon}
    if len(streams.cmd_t) > 1:
        if len(streams.odom_t) > 1:
            d, r = estimate_delay(streams.cmd_t, streams.cmd_w, streams.odom_t, streams.odom_w)
            dv, rv = estimate_delay(streams.cmd_t, streams.cmd_v, streams.odom_t, streams.odom_v)
            res["cmd_to_odom_delay_s"] = {"angular": d, "angular_corr": r, "linear": dv, "linear_corr": rv}
            delay = d if r > 0.3 else (dv if rv > 0.3 else 0.0)
        else:
            delay = 0.0
        tracks["cmd"] = integrate_twist(streams.cmd_t, streams.cmd_v, streams.cmd_w, delay=delay)
        res["cmd_delay_used_s"] = delay
    if len(streams.loc_t) > 1:
        tracks["localization"] = streams.loc_track()
    if not tracks:
        raise SystemExit("no odometry / command / localization in the bag")
    t_lo = max(tr.t[0] for tr in tracks.values())
    t_hi = min(tr.t[-1] for tr in tracks.values()) - horizon
    t0 = np.arange(t_lo, t_hi, 1.0 / sample_rate)
    names = list(tracks)
    res["pairs"] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            res["pairs"][f"{names[i]} vs {names[j]}"] = compare_tracks(tracks[names[i]], tracks[names[j]], t0, horizon)
    # 指示値と実際の速度の比 (動いている時)
    if "odom" in tracks and len(streams.cmd_t) > 1:
        tq = streams.odom_t
        cv = np.interp(tq, streams.cmd_t + res.get("cmd_delay_used_s", 0.0), streams.cmd_v)
        cw = np.interp(tq, streams.cmd_t + res.get("cmd_delay_used_s", 0.0), streams.cmd_w)
        mv = np.abs(cv) > 0.1
        mw = np.abs(cw) > 0.2
        res["odom_over_cmd"] = {
            "linear_median": float(np.median(streams.odom_v[mv] / cv[mv])) if mv.any() else None,
            "angular_median": float(np.median(streams.odom_w[mw] / cw[mw])) if mw.any() else None,
        }
    return res, tracks


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("bag")
    ap.add_argument("--horizon", type=float, default=8 / 3.0, help="比べる動きの長さ [s] (既定 8 フレーム @3Hz)")
    ap.add_argument("--start_sec", type=float)
    ap.add_argument("--end_sec", type=float)
    ap.add_argument("--out", default="", help="結果 (json と図) の保存先ディレクトリ")
    args = ap.parse_args(argv)
    robot = RobotConfig.load(args.robot)
    robot.topics.image = ""  # 画像は読まない
    with open_run(args.bag, args.typestore) as bag:
        streams = read_streams(bag, robot, None, 3.0, args.start_sec, args.end_sec, run_name(args.bag),
                               decode_images=False)
    res, tracks = analyze(streams, args.horizon)
    print(json.dumps(res, indent=1))
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "check_odometry.json"), "w") as f:
            json.dump(res, f, indent=1)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
            for name, tr in tracks.items():
                ax.plot(tr.x - tr.x[0], tr.y - tr.y[0], label=name, lw=1)
            ax.set_aspect("equal")
            ax.legend()
            ax.set_title("paths from each source (start aligned, no rotation)")
            fig.savefig(os.path.join(args.out, "check_odometry.png"))
            print("->", os.path.join(args.out, "check_odometry.png"))
        except ImportError:
            pass


if __name__ == "__main__":
    main()
