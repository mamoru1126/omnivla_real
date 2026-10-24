#!/usr/bin/env python3
"""学習データの中で「曲がる場面」がどれだけあるかを、しきい値ごとに数える (turn_threshold_deg を決める目安).

  python3 tools/turn_stats.py /data/dataset
  python3 tools/turn_stats.py /data/dataset --horizon 10 --thresholds 10 15 20 30 45

学習の設定 (configs/finetune_*.yaml) の turn_threshold_deg / turn_horizon と同じ判定 (omnivla_real.data_utils.turn_flags).
「曲がる」= この先 horizon フレーム以内に、向きが threshold 度以上変わる、または horizon フレーム先の位置が
threshold 度以上横にある.

bag ごとの運転の仕方 (速さ, 曲がる指令の出し方) も出す. スティックを倒し切って短く曲がる操作が多いと、
モデルは「いつ曲がるか」が分からず遅くて弱い曲がりを覚えるので、label_smooth_sec (正解をならす) の目安にする.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omnivla_real.data_utils import turn_flags  # noqa: E402
from omnivla_real.trajectory_io import find_trajectories, load_meta, load_trajectory  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dirs", nargs="+")
    ap.add_argument("--horizon", type=int, default=10, help="何フレーム先まで見るか (3Hz なら 10 フレーム = 約 3.3 秒)")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[10, 15, 20, 25, 30, 45])
    args = ap.parse_args(argv)

    trajs = find_trajectories(args.data_dirs)
    if not trajs:
        print(f"no trajectories in {args.data_dirs}")
        return 1
    per_bag = defaultdict(lambda: defaultdict(lambda: [0, 0]))   # bag -> thr -> [turn, total]
    max_dyaw = []                                                   # 各フレームの「この先の最大の向きの変化」
    drive = defaultdict(lambda: defaultdict(list))                  # bag -> 値の種類 -> 配列
    for d in trajs:
        data = load_trajectory(d)
        pos, yaw = np.asarray(data["position"]), np.asarray(data["yaw"]).reshape(-1)
        bag = str(load_meta(d).get("bag", "?"))
        n = len(yaw)
        for k in ("cmd_v", "cmd_w", "odom_v", "odom_w"):
            if k in data:
                drive[bag][k].append(np.asarray(data[k], dtype=float).reshape(-1))
        for thr in args.thresholds:
            f = turn_flags(pos, yaw, args.horizon, thr)
            per_bag[bag][thr][0] += int(f.sum())
            per_bag[bag][thr][1] += n
        for t in range(n - 1):
            hi = min(n - 1, t + args.horizon)
            dy = np.abs((yaw[t + 1:hi + 1] - yaw[t] + np.pi) % (2 * np.pi) - np.pi)
            max_dyaw.append(math.degrees(dy.max()))

    thrs = args.thresholds
    print(f"trajectories: {len(trajs)}, horizon: {args.horizon} frames")
    print("曲がる場面の割合 [%]  (しきい値 [deg] ごと)")
    print(f"{'bag':<24}" + "".join(f"{t:>8.0f}" for t in thrs))
    tot = defaultdict(lambda: [0, 0])
    for bag in sorted(per_bag):
        row = per_bag[bag]
        print(f"{bag:<24}" + "".join(f"{100 * row[t][0] / max(1, row[t][1]):>8.1f}" for t in thrs))
        for t in thrs:
            tot[t][0] += row[t][0]
            tot[t][1] += row[t][1]
    print(f"{'(all)':<24}" + "".join(f"{100 * tot[t][0] / max(1, tot[t][1]):>8.1f}" for t in thrs))
    if max_dyaw:
        q = np.percentile(max_dyaw, [50, 90, 95, 99, 100])
        print(f"\n{args.horizon} フレーム以内の向きの変化 [deg]: 中央値 {q[0]:.0f}, 90% {q[1]:.0f}, 95% {q[2]:.0f}, "
              f"99% {q[3]:.0f}, 最大 {q[4]:.0f}")
    print("\n目安: (all) が 5〜20% になるしきい値を configs/finetune_*.yaml の turn_threshold_deg にする")
    print_driving(drive)
    return 0


def driving_style(cmd_v, cmd_w, odom_v, odom_w) -> dict:
    """運転の仕方: 走っている間の速さ、曲がる指令を出していた割合、そのうち「倒し切り」の割合."""
    def cat(a):
        return np.concatenate(a) if a else np.zeros(0)
    cv, cw, ov, ow = cat(cmd_v), cat(cmd_w), cat(odom_v), cat(odom_w)
    v = ov if ov.size and np.isfinite(ov).any() else cv
    moving = np.isfinite(v) & (np.abs(v) > 0.05)
    out = {"frames": int(len(v)), "moving": int(moving.sum())}
    if not moving.any():
        return out
    out["speed_median"] = float(np.median(np.abs(v[moving])))
    if cw.size == v.size and np.isfinite(cw[moving]).any():
        w = np.abs(cw[moving])
        steer = w > 0.02
        out["steer_ratio"] = float(steer.mean())
        if steer.any():
            wmax = float(w[steer].max())
            out["cmd_w_max"] = wmax
            out["full_stick_ratio"] = float((w[steer] >= 0.9 * wmax).mean())   # 倒し切りの割合
    if ow.size == v.size and np.isfinite(ow[moving]).any():
        out["yaw_rate_p90"] = float(np.percentile(np.abs(ow[moving]), 90))
    return out


def print_driving(drive) -> None:
    if not drive:
        return
    print("\n運転の仕方 (走っている間)")
    print(f"{'bag':<24}{'速さ m/s':>10}{'曲がる指令 %':>14}{'倒し切り %':>12}{'最大 rad/s':>12}{'|w| p90':>9}")
    taps = []
    for bag in sorted(drive):
        s = driving_style(*(drive[bag].get(k, []) for k in ("cmd_v", "cmd_w", "odom_v", "odom_w")))
        f = lambda k, m=1.0, d=2: "-" if s.get(k) is None else f"{s[k] * m:.{d}f}"  # noqa: E731
        print(f"{bag:<24}{f('speed_median'):>10}{f('steer_ratio', 100, 1):>14}{f('full_stick_ratio', 100, 0):>12}"
              f"{f('cmd_w_max'):>12}{f('yaw_rate_p90'):>9}")
        if (s.get("full_stick_ratio") or 0) > 0.5:
            taps.append(bag)
    if taps:
        print(f"\n倒し切りが多い bag: {', '.join(taps)}")
        print("  -> 曲がる指令がほぼ「0 か最大」= 短く叩いて曲がっている. いつ曲がるかは画像で決まらないので、")
        print("     configs/finetune_7b.yaml の label_smooth_sec (0.5〜1.0) で正解をならす. 取り直すなら、なめらかに曲がる")


if __name__ == "__main__":
    sys.exit(main())
