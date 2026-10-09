#!/usr/bin/env python3
"""学習データの中で「曲がる場面」がどれだけあるかを、しきい値ごとに数える (turn_threshold_deg を決める目安).

  python3 tools/turn_stats.py /data/dataset
  python3 tools/turn_stats.py /data/dataset --horizon 10 --thresholds 10 15 20 30 45

学習の設定 (configs/finetune_*.yaml) の turn_threshold_deg / turn_horizon と同じ判定 (omnivla_real.data_utils.turn_flags).
「曲がる」= この先 horizon フレーム以内に、向きが threshold 度以上変わる、または horizon フレーム先の位置が
threshold 度以上横にある.
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
    for d in trajs:
        data = load_trajectory(d)
        pos, yaw = np.asarray(data["position"]), np.asarray(data["yaw"]).reshape(-1)
        bag = str(load_meta(d).get("bag", "?"))
        n = len(yaw)
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
