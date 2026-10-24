#!/usr/bin/env python3
"""机上評価の結果 (report.json) を並べて表にする (曲がる量の比, 遅れ, ADE, 曲がる所の向き誤差, サブゴール).

  python3 tools/desk_eval_table.py runs/desk_eval/a runs/desk_eval/b ...      # 各ディレクトリの report.json
  python3 tools/desk_eval_table.py --root runs/overnight_x                     # その下の desk_eval_* / 数字_* を全部
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def fmt(v, n=2):
    return "-" if v is None else f"{v:.{n}f}"


def row(name: str, r) -> str:
    if r is None:
        return f"{name:44s} (失敗 / まだ)"
    w, c, g = r.get("waypoints", {}), r.get("commands", {}), r.get("subgoals", {})
    lag = w.get("turn_lag_frames")
    goal = r.get("goal_mode", "-")
    sub = f"{g.get('last_subgoal', '-')}/{g.get('num_nodes', 1) - 1}" if g else "-"
    return (f"{name:44s} {goal:9s} {str(r.get('pose_source', '-')):12s} {fmt(w.get('turn_gain')):>6s} "
            f"{('-' if lag is None else str(-lag)):>4s} {fmt(w.get('ade_m'), 3):>6s} "
            f"{fmt((w.get('turn') or {}).get('heading_err_deg'), 1):>6s} {fmt(c.get('w_corr')):>6s} {sub:>9s} "
            f"{fmt(g.get('index_minus_truth_abs_p90'), 1) if g else '-':>6s}")


def table(dirs) -> str:
    L = ["曲がる量の比: 予測した 8 点目の向き / 実際 (10 度以上曲がる所). 1 に近いほどよい. 遅れ: フレーム (1/3 秒)",
         "ゴール: topomap = 実機と同じくサブゴール画像をたどる / hindsight = 同じ走行の 1.5 m 先 (理想のゴール)",
         "正解: 正解の軌跡に使った位置 (odom = 学習のラベルと同じ / localization = 自己位置)", "",
         f"{'評価':44s} {'ゴール':9s} {'正解':12s} {'比':>6s} {'遅れ':>4s} {'ADE':>6s} {'向き誤差':>6s} {'相関w':>6s} "
         f"{'サブゴール':>9s} {'番号ずれ':>6s}"]
    for d in dirs:
        p = os.path.join(d, "report.json")
        r = json.load(open(p)) if os.path.exists(p) else None
        L.append(row(os.path.basename(os.path.normpath(d)), r))
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--root", default="")
    ap.add_argument("--out", default="", help="表を書くファイル")
    args = ap.parse_args(argv)
    dirs = list(args.dirs)
    if args.root:
        dirs += sorted(d for d in glob.glob(os.path.join(args.root, "*")) if os.path.isdir(d))
    txt = table(dirs)
    print(txt)
    if args.out:
        with open(args.out, "w") as f:
            f.write(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
