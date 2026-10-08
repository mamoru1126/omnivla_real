#!/usr/bin/env python3
"""机上評価: 記録した bag (コースを走った 1 回) をモデルに通し、実際の走行と比べる.

  # 実機と同じく、サブゴール画像列 (別の走行から作った topomap) をたどる
  python3 tools/desk_eval.py --bag /data/bags/run2 --topomap /data/topomaps/course_a \
      --finetuned_dir /runs/<run>/checkpoints/step_010000 --out /runs/desk_eval/run2

  # モデルを比べる: --model 7b --weights /checkpoints/omnivla-original [--finetuned_dir ...]
  # 配管の確認 (正解をそのまま出すダミーのモデル): --policy oracle
  # サブゴールの切り替えを使わず、同じ bag の 1.5m 先をゴールにする: --goal_mode hindsight

出力 (<out>/):
  report.txt / report.json   指標のまとめ (下の「読み方」)
  overview.png               地図: 実際の走行, モデルの指示値を積算した軌跡, 記録の指示値を積算した軌跡, サブゴール
  timeline.png               時系列: 速度・角速度 (モデル vs 記録), サブゴール番号, 類似度
  steps.csv                  フレームごとの値
  debug/                     --debug_every N でデバッグ画像

読み方:
  waypoints.ade_m            予測 8 点と、実際にその後走った 8 点のずれ (平均) [m]
  commands.turn_direction_agreement  記録が曲がっている所で、モデルの角速度の向きが合っている割合
  integrated_windows         実際の位置から N 秒だけモデルの指示値で進めた時の終点のずれ.
                             recorded_cmd (記録された指示値を同じように積算) と同程度なら十分に近い
  subgoals.reached_goal      最後のサブゴールまで切り替わったか
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from omnivla_real.desk_eval import (DeskEvalConfig, OraclePolicy, prepare_frames, run_desk_eval,  # noqa: E402
                                    similarity_table)
from omnivla_real.extract import read_streams  # noqa: E402
from omnivla_real.geometry import relative_pose  # noqa: E402
from omnivla_real.nav_config import load_nav_config, make_policy  # noqa: E402
from omnivla_real.robot_config import RobotConfig  # noqa: E402
from omnivla_real.runio import add_common_args, open_run, run_name  # noqa: E402
from omnivla_real.topomap import compose, load_topomap  # noqa: E402


def plots(out: str, ft, rec, summary, nodes_xy):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(9, 9), dpi=100)
    ax.plot(ft.x, ft.y, color="#222", lw=2.5, label="actual (ground truth)")
    pm = np.asarray(summary["paths"]["model"])
    pc = np.asarray(summary["paths"]["recorded_cmd"])
    ax.plot(pm[:, 0], pm[:, 1], color="#d0602f", lw=1.5, label="model commands, integrated")
    ax.plot(pc[:, 0], pc[:, 1], color="#2f6fdb", lw=1, ls="--", label="recorded commands, integrated")
    every = max(1, len(rec) // 60)
    for r in rec[::every]:
        if "wps" in r:
            w = np.asarray(r["wps"])
            c, s = math.cos(r["yaw"]), math.sin(r["yaw"])
            xs = r["x"] + c * w[:, 0] - s * w[:, 1]
            ys = r["y"] + s * w[:, 0] + c * w[:, 1]
            ax.plot(np.r_[r["x"], xs], np.r_[r["y"], ys], color="#d0602f", lw=0.6, alpha=0.5)
    if nodes_xy is not None and len(nodes_xy):
        ax.plot(nodes_xy[:, 0], nodes_xy[:, 1], "s", color="#b7791f", ms=4, label="subgoals")
    ax.plot(ft.x[0], ft.y[0], "o", color="green", ms=9)
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.set_title("thin orange: predicted 8 waypoints (every few frames)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "overview.png"))
    plt.close(fig)

    t = np.array([r["t"] for r in rec]) - rec[0]["t"]
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), dpi=100, sharex=True)
    axs[0].plot(t, [r["cmd_v"] for r in rec], label="recorded v", color="#2f6fdb")
    axs[0].plot(t, [r.get("v", np.nan) for r in rec], label="model v", color="#d0602f")
    axs[0].set_ylabel("m/s")
    axs[0].legend(fontsize=8)
    axs[1].plot(t, [r["cmd_w"] for r in rec], label="recorded w", color="#2f6fdb")
    axs[1].plot(t, [r.get("w", np.nan) for r in rec], label="model w", color="#d0602f")
    axs[1].axhline(0, color="#999", lw=0.8)
    axs[1].set_ylabel("rad/s")
    axs[1].legend(fontsize=8)
    if any("subgoal" in r for r in rec):
        axs[2].step(t, [r.get("subgoal", np.nan) for r in rec], where="post", label="subgoal (model)", color="#7a4fd0")
        if any(r.get("subgoal_gt") is not None for r in rec):
            axs[2].step(t, [r.get("subgoal_gt") if r.get("subgoal_gt") is not None else np.nan for r in rec],
                        where="post", label="subgoal (by position)", color="#999", ls="--")
        ax2 = axs[2].twinx()
        ax2.plot(t, [r.get("similarity") if r.get("similarity") is not None else np.nan for r in rec],
                 color="#2e8b57", lw=0.8, label="similarity")
        ax2.set_ylabel("similarity")
        axs[2].legend(fontsize=8, loc="upper left")
    else:
        axs[2].plot(t, [r.get("ade", np.nan) for r in rec], label="ADE [m]")
        axs[2].legend(fontsize=8)
    axs[2].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "timeline.png"))
    plt.close(fig)


def text_report(s: dict) -> str:
    L = []
    w, c, f = s.get("waypoints", {}), s.get("commands", {}), s.get("integrated_full", {})
    fmt = lambda v, u="": "-" if v is None else (f"{v:.3f}{u}" if isinstance(v, float) else f"{v}{u}")  # noqa: E731
    L.append(f"frames: {s.get('frames_evaluated')} / {s.get('frames')}  (sample_rate {s.get('sample_rate')} Hz, "
             f"controller {s.get('controller')})")
    L.append("")
    L.append("[予測軌跡 vs 実際にその後走った軌跡]")
    L.append(f"  ADE {fmt(w.get('ade_m'), ' m')}  FDE {fmt(w.get('fde_m'), ' m')}  向き {fmt(w.get('heading_err_deg'), ' deg')}")
    t = w.get("turn", {})
    L.append(f"  曲がる場面 (n={t.get('n')}): ADE {fmt(t.get('ade_m'), ' m')}  向き {fmt(t.get('heading_err_deg'), ' deg')}")
    L.append("")
    L.append("[指示値 vs 記録された指示値]")
    L.append(f"  |v| 誤差 {fmt(c.get('v_mae'), ' m/s')}  |w| 誤差 {fmt(c.get('w_mae'), ' rad/s')}  "
             f"相関(w) {fmt(c.get('w_corr'))}")
    L.append(f"  曲がる向きの一致率 {fmt(c.get('turn_direction_agreement'))} (曲がっているフレーム {c.get('turn_frames')})")
    L.append("")
    L.append("[指示値を積算した軌跡 vs 実際の走行]")
    L.append(f"  全体 (スタートから積算, {fmt(f.get('path_length_m'), ' m')}): モデル 平均 {fmt(f.get('model_mean_err_m'), ' m')} / "
             f"終点 {fmt(f.get('model_final_err_m'), ' m')}   (記録の指示値: 平均 {fmt(f.get('recorded_cmd_mean_err_m'), ' m')})")
    for k, v in s.get("integrated_windows", {}).items():
        m, r = v["model"], v["recorded_cmd"]
        L.append(f"  {k} 窓: モデル 中央値 {fmt(m.get('pos_err_median_m'), ' m')} (p90 {fmt(m.get('pos_err_p90_m'), ' m')}, "
                 f"向き {fmt(m.get('yaw_err_median_deg'), ' deg')})   記録の指示値 {fmt(r.get('pos_err_median_m'), ' m')}")
    if "subgoals" in s:
        g = s["subgoals"]
        L.append("")
        L.append(f"[サブゴール ({g['reach_check']})]")
        L.append(f"  最後まで到達: {g['reached_goal']}  最後のサブゴール {g['last_subgoal']}/{g['num_nodes'] - 1}")
        L.append(f"  位置から見た正解との差 (中央値) {fmt(g.get('index_minus_truth_median'))}  "
                 f"|差| p90 {fmt(g.get('index_minus_truth_abs_p90'))}  類似度 中央値 {fmt(g.get('similarity_median'))}")
    if s.get("similarity_threshold"):
        st = s["similarity_threshold"]
        L.append(f"  類似度の目安: 近いノード 中央値 {st['near_median']:.3f} / 遠いノード p99 {st['far_p99']:.3f} "
                 f"-> image_threshold ~ {st['suggested_image_threshold']:.3f}")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--bag", required=True, help="評価する走行の bag (topomap を作った走行とは別の走行を推奨)")
    ap.add_argument("--topomap", default="", help="サブゴール画像列 (tools/make_topomap.py の出力)")
    ap.add_argument("--nav", default="configs/navigator.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", default="model", choices=["model", "oracle"])
    ap.add_argument("--model", default="", help="edge | 7b | remote (navigator.yaml を上書き)")
    ap.add_argument("--url", default="", help="--model remote: 推論サーバ (tools/policy_server.py) の URL")
    ap.add_argument("--weights", default="")
    ap.add_argument("--finetuned_dir", default="")
    ap.add_argument("--device", default="")
    ap.add_argument("--goal_mode", default="", choices=["", "topomap", "hindsight"])
    ap.add_argument("--goal_ahead_m", type=float, default=1.5)
    ap.add_argument("--pose_source", default="auto", choices=["auto", "odom", "odom_twist", "cmd", "localization"],
                    help="正解の位置 (auto: 自己位置があればそれ, 無ければ odom)")
    ap.add_argument("--start_sec", type=float)
    ap.add_argument("--end_sec", type=float)
    ap.add_argument("--sample_rate", type=float, default=0.0, help="0 = モデルの学習データの周期")
    ap.add_argument("--debug_every", type=int, default=0)
    args = ap.parse_args(argv)

    robot = RobotConfig.load(args.robot)
    nav = load_nav_config(args.nav if os.path.exists(args.nav) else None,
                          {"model.model": args.model, "model.weights": args.weights,
                           "model.finetuned_dir": args.finetuned_dir, "model.device": args.device,
                           "model.url": args.url})
    goal_mode = args.goal_mode or ("topomap" if args.topomap else "hindsight")
    os.makedirs(args.out, exist_ok=True)
    policy = None
    if args.policy == "model":
        policy = make_policy(nav.model)
    meta = getattr(policy, "meta", None) or getattr(getattr(policy, "c", None), "meta", None) or {}
    rate = args.sample_rate or nav.engine.sample_rate or float(meta.get("sample_rate") or 3.0)
    nav.engine.sample_rate = rate
    tmp = tempfile.mkdtemp(prefix="desk_eval_")
    try:
        print(f"reading {args.bag} at {rate} Hz ...")
        with open_run(args.bag, args.typestore) as bag:
            streams = read_streams(bag, robot, tmp, rate, args.start_sec, args.end_sec, run_name(args.bag))
        ft, odom_xyz, src, offset = prepare_frames(streams, rate, args.pose_source)
        if args.policy == "oracle":
            policy = OraclePolicy(ft, rate)
        topomap = load_topomap(args.topomap) if goal_mode == "topomap" else None
        dcfg = DeskEvalConfig(goal_mode=goal_mode, goal_ahead_m=args.goal_ahead_m, debug_every=args.debug_every)
        dbg_dir = os.path.join(args.out, "debug")
        if args.debug_every:
            os.makedirs(dbg_dir, exist_ok=True)
        rec, summary = run_desk_eval(ft, policy, robot, nav.engine, dcfg, topomap, odom_xyz,
                                     progress=lambda s: print("  ", s, flush=True),
                                     debug_cb=lambda i, img: img.save(os.path.join(dbg_dir, f"{i:06d}.jpg"), quality=85))
        nodes_xy = None
        if topomap is not None and topomap.has_poses():
            if topomap.frame == "map" and src == "localization":
                nodes_xy = np.array([n.pose[:2] for n in topomap.nodes])
            elif topomap.start is not None:
                origin = (ft.x[0], ft.y[0], ft.yaw[0])
                nodes_xy = np.array([compose(origin, relative_pose(topomap.start, n.pose))[:2] for n in topomap.nodes])
            course = [r.get("course_pose") for r in rec]
            if args.policy == "model" and all(c is not None for c in course):
                summary["similarity_threshold"] = similarity_table(policy, ft.image_paths, topomap.nodes,
                                                                   np.asarray(course))
        summary.update({"bag": args.bag, "topomap": args.topomap, "goal_mode": goal_mode, "pose_source": src,
                        "policy": args.policy, "model": vars(nav.model), "trim_offset_frames": offset})
        plots(args.out, ft, rec, summary, nodes_xy)
        with open(os.path.join(args.out, "report.json"), "w") as f:
            json.dump(summary, f, indent=1, default=str)
        keys = ["i", "t", "x", "y", "yaw", "cmd_v", "cmd_w", "v", "w", "state", "subgoal", "subgoal_gt",
                "similarity", "ade", "fde", "yaw_err_deg", "gt_turn_deg", "pred_turn_deg", "latency"]
        with open(os.path.join(args.out, "steps.csv"), "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(keys)
            for r in rec:
                wr.writerow([r.get(k, "") for k in keys])
        txt = text_report(summary)
        with open(os.path.join(args.out, "report.txt"), "w") as f:
            f.write(txt)
        print("\n" + txt + f"-> {args.out}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
