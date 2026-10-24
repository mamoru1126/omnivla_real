#!/usr/bin/env python3
"""実走の評価: 自律走行した bag の自己位置を、コース (自己位置で作った topomap) と比べる.

自己位置は評価だけに使う (学習・走行には使わない). robot.yaml の topics.eval_localization (PoseStamped など)
か --loc_topic で指定する. コースの topomap は、同じ自己位置が入った bag から作る (frame: map になる).

  python3 tools/run_eval.py --bag /bags/auto3 --topomap /data/topomaps/course_b --out /runs/run_eval/auto3
  python3 tools/run_eval.py --bag /bags/auto3 --topomap /data/topomaps/course_b --loc_topic /localization --out ...

出力 (<out>/):
  report.txt / report.json   どこまで進んだか, コースからの横ずれ, 向きの差, カーブごとの膨らみ
  overview.png               コース (灰色) と走った所 (横ずれの大きさで色分け), カーブの番号
  xtrack.png                 コースの道のりに対する横ずれ (カーブを色付け)
  track.csv                  時刻ごとの値
/omnivla/status が bag にあれば、ナビが running だった間 (自律で走っていた間) だけの集計も出す.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from omnivla_real.extract import read_streams  # noqa: E402
from omnivla_real.robot_config import RobotConfig  # noqa: E402
from omnivla_real.run_eval import CoursePath, evaluate_run, report_text, running_mask, save_json  # noqa: E402
from omnivla_real.runio import add_common_args, open_run, run_name  # noqa: E402
from omnivla_real.topomap import load_topomap  # noqa: E402

STATUS_TOPIC = "/omnivla/status"


def read_status(bag) -> list:
    if STATUS_TOPIC not in bag.topics():
        return []
    out = []
    for m in bag.messages([STATUS_TOPIC]):
        try:
            out.append((m.t_bag, json.loads(m.msg.data).get("state", "")))
        except (ValueError, AttributeError):
            continue
    return out


def plot(path: CoursePath, per: dict, summary: dict, out_dir: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = np.isfinite(per["s"])
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.plot(path.x, path.y, color="0.7", lw=6, solid_capstyle="round", label="course")
    sc = ax.scatter(per["x"][ok], per["y"][ok], c=np.abs(per["xtrack"][ok]), cmap="viridis_r", s=6,
                    vmin=0, vmax=max(1.0, float(np.nanpercentile(np.abs(per["xtrack"][ok]), 99))), label="run")
    if "running" in per:
        man = ok & ~per["running"]
        ax.scatter(per["x"][man], per["y"][man], color="red", s=3, label="not running (manual?)")
    for k, c in enumerate(summary.get("corners") or []):
        sm = 0.5 * (c["s0"] + c["s1"])
        j = int(np.clip(np.searchsorted(path.s, sm) - 1, 0, len(path.x) - 1))
        ax.annotate(str(k + 1), (path.x[j], path.y[j]), fontsize=10, color="C3", weight="bold")
    ax.plot(path.x[0], path.y[0], "k^", ms=10, label="course start")
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=8)
    fig.colorbar(sc, ax=ax, label="|cross-track| [m]")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "overview.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    for c in summary.get("corners") or []:
        ax.axvspan(c["s0"], c["s1"], color="C1", alpha=0.15)
    ax.plot(per["s"][ok], per["xtrack"][ok], ".", ms=2, color="C0")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("course distance s [m]")
    ax.set_ylabel("cross-track [m] (left +)")
    ax.set_title(f"{title} (orange: curves)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "xtrack.png"), dpi=120)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--bag", required=True, help="自律走行した bag (自己位置が入っていること)")
    ap.add_argument("--topomap", required=True, help="コース (自己位置で作った topomap, frame: map)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--loc_topic", default="", help="自己位置のトピック (robot.yaml の topics.eval_localization を上書き)")
    ap.add_argument("--speed_cap", type=float, default=0.3, help="走行の速度の上限 [m/s] (これより速い所は手で動かした可能性)")
    ap.add_argument("--min_turn_deg", type=float, default=20.0, help="4 m の間にこれ以上向きが変わる所をカーブとする")
    ap.add_argument("--rate", type=float, default=5.0, help="評価する周期 [Hz]")
    ap.add_argument("--start_sec", type=float)
    ap.add_argument("--end_sec", type=float)
    args = ap.parse_args(argv)

    robot = copy.deepcopy(RobotConfig.load(args.robot))
    if args.loc_topic:
        robot.topics.eval_localization = args.loc_topic
        robot.topics.localization = ""
    if not (robot.topics.localization or robot.topics.eval_localization):
        ap.error("自己位置のトピックがありません: robot.yaml の topics.eval_localization か --loc_topic")
    robot.topics.image = ""          # 画像は使わない (速い)
    robot.topics.cmd = ""
    tm = load_topomap(args.topomap)
    path = CoursePath.from_topomap(tm)
    os.makedirs(args.out, exist_ok=True)
    name = run_name(args.bag)
    with open_run(args.bag, args.typestore) as bag:
        streams = read_streams(bag, robot, None, args.rate, args.start_sec, args.end_sec, name, decode_images=False)
        status = read_status(bag)
    for n in streams.notes:
        print("  note:", n)
    if len(streams.loc_t) < 2:
        print(f"自己位置 ({robot.topics.localization or robot.topics.eval_localization}) が bag にありません")
        return 1
    tr = streams.loc_track()
    t = np.arange(streams.loc_t[0], streams.loc_t[-1], 1.0 / args.rate)
    x, y, yaw, valid = tr.at(t, 1.0)
    x, y, yaw = (np.where(valid, a, np.nan) for a in (x, y, yaw))
    running = running_mask(t, status)
    summary, per = evaluate_run(path, t, x, y, yaw, running, args.speed_cap, args.min_turn_deg)
    summary.update({"bag": args.bag, "topomap": args.topomap, "loc_topic": robot.topics.localization
                    or robot.topics.eval_localization, "status_messages": len(status)})
    title = f"{name} on {os.path.basename(os.path.normpath(args.topomap))}"
    text = report_text(summary, title)
    print(text)
    with open(os.path.join(args.out, "report.txt"), "w") as f:
        f.write(text)
    save_json(os.path.join(args.out, "report.json"), summary)
    with open(os.path.join(args.out, "track.csv"), "w", newline="") as f:
        w = csv.writer(f)
        keys = [k for k in ("t", "x", "y", "yaw", "speed", "s", "xtrack", "heading_err", "running") if k in per]
        w.writerow(keys)
        for i in range(len(t)):
            w.writerow([f"{float(per[k][i]):.4f}" if k != "running" else int(per[k][i]) for k in keys])
    plot(path, per, summary, args.out, title)
    print(f"-> {args.out} (report.txt, overview.png, xtrack.png)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
