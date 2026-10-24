#!/usr/bin/env python3
"""走らせる前の確認 (ROS 2). nav コンテナの中で動かす (bin/field.sh check / where が呼ぶ).

数秒トピックを受け取って、次を調べて OK / 注意 / NG で表示する:
  - カメラ画像 (robot.yaml の topics.image): 届いているか, 周期, 「受け取った時刻 - 画像の時刻」(カメラ・通信の遅れ + 時計のずれ),
    画像の大きさ, 真っ黒でないか
  - オドメトリ (topics.odom): 届いているか, 周期, 動いた距離
  - ナビゲーションノード (/omnivla/status): 状態, 推論の結果 (画像待ちか), 判定方法, 類似度, 推論時間
  - 指示値 (CMD_VEL_TOPIC と /cmd_vel)
  - --where: いまのカメラ画像が topomap のどのサブゴールに一番似ているか (推論サーバで特徴を計算).
    スタート位置に置いたら 0 番の近くになるはず. 類似度が低いなら、topomap を撮ったときとカメラ・景色が違う.
結果は log/field/probe_<時刻>.json にも保存する.

  python3 tools/field_probe.py [--sec 5] [--where] [--topomap /data/topomaps/course_a]
  python3 tools/field_probe.py --status            # /omnivla/status を 1 回 (1 行で)
  python3 tools/field_probe.py --enable true       # 開始 (false で停止). 受け手 (nav) が見つかってから送る
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import threading
import time
from collections import defaultdict

import numpy as np

ROOT = os.environ.get("OMNIVLA_REAL_ROOT", "/workspace")
sys.path.insert(0, ROOT)
from omnivla_real.bag.messages import header_stamp, image_to_rgb  # noqa: E402
from omnivla_real.robot_config import RobotConfig, preprocess_image  # noqa: E402

OK, WARN, NG = "OK  ", "注意", "NG  "


def med(v):
    return statistics.median(v) if v else None


def hz_of(times):
    if len(times) < 2 or times[-1] - times[0] <= 0:
        return None
    return (len(times) - 1) / (times[-1] - times[0])


class Probe:
    def __init__(self):
        self.lock = threading.Lock()
        self.recv = defaultdict(list)       # topic -> 受け取った時刻 (この PC の時計)
        self.offset = defaultdict(list)     # topic -> 受け取った時刻 - header.stamp
        self.last = {}                      # topic -> 最後のメッセージ
        self.first = {}

    def add(self, topic, msg):
        t = time.time()
        s = header_stamp(msg) if hasattr(msg, "header") else None
        with self.lock:
            self.recv[topic].append(t)
            if s:
                self.offset[topic].append(t - s)
            self.first.setdefault(topic, msg)
            self.last[topic] = msg


def where(img, topomap_path, url, step, log):
    """いまの画像に一番似ている topomap のノード (推論サーバの /embed)."""
    from omnivla_real.remote import RemotePolicy
    from omnivla_real.topomap import load_topomap
    tm = load_topomap(topomap_path)
    pol = RemotePolicy(url, timeout=60.0, wait=30.0, log=log)
    cur = pol.embed(img)
    sims = {}

    def sim(j):
        if j not in sims:
            e = pol.embed(tm.nodes[j].image, cache_key=f"n{j}")
            sims[j] = float(np.dot(cur, e) / (np.linalg.norm(cur) * np.linalg.norm(e) + 1e-9))
        return sims[j]
    n = len(tm.nodes)
    t0 = time.time()
    for j in range(0, n, max(1, step)):
        sim(j)
    best = max(sims, key=sims.get)
    for j in range(max(0, best - step), min(n, best + step + 1)):   # 一番似ている所の前後を細かく
        sim(j)
    sim(0)
    top = sorted(sims.items(), key=lambda kv: -kv[1])[:5]
    return {"topomap": topomap_path, "num_nodes": n, "compared": len(sims), "seconds": time.time() - t0,
            "best_node": top[0][0], "best_similarity": top[0][1], "top5": top, "node0_similarity": sims[0],
            "median_similarity": med(list(sims.values()))}


def status_line(d: dict) -> str:
    lat = d.get("latency")
    s = (f"state={d.get('state')} step_state={d.get('step_state', '?')} reach_check={d.get('reach_check')} "
         f"subgoal={d.get('subgoal')}/{(d.get('num_nodes') or 1) - 1} similarity={d.get('similarity')} "
         f"latency={'-' if lat is None else f'{lat * 1000:.0f} ms'} v={d.get('v')} w={d.get('w')}")
    if d.get("last_error"):
        s += f"\n  最後のエラー: {d['last_error']}  (bin/jetson.sh logs nav / logs policy)"
    if d.get("step_state") == "waiting_image":
        s += (f"\n  画像待ち (指示値 0, 類似度 null): ノードが見た画像の遅れ {d.get('image_age')} 秒 "
              f"(image_age_from={d.get('image_age_from')}, 受け取り - 画像の時刻 = {d.get('image_stamp_offset')} 秒). "
              "bin/field.sh check で確認")
    return s


def quick(args) -> int:
    """--status / --enable: 受け手・送り手が見つかるのを待ってから (ros2 topic pub --once は相手を待たずに送ることがある)."""
    import rclpy
    from std_msgs.msg import Bool, String
    rclpy.init()
    node = rclpy.create_node("omnivla_field_cmd")
    got = {}
    node.create_subscription(String, "/omnivla/status", lambda m: got.setdefault("s", m.data), 10)
    rc = 0
    if args.enable:
        pub = node.create_publisher(Bool, "/omnivla/enable", 10)
        t_end = time.time() + 10
        while pub.get_subscription_count() == 0 and time.time() < t_end:
            rclpy.spin_once(node, timeout_sec=0.1)
        if pub.get_subscription_count() == 0:
            print("NG  /omnivla/enable の受け手 (ナビゲーションノード) が見つかりません (bin/field.sh up)")
            rc = 1
        else:
            pub.publish(Bool(data=args.enable == "true"))
            t_end = time.time() + 0.5
            while time.time() < t_end:
                rclpy.spin_once(node, timeout_sec=0.05)
            print(f"/omnivla/enable {args.enable}")
            if args.enable == "true":
                time.sleep(3.0)      # 最初の推論が終わるのを少し待つ (7B は 1 回 0.5〜1 秒)
                got.clear()
    if rc == 0:
        t_end = time.time() + 5
        while "s" not in got and time.time() < t_end:
            rclpy.spin_once(node, timeout_sec=0.1)
        if "s" in got:
            try:
                st = json.loads(got["s"])
                print("  " + status_line(st))
                if args.enable == "true" and st.get("reach_check") == "image":
                    print("注意  reach_check=image: オドメトリが届いていないので、サブゴールを画像だけで判定しています "
                          "(似ているだけで先へ飛びやすい). 届けば自動で image_odom に切り替わります. "
                          "確認: bin/field.sh check")
            except ValueError:
                print("  " + got["s"])
        else:
            print("注意  /omnivla/status が来ません (ナビゲーションノードが動いていない?)")
            rc = rc or (1 if args.status else 0)
    node.destroy_node()
    rclpy.shutdown()
    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sec", type=float, default=5.0, help="受け取る時間 [s]")
    ap.add_argument("--robot", default=os.environ.get("ROBOT_CONFIG", os.path.join(ROOT, "configs/robot.yaml")))
    ap.add_argument("--cmd", default=os.environ.get("CMD_VEL_TOPIC", "") or "/omnivla/cmd_vel/raw")
    ap.add_argument("--where", action="store_true", help="いまの画像が topomap のどこに似ているか")
    ap.add_argument("--topomap", default="", help="--where: 省くと動いているノードの topomap (/omnivla/status). 無ければ環境変数 TOPOMAP")
    ap.add_argument("--step", type=int, default=5, help="--where: 何ノードおきに比べるか (粗く探してから細かく)")
    ap.add_argument("--url", default=f"http://127.0.0.1:{os.environ.get('POLICY_PORT', '8765')}")
    ap.add_argument("--max_image_age", type=float, default=0.5)
    ap.add_argument("--out", default=os.path.join(ROOT, "log/field"))
    ap.add_argument("--status", action="store_true", help="/omnivla/status を 1 回表示するだけ")
    ap.add_argument("--enable", choices=["true", "false"], help="/omnivla/enable を送る (開始 / 停止)")
    args = ap.parse_args()
    if args.status or args.enable:
        sys.exit(quick(args))

    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry, Path
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage
    from sensor_msgs.msg import Image as ImageMsg
    from std_msgs.msg import String

    robot = RobotConfig.load(args.robot)
    tp = robot.topics
    rclpy.init()
    node = rclpy.create_node("omnivla_field_probe")
    pr = Probe()
    img_type = CompressedImage if tp.image.rstrip("/").endswith("compressed") else ImageMsg
    subs = {tp.image: (img_type, qos_profile_sensor_data)}
    if tp.odom:
        subs[tp.odom] = (Odometry, qos_profile_sensor_data)
    subs["/omnivla/status"] = (String, 10)
    subs["/omnivla/path"] = (Path, 10)
    for c in dict.fromkeys([args.cmd, "/cmd_vel"]):
        subs[c] = (Twist, 10)
    for topic, (typ, qos) in subs.items():
        node.create_subscription(typ, topic, lambda m, t=topic: pr.add(t, m), qos)

    print(f"[probe] {args.sec:.0f} 秒トピックを受け取ります (ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '0')}) ...",
          flush=True)
    t_end = time.time() + 2.5 + args.sec        # 最初の 2.5 秒は相手を見つける時間
    while time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.1)
    names = dict(node.get_topic_names_and_types())
    pubs = {t: node.count_publishers(t) for t in subs}

    rep = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "host_time": time.time(), "topics": {}, "checks": []}

    def check(level, what, detail=""):
        rep["checks"].append({"level": level.strip(), "what": what, "detail": detail})

    # --- カメラ ---
    img = None
    with pr.lock:
        recv, off, last = dict(pr.recv), dict(pr.offset), dict(pr.last)
    ti = {"type": names.get(tp.image, ["(無い)"])[0] if tp.image in names else "(無い)", "publishers": pubs[tp.image],
          "count": len(recv.get(tp.image, [])), "hz": hz_of(recv.get(tp.image, [])),
          "received_minus_stamp_s": med(off.get(tp.image, []))}
    if tp.image in last:
        m = last[tp.image]
        try:
            rgb = image_to_rgb(m)
            img = preprocess_image(rgb, robot.image)
            ti.update(size=f"{rgb.shape[1]}x{rgb.shape[0]}", brightness=float(rgb.mean()),
                      format=getattr(m, "format", None) or getattr(m, "encoding", None))
        except Exception as e:  # noqa: BLE001
            ti["decode_error"] = f"{type(e).__name__}: {e}"
    rep["topics"]["image"] = {tp.image: ti}
    if ti["count"] == 0:
        cand = [t for t in names if any(k in t for k in ("image", "camera", "compressed"))]
        check(NG, f"カメラ画像 {tp.image} が届いていない",
              f"送り手 {ti['publishers']} 個. 似た名前のトピック: {cand or 'なし'}. "
              "robot.yaml の topics.image / ROS_DOMAIN_ID / DDS の設定 (DDS_PROFILE_FILE) を確認")
    else:
        hz = ti["hz"] or 0.0
        check(OK if hz >= 3 else WARN, f"カメラ画像 {hz:.1f} Hz ({ti.get('size', '?')}, {ti.get('format', '?')})",
              "" if hz >= 3 else "周期が低い (推論は画像が来るたびに行う)")
        if "decode_error" in ti:
            check(NG, "カメラ画像を読めない", ti["decode_error"])
        elif ti.get("brightness", 255) < 8:
            check(WARN, "カメラ画像がほぼ真っ黒", f"平均の明るさ {ti['brightness']:.1f}")
        o = ti["received_minus_stamp_s"]
        if o is None:
            check(WARN, "カメラ画像に時刻 (header.stamp) が無い", "受け取った時刻で判定される")
        elif abs(o) > 1e5:
            check(NG, f"画像の時刻がこの PC の時計と全く違う ({o:.0f} 秒)",
                  "カメラが起動してからの時間などを付けている. image_age_from=receive で動かす (bin/field.sh up の既定)")
        elif o < -0.05:
            check(NG, f"画像の時刻が未来 ({-o:.2f} 秒先)", "ATMOBI PC の時計が進んでいる. bin/field.sh time で合わせる")
        elif o > args.max_image_age:
            check(NG, f"画像が {o:.2f} 秒遅れて届く (max_image_age {args.max_image_age})",
                  "時計のずれ (bin/field.sh time) か、カメラ・通信の遅れ. image_age_from=receive なら止まらない")
        elif o > 0.25:
            check(WARN, f"画像が {o:.2f} 秒遅れて届く", "カメラ・通信の遅れか時計のずれ")
        else:
            check(OK, f"画像の遅れ {o:.2f} 秒")
    # --- オドメトリ ---
    if tp.odom:
        r = recv.get(tp.odom, [])
        to = {"publishers": pubs[tp.odom], "count": len(r), "hz": hz_of(r), "received_minus_stamp_s": med(off.get(tp.odom, []))}
        if tp.odom in last and tp.odom in pr.first:
            a, b = pr.first[tp.odom].pose.pose.position, last[tp.odom].pose.pose.position
            to["moved_m"] = math.hypot(b.x - a.x, b.y - a.y)
        rep["topics"]["odom"] = {tp.odom: to}
        if not r:
            check(WARN, f"オドメトリ {tp.odom} が届いていない",
                  "このまま start するとサブゴールは画像だけで判定 (image) になり、似ているだけで先へ飛びやすい. "
                  "start のあとで届けば自動で道のりも使う (image_odom) が、届いてから start するのが安全")
        else:
            check(OK, f"オドメトリ {to['hz'] or 0:.1f} Hz")
    # --- ナビゲーションノード ---
    st = None
    if "/omnivla/status" in last:
        try:
            st = json.loads(last["/omnivla/status"].data)
        except ValueError:
            st = None
    rep["status"] = st
    if st is None:
        check(WARN, "ナビゲーションノードの状態 (/omnivla/status) が来ない", "nav が動いていない (bin/field.sh up)")
    else:
        desc = status_line(st).split("\n")[0]
        if st.get("num_nodes", 0) == 0:
            check(NG, "topomap が読み込まれていない", "TOPOMAP を指定する (bin/field.sh up --topomap <名前>)")
        if st.get("step_state") == "error":
            check(NG, "推論・判定でエラーが起きている (指示値 0)", f"{st.get('last_error')}. bin/jetson.sh logs nav / logs policy")
        elif st.get("step_state") == "waiting_image":
            check(NG, "ナビゲーションが画像待ち (指示値 0, 類似度 null)",
                  f"{desc}. ノードが見た画像の遅れ {st.get('image_age')} 秒 (image_age_from={st.get('image_age_from')})")
        else:
            check(OK, "ナビゲーションノード", desc)
    for c in dict.fromkeys([args.cmd, "/cmd_vel"]):
        r = recv.get(c, [])
        v = None
        if c in last:
            v = (round(last[c].linear.x, 3), round(last[c].angular.z, 3))
        rep["topics"].setdefault("cmd", {})[c] = {"publishers": pubs[c], "count": len(r), "hz": hz_of(r), "last": v}
    rep["topics"]["all"] = {k: v for k, v in sorted(names.items())}

    # --- topomap のどこに似ているか ---
    if args.where:
        args.topomap = args.topomap or (st or {}).get("topomap") or os.environ.get("TOPOMAP", "")
        if img is None:
            check(NG, "topomap との比較ができない", "カメラ画像が無い")
        elif not args.topomap:
            check(NG, "topomap との比較ができない", "TOPOMAP が空 (bin/field.sh where <topomap>)")
        else:
            try:
                w = where(img, args.topomap, args.url, args.step, lambda s: print(s, flush=True))
                rep["where"] = w
                b, s0 = w["best_similarity"], w["node0_similarity"]
                lvl = OK if b >= 0.9 else (WARN if b >= 0.85 else NG)
                check(lvl, f"いまの画像に一番似ているのは {w['best_node']} 番 / {w['num_nodes'] - 1} (類似度 {b:.3f})",
                      f"0 番 (スタート) との類似度 {s0:.3f}, 比べた全体の中央値 {w['median_similarity']:.3f}. "
                      "スタート位置なら 0 番の近く. 類似度が 0.9 未満なら topomap を撮った時とカメラ・景色が違う可能性")
            except Exception as e:  # noqa: BLE001
                check(NG, "topomap との比較に失敗", f"{type(e).__name__}: {e} (推論サーバは動いている? bin/field.sh up)")

    # --- 表示と保存 ---
    print("")
    for c in rep["checks"]:
        print(f"[{c['level']:<2}] {c['what']}" + (f"\n        {c['detail']}" if c["detail"] else ""))
    cmds = rep["topics"].get("cmd", {})
    print("        指示値: " + ", ".join(f"{k} {v['count']}件 最後 {v['last']}" for k, v in cmds.items()))
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"probe_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1, default=str)
    print(f"-> {path}")
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(1 if any(c["level"] == "NG" for c in rep["checks"]) else 0)


if __name__ == "__main__":
    main()
