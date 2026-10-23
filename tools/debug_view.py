#!/usr/bin/env python3
"""走行中のナビゲーションノードを外から見るデバッグ画面 (ROS 2. Python のノードにはブラウザの画面が無いので).

nav コンテナの中で動かす (同じ ROS_DOMAIN_ID / DDS の設定でトピックを受け取る):
  docker compose -f docker-compose.jetson.yml -f docker-compose.jetson-humble.yml exec nav \\
    bash -c "source /opt/omnivla_ws/install/setup.bash && python3 /workspace/tools/debug_view.py"
  → ブラウザで http://<Orin の IP>:8090   (ATMOBI PC からなら http://192.168.0.200:8090)

表示 (1 秒ごと):
  左上: モデルに入れる画像 (robot.yaml の前処理後) + 予測した軌跡 (/omnivla/path)
  右上: いま目指しているサブゴール画像 (topomap の <番号>.jpg)
  下  : 状態, サブゴール番号, 類似度, 推論時間, 指示値 (v, w), 画像の遅れ・周期 など
保存: log/debug_view/<時刻>/  (frames/*.jpg を 1 秒ごと, steps.csv)
Ctrl-C で終わると、まとめ (summary.txt) を表示する. それを貼れば状況が分かる.
"""
from __future__ import annotations

import argparse
import csv
import http.server
import io
import json
import math
import os
import statistics
import sys
import threading
import time
from collections import deque
from typing import Optional

import numpy as np
from PIL import Image

ROOT = os.environ.get("OMNIVLA_REAL_ROOT", "/workspace")
sys.path.insert(0, ROOT)
from omnivla_real.robot_config import RobotConfig, camera_model, preprocess_image  # noqa: E402
from omnivla_real.viz import render_debug  # noqa: E402

PAGE = """<!doctype html><meta charset="utf-8"><title>OmniVLA debug</title>
<style>body{background:#1e1e1e;color:#ddd;font:14px monospace;margin:12px}img{max-width:100%%;border:1px solid #555}
pre{background:#111;padding:8px;white-space:pre-wrap}</style>
<h3>OmniVLA debug view</h3><img id=i src="frame.jpg"><pre id=s></pre>
<script>
setInterval(()=>{document.getElementById('i').src='frame.jpg?'+Date.now();
fetch('state.json?'+Date.now()).then(r=>r.json()).then(j=>{document.getElementById('s').textContent=JSON.stringify(j,null,1)})},%d);
</script>"""


class State:
    """受け取った最新の値 (コールバックと画面の間で共有)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.img_msg = None
        self.img_stamp: Optional[float] = None
        self.img_recv = deque(maxlen=60)       # 受信時刻 (周期の計算)
        self.n_img = 0
        self.path: Optional[np.ndarray] = None  # (8, 4) x, y, cos, sin
        self.path_time: Optional[float] = None
        self.n_path = 0
        self.status: dict = {}
        self.status_time: Optional[float] = None
        self.cmd = None                          # (v, w)
        self.cmd_time: Optional[float] = None
        self.n_cmd = 0
        self.odom_recv = deque(maxlen=60)


def hz(times) -> Optional[float]:
    t = list(times)
    if len(t) < 2 or t[-1] - t[0] <= 0:
        return None
    return (len(t) - 1) / (t[-1] - t[0])


def fmt(v, f="{:.2f}"):
    return "-" if v is None else f.format(v)


def compose(st: State, robot: RobotConfig, cam, now: float):
    """画面 1 枚と、その時点の値 (dict)."""
    from omnivla_real.bag.messages import image_to_rgb
    with st.lock:
        msg, stamp, path = st.img_msg, st.img_stamp, st.path
        status, cmd = dict(st.status), st.cmd
        info = {
            "image_hz": hz(st.img_recv), "odom_hz": hz(st.odom_recv),
            "image_age_s": None if stamp is None else now - stamp,          # 時計のずれ + 遅れ
            "image_since_recv_s": None if not st.img_recv else now - st.img_recv[-1],
            "path_age_s": None if st.path_time is None else now - st.path_time,
            "cmd_age_s": None if st.cmd_time is None else now - st.cmd_time,
            "n_image": st.n_img, "n_path": st.n_path, "n_cmd": st.n_cmd,
        }
    cur = None
    if msg is not None:
        try:
            cur = preprocess_image(image_to_rgb(msg), robot.image)
        except Exception as e:  # noqa: BLE001
            info["image_error"] = f"{type(e).__name__}: {e}"
    goal = None
    tm, sg = status.get("topomap"), status.get("subgoal")
    if tm and sg is not None:
        p = os.path.join(tm, f"{int(sg)}.jpg")
        if os.path.exists(p):
            goal = Image.open(p).convert("RGB")
    lat = status.get("latency")
    lines = [
        f"state {status.get('state', '-')}/{status.get('step_state') or '-'}   ({status.get('reach_check', '-')})"
        + (f"  node img age {status['image_age']:.2f}/{status.get('max_image_age')}s" if status.get("image_age") is not None else ""),
        f"subgoal {status.get('subgoal', '-')}/{(status.get('num_nodes') or 1) - 1}   sim {fmt(status.get('similarity'), '{:.3f}')}",
        f"cmd v={fmt(cmd[0] if cmd else None)} w={fmt(cmd[1] if cmd else None)}   latency {fmt(lat * 1000 if lat else None, '{:.0f}')} ms",
        f"image {fmt(info['image_hz'], '{:.1f}')} Hz  age {fmt(info['image_age_s'])} s   odom {fmt(info['odom_hz'], '{:.1f}')} Hz",
    ]
    frame = render_debug(cur if cur is not None else Image.new("RGB", (320, 240), (60, 0, 0)),
                         goal, path, cam=cam, lines=lines)
    row = {"time": now, "state": status.get("state"), "step_state": status.get("step_state"),
           "node_image_age_s": status.get("image_age"), "subgoal": sg, "num_nodes": status.get("num_nodes"),
           "similarity": status.get("similarity"), "latency": lat, "travel_m": status.get("travel_m"),
           "v": cmd[0] if cmd else None, "w": cmd[1] if cmd else None, **info,
           "path_end_x": float(path[-1, 0]) if path is not None else None,
           "path_end_y": float(path[-1, 1]) if path is not None else None}
    return frame, row


def summarize(rows: list, topics: dict) -> str:
    def vals(k):
        return [r[k] for r in rows if r.get(k) is not None]

    def stat(k, scale=1.0, f="{:.2f}"):
        v = [x * scale for x in vals(k)]
        if not v:
            return "-"
        return f"中央値 {f.format(statistics.median(v))} / 最小 {f.format(min(v))} / 最大 {f.format(max(v))}"
    L = [f"debug_view まとめ ({len(rows)} 秒分)", f"トピック: {json.dumps(topics, ensure_ascii=False)}"]
    states = {}
    for r in rows:
        states[r.get("state")] = states.get(r.get("state"), 0) + 1
    L.append(f"状態: {states}")
    steps = {}
    for r in rows:
        steps[r.get("step_state")] = steps.get(r.get("step_state"), 0) + 1
    L.append(f"推論の結果 (step_state): {steps}   ノードが見た画像の遅れ [s]: {stat('node_image_age_s')}")
    sg = vals("subgoal")
    L.append(f"サブゴール: {sg[0] if sg else '-'} -> {sg[-1] if sg else '-'} / {rows[-1].get('num_nodes') if rows else '-'}")
    L.append(f"類似度: {stat('similarity', f='{:.3f}')}")
    L.append(f"推論時間 [ms]: {stat('latency', 1000, '{:.0f}')}")
    L.append(f"v [m/s]: {stat('v')}   w [rad/s]: {stat('w')}")
    w = vals("w")
    if w:
        L.append(f"w の向き: 左 {sum(x > 0.05 for x in w)} / 右 {sum(x < -0.05 for x in w)} / ほぼ直進 {sum(abs(x) <= 0.05 for x in w)} (秒)")
    L.append(f"予測軌跡の終点 y (左 +) [m]: {stat('path_end_y')}   x [m]: {stat('path_end_x')}")
    L.append(f"画像: {stat('image_hz', f='{:.1f}')} Hz   遅れ (今 - 画像の時刻) [s]: {stat('image_age_s')}")
    L.append(f"オドメトリ: {stat('odom_hz', f='{:.1f}')} Hz")
    errs = {r['image_error'] for r in rows if r.get('image_error')}
    if errs:
        L.append(f"画像のエラー: {errs}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8090, help="ブラウザのポート (0 で画面なし)")
    ap.add_argument("--robot", default=os.environ.get("ROBOT_CONFIG", os.path.join(ROOT, "configs/robot.yaml")))
    ap.add_argument("--cmd", default=os.environ.get("CMD_VEL_TOPIC", "") or "/cmd_vel", help="指示値のトピック")
    ap.add_argument("--out", default=os.path.join(ROOT, "log/debug_view", time.strftime("%Y%m%d_%H%M%S")))
    ap.add_argument("--every", type=float, default=1.0, help="画面の更新と保存の間隔 [s]")
    ap.add_argument("--no_save", action="store_true", help="画像を保存しない (steps.csv と summary.txt は保存)")
    args = ap.parse_args()

    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry, Path
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage
    from sensor_msgs.msg import Image as ImageMsg
    from std_msgs.msg import String

    robot = RobotConfig.load(args.robot)
    cam = camera_model(robot)
    tp = robot.topics
    topics = {"image": tp.image, "odom": tp.odom, "cmd": args.cmd, "path": "/omnivla/path", "status": "/omnivla/status"}
    st = State()
    os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)

    rclpy.init()
    node = rclpy.create_node("omnivla_debug_view")
    clock_now = time.time   # 画像の遅れは壁時計で見る (Orin と ATMOBI PC の時計のずれも分かる)

    def on_img(m):
        t = clock_now()
        with st.lock:
            st.img_msg = m
            s = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            st.img_stamp = s if s > 0 else None
            st.img_recv.append(t)
            st.n_img += 1

    def on_path(m):
        pts = []
        for ps in m.poses[1:]:
            q = ps.pose.orientation
            yaw = 2.0 * math.atan2(q.z, q.w)
            pts.append([ps.pose.position.x, ps.pose.position.y, math.cos(yaw), math.sin(yaw)])
        with st.lock:
            st.path = np.array(pts) if pts else None
            st.path_time = clock_now()
            st.n_path += 1

    def on_status(m):
        try:
            d = json.loads(m.data)
        except ValueError:
            return
        with st.lock:
            st.status = d
            st.status_time = clock_now()

    def on_cmd(m):
        with st.lock:
            st.cmd = (m.linear.x, m.angular.z)
            st.cmd_time = clock_now()
            st.n_cmd += 1

    def on_odom(m):
        with st.lock:
            st.odom_recv.append(clock_now())

    img_type = CompressedImage if tp.image.rstrip("/").endswith("compressed") else ImageMsg
    node.create_subscription(img_type, tp.image, on_img, qos_profile_sensor_data)
    node.create_subscription(Path, "/omnivla/path", on_path, 10)
    node.create_subscription(String, "/omnivla/status", on_status, 10)
    node.create_subscription(Twist, args.cmd, on_cmd, 10)
    if tp.odom:
        node.create_subscription(Odometry, tp.odom, on_odom, qos_profile_sensor_data)

    latest = {"jpg": b"", "row": {}}
    rows: list = []
    fcsv = open(os.path.join(args.out, "steps.csv"), "w", newline="")
    writer = None

    def tick():
        nonlocal writer
        now = clock_now()
        frame, row = compose(st, robot, cam, now)
        buf = io.BytesIO()
        frame.save(buf, "JPEG", quality=85)
        latest["jpg"], latest["row"] = buf.getvalue(), row
        rows.append(row)
        if not args.no_save:
            with open(os.path.join(args.out, "frames", f"{len(rows):05d}.jpg"), "wb") as f:
                f.write(latest["jpg"])
        if writer is None:
            writer = csv.DictWriter(fcsv, fieldnames=list(row.keys()), extrasaction="ignore")
            writer.writeheader()
        writer.writerow(row)
        fcsv.flush()

    node.create_timer(args.every, tick)

    if args.port:
        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                p = self.path.split("?")[0]
                if p == "/frame.jpg":
                    body, ctype = latest["jpg"], "image/jpeg"
                elif p == "/state.json":
                    body, ctype = json.dumps(latest["row"], default=str).encode(), "application/json"
                else:
                    body, ctype = (PAGE % int(args.every * 1000)).encode(), "text/html; charset=utf-8"
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

        srv = http.server.ThreadingHTTPServer(("0.0.0.0", args.port), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        print(f"[debug_view] ブラウザ: http://<この PC の IP>:{args.port}", flush=True)
    print(f"[debug_view] トピック: {topics}", flush=True)
    print(f"[debug_view] 保存先: {args.out}  (Ctrl-C で終わり、まとめを表示)", flush=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        fcsv.close()
        txt = summarize(rows, topics)
        with open(os.path.join(args.out, "summary.txt"), "w") as f:
            f.write(txt + "\n")
        print("\n" + txt + f"\n-> {args.out}", flush=True)
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
