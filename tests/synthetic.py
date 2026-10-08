"""テスト用: コースを走った rosbag と同じ形のメッセージ列を合成する (ROS / rosbags 不要).

ロボットは長方形のコース (角は円弧) を経路追従で 1 周する。
  * 指示値 /cmd_vel (Twist, 20Hz)
  * 実際の動き = 指示値を delay 秒遅らせ、前進は slip 倍 (タイヤの滑り)
  * ホイールオドメトリ /odom (Odometry, 50Hz): 実際の動き + 小さなヨードリフト
  * カメラ /camera/image_raw/compressed (CompressedImage, 15Hz): 位置と向きで決まる簡単な絵 (JPEG)
  * 自己位置 /localization (PoseStamped, 10Hz): 実際の位置
途中で stop_at 秒から stop_len 秒止まり、最初と最後にも止まっている時間を入れる。
"""
from __future__ import annotations

import io
import math
from types import SimpleNamespace as NS
from typing import List, Optional

import numpy as np
from PIL import Image

from omnivla_real.bag.reader import BagMessage
from omnivla_real.expert import FollowerConfig, PathFollower


def stamp(t: float) -> NS:
    sec = int(math.floor(t))
    return NS(sec=sec, nanosec=int(round((t - sec) * 1e9)))


def header(t: float, frame: str = "") -> NS:
    return NS(stamp=stamp(t), frame_id=frame)


def quat(yaw: float) -> NS:
    return NS(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))


def course_path(width: float = 8.0, height: float = 5.0, r: float = 1.2, step: float = 0.05,
                closed: bool = False) -> np.ndarray:
    """原点から +x 向きにスタートする角丸長方形 (反時計回り). closed=False なら最後の角の手前で終わる."""
    pts = []

    def line(p0, p1):
        n = max(2, int(np.hypot(*(np.subtract(p1, p0))) / step))
        for k in range(n):
            pts.append(np.add(p0, np.subtract(p1, p0) * k / n))

    def arc(c, a0, a1):
        n = max(4, int(abs(a1 - a0) * r / step))
        for k in range(n):
            a = a0 + (a1 - a0) * k / n
            pts.append((c[0] + r * math.cos(a), c[1] + r * math.sin(a)))

    w, h = width, height
    line((0, 0), (w - r, 0))
    arc((w - r, r), -math.pi / 2, 0)
    line((w, r), (w, h - r))
    arc((w - r, h - r), 0, math.pi / 2)
    line((w - r, h), (r, h))
    arc((r, h - r), math.pi / 2, math.pi)
    line((0, h - r), (0, r))
    if closed:
        arc((r, r), math.pi, 1.5 * math.pi)
        pts.append((0.0, 0.0))
    else:
        pts.append((0.0, r))
    return np.asarray(pts, dtype=np.float64)


def render(x: float, y: float, yaw: float, size=(64, 48)) -> Image.Image:
    """位置と向きで変わる簡単な絵 (近い位置・同じ向きなら似た画像になる)."""
    w, h = size
    cols = np.arange(w) / w - 0.5
    ang = yaw + cols * 1.6                       # 水平画角 ~ 90deg
    # 遠くの「壁」の模様: 方位角で色が変わる + 位置で少しずれる
    phase = ang * 3.0 + 0.15 * x - 0.1 * y
    r = 127 + 120 * np.sin(phase)
    g = 127 + 120 * np.sin(phase * 0.7 + 1.0 + 0.2 * y)
    b = 127 + 120 * np.cos(ang * 1.3 - 0.1 * x)
    row = np.stack([r, g, b], axis=-1)
    img = np.repeat(row[None], h, axis=0)
    img[h // 2:] *= np.linspace(0.6, 0.3, h - h // 2)[:, None, None]  # 床
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def jpeg_bytes(img: Image.Image) -> np.ndarray:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return np.frombuffer(buf.getvalue(), dtype=np.uint8)


def simulate_course(t0: float = 1000.0, speed: float = 0.6, delay: float = 0.15, slip: float = 0.97,
                    yaw_drift: float = 0.002, idle_start: float = 2.0, idle_end: float = 3.0,
                    stop_at: Optional[float] = 20.0, stop_len: float = 6.0, dt: float = 0.01,
                    image_rate: float = 15.0, with_localization: bool = True, raw_image: bool = False,
                    path: Optional[np.ndarray] = None, image_size=(64, 48), laps: int = 1,
                    twist_stamped: bool = False) -> List[BagMessage]:
    if path is None:
        path = course_path()
        if laps > 1:
            loop = course_path(closed=True)
            path = np.concatenate([loop] * (laps - 1) + [path])
    fol = PathFollower(path, FollowerConfig(speed=speed, lookahead=0.6, max_angular=1.0, goal_tolerance=0.15))
    msgs: List[BagMessage] = []
    true = [0.0, 0.0, 0.0]
    odom = [0.0, 0.0, 0.0]
    cmd_hist = []                   # (t, v, w)
    t = t0
    t_end_motion = None
    next_cmd = next_odom = next_img = next_loc = t0
    done = False
    stopped_until = None
    while True:
        rel = t - t0
        # ---- 指示値 (20Hz) ----
        if t >= next_cmd - 1e-9:
            next_cmd += 0.05
            if rel < idle_start or done:
                v = w = 0.0
            elif stop_at is not None and stop_at <= rel < stop_at + stop_len:
                v = w = 0.0
            else:
                v, w, reached = fol.step(tuple(true))
                if reached:
                    done = True
                    t_end_motion = t
                    v = w = 0.0
            cmd_hist.append((t, v, w))
            tw = NS(linear=NS(x=v, y=0.0, z=0.0), angular=NS(x=0.0, y=0.0, z=w))
            if twist_stamped:
                msgs.append(BagMessage("/cmd_vel", "geometry_msgs/msg/TwistStamped", t, NS(header=header(t), twist=tw)))
            else:
                msgs.append(BagMessage("/cmd_vel", "geometry_msgs/msg/Twist", t, tw))
        # ---- 実際の動き (delay 秒前の指示値) ----
        ca = [c for c in cmd_hist if c[0] <= t - delay]
        v_act, w_act = (ca[-1][1] * slip, ca[-1][2]) if ca else (0.0, 0.0)
        mid = true[2] + 0.5 * w_act * dt
        true = [true[0] + v_act * math.cos(mid) * dt, true[1] + v_act * math.sin(mid) * dt, true[2] + w_act * dt]
        # オドメトリ: 前進は指示どおり (滑りを知らない), ヨーは少しドリフト
        v_od = v_act / slip
        w_od = w_act + (yaw_drift if abs(v_act) > 1e-3 else 0.0)
        mid = odom[2] + 0.5 * w_od * dt
        odom = [odom[0] + v_od * math.cos(mid) * dt, odom[1] + v_od * math.sin(mid) * dt, odom[2] + w_od * dt]
        if t >= next_odom - 1e-9:
            next_odom += 0.02
            m = NS(header=header(t, "odom"), child_frame_id="base_link",
                   pose=NS(pose=NS(position=NS(x=odom[0], y=odom[1], z=0.0), orientation=quat(odom[2])),
                           covariance=[0.0] * 36),
                   twist=NS(twist=NS(linear=NS(x=v_od, y=0.0, z=0.0), angular=NS(x=0.0, y=0.0, z=w_od)),
                            covariance=[0.0] * 36))
            msgs.append(BagMessage("/odom", "nav_msgs/msg/Odometry", t + 0.002, m))
        if with_localization and t >= next_loc - 1e-9:
            next_loc += 0.1
            m = NS(header=header(t, "map"), pose=NS(position=NS(x=true[0] + 3.0, y=true[1] - 2.0, z=0.0),
                                                     orientation=quat(true[2])))
            msgs.append(BagMessage("/localization", "geometry_msgs/msg/PoseStamped", t + 0.003, m))
        if t >= next_img - 1e-9:
            next_img += 1.0 / image_rate
            img = render(true[0], true[1], true[2], image_size)
            if raw_image:
                arr = np.asarray(img)[..., ::-1]  # bgr8
                m = NS(header=header(t, "camera"), height=img.size[1], width=img.size[0], encoding="bgr8",
                       is_bigendian=0, step=img.size[0] * 3, data=arr.reshape(-1).copy())
                msgs.append(BagMessage("/camera/image_raw", "sensor_msgs/msg/Image", t + 0.01, m))
            else:
                m = NS(header=header(t, "camera"), format="jpeg", data=jpeg_bytes(img))
                msgs.append(BagMessage("/camera/image_raw/compressed", "sensor_msgs/msg/CompressedImage",
                                       t + 0.01, m))
        t += dt
        if t_end_motion is not None and t - t_end_motion > idle_end:
            break
        if t - t0 > 600:
            raise RuntimeError("simulation did not finish")
    return msgs


def true_path_from_messages(msgs: List[BagMessage]) -> np.ndarray:
    pts = [(m.msg.pose.position.x - 3.0, m.msg.pose.position.y + 2.0) for m in msgs if m.topic == "/localization"]
    return np.asarray(pts)
