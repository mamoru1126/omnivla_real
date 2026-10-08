"""bag から「画像 (一定周期で間引き)・指示値・オドメトリ・自己位置」の時系列を 1 回の読み込みで取り出す.

画像は sample_rate [Hz] ごとに 1 枚だけデコードして JPEG で保存する (全フレームはメモリに載せない)。
時刻は robot.yaml の time_source に従う:
  auto   : メッセージに header.stamp があればそれ、無ければ bag の記録時刻
  header : header.stamp (無いメッセージ (Twist など) は記録時刻)
  bag    : 全て記録時刻 (センサの時計がずれているロボット向け)
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .bag.messages import (IMAGE_TYPES, header_stamp, image_to_rgb, pose_values, short_type, transform_stamp,
                           twist_values)
from .odometry import PoseTrack
from .robot_config import RobotConfig, preprocess_image


@dataclass
class Streams:
    """bag 1 本 (1 走行) 分の時系列. 時刻は全て秒."""
    name: str = ""
    image_t: np.ndarray = field(default_factory=lambda: np.zeros(0))   # 保存した (間引いた) 画像の時刻
    image_paths: List[str] = field(default_factory=list)
    image_all_t: np.ndarray = field(default_factory=lambda: np.zeros(0))  # 全画像の時刻 (周期の確認用)
    image_size: Optional[tuple] = None                                  # 保存画像の (W, H)
    cmd_t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cmd_v: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cmd_w: np.ndarray = field(default_factory=lambda: np.zeros(0))
    odom_t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    odom_x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    odom_y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    odom_yaw: np.ndarray = field(default_factory=lambda: np.zeros(0))
    odom_v: np.ndarray = field(default_factory=lambda: np.zeros(0))
    odom_w: np.ndarray = field(default_factory=lambda: np.zeros(0))
    loc_t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    loc_x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    loc_y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    loc_yaw: np.ndarray = field(default_factory=lambda: np.zeros(0))
    bag_start: float = 0.0
    bag_end: float = 0.0
    clock_offsets: Dict[str, float] = field(default_factory=dict)       # topic -> median(header - bag) [s]

    def odom_track(self) -> PoseTrack:
        return PoseTrack.from_arrays(self.odom_t, self.odom_x, self.odom_y, self.odom_yaw)

    def loc_track(self) -> PoseTrack:
        return PoseTrack.from_arrays(self.loc_t, self.loc_x, self.loc_y, self.loc_yaw)

    def summary(self) -> dict:
        def rate(t):
            return float((len(t) - 1) / (t[-1] - t[0])) if len(t) > 1 and t[-1] > t[0] else 0.0
        return {
            "duration_s": float(self.bag_end - self.bag_start),
            "images_total": int(len(self.image_all_t)), "image_rate_hz": rate(self.image_all_t),
            "images_saved": int(len(self.image_t)), "image_size": self.image_size,
            "cmd_msgs": int(len(self.cmd_t)), "cmd_rate_hz": rate(self.cmd_t),
            "odom_msgs": int(len(self.odom_t)), "odom_rate_hz": rate(self.odom_t),
            "loc_msgs": int(len(self.loc_t)), "loc_rate_hz": rate(self.loc_t),
            "clock_offsets_s": self.clock_offsets,
        }


def _msg_time(stamp: Optional[float], t_bag: float, mode: str) -> float:
    if mode == "bag" or stamp is None:
        return t_bag
    return stamp


def read_streams(source, robot: RobotConfig, frame_dir: Optional[str], sample_rate: float,
                 start_sec: Optional[float] = None, end_sec: Optional[float] = None, name: str = "",
                 progress: Optional[Callable[[str], None]] = None, decode_images: bool = True) -> Streams:
    """source: BagReader (open 済み) か ListSource. start_sec/end_sec は bag 先頭からの秒数."""
    tp = robot.topics
    topics = source.topics()
    wanted = [t for t in (tp.image, tp.odom, tp.cmd, tp.localization) if t]
    for t in wanted:
        if t not in topics:
            raise KeyError(f"topic '{t}' is not in the bag. available topics:\n  "
                           + "\n  ".join(f"{k}  [{v[0]}]" for k, v in sorted(topics.items())))
    if tp.image and short_type(topics[tp.image][0]) not in IMAGE_TYPES:
        raise ValueError(f"{tp.image} is {topics[tp.image][0]}, not an image")
    s = Streams(name=name, bag_start=source.start_time, bag_end=source.end_time)
    t_start = s.bag_start + start_sec if start_sec is not None else None
    t_stop = s.bag_start + end_sec if end_sec is not None else None
    if frame_dir and decode_images:
        os.makedirs(frame_dir, exist_ok=True)

    img_all, img_t, img_paths = [], [], []
    cmd, odom, loc = [], [], []
    offsets: Dict[str, List[float]] = {}
    period = 1.0 / sample_rate if sample_rate > 0 else 0.0
    next_due = None
    count = 0
    mode = robot.time_source
    for m in source.messages(wanted, start=t_start, stop=t_stop):
        count += 1
        if progress and count % 5000 == 0:
            progress(f"{count} messages ({m.t_bag - s.bag_start:.0f}s)")
        if m.topic == tp.image:
            stamp = header_stamp(m.msg)
            t = _msg_time(stamp, m.t_bag, mode)
            if stamp is not None:
                offsets.setdefault(m.topic, []).append(stamp - m.t_bag)
            img_all.append(t)
            if next_due is None:
                next_due = t
            if t + 1e-6 >= next_due:
                # 周期のずれが積み重ならないよう予定時刻から進める (大きく遅れたら今から数え直す)
                next_due = next_due + period if t - next_due < period else t + period
                if decode_images and frame_dir:
                    img = preprocess_image(image_to_rgb(m.msg, m.msgtype), robot.image)
                    path = os.path.join(frame_dir, f"{len(img_t):06d}.jpg")
                    img.save(path, quality=robot.image.jpeg_quality)
                    s.image_size = img.size
                    img_paths.append(path)
                else:
                    img_paths.append("")
                img_t.append(t)
        elif m.topic == tp.cmd:
            stamp = header_stamp(m.msg)
            v, w = twist_values(m.msg, m.msgtype, tp.cmd_v_field, tp.cmd_w_field)
            cmd.append((_msg_time(stamp, m.t_bag, mode), v, w))
        elif m.topic == tp.odom:
            stamp = header_stamp(m.msg)
            if stamp is not None:
                offsets.setdefault(m.topic, []).append(stamp - m.t_bag)
            x, y, yaw = pose_values(m.msg, m.msgtype)
            v, w = twist_values(m.msg, m.msgtype)
            odom.append((_msg_time(stamp, m.t_bag, mode), x, y, yaw, v, w))
        elif m.topic == tp.localization:
            pose = pose_values(m.msg, m.msgtype, tp.localization_frame, tp.localization_child_frame)
            if pose is None:
                continue
            stamp = transform_stamp(m.msg, tp.localization_frame, tp.localization_child_frame)
            loc.append((_msg_time(stamp, m.t_bag, mode),) + tuple(pose))

    s.image_all_t = np.asarray(img_all)
    order = np.argsort(np.asarray(img_t), kind="stable") if img_t else np.zeros(0, int)
    s.image_t = np.asarray(img_t)[order] if img_t else np.zeros(0)
    s.image_paths = [img_paths[i] for i in order]
    if cmd:
        a = np.asarray(sorted(cmd))
        s.cmd_t, s.cmd_v, s.cmd_w = a[:, 0], a[:, 1], a[:, 2]
    if odom:
        a = np.asarray(sorted(odom))
        s.odom_t, s.odom_x, s.odom_y, s.odom_yaw, s.odom_v, s.odom_w = (a[:, i] for i in range(6))
    if loc:
        a = np.asarray(sorted(loc))
        s.loc_t, s.loc_x, s.loc_y, s.loc_yaw = (a[:, i] for i in range(4))
    s.clock_offsets = {k: float(np.median(v)) for k, v in offsets.items() if v}
    return s


def check_clock(streams: Streams, warn_threshold: float = 0.3) -> List[str]:
    """header の時刻と記録時刻が大きくずれているトピックがあれば警告文を返す."""
    msgs = []
    for topic, off in streams.clock_offsets.items():
        if abs(off) > warn_threshold:
            msgs.append(f"{topic}: header.stamp is {off:+.2f}s from the bag time. If sensors use different "
                        f"clocks, set time_source: bag in robot.yaml")
    return msgs


def frame_period_stats(t: np.ndarray) -> dict:
    if len(t) < 2:
        return {}
    d = np.diff(t)
    return {"median_s": float(np.median(d)), "max_s": float(d.max()), "n_gaps_over_1s": int((d > 1.0).sum())}


def nan_safe(x: float) -> Optional[float]:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)
