"""コースを走った rosbag -> OmniVLA の学習データ (GNM 形式の軌跡) への変換.

与えられるのはエピソード単位ではなく「指定コースを走り続けた」bag なので、次の手順で学習用の区間に分ける:
  1. 画像を sample_rate [Hz] (既定 3Hz = OmniVLA の推論周期) で間引き、その時刻の位置姿勢を補間
  2. 位置が取れない・画像が途切れた・後退した所で区切る
  3. 長く止まっている所 (max_stop_sec 以上) で区切り、止まっている部分は削る (最後の keep_stop_sec 秒だけ残す)
  4. 長い区間は chunk_sec ごとに分ける (学習/検証を時間の塊で分けるため)
各区間を <out>/<bag名>_<区間>_<塊>/ に保存する:
  0.jpg, 1.jpg, ...            画像
  traj_data.pkl                {"position": (N,2), "yaw": (N,), "stamp", "cmd_v", "cmd_w", "odom_v", "odom_w",
                                "exclude": (N,) 学習の起点に使わないフレーム (途中で止まっている所)}
  meta.json                    元の bag, 時刻範囲, 位置の出所, 周期など
<out>/dataset_info.json に全体の統計 (1 フレームあたりの移動量 = metric_waypoint_spacing の目安) をまとめる。
"""
from __future__ import annotations

import json
import math
import os
import pickle
import shutil
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .data_utils import NUM_ACTIONS_CHUNK, summarize_spacing, turn_flags
from .extract import Streams
from .geometry import wrap_angle
from .odometry import PoseTrack, integrate_twist, zero_order_hold
from .trajectory_io import META_FILE, TRAJ_FILE, find_trajectories, load_meta

DATASET_INFO = "dataset_info.json"
POSE_SOURCES = ("odom", "odom_twist", "cmd", "localization")


@dataclass
class ConvertConfig:
    sample_rate: float = 3.0          # 学習フレームの周期 [Hz]. OmniVLA の推論周期 (3Hz) と合わせる
    pose_source: str = "odom"         # odom | odom_twist | cmd | localization (ラベルに使う位置)
    cmd_delay: float = 0.0            # pose_source=cmd のとき: 指令から動き出すまでの遅れ [s]
    cmd_timeout: float = 0.5          # この秒数指令が来なければ 0 (停止) とみなす
    max_pose_gap: float = 0.5         # 位置のサンプル間隔がこれ以上空いたら補間しない [s]
    max_image_gap: float = 1.0        # 画像がこれ以上途切れたら区切る [s]
    min_speed: float = 0.05           # 「動いている」の判定 [m/s]
    min_yaw_rate: float = 0.1         # 「その場旋回している」の判定 [rad/s]
    max_stop_sec: float = 3.0         # これより長く止まったら区切る [s]
    keep_stop_sec: float = 1.0        # 区間の最後に残す停止フレーム [s] (「着いたら止まる」の学習用)
    allow_reverse: bool = False       # 後退を学習に含めるか
    min_segment_sec: float = 5.0
    min_segment_m: float = 1.0
    chunk_sec: float = 60.0           # 長い区間をこの長さで分ける (0 で分けない)
    start_sec: Optional[float] = None  # bag 先頭からの秒数で使う範囲を絞る
    end_sec: Optional[float] = None


@dataclass
class FrameTable:
    """間引いた画像フレームごとの値."""
    t: np.ndarray
    image_paths: List[str]
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    valid: np.ndarray
    cmd_v: np.ndarray
    cmd_w: np.ndarray
    odom_v: np.ndarray
    odom_w: np.ndarray
    loc: Optional[np.ndarray] = None  # (N, 3) 自己位置 (無ければ None)
    loc_valid: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.t)


# ---------------------------------------------------------------------------
# 位置の出所
# ---------------------------------------------------------------------------
def build_pose_track(s: Streams, cfg: ConvertConfig) -> PoseTrack:
    src = cfg.pose_source
    if src not in POSE_SOURCES:
        raise ValueError(f"pose_source must be one of {POSE_SOURCES}, got {src}")
    if src == "odom":
        if len(s.odom_t) < 2:
            raise ValueError("no odometry in the bag (topics.odom). use pose_source: cmd")
        if _pose_is_static(s) and np.abs(s.odom_v).max() > 0.05:
            raise ValueError("odometry pose does not change although the twist does. use pose_source: odom_twist")
        return s.odom_track()
    if src == "odom_twist":
        if len(s.odom_t) < 2:
            raise ValueError("no odometry in the bag (topics.odom)")
        return integrate_twist(s.odom_t, s.odom_v, s.odom_w, timeout=cfg.max_pose_gap)
    if src == "cmd":
        if len(s.cmd_t) < 2:
            raise ValueError("no command in the bag (topics.cmd)")
        return integrate_twist(s.cmd_t, s.cmd_v, s.cmd_w, delay=cfg.cmd_delay, timeout=cfg.cmd_timeout)
    if len(s.loc_t) < 2:
        raise ValueError("no localization in the bag (topics.localization)")
    return s.loc_track()


def _pose_is_static(s: Streams) -> bool:
    return (np.ptp(s.odom_x) < 1e-3 and np.ptp(s.odom_y) < 1e-3) if len(s.odom_x) else True


def build_frames(s: Streams, cfg: ConvertConfig, track: Optional[PoseTrack] = None) -> FrameTable:
    track = track or build_pose_track(s, cfg)
    t = s.image_t
    gap = cfg.max_pose_gap if cfg.pose_source in ("odom", "localization") else 1e9
    x, y, yaw, valid = track.at(t, gap)
    cmd_v = zero_order_hold(s.cmd_t, s.cmd_v, t, cfg.cmd_timeout)
    cmd_w = zero_order_hold(s.cmd_t, s.cmd_w, t, cfg.cmd_timeout)
    if len(s.odom_t) >= 2:
        odom_v = np.interp(t, s.odom_t, s.odom_v)
        odom_w = np.interp(t, s.odom_t, s.odom_w)
    else:
        odom_v = np.full(len(t), np.nan)
        odom_w = np.full(len(t), np.nan)
    loc = loc_valid = None
    if len(s.loc_t) >= 2:
        lx, ly, lyaw, loc_valid = s.loc_track().at(t, cfg.max_pose_gap)
        loc = np.stack([lx, ly, lyaw], axis=1)
    return FrameTable(t, list(s.image_paths), x, y, yaw, valid, cmd_v, cmd_w, odom_v, odom_w, loc, loc_valid)


# ---------------------------------------------------------------------------
# 区間分け
# ---------------------------------------------------------------------------
def motion_flags(ft: FrameTable, cfg: ConvertConfig) -> Tuple[np.ndarray, np.ndarray]:
    """各フレームから次のフレームまでで (動いている, 後退している)."""
    n = len(ft)
    moving = np.zeros(n, bool)
    reverse = np.zeros(n, bool)
    if n < 2:
        return moving, reverse
    dt = np.maximum(np.diff(ft.t), 1e-3)
    dx, dy = np.diff(ft.x), np.diff(ft.y)
    fwd = (dx * np.cos(ft.yaw[:-1]) + dy * np.sin(ft.yaw[:-1])) / dt
    speed = np.hypot(dx, dy) / dt
    yaw_rate = np.abs(wrap_angle(np.diff(ft.yaw))) / dt
    moving[:-1] = (speed > cfg.min_speed) | (yaw_rate > cfg.min_yaw_rate)
    reverse[:-1] = fwd < -cfg.min_speed
    return moving, reverse


def segment_frames(ft: FrameTable, cfg: ConvertConfig) -> List[Tuple[int, int]]:
    """学習に使う区間 [start, end) のリスト."""
    n = len(ft)
    if n < 2:
        return []
    moving, reverse = motion_flags(ft, cfg)
    ok = ft.valid.copy()
    if not cfg.allow_reverse:
        ok &= ~reverse
    breaks = np.zeros(n, bool)
    breaks[1:] = np.diff(ft.t) > cfg.max_image_gap
    rate = cfg.sample_rate
    max_stop = max(1, int(round(cfg.max_stop_sec * rate)))
    keep_stop = int(round(cfg.keep_stop_sec * rate))
    segs: List[Tuple[int, int]] = []

    def close(a: int, b: int) -> None:
        """[a, b) の中で、長い停止で区切り、前後の停止を削る."""
        idx = np.nonzero(moving[a:b])[0] + a
        if len(idx) == 0:
            return
        start = idx[0]
        prev = idx[0]
        for i in list(idx[1:]) + [None]:
            if i is not None and i - prev <= max_stop:
                prev = i
                continue
            end = min(prev + 1 + keep_stop, b) if i is None else min(prev + 1 + keep_stop, i)
            # 区間の最後 (止まった時刻) まで含める: prev は最後に動いていたフレーム, prev+1 で止まる
            end = max(end, min(prev + 2, b))
            segs.append((int(start), int(end)))
            if i is not None:
                start = prev = i

    a = 0
    for i in range(n + 1):
        if i == n or not ok[i] or breaks[i]:
            if i - a >= 2:
                close(a, i)
            a = i + 1 if (i < n and not ok[i]) else i
    out = []
    for a, b in segs:
        dur = ft.t[b - 1] - ft.t[a]
        length = float(np.sum(np.hypot(np.diff(ft.x[a:b]), np.diff(ft.y[a:b]))))
        if b - a >= NUM_ACTIONS_CHUNK + 2 and dur >= cfg.min_segment_sec and length >= cfg.min_segment_m:
            out.append((a, b))
    return out


def chunk_segment(a: int, b: int, cfg: ConvertConfig) -> List[Tuple[int, int]]:
    if cfg.chunk_sec <= 0:
        return [(a, b)]
    size = int(round(cfg.chunk_sec * cfg.sample_rate))
    n = b - a
    k = max(1, int(round(n / size)))
    edges = np.linspace(a, b, k + 1).round().astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(k) if edges[i + 1] - edges[i] >= NUM_ACTIONS_CHUNK + 2]


def idle_mask(x: np.ndarray, y: np.ndarray, yaw: np.ndarray, keep_tail: int, horizon: int = NUM_ACTIONS_CHUNK,
              min_move: float = 0.05, min_turn: float = math.radians(5)) -> np.ndarray:
    """途中で止まっているフレーム (この先 horizon フレーム動かない) を学習の起点から外す. 区間の最後は残す."""
    n = len(x)
    out = np.zeros(n, bool)
    for t in range(n - keep_tail - 1):
        hi = min(n - 1, t + horizon)
        move = np.hypot(x[hi] - x[t], y[hi] - y[t])
        turn = abs(wrap_angle(yaw[hi] - yaw[t]))
        out[t] = move < min_move and turn < min_turn
    return out


# ---------------------------------------------------------------------------
# 書き出し
# ---------------------------------------------------------------------------
def _link_or_copy(src: str, dst: str) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def write_trajectory(ft: FrameTable, a: int, b: int, out_dir: str, meta: dict, keep_tail: int) -> str:
    os.makedirs(out_dir, exist_ok=False)
    for k, i in enumerate(range(a, b)):
        _link_or_copy(ft.image_paths[i], os.path.join(out_dir, f"{k}.jpg"))
    x, y, yaw = ft.x[a:b], ft.y[a:b], ft.yaw[a:b]
    data = {
        "position": np.stack([x, y], axis=1).astype(np.float64),
        "yaw": yaw.astype(np.float64),
        "stamp": ft.t[a:b].astype(np.float64),
        "cmd_v": ft.cmd_v[a:b].astype(np.float32), "cmd_w": ft.cmd_w[a:b].astype(np.float32),
        "odom_v": ft.odom_v[a:b].astype(np.float32), "odom_w": ft.odom_w[a:b].astype(np.float32),
        "exclude": idle_mask(x, y, yaw, keep_tail),
    }
    if ft.loc is not None:
        data["loc_pose"] = ft.loc[a:b].astype(np.float64)
        data["loc_valid"] = ft.loc_valid[a:b].astype(bool)
    with open(os.path.join(out_dir, TRAJ_FILE), "wb") as f:
        pickle.dump(data, f)
    pos = data["position"]
    meta = dict(meta)
    meta.update({
        "num_frames": int(b - a),
        "duration_s": float(ft.t[b - 1] - ft.t[a]),
        "path_length_m": float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1))),
        "spacing_m": summarize_spacing(pos),
        "excluded_frames": int(data["exclude"].sum()),
        "turn_fraction": float(turn_flags(pos, yaw, 10, 45.0).mean()),
        "t_start": float(ft.t[a]), "t_end": float(ft.t[b - 1]),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    with open(os.path.join(out_dir, META_FILE), "w") as f:
        json.dump(meta, f, indent=1)
    return out_dir


def convert_streams(s: Streams, cfg: ConvertConfig, out_root: str, name: str, extra_meta: Optional[dict] = None,
                    overwrite: bool = False) -> dict:
    """読み込んだ時系列 -> 軌跡ディレクトリ群. 戻り値は変換結果のまとめ."""
    ft = build_frames(s, cfg)
    segs = segment_frames(ft, cfg)
    keep_tail = int(round(cfg.keep_stop_sec * cfg.sample_rate))
    os.makedirs(out_root, exist_ok=True)
    written = []
    meta = {"bag": name, "pose_source": cfg.pose_source, "sample_rate": cfg.sample_rate,
            "convert_config": asdict(cfg)}
    meta.update(extra_meta or {})
    for si, (a, b) in enumerate(segs):
        for ci, (ca, cb) in enumerate(chunk_segment(a, b, cfg)):
            d = os.path.join(out_root, f"{name}_{si:03d}_{ci:02d}")
            if os.path.exists(d):
                if not overwrite:
                    raise FileExistsError(f"{d} exists (use --overwrite)")
                shutil.rmtree(d)
            m = dict(meta, segment=si, chunk=ci)
            written.append(write_trajectory(ft, ca, cb, d, m, keep_tail if cb == b else 0))
    used = sum(b - a for a, b in segs)
    moving, _ = motion_flags(ft, cfg)
    return {
        "bag": name, "frames": len(ft), "valid_frames": int(ft.valid.sum()), "moving_frames": int(moving.sum()),
        "used_frames": int(used), "segments": len(segs), "trajectories": written,
        "segments_time": [(float(ft.t[a] - s.bag_start), float(ft.t[b - 1] - s.bag_start)) for a, b in segs],
    }


def update_dataset_info(out_root: str) -> dict:
    """out_root 以下の全軌跡から統計をまとめて dataset_info.json に書く."""
    trajs = find_trajectories(out_root)
    spacings, frames, lengths, bags, rates, sources = [], 0, 0.0, set(), set(), set()
    turn = []
    for d in trajs:
        m = load_meta(d)
        if m.get("spacing_m"):
            spacings.extend([m["spacing_m"]] * max(1, int(m.get("num_frames", 1))))
        frames += int(m.get("num_frames", 0))
        lengths += float(m.get("path_length_m", 0.0))
        bags.add(m.get("bag", "?"))
        rates.add(m.get("sample_rate"))
        sources.add(m.get("pose_source"))
        turn.append((m.get("turn_fraction", 0.0), int(m.get("num_frames", 0))))
    info = {
        "trajectories": len(trajs), "frames": frames, "path_length_m": lengths, "bags": sorted(bags),
        "sample_rate": sorted(r for r in rates if r is not None),
        "pose_source": sorted(s for s in sources if s),
        "metric_waypoint_spacing": float(np.median(spacings)) if spacings else None,
        "turn_fraction": float(sum(f * n for f, n in turn) / max(1, sum(n for _, n in turn))),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if len(info["sample_rate"]) > 1 or len(info["pose_source"]) > 1:
        info["warning"] = "trajectories were converted with different sample_rate / pose_source"
    with open(os.path.join(out_root, DATASET_INFO), "w") as f:
        json.dump(info, f, indent=1)
    return info


def read_dataset_info(dirs) -> dict:
    for d in ([dirs] if isinstance(dirs, str) else dirs):
        p = os.path.join(os.path.expanduser(d), DATASET_INFO)
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return {}
