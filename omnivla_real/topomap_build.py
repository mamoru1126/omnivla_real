"""コースを走った bag (の時系列) からサブゴール画像列 (topomap) を作る (tools/make_topomap.py の中身)."""
from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

from .convert import ConvertConfig, build_frames, motion_flags
from .extract import Streams
from .topomap import write_topomap


def select_by_distance(x: np.ndarray, y: np.ndarray, valid: np.ndarray, spacing: float, first: int, last: int):
    """first..last のフレームから、道のり spacing [m] ごとのインデックスと各点の道のり."""
    idx, s_list = [first], [0.0]
    s = 0.0
    next_s = spacing
    for i in range(first + 1, last + 1):
        if not valid[i]:
            continue
        s += float(np.hypot(x[i] - x[i - 1], y[i] - y[i - 1]))
        if s >= next_s:
            idx.append(i)
            s_list.append(s)
            next_s = s + spacing
    if idx[-1] != last and s - s_list[-1] > 0.3 * spacing:
        idx.append(last)
        s_list.append(s)
    elif idx[-1] != last:
        idx[-1], s_list[-1] = last, s
    return idx, s_list



def build_topomap(streams: Streams, out_dir: str, spacing: float = 1.0, pose_source: str = "auto",
                  meta: Optional[dict] = None, overwrite: bool = False, jpeg_quality: int = 95) -> dict:
    """streams は画像付きで (read_streams の frame_dir を指定して) 読んだもの. 戻り値は作成結果."""
    src = pose_source
    if src == "auto":
        src = "localization" if len(streams.loc_t) > 1 else ("odom" if len(streams.odom_t) > 1 else "cmd")
    rate = 1.0 / np.median(np.diff(streams.image_t)) if len(streams.image_t) > 1 else 10.0
    cfg = ConvertConfig(sample_rate=float(rate), pose_source=src)
    ft = build_frames(streams, cfg)
    moving, _ = motion_flags(ft, cfg)
    mv = np.nonzero(moving & ft.valid)[0]
    if len(mv) == 0:
        raise ValueError("the robot does not move in the selected range")
    first, last = int(mv[0]), int(min(len(ft) - 1, mv[-1] + 1))
    sel, s_list = select_by_distance(ft.x, ft.y, ft.valid, spacing, first, last)
    images = [Image.open(ft.image_paths[i]).convert("RGB") for i in sel]
    poses = [(float(ft.x[i]), float(ft.y[i]), float(ft.yaw[i])) for i in sel]
    frame = "map" if src == "localization" else "odom"
    m = dict(meta or {})
    m.update({"spacing_m": spacing, "pose_source": src,
              "t_start": float(ft.t[first] - streams.bag_start), "t_end": float(ft.t[last] - streams.bag_start),
              "course_length_m": float(s_list[-1])})
    write_topomap(out_dir, images, poses, s_list, frame=frame, start=poses[0], meta=m, overwrite=overwrite,
                  jpeg_quality=jpeg_quality)
    return {"out": out_dir, "num_nodes": len(sel), "frame": frame, "pose_source": src, "meta": m,
            "path_xy": np.stack([ft.x[first:last + 1], ft.y[first:last + 1]], 1),
            "selected": [i - first for i in sel]}
