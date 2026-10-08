"""ロボット固有の設定 (トピック名・画像の前処理) を configs/robot.yaml から読む.

同じ設定を「bag の変換」「サブゴール画像の作成」「机上評価」「実機ナビゲーション」で共有する。
特に画像の切り抜き (crop) は学習時と走行時で同じでないと見え方がずれるので、必ずここで一元管理する。
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from PIL import Image


@dataclass
class TopicConfig:
    image: str = "/camera/image_raw/compressed"   # sensor_msgs/Image or CompressedImage
    odom: str = "/odom"                            # nav_msgs/Odometry (ホイールオドメトリ)
    cmd: str = "/cmd_vel"                          # geometry_msgs/Twist or TwistStamped (指示値)
    localization: str = ""                         # 任意: PoseStamped / PoseWithCovarianceStamped / Odometry / /tf
    localization_frame: str = "map"                # /tf の場合の親フレーム
    localization_child_frame: str = "base_link"    # /tf の場合の子フレーム
    cmd_v_field: str = ""                          # 独自メッセージの場合: 例 "drive.speed"
    cmd_w_field: str = ""                          #                       例 "drive.steering_rate"


@dataclass
class ImageConfig:
    crop: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])  # 上, 下, 左, 右 を割合で削る
    width: int = 320          # 保存する画像の幅 [px] (縦横比は維持). 0 で元のまま
    rotate180: bool = False   # カメラが逆さまに付いている場合
    jpeg_quality: int = 95


@dataclass
class CameraConfig:
    """予測軌跡を画像に重ねて表示するためだけに使う (学習・推論には影響しない)."""
    hfov_deg: float = 90.0     # 水平画角 (切り抜き後) [deg]
    height: float = 0.5        # 地面からの高さ [m]
    x_offset: float = 0.2      # ロボット原点からの前方オフセット [m]


@dataclass
class RobotConfig:
    name: str = "robot"
    topics: TopicConfig = field(default_factory=TopicConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    time_source: str = "auto"  # auto (header の時刻があれば使う) | header | bag (記録時刻)

    @staticmethod
    def load(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> "RobotConfig":
        data: Dict[str, Any] = {}
        if path:
            with open(os.path.expanduser(path)) as f:
                data = yaml.safe_load(f) or {}
        data = _deep_update(data, overrides or {})
        cfg = RobotConfig()
        cfg.name = data.get("name", cfg.name)
        cfg.time_source = data.get("time_source", cfg.time_source)
        cfg.topics = _from_dict(TopicConfig, data.get("topics", {}))
        cfg.image = _from_dict(ImageConfig, data.get("image", {}))
        cfg.camera = _from_dict(CameraConfig, data.get("camera", {}))
        if cfg.time_source not in ("auto", "header", "bag"):
            raise ValueError(f"time_source must be auto/header/bag, got {cfg.time_source}")
        if len(cfg.image.crop) != 4:
            raise ValueError("image.crop must be [top, bottom, left, right]")
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)


def camera_model(cfg: "RobotConfig"):
    from .viz import CameraModel
    import math
    return CameraModel(hfov=math.radians(cfg.camera.hfov_deg), height=cfg.camera.height, x_offset=cfg.camera.x_offset)


def _from_dict(cls, d: Dict[str, Any]):
    names = {f.name for f in fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**{k: v for k, v in d.items() if k in names})


def _deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def preprocess_image(rgb, cfg: ImageConfig) -> Image.Image:
    """カメラ画像 -> 学習/推論に使う画像 (切り抜き・回転・縮小). 変換/サブゴール/走行で共通."""
    img = Image.fromarray(np.asarray(rgb, dtype=np.uint8)) if not isinstance(rgb, Image.Image) else rgb
    img = img.convert("RGB")
    if cfg.rotate180:
        img = img.rotate(180)
    top, bottom, left, right = [float(v) for v in cfg.crop]
    if any(v > 0 for v in (top, bottom, left, right)):
        w, h = img.size
        box = (int(round(left * w)), int(round(top * h)), int(round(w - right * w)), int(round(h - bottom * h)))
        if box[2] - box[0] < 8 or box[3] - box[1] < 8:
            raise ValueError(f"image.crop {cfg.crop} leaves no image")
        img = img.crop(box)
    if cfg.width and img.size[0] != cfg.width:
        w, h = img.size
        img = img.resize((int(cfg.width), max(1, int(round(h * cfg.width / w)))), Image.BILINEAR)
    return img
