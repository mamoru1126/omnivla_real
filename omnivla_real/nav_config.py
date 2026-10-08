"""ナビゲーション設定 (configs/navigator.yaml) の読み込み. ROS1 / ROS2 ノードと机上評価で共通."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Optional

import yaml

from .controller import ControllerConfig
from .engine import EngineConfig
from .topomap import TrackerConfig


@dataclass
class ModelConfig:
    model: str = "edge"                 # edge | 7b | remote (推論サーバ tools/policy_server.py を呼ぶ)
    weights: str = ""                   # 7b: ベースモデル (/checkpoints/omnivla-original), edge: 公式重み
    finetuned_dir: str = ""             # 学習結果 (runs/<run>/checkpoints/step_XXXXXX)
    device: str = "cuda:0"
    half: bool = False                  # edge: fp16 で推論
    metric_waypoint_spacing: float = 0.0  # 0 = 学習結果の値
    url: str = ""                       # remote: 推論サーバの URL (空なら http://127.0.0.1:8765)
    timeout: float = 10.0               # remote: 1 回の推論の待ち時間の上限 [s]


@dataclass
class IOConfig:
    cmd_vel: str = "/cmd_vel"
    cmd_stamped: bool = False           # geometry_msgs/TwistStamped で出す
    base_frame: str = "base_link"       # 予測軌跡 (Path) の frame_id
    path: str = "/omnivla/path"
    debug_image: str = "/omnivla/debug_image"
    status: str = "/omnivla/status"
    enable: str = "/omnivla/enable"     # std_msgs/Bool: true で開始, false で停止
    topomap: str = "/omnivla/topomap"   # std_msgs/String: topomap のディレクトリを送ると読み直して開始
    localization_type: str = "PoseWithCovarianceStamped"  # PoseStamped | PoseWithCovarianceStamped | Odometry
    control_rate: float = 10.0          # 指示値を出す周期 [Hz] (推論結果を保持して出し続ける)
    cmd_timeout: float = 1.0            # 推論結果がこれより古くなったら止める [s]
    max_inference_rate: float = 0.0     # 推論の上限 [Hz] (0 = 新しい画像が来るたび. カメラの周期が上限)
    log_dir: str = "/workspace/log/nav"
    log_images: bool = True


@dataclass
class NavConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    io: IOConfig = field(default_factory=IOConfig)


def _apply(obj, d: Dict[str, Any], where: str):
    names = {f.name for f in fields(obj)}
    for k, v in (d or {}).items():
        if k not in names:
            raise ValueError(f"unknown key '{k}' in {where}")
        setattr(obj, k, v)
    return obj


def load_nav_config(path: Optional[str], overrides: Optional[Dict[str, Any]] = None) -> NavConfig:
    data: Dict[str, Any] = {}
    if path:
        with open(os.path.expanduser(path)) as f:
            data = yaml.safe_load(f) or {}
    for k, v in (overrides or {}).items():
        if v is None or v == "":
            continue
        sec, _, key = k.partition(".")
        data.setdefault(sec, {})[key] = v
    cfg = NavConfig()
    _apply(cfg.model, data.get("model", {}), "model")
    _apply(cfg.io, data.get("io", {}), "io")
    eng = dict(data.get("engine", {}))
    ctrl = eng.pop("controller", {})
    trk = eng.pop("tracker", {})
    _apply(cfg.engine, eng, "engine")
    cfg.engine.controller = _apply(ControllerConfig(mode="trajectory"), ctrl, "engine.controller")
    cfg.engine.tracker = _apply(TrackerConfig(), trk, "engine.tracker")
    unknown = set(data) - {"model", "engine", "io"}
    if unknown:
        raise ValueError(f"unknown sections in navigator config: {sorted(unknown)}")
    return cfg


def make_policy(m: ModelConfig):
    from .policy_base import load_policy
    return load_policy(m.model, m.weights, m.finetuned_dir, m.device, m.metric_waypoint_spacing or None, m.half,
                       url=m.url, timeout=m.timeout)
