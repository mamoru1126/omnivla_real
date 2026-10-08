"""推論結果の共通型と、設定からポリシー (7B / edge) を作る関数 (torch 以外の重い依存なし)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class PolicyOutput:
    waypoints: np.ndarray        # (8, 4) [x[m], y[m], cos, sin] ロボット座標 (x前, y左). k 点目は (k+1)/sample_rate 秒後
    normalized: np.ndarray       # (8, 4) モデル生出力
    modality: int
    latency: float               # [s]
    goal_pose_input: np.ndarray  # (4,) モデルに入れた正規化 goal pose
    distance: Optional[float] = None  # edge のみ: 距離ヘッドの出力 (参考)


def load_policy(model: str, weights: str = "", finetuned_dir: str = "", device: str = "cuda:0",
                metric_waypoint_spacing: Optional[float] = None, half: bool = False):
    """model: '7b' (OmniVLA) | 'edge' (OmniVLA-edge)."""
    model = model.lower()
    if model in ("7b", "omnivla"):
        from .policy import OmniVLAPolicy, PolicyConfig
        return OmniVLAPolicy(PolicyConfig(vla_path=weights or "/checkpoints/omnivla-original",
                                          finetuned_dir=finetuned_dir or None, device=device,
                                          metric_waypoint_spacing=metric_waypoint_spacing))
    if model in ("edge", "omnivla-edge"):
        from .edge import EdgePolicy, EdgePolicyConfig
        return EdgePolicy(EdgePolicyConfig(weights=finetuned_dir or weights or "/checkpoints/omnivla-edge",
                                           device=device, metric_waypoint_spacing=metric_waypoint_spacing,
                                           half=half))
    raise ValueError(f"unknown model '{model}' (7b | edge)")
