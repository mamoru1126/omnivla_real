"""OmniVLA-edge (軽量版, EfficientNet + 小さな Transformer) の読み込み・入力作成・推論.

公式 inference/run_omnivla_edge.py との違い:
  * 観測履歴 (過去 5 フレーム + 現在) に本当の過去画像を入れる (公式サンプルは現在画像を 6 回複製している)
  * ゴール画像・ゴール姿勢・言語を引数で受け取り、7B 版 (policy.OmniVLAPolicy) と同じ PolicyOutput を返す
  * 本リポジトリのファインチューニング結果 (finetune_meta.json 付き) を読める

入力の形 (公式と同じ):
  obs_images  (B, 3*(context+1), 96, 96)  過去 -> 現在 の順に 3ch ずつ連結 (ImageNet 正規化)
  goal_pose   (B, 4)                       [x/s, y/s, cos, sin]
  map_images  (B, 9, 96, 96)               [衛星画像(現在), 衛星画像(ゴール), 現在画像] 衛星画像は黒で埋める
  goal_image  (B, 3, 96, 96)
  modality_id (B,)                          7B 版と同じ id (6 = 画像のみ)
  feat_text   (B, 512)                      CLIP ViT-B/32 のテキスト特徴 (言語を使わない時は "xxxx")
  cur_large   (B, 3, 224, 224)             言語用 FiLM の入力 (現在画像)
出力: (B, 8, 4) 正規化 waypoint (位置は累積済み, cos/sin は正規化済み) と距離予測 (B, 1)
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .data_utils import (IMAGE_MODALITIES, LANGUAGE_MODALITIES, MODALITY_NAMES, POSE_MODALITIES, SUPPORTED_MODALITIES,
                         denormalize_actions, modality_id, normalize_goal_pose)
from .policy_base import PolicyOutput

EDGE_PARAMS = dict(  # 公式 run_omnivla_edge.py と同じ
    context_size=5, len_traj_pred=8, learn_angle=True, obs_encoder="efficientnet-b0", obs_encoding_size=1024,
    late_fusion=False, mha_num_attention_heads=4, mha_num_attention_layers=4, mha_ff_dim_factor=4,
)
CLIP_TYPE = "ViT-B/32"
TEXT_DIM = 512
NO_LANGUAGE = "xxxx"  # 公式: 言語を使わない modality では "xxxx" を CLIP に通す
EDGE_WEIGHTS = "omnivla-edge.pth"
FINETUNE_META = "finetune_meta.json"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
ImageLike = Union[Image.Image, np.ndarray]


def build_edge_model(**overrides):
    """公式 inference/model_omnivla_edge.py (third_party にコピー. CPU でも動くよう device の扱いだけ修正)."""
    from .third_party.model_omnivla_edge import OmniVLA_edge
    params = dict(EDGE_PARAMS)
    params.update(overrides)
    return OmniVLA_edge(**params)


def to_tensor_norm(img: Image.Image, size: int) -> torch.Tensor:
    """resize (縦横比無視) -> [0,1] -> ImageNet 正規化. (3, size, size)."""
    img = img.convert("RGB").resize((size, size), Image.BILINEAR)
    t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return (t - IMAGENET_MEAN) / IMAGENET_STD


def black_map(size: int = 96) -> torch.Tensor:
    """公式: 衛星画像は使わない場合は黒画像を正規化したもの."""
    return ((torch.zeros(3, size, size) - IMAGENET_MEAN) / IMAGENET_STD)


def load_clip(clip_type: str, device: str = "cpu"):
    """clip.load. 公式の配布元 (openaipublic.azureedge.net) に届かない場合は同じファイルを blob から取る."""
    import clip  # openai-clip
    try:
        return clip.load(clip_type, device=device)
    except Exception as e:  # noqa: BLE001  (ダウンロード失敗)
        url = getattr(clip.clip, "_MODELS", {}).get(clip_type)
        if not url or "azureedge.net" not in url:
            raise
        alt = url.replace("openaipublic.azureedge.net", "openaipublic.blob.core.windows.net")
        print(f"[clip] {e!r} -> retry from {alt}")
        path = clip.clip._download(alt, os.path.expanduser("~/.cache/clip"))
        return clip.load(path, device=device)


class TextEncoder:
    """CLIP のテキスト特徴 (キャッシュ付き). clip_type=None ならゼロベクトル (テスト・言語を使わない場合用).
    preset: 計算済みの特徴 {テキスト: (512,)}. 学習結果には「言語なし」の特徴が入っているので、
    言語を使わない走行では CLIP を読み込まない (CLIP は必要になった時に初めて読む)."""

    def __init__(self, clip_type: Optional[str] = CLIP_TYPE, device: str = "cpu",
                 preset: Optional[Dict[str, torch.Tensor]] = None):
        self.device = device
        self.clip_type = clip_type
        self.model = None
        self.cache: Dict[str, torch.Tensor] = dict(preset or {})

    def _ensure_model(self) -> None:
        if self.model is None:
            import clip  # openai-clip
            self.model, _ = load_clip(self.clip_type, self.device)
            self.model = self.model.float().eval()
            self._tokenize = clip.tokenize

    @torch.no_grad()
    def __call__(self, text: Optional[str]) -> torch.Tensor:
        key = text or NO_LANGUAGE
        if key not in self.cache:
            if not self.clip_type:
                self.cache[key] = torch.zeros(TEXT_DIM)
            else:
                self._ensure_model()
                tok = self._tokenize(key, truncate=True).to(self.device)
                self.cache[key] = self.model.encode_text(tok)[0].float().cpu()
        return self.cache[key]


def make_edge_batch(obs: Sequence[Image.Image], goal: Image.Image, goal_pose: np.ndarray, modality: int,
                    text_feat: torch.Tensor, context_size: int = 5) -> Dict[str, torch.Tensor]:
    """1 サンプル分の入力 (バッチ次元なし). obs は古い順で長さ context_size+1 (足りなければ先頭を複製)."""
    obs = list(obs)[-(context_size + 1):]
    while len(obs) < context_size + 1:
        obs.insert(0, obs[0])
    obs_t = [to_tensor_norm(im, 96) for im in obs]
    cur96 = obs_t[-1]
    return {
        "obs_images": torch.cat(obs_t, dim=0),
        "goal_pose": torch.as_tensor(goal_pose, dtype=torch.float32),
        "map_images": torch.cat([black_map(96), black_map(96), cur96], dim=0),
        "goal_image": to_tensor_norm(goal, 96),
        "modality_id": torch.tensor(int(modality), dtype=torch.long),
        "feat_text": text_feat.float(),
        "cur_large": to_tensor_norm(obs[-1], 224),
    }


def edge_forward(model, batch: Dict[str, torch.Tensor], device) -> tuple:
    """(actions (B,8,4), dist (B,1))."""
    b = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    actions, dist, _ = model(b["obs_images"], b["goal_pose"], b["map_images"], b["goal_image"], b["modality_id"],
                             b["feat_text"], b["cur_large"])
    return actions, dist


def read_meta(path: str) -> dict:
    d = path if os.path.isdir(path) else os.path.dirname(path)
    p = os.path.join(d, FINETUNE_META)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def resolve_weights(path: str) -> str:
    """ディレクトリなら中の omnivla-edge.pth."""
    path = os.path.expanduser(path)
    if os.path.isdir(path):
        path = os.path.join(path, EDGE_WEIGHTS)
    if not os.path.exists(path):
        raise FileNotFoundError(f"OmniVLA-edge weights not found: {path}")
    return path


def load_edge_weights(model, path: str) -> None:
    state = torch.load(resolve_weights(path), map_location="cpu")
    if isinstance(state, dict) and "model" in state and not any(k.startswith("obs_encoder") for k in state):
        state = state["model"]
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)


@dataclass
class EdgePolicyConfig:
    weights: str = "/checkpoints/omnivla-edge"      # 公式重みのディレクトリ or 学習結果 (step_XXXXXX/)
    device: str = "cuda:0"
    clip_type: Optional[str] = CLIP_TYPE           # None でゼロ (言語を使わない場合. 学習時と揃えること)
    metric_waypoint_spacing: Optional[float] = None  # None: finetune_meta.json の値 or 0.1
    max_goal_dist: float = 30.0
    context_stride: Optional[int] = None           # 観測履歴の間隔 (フレーム). None: meta の値 or 1
    half: bool = False                             # fp16 で推論 (Jetson で速くしたい場合)


class EdgePolicy:
    """OmniVLA-edge の推論. 7B 版 OmniVLAPolicy と同じ predict() / embed() を持つ."""

    def __init__(self, cfg: EdgePolicyConfig, model=None):
        self.cfg = cfg
        dev = cfg.device if torch.cuda.is_available() or not str(cfg.device).startswith("cuda") else "cpu"
        self.device = torch.device(dev)
        self.meta = read_meta(cfg.weights) if model is None else {}
        if model is None:
            model = build_edge_model()
            load_edge_weights(model, cfg.weights)
        self.model = model.to(self.device).eval()
        if cfg.half and self.device.type == "cuda":
            self.model = self.model.half()
        ms = cfg.metric_waypoint_spacing or self.meta.get("metric_waypoint_spacing") or 0.1
        self.metric_spacing = float(ms)
        self.context_size = int(EDGE_PARAMS["context_size"])
        self.context_stride = int(cfg.context_stride or self.meta.get("context_stride") or 1)
        clip_type = self.meta.get("clip_type", cfg.clip_type) if self.meta else cfg.clip_type
        preset = None
        if self.meta.get("text_feature_no_language") is not None:   # 学習時に計算した「言語なし」の特徴
            preset = {NO_LANGUAGE: torch.tensor(self.meta["text_feature_no_language"], dtype=torch.float32)}
        self.text = TextEncoder(clip_type, str(self.device), preset=preset)
        self.history: Deque[Image.Image] = deque(maxlen=self.context_size * self.context_stride + 1)
        self._emb_cache: Dict[str, np.ndarray] = {}
        print(f"[edge] ready (device={self.device}, metric_waypoint_spacing={self.metric_spacing}, "
              f"context_stride={self.context_stride})")

    # -- 観測履歴 ------------------------------------------------------------
    def push(self, image: ImageLike) -> None:
        """推論周期 (学習時の sample_rate と同じ周期) ごとに現在画像を積む."""
        self.history.append(_to_pil(image))

    def reset_history(self) -> None:
        self.history.clear()

    def observation_window(self, current: Image.Image) -> List[Image.Image]:
        hist = list(self.history)
        if not hist or hist[-1] is not current:
            hist.append(current)
        s = self.context_stride
        idx = [len(hist) - 1 - s * k for k in range(self.context_size, -1, -1)]
        return [hist[max(0, i)] for i in idx]

    # -- 推論 ---------------------------------------------------------------
    @torch.no_grad()
    def predict(self, current: ImageLike, goal_image: Optional[ImageLike] = None,
                goal_pose: Optional[Sequence[float]] = None, instruction: Optional[str] = None,
                modality: Union[str, int] = "image", observations: Optional[Sequence[ImageLike]] = None) -> PolicyOutput:
        mid = modality_id(modality)
        if mid not in SUPPORTED_MODALITIES:
            raise ValueError(f"modality {MODALITY_NAMES[mid]} is not supported")
        if mid in IMAGE_MODALITIES and goal_image is None:
            raise ValueError("goal_image is required")
        if mid in POSE_MODALITIES and goal_pose is None:
            raise ValueError("goal_pose is required")
        t0 = time.time()
        cur = _to_pil(current)
        obs = [_to_pil(o) for o in observations] if observations is not None else self.observation_window(cur)
        goal = _to_pil(goal_image) if goal_image is not None else cur
        if goal_pose is not None and mid in POSE_MODALITIES:
            gp = normalize_goal_pose(float(goal_pose[0]), float(goal_pose[1]), float(goal_pose[2]),
                                     self.metric_spacing, self.cfg.max_goal_dist)
        else:
            gp = np.zeros(4, dtype=np.float32)
        feat = self.text(instruction if mid in LANGUAGE_MODALITIES else None)
        sample = make_edge_batch(obs, goal, gp, mid, feat, self.context_size)
        batch = {k: v.unsqueeze(0) for k, v in sample.items()}
        if self.cfg.half and self.device.type == "cuda":
            batch = {k: (v.half() if v.is_floating_point() else v) for k, v in batch.items()}
        actions, dist = edge_forward(self.model, batch, self.device)
        norm = actions[0].float().cpu().numpy()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return PolicyOutput(denormalize_actions(norm, self.metric_spacing), norm, mid, time.time() - t0, gp,
                            distance=float(dist[0, 0].float().cpu()))

    @torch.no_grad()
    def embed(self, image: ImageLike, cache_key: Optional[str] = None) -> np.ndarray:
        """サブゴール判定用の画像特徴 (観測エンコーダ EfficientNet の平均プーリング, L2 正規化)."""
        if cache_key is not None and cache_key in self._emb_cache:
            return self._emb_cache[cache_key]
        x = to_tensor_norm(_to_pil(image), 96).unsqueeze(0).to(self.device)
        if self.cfg.half and self.device.type == "cuda":
            x = x.half()
        enc = self.model.obs_encoder
        f = enc._avg_pooling(enc.extract_features(x)).flatten(1)
        emb = F.normalize(f.float(), dim=-1)[0].cpu().numpy()
        if cache_key is not None:
            self._emb_cache[cache_key] = emb
        return emb

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _to_pil(img: ImageLike) -> Image.Image:
    if isinstance(img, np.ndarray):
        return Image.fromarray(img.astype(np.uint8)).convert("RGB")
    return img.convert("RGB")
