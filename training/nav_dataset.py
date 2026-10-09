"""bag から変換した軌跡 (GNM 形式) から OmniVLA / OmniVLA-edge の学習サンプルを作る Dataset.

  * ゴール画像は同じ軌跡の未来フレーム (hindsight relabeling), ゴール姿勢は位置の時系列から計算
  * 正解はこの先 8 フレームの位置と向き (metric_waypoint_spacing で正規化)
  * 途中で止まっているフレーム (traj_data.pkl の exclude) は学習の起点にしない
  * 軌跡末尾は最終姿勢でパディング (= ゴール付近では「止まる」ことを学習)
OmniVLADataset (7B) は prismatic に、EdgeDataset は OmniVLA-edge の入力形式に合わせる。
"""
from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from omnivla_real.data_utils import (AugmentConfig, GoalSamplingConfig, augment_pair, balanced_weights,  # noqa: E402
                                     build_sample_index, color_jitter, flip_left_right, label_valid_mask,
                                     make_targets, modality_id, random_crop_box, reweight, turn_flags)
from omnivla_real.trajectory_io import image_path, load_meta, load_trajectory  # noqa: E402


@dataclass
class NavDatasetConfig:
    metric_waypoint_spacing: float = 0.1   # 正規化に使う 1 単位 [m] (dataset_info.json の値を使う)
    waypoint_spacing: int = 1              # 何フレームおきに waypoint を取るか
    max_goal_dist: float = 30.0
    goal: GoalSamplingConfig = field(default_factory=GoalSamplingConfig)
    aug: AugmentConfig = field(default_factory=AugmentConfig)
    turn_horizon: int = 10                 # 「曲がるサンプル」の判定: この先何フレーム以内に
    turn_threshold_deg: float = 45.0       #   何度以上曲がるか
    context_size: int = 5                  # edge: 過去何フレームを入れるか
    context_stride: int = 1                # edge: 過去フレームの間隔 (フレーム数)


def parse_modality_weights(weights: Dict) -> Dict[int, float]:
    return {modality_id(k): float(v) for k, v in weights.items()}


class BaseNavDataset(Dataset):
    epoch = 0

    def __init__(self, traj_dirs: Sequence[str], cfg: NavDatasetConfig, train: bool = True,
                 force_modality: Optional[int] = None, max_samples: Optional[int] = None, seed: int = 0,
                 turn_ratio: float = 0.0, recovery_ratio: float = 0.0):
        """turn_ratio / recovery_ratio: max_samples で間引くとき (検証セット) の曲がるサンプルなどの割合."""
        self.cfg = cfg
        self.train = train
        self.force_modality = force_modality
        self.seed = seed
        self.trajs: List[dict] = []
        for d in traj_dirs:
            data = load_trajectory(d)
            n = len(data["position"])
            if n < 2 or not os.path.exists(image_path(d, n - 1)):
                print(f"[dataset] skip {d} (frames={n})")
                continue
            valid = label_valid_mask(data["perturbed"], 8, cfg.waypoint_spacing) & ~data["exclude"]
            self.trajs.append({"dir": d, "name": os.path.basename(d), "position": data["position"],
                               "yaw": data["yaw"], "n": n,
                               "turn": turn_flags(data["position"], data["yaw"], cfg.turn_horizon,
                                                  cfg.turn_threshold_deg),
                               "perturbed": data["perturbed"], "valid": valid})
        self.index = [(ti, t) for ti, t in build_sample_index([t["n"] for t in self.trajs])
                      if self.trajs[ti]["valid"][t]]
        self.recovery = np.array([bool(self.trajs[ti]["perturbed"][max(0, t - 6):t + 1].any())
                                  for ti, t in self.index], dtype=bool)
        self.turn = np.array([bool(self.trajs[ti]["turn"][t]) for ti, t in self.index], dtype=bool)
        if max_samples is not None and len(self.index) > max_samples:
            rng = np.random.default_rng(seed)
            p = reweight(balanced_weights(self.turn, turn_ratio), self.recovery, recovery_ratio)
            keep = np.sort(rng.choice(len(self.index), size=max_samples, replace=False, p=p))
            self.index = [self.index[i] for i in keep]
            self.turn = self.turn[keep]
            self.recovery = self.recovery[keep]
        if not self.index:
            raise ValueError("dataset is empty")

    def __len__(self) -> int:
        return len(self.index)

    def turn_fraction(self) -> float:
        return float(self.turn.mean()) if len(self.turn) else 0.0

    def recovery_fraction(self) -> float:
        return float(self.recovery.mean()) if len(self.recovery) else 0.0

    def sample_weights(self, turn_ratio: float, recovery_ratio: float = 0.0) -> np.ndarray:
        return reweight(balanced_weights(self.turn, turn_ratio), self.recovery, recovery_ratio)

    def num_frames(self) -> int:
        return int(sum(t["n"] for t in self.trajs))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng(self, i: int) -> np.random.Generator:
        if self.train:
            return np.random.default_rng([torch.initial_seed() % (2 ** 32), self.epoch, i])
        return np.random.default_rng([self.seed, i])  # 検証は毎回同じサンプル

    def targets(self, i: int):
        ti, t = self.index[i]
        tr = self.trajs[ti]
        rng = self._rng(i)
        mod, goal_t, actions, goal_pose = make_targets(
            rng, tr["position"], tr["yaw"], t, self.cfg.goal, self.cfg.metric_waypoint_spacing,
            self.cfg.waypoint_spacing, self.cfg.max_goal_dist, self.force_modality)
        return tr, t, rng, mod, goal_t, actions, goal_pose

    def meta(self, i: int, tr: dict, t: int, goal_t: int, mod: int) -> dict:
        return {"traj_dir": tr["dir"], "t": int(t), "goal_t": int(goal_t), "modality": mod,
                "turn": bool(self.turn[i]), "recovery": bool(self.recovery[i])}


class OmniVLADataset(BaseNavDataset):
    """OmniVLA 7B 用 (プロンプト・224x224 画像 2 枚)."""

    def __init__(self, traj_dirs, processor, action_tokenizer, cfg: NavDatasetConfig, **kw):
        super().__init__(traj_dirs, cfg, **kw)
        self.tokenizer = processor.tokenizer
        self.image_transform = processor.image_processor.apply_transform
        self.action_tokenizer = action_tokenizer

    def __getitem__(self, i: int) -> dict:
        from omnivla_real.omnivla_model import build_prompt, make_sample
        tr, t, rng, mod, goal_t, actions, goal_pose = self.targets(i)
        cur = Image.open(image_path(tr["dir"], t)).convert("RGB")
        goal = Image.open(image_path(tr["dir"], goal_t)).convert("RGB")
        cur, goal, actions, goal_pose = augment_pair(rng, cur, goal, actions, goal_pose, self.cfg.aug, self.train)
        input_ids, labels = build_prompt(self.tokenizer, self.action_tokenizer, None, actions)
        sample = make_sample(self.image_transform, cur, goal, input_ids, labels, goal_pose, mod, actions)
        sample["meta"] = self.meta(i, tr, t, goal_t, mod)
        return sample


class EdgeDataset(BaseNavDataset):
    """OmniVLA-edge 用 (過去 context_size フレーム + 現在, ゴール画像 96x96, 現在画像 224x224)."""

    def __init__(self, traj_dirs, cfg: NavDatasetConfig, text_features: Dict[str, torch.Tensor], **kw):
        super().__init__(traj_dirs, cfg, **kw)
        self.text_features = text_features  # {"": 言語なし ("xxxx") の特徴}

    def __getitem__(self, i: int) -> dict:
        from omnivla_real.edge import make_edge_batch
        tr, t, rng, mod, goal_t, actions, goal_pose = self.targets(i)
        c = self.cfg
        hist_idx = [max(0, t - c.context_stride * k) for k in range(c.context_size, -1, -1)]
        obs = [Image.open(image_path(tr["dir"], j)).convert("RGB") for j in hist_idx]
        goal = Image.open(image_path(tr["dir"], goal_t)).convert("RGB")
        if self.train and c.aug.enabled:
            u = (rng.random(), rng.random())
            box = lambda im: random_crop_box(None, im.width, im.height, c.aug.crop_v, c.aug.crop_h, u)  # noqa: E731
            obs = [im.crop(box(im)) for im in obs]
            goal = goal.crop(box(goal))
            if rng.random() < c.aug.flip_prob:
                obs = [im.transpose(Image.FLIP_LEFT_RIGHT) for im in obs]
                goal = goal.transpose(Image.FLIP_LEFT_RIGHT)
                actions, goal_pose = flip_left_right(actions, goal_pose)
            seed = int(rng.integers(1 << 31))
            obs = [color_jitter(im, np.random.default_rng(seed), c.aug.color_jitter) for im in obs]  # 履歴は同じ変動
            goal = color_jitter(goal, rng, c.aug.color_jitter)
        sample = make_edge_batch(obs, goal, goal_pose, mod, self.text_features[""], c.context_size)
        sample["actions"] = torch.as_tensor(actions, dtype=torch.float32)
        sample["distance"] = torch.tensor(float(goal_t - t), dtype=torch.float32)  # ゴールまでのフレーム数
        sample["meta"] = self.meta(i, tr, t, goal_t, mod)
        return sample


def edge_collate(items: List[dict]) -> dict:
    out = {k: torch.stack([x[k] for x in items]) for k in items[0] if k != "meta"}
    out["meta"] = [x["meta"] for x in items]
    return out


class WeightedEpochSampler(torch.utils.data.Sampler):
    """重み付き復元抽出のサンプラ (epoch ごとに変わる, DDP では rank ごとに別の乱数)."""

    def __init__(self, weights, num_samples: int, seed: int = 0, rank: int = 0):
        self.weights = torch.as_tensor(np.asarray(weights), dtype=torch.double)
        self.num_samples = int(num_samples)
        self.seed, self.rank, self.epoch = int(seed), int(rank), 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed * 100003 + self.epoch * 1009 + self.rank)
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True, generator=g).tolist())

    def __len__(self) -> int:
        return self.num_samples


def split_trajectories(traj_dirs: Sequence[str], val_ratio: float, seed: int, val_bags: Sequence[str] = ()):
    """train/val に分ける. val_bags を指定するとその bag の軌跡を全て検証に使う (同じコースの別の走行で評価).
    指定が無ければ軌跡 (= 最大 chunk_sec 秒の塊) 単位でランダムに分ける."""
    dirs = sorted(traj_dirs)
    if val_bags:
        # YAML では引用符なしの 20260919_150554 が数値 20260919150554 になる (_ は桁区切り扱い). 数値で来たら _ を除いた名前と比べる
        vb = {str(v) for v in val_bags}
        loose = {str(v) for v in val_bags if isinstance(v, int)}

        def hit(name) -> bool:
            name = str(name)
            return name in vb or name.replace("_", "") in loose

        val = [d for d in dirs if hit(load_meta(d).get("bag"))]
        if not val:
            names = sorted({str(load_meta(d).get("bag")) for d in dirs})
            raise ValueError(f"no trajectories from val_bags {sorted(vb)}. bag names in the dataset: {names} "
                             "(数字だけの名前は引用符で囲む: val_bags: [\"20260919_150554\"])")
        return [d for d in dirs if d not in val], val
    rng = random.Random(seed)
    rng.shuffle(dirs)
    n_val = int(round(len(dirs) * val_ratio))
    if val_ratio > 0 and len(dirs) >= 2:
        n_val = max(1, n_val)
    n_val = min(n_val, len(dirs) - 1)
    return sorted(dirs[n_val:]), sorted(dirs[:n_val])


def resolve_metric_spacing(data_dirs: Sequence[str], traj_dirs: Sequence[str], value: float) -> float:
    """metric_waypoint_spacing [m/frame]. value > 0 ならそのまま、0 なら dataset_info.json か軌跡から求める."""
    if value and value > 0:
        return float(value)
    from omnivla_real.convert import read_dataset_info
    from omnivla_real.data_utils import summarize_spacing
    info = read_dataset_info(data_dirs)
    if info.get("metric_waypoint_spacing"):
        return float(info["metric_waypoint_spacing"])
    vals = [summarize_spacing(load_trajectory(d)["position"]) for d in traj_dirs]
    vals = [v for v in vals if v]
    if not vals:
        raise ValueError("cannot determine metric_waypoint_spacing; set it explicitly")
    return float(np.median(vals))


def data_summary(data_dirs: Sequence[str], traj_dirs: Sequence[str]) -> dict:
    """推論側 (navigator / desk_eval) に引き継ぐ学習データの情報 (sample_rate = waypoint の時間間隔)."""
    from omnivla_real.convert import read_dataset_info
    info = read_dataset_info(data_dirs)
    rates = info.get("sample_rate") or sorted({load_meta(d).get("sample_rate") for d in traj_dirs} - {None})
    if len(rates) > 1:
        raise ValueError(f"trajectories have different sample_rate {rates}; convert them with the same rate")
    return {"sample_rate": float(rates[0]) if rates else 3.0,
            "pose_source": (info.get("pose_source") or [None])[0],
            "dataset_bags": info.get("bags", [])}
