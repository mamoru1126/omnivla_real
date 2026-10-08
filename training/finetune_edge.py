#!/usr/bin/env python3
"""OmniVLA-edge (軽量版) を実機の rosbag から変換したデータでファインチューニングする.

公式リポジトリには edge 版の学習コードが無いため、推論コード (inference/run_omnivla_edge.py) と
同じ入力形式・出力形式 (正規化 waypoint) で、モデル全体を学習する:
  損失 = MSE(8 点の [x, y, cos, sin]) + smooth_loss_weight * MSE(隣接点の差) + dist_loss_weight * 距離ヘッド
  観測履歴は過去 context_size=5 フレーム + 現在 (間隔 context_stride フレーム)。走行時も同じ間隔で履歴を積む。

  python3 training/finetune_edge.py --config configs/finetune_edge.yaml
  python3 training/finetune_edge.py --data_dirs /data/dataset --max_steps 2000 --batch_size 16
出力: /runs/<run>/checkpoints/step_XXXXXX/{omnivla-edge.pth, finetune_meta.json}
       走行・机上評価で model: edge, weights: <この step のディレクトリ> を指定する。
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from omnivla_real.data_utils import AugmentConfig, GoalSamplingConfig, denormalize_actions  # noqa: E402
from omnivla_real.edge import (CLIP_TYPE, EDGE_PARAMS, EDGE_WEIGHTS, FINETUNE_META, TextEncoder,  # noqa: E402
                               build_edge_model, edge_forward, load_edge_weights, read_meta)
from omnivla_real.trajectory_io import find_trajectories, image_path  # noqa: E402
from omnivla_real.viz import render_debug  # noqa: E402

from common import parse_dataclass, trajectory_errors  # noqa: E402
from nav_dataset import (EdgeDataset, NavDatasetConfig, WeightedEpochSampler, data_summary,  # noqa: E402
                         edge_collate, parse_modality_weights, resolve_metric_spacing, split_trajectories)


@dataclass
class EdgeFinetuneConfig:
    # --- model ---
    weights: str = "/checkpoints/omnivla-edge"   # 学習の起点: 公式 omnivla-edge (ディレクトリ or .pth) か本スクリプトの出力
    clip_type: str = CLIP_TYPE                   # 言語特徴 (言語を使わない場合も公式は "xxxx" を CLIP に通す). "" でゼロ
    device: str = "cuda:0"
    # --- data ---
    data_dirs: List[str] = field(default_factory=lambda: ["/data/dataset"])
    val_ratio: float = 0.1
    val_bags: List[str] = field(default_factory=list)
    metric_waypoint_spacing: float = 0.0          # 0 = dataset_info.json の値
    waypoint_spacing: int = 1
    max_goal_dist: float = 30.0
    image_goal_offset: List[int] = field(default_factory=lambda: [2, 30])
    pose_goal_offset: List[int] = field(default_factory=lambda: [2, 300])
    modality_weights: Dict[str, float] = field(default_factory=lambda: {"image": 1.0})
    context_stride: int = 1                       # 観測履歴の間隔 [フレーム] (sample_rate 3Hz なら 1/3 秒おき)
    turn_sample_ratio: float = 0.5
    turn_threshold_deg: float = 45.0
    turn_horizon: int = 10
    val_turn_ratio: float = 0.5
    augment: bool = True
    crop_v: float = 0.1
    crop_h: float = 0.05
    flip_prob: float = 0.5
    color_jitter: float = 0.2
    num_workers: int = 6
    # --- optimization ---
    batch_size: int = 32
    max_steps: int = 10000
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    lr_warmup_steps: int = 200
    cosine_decay: bool = True
    max_grad_norm: float = 1.0
    smooth_loss_weight: float = 0.1
    dist_loss_weight: float = 0.0               # 距離ヘッド (ゴールまでのフレーム数) も学習するか
    amp: bool = True
    # --- logging / checkpoints ---
    run_root: str = "/runs"
    run_name: str = ""
    log_freq: int = 50
    val_freq: int = 1000
    val_batches: int = 30
    val_at_start: bool = True
    num_viz: int = 8
    save_freq: int = 2000
    keep_last: int = 3
    seed: int = 42
    dry_run: bool = False


def lr_at(step: int, cfg: EdgeFinetuneConfig) -> float:
    if step < cfg.lr_warmup_steps:
        return cfg.learning_rate * (0.1 + 0.9 * step / max(1, cfg.lr_warmup_steps))
    if not cfg.cosine_decay:
        return cfg.learning_rate
    p = (step - cfg.lr_warmup_steps) / max(1, cfg.max_steps - cfg.lr_warmup_steps)
    return cfg.learning_rate * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


def compute_loss(model, batch, device, cfg: EdgeFinetuneConfig):
    actions, dist = edge_forward(model, batch, device)
    gt = batch["actions"].to(device)
    pred = actions.float()
    l_act = F.mse_loss(pred, gt)
    l_smooth = F.mse_loss(pred[:, :-1], pred[:, 1:])
    loss = l_act + cfg.smooth_loss_weight * l_smooth
    parts = {"l2_action": float(l_act), "l2_smooth": float(l_smooth)}
    if cfg.dist_loss_weight > 0:
        l_dist = F.mse_loss(dist.float()[:, 0], batch["distance"].to(device))
        loss = loss + cfg.dist_loss_weight * l_dist
        parts["l2_dist"] = float(l_dist)
    return loss, pred, parts


@torch.no_grad()
def evaluate(model, loader, device, cfg: EdgeFinetuneConfig, max_batches: int, viz_dir: str = "",
             num_viz: int = 0) -> Dict[str, float]:
    model.eval()
    sums: Dict[str, list] = defaultdict(list)
    per = {"turn": defaultdict(list), "straight": defaultdict(list)}
    n_viz = 0
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=cfg.amp and device.type == "cuda"):
            actions, _ = edge_forward(model, batch, device)
        pred = actions.float()
        gt = batch["actions"].to(device)
        errs = trajectory_errors(pred, gt, cfg.metric_waypoint_spacing)
        for k, v in errs.items():
            vals = v.cpu().numpy().tolist()
            sums[k].extend(vals)
            for m, val in zip(batch["meta"], vals):
                per["turn" if m["turn"] else "straight"][k].append(val)
        if viz_dir and n_viz < num_viz:
            os.makedirs(viz_dir, exist_ok=True)
            from PIL import Image
            for j, m in enumerate(batch["meta"]):
                if n_viz >= num_viz:
                    break
                cur = Image.open(image_path(m["traj_dir"], m["t"])).convert("RGB")
                goal = Image.open(image_path(m["traj_dir"], m["goal_t"])).convert("RGB")
                p_m = denormalize_actions(pred[j].cpu().numpy(), cfg.metric_waypoint_spacing)
                g_m = denormalize_actions(gt[j].cpu().numpy(), cfg.metric_waypoint_spacing)
                lines = [os.path.basename(m["traj_dir"]), f"t={m['t']} goal_t={m['goal_t']}",
                         f"ADE {errs['ade'][j].item():.3f} m", "green: ground truth / orange: prediction"]
                render_debug(cur, goal, p_m, lines=lines, gt_waypoints=g_m).save(
                    os.path.join(viz_dir, f"sample_{n_viz:03d}.jpg"), quality=90)
                n_viz += 1
    model.train()
    out = {k: float(np.mean(v)) for k, v in sums.items() if v}
    out["num_samples"] = len(sums.get("ade", []))
    for name, d in per.items():
        if d["ade"]:
            out[f"{name}/ade"] = float(np.mean(d["ade"]))
            out[f"{name}/fde"] = float(np.mean(d["fde"]))
            out[f"{name}/yaw_err_deg"] = float(np.degrees(np.mean(d["yaw_err"])))
    return out


def save_checkpoint(run_dir: Path, step: int, model, cfg: EdgeFinetuneConfig, extra: dict) -> Path:
    root = run_dir / "checkpoints"
    d = root / f"step_{step:06d}"
    d.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), d / EDGE_WEIGHTS)
    meta = {"model": "edge", "step": step, "base_weights": cfg.weights,
            "metric_waypoint_spacing": cfg.metric_waypoint_spacing, "waypoint_spacing": cfg.waypoint_spacing,
            "context_size": EDGE_PARAMS["context_size"], "context_stride": cfg.context_stride,
            "clip_type": cfg.clip_type or None, "data_dirs": cfg.data_dirs,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"), **extra}
    with open(d / FINETUNE_META, "w") as f:
        json.dump(meta, f, indent=2)
    with open(root / "latest.txt", "w") as f:
        f.write(str(d) + "\n")
    if cfg.keep_last > 0:
        for p in sorted(p for p in root.glob("step_*") if p.is_dir())[:-cfg.keep_last]:
            shutil.rmtree(p, ignore_errors=True)
    return d


def main(argv=None):
    cfg = parse_dataclass(EdgeFinetuneConfig, argv, __doc__)
    dev = cfg.device if torch.cuda.is_available() or not cfg.device.startswith("cuda") else "cpu"
    device = torch.device(dev)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    run_dir = Path(cfg.run_root) / (cfg.run_name or f"omnivla_edge_{time.strftime('%Y%m%d_%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)

    def say(*a):
        print("[edge]", *a, flush=True)

    trajs = find_trajectories(cfg.data_dirs)
    if len(trajs) < 2:
        raise RuntimeError(f"need >= 2 trajectories, found {len(trajs)} in {cfg.data_dirs}")
    train_dirs, val_dirs = split_trajectories(trajs, cfg.val_ratio, cfg.seed, cfg.val_bags)
    cfg.metric_waypoint_spacing = resolve_metric_spacing(cfg.data_dirs, trajs, cfg.metric_waypoint_spacing)
    data_meta = data_summary(cfg.data_dirs, trajs)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(dataclasses.asdict(cfg), f, sort_keys=False)
    with open(run_dir / "split.json", "w") as f:
        json.dump({"train": train_dirs, "val": val_dirs}, f, indent=1)
    say(f"trajectories: train={len(train_dirs)} val={len(val_dirs)}, "
        f"metric_waypoint_spacing={cfg.metric_waypoint_spacing:.4f} m/frame, {data_meta}")

    # ---- model ----
    model = build_edge_model()
    if cfg.weights:
        base_meta = read_meta(cfg.weights)
        if base_meta and abs(float(base_meta.get("metric_waypoint_spacing", cfg.metric_waypoint_spacing))
                             - cfg.metric_waypoint_spacing) > 1e-6:
            say(f"WARNING: {cfg.weights} was trained with metric_waypoint_spacing "
                f"{base_meta.get('metric_waypoint_spacing')}")
        load_edge_weights(model, cfg.weights)
        say(f"loaded {cfg.weights}")
    else:
        say("WARNING: training from scratch (no weights)")
    model = model.to(device).train()
    say(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    text = TextEncoder(cfg.clip_type or None, str(device))
    text_features = {"": text(None).cpu()}
    data_meta["text_feature_no_language"] = [float(v) for v in text_features[""]]   # 走行時に CLIP を読まずに済む
    del text

    ds_cfg = NavDatasetConfig(
        metric_waypoint_spacing=cfg.metric_waypoint_spacing, waypoint_spacing=cfg.waypoint_spacing,
        max_goal_dist=cfg.max_goal_dist,
        goal=GoalSamplingConfig(image_goal_offset=tuple(cfg.image_goal_offset),
                                pose_goal_offset=tuple(cfg.pose_goal_offset),
                                modality_weights=parse_modality_weights(cfg.modality_weights)),
        aug=AugmentConfig(enabled=cfg.augment, crop_v=cfg.crop_v, crop_h=cfg.crop_h, flip_prob=cfg.flip_prob,
                          color_jitter=cfg.color_jitter),
        turn_horizon=cfg.turn_horizon, turn_threshold_deg=cfg.turn_threshold_deg,
        context_stride=cfg.context_stride)
    train_ds = EdgeDataset(train_dirs, ds_cfg, text_features, train=True, seed=cfg.seed)
    val_ds = EdgeDataset(val_dirs, ds_cfg, text_features, train=False, seed=cfg.seed,
                         max_samples=cfg.val_batches * cfg.batch_size, turn_ratio=cfg.val_turn_ratio) if val_dirs else None
    say(f"samples: train={len(train_ds)} val={len(val_ds) if val_ds else 0}, "
        f"turning {train_ds.turn_fraction() * 100:.1f}% -> sampled at {cfg.turn_sample_ratio * 100:.0f}%")
    sampler = WeightedEpochSampler(train_ds.sample_weights(cfg.turn_sample_ratio), len(train_ds), cfg.seed) \
        if cfg.turn_sample_ratio > 0 else None
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler, shuffle=sampler is None,
                              num_workers=cfg.num_workers, collate_fn=edge_collate, drop_last=True,
                              persistent_workers=cfg.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                            collate_fn=edge_collate) if val_ds else None

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    metrics_f = open(run_dir / "metrics.csv", "a")

    def log(step, data):
        metrics_f.write(json.dumps({"step": step, **data}) + "\n")
        metrics_f.flush()

    def validate(step):
        if val_loader is None:
            return {}
        m = evaluate(model, val_loader, device, cfg, cfg.val_batches, str(run_dir / "viz" / f"step_{step:06d}"),
                     cfg.num_viz)
        say(f"[val step {step}] ADE={m.get('ade', float('nan')):.3f}m FDE={m.get('fde', float('nan')):.3f}m "
            f"turn: ADE={m.get('turn/ade', float('nan')):.3f}m heading_err={m.get('turn/yaw_err_deg', float('nan')):.1f}deg")
        log(step, {f"val/{k}": v for k, v in m.items()})
        return m

    if cfg.val_at_start and not cfg.dry_run:
        validate(0)
    max_steps = 3 if cfg.dry_run else cfg.max_steps
    step, epoch = 0, 0
    t_last = time.time()
    hist: Dict[str, list] = defaultdict(list)
    last_val: dict = {}
    while step < max_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        train_ds.set_epoch(epoch)
        for batch in train_loader:
            for g in opt.param_groups:
                g["lr"] = lr_at(step, cfg)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=cfg.amp and device.type == "cuda"):
                loss, _, parts = compute_loss(model, batch, device, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
            step += 1
            hist["loss"].append(float(loss))
            for k, v in parts.items():
                hist[k].append(v)
            if step % cfg.log_freq == 0 or step == 1 or cfg.dry_run:
                dt = (time.time() - t_last) / max(1, cfg.log_freq if step > 1 else 1)
                t_last = time.time()
                data = {f"train/{k}": float(np.mean(v)) for k, v in hist.items()}
                data.update({"train/lr": lr_at(step, cfg), "train/sec_per_step": dt})
                say(f"step {step}/{max_steps} " + " ".join(f"{k.split('/')[1]}={v:.4g}" for k, v in data.items()))
                log(step, data)
                hist.clear()
            if not cfg.dry_run and step % cfg.val_freq == 0:
                last_val = validate(step)
            if step % cfg.save_freq == 0 or step == max_steps:
                d = save_checkpoint(run_dir, step, model, cfg, {"val": last_val, **data_meta})
                say(f"saved {d}")
            if step >= max_steps:
                break
        epoch += 1
    if not cfg.dry_run and step % cfg.val_freq != 0:
        last_val = validate(step)
        save_checkpoint(run_dir, step, model, cfg, {"val": last_val, **data_meta})
    metrics_f.close()
    say(f"done. run dir: {run_dir}")


if __name__ == "__main__":
    main()
