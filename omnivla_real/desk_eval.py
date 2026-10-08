"""机上評価: 記録した bag の画像を順にモデルへ入れ、出てくる軌跡・指示値を実際の走行と比べる.

実機を動かさずに「このモデルならコースに沿って走れそうか」を見るためのもの。各フレームで
  1. 実機と同じ処理 (NavEngine: 前処理 -> サブゴール切り替え -> 推論 -> 指示値) を行う
  2. 予測軌跡 (8 点) を、実際にその後 8 フレームで走った軌跡と比べる (ADE / FDE / 向き)
  3. 指示値 (v, w) を、記録された指示値と比べる (誤差, 曲がる向きの一致率)
  4. 指示値を時間で積算して走行軌跡を作り、記録された走行軌跡と比べる
     - 全体: スタートから積算 (誤差が積み重なるので参考)
     - 窓: 実際の位置から window 秒だけ積算した終点のずれ (局所的に「コースに沿う」か)
注意: 画像は実際に走った位置から撮られたものなので、モデルの指示値どおりに動いた場合の画像ではない
     (ずれが自分で大きくなっていく状況は再現できない)。最終的な確認は実機で行う。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .convert import ConvertConfig, FrameTable, build_frames, motion_flags
from .data_utils import NUM_ACTIONS_CHUNK
from .engine import EngineConfig, NavEngine
from .geometry import relative_pose, to_local, wrap_angle
from .policy_base import PolicyOutput
from .robot_config import RobotConfig
from .topomap import GoalNode, Topomap


@dataclass
class DeskEvalConfig:
    goal_mode: str = "topomap"           # topomap: サブゴール画像列をたどる (実機と同じ) | hindsight: 同じ bag の少し先
    goal_ahead_m: float = 1.5            # hindsight: 何 m 先のフレームをゴール画像にするか
    window_sec: Tuple[float, ...] = (3.0, 6.0, 10.0)
    window_step_sec: float = 1.0
    turn_w: float = 0.15                 # これ以上の角速度を「曲がっている」とみなす [rad/s]
    debug_every: int = 0                 # 何フレームごとにデバッグ画像を保存するか (0 で保存しない)


# ---------------------------------------------------------------------------
# 比較用のポリシー (配管の確認用)
# ---------------------------------------------------------------------------
class OraclePolicy:
    """正解 (実際にその後走った軌跡) をそのまま返す. 評価の配管・制御・積算が正しいかの確認用."""

    meta: dict = {}

    def __init__(self, ft: FrameTable, sample_rate: float, noise_m: float = 0.0, seed: int = 0):
        self.ft = ft
        self.meta = {"sample_rate": sample_rate}
        self.i = 0
        self.noise = noise_m
        self.rng = np.random.default_rng(seed)

    def predict(self, current, goal_image=None, goal_pose=None, instruction=None, modality="image") -> PolicyOutput:
        wps = ground_truth_waypoints(self.ft, self.i)
        if self.noise > 0:
            wps[:, :2] += self.rng.normal(0, self.noise, (len(wps), 2)) * np.linspace(0.3, 1, len(wps))[:, None]
        return PolicyOutput(wps, wps.copy(), 6, 0.0, np.zeros(4, np.float32))

    def embed(self, image, cache_key=None) -> np.ndarray:
        return simple_embedding(image)

    def push(self, image) -> None:
        pass

    def reset_history(self) -> None:
        pass


def simple_embedding(image) -> np.ndarray:
    """モデルを使わない簡易な画像特徴 (縮小した画素). 配管の確認用."""
    img = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image, np.uint8))
    a = np.asarray(img.convert("RGB").resize((32, 24), Image.BILINEAR), dtype=np.float32).reshape(-1)
    a = a - a.mean()
    return a / (np.linalg.norm(a) + 1e-9)


def ground_truth_waypoints(ft: FrameTable, i: int, n: int = NUM_ACTIONS_CHUNK) -> np.ndarray:
    idx = np.minimum(np.arange(i + 1, i + 1 + n), len(ft) - 1)
    loc = to_local(np.stack([ft.x[idx], ft.y[idx]], 1), (ft.x[i], ft.y[i]), ft.yaw[i])
    dyaw = ft.yaw[idx] - ft.yaw[i]
    return np.concatenate([loc, np.cos(dyaw)[:, None], np.sin(dyaw)[:, None]], axis=1)


# ---------------------------------------------------------------------------
# 積算
# ---------------------------------------------------------------------------
def integrate_commands(t: np.ndarray, v: np.ndarray, w: np.ndarray, pose0, substeps: int = 10):
    """各フレームの指示値をそのフレーム間だけ保持して積算. 戻り値 (N, 3)."""
    out = np.zeros((len(t), 3))
    x, y, yaw = pose0
    out[0] = (x, y, yaw)
    for i in range(1, len(t)):
        dt = (t[i] - t[i - 1]) / substeps
        for _ in range(substeps):
            mid = yaw + 0.5 * w[i - 1] * dt
            x += v[i - 1] * math.cos(mid) * dt
            y += v[i - 1] * math.sin(mid) * dt
            yaw += w[i - 1] * dt
        out[i] = (x, y, yaw)
    return out


def window_drift(t, v, w, gx, gy, gyaw, window: float, step: float) -> Dict[str, float]:
    """実際の位置から window 秒だけ指示値を積算したときの終点のずれ."""
    errs, yerrs, dists = [], [], []
    starts = np.arange(t[0], t[-1] - window, step)
    for ts in starts:
        a = int(np.searchsorted(t, ts))
        b = int(np.searchsorted(t, t[a] + window))
        if b >= len(t) or b - a < 2:
            continue
        p = integrate_commands(t[a:b + 1], v[a:b + 1], w[a:b + 1], (gx[a], gy[a], gyaw[a]))[-1]
        errs.append(math.hypot(p[0] - gx[b], p[1] - gy[b]))
        yerrs.append(abs(wrap_angle(p[2] - gyaw[b])))
        dists.append(math.hypot(gx[b] - gx[a], gy[b] - gy[a]))
    if not errs:
        return {"n": 0}
    return {"n": len(errs), "pos_err_median_m": float(np.median(errs)), "pos_err_p90_m": float(np.percentile(errs, 90)),
            "yaw_err_median_deg": float(np.degrees(np.median(yerrs))),
            "relative_err_median": float(np.median(np.asarray(errs) / np.maximum(np.asarray(dists), 0.1))),
            "mean_distance_m": float(np.mean(dists))}


# ---------------------------------------------------------------------------
# 本体
# ---------------------------------------------------------------------------
def hindsight_goal_index(ft: FrameTable, i: int, ahead_m: float) -> int:
    s = 0.0
    for j in range(i + 1, len(ft)):
        s += math.hypot(ft.x[j] - ft.x[j - 1], ft.y[j] - ft.y[j - 1])
        if s >= ahead_m:
            return j
    return len(ft) - 1


def nearest_node_ahead(nodes: Sequence[GoalNode], pose) -> Optional[int]:
    """今の位置から見て、まだ通り過ぎていない最初のノード (正解のサブゴール番号の目安)."""
    if pose is None or any(n.pose is None for n in nodes):
        return None
    d = [math.hypot(n.pose[0] - pose[0], n.pose[1] - pose[1]) for n in nodes]
    k = int(np.argmin(d))
    rel = relative_pose(pose, nodes[k].pose)
    return k if rel[0] > 0 or k == len(nodes) - 1 else k + 1


def run_desk_eval(ft: FrameTable, policy, robot: RobotConfig, engine_cfg: EngineConfig, cfg: DeskEvalConfig,
                  topomap: Optional[Topomap] = None, odom_xyz: Optional[np.ndarray] = None,
                  progress: Optional[Callable[[str], None]] = None, debug_cb: Optional[Callable] = None):
    """ft: 評価する bag のフレーム (正解の位置は ft.x/y/yaw). odom_xyz: 各フレームのオドメトリ (サブゴール判定用).
    戻り値: (フレームごとの記録 list[dict], まとめ dict)."""
    rec: List[dict] = []
    if cfg.goal_mode == "topomap":
        if topomap is None:
            raise ValueError("goal_mode=topomap needs a topomap")
        engine = NavEngine(policy, topomap, engine_cfg, robot)
        if odom_xyz is not None:
            engine.on_odom(ft.t[0], tuple(odom_xyz[0]))
        if ft.loc is not None and ft.loc_valid[0]:
            engine.on_localization(ft.t[0], tuple(ft.loc[0]))
        engine.start()
    else:
        engine = NavEngine(policy, Topomap([GoalNode(Image.new("RGB", (8, 8)))]), engine_cfg, robot)
    ctrl = engine.cfg.controller
    n = len(ft)
    for i in range(n):
        if isinstance(policy, OraclePolicy):
            policy.i = i
        img_path = ft.image_paths[i]
        engine.on_image(ft.t[i], lambda p=img_path: Image.open(p).convert("RGB"))
        if odom_xyz is not None:
            engine.on_odom(ft.t[i], tuple(odom_xyz[i]))
        if ft.loc is not None and ft.loc_valid[i]:
            engine.on_localization(ft.t[i], tuple(ft.loc[i]))
        r = {"i": i, "t": float(ft.t[i]), "x": float(ft.x[i]), "y": float(ft.y[i]), "yaw": float(ft.yaw[i]),
             "cmd_v": float(ft.cmd_v[i]), "cmd_w": float(ft.cmd_w[i])}
        if cfg.goal_mode == "topomap":
            if engine.state != "running":
                r.update(state=engine.state, v=0.0, w=0.0)
                rec.append(r)
                continue
            res = engine.step(ft.t[i])
            wps, v, w = res.waypoints, res.v, res.w
            r.update(state=res.state, subgoal=res.subgoal, similarity=res.similarity, latency=res.latency,
                     course_pose=engine.course_pose())
            gt_node = nearest_node_ahead(topomap.nodes, engine.course_pose()) if engine.mode != "image" else None
            r["subgoal_gt"] = gt_node
            if res.debug is not None and debug_cb and cfg.debug_every and i % cfg.debug_every == 0:
                debug_cb(i, res.debug)
        else:
            from .controller import compute_command
            j = hindsight_goal_index(ft, i, cfg.goal_ahead_m)
            cur = engine.current_image()
            goal = Image.open(ft.image_paths[j]).convert("RGB")
            if hasattr(policy, "push") and getattr(policy, "history", None) is not None:
                policy.push(cur)
            out = policy.predict(cur, goal_image=goal, modality="image")
            wps = out.waypoints
            v, w = compute_command(wps, ctrl)
            r.update(state="running", goal_i=j, latency=out.latency)
            if debug_cb and cfg.debug_every and i % cfg.debug_every == 0:
                from .robot_config import camera_model
                from .viz import render_debug
                debug_cb(i, render_debug(cur, goal, wps, cam=camera_model(robot),
                                         gt_waypoints=ground_truth_waypoints(ft, i),
                                         lines=[f"frame {i}", f"goal = frame {j}", f"v={v:.2f} w={w:.2f}"]))
        r.update(v=float(v), w=float(w))
        if wps is not None:
            gt = ground_truth_waypoints(ft, i)
            d = np.linalg.norm(wps[:, :2] - gt[:, :2], axis=1)
            yaw_p = np.arctan2(wps[-1, 3], wps[-1, 2])
            yaw_g = np.arctan2(gt[-1, 3], gt[-1, 2])
            r.update(ade=float(d.mean()), fde=float(d[-1]), yaw_err_deg=float(np.degrees(abs(wrap_angle(yaw_p - yaw_g)))),
                     gt_turn_deg=float(np.degrees(yaw_g)), pred_turn_deg=float(np.degrees(yaw_p)),
                     wps=wps[:, :2].tolist())
        rec.append(r)
        if progress and (i % 50 == 0 or i == n - 1):
            progress(f"frame {i + 1}/{n}" + (f" subgoal {r.get('subgoal')}/{len(topomap) - 1}" if topomap else ""))
        if cfg.goal_mode == "topomap" and engine.state == "reached":
            r["reached"] = True
    return rec, summarize(rec, ft, cfg, engine)


def summarize(rec: List[dict], ft: FrameTable, cfg: DeskEvalConfig, engine: NavEngine) -> dict:
    used = [r for r in rec if "ade" in r]
    out: dict = {"frames": len(rec), "frames_evaluated": len(used),
                 "sample_rate": engine.sample_rate, "controller": engine.cfg.controller.mode}
    if not used:
        return out
    t = np.array([r["t"] for r in rec])
    v = np.array([r.get("v", 0.0) for r in rec])
    w = np.array([r.get("w", 0.0) for r in rec])
    cv = np.array([r["cmd_v"] for r in rec])
    cw = np.array([r["cmd_w"] for r in rec])
    turn = np.array([abs(r.get("gt_turn_deg", 0.0)) > 20 for r in used])
    ade = np.array([r["ade"] for r in used])
    fde = np.array([r["fde"] for r in used])
    yerr = np.array([r["yaw_err_deg"] for r in used])
    out["waypoints"] = {
        "ade_m": float(ade.mean()), "fde_m": float(fde.mean()), "heading_err_deg": float(yerr.mean()),
        "turn": {"n": int(turn.sum()), "ade_m": float(ade[turn].mean()) if turn.any() else None,
                 "heading_err_deg": float(yerr[turn].mean()) if turn.any() else None},
        "straight": {"n": int((~turn).sum()), "ade_m": float(ade[~turn].mean()) if (~turn).any() else None},
    }
    ev = np.array(["v" in r and "ade" in r for r in rec])
    moving = ev & (np.abs(cv) + np.abs(cw) > 1e-3)
    turning = moving & (np.abs(cw) > cfg.turn_w)
    out["commands"] = {
        "v_mae": float(np.mean(np.abs(v[moving] - cv[moving]))) if moving.any() else None,
        "w_mae": float(np.mean(np.abs(w[moving] - cw[moving]))) if moving.any() else None,
        "turn_direction_agreement": float(np.mean(np.sign(w[turning]) == np.sign(cw[turning]))) if turning.any() else None,
        "turn_frames": int(turning.sum()),
        "w_corr": float(np.corrcoef(w[moving], cw[moving])[0, 1]) if moving.sum() > 2 and np.std(w[moving]) > 1e-6
        and np.std(cw[moving]) > 1e-6 else None,
    }
    gx, gy, gyaw = ft.x[:len(rec)], ft.y[:len(rec)], ft.yaw[:len(rec)]
    pm = integrate_commands(t, v, w, (gx[0], gy[0], gyaw[0]))
    pc = integrate_commands(t, cv, cw, (gx[0], gy[0], gyaw[0]))
    out["integrated_full"] = {
        "model_final_err_m": float(math.hypot(pm[-1, 0] - gx[-1], pm[-1, 1] - gy[-1])),
        "model_mean_err_m": float(np.mean(np.hypot(pm[:, 0] - gx, pm[:, 1] - gy))),
        "recorded_cmd_final_err_m": float(math.hypot(pc[-1, 0] - gx[-1], pc[-1, 1] - gy[-1])),
        "recorded_cmd_mean_err_m": float(np.mean(np.hypot(pc[:, 0] - gx, pc[:, 1] - gy))),
        "path_length_m": float(np.sum(np.hypot(np.diff(gx), np.diff(gy)))),
    }
    out["integrated_windows"] = {}
    for win in cfg.window_sec:
        out["integrated_windows"][f"{win:g}s"] = {
            "model": window_drift(t, v, w, gx, gy, gyaw, win, cfg.window_step_sec),
            "recorded_cmd": window_drift(t, cv, cw, gx, gy, gyaw, win, cfg.window_step_sec),
        }
    if any("subgoal" in r for r in rec):
        sg = [(r["subgoal"], r.get("subgoal_gt")) for r in rec if "subgoal" in r]
        lag = [a - b for a, b in sg if b is not None]
        out["subgoals"] = {
            "reached_goal": bool(any(r.get("reached") for r in rec)) or engine.state == "reached",
            "last_subgoal": int(sg[-1][0]), "num_nodes": len(engine.map),
            "reach_check": engine.mode,
            "index_minus_truth_median": float(np.median(lag)) if lag else None,
            "index_minus_truth_abs_p90": float(np.percentile(np.abs(lag), 90)) if lag else None,
            "similarity_median": float(np.median([r["similarity"] for r in rec if r.get("similarity") is not None]))
            if any(r.get("similarity") is not None for r in rec) else None,
        }
    out["paths"] = {"model": pm.tolist(), "recorded_cmd": pc.tolist()}
    return out


def similarity_table(policy, frames: Sequence[str], nodes: Sequence[GoalNode], frame_poses: np.ndarray,
                     stride: int = 3) -> Dict[str, float]:
    """類似度のしきい値の目安: 各フレームと、近い (< 0.5m) ノード / 遠い (> 3m) ノードの類似度の分布."""
    if any(n.pose is None for n in nodes):
        return {}
    embs = [policy.embed(n.image, cache_key=f"tbl{k}") for k, n in enumerate(nodes)]
    near, far = [], []
    for i in range(0, len(frames), stride):
        e = policy.embed(Image.open(frames[i]).convert("RGB"))
        for k, n in enumerate(nodes):
            d = math.hypot(n.pose[0] - frame_poses[i, 0], n.pose[1] - frame_poses[i, 1])
            dyaw = abs(wrap_angle(n.pose[2] - frame_poses[i, 2]))
            s = float(np.dot(e, embs[k]) / (np.linalg.norm(e) * np.linalg.norm(embs[k]) + 1e-9))
            if d < 0.5 and dyaw < math.radians(30):
                near.append(s)
            elif d > 3.0:
                far.append(s)
    if not near or not far:
        return {}
    return {"near_median": float(np.median(near)), "near_p10": float(np.percentile(near, 10)),
            "far_median": float(np.median(far)), "far_p99": float(np.percentile(far, 99)),
            "suggested_image_threshold": float((np.percentile(near, 25) + np.percentile(far, 99)) / 2)}


def trim_frames(ft: FrameTable, cfg: ConvertConfig, keep: int = 3):
    """動き始める直前から止まった直後までに絞る. 戻り値 (ft, 先頭で削ったフレーム数)."""
    moving, _ = motion_flags(ft, cfg)
    idx = np.nonzero(moving & ft.valid)[0]
    if len(idx) == 0:
        raise ValueError("the robot does not move in this bag (range)")
    a, b = max(0, int(idx[0]) - 1), min(len(ft), int(idx[-1]) + 1 + keep)
    for name in ("t", "x", "y", "yaw", "valid", "cmd_v", "cmd_w", "odom_v", "odom_w"):
        setattr(ft, name, getattr(ft, name)[a:b])
    ft.image_paths = ft.image_paths[a:b]
    if ft.loc is not None:
        ft.loc, ft.loc_valid = ft.loc[a:b], ft.loc_valid[a:b]
    return ft, a


def prepare_frames(streams, rate: float, pose_source: str = "auto"):
    """評価用のフレーム (正解の位置付き) とオドメトリ. 戻り値 (ft, odom_xyz or None, 使った位置の出所, offset)."""
    src = pose_source
    if src == "auto":
        src = "localization" if len(streams.loc_t) > 1 else ("odom" if len(streams.odom_t) > 1 else "cmd")
    ccfg = ConvertConfig(sample_rate=rate, pose_source=src)
    ft, offset = trim_frames(build_frames(streams, ccfg), ccfg)
    odom_xyz = None
    if len(streams.odom_t) > 1:
        ox, oy, oyaw, _ = streams.odom_track().at(ft.t, 1e9)
        odom_xyz = np.stack([ox, oy, oyaw], 1)
    return ft, odom_xyz, src, offset
