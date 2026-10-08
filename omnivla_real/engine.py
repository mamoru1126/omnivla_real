"""実機ナビゲーションの中身 (ROS 非依存). ROS1 / ROS2 のノードはこれを呼ぶだけの薄いラッパ.

入力 : カメラ画像 (必須), ホイールオドメトリ (任意), 自己位置 (任意)
出力 : 指示値 (v, w) と、この先 約 (8 / sample_rate) 秒の予測軌跡 (ロボット座標 x前, y左 の 8 点)

1 ステップ (step()) の処理:
  1. 最新のカメラ画像を学習時と同じ前処理 (robot.yaml の crop / 縮小) にかける
  2. サブゴールに着いたか判定して切り替える (topomap.SubgoalTracker)
  3. OmniVLA (7B / edge) に「現在画像 + 今のサブゴール画像」を入れて 8 点の軌跡を予測
  4. 予測軌跡を再現する (v, w) を計算 (controller: trajectory)
  5. 最終ゴールに着いたら止まる
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from .controller import ControllerConfig, StuckDetector, compute_command
from .geometry import relative_pose
from .robot_config import RobotConfig, camera_model, preprocess_image
from .topomap import SubgoalTracker, Topomap, TrackerConfig, align_to_start, resolve_mode
from .viz import render_debug

Pose = Tuple[float, float, float]
ImageSource = Union[np.ndarray, Image.Image, Callable[[], Any]]


@dataclass
class EngineConfig:
    modality: str = "image"            # image | image_pose (自己位置/オドメトリで相対ゴール位置も入れる)
    sample_rate: float = 0.0           # 学習データの周期 [Hz] (waypoint の時間間隔). 0 = モデルの meta から (無ければ 3)
    controller: ControllerConfig = field(default_factory=lambda: ControllerConfig(mode="trajectory"))
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    max_image_age: float = 0.5         # これより古い画像しか無ければ止まる [s]
    stuck_timeout: float = 0.0         # 前進指令中にこの秒数動かなければ止める (オドメトリがある場合. 0 で無効)
    stop_at_goal: bool = True
    debug_image: bool = True


@dataclass
class StepResult:
    v: float
    w: float
    waypoints: Optional[np.ndarray]    # (8, 4) [x, y, cos, sin] ロボット座標 (m)
    state: str                         # running | reached | idle | waiting_image | stopped
    subgoal: int
    num_nodes: int
    latency: float = 0.0
    similarity: Optional[float] = None
    reason: str = ""
    debug: Optional[Image.Image] = None
    events: List[str] = field(default_factory=list)


class NavEngine:
    def __init__(self, policy, topomap: Topomap, cfg: EngineConfig, robot: RobotConfig, logger=None):
        self.policy = policy
        self.map = topomap
        self.cfg = cfg
        self.robot = robot
        self.logger = logger
        meta = getattr(policy, "meta", None) or getattr(getattr(policy, "c", None), "meta", None) or {}
        self.sample_rate = float(cfg.sample_rate or meta.get("sample_rate") or 3.0)
        self.cfg.controller.dt = 1.0 / self.sample_rate  # waypoint k は (k+1)/sample_rate 秒後
        self.cam = camera_model(robot)
        self.state = "idle"
        self.tracker: Optional[SubgoalTracker] = None
        self.mode = "image"
        self.stuck = StuckDetector(timeout=cfg.stuck_timeout) if cfg.stuck_timeout > 0 else None
        # 入力
        self._img_t: Optional[float] = None
        self._img_src: Optional[ImageSource] = None
        self._img_cache: Optional[Image.Image] = None
        self._next_hist: Optional[float] = None
        self.odom: Optional[Pose] = None
        self.odom_start: Optional[Pose] = None
        self.travel = 0.0
        self.loc: Optional[Pose] = None
        self.n_step = 0
        self._node_emb: Dict[int, np.ndarray] = {}
        self._emb_now: Dict[str, np.ndarray] = {}
        self.last: Optional[StepResult] = None
        self._img_seq = 0          # 受け取った画像の通し番号
        self._used_seq = -1        # 最後に推論に使った画像の番号 (同じ画像で何度も推論しない)

    # ------------------------------------------------------------------ 入力
    def on_image(self, t: float, image: ImageSource) -> None:
        """カメラ画像 (ndarray / PIL / デコードする関数). 観測履歴 (edge) は sample_rate ごとに積む."""
        self._img_t, self._img_src, self._img_cache = t, image, None
        self._img_seq += 1
        if self._next_hist is None or t + 1e-6 >= self._next_hist:
            period = 1.0 / self.sample_rate
            self._next_hist = (self._next_hist + period) if (self._next_hist and t - self._next_hist < period) \
                else t + period
            if hasattr(self.policy, "push") and getattr(self.policy, "history", None) is not None:
                self.policy.push(self.current_image())

    def has_new_image(self) -> bool:
        """まだ推論に使っていない画像があるか (推論はカメラの周期より速くは回さない)."""
        return self._img_t is not None and self._img_seq != self._used_seq

    def current_image(self) -> Optional[Image.Image]:
        if self._img_cache is None and self._img_src is not None:
            src = self._img_src() if callable(self._img_src) else self._img_src
            self._img_cache = preprocess_image(src, self.robot.image)
        return self._img_cache

    def on_odom(self, t: float, pose: Pose) -> None:
        if self.odom is not None:
            self.travel += math.hypot(pose[0] - self.odom[0], pose[1] - self.odom[1])
        self.odom = tuple(pose)

    def on_localization(self, t: float, pose: Pose) -> None:
        self.loc = tuple(pose)

    # ------------------------------------------------------------------ 制御
    def start(self, topomap: Optional[Topomap] = None, reason: str = "start") -> str:
        """走行開始. ロボットは topomap を作った走行のスタート地点・向きに置いておくこと."""
        if topomap is not None:
            self.map = topomap
            self._node_emb.clear()
        self.mode = resolve_mode(self.cfg.tracker.reach_check, self.loc is not None, self.odom is not None, self.map)
        self.tracker = SubgoalTracker(self.map, self.cfg.tracker, self.mode)
        self.odom_start = self.odom
        self.travel = 0.0
        self.state = "running"
        if hasattr(self.policy, "reset_history"):
            self.policy.reset_history()
        if self.stuck:
            self.stuck.reset()
        self._event(f"{reason}: {len(self.map)} subgoals, reach_check={self.mode}, sample_rate={self.sample_rate}")
        return self.mode

    def stop(self, reason: str = "stopped") -> None:
        if self.state == "running":
            self._event(f"stop: {reason}")
        self.state = "stopped" if reason != "reached" else "reached"

    def course_pose(self) -> Optional[Pose]:
        """topomap と同じ座標での今の位置 (自己位置 or スタートで重ねたオドメトリ)."""
        if self.mode == "pose":
            return self.loc
        if self.odom is not None and self.odom_start is not None and self.map.start is not None:
            return align_to_start(self.odom, self.odom_start, self.map.start)
        return None

    def _similarity(self, j: int) -> float:
        if j not in self._node_emb:
            self._node_emb[j] = self.policy.embed(self.map.nodes[j].image, cache_key=f"node{j}:{id(self.map)}")
        if "cur" not in self._emb_now:
            self._emb_now["cur"] = self.policy.embed(self.current_image())
        return float(np.dot(self._emb_now["cur"], self._node_emb[j]) /
                     (np.linalg.norm(self._emb_now["cur"]) * np.linalg.norm(self._node_emb[j]) + 1e-9))

    def step(self, t: float) -> StepResult:
        n = len(self.map)
        events: List[str] = []
        if self.state != "running" or self.tracker is None:
            return self._result(0.0, 0.0, None, self.state, events=events)
        if self._img_t is None or t - self._img_t > self.cfg.max_image_age:
            return self._result(0.0, 0.0, None, "waiting_image", reason="no recent image", events=events)
        self._used_seq = self._img_seq
        cur = self.current_image()
        pose = self.course_pose()
        # --- サブゴールの切り替え ---
        self._emb_now = {}
        before = self.tracker.index
        sim_fn = self._similarity if self.mode in ("image", "image_odom") else None
        if self.tracker.update(pose, sim_fn, self.travel if self.odom is not None else None):
            if self.tracker.done:
                events.append(f"final goal reached [{self.tracker.last_reason}]")
            else:
                events.append(f"subgoal {before} -> {self.tracker.index}/{n - 1} [{self.tracker.last_reason}]")
        for e in events:
            self._event(e, t)
        if self.tracker.done and self.cfg.stop_at_goal:
            self.state = "reached"
            res = self._result(0.0, 0.0, None, "reached", reason=self.tracker.last_reason or "", events=events)
            if self.logger:
                self.logger.close("reached", reached=True)
            return res
        # --- 推論 ---
        node = self.tracker.current
        goal_pose = None
        if "pose" in self.cfg.modality and pose is not None and node.pose is not None:
            goal_pose = relative_pose(pose, node.pose)
        out = self.policy.predict(cur, goal_image=node.image, goal_pose=goal_pose, modality=self.cfg.modality)
        v, w = compute_command(out.waypoints, self.cfg.controller)
        # --- 動けない (押し付け) ---
        if self.stuck is not None and self.odom is not None and self.stuck.update(t, self.odom, v):
            self.stop("stuck")
            self._event(f"stuck: no movement for {self.cfg.stuck_timeout}s", t)
            if self.logger:
                self.logger.close("stuck")
            return self._result(0.0, 0.0, out.waypoints, "stopped", latency=out.latency, reason="stuck",
                                events=events)
        self.n_step += 1
        dbg = self._debug(cur, node, out, v, w, t) if self.cfg.debug_image else None
        res = self._result(v, w, out.waypoints, "running", latency=out.latency, events=events, debug=dbg)
        if self.logger:
            gl = relative_pose(pose, node.pose)[:2] if pose is not None and node.pose is not None else None
            self.logger.step(self.n_step, t, pose, self.tracker.index, n, node.pose, gl, out.modality,
                             self.cfg.controller.mode, v, w, out.latency, out.waypoints,
                             self.tracker.last_distance, self.tracker.last_similarity, "running", dbg, cur)
        return res

    # ------------------------------------------------------------------ util
    def _result(self, v, w, wps, state, latency=0.0, reason="", events=None, debug=None) -> StepResult:
        tr = self.tracker
        r = StepResult(float(v), float(w), wps, state, tr.index if tr else 0, len(self.map), latency,
                       tr.last_similarity if tr else None, reason, debug, events or [])
        self.last = r
        return r

    def _event(self, text: str, t: Optional[float] = None) -> None:
        if self.logger:
            self.logger.event(t, text)

    def _debug(self, cur, node, out, v, w, t) -> Image.Image:
        tr = self.tracker
        lines = [f"step {self.n_step}  t={t:.1f}", f"subgoal {tr.index}/{len(self.map) - 1}  ({self.mode})",
                 f"v={v:.2f} m/s  w={w:.2f} rad/s", f"latency {out.latency * 1000:.0f} ms"]
        if tr.last_similarity is not None:
            lines.append(f"similarity {tr.last_similarity:.3f}")
        if tr.last_distance is not None:
            lines.append(f"dist to subgoal {tr.last_distance:.2f} m")
        return render_debug(cur, node.image, out.waypoints, cam=self.cam, lines=lines)

    def status(self) -> dict:
        tr = self.tracker
        return {"state": self.state, "subgoal": tr.index if tr else None, "num_nodes": len(self.map),
                "reach_check": self.mode, "similarity": tr.last_similarity if tr else None,
                "distance": tr.last_distance if tr else None, "travel_m": self.travel,
                "v": self.last.v if self.last else 0.0, "w": self.last.w if self.last else 0.0,
                "latency": self.last.latency if self.last else None, "time": time.time()}
