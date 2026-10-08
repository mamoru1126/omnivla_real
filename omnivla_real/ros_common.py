"""ROS1 / ROS2 ノード共通の処理 (ROS のパッケージには依存しない).

ノード側は「メッセージを受けて NavRunner に渡す」「NavRunner の出力をメッセージにして出す」だけにする。
推論 (遅い) は別スレッドで回し、指示値は control_rate で最新の結果を出し続ける (古くなったら 0)。
推論は新しいカメラ画像が来たときだけ行う (カメラ 10Hz なら最大 10Hz. 推論に時間がかかればその分遅くなる)。
"""
from __future__ import annotations

import json
import math
import threading
import time
from typing import Callable, Optional, Tuple

import numpy as np

from .engine import NavEngine, StepResult
from .nav_config import NavConfig, load_nav_config, make_policy
from .navlog import NavRunLogger
from .robot_config import RobotConfig
from .topomap import Topomap, load_topomap


class NavRunner:
    """エンジン + 推論スレッド + 指示値の保持. clock() は ROS の時刻 (bag 再生時は /clock) を返す関数."""

    def __init__(self, robot_cfg: str, nav_cfg: str, topomap: str = "", overrides: Optional[dict] = None,
                 clock: Callable[[], float] = time.time, log: Callable[[str], None] = print):
        self.robot = RobotConfig.load(robot_cfg)
        self.cfg: NavConfig = load_nav_config(nav_cfg, overrides)
        self.clock = clock
        self.log = log
        log(f"loading model ({self.cfg.model.model}) ...")
        self.policy = make_policy(self.cfg.model)
        tm = load_topomap(topomap) if topomap else Topomap([])
        self.engine = NavEngine(self.policy, tm, self.cfg.engine, self.robot)
        self.lock = threading.Lock()
        self.cmd: Tuple[float, float] = (0.0, 0.0)
        self.cmd_time = -1e9
        self.stopped_at = -1e9
        self.last_result: Optional[StepResult] = None
        self.topomap_path = topomap
        self._thread: Optional[threading.Thread] = None
        self._alive = True
        self.on_result: Optional[Callable[[StepResult], None]] = None
        log(f"ready: model={self.cfg.model.model}, sample_rate={self.engine.sample_rate} Hz, "
            f"topomap={topomap or '(none)'} ({len(tm)} subgoals)")

    # ------------------------------------------------------------------ 開始/停止
    def start(self, topomap: str = "") -> bool:
        with self.lock:
            if topomap:
                self.topomap_path = topomap
                tm = load_topomap(topomap)
            else:
                tm = self.engine.map
            if len(tm) == 0:
                self.log("no topomap. publish a directory to the topomap topic")
                return False
            if self.engine.logger:
                self.engine.logger.close("restarted")
            self.engine.logger = self._make_logger(tm)
            mode = self.engine.start(tm if topomap else None)
            self.cmd, self.cmd_time = (0.0, 0.0), -1e9
        self.log(f"start: {len(tm)} subgoals, reach_check={mode}, log={self.engine.logger.dir if self.engine.logger else '-'}")
        self._ensure_thread()
        return True

    def stop(self, reason: str = "stopped by user") -> None:
        with self.lock:
            self.engine.stop(reason)
            if self.engine.logger:
                self.engine.logger.close(reason)
            self.cmd, self.stopped_at = (0.0, 0.0), self.clock()
        self.log(f"stop: {reason}")

    def shutdown(self) -> None:
        self._alive = False
        if self.engine.state == "running":
            self.stop("shutdown")

    def _make_logger(self, tm: Topomap) -> Optional[NavRunLogger]:
        if not self.cfg.io.log_dir:
            return None
        final = tm.nodes[-1].pose if tm.nodes and tm.nodes[-1].pose is not None else None
        meta = {"topomap": self.topomap_path, "topomap_meta": tm.meta, "num_nodes": len(tm),
                "final_goal_pose": final, "model": vars(self.cfg.model), "robot": self.robot.to_dict(),
                "engine": {"modality": self.cfg.engine.modality, "sample_rate": self.engine.sample_rate,
                           "controller": vars(self.cfg.engine.controller), "tracker": vars(self.cfg.engine.tracker)}}
        try:
            return NavRunLogger(self.cfg.io.log_dir, meta, [n.image for n in tm.nodes],
                                save_debug=self.cfg.io.log_images, save_raw=self.cfg.io.log_images)
        except OSError as e:
            self.log(f"cannot write logs to {self.cfg.io.log_dir}: {e}")
            return None

    # ------------------------------------------------------------------ 推論スレッド
    def _ensure_thread(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def _loop(self) -> None:
        period = 1.0 / self.cfg.io.max_inference_rate if self.cfg.io.max_inference_rate > 0 else 0.0
        while self._alive:
            if self.engine.state != "running":
                time.sleep(0.05)
                continue
            if not self.engine.has_new_image():   # 新しい画像が来るまで待つ (同じ画像で推論し直さない)
                time.sleep(0.005)
                continue
            t0 = time.time()
            try:
                with self.lock:
                    res = self.engine.step(self.clock())
                    if res.state == "running":
                        self.cmd, self.cmd_time = (res.v, res.w), self.clock()
                    elif res.state in ("reached", "stopped"):
                        self.cmd, self.stopped_at = (0.0, 0.0), self.clock()
                    self.last_result = res
                for e in res.events:
                    self.log(e)
                if res.state in ("reached", "stopped"):
                    self.log(f"finished: {res.state} {res.reason}")
                if self.on_result:
                    self.on_result(res)
            except Exception as e:  # 推論の例外でノードごと落ちないようにする
                self.log(f"ERROR in step: {e!r}")
                self.cmd = (0.0, 0.0)
                time.sleep(0.5)
            wait = period - (time.time() - t0)
            if wait > 0:
                time.sleep(wait)
            if self.last_result is not None and self.last_result.state == "waiting_image":
                time.sleep(0.05)  # 画像が来るまで空回りしない

    # ------------------------------------------------------------------ 出力
    def command(self) -> Optional[Tuple[float, float]]:
        """今出すべき指示値. None なら何も出さない (止めてから 1 秒以上経った時)."""
        now = self.clock()
        if self.engine.state == "running":
            if now - self.cmd_time <= self.cfg.io.cmd_timeout:
                return self.cmd
            return (0.0, 0.0)
        if now - self.stopped_at < 1.0:
            return (0.0, 0.0)
        return None

    def status_json(self) -> str:
        st = self.engine.status()
        st["topomap"] = self.topomap_path
        return json.dumps(st, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))


def path_points(waypoints: Optional[np.ndarray]):
    """予測 8 点 -> [(x, y, yaw)] (原点 (0,0,0) を先頭に付ける)."""
    pts = [(0.0, 0.0, 0.0)]
    if waypoints is None:
        return pts
    for wp in waypoints:
        pts.append((float(wp[0]), float(wp[1]), math.atan2(float(wp[3]), float(wp[2]))))
    return pts


def quaternion_from_yaw(yaw: float):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)
