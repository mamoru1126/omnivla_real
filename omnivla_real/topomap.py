"""サブゴール画像列 (topomap) の読み書きと、走行中のサブゴール切り替え (ROS 非依存).

ディレクトリ形式 (tools/make_topomap.py が bag から作る):
    <topomap>/
        0.jpg, 1.jpg, ..., N.jpg   コースに沿って一定距離ごとの画像 (N が最終ゴール)
        poses.yaml                 frame: odom | map, start: {x,y,yaw}, nodes: [{image, x, y, yaw, s}]
                                   (s はスタートからの道のり [m])
        topomap.json               作成元の bag・間隔など

サブゴールの切り替え (reach_check):
  pose      自己位置 (map 座標) とノード位置の距離・向きで判定. topomap も自己位置付きの bag から作る必要がある
  odom      ホイールオドメトリで判定. 走行開始時のロボット位置を topomap の start と同じ場所とみなして重ねる
            (長いコースではドリフトするので単独では非推奨)
  image     画像の類似度で「今どのノードの近くにいるか」を推定 (現在のノード〜search_window 先まで比較)
  image_odom image と同じだが、オドメトリで明らかに遠いノードには切り替えない / 画像で判定できないまま
            オドメトリ上で通り過ぎたら進める (見た目が似た場所での誤判定を防ぐ)
  auto      自己位置があれば pose, オドメトリがあれば image_odom, どちらも無ければ image
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from PIL import Image

from .geometry import relative_pose, wrap_angle

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
REACH_MODES = ("auto", "pose", "odom", "image", "image_odom", "none")
Pose = Tuple[float, float, float]


@dataclass
class GoalNode:
    image: Image.Image
    pose: Optional[Pose] = None   # topomap を作った bag の座標 (frame)
    s: Optional[float] = None     # スタートからの道のり [m]
    path: str = ""


@dataclass
class Topomap:
    nodes: List[GoalNode]
    frame: str = "odom"
    start: Optional[Pose] = None
    meta: dict = field(default_factory=dict)
    directory: str = ""

    def __len__(self) -> int:
        return len(self.nodes)

    @property
    def spacing(self) -> Optional[float]:
        return self.meta.get("spacing_m")

    def has_poses(self) -> bool:
        return all(n.pose is not None for n in self.nodes)


def _numeric_key(path: str):
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"^(\d+)$", stem)
    return (0, int(m.group(1)), stem) if m else (1, 0, stem)


def _pose(entry) -> Optional[Pose]:
    if entry is None:
        return None
    if isinstance(entry, dict):
        if "x" not in entry:
            return None
        return float(entry["x"]), float(entry["y"]), float(entry.get("yaw", 0.0))
    vals = list(entry)
    return float(vals[0]), float(vals[1]), float(vals[2]) if len(vals) > 2 else 0.0


def topomap_index(path: str) -> dict:
    """topomap の中身の一覧 (画像は読まない). C++ ノードの設定 (tools/nav_config_json.py) にも使う.
    {"directory", "frame", "start": (x,y,yaw)|None, "meta", "nodes": [{"path", "pose": (x,y,yaw)|None, "s"}]}"""
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(path):
        return {"directory": os.path.dirname(path), "frame": "odom", "start": None, "meta": {},
                "nodes": [{"path": path, "pose": None, "s": None}]}
    if not os.path.isdir(path):
        raise FileNotFoundError(f"topomap not found: {path}")
    info: Dict[str, dict] = {}
    frame, start = "odom", None
    pf = os.path.join(path, "poses.yaml")
    if os.path.exists(pf):
        with open(pf) as f:
            data = yaml.safe_load(f) or {}
        frame = data.get("frame", frame)
        start = _pose(data.get("start"))
        for item in data.get("nodes", []):
            info[str(item["image"])] = item
    if info:   # poses.yaml があればそこに並んだ画像だけ (overview.png などは使わない)
        images = [os.path.join(path, name) for name in info]
        missing = [p for p in images if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"images listed in poses.yaml are missing: {missing[:3]}")
    else:      # 画像だけのディレクトリ: 0.jpg, 1.jpg, ... (名前が数字のもの) を番号順に
        images = sorted([os.path.join(path, f) for f in os.listdir(path)
                         if f.lower().endswith(IMAGE_EXTS) and os.path.splitext(f)[0].isdigit()], key=_numeric_key)
    if not images:
        raise FileNotFoundError(f"no subgoal images (0.jpg, 1.jpg, ...) in {path}")
    meta = {}
    mf = os.path.join(path, "topomap.json")
    if os.path.exists(mf):
        with open(mf) as f:
            meta = json.load(f)
    nodes = []
    for p in images:
        it = info.get(os.path.basename(p), {})
        nodes.append({"path": p, "pose": _pose(it), "s": it.get("s")})
    if start is None and nodes and nodes[0]["pose"] is not None:
        start = nodes[0]["pose"]
    return {"directory": path, "frame": frame, "start": start, "meta": meta, "nodes": nodes}


def load_topomap(path: str) -> Topomap:
    """ディレクトリ (topomap) か画像 1 枚 (ゴール 1 つ) を読む."""
    idx = topomap_index(path)
    nodes = [GoalNode(Image.open(n["path"]).convert("RGB"), n["pose"], n["s"], n["path"]) for n in idx["nodes"]]
    return Topomap(nodes, idx["frame"], idx["start"], idx["meta"], idx["directory"])


def write_topomap(out_dir: str, images: Sequence[Image.Image], poses: Sequence[Optional[Pose]],
                  s: Sequence[Optional[float]], frame: str = "odom", start: Optional[Pose] = None,
                  meta: Optional[dict] = None, overwrite: bool = False, jpeg_quality: int = 95) -> str:
    out_dir = os.path.abspath(os.path.expanduser(out_dir))
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        if not overwrite:
            raise FileExistsError(f"{out_dir} is not empty (use --overwrite)")
        for f in os.listdir(out_dir):
            if f.lower().endswith(IMAGE_EXTS) or f in ("poses.yaml", "topomap.json"):
                os.remove(os.path.join(out_dir, f))
    os.makedirs(out_dir, exist_ok=True)
    nodes = []
    for i, (img, pose, si) in enumerate(zip(images, poses, s)):
        name = f"{i}.jpg"
        img.convert("RGB").save(os.path.join(out_dir, name), quality=jpeg_quality)
        e = {"image": name}
        if pose is not None:
            e.update({"x": float(pose[0]), "y": float(pose[1]), "yaw": float(pose[2])})
        if si is not None:
            e["s"] = float(si)
        nodes.append(e)
    data = {"frame": frame, "nodes": nodes}
    if start is not None:
        data["start"] = {"x": float(start[0]), "y": float(start[1]), "yaw": float(start[2])}
    with open(os.path.join(out_dir, "poses.yaml"), "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    m = dict(meta or {})
    m.update({"num_nodes": len(nodes), "frame": frame, "created_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    with open(os.path.join(out_dir, "topomap.json"), "w") as f:
        json.dump(m, f, indent=1)
    return out_dir


def compose(a: Pose, b: Pose) -> Pose:
    """a の座標系で表した b をワールドへ (a ∘ b)."""
    c, s_ = math.cos(a[2]), math.sin(a[2])
    return a[0] + c * b[0] - s_ * b[1], a[1] + s_ * b[0] + c * b[1], wrap_angle(a[2] + b[2])


def align_to_start(robot_odom: Pose, robot_odom_at_start: Pose, topomap_start: Pose) -> Pose:
    """走行開始時のロボット位置 = topomap の start とみなして、今のオドメトリ位置を topomap の座標へ."""
    rel = relative_pose(robot_odom_at_start, robot_odom)  # 開始位置から見た今の位置
    return compose(topomap_start, rel)


# ---------------------------------------------------------------------------
# サブゴールの切り替え
# ---------------------------------------------------------------------------
@dataclass
class TrackerConfig:
    reach_check: str = "auto"
    subgoal_radius: float = 0.5         # pose/odom: これ以内で到達 [m]
    goal_radius: float = 0.5            # 最終ゴール [m]
    reach_angle_deg: float = 30.0       # pose/odom: 途中のサブゴールは向きの差もこれ以内 (0 で距離のみ)
    pass_radius: float = 1.5            # pose/odom: この距離以内でサブゴールが真横より後ろなら通過扱い
    pass_angle_deg: float = 90.0
    image_threshold: float = 0.80       # image: これ以上似ていたら「そのノードに着いた」
    goal_image_threshold: float = 0.85  # image: 最終ゴールの判定 (誤って止まらないよう高め)
    search_window: int = 2              # image: 今のノードから何個先まで比較するか
    confirm: int = 2                    # image: 何回続けて同じ判定なら切り替えるか
    odom_gate_m: float = 2.5            # image_odom: オドメトリ上これより遠いノードには切り替えない
    odom_pass_m: float = 1.0            # image_odom: オドメトリ上ノードをこれだけ通り過ぎたら進める
    goal_odom_gate_m: float = 1.5       # image_odom: 最終ゴールはオドメトリ上これ以内でだけ画像で判定 (手前で止まらない)
    min_travel_m: float = 0.2           # image: 切り替えてからこれだけ進むまで次に切り替えない (odom がある場合)


def resolve_mode(mode: str, has_localization: bool, has_odom: bool, topomap: Topomap) -> str:
    if mode not in REACH_MODES:
        raise ValueError(f"reach_check must be one of {REACH_MODES}")
    if mode != "auto":
        return mode
    if has_localization and topomap.frame == "map" and topomap.has_poses():
        return "pose"
    if has_odom and topomap.has_poses() and topomap.start is not None:
        return "image_odom"
    return "image"


class SubgoalTracker:
    def __init__(self, topomap: Topomap, cfg: TrackerConfig, mode: str):
        if not topomap.nodes:
            raise ValueError("empty topomap")
        if mode in ("pose", "odom", "image_odom") and not topomap.has_poses():
            raise ValueError(f"reach_check={mode} needs node poses in poses.yaml")
        self.map = topomap
        self.cfg = cfg
        self.mode = mode
        self.index = 0
        self.done = False
        self.last_reason: Optional[str] = None
        self.last_similarity: Optional[float] = None
        self.last_distance: Optional[float] = None
        self.similarities: Dict[int, float] = {}
        self._pending: Optional[int] = None
        self._pending_count = 0
        self._travel_at_switch = 0.0

    @property
    def nodes(self) -> List[GoalNode]:
        return self.map.nodes

    @property
    def current(self) -> GoalNode:
        return self.nodes[self.index]

    @property
    def is_final(self) -> bool:
        return self.index == len(self.nodes) - 1

    def _advance(self, to_index: int, reason: str, travel: Optional[float]) -> bool:
        self.last_reason = reason
        self._pending, self._pending_count = None, 0
        if travel is not None:
            self._travel_at_switch = travel
        if to_index >= len(self.nodes):
            self.index = len(self.nodes) - 1
            self.done = True
        else:
            self.index = to_index
        return True

    # -- pose / odom --------------------------------------------------------
    def _pose_check(self, pose: Pose, travel: Optional[float]) -> bool:
        c = self.cfg
        last = len(self.nodes) - 1
        for j in range(min(last, self.index + 1), self.index - 1, -1):  # 1 つ先まで (飛ばし対応)
            node = self.nodes[j]
            rel = relative_pose(pose, node.pose)
            d = math.hypot(rel[0], rel[1])
            if j == self.index:
                self.last_distance = d
            final = j == last
            if d < (c.goal_radius if final else c.subgoal_radius):
                dyaw = abs(wrap_angle(pose[2] - node.pose[2]))
                if final or c.reach_angle_deg <= 0 or dyaw <= math.radians(c.reach_angle_deg):
                    return self._advance(j + 1, f"within radius (d={d:.2f}m, dyaw={math.degrees(dyaw):.0f}deg)",
                                         travel)
            if not final and d < c.pass_radius and abs(math.atan2(rel[1], rel[0])) > math.radians(c.pass_angle_deg):
                return self._advance(j + 1, f"passed (d={d:.2f}m)", travel)
        return False

    # -- image --------------------------------------------------------------
    def _image_check(self, sim_fn: Callable[[int], float], course_pose: Optional[Pose],
                     travel: Optional[float]) -> bool:
        c = self.cfg
        last = len(self.nodes) - 1
        hi = min(last, self.index + max(0, c.search_window))
        sims = {j: float(sim_fn(j)) for j in range(self.index, hi + 1)}
        self.similarities = sims
        self.last_similarity = sims[self.index]
        if course_pose is not None:
            self.last_distance = math.hypot(*relative_pose(course_pose, self.current.pose)[:2])
        # オドメトリで明らかに遠いノードは候補から外す
        cand = dict(sims)
        if self.mode == "image_odom" and course_pose is not None:
            for j in list(cand):
                gate = min(c.odom_gate_m, c.goal_odom_gate_m) if j == last else c.odom_gate_m
                if math.hypot(*relative_pose(course_pose, self.nodes[j].pose)[:2]) > gate:
                    cand.pop(j)
        best = max(cand, key=cand.get) if cand else None
        thr = lambda j: c.goal_image_threshold if j == last else c.image_threshold  # noqa: E731
        if best is not None and cand[best] >= thr(best):
            moved = travel is None or travel - self._travel_at_switch >= c.min_travel_m or best > self.index
            if moved:
                self._pending_count = self._pending_count + 1 if self._pending == best else 1
                self._pending = best
                if self._pending_count >= max(1, c.confirm):
                    return self._advance(best + 1, f"image similarity {cand[best]:.3f} (node {best})", travel)
                return False
        else:
            self._pending, self._pending_count = None, 0
        # 画像で判定できないまま、オドメトリ上で今のノードを通り過ぎた (最終ゴールなら止まる)
        if self.mode == "image_odom" and course_pose is not None:
            rel = relative_pose(course_pose, self.current.pose)
            if rel[0] < -c.odom_pass_m:
                return self._advance(self.index + 1, f"passed by odometry ({-rel[0]:.2f}m behind)", travel)
        return False

    # -- main ---------------------------------------------------------------
    def update(self, pose: Optional[Pose] = None, sim_fn: Optional[Callable[[int], float]] = None,
               travel: Optional[float] = None) -> bool:
        """pose: pose/odom/image_odom では topomap と同じ座標の位置. sim_fn(j): 現在画像とノード j の類似度.
        travel: オドメトリ上の累積走行距離 [m] (任意). 進んだら True."""
        if self.done or self.mode == "none":
            return False
        if self.mode in ("pose", "odom"):
            if pose is None:
                return False
            return self._pose_check(pose, travel)
        if sim_fn is None:
            return False
        return self._image_check(sim_fn, pose if self.mode == "image_odom" else None, travel)
