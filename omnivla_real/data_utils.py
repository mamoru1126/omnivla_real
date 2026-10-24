"""学習データ作成に関する numpy/PIL のみの処理 (torch 非依存なので単体テスト可能).

OmniVLA (GNM_Dataset / 公式 inference) と同じ表現を使う:
  actions   : (8, 4) = [x/s, y/s, cos(dyaw), sin(dyaw)]   (s = metric_waypoint_spacing [m])
  goal_pose : (4,)   = [x/s, y/s, cos(dyaw), sin(dyaw)]   (距離 max_goal_dist でクリップ後に正規化)
  ロボット座標は x=前, y=左。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageEnhance

from .geometry import to_local

# OmniVLA の modality id (modeling_prismatic.py の _build_multimodal_attention_MMN と同じ定義)
MODALITY_IDS: Dict[str, int] = {
    "satellite": 0,
    "pose_satellite": 1,
    "satellite_image": 2,
    "all": 3,
    "pose": 4,
    "image_pose": 5,
    "image": 6,
    "language": 7,
    "language_pose": 8,
}
MODALITY_NAMES: Dict[int, str] = {v: k for k, v in MODALITY_IDS.items()}
POSE_MODALITIES = {1, 3, 4, 5, 8}
IMAGE_MODALITIES = {2, 3, 5, 6}
LANGUAGE_MODALITIES = {7, 8}
# 本リポジトリで扱う modality (衛星画像は使わない)
SUPPORTED_MODALITIES = {4, 5, 6, 7, 8}

NUM_ACTIONS_CHUNK = 8  # prismatic/vla/constants.py と同じ
ACTION_DIM = 4


def modality_id(name_or_id) -> int:
    if isinstance(name_or_id, (int, np.integer)):
        mid = int(name_or_id)
    else:
        key = str(name_or_id).strip().lower()
        if key.isdigit():
            mid = int(key)
        elif key in MODALITY_IDS:
            mid = MODALITY_IDS[key]
        else:
            raise ValueError(f"unknown modality '{name_or_id}'. choose from {sorted(MODALITY_IDS)}")
    if mid not in MODALITY_NAMES:
        raise ValueError(f"invalid modality id {mid}")
    return mid


# ---------------------------------------------------------------------------
# 目標・行動の計算
# ---------------------------------------------------------------------------
def normalize_goal_pose(x: float, y: float, dyaw: float, metric_spacing: float,
                        max_goal_dist: float = 30.0) -> np.ndarray:
    """ロボット座標の相対ゴール (m, m, rad) を OmniVLA の goal_pose 表現に変換.

    公式 run_omnivla.py と同じく、距離が max_goal_dist (thres_dist=30m) を超える場合は
    方向を保ったまま max_goal_dist に縮めてから metric_spacing で割る。
    """
    r = float(np.hypot(x, y))
    if r > max_goal_dist > 0:
        x *= max_goal_dist / r
        y *= max_goal_dist / r
    return np.array([x / metric_spacing, y / metric_spacing, np.cos(dyaw), np.sin(dyaw)], dtype=np.float32)


def future_indices(t: int, n: int, len_pred: int = NUM_ACTIONS_CHUNK, spacing: int = 1) -> np.ndarray:
    idx = t + spacing * np.arange(1, len_pred + 1)
    return np.minimum(idx, n - 1)  # 軌跡末尾を超える分は最終姿勢でパディング (= 停止)


def compute_action_targets(positions: np.ndarray, yaws: np.ndarray, t: int,
                           len_pred: int = NUM_ACTIONS_CHUNK, spacing: int = 1,
                           metric_spacing: float = 0.1) -> np.ndarray:
    """時刻 t から見た未来 len_pred 点の waypoint (正規化済み) を返す. shape (len_pred, 4)."""
    positions = np.asarray(positions, dtype=np.float64)
    yaws = np.asarray(yaws, dtype=np.float64).reshape(-1)
    idx = future_indices(t, len(positions), len_pred, spacing)
    local = to_local(positions[idx], positions[t], yaws[t]) / metric_spacing
    dyaw = yaws[idx] - yaws[t]
    return np.concatenate([local, np.cos(dyaw)[:, None], np.sin(dyaw)[:, None]], axis=1).astype(np.float32)


def compute_goal_pose(positions: np.ndarray, yaws: np.ndarray, t: int, goal_t: int,
                      metric_spacing: float = 0.1, max_goal_dist: float = 30.0) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    yaws = np.asarray(yaws, dtype=np.float64).reshape(-1)
    local = to_local(positions[goal_t], positions[t], yaws[t])
    return normalize_goal_pose(float(local[0]), float(local[1]), float(yaws[goal_t] - yaws[t]),
                               metric_spacing, max_goal_dist)


def denormalize_actions(actions_norm: np.ndarray, metric_spacing: float) -> np.ndarray:
    """(8,4) 正規化 waypoint -> [x[m], y[m], cos, sin] (cos/sin は単位ベクトルに正規化)."""
    a = np.array(actions_norm, dtype=np.float64, copy=True)
    a[:, :2] *= metric_spacing
    norm = np.linalg.norm(a[:, 2:4], axis=1, keepdims=True)
    a[:, 2:4] = np.where(norm > 1e-6, a[:, 2:4] / np.maximum(norm, 1e-6), np.array([1.0, 0.0]))
    return a


def rotate_labels(actions: np.ndarray, goal_pose: np.ndarray, dyaw: float) -> Tuple[np.ndarray, np.ndarray]:
    """ロボットの向きが dyaw [rad] (左が正) だけずれていたとしたときの正解 (同じ位置・同じ未来の経路を、
    ずれた向きのロボット座標で見たもの). 位置は -dyaw 回転、向きは -dyaw."""
    actions = np.array(actions, dtype=np.float32, copy=True)
    goal_pose = np.array(goal_pose, dtype=np.float32, copy=True)
    c, s = math.cos(dyaw), math.sin(dyaw)
    for a in (actions.reshape(-1, 4), goal_pose.reshape(-1, 4)):
        x, y = a[:, 0].copy(), a[:, 1].copy()
        a[:, 0], a[:, 1] = c * x + s * y, -s * x + c * y
        co, si = a[:, 2].copy(), a[:, 3].copy()
        a[:, 2], a[:, 3] = co * c + si * s, si * c - co * s   # cos/sin(θ - dyaw)
    return actions, goal_pose


def flip_left_right(actions: np.ndarray, goal_pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """画像の左右反転に合わせて y と sin(yaw) の符号を反転 (公式 dataset と同じ)."""
    actions = np.array(actions, copy=True)
    goal_pose = np.array(goal_pose, copy=True)
    actions[:, 1] *= -1.0
    actions[:, 3] *= -1.0
    goal_pose[1] *= -1.0
    goal_pose[3] *= -1.0
    return actions, goal_pose


# ---------------------------------------------------------------------------
# サンプリング
# ---------------------------------------------------------------------------
@dataclass
class GoalSamplingConfig:
    # ゴールまでのフレーム数 (3Hz で記録. 1 フレームの移動量は速度による)
    image_goal_offset: Tuple[int, int] = (2, 30)    # 画像のみ (modality 6) の場合
    pose_goal_offset: Tuple[int, int] = (2, 300)    # ポーズを含む modality の場合
    # modality id -> 重み
    modality_weights: Dict[int, float] = field(default_factory=lambda: {6: 0.5, 5: 0.25, 4: 0.25})
    # 画像のみの場合、ゴールを「道のりで何 m 先」で選ぶ (lo, hi). None ならフレーム数 (image_goal_offset).
    # 走行中のサブゴール (topomap, 1m 間隔) は 0.3〜1.5m 先なので、速さの違う bag を混ぜても同じ距離になるようにする
    image_goal_dist_m: Optional[Tuple[float, float]] = None


def build_sample_index(traj_lengths: Sequence[int], min_len: int = 2) -> List[Tuple[int, int]]:
    """(traj_idx, t) の一覧. t は少なくとも 1 フレーム未来がある位置."""
    index = []
    for i, n in enumerate(traj_lengths):
        if n < max(min_len, 2):
            continue
        index.extend((i, t) for t in range(0, n - 1))
    return index


def label_valid_mask(perturbed: Sequence[bool], len_pred: int = NUM_ACTIONS_CHUNK, spacing: int = 1) -> np.ndarray:
    """時刻 t を学習サンプルの「現在」に使ってよいか.

    正解ラベル (t+1 .. t+len_pred の姿勢) の区間に外乱フレームが含まれると、ラベルがお手本の動きではなくなるので除外する。
    t 自身が外乱中でも、その先がお手本の動き (= 外乱からの立て直し) なら有効。これが「外れた状態から戻る」学習データになる。
    """
    p = np.asarray(perturbed, dtype=bool)
    n = len(p)
    valid = np.ones(n, dtype=bool)
    for t in range(n):
        fut = p[t + 1:min(n, t + spacing * len_pred + 1)]
        if fut.any():
            valid[t] = False
    return valid


def choose_modality(rng: np.random.Generator, weights: Dict[int, float]) -> int:
    ids = np.array(sorted(weights.keys()), dtype=np.int64)
    p = np.array([float(weights[int(i)]) for i in ids], dtype=np.float64)
    if np.any(p < 0) or p.sum() <= 0:
        raise ValueError(f"invalid modality weights: {weights}")
    unknown = set(int(i) for i in ids) - SUPPORTED_MODALITIES
    if unknown:
        raise ValueError(f"unsupported modality ids: {sorted(unknown)} (satellite modalities are not used)")
    return int(rng.choice(ids, p=p / p.sum()))


def arc_length(positions: np.ndarray) -> np.ndarray:
    """各フレームまでの道のり [m] (先頭 0)."""
    p = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
    if len(p) == 0:
        return np.zeros(0)
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])


def sample_goal_offset(rng: np.random.Generator, t: int, n: int, modality: int,
                       cfg: GoalSamplingConfig, arc: Optional[np.ndarray] = None) -> int:
    """modality に応じたゴールのフレームオフセット (>=1) を返す.
    arc (各フレームの道のり) と cfg.image_goal_dist_m があれば、画像のみのゴールは道のりで選ぶ。"""
    remaining = n - 1 - t
    if remaining < 1:
        raise ValueError("t must have at least one future frame")
    if modality == 6 and cfg.image_goal_dist_m and arc is not None:
        lo, hi = float(cfg.image_goal_dist_m[0]), float(cfg.image_goal_dist_m[1])
        d = np.asarray(arc[t + 1:n], dtype=np.float64) - float(arc[t])
        cand = np.nonzero((d >= lo) & (d <= hi))[0]
        if cand.size:
            return int(cand[rng.integers(cand.size)]) + 1
        within = np.nonzero(d <= hi)[0]   # 止まっている・軌跡の終わりが近い: hi 以内で一番先
        return int(within[-1]) + 1 if within.size else 1
    if modality == 6:  # 画像のみ: 近距離ゴール
        lo, hi = cfg.image_goal_offset
    else:
        lo, hi = cfg.pose_goal_offset
    hi_eff = max(1, min(int(hi), remaining))
    lo_eff = max(1, min(int(lo), hi_eff))
    return int(rng.integers(lo_eff, hi_eff + 1))


# ---------------------------------------------------------------------------
# 画像 augmentation (PIL)
# ---------------------------------------------------------------------------
def random_crop_box(rng: Optional[np.random.Generator], width: int, height: int,
                    max_v_frac: float = 0.2, max_h_frac: float = 0.1,
                    u: Optional[Tuple[float, float]] = None) -> Tuple[int, int, int, int]:
    """公式 LeLaN/Dummy dataset と同じ「上下左右対称に縁を削る」クロップ (画像中心は不変).

    片側あたり最大 max_v_frac*H, max_h_frac*W を削る (公式の v_random=0.2, h_random=0.1 と同じ意味)。
    公式実装は 224x224 前提で座標を決め打ちしているが、ここでは画像サイズに比例させる。
    u=(uv, uh) を与えると乱数の代わりに使う (現在画像とゴール画像で同じクロップをするため)。
    """
    uv, uh = u if u is not None else (rng.random(), rng.random())
    voff = int(height * max_v_frac * uv)
    hoff = int(width * max_h_frac * uh)
    voff = min(voff, (height - 2) // 2)
    hoff = min(hoff, (width - 2) // 2)
    return hoff, voff, width - hoff, height - voff


def color_jitter(img: Image.Image, rng: np.random.Generator, strength: float = 0.2) -> Image.Image:
    """明るさ・コントラスト・彩度を ±strength の範囲でランダムに変える."""
    if strength <= 0:
        return img
    for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        factor = float(1.0 + rng.uniform(-strength, strength))
        img = enhancer(img).enhance(factor)
    return img


def resize_naive(img: Image.Image, size: int = 224) -> Image.Image:
    """OmniVLA の preprocessor (image_resize_strategy=resize-naive) と同じくアスペクト比を無視して縮小."""
    return img.resize((size, size), Image.BICUBIC)


@dataclass
class AugmentConfig:
    enabled: bool = True
    crop_v: float = 0.1        # 片側あたりの最大クロップ率 (縦)
    crop_h: float = 0.05       # 片側あたりの最大クロップ率 (横)
    flip_prob: float = 0.5     # 左右反転 (行動・ゴールの y と sin も反転)
    color_jitter: float = 0.2  # 明るさ/コントラスト/彩度の変動幅
    image_size: int = 224      # OmniVLA の入力解像度 (固定)
    # 向きのずれの水増し: 現在画像だけを左右にずらし (= カメラを最大 yaw_deg 回したのと同じ見え方)、
    # 正解をそのずれから元の経路に戻る向きに回す。「向きがずれたら戻る」(コース外からの立て直し) を教える。
    # hfov_deg はカメラの実際の水平画角 (robot.yaml の camera.hfov_deg. 画像のずらし量 <-> 角度の換算に使う)
    yaw_deg: float = 0.0
    hfov_deg: float = 90.0


def make_targets(rng: np.random.Generator, positions: np.ndarray, yaws: np.ndarray, t: int,
                 goal_cfg: GoalSamplingConfig, metric_spacing: float, waypoint_spacing: int = 1,
                 max_goal_dist: float = 30.0, force_modality: Optional[int] = None,
                 arc: Optional[np.ndarray] = None):
    """1 サンプル分の (modality, goal_t, actions(8,4), goal_pose(4,)) を作る (hindsight relabeling)."""
    n = len(positions)
    mod = int(force_modality) if force_modality is not None else choose_modality(rng, goal_cfg.modality_weights)
    if arc is None and goal_cfg.image_goal_dist_m:
        arc = arc_length(positions)
    goal_t = t + sample_goal_offset(rng, t, n, mod, goal_cfg, arc)
    actions = compute_action_targets(positions, yaws, t, NUM_ACTIONS_CHUNK, waypoint_spacing, metric_spacing)
    goal_pose = compute_goal_pose(positions, yaws, t, goal_t, metric_spacing, max_goal_dist)
    return mod, goal_t, actions, goal_pose


def yaw_shift_boxes(width: int, height: int, box: Tuple[int, int, int, int], yaw: float, hfov: float):
    """切り抜き box (左, 上, 右, 下. 左右対称) を左右に動かして、カメラを yaw [rad] (左が正) 回したのと
    ほぼ同じ見え方にする (小さい角度なら回転 = 横ずらし). 動かす余白が足りなければ窓を左右とも細くする.

    戻り値: (現在画像の box, ゴール画像の box (同じ大きさで中央), 実際にずらした角度 [rad])
    """
    left, top, right, bottom = box
    f = (width / 2.0) / math.tan(hfov / 2.0)          # 焦点距離 [px]
    dx = int(round(-f * math.tan(yaw)))                 # 左を向く = 窓を左へ
    margin = min(left, width - right)
    if abs(dx) > margin:
        extra = abs(dx) - margin
        if right - left - 2 * extra < max(8, width // 4):   # 細くなりすぎる: ずらせる所まで
            extra = max(0, (right - left - max(8, width // 4)) // 2)
            dx = int(np.sign(dx)) * (margin + extra)
        left, right = left + extra, right - extra
    cur = (left + dx, top, right + dx, bottom)
    return cur, (left, top, right, bottom), math.atan(-dx / f)


def augment_pair(rng: np.random.Generator, cur: Image.Image, goal: Image.Image, actions: np.ndarray,
                 goal_pose: np.ndarray, aug: AugmentConfig, train: bool):
    """クロップ (両画像同じ. yaw_deg>0 なら現在画像だけ左右にずらして正解を回す) -> 224x224 に縮小
    -> 左右反転 (両画像+ラベル) -> 色変動 (画像ごと)."""
    if train and aug.enabled:
        u = (rng.random(), rng.random())
        box_c = random_crop_box(None, cur.width, cur.height, aug.crop_v, aug.crop_h, u)
        box_g = random_crop_box(None, goal.width, goal.height, aug.crop_v, aug.crop_h, u)
        if aug.yaw_deg > 0:
            want = math.radians(float(rng.uniform(-aug.yaw_deg, aug.yaw_deg)))
            box_c, same_g, applied = yaw_shift_boxes(cur.width, cur.height, box_c, want, math.radians(aug.hfov_deg))
            if goal.size == cur.size:
                box_g = same_g
            actions, goal_pose = rotate_labels(actions, goal_pose, applied)
        cur = cur.crop(box_c)
        goal = goal.crop(box_g)
    cur = resize_naive(cur, aug.image_size)
    goal = resize_naive(goal, aug.image_size)
    if train and aug.enabled:
        if rng.random() < aug.flip_prob:
            cur = cur.transpose(Image.FLIP_LEFT_RIGHT)
            goal = goal.transpose(Image.FLIP_LEFT_RIGHT)
            actions, goal_pose = flip_left_right(actions, goal_pose)
        cur = color_jitter(cur, rng, aug.color_jitter)
        goal = color_jitter(goal, rng, aug.color_jitter)
    return cur, goal, actions, goal_pose


def turn_flags(positions: np.ndarray, yaws: np.ndarray, horizon: int = 10, threshold_deg: float = 20.0) -> np.ndarray:
    """各フレーム t について「この先 horizon フレーム以内に曲がるか」を返す (bool, 長さ N).

    次のどちらかが threshold_deg を超えたら「曲がる」:
      * 向きの変化 |yaw[t+k] - yaw[t]| (k <= horizon)
      * horizon フレーム先の位置がどれだけ横にあるか (ロボット座標での方位角)
    """
    positions = np.asarray(positions, dtype=np.float64)
    yaws = np.asarray(yaws, dtype=np.float64).reshape(-1)
    n = len(yaws)
    thr = math.radians(threshold_deg)
    out = np.zeros(n, dtype=bool)
    for t in range(n):
        hi = min(n - 1, t + horizon)
        if hi <= t:
            continue
        dyaw = np.abs((yaws[t + 1:hi + 1] - yaws[t] + np.pi) % (2 * np.pi) - np.pi)
        if dyaw.size and dyaw.max() > thr:
            out[t] = True
            continue
        local = to_local(positions[hi], positions[t], yaws[t])
        if np.hypot(local[0], local[1]) > 0.2 and abs(math.atan2(local[1], local[0])) > thr:
            out[t] = True
    return out


def balanced_weights(flags: Sequence[bool], ratio: float) -> np.ndarray:
    """flags=True のサンプルが全体の ratio の割合で引かれるようなサンプリング重み (和 = 1)."""
    flags = np.asarray(flags, dtype=bool)
    n_pos, n_neg = int(flags.sum()), int((~flags).sum())
    if ratio <= 0 or n_pos == 0 or n_neg == 0:
        return np.full(len(flags), 1.0 / max(1, len(flags)))
    ratio = min(max(float(ratio), 0.0), 1.0)
    w = np.where(flags, ratio / n_pos, (1.0 - ratio) / n_neg)
    return w / w.sum()


def reweight(weights: Sequence[float], flags: Sequence[bool], ratio: float) -> np.ndarray:
    """既存のサンプリング重みを、flags=True のサンプルの合計が ratio になるように掛け直す (和 = 1).

    flags=True の合計が既に ratio 以上なら何もしない (少ないときだけ増やす)。
    """
    w = np.asarray(weights, dtype=np.float64).copy()
    flags = np.asarray(flags, dtype=bool)
    w = w / max(w.sum(), 1e-12)
    m_pos = float(w[flags].sum())
    if ratio <= 0 or m_pos <= 0 or m_pos >= 1 or m_pos >= ratio:
        return w
    ratio = min(float(ratio), 1.0)
    w = np.where(flags, w * ratio / m_pos, w * (1.0 - ratio) / (1.0 - m_pos))
    return w / w.sum()


def cut_end_mask(n: int, len_pred: int = NUM_ACTIONS_CHUNK, spacing: int = 1) -> np.ndarray:
    """区切っただけの軌跡の末尾で、正解 (この先 len_pred 点) が軌跡の外にはみ出すフレーム (True = 使わない)."""
    out = np.zeros(int(n), dtype=bool)
    k = int(len_pred) * max(1, int(spacing))
    out[max(0, int(n) - k):] = True
    return out


def smooth_trajectory(positions: np.ndarray, yaws: np.ndarray, sigma_frames: float) -> Tuple[np.ndarray, np.ndarray]:
    """位置と向きを時間方向にガウスでならす (sigma_frames [フレーム]).

    手動操作で「スティックを短く倒す」を繰り返した曲がり方は、いつ倒すかが画像から決まらないので、そのまま
    正解にするとモデルは平均 (遅くて弱い曲がり) を覚える。ならして「なめらかに曲がる経路」を正解にする。
    端は点対称に折り返す (端での速度を保つ). 向きは unwrap してからならす (返す向きも unwrap のまま)."""
    p = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
    y = np.unwrap(np.asarray(yaws, dtype=np.float64).reshape(-1))
    n = len(y)
    if sigma_frames <= 0 or n < 3:
        return p.copy(), y
    m = min(int(math.ceil(3.0 * sigma_frames)), n - 1)
    k = np.exp(-0.5 * (np.arange(-m, m + 1) / float(sigma_frames)) ** 2)
    k /= k.sum()

    def sm(a: np.ndarray) -> np.ndarray:
        pad = np.concatenate([2 * a[0] - a[m:0:-1], a, 2 * a[-1] - a[-2:-m - 2:-1]])
        return np.convolve(pad, k, mode="valid")

    return np.stack([sm(p[:, 0]), sm(p[:, 1])], axis=1), sm(y)


def summarize_spacing(positions: np.ndarray) -> Optional[float]:
    """隣接フレーム間の平均移動量 [m] (停止中フレームを除く). metric_waypoint_spacing の目安."""
    positions = np.asarray(positions, dtype=np.float64)
    if len(positions) < 2:
        return None
    d = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    moving = d[d > 1e-3]
    return float(moving.mean()) if len(moving) else None
