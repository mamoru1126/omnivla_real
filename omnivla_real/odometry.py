"""位置の時系列 (オドメトリ・指令値の積算・自己位置) の補間と比較 (numpy のみ).

学習ラベル (この先 8 点の位置と向き) は「ロボットが実際にどう動いたか」から作る。その元として
  odom          : ホイールオドメトリの位置 (nav_msgs/Odometry の pose)
  odom_twist    : ホイールオドメトリの速度を積算 (pose が無い/おかしいロボット向け)
  cmd           : 指示値 (cmd_vel) を積算 (オドメトリが無い/信用できない場合. 遅れ cmd_delay を考慮)
  localization  : 自己位置 (任意)
のどれでも使えるようにしている。ラベルに使うのは 2〜3 秒先までの相対的な動きだけなので、
長時間のドリフトは問題にならないが、短時間のずれ (スリップ・指令と実際の差) は効く。
check_odometry.py で比較できる。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .geometry import to_local, wrap_angle


@dataclass
class PoseTrack:
    t: np.ndarray      # (N,) [s] 昇順
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray    # unwrap 済み

    @staticmethod
    def from_arrays(t, x, y, yaw) -> "PoseTrack":
        t = np.asarray(t, dtype=np.float64)
        order = np.argsort(t, kind="stable")
        t = t[order]
        keep = np.concatenate([[True], np.diff(t) > 0]) if len(t) else np.zeros(0, bool)
        sel = order[keep]
        return PoseTrack(t[keep], np.asarray(x, np.float64)[sel], np.asarray(y, np.float64)[sel],
                         np.unwrap(np.asarray(yaw, np.float64)[sel]))

    def __len__(self) -> int:
        return len(self.t)

    def at(self, tq, max_gap: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """時刻 tq の (x, y, yaw, valid). 範囲外や前後のサンプルが max_gap 以上離れていれば valid=False."""
        tq = np.atleast_1d(np.asarray(tq, dtype=np.float64))
        n = len(self.t)
        if n < 2:
            z = np.zeros(len(tq))
            return z, z, z, np.zeros(len(tq), bool)
        x = np.interp(tq, self.t, self.x)
        y = np.interp(tq, self.t, self.y)
        yaw = np.interp(tq, self.t, self.yaw)
        idx = np.clip(np.searchsorted(self.t, tq), 1, n - 1)
        gap = self.t[idx] - self.t[idx - 1]
        valid = (tq >= self.t[0] - 1e-6) & (tq <= self.t[-1] + 1e-6) & (gap <= max_gap)
        return x, y, wrap_angle(yaw), valid

    def speed(self) -> Tuple[np.ndarray, np.ndarray]:
        """各サンプル区間の (前進速度 [m/s], 角速度 [rad/s]) (長さ N-1)."""
        dt = np.maximum(np.diff(self.t), 1e-6)
        dx, dy = np.diff(self.x), np.diff(self.y)
        fwd = dx * np.cos(self.yaw[:-1]) + dy * np.sin(self.yaw[:-1])
        return fwd / dt, np.diff(self.yaw) / dt


def zero_order_hold(t_src: np.ndarray, values: np.ndarray, tq: np.ndarray, timeout: float = 0.5,
                    default: float = 0.0) -> np.ndarray:
    """最新の値を保持. 最後のサンプルから timeout 秒以上経ったら default (指令が途切れたら停止扱い)."""
    t_src = np.asarray(t_src, np.float64)
    values = np.asarray(values, np.float64)
    tq = np.asarray(tq, np.float64)
    out = np.full(tq.shape, default, dtype=np.float64)
    if len(t_src) == 0:
        return out
    idx = np.searchsorted(t_src, tq, side="right") - 1
    ok = idx >= 0
    age = np.where(ok, tq - t_src[np.clip(idx, 0, None)], np.inf)
    ok &= age <= timeout
    out[ok] = values[idx[ok]]
    return out


def integrate_twist(t: Sequence[float], v: Sequence[float], w: Sequence[float], delay: float = 0.0,
                    timeout: float = 0.5, dt: float = 0.01, pose0=(0.0, 0.0, 0.0)) -> PoseTrack:
    """速度 (v, w) の時系列を積算して位置にする (デッドレコニング). delay: 指令から動き出すまでの遅れ [s]."""
    t = np.asarray(t, np.float64) + float(delay)
    if len(t) < 2:
        return PoseTrack.from_arrays([], [], [], [])
    order = np.argsort(t, kind="stable")
    t, v, w = t[order], np.asarray(v, np.float64)[order], np.asarray(w, np.float64)[order]
    grid = np.arange(t[0], t[-1] + timeout, dt)
    vg = zero_order_hold(t, v, grid, timeout)
    wg = zero_order_hold(t, w, grid, timeout)
    yaw = pose0[2] + np.concatenate([[0.0], np.cumsum(wg[:-1] * dt)])
    mid = yaw[:-1] + 0.5 * wg[:-1] * dt                     # 中点法 (円弧に近い)
    x = pose0[0] + np.concatenate([[0.0], np.cumsum(vg[:-1] * np.cos(mid) * dt)])
    y = pose0[1] + np.concatenate([[0.0], np.cumsum(vg[:-1] * np.sin(mid) * dt)])
    return PoseTrack(grid, x, y, yaw)


def estimate_delay(t_cmd: np.ndarray, w_cmd: np.ndarray, t_meas: np.ndarray, w_meas: np.ndarray,
                   max_delay: float = 1.0, step: float = 0.02) -> Tuple[float, float]:
    """指令 (w_cmd) と実際 (w_meas) の相互相関が最大になる遅れ [s] と、その相関係数."""
    if len(t_cmd) < 10 or len(t_meas) < 10:
        return 0.0, 0.0
    t0, t1 = max(t_cmd[0], t_meas[0]), min(t_cmd[-1], t_meas[-1]) - max_delay
    if t1 - t0 < 5.0:
        return 0.0, 0.0
    grid = np.arange(t0, t1, 0.05)
    best = (0.0, -2.0)
    meas_interp = lambda tq: np.interp(tq, t_meas, w_meas)  # noqa: E731
    c = zero_order_hold(t_cmd, w_cmd, grid, timeout=1.0)
    if np.std(c) < 1e-6:
        return 0.0, 0.0
    for d in np.arange(0.0, max_delay + 1e-9, step):
        m = meas_interp(grid + d)
        if np.std(m) < 1e-6:
            continue
        r = float(np.corrcoef(c, m)[0, 1])
        if r > best[1]:
            best = (float(d), r)
    return best


def window_displacements(track: PoseTrack, t0: np.ndarray, horizon: float, max_gap: float = 0.5):
    """各 t0 から horizon 秒後までの相対移動 (ロボット座標 x前, y左, dyaw)."""
    xa, ya, ra, va = track.at(t0, max_gap)
    xb, yb, rb, vb = track.at(np.asarray(t0) + horizon, max_gap)
    out = np.zeros((len(t0), 3))
    for i in range(len(t0)):
        loc = to_local(np.array([xb[i], yb[i]]), (xa[i], ya[i]), ra[i])
        out[i] = (loc[0], loc[1], wrap_angle(rb[i] - ra[i]))
    return out, va & vb


def compare_tracks(a: PoseTrack, b: PoseTrack, t0: np.ndarray, horizon: float) -> Dict[str, float]:
    """2 つの位置の時系列の「horizon 秒間の動き」の差 (ラベルとして使えるかの目安)."""
    da, va = window_displacements(a, t0, horizon)
    db, vb = window_displacements(b, t0, horizon)
    ok = va & vb
    if not ok.any():
        return {"n": 0}
    pos_err = np.linalg.norm(da[ok, :2] - db[ok, :2], axis=1)
    yaw_err = np.abs(wrap_angle(da[ok, 2] - db[ok, 2]))
    dist_a = np.linalg.norm(da[ok, :2], axis=1)
    moving = dist_a > 0.1
    scale = float(np.median(np.linalg.norm(db[ok][moving, :2], axis=1) / dist_a[moving])) if moving.any() else math.nan
    return {
        "n": int(ok.sum()),
        "pos_err_median_m": float(np.median(pos_err)),
        "pos_err_p90_m": float(np.percentile(pos_err, 90)),
        "yaw_err_median_deg": float(np.degrees(np.median(yaw_err))),
        "yaw_err_p90_deg": float(np.degrees(np.percentile(yaw_err, 90))),
        "distance_ratio_b_over_a": scale,
        "mean_distance_a_m": float(dist_a.mean()),
    }
