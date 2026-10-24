"""実走の評価: 自律走行した時の自己位置 (地図の座標) を、コース (自己位置で作った topomap) と比べる.

自己位置は評価だけに使う (学習・走行には使わない)。コースはサブゴール画像列 (topomap) のノードの位置
(poses.yaml, frame: map) をつないだ折れ線。走行の各時刻について:
  s        コースに沿ってどこまで進んだか [m] (前回の近くだけを探すので、コースが近くを 2 回通っても飛ばない)
  xtrack   コースからの横ずれ [m] (左が正)
  heading  コースの向きとの差 [deg] (左が正)
曲がる所 (カーブ) ごとに、外に膨らんだ量 (outward) と内側に切り込んだ量 (inward) を出す。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geometry import wrap_angle


@dataclass
class CoursePath:
    x: np.ndarray
    y: np.ndarray
    s: np.ndarray            # 各ノードまでの道のり [m]
    heading: np.ndarray      # 各区間 (ノード j -> j+1) の向き [rad] (長さ N-1)

    @staticmethod
    def from_points(xy: np.ndarray) -> "CoursePath":
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        keep = [0]
        for k in range(1, len(xy)):                      # 同じ位置のノードは 1 つに (長さ 0 の区間を作らない)
            if math.hypot(*(xy[k] - xy[keep[-1]])) > 1e-6:
                keep.append(k)
        xy = xy[keep]
        if len(xy) < 2:
            raise ValueError("course needs at least 2 distinct points")
        d = np.hypot(*np.diff(xy, axis=0).T)
        s = np.concatenate([[0.0], np.cumsum(d)])
        h = np.arctan2(np.diff(xy[:, 1]), np.diff(xy[:, 0]))
        return CoursePath(xy[:, 0], xy[:, 1], s, h)

    @staticmethod
    def from_topomap(tm) -> "CoursePath":
        if not tm.has_poses():
            raise ValueError("the topomap has no node poses (poses.yaml)")
        if tm.frame != "map":
            raise ValueError(f"the topomap is in the '{tm.frame}' frame. Build it from a bag with localization "
                             "(robot.yaml topics.eval_localization) so its poses are in the map frame")
        return CoursePath.from_points(np.array([[n.pose[0], n.pose[1]] for n in tm.nodes]))

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def heading_at(self, s: np.ndarray) -> np.ndarray:
        """道のり s でのコースの向き (区間の向きを unwrap したもの)."""
        hu = np.unwrap(self.heading)
        j = np.clip(np.searchsorted(self.s, s, side="right") - 1, 0, len(hu) - 1)
        return hu[j]

    def corners(self, min_turn_deg: float = 20.0, window_m: float = 4.0, merge_m: float = 2.0) -> List[dict]:
        """window_m の間で向きが min_turn_deg 以上変わる所をカーブとして返す (道のり順).
        turn_deg: カーブ全体での向きの変化 (左が正)."""
        if self.length < window_m:
            return []
        grid = np.arange(0.0, self.length, 0.5)
        half = window_m / 2
        dh = self.heading_at(np.minimum(grid + half, self.length)) - self.heading_at(np.maximum(grid - half, 0.0))
        hot = np.abs(np.degrees(dh)) >= min_turn_deg
        spans: List[List[float]] = []
        for g, flag in zip(grid, hot):
            if not flag:
                continue
            if spans and g - spans[-1][1] <= merge_m:
                spans[-1][1] = g
            else:
                spans.append([g, g])
        out = []
        for a, b in spans:
            a0, b0 = max(0.0, a - half), min(self.length, b + half)
            turn = float(np.degrees(self.heading_at(np.array([b0]))[0] - self.heading_at(np.array([a0]))[0]))
            out.append({"s0": float(a0), "s1": float(b0), "turn_deg": turn})
        return out


def project_track(path: CoursePath, x: np.ndarray, y: np.ndarray, yaw: np.ndarray,
                  back_m: float = 5.0, ahead_m: float = 15.0) -> Dict[str, np.ndarray]:
    """各時刻の位置をコースに投影する. 戻り値: s, xtrack (左が正), heading_err [rad], dist (コースまでの距離).
    最初は全体から一番近い所、そのあとは前回の s の back_m 手前〜ahead_m 先だけを探す."""
    n = len(x)
    S = np.full(n, np.nan)
    X = np.full(n, np.nan)
    H = np.full(n, np.nan)
    D = np.full(n, np.nan)
    ax, ay = path.x[:-1], path.y[:-1]
    bx, by = path.x[1:], path.y[1:]
    seg = np.hypot(bx - ax, by - ay)
    last: Optional[float] = None
    for i in range(n):
        if not (np.isfinite(x[i]) and np.isfinite(y[i])):
            continue
        if last is None:
            js = np.arange(len(seg))
        else:
            js = np.nonzero((path.s[1:] >= last - back_m) & (path.s[:-1] <= last + ahead_m))[0]
            if js.size == 0:
                js = np.arange(len(seg))
        ux = ((x[i] - ax[js]) * (bx[js] - ax[js]) + (y[i] - ay[js]) * (by[js] - ay[js])) / (seg[js] ** 2)
        u = np.clip(ux, 0.0, 1.0)
        qx = ax[js] + u * (bx[js] - ax[js])
        qy = ay[js] + u * (by[js] - ay[js])
        d = np.hypot(x[i] - qx, y[i] - qy)
        k = int(np.argmin(d))
        j = int(js[k])
        cross = (bx[j] - ax[j]) * (y[i] - ay[j]) - (by[j] - ay[j]) * (x[i] - ax[j])
        S[i] = path.s[j] + u[k] * seg[j]
        X[i] = math.copysign(float(d[k]), cross) if d[k] > 0 else 0.0
        D[i] = float(d[k])
        if np.isfinite(yaw[i]):
            H[i] = float(wrap_angle(yaw[i] - path.heading[j]))
        last = S[i]
    return {"s": S, "xtrack": X, "heading_err": H, "dist": D}


def running_mask(t: np.ndarray, status: Sequence[Tuple[float, str]], hold_s: float = 2.0) -> Optional[np.ndarray]:
    """/omnivla/status (時刻, state) から、各時刻にナビが running だったか. 状態が無ければ None.
    状態は 1 秒ごとなので、最後の状態を hold_s 秒まで保持する."""
    if not status:
        return None
    ts = np.array([a for a, _ in status], dtype=np.float64)
    st = np.array([b == "running" for _, b in status], dtype=bool)
    idx = np.searchsorted(ts, t, side="right") - 1
    ok = idx >= 0
    age = np.where(ok, t - ts[np.clip(idx, 0, None)], np.inf)
    return ok & (age <= hold_s) & st[np.clip(idx, 0, None)]


def _stats(a: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
            "max": float(a.max())}


def evaluate_run(path: CoursePath, t: np.ndarray, x: np.ndarray, y: np.ndarray, yaw: np.ndarray,
                 running: Optional[np.ndarray] = None, speed_cap: Optional[float] = None,
                 min_turn_deg: float = 20.0) -> Tuple[dict, Dict[str, np.ndarray]]:
    """戻り値: (まとめ, 時刻ごとの値). running があれば「自律で走っていた時間」だけでも集計する."""
    t = np.asarray(t, dtype=np.float64)
    pr = project_track(path, x, y, yaw)
    sp = np.full(len(t), np.nan)
    if len(t) > 1:
        dt = np.maximum(np.gradient(t), 1e-3)
        sp = np.hypot(np.gradient(x), np.gradient(y)) / dt
    per = {"t": t, "x": np.asarray(x), "y": np.asarray(y), "yaw": np.asarray(yaw), "speed": sp, **pr}
    if running is not None:
        per["running"] = np.asarray(running, dtype=bool)
    ok = np.isfinite(pr["s"])
    out: dict = {"course_length_m": path.length, "duration_s": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
                 "driven_m": float(np.nansum(np.hypot(np.diff(x), np.diff(y)))) if len(t) > 1 else 0.0}
    if not ok.any():
        out["error"] = "no localization samples"
        return out, per
    s_ok = pr["s"][ok]
    out["progress"] = {"start_s": float(s_ok[0]), "max_s": float(np.max(s_ok)),
                       "completion": float((np.max(s_ok) - s_ok[0]) / max(path.length - s_ok[0], 1e-6))}

    def block(mask: np.ndarray) -> dict:
        m = mask & ok
        b = {"samples": int(m.sum()), "seconds": float(np.sum(np.gradient(t)[m])) if len(t) > 1 else 0.0,
             "abs_xtrack_m": _stats(np.abs(pr["xtrack"][m])),
             "abs_heading_err_deg": _stats(np.degrees(np.abs(pr["heading_err"][m]))),
             "off_course_0p5m_ratio": float(np.mean(np.abs(pr["xtrack"][m]) > 0.5)) if m.any() else None,
             "off_course_1m_ratio": float(np.mean(np.abs(pr["xtrack"][m]) > 1.0)) if m.any() else None}
        if m.any():
            k = int(np.nanargmax(np.where(m, np.abs(pr["xtrack"]), -1)))
            b["worst"] = {"t": float(t[k] - t[0]), "s": float(pr["s"][k]), "xtrack_m": float(pr["xtrack"][k])}
        if speed_cap is not None and m.any():
            b["faster_than_cap_ratio"] = float(np.mean(sp[m] > speed_cap + 0.1))
        return b

    out["all"] = block(np.ones(len(t), dtype=bool))
    if running is not None:
        out["running"] = block(per["running"])
    corners = []
    reached = float(np.max(s_ok))
    use = per.get("running", np.ones(len(t), dtype=bool)) & ok
    for c in path.corners(min_turn_deg):
        inside = use & (pr["s"] >= c["s0"]) & (pr["s"] <= c["s1"])
        d = dict(c)
        d["reached_end"] = bool(reached >= c["s1"])
        if inside.any():
            sign = 1.0 if c["turn_deg"] > 0 else -1.0       # 左に曲がる所は右 (xtrack 負) が外
            outward = -sign * pr["xtrack"][inside]
            d.update(samples=int(inside.sum()), max_outward_m=float(np.max(outward)),
                     max_inward_m=float(np.max(-outward)), mean_abs_xtrack_m=float(np.mean(np.abs(pr["xtrack"][inside]))))
        corners.append(d)
    out["corners"] = corners
    return out, per


def report_text(summary: dict, title: str = "") -> str:
    def f(v, u="", d=2):
        return "-" if v is None else f"{v:.{d}f}{u}"
    L = []
    if title:
        L.append(title)
    if "error" in summary:
        L.append(f"エラー: {summary['error']}")
        return "\n".join(L) + "\n"
    p = summary["progress"]
    L.append(f"コース {summary['course_length_m']:.1f} m のうち {p['start_s']:.1f} m 地点から {p['max_s']:.1f} m 地点まで "
             f"(進んだ割合 {p['completion'] * 100:.0f}%)   時間 {summary['duration_s']:.0f} 秒, 走った距離 {summary['driven_m']:.1f} m")
    for key, name in (("running", "自律で走っていた間 (/omnivla/status が running)"), ("all", "全体")):
        if key not in summary:
            continue
        b = summary[key]
        x, h = b["abs_xtrack_m"], b["abs_heading_err_deg"]
        L.append("")
        L.append(f"[{name}]  {b['seconds']:.0f} 秒")
        if x.get("n"):
            L.append(f"  コースからの横ずれ |m|: 中央値 {f(x['median'])}  95% {f(x['p95'])}  最大 {f(x['max'])}")
            L.append(f"  向きの差 |deg|: 中央値 {f(h.get('median'), d=1)}  95% {f(h.get('p95'), d=1)}")
            L.append(f"  0.5 m 以上ずれていた割合 {f((b['off_course_0p5m_ratio'] or 0) * 100, '%', 0)}  "
                     f"1 m 以上 {f((b['off_course_1m_ratio'] or 0) * 100, '%', 0)}")
            if b.get("worst"):
                w = b["worst"]
                L.append(f"  一番ずれた所: {w['t']:.0f} 秒, コースの {w['s']:.1f} m 地点, 横ずれ {w['xtrack_m']:+.2f} m (左が正)")
            if b.get("faster_than_cap_ratio") is not None:
                L.append(f"  速度の上限より速かった割合 {b['faster_than_cap_ratio'] * 100:.0f}% (手で動かしていた可能性)")
    cs = summary.get("corners") or []
    L.append("")
    L.append(f"[カーブ] {len(cs)} か所 (外 = 外に膨らんだ量, 内 = 内側に切り込んだ量)")
    for k, c in enumerate(cs):
        if "samples" in c:
            L.append(f"  {k + 1:>2}. {c['s0']:6.1f}-{c['s1']:6.1f} m  {c['turn_deg']:+5.0f} deg  外 {c['max_outward_m']:+.2f} m  "
                     f"内 {c['max_inward_m']:+.2f} m  平均 |横ずれ| {c['mean_abs_xtrack_m']:.2f} m"
                     + ("" if c["reached_end"] else "  (抜けていない)"))
        else:
            L.append(f"  {k + 1:>2}. {c['s0']:6.1f}-{c['s1']:6.1f} m  {c['turn_deg']:+5.0f} deg  (通っていない)")
    return "\n".join(L) + "\n"


def save_json(path: str, summary: dict) -> None:
    with open(path, "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
