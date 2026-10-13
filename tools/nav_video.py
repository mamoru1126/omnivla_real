#!/usr/bin/env python3
"""走行ログ (ナビゲーションノードの記録) から、デバッグ画面と同じ内容の動画を作る (ブラウザ不要).

  python3 tools/nav_video.py runs/replay/nav/latest                 # -> <ログ>/video.mp4
  python3 tools/nav_video.py log/nav/20261013_120000 --speed 2 --out /runs/run1.mp4

1 フレームの中身 (デバッグ画面と同じ):
  - カメラ画像 (モデルに入れた切り抜き・縮小後) に予測した 8 点の軌跡を重ねたもの
  - 目指しているサブゴールの画像、サブゴール番号・残りの道のり・類似度
  - 予測軌跡の俯瞰図 (0.5 m 方眼)、指示値 v / ω、直近 60 秒の履歴
  - コースの進み具合と、サブゴールの切り替えなどの出来事

ログには io.log_images: true (既定) でカメラ画像 (raw/) が入っている必要がある.
動画のエンコードに ffmpeg を使う.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omnivla_real.robot_config import RobotConfig, camera_model, preprocess_image  # noqa: E402

W, H = 1280, 720
BG = (27, 34, 40)
PANEL = (35, 44, 51)
LINE = (51, 64, 74)
TEXT = (227, 232, 236)
MUTED = (141, 154, 165)
PATH = (242, 193, 78)
PATH_EDGE = (42, 35, 6)
V_COL = (79, 179, 169)
W_COL = (160, 140, 240)
STOP = (224, 90, 79)
OK = (92, 196, 138)
WARN = (224, 168, 76)
HISTORY_SEC = 60.0


# --------------------------------------------------------------------------- 読み込み
def resolve_run_dir(path: str) -> str:
    path = os.path.abspath(path)
    if os.path.basename(path) == "latest":
        runs = sorted(d for d in glob.glob(os.path.join(os.path.dirname(path), "*"))
                      if os.path.isfile(os.path.join(d, "steps.csv")))
        if not runs:
            raise SystemExit(f"no runs in {os.path.dirname(path)}")
        return runs[-1]
    if not os.path.isfile(os.path.join(path, "steps.csv")):
        raise SystemExit(f"{path} has no steps.csv")
    return path


def _f(v: str) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def read_steps(run_dir: str) -> List[dict]:
    rows = []
    with open(os.path.join(run_dir, "steps.csv")) as f:
        for r in csv.DictReader(f):
            wps = []
            for i in range(8):
                x, y = _f(r.get(f"wp{i}_x")), _f(r.get(f"wp{i}_y"))
                if x is not None and y is not None:
                    wps.append((x, y))
            rows.append({"step": int(r["step"]), "t": _f(r.get("sim_time")), "subgoal": int(r.get("subgoal") or 0),
                         "num_nodes": int(r.get("num_nodes") or 0), "v": _f(r.get("v")) or 0.0,
                         "w": _f(r.get("w")) or 0.0, "latency": _f(r.get("latency")), "dist": _f(r.get("dist")),
                         "similarity": _f(r.get("similarity")), "state": r.get("state", ""),
                         "goal_local": (_f(r.get("goal_local_x")), _f(r.get("goal_local_y"))),
                         "wps": np.array(wps) if wps else None})
    if not rows:
        raise SystemExit("steps.csv is empty (ナビゲーションが推論する前に終わった)")
    if any(r["t"] is None for r in rows):   # 時刻が無ければ 3Hz とみなす
        for k, r in enumerate(rows):
            r["t"] = k / 3.0
    return rows


def read_events(run_dir: str) -> List[Tuple[float, str]]:
    out = []
    p = os.path.join(run_dir, "events.log")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8", errors="replace"):
            m = re.match(r"^\[([-0-9.]+)\]\s*(.*)$", line.rstrip("\n"))
            if m and m.group(1) != "-":
                out.append((float(m.group(1)), m.group(2)))
    return out


def raw_image(run_dir: str, step: int) -> Optional[Image.Image]:
    for ext in ("jpg", "png"):
        p = os.path.join(run_dir, "raw", f"{step:06d}.{ext}")
        if os.path.exists(p):
            try:
                return Image.open(p).convert("RGB")
            except OSError:
                return None
    return None


# --------------------------------------------------------------------------- 描画の部品
def load_font(size: int):
    cands = []
    try:
        from matplotlib import font_manager
        cands.append(font_manager.findfont("DejaVu Sans", fallback_to_default=True))
    except Exception:  # noqa: BLE001
        pass
    cands += ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    for c in cands:
        try:
            return ImageFont.truetype(c, size)
        except (OSError, TypeError):
            continue
    return ImageFont.load_default()


def fit(img: Image.Image, w: int, h: int) -> Tuple[Image.Image, Tuple[int, int]]:
    """縦横比を保って w x h に収める. 戻り値: (画像, 枠内での左上)."""
    s = min(w / img.width, h / img.height)
    im = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.BILINEAR)
    return im, ((w - im.width) // 2, (h - im.height) // 2)


def draw_path_on_image(img: Image.Image, wps: Optional[np.ndarray], cam, scale: float) -> None:
    if wps is None or len(wps) == 0:
        return
    pts = np.concatenate([[[cam.x_offset + 0.16, 0.0]], wps[:, :2]], axis=0)
    u, v, ok = cam.project(pts, img.width, img.height)
    coords = [(float(a), float(b)) for a, b, k in zip(u, v, ok) if k]
    d = ImageDraw.Draw(img)
    lw = max(3, int(6 * scale))
    if len(coords) >= 2:
        d.line(coords, fill=PATH_EDGE, width=lw + 4, joint="curve")
        d.line(coords, fill=PATH, width=lw, joint="curve")
    r = max(3, int(5 * scale))
    for (a, b) in coords[1:]:
        d.ellipse([a - r - 2, b - r - 2, a + r + 2, b + r + 2], fill=PATH_EDGE)
        d.ellipse([a - r, b - r, a + r, b + r], fill=PATH)


def topdown(wps: Optional[np.ndarray], goal_local, size: Tuple[int, int], font) -> Image.Image:
    w, h = size
    img = Image.new("RGB", size, PANEL)
    d = ImageDraw.Draw(img)
    x_range = (-0.5, 2.5)
    s = (h - 8) / (x_range[1] - x_range[0])

    def px(x, y):
        return w / 2.0 - y * s, h - 4 - (x - x_range[0]) * s

    for gx in np.arange(-0.5, 2.51, 0.5):
        _, yy = px(gx, 0)
        d.line([(0, yy), (w, yy)], fill=LINE)
    for gy in np.arange(-3.0, 3.01, 0.5):
        xx, _ = px(0, gy)
        d.line([(xx, 0), (xx, h)], fill=LINE)
    rx, ry = px(0, 0)
    d.polygon([(rx, ry - 11), (rx - 8, ry + 7), (rx + 8, ry + 7)], fill=MUTED)
    if goal_local is not None and goal_local[0] is not None and goal_local[1] is not None:
        gx, gy = px(*goal_local)
        gx, gy = min(max(gx, 7), w - 7), min(max(gy, 7), h - 7)
        d.regular_polygon((gx, gy, 7), 5, fill=STOP)
    if wps is not None and len(wps):
        pts = [(rx, ry)] + [px(p[0], p[1]) for p in wps]
        d.line(pts, fill=PATH_EDGE, width=7)
        d.line(pts, fill=PATH, width=4)
        for (a, b) in pts[1:]:
            d.ellipse([a - 3, b - 3, a + 3, b + 3], fill=PATH)
    d.text((6, 4), "predicted path (0.5 m grid)", fill=MUTED, font=font)
    return img


def meter(d: ImageDraw.ImageDraw, box, label: str, value: float, vmax: float, color, font_big, font) -> None:
    x0, y0, x1, y1 = box
    d.rectangle(box, fill=PANEL)
    d.text((x0 + 10, y0 + 6), label, fill=MUTED, font=font)
    d.text((x0 + 10, y0 + 24), f"{value:+.2f}", fill=TEXT, font=font_big)
    bx0, bx1, by = x0 + 10, x1 - 10, y1 - 16
    mid = (bx0 + bx1) / 2
    d.rectangle([bx0, by, bx1, by + 8], fill=LINE)
    frac = max(-1.0, min(1.0, value / vmax)) if vmax > 0 else 0.0
    end = mid + frac * (bx1 - bx0) / 2
    d.rectangle([min(mid, end), by, max(mid, end), by + 8], fill=color)
    d.line([(mid, by - 3), (mid, by + 11)], fill=TEXT)


def history_chart(rows: List[dict], k: int, size, font) -> Image.Image:
    w, h = size
    img = Image.new("RGB", size, PANEL)
    d = ImageDraw.Draw(img)
    t_now = rows[k]["t"]
    t0 = t_now - HISTORY_SEC
    lo = bisect.bisect_left([r["t"] for r in rows], t0)
    seg = rows[lo:k + 1]
    ymax = 1.5

    def py(v):
        return h / 2 - (max(-ymax, min(ymax, v)) / ymax) * (h / 2 - 6)

    for gv in (-1.0, -0.5, 0.0, 0.5, 1.0):
        d.line([(0, py(gv)), (w, py(gv))], fill=LINE if gv else MUTED)
    for key, col in (("v", V_COL), ("w", W_COL)):
        pts = [((r["t"] - t0) / HISTORY_SEC * w, py(r[key])) for r in seg]
        if len(pts) >= 2:
            d.line(pts, fill=col, width=2)
    d.text((6, 3), "last 60 s", fill=MUTED, font=font)
    d.text((w - 150, 3), "v [m/s]", fill=V_COL, font=font)
    d.text((w - 80, 3), "w [rad/s]", fill=W_COL, font=font)
    return img


# --------------------------------------------------------------------------- 1 フレーム
class Renderer:
    def __init__(self, run_dir: str, rows: List[dict], events, robot: RobotConfig, meta: dict):
        self.run_dir, self.rows, self.events, self.robot, self.meta = run_dir, rows, events, robot, meta
        self.cam = camera_model(robot)
        self.ev_t = [e[0] for e in events]
        self.goals: Dict[int, Optional[Image.Image]] = {}
        self.f_small, self.f, self.f_big, self.f_title = load_font(13), load_font(16), load_font(30), load_font(20)
        model = meta.get("model", {}) if isinstance(meta, dict) else {}
        self.model_name = model.get("server_model") or model.get("model") or ""
        self.v_max = 1.0

    def goal(self, k: int) -> Optional[Image.Image]:
        if k not in self.goals:
            p = os.path.join(self.run_dir, "goals", f"{k}.jpg")
            self.goals[k] = Image.open(p).convert("RGB") if os.path.exists(p) else None
        return self.goals[k]

    def render(self, k: int) -> Image.Image:
        r = self.rows[k]
        canvas = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(canvas)
        # --- 見出し
        state = r["state"] or "-"
        scol = OK if state == "running" else (WARN if state in ("waiting_image", "stopped") else STOP)
        d.text((16, 10), "OmniVLA", fill=TEXT, font=self.f_title)
        d.text((120, 13), f"t = {r['t'] - self.rows[0]['t']:7.1f} s    step {r['step']}", fill=MUTED, font=self.f)
        d.rounded_rectangle([380, 9, 380 + 14 + d.textlength(state, font=self.f), 33], 6, fill=scol)
        d.text((387, 13), state, fill=BG, font=self.f)
        info = f"model {self.model_name}" + (f"   latency {r['latency'] * 1000:.0f} ms" if r["latency"] else "")
        d.text((W - 16 - d.textlength(info, font=self.f), 13), info, fill=MUTED, font=self.f)
        # --- カメラ画像 + 予測軌跡
        L = (16, 48, 816, 648)
        d.rectangle(L, fill=PANEL)
        raw = raw_image(self.run_dir, r["step"])
        if raw is not None:
            cur = preprocess_image(raw, self.robot.image)
            big, (ox, oy) = fit(cur, L[2] - L[0], L[3] - L[1])
            draw_path_on_image(big, r["wps"], self.cam, big.width / 640.0)
            canvas.paste(big, (L[0] + ox, L[1] + oy))
        else:
            d.text((L[0] + 20, L[1] + 20), "no camera image in the log (io.log_images: false?)", fill=MUTED, font=self.f)
        d.rectangle([L[0], L[1], L[0] + 250, L[1] + 26], fill=BG)
        d.text((L[0] + 8, L[1] + 4), "camera (model input) + prediction", fill=TEXT, font=self.f_small)
        # --- サブゴール
        R0, R1 = 832, W - 16
        g = self.goal(r["subgoal"])
        gb = (R0, 48, R1, 48 + 324)
        d.rectangle(gb, fill=PANEL)
        if g is not None:
            gi, (ox, oy) = fit(g, gb[2] - gb[0], gb[3] - gb[1])
            canvas.paste(gi, (gb[0] + ox, gb[1] + oy))
        n = max(1, r["num_nodes"])
        d.rectangle([gb[0], gb[1], gb[2], gb[1] + 26], fill=BG)
        d.text((gb[0] + 8, gb[1] + 4), f"subgoal {r['subgoal']} / {n - 1}", fill=TEXT, font=self.f_small)
        sub = []
        if r["dist"] is not None:
            sub.append(f"{r['dist']:+.1f} m to go")
        if r["similarity"] is not None:
            sub.append(f"similarity {r['similarity']:.3f}")
        if sub:
            txt = "   ".join(sub)
            d.rectangle([gb[0], gb[3] - 26, gb[2], gb[3]], fill=BG)
            d.text((gb[0] + 8, gb[3] - 22), txt, fill=TEXT, font=self.f_small)
        # --- 俯瞰図 + v / w
        canvas.paste(topdown(r["wps"], r["goal_local"], (200, 180), self.f_small), (R0, 384))
        meter(d, (R0 + 212, 384, R1, 468), "v  [m/s]", r["v"], self.v_max, V_COL, self.f_big, self.f_small)
        meter(d, (R0 + 212, 480, R1, 564), "w  [rad/s]", r["w"], 1.5, W_COL, self.f_big, self.f_small)
        canvas.paste(history_chart(self.rows, k, (R1 - R0, 72), self.f_small), (R0, 576))
        # --- 進み具合と出来事
        d.rectangle([16, 660, W - 16, 668], fill=LINE)
        frac = r["subgoal"] / max(1, n - 1)
        d.rectangle([16, 660, 16 + frac * (W - 32), 668], fill=PATH)
        j = bisect.bisect_right(self.ev_t, r["t"])
        recent = [e for e in self.events[max(0, j - 2):j] if r["t"] - e[0] < 15.0]
        y = 676
        for i, (et, text) in enumerate(recent):
            col = TEXT if i == len(recent) - 1 else MUTED
            d.text((16, y), f"[{et - self.rows[0]['t']:6.1f} s] {text}"[:150], fill=col, font=self.f_small)
            y += 18
        return canvas


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="走行ログのディレクトリ (…/nav/<時刻> または …/nav/latest)")
    ap.add_argument("--out", default="", help="出力 (既定: <ログ>/video.mp4)")
    ap.add_argument("--robot", default="", help="robot.yaml (既定: ログの meta.json に入っている設定)")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--speed", type=float, default=1.0, help="再生速度 (2 なら 2 倍速)")
    ap.add_argument("--start", type=float, default=0.0, help="走行開始から何秒目から [s]")
    ap.add_argument("--end", type=float, default=0.0, help="何秒目まで [s] (0 で最後まで)")
    ap.add_argument("--crf", type=int, default=23, help="画質 (小さいほど高画質. 18〜28)")
    args = ap.parse_args(argv)

    run_dir = resolve_run_dir(args.run_dir)
    out = args.out or os.path.join(run_dir, "video.mp4")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg が見つかりません (apt install ffmpeg)")
    meta = {}
    if os.path.exists(os.path.join(run_dir, "meta.json")):
        with open(os.path.join(run_dir, "meta.json")) as f:
            meta = json.load(f)
    if args.robot:
        robot = RobotConfig.load(args.robot)
    else:
        robot = RobotConfig.load(None, meta.get("robot") or {})
    rows = read_steps(run_dir)
    events = read_events(run_dir)
    if not os.path.isdir(os.path.join(run_dir, "raw")):
        print("注意: ログにカメラ画像 (raw/) がありません. navigator.yaml の io.log_images: true で走らせてください")
    rend = Renderer(run_dir, rows, events, robot, meta)

    times = [r["t"] for r in rows]
    t0 = times[0] + max(0.0, args.start)
    t1 = times[-1] if args.end <= 0 else min(times[-1], times[0] + args.end)
    dt = max(1e-3, args.speed) / max(1.0, args.fps)
    n_frames = max(1, int(math.floor((t1 - t0) / dt)) + 1)
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", f"{args.fps:g}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(args.crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    last_k, frame = -1, None
    print(f"{run_dir}: {len(rows)} steps, {t1 - t0:.1f} s -> {n_frames} frames ({args.fps:g} fps, x{args.speed:g})")
    try:
        for i in range(n_frames):
            t = t0 + i * dt
            k = max(0, bisect.bisect_right(times, t) - 1)
            if k != last_k:
                frame = rend.render(k).tobytes()
                last_k = k
            proc.stdin.write(frame)
            if i % 300 == 0:
                print(f"  {i}/{n_frames}", flush=True)
    finally:
        proc.stdin.close()
        rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"ffmpeg failed (exit {rc})")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
