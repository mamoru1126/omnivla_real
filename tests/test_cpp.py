"""C++ の ROS 1 ノードの中身 (ros1/omnivla_real_ros1) が Python の実装と同じ動きをするかの確認.

  python3 -m pytest -q tests/test_cpp.py
  python3 tests/test_cpp.py

core_check (C++) は OMNIVLA_CORE_CHECK で場所を指定するか、無ければ cmake でここでビルドする (g++ と cmake が必要)。
"""
from __future__ import annotations

import json
import math
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from omnivla_real.controller import ControllerConfig, compute_command  # noqa: E402
from omnivla_real.topomap import SubgoalTracker, TrackerConfig, Topomap, GoalNode, resolve_mode  # noqa: E402

PKG = os.path.join(REPO, "ros1", "omnivla_real_ros1")
_BIN = None


def core_check() -> str:
    global _BIN
    if _BIN:
        return _BIN
    cands = [os.environ.get("OMNIVLA_CORE_CHECK", ""), os.path.join(HERE, ".build_cpp", "core_check"),
             "/opt/catkin_ws/devel/lib/omnivla_real_ros1/core_check"]
    for c in cands:
        if c and os.path.exists(c):
            _BIN = c
            return c
    if not shutil.which("cmake") or not shutil.which("g++"):
        return ""
    build = os.path.join(HERE, ".build_cpp")
    subprocess.run(["cmake", "-S", PKG, "-B", build, "-DCMAKE_BUILD_TYPE=Release"], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["cmake", "--build", build, "-j", str(os.cpu_count() or 2), "--target", "omnivla_core_check"], check=True,
                   stdout=subprocess.DEVNULL)
    _BIN = os.path.join(build, "core_check")
    return _BIN


def _skip(msg: str) -> bool:
    print(f"  (skip: {msg})")
    try:
        import pytest
        pytest.skip(msg)
    except ImportError:
        pass
    return True


def run(mode: str, payload=None, args=()):
    p = subprocess.run([core_check(), mode, *args], input=None if payload is None else json.dumps(payload),
                       capture_output=True, text=True, timeout=300)
    if p.returncode != 0:
        raise RuntimeError(f"core_check {mode} failed:\n{p.stderr[-3000:]}")
    return json.loads(p.stdout)


def _random_waypoints(rng, kind: str) -> np.ndarray:
    t = (np.arange(8) + 1) / 3.0
    if kind == "arc":
        v, w = rng.uniform(0, 0.6), rng.uniform(-1.2, 1.2)
        yaw = w * t
        x = np.where(abs(w) > 1e-6, v / (w + 1e-12) * np.sin(yaw), v * t)
        y = np.where(abs(w) > 1e-6, v / (w + 1e-12) * (1 - np.cos(yaw)), 0 * t)
    elif kind == "rotate":
        yaw = rng.uniform(-2, 2) * t
        x, y = rng.normal(0, 0.01, 8), rng.normal(0, 0.01, 8)
    elif kind == "back":
        yaw = rng.uniform(-3.1, 3.1, 8)
        x, y = -rng.uniform(0, 0.5, 8), rng.normal(0, 0.3, 8)
    elif kind == "zero":
        yaw, x, y = np.zeros(8), np.zeros(8), np.zeros(8)
    else:  # noisy: 向きが -pi/pi をまたぐ
        yaw = np.cumsum(rng.normal(0, 1.0, 8)) + math.pi * 0.95
        x, y = np.cumsum(rng.uniform(0, 0.2, 8)), np.cumsum(rng.normal(0, 0.1, 8))
    return np.stack([x, y, np.cos(yaw), np.sin(yaw)], 1)


def test_controller_matches_python():
    if not core_check():
        return _skip("cmake / g++ not available")
    rng = np.random.default_rng(0)
    cases = []
    for i in range(300):
        mode = ["trajectory", "upstream", "pure_pursuit"][i % 3]
        cfg = {"mode": mode, "dt": [1 / 3, 0.2, 0.5][i % 3 if i % 7 else 0], "track_max_v": rng.uniform(0.2, 0.8),
               "track_max_w": rng.uniform(0.5, 1.5), "track_horizon": int(rng.integers(0, 8)),
               "respect_predicted_speed": bool(i % 2), "lookahead": rng.uniform(0.2, 1.0)}
        wps = _random_waypoints(rng, ["arc", "rotate", "back", "zero", "noisy"][i % 5])
        cases.append({"cfg": cfg, "waypoints": wps.tolist()})
    got = run("controller", {"cases": cases})
    for c, (v, w) in zip(cases, got):
        ev, ew = compute_command(np.asarray(c["waypoints"]), ControllerConfig(**c["cfg"]))
        assert abs(ev - v) < 1e-9 and abs(ew - w) < 1e-9, (c, (v, w), (ev, ew))


def _course_nodes(n=12, spacing=1.0):
    nodes = []
    for k in range(n):   # 直進 -> 左に 90 度曲がる
        if k < 6:
            nodes.append((k * spacing, 0.0, 0.0))
        else:
            nodes.append((5 * spacing, (k - 5) * spacing, math.pi / 2))
    return nodes


def test_tracker_matches_python():
    if not core_check():
        return _skip("cmake / g++ not available")
    from PIL import Image
    rng = np.random.default_rng(1)
    poses = _course_nodes()
    for mode, frame, has_loc, has_odom in (("pose", "map", True, False), ("odom", "odom", False, True),
                                           ("image", "odom", False, False), ("image_odom", "odom", False, True),
                                           ("auto", "map", True, True), ("auto", "odom", False, True)):
        for trial in range(4):
            cfg = TrackerConfig(reach_check=mode, confirm=int(rng.integers(1, 3)), search_window=int(rng.integers(0, 3)),
                                image_threshold=0.8, goal_image_threshold=0.85, min_travel_m=0.2,
                                reach_angle_deg=float(rng.choice([0.0, 30.0])))
            tm_py = Topomap([GoalNode(Image.new("RGB", (4, 4)), p) for p in poses], frame=frame, start=poses[0])
            tm_js = {"directory": "", "frame": frame, "start": list(poses[0]),
                     "nodes": [{"path": "", "pose": list(p), "s": None} for p in poses]}
            m = resolve_mode(mode, has_loc, has_odom, tm_py)
            tr = SubgoalTracker(tm_py, cfg, m)
            steps, expected = [], []
            travel = 0.0
            for i in range(160):   # コースに沿って (ゆらぎ付きで) 進む
                s = min(i * 0.09, 16.0)
                if s <= 5:
                    x, y, yaw = s, 0.0, 0.0
                else:
                    x, y, yaw = 5.0, s - 5.0, math.pi / 2
                pose = [x + rng.normal(0, 0.15), y + rng.normal(0, 0.15), yaw + rng.normal(0, 0.3)]
                travel += 0.09
                sims = [[j, float(np.clip(1.0 - 0.35 * math.hypot(poses[j][0] - x, poses[j][1] - y) + rng.normal(0, 0.05),
                                          -1, 1))] for j in range(len(poses))]
                use_pose = None if rng.random() < 0.05 else pose
                step = {"pose": use_pose, "sims": sims, "travel": travel if has_odom else None}
                steps.append(step)
                d = dict(sims)
                adv = tr.update(tuple(use_pose) if use_pose else None, (lambda j, d=d: d.get(j, 0.0)),
                                travel if has_odom else None)
                expected.append({"advanced": adv, "index": tr.index, "done": tr.done, "reason": tr.last_reason or "",
                                 "similarity": tr.last_similarity, "distance": tr.last_distance})
            got = run("tracker", {"topomap": tm_js, "cfg": vars(cfg), "has_localization": has_loc,
                                  "has_odom": has_odom, "steps": steps})
            assert got["mode"] == m, (got["mode"], m)
            for k, (g, e) in enumerate(zip(got["steps"], expected)):
                for key in ("advanced", "index", "done", "reason"):
                    assert g[key] == e[key], (mode, trial, k, key, g, e)
                for key in ("similarity", "distance"):
                    assert (g[key] is None) == (e[key] is None) and (e[key] is None or abs(g[key] - e[key]) < 1e-9), \
                        (mode, k, key, g, e)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _start_fake_server(port: int):
    proc = subprocess.Popen([sys.executable, os.path.join(HERE, "fake_policy.py"), "--port", str(port)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    import urllib.request
    for _ in range(100):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/info", timeout=1)
            return proc
        except OSError:
            time.sleep(0.1)
    proc.kill()
    raise RuntimeError("fake policy server did not start")


def make_replay_case(out_dir: str, laps: int = 1):
    """合成コース -> 3Hz の画像 (JPEG) + オドメトリ と、それから作った topomap."""
    from omnivla_real.bag.reader import ListSource
    from omnivla_real.desk_eval import prepare_frames
    from omnivla_real.extract import read_streams
    from omnivla_real.robot_config import RobotConfig
    from omnivla_real.topomap_build import build_topomap
    from synthetic import simulate_course

    robot = RobotConfig.load(os.path.join(REPO, "configs", "robot.yaml"))
    msgs = simulate_course(image_size=(160, 120), laps=laps)
    s = read_streams(ListSource(msgs), robot, os.path.join(out_dir, "frames"), 3.0)
    topomap = os.path.join(out_dir, "topomap")
    build_topomap(s, topomap, spacing=1.0)
    ft, odom_xyz, _, _ = prepare_frames(s, 3.0, "odom")
    frames = [{"t": float(ft.t[i]), "path": ft.image_paths[i], "odom": [float(v) for v in odom_xyz[i]], "loc": None}
              for i in range(len(ft))]
    with open(os.path.join(out_dir, "frames.json"), "w") as f:
        json.dump({"frames": frames}, f)
    return topomap, frames


def test_engine_replay_matches_python():
    """同じ推論サーバ (偽物) に対して、C++ の NavEngine と Python の NavEngine が同じ結果を出す."""
    if not core_check():
        return _skip("cmake / g++ not available")
    from PIL import Image
    from omnivla_real.engine import NavEngine
    from omnivla_real.nav_config import load_nav_config
    from omnivla_real.remote import RemotePolicy
    from omnivla_real.robot_config import RobotConfig
    from omnivla_real.topomap import load_topomap

    tmp = tempfile.mkdtemp()
    topomap, frames = make_replay_case(tmp)
    port = _free_port()
    server = _start_fake_server(port)
    try:
        url = f"http://127.0.0.1:{port}"
        # --- Python ---
        nav = load_nav_config(os.path.join(REPO, "configs", "navigator.yaml"))
        robot = RobotConfig.load(os.path.join(REPO, "configs", "robot.yaml"))
        pol = RemotePolicy(url, log=lambda s: None)
        eng = NavEngine(pol, load_topomap(topomap), nav.engine, robot)
        eng.on_odom(frames[0]["t"], tuple(frames[0]["odom"]))
        eng.start()
        py = []
        for fr in frames:
            img = Image.open(fr["path"]).convert("RGB")
            eng.on_image(fr["t"], img)
            eng.on_odom(fr["t"], tuple(fr["odom"]))
            if eng.state == "running":
                r = eng.step(fr["t"])
                py.append((r.state, r.subgoal, r.v, r.w))
            else:
                py.append((eng.state, None, None, None))
        # --- C++ ---
        got = run("replay", None, ["--repo", REPO, "--frames", os.path.join(tmp, "frames.json"), "--topomap", topomap,
                                   "--set", f"model.url={url}", "--set", "io.web_port=0",
                                   "--set", f"io.log_dir={tmp}/log"])
        assert got["mode"] == eng.mode == "image_odom", (got["mode"], eng.mode)
        cpp = [(f.get("state"), f.get("subgoal"), f.get("v"), f.get("w")) for f in got["frames"]]
        assert len(cpp) == len(py)
        n_run = 0
        for k, (a, b) in enumerate(zip(cpp, py)):
            assert a[0] == b[0], (k, a, b)
            if b[0] == "running":
                n_run += 1
                assert a[1] == b[1] and abs(a[2] - b[2]) < 1e-9 and abs(a[3] - b[3]) < 1e-9, (k, a, b)
        assert n_run > 50 and any(b[0] == "reached" for b in py), py[-3:]
        # 走行ログ (plot_nav_log.py で読める)
        logs = sorted(os.listdir(os.path.join(tmp, "log")))
        assert logs and os.path.exists(os.path.join(tmp, "log", logs[-1], "steps.csv"))
        sys.path.insert(0, os.path.join(REPO, "tools"))
        import plot_nav_log
        plot_nav_log.main([os.path.join(tmp, "log", "latest")])
    finally:
        server.kill()


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            t0 = time.time()
            try:
                fn()
                print(f"PASS {name} ({time.time() - t0:.1f}s)")
            except Exception as e:  # noqa: BLE001
                failed += 1
                import traceback

                traceback.print_exc()
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failed else 0)
