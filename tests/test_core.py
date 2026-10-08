"""ROS / GPU / rosbags 無しで動く部分の単体テスト (合成したメッセージ列を使う).

  python3 -m pytest -q tests        (コンテナ内)
  python3 tests/test_core.py        (pytest が無い環境でも実行可)
"""
from __future__ import annotations

import io
import json
import math
import os
import shutil
import sys
import tempfile
import threading
import time
from types import SimpleNamespace as NS

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from omnivla_real import controller, data_utils, geometry, odometry  # noqa: E402
from omnivla_real.bag.messages import image_to_rgb, pose_values, stamp_to_sec, twist_values  # noqa: E402
from omnivla_real.bag.reader import ListSource  # noqa: E402
from omnivla_real.convert import (ConvertConfig, build_frames, convert_streams, idle_mask,  # noqa: E402
                                  segment_frames, update_dataset_info)
from omnivla_real.desk_eval import (DeskEvalConfig, OraclePolicy, integrate_commands, prepare_frames,  # noqa: E402
                                    run_desk_eval, simple_embedding)
from omnivla_real.engine import NavEngine  # noqa: E402
from omnivla_real.extract import read_streams  # noqa: E402
from omnivla_real.nav_config import load_nav_config  # noqa: E402
from omnivla_real.robot_config import ImageConfig, RobotConfig, preprocess_image  # noqa: E402
from omnivla_real.topomap import (GoalNode, SubgoalTracker, Topomap, TrackerConfig, align_to_start,  # noqa: E402
                                  load_topomap, resolve_mode)
from omnivla_real.topomap_build import build_topomap  # noqa: E402
from omnivla_real.trajectory_io import find_trajectories, load_meta, load_trajectory  # noqa: E402
from synthetic import header, jpeg_bytes, quat, render, simulate_course, stamp  # noqa: E402

CONFIGS = os.path.join(REPO, "configs")


def robot_cfg(localization: bool = False, raw: bool = False) -> RobotConfig:
    topics = {"localization": "/localization" if localization else ""}
    if raw:
        topics["image"] = "/camera/image_raw"
    return RobotConfig.load(os.path.join(CONFIGS, "robot.yaml"), {"topics": topics})


_CACHE = {}


def course(**kw):
    key = json.dumps(kw, sort_keys=True)
    if key not in _CACHE:
        _CACHE[key] = simulate_course(**kw)
    return _CACHE[key]


# ---------------------------------------------------------------- messages
def test_decode_images():
    img = render(1.0, 2.0, 0.3, (40, 30))
    ref = np.asarray(img)
    # jpeg
    m = NS(header=header(1.0), format="jpeg", data=jpeg_bytes(img))
    out = image_to_rgb(m, "sensor_msgs/msg/CompressedImage")
    assert out.shape == ref.shape and np.abs(out.astype(int) - ref).mean() < 8
    # raw: rgb8 (行末にパディングあり), bgr8, mono8, yuyv
    pad = np.zeros((30, 40 * 3 + 8), np.uint8)
    pad[:, :120] = ref.reshape(30, -1)
    assert np.array_equal(image_to_rgb(NS(height=30, width=40, step=128, encoding="rgb8", data=pad.reshape(-1))), ref)
    bgr = ref[..., ::-1].copy()
    assert np.array_equal(image_to_rgb(NS(height=30, width=40, step=120, encoding="bgr8", data=bgr.tobytes())), ref)
    g = image_to_rgb(NS(height=30, width=40, step=40, encoding="mono8", data=ref[..., 0].tobytes()))
    assert g.shape == (30, 40, 3) and np.array_equal(g[..., 1], ref[..., 0])
    yuyv = np.full((30, 80), 128, np.uint8)
    yuyv[:, 0::2] = 200  # Y=200, U=V=128 -> 灰色 200
    out = image_to_rgb(NS(height=30, width=40, step=80, encoding="yuyv", data=yuyv.tobytes()))
    assert out.shape == (30, 40, 3) and np.all(np.abs(out.astype(int) - 200) <= 1)


def test_decode_twist_pose_stamp():
    assert math.isclose(stamp_to_sec(stamp(12.25)), 12.25, abs_tol=1e-9)
    assert stamp_to_sec(NS(secs=3, nsecs=500000000)) == 3.5      # rospy
    assert stamp_to_sec(NS(sec=0, nanosec=0)) is None
    tw = NS(linear=NS(x=0.4, y=0, z=0), angular=NS(x=0, y=0, z=-0.2))
    assert twist_values(tw, "geometry_msgs/msg/Twist") == (0.4, -0.2)
    assert twist_values(NS(header=header(1), twist=tw), "geometry_msgs/msg/TwistStamped") == (0.4, -0.2)
    odom = NS(header=header(1), child_frame_id="base_link",
              pose=NS(pose=NS(position=NS(x=1, y=2, z=0), orientation=quat(0.5))),
              twist=NS(twist=tw))
    assert twist_values(odom, "nav_msgs/msg/Odometry") == (0.4, -0.2)
    assert np.allclose(pose_values(odom, "nav_msgs/msg/Odometry"), (1, 2, 0.5))
    ps = NS(header=header(1), pose=NS(position=NS(x=3, y=4, z=0), orientation=quat(-1.0)))
    assert np.allclose(pose_values(ps, "geometry_msgs/msg/PoseStamped"), (3, 4, -1.0))
    pwc = NS(header=header(1), pose=NS(pose=ps.pose, covariance=[0] * 36))
    assert np.allclose(pose_values(pwc, "geometry_msgs/msg/PoseWithCovarianceStamped"), (3, 4, -1.0))
    tf = NS(transforms=[NS(header=header(1, "odom"), child_frame_id="base_link",
                           transform=NS(translation=NS(x=9, y=9, z=0), rotation=quat(0))),
                        NS(header=header(1, "map"), child_frame_id="base_link",
                           transform=NS(translation=NS(x=5, y=6, z=0), rotation=quat(1.0)))])
    assert np.allclose(pose_values(tf, "tf2_msgs/msg/TFMessage", "map", "base_link"), (5, 6, 1.0))
    assert pose_values(tf, "tf2_msgs/msg/TFMessage", "map", "laser") is None
    custom = NS(drive=NS(speed=0.7, yaw_rate=0.1))
    assert twist_values(custom, "my_msgs/msg/Drive", "drive.speed", "drive.yaw_rate") == (0.7, 0.1)


# ---------------------------------------------------------------- odometry
def test_integrate_and_interpolate():
    t = np.arange(0, 10, 0.05)
    tr = odometry.integrate_twist(t, np.full(len(t), 0.5), np.full(len(t), 0.25))  # 半径 2m の円
    x, y, yaw, ok = tr.at([2 * math.pi / 0.25 / 4], 1e9)      # 1/4 周 (t = 2π)
    assert ok[0] and abs(x[0] - 2.0) < 0.03 and abs(y[0] - 2.0) < 0.03 and abs(yaw[0] - math.pi / 2) < 0.02
    tq = np.array([0.0, 1.0, 1.04, 5.0])
    z = odometry.zero_order_hold(np.array([0.0, 1.0]), np.array([1.0, 2.0]), tq, timeout=0.5)
    assert list(z) == [1.0, 2.0, 2.0, 0.0]                     # 途切れたら 0
    pt = odometry.PoseTrack.from_arrays([0, 1, 3], [0, 1, 3], [0, 0, 0], [0, 0, 0])
    _, _, _, ok = pt.at([0.5, 2.0, 4.0], max_gap=1.5)
    assert list(ok) == [True, False, False]                    # 2s の空きと範囲外は使わない


def test_estimate_delay():
    t = np.arange(0, 60, 0.05)
    w = np.sin(t * 0.7) * (t % 10 < 7)
    meas_t = np.arange(0, 60, 0.02)
    meas = np.interp(meas_t - 0.3, t, w)
    d, r = odometry.estimate_delay(t, w, meas_t, meas)
    assert abs(d - 0.3) < 0.05 and r > 0.9


# ---------------------------------------------------------------- conversion
def test_extract_and_convert_course_run():
    msgs = course()
    tmp = tempfile.mkdtemp()
    try:
        s = read_streams(ListSource(msgs), robot_cfg(), os.path.join(tmp, "_frames", "r"), 3.0, name="r")
        info = s.summary()
        assert 14.5 < info["image_rate_hz"] < 15.5 and len(s.image_t) == len(s.image_paths) > 100
        assert abs(np.median(np.diff(s.image_t)) - 1 / 3) < 0.03     # 3Hz に間引かれている
        assert Image.open(s.image_paths[0]).size == (320, 240)      # robot.yaml image.width
        cfg = ConvertConfig()
        ft = build_frames(s, cfg)
        segs = segment_frames(ft, cfg)
        assert len(segs) == 2                                       # 途中の 6 秒停止で 2 区間に分かれる
        t_rel = [(ft.t[a] - s.bag_start, ft.t[b - 1] - s.bag_start) for a, b in segs]
        assert t_rel[0][0] < 3.0 and 19.5 < t_rel[0][1] < 21.6 and 25.5 < t_rel[1][0] < 27.0
        res = convert_streams(s, cfg, os.path.join(tmp, "ds"), "r")
        info = update_dataset_info(os.path.join(tmp, "ds"))
        assert info["trajectories"] == len(res["trajectories"]) == 2
        assert 0.15 < info["metric_waypoint_spacing"] < 0.21       # 0.6 m/s / 3 Hz = 0.2 m (角で遅い)
        d = find_trajectories(os.path.join(tmp, "ds"))[1]
        tr = load_trajectory(d)
        assert set(tr) >= {"position", "yaw", "stamp", "cmd_v", "cmd_w", "exclude", "perturbed"}
        assert load_meta(d)["pose_source"] == "odom"
        # 正解ラベル: 左回りのコースなので、曲がっている所では y (左) が正
        n = len(tr["position"])
        yaws = []
        for t in range(n - 9):
            a = data_utils.compute_action_targets(tr["position"], tr["yaw"], t, metric_spacing=0.2)
            yaws.append(math.degrees(math.atan2(a[-1, 3], a[-1, 2])))
        assert max(yaws) > 40 and min(yaws) > -5
    finally:
        shutil.rmtree(tmp)


def test_pose_sources_agree_short_term():
    msgs = course()
    s = read_streams(ListSource(msgs), robot_cfg(localization=True), None, 3.0, decode_images=False)
    t0 = np.arange(s.cmd_t[0] + 3, s.cmd_t[-1] - 5, 1.0)
    true = s.loc_track()
    cmd = odometry.integrate_twist(s.cmd_t, s.cmd_v, s.cmd_w, delay=0.15)
    r = odometry.compare_tracks(true, cmd, t0, 8 / 3)
    assert r["pos_err_median_m"] < 0.1 and 0.95 < r["distance_ratio_b_over_a"] < 1.1   # 滑り 3% 程度
    r2 = odometry.compare_tracks(true, s.odom_track(), t0, 8 / 3)
    assert r2["pos_err_median_m"] < 0.1


def test_idle_mask_and_raw_images():
    # 0..11 停止 → 12..21 前進 → 22..31 停止 (horizon=8 なので 0..4 は「この先 8 フレーム動かない」)
    x = np.r_[np.zeros(12), np.linspace(0, 2, 10), np.full(10, 2.0)]
    m = idle_mask(x, np.zeros_like(x), np.zeros_like(x), keep_tail=3)
    assert m[:5].all()                                # 動き出すまでの停止は学習起点から外す
    assert not m[5:21].any()                          # 動き出し・走行中は使う
    assert m[22:28].all() and not m[-4:].any()        # 途中停止は外す / 区間末尾 (keep_tail) は残す
    msgs = course(raw_image=True, stop_at=None, twist_stamped=True)
    s = read_streams(ListSource(msgs), robot_cfg(raw=True), tempfile.mkdtemp(), 3.0)
    assert len(s.image_t) > 50 and len(s.cmd_t) > 100


def test_preprocess_crop():
    img = Image.new("RGB", (640, 480))
    out = preprocess_image(np.asarray(img), ImageConfig(crop=[0.0, 0.25, 0.1, 0.1], width=320))
    assert out.size == (320, 240 * 0.75 * 320 / 512 / 0.75 * 0.75 // 1 + 0) or out.size[0] == 320
    w, h = 640 * 0.8, 480 * 0.75
    assert out.size == (320, int(round(h * 320 / w)))


# ---------------------------------------------------------------- topomap / tracker
def test_topomap_from_run_and_tracker_modes():
    tmp = tempfile.mkdtemp()
    try:
        s = read_streams(ListSource(course(stop_at=None)), robot_cfg(localization=True), os.path.join(tmp, "f"), 10.0)
        res = build_topomap(s, os.path.join(tmp, "topo"), spacing=1.0)
        assert res["frame"] == "map" and 20 <= res["num_nodes"] <= 26
        tm = load_topomap(os.path.join(tmp, "topo"))
        assert len(tm) == res["num_nodes"] and tm.has_poses() and tm.start is not None
        ss = [n.s for n in tm.nodes]
        assert np.all(np.diff(ss[:-1]) > 0.95) and tm.meta["spacing_m"] == 1.0
        assert resolve_mode("auto", True, True, tm) == "pose"
        # pose: ノードの位置を順にたどれば最後まで進む
        tr = SubgoalTracker(tm, TrackerConfig(), "pose")
        for n in tm.nodes:
            for k in range(3):
                tr.update(n.pose)
        assert tr.done
        # odom: 開始位置を start に重ねる
        start_odom = (10.0, -3.0, 1.0)
        rel_goal = geometry.relative_pose(tm.start, tm.nodes[3].pose)
        from omnivla_real.topomap import compose
        odom_at_node3 = compose(start_odom, rel_goal)
        p = align_to_start(odom_at_node3, start_odom, tm.start)
        assert np.allclose(p[:2], tm.nodes[3].pose[:2], atol=1e-6)
        # image: 画像の類似度で切り替わる
        tr = SubgoalTracker(tm, TrackerConfig(image_threshold=0.95, confirm=1), "image")
        emb = [simple_embedding(n.image) for n in tm.nodes]
        for k in range(len(tm)):
            cur = emb[k]
            tr.update(None, lambda j, c=cur: float(np.dot(c, emb[j])))
        assert tr.done
    finally:
        shutil.rmtree(tmp)


# ---------------------------------------------------------------- controller
def test_trajectory_controller_reproduces_arc():
    cfg = controller.ControllerConfig(mode="trajectory", dt=1 / 3)
    for v, w in [(0.5, 0.0), (0.6, 0.4), (0.3, -0.8), (0.0, 0.5)]:
        t = (np.arange(8) + 1) / 3
        yaw = w * t
        if abs(w) < 1e-9:
            x, y = v * t, 0 * t
        else:
            x, y = v / w * np.sin(yaw), v / w * (1 - np.cos(yaw))
        vc, wc = controller.compute_command(np.stack([x, y, np.cos(yaw), np.sin(yaw)], 1), cfg)
        assert abs(vc - min(v, cfg.track_max_v)) < 0.02 or abs(vc / max(v, 1e-6) - wc / max(w, 1e-6)) < 0.05
        assert abs(wc * (v / max(vc, 1e-6) if vc > 0.01 and v > cfg.track_max_v else 1.0) - w) < 0.05


# ---------------------------------------------------------------- desk eval / engine
def test_desk_eval_oracle_follows_course():
    tmp = tempfile.mkdtemp()
    try:
        robot = robot_cfg()
        sa = read_streams(ListSource(course(stop_at=None)), robot, os.path.join(tmp, "a"), 10.0)
        build_topomap(sa, os.path.join(tmp, "topo"), 1.0)
        tm = load_topomap(os.path.join(tmp, "topo"))
        sb = read_streams(ListSource(course(stop_at=12.0, stop_len=2.0, slip=0.95, delay=0.2)), robot,
                          os.path.join(tmp, "b"), 3.0)
        ft, odom_xyz, src, _ = prepare_frames(sb, 3.0)
        assert src == "odom"
        nav = load_nav_config(os.path.join(CONFIGS, "navigator.yaml"))
        nav.engine.tracker.image_threshold = 0.9
        nav.engine.tracker.goal_image_threshold = 0.92
        rec, s = run_desk_eval(ft, OraclePolicy(ft, 3.0), robot, nav.engine, DeskEvalConfig(), tm, odom_xyz)
        assert s["subgoals"]["reach_check"] == "image_odom" and s["subgoals"]["reached_goal"]
        assert s["waypoints"]["ade_m"] < 1e-6
        assert s["commands"]["turn_direction_agreement"] > 0.95
        w6 = s["integrated_windows"]["6s"]
        assert w6["model"]["pos_err_median_m"] < 0.35                 # 6 秒積算して 35cm 以内
        # 積算の関数自体: 一定の指令で直進
        p = integrate_commands(np.arange(0, 3.01, 1 / 3), np.full(10, 0.3), np.zeros(10), (0, 0, 0))
        assert abs(p[-1, 0] - 0.9) < 1e-6
    finally:
        shutil.rmtree(tmp)


class _FakePolicy:
    """常に 0.4m/s で左へ緩く曲がる軌跡を返す."""
    meta = {"sample_rate": 3.0}

    calls = 0

    def predict(self, current, goal_image=None, goal_pose=None, instruction=None, modality="image"):
        from omnivla_real.policy_base import PolicyOutput
        _FakePolicy.calls += 1
        t = (np.arange(8) + 1) / 3
        yaw = 0.2 * t
        wps = np.stack([2 * np.sin(yaw), 2 * (1 - np.cos(yaw)), np.cos(yaw), np.sin(yaw)], 1)
        return PolicyOutput(wps, wps, 6, 0.01, np.zeros(4))

    def embed(self, image, cache_key=None):
        return simple_embedding(image)


class _HistPolicy:
    """edge と同じ観測履歴の扱いをする偽のポリシー (出力は入力画像の明るさで決まる)."""
    meta = {"sample_rate": 3.0, "metric_waypoint_spacing": 0.2}
    context_size, context_stride, metric_spacing = 3, 2, 0.2

    def __init__(self):
        from collections import deque
        self.history = deque(maxlen=self.context_size * self.context_stride + 1)

    def push(self, image):
        self.history.append(image)

    def reset_history(self):
        self.history.clear()

    def observation_window(self, current):
        hist = list(self.history)
        if not hist or hist[-1] is not current:
            hist.append(current)
        s = self.context_stride
        return [hist[max(0, len(hist) - 1 - s * k)] for k in range(self.context_size, -1, -1)]

    def predict(self, current, goal_image=None, goal_pose=None, instruction=None, modality="image", observations=None):
        from omnivla_real.policy_base import PolicyOutput
        if instruction == "boom":
            raise ValueError("boom")
        obs = observations if observations is not None else self.observation_window(current)
        f = [float(np.asarray(o, np.float64).mean()) for o in obs] + [float(np.asarray(goal_image).mean())]
        wps = np.zeros((8, 4))
        wps[:, 0] = (np.arange(8) + 1) * 0.1 + f[0] * 1e-3
        wps[:, 1] = (f[-1] - f[-2]) * 1e-3 + np.arange(8) * f[1] * 1e-5
        wps[:, 2] = 1.0
        gp = np.zeros(4) if goal_pose is None else np.r_[np.asarray(goal_pose, float), 0.0]
        return PolicyOutput(wps, wps * 2, 6, 0.0, gp, distance=sum(f))

    def embed(self, image, cache_key=None):
        return simple_embedding(image)


def test_remote_policy_matches_local():
    """推論サーバ経由 (ROS 1 コンテナ -> PyTorch コンテナ) でも、同じ入力なら同じ出力になる."""
    from omnivla_real.policy_base import load_policy
    from omnivla_real.remote import PolicyServer, RemotePolicy
    server = PolicyServer(_HistPolicy(), "fake", "127.0.0.1", 0, log=lambda s: None)
    server.start_background()
    try:
        url = f"http://127.0.0.1:{server.address[1]}"
        remote = load_policy("remote", url=url)
        assert isinstance(remote, RemotePolicy) and remote.history is not None and remote.meta["sample_rate"] == 3.0
        local = _HistPolicy()
        frames = [render(0.3 * k, 0.1 * k, 0.05 * k) for k in range(9)]
        goal = render(3, 0, 0)
        for k, img in enumerate(frames):
            remote.push(img)
            local.push(img)
            if k % 2 == 0:
                a = remote.predict(img, goal_image=goal, goal_pose=(1.0, 0.5, 0.2), modality="image_pose")
                b = local.predict(img, goal_image=goal, goal_pose=(1.0, 0.5, 0.2), modality="image_pose")
                assert np.array_equal(a.waypoints, b.waypoints) and np.array_equal(a.normalized, b.normalized)
                assert a.distance == b.distance and np.allclose(a.goal_pose_input, b.goal_pose_input)
        assert np.allclose(remote.embed(goal, cache_key="g"), simple_embedding(goal), atol=1e-6)
        assert "g" in remote._emb_cache
        try:
            remote.predict(frames[0], goal_image=goal, instruction="boom")
            raise AssertionError("server error should propagate")
        except RuntimeError as e:
            assert "boom" in str(e)
        # NavEngine からも使える (画像の履歴は sample_rate ごとに push される)
        tm = Topomap([GoalNode(render(0, 0, 0)), GoalNode(goal)], start=None)
        nav = load_nav_config(os.path.join(CONFIGS, "navigator.yaml"))
        eng = NavEngine(remote, tm, nav.engine, robot_cfg())
        eng.start()
        states = []
        for k in range(4):
            eng.on_image(10.0 + k / 3, np.asarray(frames[k]))
            r = eng.step(10.0 + k / 3 + 0.05)
            states.append(r.state)
            assert r.state != "running" or r.waypoints.shape == (8, 4)
        assert states[:2] == ["running", "running"] and len(remote.history) == 4
        remote.close()
    finally:
        server.shutdown()
    try:
        RemotePolicy(url, wait=0.5, log=lambda s: None)      # サーバが無い
        raise AssertionError("should fail")
    except ConnectionError:
        pass


def test_engine_and_runner():
    from omnivla_real import ros_common
    tm = Topomap([GoalNode(render(0, 0, 0)), GoalNode(render(2, 0, 0))], start=None)
    nav = load_nav_config(os.path.join(CONFIGS, "navigator.yaml"))
    nav.io.log_dir = tempfile.mkdtemp()
    eng = NavEngine(_FakePolicy(), tm, nav.engine, robot_cfg())
    assert eng.start() == "image"
    assert eng.step(0.0).state == "waiting_image"
    eng.on_image(1.0, lambda: np.asarray(render(0.5, 0, 0.1)))
    r = eng.step(1.1)
    assert r.state == "running" and abs(r.v - 0.4) < 0.02 and abs(r.w - 0.2) < 0.02 and r.debug is not None
    assert eng.step(2.0).state == "waiting_image"                    # 画像が古い
    # NavRunner (ROS ノードの中身): 推論スレッド + 指示値の保持
    clock = [10.0]
    runner = ros_common.NavRunner.__new__(ros_common.NavRunner)
    runner.robot, runner.cfg, runner.clock, runner.log = robot_cfg(), nav, (lambda: clock[0]), (lambda s: None)
    runner.policy = _FakePolicy()
    runner.engine = NavEngine(runner.policy, tm, nav.engine, runner.robot)
    runner.lock, runner.cmd, runner.cmd_time, runner.stopped_at = threading.Lock(), (0.0, 0.0), -1e9, -1e9
    runner.last_result, runner.topomap_path, runner._thread, runner._alive, runner.on_result = None, "x", None, True, None
    assert runner.command() is None                                  # 開始前は何も出さない
    runner.engine.on_image(clock[0], np.asarray(render(0.5, 0, 0)))
    assert runner.start()
    for _ in range(500):                                             # CI の遅いマシンでも待てるように長めに
        if runner.cmd_time > 0:
            break
        time.sleep(0.02)
    cmd = runner.command()
    assert cmd is not None and abs(cmd[0] - 0.4) < 0.02
    n = _FakePolicy.calls
    time.sleep(0.2)
    assert _FakePolicy.calls == n                                    # 新しい画像が来るまで推論しない
    runner.engine.on_image(clock[0] + 0.1, np.asarray(render(0.6, 0, 0)))
    for _ in range(500):
        if _FakePolicy.calls > n:
            break
        time.sleep(0.01)
    time.sleep(0.1)
    assert _FakePolicy.calls == n + 1                                # 1 枚につき 1 回
    clock[0] += 5.0                                                  # 推論結果が古くなったら 0
    runner.engine.on_image(clock[0] + 100, np.asarray(render(0.5, 0, 0)))
    assert json.loads(runner.status_json())["state"] == "running"
    runner.stop("test")
    assert runner.command() == (0.0, 0.0)
    clock[0] += 2.0
    assert runner.command() is None
    runner.shutdown()
    logs = os.listdir(nav.io.log_dir)
    assert len(logs) == 1 and os.path.exists(os.path.join(nav.io.log_dir, logs[0], "summary.json"))
    # 走行ログの解析ツール
    sys.path.insert(0, os.path.join(REPO, "tools"))
    import plot_nav_log
    plot_nav_log.main([os.path.join(nav.io.log_dir, "latest")])
    assert os.path.exists(os.path.join(nav.io.log_dir, logs[0], "overview.png"))


def test_configs_load():
    nav = load_nav_config(os.path.join(CONFIGS, "navigator.yaml"), {"model.model": "7b", "engine.modality": "image"})
    assert nav.model.model == "7b" and nav.engine.controller.mode == "trajectory"
    r = RobotConfig.load(os.path.join(CONFIGS, "robot.yaml"))
    assert r.topics.cmd == "/cmd_vel" and len(r.image.crop) == 4
    from omnivla_real.runio import load_convert_config
    c = load_convert_config(os.path.join(CONFIGS, "convert.yaml"), {"pose_source": "cmd"})
    assert c.pose_source == "cmd" and c.sample_rate == 3.0


# ---------------------------------------------------------------- edge (torch がある場合のみ)
def test_edge_model_cpu():
    try:
        import torch  # noqa: F401
        import efficientnet_pytorch  # noqa: F401
    except ImportError:
        print("  (skip: torch / efficientnet_pytorch not installed)")
        return
    from omnivla_real.edge import EdgePolicy, EdgePolicyConfig, build_edge_model, make_edge_batch
    model = build_edge_model()
    pol = EdgePolicy(EdgePolicyConfig(device="cpu", clip_type=None, metric_waypoint_spacing=0.2), model=model)
    for k in range(8):
        pol.push(render(0.1 * k, 0, 0))
    out = pol.predict(render(0.8, 0, 0), goal_image=render(2, 0, 0))
    assert out.waypoints.shape == (8, 4) and np.isfinite(out.waypoints).all()
    assert np.allclose(np.linalg.norm(out.waypoints[:, 2:], axis=1), 1.0, atol=1e-4)
    emb = pol.embed(render(0, 0, 0))
    assert emb.ndim == 1 and abs(np.linalg.norm(emb) - 1) < 1e-4
    # 学習 1 step (勾配が流れる)
    sample = make_edge_batch([render(0, 0, 0)] * 6, render(1, 0, 0), np.zeros(4, np.float32), 6, torch.zeros(512))
    batch = {k: v.unsqueeze(0).repeat(2, *([1] * v.dim())) for k, v in sample.items()}
    model.train()
    from omnivla_real.edge import edge_forward
    actions, _ = edge_forward(model, batch, torch.device("cpu"))
    loss = (actions.float() ** 2).mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


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
