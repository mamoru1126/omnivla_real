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
from omnivla_real.bag.reader import BagMessage, ListSource  # noqa: E402
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
from omnivla_real.trajectory_io import find_trajectories, load_meta, load_trajectory, trajectory_cut_end  # noqa: E402
from synthetic import header, jpeg_bytes, quat, render, simulate_course, stamp  # noqa: E402

CONFIGS = os.path.join(REPO, "tests", "configs")   # configs/ はロボットに合わせて書き換えるので、テストは固定の写しを使う


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


def test_foreign_camera_clock_and_missing_cmd():
    """カメラが別の時計 (0 = 1970 年, 起動からの時間など) を付けている bag と、/cmd_vel が無い bag (Orin で録った bag)."""
    from omnivla_real.extract import check_clock
    alt = []
    for m in course():
        if m.topic == "/cmd_vel":
            continue
        if m.topic == "/camera/image_raw/compressed":
            m = BagMessage(m.topic, m.msgtype, m.t_bag,
                           NS(header=header(m.t_bag - 950.0, "camera"), format=m.msg.format, data=m.msg.data))
        alt.append(m)
    tmp = tempfile.mkdtemp()
    try:
        s = read_streams(ListSource(alt), robot_cfg(), os.path.join(tmp, "_frames", "r"), 3.0, name="r")
        assert s.time_source_used == "bag" and any("another clock" in n for n in s.notes)
        assert any("/cmd_vel" in n for n in s.notes) and len(s.cmd_t) == 0
        assert abs(s.image_t[0] - s.odom_t[0]) < 1.0              # 画像もオドメトリも記録時刻にそろう
        assert any("another clock" in w for w in check_clock(s))
        res = convert_streams(s, ConvertConfig(), os.path.join(tmp, "ds"), "r")
        assert len(res["trajectories"]) == 2                        # 普通の bag と同じく 2 区間
        tr = load_trajectory(res["trajectories"][0])
        assert np.isnan(tr["cmd_v"]).all() and np.isfinite(tr["position"]).all()
        s2 = read_streams(ListSource(course()), robot_cfg(), None, 3.0, decode_images=False)
        assert s2.time_source_used == "auto" and not s2.notes       # 普通の bag は今までどおり header の時刻
    finally:
        shutil.rmtree(tmp)


def _convert_chunked(tmp: str, chunk_sec: float = 6.0):
    s = read_streams(ListSource(course()), robot_cfg(), os.path.join(tmp, "_frames", "r"), 3.0, name="r")
    convert_streams(s, ConvertConfig(chunk_sec=chunk_sec), os.path.join(tmp, "ds"), "r")
    return find_trajectories(os.path.join(tmp, "ds"))


def test_chunk_cut_end_is_marked():
    tmp = tempfile.mkdtemp()
    try:
        dirs = _convert_chunked(tmp)
        by_seg = {}
        for d in dirs:
            m = load_meta(d)
            by_seg.setdefault(m["segment"], []).append((m["chunk"], m["cut_end"], d))
        assert max(len(v) for v in by_seg.values()) >= 2
        for items in by_seg.values():
            items.sort()
            # 区間の最後の塊だけが本当の終わり (止まった所). それ以外は区切っただけ
            assert [c for _, c, _ in items] == [True] * (len(items) - 1) + [False]
            for _, cut, d in items:   # 古い変換 (meta に cut_end が無い) でも次の塊があるかで分かる
                m = load_meta(d)
                m.pop("cut_end")
                assert trajectory_cut_end(d, m) == cut
        mask = data_utils.cut_end_mask(20, 8, 1)
        assert not mask[:12].any() and mask[12:].all()
        assert data_utils.cut_end_mask(20, 8, 2)[4:].all() and not data_utils.cut_end_mask(20, 8, 2)[:4].any()
    finally:
        shutil.rmtree(tmp)


def test_smooth_trajectory_spreads_tapped_turns():
    # 0.25 m/フレームで直進しながら、スティックを 3 回だけ短く倒して 45 度曲がる (手動操作)
    n = 60
    w = np.zeros(n)
    w[[20, 24, 28]] = math.radians(15)
    yaw = np.cumsum(w)
    pos = np.zeros((n, 2))
    for k in range(1, n):
        pos[k] = pos[k - 1] + 0.25 * np.array([math.cos(yaw[k - 1]), math.sin(yaw[k - 1])])
    p, yw = data_utils.smooth_trajectory(pos, yaw, 2.0)
    assert abs((yw[-1] - yw[0]) - (yaw[-1] - yaw[0])) < 1e-6           # 曲がる量の合計は同じ
    assert np.abs(np.diff(yw)).max() < 0.5 * np.abs(np.diff(yaw)).max()  # 1 フレームあたりは均される
    assert np.linalg.norm(p[-1] - pos[-1]) < 0.3
    line = np.stack([np.arange(n) * 0.25, np.zeros(n)], axis=1)          # 直線はそのまま (端も)
    p2, y2 = data_utils.smooth_trajectory(line, np.zeros(n), 2.0)
    assert np.allclose(p2, line, atol=1e-9) and np.allclose(y2, 0)
    p3, y3 = data_utils.smooth_trajectory(pos, yaw, 0.0)
    assert np.allclose(p3, pos) and np.allclose(y3, yaw)
    # 正解 (この先 8 点) で見ると、ならした方が曲がり始めの手前から少しずつ曲がる
    raw = data_utils.compute_action_targets(pos, yaw, 14, metric_spacing=0.25)
    smo = data_utils.compute_action_targets(p, yw, 14, metric_spacing=0.25)
    assert abs(raw[2, 3]) < 1e-9 < smo[2, 3]


def test_goal_sampling_by_distance():
    pos = np.stack([np.arange(40) * 0.1, np.zeros(40)], axis=1)          # 0.1 m/フレーム (ゆっくり)
    arc = data_utils.arc_length(pos)
    assert abs(arc[-1] - 3.9) < 1e-9
    cfg = data_utils.GoalSamplingConfig(image_goal_dist_m=(0.45, 2.05), modality_weights={6: 1.0})
    rng = np.random.default_rng(0)
    offs = [data_utils.sample_goal_offset(rng, 0, 40, 6, cfg, arc) for _ in range(300)]
    assert min(offs) == 5 and max(offs) == 20                           # 0.5〜2.0 m 先
    assert data_utils.sample_goal_offset(rng, 37, 40, 6, cfg, arc) == 2   # 終わり近く: 届く所で一番先
    mod, goal_t, a, gp = data_utils.make_targets(rng, pos, np.zeros(40), 0, cfg, 0.1)
    assert mod == 6 and 5 <= goal_t <= 20 and a.shape == (8, 4)
    frames = data_utils.GoalSamplingConfig(modality_weights={6: 1.0})    # 指定なしは今までどおりフレーム数
    assert all(2 <= data_utils.sample_goal_offset(rng, 0, 40, 6, frames, arc) <= 30 for _ in range(50))


def test_yaw_shift_augmentation_turns_back_to_the_path():
    line = np.stack([np.arange(12) * 0.2, np.zeros(12)], axis=1)
    a = data_utils.compute_action_targets(line, np.zeros(12), 0, metric_spacing=0.2)   # 真っ直ぐ進む
    gp = np.array([5.0, 0.0, 1.0, 0.0], np.float32)
    a2, g2 = data_utils.rotate_labels(*data_utils.rotate_labels(a, gp, 0.2), -0.2)
    assert np.allclose(a2, a, atol=1e-5) and np.allclose(g2, gp, atol=1e-5)
    a3, g3 = data_utils.rotate_labels(a, gp, 0.2)          # 左を向いたロボットから見ると、元の経路は右
    assert a3[-1, 1] < 0 and a3[-1, 3] < 0 and g3[1] < 0
    assert abs(math.atan2(a3[-1, 1], a3[-1, 0]) + 0.2) < 1e-5
    # 切り抜き: 5 度左を向く = 窓を 160*tan(5deg) = 14px 左へ. ゴールは同じ大きさで中央
    cb, gb, applied = data_utils.yaw_shift_boxes(320, 240, (16, 0, 304, 240), math.radians(5), math.radians(90))
    assert cb == (2, 0, 290, 240) and gb == (16, 0, 304, 240) and abs(math.degrees(applied) - 5.0) < 0.05
    cb, gb, _ = data_utils.yaw_shift_boxes(320, 240, (5, 0, 315, 240), math.radians(-5), math.radians(90))
    assert cb == (28, 0, 320, 240) and gb == (14, 0, 306, 240)   # 余白が足りない: 窓を細くしてずらす
    # 画像の中央の縦線 (= 元の進む先) が右に見えたら、正解も右へ曲がる
    img = np.zeros((240, 320, 3), np.uint8)
    img[:, 158:162] = 255
    cur = goal = Image.fromarray(img)
    aug = data_utils.AugmentConfig(crop_v=0.0, crop_h=0.05, flip_prob=0.0, color_jitter=0.0, yaw_deg=8.0)
    seen = 0
    for seed in range(30):
        c, g, a4, gp4 = data_utils.augment_pair(np.random.default_rng(seed), cur, goal, a, gp, aug, True)
        shift = int(np.argmax(np.asarray(c.convert("L"), float).mean(axis=0))) - (c.width - 1) / 2
        assert abs(int(np.argmax(np.asarray(g.convert("L"), float).mean(axis=0))) - (g.width - 1) / 2) <= 2
        if abs(shift) > 3:
            seen += 1
            assert np.sign(shift) == -np.sign(a4[-1, 1]) == -np.sign(gp4[1])
    assert seen > 10
    off = data_utils.AugmentConfig(crop_v=0.0, crop_h=0.05, flip_prob=0.0, color_jitter=0.0)   # 既定は回さない
    _, _, a5, _ = data_utils.augment_pair(np.random.default_rng(0), cur, goal, a, gp, off, True)
    assert np.allclose(a5, a)


def test_nav_dataset_cut_ends_smoothing_and_goal_distance():
    try:
        import torch  # noqa: F401
    except ImportError:
        print("  (skip: torch not installed)")
        return
    sys.path.insert(0, os.path.join(REPO, "training"))
    from nav_dataset import BaseNavDataset, NavDatasetConfig
    tmp = tempfile.mkdtemp()
    try:
        dirs = _convert_chunked(tmp)
        ds = BaseNavDataset(dirs, NavDatasetConfig(metric_waypoint_spacing=0.2), train=False)
        assert ds.num_cut_dropped > 0
        for ti, t in ds.index:   # 区切っただけの塊では、正解 8 点が全部その塊の中にある
            tr = ds.trajs[ti]
            if trajectory_cut_end(tr["dir"]):
                assert t + 8 <= tr["n"] - 1
        # bag ごとの倍率: 同じ走行を別名 (manual_x) でもう 1 本入れて、そちらを 3 倍引く
        s2 = read_streams(ListSource(course()), robot_cfg(), os.path.join(tmp, "_frames", "m"), 3.0, name="manual_x")
        convert_streams(s2, ConvertConfig(chunk_sec=6.0), os.path.join(tmp, "ds"), "manual_x")
        ds2 = BaseNavDataset(find_trajectories(os.path.join(tmp, "ds")), NavDatasetConfig(metric_waypoint_spacing=0.2),
                             train=False)
        counts = ds2.bag_counts()
        assert counts["r"] == counts["manual_x"] > 0
        is_m = ds2.bag_factors({"manual": 3.0}) == 3.0
        assert abs(ds2.sample_weights(0.0)[is_m].sum() - 0.5) < 1e-9
        assert abs(ds2.sample_weights(0.0, bag_weights={"manual": 3.0})[is_m].sum() - 0.75) < 1e-9
        assert abs(ds2.sample_weights(0.5, bag_weights={"manual": 3.0}).sum() - 1.0) < 1e-9
        cfg = NavDatasetConfig(metric_waypoint_spacing=0.2, label_smooth_sec=0.7,
                               goal=data_utils.GoalSamplingConfig(image_goal_dist_m=(0.5, 2.0),
                                                                 modality_weights={6: 1.0}))
        ds2 = BaseNavDataset(dirs, cfg, train=False)
        for i in range(0, len(ds2), 7):
            tr, t, _, mod, goal_t, actions, _ = ds2.targets(i)
            assert np.isfinite(actions).all() and mod == 6
            d = tr["arc"][goal_t] - tr["arc"][t]
            assert d <= 2.0 + 1e-6 and (d >= 0.5 - 1e-6 or tr["arc"][-1] - tr["arc"][t] < 0.5
                                         or goal_t == tr["n"] - 1)
    finally:
        shutil.rmtree(tmp)


def test_turn_gain_and_driving_style():
    from omnivla_real.desk_eval import turn_gain
    t = np.arange(300)
    gt = 30 * np.sin(t / 15.0)
    pred = 0.7 * np.roll(gt, 1)          # 曲がり 3 割不足, 1 フレーム遅れ
    r = turn_gain(gt.tolist(), pred.tolist())
    assert abs(r["turn_gain"] - 0.7) < 0.03 and r["turn_lag_frames"] == -1 and r["turn_gain_n"] > 100
    assert turn_gain([None, 1.0], [0.0, None])["turn_gain"] is None
    sys.path.insert(0, os.path.join(REPO, "tools"))
    from turn_stats import driving_style
    v = np.full(100, 0.8)
    taps = np.zeros(100)
    taps[::10] = 1.24                     # 0 か倒し切り
    s = driving_style([v], [taps], [v], [np.zeros(100)])
    assert abs(s["speed_median"] - 0.8) < 1e-9 and abs(s["steer_ratio"] - 0.1) < 1e-9
    assert s["full_stick_ratio"] == 1.0
    smooth = 0.3 * np.sin(np.arange(100) / 5.0)
    assert driving_style([v], [smooth], [v], [smooth])["full_stick_ratio"] < 0.5


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
def test_rebuild_step_times_for_old_logs():
    """古いログ (sim_time を有効数字 5 桁で書いていた) は events.log のサブゴール切り替え時刻からステップの時刻を推定する."""
    from omnivla_real.navlog import rebuild_step_times, times_usable
    true = [1.76e9 + k / 6.0 for k in range(120)]                 # 6 Hz
    sub = [k // 10 for k in range(120)]
    events = [(0.0, "start: 12 subgoals")] + [(true[k], f"subgoal {sub[k] - 1} -> {sub[k]}/11 [image]")
                                               for k in range(1, 120) if sub[k] != sub[k - 1]]
    assert not times_usable([float(f"{t:.5g}") for t in true]) and times_usable(true)
    est = rebuild_step_times(sub, events)
    assert max(abs(a - b) for a, b in zip(est, true)) < 1e-3


def test_image_odom_does_not_run_ahead_on_long_course():
    """長いコースで、オドメトリの位置 (x, y) が向きのドリフトで大きくずれ、画像の類似度がどこでも高くても、
    image_odom のサブゴールが実際の位置より先走らない (道のりで候補を絞り、手前のノードでも道のりを直す)."""
    rng = np.random.default_rng(0)
    n = 200
    nodes = [GoalNode(Image.new("RGB", (2, 2)), (float(k), 0.0, 0.0)) for k in range(n)]   # 1 m 間隔の直線
    tm = Topomap(nodes, frame="odom", start=nodes[0].pose)
    tr = SubgoalTracker(tm, TrackerConfig(reach_check="image_odom"), "image_odom")
    travel, leads = 0.0, []
    for i in range(1200):
        s_true = i * 0.2                                    # 0.6 m/s, 3 Hz
        travel = s_true * 1.02                              # オドメトリの道のりは 2% 多め
        th = math.radians(0.1) * s_true                     # 向きのドリフト (200 m で 20 度)
        pose = (s_true * math.cos(th), s_true * math.sin(th), th)
        sims = lambda j: 0.93 - 0.01 * abs(j - s_true) + rng.normal(0, 0.03)   # noqa: E731  どこでもよく似ている
        tr.update(pose, sims, travel)
        truth = min(n - 1, int(math.floor(s_true)) + 1)
        leads.append(tr.index - truth)
        if tr.done:
            break
    assert s_true > n - 5, f"finished too early at {s_true:.0f} m of {n - 1} m"
    assert max(leads) <= 4 and np.median(np.abs(leads)) <= 2, (max(leads), np.median(leads))


def test_image_odom_keeps_up_when_images_are_ambiguous():
    """近くも遠くも同じくらい似ていて、画像の判定が確定しない (一番似たノードが入れ替わり続ける) 場合でも、
    道のりで通り過ぎたら進める (サブゴールが止まって遅れない)."""
    rng = np.random.default_rng(1)
    n = 200
    nodes = [GoalNode(Image.new("RGB", (2, 2)), (float(k), 0.0, 0.0)) for k in range(n)]
    tr = SubgoalTracker(Topomap(nodes, frame="odom", start=nodes[0].pose), TrackerConfig(reach_check="image_odom"),
                        "image_odom")
    leads = []
    for i in range(1100):
        s = i * 0.2
        tr.update(None, lambda j: 0.91 + rng.normal(0, 0.03), s)   # noqa: B023
        leads.append(tr.index - min(n - 1, int(math.floor(s)) + 1))
        if tr.done:
            break
    assert tr.done and s > n - 5, s
    assert min(leads) >= -3, min(leads)


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
        assert s["latency"]["n"] > 0 and s["latency"]["median_s"] >= 0  # 推論時間 (Jetson での確認用)
        sys.path.insert(0, os.path.join(REPO, "tools"))
        from desk_eval import text_report  # tools/desk_eval.py
        s["model"] = {"server": {"model": "edge", "checkpoint": "/runs/x"}}
        txt = text_report(s)
        assert "推論時間: 中央値" in txt and "推論サーバのモデル: edge /runs/x" in txt
        # 積算の関数自体: 一定の指令で直進
        p = integrate_commands(np.arange(0, 3.01, 1 / 3), np.full(10, 0.3), np.zeros(10), (0, 0, 0))
        assert abs(p[-1, 0] - 0.9) < 1e-6
    finally:
        shutil.rmtree(tmp)


def test_eval_localization_is_truth_only():
    """topics.eval_localization: 自己位置は答え合わせだけ (topomap は地図の座標, ナビは実機と同じく画像 + 道のり)."""
    tmp = tempfile.mkdtemp()
    try:
        robot = RobotConfig.load(os.path.join(CONFIGS, "robot.yaml"),
                                 {"topics": {"localization": "", "eval_localization": "/localization"}})
        sa = read_streams(ListSource(course(stop_at=None)), robot, os.path.join(tmp, "a"), 10.0)
        assert sa.loc_eval_only and len(sa.loc_t) > 10
        build_topomap(sa, os.path.join(tmp, "topo"), 1.0)
        tm = load_topomap(os.path.join(tmp, "topo"))
        assert tm.frame == "map"
        nav = load_nav_config(os.path.join(CONFIGS, "navigator.yaml"))
        nav.engine.tracker.image_threshold = 0.9
        nav.engine.tracker.goal_image_threshold = 0.92
        run_b = course(stop_at=12.0, stop_len=2.0, slip=0.95, delay=0.2)
        sb = read_streams(ListSource(run_b), robot, os.path.join(tmp, "b"), 3.0)
        ft, odom_xyz, src, _ = prepare_frames(sb, 3.0)
        assert src == "localization" and ft.loc_eval_only
        rec, s = run_desk_eval(ft, OraclePolicy(ft, 3.0), robot, nav.engine, DeskEvalConfig(), tm, odom_xyz)
        g = s["subgoals"]
        assert g["reach_check"] == "image_odom" and g["reached_goal"]       # ナビには自己位置を渡さない
        assert g["index_minus_truth_abs_p90"] is not None and g["index_minus_truth_abs_p90"] <= 3
        # 自己位置をナビにも使う構成 (topics.localization) は今までどおり pose で判定
        robot2 = RobotConfig.load(os.path.join(CONFIGS, "robot.yaml"), {"topics": {"localization": "/localization"}})
        sb2 = read_streams(ListSource(run_b), robot2, os.path.join(tmp, "c"), 3.0)
        ft2, odom2, _, _ = prepare_frames(sb2, 3.0)
        assert not ft2.loc_eval_only
        _, s2 = run_desk_eval(ft2, OraclePolicy(ft2, 3.0), robot2, nav.engine, DeskEvalConfig(), tm, odom2)
        assert s2["subgoals"]["reach_check"] == "pose"
    finally:
        shutil.rmtree(tmp)


def test_pose_message_without_header():
    """geometry_msgs/Pose (時刻なし) も自己位置として読める. 時刻は記録時刻になる."""
    from omnivla_real.bag.messages import header_stamp
    m = NS(position=NS(x=1.0, y=2.0, z=0.0), orientation=quat(0.5))
    x, y, yaw = pose_values(m, "geometry_msgs/msg/Pose")
    assert (x, y) == (1.0, 2.0) and abs(yaw - 0.5) < 1e-9 and header_stamp(m) is None
    alt = [BagMessage("/loc_pose", "geometry_msgs/msg/Pose", m.t_bag, m.msg.pose) if m.topic == "/localization" else m
           for m in course()]
    robot = RobotConfig.load(os.path.join(CONFIGS, "robot.yaml"), {"topics": {"eval_localization": "/loc_pose"}})
    s = read_streams(ListSource(alt), robot, None, 3.0, decode_images=False)
    assert s.loc_eval_only and len(s.loc_t) > 10 and abs(s.loc_x[0] - 3.0) < 0.05


def test_run_eval_cross_track_and_corners():
    from omnivla_real.run_eval import CoursePath, evaluate_run, report_text, running_mask
    # コース: 10 m 直進 -> 半径 2 m で左へ 90 度 -> 10 m
    a = [(0.25 * k, 0.0) for k in range(41)]
    arc = [(10 + 2 * math.sin(q), 2 - 2 * math.cos(q)) for q in np.linspace(0, math.pi / 2, 20)[1:]]
    b = [(12.0, 2 + 0.25 * k) for k in range(1, 41)]
    path = CoursePath.from_points(np.array(a + arc + b))
    cs = path.corners(20.0)
    assert len(cs) == 1 and 80 < cs[0]["turn_deg"] < 100 and cs[0]["s0"] < 11 < 13 < cs[0]["s1"]
    # 走行: いつもはコースの 0.3 m 右, カーブでは 0.8 m 外 (= 右) に膨らむ
    ss = np.arange(0.0, path.length, 0.05)
    hh = path.heading_at(ss)
    px = np.interp(ss, path.s, path.x)
    py = np.interp(ss, path.s, path.y)
    in_curve = (ss > 10.5) & (ss < 12.6)
    off = np.where(in_curve, -0.8, -0.3)
    x, y = px - off * np.sin(hh), py + off * np.cos(hh)
    t = ss / 0.3
    run = np.ones(len(t), bool)
    run[:20] = False                                    # 最初の 1 m は手で動かした
    summary, per = evaluate_run(path, t, x, y, hh, running=run, speed_cap=0.3)
    assert summary["progress"]["completion"] > 0.98
    xs = per["xtrack"][~in_curve & run]
    assert np.allclose(np.median(xs), -0.3, atol=0.02)
    c = summary["corners"][0]
    assert abs(c["max_outward_m"] - 0.8) < 0.05 and c["reached_end"]
    assert summary["running"]["samples"] == run.sum() and summary["all"]["samples"] == len(t)
    txt = report_text(summary, "t")
    assert "外 +0.8" in txt and "進んだ割合 10" in txt
    # 行って戻るコース (2 本が 0.6 m 離れて並ぶ): 戻りの区間を走っても行きの区間に飛ばない
    out_back = np.array([(0.25 * k, 0.0) for k in range(41)] + [(10.0, 0.3)] + [(10 - 0.25 * k, 0.6) for k in range(41)])
    p2 = CoursePath.from_points(out_back)
    xr = np.concatenate([np.linspace(0, 10, 101), np.linspace(10, 0, 101)])
    yr = np.concatenate([np.full(101, 0.1), np.full(101, 0.5)])
    yawr = np.concatenate([np.zeros(101), np.full(101, math.pi)])
    _, per2 = evaluate_run(p2, np.arange(202) * 0.3, xr, yr, yawr)
    assert per2["s"][-1] > 19.0 and np.all(np.diff(per2["s"]) > -0.5)
    # /omnivla/status の running
    m = running_mask(np.array([0.5, 1.5, 2.5, 10.0]), [(0.0, "idle"), (1.0, "running"), (2.0, "running")])
    assert m.tolist() == [False, True, True, False]
    assert running_mask(np.array([1.0]), []) is None


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
        # 画像は 1 度だけ送る: 過去画像・サブゴール画像は 2 回目からは key だけ
        assert remote.protocol == 2 and remote.bytes_sent < 9 * 64 * 48 * 3 * 2
        server._images.clear()                                  # サーバ再起動相当 -> 409 -> 送り直し
        a = remote.predict(frames[-1], goal_image=goal)
        b = local.predict(frames[-1], goal_image=goal)
        assert np.array_equal(a.waypoints, b.waypoints)
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


def test_remote_server_preprocess_and_keys():
    """C++ ノードの送り方: カメラの JPEG / raw をそのまま送り、サーバが robot.yaml の前処理をかける."""
    import http.client
    from fake_policy import FakeEdgePolicy
    from omnivla_real.remote import PolicyServer, pack, unpack
    cfg = ImageConfig(crop=[0.1, 0.2, 0.05, 0.05], width=48)
    server = PolicyServer(FakeEdgePolicy(context_size=2), "fake", "127.0.0.1", 0, log=lambda s: None, image_cfg=cfg)
    server.start_background()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.address[1], timeout=5)

        def post(path, header, arrays):
            conn.request("POST", path, pack(header, arrays))
            r = conn.getresponse()
            return r.status, unpack(r.read())

        cam_a, cam_b = render(1, 0, 0.3, (96, 72)), render(1.5, 0.2, -0.2, (96, 72))
        jpg = jpeg_bytes(cam_a)
        bgr = np.asarray(cam_b)[..., ::-1].copy()
        goal = preprocess_image(render(3, 0, 0, (96, 72)), cfg)       # topomap の画像は前処理済み
        st, (h, arr) = post("/predict", {
            "images": [{"key": "f1", "array": 0, "kind": "encoded"},
                       {"key": "f2", "array": 1, "kind": "raw", "encoding": "bgr8", "width": 96, "height": 72,
                        "step": 96 * 3},
                       {"key": "g0", "array": 2}],
            "current": 0, "goal": 2, "observations": [1, 1, 0], "want_preview": True},
            [np.frombuffer(jpg.tobytes(), np.uint8), bgr.reshape(-1), np.asarray(goal)])
        assert st == 200, h
        local = FakeEdgePolicy(context_size=2)
        cur = preprocess_image(Image.open(io.BytesIO(jpg.tobytes())), cfg)
        prev = preprocess_image(cam_b, cfg)
        exp = local.predict(cur, goal_image=goal, observations=[prev, prev, cur])
        assert np.array_equal(arr[0], exp.waypoints)
        prev_img = Image.open(io.BytesIO(arr[h["preview"]].tobytes()))
        assert prev_img.size == cur.size == tuple(h["preview_size"])
        # 2 回目からは key だけ (中身を送らない)
        st, (h2, arr2) = post("/predict", {"images": [{"key": "f1"}, {"key": "g0"}, {"key": "f2"}],
                                           "current": 0, "goal": 1, "observations": [2, 2, 0]}, [])
        assert st == 200 and np.array_equal(arr2[0], exp.waypoints)
        st, (h3, _) = post("/predict", {"images": [{"key": "nope"}], "current": 0}, [])
        assert st == 409 and h3["missing"] == ["nope"]
        st, (h4, emb) = post("/embed", {"images": [{"key": "g0"}], "image": 0}, [])
        assert st == 200 and np.allclose(emb[0], simple_embedding(goal), atol=1e-6)
        conn.close()
    finally:
        server.shutdown()


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


def test_nav_overrides_env_nested():
    """NAV_OVERRIDES (環境変数) の "engine.tracker.x=..." を Python のノードでも使える."""
    from omnivla_real.nav_config import load_nav_config
    from omnivla_real.ros_common import parse_overrides
    o = parse_overrides("engine.max_image_age=1.5 engine.tracker.image_threshold=0.93 io.log_images=false bogus")
    assert o == {"engine.max_image_age": 1.5, "engine.tracker.image_threshold": 0.93, "io.log_images": False}
    c = load_nav_config(os.path.join(CONFIGS, "navigator.yaml"), o)
    assert c.engine.max_image_age == 1.5 and c.engine.tracker.image_threshold == 0.93 and c.io.log_images is False


def test_image_age_from_receive():
    """image_age_from=receive: 画像の時刻 (別の PC の時計・カメラの時刻) が大きくずれていても、受け取ってすぐなら使う."""
    tm = Topomap([GoalNode(render(0, 0, 0)), GoalNode(render(2, 0, 0))], start=None)
    nav = load_nav_config(os.path.join(CONFIGS, "navigator.yaml"))
    nav.io.log_dir = tempfile.mkdtemp()
    eng = NavEngine(_FakePolicy(), tm, nav.engine, robot_cfg())
    eng.start()
    eng.on_image(1000.0, lambda: np.asarray(render(0.5, 0, 0.1)))     # 画像の時刻が今 (t=5000) より 4000 秒前
    assert eng.step(5000.0).state == "waiting_image"                  # stamp: 古いと判定
    assert eng.status()["step_state"] == "waiting_image" and eng.status()["image_age"] > 3000
    nav2 = load_nav_config(os.path.join(CONFIGS, "navigator.yaml"), {"engine.image_age_from": "receive"})
    nav2.io.log_dir = tempfile.mkdtemp()
    eng = NavEngine(_FakePolicy(), tm, nav2.engine, robot_cfg())
    eng.start()
    eng.on_image(1000.0, lambda: np.asarray(render(0.5, 0, 0.1)))
    r = eng.step(5000.0)
    st = eng.status()
    assert r.state == "running" and st["step_state"] == "running" and st["image_age"] < 1.0
    assert st["similarity"] is not None and st["image_age_from"] == "receive" and st["image_stamp_offset"] > 1e6


def test_odometry_arriving_after_start_switches_to_image_odom():
    """start の時点でオドメトリが無い (届くのが遅れた) と auto は image になる. 届いたら image_odom に切り替える."""
    tm = Topomap([GoalNode(render(k, 0, 0), pose=(float(k), 0.0, 0.0), s=float(k)) for k in range(6)],
                 start=(0.0, 0.0, 0.0))
    nav = load_nav_config(os.path.join(CONFIGS, "navigator.yaml"), {"engine.image_age_from": "receive"})
    nav.io.log_dir = tempfile.mkdtemp()
    eng = NavEngine(_FakePolicy(), tm, nav.engine, robot_cfg())
    assert eng.start() == "image"                                   # まだオドメトリが無い
    eng.on_image(1.0, lambda: np.asarray(render(0.2, 0, 0)))
    eng.step(1.0)
    eng.on_odom(1.1, (5.0, 3.0, 0.5))                               # 遅れて届いた
    st = eng.status()
    assert st["reach_check"] == "image_odom" and eng.tracker.mode == "image_odom"
    assert eng.odom_start == (5.0, 3.0, 0.5) and eng.travel == 0.0
    eng.on_odom(1.2, (5.5, 3.0, 0.5))
    assert abs(eng.travel - 0.5) < 1e-9
    eng.on_image(1.3, lambda: np.asarray(render(0.6, 0, 0)))
    r = eng.step(1.3)
    assert r.state == "running" and any("odometry arrived after start" in e for e in r.events)
    # 最初からオドメトリがあれば今までどおり
    eng2 = NavEngine(_FakePolicy(), tm, nav.engine, robot_cfg())
    eng2.on_odom(0.0, (0.0, 0.0, 0.0))
    assert eng2.start() == "image_odom"


def test_trajectory_turn_gain_and_keep_w():
    from omnivla_real.controller import ControllerConfig, trajectory_command
    # 0.45 m/s, 0.6 rad/s の円弧
    v0, w0, dt = 0.45, 0.6, 1.0 / 3.0
    wps = []
    for k in range(8):
        t = (k + 1) * dt
        th = w0 * t
        r = v0 / w0
        wps.append([r * np.sin(th), r * (1 - np.cos(th)), np.cos(th), np.sin(th)])
    wps = np.array(wps)
    base = ControllerConfig(track_max_v=0.3, track_max_w=1.0)
    v, w = trajectory_command(wps, base)
    assert abs(v - 0.3) < 0.02 and abs(w - 0.4) < 0.03          # 曲率を保って両方落ちる
    v, w = trajectory_command(wps, ControllerConfig(track_max_v=0.3, track_max_w=1.0, track_keep_w=True))
    assert abs(v - 0.3) < 0.02 and abs(w - 0.6) < 0.03          # w は落とさない
    v, w = trajectory_command(wps, ControllerConfig(track_max_v=0.6, track_max_w=1.0, track_w_gain=1.5))
    assert abs(w - 0.9) < 0.04
    v, w = trajectory_command(wps, ControllerConfig(track_max_v=0.6, track_max_w=0.7, track_w_gain=1.5,
                                                    track_keep_w=True))
    assert abs(w - 0.7) < 1e-6
