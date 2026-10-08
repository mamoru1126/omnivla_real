"""本物の rosbag (ROS1 .bag / ROS2 sqlite3 / ROS2 mcap) を書いて読み戻すテスト. rosbags が必要.

  python3 -m pytest -q tests/test_bags.py
  python3 tests/test_bags.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

try:
    import rosbags  # noqa: F401
    HAVE_ROSBAGS = True
except ImportError:  # pragma: no cover
    HAVE_ROSBAGS = False

from omnivla_real.bag.reader import ListSource  # noqa: E402
from omnivla_real.convert import ConvertConfig, convert_streams  # noqa: E402
from omnivla_real.extract import read_streams  # noqa: E402
from omnivla_real.robot_config import RobotConfig  # noqa: E402
from omnivla_real.runio import open_run  # noqa: E402
from synthetic import simulate_course  # noqa: E402

CONFIGS = os.path.join(REPO, "configs")
_MSGS = {}


def _msgs(**kw):
    key = tuple(sorted(kw.items()))
    if key not in _MSGS:
        _MSGS[key] = simulate_course(stop_at=None, **kw)
    return _MSGS[key]


def _robot(**topics) -> RobotConfig:
    return RobotConfig.load(os.path.join(CONFIGS, "robot.yaml"), {"topics": topics} if topics else None)


def _skip() -> bool:
    if not HAVE_ROSBAGS:
        print("  (skip: rosbags not installed)")
        try:
            import pytest

            pytest.skip("rosbags not installed")
        except ImportError:
            pass
        return True
    return False


def _compare_with_listsource(spec: str, msgs, robot: RobotConfig) -> None:
    ref = read_streams(ListSource(msgs), robot, tempfile.mkdtemp(), 3.0)
    with open_run(spec) as bag:
        tp = bag.topics()
        for t in ("/cmd_vel", "/odom", "/camera/image_raw/compressed", "/camera/image_raw"):
            if t in tp:
                n_ref = sum(m.topic == t for m in msgs)
                assert tp[t][1] == n_ref, (spec, t, tp[t], n_ref)
        got = read_streams(bag, robot, tempfile.mkdtemp(), 3.0)
    assert len(got.image_t) == len(ref.image_t) > 50, (len(got.image_t), len(ref.image_t))
    assert np.allclose(got.image_t, ref.image_t, atol=1e-6)
    assert np.allclose(got.cmd_v, ref.cmd_v) and np.allclose(got.cmd_w, ref.cmd_w)
    for k in ("odom_t", "odom_x", "odom_y", "odom_yaw", "odom_v", "loc_x", "loc_yaw"):
        assert np.allclose(getattr(got, k), getattr(ref, k), atol=1e-9), k
    # 画像も同じに復号できる
    from PIL import Image

    a = np.asarray(Image.open(got.image_paths[10]), dtype=np.int16)
    b = np.asarray(Image.open(ref.image_paths[10]), dtype=np.int16)
    assert a.shape == b.shape and np.abs(a - b).mean() < 1.0


def test_roundtrip_all_formats():
    if _skip():
        return
    from bagfiles import write_bag

    msgs = _msgs(image_size=(96, 72))
    out = tempfile.mkdtemp()
    for fmt in ("ros1", "sqlite3", "mcap"):
        path = write_bag(msgs, os.path.join(out, f"c_{fmt}"), fmt)
        _compare_with_listsource(path, msgs, _robot(localization="/localization"))
        if fmt != "ros1":  # .db3 / .mcap ファイルを直接渡しても開ける
            inner = [f for f in os.listdir(path) if f.endswith((".db3", ".mcap"))][0]
            with open_run(os.path.join(path, inner)) as bag:
                assert "/odom" in bag.topics()


def test_split_ros1_and_raw_twiststamped():
    if _skip():
        return
    from bagfiles import write_bag

    msgs = _msgs(raw_image=True, twist_stamped=True, with_localization=False, image_size=(64, 48))
    out = tempfile.mkdtemp()
    mid = msgs[len(msgs) // 2].t_bag
    a = write_bag([m for m in msgs if m.t_bag < mid], os.path.join(out, "r_0"), "ros1")
    b = write_bag([m for m in msgs if m.t_bag >= mid], os.path.join(out, "r_1"), "ros1")
    robot = _robot(image="/camera/image_raw")
    _compare_with_listsource(f"{a},{b}", msgs, robot)
    with open_run(f"{a},{b}") as bag:
        s = read_streams(bag, robot, tempfile.mkdtemp(), 3.0)
    res = convert_streams(s, ConvertConfig(), tempfile.mkdtemp(), name="r")
    assert len(res["trajectories"]) >= 1 and res["segments"] >= 1, res


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
