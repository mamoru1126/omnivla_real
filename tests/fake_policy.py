#!/usr/bin/env python3
"""テスト用の偽の推論サーバ (torch 不要). 本物の tools/policy_server.py と同じ通信をする.

  python3 tests/fake_policy.py --port 8799 [--robot configs/robot.yaml]

出力は入力画像の明るさなどから決まる (同じ入力なら同じ出力). 観測履歴の扱いは OmniVLA-edge と同じ。
C++ ノードと Python の NavEngine が同じ動きをするかの確認 (tests/test_cpp.py) に使う。
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from omnivla_real.policy_base import PolicyOutput  # noqa: E402


class FakeEdgePolicy:
    """edge と同じ観測履歴. 予測は「画像の明るさの左右差」で曲がり、ゴール画像との差で速さが変わる軌跡."""
    meta = {"sample_rate": 3.0, "metric_waypoint_spacing": 0.2}

    def __init__(self, context_size: int = 5, context_stride: int = 1, speed: float = 0.15):
        self.context_size, self.context_stride, self.metric_spacing = context_size, context_stride, 0.2
        self.history = deque(maxlen=context_size * context_stride + 1)
        self.calls = 0
        self.base_speed = speed

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
        self.calls += 1
        obs = observations if observations is not None else self.observation_window(current)
        cur = np.asarray(current, np.float64) / 255.0
        w = cur.shape[1]
        turn = float(cur[:, : w // 2].mean() - cur[:, w // 2:].mean())          # 左が明るい -> 左へ
        prev = float(np.asarray(obs[0], np.float64).mean() / 255.0)
        goal = float(np.asarray(goal_image, np.float64).mean() / 255.0) if goal_image is not None else 0.5
        speed = self.base_speed + 0.1 * abs(goal - prev)
        yaw_rate = 2.0 * turn
        wps = np.zeros((8, 4))
        for k in range(8):
            t = (k + 1) / 3.0
            yaw = yaw_rate * t
            if abs(yaw_rate) > 1e-6:
                r = speed / yaw_rate
                wps[k, 0], wps[k, 1] = r * math.sin(yaw), r * (1 - math.cos(yaw))
            else:
                wps[k, 0] = speed * t
            wps[k, 2], wps[k, 3] = math.cos(yaw), math.sin(yaw)
        gp = np.zeros(4) if goal_pose is None else np.r_[np.asarray(goal_pose, float)[:2], 0.0, 0.0]
        return PolicyOutput(wps, wps / 0.2, 6, 0.001, gp, distance=goal)

    def embed(self, image, cache_key=None):
        from omnivla_real.desk_eval import simple_embedding
        return simple_embedding(image)


def main(argv=None):
    from omnivla_real.remote import PolicyServer
    from omnivla_real.robot_config import RobotConfig

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8799)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--robot", default=os.path.join(REPO, "configs/robot.yaml"))
    ap.add_argument("--speed", type=float, default=0.15, help="予測する速さ [m/s] (画面の確認用)")
    args = ap.parse_args(argv)
    server = PolicyServer(FakeEdgePolicy(speed=args.speed), "fake", args.host, args.port, log=lambda s: print(s, flush=True),
                          image_cfg=RobotConfig.load(args.robot).image)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
