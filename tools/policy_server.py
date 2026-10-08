#!/usr/bin/env python3
"""推論サーバ: OmniVLA (7B / edge) を GPU で動かし、ROS ノード (model: remote) から HTTP で呼べるようにする.

ROS 1 のノードと PyTorch を別のコンテナ (別の Python) に分けるためのもの。ROS は不要。

  python3 tools/policy_server.py --model edge --weights /runs/<run>/checkpoints/step_010000
  python3 tools/policy_server.py --model 7b --finetuned_dir /runs/<run>/checkpoints/step_005000
  # 何も指定しなければ configs/navigator.yaml の model: の値 (model: remote は不可)

ROS 側: navigator.launch の policy_url:=http://127.0.0.1:8765 (同じ PC なら既定のままで良い)。
別の PC から呼ぶ場合は --host 0.0.0.0 (認証なし. ロボットの中のネットワークだけで使うこと)。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from omnivla_real.nav_config import load_nav_config, make_policy  # noqa: E402
from omnivla_real.remote import DEFAULT_PORT, PolicyServer  # noqa: E402
from omnivla_real.robot_config import RobotConfig  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nav", default=os.path.join(REPO, "configs/navigator.yaml"), help="model: の既定値")
    ap.add_argument("--robot", default=os.path.join(REPO, "configs/robot.yaml"),
                    help="画像の前処理 (C++ ノードが送るカメラ画像をここで切り抜き・縮小する)")
    ap.add_argument("--model", default="", help="edge | 7b")
    ap.add_argument("--weights", default="")
    ap.add_argument("--finetuned_dir", default="")
    ap.add_argument("--device", default="")
    ap.add_argument("--half", action="store_true", help="edge を fp16 で推論")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--warmup", type=int, default=3, help="起動時に空推論する回数 (初回の遅さを避ける)")
    args = ap.parse_args(argv)

    nav = load_nav_config(args.nav if os.path.exists(args.nav) else None,
                          {"model.model": args.model, "model.weights": args.weights,
                           "model.finetuned_dir": args.finetuned_dir, "model.device": args.device,
                           "model.half": True if args.half else None})
    if nav.model.model.lower() == "remote":
        raise SystemExit("--model edge or 7b (the server runs the model itself)")
    print(f"[policy_server] loading {nav.model.model} ...", flush=True)
    t0 = time.time()
    policy = make_policy(nav.model)
    print(f"[policy_server] loaded in {time.time() - t0:.1f}s", flush=True)
    if args.warmup > 0:
        img = Image.fromarray(np.full((240, 320, 3), 128, np.uint8))
        for _ in range(args.warmup):
            out = policy.predict(img, goal_image=img, modality="image")
        policy.embed(img)
        if hasattr(policy, "reset_history"):
            policy.reset_history()
        print(f"[policy_server] warmup done (latency {out.latency * 1000:.0f} ms)", flush=True)
    robot = RobotConfig.load(args.robot)
    server = PolicyServer(policy, nav.model.model.lower(), args.host, args.port,
                          log=lambda s: print(s, flush=True), image_cfg=robot.image)
    print(f"[policy_server] image preprocess (robot.yaml): crop={robot.image.crop} width={robot.image.width} "
          f"rotate180={robot.image.rotate180}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
