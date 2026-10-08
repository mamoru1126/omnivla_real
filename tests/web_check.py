#!/usr/bin/env python3
"""デバッグ画面 (C++ ノードの内蔵 Web サーバ) が動いているかの確認 (CI 用. 標準ライブラリだけ).

  python3 tests/web_check.py http://127.0.0.1:8090 40     # 40 秒の間に推論の結果が届くか
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request


def get(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.headers.get("Content-Type", ""), r.read()


def main(base: str, duration: float) -> int:
    ok = True
    t_end = time.time() + duration
    while True:   # ノードが起動するまで待つ
        try:
            st, ctype, body = get(base + "/")
            break
        except OSError:
            if time.time() > t_end:
                print("NG: debug page is not reachable")
                return 1
            time.sleep(1)
    print("GET /", st, ctype, len(body), "bytes")
    if st != 200 or b"OmniVLA" not in body:
        print("NG: index page")
        ok = False
    st, _, body = get(base + "/api/config")
    cfg = json.loads(body)
    print("GET /api/config", st, "camera", cfg.get("robot", {}).get("camera"), "nodes", cfg.get("topomap", {}).get("num_nodes"))
    if "robot" not in cfg or not cfg.get("topomap", {}).get("num_nodes"):
        print("NG: config")
        ok = False
    # Server-Sent Events: 推論の結果が届くか
    steps = statuses = 0
    last = None
    req = urllib.request.Request(base + "/api/events")
    with urllib.request.urlopen(req, timeout=10) as r:
        while time.time() < t_end and steps < 15:
            line = r.readline()
            if not line:
                break
            if line.startswith(b"data: "):
                m = json.loads(line[6:])
                if m.get("kind") == "step":
                    steps += 1
                    last = m
                else:
                    statuses += 1
    print(f"SSE: {steps} step messages, {statuses} status messages")
    if steps < 3 or last is None or len(last.get("waypoints") or []) != 8:
        print("NG: no step messages with 8 waypoints")
        ok = False
    else:
        print("   last step:", {k: last.get(k) for k in ("state", "subgoal", "num_nodes", "v", "w", "latency", "preview")})
        if not last.get("preview"):
            print("NG: preview image was not requested while the page was open")
            ok = False
        st, ctype, body = get(base + "/img/current.jpg")
        print("GET /img/current.jpg", st, ctype, len(body))
        if st != 200 or not body.startswith(b"\xff\xd8"):
            print("NG: current image")
            ok = False
        st, ctype, body = get(base + f"/img/goal/{last.get('subgoal', 0)}.jpg")
        print("GET /img/goal", st, ctype, len(body))
        if st != 200:
            print("NG: goal image")
            ok = False
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1].rstrip("/"), float(sys.argv[2]) if len(sys.argv) > 2 else 40.0))
