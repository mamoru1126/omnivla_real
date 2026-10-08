"""推論サーバ (GPU / PyTorch 側) と、それを呼ぶポリシー (ROS 側) . 標準ライブラリ + numpy + PIL だけで動く.

実機では ROS 1 (Noetic = Ubuntu 20.04) と Jetson の PyTorch (JetPack ごとに Ubuntu / Python が決まる) を
同じ環境に入れにくいので、2 つのコンテナに分けて localhost の HTTP でつなぐ:

  [推論コンテナ] python3 tools/policy_server.py --model edge --weights ...      (PyTorch, GPU, ROS 不要)
  [ROS コンテナ] navigator (model: remote, url: http://127.0.0.1:8765)          (rospy, numpy, PIL だけ)

RemotePolicy は EdgePolicy / OmniVLAPolicy と同じ predict() / embed() / push() を持つので、
NavEngine・机上評価からは区別なく使える。画像は無圧縮 (uint8) で送るので、同じ重みなら
プロセス内で推論した場合と同じ結果になる。

通信: POST /predict, /embed, GET /info. 本文 = [4 byte: JSON の長さ][JSON][配列のバイト列 ...]
"""
from __future__ import annotations

import http.client
import json
import struct
import threading
import time
import urllib.parse
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .policy_base import PolicyOutput

DEFAULT_PORT = 8765


# ---------------------------------------------------------------------------
# 通信の形式
# ---------------------------------------------------------------------------
def pack(header: Dict[str, Any], arrays: Sequence[np.ndarray] = ()) -> bytes:
    arrays = [np.ascontiguousarray(a) for a in arrays]
    header = dict(header, arrays=[[list(a.shape), a.dtype.str] for a in arrays])
    hb = json.dumps(header, default=_json_default).encode()
    return struct.pack(">I", len(hb)) + hb + b"".join(a.tobytes() for a in arrays)


def unpack(body: bytes) -> Tuple[Dict[str, Any], List[np.ndarray]]:
    n = struct.unpack(">I", body[:4])[0]
    header = json.loads(body[4:4 + n].decode())
    off = 4 + n
    arrays = []
    for shape, dt in header.pop("arrays", []):
        dt = np.dtype(dt)
        size = int(np.prod(shape, dtype=np.int64)) * dt.itemsize
        arrays.append(np.frombuffer(body, dtype=dt, count=int(np.prod(shape, dtype=np.int64)), offset=off).reshape(shape))
        off += size
    return header, arrays


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _rgb(img) -> np.ndarray:
    if isinstance(img, Image.Image):
        return np.asarray(img.convert("RGB"), dtype=np.uint8)
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return arr[..., :3].astype(np.uint8, copy=False)


def _pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.asarray(arr, dtype=np.uint8)).convert("RGB")


def _policy_meta(policy) -> dict:
    meta = getattr(policy, "meta", None) or getattr(getattr(policy, "c", None), "meta", None) or {}
    return json.loads(json.dumps(meta, default=_json_default))


# ---------------------------------------------------------------------------
# サーバ (PyTorch 側)
# ---------------------------------------------------------------------------
class PolicyServer:
    """policy (EdgePolicy / OmniVLAPolicy) を HTTP で公開する. 推論は 1 つずつ (ロック)."""

    def __init__(self, policy, model_name: str, host: str = "127.0.0.1", port: int = DEFAULT_PORT,
                 log=print):
        self.policy = policy
        self.model_name = model_name
        self.lock = threading.Lock()
        self.log = log
        self.n_requests = 0
        self.busy_time = 0.0
        hist = getattr(policy, "history", None)
        self.info = {
            "model": model_name,
            "history": hist is not None,
            "context_size": int(getattr(policy, "context_size", 0) or 0),
            "context_stride": int(getattr(policy, "context_stride", 1) or 1),
            "metric_spacing": float(getattr(policy, "metric_spacing", 0.0) or 0.0),
            "meta": _policy_meta(policy),
        }
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):  # リクエストごとのログは出さない
                pass

            def _send(self, code: int, body: bytes, ctype: str = "application/octet-stream") -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.rstrip("/") in ("/info", ""):
                    self._send(200, json.dumps(server.info).encode(), "application/json")
                else:
                    self._send(404, b"not found", "text/plain")

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                try:
                    header, arrays = unpack(body)
                    op = self.path.strip("/")
                    t0 = time.time()
                    with server.lock:
                        out = server.handle(op, header, arrays)
                        server.n_requests += 1
                        server.busy_time += time.time() - t0
                    self._send(200, out)
                except Exception as e:  # noqa: BLE001  (クライアントにエラーを返す)
                    self._send(500, pack({"error": f"{type(e).__name__}: {e}"}))

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.httpd.daemon_threads = True
        self.address = self.httpd.server_address

    def handle(self, op: str, header: dict, arrays: List[np.ndarray]) -> bytes:
        if op == "predict":
            imgs = [_pil(a) for a in arrays]
            cur = imgs[header["current"]]
            goal = imgs[header["goal"]] if header.get("goal") is not None else None
            kw = dict(goal_image=goal, goal_pose=header.get("goal_pose"), modality=header.get("modality", "image"))
            if header.get("instruction"):
                kw["instruction"] = header["instruction"]
            if header.get("observations") is not None and self.info["history"]:
                kw["observations"] = [imgs[i] for i in header["observations"]]
            out: PolicyOutput = self.policy.predict(cur, **kw)
            return pack({"modality": int(out.modality), "latency": float(out.latency),
                         "distance": None if out.distance is None else float(out.distance)},
                        [np.asarray(out.waypoints, np.float64), np.asarray(out.normalized, np.float64),
                         np.asarray(out.goal_pose_input, np.float64)])
        if op == "embed":
            emb = self.policy.embed(_pil(arrays[0]))
            return pack({}, [np.asarray(emb, np.float32)])
        raise ValueError(f"unknown op '{op}'")

    def serve_forever(self) -> None:
        self.log(f"[policy_server] {self.model_name} on http://{self.address[0]}:{self.address[1]}")
        self.httpd.serve_forever()

    def start_background(self) -> threading.Thread:
        th = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        th.start()
        return th

    def shutdown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


# ---------------------------------------------------------------------------
# クライアント (ROS 側)
# ---------------------------------------------------------------------------
class RemotePolicy:
    """推論サーバを呼ぶポリシー. 観測履歴 (edge) はこちらで持ち、推論のたびに送る."""

    def __init__(self, url: str = f"http://127.0.0.1:{DEFAULT_PORT}", timeout: float = 10.0, wait: float = 600.0,
                 log=print):
        u = urllib.parse.urlparse(url if "://" in url else "http://" + url)
        self.url = url
        self.host, self.port = u.hostname or "127.0.0.1", u.port or DEFAULT_PORT
        self.timeout = timeout
        self._conn: Optional[http.client.HTTPConnection] = None
        self._lock = threading.Lock()
        self._emb_cache: Dict[str, np.ndarray] = {}
        info = self._wait_info(wait, log)
        self.server_info = info
        self.model_name = info.get("model", "?")
        self.meta = info.get("meta") or {}
        self.metric_spacing = float(info.get("metric_spacing") or 0.0)
        if info.get("history"):
            self.context_size = int(info["context_size"])
            self.context_stride = int(info.get("context_stride") or 1)
            self.history: Optional[Deque[Image.Image]] = deque(maxlen=self.context_size * self.context_stride + 1)
        else:
            self.context_size, self.context_stride, self.history = 0, 1, None
        log(f"[remote] {self.model_name} at {self.host}:{self.port} (history={self.history is not None})")

    # -- 通信 ----------------------------------------------------------------
    def _wait_info(self, wait: float, log) -> dict:
        t_end = time.time() + wait
        last = 0.0
        while True:
            try:
                return json.loads(self._request("GET", "/info").decode())
            except (OSError, http.client.HTTPException) as e:
                if time.time() > t_end:
                    raise ConnectionError(f"policy server {self.host}:{self.port} is not reachable: {e}") from e
                if time.time() - last > 10:
                    log(f"[remote] waiting for policy server {self.host}:{self.port} ... ({e.__class__.__name__})")
                    last = time.time()
                time.sleep(1.0)

    def _request(self, method: str, path: str, body: Optional[bytes] = None) -> bytes:
        with self._lock:
            for attempt in range(2):
                try:
                    if self._conn is None:
                        self._conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
                    self._conn.request(method, path, body=body,
                                       headers={"Content-Type": "application/octet-stream"} if body else {})
                    resp = self._conn.getresponse()
                    data = resp.read()
                    if resp.status != 200:
                        msg = data.decode(errors="replace")
                        try:
                            msg = unpack(data)[0].get("error", msg)
                        except Exception:  # noqa: BLE001
                            pass
                        raise RuntimeError(f"policy server error ({resp.status}): {msg}")
                    return data
                except (OSError, http.client.HTTPException):
                    if self._conn is not None:
                        self._conn.close()
                    self._conn = None
                    if attempt == 1:
                        raise
        raise RuntimeError("unreachable")

    # -- ポリシーとしてのインターフェース ------------------------------------
    def push(self, image) -> None:
        if self.history is not None:
            self.history.append(image if isinstance(image, Image.Image) else _pil(_rgb(image)))

    def reset_history(self) -> None:
        if self.history is not None:
            self.history.clear()

    def observation_window(self, current: Image.Image) -> List[Image.Image]:
        """EdgePolicy.observation_window と同じ並び (古い順に context_size 枚 + 現在)."""
        hist = list(self.history or [])
        if not hist or hist[-1] is not current:
            hist.append(current)
        s = self.context_stride
        idx = [len(hist) - 1 - s * k for k in range(self.context_size, -1, -1)]
        return [hist[max(0, i)] for i in idx]

    def predict(self, current, goal_image=None, goal_pose=None, instruction: Optional[str] = None,
                modality="image", observations=None) -> PolicyOutput:
        t0 = time.time()
        images: List = []
        index: Dict[int, int] = {}

        def add(img) -> int:
            k = id(img)
            if k not in index:
                index[k] = len(images)
                images.append(img)
            return index[k]

        header: Dict[str, Any] = {"current": add(current), "modality": modality, "instruction": instruction,
                                  "goal_pose": None if goal_pose is None else [float(v) for v in goal_pose]}
        header["goal"] = add(goal_image) if goal_image is not None else None
        if self.history is not None:
            obs = list(observations) if observations is not None else self.observation_window(current)
            header["observations"] = [add(o) for o in obs]
        data = self._request("POST", "/predict", pack(header, [_rgb(im) for im in images]))
        h, arr = unpack(data)
        return PolicyOutput(arr[0].copy(), arr[1].copy(), int(h["modality"]), time.time() - t0, arr[2].copy(),
                            distance=h.get("distance"))

    def embed(self, image, cache_key: Optional[str] = None) -> np.ndarray:
        if cache_key is not None and cache_key in self._emb_cache:
            return self._emb_cache[cache_key]
        _, arr = unpack(self._request("POST", "/embed", pack({}, [_rgb(image)])))
        emb = arr[0].copy()
        if cache_key is not None:
            self._emb_cache[cache_key] = emb
        return emb

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
