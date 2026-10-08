"""ROS メッセージ (ROS1/ROS2 共通) を numpy / PIL に変換する (ROS も cv_bridge も不要).

rosbags で deserialize したメッセージ、rospy / rclpy のメッセージのどちらも
「属性でフィールドにアクセスできるオブジェクト」として同じように扱える。

対応:
  画像    : sensor_msgs/Image (rgb8, bgr8, rgba8, bgra8, mono8, mono16, yuv422/uyvy, yuyv, bayer_*),
            sensor_msgs/CompressedImage (jpeg / png)
  速度    : geometry_msgs/Twist, geometry_msgs/TwistStamped, nav_msgs/Odometry (twist), 任意フィールド指定
  姿勢    : nav_msgs/Odometry, geometry_msgs/PoseStamped, PoseWithCovarianceStamped, TransformStamped,
            tf2_msgs/TFMessage (frame_id / child_frame_id で 1 本の変換を選ぶ)
"""
from __future__ import annotations

import io
import math
from typing import Any, Optional, Tuple

import numpy as np
from PIL import Image

Pose2D = Tuple[float, float, float]


# ---------------------------------------------------------------------------
# 共通
# ---------------------------------------------------------------------------
def get_field(msg: Any, path: str) -> Any:
    """'twist.linear.x' のようなドット区切りでフィールドを取り出す."""
    obj = msg
    for name in path.split("."):
        if name == "":
            continue
        obj = getattr(obj, name)
    return obj


def stamp_to_sec(stamp: Any) -> Optional[float]:
    """builtin_interfaces/Time (sec, nanosec) と rospy.Time (secs, nsecs) の両方に対応. 0 なら None."""
    if stamp is None:
        return None
    if hasattr(stamp, "sec"):
        sec, nsec = stamp.sec, getattr(stamp, "nanosec", 0)
    elif hasattr(stamp, "secs"):
        sec, nsec = stamp.secs, getattr(stamp, "nsecs", 0)
    elif hasattr(stamp, "to_sec"):
        return float(stamp.to_sec()) or None
    else:
        return None
    t = float(sec) + float(nsec) * 1e-9
    return t if t > 0 else None


def header_stamp(msg: Any) -> Optional[float]:
    header = getattr(msg, "header", None)
    return stamp_to_sec(getattr(header, "stamp", None)) if header is not None else None


def yaw_from_quaternion(q: Any) -> float:
    x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def short_type(msgtype: str) -> str:
    """'sensor_msgs/msg/Image' / 'sensor_msgs/Image' -> 'sensor_msgs/Image'."""
    parts = [p for p in str(msgtype).split("/") if p and p != "msg"]
    return "/".join(parts[-2:]) if len(parts) >= 2 else str(msgtype)


def _bytes(data: Any) -> bytes:
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if isinstance(data, np.ndarray):
        return data.tobytes()
    return bytes(bytearray(data))


# ---------------------------------------------------------------------------
# 画像
# ---------------------------------------------------------------------------
IMAGE_TYPES = ("sensor_msgs/Image", "sensor_msgs/CompressedImage")


def is_compressed(msg: Any, msgtype: Optional[str] = None) -> bool:
    if msgtype:
        return short_type(msgtype) == "sensor_msgs/CompressedImage"
    return hasattr(msg, "format") and not hasattr(msg, "encoding")


def decode_compressed(msg: Any) -> np.ndarray:
    fmt = str(getattr(msg, "format", "")).lower()
    if "compresseddepth" in fmt:
        raise ValueError("compressedDepth images are not supported")
    img = Image.open(io.BytesIO(_bytes(msg.data)))
    return np.asarray(img.convert("RGB"))


def _bayer_to_rgb(raw: np.ndarray, pattern: str) -> np.ndarray:
    """2x2 ブロックを 1 画素にまとめる簡易デモザイク (解像度は半分になる)."""
    h, w = raw.shape[:2]
    raw = raw[: h - h % 2, : w - w % 2].astype(np.float32)
    p00, p01, p10, p11 = raw[0::2, 0::2], raw[0::2, 1::2], raw[1::2, 0::2], raw[1::2, 1::2]
    pattern = pattern.lower()
    if pattern == "rggb":
        r, g, b = p00, (p01 + p10) / 2, p11
    elif pattern == "bggr":
        r, g, b = p11, (p01 + p10) / 2, p00
    elif pattern == "gbrg":
        r, g, b = p10, (p00 + p11) / 2, p01
    elif pattern == "grbg":
        r, g, b = p01, (p00 + p11) / 2, p10
    else:
        raise ValueError(f"unknown bayer pattern {pattern}")
    return np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8)


def _yuv422_to_rgb(buf: np.ndarray, order: str) -> np.ndarray:
    """buf: (H, W*2) uint8. order: 'uyvy' or 'yuyv'."""
    h, w2 = buf.shape
    px = buf.reshape(h, w2 // 4, 4).astype(np.float32)
    if order == "uyvy":
        u, y0, v, y1 = px[..., 0], px[..., 1], px[..., 2], px[..., 3]
    else:
        y0, u, y1, v = px[..., 0], px[..., 1], px[..., 2], px[..., 3]
    y = np.stack([y0, y1], axis=-1).reshape(h, -1)
    u = np.repeat(u, 2, axis=1) - 128.0
    v = np.repeat(v, 2, axis=1) - 128.0
    r = y + 1.402 * v
    g = y - 0.344136 * u - 0.714136 * v
    b = y + 1.772 * u
    return np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8)


def decode_raw(msg: Any) -> np.ndarray:
    enc = str(msg.encoding).lower()
    h, w, step = int(msg.height), int(msg.width), int(msg.step)
    data = np.frombuffer(_bytes(msg.data), dtype=np.uint8)
    rows = data[: h * step].reshape(h, step)
    if enc in ("rgb8", "bgr8"):
        img = rows[:, : w * 3].reshape(h, w, 3)
        return img[..., ::-1].copy() if enc == "bgr8" else img.copy()
    if enc in ("rgba8", "bgra8"):
        img = rows[:, : w * 4].reshape(h, w, 4)[..., :3]
        return img[..., ::-1].copy() if enc == "bgra8" else img.copy()
    if enc in ("mono8", "8uc1"):
        g = rows[:, :w]
        return np.repeat(g[..., None], 3, axis=2)
    if enc in ("mono16", "16uc1"):
        dt = ">u2" if int(getattr(msg, "is_bigendian", 0)) else "<u2"
        g = np.frombuffer(rows[:, : w * 2].tobytes(), dtype=dt).reshape(h, w).astype(np.float32)
        hi = np.percentile(g, 99.5) or 1.0
        g8 = np.clip(g / hi * 255.0, 0, 255).astype(np.uint8)
        return np.repeat(g8[..., None], 3, axis=2)
    if enc in ("yuv422", "uyvy", "yuv422_uyvy"):
        return _yuv422_to_rgb(rows[:, : w * 2], "uyvy")
    if enc in ("yuyv", "yuv422_yuy2", "yuy2"):
        return _yuv422_to_rgb(rows[:, : w * 2], "yuyv")
    if enc.startswith("bayer_") and enc.endswith("8"):
        return _bayer_to_rgb(rows[:, :w], enc[len("bayer_"):-1])
    raise ValueError(f"unsupported image encoding '{msg.encoding}'")


def image_to_rgb(msg: Any, msgtype: Optional[str] = None) -> np.ndarray:
    """(H, W, 3) uint8 RGB."""
    if is_compressed(msg, msgtype):
        return decode_compressed(msg)
    return decode_raw(msg)


# ---------------------------------------------------------------------------
# 速度 (指令値 / オドメトリ)
# ---------------------------------------------------------------------------
def twist_values(msg: Any, msgtype: Optional[str] = None, v_field: str = "", w_field: str = "") -> Tuple[float, float]:
    """(v [m/s], w [rad/s]). v_field/w_field を指定すると任意のメッセージから取り出す."""
    if v_field and w_field:
        return float(get_field(msg, v_field)), float(get_field(msg, w_field))
    st = short_type(msgtype) if msgtype else ""
    if st == "nav_msgs/Odometry" or (not st and hasattr(msg, "child_frame_id") and hasattr(msg, "twist")):
        tw = msg.twist.twist
    elif st == "geometry_msgs/TwistStamped" or (not st and hasattr(msg, "twist") and hasattr(msg, "header")):
        tw = msg.twist
    elif hasattr(msg, "linear") and hasattr(msg, "angular"):
        tw = msg
    elif hasattr(msg, "twist"):
        tw = msg.twist
        tw = getattr(tw, "twist", tw)
    else:
        raise ValueError(f"cannot read a twist from {msgtype or type(msg)}; set v_field / w_field")
    return float(tw.linear.x), float(tw.angular.z)


# ---------------------------------------------------------------------------
# 姿勢 (オドメトリ / 自己位置)
# ---------------------------------------------------------------------------
def pose_values(msg: Any, msgtype: Optional[str] = None, frame_id: str = "", child_frame_id: str = "") -> Optional[Pose2D]:
    """(x, y, yaw). TFMessage で該当する変換が無ければ None."""
    st = short_type(msgtype) if msgtype else ""
    if st == "tf2_msgs/TFMessage" or (not st and hasattr(msg, "transforms")):
        for tf in msg.transforms:
            if child_frame_id and tf.child_frame_id.lstrip("/") != child_frame_id.lstrip("/"):
                continue
            if frame_id and tf.header.frame_id.lstrip("/") != frame_id.lstrip("/"):
                continue
            tr = tf.transform
            return float(tr.translation.x), float(tr.translation.y), yaw_from_quaternion(tr.rotation)
        return None
    if hasattr(msg, "transform"):                      # TransformStamped
        tr = msg.transform
        return float(tr.translation.x), float(tr.translation.y), yaw_from_quaternion(tr.rotation)
    pose = msg.pose
    pose = getattr(pose, "pose", pose)                 # Odometry / PoseWithCovarianceStamped は pose.pose
    return float(pose.position.x), float(pose.position.y), yaw_from_quaternion(pose.orientation)


def transform_stamp(msg: Any, frame_id: str = "", child_frame_id: str = "") -> Optional[float]:
    if hasattr(msg, "transforms"):
        for tf in msg.transforms:
            if child_frame_id and tf.child_frame_id.lstrip("/") != child_frame_id.lstrip("/"):
                continue
            if frame_id and tf.header.frame_id.lstrip("/") != frame_id.lstrip("/"):
                continue
            return stamp_to_sec(tf.header.stamp)
        return None
    return header_stamp(msg)
