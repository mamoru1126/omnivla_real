"""テスト用: 合成したコース走行 (synthetic.simulate_course) を本物の rosbag に書き出す.

  python3 tests/bagfiles.py OUT_DIR            # ROS1 .bag / ROS2 sqlite3 / ROS2 mcap を作る

rosbags の Writer を使うので ROS は不要。ROS1 は Noetic、ROS2 は Humble の型で書く。
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

FORMATS = ("ros1", "sqlite3", "mcap")
BASE, NAME, ARRAY, SEQUENCE = 1, 2, 3, 4      # rosbags の Nodetype

_NUMPY = {"bool": np.bool_, "byte": np.uint8, "char": np.uint8, "uint8": np.uint8, "int8": np.int8,
          "uint16": np.uint16, "int16": np.int16, "uint32": np.uint32, "int32": np.int32,
          "uint64": np.uint64, "int64": np.int64, "float32": np.float32, "float64": np.float64}


def _base_default(t: str) -> Any:
    if t == "string":
        return ""
    if t == "bool":
        return False
    return 0.0 if t.startswith("float") else 0


def to_ros(ts, msgtype: str, src: Any) -> Any:
    """SimpleNamespace のメッセージを rosbags の型に詰め替える (無いフィールドは 0 / 空)."""
    cls = ts.types[msgtype]
    _, fields = ts.fielddefs[msgtype]
    kw = {}
    for name, (kind, detail) in fields:
        val = getattr(src, name, None) if src is not None else None
        if int(kind) == BASE:
            t = detail[0]
            kw[name] = _base_default(t) if val is None else (str(val) if t == "string" else val)
        elif int(kind) == NAME:
            kw[name] = to_ros(ts, detail, val)
        else:  # ARRAY / SEQUENCE
            (sub_kind, sub), length = detail
            if int(sub_kind) == BASE and sub[0] in _NUMPY:
                arr = np.asarray([] if val is None else val, dtype=_NUMPY[sub[0]])
                if int(kind) == ARRAY and arr.size == 0:
                    arr = np.zeros(length, dtype=_NUMPY[sub[0]])
                kw[name] = arr
            elif int(sub_kind) == BASE:
                kw[name] = [str(v) for v in (val or [])]
            else:
                kw[name] = [to_ros(ts, sub, v) for v in (val or [])]
    return cls(**kw)


def write_bag(msgs: Sequence, path: str, fmt: str) -> str:
    """msgs (BagMessage のリスト) を fmt 形式で path に書く. 返り値は bag のパス."""
    from rosbags.typesys import Stores, get_typestore

    msgs = sorted(msgs, key=lambda m: m.t_bag)
    types: Dict[str, str] = {}
    for m in msgs:
        types.setdefault(m.topic, m.msgtype)
    if fmt == "ros1":
        from rosbags.rosbag1 import Writer

        ts = get_typestore(Stores.ROS1_NOETIC)
        if not path.endswith(".bag"):
            path += ".bag"
        with Writer(path) as w:
            conns = {t: w.add_connection(t, mt, typestore=ts) for t, mt in types.items()}
            for m in msgs:
                w.write(conns[m.topic], int(round(m.t_bag * 1e9)), ts.serialize_ros1(to_ros(ts, m.msgtype, m.msg), m.msgtype))
        return path
    if fmt in ("sqlite3", "mcap"):
        from rosbags.rosbag2 import StoragePlugin, Writer

        ts = get_typestore(Stores.ROS2_HUMBLE)
        plugin = StoragePlugin.MCAP if fmt == "mcap" else StoragePlugin.SQLITE3
        with Writer(path, version=8, storage_plugin=plugin) as w:
            conns = {t: w.add_connection(t, mt, typestore=ts) for t, mt in types.items()}
            for m in msgs:
                w.write(conns[m.topic], int(round(m.t_bag * 1e9)), ts.serialize_cdr(to_ros(ts, m.msgtype, m.msg), m.msgtype))
        return path
    raise ValueError(f"unknown format {fmt}")


def write_course_bags(out_dir: str, formats: Sequence[str] = FORMATS, split: bool = False, **sim_kw) -> List[str]:
    """合成コースを各形式で書く. split=True なら ROS1 は 2 ファイルに分割 (分割 bag の結合テスト用)."""
    from synthetic import simulate_course

    os.makedirs(out_dir, exist_ok=True)
    msgs = simulate_course(**sim_kw)
    out = []
    for fmt in formats:
        name = os.path.join(out_dir, f"course_{fmt}")
        if fmt == "ros1" and split:
            t_mid = msgs[len(msgs) // 2].t_bag
            a = write_bag([m for m in msgs if m.t_bag < t_mid], name + "_0", fmt)
            b = write_bag([m for m in msgs if m.t_bag >= t_mid], name + "_1", fmt)
            out.append(f"{a},{b}")
        else:
            out.append(write_bag(msgs, name, fmt))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--formats", default=",".join(FORMATS))
    ap.add_argument("--laps", type=int, default=1)
    ap.add_argument("--no_localization", action="store_true")
    ap.add_argument("--raw_image", action="store_true")
    ap.add_argument("--image_size", default="160,120")
    ap.add_argument("--split", action="store_true")
    ap.add_argument("--speed", type=float, default=0.6)
    ap.add_argument("--slip", type=float, default=0.97)
    ap.add_argument("--no_stop", action="store_true", help="途中の一時停止なし")
    a = ap.parse_args(argv)
    size = tuple(int(v) for v in a.image_size.split(","))
    paths = write_course_bags(a.out_dir, a.formats.split(","), split=a.split, laps=a.laps,
                              with_localization=not a.no_localization, raw_image=a.raw_image, image_size=size,
                              speed=a.speed, slip=a.slip, **({"stop_at": None} if a.no_stop else {}))
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
