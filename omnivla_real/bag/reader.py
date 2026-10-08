"""rosbag (ROS1 .bag / ROS2 sqlite3・mcap) を ROS 無しで読む.

rosbags (https://gitlab.com/ternaris/rosbags, pure Python) の AnyReader を使う。
  * ROS1 .bag  : メッセージ定義が bag に入っているので、独自メッセージもそのまま読める
  * ROS2       : bag ディレクトリ (metadata.yaml がある所) を渡す。.db3 / .mcap ファイルを渡しても
                 親ディレクトリに metadata.yaml があればそちらを開く。
                 定義が bag に無い型は ROS 2 Humble の標準型 (sensor_msgs, nav_msgs など) として読む。
  * 分割された bag (a_0.bag, a_1.bag ...) は複数まとめて渡すと時刻順に 1 本として読む。

使い方:
    with BagReader(["run1.bag"]) as bag:
        print(bag.topics())                       # {topic: (msgtype, count)}
        for m in bag.messages(["/odom", "/cmd_vel"]):
            m.topic, m.msgtype, m.t_bag, m.msg
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union


@dataclass
class BagMessage:
    topic: str
    msgtype: str
    t_bag: float          # bag に記録された受信時刻 [s]
    msg: Any


def resolve_bag_path(path: Union[str, Path]) -> Path:
    p = Path(os.path.expanduser(str(path))).resolve()
    if p.is_file() and p.suffix in (".db3", ".mcap") and (p.parent / "metadata.yaml").exists():
        return p.parent
    if p.is_file() and p.name == "metadata.yaml":
        return p.parent
    if not p.exists():
        raise FileNotFoundError(f"bag not found: {path}")
    if p.is_file() and p.suffix in (".db3", ".mcap"):
        raise FileNotFoundError(
            f"{p} has no metadata.yaml next to it. Run `ros2 bag reindex {p.parent}` to create it, "
            "or pass the bag directory.")
    return p


def bag_name(paths: Sequence[Union[str, Path]]) -> str:
    p = resolve_bag_path(paths[0])
    return p.stem if p.is_file() else p.name


class BagReader:
    def __init__(self, paths: Sequence[Union[str, Path]], typestore: str = "ROS2_HUMBLE"):
        if isinstance(paths, (str, Path)):
            paths = [paths]
        self.paths = [resolve_bag_path(p) for p in paths]
        self.typestore_name = typestore
        self._reader = None

    # -- context -----------------------------------------------------------
    def open(self) -> "BagReader":
        try:
            from rosbags.highlevel import AnyReader
            from rosbags.typesys import Stores, get_typestore
        except ImportError as e:  # pragma: no cover
            raise ImportError("rosbags is required to read bags: pip install rosbags") from e
        store = get_typestore(getattr(Stores, self.typestore_name))
        try:
            self._reader = AnyReader(self.paths, default_typestore=store)
        except TypeError:  # 古い rosbags (default_typestore 引数なし)
            self._reader = AnyReader(self.paths)
        self._reader.open()
        return self

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def __enter__(self) -> "BagReader":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    # -- info --------------------------------------------------------------
    @property
    def start_time(self) -> float:
        return self._reader.start_time * 1e-9

    @property
    def end_time(self) -> float:
        return self._reader.end_time * 1e-9

    def topics(self) -> Dict[str, Tuple[str, int]]:
        out: Dict[str, Tuple[str, int]] = {}
        for c in self._reader.connections:
            mt, n = out.get(c.topic, (c.msgtype, 0))
            out[c.topic] = (mt, n + int(getattr(c, "msgcount", 0) or 0))
        return out

    def msgtype(self, topic: str) -> Optional[str]:
        return self.topics().get(topic, (None, 0))[0]

    # -- data --------------------------------------------------------------
    def messages(self, topics: Sequence[str], start: Optional[float] = None,
                 stop: Optional[float] = None) -> Iterator[BagMessage]:
        """topics のメッセージを bag の時刻順に返す. start/stop は bag 時刻 [s]."""
        wanted = [t for t in topics if t]
        conns = [c for c in self._reader.connections if c.topic in wanted]
        missing = set(wanted) - {c.topic for c in conns}
        if missing:
            raise KeyError(f"topics not in bag: {sorted(missing)}. available: {sorted(self.topics())}")
        kw = {}
        if start is not None:
            kw["start"] = int(start * 1e9)
        if stop is not None:
            kw["stop"] = int(stop * 1e9)
        for conn, t_ns, raw in self._reader.messages(connections=conns, **kw):
            yield BagMessage(conn.topic, conn.msgtype, t_ns * 1e-9, self._reader.deserialize(raw, conn.msgtype))


class ListSource:
    """テスト・デバッグ用: BagMessage のリストを BagReader と同じように扱う."""

    def __init__(self, messages: List[BagMessage]):
        self._msgs = sorted(messages, key=lambda m: m.t_bag)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    @property
    def start_time(self) -> float:
        return self._msgs[0].t_bag if self._msgs else 0.0

    @property
    def end_time(self) -> float:
        return self._msgs[-1].t_bag if self._msgs else 0.0

    def topics(self) -> Dict[str, Tuple[str, int]]:
        out: Dict[str, Tuple[str, int]] = {}
        for m in self._msgs:
            mt, n = out.get(m.topic, (m.msgtype, 0))
            out[m.topic] = (mt, n + 1)
        return out

    def msgtype(self, topic: str) -> Optional[str]:
        return self.topics().get(topic, (None, 0))[0]

    def messages(self, topics: Sequence[str], start: Optional[float] = None, stop: Optional[float] = None):
        wanted = set(t for t in topics if t)
        for m in self._msgs:
            if m.topic in wanted and (start is None or m.t_bag >= start) and (stop is None or m.t_bag < stop):
                yield m
