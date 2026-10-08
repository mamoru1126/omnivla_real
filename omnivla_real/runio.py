"""コマンドラインツール共通: 設定ファイルの読み込みと bag (走行) の指定."""
from __future__ import annotations

import os
from dataclasses import fields
from typing import Any, Dict, List, Optional, Sequence

import yaml

from .bag.reader import BagReader, bag_name
from .convert import ConvertConfig


def load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(os.path.expanduser(path)) as f:
        return yaml.safe_load(f) or {}


def load_convert_config(path: Optional[str], overrides: Optional[Dict[str, Any]] = None) -> ConvertConfig:
    data = load_yaml(path)
    data.update({k: v for k, v in (overrides or {}).items() if v is not None})
    names = {f.name for f in fields(ConvertConfig)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"unknown convert options: {sorted(unknown)}")
    return ConvertConfig(**data)


def split_run(spec: str) -> List[str]:
    """'a_0.bag,a_1.bag' のようにカンマでつなぐと分割 bag を 1 走行として扱う."""
    return [p for p in spec.split(",") if p]


def open_run(spec: str, typestore: str = "ROS2_HUMBLE") -> BagReader:
    return BagReader(split_run(spec), typestore=typestore)


def run_name(spec: str) -> str:
    return bag_name(split_run(spec))


def add_common_args(ap) -> None:
    ap.add_argument("--robot", default="configs/robot.yaml", help="トピック名・画像の前処理 (robot.yaml)")
    ap.add_argument("--typestore", default="ROS2_HUMBLE",
                    help="ROS2 bag で定義が入っていない型をどの ROS の標準型として読むか (例 ROS2_HUMBLE, ROS2_FOXY)")


def overrides_from_args(args, keys: Sequence[str]) -> Dict[str, Any]:
    return {k: getattr(args, k) for k in keys if getattr(args, k, None) is not None}
