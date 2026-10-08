#!/usr/bin/env bash
# 1. bag の中身 (トピック・型・周期) を見る. 表示されたトピック名を configs/robot.yaml の topics に書く.
#
#   bin/bag_info.sh run1.bag                # ROS 1 (BAG_DIR の中なら名前だけでよい)
#   bin/bag_info.sh run1_0.bag,run1_1.bag   # 分割された bag
#   bin/bag_info.sh run1                    # ROS 2 (bag のディレクトリ)
source "$(dirname "$0")/_lib.sh"
[ $# -ge 1 ] || usage 1
BAG=$(cbags "$1")
shift
in_shell python3 tools/bag_info.py "$BAG" "$@"
