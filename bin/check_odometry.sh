#!/usr/bin/env bash
# 2. オドメトリが学習ラベルに使えるか確かめる. 結果は runs/check/<bag>/ (図と json).
#
#   bin/check_odometry.sh run1.bag
#   bin/check_odometry.sh run1.bag --start_sec 10 --end_sec 200   # 後ろのオプションは check_odometry.py に渡す
source "$(dirname "$0")/_lib.sh"
[ $# -ge 1 ] || usage 1
BAG=$(cbags "$1")
shift
OUT=/runs/check/$(bag_name "$BAG")
in_shell python3 tools/check_odometry.py --out "$OUT" "$BAG" "$@"
say "結果: $(hpath "$OUT")"
