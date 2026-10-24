#!/usr/bin/env bash
# 7. 実走の評価: 自律走行した bag の自己位置を、コース (自己位置で作った topomap) と比べる.
#    結果は runs/run_eval/<bag>_<時刻>/ (report.txt, overview.png, xtrack.png). 自己位置は評価だけに使う.
#
#   bin/run_eval.sh auto3 course_b                            # 自己位置は robot.yaml の topics.eval_localization
#   bin/run_eval.sh auto3 course_b --loc_topic /localization  # トピックを指定する
#   bin/run_eval.sh log/bags/auto3_20261025_101010 course_b   # Orin で録った bag (Orin でもそのまま動く)
#   (後ろのオプションは tools/run_eval.py に渡す: --speed_cap 0.3 --start_sec 10 --end_sec 300 など)
source "$(dirname "$0")/_lib.sh"
[ $# -ge 2 ] || usage 1
BAG=$(cbags "$1")
TOPO=$(topomap_path "$2")
shift 2
OUT=/runs/run_eval/$(bag_name "$BAG")_$(date +%Y%m%d_%H%M%S)
in_shell python3 tools/run_eval.py --bag "$BAG" --topomap "$TOPO" --out "$OUT" "$@"
say "結果: $(hpath "$OUT")  (report.txt, overview.png, xtrack.png)"
