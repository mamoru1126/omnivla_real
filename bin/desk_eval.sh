#!/usr/bin/env bash
# 6. 机上評価: 別の走行の bag をモデルに入れて、記録と比べる. 結果は runs/desk_eval/<bag>_<時刻>/
# モデルは .env の NAV_MODEL / NAV_WEIGHTS / FINETUNED_DIR (走行と同じ). topomap を省くと .env の TOPOMAP.
#
#   bin/desk_eval.sh run2.bag course_a
#   bin/desk_eval.sh run2.bag course_a --policy oracle       # モデルの代わりに正解を返す (仕組みの確認)
#   bin/desk_eval.sh run2.bag --goal_mode hindsight          # topomap を使わない
#   (後ろのオプションは desk_eval.py に渡す)
source "$(dirname "$0")/_lib.sh"
[ $# -ge 1 ] || usage 1
BAG=$(cbags "$1")
shift
TOPO=$(env_get TOPOMAP "")
if [ $# -gt 0 ] && [[ $1 != -* ]]; then
    TOPO=$1
    shift
fi
ARGS=()
[ -n "$TOPO" ] && ARGS+=(--topomap "$(topomap_path "$TOPO")")
M=$(env_get NAV_MODEL "")
W=$(env_get NAV_WEIGHTS "")
F=$(env_get FINETUNED_DIR "")
[ -n "$M" ] && ARGS+=(--model "$M")
[ -n "$W" ] && ARGS+=(--weights "$(cpath runs "" "$W")")
[ -n "$F" ] && ARGS+=(--finetuned_dir "$(cpath runs "" "$F")")
OUT=/runs/desk_eval/$(bag_name "$BAG")_$(date +%Y%m%d_%H%M%S)
in_shell python3 tools/desk_eval.py --bag "$BAG" --out "$OUT" --debug_every 10 "${ARGS[@]}" "$@"
say "結果: $(hpath "$OUT")  (overview.png, timeline.png, report.txt)"
