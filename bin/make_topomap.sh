#!/usr/bin/env bash
# 5. コースを 1 回走った bag から、サブゴール画像列 (topomap) を作る. 出力は data/topomaps/<名前>/
#
#   bin/make_topomap.sh course_a run1.bag
#   bin/make_topomap.sh course_a run1.bag --spacing 1.5 --start_sec 12 --end_sec 240 --overwrite
#   (後ろのオプションは make_topomap.py に渡す. 確認は data/topomaps/<名前>/overview.png)
source "$(dirname "$0")/_lib.sh"
[ $# -ge 2 ] || usage 1
OUT=$(cpath data topomaps "$1")
BAG=$(cbags "$2")
shift 2
in_shell python3 tools/make_topomap.py --out "$OUT" "$BAG" "$@"
say "topomap: $(hpath "$OUT")  (走行で使うには .env に TOPOMAP=$OUT)"
