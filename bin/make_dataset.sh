#!/usr/bin/env bash
# 3. bag を学習データに変換し (data/dataset に追加)、正解の軌跡を画像に重ねた確認用の図を作る (runs/inspect).
#
#   bin/make_dataset.sh run1.bag run2.bag
#   bin/make_dataset.sh run3_0.bag,run3_1.bag             # 分割された bag は 1 本の走行として扱う
#   bin/make_dataset.sh run1.bag --pose_source cmd        # オプション (- で始まる引数から後) は bag_to_dataset.py に渡す
# 出力先を変えるとき: DATASET=/data/dataset_b bin/make_dataset.sh ...  (学習の設定の data_dirs も合わせる)
source "$(dirname "$0")/_lib.sh"
BAGS=()
while [ $# -gt 0 ] && [[ $1 != -* ]]; do
    BAGS+=("$(cbags "$1")")
    shift
done
[ ${#BAGS[@]} -ge 1 ] || usage 1
DATASET=$(cpath data "" "${DATASET:-/data/dataset}")
in_shell bash -c 'python3 tools/bag_to_dataset.py --out "$0" "$@" && python3 training/inspect_dataset.py "$0" --num_viz 16 --out /runs/inspect' \
    "$DATASET" "${BAGS[@]}" "$@"
say "学習データ: $(hpath "$DATASET")  (使った区間: _reports/<bag>.png)"
say "確認用の図: $(hpath /runs/inspect)"
