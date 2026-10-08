#!/usr/bin/env bash
# 4. 学習. 結果は runs/<run>/checkpoints/step_XXXXXX/
#
#   bin/train.sh 7b                      # OmniVLA 7B (LoRA). 設定は .env の TRAIN_7B_CONFIG (既定 configs/finetune_7b.yaml)
#   bin/train.sh edge                    # OmniVLA-edge.     設定は .env の EDGE_CONFIG (既定 configs/finetune_edge.yaml)
#   bin/train.sh 7b --config configs/my.yaml --max_steps 3000   # 設定ファイルの値は後ろのオプションで上書きできる
#   bin/train.sh 7b -d                   # 裏で動かす (端末を閉じても続く). ログは docker logs -f omnivla_train_7b
source "$(dirname "$0")/_lib.sh"
MODEL=${1:-}
shift || true
case $MODEL in
    7b) SVC=train_7b SCRIPT=training/finetune_omnivla.py CFG=$(env_get TRAIN_7B_CONFIG configs/finetune_7b.yaml) ;;
    edge) SVC=train_edge SCRIPT=training/finetune_edge.py CFG=$(env_get EDGE_CONFIG configs/finetune_edge.yaml) ;;
    *) usage 1 ;;
esac
DETACH=()
ARGS=()
while [ $# -gt 0 ]; do
    case $1 in
        -d | --detach) DETACH=(-d --name "omnivla_train_$MODEL") ;;
        --config) CFG=$2; shift ;;
        *) ARGS+=("$1") ;;
    esac
    shift
done
case $CFG in /*) CFG=$(cpath data "" "$CFG") ;; esac
[ -f "$(hpath "$CFG")" ] || [ -f "$ROOT/$CFG" ] || die "設定ファイルが見つかりません: $CFG"
ensure_env
say "$SCRIPT --config $CFG ${ARGS[*]}"
pc run --rm "${TTY[@]}" "${DETACH[@]}" "$SVC" python3 "$SCRIPT" --config "$CFG" "${ARGS[@]}"
if [ ${#DETACH[@]} -gt 0 ]; then
    say "裏で学習しています. ログ: docker logs -f omnivla_train_$MODEL   止める: docker stop omnivla_train_$MODEL"
else
    say "結果: $(host_dir runs)/<run>/checkpoints/"
fi
