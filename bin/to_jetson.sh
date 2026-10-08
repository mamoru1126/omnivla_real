#!/usr/bin/env bash
# 学習 PC → Jetson: 走行に使う重みと topomap を rsync で送る (ssh で入れること. Jetson 側にもリポジトリを clone 済み).
# 送るものは .env の NAV_MODEL / FINETUNED_DIR / NAV_WEIGHTS / TOPOMAP (学習 PC で bin/replay.sh したときと同じもの).
# 元のモデル (公式の重み. 7B は約 16GB) も送る. rsync なので 2 回目からは変わった分だけ.
#
#   bin/to_jetson.sh user@jetson                       # Jetson の ~/omnivla_real に送る
#   bin/to_jetson.sh user@jetson --dest /opt/omnivla_real --configs
#        --dest DIR   Jetson 側のリポジトリ   --configs  configs/robot.yaml と navigator.yaml も送る
#        --dry-run    送るものを表示するだけ
# Jetson 側の .env の BAG_DIR / DATA_DIR / RUNS_DIR / CHECKPOINT_DIR は既定 (./bags など) のままとする.
source "$(dirname "$0")/_lib.sh"
[ $# -ge 1 ] && [[ $1 != -* ]] || usage 1
HOST=$1
shift
DEST=omnivla_real
CONFIGS=0
RSYNC=(rsync -aR --info=progress2)
while [ $# -gt 0 ]; do
    case $1 in
        --dest) DEST=$2; shift ;;
        --configs) CONFIGS=1 ;;
        --dry-run | -n) RSYNC+=(-n -v) ;;
        *) die "知らないオプション: $1" ;;
    esac
    shift
done

MODEL=$(env_get NAV_MODEL "")
F=$(cpath runs "" "$(env_get FINETUNED_DIR "")")
W=$(cpath runs "" "$(env_get NAV_WEIGHTS "")")
T=$(env_get TOPOMAP "")
[ -n "$T" ] && T=$(topomap_path "$T")
[ -n "$MODEL" ] || MODEL=$(sed -n -E 's/^[[:space:]]+model:[[:space:]]*([a-z0-9]+).*/\1/p' configs/navigator.yaml | head -1)
# 元のモデル: NAV_WEIGHTS か公式の重み (7B の学習結果は finetune_meta.json に書いてある元のモデル)
if [ -n "$W" ]; then
    BASE=$W
elif [ "$MODEL" = 7b ]; then
    BASE=/checkpoints/omnivla-original
    if [ -n "$F" ] && [ -f "$(hpath "$F")/finetune_meta.json" ]; then
        B=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("base_vla_path") or "")' \
            "$(hpath "$F")/finetune_meta.json" 2> /dev/null || true)
        BASE=${B:-$BASE}
    fi
else
    BASE=/checkpoints/omnivla-edge
fi
SEND=()
if [ -e "$(hpath "$BASE")" ]; then
    SEND+=("$BASE")
else
    say "注意: $BASE がこの PC にないので送りません (Jetson 側にあれば問題なし)"
fi
for p in "$F" "$T"; do
    [ -n "$p" ] || continue
    need "$p"
    SEND+=("$p")
done
[ ${#SEND[@]} -gt 0 ] || die ".env に送るもの (FINETUNED_DIR / NAV_WEIGHTS / TOPOMAP) がありません"

say "送り先: $HOST:$DEST  (モデル: $MODEL)"
if [ "${OMNIVLA_DRY_RUN:-0}" = 1 ]; then
    ssh() { echo "+ ssh $*"; }
    rsync() { echo "+ rsync $*"; }
fi
ssh "$HOST" "mkdir -p $DEST/bags $DEST/data $DEST/runs $DEST/checkpoints"
for p in "${SEND[@]}"; do
    case $p in
        /runs/* | /data/* | /checkpoints/*) ;;
        *) die "$p は /runs /data /checkpoints の外にあるので送れません" ;;
    esac
    k=${p#/}
    k=${k%%/*}
    need "$p"
    say "$p"
    "${RSYNC[@]}" "$(host_dir "$k")/./${p#/"$k"/}" "$HOST:$DEST/$k/"
done
if [ "$CONFIGS" = 1 ]; then
    say "configs/robot.yaml configs/navigator.yaml"
    "${RSYNC[@]}" configs/./robot.yaml configs/./navigator.yaml "$HOST:$DEST/configs/"
fi
say "送りました. Jetson の $DEST/.env に次を書いて bin/jetson.sh up:"
echo "NAV_MODEL=$MODEL"
[ -n "$F" ] && echo "FINETUNED_DIR=$F"
[ -n "$W" ] && echo "NAV_WEIGHTS=$W"
[ -n "$T" ] && echo "TOPOMAP=$T"
exit 0
