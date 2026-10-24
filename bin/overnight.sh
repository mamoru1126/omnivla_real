#!/usr/bin/env bash
# 夜のあいだに学習 PC で回す一式: 新しいコースの bag を足して 7B を学習し直し、新旧のモデルを机上評価で比べる.
# 最初の確認 (1〜2 分) だけ端末で行い、あとは裏で動く (端末を閉じても続く). 朝に summary.txt を見る.
#
#   bin/overnight.sh <新しい bag> [--loc <自己位置のトピック>] [--course course_b] [--weight 3] [--edge]
#
#     --loc     評価だけに使う自己位置 (PoseStamped). robot.yaml の topics.eval_localization に書く
#               (course はその自己位置の座標で作る = bin/run_eval.sh で実走を評価できる)
#     --course  新しいコースの topomap の名前 (既定 course_b)
#     --weight  新しい bag を何倍引くか (既定 3. 前のコースの bag も一緒に使う)
#     --edge    OmniVLA-edge も学習して比べる (+1〜2 時間)
#
# 裏でやること (ログ: log/overnight_<時刻>/overnight.log, 結果: runs/overnight_<時刻>/):
#   1. 学習データに新しい bag を追加 (bin/make_dataset.sh) と 運転の仕方 (tools/turn_stats.py)
#   2. 新しいコースの topomap (bin/make_topomap.sh)
#   3. 7B の学習 (configs/finetune_7b.yaml. 検証は今までどおり 20260919_150554 を取り分けたまま)
#   4. 机上評価 (新しいモデル / 今のモデル = .env の FINETUNED_DIR / 元のモデル)
#        a. 取り分けた 20260919_150554 + course_a: どちらのモデルも学習に使っていない走行 = 学習方法の比較
#        b. 新しい bag + 新しいコース (新しいモデルは学習に使った走行なので甘めに出る)
#   5. まとめ: log/overnight_<時刻>/summary.txt (と debug_log/overnight_<時刻>/ に持ち帰り用の文字のファイル)
# 途中で失敗しても、できる所までは続ける (学習が失敗しても今のモデルの机上評価は回す).
source "$(dirname "$0")/_lib.sh"
not_on_jetson "夜の学習"
warn() { say "注意: $*" >&2; }
[ $# -ge 1 ] && [[ $1 != -* ]] || usage 1
BAG=$(cbags "$1")
shift
LOC=""
COURSE=course_b
WEIGHT=3
EDGE=0
while [ $# -gt 0 ]; do
    case $1 in
        --loc) LOC=$2; shift ;;
        --course) COURSE=$2; shift ;;
        --weight) WEIGHT=$2; shift ;;
        --edge) EDGE=1 ;;
        *) die "知らないオプション: $1" ;;
    esac
    shift
done
case ${BAG%%,*} in
    *.mcap | *.db3) NAME=$(basename "$(dirname "${BAG%%,*}")") ;;   # ROS 2 の bag のファイル -> そのフォルダの名前
    *) NAME=$(bag_name "$BAG") ;;
esac
CHILD_ARGS=("$BAG" --course "$COURSE" --weight "$WEIGHT")
[ -n "$LOC" ] && CHILD_ARGS+=(--loc "$LOC")
[ "$EDGE" = 1 ] && CHILD_ARGS+=(--edge)
VAL_BAG=$(env_get OVERNIGHT_VAL_BAG 20260919_150554)
VAL_TOPO=$(env_get OVERNIGHT_VAL_TOPOMAP course_a)
CUR=$(env_get FINETUNED_DIR "")
DRY=${OMNIVLA_DRY_RUN:-0}

# ----------------------------------------------------------------------------------------------- 確認 (端末で)
if [ "${OVERNIGHT_STAGE:-}" != run ]; then
    STAMP=$(date +%Y%m%d_%H%M%S)
    say "新しい bag: $BAG (学習データでの名前 $NAME), コース: $COURSE, 新しい bag を ${WEIGHT} 倍"
    if [ "$DRY" != 1 ]; then
        # 学習・推論のコンテナが動いていると GPU のメモリが足りなくなる
        BUSY=$(command docker ps --format '{{.Names}}' 2> /dev/null | grep -i omnivla || true)
        [ -z "$BUSY" ] || die "動いているコンテナがあります (GPU を使う): $BUSY  -> 止めてから: docker stop <名前> (bin/replay.sh down など)"
        if command -v nvidia-smi > /dev/null; then
            USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null | head -1 || true)
            [ -z "$USED" ] || [ "$USED" -lt 4000 ] || warn "GPU のメモリが ${USED} MiB 使われています (学習で足りなくなるかも)"
        else
            warn "nvidia-smi がありません (GPU が見えていない?)"
        fi
        AVAIL=$(df -Pk "$(host_dir runs)" 2> /dev/null | awk 'NR==2 {print int($4 / 1024 / 1024)}')
        [ -z "$AVAIL" ] || [ "$AVAIL" -ge 30 ] || warn "runs の空きが ${AVAIL} GB (30 GB 以上あると安心)"
    fi
    if [ -n "$LOC" ]; then
        RY="$ROOT/configs/robot.yaml"
        if [ "$DRY" = 1 ]; then
            echo "+ $RY: topics.eval_localization: $LOC"
        elif grep -qE '^[[:space:]]+eval_localization:' "$RY"; then
            sed -i -E "s|^([[:space:]]+eval_localization:[[:space:]]*)[^ #]*|\\1$LOC|" "$RY"
        else
            sed -i -E "/^[[:space:]]+localization:/a\\  eval_localization: $LOC" "$RY"
        fi
        say "configs/robot.yaml: topics.eval_localization = $LOC (評価だけに使う)"
    fi
    say "bag のトピックを確かめます"
    if ! in_shell python3 tools/bag_info.py "$BAG" --robot configs/robot.yaml; then
        die "configs/robot.yaml の画像かオドメトリのトピックが bag にありません. 上の候補を見て robot.yaml を直してから、もう一度"
    fi
    for p in "/bags/$VAL_BAG.bag" "/data/topomaps/$VAL_TOPO"; do
        [ -e "$(hpath "$p")" ] || warn "$p がありません (取り分けた走行での比較は飛ばします)"
    done
    [ -n "$CUR" ] || warn ".env に FINETUNED_DIR がありません (今のモデルとの比較は飛ばします)"
    LOGDIR="$ROOT/log/overnight_$STAMP"
    mkdir -p "$LOGDIR"
    if [ "$DRY" = 1 ]; then
        OVERNIGHT_STAGE=run OVERNIGHT_STAMP=$STAMP bash "$ROOT/bin/overnight.sh" "${CHILD_ARGS[@]}"
        exit 0
    fi
    OVERNIGHT_STAGE=run OVERNIGHT_STAMP=$STAMP setsid nohup bash "$ROOT/bin/overnight.sh" "${CHILD_ARGS[@]}" \
        > "$LOGDIR/overnight.log" 2>&1 < /dev/null &
    say "裏で始めました (pid $!). 端末を閉じても続きます. 7B の学習 6〜7 時間 + 机上評価 1 時間くらい"
    say "進み具合: tail -f log/overnight_$STAMP/overnight.log"
    say "朝: log/overnight_$STAMP/summary.txt"
    exit 0
fi

# ----------------------------------------------------------------------------------------------- 本番 (裏で)
STAMP=${OVERNIGHT_STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT=/runs/overnight_$STAMP
LOGDIR="$ROOT/log/overnight_$STAMP"
mkdir -p "$LOGDIR"
RUN=omnivla7b_overnight_$STAMP
EDGE_RUN=edge_overnight_$STAMP
FAILED=()
stage() { echo; say "==== $(date '+%H:%M:%S') $*"; }
try() {
    if "$@"; then
        return 0
    fi
    warn "失敗しました: $*"
    FAILED+=("$*")
    return 1
}
# 机上評価 1 回: desk TAG BAG TOPOMAP --finetuned_dir ...
desk() {
    local tag=$1 bag=$2 topo=$3
    shift 3
    stage "机上評価 $tag ($bag / $topo)"
    try "$ROOT/bin/desk_eval.sh" "$bag" "$topo" --out "$OUT/desk_eval_$tag" "$@" || true
    summary
}
summary() {
    [ "$DRY" = 1 ] && return 0
    python3 - "$(hpath "$OUT")" "$LOGDIR/summary.txt" "$NAME" "$RUN" << 'PY' || true
import glob, json, os, sys
out, dst, name, run = sys.argv[1:5]
rows = []
for d in sorted(glob.glob(os.path.join(out, "desk_eval_*"))):
    p = os.path.join(d, "report.json")
    if not os.path.exists(p):
        rows.append((os.path.basename(d)[10:], None))
        continue
    rows.append((os.path.basename(d)[10:], json.load(open(p))))
def f(v, n=2):
    return "-" if v is None else f"{v:.{n}f}"
L = [f"夜の学習のまとめ  新しい bag: {name}  新しいモデル: /runs/{run}", "",
     "曲がる量の比 (turn_gain): 予測した 8 点目の向き / 実際 (10 度以上曲がる所). 1 に近いほどよい",
     "遅れ: 予測が実際より何フレーム遅れているか (1 フレーム = 1/3 秒)", "",
     f"{'評価':34s} {'曲がる量の比':>10s} {'遅れ':>5s} {'ADE[m]':>7s} {'曲がる所の向き誤差':>14s} {'相関(w)':>8s} "
     f"{'最後のサブゴール':>14s} {'|番号-正解| p90':>14s}"]
for tag, r in rows:
    if r is None:
        L.append(f"{tag:34s} (失敗 / まだ)")
        continue
    w, c, g = r.get("waypoints", {}), r.get("commands", {}), r.get("subgoals", {})
    lag = w.get("turn_lag_frames")
    L.append(f"{tag:34s} {f(w.get('turn_gain')):>10s} {('-' if lag is None else str(-lag)):>5s} {f(w.get('ade_m'), 3):>7s} "
             f"{f((w.get('turn') or {}).get('heading_err_deg'), 1):>14s} {f(c.get('w_corr')):>8s} "
             f"{str(g.get('last_subgoal', '-')) + '/' + str(g.get('num_nodes', 1) - 1) if g else '-':>14s} "
             f"{f(g.get('index_minus_truth_abs_p90'), 1) if g else '-':>14s}")
txt = "\n".join(L) + "\n"
open(dst, "w").write(txt)
print(txt)
PY
}

stage "1. 学習データに追加: $BAG"
try "$ROOT/bin/make_dataset.sh" "$BAG" --overwrite || true
stage "運転の仕方 (倒し切り % が高い bag は label_smooth_sec でならす. 設定は configs/finetune_7b.yaml)"
try "$ROOT/bin/shell.sh" python3 tools/turn_stats.py /data/dataset || true

stage "2. 新しいコースの topomap: $COURSE"
try "$ROOT/bin/make_topomap.sh" "$COURSE" "$BAG" --overwrite || true

stage "3. 7B の学習: /runs/$RUN (新しい bag を ${WEIGHT} 倍, 検証は $VAL_BAG を取り分け)"
NEW=""
if try "$ROOT/bin/train.sh" 7b --run_name "$RUN" --bag_weights "{\"$NAME\": $WEIGHT}"; then
    L=$(cat "$(hpath "/runs/$RUN/checkpoints/latest.txt")" 2> /dev/null || true)
    NEW=${L:-/runs/$RUN/checkpoints/step_005000}
fi
[ "$DRY" = 1 ] && NEW=/runs/$RUN/checkpoints/step_005000

stage "4. 机上評価"
HAVE_VAL=0
[ -e "$(hpath "/bags/$VAL_BAG.bag")" ] && [ -e "$(hpath "/data/topomaps/$VAL_TOPO")" ] && HAVE_VAL=1
[ "$DRY" = 1 ] && HAVE_VAL=1
if [ "$HAVE_VAL" = 1 ]; then
    [ -n "$NEW" ] && desk "a_held_out_new_model" "/bags/$VAL_BAG.bag" "$VAL_TOPO" --finetuned_dir "$NEW"
    [ -n "$CUR" ] && desk "a_held_out_current_model" "/bags/$VAL_BAG.bag" "$VAL_TOPO" --finetuned_dir "$CUR"
fi
[ -n "$NEW" ] && desk "b_new_course_new_model" "$BAG" "$COURSE" --finetuned_dir "$NEW"
[ -n "$CUR" ] && desk "b_new_course_current_model" "$BAG" "$COURSE" --finetuned_dir "$CUR"
desk "b_new_course_original_model" "$BAG" "$COURSE" --finetuned_dir ""

if [ "$EDGE" = 1 ]; then
    stage "edge の学習: /runs/$EDGE_RUN"
    if try "$ROOT/bin/train.sh" edge --run_name "$EDGE_RUN" --bag_weights "{\"$NAME\": $WEIGHT}"; then
        L=$(ls -1d "$(hpath "/runs/$EDGE_RUN/checkpoints")"/step_* 2> /dev/null | tail -1 || true)
        E=/runs/$EDGE_RUN/checkpoints/$(basename "${L:-step_010000}")
        [ "$HAVE_VAL" = 1 ] && desk "a_held_out_edge" "/bags/$VAL_BAG.bag" "$VAL_TOPO" --model edge --weights "$E" --finetuned_dir ""
        desk "b_new_course_edge" "$BAG" "$COURSE" --model edge --weights "$E" --finetuned_dir ""
    fi
fi

stage "5. まとめ"
summary
if [ "$DRY" != 1 ]; then
    KEEP="$ROOT/debug_log/overnight_$STAMP"
    mkdir -p "$KEEP"
    cp "$LOGDIR/summary.txt" "$LOGDIR/overnight.log" "$KEEP/" 2> /dev/null || true
    for d in "$(hpath "$OUT")"/desk_eval_*; do
        [ -d "$d" ] || continue
        mkdir -p "$KEEP/$(basename "$d")"
        cp "$d"/report.txt "$d"/report.json "$d"/overview.png "$d"/timeline.png "$KEEP/$(basename "$d")/" 2> /dev/null || true
    done
    cp "$(hpath "/runs/$RUN/metrics.csv")" "$KEEP/train_metrics.csv" 2> /dev/null || true
    cp "$(hpath "/data/topomaps/$COURSE/overview.png")" "$KEEP/${COURSE}_overview.png" 2> /dev/null || true
    say "持ち帰り用: debug_log/overnight_$STAMP/ (git add debug_log/overnight_$STAMP -> commit -> push で共有)"
fi
if [ ${#FAILED[@]} -gt 0 ]; then
    warn "失敗したもの: ${FAILED[*]}"
fi
say "新しいモデル: ${NEW:-(学習に失敗)}  コース: /data/topomaps/$COURSE"
say "終わり: $(date '+%Y-%m-%d %H:%M:%S')"
