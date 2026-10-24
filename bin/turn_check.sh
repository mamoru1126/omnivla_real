#!/usr/bin/env bash
# 曲がりが足りない原因の切り分け (学習はしない. 机上評価を 7 回, 1 時間半ほど. 裏で動く).
# 結果: log/turn_check_<時刻>/summary.txt (と debug_log/turn_check_<時刻>/ に持ち帰り用の文字のファイル)
#
#   bin/turn_check.sh <新しいコースの bag> [--new /runs/<run>/checkpoints/step_XXXXXX]
#     --new  学習し直したモデル (省くと一番新しい runs/omnivla7b_overnight_*). 今のモデルは .env の FINETUNED_DIR
#
# 見ること (どれも同じ bag で、条件を 1 つずつ変える):
#   1/2  取り分けた走行 (20260919_150554) で、ゴールを理想 (同じ走行の 1.5 m 先) にした時の新旧モデル
#   3    新しいコースで理想のゴール                         -> モデル自体がどこまで曲がれるか
#   4    新しいコースで実機と同じゴール (正解はオドメトリ)    -> 3 との差 = サブゴールの選び方の影響
#   5    4 + サブゴール切り替えのしきい値を厳しく (0.97)      -> 画像の判定が甘くて先へ進みすぎていないか
#   6    4 + サブゴールを自己位置で切り替え (理想の切り替え)  -> 切り替えが正しければどこまで曲がれるか
#   7    4 の正解を自己位置にした時                          -> 正解の位置の出所での違い
source "$(dirname "$0")/_lib.sh"
not_on_jetson "曲がりの切り分け"
warn() { say "注意: $*" >&2; }
[ $# -ge 1 ] && [[ $1 != -* ]] || usage 1
BAG=$(cbags "$1")
shift
NEW=""
while [ $# -gt 0 ]; do
    case $1 in
        --new) NEW=$(cpath runs "" "$2"); shift ;;
        *) die "知らないオプション: $1" ;;
    esac
    shift
done
DRY=${OMNIVLA_DRY_RUN:-0}
if [ -z "$NEW" ]; then
    L=$(ls -1dt "$(host_dir runs)"/omnivla7b_overnight_*/checkpoints/latest.txt 2> /dev/null | head -1 || true)
    [ -n "$L" ] && NEW=$(cat "$L")
    [ "$DRY" = 1 ] && NEW=${NEW:-/runs/omnivla7b_overnight_x/checkpoints/step_005000}
    [ -n "$NEW" ] || die "学習し直したモデルが見つかりません: --new /runs/<run>/checkpoints/step_XXXXXX"
fi
CUR=$(env_get FINETUNED_DIR "")
HB=/bags/$(env_get OVERNIGHT_VAL_BAG 20260919_150554).bag
HT=$(env_get OVERNIGHT_VAL_TOPOMAP course_a)
COURSE=$(env_get TURN_CHECK_COURSE course_b)
RY="$ROOT/configs/robot.yaml"
LOCT=$(sed -n -E 's/^[[:space:]]+eval_localization:[[:space:]]*"?([^" #]+)"?.*/\1/p' "$RY" | head -1)

if [ "${TURN_CHECK_STAGE:-}" != run ]; then
    say "新しいモデル: $NEW"
    say "今のモデル:   ${CUR:-(.env の FINETUNED_DIR が無い -> 比較を飛ばす)}"
    say "bag: $BAG  コース: $COURSE  取り分けた走行: $HB / $HT  自己位置: ${LOCT:-(無し -> 6, 7 を飛ばす)}"
    if [ "$DRY" != 1 ]; then
        BUSY=$(command docker ps --format '{{.Names}}' 2> /dev/null | grep -i omnivla || true)
        [ -z "$BUSY" ] || die "動いているコンテナがあります (GPU を使う): $BUSY"
    fi
    STAMP=$(date +%Y%m%d_%H%M%S)
    LOGDIR="$ROOT/log/turn_check_$STAMP"
    mkdir -p "$LOGDIR"
    ARGS=("$BAG" --new "$NEW")
    if [ "$DRY" = 1 ]; then
        TURN_CHECK_STAGE=run TURN_CHECK_STAMP=$STAMP bash "$ROOT/bin/turn_check.sh" "${ARGS[@]}"
        exit 0
    fi
    TURN_CHECK_STAGE=run TURN_CHECK_STAMP=$STAMP setsid nohup bash "$ROOT/bin/turn_check.sh" "${ARGS[@]}" \
        > "$LOGDIR/turn_check.log" 2>&1 < /dev/null &
    say "裏で始めました (pid $!). 1 時間半くらい"
    say "進み具合: tail -f log/turn_check_$STAMP/turn_check.log"
    say "結果: cat log/turn_check_$STAMP/summary.txt"
    exit 0
fi

STAMP=${TURN_CHECK_STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT=/runs/turn_check_$STAMP
LOGDIR="$ROOT/log/turn_check_$STAMP"
mkdir -p "$LOGDIR"
# 切り替えのしきい値を厳しくした navigator.yaml / 自己位置でサブゴールを切り替える robot.yaml (この確認だけで使う)
sed -E -e 's/^([[:space:]]+image_threshold:[[:space:]]*)[0-9.]+/\10.97/' \
    -e 's/^([[:space:]]+goal_image_threshold:[[:space:]]*)[0-9.]+/\10.98/' "$ROOT/configs/navigator.yaml" > "$LOGDIR/navigator_strict.yaml"
if [ -n "$LOCT" ]; then
    sed -E "s|^([[:space:]]+localization:[[:space:]]*)\"\"|\\1$LOCT|" "$RY" > "$LOGDIR/robot_pose.yaml"
fi
W=/workspace/log/turn_check_$STAMP
stage() { echo; say "==== $(date '+%H:%M:%S') $*"; }
desk() {
    local tag=$1
    shift
    stage "$tag"
    "$ROOT/bin/desk_eval.sh" "$@" --out "$OUT/$tag" || warn "失敗しました: $tag"
    [ "$DRY" = 1 ] || python3 "$ROOT/tools/desk_eval_table.py" --root "$(hpath "$OUT")" --out "$LOGDIR/summary.txt" > /dev/null || true
}
desk 1_held_out_hindsight_new "$HB" --goal_mode hindsight --pose_source odom --finetuned_dir "$NEW"
[ -n "$CUR" ] && desk 2_held_out_hindsight_current "$HB" --goal_mode hindsight --pose_source odom --finetuned_dir "$CUR"
desk 3_new_course_hindsight_new "$BAG" --goal_mode hindsight --pose_source odom --finetuned_dir "$NEW"
desk 4_new_course_topomap_new "$BAG" "$COURSE" --pose_source odom --finetuned_dir "$NEW"
desk 5_new_course_topomap_strict_new "$BAG" "$COURSE" --pose_source odom --finetuned_dir "$NEW" --nav "$W/navigator_strict.yaml"
if [ -n "$LOCT" ]; then
    desk 6_new_course_topomap_pose_new "$BAG" "$COURSE" --pose_source odom --finetuned_dir "$NEW" --robot "$W/robot_pose.yaml"
    desk 7_new_course_topomap_locgt_new "$BAG" "$COURSE" --pose_source localization --finetuned_dir "$NEW"
fi
stage "まとめ"
if [ "$DRY" != 1 ]; then
    python3 "$ROOT/tools/desk_eval_table.py" --root "$(hpath "$OUT")" --out "$LOGDIR/summary.txt"
    KEEP="$ROOT/debug_log/turn_check_$STAMP"
    mkdir -p "$KEEP"
    cp "$LOGDIR/summary.txt" "$LOGDIR/turn_check.log" "$KEEP/" 2> /dev/null || true
    for d in "$(hpath "$OUT")"/*/; do
        d=${d%/}
        mkdir -p "$KEEP/$(basename "$d")"
        cp "$d"/report.txt "$d"/steps.csv "$d"/timeline.png "$KEEP/$(basename "$d")/" 2> /dev/null || true
    done
    say "持ち帰り用: debug_log/turn_check_$STAMP/ (git add -> commit -> push で共有)"
fi
say "終わり: $(date '+%Y-%m-%d %H:%M:%S')"
