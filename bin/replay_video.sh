#!/usr/bin/env bash
# bin/replay.sh と同じ構成 (推論サーバ + Jetson と同じ nav + bag をロボット代わり) で bag を最後まで流し、
# デバッグ画面と同じ内容 (カメラ画像 + 予測軌跡、サブゴール、指示値、俯瞰図、履歴、出来事) を動画に保存する.
# ブラウザは使わない. 動画は走行ログから作る (tools/nav_video.py, ffmpeg).
# 学習 PC でも Jetson でも同じ.
#
#   bin/replay_video.sh run2.bag course_a                    # -> runs/replay/<時刻>.mp4
#   bin/replay_video.sh run2.bag course_a --rate 2 --speed 4 # bag を 2 倍速で流し、動画は 4 倍速
#        --rate R    bag の再生速度   --start S  bag の先頭から S 秒飛ばす   --args "…" rosbag play に渡す
#        --speed X   動画の再生速度 (既定 1 = 実時間)   --fps N  (既定 10)   --out FILE  出力先
#   bin/replay_video.sh --log runs/replay/nav/latest         # 既にある走行ログから動画だけ作る
#   bin/replay_video.sh --log log/nav/20261013_120000        # Jetson から持ってきた実機の走行ログも同じ
# モデルは .env の NAV_MODEL / NAV_WEIGHTS / FINETUNED_DIR. topomap を省くと .env の TOPOMAP.
# 終わったあと推論サーバは動いたまま (次も速く始まる). 片付け: bin/replay.sh down
source "$(dirname "$0")/_lib.sh"

LOG=""
OUT=""
VIDEO_ARGS=()
POS=()
while [ $# -gt 0 ]; do
    case $1 in
        --log) LOG=$2; shift ;;
        --out) OUT=$2; shift ;;
        --speed | --fps) VIDEO_ARGS+=("$1" "$2"); shift ;;
        --rate) export REPLAY_RATE=$2; shift ;;
        --start) export REPLAY_START=$2; shift ;;
        --args) export REPLAY_ARGS=$2; shift ;;
        -*) die "知らないオプション: $1 (bin/replay_video.sh -h)" ;;
        *) POS+=("$1") ;;
    esac
    shift
done

# 動画を作る (ROS 1 イメージの中で. ffmpeg が無い古いイメージなら作り直す)
render() {
    local log=$1 out=$2
    ensure_image omnivla_real:ros1 replay build nav
    if [ "${OMNIVLA_DRY_RUN:-0}" != 1 ] && ! replay run --rm --no-deps -T robot bash -c 'command -v ffmpeg' > /dev/null 2>&1; then
        say "ROS 1 のイメージに ffmpeg が無いので作り直します"
        replay build nav
    fi
    replay run --rm --no-deps -T robot python3 tools/nav_video.py "$log" --out "$out" "${VIDEO_ARGS[@]}"
    say "動画: $(hpath "$out")"
}

# --- 既にある走行ログから ---
if [ -n "$LOG" ]; then
    [ ${#POS[@]} -eq 0 ] || die "--log のときは bag を指定しない"
    L=$(cpath runs "" "$LOG")
    case $L in
        */latest)   # 一番新しい走行に置き換える
            need "${L%/latest}"
            newest=$(cd "$(hpath "${L%/latest}")" && ls -1d -- */ 2> /dev/null | sort | tail -1)
            [ -n "$newest" ] || die "走行ログがありません: $(hpath "${L%/latest}")"
            L="${L%/latest}/${newest%/}"
            ;;
        *) need "$L" ;;
    esac
    O=$(cpath_new "${OUT:-$(hpath "$L")/video.mp4}")
    render "$L" "$O"
    exit 0
fi

# --- bag を流してから ---
[ ${#POS[@]} -ge 1 ] || usage 1
ensure_env
BAG=$(cbags "${POS[0]}")
TOPO=${POS[1]:-$(env_get TOPOMAP "")}
[ -n "$TOPO" ] || die "topomap を指定してください (引数か .env の TOPOMAP)"
TOPO=$(topomap_path "$TOPO")
[ -z "$OUT" ] || cpath_new "$OUT" > /dev/null   # 出力先を先に確かめる (走らせてから失敗しない)
export REPLAY_BAG=$BAG TOPOMAP=$TOPO NAV_AUTOSTART=true
export NAV_WEB_PORT=0                      # ブラウザの画面は使わない
export NAV_OVERRIDES="io.log_images=true"  # 動画にするカメラ画像をログに残す
NAVLOG=$(host_dir runs)/replay/nav
mkdir -p "$NAVLOG"
before=$(ls -1 "$NAVLOG" 2> /dev/null | sort)

say "bag: $BAG  topomap: $TOPO  model: $(env_get NAV_MODEL '(navigator.yaml)') $(env_get FINETUNED_DIR '')$(env_get NAV_WEIGHTS '')"
jetson_replay_note
ensure_image omnivla_real:ros1 replay build nav
replay up -d roscore policy
replay up -d --force-recreate --no-deps nav record robot
trap 'say "中断します"; replay stop nav record robot > /dev/null 2>&1; exit 130' INT TERM

# bag が最後まで流れるのを待つ (robot の様子を表示)
if [ "${OMNIVLA_DRY_RUN:-0}" = 1 ]; then
    run_dir=DRYRUN
else
    CID=$(replay ps -a -q robot)
    replay logs -f --no-log-prefix robot &
    LOGS=$!
    docker wait "$CID" > /dev/null
    kill "$LOGS" 2> /dev/null || true
    sleep 2
    replay stop nav record robot > /dev/null 2>&1   # 走行ログを閉じる
    run_dir=$(comm -13 <(echo "$before") <(ls -1 "$NAVLOG" | sort) | tail -1)
    [ -n "$run_dir" ] || die "走行ログが見つかりません ($NAVLOG). nav のログ: bin/replay.sh logs nav"
fi
trap - INT TERM
O=$(cpath_new "${OUT:-/runs/replay/$run_dir.mp4}")
render "/runs/replay/nav/$run_dir" "$O"
say "走行ログ: $NAVLOG/$run_dir   (図: bin/plot_nav_log.sh runs/replay/nav/$run_dir)"
