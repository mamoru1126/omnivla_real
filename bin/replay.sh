#!/usr/bin/env bash
# 7. 学習 PC で、bag をロボットの代わりにして実機と同じ構成を動かす (docker-compose.replay.yml). ブラウザで確認する.
# モデルは .env の NAV_MODEL / NAV_WEIGHTS / FINETUNED_DIR. topomap を省くと .env の TOPOMAP.
#
#   bin/replay.sh run2.bag course_a              # 起動して robot と nav のログを表示 (Ctrl-C でログの表示だけ止まる)
#   bin/replay.sh run2.bag course_a --rate 2 --start 30 --manual
#        --rate R    再生速度   --start S   先頭から S 秒飛ばす   --manual   ブラウザの「開始」で走り出す
#        --args "…"  rosbag play に渡す (例 "-u 60")
#   bin/replay.sh again           # もう一度最初から流す (nav と robot を起動し直す. 推論サーバはそのまま)
#   bin/replay.sh logs [サービス]  # ログ (policy, nav, robot, record, roscore)
#   bin/replay.sh down            # 止めて片付ける
# ブラウザ: http://localhost:8080  記録: runs/replay/ (out_<時刻>.bag, nav/<時刻>/ → bin/plot_nav_log.sh)
source "$(dirname "$0")/_lib.sh"

web_port() { env_get NAV_WEB_PORT "$(sed -n -E 's/^[[:space:]]*web_port:[[:space:]]*([0-9]+).*/\1/p' configs/navigator.yaml | head -1)"; }

follow() {
    say "ブラウザ: http://localhost:$(web_port)  (別の PC からは http://<この PC の IP>:$(web_port))"
    say "Ctrl-C でログの表示だけ止まります (動いたまま). もう一度: bin/replay.sh again  片付け: bin/replay.sh down"
    replay logs -f --tail 50 robot nav
}

case "${1:-}" in
    "") usage 1 ;;
    again)
        replay restart nav
        replay restart robot
        follow
        exit
        ;;
    logs)
        shift
        replay logs -f --tail 100 "$@"
        exit
        ;;
    down | stop)
        replay down
        exit
        ;;
esac

ensure_env
BAG=$(cbags "$1")
shift
TOPO=$(env_get TOPOMAP "")
if [ $# -gt 0 ] && [[ $1 != -* ]]; then
    TOPO=$1
    shift
fi
[ -n "$TOPO" ] || die "topomap を指定してください (引数か .env の TOPOMAP)"
TOPO=$(topomap_path "$TOPO")
while [ $# -gt 0 ]; do
    case $1 in
        --rate) export REPLAY_RATE=$2; shift ;;
        --start) export REPLAY_START=$2; shift ;;
        --args) export REPLAY_ARGS=$2; shift ;;
        --manual) export NAV_AUTOSTART=false ;;
        *) die "知らないオプション: $1 (bin/replay.sh -h)" ;;
    esac
    shift
done
export REPLAY_BAG=$BAG TOPOMAP=$TOPO
mkdir -p "$(host_dir runs)/replay"
say "bag: $BAG  topomap: $TOPO  model: $(env_get NAV_MODEL '(navigator.yaml)') $(env_get FINETUNED_DIR '')$(env_get NAV_WEIGHTS '')"
ensure_image omnivla_real:ros1 replay build nav
# 推論サーバと roscore は動いていればそのまま (モデルを読み直さない). nav・記録・robot は毎回作り直す
replay up -d roscore policy
replay up -d --force-recreate --no-deps nav record robot
follow
