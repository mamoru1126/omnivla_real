#!/usr/bin/env bash
# 8. Jetson で走らせる (docker-compose.jetson.yml). Jetson の上で実行する.
#
#   bin/jetson.sh build               # イメージを作る (JetPack に合わせた JETSON_BASE_IMAGE は .env)
#   bin/jetson.sh up                  # 推論サーバとナビゲーションを起動してログを表示 (Ctrl-C でログの表示だけ止まる)
#   bin/jetson.sh up --standalone     # roscore もここで立てる (ロボット側に master がない場合)
#   bin/jetson.sh restart             # ナビゲーションだけ起動し直す (推論サーバはそのまま)
#   bin/jetson.sh status              # コンテナの状態と推論サーバの応答
#   bin/jetson.sh logs [サービス]      # ログ (policy, nav, roscore)
#   bin/jetson.sh plot [ログ]          # 走行ログを図にする (既定: 一番新しい走行)
#   bin/jetson.sh down                # 止める
# ブラウザ: http://<Jetson の IP>:8080   重みと topomap は学習 PC から bin/to_jetson.sh で送る
source "$(dirname "$0")/_lib.sh"

ip_addr() { hostname -I 2> /dev/null | awk '{print $1}'; }
web_port() { env_get NAV_WEB_PORT "$(sed -n -E 's/^[[:space:]]*web_port:[[:space:]]*([0-9]+).*/\1/p' configs/navigator.yaml | head -1)"; }
follow() {
    say "ブラウザ: http://$(ip_addr):$(web_port)"
    say "Ctrl-C でログの表示だけ止まります (走ったまま). 止める: bin/jetson.sh down"
    jetson logs -f --tail 50 policy nav
}

CMD=${1:-}
shift || true
case $CMD in
    build)
        ensure_env
        say "ベースイメージ: $(env_get JETSON_BASE_IMAGE dustynv/l4t-pytorch:r36.4.0)  7B: $(env_get WITH_7B 1)"
        jetson build "$@"
        ;;
    up)
        ensure_env
        PROFILE=()
        if [ "${1:-}" = --standalone ]; then
            PROFILE=(--profile standalone)
            export ROSLAUNCH_ARGS=--wait   # roscore コンテナの master を待つ
        fi
        T=$(env_get TOPOMAP "")
        [ -n "$T" ] || say "注意: .env の TOPOMAP が空です (ブラウザや /omnivla/topomap で指定するまで走りません)"
        [ -z "$T" ] || [ -e "$(hpath "$T")" ] || die "TOPOMAP が見つかりません: $T (ホストでは $(hpath "$T"))"
        say "モデル: $(env_get NAV_MODEL '(navigator.yaml)') $(env_get FINETUNED_DIR '')$(env_get NAV_WEIGHTS '')  topomap: $T"
        say "ROS master: $(env_get ROS_MASTER_URI http://localhost:11311)"
        ensure_image omnivla_real:ros1 jetson build nav
        jetson "${PROFILE[@]}" up -d
        follow
        ;;
    restart)
        jetson restart nav
        follow
        ;;
    status)
        jetson --profile standalone ps
        PORT=$(env_get POLICY_PORT 8765)
        if command -v curl > /dev/null; then
            echo "policy server (:$PORT/info):"
            curl -sf --noproxy '*' "http://127.0.0.1:$PORT/info" && echo || echo "  応答なし (起動中か止まっている)"
        fi
        ;;
    logs)
        jetson --profile standalone logs -f --tail 100 "$@"
        ;;
    plot)
        LOGDIR=$(env_get NAV_LOG_DIR /workspace/log/nav)
        DIR=${1:-$LOGDIR/latest}
        case $DIR in /*) ;; *) DIR=$(cpath runs "" "$DIR") ;; esac
        jetson run --rm --no-deps "${TTY[@]}" nav python3 tools/plot_nav_log.py "$DIR"
        ;;
    down | stop)
        jetson --profile standalone down
        ;;
    *) usage 1 ;;
esac
