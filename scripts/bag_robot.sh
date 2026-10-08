#!/usr/bin/env bash
# 実機の代わりに rosbag を流す (docker-compose.replay.yml の robot. ROS 1 コンテナの中で動く).
# robot.yaml のカメラ画像・オドメトリ・自己位置のトピックだけを、シミュレーション時刻 (--clock) で流す.
# 記録されていた /cmd_vel などは流さない (指示値はナビゲーションノードが出す).
#
#   REPLAY_BAG    bag (カンマ区切りで複数. ディレクトリなら中の *.bag 全部. /bags からの相対パスでもよい)
#   REPLAY_RATE   再生速度 (既定 1.0)
#   REPLAY_START  bag の先頭から何秒飛ばすか (既定 0)
#   REPLAY_ARGS   rosbag play にそのまま渡す (例 "-u 60" で 60 秒だけ)
#   REPLAY_WAIT   流し始める前に起動を待つノード (空白区切り). ナビゲーションノードは常に待つ
set -uo pipefail
trap 'exit 130' INT TERM
ROOT=${OMNIVLA_REAL_ROOT:-/workspace}
ROBOT=${ROBOT_CONFIG:-$ROOT/configs/robot.yaml}
NAV=${NAV_CONFIG:-$ROOT/configs/navigator.yaml}
NAV_NODE=${NAV_NODE:-/omnivla_navigator}
RATE=${REPLAY_RATE:-1.0}
START=${REPLAY_START:-0}
say() { echo "[robot] $*"; }

# --- bag の一覧 ---
BAGS=()
IFS=',' read -r -a ITEMS <<< "${REPLAY_BAG:-}"
for it in "${ITEMS[@]}"; do
    it=$(echo "$it" | xargs)
    [ -z "$it" ] && continue
    if [ ! -e "$it" ] && [ -e "/bags/$it" ]; then it=/bags/$it; fi
    if [ -d "$it" ]; then
        mapfile -t found < <(find "$it" -maxdepth 1 -name "*.bag" | sort -V)
        BAGS+=("${found[@]}")
    else
        BAGS+=("$it")
    fi
done
if [ ${#BAGS[@]} -eq 0 ]; then
    say "REPLAY_BAG が空です. 例: REPLAY_BAG=run2.bag (ホストの BAG_DIR が /bags)"
    exit 2
fi
for b in "${BAGS[@]}"; do
    [ -f "$b" ] || { say "bag が見つかりません: $b (ホストの BAG_DIR が /bags)"; exit 2; }
done

# --- 流すトピックとブラウザのポート (設定ファイルから) ---
read -r IMG TOPICS WEB_PORT < <(python3 - "$ROBOT" "$NAV" <<'EOF'
import sys, yaml
t = (yaml.safe_load(open(sys.argv[1])) or {}).get("topics", {})
io = (yaml.safe_load(open(sys.argv[2])) or {}).get("io", {})
topics = [x for x in (t.get("image"), t.get("odom"), t.get("localization")) if x]
print(t.get("image"), ",".join(topics), io.get("web_port", 8080))
EOF
)
[ -n "$IMG" ] && [ "$IMG" != None ] || { say "$ROBOT の topics.image が空です"; exit 2; }
WEB_PORT=${NAV_WEB_PORT:-$WEB_PORT}
IFS=',' read -r -a TOPIC_LIST <<< "$TOPICS"

# --- ROS master と、ノードの準備を待つ ---
until rostopic list > /dev/null 2>&1; do sleep 1; done
rosparam set /use_sim_time true
t0=$SECONDS
last=-30
until rostopic info "$IMG" 2> /dev/null | grep -qF "* $NAV_NODE ("; do
    if (( SECONDS - last >= 30 )); then
        say "$NAV_NODE が $IMG を受け取れるようになるのを待っています ($((SECONDS - t0)) s. 推論サーバのモデルの読み込みに数分かかることがあります)"
        last=$SECONDS
    fi
    sleep 1
done
for n in ${REPLAY_WAIT:-}; do
    for _ in $(seq 60); do rosnode list 2> /dev/null | grep -qx "$n" && break; sleep 1; done
done

DUR=0
for b in "${BAGS[@]}"; do
    d=$(rosbag info --yaml --key=duration "$b" 2> /dev/null || echo 0)
    DUR=$(awk -v a="$DUR" -v b="$d" 'BEGIN { printf "%.1f", a + b }')
done
say "bag: ${BAGS[*]} (${DUR} s, 速度 x$RATE, ${START} s から)"
say "流すトピック: ${TOPIC_LIST[*]}"
say "ブラウザ: http://localhost:$WEB_PORT  (別の PC からは http://<この PC の IP>:$WEB_PORT)"

# shellcheck disable=SC2086
rosbag play --clock -q -d 2 -r "$RATE" -s "$START" ${REPLAY_ARGS:-} "${BAGS[@]}" --topics "${TOPIC_LIST[@]}" &
PLAY=$!
# 裏で動かしたコマンドには SIGINT が届かないので TERM で止める
trap 'kill -TERM $PLAY 2> /dev/null; exit 130' INT TERM
wait $PLAY
rc=$?
trap - INT TERM
say "bag の再生が終わりました (rc=$rc). 画面はそのまま見られます: http://localhost:$WEB_PORT"
say "記録: /runs/replay (ホストの RUNS_DIR/replay). out_<時刻>.bag と nav/<時刻>/ (python3 tools/plot_nav_log.py で図に)"
say "もう一度流す: docker compose -f docker-compose.replay.yml restart nav && docker compose -f docker-compose.replay.yml restart robot"
exit $rc
