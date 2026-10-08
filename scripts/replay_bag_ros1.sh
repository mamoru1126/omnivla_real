#!/usr/bin/env bash
# ROS 1 の bag を再生して navigator (ROS 1 ノード) を動かす. 実機に載せる前の確認用 (ROS 1 コンテナの中で実行).
# 推論サーバ (tools/policy_server.py) を先に起動しておく (docker compose up -d policy).
#
#   bash scripts/replay_bag_ros1.sh BAG[,BAG2,...] TOPOMAP [OUT_DIR]
#
# bag からは robot.yaml の画像・オドメトリ・自己位置のトピックだけを流す (記録された /cmd_vel は流さない)。
# 出力: OUT_DIR/out.bag (/cmd_vel, /omnivla/path, /omnivla/status), OUT_DIR/nav/<時刻>/ (走行ログと図), node.log
# 走行中の様子は別の端末で: rqt_image_view /omnivla/debug_image
set -euo pipefail
BAGS=${1:?usage: replay_bag_ros1.sh BAG[,BAG2,...] TOPOMAP [OUT_DIR]}
TOPOMAP=${2:?topomap directory}
OUT=${3:-/runs/replay/$(date +%Y%m%d_%H%M%S)}
ROOT=${OMNIVLA_REAL_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
ROBOT=${ROBOT_CONFIG:-$ROOT/configs/robot.yaml}
NAV=${NAV_CONFIG:-$ROOT/configs/navigator.yaml}
URL=${POLICY_URL:-http://127.0.0.1:${POLICY_PORT:-8765}}
mkdir -p "$OUT"
IFS=',' read -r -a BAG_LIST <<< "$BAGS"
TOPICS=$(python3 - "$ROBOT" <<'EOF'
import sys, yaml
t = (yaml.safe_load(open(sys.argv[1])) or {}).get("topics", {})
print(" ".join(x for x in (t.get("image"), t.get("odom"), t.get("localization")) if x))
EOF
)
echo "bags: ${BAG_LIST[*]}"
echo "topics: $TOPICS"
echo "out: $OUT"

PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill -INT "$p" 2>/dev/null || true; done; sleep 2; }
trap cleanup EXIT

if ! rostopic list > /dev/null 2>&1; then
    roscore > "$OUT/roscore.log" 2>&1 &
    PIDS+=($!)
    for _ in $(seq 30); do rostopic list > /dev/null 2>&1 && break; sleep 1; done
fi
rosparam set /use_sim_time true

roslaunch omnivla_real_ros1 navigator.launch topomap:="$TOPOMAP" policy_url:="$URL" robot_config:="$ROBOT" \
    nav_config:="$NAV" log_dir:="$OUT/nav" use_sim_time:=true > "$OUT/node.log" 2>&1 &
PIDS+=($!)
for _ in $(seq 600); do grep -q "ready:" "$OUT/node.log" && break; sleep 1; done
grep -q "ready:" "$OUT/node.log" || { echo "navigator did not start:"; tail -30 "$OUT/node.log"; exit 1; }
echo "navigator ready"

rosbag record -O "$OUT/out.bag" /cmd_vel /omnivla/path /omnivla/status __name:=omnivla_record > /dev/null 2>&1 &
PIDS+=($!)
sleep 2
# shellcheck disable=SC2086
rosbag play --clock "${BAG_LIST[@]}" --topics $TOPICS
sleep 3
rosnode kill /omnivla_record > /dev/null 2>&1 || true
echo "----- navigator (tail)"
grep -E "start:|subgoal|finished|reached|ERROR" "$OUT/node.log" | tail -40 || true
if python3 -c "import matplotlib" 2> /dev/null && ls "$OUT"/nav/*/steps.csv > /dev/null 2>&1; then
    python3 "$ROOT/tools/plot_nav_log.py" "$OUT/nav/latest" | tail -5
fi
echo "-> $OUT"
