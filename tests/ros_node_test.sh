#!/usr/bin/env bash
# ROS ノードの結合テスト (CI 用): 合成 bag をシミュレーション時刻で再生し、ノードが
# カメラ画像 (+オドメトリ) から 指示値 /cmd_vel と 予測軌跡 /omnivla/path を出すことを確認する。
#
#   bash tests/ros_node_test.sh ros2 E2E_DIR     # ROS 2 Humble (colcon build 済みの install を source してから)
#   bash tests/ros_node_test.sh ros1 E2E_DIR     # ROS 1 Noetic
# E2E_DIR は tests/e2e_cli.sh の出力 (bags/, topomap/, runs/edge_ci/checkpoints/step_*).
# POLICY_URL=http://127.0.0.1:8765 を付けると、ノードの中では推論せず推論サーバを呼ぶ (実機の構成).
# ROS 1 は NODE_IMPL=cpp (既定. C++ ノード + デバッグ画面の確認. POLICY_URL が必要) | py (Python ノード)
set -uo pipefail
WHICH=$1
E2E=$(cd "$2" && pwd)
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export OMNIVLA_REAL_ROOT=$ROOT
export PYTHONUNBUFFERED=1
CKPT=$(ls -d "$E2E"/runs/edge_ci/checkpoints/step_* 2>/dev/null | tail -1)
OUT="$E2E/${WHICH}_node"
rm -rf "$OUT"
mkdir -p "$OUT"
PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done; sleep 1; }
trap cleanup EXIT

POLICY_URL=${POLICY_URL:-}
EXTRA2=()
if [ -n "$POLICY_URL" ]; then EXTRA2=(-p policy_url:="$POLICY_URL"); fi
if [ "$WHICH" = ros2 ]; then
    ros2 run omnivla_real_ros navigator --ros-args -p topomap:="$E2E/topomap" -p model:=edge -p weights:="$CKPT" \
        -p device:=cpu -p log_dir:="$OUT/nav" "${EXTRA2[@]}" -p use_sim_time:=true > "$OUT/node.log" 2>&1 &
    PIDS+=($!)
else
    NODE_IMPL=${NODE_IMPL:-cpp}
    roscore > "$OUT/roscore.log" 2>&1 &
    PIDS+=($!)
    for _ in $(seq 30); do rostopic list > /dev/null 2>&1 && break; sleep 1; done
    rosparam set /use_sim_time true
    if [ "$NODE_IMPL" = cpp ]; then
        rosrun omnivla_real_ros1 navigator _repo_root:="$ROOT" _topomap:="$E2E/topomap" \
            _policy_url:="${POLICY_URL:-http://127.0.0.1:8765}" _log_dir:="$OUT/nav" _web_port:=8090 > "$OUT/node.log" 2>&1 &
    else
        python3 "$ROOT/ros1/omnivla_real_ros1/scripts/navigator_node.py" _topomap:="$E2E/topomap" _model:=edge \
            _weights:="$CKPT" _device:=cpu _log_dir:="$OUT/nav" _policy_url:="$POLICY_URL" > "$OUT/node.log" 2>&1 &
    fi
    PIDS+=($!)
fi

for _ in $(seq 180); do grep -q "ready:" "$OUT/node.log" && break; sleep 1; done
if ! grep -q "ready:" "$OUT/node.log"; then
    echo "node did not become ready"; cat "$OUT/node.log"
    echo "----- processes"; ps aux | grep -E "navigator|nav_config|policy" | grep -v grep | head
    exit 1
fi
echo "node ready"

python3 "$ROOT/tests/ros_check.py" "$WHICH" 50 > "$OUT/check.txt" 2>&1 &
CHECK=$!
WEB=""
if [ "$WHICH" = ros1 ] && [ "${NODE_IMPL:-cpp}" = cpp ]; then
    python3 "$ROOT/tests/web_check.py" http://127.0.0.1:8090 45 > "$OUT/web_check.txt" 2>&1 &
    WEB=$!
fi
sleep 3
if [ "$WHICH" = ros2 ]; then
    ros2 bag play "$E2E/bags/course_sqlite3" --topics /camera/image_raw/compressed /odom --clock > "$OUT/play.log" 2>&1 &
else
    rosbag play --clock "$E2E/bags/course_ros1_0.bag" "$E2E/bags/course_ros1_1.bag" \
        --topics /camera/image_raw/compressed /odom > "$OUT/play.log" 2>&1 &
fi
PIDS+=($!)
if wait $CHECK; then rc=0; else rc=$?; fi
if [ -n "$WEB" ]; then
    if ! wait $WEB; then rc=1; fi
    echo "----- web_check"; cat "$OUT/web_check.txt"
fi
echo "----- check"; cat "$OUT/check.txt"
echo "----- node.log (tail)"; tail -40 "$OUT/node.log"
echo "----- play.log (tail)"; tail -10 "$OUT/play.log"
exit $rc
