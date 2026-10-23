#!/bin/bash
# ROS 2 Humble と colcon ワークスペースを読み込んでからコマンドを実行する.
# /workspace をマウントして C++ のソース (ros2/omnivla_real_ros2 と ros1/omnivla_real_ros1 の core) が変わっていたら、
# ここでビルドし直す (数分かかることがある).
#   entrypoint-ros2.sh --hash   ソースのハッシュを表示するだけ (Dockerfile で使う)
set -e
ROOT="${OMNIVLA_REAL_ROOT:-/workspace}"
WS=/opt/ros2_ws
src_hash() {
  (cd "$ROOT" && find ros2/omnivla_real_ros2 ros1/omnivla_real_ros1/src/core ros1/omnivla_real_ros1/include \
     ros1/omnivla_real_ros1/web ros1/omnivla_real_ros1/third_party -type f 2>/dev/null | sort | xargs cat | md5sum | cut -d' ' -f1)
}
if [ "${1:-}" = --hash ]; then src_hash; exit 0; fi
source /opt/ros/humble/setup.bash
export OMNIVLA_REAL_ROOT="$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [ "${OMNIVLA_SKIP_BUILD:-0}" != "1" ] && [ -d "$ROOT/ros2/omnivla_real_ros2" ]; then
  if [ "$(src_hash)" != "$(cat $WS/.src_hash 2>/dev/null)" ]; then
    echo "[entrypoint] C++ sources changed: colcon build (log: $WS/build.log)" >&2
    if (colcon --log-base $WS/log build --base-paths "$ROOT/ros2" --build-base $WS/build --install-base $WS/install \
          --cmake-args -DCMAKE_BUILD_TYPE=Release > $WS/build.log 2>&1); then
      src_hash > $WS/.src_hash
    else
      echo "[entrypoint] colcon build failed:" >&2; tail -n 30 $WS/build.log >&2
    fi
  fi
fi
if [ -f $WS/install/setup.bash ]; then
  source $WS/install/setup.bash
fi
# 空の変数は消す (.env で空のまま渡ってくる. Fast DDS は空のプロファイル名を読もうとして失敗する)
for v in FASTRTPS_DEFAULT_PROFILES_FILE CYCLONEDDS_URI ROS_LOCALHOST_ONLY RMW_IMPLEMENTATION; do
  if [ -z "${!v:-}" ]; then unset "$v"; fi
done
exec "$@"
