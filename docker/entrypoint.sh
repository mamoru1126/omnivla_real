#!/bin/bash
# ROS 2 と本リポジトリの環境を読み込んでからコマンドを実行する
set -e
if [ -f /opt/ros/humble/setup.bash ]; then
  source /opt/ros/humble/setup.bash
fi
export OMNIVLA_REAL_ROOT="${OMNIVLA_REAL_ROOT:-/workspace}"
export PYTHONPATH="${OMNIVLA_REAL_ROOT}:${OMNIVLA_ROOT:-/opt/OmniVLA}${PYTHONPATH:+:${PYTHONPATH}}"
WS=/opt/omnivla_ws
# マウントしたソースで ROS 2 パッケージを (再) ビルド (symlink-install なので Python の編集は即反映)
if [ "${OMNIVLA_SKIP_BUILD:-0}" != "1" ] && [ -d "${OMNIVLA_REAL_ROOT}/ros2" ] && command -v colcon >/dev/null; then
  mkdir -p "${WS}"
  colcon --log-base "${WS}/log" build --symlink-install --base-paths "${OMNIVLA_REAL_ROOT}/ros2" \
      --build-base "${WS}/build" --install-base "${WS}/install" > "${WS}/build.log" 2>&1 \
      || { echo "[entrypoint] colcon build failed:" >&2; tail -n 30 "${WS}/build.log" >&2; }
fi
if [ -f "${WS}/install/setup.bash" ]; then
  source "${WS}/install/setup.bash"
fi
exec "$@"
