#!/bin/bash
# ROS 1 Noetic と catkin ワークスペースを読み込んでからコマンドを実行する.
# /workspace をマウントして C++ のソースが変わっていたら、ここでビルドし直す (数分かかることがある).
set -e
source /opt/ros/noetic/setup.bash
export OMNIVLA_REAL_ROOT="${OMNIVLA_REAL_ROOT:-/workspace}"
export PYTHONPATH="${OMNIVLA_REAL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
WS=/opt/catkin_ws
PKG="${OMNIVLA_REAL_ROOT}/ros1/omnivla_real_ros1"
src_hash() { (cd "$PKG" && find CMakeLists.txt package.xml include src web -type f | sort | xargs cat | md5sum | cut -d' ' -f1); }
if [ "${OMNIVLA_SKIP_BUILD:-0}" != "1" ] && [ -d "$PKG" ]; then
  if [ "$(src_hash)" != "$(cat $WS/.src_hash 2>/dev/null)" ]; then
    echo "[entrypoint] C++ sources changed: catkin_make (log: $WS/build.log)" >&2
    if (cd $WS && catkin_make -DCMAKE_BUILD_TYPE=Release > $WS/build.log 2>&1); then
      src_hash > $WS/.src_hash
    else
      echo "[entrypoint] catkin_make failed:" >&2; tail -n 30 $WS/build.log >&2
    fi
  fi
fi
if [ -f $WS/devel/setup.bash ]; then
  source $WS/devel/setup.bash
fi
exec "$@"
