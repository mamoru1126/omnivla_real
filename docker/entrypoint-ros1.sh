#!/bin/bash
# ROS 1 Noetic と catkin ワークスペースを読み込んでからコマンドを実行する
set -e
source /opt/ros/noetic/setup.bash
if [ -f /opt/catkin_ws/devel/setup.bash ]; then
  source /opt/catkin_ws/devel/setup.bash
fi
export OMNIVLA_REAL_ROOT="${OMNIVLA_REAL_ROOT:-/workspace}"
export PYTHONPATH="${OMNIVLA_REAL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "$@"
