#!/usr/bin/env bash
# ROS 2 のナビゲーションノード (C++) を環境変数から起動する (docker-compose.jetson-ros2.yml の nav).
# 空の値は渡さない (navigator.yaml の値になる. ROS 2 は「-p topomap:=」のような空の値を受け付けない).
#   TOPOMAP NAV_AUTOSTART POLICY_PORT NAV_WEB_PORT NAV_LOG_DIR NAV_OVERRIDES CMD_VEL_TOPIC USE_SIM_TIME
set -euo pipefail
ARGS=(-p "policy_url:=http://127.0.0.1:${POLICY_PORT:-8765}")
[ -n "${TOPOMAP:-}" ] && ARGS+=(-p "topomap:=$TOPOMAP")
[ -n "${NAV_AUTOSTART:-}" ] && ARGS+=(-p "autostart:=$NAV_AUTOSTART")
[ -n "${NAV_WEB_PORT:-}" ] && ARGS+=(-p "web_port:=$NAV_WEB_PORT")
[ -n "${NAV_LOG_DIR:-}" ] && ARGS+=(-p "log_dir:=$NAV_LOG_DIR")
[ -n "${USE_SIM_TIME:-}" ] && ARGS+=(-p "use_sim_time:=$USE_SIM_TIME")
OV=${NAV_OVERRIDES:-}
[ -n "${CMD_VEL_TOPIC:-}" ] && OV="$OV io.cmd_vel=$CMD_VEL_TOPIC"
OV=$(echo "$OV" | xargs)
[ -n "$OV" ] && ARGS+=(-p "overrides:=$OV")
echo "[nav_ros2] ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0} ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY:-0} ${ARGS[*]}"
exec ros2 run omnivla_real_ros2 navigator --ros-args "${ARGS[@]}"
