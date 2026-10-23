"""OmniVLA 実機ナビゲーション (ROS 2, C++ ノード + ブラウザのデバッグ画面).

  ros2 launch omnivla_real_ros2 navigator.launch.py topomap:=/data/topomaps/course_a
  ros2 launch omnivla_real_ros2 navigator.launch.py cmd_vel:=/omnivla/cmd_vel/raw autostart:=false

推論は推論サーバ (tools/policy_server.py, 別コンテナ). ブラウザで http://<この PC>:8080.
空の引数は渡さない (navigator.yaml の値になる).
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node

ROOT = os.environ.get("OMNIVLA_REAL_ROOT", "/workspace")
ARGS = {
    "repo_root": ROOT,
    "robot_config": os.path.join(ROOT, "configs/robot.yaml"),
    "nav_config": os.path.join(ROOT, "configs/navigator.yaml"),
    "topomap": "",
    "autostart": "true",
    "policy_url": "http://127.0.0.1:8765",
    "log_dir": "",
    "web_port": "",          # 空なら navigator.yaml の io.web_port (0 で無効)
    "cmd_vel": "",           # 空なら navigator.yaml の io.cmd_vel
    "overrides": "",         # navigator.yaml の上書き. 例 "engine.tracker.reach_check=pose io.control_rate=20"
    "use_sim_time": "false",
}


def _node(context):
    v = {k: context.launch_configurations[k] for k in ARGS}
    params = {k: v[k] for k in ("repo_root", "robot_config", "nav_config", "topomap", "policy_url", "log_dir")
              if v[k]}
    params["autostart"] = v["autostart"].lower() in ("1", "true", "yes")
    params["use_sim_time"] = v["use_sim_time"].lower() in ("1", "true", "yes")
    if v["web_port"]:
        params["web_port"] = int(v["web_port"])
    overrides = v["overrides"].split()
    if v["cmd_vel"]:
        overrides.append("io.cmd_vel=" + v["cmd_vel"])
    if overrides:
        params["overrides"] = " ".join(overrides)
    return [Node(package="omnivla_real_ros2", executable="navigator", name="omnivla_navigator",
                 output="screen", emulate_tty=True, parameters=[params])]


def generate_launch_description():
    return LaunchDescription([DeclareLaunchArgument(k, default_value=d) for k, d in ARGS.items()]
                             + [OpaqueFunction(function=_node)])
