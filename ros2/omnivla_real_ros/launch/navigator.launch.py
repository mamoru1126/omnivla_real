"""OmniVLA 実機ナビゲーション.

  ros2 launch omnivla_real_ros navigator.launch.py topomap:=/data/topomaps/course_a
  # モデルを切り替える: model:=7b weights:=/checkpoints/omnivla-original finetuned_dir:=/runs/<run>/checkpoints/step_005000
  # bag 再生で試す: use_sim_time:=true (ros2 bag play --clock)
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

ROOT = os.environ.get("OMNIVLA_REAL_ROOT", "/workspace")


def generate_launch_description():
    args = {
        "robot_config": (os.path.join(ROOT, "configs/robot.yaml"), str),
        "nav_config": (os.path.join(ROOT, "configs/navigator.yaml"), str),
        "topomap": ("", str),
        "autostart": ("true", bool),
        "model": ("", str),
        "weights": ("", str),
        "finetuned_dir": ("", str),
        "device": ("", str),
        "log_dir": ("", str),
        "use_sim_time": ("false", bool),
    }
    decls = [DeclareLaunchArgument(k, default_value=v[0]) for k, v in args.items()]
    params = {k: ParameterValue(LaunchConfiguration(k), value_type=v[1]) for k, v in args.items()}
    return LaunchDescription(decls + [
        Node(package="omnivla_real_ros", executable="navigator", name="omnivla_navigator",
             parameters=[params], output="screen", emulate_tty=True),
    ])
