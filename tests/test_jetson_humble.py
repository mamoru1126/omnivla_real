"""JetsonのROSディストリビューション選択を検証します。"""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


class JetsonDistroTest(unittest.TestCase):
    """既存のNoetic起動とHumbleの切り替えを確認します。"""

    def test_humble_defaults_to_robot_links_profile(self) -> None:
        """Humbleの既定値が専用LAN・USBと専用速度トピックを選ぶことを確認します。"""
        settings = dict(
            line.split("=", 1)
            for line in (ROOT / ".env.example").read_text().splitlines()
            if line and not line.startswith("#") and "=" in line
        )
        profile = "../nakanoshima_2026/docker/fastdds_robot_links.xml"
        self.assertEqual(settings["DDS_PROFILE_FILE"], profile)
        self.assertEqual(settings["ROS_DOMAIN_ID"], "42")
        self.assertEqual(settings["CMD_VEL_TOPIC"], "/omnivla/cmd_vel/raw")
        self.assertEqual(settings["ROBOT_CONFIG"], "/workspace/configs/robot.yaml")
        self.assertEqual(settings["NAV_CONFIG"], "/workspace/configs/navigator.yaml")
        compose = (ROOT / "docker-compose.jetson-humble.yml").read_text()
        self.assertIn(f"${{DDS_PROFILE_FILE:-{profile}}}:/etc/omnivla/fastdds.xml:ro", compose)
        self.assertIn("ROBOT_CONFIG: ${ROBOT_CONFIG:-/workspace/configs/robot.yaml}", compose)
        self.assertIn("NAV_CONFIG: ${NAV_CONFIG:-/workspace/configs/navigator.yaml}", compose)
        self.assertIn("-r /cmd_vel:=$$CMD_VEL_TOPIC", compose)

    def test_standard_navigation_restores_master_goal_settings(self) -> None:
        nav = yaml.safe_load((ROOT / "configs/navigator.yaml").read_text())
        robot = yaml.safe_load((ROOT / "configs/robot.yaml").read_text())
        self.assertEqual(nav["engine"]["tracker"]["reach_check"], "auto")
        self.assertTrue(nav["engine"]["stop_at_goal"])
        self.assertEqual(nav["engine"]["controller"]["track_max_v"], 0.6)
        self.assertEqual(nav["engine"]["controller"]["track_max_w"], 1.0)
        self.assertEqual(robot["topics"]["odom"], "/odom")
        settings = dict(
            line.split("=", 1)
            for line in (ROOT / ".env.example").read_text().splitlines()
            if line and not line.startswith("#") and "=" in line
        )
        self.assertEqual(settings["TOPOMAP"], "")

    def test_humble_topomap_argument_handles_empty_and_spaced_paths(self) -> None:
        compose = yaml.safe_load((ROOT / "docker-compose.jetson-humble.yml").read_text())
        command = compose["services"]["nav"]["command"][-1].replace("$$", "$")
        with tempfile.TemporaryDirectory() as directory:
            ros2 = Path(directory) / "ros2"
            ros2.write_text(
                "#!/usr/bin/env python3\n"
                "import json, sys\n"
                "print(json.dumps(sys.argv[1:]))\n"
            )
            ros2.chmod(0o755)
            for topomap in ("", "/data/topomaps/course_a", "/data/topomaps/course a"):
                with self.subTest(topomap=topomap):
                    env = dict(
                        os.environ, PATH=directory + os.pathsep + os.environ["PATH"],
                        TOPOMAP=topomap, NAV_AUTOSTART="false", POLICY_PORT="8765",
                        CMD_VEL_TOPIC="/omnivla/cmd_vel/raw",
                        ROBOT_CONFIG="/workspace/configs/robot.yaml",
                        NAV_CONFIG="/workspace/configs/navigator.yaml",
                    )
                    result = subprocess.run(
                        ["bash", "-c", "source() { :; }\n" + command],
                        env=env, text=True, capture_output=True, check=False,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    arguments = json.loads(result.stdout)
                    values = [value for value in arguments if value.startswith("topomap:=")]
                    self.assertEqual(values, ["topomap:=" + topomap] if topomap else [])
                    self.assertIn("autostart:=false", arguments)
                    self.assertIn("/cmd_vel:=/omnivla/cmd_vel/raw", arguments)

    def run_jetson(self, distro: str, *args: str) -> subprocess.CompletedProcess:
        env = dict(os.environ, OMNIVLA_DRY_RUN="1", JETSON_ROS_DISTRO=distro)
        return subprocess.run(
            ["bash", str(ROOT / "bin/jetson.sh"), *args],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_noetic_preserves_existing_compose(self) -> None:
        result = self.run_jetson("noetic", "build")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("-f docker-compose.jetson.yml build", result.stdout)
        self.assertNotIn("docker-compose.jetson-humble.yml", result.stdout)

    def test_humble_adds_compose_override(self) -> None:
        result = self.run_jetson("humble", "build")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("-f docker-compose.jetson-humble.yml build", result.stdout)

    def test_unknown_distro_fails(self) -> None:
        result = self.run_jetson("invalid", "build")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("JETSON_ROS_DISTRO", result.stderr)

    def test_humble_rejects_roscore(self) -> None:
        result = self.run_jetson("humble", "up", "--standalone")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--standalone", result.stderr)


if __name__ == "__main__":
    unittest.main()
