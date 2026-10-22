"""JetsonのROSディストリビューション選択を検証します。"""

import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


class JetsonDistroTest(unittest.TestCase):
    """既存のNoetic起動とHumbleの切り替えを確認します。"""

    def test_humble_defaults_to_direct_lan_profile(self) -> None:
        """Humbleの既定値が直接接続LANと専用速度トピックを選ぶことを確認します。"""
        settings = dict(
            line.split("=", 1)
            for line in (ROOT / ".env.example").read_text().splitlines()
            if line and not line.startswith("#") and "=" in line
        )
        profile = "../nakanoshima_2026/docker/fastdds_direct_lan.xml"
        self.assertEqual(settings["DDS_PROFILE_FILE"], profile)
        self.assertEqual(settings["ROS_DOMAIN_ID"], "42")
        self.assertEqual(settings["CMD_VEL_TOPIC"], "/omnivla/cmd_vel/raw")
        compose = (ROOT / "docker-compose.jetson-humble.yml").read_text()
        self.assertIn(f"${{DDS_PROFILE_FILE:-{profile}}}:/etc/omnivla/fastdds.xml:ro", compose)

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
