"""JetsonのROSディストリビューション選択を検証します。"""

import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


class JetsonDistroTest(unittest.TestCase):
    """既存のNoetic起動とHumbleの切り替えを確認します。"""

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
