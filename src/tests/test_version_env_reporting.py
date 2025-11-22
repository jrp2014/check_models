"""Version and environment reporting tests.

Test that version and environment info is correctly reported in logs and outputs
for reproducibility.
"""

import subprocess
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).parent.parent
_CHECK_MODELS_SCRIPT = _SRC_DIR / "check_models.py"
_OUTPUT_DIR = _SRC_DIR / "output"

# This test assumes the CLI writes environment info to the log file specified by --output-env


def test_cli_version_and_env_reporting(tmp_path: Path) -> None:
    env_log = tmp_path / "test_env_reporting.log"
    subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(_CHECK_MODELS_SCRIPT),
            "--folder",
            str(tmp_path),
            "--output-env",
            str(env_log),
            "--output-log",
            str(tmp_path / "test_env_reporting_cli.log"),
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    # Check that the log file was created and contains version/environment info
    assert env_log.exists()
    content = env_log.read_text().lower()
    assert "python" in content
    assert "mlx" in content or "mlx-vlm" in content
    assert "mypy" in content or "ruff" in content
    assert "pip" in content or "conda" in content
