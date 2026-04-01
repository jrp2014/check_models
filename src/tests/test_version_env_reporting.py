"""Version and environment reporting tests.

Test that version and environment info is correctly reported in logs and outputs
for reproducibility.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Import check_models
import check_models

_SRC_DIR = Path(__file__).parent.parent
_OUTPUT_DIR = _SRC_DIR / "output"


def test_cli_version_and_env_reporting(tmp_path: Path) -> None:
    env_log = tmp_path / "test_env_reporting.log"
    cli_log = tmp_path / "test_env_reporting_cli.log"

    test_args = [
        "check_models.py",
        "--folder",
        str(tmp_path),
        "--output-env",
        str(env_log),
        "--output-log",
        str(cli_log),
    ]

    with patch.object(sys, "argv", test_args), pytest.raises(SystemExit):
        check_models.main_cli()

    # Check that the log file was created and contains version/environment info
    assert env_log.exists()
    content = env_log.read_text(encoding="utf-8").lower()
    assert "python" in content
    assert "mlx" in content or "mlx-vlm" in content
    # Environment dump now uses importlib.metadata (not pip/conda subprocess calls)
    assert "packages" in content or "environment" in content
