"""CLI help/usage output tests.

Test that CLI help and usage messages are clear, complete, and up-to-date.
"""

import subprocess
import sys
from pathlib import Path

import pytest  # type: ignore[import]

_SRC_DIR = Path(__file__).parent.parent
_CHECK_MODELS_SCRIPT = _SRC_DIR / "check_models.py"


@pytest.mark.parametrize("help_flag", ["-h", "--help"])
def test_cli_help_output(help_flag: str) -> None:
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(_CHECK_MODELS_SCRIPT), help_flag],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0
    # Check for key sections and argument descriptions
    assert "usage" in output.lower()
    assert "folder" in output.lower()
    assert "output-html" in output.lower()
    assert "temperature" in output.lower()
    assert "max-tokens" in output.lower()
    assert "models" in output.lower()
    assert "version" not in output.lower()  # If version flag is not present
