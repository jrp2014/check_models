"""Error message consistency tests for CLI and logs.

Verify that all error paths produce clear, actionable, and consistently formatted messages.
"""

import subprocess
import sys
from pathlib import Path

import pytest

_SRC_DIR = Path(__file__).parent.parent
_CHECK_MODELS_SCRIPT = _SRC_DIR / "check_models.py"
_OUTPUT_DIR = _SRC_DIR / "output"


@pytest.mark.parametrize(
    ("args", "expected_phrases"),
    [
        (["--folder", "/nonexistent/folder/path"], ["does not exist", "Exiting."]),
        (["--max-tokens", "0"], ["max_tokens must be > 0", "Fatal error"]),
        (["--temperature", "-1"], ["Temperature must be non-negative", "Fatal error"]),
        (["--kv-bits", "99"], ["kv_bits must be", "Fatal error"]),
    ],
)
def test_error_message_consistency(args: list[str], expected_phrases: list[str]) -> None:
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(_CHECK_MODELS_SCRIPT),
            *args,
            "--output-log",
            str(_OUTPUT_DIR / "test_error_consistency.log"),
            "--output-env",
            str(_OUTPUT_DIR / "test_error_consistency_env.log"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0
    for phrase in expected_phrases:
        if phrase in {"kv_bits must be", "Fatal error"}:
            # Accept argparse's invalid choice error for kv-bits
            assert phrase.lower() in output.lower() or "invalid choice" in output.lower()
        else:
            assert phrase.lower() in output.lower()
