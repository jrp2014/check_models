"""Invalid argument tests for CLI robustness.

Test CLI with invalid argument combinations, missing required arguments, and boundary values.
Confirm proper error handling and messaging.
"""

import subprocess
import sys
from pathlib import Path

import pytest

_SRC_DIR = Path(__file__).parent.parent
_CHECK_MODELS_SCRIPT = _SRC_DIR / "check_models.py"
_OUTPUT_DIR = _SRC_DIR / "output"


@pytest.mark.parametrize(
    ("args", "expected_error"),
    [
        (["--max-tokens", "-1"], "max_tokens must be > 0"),
        (["--temperature", "-5"], "Temperature must be non-negative"),
        (["--kv-bits", "16"], "kv_bits must be"),
        (["--folder"], "Folder"),  # Missing folder path
        (["--unknown-flag"], "unrecognized arguments"),
    ],
)
def test_cli_invalid_arguments(args: list[str], expected_error: str) -> None:
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(_CHECK_MODELS_SCRIPT),
            *args,
            "--output-log",
            str(_OUTPUT_DIR / "test_invalid_args.log"),
            "--output-env",
            str(_OUTPUT_DIR / "test_invalid_args_env.log"),
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0
    if expected_error == "kv_bits must be":
        # Accept argparse's invalid choice error for kv-bits
        assert expected_error.lower() in output.lower() or "invalid choice" in output.lower()
    else:
        assert expected_error.lower() in output.lower()
