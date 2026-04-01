"""Error message consistency tests for CLI and logs.

Verify that all error paths produce clear, actionable, and consistently formatted messages.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Import check_models
import check_models

_SRC_DIR = Path(__file__).parent.parent
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
def test_error_message_consistency(
    args: list[str], expected_phrases: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    test_args = [
        "check_models.py",
        *args,
        "--output-log",
        str(_OUTPUT_DIR / "test_error_consistency.log"),
        "--output-env",
        str(_OUTPUT_DIR / "test_error_consistency_env.log"),
    ]

    with patch.object(sys, "argv", test_args), pytest.raises(SystemExit) as excinfo:
        check_models.main_cli()

    assert excinfo.value.code != 0
    captured = capsys.readouterr()
    output = captured.out + captured.err

    for phrase in expected_phrases:
        if phrase in {"kv_bits must be", "Fatal error"}:
            # Accept argparse's invalid choice error for kv-bits
            assert phrase.lower() in output.lower() or "invalid choice" in output.lower()
        else:
            assert phrase.lower() in output.lower()
