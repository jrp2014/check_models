"""Invalid argument tests for CLI robustness.

Test CLI with invalid argument combinations, missing required arguments, and boundary values.
Confirm proper error handling and messaging.
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
    ("args", "expected_error"),
    [
        (["--max-tokens", "-1"], "max_tokens must be > 0"),
        (["--temperature", "-5"], "Temperature must be non-negative"),
        (["--kv-bits", "16"], "kv_bits must be"),
        (["--folder"], "Folder"),  # Missing folder path
        (["--unknown-flag"], "unrecognized arguments"),
    ],
)
def test_cli_invalid_arguments(
    args: list[str],
    expected_error: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    test_args = [
        "check_models.py",
        *args,
        "--output-log",
        str(_OUTPUT_DIR / "test_invalid_args.log"),
        "--output-env",
        str(_OUTPUT_DIR / "test_invalid_args_env.log"),
    ]

    with patch.object(sys, "argv", test_args), pytest.raises(SystemExit) as excinfo:
        check_models.main_cli()

    assert excinfo.value.code != 0
    captured = capsys.readouterr()
    output = captured.out + captured.err

    if expected_error == "kv_bits must be":
        # Accept argparse's invalid choice error for kv-bits
        assert expected_error.lower() in output.lower() or "invalid choice" in output.lower()
    else:
        assert expected_error.lower() in output.lower()
