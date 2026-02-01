"""CLI help/usage output tests.

Test that CLI help and usage messages are clear, complete, and up-to-date.
"""

import sys
from unittest.mock import patch

import pytest

# Import check_models
import check_models


@pytest.mark.parametrize("help_flag", ["-h", "--help"])
def test_cli_help_output(help_flag: str, capsys: pytest.CaptureFixture[str]) -> None:
    test_args = ["check_models.py", help_flag]

    with patch.object(sys, "argv", test_args), pytest.raises(SystemExit) as excinfo:
        check_models.main_cli()

    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    output = captured.out + captured.err

    # Check for key sections and argument descriptions
    assert "usage" in output.lower()
    assert "folder" in output.lower()
    assert "output-html" in output.lower()
    assert "temperature" in output.lower()
    assert "max-tokens" in output.lower()
    assert "models" in output.lower()
