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
    normalized_output = " ".join(output.split())

    # Check for key sections and argument descriptions
    assert "usage" in output.lower()
    assert "folder" in output.lower()
    assert "output-html" in output.lower()
    assert "output-gallery-markdown" in output.lower()
    assert "temperature" in output.lower()
    assert "max-tokens" in output.lower()
    assert "models" in output.lower()
    assert "--resize-shape" in output
    assert "--eos-tokens" in output
    assert "--skip-special-tokens" in output
    assert "--processor-kwargs" in output
    assert "--enable-thinking" in output
    assert "--thinking-budget" in output
    assert "--thinking-start-token" in output
    assert "--thinking-end-token" in output
    assert "--detailed-metrics" in output
    assert "--min-p" in output
    assert "--top-k" in output
    assert "--kv-quant-scheme" in output
    assert "ignored unless" in output
    assert "--verbose" in output
    assert "most recently modified image file in the folder" in normalized_output
    assert str(check_models.DEFAULT_FOLDER) in output
    assert "requires a path when provided" in normalized_output.lower()
    assert "verification prompt is used" in normalized_output
    assert "Prompt. (default: None)" not in output
    assert "--eval-mode" in output
    assert "--prune-repro-days" in output
    assert "exclusions accumulate" in normalized_output
    assert "model lists accumulate" in normalized_output
    assert "token lists accumulate" in normalized_output
    assert "single flag occurrence" in normalized_output
