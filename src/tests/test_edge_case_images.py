"""Tests for CLI robustness with edge-case images."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# src/tests is a package, and src is the parent. We can import check_models.
# We add it to sys.path if needed, but normally pytest handles this if run from src.
import check_models

_SRC_DIR = Path(__file__).parent.parent
_OUTPUT_DIR = _SRC_DIR / "output"


def test_cli_handles_corrupted_image(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI exits with error for corrupted image files."""
    img_path = tmp_path / "corrupted.png"
    img_path.write_bytes(b"not an image")

    test_args = [
        "check_models.py",
        "--image",
        str(img_path),
        "--output-log",
        str(_OUTPUT_DIR / "test_edge_case.log"),
        "--output-env",
        str(_OUTPUT_DIR / "test_edge_case_env.log"),
    ]

    with patch.object(sys, "argv", test_args), pytest.raises(SystemExit) as excinfo:
        check_models.main_cli()

    # In check_models.py, SystemExit(1) is raised on handled errors
    assert excinfo.value.code != 0
    captured = capsys.readouterr()
    # Errors are often logged, so they might be in stderr or stdout depending on logger config
    # In our script, exceptions are logged to the console (stderr usually)
    out_err = (captured.out + captured.err).lower()
    assert "error" in out_err or "cannot open" in out_err


def test_cli_handles_unsupported_format(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI exits with error for unsupported file formats."""
    txt_path = tmp_path / "not_an_image.txt"
    txt_path.write_text("hello world")

    test_args = [
        "check_models.py",
        "--image",
        str(txt_path),
        "--output-log",
        str(_OUTPUT_DIR / "test_edge_case.log"),
        "--output-env",
        str(_OUTPUT_DIR / "test_edge_case_env.log"),
    ]

    with patch.object(sys, "argv", test_args), pytest.raises(SystemExit) as excinfo:
        check_models.main_cli()

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    out_err = (captured.out + captured.err).lower()
    assert "cannot" in out_err or "error" in out_err
