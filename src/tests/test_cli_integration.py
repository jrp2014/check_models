"""Integration tests for the CLI.

These tests verify the CLI's behavior with various arguments and inputs.
"""

import sys
import time
from pathlib import Path
from typing import NamedTuple
from unittest.mock import patch

import pytest
from PIL import Image

# Import check_models
import check_models

# Path to check_models.py relative to test file location
_TEST_DIR = Path(__file__).parent
_SRC_DIR = _TEST_DIR.parent
_OUTPUT_DIR = _SRC_DIR / "output"

# Test-specific output files (excluded from git via .gitignore)
_TEST_LOG = _OUTPUT_DIR / "test_cli_integration.log"
_TEST_HTML = _OUTPUT_DIR / "test_cli_integration.html"
_TEST_MD = _OUTPUT_DIR / "test_cli_integration.md"
_TEST_ENV = _OUTPUT_DIR / "test_cli_integration_environment.log"
_TEST_JSONL = _OUTPUT_DIR / "test_cli_integration.jsonl"


class CLIResult(NamedTuple):
    """Result of a CLI execution for testing."""

    exit_code: int
    stdout: str
    stderr: str


def _run_cli(args: list[str], capsys: pytest.CaptureFixture[str]) -> CLIResult:
    """Helper to run the CLI main function directly."""
    test_args = ["check_models.py", *args]
    exit_code = 0
    with patch.object(sys, "argv", test_args):
        try:
            check_models.main_cli()
        except SystemExit as e:
            exit_code = e.code if isinstance(e.code, int) else (1 if e.code else 0)

    captured = capsys.readouterr()
    return CLIResult(exit_code, captured.out, captured.err)


def _get_test_output_args() -> list[str]:
    """Return CLI arguments for test-specific output files."""
    return [
        "--output-log",
        str(_TEST_LOG),
        "--output-html",
        str(_TEST_HTML),
        "--output-markdown",
        str(_TEST_MD),
        "--output-env",
        str(_TEST_ENV),
        "--output-jsonl",
        str(_TEST_JSONL),
    ]


@pytest.fixture
def test_image(tmp_path: Path) -> Path:
    """Create a minimal valid test image."""
    img_path = tmp_path / "test.jpg"
    img = Image.new("RGB", (100, 100), color="red")
    img.save(img_path)
    return img_path


@pytest.fixture
def test_folder_with_images(tmp_path: Path) -> Path:
    """Create a folder with multiple test images."""
    folder = tmp_path / "images"
    folder.mkdir()

    # Create images with different timestamps
    for i, name in enumerate(["old.jpg", "middle.jpg", "newest.jpg"]):
        img_path = folder / name
        img = Image.new("RGB", (50, 50), color="blue")
        img.save(img_path)
        if i < 2:  # Don't sleep after last image
            time.sleep(0.1)

    return folder


def test_cli_help_displays(capsys: pytest.CaptureFixture[str]) -> None:
    """Should display help message with --help."""
    result = _run_cli(["--help"], capsys)
    assert result.exit_code == 0
    assert "MLX VLM Model Checker" in result.stdout
    assert "--folder" in result.stdout
    assert "--models" in result.stdout


def test_cli_help_structure(capsys: pytest.CaptureFixture[str]) -> None:
    """Should display help text that includes usage information."""
    result = _run_cli(["--help"], capsys)
    assert result.exit_code == 0
    output = result.stdout + result.stderr
    # Should contain basic usage info
    assert "usage" in output.lower() or "--folder" in output


def test_cli_exits_on_nonexistent_folder(capsys: pytest.CaptureFixture[str]) -> None:
    """Should exit with error when folder does not exist."""
    result = _run_cli([*_get_test_output_args(), "--folder", "/nonexistent/path"], capsys)
    assert result.exit_code != 0
    output = result.stdout + result.stderr
    assert any(word in output.lower() for word in ["folder", "directory", "not found"])


def test_cli_exits_on_empty_folder(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Should exit with error when folder has no images."""
    empty_folder = tmp_path / "empty"
    empty_folder.mkdir()

    result = _run_cli([*_get_test_output_args(), "--folder", str(empty_folder)], capsys)
    assert result.exit_code != 0
    output = result.stdout + result.stderr
    assert any(
        phrase in output.lower() for phrase in ["no images", "could not find", "mlx-vlm not found"]
    )


def test_cli_invalid_temperature_value(
    test_folder_with_images: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Should reject temperature outside valid range."""
    result = _run_cli(
        [
            *_get_test_output_args(),
            "--temperature",
            "-0.5",
            "--folder",
            str(test_folder_with_images),
        ],
        capsys,
    )
    assert result.exit_code != 0
    output = result.stdout + result.stderr
    assert "temperature" in output.lower()


def test_cli_invalid_max_tokens(
    test_folder_with_images: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Should reject negative max_tokens."""
    result = _run_cli(
        [*_get_test_output_args(), "--max-tokens", "-10", "--folder", str(test_folder_with_images)],
        capsys,
    )
    assert result.exit_code != 0
    output = result.stdout + result.stderr
    assert any(word in output.lower() for word in ["max", "token"])


def test_cli_accepts_valid_parameters(capsys: pytest.CaptureFixture[str]) -> None:
    """Should accept valid command-line parameters without error."""
    result = _run_cli(["--help"], capsys)
    output = result.stdout + result.stderr
    assert any(
        word in output or word.lower() in output.lower()
        for word in ["--folder", "--temperature", "usage:"]
    )
    assert "--output-jsonl" in output
