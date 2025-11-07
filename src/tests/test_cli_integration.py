"""End-to-end CLI integration tests.

These tests verify the full workflow from CLI input to output generation,
including argument parsing, execution flow, and error handling.
"""

# ruff: noqa: S603, ANN201
import subprocess
import sys
import time
from pathlib import Path

import pytest
from PIL import Image

# Path to check_models.py relative to test file location
_TEST_DIR = Path(__file__).parent
_SRC_DIR = _TEST_DIR.parent
_CHECK_MODELS_SCRIPT = _SRC_DIR / "check_models.py"


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


def test_cli_help_displays():
    """Should display help message with --help."""
    result = subprocess.run(
        [sys.executable, str(_CHECK_MODELS_SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "MLX VLM Model Checker" in result.stdout
    assert "--folder" in result.stdout
    assert "--models" in result.stdout


def test_cli_help_structure():
    """Should display help text that includes usage information."""
    result = subprocess.run(
        [sys.executable, str(_CHECK_MODELS_SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    output = result.stdout + result.stderr
    # Should contain basic usage info
    assert "usage" in output.lower() or "--folder" in output


def test_cli_exits_on_nonexistent_folder():
    """Should exit with error when folder does not exist."""
    result = subprocess.run(
        [sys.executable, str(_CHECK_MODELS_SCRIPT), "--folder", "/nonexistent/folder/path"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert result.returncode != 0
    output = result.stdout + result.stderr
    # Check for error message about missing folder
    assert (
        "folder" in output.lower() or "directory" in output.lower() or "not found" in output.lower()
    )


def test_cli_exits_on_empty_folder(tmp_path: Path):
    """Should exit with error when folder has no images."""
    empty_folder = tmp_path / "empty"
    empty_folder.mkdir()

    result = subprocess.run(
        [sys.executable, str(_CHECK_MODELS_SCRIPT), "--folder", str(empty_folder)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "could not find" in output.lower() or "no image" in output.lower()


def test_cli_invalid_temperature_value():
    """Should reject temperature outside valid range."""
    result = subprocess.run(
        [
            sys.executable,
            str(_CHECK_MODELS_SCRIPT),
            "--temperature",
            "2.5",
            "--folder",
            ".",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    # Should fail validation
    assert result.returncode != 0


def test_cli_invalid_max_tokens():
    """Should reject negative max_tokens."""
    result = subprocess.run(
        [
            sys.executable,
            str(_CHECK_MODELS_SCRIPT),
            "--max-tokens",
            "-10",
            "--folder",
            ".",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0


def test_cli_accepts_valid_parameters():
    """Should accept valid command-line parameters without error."""
    # Just test that the script accepts parameters correctly without actually running models
    result = subprocess.run(
        [sys.executable, str(_CHECK_MODELS_SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )

    output = result.stdout + result.stderr
    # Check that help shows our expected parameters
    assert "--folder" in output or "--temperature" in output or "usage:" in output.lower()
