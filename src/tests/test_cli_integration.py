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
        [sys.executable, "check_models.py", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "MLX VLM Model Checker" in result.stdout
    assert "--folder" in result.stdout
    assert "--models" in result.stdout


def test_cli_version_info():
    """Should display version info with --version."""
    result = subprocess.run(
        [sys.executable, "check_models.py", "--version"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    # Should show some version information
    assert len(result.stdout) > 0


def test_cli_exits_on_nonexistent_folder():
    """Should exit with error for nonexistent folder."""
    result = subprocess.run(
        [sys.executable, "check_models.py", "--folder", "/nonexistent/test/path"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    # Should mention the folder issue in output
    output = result.stdout + result.stderr
    assert "not found" in output.lower() or "does not exist" in output.lower()


def test_cli_exits_on_empty_folder(tmp_path: Path):
    """Should exit with error when folder has no images."""
    empty_folder = tmp_path / "empty"
    empty_folder.mkdir()

    result = subprocess.run(
        [sys.executable, "check_models.py", "--folder", str(empty_folder)],
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
            "check_models.py",
            "--temperature",
            "2.5",
            "--folder",
            ".",
        ],
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
            "check_models.py",
            "--max-tokens",
            "-10",
            "--folder",
            ".",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0


def test_cli_basic_run_structure(test_folder_with_images: Path):
    """Should execute basic workflow and show expected output structure."""
    result = subprocess.run(
        [
            sys.executable,
            "check_models.py",
            "--folder",
            str(test_folder_with_images),
            "--max-tokens",
            "5",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    # Should succeed or fail gracefully
    assert result.returncode in (0, 1)

    output = result.stdout + result.stderr

    # Should show header
    assert "MLX Vision Language Model" in output or "MLX VLM" in output

    # Should attempt to process image
    assert "newest.jpg" in output or "Scanning folder" in output


def test_cli_verbose_flag_increases_output(test_folder_with_images: Path):
    """Should produce more output with --verbose flag."""
    # Run without verbose
    result_normal = subprocess.run(
        [
            sys.executable,
            "check_models.py",
            "--folder",
            str(test_folder_with_images),
            "--max-tokens",
            "5",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    # Run with verbose
    result_verbose = subprocess.run(
        [
            sys.executable,
            "check_models.py",
            "--folder",
            str(test_folder_with_images),
            "--max-tokens",
            "5",
            "--verbose",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    # Verbose should produce equal or more output
    verbose_output = result_verbose.stdout + result_verbose.stderr
    normal_output = result_normal.stdout + result_normal.stderr

    # At minimum, verbose should mention logging/debug info
    assert len(verbose_output) >= len(normal_output) or "verbose" in verbose_output.lower()


def test_cli_custom_prompt_parameter(test_folder_with_images: Path):
    """Should accept custom prompt via --prompt."""
    custom_prompt = "Describe this test image in detail."

    result = subprocess.run(
        [
            sys.executable,
            "check_models.py",
            "--folder",
            str(test_folder_with_images),
            "--prompt",
            custom_prompt,
            "--max-tokens",
            "5",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    # Should execute without argument parsing errors
    assert result.returncode in (0, 1)  # May fail if no models, but args should parse
    output = result.stdout + result.stderr

    # Should show the prompt somewhere
    assert custom_prompt in output or "prompt" in output.lower()


def test_cli_model_exclusion_parameter(test_folder_with_images: Path):
    """Should accept --exclude parameter for filtering models."""
    result = subprocess.run(
        [
            sys.executable,
            "check_models.py",
            "--folder",
            str(test_folder_with_images),
            "--exclude",
            "some-model",
            "--max-tokens",
            "5",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    # Should parse arguments successfully
    assert result.returncode in (0, 1)


def test_cli_timeout_parameter(test_folder_with_images: Path):
    """Should accept --timeout parameter."""
    result = subprocess.run(
        [
            sys.executable,
            "check_models.py",
            "--folder",
            str(test_folder_with_images),
            "--timeout",
            "5",
            "--max-tokens",
            "5",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    # Should parse successfully
    assert result.returncode in (0, 1)
