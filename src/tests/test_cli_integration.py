"""Integration tests for the CLI.

These tests verify the CLI's behavior with various arguments and inputs.
"""

import sys
from pathlib import Path
from typing import NamedTuple
from unittest.mock import patch

import pytest

# Import check_models
import check_models


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


def _get_test_output_args(tmp_path: Path) -> list[str]:
    """Redirect the whole retained layout outside the tree (never into src/output)."""
    return ["--output-dir", str(tmp_path / "output")]


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


def test_cli_parser_accumulates_repeated_exclude_flags() -> None:
    """Repeated -e/--exclude flags should accumulate all excluded models."""
    parser = check_models._build_cli_parser()

    args = parser.parse_args(
        [
            "--folder",
            "test-folder",
            "-e",
            "Qwen/Qwen3-VL-2B-Instruct",
            "-e",
            "mlx-community/Qwen3-VL-2B-Thinking-bf16",
        ]
    )

    assert args.exclude == [
        "Qwen/Qwen3-VL-2B-Instruct",
        "mlx-community/Qwen3-VL-2B-Thinking-bf16",
    ]


def test_cli_parser_accumulates_repeated_model_flags() -> None:
    """Repeated -m/--models flags should accumulate all model identifiers."""
    parser = check_models._build_cli_parser()

    args = parser.parse_args(
        [
            "--folder",
            "test-folder",
            "-m",
            "model-a",
            "-m",
            "model-b",
            "model-c",
        ]
    )

    assert args.models == ["model-a", "model-b", "model-c"]


def test_cli_parser_accumulates_repeated_eos_token_flags() -> None:
    """Repeated --eos-tokens flags should accumulate all stop tokens."""
    parser = check_models._build_cli_parser()

    args = parser.parse_args(
        [
            "--folder",
            "test-folder",
            "--eos-tokens",
            "</think>",
            "--eos-tokens",
            r"\n",
            "<END>",
        ]
    )

    assert args.eos_tokens == ["</think>", r"\n", "<END>"]


def test_cli_output_dir_default_is_the_canonical_output_root() -> None:
    """The single output root defaults to src/output."""
    args = check_models._build_cli_parser().parse_args(["--folder", "test-folder"])

    assert args.output_dir == check_models._SCRIPT_DIR / "output"


def test_cli_exits_on_nonexistent_folder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Should exit with error when folder does not exist."""
    result = _run_cli([*_get_test_output_args(tmp_path), "--folder", "/nonexistent/path"], capsys)
    assert result.exit_code != 0
    output = result.stdout + result.stderr
    # Should mention the folder doesn't exist (exact message from exit_with_cli_error)
    assert "does not exist" in output.lower() or "not found" in output.lower()


def test_cli_exits_on_empty_folder(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Should exit with error when folder has no images."""
    empty_folder = tmp_path / "empty"
    empty_folder.mkdir()

    result = _run_cli([*_get_test_output_args(tmp_path), "--folder", str(empty_folder)], capsys)
    assert result.exit_code != 0
    output = result.stdout + result.stderr
    # Should mention no images found in the provided folder.
    assert "could not find" in output.lower()


def test_cli_invalid_temperature_value(
    folder_with_images: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Should reject temperature outside valid range."""
    result = _run_cli(
        [
            *_get_test_output_args(tmp_path),
            "--temperature",
            "-0.5",
            "--folder",
            str(folder_with_images),
        ],
        capsys,
    )
    assert result.exit_code != 0
    output = result.stdout + result.stderr
    assert "temperature" in output.lower()


def test_cli_invalid_max_tokens(
    folder_with_images: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Should reject negative max_tokens."""
    result = _run_cli(
        [
            *_get_test_output_args(tmp_path),
            "--max-tokens",
            "-10",
            "--folder",
            str(folder_with_images),
        ],
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
    assert "--output-dir" in output
    assert "--output-jsonl" not in output
    assert "--output-gallery-markdown" not in output


def test_cli_rejects_url_passed_to_image(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--image with a URL must fail with guidance, not a mangled-path ENOENT.

    argparse wraps the value in a Path, so a URL previously surfaced only as
    "No such file: .../src/https:/github.com/..." after resolution.
    """
    result = _run_cli(
        [
            *_get_test_output_args(tmp_path),
            "--image",
            "https://github.com/Blaizzy/mlx-vlm/blob/main/examples/images/cats.jpg",
        ],
        capsys,
    )

    assert result.exit_code != 0
    output = result.stdout + result.stderr
    assert "expects a local file path, not a URL" in output
    assert "--image-source-url" in output
    assert "/raw/" in output
    assert "No such file" not in output


@pytest.mark.parametrize(
    ("value", "url_like"),
    [
        ("https://github.com/x/y.jpg", True),
        ("http://host/i.jpg", True),
        ("file:///tmp/a.jpg", True),
        ("cats.jpg", False),
        ("/Users/someone/Pictures/a.jpg", False),
        ("C:/photos/a.jpg", False),  # single-letter drive spec is a path, not a scheme
        ("./sub/dir/a.jpg", False),
    ],
)
def test_url_like_image_argument_detection(value: str, url_like: bool) -> None:
    """URL schemes are detected after Path() collapses '://' to ':/'; paths are not."""
    assert bool(check_models._URL_LIKE_IMAGE_ARG_RE.match(str(Path(value)))) is url_like
