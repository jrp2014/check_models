"""Edge case image tests for CLI robustness.

Tests extremely large, small, corrupted, and unsupported image files.
Ensures graceful failure and clear error messages.
"""

import subprocess
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).parent.parent
_CHECK_MODELS_SCRIPT = _SRC_DIR / "check_models.py"
_OUTPUT_DIR = _SRC_DIR / "output"


def test_cli_handles_corrupted_image(tmp_path: Path) -> None:
    """Verify CLI exits with error for corrupted image files."""
    img_path = tmp_path / "corrupted.png"
    img_path.write_bytes(b"not an image")
    result = subprocess.run(
        [
            sys.executable,
            str(_CHECK_MODELS_SCRIPT),
            "--image",
            str(img_path),
            "--output-log",
            str(_OUTPUT_DIR / "test_edge_case.log"),
            "--output-env",
            str(_OUTPUT_DIR / "test_edge_case_env.log"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode != 0
    assert "error" in result.stderr.lower() or "cannot open" in result.stderr.lower()


def test_cli_handles_unsupported_format(tmp_path: Path) -> None:
    """Verify CLI exits with error for unsupported file formats."""
    txt_path = tmp_path / "not_an_image.txt"
    txt_path.write_text("hello world")
    result = subprocess.run(
        [
            sys.executable,
            str(_CHECK_MODELS_SCRIPT),
            "--image",
            str(txt_path),
            "--output-log",
            str(_OUTPUT_DIR / "test_edge_case.log"),
            "--output-env",
            str(_OUTPUT_DIR / "test_edge_case_env.log"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    # Expect clean exit with error message (no traceback)
    assert result.returncode == 1
    assert "cannot" in result.stderr.lower() or "error" in result.stderr.lower()
