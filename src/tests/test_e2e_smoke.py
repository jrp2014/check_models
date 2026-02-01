"""End-to-end smoke tests for the VLM check pipeline.

These tests verify the complete workflow with actual model inference.
They are marked with `pytest.mark.slow` and `pytest.mark.e2e` for selective
execution, as they require downloading and running models.

Run these tests explicitly with:
    pytest tests/test_e2e_smoke.py -v

Or run all tests including slow ones:
    pytest --run-slow

Skip these in CI by default (they require MLX hardware and model downloads).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple
from unittest.mock import patch

# =============================================================================
# EARLY ENVIRONMENT SETUP (MUST happen before huggingface_hub imports)
# =============================================================================

# Set up HF cache directory early, before any huggingface_hub functions cache the path.
# This is needed for CI environments that don't have ~/.cache/huggingface/hub
_HF_CACHE_DIR = Path(tempfile.gettempdir()) / "pytest_hf_cache" / "hub"
_HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HUB_CACHE"] = str(_HF_CACHE_DIR)
os.environ["HF_HOME"] = str(_HF_CACHE_DIR.parent)

# Now import huggingface_hub after environment is configured
import pytest  # noqa: E402
from huggingface_hub import scan_cache_dir  # noqa: E402
from huggingface_hub.errors import CacheNotFound  # noqa: E402
from PIL import Image  # noqa: E402

import check_models  # noqa: E402

# Fixture model - small, fast, reliable
# nanoLLaVA is ~600MB, fastest load, lowest memory (4.5GB peak)
FIXTURE_MODEL = "qnguyen3/nanoLLaVA"


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


def _get_e2e_output_args(output_dir: Path) -> list[str]:
    """Return CLI arguments for E2E test-specific output files."""
    return [
        "--output-log",
        str(output_dir / "e2e.log"),
        "--output-html",
        str(output_dir / "e2e.html"),
        "--output-markdown",
        str(output_dir / "e2e.md"),
        "--output-tsv",
        str(output_dir / "e2e.tsv"),
        "--output-jsonl",
        str(output_dir / "e2e.jsonl"),
        "--output-env",
        str(output_dir / "e2e_env.log"),
    ]


@pytest.fixture
def e2e_output_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for E2E test outputs."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def e2e_test_image(tmp_path: Path) -> Path:
    """Create a realistic test image for E2E testing."""
    img_path = tmp_path / "e2e_test.jpg"
    img = Image.new("RGB", (640, 480), color=(135, 206, 235))
    pixels = img.load()
    if pixels:
        for x in range(540, 600):
            for y in range(40, 100):
                if (x - 570) ** 2 + (y - 70) ** 2 < 900:
                    pixels[x, y] = (255, 255, 0)
        for x in range(640):
            for y in range(380, 480):
                pixels[x, y] = (34, 139, 34)
        for x in range(200, 350):
            for y in range(250, 380):
                pixels[x, y] = (139, 90, 43)
        for x in range(180, 370):
            for y in range(200, 250):
                pixels[x, y] = (178, 34, 34)
    img.save(img_path, "JPEG", quality=85)
    return img_path


def _check_mlx_vlm_available() -> bool:
    """Check if mlx-vlm is available in the current environment."""
    return importlib.util.find_spec("mlx_vlm") is not None


def _check_model_cached(model_id: str) -> bool:
    """Check if a model is already cached locally."""
    try:
        repos = scan_cache_dir().repos
    except (OSError, ValueError, RuntimeError, CacheNotFound):
        return False
    else:
        repo_ids = [r.repo_id for r in repos]
        return model_id in repo_ids


# Mark all tests in this module as slow and e2e
pytestmark = [
    pytest.mark.slow,
    pytest.mark.e2e,
]


@pytest.mark.skipif(
    not _check_mlx_vlm_available(),
    reason="mlx-vlm not available in environment",
)
class TestE2ESmoke:
    """End-to-end smoke tests that run actual model inference."""

    def test_dry_run_with_fixture_model(
        self,
        e2e_test_image: Path,
        e2e_output_dir: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Dry-run should validate setup without invoking the model."""
        args = [
            *_get_e2e_output_args(e2e_output_dir),
            "--image",
            str(e2e_test_image),
            "--models",
            FIXTURE_MODEL,
            "--dry-run",
            "--prompt",
            "Describe this image.",
        ]
        result = _run_cli(args, capsys)
        assert result.exit_code == 0
        output = result.stdout + result.stderr
        assert "dry run" in output.lower()
        assert FIXTURE_MODEL in output
        assert "Describe this image" in output

    @pytest.mark.skipif(
        not _check_model_cached(FIXTURE_MODEL),
        reason=f"Model {FIXTURE_MODEL} not cached (run manually first)",
    )
    def test_full_inference_with_fixture_model(
        self,
        e2e_test_image: Path,
        e2e_output_dir: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Full inference run should complete successfully and produce outputs."""
        args = [
            *_get_e2e_output_args(e2e_output_dir),
            "--image",
            str(e2e_test_image),
            "--models",
            FIXTURE_MODEL,
            "--prompt",
            "Describe the main elements in this image briefly.",
            "--max-tokens",
            "100",
            "--timeout",
            "120",
        ]
        result = _run_cli(args, capsys)
        assert result.exit_code == 0

        # Verify output files were created
        e2e_html = e2e_output_dir / "e2e.html"
        e2e_md = e2e_output_dir / "e2e.md"
        e2e_jsonl = e2e_output_dir / "e2e.jsonl"
        assert e2e_html.exists()
        assert e2e_md.exists()
        assert e2e_jsonl.exists()

        with e2e_jsonl.open() as f:
            records = [json.loads(line) for line in f if line.strip()]
        assert len(records) >= 1
        record = records[0]
        assert record["model"] == FIXTURE_MODEL
        assert record["success"] is True

    @pytest.mark.skipif(
        not _check_model_cached(FIXTURE_MODEL),
        reason=f"Model {FIXTURE_MODEL} not cached (run manually first)",
    )
    def test_quality_analysis_produces_output(
        self,
        e2e_test_image: Path,
        e2e_output_dir: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Quality analysis should run and detect potential issues."""
        args = [
            *_get_e2e_output_args(e2e_output_dir),
            "--image",
            str(e2e_test_image),
            "--models",
            FIXTURE_MODEL,
            "--prompt",
            "Describe this image in detail.",
            "--max-tokens",
            "150",
            "--verbose",
        ]
        result = _run_cli(args, capsys)
        assert result.exit_code == 0
        output = result.stdout + result.stderr
        assert any(
            word in output.lower() for word in ["quality", "tps", "generated", "tokens", "memory"]
        )

    def test_invalid_model_produces_error(
        self,
        e2e_test_image: Path,
        e2e_output_dir: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Non-existent model should produce a clear error."""
        args = [
            *_get_e2e_output_args(e2e_output_dir),
            "--image",
            str(e2e_test_image),
            "--models",
            "nonexistent/fake-model-12345",
            "--timeout",
            "30",
        ]
        result = _run_cli(args, capsys)
        output = result.stdout + result.stderr
        # Model loading should fail with repository/model not found error
        assert (
            "not found" in output.lower()
            or "could not" in output.lower()
            or "failed" in output.lower()
        )


# Note: No cleanup fixture needed - tmp_path fixture handles cleanup automatically
