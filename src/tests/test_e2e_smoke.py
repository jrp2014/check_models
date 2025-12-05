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

# ruff: noqa: S603

import contextlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

# Path configuration
_TEST_DIR = Path(__file__).parent
_SRC_DIR = _TEST_DIR.parent
_CHECK_MODELS_SCRIPT = _SRC_DIR / "check_models.py"
_OUTPUT_DIR = _SRC_DIR / "output"

# Test-specific output files
_E2E_LOG = _OUTPUT_DIR / "test_e2e_smoke.log"
_E2E_HTML = _OUTPUT_DIR / "test_e2e_smoke.html"
_E2E_MD = _OUTPUT_DIR / "test_e2e_smoke.md"
_E2E_TSV = _OUTPUT_DIR / "test_e2e_smoke.tsv"
_E2E_JSONL = _OUTPUT_DIR / "test_e2e_smoke.jsonl"
_E2E_ENV = _OUTPUT_DIR / "test_e2e_smoke_env.log"

# Fixture model - small, fast, reliable
# nanoLLaVA is ~600MB, fastest load, lowest memory (4.5GB peak)
FIXTURE_MODEL = "qnguyen3/nanoLLaVA"

# Timeout for model operations (generous for first-time downloads)
E2E_TIMEOUT = 300  # 5 minutes


def _get_e2e_output_args() -> list[str]:
    """Return CLI arguments for E2E test-specific output files."""
    return [
        "--output-log",
        str(_E2E_LOG),
        "--output-html",
        str(_E2E_HTML),
        "--output-markdown",
        str(_E2E_MD),
        "--output-tsv",
        str(_E2E_TSV),
        "--output-jsonl",
        str(_E2E_JSONL),
        "--output-env",
        str(_E2E_ENV),
    ]


@pytest.fixture
def e2e_test_image(tmp_path: Path) -> Path:
    """Create a realistic test image for E2E testing.

    Creates a larger image with some visual content to give the model
    something to describe.
    """
    img_path = tmp_path / "e2e_test.jpg"
    # Create a simple image with visual elements
    img = Image.new("RGB", (640, 480), color=(135, 206, 235))  # Sky blue

    # Add some basic visual elements by drawing rectangles
    # (Without PIL.ImageDraw, we manually set pixel regions)
    pixels = img.load()

    # Draw a "sun" (yellow circle approximation in top-right)
    for x in range(540, 600):
        for y in range(40, 100):
            if (x - 570) ** 2 + (y - 70) ** 2 < 900:  # Circle radius ~30
                pixels[x, y] = (255, 255, 0)  # Yellow

    # Draw "grass" (green strip at bottom)
    for x in range(640):
        for y in range(380, 480):
            pixels[x, y] = (34, 139, 34)  # Forest green

    # Draw a simple "house" (brown rectangle with red roof)
    for x in range(200, 350):
        for y in range(250, 380):
            pixels[x, y] = (139, 90, 43)  # Brown (house body)

    for x in range(180, 370):
        for y in range(200, 250):
            # Triangle-ish roof (simplified as rectangle for now)
            pixels[x, y] = (178, 34, 34)  # Firebrick red (roof)

    img.save(img_path, "JPEG", quality=85)
    return img_path


def _check_mlx_vlm_available() -> bool:
    """Check if mlx-vlm is available in the current environment."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import mlx_vlm"],
            check=False,
            capture_output=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    else:
        return result.returncode == 0


def _check_model_cached(model_id: str) -> bool:
    """Check if a model is already cached locally."""
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                f"from huggingface_hub import scan_cache_dir; "
                f"ids = [r.repo_id for r in scan_cache_dir().repos]; "
                f"print('{model_id}' in ids)",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    else:
        return "True" in result.stdout


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

    def test_dry_run_with_fixture_model(self, e2e_test_image: Path) -> None:
        """Dry-run should validate setup without invoking the model."""
        result = subprocess.run(
            [
                sys.executable,
                str(_CHECK_MODELS_SCRIPT),
                *_get_e2e_output_args(),
                "--image",
                str(e2e_test_image),
                "--models",
                FIXTURE_MODEL,
                "--dry-run",
                "--prompt",
                "Describe this image.",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0, f"Dry run failed: {result.stderr}"
        output = result.stdout + result.stderr

        # Verify dry-run output contains expected elements
        assert "Dry Run" in output or "dry run" in output.lower()
        assert FIXTURE_MODEL in output
        assert "Describe this image" in output

    @pytest.mark.skipif(
        not _check_model_cached(FIXTURE_MODEL),
        reason=f"Model {FIXTURE_MODEL} not cached (run manually first)",
    )
    def test_full_inference_with_fixture_model(self, e2e_test_image: Path) -> None:
        """Full inference run should complete successfully and produce outputs."""
        result = subprocess.run(
            [
                sys.executable,
                str(_CHECK_MODELS_SCRIPT),
                *_get_e2e_output_args(),
                "--image",
                str(e2e_test_image),
                "--models",
                FIXTURE_MODEL,
                "--prompt",
                "Describe the main elements in this image briefly.",
                "--max-tokens",
                "100",  # Limit for faster completion
                "--timeout",
                "120",  # Model timeout
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=E2E_TIMEOUT,
        )

        # Check successful completion
        assert result.returncode == 0, f"Inference failed: {result.stderr}"

        # Verify output files were created
        assert _E2E_HTML.exists(), "HTML report not created"
        assert _E2E_MD.exists(), "Markdown report not created"
        assert _E2E_JSONL.exists(), "JSONL report not created"

        # Verify JSONL contains valid result
        with _E2E_JSONL.open() as f:
            records = [json.loads(line) for line in f if line.strip()]

        assert len(records) >= 1, "No results in JSONL"
        record = records[0]
        assert record["model"] == FIXTURE_MODEL
        assert record["success"] is True, f"Model failed: {record.get('error_message')}"

        # Verify metrics are present
        metrics = record.get("metrics", {})
        assert metrics.get("generation_tokens", 0) > 0, "No tokens generated"
        assert metrics.get("generation_tps", 0) > 0, "TPS not recorded"

    @pytest.mark.skipif(
        not _check_model_cached(FIXTURE_MODEL),
        reason=f"Model {FIXTURE_MODEL} not cached (run manually first)",
    )
    def test_quality_analysis_produces_output(self, e2e_test_image: Path) -> None:
        """Quality analysis should run and detect potential issues."""
        result = subprocess.run(
            [
                sys.executable,
                str(_CHECK_MODELS_SCRIPT),
                *_get_e2e_output_args(),
                "--image",
                str(e2e_test_image),
                "--models",
                FIXTURE_MODEL,
                "--prompt",
                "Describe this image in detail. Include colors and shapes.",
                "--max-tokens",
                "150",
                "--verbose",  # Enable quality analysis logging
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=E2E_TIMEOUT,
        )

        assert result.returncode == 0, f"Quality analysis failed: {result.stderr}"
        output = result.stdout + result.stderr

        # Check that quality analysis ran (may or may not find issues)
        # Look for quality-related indicators in output
        assert any(
            indicator in output.lower()
            for indicator in [
                "quality",
                "tps",
                "generated",
                "tokens",
                "memory",
            ]
        ), "Expected quality/metrics output not found"

    def test_invalid_model_produces_error(self, e2e_test_image: Path) -> None:
        """Non-existent model should produce a clear error."""
        result = subprocess.run(
            [
                sys.executable,
                str(_CHECK_MODELS_SCRIPT),
                *_get_e2e_output_args(),
                "--image",
                str(e2e_test_image),
                "--models",
                "nonexistent/fake-model-12345",
                "--timeout",
                "30",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )

        # Should either fail or produce error result in output
        output = result.stdout + result.stderr
        # Check for error indicators
        assert any(
            indicator in output.lower()
            for indicator in ["error", "failed", "not found", "could not"]
        ), f"Expected error message not found in: {output[:500]}"


# Cleanup fixture
@pytest.fixture(autouse=True)
def cleanup_e2e_outputs() -> None:
    """Clean up E2E test output files after each test."""
    yield
    # Cleanup after test
    for file in [_E2E_LOG, _E2E_HTML, _E2E_MD, _E2E_TSV, _E2E_JSONL, _E2E_ENV]:
        if file.exists():
            with contextlib.suppress(OSError):
                file.unlink()
