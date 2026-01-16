"""Pytest configuration and shared fixtures for test suite.

This module provides:
- Shared fixtures for test images, folders, and paths
- Session-scoped fixtures for expensive resources
- Helper utilities for common test patterns
- Configuration for test markers and plugins

Performance optimizations:
- Session-scoped fixtures avoid repeated setup
- Module-scoped fixtures share resources across tests in same file
- Lazy imports defer heavy module loading
"""

from __future__ import annotations

import contextlib
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Generator

# =============================================================================
# PATH CONSTANTS
# =============================================================================

# All paths relative to test file locations for portability
TEST_DIR = Path(__file__).parent
SRC_DIR = TEST_DIR.parent
CHECK_MODELS_SCRIPT = SRC_DIR / "check_models.py"
OUTPUT_DIR = SRC_DIR / "output"


# =============================================================================
# SUBPROCESS HELPERS
# =============================================================================


def run_cli(
    args: list[str],
    *,
    timeout: float = 30.0,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run check_models CLI with given arguments.

    This is a shared helper that standardizes subprocess calls across tests,
    ensuring consistent timeout and capture behavior.

    Args:
        args: CLI arguments (without python/script path)
        timeout: Maximum execution time in seconds
        capture: Whether to capture stdout/stderr

    Returns:
        CompletedProcess with returncode, stdout, stderr
    """
    cmd = [sys.executable, str(CHECK_MODELS_SCRIPT), *args]
    return subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        timeout=timeout,
        check=False,
    )


def get_default_output_args(prefix: str = "test") -> list[str]:
    """Return CLI arguments for test-specific output files.

    Args:
        prefix: Prefix for output filenames (e.g., 'test_cli' -> test_cli.log)

    Returns:
        List of CLI arguments for output file paths
    """
    return [
        "--output-log",
        str(OUTPUT_DIR / f"{prefix}.log"),
        "--output-html",
        str(OUTPUT_DIR / f"{prefix}.html"),
        "--output-markdown",
        str(OUTPUT_DIR / f"{prefix}.md"),
        "--output-tsv",
        str(OUTPUT_DIR / f"{prefix}.tsv"),
        "--output-jsonl",
        str(OUTPUT_DIR / f"{prefix}.jsonl"),
        "--output-env",
        str(OUTPUT_DIR / f"{prefix}_env.log"),
    ]


# =============================================================================
# IMAGE FIXTURES
# =============================================================================


@pytest.fixture
def minimal_test_image(tmp_path: Path) -> Path:
    """Create a minimal valid test image (fastest, for CLI validation tests)."""
    img_path = tmp_path / "minimal.jpg"
    img = Image.new("RGB", (10, 10), color="red")
    img.save(img_path, "JPEG", quality=50)
    return img_path


@pytest.fixture
def test_image(tmp_path: Path) -> Path:
    """Create a small valid test image (100x100, for standard tests)."""
    img_path = tmp_path / "test.jpg"
    img = Image.new("RGB", (100, 100), color="red")
    img.save(img_path, "JPEG", quality=85)
    return img_path


@pytest.fixture
def realistic_test_image(tmp_path: Path) -> Path:
    """Create a realistic test image with visual elements (for E2E tests).

    Larger image (640x480) with sky, sun, grass, and house elements
    to give models something meaningful to describe.
    """
    img_path = tmp_path / "realistic.jpg"
    img = Image.new("RGB", (640, 480), color=(135, 206, 235))  # Sky blue

    pixels = img.load()
    assert pixels is not None

    # Draw a "sun" (yellow circle in top-right)
    for x in range(540, 600):
        for y in range(40, 100):
            if (x - 570) ** 2 + (y - 70) ** 2 < 900:
                pixels[x, y] = (255, 255, 0)

    # Draw "grass" (green strip at bottom)
    for x in range(640):
        for y in range(380, 480):
            pixels[x, y] = (34, 139, 34)

    # Draw "house" (brown rectangle)
    for x in range(200, 350):
        for y in range(250, 380):
            pixels[x, y] = (139, 90, 43)

    # Draw "roof" (red rectangle)
    for x in range(180, 370):
        for y in range(200, 250):
            pixels[x, y] = (178, 34, 34)

    img.save(img_path, "JPEG", quality=85)
    return img_path


# =============================================================================
# FOLDER FIXTURES
# =============================================================================


@pytest.fixture
def empty_folder(tmp_path: Path) -> Path:
    """Create an empty folder for testing no-image scenarios."""
    folder = tmp_path / "empty"
    folder.mkdir()
    return folder


@pytest.fixture
def folder_with_images(tmp_path: Path) -> Path:
    """Create a folder with multiple test images (different timestamps)."""
    folder = tmp_path / "images"
    folder.mkdir()

    for i, name in enumerate(["old.jpg", "middle.jpg", "newest.jpg"]):
        img_path = folder / name
        img = Image.new("RGB", (50, 50), color="blue")
        img.save(img_path)
        if i < 2:
            time.sleep(0.05)  # Reduced from 0.1s - enough for timestamp difference

    return folder


@pytest.fixture
def folder_with_single_image(tmp_path: Path) -> Path:
    """Create a folder with exactly one image."""
    folder = tmp_path / "single"
    folder.mkdir()
    img_path = folder / "only.jpg"
    img = Image.new("RGB", (50, 50), color="green")
    img.save(img_path)
    return folder


# =============================================================================
# ENVIRONMENT DETECTION FIXTURES
# =============================================================================


@pytest.fixture(scope="session")
def mlx_vlm_available() -> bool:
    """Check if mlx-vlm is available (session-scoped, checked once)."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import mlx_vlm"],
            check=False,
            capture_output=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0


@pytest.fixture(scope="session")
def fixture_model_cached() -> bool:
    """Check if the fixture model (nanoLLaVA) is cached (session-scoped)."""
    fixture_model = "qnguyen3/nanoLLaVA"
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                f"from huggingface_hub import scan_cache_dir; "
                f"ids = [r.repo_id for r in scan_cache_dir().repos]; "
                f"print('{fixture_model}' in ids)",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return "True" in result.stdout


# =============================================================================
# CLEANUP FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def cleanup_test_outputs() -> Generator[None]:
    """Clean up test output files after each test (autouse)."""
    yield
    # Cleanup after test - remove any test-prefixed output files
    for pattern in ["test_*.log", "test_*.html", "test_*.md", "test_*.tsv", "test_*.jsonl"]:
        for file in OUTPUT_DIR.glob(pattern):
            with contextlib.suppress(OSError):
                file.unlink()


# =============================================================================
# PYTEST HOOKS & CONFIGURATION
# =============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end integration tests")
    config.addinivalue_line("markers", "subprocess: marks tests that spawn subprocesses")


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Auto-mark tests based on their characteristics."""
    for item in items:
        # Auto-mark tests in test_e2e_smoke.py as slow and e2e
        if "test_e2e_smoke" in str(item.fspath):
            item.add_marker(pytest.mark.slow)
            item.add_marker(pytest.mark.e2e)
