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
import importlib.util
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from huggingface_hub import scan_cache_dir
from huggingface_hub.errors import CacheNotFound
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Generator

# All paths relative to test file locations for portability
TEST_DIR = Path(__file__).parent
SRC_DIR = TEST_DIR.parent
OUTPUT_DIR = SRC_DIR / "output"

# =============================================================================
# EARLY ENVIRONMENT SETUP (before any HuggingFace imports cache paths)
# =============================================================================

# Set up HF cache directory early, before any huggingface_hub functions cache the path.
# This is needed for CI environments that don't have ~/.cache/huggingface/hub
_HF_CACHE_DIR = Path(tempfile.gettempdir()) / "pytest_hf_cache" / "hub"
_HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HUB_CACHE", str(_HF_CACHE_DIR))
os.environ.setdefault("HF_HOME", str(_HF_CACHE_DIR.parent))


# =============================================================================
# ENVIRONMENT FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def setup_hf_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure HuggingFace cache directory exists for all tests.

    CI environments like GitHub Actions runners may not have the default
    HF cache directory, causing CacheNotFound errors. This fixture creates
    a temporary cache directory for each test.
    """
    cache_dir = tmp_path / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HF_HUB_CACHE", str(cache_dir))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf_home"))


# =============================================================================
# LOGGING FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def reset_logger_handlers() -> Generator[None]:
    """Reset check_models logger handlers before each test to avoid closed stream issues.

    The check_models module configures its logger with sys.stderr at import time.
    When pytest captures output, the stream may be swapped, causing "I/O operation
    on closed file" errors. This fixture ensures handlers use a fresh stream and
    propagate is enabled for caplog to work correctly.
    """
    # Import lazily to avoid circular imports at module load time
    from check_models import logger  # noqa: PLC0415

    # Save original state
    original_propagate = logger.propagate
    original_handlers = logger.handlers[:]

    # Enable propagation so caplog can capture records
    logger.propagate = True

    yield

    # After test: restore original state and reset handlers to use current sys.stderr
    logger.propagate = original_propagate
    logger.handlers = original_handlers

    # Ensure handlers use current sys.stderr to prevent "closed file" errors
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.stream = sys.stderr


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
    """Create a realistic test image with visual elements (for E2E tests)."""
    img_path = tmp_path / "realistic.jpg"
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
            time.sleep(0.05)
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
    return importlib.util.find_spec("mlx_vlm") is not None


@pytest.fixture(scope="session")
def fixture_model_cached() -> bool:
    """Check if the fixture model (nanoLLaVA) is cached (session-scoped)."""
    fixture_model = "qnguyen3/nanoLLaVA"
    try:
        repo_ids = [r.repo_id for r in scan_cache_dir().repos]
    except (OSError, ValueError, RuntimeError, CacheNotFound):
        return False
    else:
        return fixture_model in repo_ids


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
