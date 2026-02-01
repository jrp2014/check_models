"""Tests for model discovery and filtering."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# =============================================================================
# EARLY ENVIRONMENT SETUP (MUST happen before huggingface_hub imports)
# =============================================================================

# Set up HF cache directory early, before any huggingface_hub functions cache the path.
# Strategy (following HuggingFace documentation):
# 1. If HF_HUB_CACHE is set → use it (user explicitly configured)
# 2. Else if default cache exists (~/.cache/huggingface/hub) → use it
# 3. Else create temp cache (CI environment without cache)
_DEFAULT_HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

if "HF_HUB_CACHE" not in os.environ and not _DEFAULT_HF_CACHE.exists():
    # CI environment - create temp cache to prevent CacheNotFound
    _temp_hf_cache = Path(tempfile.gettempdir()) / "pytest_hf_cache"
    _temp_hf_cache.mkdir(parents=True, exist_ok=True)
    (_temp_hf_cache / "hub").mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_CACHE"] = str(_temp_hf_cache / "hub")
    os.environ["HF_HOME"] = str(_temp_hf_cache)

# Now import huggingface_hub after environment is configured
import pytest  # noqa: E402
from huggingface_hub.errors import CacheNotFound  # noqa: E402

import check_models  # noqa: E402


def test_get_cached_model_ids_returns_list() -> None:
    """Should return a list of model IDs from cache."""
    try:
        model_ids = check_models.get_cached_model_ids()
        assert isinstance(model_ids, list)
        # May be empty if no models cached
        for model_id in model_ids:
            assert isinstance(model_id, str)
    except CacheNotFound:
        pytest.skip("HuggingFace cache directory not found (expected in CI)")


def test_validate_model_identifier_accepts_valid_huggingface_format() -> None:
    """Should accept standard HuggingFace model identifiers."""
    # Should not raise
    check_models.validate_model_identifier("mlx-community/Qwen2-VL-2B-Instruct-4bit")
    check_models.validate_model_identifier("microsoft/Phi-3-vision-128k-instruct")
    check_models.validate_model_identifier("apple/OpenELM-270M")


def test_validate_model_identifier_accepts_local_paths(tmp_path: Path) -> None:
    """Should accept valid local paths."""
    # Create a dummy model directory
    model_dir = tmp_path / "local_model"
    model_dir.mkdir()
    check_models.validate_model_identifier(str(model_dir))


def test_validate_model_identifier_rejects_empty_string() -> None:
    """Should reject empty model identifier."""
    with pytest.raises(ValueError, match="Model identifier cannot be empty"):
        check_models.validate_model_identifier("")


def test_validate_model_identifier_rejects_whitespace_only() -> None:
    """Should reject whitespace-only identifiers."""
    with pytest.raises(ValueError, match="Model identifier cannot be empty"):
        check_models.validate_model_identifier("   ")
    with pytest.raises(ValueError, match="Model identifier cannot be empty"):
        check_models.validate_model_identifier("\t\n")


def test_validate_kv_params_valid_combinations() -> None:
    """Should accept valid KV cache parameter combinations."""
    # Should not raise
    check_models.validate_kv_params(kv_bits=None, max_kv_size=None)
    check_models.validate_kv_params(kv_bits=4, max_kv_size=1024)
    check_models.validate_kv_params(kv_bits=8, max_kv_size=2048)


def test_validate_kv_params_rejects_invalid_bits() -> None:
    """Should reject invalid kv_bits values."""
    with pytest.raises(ValueError, match="kv_bits must be"):
        check_models.validate_kv_params(kv_bits=16, max_kv_size=1024)


def test_validate_kv_params_rejects_negative_size() -> None:
    """Should reject negative max_kv_size."""
    with pytest.raises(ValueError, match="max_kv_size must be > 0"):
        check_models.validate_kv_params(kv_bits=4, max_kv_size=-100)


def test_validate_kv_params_rejects_zero_size() -> None:
    """Should reject zero max_kv_size."""
    with pytest.raises(ValueError, match="max_kv_size must be > 0"):
        check_models.validate_kv_params(kv_bits=4, max_kv_size=0)


def test_is_numeric_field_identifies_numeric_fields() -> None:
    """Should correctly identify numeric field names."""
    assert check_models.is_numeric_field("prompt_tps")
    assert check_models.is_numeric_field("generation_tps")
    assert check_models.is_numeric_field("total_time")
    assert check_models.is_numeric_field("peak_memory_gb")


def test_is_numeric_field_rejects_text_fields() -> None:
    """Should correctly identify non-numeric fields."""
    assert not check_models.is_numeric_field("model_identifier")
    assert not check_models.is_numeric_field("response")
    assert not check_models.is_numeric_field("error_message")


def test_is_numeric_value_identifies_numbers() -> None:
    """Should correctly identify numeric values."""
    assert check_models.is_numeric_value(42)
    assert check_models.is_numeric_value(3.14)
    assert check_models.is_numeric_value(0)
    assert check_models.is_numeric_value(-1.5)


def test_is_numeric_value_rejects_non_numbers() -> None:
    """Should reject non-numeric values."""
    assert not check_models.is_numeric_value("text")
    # Note: "42" is numeric (can be parsed as number)
    assert not check_models.is_numeric_value(None)
    assert not check_models.is_numeric_value([1, 2, 3])
