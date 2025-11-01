"""Tests for model discovery and filtering."""

# ruff: noqa: ANN201
import pytest

import check_models


def test_get_cached_model_ids_returns_list():
    """Should return a list of model IDs from cache."""
    model_ids = check_models.get_cached_model_ids()
    assert isinstance(model_ids, list)
    # May be empty if no models cached
    for model_id in model_ids:
        assert isinstance(model_id, str)


def test_validate_model_identifier_accepts_valid_huggingface_format():
    """Should accept standard HuggingFace model identifiers."""
    # Should not raise
    check_models.validate_model_identifier("mlx-community/Qwen2-VL-2B-Instruct-4bit")
    check_models.validate_model_identifier("microsoft/Phi-3-vision-128k-instruct")
    check_models.validate_model_identifier("apple/OpenELM-270M")


def test_validate_model_identifier_accepts_local_paths():
    """Should accept local filesystem paths."""
    # Should not raise
    check_models.validate_model_identifier("./local/model")
    check_models.validate_model_identifier("/absolute/path/to/model")
    check_models.validate_model_identifier("../relative/path")


def test_validate_model_identifier_rejects_empty_string():
    """Should reject empty model identifier."""
    with pytest.raises(ValueError, match="Model identifier cannot be empty"):
        check_models.validate_model_identifier("")


def test_validate_model_identifier_rejects_whitespace_only():
    """Should reject whitespace-only identifiers."""
    with pytest.raises(ValueError, match="Model identifier cannot be empty"):
        check_models.validate_model_identifier("   ")
    with pytest.raises(ValueError, match="Model identifier cannot be empty"):
        check_models.validate_model_identifier("\t\n")


def test_validate_kv_params_valid_combinations():
    """Should accept valid KV cache parameter combinations."""
    # Should not raise
    check_models.validate_kv_params(kv_bits=None, max_kv_size=None)
    check_models.validate_kv_params(kv_bits=4, max_kv_size=1024)
    check_models.validate_kv_params(kv_bits=8, max_kv_size=2048)


def test_validate_kv_params_rejects_invalid_bits():
    """Should reject invalid kv_bits values."""
    with pytest.raises(ValueError, match="kv_bits must be"):
        check_models.validate_kv_params(kv_bits=16, max_kv_size=1024)


def test_validate_kv_params_rejects_negative_size():
    """Should reject negative max_kv_size."""
    with pytest.raises(ValueError, match="max_kv_size must be positive"):
        check_models.validate_kv_params(kv_bits=4, max_kv_size=-100)


def test_validate_kv_params_rejects_zero_size():
    """Should reject zero max_kv_size."""
    with pytest.raises(ValueError, match="max_kv_size must be positive"):
        check_models.validate_kv_params(kv_bits=4, max_kv_size=0)


def test_is_numeric_field_identifies_numeric_fields():
    """Should correctly identify numeric field names."""
    assert check_models.is_numeric_field("prompt_tps")
    assert check_models.is_numeric_field("generation_tps")
    assert check_models.is_numeric_field("total_time")
    assert check_models.is_numeric_field("peak_memory_gb")


def test_is_numeric_field_rejects_text_fields():
    """Should correctly identify non-numeric fields."""
    assert not check_models.is_numeric_field("model_identifier")
    assert not check_models.is_numeric_field("response")
    assert not check_models.is_numeric_field("error_message")


def test_is_numeric_value_identifies_numbers():
    """Should correctly identify numeric values."""
    assert check_models.is_numeric_value(42)
    assert check_models.is_numeric_value(3.14)
    assert check_models.is_numeric_value(0)
    assert check_models.is_numeric_value(-1.5)


def test_is_numeric_value_rejects_non_numbers():
    """Should reject non-numeric values."""
    assert not check_models.is_numeric_value("42")
    assert not check_models.is_numeric_value("text")
    assert not check_models.is_numeric_value(None)
    assert not check_models.is_numeric_value([1, 2, 3])
