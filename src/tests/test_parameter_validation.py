"""Tests for parameter validation functions."""

import pytest

from check_models import (
    validate_kv_params,
    validate_sampling_params,
    validate_temperature,
)


class TestTemperatureValidation:
    """Test validate_temperature function."""

    def test_valid_temperatures(self) -> None:
        """Test that valid temperatures pass validation."""
        validate_temperature(0.0)  # Minimum valid
        validate_temperature(0.1)  # Default
        validate_temperature(0.5)  # Mid-range
        validate_temperature(1.0)  # Common value
        validate_temperature(1.5)  # Higher but reasonable

    def test_negative_temperature_raises_error(self) -> None:
        """Test that negative temperature raises ValueError."""
        with pytest.raises(ValueError, match="Temperature must be non-negative"):
            validate_temperature(-0.1)

        with pytest.raises(ValueError, match="Temperature must be non-negative"):
            validate_temperature(-1.0)

    def test_high_temperature_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that very high temperature triggers warning."""
        validate_temperature(2.5)  # Should warn but not raise
        assert "unusually high" in caplog.text.lower()


class TestSamplingParamsValidation:
    """Test validate_sampling_params function."""

    def test_valid_top_p_values(self) -> None:
        """Test that valid top_p values pass validation."""
        validate_sampling_params(top_p=0.0, repetition_penalty=None)
        validate_sampling_params(top_p=0.5, repetition_penalty=None)
        validate_sampling_params(top_p=0.9, repetition_penalty=None)
        validate_sampling_params(top_p=1.0, repetition_penalty=None)

    def test_invalid_top_p_raises_error(self) -> None:
        """Test that invalid top_p raises ValueError."""
        with pytest.raises(ValueError, match="top_p must be between"):
            validate_sampling_params(top_p=-0.1, repetition_penalty=None)

        with pytest.raises(ValueError, match="top_p must be between"):
            validate_sampling_params(top_p=1.1, repetition_penalty=None)

        with pytest.raises(ValueError, match="top_p must be between"):
            validate_sampling_params(top_p=2.0, repetition_penalty=None)

    def test_valid_repetition_penalty_values(self) -> None:
        """Test that valid repetition_penalty values pass validation."""
        validate_sampling_params(top_p=1.0, repetition_penalty=None)  # Disabled
        validate_sampling_params(top_p=1.0, repetition_penalty=1.0)  # Minimum
        validate_sampling_params(top_p=1.0, repetition_penalty=1.2)  # Typical
        validate_sampling_params(top_p=1.0, repetition_penalty=2.0)  # High

    def test_invalid_repetition_penalty_raises_error(self) -> None:
        """Test that repetition_penalty < 1.0 raises ValueError."""
        with pytest.raises(ValueError, match=r"repetition_penalty must be >= 1\.0"):
            validate_sampling_params(top_p=1.0, repetition_penalty=0.9)

        with pytest.raises(ValueError, match=r"repetition_penalty must be >= 1\.0"):
            validate_sampling_params(top_p=1.0, repetition_penalty=0.0)

        with pytest.raises(ValueError, match=r"repetition_penalty must be >= 1\.0"):
            validate_sampling_params(top_p=1.0, repetition_penalty=-1.0)

    def test_combined_valid_params(self) -> None:
        """Test valid combinations of sampling parameters."""
        validate_sampling_params(top_p=0.9, repetition_penalty=1.2)
        validate_sampling_params(top_p=0.95, repetition_penalty=1.1)
        validate_sampling_params(top_p=1.0, repetition_penalty=1.5)


class TestKVParamsValidation:
    """Test validate_kv_params function."""

    def test_valid_max_kv_size_values(self) -> None:
        """Test that valid max_kv_size values pass validation."""
        validate_kv_params(max_kv_size=None, kv_bits=None)  # Disabled
        validate_kv_params(max_kv_size=1024, kv_bits=None)
        validate_kv_params(max_kv_size=4096, kv_bits=None)
        validate_kv_params(max_kv_size=8192, kv_bits=None)

    def test_invalid_max_kv_size_raises_error(self) -> None:
        """Test that max_kv_size <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="max_kv_size must be > 0"):
            validate_kv_params(max_kv_size=0, kv_bits=None)

        with pytest.raises(ValueError, match="max_kv_size must be > 0"):
            validate_kv_params(max_kv_size=-1, kv_bits=None)

    def test_valid_kv_bits_values(self) -> None:
        """Test that valid kv_bits values pass validation."""
        validate_kv_params(max_kv_size=None, kv_bits=None)  # Disabled
        validate_kv_params(max_kv_size=None, kv_bits=4)
        validate_kv_params(max_kv_size=None, kv_bits=8)

    def test_invalid_kv_bits_raises_error(self) -> None:
        """Test that invalid kv_bits raises ValueError."""
        with pytest.raises(ValueError, match="kv_bits must be 4 or 8"):
            validate_kv_params(max_kv_size=None, kv_bits=2)

        with pytest.raises(ValueError, match="kv_bits must be 4 or 8"):
            validate_kv_params(max_kv_size=None, kv_bits=16)

        with pytest.raises(ValueError, match="kv_bits must be 4 or 8"):
            validate_kv_params(max_kv_size=None, kv_bits=1)

    def test_combined_valid_kv_params(self) -> None:
        """Test valid combinations of KV cache parameters."""
        validate_kv_params(max_kv_size=4096, kv_bits=4)
        validate_kv_params(max_kv_size=8192, kv_bits=8)
        validate_kv_params(max_kv_size=2048, kv_bits=4)
