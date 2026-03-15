"""Tests for parameter validation functions."""

import argparse

import pytest

from check_models import (
    validate_cli_arguments,
    validate_kv_params,
    validate_sampling_params,
    validate_temperature,
)


class TestTemperatureValidation:
    """Test validate_temperature function."""

    def test_valid_temperatures(self) -> None:
        """Test that valid temperatures pass validation."""
        validate_temperature(temp=0.0)  # Minimum valid
        validate_temperature(temp=0.1)  # Default
        validate_temperature(temp=0.5)  # Mid-range
        validate_temperature(temp=1.0)  # Common value
        validate_temperature(temp=1.5)  # Higher but reasonable

    def test_negative_temperature_raises_error(self) -> None:
        """Test that negative temperature raises ValueError."""
        with pytest.raises(ValueError, match="Temperature must be non-negative"):
            validate_temperature(temp=-0.1)

        with pytest.raises(ValueError, match="Temperature must be non-negative"):
            validate_temperature(temp=-1.0)

    def test_high_temperature_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that very high temperature triggers warning."""
        validate_temperature(temp=2.5)  # Should warn but not raise
        assert "unusually high" in caplog.text.lower()


class TestSamplingParamsValidation:
    """Test validate_sampling_params function."""

    def test_valid_top_p_values(self) -> None:
        """Test that valid top_p values pass validation."""
        validate_sampling_params(top_p=0.0, min_p=0.0, top_k=0, repetition_penalty=None)
        validate_sampling_params(top_p=0.5, min_p=0.0, top_k=0, repetition_penalty=None)
        validate_sampling_params(top_p=0.9, min_p=0.0, top_k=0, repetition_penalty=None)
        validate_sampling_params(top_p=1.0, min_p=0.0, top_k=0, repetition_penalty=None)

    def test_invalid_top_p_raises_error(self) -> None:
        """Test that invalid top_p raises ValueError."""
        with pytest.raises(ValueError, match="top_p must be between"):
            validate_sampling_params(top_p=-0.1, min_p=0.0, top_k=0, repetition_penalty=None)

        with pytest.raises(ValueError, match="top_p must be between"):
            validate_sampling_params(top_p=1.1, min_p=0.0, top_k=0, repetition_penalty=None)

        with pytest.raises(ValueError, match="top_p must be between"):
            validate_sampling_params(top_p=2.0, min_p=0.0, top_k=0, repetition_penalty=None)

    def test_valid_min_p_values(self) -> None:
        """Test that valid min_p values pass validation."""
        validate_sampling_params(top_p=1.0, min_p=0.0, top_k=0, repetition_penalty=None)
        validate_sampling_params(top_p=1.0, min_p=0.2, top_k=0, repetition_penalty=None)
        validate_sampling_params(top_p=1.0, min_p=1.0, top_k=0, repetition_penalty=None)

    def test_invalid_min_p_raises_error(self) -> None:
        """Test that invalid min_p raises ValueError."""
        with pytest.raises(ValueError, match="min_p must be between"):
            validate_sampling_params(top_p=1.0, min_p=-0.1, top_k=0, repetition_penalty=None)

        with pytest.raises(ValueError, match="min_p must be between"):
            validate_sampling_params(top_p=1.0, min_p=1.1, top_k=0, repetition_penalty=None)

    def test_valid_top_k_values(self) -> None:
        """Test that valid top_k values pass validation."""
        validate_sampling_params(top_p=1.0, min_p=0.0, top_k=0, repetition_penalty=None)
        validate_sampling_params(top_p=1.0, min_p=0.0, top_k=1, repetition_penalty=None)
        validate_sampling_params(top_p=1.0, min_p=0.0, top_k=40, repetition_penalty=None)

    def test_invalid_top_k_raises_error(self) -> None:
        """Test that invalid top_k raises ValueError."""
        with pytest.raises(ValueError, match=r"top_k must be >= 0"):
            validate_sampling_params(top_p=1.0, min_p=0.0, top_k=-1, repetition_penalty=None)

    def test_valid_repetition_penalty_values(self) -> None:
        """Test that valid repetition_penalty values pass validation."""
        validate_sampling_params(top_p=1.0, min_p=0.0, top_k=0, repetition_penalty=None)
        validate_sampling_params(top_p=1.0, min_p=0.0, top_k=0, repetition_penalty=1.0)
        validate_sampling_params(top_p=1.0, min_p=0.0, top_k=0, repetition_penalty=1.2)
        validate_sampling_params(top_p=1.0, min_p=0.0, top_k=0, repetition_penalty=2.0)

    def test_invalid_repetition_penalty_raises_error(self) -> None:
        """Test that repetition_penalty < 1.0 raises ValueError."""
        with pytest.raises(ValueError, match=r"repetition_penalty must be >= 1\.0"):
            validate_sampling_params(top_p=1.0, min_p=0.0, top_k=0, repetition_penalty=0.9)

        with pytest.raises(ValueError, match=r"repetition_penalty must be >= 1\.0"):
            validate_sampling_params(top_p=1.0, min_p=0.0, top_k=0, repetition_penalty=0.0)

        with pytest.raises(ValueError, match=r"repetition_penalty must be >= 1\.0"):
            validate_sampling_params(top_p=1.0, min_p=0.0, top_k=0, repetition_penalty=-1.0)

    def test_combined_valid_params(self) -> None:
        """Test valid combinations of sampling parameters."""
        validate_sampling_params(top_p=0.9, min_p=0.05, top_k=40, repetition_penalty=1.2)
        validate_sampling_params(top_p=0.95, min_p=0.1, top_k=8, repetition_penalty=1.1)
        validate_sampling_params(top_p=1.0, min_p=0.0, top_k=0, repetition_penalty=1.5)


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


class TestCliArgumentNormalization:
    """Test CLI-only normalization and validation helpers."""

    @staticmethod
    def _build_args(**overrides: object) -> argparse.Namespace:
        base: dict[str, object] = {
            "temperature": 0.0,
            "max_tokens": 10,
            "top_p": 1.0,
            "min_p": 0.0,
            "top_k": 0,
            "repetition_penalty": None,
            "max_kv_size": None,
            "kv_bits": None,
            "verbose": False,
            "detailed_metrics": False,
            "resize_shape": None,
            "eos_tokens": None,
            "processor_kwargs": None,
            "enable_thinking": False,
            "thinking_budget": None,
            "thinking_start_token": None,
            "thinking_end_token": "</think>",
        }
        base.update(overrides)
        return argparse.Namespace(**base)

    def test_cli_argument_normalization_decodes_and_shapes_values(self) -> None:
        """CLI validation should normalize resize, EOS, and processor kwargs values."""
        args = self._build_args(
            resize_shape=[512],
            eos_tokens=[r"</think>", r"\n"],
            processor_kwargs={"cropping": False, "max_patches": 3},
        )

        validate_cli_arguments(args)

        assert args.resize_shape == (512, 512)
        assert args.eos_tokens == ("</think>", "\n")
        assert args.processor_kwargs == {"cropping": False, "max_patches": 3}

    def test_invalid_resize_shape_raises_error(self) -> None:
        """Resize shape should reject anything other than one or two positive ints."""
        args = self._build_args(resize_shape=[224, 224, 224])

        with pytest.raises(ValueError, match="resize_shape must contain 1 or 2 integers"):
            validate_cli_arguments(args)

    def test_reserved_processor_kwargs_raise_error(self) -> None:
        """Processor kwargs should not be allowed to override dedicated CLI flags."""
        args = self._build_args(processor_kwargs={"top_k": 10, "cropping": False})

        with pytest.raises(ValueError, match="processor_kwargs cannot override dedicated"):
            validate_cli_arguments(args)

    def test_invalid_min_p_in_cli_args_raises_error(self) -> None:
        """CLI validation should reject min_p outside the upstream range."""
        args = self._build_args(min_p=1.2)

        with pytest.raises(ValueError, match="min_p must be between"):
            validate_cli_arguments(args)

    def test_invalid_top_k_in_cli_args_raises_error(self) -> None:
        """CLI validation should reject negative top_k values."""
        args = self._build_args(top_k=-5)

        with pytest.raises(ValueError, match=r"top_k must be >= 0"):
            validate_cli_arguments(args)

    def test_thinking_budget_requires_enable_thinking(self) -> None:
        """Thinking budget should be rejected unless thinking mode is explicitly enabled."""
        args = self._build_args(thinking_budget=64)

        with pytest.raises(ValueError, match="require --enable-thinking"):
            validate_cli_arguments(args)

    def test_invalid_thinking_budget_raises_error(self) -> None:
        """Thinking budget should reject non-positive values."""
        args = self._build_args(enable_thinking=True, thinking_budget=0)

        with pytest.raises(ValueError, match="thinking_budget must be > 0"):
            validate_cli_arguments(args)

    def test_empty_thinking_end_token_raises_error(self) -> None:
        """Thinking mode should require a non-empty end token."""
        args = self._build_args(enable_thinking=True, thinking_end_token="")

        with pytest.raises(ValueError, match="thinking_end_token must be non-empty"):
            validate_cli_arguments(args)

    def test_detailed_metrics_without_verbose_warns(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Detailed metrics should warn when requested without verbose mode."""
        args = self._build_args(detailed_metrics=True, verbose=False)

        validate_cli_arguments(args)

        assert "has no effect unless --verbose is also set" in caplog.text
