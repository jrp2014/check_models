"""Tests for parameter validation functions."""

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import pytest

import check_models
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
        for bits in (2, 3, 4, 5, 6, 8):
            validate_kv_params(max_kv_size=None, kv_bits=bits)
        validate_kv_params(max_kv_size=None, kv_bits=3.5)
        validate_kv_params(
            max_kv_size=None,
            kv_bits=3.0,
            kv_quant_scheme="turboquant",
        )

    def test_invalid_kv_bits_raises_error(self) -> None:
        """Test that invalid kv_bits raises ValueError."""
        with pytest.raises(ValueError, match="kv_bits must be >= 1"):
            validate_kv_params(max_kv_size=None, kv_bits=0.5)

        with pytest.raises(ValueError, match=r"integer or \.5 increment"):
            validate_kv_params(max_kv_size=None, kv_bits=3.25)

        with pytest.raises(ValueError, match="uniform kv_bits must be one of"):
            validate_kv_params(max_kv_size=None, kv_bits=16)

    def test_combined_valid_kv_params(self) -> None:
        """Test valid combinations of KV cache parameters."""
        validate_kv_params(max_kv_size=4096, kv_bits=4)
        validate_kv_params(max_kv_size=8192, kv_bits=8)
        validate_kv_params(max_kv_size=2048, kv_bits=3.5)

    def test_valid_per_tensor_kv_overrides(self) -> None:
        """Per-tensor KV overrides pass validation alongside a base kv_bits."""
        validate_kv_params(max_kv_size=None, kv_bits=4, kv_key_bits=8, kv_value_bits=4)
        # Fractional kv_bits implies a TurboQuant base, so .5 overrides are fine.
        validate_kv_params(max_kv_size=None, kv_bits=3.5, kv_key_bits=3.5, kv_value_bits=4.5)
        # An explicit turboquant per-tensor scheme lifts the uniform bit set.
        validate_kv_params(
            max_kv_size=None,
            kv_bits=8,
            kv_value_bits=3.5,
            kv_value_scheme="turboquant",
        )
        # Scheme-only overrides are valid without per-tensor bit widths.
        validate_kv_params(
            max_kv_size=None,
            kv_bits=8,
            kv_key_scheme="turboquant",
            kv_value_scheme="uniform",
        )

    def test_per_tensor_kv_overrides_require_kv_bits(self) -> None:
        """Per-tensor overrides without kv_bits would be silent upstream no-ops."""
        with pytest.raises(ValueError, match="require kv_bits: kv_key_bits"):
            validate_kv_params(max_kv_size=None, kv_bits=None, kv_key_bits=8)

        with pytest.raises(ValueError, match="require kv_bits: kv_value_scheme"):
            validate_kv_params(max_kv_size=None, kv_bits=None, kv_value_scheme="turboquant")

    def test_invalid_per_tensor_kv_bits_raise_error(self) -> None:
        """Per-tensor bit widths follow the same range and increment rules."""
        with pytest.raises(ValueError, match="kv_key_bits must be >= 1"):
            validate_kv_params(max_kv_size=None, kv_bits=4, kv_key_bits=0.5)

        with pytest.raises(ValueError, match=r"kv_value_bits must be an integer or \.5 increment"):
            validate_kv_params(max_kv_size=None, kv_bits=4, kv_value_bits=3.25)

        # Uniform base scheme restricts per-tensor bit widths to the mx.quantize set.
        with pytest.raises(ValueError, match="uniform kv_key_bits must be one of"):
            validate_kv_params(max_kv_size=None, kv_bits=4, kv_key_bits=16)

        # An explicit uniform per-tensor scheme rejects fractional bit widths.
        with pytest.raises(ValueError, match="uniform kv_value_bits must be one of"):
            validate_kv_params(
                max_kv_size=None,
                kv_bits=3.5,
                kv_value_bits=4.5,
                kv_value_scheme="uniform",
            )


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
            "seed": None,
            "repetition_penalty": None,
            "max_kv_size": None,
            "kv_bits": None,
            "kv_quant_scheme": "uniform",
            "presence_penalty": None,
            "presence_context_size": 20,
            "frequency_penalty": None,
            "frequency_context_size": 20,
            "logit_bias": None,
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

    def test_cli_argument_normalization_handles_accumulated_eos_tokens(self) -> None:
        """Normalization should decode EOS tokens accumulated across repeated flags."""
        parser = __import__("check_models")._build_cli_parser()
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

        validate_cli_arguments(args)

        assert args.eos_tokens == ("</think>", "\n", "<END>")

    def test_image_source_url_accepts_absolute_https_metadata(self) -> None:
        """A public source URL should be retained separately from the local input."""
        parser = check_models._build_cli_parser()

        args = parser.parse_args(
            [
                "--image",
                "local.jpg",
                "--image-source-url",
                "https://example.test/images/cats.jpg",
            ]
        )

        assert args.image == Path("local.jpg")
        assert args.image_source_url == "https://example.test/images/cats.jpg"

    @pytest.mark.parametrize(
        "value",
        [
            "cats.jpg",
            "file:///tmp/cats.jpg",
            "ftp://example.test/cats.jpg",
            "https:///cats.jpg",
        ],
    )
    def test_image_source_url_rejects_non_public_sources(self, value: str) -> None:
        """Source metadata must identify an absolute HTTP(S) location."""
        parser = check_models._build_cli_parser()

        with pytest.raises(SystemExit):
            parser.parse_args(["--image-source-url", value])

    def test_cli_argument_normalization_accepts_server_shared_request_controls(self) -> None:
        """Server request controls shared with generate() should parse and validate."""
        parser = __import__("check_models")._build_cli_parser()
        args = parser.parse_args(
            [
                "--folder",
                "test-folder",
                "--seed",
                "123",
                "--presence-penalty",
                "0.25",
                "--presence-context-size",
                "32",
                "--frequency-penalty",
                "0.5",
                "--frequency-context-size",
                "64",
                "--logit-bias",
                '{"42": -1.5, "123": 2}',
            ]
        )

        validate_cli_arguments(args)

        assert args.seed == 123
        assert args.presence_penalty == 0.25
        assert args.presence_context_size == 32
        assert args.frequency_penalty == 0.5
        assert args.frequency_context_size == 64
        assert args.logit_bias == {42: -1.5, 123: 2.0}

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

    def test_reserved_kv_quant_scheme_processor_kwarg_raises_error(self) -> None:
        """Processor kwargs should not be allowed to override KV quantization backend."""
        args = self._build_args(processor_kwargs={"kv_quant_scheme": "turboquant"})

        with pytest.raises(ValueError, match="processor_kwargs cannot override dedicated"):
            validate_cli_arguments(args)

    def test_reserved_server_shared_processor_kwarg_raises_error(self) -> None:
        """Processor kwargs should not override dedicated server-shared request controls."""
        args = self._build_args(processor_kwargs={"presence_penalty": 0.5})

        with pytest.raises(ValueError, match="processor_kwargs cannot override dedicated"):
            validate_cli_arguments(args)

    def test_reserved_per_tensor_kv_processor_kwarg_raises_error(self) -> None:
        """Processor kwargs should not override the per-tensor KV override flags."""
        args = self._build_args(processor_kwargs={"kv_key_bits": 8})

        with pytest.raises(ValueError, match="processor_kwargs cannot override dedicated"):
            validate_cli_arguments(args)

    def test_per_tensor_kv_flags_parse_and_validate(self) -> None:
        """The per-tensor KV override flags should parse and pass CLI validation."""
        parser = check_models._build_cli_parser()
        args = parser.parse_args(
            [
                "--folder",
                "test-folder",
                "--kv-bits",
                "8",
                "--kv-key-bits",
                "8",
                "--kv-value-bits",
                "3.5",
                "--kv-value-scheme",
                "turboquant",
            ]
        )

        validate_cli_arguments(args)

        assert args.kv_key_bits == 8.0
        assert args.kv_value_bits == 3.5
        assert args.kv_key_scheme is None
        assert args.kv_value_scheme == "turboquant"

    def test_per_tensor_kv_flags_without_kv_bits_raise(self) -> None:
        """Per-tensor KV flags without --kv-bits should fail CLI validation."""
        parser = check_models._build_cli_parser()
        args = parser.parse_args(["--folder", "test-folder", "--kv-key-bits", "8"])

        with pytest.raises(ValueError, match="require kv_bits"):
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

    def test_invalid_presence_context_size_raises_error(self) -> None:
        """Presence penalty context must be positive when provided."""
        args = self._build_args(presence_context_size=0)

        with pytest.raises(ValueError, match="presence_context_size must be > 0"):
            validate_cli_arguments(args)

    def test_invalid_frequency_context_size_raises_error(self) -> None:
        """Frequency penalty context must be positive when provided."""
        args = self._build_args(frequency_context_size=-1)

        with pytest.raises(ValueError, match="frequency_context_size must be > 0"):
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

    def test_parser_defaults_eval_mode_to_auto(self) -> None:
        """Default eval mode should be resolved after image metadata is known."""
        parser = check_models._build_cli_parser()
        args = parser.parse_args(["--folder", "test-folder"])

        assert args.eval_mode == "auto"

    def test_retained_output_defaults(self) -> None:
        """Parser defaults should expose only the retained configurable artifacts."""
        parser = check_models._build_cli_parser()
        args = parser.parse_args([])

        assert args.output_html == check_models.DEFAULT_HTML_OUTPUT
        assert args.output_gallery_markdown == check_models.DEFAULT_GALLERY_MD_OUTPUT
        assert args.output_jsonl == check_models.DEFAULT_JSONL_OUTPUT
        assert args.output_run_json == check_models.DEFAULT_RUN_JSON_OUTPUT
        assert args.output_diagnostics == check_models.DEFAULT_DIAGNOSTICS_OUTPUT

    @pytest.mark.parametrize(
        "retired_flag",
        [
            "--output-markdown",
            "--output-review",
            "--output-model-selection",
            "--output-model-capabilities",
            "--output-model-capabilities-json",
            "--output-tsv",
        ],
    )
    def test_retired_output_flags_are_rejected(self, retired_flag: str) -> None:
        """Removed artifact flags must fail loudly instead of being accepted and ignored."""
        parser = check_models._build_cli_parser()

        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args([retired_flag, "retired-output"])

        assert exc_info.value.code == 2

    def test_retained_output_destinations_can_be_overridden(
        self,
        tmp_path: Path,
    ) -> None:
        """Parser should accept explicit retained artifact destinations."""
        parser = check_models._build_cli_parser()
        html = tmp_path / "results.html"
        gallery = tmp_path / "gallery.md"
        jsonl = tmp_path / "results.jsonl"
        run_json = tmp_path / "run.json"
        diagnostics = tmp_path / "diagnostics.md"

        args = parser.parse_args(
            [
                "--output-html",
                str(html),
                "--output-gallery-markdown",
                str(gallery),
                "--output-jsonl",
                str(jsonl),
                "--output-run-json",
                str(run_json),
                "--output-diagnostics",
                str(diagnostics),
            ],
        )

        assert args.output_html == html
        assert args.output_gallery_markdown == gallery
        assert args.output_jsonl == jsonl
        assert args.output_run_json == run_json
        assert args.output_diagnostics == diagnostics

    def test_auto_eval_mode_uses_assisted_lane_when_descriptive_metadata_exists(self) -> None:
        """Auto mode should use metadata-assisted cataloguing when references exist."""
        args = self._build_args(eval_mode="auto", max_tokens=None)

        check_models._apply_eval_mode_defaults(
            args,
            {
                "date": "2026-06-12",
                "description": "Two cats lounging on a couch.",
                "keywords": "cats, couch, remote controls",
                "exif": "{...}",
            },
        )

        assert args.eval_mode == "assisted"
        assert args.max_tokens == check_models.DEFAULT_MAX_TOKENS

    def test_auto_eval_mode_uses_blind_lane_with_capture_metadata_only(self) -> None:
        """Auto mode should withhold capture-only metadata in the blind lane."""
        args = self._build_args(eval_mode="auto", max_tokens=None)

        check_models._apply_eval_mode_defaults(
            args,
            {
                "date": "2026-06-12",
                "time": "12:34:56",
                "gps": "51.5074,-0.1278",
                "exif": "{...}",
            },
        )

        assert args.eval_mode == "blind"
        assert args.max_tokens == check_models.DEFAULT_MAX_TOKENS

    def test_auto_eval_mode_uses_blind_lane_without_metadata(self) -> None:
        """Auto mode should run the structured blind benchmark without references."""
        args = self._build_args(eval_mode="auto", max_tokens=None)

        check_models._apply_eval_mode_defaults(
            args,
            {
                "date": None,
                "time": None,
                "gps": None,
                "description": None,
                "title": None,
                "keywords": None,
                "exif": "{}",
            },
        )

        assert args.eval_mode == "blind"
        assert args.max_tokens == check_models.DEFAULT_MAX_TOKENS

    def test_auto_eval_mode_preserves_custom_token_cap_without_metadata(self) -> None:
        """Metadata-aware defaults should not overwrite an already-custom token cap."""
        args = self._build_args(eval_mode="auto", max_tokens=321)

        check_models._apply_eval_mode_defaults(args, {})

        assert args.eval_mode == "blind"
        assert args.max_tokens == 321

    def test_parser_defaults_max_tokens_to_unset_sentinel(self) -> None:
        """The parser must not pre-fill max_tokens; the lane resolves the default."""
        parser = check_models._build_cli_parser()

        args = parser.parse_args(["--dry-run"])

        assert args.max_tokens is None

    def test_triage_lane_applies_its_token_cap_when_unset(self) -> None:
        """An unset token cap should resolve to the triage budget in the triage lane."""
        args = self._build_args(eval_mode="triage", max_tokens=None)

        check_models._apply_eval_mode_defaults(args, {})

        assert args.eval_mode == "triage"
        assert args.max_tokens == check_models.TRIAGE_MAX_TOKENS

    def test_explicit_default_valued_token_cap_survives_triage(self) -> None:
        """--max-tokens 500 must survive triage even though it equals the old default.

        The previous value-comparison sentinel could not distinguish an explicit
        500 from "unset" and silently replaced it with the triage cap.
        """
        args = self._build_args(
            eval_mode="triage",
            max_tokens=check_models.DEFAULT_MAX_TOKENS,
        )

        check_models._apply_eval_mode_defaults(args, {})

        assert args.eval_mode == "triage"
        assert args.max_tokens == check_models.DEFAULT_MAX_TOKENS

    def test_explicit_assisted_lane_requires_descriptive_metadata(self) -> None:
        """Assisted selection should fail instead of silently becoming a blind run."""
        args = self._build_args(eval_mode="assisted", max_tokens=None)

        with pytest.raises(ValueError, match=r"assisted.*descriptive metadata"):
            check_models._apply_eval_mode_defaults(args, {})

    def test_legacy_quality_mode_maps_to_assisted_with_quality_budget(self) -> None:
        """The quality alias should retain its token budget while recording the new lane."""
        args = self._build_args(eval_mode="quality", max_tokens=None)

        check_models._apply_eval_mode_defaults(args, {"description": "Reference caption"})

        assert args.eval_mode == "assisted"
        assert args.max_tokens == check_models.QUALITY_MAX_TOKENS

    @pytest.mark.parametrize("legacy_mode", ["stress", "quality"])
    def test_legacy_modes_map_to_blind_without_descriptive_metadata(
        self,
        legacy_mode: str,
    ) -> None:
        """Legacy inputs should remain aliases and never become persisted lanes."""
        args = self._build_args(eval_mode=legacy_mode, max_tokens=None)

        check_models._apply_eval_mode_defaults(args, {"date": "2026-07-10"})

        assert args.eval_mode == "blind"


class TestUpstreamCliParity:
    """Shared CLI flags must track mlx-vlm's generate CLI unless deliberately divergent."""

    # Flag -> reason the default deliberately differs from upstream generate.
    DELIBERATE_DEFAULT_DIVERGENCES: ClassVar[dict[str, str]] = {
        "--max-tokens": "lane-resolved default (500/200/1000) vs upstream free-form 2048",
        "--prompt": "harness builds a metadata-cataloguing prompt when unset",
        "--revision": "None distinguishes requested vs resolved revisions in reports",
        "--thinking-start-token": "None defers to the upstream default at call time",
        "--trust-remote-code": "harness defaults on, with a security warning and opt-out",
    }
    # Display/console flags are outside the generation contract; their
    # upstream defaults also differ between released and git-HEAD mlx-vlm,
    # and this test must pass against both.
    DISPLAY_ONLY_FLAGS: ClassVar[frozenset[str]] = frozenset({"--verbose"})

    @staticmethod
    def _capture_parser_defaults(build: Callable[[], object]) -> dict[str, object]:
        captured: dict[str, object] = {}
        real_parse_args = argparse.ArgumentParser.parse_args

        def _grab(self: argparse.ArgumentParser, *_args: object, **_kw: object) -> object:
            for action in self._actions:
                for option in action.option_strings:
                    if option.startswith("--"):
                        captured[option] = action.default
            return argparse.Namespace()

        argparse.ArgumentParser.parse_args = _grab  # type: ignore[method-assign, assignment]  # deliberate monkeypatch to capture parser defaults
        try:
            build()
        finally:
            argparse.ArgumentParser.parse_args = real_parse_args  # type: ignore[method-assign]  # restore the real parser
        return captured

    def test_shared_flag_defaults_match_mlx_vlm_generate(self) -> None:
        """Overlapping flags keep upstream defaults except documented divergences."""
        dispatch = pytest.importorskip("mlx_vlm.generate.dispatch")

        upstream = self._capture_parser_defaults(dispatch.parse_arguments)
        parser = check_models._build_cli_parser()
        ours: dict[str, object] = {}
        for action in parser._actions:
            for option in action.option_strings:
                if option.startswith("--"):
                    ours[option] = action.default

        shared = sorted((set(ours) & set(upstream)) - {"--help"})
        assert len(shared) >= 25, "shared CLI surface unexpectedly shrank"

        unexpected: dict[str, tuple[object, object]] = {}
        for option in shared:
            if option in self.DELIBERATE_DEFAULT_DIVERGENCES or option in self.DISPLAY_ONLY_FLAGS:
                continue
            # None on our side means "defer to upstream default at call time".
            if ours[option] is None and upstream[option] is not None:
                continue
            if ours[option] != upstream[option]:
                unexpected[option] = (ours[option], upstream[option])
        assert not unexpected, f"undocumented CLI default drift vs upstream: {unexpected}"
        # No reverse "allowlist must diverge" assertion: upstream defaults move
        # between releases and git HEAD, and this test must pass against both.


def test_dry_run_setup_writes_no_log_or_environment_files(
    tmp_path: Path,
    test_image: Path,
) -> None:
    """--dry-run must not overwrite the retained log/environment artifacts."""
    log_path = tmp_path / "check_models.log"
    env_path = tmp_path / "environment.log"
    parser = check_models._build_cli_parser()
    args = parser.parse_args(
        [
            "--dry-run",
            "--image",
            str(test_image),
            "--output-log",
            str(log_path),
            "--output-env",
            str(env_path),
        ]
    )

    check_models.setup_environment(args)

    assert not log_path.exists()
    assert not env_path.exists()

    args = parser.parse_args(
        [
            "--image",
            str(test_image),
            "--output-log",
            str(log_path),
            "--output-env",
            str(env_path),
        ]
    )
    check_models.setup_environment(args)
    assert log_path.exists()
    assert env_path.exists()


def test_per_tensor_kv_keywords_match_upstream_generate_kwargs() -> None:
    """Per-tensor KV keywords we send must exist in the installed GenerateKwargs.

    CI installs PyPI mlx-vlm releases, which may predate per-tensor KV cache
    quantization, so skip (rather than fail) when the installed contract lacks
    the keys.  Against a git-HEAD install this locks the forwarded names to the
    upstream TypedDict spelling.
    """
    types_module = pytest.importorskip("mlx_vlm.generate.types")
    annotations: dict[str, object] = getattr(types_module.GenerateKwargs, "__annotations__", {})
    per_tensor_keys = {"kv_key_bits", "kv_value_bits", "kv_key_scheme", "kv_value_scheme"}
    if not per_tensor_keys <= set(annotations):
        pytest.skip("installed mlx-vlm predates per-tensor KV cache quantization")

    assert per_tensor_keys <= set(check_models._SENT_GENERATE_KEYWORDS)
