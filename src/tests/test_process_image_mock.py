"""Mock-based tests for process_image_with_model."""

from __future__ import annotations

import contextlib
import sys
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

import check_models

OPEN_THINK_MARKER = "<think>"
CLOSE_THINK_MARKER = "</think>"


@dataclass
class _FakeGenerationResult:
    """Minimal stand-in for mlx_vlm GenerationResult."""

    text: str = "Hello world"
    prompt_tokens: int = 50
    generation_tokens: int = 20
    prompt_tps: float = 100.0
    generation_tps: float = 42.0
    peak_memory: float = 1.2
    time: float = 0.0
    active_memory: float = 0.5
    cache_memory: float = 0.3


class _FakeModel:
    config: object = object()

    @staticmethod
    def parameters() -> list[object]:
        return []


class _FakeMxRuntime:
    """Minimal MLX runtime stand-in for mock-based generation tests."""

    @staticmethod
    def synchronize() -> None:
        return None

    @staticmethod
    def get_active_memory() -> float:
        return 0.0

    @staticmethod
    def get_cache_memory() -> float:
        return 0.0

    @staticmethod
    def get_peak_memory() -> float:
        return 0.0

    @staticmethod
    def eval(_params: object) -> None:
        return None


def _build_params(image_path: Path) -> check_models.ProcessImageParams:
    """Return default ProcessImageParams for testing."""
    return check_models.ProcessImageParams(
        model_identifier="test/fake-model",
        image_path=str(image_path),
        prompt="Describe this image.",
        max_tokens=50,
        temperature=0.0,
        timeout=30.0,
        verbose=False,
        trust_remote_code=True,
        top_p=1.0,
        min_p=0.0,
        top_k=0,
        repetition_penalty=None,
        repetition_context_size=20,
        lazy=False,
        max_kv_size=None,
        kv_bits=None,
        kv_group_size=64,
        quantized_kv_start=0,
    )


class TestProcessImageWithModelMock:
    """Tests using mocked internals to verify process_image_with_model orchestration."""

    def test_success_returns_performance_result(self, test_image: Path) -> None:
        """Successful generation should produce a PerformanceResult with success=True."""
        fake_result = _FakeGenerationResult()
        params = _build_params(test_image)

        with patch.object(
            check_models,
            "_run_model_generation",
            return_value=fake_result,
        ):
            result = check_models.process_image_with_model(params)

        assert result.success is True
        assert result.model_name == "test/fake-model"
        assert result.generation is not None
        assert result.quality_analysis is not None
        assert result.quality_issues is None
        assert result.runtime_diagnostics is not None
        assert result.runtime_diagnostics.first_token_latency_s == 0.5

    def test_timeout_returns_failure(self, test_image: Path) -> None:
        """TimeoutError during generation should produce success=False."""
        params = _build_params(test_image)

        with patch.object(
            check_models,
            "_run_model_generation",
            side_effect=TimeoutError("timed out"),
        ):
            result = check_models.process_image_with_model(params)

        assert result.success is False
        assert result.error_type == "TimeoutError"
        assert result.failure_phase is not None
        assert result.error_code is not None
        assert result.error_signature is not None

    def test_value_error_returns_failure(self, test_image: Path) -> None:
        """ValueError during generation should produce success=False with error info."""
        params = _build_params(test_image)

        with patch.object(
            check_models,
            "_run_model_generation",
            side_effect=ValueError("bad config"),
        ):
            result = check_models.process_image_with_model(params)

        assert result.success is False
        assert result.error_type == "ValueError"
        assert "bad config" in (result.error_message or "")
        assert result.failure_phase is not None
        assert result.error_code is not None

    def test_os_error_returns_failure(self, test_image: Path) -> None:
        """OSError during generation should produce success=False."""
        params = _build_params(test_image)

        with patch.object(
            check_models,
            "_run_model_generation",
            side_effect=OSError("disk full"),
        ):
            result = check_models.process_image_with_model(params)

        assert result.success is False
        assert result.error_type == "OSError"
        assert result.failure_phase is not None

    def test_failure_captures_stdout_and_stderr(self, test_image: Path) -> None:
        """Failure result should include captured stdout/stderr text."""
        params = _build_params(test_image)

        def _raise_after_output(*_args: object, **_kwargs: object) -> _FakeGenerationResult:
            sys.stdout.write("stdout marker\n")
            sys.stderr.write("stderr marker\n")
            error_message = "bad config"
            raise ValueError(error_message)

        with patch.object(
            check_models,
            "_run_model_generation",
            side_effect=_raise_after_output,
        ):
            result = check_models.process_image_with_model(params)

        assert result.success is False
        assert result.captured_output_on_fail is not None
        assert "stdout marker" in result.captured_output_on_fail
        assert "stderr marker" in result.captured_output_on_fail

    def test_failure_stdout_quality_is_analyzed(self, test_image: Path) -> None:
        """Captured stdout with model-like output should retain quality flags on failures."""
        params = _build_params(test_image)
        decode_error_message = "decode failed"

        def _raise_decode_failed() -> None:
            raise ValueError(decode_error_message)

        def _raise_after_repetitive_output(
            *_args: object,
            **_kwargs: object,
        ) -> _FakeGenerationResult:
            unreachable_message = "unreachable"
            sys.stdout.write(("loop " * 25).strip() + "\n")
            _raise_decode_failed()
            raise AssertionError(unreachable_message)

        with patch.object(
            check_models,
            "_run_model_generation",
            side_effect=_raise_after_repetitive_output,
        ):
            result = check_models.process_image_with_model(params)

        assert result.success is False
        assert result.quality_analysis is not None
        assert result.quality_analysis.is_repetitive is True
        assert result.quality_issues is not None
        assert "repetitive" in result.quality_issues

    def test_build_failure_result_helper_preserves_capture(self) -> None:
        """Centralized failure builder should preserve diagnostics fields."""
        result: check_models.PerformanceResult
        try:
            int("not-an-int")
        except ValueError as err:
            result = check_models._build_failure_result(
                model_name="test/fake-model",
                error=err,
                captured_output="=== STDERR ===\ntemplate failure",
            )
        else:  # pragma: no cover - defensive guard for static analysis
            raise AssertionError

        assert result.success is False
        assert result.error_type == "ValueError"
        assert result.failure_phase is None
        assert result.error_stage is not None
        assert result.error_code is not None
        assert result.error_signature is not None
        assert result.error_traceback is not None
        assert "template failure" in (result.captured_output_on_fail or "")

    def test_build_failure_result_preserves_quality_fields(self) -> None:
        """Failure builder should carry precomputed quality diagnostics when provided."""
        repeated_phrase = "loop"
        decode_error_message = "decode failed"

        def _raise_decode_failed() -> None:
            raise ValueError(decode_error_message)

        analysis = check_models.GenerationQualityAnalysis(
            is_repetitive=True,
            repeated_token=repeated_phrase,
            hallucination_issues=[],
            is_verbose=False,
            formatting_issues=[],
            has_excessive_bullets=False,
            bullet_count=0,
            is_context_ignored=False,
            missing_context_terms=[],
            is_refusal=False,
            refusal_type=None,
            is_generic=False,
            specificity_score=0.0,
            has_language_mixing=False,
            language_mixing_issues=[],
            has_degeneration=False,
            degeneration_type=None,
            has_fabrication=False,
            fabrication_issues=[],
            has_harness_issue=False,
            harness_issue_type=None,
            harness_issue_details=[],
            word_count=25,
            unique_ratio=0.1,
        )
        result: check_models.PerformanceResult
        try:
            _raise_decode_failed()
        except ValueError as err:
            result = check_models._build_failure_result(
                model_name="test/fake-model",
                error=err,
                captured_output="=== STDOUT ===\nloop loop loop",
                quality_issues="repetitive(loop)",
                quality_analysis=analysis,
            )
        else:  # pragma: no cover - defensive guard for static analysis
            raise AssertionError

        assert result.quality_analysis is analysis
        assert result.quality_issues == "repetitive(loop)"

    def test_build_failure_result_respects_tagged_phase(self) -> None:
        """Failure phase tags should flow into the final result payload."""
        err = check_models._tag_exception_failure_phase(ValueError("decode issue"), "decode")
        result = check_models._build_failure_result(
            model_name="test/fake-model",
            error=err,
            captured_output=None,
        )
        assert result.failure_phase == "decode"
        assert result.error_code is not None
        assert "_DECODE_" in result.error_code

    def test_ensure_generation_runtime_symbols_raises_for_api_drift(self) -> None:
        """Runtime contract drift should fail before model invocation starts."""
        with (
            patch.object(
                check_models,
                "_detect_runtime_api_drift_issues",
                return_value=(
                    "mlx_vlm.generate.generate is missing required keyword parameter(s): verbose.",
                ),
            ),
            pytest.raises(RuntimeError, match="Generation runtime API drift"),
        ):
            check_models._ensure_generation_runtime_symbols()

    def test_run_model_generation_passes_phase1_generate_kwargs(self, test_image: Path) -> None:
        """Phase-1 upstream-compatible CLI params should reach mlx_vlm.generate."""
        params = replace(
            _build_params(test_image),
            min_p=0.15,
            top_k=12,
            prefill_step_size=256,
            resize_shape=(512, 384),
            eos_tokens=("</think>", "\n"),
            skip_special_tokens=True,
            processor_kwargs={"cropping": False, "max_patches": 3},
        )

        fake_model = _FakeModel()
        fake_processor = object()
        fake_generation = _FakeGenerationResult()

        with (
            patch.object(check_models, "_ensure_generation_runtime_symbols"),
            patch.object(
                check_models,
                "_load_model",
                return_value=(fake_model, fake_processor, None),
            ),
            patch.object(check_models, "_run_model_preflight_validators"),
            patch.object(check_models, "apply_chat_template", return_value="formatted prompt"),
            patch.object(check_models, "generate", return_value=fake_generation) as mock_generate,
            patch.object(check_models, "mx", _FakeMxRuntime()),
        ):
            result = check_models._run_model_generation(params)

        assert result is fake_generation
        generate_kwargs = mock_generate.call_args.kwargs
        assert generate_kwargs["prompt"] == "formatted prompt"
        assert generate_kwargs["image"] == str(test_image)
        assert generate_kwargs["min_p"] == 0.15
        assert generate_kwargs["top_k"] == 12
        assert generate_kwargs["prefill_step_size"] == 256
        assert generate_kwargs["resize_shape"] == (512, 384)
        assert generate_kwargs["eos_tokens"] == ["</think>", "\n"]
        assert generate_kwargs["skip_special_tokens"] is True
        assert generate_kwargs["cropping"] is False
        assert generate_kwargs["max_patches"] == 3

    def test_run_model_generation_passes_thinking_kwargs(self, test_image: Path) -> None:
        """Thinking-mode flags should reach both chat templating and generation."""
        params = replace(
            _build_params(test_image),
            enable_thinking=True,
            thinking_budget=96,
            thinking_start_token=OPEN_THINK_MARKER,
            thinking_end_token=CLOSE_THINK_MARKER,
        )

        fake_model = _FakeModel()
        fake_processor = object()
        fake_generation = _FakeGenerationResult()

        with (
            patch.object(check_models, "_ensure_generation_runtime_symbols"),
            patch.object(
                check_models,
                "_load_model",
                return_value=(fake_model, fake_processor, None),
            ),
            patch.object(check_models, "_run_model_preflight_validators"),
            patch.object(
                check_models,
                "apply_chat_template",
                return_value="formatted prompt",
            ) as mock_template,
            patch.object(check_models, "generate", return_value=fake_generation) as mock_generate,
            patch.object(check_models, "mx", _FakeMxRuntime()),
        ):
            result = check_models._run_model_generation(params)

        assert result is fake_generation
        assert mock_template.call_args.kwargs["enable_thinking"] is True
        generate_kwargs = mock_generate.call_args.kwargs
        assert generate_kwargs["enable_thinking"] is True
        assert generate_kwargs["thinking_budget"] == 96
        assert generate_kwargs["thinking_start_token"] == "<think>"
        assert generate_kwargs["thinking_end_token"] == "</think>"

    def test_run_model_generation_retries_utf8_detokenizer_failure(self, test_image: Path) -> None:
        """Known mlx-vlm UTF-8 detokenizer failures should retry once with the patch."""
        params = _build_params(test_image)
        fake_model = _FakeModel()
        fake_processor = object()
        fake_generation = _FakeGenerationResult()
        generate_attempts: list[str] = []
        retry_patch_entries: list[str] = []
        decode_error = UnicodeDecodeError("utf-8", b"\xab", 0, 1, "invalid start byte")

        def _generate_side_effect(*_args: object, **_kwargs: object) -> _FakeGenerationResult:
            generate_attempts.append("attempt")
            if len(generate_attempts) == 1:
                raise decode_error
            return fake_generation

        @contextlib.contextmanager
        def _record_retry_patch() -> Generator[None]:
            retry_patch_entries.append("entered")
            yield

        with (
            patch.object(check_models, "_ensure_generation_runtime_symbols"),
            patch.object(
                check_models,
                "_load_model",
                return_value=(fake_model, fake_processor, None),
            ),
            patch.object(check_models, "_run_model_preflight_validators"),
            patch.object(check_models, "apply_chat_template", return_value="formatted prompt"),
            patch.object(
                check_models,
                "_is_mlx_vlm_bpe_detokenizer_decode_failure",
                return_value=True,
            ) as mock_detector,
            patch.object(
                check_models,
                "_temporary_mlx_vlm_lossy_bpe_detokenizer_patch",
                _record_retry_patch,
            ),
            patch.object(
                check_models,
                "generate",
                side_effect=_generate_side_effect,
            ) as mock_generate,
            patch.object(check_models, "mx", _FakeMxRuntime()),
        ):
            result = check_models._run_model_generation(params)

        assert result is fake_generation
        assert mock_generate.call_count == 2
        assert mock_detector.call_count == 1
        assert retry_patch_entries == ["entered"]

    def test_run_model_generation_retries_only_once_for_utf8_detokenizer_failure(
        self,
        test_image: Path,
    ) -> None:
        """The lossy detokenizer workaround should be a single retry, not an open loop."""
        params = _build_params(test_image)
        fake_model = _FakeModel()
        fake_processor = object()
        generate_attempts: list[str] = []
        retry_patch_entries: list[str] = []
        first_error = UnicodeDecodeError("utf-8", b"\xab", 0, 1, "invalid start byte")
        second_error = UnicodeDecodeError("utf-8", b"\xab", 0, 1, "invalid start byte")

        def _generate_side_effect(*_args: object, **_kwargs: object) -> _FakeGenerationResult:
            generate_attempts.append("attempt")
            if len(generate_attempts) == 1:
                raise first_error
            raise second_error

        @contextlib.contextmanager
        def _record_retry_patch() -> Generator[None]:
            retry_patch_entries.append("entered")
            yield

        with (
            patch.object(check_models, "_ensure_generation_runtime_symbols"),
            patch.object(
                check_models,
                "_load_model",
                return_value=(fake_model, fake_processor, None),
            ),
            patch.object(check_models, "_run_model_preflight_validators"),
            patch.object(check_models, "apply_chat_template", return_value="formatted prompt"),
            patch.object(
                check_models,
                "_is_mlx_vlm_bpe_detokenizer_decode_failure",
                return_value=True,
            ),
            patch.object(
                check_models,
                "_temporary_mlx_vlm_lossy_bpe_detokenizer_patch",
                _record_retry_patch,
            ),
            patch.object(
                check_models,
                "generate",
                side_effect=_generate_side_effect,
            ) as mock_generate,
            patch.object(check_models, "mx", _FakeMxRuntime()),
        ):
            try:
                check_models._run_model_generation(params)
            except ValueError as err:
                error_message = str(err)
            else:  # pragma: no cover - defensive guard for static analysis
                raise AssertionError

        assert "invalid start byte" in error_message
        assert mock_generate.call_count == 2
        assert retry_patch_entries == ["entered"]

    def test_run_model_generation_does_not_retry_other_value_errors(self, test_image: Path) -> None:
        """Only the known upstream detokenizer failure should trigger a retry."""
        params = _build_params(test_image)
        fake_model = _FakeModel()
        fake_processor = object()
        retry_patch_entries: list[str] = []

        @contextlib.contextmanager
        def _record_retry_patch() -> Generator[None]:
            retry_patch_entries.append("entered")
            yield

        with (
            patch.object(check_models, "_ensure_generation_runtime_symbols"),
            patch.object(
                check_models,
                "_load_model",
                return_value=(fake_model, fake_processor, None),
            ),
            patch.object(check_models, "_run_model_preflight_validators"),
            patch.object(check_models, "apply_chat_template", return_value="formatted prompt"),
            patch.object(
                check_models,
                "_is_mlx_vlm_bpe_detokenizer_decode_failure",
                return_value=False,
            ) as mock_detector,
            patch.object(
                check_models,
                "_temporary_mlx_vlm_lossy_bpe_detokenizer_patch",
                _record_retry_patch,
            ),
            patch.object(
                check_models,
                "generate",
                side_effect=ValueError("bad config"),
            ) as mock_generate,
            patch.object(check_models, "mx", _FakeMxRuntime()),
        ):
            try:
                check_models._run_model_generation(params)
            except ValueError as err:
                error_message = str(err)
            else:  # pragma: no cover - defensive guard for static analysis
                raise AssertionError

        assert error_message == "Model generation failed for test/fake-model: bad config"
        assert mock_generate.call_count == 1
        assert mock_detector.call_count == 1
        assert retry_patch_entries == []

    def test_run_model_generation_backfills_peak_memory_from_mlx(self, test_image: Path) -> None:
        """MLX peak memory should backfill result objects that omit it."""
        params = _build_params(test_image)
        fake_model = _FakeModel()
        fake_processor = object()
        fake_generation = _FakeGenerationResult(peak_memory=0.0)

        class _FakeMxRuntimeWithPeak(_FakeMxRuntime):
            @staticmethod
            def get_peak_memory() -> float:
                return float(3 * (1024**3))

        with (
            patch.object(check_models, "_ensure_generation_runtime_symbols"),
            patch.object(
                check_models,
                "_load_model",
                return_value=(fake_model, fake_processor, None),
            ),
            patch.object(check_models, "_run_model_preflight_validators"),
            patch.object(check_models, "apply_chat_template", return_value="formatted prompt"),
            patch.object(check_models, "generate", return_value=fake_generation),
            patch.object(check_models, "mx", _FakeMxRuntimeWithPeak()),
        ):
            result = check_models._run_model_generation(params)

        assert result is fake_generation
        assert result.peak_memory == 3.0

    def test_log_perf_block_reads_cache_memory_field(self) -> None:
        """Compact memory logging should use the stored cache_memory field name."""
        result = check_models.PerformanceResult(
            model_name="test/fake-model",
            success=True,
            generation=_FakeGenerationResult(active_memory=0.5, cache_memory=0.3, peak_memory=1.2),
        )
        logged_values: list[tuple[str, str]] = []

        def _capture_tree(_prefix: str, label: str, value: str, *, indent: str = "") -> None:
            del indent
            logged_values.append((label, value))

        with (
            patch.object(check_models, "log_metric_label"),
            patch.object(check_models, "log_metric_tree", side_effect=_capture_tree),
        ):
            check_models._log_perf_block(result)

        assert ("Cache Δ:", " 0.30 GB") in logged_values

    def test_finalize_process_result_preserves_first_token_latency(self, test_image: Path) -> None:
        """Final cleanup should not discard previously derived first-token latency."""
        phase_timer = check_models.PhaseTimer()
        result_payload = check_models.PerformanceResult(
            model_name="test/fake-model",
            success=True,
            generation=_FakeGenerationResult(),
            runtime_diagnostics=check_models.RuntimeDiagnostics(
                input_validation_time_s=0.01,
                model_load_time_s=0.02,
                prompt_prep_time_s=0.03,
                decode_time_s=0.04,
                cleanup_time_s=0.05,
                first_token_latency_s=0.25,
                stop_reason="completed",
            ),
        )

        finalized = check_models._finalize_process_result(
            result_payload=result_payload,
            params=_build_params(test_image),
            phase_timer=phase_timer,
            stop_reason="completed",
            current_phase="cleanup",
            total_start_time=0.0,
        )

        assert finalized.runtime_diagnostics is not None
        assert finalized.runtime_diagnostics.first_token_latency_s == 0.25
