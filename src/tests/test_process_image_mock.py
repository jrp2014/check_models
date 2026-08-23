"""Mock-based tests for process_image_with_model."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from transformers.processing_utils import ProcessorMixin

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable, Sequence

import check_models

OPEN_THINK_MARKER = "<think>"
CLOSE_THINK_MARKER = "</think>"


@dataclass
class _FakeGenerationResult:
    """Minimal stand-in for mlx_vlm GenerationResult."""

    text: str = "Hello world"
    token: object | None = None
    logprobs: object | None = None
    prompt_tokens: int = 50
    generation_tokens: int = 20
    total_tokens: int = 70
    prompt_tps: float = 100.0
    generation_tps: float = 42.0
    peak_memory: float = 1.2
    time: float = 0.0
    active_memory: float = 0.5
    cache_memory: float = 0.3
    model_load_active_memory: float | None = None
    finish_reason: str | None = None
    _check_models_prompt_diagnostics: check_models.PromptDiagnostics | None = None


class _FakeModel:
    config: object = object()

    @staticmethod
    def parameters() -> list[object]:
        return []


class _FakeProcessor:
    """Minimal processor satisfying mlx-vlm's typed generation protocol."""

    tokenizer: object = object()
    detokenizer: object = object()


class _FakeLegacyProcessor:
    """Tokenizer-like processor accepted by mlx-vlm's legacy generation path."""

    detokenizer: object = object()

    @staticmethod
    def decode(*_args: object, **_kwargs: object) -> str:
        return ""

    @staticmethod
    def batch_decode(*_args: object, **_kwargs: object) -> list[str]:
        return []


class _FakeStepProcessor(_FakeLegacyProcessor, ProcessorMixin):
    """Callable image processor without an ``image_processor`` attribute."""

    def __init__(self) -> None:
        """Skip ProcessorMixin construction; the preflight needs only interfaces."""

    def __call__(self, *_args: object, **_kwargs: object) -> dict[str, object]:
        return {}


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


class _RecordingMxRuntime:
    """MLX runtime stand-in that records synchronization and memory probes."""

    def __init__(self) -> None:
        self.sync_calls = 0
        self.eval_calls = 0
        self.active_calls = 0
        self.cache_calls = 0
        self.peak_calls = 0

    def synchronize(self) -> None:
        self.sync_calls += 1

    def get_active_memory(self) -> float:
        self.active_calls += 1
        return 0.0

    def get_cache_memory(self) -> float:
        self.cache_calls += 1
        return 0.0

    def get_peak_memory(self) -> float:
        self.peak_calls += 1
        return 0.0

    def eval(self, _params: object) -> None:
        self.eval_calls += 1


class _SequencedMxRuntime(_RecordingMxRuntime):
    """MLX runtime stand-in with deterministic active-memory samples."""

    def __init__(self, active_values: Sequence[float], *, cache_value: float = 0.0) -> None:
        super().__init__()
        self._active_values = tuple(active_values)
        self._cache_value = cache_value

    def get_active_memory(self) -> float:
        self.active_calls += 1
        if not self._active_values:
            return 0.0
        value_index = min(self.active_calls - 1, len(self._active_values) - 1)
        return self._active_values[value_index]

    def get_cache_memory(self) -> float:
        self.cache_calls += 1
        return self._cache_value


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
        kv_quant_scheme="uniform",
        kv_group_size=64,
        quantized_kv_start=check_models.DEFAULT_QUANTIZED_KV_START,
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

    def test_process_records_upstream_generation_entry(self, test_image: Path) -> None:
        """A successful upstream call should retain the deepest entered boundary."""
        params = _build_params(test_image)

        def _run_with_boundaries(
            *_args: object,
            phase_callback: Callable[[str], None],
            **_kwargs: object,
        ) -> _FakeGenerationResult:
            phase_callback("model_load")
            phase_callback("decode")
            return _FakeGenerationResult()

        with patch.object(
            check_models,
            "_run_model_generation",
            side_effect=_run_with_boundaries,
        ):
            result = check_models.process_image_with_model(params)

        assert result.upstream_boundary == "generation_started"
        assert result.generation is not None
        assert result.generation.text == "Hello world"

    def test_process_records_upstream_load_failure(self, test_image: Path) -> None:
        """A load exception should retain that upstream loading was entered."""
        params = _build_params(test_image)
        load_error = "loader raised"

        def _fail_during_load(
            *_args: object,
            phase_callback: Callable[[str], None],
            **_kwargs: object,
        ) -> _FakeGenerationResult:
            phase_callback("model_load")
            raise ValueError(load_error)

        with patch.object(
            check_models,
            "_run_model_generation",
            side_effect=_fail_during_load,
        ):
            result = check_models.process_image_with_model(params)

        assert result.upstream_boundary == "load_started"

    def test_process_records_upstream_generation_failure(self, test_image: Path) -> None:
        """A decode exception should retain that upstream generation was entered."""
        params = _build_params(test_image)
        generation_error = "generator raised"

        def _fail_during_generation(
            *_args: object,
            phase_callback: Callable[[str], None],
            **_kwargs: object,
        ) -> _FakeGenerationResult:
            phase_callback("model_load")
            phase_callback("decode")
            raise ValueError(generation_error)

        with patch.object(
            check_models,
            "_run_model_generation",
            side_effect=_fail_during_generation,
        ):
            result = check_models.process_image_with_model(params)

        assert result.upstream_boundary == "generation_started"

    def test_extract_generation_performance_data_uses_generation_result(self) -> None:
        """Performance snapshots should prefer upstream GenerationResult metrics."""
        fake_result = _FakeGenerationResult(
            prompt_tokens=11,
            generation_tokens=7,
            total_tokens=18,
            prompt_tps=22.0,
            generation_tps=33.0,
            peak_memory=4.5,
            time=1.25,
            active_memory=0.75,
            cache_memory=0.25,
        )

        metrics = check_models._extract_generation_performance_data(fake_result)

        assert metrics.prompt_tokens == 11
        assert metrics.generation_tokens == 7
        assert metrics.total_tokens == 18
        assert metrics.prompt_tps == 22.0
        assert metrics.generation_tps == 33.0
        assert metrics.peak_memory_gb == 4.5
        assert metrics.generation_time_s == 1.25
        assert metrics.active_memory_gb == 0.75
        assert metrics.cache_memory_gb == 0.25
        assert metrics.first_token_latency_s == 0.5

    def test_load_model_forwards_upstream_load_flags(self, test_image: Path) -> None:
        """Image-relevant mlx-vlm load flags should reach mlx_vlm.utils.load."""
        params = replace(
            _build_params(test_image),
            force_download=True,
            quantize_activations=True,
        )
        fake_model = _FakeModel()
        fake_processor = _FakeProcessor()

        with patch.object(
            check_models,
            "load",
            return_value=(fake_model, fake_processor),
        ) as mock_load:
            model, processor, config = check_models._load_model(params)

        assert model is fake_model
        assert processor is fake_processor
        assert config is fake_model.config
        assert mock_load.call_args.kwargs["force_download"] is True
        assert mock_load.call_args.kwargs["quantize_activations"] is True

    def test_load_model_retries_connectivity_failure_from_local_snapshot(
        self,
        test_image: Path,
        tmp_path: Path,
    ) -> None:
        """A transient Hub read must not block an already-cached model."""
        params = _build_params(test_image)
        snapshot = tmp_path / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)
        fake_model = _FakeModel()
        fake_processor = _FakeProcessor()

        with (
            patch.object(
                check_models,
                "load",
                side_effect=[OSError("connection reset by peer"), (fake_model, fake_processor)],
            ) as mock_load,
            patch.object(
                check_models,
                "_resolve_model_snapshot_path",
                return_value=snapshot,
            ),
        ):
            model, processor, config = check_models._load_model(params)

        assert model is fake_model
        assert processor is fake_processor
        assert config is fake_model.config
        assert [call.kwargs["path_or_hf_repo"] for call in mock_load.call_args_list] == [
            params.model_identifier,
            str(snapshot),
        ]

    @pytest.mark.parametrize(
        ("load_error", "revision", "force_download"),
        [
            (ValueError("unsupported model config"), None, False),
            (OSError("connection reset by peer"), "different-revision", False),
            (OSError("connection reset by peer"), None, True),
        ],
    )
    def test_load_model_does_not_retry_unsafe_local_fallbacks(
        self,
        test_image: Path,
        tmp_path: Path,
        load_error: Exception,
        revision: str | None,
        force_download: bool,
    ) -> None:
        """Only unpinned ordinary connectivity failures may use cached state."""
        params = replace(
            _build_params(test_image),
            revision=revision,
            force_download=force_download,
        )
        snapshot = tmp_path / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)

        with (
            patch.object(check_models, "load", side_effect=load_error) as mock_load,
            patch.object(
                check_models,
                "_resolve_model_snapshot_path",
                return_value=snapshot,
            ),
            pytest.raises(type(load_error), match=str(load_error)),
        ):
            check_models._load_model(params)

        assert mock_load.call_count == 1

    def test_preflight_accepts_custom_image_processor_without_attribute(self) -> None:
        """Native-compatible custom processors need not expose HF internals."""
        check_models._run_model_preflight_validators(
            model_identifier="org/step",
            processor=_FakeStepProcessor(),
            config={"model_type": "step3p7"},
        )

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

    def test_failure_capture_omits_self_logged_rich_traceback(
        self,
        test_image: Path,
    ) -> None:
        """Captured stderr should keep tool output but drop our own Rich traceback block."""
        params = _build_params(test_image)

        def _raise_after_rich_log_output(
            *_args: object, **_kwargs: object
        ) -> _FakeGenerationResult:
            sys.stderr.write("Downloading (incomplete total...): 0.00B [00:00, ?B/s]\n")
            sys.stderr.write("[19:36:43] ERROR    Failed to load model org/broken\n")
            sys.stderr.write("╭──────── Traceback (most recent call last) ────────╮\n")
            sys.stderr.write("│ /repo/src/check_models.py:18883 in _run_model_generation │\n")
            sys.stderr.write("╰──────────────────────────────────────────────────╯\n")
            sys.stderr.write("ValueError: Missing 2 parameters\n")
            error_message = "bad config"
            raise ValueError(error_message)

        with patch.object(
            check_models,
            "_run_model_generation",
            side_effect=_raise_after_rich_log_output,
        ):
            result = check_models.process_image_with_model(params)

        assert result.success is False
        captured_output = result.captured_output_on_fail or ""
        assert "Downloading (incomplete total" in captured_output
        assert "Failed to load model org/broken" not in captured_output
        assert "Traceback (most recent call last)" not in captured_output
        assert "_run_model_generation" not in captured_output

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

    def test_build_failure_result_preserves_root_exception_identity(self) -> None:
        """Wrapped upstream errors should keep their original exception identity."""
        upstream_message = "upstream shape mismatch"
        wrapper_message = "Model loading failed: upstream shape mismatch"
        upstream_error = RuntimeError(upstream_message)
        wrapper_error = ValueError(wrapper_message)
        wrapper_error.__cause__ = upstream_error

        def _raise_wrapped_error(error: ValueError) -> None:
            raise error

        result: check_models.PerformanceResult
        try:
            _raise_wrapped_error(wrapper_error)
        except ValueError as err:
            result = check_models._build_failure_result(
                model_name="test/fake-model",
                error=err,
                captured_output=None,
            )
        else:  # pragma: no cover - defensive guard for static analysis
            raise AssertionError

        assert result.error_type == "ValueError"
        assert result.root_error_type == "RuntimeError"
        assert result.root_error_module == "builtins"
        assert result.root_error_message == upstream_message

    def test_build_failure_result_preserves_exception_chain_order(self) -> None:
        """Failure narratives should preserve root-to-wrapper exception chronology."""
        index_error = IndexError("token id 999 outside detokenizer table")
        runtime_error = RuntimeError("METAL command buffer out of memory")
        runtime_error.__cause__ = index_error
        wrapped = ValueError("generation failed")
        wrapped.__cause__ = runtime_error

        try:
            raise wrapped
        except ValueError as error:
            result = check_models._build_failure_result(
                model_name="org/model",
                error=error,
                captured_output=None,
            )

        assert [entry.exception_type for entry in result.exception_chain] == [
            "IndexError",
            "RuntimeError",
            "ValueError",
        ]
        assert [entry.message for entry in result.exception_chain] == [
            "token id 999 outside detokenizer table",
            "METAL command buffer out of memory",
            "generation failed",
        ]

    def test_build_failure_result_reuses_one_root_selection_traversal(self) -> None:
        """Failure building should reuse the canonical root selector over one chain walk."""
        root_error = IndexError("bad token")
        wrapper_error = ValueError("generation failed")
        wrapper_error.__cause__ = root_error

        with (
            patch.object(
                check_models,
                "_exception_chain",
                wraps=check_models._exception_chain,
            ) as chain_walk,
            patch.object(
                check_models,
                "_root_cause_exception",
                wraps=check_models._root_cause_exception,
            ) as root_selector,
        ):
            try:
                raise wrapper_error
            except ValueError as error:
                result = check_models._build_failure_result(
                    model_name="org/model",
                    error=error,
                    captured_output=None,
                )

        root_selector.assert_called_once()
        assert chain_walk.call_count == 1
        assert result.root_error_type == "IndexError"
        assert result.root_error_message == "bad token"

    def test_failure_result_preserves_mixed_runtime_exception_modules(self) -> None:
        """A mixed runtime chain retains exact modules without an owner projection."""
        runtime_error = RuntimeError("kIOGPUCommandBufferCallbackErrorOutOfMemory")
        wrapper_error = ValueError("mlx_vlm/generate.py generation failed")
        wrapper_error.__cause__ = runtime_error

        try:
            raise wrapper_error
        except ValueError as error:
            result = check_models._build_failure_result(
                model_name="org/model",
                error=error,
                captured_output=None,
            )

        assert [entry.module for entry in result.exception_chain] == [
            "builtins",
            "builtins",
        ]

    def test_build_failure_result_preserves_quality_fields(self) -> None:
        """Failure builder should carry precomputed quality diagnostics when provided."""
        repeated_phrase = "loop"
        decode_error_message = "decode failed"

        def _raise_decode_failed() -> None:
            raise ValueError(decode_error_message)

        analysis = check_models.GenerationQualityAnalysis(
            is_repetitive=True,
            repeated_token=repeated_phrase,
            word_count=25,
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
        fake_processor = _FakeProcessor()
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
        prompt_diagnostics = result._check_models_prompt_diagnostics
        assert prompt_diagnostics is not None
        assert prompt_diagnostics.processed_image_width == 384
        assert prompt_diagnostics.processed_image_height == 512
        assert prompt_diagnostics.image_patch_count is None

    def test_run_model_generation_accepts_legacy_tokenizer_processor(
        self,
        test_image: Path,
    ) -> None:
        """Tokenizer-like legacy processors should reach upstream generation unchanged."""
        params = _build_params(test_image)
        fake_model = _FakeModel()
        fake_processor = _FakeLegacyProcessor()
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
        assert mock_generate.call_args.kwargs["processor"] is fake_processor

    def test_run_model_generation_passes_thinking_kwargs(self, test_image: Path) -> None:
        """Thinking-mode flags should reach both chat templating and generation."""
        params = replace(
            _build_params(test_image),
            eos_tokens=(CLOSE_THINK_MARKER,),
            skip_special_tokens=True,
            enable_thinking=True,
            thinking_budget=96,
            thinking_start_token=OPEN_THINK_MARKER,
            thinking_end_token=CLOSE_THINK_MARKER,
        )

        fake_model = _FakeModel()
        fake_processor = _FakeProcessor()
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
        assert mock_template.call_args.kwargs["thinking_budget"] == 96
        assert mock_template.call_args.kwargs["thinking_start_token"] == "<think>"
        assert mock_template.call_args.kwargs["thinking_end_token"] == "</think>"
        generate_kwargs = mock_generate.call_args.kwargs
        assert generate_kwargs["eos_tokens"] == ["</think>"]
        assert generate_kwargs["skip_special_tokens"] is True
        assert generate_kwargs["enable_thinking"] is True
        assert generate_kwargs["thinking_budget"] == 96
        assert generate_kwargs["thinking_start_token"] == "<think>"
        assert generate_kwargs["thinking_end_token"] == "</think>"

    def test_run_model_generation_passes_server_shared_request_kwargs(
        self,
        test_image: Path,
    ) -> None:
        """Server request controls shared with generate() should reach mlx_vlm.generate."""
        params = replace(
            _build_params(test_image),
            seed=0,
            presence_penalty=0.25,
            presence_context_size=32,
            frequency_penalty=0.5,
            frequency_context_size=64,
            logit_bias={42: -1.5, 123: 2.0},
        )

        fake_model = _FakeModel()
        fake_processor = _FakeProcessor()
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
        assert generate_kwargs["seed"] == 0
        assert generate_kwargs["presence_penalty"] == 0.25
        assert generate_kwargs["presence_context_size"] == 32
        assert generate_kwargs["frequency_penalty"] == 0.5
        assert generate_kwargs["frequency_context_size"] == 64
        assert generate_kwargs["logit_bias"] == {42: -1.5, 123: 2.0}
        prompt_diagnostics = result._check_models_prompt_diagnostics
        assert prompt_diagnostics is not None
        assert prompt_diagnostics.generate_kwargs["seed"] == 0
        assert prompt_diagnostics.generate_kwargs["presence_penalty"] == 0.25
        assert prompt_diagnostics.generate_kwargs["presence_context_size"] == 32
        assert prompt_diagnostics.generate_kwargs["frequency_penalty"] == 0.5
        assert prompt_diagnostics.generate_kwargs["frequency_context_size"] == 64
        assert prompt_diagnostics.generate_kwargs["logit_bias"] == {"42": -1.5, "123": 2.0}
        assert "verbose" not in prompt_diagnostics.generate_kwargs

    def test_run_model_generation_fails_fast_on_generation_errors(
        self,
        test_image: Path,
    ) -> None:
        """Generation errors surface once with decode-phase tagging (no retry loop)."""
        params = _build_params(test_image)
        fake_model = _FakeModel()
        fake_processor = _FakeProcessor()

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
                "generate",
                side_effect=ValueError("bad config"),
            ) as mock_generate,
            patch.object(check_models, "mx", _FakeMxRuntime()),
            pytest.raises(ValueError, match="Model generation failed for test/fake-model"),
        ):
            check_models._run_model_generation(params)

        assert mock_generate.call_count == 1

    def test_run_model_generation_samples_memory_without_local_peak_probe(
        self,
        test_image: Path,
    ) -> None:
        """Generation should sample active/cache memory without a local peak probe."""
        params = _build_params(test_image)
        fake_model = _FakeModel()
        fake_processor = _FakeProcessor()
        fake_generation = _FakeGenerationResult()
        runtime = _RecordingMxRuntime()

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
            patch.object(check_models, "mx", runtime),
        ):
            result = check_models._run_model_generation(params)

        assert result is fake_generation
        assert runtime.sync_calls == 1
        assert runtime.active_calls == 2
        assert runtime.cache_calls == 1
        assert runtime.peak_calls == 0
        assert runtime.eval_calls == 0

    def test_run_model_generation_records_post_load_active_memory_baseline(
        self,
        test_image: Path,
    ) -> None:
        """Generation should retain a post-load memory baseline for image-density metrics."""
        params = _build_params(test_image)
        fake_model = _FakeModel()
        fake_processor = _FakeProcessor()
        fake_generation = _FakeGenerationResult()
        runtime = _SequencedMxRuntime(
            active_values=(
                2.0 * check_models.DECIMAL_GB,
                4.0 * check_models.DECIMAL_GB,
            ),
            cache_value=3.0 * check_models.DECIMAL_GB,
        )

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
            patch.object(check_models, "mx", runtime),
        ):
            result = check_models._run_model_generation(params)

        assert result is fake_generation
        assert getattr(result, "model_load_active_memory", None) == 2.0
        assert result.active_memory == 4.0
        assert result.cache_memory == 3.0
        assert runtime.active_calls == 2

    @pytest.mark.parametrize(
        ("finish_reason", "generation_tokens", "expected"),
        [
            ("stop", 50, "completed"),
            ("length", 20, "max_tokens"),
            (None, 50, "max_tokens"),
        ],
    )
    def test_process_image_with_model_prefers_upstream_finish_reason(
        self,
        test_image: Path,
        finish_reason: str | None,
        generation_tokens: int,
        expected: str,
    ) -> None:
        """Explicit upstream termination should win over the token-count fallback."""
        params = _build_params(test_image)
        fake_result = _FakeGenerationResult(
            generation_tokens=generation_tokens,
            finish_reason=finish_reason,
        )

        with patch.object(check_models, "_run_model_generation", return_value=fake_result):
            result = check_models.process_image_with_model(params)

        assert result.runtime_diagnostics is not None
        assert result.runtime_diagnostics.stop_reason == expected

    def test_process_image_with_model_skips_cleanup_sync_after_success(
        self,
        test_image: Path,
    ) -> None:
        """Cleanup should skip a second synchronize after a successful synced generation."""
        params = _build_params(test_image)
        cleanup_flags: list[bool] = []

        def _record_cleanup(*, synchronize_first: bool = True) -> None:
            cleanup_flags.append(synchronize_first)

        with (
            patch.object(
                check_models,
                "_run_model_generation",
                return_value=_FakeGenerationResult(),
            ),
            patch.object(check_models, "_cleanup_runtime_resources", side_effect=_record_cleanup),
        ):
            result = check_models.process_image_with_model(params)

        assert result.success is True
        assert cleanup_flags == [False]

    def test_process_image_with_model_keeps_cleanup_sync_on_failure(
        self,
        test_image: Path,
    ) -> None:
        """Cleanup should still synchronize when generation fails before the success barrier."""
        params = _build_params(test_image)
        cleanup_flags: list[bool] = []

        def _record_cleanup(*, synchronize_first: bool = True) -> None:
            cleanup_flags.append(synchronize_first)

        with (
            patch.object(
                check_models,
                "_run_model_generation",
                side_effect=ValueError("bad config"),
            ),
            patch.object(check_models, "_cleanup_runtime_resources", side_effect=_record_cleanup),
        ):
            result = check_models.process_image_with_model(params)

        assert result.success is False
        assert cleanup_flags == [True]

    @pytest.mark.parametrize("generation_fails", [False, True])
    def test_process_image_with_model_records_post_cleanup_memory_after_every_attempt(
        self,
        test_image: Path,
        *,
        generation_fails: bool,
    ) -> None:
        """Successful and crashed attempts should expose allocator state after cleanup."""
        params = _build_params(test_image)
        runtime = _SequencedMxRuntime(
            active_values=(0.125 * check_models.DECIMAL_GB,),
            cache_value=0.25 * check_models.DECIMAL_GB,
        )
        run_outcome: _FakeGenerationResult | ValueError = (
            ValueError("bad config") if generation_fails else _FakeGenerationResult()
        )

        with (
            patch.object(check_models, "mx", runtime),
            patch.object(
                check_models,
                "_run_model_generation",
                side_effect=run_outcome if generation_fails else None,
                return_value=None if generation_fails else run_outcome,
            ),
        ):
            result = check_models.process_image_with_model(params)

        assert result.runtime_diagnostics is not None
        assert result.runtime_diagnostics.post_cleanup_active_memory_gb == 0.125
        assert result.runtime_diagnostics.post_cleanup_cache_memory_gb == 0.25

    def test_process_image_logs_teed_console_output_to_file_only(
        self,
        test_image: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Live mlx-vlm stdout should land in the file log without a second console echo."""
        params = _build_params(test_image)
        live_block = (
            "==========\n"
            "Prompt: <image>Describe this image\n"
            "A top-down view of two cats laying on a pink blanket.\n"
            "==========\n"
            "Prompt: 1029 tokens, 915.573 tokens-per-sec\n"
            "Generation: 159 tokens, 5.341 tokens-per-sec\n"
            "Peak memory: 25.836 GB\n"
        )

        def _run_and_print(*_args: object, **_kwargs: object) -> _FakeGenerationResult:
            sys.stdout.write(live_block)
            return _FakeGenerationResult(
                text="A top-down view of two cats laying on a pink blanket."
            )

        with (
            patch.object(check_models, "_run_model_generation", side_effect=_run_and_print),
            caplog.at_level(logging.DEBUG, logger=check_models.LOGGER_NAME),
        ):
            result = check_models.process_image_with_model(params)

        assert result.success is True
        capture_records = [
            record
            for record in caplog.records
            if "Captured mlx-vlm console output for" in record.getMessage()
        ]
        assert len(capture_records) == 1
        message = capture_records[0].getMessage()
        assert "two cats laying on a pink blanket" in message
        assert "Peak memory: 25.836 GB" in message
        assert getattr(capture_records[0], "log_destination", None) == "file"

    def test_compose_stream_capture_for_file_log_bounds_size(self) -> None:
        """Pathological tee buffers should truncate rather than inflate the log unboundedly."""
        huge = "x" * (check_models.MAX_FILE_STREAM_CAPTURE_CHARS + 50)
        body = check_models._compose_stream_capture_for_file_log(
            stdout_text=huge,
            stderr_text="",
        )
        assert body is not None
        assert "truncated 50 characters for file-log size bound" in body
        assert len(body) < len(huge) + 80

    def test_log_perf_block_reads_cache_memory_field(self) -> None:
        """Compact memory logging should use the stored cache_memory field name."""
        result = check_models.PerformanceResult(
            model_name="test/fake-model",
            success=True,
            generation=_FakeGenerationResult(active_memory=0.5, cache_memory=0.3, peak_memory=1.2),
        )
        logged_values: list[tuple[str, str]] = []

        def _capture_tree(
            _title: str,
            rows: Sequence[tuple[str, str]],
            *,
            emoji: str = "",
            indent: str = "",
        ) -> None:
            del emoji, indent
            logged_values.extend(rows)

        with patch.object(check_models, "_log_metric_tree", side_effect=_capture_tree):
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


def test_sent_generate_keywords_mirror_the_kwargs_builders(test_image: Path) -> None:
    """The drift-detector contract must equal what the builders actually send."""
    params = replace(
        _build_params(test_image),
        min_p=0.05,
        top_k=40,
        prefill_step_size=2048,
        resize_shape=(448, 448),
        eos_tokens=("<eos>",),
        skip_special_tokens=True,
        enable_thinking=True,
        thinking_budget=128,
        thinking_start_token="<think>",  # noqa: S106 - thinking delimiter, not a credential
        thinking_end_token="</think>",  # noqa: S106 - thinking delimiter, not a credential
        kv_key_bits=8.0,
        kv_value_bits=3.0,
        kv_key_scheme="uniform",
        kv_value_scheme="turboquant",
    )

    sent = check_models._build_generate_kwargs(
        params,
        check_models._build_generate_extra_kwargs(params),
    )

    assert set(sent) == set(check_models._SENT_GENERATE_KEYWORDS)


class TestPerModelIsolationBoundary:
    """process_image_with_model must isolate any per-model exception."""

    def test_unexpected_exception_recorded_as_model_failure(self, test_image: Path) -> None:
        """A TypeError raised inside generation becomes a failure result, not an abort."""
        params = _build_params(test_image)

        def _raise_type_error(*_args: object, **_kwargs: object) -> None:
            msg = "mlx-vlm load returned no generation-compatible processor or tokenizer"
            raise TypeError(msg)

        with patch.object(check_models, "_run_model_generation", side_effect=_raise_type_error):
            result = check_models.process_image_with_model(params)

        assert result.success is False
        assert result.error_type == "TypeError"
        assert "generation-compatible processor" in (result.error_message or "")
        assert result.error_traceback is not None
        assert result.failure_phase is not None

    def test_arbitrary_exception_class_is_isolated(self, test_image: Path) -> None:
        """Even a custom exception type from a processor is recorded, not propagated."""
        params = _build_params(test_image)

        class ProcessorQuirkError(Exception):
            pass

        def _raise_custom(*_args: object, **_kwargs: object) -> None:
            msg = "unexpected processor state"
            raise ProcessorQuirkError(msg)

        with patch.object(check_models, "_run_model_generation", side_effect=_raise_custom):
            result = check_models.process_image_with_model(params)

        assert result.success is False
        assert result.error_type == "ProcessorQuirkError"

    def test_keyboard_interrupt_propagates(self, test_image: Path) -> None:
        """Operator interrupts must escape the boundary and stop the sweep."""
        params = _build_params(test_image)

        with (
            patch.object(check_models, "_run_model_generation", side_effect=KeyboardInterrupt),
            pytest.raises(KeyboardInterrupt),
        ):
            check_models.process_image_with_model(params)

    def test_system_exit_propagates(self, test_image: Path) -> None:
        """SystemExit must escape the boundary and stop the sweep."""
        params = _build_params(test_image)

        with (
            patch.object(check_models, "_run_model_generation", side_effect=SystemExit(3)),
            pytest.raises(SystemExit),
        ):
            check_models.process_image_with_model(params)


# ── Isolated execution (one child interpreter per model) ─────────────────────


class TestIsolatedExecution:
    """The parent must survive a native child crash and round-trip full results."""

    @staticmethod
    def _args(test_image: Path) -> argparse.Namespace:
        parser = check_models._build_cli_parser()
        return parser.parse_args(["--image", str(test_image), "--isolate", "--max-tokens", "5"])

    def test_namespace_json_round_trip_preserves_paths(self, test_image: Path) -> None:
        """Path-valued CLI args survive the JSON hand-off to the child."""
        args = self._args(test_image)
        restored = check_models._namespace_from_json(check_models._namespace_to_json(args))
        assert restored.image == test_image
        assert isinstance(restored.output_jsonl, Path)
        assert restored.max_tokens == 5
        assert restored.isolate is True

    def test_performance_result_json_round_trip(self) -> None:
        """Nested dataclasses and a duck-typed generation survive the round trip."""
        diagnostics = check_models.PromptDiagnostics(
            model_type="fake",
            rendered_prompt="p",
            rendered_prompt_token_count=7,
            special_token_ids=(1, 2),
            snapshot_notes=("note",),
            generate_kwargs={"max_tokens": 5},
        )
        result = check_models.PerformanceResult(
            model_name="org/m",
            success=True,
            generation=_FakeGenerationResult(text="hello", prompt_tokens=50, generation_tps=42.0),
            generation_time=1.5,
            prompt_diagnostics=diagnostics,
            exception_chain=(
                check_models.FailureException(
                    exception_type="ValueError", module="builtins", message="x"
                ),
            ),
            runtime_diagnostics=check_models.RuntimeDiagnostics(stop_reason="completed"),
            completed_at="2026-08-23 12:00:00 BST",
        )
        payload = json.loads(json.dumps(check_models._performance_result_to_json(result)))
        restored = check_models._performance_result_from_json(payload)

        assert restored.model_name == "org/m"
        assert restored.success is True
        assert restored.generation is not None
        assert restored.generation.text == "hello"
        assert getattr(restored.generation, "prompt_tokens", None) == 50
        assert getattr(restored.generation, "generation_tps", None) == 42.0
        assert getattr(restored.generation, "missing_field", None) is None
        assert check_models._generation_int_metric(restored.generation, "prompt_tokens") == 50
        assert restored.prompt_diagnostics == diagnostics
        assert restored.exception_chain == result.exception_chain
        assert restored.runtime_diagnostics == result.runtime_diagnostics
        assert restored.generation_time == 1.5

    def test_signal_names(self) -> None:
        """Negative and 128+N return codes map to signal names."""
        assert check_models._signal_name_for_returncode(-6) == "SIGABRT"
        assert check_models._signal_name_for_returncode(134) == "SIGABRT"
        assert check_models._signal_name_for_returncode(-11) == "SIGSEGV"
        assert check_models._signal_name_for_returncode(1) is None
        assert check_models._signal_name_for_returncode(0) is None

    def test_crashed_child_becomes_phase_tagged_failure(self, test_image: Path) -> None:
        """A child dying natively is recorded with the phase it reached, not fatal."""
        args = self._args(test_image)

        def _fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            worker_dir = Path(command[-1]).parent
            check_models._write_text_file(worker_dir / "phase.txt", "decode")
            return subprocess.CompletedProcess(command, -11)

        with patch.object(check_models.subprocess, "run", side_effect=_fake_run):
            result = check_models._run_model_isolated(
                args, model_identifier="org/crasher", image_path=test_image, prompt="p"
            )

        assert result.success is False
        assert result.failure_phase == "decode"
        assert result.error_type == "IsolatedWorkerCrashError"
        assert result.error_message is not None
        assert "SIGSEGV" in result.error_message
        assert result.upstream_boundary != "not_started"

    def test_successful_child_result_is_returned(self, test_image: Path) -> None:
        """A child that writes a result has it returned verbatim."""
        args = self._args(test_image)
        child_result = check_models.PerformanceResult(
            model_name="org/ok",
            success=True,
            generation=_FakeGenerationResult(text="child says hi"),
            generation_time=0.2,
        )

        def _fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            worker_dir = Path(command[-1]).parent
            check_models._write_text_file(
                worker_dir / "result.json",
                json.dumps(check_models._performance_result_to_json(child_result)),
            )
            return subprocess.CompletedProcess(command, 0)

        with patch.object(check_models.subprocess, "run", side_effect=_fake_run):
            result = check_models._run_model_isolated(
                args, model_identifier="org/ok", image_path=test_image, prompt="p"
            )
        assert result.success is True
        assert result.generation is not None
        assert result.generation.text == "child says hi"
        assert result.generation_time == 0.2

    def test_timed_out_child_is_terminated_and_classified(self, test_image: Path) -> None:
        """A child that exceeds the model timeout plus grace is a phase-tagged timeout."""
        args = self._args(test_image)

        def _fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            worker_dir = Path(command[-1]).parent
            check_models._write_text_file(worker_dir / "phase.txt", "model_load")
            deadline = kwargs.get("timeout")
            assert isinstance(deadline, float)
            raise subprocess.TimeoutExpired(command, deadline)

        with patch.object(check_models.subprocess, "run", side_effect=_fake_run):
            result = check_models._run_model_isolated(
                args, model_identifier="org/hang", image_path=test_image, prompt="p"
            )
        assert result.success is False
        assert result.failure_phase == "model_load"
        assert result.error_type == "IsolatedWorkerTimeoutError"
        assert result.error_message is not None
        assert "exceeded" in result.error_message

    def test_reruns_use_the_selected_execution_boundary(self, test_image: Path) -> None:
        """--rerun-triage must not bypass --isolate."""
        args = self._args(test_image)
        args.rerun_triage = True
        first = check_models.PerformanceResult(model_name="org/m", success=False, generation=None)
        seen: list[dict[str, object]] = []

        def _fake_isolated(_args: object, **kwargs: object) -> check_models.PerformanceResult:
            seen.append(kwargs)
            return check_models.PerformanceResult(
                model_name="org/m", success=True, generation=_FakeGenerationResult(text="rerun ok")
            )

        with (
            patch.object(check_models, "_run_model_isolated", side_effect=_fake_isolated),
            patch.object(
                check_models,
                "process_image_with_model",
                side_effect=AssertionError("bypassed isolation"),
            ),
        ):
            updated = check_models._run_differential_reruns([first], args, test_image)
        assert seen
        assert seen[0]["prompt"] == check_models.TRIAGE_PROMPT
        assert seen[0]["max_tokens"] == check_models.RERUN_TRIAGE_MAX_TOKENS
        assert updated[0].rerun_evidence is not None
        assert updated[0].rerun_evidence.rerun_success is True

    def test_dynamic_generation_attributes_survive_isolation(self) -> None:
        """Metrics check_models attaches to the upstream object are not declared fields."""

        @dataclass
        class _UpstreamLike:
            text: str = "hi"
            prompt_tokens: int = 3

        generation = _UpstreamLike()
        # Attached the way check_models attaches runtime metrics: dynamically.
        setattr(generation, "active_memory", 0.5)  # noqa: B010 - deliberately dynamic
        setattr(generation, "cache_memory", 0.25)  # noqa: B010 - deliberately dynamic
        result = check_models.PerformanceResult(
            model_name="org/m", success=True, generation=generation
        )
        restored = check_models._performance_result_from_json(
            json.loads(json.dumps(check_models._performance_result_to_json(result)))
        )
        assert getattr(restored.generation, "active_memory", None) == 0.5
        assert getattr(restored.generation, "cache_memory", None) == 0.25
        assert getattr(restored.generation, "prompt_tokens", None) == 3

    def test_isolated_download_timeout_is_environmental(self, test_image: Path) -> None:
        """A worker deadline that expired mid-download stays indeterminate, not actionable."""
        args = self._args(test_image)

        def _fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            worker_dir = Path(command[-1]).parent
            check_models._write_text_file(worker_dir / "phase.txt", "model_load")
            check_models._write_text_file(
                worker_dir / "stderr.txt",
                "model-00001-of-00005.safetensors:  37%|███  | 7.4G/20.1G [04:02<07:12, 29.3MB/s]",
            )
            deadline = kwargs.get("timeout")
            assert isinstance(deadline, float)
            raise subprocess.TimeoutExpired(command, deadline)

        with patch.object(check_models.subprocess, "run", side_effect=_fake_run):
            result = check_models._run_model_isolated(
                args, model_identifier="org/coldstart", image_path=test_image, prompt="p"
            )
        assert result.error_type == "IsolatedWorkerTimeoutError"
        assert check_models._is_download_timeout_failure(result) is True
        assessment = check_models._assess_result(result)
        assert assessment.execution == "indeterminate"
        assert assessment.maintainer_status != "actionable_failure"
