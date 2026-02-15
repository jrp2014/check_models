"""Mock-based tests for process_image_with_model."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from pathlib import Path

import check_models


@dataclass
class _FakeGenerationResult:
    """Minimal stand-in for mlx_vlm GenerationResult."""

    text: str = "Hello world"
    prompt_tokens: int = 50
    generation_tokens: int = 20
    generation_tps: float = 42.0
    peak_memory: float = 1.2
    time: float = 0.0
    active_memory: float = 0.5
    cache_memory: float = 0.3


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
