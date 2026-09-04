"""Tests for overall runtime inclusion in generated reports.

This focuses on ensuring that the recently added overall runtime metric
appears in the retained HTML report without
executing full model runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from check_models import (
    PerformanceResult,
    RuntimeDiagnostics,
    generate_html_report,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pathlib import Path


class _StubGeneration:
    """Lightweight object matching SupportsGenerationResult attributes used in reporting.

    Attributes are annotated to satisfy the structural Protocol imported in the
    main module (`SupportsGenerationResult`). Optional-like fields use the
    same value semantics as real generation results (ints/floats or None).
    """

    prompt_tokens: int | None
    generation_tokens: int | None
    generation_tps: float | None
    peak_memory: float | None
    active_memory: float | None
    cache_memory: float | None
    time: float | None
    text: str | None

    def __init__(self) -> None:
        self.prompt_tokens = 5
        self.generation_tokens = 10
        self.generation_tps = 20.0
        self.peak_memory = 0.25
        self.active_memory = None
        self.cache_memory = None
        self.time = 1.23
        self.text = "Sample output"


def _build_single_result() -> PerformanceResult:
    """Return a synthetic successful PerformanceResult for testing."""
    return PerformanceResult(
        model_name="dummy/model",
        generation=_StubGeneration(),
        success=True,
        generation_time=0.50,
        model_load_time=0.40,
        total_time=0.90,
    )


def _build_result_with_runtime() -> PerformanceResult:
    """Return a synthetic successful PerformanceResult with detailed runtime metadata."""
    return PerformanceResult(
        model_name="dummy/model",
        generation=_StubGeneration(),
        success=True,
        generation_time=0.50,
        model_load_time=0.40,
        total_time=0.95,
        runtime_diagnostics=RuntimeDiagnostics(
            input_validation_time_s=0.05,
            model_load_time_s=0.40,
            prompt_prep_time_s=0.10,
            decode_time_s=0.50,
            cleanup_time_s=0.05,
            first_token_latency_s=0.20,
            stop_reason="completed",
        ),
    )


def test_html_report_includes_runtime(tmp_path: Path) -> None:
    """HTML report labels the pre-report sweep duration and formats it with an 's' suffix."""
    results = [_build_single_result()]
    html_file = tmp_path / "report.html"
    generate_html_report(
        results=results,
        filename=html_file,
        versions={"mlx": "0.0.0", "mlx-vlm": "0.0.0"},
        prompt="Test prompt",
        total_runtime_seconds=56.78,
    )
    content = html_file.read_text(encoding="utf-8")
    msg: str
    if "Model sweep runtime:" not in content:
        msg = "Missing model sweep runtime label in HTML report"
        raise AssertionError(msg)
    if "56.78s" not in content:
        msg = "Expected formatted runtime '56.78s' not found in HTML report"
        raise AssertionError(msg)


def test_html_report_includes_timing_snapshot(tmp_path: Path) -> None:
    """HTML report should surface aggregate timing inside the Runtime stanza."""
    results = [_build_result_with_runtime()]
    html_file = tmp_path / "report.html"
    generate_html_report(
        results=results,
        filename=html_file,
        versions={"mlx": "0.0.0", "mlx-vlm": "0.0.0"},
        prompt="Test prompt",
        total_runtime_seconds=56.78,
    )
    content = html_file.read_text(encoding="utf-8")
    assert "<p><b>Runtime</b></p>" in content
    assert "Validation overhead:" in content
    assert "Upstream model prefill / first-token time:" in content
    assert "Generation total:" in content


def test_html_long_runtime_reads_as_hours_minutes_seconds(tmp_path: Path) -> None:
    """Long runtimes read at a glance; no raw four-digit second counts."""
    results = [_build_single_result()]
    html_file = tmp_path / "long.html"
    long_seconds = 7_245.9  # 2h 0m 45.9s
    generate_html_report(
        results=results,
        filename=html_file,
        versions={"mlx": "0.0.0", "mlx-vlm": "0.0.0"},
        prompt="Test prompt",
        total_runtime_seconds=long_seconds,
    )
    content = html_file.read_text(encoding="utf-8")
    assert "2h 00m 45s" in content
    assert "7245.90" not in content
