"""Tests for metrics alignment helper and mode selection output paths."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from check_models import (
    PerformanceResult,
    _align_metric_parts,
    print_model_result,
)

if TYPE_CHECKING:  # pragma: no cover - only for type hints
    import pytest


class _StubGeneration:
    """Lightweight object matching attributes used by print_model_result."""

    prompt_tokens: int | None
    generation_tokens: int | None
    generation_tps: float | None
    peak_memory: float | None
    active_memory: float | None
    cache_memory: float | None
    time: float | None
    text: str | None

    def __init__(self) -> None:
        self.prompt_tokens = 10
        self.generation_tokens = 5
        self.generation_tps = 50.0
        self.peak_memory = 0.25
        self.active_memory = None
        self.cache_memory = None
        self.time = 1.0
        self.text = "hello"


def _build_perf() -> PerformanceResult:
    return PerformanceResult(
        model_name="dummy/model",
        generation=_StubGeneration(),
        success=True,
        generation_time=1.0,
        model_load_time=0.5,
        total_time=1.5,
    )


def test_align_metric_parts_alignment() -> None:
    """The metric parts should remain in compact key=value format."""
    parts = [
        "total=1.23s",
        "gen=1.00s",
        "peak_mem=5.5GB",
        "tokens(total/prompt/gen)=15/10/5",
    ]
    aligned = _align_metric_parts(parts)
    # Should return parts unchanged (no padding)
    assert aligned == parts, f"Expected {parts}, got {aligned}"
    # Verify all parts are in key=value format
    for part in aligned:
        assert "=" in part, f"Part {part!r} should contain '='"


def test_metrics_mode_compact_smoke(caplog: pytest.LogCaptureFixture) -> None:
    """Compact mode should emit Timing and Tokens lines."""
    caplog.set_level(logging.INFO)
    res = _build_perf()
    print_model_result(res, verbose=True, detailed_metrics=False)
    # New format uses "Timing:" (line 1) and "Tokens:" (line 2)
    timing_lines = [r.message for r in caplog.records if "Timing:" in r.message]
    assert timing_lines, "Expected Timing line in compact mode logs"


def test_metrics_mode_detailed_smoke(caplog: pytest.LogCaptureFixture) -> None:
    """Detailed mode should emit token lines plus Performance Metrics header."""
    caplog.set_level(logging.INFO)
    res = _build_perf()
    print_model_result(res, verbose=True, detailed_metrics=True)
    # Detailed mode uses "Performance Metrics:" header and separate "Tokens:" section
    perf_lines = [r.message for r in caplog.records if "Performance Metrics:" in r.message]
    token_lines = [r.message for r in caplog.records if "Tokens:" in r.message]
    assert token_lines, "Expected token summary lines in detailed mode"
    assert perf_lines, "Expected Performance Metrics header in detailed mode"
