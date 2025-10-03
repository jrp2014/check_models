"""Tests for metrics alignment helper and mode selection output paths."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from check_models import (  # type: ignore[attr-defined]
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
    cached_memory: float | None
    time: float | None
    text: str | None

    def __init__(self) -> None:
        self.prompt_tokens = 10
        self.generation_tokens = 5
        self.generation_tps = 50.0
        self.peak_memory = 0.25
        self.active_memory = None
        self.cached_memory = None
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
    """The '=' characters for first metrics should be column-aligned."""
    parts = [
        "total=1.23s",
        "gen=1.00s",
        "peak_mem=5.5GB",
        "tokens(total/prompt/gen)=15/10/5",
    ]
    aligned = _align_metric_parts(parts)
    eq_positions = [p.index("=") for p in aligned[:3]]
    assert len(set(eq_positions)) == 1, aligned


def test_metrics_mode_compact_smoke(caplog: pytest.LogCaptureFixture) -> None:  # type: ignore[name-defined]
    """Compact mode should emit a single 'Metrics:' line."""
    caplog.set_level(logging.INFO)
    res = _build_perf()
    print_model_result(res, verbose=True, detailed_metrics=False)
    metrics_lines = [r.message for r in caplog.records if "Metrics:" in r.message]
    assert metrics_lines, "Expected compact Metrics line in logs"


def test_metrics_mode_detailed_smoke(caplog: pytest.LogCaptureFixture) -> None:  # type: ignore[name-defined]
    """Detailed mode should emit token lines plus Metrics header."""
    caplog.set_level(logging.INFO)
    res = _build_perf()
    print_model_result(res, verbose=True, detailed_metrics=True)
    metrics_lines = [r.message for r in caplog.records if "Metrics:" in r.message]
    token_lines = [r.message for r in caplog.records if "Tokens:" in r.message]
    assert token_lines, "Expected token summary lines in detailed mode"
    assert metrics_lines, "Expected Metrics header in detailed mode"
