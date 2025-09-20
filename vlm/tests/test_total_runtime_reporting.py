"""Tests for overall runtime inclusion in generated reports.

This focuses on ensuring that the recently added overall runtime metric
appears in CLI-related report outputs (Markdown & HTML builders) without
executing full model runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vlm.check_models import PerformanceResult, generate_html_report, generate_markdown_report

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
    cached_memory: float | None
    time: float | None
    text: str | None

    def __init__(self) -> None:
        self.prompt_tokens = 5
        self.generation_tokens = 10
        self.generation_tps = 20.0
        self.peak_memory = 0.25
        self.active_memory = None
        self.cached_memory = None
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


def test_markdown_report_includes_runtime(tmp_path: Path) -> None:  # type: ignore[name-defined]
    """Markdown report should contain an overall runtime line with seconds."""
    results = [_build_single_result()]
    md_file = tmp_path / "report.md"
    generate_markdown_report(
        results=results,
        filename=md_file,
        versions={"mlx": "0.0.0", "mlx-vlm": "0.0.0"},
        prompt="Test prompt",
        total_runtime_seconds=12.34,
    )
    content = md_file.read_text(encoding="utf-8")
    msg: str
    if "Overall runtime:" not in content:
        msg = "Missing overall runtime label in markdown report"
        raise AssertionError(msg)
    if "12.34" not in content:
        msg = "Expected formatted runtime '12.34' not found in markdown report"
        raise AssertionError(msg)


def test_html_report_includes_runtime(tmp_path: Path) -> None:  # type: ignore[name-defined]
    """HTML report should include the formatted overall runtime string with 's' suffix."""
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
    if "Overall runtime:" not in content:
        msg = "Missing overall runtime label in HTML report"
        raise AssertionError(msg)
    if "56.78s" not in content:
        msg = "Expected formatted runtime '56.78s' not found in HTML report"
        raise AssertionError(msg)


def test_markdown_long_runtime_hms(tmp_path: Path) -> None:  # type: ignore[name-defined]
    """Markdown should show HH:MM:SS plus seconds for long runtimes (>= 1 hour)."""
    results = [_build_single_result()]
    md_file = tmp_path / "long.md"
    long_seconds = 3_726.4  # 1h 2m 6.4s
    generate_markdown_report(
        results=results,
        filename=md_file,
        versions={"mlx": "0.0.0", "mlx-vlm": "0.0.0"},
        prompt="Test prompt",
        total_runtime_seconds=long_seconds,
    )
    content = md_file.read_text(encoding="utf-8")
    if "01:02:06" not in content:
        msg = "Missing HH:MM:SS component in long runtime markdown report"
        raise AssertionError(msg)
    if "3726.40" not in content:
        msg = "Missing precise seconds component in long runtime markdown report"
        raise AssertionError(msg)


def test_html_long_runtime_hms(tmp_path: Path) -> None:  # type: ignore[name-defined]
    """HTML should show HH:MM:SS plus seconds for long runtimes (>= 1 hour)."""
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
    if "02:00:45" not in content:
        msg = "Missing HH:MM:SS component in long runtime HTML report"
        raise AssertionError(msg)
    if "7245.90" not in content:
        msg = "Missing precise seconds component in long runtime HTML report"
        raise AssertionError(msg)
