"""Tests for metrics mode selection output paths."""

from __future__ import annotations

import argparse
import logging
import time
from typing import TYPE_CHECKING
from unittest.mock import patch

from check_models import (
    FileSafeFormatter,
    PerformanceResult,
    finalize_execution,
    log_summary,
    print_model_result,
    print_model_stats,
)

if TYPE_CHECKING:  # pragma: no cover - only for type hints
    from pathlib import Path

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

    def __init__(
        self,
        *,
        prompt_tokens: int = 10,
        generation_tokens: int = 5,
        generation_tps: float = 50.0,
        peak_memory: float = 0.25,
        text: str = "hello",
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.generation_tokens = generation_tokens
        self.generation_tps = generation_tps
        self.peak_memory = peak_memory
        self.active_memory = None
        self.cache_memory = None
        self.time = 1.0
        self.text = text


def _build_perf() -> PerformanceResult:
    return PerformanceResult(
        model_name="dummy/model",
        generation=_StubGeneration(),
        success=True,
        generation_time=1.0,
        model_load_time=0.5,
        total_time=1.5,
    )


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


def test_log_summary_uses_model_load_time_for_fastest_load(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Fastest-load metric should use model_load_time (not legacy load_time attr)."""
    caplog.set_level(logging.INFO)
    results = [
        PerformanceResult(
            model_name="model/slow-load",
            generation=_StubGeneration(generation_tps=10.0, peak_memory=2.0, text="good output"),
            success=True,
            generation_time=1.0,
            model_load_time=3.0,
            total_time=4.0,
        ),
        PerformanceResult(
            model_name="model/fast-load",
            generation=_StubGeneration(generation_tps=9.0, peak_memory=2.2, text="good output"),
            success=True,
            generation_time=1.1,
            model_load_time=0.5,
            total_time=1.6,
        ),
    ]

    log_summary(results)

    assert any(
        "Fastest load: model/fast-load (0.50s)" in record.message for record in caplog.records
    )


def test_print_model_stats_excludes_output_column(caplog: pytest.LogCaptureFixture) -> None:
    """CLI performance table should avoid huge output-preview column."""
    caplog.set_level(logging.INFO)
    results = [
        PerformanceResult(
            model_name="model/table",
            generation=_StubGeneration(text="very long " * 200),
            success=True,
            generation_time=1.0,
            model_load_time=0.4,
            total_time=1.4,
            quality_issues="verbose",
        ),
    ]

    print_model_stats(results)

    all_messages = "\n".join(record.message for record in caplog.records)
    assert "Output" not in all_messages


def test_print_model_result_failure_logs_actionable_details(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Failure output should include package/type and traceback/captured-output sections."""
    caplog.set_level(logging.INFO)
    tb_lines = "\n".join(f"line{i}" for i in range(1, 10))
    result = PerformanceResult(
        model_name="broken/model",
        generation=None,
        success=False,
        error_stage="Model Error",
        error_message="Model loading failed",
        error_type="ValueError",
        error_package="mlx-vlm",
        error_traceback=tb_lines,
        captured_output_on_fail="stdout sample\nstderr sample",
    )

    print_model_result(result, verbose=True, run_index=1, total_runs=1)

    assert any("Error package: mlx-vlm" in record.message for record in caplog.records)
    assert any("Error type: ValueError" in record.message for record in caplog.records)
    assert any("Traceback tail:" in record.message for record in caplog.records)
    assert any("Captured output:" in record.message for record in caplog.records)


def test_file_safe_formatter_strips_ansi() -> None:
    """File formatter should remove ANSI escapes from persisted logs."""
    formatter = FileSafeFormatter("%(message)s")
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="\x1b[91mred text\x1b[0m",
        args=(),
        exc_info=None,
    )

    assert formatter.format(record) == "red text"


def test_finalize_execution_logs_configured_log_and_env_paths(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Final summary should report configured --output-log/--output-env paths."""
    caplog.set_level(logging.INFO)
    custom_log = tmp_path / "custom.log"
    custom_env = tmp_path / "custom.env.log"
    custom_env.write_text("env", encoding="utf-8")

    args = argparse.Namespace(
        output_html=tmp_path / "report.html",
        output_markdown=tmp_path / "report.md",
        output_tsv=tmp_path / "report.tsv",
        output_jsonl=tmp_path / "report.jsonl",
        output_diagnostics=tmp_path / "diagnostics.md",
        output_log=custom_log,
        output_env=custom_env,
    )
    result = PerformanceResult(
        model_name="dummy/model",
        generation=_StubGeneration(),
        success=True,
        generation_time=1.0,
        model_load_time=0.5,
        total_time=1.5,
    )
    history_record: dict[str, object] = {
        "_type": "run",
        "timestamp": "2026-02-13 00:00:00",
        "model_results": {},
    }

    with (
        patch("check_models.print_cli_section"),
        patch("check_models.print_version_info"),
        patch("check_models.get_system_characteristics", return_value={}),
        patch("check_models.generate_html_report"),
        patch("check_models.generate_markdown_report"),
        patch("check_models.generate_tsv_report"),
        patch("check_models.save_jsonl_report"),
        patch("check_models.append_history_record", return_value=history_record),
        patch("check_models.generate_diagnostics_report", return_value=False),
        patch("check_models._log_history_comparison"),
    ):
        finalize_execution(
            args=args,
            results=[result],
            library_versions={"mlx": "0.0.0", "mlx-vlm": "0.0.0"},
            overall_start_time=time.perf_counter() - 0.5,
            prompt="test prompt",
            image_path=None,
        )

    messages = [record.message for record in caplog.records]
    assert any(str(custom_log.resolve()) in msg for msg in messages)
    assert any(str(custom_env.resolve()) in msg for msg in messages)
