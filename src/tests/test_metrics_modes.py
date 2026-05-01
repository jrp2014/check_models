"""Tests for metrics mode selection output paths."""

from __future__ import annotations

import argparse
import logging
import time
from typing import TYPE_CHECKING
from unittest.mock import patch

import check_models
from check_models import (
    FileSafeFormatter,
    GenerationQualityAnalysis,
    HistoryModelResultRecord,
    HistoryRunRecord,
    PerformanceResult,
    RuntimeDiagnostics,
    _log_canonical_model_review,
    _log_history_comparison,
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
    prompt_tps: float | None
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
        prompt_tps: float = 100.0,
        generation_tokens: int = 5,
        generation_tps: float = 50.0,
        peak_memory: float = 0.25,
        text: str = "hello",
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.prompt_tps = prompt_tps
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


def _history_run_record(
    outcomes: dict[str, bool],
    *,
    prompt_hash: str = "abc123456789",
    image_path: str | None = "/Users/test/images/test.jpg",
    library_versions: dict[str, str | None] | None = None,
) -> HistoryRunRecord:
    """Create typed history run records for _log_history_comparison tests."""
    model_results: dict[str, HistoryModelResultRecord] = {}
    for model_name, success in outcomes.items():
        model_results[model_name] = {
            "success": success,
            "error_stage": None if success else "Model Error",
            "error_type": None if success else "RuntimeError",
            "error_package": None if success else "mlx-vlm",
        }

    return {
        "_type": "run",
        "format_version": "1.0",
        "timestamp": "2026-02-15 00:00:00 GMT",
        "prompt_hash": prompt_hash,
        "prompt_preview": "Describe this image.",
        "image_path": image_path,
        "model_results": model_results,
        "system": {"OS": "macOS"},
        "library_versions": library_versions or {"mlx": "0.27.0", "mlx-vlm": "0.4.0"},
    }


def test_metrics_mode_compact_smoke(caplog: pytest.LogCaptureFixture) -> None:
    """Compact mode should emit Timing and Tokens lines."""
    caplog.set_level(logging.INFO)
    res = _build_perf()
    print_model_result(res, verbose=True, detailed_metrics=False)
    # New format uses "Timing:" (line 1) and "Tokens:" (line 2)
    timing_lines = [r.message for r in caplog.records if "Timing:" in r.message]
    assert timing_lines, "Expected Timing line in compact mode logs"


def test_metrics_mode_compact_surfaces_runtime_hints(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Compact mode should surface prep, first-token, and abnormal stop hints."""
    caplog.set_level(logging.INFO)
    res = PerformanceResult(
        model_name="dummy/model",
        generation=_StubGeneration(),
        success=True,
        generation_time=1.0,
        model_load_time=0.5,
        total_time=1.8,
        runtime_diagnostics=RuntimeDiagnostics(
            input_validation_time_s=0.05,
            model_load_time_s=0.5,
            prompt_prep_time_s=0.15,
            decode_time_s=1.0,
            cleanup_time_s=0.1,
            first_token_latency_s=0.3,
            stop_reason="timeout",
        ),
    )

    print_model_result(res, verbose=True, detailed_metrics=False)

    messages = "\n".join(record.message for record in caplog.records)
    assert "prep=0.15s" in messages
    assert "first=0.30s" in messages
    assert "stop=timeout" in messages


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


def test_metrics_mode_detailed_logs_runtime_phase_details(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Detailed mode should surface extra runtime phases and stop reason."""
    caplog.set_level(logging.INFO)
    res = PerformanceResult(
        model_name="dummy/model",
        generation=_StubGeneration(),
        success=True,
        generation_time=1.0,
        model_load_time=0.5,
        total_time=1.8,
        runtime_diagnostics=RuntimeDiagnostics(
            input_validation_time_s=0.05,
            model_load_time_s=0.5,
            prompt_prep_time_s=0.15,
            decode_time_s=1.0,
            cleanup_time_s=0.1,
            first_token_latency_s=0.3,
            stop_reason="completed",
        ),
    )

    print_model_result(res, verbose=True, detailed_metrics=True)

    messages = "\n".join(record.message for record in caplog.records)
    assert "Validation:" in messages
    assert "Prompt prep:" in messages
    assert "Cleanup:" in messages
    assert "First token:" in messages
    assert "Stop reason:" in messages


def test_print_model_result_non_verbose_labels_generated_text_preview(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Non-verbose preview mode should still label the emitted model output block."""
    caplog.set_level(logging.INFO)
    preview_text = (
        "- Keywords hint: St Pancras, clock tower, Victorian Gothic\n"
        "architecture, public square, urban space"
    )
    analysis = GenerationQualityAnalysis(
        is_repetitive=False,
        repeated_token=None,
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
        missing_sections=["title", "description", "keywords"],
    )
    result = PerformanceResult(
        model_name="dummy/model",
        generation=_StubGeneration(text=preview_text, generation_tokens=32),
        success=True,
        generation_time=1.0,
        model_load_time=0.5,
        total_time=1.5,
        quality_analysis=analysis,
    )

    print_model_result(result, verbose=False)

    messages = [record.message for record in caplog.records]
    assert any("Missing sections:" in message for message in messages)
    label_index = next(i for i, message in enumerate(messages) if "Generated Text:" in message)
    output_index = next(
        i for i, message in enumerate(messages) if "- Keywords hint: St Pancras" in message
    )
    assert label_index < output_index


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


def test_log_summary_emits_comparison_table_and_ascii_charts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Summary should include tabulated model comparison and compact metric charts."""
    caplog.set_level(logging.INFO)
    results = [
        PerformanceResult(
            model_name="org/model-a",
            generation=_StubGeneration(generation_tps=35.0, peak_memory=1.2, text="clean output"),
            success=True,
            generation_time=1.0,
            model_load_time=0.4,
            total_time=1.4,
        ),
        PerformanceResult(
            model_name="org/model-b",
            generation=_StubGeneration(generation_tps=20.0, peak_memory=1.5, text="clean output"),
            success=True,
            generation_time=1.3,
            model_load_time=0.7,
            total_time=2.0,
        ),
        PerformanceResult(
            model_name="org/model-c",
            generation=None,
            success=False,
            error_stage="Generation Error",
            error_message="timeout",
            generation_time=0.5,
            model_load_time=0.2,
            total_time=0.7,
        ),
    ]

    log_summary(results, prompt="Describe this image.")

    messages = "\n".join(record.message for record in caplog.records)
    assert "Model Comparison (current run):" in messages
    assert "|   # | Model" in messages
    assert "|   TPS |   Total(s) |   Load(s) |   PeakGB |" in messages
    assert "TPS comparison chart:" in messages
    assert "Efficiency chart (higher is faster overall):" in messages
    assert "Failure stage frequency:" in messages


def test_log_summary_single_model_omits_efficiency_chart(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Single-model runs should still show comparison table without cross-model efficiency chart."""
    caplog.set_level(logging.INFO)
    result = PerformanceResult(
        model_name="org/single-model",
        generation=_StubGeneration(generation_tps=15.0, peak_memory=1.1, text="good output"),
        success=True,
        generation_time=1.0,
        model_load_time=0.5,
        total_time=1.5,
    )

    log_summary([result], prompt="Describe this image.")

    messages = "\n".join(record.message for record in caplog.records)
    assert "Model Comparison (current run):" in messages
    assert "TPS comparison chart:" in messages
    assert "Efficiency chart (higher is faster overall):" not in messages


def test_log_summary_comparison_table_sanitizes_non_ascii_notes(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Comparison-table notes should drop emoji to preserve terminal alignment."""
    caplog.set_level(logging.INFO)
    result = PerformanceResult(
        model_name="org/emoji-note",
        generation=_StubGeneration(generation_tps=15.0, peak_memory=1.1, text="good output"),
        success=True,
        generation_time=1.0,
        model_load_time=0.5,
        total_time=1.5,
        quality_issues="⚠️harness(stop_token), output:zero_tokens",
    )

    log_summary([result], prompt="Describe this image.")

    table_row_messages = [
        record.message
        for record in caplog.records
        if "emoji-note" in record.message and record.message.strip().startswith("|")
    ]
    assert table_row_messages
    row = table_row_messages[0]
    assert "harness(stop_token)" in row
    assert "⚠" not in row


def test_log_summary_reports_metadata_baseline_delta_when_context_present(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Cataloging triage should report baseline and delta when prompt includes metadata context."""
    caplog.set_level(logging.INFO)
    prompt = (
        "Analyze this image.\n\n"
        "Context: Existing metadata hints (use only if visually consistent):\n"
        "- Title hint: Sunset over mountain lake\n"
        "- Description hint: A colorful sunset behind mountains with lake reflection and trees.\n"
        "- Keyword hints: sunset, mountains, lake, reflection, trees, nature, landscape\n"
        "\n"
        "Prioritize what is visibly present."
    )
    results = [
        PerformanceResult(
            model_name="org/model-good",
            generation=_StubGeneration(
                generation_tps=24.0,
                peak_memory=1.3,
                generation_tokens=180,
                text=(
                    "Title: Golden alpine sunset over mirrored mountain lake\n"
                    "Description: Warm orange light outlines the ridgeline while evergreen "
                    "silhouettes frame a still alpine lake that mirrors streaked clouds.\n"
                    "Keywords: alpine sunset, mirrored lake, mountain ridgeline, evergreen "
                    "silhouettes, orange sky, reflection, wilderness, scenic vista"
                ),
            ),
            success=True,
            generation_time=1.0,
            model_load_time=0.4,
            total_time=1.4,
        ),
        PerformanceResult(
            model_name="org/model-bad",
            generation=_StubGeneration(
                generation_tps=12.0,
                peak_memory=1.1,
                generation_tokens=12,
                text="Title: Sunset. Description: sunset image. Keywords: sunset, mountain",
            ),
            success=True,
            generation_time=1.1,
            model_load_time=0.5,
            total_time=1.6,
        ),
    ]

    log_summary(results, prompt=prompt)

    messages = "\n".join(record.message for record in caplog.records)
    assert "Metadata baseline:" in messages
    assert "Vs metadata:" in messages


def test_log_history_comparison_emits_tables_and_transition_chart(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """History comparison should show run-over-run tables plus transition chart/details."""
    caplog.set_level(logging.INFO)
    previous = _history_run_record(
        {"org/model-a": True, "org/model-b": False, "org/model-old": True},
        prompt_hash="1111222233334444",
        image_path="/Users/test/images/previous.jpg",
        library_versions={"mlx": "0.26.0", "mlx-vlm": "0.3.0"},
    )
    current = _history_run_record(
        {"org/model-a": False, "org/model-b": True, "org/model-new": False},
        prompt_hash="aaaa222233334444",
        image_path="/Users/test/images/current.jpg",
        library_versions={"mlx": "0.27.0", "mlx-vlm": "0.4.0"},
    )

    _log_history_comparison(previous, current)

    messages = "\n".join(record.message for record in caplog.records)
    assert "Run-over-run comparison:" in messages
    assert "| Metric" in messages
    assert "Comparison context:" in messages
    assert "Status transition counts:" in messages
    assert "Detailed model transitions:" in messages
    assert "Prompt differs from previous run" in messages
    assert "Image path differs from previous run" in messages
    assert "Regression" in messages
    assert "Recovery" in messages


def test_log_history_comparison_baseline_emits_current_status_chart(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Baseline history should render table/context and current status chart without transitions."""
    caplog.set_level(logging.INFO)
    current = _history_run_record({"org/single": True})

    _log_history_comparison(None, current)

    messages = "\n".join(record.message for record in caplog.records)
    assert "No prior history run available. Baseline created." in messages
    assert "Run-over-run comparison:" in messages
    assert "Comparison context:" in messages
    assert "Current run status counts:" in messages
    assert "Detailed model transitions:" not in messages


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


def test_canonical_review_log_emits_verdict_block(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Canonical review logging should emit the ordered verdict block and full output."""
    caplog.set_level(logging.DEBUG)
    result = PerformanceResult(
        model_name="dummy/model",
        generation=_StubGeneration(text="Title: Example output"),
        success=True,
        quality_analysis=check_models.analyze_generation_text(
            "Title: Example output",
            generated_tokens=6,
            requested_max_tokens=32,
        ),
        requested_max_tokens=32,
    )

    _log_canonical_model_review(result)

    messages = "\n".join(record.message for record in caplog.records)
    assert "=== CANONICAL REVIEW: dummy/model ===" in messages
    assert "Verdict:" in messages
    assert "Trusted hints:" in messages
    assert "Token accounting:" in messages
    assert "Full output:" in messages


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
        output_gallery_markdown=tmp_path / "gallery.md",
        output_review=tmp_path / "review.md",
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
        patch("check_models.generate_markdown_gallery_report"),
        patch("check_models.generate_review_report"),
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
            metadata=None,
        )

    messages = [record.message for record in caplog.records]
    assert any(str(custom_log.resolve()) in msg for msg in messages)
    assert any(str(custom_env.resolve()) in msg for msg in messages)
    assert any(str(args.output_gallery_markdown.resolve()) in msg for msg in messages)
    assert any(str(args.output_review.resolve()) in msg for msg in messages)


def test_finalize_execution_prunes_canonical_repro_bundle_dir(tmp_path: Path) -> None:
    """Finalization should prune output/repro_bundles, not output/reports/repro_bundles."""
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    args = argparse.Namespace(
        output_html=reports_dir / "report.html",
        output_markdown=reports_dir / "report.md",
        output_gallery_markdown=reports_dir / "gallery.md",
        output_review=reports_dir / "review.md",
        output_tsv=reports_dir / "report.tsv",
        output_jsonl=tmp_path / "report.jsonl",
        output_diagnostics=reports_dir / "diagnostics.md",
        output_log=tmp_path / "check_models.log",
        output_env=tmp_path / "environment.log",
        prune_repro_days=90,
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
        patch("check_models.generate_markdown_gallery_report"),
        patch("check_models.generate_review_report"),
        patch("check_models.generate_tsv_report"),
        patch("check_models.save_jsonl_report"),
        patch("check_models.append_history_record", return_value=history_record),
        patch("check_models.generate_diagnostics_report", return_value=False),
        patch("check_models._log_history_comparison"),
        patch("check_models._prune_repro_bundles", return_value=0) as prune_repro_bundles,
    ):
        finalize_execution(
            args=args,
            results=[result],
            library_versions={"mlx": "0.0.0", "mlx-vlm": "0.0.0"},
            overall_start_time=time.perf_counter() - 0.5,
            prompt="test prompt",
            image_path=None,
            metadata=None,
        )

    prune_repro_bundles.assert_called_once_with(tmp_path / "repro_bundles", 90)
