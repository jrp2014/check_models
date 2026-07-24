"""Tests for metrics mode selection output paths."""

from __future__ import annotations

import argparse
import io
import logging
import time
from contextlib import ExitStack
from typing import TYPE_CHECKING
from unittest.mock import patch

from rich.console import Console

import check_models
from check_models import (
    FileSafeFormatter,
    GenerationQualityAnalysis,
    PerformanceResult,
    RuntimeDiagnostics,
    StyleAwareRichHandler,
    finalize_execution,
    log_summary,
    print_model_result,
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


_FINALIZE_REPORT_PATCHES = (
    "check_models.print_cli_section",
    "check_models.print_version_info",
    "check_models.generate_html_report",
    "check_models.generate_markdown_gallery_report",
    "check_models.save_jsonl_report",
    "check_models.save_run_json_report",
)

_EXPECTED_REPORT_ARTIFACT_LOG_LABELS = (
    "Output Index:",
    "HTML Report:",
    "Gallery Report:",
    "Diagnostics:",
    "JSONL Report:",
    "Run JSON:",
)


def _finalize_history_stub() -> dict[str, object]:
    return {
        "_type": "run",
        "timestamp": "2026-02-13 00:00:00",
        "model_results": {},
    }


def _run_finalize_with_report_patches(
    *,
    args: argparse.Namespace,
    result: PerformanceResult,
    overall_start_time: float,
) -> None:
    """Run finalization with report writers patched out for path/log assertions."""
    with ExitStack() as stack:
        for patch_target in _FINALIZE_REPORT_PATCHES:
            stack.enter_context(patch(patch_target))
        stack.enter_context(patch("check_models.get_system_characteristics", return_value={}))
        stack.enter_context(
            patch("check_models.append_history_record", return_value=_finalize_history_stub())
        )
        stack.enter_context(patch("check_models.generate_diagnostics_report", return_value=False))
        finalize_execution(
            args=args,
            results=[result],
            library_versions={"mlx": "0.0.0", "mlx-vlm": "0.0.0"},
            overall_start_time=overall_start_time,
            prompt="test prompt",
            image_path=None,
            metadata=None,
        )


def _assert_logged_paths(messages: list[str], *paths: Path) -> None:
    for path in paths:
        assert any(str(path.resolve()) in message for message in messages)


def _message_index(messages: list[str], label: str) -> int:
    return next(index for index, message in enumerate(messages) if label in message)


def _assert_report_artifact_log_order(messages: list[str]) -> None:
    report_start = _message_index(messages, "Reports successfully generated:")
    report_messages = messages[report_start:]
    positions = [
        _message_index(report_messages, label) for label in _EXPECTED_REPORT_ARTIFACT_LOG_LABELS
    ]
    assert positions == sorted(positions)


def test_console_handler_keeps_repeated_timestamps_visible() -> None:
    """Console logs should timestamp every record, including same-second records."""
    stream = io.StringIO()
    console = Console(
        file=stream,
        width=100,
        no_color=True,
        force_terminal=False,
        markup=False,
        highlight=False,
    )
    with patch.object(check_models, "_make_rich_console", return_value=console):
        handler = check_models._make_console_log_handler(
            level=logging.INFO,
            verbose=False,
            width=100,
        )

    test_logger = logging.getLogger("check-models-rich-timestamp-test")
    old_handlers = test_logger.handlers[:]
    old_level = test_logger.level
    old_propagate = test_logger.propagate
    try:
        test_logger.handlers.clear()
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.INFO)
        test_logger.propagate = False
        test_logger.info("first")
        test_logger.info("second")
    finally:
        test_logger.handlers[:] = old_handlers
        test_logger.setLevel(old_level)
        test_logger.propagate = old_propagate

    lines = [line for line in stream.getvalue().splitlines() if line.strip()]
    assert len(lines) == 2
    assert lines[0].startswith("[")
    assert lines[1].startswith("[")
    assert "first" in lines[0]
    assert "second" in lines[1]


def test_console_handler_hides_file_only_records() -> None:
    """Console handler should suppress records tagged for the file log only."""
    stream = io.StringIO()
    console = Console(
        file=stream,
        width=100,
        no_color=True,
        force_terminal=False,
        markup=False,
        highlight=False,
    )
    with patch.object(check_models, "_make_rich_console", return_value=console):
        handler = check_models._make_console_log_handler(
            level=logging.DEBUG,
            verbose=True,
            width=100,
        )

    test_logger = logging.getLogger("check-models-file-only-filter-test")
    old_handlers = test_logger.handlers[:]
    old_level = test_logger.level
    old_propagate = test_logger.propagate
    try:
        test_logger.handlers.clear()
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.DEBUG)
        test_logger.propagate = False
        test_logger.debug("file-only message", extra={"log_destination": "file"})
        test_logger.info("console message")
    finally:
        test_logger.handlers[:] = old_handlers
        test_logger.setLevel(old_level)
        test_logger.propagate = old_propagate

    output = stream.getvalue()
    assert "file-only message" not in output
    assert "console message" in output


def test_metrics_mode_compact_smoke(caplog: pytest.LogCaptureFixture) -> None:
    """Compact mode should emit Timing and Tokens lines."""
    caplog.set_level(logging.INFO)
    res = _build_perf()
    print_model_result(res, verbose=True, detailed_metrics=False)
    # New format uses "Timing:" (line 1) and "Tokens:" (line 2)
    timing_lines = [r.message for r in caplog.records if "Timing:" in r.message]
    assert timing_lines, "Expected Timing line in compact mode logs"


def test_metrics_mode_compact_shows_working_set_context(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Compact memory output should include the detected Metal denominator."""
    caplog.set_level(logging.INFO)
    res = PerformanceResult(
        model_name="dummy/model",
        generation=_StubGeneration(peak_memory=1.0),
        success=True,
        generation_time=1.0,
        model_load_time=0.5,
        total_time=1.5,
    )

    with patch("check_models._get_recommended_working_set_bytes", return_value=2_000_000_000):
        print_model_result(res, verbose=True, detailed_metrics=False)

    messages = "\n".join(record.message for record in caplog.records)
    assert "1.0 GB (50% of 1.86 GB recommended working set)" in messages


def test_metrics_mode_verbose_does_not_repeat_generated_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verbose mode should keep streamed output out of the post-run summary block."""
    caplog.set_level(logging.INFO)
    res = PerformanceResult(
        model_name="dummy/model",
        generation=_StubGeneration(text="Distinct streamed output", generation_tokens=5),
        success=True,
        generation_time=1.0,
        model_load_time=0.5,
        total_time=1.5,
    )

    print_model_result(res, verbose=True, detailed_metrics=False)

    messages = [record.message for record in caplog.records]
    assert not any("Generated Text:" in message for message in messages)
    assert not any("Distinct streamed output" in message for message in messages)
    assert any("Timing:" in message for message in messages)


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


def test_metrics_mode_detailed_shows_working_set_context(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Detailed peak-memory rows should include the same Metal context."""
    caplog.set_level(logging.INFO)
    res = PerformanceResult(
        model_name="dummy/model",
        generation=_StubGeneration(peak_memory=1.0),
        success=True,
        generation_time=1.0,
        model_load_time=0.5,
        total_time=1.5,
    )

    with patch("check_models._get_recommended_working_set_bytes", return_value=2_000_000_000):
        print_model_result(res, verbose=True, detailed_metrics=True)

    messages = "\n".join(record.message for record in caplog.records)
    assert "1.0 GB (50% of 1.86 GB recommended working set)" in messages


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
    assert "Upstream model prefill / first token:" in messages
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

    log_summary(results)

    messages = "\n".join(record.message for record in caplog.records)
    assert "Model Comparison (current run):" in messages
    assert "│ # │ Model" in messages
    assert "TPS │" in messages
    assert "│ Total(s)" in messages
    assert "│ PeakGB" in messages
    assert "TPS comparison chart:" in messages
    assert "Efficiency chart (higher is faster overall):" in messages
    assert "Failure stage frequency:" in messages


def test_log_summary_contextualizes_comparison_and_average_peak_memory(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Summary memory values should use the same detected Metal denominator."""
    caplog.set_level(logging.INFO)
    results = [
        PerformanceResult(
            model_name="org/model-a",
            generation=_StubGeneration(generation_tps=35.0, peak_memory=1.2, text="clean"),
            success=True,
            generation_time=1.0,
            model_load_time=0.4,
            total_time=1.4,
        ),
        PerformanceResult(
            model_name="org/model-b",
            generation=_StubGeneration(generation_tps=20.0, peak_memory=1.5, text="clean"),
            success=True,
            generation_time=1.3,
            model_load_time=0.7,
            total_time=2.0,
        ),
    ]

    with patch("check_models._get_recommended_working_set_bytes", return_value=2_000_000_000):
        log_summary(results)

    messages = "\n".join(record.message for record in caplog.records)
    assert "Recommended working set: 1.86 GB" in messages
    assert "67.5% of 1.86 GB recommended working set" in messages


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

    log_summary([result])

    messages = "\n".join(record.message for record in caplog.records)
    assert "Model Comparison (current run):" in messages
    assert "TPS comparison chart:" in messages
    assert "Efficiency chart (higher is faster overall):" not in messages


def test_log_summary_comparison_table_preserves_unicode_notes(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Rich comparison-table notes can keep Unicode without breaking alignment."""
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

    log_summary([result])

    # Collect the model-data row (contains "emoji-note") and all continuation
    # rows (│ … │ rows without "emoji-note" that immediately follow it).
    all_table_rows = [
        record.message for record in caplog.records if record.message.strip().startswith("│")
    ]
    assert all_table_rows
    emoji_note_rows = [r for r in all_table_rows if "emoji-note" in r]
    assert emoji_note_rows, "Model row not found in table output"

    # Unicode emoji must survive on the primary model row.
    assert "⚠" in emoji_note_rows[0]

    # Notes cell content may wrap across continuation rows at narrow render widths;
    # verify each significant token appears somewhere in the table section.
    table_text = " ".join(all_table_rows)
    assert "harness" in table_text
    assert "stop_token" in table_text


def test_log_summary_reports_execution_and_mechanical_observations(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Run summaries should emit facts without semantic scorecards."""
    caplog.set_level(logging.INFO)
    result = PerformanceResult(
        model_name="org/caption-model",
        generation=_StubGeneration(
            generation_tps=42.0,
            peak_memory=1.2,
            text="Two cats resting on a bright pink couch.",
        ),
        success=True,
        generation_time=1.0,
        model_load_time=0.4,
        total_time=1.4,
    )

    log_summary([result])

    messages = "\n".join(record.message for record in caplog.records)
    assert "Execution outcomes: completed=1" in messages
    assert "Mechanical observations: none" in messages
    assert "Cataloging Utility Snapshot:" not in messages
    assert "Metadata baseline:" not in messages
    assert "Best description:" not in messages
    assert "Best keywording:" not in messages


def test_log_summary_failure_uses_only_recorded_failure_facts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Failure summaries must not infer an owner or likely cause."""
    caplog.set_level(logging.INFO)
    result = PerformanceResult(
        model_name="broken/model",
        generation=None,
        success=False,
        failure_phase="decode",
        error_stage="Model Error",
        error_code="MODEL_MLX_VLM_DECODE",
        error_message="generation failed",
        error_type="ValueError",
        root_error_module="builtins",
        error_package="mlx-vlm",
        error_traceback="Traceback (most recent call last):\nValueError: generation failed",
    )

    log_summary([result])

    messages = "\n".join(record.message for record in caplog.records)
    assert (
        "broken/model | phase=decode | stage=Model Error | module=builtins | "
        "package=mlx-vlm | code=MODEL_MLX_VLM_DECODE | traceback=available"
    ) in messages
    assert "module=builtins" in messages
    assert "error=ValueError: generation failed" in messages
    assert "traceback=available" in messages
    assert "owner≈" not in messages
    assert "likely=" not in messages
    assert "generation/integration path" not in messages
    assert "model runtime failure" not in messages


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


def test_file_safe_formatter_uses_project_timestamp_shape() -> None:
    """File formatter should use stable second-resolution local timestamps."""
    formatter = FileSafeFormatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt=check_models.LOCAL_TIMESTAMP_FORMAT,
    )
    formatter.converter = time.gmtime
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="plain text",
        args=(),
        exc_info=None,
    )
    record.created = 0.0
    record.msecs = 321.0

    formatted = formatter.format(record)

    assert formatted in {
        "1970-01-01 00:00:00 GMT - INFO - plain text",
        "1970-01-01 00:00:00 UTC - INFO - plain text",
    }


def test_rich_debug_level_label_is_dim() -> None:
    """Only the DEBUG level label should render dim/gray."""
    handler = StyleAwareRichHandler(
        console=Console(file=io.StringIO(), force_terminal=True),
        show_level=True,
        show_time=False,
        show_path=False,
    )
    record = logging.LogRecord(
        name="check_models",
        level=logging.DEBUG,
        pathname=__file__,
        lineno=1,
        msg="debug details",
        args=(),
        exc_info=None,
    )

    level_text = handler.get_level_text(record)

    assert level_text.plain == "DEBUG   "
    assert level_text.spans
    assert str(level_text.spans[0].style) == "dim"


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
        output_gallery_markdown=tmp_path / "gallery.md",
        output_jsonl=tmp_path / "report.jsonl",
        output_run_json=tmp_path / "run.json",
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

    _run_finalize_with_report_patches(
        args=args,
        result=result,
        overall_start_time=time.perf_counter() - 0.5,
    )

    messages = [record.message for record in caplog.records]
    _assert_logged_paths(
        messages,
        custom_log,
        custom_env,
        args.output_gallery_markdown,
        args.output_diagnostics,
    )
    _assert_report_artifact_log_order(messages)


def test_report_generation_uses_single_artifact_plan(tmp_path: Path) -> None:
    """Report generation should expose one ordered artifact plan for jobs and logs."""
    args = argparse.Namespace(
        output_html=tmp_path / "report.html",
        output_gallery_markdown=tmp_path / "gallery.md",
        output_jsonl=tmp_path / "report.jsonl",
        output_run_json=tmp_path / "run.json",
        output_diagnostics=tmp_path / "diagnostics.md",
        output_log=tmp_path / "check_models.log",
        output_env=tmp_path / "environment.log",
    )
    inputs = check_models.ReportGenerationInputs(
        results=[],
        library_versions={"mlx": "0.0.0"},
        prompt="prompt",
        metadata=None,
        overall_time=1.0,
        image_path=None,
        system_info={},
        report_context=check_models._build_report_render_context(results=[], prompt="prompt"),
        output_paths=check_models._resolve_report_output_paths(args),
        runtime_fingerprint={},
    )

    artifacts = check_models._build_report_artifacts(inputs)

    assert [artifact.key for artifact in artifacts] == [
        "output_index",
        "html",
        "markdown_gallery",
        "diagnostics",
        "jsonl",
        "run_json",
    ]
    assert [artifact.label.strip() for artifact in artifacts] == [
        "Output Index:",
        "HTML Report:",
        "Gallery Report:",
        "Diagnostics:",
        "JSONL Report:",
        "Run JSON:",
    ]
    assert all(artifact.path.is_absolute() for artifact in artifacts)
    assert all(artifact.job is not None for artifact in artifacts if artifact.key != "diagnostics")


def test_report_artifact_specs_are_the_metadata_source(tmp_path: Path) -> None:
    """Generated report path, run-json, and dashboard metadata should share specs."""
    args = argparse.Namespace(
        output_html=tmp_path / "report.html",
        output_gallery_markdown=tmp_path / "gallery.md",
        output_jsonl=tmp_path / "report.jsonl",
        output_run_json=tmp_path / "run.json",
        output_diagnostics=tmp_path / "diagnostics.md",
        output_log=tmp_path / "check_models.log",
        output_env=tmp_path / "environment.log",
    )
    paths = check_models._resolve_report_output_paths(args)

    specs = check_models._build_report_artifact_specs(paths)

    assert tuple(spec.key for spec in specs) == (
        "output_index",
        "html",
        "markdown_gallery",
        "diagnostics",
        "jsonl",
        "run_json",
    )
    public_map = check_models._public_output_artifact_map(paths)
    assert set(public_map) == {
        "output_index",
        "results_html",
        "model_gallery",
        "diagnostics",
        "results_jsonl",
        "run_json",
        "log",
        "environment",
    }


def test_output_index_links_only_retained_artifacts(tmp_path: Path) -> None:
    """The navigation index must not reintroduce retired or conditional artifacts."""
    reports_dir = tmp_path / "reports"
    paths = check_models.ReportOutputPaths(
        index=tmp_path / "index.md",
        html=reports_dir / "results.html",
        gallery_markdown=reports_dir / "model_gallery.md",
        diagnostics=reports_dir / "diagnostics.md",
        jsonl=tmp_path / "results.jsonl",
        run_json=tmp_path / "run.json",
        log=tmp_path / "check_models.log",
        environment=tmp_path / "environment.log",
    )

    with patch.object(check_models._LinkStyleState, "value", "relative"):
        check_models.generate_output_index_report(paths.index, output_paths=paths)

    assert paths.index.read_text(encoding="utf-8").splitlines() == [
        "# Check Models Output Index",
        "",
        "- [results.html](reports/results.html)",
        "- [model_gallery.md](reports/model_gallery.md)",
        "- [diagnostics.md](reports/diagnostics.md)",
        "- [results.jsonl](results.jsonl)",
        "- [run.json](run.json)",
        "- [check_models.log](check_models.log)",
        "- [environment.log](environment.log)",
    ]


def test_finalize_execution_does_not_read_history_for_current_reports(tmp_path: Path) -> None:
    """Current report statuses must not depend on append-only history."""
    args = argparse.Namespace(
        output_html=tmp_path / "report.html",
        output_gallery_markdown=tmp_path / "gallery.md",
        output_jsonl=tmp_path / "report.jsonl",
        output_run_json=tmp_path / "run.json",
        output_diagnostics=tmp_path / "diagnostics.md",
        output_log=tmp_path / "check_models.log",
        output_env=tmp_path / "environment.log",
    )
    result = PerformanceResult(
        model_name="dummy/model",
        generation=_StubGeneration(text="A complete response."),
        success=True,
    )
    expected = check_models._assess_result(result)

    with patch.object(
        check_models,
        "_load_history_run_records",
        side_effect=AssertionError("current reports must not read history"),
        create=True,
    ):
        _run_finalize_with_report_patches(
            args=args,
            result=result,
            overall_start_time=time.perf_counter() - 0.5,
        )

    assert check_models._assess_result(result) == expected
