"""Tests for report generation edge cases (empty input, all-failed results)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import check_models

if TYPE_CHECKING:
    from pathlib import Path

    from check_models import HistoryModelResultRecord, HistoryRunRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _MockGeneration:
    """Minimal stand-in for GenerationResult used by report generators."""

    text: str | None = "output"
    prompt_tokens: int | None = 10
    generation_tokens: int | None = 5
    time: float | None = None
    active_memory: float | None = None
    cache_memory: float | None = None


def _stub_versions() -> check_models.LibraryVersionDict:
    return {
        "numpy": "1.0",
        "mlx": "0.1",
        "mlx-metal": None,
        "mlx-vlm": "0.1",
        "mlx-lm": None,
        "huggingface-hub": "0.1",
        "transformers": "4.0",
        "tokenizers": "0.1",
        "Pillow": "10.0",
    }


def _make_success(name: str = "org/model-ok") -> check_models.PerformanceResult:
    return check_models.PerformanceResult(
        model_name=name,
        success=True,
        generation=_MockGeneration(),
        total_time=1.0,
        generation_time=0.5,
        model_load_time=0.5,
    )


def _make_failure(
    name: str = "org/model-fail",
    error_type: str = "ValueError",
    error_package: str = "mlx-vlm",
) -> check_models.PerformanceResult:
    return check_models.PerformanceResult(
        model_name=name,
        success=False,
        generation=None,
        error_stage="load",
        error_message="boom",
        error_type=error_type,
        error_package=error_package,
    )


def _history_run(
    model_success: dict[str, bool],
    *,
    timestamp: str,
) -> HistoryRunRecord:
    """Build a fully shaped history run record for diagnostics-history tests."""
    model_results: dict[str, HistoryModelResultRecord] = {}
    for model, success in model_success.items():
        model_results[model] = {
            "success": success,
            "error_stage": None,
            "error_type": None,
            "error_package": None,
        }

    return {
        "_type": "run",
        "format_version": "1.0",
        "timestamp": timestamp,
        "prompt_hash": "hash",
        "prompt_preview": "preview",
        "image_path": None,
        "model_results": model_results,
        "system": {},
        "library_versions": {},
    }


# ===================================================================
# HTML report
# ===================================================================


class TestHtmlReportEdgeCases:
    """Edge-case coverage for generate_html_report."""

    def test_empty_results_does_not_write(self, tmp_path: Path) -> None:
        """Empty result list should produce no file."""
        out = tmp_path / "empty.html"
        check_models.generate_html_report(
            results=[],
            filename=out,
            versions=_stub_versions(),
            prompt="unused",
            total_runtime_seconds=0.0,
        )
        assert not out.exists()

    def test_all_failed_results_produces_file(self, tmp_path: Path) -> None:
        """All-failed result list should still produce a report."""
        out = tmp_path / "failed.html"
        check_models.generate_html_report(
            results=[_make_failure("org/a"), _make_failure("org/b")],
            filename=out,
            versions=_stub_versions(),
            prompt="describe",
            total_runtime_seconds=5.0,
        )
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "org/a" in content
        assert "org/b" in content

    def test_mixed_results_contains_both(self, tmp_path: Path) -> None:
        """Report with mixed success/failure should contain both models."""
        out = tmp_path / "mixed.html"
        check_models.generate_html_report(
            results=[_make_success("org/good"), _make_failure("org/bad")],
            filename=out,
            versions=_stub_versions(),
            prompt="describe",
            total_runtime_seconds=3.0,
        )
        content = out.read_text(encoding="utf-8")
        assert "org/good" in content
        assert "org/bad" in content


# ===================================================================
# Markdown report
# ===================================================================


class TestMarkdownReportEdgeCases:
    """Edge-case coverage for generate_markdown_report."""

    def test_empty_results_does_not_write(self, tmp_path: Path) -> None:
        """Empty result list should produce no file."""
        out = tmp_path / "empty.md"
        check_models.generate_markdown_report(
            results=[],
            filename=out,
            versions=_stub_versions(),
            prompt="unused",
            total_runtime_seconds=0.0,
        )
        assert not out.exists()

    def test_all_failed_results_produces_file(self, tmp_path: Path) -> None:
        """All-failed result list should still produce a report."""
        out = tmp_path / "failed.md"
        check_models.generate_markdown_report(
            results=[_make_failure("org/c"), _make_failure("org/d")],
            filename=out,
            versions=_stub_versions(),
            prompt="describe",
            total_runtime_seconds=4.0,
        )
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "org/c" in content
        assert "org/d" in content

    def test_mixed_results_contains_both(self, tmp_path: Path) -> None:
        """Report with mixed success/failure should contain both models."""
        out = tmp_path / "mixed.md"
        check_models.generate_markdown_report(
            results=[_make_success("org/good"), _make_failure("org/bad")],
            filename=out,
            versions=_stub_versions(),
            prompt="describe",
            total_runtime_seconds=2.0,
        )
        content = out.read_text(encoding="utf-8")
        assert "org/good" in content
        assert "org/bad" in content


# ===================================================================
# TSV report
# ===================================================================


class TestTsvReportEdgeCases:
    """Edge-case coverage for generate_tsv_report."""

    def test_empty_results_does_not_write(self, tmp_path: Path) -> None:
        """Empty result list should produce no file."""
        out = tmp_path / "empty.tsv"
        check_models.generate_tsv_report(results=[], filename=out)
        assert not out.exists()

    def test_all_failed_results_produces_file(self, tmp_path: Path) -> None:
        """All-failed result list should still produce a report."""
        out = tmp_path / "failed.tsv"
        check_models.generate_tsv_report(
            results=[_make_failure("org/e"), _make_failure("org/f")],
            filename=out,
        )
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "org/e" in content
        assert "org/f" in content

    def test_tsv_has_metadata_comment(self, tmp_path: Path) -> None:
        """TSV output should start with a generated_at metadata comment."""
        out = tmp_path / "meta.tsv"
        check_models.generate_tsv_report(
            results=[_make_success()],
            filename=out,
        )
        first_line = out.read_text(encoding="utf-8").splitlines()[0]
        assert first_line.startswith("# generated_at:")

    def test_tsv_has_error_columns(self, tmp_path: Path) -> None:
        """TSV should include error_type and error_package columns."""
        out = tmp_path / "cols.tsv"
        check_models.generate_tsv_report(
            results=[_make_failure(error_type="RuntimeError", error_package="transformers")],
            filename=out,
        )
        content = out.read_text(encoding="utf-8")
        # Header row (skip the comment line)
        header_line = content.splitlines()[1]
        assert "error_type" in header_line
        assert "error_package" in header_line
        # Data row
        data_line = content.splitlines()[2]
        assert "RuntimeError" in data_line
        assert "transformers" in data_line

    def test_tsv_error_columns_empty_for_success(self, tmp_path: Path) -> None:
        """Successful models should have empty error columns in the header."""
        out = tmp_path / "ok.tsv"
        check_models.generate_tsv_report(
            results=[_make_success()],
            filename=out,
        )
        content = out.read_text(encoding="utf-8")
        # The header must advertise error_type / error_package columns
        header_line = content.splitlines()[1]
        assert "error_type" in header_line
        assert "error_package" in header_line
        # For a success row, those columns are empty strings; tabulate may
        # trim trailing whitespace-only fields, so just verify the data row
        # does NOT contain a populated error value.
        data_line = content.splitlines()[2]
        stripped_fields = [f.strip() for f in data_line.split("\t")]
        # error_type and error_package should not contain real values
        assert "RuntimeError" not in stripped_fields
        assert "transformers" not in stripped_fields


# ===================================================================
# Diagnostics report
# ===================================================================


def _make_failure_with_details(
    name: str = "org/model-fail",
    *,
    error_msg: str = "boom",
    error_type: str = "ValueError",
    error_package: str = "mlx-vlm",
    error_stage: str = "Model Error",
    traceback_str: str | None = None,
    captured_output: str | None = None,
) -> check_models.PerformanceResult:
    """Create a failure result with full error details for diagnostics tests."""
    return check_models.PerformanceResult(
        model_name=name,
        success=False,
        generation=None,
        error_stage=error_stage,
        error_message=error_msg,
        error_type=error_type,
        error_package=error_package,
        captured_output_on_fail=captured_output,
        error_traceback=traceback_str,
    )


class TestDiagnosticsReport:
    """Tests for generate_diagnostics_report and its helpers."""

    def test_no_report_when_all_succeed(self, tmp_path: Path) -> None:
        """No diagnostics file when every model succeeds without harness issues."""
        out = tmp_path / "diag.md"
        result = check_models.generate_diagnostics_report(
            results=[_make_success()],
            filename=out,
            versions=_stub_versions(),
            system_info={"Python Version": "3.13", "GPU/Chip": "M4"},
            prompt="test",
        )
        assert result is False
        assert not out.exists()

    def test_report_written_on_failure(self, tmp_path: Path) -> None:
        """Diagnostics file created when a model fails."""
        out = tmp_path / "diag.md"
        result = check_models.generate_diagnostics_report(
            results=[
                _make_success(),
                _make_failure_with_details(
                    "org/bad-model",
                    error_msg="[broadcast_shapes] Shapes (3,1,2048) and (3,1,6065) mismatch",
                ),
            ],
            filename=out,
            versions=_stub_versions(),
            system_info={"Python Version": "3.13"},
            prompt="test prompt",
        )
        assert result is True
        content = out.read_text(encoding="utf-8")
        assert "# Diagnostics Report" in content
        assert "org/bad-model" in content
        assert "broadcast_shapes" in content

    def test_environment_table_includes_versions(self, tmp_path: Path) -> None:
        """Environment table should include library versions and system info."""
        out = tmp_path / "diag.md"
        versions = _stub_versions()
        versions["mlx-vlm"] = "0.3.11"
        check_models.generate_diagnostics_report(
            results=[_make_failure_with_details()],
            filename=out,
            versions=versions,
            system_info={"Python Version": "3.13.9", "GPU/Chip": "Apple M4 Max"},
            prompt="test",
        )
        content = out.read_text(encoding="utf-8")
        assert "| mlx-vlm | 0.3.11 |" in content
        assert "| Python Version | 3.13.9 |" in content
        assert "| GPU/Chip | Apple M4 Max |" in content

    def test_failure_clustering_groups_similar_errors(self, tmp_path: Path) -> None:
        """Models with the same error pattern (differing only in numbers) should cluster."""
        out = tmp_path / "diag.md"
        check_models.generate_diagnostics_report(
            results=[
                _make_failure_with_details(
                    "org/model-a",
                    error_msg=(
                        "[broadcast_shapes] Shapes (3,1,2048) and (3,1,6065) cannot be broadcast"
                    ),
                ),
                _make_failure_with_details(
                    "org/model-b",
                    error_msg=(
                        "[broadcast_shapes] Shapes (984,2048) and (1,0,2048) cannot be broadcast"
                    ),
                ),
            ],
            filename=out,
            versions=_stub_versions(),
            system_info={},
            prompt="test",
        )
        content = out.read_text(encoding="utf-8")
        # Both should be in the same cluster section, not separate sections
        # Count the number of "##" headings that mention "Model Error"
        error_sections = [
            ln for ln in content.splitlines() if ln.startswith("## ") and "Model Error" in ln
        ]
        assert len(error_sections) == 1
        # Both models mentioned in that section
        assert "org/model-a" in content
        assert "org/model-b" in content

    def test_different_errors_get_separate_clusters(self, tmp_path: Path) -> None:
        """Different error types should produce separate sections."""
        out = tmp_path / "diag.md"
        check_models.generate_diagnostics_report(
            results=[
                _make_failure_with_details(
                    "org/model-a",
                    error_msg="chat_template is not set",
                    error_stage="No Chat Template",
                ),
                _make_failure_with_details(
                    "org/model-b",
                    error_msg="Missing 1 parameters: lm_head.weight",
                    error_stage="Weight Mismatch",
                    error_package="mlx",
                ),
            ],
            filename=out,
            versions=_stub_versions(),
            system_info={},
            prompt="test",
        )
        content = out.read_text(encoding="utf-8")
        assert "No Chat Template" in content
        assert "Weight Mismatch" in content

    def test_traceback_tail_included(self, tmp_path: Path) -> None:
        """Traceback excerpt should appear in the report."""
        tb = (
            "Traceback (most recent call last):\n"
            '  File "foo.py", line 10, in bar\n'
            "    do_stuff()\n"
            '  File "baz.py", line 20, in do_stuff\n'
            "    raise ValueError('bad shape')\n"
            "ValueError: bad shape\n"
        )
        out = tmp_path / "diag.md"
        check_models.generate_diagnostics_report(
            results=[
                _make_failure_with_details("org/m", traceback_str=tb, error_msg="bad shape"),
            ],
            filename=out,
            versions=_stub_versions(),
            system_info={},
            prompt="test",
        )
        content = out.read_text(encoding="utf-8")
        assert "Traceback (tail)" in content
        assert "ValueError: bad shape" in content

    def test_full_tracebacks_in_collapsed_section(self, tmp_path: Path) -> None:
        """Diagnostics should include full traceback blocks per failed model."""
        tb = "\n".join(f"trace-line-{i}" for i in range(12))
        out = tmp_path / "diag.md"
        check_models.generate_diagnostics_report(
            results=[_make_failure_with_details("org/m", traceback_str=tb, error_msg="bad shape")],
            filename=out,
            versions=_stub_versions(),
            system_info={},
            prompt="test",
        )
        content = out.read_text(encoding="utf-8")
        assert "Full tracebacks (all models in this cluster)" in content
        # line 0 is not in the 6-line tail and proves full traceback inclusion.
        assert "trace-line-0" in content

    def test_captured_output_in_collapsed_section(self, tmp_path: Path) -> None:
        """Diagnostics should include captured stdout/stderr blocks when available."""
        out = tmp_path / "diag.md"
        check_models.generate_diagnostics_report(
            results=[
                _make_failure_with_details(
                    "org/m",
                    error_msg="bad shape",
                    captured_output="=== STDERR ===\nTokenizer warning here",
                ),
            ],
            filename=out,
            versions=_stub_versions(),
            system_info={},
            prompt="test",
        )
        content = out.read_text(encoding="utf-8")
        assert "Captured stdout/stderr (all models in this cluster)" in content
        assert "Tokenizer warning here" in content

    def test_history_context_includes_regressions_recoveries_and_repro(
        self, tmp_path: Path
    ) -> None:
        """Diagnostics should show history-based regression/recovery and retry context."""
        out = tmp_path / "diag.md"
        history_path = tmp_path / "results.history.jsonl"

        old_record = _history_run(
            {"org/a": False, "org/b": False},
            timestamp="2026-02-10 10:00:00 GMT",
        )
        previous_record = _history_run(
            {"org/a": True, "org/b": False},
            timestamp="2026-02-11 10:00:00 GMT",
        )
        current_record = _history_run(
            {"org/a": False, "org/b": True},
            timestamp="2026-02-12 10:00:00 GMT",
        )
        history_path.write_text(
            "\n".join(
                [
                    json.dumps(old_record),
                    json.dumps(previous_record),
                    json.dumps(current_record),
                ],
            )
            + "\n",
            encoding="utf-8",
        )

        check_models.generate_diagnostics_report(
            results=[_make_failure_with_details("org/a", error_msg="boom")],
            filename=out,
            versions=_stub_versions(),
            system_info={},
            prompt="test",
            history_path=history_path,
            previous_history=previous_record,
            current_history=current_record,
        )
        content = out.read_text(encoding="utf-8")
        assert "## History Context" in content
        assert "`org/a`" in content
        assert "new regression" in content
        assert "Recoveries since previous run" in content
        assert "`org/b`" in content
        assert "2/3 recent runs failed" in content
        assert "2026-02-10 10:00:00 GMT" in content

    def test_priority_table_present(self, tmp_path: Path) -> None:
        """Priority Summary table should appear with correct structure."""
        out = tmp_path / "diag.md"
        check_models.generate_diagnostics_report(
            results=[_make_failure_with_details()],
            filename=out,
            versions=_stub_versions(),
            system_info={},
            prompt="test",
        )
        content = out.read_text(encoding="utf-8")
        assert "## Priority Summary" in content
        assert "| Priority | Issue |" in content

    def test_reproducibility_section(self, tmp_path: Path) -> None:
        """Reproducibility section should include model-specific commands."""
        out = tmp_path / "diag.md"
        check_models.generate_diagnostics_report(
            results=[_make_failure_with_details("org/broken-model")],
            filename=out,
            versions=_stub_versions(),
            system_info={},
            prompt="test",
        )
        content = out.read_text(encoding="utf-8")
        assert "## Reproducibility" in content
        assert "python src/check_models.py --model org/broken-model" in content

    def test_prompt_in_details_block(self, tmp_path: Path) -> None:
        """Prompt should be in a collapsible details block."""
        out = tmp_path / "diag.md"
        check_models.generate_diagnostics_report(
            results=[_make_failure_with_details()],
            filename=out,
            versions=_stub_versions(),
            system_info={},
            prompt="Analyze this image carefully.",
        )
        content = out.read_text(encoding="utf-8")
        assert "<details>" in content
        assert "Analyze this image carefully." in content

    def test_high_priority_for_multi_model_cluster(self, tmp_path: Path) -> None:
        """Clusters with ≥2 models should be tagged High priority."""
        out = tmp_path / "diag.md"
        check_models.generate_diagnostics_report(
            results=[
                _make_failure_with_details("org/a", error_msg="same error for all"),
                _make_failure_with_details("org/b", error_msg="same error for all"),
            ],
            filename=out,
            versions=_stub_versions(),
            system_info={},
            prompt="test",
        )
        content = out.read_text(encoding="utf-8")
        assert "Priority: High" in content


class TestClusterFailuresByPattern:
    """Unit tests for the _cluster_failures_by_pattern helper."""

    def test_strips_model_name_from_wrapper(self) -> None:
        """Model-specific prefixes should be removed before clustering."""
        results = [
            _make_failure_with_details(
                "org/model-a",
                error_msg="Model generation failed for org/model-a: token mismatch",
            ),
            _make_failure_with_details(
                "org/model-b",
                error_msg="Model generation failed for org/model-b: token mismatch",
            ),
        ]
        clusters = check_models._cluster_failures_by_pattern(results)
        assert len(clusters) == 1

    def test_normalises_numbers(self) -> None:
        """Varying numeric values (shape dimensions) should be normalised."""
        results = [
            _make_failure_with_details(
                "org/a",
                error_msg="Shapes (3,1,2048) and (3,1,6065) mismatch",
            ),
            _make_failure_with_details(
                "org/b",
                error_msg="Shapes (984,2048) and (1,0,2048) mismatch",
            ),
        ]
        clusters = check_models._cluster_failures_by_pattern(results)
        assert len(clusters) == 1
        assert len(next(iter(clusters.values()))) == 2

    def test_different_patterns_separate(self) -> None:
        """Fundamentally different errors should not cluster together."""
        results = [
            _make_failure_with_details("org/a", error_msg="chat_template is not set"),
            _make_failure_with_details("org/b", error_msg="Missing 1 parameters"),
        ]
        clusters = check_models._cluster_failures_by_pattern(results)
        assert len(clusters) == 2


class TestFormatTracebackTail:
    """Unit tests for _format_traceback_tail."""

    def test_none_input(self) -> None:
        """None input returns None."""
        assert check_models._format_traceback_tail(None) is None

    def test_empty_string(self) -> None:
        """Empty string returns None."""
        assert check_models._format_traceback_tail("") is None

    def test_extracts_tail(self) -> None:
        """Long tracebacks are truncated to the tail lines."""
        tb = "\n".join(f"line {i}" for i in range(20))
        result = check_models._format_traceback_tail(tb)
        assert result is not None
        lines = result.splitlines()
        assert len(lines) == 6  # _DIAGNOSTICS_TRACEBACK_TAIL_LINES
        assert "line 19" in lines[-1]

    def test_short_traceback_returned_fully(self) -> None:
        """Short tracebacks are returned without truncation."""
        tb = "ValueError: bad\n  at foo.py:10"
        result = check_models._format_traceback_tail(tb)
        assert result is not None
        assert "ValueError: bad" in result


class TestDiagnosticsPriority:
    """Unit tests for _diagnostics_priority."""

    def test_high_for_multi_model(self) -> None:
        """Clusters with >=2 models are High priority."""
        assert check_models._diagnostics_priority(2, "Model Error") == "High"
        assert check_models._diagnostics_priority(5, "Model Error") == "High"

    def test_low_for_weight_mismatch(self) -> None:
        """Weight mismatch and config issues are Low priority."""
        assert check_models._diagnostics_priority(1, "Weight Mismatch") == "Low"
        assert check_models._diagnostics_priority(1, "Config Missing") == "Low"

    def test_medium_default(self) -> None:
        """Single-model actionable errors default to Medium."""
        assert check_models._diagnostics_priority(1, "Model Error") == "Medium"
        assert check_models._diagnostics_priority(1, "No Chat Template") == "Medium"


class TestDiagnosticsContextBuilder:
    """Unit tests for _build_diagnostics_context."""

    def test_builds_sets_and_history_context(self) -> None:
        """Context builder should materialize comparison sets and retry stats."""
        history_records: list[HistoryRunRecord] = [
            _history_run(
                {"org/a": False, "org/b": True},
                timestamp="2026-02-01 10:00:00 GMT",
            ),
            _history_run(
                {"org/a": False, "org/b": False},
                timestamp="2026-02-02 10:00:00 GMT",
            ),
        ]
        comparison = {
            "regressions": ["org/b"],
            "recoveries": [],
            "new_models": [],
            "missing_models": ["org/c"],
        }

        context = check_models._build_diagnostics_context(
            failed_models={"org/a", "org/b"},
            history_records=history_records,
            comparison=comparison,
        )

        assert "org/b" in context.regressions
        assert "org/c" in context.missing_models
        assert context.failure_history["org/a"].first_failure_timestamp == "2026-02-01 10:00:00 GMT"
        assert context.failure_history["org/a"].recent_failures == 2
        assert context.failure_history["org/a"].recent_considered == 2
