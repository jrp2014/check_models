"""Tests for report generation edge cases (empty input, all-failed results)."""

from __future__ import annotations

import json
from argparse import Namespace
from dataclasses import dataclass
from typing import TYPE_CHECKING

from check_models import (
    DiagnosticsHistoryInputs,
    GenerationQualityAnalysis,
    LibraryVersionDict,
    PerformanceResult,
    _build_diagnostics_context,
    _cluster_failures_by_pattern,
    _diagnostics_priority,
    _format_traceback_tail,
    export_failure_repro_bundles,
    generate_diagnostics_report,
    generate_html_report,
    generate_markdown_report,
    generate_tsv_report,
)

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


def _stub_versions() -> LibraryVersionDict:
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


def _make_success(name: str = "org/model-ok") -> PerformanceResult:
    return PerformanceResult(
        model_name=name,
        success=True,
        generation=_MockGeneration(),
        total_time=1.0,
        generation_time=0.5,
        model_load_time=0.5,
    )


def _make_harness_success(
    name: str = "org/model-harness",
    *,
    text: str = "",
    prompt_tokens: int = 4000,
    generation_tokens: int = 0,
    harness_type: str = "prompt_template",
    harness_detail: str = "output:zero_tokens",
) -> PerformanceResult:
    qa = GenerationQualityAnalysis(
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
        has_harness_issue=True,
        harness_issue_type=harness_type,
        harness_issue_details=[harness_detail],
        word_count=0,
        unique_ratio=0.0,
    )
    return PerformanceResult(
        model_name=name,
        success=True,
        generation=_MockGeneration(
            text=text,
            prompt_tokens=prompt_tokens,
            generation_tokens=generation_tokens,
        ),
        total_time=1.0,
        generation_time=0.5,
        model_load_time=0.5,
        quality_issues=f"⚠️harness({harness_type})",
        quality_analysis=qa,
    )


def _make_failure(
    name: str = "org/model-fail",
    error_type: str = "ValueError",
    error_package: str = "mlx-vlm",
) -> PerformanceResult:
    return PerformanceResult(
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
        generate_html_report(
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
        generate_html_report(
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
        generate_html_report(
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
        generate_markdown_report(
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
        generate_markdown_report(
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
        generate_markdown_report(
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
        generate_tsv_report(results=[], filename=out)
        assert not out.exists()

    def test_all_failed_results_produces_file(self, tmp_path: Path) -> None:
        """All-failed result list should still produce a report."""
        out = tmp_path / "failed.tsv"
        generate_tsv_report(
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
        generate_tsv_report(
            results=[_make_success()],
            filename=out,
        )
        first_line = out.read_text(encoding="utf-8").splitlines()[0]
        assert first_line.startswith("# generated_at:")

    def test_tsv_has_error_columns(self, tmp_path: Path) -> None:
        """TSV should include error_type and error_package columns."""
        out = tmp_path / "cols.tsv"
        generate_tsv_report(
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
        generate_tsv_report(
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
) -> PerformanceResult:
    """Create a failure result with full error details for diagnostics tests."""
    return PerformanceResult(
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
        result = generate_diagnostics_report(
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
        result = generate_diagnostics_report(
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
        generate_diagnostics_report(
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
        generate_diagnostics_report(
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
        generate_diagnostics_report(
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
        generate_diagnostics_report(
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
        generate_diagnostics_report(
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
        generate_diagnostics_report(
            results=[
                _make_failure_with_details(
                    "org/m",
                    error_msg="bad shape",
                    captured_output=(
                        "=== STDERR ===\n\x1b[31mTokenizer warning here\x1b[0m\rDownloading: 100%\n"
                    ),
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
        assert "\x1b[" not in content
        assert "\r" not in content
        assert "#### `" not in content
        assert "### `org/m`" in content

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

        generate_diagnostics_report(
            results=[_make_failure_with_details("org/a", error_msg="boom")],
            filename=out,
            versions=_stub_versions(),
            system_info={},
            prompt="test",
            history=DiagnosticsHistoryInputs(
                history_path=history_path,
                previous_history=previous_record,
                current_history=current_record,
            ),
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
        generate_diagnostics_report(
            results=[_make_failure_with_details()],
            filename=out,
            versions=_stub_versions(),
            system_info={},
            prompt="test",
        )
        content = out.read_text(encoding="utf-8")
        assert "## Priority Summary" in content
        assert "| Priority | Issue |" in content

    def test_report_written_for_stack_signal_without_failures(self, tmp_path: Path) -> None:
        """Suspicious successful runs should still produce diagnostics for stack triage."""
        out = tmp_path / "diag.md"
        success = PerformanceResult(
            model_name="org/suspicious-success",
            success=True,
            generation=_MockGeneration(text="", prompt_tokens=5000, generation_tokens=0),
            total_time=1.0,
            generation_time=0.5,
            model_load_time=0.5,
        )
        result = generate_diagnostics_report(
            results=[success],
            filename=out,
            versions=_stub_versions(),
            system_info={},
            prompt="test",
        )
        assert result is True
        content = out.read_text(encoding="utf-8")
        assert "## Potential Stack Issues" in content
        assert "org/suspicious-success" in content

    def test_harness_section_includes_tokens_and_empty_marker(self, tmp_path: Path) -> None:
        """Harness section should include prompt/output token evidence for empty runs."""
        out = tmp_path / "diag.md"
        generate_diagnostics_report(
            results=[
                _make_harness_success(
                    name="org/harness-empty",
                    text="",
                    prompt_tokens=5000,
                    generation_tokens=0,
                    harness_type="long_context",
                    harness_detail="long_context_empty(5000tok)",
                ),
            ],
            filename=out,
            versions=_stub_versions(),
            system_info={},
            prompt="test",
        )
        content = out.read_text(encoding="utf-8")
        assert "## Harness/Integration Issues" in content
        assert "prompt=5,000" in content
        assert "ratio=0.00%" in content
        assert "<empty output>" in content
        assert "Likely package" in content

    def test_reproducibility_section(self, tmp_path: Path) -> None:
        """Reproducibility section should include model-specific commands."""
        out = tmp_path / "diag.md"
        generate_diagnostics_report(
            results=[_make_failure_with_details("org/broken-model")],
            filename=out,
            versions=_stub_versions(),
            system_info={},
            prompt="test",
        )
        content = out.read_text(encoding="utf-8")
        assert "## Reproducibility" in content
        assert "### Target specific failing models" in content
        assert "### Target specific failing models:" not in content
        assert "python -m check_models" in content
        assert "python -m check_models --models org/broken-model" in content

    def test_issue_template_contains_fingerprint_and_canonical_metadata(
        self, tmp_path: Path
    ) -> None:
        """Failure clusters should include copy/paste issue templates."""
        out = tmp_path / "diag.md"
        generate_diagnostics_report(
            results=[
                _make_failure_with_details(
                    "org/broken-model",
                    error_msg="got an unexpected keyword argument 'images'",
                    error_package="transformers",
                    error_stage="API Mismatch",
                    traceback_str='Traceback\nFile "x.py", line 12, in y\nTypeError: bad',
                ),
            ],
            filename=out,
            versions=_stub_versions(),
            system_info={"Python Version": "3.13", "Platform": "macOS", "GPU/Chip": "Apple M4"},
            prompt="test",
        )
        content = out.read_text(encoding="utf-8")
        assert "### Issue Template" in content
        assert "Copy/paste GitHub issue template" in content
        assert "Canonical code" in content
        assert "Signature" in content
        assert "Environment Fingerprint" in content
        assert "Minimal Reproduction" in content
        assert "transformers/issues/new" in content

    def test_failure_repro_bundles_written(self, tmp_path: Path) -> None:
        """Each failed model should produce a machine-readable repro bundle."""
        failure = _make_failure_with_details(
            "org/broken-model",
            error_msg="bad shape",
            error_stage="Model Error",
            error_package="mlx-vlm",
            traceback_str="Traceback\nValueError: bad shape",
        )
        bundles = export_failure_repro_bundles(
            results=[failure],
            output_dir=tmp_path / "repro_bundles",
            run_args=Namespace(),
            versions=_stub_versions(),
            system_info={"Python Version": "3.13"},
            prompt="Describe this image.",
            image_path=None,
        )
        assert "org/broken-model" in bundles
        bundle_path = bundles["org/broken-model"]
        assert bundle_path.exists()
        payload = json.loads(bundle_path.read_text(encoding="utf-8"))
        assert payload["model"] == "org/broken-model"
        assert payload["failure"]["stage"] == "Model Error"
        assert payload["repro"]["prompt_hash_sha256"]

    def test_diagnostics_file_ends_with_single_trailing_newline(self, tmp_path: Path) -> None:
        """Diagnostics markdown should end with a trailing newline for markdownlint."""
        out = tmp_path / "diag.md"
        generate_diagnostics_report(
            results=[_make_failure_with_details("org/broken-model")],
            filename=out,
            versions=_stub_versions(),
            system_info={},
            prompt="test",
        )
        content = out.read_text(encoding="utf-8")
        assert content.endswith("\n")

    def test_reproducibility_section_includes_image_path_when_available(
        self, tmp_path: Path
    ) -> None:
        """Repro commands should include --image when run context contains an image path."""
        out = tmp_path / "diag.md"
        image_path = tmp_path / "sample image.jpg"
        generate_diagnostics_report(
            results=[_make_failure_with_details("org/broken-model")],
            filename=out,
            versions=_stub_versions(),
            system_info={},
            prompt="test",
            image_path=image_path,
        )
        content = out.read_text(encoding="utf-8")
        assert f"--image '{image_path}'" in content

    def test_reproducibility_section_includes_sampling_and_runtime_flags(
        self, tmp_path: Path
    ) -> None:
        """Reproducibility should include the key generation/runtime CLI settings used."""
        out = tmp_path / "diag.md"
        run_args = Namespace(
            image=None,
            folder=None,
            models=None,
            exclude=None,
            trust_remote_code=False,
            revision=None,
            adapter_path=None,
            prompt=None,
            detailed_metrics=False,
            max_tokens=123,
            temperature=0.7,
            top_p=0.92,
            repetition_penalty=1.1,
            repetition_context_size=50,
            lazy_load=False,
            max_kv_size=None,
            kv_bits=None,
            kv_group_size=64,
            quantized_kv_start=0,
            prefill_step_size=None,
            timeout=42.0,
            verbose=True,
            no_color=False,
            force_color=False,
            width=None,
            quality_config=None,
            context_marker="Context:",
        )
        generate_diagnostics_report(
            results=[_make_failure_with_details("org/broken-model")],
            filename=out,
            versions=_stub_versions(),
            system_info={},
            prompt="test",
            run_args=run_args,
        )
        content = out.read_text(encoding="utf-8")
        assert "--max-tokens 123" in content
        assert "--temperature 0.7" in content
        assert "--top-p 0.92" in content
        assert "--repetition-penalty 1.1" in content
        assert "--repetition-context-size 50" in content
        assert "--timeout 42.0" in content
        assert "--no-trust-remote-code" in content
        assert "--verbose" in content

    def test_reproducibility_lists_each_failing_model(self, tmp_path: Path) -> None:
        """Targeted repro commands should include every failed model (no truncation)."""
        out = tmp_path / "diag.md"
        generate_diagnostics_report(
            results=[
                _make_failure_with_details("org/a"),
                _make_failure_with_details("org/b"),
            ],
            filename=out,
            versions=_stub_versions(),
            system_info={},
            prompt="test",
        )
        content = out.read_text(encoding="utf-8")
        assert "python -m check_models --models org/a" in content
        assert "python -m check_models --models org/b" in content

    def test_prompt_in_details_block(self, tmp_path: Path) -> None:
        """Prompt should be in a collapsible details block."""
        out = tmp_path / "diag.md"
        generate_diagnostics_report(
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
        generate_diagnostics_report(
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
        clusters = _cluster_failures_by_pattern(results)
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
        clusters = _cluster_failures_by_pattern(results)
        assert len(clusters) == 1
        assert len(next(iter(clusters.values()))) == 2

    def test_different_patterns_separate(self) -> None:
        """Fundamentally different errors should not cluster together."""
        results = [
            _make_failure_with_details("org/a", error_msg="chat_template is not set"),
            _make_failure_with_details("org/b", error_msg="Missing 1 parameters"),
        ]
        clusters = _cluster_failures_by_pattern(results)
        assert len(clusters) == 2


class TestFormatTracebackTail:
    """Unit tests for _format_traceback_tail."""

    def test_none_input(self) -> None:
        """None input returns None."""
        assert _format_traceback_tail(None) is None

    def test_empty_string(self) -> None:
        """Empty string returns None."""
        assert _format_traceback_tail("") is None

    def test_extracts_tail(self) -> None:
        """Long tracebacks are truncated to the tail lines."""
        tb = "\n".join(f"line {i}" for i in range(20))
        result = _format_traceback_tail(tb)
        assert result is not None
        lines = result.splitlines()
        assert len(lines) == 6  # _DIAGNOSTICS_TRACEBACK_TAIL_LINES
        assert "line 19" in lines[-1]

    def test_short_traceback_returned_fully(self) -> None:
        """Short tracebacks are returned without truncation."""
        tb = "ValueError: bad\n  at foo.py:10"
        result = _format_traceback_tail(tb)
        assert result is not None
        assert "ValueError: bad" in result


class TestDiagnosticsPriority:
    """Unit tests for _diagnostics_priority."""

    def test_high_for_multi_model(self) -> None:
        """Clusters with >=2 models are High priority."""
        assert _diagnostics_priority(2, "Model Error") == "High"
        assert _diagnostics_priority(5, "Model Error") == "High"

    def test_low_for_weight_mismatch(self) -> None:
        """Weight mismatch and config issues are Low priority."""
        assert _diagnostics_priority(1, "Weight Mismatch") == "Low"
        assert _diagnostics_priority(1, "Config Missing") == "Low"

    def test_medium_default(self) -> None:
        """Single-model actionable errors default to Medium."""
        assert _diagnostics_priority(1, "Model Error") == "Medium"
        assert _diagnostics_priority(1, "No Chat Template") == "Medium"


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

        context = _build_diagnostics_context(
            failed_models={"org/a", "org/b"},
            history_records=history_records,
            comparison=comparison,
        )

        assert "org/b" in context.regressions
        assert "org/c" in context.missing_models
        assert context.failure_history["org/a"].first_failure_timestamp == "2026-02-01 10:00:00 GMT"
        assert context.failure_history["org/a"].recent_failures == 2
        assert context.failure_history["org/a"].recent_considered == 2
