"""Tests for report generation edge cases (empty input, all-failed results)."""

from __future__ import annotations

import json
import os
import time
from argparse import Namespace
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import check_models
from check_models import (
    DiagnosticsArtifacts,
    DiagnosticsHistoryInputs,
    DiagnosticsSnapshot,
    GenerationQualityAnalysis,
    LibraryVersionDict,
    PerformanceResult,
    RuntimeDiagnostics,
    _append_review_user_buckets,
    _build_diagnostics_context,
    _build_diagnostics_snapshot,
    _build_report_render_context,
    _clean_stale_toplevel_reports,
    _cluster_failures_by_pattern,
    _format_traceback_tail,
    _generate_github_issue_reports,
    _log_maintainer_summary,
    _prune_repro_bundles,
    export_failure_repro_bundles,
    generate_diagnostics_report,
    generate_html_report,
    generate_markdown_gallery_report,
    generate_markdown_report,
    generate_review_report,
    generate_tsv_report,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import pytest

    from check_models import HistoryModelResultRecord, HistoryRunRecord

THINKING_START_TOKEN = "<think>"
THINKING_END_TOKEN = "</think>"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _MockGeneration:
    """Minimal stand-in for GenerationResult used by report generators."""

    text: str | None = "output"
    token: object | None = None
    logprobs: object | None = None
    prompt_tokens: int | None = 10
    generation_tokens: int | None = 5
    total_tokens: int | None = 15
    prompt_tps: float | None = 1200.0
    generation_tps: float | None = 80.0
    peak_memory: float | None = 4.5
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


def _extract_markdown_subsection(
    content: str,
    heading: str,
    *,
    end_headings: Sequence[str],
) -> str:
    start = content.index(heading)
    end_positions = [
        content.find(candidate, start + len(heading))
        for candidate in end_headings
        if content.find(candidate, start + len(heading)) != -1
    ]
    end = min(end_positions) if end_positions else len(content)
    return content[start:end]


def _make_success(name: str = "org/model-ok") -> PerformanceResult:
    return PerformanceResult(
        model_name=name,
        success=True,
        generation=_MockGeneration(
            text=(
                "Title: Brick storefront with outdoor seating\n"
                "Description: A brick storefront has outdoor seating beside a sidewalk. "
                "People sit outside under clear daylight.\n"
                "Keywords: brick storefront, outdoor seating, sidewalk, people, daylight, "
                "sign, windows, street, town, facade"
            ),
            prompt_tokens=120,
            generation_tokens=48,
        ),
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
        prompt_checks_ran=True,
        verdict="harness" if harness_type != "long_context" else "context_budget",
        owner="mlx-vlm" if harness_type != "long_context" else "mlx",
        user_bucket="avoid" if harness_type != "long_context" else "caveat",
        evidence=[f"harness:{harness_type}"],
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


def _make_quality_success(
    name: str,
    *,
    with_quality_issue: bool,
) -> PerformanceResult:
    """Create a successful result with explicit quality analysis state."""
    qa = GenerationQualityAnalysis(
        is_repetitive=False,
        repeated_token=None,
        hallucination_issues=[],
        is_verbose=False,
        formatting_issues=["Formatting marker leak"] if with_quality_issue else [],
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
        has_harness_issue=False,
        harness_issue_type=None,
        harness_issue_details=[],
        word_count=20,
        unique_ratio=0.9,
        prompt_checks_ran=True,
    )
    return PerformanceResult(
        model_name=name,
        success=True,
        generation=_MockGeneration(text="quality output", prompt_tokens=120, generation_tokens=80),
        total_time=1.0,
        generation_time=0.6,
        model_load_time=0.4,
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


def test_build_report_render_context_backfills_quality_analysis() -> None:
    """Shared report context should populate missing quality analysis for successful results."""
    result = _make_success("org/model-clean")
    assert result.quality_analysis is None

    context = _build_report_render_context(results=[result], prompt="Describe this image.")

    populated = context.result_set.results[0]
    assert populated.quality_analysis is not None


def test_build_report_render_context_refreshes_prompt_dependent_checks() -> None:
    """Prompt-aware report rendering should upgrade prompt-less cached analyses."""
    prompt = (
        "Analyze this image.\n"
        "Context: Existing metadata hints:\n"
        "- Title hint: Brick storefront with outdoor seating\n"
        "- Description hint: A brick storefront has outdoor seating beside a sidewalk.\n"
    )
    echoed_text = (
        "Context: Existing metadata hints:\n"
        "Title hint: Brick storefront with outdoor seating\n"
        "Description hint: A brick storefront has outdoor seating beside a sidewalk."
    )
    stale_analysis = check_models.analyze_generation_text(
        echoed_text,
        generated_tokens=32,
    )
    result = PerformanceResult(
        model_name="org/model-echo",
        success=True,
        generation=_MockGeneration(text=echoed_text, generation_tokens=32),
        total_time=1.0,
        generation_time=0.5,
        model_load_time=0.5,
        quality_analysis=stale_analysis,
    )

    context = _build_report_render_context(results=[result], prompt=prompt)

    populated = context.result_set.results[0]
    assert populated.quality_analysis is not None
    assert populated.quality_analysis.prompt_checks_ran is True
    assert populated.quality_analysis.has_context_echo is True


def test_unflagged_models_section_marks_prompt_incomplete_analysis() -> None:
    """Diagnostics should avoid labeling prompt-less cached analysis as clean output."""
    stale_analysis = check_models.analyze_generation_text(
        "Title: Brick storefront with outdoor seating",
        generated_tokens=18,
    )
    result = PerformanceResult(
        model_name="org/promptless",
        success=True,
        generation=_MockGeneration(
            text="Title: Brick storefront with outdoor seating",
            generation_tokens=18,
        ),
        total_time=1.0,
        generation_time=0.5,
        model_load_time=0.5,
        quality_analysis=stale_analysis,
    )

    section = check_models._diagnostics_unflagged_success_section(
        unflagged_successful=[result],
    )
    content = "\n".join(section)

    assert "Passed (prompt-dependent quality checks unavailable)" in content
    assert "`org/promptless`" in content


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

    def test_html_report_includes_shared_triage_sections(self, tmp_path: Path) -> None:
        """HTML report should reuse shared action, review, and failure sections."""
        out = tmp_path / "triage.html"
        results = [
            _make_success("org/good"),
            _make_harness_success("org/risky"),
            _make_failure("org/bad", error_package="transformers"),
        ]
        report_context = _build_report_render_context(results=results, prompt="describe")

        generate_html_report(
            results=results,
            filename=out,
            versions=_stub_versions(),
            prompt="describe",
            total_runtime_seconds=3.0,
            report_context=report_context,
        )

        content = out.read_text(encoding="utf-8")
        assert "Action Snapshot" in content
        assert "Review Shortlist" in content
        assert "Failures by Package" in content
        assert "org/risky" in content
        assert "transformers" in content

    def test_html_report_includes_preflight_guidance_in_action_snapshot(
        self, tmp_path: Path
    ) -> None:
        """HTML report should explain how to interpret preflight compatibility warnings."""
        out = tmp_path / "preflight-triage.html"
        results = [_make_success("org/good")]
        report_context = _build_report_render_context(
            results=results,
            prompt="describe",
            preflight_issues=(
                "transformers==5.4.0 is below minimum 5.5.3 required by check_models.",
            ),
        )

        generate_html_report(
            results=results,
            filename=out,
            versions=_stub_versions(),
            prompt="describe",
            total_runtime_seconds=1.0,
            report_context=report_context,
        )

        content = out.read_text(encoding="utf-8")
        assert "Preflight compatibility" in content
        assert "informational warning(s); do not treat these alone as run failures" in content
        assert "API mismatches, startup hangs, or backend/runtime crashes" in content

    def test_html_report_preserves_full_output_in_details(self, tmp_path: Path) -> None:
        """HTML table should reuse the shared preview while keeping full text expandable."""
        out = tmp_path / "details.html"
        long_text = (
            "START " + ("filler " * 70) + "MIDDLE-MARKER " + ("more filler " * 70) + "END-MARKER"
        )
        results = [
            PerformanceResult(
                model_name="org/preview-model",
                success=True,
                generation=_MockGeneration(text=long_text),
                total_time=1.0,
                generation_time=0.5,
                model_load_time=0.5,
                quality_issues="context-echo",
            ),
        ]

        generate_html_report(
            results=results,
            filename=out,
            versions=_stub_versions(),
            prompt="describe",
            total_runtime_seconds=1.0,
        )

        content = out.read_text(encoding="utf-8")
        assert "context-echo" in content
        assert "[tail]" in content
        assert "MIDDLE-MARKER" in content
        assert "END-MARKER" in content


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
        assert "## 🎯 Action Snapshot" in content
        assert "org/good" in content
        assert "org/bad" in content

    def test_markdown_report_includes_preflight_guidance_in_action_snapshot(
        self,
        tmp_path: Path,
    ) -> None:
        """Markdown report should explain how to interpret preflight compatibility warnings."""
        out = tmp_path / "preflight-triage.md"
        results = [_make_success("org/good")]
        report_context = _build_report_render_context(
            results=results,
            prompt="describe",
            preflight_issues=(
                "transformers==5.4.0 is below minimum 5.5.3 required by check_models.",
            ),
        )

        generate_markdown_report(
            results=results,
            filename=out,
            versions=_stub_versions(),
            prompt="describe",
            total_runtime_seconds=1.0,
            report_context=report_context,
        )

        content = out.read_text(encoding="utf-8")
        assert "## 🎯 Action Snapshot" in content
        assert "_Preflight compatibility:_ 1 informational warning(s);" in content
        assert "do not treat" in content
        assert "run failures." in content
        assert "_Escalate only if:_" in content
        assert "API mismatches" in content
        assert "backend/runtime crashes." in content

    def test_prompt_section_uses_wrapped_blockquote(self, tmp_path: Path) -> None:
        """Prompt section should use the wrapped blockquote helper for readable Markdown."""
        out = tmp_path / "blockquote.md"
        generate_markdown_report(
            results=[_make_success("org/good")],
            filename=out,
            versions=_stub_versions(),
            prompt="line one\n\nline two",
            total_runtime_seconds=1.0,
        )
        content = out.read_text(encoding="utf-8")
        assert "_Prompt used:_" in content
        assert "<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->" in content
        assert "> [!NOTE]" not in content
        assert "> line one" in content
        assert "\n>\n> line two" in content
        assert "```text" not in content
        assert "> _Prompt used:_" not in content

    def test_report_links_to_dedicated_gallery_artifact(self, tmp_path: Path) -> None:
        """Main markdown report should point readers at the standalone gallery artifact."""
        out = tmp_path / "results.md"
        gallery = tmp_path / "model_gallery.md"
        review = tmp_path / "review.md"
        log_file = tmp_path / "check_models.log"
        generate_markdown_report(
            results=[_make_quality_success("org/good", with_quality_issue=True)],
            filename=out,
            versions=_stub_versions(),
            prompt="describe",
            total_runtime_seconds=1.0,
            gallery_filename=gallery,
            review_filename=review,
            log_filename=log_file,
        )
        content = out.read_text(encoding="utf-8")
        assert "Review artifacts:" in content
        assert "_Review artifacts:_\n\n- _Standalone output gallery:_" in content
        assert "Standalone output gallery" in content
        assert "Automated review digest" in content
        assert "Canonical run log" in content
        assert "[model_gallery.md](model_gallery.md)" in content
        assert "[review.md](review.md)" in content
        assert "[check_models.log](check_models.log)" in content
        assert "## Model Gallery" not in content
        assert "## ✅ Recommended Models" in content
        assert "_Best end-to-end cataloging:_" in content
        assert "_Best descriptions:_" in content
        assert "_Best keywording:_" in content
        assert "## 🔍 Quality Pattern Breakdown" in content

    def test_markdown_report_uses_shared_output_preview_text(self) -> None:
        """Markdown compact views should rely on the shared preview builder semantics."""
        long_text = "Start of answer. " + ("filler text " * 40) + "TRAILING-SIGNAL"
        result = PerformanceResult(
            model_name="org/preview-model",
            success=True,
            generation=_MockGeneration(text=long_text),
            total_time=1.0,
            generation_time=0.5,
            model_load_time=0.5,
            quality_issues="context-echo, reasoning-leak",
        )

        preview = check_models._build_result_output_preview(result)
        assert "[context-echo; reasoning-leak]" in preview
        assert "[tail]" in preview
        assert "TRAILING-SIGNAL" in preview


class TestMarkdownGalleryReport:
    """Coverage for the standalone markdown gallery artifact."""

    def test_empty_results_does_not_write(self, tmp_path: Path) -> None:
        """Empty result list should produce no gallery file."""
        out = tmp_path / "model_gallery.md"
        generate_markdown_gallery_report(
            results=[],
            filename=out,
            prompt="unused",
        )
        assert not out.exists()

    def test_gallery_includes_metadata_prompt_and_models(self, tmp_path: Path) -> None:
        """Gallery artifact should include selected metadata, prompt, and model sections."""
        out = tmp_path / "model_gallery.md"
        generate_markdown_gallery_report(
            results=[_make_success("org/good"), _make_failure("org/bad")],
            filename=out,
            prompt="Describe this image fully.",
            metadata={
                "title": "Harbor Sunset",
                "description": "Fishing boats at dusk.",
                "keywords": "harbor, boats, sunset",
                "date": "2026-03-08",
                "time": "18:42:00",
                "gps": "51.5000, -0.1200",
                "exif": "ignored raw blob",
            },
        )
        content = out.read_text(encoding="utf-8")
        assert "# Model Output Gallery" in content
        assert "## Image Metadata" in content
        assert "_Title:_ Harbor Sunset" in content
        assert "_Description:_ Fishing boats at dusk." in content
        assert "_Keywords:_ harbor, boats, sunset" in content
        assert "_Date:_ 2026-03-08" in content
        assert "_Time:_ 18:42:00" in content
        assert "_GPS:_ 51.5000, -0.1200" in content
        assert "ignored raw blob" not in content
        assert "## Prompt" in content
        assert "## Quick Navigation" in content
        assert "<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->" in content
        assert "> [!NOTE]" not in content
        assert "Describe this image fully." in content
        assert "```text" not in content
        assert '<a id="model-org-good"></a>' in content
        assert "_Recommendation:_" in content
        assert "_Owner:_" in content
        assert "_Next step:_" in content
        assert "### ✅ org/good" in content
        assert "### ❌ org/bad" in content

    def test_review_report_groups_owner_and_user_buckets(self, tmp_path: Path) -> None:
        """Review digest should group maintainer ownership and user-facing buckets."""
        out = tmp_path / "review.md"
        log_file = tmp_path / "check_models.log"
        gallery = tmp_path / "model_gallery.md"
        results = [
            _make_success("org/good"),
            _make_harness_success(
                "org/risky", harness_type="stop_token", harness_detail="token_leak:<s>"
            ),
            _make_failure("org/bad", error_package="transformers"),
        ]
        report_context = _build_report_render_context(results=results, prompt="describe")

        generate_review_report(
            results=results,
            filename=out,
            prompt="describe",
            report_context=report_context,
            log_filename=log_file,
            gallery_filename=gallery,
        )

        content = out.read_text(encoding="utf-8")
        assert "# Automated Review Digest" in content
        assert "## Maintainer Escalations" in content
        assert "issues/index.md" in content
        assert "## 🧭 Review Shortlist" in content
        assert "## User Buckets" in content
        assert "## Model Verdicts" in content
        assert "## Maintainer Queue" not in content
        assert "`mlx-vlm`" in content or "`transformers`" in content
        assert "`recommended`" in content
        assert "`avoid`" in content
        assert content.index("## User Buckets") < content.index("## Maintainer Escalations")
        assert content.index("## Maintainer Escalations") < content.index("## Model Verdicts")
        assert "Model" in content
        assert "Hint Handling" in content
        assert "Key Evidence" in content
        assert "Evidence Bundle" in content
        assert "Fixed When" in content
        assert "Canonical run log" in content
        maintainer_queue = content.split("## Maintainer Escalations", maxsplit=1)[1].split(
            "## Model Verdicts",
            maxsplit=1,
        )[0]
        assert "org/good" not in maintainer_queue
        assert "org/risky" in maintainer_queue
        assert "org/bad" in maintainer_queue

    def test_gallery_includes_shared_triage_sections_and_review_status(
        self,
        tmp_path: Path,
    ) -> None:
        """Gallery should surface shared triage sections and per-model review status."""
        out = tmp_path / "triage_gallery.md"
        results = [
            _make_success("org/good"),
            _make_harness_success("org/risky"),
            _make_failure("org/bad", error_package="mlx-vlm"),
        ]
        report_context = _build_report_render_context(results=results, prompt="describe")

        generate_markdown_gallery_report(
            results=results,
            filename=out,
            prompt="Describe this image fully.",
            report_context=report_context,
        )

        content = out.read_text(encoding="utf-8")
        assert "Action Snapshot" in content  # cross-reference to results.md
        assert "## 🧭 Review Shortlist" in content
        assert "## 🚨 Failures by Package (Actionable)" in content
        assert "_Review focus:_" in content
        assert "strong candidate for first-pass review" in content or "watchlist" in content
        assert "_Error summary:_" in content
        assert "_Next step:_" in content


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
    generated_text: str | None = None,
) -> PerformanceResult:
    """Create a failure result with full error details for diagnostics tests."""
    generation = (
        _MockGeneration(text=generated_text, prompt_tokens=32, generation_tokens=16)
        if generated_text is not None
        else None
    )
    return PerformanceResult(
        model_name=name,
        success=False,
        generation=generation,
        error_stage=error_stage,
        error_message=error_msg,
        error_type=error_type,
        error_package=error_package,
        captured_output_on_fail=captured_output,
        error_traceback=traceback_str,
    )


class TestDiagnosticsReport:
    """Tests for generate_diagnostics_report and its helpers."""

    def test_classify_review_owner_prefers_failure_owner_then_harness(self) -> None:
        """Review ownership should keep failure-owner precedence and harness fallbacks."""
        assert (
            check_models._classify_review_owner(
                harness_type="long_context",
                failure_owner="transformers / mlx-vlm",
            )
            == "transformers"
        )
        assert (
            check_models._classify_review_owner(
                harness_type="prompt_template",
                failure_owner=None,
            )
            == "model-config"
        )
        assert (
            check_models._classify_review_owner(
                harness_type=None,
                failure_owner="huggingface-hub",
            )
            == "huggingface-hub"
        )

    def test_review_next_action_calls_out_hub_connectivity_failures(self) -> None:
        """Canonical review text should flag transient Hub disconnects explicitly."""
        action = check_models._review_next_action_text(
            {
                "verdict": "harness",
                "hint_relationship": "preserves_trusted_hints",
                "instruction_echo": False,
                "metadata_borrowing": False,
                "likely_capped": False,
                "owner": "huggingface-hub",
                "user_bucket": "avoid",
                "evidence": [
                    "model_error",
                    "huggingface_hub_model_load_model",
                    "hub_connectivity",
                ],
                "requested_max_tokens": 100,
                "hit_max_tokens": False,
                "prompt_tokens_total": None,
                "prompt_tokens_text_est": None,
                "prompt_tokens_nontext_est": None,
                "prompt_output_ratio": None,
                "nontext_prompt_ratio": None,
                "missing_terms": [],
                "missing_sections": [],
                "harness_details": [],
            },
        )
        assert "Hugging Face was reachable" in action
        assert "transient Hub/network outage" in action

    def test_diagnostics_next_action_uses_composite_owner_rules(self) -> None:
        """Composite owner keys should keep their specialized next-step guidance."""
        assert (
            check_models._diagnostics_next_action("model-config / mlx-vlm")
            == "validate chat-template/config expectations and mlx-vlm prompt formatting for this model."
        )
        assert (
            check_models._diagnostics_next_action("mlx-vlm / mlx")
            == "validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime."
        )

    def test_infer_harness_issue_owner_uses_detail_prefix_fallbacks(self) -> None:
        """Harness owner inference should still honor detail-prefix fallbacks."""
        training_leak = _make_harness_success(
            "org/harness-generation-loop",
            harness_type="generation_loop",
            harness_detail="training_leak:instruction_header",
        )
        output_shape = _make_harness_success(
            "org/harness-output-shape",
            harness_type="stop_token",
            harness_detail="output:zero_tokens",
        )

        assert check_models._infer_harness_issue_owner(training_leak) == "mlx-vlm / mlx-lm"
        assert check_models._infer_harness_issue_owner(output_shape) == "model-config / mlx-vlm"

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

    def test_report_written_for_preflight_warnings_only(self, tmp_path: Path) -> None:
        """Preflight compatibility warnings should be captured in diagnostics output."""
        out = tmp_path / "diag.md"
        result = generate_diagnostics_report(
            results=[_make_success()],
            filename=out,
            versions=_stub_versions(),
            system_info={"Python Version": "3.13"},
            prompt="test",
            history=DiagnosticsHistoryInputs(
                preflight_issues=(
                    "transformers==5.4.0 is below minimum 5.5.3 required by check_models.",
                ),
            ),
        )
        assert result is True
        content = out.read_text(encoding="utf-8")
        assert "## Preflight Compatibility Warnings" in content
        assert "transformers==5.4.0 is below minimum 5.5.3 required by check_models." in content
        assert "They are informational by" in content
        assert "default and do not invalidate successful runs on their own." in content
        assert "transformers/issues/new" in content
        assert "These warnings were detected before inference." in content

    def test_preflight_section_splits_mixed_owner_groups(self, tmp_path: Path) -> None:
        """Mixed preflight owners should render as separate detailed owner buckets."""
        out = tmp_path / "diag.md"
        result = generate_diagnostics_report(
            results=[_make_success()],
            filename=out,
            versions=_stub_versions(),
            system_info={"Python Version": "3.13"},
            prompt="test",
            history=DiagnosticsHistoryInputs(
                preflight_issues=(
                    "transformers==5.4.0 is below minimum 5.5.3 required by check_models.",
                    "mlx runtime probe reported a suspicious cache incompatibility",
                ),
            ),
        )
        assert result is True
        content = out.read_text(encoding="utf-8")
        assert "### `transformers`" in content
        assert "### `mlx`" in content
        assert "verify API compatibility and pinned version floor." in content
        assert "mlx runtime probe reported a suspicious cache incompatibility" in content
        assert "transformers/issues/new" in content

    def test_report_written_for_stack_signal_uses_preflight_owner_hint(
        self,
        tmp_path: Path,
    ) -> None:
        """Stack-signal rows should reuse matching preflight ownership hints when available."""
        out = tmp_path / "diag.md"
        success = PerformanceResult(
            model_name="org/stack-transformers",
            success=True,
            generation=_MockGeneration(
                text="echoed context",
                prompt_tokens=15000,
                generation_tokens=80,
            ),
            total_time=1.0,
            generation_time=0.5,
            model_load_time=0.5,
            quality_analysis=GenerationQualityAnalysis(
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
                has_reasoning_leak=False,
                reasoning_leak_markers=[],
                has_context_echo=True,
                context_echo_ratio=0.9,
                has_harness_issue=False,
                harness_issue_type=None,
                harness_issue_details=[],
                word_count=80,
                unique_ratio=0.3,
                prompt_checks_ran=True,
            ),
        )

        result = generate_diagnostics_report(
            results=[success],
            filename=out,
            versions=_stub_versions(),
            system_info={"Python Version": "3.13"},
            prompt="test",
            history=DiagnosticsHistoryInputs(
                preflight_issues=(
                    "transformers==5.4.0 is below minimum 5.5.3 required by check_models.",
                ),
            ),
        )

        assert result is True
        content = out.read_text(encoding="utf-8")
        assert "`transformers / mlx-vlm`" in content
        assert "Long-Context Degradation / Potential Stack Issues" in content

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

    def test_action_summary_escapes_double_underscore_error_text(self, tmp_path: Path) -> None:
        """Diagnostics action summary should escape double underscores in failure text."""
        out = tmp_path / "diag.md"
        generate_diagnostics_report(
            results=[
                _make_failure_with_details(
                    "org/bad-model",
                    error_msg=(
                        "Failed to process inputs with error: "
                        "ImagesKwargs.__init__() got an unexpected keyword argument"
                    ),
                    error_package="transformers",
                    error_stage="Processor Error",
                ),
            ],
            filename=out,
            versions=_stub_versions(),
            system_info={"Python Version": "3.13"},
            prompt="test prompt",
        )
        content = out.read_text(encoding="utf-8")
        assert "ImagesKwargs.\\_\\_init\\_\\_()" in content

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
        assert "mlx-vlm" in content
        assert "0.3.11" in content
        assert "Python Version" in content
        assert "3.13.9" in content
        assert "GPU/Chip" in content
        assert "Apple M4 Max" in content

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
        # Count the number of "##" headings that mention grouped failure heading
        error_sections = [
            ln for ln in content.splitlines() if ln.startswith("## ") and "Failure affecting" in ln
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
        assert "chat_template is not set" in content
        assert "Missing 1 parameters: lm_head.weight" in content

    def test_traceback_tail_included(self, tmp_path: Path) -> None:
        """Detailed trace dropdown should include traceback text."""
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
        assert "Detailed trace logs (affected model)" in content
        assert "ValueError: bad shape" in content

    def test_traceback_omits_local_check_models_frames(self, tmp_path: Path) -> None:
        """Traceback rendering should omit local check_models frames from issue output."""
        tb = (
            "Traceback (most recent call last):\n"
            '  File "/Users/test/check_models.py", line 99, in local_runner\n'
            "    local_wrapper()\n"
            '  File "/opt/homebrew/lib/python3.13/site-packages/mlx_vlm/utils.py", '
            "line 10, in process_inputs\n"
            "    raise ValueError('bad shape')\n"
            "ValueError: bad shape\n"
        )
        out = tmp_path / "diag.md"
        generate_diagnostics_report(
            results=[_make_failure_with_details("org/m", traceback_str=tb, error_msg="bad shape")],
            filename=out,
            versions=_stub_versions(),
            system_info={},
            prompt="test",
        )
        content = out.read_text(encoding="utf-8")
        assert "/Users/test/check_models.py" not in content
        assert "local_wrapper()" not in content
        assert "mlx_vlm/utils.py" in content

    def test_traceback_tail_used_in_collapsed_section(self, tmp_path: Path) -> None:
        """Diagnostics should include only the traceback tail in collapsed details."""
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
        assert "Detailed trace logs (affected model)" in content
        assert "<details>" in content
        assert "Traceback tail:" in content
        assert "trace-line-0" not in content
        assert "trace-line-6" in content
        assert "trace-line-11" in content

    def test_captured_output_in_collapsed_section(self, tmp_path: Path) -> None:
        """Diagnostics should include captured stdout/stderr in the detailed log section."""
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
        assert "Detailed trace logs (affected model)" in content
        assert "Tokenizer warning here" in content
        assert "Downloading: 100%" not in content
        assert "\x1b[" not in content
        assert "\r" not in content
        assert "#### `org/m`" in content

    def test_history_context_includes_regressions_recoveries_and_repro(
        self,
        tmp_path: Path,
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

    def test_issue_queue_present(self, tmp_path: Path) -> None:
        """Issue Queue table should appear with clustered issue draft links."""
        out = tmp_path / "diag.md"
        generate_diagnostics_report(
            results=[_make_failure_with_details()],
            filename=out,
            versions=_stub_versions(),
            system_info={},
            prompt="test",
        )
        content = out.read_text(encoding="utf-8")
        assert "## Issue Queue" in content
        assert "Issue Draft" in content
        assert "Target" in content
        assert "Evidence Bundle" in content
        assert "Fixed When" in content
        assert "issues/index.md" in content
        assert "Priority" not in content
        assert content.index("## Issue Queue") < content.index("## 1. Failure")
        assert content.index("## Issue Queue") < content.index("## Environment")
        assert content.index("## Environment") < content.index("## Reproducibility")

    def test_action_summary_and_portable_triage_sections_present(self, tmp_path: Path) -> None:
        """Diagnostics should include compact action triage and portable probe commands."""
        out = tmp_path / "diag.md"
        image_path = tmp_path / "sample image.jpg"
        image_path.write_text("placeholder", encoding="utf-8")
        adapter_path = tmp_path / "portable-adapter"
        run_args = Namespace(
            image=image_path,
            folder=None,
            models=None,
            exclude=None,
            trust_remote_code=False,
            revision="main",
            adapter_path=adapter_path,
            prompt=None,
            detailed_metrics=False,
            resize_shape=None,
            eos_tokens=None,
            skip_special_tokens=False,
            processor_kwargs=None,
            enable_thinking=False,
            thinking_budget=None,
            thinking_start_token=None,
            thinking_end_token=None,
            max_tokens=123,
            temperature=0.7,
            top_p=0.92,
            min_p=0.08,
            top_k=24,
            repetition_penalty=1.1,
            repetition_context_size=50,
            lazy_load=False,
            max_kv_size=None,
            kv_bits=None,
            kv_quant_scheme=check_models.DEFAULT_KV_QUANT_SCHEME,
            prefill_step_size=16,
            kv_group_size=64,
            quantized_kv_start=2048,
            timeout=42.0,
            verbose=True,
            no_color=False,
            force_color=False,
            width=None,
            quality_config=None,
            context_marker="Context:",
        )
        stack_signal = PerformanceResult(
            model_name="org/stack-probe",
            success=True,
            generation=_MockGeneration(
                text="echoed context",
                prompt_tokens=15000,
                generation_tokens=80,
            ),
            total_time=1.0,
            generation_time=0.5,
            model_load_time=0.5,
            quality_analysis=GenerationQualityAnalysis(
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
                has_reasoning_leak=False,
                reasoning_leak_markers=[],
                has_context_echo=True,
                context_echo_ratio=0.94,
                has_harness_issue=False,
                harness_issue_type=None,
                harness_issue_details=[],
                word_count=80,
                unique_ratio=0.3,
                prompt_checks_ran=True,
            ),
        )
        generate_diagnostics_report(
            results=[
                _make_failure_with_details("org/broken-model"),
                _make_harness_success("org/harness-probe"),
                stack_signal,
            ],
            filename=out,
            versions=_stub_versions(),
            system_info={},
            prompt="test",
            image_path=image_path,
            run_args=run_args,
        )
        content = out.read_text(encoding="utf-8")
        portable_block = _extract_markdown_subsection(
            content,
            "### Portable upstream probes (no local image required)",
            end_headings=("### Target specific failing models",),
        )
        assert "## Upstream Filing Notes" in content
        assert "## Appendix" in content
        assert "### Portable upstream probes (no local image required)" in content
        assert "mlx_vlm.utils.load" in portable_block
        assert "org/broken-model" in portable_block
        assert "org/harness-probe" in portable_block
        assert "org/stack-probe" in portable_block
        assert "check_models_portable_probe.png" in portable_block
        assert str(image_path) not in portable_block
        assert "python -m pip show mlx mlx-vlm mlx-lm transformers" in content

    def test_unflagged_models_section_lists_successes(self, tmp_path: Path) -> None:
        """Diagnostics should backfill quality analysis for clean successful models."""
        out = tmp_path / "diag.md"
        generate_diagnostics_report(
            results=[
                _make_failure_with_details("org/fail"),
                _make_success("org/pass-a"),
                _make_success("org/pass-b"),
            ],
            filename=out,
            versions=_stub_versions(),
            system_info={},
            prompt="test",
        )
        content = out.read_text(encoding="utf-8")
        assert "## Models Not Flagged (2 model(s))" in content
        assert "### Clean output (2 model(s))" in content
        assert "`org/pass-a`" in content
        assert "`org/pass-b`" in content

    def test_build_diagnostics_snapshot_classifies_prompt_backfilled_harness_run(self) -> None:
        """Prompt-aware snapshot building should classify harness issues without cached analysis."""
        failure = _make_failure_with_details("org/fail")
        harness_candidate = PerformanceResult(
            model_name="org/harness-implicit",
            success=True,
            generation=_MockGeneration(text="", prompt_tokens=5000, generation_tokens=0),
            total_time=1.0,
            generation_time=0.5,
            model_load_time=0.5,
        )

        snapshot = _build_diagnostics_snapshot(
            results=[failure, harness_candidate],
            prompt="Describe the image.",
        )

        assert len(snapshot.harness_results) == 1
        assert snapshot.harness_results[0][0].model_name == "org/harness-implicit"
        assert not snapshot.unflagged_successful

    def test_unflagged_models_section_categorizes_by_quality(self, tmp_path: Path) -> None:
        """Unflagged successful models should be grouped by available quality signals."""
        out = tmp_path / "diag.md"
        generate_diagnostics_report(
            results=[
                _make_failure_with_details("org/fail"),
                _make_quality_success("org/clean", with_quality_issue=False),
                _make_quality_success("org/warn", with_quality_issue=True),
            ],
            filename=out,
            versions=_stub_versions(),
            system_info={},
            prompt="test",
        )
        content = out.read_text(encoding="utf-8")
        assert "## Models Not Flagged (2 model(s))" in content
        assert "### Clean output (1 model(s))" in content
        assert "### Ran, but with quality warnings (1 model(s))" in content
        assert "`org/clean`" in content
        assert "`org/warn`:" in content
        assert "Output formatting deviated from the requested structure." in content

    def test_coverage_and_runtime_metrics_section(self, tmp_path: Path) -> None:
        """Diagnostics should verify exclusive model coverage and report aggregate runtime."""
        out = tmp_path / "diag.md"
        stack_only_success = PerformanceResult(
            model_name="org/stack",
            success=True,
            generation=_MockGeneration(text="", prompt_tokens=5000, generation_tokens=0),
            total_time=2.0,
            generation_time=1.0,
            model_load_time=1.0,
            runtime_diagnostics=RuntimeDiagnostics(
                input_validation_time_s=0.05,
                model_load_time_s=1.0,
                prompt_prep_time_s=0.15,
                decode_time_s=0.8,
                cleanup_time_s=0.1,
                first_token_latency_s=0.25,
                stop_reason="completed",
            ),
        )
        generate_diagnostics_report(
            results=[
                _make_failure_with_details("org/fail"),
                _make_harness_success("org/harness"),
                stack_only_success,
                _make_success("org/clean"),
            ],
            filename=out,
            versions=_stub_versions(),
            system_info={},
            prompt="test",
        )

        content = out.read_text(encoding="utf-8")
        assert "## Coverage & Runtime Metrics" in content
        assert "**Detailed diagnostics models:** 3" in content
        assert "**Summary diagnostics models:** 1" in content
        assert "**Coverage check:** ✅ Complete (each model appears exactly once)." in content
        assert "**Total model runtime (sum):**" in content
        assert "**Average runtime per model:**" in content
        assert "Runtime aggregates: unavailable" not in content
        assert "**Runtime note:**" in content
        assert "**Dominant runtime phase:**" in content
        assert "local prompt prep=" in content
        assert "upstream prefill / first-token=" in content
        assert "post-prefill decode=" in content
        assert "**Generation total:**" in content
        assert "**Validation overhead:**" in content
        assert "**Upstream prefill / first-token latency:**" in content
        assert "**What this likely means:**" in content
        assert "**Suggested next action:**" in content

    def test_runtime_metrics_split_prefill_from_post_prefill_decode(
        self,
        tmp_path: Path,
    ) -> None:
        """Diagnostics should expose upstream prefill as a distinct derived phase."""
        out = tmp_path / "diag.md"
        prefill_heavy = PerformanceResult(
            model_name="org/prefill-heavy",
            success=True,
            generation=_MockGeneration(text="ok", prompt_tokens=1000, generation_tokens=10),
            total_time=10.4,
            generation_time=10.0,
            model_load_time=0.2,
            runtime_diagnostics=RuntimeDiagnostics(
                model_load_time_s=0.2,
                prompt_prep_time_s=0.1,
                decode_time_s=10.0,
                cleanup_time_s=0.1,
                first_token_latency_s=8.0,
                stop_reason="completed",
            ),
        )
        decode_heavy = replace(
            prefill_heavy,
            model_name="org/decode-heavy",
            total_time=6.4,
            generation_time=6.0,
            runtime_diagnostics=RuntimeDiagnostics(
                model_load_time_s=0.2,
                prompt_prep_time_s=0.1,
                decode_time_s=6.0,
                cleanup_time_s=0.1,
                first_token_latency_s=1.0,
                stop_reason="completed",
            ),
        )

        generate_diagnostics_report(
            results=[prefill_heavy, decode_heavy],
            filename=out,
            versions=_stub_versions(),
            system_info={},
            prompt="test",
        )

        content = out.read_text(encoding="utf-8")
        assert "**Dominant runtime phase:** upstream prefill / first-token dominated" in content
        assert "upstream prefill / first-token=9.00s" in content
        assert "post-prefill decode=7.00s" in content
        assert (
            "**Generation total:** 16.00s across 2 model(s); upstream prefill / "
            "first-token split available for 2/2 model(s)."
        ) in content

    def test_runtime_metrics_keep_unsplit_generation_when_first_token_missing(
        self,
        tmp_path: Path,
    ) -> None:
        """Missing first-token latency should preserve the old generation total fallback."""
        out = tmp_path / "diag.md"
        result = PerformanceResult(
            model_name="org/no-first-token",
            success=True,
            generation=_MockGeneration(text="ok", prompt_tokens=100, generation_tokens=10),
            total_time=3.3,
            generation_time=3.0,
            model_load_time=0.2,
            runtime_diagnostics=RuntimeDiagnostics(
                model_load_time_s=0.2,
                prompt_prep_time_s=0.1,
                decode_time_s=3.0,
                cleanup_time_s=0.0,
                first_token_latency_s=None,
                stop_reason="completed",
            ),
        )

        generate_diagnostics_report(
            results=[_make_failure_with_details("org/fail"), result],
            filename=out,
            versions=_stub_versions(),
            system_info={},
            prompt="test",
        )

        content = out.read_text(encoding="utf-8")
        assert "generation total (unsplit)=3.00s" in content
        assert "post-prefill decode=" not in content
        assert "upstream prefill / first-token=" not in content

    def test_report_written_for_stack_signal_without_failures(self, tmp_path: Path) -> None:
        """Suspicious successful runs should still produce diagnostics for stack triage."""
        out = tmp_path / "diag.md"
        repeated_phrase = "loop"
        analysis = GenerationQualityAnalysis(
            is_repetitive=True,
            repeated_token=repeated_phrase,
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
            has_harness_issue=False,
            harness_issue_type=None,
            harness_issue_details=[],
            word_count=40,
            unique_ratio=0.05,
            prompt_checks_ran=True,
        )
        success = PerformanceResult(
            model_name="org/suspicious-success",
            success=True,
            generation=_MockGeneration(text=("loop " * 40).strip(), prompt_tokens=15000),
            total_time=1.0,
            generation_time=0.5,
            model_load_time=0.5,
            quality_analysis=analysis,
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
        assert "### Long-Context Degradation / Potential Stack Issues" in content
        assert "org/suspicious-success" in content

    def test_report_written_for_context_echo_stack_signal(self, tmp_path: Path) -> None:
        """Extreme prompt-length context echo should be surfaced as a stack-signal candidate."""
        out = tmp_path / "diag.md"
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
            has_reasoning_leak=False,
            reasoning_leak_markers=[],
            has_context_echo=True,
            context_echo_ratio=0.94,
            has_harness_issue=False,
            harness_issue_type=None,
            harness_issue_details=[],
            word_count=80,
            unique_ratio=0.3,
            prompt_checks_ran=True,
        )
        success = PerformanceResult(
            model_name="org/context-echo",
            success=True,
            generation=_MockGeneration(
                text="echoed context",
                prompt_tokens=15000,
                generation_tokens=80,
            ),
            total_time=1.0,
            generation_time=0.5,
            model_load_time=0.5,
            quality_analysis=analysis,
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
        assert "### Long-Context Degradation / Potential Stack Issues" in content
        assert "`mlx-vlm / mlx`" in content
        assert "org/context-echo" in content

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
        assert "output/prompt=0.00%" in content
        assert "<empty output>" in content
        assert "Likely owner" in content
        assert "<summary>Sample output</summary>" not in content

    def test_harness_section_uses_prompt_template_owner_hint(self, tmp_path: Path) -> None:
        """Prompt-template harness issues should not be flattened into mlx runtime ownership."""
        out = tmp_path / "diag.md"
        generate_diagnostics_report(
            results=[
                _make_harness_success(
                    name="org/harness-template",
                    text="",
                    prompt_tokens=4000,
                    generation_tokens=0,
                    harness_type="prompt_template",
                    harness_detail="output:zero_tokens",
                ),
            ],
            filename=out,
            versions=_stub_versions(),
            system_info={},
            prompt="test",
        )
        content = out.read_text(encoding="utf-8")
        assert "`model-config / mlx-vlm`" in content
        assert "validate chat-template/config expectations" in content

    def test_failure_section_includes_representative_maintainer_triage(
        self,
        tmp_path: Path,
    ) -> None:
        """Failure clusters should include a compact representative triage block."""
        out = tmp_path / "diag.md"
        failed = PerformanceResult(
            model_name="org/broken-model",
            generation=None,
            success=False,
            error_message="RuntimeError: shape mismatch",
            error_stage="Model Error",
            error_code="MLX_VLM_DECODE_RUNTIME",
            error_package="mlx-vlm",
            error_traceback="Traceback (most recent call last):\nRuntimeError: shape mismatch",
        )

        generate_diagnostics_report(
            results=[failed],
            filename=out,
            versions=_stub_versions(),
            system_info={},
            prompt="test",
        )

        content = out.read_text(encoding="utf-8")
        assert "Representative maintainer triage" in content
        assert "- _Likely owner:_" in content
        assert "MLX_VLM_DECODE_RUNTIME" in content
        assert "confidence=high" in content
        assert "runtime_failure" in content

    def test_action_summary_splits_mixed_harness_owners(self, tmp_path: Path) -> None:
        """Mixed harness owner classes should render separate maintainer triage rows."""
        out = tmp_path / "diag.md"
        generate_diagnostics_report(
            results=[
                _make_harness_success(
                    name="org/harness-template",
                    harness_type="prompt_template",
                    harness_detail="output:zero_tokens",
                ),
                _make_harness_success(
                    name="org/harness-runtime",
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
        assert "`model-config / mlx-vlm`" in content
        assert "`mlx-vlm / mlx`" in content
        assert "validate chat-template/config expectations" in content
        assert "validate long-context handling" in content

    def test_action_summary_splits_mixed_stack_signal_owners(self, tmp_path: Path) -> None:
        """Mixed stack-signal owner classes should render separate maintainer triage rows."""
        out = tmp_path / "diag.md"
        generate_diagnostics_report(
            results=[
                PerformanceResult(
                    model_name="org/stack-empty",
                    success=True,
                    generation=_MockGeneration(text="", prompt_tokens=5000, generation_tokens=0),
                    total_time=1.0,
                    generation_time=0.5,
                    model_load_time=0.5,
                    quality_analysis=GenerationQualityAnalysis(
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
                        has_reasoning_leak=False,
                        reasoning_leak_markers=[],
                        has_context_echo=False,
                        context_echo_ratio=0.0,
                        has_harness_issue=False,
                        harness_issue_type=None,
                        harness_issue_details=[],
                        word_count=0,
                        unique_ratio=0.0,
                        prompt_checks_ran=True,
                    ),
                ),
                PerformanceResult(
                    model_name="org/stack-context",
                    success=True,
                    generation=_MockGeneration(
                        text="echoed context",
                        prompt_tokens=15000,
                        generation_tokens=80,
                    ),
                    total_time=1.0,
                    generation_time=0.5,
                    model_load_time=0.5,
                    quality_analysis=GenerationQualityAnalysis(
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
                        has_reasoning_leak=False,
                        reasoning_leak_markers=[],
                        has_context_echo=True,
                        context_echo_ratio=0.9,
                        has_harness_issue=False,
                        harness_issue_type=None,
                        harness_issue_details=[],
                        word_count=80,
                        unique_ratio=0.3,
                        prompt_checks_ran=True,
                    ),
                ),
            ],
            filename=out,
            versions=_stub_versions(),
            system_info={},
            prompt="test",
        )
        content = out.read_text(encoding="utf-8")
        assert "`mlx-vlm`" in content
        assert "`mlx-vlm / mlx`" in content
        assert "org/stack-empty" in content
        assert "org/stack-context" in content

    def test_action_summary_splits_mixed_preflight_owners(self, tmp_path: Path) -> None:
        """Mixed preflight owner classes should render separate maintainer triage rows."""
        out = tmp_path / "diag.md"
        generate_diagnostics_report(
            results=[_make_success()],
            filename=out,
            versions=_stub_versions(),
            system_info={},
            prompt="test",
            history=DiagnosticsHistoryInputs(
                preflight_issues=(
                    "transformers==5.4.0 is below minimum 5.5.3 required by check_models.",
                    "mlx runtime probe reported a suspicious cache incompatibility",
                ),
            ),
        )
        content = out.read_text(encoding="utf-8")
        assert "`transformers`" in content
        assert "`mlx`" in content
        assert "verify API compatibility and pinned version floor." in content
        assert "mlx runtime probe reported a suspicious cache incompatibility" in content

    def test_harness_token_leak_details_are_escaped(self, tmp_path: Path) -> None:
        """Token leak details should be escaped so markdown does not treat them as HTML."""
        out = tmp_path / "diag.md"
        generate_diagnostics_report(
            results=[
                _make_harness_success(
                    name="org/harness-token",
                    text="safe output",
                    harness_type="stop_token",
                    harness_detail="token_leak:<|end|>",
                ),
            ],
            filename=out,
            versions=_stub_versions(),
            system_info={},
            prompt="test",
        )
        content = out.read_text(encoding="utf-8")
        assert "Special control token &lt;|end|&gt; appeared in generated text." in content
        assert "Special control token <|end|> appeared in generated text." not in content

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

    def test_filing_guidance_points_to_footer_repro_section(self, tmp_path: Path) -> None:
        """Failure clusters should point readers to the footer repro appendix."""
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
        assert "### To reproduce" in content
        assert "### Filing Guidance" not in content
        assert "Copy/paste GitHub issue template" not in content
        assert "Observed" in content
        assert "Failure phase" not in content
        assert "Canonical code" not in content
        assert "Signature" not in content
        assert "Environment fingerprint" not in content
        assert "Exact model-specific repro command appears below" in content
        assert "Representative failing model: `org/broken-model`" in content
        assert "Repro command (exact run)" not in content
        assert "issues/new" not in content
        assert "Repro bundle" not in content

    def test_failure_cluster_shows_generated_output_inline(self, tmp_path: Path) -> None:
        """Failure clusters should show generated output inline when present."""
        out = tmp_path / "diag.md"
        generate_diagnostics_report(
            results=[
                _make_failure_with_details(
                    "org/partial",
                    error_msg="decoder crashed after partial output",
                    generated_text="Partial decoded answer before crash.",
                ),
            ],
            filename=out,
            versions=_stub_versions(),
            system_info={},
            prompt="test",
        )
        content = out.read_text(encoding="utf-8")
        assert "**Observed model output (`org/partial`):**" in content
        assert "Partial decoded answer before crash." in content

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

    def test_harness_repro_bundles_written_and_linked(self, tmp_path: Path) -> None:
        """Successful issue-clustered harness anomalies should get repro bundles."""
        prompt = "Describe this image."
        harness = replace(
            _make_harness_success(
                "org/model-harness",
                text="<|end|> leaked control token",
                harness_type="prompt_template",
                harness_detail="output:zero_tokens",
            ),
            prompt_diagnostics=check_models.PromptDiagnostics(
                model_type="qwen2_vl",
                processor_class="transformers.AutoProcessor",
                tokenizer_class="transformers.PreTrainedTokenizerFast",
                rendered_prompt_hash_sha256="rendered-hash",
                rendered_prompt_preview="<|im_start|>user <image> Describe this image.",
                rendered_prompt_chars=44,
                image_placeholder_count=1,
                eos_token_id=151645,
                special_token_ids=(151645,),
                special_tokens=("<|end|>", "</think>"),
                generate_kwargs={
                    "max_tokens": 500,
                    "quantized_kv_start": check_models.DEFAULT_QUANTIZED_KV_START,
                },
            ),
        )
        bundles = export_failure_repro_bundles(
            results=[harness],
            output_dir=tmp_path / "repro_bundles",
            run_args=Namespace(),
            versions=_stub_versions(),
            system_info={"Python Version": "3.13"},
            prompt=prompt,
            image_path=None,
        )

        assert "org/model-harness" in bundles
        bundle_path = bundles["org/model-harness"]
        payload = json.loads(bundle_path.read_text(encoding="utf-8"))
        assert payload["result"]["success"] is True
        assert payload["issue_cluster_id"] == "model-config-mlx-vlm_prompt-template_001"
        assert payload["repro"]["prompt_diagnostics"]["rendered_prompt_hash_sha256"] == (
            "rendered-hash"
        )
        assert payload["repro"]["prompt_diagnostics"]["special_tokens"] == [
            "<|end|>",
            "</think>",
        ]

        issue_reports = _generate_github_issue_reports(
            diagnostics_snapshot=_build_diagnostics_snapshot(results=[harness], prompt=prompt),
            output_dir=tmp_path,
            versions=_stub_versions(),
            system_info={"Python Version": "3.13"},
            repro_bundles=bundles,
            run_args=Namespace(),
        )
        issue_content = next(iter(issue_reports.values())).read_text(encoding="utf-8")
        assert bundle_path.name in issue_content
        assert "Repro bundles:" in issue_content

    def test_repro_command_omits_upstream_quantized_kv_default(self) -> None:
        """Default KV quantization start should not be forwarded as a repro override."""
        tokens = check_models._build_repro_command_tokens(
            image_path=None,
            run_args=Namespace(
                quantized_kv_start=check_models.DEFAULT_QUANTIZED_KV_START,
                trust_remote_code=True,
            ),
            include_selection=False,
        )
        assert "--quantized-kv-start" not in tokens

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
        self,
        tmp_path: Path,
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
        self,
        tmp_path: Path,
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
            resize_shape=(768, 512),
            eos_tokens=("</think>",),
            skip_special_tokens=True,
            processor_kwargs={"cropping": False},
            enable_thinking=True,
            thinking_budget=96,
            thinking_start_token=THINKING_START_TOKEN,
            thinking_end_token=THINKING_END_TOKEN,
            max_tokens=123,
            temperature=0.7,
            top_p=0.92,
            min_p=0.08,
            top_k=24,
            repetition_penalty=1.1,
            repetition_context_size=50,
            lazy_load=False,
            max_kv_size=None,
            kv_bits=None,
            kv_group_size=64,
            quantized_kv_start=2048,
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
        assert "--min-p 0.08" in content
        assert "--top-k 24" in content
        assert "--resize-shape 768 512" in content
        assert "--eos-tokens '</think>'" in content
        assert "--skip-special-tokens" in content
        assert "--processor-kwargs '{\"cropping\": false}'" in content
        assert "--enable-thinking" in content
        assert "--thinking-budget 96" in content
        assert "--thinking-start-token '<think>'" in content
        assert "--repetition-penalty 1.1" in content
        assert "--repetition-context-size 50" in content
        assert "--quantized-kv-start 2048" in content
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

    def test_portable_upstream_probes_include_harness_and_stack_models_only(
        self,
        tmp_path: Path,
    ) -> None:
        """Harness and stack-only reports should still get portable generation probes."""
        out = tmp_path / "diag.md"
        image_path = tmp_path / "portable original image.jpg"
        image_path.write_text("placeholder", encoding="utf-8")
        stack_signal = PerformanceResult(
            model_name="org/stack-only",
            success=True,
            generation=_MockGeneration(
                text="echoed context",
                prompt_tokens=15000,
                generation_tokens=80,
            ),
            total_time=1.0,
            generation_time=0.5,
            model_load_time=0.5,
            quality_analysis=GenerationQualityAnalysis(
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
                has_reasoning_leak=False,
                reasoning_leak_markers=[],
                has_context_echo=True,
                context_echo_ratio=0.94,
                has_harness_issue=False,
                harness_issue_type=None,
                harness_issue_details=[],
                word_count=80,
                unique_ratio=0.3,
                prompt_checks_ran=True,
            ),
        )
        run_args = Namespace(
            image=image_path,
            folder=None,
            models=None,
            exclude=None,
            trust_remote_code=True,
            revision=None,
            adapter_path=None,
            prompt=None,
            max_tokens=check_models.DEFAULT_MAX_TOKENS,
            temperature=check_models.DEFAULT_TEMPERATURE,
            top_p=1.0,
            repetition_penalty=None,
            repetition_context_size=None,
            min_p=0.0,
            top_k=0,
            max_kv_size=None,
            kv_bits=None,
            kv_quant_scheme=check_models.DEFAULT_KV_QUANT_SCHEME,
            prefill_step_size=None,
            thinking_budget=None,
            thinking_start_token=None,
            thinking_end_token=None,
            kv_group_size=64,
            quantized_kv_start=check_models.DEFAULT_QUANTIZED_KV_START,
            timeout=check_models.DEFAULT_TIMEOUT,
            detailed_metrics=False,
            lazy_load=False,
            skip_special_tokens=False,
            enable_thinking=False,
            resize_shape=None,
            eos_tokens=None,
            processor_kwargs=None,
            verbose=False,
            no_color=False,
            force_color=False,
            width=None,
            quality_config=None,
            context_marker="Context:",
        )
        generate_diagnostics_report(
            results=[
                _make_harness_success("org/harness-only"),
                stack_signal,
            ],
            filename=out,
            versions=_stub_versions(),
            system_info={},
            prompt="Describe this image.",
            image_path=image_path,
            run_args=run_args,
        )

        content = out.read_text(encoding="utf-8")
        portable_block = _extract_markdown_subsection(
            content,
            "### Portable upstream probes (no local image required)",
            end_headings=("### Prompt Used",),
        )
        assert "org/harness-only" in portable_block
        assert "org/stack-only" in portable_block
        assert "check_models_portable_probe.png" in portable_block
        assert "--models org/harness-only org/stack-only" in portable_block
        assert str(image_path) not in portable_block

    def test_prompt_in_section(self, tmp_path: Path) -> None:
        """Prompt should be rendered in a dedicated markdown section."""
        out = tmp_path / "diag.md"
        generate_diagnostics_report(
            results=[_make_failure_with_details()],
            filename=out,
            versions=_stub_versions(),
            system_info={},
            prompt="Analyze this image carefully.",
        )
        content = out.read_text(encoding="utf-8")
        assert "### Prompt Used" in content
        assert "### Run details" in content
        assert "### Run details\n\n- Input image:" in content
        assert "Analyze this image carefully." in content

    def test_multi_model_cluster_has_no_priority_label(self, tmp_path: Path) -> None:
        """Clusters with multiple models should not emit priority labels."""
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
        assert "Failure affecting 2 models" in content
        assert "Priority" not in content

    def test_generate_markdown_report_uses_provided_report_context(self, tmp_path: Path) -> None:
        """Markdown generation should reuse a supplied cached report context."""
        out = tmp_path / "results.md"
        results = [_make_success("org/good"), _make_failure("org/bad")]
        report_context = _build_report_render_context(results=results, prompt="test prompt")

        with (
            patch.object(check_models, "_build_report_render_context", side_effect=AssertionError),
            patch.object(check_models, "analyze_model_issues", side_effect=AssertionError),
            patch.object(
                check_models,
                "compute_performance_statistics",
                side_effect=AssertionError,
            ),
            patch.object(check_models, "get_system_characteristics", side_effect=AssertionError),
        ):
            generate_markdown_report(
                results=results,
                filename=out,
                versions=_stub_versions(),
                prompt="test prompt",
                total_runtime_seconds=1.0,
                report_context=report_context,
            )

        content = out.read_text(encoding="utf-8")
        assert "# Model Performance Results" in content
        assert "org/good" in content

    def test_generate_tsv_report_uses_provided_report_context(self, tmp_path: Path) -> None:
        """TSV generation should reuse a supplied cached report context."""
        out = tmp_path / "results.tsv"
        results = [_make_success("org/good"), _make_failure("org/bad")]
        report_context = _build_report_render_context(results=results, prompt="test prompt")

        with patch.object(check_models, "_build_report_render_context", side_effect=AssertionError):
            generate_tsv_report(
                results=results,
                filename=out,
                report_context=report_context,
            )

        content = out.read_text(encoding="utf-8")
        assert "org/good" in content
        assert "error_type" in content

    def test_generate_tsv_report_standalone_uses_prepared_table_path(
        self,
        tmp_path: Path,
    ) -> None:
        """Standalone TSV generation should still render results without cached context."""
        out = tmp_path / "standalone.tsv"
        results = [_make_success("org/good"), _make_failure("org/bad")]

        generate_tsv_report(
            results=results,
            filename=out,
        )

        content = out.read_text(encoding="utf-8")
        assert "org/good" in content
        assert "org/bad" in content

    def test_build_diagnostics_snapshot_partitions_results(self) -> None:
        """Diagnostics snapshot should preserve failure and harness buckets once."""
        failure = _make_failure_with_details("org/fail")
        harness = _make_harness_success("org/harness", harness_type="long_context")
        clean = _make_success("org/clean")

        snapshot = _build_diagnostics_snapshot(
            results=[failure, harness, clean],
            preflight_issues=("mlx-vlm mismatch",),
        )

        assert len(snapshot.failed) == 1
        assert len(snapshot.harness_results) == 1
        assert len(snapshot.unflagged_successful) == 1
        assert len(snapshot.preflight_issues) == 1
        assert len(snapshot.failure_clusters) == 1

    def test_log_maintainer_summary_includes_counts_and_paths(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Maintainer summary should emit concise diagnostics counts and artifact hints."""
        diagnostics_path = tmp_path / "diagnostics.md"
        diagnostics_path.write_text("summary\n", encoding="utf-8")
        failure = _make_failure_with_details("org/fail", error_package="mlx-vlm")
        snapshot = DiagnosticsSnapshot(
            failed=(failure,),
            harness_results=((_make_harness_success("org/harness"), "sample output"),),
            preflight_issues=("transformers warning",),
            failure_clusters=(("cluster", (failure,)),),
        )
        artifacts = DiagnosticsArtifacts(
            snapshot=snapshot,
            diagnostics_written=True,
            repro_bundles={"org/fail": tmp_path / "repro.json"},
        )

        with caplog.at_level("INFO", logger=check_models.LOGGER_NAME):
            _log_maintainer_summary(
                artifacts=artifacts,
                diagnostics_path=diagnostics_path,
            )

        assert "Diagnostics signals: failures=1, harness=1, stack=0, preflight=1" in caplog.text
        assert "Likely owners:" in caplog.text
        assert "Repro bundles available for 1 issue-linked model(s)." in caplog.text

    def test_log_maintainer_summary_mentions_stack_signal_owner_hint(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Maintainer summary should report inferred owners for stack-signal anomalies."""
        diagnostics_path = tmp_path / "diagnostics.md"
        diagnostics_path.write_text("summary\n", encoding="utf-8")
        stack_result = PerformanceResult(
            model_name="org/stack-transformers",
            success=True,
            generation=_MockGeneration(
                text="echoed context",
                prompt_tokens=15000,
                generation_tokens=80,
            ),
            total_time=1.0,
            generation_time=0.5,
            model_load_time=0.5,
        )
        snapshot = DiagnosticsSnapshot(
            stack_signals=(
                (
                    stack_result,
                    "Context echo under extreme prompt length",
                    "transformers / mlx-vlm",
                ),
            ),
            preflight_issues=(
                "transformers==5.4.0 is below minimum 5.5.3 required by check_models.",
            ),
        )
        artifacts = DiagnosticsArtifacts(
            snapshot=snapshot,
            diagnostics_written=True,
            repro_bundles={},
        )

        with caplog.at_level("INFO", logger=check_models.LOGGER_NAME):
            _log_maintainer_summary(
                artifacts=artifacts,
                diagnostics_path=diagnostics_path,
            )

        assert (
            "Long-context or stack-signal anomalies: 1 model(s) likely owned by "
            "transformers / mlx-vlm."
        ) in caplog.text

    def test_log_maintainer_summary_mentions_harness_owner_hint(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Maintainer summary should report inferred owners for harness anomalies."""
        diagnostics_path = tmp_path / "diagnostics.md"
        diagnostics_path.write_text("summary\n", encoding="utf-8")
        snapshot = DiagnosticsSnapshot(
            harness_results=((_make_harness_success("org/harness-template"), ""),),
        )
        artifacts = DiagnosticsArtifacts(
            snapshot=snapshot,
            diagnostics_written=True,
            repro_bundles={},
        )

        with caplog.at_level("INFO", logger=check_models.LOGGER_NAME):
            _log_maintainer_summary(
                artifacts=artifacts,
                diagnostics_path=diagnostics_path,
            )

        assert (
            "Harness/runtime anomalies: 1 model(s) likely owned by model-config / mlx-vlm."
        ) in caplog.text

    def test_log_maintainer_summary_splits_mixed_harness_owner_hints(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Maintainer summary should emit one harness line per inferred owner bucket."""
        diagnostics_path = tmp_path / "diagnostics.md"
        diagnostics_path.write_text("summary\n", encoding="utf-8")
        snapshot = DiagnosticsSnapshot(
            harness_results=(
                (_make_harness_success("org/harness-template"), ""),
                (
                    _make_harness_success(
                        "org/harness-runtime",
                        harness_type="long_context",
                        harness_detail="long_context_empty(5000tok)",
                    ),
                    "",
                ),
            ),
        )
        artifacts = DiagnosticsArtifacts(
            snapshot=snapshot,
            diagnostics_written=True,
            repro_bundles={},
        )

        with caplog.at_level("INFO", logger=check_models.LOGGER_NAME):
            _log_maintainer_summary(
                artifacts=artifacts,
                diagnostics_path=diagnostics_path,
            )

        assert (
            "Harness/runtime anomalies: 1 model(s) likely owned by model-config / mlx-vlm. "
            "Next: validate chat-template/config expectations and mlx-vlm prompt formatting for this model."
        ) in caplog.text
        assert (
            "Harness/runtime anomalies: 1 model(s) likely owned by mlx-vlm / mlx. "
            "Next: validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime."
        ) in caplog.text

    def test_log_maintainer_summary_splits_mixed_stack_and_preflight_owner_hints(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Maintainer summary should emit per-owner lines for stack and preflight signals."""
        diagnostics_path = tmp_path / "diagnostics.md"
        diagnostics_path.write_text("summary\n", encoding="utf-8")
        stack_runtime = PerformanceResult(
            model_name="org/stack-runtime",
            success=True,
            generation=_MockGeneration(text="echo", prompt_tokens=15000, generation_tokens=80),
            total_time=1.0,
            generation_time=0.5,
            model_load_time=0.5,
        )
        stack_transformers = PerformanceResult(
            model_name="org/stack-transformers",
            success=True,
            generation=_MockGeneration(text="echo", prompt_tokens=15000, generation_tokens=80),
            total_time=1.0,
            generation_time=0.5,
            model_load_time=0.5,
        )
        snapshot = DiagnosticsSnapshot(
            stack_signals=(
                (
                    stack_runtime,
                    "Context echo under long prompt length",
                    "mlx-vlm / mlx",
                ),
                (
                    stack_transformers,
                    "Context echo under extreme prompt length",
                    "transformers / mlx-vlm",
                ),
            ),
            preflight_issues=(
                "transformers==5.4.0 is below minimum 5.5.3 required by check_models.",
                "mlx runtime probe reported a suspicious cache incompatibility",
            ),
        )
        artifacts = DiagnosticsArtifacts(
            snapshot=snapshot,
            diagnostics_written=True,
            repro_bundles={},
        )

        with caplog.at_level("INFO", logger=check_models.LOGGER_NAME):
            _log_maintainer_summary(
                artifacts=artifacts,
                diagnostics_path=diagnostics_path,
            )

        assert (
            "Long-context or stack-signal anomalies: 1 model(s) likely owned by mlx-vlm / mlx. "
            "Next: validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime."
        ) in caplog.text
        assert (
            "Long-context or stack-signal anomalies: 1 model(s) likely owned by transformers / mlx-vlm. "
            "Next: verify API compatibility and pinned version floor."
        ) in caplog.text
        assert (
            "Preflight compatibility warnings: 1 issue(s) likely owned by mlx. "
            "Next: check tensor/cache behavior and memory pressure handling."
        ) in caplog.text
        assert (
            "Preflight compatibility warnings: 1 issue(s) likely owned by transformers. "
            "Next: verify API compatibility and pinned version floor."
        ) in caplog.text


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
        comparison = cast(
            "check_models.HistoryComparisonSummary",
            {
                "regressions": ["org/b"],
                "recoveries": [],
                "new_models": [],
                "missing_models": ["org/c"],
                "quality_regressions": [],
                "quality_recoveries": [],
                "harness_regressions": [],
                "harness_recoveries": [],
                "owner_changes": [],
            },
        )

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


class TestPruneReproBundles:
    """Tests for _prune_repro_bundles."""

    def test_removes_old_bundles(self, tmp_path: Path) -> None:
        """JSON bundles older than max_age_days are removed."""
        old_file = tmp_path / "20240101T000000_001_model_sig.json"
        old_file.write_text("{}")
        old_time = time.time() - 200 * 86400
        os.utime(old_file, (old_time, old_time))

        new_file = tmp_path / "20260401T000000_001_model_sig.json"
        new_file.write_text("{}")

        removed = _prune_repro_bundles(tmp_path, 90, max_runs=100)
        assert removed == 1
        assert not old_file.exists()
        assert new_file.exists()

    def test_zero_days_disables_pruning(self, tmp_path: Path) -> None:
        """max_age_days=0 disables pruning."""
        (tmp_path / "20260401T000000_001_model_sig.json").write_text("{}")
        removed = _prune_repro_bundles(tmp_path, 0)
        assert removed == 0

    def test_nonexistent_dir(self, tmp_path: Path) -> None:
        """Non-existent directory returns 0."""
        removed = _prune_repro_bundles(tmp_path / "nonexistent", 90)
        assert removed == 0

    def test_prunes_json_files_by_run_count(self, tmp_path: Path) -> None:
        """JSON bundle files beyond max_runs are pruned."""
        # Create files from 3 different "runs" (distinct 16-char prefixes)
        for i, prefix in enumerate(["20260401T000000", "20260402T000000", "20260403T000000"]):
            f = tmp_path / f"{prefix}_{i:03d}_model_sig.json"
            f.write_text("{}")
        removed = _prune_repro_bundles(tmp_path, max_age_days=9999, max_runs=2)
        assert removed == 1
        # Oldest run should be gone
        assert not (tmp_path / "20260401T000000_000_model_sig.json").exists()
        assert (tmp_path / "20260402T000000_001_model_sig.json").exists()
        assert (tmp_path / "20260403T000000_002_model_sig.json").exists()

    def test_removes_empty_directories(self, tmp_path: Path) -> None:
        """Empty subdirectories are cleaned up."""
        empty = tmp_path / "empty_dir"
        empty.mkdir()
        _prune_repro_bundles(tmp_path, max_age_days=9999, max_runs=100)
        assert not empty.exists()


class TestCleanStaleToplevelReports:
    """Tests for _clean_stale_toplevel_reports."""

    def test_removes_stale_files_when_canonical_exists(self, tmp_path: Path) -> None:
        """Stale top-level file removed when reports/ copy exists."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        (tmp_path / "results.md").write_text("old")
        (reports_dir / "results.md").write_text("canonical")
        removed = _clean_stale_toplevel_reports(tmp_path, reports_dir)
        assert removed == 1
        assert not (tmp_path / "results.md").exists()

    def test_keeps_file_when_no_canonical(self, tmp_path: Path) -> None:
        """Top-level file kept when reports/ copy does not exist."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        (tmp_path / "results.md").write_text("only copy")
        removed = _clean_stale_toplevel_reports(tmp_path, reports_dir)
        assert removed == 0
        assert (tmp_path / "results.md").exists()


class TestEmptyRecommendedBucketExplanation:
    """Test that an empty 'recommended' bucket includes an explanation."""

    def test_recommended_bucket_shows_explanation(self) -> None:
        """Recommended bucket text explains why no models qualified."""
        md: list[str] = []
        _append_review_user_buckets(
            md, {"recommended": [], "caveat": [], "needs_triage": [], "avoid": []}
        )
        text = "\n".join(md)
        assert "quality thresholds" in text
        # Other empty buckets should just say "None."
        lines = [line for line in md if line.startswith("- None")]
        explanation_lines = [ln for ln in lines if "quality thresholds" in ln]
        plain_none_lines = [ln for ln in lines if ln.strip() == "- None."]
        assert len(explanation_lines) == 1
        assert len(plain_none_lines) == 3


class TestGithubIssueReportsCleanup:
    """Tests for _generate_github_issue_reports stale file cleanup."""

    def test_stale_issue_files_removed(self, tmp_path: Path) -> None:
        """Old issue_*.md files are removed before writing new ones."""
        issues_dir = tmp_path / "issues"
        issues_dir.mkdir()
        (issues_dir / "issue_001_crash.md").write_text("stale crash report")
        (issues_dir / "issue_002_harness.md").write_text("stale harness report")
        # A non-issue file should be left alone
        (issues_dir / "README.md").write_text("keep me")

        # Empty snapshot → no new issue files written
        snapshot = DiagnosticsSnapshot()
        _generate_github_issue_reports(
            diagnostics_snapshot=snapshot,
            output_dir=tmp_path,
            versions=_stub_versions(),
            system_info={"Python Version": "3.13"},
            repro_bundles={},
            run_args=None,
        )

        assert not (issues_dir / "issue_001_crash.md").exists()
        assert not (issues_dir / "issue_002_harness.md").exists()
        assert (issues_dir / "index.md").exists()
        assert (issues_dir / "README.md").exists()


class TestGithubIssueReportContent:
    """Content checks for generated standalone GitHub issue reports."""

    def test_crash_issue_includes_traceback_repro_bundle_and_environment(
        self, tmp_path: Path
    ) -> None:
        """Crash issue templates should include traceback, repro bundle, and environment."""
        failed_result = PerformanceResult(
            model_name="org/broken-model",
            generation=None,
            success=False,
            error_message="RuntimeError: shape mismatch",
            error_stage="Model Error",
            error_code="MLX_VLM_DECODE_RUNTIME",
            error_package="mlx-vlm",
            error_signature="MLX_DECODE_ERROR:abc123",
            error_traceback="Traceback (most recent call last):\nRuntimeError: shape mismatch",
            total_time=0.5,
        )
        snapshot = DiagnosticsSnapshot(
            failed=(failed_result,),
            failure_clusters=(("MLX_DECODE_ERROR:abc123", (failed_result,)),),
        )
        bundle_path = tmp_path / "repro_bundles" / "broken.json"
        bundle_path.parent.mkdir()
        bundle_path.write_text("{}", encoding="utf-8")

        generated = _generate_github_issue_reports(
            diagnostics_snapshot=snapshot,
            output_dir=tmp_path,
            versions=_stub_versions(),
            system_info={"Python Version": "3.13"},
            repro_bundles={"org/broken-model": bundle_path},
            run_args=None,
        )

        assert len(generated) == 1
        content = next(iter(generated.values())).read_text(encoding="utf-8")
        assert content.startswith("# \\[mlx-vlm\\]\\[MLX VLM decode runtime\\]")
        assert "## Summary" in content
        assert "## Affected Models" in content
        assert "## Minimal Evidence" in content
        assert "## Appendix: Detailed Evidence" in content
        assert "## Likely Root Cause" in content
        assert "## Repro Commands" in content
        assert "## Fix Checklist" in content
        assert "## Expected Fix Signal" in content
        assert "## Appendix: Environment" in content
        assert "MLX_VLM_DECODE_RUNTIME" in content
        assert "runtime_failure" in content
        assert "Traceback (most recent call last)" in content
        assert "[repro JSON](../repro_bundles/broken.json)" in content
        assert "attach or publish the JSON when filing upstream" in content
        assert "Python Version" in content
        assert "Priority" not in content

    def test_harness_issue_humanizes_details_and_includes_checklist(self, tmp_path: Path) -> None:
        """Harness issue templates should show humanized details and fix guidance."""
        harness_result = _make_harness_success(
            name="org/harness-empty",
            harness_type="long_context",
            harness_detail="long_context_empty(5000tok)",
        )
        snapshot = DiagnosticsSnapshot(
            harness_results=((harness_result, "long_context"),),
        )

        generated = _generate_github_issue_reports(
            diagnostics_snapshot=snapshot,
            output_dir=tmp_path,
            versions=_stub_versions(),
            system_info={"Python Version": "3.13"},
            repro_bundles={},
            run_args=None,
        )

        content = next(iter(generated.values())).read_text(encoding="utf-8")
        assert content.startswith("# \\[mlx-vlm / mlx\\]\\[Long-context collapse\\]")
        assert "## Likely Root Cause" in content
        assert "mlx-vlm first; MLX if cache/runtime reproduces" in content
        assert "`mlx-vlm / mlx`" in content
        assert "At long prompt length (5000 tokens), generation returned empty output." in content
        assert "context_budget" in content
        assert "long_context" in content
        assert "Rerun with reduced image/text burden" in content
        assert "Expected Fix Signal" in content
        assert "Priority" not in content

    def test_multiple_stop_token_models_produce_one_issue(self, tmp_path: Path) -> None:
        """Multiple stop-token harness models should cluster into one issue draft."""
        first = _make_harness_success(
            name="org/stop-a",
            text="caption <|end|>",
            generation_tokens=32,
            harness_type="stop_token",
            harness_detail="token_leak:<|end|>",
        )
        second = _make_harness_success(
            name="org/stop-b",
            text="caption </think>",
            generation_tokens=32,
            harness_type="stop_token",
            harness_detail="token_leak:</think>",
        )
        snapshot = DiagnosticsSnapshot(
            harness_results=(
                (first, cast("_MockGeneration", first.generation).text or ""),
                (second, cast("_MockGeneration", second.generation).text or ""),
            )
        )

        generated = _generate_github_issue_reports(
            diagnostics_snapshot=snapshot,
            output_dir=tmp_path,
            versions=_stub_versions(),
            system_info={"Python Version": "3.13"},
            repro_bundles={},
            run_args=None,
        )

        assert len(generated) == 1
        issue_path = next(iter(generated.values()))
        content = issue_path.read_text(encoding="utf-8")
        assert "affecting 2 model(s)" in content
        assert "org/stop-a" in content
        assert "org/stop-b" in content
        assert "`mlx-vlm_stop-token_001`" in content
        assert "&lt;\\|end\\|&gt;" in content
        assert "&lt;/think&gt;" in content

    def test_unrelated_harness_subtypes_produce_separate_issues(self, tmp_path: Path) -> None:
        """Different harness subtypes should not be merged into one draft."""
        stop = _make_harness_success(
            name="org/stop",
            text="caption <|end|>",
            generation_tokens=32,
            harness_type="stop_token",
            harness_detail="token_leak:<|end|>",
        )
        encoding = _make_harness_success(
            name="org/encoding",
            text="A Ġcaption",
            generation_tokens=32,
            harness_type="encoding",
            harness_detail="token_encoding:bpe_space_leak(1)",
        )
        snapshot = DiagnosticsSnapshot(
            harness_results=(
                (stop, cast("_MockGeneration", stop.generation).text or ""),
                (encoding, cast("_MockGeneration", encoding.generation).text or ""),
            )
        )

        generated = _generate_github_issue_reports(
            diagnostics_snapshot=snapshot,
            output_dir=tmp_path,
            versions=_stub_versions(),
            system_info={"Python Version": "3.13"},
            repro_bundles={},
            run_args=None,
        )

        assert len(generated) == 2
        index = (tmp_path / "issues" / "index.md").read_text(encoding="utf-8")
        assert "Stop/control tokens leaked into generated text" in index
        assert "Tokenizer decode leaked BPE/byte markers" in index
        assert "Issue Draft" in index
        assert "Fixed When" in index
        assert "Priority" not in index

    def test_issue_queue_humanizes_runtime_error_codes(self, tmp_path: Path) -> None:
        """Canonical runtime error-code subtypes should render as readable queue labels."""
        failed_result = PerformanceResult(
            model_name="org/broken-model",
            generation=None,
            success=False,
            error_message="RuntimeError: shape mismatch",
            error_stage="Model Error",
            error_code="MLX_MODEL_LOAD_MODEL",
            error_package="mlx",
            error_signature="MLX_MODEL_LOAD_MODEL:abc123",
            error_traceback="Traceback (most recent call last):\nRuntimeError: shape mismatch",
            total_time=0.5,
        )
        snapshot = DiagnosticsSnapshot(
            failed=(failed_result,),
            failure_clusters=(("MLX_MODEL_LOAD_MODEL:abc123", (failed_result,)),),
        )

        _generate_github_issue_reports(
            diagnostics_snapshot=snapshot,
            output_dir=tmp_path,
            versions=_stub_versions(),
            system_info={"Python Version": "3.13"},
            repro_bundles={},
            run_args=None,
        )

        index = (tmp_path / "issues" / "index.md").read_text(encoding="utf-8")
        assert "MLX: Model load / model error" in index
        assert "RuntimeError: shape mismatch" in index
        assert "Priority" not in index

    def test_issue_queue_summarizes_unsupported_model_failures(self, tmp_path: Path) -> None:
        """Unsupported model-type failures should produce filing-ready wording/actions."""
        failed_result = PerformanceResult(
            model_name="org/granite-vlm",
            generation=None,
            success=False,
            error_message=(
                "Model loading failed: Model type granite not supported. "
                "Error: No module named 'mlx_vlm.specifics'"
            ),
            root_error_message=(
                "Model type granite not supported. Error: No module named 'mlx_vlm.specifics'"
            ),
            error_stage="Model Error",
            error_code="MLX_VLM_MODEL_LOAD_MODEL",
            error_package="mlx-vlm",
            error_signature="MLX_VLM_MODEL_LOAD_MODEL:abc123",
            error_traceback="Traceback (most recent call last):\nModuleNotFoundError: no module",
            total_time=0.5,
        )
        snapshot = DiagnosticsSnapshot(
            failed=(failed_result,),
            failure_clusters=(("MLX_VLM_MODEL_LOAD_MODEL:abc123", (failed_result,)),),
        )

        generated = _generate_github_issue_reports(
            diagnostics_snapshot=snapshot,
            output_dir=tmp_path,
            versions=_stub_versions(),
            system_info={"Python Version": "3.13"},
            repro_bundles={},
            run_args=None,
        )

        index = (tmp_path / "issues" / "index.md").read_text(encoding="utf-8")
        content = next(iter(generated.values())).read_text(encoding="utf-8")
        normalized_content = " ".join(content.split())
        assert "Unsupported Granite model type/import path" in index
        assert "No module named" not in index
        assert "model-type registration/import handling for Granite" in normalized_content
        assert "model-type registry" in content
        assert "Inspect prompt-template, stop-token" not in content

    def test_issue_queue_summarizes_weight_mismatch_failures(self, tmp_path: Path) -> None:
        """Weight/config mismatches should point maintainers at keys and loader compatibility."""
        failed_result = PerformanceResult(
            model_name="org/mismatch-vlm",
            generation=None,
            success=False,
            error_message=(
                "Model loading failed: received the following parameters not in model: "
                "vision_model.merger.mlp.0.scale"
            ),
            root_error_message=(
                "received the following parameters not in model: vision_model.merger.mlp.0.scale"
            ),
            error_stage="Weight Mismatch",
            error_code="MLX_MODEL_LOAD_WEIGHT_MISMATCH",
            error_package="mlx",
            error_signature="MLX_MODEL_LOAD_WEIGHT_MISMATCH:abc123",
            error_traceback="Traceback (most recent call last):\nValueError: missing parameters",
            total_time=0.5,
        )
        snapshot = DiagnosticsSnapshot(
            failed=(failed_result,),
            failure_clusters=(("MLX_MODEL_LOAD_WEIGHT_MISMATCH:abc123", (failed_result,)),),
        )

        generated = _generate_github_issue_reports(
            diagnostics_snapshot=snapshot,
            output_dir=tmp_path,
            versions=_stub_versions(),
            system_info={"Python Version": "3.13"},
            repro_bundles={},
            run_args=None,
        )

        index = (tmp_path / "issues" / "index.md").read_text(encoding="utf-8")
        content = next(iter(generated.values())).read_text(encoding="utf-8")
        assert "Weight/config mismatch during model load" in index
        assert "checkpoint keys" in content
        assert "scale, bias" in content
        assert "KV/cache behavior" not in content

    def test_stack_signal_anomaly_produces_issue_draft(self, tmp_path: Path) -> None:
        """Successful-run stack anomalies should also produce clustered issue drafts."""
        result = PerformanceResult(
            model_name="org/stack-context",
            success=True,
            generation=_MockGeneration(
                text="echoed context",
                prompt_tokens=15000,
                generation_tokens=80,
            ),
            total_time=1.0,
            generation_time=0.5,
            model_load_time=0.5,
        )
        snapshot = DiagnosticsSnapshot(
            stack_signals=((result, "Context echo under long prompt length", "mlx-vlm / mlx"),)
        )

        generated = _generate_github_issue_reports(
            diagnostics_snapshot=snapshot,
            output_dir=tmp_path,
            versions=_stub_versions(),
            system_info={"Python Version": "3.13"},
            repro_bundles={},
            run_args=None,
        )

        assert len(generated) == 1
        content = next(iter(generated.values())).read_text(encoding="utf-8")
        assert "## Appendix: Detailed Evidence" in content
        assert "Stack Signals" in content
        assert "long_context" in content
        assert "Inspect cache allocation" in content


class TestIssueDirectoryInvariants:
    """Invariants: issue directory reflects exactly the current run."""

    def test_issue_dir_contains_only_current_run_files(self, tmp_path: Path) -> None:
        """After generation, issue_*.md files match exactly what was generated."""
        issues_dir = tmp_path / "issues"
        issues_dir.mkdir()
        (issues_dir / "issue_001_crash.md").write_text("stale")
        (issues_dir / "issue_099_harness.md").write_text("stale")
        (issues_dir / "README.md").write_text("keep me")

        # Build a snapshot with one crash cluster
        failed_result = PerformanceResult(
            model_name="org/broken-model",
            generation=None,
            success=False,
            error_message="RuntimeError: shape mismatch",
            error_signature="MLX_DECODE_ERROR:abc123",
            total_time=0.5,
        )
        snapshot = DiagnosticsSnapshot(
            failed=(failed_result,),
            failure_clusters=(("MLX_DECODE_ERROR:abc123", (failed_result,)),),
        )
        generated = _generate_github_issue_reports(
            diagnostics_snapshot=snapshot,
            output_dir=tmp_path,
            versions=_stub_versions(),
            system_info={"Python Version": "3.13"},
            repro_bundles={},
            run_args=None,
        )

        issue_files = sorted(f.name for f in issues_dir.glob("issue_*.md"))
        assert len(issue_files) == len(generated), (
            f"Expected {len(generated)} issue files, found {issue_files}"
        )
        assert (issues_dir / "index.md").exists()
        # Non-issue files must survive
        assert (issues_dir / "README.md").exists()
        # Stale files must be gone
        assert not (issues_dir / "issue_099_harness.md").exists()

    def test_empty_run_clears_all_issue_files(self, tmp_path: Path) -> None:
        """A run with no failures must leave zero issue_*.md files and refresh index."""
        issues_dir = tmp_path / "issues"
        issues_dir.mkdir()
        (issues_dir / "issue_001_crash.md").write_text("stale")

        _generate_github_issue_reports(
            diagnostics_snapshot=DiagnosticsSnapshot(),
            output_dir=tmp_path,
            versions=_stub_versions(),
            system_info={"Python Version": "3.13"},
            repro_bundles={},
            run_args=None,
        )

        issue_files = list(issues_dir.glob("issue_*.md"))
        assert issue_files == [], f"Expected no issue files, found {issue_files}"
        assert (issues_dir / "index.md").exists()
