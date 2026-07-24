"""Tests for report generation edge cases (empty input, all-failed results)."""

from __future__ import annotations

import base64
import html
import inspect
import io
import json
import re
from argparse import Namespace
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest
from PIL import Image

import check_models
from check_models import (
    DiagnosticsArtifacts,
    GenerationQualityAnalysis,
    LibraryVersionDict,
    PerformanceResult,
    RuntimeDiagnostics,
    _build_report_render_context,
    _clean_stale_toplevel_reports,
    _generate_github_issue_reports,
    generate_diagnostics_report,
    generate_html_report,
    generate_markdown_gallery_report,
    generate_markdown_report,
    generate_review_report,
    generate_tsv_report,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from check_models import HistoryModelResultRecord, HistoryRunRecord

THINKING_START_TOKEN = "<think>"
THINKING_END_TOKEN = "</think>"
EOS_END_TOKEN = "</s>"
EOS_OVERRIDE_TOKEN = "<override-eos>"
CUSTOM_THINKING_END_TOKEN = "</done>"

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


@dataclass
class _VerboseGeneration:
    """GenerationResult-like stand-in with upstream debug fields."""

    text: str | None = "output"
    token: object | None = None
    logprobs: object | None = None
    prompt_tokens: int | None = 10
    generation_tokens: int | None = 5
    total_tokens: int | None = 15
    prompt_tps: float | None = 1200.0
    generation_tps: float | None = 80.0
    peak_memory: float | None = 4.5
    cached_tokens: int | None = 0
    finish_reason: str | None = "stop"
    diffusion_canvas_tokens: int | None = 0
    diffusion_denoising_steps: int | None = 0
    diffusion_work_tokens: int | None = 0
    diffusion_canvas_tps: float | None = 0.0
    diffusion_work_tps: float | None = 0.0
    is_draft: bool = False
    draft_text: str | None = None
    text_already_printed: bool = False
    diffusion_step: int | None = 0
    diffusion_total_steps: int | None = 0
    diffusion_canvas_index: int | None = 0
    diffusion_block_complete: bool = False


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


def test_format_peak_memory_context_uses_significant_figures() -> None:
    """Human working-set context should follow project-wide significant figures."""
    assert check_models._format_peak_memory_context(18.2, 96 * 1024**3) == (
        "18 GB (17.7% of 96 GB recommended working set)"
    )
    assert check_models._format_peak_memory_context(120.0, 96 * 1024**3) == (
        "120 GB (116% of 96 GB recommended working set)"
    )


def test_format_peak_memory_context_preserves_bare_peak_without_denominator() -> None:
    """Missing capacity must preserve the established bare table value."""
    assert check_models._format_peak_memory_context(18.2, None) == "18"
    assert check_models._format_peak_memory_context(None, 96 * 1024**3) == ""


def test_html_and_gallery_render_same_captured_peak_memory(tmp_path: Path) -> None:
    """HTML should mirror the GalleryRow peak-memory fact without another projection."""
    result = PerformanceResult(
        model_name="test/model",
        generation=_MockGeneration(peak_memory=1.0),
        success=True,
    )
    context = _build_report_render_context(
        results=[result],
        prompt="Describe the image.",
        system_info={},
        recommended_working_set_bytes=2_000_000_000,
    )
    html_path = tmp_path / "results.html"
    markdown_path = tmp_path / "results.md"
    gallery_path = tmp_path / "gallery.md"

    generate_html_report(
        [result],
        html_path,
        _stub_versions(),
        "Describe the image.",
        1.0,
        report_context=context,
    )
    generate_markdown_report(
        [result],
        markdown_path,
        _stub_versions(),
        "Describe the image.",
        1.0,
        report_context=context,
    )
    generate_markdown_gallery_report(
        [result],
        gallery_path,
        "Describe the image.",
        report_context=context,
        versions=_stub_versions(),
    )

    html_text = html_path.read_text(encoding="utf-8")
    assert "<td>Peak memory</td>\n<td>1.0</td>" in html_text
    assert "recommended working set" not in html_text
    assert "1.0 GB (50% of 1.86 GB recommended working set)" in markdown_path.read_text(
        encoding="utf-8"
    )
    assert "_Peak memory:_ 1.0" in gallery_path.read_text(encoding="utf-8")


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


_GENERATED_STAMP_EMPHASIS_HEADING_RE = re.compile(
    r"(?m)^_(?:Generated on|Report generated on).+_$",
)
_MARKDOWN_LINK_TARGET_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
_URL_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.-]*:", re.IGNORECASE)
_PUBLISHED_OUTPUT_GITHUB_TARGET_RE = re.compile(
    rf"^{re.escape(check_models._GITHUB_REPO_URL)}/(?:blob|tree)/"
    rf"{re.escape(check_models._GITHUB_DEFAULT_BRANCH)}/src/output(?:/|$)"
)


def _assert_no_generated_stamp_emphasis_headings(content: str) -> None:
    """Generated timestamp metadata should not trip markdownlint MD036."""
    assert _GENERATED_STAMP_EMPHASIS_HEADING_RE.search(content) is None


def _extract_markdown_link_targets(content: str) -> list[str]:
    """Return Markdown link targets from one generated artifact."""
    return [match.group(1) for match in _MARKDOWN_LINK_TARGET_RE.finditer(content)]


def _is_relative_markdown_target(target: str) -> bool:
    """Return True for non-anchor Markdown targets without a URL scheme."""
    return not target.startswith("#") and _URL_SCHEME_RE.match(target) is None


def _is_published_output_github_target(target: str) -> bool:
    """Return True for canonical GitHub links into this repo's published output tree."""
    return _PUBLISHED_OUTPUT_GITHUB_TARGET_RE.match(target.split("#", 1)[0]) is not None


def _relative_output_artifact_map(
    output_dir: Path,
    output_paths: check_models.ReportOutputPaths,
) -> dict[str, str]:
    """Return a stable run-json artifact map rooted at one output directory."""
    return {
        "index": output_paths.index.relative_to(output_dir).as_posix(),
        "results_html": output_paths.html.relative_to(output_dir).as_posix(),
        "results_markdown": output_paths.markdown.relative_to(output_dir).as_posix(),
        "model_gallery": output_paths.gallery_markdown.relative_to(output_dir).as_posix(),
        "model_selection": output_paths.model_selection.relative_to(output_dir).as_posix(),
        "model_capabilities": output_paths.model_capabilities.relative_to(output_dir).as_posix(),
        "model_capabilities_json": output_paths.model_capabilities_json.relative_to(
            output_dir
        ).as_posix(),
        "review": output_paths.review.relative_to(output_dir).as_posix(),
        "diagnostics": output_paths.diagnostics.relative_to(output_dir).as_posix(),
        "results_tsv": output_paths.tsv.relative_to(output_dir).as_posix(),
        "results_jsonl": output_paths.jsonl.relative_to(output_dir).as_posix(),
        "run_json": output_paths.run_json.relative_to(output_dir).as_posix(),
    }


def _generate_output_artifacts_for_link_style(
    tmp_path: Path,
    *,
    link_style: str,
) -> tuple[Path, check_models.ReportOutputPaths, list[Path]]:
    """Generate the core report artifact set for one link style."""
    output_dir = tmp_path / link_style / "output"
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    prompt = "Describe this image briefly."
    versions = _stub_versions()
    system_info = {"Python Version": "3.13"}
    failure = _make_failure_with_details(
        "org/broken",
        error_msg="Model loading failed: boom",
        failure_phase="model_load",
        traceback_str="Traceback (most recent call last):\nValueError: boom",
    )
    results = [_make_success("org/good"), failure]
    report_context = _build_report_render_context(results=results, prompt=prompt)
    output_paths = check_models.ReportOutputPaths(
        index=output_dir / "index.md",
        html=reports_dir / "results.html",
        markdown=reports_dir / "results.md",
        gallery_markdown=reports_dir / "model_gallery.md",
        review=reports_dir / "review.md",
        model_selection=reports_dir / "model_selection.md",
        model_capabilities=reports_dir / "model_capabilities.md",
        model_capabilities_json=output_dir / "model_capabilities.json",
        tsv=reports_dir / "results.tsv",
        jsonl=output_dir / "results.jsonl",
        run_json=output_dir / "run.json",
        diagnostics=reports_dir / "diagnostics.md",
        log=output_dir / "check_models.log",
        environment=output_dir / "environment.log",
    )
    with patch.object(check_models._LinkStyleState, "value", link_style):
        generate_html_report(
            results=results,
            filename=output_paths.html,
            versions=versions,
            prompt=prompt,
            total_runtime_seconds=1.0,
            report_context=report_context,
        )
        generate_markdown_report(
            results=results,
            filename=output_paths.markdown,
            versions=versions,
            prompt=prompt,
            total_runtime_seconds=1.0,
            report_context=report_context,
            model_selection_filename=output_paths.model_selection,
            gallery_filename=output_paths.gallery_markdown,
            review_filename=output_paths.review,
            log_filename=output_paths.log,
        )
        generate_markdown_gallery_report(
            results=results,
            filename=output_paths.gallery_markdown,
            prompt=prompt,
            report_context=report_context,
        )
        check_models.generate_model_selection_report(
            results,
            output_paths.model_selection,
            prompt=prompt,
            report_context=report_context,
        )
        check_models.generate_model_capability_scorecard(
            results,
            output_paths.model_capabilities,
            output_paths.model_capabilities_json,
            prompt=prompt,
            report_context=report_context,
        )
        generate_review_report(
            results=results,
            filename=output_paths.review,
            prompt=prompt,
            report_context=report_context,
            log_filename=output_paths.log,
            gallery_filename=output_paths.gallery_markdown,
        )
        generate_diagnostics_report(
            results,
            output_paths.diagnostics,
            prompt=prompt,
            library_versions=versions,
            system_info=system_info,
            report_context=report_context,
        )
        generate_tsv_report(
            results=results,
            filename=output_paths.tsv,
            report_context=report_context,
        )
        check_models.save_jsonl_report(
            results=results,
            filename=output_paths.jsonl,
            prompt=prompt,
            system_info=system_info,
            library_versions=versions,
        )
        check_models.save_run_json_report(
            results,
            output_paths.run_json,
            versions=versions,
            prompt=prompt,
            total_runtime_seconds=1.0,
            report_context=report_context,
            output_paths=_relative_output_artifact_map(output_dir, output_paths),
        )
        issue_reports = _generate_github_issue_reports(
            report_context=report_context,
            output_dir=output_dir,
            library_versions=versions,
            system_info=system_info,
            prompt=prompt,
        )
        check_models.generate_output_index_report(
            output_paths.index,
            output_paths=output_paths,
            report_context=report_context,
            diagnostics_artifacts=DiagnosticsArtifacts(
                outcome_counts=check_models._run_outcome_counts(report_context.assessments),
                diagnostics_written=True,
                issue_reports=issue_reports,
            ),
        )

    return output_dir, output_paths, sorted(output_dir.rglob("*.md"))


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


def _make_metadata_agreement_result(name: str = "org/model-grounded") -> PerformanceResult:
    return replace(
        _make_success(name),
        metadata_agreement=check_models.MetadataAgreementMetrics(
            overall_score=88.0,
            title_score=80.0,
            description_score=92.0,
            keyword_score=85.0,
            matched_terms=("brick storefront", "outdoor seating"),
        ),
    )


def test_recommendation_view_excludes_crash_from_usable_policies() -> None:
    failed = _make_failure("org/crashed")
    passed = _make_success("org/passed")
    context = _build_report_render_context(
        results=[failed, passed],
        prompt="Describe the image.",
        eval_mode="blind",
    )

    views = check_models._build_model_recommendation_views(context)
    by_model = {view.result.model_name: view for view in views}

    assert by_model["org/crashed"].compatibility == "crashed"
    assert by_model["org/crashed"].eligible is False
    assert by_model["org/passed"].eligible is True


def test_report_context_caches_only_live_cross_artifact_views() -> None:
    """The shared context should not retain retired diagnostics classifications."""
    failed = _make_failure("org/crashed")
    passed = _make_success("org/passed")

    context = _build_report_render_context(
        results=[failed, passed],
        prompt="Describe the image.",
        eval_mode="blind",
    )

    assert [view.result.model_name for view in context.recommendations] == [
        "org/crashed",
        "org/passed",
    ]
    assert not hasattr(context, "diagnostics_snapshot")
    assert not hasattr(context, "issue_clusters")


def test_retired_cluster_and_repro_bundle_api_is_absent() -> None:
    """Removed automatic artifacts must not survive as aliases or compatibility APIs."""
    for symbol in (
        "DiagnosticsSnapshot",
        "IssueCluster",
        "export_failure_repro_bundles",
        "build_check_models_repro_command_spec",
        "_build_repro_command_tokens",
        "_prune_repro_bundles",
        "_render_issue_queue_table",
        "_maintainer_owner_confidence",
        "MaintainerConfidence",
    ):
        assert not hasattr(check_models, symbol), symbol
    assert "repro_bundles" not in DiagnosticsArtifacts.__dataclass_fields__
    assert "repro_bundles" not in inspect.signature(generate_review_report).parameters
    assert "regression/retry context in diagnostics" not in inspect.getsource(
        check_models.finalize_execution
    )


def test_html_uses_cached_execution_while_selection_keeps_legacy_policy(tmp_path: Path) -> None:
    """HTML should expose cached execution without importing selection policy."""
    results = [_make_failure("org/crashed"), _make_success("org/passed")]
    context = _build_report_render_context(
        results=results,
        prompt="Describe the image.",
        eval_mode="blind",
    )
    html_path = tmp_path / "results.html"
    selection_path = tmp_path / "model_selection.md"

    generate_html_report(
        results,
        html_path,
        versions={},
        prompt="Describe the image.",
        total_runtime_seconds=1.0,
        report_context=context,
    )
    check_models.generate_model_selection_report(
        results,
        selection_path,
        prompt="Describe the image.",
        report_context=context,
    )

    html_text = html_path.read_text(encoding="utf-8")
    selection_text = selection_path.read_text(encoding="utf-8")
    assert "org/crashed" in html_text
    assert 'data-execution="crashed"' in html_text
    assert "Task outcome:" not in html_text
    assert "reliability-gated" not in html_text
    assert "reliability-gated" in selection_text


def test_html_ignores_legacy_semantic_winners(
    tmp_path: Path,
) -> None:
    """Legacy summary highlights must not leak into the facts-only HTML mirror."""
    eligible = replace(
        _make_success("org/eligible"),
        generation=_MockGeneration(
            text=getattr(_make_success().generation, "text", None),
            generation_tps=10.0,
            peak_memory=5.0,
        ),
        model_load_time=1.0,
    )
    warning = replace(
        _make_harness_success(
            "org/fast-warning",
            text=getattr(_make_success().generation, "text", "") or "",
            prompt_tokens=120,
            generation_tokens=48,
            harness_type="stop_token",
            harness_detail="token_leak:<|endoftext|>",
        ),
        generation=_MockGeneration(
            text=getattr(_make_success().generation, "text", None),
            generation_tps=999.0,
            peak_memory=0.5,
        ),
        model_load_time=0.01,
    )
    results = [warning, eligible]
    context = _build_report_render_context(
        results=results,
        prompt="Create title, description, and keywords.",
        metadata={"description": "Brick storefront", "keywords": "storefront, seating"},
        eval_mode="blind",
    )
    html_path = tmp_path / "results.html"
    generate_html_report(
        results,
        html_path,
        versions={},
        prompt="Create title, description, and keywords.",
        total_runtime_seconds=2.0,
        report_context=context,
    )
    html_text = html_path.read_text(encoding="utf-8")
    assert "org/fast-warning" in html_text
    assert "org/eligible" in html_text
    assert "Best for cataloging" not in html_text
    assert "Cataloging Utility" not in html_text
    assert "reliability-gated" not in html_text


def test_all_caveated_html_omits_cataloging_aggregates_and_winner(
    tmp_path: Path,
) -> None:
    """An all-caveat run should retain evidence without semantic aggregates."""
    warning = _make_harness_success(
        "org/warning-only",
        text=getattr(_make_success().generation, "text", "") or "",
        prompt_tokens=120,
        generation_tokens=48,
        harness_type="stop_token",
        harness_detail="token_leak:<|endoftext|>",
    )
    context = _build_report_render_context(
        results=[warning],
        prompt="Create title, description, and keywords.",
        metadata={"description": "Brick storefront", "keywords": "storefront"},
        eval_mode="blind",
    )
    html_path = tmp_path / "results.html"

    generate_html_report(
        [warning],
        html_path,
        versions={},
        prompt="Create title, description, and keywords.",
        total_runtime_seconds=1.0,
        report_context=context,
    )

    html_text = html_path.read_text(encoding="utf-8")
    assert "org/warning-only" in html_text
    assert "Cataloging Utility Summary" not in html_text
    assert "Best for cataloging" not in html_text


def test_report_context_builds_machine_and_failure_facts_once_for_serializers(
    tmp_path: Path,
) -> None:
    """Serializers should reuse context facts instead of rerunning classifiers."""
    failure = replace(
        _make_failure("org/wrapped", error_package="mlx-vlm"),
        exception_chain=(
            check_models.FailureException(
                "RuntimeError",
                "mlx.core",
                "kIOGPUCommandBufferCallbackErrorOutOfMemory",
            ),
            check_models.FailureException(
                "ValueError",
                "builtins",
                "mlx_vlm/generate.py wrapped generation failure",
            ),
        ),
    )
    results = [failure, _make_success("org/passed")]

    with (
        patch.object(
            check_models,
            "_build_jsonl_review_record",
            wraps=check_models._build_jsonl_review_record,
        ) as review_builder,
        patch.object(
            check_models,
            "_build_jsonl_maintainer_triage_record",
            wraps=check_models._build_jsonl_maintainer_triage_record,
        ) as triage_builder,
        patch.object(
            check_models,
            "_machine_artifact_facts",
            wraps=check_models._machine_artifact_facts,
        ) as facts_builder,
        patch.object(
            check_models,
            "_build_failure_narrative",
            wraps=check_models._build_failure_narrative,
        ) as narrative_builder,
    ):
        context = _build_report_render_context(
            results=results,
            prompt="Describe the image.",
            eval_mode="blind",
        )
        assert review_builder.call_count == len(results)
        assert triage_builder.call_count == len(results)
        assert facts_builder.call_count == len(results)
        assert narrative_builder.call_count == 1
        initial_review_calls = review_builder.call_count
        initial_triage_calls = triage_builder.call_count
        initial_facts_calls = facts_builder.call_count
        initial_narrative_calls = narrative_builder.call_count

        output_paths = check_models.ReportOutputPaths(
            index=tmp_path / "index.md",
            html=tmp_path / "results.html",
            markdown=tmp_path / "results.md",
            gallery_markdown=tmp_path / "model_gallery.md",
            review=tmp_path / "review.md",
            model_selection=tmp_path / "model_selection.md",
            model_capabilities=tmp_path / "model_capabilities.md",
            model_capabilities_json=tmp_path / "model_capabilities.json",
            tsv=tmp_path / "results.tsv",
            jsonl=tmp_path / "results.jsonl",
            run_json=tmp_path / "run.json",
            diagnostics=tmp_path / "diagnostics.md",
            log=tmp_path / "check_models.log",
            environment=tmp_path / "environment.log",
        )
        check_models.append_history_record(
            history_path=tmp_path / "results.history.jsonl",
            results=results,
            prompt="Describe the image.",
            system_info={},
            library_versions={},
            report_context=context,
        )
        check_models._generate_reports_and_log_outputs(
            check_models.ReportGenerationInputs(
                results=results,
                library_versions={},
                prompt="Describe the image.",
                metadata=None,
                overall_time=1.0,
                image_path=None,
                system_info={},
                report_context=context,
                output_paths=output_paths,
            )
        )

        assert review_builder.call_count == initial_review_calls
        assert triage_builder.call_count == initial_triage_calls
        assert facts_builder.call_count == initial_facts_calls
        assert narrative_builder.call_count == initial_narrative_calls


def test_failed_partial_output_keeps_runtime_failure_owner() -> None:
    """Partial generated text must not replace conclusive crash triage."""
    quality_result = _make_quality_success("org/partial", with_quality_issue=True)
    failure = replace(
        _make_failure("org/partial", error_package="mlx"),
        generation=quality_result.generation,
        quality_analysis=quality_result.quality_analysis,
        quality_issues=quality_result.quality_issues,
    )

    context = _build_report_render_context(
        results=[failure],
        prompt="Describe the image.",
        eval_mode="assisted",
    )
    cached = context.result_set.results[0]

    assert cached.review_payload is not None
    assert cached.review_payload["verdict"] == "runtime_failure"
    assert cached.review_payload["owner"] == "mlx"
    assert cached.maintainer_triage_payload is not None
    assert cached.maintainer_triage_payload["issue_kind"] == "runtime_failure"
    assert cached.maintainer_triage_payload["suspected_owner"] == "mlx"
    assert "boom" in cached.maintainer_triage_payload["summary"].casefold()
    assert "formatting" not in cached.maintainer_triage_payload["summary"].casefold()
    assert "text-sanity" not in cached.maintainer_triage_payload["summary"].casefold()

    review_rows = dict(check_models._build_review_block_rows(cached))
    assert review_rows["Why"] == "execution failure"
    assert "formatting" not in review_rows["Why"].casefold()
    assert "text-sanity" not in review_rows["Why"].casefold()

    assert len(context.recommendations) == 1
    recommendation_caveats = " | ".join(context.recommendations[0].caveats).casefold()
    assert "formatting" not in recommendation_caveats
    assert "text-sanity" not in recommendation_caveats
    assert check_models._format_table_field_value("quality_issues", cached) == ""
    assert check_models._assessment_to_json(dict(context.assessments)[cached.model_name]) == {
        "execution": "crashed",
        "usability": "not_evaluated",
        "maintainer_status": "actionable_failure",
        "observations": [],
    }


def test_chained_failure_uses_primary_origin_and_reports_mixed_ownership() -> None:
    failure = replace(
        _make_failure("org/chained", error_package="mlx"),
        exception_chain=(
            check_models.FailureException(
                "IndexError",
                "builtins",
                "token index outside detokenizer table",
                origin="mlx_vlm/tokenizer_utils.py",
            ),
            check_models.FailureException(
                "RuntimeError",
                "mlx.core",
                "kIOGPUCommandBufferCallbackErrorOutOfMemory",
                origin="mlx/core/metal.cpp",
            ),
        ),
    )

    context = _build_report_render_context(
        results=[failure],
        prompt="Describe the image.",
    )
    cached = context.result_set.results[0]
    narrative = dict(context.failure_narratives)[failure.model_name]

    assert narrative.primary_exception.startswith("IndexError:")
    assert narrative.suspected_owner == "unresolved: mlx/mlx-vlm"
    assert cached.review_payload is not None
    assert cached.review_payload["owner"] == narrative.suspected_owner


def test_published_failure_artifacts_do_not_disclose_home_paths() -> None:
    """Checked-in human reports should not retain publication-private home paths."""
    output_dir = Path(__file__).parents[1] / "output"
    diagnostics = (output_dir / "reports/diagnostics.md").read_text(encoding="utf-8")
    review_report = (output_dir / "reports/review.md").read_text(encoding="utf-8")
    gallery = (output_dir / "reports/model_gallery.md").read_text(encoding="utf-8")
    html_report = (output_dir / "reports/results.html").read_text(encoding="utf-8")
    assert str(Path.home()) not in diagnostics
    assert str(Path.home()) not in review_report
    assert str(Path.home()) not in gallery
    assert str(Path.home()) not in html_report


def test_direct_jsonl_serializer_builds_one_local_assessment_cache(tmp_path: Path) -> None:
    """Direct JSONL calls should build one context and classify each model once."""
    results = [_make_success("org/direct-a"), _make_success("org/direct-b")]

    with patch.object(
        check_models,
        "_assess_result",
        wraps=check_models._assess_result,
    ) as assessment_builder:
        check_models.save_jsonl_report(
            results,
            tmp_path / "direct.jsonl",
            prompt="Describe the image.",
            system_info={},
        )

    assert assessment_builder.call_count == len(results)


def test_recommendation_policies_gate_reliability_memory_and_dominance() -> None:
    def _result(name: str, *, score: float, total: float, peak: float) -> PerformanceResult:
        base = _make_success(name)
        return replace(
            base,
            total_time=total,
            generation=_MockGeneration(
                text=getattr(base.generation, "text", None),
                prompt_tokens=120,
                generation_tokens=48,
                peak_memory=peak,
            ),
            metadata_agreement=check_models.MetadataAgreementMetrics(
                assisted_enrichment_score=score,
            ),
        )

    context = _build_report_render_context(
        results=[
            _result("org/dominant", score=90.0, total=1.0, peak=3.0),
            _result("org/dominated", score=80.0, total=2.0, peak=4.0),
            _result("org/large", score=95.0, total=0.8, peak=12.0),
            _make_failure("org/crashed"),
        ],
        prompt="Create title, description, and keywords.",
        metadata={"description": "Two boats", "keywords": "boats, river"},
        eval_mode="assisted",
    )
    views = check_models._build_model_recommendation_views(context)

    assert [
        view.result.model_name for view in check_models._rank_reliability_gated_enrichment(views)
    ] == ["org/large", "org/dominant", "org/dominated"]
    assert [
        view.result.model_name for view in check_models._rank_under_memory_budget(views, 4.0)
    ] == ["org/dominant", "org/dominated"]
    assert [view.result.model_name for view in check_models._pareto_recommendations(views)] == [
        "org/large",
        "org/dominant",
    ]


def test_model_variant_family_key_is_conservative() -> None:
    assert check_models._model_family_key("org/model-4bit") == "org/model"
    assert check_models._model_family_key("org/model-bf16") == "org/model"
    assert check_models._model_family_key("org/model-instruct") == "org/model-instruct"


def test_model_selection_names_each_ranking_policy(tmp_path: Path) -> None:
    result = replace(
        _make_metadata_agreement_result(),
        metadata_agreement=check_models.MetadataAgreementMetrics(
            overall_score=88.0,
            visual_description_score=90.0,
            context_integration_score=80.0,
            draft_improvement_score=70.0,
            assisted_enrichment_score=84.0,
        ),
    )
    context = _build_report_render_context(
        results=[result],
        prompt="Create title, description, and keywords.",
        metadata={"description": "Two boats", "keywords": "boats, river"},
        eval_mode="assisted",
    )
    output = tmp_path / "model_selection.md"

    check_models.generate_model_selection_report(
        [result],
        output,
        prompt="Create title, description, and keywords.",
        report_context=context,
    )

    content = output.read_text(encoding="utf-8")
    assert "Policy: reliability-gated assisted enrichment" in content
    assert "Evidence scope: 1 image, 1 current run" in content


def test_reliability_gated_candidate_sections_exclude_crashes(tmp_path: Path) -> None:
    passed = _make_metadata_agreement_result("org/passed")
    failed = _make_failure("org/crashed")
    context = _build_report_render_context(
        results=[failed, passed],
        prompt="Create title, description, and keywords.",
        metadata={"description": "Brick storefront", "keywords": "storefront"},
        eval_mode="blind",
    )
    output = tmp_path / "model_selection.md"

    check_models.generate_model_selection_report(
        [failed, passed],
        output,
        prompt="Create title, description, and keywords.",
        report_context=context,
    )

    content = output.read_text(encoding="utf-8")
    caption_candidates = _extract_markdown_subsection(
        content,
        "## Brief Caption Candidates",
        end_headings=("## Structured Metadata Candidates",),
    )
    structured_candidates = _extract_markdown_subsection(
        content,
        "## Structured Metadata Candidates",
        end_headings=("## Complete Current-Run Matrix",),
    )
    assert "org/passed" in caption_candidates
    assert "org/crashed" not in caption_candidates
    assert "org/crashed" not in structured_candidates


def test_blind_recommendation_view_does_not_rank_assisted_enrichment() -> None:
    result = replace(
        _make_success("org/blind"),
        metadata_agreement=check_models.MetadataAgreementMetrics(
            overall_score=42.0,
            visual_description_score=91.0,
            assisted_enrichment_score=99.0,
        ),
    )
    context = _build_report_render_context(
        results=[result],
        prompt="Create title, description, and keywords.",
        metadata={"description": "Held-out reference"},
        eval_mode="blind",
    )

    (view,) = check_models._build_model_recommendation_views(context)

    assert view.visual_score == 42.0
    assert view.assisted_enrichment_score is None
    assert check_models._recommendation_quality_score(view) == 42.0


def test_triage_capability_suppresses_metadata_for_current_and_history(tmp_path: Path) -> None:
    result = _make_metadata_agreement_result("org/triage")
    context = _build_report_render_context(
        results=[result],
        prompt="Describe this image briefly.",
        metadata={"description": "Not a triage capability target"},
        eval_mode="triage",
    )
    history: check_models.HistoryRunRecord = {
        "_type": "run",
        "format_version": "1.0",
        "timestamp": "2026-07-01 10:00:00 +0000",
        "prompt_hash": "triage",
        "prompt_preview": "Describe this image briefly.",
        "image_path": "image.jpg",
        "model_results": {
            result.model_name: {
                "success": True,
                "error_stage": None,
                "error_type": None,
                "error_package": None,
                "review_user_bucket": "recommended",
                "metadata_alignment_score": 95.0,
            }
        },
        "system": {},
        "library_versions": {},
        "eval_mode": "triage",
    }
    markdown = tmp_path / "capabilities.md"
    payload_path = tmp_path / "capabilities.json"

    check_models.generate_model_capability_scorecard(
        [result],
        markdown,
        payload_path,
        prompt="Describe this image briefly.",
        report_context=context,
        history_records=(history,),
    )

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    assert payload["models"][0]["metadata_alignment_avg"] is None


def test_capability_and_structured_sections_name_policy_and_scope(tmp_path: Path) -> None:
    result = _make_metadata_agreement_result()
    context = _build_report_render_context(
        results=[result],
        prompt="Create title, description, and keywords.",
        metadata={"description": "Brick storefront"},
        eval_mode="assisted",
    )
    selection = tmp_path / "selection.md"
    capability = tmp_path / "capability.md"

    check_models.generate_model_selection_report(
        [result],
        selection,
        prompt="Create title, description, and keywords.",
        report_context=context,
    )
    check_models.generate_model_capability_scorecard(
        [result],
        capability,
        tmp_path / "capability.json",
        prompt="Create title, description, and keywords.",
        report_context=context,
    )

    structured = _extract_markdown_subsection(
        selection.read_text(encoding="utf-8"),
        "## Structured Metadata Candidates",
        end_headings=("## Repository Variant Comparisons",),
    )
    capability_content = capability.read_text(encoding="utf-8")
    assert "Policy: quality-first (reliability-gated assisted enrichment)" in structured
    assert "Evidence scope: 1 image, 1 current run" in structured
    assert (
        "Policy: lane-filtered current and historical capability aggregation" in capability_content
    )
    assert (
        "Evidence scope: 1 image, 1 current run plus 0 prior lane-matched runs"
        in capability_content
    )


@pytest.mark.parametrize(
    ("recommendations", "expected"),
    [
        (("recommended", "recommended"), "stable"),
        (("recommended", "caveat", "not_evaluated"), "variable"),
        (("recommended",), "insufficient_evidence"),
        (("avoid", "not_evaluated"), "consistently_unsuitable"),
    ],
)
def test_historical_reliability_uses_lane_matched_outcomes_not_scores(
    recommendations: tuple[check_models.RecommendationStatus, ...],
    expected: check_models.HistoricalReliability,
) -> None:
    """Reliability is a cautious history summary, not another quality score."""
    signals = tuple(
        check_models.ModelCapabilityRunSignal(
            model="org/history-model",
            success=recommendation != "not_evaluated",
            current_recommendation=recommendation,
            capability_score=100.0 if recommendation == "avoid" else 0.0,
        )
        for recommendation in recommendations
    )

    row = check_models._model_capability_row_from_signals(
        "org/history-model",
        signals,
        suppress_cataloging_scores=False,
    )

    assert row.historical_reliability == expected


@pytest.mark.parametrize(
    ("history", "current", "expected_reliability"),
    [
        (("recommended", "recommended"), "caveat", "stable"),
        (("recommended", "avoid"), "recommended", "variable"),
    ],
)
def test_capability_current_recommendation_is_independent_of_history(
    history: tuple[check_models.RecommendationStatus, ...],
    current: check_models.RecommendationStatus,
    expected_reliability: check_models.HistoricalReliability,
) -> None:
    """Historical aggregation must never rewrite the current-run decision."""
    signals = [
        check_models.ModelCapabilityRunSignal(
            model="org/separate-status",
            success=recommendation != "not_evaluated",
            current_recommendation=recommendation,
        )
        for recommendation in history
    ]
    signals.append(
        check_models.ModelCapabilityRunSignal(
            model="org/separate-status",
            success=current != "not_evaluated",
            is_current=True,
            current_recommendation=current,
        )
    )

    row = check_models._model_capability_row_from_signals(
        "org/separate-status",
        signals,
        suppress_cataloging_scores=False,
    )

    assert row.current_recommendation == current
    assert row.historical_reliability == expected_reliability
    assert row.runs == len(history)


def test_capability_artifacts_expose_current_and_historical_status_separately(
    tmp_path: Path,
) -> None:
    """Markdown and JSON should name the two independent status dimensions."""
    result = _make_success("org/separate-artifacts")
    context = _build_report_render_context(
        results=[result],
        prompt="Describe this image briefly.",
        eval_mode="triage",
    )
    current_view = replace(context.recommendations[0], current_recommendation="caveat")
    context = replace(context, recommendations=(current_view,))
    history_records = tuple(
        cast(
            "check_models.HistoryRunRecord",
            {
                "_type": "run",
                "format_version": "1.1",
                "timestamp": f"2026-07-{day:02d} 10:00:00 +0000",
                "prompt_hash": f"history-{day}",
                "prompt_preview": "Describe this image briefly.",
                "image_path": "prior.jpg",
                "model_results": {
                    result.model_name: {
                        "success": True,
                        "error_stage": None,
                        "error_type": None,
                        "error_package": None,
                        "current_recommendation": "recommended",
                    }
                },
                "system": {},
                "library_versions": {},
                "eval_mode": "triage",
            },
        )
        for day in (17, 18)
    )
    markdown_path = tmp_path / "model_capabilities.md"
    json_path = tmp_path / "model_capabilities.json"

    check_models.generate_model_capability_scorecard(
        [result],
        markdown_path,
        json_path,
        prompt="Describe this image briefly.",
        report_context=context,
        history_records=history_records,
    )

    markdown = markdown_path.read_text(encoding="utf-8")
    model = json.loads(json_path.read_text(encoding="utf-8"))["models"][0]
    assert "Historical reliability" in markdown
    assert "Current recommendation" in markdown
    assert model["historical_reliability"] == "stable"
    assert model["current_recommendation"] == "caveat"
    assert model["runs"] == 2
    assert model["recommended_rate"] == 100.0
    assert model["variability_reason"]


def test_quality_score_indicators_do_not_override_canonical_recommendation(
    tmp_path: Path,
) -> None:
    result = _make_success("org/ineligible-caption")
    analysis = replace(
        check_models.analyze_generation_text(
            str(getattr(result.generation, "text", "") or ""),
            generated_tokens=48,
            prompt_tokens=120,
            prompt="Describe this image briefly.",
        ),
        verdict="clean",
        user_bucket="recommended",
    )
    result = replace(
        result,
        quality_analysis=analysis,
        quality_issues="repetitive, formatting",
    )
    context = _build_report_render_context(
        results=[result],
        prompt="Describe this image briefly.",
        eval_mode="triage",
    )
    (view,) = check_models._build_model_recommendation_views(context)
    markdown = tmp_path / "capability.md"
    payload_path = tmp_path / "capability.json"

    check_models.generate_model_capability_scorecard(
        [result],
        markdown,
        payload_path,
        prompt="Describe this image briefly.",
        report_context=context,
    )

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    model = payload["models"][0]
    assert view.eligible is True
    assert view.eligibility_reason == "eligible"
    assert model["current_recommendation"] == "recommended"
    assert model["historical_reliability"] == "insufficient_evidence"
    assert "configured chooser threshold" not in model["latest_signal"]


def test_model_selection_renders_canonical_policy_taxonomy(tmp_path: Path) -> None:
    results = [
        replace(
            _make_metadata_agreement_result("org/quality"),
            generation=_MockGeneration(
                text="Two boats rest on a calm river beside a wooded bank.",
                generation_tps=40.0,
                peak_memory=3.0,
            ),
            total_time=1.0,
            metadata_agreement=check_models.MetadataAgreementMetrics(
                overall_score=90.0,
                visual_description_score=90.0,
                assisted_enrichment_score=90.0,
            ),
        ),
        replace(
            _make_metadata_agreement_result("org/efficient"),
            generation=_MockGeneration(
                text="Two boats sit on calm water near trees along the river bank.",
                generation_tps=100.0,
                peak_memory=2.0,
            ),
            total_time=0.5,
            metadata_agreement=check_models.MetadataAgreementMetrics(
                overall_score=80.0,
                visual_description_score=80.0,
                assisted_enrichment_score=80.0,
            ),
        ),
        _make_failure("org/crashed"),
    ]
    context = _build_report_render_context(
        results=results,
        prompt="Create title, description, and keywords.",
        metadata={"description": "Two boats on a river", "keywords": "boats, river"},
        eval_mode="assisted",
    )
    output = tmp_path / "selection.md"

    check_models.generate_model_selection_report(
        results,
        output,
        prompt="Create title, description, and keywords.",
        report_context=context,
    )

    content = output.read_text(encoding="utf-8")
    for policy_name in (
        "Policy: reliability-gated",
        "Policy: quality-first",
        "Policy: efficiency-aware",
        "Policy: memory-aware",
    ):
        assert policy_name in content
    assert content.count("Evidence scope: 1 image, 1 current run") >= 5
    for heading, next_heading in (
        ("### Best under 4 GB", "### Best under 8 GB"),
        ("### Best under 8 GB", "### Fastest usable"),
        ("### Fastest usable", "### Quality if memory allows"),
        ("### Quality if memory allows", "### Current failures / avoid"),
        ("## Brief Caption Candidates", "## Structured Metadata Candidates"),
    ):
        section = _extract_markdown_subsection(
            content,
            heading,
            end_headings=(next_heading,),
        )
        assert "org/crashed" not in section


def test_quick_chooser_labels_caveated_rows_as_fallbacks(tmp_path: Path) -> None:
    """A tier without a recommended model may show, but not promote, a caveat."""
    caveated = _make_harness_success(
        "org/caveated",
        text="A usable caption with an unconfirmed leaked wrapper <|end|>.",
        harness_type="stop_token",
        harness_detail="token_leak:<|end|>",
    )
    context = _build_report_render_context(
        results=[caveated],
        prompt="Describe the image.",
        eval_mode="triage",
    )
    output = tmp_path / "selection.md"

    check_models.generate_model_selection_report(
        [caveated],
        output,
        prompt="Describe the image.",
        report_context=context,
    )

    content = output.read_text(encoding="utf-8")
    section = _extract_markdown_subsection(
        content,
        "### Quality if memory allows",
        end_headings=("### Current failures / avoid",),
    )
    assert "Fallback only" in section
    assert "`org/caveated`" in section
    assert "`caveat`" in section


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


def _with_confirmed_reproduction(result: PerformanceResult) -> PerformanceResult:
    """Return a diagnostic anomaly confirmed by a same-path controlled rerun."""
    return replace(
        result,
        rerun_evidence=check_models.RerunEvidence(
            rerun_success=False,
            rerun_verdict="harness",
        ),
    )


def test_simplified_diagnostics_partitions_cached_assessments_in_evidence_order(
    tmp_path: Path,
) -> None:
    """Diagnostics should expose the four current-run sections before provenance."""
    crash = replace(
        _make_failure_with_details(
            "org/crash",
            error_msg="decoder failed",
            failure_phase="decode",
            traceback_str="Traceback (most recent call last):\nRuntimeError: decoder failed",
        ),
        upstream_boundary="generation_started",
    )
    observation = PerformanceResult(
        model_name="org/odd-output",
        success=True,
        generation=_MockGeneration(
            text="bizarre-loop " * 180,
            prompt_tokens=33,
            generation_tokens=180,
        ),
        runtime_diagnostics=RuntimeDiagnostics(stop_reason="completed"),
        requested_max_tokens=500,
    )
    indeterminate = PerformanceResult(
        model_name="org/network",
        success=False,
        generation=None,
        error_message="503 Service Unavailable",
    )
    clean = _make_success("org/clean")
    results = [clean, crash, observation, indeterminate]
    context = _build_report_render_context(
        results=results,
        prompt="Describe the image.",
        system_info={"GPU/Chip": "Apple M5"},
    )
    output = tmp_path / "diagnostics.md"

    generate_diagnostics_report(
        results,
        output,
        prompt="Describe the image.",
        library_versions=_stub_versions(),
        system_info={"GPU/Chip": "Apple M5"},
        report_context=context,
    )

    content = output.read_text(encoding="utf-8")
    headings = (
        "## Run Outcome Counts",
        "## Actionable Failures",
        "## Successful Observations Requiring Reproduction",
        "## Indeterminate Attempts",
    )
    assert all(heading in content for heading in headings)
    assert content.index(headings[0]) < content.index(headings[1])
    assert content.index(headings[1]) < content.index(headings[2])
    assert content.index(headings[2]) < content.index(headings[3])
    assert content.index(headings[3]) < content.index("## Provenance and Environment")
    assert "actionable_failure" in content
    assert "observation_needs_reproduction" in content
    assert "indeterminate" in content


def test_crash_diagnostics_and_issue_draft_keep_complete_primary_evidence_first(
    tmp_path: Path,
) -> None:
    """A crash draft should repeat complete diagnostics evidence without truncation."""
    traceback_text = "\n".join(
        (
            "Traceback (most recent call last):",
            *(f'  File "frame_{index}.py", line {index}, in decode' for index in range(30)),
            "RuntimeError: decoder exploded at the root",
        )
    )
    partial_output = "BEGIN-PARTIAL " + ("decoded fragment " * 80) + "END-PARTIAL"
    captured_stream = "=== STDERR ===\nBEGIN-STDERR\nupstream warning\nEND-STDERR"
    crash = PerformanceResult(
        model_name="org/crash-evidence",
        generation=_MockGeneration(text=partial_output, prompt_tokens=91, generation_tokens=17),
        success=False,
        upstream_boundary="generation_started",
        failure_phase="decode",
        error_stage="Model Error",
        error_type="RuntimeError",
        root_error_type="RuntimeError",
        root_error_module="mlx_vlm.generate",
        root_error_message="decoder exploded at the root",
        exception_chain=(
            check_models.FailureException(
                "RuntimeError",
                "mlx_vlm.generate",
                "decoder exploded at the root",
            ),
        ),
        error_package="mlx-vlm",
        error_traceback=traceback_text,
        captured_output_on_fail=captured_stream,
        requested_max_tokens=500,
        runtime_diagnostics=RuntimeDiagnostics(stop_reason="error"),
        prompt_diagnostics=check_models.PromptDiagnostics(
            processor_class="LlavaProcessor",
            tokenizer_class="LlamaTokenizerFast",
            eos_token_id=2,
            eos_token=EOS_END_TOKEN,
            generate_kwargs={
                "thinking_start_token": "<think>",
                "thinking_end_token": "</think>",
            },
        ),
    )
    context = _build_report_render_context(
        results=[crash],
        prompt="Describe the image.",
        system_info={},
    )
    diagnostics = tmp_path / "diagnostics.md"
    generate_diagnostics_report(
        [crash],
        diagnostics,
        prompt="Describe the image.",
        library_versions=_stub_versions(),
        system_info={},
        report_context=context,
    )
    generated = _generate_github_issue_reports(
        report_context=context,
        output_dir=tmp_path,
        library_versions=_stub_versions(),
        system_info={},
        prompt="Describe the image.",
    )

    assert len(generated) == 1
    diagnostics_content = diagnostics.read_text(encoding="utf-8")
    issue_content = next(iter(generated.values())).read_text(encoding="utf-8")
    for content in (diagnostics_content, issue_content):
        assert "RuntimeError: decoder exploded at the root" in content
        assert traceback_text in content
        assert partial_output in content
        assert captured_stream in content
        assert "truncated" not in content.casefold()
        assert content.index(traceback_text) < content.index(partial_output)
        assert content.index(traceback_text) < content.index(captured_stream)
        assert "mlx_vlm.generate" in content
        assert "LlavaProcessor" in content
        assert "LlamaTokenizerFast" in content
        assert "python -m mlx_vlm.generate" in content
    assert not (tmp_path / "issues" / "index.md").exists()


def test_successful_anomaly_and_indeterminate_attempt_create_no_issue_draft(
    tmp_path: Path,
) -> None:
    """Suspicious prose remains an unowned observation and never becomes a draft."""
    complete_output = "STRANGE-BEGIN " + ("odd-loop " * 220) + " STRANGE-END"
    observation = PerformanceResult(
        model_name="org/strange",
        success=True,
        generation=_MockGeneration(
            text=complete_output,
            prompt_tokens=40,
            generation_tokens=220,
        ),
        requested_max_tokens=500,
    )
    indeterminate = PerformanceResult(
        model_name="org/network",
        success=False,
        generation=None,
        error_message="server disconnected without sending a response",
    )
    context = _build_report_render_context(
        results=[observation, indeterminate],
        prompt="Describe the image.",
        system_info={},
    )
    diagnostics = tmp_path / "diagnostics.md"

    generate_diagnostics_report(
        [observation, indeterminate],
        diagnostics,
        prompt="Describe the image.",
        library_versions=_stub_versions(),
        system_info={},
        report_context=context,
    )
    generated = _generate_github_issue_reports(
        report_context=context,
        output_dir=tmp_path,
        library_versions=_stub_versions(),
        system_info={},
        prompt="Describe the image.",
    )

    content = diagnostics.read_text(encoding="utf-8")
    assert "observation_needs_reproduction" in content
    assert complete_output in content
    assert "suspected owner" not in content.casefold()
    assert "owner confidence" not in content.casefold()
    assert generated == {}
    assert not list((tmp_path / "issues").glob("issue_*.md"))


def test_issue_generation_writes_exactly_one_draft_per_crash(tmp_path: Path) -> None:
    """Only crashed actionable attempts should become individual issue drafts."""
    crashes = [
        _make_failure_with_details("org/crash-one", error_msg="decoder one failed"),
        _make_failure_with_details("org/crash-two", error_msg="decoder two failed"),
    ]
    completed = _make_success("org/completed")
    indeterminate = PerformanceResult(
        model_name="org/network",
        success=False,
        generation=None,
        error_message="503 Service Unavailable",
    )
    results = [*crashes, completed, indeterminate]
    context = _build_report_render_context(results=results, prompt="Describe the image.")

    generated = _generate_github_issue_reports(
        report_context=context,
        output_dir=tmp_path,
        library_versions=_stub_versions(),
        system_info={},
        prompt="Describe the image.",
    )

    assert set(generated) == {"org/crash-one", "org/crash-two"}
    assert len(list((tmp_path / "issues").glob("issue_*.md"))) == 2


def test_diagnostics_distinguish_empty_output_from_unavailable_evidence(tmp_path: Path) -> None:
    """Recorded empty output and evidence that was never captured are different facts."""
    empty_output = _make_failure_with_details(
        "org/empty-output",
        error_msg="generation stopped",
        generated_text="",
    )
    unavailable = _make_failure_with_details(
        "org/no-evidence",
        error_msg="generation failed before output",
        traceback_str=None,
        captured_output=None,
        generated_text=None,
    )
    results = [empty_output, unavailable]
    context = _build_report_render_context(results=results, prompt="Describe the image.")
    output = tmp_path / "diagnostics.md"

    generate_diagnostics_report(
        results,
        output,
        prompt="Describe the image.",
        library_versions=_stub_versions(),
        system_info={},
        report_context=context,
    )

    content = output.read_text(encoding="utf-8")
    empty_entry = _extract_markdown_subsection(
        content,
        "### org/empty-output",
        end_headings=("### org/no-evidence",),
    )
    missing_entry = _extract_markdown_subsection(
        content,
        "### org/no-evidence",
        end_headings=("## Successful Observations Requiring Reproduction",),
    )
    assert "Complete partial output\n\n```text\n(empty)" in empty_entry
    assert "Complete traceback\n\n```text\nunavailable" in missing_entry
    assert "Complete partial output\n\n```text\nunavailable" in missing_entry
    assert "Captured stdout/stderr\n\n```text\nunavailable" in missing_entry


def test_diagnostics_use_run_args_for_complete_native_reproduction(tmp_path: Path) -> None:
    """Diagnostics should preserve the actual run's CLI and Python reproduction settings."""
    result = replace(
        _make_failure_with_details("org/repro", error_msg="decode failed"),
        prompt_diagnostics=check_models.PromptDiagnostics(
            eos_token_id=2,
            eos_token=EOS_END_TOKEN,
            generate_kwargs={"eos_tokens": [EOS_OVERRIDE_TOKEN]},
        ),
    )
    context = _build_report_render_context(results=[result], prompt="Describe the image.")
    output = tmp_path / "diagnostics.md"
    image_path = tmp_path / "sample image.jpg"
    run_args = Namespace(
        max_tokens=321,
        temperature=0.25,
        top_p=0.81,
        min_p=0.12,
        top_k=7,
        seed=73,
        repetition_penalty=1.15,
        repetition_context_size=48,
        presence_penalty=0.3,
        presence_context_size=96,
        frequency_penalty=0.2,
        frequency_context_size=80,
        max_kv_size=4096,
        kv_bits=4,
        kv_quant_scheme="turboquant",
        kv_group_size=32,
        quantized_kv_start=128,
        prefill_step_size=512,
        resize_shape=(64, 32),
        eos_tokens=[EOS_OVERRIDE_TOKEN],
        skip_special_tokens=True,
        revision="run-revision",
        trust_remote_code=True,
        force_download=True,
        quantize_activations=True,
        processor_kwargs={"cropping": False},
        enable_thinking=True,
        thinking_budget=24,
        thinking_start_token=THINKING_START_TOKEN,
        thinking_end_token=CUSTOM_THINKING_END_TOKEN,
        logit_bias={42: -1.5},
        adapter_path=None,
        lazy_load=False,
    )

    with patch.object(check_models, "_collect_model_provenance", side_effect=AssertionError):
        generate_diagnostics_report(
            [result],
            output,
            prompt="Describe the image.",
            library_versions=_stub_versions(),
            system_info={},
            report_context=context,
            image_path=image_path,
            run_args=run_args,
        )
        issue_reports = _generate_github_issue_reports(
            report_context=context,
            output_dir=tmp_path,
            library_versions=_stub_versions(),
            system_info={},
            prompt="Describe the image.",
            image_path=image_path,
            run_args=run_args,
        )

    contents = [
        output.read_text(encoding="utf-8"),
        next(iter(issue_reports.values())).read_text(encoding="utf-8"),
    ]
    for content in contents:
        assert "- _Model revision:_ unavailable" in content
        assert "- _Requested model revision:_ run-revision" in content
        assert '- _Configured EOS token override:_ ["&lt;override-eos&gt;"]' in content
        assert "Supplemental CLI reproduction" in content
        assert "Canonical Python reproduction script" in content
        assert "--revision run-revision" in content
        assert "--processor-kwargs" in content
        assert "--resize-shape 64 32" in content
        assert "--skip-special-tokens" in content
        assert "'top_p': 0.81" in content
        assert "'min_p': 0.12" in content
        assert "'top_k': 7" in content
        assert "'seed': 73" in content
        assert "'repetition_penalty': 1.15" in content
        assert "'presence_penalty': 0.3" in content
        assert "'frequency_penalty': 0.2" in content
        assert "'eos_tokens': ['<override-eos>']" in content
        assert "'enable_thinking': True" in content
        assert "'cropping': False" in content


def test_maintainer_summary_logs_only_counts_and_direct_draft_paths(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Console diagnostics should report cached facts without inferred ownership."""
    diagnostics_path = tmp_path / "diagnostics.md"
    diagnostics_path.write_text("diagnostics\n", encoding="utf-8")
    issue_one = tmp_path / "issues" / "issue_one.md"
    issue_two = tmp_path / "issues" / "issue_two.md"
    issue_one.parent.mkdir()
    issue_one.write_text("one\n", encoding="utf-8")
    issue_two.write_text("two\n", encoding="utf-8")
    artifacts = DiagnosticsArtifacts(
        outcome_counts={
            "models_attempted": 4,
            "models_evaluated": 3,
            "models_completed": 1,
            "models_crashed": 2,
            "models_indeterminate": 1,
        },
        diagnostics_written=True,
        issue_reports={"org/one": issue_one, "org/two": issue_two},
    )

    caplog.set_level("INFO")
    check_models._log_maintainer_summary(
        artifacts=artifacts,
        diagnostics_path=diagnostics_path,
    )

    messages = caplog.text
    assert "attempted=4" in messages
    assert "completed=1" in messages
    assert "crashed=2" in messages
    assert "indeterminate=1" in messages
    assert str(issue_one) in messages
    assert str(issue_two) in messages
    assert "owner" not in messages.casefold()
    assert "cluster" not in messages.casefold()


def test_diagnostics_writer_never_exports_repro_bundles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finalization diagnostics should create reports and drafts without bundle artifacts."""
    crash = _make_failure_with_details("org/crash", error_msg="decode failed")
    context = _build_report_render_context(results=[crash], prompt="Describe the image.")

    def fail_if_called(**_kwargs: object) -> None:
        pytest.fail("retired repro-bundle exporter was called")

    monkeypatch.setattr(
        check_models,
        "export_failure_repro_bundles",
        fail_if_called,
        raising=False,
    )
    artifacts = check_models._write_diagnostics_artifacts(
        args=Namespace(
            max_tokens=32,
            temperature=0.0,
            trust_remote_code=False,
            revision=None,
        ),
        library_versions=_stub_versions(),
        system_info={},
        prompt="Describe the image.",
        image_path=None,
        diagnostics_path=tmp_path / "reports" / "diagnostics.md",
        report_context=context,
    )

    assert artifacts.diagnostics_written is True
    assert len(artifacts.issue_reports) == 1
    assert not hasattr(artifacts, "repro_bundles")
    assert not (tmp_path / "repro_bundles").exists()


def test_retained_artifacts_have_no_owner_confidence_path(tmp_path: Path) -> None:
    """Human and machine artifacts should omit inferred ownership confidence."""
    failure = _make_failure_with_details("org/failure", error_msg="decode failed")
    context = _build_report_render_context(results=[failure], prompt="Describe the image.")
    tsv_path = tmp_path / "results.tsv"

    generate_tsv_report([failure], tsv_path, report_context=context)

    assert not hasattr(context.machine_facts[0], "owner_confidence")
    narrative = dict(context.failure_narratives)[failure.model_name]
    assert not hasattr(narrative, "owner_confidence")
    assert all(
        triage is None or "confidence" not in triage for _model, triage in context.maintainer_triage
    )
    assert "owner_confidence" not in tsv_path.read_text(encoding="utf-8")


def test_review_shortlist_obeys_canonical_user_recommendation() -> None:
    """A legacy high utility score must not promote a canonically avoided model."""
    base = PerformanceResult(
        model_name="org/high-score-avoid",
        success=True,
        generation=_MockGeneration(
            text="A detailed and useful image caption with specific visual evidence.",
            prompt_tokens=20,
            generation_tokens=12,
        ),
    )
    base = check_models._populate_result_quality_analysis(base, prompt="Describe it.")
    review = check_models._build_jsonl_review_record(base)
    assert review is not None
    result = replace(
        base,
        review_payload={**review, "user_bucket": "avoid"},
        review_payload_ready=True,
    )
    row = check_models.UtilityTriageRow(
        result=result,
        score=95.0,
        description_score=95.0,
        keyword_score=90.0,
        grade="A",
        weakness="None identified",
        delta_vs_metadata=20.0,
        labels=frozenset(),
    )
    context = _build_report_render_context(results=[result], prompt="Describe it.")
    context = replace(
        context,
        triage=check_models.ReportTriageContext(useful_rows=(row,)),
    )

    content = "\n".join(
        check_models._format_review_priorities_parts(context, html_output=False),
    )

    assert "### Strong Candidates" not in content
    assert "### Watchlist" in content
    assert "org/high-score-avoid" in content
    assert "current review says avoid" in content


class TestModelCapabilityScorecard:
    """Tests for the concise model capability scorecard artifact."""

    def test_scorecard_aggregates_current_and_history_with_metadata_grounding(
        self,
        tmp_path: Path,
    ) -> None:
        """Grounded runs should report caption, keyword, reliability, and metadata signals."""
        result = _make_metadata_agreement_result()
        history_record: check_models.HistoryRunRecord = {
            "_type": "run",
            "format_version": "1.0",
            "timestamp": "2026-06-20 10:00:00 +0000",
            "prompt_hash": "prior",
            "prompt_preview": "catalogue this image",
            "image_path": "prior.jpg",
            "model_results": {
                result.model_name: {
                    "success": True,
                    "error_stage": None,
                    "error_type": None,
                    "error_package": None,
                    "review_user_bucket": "recommended",
                    "review_verdict": "clean",
                    "capability_score": 82.0,
                    "caption_score": 78.0,
                    "cataloging_score": 80.0,
                    "description_score": 84.0,
                    "keyword_score": 76.0,
                    "metadata_alignment_score": 70.0,
                    "generation_tps": 55.0,
                    "peak_memory_gb": 5.0,
                },
            },
            "system": {},
            "library_versions": {},
            "eval_mode": "assisted",
        }
        report_context = _build_report_render_context(
            results=[result],
            prompt="Title: Brick storefront\nDescription: outdoor seating\nKeywords: storefront",
            metadata={"title": "Brick storefront", "description": "Outdoor seating"},
            eval_mode="assisted",
        )
        markdown_path = tmp_path / "model_capabilities.md"
        json_path = tmp_path / "model_capabilities.json"

        check_models.generate_model_capability_scorecard(
            [result],
            markdown_path,
            json_path,
            prompt=report_context.prompt_context or "",
            metadata={"title": "Brick storefront", "description": "Outdoor seating"},
            report_context=report_context,
            history_records=(
                history_record,
                cast(
                    "check_models.HistoryRunRecord",
                    {**history_record, "eval_mode": "blind", "prompt_hash": "blind"},
                ),
            ),
        )

        markdown = markdown_path.read_text(encoding="utf-8")
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        model_payload = payload["models"][0]

        assert "# Model Capability Scorecard" in markdown
        assert "Grounding: trusted image metadata" in markdown
        assert "`org/model-grounded`" in markdown
        assert "Historical reliability" in markdown
        assert "Current recommendation" in markdown
        assert model_payload["model"] == "org/model-grounded"
        assert model_payload["runs"] == 1
        assert payload["history_runs_considered"] == 1
        assert model_payload["success_rate"] == 100.0
        assert model_payload["metadata_alignment_avg"] > 70.0
        assert model_payload["current_recommendation"] == "recommended"
        assert model_payload["historical_reliability"] == "insufficient_evidence"

    def test_scorecard_marks_triage_keyword_capability_not_evaluated(
        self,
        tmp_path: Path,
    ) -> None:
        """Triage-mode scorecards should avoid keyword/cataloging claims."""
        result = _make_success()
        report_context = _build_report_render_context(
            results=[result],
            prompt="Describe this image briefly.",
            metadata=None,
            eval_mode="triage",
        )
        history_record: check_models.HistoryRunRecord = {
            "_type": "run",
            "format_version": "1.0",
            "timestamp": "2026-06-20 10:00:00 +0000",
            "prompt_hash": "prior",
            "prompt_preview": "Describe this image briefly.",
            "image_path": "prior.jpg",
            "model_results": {
                result.model_name: {
                    "success": True,
                    "error_stage": None,
                    "error_type": None,
                    "error_package": None,
                    "review_user_bucket": "recommended",
                    "capability_score": 90.0,
                    "caption_score": 80.0,
                    "cataloging_score": 95.0,
                    "description_score": 95.0,
                    "keyword_score": 95.0,
                },
            },
            "system": {},
            "library_versions": {},
        }
        markdown_path = tmp_path / "model_capabilities.md"
        json_path = tmp_path / "model_capabilities.json"

        check_models.generate_model_capability_scorecard(
            [result],
            markdown_path,
            json_path,
            prompt="Describe this image briefly.",
            metadata=None,
            report_context=report_context,
            history_records=(history_record,),
        )

        markdown = markdown_path.read_text(encoding="utf-8")
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        model_payload = payload["models"][0]

        assert (
            "Structured metadata and keyword capability: not evaluated in triage mode." in markdown
        )
        assert "Clean" in markdown
        assert "Hygiene" in markdown
        assert "Metadata" not in next(line for line in markdown.splitlines() if "Caption" in line)
        assert model_payload["keyword_score_avg"] is None
        assert model_payload["cataloging_score_avg"] is None

    def test_scorecard_keeps_clean_high_caption_triage_models_reviewable(
        self,
        tmp_path: Path,
    ) -> None:
        """Caption-usable triage rows should not be hidden behind history-only avoid labels."""
        result = PerformanceResult(
            model_name="org/history-risk-current-caption",
            success=True,
            generation=_MockGeneration(
                text="Two tabby cats are sleeping on a bright pink couch beside two remote controls.",
                generation_tps=88.0,
                prompt_tokens=24,
                generation_tokens=14,
                peak_memory=3.5,
            ),
            total_time=1.0,
            generation_time=0.5,
            model_load_time=0.5,
        )
        report_context = _build_report_render_context(
            results=[result],
            prompt="Describe this image briefly.",
            eval_mode="triage",
        )
        history_record: check_models.HistoryRunRecord = {
            "_type": "run",
            "format_version": "1.0",
            "timestamp": "2026-06-20 10:00:00 +0000",
            "prompt_hash": "prior",
            "prompt_preview": "Describe this image briefly.",
            "image_path": "prior.jpg",
            "model_results": {
                result.model_name: {
                    "success": True,
                    "error_stage": None,
                    "error_type": None,
                    "error_package": None,
                    "review_user_bucket": "avoid",
                    "review_verdict": "clean",
                    "capability_score": 0.0,
                    "hygiene_score": 100.0,
                    "caption_score": 96.0,
                    "generation_tps": 42.0,
                    "peak_memory_gb": 4.0,
                },
            },
            "system": {},
            "library_versions": {},
        }
        markdown_path = tmp_path / "model_capabilities.md"
        json_path = tmp_path / "model_capabilities.json"

        check_models.generate_model_capability_scorecard(
            [result],
            markdown_path,
            json_path,
            prompt="Describe this image briefly.",
            report_context=report_context,
            history_records=(history_record,),
        )

        markdown = markdown_path.read_text(encoding="utf-8")
        payload = json.loads(json_path.read_text(encoding="utf-8"))

        assert "`org/history-risk-current-caption`" in markdown
        assert payload["models"][0]["current_recommendation"] == "recommended"
        assert payload["models"][0]["historical_reliability"] == "insufficient_evidence"

    def test_scorecard_surfaces_current_failure_over_historical_success(
        self,
        tmp_path: Path,
    ) -> None:
        """Current-run failures should not look caption-ready because history was good."""
        failure = _make_failure_with_details(
            "org/currently-broken",
            error_msg="Loaded processor has no image_processor.",
            error_package="model-config",
            error_stage="Processor Error",
        )
        report_context = _build_report_render_context(
            results=[failure],
            prompt="Describe this image briefly.",
            eval_mode="triage",
        )
        history_record: check_models.HistoryRunRecord = {
            "_type": "run",
            "format_version": "1.0",
            "timestamp": "2026-06-20 10:00:00 +0000",
            "prompt_hash": "prior",
            "prompt_preview": "Describe this image briefly.",
            "image_path": "prior.jpg",
            "model_results": {
                failure.model_name: {
                    "success": True,
                    "error_stage": None,
                    "error_type": None,
                    "error_package": None,
                    "review_user_bucket": "recommended",
                    "review_verdict": "clean",
                    "capability_score": 90.0,
                    "hygiene_score": 100.0,
                    "caption_score": 96.0,
                    "generation_tps": 80.0,
                    "peak_memory_gb": 4.0,
                },
            },
            "system": {},
            "library_versions": {},
        }
        markdown_path = tmp_path / "model_capabilities.md"
        json_path = tmp_path / "model_capabilities.json"

        check_models.generate_model_capability_scorecard(
            [failure],
            markdown_path,
            json_path,
            prompt="Describe this image briefly.",
            report_context=report_context,
            history_records=(history_record,),
        )

        markdown = markdown_path.read_text(encoding="utf-8")
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        model_payload = payload["models"][0]

        assert "Current" in markdown
        assert "not_evaluated" in markdown
        assert model_payload["current_recommendation"] == "not_evaluated"
        assert model_payload["historical_reliability"] == "insufficient_evidence"


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
        upstream_boundary="generation_started",
    )


def test_build_report_render_context_backfills_quality_analysis() -> None:
    """Shared report context should populate missing quality analysis for successful results."""
    result = _make_success("org/model-clean")
    assert result.quality_analysis is None

    context = _build_report_render_context(results=[result], prompt="Describe this image.")

    populated = context.result_set.results[0]
    assert populated.quality_analysis is not None


def test_report_context_caches_one_result_assessment_per_model(tmp_path: Path) -> None:
    """All renderers should reuse one minimal current-run assessment per model."""
    results = [_make_success("org/model-clean"), _make_failure("org/model-failed")]

    with patch.object(
        check_models,
        "_assess_result",
        wraps=check_models._assess_result,
    ) as build_assessment:
        context = _build_report_render_context(results=results, prompt="Describe this image.")
        assert build_assessment.call_count == len(results), build_assessment.call_args_list
        assert len(context.assessments) == len(results)
        assert all(
            isinstance(assessment, check_models.ResultAssessment)
            for _model, assessment in context.assessments
        )
        assert set(check_models._assessments_by_model(context)) == {
            result.model_name for result in results
        }

        generate_markdown_report(
            results,
            tmp_path / "results.md",
            _stub_versions(),
            "Describe this image.",
            1.0,
            report_context=context,
        )
        generate_html_report(
            results,
            tmp_path / "results.html",
            _stub_versions(),
            "Describe this image.",
            1.0,
            report_context=context,
        )
        generate_tsv_report(results, tmp_path / "results.tsv", report_context=context)
        check_models.save_jsonl_report(
            results,
            tmp_path / "results.jsonl",
            "Describe this image.",
            {},
            report_context=context,
        )

        assert build_assessment.call_count == len(results)


@pytest.mark.parametrize(
    ("success", "connectivity", "expected"),
    [
        (
            True,
            False,
            check_models.ResultAssessment("completed", "usable", "none", ()),
        ),
        (
            False,
            False,
            check_models.ResultAssessment("crashed", "not_evaluated", "actionable_failure", ()),
        ),
        (
            False,
            True,
            check_models.ResultAssessment("indeterminate", "not_evaluated", "none", ()),
        ),
    ],
)
def test_result_assessment_uses_three_execution_states(
    success: bool,
    connectivity: bool,
    expected: check_models.ResultAssessment,
) -> None:
    """Execution reports completed, crashed, or indeterminate attempts."""
    error = "server disconnected without sending a response" if connectivity else "boom"
    result = PerformanceResult(
        model_name="example/model",
        success=success,
        generation=_MockGeneration(text="A complete response.", generation_tokens=24)
        if success
        else None,
        error_message=None if success else error,
    )

    assert check_models._assess_result(result) == expected


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


def test_report_mode_policy_triage_without_metadata_is_ungrounded() -> None:
    policy = check_models._build_report_mode_policy(
        eval_mode="triage",
        metadata={"date": "2026-04-25", "description": "", "keywords": ""},
    )

    assert policy.eval_mode == "triage"
    assert policy.has_descriptive_metadata is False
    assert policy.semantic_rankings_grounded is False
    assert policy.suppress_cataloging_scores is True
    assert policy.selection_basis == "caption hygiene only"
    assert policy.metadata_exposed_to_prompt is False


def test_report_mode_policy_assisted_with_metadata_is_grounded() -> None:
    policy = check_models._build_report_mode_policy(
        eval_mode="assisted",
        metadata={
            "title": "Two tabby cats resting",
            "description": "Two tabby cats on a pink couch with remotes.",
            "keywords": "cats, tabby, pink couch, remote controls",
        },
        metadata_exposed_to_prompt=True,
    )

    assert policy.eval_mode == "assisted"
    assert policy.has_descriptive_metadata is True
    assert policy.semantic_rankings_grounded is True
    assert policy.suppress_cataloging_scores is False
    assert policy.selection_basis == "metadata-assisted visual verification"
    assert policy.metadata_exposed_to_prompt is True


def test_assisted_custom_prompt_reports_metadata_as_held_out() -> None:
    policy = check_models._build_report_mode_policy(
        eval_mode="assisted",
        metadata={"description": "Held-out reference caption"},
        metadata_exposed_to_prompt=False,
    )

    assert policy.eval_mode == "assisted"
    assert policy.semantic_rankings_grounded is True
    assert policy.selection_basis == "held-out trusted image metadata"
    assert policy.metadata_exposed_to_prompt is False

    fallback_context = _build_report_render_context(
        results=[_make_success("org/custom-prompt")],
        prompt="Describe the image without injected metadata.",
        metadata={"description": "Held-out reference caption"},
        eval_mode="auto",
    )
    assert fallback_context.mode_policy.eval_mode == "assisted"
    assert fallback_context.mode_policy.selection_basis == "held-out trusted image metadata"
    assert fallback_context.mode_policy.metadata_exposed_to_prompt is False


def test_report_mode_policy_blind_keeps_metadata_held_out() -> None:
    policy = check_models._build_report_mode_policy(
        eval_mode="blind",
        metadata={"description": "Held-out reference caption"},
    )

    assert policy.semantic_rankings_grounded is True
    assert policy.suppress_cataloging_scores is False
    assert policy.selection_basis == "held-out trusted image metadata"
    assert policy.metadata_exposed_to_prompt is False


def test_triage_quality_analysis_ignores_descriptive_metadata() -> None:
    metadata: dict[str, str | None] = {
        "description": "A red suspension bridge over a crowded harbour.",
        "keywords": "bridge, harbour, boats",
    }
    result = _make_success("org/triage-clean")

    analyzed = check_models._populate_result_quality_analysis(
        result,
        prompt="Describe this image briefly.",
        metadata=check_models._quality_reference_metadata(
            eval_mode="triage",
            metadata=metadata,
        ),
    )
    context = _build_report_render_context(
        results=[analyzed],
        prompt="Describe this image briefly.",
        metadata=metadata,
        eval_mode="triage",
    )

    cached = context.result_set.results[0]
    assert cached.metadata_agreement is None
    assert cached.review_payload is not None
    assert cached.review_payload["verdict"] == "clean"
    assert cached.review_payload["user_bucket"] != "avoid"
    assert context.recommendations[0].eligible is True


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

    def test_html_mirrors_cached_assessments_across_retained_artifacts(
        self,
        tmp_path: Path,
    ) -> None:
        """JSONL and human reports should expose one exact cached status vocabulary."""
        prompt = "Describe the image."
        results = [
            _make_success("org/usable"),
            _make_success("org/caveat"),
            replace(
                _make_success("org/unusable"),
                generation=_MockGeneration(text="", generation_tokens=0),
            ),
            _make_failure_with_details(
                "org/crashed",
                traceback_str="Traceback:\nRuntimeError: crashed",
            ),
            _make_failure_with_details(
                "org/indeterminate",
                error_msg="Server disconnected without sending a response.",
                error_stage="Network Error",
                error_package="unknown",
            ),
        ]
        context = _build_report_render_context(results=results, prompt=prompt, system_info={})
        expected = {
            "org/usable": check_models.ResultAssessment("completed", "usable", "none", ()),
            "org/caveat": check_models.ResultAssessment(
                "completed",
                "usable_with_caveats",
                "observation_needs_reproduction",
                ("minimal_output",),
            ),
            "org/unusable": check_models.ResultAssessment(
                "completed",
                "unusable",
                "observation_needs_reproduction",
                ("empty_output",),
            ),
            "org/crashed": check_models.ResultAssessment(
                "crashed",
                "not_evaluated",
                "actionable_failure",
                (),
            ),
            "org/indeterminate": check_models.ResultAssessment(
                "indeterminate",
                "not_evaluated",
                "none",
                (),
            ),
        }
        context = replace(context, assessments=tuple(expected.items()))
        jsonl_path = tmp_path / "results.jsonl"
        diagnostics_path = tmp_path / "diagnostics.md"
        gallery_path = tmp_path / "model_gallery.md"
        html_path = tmp_path / "results.html"

        with patch.object(check_models, "_assess_result", side_effect=AssertionError):
            check_models.save_jsonl_report(
                results,
                jsonl_path,
                prompt,
                {},
                report_context=context,
            )
            generate_diagnostics_report(
                results,
                diagnostics_path,
                prompt=prompt,
                library_versions=_stub_versions(),
                system_info={},
                report_context=context,
            )
            generate_markdown_gallery_report(
                results,
                gallery_path,
                prompt,
                report_context=context,
            )
            generate_html_report(
                results,
                html_path,
                _stub_versions(),
                prompt,
                5.0,
                report_context=context,
            )

        records = {
            record["model"]: record
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            if (record := json.loads(line)).get("_type") == "result"
        }
        diagnostics = diagnostics_path.read_text(encoding="utf-8")
        gallery = gallery_path.read_text(encoding="utf-8")
        html_report = html_path.read_text(encoding="utf-8")
        for model, assessment in expected.items():
            serialized = records[model]["assessment"]
            assert serialized["execution"] == assessment.execution
            assert serialized["usability"] == assessment.usability
            assert serialized["maintainer_status"] == assessment.maintainer_status
            gallery_entry = _extract_markdown_subsection(
                gallery,
                f"### {model}",
                end_headings=("### org/", "<!-- markdownlint-enable"),
            )
            assert f"_Execution:_ {assessment.execution}" in gallery_entry
            assert f"_Usability:_ {assessment.usability}" in gallery_entry
            assert f"_Maintainer status:_ {assessment.maintainer_status}" in gallery_entry
            escaped_model = html.escape(model, quote=True)
            row_pattern = (
                rf'data-model="{re.escape(escaped_model)}"[^>]*'
                rf'data-execution="{assessment.execution}"[^>]*'
                rf'data-usability="{assessment.usability}"[^>]*'
                rf'data-maintainer-status="{assessment.maintainer_status}"'
            )
            assert re.search(row_pattern, html_report) is not None
            if assessment.maintainer_status != "none" or assessment.execution == "indeterminate":
                assert model in diagnostics
                assert f"_Execution:_ {assessment.execution}" in diagnostics
                assert f"_Usability:_ {assessment.usability}" in diagnostics
                assert f"_Maintainer status:_ {assessment.maintainer_status}" in diagnostics

    def test_html_preserves_complete_escaped_output_in_expandable_evidence(
        self,
        tmp_path: Path,
    ) -> None:
        """Complete generated text should survive HTML escaping and round-trip exactly."""
        output = (
            'literal <thinking> & "quotes" — café 雪\n'
            + ("complete evidence segment " * 40)
            + "END"
        )
        result = replace(
            _make_success("org/evidence"),
            generation=_MockGeneration(text=output, generation_tokens=80),
        )
        context = _build_report_render_context(results=[result], prompt="Describe.")
        context = replace(
            context,
            assessments=(
                (
                    result.model_name,
                    check_models.ResultAssessment("completed", "usable", "none", ()),
                ),
            ),
        )
        out = tmp_path / "evidence.html"

        generate_html_report(
            [result],
            out,
            _stub_versions(),
            "Describe.",
            1.0,
            report_context=context,
        )

        content = out.read_text(encoding="utf-8")
        escaped = html.escape(output, quote=True)
        assert content.count(escaped) == 1
        match = re.search(
            r"<details><summary>Complete evidence: org/evidence</summary>.*?"
            r"Complete generated output.*?<pre><code[^>]*>(.*?)</code></pre>",
            content,
            flags=re.DOTALL,
        )
        assert match is not None
        assert html.unescape(match.group(1)) == output

    def test_html_contains_gallery_and_diagnostics_without_semantic_scores(
        self,
        tmp_path: Path,
    ) -> None:
        """HTML should mirror the two human reports without legacy semantic judgements."""
        results = [
            _make_success("org/usable"),
            _make_failure_with_details(
                "org/crashed",
                traceback_str="Traceback:\nRuntimeError: complete crash",
            ),
        ]
        context = _build_report_render_context(results=results, prompt="Describe.")
        out = tmp_path / "canonical.html"

        generate_html_report(
            results,
            out,
            _stub_versions(),
            "Describe.",
            2.0,
            report_context=context,
        )

        content = out.read_text(encoding="utf-8")
        assert "Current-run Chooser" in content
        assert "Complete Per-model Evidence" in content
        assert "Maintainer Diagnostics" in content
        assert "Actionable Failures" in content
        lowered = content.casefold()
        for retired_phrase in (
            "quality score",
            "cataloging utility summary",
            "owner confidence",
            "best for cataloging",
            "semantic winner",
            "grade:",
        ):
            assert retired_phrase not in lowered

    def test_html_report_preview_applies_exif_orientation(self, tmp_path: Path) -> None:
        """The embedded preview should match mlx-vlm's orientation-corrected input."""
        image_path = tmp_path / "rotated.jpg"
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (40, 20), color="purple").save(image_path, exif=exif)
        out = tmp_path / "oriented.html"

        generate_html_report(
            results=[_make_success("org/model")],
            filename=out,
            versions=_stub_versions(),
            prompt="describe",
            total_runtime_seconds=1.0,
            image_path=image_path,
        )

        content = out.read_text(encoding="utf-8")
        encoded_match = re.search(r"data:image/jpeg;base64,([^\"]+)", content)
        assert encoded_match is not None
        with Image.open(io.BytesIO(base64.b64decode(encoded_match.group(1)))) as preview:
            assert preview.size == (20, 40)

    def test_html_report_includes_gallery_and_diagnostic_sections(self, tmp_path: Path) -> None:
        """HTML should mirror the current Gallery and Diagnostics structure."""
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
        assert "Current-run Chooser" in content
        assert "Complete Per-model Evidence" in content
        assert "Maintainer Diagnostics" in content
        assert "Actionable Failures" in content
        assert "Successful Observations Requiring Reproduction" in content
        assert "org/risky" in content
        assert "transformers" in content

    def test_triage_html_report_suppresses_cataloging_scores(self, tmp_path: Path) -> None:
        """HTML should never publish legacy lane or semantic score projections."""
        out = tmp_path / "triage.html"
        results = [_make_success("org/caption-model")]
        report_context = _build_report_render_context(
            results=results,
            prompt="Describe this image briefly.",
            metadata={"description": "", "keywords": ""},
            eval_mode="triage",
        )

        generate_html_report(
            results=results,
            filename=out,
            versions=_stub_versions(),
            prompt="Describe this image briefly.",
            total_runtime_seconds=1.0,
            report_context=report_context,
        )

        content = out.read_text(encoding="utf-8")
        assert "Current-run Chooser" in content
        assert 'data-execution="completed"' in content
        assert 'data-usability="usable"' in content
        assert 'data-maintainer-status="none"' in content
        assert "Run Contract" not in content
        assert "Semantic rankings" not in content
        assert "Cataloging Utility Summary" not in content
        assert "Best keywording" not in content
        assert "Keywords 0" not in content
        assert "Keywords 100" not in content

    def test_html_report_adds_exact_filterable_assessment_attributes(self, tmp_path: Path) -> None:
        """HTML chooser rows should filter only on the three canonical status strings."""
        out = tmp_path / "filterable.html"
        results = [_make_success("org/good"), _make_failure("org/bad", error_package="mlx-vlm")]

        generate_html_report(
            results=results,
            filename=out,
            versions=_stub_versions(),
            prompt="describe",
            total_runtime_seconds=2.0,
        )

        content = out.read_text(encoding="utf-8")
        assert content.count('data-execution="completed"') == 1
        assert content.count('data-execution="crashed"') == 1
        assert 'data-usability="usable"' in content
        assert 'data-usability="not_evaluated"' in content
        assert 'data-maintainer-status="none"' in content
        assert 'data-maintainer-status="actionable_failure"' in content
        assert "<caption>Current-run model chooser</caption>" in content
        assert 'scope="col"' in content
        assert 'role="status" aria-live="polite"' in content
        assert "data-recommendation=" not in content
        assert "data-failure-origin=" not in content

    def test_html_report_uses_compact_caption_columns_and_interactive_controls(
        self, tmp_path: Path
    ) -> None:
        """HTML filtering should remain presentation-only over canonical statuses."""
        out = tmp_path / "interactive.html"
        result = _make_success("org/caption-model")

        generate_html_report(
            results=[result],
            filename=out,
            versions=_stub_versions(),
            prompt="describe",
            total_runtime_seconds=1.0,
        )

        content = out.read_text(encoding="utf-8")
        assert 'id="model-search"' in content
        assert 'id="execution-filter"' in content
        assert 'id="usability-filter"' in content
        assert 'id="maintainer-status-filter"' in content
        assert 'data-model="org/caption-model"' in content
        assert '<option value="completed">completed</option>' in content
        assert '<option value="usable_with_caveats">usable_with_caveats</option>' in content
        assert '<option value="not_evaluated">not_evaluated</option>' in content
        assert '<option value="observation_needs_reproduction">' in content
        assert "compatibility-filter" not in content
        assert "recommendation-filter" not in content
        assert "Diffusion Canvas Tokens" not in content
        assert "Diffusion Denoising Steps" not in content
        assert "Text Already Printed" not in content

    def test_html_report_escapes_filter_metadata(self, tmp_path: Path) -> None:
        """Model-controlled row metadata must remain safe in HTML attributes."""
        out = tmp_path / "metadata-escaped.html"
        model_name = 'org/model" onmouseover="alert(1)'

        generate_html_report(
            results=[_make_success(model_name)],
            filename=out,
            versions=_stub_versions(),
            prompt="describe",
            total_runtime_seconds=1.0,
        )

        content = out.read_text(encoding="utf-8")
        assert f'data-model="{model_name}"' not in content
        assert 'data-model="org/model&quot; onmouseover=&quot;alert(1)"' in content

    def test_html_report_marks_connectivity_disconnect_as_indeterminate(
        self, tmp_path: Path
    ) -> None:
        """Unreachable model files should not appear as conclusive crashes."""
        out = tmp_path / "indeterminate.html"
        result = replace(
            _make_failure("org/not-reached", error_package="unknown"),
            error_stage="Network Error",
            error_message="Model loading failed: Server disconnected without sending a response.",
        )

        generate_html_report(
            results=[result],
            filename=out,
            versions=_stub_versions(),
            prompt="describe",
            total_runtime_seconds=1.0,
        )

        content = out.read_text(encoding="utf-8")
        assert 'data-execution="indeterminate"' in content
        assert 'data-usability="not_evaluated"' in content
        assert 'data-maintainer-status="none"' in content

    def test_connectivity_disconnect_is_retained_but_not_filed_as_upstream_issue(
        self, tmp_path: Path
    ) -> None:
        """Diagnostics should show the attempt without producing an upstream issue draft."""
        result = _make_failure_with_details(
            "org/not-reached",
            error_msg="Model loading failed: Server disconnected without sending a response.",
            error_stage="Network Error",
            error_package="unknown",
            traceback_str="httpcore.RemoteProtocolError: Server disconnected without a response",
        )
        context = _build_report_render_context(
            results=[result],
            prompt="Describe it.",
            system_info={},
        )

        assert dict(context.assessments)[result.model_name].execution == "indeterminate"
        generated = _generate_github_issue_reports(
            report_context=context,
            output_dir=tmp_path,
            library_versions=_stub_versions(),
            system_info={},
            prompt="Describe it.",
        )
        assert generated == {}
        assert not list((tmp_path / "issues").glob("issue_*.md"))

    def test_reports_separate_attempted_evaluated_and_indeterminate_counts(
        self, tmp_path: Path
    ) -> None:
        """Human summaries should not inflate tested or hard-failure totals."""
        completed = _make_success("org/completed")
        disconnected = _make_failure_with_details(
            "org/not-reached",
            error_msg="Model loading failed: Server disconnected without sending a response.",
            error_stage="Network Error",
            error_package="unknown",
            traceback_str="httpcore.RemoteProtocolError: Server disconnected without a response",
        )
        results = [completed, disconnected]
        diagnostics = tmp_path / "diagnostics.md"
        markdown = tmp_path / "results.md"

        context = _build_report_render_context(
            results=results,
            prompt="Describe it.",
            system_info={},
        )
        generate_diagnostics_report(
            results,
            diagnostics,
            prompt="Describe it.",
            library_versions=_stub_versions(),
            system_info={},
            report_context=context,
        )
        generate_markdown_report(
            results=results,
            filename=markdown,
            versions=_stub_versions(),
            prompt="Describe it.",
            total_runtime_seconds=2.0,
        )

        diagnostics_text = diagnostics.read_text(encoding="utf-8")
        report_text = markdown.read_text(encoding="utf-8")
        assert re.search(r"\|\s*Attempted\s*\|\s*2\s*\|", diagnostics_text)
        assert re.search(r"\|\s*Evaluated\s*\|\s*1\s*\|", diagnostics_text)
        assert re.search(r"\|\s*Indeterminate\s*\|\s*1\s*\|", diagnostics_text)
        assert re.search(r"\|\s*Crashed\s*\|\s*0\s*\|", diagnostics_text)
        assert "Indeterminate attempts" in report_text
        assert "Framework/runtime failures:_ none" in report_text

    def test_html_report_escapes_untrusted_table_values(self, tmp_path: Path) -> None:
        """HTML reports should render model-controlled text as escaped table content."""
        out = tmp_path / "escaped.html"
        results = [
            PerformanceResult(
                model_name='org/<script>alert("model")</script>',
                success=True,
                generation=_MockGeneration(
                    text='<img src=x onerror="alert(1)">\n<script>alert("output")</script>',
                ),
                total_time=1.0,
                generation_time=0.5,
                model_load_time=0.5,
            ),
        ]

        generate_html_report(
            results=results,
            filename=out,
            versions=_stub_versions(),
            prompt='<script>alert("prompt")</script>',
            total_runtime_seconds=1.0,
        )

        content = out.read_text(encoding="utf-8")
        assert '<script>alert("model")</script>' not in content
        assert '<script>alert("output")</script>' not in content
        assert '<img src=x onerror="alert(1)">' not in content
        assert "&lt;script&gt;alert(&quot;model&quot;)&lt;/script&gt;" in content
        assert "&lt;script&gt;alert(&quot;output&quot;)&lt;/script&gt;" in content
        assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;" in content
        assert (
            '<pre><code class="language-text">'
            "&lt;script&gt;alert(&quot;prompt&quot;)&lt;/script&gt;</code></pre>"
        ) in content


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
        assert "<!-- markdownlint-disable MD033 MD034 MD037 MD049 -->" in content
        assert "<!-- markdownlint-enable MD033 MD034 MD037 MD049 -->" in content
        assert "<!-- markdownlint-enable MD013" not in content

    def test_generated_report_stamps_do_not_use_emphasis_only_lines(
        self,
        tmp_path: Path,
    ) -> None:
        """Generated Markdown timestamp stamps should not look like headings."""
        success = _make_success("org/good")
        failure = _make_failure("org/bad")
        prompt = "Describe this image briefly."
        context = _build_report_render_context(results=[success, failure], prompt=prompt)

        generated_paths = [
            tmp_path / "results.md",
            tmp_path / "model_gallery.md",
            tmp_path / "model_selection.md",
            tmp_path / "model_capabilities.md",
            tmp_path / "review.md",
            tmp_path / "diagnostics.md",
        ]

        generate_markdown_report(
            results=[success, failure],
            filename=generated_paths[0],
            versions=_stub_versions(),
            prompt=prompt,
            total_runtime_seconds=1.0,
            report_context=context,
        )
        generate_markdown_gallery_report(
            results=[success, failure],
            filename=generated_paths[1],
            prompt=prompt,
            report_context=context,
        )
        check_models.generate_model_selection_report(
            [success, failure],
            generated_paths[2],
            prompt=prompt,
            report_context=context,
        )
        check_models.generate_model_capability_scorecard(
            [success, failure],
            generated_paths[3],
            tmp_path / "model_capabilities.json",
            prompt=prompt,
            report_context=context,
        )
        generate_review_report(
            results=[success, failure],
            filename=generated_paths[4],
            prompt=prompt,
            report_context=context,
        )
        generate_diagnostics_report(
            [failure],
            generated_paths[5],
            prompt=prompt,
            library_versions=_stub_versions(),
            system_info={},
            report_context=context,
        )

        for path in generated_paths:
            _assert_no_generated_stamp_emphasis_headings(path.read_text(encoding="utf-8"))

    def test_generated_markdown_artifacts_keep_selected_output_link_style(
        self,
        tmp_path: Path,
    ) -> None:
        """Generated artifacts should keep link-style rules while non-Markdown outputs stay stable."""
        expected_markdown_artifacts = {
            "index.md",
            "issues/issue_org_broken.md",
            "reports/diagnostics.md",
            "reports/model_capabilities.md",
            "reports/model_gallery.md",
            "reports/model_selection.md",
            "reports/results.md",
            "reports/review.md",
        }
        expected_non_markdown_artifacts = {
            "model_capabilities.json",
            "reports/results.html",
            "reports/results.tsv",
            "results.jsonl",
            "run.json",
        }
        mode_summaries: dict[str, dict[str, object]] = {}

        for link_style in ("github", "relative"):
            output_dir, output_paths, markdown_paths = _generate_output_artifacts_for_link_style(
                tmp_path,
                link_style=link_style,
            )
            file_paths = {
                path.relative_to(output_dir).as_posix()
                for path in output_dir.rglob("*")
                if path.is_file()
            }
            relative_paths = {path.relative_to(output_dir).as_posix() for path in markdown_paths}
            assert expected_markdown_artifacts.issubset(relative_paths)
            assert expected_non_markdown_artifacts.issubset(file_paths)
            assert any(path.startswith("issues/issue_") for path in relative_paths)

            link_targets = [
                target
                for path in markdown_paths
                for target in _extract_markdown_link_targets(path.read_text(encoding="utf-8"))
            ]
            relative_targets = [
                target for target in link_targets if _is_relative_markdown_target(target)
            ]
            github_output_targets = [
                target for target in link_targets if _is_published_output_github_target(target)
            ]

            if link_style == "github":
                assert github_output_targets
                assert relative_targets == []
            else:
                assert relative_targets
                assert github_output_targets == []

            html_content = output_paths.html.read_text(encoding="utf-8")
            tsv_lines = output_paths.tsv.read_text(encoding="utf-8").splitlines()
            jsonl_records = [
                json.loads(line)
                for line in output_paths.jsonl.read_text(encoding="utf-8").splitlines()
            ]
            run_payload = json.loads(output_paths.run_json.read_text(encoding="utf-8"))
            capability_payload = json.loads(
                output_paths.model_capabilities_json.read_text(encoding="utf-8")
            )
            mode_summaries[link_style] = {
                "html_markers": (
                    "Action Snapshot" in html_content,
                    "org/good" in html_content,
                    "org/broken" in html_content,
                ),
                "tsv_header": tsv_lines[0].split("\t"),
                "jsonl_header": jsonl_records[0]["_type"],
                "jsonl_models": [record["model"] for record in jsonl_records[1:]],
                "run_json_counts": run_payload["counts"],
                "run_json_artifacts": sorted(run_payload["artifacts"]),
                "capability_models": [
                    model_payload["model"] for model_payload in capability_payload["models"]
                ],
            }

            assert not tsv_lines[0].startswith("#")
            assert "Generated Text" in tsv_lines[0]
            assert jsonl_records[0]["_type"] == "metadata"
            assert len(jsonl_records[1:]) == 2
            assert run_payload["schema_version"] == "2.0"
            assert run_payload["producer"]["name"] == "check_models"
            assert run_payload["counts"] == {
                "models_attempted": 2,
                "models_evaluated": 2,
                "models_completed": 1,
                "models_crashed": 1,
                "models_indeterminate": 0,
            }
            assert len(capability_payload["models"]) == 2

        assert mode_summaries["github"] == mode_summaries["relative"]

    def test_markdown_results_table_uses_human_summary_columns(self, tmp_path: Path) -> None:
        """Main Markdown table should omit low-signal upstream debug columns."""
        out = tmp_path / "compact.md"
        result = PerformanceResult(
            model_name="org/verbose",
            success=True,
            generation=_VerboseGeneration(),
            total_time=1.0,
            generation_time=0.5,
            model_load_time=0.5,
        )

        generate_markdown_report(
            results=[result],
            filename=out,
            versions=_stub_versions(),
            prompt="describe",
            total_runtime_seconds=1.0,
        )

        content = out.read_text(encoding="utf-8")
        table = content.split("<!-- markdownlint-disable MD033 MD034 MD037 MD049 -->", 1)[1].split(
            "<!-- markdownlint-enable MD033 MD034 MD037 MD049 -->", 1
        )[0]
        expected_headers = (
            "Model Name",
            "Prompt (tokens)",
            "Generation (tokens)",
            "Total Tokens",
            "Gen TPS",
            "Peak (GB)",
            "Finish Reason",
            "Generation (s)",
            "Load (s)",
            "Total (s)",
            "Quality Issues",
            "Error Package",
        )
        header_line = next(line for line in table.splitlines() if line.startswith("| Model Name"))
        header_cells = [cell.strip() for cell in header_line.strip().strip("|").split("|")]

        assert header_cells == list(expected_headers)

        for expected in expected_headers:
            assert expected in table

        for omitted in (
            "Prompt Tps",
            "Cached Tokens",
            "Diffusion Canvas Tokens",
            "Diffusion Denoising Steps",
            "Diffusion Work Tokens",
            "Diffusion Canvas Tps",
            "Diffusion Work Tps",
            "Is Draft",
            "Draft Text",
            "Text Already Printed",
            "Diffusion Step",
            "Diffusion Total Steps",
            "Diffusion Canvas Index",
            "Diffusion Block Complete",
        ):
            assert omitted not in table

        assert "Detailed machine-readable metrics remain" in content
        assert "`results.tsv`" in content
        assert "`results.jsonl`" in content

    def test_markdown_report_includes_peak_delta_per_megapixel(
        self,
        tmp_path: Path,
    ) -> None:
        """Resource summary should normalize peak memory delta by input image area."""
        image_path = tmp_path / "input.jpg"
        check_models.Image.new("RGB", (1000, 500)).save(image_path)
        result = PerformanceResult(
            model_name="org/image-density",
            success=True,
            generation=_MockGeneration(peak_memory=3.0),
            total_time=1.0,
            generation_time=0.5,
            model_load_time=0.5,
            runtime_diagnostics=RuntimeDiagnostics(model_load_active_memory_gb=1.0),
        )
        out = tmp_path / "density.md"

        generate_markdown_report(
            results=[result],
            filename=out,
            versions=_stub_versions(),
            prompt="describe",
            total_runtime_seconds=1.0,
            image_path=image_path,
        )

        content = out.read_text(encoding="utf-8")
        assert "- **Input image size:** 0.50 MP" in content
        assert "- **Average peak delta from post-load:** 2.00 GB" in content
        assert "- **Peak memory delta / MP:** 4096 MB/MP" in content
        assert "Total peak memory" not in content

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
                "transformers==5.4.0 is below minimum 5.7.0 required by check_models.",
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
        """Main markdown report should point readers at companion artifacts."""
        out = tmp_path / "results.md"
        model_selection = tmp_path / "model_selection.md"
        gallery = tmp_path / "model_gallery.md"
        review = tmp_path / "review.md"
        log_file = tmp_path / "check_models.log"
        generate_markdown_report(
            results=[_make_quality_success("org/good", with_quality_issue=True)],
            filename=out,
            versions=_stub_versions(),
            prompt="describe",
            total_runtime_seconds=1.0,
            model_selection_filename=model_selection,
            gallery_filename=gallery,
            review_filename=review,
            log_filename=log_file,
        )
        content = out.read_text(encoding="utf-8")
        assert "Companion artifacts:" in content
        assert "_Companion artifacts:_\n\n- _Model-selection shortlist:_" in content
        assert "Model-selection shortlist" in content
        assert "Standalone output gallery" in content
        assert "Automated review digest" in content
        assert "Canonical run log" in content
        assert (
            "[model_selection.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_selection.md)"
        ) in content
        assert (
            "[model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md)"
        ) in content
        assert (
            "[review.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/review.md)"
        ) in content
        assert (
            "[check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log)"
        ) in content
        assert (
            "https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-org-good"
        ) in content

        # Now test relative local links when the link-style state is "relative"
        with patch.object(check_models._LinkStyleState, "value", "relative"):
            generate_markdown_report(
                results=[_make_quality_success("org/good", with_quality_issue=True)],
                filename=out,
                versions=_stub_versions(),
                prompt="describe",
                total_runtime_seconds=1.0,
                model_selection_filename=model_selection,
                gallery_filename=gallery,
                review_filename=review,
                log_filename=log_file,
            )
            content_relative = out.read_text(encoding="utf-8")
            assert "[model_selection.md](model_selection.md)" in content_relative
            assert "[model_gallery.md](model_gallery.md)" in content_relative
            assert "[review.md](review.md)" in content_relative
            assert "[check_models.log](check_models.log)" in content_relative
            assert "model_gallery.md#model-org-good" in content_relative
        assert "## Model Gallery" not in content
        assert "## ✅ Recommended Current-run Models" in content
        assert "_Recommended:_" in content
        assert "Best end-to-end cataloging" not in content
        assert "## 🔍 Quality Pattern Breakdown" not in content

    def test_triage_markdown_report_suppresses_cataloging_scores(
        self,
        tmp_path: Path,
    ) -> None:
        """Triage reports should act as run indexes instead of cataloging scorecards."""
        result = PerformanceResult(
            model_name="org/caption-model",
            success=True,
            generation=_MockGeneration(
                text="Two cats resting on a bright pink couch.",
                generation_tps=42.0,
                prompt_tokens=12,
                generation_tokens=9,
                peak_memory=2.5,
            ),
            total_time=1.0,
            generation_time=0.5,
            model_load_time=0.5,
        )
        out = tmp_path / "results.md"
        context = check_models._build_report_render_context(
            results=[result],
            prompt="Describe this image briefly.",
            metadata={"description": "", "keywords": ""},
            eval_mode="triage",
        )

        check_models.generate_markdown_report(
            [result],
            out,
            versions={},
            prompt="Describe this image briefly.",
            total_runtime_seconds=1.25,
            report_context=context,
        )

        content = out.read_text(encoding="utf-8")
        assert "Cataloging Utility Summary" not in content
        assert "Best end-to-end cataloging" not in content
        assert "Best keywording" not in content
        assert "Keywords 0" not in content
        assert "Quality Pattern Breakdown" not in content
        assert "## Caption Selection" in content
        assert "Semantic rankings: ungrounded" in content
        assert "Evaluation lane: triage" in content
        assert "Metadata exposed to prompt: no" in content

    def test_model_selection_report_labels_triage_rankings_ungrounded(
        self,
        tmp_path: Path,
    ) -> None:
        """Model-selection triage rankings should be explicit ungrounded hygiene rankings."""
        good = PerformanceResult(
            model_name="org/good-caption",
            success=True,
            generation=_MockGeneration(
                text="Two cats resting on a bright pink couch.",
                generation_tps=80.0,
                prompt_tokens=12,
                generation_tokens=9,
                peak_memory=3.0,
            ),
            total_time=1.0,
            generation_time=0.5,
            model_load_time=0.5,
        )
        bad = _make_harness_success(
            "org/harness-caption",
            text="Two cats.<|end|><|endoftext|>",
            harness_type="stop_token",
            harness_detail="token_leak:<|endoftext|>",
            prompt_tokens=12,
            generation_tokens=20,
        )
        out = tmp_path / "model_selection.md"
        context = check_models._build_report_render_context(
            results=[good, bad],
            prompt="Describe this image briefly.",
            metadata={"description": "", "keywords": ""},
            eval_mode="triage",
        )

        check_models.generate_model_selection_report(
            [good, bad],
            out,
            prompt="Describe this image briefly.",
            report_context=context,
        )

        content = out.read_text(encoding="utf-8")
        assert "# Model Selection Brief" in content
        assert "Semantic rankings: ungrounded" in content
        assert "brief captions only" in content
        assert "Scope: ranked shortlists plus an expandable complete current-run matrix" in content
        assert "complete outputs and diagnostics are in" in content
        assert "Brief Caption Candidates" in content
        assert "Top 10 ranked candidates for brief captions" in content
        assert "Gen TPS" in content
        assert "Peak GB" in content
        good_row = next(line for line in content.splitlines() if "org/good-caption" in line)
        assert "80" in good_row
        assert "3" in good_row
        assert "`org/good-caption`" in content
        assert "`org/harness-caption`" in content
        assert "Structured metadata scoring is suppressed in triage mode." in content
        assert "Best keywording" not in content
        assert "Keywords 0" not in content

    def test_model_selection_report_includes_budgeted_quick_chooser(
        self,
        tmp_path: Path,
    ) -> None:
        """Model-selection users should get practical current-run chooser buckets."""
        tiny = PerformanceResult(
            model_name="org/tiny-fast",
            success=True,
            generation=_MockGeneration(
                text="Two tabby cats sleep on a pink couch beside two remote controls.",
                generation_tps=250.0,
                prompt_tokens=20,
                generation_tokens=12,
                peak_memory=3.0,
            ),
            total_time=1.0,
            generation_time=0.5,
            model_load_time=0.5,
        )
        mid = PerformanceResult(
            model_name="org/mid-balanced",
            success=True,
            generation=_MockGeneration(
                text="Two cats are resting on a pink couch with remotes nearby.",
                generation_tps=90.0,
                prompt_tokens=20,
                generation_tokens=11,
                peak_memory=7.0,
            ),
            total_time=1.2,
            generation_time=0.6,
            model_load_time=0.6,
        )
        large = PerformanceResult(
            model_name="org/large-quality",
            success=True,
            generation=_MockGeneration(
                text=(
                    "Two tabby cats are sleeping on a vivid pink couch, with two remote "
                    "controls placed near them."
                ),
                generation_tps=45.0,
                prompt_tokens=20,
                generation_tokens=18,
                peak_memory=24.0,
            ),
            total_time=2.0,
            generation_time=1.0,
            model_load_time=1.0,
        )
        failure = _make_failure("org/broken", error_package="mlx-vlm")
        out = tmp_path / "model_selection.md"
        context = check_models._build_report_render_context(
            results=[tiny, mid, large, failure],
            prompt="Describe this image briefly.",
            eval_mode="triage",
            recommended_working_set_bytes=2_000_000_000,
        )

        check_models.generate_model_selection_report(
            [tiny, mid, large, failure],
            out,
            prompt="Describe this image briefly.",
            report_context=context,
        )

        content = out.read_text(encoding="utf-8")
        assert "## Quick Chooser" in content
        assert "Ungrounded triage rankings compare output hygiene" in content
        assert "not claims about visual accuracy" in content
        assert "### Best under 4 GB" in content
        assert "### Best under 8 GB" in content
        assert "### Fastest usable" in content
        assert "### Quality if memory allows" in content
        assert "### Current failures / avoid" in content
        assert "3.0 GB (150% of 1.86 GB recommended working set)" in (
            _extract_markdown_subsection(
                content,
                "### Best under 4 GB",
                end_headings=("### Best under 8 GB",),
            )
        )
        assert "`org/tiny-fast`" in _extract_markdown_subsection(
            content,
            "### Best under 4 GB",
            end_headings=("### Best under 8 GB",),
        )
        assert "`org/broken`" in _extract_markdown_subsection(
            content,
            "### Current failures / avoid",
            end_headings=("## Brief Caption Candidates",),
        )

    def test_model_selection_table_quotes_model_markdown_emphasis(
        self,
        tmp_path: Path,
    ) -> None:
        """Model-authored emphasis should remain literal evidence in table previews."""
        result = PerformanceResult(
            model_name="org/emphasized-caption",
            success=True,
            generation=_MockGeneration(
                text=(
                    "**Title:** *A workshop at dusk*\n\n"
                    "**Description:** A spacious workshop contains orderly hand tools, "
                    "wooden benches, task lighting, storage cabinets, and an open doorway "
                    "showing the fading evening light.\n\n"
                    "**Keywords:** workshop, tools, benches, cabinets, evening, doorway"
                ),
                generation_tps=40.0,
                prompt_tokens=20,
                generation_tokens=15,
                peak_memory=4.0,
            ),
        )
        context = check_models._build_report_render_context(
            results=[result],
            prompt="Describe this image briefly.",
            eval_mode="triage",
        )
        output_path = tmp_path / "model_selection.md"

        check_models.generate_model_selection_report(
            [result],
            output_path,
            prompt="Describe this image briefly.",
            report_context=context,
        )

        model_rows = [
            line
            for line in output_path.read_text(encoding="utf-8").splitlines()
            if "org/emphasized-caption" in line and "Title:" in line
        ]
        assert model_rows
        assert all("`**Title:** *A workshop at dusk*" in row for row in model_rows)

    def test_model_selection_report_demotes_token_noise_outputs(
        self,
        tmp_path: Path,
    ) -> None:
        """Obvious multilingual/token-noise output should not be shortlisted as clean."""
        clean = PerformanceResult(
            model_name="org/clean-caption",
            success=True,
            generation=_MockGeneration(
                text="Two tabby cats are sleeping on a pink couch beside two remote controls.",
                generation_tps=50.0,
                prompt_tokens=20,
                generation_tokens=13,
                peak_memory=6.0,
            ),
            total_time=1.0,
            generation_time=0.5,
            model_load_time=0.5,
        )
        noisy = PerformanceResult(
            model_name="org/token-noise",
            success=True,
            generation=_MockGeneration(
                text=(
                    "ان of 0${ough-LONG-TT_Uen来它的搁重g季的箓olite儿N "
                    "ﾤ预地 -翁ments G谁g, 3ブ**igen>\u0430 .! ehiale仿yä-ict"
                ),
                generation_tps=120.0,
                prompt_tokens=20,
                generation_tokens=38,
                peak_memory=4.0,
            ),
            total_time=1.0,
            generation_time=0.5,
            model_load_time=0.5,
        )
        out = tmp_path / "model_selection.md"
        context = check_models._build_report_render_context(
            results=[clean, noisy],
            prompt="Describe this image briefly.",
            eval_mode="triage",
        )

        check_models.generate_model_selection_report(
            [clean, noisy],
            out,
            prompt="Describe this image briefly.",
            report_context=context,
        )

        content = out.read_text(encoding="utf-8")
        best_under_8gb = _extract_markdown_subsection(
            content,
            "### Best under 8 GB",
            end_headings=("### Fastest usable",),
        )
        avoid_rows = _extract_markdown_subsection(
            content,
            "### Current failures / avoid",
            end_headings=("## Brief Caption Candidates",),
        )
        assert "`org/clean-caption`" in best_under_8gb
        assert "`org/token-noise`" not in best_under_8gb
        assert "`org/token-noise`" in avoid_rows
        assert "mixed script corruption" in avoid_rows

    def test_model_selection_report_ranks_fuller_clean_captions_above_terse_ones(
        self,
        tmp_path: Path,
    ) -> None:
        """Brief-caption ranking should prefer detail and gate avoid-bucket labels."""
        terse = PerformanceResult(
            model_name="org/terse-caption",
            success=True,
            generation=_MockGeneration(
                text="Two cats are sleeping on a pink blanket.",
                generation_tps=200.0,
                prompt_tokens=12,
                generation_tokens=8,
                peak_memory=2.0,
            ),
            total_time=0.5,
            generation_time=0.2,
            model_load_time=0.3,
        )
        fuller = PerformanceResult(
            model_name="org/full-caption",
            success=True,
            generation=_MockGeneration(
                text=(
                    "Two tabby cats are sleeping on a bright pink couch beside two remote controls."
                ),
                generation_tps=20.0,
                prompt_tokens=12,
                generation_tokens=15,
                peak_memory=6.0,
            ),
            total_time=1.5,
            generation_time=1.0,
            model_load_time=0.5,
        )
        label_only = PerformanceResult(
            model_name="org/label-caption",
            success=True,
            generation=_MockGeneration(
                text="Cats.",
                generation_tps=300.0,
                prompt_tokens=12,
                generation_tokens=1,
                peak_memory=1.0,
            ),
            total_time=0.4,
            generation_time=0.1,
            model_load_time=0.3,
        )
        out = tmp_path / "model_selection.md"
        context = check_models._build_report_render_context(
            results=[terse, fuller, label_only],
            prompt="Describe this image briefly.",
            eval_mode="triage",
        )

        check_models.generate_model_selection_report(
            [terse, fuller, label_only],
            out,
            prompt="Describe this image briefly.",
            report_context=context,
        )

        content = out.read_text(encoding="utf-8")
        assert "`org/label-caption`" in content
        assert "`org/terse-caption`" in content
        assert "`org/full-caption`" in content
        shortlist = _extract_markdown_subsection(
            content,
            "## Brief Caption Candidates",
            end_headings=("## Structured Metadata Candidates",),
        )
        assert shortlist.index("`org/full-caption`") < shortlist.index("`org/terse-caption`")
        assert "`org/label-caption`" not in shortlist
        avoid = _extract_markdown_subsection(
            content,
            "### Current failures / avoid",
            end_headings=("## Brief Caption Candidates",),
        )
        assert "`org/label-caption`" not in avoid
        assert "`caveat`" in content

    def test_model_selection_report_uses_metadata_when_available(
        self,
        tmp_path: Path,
    ) -> None:
        """Model-selection reports should surface metadata agreement when grounded."""
        result = PerformanceResult(
            model_name="org/metadata-model",
            success=True,
            generation=_MockGeneration(
                text=(
                    "Title: Two tabby cats resting\n"
                    "Description: Two tabby cats rest on a bright pink couch with two remotes.\n"
                    "Keywords: cats, tabby, pink couch, remote controls"
                ),
                generation_tps=55.0,
                prompt_tokens=80,
                generation_tokens=34,
                peak_memory=4.0,
            ),
            total_time=1.0,
            generation_time=0.5,
            model_load_time=0.5,
        )
        out = tmp_path / "model_selection.md"
        metadata: dict[str, str | None] = {
            "title": "Two tabby cats resting",
            "description": "Two tabby cats rest on a bright pink couch with two remotes.",
            "keywords": "cats, tabby, pink couch, remote controls",
        }
        enriched = check_models._populate_result_quality_analysis(
            result,
            prompt="Create title, description, and keywords.",
            metadata=metadata,
            requested_max_tokens=200,
        )
        context = check_models._build_report_render_context(
            results=[enriched],
            prompt="Create title, description, and keywords.",
            metadata=metadata,
            metadata_exposed_to_prompt=True,
            eval_mode="quality",
        )

        check_models.generate_model_selection_report(
            [enriched],
            out,
            prompt="Create title, description, and keywords.",
            metadata=metadata,
            report_context=context,
        )

        content = out.read_text(encoding="utf-8")
        assert "Semantic rankings: grounded (metadata-assisted visual verification)" in content
        assert "Metadata exposed to prompt: yes" in content
        assert "Structured Metadata Candidates" in content
        assert "Top 10 ranked candidates for structured title/description/keywords" in content
        assert "Metadata agreement" in content
        assert "`org/metadata-model`" in content

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

    def test_build_result_output_cues_preserves_priority_order_and_limit(self) -> None:
        """Cue helper should keep the stable cue order before compact preview truncation."""
        result = _make_harness_success("org/cues", harness_type="stop_token")
        assert result.quality_analysis is not None

        analysis = replace(
            result.quality_analysis,
            is_repetitive=True,
            has_context_echo=True,
            instruction_echo=True,
            metadata_borrowing=True,
            has_reasoning_leak=True,
            has_degeneration=True,
            is_context_ignored=True,
            missing_sections=["keywords"],
            formatting_issues=["Formatting marker leak"],
            is_generic=True,
            verdict="cutoff",
        )
        result = replace(
            result,
            quality_analysis=analysis,
            quality_issues=(
                "⚠️harness(stop_token), repetitive(loop), context-echo(0.94), "
                "instruction_echo, metadata_borrowing, cutoff, reasoning_leak, "
                "degeneration, context_ignored, missing_sections(keywords), "
                "formatting(marker), generic"
            ),
        )

        expected_order = [
            "harness:stop-token",
            "repetitive",
            "context-echo",
            "instruction-echo",
            "metadata-borrowing",
            "cutoff",
            "reasoning-leak",
            "degeneration",
            "context-ignored",
            "missing-sections",
            "formatting",
            "generic",
        ]

        assert (
            check_models._build_result_output_cues(result)
            == expected_order[: check_models.OUTPUT_PREVIEW_CUE_LIMIT]
        )

    def test_review_surfaces_use_canonical_assisted_enrichment_evidence(self) -> None:
        """Review surfaces should reuse canonical assisted enrichment evidence."""
        analysis = replace(
            check_models.analyze_generation_text("A concise river caption.", 6),
            metadata_borrowing=True,
            evidence=["unverified-context-copy", "low-draft-improvement"],
        )
        review = check_models._build_jsonl_review_record(
            replace(_make_success("org/enrichment"), quality_analysis=analysis)
        )

        assert review is not None
        focus_text = check_models._review_focus_text(review, analysis)
        assert "unverified-context-copy" in focus_text
        assert "low-draft-improvement" in focus_text
        assert "nonvisual metadata reused" not in focus_text
        assert "metadata borrowing" not in focus_text


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
        results = [_make_success("org/good"), _make_failure("org/bad")]
        context = _build_report_render_context(results=results, prompt="Describe this image fully.")
        generate_markdown_gallery_report(
            results=results,
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
            report_context=context,
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
        assert "## Current-run Chooser" in content
        assert "## Avoid for This Run" in content
        assert "## Lowest-memory Usable Models" in content
        assert "## Fastest Valid Generation" in content
        assert "> [!NOTE]" not in content
        assert "Describe this image fully." in content
        assert "```text\nDescribe this image fully." not in content
        assert "<summary>Complete evidence: org/good</summary>" in content
        assert "```text" in content
        assert '<a id="model-org-good"></a>' in content
        assert "_Usability:_" in content
        assert "_Observations:_" in content
        assert "_Verdict:_" not in content
        assert "_Maintainer:_" not in content
        assert "_Next action:_" not in content
        assert "### org/good" in content
        assert "### org/bad" in content

    def test_gallery_uses_cached_usability_not_recommendation_icons(self, tmp_path: Path) -> None:
        """Completed output should expose cached usability without recommendation policy."""
        text = "<think>Inspect.</think> A useful final caption."
        result = _make_success("org/thinking")
        analysis = replace(
            check_models.analyze_generation_text(
                text,
                generated_tokens=12,
                model_name="org/thinking",
                prompt="Describe this image.",
            ),
            has_reasoning_leak=False,
            has_thinking_trace=True,
            user_bucket="caveat",
        )
        result = replace(
            result,
            generation=_MockGeneration(text=text, generation_tokens=12),
            quality_analysis=analysis,
        )
        out = tmp_path / "model_gallery.md"
        context = _build_report_render_context(results=[result], prompt="Describe this image.")

        generate_markdown_gallery_report(
            results=[result],
            filename=out,
            prompt="Describe this image.",
            report_context=context,
        )

        content = out.read_text(encoding="utf-8")
        chooser_row = next(line for line in content.splitlines() if "org/thinking" in line)
        assert "usable_with_caveats" in chooser_row
        assert "### org/thinking" in content
        assert "### ⚠️ org/thinking" not in content

    def test_gallery_is_evidence_only_without_scoreboard_duplication(
        self,
        tmp_path: Path,
    ) -> None:
        """Gallery should keep output evidence without duplicating selection scoreboards."""
        result = PerformanceResult(
            model_name="org/evidence-model",
            success=True,
            generation=_MockGeneration(
                text="Two cats resting on a pink couch.",
                generation_tps=50.0,
                prompt_tokens=12,
                generation_tokens=8,
                peak_memory=2.0,
            ),
            total_time=1.0,
            generation_time=0.5,
            model_load_time=0.5,
        )
        out = tmp_path / "model_gallery.md"
        context = check_models._build_report_render_context(
            results=[result],
            prompt="Describe this image briefly.",
            eval_mode="triage",
        )

        generate_markdown_gallery_report(
            [result],
            out,
            prompt="Describe this image briefly.",
            metadata={"description": ""},
            report_context=context,
            versions={},
        )

        content = out.read_text(encoding="utf-8")
        assert "# Model Output Gallery" in content
        assert "Complete generated or crash evidence for every attempted model" in content
        assert "Review Shortlist" not in content
        assert "Failures by Package" not in content
        assert "Best keywording" not in content

    def test_gallery_suppresses_cataloging_score_rows_in_triage(
        self,
        tmp_path: Path,
    ) -> None:
        """Triage gallery output should not leak cataloging or keyword score rows."""
        result = PerformanceResult(
            model_name="org/brief-caption",
            success=True,
            generation=_MockGeneration(
                text=(
                    "Title: Two cats on a couch\n"
                    "Description: Two cats rest on a bright pink couch beside remote controls.\n"
                    "Keywords: cats, cats, cats, cats"
                ),
                prompt_tokens=12,
                generation_tokens=28,
            ),
            total_time=1.0,
            generation_time=0.5,
            model_load_time=0.5,
        )
        out = tmp_path / "model_gallery.md"
        context = check_models._build_report_render_context(
            results=[result],
            prompt="Describe this image briefly.",
            eval_mode="triage",
        )

        generate_markdown_gallery_report(
            [result],
            out,
            prompt="Describe this image briefly.",
            metadata={"description": ""},
            report_context=context,
            versions={},
        )

        content = out.read_text(encoding="utf-8")
        assert "_Score:_" not in content
        assert "Keywords are not specific" not in content
        assert "_Review focus:_" not in content

    def test_gallery_includes_consolidated_summary_and_version_stamps(
        self,
        tmp_path: Path,
    ) -> None:
        """Gallery should provide a pasteable run summary with package version stamps."""
        out = tmp_path / "model_gallery.md"
        results = [
            _make_quality_success("org/good", with_quality_issue=False),
            _make_harness_success(
                "org/risky",
                text="answer with | pipe and <think>leaked marker</think>",
                harness_type="stop_token",
                harness_detail="token_leak:<|end|>",
            ),
            _make_failure("org/bad", error_package="mlx-vlm"),
        ]
        context = _build_report_render_context(
            results=results,
            prompt="Describe this image briefly.",
            system_info={
                "GPU Architecture": "applegpu_g17s",
                "Recommended Working Set": "96 GB",
                "Fused Attention": "available",
            },
        )
        generate_markdown_gallery_report(
            results=results,
            filename=out,
            prompt="Describe this image briefly.",
            versions=_stub_versions(),
            report_context=context,
        )

        content = out.read_text(encoding="utf-8")
        assert "## Run Stamps" in content
        assert "- `mlx-vlm`: `0.1`" in content
        assert "- `mlx`: `0.1`" in content
        assert "- _GPU Architecture:_ applegpu_g17s" in content
        assert "- _Recommended Working Set:_ 96 GB" in content
        assert "- _Fused Attention:_ available" in content
        assert "## Current-run Chooser" in content
        assert "## Model Quality Summary" not in content
        assert "## All Model Output and Cost Summary" not in content
        assert "<!-- markdownlint-disable MD034 MD049 -->" in content
        assert "<!-- markdownlint-enable MD034 MD049 -->" in content
        assert "<!-- markdownlint-disable MD013 MD034 -->" not in content
        assert "<!-- markdownlint-enable MD013" not in content

        chooser = _extract_markdown_subsection(
            content,
            "## Current-run Chooser",
            end_headings=("## Avoid for This Run",),
        )
        assert "Output preview" in chooser
        assert "[`org/good`](#model-org-good)" in chooser
        assert "quality output" in chooser
        assert "[`org/risky`](#model-org-risky)" in chooser
        assert "unexpected special token" in chooser
        assert r"answer with \| pipe" in chooser
        assert "&lt;think&gt;leaked marker&lt;/think&gt;" in chooser
        assert "[`org/bad`](#model-org-bad)" in chooser
        assert "not_evaluated" in chooser
        assert "boom" in chooser

    def test_gallery_keeps_complete_output_once_in_expandable_code_block(
        self,
        tmp_path: Path,
    ) -> None:
        """The gallery should keep full evidence once without making the summary unwieldy."""
        complete_text = (
            "**BEGIN:** *model emphasis* " + ("distinct middle evidence " * 30) + "END-SENTINEL"
        )
        result = PerformanceResult(
            model_name="org/complete-output",
            success=True,
            generation=_MockGeneration(
                text=complete_text,
                prompt_tokens=18,
                generation_tokens=200,
                generation_tps=42.0,
                peak_memory=2.5,
            ),
            total_time=1.25,
            generation_time=0.75,
            model_load_time=0.50,
        )
        out = tmp_path / "model_gallery.md"
        context = _build_report_render_context(
            results=[result],
            prompt="Describe this image briefly.",
        )

        generate_markdown_gallery_report(
            results=[result],
            filename=out,
            prompt="Describe this image briefly.",
            report_context=context,
        )

        content = out.read_text(encoding="utf-8")
        chooser = _extract_markdown_subsection(
            content,
            "## Current-run Chooser",
            end_headings=("## Avoid for This Run",),
        )
        assert "## Model Quality Summary" not in content
        assert "## All Model Output and Cost Summary" not in content
        assert "BEGIN" in chooser
        assert "END-SENTINEL" not in chooser
        assert "<!-- markdownlint-disable MD034 MD049 -->" in chooser
        assert "Gen tok" in chooser
        assert "Peak GB" in chooser
        assert "Observations" in chooser
        assert "<summary>Complete evidence: org/complete-output</summary>" in content
        assert "```text" in content
        assert content.count("END-SENTINEL") == 1

    def test_gallery_includes_all_model_output_and_cost_summary(
        self,
        tmp_path: Path,
    ) -> None:
        """Gallery should summarize every model's output beside runtime and memory cost."""
        success = PerformanceResult(
            model_name="org/full-caption",
            success=True,
            generation=_MockGeneration(
                text=(
                    "Title: Two cats on a sofa\n"
                    "Description: Two cats sit together on a pink sofa beside remote controls.\n"
                    "Keywords: cats, sofa, remote controls, indoor, pet portrait"
                ),
                prompt_tokens=18,
                generation_tokens=24,
                generation_tps=42.0,
                peak_memory=2.5,
            ),
            total_time=1.25,
            generation_time=0.75,
            model_load_time=0.50,
        )
        failure = replace(
            _make_failure("org/crashed", error_package="transformers"),
            total_time=0.33,
        )
        harness = _make_harness_success(
            "org/risky-output",
            text="cats",
            generation_tokens=3,
            harness_type="prompt_template",
        )
        out = tmp_path / "model_gallery.md"
        context = _build_report_render_context(
            results=[success, failure, harness],
            prompt="Describe this image briefly.",
        )

        generate_markdown_gallery_report(
            results=[success, failure, harness],
            filename=out,
            prompt="Describe this image briefly.",
            report_context=context,
        )

        content = out.read_text(encoding="utf-8")
        chooser = _extract_markdown_subsection(
            content,
            "## Current-run Chooser",
            end_headings=("## Avoid for This Run",),
        )
        assert "Output preview" in chooser
        assert "Peak GB" in chooser
        assert "Observations" in chooser
        assert "[`org/full-caption`](#model-org-full-caption)" in chooser
        assert "Two cats sit together on a pink sofa" in chooser
        assert "24" in chooser
        assert "42.0" in chooser
        assert "2.5" in chooser
        risky_row = next(line for line in chooser.splitlines() if "org/risky-output" in line)
        assert "| cats " in risky_row
        assert "insufficient sample" in risky_row
        assert "[`org/crashed`](#model-org-crashed)" in chooser
        assert "boom" in chooser
        crashed_evidence = _extract_markdown_subsection(
            content,
            "### org/crashed",
            end_headings=("### org/full-caption", "### org/risky-output"),
        )
        assert "_Total time:_ 0.33s" in crashed_evidence

    def test_gallery_uses_skim_first_chooser_order_and_cached_assessments(
        self,
        tmp_path: Path,
    ) -> None:
        """Gallery order should move from chooser policy to complete evidence."""
        results = [
            _make_success("org/usable"),
            _make_success("org/unusable"),
            _make_failure("org/not-evaluated"),
        ]
        context = _build_report_render_context(results=results, prompt="Describe the image.")
        context = replace(
            context,
            assessments=(
                (
                    "org/usable",
                    check_models.ResultAssessment("completed", "usable", "none", ()),
                ),
                (
                    "org/unusable",
                    check_models.ResultAssessment(
                        "completed",
                        "unusable",
                        "observation_needs_reproduction",
                        ("repeated_output",),
                    ),
                ),
                (
                    "org/not-evaluated",
                    check_models.ResultAssessment(
                        "crashed",
                        "not_evaluated",
                        "actionable_failure",
                        (),
                    ),
                ),
            ),
        )
        out = tmp_path / "model_gallery.md"

        with (
            patch.object(check_models, "_review_for_result", side_effect=AssertionError),
            patch.object(check_models, "_model_selection_score", side_effect=AssertionError),
            patch.object(
                check_models,
                "_recommendation_status_for_result",
                side_effect=AssertionError,
            ),
        ):
            generate_markdown_gallery_report(
                results=results,
                filename=out,
                prompt="Describe the image.",
                report_context=context,
            )

        content = out.read_text(encoding="utf-8")
        headings = [
            "## Current-run Chooser",
            "## Avoid for This Run",
            "## Lowest-memory Usable Models",
            "## Fastest Valid Generation",
            "## Complete Per-model Evidence",
        ]
        assert [content.index(heading) for heading in headings] == sorted(
            content.index(heading) for heading in headings
        )
        assert "_Verdict:_" not in content
        assert "_Maintainer:_" not in content
        assert "_Next action:_" not in content
        assert "_Score:_" not in content

    def test_gallery_complete_output_uses_safe_fence_without_shortening(
        self,
        tmp_path: Path,
    ) -> None:
        """Complete output should survive prior limits and nested Markdown fences."""
        complete_text = (
            "BEGIN-COMPLETE\n```python\nprint('nested')\n```\n"
            + ("evidence-line-0123456789\n" * 600)
            + "END-COMPLETE"
        )
        result = replace(
            _make_success("org/complete"),
            generation=_MockGeneration(
                text=complete_text,
                prompt_tokens=32,
                generation_tokens=500,
                generation_tps=25.0,
            ),
        )
        context = _build_report_render_context(results=[result], prompt="Describe the image.")
        out = tmp_path / "model_gallery.md"

        generate_markdown_gallery_report(
            results=[result],
            filename=out,
            prompt="Describe the image.",
            report_context=context,
        )

        content = out.read_text(encoding="utf-8")
        evidence = _extract_markdown_subsection(
            content,
            "### org/complete",
            end_headings=("<!-- markdownlint-enable",),
        )
        assert "<details>" in evidence
        assert "````text\n" in evidence
        assert complete_text in evidence
        assert content.count(complete_text) == 1

    def test_short_generation_is_not_valid_throughput_but_keeps_raw_metrics(
        self,
        tmp_path: Path,
    ) -> None:
        """A short sample should affect throughput validity, not model usability."""
        short = replace(
            _make_success("org/short"),
            generation=_MockGeneration(
                text="A usable short response.",
                prompt_tokens=20,
                generation_tokens=8,
                generation_tps=999.0,
                peak_memory=1.0,
            ),
            generation_time=0.25,
        )
        valid = replace(
            _make_success("org/valid"),
            generation=_MockGeneration(
                text="A sufficiently measured response.",
                prompt_tokens=20,
                generation_tokens=20,
                generation_tps=40.0,
                peak_memory=2.0,
            ),
        )
        context = _build_report_render_context(results=[short, valid], prompt="Describe the image.")
        context = replace(
            context,
            assessments=tuple(
                (
                    result.model_name,
                    check_models.ResultAssessment("completed", "usable", "none", ()),
                )
                for result in (short, valid)
            ),
        )
        out = tmp_path / "model_gallery.md"

        generate_markdown_gallery_report(
            results=[short, valid],
            filename=out,
            prompt="Describe the image.",
            report_context=context,
        )

        content = out.read_text(encoding="utf-8")
        chooser = _extract_markdown_subsection(
            content,
            "## Current-run Chooser",
            end_headings=("## Avoid for This Run",),
        )
        short_row = next(line for line in chooser.splitlines() if "org/short" in line)
        assert "usable" in short_row
        assert "insufficient sample" in short_row
        assert "999" not in short_row
        assert "Fastest valid generation: `org/valid` at 40.0 tok/s" in content
        assert "Average valid generation throughput: 40.0 tok/s" in content
        evidence = _extract_markdown_subsection(
            content,
            "### org/short",
            end_headings=("### org/valid", "<!-- markdownlint-enable"),
        )
        assert "_Generation time:_ 0.25s" in evidence
        assert "_Generation throughput (raw):_ 999 tok/s" in evidence
        assert "_Generation tokens:_ 8" in evidence

    def test_gallery_resource_policies_are_deterministic(self, tmp_path: Path) -> None:
        """Avoid, memory, and speed policies should have explicit stable ordering."""

        def result(
            name: str,
            *,
            memory: float | None,
            tokens: int,
            throughput: float | None,
        ) -> PerformanceResult:
            return PerformanceResult(
                model_name=name,
                success=True,
                generation=_MockGeneration(
                    text=f"output for {name}",
                    prompt_tokens=20,
                    generation_tokens=tokens,
                    generation_tps=throughput,
                    peak_memory=memory,
                ),
            )

        usable_results = [
            result("org/zeta", memory=None, tokens=8, throughput=900.0),
            result("org/beta", memory=2.0, tokens=20, throughput=30.0),
            result("org/alpha", memory=2.0, tokens=20, throughput=30.0),
            result("org/gamma", memory=1.0, tokens=20, throughput=10.0),
        ]
        avoided_results = [
            _make_success("org/z-unusable"),
            _make_success("org/a-unusable"),
            _make_failure("org/a-not-evaluated"),
        ]
        results = [*usable_results, *avoided_results]
        context = _build_report_render_context(results=results, prompt="Describe the image.")
        assessments = {
            result.model_name: check_models.ResultAssessment(
                "completed",
                "usable_with_caveats" if result.model_name == "org/beta" else "usable",
                "none",
                (),
            )
            for result in usable_results
        }
        assessments.update(
            {
                "org/z-unusable": check_models.ResultAssessment(
                    "completed", "unusable", "observation_needs_reproduction", ("empty_output",)
                ),
                "org/a-unusable": check_models.ResultAssessment(
                    "completed", "unusable", "observation_needs_reproduction", ("empty_output",)
                ),
                "org/a-not-evaluated": check_models.ResultAssessment(
                    "crashed", "not_evaluated", "actionable_failure", ()
                ),
            }
        )
        context = replace(context, assessments=tuple(assessments.items()))
        out = tmp_path / "model_gallery.md"

        generate_markdown_gallery_report(
            results=results,
            filename=out,
            prompt="Describe the image.",
            report_context=context,
        )

        content = out.read_text(encoding="utf-8")
        avoid = _extract_markdown_subsection(
            content,
            "## Avoid for This Run",
            end_headings=("## Lowest-memory Usable Models",),
        )
        memory = _extract_markdown_subsection(
            content,
            "## Lowest-memory Usable Models",
            end_headings=("## Fastest Valid Generation",),
        )
        speed = _extract_markdown_subsection(
            content,
            "## Fastest Valid Generation",
            end_headings=("## Complete Per-model Evidence",),
        )
        assert avoid.index("org/a-unusable") < avoid.index("org/z-unusable")
        assert avoid.index("org/z-unusable") < avoid.index("org/a-not-evaluated")
        assert memory.index("org/gamma") < memory.index("org/alpha")
        assert memory.index("org/alpha") < memory.index("org/beta")
        assert memory.index("org/beta") < memory.index("org/zeta")
        assert speed.index("org/alpha") < speed.index("org/beta")
        assert speed.index("org/beta") < speed.index("org/gamma")
        assert speed.index("org/gamma") < speed.index("org/zeta")

    def test_gallery_crash_evidence_keeps_traceback_before_captured_output(
        self,
        tmp_path: Path,
    ) -> None:
        """Crash evidence should retain factual context and complete evidence priority."""
        result = replace(
            _make_failure("org/crashed", error_package="mlx-vlm"),
            failure_phase="decode",
            error_code="generation-failed",
            error_traceback="Traceback (most recent call last):\nRuntimeError: complete trace",
            captured_output_on_fail="complete captured stderr",
        )
        context = _build_report_render_context(results=[result], prompt="Describe the image.")
        out = tmp_path / "model_gallery.md"

        generate_markdown_gallery_report(
            results=[result],
            filename=out,
            prompt="Describe the image.",
            report_context=context,
        )

        content = out.read_text(encoding="utf-8")
        evidence = _extract_markdown_subsection(
            content,
            "### org/crashed",
            end_headings=("<!-- markdownlint-enable",),
        )
        assert "_Failure phase:_ decode" in evidence
        assert "_Error code:_ generation-failed" in evidence
        assert "_Error package:_ mlx-vlm" in evidence
        assert evidence.index("RuntimeError: complete trace") < evidence.index(
            "complete captured stderr"
        )

    def test_gallery_uses_cached_indeterminate_execution(self, tmp_path: Path) -> None:
        """Per-model evidence should not turn indeterminate attempts into crashes."""
        result = _make_failure("org/indeterminate", error_package="huggingface-hub")
        context = _build_report_render_context(results=[result], prompt="Describe the image.")
        context = replace(
            context,
            assessments=(
                (
                    result.model_name,
                    check_models.ResultAssessment(
                        "indeterminate",
                        "not_evaluated",
                        "observation_needs_reproduction",
                        (),
                    ),
                ),
            ),
        )
        out = tmp_path / "model_gallery.md"

        generate_markdown_gallery_report(
            results=[result],
            filename=out,
            prompt="Describe the image.",
            report_context=context,
        )

        content = out.read_text(encoding="utf-8")
        assert "_Execution:_ indeterminate" in content
        assert "_Execution:_ crashed" not in content

    def test_gallery_marks_missing_nested_diagnostic_fields_not_captured(
        self,
        tmp_path: Path,
    ) -> None:
        """Present diagnostic objects should not make absent nested facts disappear."""
        result = replace(
            _make_success("org/missing-nested"),
            runtime_diagnostics=RuntimeDiagnostics(stop_reason=None),
            prompt_diagnostics=check_models.PromptDiagnostics(
                processor_class=None,
                tokenizer_class=None,
            ),
        )
        context = _build_report_render_context(results=[result], prompt="Describe the image.")
        out = tmp_path / "model_gallery.md"

        generate_markdown_gallery_report(
            results=[result],
            filename=out,
            prompt="Describe the image.",
            report_context=context,
        )

        content = out.read_text(encoding="utf-8")
        evidence = _extract_markdown_subsection(
            content,
            "### org/missing-nested",
            end_headings=("<!-- markdownlint-enable",),
        )
        assert "_Stop reason:_ not captured" in evidence
        assert "_Processor:_ not captured" in evidence
        assert "_Tokenizer:_ not captured" in evidence

    def test_review_report_keeps_user_buckets_without_legacy_issue_queue(
        self,
        tmp_path: Path,
    ) -> None:
        """Review digest should not advertise inferred owners or retired issue indexes."""
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
        assert content.startswith(
            "<!-- markdownlint-disable MD012 MD013 -->\n\n# Automated Review Digest"
        )
        assert "# Automated Review Digest" in content
        assert "## Maintainer Escalations" not in content
        assert "issues/index.md" not in content

        # Relative link rendering must not revive the retired issue queue.
        with patch.object(check_models._LinkStyleState, "value", "relative"):
            generate_review_report(
                results=results,
                filename=out,
                prompt="describe",
                report_context=report_context,
                log_filename=log_file,
                gallery_filename=gallery,
            )
            content_relative = out.read_text(encoding="utf-8")
            assert "issues/index.md" not in content_relative
        assert "## 🧭 Review Shortlist" in content
        assert "## User Buckets" in content
        assert "## Model Verdicts" in content
        assert "## Maintainer Queue" not in content
        assert "`clean-triage-pass`" in content
        assert "`avoid`" in content
        assert content.index("## User Buckets") < content.index("## Model Verdicts")
        assert "Model" in content
        assert "Hint Handling" in content
        assert "Key Evidence" in content
        assert "Canonical run log" in content
        assert "Treat as a model-quality limitation" not in content

    def test_review_report_marks_hint_handling_not_evaluated_without_metadata(
        self,
        tmp_path: Path,
    ) -> None:
        """Plain triage prompts should not claim trusted metadata hints were preserved."""
        prompt = "Describe this image briefly."
        text = "Two cats are sleeping on a pink blanket on a couch."
        analysis = check_models.analyze_generation_text(
            text,
            generated_tokens=13,
            prompt_tokens=1196,
            prompt=prompt,
            requested_max_tokens=200,
        )
        result = PerformanceResult(
            model_name="org/plain-caption",
            success=True,
            generation=_MockGeneration(
                text=text,
                prompt_tokens=1196,
                generation_tokens=13,
            ),
            quality_analysis=analysis,
            total_time=1.0,
            generation_time=0.5,
            model_load_time=0.5,
        )
        out = tmp_path / "review.md"

        generate_review_report(
            results=[result],
            filename=out,
            prompt=prompt,
            report_context=_build_report_render_context(
                results=[result],
                prompt=prompt,
                eval_mode="triage",
            ),
        )

        content = out.read_text(encoding="utf-8")
        assert "not evaluated" in content
        assert "preserves trusted hints" not in content

    def test_review_report_keeps_hint_handling_when_metadata_is_present(
        self,
        tmp_path: Path,
    ) -> None:
        """Metadata-grounded prompts should still evaluate trusted visual hints."""
        prompt = (
            "Context:\n"
            "Title: Two tabby cats resting\n"
            "Description: Two tabby cats rest on a bright pink couch with two remotes.\n"
            "Keywords: cats, tabby, pink couch, remote controls\n\n"
            "Describe this image briefly."
        )
        text = "Two tabby cats rest on a bright pink couch with two remote controls."
        analysis = check_models.analyze_generation_text(
            text,
            generated_tokens=16,
            prompt_tokens=260,
            prompt=prompt,
            requested_max_tokens=200,
        )
        result = PerformanceResult(
            model_name="org/metadata-caption",
            success=True,
            generation=_MockGeneration(
                text=text,
                prompt_tokens=260,
                generation_tokens=16,
            ),
            quality_analysis=analysis,
            total_time=1.0,
            generation_time=0.5,
            model_load_time=0.5,
        )
        out = tmp_path / "review.md"

        generate_review_report(
            results=[result],
            filename=out,
            prompt=prompt,
            report_context=_build_report_render_context(
                results=[result],
                prompt=prompt,
                metadata={
                    "title": "Two tabby cats resting",
                    "description": "Two tabby cats rest on a bright pink couch with two remotes.",
                    "keywords": "cats, tabby, pink couch, remote controls",
                },
                eval_mode="quality",
            ),
        )

        content = out.read_text(encoding="utf-8")
        assert "preserves trusted hints" in content
        assert "not evaluated" not in content

    def test_clean_image_heavy_review_focus_omits_nontext_burden(self) -> None:
        """Non-text prompt burden should be context, not key evidence, for clean captions."""
        prompt = "Describe this image briefly."
        text = "Two cats are sleeping on a pink blanket on a couch."
        analysis = check_models.analyze_generation_text(
            text,
            generated_tokens=13,
            prompt_tokens=1196,
            prompt=prompt,
            requested_max_tokens=200,
        )
        result = PerformanceResult(
            model_name="org/plain-caption",
            success=True,
            generation=_MockGeneration(
                text=text,
                prompt_tokens=1196,
                generation_tokens=13,
            ),
            quality_analysis=analysis,
        )
        review = check_models._build_jsonl_review_record(result)

        assert review is not None
        assert "nontext prompt burden" not in check_models._review_focus_text(review, analysis)

    def test_context_budget_review_focus_keeps_nontext_burden(self) -> None:
        """Real context-collapse cases should expose canonical image-token pressure."""
        analysis = check_models.analyze_generation_text(
            "Cat.",
            generated_tokens=3,
            prompt_tokens=4103,
            prompt="Describe this image briefly.",
            requested_max_tokens=200,
        )
        result = PerformanceResult(
            model_name="org/context-collapse",
            success=True,
            generation=_MockGeneration(
                text="Cat.",
                prompt_tokens=4103,
                generation_tokens=3,
            ),
            quality_analysis=analysis,
            prompt_diagnostics=check_models.PromptDiagnostics(image_placeholder_count=1),
        )
        review = check_models._build_jsonl_review_record(result)

        assert review is not None
        focus = check_models._review_focus_text(review, analysis)
        assert analysis.verdict == "context_budget"
        assert "visual input burden" in focus
        assert "nontext prompt burden" not in focus

    def test_unavailable_prompt_components_do_not_claim_normal_burden(self) -> None:
        """Unavailable component estimates should produce uncertainty-aware guidance."""
        analysis = replace(
            check_models.analyze_generation_text(
                "Cat.",
                generated_tokens=3,
                prompt_tokens=4103,
                prompt="Describe this image briefly.",
            ),
            prompt_tokens_text_est=None,
            prompt_tokens_nontext_est=None,
            verdict="context_budget",
        )
        result = PerformanceResult(
            model_name="org/unavailable-components",
            success=True,
            generation=_MockGeneration(
                text="Cat.",
                prompt_tokens=4103,
                generation_tokens=3,
            ),
            quality_analysis=analysis,
        )
        review = check_models._build_jsonl_review_record(result)

        assert review is not None
        guidance = check_models._review_next_action_for_result(result, review)
        assert review["prompt_burden_kind"] == "unavailable"
        assert "normal burden issue" not in guidance
        assert "controlled" in guidance

    def test_gallery_keeps_chooser_and_per_model_factual_status(
        self,
        tmp_path: Path,
    ) -> None:
        """Gallery should keep cached status without legacy review projections."""
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
        assert "## Current-run Chooser" in content
        assert "Action Snapshot" not in content
        assert "## 🧭 Review Shortlist" not in content
        assert "## 🚨 Failures by Package (Actionable)" not in content
        assert "_Review focus:_" not in content
        assert "_Score:_" not in content
        assert "_Usability:_" in content
        assert "_Execution:_" in content
        assert "_Next action:_" not in content


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

    def test_tsv_starts_with_standard_header(self, tmp_path: Path) -> None:
        """TSV output should import without a non-standard comment preamble."""
        out = tmp_path / "meta.tsv"
        generate_tsv_report(
            results=[_make_success()],
            filename=out,
        )
        first_line = out.read_text(encoding="utf-8").splitlines()[0]
        assert not first_line.startswith("#")
        assert "Model" in first_line

    def test_tsv_has_error_columns(self, tmp_path: Path) -> None:
        """TSV should include error_type and error_package columns."""
        out = tmp_path / "cols.tsv"
        generate_tsv_report(
            results=[_make_failure(error_type="RuntimeError", error_package="transformers")],
            filename=out,
        )
        content = out.read_text(encoding="utf-8")
        header_line = content.splitlines()[0]
        assert "error_type" in header_line
        assert "error_package" in header_line
        data_line = content.splitlines()[1]
        assert "RuntimeError" in data_line
        assert "transformers" in data_line

    def test_tsv_omits_empty_error_columns_for_success(self, tmp_path: Path) -> None:
        """Successful-only runs should omit wholly empty error columns."""
        out = tmp_path / "ok.tsv"
        generate_tsv_report(
            results=[_make_success()],
            filename=out,
        )
        content = out.read_text(encoding="utf-8")
        header_line = content.splitlines()[1]
        assert "error_type" not in header_line
        assert "error_package" not in header_line


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
    failure_phase: str | None = None,
    traceback_str: str | None = None,
    captured_output: str | None = None,
    generated_text: str | None = None,
    upstream_boundary: check_models.UpstreamBoundary = "generation_started",
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
        failure_phase=failure_phase,
        error_message=error_msg,
        error_type=error_type,
        error_package=error_package,
        captured_output_on_fail=captured_output,
        error_traceback=traceback_str,
        upstream_boundary=upstream_boundary,
    )


class TestSharedReportSections:
    """Tests for shared Markdown/HTML report section primitives."""

    def test_report_details_separates_table_from_closing_tag(self) -> None:
        """Tables inside details blocks should retain the required trailing blank line."""
        details = check_models.ReportDetails(
            summary="Complete matrix",
            blocks=(
                check_models.ReportTable(
                    headers=("Model", "Status"),
                    rows=(("org/model", "recommended"),),
                ),
            ),
        )

        markdown = "\n".join(check_models.render_report_markdown((details,)))

        assert "| org/model" in markdown
        assert "\n\n</details>" in markdown

    def test_markdown_block_helper_matches_public_renderer(self) -> None:
        """Private block renderer should preserve public Markdown output."""
        block = check_models.ReportParagraph("Observed <tag> & value")
        block_lines = check_models._render_report_markdown_block(block)
        while block_lines and block_lines[-1] == "":
            block_lines.pop()

        assert block_lines == check_models.render_report_markdown((block,))

    def test_report_section_renders_markdown_and_html_from_same_model(self) -> None:
        """A shared section model should render escaped Markdown and HTML variants."""
        section = check_models.ReportSection(
            title="Queue <Summary>",
            level=2,
            blocks=(
                check_models.ReportParagraph("Observed <tag> & value"),
                check_models.ReportKeyValues(
                    rows=(
                        ("Owner", "mlx-vlm <runtime>"),
                        ("Evidence", "stage=model_load | type=ValueError"),
                    )
                ),
                check_models.ReportBulletList(("first <signal>", "second signal")),
                check_models.ReportTable(
                    headers=("Model", "Problem"),
                    rows=(("org/model", "shape <mismatch>"),),
                ),
                check_models.ReportCodeBlock("print('hello')", language="python"),
                check_models.ReportDetails(
                    summary="Trace <details>",
                    blocks=(check_models.ReportParagraph("inside <frame>"),),
                ),
            ),
        )

        markdown = "\n".join(check_models.render_report_markdown((section,)))
        html_output = "\n".join(check_models.render_report_html((section,)))

        assert "## Queue &lt;Summary&gt;" in markdown
        assert "Observed &lt;tag&gt; &amp; value" in markdown
        assert "| Model" in markdown
        assert "```python" in markdown
        assert "<summary>Trace &lt;details&gt;</summary>" in markdown

        assert "<h2>Queue &lt;Summary&gt;</h2>" in html_output
        assert "Observed &lt;tag&gt; &amp; value" in html_output
        assert "<table>" in html_output
        assert '<pre><code class="language-python">' in html_output
        assert "<summary>Trace &lt;details&gt;</summary>" in html_output


class TestReproCommandNormalization:
    """Tests for spec-driven repro command generation."""

    def test_native_mlx_vlm_cli_omits_non_cli_generate_kwargs(self, tmp_path: Path) -> None:
        """Native CLI repros should not invent upstream flags absent from mlx-vlm CLI."""
        image_path = tmp_path / "probe.png"
        adapter_path = tmp_path / "adapter"
        run_args = Namespace(
            adapter_path=adapter_path,
            resize_shape=(64, 32),
            eos_tokens=["</s>"],
            max_kv_size=4096,
            kv_bits=4,
            kv_quant_scheme="turboquant",
            kv_group_size=32,
            quantized_kv_start=128,
            skip_special_tokens=True,
            force_download=True,
            revision="main",
            trust_remote_code=True,
            quantize_activations=True,
            processor_kwargs={"cropping": False},
            prefill_step_size=512,
            enable_thinking=True,
            thinking_budget=32,
            thinking_start_token=THINKING_START_TOKEN,
            thinking_end_token=THINKING_END_TOKEN,
            max_tokens=123,
            temperature=0.2,
            top_p=0.8,
            min_p=0.1,
            top_k=4,
            repetition_penalty=1.1,
            repetition_context_size=64,
        )

        tokens = check_models._build_native_mlx_vlm_cli_tokens(
            model_name="org/model",
            prompt="Describe this.",
            image_ref=str(image_path),
            run_args=run_args,
        )
        script = check_models._build_native_mlx_vlm_python_script(
            model_name="org/model",
            prompt="Describe this.",
            image_ref=str(image_path),
            run_args=run_args,
        )
        for unsupported_cli_flag in (
            "--top-p",
            "--min-p",
            "--top-k",
            "--repetition-penalty",
            "--repetition-context-size",
        ):
            assert unsupported_cli_flag not in tokens
        assert "--processor-kwargs" in tokens
        assert "--prefill-step-size" in tokens

        assert "'top_p': 0.8" in script
        assert "'min_p': 0.1" in script
        assert "'top_k': 4" in script
        assert "'repetition_penalty': 1.1" in script
        assert "'repetition_context_size': 64" in script
        assert "'cropping': False" in script
        assert "from mlx_vlm.prompt_utils import apply_chat_template" in script
        assert "formatted_prompt = apply_chat_template(" in script
        assert "processor," in script
        assert "model.config," in script
        assert "PROMPT," in script
        assert "num_images=1," in script
        assert (
            "result = generate(model, processor, formatted_prompt, image=IMAGE, **GENERATE_KWARGS)"
            in script
        )
        assert "generate(model, processor, PROMPT" not in script

    def test_native_python_repro_preserves_template_kwargs_and_false_trust(self) -> None:
        """Canonical Python repros should match thinking setup and explicit trust policy."""
        run_args = Namespace(
            trust_remote_code=False,
            enable_thinking=True,
            thinking_budget=19,
            thinking_start_token=THINKING_START_TOKEN,
            thinking_end_token=CUSTOM_THINKING_END_TOKEN,
            max_tokens=64,
            temperature=0.1,
        )

        script = check_models._build_native_mlx_vlm_python_script(
            model_name="org/model",
            prompt="Describe this.",
            image_ref="image.jpg",
            run_args=run_args,
        )

        assert "LOAD_KWARGS = {'trust_remote_code': False}" in script
        assert "TEMPLATE_KWARGS = {" in script
        assert "'enable_thinking': True" in script
        assert "'thinking_budget': 19" in script
        assert f"'thinking_start_token': {THINKING_START_TOKEN!r}" in script
        assert f"'thinking_end_token': {CUSTOM_THINKING_END_TOKEN!r}" in script
        assert "    **TEMPLATE_KWARGS," in script


def test_output_index_routes_maintainers_and_model_users(tmp_path: Path) -> None:
    """Run-level output index should tell each audience where to start."""
    good = _make_success("org/good")
    failure = _make_failure("org/bad", error_package="mlx-vlm")
    report_context = _build_report_render_context(
        results=[good, failure],
        prompt="Describe this image briefly.",
        eval_mode="triage",
    )
    output_dir = tmp_path / "output"
    reports_dir = output_dir / "reports"
    issues_dir = output_dir / "issues"
    paths = check_models.ReportOutputPaths(
        index=output_dir / "index.md",
        html=reports_dir / "results.html",
        markdown=reports_dir / "results.md",
        gallery_markdown=reports_dir / "model_gallery.md",
        review=reports_dir / "review.md",
        model_selection=reports_dir / "model_selection.md",
        model_capabilities=reports_dir / "model_capabilities.md",
        model_capabilities_json=output_dir / "model_capabilities.json",
        tsv=reports_dir / "results.tsv",
        jsonl=output_dir / "results.jsonl",
        run_json=output_dir / "run.json",
        diagnostics=reports_dir / "diagnostics.md",
        log=output_dir / "check_models.log",
        environment=output_dir / "environment.log",
    )
    artifacts = DiagnosticsArtifacts(
        outcome_counts=check_models._run_outcome_counts(report_context.assessments),
        diagnostics_written=True,
        issue_reports={"org/bad": issues_dir / "issue_org_bad.md"},
    )

    check_models.generate_output_index_report(
        paths.index,
        output_paths=paths,
        report_context=report_context,
        diagnostics_artifacts=artifacts,
    )

    content = paths.index.read_text(encoding="utf-8")
    assert "# Check Models Output Index" in content
    assert "- Models attempted: 2" in content
    assert "- Models evaluated: 2" in content
    assert "- Successful: 1" in content
    assert "- Failed: 1" in content
    assert "- Indeterminate: 0" in content
    assert "## For Model Users" in content
    assert "## For Maintainers" in content
    assert "model_selection.md" in content
    assert "model_capabilities.md" in content
    assert "issue_org_bad.md" in content
    assert "issues/index.md" not in content
    assert "latest_by_cluster.json" not in content
    assert "## Primary Artifacts" in content
    assert "## Supporting Artifacts" in content
    primary = _extract_markdown_subsection(
        content,
        "## Primary Artifacts",
        end_headings=("## Supporting Artifacts",),
    )
    supporting = _extract_markdown_subsection(
        content,
        "## Supporting Artifacts",
        end_headings=("## For Model Users",),
    )
    for artifact in (
        "diagnostics.md",
        "results.html",
        "model_selection.md",
        "model_gallery.md",
        "results.jsonl",
    ):
        assert artifact in primary
    for artifact in (
        "results.md",
        "review.md",
        "model_capabilities.md",
        "results.tsv",
        "results.history.jsonl",
        "issue_org_bad.md",
    ):
        assert artifact in supporting


def test_reports_dashboard_lists_each_real_issue_draft(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The console dashboard should link actual drafts, never a deleted index."""
    output_dir = tmp_path / "output"
    reports_dir = output_dir / "reports"
    issues_dir = output_dir / "issues"
    reports_dir.mkdir(parents=True)
    issues_dir.mkdir()
    diagnostics = reports_dir / "diagnostics.md"
    diagnostics.write_text("diagnostics\n", encoding="utf-8")
    issue_one = issues_dir / "issue_org_one.md"
    issue_two = issues_dir / "issue_org_two.md"
    issue_one.write_text("one\n", encoding="utf-8")
    issue_two.write_text("two\n", encoding="utf-8")
    paths = check_models.ReportOutputPaths(
        index=output_dir / "index.md",
        html=reports_dir / "results.html",
        markdown=reports_dir / "results.md",
        gallery_markdown=reports_dir / "model_gallery.md",
        review=reports_dir / "review.md",
        model_selection=reports_dir / "model_selection.md",
        model_capabilities=reports_dir / "model_capabilities.md",
        model_capabilities_json=output_dir / "model_capabilities.json",
        tsv=reports_dir / "results.tsv",
        jsonl=output_dir / "results.jsonl",
        run_json=output_dir / "run.json",
        diagnostics=diagnostics,
        log=output_dir / "check_models.log",
        environment=output_dir / "environment.log",
    )
    artifacts = DiagnosticsArtifacts(
        diagnostics_written=True,
        issue_reports={"org/one": issue_one, "org/two": issue_two},
    )

    check_models._print_reports_dashboard(paths, diagnostics_artifacts=artifacts)

    captured = capsys.readouterr().err
    assert "issue_org_one.md" in captured
    assert "issue_org_two.md" in captured
    assert "issues/index.md" not in captured


def test_generate_markdown_report_uses_provided_report_context(tmp_path: Path) -> None:
    """Markdown generation should reuse a supplied cached report context."""
    out = tmp_path / "results.md"
    results = [_make_success("org/good"), _make_failure("org/bad")]
    report_context = _build_report_render_context(results=results, prompt="test prompt")

    with (
        patch.object(check_models, "_build_report_render_context", side_effect=AssertionError),
        patch.object(check_models, "analyze_model_issues", side_effect=AssertionError),
        patch.object(check_models, "compute_performance_statistics", side_effect=AssertionError),
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


def test_generate_tsv_report_uses_provided_report_context(tmp_path: Path) -> None:
    """TSV generation should reuse a supplied cached report context."""
    out = tmp_path / "results.tsv"
    results = [_make_success("org/good"), _make_failure("org/bad")]
    report_context = _build_report_render_context(results=results, prompt="test prompt")

    with patch.object(check_models, "_build_report_render_context", side_effect=AssertionError):
        generate_tsv_report(results=results, filename=out, report_context=report_context)

    content = out.read_text(encoding="utf-8")
    assert "org/good" in content
    assert "error_type" in content


def test_generate_tsv_report_includes_full_generated_text_for_analysis(tmp_path: Path) -> None:
    """Spreadsheet output should preserve exact generated text separately from previews."""
    out = tmp_path / "results.tsv"
    full_text = (
        "Two cats are sleeping on a pink couch. "
        + "context words " * 40
        + "</think> exact leak marker after a long reasoning preface."
    )
    result = PerformanceResult(
        model_name="org/full-output",
        success=True,
        generation=_MockGeneration(
            text=full_text,
            prompt_tokens=317,
            generation_tokens=196,
        ),
        total_time=1.0,
        generation_time=0.5,
        model_load_time=0.5,
    )

    generate_tsv_report(results=[result], filename=out)

    content = out.read_text(encoding="utf-8")
    assert "Generated Text" in content
    assert "</think> exact leak marker" in content


def test_generate_tsv_report_standalone_uses_prepared_table_path(tmp_path: Path) -> None:
    """Standalone TSV generation should still render results without cached context."""
    out = tmp_path / "standalone.tsv"
    results = [_make_success("org/good"), _make_failure("org/bad")]

    generate_tsv_report(results=results, filename=out)

    content = out.read_text(encoding="utf-8")
    assert "org/good" in content
    assert "org/bad" in content


class TestCleanStaleToplevelReports:
    """Regression coverage for stale top-level report cleanup."""

    def test_removes_stale_files_when_canonical_exists(self, tmp_path: Path) -> None:
        """A stale top-level file is removed when the reports copy exists."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        (tmp_path / "results.md").write_text("old", encoding="utf-8")
        (reports_dir / "results.md").write_text("canonical", encoding="utf-8")
        (tmp_path / "model_selection.md").write_text("old selection", encoding="utf-8")
        (reports_dir / "model_selection.md").write_text(
            "canonical selection",
            encoding="utf-8",
        )

        removed = _clean_stale_toplevel_reports(tmp_path, reports_dir)

        assert removed == 2
        assert not (tmp_path / "results.md").exists()
        assert not (tmp_path / "model_selection.md").exists()

    def test_keeps_file_when_no_canonical(self, tmp_path: Path) -> None:
        """A top-level file is kept when no reports copy exists."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        only_copy = tmp_path / "results.md"
        only_copy.write_text("only copy", encoding="utf-8")

        assert _clean_stale_toplevel_reports(tmp_path, reports_dir) == 0
        assert only_copy.exists()


class TestGithubIssueReportsCleanup:
    """Regression coverage for live stale crash-draft cleanup."""

    def test_stale_issue_files_removed(self, tmp_path: Path) -> None:
        """Old issue_*.md files are removed even when the next run has no crashes."""
        issues_dir = tmp_path / "issues"
        issues_dir.mkdir()
        stale_crash = issues_dir / "issue_001_crash.md"
        stale_harness = issues_dir / "issue_002_harness.md"
        stale_index = issues_dir / "index.md"
        readme = issues_dir / "README.md"
        stale_crash.write_text("stale crash report", encoding="utf-8")
        stale_harness.write_text("stale harness report", encoding="utf-8")
        stale_index.write_text("stale index", encoding="utf-8")
        readme.write_text("keep me", encoding="utf-8")
        context = _build_report_render_context(results=[], prompt="Describe the image.")

        generated = _generate_github_issue_reports(
            report_context=context,
            output_dir=tmp_path,
            library_versions=_stub_versions(),
            system_info={"Python Version": "3.13"},
            prompt="Describe the image.",
        )

        assert generated == {}
        assert not stale_crash.exists()
        assert not stale_harness.exists()
        assert not stale_index.exists()
        assert readme.exists()


class TestEmptyRecommendedBucketExplanation:
    """Regression coverage for the empty recommended bucket explanation."""

    def test_recommended_bucket_shows_explanation(self) -> None:
        """Only the recommended bucket should explain why no model qualified."""
        markdown: list[str] = []
        check_models._append_review_user_buckets(
            markdown,
            {"recommended": [], "caveat": [], "needs_triage": [], "avoid": []},
        )

        none_lines = [line for line in markdown if line.startswith("- None")]
        explanation_lines = [line for line in none_lines if "quality thresholds" in line]
        plain_none_lines = [line for line in none_lines if line.strip() == "- None."]
        assert len(explanation_lines) == 1
        assert len(plain_none_lines) == 3


def test_quality_signal_summary_reports_incomplete_thinking_without_fault_language() -> None:
    """Developer prose should describe an unfinished expected thinking protocol."""
    analysis = check_models.analyze_generation_text(
        "◁think▷Inspecting the image step by step.",
        generated_tokens=500,
        requested_max_tokens=500,
        model_name="mlx-community/Kimi-VL-A3B-Thinking-8bit",
    )

    summary = " ".join(check_models._summarize_quality_signals(analysis))

    assert "Thinking trace incomplete" in summary
    assert "expected model protocol" in summary
    assert "leaked reasoning" not in summary
