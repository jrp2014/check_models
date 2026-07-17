"""Tests for report generation edge cases (empty input, all-failed results)."""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from argparse import Namespace
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

from PIL import Image

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
    bundle_path = output_dir / "repro_bundles" / "broken.json"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    check_models._write_text_file(bundle_path, "{}")
    repro_bundles = {failure.model_name: bundle_path}
    diagnostics_snapshot = _build_diagnostics_snapshot(results=results, prompt=prompt)

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
            repro_bundles=repro_bundles,
        )
        generate_diagnostics_report(
            results=results,
            filename=output_paths.diagnostics,
            versions=versions,
            system_info=system_info,
            prompt=prompt,
            repro_bundles=repro_bundles,
            diagnostics_snapshot=diagnostics_snapshot,
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
            diagnostics_snapshot=diagnostics_snapshot,
            output_dir=output_dir,
            versions=versions,
            system_info=system_info,
            repro_bundles=repro_bundles,
            run_args=None,
            prompt=prompt,
        )
        check_models.generate_output_index_report(
            output_paths.index,
            output_paths=output_paths,
            report_context=report_context,
            diagnostics_artifacts=DiagnosticsArtifacts(
                snapshot=diagnostics_snapshot,
                diagnostics_written=True,
                repro_bundles=repro_bundles,
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


def test_report_context_caches_cross_artifact_views() -> None:
    """One context should own recommendations, diagnostics, and issue clusters."""
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
    assert context.diagnostics_snapshot.failed == (failed,)
    assert context.issue_clusters == check_models._build_issue_clusters(
        context.diagnostics_snapshot
    )


def test_html_and_markdown_share_task_outcome_and_policy(tmp_path: Path) -> None:
    """HTML and the selection brief should expose the same reliability policy."""
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
    assert "Task outcome: crashed" in html_text
    assert "reliability-gated" in html_text
    assert "reliability-gated" in selection_text


def test_html_and_supporting_markdown_do_not_name_ineligible_legacy_winners(
    tmp_path: Path,
) -> None:
    """Legacy summary highlights must obey the canonical reliability gate."""
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
    markdown_path = tmp_path / "results.md"

    generate_html_report(
        results,
        html_path,
        versions={},
        prompt="Create title, description, and keywords.",
        total_runtime_seconds=2.0,
        report_context=context,
    )
    generate_markdown_report(
        results,
        markdown_path,
        versions={},
        prompt="Create title, description, and keywords.",
        total_runtime_seconds=2.0,
        report_context=context,
    )

    assert context.summary["fastest_model"][0] == "org/eligible"
    assert context.summary["most_efficient_model"][0] == "org/eligible"
    assert context.summary["fastest_load_model"][0] == "org/eligible"
    cataloging_best = context.summary["cataloging_best"]
    if cataloging_best is None:
        message = "cataloging_best"
        raise AssertionError(message)
    assert cataloging_best[0] == "org/eligible"
    html_text = html_path.read_text(encoding="utf-8")
    markdown_text = markdown_path.read_text(encoding="utf-8")
    assert "org/fast-warning" in html_text
    assert "org/fast-warning" in markdown_text
    assert "<b>Fastest:</b> <code>org/eligible</code>" in html_text
    assert "- **Fastest:** `org/eligible`" in markdown_text
    assert "<b>Best for cataloging:</b> <code>org/eligible</code>" in html_text
    assert "- **Best for cataloging:** `org/eligible`" in markdown_text


def test_all_ineligible_html_keeps_cataloging_aggregates_without_winner(
    tmp_path: Path,
) -> None:
    """An all-warning run should retain aggregate evidence without naming a winner."""
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
    assert context.summary["cataloging_best"] is None
    assert "Cataloging Utility Summary" in html_text
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
    assert "load" in cached.maintainer_triage_payload["summary"].casefold()
    assert "formatting" not in cached.maintainer_triage_payload["summary"].casefold()
    assert "text-sanity" not in cached.maintainer_triage_payload["summary"].casefold()

    review_rows = dict(check_models._build_review_block_rows(cached))
    assert "load" in review_rows["Why"].casefold()
    assert "formatting" not in review_rows["Why"].casefold()
    assert "text-sanity" not in review_rows["Why"].casefold()

    assert len(context.recommendations) == 1
    recommendation_caveats = " | ".join(context.recommendations[0].caveats).casefold()
    assert "formatting" not in recommendation_caveats
    assert "text-sanity" not in recommendation_caveats
    assert check_models._format_table_field_value("quality_issues", cached) == ""
    assert check_models._build_jsonl_result_record_base(cached)["quality_issues"] == []


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
    assert narrative.owner_confidence == "low"
    assert cached.review_payload is not None
    assert cached.review_payload["owner"] == narrative.suspected_owner
    assert context.issue_clusters[0].owner == narrative.suspected_owner


def test_published_failure_artifacts_match_canonical_runtime_triage() -> None:
    """Checked-in issue-ready reports must not retain pre-fix quality classifications."""
    output_dir = Path(__file__).parents[1] / "output"
    records = [
        json.loads(line)
        for line in (output_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    failures = [
        record for record in records if record.get("_type") == "result" and not record["success"]
    ]

    assert failures
    diagnostics = (output_dir / "reports/diagnostics.md").read_text(encoding="utf-8")
    review_report = (output_dir / "reports/review.md").read_text(encoding="utf-8")
    gallery = (output_dir / "reports/model_gallery.md").read_text(encoding="utf-8")
    html_report = (output_dir / "reports/results.html").read_text(encoding="utf-8")
    assert str(Path.home()) not in diagnostics
    assert str(Path.home()) not in review_report
    assert str(Path.home()) not in gallery
    assert str(Path.home()) not in html_report
    for failure in failures:
        review = failure["review"]
        triage = failure["maintainer_triage"]
        assert review["verdict"] == "runtime_failure"
        assert triage["issue_kind"] == "runtime_failure"
        assert failure["compatibility_status"] == "crashed"
        assert failure["quality_issues"] == []
        assert review["owner"] == triage["suspected_owner"]
        assert review["owner"] in diagnostics
        assert f"`{failure['model']}`" in review_report
        assert "`runtime_failure`" in review_report
        assert (output_dir / triage["issue_cluster_path"]).is_file()


def test_direct_serializers_build_one_local_review_cache(tmp_path: Path) -> None:
    """Legacy direct serializers should classify once through their fallback context."""
    result = _make_success("org/direct")

    with patch.object(
        check_models,
        "_build_jsonl_review_record",
        wraps=check_models._build_jsonl_review_record,
    ) as review_builder:
        check_models.save_jsonl_report(
            [result],
            tmp_path / "direct.jsonl",
            prompt="Describe the image.",
            system_info={},
        )
        check_models.append_history_record(
            history_path=tmp_path / "direct.history.jsonl",
            results=[result],
            prompt="Describe the image.",
            system_info={},
            library_versions={},
        )

    assert review_builder.call_count == 2


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
        end_headings=("## Repository Variant Comparisons",),
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


def test_ineligible_current_view_cannot_receive_positive_capability_recommendation(
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
    assert view.eligible is False
    assert "configured chooser threshold" in view.eligibility_reason
    assert model["current_status"] == "ineligible"
    assert model["recommendation"] == "current-run-ineligible"
    assert model["recommendation"] not in {"caption", "caption+keywords", "keywords"}
    assert "configured chooser threshold" in model["latest_signal"]


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
        assert "caption+keywords" in markdown
        assert model_payload["model"] == "org/model-grounded"
        assert model_payload["runs"] == 2
        assert payload["history_runs_considered"] == 1
        assert model_payload["success_rate"] == 100.0
        assert model_payload["metadata_alignment_avg"] > 70.0
        assert model_payload["recommendation"] == "caption+keywords"

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
        assert "caption" in payload["models"][0]["recommendation"]
        assert payload["models"][0]["recommendation"] != "avoid"

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
        assert "current-run-blocked" in markdown
        assert model_payload["current_status"] == "failed"
        assert model_payload["recommendation"] == "current-run-blocked"


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

    def test_triage_html_report_suppresses_cataloging_scores(self, tmp_path: Path) -> None:
        """Triage HTML should publish run-index context instead of cataloging scorecards."""
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
        assert "Run Contract" in content
        assert "<b>Evaluation lane:</b> triage" in content
        assert "<b>Metadata exposed to prompt:</b> no" in content
        assert "<b>Semantic rankings:</b> ungrounded" in content
        assert "Cataloging Utility Summary" not in content
        assert "Best keywording" not in content
        assert "Keywords 0" not in content
        assert "Keywords 100" not in content
        assert "caption-review candidate" in content

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
                "transformers==5.4.0 is below minimum 5.7.0 required by check_models.",
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

    def test_html_report_adds_filterable_row_attributes_and_numeric_classes(
        self, tmp_path: Path
    ) -> None:
        """HTML results table should expose row metadata and numeric alignment classes."""
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
        assert content.count('data-status="success"') == 1
        assert content.count('data-status="failed"') == 1
        assert 'data-error-stage="load"' in content
        assert 'data-error-type="ValueError"' in content
        assert 'data-error-package="mlx-vlm"' in content
        assert re.search(r'class="numeric"', content) is not None
        assert 'class="text failed"' in content

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
        assert "<pre>&lt;script&gt;alert(&quot;prompt&quot;)&lt;/script&gt;</pre>" in content


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
            results=[_make_failure_with_details("org/broken")],
            filename=generated_paths[5],
            versions=_stub_versions(),
            system_info={},
            prompt=prompt,
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
            "issues/index.md",
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
                "tsv_header": tsv_lines[1].split("\t"),
                "jsonl_header": jsonl_records[0]["_type"],
                "jsonl_models": [record["model"] for record in jsonl_records[1:]],
                "run_json_counts": run_payload["counts"],
                "run_json_artifacts": sorted(run_payload["artifacts"]),
                "capability_models": [
                    model_payload["model"] for model_payload in capability_payload["models"]
                ],
            }

            assert tsv_lines[0].startswith("# generated_at:")
            assert "Generated Text" in tsv_lines[1]
            assert jsonl_records[0]["_type"] == "metadata"
            assert len(jsonl_records[1:]) == 2
            assert run_payload["schema_version"] == "1.0"
            assert run_payload["counts"] == {
                "models_total": 2,
                "models_successful": 1,
                "models_failed": 1,
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
        assert "## ✅ Usable Diagnostic Candidates" in content
        assert "_Best end-to-end cataloging:_" in content
        assert "_Best descriptions:_" in content
        assert "_Best keywording:_" in content
        assert "## 🔍 Quality Pattern Breakdown" in content

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
        assert "Scope: ranked shortlist, not the complete run" in content
        assert "complete per-model outputs and diagnostics are in" in content
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
        )

        check_models.generate_model_selection_report(
            [tiny, mid, large, failure],
            out,
            prompt="Describe this image briefly.",
            report_context=context,
        )

        content = out.read_text(encoding="utf-8")
        assert "## Quick Chooser" in content
        assert "### Best under 4 GB" in content
        assert "### Best under 8 GB" in content
        assert "### Fastest usable" in content
        assert "### Quality if memory allows" in content
        assert "### Current failures / avoid" in content
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
        assert "text-sanity" in avoid_rows

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
        assert "`org/label-caption`" in avoid

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

    def test_jsonl_metadata_agreement_includes_assisted_enrichment_components(self) -> None:
        """JSONL should expose available assisted enrichment component scores."""
        metrics = check_models.MetadataAgreementMetrics(
            overall_score=84.0,
            context_integration_score=75.0,
            draft_improvement_score=45.0,
            visual_description_score=90.0,
            assisted_enrichment_score=76.5,
        )

        payload = check_models._build_jsonl_metadata_agreement_record(metrics)

        assert payload is not None
        assert payload["context_integration_score"] == 75.0
        assert payload["draft_improvement_score"] == 45.0
        assert payload["visual_description_score"] == 90.0
        assert payload["assisted_enrichment_score"] == 76.5

    def test_authoritative_only_assisted_record_keeps_visual_description_component(self) -> None:
        """Authoritative-only assisted records should retain visual-description scoring."""
        metrics = check_models.compute_metadata_agreement(
            (
                "Title: Sailboats on Deben Estuary at Woodbridge\n"
                "Description: Two white sailboats rest on calm water before a wooded bank.\n"
                "Keywords: sailboats, estuary, Woodbridge, calm water, wooded bank"
            ),
            {"keywords": "Deben Estuary, Woodbridge"},
        )

        payload = check_models._build_jsonl_metadata_agreement_record(metrics)

        assert metrics.context_integration_score is not None
        assert metrics.draft_improvement_score is None
        assert metrics.visual_description_score is not None
        assert metrics.assisted_enrichment_score is not None
        assert payload is not None
        assert payload["visual_description_score"] == metrics.visual_description_score

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
        hint_text = check_models._review_hint_text(review, analysis)
        utility_text = check_models._review_utility_text(review, analysis)
        focus_text = check_models._review_focus_text(review, analysis)
        combined = f"{hint_text} | {utility_text} | {focus_text}"
        assert "unverified-context-copy" in combined
        assert "low-draft-improvement" in combined
        assert "nonvisual metadata reused" not in combined
        assert "metadata borrowing" not in combined


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
        assert "Full generated output by model" in content
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

    def test_gallery_includes_issue_style_quality_summary_and_version_stamps(
        self,
        tmp_path: Path,
    ) -> None:
        """Gallery should provide a pasteable run summary with package version stamps."""
        out = tmp_path / "model_gallery.md"
        generate_markdown_gallery_report(
            results=[
                _make_quality_success("org/good", with_quality_issue=False),
                _make_harness_success(
                    "org/risky",
                    text="answer with | pipe and <think>leaked marker</think>",
                    harness_type="stop_token",
                    harness_detail="token_leak:<|end|>",
                ),
                _make_failure("org/bad", error_package="mlx-vlm"),
            ],
            filename=out,
            prompt="Describe this image briefly.",
            versions=_stub_versions(),
        )

        content = out.read_text(encoding="utf-8")
        assert "## Run Stamps" in content
        assert "- `mlx-vlm`: `0.1`" in content
        assert "- `mlx`: `0.1`" in content
        assert "## Model Quality Summary" in content
        assert "<!-- markdownlint-disable MD034 -->" in content
        assert "<!-- markdownlint-enable MD034 -->" in content
        assert "<!-- markdownlint-disable MD013 MD034 -->" not in content
        assert "<!-- markdownlint-enable MD013" not in content

        summary = _extract_markdown_subsection(
            content,
            "## Model Quality Summary",
            end_headings=(
                "## Image Metadata",
                "## Prompt",
                "## All Model Output and Cost Summary",
                "## Quick Navigation",
            ),
        )
        assert "Response / diagnostic" in summary
        assert "[`org/good`](#model-org-good)" in summary
        assert "quality output" in summary
        assert "[`org/risky`](#model-org-risky)" in summary
        assert "`avoid` / `harness`" in summary
        assert "harness:stop-token" in summary
        assert r"answer with \| pipe" in summary
        assert "[harness:stop-token] answer" not in summary
        assert "&lt;think&gt;leaked marker&lt;/think&gt;" in summary
        assert "[`org/bad`](#model-org-bad)" in summary
        assert "`avoid` / `runtime failure`" in summary
        assert "mlx-vlm; load" in summary
        assert "Error: load - boom" in summary

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

        generate_markdown_gallery_report(
            results=[success, failure, harness],
            filename=out,
            prompt="Describe this image briefly.",
        )

        content = out.read_text(encoding="utf-8")
        summary = _extract_markdown_subsection(
            content,
            "## All Model Output and Cost Summary",
            end_headings=("## Quick Navigation", "## Model Gallery"),
        )
        assert "Output / diagnostic" in summary
        assert "Peak GB" in summary
        assert "Quality signal" in summary
        assert "[`org/full-caption`](#model-org-full-caption)" in summary
        assert "Two cats sit together on a pink sofa" in summary
        assert "24" in summary
        assert "1.25s" in summary
        assert "42.0" in summary
        assert "2.5" in summary
        assert "clean" in summary
        risky_row = next(line for line in summary.splitlines() if "org/risky-output" in line)
        assert "| cats " in risky_row
        assert "[harness" not in risky_row
        assert "harness:prompt-template" in risky_row
        assert "[`org/crashed`](#model-org-crashed)" in summary
        assert "Error: load - boom" in summary
        assert "transformers; load" in summary
        assert "0.33s" in summary

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
        assert content.startswith(
            "<!-- markdownlint-disable MD012 MD013 -->\n\n# Automated Review Digest"
        )
        assert "# Automated Review Digest" in content
        assert "## Maintainer Escalations" in content
        assert "issues/index.md" in content
        assert (
            "https://github.com/jrp2014/check_models/blob/main/src/output/issues/index.md"
        ) in content

        # Now test relative local links when the link-style state is "relative"
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
            assert "../issues/index.md" in content_relative
        assert "## 🧭 Review Shortlist" in content
        assert "## User Buckets" in content
        assert "## Model Verdicts" in content
        assert "## Maintainer Queue" not in content
        assert "`mlx-vlm`" in content or "`transformers`" in content
        assert "`clean-triage-pass`" in content
        assert "`avoid`" in content
        assert content.index("## User Buckets") < content.index("## Maintainer Escalations")
        assert content.index("## Maintainer Escalations") < content.index("## Model Verdicts")
        assert "Model" in content
        assert "Hint Handling" in content
        assert "Key Evidence" in content
        assert "Evidence Bundle" in content
        assert "Fixed When" in content
        assert "Canonical run log" in content
        assert "Treat as a model-quality limitation" not in content
        maintainer_queue = content.split("## Maintainer Escalations", maxsplit=1)[1].split(
            "## Model Verdicts",
            maxsplit=1,
        )[0]
        assert "<!-- markdownlint-disable MD060 -->" in maintainer_queue
        assert "<!-- markdownlint-enable MD060 -->" in maintainer_queue
        assert "org/good" not in maintainer_queue
        assert "org/risky" in maintainer_queue

    def test_review_report_links_issue_repro_bundles_when_available(
        self,
        tmp_path: Path,
    ) -> None:
        """Review maintainer queue should carry repro bundle links, not placeholder dashes."""
        out = tmp_path / "review.md"
        risky = _make_harness_success(
            "org/risky", harness_type="stop_token", harness_detail="token_leak:<s>"
        )
        report_context = _build_report_render_context(results=[risky], prompt="describe")
        bundle_path = tmp_path / "repro_bundles" / "risky.json"

        generate_review_report(
            results=[risky],
            filename=out,
            prompt="describe",
            report_context=report_context,
            repro_bundles={risky.model_name: bundle_path},
        )

        content = out.read_text(encoding="utf-8")
        maintainer_queue = content.split("## Maintainer Escalations", maxsplit=1)[1].split(
            "## Model Verdicts",
            maxsplit=1,
        )[0]
        assert "[repro JSON]" in maintainer_queue
        assert "risky.json" in maintainer_queue

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
        guidance = check_models._review_next_action_text(review)
        assert review["prompt_burden_kind"] == "unavailable"
        assert "normal burden issue" not in guidance
        assert "input composition is unavailable" in guidance

    def test_gallery_includes_summary_pointer_and_per_model_review_status(
        self,
        tmp_path: Path,
    ) -> None:
        """Gallery should point to summaries while keeping per-model review status."""
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
        assert "## 🧭 Review Shortlist" not in content
        assert "## 🚨 Failures by Package (Actionable)" not in content
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
    failure_phase: str | None = None,
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
        failure_phase=failure_phase,
        error_message=error_msg,
        error_type=error_type,
        error_package=error_package,
        captured_output_on_fail=captured_output,
        error_traceback=traceback_str,
    )


class TestDiagnosticsReport:
    """Tests for generate_diagnostics_report and its helpers."""

    def test_diagnostics_is_self_contained_mlx_vlm_issue(self, tmp_path: Path) -> None:
        """Diagnostics should be directly pasteable as an mlx-vlm issue."""
        failure = _make_failure_with_details(
            "org/crashing-model",
            error_msg="generation failed",
            error_stage="Model Error",
            error_package="mlx-vlm",
            traceback_str="Traceback\nIndexError: token id out of range",
        )
        output = tmp_path / "diagnostics.md"

        written = generate_diagnostics_report(
            results=[failure],
            filename=output,
            versions={"mlx-vlm": "0.6.5", "mlx": "0.32.1"},
            system_info={"GPU/Chip": "Apple M5 Max", "RAM": "128 GB"},
            prompt="Describe the image.",
            image_path=tmp_path / "fixture.jpg",
            run_args=Namespace(max_tokens=500, temperature=0.0, top_p=1.0),
        )

        content = output.read_text(encoding="utf-8")
        assert written is True
        assert "## Crash / Failure Matrix" in content
        assert "Task outcome: crashed" in content
        assert "python -m mlx_vlm.generate" in content
        assert "## Expected and Actual Behaviour" in content
        assert "## Models Not Flagged" not in content
        assert "Total model runtime (sum)" not in content

    def test_model_only_text_quality_is_not_confirmed_mlx_vlm_issue(
        self,
        tmp_path: Path,
    ) -> None:
        """Model-owned text quality should remain an observation."""
        result = _make_quality_success("org/model-only", with_quality_issue=False)
        analysis = cast("GenerationQualityAnalysis", result.quality_analysis)
        result = replace(
            result,
            quality_analysis=replace(
                analysis,
                text_sanity_issue_type="numeric_loop",
                owner="model",
                user_bucket="avoid",
            ),
        )
        output = tmp_path / "diagnostics.md"

        generate_diagnostics_report(
            results=[result],
            filename=output,
            versions={"mlx-vlm": "0.6.5"},
            system_info={},
            prompt="Describe the image.",
        )

        content = output.read_text(encoding="utf-8")
        assert "mlx-vlm / MLX Issue Matrix" not in content
        assert "Model/config observations" in content

    def test_non_scope_crash_stays_in_crash_matrix_and_observations(
        self,
        tmp_path: Path,
    ) -> None:
        """Every crash remains visible even when its owner is out of filing scope."""
        output = tmp_path / "diagnostics.md"
        failure = _make_failure_with_details(
            "org/transformers-crash",
            error_msg="processor API mismatch",
            error_package="transformers",
            error_stage="API Mismatch",
        )

        generate_diagnostics_report(
            results=[failure],
            filename=output,
            versions={"mlx-vlm": "0.6.5", "transformers": "5.12.0"},
            system_info={},
            prompt="Describe the image.",
        )

        content = output.read_text(encoding="utf-8")
        assert "## Crash / Failure Matrix" in content
        assert "Task outcome: crashed" in content
        assert "## mlx-vlm / MLX Issue Matrix" not in content
        assert "## Model/config observations" in content
        assert "org/transformers-crash" in content

    def test_preflight_warnings_are_model_config_observations(self, tmp_path: Path) -> None:
        """Preflight-only signals should stay outside confirmed upstream issues."""
        output = tmp_path / "diagnostics.md"

        written = generate_diagnostics_report(
            results=[_make_success()],
            filename=output,
            versions=_stub_versions(),
            system_info={},
            prompt="Describe the image.",
            history=DiagnosticsHistoryInputs(
                preflight_issues=(
                    "transformers compatibility warning",
                    "mlx runtime cache warning",
                ),
            ),
        )

        content = output.read_text(encoding="utf-8")
        assert written is True
        assert "## Model/config observations" in content
        assert "transformers compatibility warning" in content
        assert "mlx runtime cache warning" in content
        assert "## mlx-vlm / MLX Issue Matrix" not in content

    def test_scoped_harness_cluster_is_inline_issue(self, tmp_path: Path) -> None:
        """An mlx-vlm-owned harness cluster should include its complete issue body."""
        output = tmp_path / "diagnostics.md"
        result = _make_harness_success(
            "org/stop-token",
            text="caption <|end|>",
            harness_type="stop_token",
            harness_detail="token_leak:<|end|>",
        )

        generate_diagnostics_report(
            results=[result],
            filename=output,
            versions={"mlx-vlm": "0.6.5"},
            system_info={},
            prompt="Describe the image.",
        )

        content = output.read_text(encoding="utf-8")
        assert "## mlx-vlm / MLX Issue Matrix" in content
        assert "## Expected and Actual Behaviour" in content
        assert "python -m mlx_vlm.generate" in content
        assert "issues/index.md" not in content

    def test_text_sanity_cluster_requires_output_excerpt(self) -> None:
        """A detector label without generated evidence should not form a text issue."""
        result = _make_quality_success("org/no-excerpt", with_quality_issue=False)
        analysis = cast("GenerationQualityAnalysis", result.quality_analysis)
        result = replace(
            result,
            generation=replace(cast("_MockGeneration", result.generation), text=""),
            quality_analysis=replace(analysis, text_sanity_issue_type="numeric_loop"),
        )

        snapshot = _build_diagnostics_snapshot(
            results=[result],
            prompt="Describe the image.",
        )

        assert snapshot.text_sanity_results == ()

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

    def test_diagnostics_stack_routing_matches_exact_package_components(self) -> None:
        """Composite MLX owners route upstream without substring false positives."""

        def cluster(owner: str) -> check_models.IssueCluster:
            return check_models.IssueCluster(
                cluster_id="stack",
                issue_filename="issue_stack.md",
                owner=owner,
                issue_kind="stack_signal",
                issue_subtype="long_context",
                symptom_family="context",
                symptom="context collapse",
                acceptance_signal="generation completes",
                source="stack_signal",
                sort_rank=1,
            )

        assert check_models._issue_cluster_is_mlx_vlm_scope(cluster("mlx-vlm / mlx"))
        assert check_models._issue_cluster_is_mlx_vlm_scope(cluster("transformers / mlx-vlm"))
        assert not check_models._issue_cluster_is_mlx_vlm_scope(cluster("not-mlx-vlm"))

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

    def test_diagnostics_harness_sample_output_keeps_late_leak_marker(self) -> None:
        """Diagnostics samples should show the exact leaked marker, not only the output head."""
        leaked_text = "background context " * 80 + "leaked control token <|end|> after."
        result = _make_harness_success(
            text=leaked_text,
            harness_type="stop_token",
            harness_detail="token_leak:<|end|>",
        )

        excerpt = check_models._issue_output_excerpt(result, max_chars=240)
        assert "leaked control token <|end|> after" in excerpt

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

    def test_action_summary_escapes_full_multi_underscore_runs(self, tmp_path: Path) -> None:
        """Diagnostics should fully escape odd-length underscore runs."""
        out = tmp_path / "diag.md"
        generate_diagnostics_report(
            results=[
                _make_failure_with_details(
                    "org/bad-model",
                    error_msg="Tokenizer produced _____ is _____ in summary output",
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
        assert r"\_\_\_\_\_ is \_\_\_\_\_" in content
        assert r"\_\_\_\__ is" not in content

    def test_environment_table_includes_versions(self, tmp_path: Path) -> None:
        """Environment table should include library versions and system info."""
        out = tmp_path / "diag.md"
        versions = _stub_versions()
        versions["mlx-vlm"] = "0.3.11"
        versions["mlx-metal"] = "0.3.11"
        generate_diagnostics_report(
            results=[_make_failure_with_details()],
            filename=out,
            versions=versions,
            system_info={
                "Python Version": "3.13.9",
                "GPU/Chip": "Apple M4 Max",
                "SDK Version": "26.5",
                "Xcode Version": "26.5",
                "Xcode Build": "17F42",
                "Metal Compiler Version": "Apple metal version 32023.883",
                "Metallib Linker Version": "AIR-LLD 32023.883",
                "MLX Install Type": "wheel/site-packages",
                "MLX Metallib": "/site-packages/mlx/lib/mlx.metallib (sha256=abc123)",
            },
            prompt="test",
        )
        content = out.read_text(encoding="utf-8")
        assert "mlx-vlm" in content
        assert "0.3.11" in content
        assert "mlx-metal" in content
        assert "Python Version" in content
        assert "3.13.9" in content
        assert "GPU/Chip" in content
        assert "Apple M4 Max" in content
        assert "SDK Version" in content
        assert "26.5" in content
        assert "Xcode Version" in content
        assert "Metal Compiler Version" in content
        assert "AIR-LLD 32023.883" in content
        assert "MLX Metallib" in content
        assert "sha256=abc123" in content

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
        assert content.count("## Expected and Actual Behaviour") == 1
        assert "Affected models:_ org/model-a, org/model-b" in content
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

    def test_multiple_diagnostic_clusters_use_lint_safe_heading_hierarchy(
        self,
        tmp_path: Path,
    ) -> None:
        """Inline issue bodies should be distinct and Markdown-lint clean."""
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
        assert content.count("\n## Issue ") == 2
        assert content.count("\n### Expected and Actual Behaviour\n") == 2
        assert content.count("\n### Native mlx-vlm reproduction\n") == 2
        assert content.count("\n### Expected Fix Signal\n") == 2
        assert "\n- _Filing target:_ mlx-vlm\n\n### Native mlx-vlm reproduction\n" in content
        assert "\n- _Filing target:_ mlx\n\n### Native mlx-vlm reproduction\n" in content

    def test_captured_output_omitted_from_compact_diagnostics(self, tmp_path: Path) -> None:
        """Captured stdout/stderr belongs in issue drafts and bundles, not diagnostics."""
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
        assert "Detailed trace logs" not in content
        assert "Tokenizer warning here" not in content
        assert "Captured stdout/stderr" not in content
        assert "Downloading: 100%" not in content
        assert "\x1b[" not in content
        assert "\r" not in content
        assert "`org/m`" in content

    def test_failure_section_uses_upstream_traceback_not_local_wrappers(
        self,
        tmp_path: Path,
    ) -> None:
        """Failure sections should surface upstream stack frames before harness wrappers."""
        traceback_text = (
            "Traceback (most recent call last):\n"
            '  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", '
            "line 19044, in _run_model_generation\n"
            "    model, processor, config = _load_model(params)\n"
            '  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", '
            "line 534, in load_model\n"
            "    model_config = apply_generation_config_defaults(model_config, config)\n"
            '  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", '
            "line 69, in apply_generation_config_defaults\n"
            "    setattr(model_config, key, config[key])\n"
            "AttributeError: property 'eos_token_id' of 'ModelConfig' object has no setter\n"
        )
        out = tmp_path / "diag.md"

        generate_diagnostics_report(
            results=[
                _make_failure_with_details(
                    "org/upstream-fail",
                    error_msg=(
                        "Model loading failed: property 'eos_token_id' of "
                        "'ModelConfig' object has no setter"
                    ),
                    traceback_str=traceback_text,
                ),
            ],
            filename=out,
            versions=_stub_versions(),
            system_info={},
            prompt="test",
        )

        content = out.read_text(encoding="utf-8")
        failure_section = _extract_markdown_subsection(
            content,
            "## Crash / Failure Matrix",
            end_headings=["## mlx-vlm / MLX Issue Matrix"],
        )
        assert "Crash evidence: org/upstream-fail" in failure_section
        assert "mlx_vlm/utils.py" in failure_section
        assert "AttributeError: property 'eos_token_id'" in failure_section
        assert "check_models/src/check_models.py" not in failure_section
        assert "_run_model_generation" not in failure_section

    def test_diagnostics_make_local_paths_home_relative(self, tmp_path: Path) -> None:
        """Public issue artifacts should not expose the local account path."""
        home = str(Path.home())
        traceback_text = (
            "Traceback (most recent call last):\n"
            f'  File "{home}/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", '
            "line 69, in load\n"
            "    raise ValueError('bad config')\n"
            "ValueError: bad config\n"
        )
        out = tmp_path / "diag.md"

        generate_diagnostics_report(
            results=[
                _make_failure_with_details(
                    "org/upstream-fail",
                    error_msg="bad config",
                    traceback_str=traceback_text,
                ),
            ],
            filename=out,
            versions=_stub_versions(),
            system_info={
                "MLX Distribution Root": f"{home}/miniconda3/envs/mlx-vlm/site-packages",
            },
            prompt="test",
        )

        content = out.read_text(encoding="utf-8")
        assert home not in content
        assert 'File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py"' in content
        assert "~/miniconda3/envs/mlx-vlm/site-packages" in content

    def test_failure_section_uses_one_chronological_failure_narrative(
        self,
        tmp_path: Path,
    ) -> None:
        """Failure reports should lead with the root and retain later wrappers."""
        out = tmp_path / "diag.md"
        failure = replace(
            _make_failure_with_details(
                "org/chained-fail",
                error_msg="generation failed",
            ),
            exception_chain=(
                check_models.FailureException(
                    "IndexError",
                    "builtins",
                    "token id 999 outside detokenizer table",
                ),
                check_models.FailureException(
                    "RuntimeError",
                    "builtins",
                    "METAL command buffer out of memory",
                ),
                check_models.FailureException(
                    "ValueError",
                    "builtins",
                    "generation failed",
                ),
            ),
        )

        generate_diagnostics_report(
            results=[failure],
            filename=out,
            versions=_stub_versions(),
            system_info={},
            prompt="test",
        )

        content = out.read_text(encoding="utf-8")
        assert "IndexError: token id 999 outside detokenizer table" in content
        assert "RuntimeError: METAL command buffer out of memory" in content
        assert "ValueError: generation failed" in content

    def test_issue_queue_present(self, tmp_path: Path) -> None:
        """Issue matrix should contain inline evidence without draft dependencies."""
        out = tmp_path / "diag.md"
        generate_diagnostics_report(
            results=[
                _make_failure_with_details(
                    failure_phase="model_load",
                    error_type="ValueError",
                )
            ],
            filename=out,
            versions=_stub_versions(),
            system_info={},
            prompt="test",
        )
        content = out.read_text(encoding="utf-8")
        assert content.startswith("<!-- markdownlint-disable MD013 -->\n\n# Diagnostics Report")
        assert "<!-- markdownlint-disable MD060 -->" in content
        assert "<!-- markdownlint-enable MD060 -->" in content
        assert "## mlx-vlm / MLX Issue Matrix" in content
        assert "Target" in content
        assert "Evidence Snapshot" in content
        assert "Model Error" in content
        assert "phase model_load" in content
        assert "ValueError" in content
        assert "Confidence" in content
        assert "Evidence Type" in content
        assert "Fixed When" in content
        assert "issues/index.md" not in content
        assert "issue draft" not in content.casefold()
        assert "Priority" not in content
        assert content.index("## Crash / Failure Matrix") < content.index(
            "## mlx-vlm / MLX Issue Matrix"
        )
        assert content.index("## Native mlx-vlm reproduction") < content.index("## Environment")

    def test_failure_observed_behavior_keeps_multiline_error_details(
        self,
        tmp_path: Path,
    ) -> None:
        """Diagnostics should keep actionable details from multiline load errors."""
        out = tmp_path / "diag.md"
        generate_diagnostics_report(
            results=[
                _make_failure_with_details(
                    error_msg=(
                        "Model loading failed: Received 2 parameters not in model:\n"
                        "multi_modal_projector.layer_norm.bias,\n"
                        "multi_modal_projector.layer_norm.weight."
                    ),
                    error_type="ValueError",
                    error_stage="Model Error",
                    failure_phase="model_load",
                )
            ],
            filename=out,
            versions=_stub_versions(),
            system_info={},
            prompt="test",
        )
        content = out.read_text(encoding="utf-8")
        assert (
            "Received 2 parameters not in model: multi_modal_projector.layer_norm.bias,"
        ) in content
        assert "multi_modal_projector.layer_norm.weight." in content

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

    def test_report_written_for_text_sanity_success_without_failures(self, tmp_path: Path) -> None:
        """Successful token-soup generations should be detailed in diagnostics."""
        out = tmp_path / "diag.md"
        junk_text = (
            'open对不同方面">black/ with小猫小猫kotPicture •0超高清比!y表面处理超经典的!'
            "张图片'七- object Tno-go-head-or U0.C在其他 ** ,Not只!被i animal"
        )
        success = PerformanceResult(
            model_name="org/token-soup",
            success=True,
            generation=_MockGeneration(
                text=junk_text,
                prompt_tokens=319,
                generation_tokens=85,
            ),
            total_time=1.0,
            generation_time=0.5,
            model_load_time=0.5,
        )

        result = generate_diagnostics_report(
            results=[success],
            filename=out,
            versions=_stub_versions(),
            system_info={},
            prompt="Describe this image briefly.",
        )

        assert result is True
        content = out.read_text(encoding="utf-8")
        assert "## Model/config observations" in content
        assert "org/token-soup" in content
        assert "Generated text is mixed-script token-soup" in content
        assert "open对不同方面" in content
        assert "## mlx-vlm / MLX Issue Matrix" not in content
        assert "## Models Not Flagged" not in content

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
        assert "Crash evidence: org/partial" in content
        assert "Partial decoded answer before crash." in content

    def test_failure_repro_bundles_written(self, tmp_path: Path) -> None:
        """Failure repros should add one narrative while preserving legacy fields."""
        failure = replace(
            _make_failure_with_details(
                "org/broken-model",
                error_msg="generation failed",
                error_type="ValueError",
                error_stage="Model Error",
                error_package="mlx-vlm",
                traceback_str="Traceback\nValueError: generation failed",
            ),
            exception_chain=(
                check_models.FailureException(
                    "RuntimeError",
                    "mlx.core",
                    "kIOGPUCommandBufferCallbackErrorOutOfMemory",
                ),
                check_models.FailureException(
                    "ValueError",
                    "builtins",
                    "mlx_vlm/generate.py generation failed",
                ),
            ),
        )
        bundles = export_failure_repro_bundles(
            results=[failure],
            output_dir=tmp_path / "repro_bundles",
            run_args=Namespace(eval_mode="blind"),
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
        assert payload["failure"]["type"] == "ValueError"
        assert payload["failure"]["package"] == "mlx-vlm"
        assert payload["failure"]["message"] == "generation failed"
        assert payload["failure"]["stage"] == "Model Error"
        assert payload["failure"]["exception_chain"] == [
            {
                "type": "RuntimeError",
                "module": "mlx.core",
                "message": "kIOGPUCommandBufferCallbackErrorOutOfMemory",
            },
            {
                "type": "ValueError",
                "module": "builtins",
                "message": "mlx_vlm/generate.py generation failed",
            },
        ]
        assert payload["failure"]["task_outcome"] == "crashed"
        assert (
            payload["failure"]["primary_exception"]
            == "RuntimeError: kIOGPUCommandBufferCallbackErrorOutOfMemory"
        )
        assert payload["failure"]["secondary_exceptions"] == [
            "ValueError: mlx_vlm/generate.py generation failed"
        ]
        assert payload["failure"]["suspected_owner"] == "unresolved: mlx/mlx-vlm"
        assert payload["failure"]["owner_confidence"] == "low"
        assert (
            payload["result"]["maintainer_triage"]["suspected_owner"] == "unresolved: mlx/mlx-vlm"
        )
        assert payload["repro"]["prompt_hash_sha256"]
        assert payload["repro"]["eval_mode"] == "blind"

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
        assert "Optional advanced context:" in issue_content

    def test_repro_bundles_write_latest_cluster_index(self, tmp_path: Path) -> None:
        """Current-run bundles should be indexed separately from the historical archive."""
        failure = _make_failure_with_details(
            "org/broken-model",
            error_msg="bad shape",
            error_stage="Model Error",
            error_package="mlx-vlm",
            traceback_str="Traceback\nValueError: bad shape",
        )
        output_dir = tmp_path / "repro_bundles"

        bundles = export_failure_repro_bundles(
            results=[failure],
            output_dir=output_dir,
            run_args=Namespace(),
            versions=_stub_versions(),
            system_info={"Python Version": "3.13"},
            prompt="Describe this image.",
            image_path=None,
        )

        index_path = output_dir / "latest_by_cluster.json"
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        assert "org/broken-model" in bundles
        assert payload["schema_version"] == "1.0"
        assert payload["models"]["org/broken-model"]["bundle"] == bundles["org/broken-model"].name
        assert payload["models"]["org/broken-model"]["issue_cluster_id"]

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
        assert "--image 'sample image.jpg'" in content
        assert str(image_path) not in content

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
            force_download=True,
            quantize_activations=True,
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
        assert "--top-p" not in content
        assert "--min-p" not in content
        assert "--top-k" not in content
        assert "--resize-shape 768 512" in content
        assert "--eos-tokens '</think>'" in content
        assert "--skip-special-tokens" in content
        assert "--processor-kwargs '{\"cropping\": false}'" in content
        assert "--force-download" in content
        assert "--quantize-activations" in content
        assert "--enable-thinking" in content
        assert "--thinking-budget 96" in content
        assert "--thinking-start-token '<think>'" in content
        assert "--repetition-penalty" not in content
        assert "--repetition-context-size" not in content
        assert "--quantized-kv-start 2048" in content
        assert "--timeout" not in content
        assert "--no-trust-remote-code" not in content
        assert "--verbose" not in content

    def test_prompt_in_section(self, tmp_path: Path) -> None:
        """Prompt text should be referenced instead of pasted into diagnostics."""
        out = tmp_path / "diag.md"
        generate_diagnostics_report(
            results=[_make_failure_with_details()],
            filename=out,
            versions=_stub_versions(),
            system_info={},
            prompt="Analyze this image carefully.",
        )
        content = out.read_text(encoding="utf-8")
        assert "### Prompt Used" not in content
        assert "## Impact and Run Conditions" in content
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
        assert "Affected models:_ org/a, org/b" in content
        assert content.count("## Expected and Actual Behaviour") == 1
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

    def test_generate_tsv_report_includes_full_generated_text_for_analysis(
        self,
        tmp_path: Path,
    ) -> None:
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

        generate_tsv_report(
            results=[result],
            filename=out,
        )

        content = out.read_text(encoding="utf-8")
        assert "Generated Text" in content
        assert "</think> exact leak marker" in content

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

        assert (
            "Diagnostics signals: failures=1, harness=1, stack=0, text_sanity=0, preflight=1"
        ) in caplog.text
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
                "transformers==5.4.0 is below minimum 5.7.0 required by check_models.",
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
                "transformers==5.4.0 is below minimum 5.7.0 required by check_models.",
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

    def test_chained_exception_prefers_upstream_traceback_segment(self) -> None:
        """Wrapped local failures should not hide the upstream root traceback."""
        tb = (
            "Traceback (most recent call last):\n"
            '  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", '
            "line 19044, in _run_model_generation\n"
            "    model, processor, config = _load_model(params)\n"
            '  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", '
            "line 534, in load_model\n"
            "    model_config = apply_generation_config_defaults(model_config, config)\n"
            '  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", '
            "line 69, in apply_generation_config_defaults\n"
            "    setattr(model_config, key, config[key])\n"
            "AttributeError: property 'eos_token_id' of 'ModelConfig' object has no setter\n"
            "\n"
            "The above exception was the direct cause of the following exception:\n"
            "\n"
            "Traceback (most recent call last):\n"
            '  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", '
            "line 19285, in process_image_with_model\n"
            "    output = _run_model_generation(params=params)\n"
            '  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", '
            "line 19059, in _run_model_generation\n"
            "    raise ValueError(error_details) from load_err\n"
            "ValueError: Model loading failed: property 'eos_token_id' of "
            "'ModelConfig' object has no setter\n"
        )

        result = _format_traceback_tail(tb)

        assert result is not None
        assert "mlx_vlm/utils.py" in result
        assert "AttributeError: property 'eos_token_id'" in result
        assert "Model loading failed" not in result
        assert "check_models/src/check_models.py" not in result


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
        (tmp_path / "model_selection.md").write_text("old selection")
        (reports_dir / "model_selection.md").write_text("canonical selection")
        removed = _clean_stale_toplevel_reports(tmp_path, reports_dir)
        assert removed == 2
        assert not (tmp_path / "results.md").exists()
        assert not (tmp_path / "model_selection.md").exists()

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


class TestSharedReportSections:
    """Tests for shared Markdown/HTML report section primitives."""

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

    def test_check_models_repro_command_uses_shared_spec_path(self, tmp_path: Path) -> None:
        """Diagnostics and bundles should share canonical check_models command tokens."""
        image_path = tmp_path / "sample image.jpg"
        adapter_path = tmp_path / "adapter"
        quality_config_path = tmp_path / "quality.yaml"
        custom_end_marker = "</done>"
        run_args = Namespace(
            image=image_path,
            folder=None,
            models=["org/a", "org/b"],
            exclude=["org/skip"],
            trust_remote_code=True,
            revision="main",
            adapter_path=adapter_path,
            prompt="Describe this.",
            detailed_metrics=True,
            lazy_load=True,
            force_download=True,
            quantize_activations=True,
            skip_special_tokens=True,
            enable_thinking=True,
            max_tokens=123,
            temperature=0.2,
            top_p=0.8,
            min_p=0.1,
            top_k=4,
            resize_shape=(64, 32),
            eos_tokens=["</s>", "<|end|>"],
            processor_kwargs={"cropping": False},
            repetition_penalty=1.1,
            repetition_context_size=64,
            max_kv_size=4096,
            kv_bits=4,
            kv_quant_scheme="turboquant",
            prefill_step_size=512,
            thinking_budget=32,
            thinking_start_token=THINKING_START_TOKEN,
            thinking_end_token=custom_end_marker,
            kv_group_size=32,
            quantized_kv_start=128,
            timeout=33.0,
            verbose=True,
            no_color=True,
            force_color=False,
            width=100,
            quality_config=quality_config_path,
            context_marker="Visible context:",
        )

        spec = check_models.build_check_models_repro_command_spec(
            image_path=image_path,
            run_args=run_args,
            include_selection=True,
        )
        tokens = check_models._build_repro_command_tokens(
            image_path=image_path,
            run_args=run_args,
            include_selection=True,
        )

        assert tokens == list(spec.tokens())
        assert tokens[:5] == ["python", "-m", "check_models", "--image", str(image_path)]
        assert "--models" in tokens
        assert "--exclude" in tokens
        assert "--processor-kwargs" in tokens
        assert json.loads(tokens[tokens.index("--processor-kwargs") + 1]) == {"cropping": False}

    def test_check_models_repro_command_emits_logit_bias_once(self, tmp_path: Path) -> None:
        """Canonical check_models repro commands should not duplicate --logit-bias."""
        image_path = tmp_path / "sample.jpg"
        run_args = Namespace(
            trust_remote_code=True,
            logit_bias={42: -1.5},
        )

        spec = check_models.build_check_models_repro_command_spec(
            image_path=image_path,
            run_args=run_args,
            include_selection=False,
        )
        tokens = list(spec.tokens())

        assert tokens.count("--logit-bias") == 1
        assert json.loads(tokens[tokens.index("--logit-bias") + 1]) == {"42": -1.5}

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
        config_json = check_models._build_issue_inline_config_json(
            model_name="org/model",
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

        config = json.loads(config_json)
        assert config["generate_kwargs"]["top_p"] == 0.8
        assert config["generate_kwargs"]["min_p"] == 0.1
        assert config["generate_kwargs"]["top_k"] == 4
        assert config["generate_kwargs"]["repetition_penalty"] == 1.1
        assert config["generate_kwargs"]["repetition_context_size"] == 64


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

    def test_issue_draft_uses_native_mlx_vlm_repro_and_prunes_internal_jargon(
        self,
        tmp_path: Path,
    ) -> None:
        """Pasteable issue drafts should use native mlx-vlm repros, not harness commands."""
        image_path = tmp_path / "sample.jpg"
        image_path.write_text("placeholder", encoding="utf-8")
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
        run_args = Namespace(
            image=image_path,
            max_tokens=123,
            temperature=0.0,
            revision="main",
            trust_remote_code=True,
        )

        generated = _generate_github_issue_reports(
            diagnostics_snapshot=snapshot,
            output_dir=tmp_path,
            versions=_stub_versions(),
            system_info={"Python Version": "3.13"},
            repro_bundles={},
            run_args=run_args,
            prompt="Analyze this image.",
            image_path=image_path,
        )

        content = next(iter(generated.values())).read_text(encoding="utf-8")
        assert "python -m mlx_vlm.generate" in content
        assert "--model org/broken-model" in content
        assert "Analyze this image." in content
        assert "from mlx_vlm.utils import load" in content
        assert "from mlx_vlm.generate import generate" in content
        assert "python -m check_models" not in content
        assert "Raw cluster" not in content
        assert "Issue kind" not in content
        assert "Raw owner hint" not in content
        assert "Why this classification is credible" not in content
        assert "MLX_VLM_DECODE_RUNTIME" not in content

    def test_issue_draft_uses_portable_image_reference_with_hash(
        self,
        tmp_path: Path,
    ) -> None:
        """Pasteable upstream repros should not require the reporter's absolute paths."""
        image_path = tmp_path / "cats.jpg"
        image_path.write_bytes(b"fake image bytes")
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

        generated = _generate_github_issue_reports(
            diagnostics_snapshot=snapshot,
            output_dir=tmp_path,
            versions=_stub_versions(),
            system_info={"Python Version": "3.13"},
            repro_bundles={},
            run_args=Namespace(image=image_path, max_tokens=123, temperature=0.0),
            prompt="Analyze this image.",
            image_path=image_path,
        )

        content = next(iter(generated.values())).read_text(encoding="utf-8")
        assert str(image_path) not in content
        assert "--image cats.jpg" in content
        assert '"image": "cats.jpg"' in content
        assert "Image SHA256:" in content

    def test_issue_index_includes_run_context_header(self, tmp_path: Path) -> None:
        """Issue queue index should explain the run context before the table."""
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

        _generate_github_issue_reports(
            diagnostics_snapshot=snapshot,
            output_dir=tmp_path,
            versions={**_stub_versions(), "mlx-vlm": "0.5.0"},
            system_info={"OS": "macOS 26.4.1", "GPU/Chip": "Apple M5 Max"},
            repro_bundles={},
            run_args=None,
        )

        index = (tmp_path / "issues" / "index.md").read_text(encoding="utf-8")
        assert index.startswith("# Check Models Issue Queue")
        assert "Generated on:" in index
        assert "Test Environment:" in index
        assert "Apple M5 Max" in index
        assert "mlx-vlm 0.5.0" in index
        assert "Summary:" in index
        assert "1 hard failure" in index
        assert "Evidence Snapshot" in index
        assert "Model Error" in index
        assert "RuntimeError" in index

    def test_issue_affected_models_use_failure_message_not_generic_evidence(
        self,
        tmp_path: Path,
    ) -> None:
        """Issue affected-model rows should show the actionable runtime error."""
        failed_result = PerformanceResult(
            model_name="org/broken-model",
            generation=None,
            success=False,
            error_message=(
                "Model loading failed: Received 2 parameters not in model:\n"
                "multi_modal_projector.layer_norm.bias,\n"
                "multi_modal_projector.layer_norm.weight."
            ),
            root_error_message=(
                "Received 2 parameters not in model:\n"
                "multi_modal_projector.layer_norm.bias,\n"
                "multi_modal_projector.layer_norm.weight."
            ),
            root_error_type="ValueError",
            root_error_module="builtins",
            error_stage="Model Error",
            error_code="MLX_MODEL_LOAD_MODEL",
            error_package="mlx",
            error_signature="MLX_MODEL_LOAD_MODEL:abc123",
            error_traceback="Traceback (most recent call last):\nValueError: shape mismatch",
            total_time=0.5,
        )
        snapshot = DiagnosticsSnapshot(
            failed=(failed_result,),
            failure_clusters=(("MLX_MODEL_LOAD_MODEL:abc123", (failed_result,)),),
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
        affected_models = _extract_markdown_subsection(
            content,
            "## Affected Models",
            end_headings=["## Minimal Evidence"],
        )
        detailed_evidence = _extract_markdown_subsection(
            content,
            "## Appendix: Detailed Evidence",
            end_headings=[],
        )
        assert "<!-- markdownlint-disable MD060 -->" in affected_models
        assert "<!-- markdownlint-enable MD060 -->" in affected_models
        assert "Received 2 parameters not in model" in affected_models
        assert "multi_modal_projector.layer_norm.bias" not in affected_models
        assert "multi_modal_projector.layer_norm.bias" in detailed_evidence
        assert "model error | mlx model load model" not in affected_models

    def test_issue_traceback_omits_local_check_models_frames(self, tmp_path: Path) -> None:
        """Issue draft tracebacks should start at upstream frames when local wrappers exist."""
        traceback_text = (
            "Traceback (most recent call last):\n"
            '  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", '
            "line 17284, in _run_model_generation\n"
            "    output = generate(...)\n"
            '  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", '
            "line 419, in load\n"
            "    raise RuntimeError('shape mismatch')\n"
            "RuntimeError: shape mismatch\n"
        )
        failed_result = PerformanceResult(
            model_name="org/broken-model",
            generation=None,
            success=False,
            error_message="RuntimeError: shape mismatch",
            error_stage="Model Error",
            error_code="MLX_VLM_DECODE_RUNTIME",
            error_package="mlx-vlm",
            error_signature="MLX_DECODE_ERROR:abc123",
            error_traceback=traceback_text,
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

        content = next(iter(generated.values())).read_text(encoding="utf-8")
        assert "check_models.py" not in content
        assert "output = generate(...)" not in content
        assert "mlx_vlm/utils.py" in content

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
        assert content.startswith(
            "<!-- markdownlint-disable MD012 MD013 MD033 MD060 -->\n\n"
            "# \\[mlx-vlm\\]\\[MLX VLM decode runtime\\]"
        )
        assert "## Summary" in content
        assert "## Affected Models" in content
        assert "## Minimal Evidence" in content
        assert "## Appendix: Detailed Evidence" in content
        assert "## Minimal Reproduction" in content
        assert "## Fix Checklist" in content
        assert "## Expected Fix Signal" in content
        assert "## Appendix: Environment" in content
        assert "python -m mlx_vlm.generate" in content
        assert "python -m check_models" not in content
        assert "Raw cluster" not in content
        assert "MLX_VLM_DECODE_RUNTIME" not in content
        assert "runtime_failure" not in content
        assert "Traceback (most recent call last)" in content
        assert (
            "[optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/broken.json)"
        ) in content

        # Now test relative local links when the link-style state is "relative"
        with patch.object(check_models._LinkStyleState, "value", "relative"):
            generated_github = _generate_github_issue_reports(
                diagnostics_snapshot=snapshot,
                output_dir=tmp_path,
                versions=_stub_versions(),
                system_info={"Python Version": "3.13"},
                repro_bundles={"org/broken-model": bundle_path},
                run_args=None,
            )
            content_relative = next(iter(generated_github.values())).read_text(encoding="utf-8")
            assert "[optional JSON](../repro_bundles/broken.json)" in content_relative
        assert "Optional advanced context:" in content
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
        assert content.startswith(
            "<!-- markdownlint-disable MD012 MD013 MD033 MD060 -->\n\n"
            "# \\[mlx-vlm / mlx\\]\\[Long-context collapse\\]"
        )
        assert "## Minimal Reproduction" in content
        assert "mlx-vlm first; MLX if cache/runtime reproduces" in content
        assert "At long prompt length (5000 tokens), generation returned empty output." in content
        assert "context_budget" not in content
        assert "long_context" not in content
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
        assert "mlx-vlm_stop-token_001" not in content
        assert "&lt;\\|end\\|&gt;" in content
        assert "&lt;/think&gt;" in content

    def test_stop_token_issue_excerpt_keeps_late_leak_marker(self, tmp_path: Path) -> None:
        """Issue excerpts should center on exact late leak markers instead of only the head."""
        leaked_text = (
            "A long answer starts with plausible visual details. "
            + "background context " * 80
            + "</think> leaked control token after the long preface."
        )
        result = _make_harness_success(
            name="org/late-stop",
            text=leaked_text,
            generation_tokens=220,
            harness_type="stop_token",
            harness_detail="token_leak:</think>",
        )
        snapshot = DiagnosticsSnapshot(
            harness_results=((result, leaked_text),),
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
        assert "</think> leaked control token" in content
        assert "Output excerpt:" in content

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
        assert "Unsupported Granite model type/import path" in index
        assert "No module named" not in index
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
        assert "Long-context collapse" in content
        assert "Inspect cache allocation" in content


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
        snapshot=_build_diagnostics_snapshot(results=[good, failure], prompt="describe"),
        diagnostics_written=True,
        repro_bundles={"org/bad": output_dir / "repro_bundles" / "bad.json"},
        issue_reports={"cluster": issues_dir / "index.md"},
    )

    check_models.generate_output_index_report(
        paths.index,
        output_paths=paths,
        report_context=report_context,
        diagnostics_artifacts=artifacts,
    )

    content = paths.index.read_text(encoding="utf-8")
    assert "# Check Models Output Index" in content
    assert "- Models tested: 2" in content
    assert "- Successful: 1" in content
    assert "- Failed: 1" in content
    assert "## For Model Users" in content
    assert "## For Maintainers" in content
    assert "model_selection.md" in content
    assert "model_capabilities.md" in content
    assert "issues/index.md" in content
    assert "latest_by_cluster.json" in content
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
        "issues/index.md",
        "latest_by_cluster.json",
    ):
        assert artifact in supporting


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


def test_cached_thinking_model_analysis_is_refreshed_with_model_protocol() -> None:
    """Legacy cached reasoning labels should be reclassified for thinking models."""
    text = "◁think▷Inspecting the image step by step."
    legacy_analysis = check_models.analyze_generation_text(text, generated_tokens=500)
    result = PerformanceResult(
        model_name="mlx-community/Kimi-VL-A3B-Thinking-8bit",
        success=True,
        generation=_MockGeneration(text=text, generation_tokens=500),
        quality_analysis=legacy_analysis,
        quality_issues="reasoning-leak",
        requested_max_tokens=500,
    )

    refreshed = check_models._populate_result_quality_analysis(result)

    assert refreshed.quality_analysis is not None
    assert refreshed.quality_analysis.has_thinking_trace is True
    assert refreshed.quality_analysis.thinking_trace_incomplete is True
    assert refreshed.quality_analysis.has_reasoning_leak is False
    assert "thinking-incomplete" in (refreshed.quality_issues or "")
    assert "reasoning-leak" not in (refreshed.quality_issues or "")
