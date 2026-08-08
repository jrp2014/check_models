"""Tests for report generation edge cases (empty input, all-failed results)."""

from __future__ import annotations

import base64
import html
import io
import json
import logging
import re
from argparse import Namespace
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, get_args
from unittest.mock import patch

import pytest
from PIL import Image

import check_models
from check_models import (
    DiagnosticsArtifacts,
    GenerationQualityAnalysis,
    PerformanceResult,
    RuntimeDiagnostics,
    _build_report_render_context,
    _clean_stale_toplevel_reports,
    _generate_github_issue_reports,
    generate_diagnostics_report,
    generate_html_report,
    generate_markdown_gallery_report,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

type ExpectedUpstreamBoundary = Literal["not_started", "load_started", "generation_started"]

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


def _stub_versions() -> dict[str, str | None]:
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


def _issue_summary_output_paths(output_dir: Path) -> check_models.ReportOutputPaths:
    """Return canonical retained paths for aggregate issue-summary tests."""
    return check_models.ReportOutputPaths(
        index=output_dir / "index.md",
        html=output_dir / "reports" / "results.html",
        gallery_markdown=output_dir / "reports" / "model_gallery.md",
        jsonl=output_dir / "results.jsonl",
        run_json=output_dir / "run.json",
        diagnostics=output_dir / "reports" / "diagnostics.md",
        log=output_dir / "check_models.log",
        environment=output_dir / "environment.log",
    )


def _issue_summary_result(
    model: str,
    *,
    execution: str = "completed",
    usability: str = "usable",
    maintainer_status: str = "none",
    observations: list[str] | None = None,
    details: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build one literal schema-2.0 result without production serializers."""
    crashed = execution == "crashed"
    return {
        "_type": "result",
        "model": model,
        "timestamp": "2026-07-31 12:01:00 BST",
        "assessment": {
            "execution": execution,
            "usability": usability,
            "maintainer_status": maintainer_status,
            "observations": observations or [],
            **({"details": details} if details is not None else {}),
        },
        "generated_text": "generated output that must not be copied",
        "captured_output_on_fail": "captured output that must not be copied",
        "failure": (
            {
                "phase": "processor_load",
                "stage": "Processor Error",
                "message": "processor missing image support",
                "exception_type": "ValueError",
                "traceback": "Traceback (most recent call last):\nheavy evidence",
                "exception_chain": [
                    {
                        "type": "ValueError",
                        "module": "builtins",
                        "message": "processor missing image support",
                        "origin": "check_models.py",
                    }
                ],
            }
            if crashed
            else None
        ),
        "metrics": {},
        "timing": {},
        "model_provenance": {
            "model": model,
            "requested_revision": None,
            "resolved_revision": f"revision-{model.rsplit('/', maxsplit=1)[-1]}",
            "snapshot_path": None,
        },
        "prompt_diagnostics": None,
    }


def _write_issue_summary_fixture(
    output_paths: check_models.ReportOutputPaths,
    *,
    results: Sequence[dict[str, object]],
    image_source_url: str | None = None,
    image_sha256: str | None = "a" * 64,
    trust_remote_code: bool | None = False,
    total_runtime_seconds: float | None = None,
) -> None:
    """Write hand-authored retained input for issue-summary tests."""
    output_paths.jsonl.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "_type": "metadata",
        "format_version": "2.0",
        "prompt": "full prompt that must not be copied",
        "system": {
            "macOS Version": "26.6",
            "GPU/Chip": "Apple M5 Max",
            "Python Version": "3.13.13",
        },
        "timestamp": "2026-07-31 12:00:00 BST",
        "eval_mode": "assisted",
        "metadata_exposed_to_prompt": True,
        "library_versions": {
            "mlx-vlm": "0.6.8",
            "mlx": "0.32.1",
            "transformers": "5.14.1",
        },
        "component_provenance": {},
        "runtime_fingerprint": {},
    }
    rows = (metadata, *results)
    check_models._write_text_file(
        output_paths.jsonl,
        "".join(json.dumps(row) + "\n" for row in rows),
    )
    image: dict[str, object] = {
        "name": "fixture.jpg",
        "sha256": image_sha256,
        "size_bytes": 12_345,
        "width": 640,
        "height": 480,
        "megapixels": 0.3072,
    }
    if image_source_url is not None:
        image["source_url"] = image_source_url
    check_models._write_text_file(
        output_paths.run_json,
        json.dumps(
            {
                "generated_at": "2026-07-31 12:02:00 BST",
                **(
                    {"total_runtime_seconds": total_runtime_seconds}
                    if total_runtime_seconds is not None
                    else {}
                ),
                "eval_mode": "assisted",
                "generation_settings": {"max_tokens": 500, "temperature": 0.0},
                "image": image,
                **(
                    {"trust_remote_code": trust_remote_code}
                    if trust_remote_code is not None
                    else {}
                ),
                "producer": {
                    "name": "check_models",
                    "version": "0.8.9",
                    "git_revision": "abc123",
                    "install_type": "source-tree",
                    "dirty": False,
                },
            }
        )
        + "\n",
    )


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


def test_human_observation_labels_are_readable_and_severity_ordered() -> None:
    labels = check_models._human_observation_labels(
        (
            "no_keyword_overlap",
            "missing_requested_sections",
            "unexpected_special_token",
            "repeated_output",
        ),
        details={"missing_sections": ["title", "keywords"]},
    )

    assert labels == (
        "Response repeats the same text; "
        "Unrecognised model control tokens remain visible; "
        "Missing or empty fields: Title, Keywords; "
        "Keywords do not overlap the supplied keyword hints"
    )


def test_human_observation_labels_cover_every_stable_code() -> None:
    all_codes = get_args(check_models.ObservationCode.__value__)

    labels = check_models._human_observation_labels(all_codes)

    assert labels.count("; ") == len(all_codes) - 1
    assert "_" not in labels
    assert "catalogue instructions" not in labels.casefold()


def test_run_issue_summary_expands_crash_and_tables_other_findings(tmp_path: Path) -> None:
    """A paste-ready issue should prioritize crashes without copying heavy evidence."""
    output_paths = _issue_summary_output_paths(tmp_path / "output")
    _write_issue_summary_fixture(
        output_paths,
        results=(
            _issue_summary_result(
                "org/crash",
                execution="crashed",
                usability="not_evaluated",
                maintainer_status="actionable_failure",
            ),
            _issue_summary_result(
                "org/observed",
                usability="unusable",
                maintainer_status="observation_needs_reproduction",
                observations=["missing_requested_sections", "repeated_output"],
                details={"missing_sections": ["title", "keywords"]},
            ),
            _issue_summary_result(
                "org/crashed-observed",
                execution="crashed",
                usability="not_evaluated",
                maintainer_status="observation_needs_reproduction",
                observations=["unexpected_special_token"],
            ),
            _issue_summary_result(
                "org/indeterminate",
                execution="indeterminate",
                usability="not_evaluated",
                observations=["empty_output"],
            ),
            _issue_summary_result("org/clean"),
        ),
    )
    issue_draft = output_paths.index.parent / "issues" / "issue_org_crash.md"
    issue_draft.parent.mkdir(parents=True, exist_ok=True)
    check_models._write_text_file(issue_draft, "# Exact crash draft\n")

    with patch.object(check_models._LinkStyleState, "value", "relative"):
        summary = check_models.generate_run_issue_summary_report(
            output_paths,
            issue_reports={"org/crash": issue_draft},
        )

    assert summary == output_paths.index.parent / "issues" / "run_summary.md"
    if summary is None:
        pytest.fail("surfaced results must produce a run issue summary")
    content = summary.read_text(encoding="utf-8")
    assert content.startswith(
        "# mlx-vlm compatibility findings across 5 cached vision-language models\n"
    )
    assert "## Run summary" in content
    assert "mechanical facts from one image" in content
    assert "## Crashes requiring action" in content
    assert "### org/crash" in content
    assert "processor_load" in content
    assert "ValueError: processor missing image support" in content
    assert "The original local input is not published" in content
    assert "JPEG" in content
    assert "640 x 480" in content
    assert "12,345 bytes" in content
    assert "a" * 64 in content
    assert "full prompt that must not be copied" in content
    assert "reproduce.py" not in content
    assert "prompt.txt" not in content
    assert "--image fixture.jpg" not in content
    assert "## Completed attempts requiring review" in content
    assert "## Crashed attempts requiring review" in content
    assert "## Indeterminate attempts requiring review" in content
    assert content.count("| Model | Usability | Observed result | Evidence |") == 3
    assert "| Model | Execution / usability | Observations | Full evidence |" not in content
    assert "## Observation clusters" in content
    # Clusters group by observation codes only (no per-model detail expansion).
    assert (
        "| Response repeats the same text; Required fields are missing or empty | 1 |"
    ) in content
    assert (
        "| org/observed | unusable | Response repeats the same text; "
        "Missing or empty fields: Title, Keywords |"
    ) in content
    crashed_table = _extract_markdown_subsection(
        content,
        "## Crashed attempts requiring review",
        end_headings=("## Indeterminate attempts requiring review",),
    )
    assert "org/crashed-observed" in crashed_table
    assert "| org/crash |" not in crashed_table
    link_targets = _extract_markdown_link_targets(content)
    assert link_targets
    blob_prefix = (
        "https://github.com/jrp2014/check_models/blob/"
        f"{check_models._github_blob_ref()}/src/output/"
    )
    assert all(target.startswith(blob_prefix) for target in link_targets)
    assert (
        "1 clean completion; see the full model gallery (model_gallery.md, producer-local)."
        in content
    )
    assert "Trust remote code" in content
    assert "check_models" in content
    assert "0.8.9" in content
    assert "abc123" in content
    assert "GitHub links" in content
    # The link caveat is dynamic: pinned wording for a clean-worktree SHA ref,
    # mutable-branch wording otherwise.
    if re.fullmatch(r"[0-9a-f]{40}", check_models._github_blob_ref()):
        assert "pinned to producer commit" in content
    else:
        assert "mutable" in content
    assert "org/clean" not in content
    assert "Traceback (most recent call last)" not in content
    assert "generated output that must not be copied" not in content


def test_run_issue_summary_uses_failure_reason_when_observations_are_empty(
    tmp_path: Path,
) -> None:
    """An indeterminate row should say what prevented evaluation."""
    output_paths = _issue_summary_output_paths(tmp_path / "output")
    result = _issue_summary_result(
        "org/network",
        execution="indeterminate",
        usability="not_evaluated",
    )
    result["failure"] = {
        "phase": "model_load",
        "stage": "Network Error",
        "code": "UNKNOWN_MODEL_LOAD_NETWORK_ERROR",
        "message": "Model loading failed: [Errno 54] Connection reset by peer",
        "exception_type": "ReadError",
        "exception_module": "httpx",
        "package": "unknown",
        "traceback": "heavy evidence",
        "exception_chain": [],
    }
    _write_issue_summary_fixture(output_paths, results=(result,))

    summary = check_models.generate_run_issue_summary_report(output_paths)

    if summary is None:
        pytest.fail("the indeterminate result must produce a summary")
    content = summary.read_text(encoding="utf-8")
    assert "Network connection reset during model loading" in content
    assert "| org/network | not evaluated | none |" not in content


def test_run_issue_summary_sorts_review_rows_by_observation_severity(tmp_path: Path) -> None:
    """Grossly unusable outputs should be visible before repairable caveats."""
    output_paths = _issue_summary_output_paths(tmp_path / "output")
    _write_issue_summary_fixture(
        output_paths,
        results=(
            _issue_summary_result(
                "org/count-caveat",
                usability="usable_with_caveats",
                maintainer_status="observation_needs_reproduction",
                observations=["catalog_constraint_violation"],
                details={"title_word_count": 4, "title_word_range": [5, 10]},
            ),
            _issue_summary_result(
                "org/missing",
                usability="unusable",
                maintainer_status="observation_needs_reproduction",
                observations=["missing_requested_sections"],
            ),
            _issue_summary_result(
                "org/repeated",
                usability="unusable",
                maintainer_status="observation_needs_reproduction",
                observations=["repeated_output"],
            ),
        ),
    )

    summary = check_models.generate_run_issue_summary_report(output_paths)

    if summary is None:
        pytest.fail("the observed results must produce a summary")
    content = summary.read_text(encoding="utf-8")
    assert content.index("| org/repeated |") < content.index("| org/missing |")
    assert content.index("| org/missing |") < content.index("| org/count-caveat |")
    assert "Title has 4 words (requested 5-10)" in content


def test_diagnostics_sorts_triage_and_evidence_by_actionability(tmp_path: Path) -> None:
    """Grossly unusable output should precede less severe observations everywhere."""
    minimal = PerformanceResult(
        model_name="org/a-minimal",
        success=True,
        generation=_MockGeneration(text="Cat", generation_tokens=1),
    )
    repeated = PerformanceResult(
        model_name="org/z-repeated",
        success=True,
        generation=_MockGeneration(text="word " * 100, generation_tokens=100),
    )
    results = [minimal, repeated]
    context = _build_report_render_context(
        results=results,
        prompt="Describe this image.",
        system_info={},
    )
    output = tmp_path / "diagnostics.md"

    generate_diagnostics_report(
        results,
        output,
        prompt="Describe this image.",
        library_versions=_stub_versions(),
        system_info={},
        report_context=context,
    )

    content = output.read_text(encoding="utf-8")
    triage = _extract_markdown_subsection(
        content,
        "## Triage",
        end_headings=("## Crashes requiring action",),
    )
    observations = _extract_markdown_subsection(
        content,
        "## Completed Runs with Observations",
        end_headings=("## Indeterminate Attempts",),
    )
    assert triage.index("org/z-repeated") < triage.index("org/a-minimal")
    assert observations.index("org/z-repeated") < observations.index("org/a-minimal")


def test_run_issue_summary_builds_complete_public_image_reproduction(tmp_path: Path) -> None:
    """A public source should make the aggregate crash reproduction runnable."""
    output_paths = _issue_summary_output_paths(tmp_path / "output")
    result = _issue_summary_result(
        "org/crash",
        execution="crashed",
        usability="not_evaluated",
        maintainer_status="actionable_failure",
    )
    _write_issue_summary_fixture(
        output_paths,
        results=(result,),
        image_source_url="https://example.test/images/cats.jpg",
        trust_remote_code=True,
    )

    summary = check_models.generate_run_issue_summary_report(output_paths)

    if summary is None:
        pytest.fail("the crash must produce a run issue summary")
    content = summary.read_text(encoding="utf-8")
    assert "https://example.test/images/cats.jpg" in content
    assert "curl --fail --location" in content
    assert "set -euo pipefail\ncurl --fail --location" in content
    assert "shasum -a 256 --check" in content
    assert "python -m mlx_vlm.generate" in content
    assert "--model org/crash" in content
    assert "--revision revision-crash" in content
    assert "--prompt 'full prompt that must not be copied'" in content
    assert "--image repro-image.jpg" in content
    assert "--trust-remote-code" in content
    assert "reproduce.py" not in content
    assert "prompt.txt" not in content


def test_run_issue_summary_withholds_stale_log_and_environment_links(tmp_path: Path) -> None:
    """Issue-ready evidence must not attribute prior-run logs to the current run."""
    output_paths = _issue_summary_output_paths(tmp_path / "output")
    _write_issue_summary_fixture(
        output_paths,
        results=(
            _issue_summary_result(
                "org/observed",
                usability="usable_with_caveats",
                maintainer_status="observation_needs_reproduction",
                observations=["minimal_output"],
            ),
        ),
        total_runtime_seconds=120.0,
    )
    check_models._write_text_file(
        output_paths.log,
        "2026-07-31 11:00:00 BST - INFO - prior run\n",
    )
    check_models._write_text_file(
        output_paths.environment,
        "FULL ENVIRONMENT DUMP - 2026-07-31 11:00:00 BST\n",
    )

    summary = check_models.generate_run_issue_summary_report(output_paths)

    assert summary is not None
    content = summary.read_text(encoding="utf-8")
    assert "Stale retained artifacts omitted" in content
    assert "check_models.log" in content
    assert "environment.log" in content
    assert "src/output/check_models.log" not in content
    assert "src/output/environment.log" not in content
    assert "| Environment |" not in content
    assert "| Log |" not in content


def test_run_issue_summary_keeps_current_log_and_environment_links(tmp_path: Path) -> None:
    """Artifacts beginning inside the retained run window should remain linked."""
    output_paths = _issue_summary_output_paths(tmp_path / "output")
    _write_issue_summary_fixture(
        output_paths,
        results=(
            _issue_summary_result(
                "org/observed",
                usability="usable_with_caveats",
                maintainer_status="observation_needs_reproduction",
                observations=["minimal_output"],
            ),
        ),
        total_runtime_seconds=120.0,
    )
    check_models._write_text_file(
        output_paths.log,
        "2026-07-31 12:00:00 BST - INFO - current run\n",
    )
    check_models._write_text_file(
        output_paths.environment,
        "FULL ENVIRONMENT DUMP - 2026-07-31 12:00:00 BST\n",
    )

    summary = check_models.generate_run_issue_summary_report(output_paths)

    assert summary is not None
    content = summary.read_text(encoding="utf-8")
    assert "Stale retained artifacts omitted" not in content
    assert "check_models.log (producer-local, not published)" in content
    assert "src/output/environment.log" in content
    assert "| Environment |" in content
    assert "| Log |" in content


def test_run_issue_summary_compacts_large_unexpected_parameter_errors(tmp_path: Path) -> None:
    """Aggregate crash evidence should group repeated parameter paths."""
    output_paths = _issue_summary_output_paths(tmp_path / "output")
    result = _issue_summary_result(
        "org/crash",
        execution="crashed",
        usability="not_evaluated",
        maintainer_status="actionable_failure",
    )
    parameters = [
        "audio_tower.encoder.biases",
        "audio_tower.encoder.scales",
        *(
            f"language_model.model.layers.{layer}.mlp.experts.down_proj.weight"
            for layer in range(10)
        ),
    ]
    message = "Received 12 parameters not in model: \n" + ",\n".join(parameters) + "."
    failure = result["failure"]
    assert isinstance(failure, dict)
    failure["message"] = message
    failure["exception_chain"] = [
        {
            "type": "ValueError",
            "module": "builtins",
            "message": message,
            "origin": "check_models.py",
        },
        {
            "type": "ValueError",
            "module": "builtins",
            "message": f"Model loading failed: {message}",
            "origin": "check_models.py",
        },
    ]
    _write_issue_summary_fixture(output_paths, results=(result,))

    summary = check_models.generate_run_issue_summary_report(output_paths)

    assert summary is not None
    content = summary.read_text(encoding="utf-8")
    assert "ValueError: Received 12 parameters not in model" in content
    assert "audio_tower" in content
    assert "language_model" in content
    assert parameters[0] in content
    assert parameters[-1] not in content
    assert "model evidence" in content


def test_diagnostics_and_issue_draft_compact_large_unexpected_parameter_errors(
    tmp_path: Path,
) -> None:
    """Maintainer paste surfaces should compact large unexpected-parameter lists."""
    parameters = [
        "audio_tower.encoder.biases",
        "audio_tower.encoder.scales",
        *(
            f"language_model.model.layers.{layer}.mlp.experts.down_proj.weight"
            for layer in range(10)
        ),
    ]
    message = "Received 12 parameters not in model: \n" + ",\n".join(parameters) + "."
    crash = PerformanceResult(
        model_name="org/param-mismatch",
        generation=None,
        success=False,
        failure_phase="model_load",
        error_stage="Model Error",
        error_type="ValueError",
        error_message=message,
        root_error_type="ValueError",
        root_error_module="builtins",
        root_error_message=message,
        exception_chain=(check_models.FailureException("ValueError", "builtins", message),),
        error_package="mlx-vlm",
        error_traceback=f"Traceback (most recent call last):\nValueError: {message}",
    )
    context = _build_report_render_context(
        results=[crash],
        prompt="Describe the image.",
        system_info={"Python Version": "3.13.13"},
    )
    diagnostics = tmp_path / "diagnostics.md"
    generate_diagnostics_report(
        [crash],
        diagnostics,
        prompt="Describe the image.",
        library_versions=_stub_versions(),
        system_info={"Python Version": "3.13.13"},
        report_context=context,
    )
    generated = _generate_github_issue_reports(
        report_context=context,
        output_dir=tmp_path,
        library_versions=_stub_versions(),
        system_info={"Python Version": "3.13.13"},
        prompt="Describe the image.",
    )

    diagnostics_content = diagnostics.read_text(encoding="utf-8")
    issue_content = next(iter(generated.values())).read_text(encoding="utf-8")
    for content in (diagnostics_content, issue_content):
        assert "Received 12 parameters not in model" in content
        assert "families: audio_tower, language_model" in content
        assert "representative parameters:" in content
        # Compacted exception presentation keeps a short sample, not the full list.
        assert parameters[0] in content
    # Full traceback remains available for deep inspection.
    assert parameters[-1] in diagnostics_content
    assert parameters[-1] in issue_content


def test_github_blob_ref_uses_clean_producer_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clean producer revisions should pin GitHub artifact links to that commit."""
    monkeypatch.setattr(check_models, "_GITHUB_REF_OVERRIDE", None)
    monkeypatch.setattr(
        check_models,
        "_collect_check_models_provenance",
        lambda: {
            "name": "check_models",
            "version": "0.8.9",
            "git_revision": "deadbeefcafebabe",
            "install_type": "source-tree",
            "dirty": False,
        },
    )
    assert check_models._github_blob_ref() == "deadbeefcafebabe"
    monkeypatch.setattr(
        check_models,
        "_collect_check_models_provenance",
        lambda: {
            "name": "check_models",
            "version": "0.8.9",
            "git_revision": "deadbeefcafebabe",
            "install_type": "source-tree",
            "dirty": True,
        },
    )
    assert check_models._github_blob_ref() == check_models._GITHUB_DEFAULT_BRANCH


def test_observation_display_registry_covers_literal_codes() -> None:
    """Observation display metadata must stay aligned with ObservationCode."""
    codes = check_models._literal_values(check_models.ObservationCode)
    assert set(check_models._OBSERVATION_DISPLAY_BY_CODE) == codes
    assert codes == check_models._RUN_ISSUE_OBSERVATION_VALUES
    assert check_models._RUN_ISSUE_EXECUTION_VALUES == check_models._EXECUTION_STATUS_VALUES
    assert "empty_output" in check_models._UNUSABLE_OBSERVATIONS
    assert "thinking_trace_present" not in check_models._UNUSABLE_OBSERVATIONS
    assert (
        check_models._gallery_observation_labels(("token_cap_truncation", "repeated_output"))
        == "repeated text; cut off at token limit"
    )


@pytest.mark.parametrize("image_sha256", [None, "abc123"])
def test_run_issue_summary_withholds_command_without_valid_digest(
    tmp_path: Path,
    image_sha256: str | None,
) -> None:
    """A public URL alone must not be presented as a verified exact input."""
    output_paths = _issue_summary_output_paths(tmp_path / "output")
    result = _issue_summary_result(
        "org/crash",
        execution="crashed",
        usability="not_evaluated",
        maintainer_status="actionable_failure",
    )
    _write_issue_summary_fixture(
        output_paths,
        results=(result,),
        image_source_url="https://example.test/images/cats.jpg",
        image_sha256=image_sha256,
    )

    summary = check_models.generate_run_issue_summary_report(output_paths)

    if summary is None:
        pytest.fail("the crash must produce a run issue summary")
    content = summary.read_text(encoding="utf-8")
    assert "A valid SHA-256 digest is unavailable" in content
    assert "python -m mlx_vlm.generate" not in content
    assert "shasum -a 256 --check" not in content


@pytest.mark.parametrize(
    ("trust_remote_code", "expected_flag"),
    [(True, True), (False, False), (None, False)],
)
def test_run_issue_summary_preserves_remote_code_policy(
    tmp_path: Path,
    trust_remote_code: bool | None,
    expected_flag: bool,
) -> None:
    """A retained reproduction must not silently broaden remote-code trust."""
    output_paths = _issue_summary_output_paths(tmp_path / "output")
    result = _issue_summary_result(
        "org/crash",
        execution="crashed",
        usability="not_evaluated",
        maintainer_status="actionable_failure",
    )
    _write_issue_summary_fixture(
        output_paths,
        results=(result,),
        image_source_url="https://example.test/images/cats.jpg",
        trust_remote_code=trust_remote_code,
    )

    summary = check_models.generate_run_issue_summary_report(output_paths)

    if summary is None:
        pytest.fail("the crash must produce a run issue summary")
    content = summary.read_text(encoding="utf-8")
    assert ("--trust-remote-code" in content) is expected_flag


def test_run_issue_summary_uses_cached_assessment_without_reclassification(
    tmp_path: Path,
) -> None:
    """Report-only rendering must preserve cached schema-2.0 assessment values."""
    output_paths = _issue_summary_output_paths(tmp_path / "output")
    _write_issue_summary_fixture(
        output_paths,
        results=(
            _issue_summary_result(
                "org/cached",
                usability="usable_with_caveats",
                maintainer_status="observation_needs_reproduction",
                observations=["minimal_output"],
            ),
        ),
    )

    with (
        patch.object(
            check_models,
            "_assess_result",
            side_effect=AssertionError("cached assessment was reclassified"),
        ),
        patch.object(check_models._LinkStyleState, "value", "relative"),
    ):
        summary = check_models.generate_run_issue_summary_report(output_paths)

    assert summary is not None
    content = summary.read_text(encoding="utf-8")
    assert "## Completed attempts requiring review" in content
    assert "| org/cached | usable with caveats |" in content
    assert "Response is unusually short" in content


def test_run_issue_summary_repro_prefers_resolved_revision(tmp_path: Path) -> None:
    """Crash reproduction should pin the immutable resolved snapshot when available."""
    output_paths = _issue_summary_output_paths(tmp_path / "output")
    result = _issue_summary_result(
        "org/crash",
        execution="crashed",
        usability="not_evaluated",
        maintainer_status="actionable_failure",
    )
    provenance = result["model_provenance"]
    assert isinstance(provenance, dict)
    provenance["requested_revision"] = "moving-branch"
    provenance["resolved_revision"] = "immutable-commit"
    _write_issue_summary_fixture(
        output_paths,
        results=(result,),
        image_source_url="https://example.test/images/cats.jpg",
    )

    summary = check_models.generate_run_issue_summary_report(output_paths)

    if summary is None:
        pytest.fail("the crash must produce a run issue summary")
    content = summary.read_text(encoding="utf-8")
    assert "--revision immutable-commit" in content
    assert "--revision moving-branch" not in content


def test_run_issue_summary_removes_stale_artifact_for_clean_run(tmp_path: Path) -> None:
    """A run with no surfaced result should not leave an obsolete issue body."""
    output_paths = _issue_summary_output_paths(tmp_path / "output")
    _write_issue_summary_fixture(
        output_paths,
        results=(_issue_summary_result("org/clean"),),
    )
    stale = output_paths.index.parent / "issues" / "run_summary.md"
    stale.parent.mkdir(parents=True, exist_ok=True)
    check_models._write_text_file(stale, "stale issue\n")

    assert check_models.generate_run_issue_summary_report(output_paths) is None
    assert not stale.exists()


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        ((), "Missing JSONL metadata"),
        (({"_type": "metadata", "format_version": "1.0"},), "format_version 2.0"),
        (
            (
                {
                    "_type": "metadata",
                    "format_version": "2.0",
                    "prompt": "prompt",
                    "system": {},
                    "timestamp": "now",
                },
                {"_type": "result", "model": "org/model"},
            ),
            "cached assessment",
        ),
    ],
)
def test_run_issue_summary_rejects_invalid_jsonl_contract(
    tmp_path: Path,
    rows: tuple[dict[str, object], ...],
    expected: str,
) -> None:
    """Missing metadata, wrong schemas, and missing assessments must fail clearly."""
    output_paths = _issue_summary_output_paths(tmp_path / "output")
    output_paths.jsonl.parent.mkdir(parents=True, exist_ok=True)
    check_models._write_text_file(
        output_paths.jsonl,
        "".join(json.dumps(row) + "\n" for row in rows),
    )

    with pytest.raises(ValueError, match=expected):
        check_models.generate_run_issue_summary_report(output_paths)


@pytest.mark.parametrize(
    ("replacement", "expected"),
    [
        ({"model_provenance": "not a mapping"}, "model provenance"),
        ({"model_provenance": {"model": "org/model", "resolved_revision": []}}, "revision"),
        (
            {
                "model_provenance": {
                    "model": "org/different-model",
                    "requested_revision": None,
                    "resolved_revision": "commit",
                }
            },
            "does not match",
        ),
        ({"failure": "not a mapping"}, "failure"),
        (
            {
                "failure": {
                    "phase": "model_load",
                    "exception_chain": ["not a mapping"],
                }
            },
            "exception chain",
        ),
        (
            {
                "assessment": {
                    "execution": [],
                    "usability": "unusable",
                    "maintainer_status": "observation_needs_reproduction",
                    "observations": [],
                }
            },
            "cached assessment",
        ),
    ],
)
def test_run_issue_summary_rejects_malformed_consumed_result_structures(
    tmp_path: Path,
    replacement: dict[str, object],
    expected: str,
) -> None:
    """Every retained structure dereferenced by the renderer must be validated."""
    output_paths = _issue_summary_output_paths(tmp_path / "output")
    result = _issue_summary_result(
        "org/model",
        execution="crashed",
        usability="not_evaluated",
        maintainer_status="actionable_failure",
    )
    result.update(replacement)
    _write_issue_summary_fixture(output_paths, results=(result,))

    with pytest.raises(ValueError, match=expected):
        check_models.generate_run_issue_summary_report(output_paths)


def test_run_issue_summary_ignores_malformed_optional_run_json(tmp_path: Path) -> None:
    """Malformed optional enrichment must not block rendering cached assessments."""
    output_paths = _issue_summary_output_paths(tmp_path / "output")
    _write_issue_summary_fixture(
        output_paths,
        results=(
            _issue_summary_result(
                "org/observed",
                maintainer_status="observation_needs_reproduction",
                observations=["minimal_output"],
            ),
        ),
    )
    check_models._write_text_file(output_paths.run_json, "{not json\n")

    summary = check_models.generate_run_issue_summary_report(output_paths)

    assert summary is not None
    assert "org/observed" in summary.read_text(encoding="utf-8")


def test_regenerate_run_issue_summary_only_writes_derived_artifact(tmp_path: Path) -> None:
    """Report-only regeneration must leave every retained source byte-identical."""
    output_paths = _issue_summary_output_paths(tmp_path / "output")
    _write_issue_summary_fixture(
        output_paths,
        results=(
            _issue_summary_result(
                "org/crash",
                execution="crashed",
                usability="not_evaluated",
                maintainer_status="actionable_failure",
            ),
        ),
    )
    issue_draft = output_paths.index.parent / "issues" / "issue_org_crash.md"
    issue_draft.parent.mkdir(parents=True, exist_ok=True)
    check_models._write_text_file(issue_draft, "# Existing crash draft\n")
    retained = {
        path: path.read_bytes() for path in (output_paths.jsonl, output_paths.run_json, issue_draft)
    }

    with patch.object(check_models._LinkStyleState, "value", "relative"):
        generated = check_models.regenerate_run_issue_summary(output_paths.index.parent)

    assert generated == output_paths.index.parent / "issues" / "run_summary.md"
    if generated is None:
        pytest.fail("the actionable retained run must regenerate an issue summary")
    assert {path: path.read_bytes() for path in retained} == retained
    crash_draft_url = (
        "https://github.com/jrp2014/check_models/blob/"
        f"{check_models._github_blob_ref()}/src/output/issues/issue_org_crash.md"
    )
    assert f"[crash draft]({crash_draft_url})" in generated.read_text(encoding="utf-8")


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
    gallery_path = tmp_path / "gallery.md"

    generate_html_report(
        [result],
        html_path,
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
    assert "*Peak memory:* 1.0" in gallery_path.read_text(encoding="utf-8")


def test_markdown_gallery_publishes_reference_image_beside_report(tmp_path: Path) -> None:
    image_path = tmp_path / "input.jpg"
    Image.new("RGB", (2048, 1024), color="purple").save(image_path)
    gallery_path = tmp_path / "reports" / "model_gallery.md"
    result = _make_success("org/model")
    context = _build_report_render_context(results=[result], prompt="Describe the image.")

    generate_markdown_gallery_report(
        [result],
        gallery_path,
        "Describe the image.",
        report_context=context,
        image_path=image_path,
    )

    assert "![Reference image](assets/source-image.jpg)" in gallery_path.read_text(encoding="utf-8")
    with Image.open(gallery_path.parent / "assets" / "source-image.jpg") as preview:
        assert preview.size == (1024, 512)


def test_markdown_gallery_does_not_follow_reference_asset_symlink(tmp_path: Path) -> None:
    image_path = tmp_path / "input.jpg"
    Image.new("RGB", (16, 8), color="purple").save(image_path)
    gallery_path = tmp_path / "reports" / "model_gallery.md"
    asset = gallery_path.parent / "assets" / "source-image.jpg"
    asset.parent.mkdir(parents=True)
    victim = tmp_path / "victim.jpg"
    victim.write_bytes(b"keep-me")
    asset.symlink_to(victim)
    result = _make_success("org/model")

    generate_markdown_gallery_report(
        [result],
        gallery_path,
        "Describe the image.",
        report_context=_build_report_render_context(
            results=[result],
            prompt="Describe the image.",
        ),
        image_path=image_path,
    )

    assert victim.read_bytes() == b"keep-me"
    assert "![Reference image]" not in gallery_path.read_text(encoding="utf-8")


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


def _extract_markdown_model_section(content: str, model_name: str) -> str:
    """Return one model's heading-scoped section without crossing into another model."""
    match = re.search(
        rf"(?ms)^### {re.escape(model_name)}\n.*?(?=^### |^## |\Z)",
        content,
    )
    assert match is not None, f"Missing Markdown section for {model_name}"
    return match.group(0)


def _extract_markdown_diagnostic_entry(content: str, model_name: str) -> str:
    """Return one diagnostics entry through its triage-table evidence link."""
    link = re.search(rf"\[{re.escape(model_name)}\]\(#([^)]+)\)", content)
    assert link is not None, f"Missing diagnostics link for {model_name}"
    marker = f'<a id="{link.group(1)}"></a>'
    start = content.index(marker)
    tail = content[start + len(marker) :]
    boundaries = [
        index for token in ('<a id="diagnostic-', "\n## ") if (index := tail.find(token)) >= 0
    ]
    end = start + len(marker) + min(boundaries) if boundaries else len(content)
    return content[start:end]


_GENERATED_STAMP_EMPHASIS_HEADING_RE = re.compile(
    r"(?m)^_(?:Generated on|Report generated on).+_$",
)
_MARKDOWN_LINK_TARGET_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
_URL_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.-]*:", re.IGNORECASE)
_PUBLISHED_OUTPUT_GITHUB_TARGET_RE = re.compile(
    rf"^{re.escape(check_models._GITHUB_REPO_URL)}/(?:blob|tree)/"
    rf"(?:{re.escape(check_models._GITHUB_DEFAULT_BRANCH)}|[0-9a-f]{{7,40}})/"
    r"src/output(?:/|$)"
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


def test_custom_published_index_and_issue_drafts_use_distinct_repo_paths(
    tmp_path: Path,
) -> None:
    """A retained index must publish at the output root while drafts stay under issues."""
    custom_index = tmp_path / "custom-run" / "index.md"
    issue_draft = tmp_path / "custom-run" / "issues" / "issue_org_model.md"

    index_path = check_models._published_output_repo_path(custom_index)
    issue_path = check_models._published_output_repo_path(issue_draft)

    assert index_path is not None
    assert issue_path is not None
    assert index_path.as_posix() == "src/output/index.md"
    assert issue_path.as_posix() == "src/output/issues/issue_org_model.md"


def test_output_index_links_only_current_run_artifacts(tmp_path: Path) -> None:
    """The tiny index should link current evidence, not history or retired reports."""
    output_dir = tmp_path / "output"
    output_paths = check_models.ReportOutputPaths(
        index=output_dir / "index.md",
        html=output_dir / "reports" / "results.html",
        gallery_markdown=output_dir / "reports" / "model_gallery.md",
        jsonl=output_dir / "results.jsonl",
        run_json=output_dir / "run.json",
        diagnostics=output_dir / "reports" / "diagnostics.md",
        log=output_dir / "check_models.log",
        environment=output_dir / "environment.log",
    )

    with patch.object(check_models._LinkStyleState, "value", "relative"):
        check_models.generate_output_index_report(output_paths.index, output_paths=output_paths)

    assert output_paths.index.read_text(encoding="utf-8") == (
        "# Check Models Output Index\n"
        "\n"
        "- [results.html](reports/results.html) (local only, not tracked)\n"
        "- [model_gallery.md](reports/model_gallery.md) (local only, not tracked)\n"
        "- [diagnostics.md](reports/diagnostics.md)\n"
        "- [results.jsonl](results.jsonl)\n"
        "- [run.json](run.json)\n"
        "- [check_models.log](check_models.log) (local only, not tracked)\n"
        "- [environment.log](environment.log)\n"
    )


def test_local_only_artifacts_never_publish_github_paths(tmp_path: Path) -> None:
    """Untracked bulky artifacts must resolve to relative links, never GitHub URLs."""
    for name in (
        "reports/results.html",
        "reports/model_gallery.md",
        "check_models.log",
        "results.history.jsonl",
    ):
        assert check_models._published_output_repo_path(tmp_path / name) is None
        # Location-based inference must not resurrect a GitHub path either.
        repo_local = check_models._REPO_ROOT / "src" / "output" / name
        assert check_models._published_output_repo_path(repo_local) is None

    tracked = check_models._REPO_ROOT / "src" / "output" / "reports" / "diagnostics.md"
    tracked_path = check_models._published_output_repo_path(tracked)
    assert tracked_path is not None
    assert tracked_path.as_posix() == "src/output/reports/diagnostics.md"


def test_output_index_links_current_run_issue_drafts_in_model_order(tmp_path: Path) -> None:
    """The output index should expose only the issue paths produced for this run."""
    output_dir = tmp_path / "output"
    output_paths = check_models.ReportOutputPaths(
        index=output_dir / "index.md",
        html=output_dir / "reports" / "results.html",
        gallery_markdown=output_dir / "reports" / "model_gallery.md",
        jsonl=output_dir / "results.jsonl",
        run_json=output_dir / "run.json",
        diagnostics=output_dir / "reports" / "diagnostics.md",
        log=output_dir / "check_models.log",
        environment=output_dir / "environment.log",
    )
    issue_reports = {
        "org/z": output_dir / "issues" / "issue_org_z.md",
        "org/a": output_dir / "issues" / "issue_org_a.md",
    }

    with patch.object(check_models._LinkStyleState, "value", "relative"):
        check_models.generate_output_index_report(
            output_paths.index,
            output_paths=output_paths,
            issue_reports=issue_reports,
        )

    content = output_paths.index.read_text(encoding="utf-8")
    assert "## Issue drafts" in content
    assert "[org/a](issues/issue_org_a.md)" in content
    assert "[org/z](issues/issue_org_z.md)" in content
    assert content.index("[org/a]") < content.index("[org/z]")


def test_output_index_renders_run_dashboard(tmp_path: Path) -> None:
    """The index should lead with run counts, usability, and top observations."""
    output_dir = tmp_path / "output"
    output_paths = check_models.ReportOutputPaths(
        index=output_dir / "index.md",
        html=output_dir / "reports" / "results.html",
        gallery_markdown=output_dir / "reports" / "model_gallery.md",
        jsonl=output_dir / "results.jsonl",
        run_json=output_dir / "run.json",
        diagnostics=output_dir / "reports" / "diagnostics.md",
        log=output_dir / "check_models.log",
        environment=output_dir / "environment.log",
    )
    assessments = (
        ("org/good", check_models.ResultAssessment("completed", "usable", "none", ())),
        (
            "org/warn",
            check_models.ResultAssessment(
                "completed",
                "usable_with_caveats",
                "observation_needs_reproduction",
                ("minimal_output",),
            ),
        ),
        (
            "org/crash",
            check_models.ResultAssessment("crashed", "not_evaluated", "actionable_failure", ()),
        ),
    )

    with patch.object(check_models._LinkStyleState, "value", "relative"):
        check_models.generate_output_index_report(
            output_paths.index,
            output_paths=output_paths,
            assessments=assessments,
        )

    content = output_paths.index.read_text(encoding="utf-8")
    assert "## Run at a glance" in content
    assert "- Models attempted: 3 (completed 2, crashed 1, indeterminate 0)" in content
    assert "- Usability: usable 1, usable with caveats 1, unusable 0, not evaluated 1" in content
    minimal_label = check_models._OBSERVATION_DISPLAY_LABELS["minimal_output"]
    assert f"- Top observations: {minimal_label} (1)" in content
    assert "## Artifacts" in content
    assert content.index("## Run at a glance") < content.index("## Artifacts")


def test_run_issue_summary_link_caveat_reflects_blob_ref(tmp_path: Path) -> None:
    """The link caveat must say pinned for SHA refs and mutable for branch refs."""
    output_paths = _issue_summary_output_paths(tmp_path / "output")
    _write_issue_summary_fixture(
        output_paths,
        results=(
            _issue_summary_result(
                "org/observed",
                usability="usable_with_caveats",
                maintainer_status="observation_needs_reproduction",
                observations=["minimal_output"],
            ),
        ),
        total_runtime_seconds=120.0,
    )

    pinned_sha = "a" * 40
    with patch.object(check_models, "_GITHUB_REF_OVERRIDE", pinned_sha):
        summary = check_models.generate_run_issue_summary_report(output_paths)
    assert summary is not None
    pinned_content = summary.read_text(encoding="utf-8")
    assert f"pinned to producer commit `{pinned_sha[:12]}`" in pinned_content
    assert "mutable" not in pinned_content

    with patch.object(check_models, "_GITHUB_REF_OVERRIDE", "main"):
        summary = check_models.generate_run_issue_summary_report(output_paths)
    assert summary is not None
    branch_content = summary.read_text(encoding="utf-8")
    assert "mutable main branch" in branch_content
    assert "pinned to producer commit" not in branch_content


def _report_outcome(
    outcomes: Sequence[check_models.ReportArtifactOutcome],
    key: str,
) -> check_models.ReportArtifactOutcome:
    """Return one named report outcome for concise orchestration assertions."""
    return next(outcome for outcome in outcomes if outcome.key == key)


def test_report_orchestration_passes_generated_issue_drafts_to_index(tmp_path: Path) -> None:
    """Final report orchestration should index the drafts generated by diagnostics."""
    args = Namespace(
        output_html=tmp_path / "output" / "reports" / "results.html",
        output_gallery_markdown=tmp_path / "output" / "reports" / "model_gallery.md",
        output_jsonl=tmp_path / "output" / "results.jsonl",
        output_run_json=tmp_path / "output" / "run.json",
        output_diagnostics=tmp_path / "output" / "reports" / "diagnostics.md",
        output_log=tmp_path / "output" / "check_models.log",
        output_env=tmp_path / "output" / "environment.log",
    )
    result = _make_failure_with_details(
        "org/broken",
        error_msg="Model loading failed: boom",
        failure_phase="model_load",
        traceback_str="Traceback (most recent call last):\nValueError: boom",
    )
    context = _build_report_render_context(results=[result], prompt="Describe the image.")
    output_paths = check_models._resolve_report_output_paths(args)
    inputs = check_models.ReportGenerationInputs(
        results=[result],
        library_versions=_stub_versions(),
        prompt="Describe the image.",
        metadata=None,
        overall_time=1.0,
        image_path=None,
        system_info={},
        report_context=context,
        output_paths=output_paths,
        run_args=args,
        runtime_fingerprint={},
    )

    with patch.object(check_models._LinkStyleState, "value", "relative"):
        outcomes = check_models._generate_reports_and_log_outputs(inputs)

    index_content = output_paths.index.read_text(encoding="utf-8")
    summary_path = output_paths.index.parent / "issues" / "run_summary.md"
    assert "[Run issue summary](issues/run_summary.md)" in index_content
    assert "[org/broken](issues/issue_org_broken.md)" in index_content
    assert index_content.index("[Run issue summary]") < index_content.index("[org/broken]")
    assert _report_outcome(outcomes, "run_issue_summary").succeeded

    with patch.object(
        check_models,
        "generate_run_issue_summary_report",
        side_effect=ValueError("summary fixture failure"),
    ):
        failed_summary_outcomes = check_models._generate_reports_and_log_outputs(inputs)

    failed_summary = _report_outcome(failed_summary_outcomes, "run_issue_summary")
    assert not failed_summary.succeeded
    assert failed_summary.error_message == "summary fixture failure"
    assert all(
        path.exists()
        for path in (
            output_paths.index,
            output_paths.html,
            output_paths.gallery_markdown,
            output_paths.jsonl,
            output_paths.run_json,
            output_paths.diagnostics,
        )
    )
    assert not summary_path.exists()

    check_models._write_text_file(summary_path, "stale prior-run summary\n")
    with (
        patch.object(
            check_models,
            "save_jsonl_report",
            side_effect=OSError("current JSONL write failed"),
        ),
        patch.object(
            check_models,
            "generate_run_issue_summary_report",
            side_effect=AssertionError("summary must not read stale JSONL"),
        ),
    ):
        stale_jsonl_outcomes = check_models._generate_reports_and_log_outputs(inputs)

    assert not summary_path.exists()
    assert not _report_outcome(stale_jsonl_outcomes, "jsonl").succeeded

    check_models._write_text_file(summary_path, "undeletable stale summary\n")
    cleanup_error = PermissionError("summary cleanup denied")
    with (
        patch.object(
            check_models,
            "save_jsonl_report",
            side_effect=OSError("current JSONL write failed"),
        ),
        patch.object(check_models, "_remove_run_issue_summary", return_value=cleanup_error),
    ):
        cleanup_failure_outcomes = check_models._generate_reports_and_log_outputs(inputs)

    assert summary_path.exists()
    assert "Run issue summary" not in output_paths.index.read_text(encoding="utf-8")
    cleanup_failure = _report_outcome(cleanup_failure_outcomes, "run_issue_summary")
    assert not cleanup_failure.succeeded
    assert "cleanup denied" in (cleanup_failure.error_message or "")

    check_models._write_text_file(summary_path, "stale prior-run summary\n")
    check_models._write_text_file(output_paths.diagnostics, "stale prior-run diagnostics\n")
    with (
        patch.object(
            check_models,
            "_write_diagnostics_artifacts",
            side_effect=OSError("current diagnostics write failed"),
        ),
        patch.object(
            check_models,
            "generate_run_issue_summary_report",
            side_effect=AssertionError("summary must not link stale diagnostics"),
        ),
    ):
        stale_diagnostics_outcomes = check_models._generate_reports_and_log_outputs(inputs)

    assert not summary_path.exists()
    assert "Run issue summary" not in output_paths.index.read_text(encoding="utf-8")
    assert not _report_outcome(stale_diagnostics_outcomes, "diagnostics").succeeded
    stale_diagnostics_summary = _report_outcome(
        stale_diagnostics_outcomes,
        "run_issue_summary",
    )
    assert not stale_diagnostics_summary.succeeded
    assert "diagnostics" in (stale_diagnostics_summary.error_message or "").lower()

    check_models._write_text_file(
        output_paths.run_json,
        json.dumps(
            {
                "image": {"name": "stale-prior-run.jpg"},
                "generation_settings": {"max_tokens": 9999},
            }
        ),
    )
    with patch.object(
        check_models,
        "save_run_json_report",
        side_effect=OSError("current run JSON write failed"),
    ):
        stale_run_json_outcomes = check_models._generate_reports_and_log_outputs(inputs)

    assert not _report_outcome(stale_run_json_outcomes, "run_json").succeeded
    stale_run_json_summary = summary_path.read_text(encoding="utf-8")
    assert "stale-prior-run.jpg" not in stale_run_json_summary
    assert "run.json" not in stale_run_json_summary

    check_models._write_text_file(
        output_paths.gallery_markdown,
        "stale prior-run gallery\n",
    )
    with patch.object(
        check_models,
        "generate_markdown_gallery_report",
        side_effect=OSError("current gallery write failed"),
    ):
        stale_gallery_outcomes = check_models._generate_reports_and_log_outputs(inputs)

    stale_gallery_summary = summary_path.read_text(encoding="utf-8")
    assert "model_gallery.md" not in stale_gallery_summary
    assert "full model gallery" not in stale_gallery_summary
    assert not _report_outcome(stale_gallery_outcomes, "markdown_gallery").succeeded
    assert _report_outcome(stale_gallery_outcomes, "run_issue_summary").succeeded


def test_report_dashboard_only_shows_current_successful_run_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An existing stale summary must stay hidden unless this run produced it."""
    output_paths = _issue_summary_output_paths(tmp_path / "output")
    stale_summary = output_paths.index.parent / "issues" / "run_summary.md"
    check_models._write_text_file(stale_summary, "stale\n")

    check_models._print_reports_dashboard(output_paths, run_issue_summary=None)
    without_summary = capsys.readouterr().err
    check_models._print_reports_dashboard(
        output_paths,
        run_issue_summary=stale_summary,
    )
    with_summary = capsys.readouterr().err

    assert "Run Issue Summary" not in without_summary
    assert "Run Issue Summary" in with_summary


def _relative_output_artifact_map(
    output_dir: Path,
    output_paths: check_models.ReportOutputPaths,
) -> dict[str, str]:
    """Return the retained run-json artifact map rooted at one output directory."""
    return {
        "output_index": output_paths.index.relative_to(output_dir).as_posix(),
        "results_html": output_paths.html.relative_to(output_dir).as_posix(),
        "model_gallery": output_paths.gallery_markdown.relative_to(output_dir).as_posix(),
        "diagnostics": output_paths.diagnostics.relative_to(output_dir).as_posix(),
        "results_jsonl": output_paths.jsonl.relative_to(output_dir).as_posix(),
        "run_json": output_paths.run_json.relative_to(output_dir).as_posix(),
        "log": output_paths.log.relative_to(output_dir).as_posix(),
        "environment": output_paths.environment.relative_to(output_dir).as_posix(),
    }


def _generate_output_artifacts_for_link_style(
    tmp_path: Path,
    *,
    link_style: str,
) -> tuple[Path, check_models.ReportOutputPaths, list[Path]]:
    """Generate the retained artifact set for one link style."""
    output_dir = tmp_path / link_style / "output"
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    prompt = "Describe this image briefly."
    versions = _stub_versions()
    system_info = {"Python Version": "3.13"}
    results = [
        _make_success("org/good"),
        _make_failure_with_details(
            "org/broken",
            error_msg="Model loading failed: boom",
            failure_phase="model_load",
            traceback_str="Traceback (most recent call last):\nValueError: boom",
        ),
    ]
    report_context = _build_report_render_context(results=results, prompt=prompt)
    output_paths = check_models.ReportOutputPaths(
        index=output_dir / "index.md",
        html=reports_dir / "results.html",
        gallery_markdown=reports_dir / "model_gallery.md",
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
        generate_markdown_gallery_report(
            results=results,
            filename=output_paths.gallery_markdown,
            prompt=prompt,
            report_context=report_context,
        )
        generate_diagnostics_report(
            results,
            output_paths.diagnostics,
            prompt=prompt,
            library_versions=versions,
            system_info=system_info,
            report_context=report_context,
        )
        check_models.save_jsonl_report(
            results=results,
            filename=output_paths.jsonl,
            prompt=prompt,
            system_info=system_info,
            library_versions=versions,
            report_context=report_context,
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
            issue_reports=issue_reports,
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
    upstream_boundary: ExpectedUpstreamBoundary = "generation_started",
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


def _make_quality_success(
    name: str,
    *,
    with_quality_issue: bool,
) -> PerformanceResult:
    """Create a successful result with explicit quality analysis state."""
    qa = GenerationQualityAnalysis(
        is_repetitive=False,
        repeated_token=None,
        word_count=20,
        prompt_checks_ran=True,
        unexpected_special_tokens=["<|unexpected|>"] if with_quality_issue else [],
    )
    return PerformanceResult(
        model_name=name,
        success=True,
        generation=_MockGeneration(
            text="quality output",
            prompt_tokens=120,
            generation_tokens=80,
        ),
        total_time=1.0,
        generation_time=0.6,
        model_load_time=0.4,
        quality_analysis=qa,
    )


def test_report_context_caches_only_live_cross_artifact_views() -> None:
    """The shared context should retain only current-run factual assessments."""
    failed = _make_failure("org/crashed")
    passed = _make_success("org/passed")

    context = _build_report_render_context(
        results=[failed, passed],
        prompt="Describe the image.",
        eval_mode="blind",
    )

    assert [model for model, _assessment in context.assessments] == [
        "org/crashed",
        "org/passed",
    ]
    assert not hasattr(context, "recommendations")
    assert not hasattr(context, "triage")
    assert not hasattr(context, "machine_facts")
    assert not hasattr(context, "diagnostics_snapshot")
    assert not hasattr(context, "issue_clusters")


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


def test_chained_failure_retains_exact_exception_chain() -> None:
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

    assert [entry.exception_type for entry in failure.exception_chain] == [
        "IndexError",
        "RuntimeError",
    ]
    assert [entry.module for entry in failure.exception_chain] == ["builtins", "mlx.core"]


def test_published_failure_artifacts_do_not_disclose_home_paths() -> None:
    """Checked-in human reports should not retain publication-private home paths."""
    output_dir = Path(__file__).parents[1] / "output"
    diagnostics = (output_dir / "reports/diagnostics.md").read_text(encoding="utf-8")
    gallery = (output_dir / "reports/model_gallery.md").read_text(encoding="utf-8")
    html_report = (output_dir / "reports/results.html").read_text(encoding="utf-8")
    assert str(Path.home()) not in diagnostics
    assert str(Path.home()) not in gallery
    assert str(Path.home()) not in html_report


def test_public_failure_evidence_sanitizes_paths_without_mutating_model_text(
    tmp_path: Path,
) -> None:
    """Public operational evidence is portable while generated text stays exact."""
    generated_text = "Model says /Users/alice/source and /private/cache exactly."
    success = replace(
        _make_success("org/generated-paths"),
        generation=_MockGeneration(text=generated_text, generation_tokens=20),
    )
    crash = replace(
        _make_failure_with_details(
            "org/crash-paths",
            error_msg="failed under /Users/alice/project/model.py using /private/tmp/cache",
            traceback_str=(
                "Traceback (most recent call last):\n"
                '  File "/Users/alice/project/model.py", line 7, in run\n'
                "RuntimeError: cache /private/tmp/cache failed"
            ),
            captured_output=(
                "stderr from /Users/alice/project/model.py\nprivate=/private/tmp/cache"
            ),
        ),
        root_error_message="root at /Users/alice/project/model.py",
        exception_chain=(
            check_models.FailureException(
                "RuntimeError",
                "builtins",
                "cache /private/tmp/cache failed",
                origin="/Users/alice/project/model.py",
            ),
        ),
    )
    results = [success, crash]
    provenance: dict[str, check_models.ModelProvenanceRecord] = {
        result.model_name: check_models.ModelProvenanceRecord(
            model=result.model_name,
            requested_revision=None,
            resolved_revision="sha",
            snapshot_path="/Users/alice/.cache/models/snapshots/sha",
        )
        for result in results
    }
    context = _build_report_render_context(
        results=results,
        prompt="Describe the image.",
        system_info={},
        model_provenance=provenance,
    )
    gallery_path = tmp_path / "model_gallery.md"
    diagnostics_path = tmp_path / "diagnostics.md"
    html_path = tmp_path / "results.html"
    jsonl_path = tmp_path / "results.jsonl"
    run_path = tmp_path / "run.json"

    generate_markdown_gallery_report(
        results,
        gallery_path,
        prompt="Describe the image.",
        report_context=context,
    )
    generate_diagnostics_report(
        results,
        diagnostics_path,
        prompt="Describe the image.",
        library_versions=_stub_versions(),
        system_info={},
        report_context=context,
    )
    generate_html_report(
        results,
        html_path,
        _stub_versions(),
        "Describe the image.",
        1.0,
        report_context=context,
    )
    check_models.save_jsonl_report(
        results,
        jsonl_path,
        prompt="Describe the image.",
        system_info={},
        report_context=context,
    )
    check_models.save_run_json_report(
        results,
        run_path,
        versions=_stub_versions(),
        prompt="Describe the image.",
        total_runtime_seconds=1.0,
        report_context=context,
        output_paths={
            "external": "/Users/alice/published/results.html",
            "private": "/private/tmp/results.jsonl",
        },
    )

    gallery = gallery_path.read_text(encoding="utf-8")
    diagnostics = diagnostics_path.read_text(encoding="utf-8")
    html_report = html.unescape(html_path.read_text(encoding="utf-8"))
    crash_gallery = _extract_markdown_model_section(gallery, crash.model_name)
    assert generated_text in gallery
    assert generated_text in html_report
    for crash_evidence in (crash_gallery, diagnostics):
        assert "/Users/alice/" not in crash_evidence
        assert "/private/" not in crash_evidence
        assert "~/project/model.py" in crash_evidence
        assert "<private>/tmp/cache" in crash_evidence
    crash_html_match = re.search(
        r'<article id="model-org-crash-paths".*?</article>',
        html_report,
        re.DOTALL,
    )
    assert crash_html_match is not None
    crash_html_articles = crash_html_match.group(0)
    assert "/Users/alice/" not in crash_html_articles
    assert "/private/" not in crash_html_articles

    records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    rows = {record["model"]: record for record in records if record.get("_type") == "result"}
    assert rows[success.model_name]["generated_text"] == generated_text
    crash_row = rows[crash.model_name]
    assert "/Users/alice/" not in json.dumps(crash_row)
    assert "/private/" not in json.dumps(crash_row)
    assert crash_row["failure"]["exception_chain"][0]["origin"] == "~/project/model.py"
    run_payload = json.loads(run_path.read_text(encoding="utf-8"))
    assert run_payload["artifacts"] == {
        "external": "~/published/results.html",
        "private": "<private>/tmp/results.jsonl",
        "run_issue_summary": "issues/run_summary.md",
    }


def test_tabs_round_trip_across_every_public_model_evidence_artifact(tmp_path: Path) -> None:
    """Hard tabs in captured model output must survive JSON, Markdown, and HTML."""
    output = "left\tright"
    result = replace(
        _make_success("org/tabbed"),
        generation=_MockGeneration(text=output, generation_tokens=2),
    )
    context = _build_report_render_context(results=[result], prompt="Describe the image.")
    jsonl_path = tmp_path / "results.jsonl"
    gallery_path = tmp_path / "model_gallery.md"
    diagnostics_path = tmp_path / "diagnostics.md"
    html_path = tmp_path / "results.html"

    check_models.save_jsonl_report(
        [result],
        jsonl_path,
        prompt="Describe the image.",
        system_info={},
        report_context=context,
    )
    generate_markdown_gallery_report(
        [result],
        gallery_path,
        prompt="Describe the image.",
        report_context=context,
    )
    generate_diagnostics_report(
        [result],
        diagnostics_path,
        prompt="Describe the image.",
        library_versions=_stub_versions(),
        system_info={},
        report_context=context,
    )
    generate_html_report(
        [result],
        html_path,
        _stub_versions(),
        "Describe the image.",
        1.0,
        report_context=context,
    )

    records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    row = next(record for record in records if record.get("_type") == "result")
    assert row["generated_text"] == output
    assert output in gallery_path.read_text(encoding="utf-8")
    assert output in diagnostics_path.read_text(encoding="utf-8")
    html_report = html_path.read_text(encoding="utf-8")
    match = re.search(
        r"Complete generated output.*?<pre><code[^>]*>(.*?)</code></pre>",
        html_report,
        re.DOTALL,
    )
    assert match is not None
    assert html.unescape(match.group(1)) == output


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


def test_machine_reports_share_the_cached_resolved_model_provenance(tmp_path: Path) -> None:
    """Every retained model artifact should serialize one exact snapshot identity."""
    result = _make_success("org/pinned")
    provenance: check_models.ModelProvenanceRecord = {
        "model": result.model_name,
        "requested_revision": "requested-tag",
        "resolved_revision": "abcdef0123456789abcdef0123456789abcdef01",
        "snapshot_path": "~/.cache/snapshots/abcdef0123456789abcdef0123456789abcdef01",
    }
    context = _build_report_render_context(
        results=[result],
        prompt="Describe the image.",
        model_provenance={result.model_name: provenance},
    )
    jsonl_path = tmp_path / "results.jsonl"
    run_json_path = tmp_path / "run.json"
    gallery_path = tmp_path / "model_gallery.md"
    html_path = tmp_path / "results.html"

    with patch.object(check_models, "_collect_model_provenance", side_effect=AssertionError):
        check_models.save_jsonl_report(
            [result],
            jsonl_path,
            prompt="Describe the image.",
            system_info={},
            requested_revision="requested-tag",
            report_context=context,
        )
        check_models.save_run_json_report(
            [result],
            run_json_path,
            versions=_stub_versions(),
            prompt="Describe the image.",
            total_runtime_seconds=1.0,
            report_context=context,
            output_paths={},
            requested_revision="requested-tag",
        )
        generate_markdown_gallery_report(
            [result],
            gallery_path,
            prompt="Describe the image.",
            report_context=context,
        )
        generate_html_report(
            [result],
            html_path,
            _stub_versions(),
            "Describe the image.",
            1.0,
            report_context=context,
        )

    jsonl_record = json.loads(jsonl_path.read_text(encoding="utf-8").splitlines()[1])
    run_record = json.loads(run_json_path.read_text(encoding="utf-8"))
    assert jsonl_record["model_provenance"] == provenance
    assert run_record["model_provenance"] == {result.model_name: provenance}
    gallery = gallery_path.read_text(encoding="utf-8")
    html_report = html.unescape(html_path.read_text(encoding="utf-8"))
    assert "*Requested model revision:* requested-tag" in gallery
    assert f"*Resolved model revision:* {provenance['resolved_revision']}" in gallery
    assert "<td>Requested model revision</td>\n<td>requested-tag</td>" in html_report
    assert (
        f"<td>Resolved model revision</td>\n<td>{provenance['resolved_revision']}</td>"
    ) in html_report


def test_run_context_validator_accepts_exact_mixed_partition() -> None:
    """One validated context must partition every attempted model exactly once."""
    results = [
        _make_success("org/usable"),
        replace(
            _make_success("org/caveat"),
            generation=_MockGeneration(text="Brief reply", generation_tokens=2),
        ),
        replace(
            _make_success("org/unusable"),
            generation=_MockGeneration(text="", generation_tokens=0),
        ),
        _make_failure_with_details("org/crashed", error_msg="decode crashed"),
        _make_failure_with_details(
            "org/indeterminate",
            error_msg="Server disconnected without sending a response.",
        ),
    ]
    provenance: dict[str, check_models.ModelProvenanceRecord] = {
        result.model_name: {
            "model": result.model_name,
            "requested_revision": None,
            "resolved_revision": f"sha-{index}",
            "snapshot_path": f"~/.cache/snapshots/sha-{index}",
        }
        for index, result in enumerate(results)
    }
    context = _build_report_render_context(
        results=results,
        prompt="Describe the image.",
        system_info={},
        model_provenance=provenance,
    )

    check_models._validate_report_render_context(context)

    assert check_models._run_outcome_counts(context.assessments) == {
        "models_attempted": 5,
        "models_evaluated": 4,
        "models_completed": 3,
        "models_crashed": 1,
        "models_indeterminate": 1,
    }


def test_run_context_validator_rejects_duplicate_result_identity() -> None:
    """Duplicate result keys must fail before tuple-to-dict conversion can hide them."""
    result = _make_success("org/duplicate")
    provenance: check_models.ModelProvenanceRecord = {
        "model": result.model_name,
        "requested_revision": None,
        "resolved_revision": "sha",
        "snapshot_path": "~/.cache/snapshots/sha",
    }
    context = _build_report_render_context(
        results=[result, result],
        prompt="Describe the image.",
        system_info={},
        model_provenance={result.model_name: provenance},
    )

    with pytest.raises(ValueError, match="duplicate"):
        check_models._validate_report_render_context(context)


def test_run_context_validator_rejects_key_misalignment() -> None:
    """Result, assessment, and provenance identities must align exactly."""
    result = _make_success("org/model")
    context = _build_report_render_context(
        results=[result],
        prompt="Describe the image.",
        system_info={},
    )

    with pytest.raises(ValueError, match="provenance"):
        check_models._validate_report_render_context(context)


def test_run_context_validator_rejects_illegal_axis_combination() -> None:
    """Completed results cannot carry the not-evaluated usability axis."""
    result = _make_success("org/model")
    context = _build_report_render_context(
        results=[result],
        prompt="Describe the image.",
        system_info={},
        model_provenance={
            result.model_name: {
                "model": result.model_name,
                "requested_revision": None,
                "resolved_revision": "sha",
                "snapshot_path": "~/.cache/snapshots/sha",
            }
        },
    )
    context = replace(
        context,
        assessments=(
            (
                result.model_name,
                check_models.ResultAssessment("completed", "not_evaluated", "none", ()),
            ),
        ),
    )

    with pytest.raises(ValueError, match="illegal"):
        check_models._validate_report_render_context(context)


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
        word_count=0,
        prompt_checks_ran=True,
        unexpected_special_tokens=(
            [harness_detail.split(":", maxsplit=1)[-1]]
            if harness_detail.startswith("token_leak:")
            else []
        ),
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
        "## Run Summary",
        "## Triage",
        "## Crashes requiring action",
        "## Completed Runs with Observations",
        "## Indeterminate Attempts",
        "## Clean Completion Context",
    )
    assert all(heading in content for heading in headings)
    assert content.index(headings[0]) < content.index(headings[1])
    assert content.index(headings[1]) < content.index(headings[2])
    assert content.index(headings[2]) < content.index(headings[3])
    assert content.index(headings[-1]) < content.index("## Shared Reproduction and Provenance")
    assert "actionable_failure" in content
    assert "observation_needs_reproduction" in content
    assert "indeterminate" in content


def test_diagnostics_facts_surface_exact_observation_evidence_without_empty_noise() -> None:
    repeated_fragment = 'keyword: "remote control"'
    analysis = check_models.GenerationQualityAnalysis(
        is_repetitive=True,
        repeated_token=repeated_fragment,
        missing_sections=["title"],
        instruction_echo=True,
        instruction_echo_fragments=["prompt instructions"],
        unexpected_special_tokens=["<|im_user|>"],
    )
    result = PerformanceResult(
        model_name="org/observed",
        success=True,
        generation=_MockGeneration(text="output", generation_tokens=20),
        quality_analysis=analysis,
    )
    assessment = check_models.ResultAssessment(
        "completed",
        "unusable",
        "observation_needs_reproduction",
        ("repeated_output", "missing_requested_sections", "prompt_instruction_echo"),
    )

    facts = dict(
        check_models._diagnostics_result_facts(
            result,
            assessment,
            run_args=None,
            model_provenance=None,
        )
    )

    assert facts["Missing sections"] == '["title"]'
    assert facts["Repeated fragment"] == 'keyword: "remote control"'
    assert facts["Echoed instruction fragments"] == '["prompt instructions"]'
    assert facts["Unexpected special tokens"] == '["<|im_user|>"]'
    assert "Error type" not in facts
    assert "Configured EOS token" not in facts


def test_diagnostics_facts_render_catalog_constraint_evidence() -> None:
    analysis = check_models.GenerationQualityAnalysis(
        is_repetitive=False,
        repeated_token=None,
        title_word_count=4,
        title_word_range=(5, 10),
        keyword_count=10,
        keyword_count_range=(10, 18),
        duplicate_keywords=["building"],
    )
    result = PerformanceResult(
        model_name="org/catalog-constraint",
        success=True,
        generation=_MockGeneration(text="catalogue output", generation_tokens=20),
        quality_analysis=analysis,
    )
    assessment = check_models.ResultAssessment(
        "completed",
        "usable_with_caveats",
        "observation_needs_reproduction",
        ("catalog_constraint_violation",),
    )

    facts = dict(
        check_models._diagnostics_result_facts(
            result,
            assessment,
            run_args=None,
            model_provenance=None,
        )
    )

    assert facts["Title word count"] == "4"
    assert facts["Requested title word range"] == "[5, 10]"
    assert facts["Keyword count"] == "10"
    assert facts["Requested keyword count range"] == "[10, 18]"
    assert facts["Duplicate keywords"] == '["building"]'


def test_diagnostics_are_skim_first_and_share_reproduction_context_once(
    tmp_path: Path,
) -> None:
    """Issue-ready diagnostics should expand faults, collapse context, and avoid repetition."""
    prompt = "Exact multiline prompt.\nSecond distinctive line."
    crash = replace(
        _make_failure_with_details(
            "org/crash",
            error_msg="decoder failed",
            generated_text="CRASH-PARTIAL",
            traceback_str="TRACEBACK-FIRST\nRuntimeError: decoder failed",
            captured_output="CAPTURED-AFTER-PARTIAL",
        ),
        prompt_diagnostics=check_models.PromptDiagnostics(
            processor_class="mlx_vlm.processors.CrashProcessor",
            tokenizer_class="transformers.CrashTokenizer",
        ),
    )
    repeated_fragment = 'phrase: "OBSERVED-OUTPUT-MUST-APPEAR"'
    observed = PerformanceResult(
        model_name="org/observed",
        success=True,
        generation=_MockGeneration(
            text="OBSERVED-OUTPUT-MUST-APPEAR",
            prompt_tokens=30,
            generation_tokens=80,
            generation_tps=20.0,
            peak_memory=2.0,
        ),
        prompt_diagnostics=check_models.PromptDiagnostics(
            processor_class="mlx_vlm.processors.ObservedProcessor",
        ),
        runtime_diagnostics=RuntimeDiagnostics(stop_reason="eos"),
        quality_analysis=check_models.GenerationQualityAnalysis(
            is_repetitive=True,
            repeated_token=repeated_fragment,
            prompt_checks_ran=True,
        ),
    )
    indeterminate = PerformanceResult(
        model_name="org/network",
        success=False,
        generation=None,
        error_message="503 Service Unavailable — INDETERMINATE-EVIDENCE",
        captured_output_on_fail="SERVER-COULD-NOT-BE-CONTACTED",
    )
    clean_one = replace(
        _make_success("org/clean-one"),
        generation=_MockGeneration(
            text="CLEAN-OUTPUT-MUST-NOT-APPEAR",
            prompt_tokens=44,
            generation_tokens=40,
            generation_tps=16.5,
            peak_memory=1.5,
        ),
        prompt_diagnostics=check_models.PromptDiagnostics(
            processor_class="mlx_vlm.processors.CleanProcessor",
        ),
        runtime_diagnostics=RuntimeDiagnostics(stop_reason="eos"),
    )
    clean_two = replace(
        _make_success("org/clean-two"),
        generation=_MockGeneration(
            text="SECOND-CLEAN-OUTPUT-MUST-NOT-APPEAR",
            prompt_tokens=50,
            generation_tokens=8,
            generation_tps=999.0,
            peak_memory=3.0,
        ),
        prompt_diagnostics=check_models.PromptDiagnostics(
            processor_class="OtherProcessor",
        ),
        runtime_diagnostics=RuntimeDiagnostics(stop_reason="length"),
    )
    results = [crash, observed, indeterminate, clean_one, clean_two]
    assessments = {
        "org/crash": check_models.ResultAssessment(
            "crashed", "not_evaluated", "actionable_failure", ()
        ),
        "org/observed": check_models.ResultAssessment(
            "completed", "unusable", "observation_needs_reproduction", ("repeated_output",)
        ),
        "org/network": check_models.ResultAssessment("indeterminate", "not_evaluated", "none", ()),
        "org/clean-one": check_models.ResultAssessment("completed", "usable", "none", ()),
        "org/clean-two": check_models.ResultAssessment("completed", "usable", "none", ()),
    }
    revisions: dict[str, check_models.ModelProvenanceRecord] = {
        result.model_name: {
            "model": result.model_name,
            "requested_revision": None,
            "resolved_revision": f"{index:012d}full-revision",
            "snapshot_path": None,
        }
        for index, result in enumerate(results, start=1)
    }
    context = _build_report_render_context(
        results=results,
        prompt=prompt,
        system_info={"GPU/Chip": "Apple M5"},
        model_provenance=revisions,
    )
    context = replace(context, assessments=tuple(assessments.items()))
    markdown_path = tmp_path / "diagnostics.md"
    html_path = tmp_path / "results.html"

    generate_diagnostics_report(
        results,
        markdown_path,
        prompt=prompt,
        library_versions=_stub_versions(),
        system_info=context.system_info,
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

    diagnostics = markdown_path.read_text(encoding="utf-8")
    html_report = html_path.read_text(encoding="utf-8")
    assert all(
        label in diagnostics
        for label in (
            "Outcome counts",
            "Maintainer status counts",
            "Usability counts",
            "Observation counts",
        )
    )
    assert "[org/crash](#diagnostic-org-crash)" in diagnostics
    assert "[org/observed](#diagnostic-org-observed)" in diagnostics
    assert "[org/network](#diagnostic-org-network)" in diagnostics
    triage = _extract_markdown_subsection(
        diagnostics,
        "## Triage",
        end_headings=("## Crashes requiring action",),
    )
    assert "org/clean-one" not in triage
    assert "org/clean-two" not in triage
    assert diagnostics.index("TRACEBACK-FIRST") < diagnostics.index("CRASH-PARTIAL")
    assert "<summary>org/observed" in diagnostics
    assert "<summary>org/network" in diagnostics
    assert re.search(
        r"\| Response repeats the same text\s+\|\s+1 \|\n\n## Triage",
        diagnostics,
    )
    assert re.search(
        r"\| \[org/network\].*\|\n\n## Crashes requiring action",
        diagnostics,
    )
    assert "OBSERVED-OUTPUT-MUST-APPEAR" in diagnostics
    assert diagnostics.count("#### Complete output") == 1
    assert "Repeated fragment" in diagnostics
    assert "SERVER-COULD-NOT-BE-CONTACTED" in diagnostics
    assert "<summary>Clean completions</summary>" in diagnostics
    assert "000000000004" in diagnostics
    assert "CleanProcessor" in diagnostics
    assert "eos" in diagnostics
    assert "44 prompt / 40 generated" in diagnostics
    assert "16.5 tok/s" in diagnostics
    assert "1.5 GB" in diagnostics
    assert "insufficient sample" in diagnostics
    assert "CLEAN-OUTPUT-MUST-NOT-APPEAR" not in diagnostics
    assert "SECOND-CLEAN-OUTPUT-MUST-NOT-APPEAR" not in diagnostics
    assert diagnostics.count(prompt) == 1
    assert diagnostics.count("The original local input is not published") == 1
    assert diagnostics.count("Exact prompt") == 1
    assert all(
        unavailable_ref not in diagnostics
        for unavailable_ref in ("reproduce.py", "prompt.txt", "python -m mlx_vlm.generate")
    )
    for model in ("org/crash", "org/observed", "org/network"):
        assert model in diagnostics
        revision = revisions[model]["resolved_revision"]
        assert revision is not None
        assert revision in diagnostics
    maintainer_html = html_report.split('<section id="maintainer-diagnostics">', maxsplit=1)[1]
    maintainer_html = maintainer_html.split("</section>", maxsplit=1)[0]
    assert "CLEAN-OUTPUT-MUST-NOT-APPEAR" not in maintainer_html
    assert "OBSERVED-OUTPUT-MUST-APPEAR" in maintainer_html
    assert 'href="#diagnostic-org-crash"' in maintainer_html
    assert html_report.count("The original local input is not published") == 1
    assert all(
        unavailable_ref not in html_report for unavailable_ref in ("reproduce.py", "prompt.txt")
    )


def test_html_chooser_is_sortable_and_surfaces_prefill_first_token_time(
    tmp_path: Path,
) -> None:
    """HTML alone should expose sortable per-model prefill/first-token latency."""
    result = replace(
        _make_success("org/timed"),
        runtime_diagnostics=RuntimeDiagnostics(first_token_latency_s=0.375),
    )
    html_path = tmp_path / "results.html"

    generate_html_report(
        [result],
        html_path,
        _stub_versions(),
        "Describe the image.",
        1.0,
    )

    content = html_path.read_text(encoding="utf-8")
    chooser = content.split('<section id="current-run-chooser">', maxsplit=1)[1]
    chooser = chooser.split("</section>", maxsplit=1)[0]
    assert "Prefill/first s" in chooser
    assert 'data-sort-column="6"' in chooser
    assert 'data-sort-value="0.375"' in chooser
    assert "sortChooserColumn" in chooser


def test_crash_diagnostics_and_issue_draft_keep_complete_primary_evidence_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crash draft should repeat complete diagnostics evidence without truncation."""
    monkeypatch.setattr(
        check_models,
        "_collect_check_models_provenance",
        lambda: {
            "name": "check_models",
            "version": "0.8.9",
            "git_revision": "abc123def456",
            "install_type": "source-tree",
            "dirty": False,
        },
    )
    system_info = {
        "Python Version": "3.13.13",
        "macOS Version": "26.6",
        "GPU/Chip": "Apple M5 Max",
        "SDK Version": "26.0",
        "Apple Clang Version": "17.0.0",
    }
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
        system_info=system_info,
    )
    diagnostics = tmp_path / "diagnostics.md"
    generate_diagnostics_report(
        [crash],
        diagnostics,
        prompt="Describe the image.",
        library_versions=_stub_versions(),
        system_info=system_info,
        report_context=context,
    )
    generated = _generate_github_issue_reports(
        report_context=context,
        output_dir=tmp_path,
        library_versions=_stub_versions(),
        system_info=system_info,
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
    assert "python -m mlx_vlm.generate" not in diagnostics_content
    assert "The original local input is not published" in diagnostics_content
    assert "reproduce.py" not in diagnostics_content
    assert "prompt.txt" not in diagnostics_content
    for content in (diagnostics_content, issue_content):
        assert content.index("#### Root exception and chain") < content.index(
            "#### Execution and provenance"
        )
        assert content.index("#### Execution and provenance") < content.index("Complete traceback")
    assert "<summary>Complete traceback</summary>" in issue_content
    # Diagnostics folds the complete traceback too, so one crash's dump cannot
    # bury the triage tables; the exact evidence stays inside the details block.
    assert "<summary>Complete traceback</summary>" in diagnostics_content
    assert "The original local input is not published" in issue_content
    assert "python -m mlx_vlm.generate" not in issue_content
    assert issue_content.index("## Reproduction inputs") < issue_content.index(
        "## Provenance and Environment"
    )
    environment_url = (
        "https://github.com/jrp2014/check_models/blob/"
        f"{check_models._github_blob_ref()}/src/output/environment.log"
    )
    for expected in (
        "mlx-vlm",
        "mlx",
        "transformers",
        "tokenizers",
        "Python Version",
        "macOS Version",
        "GPU/Chip",
        "abc123def456",
        environment_url,
    ):
        assert expected in issue_content
    assert "SDK Version" not in issue_content
    assert "Apple Clang Version" not in issue_content
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
        end_headings=("## Completed Runs with Observations",),
    )
    assert "Complete partial output\n\n```text\n(empty)" in empty_entry
    assert "generation failed before output" in missing_entry
    assert "Complete traceback" not in missing_entry
    assert "Complete partial output" not in missing_entry
    assert "Captured stdout/stderr" not in missing_entry


def test_diagnostics_describe_local_reproduction_input_without_fake_command(
    tmp_path: Path,
) -> None:
    """Diagnostics should preserve local input facts without inventing a runnable command."""
    result = replace(
        _make_failure_with_details("org/repro", error_msg="decode failed"),
        prompt_diagnostics=check_models.PromptDiagnostics(
            eos_token_id=2,
            eos_token=EOS_END_TOKEN,
            generate_kwargs={"eos_tokens": [EOS_OVERRIDE_TOKEN]},
        ),
    )
    resolved_revision = "0123456789abcdef0123456789abcdef01234567"
    context = _build_report_render_context(
        results=[result],
        prompt="Describe the image.",
        model_provenance={
            result.model_name: {
                "model": result.model_name,
                "requested_revision": "run-revision",
                "resolved_revision": resolved_revision,
                "snapshot_path": f"~/.cache/snapshots/{resolved_revision}",
            }
        },
    )
    output = tmp_path / "diagnostics.md"
    image_path = Path(__file__).parent / "fixtures/check_models-task9-fixture.jpg"
    assert image_path.is_file()
    assert (
        check_models._sha256_file(image_path)
        == "251712968e443405f6e1ff145de15a91a082dc073209938c5305db0e8e80134c"
    )
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

    diagnostics_content = output.read_text(encoding="utf-8")
    issue_content = next(iter(issue_reports.values())).read_text(encoding="utf-8")
    for content in (diagnostics_content, issue_content):
        assert f"- *Resolved model revision:* {resolved_revision}" in content
        assert "- *Requested model revision:* run-revision" in content
        assert '- *Configured EOS token override:* ["&lt;override-eos&gt;"]' in content
    assert "Supplemental CLI reproduction" not in diagnostics_content
    assert "The original local input is not published" in diagnostics_content
    assert "reproduce.py" not in diagnostics_content
    assert "prompt.txt" not in diagnostics_content
    assert "python -m mlx_vlm.generate" not in diagnostics_content
    assert resolved_revision in diagnostics_content
    assert "Reproduction inputs" in diagnostics_content
    assert "JPEG" in diagnostics_content
    assert "17,235 bytes" in diagnostics_content
    assert "251712968e443405f6e1ff145de15a91a082dc073209938c5305db0e8e80134c" in diagnostics_content
    assert "check_models-task9-fixture.jpg" not in diagnostics_content
    assert "Reproduction inputs" in issue_content
    assert "The original local input is not published" in issue_content
    assert "JPEG" in issue_content
    assert "17,235 bytes" in issue_content
    assert "251712968e443405f6e1ff145de15a91a082dc073209938c5305db0e8e80134c" in issue_content
    assert "Supplemental CLI reproduction" not in issue_content
    assert "Canonical Python reproduction script" not in issue_content
    assert "check_models-task9-fixture.jpg" not in issue_content


def test_crash_issue_draft_builds_complete_public_image_reproduction(tmp_path: Path) -> None:
    """A direct crash draft should fetch, verify, and use a public exact input."""
    result = _make_failure_with_details("org/public-repro", error_msg="decode failed")
    resolved_revision = "0123456789abcdef0123456789abcdef01234567"
    image_path = Path(__file__).parent / "fixtures/check_models-task9-fixture.jpg"
    context = _build_report_render_context(
        results=[result],
        prompt="Describe the image exactly.",
        image_path=image_path,
        model_provenance={
            result.model_name: {
                "model": result.model_name,
                "requested_revision": "main",
                "resolved_revision": resolved_revision,
                "snapshot_path": None,
            }
        },
    )
    run_args = Namespace(
        image_source_url="https://example.test/images/cats.jpg",
        max_tokens=321,
        temperature=0.0,
        revision="main",
        trust_remote_code=True,
    )

    issue_reports = _generate_github_issue_reports(
        report_context=context,
        output_dir=tmp_path,
        library_versions=_stub_versions(),
        system_info={},
        prompt="Describe the image exactly.",
        image_path=image_path,
        run_args=run_args,
    )

    content = next(iter(issue_reports.values())).read_text(encoding="utf-8")
    assert "https://example.test/images/cats.jpg" in content
    assert "curl --fail --location" in content
    assert "set -euo pipefail\ncurl --fail --location" in content
    assert "shasum -a 256 --check" in content
    assert "python -m mlx_vlm.generate" in content
    assert "--model org/public-repro" in content
    assert f"--revision {resolved_revision}" in content
    assert "--prompt 'Describe the image exactly.'" in content
    assert "--image repro-image.jpg" in content
    assert "reproduce.py" not in content
    assert "prompt.txt" not in content


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
    jsonl_path = tmp_path / "results.jsonl"
    check_models.save_jsonl_report(
        [failure],
        jsonl_path,
        prompt="Describe the image.",
        system_info={},
        report_context=context,
    )

    jsonl_text = jsonl_path.read_text(encoding="utf-8")
    assert "owner_confidence" not in jsonl_text
    assert "suspected_owner" not in jsonl_text


class TestHtmlReportEdgeCases:
    """Edge-case coverage for generate_html_report."""

    def test_html_mirrors_cached_assessments_across_retained_artifacts(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Every retained consumer should expose one exact cached status vocabulary."""
        caplog.set_level(logging.INFO)
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
        run_path = tmp_path / "run.json"

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
            check_models.save_run_json_report(
                results,
                run_path,
                versions=_stub_versions(),
                prompt=prompt,
                total_runtime_seconds=5.0,
                report_context=context,
                output_paths={},
            )
            check_models.log_summary(results, assessments=expected)

        records = {
            record["model"]: record
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            if (record := json.loads(line)).get("_type") == "result"
        }
        diagnostics = diagnostics_path.read_text(encoding="utf-8")
        gallery = gallery_path.read_text(encoding="utf-8")
        html_report = html_path.read_text(encoding="utf-8")
        run_report = json.loads(run_path.read_text(encoding="utf-8"))
        assert run_report["counts"] == {
            "models_attempted": 5,
            "models_evaluated": 4,
            "models_completed": 3,
            "models_crashed": 1,
            "models_indeterminate": 1,
        }
        log_text = "\n".join(record.message for record in caplog.records)
        assert "status=OK" not in log_text
        assert "Successful Models" not in log_text
        assert "Execution outcomes: completed=3, crashed=1, indeterminate=1" in log_text
        for model, assessment in expected.items():
            serialized = records[model]["assessment"]
            assert serialized["execution"] == assessment.execution
            assert serialized["usability"] == assessment.usability
            assert serialized["maintainer_status"] == assessment.maintainer_status
            gallery_entry = _extract_markdown_model_section(gallery, model)
            assert f"*Execution:* {assessment.execution}" in gallery_entry
            assert f"*Usability:* {assessment.usability}" in gallery_entry
            assert f"*Maintainer status:* {assessment.maintainer_status}" in gallery_entry
            escaped_model = html.escape(model, quote=True)
            row_pattern = (
                rf'data-model="{re.escape(escaped_model)}"[^>]*'
                rf'data-execution="{assessment.execution}"[^>]*'
                rf'data-usability="{assessment.usability}"[^>]*'
                rf'data-maintainer-status="{assessment.maintainer_status}"'
            )
            assert re.search(row_pattern, html_report) is not None
            if assessment.maintainer_status != "none" or assessment.execution == "indeterminate":
                diagnostics_entry = _extract_markdown_diagnostic_entry(diagnostics, model)
                assert f"*Execution:* {assessment.execution}" in diagnostics_entry
                assert f"*Usability:* {assessment.usability}" in diagnostics_entry
                assert f"*Maintainer status:* {assessment.maintainer_status}" in diagnostics_entry

    def test_standalone_html_does_not_build_legacy_semantic_context(
        self,
        tmp_path: Path,
    ) -> None:
        """Standalone HTML should build only its canonical gallery/diagnostic context."""
        result = _make_success("org/standalone")
        out = tmp_path / "standalone.html"

        with patch.object(
            check_models,
            "_build_report_render_context",
            side_effect=AssertionError,
        ):
            generate_html_report(
                [result],
                out,
                _stub_versions(),
                "Describe.",
                1.0,
            )

        content = out.read_text(encoding="utf-8")
        assert 'data-execution="completed"' in content
        assert 'data-usability="usable"' in content
        assert 'data-maintainer-status="none"' in content

    def test_html_diagnostics_preserve_nondefault_run_arguments(
        self,
        tmp_path: Path,
    ) -> None:
        """HTML maintainer facts and shared repro should retain the run configuration."""
        result = _make_failure_with_details("org/repro", error_msg="decode failed")
        context = _build_report_render_context(results=[result], prompt="Describe the image.")
        output = tmp_path / "results.html"
        image_path = tmp_path / "sample image.jpg"
        Image.new("RGB", (12, 8), "blue").save(image_path)
        run_args = Namespace(
            adapter_path=tmp_path / "adapter",
            revision="refs/pr/42",
            trust_remote_code=False,
            enable_thinking=True,
            thinking_budget=19,
            thinking_start_token=THINKING_START_TOKEN,
            thinking_end_token=CUSTOM_THINKING_END_TOKEN,
            max_tokens=321,
            temperature=0.42,
            processor_kwargs={"cropping": False},
            image_source_url="https://example.test/images/cats.jpg",
        )

        generate_html_report(
            [result],
            output,
            _stub_versions(),
            "Describe the image.",
            1.0,
            image_path=image_path,
            report_context=context,
            run_args=run_args,
        )

        content = html.unescape(output.read_text(encoding="utf-8"))
        assert "<li><b>Requested model revision:</b> refs/pr/42</li>" in content
        assert "curl --fail --location" in content
        assert "shasum -a 256 --check" in content
        assert "python -m mlx_vlm.generate" in content
        assert "MODEL_ID" in content
        assert "RESOLVED_REVISION" in content
        assert "reproduce.py" not in content
        assert "prompt.txt" not in content

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
        # HTML escaping is lossless, so the readable pre block is the single
        # exact copy; a second collapsed raw copy would always be identical.
        assert content.count(escaped) == 1
        match = re.search(
            r"<details><summary>Complete evidence: org/evidence</summary>.*?"
            r'<pre class="model-output-readable">(.*?)</pre>',
            content,
            flags=re.DOTALL,
        )
        assert match is not None
        assert html.unescape(match.group(1)) == output

    def test_html_gallery_renders_readable_and_exact_escaped_model_output(
        self,
        tmp_path: Path,
    ) -> None:
        """HTML should expose both readable preformatted text and collapsed exact evidence."""
        output = "## Title\n\n- cat\n\n@maintainer <details>unsafe</details>"
        result = replace(
            _make_success("org/formatted"),
            generation=_MockGeneration(text=output, generation_tokens=80),
        )
        context = _build_report_render_context(results=[result], prompt="Describe.")
        out = tmp_path / "formatted.html"

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
        model_entry = re.search(
            r'<article id="model-org-formatted">.*?</article>',
            content,
            flags=re.DOTALL,
        )
        assert model_entry is not None
        assert f'<pre class="model-output-readable">{escaped}</pre>' in model_entry.group(0)
        # The readable pre already carries the exact escaped bytes; no second
        # collapsed raw copy is emitted.
        assert "<summary>Exact raw output</summary>" not in model_entry.group(0)
        assert model_entry.group(0).count(escaped) == 1

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
        assert "Crashes requiring action" in content
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
        assert "Crashes requiring action" in content
        assert "Completed Runs with Observations" in content
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

        diagnostics_text = diagnostics.read_text(encoding="utf-8")
        assert re.search(r"\|\s*Attempted\s*\|\s*2\s*\|", diagnostics_text)
        assert re.search(r"\|\s*Conclusive outcomes\s*\|\s*1\s*\|", diagnostics_text)
        assert re.search(r"\|\s*Indeterminate\s*\|\s*1\s*\|", diagnostics_text)
        assert re.search(r"\|\s*Crashed\s*\|\s*0\s*\|", diagnostics_text)

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
# Retained Markdown artifacts
# ===================================================================


class TestRetainedMarkdownArtifactEdges:
    """Cross-artifact coverage for the retained Markdown surfaces."""

    def test_generated_report_stamps_do_not_use_emphasis_only_lines(
        self,
        tmp_path: Path,
    ) -> None:
        """Generated Markdown timestamp stamps should not look like headings."""
        success = _make_success("org/good")
        failure = _make_failure("org/bad")
        prompt = "Describe this image briefly."
        context = _build_report_render_context(results=[success, failure], prompt=prompt)

        generated_paths = [tmp_path / "model_gallery.md", tmp_path / "diagnostics.md"]

        generate_markdown_gallery_report(
            results=[success, failure],
            filename=generated_paths[0],
            prompt=prompt,
            report_context=context,
        )
        generate_diagnostics_report(
            [failure],
            generated_paths[1],
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
            "reports/model_gallery.md",
        }
        expected_non_markdown_artifacts = {
            "reports/results.html",
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
                # Local-only artifacts are never published, so they are the only
                # links allowed to stay relative in GitHub link style.
                assert relative_targets
                assert all(
                    target.split("#", 1)[0].rsplit("/", 1)[-1]
                    in check_models._LOCAL_ONLY_OUTPUT_ARTIFACT_NAMES
                    for target in relative_targets
                )
            else:
                assert relative_targets
                environment_url = (
                    "https://github.com/jrp2014/check_models/blob/"
                    f"{check_models._github_blob_ref()}/src/output/environment.log"
                )
                assert github_output_targets == [environment_url]

            html_content = output_paths.html.read_text(encoding="utf-8")
            jsonl_records = [
                json.loads(line)
                for line in output_paths.jsonl.read_text(encoding="utf-8").splitlines()
            ]
            run_payload = json.loads(output_paths.run_json.read_text(encoding="utf-8"))
            mode_summaries[link_style] = {
                "html_markers": (
                    "Action Snapshot" in html_content,
                    "org/good" in html_content,
                    "org/broken" in html_content,
                ),
                "jsonl_header": jsonl_records[0]["_type"],
                "jsonl_models": [record["model"] for record in jsonl_records[1:]],
                "run_json_counts": run_payload["counts"],
                "run_json_artifacts": sorted(run_payload["artifacts"]),
            }

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

        assert mode_summaries["github"] == mode_summaries["relative"]


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
        assert "*Title:* Harbor Sunset" in content
        assert "*Description:* Fishing boats at dusk." in content
        assert "*Keywords:* harbor, boats, sunset" in content
        assert "*Date:* 2026-03-08" in content
        assert "*Time:* 18:42:00" in content
        assert "*GPS:* 51.5000, -0.1200" in content
        assert "ignored raw blob" not in content
        assert "## Prompt" in content
        assert "## Current-run Chooser" in content
        assert "## Avoid for This Run" in content
        assert "## Lowest-memory Usable Models (Including Caveats)" in content
        assert "## Fastest Usable Models (Including Caveats)" in content
        assert "> [!NOTE]" not in content
        assert "Describe this image fully." in content
        assert "```text\nDescribe this image fully." not in content
        assert "<summary>Complete evidence: org/good</summary>" in content
        assert '<pre class="model-output-readable">' in content
        assert '<a id="model-org-good"></a>' in content
        assert "*Usability:*" in content
        assert "*Observations:*" in content
        assert "*Verdict:*" not in content
        assert "*Maintainer:*" not in content
        assert "*Next action:*" not in content
        assert "### org/good" in content
        assert "### org/bad" in content

    def test_gallery_uses_cached_usability_not_recommendation_icons(self, tmp_path: Path) -> None:
        """Completed output should expose cached usability without recommendation policy."""
        text = "<think>Inspect.</think> A useful final caption."
        result = _make_success("org/thinking")
        analysis = check_models.analyze_generation_text(
            text,
            generated_tokens=12,
            prompt="Describe this image.",
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
        assert "`usable`" in chooser_row
        assert "none" in chooser_row
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
        assert "*Score:*" not in content
        assert "Keywords are not specific" not in content
        assert "*Review focus:*" not in content

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
        assert "- *GPU Architecture:* applegpu_g17s" in content
        assert "- *Recommended Working Set:* 96 GB" in content
        assert "- *Fused Attention:* available" in content
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
        assert "Output preview" not in chooser
        assert "[`org/good`](#model-org-good)" in chooser
        assert "quality output" not in chooser
        assert "[`org/risky`](#model-org-risky)" in chooser
        assert "control tokens visible" in chooser
        assert "Prefill/first s" in chooser
        assert r"answer with \| pipe" not in chooser
        assert "&lt;think&gt;leaked marker&lt;/think&gt;" not in chooser
        assert "[`org/bad`](#model-org-bad)" in chooser
        assert "not_evaluated" in chooser
        assert "boom" not in chooser
        avoid = _extract_markdown_subsection(
            content,
            "## Avoid for This Run",
            end_headings=("## Lowest-memory Usable Models (Including Caveats)",),
        )
        assert "Output preview" not in avoid
        assert "boom" not in avoid

    def test_gallery_keeps_exact_output_in_expandable_code_block(
        self,
        tmp_path: Path,
    ) -> None:
        """The gallery should keep exact evidence without making the chooser unwieldy."""
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
        assert "Output preview" not in chooser
        assert "BEGIN" not in chooser
        assert "END-SENTINEL" not in chooser
        assert "<!-- markdownlint-disable MD034 MD049 -->" in chooser
        assert chooser.index("Total s") < chooser.index("Gen TPS")
        assert "Gen tok" in chooser
        assert "Peak GB" in chooser
        assert "Observations" in chooser
        assert "<summary>Complete evidence: org/complete-output</summary>" in content
        # Plain text renders once as the readable view; the raw fence would be
        # byte-identical, so exactly one exact copy is retained.
        assert f"```text\n{complete_text}\n```" not in content
        assert complete_text in content
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
        assert "Output preview" not in chooser
        assert "Peak GB" in chooser
        assert "Prefill/first s" in chooser
        assert "Observations" in chooser
        assert "[`org/full-caption`](#model-org-full-caption)" in chooser
        assert "Two cats sit together on a pink sofa" not in chooser
        assert "24" in chooser
        assert "42.0" in chooser
        assert "2.5" in chooser
        risky_row = next(line for line in chooser.splitlines() if "org/risky-output" in line)
        assert "| cats " not in risky_row
        assert "insufficient sample" in risky_row
        assert "[`org/crashed`](#model-org-crashed)" in chooser
        assert "boom" not in chooser
        crashed_evidence = _extract_markdown_subsection(
            content,
            "### org/crashed",
            end_headings=("### org/full-caption", "### org/risky-output"),
        )
        assert "*Total time:* 0.33s" in crashed_evidence

    def test_gallery_uses_skim_first_chooser_order_and_cached_assessments(
        self,
        tmp_path: Path,
    ) -> None:
        """Gallery order should move from chooser policy to complete evidence."""
        results = [
            _make_success("org/usable"),
            _make_success("org/caveated"),
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
                    "org/caveated",
                    check_models.ResultAssessment("completed", "usable_with_caveats", "none", ()),
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
            "## Lowest-memory Usable Models (Including Caveats)",
            "## Fastest Usable Models (Including Caveats)",
            "## Complete Per-model Evidence",
        ]
        assert [content.index(heading) for heading in headings] == sorted(
            content.index(heading) for heading in headings
        )
        chooser = _extract_markdown_subsection(
            content,
            "## Current-run Chooser",
            end_headings=("## Avoid for This Run",),
        )
        expected_model_order = (
            "org/usable",
            "org/caveated",
            "org/unusable",
            "org/not-evaluated",
        )
        assert [chooser.index(model) for model in expected_model_order] == sorted(
            chooser.index(model) for model in expected_model_order
        )
        assert [content.index(f"### {model}") for model in expected_model_order] == sorted(
            content.index(f"### {model}") for model in expected_model_order
        )
        assert "*Verdict:*" not in content
        assert "*Maintainer:*" not in content
        assert "*Next action:*" not in content
        assert "*Score:*" not in content

        html_out = tmp_path / "results.html"
        generate_html_report(
            results,
            html_out,
            versions={},
            prompt="Describe the image.",
            total_runtime_seconds=1.0,
            report_context=context,
        )
        html_content = html_out.read_text(encoding="utf-8")
        html_chooser = html_content[
            html_content.index('<div id="chooser-table">') : html_content.index(
                "</div>", html_content.index('<div id="chooser-table">')
            )
        ]
        assert [html_chooser.index(model) for model in expected_model_order] == sorted(
            html_chooser.index(model) for model in expected_model_order
        )
        html_complete_evidence = html_content[
            html_content.index('<section id="complete-model-evidence">') :
        ]
        assert [html_complete_evidence.index(model) for model in expected_model_order] == sorted(
            html_complete_evidence.index(model) for model in expected_model_order
        )

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

    def test_gallery_omits_preview_but_preserves_readable_model_formatting(
        self,
        tmp_path: Path,
    ) -> None:
        """Complete evidence should retain useful formatting without chooser duplication."""
        formatted_output = (
            "## Title\n\n"
            "Two cats resting\n\n"
            "- pink sofa\n"
            "- remote control\n\n"
            "@maintainer <details>unsafe</details>\n"
            "```text\nnested\n```"
        )
        result = replace(
            _make_success("org/formatted"),
            generation=_MockGeneration(text=formatted_output, generation_tokens=80),
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
        chooser = _extract_markdown_subsection(
            content,
            "## Current-run Chooser",
            end_headings=("## Avoid for This Run",),
        )
        assert "## Title<br><br>Two cats resting" not in chooser
        assert '<pre class="model-output-readable">' in content
        assert "## Title\n\nTwo cats resting\n\n- pink sofa" in content
        assert "&#96;&#96;&#96;text" in content
        assert "&#64;maintainer &lt;details&gt;unsafe&lt;/details&gt;" in content
        assert "<summary>Exact raw output</summary>" in content
        assert content.count(formatted_output) == 1

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
        assert "*Generation time:* 0.25s" in evidence
        assert "*Generation throughput (raw):* 999 tok/s" in evidence
        assert "*Generation tokens:* 8" in evidence

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
            end_headings=("## Lowest-memory Usable Models (Including Caveats)",),
        )
        memory = _extract_markdown_subsection(
            content,
            "## Lowest-memory Usable Models (Including Caveats)",
            end_headings=("## Fastest Usable Models (Including Caveats)",),
        )
        speed = _extract_markdown_subsection(
            content,
            "## Fastest Usable Models (Including Caveats)",
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
        assert "*Failure phase:* decode" in evidence
        assert "*Error code:* generation-failed" in evidence
        assert "*Error package:* mlx-vlm" in evidence
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
        assert "*Execution:* indeterminate" in content
        assert "*Execution:* crashed" not in content

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
        assert "*Stop reason:* not captured" in evidence
        assert "*Processor:* not captured" in evidence
        assert "*Tokenizer:* not captured" in evidence

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
        assert "*Review focus:*" not in content
        assert "*Score:*" not in content
        assert "*Usability:*" in content
        assert "*Execution:*" in content
        assert "*Next action:*" not in content


# ===================================================================
# Stale retained report copies
# ===================================================================


class TestCleanStaleToplevelReports:
    """Regression coverage for stale top-level report cleanup."""

    def test_removes_stale_files_when_canonical_exists(self, tmp_path: Path) -> None:
        """A stale top-level file is removed when the reports copy exists."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        (tmp_path / "results.html").write_text("old", encoding="utf-8")
        (reports_dir / "results.html").write_text("canonical", encoding="utf-8")
        (tmp_path / "model_gallery.md").write_text("old gallery", encoding="utf-8")
        (reports_dir / "model_gallery.md").write_text(
            "canonical gallery",
            encoding="utf-8",
        )

        removed = _clean_stale_toplevel_reports(tmp_path, reports_dir)

        assert removed == 2
        assert not (tmp_path / "results.html").exists()
        assert not (tmp_path / "model_gallery.md").exists()

    def test_keeps_file_when_no_canonical(self, tmp_path: Path) -> None:
        """A top-level file is kept when no reports copy exists."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        only_copy = tmp_path / "results.html"
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
