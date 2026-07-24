"""Tests for JSONL output generation."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

from PIL import Image

import check_models
from check_models import (
    JsonlMetadataRecord,
    JsonlResultRecord,
    PerformanceResult,
    RuntimeDiagnostics,
    _history_path_for_jsonl,
    append_history_record,
    save_jsonl_report,
)
from tools import safe_io

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _read_jsonl(path: Path) -> tuple[JsonlMetadataRecord, list[JsonlResultRecord]]:
    """Read JSONL file returning (metadata_header, result_rows)."""
    lines = safe_io.read_text_no_follow(path).strip().split("\n")
    header = cast("JsonlMetadataRecord", json.loads(lines[0]))
    results = [cast("JsonlResultRecord", json.loads(line)) for line in lines[1:]]
    return header, results


def _require_present[T](value: T | None, *, field_name: str) -> T:
    """Return an optional test payload after asserting that it exists."""
    if value is None:
        raise AssertionError(field_name)
    return value


@dataclass
class MockGeneration:
    """Mock generation result for testing."""

    text: str | None = "generated text"
    token: object | None = None
    logprobs: object | None = None
    prompt_tokens: int | None = 10
    generation_tokens: int | None = 20
    total_tokens: int | None = 30
    prompt_tps: float | None = 2.0
    generation_tps: float | None = 5.0
    peak_memory: float | None = 1.5
    time: float | None = None
    active_memory: float | None = None
    cache_memory: float | None = None
    quality_analysis: object | None = None


def test_save_jsonl_report_creates_file(tmp_path: Path) -> None:
    """Test that save_jsonl_report creates a file with metadata header."""
    output_file = tmp_path / "results.jsonl"
    results: list[PerformanceResult] = []
    save_jsonl_report(
        results,
        output_file,
        prompt="test",
        system_info={},
        eval_mode="blind",
        metadata_exposed_to_prompt=False,
    )

    assert output_file.exists()
    header, rows = _read_jsonl(output_file)
    assert header["_type"] == "metadata"
    assert header["format_version"] == "2.0"
    assert header["prompt"] == "test"
    assert header["eval_mode"] == "blind"
    assert header["metadata_exposed_to_prompt"] is False
    assert rows == []


def test_save_jsonl_report_includes_library_versions_in_metadata(tmp_path: Path) -> None:
    """Metadata header should preserve the shared library-version snapshot."""
    output_file = tmp_path / "results.jsonl"
    versions = cast(
        "check_models.LibraryVersionDict",
        {"mlx": "0.31.1", "mlx-vlm": "0.4.4", "transformers": "5.7.0"},
    )

    save_jsonl_report(
        [],
        output_file,
        prompt="test",
        system_info={},
        library_versions=versions,
    )

    header, rows = _read_jsonl(output_file)
    assert header.get("library_versions") == versions
    assert rows == []


def test_save_run_json_report_captures_public_snapshot_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run JSON should capture stable public snapshot metadata."""
    analysis = dataclasses.replace(
        check_models.analyze_generation_text(
            "Two cats on a pink couch.",
            generated_tokens=7,
            prompt_tokens=80,
            prompt="Describe this image briefly.",
        ),
        prompt_tokens_total=80,
        prompt_tokens_text_est=12,
        prompt_tokens_nontext_est=68,
    )
    result = PerformanceResult(
        model_name="org/caption-model",
        generation=MockGeneration(
            text="Two cats on a pink couch.",
            generation_tps=12.0,
            prompt_tokens=8,
            generation_tokens=7,
            peak_memory=1.5,
        ),
        success=True,
        generation_time=1.0,
        model_load_time=0.5,
        total_time=1.5,
        quality_analysis=analysis,
        prompt_diagnostics=check_models.PromptDiagnostics(
            processed_image_width=640,
            processed_image_height=480,
            image_patch_count=120,
            generate_kwargs={
                "max_tokens": 500,
                "temperature": 0.0,
                "prefill_step_size": 4096,
            },
        ),
    )
    out = tmp_path / "run.json"
    image_path = tmp_path / "catalogue.jpg"
    Image.new("RGB", (12, 8), "blue").save(image_path)
    context = check_models._build_report_render_context(
        results=[result],
        prompt="Describe this image briefly.",
        image_path=image_path,
        metadata={"description": ""},
        eval_mode="triage",
    )
    monkeypatch.setattr(
        check_models,
        "_collect_model_provenance",
        lambda model, requested_revision=None: {
            "model": model,
            "requested_revision": requested_revision,
            "resolved_revision": "snapshot123",
            "snapshot_path": "~/.cache/snapshots/snapshot123",
        },
    )

    check_models.save_run_json_report(
        [result],
        out,
        versions={"mlx-vlm": "0.6.3"},
        prompt="Describe this image briefly.",
        total_runtime_seconds=3.0,
        report_context=context,
        image_path=image_path,
        trust_remote_code=False,
        requested_revision="release-branch",
        output_paths={
            "output_index": "index.md",
            "results_html": "reports/results.html",
            "model_gallery": "reports/model_gallery.md",
            "diagnostics": "reports/diagnostics.md",
            "results_jsonl": "results.jsonl",
            "run_json": "run.json",
            "log": "check_models.log",
            "environment": "environment.log",
        },
        producer={
            "name": "check_models",
            "version": "0.8.6",
            "git_revision": "abc123",
            "install_type": "editable",
        },
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert set(payload) == {
        "schema_version",
        "generated_at",
        "eval_mode",
        "prompt",
        "prompt_sha256",
        "metadata_exposed_to_prompt",
        "total_runtime_seconds",
        "counts",
        "artifacts",
        "library_versions",
        "component_provenance",
        "producer",
        "image",
        "generation_settings",
        "trust_remote_code",
        "model_provenance",
        "prompt_burden",
    }
    assert payload["schema_version"] == "2.0"
    assert payload["eval_mode"] == "triage"
    assert "semantic_rankings_grounded" not in payload
    assert "selection_basis" not in payload
    assert "has_descriptive_metadata" not in payload
    assert payload["metadata_exposed_to_prompt"] is False
    assert payload["counts"] == {
        "models_attempted": 1,
        "models_evaluated": 1,
        "models_completed": 1,
        "models_crashed": 0,
        "models_indeterminate": 0,
    }
    assert payload["artifacts"] == {
        "output_index": "index.md",
        "results_html": "reports/results.html",
        "model_gallery": "reports/model_gallery.md",
        "diagnostics": "reports/diagnostics.md",
        "results_jsonl": "results.jsonl",
        "run_json": "run.json",
        "log": "check_models.log",
        "environment": "environment.log",
    }
    assert payload["library_versions"]["mlx-vlm"] == "0.6.3"
    assert payload["image"]["name"] == "catalogue.jpg"
    assert payload["image"]["width"] == 12
    assert payload["image"]["height"] == 8
    assert payload["image"]["sha256"]
    assert payload["image"]["size_bytes"] > 0
    assert payload["generation_settings"] == {
        "max_tokens": 500,
        "prefill_step_size": 4096,
        "temperature": 0.0,
    }
    assert payload["trust_remote_code"] is False
    assert payload["prompt_sha256"] == check_models._sha256_text("Describe this image briefly.")
    assert payload["model_provenance"][result.model_name] == {
        "model": result.model_name,
        "requested_revision": "release-branch",
        "resolved_revision": "snapshot123",
        "snapshot_path": "~/.cache/snapshots/snapshot123",
    }
    assert payload["prompt_burden"][result.model_name] == {
        "total_tokens": 80,
        "text_tokens_est": 12,
        "nontext_tokens_est": 68,
        "processed_image_width": 640,
        "processed_image_height": 480,
        "image_patch_count": 120,
    }
    assert payload["producer"] == {
        "name": "check_models",
        "version": "0.8.6",
        "git_revision": "abc123",
        "install_type": "editable",
    }


def test_run_json_counts_completed_crashed_and_indeterminate_results_consistently(
    tmp_path: Path,
) -> None:
    """Run counts should partition attempts while evaluated outcomes remain conclusive."""
    completed = PerformanceResult(model_name="org/completed", generation=None, success=True)
    crashed = PerformanceResult(
        model_name="org/crashed",
        generation=None,
        success=False,
        error_stage="Generation",
        error_message="decode failed",
    )
    disconnected = PerformanceResult(
        model_name="org/not-reached",
        generation=None,
        success=False,
        error_stage="Network Error",
        error_message="Model loading failed: Server disconnected without sending a response.",
        error_package="unknown",
    )
    results = [completed, crashed, disconnected]
    context = check_models._build_report_render_context(results=results, prompt="Describe it.")
    out = tmp_path / "run.json"

    check_models.save_run_json_report(
        results,
        out,
        versions={},
        prompt="Describe it.",
        total_runtime_seconds=2.0,
        report_context=context,
        output_paths={},
        producer={
            "name": "check_models",
            "version": "test",
            "git_revision": None,
            "install_type": "unknown",
        },
    )

    counts = json.loads(out.read_text(encoding="utf-8"))["counts"]
    assert counts == {
        "models_attempted": 3,
        "models_evaluated": 2,
        "models_completed": 1,
        "models_crashed": 1,
        "models_indeterminate": 1,
    }
    assert counts["models_attempted"] == (
        counts["models_completed"] + counts["models_crashed"] + counts["models_indeterminate"]
    )
    assert counts["models_evaluated"] == (counts["models_completed"] + counts["models_crashed"])


def test_check_models_provenance_degrades_without_install_or_git_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run metadata collection should remain usable outside an installed Git checkout."""

    def missing_version(_distribution_name: str) -> str:
        raise check_models.PackageNotFoundError

    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.setattr(check_models, "version", missing_version)
    monkeypatch.setattr(check_models, "_distribution_is_editable", lambda _name: False)
    monkeypatch.setattr(check_models, "_run_macos_toolchain_command", lambda _cmd: None)

    assert check_models._collect_check_models_provenance() == {
        "name": "check_models",
        "version": "unknown",
        "git_revision": None,
        "install_type": "unknown",
    }


def test_component_provenance_captures_editable_source_without_home_disclosure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Editable component metadata should retain source and revision safely."""
    monkeypatch.setattr(
        check_models,
        "_distribution_direct_url",
        lambda _name: {
            "url": "file:///Users/example/src/mlx-vlm",
            "dir_info": {"editable": True},
        },
    )
    monkeypatch.setattr(
        check_models,
        "_distribution_location",
        lambda _name: "/Users/example/miniconda/envs/mlx-vlm/lib/python3.13/site-packages",
    )
    monkeypatch.setattr(
        check_models,
        "_local_source_revision",
        lambda _path: "abc123",
    )
    monkeypatch.setattr(
        check_models.Path, "home", classmethod(lambda _cls: check_models.Path("/Users/example"))
    )

    provenance = check_models._collect_component_provenance({"mlx-vlm": "0.6.4"})

    assert provenance["mlx-vlm"] == {
        "version": "0.6.4",
        "install_type": "editable",
        "source_location": "~/src/mlx-vlm",
        "source_revision": "abc123",
        "direct_url": "file://~/src/mlx-vlm",
        "vcs_revision": None,
    }


def test_model_provenance_distinguishes_requested_and_resolved_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local snapshot identity should not be confused with the requested ref."""
    snapshot_sha = "0123456789abcdef"
    snapshot = check_models.Path(
        f"/Users/example/.cache/huggingface/hub/models--org--model/snapshots/{snapshot_sha}"
    )
    monkeypatch.setattr(check_models, "_resolve_model_snapshot_path", lambda _model: snapshot)
    monkeypatch.setattr(
        check_models.Path, "home", classmethod(lambda _cls: check_models.Path("/Users/example"))
    )

    provenance = check_models._collect_model_provenance(
        "org/model",
        requested_revision="main",
    )

    assert provenance == {
        "model": "org/model",
        "requested_revision": "main",
        "resolved_revision": snapshot_sha,
        "snapshot_path": ("~/.cache/huggingface/hub/models--org--model/snapshots/" + snapshot_sha),
    }


def test_jsonl_and_run_json_include_shared_component_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Primary machine artifacts should expose the same component identity payload."""
    components = {
        "mlx-vlm": {
            "version": "0.6.4",
            "install_type": "wheel",
            "source_location": "~/env/site-packages",
            "source_revision": None,
            "direct_url": None,
            "vcs_revision": None,
        }
    }
    monkeypatch.setattr(
        check_models, "_collect_component_provenance", lambda _versions=None: components
    )
    monkeypatch.setattr(
        check_models,
        "_collect_model_provenance",
        lambda model, requested_revision=None: {
            "model": model,
            "requested_revision": requested_revision,
            "resolved_revision": "snapshot123",
            "snapshot_path": "~/.cache/snapshots/snapshot123",
        },
    )
    result = PerformanceResult(model_name="org/model", generation=MockGeneration(), success=True)
    context = check_models._build_report_render_context(results=[result], prompt="describe")
    jsonl_path = tmp_path / "results.jsonl"
    run_path = tmp_path / "run.json"

    save_jsonl_report(
        [result],
        jsonl_path,
        prompt="describe",
        system_info={},
        library_versions={"mlx-vlm": "0.6.4"},
        requested_revision="release-branch",
        report_context=context,
    )
    check_models.save_run_json_report(
        [result],
        run_path,
        versions={"mlx-vlm": "0.6.4"},
        prompt="describe",
        total_runtime_seconds=1.0,
        report_context=context,
        output_paths={},
    )

    header, rows = _read_jsonl(jsonl_path)
    run_payload = json.loads(run_path.read_text(encoding="utf-8"))
    assert header["component_provenance"] == components
    assert run_payload["component_provenance"] == components
    assert rows[0]["model_provenance"]["resolved_revision"] == "snapshot123"
    assert rows[0]["model_provenance"]["requested_revision"] == "release-branch"


def test_jsonl_metrics_fall_back_to_generation_runtime_fields(tmp_path: Path) -> None:
    """JSONL metrics should use performance fields attached to GenerationResult."""
    result = PerformanceResult(
        model_name="fake/model",
        generation=MockGeneration(active_memory=0.75, cache_memory=0.25),
        success=True,
        active_memory=None,
        cache_memory=None,
        runtime_diagnostics=RuntimeDiagnostics(model_load_active_memory_gb=1.0),
    )
    output_file = tmp_path / "results.jsonl"
    save_jsonl_report([result], output_file, prompt="describe", system_info={})
    _header, rows = _read_jsonl(output_file)
    record = rows[0]

    metrics = record["metrics"]
    assert metrics["prompt_tokens"] == 10
    assert metrics["generation_tps"] == 5.0
    assert metrics["peak_memory_gb"] == 1.5
    assert metrics["active_memory_gb"] == 0.75
    assert metrics["cache_memory_gb"] == 0.25
    assert metrics["model_load_active_memory_gb"] == 1.0
    assert metrics["peak_memory_delta_gb"] == 0.5


def test_working_set_percentage_stays_in_current_run_jsonl(tmp_path: Path) -> None:
    """Derived working-set percentages belong in current-run JSONL, not raw history."""
    result = PerformanceResult(
        model_name="test-model",
        generation=MockGeneration(peak_memory=1.0),
        success=True,
    )
    context = check_models._build_report_render_context(
        results=[result],
        prompt="test",
        system_info={},
        recommended_working_set_bytes=2_000_000_000,
    )

    output_file = tmp_path / "working-set.jsonl"
    save_jsonl_report(
        [result],
        output_file,
        prompt="test",
        system_info={},
        report_context=context,
    )
    _header, rows = _read_jsonl(output_file)
    assert rows[0]["metrics"]["peak_memory_working_set_pct"] == 50.0

    history = append_history_record(
        history_path=tmp_path / "working-set.history.jsonl",
        results=[result],
        prompt="test",
        system_info={},
        library_versions={},
    )
    assert "peak_memory_working_set_pct" not in history["model_results"]["test-model"]


def test_missing_working_set_omits_jsonl_percentage(tmp_path: Path) -> None:
    """An unavailable denominator should not create a guessed structured fact."""
    result = PerformanceResult(
        model_name="test-model",
        generation=MockGeneration(peak_memory=1.0),
        success=True,
    )
    context = check_models._build_report_render_context(
        results=[result],
        prompt="test",
        system_info={},
        recommended_working_set_bytes=None,
    )

    output_file = tmp_path / "no-working-set.jsonl"
    save_jsonl_report(
        [result],
        output_file,
        prompt="test",
        system_info={},
        report_context=context,
    )
    _header, rows = _read_jsonl(output_file)
    assert "peak_memory_working_set_pct" not in rows[0]["metrics"]


def test_save_jsonl_report_content(tmp_path: Path) -> None:
    """Test that save_jsonl_report writes correct content with generation."""
    output_file = tmp_path / "results.jsonl"

    gen = MockGeneration(
        text="A detailed image description with enough words to be useful without caveats."
    )
    result = PerformanceResult(
        model_name="test-model",
        generation=gen,
        success=True,
        generation_time=1.5,
        model_load_time=0.5,
        total_time=2.0,
        runtime_diagnostics=RuntimeDiagnostics(
            input_validation_time_s=0.1,
            model_load_time_s=0.5,
            prompt_prep_time_s=0.2,
            decode_time_s=1.5,
            cleanup_time_s=0.05,
            first_token_latency_s=None,
            stop_reason="completed",
        ),
    )

    results = [result]
    save_jsonl_report(results, output_file, prompt="test", system_info={})

    assert output_file.exists()
    header, rows = _read_jsonl(output_file)
    assert header["_type"] == "metadata"
    assert len(rows) == 1

    data = rows[0]
    assert set(data) == {
        "_type",
        "model",
        "timestamp",
        "assessment",
        "generated_text",
        "captured_output_on_fail",
        "failure",
        "metrics",
        "timing",
        "model_provenance",
        "prompt_diagnostics",
    }
    assert data["_type"] == "result"
    assert data["model"] == "test-model"
    assert data["assessment"] == {
        "execution": "completed",
        "usability": "usable",
        "maintainer_status": "none",
        "observations": [],
    }
    assert data["generated_text"] == gen.text
    assert data["captured_output_on_fail"] == ""
    assert data["failure"] is None
    assert data["prompt_diagnostics"] is None
    metrics = data["metrics"]
    assert metrics.get("generation_tps") == 5.0
    assert metrics.get("prompt_tokens") == 10
    assert metrics.get("total_tokens") == 30
    assert metrics.get("prompt_tps") == 2.0
    timing = data["timing"]
    assert timing["input_validation_time_s"] == 0.1
    assert timing["prompt_prep_time_s"] == 0.2
    assert timing["cleanup_time_s"] == 0.05
    assert timing["stop_reason"] == "completed"


def test_save_jsonl_report_serializes_only_cached_result_assessment(tmp_path: Path) -> None:
    """Successful rows should expose one assessment without legacy status projections."""
    output_file = tmp_path / "results.jsonl"
    prompt = (
        "Analyze this image.\n"
        "Context: Existing metadata hints:\n"
        "- Title hint: Brick storefront with outdoor seating\n"
        "- Description hint: A brick storefront has outdoor seating beside a sidewalk.\n"
        "- Keyword hints: brick storefront, outdoor seating, sidewalk, people\n"
    )
    gen = MockGeneration(
        text=(
            "Title: Brick storefront with outdoor seating\n"
            "Description: A brick storefront has outdoor seating beside a sidewalk.\n"
            "Keywords: brick storefront, outdoor seating, sidewalk, people"
        ),
        prompt_tokens=320,
        generation_tokens=64,
    )
    analysis = check_models.analyze_generation_text(
        gen.text or "",
        generated_tokens=64,
        prompt_tokens=320,
        prompt=prompt,
        requested_max_tokens=128,
    )
    result = PerformanceResult(
        model_name="test-model",
        generation=gen,
        success=True,
        quality_analysis=analysis,
        requested_max_tokens=128,
    )

    save_jsonl_report([result], output_file, prompt=prompt, system_info={})

    _header, rows = _read_jsonl(output_file)
    row = rows[0]
    assert row["assessment"] == {
        "execution": "completed",
        "usability": "usable",
        "maintainer_status": "none",
        "observations": [],
    }
    assert "review" not in row
    assert "maintainer_triage" not in row
    assert "current_recommendation" not in row
    assert "compatibility_status" not in row


def test_save_jsonl_report_serializes_crash_assessment_and_failure(tmp_path: Path) -> None:
    """Failure rows should separate the assessment from raw failure evidence."""
    output_file = tmp_path / "results.jsonl"
    result = PerformanceResult(
        model_name="failed-model",
        generation=None,
        success=False,
        upstream_boundary="generation_started",
        error_message="runtime error",
        error_stage="Model Error",
        error_code="MLX_VLM_DECODE_RUNTIME",
        error_package="mlx-vlm",
    )

    save_jsonl_report([result], output_file, prompt="test", system_info={})

    _header, rows = _read_jsonl(output_file)
    row = rows[0]
    assert row["assessment"] == {
        "execution": "crashed",
        "usability": "not_evaluated",
        "maintainer_status": "actionable_failure",
        "observations": [],
    }
    assert row["failure"] == {
        "phase": None,
        "stage": "Model Error",
        "code": "MLX_VLM_DECODE_RUNTIME",
        "message": "runtime error",
        "exception_type": None,
        "exception_module": None,
        "package": "mlx-vlm",
        "traceback": None,
    }
    assert "review" not in row
    assert "maintainer_triage" not in row


def test_save_jsonl_report_omits_semantic_score_payloads(tmp_path: Path) -> None:
    """The narrow machine contract should not publish report-ranking scores."""
    output_file = tmp_path / "results.jsonl"
    result = PerformanceResult(
        model_name="test-model",
        generation=MockGeneration(),
        success=True,
    )

    save_jsonl_report([result], output_file, prompt="test", system_info={})

    _header, rows = _read_jsonl(output_file)
    row = rows[0]
    assert "metadata_agreement" not in row
    assert "quality_analysis" not in row
    assert "context_integration_score" not in row
    assert "draft_improvement_score" not in row


def test_save_jsonl_report_marks_external_connectivity_as_indeterminate(tmp_path: Path) -> None:
    """Transport failures should be recorded as indeterminate attempts."""
    output_file = tmp_path / "results.jsonl"
    result = PerformanceResult(
        model_name="failed-model",
        generation=None,
        success=False,
        error_message="Model loading failed: Server disconnected without sending a response.",
        error_stage="Model Error",
        error_code="HUGGINGFACE_HUB_MODEL_LOAD_MODEL",
        error_package="huggingface-hub",
        error_traceback=(
            "Traceback (most recent call last):\n"
            "httpx.RemoteProtocolError: Server disconnected without sending a response."
        ),
    )

    save_jsonl_report([result], output_file, prompt="test", system_info={})

    _header, rows = _read_jsonl(output_file)
    assert rows[0]["assessment"] == {
        "execution": "indeterminate",
        "usability": "not_evaluated",
        "maintainer_status": "none",
        "observations": [],
    }


def test_save_jsonl_report_no_generation(tmp_path: Path) -> None:
    """Test that save_jsonl_report handles missing generation."""
    output_file = tmp_path / "results.jsonl"

    result = PerformanceResult(
        model_name="test-model",
        generation=None,
        success=True,
        generation_time=1.5,
        model_load_time=0.5,
        total_time=2.0,
    )

    results = [result]
    save_jsonl_report(results, output_file, prompt="test", system_info={})

    _header, rows = _read_jsonl(output_file)
    data = rows[0]

    assert data["model"] == "test-model"
    assert "metrics" in data
    assert data["metrics"] == {}


def test_save_jsonl_report_failed_model(tmp_path: Path) -> None:
    """Test that save_jsonl_report handles failed models correctly."""
    output_file = tmp_path / "results.jsonl"

    result = PerformanceResult(
        model_name="failed-model",
        generation=None,
        success=False,
        error_message="Something went wrong",
        error_stage="Model Load",
    )

    results = [result]
    save_jsonl_report(results, output_file, prompt="test", system_info={})

    _header, rows = _read_jsonl(output_file)
    data = rows[0]

    assert data["model"] == "failed-model"
    assert data["failure"] == {
        "phase": None,
        "stage": "Model Load",
        "code": None,
        "message": "Something went wrong",
        "exception_type": None,
        "exception_module": None,
        "package": None,
        "traceback": None,
    }


def test_save_jsonl_report_includes_failure_phase_and_code(tmp_path: Path) -> None:
    """Failure metadata should remain nested raw evidence."""
    output_file = tmp_path / "results.jsonl"
    result = PerformanceResult(
        model_name="failed-model",
        generation=None,
        success=False,
        failure_phase="decode",
        error_stage="API Mismatch",
        error_code="TRANSFORMERS_DECODE_API_MISMATCH",
        error_signature="TRANSFORMERS_DECODE_API_MISMATCH:abc123",
        error_message="unexpected keyword argument",
    )
    save_jsonl_report([result], output_file, prompt="test", system_info={})

    _header, rows = _read_jsonl(output_file)
    failure = _require_present(rows[0]["failure"], field_name="failure")
    assert failure["phase"] == "decode"
    assert failure["code"] == "TRANSFORMERS_DECODE_API_MISMATCH"
    assert "error_signature" not in rows[0]


def test_save_jsonl_report_includes_traceback_and_type(tmp_path: Path) -> None:
    """Test that save_jsonl_report includes error_traceback and error_type for failures."""
    output_file = tmp_path / "results.jsonl"

    result = PerformanceResult(
        model_name="failed-model",
        generation=None,
        success=False,
        error_message="ValueError: Missing parameters",
        error_stage="Weight Mismatch",
        error_type="ValueError",
        error_package="mlx",
        error_traceback="Traceback (most recent call last):\n  File 'test.py', line 1\nValueError: Missing parameters",
    )

    results = [result]
    save_jsonl_report(results, output_file, prompt="test", system_info={})

    _header, rows = _read_jsonl(output_file)
    data = rows[0]

    assert data["model"] == "failed-model"
    failure = _require_present(data["failure"], field_name="failure")
    assert failure["exception_type"] == "ValueError"
    assert failure["package"] == "mlx"
    assert failure["traceback"] is not None
    assert "Traceback" in failure["traceback"]


def test_save_jsonl_report_includes_root_exception_fields(tmp_path: Path) -> None:
    """Optional root exception fields should serialize without changing error_type."""
    output_file = tmp_path / "results.jsonl"
    result = PerformanceResult(
        model_name="failed-model",
        generation=None,
        success=False,
        error_message="Model loading failed: upstream shape mismatch",
        error_type="ValueError",
        root_error_type="RuntimeError",
        root_error_module="builtins",
        root_error_message="upstream shape mismatch",
    )

    save_jsonl_report([result], output_file, prompt="test", system_info={})
    _header, rows = _read_jsonl(output_file)

    failure = _require_present(rows[0]["failure"], field_name="failure")
    assert failure["exception_type"] == "RuntimeError"
    assert failure["exception_module"] == "builtins"
    assert failure["message"] == "Model loading failed: upstream shape mismatch"


def test_save_jsonl_report_includes_exception_chain_in_chronological_order(
    tmp_path: Path,
) -> None:
    """Exception chains serialize additively from root cause to outer wrapper."""
    output_file = tmp_path / "results.jsonl"
    result = PerformanceResult(
        model_name="failed-model",
        generation=None,
        success=False,
        error_message="generation failed",
        exception_chain=(
            check_models.FailureException("IndexError", "builtins", "bad token"),
            check_models.FailureException("ValueError", "builtins", "generation failed"),
        ),
    )

    save_jsonl_report([result], output_file, prompt="test", system_info={})
    _header, rows = _read_jsonl(output_file)

    failure = _require_present(rows[0]["failure"], field_name="failure")
    assert failure.get("exception_chain") == [
        {"type": "IndexError", "module": "builtins", "message": "bad token"},
        {"type": "ValueError", "module": "builtins", "message": "generation failed"},
    ]


def test_save_jsonl_report_includes_prompt_diagnostics(tmp_path: Path) -> None:
    """Rendered prompt diagnostics should be optional JSONL metadata."""
    output_file = tmp_path / "results.jsonl"
    result = PerformanceResult(
        model_name="ok-model",
        generation=MockGeneration(),
        success=True,
        prompt_diagnostics=check_models.PromptDiagnostics(
            model_type="qwen2_vl",
            processor_class="transformers.AutoProcessor",
            tokenizer_class="transformers.PreTrainedTokenizerFast",
            rendered_prompt_hash_sha256="abc123",
            rendered_prompt_preview="<image> Describe this.",
            rendered_prompt_chars=22,
            image_placeholder_count=1,
            processed_image_width=512,
            processed_image_height=384,
            image_patch_count=4,
            eos_token_id=151645,
            special_token_ids=(151645,),
            special_tokens=("<|end|>",),
            generate_kwargs={
                "max_tokens": 500,
                "quantized_kv_start": check_models.DEFAULT_QUANTIZED_KV_START,
            },
        ),
    )

    save_jsonl_report([result], output_file, prompt="test", system_info={})
    _header, rows = _read_jsonl(output_file)

    prompt_diagnostics = _require_present(
        rows[0].get("prompt_diagnostics"),
        field_name="prompt_diagnostics",
    )
    assert prompt_diagnostics["rendered_prompt_hash_sha256"] == "abc123"
    assert prompt_diagnostics["image_placeholder_count"] == 1
    assert prompt_diagnostics["processed_image_width"] == 512
    assert prompt_diagnostics["processed_image_height"] == 384
    assert prompt_diagnostics["image_patch_count"] == 4
    assert prompt_diagnostics["special_tokens"] == ["<|end|>"]
    assert prompt_diagnostics["generate_kwargs"] == {
        "max_tokens": 500,
        "quantized_kv_start": check_models.DEFAULT_QUANTIZED_KV_START,
    }


def test_jsonl_does_not_back_project_legacy_machine_facts(tmp_path: Path) -> None:
    """Machine rows should expose the assessment without legacy report aliases."""
    output_file = tmp_path / "results.jsonl"
    prompt = "Create title, description, and keywords."
    analysis = check_models.analyze_generation_text(
        "Title: Cat\nDescription: A cat rests on a chair.\nKeywords: cat, chair",
        generated_tokens=18,
        prompt_tokens=4100,
        prompt=prompt,
    )
    result = PerformanceResult(
        model_name="org/enriched",
        generation=MockGeneration(
            text="Title: Cat\nDescription: A cat rests on a chair.\nKeywords: cat, chair",
            prompt_tokens=4100,
            generation_tokens=18,
        ),
        success=True,
        quality_analysis=analysis,
        prompt_diagnostics=check_models.PromptDiagnostics(image_placeholder_count=1),
    )
    context = check_models._build_report_render_context(
        results=[result],
        prompt=prompt,
        metadata={"description": "A cat rests on a chair."},
        eval_mode="assisted",
    )

    save_jsonl_report(
        [result],
        output_file,
        prompt=prompt,
        system_info={},
        eval_mode="assisted",
        metadata_exposed_to_prompt=True,
        report_context=context,
    )
    header, rows = _read_jsonl(output_file)
    row = rows[0]
    assert header["format_version"] == "2.0"
    assert row["assessment"]["execution"] == "completed"
    assert (
        not {
            "compatibility_status",
            "current_recommendation",
            "failure_origin",
            "maintainer_readiness",
            "reproduction_status",
            "keyword_overlap",
            "context_integration_score",
            "draft_improvement_score",
            "visual_description_score",
            "assisted_enrichment_score",
            "prompt_burden_kind",
            "prompt_burden_source",
            "owner_confidence",
        }
        & row.keys()
    )


def test_save_jsonl_report_includes_captured_output(tmp_path: Path) -> None:
    """Failure rows should retain captured stdout/stderr for diagnostics workflows."""
    output_file = tmp_path / "results.jsonl"

    result = PerformanceResult(
        model_name="failed-model",
        generation=None,
        success=False,
        error_message="runtime error",
        error_stage="Model Error",
        captured_output_on_fail="=== STDERR ===\nTokenizer warning",
    )

    save_jsonl_report([result], output_file, prompt="test", system_info={})

    _header, rows = _read_jsonl(output_file)
    data = rows[0]
    assert data["captured_output_on_fail"] == "=== STDERR ===\nTokenizer warning"


def test_save_jsonl_report_includes_timing(tmp_path: Path) -> None:
    """Test that save_jsonl_report includes timing information."""
    output_file = tmp_path / "results.jsonl"

    gen = MockGeneration()
    result = PerformanceResult(
        model_name="test-model",
        generation=gen,
        success=True,
        generation_time=2.5,
        model_load_time=1.0,
        total_time=3.5,
    )

    results = [result]
    save_jsonl_report(results, output_file, prompt="test", system_info={})

    _header, rows = _read_jsonl(output_file)
    data = rows[0]

    assert "timing" in data
    assert data["timing"]["generation_time_s"] == 2.5
    assert data["timing"]["model_load_time_s"] == 1.0
    assert data["timing"]["total_time_s"] == 3.5


def test_save_jsonl_report_round_trips_complete_generated_text(tmp_path: Path) -> None:
    """JSON escaping should preserve every captured output byte after decoding."""
    output_file = tmp_path / "results.jsonl"
    output = (
        "Title:\tCafé 雪\n"
        "```markdown\n**unchanged**\n```\n"
        "<think>HTML-looking, not markup</think>\n"
        "Final line\n"
    )
    gen = MockGeneration(text=output)
    result = PerformanceResult(
        model_name="test-model",
        generation=gen,
        success=True,
        generation_time=1.5,
        model_load_time=0.5,
        total_time=2.0,
    )

    results = [result]
    save_jsonl_report(results, output_file, prompt="test", system_info={})

    _header, rows = _read_jsonl(output_file)
    data = rows[0]

    assert "generated_text" in data
    assert data["generated_text"] == output


def test_save_jsonl_report_preserves_empty_generated_text(tmp_path: Path) -> None:
    """Empty generated text should still be serialized for diagnostics triage."""
    output_file = tmp_path / "results.jsonl"

    gen = MockGeneration(text="")
    result = PerformanceResult(
        model_name="test-model",
        generation=gen,
        success=True,
    )

    save_jsonl_report([result], output_file, prompt="test", system_info={})

    _header, rows = _read_jsonl(output_file)
    data = rows[0]
    assert "generated_text" in data
    assert data["generated_text"] == ""


def test_append_history_record_creates_file(tmp_path: Path) -> None:
    """Test that append_history_record writes a per-run history entry."""
    history_file = tmp_path / "results.history.jsonl"
    result = PerformanceResult(
        model_name="test-model",
        generation=None,
        success=True,
        generation_time=1.0,
        model_load_time=0.5,
        total_time=1.5,
    )

    append_history_record(
        history_path=history_file,
        results=[result],
        prompt="test prompt",
        system_info={"OS": "test"},
        library_versions={},
        image_path=None,
    )

    assert history_file.exists()
    lines = history_file.read_text().strip().split("\n")
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["_type"] == "run"
    assert record["model_results"]["test-model"]["success"] is True


def test_append_history_record_contains_only_raw_execution_and_resource_facts(
    tmp_path: Path,
) -> None:
    """History rows must not persist current semantic fields or recommendations."""
    history_file = tmp_path / "results.history.jsonl"
    prompt = (
        "Analyze this image.\n"
        "Context: Existing metadata hints:\n"
        "- Title hint: Brick storefront with outdoor seating\n"
        "- Description hint: A brick storefront has outdoor seating beside a sidewalk.\n"
        "- Keyword hints: brick storefront, outdoor seating, sidewalk, people\n"
    )
    gen = MockGeneration(
        text=(
            "Title: Brick storefront with outdoor seating\n"
            "Description: A brick storefront has outdoor seating beside a sidewalk.\n"
            "Keywords: brick storefront, outdoor seating, sidewalk, people"
        ),
        prompt_tokens=320,
        generation_tokens=64,
    )
    result = PerformanceResult(
        model_name="test-model",
        generation=gen,
        success=True,
        generation_time=1.25,
        model_load_time=0.5,
        total_time=1.75,
        requested_max_tokens=128,
    )

    record = append_history_record(
        history_path=history_file,
        results=[result],
        prompt=prompt,
        system_info={},
        library_versions={},
        image_path=None,
    )

    model_results = _require_present(record.get("model_results"), field_name="model_results")
    model_record = model_results["test-model"]
    assert model_record == {
        "success": True,
        "failure_phase": None,
        "error_stage": None,
        "error_type": None,
        "error_package": None,
        "error_code": None,
        "error_signature": None,
        "generation_time_s": 1.25,
        "model_load_time_s": 0.5,
        "total_time_s": 1.75,
        "prompt_tokens": 320,
        "generation_tokens": 64,
        "total_tokens": 30,
        "generation_tps": 5.0,
        "peak_memory_gb": 1.5,
        "active_memory_gb": 0.0,
        "cache_memory_gb": 0.0,
    }


def test_history_path_for_jsonl_derives_name(tmp_path: Path) -> None:
    """Test that history path inserts '.history' before '.jsonl'."""
    result = _history_path_for_jsonl(tmp_path / "results.jsonl")
    assert result == tmp_path / "results.history.jsonl"


def test_history_path_for_jsonl_custom_stem(tmp_path: Path) -> None:
    """Test history path derivation with a non-default stem."""
    result = _history_path_for_jsonl(tmp_path / "my_output.jsonl")
    assert result == tmp_path / "my_output.history.jsonl"


# ---------------------------------------------------------------------------
# --- Runtime Fingerprint Canary Tests ---


class TestRuntimeFingerprint:
    """Mock canary tests for runtime capability fingerprint collection."""

    def test_collect_runtime_fingerprint_returns_all_probes(self) -> None:
        """Fingerprint must include every probe key (G2: never silently omit)."""
        fingerprint = check_models.collect_runtime_fingerprint()
        expected_probes = {
            "metal_gpu",
            "mlx_framework",
            "mlx_vlm",
            "gpu_memory",
            "fused_attention",
        }
        assert set(fingerprint.keys()) == expected_probes

    def test_each_probe_has_valid_status(self) -> None:
        """Every probe result must have a status in the allowed set."""
        fingerprint = check_models.collect_runtime_fingerprint()
        valid_statuses = {"ok", "unavailable", "errored", "timed_out"}
        for probe_name, result in fingerprint.items():
            assert result["status"] in valid_statuses, (
                f"Probe '{probe_name}' has invalid status: {result['status']}"
            )

    def test_collect_runtime_fingerprint_reports_mlx_vlm_available(self) -> None:
        """An imported mlx-vlm runtime should be recorded as available."""
        with patch.dict(check_models.MISSING_DEPENDENCIES, {}, clear=True):
            fingerprint = check_models.collect_runtime_fingerprint()

        assert fingerprint["mlx_vlm"] == {"status": "ok"}

    def test_collect_runtime_fingerprint_reports_mlx_vlm_unavailable(self) -> None:
        """A captured mlx-vlm import failure should remain actionable."""
        with patch.dict(
            check_models.MISSING_DEPENDENCIES,
            {"mlx-vlm": "not imported"},
            clear=True,
        ):
            fingerprint = check_models.collect_runtime_fingerprint()

        assert fingerprint["mlx_vlm"] == {
            "status": "unavailable",
            "detail": "not imported",
        }

    def test_collect_runtime_fingerprint_uses_top_level_mlx_memory_probe(self) -> None:
        """GPU memory probe should use the current top-level MLX memory API."""

        class _FakeMxRuntime:
            @staticmethod
            def get_active_memory() -> float:
                return 2 * check_models.DECIMAL_GB

        with patch.object(check_models, "mx", _FakeMxRuntime()):
            fingerprint = check_models.collect_runtime_fingerprint()

        assert fingerprint["gpu_memory"]["status"] == "ok"
        assert fingerprint["gpu_memory"].get("detail") == "active=2.00GB"

    def test_collect_runtime_fingerprint_reports_fused_attention_available(self) -> None:
        """Callable MLX fused attention should be recorded as available."""
        runtime = SimpleNamespace(
            fast=SimpleNamespace(scaled_dot_product_attention=lambda: None),
        )
        with patch.object(check_models, "mx", runtime):
            fingerprint = check_models.collect_runtime_fingerprint()

        assert fingerprint["fused_attention"] == {"status": "ok"}

    def test_collect_runtime_fingerprint_reports_fused_attention_unavailable(self) -> None:
        """A missing fused-attention surface should remain explicit."""
        with patch.object(check_models, "mx", SimpleNamespace()):
            fingerprint = check_models.collect_runtime_fingerprint()

        assert fingerprint["fused_attention"]["status"] == "unavailable"

    def test_probe_fused_attention_reports_attribute_error(self) -> None:
        """Runtime attribute errors should become bounded probe state."""

        class RaisingRuntime:
            @property
            def fast(self) -> object:
                message = "runtime unavailable"
                raise RuntimeError(message)

        with patch.object(check_models, "mx", RaisingRuntime()):
            result = check_models._probe_fused_attention()

        assert result == {"status": "errored", "detail": "runtime unavailable"}

    def test_jsonl_metadata_includes_fingerprint(self) -> None:
        """JSONL metadata record includes runtime_fingerprint when provided."""
        fingerprint = {"metal_gpu": check_models.RuntimeProbeResult(status="ok")}
        record = check_models._build_jsonl_metadata_record(
            prompt="test",
            system_info={},
            runtime_fingerprint=fingerprint,
        )
        assert "runtime_fingerprint" in record
        runtime_fingerprint = _require_present(
            record.get("runtime_fingerprint"),
            field_name="runtime_fingerprint",
        )
        assert runtime_fingerprint["metal_gpu"]["status"] == "ok"

    def test_jsonl_metadata_omits_fingerprint_when_none(self) -> None:
        """JSONL metadata record omits runtime_fingerprint when not provided."""
        record = check_models._build_jsonl_metadata_record(
            prompt="test",
            system_info={},
        )
        assert "runtime_fingerprint" not in record

    def test_history_record_includes_fingerprint(self, tmp_path: Path) -> None:
        """History record includes runtime_fingerprint when provided."""
        fingerprint = {"mlx_vlm": check_models.RuntimeProbeResult(status="ok")}
        history_path = tmp_path / "test.history.jsonl"
        record = check_models.append_history_record(
            history_path=history_path,
            results=[],
            prompt="test prompt",
            system_info={},
            library_versions=cast("check_models.LibraryVersionDict", {}),
            runtime_fingerprint=fingerprint,
        )
        assert record.get("runtime_fingerprint") == fingerprint
        # Verify it's persisted to disk
        lines = history_path.read_text().strip().splitlines()
        assert len(lines) == 1
        persisted = json.loads(lines[0])
        assert persisted["runtime_fingerprint"]["mlx_vlm"]["status"] == "ok"

    def test_save_jsonl_includes_fingerprint(self, tmp_path: Path) -> None:
        """save_jsonl_report includes fingerprint in metadata header."""
        fingerprint = {"metal_gpu": check_models.RuntimeProbeResult(status="ok")}
        out_path = tmp_path / "results.jsonl"
        check_models.save_jsonl_report(
            [],
            out_path,
            prompt="test",
            system_info={},
            runtime_fingerprint=fingerprint,
        )
        lines = out_path.read_text().strip().splitlines()
        header = json.loads(lines[0])
        assert header["_type"] == "metadata"
        assert header["runtime_fingerprint"]["metal_gpu"]["status"] == "ok"


class TestSchemaVersioning:
    """Tests for JSONL schema versioning and round-trip integrity."""

    def test_metadata_format_version_is_2_0(self, tmp_path: Path) -> None:
        """Current JSONL output uses the narrow 2.0 machine contract."""
        out = tmp_path / "results.jsonl"
        check_models.save_jsonl_report([], out, prompt="test", system_info={})
        header, _ = _read_jsonl(out)
        assert header["format_version"] == "2.0"

    def test_round_trip_metadata_keys(self, tmp_path: Path) -> None:
        """Metadata record round-trips through JSON with expected keys."""
        fingerprint = {"metal_gpu": check_models.RuntimeProbeResult(status="ok")}
        out = tmp_path / "results.jsonl"
        check_models.save_jsonl_report(
            [],
            out,
            prompt="hello",
            system_info={"os": "macOS"},
            runtime_fingerprint=fingerprint,
        )
        header, _ = _read_jsonl(out)
        assert header["_type"] == "metadata"
        assert header["prompt"] == "hello"
        assert header["system"]["os"] == "macOS"
        assert "timestamp" in header
        runtime_fingerprint = _require_present(
            header.get("runtime_fingerprint"),
            field_name="runtime_fingerprint",
        )
        assert runtime_fingerprint["metal_gpu"]["status"] == "ok"

    def test_round_trip_result_record_success(self, tmp_path: Path) -> None:
        """Successful result record round-trips with all required keys."""
        result = PerformanceResult(
            model_name="org/good",
            generation=MockGeneration(),
            success=True,
        )
        out = tmp_path / "results.jsonl"
        check_models.save_jsonl_report([result], out, prompt="t", system_info={})
        _, rows = _read_jsonl(out)
        row = rows[0]
        assert row["_type"] == "result"
        assert row["model"] == "org/good"
        assert row["assessment"]["execution"] == "completed"
        assert row["failure"] is None

    def test_round_trip_result_record_failure(self, tmp_path: Path) -> None:
        """Failed result record round-trips with nested raw failure evidence."""
        result = PerformanceResult(
            model_name="org/bad",
            generation=None,
            success=False,
            error_message="ValueError: bad shape",
            error_code="DECODE_ERR",
            error_traceback="File x.py line 1\n  raise ValueError",
        )
        out = tmp_path / "results.jsonl"
        check_models.save_jsonl_report([result], out, prompt="t", system_info={})
        _, rows = _read_jsonl(out)
        row = rows[0]
        assert row["assessment"]["execution"] == "crashed"
        failure = _require_present(row["failure"], field_name="failure")
        assert failure["code"] == "DECODE_ERR"
        assert failure["traceback"] == "File x.py line 1\n  raise ValueError"

    def test_round_trip_all_fields_json_serializable(self, tmp_path: Path) -> None:
        """Every field in the JSONL output is JSON-serializable (no crash)."""
        result = PerformanceResult(
            model_name="org/model",
            generation=MockGeneration(),
            success=True,
            runtime_diagnostics=RuntimeDiagnostics(),
        )
        out = tmp_path / "results.jsonl"
        check_models.save_jsonl_report([result], out, prompt="p", system_info={})
        # Re-parse every line — will raise if any field isn't serializable
        for line in out.read_text().strip().splitlines():
            parsed = json.loads(line)
            json.dumps(parsed)  # round-trip back to string

    def test_history_format_version_unchanged(self, tmp_path: Path) -> None:
        """History records keep format_version 1.0 (separate schema)."""
        hist = tmp_path / "results.history.jsonl"
        check_models.append_history_record(
            results=[],
            prompt="t",
            image_path=None,
            system_info={},
            library_versions={},
            history_path=hist,
            eval_mode="blind",
        )
        data = json.loads(hist.read_text().strip())
        assert data["format_version"] == "1.0"
        assert data["eval_mode"] == "blind"

    def test_legacy_mode_is_resolved_before_history_persistence(self, tmp_path: Path) -> None:
        """Compatibility aliases should never appear as stored lane identities."""
        hist = tmp_path / "results.history.jsonl"

        check_models.append_history_record(
            results=[],
            prompt="t",
            image_path=None,
            system_info={},
            library_versions={},
            history_path=hist,
            eval_mode="stress",
        )

        data = json.loads(hist.read_text().strip())
        assert data["eval_mode"] == "blind"


class TestRerunEvidence:
    """Tests for differential rerun evidence in JSONL output."""

    def test_select_rerun_candidates_picks_failures(self) -> None:
        """_select_rerun_candidates picks failed models without verdicts."""
        ok = PerformanceResult(model_name="ok", generation=MockGeneration(), success=True)
        fail = PerformanceResult(model_name="fail", generation=None, success=False)
        candidates = check_models._select_rerun_candidates([ok, fail])
        assert len(candidates) == 1
        assert candidates[0].model_name == "fail"

    def test_select_rerun_candidates_skips_deterministic_verdicts(self) -> None:
        """Models with harness/model_shortcoming verdicts are not rerun candidates."""
        # Create a mock quality_analysis with verdict="harness"
        qa = MagicMock()
        qa.verdict = "harness"
        result = PerformanceResult(
            model_name="harness-model",
            generation=MockGeneration(),
            success=True,
            quality_analysis=qa,
        )
        candidates = check_models._select_rerun_candidates([result])
        assert len(candidates) == 0
