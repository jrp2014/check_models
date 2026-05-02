"""Tests for JSONL output generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

import check_models
from check_models import (
    JsonlMetadataRecord,
    JsonlResultRecord,
    MetadataAgreementMetrics,
    PerformanceResult,
    RuntimeDiagnostics,
    _history_path_for_jsonl,
    _load_latest_history_record,
    append_history_record,
    compare_history_records,
    save_jsonl_report,
)

if TYPE_CHECKING:
    from pathlib import Path

    from check_models import HistoryModelResultRecord, HistoryRunRecord


def _read_jsonl(path: Path) -> tuple[JsonlMetadataRecord, list[JsonlResultRecord]]:
    """Read JSONL file returning (metadata_header, result_rows)."""
    lines = path.read_text().strip().split("\n")
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


def _history_run(
    model_success: dict[str, bool],
    *,
    timestamp: str = "2026-01-01 00:00:00 GMT",
) -> HistoryRunRecord:
    """Build a fully shaped history run record for typed history tests."""
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


def test_save_jsonl_report_creates_file(tmp_path: Path) -> None:
    """Test that save_jsonl_report creates a file with metadata header."""
    output_file = tmp_path / "results.jsonl"
    results: list[PerformanceResult] = []
    save_jsonl_report(results, output_file, prompt="test", system_info={})

    assert output_file.exists()
    header, rows = _read_jsonl(output_file)
    assert header["_type"] == "metadata"
    assert header["format_version"] == "2.0"
    assert header["prompt"] == "test"
    assert rows == []


def test_save_jsonl_report_includes_library_versions_in_metadata(tmp_path: Path) -> None:
    """Metadata header should preserve the shared library-version snapshot."""
    output_file = tmp_path / "results.jsonl"
    versions = cast(
        "check_models.LibraryVersionDict",
        {"mlx": "0.31.1", "mlx-vlm": "0.4.4", "transformers": "5.5.3"},
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


def test_save_jsonl_report_content(tmp_path: Path) -> None:
    """Test that save_jsonl_report writes correct content with generation."""
    output_file = tmp_path / "results.jsonl"

    gen = MockGeneration()
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
    assert data["_type"] == "result"
    assert data["model"] == "test-model"
    assert data["success"] is True
    assert data["failure_phase"] is None
    assert data["error_code"] is None
    assert data["error_signature"] is None
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


def test_save_jsonl_report_includes_review_payload_for_success(tmp_path: Path) -> None:
    """Successful rows should include the canonical automated review payload."""
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
    review = _require_present(rows[0].get("review"), field_name="review")
    triage = _require_present(rows[0].get("maintainer_triage"), field_name="maintainer_triage")
    assert review["verdict"] in {"clean", "model_shortcoming", "context_budget"}
    assert review["hint_relationship"] in {
        "improves_trusted_hints",
        "preserves_trusted_hints",
        "degrades_trusted_hints",
        "ignores_trusted_hints",
    }
    assert review["requested_max_tokens"] == 128
    assert review["prompt_tokens_total"] == 320
    assert review["prompt_tokens_text_est"] is not None
    assert review["prompt_tokens_nontext_est"] is not None
    assert triage["issue_kind"] == review["verdict"]
    assert triage["suspected_owner"] == review["owner"]
    assert triage["user_bucket"] == review["user_bucket"]
    assert triage["summary"]
    assert "issue_cluster_id" not in triage
    assert "priority" not in json.dumps(triage).casefold()
    if review["verdict"] == "clean":
        assert triage["next_action"] == "No immediate maintainer action."


def test_save_jsonl_report_includes_review_payload_for_failures(tmp_path: Path) -> None:
    """Failure rows should still include owner and verdict review fields."""
    output_file = tmp_path / "results.jsonl"
    result = PerformanceResult(
        model_name="failed-model",
        generation=None,
        success=False,
        error_message="runtime error",
        error_stage="Model Error",
        error_code="MLX_VLM_DECODE_RUNTIME",
        error_package="mlx-vlm",
    )

    save_jsonl_report([result], output_file, prompt="test", system_info={})

    _header, rows = _read_jsonl(output_file)
    review = _require_present(rows[0].get("review"), field_name="review")
    triage = _require_present(rows[0].get("maintainer_triage"), field_name="maintainer_triage")
    assert review["verdict"] == "runtime_failure"
    assert review["owner"] == "mlx-vlm"
    assert review["user_bucket"] == "avoid"
    assert review["evidence"]
    assert triage["issue_kind"] == "runtime_failure"
    assert triage.get("issue_subtype") == "MLX_VLM_DECODE_RUNTIME"
    assert triage.get("issue_cluster_id") == "mlx-vlm_mlx-vlm-decode-runtime_001"
    issue_cluster_path = _require_present(
        triage.get("issue_cluster_path"),
        field_name="issue_cluster_path",
    )
    assert issue_cluster_path.startswith("issues/issue_001_")
    assert triage.get("acceptance_signal")
    assert triage["confidence"] == "high"
    assert triage["suspected_owner"] == "mlx-vlm"
    assert "Inspect prompt-template" in triage["next_action"]
    assert "priority" not in json.dumps(triage).casefold()


def test_save_jsonl_report_includes_metadata_agreement_payload(tmp_path: Path) -> None:
    """Successful JSONL rows should carry metadata-agreement benchmark fields."""
    output_file = tmp_path / "results.jsonl"
    result = PerformanceResult(
        model_name="test-model",
        generation=MockGeneration(),
        success=True,
        metadata_agreement=MetadataAgreementMetrics(
            overall_score=82.5,
            title_score=100.0,
            description_score=75.0,
            keyword_score=80.0,
            nonvisual_penalty=6.7,
            matched_terms=("brick", "storefront", "outdoor seating"),
            missed_terms=("pedestrians",),
            nonvisual_hits=("51.5000,-0.1200",),
        ),
    )

    save_jsonl_report([result], output_file, prompt="test", system_info={})

    _header, rows = _read_jsonl(output_file)
    payload = _require_present(
        rows[0].get("metadata_agreement"),
        field_name="metadata_agreement",
    )
    assert payload["overall_score"] == 82.5
    assert payload["title_score"] == 100.0
    assert payload["description_score"] == 75.0
    assert payload["keyword_score"] == 80.0
    assert payload["nonvisual_penalty"] == 6.7
    assert payload["matched_terms"] == ["brick", "storefront", "outdoor seating"]
    assert payload["missed_terms"] == ["pedestrians"]
    assert payload["nonvisual_hits"] == ["51.5000,-0.1200"]


def test_save_jsonl_report_flags_huggingface_hub_connectivity_failures(tmp_path: Path) -> None:
    """Hub transport failures should carry a connectivity-oriented review hint."""
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
    review = _require_present(rows[0].get("review"), field_name="review")
    assert review["verdict"] == "runtime_failure"
    assert review["owner"] == "huggingface-hub"
    assert "hub_connectivity" in review["evidence"]


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
    assert data["success"] is False
    assert data["failure_phase"] is None
    assert data["error_message"] == "Something went wrong"
    assert data["error_stage"] == "Model Load"


def test_save_jsonl_report_includes_phase_code_and_signature(tmp_path: Path) -> None:
    """Failure metadata fields should serialize for diagnostics tooling."""
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
    data = rows[0]
    assert data["failure_phase"] == "decode"
    assert data["error_code"] == "TRANSFORMERS_DECODE_API_MISMATCH"
    assert data["error_signature"] == "TRANSFORMERS_DECODE_API_MISMATCH:abc123"


def test_save_jsonl_report_quality_issues_as_list(tmp_path: Path) -> None:
    """Test that quality_issues is saved as a list of strings in JSONL."""
    output_file = tmp_path / "results.jsonl"

    gen = MockGeneration()
    result = PerformanceResult(
        model_name="test-model",
        generation=gen,
        success=True,
        quality_issues="repetitive(<s>), verbose, formatting",
        generation_time=1.5,
        model_load_time=0.5,
        total_time=2.0,
    )

    results = [result]
    save_jsonl_report(results, output_file, prompt="test", system_info={})

    _header, rows = _read_jsonl(output_file)
    data = rows[0]

    assert data["model"] == "test-model"
    assert isinstance(data["quality_issues"], list)
    assert data["quality_issues"] == ["repetitive(<s>)", "verbose", "formatting"]


def test_save_jsonl_report_quality_issues_with_internal_commas(tmp_path: Path) -> None:
    """Commas inside one issue item (e.g., phrase preview) should not split that item."""
    output_file = tmp_path / "results.jsonl"

    gen = MockGeneration()
    result = PerformanceResult(
        model_name="test-model",
        generation=gen,
        success=True,
        quality_issues='repetitive(phrase: "a, b..."), context-echo(0.91)',
        generation_time=1.5,
        model_load_time=0.5,
        total_time=2.0,
    )

    save_jsonl_report([result], output_file, prompt="test", system_info={})

    _header, rows = _read_jsonl(output_file)
    data = rows[0]

    assert data["quality_issues"] == ['repetitive(phrase: "a, b...")', "context-echo(0.91)"]


def test_save_jsonl_report_no_quality_issues(tmp_path: Path) -> None:
    """Test that quality_issues is an empty list when None."""
    output_file = tmp_path / "results.jsonl"

    gen = MockGeneration()
    result = PerformanceResult(
        model_name="test-model",
        generation=gen,
        success=True,
        quality_issues=None,
        generation_time=1.5,
        model_load_time=0.5,
        total_time=2.0,
    )

    results = [result]
    save_jsonl_report(results, output_file, prompt="test", system_info={})

    _header, rows = _read_jsonl(output_file)
    data = rows[0]

    assert data["model"] == "test-model"
    assert data["quality_issues"] == []


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
    assert data["success"] is False
    assert data["error_type"] == "ValueError"
    assert data["error_package"] == "mlx"
    assert data["error_traceback"] is not None
    assert "Traceback" in data["error_traceback"]


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

    data = rows[0]
    assert data["error_type"] == "ValueError"
    assert data.get("root_error_type") == "RuntimeError"
    assert data.get("root_error_module") == "builtins"
    assert data.get("root_error_message") == "upstream shape mismatch"


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
    assert prompt_diagnostics["special_tokens"] == ["<|end|>"]
    assert prompt_diagnostics["generate_kwargs"] == {
        "max_tokens": 500,
        "quantized_kv_start": check_models.DEFAULT_QUANTIZED_KV_START,
    }


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


def test_save_jsonl_report_includes_generated_text(tmp_path: Path) -> None:
    """Test that save_jsonl_report includes generated_text for successful models."""
    output_file = tmp_path / "results.jsonl"

    gen = MockGeneration(text="This is the generated output text.")
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
    assert data["generated_text"] == "This is the generated output text."


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


def test_append_history_record_captures_review_fields_for_quality_tracking(
    tmp_path: Path,
) -> None:
    """History rows should keep enough review context for non-binary comparisons."""
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
    assert model_record.get("review_verdict") == analysis.verdict
    assert model_record.get("review_owner") == analysis.owner
    assert model_record.get("review_user_bucket") == analysis.user_bucket
    assert model_record.get("prompt_output_ratio") == 64 / 320


def test_compare_history_records_detects_regressions_and_recoveries() -> None:
    """Test regression/recovery detection between history records."""
    previous = _history_run(
        {
            "model-a": True,
            "model-b": False,
            "model-c": True,
        },
    )
    current = _history_run(
        {
            "model-a": False,
            "model-b": True,
            "model-d": True,
        },
        timestamp="2026-01-02 00:00:00 GMT",
    )

    summary = compare_history_records(previous, current)
    assert summary["regressions"] == ["model-a"]
    assert summary["recoveries"] == ["model-b"]
    assert summary["new_models"] == ["model-d"]
    assert summary["missing_models"] == ["model-c"]


def test_compare_history_records_detects_quality_harness_and_owner_changes() -> None:
    """Stable-success models should still report maintainer-relevant quality shifts."""
    previous = _history_run({"model-a": True, "model-b": True})
    current = _history_run({"model-a": True, "model-b": True})

    previous_model_results = _require_present(
        previous.get("model_results"),
        field_name="model_results",
    )
    current_model_results = _require_present(
        current.get("model_results"),
        field_name="model_results",
    )

    previous_model_results["model-a"]["review_user_bucket"] = "recommended"
    previous_model_results["model-a"]["review_verdict"] = "clean"
    previous_model_results["model-a"]["review_owner"] = "model"

    current_model_results["model-a"]["review_user_bucket"] = "avoid"
    current_model_results["model-a"]["review_verdict"] = "context_budget"
    current_model_results["model-a"]["review_owner"] = "mlx"
    current_model_results["model-a"]["harness_issue_type"] = "long_context"

    previous_model_results["model-b"]["review_user_bucket"] = "avoid"
    previous_model_results["model-b"]["review_verdict"] = "harness"
    previous_model_results["model-b"]["review_owner"] = "mlx-vlm"
    previous_model_results["model-b"]["harness_issue_type"] = "token_leak"

    current_model_results["model-b"]["review_user_bucket"] = "recommended"
    current_model_results["model-b"]["review_verdict"] = "clean"
    current_model_results["model-b"]["review_owner"] = "model"

    summary = compare_history_records(previous, current)
    assert summary["quality_regressions"] == ["model-a"]
    assert summary["quality_recoveries"] == ["model-b"]
    assert summary["harness_regressions"] == ["model-a"]
    assert summary["harness_recoveries"] == ["model-b"]
    assert summary["owner_changes"] == ["model-a", "model-b"]


# ---------------------------------------------------------------------------
# _history_path_for_jsonl
# ---------------------------------------------------------------------------


def test_history_path_for_jsonl_derives_name(tmp_path: Path) -> None:
    """Test that history path inserts '.history' before '.jsonl'."""
    result = _history_path_for_jsonl(tmp_path / "results.jsonl")
    assert result == tmp_path / "results.history.jsonl"


def test_history_path_for_jsonl_custom_stem(tmp_path: Path) -> None:
    """Test history path derivation with a non-default stem."""
    result = _history_path_for_jsonl(tmp_path / "my_output.jsonl")
    assert result == tmp_path / "my_output.history.jsonl"


# ---------------------------------------------------------------------------
# _load_latest_history_record
# ---------------------------------------------------------------------------


def test_load_latest_history_record_missing_file(tmp_path: Path) -> None:
    """Return None when the history file does not exist."""
    assert _load_latest_history_record(tmp_path / "missing.jsonl") is None


def test_load_latest_history_record_empty_file(tmp_path: Path) -> None:
    """Return None when the history file is empty."""
    history = tmp_path / "empty.jsonl"
    history.write_text("")
    assert _load_latest_history_record(history) is None


def test_load_latest_history_record_only_blank_lines(tmp_path: Path) -> None:
    """Return None when the file contains only blank lines."""
    history = tmp_path / "blanks.jsonl"
    history.write_text("\n\n\n")
    assert _load_latest_history_record(history) is None


def test_load_latest_history_record_corrupted_lines(tmp_path: Path) -> None:
    """Skip corrupted lines and return the valid record."""
    history = tmp_path / "mixed.jsonl"
    valid = json.dumps(_history_run({"m": True}))
    history.write_text(f'{valid}\nNOT-JSON\n{{"bad": true}}\n')

    record = _load_latest_history_record(history)
    assert record is not None
    assert record.get("_type") == "run"
    assert "model_results" in record
    model_results = record["model_results"]
    assert model_results["m"]["success"] is True


def test_load_latest_history_record_only_corrupted(tmp_path: Path) -> None:
    """Return None when every line is invalid JSON."""
    history = tmp_path / "corrupt.jsonl"
    history.write_text("NOT-JSON-1\nNOT-JSON-2\n")
    assert _load_latest_history_record(history) is None


def test_load_latest_history_record_returns_last_run(tmp_path: Path) -> None:
    """When multiple run records exist, return the last one."""
    history = tmp_path / "multi.jsonl"
    first = json.dumps(_history_run({"m": True}, timestamp="2026-01-01 00:00:00 GMT"))
    second = json.dumps(_history_run({"m": False}, timestamp="2026-01-02 00:00:00 GMT"))
    history.write_text(f"{first}\n{second}\n")

    record = _load_latest_history_record(history)
    assert record is not None
    assert record.get("timestamp") == "2026-01-02 00:00:00 GMT"


def test_load_latest_history_record_skips_non_run_types(tmp_path: Path) -> None:
    """Skip records whose _type is not 'run'."""
    history = tmp_path / "types.jsonl"
    run_record = _history_run({"m": True}, timestamp="2026-01-03 00:00:00 GMT")
    metadata = json.dumps({"_type": "metadata", "info": "x"})
    history.write_text(f"{json.dumps(run_record)}\n{metadata}\n")

    record = _load_latest_history_record(history)
    assert record is not None
    assert record.get("_type") == "run"
    assert record.get("timestamp") == "2026-01-03 00:00:00 GMT"


# ---------------------------------------------------------------------------
# compare_history_records — baseline (no previous)
# ---------------------------------------------------------------------------


def test_compare_history_records_no_previous() -> None:
    """With no previous record, all current models are 'new'."""
    current = _history_run({"model-x": True, "model-y": False})

    summary = compare_history_records(None, current)
    assert summary["regressions"] == []
    assert summary["recoveries"] == []
    assert summary["quality_regressions"] == []
    assert summary["quality_recoveries"] == []
    assert summary["harness_regressions"] == []
    assert summary["harness_recoveries"] == []
    assert summary["owner_changes"] == []
    assert summary["new_models"] == ["model-x", "model-y"]
    assert summary["missing_models"] == []


# --- Runtime Fingerprint Canary Tests ---


class TestRuntimeFingerprint:
    """Mock canary tests for runtime capability fingerprint collection."""

    def test_collect_runtime_fingerprint_returns_all_probes(self) -> None:
        """Fingerprint must include all four probe keys (G2: never silently omit)."""
        fingerprint = check_models.collect_runtime_fingerprint()
        expected_probes = {"metal_gpu", "mlx_framework", "mlx_vlm", "gpu_memory"}
        assert set(fingerprint.keys()) == expected_probes

    def test_each_probe_has_valid_status(self) -> None:
        """Every probe result must have a status in the allowed set."""
        fingerprint = check_models.collect_runtime_fingerprint()
        valid_statuses = {"ok", "unavailable", "errored", "timed_out"}
        for probe_name, result in fingerprint.items():
            assert result["status"] in valid_statuses, (
                f"Probe '{probe_name}' has invalid status: {result['status']}"
            )

    def test_collect_runtime_fingerprint_uses_top_level_mlx_memory_probe(self) -> None:
        """GPU memory probe should use the current top-level MLX memory API."""

        class _FakeMxRuntime:
            @staticmethod
            def get_active_memory() -> float:
                return float(2 * 1024**3)

        with patch.object(check_models, "mx", _FakeMxRuntime()):
            fingerprint = check_models.collect_runtime_fingerprint()

        assert fingerprint["gpu_memory"]["status"] == "ok"
        assert fingerprint["gpu_memory"].get("detail") == "active=2.00GB"

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


class TestSignatureComponents:
    """Tests for structured diagnostic signature components in JSONL output."""

    def test_failed_result_includes_signature_components(self, tmp_path: Path) -> None:
        """Failed results emit signature_components with normalized fields."""
        result = PerformanceResult(
            model_name="org/broken",
            generation=None,
            success=False,
            error_message="RuntimeError: shape mismatch [4, 128] vs [4, 256]",
            error_code="MLX_DECODE_ERROR",
            error_traceback="File model.py line 42\n  raise RuntimeError",
        )
        out = tmp_path / "results.jsonl"
        check_models.save_jsonl_report([result], out, prompt="test", system_info={})
        _, rows = _read_jsonl(out)
        assert len(rows) == 1
        components = rows[0].get("signature_components")
        assert components is not None
        assert components.get("error_code") == "MLX_DECODE_ERROR"
        assert "normalized_message" in components
        assert "traceback_signature" in components

    def test_successful_result_omits_signature_components(self, tmp_path: Path) -> None:
        """Successful results must not include signature_components."""
        result = PerformanceResult(
            model_name="org/good",
            generation=MockGeneration(),
            success=True,
        )
        out = tmp_path / "results.jsonl"
        check_models.save_jsonl_report([result], out, prompt="test", system_info={})
        _, rows = _read_jsonl(out)
        assert "signature_components" not in rows[0]

    def test_signature_components_normalized_message_is_stable(self) -> None:
        """Normalized message strips variable numbers for stable clustering."""
        record1 = check_models._build_jsonl_result_record_base(
            PerformanceResult(
                model_name="org/a",
                generation=None,
                success=False,
                error_message="RuntimeError: shape [4, 128] mismatch",
                error_code="ERR",
            )
        )
        record2 = check_models._build_jsonl_result_record_base(
            PerformanceResult(
                model_name="org/b",
                generation=None,
                success=False,
                error_message="RuntimeError: shape [8, 256] mismatch",
                error_code="ERR",
            )
        )
        c1 = record1.get("signature_components", {})
        c2 = record2.get("signature_components", {})
        assert c1.get("normalized_message") == c2.get("normalized_message")


class TestSchemaVersioning:
    """Tests for JSONL schema versioning and round-trip integrity."""

    def test_metadata_format_version_is_2_0(self, tmp_path: Path) -> None:
        """Current JSONL output uses format_version 2.0."""
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
        assert row["success"] is True
        assert row["error_signature"] is None
        assert "signature_components" not in row

    def test_round_trip_result_record_failure(self, tmp_path: Path) -> None:
        """Failed result record round-trips with signature_components."""
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
        assert row["success"] is False
        assert row["error_code"] == "DECODE_ERR"
        sc = _require_present(
            row.get("signature_components"),
            field_name="signature_components",
        )
        assert "normalized_message" in sc
        assert "traceback_signature" in sc
        assert sc.get("error_code") == "DECODE_ERR"

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
        )
        data = json.loads(hist.read_text().strip())
        assert data["format_version"] == "1.0"


class TestRerunEvidence:
    """Tests for differential rerun evidence in JSONL output."""

    def test_rerun_summary_emitted_when_evidence_present(self, tmp_path: Path) -> None:
        """JSONL result includes rerun_summary when rerun_evidence is set."""
        evidence = check_models.RerunEvidence(
            rerun_success=True,
            rerun_generated_chars=42,
            rerun_generation_time=1.5,
            rerun_prompt="Describe this image briefly.",
        )
        result = PerformanceResult(
            model_name="org/model",
            generation=None,
            success=False,
            error_message="RuntimeError: something",
            rerun_evidence=evidence,
        )
        out = tmp_path / "results.jsonl"
        check_models.save_jsonl_report([result], out, prompt="t", system_info={})
        _, rows = _read_jsonl(out)
        summary = rows[0].get("rerun_summary")
        assert summary is not None
        assert summary["rerun_success"] is True
        assert summary.get("rerun_generated_chars") == 42
        assert summary.get("rerun_prompt") == "Describe this image briefly."

    def test_no_rerun_summary_when_no_evidence(self, tmp_path: Path) -> None:
        """JSONL result omits rerun_summary when no rerun was performed."""
        result = PerformanceResult(
            model_name="org/model",
            generation=None,
            success=False,
            error_message="RuntimeError: something",
        )
        out = tmp_path / "results.jsonl"
        check_models.save_jsonl_report([result], out, prompt="t", system_info={})
        _, rows = _read_jsonl(out)
        assert "rerun_summary" not in rows[0]

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
