"""Tests for JSONL output generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from check_models import (
    PerformanceResult,
    _history_path_for_jsonl,
    _load_latest_history_record,
    append_history_record,
    compare_history_records,
    save_jsonl_report,
)

if TYPE_CHECKING:
    from pathlib import Path


def _read_jsonl(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read JSONL file returning (metadata_header, result_rows)."""
    lines = path.read_text().strip().split("\n")
    header: dict[str, Any] = json.loads(lines[0])
    results: list[dict[str, Any]] = [json.loads(line) for line in lines[1:]]
    return header, results


@dataclass
class MockGeneration:
    """Mock generation result for testing."""

    text: str | None = "generated text"
    prompt_tokens: int | None = 10
    generation_tokens: int | None = 20
    generation_tps: float | None = 5.0
    peak_memory: float | None = 1.5
    time: float | None = None
    active_memory: float | None = None
    cache_memory: float | None = None


def test_save_jsonl_report_creates_file(tmp_path: Path) -> None:
    """Test that save_jsonl_report creates a file with metadata header."""
    output_file = tmp_path / "results.jsonl"
    results: list[PerformanceResult] = []
    save_jsonl_report(results, output_file, prompt="test", system_info={})

    assert output_file.exists()
    header, rows = _read_jsonl(output_file)
    assert header["_type"] == "metadata"
    assert header["prompt"] == "test"
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
    assert data["metrics"]["generation_tps"] == 5.0
    assert data["metrics"]["prompt_tokens"] == 10


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
    assert data["error_message"] == "Something went wrong"
    assert data["error_stage"] == "Model Load"


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


def test_compare_history_records_detects_regressions_and_recoveries() -> None:
    """Test regression/recovery detection between history records."""
    previous = {
        "model_results": {
            "model-a": {"success": True},
            "model-b": {"success": False},
            "model-c": {"success": True},
        },
    }
    current = {
        "model_results": {
            "model-a": {"success": False},
            "model-b": {"success": True},
            "model-d": {"success": True},
        },
    }

    summary = compare_history_records(previous, current)
    assert summary["regressions"] == ["model-a"]
    assert summary["recoveries"] == ["model-b"]
    assert summary["new_models"] == ["model-d"]
    assert summary["missing_models"] == ["model-c"]


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
    valid = json.dumps({"_type": "run", "model_results": {"m": {"success": True}}})
    history.write_text(f'{valid}\nNOT-JSON\n{{"bad": true}}\n')

    record = _load_latest_history_record(history)
    assert record is not None
    assert record["_type"] == "run"
    assert record["model_results"]["m"]["success"] is True


def test_load_latest_history_record_only_corrupted(tmp_path: Path) -> None:
    """Return None when every line is invalid JSON."""
    history = tmp_path / "corrupt.jsonl"
    history.write_text("NOT-JSON-1\nNOT-JSON-2\n")
    assert _load_latest_history_record(history) is None


def test_load_latest_history_record_returns_last_run(tmp_path: Path) -> None:
    """When multiple run records exist, return the last one."""
    history = tmp_path / "multi.jsonl"
    first = json.dumps({"_type": "run", "seq": 1})
    second = json.dumps({"_type": "run", "seq": 2})
    history.write_text(f"{first}\n{second}\n")

    record = _load_latest_history_record(history)
    assert record is not None
    assert record["seq"] == 2


def test_load_latest_history_record_skips_non_run_types(tmp_path: Path) -> None:
    """Skip records whose _type is not 'run'."""
    history = tmp_path / "types.jsonl"
    run_record = json.dumps({"_type": "run", "ok": True})
    metadata = json.dumps({"_type": "metadata", "info": "x"})
    history.write_text(f"{run_record}\n{metadata}\n")

    record = _load_latest_history_record(history)
    assert record is not None
    assert record["_type"] == "run"
    assert record["ok"] is True


# ---------------------------------------------------------------------------
# compare_history_records — baseline (no previous)
# ---------------------------------------------------------------------------


def test_compare_history_records_no_previous() -> None:
    """With no previous record, all current models are 'new'."""
    current = {
        "model_results": {
            "model-x": {"success": True},
            "model-y": {"success": False},
        },
    }

    summary = compare_history_records(None, current)
    assert summary["regressions"] == []
    assert summary["recoveries"] == []
    assert summary["new_models"] == ["model-x", "model-y"]
    assert summary["missing_models"] == []
