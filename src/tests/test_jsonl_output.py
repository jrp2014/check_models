"""Tests for JSONL output generation."""

import json
from dataclasses import dataclass
from pathlib import Path

from check_models import PerformanceResult, save_jsonl_report


@dataclass
class MockGeneration:
    """Mock generation result for testing."""

    text: str = "generated text"
    prompt_tokens: int = 10
    generation_tokens: int = 20
    generation_tps: float = 5.0
    peak_memory: float = 1.5


def test_save_jsonl_report_creates_file(tmp_path: Path) -> None:
    """Test that save_jsonl_report creates a file."""
    output_file = tmp_path / "results.jsonl"
    results: list[PerformanceResult] = []
    save_jsonl_report(results, output_file)

    assert output_file.exists()
    assert output_file.read_text() == ""


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
    save_jsonl_report(results, output_file)

    assert output_file.exists()
    lines = output_file.read_text().strip().split("\n")
    assert len(lines) == 1

    data = json.loads(lines[0])
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
    save_jsonl_report(results, output_file)

    lines = output_file.read_text().strip().split("\n")
    data = json.loads(lines[0])

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
    save_jsonl_report(results, output_file)

    lines = output_file.read_text().strip().split("\n")
    data = json.loads(lines[0])

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
    save_jsonl_report(results, output_file)

    lines = output_file.read_text().strip().split("\n")
    data = json.loads(lines[0])

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
    save_jsonl_report(results, output_file)

    lines = output_file.read_text().strip().split("\n")
    data = json.loads(lines[0])

    assert data["model"] == "test-model"
    assert data["quality_issues"] == []
