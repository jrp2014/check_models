"""Tests for TSV output generation."""

import tempfile
from dataclasses import dataclass
from pathlib import Path

import check_models


@dataclass
class MockGenerationResult:
    """Mock GenerationResult for testing."""

    text: str | None = "Generated text"
    prompt_tokens: int | None = 100
    generation_tokens: int | None = 50
    time: float | None = None
    active_memory: float | None = None
    cache_memory: float | None = None


def test_generate_tsv_report_basic(tmp_path: Path) -> None:
    """Should generate basic TSV report with headers and data."""
    # Create a simple test result
    results = [
        check_models.PerformanceResult(
            model_name="test/model-1",
            success=True,
            generation=MockGenerationResult(text="Test output"),
            total_time=1.5,
            generation_time=1.0,
            model_load_time=0.5,
        ),
    ]

    output_file = tmp_path / "test_output.tsv"
    check_models.generate_tsv_report(results, output_file)

    # Verify file was created
    assert output_file.exists()

    # Read and verify content
    content = output_file.read_text(encoding="utf-8")
    lines = content.strip().split("\n")

    # Should have at least header + 1 data row
    assert len(lines) >= 2

    # Verify it's tab-separated
    assert "\t" in lines[0]  # Header line
    assert "\t" in lines[1]  # Data line


def test_tsv_escapes_tabs_in_values() -> None:
    """Should replace tabs with spaces in field values."""
    results = [
        check_models.PerformanceResult(
            model_name="test/model",
            success=True,
            generation=MockGenerationResult(text="Line with\ttab character"),
            total_time=1.0,
            generation_time=0.5,
            model_load_time=0.5,
        ),
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
        output_file = Path(f.name)

    try:
        check_models.generate_tsv_report(results, output_file)
        content = output_file.read_text(encoding="utf-8")

        # The output text should not contain actual tabs within the field value
        # (but will have tabs as column separators)
        lines = content.strip().split("\n")
        # Get the last column which contains the output
        last_column = lines[1].split("\t")[-1]
        # Tabs should be replaced with 4 spaces
        assert "    " in last_column
    finally:
        output_file.unlink()


def test_tsv_escapes_newlines_in_values() -> None:
    r"""Should replace newlines with escaped \n sequence."""
    results = [
        check_models.PerformanceResult(
            model_name="test/model",
            success=True,
            generation=MockGenerationResult(text="Line 1\nLine 2\nLine 3"),
            total_time=1.0,
            generation_time=0.5,
            model_load_time=0.5,
        ),
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
        output_file = Path(f.name)

    try:
        check_models.generate_tsv_report(results, output_file)
        content = output_file.read_text(encoding="utf-8")

        # Content should be 2 lines (header + 1 data row)
        lines = content.strip().split("\n")
        assert len(lines) == 2

        # The newlines should be escaped as literal \n
        last_column = lines[1].split("\t")[-1]
        assert "\\n" in last_column
        # Should not have actual newlines in the data
        assert "\n" not in last_column or last_column.count("\n") == 0
    finally:
        output_file.unlink()


def test_tsv_removes_html_tags_from_headers() -> None:
    """Should remove HTML tags like <br> from headers."""
    results = [
        check_models.PerformanceResult(
            model_name="test/model",
            success=True,
            generation=MockGenerationResult(text="output"),
            total_time=1.0,
            generation_time=0.5,
            model_load_time=0.5,
        ),
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
        output_file = Path(f.name)

    try:
        check_models.generate_tsv_report(results, output_file)
        content = output_file.read_text(encoding="utf-8")

        # Header should not contain HTML tags
        header_line = content.split("\n")[0]
        assert "<br>" not in header_line
        assert "<" not in header_line
        assert ">" not in header_line
    finally:
        output_file.unlink()


def test_tsv_handles_failed_results() -> None:
    """Should handle failed results with error messages."""
    results = [
        check_models.PerformanceResult(
            model_name="test/failed-model",
            success=False,
            error_stage="load",
            error_message="Failed to load model",
            generation=None,
        ),
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
        output_file = Path(f.name)

    try:
        check_models.generate_tsv_report(results, output_file)
        content = output_file.read_text(encoding="utf-8")

        # Should have header + data row
        lines = content.strip().split("\n")
        assert len(lines) == 2

        # Error message should be in the output
        assert "Error" in content or "Failed to load model" in content
    finally:
        output_file.unlink()


def test_tsv_empty_results() -> None:
    """Should handle empty results list gracefully."""
    results: list[check_models.PerformanceResult] = []

    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
        output_file = Path(f.name)

    try:
        check_models.generate_tsv_report(results, output_file)
        # Should not create file or create empty file for empty results
        # Based on the implementation, it returns early if no results
        assert not output_file.exists() or output_file.stat().st_size == 0
    finally:
        if output_file.exists():
            output_file.unlink()


def test_tsv_full_model_name(tmp_path: Path) -> None:
    """Should preserve the full model name (including organization) in TSV output."""
    full_model_name = "organization/specific-model-v1"
    results = [
        check_models.PerformanceResult(
            model_name=full_model_name,
            success=True,
            generation=MockGenerationResult(text="Output"),
            total_time=1.0,
            generation_time=0.5,
            model_load_time=0.5,
        ),
    ]

    output_file = tmp_path / "test_full_name.tsv"
    check_models.generate_tsv_report(results, output_file)

    content = output_file.read_text(encoding="utf-8")
    lines = content.strip().split("\n")

    # Check the data row (index 1)
    data_row = lines[1]
    # The model name is typically the first column
    assert full_model_name in data_row
    # Ensure it wasn't truncated to just "specific-model-v1"
    assert f"\t{full_model_name}\t" in f"\t{data_row}\t" or data_row.startswith(
        f"{full_model_name}\t",
    )
