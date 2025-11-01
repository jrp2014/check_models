"""Tests for report generation."""

# ruff: noqa: ANN201
from pathlib import Path

import check_models


def test_generate_html_report_creates_file(tmp_path: Path):
    """Should create HTML file with valid structure."""
    output_file = tmp_path / "report.html"

    # Create mock results
    mock_results = [
        check_models.PerformanceResult(
            model_identifier="test/model",
            image_path=Path("test.jpg"),
            prompt="Test prompt",
            response="Test response from model",
            prompt_tps=10.5,
            generation_tps=25.3,
            total_time=2.5,
            peak_memory_gb=4.2,
            success=True,
            error_message=None,
        ),
    ]

    check_models.generate_html_report(
        results=mock_results,
        output_path=output_file,
        library_versions={"mlx": "0.1.0", "mlx-vlm": "0.2.0"},
        prompt="Test prompt",
    )

    assert output_file.exists()
    content = output_file.read_text()

    # Check for HTML structure
    assert "<!DOCTYPE html>" in content or "<html" in content.lower()
    assert "</html>" in content.lower()

    # Check for data presence
    assert "test/model" in content
    assert "Test response from model" in content
    assert "10.5" in content or "10." in content  # TPS value


def test_generate_markdown_report_creates_file(tmp_path: Path):
    """Should create Markdown file with tables."""
    output_file = tmp_path / "report.md"

    mock_results = [
        check_models.PerformanceResult(
            model_identifier="test/model-name",
            image_path=Path("test.jpg"),
            prompt="Describe this image",
            response="A test response",
            prompt_tps=15.2,
            generation_tps=30.1,
            total_time=1.8,
            peak_memory_gb=3.5,
            success=True,
            error_message=None,
        ),
    ]

    check_models.generate_markdown_report(
        results=mock_results,
        output_path=output_file,
        library_versions={"mlx": "0.1.0"},
        prompt="Describe this image",
    )

    assert output_file.exists()
    content = output_file.read_text()

    # Check for Markdown structure
    assert "#" in content  # Headers
    assert "|" in content  # Tables

    # Check for data
    assert "test/model-name" in content
    assert "A test response" in content


def test_generate_html_report_handles_failed_result(tmp_path: Path):
    """Should include error information for failed results."""
    output_file = tmp_path / "report.html"

    mock_results = [
        check_models.PerformanceResult(
            model_identifier="failed/model",
            image_path=Path("test.jpg"),
            prompt="Test",
            response="",
            prompt_tps=0.0,
            generation_tps=0.0,
            total_time=0.0,
            peak_memory_gb=0.0,
            success=False,
            error_message="Model loading failed",
        ),
    ]

    check_models.generate_html_report(
        results=mock_results,
        output_path=output_file,
        library_versions={"mlx": "0.1.0"},
        prompt="Test",
    )

    assert output_file.exists()
    content = output_file.read_text()
    assert "failed/model" in content
    assert "Model loading failed" in content or "error" in content.lower()


def test_generate_markdown_report_handles_multiple_results(tmp_path: Path):
    """Should include all results in the report."""
    output_file = tmp_path / "report.md"

    mock_results = [
        check_models.PerformanceResult(
            model_identifier="model/one",
            image_path=Path("test.jpg"),
            prompt="Test",
            response="Response 1",
            prompt_tps=10.0,
            generation_tps=20.0,
            total_time=1.0,
            peak_memory_gb=2.0,
            success=True,
            error_message=None,
        ),
        check_models.PerformanceResult(
            model_identifier="model/two",
            image_path=Path("test.jpg"),
            prompt="Test",
            response="Response 2",
            prompt_tps=15.0,
            generation_tps=25.0,
            total_time=1.5,
            peak_memory_gb=3.0,
            success=True,
            error_message=None,
        ),
    ]

    check_models.generate_markdown_report(
        results=mock_results,
        output_path=output_file,
        library_versions={"mlx": "0.1.0"},
        prompt="Test",
    )

    content = output_file.read_text()
    assert "model/one" in content
    assert "model/two" in content
    assert "Response 1" in content
    assert "Response 2" in content


def test_format_field_label_standardizes_labels():
    """Should format field labels consistently."""
    assert check_models.format_field_label("model_identifier") == "Model"
    assert check_models.format_field_label("prompt_tps") == "Prompt TPS"
    assert check_models.format_field_label("total_time") == "Total Time"


def test_format_field_value_formats_numbers():
    """Should format numeric values appropriately."""
    # TPS values
    assert "10.5" in check_models.format_field_value("prompt_tps", 10.5)
    
    # Time values
    assert "s" in check_models.format_field_value("total_time", 2.5)
    
    # Memory values
    assert "GB" in check_models.format_field_value("peak_memory_gb", 4.2)


def test_format_field_value_preserves_strings():
    """Should preserve string values without modification."""
    result = check_models.format_field_value("model_identifier", "test/model")
    assert result == "test/model"

    result = check_models.format_field_value("response", "Test response")
    assert result == "Test response"
