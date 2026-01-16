"""Tests for Markdown formatting utilities."""

import check_models


def test_escape_markdown_in_text_pipes() -> None:
    """Should escape pipe characters."""
    result = check_models._escape_markdown_in_text("a|b|c")
    assert result == "a\\|b\\|c"


def test_escape_markdown_in_text_single_pipe() -> None:
    """Should escape single pipe."""
    result = check_models._escape_markdown_in_text("before|after")
    assert result == "before\\|after"


def test_escape_markdown_in_text_no_pipes() -> None:
    """Should leave text without pipes unchanged."""
    text = "normal text without pipes"
    result = check_models._escape_markdown_in_text(text)
    assert result == text


def test_escape_markdown_in_text_empty() -> None:
    """Should handle empty string."""
    result = check_models._escape_markdown_in_text("")
    assert result == ""


def test_escape_markdown_in_text_only_pipes() -> None:
    """Should escape text with only pipes."""
    result = check_models._escape_markdown_in_text("|||")
    assert result == "\\|\\|\\|"


def test_escape_markdown_in_text_mixed_content() -> None:
    """Should escape pipes in mixed content."""
    result = check_models._escape_markdown_in_text("model|size|speed")
    assert result == "model\\|size\\|speed"


def test_escape_markdown_in_text_whitespace() -> None:
    """Should preserve whitespace around escaped pipes."""
    result = check_models._escape_markdown_in_text("a | b | c")
    assert result == "a \\| b \\| c"


def test_format_failures_by_package_empty() -> None:
    """Should return empty list when no failures."""
    results = [
        check_models.PerformanceResult(
            model_name="success-model",
            generation=None,
            success=True,
        ),
    ]
    output = check_models._format_failures_by_package_text(results)
    assert output == []


def test_format_failures_by_package_groups_by_package() -> None:
    """Should group failures by error_package."""
    results = [
        check_models.PerformanceResult(
            model_name="model-a",
            generation=None,
            success=False,
            error_package="mlx",
            error_stage="OOM",
            error_message="Out of memory",
        ),
        check_models.PerformanceResult(
            model_name="model-b",
            generation=None,
            success=False,
            error_package="mlx-vlm",
            error_stage="Processor Error",
            error_message="Processor failed",
        ),
        check_models.PerformanceResult(
            model_name="model-c",
            generation=None,
            success=False,
            error_package="mlx",
            error_stage="Type Cast Error",
            error_message="std::bad_cast",
        ),
    ]
    output = check_models._format_failures_by_package_text(results)

    # Should contain the section header
    assert "## 🚨 Failures by Package (Actionable)" in output

    # Should have a table with packages
    output_text = "\n".join(output)
    assert "`mlx`" in output_text
    assert "`mlx-vlm`" in output_text

    # Should show 2 failures for mlx (OOM, Type Cast Error)
    assert "2" in output_text  # mlx has 2 failures
    assert "1" in output_text  # mlx-vlm has 1 failure


def test_format_failures_by_package_includes_actionable_items() -> None:
    """Should include actionable items section with model details."""
    results = [
        check_models.PerformanceResult(
            model_name="test-model",
            generation=None,
            success=False,
            error_package="transformers",
            error_stage="Lib Version",
            error_message="cannot import name 'some_function'",
            error_type="ImportError",
        ),
    ]
    output = check_models._format_failures_by_package_text(results)
    output_text = "\n".join(output)

    # Should have actionable items section
    assert "### Actionable Items by Package" in output_text
    assert "#### transformers" in output_text
    assert "test-model" in output_text
    assert "ImportError" in output_text
