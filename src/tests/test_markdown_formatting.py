"""Tests for Markdown formatting utilities."""

import re

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


# ── Bare URL wrapping (MD034) tests ────────────────────────────────────────

_BARE_URL_RE = re.compile(r"(?<![<(])https?://")
"""Matches a bare URL not already wrapped in < > or ( )."""


def _gallery_lines_for(result: check_models.PerformanceResult) -> str:
    """Return joined gallery markdown for a single result."""
    return "\n".join(check_models._generate_model_gallery_section([result]))


def test_bare_url_in_long_error_is_wrapped() -> None:
    """Error messages with bare URLs should get <angle brackets> in markdown."""
    result = check_models.PerformanceResult(
        model_name="test/model",
        generation=None,
        success=False,
        error_stage="No Chat Template",
        error_message=(
            "Cannot use chat template because tokenizer.chat_template is not set. "
            "See https://huggingface.co/docs/transformers/main/en/chat_templating"
        ),
    )
    md = _gallery_lines_for(result)
    # The URL must be wrapped in angle brackets
    assert "<https://huggingface.co/docs/transformers/main/en/chat_templating>" in md
    # No bare URL should remain
    assert not _BARE_URL_RE.search(md), f"Bare URL found in:\n{md}"


def test_bare_url_in_short_error_is_wrapped() -> None:
    """Even short inline errors with URLs should be wrapped."""
    result = check_models.PerformanceResult(
        model_name="test/model",
        generation=None,
        success=False,
        error_stage="Error",
        error_message="See https://example.com/help",
    )
    md = _gallery_lines_for(result)
    assert "<https://example.com/help>" in md
    assert not _BARE_URL_RE.search(md)


def test_already_wrapped_url_not_double_wrapped() -> None:
    """URLs already in angle brackets should not be double-wrapped."""
    result = check_models.PerformanceResult(
        model_name="test/model",
        generation=None,
        success=False,
        error_stage="Error",
        error_message="See <https://example.com/help> for details",
    )
    md = _gallery_lines_for(result)
    assert "<https://example.com/help>" in md
    assert "<<https://" not in md


def test_error_without_url_unchanged() -> None:
    """Error messages without URLs should be unaffected."""
    result = check_models.PerformanceResult(
        model_name="test/model",
        generation=None,
        success=False,
        error_stage="OOM",
        error_message="Out of memory during generation",
    )
    md = _gallery_lines_for(result)
    assert "Out of memory during generation" in md
