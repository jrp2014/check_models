"""Tests for Markdown formatting utilities."""

# ruff: noqa: SLF001
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
