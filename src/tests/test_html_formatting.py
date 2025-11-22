"""Tests for HTML formatting utilities."""

import check_models


def test_escape_html_tags_selective_basic() -> None:
    """Should escape < and > in text."""
    result = check_models.HTML_ESCAPER.escape("Use <model> here")
    assert result == "Use &lt;model&gt; here"


def test_escape_html_tags_selective_multiple() -> None:
    """Should escape non-allowed tags but keep allowed tags like <b>."""
    text = "<a> and <b> tags"
    result = check_models.HTML_ESCAPER.escape(text)
    # <a> is not allowed, so it's escaped; <b> is allowed, so it's kept
    assert result == "&lt;a&gt; and <b> tags"


def test_escape_html_tags_selective_preserves_entities() -> None:
    """Should preserve existing HTML entities."""
    result = check_models.HTML_ESCAPER.escape("5 &lt; 10")
    assert result == "5 &lt; 10"


def test_escape_html_tags_selective_mixed() -> None:
    """Should handle mixed content with entities."""
    # Function only escapes tag-like patterns, not bare angle brackets
    result = check_models.HTML_ESCAPER.escape("5 < 10 &amp; 10 > 5")
    # Bare angle brackets and entities are preserved as-is
    assert result == "5 < 10 &amp; 10 > 5"


def test_escape_html_tags_selective_ampersands() -> None:
    """Should handle standalone ampersands."""
    # Function only escapes tags, not ampersands
    result = check_models.HTML_ESCAPER.escape("Tom & Jerry")
    # Ampersand is preserved as-is (not escaped by this function)
    assert result == "Tom & Jerry"


def test_escape_html_tags_selective_no_tags() -> None:
    """Should leave text without tags unchanged."""
    text = "Plain text without any special characters"
    result = check_models.HTML_ESCAPER.escape(text)
    assert result == text


def test_escape_html_tags_selective_empty() -> None:
    """Should handle empty string."""
    assert check_models.HTML_ESCAPER.escape("") == ""


def test_escape_html_tags_selective_entity_chain() -> None:
    """Should preserve complex entity chains."""
    result = check_models.HTML_ESCAPER.escape("&lt;&gt; &amp; &quot;")
    assert "&lt;" in result
    assert "&gt;" in result
    assert "&amp;" in result
    assert "&quot;" in result
