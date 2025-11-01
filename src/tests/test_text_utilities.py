"""Tests for text padding utilities."""

# ruff: noqa: SLF001
import check_models


def test_pad_text_left_align() -> None:
    """Should pad text on the right (left align)."""
    result = check_models._pad_text("hi", 5, right_align=False)
    assert result == "hi   "
    assert len(result) == 5


def test_pad_text_right_align() -> None:
    """Should pad text on the left (right align)."""
    result = check_models._pad_text("hi", 5, right_align=True)
    assert result == "   hi"
    assert len(result) == 5


def test_pad_text_exact_width() -> None:
    """Should not pad if text is exact width."""
    result = check_models._pad_text("hello", 5)
    assert result == "hello"
    assert len(result) == 5


def test_pad_text_overflow() -> None:
    """Should not truncate if text exceeds width."""
    result = check_models._pad_text("toolong", 3)
    assert result == "toolong"
    assert len(result) == 7


def test_pad_text_zero_width() -> None:
    """Should handle zero width."""
    result = check_models._pad_text("text", 0)
    assert result == "text"


def test_pad_text_empty_string() -> None:
    """Should pad empty string."""
    result = check_models._pad_text("", 5)
    assert result == "     "
    assert len(result) == 5


def test_pad_text_single_char() -> None:
    """Should pad single character."""
    result = check_models._pad_text("x", 3, right_align=True)
    assert result == "  x"


def test_pad_text_unicode() -> None:
    """Should handle unicode characters."""
    result = check_models._pad_text("🎉", 3)
    # Note: Display width may differ from string length for unicode
    assert "🎉" in result
