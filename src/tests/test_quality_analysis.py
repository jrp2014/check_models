"""Tests for quality analysis utilities."""

from __future__ import annotations

from check_models import GenerationQualityAnalysis, analyze_generation_text


def test_analyze_generation_text_clean_output() -> None:
    """Test that clean text has no issues detected."""
    text = "This is a normal, varied caption describing an image."
    analysis = analyze_generation_text(text, 12)

    assert not analysis.is_repetitive
    assert analysis.repeated_token is None
    assert not analysis.hallucination_issues
    assert not analysis.is_verbose
    assert not analysis.formatting_issues
    assert not analysis.has_excessive_bullets
    assert analysis.bullet_count == 0


def test_analyze_generation_text_repetitive() -> None:
    """Test detection of repetitive output."""
    # Repetitive detection requires 80% of TOKENS to be the same
    # "word" repeated 100 times = 100 tokens, all same = 100% repetitive
    text = "word " * 100
    analysis = analyze_generation_text(text, 50)

    assert analysis.is_repetitive
    assert analysis.repeated_token == "word"  # noqa: S105 - not a password, it's a token


def test_analyze_generation_text_hallucination_table() -> None:
    """Test detection of hallucinated table structures."""
    text = "Caption: Nice photo\n\n| Grade | Count |\n|-------|-------|\n| A     | 42    |"
    analysis = analyze_generation_text(text, 20)

    assert len(analysis.hallucination_issues) > 0
    assert any("table" in issue.lower() for issue in analysis.hallucination_issues)


def test_analyze_generation_text_hallucination_multiple_choice() -> None:
    """Test detection of hallucinated multiple choice patterns."""
    text = "A) The cat\nB) The dog\nC) The bird\nD) None of the above"
    analysis = analyze_generation_text(text, 15)

    assert len(analysis.hallucination_issues) > 0
    assert any("choice" in issue.lower() for issue in analysis.hallucination_issues)


def test_analyze_generation_text_excessive_verbosity() -> None:
    """Test detection of overly verbose output."""
    # Verbosity requires 300+ tokens AND meta-commentary or section headers
    text = "The image shows " * 50 + "\n### Analysis\n" + "content " * 50
    # Must have 300+ generated tokens
    analysis = analyze_generation_text(text, 350)

    assert analysis.is_verbose


def test_analyze_generation_text_formatting_violations() -> None:
    """Test detection of formatting issues."""
    # HTML tags are a formatting violation
    text = "Here is the answer:\n<div>content</div>\n<span>more</span>"
    analysis = analyze_generation_text(text, 10)

    assert len(analysis.formatting_issues) > 0


def test_analyze_generation_text_excessive_bullets() -> None:
    """Test detection of too many bullet points."""
    # Excessive bullets = more than 15
    text = "\n".join([f"- Item {i}" for i in range(20)])
    analysis = analyze_generation_text(text, 30)

    assert analysis.has_excessive_bullets
    assert analysis.bullet_count == 20


def test_analyze_generation_text_multiple_issues() -> None:
    """Test that multiple issues can be detected simultaneously."""
    # Repetitive (80%+ same token) + excessive bullets (>15)
    # Each line has only 2 tokens: "-" and "same", so "same" is 50% repetition (not enough)
    # Need to make "same" appear 80%+ of the time
    text = "\n".join(["- same same same same"] * 20)  # "same" appears 80 times, "-" 20 times
    analysis = analyze_generation_text(text, 100)

    assert analysis.is_repetitive
    assert analysis.has_excessive_bullets
    assert analysis.bullet_count == 20


def test_analyze_generation_text_returns_correct_type() -> None:
    """Test that function returns GenerationQualityAnalysis dataclass."""
    text = "Normal text"
    result = analyze_generation_text(text, 5)

    assert isinstance(result, GenerationQualityAnalysis)
    assert hasattr(result, "is_repetitive")
    assert hasattr(result, "repeated_token")
    assert hasattr(result, "hallucination_issues")
    assert hasattr(result, "is_verbose")
    assert hasattr(result, "formatting_issues")
    assert hasattr(result, "has_excessive_bullets")
    assert hasattr(result, "bullet_count")


def test_analyze_generation_text_empty_input() -> None:
    """Test handling of empty text input."""
    analysis = analyze_generation_text("", 0)

    # Empty text should not trigger most issues
    assert not analysis.is_repetitive
    assert analysis.repeated_token is None
    assert not analysis.hallucination_issues
    assert not analysis.is_verbose
    assert analysis.bullet_count == 0
