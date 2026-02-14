"""Tests for quality analysis utilities."""

from __future__ import annotations

import check_models


def test_analyze_generation_text_clean_output() -> None:
    """Test that clean text has no issues detected."""
    text = "This is a normal, varied caption describing an image."
    analysis = check_models.analyze_generation_text(text, 12)

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
    analysis = check_models.analyze_generation_text(text, 50)

    assert analysis.is_repetitive
    assert analysis.repeated_token == "word"


def test_analyze_generation_text_hallucination_table() -> None:
    """Test detection of hallucinated table structures."""
    text = "Caption: Nice photo\n\n| Grade | Count |\n|-------|-------|\n| A     | 42    |"
    analysis = check_models.analyze_generation_text(text, 20)

    assert len(analysis.hallucination_issues) > 0
    assert any("table" in issue.lower() for issue in analysis.hallucination_issues)


def test_analyze_generation_text_hallucination_multiple_choice() -> None:
    """Test detection of hallucinated multiple choice patterns."""
    text = "A) The cat\nB) The dog\nC) The bird\nD) None of the above"
    analysis = check_models.analyze_generation_text(text, 15)

    assert len(analysis.hallucination_issues) > 0
    assert any("choice" in issue.lower() for issue in analysis.hallucination_issues)


def test_analyze_generation_text_excessive_verbosity() -> None:
    """Test detection of overly verbose output."""
    # Verbosity requires 300+ tokens AND meta-commentary or section headers
    text = "The image shows " * 50 + "\n### Analysis\n" + "content " * 50
    # Must have 300+ generated tokens
    analysis = check_models.analyze_generation_text(text, 350)

    assert analysis.is_verbose


def test_analyze_generation_text_formatting_violations() -> None:
    """Test detection of formatting issues."""
    # HTML tags are a formatting violation
    text = "Here is the answer:\n<div>content</div>\n<span>more</span>"
    analysis = check_models.analyze_generation_text(text, 10)

    assert len(analysis.formatting_issues) > 0


def test_analyze_generation_text_excessive_bullets() -> None:
    """Test detection of too many bullet points."""
    # Excessive bullets = more than 25 (threshold from quality_config.yaml)
    # Set high for cataloging prompts that request keyword lists
    text = "\n".join([f"- Item {i}" for i in range(30)])
    analysis = check_models.analyze_generation_text(text, 40)

    assert analysis.has_excessive_bullets
    assert analysis.bullet_count == 30


def test_analyze_generation_text_multiple_issues() -> None:
    """Test that multiple issues can be detected simultaneously."""
    # Repetitive (80%+ same token) + excessive bullets (>25)
    # Each line has only 2 tokens: "-" and "same", so "same" is 50% repetition (not enough)
    # Need to make "same" appear 80%+ of the time
    text = "\n".join(["- same same same same"] * 30)  # "same" appears 120 times, "-" 30 times
    analysis = check_models.analyze_generation_text(text, 150)

    assert analysis.is_repetitive
    assert analysis.has_excessive_bullets
    assert analysis.bullet_count == 30


def test_analyze_generation_text_returns_correct_type() -> None:
    """Test that function returns GenerationQualityAnalysis dataclass."""
    text = "Normal text"
    result = check_models.analyze_generation_text(text, 5)

    assert isinstance(result, check_models.GenerationQualityAnalysis)
    assert hasattr(result, "is_repetitive")
    assert hasattr(result, "repeated_token")
    assert hasattr(result, "hallucination_issues")
    assert hasattr(result, "is_verbose")
    assert hasattr(result, "formatting_issues")
    assert hasattr(result, "has_excessive_bullets")
    assert hasattr(result, "bullet_count")


def test_analyze_generation_text_empty_input() -> None:
    """Test handling of empty text input."""
    analysis = check_models.analyze_generation_text("", 0)

    # Empty text should be surfaced as a harness/prompt-template issue.
    assert not analysis.is_repetitive
    assert analysis.repeated_token is None
    assert not analysis.hallucination_issues
    assert not analysis.is_verbose
    assert analysis.bullet_count == 0
    assert analysis.has_harness_issue is True
    assert analysis.harness_issue_type == "prompt_template"
    assert "output:zero_tokens" in analysis.harness_issue_details


def test_analyze_generation_text_context_ignorance_custom_marker() -> None:
    """Test detection of context ignorance with a custom marker."""
    # Use prompt with at least 3 key terms to meet minimum threshold
    prompt = "MyMarker: Located at White Rock, Hastings, England.\n\nDescribe the image."
    text = "A nice landscape."

    # Should detect missing context terms if we use the correct marker
    # (White, Rock, Hastings, England = 4 terms, all missing = 100% > 75%)
    analysis = check_models.analyze_generation_text(
        text,
        10,
        prompt=prompt,
        context_marker="MyMarker:",
    )

    assert analysis.is_context_ignored
    assert len(analysis.missing_context_terms) >= 3

    # Should NOT detect if we use the default marker (since "Context:" is not in prompt)
    analysis_default = check_models.analyze_generation_text(
        text,
        10,
        prompt=prompt,
    )
    # If "Context:" is not found, it returns False, []
    assert not analysis_default.is_context_ignored


def test_repetitive_phrase_detection_uses_quality_thresholds() -> None:
    """Test that phrase repetition detection uses QUALITY config thresholds."""
    # Verify the QUALITY constants exist and have expected types
    assert hasattr(check_models.QUALITY, "min_phrase_repetitions")
    assert hasattr(check_models.QUALITY, "max_phrase_repetitions")
    assert hasattr(check_models.QUALITY, "phrase_coverage_threshold")
    assert isinstance(check_models.QUALITY.min_phrase_repetitions, int)
    assert isinstance(check_models.QUALITY.max_phrase_repetitions, int)
    assert isinstance(check_models.QUALITY.phrase_coverage_threshold, float)

    # Verify defaults are sensible
    assert check_models.QUALITY.min_phrase_repetitions >= 2
    assert check_models.QUALITY.max_phrase_repetitions > check_models.QUALITY.min_phrase_repetitions
    assert 0 < check_models.QUALITY.phrase_coverage_threshold < 1


# =============================================================================
# Tests for new diagnostic functions
# =============================================================================


def test_compute_vocabulary_diversity_basic() -> None:
    """Test vocabulary diversity computation."""
    # High diversity - all unique words
    ttr, unique, total = check_models.compute_vocabulary_diversity("The quick brown fox jumps")
    assert total == 5
    assert unique == 5
    assert ttr == 1.0

    # Low diversity - repetitive
    ttr, unique, total = check_models.compute_vocabulary_diversity("yes yes yes yes")
    assert total == 4
    assert unique == 1
    assert ttr == 0.25

    # Empty text
    ttr, unique, total = check_models.compute_vocabulary_diversity("")
    assert ttr == 0.0
    assert unique == 0
    assert total == 0


def test_compute_efficiency_metrics() -> None:
    """Test efficiency metrics computation."""
    metrics = check_models.compute_efficiency_metrics(
        tokens_generated=100,
        generation_time=2.0,
        peak_memory_gb=5.0,
    )
    assert metrics["tokens_per_second"] == 50.0
    assert metrics["tokens_per_gb"] == 20.0
    assert metrics["tokens_per_second_per_gb"] == 10.0

    # Missing data returns None
    metrics = check_models.compute_efficiency_metrics(
        tokens_generated=100,
        generation_time=None,
        peak_memory_gb=None,
    )
    assert metrics["tokens_per_second"] is None
    assert metrics["tokens_per_gb"] is None


def test_detect_response_structure() -> None:
    """Test response structure detection."""
    # Text with clear structure
    text = """Caption: A beautiful sunset

    Keywords: sunset, beach, ocean

    Description: The image shows a sunset over the ocean."""
    structure = check_models.detect_response_structure(text)
    assert structure["has_caption"] is True
    assert structure["has_keywords"] is True
    assert structure["has_description"] is True

    # Plain text without structure
    text_plain = "A beautiful sunset over the ocean with orange and pink colors."
    structure_plain = check_models.detect_response_structure(text_plain)
    assert structure_plain["has_caption"] is False
    assert structure_plain["has_keywords"] is False

    # Empty text
    structure_empty = check_models.detect_response_structure("")
    assert structure_empty["has_caption"] is False


def test_compute_confidence_indicators() -> None:
    """Test confidence indicator computation."""
    # High confidence text
    confident = "The image shows a red car. It is parked in a garage."
    indicators = check_models.compute_confidence_indicators(confident)
    assert indicators["definitive_count"] > 0
    assert indicators["confidence_ratio"] > 0.5

    # Uncertain text
    uncertain = "It appears to be a car. It might be red. Perhaps it's in a garage."
    indicators = check_models.compute_confidence_indicators(uncertain)
    assert indicators["hedge_count"] > 0
    assert indicators["confidence_ratio"] < 0.5

    # Empty text
    empty_indicators = check_models.compute_confidence_indicators("")
    assert empty_indicators["hedge_count"] == 0
    assert empty_indicators["definitive_count"] == 0


# =============================================================================
# Tests for LLM Failure Mode Detection
# =============================================================================


def test_detect_output_degeneration_clean() -> None:
    """Clean text should not trigger degeneration detection."""
    text = "This is a normal sentence that ends properly."
    has_degen, degen_type = check_models._detect_output_degeneration(text)
    assert has_degen is False
    assert degen_type is None


def test_detect_output_degeneration_repeated_punctuation() -> None:
    """Detect repeated punctuation at end (common failure mode)."""
    text = "This is a sentence that goes wrong..........."
    has_degen, degen_type = check_models._detect_output_degeneration(text)
    assert has_degen is True
    assert degen_type is not None
    assert "punctuation" in degen_type


def test_detect_output_degeneration_character_loop() -> None:
    """Detect character-level repetition loops."""
    text = "Normal text then aaaaaaaaaaaaaaaaaaaaaaaaa"
    has_degen, degen_type = check_models._detect_output_degeneration(text)
    assert has_degen is True
    assert degen_type is not None
    assert "loop" in degen_type or "character" in degen_type


def test_detect_output_degeneration_excessive_newlines() -> None:
    """Detect excessive newline degeneration."""
    text = "Normal sentence.\n\n\n\n\n\n\n\n\n\n"
    has_degen, degen_type = check_models._detect_output_degeneration(text)
    assert has_degen is True
    assert degen_type == "excessive_newlines"


def test_detect_output_degeneration_short_text() -> None:
    """Short text should not be checked for degeneration."""
    text = "Short."
    has_degen, _degen_type = check_models._detect_output_degeneration(text)
    assert has_degen is False


def test_detect_fabricated_details_clean() -> None:
    """Clean text should not trigger fabrication detection."""
    text = "A red car is parked in front of a blue house."
    has_fab, issues = check_models._detect_fabricated_details(text)
    assert has_fab is False
    assert issues == []


def test_detect_fabricated_details_suspicious_url() -> None:
    """Detect fabricated/placeholder URLs."""
    text = "Learn more at https://example.com/fake/page"
    has_fab, issues = check_models._detect_fabricated_details(text)
    assert has_fab is True
    assert any("url" in issue.lower() for issue in issues)


def test_detect_fabricated_details_uses_configured_patterns() -> None:
    """Configured fabrication patterns should override built-in defaults."""
    original_patterns = check_models.QUALITY.patterns
    check_models.QUALITY.patterns = {
        "fabrication_url_patterns": [r"https?://[^\s]+"],
        "fabrication_suspicious_url_keywords": ["customtokenxyz"],
        "fabrication_precise_stat_patterns": [r"\bwonky\d+\b"],
    }
    try:
        text = "See https://domain.test/customtokenxyz/path with wonky1 and wonky2."
        has_fab, issues = check_models._detect_fabricated_details(text)
        assert has_fab is True
        assert any("suspicious_url" in issue for issue in issues)
        assert any("suspicious_precision" in issue for issue in issues)
    finally:
        check_models.QUALITY.patterns = original_patterns


def test_detect_fabricated_details_future_date() -> None:
    """Detect references to future years (LLM can't know the future)."""
    text = "According to the 2035 census data, the population increased."
    has_fab, issues = check_models._detect_fabricated_details(text)
    assert has_fab is True
    assert any("future" in issue.lower() for issue in issues)


def test_detect_fabricated_details_fake_citation() -> None:
    """Detect fake academic citations."""
    text = "The findings were significant (Smith et al., 2024)."
    has_fab, issues = check_models._detect_fabricated_details(text)
    assert has_fab is True
    assert any("citation" in issue.lower() for issue in issues)


def test_analyze_generation_text_includes_degeneration() -> None:
    """Verify analyze_generation_text detects degeneration."""
    text = "Normal start then goes bad.........."
    analysis = check_models.analyze_generation_text(text, 10)
    assert analysis.has_degeneration is True
    assert analysis.degeneration_type is not None


def test_analyze_generation_text_includes_fabrication() -> None:
    """Verify analyze_generation_text detects fabrication."""
    text = "Check out https://example.com/placeholder for more info."
    analysis = check_models.analyze_generation_text(text, 15)
    assert analysis.has_fabrication is True
    assert len(analysis.fabrication_issues) > 0


# =============================================================================
# Tests for harness/integration issue detection
# =============================================================================


def test_detect_token_encoding_issues_bpe_space_leak() -> None:
    """Detect BPE space marker (Ġ) leaking into output."""
    # This is the exact pattern seen in Devstral output
    text = 'Factual\u0120Caption:\u0120"Evening\u0120view'
    has_issue, issue_type = check_models._detect_token_encoding_issues(text)
    assert has_issue is True
    assert issue_type is not None
    assert "bpe_space_leak" in issue_type


def test_detect_token_encoding_issues_clean() -> None:
    """Clean text should not trigger encoding detection."""
    text = "A normal caption with proper spacing."
    has_issue, issue_type = check_models._detect_token_encoding_issues(text)
    assert has_issue is False
    assert issue_type is None


def test_detect_special_token_leakage_end_token() -> None:
    """Detect special end tokens leaking into output."""
    text = "Good output ends here.<|end|><|endoftext|>And then training data follows."
    has_leak, leaked = check_models._detect_special_token_leakage(text)
    assert has_leak is True
    assert any("<|end|>" in tok for tok in leaked)


def test_detect_special_token_leakage_instruction_marker() -> None:
    """Detect instruction markers leaking (training data leak)."""
    text = "Normal output.\n# INSTRUCTION\nWrite a story about..."
    has_leak, leaked = check_models._detect_special_token_leakage(text)
    assert has_leak is True
    assert any("INSTRUCTION" in tok for tok in leaked)


def test_detect_special_token_leakage_clean() -> None:
    """Clean output should not trigger token leakage detection."""
    text = "A well-formed response describing an image."
    has_leak, leaked = check_models._detect_special_token_leakage(text)
    assert has_leak is False
    assert leaked == []


def test_detect_minimal_output_zero_tokens() -> None:
    """Zero tokens should always be flagged as harness issue."""
    has_minimal, reason = check_models._detect_minimal_output("", 0)
    assert has_minimal is True
    assert reason == "zero_tokens"


def test_detect_minimal_output_filler_response() -> None:
    """Detect minimal filler responses."""
    text = "The image is a photograph."
    has_minimal, reason = check_models._detect_minimal_output(text, 8)
    assert has_minimal is True
    assert "filler_response" in (reason or "")


def test_detect_minimal_output_normal() -> None:
    """Normal output should not be flagged as minimal."""
    text = "A detailed description of the image showing buildings and people."
    has_minimal, reason = check_models._detect_minimal_output(text, 50)
    assert has_minimal is False
    assert reason is None


def test_analyze_generation_text_detects_long_context_breakdown() -> None:
    """Very long prompt contexts with tiny outputs should be flagged."""
    analysis = check_models.analyze_generation_text(
        text="Be concise.",
        generated_tokens=4,
        prompt_tokens=14000,
    )
    assert analysis.has_harness_issue is True
    assert analysis.harness_issue_type == "long_context"
    assert any("long_context" in item for item in analysis.harness_issue_details)


def test_detect_training_data_leak_instruction() -> None:
    """Detect training data instruction patterns leaking into output."""
    text = (
        "Good response about the image showing buildings.\n"
        "The architecture is typical of the region.\n"
        "# INSTRUCTION\n"
        "Write a short story about a young inventor named Alex."
    )
    has_leak, leak_type = check_models._detect_training_data_leak(text)
    assert has_leak is True
    assert leak_type == "instruction_header"


def test_detect_training_data_leak_clean() -> None:
    """Normal output should not trigger training data leak detection."""
    text = "A well-formed response about an image showing historic buildings in England."
    has_leak, leak_type = check_models._detect_training_data_leak(text)
    assert has_leak is False
    assert leak_type is None


def test_detect_training_data_leak_uses_configured_patterns() -> None:
    """Configured training-leak pattern groups should be used."""
    original_patterns = check_models.QUALITY.patterns
    check_models.QUALITY.patterns = {
        "training_leak_instruction_header_patterns": [r"\n### CUSTOM LEAK ###"],
    }
    try:
        text = (
            "A detailed response about architecture and people in the square. "
            "It includes grounded visual details and contextual cues.\n"
            "### CUSTOM LEAK ###\n"
            "Write a short story about a traveler."
        )
        has_leak, leak_type = check_models._detect_training_data_leak(text)
        assert has_leak is True
        assert leak_type == "instruction_header"
    finally:
        check_models.QUALITY.patterns = original_patterns


def test_analyze_generation_text_includes_harness_issues() -> None:
    """Verify analyze_generation_text detects harness issues."""
    # Text with BPE space leak
    text = "Factual\u0120Caption:\u0120A\u0120nice\u0120image"
    analysis = check_models.analyze_generation_text(text, 10)
    assert analysis.has_harness_issue is True
    assert analysis.harness_issue_type == "encoding"
    assert len(analysis.harness_issue_details) > 0


def test_build_quality_issues_string_includes_harness() -> None:
    """Verify harness issues appear prominently in quality string."""
    # Create analysis with harness issue
    analysis = check_models.GenerationQualityAnalysis(
        is_repetitive=False,
        repeated_token=None,
        hallucination_issues=[],
        is_verbose=False,
        formatting_issues=[],
        has_excessive_bullets=False,
        bullet_count=0,
        is_context_ignored=False,
        missing_context_terms=[],
        is_refusal=False,
        refusal_type=None,
        is_generic=False,
        specificity_score=0.0,
        has_language_mixing=False,
        language_mixing_issues=[],
        has_degeneration=False,
        degeneration_type=None,
        has_fabrication=False,
        fabrication_issues=[],
        has_harness_issue=True,
        harness_issue_type="encoding",
        harness_issue_details=["bpe_space_leak(5)"],
    )
    result = check_models._build_quality_issues_string(analysis)
    assert result is not None
    assert "⚠️harness" in result
    assert result.startswith("⚠️harness")  # Should be first


def test_has_harness_issues_only() -> None:
    """Test method to check if only harness issues present."""
    # Harness issue only
    harness_only = check_models.GenerationQualityAnalysis(
        is_repetitive=False,
        repeated_token=None,
        hallucination_issues=[],
        is_verbose=False,
        formatting_issues=[],
        has_excessive_bullets=False,
        bullet_count=0,
        is_context_ignored=False,
        missing_context_terms=[],
        is_refusal=False,
        refusal_type=None,
        is_generic=False,
        specificity_score=0.0,
        has_language_mixing=False,
        language_mixing_issues=[],
        has_degeneration=False,
        degeneration_type=None,
        has_fabrication=False,
        fabrication_issues=[],
        has_harness_issue=True,
        harness_issue_type="encoding",
        harness_issue_details=["bpe_space_leak"],
    )
    assert harness_only.has_harness_issues_only() is True

    # Harness issue + model quality issue
    mixed_issues = check_models.GenerationQualityAnalysis(
        is_repetitive=True,  # Model quality issue
        repeated_token="same",  # noqa: S106 - test value, not a password
        hallucination_issues=[],
        is_verbose=False,
        formatting_issues=[],
        has_excessive_bullets=False,
        bullet_count=0,
        is_context_ignored=False,
        missing_context_terms=[],
        is_refusal=False,
        refusal_type=None,
        is_generic=False,
        specificity_score=0.0,
        has_language_mixing=False,
        language_mixing_issues=[],
        has_degeneration=False,
        degeneration_type=None,
        has_fabrication=False,
        fabrication_issues=[],
        has_harness_issue=True,  # Also harness issue
        harness_issue_type="encoding",
        harness_issue_details=["bpe_space_leak"],
    )
    assert mixed_issues.has_harness_issues_only() is False
