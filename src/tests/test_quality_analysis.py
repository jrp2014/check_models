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


def test_detect_hallucination_patterns_uses_configured_keyword_lists() -> None:
    """Configured substring lists should override built-in hallucination defaults."""
    original_patterns = check_models.QUALITY.patterns
    check_models.QUALITY.patterns = {
        "hallucination_question_indicators": ["bespoke hallucination prompt"],
        "hallucination_edu_keywords": ["custom classroom leak"],
    }
    try:
        issues = check_models._detect_hallucination_patterns(
            "This long output includes a bespoke hallucination prompt and a custom classroom leak. "
            * 4,
        )
        assert any("question" in issue.lower() for issue in issues)
        assert any("educational" in issue.lower() for issue in issues)
    finally:
        check_models.QUALITY.patterns = original_patterns


def test_analyze_generation_text_excessive_verbosity() -> None:
    """Test detection of overly verbose output."""
    # Verbosity requires 400+ tokens AND meta-commentary or section headers
    text = "The image shows " * 50 + "\n### Analysis\n" + "content " * 50
    # Must have 400+ generated tokens
    analysis = check_models.analyze_generation_text(text, 450)

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
    # Repetitive (90%+ same token) + excessive bullets (>25)
    text = "\n".join(["- " + " ".join(["same"] * 9)] * 30)
    analysis = check_models.analyze_generation_text(text, 300)

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
    # Use trusted hints with at least 5 key terms to meet minimum threshold
    prompt = (
        "MyMarker: Title hint: White Rock shoreline. "
        "Description hint: White Rock shoreline with wooden posts.\n\nDescribe the image."
    )
    text = "A nice landscape."

    # Should detect missing context terms if we use the correct marker
    # (White, Rock, shoreline, wooden, posts = trusted terms, most missing)
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


def test_analyze_generation_text_catalog_contract_missing_sections() -> None:
    """Prompt-level catalog contract should flag missing required sections."""
    prompt = (
        "Analyze this image for cataloguing metadata.\n"
        "Return exactly these three sections, and nothing else:\n"
        "Title: 5-10 words.\nDescription: 1-2 factual sentences.\nKeywords: 10-18 terms."
    )
    text = "A misty lakeshore with trees and power lines."

    analysis = check_models.analyze_generation_text(
        text,
        generated_tokens=24,
        prompt=prompt,
    )

    assert "title" in analysis.missing_sections
    assert "description" in analysis.missing_sections
    assert "keywords" in analysis.missing_sections


def test_analyze_generation_text_catalog_contract_length_violations() -> None:
    """Contract checks should flag title and keyword-count violations."""
    prompt = (
        "Analyze this image for cataloguing metadata.\n"
        "Return exactly these three sections, and nothing else:\n"
        "Title: 5-10 words.\nDescription: 1-2 factual sentences.\nKeywords: 10-18 terms."
    )
    text = (
        "Title: Short title\n"
        "Description: A misty loch scene with conifer trees and utility infrastructure.\n"
        "Keywords: loch, mist, trees, building"
    )

    analysis = check_models.analyze_generation_text(
        text,
        generated_tokens=48,
        prompt=prompt,
    )

    assert analysis.title_word_count is not None
    assert analysis.title_word_count < check_models.QUALITY.min_title_words
    assert analysis.keyword_count is not None
    assert analysis.keyword_count < check_models.QUALITY.min_keywords_count


def test_analyze_generation_text_detects_reasoning_leak() -> None:
    """Chain-of-thought markers should be flagged explicitly."""
    text = "<think>Let's analyze the image before answering.</think> Title: Misty loch landscape"

    analysis = check_models.analyze_generation_text(text, generated_tokens=40)

    assert analysis.has_reasoning_leak is True
    assert analysis.reasoning_leak_markers


def test_analyze_generation_text_detects_context_echo() -> None:
    """Verbatim prompt-context blocks in output should be flagged as context echo."""
    prompt = (
        "Analyze this image.\n"
        "Context: Existing metadata hints:\n"
        "- Title hint: Misty Loch Katrine shoreline\n"
        "- Description hint: A misty loch scene with evergreen trees and a utility building.\n"
        "- Capture metadata: Taken on 2026-02-21 16:14:55 GMT.\n"
    )
    text = (
        "Context: Existing metadata hints:\n"
        "Title hint: Misty Loch Katrine shoreline\n"
        "Description hint: A misty loch scene with evergreen trees and a utility building.\n"
        "Capture metadata: Taken on 2026-02-21 16:14:55 GMT."
    )

    analysis = check_models.analyze_generation_text(
        text,
        generated_tokens=64,
        prompt=prompt,
    )

    assert analysis.has_context_echo is True
    assert analysis.context_echo_ratio > 0


def test_analyze_generation_text_marks_prompt_check_availability() -> None:
    """Analysis should record whether prompt-dependent checks were able to run."""
    prompt = "Describe the image.\nContext: Title hint: Brick storefront scene"

    with_prompt = check_models.analyze_generation_text(
        "Title: Brick storefront scene",
        generated_tokens=12,
        prompt=prompt,
    )
    without_prompt = check_models.analyze_generation_text(
        "Title: Brick storefront scene",
        generated_tokens=12,
    )

    assert with_prompt.prompt_checks_ran is True
    assert without_prompt.prompt_checks_ran is False


def test_analyze_generation_text_classifies_instruction_echo() -> None:
    """Prompt-instruction parroting should be surfaced as a model shortcoming."""
    prompt = (
        "Analyze this image for cataloguing metadata.\n"
        "Return exactly these three sections, and nothing else:\n"
        "Title: 5-10 words.\nDescription: 1-2 factual sentences.\nKeywords: 10-18 terms."
    )
    text = "Return exactly these three sections, and nothing else. Title: 5-10 words."

    analysis = check_models.analyze_generation_text(
        text,
        generated_tokens=24,
        prompt=prompt,
    )

    assert analysis.instruction_echo is True
    assert analysis.verdict == "model_shortcoming"
    assert "instruction_echo" in analysis.evidence


def test_analyze_generation_text_distinguishes_metadata_borrowing_from_trusted_hint_reuse() -> None:
    """Reuse of stripped metadata should be flagged separately from trusted hint preservation."""
    prompt = (
        "Analyze this image.\n"
        "Context: Existing metadata hints:\n"
        "- Title hint: Brick storefront with outdoor seating\n"
        "- Description hint: A brick storefront has outdoor seating beside a sidewalk.\n"
        "- Keyword hints: brick storefront, outdoor seating, sidewalk, people\n"
        "- Capture metadata: Taken on 2026-02-21 16:14:55 GMT in Welwyn Garden City.\n"
    )
    text = (
        "Title: Brick storefront with outdoor seating\n"
        "Description: A brick storefront has outdoor seating beside a sidewalk. "
        "Taken on 2026-02-21 16:14:55 GMT in Welwyn Garden City.\n"
        "Keywords: brick storefront, outdoor seating, sidewalk, people"
    )

    analysis = check_models.analyze_generation_text(
        text,
        generated_tokens=48,
        prompt=prompt,
    )

    assert analysis.metadata_borrowing is True
    assert analysis.hint_relationship in {
        "preserves_trusted_hints",
        "improves_trusted_hints",
    }
    assert analysis.verdict == "model_shortcoming"


def test_analyze_generation_text_detects_cutoff_when_cap_is_hit() -> None:
    """Exact-cap repetitive incomplete outputs should be classified as cutoff."""
    prompt = (
        "Analyze this image for cataloguing metadata.\n"
        "Return exactly these three sections, and nothing else:\n"
        "Title: 5-10 words.\nDescription: 1-2 factual sentences.\nKeywords: 10-18 terms."
    )
    text = (
        "Title: Brick storefront with outdoor seating\n"
        "Description: A brick storefront has outdoor seating beside a sidewalk.\n"
        "Keywords: brick storefront brick storefront brick storefront brick storefront"
    )

    analysis = check_models.analyze_generation_text(
        text,
        generated_tokens=60,
        requested_max_tokens=60,
        prompt=prompt,
    )

    assert analysis.likely_capped is True
    assert analysis.verdict in {"cutoff_degraded", "token_cap"}
    assert "token_cap" in analysis.evidence


def test_analyze_generation_text_uses_nontext_prompt_burden_for_context_budget() -> None:
    """Heavy non-text prompt burden should not be mislabeled as short-vs-prompt failure."""
    prompt = (
        "Analyze this image.\n"
        "Context: Existing metadata hints:\n"
        "- Title hint: Brick storefront with outdoor seating\n"
        "- Description hint: A brick storefront has outdoor seating beside a sidewalk.\n"
        "- Keyword hints: brick storefront, outdoor seating, sidewalk, people\n"
    )
    analysis = check_models.analyze_generation_text(
        "Title: Storefront scene\nDescription: A storefront.\nKeywords: storefront, street",
        generated_tokens=14,
        prompt_tokens=5000,
        prompt=prompt,
    )

    assert analysis.prompt_tokens_total == 5000
    assert analysis.prompt_tokens_nontext_est is not None
    assert analysis.prompt_tokens_text_est is not None
    assert analysis.prompt_tokens_nontext_est > analysis.prompt_tokens_text_est
    assert analysis.verdict == "context_budget"


def test_analyze_generation_text_ignores_nonvisual_location_and_time_metadata() -> None:
    """Location, GPS, and timestamps should not drive trusted-hint penalties."""
    prompt = (
        "Analyze this image.\n"
        "MyMarker: Title hint: Brick storefront scene. Description hint: A brick shopfront. "
        "Capture metadata: Taken on 2026-02-21 16:14:55 GMT. GPS: 51.8000, -0.2000. "
        "Welwyn Garden City, England.\n"
    )
    text = "Title: Brick storefront scene\nDescription: A brick shopfront.\nKeywords: brick, shop"

    analysis = check_models.analyze_generation_text(
        text,
        generated_tokens=18,
        prompt=prompt,
        context_marker="MyMarker:",
    )

    assert analysis.is_context_ignored is False
    assert "Welwyn" not in " ".join(analysis.missing_context_terms)


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


def test_detect_language_mixing_uses_passed_quality_thresholds() -> None:
    """Detector should honor per-call pattern overrides via QualityThresholds."""
    custom_quality = check_models.QualityThresholds(
        patterns={
            "tokenizer_artifacts": [r"CUSTOMTOKEN"],
            "code_patterns": [r"\bCUSTOMCALL\("],
        },
    )

    has_mixing, issues = check_models._detect_language_mixing(
        "CUSTOMTOKEN and then CUSTOMCALL(value)",
        quality_thresholds=custom_quality,
    )

    assert has_mixing is True
    assert "tokenizer_artifact" in issues
    assert "code_snippet" in issues


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


def test_generation_quality_issues_property_handles_missing_keyword_dup_ratio() -> None:
    """Issues property should not format None keyword_duplication_ratio."""
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

    issues = analysis.issues
    assert issues
    assert any("HARNESS" in issue for issue in issues)


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


class TestClassifyHintRelationship:
    """Tests for _classify_hint_relationship nonvisual-hint handling."""

    def test_metadata_only_hints_not_scored_as_ignores(self) -> None:
        """When all trusted terms are metadata-only, don't penalise for low visual overlap."""
        bundle = check_models.TrustedHintBundle(
            trusted_text="Taken on 2026-02-21, GPS coordinates 51.8N",
            trusted_terms=("taken", "2026-02-21", "gps", "timestamp"),
            nonvisual_terms=("taken", "2026-02-21", "gps", "timestamp"),
        )
        text = (
            "Title: Red brick storefront\n"
            "Description: A red brick storefront with outdoor seating.\n"
            "Keywords: brick, storefront, seating"
        )
        relationship, _ = check_models._classify_hint_relationship(text, bundle)
        assert relationship != "ignores_trusted_hints", (
            "Metadata-only hints should not trigger ignores_trusted_hints"
        )

    def test_visual_hints_still_detected_as_ignores(self) -> None:
        """When visual terms are present and not reflected, still flag as ignores."""
        bundle = check_models.TrustedHintBundle(
            trusted_text="Brick storefront with outdoor seating beside sidewalk",
            trusted_terms=("Brick", "storefront", "outdoor", "seating", "sidewalk"),
            nonvisual_terms=(),
        )
        text = "Title: Abstract painting\nDescription: Colorful abstract art.\nKeywords: art"
        relationship, _ = check_models._classify_hint_relationship(text, bundle)
        assert relationship == "ignores_trusted_hints"


class TestClassifyReviewVerdict:
    """Tests for _classify_review_verdict verdict splitting."""

    def test_token_cap_without_degradation(self) -> None:
        """Token cap hit with no degradation reasons returns token_cap."""
        verdict, _ = check_models._classify_review_verdict(
            has_harness_issue=False,
            harness_type=None,
            likely_cutoff=True,
            cutoff_reasons=[],
            prompt_tokens_total=100,
            prompt_tokens_text_est=80,
            prompt_tokens_nontext_est=20,
            missing_sections=[],
            utility_grade="A",
            instruction_echo=False,
            metadata_borrowing=False,
            has_hallucination=False,
        )
        assert verdict == "token_cap"

    def test_cutoff_degraded_with_missing_sections(self) -> None:
        """Token cap hit with missing_sections returns cutoff_degraded."""
        verdict, _ = check_models._classify_review_verdict(
            has_harness_issue=False,
            harness_type=None,
            likely_cutoff=True,
            cutoff_reasons=["missing_sections"],
            prompt_tokens_total=100,
            prompt_tokens_text_est=80,
            prompt_tokens_nontext_est=20,
            missing_sections=["Keywords"],
            utility_grade="C",
            instruction_echo=False,
            metadata_borrowing=False,
            has_hallucination=False,
        )
        assert verdict == "cutoff_degraded"

    def test_cutoff_degraded_with_abrupt_tail(self) -> None:
        """Token cap hit with abrupt_tail returns cutoff_degraded, not token_cap."""
        verdict, evidence = check_models._classify_review_verdict(
            has_harness_issue=False,
            harness_type=None,
            likely_cutoff=True,
            cutoff_reasons=["abrupt_tail"],
            prompt_tokens_total=100,
            prompt_tokens_text_est=80,
            prompt_tokens_nontext_est=20,
            missing_sections=[],
            utility_grade="C",
            instruction_echo=False,
            metadata_borrowing=False,
            has_hallucination=False,
        )
        assert verdict == "cutoff_degraded"
        assert "abrupt_tail" in evidence

    def test_harness_verdict_preserved(self) -> None:
        """Harness issues still return harness verdict."""
        verdict, _ = check_models._classify_review_verdict(
            has_harness_issue=True,
            harness_type="encoding",
            likely_cutoff=False,
            cutoff_reasons=[],
            prompt_tokens_total=100,
            prompt_tokens_text_est=80,
            prompt_tokens_nontext_est=20,
            missing_sections=[],
            utility_grade="B",
            instruction_echo=False,
            metadata_borrowing=False,
            has_hallucination=False,
        )
        assert verdict == "harness"


class TestClassifyUserBucket:
    """Tests for _classify_user_bucket with new verdict types."""

    def test_token_cap_grade_a_recommended(self) -> None:
        """A-grade token_cap models should be recommended."""
        bucket = check_models._classify_user_bucket(
            verdict="token_cap",
            hint_relationship="preserves_trusted_hints",
            has_contract_issue=False,
            utility_grade="A",
        )
        assert bucket == "recommended"

    def test_cutoff_degraded_avoid(self) -> None:
        """cutoff_degraded models should be avoid."""
        bucket = check_models._classify_user_bucket(
            verdict="cutoff_degraded",
            hint_relationship="preserves_trusted_hints",
            has_contract_issue=False,
            utility_grade="C",
        )
        assert bucket == "avoid"

    def test_runtime_failure_avoid(self) -> None:
        """runtime_failure models should be avoid."""
        bucket = check_models._classify_user_bucket(
            verdict="runtime_failure",
            hint_relationship="preserves_trusted_hints",
            has_contract_issue=False,
            utility_grade="F",
        )
        assert bucket == "avoid"

    def test_clean_grade_a_recommended(self) -> None:
        """Clean A-grade models should be recommended."""
        bucket = check_models._classify_user_bucket(
            verdict="clean",
            hint_relationship="preserves_trusted_hints",
            has_contract_issue=False,
            utility_grade="A",
        )
        assert bucket == "recommended"
