"""Tests for quality analysis utilities."""

from __future__ import annotations

import argparse
import dataclasses
from unittest.mock import patch

import pytest

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


def test_analyze_generation_text_flags_short_token_noise() -> None:
    """Mechanically short gibberish should not be classified as clean."""
    analysis = check_models.analyze_generation_text(" ''){ GREE, which is not included.", 9)

    assert analysis.text_sanity_issue_type == "gibberish(token_noise)"
    assert analysis.verdict == "semantic_mismatch"
    assert "text_sanity" in analysis.evidence


def test_analyze_generation_text_flags_cjk_latin_token_soup() -> None:
    """Mixed CJK/Latin token soup should not be classified as a clean caption."""
    text = (
        'open对不同方面">black/ with小猫小猫kotPicture •0超高清比!y表面处理超经典的!'
        "张图片'七- object Tno-go-head-or U0.C在其他 ** ,Not只!被i animal "
        "...'s*: .# • 模型被 partially \" alsoDifferent一\n"
        ',  Germanyc-under,"开Picture顶 noncolor_over宠关 feature PETwith对上 from!'
    )

    analysis = check_models.analyze_generation_text(text, 85)

    assert analysis.text_sanity_issue_type == "gibberish(mixed_script_noise)"
    assert analysis.verdict == "semantic_mismatch"
    assert analysis.user_bucket == "avoid"
    assert "text_sanity" in analysis.evidence


def test_repetitive_token_cap_is_generation_loop_diagnostic() -> None:
    """Max-token repetition should carry an explicit generation-loop signal."""
    analysis, issues = check_models._analyze_text_quality(
        "word " * 200,
        200,
        prompt_tokens=240,
        requested_max_tokens=200,
    )

    assert analysis.generation_loop_type == "repetitive_tail"
    assert analysis.verdict == "cutoff_degraded"
    assert issues is not None
    assert "generation_loop" in issues


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
    new_quality = dataclasses.replace(
        check_models.QUALITY,
        patterns={
            "hallucination_question_indicators": ["bespoke hallucination prompt"],
            "hallucination_edu_keywords": ["custom classroom leak"],
        },
    )
    with patch.object(check_models, "QUALITY", new_quality):
        issues = check_models._detect_hallucination_patterns(
            "This long output includes a bespoke hallucination prompt and a custom classroom leak. "
            * 4,
        )
        assert any("question" in issue.lower() for issue in issues)
        assert any("educational" in issue.lower() for issue in issues)


def test_analyze_generation_text_excessive_verbosity() -> None:
    """Triage-length output should be flaggable before its 200-token cap."""
    text = "The image shows " * 50 + "\n### Analysis\n" + "content " * 50
    analysis = check_models.analyze_generation_text(text, 180)

    assert analysis.is_verbose


def test_analyze_generation_text_formatting_violations() -> None:
    """Test detection of formatting issues."""
    # HTML tags are a formatting violation
    text = "Here is the answer:\n<div>content</div>\n<span>more</span>"
    analysis = check_models.analyze_generation_text(text, 10)

    assert len(analysis.formatting_issues) > 0


def test_analyze_generation_text_preserves_raw_unknown_tags_for_logs() -> None:
    """Formatting issues should keep raw tags so CLI warnings stay readable."""
    text = "<think>internal reasoning</think> Title: Misty loch landscape"

    analysis = check_models.analyze_generation_text(text, 10)

    assert "Unknown tags: <think>" in analysis.formatting_issues
    assert all("&lt;think&gt;" not in issue for issue in analysis.formatting_issues)


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


def test_near_empty_output_promotes_to_unknown_anomaly() -> None:
    """Near-empty output with grade F promotes clean → unknown_runtime_anomaly."""
    # Text must be short enough to trigger anomaly (<20 chars) but long enough
    # to avoid the harness detector's "too few tokens" check.
    analysis = check_models.analyze_generation_text("A nice photo.", 13)
    assert analysis.verdict == "unknown_runtime_anomaly"
    assert analysis.user_bucket == "caveat"


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
    assert analysis.has_thinking_trace is False


def test_analyze_generation_text_treats_expected_thinking_trace_as_information() -> None:
    """Documented delimiters from thinking models should be informational."""
    text = "◁think▷Inspect the scene.◁/think▷ Title: Two boats on a river"

    analysis = check_models.analyze_generation_text(
        text,
        generated_tokens=24,
        model_name="mlx-community/Kimi-VL-A3B-Thinking-2506-bf16",
    )

    assert analysis.has_thinking_trace is True
    assert analysis.thinking_trace_incomplete is False
    assert analysis.thinking_trace_markers == ["◁think▷", "◁/think▷"]
    assert analysis.has_reasoning_leak is False
    assert analysis.verdict == "clean"
    assert analysis.user_bucket == "caveat"
    assert check_models._build_quality_issues_string(analysis) == "thinking-trace"


def test_analyze_generation_text_flags_unclosed_expected_thinking_trace() -> None:
    """An unclosed thinking trace should explain a degraded capped result."""
    prompt = (
        "Return exactly these three sections:\n"
        "Title: 5-10 words.\n"
        "Description: 1-2 sentences.\n"
        "Keywords: 10-18 terms."
    )

    analysis = check_models.analyze_generation_text(
        "◁think▷Inspecting the image step by step.",
        generated_tokens=500,
        prompt=prompt,
        requested_max_tokens=500,
        model_name="mlx-community/Kimi-VL-A3B-Thinking-8bit",
    )

    assert analysis.has_thinking_trace is True
    assert analysis.thinking_trace_incomplete is True
    assert analysis.thinking_trace_markers == ["◁think▷"]
    assert analysis.has_reasoning_leak is False
    assert "thinking_incomplete" in analysis.evidence
    assert analysis.verdict == "cutoff_degraded"
    assert analysis.user_bucket == "avoid"
    quality_issues = check_models._build_quality_issues_string(analysis) or ""
    assert "thinking-incomplete" in quality_issues
    assert "reasoning-leak" not in quality_issues


def test_wrapped_valid_output_uses_normalized_scoring_copy() -> None:
    """Known wrappers should not hide valid sections or mutate retained raw output."""
    raw = (
        "<|assistant|>\n"
        "Title: Craftspeople restoring a wooden workbench\n"
        "Description: Two craftspeople repair a worn bench in a bright workshop.\n"
        "Keywords: workshop, craftspeople, workbench, repair, tools, timber, restoration, "
        "indoors, craft, furniture\n"
        "<|end|>"
    )
    known_tokens = ("<|assistant|>", "<|end|>")

    normalized = check_models._normalize_output_for_analysis(
        raw,
        known_special_tokens=known_tokens,
    )
    analysis = check_models.analyze_generation_text(
        raw,
        generated_tokens=64,
        prompt=(
            "Analyze this image for cataloguing metadata.\n"
            "Return exactly these three sections, and nothing else:\n"
            "Title: 5-10 words.\nDescription: 1-2 factual sentences.\n"
            "Keywords: 10-18 terms."
        ),
        known_special_tokens=known_tokens,
    )

    assert raw.startswith("<|assistant|>")
    assert normalized.text.startswith("Title:")
    assert normalized.removed_wrappers == known_tokens
    assert analysis.missing_sections == []
    assert analysis.special_token_wrappers == list(known_tokens)
    assert "special_token_wrapper" in analysis.evidence


def test_coherent_capped_reasoning_is_not_mechanical_corruption() -> None:
    """A coherent unfinished trace should be a budget issue, not token soup."""
    text = (
        "◁think▷The shoreline contains several small boats beside a timber jetty. "
        "The calm water reflects the hulls, while low buildings and trees form "
        "the background. I should now compose the requested catalog description"
    )

    analysis = check_models.analyze_generation_text(
        text,
        generated_tokens=80,
        requested_max_tokens=80,
        model_name="example/Thinking-Vision-Model",
    )

    assert analysis.thinking_trace_incomplete is True
    assert analysis.text_sanity_issue_type is None
    assert analysis.is_repetitive is False
    assert analysis.has_degeneration is False
    assert "reasoning_budget_exhausted" in analysis.evidence


def test_format_quality_analysis_for_log_distinguishes_incomplete_thinking() -> None:
    """Compact logs should preserve informational and incomplete trace states."""
    analysis = check_models.analyze_generation_text(
        "◁think▷Inspecting the image step by step.",
        generated_tokens=20,
        model_name="org/Kimi-VL-Thinking",
    )

    quality_log = check_models._format_quality_analysis_for_log(analysis)

    assert "thinking_trace=True (◁think▷)" in quality_log
    assert "thinking_incomplete=True" in quality_log
    assert "reasoning_leak" not in quality_log


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
    assert analysis.user_bucket in {"avoid", "caveat"}
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
        "The image is a photograph.",
        generated_tokens=14,
        prompt_tokens=5000,
        prompt=prompt,
    )

    assert analysis.prompt_tokens_total == 5000
    assert analysis.prompt_tokens_nontext_est is not None
    assert analysis.prompt_tokens_text_est is not None
    assert analysis.prompt_tokens_nontext_est > analysis.prompt_tokens_text_est
    assert analysis.verdict == "context_budget"


def test_large_image_short_text_is_visual_input_burden() -> None:
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
        specificity_score=1.0,
        has_language_mixing=False,
        language_mixing_issues=[],
        has_degeneration=False,
        degeneration_type=None,
        has_fabrication=False,
        fabrication_issues=[],
        prompt_tokens_total=16700,
        prompt_tokens_text_est=430,
        prompt_tokens_nontext_est=16270,
    )
    result = check_models.PerformanceResult(
        model_name="org/visual-heavy",
        generation=None,
        success=True,
        quality_analysis=analysis,
        prompt_diagnostics=check_models.PromptDiagnostics(image_placeholder_count=1),
    )
    profile = check_models.ImageInputProfile(
        width=9504,
        height=6336,
        megapixels=60.2,
        processed_width=1344,
        processed_height=896,
    )

    burden = check_models._prompt_burden_for_result(result, profile)

    assert burden.kind == "visual_input"
    assert burden.text_tokens_est == 430
    assert burden.nontext_tokens_est == 16270
    assert burden.visual_tokens_est is None
    assert burden.source == "estimated_nontext"
    assert burden.processed_width == 1344
    assert burden.processed_height == 896


def test_small_image_long_text_is_text_burden() -> None:
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
        specificity_score=1.0,
        has_language_mixing=False,
        language_mixing_issues=[],
        has_degeneration=False,
        degeneration_type=None,
        has_fabrication=False,
        fabrication_issues=[],
        prompt_tokens_total=4200,
        prompt_tokens_text_est=3900,
        prompt_tokens_nontext_est=300,
    )
    result = check_models.PerformanceResult(
        model_name="org/text-heavy",
        generation=None,
        success=True,
        quality_analysis=analysis,
        prompt_diagnostics=check_models.PromptDiagnostics(image_placeholder_count=1),
    )
    profile = check_models.ImageInputProfile(width=640, height=480, megapixels=0.3)

    burden = check_models._prompt_burden_for_result(result, profile)

    assert burden.kind == "text"
    assert burden.nontext_ratio is not None
    assert burden.nontext_ratio < 0.5


def test_known_total_without_component_estimates_is_unavailable_burden() -> None:
    analysis = dataclasses.replace(
        check_models.analyze_generation_text(
            "A concise image description.",
            generated_tokens=6,
            prompt_tokens=4200,
            prompt="Describe this image.",
        ),
        prompt_tokens_text_est=None,
        prompt_tokens_nontext_est=None,
    )
    result = check_models.PerformanceResult(
        model_name="org/unavailable-components",
        generation=None,
        success=True,
        quality_analysis=analysis,
    )

    burden = check_models._prompt_burden_for_result(result, None)

    assert burden.total_tokens == 4200
    assert burden.kind == "unavailable"
    assert burden.source == "unavailable"
    assert burden.reason == "component_estimates_unavailable"


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


def test_metadata_provenance_keeps_location_out_of_draft_terms() -> None:
    """Location hints should be authoritative rather than draft descriptive terms."""
    provenance = check_models._build_metadata_provenance(
        {
            "title": "Deben Estuary at Woodbridge",
            "description": "Two sailing boats with trees behind.",
            "keywords": "Deben Estuary, Woodbridge, England, boats, trees",
        }
    )

    assert "Deben Estuary" in provenance.authoritative_terms
    assert "Woodbridge" in provenance.authoritative_terms
    assert "boats" in {term.casefold() for term in provenance.draft_terms}
    assert "woodbridge" not in {term.casefold() for term in provenance.draft_terms}


def test_metadata_provenance_prompt_round_trips_through_trusted_hint_bundle() -> None:
    """New assisted prompt labels should remain readable by compatibility checks."""
    prompt = check_models.prepare_prompt(
        argparse.Namespace(prompt=None, eval_mode="assisted"),
        {
            "title": "Deben Estuary at Woodbridge",
            "description": "Two boats on a river.",
            "keywords": "Deben Estuary, Woodbridge, boats, river",
            "date": "2026-07-04",
            "time": "19:10:04",
            "gps": "52.0,-1.0",
        },
    )

    bundle = check_models._extract_trusted_hint_bundle(prompt)

    assert bundle.trusted_text
    assert "Description: Two boats on a river" in bundle.trusted_text
    assert "Keywords: boats, river" in bundle.trusted_text
    assert not any(line.startswith("Title:") for line in bundle.trusted_text.splitlines())
    assert {"boats", "river"}.issubset({term.casefold() for term in bundle.trusted_terms})
    nonvisual_text = " ".join(bundle.nonvisual_terms)
    assert "Deben Estuary" in nonvisual_text
    assert "Woodbridge" in nonvisual_text
    assert "2026-07-04 19:10:04" in nonvisual_text
    assert "52.0,-1.0" in nonvisual_text


def test_assisted_location_use_is_context_integration_not_borrowing() -> None:
    metadata: check_models.MetadataDict = {
        "title": "Deben Estuary at Woodbridge",
        "description": "Two boats on a river.",
        "keywords": "Deben Estuary, Woodbridge, boats, river",
    }
    text = (
        "Title: Sailboats on Deben Estuary at Woodbridge\n"
        "Description: Two moored sailboats stand before a wooded bank.\n"
        "Keywords: sailboats, Deben Estuary, Woodbridge, river, wooded bank"
    )

    metrics = check_models.compute_metadata_agreement(text, metadata)

    assert metrics.context_integration_score is not None
    assert metrics.context_integration_score > 0
    assert metrics.nonvisual_penalty == 0


def test_assisted_location_prompt_use_does_not_set_legacy_borrowing_flag() -> None:
    prompt = check_models.prepare_prompt(
        argparse.Namespace(prompt=None, eval_mode="assisted"),
        {
            "title": "Deben Estuary at Woodbridge",
            "description": "Two boats on a river.",
            "keywords": "Deben Estuary, Woodbridge, boats, river",
        },
    )
    text = (
        "Title: Sailboats on Deben Estuary at Woodbridge\n"
        "Description: Two sailboats rest on calm water.\n"
        "Keywords: sailboats, Deben Estuary, Woodbridge, river"
    )

    analysis = check_models.analyze_generation_text(text, generated_tokens=30, prompt=prompt)

    assert analysis.metadata_borrowing is False
    assert "unverified-context-copy" not in analysis.evidence


def test_assisted_capture_metadata_copy_uses_unverified_context_label() -> None:
    prompt = check_models.prepare_prompt(
        argparse.Namespace(prompt=None, eval_mode="assisted"),
        {
            "description": "Two boats on a river.",
            "keywords": "boats, river",
            "date": "2026-07-04",
            "time": "19:10:04",
            "gps": "52.0,-1.0",
        },
    )
    text = (
        "Title: Two boats on a river\n"
        "Description: Two boats captured on 2026-07-04 at 19:10:04 near 52.0,-1.0.\n"
        "Keywords: boats, river"
    )

    analysis = check_models.analyze_generation_text(text, generated_tokens=30, prompt=prompt)

    assert analysis.metadata_borrowing is True
    assert "unverified-context-copy" in analysis.evidence
    assert "unverified-context-copy" in (check_models._build_quality_issues_string(analysis) or "")


def test_assisted_date_only_copy_uses_unverified_context_label() -> None:
    prompt = check_models.prepare_prompt(
        argparse.Namespace(prompt=None, eval_mode="assisted"),
        {
            "description": "Two boats on a river.",
            "keywords": "boats, river",
            "date": "2026-07-04",
            "time": "19:10:04",
        },
    )
    text = (
        "Title: Two boats on a river\n"
        "Description: Two boats photographed on 2026-07-04 beside calm water.\n"
        "Keywords: boats, river"
    )

    analysis = check_models.analyze_generation_text(text, generated_tokens=28, prompt=prompt)

    assert analysis.metadata_borrowing is True
    assert "unverified-context-copy" in analysis.evidence


def test_assisted_time_only_copy_uses_unverified_context_label() -> None:
    prompt = check_models.prepare_prompt(
        argparse.Namespace(prompt=None, eval_mode="assisted"),
        {
            "description": "Two boats on a river.",
            "keywords": "boats, river",
            "date": "2026-07-04",
            "time": "19:10:04",
        },
    )
    text = (
        "Title: Two boats on a river\n"
        "Description: Two boats photographed at 19:10:04 beside calm water.\n"
        "Keywords: boats, river"
    )

    analysis = check_models.analyze_generation_text(text, generated_tokens=28, prompt=prompt)

    assert analysis.metadata_borrowing is True
    assert "unverified-context-copy" in analysis.evidence


def test_verbatim_draft_is_low_improvement_not_hallucination() -> None:
    metadata: check_models.MetadataDict = {
        "description": "Two boats on a river.",
        "keywords": "boats, river",
    }
    text = "Title: Two boats on a river\nDescription: Two boats on a river.\nKeywords: boats, river"
    metrics = check_models.compute_metadata_agreement(text, metadata)
    analysis = check_models.analyze_generation_text(
        text,
        generated_tokens=20,
        prompt_tokens=120,
        prompt="Return Title, Description, and Keywords.",
    )

    assert analysis.metadata_borrowing is False
    assert metrics.draft_improvement_score is not None
    assert metrics.draft_improvement_score < 50
    assert not analysis.hallucination_issues


def test_draft_improvement_rewards_corrected_enriched_description() -> None:
    provenance = check_models._build_metadata_provenance(
        {
            "description": "Two boats on a river.",
            "keywords": "boats, river",
        }
    )
    copied = check_models._score_draft_improvement(
        "Description: Two boats on a river.\nKeywords: boats, river",
        provenance,
    )
    enriched = check_models._score_draft_improvement(
        (
            "Title: Moored sailboats beside a wooded estuary\n"
            "Description: Two white sailboats rest on calm water before a dense wooded bank.\n"
            "Keywords: sailboats, estuary, moored, calm water, wooded bank, reflections"
        ),
        provenance,
    )

    assert copied is not None
    assert enriched is not None
    assert enriched > copied
    assert enriched >= 50


def test_draft_improvement_allows_concise_legitimate_overlap() -> None:
    provenance = check_models._build_metadata_provenance(
        {
            "description": "A boat on a river.",
            "keywords": "boat, river",
        }
    )

    score = check_models._score_draft_improvement(
        "A blue boat crosses the broad river at sunset.",
        provenance,
    )

    assert score is not None
    assert 20 <= score < 100


@pytest.mark.parametrize(
    "structured_text",
    [
        "GPS 51.5014, -0.1419; GPS 51.5014, -0.1419; GPS 51.5014, -0.1419",
        "51°30'05\"N 0°08'31\"W; 51°30'05\"N 0°08'31\"W",
        "Captured 12:34:56, processed 12:34:56, exported 12:34:56",
        "Image dimensions 1920x1080; preview 1920x1080; export 1920x1080",
        "Exposure 1/125 sec; alternate 1/125 sec; selected 1/125 sec",
    ],
    ids=["decimal_coordinates", "dms_coordinates", "timestamps", "dimensions", "exposure"],
)
def test_structured_numeric_metadata_is_not_a_numeric_loop(structured_text: str) -> None:
    assert check_models._text_sanity_numeric_loop_issue(structured_text) is None


def test_true_repeated_number_still_triggers_numeric_loop() -> None:
    assert check_models._text_sanity_numeric_loop_issue("42 42 42 42 42 42") == "numeric_loop"


def test_repeated_factual_values_in_prose_are_not_a_numeric_loop() -> None:
    """Separated factual reuse should not look like contiguous degeneration."""
    text = (
        "The market opened in 2024 with 42 stalls. By noon, 42 vendors had served "
        "visitors, and the 2024 programme still listed 42 independent makers."
    )

    assert check_models._text_sanity_numeric_loop_issue(text) is None


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


def test_detect_output_degeneration_allows_markdown_emphasis() -> None:
    """Markdown emphasis punctuation should not be treated as degeneration."""
    text = (
        "The image appears to show an outdoor scene.\n\n"
        "**Objects:**\n"
        "- A brick building\n"
        "- A parked car\n\n"
        "**Keywords:**"
    )

    has_degen, degen_type = check_models._detect_output_degeneration(text)

    assert has_degen is False
    assert degen_type is None


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
    new_quality = dataclasses.replace(
        check_models.QUALITY,
        patterns={
            "fabrication_url_patterns": [r"https?://[^\s]+"],
            "fabrication_suspicious_url_keywords": ["customtokenxyz"],
            "fabrication_precise_stat_patterns": [r"\bwonky\d+\b"],
        },
    )
    with patch.object(check_models, "QUALITY", new_quality):
        text = "See https://domain.test/customtokenxyz/path with wonky1 and wonky2."
        has_fab, issues = check_models._detect_fabricated_details(text)
        assert has_fab is True
        assert any("suspicious_url" in issue for issue in issues)
        assert any("suspicious_precision" in issue for issue in issues)


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


def test_image_heavy_short_caption_does_not_become_prompt_template_harness() -> None:
    """Useful short captions should not fail only because image tokens dominate the prompt."""
    analysis = check_models.analyze_generation_text(
        text="Two cats are sleeping on a pink blanket on a couch.",
        generated_tokens=13,
        prompt_tokens=1196,
        prompt="Describe this image briefly.",
        requested_max_tokens=200,
    )

    assert analysis.has_harness_issue is False
    assert analysis.harness_issue_type is None
    assert analysis.verdict == "clean"
    assert analysis.user_bucket == "recommended"
    assert not any("output_ratio" in item for item in analysis.issues)


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


def test_long_context_with_useful_caption_does_not_fail_on_ratio_alone() -> None:
    """Long image prompts need weak output evidence before becoming context-budget findings."""
    analysis = check_models.analyze_generation_text(
        text="Two cats are sleeping on a pink blanket on a couch.",
        generated_tokens=13,
        prompt_tokens=4103,
        prompt="Describe this image briefly.",
        requested_max_tokens=200,
    )

    assert analysis.has_harness_issue is False
    assert analysis.harness_issue_type is None
    assert analysis.verdict == "clean"


def test_prompt_without_trusted_metadata_marks_hint_handling_not_evaluated() -> None:
    """Plain prompts should not claim that output preserved trusted image hints."""
    analysis = check_models.analyze_generation_text(
        text="Two cats are sleeping on a pink blanket on a couch.",
        generated_tokens=13,
        prompt_tokens=1196,
        prompt="Describe this image briefly.",
        requested_max_tokens=200,
    )

    assert analysis.hint_relationship == "not_evaluated"


def test_prompt_with_visual_metadata_still_evaluates_hint_handling() -> None:
    """Metadata-grounded prompts should continue to score trusted visual hints."""
    prompt = (
        "Context:\n"
        "Title: Two tabby cats resting\n"
        "Description: Two tabby cats rest on a bright pink couch with two remotes.\n"
        "Keywords: cats, tabby, pink couch, remote controls\n\n"
        "Describe this image briefly."
    )
    analysis = check_models.analyze_generation_text(
        text="Two tabby cats rest on a bright pink couch with two remote controls.",
        generated_tokens=16,
        prompt_tokens=260,
        prompt=prompt,
        requested_max_tokens=200,
    )

    assert analysis.hint_relationship == "preserves_trusted_hints"


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
    new_quality = dataclasses.replace(
        check_models.QUALITY,
        patterns={
            "training_leak_instruction_header_patterns": [r"\n### CUSTOM LEAK ###"],
        },
    )
    with patch.object(check_models, "QUALITY", new_quality):
        text = (
            "A detailed response about architecture and people in the square. "
            "It includes grounded visual details and contextual cues.\n"
            "### CUSTOM LEAK ###\n"
            "Write a short story about a traveler."
        )
        has_leak, leak_type = check_models._detect_training_data_leak(text)
        assert has_leak is True
        assert leak_type == "instruction_header"


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


def test_format_quality_analysis_for_log_preserves_detail_order() -> None:
    """Compact log output should preserve the established field ordering."""
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
        word_count=307,
    )
    repeat_marker = "loop"
    analysis = dataclasses.replace(
        analysis,
        is_repetitive=True,
        repeated_token=repeat_marker,
        hallucination_issues=["table"],
        is_verbose=True,
        formatting_issues=["HTML tag leak"],
        has_excessive_bullets=True,
        bullet_count=7,
        is_context_ignored=True,
        is_refusal=True,
        refusal_type="safety",
        is_generic=True,
        specificity_score=0.3,
        has_language_mixing=True,
        has_degeneration=True,
        degeneration_type="encoding",
        has_fabrication=True,
        missing_sections=["title", "keywords"],
        title_word_count=4,
        description_sentence_count=3,
        keyword_count=22,
        keyword_duplication_ratio=0.57,
        has_reasoning_leak=True,
        reasoning_leak_markers=["<think>"],
        has_context_echo=True,
        context_echo_ratio=0.91,
        has_harness_issue=True,
        harness_issue_type="stop_token",
        harness_issue_details=["token_leak:<|end|>", "token_leak:<|eot|>"],
        instruction_echo=True,
        metadata_borrowing=True,
        hint_relationship="ignores_trusted_hints",
        verdict="cutoff_degraded",
        user_bucket="avoid",
        likely_capped=True,
    )

    assert check_models._format_quality_analysis_for_log(analysis) == (
        "repetitive=True (token=loop), refusal=True (type=safety), "
        "language_mixing=True, hallucination=True, generic=True (score=0.3), "
        "verbose=True, formatting_issues=True, excessive_bullets=True (count=7), "
        "context_ignored=True, degeneration=True (encoding), fabrication=True, "
        "missing_sections=title+keywords, title_words=4, description_sentences=3, "
        "keywords=22, keyword_dup=0.57, reasoning_leak=True (<think>), "
        "context_echo=True (0.91), instruction_echo=True, metadata_borrowing=True, "
        "hint_relationship=ignores_trusted_hints, verdict=cutoff_degraded, "
        "user_bucket=avoid, likely_capped=True, "
        "harness=True (stop_token; token_leak:<|end|>,token_leak:<|eot|>), words=307"
    )


def test_format_quality_analysis_for_log_clean_analysis_still_includes_word_count() -> None:
    """Clean analyses should still emit the lightweight word-count summary."""
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
        word_count=12,
    )

    assert check_models._format_quality_analysis_for_log(analysis) == "words=12"


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

    def test_visual_hints_with_no_overlap_are_a_weak_smell(self) -> None:
        """No overlap is evidence, but does not itself prove the output is unusable."""
        bundle = check_models.TrustedHintBundle(
            trusted_text="Brick storefront with outdoor seating beside sidewalk",
            trusted_terms=("Brick", "storefront", "outdoor", "seating", "sidewalk"),
            nonvisual_terms=(),
        )
        text = "Title: Abstract painting\nDescription: Colorful abstract art.\nKeywords: art"
        relationship, evidence = check_models._classify_hint_relationship(text, bundle)
        assert relationship == "no_overlap"
        assert evidence == ["no_overlap"]
        assert (
            check_models._classify_user_bucket(
                verdict="clean",
                hint_relationship=relationship,
                has_contract_issue=False,
            )
            != "avoid"
        )

    def test_omitting_individual_hint_terms_does_not_demote_clean_output(self) -> None:
        """One matching indicator is enough; absent reference terms are not recall failures."""
        bundle = check_models.TrustedHintBundle(
            trusted_text="Garden paths with flowers, hedges, benches, and trees",
            trusted_terms=("garden paths", "flowers", "hedges", "benches", "trees"),
        )
        text = (
            "Title: Curving garden path\n"
            "Description: A curved path passes flowering borders in a public garden.\n"
            "Keywords: garden path, flower border, public garden"
        )

        relationship, _evidence = check_models._classify_hint_relationship(text, bundle)

        assert relationship in {"preserves_trusted_hints", "improves_trusted_hints"}
        assert (
            check_models._classify_user_bucket(
                verdict="clean",
                hint_relationship=relationship,
                has_contract_issue=False,
            )
            == "recommended"
        )

    def test_no_overlap_plus_independent_failures_can_still_avoid(self) -> None:
        """The weak smell may support, but never replace, independent failure evidence."""
        assert (
            check_models._classify_user_bucket(
                verdict="model_shortcoming",
                hint_relationship="no_overlap",
                has_contract_issue=True,
            )
            == "avoid"
        )

    def test_primary_quality_prose_does_not_enumerate_absent_hint_terms(self) -> None:
        """Human diagnostics should describe the smell without implying keyword recall."""
        prompt = (
            "Context:\nKeyword hints: wooden benches, garden paths, flower borders\n\n"
            "Describe this image."
        )
        analysis = check_models.analyze_generation_text(
            "A night market glows beneath strings of lanterns.",
            generated_tokens=12,
            prompt=prompt,
        )

        prose = " ".join(analysis.issues)
        assert analysis.keyword_overlap == "not_assessable"
        assert "wooden benches" not in prose
        assert "garden paths" not in prose


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

    def test_grade_f_no_signals_stays_clean(self) -> None:
        """Grade F with no cutoff/harness/weak-output signals returns clean at verdict level."""
        verdict, _evidence = check_models._classify_review_verdict(
            has_harness_issue=False,
            harness_type=None,
            likely_cutoff=False,
            cutoff_reasons=[],
            prompt_tokens_total=100,
            prompt_tokens_text_est=80,
            prompt_tokens_nontext_est=20,
            missing_sections=[],
            utility_grade="F",
            instruction_echo=False,
            metadata_borrowing=False,
            has_hallucination=False,
        )
        assert verdict == "clean"


class TestClassifyUserBucket:
    """Tests for _classify_user_bucket with new verdict types."""

    def test_token_cap_grade_a_is_caveat(self) -> None:
        """Even A-grade token-capped output is not presentation-ready."""
        bucket = check_models._classify_user_bucket(
            verdict="token_cap",
            hint_relationship="preserves_trusted_hints",
            has_contract_issue=False,
            utility_grade="A",
        )
        assert bucket == "caveat"

    def test_metadata_alignment_normalizes_stale_token_cap_bucket(self) -> None:
        """Passing metadata alignment should retain canonical token-cap semantics."""
        base = check_models.analyze_generation_text(
            "A normal caption with varied image details and a complete sentence.",
            12,
        )
        analysis = dataclasses.replace(
            base,
            verdict="token_cap",
            user_bucket="recommended",
            evidence=["token_cap"],
            hint_relationship="preserves_trusted_hints",
        )
        metadata_agreement = check_models.MetadataAgreementMetrics(
            overall_score=90.0,
            matched_terms=("caption",),
            missed_terms=(),
        )

        refreshed = check_models._apply_metadata_alignment_to_analysis(
            analysis,
            metadata_agreement,
        )

        assert refreshed.verdict == "token_cap"
        assert refreshed.metadata_alignment_issue is None
        assert refreshed.user_bucket == "caveat"

    def test_low_draft_improvement_is_canonical_review_evidence(self) -> None:
        """Low draft scores should flow into canonical review evidence."""
        analysis = check_models.analyze_generation_text(
            "Title: Two boats\nDescription: Two boats on a river.\nKeywords: boats, river",
            20,
        )
        metadata_agreement = check_models.MetadataAgreementMetrics(
            overall_score=90.0,
            matched_terms=("boats",),
            draft_improvement_score=20.0,
        )

        refreshed = check_models._apply_metadata_alignment_to_analysis(
            analysis,
            metadata_agreement,
        )

        assert refreshed.draft_improvement_score == 20.0
        assert "low-draft-improvement" in refreshed.evidence

    def test_low_literal_metadata_overlap_does_not_override_clean_verdict(self) -> None:
        """Omitting optional draft terms must not turn a useful caption into a failure."""
        analysis = check_models.analyze_generation_text(
            (
                "Title: Curved glass skyscraper at night\n"
                "Description: A curved glass tower rises above a wet city street.\n"
                "Keywords: skyscraper, glass, night, street, bicycles, architecture"
            ),
            42,
        )
        metadata_agreement = check_models.MetadataAgreementMetrics(
            overall_score=12.0,
            matched_terms=("Skyscraper", "Night"),
            missed_terms=("Commuting", "Fenchurch Street", "Nightscape"),
            visual_description_score=95.0,
        )

        refreshed = check_models._apply_metadata_alignment_to_analysis(
            analysis,
            metadata_agreement,
        )

        assert refreshed.verdict == "clean"
        assert refreshed.user_bucket == "recommended"
        assert refreshed.metadata_alignment_issue == "low_metadata_alignment"

    def test_cutoff_degraded_avoid(self) -> None:
        """cutoff_degraded models should be avoid."""
        bucket = check_models._classify_user_bucket(
            verdict="cutoff_degraded",
            hint_relationship="preserves_trusted_hints",
            has_contract_issue=False,
            utility_grade="C",
        )
        assert bucket == "avoid"

    def test_unknown_runtime_anomaly_is_caveat(self) -> None:
        """Unknown runtime anomalies use the canonical caveat status."""
        bucket = check_models._classify_user_bucket(
            verdict="unknown_runtime_anomaly",
            hint_relationship="preserves_trusted_hints",
            has_contract_issue=False,
            utility_grade="F",
        )
        assert bucket == "caveat"

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


def test_assisted_prompt_keeps_complete_keyword_and_authoritative_context_policy() -> None:
    """Assisted prompts must not corrupt long metadata terms or forbid supplied facts."""
    complete_name = "The Fenchurch Building (The Walkie-Talkie)"
    prompt = check_models._build_cataloguing_prompt(
        {
            "title": f"{complete_name}, London",
            "description": "Walkie Talkie building known formally as 20 Fenchurch Street.",
            "keywords": f"Architecture, {complete_name}, London",
        },
    )

    assert complete_name in prompt
    assert "The Fenchurch Building (The Walki..." not in prompt
    assert "Authoritative context may supply identity and location" in prompt
    assert "unless supplied as authoritative context or visually obvious" in prompt


def test_empty_thinking_wrapper_is_presentation_warning_not_harness_failure() -> None:
    """An empty protocol wrapper should remain visible without becoming leaked reasoning."""
    text = (
        "<think>\n\n</think>\n\n"
        "Title: Curved glass tower at night\n"
        "Description: A curved glass tower rises above a city street.\n"
        "Keywords: tower, glass, night, street, architecture, city"
    )

    analysis = check_models.analyze_generation_text(
        text,
        45,
        prompt=(
            "Return exactly these three sections, and nothing else:\n"
            "Title:\nDescription:\nKeywords:"
        ),
    )

    assert analysis.verdict == "clean"
    assert analysis.user_bucket == "caveat"
    assert analysis.has_harness_issue is False
    assert analysis.has_reasoning_leak is False
    assert analysis.text_sanity_issue_type is None
    assert analysis.formatting_issues == ["Empty thinking wrapper present"]


class TestClassificationInvariants:
    """Property-style invariant tests guarding classification edge cases.

    These encode structural guarantees that must hold regardless of
    threshold tuning, new heuristics, or upstream library changes.
    """

    # -- Hint relationship invariants --

    @pytest.mark.parametrize(
        "nonvisual_terms",
        [
            ("taken", "2026-02-21", "gps", "timestamp"),
            ("welwyn garden city",),
            ("source", "stock", "adobe stock"),
            ("latitude", "longitude"),
        ],
        ids=["dates_gps", "location", "source_stock", "coordinates"],
    )
    def test_metadata_only_hints_never_ignores(self, nonvisual_terms: tuple[str, ...]) -> None:
        """A bundle where all trusted terms are nonvisual must never trigger ignores."""
        bundle = check_models.TrustedHintBundle(
            trusted_text=" ".join(nonvisual_terms),
            trusted_terms=nonvisual_terms,
            nonvisual_terms=nonvisual_terms,
        )
        text = (
            "Title: Mountain landscape at sunset\n"
            "Description: A mountain range glowing at sunset.\n"
            "Keywords: mountain, sunset, landscape"
        )
        relationship, _ = check_models._classify_hint_relationship(text, bundle)
        assert relationship != "ignores_trusted_hints", (
            f"Metadata-only hints {nonvisual_terms} must not trigger ignores_trusted_hints"
        )

    # -- Cutoff / truncation severity invariants --

    @pytest.mark.parametrize(
        "degradation_reason",
        ["abrupt_tail", "missing_sections", "repetitive_tail", "unfinished_section"],
    )
    def test_degradation_reason_forces_cutoff_degraded(self, degradation_reason: str) -> None:
        """Any degradation reason with a token-cap hit must produce cutoff_degraded."""
        verdict, _ = check_models._classify_review_verdict(
            has_harness_issue=False,
            harness_type=None,
            likely_cutoff=True,
            cutoff_reasons=[degradation_reason],
            prompt_tokens_total=100,
            prompt_tokens_text_est=80,
            prompt_tokens_nontext_est=20,
            missing_sections=["Keywords"] if degradation_reason == "missing_sections" else [],
            utility_grade="C",
            instruction_echo=False,
            metadata_borrowing=False,
            has_hallucination=False,
        )
        assert verdict == "cutoff_degraded", (
            f"Degradation reason '{degradation_reason}' must produce cutoff_degraded, got {verdict}"
        )

    # -- Bucket severity monotonicity --

    @pytest.mark.parametrize(
        "verdict",
        ["harness", "cutoff_degraded", "runtime_failure"],
    )
    @pytest.mark.parametrize("utility_grade", ["A", "B", "C", "D", "F"])
    @pytest.mark.parametrize(
        "hint_relationship",
        [
            "preserves_trusted_hints",
            "improves_trusted_hints",
            "degrades_trusted_hints",
            "ignores_trusted_hints",
        ],
    )
    def test_severe_verdicts_always_avoid(
        self, verdict: str, utility_grade: str, hint_relationship: str
    ) -> None:
        """harness/cutoff_degraded/runtime_failure must always map to avoid."""
        bucket = check_models._classify_user_bucket(
            verdict=verdict,
            hint_relationship=hint_relationship,
            has_contract_issue=False,
            utility_grade=utility_grade,
        )
        assert bucket == "avoid", (
            f"verdict={verdict} + grade={utility_grade} + hints={hint_relationship} "
            f"must map to 'avoid', got '{bucket}'"
        )

    # -- Unknown anomaly always triages --

    @pytest.mark.parametrize("utility_grade", ["A", "B", "C", "D", "F"])
    @pytest.mark.parametrize(
        "hint_relationship",
        [
            "preserves_trusted_hints",
            "improves_trusted_hints",
            "degrades_trusted_hints",
            "ignores_trusted_hints",
        ],
    )
    def test_unknown_anomaly_always_caveat(
        self, utility_grade: str, hint_relationship: str
    ) -> None:
        """unknown_runtime_anomaly must always map to canonical caveat."""
        bucket = check_models._classify_user_bucket(
            verdict="unknown_runtime_anomaly",
            hint_relationship=hint_relationship,
            has_contract_issue=False,
            utility_grade=utility_grade,
        )
        assert bucket == "caveat", (
            f"unknown_runtime_anomaly + grade={utility_grade} + hints={hint_relationship} "
            f"must map to 'caveat', got '{bucket}'"
        )

    # -- Clean verdict implies no breaking evidence --

    def test_clean_verdict_has_no_breaking_evidence(self) -> None:
        """A clean verdict must not carry any QUALITY_BREAKING_LABELS in evidence."""
        verdict, evidence = check_models._classify_review_verdict(
            has_harness_issue=False,
            harness_type=None,
            likely_cutoff=False,
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
        assert verdict == "clean"
        breaking_overlap = set(evidence) & check_models.QUALITY_BREAKING_LABELS
        assert not breaking_overlap, (
            f"Clean verdict must not carry breaking labels, found: {breaking_overlap}"
        )

    # -- History reader forward-compat --

    def test_history_comparison_tolerates_missing_optional_fields(self) -> None:
        """compare_history_records must handle records missing future optional fields."""
        minimal_previous: check_models.HistoryRunRecord = {
            "_type": "run",
            "format_version": "1.0",
            "timestamp": "2026-01-01T00:00:00",
            "prompt_hash": "abc123",
            "prompt_preview": "test",
            "image_path": None,
            "model_results": {
                "org/model-a": {
                    "success": True,
                    "error_stage": None,
                    "error_type": None,
                    "error_package": None,
                },
            },
            "system": {},
            "library_versions": {},
        }
        minimal_current: check_models.HistoryRunRecord = {
            "_type": "run",
            "format_version": "2.0",
            "timestamp": "2026-04-18T00:00:00",
            "prompt_hash": "abc123",
            "prompt_preview": "test",
            "image_path": None,
            "model_results": {
                "org/model-a": {
                    "success": False,
                    "error_stage": None,
                    "error_type": None,
                    "error_package": None,
                },
            },
            "system": {},
            "library_versions": {},
        }
        result = check_models.compare_history_records(minimal_previous, minimal_current)
        assert "org/model-a" in result["regressions"]
