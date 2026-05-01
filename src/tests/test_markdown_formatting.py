"""Tests for Markdown formatting utilities."""

import re
from dataclasses import dataclass
from pathlib import Path

import check_models


def test_escape_markdown_in_text_pipes() -> None:
    """Should escape pipe characters."""
    result = check_models.MARKDOWN_ESCAPER.escape("a|b|c")
    assert result == "a\\|b\\|c"


def test_escape_markdown_in_text_single_pipe() -> None:
    """Should escape single pipe."""
    result = check_models.MARKDOWN_ESCAPER.escape("before|after")
    assert result == "before\\|after"


def test_escape_markdown_in_text_no_pipes() -> None:
    """Should leave text without pipes unchanged."""
    text = "normal text without pipes"
    result = check_models.MARKDOWN_ESCAPER.escape(text)
    assert result == text


def test_escape_markdown_in_text_empty() -> None:
    """Should handle empty string."""
    result = check_models.MARKDOWN_ESCAPER.escape("")
    assert result == ""


def test_escape_markdown_in_text_only_pipes() -> None:
    """Should escape text with only pipes."""
    result = check_models.MARKDOWN_ESCAPER.escape("|||")
    assert result == "\\|\\|\\|"


def test_escape_markdown_in_text_mixed_content() -> None:
    """Should escape pipes in mixed content."""
    result = check_models.MARKDOWN_ESCAPER.escape("model|size|speed")
    assert result == "model\\|size\\|speed"


def test_escape_markdown_in_text_whitespace() -> None:
    """Should preserve whitespace around escaped pipes."""
    result = check_models.MARKDOWN_ESCAPER.escape("a | b | c")
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
    output = check_models._format_failures_by_package_parts(
        results,
        html_output=False,
    )
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
    output = check_models._format_failures_by_package_parts(
        results,
        html_output=False,
    )

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
    output = check_models._format_failures_by_package_parts(
        results,
        html_output=False,
    )
    output_text = "\n".join(output)

    # Should have actionable items section
    assert "### Actionable Items by Package" in output_text
    assert "#### transformers" in output_text
    assert "test-model" in output_text
    assert "ImportError" in output_text


# ── Bare URL wrapping (MD034) tests ────────────────────────────────────────

_BARE_URL_RE = re.compile(r"(?<![<(])https?://")
"""Matches a bare URL not already wrapped in < > or ( )."""


@dataclass
class _GalleryGeneration:
    text: str | None = "output"
    prompt_tokens: int | None = 10
    generation_tokens: int | None = 5
    prompt_tps: float | None = 100.0
    generation_tps: float | None = 50.0


def _gallery_lines_for(result: check_models.PerformanceResult) -> str:
    """Return joined gallery markdown for a single result."""
    return "\n".join(check_models._generate_model_gallery_section([result]))


def test_gallery_metrics_include_time_and_throughput_details() -> None:
    """Gallery success blocks should show timing and prompt/gen throughput."""
    result = check_models.PerformanceResult(
        model_name="test/model",
        generation=_GalleryGeneration(
            prompt_tokens=1624,
            generation_tokens=9,
            prompt_tps=1551.0,
            generation_tps=5.51,
        ),
        success=True,
        model_load_time=3.29,
        generation_time=1.60,
        total_time=5.14,
    )

    md = _gallery_lines_for(result)

    assert "_Timing:_ Load 3.29s; Gen 1.60s; Total 5.14s" in md
    assert "_Throughput:_ Prompt 1,551 TPS (1,624 tok); Gen 5.51 TPS (9 tok)" in md


def test_gallery_metrics_omit_missing_segments_cleanly() -> None:
    """Gallery metrics should omit unavailable prompt throughput without empty separators."""
    result = check_models.PerformanceResult(
        model_name="test/model",
        generation=_GalleryGeneration(
            prompt_tokens=None,
            generation_tokens=80,
            prompt_tps=None,
            generation_tps=29.7,
        ),
        success=True,
        model_load_time=None,
        generation_time=10.90,
        total_time=14.51,
    )

    md = _gallery_lines_for(result)

    assert "_Timing:_ Gen 10.90s; Total 14.51s" in md
    assert "_Throughput:_ Gen 29.7 TPS (80 tok)" in md
    assert "Prompt" not in md


def test_gallery_output_uses_wrapped_blockquote_instead_of_fence() -> None:
    """Gallery output should render model text in a wrapped blockquote."""
    result = check_models.PerformanceResult(
        model_name="test/model",
        generation=_GalleryGeneration(text="alpha\n\nbeta"),
        success=True,
        model_load_time=1.0,
        generation_time=2.0,
        total_time=3.0,
    )

    md = _gallery_lines_for(result)

    assert "```text" not in md
    assert "<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->" in md
    assert "> [!NOTE]" not in md
    assert "> alpha" in md
    assert "\n>\n> beta" in md


def test_gallery_anchor_and_heading_are_separated_by_blank_line() -> None:
    """Gallery anchors should not sit directly above headings."""
    result = check_models.PerformanceResult(
        model_name="test/model",
        generation=_GalleryGeneration(text="alpha"),
        success=True,
        model_load_time=1.0,
        generation_time=2.0,
        total_time=3.0,
    )

    md = _gallery_lines_for(result)

    assert '<a id="model-test-model"></a>\n\n### ✅ test/model' in md


def test_gallery_quality_warnings_have_blank_line_before_list() -> None:
    """Quality warning bullets should be separated from the label for MD032 compliance."""
    analysis = check_models.GenerationQualityAnalysis(
        is_repetitive=False,
        repeated_token=None,
        hallucination_issues=[],
        is_verbose=False,
        formatting_issues=[],
        has_excessive_bullets=False,
        bullet_count=0,
        is_context_ignored=True,
        missing_context_terms=["alpha", "beta"],
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
        has_harness_issue=False,
        harness_issue_type=None,
        harness_issue_details=[],
        word_count=20,
        unique_ratio=0.8,
    )
    result = check_models.PerformanceResult(
        model_name="test/model",
        generation=_GalleryGeneration(text="alpha"),
        success=True,
        model_load_time=1.0,
        generation_time=2.0,
        total_time=3.0,
        quality_analysis=analysis,
    )

    md = _gallery_lines_for(result)

    assert "⚠️ _Quality Warnings:_\n\n- Context ignored" in md
    assert "\n\n\n⚠️ _Quality Warnings:_" not in md


def test_gallery_quality_warnings_preserve_underscores_but_escape_asterisks() -> None:
    """Quality warning bullets should keep identifiers readable while escaping emphasis."""
    analysis = check_models.GenerationQualityAnalysis(
        is_repetitive=False,
        repeated_token=None,
        hallucination_issues=[],
        is_verbose=False,
        formatting_issues=["Unknown tags: <fake_token_around_image>, small */ alpha*/ beta */"],
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
        has_harness_issue=False,
        harness_issue_type=None,
        harness_issue_details=[],
        word_count=20,
        unique_ratio=0.8,
    )
    result = check_models.PerformanceResult(
        model_name="test/model",
        generation=_GalleryGeneration(text="alpha"),
        success=True,
        model_load_time=1.0,
        generation_time=2.0,
        total_time=3.0,
        quality_analysis=analysis,
    )

    md = _gallery_lines_for(result)

    assert "- Unknown tags: <fake_token_around_image>, small \\*/ alpha\\*/ beta \\*/" in md


def test_gallery_quality_warnings_escape_html_like_thinking_tags() -> None:
    """Markdown gallery warnings should escape real HTML-like tags such as <think>."""
    analysis = check_models.GenerationQualityAnalysis(
        is_repetitive=False,
        repeated_token=None,
        hallucination_issues=[],
        is_verbose=False,
        formatting_issues=["Unknown tags: <think>"],
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
        has_harness_issue=False,
        harness_issue_type=None,
        harness_issue_details=[],
        word_count=20,
        unique_ratio=0.8,
    )
    result = check_models.PerformanceResult(
        model_name="test/model",
        generation=_GalleryGeneration(text="alpha"),
        success=True,
        model_load_time=1.0,
        generation_time=2.0,
        total_time=3.0,
        quality_analysis=analysis,
    )

    md = _gallery_lines_for(result)

    assert "- Unknown tags: &lt;think&gt;" in md
    assert "- Unknown tags: <think>" not in md


def test_gallery_error_block_does_not_emit_extra_blank_lines_before_separator() -> None:
    """Error entries should not produce MD012-triggering triple blank lines."""
    result = check_models.PerformanceResult(
        model_name="test/model",
        generation=None,
        success=False,
        error_stage="Model Error",
        error_message="generation failed",
        error_traceback="Traceback (most recent call last):\nValueError: boom",
    )

    md = _gallery_lines_for(result)

    assert "<summary>Full Traceback (click to expand)</summary>\n\n\n```python" not in md
    assert "</details>\n\n\n---" not in md


def test_multiline_metadata_renders_as_single_list_item() -> None:
    """Multiline metadata values should stay within the same list item."""
    parts: list[str] = []

    check_models._append_markdown_image_metadata_section(
        parts,
        {
            "description": "First line\nSecond line\n\nThird paragraph.",
        },
    )

    md = "\n".join(parts)
    assert "- _Description:_ First line" in md
    assert "\n\n    Second line" in md
    assert "\n\n    Third paragraph." in md


def test_wrapped_blockquote_neutralizes_leading_markdown_syntax() -> None:
    """Wrapped blockquote lines should keep leading Markdown control syntax readable."""
    parts: list[str] = []

    check_models._append_markdown_wrapped_blockquote(
        parts,
        "# heading\n2. numbered\n- bullet\n[!NOTE] alert",
    )

    md = "\n".join(parts)
    assert "> &#35; heading" in md
    assert "> 2&#46; numbered" in md
    assert "> &#45; bullet" in md
    assert "> &#91;!NOTE] alert" in md


def test_wrapped_blockquote_neutralizes_inline_asterisk_emphasis() -> None:
    """Wrapped blockquote lines should render raw asterisk emphasis literally."""
    parts: list[str] = []

    check_models._append_markdown_wrapped_blockquote(parts, "*italic*\n**bold**")

    md = "\n".join(parts)
    assert "> &#42;italic&#42;" in md
    assert "> &#42;&#42;bold&#42;&#42;" in md
    assert "> *italic*" not in md
    assert "> **bold**" not in md


def test_wrapped_blockquote_neutralizes_setext_heading_underline() -> None:
    """Wrapped blockquote lines should not emit setext headings from raw model output."""
    parts: list[str] = []

    check_models._append_markdown_wrapped_blockquote(parts, "Title\n------\nKeywords\n======")

    md = "\n".join(parts)
    assert "> Title" in md
    assert "> &#45;-----" in md
    assert "> Keywords" in md
    assert "> &#61;=====" in md


def test_wrapped_blockquote_neutralizes_label_only_lines_and_lone_markers() -> None:
    """Wrapped blockquotes should keep label lines and stray markers out of heading parsing."""
    parts: list[str] = []

    check_models._append_markdown_wrapped_blockquote(
        parts,
        "Description:\n- A large white butterfly\n\n**Title:**\n- Concrete and factual.\n-",
    )

    md = "\n".join(parts)
    assert "> &#8203;Description:" in md
    assert "> &#45; A large white butterfly" in md
    assert "> &#8203;&#42;&#42;Title:&#42;&#42;" in md
    assert "> &#45; Concrete and factual." in md
    assert "> &#45;" in md
    assert "> Description:" not in md
    assert "> &#42;&#42;Title:&#42;&#42;" not in md
    assert "> -" not in md


def test_gallery_review_summary_uses_review_focus_evidence(tmp_path: Path) -> None:
    """Gallery compact review line should use the canonical evidence-focused summary."""
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
    result = check_models.PerformanceResult(
        model_name="test/model",
        generation=_GalleryGeneration(
            text=text,
            prompt_tokens=320,
            generation_tokens=60,
        ),
        success=True,
        model_load_time=1.0,
        generation_time=2.0,
        total_time=3.0,
        quality_analysis=analysis,
        requested_max_tokens=60,
    )

    out = tmp_path / "gallery.md"
    check_models.generate_markdown_gallery_report(
        results=[result],
        filename=out,
        prompt=prompt,
    )
    md = out.read_text(encoding="utf-8")

    assert "_Review summary:_" in md
    assert "hit token cap (60)" in md
    assert "keywords=1" in md


def test_wrapped_blockquote_strips_trailing_nonbreaking_spaces() -> None:
    """Wrapped blockquote lines should not preserve trailing NBSP or single spaces."""
    parts: list[str] = []

    check_models._append_markdown_wrapped_blockquote(parts, "alpha\u00a0\nbeta ")

    md = "\n".join(parts)
    assert "> alpha\u00a0" not in md
    assert "> beta " not in md
    assert "> alpha" in md
    assert "> beta" in md


def test_normalize_markdown_trailing_spaces_strips_nonbreaking_spaces() -> None:
    """Markdown trailing-space normalization should strip NBSP endings."""
    md = "alpha\u00a0\nbeta  \ngamma "

    normalized = check_models.normalize_markdown_trailing_spaces(md)

    assert normalized.splitlines() == ["alpha", "beta  ", "gamma"]


def test_normalize_markdown_trailing_spaces_strips_trailing_bom_and_zero_width() -> None:
    """Markdown trailing-space normalization should strip BOM and zero-width endings."""
    md = "alpha\ufeff\nbeta\u200b\ngamma\u2060\ndelta  "

    normalized = check_models.normalize_markdown_trailing_spaces(md)

    assert normalized.splitlines() == ["alpha", "beta", "gamma", "delta  "]


def test_wrapped_blockquote_preserves_plain_bracket_text() -> None:
    """Wrapped blockquote lines should leave ordinary bracket text readable."""
    parts: list[str] = []

    check_models._append_markdown_wrapped_blockquote(parts, "decade_data[decade]['count'] += 1")

    md = "\n".join(parts)
    assert "> decade_data[decade]['count'] += 1" in md


def test_wrapped_blockquote_disables_reversed_link_lint_for_code_like_text() -> None:
    """Wrapped blockquotes should suppress MD011 for model text that looks like code."""
    parts: list[str] = []

    check_models._append_markdown_wrapped_blockquote(
        parts,
        "df.groupby('MarketingStrategy')['EngagementLevel'].mean()",
    )

    md = "\n".join(parts)
    assert "<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->" in md
    assert "> df.groupby('MarketingStrategy')['EngagementLevel'].mean()" in md


def test_wrapped_blockquote_normalizes_wrapped_leading_spaces() -> None:
    """Wrapped continuation lines should not emit multiple spaces after '>'."""
    parts: list[str] = []

    check_models._append_markdown_wrapped_blockquote(parts, "alpha beta gamma", width=6)

    md = "\n".join(parts)
    assert ">  beta" not in md
    assert ">  gamma" not in md
    assert "> beta" in md
    assert "> gamma" in md


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


def test_error_text_escapes_underscore_emphasis_markers() -> None:
    """Error prose should escape underscores to avoid unintended strong emphasis."""
    result = check_models.PerformanceResult(
        model_name="test/model",
        generation=None,
        success=False,
        error_stage="API Mismatch",
        error_message="LanguageModel.__call__() got an unexpected keyword argument",
    )
    md = _gallery_lines_for(result)
    assert "LanguageModel.\\_\\_call\\_\\_() got an unexpected keyword argument" in md
