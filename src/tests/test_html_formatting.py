"""Tests for HTML formatting utilities."""

from pathlib import Path
from types import SimpleNamespace

import check_models


def test_model_link_keeps_hostile_names_in_escaped_text_and_a_fragment() -> None:
    """The SKY-D228 workaround must not conceal an actual HTML injection."""
    assert check_models._html_model_link('"><script>bad</script>') == (
        '<a href="#model-scriptbad-script"><code>'
        "&quot;&gt;&lt;script&gt;bad&lt;/script&gt;</code></a>"
    )


def test_escape_html_tags_selective_basic() -> None:
    """Should escape < and > in text."""
    result = check_models.escape_html_selectively("Use <model> here")
    assert result == "Use &lt;model&gt; here"


def test_escape_html_tags_selective_multiple() -> None:
    """Should escape non-allowed tags but keep allowed tags like <b>."""
    text = "<a> and <b> tags"
    result = check_models.escape_html_selectively(text)
    # <a> is not allowed, so it's escaped; <b> is allowed, so it's kept
    assert result == "&lt;a&gt; and <b> tags"


def test_escape_html_tags_selective_rejects_attributes_on_allowed_tags() -> None:
    """Allowed formatting tags should not preserve unsafe attributes."""
    result = check_models.escape_html_selectively('<b onclick="alert(1)">bold</b><br data-x="1">')
    assert result == "&lt;b onclick=&quot;alert(1)&quot;&gt;bold</b>&lt;br data-x=&quot;1&quot;&gt;"


def test_escape_html_tags_selective_preserves_entities() -> None:
    """Should preserve existing HTML entities."""
    result = check_models.escape_html_selectively("5 &lt; 10")
    assert result == "5 &lt; 10"


def test_escape_html_tags_selective_mixed() -> None:
    """Should handle mixed content with entities."""
    # Function only escapes tag-like patterns, not bare angle brackets
    result = check_models.escape_html_selectively("5 < 10 &amp; 10 > 5")
    # Bare angle brackets and entities are preserved as-is
    assert result == "5 < 10 &amp; 10 > 5"


def test_escape_html_tags_selective_ampersands() -> None:
    """Should handle standalone ampersands."""
    # Function only escapes tags, not ampersands
    result = check_models.escape_html_selectively("Tom & Jerry")
    # Ampersand is preserved as-is (not escaped by this function)
    assert result == "Tom & Jerry"


def test_escape_html_tags_selective_no_tags() -> None:
    """Should leave text without tags unchanged."""
    text = "Plain text without any special characters"
    result = check_models.escape_html_selectively(text)
    assert result == text


def test_escape_html_tags_selective_empty() -> None:
    """Should handle empty string."""
    assert check_models.escape_html_selectively("") == ""


def test_escape_html_tags_selective_entity_chain() -> None:
    """Should preserve complex entity chains."""
    result = check_models.escape_html_selectively("&lt;&gt; &amp; &quot;")
    assert "&lt;" in result
    assert "&gt;" in result
    assert "&amp;" in result
    assert "&quot;" in result


def test_html_escaper_escapes_a_tag_whose_quoted_attribute_holds_angle_brackets() -> None:
    """A quoted "<" is legal inside a CommonMark raw-HTML tag, so it must not hide the tag."""
    for tag in ('<img alt="<" src=x onerror=alert(1)>', '<img alt="<b>" onerror=x>'):
        result = check_models.escape_html_selectively(tag)
        assert result.startswith("&lt;img")
        assert "<b>" not in result


def test_table_escaping_does_not_double_escape_quotes() -> None:
    """html.escape(quote=True) yields &quot;, which must survive the ampersand pass."""
    result = check_models.escape_markdown_cell('<a href="x">')
    assert result == "&lt;a href=&quot;x&quot;&gt;"


def test_bare_url_is_wrapped_whole_and_never_swallows_markup() -> None:
    """A trailing ] must not cut the URL short, and "<" never joins it."""
    assert check_models._wrap_bare_urls("[see https://example.com]") == (
        "[see <https://example.com>]"
    )
    wrapped = check_models._wrap_bare_urls("https://x<script ")
    assert wrapped.startswith("<https://x>")
    assert check_models._escape_report_markdown_text("https://x<script ").endswith("&lt;script ")


def test_blockquote_neutralises_tilde_fences_and_short_setext_underlines() -> None:
    """~~~ opens a code block and =/== or -- underline the previous line."""
    for line in ("~~~", "=", "==", "--"):
        escaped = check_models._escape_markdown_blockquote_line(line)
        assert escaped != line
        assert escaped.startswith("&#")


def test_image_metadata_section_escapes_untrusted_caption_lines() -> None:
    """IPTC/XMP text is untrusted: raw HTML or headings must not reach the gallery."""
    parts: list[str] = []
    check_models._append_markdown_image_metadata_section(
        parts,
        {"description": "<img src=https://t/x.gif>\n# Heading\n<details open>"},
    )
    rendered = "\n".join(parts)
    assert "<img" not in rendered
    assert "<details" not in rendered
    assert "\n    # Heading" not in rendered


def test_field_aware_preview_bounds_a_single_run_on_keyword() -> None:
    """One comma-free keyword blob must respect the keyword budget."""
    answer = "Title: Boats\nDescription: Two boats.\nKeywords: " + "red car " * 250
    preview = check_models._field_aware_preview(answer, max_chars=280)
    assert preview is not None
    assert len(preview) < 400


def test_html_report_workbench_filters_compare_and_note_copy(tmp_path: Path) -> None:
    """HTML chooser exposes hardware/reasoning filters, comparison panel, and note copy button."""
    prompt = (
        "Create British-English catalogue metadata from the image and supplied context.\n\n"
        "Context: Descriptive hints:\n"
        "- Description hint: A small dog sits indoors looking at the camera.\n"
        "- Keyword hints: dog, indoors, alert, harness\n\n"
        "Return exactly these three sections and nothing else:\n"
        "Title:\nDescription:\nKeywords:"
    )
    answer = (
        "<think>Checking the image details.</think>\n"
        "Title: Small dog sitting indoors\n"
        "Description: A small dog sits indoors looking at the camera in soft light.\n"
        "Keywords: dog, indoors, living room, portrait"
    )
    result = check_models.PerformanceResult(
        model_name="org/thinking-model",
        generation=SimpleNamespace(
            text=answer,
            prompt_tokens=200,
            generation_tokens=60,
            total_tokens=260,
            prompt_tps=400.0,
            generation_tps=50.0,
            peak_memory=6.5,
        ),
        model_load_time=0.5,
        generation_time=1.2,
        total_time=1.7,
        success=True,
        assessment_profile="metadata",
    )
    out = tmp_path / "workbench.html"
    check_models.generate_html_report(
        results=[result],
        filename=out,
        versions={"mlx": "0.32.3", "mlx-vlm": "0.8.0"},
        prompt=prompt,
        total_runtime_seconds=1.7,
    )
    content = out.read_text(encoding="utf-8")
    assert 'id="peak-memory-filter"' in content
    assert 'id="total-time-filter"' in content
    assert 'id="reasoning-filter"' in content
    assert 'id="compare-panel"' in content
    assert 'class="compare-toggle"' in content
    assert ">actionable failure</option>" in content
    assert ">observation needs reproduction</option>" in content
    assert "4 (2 new / 2 hint)" in content
    assert "<th>" not in content or "Hint text" in content
    assert 'data-has-reasoning="yes"' in content
    assert 'data-peak-gb="6.5000"' in content
    assert "Copy add_accuracy_note command" in content
    assert "tools.add_accuracy_note" in content
    assert "<h4>Final answer</h4>" in content
    assert "<p><b>Runtime</b></p>" not in content
