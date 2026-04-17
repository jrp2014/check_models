"""Tests for cataloging utility metrics."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from check_models import (
    GRADE_EMOJIS,
    QUALITY,
    ModelIssueSummary,
    PerformanceResult,
    _collect_cataloging_summary_data,
    _format_cataloging_summary,
    _get_grade_display,
    analyze_model_issues,
    compute_cataloging_utility,
    compute_description_quality,
    compute_information_gain,
    compute_keyword_quality,
    compute_task_compliance,
    compute_visual_grounding,
)

# Test thresholds derived from QualityThresholds
GROUNDING_MIDPOINT = 0.5  # Threshold for "high" vs "low" grounding in tests
MIN_COLOR_TERMS_IN_LIST = 4  # Expected colors in "red, blue, green, yellow..."
MIN_SPATIAL_TERMS_IN_SENTENCE = 3  # "above", "below", "behind"


def _get_utility_score(result: dict[str, float | str]) -> float:
    """Extract utility score as float from result dict."""
    return float(result["utility_score"])


@dataclass
class _StubGeneration:
    text: str | None
    prompt_tokens: int | None = 64
    generation_tokens: int | None = 128
    generation_tps: float = 20.0
    peak_memory: float = 1.2
    time: float | None = None
    active_memory: float | None = None
    cache_memory: float | None = None


class TestComputeInformationGain:
    """Tests for compute_information_gain function."""

    def test_no_context_all_novel(self) -> None:
        """With no context, all words are novel."""
        text = "A beautiful sunset over the mountains with orange and purple clouds."
        result = compute_information_gain(text, None)

        assert result["information_gain"] == 1.0
        assert result["echo_ratio"] == 0.0
        assert result["novel_words"] == result["output_words"]

    def test_full_echo(self) -> None:
        """Text that fully echoes context should have low information gain."""
        context = "sunset mountains clouds"
        text = "sunset mountains clouds"
        result = compute_information_gain(text, context)

        assert result["information_gain"] == 0.0
        assert result["echo_ratio"] == 1.0
        assert result["novel_words"] == 0

    def test_partial_echo(self) -> None:
        """Text with some echoed and some novel words."""
        context = "mountain landscape"
        text = "A beautiful mountain landscape with a river and trees."
        result = compute_information_gain(text, context)

        # Should have some novel words
        assert 0.0 < result["information_gain"] < 1.0
        assert 0.0 < result["echo_ratio"] < 1.0
        assert result["novel_words"] > 0

    def test_empty_text(self) -> None:
        """Empty text returns zero metrics."""
        result = compute_information_gain("", "some context")

        assert result["information_gain"] == 0.0
        assert result["output_words"] == 0

    def test_penalized_echo_ratio_excludes_visual_terms(self) -> None:
        """Visual hint terms should not inflate penalized_echo_ratio."""
        context = "church sunset timestamp england"
        text = "A beautiful church at sunset with golden light."
        result = compute_information_gain(text, context)

        # echo_ratio counts all overlap (church, sunset from context)
        assert result["echo_ratio"] > 0.0
        # penalized_echo_ratio only counts nonvisual terms (timestamp, england)
        # which are NOT in the output, so it should be lower
        assert result["penalized_echo_ratio"] <= result["echo_ratio"]

    def test_penalized_echo_ratio_returned(self) -> None:
        """penalized_echo_ratio key is present in results."""
        result = compute_information_gain("some novel text here", "other context words")
        assert "penalized_echo_ratio" in result


class TestComputeTaskCompliance:
    """Tests for compute_task_compliance function."""

    def test_full_compliance(self) -> None:
        """Text with all three components matching strict length constraints."""
        text = """
        Title: A glorious golden sunset over distant rugged mountain peaks.

        Description: This image shows a beautiful sunset with orange and purple hues. The bright sky casts a warm reflection over the entire vast landscape.

        Keywords: sunset, mountains, landscape, nature, sky, orange, purple, clouds, evening, dusk, scenic, panoramic
        """
        result = compute_task_compliance(text)

        assert result["has_caption"] is True
        assert result["has_description"] is True
        assert result["has_keywords"] is True
        assert result["compliance_score"] == 1.0

    def test_no_structure(self) -> None:
        """Plain text without labeled sections."""
        text = "A nice photo of some trees."
        result = compute_task_compliance(text)

        # May detect implicit structure, but explicit markers missing
        assert result["compliance_score"] < 1.0

    def test_partial_compliance(self) -> None:
        """Text with only some components."""
        text = """
        Caption: A sunset scene.

        The colors are vivid and warm.
        """
        result = compute_task_compliance(text)

        assert result["has_caption"] is True
        # May or may not detect description without explicit marker
        assert result["compliance_score"] < 1.0

    def test_empty_text(self) -> None:
        """Empty text returns zero compliance."""
        result = compute_task_compliance("")

        assert result["compliance_score"] == 0.0


class TestComputeVisualGrounding:
    """Tests for compute_visual_grounding function."""

    def test_high_grounding(self) -> None:
        """Text with many visual/spatial/color terms."""
        text = """
        In the foreground, a red barn stands against green hills.
        The sky above is bright blue with white fluffy clouds.
        On the left side, tall oak trees frame the scene.
        """
        result = compute_visual_grounding(text, None)

        assert result["visual_terms"] > 0
        assert result["spatial_terms"] > 0
        assert result["color_terms"] > 0
        assert result["grounding_score"] > GROUNDING_MIDPOINT

    def test_low_grounding(self) -> None:
        """Abstract text with few visual references."""
        text = "This represents the concept of freedom and hope."
        result = compute_visual_grounding(text, None)

        assert result["grounding_score"] < GROUNDING_MIDPOINT

    def test_color_terms(self) -> None:
        """Text with specific color references."""
        text = "Red, blue, green, yellow, and purple flowers."
        result = compute_visual_grounding(text, None)

        assert result["color_terms"] >= MIN_COLOR_TERMS_IN_LIST

    def test_spatial_terms(self) -> None:
        """Text with spatial positioning."""
        text = "Above the horizon, below the clouds, behind the trees."
        result = compute_visual_grounding(text, None)

        assert result["spatial_terms"] >= MIN_SPATIAL_TERMS_IN_SENTENCE

    def test_empty_text(self) -> None:
        """Empty text returns zero grounding."""
        result = compute_visual_grounding("", None)

        assert result["grounding_score"] == 0.0


class TestComputeCatalogingUtility:
    """Tests for compute_cataloging_utility function."""

    def test_high_quality_output(self) -> None:
        """High-quality cataloging output gets high score."""
        text = """
        Title: A golden sunset over mountain peaks with dramatic clouds.

        Description: This landscape photograph captures the last moments of daylight as the sun descends behind snow-capped mountains. The sky is ablaze with orange, pink, and purple hues, reflecting off the scattered clouds.

        Keywords: sunset, mountains, landscape, alpine lake, reflection, golden hour, dramatic sky, nature photography, clouds, peaks, dusk, scenic
        """
        result = compute_cataloging_utility(text, None)
        score = _get_utility_score(result)

        assert score >= QUALITY.grade_b_threshold
        assert result["utility_grade"] in ("A", "B")

    def test_low_quality_output(self) -> None:
        """Low-quality output gets low score."""
        text = "image"
        result = compute_cataloging_utility(text, None)
        score = _get_utility_score(result)

        assert score < QUALITY.grade_d_threshold
        assert result["utility_grade"] in ("D", "F")

    def test_echoed_context(self) -> None:
        """Output that mostly echoes context is penalized."""
        context = "sunset mountains landscape nature"
        text = "sunset mountains landscape nature"
        result = compute_cataloging_utility(text, context)
        score = _get_utility_score(result)

        # Should be penalized for low information gain
        assert score < QUALITY.grade_b_threshold

    def test_grade_thresholds(self) -> None:
        """Verify grade assignment follows score thresholds."""
        # The actual thresholds are in QualityThresholds
        high_quality = """
        Title: Detailed image title with multiple concrete visual elements.
        Description: Comprehensive description of the scene including foreground subjects and background. The text illustrates lighting conditions and atmospheric details present throughout.
        Keywords: photography, composition, lighting, subject, background, foreground, details, atmosphere, conditions, elements, visual, landscape
        """
        result = compute_cataloging_utility(high_quality, None)

        # Grade should be determined by score
        score = _get_utility_score(result)
        grade = result["utility_grade"]

        if score >= QUALITY.grade_a_threshold:
            assert grade == "A"
        elif score >= QUALITY.grade_b_threshold:
            assert grade in ("A", "B")
        elif score >= QUALITY.grade_c_threshold:
            assert grade in ("B", "C")
        elif score >= QUALITY.grade_d_threshold:
            assert grade in ("C", "D")
        else:
            assert grade in ("D", "F")

    def test_primary_weakness_identified(self) -> None:
        """Verify primary weakness is identified."""
        text = "A photo."  # Too short, low compliance, low grounding
        result = compute_cataloging_utility(text, None)

        assert "primary_weakness" in result
        assert result["primary_weakness"]  # Should not be empty

    def test_precomputed_metrics(self) -> None:
        """Can pass pre-computed metrics to avoid redundant computation."""
        text = "Caption: A sunset. Keywords: sunset, sky"

        info_gain = compute_information_gain(text, None)
        compliance = compute_task_compliance(text)
        grounding = compute_visual_grounding(text, None)

        result = compute_cataloging_utility(
            text,
            None,
            info_gain=info_gain,
            task_compliance=compliance,
            visual_grounding=grounding,
        )

        assert "utility_score" in result
        assert "utility_grade" in result


class TestTaskSpecificScorecards:
    """Tests for description and keyword specific scoring helpers."""

    def test_description_quality_rewards_concrete_visual_detail(self) -> None:
        """Specific, image-grounded prose should beat vague filler description."""
        strong = (
            "Description: A red tram moves along wet tracks between brick buildings while "
            "soft morning fog hangs above the street."
        )
        weak = "Description: A nice image of a scene."

        strong_score = float(compute_description_quality(strong, None)["description_score"])
        weak_score = float(compute_description_quality(weak, None)["description_score"])

        assert strong_score > weak_score
        assert strong_score >= QUALITY.grade_b_threshold
        assert weak_score < QUALITY.grade_c_threshold

    def test_keyword_quality_penalizes_generic_duplicates(self) -> None:
        """Diverse keyword sets should outrank generic repeated tags."""
        strong = (
            "Keywords: red tram, wet pavement, brick buildings, city street, morning fog, "
            "urban transport, reflections, overhead wires, leading lines, travel"
        )
        weak = "Keywords: image, photo, scene, city, city, city, transport, transport, nice, visual"

        strong_metrics = compute_keyword_quality(strong, None)
        weak_metrics = compute_keyword_quality(weak, None)

        assert float(strong_metrics["keyword_score"]) > float(weak_metrics["keyword_score"])
        assert int(strong_metrics["keyword_unique_terms"]) > int(
            weak_metrics["keyword_unique_terms"]
        )
        assert float(strong_metrics["keyword_duplication_ratio"]) < float(
            weak_metrics["keyword_duplication_ratio"],
        )


class TestHelperFunctions:
    """Tests for cataloging utility helper functions."""

    def test_grade_ordering(self) -> None:
        """Higher scores should get better grades."""
        texts = [
            "x",  # Minimal
            "Caption: A photo of something.",  # Basic
            """Caption: A sunset over mountains.
            Description: Beautiful landscape with orange sky.
            Keywords: sunset, mountains, nature""",  # Good
        ]

        scores = [_get_utility_score(compute_cataloging_utility(t, None)) for t in texts]

        # Scores should generally increase with quality
        assert scores[0] < scores[2]

    @pytest.mark.parametrize(
        ("text", "expected_min_score"),
        [
            ("", 0),
            ("x", 0),
            ("Caption: A sunset photo.", 0),  # Minimal structure, low score
            (
                """Caption: Sunset over mountains.
                Description: A colorful sky at dusk.
                Keywords: sunset, mountains, sky, nature""",
                30,  # Full structure should score higher
            ),
        ],
    )
    def test_minimum_scores(self, text: str, expected_min_score: int) -> None:
        """Various text qualities should meet minimum score thresholds."""
        result = compute_cataloging_utility(text, None)
        score = _get_utility_score(result)
        assert score >= expected_min_score


class TestGradeEmojisConstant:
    """Tests for GRADE_EMOJIS constant usage."""

    def test_grade_emojis_has_all_grades(self) -> None:
        """GRADE_EMOJIS should have entries for all grades A-F."""
        for grade in ["A", "B", "C", "D", "F"]:
            assert grade in GRADE_EMOJIS
            assert isinstance(GRADE_EMOJIS[grade], str)
            assert len(GRADE_EMOJIS[grade]) > 0

    def test_grade_display_uses_constant(self) -> None:
        """_get_grade_display should return emoji-decorated grades."""
        # Check that it returns the correct format
        display_a = _get_grade_display("A")
        assert "A" in display_a
        assert "🏆" in display_a  # A should have trophy emoji

        display_f = _get_grade_display("F")
        assert "F" in display_f
        assert "❌" in display_f  # F should have X emoji


class TestCatalogingSummaryFormatters:
    """Tests for cataloging summary formatting functions."""

    def test_format_cataloging_summary_html_empty(self) -> None:
        """Empty summary should return empty list."""
        summary: ModelIssueSummary = {"cataloging_best": None}
        data = _collect_cataloging_summary_data(summary)
        result = [] if data is None else _format_cataloging_summary(data, html_output=True)
        assert result == []

    def test_format_cataloging_summary_html_with_data(self) -> None:
        """Summary with data should generate HTML."""
        summary: ModelIssueSummary = {
            "cataloging_best": ("model-a", 92.0, "A"),
            "cataloging_best_description": ("model-c", 88.0),
            "cataloging_best_keywords": ("model-d", 91.0),
            "cataloging_worst": ("model-b", 35.0, "D"),
            "cataloging_avg_score": 65.0,
            "cataloging_grades": {"A": ["model-a"], "D": ["model-b"]},
            "low_utility_models": [("model-b", 35.0, "D", "Low visual grounding")],
        }
        data = _collect_cataloging_summary_data(summary)
        assert data is not None
        result = _format_cataloging_summary(data, html_output=True)

        # Should have content
        assert len(result) > 0
        html = "".join(result)

        # Check key elements are present
        assert "Cataloging Utility Summary" in html
        assert "model-a" in html
        assert "model-b" in html
        assert "model-c" in html
        assert "model-d" in html
        assert "92" in html  # Best score
        assert "35" in html  # Worst score
        assert "Low visual grounding" in html
        assert "Best descriptions" in html
        assert "Best keywording" in html

    def test_format_cataloging_summary_html_with_metadata_delta(self) -> None:
        """Summary should include baseline and delta metadata comparison lines."""
        summary: ModelIssueSummary = {
            "cataloging_best": ("model-a", 92.0, "A"),
            "cataloging_worst": ("model-b", 35.0, "D"),
            "cataloging_avg_score": 65.0,
            "cataloging_grades": {"A": ["model-a"], "D": ["model-b"]},
            "metadata_baseline_score": 70.0,
            "metadata_baseline_grade": "B",
            "cataloging_avg_delta": -5.0,
            "cataloging_improves_metadata": ["model-a"],
            "cataloging_neutral_vs_metadata": [],
            "cataloging_worse_than_metadata": ["model-b"],
            "low_utility_models": [("model-b", 35.0, "D", "Low visual grounding")],
        }
        data = _collect_cataloging_summary_data(summary)
        assert data is not None
        result = _format_cataloging_summary(data, html_output=True)
        html = "".join(result)
        assert "Existing Metadata Baseline" in html
        assert "Vs Existing Metadata" in html
        assert "Better: 1" in html
        assert "Worse: 1" in html

    def test_format_cataloging_summary_text_empty(self) -> None:
        """Empty summary should return empty list."""
        summary: ModelIssueSummary = {"cataloging_best": None}
        data = _collect_cataloging_summary_data(summary)
        result = [] if data is None else _format_cataloging_summary(data, html_output=False)
        assert result == []

    def test_format_cataloging_summary_text_with_data(self) -> None:
        """Summary with data should generate Markdown."""
        summary: ModelIssueSummary = {
            "cataloging_best": ("model-a", 92.0, "A"),
            "cataloging_best_description": ("model-c", 88.0),
            "cataloging_best_keywords": ("model-d", 91.0),
            "cataloging_worst": ("model-b", 35.0, "D"),
            "cataloging_avg_score": 65.0,
            "cataloging_grades": {"A": ["model-a"], "D": ["model-b"]},
            "low_utility_models": [("model-b", 35.0, "D", "Low visual grounding")],
        }
        data = _collect_cataloging_summary_data(summary)
        assert data is not None
        result = _format_cataloging_summary(data, html_output=False)

        # Should have content
        assert len(result) > 0
        md = "\n".join(result)

        # Check key elements are present
        assert "Cataloging Utility Summary" in md
        assert "model-a" in md
        assert "model-b" in md
        assert "model-c" in md
        assert "model-d" in md
        assert "Best for cataloging" in md
        assert "Best descriptions" in md
        assert "Best keywording" in md
        assert "Worst for cataloging" in md
        assert "Low visual grounding" in md

    def test_format_cataloging_summary_text_with_metadata_delta(self) -> None:
        """Markdown summary should include metadata baseline and delta breakdown."""
        summary: ModelIssueSummary = {
            "cataloging_best": ("model-a", 92.0, "A"),
            "cataloging_worst": ("model-b", 35.0, "D"),
            "cataloging_avg_score": 65.0,
            "cataloging_grades": {"A": ["model-a"], "D": ["model-b"]},
            "metadata_baseline_score": 70.0,
            "metadata_baseline_grade": "B",
            "cataloging_avg_delta": -5.0,
            "cataloging_improves_metadata": ["model-a"],
            "cataloging_neutral_vs_metadata": [],
            "cataloging_worse_than_metadata": ["model-b"],
            "low_utility_models": [("model-b", 35.0, "D", "Low visual grounding")],
        }
        data = _collect_cataloging_summary_data(summary)
        assert data is not None
        result = _format_cataloging_summary(data, html_output=False)
        md = "\n".join(result)
        assert "Existing Metadata Baseline" in md
        assert "Vs Existing Metadata" in md
        assert "Better: 1" in md
        assert "Worse: 1" in md


class TestAnalyzeModelIssuesCataloging:
    """Tests for cataloging metrics in analyze_model_issues."""

    def test_analyze_model_issues_includes_cataloging(self) -> None:
        """analyze_model_issues should include cataloging summary fields."""
        # Empty results
        summary = analyze_model_issues([])

        # Check cataloging fields exist
        assert "cataloging_grades" in summary
        assert "cataloging_best" in summary
        assert "cataloging_worst" in summary
        assert "cataloging_avg_score" in summary
        assert "low_utility_models" in summary

    def test_analyze_model_issues_tracks_metadata_delta_buckets(self) -> None:
        """Models should be bucketed by whether they improve or worsen existing metadata."""
        context = (
            "Title hint: Sunset over mountain lake\n"
            "Description hint: A colorful sunset behind mountains with lake reflection and trees.\n"
            "Keyword hints: sunset, mountains, lake, reflection, trees, nature, landscape\n"
            "Capture metadata: Taken on 2025-09-10."
        )
        good_text = (
            "Title: Golden alpine sunset over mirrored mountain lake\n"
            "Description: Warm orange light outlines the ridgeline while evergreen silhouettes "
            "frame a still alpine lake that mirrors streaked clouds.\n"
            "Keywords: alpine sunset, mirrored lake, mountain ridgeline, evergreen silhouettes, "
            "orange sky, reflection, wilderness, dramatic light, scenic vista"
        )
        bad_text = "Title: Sunset. Description: sunset image. Keywords: sunset, mountain"

        results = [
            PerformanceResult(
                model_name="model-good",
                generation=_StubGeneration(text=good_text, generation_tokens=220),
                success=True,
                model_load_time=0.5,
                generation_time=1.0,
                total_time=1.5,
            ),
            PerformanceResult(
                model_name="model-bad",
                generation=_StubGeneration(text=bad_text, generation_tokens=12),
                success=True,
                model_load_time=0.6,
                generation_time=1.1,
                total_time=1.7,
            ),
        ]

        summary = analyze_model_issues(results, context=context)
        assert "metadata_baseline_score" in summary
        assert "metadata_baseline_grade" in summary
        assert "cataloging_avg_delta" in summary
        assert summary.get("cataloging_improves_metadata", []) == ["model-good"]
        assert summary.get("cataloging_worse_than_metadata", []) == ["model-bad"]
