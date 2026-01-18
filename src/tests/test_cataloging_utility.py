"""Tests for cataloging utility metrics."""

from __future__ import annotations

import pytest

from check_models import (
    QUALITY,
    compute_cataloging_utility,
    compute_information_gain,
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


class TestComputeTaskCompliance:
    """Tests for compute_task_compliance function."""

    def test_full_compliance(self) -> None:
        """Text with all three components: caption, description, keywords."""
        text = """
        Caption: A sunset over mountains.

        Description: This image shows a beautiful sunset
        with orange and purple hues across the sky.

        Keywords: sunset, mountains, landscape, nature
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
        Caption: A golden sunset over mountain peaks with dramatic clouds.

        Description: This landscape photograph captures the last moments of
        daylight as the sun descends behind snow-capped mountains. The sky
        is ablaze with orange, pink, and purple hues, reflecting off the
        scattered clouds. In the foreground, a calm alpine lake mirrors
        the colorful sky, creating a symmetrical composition.

        Keywords: sunset, mountains, landscape, alpine lake, reflection,
        golden hour, dramatic sky, nature photography
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
        Caption: Detailed image caption with visual elements.
        Description: Comprehensive description of the scene including
        foreground subjects, background elements, lighting conditions,
        and atmospheric details visible in the photograph.
        Keywords: photography, composition, lighting, subject, background
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
