"""Unit tests for pure-logic functions that require no mlx-vlm runtime.

Covers: validate_model_identifier, apply_exclusions, prepare_prompt,
compute_vocabulary_diversity, compute_efficiency_metrics,
detect_response_structure, compute_confidence_indicators,
QualityThresholds.from_config, load_quality_config.
"""

from __future__ import annotations

import argparse
import importlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import types

import pytest


@pytest.fixture(scope="module")
def mod() -> types.ModuleType:
    """Import check_models module once for all tests."""
    return importlib.import_module("check_models")


# ── validate_model_identifier ──────────────────────────────────────────────


class TestValidateModelIdentifier:
    """Tests for validate_model_identifier()."""

    def test_valid_hub_id(self, mod: types.ModuleType) -> None:
        """Standard org/name format should pass."""
        mod.validate_model_identifier("mlx-community/nanoLLaVA")

    def test_valid_single_name(self, mod: types.ModuleType) -> None:
        """Single name without slash is valid for hub IDs."""
        mod.validate_model_identifier("nanoLLaVA")

    def test_empty_string_raises(self, mod: types.ModuleType) -> None:
        """Empty string should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            mod.validate_model_identifier("")

    def test_whitespace_only_raises(self, mod: types.ModuleType) -> None:
        """Whitespace-only string should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            mod.validate_model_identifier("   ")

    def test_spaces_in_hub_id_raises(self, mod: types.ModuleType) -> None:
        """Spaces in hub ID should raise ValueError."""
        with pytest.raises(ValueError, match="contains spaces"):
            mod.validate_model_identifier("mlx community/nanoLLaVA")

    def test_local_path_nonexistent_raises(self, mod: types.ModuleType) -> None:
        """Non-existent local path should raise ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            mod.validate_model_identifier("/nonexistent/model/path")

    def test_local_path_file_raises(self, mod: types.ModuleType, tmp_path: Path) -> None:
        """File path (not directory) should raise ValueError."""
        f = tmp_path / "not_a_dir.txt"
        f.write_text("hello")
        with pytest.raises(ValueError, match="not a directory"):
            mod.validate_model_identifier(str(f))

    def test_local_path_valid_dir(self, mod: types.ModuleType, tmp_path: Path) -> None:
        """Existing directory should pass."""
        mod.validate_model_identifier(str(tmp_path))


# ── apply_exclusions ───────────────────────────────────────────────────────


class TestApplyExclusions:
    """Tests for apply_exclusions()."""

    def test_empty_exclusion_list(self, mod: types.ModuleType) -> None:
        """Empty exclusion list should return all models."""
        models = ["a", "b", "c"]
        result = mod.apply_exclusions(models, [], "test")
        assert result == ["a", "b", "c"]

    def test_excludes_matching_models(self, mod: types.ModuleType) -> None:
        """Matching models should be excluded."""
        models = ["model-a", "model-b", "model-c"]
        result = mod.apply_exclusions(models, ["model-b"], "test")
        assert result == ["model-a", "model-c"]

    def test_excludes_multiple(self, mod: types.ModuleType) -> None:
        """Multiple exclusions should all be applied."""
        models = ["a", "b", "c", "d"]
        result = mod.apply_exclusions(models, ["b", "d"], "test")
        assert result == ["a", "c"]

    def test_no_matches_returns_all(self, mod: types.ModuleType) -> None:
        """Non-matching exclusions should leave list intact."""
        models = ["a", "b"]
        result = mod.apply_exclusions(models, ["z"], "test")
        assert result == ["a", "b"]

    def test_preserves_order(self, mod: types.ModuleType) -> None:
        """Original order should be preserved after exclusion."""
        models = ["z", "a", "m"]
        result = mod.apply_exclusions(models, ["a"], "test")
        assert result == ["z", "m"]


# ── prepare_prompt ─────────────────────────────────────────────────────────


class TestPreparePrompt:
    """Tests for prepare_prompt()."""

    @staticmethod
    def _make_args(prompt: str | None = None) -> argparse.Namespace:
        return argparse.Namespace(prompt=prompt)

    def test_user_provided_prompt(self, mod: types.ModuleType) -> None:
        """User-provided prompt should be returned verbatim."""
        args = self._make_args(prompt="Describe this photo.")
        result = mod.prepare_prompt(args, {})
        assert result == "Describe this photo."

    def test_generated_prompt_with_metadata(self, mod: types.ModuleType) -> None:
        """Generated prompt should incorporate metadata fields."""
        metadata = {
            "description": "Sunset over cliffs",
            "date": "2025-10-01",
            "time": "18:30",
            "gps": "51.0N, 0.9W",
        }
        args = self._make_args()
        result = mod.prepare_prompt(args, metadata)
        assert "Sunset over cliffs" in result
        assert "2025-10-01" in result
        assert "18:30" in result
        assert "51.0N, 0.9W" in result

    def test_generated_prompt_empty_metadata(self, mod: types.ModuleType) -> None:
        """Empty metadata should still produce a cataloguing prompt."""
        args = self._make_args()
        result = mod.prepare_prompt(args, {})
        assert "caption" in result.lower() or "cataloguing" in result.lower()


# ── compute_vocabulary_diversity ───────────────────────────────────────────


class TestComputeVocabularyDiversity:
    """Tests for compute_vocabulary_diversity()."""

    def test_empty_string(self, mod: types.ModuleType) -> None:
        """Empty string should return zero diversity."""
        assert mod.compute_vocabulary_diversity("") == (0.0, 0, 0)

    def test_all_unique(self, mod: types.ModuleType) -> None:
        """All unique words should give TTR of 1.0."""
        ttr, unique, total = mod.compute_vocabulary_diversity("apple banana cherry")
        expected_words = 3
        assert unique == expected_words
        assert total == expected_words
        assert ttr == 1.0

    def test_repeated_words(self, mod: types.ModuleType) -> None:
        """Repeated words should give low TTR."""
        ttr, unique, total = mod.compute_vocabulary_diversity("yes yes yes yes")
        expected_total = 4
        expected_ttr = 0.25
        assert unique == 1
        assert total == expected_total
        assert ttr == expected_ttr

    def test_mixed_case_normalized(self, mod: types.ModuleType) -> None:
        """Mixed case should be normalized for diversity calculation."""
        _ttr, unique, total = mod.compute_vocabulary_diversity("Hello hello HELLO")
        expected_total = 3
        assert unique == 1
        assert total == expected_total


# ── compute_efficiency_metrics ─────────────────────────────────────────────


class TestComputeEfficiencyMetrics:
    """Tests for compute_efficiency_metrics()."""

    def test_all_none_inputs(self, mod: types.ModuleType) -> None:
        """All-None inputs should return None metrics."""
        result = mod.compute_efficiency_metrics(100, None, None)
        assert result["tokens_per_second"] is None
        assert result["tokens_per_gb"] is None

    def test_with_time(self, mod: types.ModuleType) -> None:
        """With time only, tokens_per_second should be computed."""
        result = mod.compute_efficiency_metrics(100, 2.0, None)
        expected_tps = 50.0
        assert result["tokens_per_second"] == expected_tps

    def test_with_time_and_memory(self, mod: types.ModuleType) -> None:
        """With both time and memory, all metrics should be computed."""
        result = mod.compute_efficiency_metrics(100, 2.0, 4.0)
        expected_tps = 50.0
        expected_tpg = 25.0
        assert result["tokens_per_second"] == expected_tps
        assert result["tokens_per_gb"] == expected_tpg
        assert result["tokens_per_second_per_gb"] is not None

    def test_zero_time(self, mod: types.ModuleType) -> None:
        """Zero time should return None for tokens_per_second."""
        result = mod.compute_efficiency_metrics(100, 0.0, 4.0)
        assert result["tokens_per_second"] is None


# ── detect_response_structure ──────────────────────────────────────────────


class TestDetectResponseStructure:
    """Tests for detect_response_structure()."""

    def test_empty_text(self, mod: types.ModuleType) -> None:
        """Empty text should have no structure detected."""
        result = mod.detect_response_structure("")
        assert result["has_caption"] is False
        assert result["has_keywords"] is False

    def test_with_keywords(self, mod: types.ModuleType) -> None:
        """Text with Keywords: label should be detected."""
        text = "Caption: A cat.\nKeywords: cat, animal, pet\nDescription: A fluffy cat."
        result = mod.detect_response_structure(text)
        assert result["has_keywords"] is True

    def test_with_sections(self, mod: types.ModuleType) -> None:
        """Text with markdown headings should detect sections."""
        text = "## Caption\nA cat on a mat.\n## Keywords\ncat, mat"
        result = mod.detect_response_structure(text)
        assert result["has_sections"] is True


# ── compute_confidence_indicators ──────────────────────────────────────────


class TestComputeConfidenceIndicators:
    """Tests for compute_confidence_indicators()."""

    def test_empty_text(self, mod: types.ModuleType) -> None:
        """Empty text should have zero hedge and definitive counts."""
        result = mod.compute_confidence_indicators("")
        assert result["hedge_count"] == 0
        assert result["definitive_count"] == 0
        assert result["confidence_ratio"] == 0.0

    def test_hedge_words(self, mod: types.ModuleType) -> None:
        """Text with hedging language should have hedge_count >= 2."""
        text = "It appears to be a cat. It might be sleeping. It seems to be outdoors."
        result = mod.compute_confidence_indicators(text)
        min_hedge_count = 2
        assert result["hedge_count"] >= min_hedge_count

    def test_definitive_text(self, mod: types.ModuleType) -> None:
        """Definitive text should produce computable confidence ratio."""
        text = "This is a red barn. The sky is blue. There are three horses."
        result = mod.compute_confidence_indicators(text)
        assert result["confidence_ratio"] >= 0.0  # at least computable


# ── QualityThresholds.from_config (YAML schema validation) ────────────────


class TestQualityThresholdsFromConfig:
    """Tests for QualityThresholds.from_config() including unknown key warnings."""

    def test_valid_config(self, mod: types.ModuleType) -> None:
        """Valid config should set the specified threshold."""
        expected_ratio = 0.9
        config = {"thresholds": {"repetition_ratio": expected_ratio}, "patterns": {}}
        qt = mod.QualityThresholds.from_config(config)
        assert qt.repetition_ratio == expected_ratio

    def test_unknown_threshold_key_warns(
        self,
        mod: types.ModuleType,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Unknown threshold key should emit a warning."""
        config = {
            "thresholds": {"repetition_ration": 0.9, "repetition_ratio": 0.5},
            "patterns": {},
        }
        with caplog.at_level(logging.WARNING):
            qt = mod.QualityThresholds.from_config(config)
        assert "repetition_ration" in caplog.text
        # The valid key should still be applied
        expected_ratio = 0.5
        assert qt.repetition_ratio == expected_ratio

    def test_unknown_top_level_section_warns(
        self,
        mod: types.ModuleType,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Unknown top-level config section should emit a warning."""
        config = {"thresholds": {}, "patterns": {}, "extra_section": {}}
        with caplog.at_level(logging.WARNING):
            mod.QualityThresholds.from_config(config)
        assert "extra_section" in caplog.text

    def test_empty_config(self, mod: types.ModuleType) -> None:
        """Empty config should use all defaults."""
        config: dict[str, object] = {}
        qt = mod.QualityThresholds.from_config(config)
        # Should use all defaults
        default_ratio = 0.8
        assert qt.repetition_ratio == default_ratio


# ── load_quality_config ───────────────────────────────────────────────────


class TestLoadQualityConfig:
    """Tests for load_quality_config()."""

    def test_nonexistent_path_warns(
        self,
        mod: types.ModuleType,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Non-existent config path should warn."""
        with caplog.at_level(logging.WARNING):
            mod.load_quality_config(Path("/nonexistent/quality_config.yaml"))
        assert "not found" in caplog.text

    def test_valid_yaml_loads(self, mod: types.ModuleType, tmp_path: Path) -> None:
        """Valid YAML config should update thresholds."""
        yaml_content = "thresholds:\n  repetition_ratio: 0.95\npatterns: {}\n"
        config_file = tmp_path / "quality_config.yaml"
        config_file.write_text(yaml_content)
        mod.load_quality_config(config_file)
        expected_ratio = 0.95
        assert mod.QUALITY.repetition_ratio == expected_ratio
        # Restore default
        mod.QUALITY.repetition_ratio = 0.8

    def test_invalid_yaml_warns(
        self,
        mod: types.ModuleType,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Invalid YAML should warn instead of crashing."""
        config_file = tmp_path / "quality_config.yaml"
        config_file.write_text("{{{{invalid yaml")
        with caplog.at_level(logging.WARNING):
            mod.load_quality_config(config_file)
        assert "Failed to load" in caplog.text
