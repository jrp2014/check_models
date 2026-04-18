"""Unit tests for pure-logic functions that require no mlx-vlm runtime.

Covers: validate_model_identifier, apply_exclusions, prepare_prompt,
compute_vocabulary_diversity, compute_efficiency_metrics,
detect_response_structure, compute_confidence_indicators,
QualityThresholds.from_config, load_quality_config.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

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

    def test_user_provided_prompt_is_logged(
        self, mod: types.ModuleType, caplog: pytest.LogCaptureFixture
    ) -> None:
        """User-provided prompt should be visible in the info log."""
        args = self._make_args(prompt="Describe this photo.")

        with caplog.at_level(logging.INFO, logger=mod.LOGGER_NAME):
            result = mod.prepare_prompt(args, {})

        assert result == "Describe this photo."
        assert "Using user-provided prompt from --prompt." in caplog.messages
        assert "User-provided prompt (--prompt): Describe this photo." in caplog.messages

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


class TestQualityIssueTruncation:
    """Tests for quality issue parsing and truncation helpers."""

    def test_parse_quality_issues_preserves_commas_inside_issue_payloads(
        self,
        mod: types.ModuleType,
    ) -> None:
        """Issue parsing should keep commas inside one parenthesized issue label."""
        quality_issues = "missing-sections(title, description, keywords), context-ignored, cutoff"

        parsed = mod._parse_quality_issues_to_list(quality_issues)

        assert parsed == [
            "missing-sections(title, description, keywords)",
            "context-ignored",
            "cutoff",
        ]

    def test_truncate_quality_issues_keeps_whole_issue_labels(
        self,
        mod: types.ModuleType,
    ) -> None:
        """Truncation should prefer complete parsed issue labels over raw comma cuts."""
        quality_issues = "missing-sections(title, description, keywords), context-ignored, cutoff"

        truncated = mod._truncate_quality_issues(quality_issues, max_len=52)

        assert truncated == "missing-sections(title, description, keywords), ..."

    def test_truncate_quality_issues_hard_clips_single_long_issue(
        self,
        mod: types.ModuleType,
    ) -> None:
        """Truncation should still hard-clip when the first issue alone exceeds the limit."""
        quality_issues = 'repetitive(phrase: "alpha, beta, gamma, delta"), cutoff'

        truncated = mod._truncate_quality_issues(quality_issues, max_len=24)

        assert truncated == 'repetitive(phrase: "a...'


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

    def test_config_driven_patterns(self, mod: types.ModuleType) -> None:
        """Configured confidence patterns should override built-in defaults."""
        original_patterns = mod.QUALITY.patterns
        mod.QUALITY.patterns = {
            "confidence_hedge_patterns": [r"\bhedgeword\b"],
            "confidence_definitive_patterns": [r"\bdefword\b"],
        }
        try:
            result = mod.compute_confidence_indicators("hedgeword defword defword")
            assert result["hedge_count"] == 1
            assert result["definitive_count"] == 2
            assert result["confidence_ratio"] == 0.67
        finally:
            mod.QUALITY.patterns = original_patterns


class TestConfigDrivenCatalogingDetectors:
    """Tests for config-driven task-compliance and visual-grounding patterns."""

    def test_task_compliance_uses_configured_labels(self, mod: types.ModuleType) -> None:
        """Configured labels should be recognized as explicit task sections."""
        original_patterns = mod.QUALITY.patterns
        mod.QUALITY.patterns = {
            "task_caption_labels": ["headline"],
            "task_description_labels": ["notes"],
            "task_keyword_labels": ["terms"],
        }
        try:
            text = "headline: Church tower\nnotes: Stone church in winter.\nterms: church, tower"
            result = mod.compute_task_compliance(text)
            assert result["has_caption"] is True
            assert result["has_description"] is True
            assert result["has_keywords"] is True
        finally:
            mod.QUALITY.patterns = original_patterns

    def test_visual_grounding_uses_configured_patterns(self, mod: types.ModuleType) -> None:
        """Configured visual/spatial/color patterns should drive grounding counts."""
        original_patterns = mod.QUALITY.patterns
        mod.QUALITY.patterns = {
            "visual_grounding_visual_patterns": [r"\bcustomobject\b"],
            "visual_grounding_spatial_patterns": [r"\bcustomspot\b"],
            "visual_grounding_color_patterns": [r"\bcustomcolor\b"],
        }
        try:
            result = mod.compute_visual_grounding(
                "customobject at customspot with customcolor",
                None,
            )
            assert result["visual_terms"] == 1
            assert result["spatial_terms"] == 1
            assert result["color_terms"] == 1
            assert result["grounding_score"] > 0.0
        finally:
            mod.QUALITY.patterns = original_patterns


class TestRegexDetectionUtilities:
    """Tests for shared regex detection helpers."""

    def test_extract_matches_ignores_invalid_regex(self, mod: types.ModuleType) -> None:
        """Invalid configured regex entries should be ignored, not raised."""
        matches = mod._extract_pattern_matches(
            "token 123",
            [r"\d+", r"[invalid"],
            debug_context="test",
        )
        assert matches == ["123"]

    def test_count_and_any_match_ignores_invalid_regex(self, mod: types.ModuleType) -> None:
        """Pattern count/any helpers should skip invalid patterns safely."""
        count = mod._count_pattern_matches("a1 b2 c3", [r"\d", r"[broken"])
        has_match = mod._matches_any_pattern("alpha", [r"[broken", r"beta"], debug_context="test")
        assert count == 3
        assert has_match is False

    def test_compile_regex_cache_reuses_compiled_pattern(self, mod: types.ModuleType) -> None:
        """Regex compiler cache should return the same compiled object for same key."""
        first = mod._compile_regex_cached(r"\d+", 0)
        second = mod._compile_regex_cached(r"\d+", 0)
        assert first is not None
        assert first is second


class TestDisplayWidthUtilities:
    """Tests for wcwidth-aware terminal width helpers."""

    def test_display_width_ignores_ansi_escape_sequences(self, mod: types.ModuleType) -> None:
        """ANSI color wrappers should not count toward rendered width."""
        colored = f"{mod.Colors.RED}abc{mod.Colors.RESET}"
        assert mod._display_width(colored) == 3

    def test_display_align_targets_display_width(
        self,
        mod: types.ModuleType,
    ) -> None:
        """Display alignment should honor display width even with wide glyphs."""
        wide_char = "界"
        padded = mod._display_align(wide_char, 4, alignment="left")
        centered = mod._display_align(wide_char, 5, alignment="center")
        assert mod._display_width(padded) == 4
        assert mod._display_width(centered) == 5

    def test_display_width_uses_wcwidth_when_available(self, mod: types.ModuleType) -> None:
        """When wcwidth is importable, a full-width glyph should occupy two columns."""
        width = mod._display_width("界")
        if mod.wcwidth_wcswidth is None:
            assert width == 1
        else:
            assert width == 2


class TestSanitizeBpeDisplay:
    """Tests for _sanitize_bpe_display BPE artifact cleanup."""

    def test_replaces_bpe_markers(self, mod: types.ModuleType) -> None:
        """BPE markers are replaced with readable equivalents."""
        raw = "Title:\u0120Stone\u0120Church\u010aDescription"
        result = mod._sanitize_bpe_display(raw)
        assert "\u0120" not in result
        assert "\u010a" not in result
        assert "Stone Church" in result

    def test_truncates_long_text(self, mod: types.ModuleType) -> None:
        """Text exceeding max_len is truncated with ellipsis."""
        result = mod._sanitize_bpe_display("a" * 200, max_len=50)
        assert len(result) <= 50
        assert result.endswith("...")

    def test_short_text_unchanged(self, mod: types.ModuleType) -> None:
        """Clean text passes through without modification."""
        result = mod._sanitize_bpe_display("clean text")
        assert result == "clean text"


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
        config: dict[str, object] = {"thresholds": {}, "patterns": {}, "extra_section": {}}
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

    def test_non_mapping_threshold_section_raises(self, mod: types.ModuleType) -> None:
        """Non-mapping threshold sections should fail with a clear schema error."""
        config: dict[str, object] = {"thresholds": [], "patterns": {}}

        with pytest.raises(TypeError, match="thresholds section must be a mapping"):
            mod.QualityThresholds.from_config(config)

    def test_invalid_threshold_bounds_raise(self, mod: types.ModuleType) -> None:
        """Inverted threshold bounds should fail fast instead of weakening checks."""
        config = {
            "thresholds": {"min_title_words": 9, "max_title_words": 4},
            "patterns": {},
        }

        with pytest.raises(ValueError, match="invalid title words bounds"):
            mod.QualityThresholds.from_config(config)

    def test_invalid_pattern_regex_raises(self, mod: types.ModuleType) -> None:
        """Malformed detector patterns should be rejected at config-load time."""
        config = {
            "thresholds": {},
            "patterns": {"hallucination_question_indicators": ["[unterminated"]},
        }

        with pytest.raises(ValueError, match="invalid regex"):
            mod.QualityThresholds.from_config(config)


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

    def test_non_mapping_yaml_warns(
        self,
        mod: types.ModuleType,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Non-mapping YAML content should warn instead of crashing."""
        config_file = tmp_path / "quality_config.yaml"
        config_file.write_text("- unexpected\n- list\n", encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            mod.load_quality_config(config_file)

        assert "top-level document must be a mapping" in caplog.text

    def test_invalid_threshold_config_warns_and_preserves_defaults(
        self,
        mod: types.ModuleType,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Invalid threshold bounds should warn and leave the existing config intact."""
        original_ratio = mod.QUALITY.repetition_ratio
        config_file = tmp_path / "quality_config.yaml"
        config_file.write_text(
            "thresholds:\n  min_title_words: 8\n  max_title_words: 3\npatterns: {}\n",
            encoding="utf-8",
        )

        with caplog.at_level(logging.WARNING):
            mod.load_quality_config(config_file)

        assert "Failed to load" in caplog.text
        assert mod.QUALITY.repetition_ratio == original_ratio

    def test_default_load_uses_packaged_resource(
        self,
        mod: types.ModuleType,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Default loading should use the packaged config resource."""
        packaged_config = tmp_path / "quality_config.yaml"
        packaged_config.write_text(
            "thresholds:\n  repetition_ratio: 0.91\npatterns: {}\n",
            encoding="utf-8",
        )
        original_ratio = mod.QUALITY.repetition_ratio

        def fake_files(package: str) -> Path:
            assert package == "check_models_data"
            return tmp_path

        monkeypatch.setattr(mod.importlib_resources, "files", fake_files)
        monkeypatch.setattr(mod.importlib_resources, "as_file", contextlib.nullcontext)

        mod.load_quality_config()

        assert mod.QUALITY.repetition_ratio == 0.91
        mod.QUALITY.repetition_ratio = original_ratio


class TestSystemProfilerParsing:
    """Tests for typed normalization of macOS system_profiler JSON output."""

    def test_get_device_info_filters_non_mapping_entries(self, mod: types.ModuleType) -> None:
        """Only mapping entries should survive normalization of system_profiler lists."""
        payload = json.dumps(
            {
                "SPDisplaysDataType": [
                    {"sppci_model": "Apple M4", "sppci_cores": 10},
                    "skip-me",
                    5,
                ],
                "SPAudioDataType": [{"_name": "MacBook Speakers"}],
                "_ignored": "scalar",
            }
        )

        mod.get_device_info.cache_clear()
        with (
            patch.object(mod.platform, "system", return_value="Darwin"),
            patch.object(mod.subprocess, "check_output", return_value=payload),
        ):
            info = mod.get_device_info()
        mod.get_device_info.cache_clear()

        assert info == {
            "SPDisplaysDataType": [{"sppci_model": "Apple M4", "sppci_cores": 10}],
            "SPAudioDataType": [{"_name": "MacBook Speakers"}],
        }

    def test_get_device_info_is_cached(self, mod: types.ModuleType) -> None:
        """Repeated device-info reads should reuse the cached system_profiler payload."""
        payload = json.dumps({"SPDisplaysDataType": [{"sppci_model": "Apple M4"}]})

        mod.get_device_info.cache_clear()
        with (
            patch.object(mod.platform, "system", return_value="Darwin"),
            patch.object(mod.subprocess, "check_output", return_value=payload) as check_output,
        ):
            first = mod.get_device_info()
            second = mod.get_device_info()
        mod.get_device_info.cache_clear()

        assert first == second
        assert check_output.call_count == 1

    def test_get_system_info_uses_first_string_gpu_name(self, mod: types.ModuleType) -> None:
        """GPU info should come from the first usable string field in display data."""
        payload = json.dumps(
            {
                "SPDisplaysDataType": [
                    {"sppci_model": "Apple M4 Max", "_name": "Fallback Name"},
                ]
            }
        )

        mod.get_device_info.cache_clear()
        mod.get_system_info.cache_clear()
        with (
            patch.object(mod.platform, "system", return_value="Darwin"),
            patch.object(mod.platform, "machine", return_value="arm64"),
            patch.object(mod.subprocess, "check_output", return_value=payload),
        ):
            arch, gpu_info = mod.get_system_info()
        mod.get_device_info.cache_clear()
        mod.get_system_info.cache_clear()

        assert arch == "arm64"
        assert gpu_info == "Apple M4 Max"


class TestPreflightDependencyDiagnostics:
    """Tests for upstream version-floor and source-pattern diagnostics."""

    def test_is_version_at_least_handles_dev_builds(self, mod: types.ModuleType) -> None:
        """Dev build strings should compare correctly against floor versions."""
        assert mod._is_version_at_least("0.30.7.dev20260214+c184262d", "0.30.4")
        assert mod._is_version_at_least("5.5.3+local", "5.5.3")
        assert not mod._is_version_at_least("5.5.3rc1", "5.5.3")

    def test_collect_upstream_requirements_tracks_strictest_floor(
        self,
        mod: types.ModuleType,
    ) -> None:
        """When multiple stacks are installed, stricter floor should win."""
        requirements = mod._collect_upstream_requirements(
            {
                "mlx-vlm": "0.4.1",
                "mlx-lm": "0.31.1",
                "mlx": "0.31.1",
                "transformers": "5.4.0",
                "huggingface-hub": "1.10.1",
            },
        )
        assert requirements["mlx"][0] == mod.PROJECT_RUNTIME_STACK_MINIMUMS["mlx"]
        assert requirements["mlx-vlm"][0] == mod.PROJECT_RUNTIME_STACK_MINIMUMS["mlx-vlm"]
        assert requirements["transformers"][0] == mod.PROJECT_MIN_TRANSFORMERS_VERSION
        assert requirements["mlx-lm"][0] == mod.PROJECT_OPTIONAL_STACK_MINIMUMS["mlx-lm"]
        assert (
            requirements["huggingface-hub"][0]
            == mod.PROJECT_RUNTIME_STACK_MINIMUMS["huggingface-hub"]
        )

    def test_detect_upstream_version_issues_reports_below_floor(
        self,
        mod: types.ModuleType,
    ) -> None:
        """Installed versions below upstream floors should be surfaced."""
        issues = mod._detect_upstream_version_issues(
            {
                "mlx-vlm": "0.4.1",
                "mlx-lm": "0.30.9",
                "mlx": "0.29.9",
                "transformers": "5.3.9",
                "huggingface-hub": "1.9.9",
            },
        )
        assert any("mlx==0.29.9" in issue and "0.31.1" in issue for issue in issues)
        assert any("mlx-vlm==0.4.1" in issue and "0.4.4" in issue for issue in issues)
        assert any("mlx-lm==0.30.9" in issue and "0.31.3" in issue for issue in issues)
        assert any("transformers==5.3.9" in issue and "5.5.3" in issue for issue in issues)
        assert any("huggingface-hub==1.9.9" in issue and "1.10.1" in issue for issue in issues)

    def test_get_callable_contract_issues_reports_keyword_drift(
        self,
        mod: types.ModuleType,
    ) -> None:
        """Callable contract helper should flag missing keyword params we rely on."""

        def _fake_generate(model: object, processor: object, prompt: str) -> object:
            return (model, processor, prompt)

        issues = mod._get_callable_contract_issues(
            qualified_name="mlx_vlm.generate.generate",
            symbol_value=_fake_generate,
            required_keyword_params=("model", "processor", "prompt", "verbose", "temperature"),
        )

        assert issues == [
            "mlx_vlm.generate.generate is missing required keyword parameter(s): verbose, temperature.",
        ]

    def test_get_generation_result_contract_issues_reports_missing_fields(
        self,
        mod: types.ModuleType,
    ) -> None:
        """GenerationResult shape checks should surface missing upstream fields clearly."""

        @dataclass
        class _IncompleteGenerationResult:
            text: str = ""
            prompt_tokens: int = 0

        issues = mod._get_generation_result_contract_issues(_IncompleteGenerationResult)

        assert len(issues) == 1
        assert "generation_tokens" in issues[0]
        assert "prompt_tps" in issues[0]

    def test_has_mlx_vlm_load_image_path_bug_detection(self, mod: types.ModuleType) -> None:
        """Source matcher should flag unguarded startswith URL branch."""
        risky_source = 'elif image_source.startswith(("http://", "https://")):\n    pass\n'
        safe_source = (
            "elif isinstance(image_source, str) and "
            'image_source.startswith(("http://", "https://")):\n    pass\n'
        )
        assert mod._has_mlx_vlm_load_image_path_bug(risky_source)
        assert not mod._has_mlx_vlm_load_image_path_bug(safe_source)

    def test_has_transformers_backend_guard_names(self, mod: types.ModuleType) -> None:
        """Guard-name detector should reflect source content."""
        assert mod._has_transformers_backend_guard_names("TRANSFORMERS_NO_TF")
        assert mod._has_transformers_backend_guard_names("USE_TF")
        assert not mod._has_transformers_backend_guard_names("USE_TORCH_XLA")

    def test_transformers_backend_guard_defaults_match_source(
        self,
        mod: types.ModuleType,
    ) -> None:
        """Only guard vars referenced by transformers should be exported."""
        source = mod._load_transformers_import_utils_source()
        expected = {
            key: value
            for key, value in mod._TRANSFORMERS_BACKEND_GUARD_ENV_CANDIDATES.items()
            if source is None or key in source
        }
        assert expected == mod._TRANSFORMERS_BACKEND_GUARD_ENV_DEFAULTS

    def test_resolve_distribution_source_file_finds_relative_path(
        self,
        mod: types.ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Distribution file resolver should locate files without importing package modules."""

        class _FakeDistribution:
            def __init__(self, base_dir: Path) -> None:
                self.base_dir = base_dir
                self.files: list[object] = []

            def locate_file(self, file_ref: object) -> Path:
                return self.base_dir / str(file_ref)

        source_file = tmp_path / "mlx_vlm" / "utils.py"
        source_file.parent.mkdir(parents=True)
        source_file.write_text("# test file\n", encoding="utf-8")

        fake_distribution = _FakeDistribution(tmp_path)
        monkeypatch.setattr(
            mod.importlib.metadata,
            "distribution",
            lambda _name: fake_distribution,
        )

        resolved = mod._resolve_distribution_source_file("mlx-vlm", "mlx_vlm/utils.py")
        assert resolved == source_file

    def test_detect_mlx_vlm_load_image_issue_uses_distribution_source_fallback(
        self,
        mod: types.ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """mlx-vlm source inspection should work even when load_image import is unavailable."""
        source_file = tmp_path / "mlx_vlm" / "utils.py"
        source_file.parent.mkdir(parents=True)
        source_file.write_text(
            'elif image_source.startswith(("http://", "https://")):\n    pass\n',
            encoding="utf-8",
        )

        def _stub_load_image(*_args: object, **_kwargs: object) -> None:
            return None

        monkeypatch.setattr(mod, "load_image", _stub_load_image)
        monkeypatch.setattr(
            mod,
            "_resolve_distribution_source_file",
            lambda _name, _relative_path: source_file,
        )

        issue = mod._detect_mlx_vlm_load_image_issue()
        assert issue is not None
        assert "unguarded URL startswith() branch" in issue

    def test_resolve_distribution_source_file_uses_module_spec_fallback(
        self,
        mod: types.ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Resolver should fall back to module search paths for editable installs."""

        class _FakeDistribution:
            files: tuple[object, ...] = ()

            def locate_file(self, _file_ref: object) -> Path:
                return tmp_path / "missing.py"

        class _FakeSpec:
            def __init__(self, location: Path) -> None:
                self.submodule_search_locations = [str(location)]

        pkg_dir = tmp_path / "mlx_vlm"
        pkg_dir.mkdir()
        source_file = pkg_dir / "utils.py"
        source_file.write_text("# editable source\n", encoding="utf-8")

        monkeypatch.setattr(
            mod.importlib.metadata,
            "distribution",
            lambda _name: _FakeDistribution(),
        )
        monkeypatch.setattr(
            mod.importlib_util,
            "find_spec",
            lambda _name: _FakeSpec(pkg_dir),
        )

        resolved = mod._resolve_distribution_source_file("mlx-vlm", "mlx_vlm/utils.py")
        assert resolved == source_file
