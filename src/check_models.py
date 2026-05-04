#!/usr/bin/env python3
"""Image analysis and caption generation using MLX Vision Language Models."""

from __future__ import annotations

import argparse
import base64
import codecs
import gc
import hashlib
import html
import inspect
import io
import json
import logging
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import textwrap
import time
import traceback
from collections import Counter
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping, Sequence
from contextlib import (
    ContextDecorator,
    ExitStack,
    contextmanager,
    redirect_stderr,
    redirect_stdout,
    suppress,
)
from dataclasses import dataclass, fields, is_dataclass, replace
from dataclasses import field as dataclass_field
from datetime import UTC, datetime
from functools import lru_cache
from importlib.metadata import (
    Distribution,
    PackageNotFoundError,
    distribution,
    distributions,
    version,
)
from importlib.resources import as_file, files
from importlib.util import find_spec
from pathlib import Path
from shlex import join as shlex_join
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Final,
    Literal,
    NoReturn,
    NotRequired,
    Protocol,
    Required,
    Self,
    TextIO,
    TypedDict,
    TypeGuard,
    Unpack,
    cast,
    runtime_checkable,
)
from urllib.error import URLError
from urllib.parse import quote, urlparse
from urllib.request import urlopen

import yaml
from huggingface_hub import HFCacheInfo, scan_cache_dir
from huggingface_hub import __version__ as hf_version
from huggingface_hub.errors import HFValidationError
from packaging.version import InvalidVersion, Version
from rich import box
from rich.bar import Bar
from rich.console import Console, ConsoleRenderable
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from tabulate import tabulate

from check_models_data.dependency_policy import (
    PROJECT_MIN_TRANSFORMERS_VERSION,
    PROJECT_RUNTIME_STACK_MINIMUMS,
    UPSTREAM_MLX_LM_MINIMUMS,
    UPSTREAM_MLX_VLM_MINIMUMS,
)

# Optional dependency: psutil for system info; degrade gracefully if missing.
# Use an intermediate variable with an explicit Optional type so mypy
# doesn't complain about assigning None to a module symbol on the except path.
psutil_mod: Any | None
try:
    import psutil as _psutil_runtime

    psutil_mod = _psutil_runtime
except ImportError:  # pragma: no cover - optional
    psutil_mod = None

psutil: types.ModuleType | None = psutil_mod

# Optional dependency: wcwidth for accurate terminal display-width calculations.
# Without wcwidth, wide Unicode glyphs may be slightly misaligned; we fall back
# to codepoint length to keep output functional.
wcwidth_wcswidth: Callable[[str], int] | None = None
if find_spec("wcwidth") is not None:
    try:
        wcwidth_module = __import__("wcwidth", fromlist=["wcswidth"])
    except ImportError:  # pragma: no cover - optional
        wcwidth_wcswidth = None
    else:
        candidate = getattr(wcwidth_module, "wcswidth", None)
        if callable(candidate):
            wcwidth_wcswidth = cast("Callable[[str], int]", candidate)

if TYPE_CHECKING:
    import types

    from mlx import nn
    from mlx_vlm.generate import GenerationResult
    from PIL.Image import Image as PILImage
    from transformers import PreTrainedTokenizer
    from transformers.configuration_utils import PreTrainedConfig
    from transformers.processing_utils import ProcessorMixin
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

# Public API (PEP 8 / PEP 561 best practice)
__all__ = [
    "GenerationQualityAnalysis",
    "PerformanceResult",
    "ProcessImageParams",
    "ResultSet",
    "analyze_generation_text",
    "append_history_record",
    "compare_history_records",
    "exit_with_cli_error",
    "extract_image_metadata",
    "find_most_recent_file",
    "format_field_label",
    "format_field_value",
    "format_overall_runtime",
    "generate_diagnostics_report",
    "generate_html_report",
    "generate_markdown_report",
    "get_device_info",
    "get_exif_data",
    "get_library_versions",
    "get_system_characteristics",
    "get_system_info",
    "get_terminal_width",
    "is_numeric_field",
    "is_numeric_value",
    "log_blank",
    "log_failure",
    "log_file_path",
    "log_generated_text",
    "log_metric_label",
    "log_model_name",
    "log_success",
    "log_warning_note",
    "main",
    "main_cli",
    "pretty_print_exif",
    "print_cli_error",
    "print_cli_header",
    "print_cli_section",
    "print_model_result",
    "print_version_info",
    "process_image_with_model",
    "validate_cli_arguments",
    "validate_image_accessible",
    "validate_inputs",
    "validate_kv_params",
    "validate_sampling_params",
    "validate_temperature",
]

# =============================================================================
# MODULE CONSTANTS
# =============================================================================

LOGGER_NAME: Final[str] = "mlx-vlm-check"
NOT_AVAILABLE: Final[str] = "N/A"

MISSING_DEPENDENCIES: dict[str, str] = {}

ERROR_MLX_MISSING: Final[str] = "Core dependency missing: mlx. Please install it."
ERROR_MLX_RUNTIME_INIT: Final[str] = (
    "Core dependency initialization failed: mlx runtime could not initialize Metal."
)
ERROR_PILLOW_MISSING: Final[str] = (
    "Error: Pillow not found. Please install it (`pip install Pillow`)."
)
ERROR_MLX_VLM_MISSING: Final[str] = (
    "Error: mlx-vlm not found. Please install it (`pip install mlx-vlm`)."
)
ERROR_MLX_LM_MISSING: Final[str] = (
    "Core dependency missing: mlx-lm. Please reinstall the runtime "
    "(`pip install -e .`) or the full model-support stack "
    '(`pip install -e ".[extras,torch]"`) from `src/`.'
)
ERROR_MLX_VLM_RUNTIME_INIT: Final[str] = (
    "Core dependency initialization failed: mlx-vlm could not be imported safely."
)
MLX_IMPORT_PROBE_TIMEOUT_SECONDS: Final[float] = 8.0
DEFAULT_KV_QUANT_SCHEME: Final = "uniform"
DEFAULT_QUANTIZED_KV_START: Final[int] = 5000
UNIFORM_KV_BITS: Final[frozenset[int]] = frozenset({2, 3, 4, 5, 6, 8})
DEFAULT_THINKING_END_MARKER: Final[str] = "</think>"


# =============================================================================
# CONFIGURATION DATACLASSES - Centralized thresholds and constants
# =============================================================================


@dataclass(frozen=True)
class FormattingThresholds:
    """Centralized thresholds for number/memory/token formatting.

    Consolidates magic numbers used throughout formatting functions to
    improve maintainability and make threshold tuning easier.
    """

    # Number formatting thresholds
    large_number: float = 100.0  # Format as integer with commas
    medium_number: float = 10.0  # One decimal place for TPS
    thousand_separator: int = 1_000  # Add comma separators

    # Memory formatting thresholds
    memory_gb_integer: float = 10.0  # Show GB as integer (no decimals)

    # Time formatting thresholds
    hour_threshold_seconds: float = 3600.0  # Show HH:MM:SS format

    # Dry-run output thresholds
    max_prompt_preview_lines: int = 10  # Max lines to show in prompt preview

    # Output truncation thresholds
    min_consecutive_repetitions_for_truncation: int = 10
    max_preview_tags: int = 3

    # Layout constants
    min_separator_chars: int = 50
    markdown_hard_break_spaces: int = 2
    markdown_wrap_width: int = 78
    generation_wrap_width: int = 100


@dataclass(frozen=True)
class QualityThresholds:
    """Centralized thresholds for quality analysis detection.

    Loaded from configuration file or defaults.
    """

    # Repetition detection
    repetition_ratio: float = 0.8
    min_text_length: int = 10
    min_token_count: int = 5

    # Phrase repetition detection (n-grams)
    min_phrase_repetitions: int = 3
    max_phrase_repetitions: int = 10
    phrase_coverage_threshold: float = 0.4
    min_phrase_length: int = 4  # Minimum n-gram length to check
    max_phrase_length: int = 10  # Maximum n-gram length to check

    # Hallucination detection
    min_pipes_for_table: int = 4
    min_table_rows: int = 2
    min_mc_answers: int = 3
    substantial_text_length: int = 200

    # Formatting violations
    max_markdown_headers: int = 5

    # Context ignorance detection (additional)
    min_context_term_length: int = 2

    # Verbosity detection
    max_verbosity_tokens: int = 300
    min_meta_patterns: int = 2
    min_section_headers: int = 3

    # Bullet point detection (set high for cataloging prompts with keyword lists)
    max_bullets: int = 25

    # Generic output detection
    min_text_length_for_generic: int = 20
    generic_filler_threshold: float = 0.15
    min_specificity_indicators: int = 2

    # Context ignorance detection
    min_key_terms_threshold: int = 3
    min_missing_ratio: float = 0.75

    # Confidence thresholds for output analysis
    high_confidence_threshold: float = 0.7
    medium_confidence_threshold: float = 0.4

    # Output degeneration detection thresholds
    min_text_for_degeneration: int = 20  # Minimum text length to check
    min_cutoff_word_length: int = 2  # Words <= this at end may be cutoff
    max_control_chars: int = 3  # Control chars threshold
    non_ascii_ratio_threshold: float = 0.3  # Threshold for encoding shift detection
    non_ascii_ratio_multiplier: int = 3  # Multiplier for tail vs head comparison
    max_url_length: int = 100  # URLs longer than this are suspicious
    min_precise_stats: int = 2  # Number of overly precise stats to flag

    # Cataloging utility thresholds
    min_useful_words: int = 5  # Minimum words for useful output
    short_output_words: int = 15  # Output considered "short"
    substantial_prose_words: int = 20  # Words needed for "substantial" prose
    max_caption_words: int = 15  # Max words for implicit caption detection
    min_title_words: int = 5  # Minimum words required in Title section
    max_title_words: int = 10  # Maximum words allowed in Title section
    min_description_sentences: int = 1  # Minimum factual description sentences
    max_description_sentences: int = 2  # Maximum factual description sentences
    min_keywords_count: int = 10  # Minimum number of keyword terms
    max_keywords_count: int = 18  # Maximum number of keyword terms
    min_keywords_for_duplication_check: int = 12  # Ignore duplication below this keyword count
    keyword_duplication_ratio_threshold: float = 0.35  # Dup ratio threshold for keyword loops
    min_useful_chars: int = 10  # Minimum chars for useful output
    severe_echo_threshold: float = 0.8  # Echo ratio triggering severe penalty
    moderate_echo_threshold: float = 0.5  # Echo ratio triggering moderate penalty
    context_echo_min_words: int = 30  # Minimum output words before context-echo scoring
    context_echo_vocab_ratio_threshold: float = 0.9  # Vocab overlap threshold for context echo
    context_echo_ngram_size: int = 8  # N-gram size for verbatim context copy detection
    context_echo_min_shared_ngrams: int = 3  # Minimum shared n-grams for context echo
    context_echo_ngram_ratio_threshold: float = 0.25  # Shared n-gram ratio threshold
    low_grounding_threshold: float = 0.3  # Visual grounding considered low
    low_compliance_threshold: float = 0.5  # Task compliance considered low
    low_info_gain_threshold: float = 0.3  # Information gain considered low
    grade_a_threshold: float = 80.0  # Score for A grade
    grade_b_threshold: float = 65.0  # Score for B grade
    grade_c_threshold: float = 50.0  # Score for C grade
    grade_d_threshold: float = 35.0  # Score for D grade
    # Cataloging utility score composition
    cataloging_weight_information_gain: float = 25.0
    cataloging_weight_compliance: float = 30.0
    cataloging_weight_grounding: float = 30.0
    cataloging_weight_length: float = 15.0
    # Cataloging utility penalties/factors
    severe_echo_penalty: float = 0.5
    moderate_echo_penalty: float = 0.8
    very_short_length_factor: float = 0.2
    short_length_factor: float = 0.6
    metadata_agreement_weight_title: float = 0.2
    metadata_agreement_weight_description: float = 0.4
    metadata_agreement_weight_keywords: float = 0.4
    metadata_agreement_nonvisual_penalty: float = 20.0

    # Harness issue detection thresholds
    min_bpe_artifact_count: int = 5  # Min BPE artifacts to flag encoding issue
    min_tokens_for_substantial: int = 10  # Tokens below this are suspicious
    min_words_for_filler_response: int = 15  # Words below this in filler response
    min_words_for_truncated: int = 5  # Words below this = truncated output
    min_prompt_tokens_for_ratio: int = 100  # Prompt tokens needed for ratio check
    min_output_tokens_for_ratio: int = 15  # Output tokens below this with large prompt
    min_output_ratio: float = 0.02  # Minimum output/prompt ratio (2%)
    long_prompt_tokens_threshold: int = 3000  # Prompt tokens above this can degrade outputs
    severe_prompt_tokens_threshold: int = 12000  # Extreme prompt token count risk threshold
    prompt_title_max_chars: int = 120  # Max chars for metadata title hints in default prompt
    prompt_description_max_chars: int = 420  # Max chars for metadata description hints
    prompt_keyword_max_items: int = 20  # Max number of metadata keyword hints
    prompt_keyword_item_max_chars: int = 36  # Max chars per metadata keyword hint
    min_text_for_leak_detection: int = 100  # Min text length for training leak detection
    prompt_word_to_token_ratio: float = 1.3  # Lightweight prompt-token estimate multiplier
    max_reported_missing_terms: int = 5  # Cap human-facing missing-term lists
    cutoff_tail_chars: int = 120  # Tail window inspected for cutoff evidence

    # Unknown anomaly detection
    anomaly_min_output_chars: int = 20  # Near-empty threshold for anomaly promotion

    # Patterns (loaded from config)
    patterns: dict[str, list[str]] | None = None

    def __post_init__(self) -> None:
        """Validate loaded threshold values so bad config cannot degrade checks silently."""
        unit_interval_fields = {
            "repetition_ratio": self.repetition_ratio,
            "phrase_coverage_threshold": self.phrase_coverage_threshold,
            "generic_filler_threshold": self.generic_filler_threshold,
            "min_missing_ratio": self.min_missing_ratio,
            "high_confidence_threshold": self.high_confidence_threshold,
            "medium_confidence_threshold": self.medium_confidence_threshold,
            "non_ascii_ratio_threshold": self.non_ascii_ratio_threshold,
            "keyword_duplication_ratio_threshold": self.keyword_duplication_ratio_threshold,
            "severe_echo_threshold": self.severe_echo_threshold,
            "moderate_echo_threshold": self.moderate_echo_threshold,
            "context_echo_vocab_ratio_threshold": self.context_echo_vocab_ratio_threshold,
            "context_echo_ngram_ratio_threshold": self.context_echo_ngram_ratio_threshold,
            "low_grounding_threshold": self.low_grounding_threshold,
            "low_compliance_threshold": self.low_compliance_threshold,
            "low_info_gain_threshold": self.low_info_gain_threshold,
            "min_output_ratio": self.min_output_ratio,
            "metadata_agreement_weight_title": self.metadata_agreement_weight_title,
            "metadata_agreement_weight_description": self.metadata_agreement_weight_description,
            "metadata_agreement_weight_keywords": self.metadata_agreement_weight_keywords,
        }
        for field_name, value in unit_interval_fields.items():
            if not 0.0 <= value <= 1.0:
                msg = (
                    f"quality_config.yaml thresholds.{field_name} must be between 0 and 1; "
                    f"got {value}"
                )
                raise ValueError(msg)

        ordered_pairs = [
            ("phrase repetitions", self.min_phrase_repetitions, self.max_phrase_repetitions),
            ("title words", self.min_title_words, self.max_title_words),
            (
                "description sentences",
                self.min_description_sentences,
                self.max_description_sentences,
            ),
            ("keywords", self.min_keywords_count, self.max_keywords_count),
        ]
        for label, lower, upper in ordered_pairs:
            if lower > upper:
                msg = f"quality_config.yaml has invalid {label} bounds: min={lower}, max={upper}"
                raise ValueError(msg)

        if self.medium_confidence_threshold > self.high_confidence_threshold:
            msg = (
                "quality_config.yaml thresholds.medium_confidence_threshold must be <= "
                "thresholds.high_confidence_threshold"
            )
            raise ValueError(msg)
        if self.moderate_echo_threshold > self.severe_echo_threshold:
            msg = (
                "quality_config.yaml thresholds.moderate_echo_threshold must be <= "
                "thresholds.severe_echo_threshold"
            )
            raise ValueError(msg)

        self._validate_metadata_agreement_thresholds()

        if self.patterns is not None:
            if not isinstance(self.patterns, dict):
                msg = "quality_config.yaml patterns section must be a mapping"
                raise TypeError(msg)
            for pattern_group, entries in self.patterns.items():
                if not isinstance(entries, list):
                    msg = f"quality_config.yaml patterns.{pattern_group} must be a list of strings"
                    raise TypeError(msg)
                for entry in entries:
                    if not isinstance(entry, str):
                        msg = (
                            f"quality_config.yaml patterns.{pattern_group} entries must be strings"
                        )
                        raise TypeError(msg)
                    try:
                        re.compile(entry)
                    except re.error as exc:
                        msg = (
                            "quality_config.yaml patterns."
                            f"{pattern_group} contains invalid regex {entry!r}: {exc}"
                        )
                        raise ValueError(msg) from exc

    def _validate_metadata_agreement_thresholds(self) -> None:
        """Validate metadata-agreement scoring thresholds."""
        metadata_weight_total = (
            self.metadata_agreement_weight_title
            + self.metadata_agreement_weight_description
            + self.metadata_agreement_weight_keywords
        )
        if not math.isclose(metadata_weight_total, 1.0, rel_tol=0.0, abs_tol=0.01):
            msg = (
                "quality_config.yaml metadata agreement weights must sum to 1.0; "
                f"got {metadata_weight_total:.3f}"
            )
            raise ValueError(msg)

        if self.metadata_agreement_nonvisual_penalty < 0.0:
            msg = (
                "quality_config.yaml thresholds.metadata_agreement_nonvisual_penalty must be "
                f">= 0; got {self.metadata_agreement_nonvisual_penalty}"
            )
            raise ValueError(msg)

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> QualityThresholds:
        """Create instance from configuration dictionary."""
        thresholds = _require_str_object_mapping(
            config.get("thresholds", {}),
            "quality_config.yaml thresholds section must be a mapping",
        )

        patterns_value = config.get("patterns", {})
        if patterns_value is None:
            patterns: dict[str, list[str]] | None = None
        else:
            patterns_mapping = _require_str_object_mapping(
                patterns_value,
                "quality_config.yaml patterns section must be a mapping",
            )
            patterns = cast("dict[str, list[str]]", dict(patterns_mapping))

        # Filter valid fields for the dataclass
        valid_fields = {f.name for f in fields(cls) if f.name != "patterns"}
        filtered_thresholds = {k: v for k, v in thresholds.items() if k in valid_fields}

        # Warn about unrecognised YAML keys (likely typos)
        unknown_threshold_keys = set(thresholds) - valid_fields
        if unknown_threshold_keys:
            logger.warning(
                "Unrecognised keys in quality_config.yaml thresholds (ignored): %s",
                ", ".join(sorted(unknown_threshold_keys)),
            )

        # Warn about unrecognised top-level config sections
        known_sections = {"thresholds", "patterns"}
        unknown_sections = set(config) - known_sections
        if unknown_sections:
            logger.warning(
                "Unrecognised top-level sections in quality_config.yaml (ignored): %s",
                ", ".join(sorted(unknown_sections)),
            )

        return cls(**cast("dict[str, Any]", filtered_thresholds), patterns=patterns)


# Instantiate singletons for runtime use
FORMATTING = FormattingThresholds()
# Default QUALITY instance (will be updated if config is loaded)
QUALITY = QualityThresholds()


def load_quality_config(config_path: Path | None = None) -> None:
    """Load quality configuration from file and update global QUALITY instance.

    If no config_path is provided, loads the bundled default quality config
    resource shipped with the distribution.
    """
    with ExitStack() as exit_stack:
        if config_path is None:
            with suppress(FileNotFoundError, ModuleNotFoundError):
                resource = files("check_models_data").joinpath(
                    "quality_config.yaml",
                )
                config_path = exit_stack.enter_context(as_file(resource))
            if config_path is None:
                logger.warning(
                    "Bundled quality config resource not found: check_models_data/quality_config.yaml",
                )
                return
        elif not config_path.exists():
            logger.warning("Quality config file not found: %s", config_path)
            return

        try:
            with config_path.open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                if config is None:
                    return
                config_mapping = _require_str_object_mapping(
                    config,
                    "quality_config.yaml top-level document must be a mapping",
                )
                new_quality = QualityThresholds.from_config(config_mapping)
                # Update existing global instance in-place to avoid 'global' keyword
                # and ensure all references see the update.
                for field in fields(QualityThresholds):
                    setattr(QUALITY, field.name, getattr(new_quality, field.name))
                logger.debug("Loaded quality configuration from %s", config_path)
        except (OSError, TypeError, ValueError, yaml.YAMLError) as e:
            logger.warning("Failed to load quality config from %s: %s", config_path, e)


_temp_logger = logging.getLogger(LOGGER_NAME)


def _probe_import_runtime(
    *,
    import_target: str,
    error_prefix: str,
    detect_metal_nsrange: bool = False,
) -> str | None:
    """Run a subprocess import probe and return an actionable error message when it fails."""
    max_output_excerpt_chars = 220
    try:
        probe_result = subprocess.run(  # noqa: S603 - fixed interpreter + fixed probe command
            [sys.executable, "-c", f"import {import_target}"],
            capture_output=True,
            text=True,
            check=False,
            timeout=MLX_IMPORT_PROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return (
            f"{error_prefix} Import probe timed out after {MLX_IMPORT_PROBE_TIMEOUT_SECONDS:.0f}s."
        )
    except OSError as probe_err:
        return f"{error_prefix} Import probe failed: {probe_err}"

    if probe_result.returncode == 0:
        return None

    combined_output = probe_result.stderr.strip() or probe_result.stdout.strip()

    if detect_metal_nsrange:
        combined_lower = combined_output.lower()
        if "nsrangeexception" in combined_lower and "objectatindex" in combined_lower:
            return (
                f"{error_prefix} No Metal device could be enumerated "
                "(NSRangeException during device discovery). This commonly happens in "
                "headless or virtualized sessions without visible Apple GPU access."
            )

    if combined_output:
        output_excerpt = " ".join(combined_output.split())
        if len(output_excerpt) > max_output_excerpt_chars:
            output_excerpt = f"{output_excerpt[: max_output_excerpt_chars - 3]}..."
    else:
        output_excerpt = "no output"
    return (
        f"{error_prefix} Import probe exited with code "
        f"{probe_result.returncode}. Probe output: {output_excerpt}"
    )


mx: Any = cast("Any", None)
mlx_probe_error = _probe_import_runtime(
    import_target="mlx.core",
    error_prefix=ERROR_MLX_RUNTIME_INIT,
    detect_metal_nsrange=True,
)
if mlx_probe_error is None:
    try:
        import mlx.core as _mx_runtime
    except ImportError:
        MISSING_DEPENDENCIES["mlx"] = ERROR_MLX_MISSING
    except (OSError, RuntimeError, ValueError) as mlx_init_err:
        MISSING_DEPENDENCIES["mlx"] = f"{ERROR_MLX_RUNTIME_INIT} {mlx_init_err}"
    else:
        mx = cast("Any", _mx_runtime)
else:
    MISSING_DEPENDENCIES["mlx"] = mlx_probe_error

ExifTags: SupportsExifTagsModule
GPSTAGS: Mapping[int, str]
TAGS: Mapping[int, str]
UnidentifiedImageError: type[Exception]
GPS: SupportsGPSEnum | None

try:
    from PIL import Image
    from PIL import UnidentifiedImageError as _PILUnidentifiedImageError

    UnidentifiedImageError = _PILUnidentifiedImageError
except ImportError:
    pillow_version = NOT_AVAILABLE

    class _PILUnavailableError(RuntimeError):
        """Raised when Pillow functionality is requested but unavailable."""

    class _ImageUnavailable:
        """Stub for PIL.Image that raises informative errors when used."""

        @staticmethod
        def open(*_args: object, **_kwargs: object) -> NoReturn:
            raise _PILUnavailableError(ERROR_PILLOW_MISSING)

    class _ExifTagsUnavailable:
        """Stub that surfaces a clear error if EXIF helpers are accessed."""

        def __getattr__(self, _name: str) -> NoReturn:
            raise _PILUnavailableError(ERROR_PILLOW_MISSING)

    ExifTags = cast("SupportsExifTagsModule", _ExifTagsUnavailable())
    Image = cast("Any", _ImageUnavailable())
    UnidentifiedImageError = _PILUnavailableError
    GPS = None
    GPSTAGS = cast("Mapping[int, str]", {})
    TAGS = cast("Mapping[int, str]", {})
    MISSING_DEPENDENCIES["Pillow"] = ERROR_PILLOW_MISSING
else:
    from PIL import ExifTags as PIL_ExifTags
    from PIL import IptcImagePlugin
    from PIL.ExifTags import GPS as PIL_GPS
    from PIL.ExifTags import GPSTAGS as PIL_GPSTAGS
    from PIL.ExifTags import TAGS as PIL_TAGS

    pillow_version = Image.__version__ if hasattr(Image, "__version__") else NOT_AVAILABLE
    ExifTags = cast("SupportsExifTagsModule", PIL_ExifTags)
    GPS = cast("SupportsGPSEnum", PIL_GPS)
    GPSTAGS = cast("Mapping[int, str]", PIL_GPSTAGS)
    TAGS = cast("Mapping[int, str]", PIL_TAGS)

# defusedxml is required by Pillow's Image.getxmp() for safe XMP/XML parsing.
# Declared explicitly in pyproject.toml even though Pillow[xmp] may also pull it
# transitively, so _extract_xmp_metadata() fails clearly if the environment drifts.
_defusedxml_available: bool
try:
    import defusedxml.ElementTree  # noqa: F401 — imported for availability check

    _defusedxml_available = True
except ImportError:
    _defusedxml_available = False
    _temp_logger.warning(
        "defusedxml not installed — XMP metadata extraction will be disabled. "
        "Install with: pip install defusedxml",
    )

try:
    import numpy as np

    numpy_version: str = getattr(np, "__version__", NOT_AVAILABLE)
except ImportError:
    numpy_version = NOT_AVAILABLE


def _raise_mlx_vlm_missing(*_args: object, **_kwargs: object) -> NoReturn:
    """Raise a consistent runtime error when mlx-vlm is unavailable."""
    raise RuntimeError(ERROR_MLX_VLM_MISSING)


vlm_version: str = NOT_AVAILABLE
generate: Callable[..., GenerationResult] = cast(
    "Callable[..., GenerationResult]", _raise_mlx_vlm_missing
)
apply_chat_template: ApplyChatTemplateCallable = cast(
    "ApplyChatTemplateCallable", _raise_mlx_vlm_missing
)
load: LoadCallable = cast("LoadCallable", _raise_mlx_vlm_missing)
load_image: LoadImageCallable = cast("LoadImageCallable", _raise_mlx_vlm_missing)


mlx_vlm_probe_error = _probe_import_runtime(
    import_target="mlx_vlm",
    error_prefix=ERROR_MLX_VLM_RUNTIME_INIT,
)
if mlx_vlm_probe_error is None:
    try:
        from mlx_vlm.generate import generate as _mlx_vlm_generate
        from mlx_vlm.prompt_utils import apply_chat_template as _mlx_vlm_apply_chat_template
        from mlx_vlm.utils import load as _mlx_vlm_load
        from mlx_vlm.utils import load_image as _mlx_vlm_load_image
        from mlx_vlm.version import __version__ as _mlx_vlm_version

        generate = _mlx_vlm_generate
        apply_chat_template = _mlx_vlm_apply_chat_template
        load = _mlx_vlm_load
        load_image = _mlx_vlm_load_image
        vlm_version = _mlx_vlm_version
    except ImportError:
        MISSING_DEPENDENCIES["mlx-vlm"] = ERROR_MLX_VLM_MISSING
else:
    MISSING_DEPENDENCIES["mlx-vlm"] = mlx_vlm_probe_error

try:
    version("mlx-lm")
except PackageNotFoundError:
    MISSING_DEPENDENCIES["mlx-lm"] = ERROR_MLX_LM_MISSING


# =============================================================================
# TYPE ALIASES & PROTOCOLS
# =============================================================================

type ExifValue = Any  # Pillow yields varied scalar / tuple EXIF types; keep permissive
type ExifDict = dict[str | int, ExifValue]
type MetadataDict = dict[str, str | None]
type PathLike = str | Path
type JsonLike = None | bool | int | float | str | list[JsonLike] | dict[str, JsonLike]
type GPSDict = dict[str, ExifValue]  # GPS EXIF data structure
type SystemProfilerEntry = dict[str, object]
type SystemProfilerDict = dict[
    str, list[SystemProfilerEntry]
]  # macOS system_profiler JSON structure
type LibraryVersionDict = dict[str, str | None]  # Library name to version mapping (optional values)
type MetricValue = int | float | str | bool | None  # Common scalar metric variants for metrics


class IPTCMetadata(TypedDict, total=False):
    """Normalized IPTC metadata extracted from an image."""

    iptc_keywords: list[str]
    iptc_caption: str


class XMPMetadata(TypedDict, total=False):
    """Normalized XMP metadata extracted from an image."""

    xmp_keywords: list[str]
    xmp_description: str
    xmp_title: str


class SupportsExifBaseNamespace(Protocol):
    """Minimal subset of ``PIL.ExifTags.Base`` used by EXIF helpers."""

    ExifOffset: int
    GPSInfo: int


class SupportsExifIfdNamespace(Protocol):
    """Minimal subset of ``PIL.ExifTags.IFD`` used by EXIF helpers."""

    Exif: int
    GPSInfo: int


class SupportsExifTagsModule(Protocol):
    """Minimal subset of Pillow's ``ExifTags`` namespace used here."""

    Base: SupportsExifBaseNamespace
    IFD: SupportsExifIfdNamespace


class SupportsGPSName(Protocol):
    """Minimal GPS enum member surface used for tag-name lookup."""

    name: str


class SupportsGPSEnum(Protocol):
    """Callable enum-like surface for Pillow GPS tag lookup."""

    def __call__(self, value: int) -> SupportsGPSName:
        """Return the GPS enum member for a numeric tag id."""
        ...


class HistoryModelResultRecord(TypedDict):
    """Per-model success/failure status stored in run history."""

    success: bool
    error_stage: str | None
    error_type: str | None
    error_package: str | None
    failure_phase: NotRequired[str | None]
    error_code: NotRequired[str | None]
    error_signature: NotRequired[str | None]
    review_verdict: NotRequired[str | None]
    review_owner: NotRequired[str | None]
    review_user_bucket: NotRequired[str | None]
    review_issue_subtype: NotRequired[str | None]
    harness_issue_type: NotRequired[str | None]
    prompt_output_ratio: NotRequired[float | None]
    nontext_prompt_ratio: NotRequired[float | None]
    stop_reason: NotRequired[str | None]
    hit_max_tokens: NotRequired[bool]


class HistoryRunRecord(TypedDict, total=False):
    """Append-only run record shape for ``*.history.jsonl``."""

    _type: Literal["run"]
    format_version: str
    timestamp: str
    prompt_hash: str
    prompt_preview: str
    image_path: str | None
    model_results: dict[str, HistoryModelResultRecord]
    system: dict[str, str]
    library_versions: LibraryVersionDict
    runtime_fingerprint: dict[str, RuntimeProbeResult]


class HistoryComparisonSummary(TypedDict):
    """Run-over-run transition summary for history comparisons."""

    regressions: list[str]
    recoveries: list[str]
    new_models: list[str]
    missing_models: list[str]
    quality_regressions: list[str]
    quality_recoveries: list[str]
    harness_regressions: list[str]
    harness_recoveries: list[str]
    owner_changes: list[str]


class JsonlTimingRecord(TypedDict):
    """Timing fields emitted per result row in ``results.jsonl``."""

    generation_time_s: float | None
    model_load_time_s: float | None
    total_time_s: float | None
    input_validation_time_s: float | None
    prompt_prep_time_s: float | None
    cleanup_time_s: float | None
    first_token_latency_s: float | None
    stop_reason: str | None


class JsonlMetricsRecord(TypedDict, total=False):
    """Performance metrics emitted for successful generations."""

    prompt_tokens: int
    generation_tokens: int
    total_tokens: int
    prompt_tps: float
    generation_tps: float
    peak_memory_gb: float
    active_memory_gb: float
    cache_memory_gb: float


class JsonlQualityAnalysisMetrics(TypedDict):
    """Subset of quality metrics included in JSONL output."""

    word_count: int
    unique_ratio: float
    bullet_count: int


class JsonlQualityAnalysisRecord(TypedDict):
    """Quality-analysis payload attached to successful rows when available."""

    issues: list[str]
    metrics: JsonlQualityAnalysisMetrics


class JsonlMetadataAgreementRecord(TypedDict):
    """Metadata-agreement payload attached to successful rows when available."""

    overall_score: float
    title_score: float
    description_score: float
    keyword_score: float
    nonvisual_penalty: float
    matched_terms: list[str]
    missed_terms: list[str]
    nonvisual_hits: list[str]


class JsonlReviewRecord(TypedDict):
    """Canonical automated review payload attached to JSONL result rows."""

    verdict: str
    hint_relationship: str
    instruction_echo: bool
    metadata_borrowing: bool
    likely_capped: bool
    owner: str
    user_bucket: str
    evidence: list[str]
    requested_max_tokens: int | None
    hit_max_tokens: bool
    prompt_tokens_total: int | None
    prompt_tokens_text_est: int | None
    prompt_tokens_nontext_est: int | None
    prompt_output_ratio: float | None
    nontext_prompt_ratio: float | None
    missing_terms: list[str]
    missing_sections: list[str]
    harness_details: list[str]


type MaintainerConfidence = Literal["high", "medium", "low"]


class JsonlMaintainerTriageRecord(TypedDict, total=False):
    """Action-oriented maintainer triage payload attached to JSONL result rows."""

    suspected_owner: Required[str]
    confidence: Required[MaintainerConfidence]
    issue_kind: Required[str]
    summary: Required[str]
    next_action: Required[str]
    user_bucket: Required[str]
    evidence: Required[list[str]]
    harness_details: Required[list[str]]
    hit_max_tokens: Required[bool]
    issue_subtype: str
    error_stage: str | None
    error_code: str | None
    stop_reason: str | None
    requested_max_tokens: int | None
    prompt_tokens_total: int | None
    prompt_output_ratio: float | None
    nontext_prompt_ratio: float | None
    issue_cluster_id: str
    issue_cluster_path: str
    acceptance_signal: str


class RuntimeProbeResult(TypedDict, total=False):
    """Status of a single runtime capability probe."""

    status: Required[Literal["ok", "unavailable", "errored", "timed_out"]]
    detail: str


class JsonlMetadataRecord(TypedDict, total=False):
    """Shared header row for ``results.jsonl`` format."""

    _type: Required[Literal["metadata"]]
    format_version: Required[str]
    prompt: Required[str]
    system: Required[dict[str, str]]
    timestamp: Required[str]
    library_versions: LibraryVersionDict
    runtime_fingerprint: dict[str, RuntimeProbeResult]


class JsonlRerunSummary(TypedDict, total=False):
    """Secondary rerun evidence attached to JSONL result rows."""

    rerun_success: Required[bool]
    rerun_verdict: str
    rerun_error_code: str
    rerun_generated_chars: int
    rerun_generation_time: float
    rerun_prompt: str


class JsonlSignatureComponents(TypedDict, total=False):
    """Structured components of the error signature for precise cross-run clustering."""

    error_code: str
    normalized_message: str
    traceback_signature: str


class JsonlResultRecord(TypedDict):
    """Per-model row shape for ``results.jsonl`` output."""

    _type: Literal["result"]
    model: str
    success: bool
    failure_phase: str | None
    error_stage: str | None
    error_code: str | None
    error_signature: str | None
    error_message: str | None
    captured_output_on_fail: str | None
    error_type: str | None
    root_error_type: NotRequired[str]
    root_error_module: NotRequired[str]
    root_error_message: NotRequired[str]
    error_package: str | None
    error_traceback: str | None
    quality_issues: list[str]
    timestamp: str
    metrics: JsonlMetricsRecord
    timing: JsonlTimingRecord
    generated_text: NotRequired[str]
    quality_analysis: NotRequired[JsonlQualityAnalysisRecord]
    metadata_agreement: NotRequired[JsonlMetadataAgreementRecord]
    review: NotRequired[JsonlReviewRecord]
    maintainer_triage: NotRequired[JsonlMaintainerTriageRecord]
    prompt_diagnostics: NotRequired[dict[str, JsonLike]]
    signature_components: NotRequired[JsonlSignatureComponents]
    rerun_summary: NotRequired[JsonlRerunSummary]


type FailedModelIssue = tuple[str, str | None, str | None]
type RepetitiveModelIssue = tuple[str, str | None]
type HallucinationModelIssue = tuple[str, list[str]]
type VerboseModelIssue = tuple[str, int]
type FormattingModelIssue = tuple[str, list[str]]
type ExcessiveBulletsIssue = tuple[str, int]
type LowUtilityModelIssue = tuple[str, float, str, str]
type ModelScoreGrade = tuple[str, float, str]
type CatalogingScoreRecord = tuple[str, float, str, str, float | None]
type TopPerformerMetric = tuple[str, str, float, str]
type ResourceUsageMetric = tuple[str, float, str]
type AggregateStatRow = tuple[str, str, str, str]
type QualityIssueEntry = tuple[str, str | None]
type QualityIssueSection = tuple[str, str, list[QualityIssueEntry]]
type RuntimePhaseName = Literal[
    "model_load",
    "prompt_prep",
    "upstream_prefill_first_token",
    "post_prefill_decode",
    "decode",
    "cleanup",
]


class RuntimeAnalysisSummary(TypedDict):
    """Aggregated runtime interpretation shared across report renderers."""

    dominant_phase: RuntimePhaseName
    dominant_phase_share: float
    dominant_phase_count: int
    measured_models: int
    interpretation: str
    next_action: str
    phase_totals: dict[RuntimePhaseName, float]
    dominant_counts: dict[RuntimePhaseName, int]
    termination_counts: dict[str, int]
    generation_total: float
    generation_models: int
    prefill_split_models: int
    validation_total: float
    validation_models: int
    first_token_latency_avg: float | None
    first_token_latency_min: float | None
    first_token_latency_max: float | None
    first_token_latency_models: int


class ModelIssueSummary(TypedDict, total=False):
    """Aggregated per-run model issue summary used in HTML/Markdown reports."""

    total_models: int
    failed_models: list[FailedModelIssue]
    repetitive_models: list[RepetitiveModelIssue]
    hallucination_models: list[HallucinationModelIssue]
    verbose_models: list[VerboseModelIssue]
    formatting_issues: list[FormattingModelIssue]
    excessive_bullets: list[ExcessiveBulletsIssue]
    cataloging_grades: dict[str, list[str]]
    cataloging_best: ModelScoreGrade | None
    cataloging_worst: ModelScoreGrade | None
    cataloging_best_description: tuple[str, float]
    cataloging_best_keywords: tuple[str, float]
    cataloging_avg_score: float
    cataloging_scores: list[CatalogingScoreRecord]
    runtime_analysis: RuntimeAnalysisSummary
    metadata_baseline_score: float
    metadata_baseline_grade: str
    cataloging_avg_delta: float
    cataloging_improves_metadata: list[str]
    cataloging_neutral_vs_metadata: list[str]
    cataloging_worse_than_metadata: list[str]
    low_utility_models: list[LowUtilityModelIssue]
    fastest_model: tuple[str, float]
    most_efficient_model: tuple[str, float]
    fastest_load_model: tuple[str, float]
    average_tps: float
    successful_count: int
    total_peak_memory: float
    average_peak_memory: float
    memory_efficiency: float


class NumericFieldStats(TypedDict):
    """Simple numeric aggregate stats for one metric field."""

    min: float
    max: float
    avg: float


type PerformanceStats = dict[str, NumericFieldStats]


@dataclass(frozen=True)
class CatalogingSummaryData:
    """Shared cataloging summary data consumed by HTML and Markdown renderers."""

    grade_counts: tuple[str, ...] = ()
    average_score: float | None = None
    metadata_breakdown: tuple[float, str, float, int, int, int] | None = None
    best_entry: ModelScoreGrade | None = None
    worst_entry: ModelScoreGrade | None = None
    best_description_entry: tuple[str, float] | None = None
    best_keyword_entry: tuple[str, float] | None = None
    low_utility_models: tuple[LowUtilityModelIssue, ...] = ()


@dataclass(frozen=True)
class DiagnosticsHistoryInputs:
    """Optional history inputs for diagnostics report generation."""

    history_path: Path | None = None
    previous_history: HistoryRunRecord | None = None
    current_history: HistoryRunRecord | None = None
    preflight_issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class DiagnosticsArtifacts:
    """Diagnostics artifacts emitted at the end of a run."""

    snapshot: DiagnosticsSnapshot
    diagnostics_written: bool = False
    repro_bundles: Mapping[str, Path] = dataclass_field(default_factory=dict)
    issue_reports: Mapping[str, Path] = dataclass_field(default_factory=dict)


@dataclass(frozen=True)
class ReportGenerationInputs:
    """Inputs required to generate final report artifacts and log their paths."""

    args: argparse.Namespace
    results: list[PerformanceResult]
    library_versions: LibraryVersionDict
    prompt: str
    metadata: MetadataDict | None
    overall_time: float
    image_path: Path | None
    system_info: dict[str, str]
    report_context: ReportRenderContext
    jsonl_output_path: Path
    log_output_path: Path
    env_output_path: Path
    review_output_path: Path
    runtime_fingerprint: dict[str, RuntimeProbeResult] | None = None


class ChatTemplateKwargs(TypedDict, total=False):
    """Supported optional kwargs forwarded to ``mlx_vlm.apply_chat_template``."""

    enable_thinking: bool


class GenerateExtraKwargs(TypedDict, total=False):
    """Optional upstream generate kwargs this CLI forwards explicitly."""

    min_p: float
    top_k: int
    prefill_step_size: int
    resize_shape: tuple[int, int]
    eos_tokens: list[str]
    skip_special_tokens: bool
    enable_thinking: bool
    thinking_end_token: str
    thinking_budget: int
    thinking_start_token: str


class SupportsTextDecoder(Protocol):
    """Minimal tokenizer/processor decode interface used by preflight checks."""

    def decode(self, *args: object, **kwargs: object) -> object:
        """Decode one token sequence."""
        del args, kwargs
        raise NotImplementedError

    def batch_decode(self, *args: object, **kwargs: object) -> object:
        """Decode one or more token sequences."""
        del args, kwargs
        raise NotImplementedError


class LoadCallable(Protocol):
    """Typed ``mlx_vlm.utils.load`` surface used by this script."""

    def __call__(
        self,
        path_or_hf_repo: str,
        adapter_path: str | None = None,
        lazy: bool = False,
        revision: str | None = None,
        *,
        trust_remote_code: bool = True,
        **kwargs: object,
    ) -> tuple[nn.Module, ProcessorMixin]:
        """Load an MLX-VLM model and processor."""
        del path_or_hf_repo, adapter_path, lazy, revision, trust_remote_code, kwargs
        raise NotImplementedError


class ApplyChatTemplateCallable(Protocol):
    """Typed ``mlx_vlm.prompt_utils.apply_chat_template`` surface."""

    def __call__(
        self,
        processor: ProcessorMixin,
        config: Mapping[str, object] | object | None,
        prompt: str | dict[str, object] | list[object],
        add_generation_prompt: bool = True,
        return_messages: bool = False,
        num_images: int = 0,
        num_audios: int = 0,
        **kwargs: Unpack[ChatTemplateKwargs],
    ) -> str | list[dict[str, object]] | object:
        """Apply an upstream chat template to a prompt payload."""
        del processor, config, prompt
        del add_generation_prompt, return_messages, num_images, num_audios, kwargs
        raise NotImplementedError


class StrictGenerateCallable(Protocol):
    """Typed ``mlx_vlm.generate.generate`` surface for known kwargs."""

    def __call__(
        self,
        model: nn.Module,
        processor: ProcessorMixin | PreTrainedTokenizer,
        prompt: str,
        image: str | list[str] | None = None,
        audio: str | list[str] | None = None,
        verbose: bool = False,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float | None = None,
        repetition_context_size: int | None = 20,
        max_kv_size: int | None = None,
        kv_bits: float | None = None,
        kv_quant_scheme: str = DEFAULT_KV_QUANT_SCHEME,
        kv_group_size: int = 64,
        quantized_kv_start: int = DEFAULT_QUANTIZED_KV_START,
        max_tokens: int = 500,
        **kwargs: Unpack[GenerateExtraKwargs],
    ) -> GenerationResult:
        """Generate a caption/response with the known CLI-controlled kwargs."""
        del model, processor, prompt, image, audio, verbose
        del temperature, top_p, repetition_penalty, repetition_context_size
        del max_kv_size, kv_bits, kv_quant_scheme, kv_group_size, quantized_kv_start
        del max_tokens, kwargs
        raise NotImplementedError


class LoadImageCallable(Protocol):
    """Typed ``mlx_vlm.utils.load_image`` surface used for validation."""

    def __call__(self, image_source: str | Path | io.BytesIO, timeout: int = 10) -> object:
        """Load an image from a path or URL."""
        del image_source, timeout
        raise NotImplementedError


if TYPE_CHECKING:
    from mlx_vlm.generate import generate as _mlx_vlm_generate_typecheck

    _TYPECHECK_MODEL = cast("nn.Module", None)
    _TYPECHECK_GENERATE_PROCESSOR = cast("PreTrainedTokenizer", None)
    _ = _mlx_vlm_generate_typecheck(
        _TYPECHECK_MODEL,
        _TYPECHECK_GENERATE_PROCESSOR,
        "",
        image=None,
        audio=None,
        verbose=False,
        max_tokens=1,
        temperature=0.0,
        repetition_penalty=None,
        repetition_context_size=20,
        top_p=1.0,
        min_p=0.0,
        top_k=0,
        max_kv_size=None,
        kv_bits=None,
        kv_quant_scheme=DEFAULT_KV_QUANT_SCHEME,
        kv_group_size=64,
        quantized_kv_start=DEFAULT_QUANTIZED_KV_START,
        prefill_step_size=None,
        resize_shape=None,
        eos_tokens=None,
        skip_special_tokens=False,
        enable_thinking=False,
        thinking_budget=None,
        thinking_end_token=DEFAULT_THINKING_END_MARKER,
        thinking_start_token=None,
    )


@runtime_checkable
class SupportsGenerationText(Protocol):
    """Minimal generation surface stored on ``PerformanceResult``.

    Report/test callers may pass lightweight stand-ins that only expose text
    plus a subset of numeric metrics. Consumers should use ``getattr`` or the
    helpers below when reading optional metrics from stored results.
    """

    @property
    def text(self) -> str | None:
        """Return generated text content when available."""
        ...


@runtime_checkable
class SupportsGenerationResult(SupportsGenerationText, Protocol):
    """Structural subset of live GenerationResult objects enriched locally.

    Using a Protocol keeps typing resilient to upstream changes in the
    concrete GenerationResult while still giving linters strong guarantees
    about the attributes actually consumed here.

    Note: `time`, `active_memory`, and `cache_memory` are added dynamically by
    our code after generation. `peak_memory` comes from the upstream
    GenerationResult returned by mlx-vlm.
    """

    token: object | None
    logprobs: object | None
    prompt_tokens: int | None
    generation_tokens: int | None
    total_tokens: int | None
    prompt_tps: float | None
    generation_tps: float | None
    time: float | None  # Dynamically added timing attribute
    active_memory: float | None  # Dynamically added active memory (GB)
    cache_memory: float | None  # Dynamically added cache memory (GB)
    peak_memory: float | None  # Upstream peak memory (GB)


class SupportsExifIfd(Protocol):
    """Minimal interface for EXIF objects providing nested IFD access."""

    def get_ifd(self, tag: object) -> Mapping[object, ExifValue] | None:
        """Retrieve a nested IFD mapping by tag identifier."""
        ...


type StoredGenerationResult = GenerationResult | SupportsGenerationText


def _generation_numeric_metric(generation: object | None, field_name: str) -> int | float | None:
    """Return an optional numeric generation metric when present."""
    value = getattr(generation, field_name, None)
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int | float) else None


def _generation_int_metric(generation: object | None, field_name: str) -> int | None:
    """Return an optional integer generation metric when present."""
    value = _generation_numeric_metric(generation, field_name)
    return value if isinstance(value, int) else None


def _generation_float_metric(generation: object | None, field_name: str) -> float | None:
    """Return an optional float generation metric when present."""
    value = _generation_numeric_metric(generation, field_name)
    return float(value) if value is not None else None


def _generation_text_value(generation: object | None) -> str:
    """Return generation text when available, else an empty string."""
    value = getattr(generation, "text", "")
    return value if isinstance(value, str) else ""


# =============================================================================
# APPLICATION CONSTANTS & DEFAULTS
# =============================================================================

# These constants define default values for various parameters used in the script.
DEFAULT_MAX_TOKENS: Final[int] = 500
DEFAULT_FOLDER: Final[Path] = Path.home() / "Pictures" / "Processed"
# Output paths relative to script's directory (not CWD) for consistency
_SCRIPT_DIR = Path(__file__).parent
DEFAULT_HTML_OUTPUT: Final[Path] = _SCRIPT_DIR / "output" / "reports" / "results.html"
DEFAULT_MD_OUTPUT: Final[Path] = _SCRIPT_DIR / "output" / "reports" / "results.md"
DEFAULT_GALLERY_MD_OUTPUT: Final[Path] = _SCRIPT_DIR / "output" / "reports" / "model_gallery.md"
DEFAULT_REVIEW_MD_OUTPUT: Final[Path] = _SCRIPT_DIR / "output" / "reports" / "review.md"
DEFAULT_TSV_OUTPUT: Final[Path] = _SCRIPT_DIR / "output" / "reports" / "results.tsv"
DEFAULT_LOG_OUTPUT: Final[Path] = _SCRIPT_DIR / "output" / "check_models.log"
DEFAULT_JSONL_OUTPUT: Final[Path] = _SCRIPT_DIR / "output" / "results.jsonl"
DEFAULT_ENV_OUTPUT: Final[Path] = _SCRIPT_DIR / "output" / "environment.log"
DEFAULT_DIAGNOSTICS_OUTPUT: Final[Path] = _SCRIPT_DIR / "output" / "reports" / "diagnostics.md"
_PREFLIGHT_ISSUES_ARG_ATTR: Final[str] = "_check_models_preflight_issues"

DEFAULT_TEMPERATURE: Final[float] = 0.0  # Greedy/deterministic (matches mlx-vlm upstream)
DEFAULT_TIMEOUT: Final[float] = 300.0  # Default timeout in seconds
MAX_REASONABLE_TEMPERATURE: Final[float] = 2.0  # Warn if temperature exceeds this

# Additional Application Constants
GRADE_EMOJIS: Final[dict[str, str]] = {"A": "🏆", "B": "✅", "C": "🟡", "D": "🟠", "F": "❌"}
IMAGE_OPEN_TIMEOUT: Final[float] = 5.0  # Timeout for opening/verifying image files
SUPPORTED_IMAGE_EXTENSIONS: Final[frozenset[str]] = frozenset({".jpg", ".jpeg", ".png", ".webp"})

# Constants - EXIF
# Use Pillow's modern ExifTags enums (Pillow 10.0+) for type safety
IMPORTANT_EXIF_TAGS: Final[frozenset[str]] = frozenset(
    {
        "DateTimeOriginal",
        "ImageDescription",
        "CreateDate",
        "Make",
        "Model",
        "LensModel",
        "ExposureTime",
        "FNumber",
        "ISOSpeedRatings",
        "FocalLength",
        "ExposureProgram",
    },
)
DATE_FORMATS: Final[tuple[str, ...]] = (
    "%Y:%m:%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y%m%d",
)
EXIF_DATE_TAGS: Final[tuple[str, ...]] = (
    "DateTimeOriginal",
    "CreateDate",
    "DateTime",
)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass(frozen=True)
class PromptDiagnostics:
    """Bounded prompt/rendering context for maintainer repro artifacts."""

    model_type: str | None = None
    processor_class: str | None = None
    tokenizer_class: str | None = None
    rendered_prompt_hash_sha256: str | None = None
    rendered_prompt_preview: str | None = None
    rendered_prompt_chars: int | None = None
    image_placeholder_count: int | None = None
    eos_token_id: JsonLike = None
    eos_token: str | None = None
    special_token_ids: tuple[JsonLike, ...] = ()
    special_tokens: tuple[str, ...] = ()
    generate_kwargs: dict[str, JsonLike] = dataclass_field(default_factory=dict)


@dataclass(frozen=True)
class PerformanceResult:
    """Encapsulates a GenerationResult and execution metadata for a model run.

    Attributes:
        model_name: HuggingFace model identifier (e.g., 'mlx-community/nanoLLaVA')
        generation: The mlx-vlm GenerationResult, or None if generation failed
        success: Whether generation completed without errors
        failure_phase: First execution phase that failed (import/model_load/prefill/decode...)
        error_stage: Human-readable error class (OOM, API Mismatch, Model Error, ...)
        error_code: Canonical machine-readable code for cross-run bucketing
        error_signature: Stable signature hash for clustering related failures
        error_message: Human-readable error description
        captured_output_on_fail: Stderr/stdout captured when error occurred
        generation_time: Wall-clock time for text generation (excludes model load)
        model_load_time: Wall-clock time to load model into GPU memory
        total_time: End-to-end time including all stages
        error_type: Exception class name for error categorization in reports
        quality_issues: Comma-separated list of detected output problems
        quality_analysis: Structured quality-analysis result for triage/reporting
        metadata_agreement: Post-generation score against trusted image metadata
        active_memory: GPU memory in use (GB), from mx.get_active_memory()
        cache_memory: GPU memory in cache (GB), from mx.get_cache_memory()
        error_package: Which package raised the error (mlx, mlx-vlm, transformers)
        error_traceback: Full traceback for actionable error reports
        requested_max_tokens: Requested generation budget for cutoff diagnosis
        prompt_diagnostics: Prompt/template and generation-kwarg context for maintainers
    """

    model_name: str
    generation: StoredGenerationResult | None
    success: bool
    failure_phase: str | None = None
    error_stage: str | None = None
    error_code: str | None = None
    error_signature: str | None = None
    error_message: str | None = None
    captured_output_on_fail: str | None = None
    generation_time: float | None = None
    model_load_time: float | None = None
    total_time: float | None = None
    error_type: str | None = None
    root_error_type: str | None = None
    root_error_module: str | None = None
    root_error_message: str | None = None
    quality_issues: str | None = None
    quality_analysis: GenerationQualityAnalysis | None = None
    metadata_agreement: MetadataAgreementMetrics | None = None
    active_memory: float | None = None
    cache_memory: float | None = None
    error_package: str | None = None
    error_traceback: str | None = None
    runtime_diagnostics: RuntimeDiagnostics | None = None
    requested_max_tokens: int | None = None
    prompt_diagnostics: PromptDiagnostics | None = None
    rerun_evidence: RerunEvidence | None = None


@dataclass(frozen=True)
class RerunEvidence:
    """Secondary evidence from a differential rerun (never overwrites first-pass)."""

    rerun_success: bool
    rerun_verdict: str | None = None
    rerun_error_code: str | None = None
    rerun_generated_chars: int | None = None
    rerun_generation_time: float | None = None
    rerun_prompt: str | None = None


@dataclass(frozen=True)
class MetadataAgreementMetrics:
    """Post-generation agreement score against trusted image metadata."""

    overall_score: float = 0.0
    title_score: float = 0.0
    description_score: float = 0.0
    keyword_score: float = 0.0
    nonvisual_penalty: float = 0.0
    matched_terms: tuple[str, ...] = ()
    missed_terms: tuple[str, ...] = ()
    nonvisual_hits: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimeDiagnostics:
    """Detailed runtime attribution and termination diagnostics for one run."""

    input_validation_time_s: float | None = None
    model_load_time_s: float | None = None
    prompt_prep_time_s: float | None = None
    decode_time_s: float | None = None
    cleanup_time_s: float | None = None
    first_token_latency_s: float | None = None
    stop_reason: str | None = None


class PhaseTimer:
    """Track named phase durations for one model run."""

    def __init__(self) -> None:
        self._durations: dict[str, float] = {}
        self._start_times: dict[str, float] = {}

    def start(self, phase: str) -> None:
        """Start timing a named phase if it is not already active."""
        if phase not in self._start_times:
            self._start_times[phase] = time.perf_counter()

    def stop(self, phase: str) -> float | None:
        """Stop timing a named phase and return the elapsed time."""
        start_time = self._start_times.pop(phase, None)
        if start_time is None:
            return None
        elapsed = time.perf_counter() - start_time
        self._durations[phase] = self._durations.get(phase, 0.0) + elapsed
        return elapsed

    @contextmanager
    def track(self, phase: str) -> Generator[None]:
        """Context manager that records elapsed time for a named phase."""
        self.start(phase)
        try:
            yield
        finally:
            self.stop(phase)

    def duration(self, phase: str) -> float | None:
        """Return the accumulated duration for a phase, if available."""
        duration = self._durations.get(phase)
        if duration is None:
            return None
        return duration


class _TeeCaptureStream(io.TextIOBase):
    """Mirror writes to an underlying stream while buffering text for diagnostics."""

    __slots__ = ("_buffer", "_stream")

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._buffer = io.StringIO()

    def writable(self) -> bool:
        return True

    def write(self, data: str) -> int:
        self._buffer.write(data)
        return self._stream.write(data)

    def flush(self) -> None:
        self._stream.flush()

    def isatty(self) -> bool:
        return self._stream.isatty()

    def getvalue(self) -> str:
        return self._buffer.getvalue()


# Gallery rendering helpers (outside class)
def _wrap_markdown_text(
    text: str,
    *,
    initial_indent: str = "",
    subsequent_indent: str = "",
    width: int = FORMATTING.markdown_wrap_width,
) -> list[str]:
    """Wrap plain Markdown text to the repository lint width."""
    if text == "":
        return [initial_indent.rstrip()]
    wrapped = textwrap.wrap(
        text,
        width=width,
        initial_indent=initial_indent,
        subsequent_indent=subsequent_indent,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return wrapped or [initial_indent.rstrip()]


def _markdown_emphasis(text: str) -> str:
    """Return repo-style Markdown emphasis for formatter-owned labels."""
    return f"_{text}_"


def _append_markdown_labeled_value(
    parts: list[str],
    *,
    label: str,
    value: str,
    bullet: bool = False,
) -> None:
    """Append a wrapped Markdown label/value line using underscore emphasis."""
    prefix = (
        f"- {_markdown_emphasis(f'{label}:')} " if bullet else f"{_markdown_emphasis(f'{label}:')} "
    )
    subsequent_indent = "  " if bullet else " " * len(prefix)
    parts.extend(
        _wrap_markdown_text(
            value,
            initial_indent=prefix,
            subsequent_indent=subsequent_indent,
        ),
    )


def _append_markdown_row_block(
    parts: list[str],
    *,
    rows: Sequence[tuple[str, str]],
    heading: str | None = None,
) -> None:
    """Append a compact Markdown stanza rendered as bullet rows."""
    if not rows:
        return
    if heading is not None:
        parts.extend([heading, ""])
    for label, value in rows:
        _append_markdown_labeled_value(parts, label=label, value=value, bullet=True)
    parts.append("")


@dataclass(frozen=True)
class ReportStanza:
    """Small shared payload for scan-friendly human-facing report sections."""

    heading: str | None
    rows: tuple[tuple[str, str], ...]


def _build_report_stanza(
    heading: str | None,
    rows: Sequence[tuple[str, str | None]],
) -> ReportStanza:
    """Build a stanza while dropping empty optional rows."""
    return ReportStanza(
        heading=heading,
        rows=tuple((label, value) for label, value in rows if value),
    )


def _append_markdown_stanza(
    parts: list[str],
    stanza: ReportStanza,
    *,
    heading_level: int = 3,
) -> None:
    """Append a reusable stanza as a short heading plus labeled bullet rows."""
    if not stanza.rows:
        return
    if stanza.heading is not None:
        parts.append(f"{'#' * heading_level} {stanza.heading}")
        parts.append("")
    _append_markdown_row_block(parts, rows=stanza.rows)


def _render_html_stanza(stanza: ReportStanza) -> list[str]:
    """Render a reusable stanza as a heading paragraph plus HTML list."""
    if not stanza.rows:
        return []
    parts: list[str] = []
    if stanza.heading is not None:
        parts.append(f"<p><b>{html.escape(stanza.heading)}</b></p>")
    parts.append("<ul>")
    for label, value in stanza.rows:
        parts.append(
            f"<li><b>{html.escape(label)}:</b> {html.escape(value)}</li>",
        )
    parts.append("</ul>")
    return parts


@dataclass(frozen=True)
class ReviewPayload:
    """Canonical review inputs reused across human-facing report renderers."""

    review: JsonlReviewRecord
    analysis: GenerationQualityAnalysis | None
    key_signals: str
    next_action: str


def _append_markdown_review_block(
    out: list[str],
    *,
    res: PerformanceResult,
) -> None:
    """Append the shared canonical review block to a Markdown artifact."""
    rows = _build_markdown_review_block_rows(res)
    _append_markdown_row_block(out, rows=rows)


def _build_gallery_error_block_lines(res: PerformanceResult) -> list[str]:
    """Build the failure block used by the Markdown gallery."""
    out: list[str] = []
    max_inline_error_length = 80

    error_msg = _escape_markdown_blockquote_line(str(res.error_message))
    review_rows = dict(_build_markdown_review_block_rows(res))
    failure_context = "; ".join(
        f"{label.lower()} `{value}`"
        for label, value in (
            ("Type", res.error_type),
            ("Phase", res.failure_phase),
            ("Code", res.error_code),
            ("Package", res.error_package),
        )
        if value
    )
    rows = [
        ("Recommendation", review_rows.get("Recommendation")),
        ("Status", f"Failed ({res.error_stage})"),
        ("Owner", review_rows.get("Owner")),
        ("Next step", review_rows.get("Next step")),
        (
            "Error summary",
            error_msg if len(error_msg) <= max_inline_error_length else "see error details below",
        ),
        ("Key signals", review_rows.get("Key signals")),
        ("Failure context", failure_context),
        ("Tokens", review_rows.get("Tokens")),
    ]
    _append_markdown_row_block(out, rows=_build_report_stanza(None, rows).rows)

    if len(error_msg) > max_inline_error_length:
        wrapped_lines = textwrap.wrap(
            error_msg,
            width=76,
            break_long_words=False,
            break_on_hyphens=False,
        )
        out.append(_markdown_emphasis("Error details:"))
        out.append("")
        out.extend(f"> {line}" for line in wrapped_lines)

    if res.error_traceback:
        out.append("")
        traceback_lines: list[str] = []
        _append_markdown_code_block(
            traceback_lines,
            res.error_traceback.rstrip(),
            language="python",
        )
        _append_markdown_details_block(
            out,
            summary="Full Traceback (click to expand)",
            body_lines=traceback_lines,
        )
    return out


def _gallery_metric_segment(label: str, field_name: str, value: MetricValue) -> str | None:
    """Return one compact timing segment for gallery rows."""
    formatted = format_field_value(field_name, value)
    if not formatted:
        return None
    return f"{label} {formatted}"


def _gallery_throughput_segment(
    label: str,
    tps_value: float | None,
    token_field_name: str,
    token_value: int | None,
) -> str | None:
    """Return one compact throughput segment for gallery rows."""
    tps_formatted = format_field_value("generation_tps", tps_value)
    tokens_formatted = format_field_value(token_field_name, token_value)
    if not tps_formatted and not tokens_formatted:
        return None
    if tps_formatted and tokens_formatted:
        return f"{label} {tps_formatted} TPS ({tokens_formatted} tok)"
    if tps_formatted:
        return f"{label} {tps_formatted} TPS"
    return f"{label} {tokens_formatted} tok"


def _build_gallery_success_triage_rows(
    res: PerformanceResult,
    *,
    summary: ModelIssueSummary | None,
    useful_now: bool,
    watchlist_reason: str | None,
) -> list[tuple[str, str]]:
    """Build score and review-focus rows for one successful gallery entry."""
    rows: list[tuple[str, str]] = []
    if summary is not None:
        score_data = _cataloging_score_index(summary).get(res.model_name)
        if score_data is not None:
            score, grade, weakness, delta = score_data
            assessment_parts = [_grade_display_parts(grade, score)]
            if delta is not None:
                assessment_parts.append(f"Δ{delta:+.0f}")
            if weakness:
                assessment_parts.append(weakness)
            rows.append(("Score", _markdown_review_text(" | ".join(assessment_parts))))

    if useful_now:
        rows.append(("Review focus", "strong candidate for first-pass review"))
    elif watchlist_reason is not None:
        rows.append(("Review focus", f"watchlist ({_humanize_watchlist_reason(watchlist_reason)})"))
    return rows


def _build_gallery_success_performance_rows(
    res: PerformanceResult,
    generation: StoredGenerationResult | None,
) -> list[tuple[str, str]]:
    """Build timing and throughput rows for one successful gallery entry."""
    time_segments = [
        segment
        for segment in (
            _gallery_metric_segment("Load", "model_load_time", res.model_load_time),
            _gallery_metric_segment("Gen", "generation_time", res.generation_time),
            _gallery_metric_segment("Total", "total_time", res.total_time),
        )
        if segment is not None
    ]
    rows = [("Timing", _markdown_review_text(" | ".join(time_segments)))] if time_segments else []
    if not time_segments or generation is None:
        return rows

    throughput_segments = [
        segment
        for segment in (
            _gallery_throughput_segment(
                "Prompt",
                _generation_float_metric(generation, "prompt_tps"),
                "prompt_tokens",
                _generation_int_metric(generation, "prompt_tokens"),
            ),
            _gallery_throughput_segment(
                "Gen",
                _generation_float_metric(generation, "generation_tps"),
                "generation_tokens",
                _generation_int_metric(generation, "generation_tokens"),
            ),
        )
        if segment is not None
    ]
    if throughput_segments:
        rows.append(("Throughput", _markdown_review_text(" | ".join(throughput_segments))))
    return rows


def _append_gallery_quality_lines(
    out: list[str],
    *,
    res: PerformanceResult,
    generation: StoredGenerationResult | None,
) -> None:
    """Append quality warnings or clean status after a generated-output block."""
    if generation is None:
        return

    analysis = _quality_analysis_for_result(res)
    if analysis and analysis.issues:
        if not out or out[-1] != "":
            out.append("")
        out.append(f"⚠️ {_markdown_emphasis('Quality Warnings:')}")
        out.append("")
        for issue in analysis.issues:
            escaped = DIAGNOSTICS_ESCAPER.escape(_collapse_preview_whitespace(issue))
            sanitized = re.sub(r"(?<!\\)([*`])", r"\\\1", escaped)
            out.append(f"- {sanitized}")
        return

    if analysis is not None and not analysis.has_any_issues():
        if not out or out[-1] != "":
            out.append("")
        _append_markdown_labeled_value(
            out,
            label="Quality Status",
            value="no quality issues detected in this run",
        )


def _build_gallery_success_block_lines(
    res: PerformanceResult,
    *,
    summary: ModelIssueSummary | None = None,
    useful_now: bool = False,
    watchlist_reason: str | None = None,
) -> list[str]:
    """Build the success block used by the Markdown gallery."""
    generation: StoredGenerationResult | None = res.generation
    rows = _build_markdown_review_block_rows(res)
    token_rows = [(label, value) for label, value in rows if label == "Tokens"]
    summary_rows = [(label, value) for label, value in rows if label != "Tokens"]
    summary_rows.extend(
        _build_gallery_success_triage_rows(
            res,
            summary=summary,
            useful_now=useful_now,
            watchlist_reason=watchlist_reason,
        )
    )
    summary_rows.extend(_build_gallery_success_performance_rows(res, generation))
    summary_rows.extend(token_rows)

    out: list[str] = []
    _append_markdown_row_block(out, rows=summary_rows)

    text = _generation_text_value(generation)
    _append_markdown_wrapped_blockquote(out, text)

    _append_gallery_quality_lines(out, res=res, generation=generation)

    return out


class ResultSet:
    """Cache-friendly wrapper around a collection of PerformanceResult.

    Provides:
        - Single-pass sorting by generation time (fastest first)
        - Lazy discovery/caching of available metric fields (generation + timing)
        - Simple iteration / length protocol support

    Rationale: Multiple rendering paths previously repeated the same sort
    and field extraction logic. Centralizing reduces duplication and
    guarantees consistent ordering across console, HTML, and Markdown
    outputs.
    """

    __slots__ = ("_failed", "_fields", "_results", "_successful")  # Sorted for lint consistency

    def __init__(self, results: list[PerformanceResult]) -> None:
        """Initialize and sort results.

        A shallow copy of ``results`` is taken to guard against external
        mutation after construction.
        """
        self._results = _sort_results_by_time(list(results))
        self._fields: list[str] | None = None
        self._successful: list[PerformanceResult] | None = None
        self._failed: list[PerformanceResult] | None = None

    # Public API -----------------------------------------------------
    @property
    def results(self) -> list[PerformanceResult]:  # Sorted
        """Return results sorted by generation time (fastest first)."""
        return self._results

    @property
    def successful(self) -> list[PerformanceResult]:
        """Return cached list of successful results."""
        if self._successful is None:
            self._successful = [r for r in self._results if r.success]
        return self._successful

    @property
    def failed(self) -> list[PerformanceResult]:
        """Return cached list of failed results."""
        if self._failed is None:
            self._failed = [r for r in self._results if not r.success]
        return self._failed

    def get_fields(self) -> list[str]:
        """Return cached list of metric field names (generation + timing)."""
        if self._fields is None:
            self._fields = _get_available_fields(self._results)
        return self._fields

    # Dunder conveniences --------------------------------------------
    def __len__(self) -> int:  # pragma: no cover - trivial
        """Return number of results."""
        return len(self._results)

    def __iter__(self) -> Iterator[PerformanceResult]:  # pragma: no cover - trivial
        """Iterate over sorted results."""
        return iter(self._results)


@dataclass(frozen=True)
class ProcessImageParams:
    """Parameters for processing an image with a model.

    Centralizes all parameters needed for model inference into a single
    immutable structure. This approach:
        - Reduces function signature complexity (single param vs many)
        - Makes parameter passing explicit and type-safe
        - Simplifies testing (one object to mock/construct)
        - Documents expected inputs clearly
    """

    model_identifier: str
    image_path: str | Path  # Support both file paths and URLs
    prompt: str
    max_tokens: int
    temperature: float
    timeout: float
    verbose: bool
    trust_remote_code: bool
    top_p: float
    min_p: float
    top_k: int
    repetition_penalty: float | None
    repetition_context_size: int
    lazy: bool
    max_kv_size: int | None
    kv_bits: float | None
    kv_quant_scheme: Literal["uniform", "turboquant"]
    kv_group_size: int
    quantized_kv_start: int
    revision: str | None = None
    adapter_path: str | None = None
    prefill_step_size: int | None = None
    resize_shape: tuple[int, int] | None = None
    eos_tokens: tuple[str, ...] | None = None
    skip_special_tokens: bool = False
    processor_kwargs: dict[str, JsonLike] | None = None
    enable_thinking: bool = False
    thinking_budget: int | None = None
    thinking_start_token: str | None = None
    thinking_end_token: str = DEFAULT_THINKING_END_MARKER
    context_marker: str = "Context:"


# =============================================================================
# INFRASTRUCTURE: Timeouts, Logging
# =============================================================================


class TimingStrategy(Protocol):
    """Protocol for timing operations."""

    def start(self) -> None:
        """Start the timer."""
        ...

    def stop(self) -> float:
        """Stop the timer and return the elapsed time in seconds."""
        ...


class PerfCounterTimer:
    """Default timing strategy using time.perf_counter()."""

    def __init__(self) -> None:
        self._start_time: float | None = None

    def start(self) -> None:
        self._start_time = time.perf_counter()

    def stop(self) -> float:
        if self._start_time is None:
            msg = "Timer was not started"
            raise RuntimeError(msg)
        return time.perf_counter() - self._start_time


# Custom timeout context manager
# Python 3.11+ has asyncio.timeout, but signal-based timeout works across sync code
# Uses SIGALRM which is Unix-only - Windows doesn't support this signal mechanism
class TimeoutManager(ContextDecorator):
    """Apply a SIGALRM-based timeout around synchronous code on Unix."""

    def __init__(self, seconds: float) -> None:
        """Store the timeout window that ``__enter__`` arms via SIGALRM.

        Args:
            seconds: Maximum runtime, in seconds, before ``TimeoutError`` is raised.

        """
        self.seconds: float = seconds
        self.timer: signal._HANDLER | None = None

    def _timeout_handler(
        self,
        _signum: int,
        _frame: types.FrameType | None,
    ) -> NoReturn:
        msg: str = f"Operation timed out after {self.seconds} seconds"
        raise TimeoutError(msg)

    def __enter__(self) -> Self:
        """Enter the timeout context manager."""
        if hasattr(signal, "SIGALRM"):
            if self.seconds > 0:
                try:
                    self.timer = signal.signal(
                        signal.SIGALRM,
                        self._timeout_handler,
                    )
                    signal.alarm(math.ceil(self.seconds))
                except ValueError as e:
                    # Signal handling restricted (threading/subprocess environment)
                    logger.warning(
                        "Could not set SIGALRM for timeout: %s. "
                        "Timeout disabled - operations may hang indefinitely.",
                        e,
                    )
                    self.seconds = 0
        elif self.seconds > 0:
            logger.warning(
                "Timeout functionality requires signal.SIGALRM, "
                "not available on this platform (e.g., Windows). "
                "Timeout disabled.",
            )
            self.seconds = 0
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: types.TracebackType | None,
    ) -> None:
        """Exit the timeout context manager and clear the alarm."""
        # Reset signal handler only if we successfully set it up
        if hasattr(signal, "SIGALRM") and self.seconds > 0 and self.timer is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self.timer)


# Configure logging - Single logger instance
logger: logging.Logger = logging.getLogger(LOGGER_NAME)

# Disable Hugging Face tokenizers parallelism to avoid fork-related warnings/deadlocks.
# This must be set before tokenizers are created/used.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Global rendering width override (set via --width); when set, all width
# calculations should honor this value instead of auto-detected terminal width.
WIDTH_OVERRIDE: int | None = None
ANSI_ESCAPE_RE: Final[re.Pattern[str]] = re.compile(
    r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])",
)


def _strip_ansi(text: str) -> str:
    """Strip ANSI escape codes from text for width calculations."""
    return ANSI_ESCAPE_RE.sub("", text)


def _display_width(text: str) -> int:
    """Return terminal display width with Unicode-aware fallback behavior."""
    sanitized = _strip_ansi(text)
    if wcwidth_wcswidth is None:
        return len(sanitized)
    width = wcwidth_wcswidth(sanitized)
    # wcwidth returns -1 for indeterminate width; fall back to codepoint length.
    return len(sanitized) if width < 0 else width


def _display_align(text: str, width: int, *, alignment: Literal["left", "center"]) -> str:
    """Pad text based on display width rather than codepoint length."""
    pad_total = max(0, width - _display_width(text))
    if alignment == "center":
        left = pad_total // 2
        right = pad_total - left
        return f"{' ' * left}{text}{' ' * right}"
    return f"{text}{' ' * pad_total}"


# =============================================================================
# UTILITY CLASSES (Logging, Timeout Management)
# =============================================================================


class _RichColorState:
    """Mutable Rich color preference shared by console handlers."""

    enabled: ClassVar[bool] = (
        True
        if os.getenv("FORCE_COLOR", "").lower() in {"1", "true", "yes"}
        else (sys.stderr.isatty() and os.getenv("NO_COLOR") is None)
    )


def _set_rich_color_enabled(*, enabled: bool) -> None:
    """Globally enable or disable Rich console color for this process."""
    _RichColorState.enabled = enabled


def _rich_color_enabled() -> bool:
    """Return whether styled terminal output is enabled."""
    return _RichColorState.enabled


class LogStyles:
    """String constants describing structured log presentation styles."""

    HEADER: ClassVar[str] = "header"
    SECTION: ClassVar[str] = "section"
    RULE: ClassVar[str] = "rule"
    ERROR: ClassVar[str] = "error"
    SUCCESS: ClassVar[str] = "success"
    WARNING: ClassVar[str] = "warning"
    DETAIL: ClassVar[str] = "detail"
    METRIC_LABEL: ClassVar[str] = "metric_label"
    METRIC_VALUE: ClassVar[str] = "metric_value"
    GENERATED_TEXT: ClassVar[str] = "generated_text"
    FILE_PATH: ClassVar[str] = "file_path"
    MODEL_NAME: ClassVar[str] = "model_name"


def _make_rich_console(*, width: int | None = None) -> Console:
    """Create the shared console object used by Rich logging."""
    colors_enabled = _rich_color_enabled()
    force_terminal = True if colors_enabled and os.getenv("FORCE_COLOR") is not None else None
    return Console(
        stderr=True,
        width=width,
        force_terminal=force_terminal,
        no_color=not colors_enabled,
        markup=False,
        highlight=False,
        emoji=True,
    )


LOCAL_TIMESTAMP_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S %Z"
CONSOLE_LOG_TIME_FORMAT: Final[str] = "[%H:%M:%S]"


class StyleAwareRichHandler(RichHandler):
    """Rich log handler that honors the script's existing structured style hints."""

    HINT_STYLES: ClassVar[dict[str, str]] = {
        LogStyles.RULE: "blue",
        LogStyles.HEADER: "bold magenta",
        LogStyles.SECTION: "bold magenta",
        LogStyles.ERROR: "bold red",
        LogStyles.SUCCESS: "bold green",
        LogStyles.WARNING: "bold yellow",
        LogStyles.DETAIL: "cyan",
        LogStyles.METRIC_LABEL: "bold white",
        LogStyles.METRIC_VALUE: "white",
        LogStyles.GENERATED_TEXT: "cyan",
        LogStyles.FILE_PATH: "cyan",
        LogStyles.MODEL_NAME: "magenta",
    }
    LEVEL_STYLES: ClassVar[dict[int, str]] = {
        logging.DEBUG: "dim",
        logging.WARNING: "yellow",
        logging.ERROR: "bold red",
        logging.CRITICAL: "bold red reverse",
    }

    def render_message(self, record: logging.LogRecord, message: str) -> ConsoleRenderable:
        """Render log records as Rich Text without changing the stored log record."""
        style_hint = getattr(record, "style_hint", None)
        rendered_message = self._message_for_hint(style_hint, message, record)
        style = self._style_for_record(record, style_hint)
        return Text(rendered_message, style=style)

    def get_level_text(self, record: logging.LogRecord) -> Text:
        """Render DEBUG's level label in gray while preserving message styling."""
        if record.levelno == logging.DEBUG:
            return Text.styled(record.levelname.ljust(8), "dim")
        return super().get_level_text(record)

    def _message_for_hint(
        self,
        style_hint: object,
        message: str,
        record: logging.LogRecord,
    ) -> str:
        """Apply lightweight structural transforms before Rich styling."""
        if style_hint == LogStyles.HEADER:
            width = int(getattr(record, "style_width", max(_display_width(message), 1)))
            return _display_align(message, width, alignment="center")
        if style_hint == LogStyles.SECTION:
            uppercase_enabled = bool(getattr(record, "style_uppercase", True))
            title = message.upper() if uppercase_enabled and "\x1b[" not in message else message
            return f"{getattr(record, 'style_prefix', '▶')} [ {title} ]"
        if style_hint == LogStyles.METRIC_LABEL:
            emoji = getattr(record, "style_emoji", "")
            return f"{emoji} {message}" if isinstance(emoji, str) and emoji else message
        return message

    def _style_for_record(self, record: logging.LogRecord, style_hint: object) -> str:
        """Return the Rich style for a log record."""
        style_override = getattr(record, "style_override", None)
        if isinstance(style_override, str) and style_override:
            return style_override
        if isinstance(style_hint, str):
            return self.HINT_STYLES.get(style_hint, "")
        return self.LEVEL_STYLES.get(record.levelno, "")


class FileSafeFormatter(logging.Formatter):
    """Formatter for file logs that strips ANSI/control formatting codes."""

    def format(self, record: logging.LogRecord) -> str:
        raw = super().format(record)
        return ANSI_ESCAPE_RE.sub("", raw)


def _make_console_log_handler(
    *,
    level: int,
    verbose: bool,
    width: int | None = None,
) -> StyleAwareRichHandler:
    """Create the Rich console log handler for stderr output."""
    handler = StyleAwareRichHandler(
        console=_make_rich_console(width=width),
        show_time=True,
        omit_repeated_times=False,
        show_level=verbose,
        show_path=False,
        markup=False,
        highlighter=None,
        rich_tracebacks=verbose,
        log_time_format=CONSOLE_LOG_TIME_FORMAT,
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(message)s"))
    return handler


# Configure an import-time console handler; setup_environment replaces it per CLI flags.
handler: StyleAwareRichHandler = _make_console_log_handler(
    level=logging.INFO,
    verbose=True,
)
logger.handlers.clear()
logger.addHandler(handler)

DECIMAL_GB: Final[float] = 1_000_000_000.0  # Decimal GB (mlx-vlm already divides by 1e9)
MEM_BYTES_TO_GB_THRESHOLD: Final[float] = 1_000_000.0  # > ~1MB treat as raw bytes from mlx
MEGAPIXEL_CONVERSION: Final[float] = 1_000_000.0  # Convert pixels to megapixels

# EXIF/GPS coordinate standards: camera hardware stores GPS as degrees-minutes-seconds tuples
MAX_GPS_COORD_LEN: Final[int] = 3  # Full GPS coordinate (degrees, minutes, seconds)
MED_GPS_COORD_LEN: Final[int] = 2  # GPS coordinate with 2 elements (degrees, minutes)
MIN_GPS_COORD_LEN: Final[int] = 1  # GPS coordinate with 1 element (degrees only)
RATIONAL_TUPLE_LEN: Final[int] = 2  # EXIF rational values are (numerator, denominator) tuples
EXIF_USERCOMMENT_PREFIX_LEN: Final[int] = 8  # EXIF UserComment has 8-byte encoding prefix

# --- EXIF & Metadata Heuristics ---
CJK_IDEOGRAPH_START: Final[int] = 0x4E00
MOJIBAKE_HEURISTIC_LEN: Final[int] = 50
MAX_TUPLE_LEN: Final[int] = 10
MAX_STR_LEN: Final[int] = 60
STR_TRUNCATE_LEN: Final[int] = 57

# Field display configuration: maps field names to (label, unit) tuples.
# Used by format_field_label() and table generators for consistent column headers.
FIELD_ABBREVIATIONS: Final[dict[str, tuple[str, str]]] = {
    "model_name": ("Model", "Name"),
    "prompt_tokens": ("Prompt", "(tokens)"),
    "generation_tokens": ("Generation", "(tokens)"),
    "total_tokens": ("Total", "Tokens"),
    "prompt_tps": ("Prompt", "Tps"),
    "generation_tps": ("Gen", "TPS"),
    "peak_memory": ("Peak", "(GB)"),
    "generation_time": ("Generation", "(s)"),
    "model_load_time": ("Load", "(s)"),
    "total_time": ("Total", "(s)"),
    "quality_issues": ("Quality", "Issues"),
    "active_memory": ("Active", "Mem (GB)"),
    "cache_memory": ("Cache", "Mem (GB)"),
    "error_package": ("Error", "Package"),
}

# Threshold for splitting long header text into multiple lines
HEADER_SPLIT_LENGTH = 10
ERROR_MESSAGE_TRUNCATE_LEN: Final[int] = 120  # Max chars for error messages in actionable reports
MAX_QUALITY_ISSUES_LEN: Final[int] = 30  # Max chars for quality issues in Markdown tables
MAX_OUTPUT_LINES: Final[int] = 3  # Max lines to show in summary table cells
MAX_OUTPUT_PREVIEW_CHARS: Final[int] = 280  # Max chars for output previews in summary tables
MAX_TSV_CELL_CHARS: Final[int] = 300  # Hard cap for any single TSV cell value
OUTPUT_PREVIEW_CUE_LIMIT: Final[int] = 3  # Max issue cues shown before compact output text
OUTPUT_PREVIEW_MIN_HEAD_CHARS: Final[int] = 96  # Minimum chars reserved for preview head
OUTPUT_PREVIEW_MIN_TAIL_CHARS: Final[int] = 48  # Minimum chars reserved for preview tail
OUTPUT_PREVIEW_MIN_BODY_CHARS: Final[int] = 24  # Smallest useful body budget after cue prefix
MAX_CAPTURED_OUTPUT_LOG_CHARS: Final[int] = 1200  # Max chars of captured stdout/stderr in logs
MAX_TRIAGE_MODELS: Final[int] = 5  # Max model rows shown in triage subsections
SUMMARY_CHART_WIDTH: Final[int] = 24  # Character width for compact Rich summary bars
SUMMARY_MODEL_LABEL_MAX: Final[int] = 32  # Max model label length in summary tables/charts
SUMMARY_CHART_MAX_ROWS: Final[int] = 8  # Max rows shown in summary charts
MIN_MODELS_FOR_EFFICIENCY_CHART: Final[int] = 2  # Min successful rows for cross-model efficiency
FLOAT_ZERO_EPSILON: Final[float] = 1e-9  # Tolerance when rendering signed deltas as zero
UTILITY_DELTA_NEUTRAL_BAND: Final[float] = 2.0  # Within ±band, model is neutral vs metadata

# Numeric fields are automatically derived from FIELD_ABBREVIATIONS for consistency
# Exclude non-numeric fields explicitly
NUMERIC_FIELD_PATTERNS: Final[frozenset[str]] = frozenset(
    k for k in FIELD_ABBREVIATIONS if k not in {"model_name", "quality_issues"}
)

# Performance timing fields: those from PerformanceResult (not GenerationResult)
# Automatically derived from FIELD_ABBREVIATIONS for consistency
PERFORMANCE_TIMING_FIELDS: Final[list[str]] = [
    field
    for field in FIELD_ABBREVIATIONS
    if field
    in {"generation_time", "model_load_time", "total_time", "quality_issues", "error_package"}
]


# =============================================================================
# FORMATTING UTILITIES (Numbers, Memory, Time, Tokens/sec, Field Values)
# =============================================================================


def fmt_num(val: float | str) -> str:
    """Format numbers consistently with thousands separators across all output formats."""
    try:
        fval = float(val)
        if abs(fval) >= FORMATTING.large_number:
            return f"{fval:,.0f}"
        # For integers or whole numbers, use comma separator if >= thousand_separator
        if fval == int(fval) and abs(fval) >= FORMATTING.thousand_separator:
            return f"{int(fval):,}"
        if abs(fval) > 0:
            return f"{fval:.3g}"
        return str(val)
    except (ValueError, TypeError, OverflowError):
        return str(val)


def format_field_label(field_name: str) -> str:
    """Return a human-friendly label for a metric field name."""
    return field_name.replace("_", " ").title()


# =============================================================================
# TEXT ESCAPING - Unified strategy for HTML/Markdown escaping
# =============================================================================


class HTMLSelectiveEscaper:
    """Selective HTML escaping preserving GitHub-safe tags.

    Escapes potentially unsafe HTML while preserving common formatting
    tags that GitHub Markdown recognizes (br, b, strong, i, em, code).
    Does NOT preserve 's' tag to avoid interpreting <s> tokens from
    model output as strikethrough.
    """

    allowed_tags: frozenset[str] = frozenset({"br", "b", "strong", "i", "em", "code"})
    tag_pattern: re.Pattern[str] = re.compile(r"</?[A-Za-z][A-Za-z0-9:-]*(?:\s+[^<>]*?)?>")

    def escape(self, text: str) -> str:
        """Escape tags except allowed safe tags."""

        def _escape_html_like(m: re.Match[str]) -> str:
            token = m.group(0)
            inner = token[1:-1].strip()
            if not inner:
                return token.replace("<", "&lt;").replace(">", "&gt;")
            core = inner.lstrip("/").split(None, 1)[0].rstrip("/").lower()
            if core in self.allowed_tags:
                return token  # Keep recognized safe tag
            return token.replace("<", "&lt;").replace(">", "&gt;")

        return self.tag_pattern.sub(_escape_html_like, text)


class MarkdownPipeEscaper:
    """Markdown escaping for table pipe characters and formatting.

    Handles pipe character escaping and converts newlines to <br> tags
    to prevent breaking Markdown table formatting. Preserves model-generated
    markdown like **bold**, *italic*, `code` while preventing table breakage.
    """

    def escape(self, text: str) -> str:
        """Escape text for safe inclusion in Markdown tables.

        Converts newlines to <br> tags, wraps bare URLs, escapes pipes,
        and neutralizes HTML-like tags while preserving markdown formatting.
        """
        # First, convert newlines to HTML <br> tags to preserve line structure
        # Handle different newline formats consistently
        result = text.replace("\r\n", "<br>").replace("\r", "<br>").replace("\n", "<br>")

        # Clean up multiple consecutive <br> tags and normalize spacing
        result = re.sub(r"(<br>\s*){2,}", "<br><br>", result)  # Max 2 consecutive breaks
        result = re.sub(r"\s+", " ", result).strip()  # Normalize other whitespace

        # Wrap bare URLs in angle brackets (MD034 compliance)
        result = _wrap_bare_urls(result)

        # Escape pipe characters (CRITICAL: breaks table structure)
        result = result.replace("|", "\\|")

        # Neutralize HTML-like tags using selective escaper
        result = HTML_ESCAPER.escape(result)

        # Prevent markdownlint MD050 (strong style) on double underscores in error messages/outputs
        result = result.replace("__", r"\_\_")

        # Escape bare ampersands that could start entities
        return re.sub(r"&(?!lt;|gt;|amp;|#)", "&amp;", result)


class DiagnosticsEscaper:
    """Markdown escaping optimized for error/diagnostic messages.

    Similar to MarkdownPipeEscaper but allows more consecutive line breaks
    to preserve traceback formatting while still preventing table breakage.
    """

    def escape(self, text: str) -> str:
        """Escape diagnostics text for Markdown tables.

        More lenient than standard escaping to preserve error message formatting.
        """
        # Convert newlines to <br> but preserve more spacing for tracebacks
        result = text.replace("\r\n", "<br>").replace("\r", "<br>").replace("\n", "<br>")

        # Limit excessive consecutive <br> (allow 3 for traceback readability)
        result = re.sub(r"(<br>\s*){3,}", "<br><br>", result)

        # Wrap bare URLs in angle brackets
        result = _wrap_bare_urls(result)

        # Escape pipes (critical for table structure)
        result = result.replace("|", "\\|")

        # Neutralize HTML-like tags
        result = HTML_ESCAPER.escape(result)

        # Prevent markdownlint MD050 (strong style) on double underscores in error messages
        result = result.replace("__", r"\_\_")

        # Escape bare ampersands
        return re.sub(r"&(?!lt;|gt;|amp;|#)", "&amp;", result)


# Instantiate default escapers for runtime use
HTML_ESCAPER = HTMLSelectiveEscaper()
MARKDOWN_ESCAPER = MarkdownPipeEscaper()
DIAGNOSTICS_ESCAPER = DiagnosticsEscaper()


def _format_memory_value_gb(num: float) -> str:
    """Format mixed-source memory value as GB string.

    Accepts raw bytes (mlx) or decimal GB (mlx-vlm). Returns a string without unit.
    """
    if num <= 0:
        return "0"
    gb: float = (num / DECIMAL_GB) if num > MEM_BYTES_TO_GB_THRESHOLD else num
    if gb >= FORMATTING.memory_gb_integer:
        return f"{gb:,.0f}"
    if gb >= 1:
        return f"{gb:,.1f}"
    return f"{gb:.2f}"


def _format_time_seconds(num: float) -> str:
    """Format seconds with two decimals and trailing 's'."""
    return f"{num:.2f}s"


def _format_tps(num: float) -> str:
    """Format tokens-per-second with adaptive precision."""
    if abs(num) >= FORMATTING.large_number:
        return f"{num:,.0f}"
    if abs(num) >= FORMATTING.medium_number:
        return f"{num:.1f}"
    return f"{num:.3g}"


def _format_hms(total_seconds: float) -> str:
    """Return HH:MM:SS string for durations >= 1 hour.

    Seconds are floored for the human-friendly component; fractional part is
    still preserved in the separate seconds display when shown.
    """
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_overall_runtime(total_seconds: float) -> str:
    """Format runtime with a short/long display mode.

    Uses seconds-only below the hour threshold. For long runs, prefixes with
    ``HH:MM:SS`` while preserving precise seconds in parentheses.
    """
    if total_seconds >= FORMATTING.hour_threshold_seconds:
        return f"{_format_hms(total_seconds)} ({total_seconds:.2f}s)"
    return f"{total_seconds:.2f}s"


def _detect_repetitive_output(text: str, threshold: float | None = None) -> tuple[bool, str | None]:
    """Detect if generated text is highly repetitive (tokens or phrases).

    Checks for:
    1. Single token repetition (e.g., "<s> <s> <s>")
    2. Phrase repetition (e.g., "The scene is very visible. The scene is very visible.")

    Args:
        text: Generated text to check
        threshold: Fraction of text that must be repetitive to flag
            (default from QUALITY.repetition_ratio)

    Returns:
        Tuple of (is_repetitive, repeated_pattern)
    """
    if threshold is None:
        threshold = QUALITY.repetition_ratio

    if not text or len(text) < QUALITY.min_text_length:
        return False, None

    # 1. Check for single token repetition
    tokens = text.split()
    if len(tokens) < QUALITY.min_token_count:
        return False, None

    token_counts = Counter(tokens)
    if token_counts:
        most_common_token, count = token_counts.most_common(1)[0]
        if count / len(tokens) >= threshold:
            return True, most_common_token

    # 2. Check for phrase repetition (n-grams)
    # Look for repeated sequences of configurable length (default 4+ tokens)
    min_phrase_len = QUALITY.min_phrase_length
    # We only care if the phrase appears multiple times and covers a significant portion of text

    # Simple heuristic: Check if any substring of length N repeats significantly
    # For efficiency, we'll check specific n-gram sizes
    text_lower = text.lower()
    words = text_lower.split()
    n_words = len(words)

    if n_words < min_phrase_len * 2:
        return False, None

    # Check n-grams up to configured maximum
    for n in range(min_phrase_len, min(QUALITY.max_phrase_length + 1, n_words // 2)):
        ngrams = [" ".join(words[i : i + n]) for i in range(n_words - n + 1)]
        if not ngrams:
            continue

        ngram_counts = Counter(ngrams)
        most_common_ngram, count = ngram_counts.most_common(1)[0]

        # If a phrase repeats more than configured threshold and covers significant portion
        # Or if it repeats excessively regardless of coverage
        if count > QUALITY.max_phrase_repetitions or (
            count > QUALITY.min_phrase_repetitions
            and (count * n) / n_words > QUALITY.phrase_coverage_threshold
        ):
            return True, f'phrase: "{most_common_ngram[:30]}..."'

    return False, None


def _detect_hallucination_patterns(text: str) -> list[str]:
    """Detect quiz/table artifacts that indicate task drift or hallucination.

    Flags markdown tables, multiple-choice answer formats, quiz-style prompts,
    and unrelated educational-content terms in otherwise descriptive output.
    """
    issues: list[str] = []

    if not text:
        return issues

    text_lower: str = text.lower()

    # Check for markdown tables (pipe-delimited)
    if "|" in text and text.count("|") >= QUALITY.min_pipes_for_table:
        # Likely a table if we see multiple pipes
        lines_with_pipes: list[str] = [line for line in text.split("\n") if "|" in line]
        if len(lines_with_pipes) >= QUALITY.min_table_rows:
            issues.append("Contains unexpected table")

    # Check for multiple choice patterns
    mc_pattern: re.Pattern[str] = re.compile(r"^[A-D]\)", re.MULTILINE)
    mc_matches: list[str] = mc_pattern.findall(text)
    if len(mc_matches) >= QUALITY.min_mc_answers:
        issues.append("Contains multiple choice pattern")

    # Check for quiz/test questions
    question_indicators: list[str] = _get_quality_pattern_list(
        "hallucination_question_indicators",
        ["what is", "how many", "based on the chart", "calculate"],
    )
    has_question: bool = any(indicator in text_lower for indicator in question_indicators)
    if has_question and len(text) > QUALITY.substantial_text_length:
        issues.append("Contains question/quiz content")

    # Check for unrelated educational content keywords
    edu_keywords: list[str] = _get_quality_pattern_list(
        "hallucination_edu_keywords",
        ["grade level", "students with adhd", "test scores", "homework"],
    )
    if any(keyword in text_lower for keyword in edu_keywords):
        issues.append("Contains unrelated educational content")

    return issues


def _detect_excessive_verbosity(text: str, generated_tokens: int) -> bool:
    """Detect if model output is excessively verbose.

    Considers output verbose if:
    - Generated tokens > 300 (substantial length)
    - Contains meta-commentary about the image/analysis
    - Has multiple sections (###, ##) suggesting over-structure

    Args:
        text: Generated text to check
        generated_tokens: Number of tokens generated

    Returns:
        True if output appears excessively verbose
    """
    if generated_tokens < QUALITY.max_verbosity_tokens:
        return False

    text_lower: str = text.lower()

    # Check for meta-commentary patterns
    meta_patterns: list[str] = _get_quality_pattern_list(
        "meta_commentary",
        [
            "the image depicts",
            "the image shows",
            "the photograph captures",
            "this image features",
            "in conclusion",
            "### analysis",
            "### conclusion",
            "based on the image",
        ],
    )

    meta_count: int = sum(1 for pattern in meta_patterns if pattern in text_lower)

    # Check for excessive sectioning
    section_headers: int = text.count("###") + text.count("## ")

    # Verbose if has meta-commentary + sections or just too many sections
    return meta_count >= QUALITY.min_meta_patterns or section_headers >= QUALITY.min_section_headers


def _detect_formatting_violations(text: str) -> list[str]:
    """Detect rendering-related formatting artifacts in generated output.

    Checks only HTML-like tags and markdown header overuse. Bullet density is
    handled separately by ``_detect_excessive_bullets`` so callers can treat
    list-heavy outputs differently from malformed formatting.

    Args:
        text: Generated text to check

    Returns:
        List of display-affecting formatting issues, excluding bullet counts
    """
    issues: list[str] = []

    if not text:
        return issues

    # Check for tags (beyond simple breaks) that may interfere with rendering
    html_tags: list[str] = re.findall(r"<(?!br>|/br>)[a-z]+[^>]*>", text, re.IGNORECASE)
    if html_tags:
        # Keep raw tags in analysis output; renderers escape them as needed for Markdown/HTML.
        tags_preview: str = ", ".join(dict.fromkeys(html_tags[: FORMATTING.max_preview_tags]))
        issues.append(f"Unknown tags: {tags_preview}")

    # Check for excessive markdown structure
    header_count: int = text.count("\n##") + text.count("\n###")
    if header_count > QUALITY.max_markdown_headers:
        issues.append(f"Excessive markdown headers ({header_count})")

    return issues


def _truncate_repetitive_output(text: str) -> str:
    """Truncate outputs with excessive token repetition for display.

    When a model produces many consecutive repetitions of the same token,
    truncate for readability while indicating the total count.

    Args:
        text: Generated text

    Returns:
        Truncated text with repetition summary if applicable
    """
    if not text:
        return text

    # Quick inline check for repetition
    is_repetitive, repeated_token = _detect_repetitive_output(text)
    if not is_repetitive or not repeated_token:
        return text

    # Count consecutive repetitions of the token (with optional whitespace between)
    pattern: str = re.escape(repeated_token)
    min_reps = FORMATTING.min_consecutive_repetitions_for_truncation
    match: re.Match[str] | None = re.search(rf"({pattern}(?:\s*{pattern}){{{min_reps},}})", text)

    if match:
        # Count total repetitions in the matched section
        repetitions: int = match.group(0).count(repeated_token)
        # Show first few occurrences + count + ellipsis
        truncated_section: str = (
            f"{repeated_token} {repeated_token} {repeated_token} "
            f"... [{repetitions} total repetitions] ..."
        )
        return text.replace(match.group(0), truncated_section)

    return text


def _detect_excessive_bullets(text: str) -> tuple[bool, int]:
    """Flag outputs whose bullet count exceeds the configured threshold.

    This is a pure count-based check. Callers decide whether a dense bullet
    list is acceptable for the current prompt.

    Args:
        text: Generated text to check

    Returns:
        Tuple of (has_excessive_bullets, bullet_count)
    """
    if not text:
        return False, 0

    bullet_prefixes: tuple[str, str, str] = ("- ", "* ", "• ")
    bullet_lines: list[str] = [
        line for line in text.split("\n") if line.strip().startswith(bullet_prefixes)
    ]
    bullet_count: int = len(bullet_lines)

    # Use config threshold if available, otherwise default to 15 (lowered for cataloging)
    threshold: int = QUALITY.max_bullets or 15
    return bullet_count > threshold, bullet_count


CONTEXT_NOISE_TERMS: Final[frozenset[str]] = frozenset(
    {
        "capture",
        "cataloguing",
        "cataloging",
        "context",
        "description",
        "existing",
        "hint",
        "hints",
        "image",
        "keyword",
        "keywords",
        "local",
        "metadata",
        "photo",
        "picture",
        "taken",
        "title",
        "trusted",
        "visual",
    },
)

CONTEXT_COMMON_WORDS: Final[frozenset[str]] = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "about",
        "as",
        "this",
        "that",
        "these",
        "those",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "should",
        "could",
        "may",
        "might",
        "must",
        "can",
    }.union(CONTEXT_NOISE_TERMS),
)

CONTEXT_TERM_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "uk": ("united kingdom", "u k", "u.k."),
    "united kingdom": ("uk", "u k", "u.k."),
    "usa": ("united states", "u s a", "u.s.a."),
    "united states": ("usa", "u s a", "u.s.a."),
}

PROMPT_ECHO_MARKERS: Final[tuple[str, ...]] = (
    "return exactly these three sections",
    "do not output reasoning",
    "do not copy context hints verbatim",
    "context: existing metadata hints",
    "title hint:",
    "description hint:",
    "keyword hints:",
    "capture metadata:",
)

NONVISUAL_CONTEXT_TERMS: Final[frozenset[str]] = frozenset(
    {
        "adobe stock",
        "any vision",
        "borough",
        "center",
        "centre",
        "country",
        "county",
        "district",
        "england",
        "europe",
        "gps",
        "hertfordshire",
        "locations",
        "municipality",
        "parish",
        "province",
        "region",
        "source",
        "state",
        "stock",
        "taken",
        "time",
        "timestamp",
        "town",
        "uk",
        "united kingdom",
        "village",
        "welwyn garden city",
    },
)

NONVISUAL_CONTEXT_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\b(?:gps|lat(?:itude)?|lon(?:gitude)?|coordinate)s?\b", re.IGNORECASE),
    re.compile(
        r"\b(?:taken on|capture metadata|capture time|timestamp|local time)\b", re.IGNORECASE
    ),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b"),
    re.compile(r"\b\d+(?:\.\d+)?°[nswe]\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class TrustedHintBundle:
    """Split prompt context into reusable trusted hints and nonvisual metadata."""

    trusted_text: str = ""
    trusted_terms: tuple[str, ...] = ()
    nonvisual_terms: tuple[str, ...] = ()


def _normalize_phrase_for_matching(text: str) -> str:
    """Normalize free-form text for alias and overlap matching."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.casefold())).strip()


def _extract_prompt_context_text(prompt: str, context_marker: str = "Context:") -> str:
    """Extract prompt context text anchored at ``context_marker``.

    Supports inline-marker context (``Context: ...``) and multi-line bullet
    context blocks. Stops on the first blank line after context content starts.
    """
    if not prompt:
        return ""

    marker_lower: str = context_marker.casefold()
    lines: list[str] = prompt.splitlines()

    for idx, raw_line in enumerate(lines):
        line: str = raw_line.strip()
        if not line.casefold().startswith(marker_lower):
            continue

        extracted: list[str] = []
        inline_remainder: str = (
            line[len(context_marker) :].strip() if len(line) >= len(context_marker) else ""
        )
        if inline_remainder:
            extracted.append(inline_remainder.lstrip("-").strip())

        for follow in lines[idx + 1 :]:
            stripped: str = follow.strip()
            if not stripped:
                if extracted:
                    break
                continue
            # Context bullets are expected; strip bullet prefix but retain content.
            if stripped.startswith(("-", "*", "•")):
                extracted.append(stripped.lstrip("-*• ").strip())
                continue
            # Treat non-bulleted lines as context continuation until first blank line.
            extracted.append(stripped)

        return "\n".join(part for part in extracted if part).strip()

    return ""


def _context_term_present(term: str, normalized_text: str) -> bool:
    """Return ``True`` when a context term (or alias) appears in normalized text."""
    canonical: str = _normalize_phrase_for_matching(term)
    if not canonical:
        return False

    variants: set[str] = {canonical}
    variants.update(
        _normalize_phrase_for_matching(v) for v in CONTEXT_TERM_ALIASES.get(canonical, ())
    )
    for variant in variants:
        if not variant:
            continue
        if re.search(rf"\b{re.escape(variant)}\b", normalized_text):
            return True
    return False


def _split_catalog_keywords(raw_keywords: str) -> list[str]:
    """Split keyword section text into normalized keyword terms."""
    if not raw_keywords:
        return []
    keywords: list[str] = []
    for item in re.split(r"[,\n]", raw_keywords):
        cleaned: str = re.sub(r"\s+", " ", item).strip(" -*•\t")
        if cleaned:
            keywords.append(cleaned)
    return keywords


def _is_nonvisual_context_term(term: str) -> bool:
    """Return True for metadata/source/location terms we should not require visually."""
    normalized = _normalize_phrase_for_matching(term)
    if not normalized:
        return False
    if normalized in NONVISUAL_CONTEXT_TERMS:
        return True
    # Check config-loaded nonvisual location terms
    if QUALITY.patterns:
        config_terms = QUALITY.patterns.get("nonvisual_location_terms")
        if config_terms and normalized in {t.lower() for t in config_terms}:
            return True
    return any(pattern.search(term) for pattern in NONVISUAL_CONTEXT_PATTERNS)


def _extract_hint_signal_terms(text: str) -> list[str]:
    """Extract simple reusable context terms from trusted hint text."""
    if not text:
        return []
    terms: list[str] = []
    for raw_term in re.findall(r"[A-Za-z0-9']+", text):
        normalized = _normalize_phrase_for_matching(raw_term)
        if (
            not normalized
            or len(normalized) < QUALITY.min_context_term_length
            or normalized in CONTEXT_COMMON_WORDS
        ):
            continue
        terms.append(raw_term)
    return _dedupe_preserve_order(terms)


def _strip_nonvisual_terms_from_text(text: str, nonvisual_terms: Sequence[str]) -> str:
    """Remove explicitly nonvisual metadata terms from trusted hint prose."""
    cleaned = text
    normalized_term_set: set[str] = {
        _normalize_phrase_for_matching(item) for item in nonvisual_terms
    }
    normalized_terms: list[str] = [term for term in normalized_term_set if term]
    normalized_terms.sort(key=len, reverse=True)
    for term in normalized_terms:
        pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
        cleaned = pattern.sub(" ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip(" ,.;:-")


def _parse_context_hint_lines(
    context_text: str,
) -> tuple[str, str, str, list[str]]:
    """Parse prompt context lines into title/description/keywords/nonvisual buckets."""
    title_hint = ""
    description_hint = ""
    keyword_hint_text = ""
    nonvisual_lines: list[str] = []
    prefix_targets: tuple[tuple[str, str], ...] = (
        ("title hint:", "title"),
        ("description hint:", "description"),
        ("keyword hints:", "keywords"),
        ("capture metadata:", "nonvisual"),
    )

    for raw_line in context_text.splitlines():
        stripped = raw_line.strip().lstrip("-").strip()
        if not stripped:
            continue
        lowered = stripped.casefold()
        for prefix, target in prefix_targets:
            if not lowered.startswith(prefix):
                continue
            value = stripped[len(prefix) :].strip()
            if target == "title":
                title_hint = value
            elif target == "description":
                description_hint = value
            elif target == "keywords":
                keyword_hint_text = value
            else:
                nonvisual_lines.append(value)
            break

    return title_hint, description_hint, keyword_hint_text, nonvisual_lines


def _partition_hint_terms(terms: Iterable[str]) -> tuple[list[str], list[str]]:
    """Split hint terms into trusted and nonvisual groups."""
    trusted_terms: list[str] = []
    nonvisual_terms: list[str] = []
    for term in terms:
        if _is_nonvisual_context_term(term):
            nonvisual_terms.append(term)
        else:
            trusted_terms.append(term)
    return trusted_terms, nonvisual_terms


def _extract_trusted_hint_bundle(
    prompt: str,
    *,
    context_marker: str = "Context:",
) -> TrustedHintBundle:
    """Parse prompt context into trusted hints and explicitly nonvisual metadata."""
    context_text = _extract_prompt_context_text(prompt, context_marker=context_marker)
    if not context_text:
        return TrustedHintBundle()

    title_hint, description_hint, keyword_hint_text, nonvisual_lines = _parse_context_hint_lines(
        context_text,
    )

    trusted_keyword_terms = _split_catalog_keywords(keyword_hint_text)
    trusted_terms, nonvisual_terms = _partition_hint_terms(trusted_keyword_terms)

    for free_text in (title_hint, description_hint):
        hint_terms, metadata_terms = _partition_hint_terms(_extract_hint_signal_terms(free_text))
        trusted_terms.extend(hint_terms)
        nonvisual_terms.extend(metadata_terms)

    for line in nonvisual_lines:
        nonvisual_terms.extend(_extract_hint_signal_terms(line))
        nonvisual_terms.append(line)

    filtered_title = _strip_nonvisual_terms_from_text(title_hint, nonvisual_terms)
    filtered_description = _strip_nonvisual_terms_from_text(description_hint, nonvisual_terms)
    filtered_keywords = [
        term for term in trusted_keyword_terms if not _is_nonvisual_context_term(term)
    ]

    trusted_parts: list[str] = []
    if filtered_title:
        trusted_parts.append(f"Title: {filtered_title}")
    if filtered_description:
        trusted_parts.append(f"Description: {filtered_description}")
    if filtered_keywords:
        trusted_parts.append("Keywords: " + ", ".join(filtered_keywords))

    return TrustedHintBundle(
        trusted_text="\n".join(trusted_parts).strip(),
        trusted_terms=tuple(_dedupe_preserve_order(trusted_terms)),
        nonvisual_terms=tuple(_dedupe_preserve_order(nonvisual_terms)),
    )


def _count_factual_sentences(text: str) -> int:
    """Count sentence-like units in description text with fallback for punctuation-free text."""
    cleaned: str = text.strip()
    if not cleaned:
        return 0
    sentence_like: list[str] = [
        s for s in re.split(r"(?<=[.!?])\s+|\n+", cleaned) if re.search(r"\w", s)
    ]
    if sentence_like:
        if len(sentence_like) == 1 and not re.search(r"[.!?]", cleaned):
            return 1
        return len(sentence_like)
    return 1


def _extract_catalog_sections(text: str) -> dict[str, str]:
    """Extract ``Title/Description/Keywords`` sections from model output."""
    if not text:
        return {}

    pattern: re.Pattern[str] = re.compile(
        r"(?im)^\s*(?:[#>*-]\s*)?(?:\*{0,2})\s*(title|description|keywords)(?:\*{0,2})\s*:\s*(.*)$",
    )
    matches: list[re.Match[str]] = list(pattern.finditer(text))
    if not matches:
        return {}

    sections: dict[str, str] = {}
    for idx, match in enumerate(matches):
        label: str = match.group(1).casefold()
        if label in sections:
            continue
        section_line: str = match.group(2).strip()
        next_start: int = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        trailing: str = text[match.end() : next_start].strip()
        combined: str = "\n".join(part for part in (section_line, trailing) if part).strip()
        sections[label] = combined
    return sections


def _prompt_requests_catalog_contract(prompt: str) -> bool:
    """Return True when prompt clearly requests strict Title/Description/Keywords output."""
    prompt_lower: str = prompt.casefold()
    has_required_labels: bool = all(
        label in prompt_lower for label in ("title:", "description:", "keywords:")
    )
    return has_required_labels and (
        "return exactly these three sections" in prompt_lower or "catalog" in prompt_lower
    )


def _analyze_catalog_contract(
    text: str,
) -> tuple[list[str], int | None, int | None, int | None, float | None]:
    """Evaluate strict cataloging contract compliance from generated text."""
    sections: dict[str, str] = _extract_catalog_sections(text)
    missing_sections: list[str] = [
        section
        for section in ("title", "description", "keywords")
        if not sections.get(section, "").strip()
    ]

    title_word_count = None
    description_sentence_count = None
    keyword_count = None
    keyword_dup_ratio = None

    title_text: str = sections.get("title", "")
    if title_text:
        title_word_count = len(re.findall(r"[A-Za-z0-9']+", title_text))

    description_text: str = sections.get("description", "")
    if description_text:
        description_sentence_count = _count_factual_sentences(description_text)

    keywords_text: str = sections.get("keywords", "")
    if keywords_text:
        keyword_terms: list[str] = _split_catalog_keywords(keywords_text)
        keyword_count = len(keyword_terms)
        if keyword_count >= QUALITY.min_keywords_for_duplication_check and keyword_count > 0:
            normalized: list[str] = [_normalize_phrase_for_matching(term) for term in keyword_terms]
            normalized = [term for term in normalized if term]
            if normalized:
                keyword_dup_ratio = 1.0 - (len(set(normalized)) / len(normalized))

    return (
        missing_sections,
        title_word_count,
        description_sentence_count,
        keyword_count,
        keyword_dup_ratio,
    )


def _detect_reasoning_leakage(text: str) -> tuple[bool, list[str]]:
    """Detect chain-of-thought or prompt-echo leakage in generated output."""
    if not text:
        return False, []

    text_lower: str = text.casefold()
    markers: list[str] = _get_quality_pattern_list(
        "reasoning_leak_markers",
        [
            "<think>",
            "◁think▷",
            "◁/think▷",
            "here are my reasoning steps",
            "the user asks:",
            "let's analyze the image",
        ],
    )

    findings: list[str] = [
        marker for marker in markers if marker and marker.casefold() in text_lower
    ]
    findings.extend(marker for marker in PROMPT_ECHO_MARKERS if marker in text_lower)
    deduped: list[str] = _dedupe_preserve_order(findings)
    return bool(deduped), deduped[:4]


def _detect_instruction_echo(text: str) -> tuple[bool, list[str]]:
    """Detect direct reuse of prompt/task instructions in the answer."""
    if not text:
        return False, []
    text_lower = text.casefold()
    findings = [marker for marker in PROMPT_ECHO_MARKERS if marker in text_lower]
    deduped = _dedupe_preserve_order(findings)
    return bool(deduped), deduped[:4]


def _detect_context_echo(
    text: str,
    prompt: str,
    context_marker: str = "Context:",
) -> tuple[bool, float]:
    """Detect likely context regurgitation rather than image-grounded synthesis."""
    has_echo: bool = False
    score: float = 0.0

    if not text or not prompt:
        return has_echo, score

    text_lower: str = text.casefold()
    has_inline_context_block: bool = "context:" in text_lower and any(
        marker in text_lower for marker in ("title hint:", "description hint:", "capture metadata:")
    )
    if has_inline_context_block:
        return True, 1.0

    context_text: str = _extract_trusted_hint_bundle(
        prompt,
        context_marker=context_marker,
    ).trusted_text
    if context_text:
        output_words: list[str] = re.findall(r"[a-z0-9']+", text_lower)
        context_words: list[str] = re.findall(r"[a-z0-9']+", context_text.casefold())
        if len(output_words) >= QUALITY.context_echo_min_words and context_words:
            output_vocab: set[str] = set(output_words)
            context_vocab: set[str] = set(context_words)
            vocab_overlap: float = len(output_vocab & context_vocab) / max(len(output_vocab), 1)
            score = round(vocab_overlap, 3)
            if vocab_overlap >= QUALITY.context_echo_vocab_ratio_threshold:
                has_echo = True
            else:
                ngram_size: int = max(2, QUALITY.context_echo_ngram_size)
                if len(output_words) >= ngram_size and len(context_words) >= ngram_size:
                    output_ngrams: set[tuple[str, ...]] = {
                        tuple(output_words[idx : idx + ngram_size])
                        for idx in range(len(output_words) - ngram_size + 1)
                    }
                    context_ngrams: set[tuple[str, ...]] = {
                        tuple(context_words[idx : idx + ngram_size])
                        for idx in range(len(context_words) - ngram_size + 1)
                    }
                    shared_ngrams: set[tuple[str, ...]] = output_ngrams & context_ngrams
                    if shared_ngrams:
                        shared_ratio: float = len(shared_ngrams) / max(len(output_ngrams), 1)
                        score = round(shared_ratio, 3)
                        has_echo = (
                            len(shared_ngrams) >= QUALITY.context_echo_min_shared_ngrams
                            and shared_ratio >= QUALITY.context_echo_ngram_ratio_threshold
                        )

    return has_echo, score


def _detect_context_ignorance(
    text: str,
    prompt: str,
    context_marker: str = "Context:",
) -> tuple[bool, list[str]]:
    """Detect when trusted hint content is mostly absent from the answer."""
    if not text or not prompt:
        return False, []

    bundle = _extract_trusted_hint_bundle(prompt, context_marker=context_marker)
    if not bundle.trusted_terms:
        return False, []

    normalized_text: str = _normalize_phrase_for_matching(text)
    missing_terms: list[str] = [
        term for term in bundle.trusted_terms if not _context_term_present(term, normalized_text)
    ]
    is_ignored: bool = (
        len(missing_terms) > 0
        and len(bundle.trusted_terms) >= QUALITY.min_key_terms_threshold
        and len(missing_terms) >= len(bundle.trusted_terms) * QUALITY.min_missing_ratio
    )

    return is_ignored, missing_terms[: QUALITY.max_reported_missing_terms]


def _detect_metadata_borrowing(
    text: str,
    bundle: TrustedHintBundle,
) -> tuple[bool, list[str]]:
    """Detect output reuse of explicitly nonvisual metadata from the prompt."""
    if not text or not bundle.nonvisual_terms:
        return False, []
    normalized_text = _normalize_phrase_for_matching(text)
    matches = [
        term for term in bundle.nonvisual_terms if _context_term_present(term, normalized_text)
    ]
    deduped = _dedupe_preserve_order(matches)
    return bool(deduped), deduped[: QUALITY.max_reported_missing_terms]


def _estimate_prompt_tokens_from_text(prompt: str | None) -> int | None:
    """Estimate text-only prompt tokens with a lightweight word-based heuristic."""
    if not prompt:
        return None
    words = len(prompt.split())
    if words == 0:
        return None
    return max(math.ceil(words * QUALITY.prompt_word_to_token_ratio), 1)


def _detect_likely_cutoff(
    text: str,
    *,
    generated_tokens: int,
    requested_max_tokens: int | None,
    is_repetitive: bool,
    missing_sections: Sequence[str],
) -> tuple[bool, list[str]]:
    """Detect likely early termination at the generation cap.

    Returns ``(hit_cap, reasons)`` where *reasons* may include degradation
    evidence such as ``"missing_sections"`` or ``"repetitive_tail"``.  A hit
    with no degradation reasons means the model produced structurally sound
    output that merely ran out of budget.
    """
    if requested_max_tokens is None or requested_max_tokens <= 0:
        return False, []
    if generated_tokens < requested_max_tokens:
        return False, []

    tail = text.strip()[-QUALITY.cutoff_tail_chars :].strip()
    reasons: list[str] = []
    if missing_sections and text.strip():
        reasons.append("missing_sections")
    if is_repetitive:
        reasons.append("repetitive_tail")
    if tail:
        if re.search(r"(title|description|keywords)\s*:?\s*$", tail, re.IGNORECASE):
            reasons.append("unfinished_section")
        if tail[-1].isalnum() and not re.search(r"[.!?\"')\]]\s*$", tail):
            reasons.append("abrupt_tail")
    return True, _dedupe_preserve_order(reasons)


def _classify_hint_relationship(
    text: str,
    bundle: TrustedHintBundle,
) -> tuple[str, list[str]]:
    """Classify how the answer relates to trusted prompt hints."""
    if not bundle.trusted_text:
        return "preserves_trusted_hints", []

    text_for_hint_eval = _strip_nonvisual_terms_from_text(text, bundle.nonvisual_terms) or text
    normalized_text = _normalize_phrase_for_matching(text_for_hint_eval)
    visual_terms = [t for t in bundle.trusted_terms if not _is_nonvisual_context_term(t)]
    visual_matched = [t for t in visual_terms if _context_term_present(t, normalized_text)]
    overlap_ratio = len(visual_matched) / max(len(visual_terms), 1) if visual_terms else 0.0
    baseline = compute_cataloging_utility(bundle.trusted_text, None)
    baseline_score = float(baseline["utility_score"])
    utility = compute_cataloging_utility(text_for_hint_eval, bundle.trusted_text)
    score = float(utility["utility_score"])
    delta = score - baseline_score

    if visual_terms and overlap_ratio < (1.0 - QUALITY.min_missing_ratio):
        return "ignores_trusted_hints", ["low_hint_overlap"]
    if delta > UTILITY_DELTA_NEUTRAL_BAND:
        return "improves_trusted_hints", ["utility_delta_positive"]
    if overlap_ratio >= QUALITY.medium_confidence_threshold:
        return "preserves_trusted_hints", ["trusted_overlap"]
    if delta < -UTILITY_DELTA_NEUTRAL_BAND:
        return "degrades_trusted_hints", ["utility_delta_negative"]
    return "preserves_trusted_hints", ["utility_delta_neutral"]


def _classify_review_owner(
    *,
    harness_type: str | None,
    failure_owner: str | None,
) -> str:
    """Return a compact single-owner label for canonical review output."""
    if failure_owner:
        owner_lower = failure_owner.casefold()
        for needle, owner in _REVIEW_OWNER_BY_FAILURE_NEEDLE:
            if needle in owner_lower:
                return owner

    harness_key = (harness_type or "").casefold()
    mapped_owner = _REVIEW_OWNER_BY_HARNESS_TYPE.get(harness_key)
    if mapped_owner is not None:
        return mapped_owner
    return "model"


def _classify_review_verdict(
    *,
    has_harness_issue: bool,
    harness_type: str | None,
    likely_cutoff: bool,
    cutoff_reasons: Sequence[str] = (),
    prompt_tokens_total: int | None,
    prompt_tokens_text_est: int | None,
    prompt_tokens_nontext_est: int | None,
    missing_sections: Sequence[str],
    utility_grade: str,
    instruction_echo: bool,
    metadata_borrowing: bool,
    has_hallucination: bool,
) -> tuple[str, list[str]]:
    """Return ordered verdict plus compact evidence labels.

    Token-cap hits are split into two verdicts:

    * ``"token_cap"`` — the model hit max_tokens but produced structurally
      sound output (no degradation evidence such as missing sections,
      repetitive tails, or abrupt stops).
    * ``"cutoff_degraded"`` — the model hit max_tokens **and** the output
      shows quality degradation.
    """
    evidence: list[str] = []
    if has_harness_issue and (harness_type or "") != "long_context":
        if harness_type:
            evidence.append(f"harness:{harness_type}")
        return "harness", evidence

    if likely_cutoff:
        evidence.append("token_cap")
        # Distinguish benign cap from degraded cap
        _degradation_reasons = {
            "missing_sections",
            "repetitive_tail",
            "unfinished_section",
            "abrupt_tail",
        }
        if set(cutoff_reasons) & _degradation_reasons:
            evidence.extend(cutoff_reasons)
            return "cutoff_degraded", evidence
        return "token_cap", evidence

    weak_output = (
        bool(missing_sections) or instruction_echo or metadata_borrowing or has_hallucination
    )
    heavy_nontext_burden = (
        prompt_tokens_total is not None
        and prompt_tokens_total >= QUALITY.long_prompt_tokens_threshold
        and prompt_tokens_text_est is not None
        and prompt_tokens_nontext_est is not None
        and prompt_tokens_nontext_est > prompt_tokens_text_est
    )
    if (harness_type or "") == "long_context" or (heavy_nontext_burden and weak_output):
        if (harness_type or "") == "long_context":
            evidence.append("long_context")
        if heavy_nontext_burden:
            evidence.append("nontext_prompt_burden")
        return "context_budget", evidence

    if weak_output:
        if instruction_echo:
            evidence.append("instruction_echo")
        if metadata_borrowing:
            evidence.append("metadata_borrowing")
        if has_hallucination:
            evidence.append("hallucination")
        if missing_sections:
            evidence.append("contract")
        if utility_grade in {"D", "F"}:
            evidence.append(f"utility:{utility_grade}")
        return "model_shortcoming", evidence

    return "clean", evidence


def _classify_user_bucket(
    *,
    verdict: str,
    hint_relationship: str,
    has_contract_issue: bool,
    utility_grade: str = "",
) -> str:
    """Bucket outputs for end users based on verdict and utility signals.

    Verdict-to-bucket mapping:

    * ``harness`` / ``cutoff_degraded`` → ``"avoid"``
    * ``token_cap`` with A/B grade → ``"recommended"`` (model is good, just
      needs more budget)
    * ``token_cap`` with C or below → ``"caveat"``
    * ``context_budget`` → ``"caveat"``
    * ``model_shortcoming`` or degraded hints → ``"avoid"``
    * ``runtime_failure`` → ``"avoid"``
    * ``unknown_runtime_anomaly`` → ``"needs_triage"``
    * ``clean`` with positive hints and no contract issues → ``"recommended"``
    * everything else → ``"caveat"``
    """
    if verdict in {"harness", "cutoff_degraded", "runtime_failure"}:
        return "avoid"
    if verdict == "token_cap":
        if utility_grade in {"A", "B"} and hint_relationship != "degrades_trusted_hints":
            return "recommended"
        return "caveat"
    if verdict in {"context_budget", "unknown_runtime_anomaly"}:
        return "needs_triage" if verdict == "unknown_runtime_anomaly" else "caveat"
    if verdict == "model_shortcoming" or hint_relationship == "degrades_trusted_hints":
        return "avoid"
    is_clean_recommended = (
        verdict == "clean"
        and hint_relationship
        in {
            "improves_trusted_hints",
            "preserves_trusted_hints",
        }
        and not has_contract_issue
    )
    return "recommended" if is_clean_recommended else "caveat"


def _detect_refusal_patterns(text: str) -> tuple[bool, str | None]:
    """Detect if model refused or expressed high uncertainty.

    Catches cases where the model can't or won't process the image.

    Args:
        text: Generated text to check

    Returns:
        Tuple of (is_refusal, refusal_type)
    """
    if not text:
        return False, None

    text_lower: str = text.lower()

    # Refusal patterns
    refusal_patterns: list[tuple[str, list[str]]] = []

    for refusal_type, pattern_key in (
        ("explicit_refusal", "refusal_explicit"),
        ("uncertainty", "refusal_uncertainty"),
        ("insufficient_info", "refusal_insufficient_info"),
    ):
        patterns: list[str] = _get_quality_pattern_list(pattern_key, [])
        if patterns:
            refusal_patterns.append((refusal_type, patterns))

    if not refusal_patterns:
        # Fallback defaults

        refusal_patterns = [
            (
                "explicit_refusal",
                [
                    "i cannot",
                    "i can't",
                    "i'm unable to",
                    "i am unable to",
                    "sorry, i can't",
                    "sorry, i cannot",
                ],
            ),
            (
                "uncertainty",
                [
                    "it's unclear",
                    "it's difficult to say",
                    "i'm not sure",
                    "i cannot determine",
                    "unable to determine",
                    "difficult to tell",
                ],
            ),
            (
                "insufficient_info",
                [
                    "not enough information",
                    "insufficient detail",
                    "cannot see clearly",
                    "too blurry",
                    "image quality",
                ],
            ),
        ]

    for refusal_type, patterns in refusal_patterns:
        if any(pattern in text_lower for pattern in patterns):
            return True, refusal_type

    return False, None


def _detect_generic_output(text: str) -> tuple[bool, float]:
    """Detect overly generic or uninformative descriptions.

    Identifies low-quality captions that lack specific details.

    Args:
        text: Generated text to check

    Returns:
        Tuple of (is_generic, specificity_score where lower = more generic)
    """
    if not text or len(text) < QUALITY.min_text_length_for_generic:
        return False, 0.0

    text_lower: str = text.lower()
    word_count: int = len(text.split())

    if word_count == 0:
        return False, 0.0

    # Count filler/hedge words
    filler_words: list[str] = _get_quality_pattern_list(
        "filler_words",
        [
            "appears to",
            "seems to",
            "looks like",
            "might be",
            "could be",
            "some",
            "several",
            "various",
            "many",
            "few",
            "very",
            "quite",
            "rather",
            "somewhat",
            "fairly",
            "thing",
            "stuff",
            "item",
            "object",
        ],
    )
    filler_count: int = sum(text_lower.count(filler) for filler in filler_words)

    # Calculate filler ratio
    filler_ratio: float = filler_count / word_count

    # Check for specific details (numbers, measurements, colors, names)
    has_numbers: bool = bool(re.search(r"\d+", text))
    has_specific_colors: bool = bool(
        re.search(
            r"\b(red|blue|green|yellow|orange|purple|pink|brown|black|white|gray|grey)\b",
            text_lower,
        ),
    )
    has_proper_nouns: bool = bool(re.search(r"\b[A-Z][a-z]+", text))

    specificity_indicators: int = sum([has_numbers, has_specific_colors, has_proper_nouns])

    # Generic if high filler ratio and low specificity
    is_generic: bool = (
        filler_ratio > QUALITY.generic_filler_threshold
        and specificity_indicators < QUALITY.min_specificity_indicators
    )

    # Specificity score: higher = more specific (0-100)
    specificity_score: float = max(
        0.0,
        100 - (filler_ratio * 200) + (specificity_indicators * 20),
    )

    return is_generic, round(specificity_score, 1)


def _detect_language_mixing(
    text: str,
    quality_thresholds: QualityThresholds | None = None,
) -> tuple[bool, list[str]]:
    """Detect unexpected language switches and tokenizer-artifact text.

    Natural-language mixing and leaked special tokens both surface as unreadable
    output noise, so this helper reports them through the same issue channel.

    Args:
        text: Generated text to check
        quality_thresholds: Optional configuration override for patterns and thresholds

    Returns:
        Tuple of (has_mixing, list of detected issues)
    """
    if not text:
        return False, []

    if quality_thresholds is None:
        quality_thresholds = QUALITY

    issues: list[str] = []

    # Check for common tokenizer artifacts
    tokenizer_artifacts: list[str] = _get_quality_pattern_list(
        "tokenizer_artifacts",
        [
            r"<\|endoftext\|>",
            r"<\|end\|>",
            r"<s>",
            r"</s>",
            r"\[SEP\]",
            r"\[CLS\]",
            r"\[PAD\]",
            r"\[UNK\]",
            r"\[MASK\]",
            r"<pad>",
            r"<unk>",
            r"<mask>",
        ],
        quality_thresholds=quality_thresholds,
    )

    for artifact in tokenizer_artifacts:
        if re.search(artifact, text, re.IGNORECASE):
            issues.append("tokenizer_artifact")
            break

    # Check for code snippets (function calls, variable assignments)
    code_patterns: list[str] = _get_quality_pattern_list(
        "code_patterns",
        [
            r"\bdef\s+\w+\(",  # Python function def
            r"\bfunction\s+\w+\(",  # JavaScript function
            r"\bclass\s+\w+",  # Class definition
            r"\bimport\s+\w+",  # Import statement
            r"\breturn\s+",  # Return statement
        ],
        quality_thresholds=quality_thresholds,
    )

    for pattern in code_patterns:
        if re.search(pattern, text):
            issues.append("code_snippet")
            break

    return bool(issues), issues


def _detect_output_degeneration(text: str) -> tuple[bool, str | None]:
    """Detect end-of-output degeneration (rubbish/nonsense at the end).

    LLMs sometimes fail to stop properly and produce:
    - Repeated special characters or punctuation
    - Incomplete sentences cut off mid-word
    - Unicode rubbish or control characters
    - Repeated newlines/whitespace patterns

    Args:
        text: Generated text to check

    Returns:
        Tuple of (has_degeneration, degeneration_type)
    """
    if not text or len(text) < QUALITY.min_text_for_degeneration:
        return False, None

    # Check the last portion of the text (where degeneration typically appears)
    tail_length: int = min(200, len(text) // 3)
    tail: str = text[-tail_length:]
    result: str | None = None

    # 1. Detect repeated punctuation/special char sequences at end
    # e.g., "......" or "?????" or "!!!!!" or "-----"
    punct_repeat: re.Match[str] | None = re.search(r"([.?!,;:\-_=+*#]{3,})\s*$", tail)
    if punct_repeat:
        result = f"repeated_punctuation: '{punct_repeat.group(1)[:10]}...'"

    # 2. Detect incomplete sentence (ends mid-word or with lowercase without punctuation)
    if result is None:
        stripped: str = text.rstrip()
        if stripped:
            last_char: str = stripped[-1]
            # Normal endings: . ! ? ) " ' ] }
            normal_endings: str = ".!?)]}'\"}"
            if last_char not in normal_endings:
                last_word_match: re.Match[str] | None = re.search(r"\b(\w+)$", stripped)
                if last_word_match:
                    last_word: str = last_word_match.group(1)
                    if len(last_word) <= QUALITY.min_cutoff_word_length and last_word.islower():
                        result = f"incomplete_sentence: ends with '{last_word}'"

    # 3. Detect Unicode rubbish/control characters (excluding normal whitespace)
    if result is None:
        control_chars: list[str] = re.findall(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", tail)
        if len(control_chars) > QUALITY.max_control_chars:
            result = f"control_characters: {len(control_chars)} found"

    # 4. Detect repeated newline patterns (degenerate spacing)
    if result is None and "\n\n\n\n\n\n" in tail:
        result = "excessive_newlines"

    # 5. Detect character-level repetition at the end
    if result is None:
        char_repeat: re.Match[str] | None = re.search(r"(.{1,3})\1{5,}\s*$", tail)
        if char_repeat:
            pattern: str = char_repeat.group(1)
            result = f"character_loop: '{pattern}' repeated"

    # 6. Detect sudden encoding shift
    if result is None and len(text) > tail_length * 2:
        head: str = text[:-tail_length]
        ascii_max: int = 127  # Standard ASCII range
        head_non_ascii: float = len([c for c in head if ord(c) > ascii_max]) / max(len(head), 1)
        tail_non_ascii: float = len([c for c in tail if ord(c) > ascii_max]) / max(len(tail), 1)
        if (
            tail_non_ascii > QUALITY.non_ascii_ratio_threshold
            and tail_non_ascii > head_non_ascii * QUALITY.non_ascii_ratio_multiplier
        ):
            result = "encoding_shift"

    return (result is not None), result


def _detect_fabricated_details(text: str) -> tuple[bool, list[str]]:
    """Detect potentially fabricated specific details (hallucination).

    LLMs sometimes invent specific details like:
    - Fake dates (especially future dates or specific historical dates)
    - Made-up URLs/links
    - Invented statistics/percentages
    - Fictional proper names in contexts where they shouldn't appear

    Args:
        text: Generated text to check

    Returns:
        Tuple of (has_fabrication, list of suspicious details)
    """
    if not text:
        return False, []

    issues: list[str] = []

    # 1. Detect suspicious URLs (models often fabricate URLs)
    url_patterns: list[str] = _get_quality_pattern_list(
        "fabrication_url_patterns",
        [r"https?://[^\s<>\"']+"],
    )
    urls: list[str] = _extract_pattern_matches(
        text,
        url_patterns,
        debug_context="fabrication URL",
        unique=True,
    )

    suspicious_url_keywords: list[str] = _get_quality_pattern_list(
        "fabrication_suspicious_url_keywords",
        ["example.com", "placeholder", "xxx", "fake"],
    )
    long_url_path_patterns: list[str] = _get_quality_pattern_list(
        "fabrication_long_url_path_patterns",
        [r"/[a-z0-9]{20,}/"],
    )
    for url in urls:
        # Fabricated URLs often have suspicious patterns
        if any(suspicious in url.lower() for suspicious in suspicious_url_keywords):
            issues.append(f"suspicious_url: {url[:50]}")
        # Very long URLs with random-looking paths
        elif len(url) > QUALITY.max_url_length and _matches_any_pattern(
            url.lower(),
            long_url_path_patterns,
            debug_context="fabrication long URL path",
        ):
            issues.append(f"fabricated_url: {url[:50]}...")

    # 2. Detect invented precise statistics (suspiciously specific numbers)
    # e.g., "exactly 73.847%" or "precisely 14,523 items"
    precise_stat_patterns: list[str] = _get_quality_pattern_list(
        "fabrication_precise_stat_patterns",
        [r"\b(\d{1,3}(?:,\d{3})*\.\d{3,})\s*%?"],
    )
    precise_stats: list[str] = _extract_pattern_matches(
        text,
        precise_stat_patterns,
        debug_context="fabrication precise stat",
    )
    if len(precise_stats) >= QUALITY.min_precise_stats:
        issues.append(f"suspicious_precision: {len(precise_stats)} overly precise numbers")

    # 3. Detect future dates (model can't know the future)
    # Years 2030+ are definitely future
    future_year_patterns: list[str] = _get_quality_pattern_list(
        "fabrication_future_year_patterns",
        [r"\b(20[3-9]\d|2[1-9]\d{2})\b"],
    )
    future_years: list[str] = _extract_pattern_matches(
        text,
        future_year_patterns,
        debug_context="fabrication future year",
        unique=True,
    )
    if future_years:
        issues.append(f"future_date: {', '.join(future_years[:3])}")

    # 4. Detect citations to non-existent sources (common hallucination)
    # Patterns like "according to Smith et al. (2024)" or "(Johnson, 2025)"
    citation_patterns: list[str] = _get_quality_pattern_list(
        "fabrication_citation_patterns",
        [r"\(([A-Z][a-z]+(?:\s+et\s+al\.?)?,?\s*\d{4})\)"],
    )
    fake_citations: list[str] = _extract_pattern_matches(
        text,
        citation_patterns,
        debug_context="fabrication citation",
        unique=True,
    )
    if fake_citations:
        issues.append(f"unverifiable_citation: {', '.join(fake_citations[:2])}")

    return bool(issues), issues


# =============================================================================
# HARNESS/INTEGRATION ISSUE DETECTION
# =============================================================================
# These detect issues that are likely bugs in mlx-vlm or model integration,
# NOT inherent model quality problems. Separating them helps users know
# whether to report issues upstream vs. use a different model.

BPE_BYTE_ARTIFACTS: Final[tuple[tuple[str, str], ...]] = (
    ("\u0100", "byte_0"),
    ("\u0101", "byte_1"),
    ("\u0102", "byte_2"),
)

SPECIAL_TOKEN_LEAK_PATTERNS: Final[tuple[tuple[str, str], ...]] = (
    (r"<\|end\|>", "<|end|>"),
    (r"<\|endoftext\|>", "<|endoftext|>"),
    (r"<\|eot_id\|>", "<|eot_id|>"),
    (r"<\|im_end\|>", "<|im_end|>"),
    (r"<\|assistant\|>", "<|assistant|>"),
    (r"<end_of_turn>", "<end_of_turn>"),
    (r"</s>(?!\w)", "</s>"),
    (r"<s>(?!\w)", "<s>"),
    (r"# INSTRUCTION", "# INSTRUCTION"),
    (r"# SOLUTION", "# SOLUTION"),
    (r"\[INST\]", "[INST]"),
    (r"\[/INST\]", "[/INST]"),
    (r"<\|think\|>", "<|think|>"),
    (r"</think>", "</think>"),
    (r"<\|pad\|>", "<|pad|>"),
    (r"<\|unk\|>", "<|unk|>"),
    (r"\[PAD\]", "[PAD]"),
    (r"\[CLS\]", "[CLS]"),
    (r"\[SEP\]", "[SEP]"),
)


def _detect_token_encoding_issues(text: str) -> tuple[bool, str | None]:
    """Detect tokenizer decoding bugs where raw BPE tokens leak through.

    Common pattern: Ġ (U+0120) appearing instead of spaces, indicating
    the tokenizer's space-prefix marker wasn't decoded properly.

    Args:
        text: Generated text to check

    Returns:
        Tuple of (has_issue, issue_type)
    """
    if not text:
        return False, None

    # Check for Ġ (U+0120) - BPE space marker leak
    # This is a specific HuggingFace tokenizer artifact
    if "\u0120" in text:
        space_leak_count: int = text.count("\u0120")
        return True, f"bpe_space_leak({space_leak_count})"

    # Check for Ċ (U+010A) - BPE newline marker leak
    if "\u010a" in text:
        newline_leak_count: int = text.count("\u010a")
        return True, f"bpe_newline_leak({newline_leak_count})"

    # Check for other common tokenizer artifacts that shouldn't be visible
    for artifact, name in BPE_BYTE_ARTIFACTS:
        if artifact in text and text.count(artifact) > QUALITY.min_bpe_artifact_count:
            return True, f"bpe_byte_leak({name})"

    return False, None


def _sanitize_bpe_display(text: str, *, max_len: int = 120) -> str:
    """Replace BPE marker characters with readable equivalents for display.

    The result is truncated to *max_len* so it fits in issue titles and log
    lines, while keeping marker leaks visible instead of rendering as blank
    spaces.
    """
    marker_labels: list[str] = []
    if "\u0120" in text:
        marker_labels.append("BPE-space-marker")
    if "\u010a" in text:
        marker_labels.append("BPE-newline-marker")

    cleaned = text.replace("\u0120", " ").replace("\u010a", "\\n")
    for artifact, _name in BPE_BYTE_ARTIFACTS:
        if artifact in cleaned:
            marker_labels.append("BPE-byte-marker")
            cleaned = cleaned.replace(artifact, "?")
    if marker_labels:
        cleaned = f"{cleaned} [markers: {', '.join(_dedupe_preserve_order(marker_labels))}]"
    if len(cleaned) > max_len:
        cleaned = cleaned[: max_len - 3].rstrip() + "..."
    return cleaned


def _detect_special_token_leakage(text: str) -> tuple[bool, list[str]]:
    """Detect special/control tokens that leaked into output.

    These indicate the generation stop logic failed to halt at EOS/EOT tokens,
    or the output wasn't properly post-processed to strip special tokens.

    Args:
        text: Generated text to check

    Returns:
        Tuple of (has_leakage, list of leaked tokens found)
    """
    if not text:
        return False, []

    leaked_tokens: list[str] = []

    for pattern, token_name in SPECIAL_TOKEN_LEAK_PATTERNS:
        if re.search(pattern, text):
            leaked_tokens.append(token_name)

    return bool(leaked_tokens), leaked_tokens


def _detect_minimal_output(
    text: str,
    generated_tokens: int,
    prompt_tokens: int | None = None,
) -> tuple[bool, str | None]:
    """Detect suspiciously minimal output suggesting prompt template issues.

    When a model generates very few tokens despite a substantial prompt,
    it often indicates:
    - Wrong chat template applied
    - Model thinks task is already complete
    - Generation parameters misconfigured

    Args:
        text: Generated text
        generated_tokens: Number of tokens generated
        prompt_tokens: Number of tokens in prompt (if known)

    Returns:
        Tuple of (is_minimal, reason)
    """
    # Zero tokens is always a harness issue
    if generated_tokens == 0:
        return True, "zero_tokens"

    # Less than threshold tokens when we have a substantial prompt
    if generated_tokens < QUALITY.min_tokens_for_substantial:
        # Check if the output is actually meaningful or just filler
        text_stripped: str = text.strip()
        word_count: int = len(text_stripped.split())

        # Single sentence filler responses
        filler_responses: list[str] = [
            "the image is a photograph",
            "the image is in the public domain",
            "i cannot",
            "i can't",
            "this image shows",
        ]
        text_lower: str = text_stripped.lower()
        for filler in filler_responses:
            if text_lower.startswith(filler) and word_count < QUALITY.min_words_for_filler_response:
                return True, f"filler_response({generated_tokens}tok)"

        if word_count < QUALITY.min_words_for_truncated:
            return True, f"truncated({generated_tokens}tok)"

    # Very low ratio of output to prompt (if prompt_tokens known)
    if (
        prompt_tokens
        and prompt_tokens > QUALITY.min_prompt_tokens_for_ratio
        and generated_tokens < QUALITY.min_output_tokens_for_ratio
    ):
        ratio: float = generated_tokens / prompt_tokens
        if ratio < QUALITY.min_output_ratio:
            return True, f"output_ratio({ratio:.1%})"

    return False, None


def _detect_long_context_breakdown(
    *,
    prompt_tokens: int | None,
    generated_tokens: int,
    text: str,
    is_repetitive: bool,
    is_context_ignored: bool,
    is_refusal: bool,
) -> tuple[bool, str | None]:
    """Detect likely long-context degradation that may indicate stack issues.

    These signals are intentionally conservative and only trigger when prompt
    token counts are high enough that prompt packing/prefill behavior can
    dominate generation quality.
    """
    if prompt_tokens is None or prompt_tokens < QUALITY.long_prompt_tokens_threshold:
        return False, None

    safe_prompt_tokens: int = max(prompt_tokens, 1)
    ratio: float = generated_tokens / safe_prompt_tokens
    text_empty: bool = not text.strip()

    if text_empty and generated_tokens == 0:
        return True, f"long_context_empty({prompt_tokens}tok)"

    if generated_tokens < QUALITY.min_output_tokens_for_ratio and ratio < QUALITY.min_output_ratio:
        return True, f"long_context_low_ratio({ratio:.1%};{prompt_tokens}->{generated_tokens})"

    if prompt_tokens >= QUALITY.severe_prompt_tokens_threshold and is_repetitive:
        return True, f"long_context_repetition({prompt_tokens}tok)"

    if (
        prompt_tokens >= QUALITY.severe_prompt_tokens_threshold
        and is_context_ignored
        and not is_refusal
    ):
        return True, f"long_context_context_drop({prompt_tokens}tok)"

    return False, None


def _detect_training_data_leak(text: str) -> tuple[bool, str | None]:
    """Detect training data or instruction template leaking into output.

    Some models fail to stop and start generating what looks like
    training examples or new instructions mid-output.

    Args:
        text: Generated text to check

    Returns:
        Tuple of (has_leak, leak_type)
    """
    if not text or len(text) < QUALITY.min_text_for_leak_detection:
        return False, None

    # Look for instruction-like patterns appearing mid-output
    training_leak_pattern_groups: list[tuple[list[str], str]] = [
        (
            _get_quality_pattern_list(
                "training_leak_instruction_header_patterns",
                [r"\n# INSTRUCTION\b"],
            ),
            "instruction_header",
        ),
        (
            _get_quality_pattern_list(
                "training_leak_task_header_patterns",
                [r"\n## (Task|Question|Instructions?):"],
            ),
            "task_header",
        ),
        (
            _get_quality_pattern_list(
                "training_leak_write_prompt_patterns",
                [r"\nWrite a (?:short )?(?:story|essay|poem|code)"],
            ),
            "write_prompt",
        ),
        (
            _get_quality_pattern_list(
                "training_leak_user_turn_patterns",
                [r"\n(?:User|Human|Question):\s*\n"],
            ),
            "user_turn",
        ),
        (
            _get_quality_pattern_list(
                "training_leak_code_example_patterns",
                [r"\n```\w+\n.*?def \w+\("],
            ),
            "code_example",
        ),
        (
            _get_quality_pattern_list(
                "training_leak_qa_pair_patterns",
                [r"\nQ:\s*\n.*?\nA:\s*\n"],
            ),
            "qa_pair",
        ),
    ]

    # Only check the latter portion of output (leaks happen after good output)
    check_portion: str = text[len(text) // 3 :]

    for patterns, leak_type in training_leak_pattern_groups:
        for pattern in patterns:
            try:
                if re.search(pattern, check_portion, re.DOTALL):
                    return True, leak_type
            except re.error:
                logger.debug("Ignoring invalid training leak regex: %s", pattern)

    return False, None


def compute_vocabulary_diversity(text: str) -> tuple[float, int, int]:
    """Compute lexical diversity as type-token ratio (TTR).

    Tokenization uses lowercase alphabetic words, returning
    ``(ttr, unique_words, total_words)``.
    """
    if not text:
        return 0.0, 0, 0

    # Normalize: lowercase, extract word tokens only
    words: list[str] = re.findall(r"\b[a-z]+\b", text.lower())
    total_words: int = len(words)

    if total_words == 0:
        return 0.0, 0, 0

    unique_words: int = len(set(words))
    ttr: float = unique_words / total_words

    return round(ttr, 3), unique_words, total_words


def compute_efficiency_metrics(
    tokens_generated: int,
    generation_time: float | None,
    peak_memory_gb: float | None,
) -> dict[str, float | None]:
    """Compute efficiency metrics combining speed and memory usage.

    Args:
        tokens_generated: Number of tokens generated
        generation_time: Time for generation in seconds
        peak_memory_gb: Peak memory usage in GB

    Returns:
        Dict with computed efficiency metrics:
        - tokens_per_second: Generation speed
        - tokens_per_gb: Tokens generated per GB of memory (efficiency)
        - tokens_per_second_per_gb: Combined efficiency metric
    """
    metrics: dict[str, float | None] = {
        "tokens_per_second": None,
        "tokens_per_gb": None,
        "tokens_per_second_per_gb": None,
    }

    if generation_time and generation_time > 0:
        metrics["tokens_per_second"] = round(tokens_generated / generation_time, 1)

    if peak_memory_gb and peak_memory_gb > 0:
        metrics["tokens_per_gb"] = round(tokens_generated / peak_memory_gb, 1)

        if generation_time and generation_time > 0:
            tps: float = tokens_generated / generation_time
            metrics["tokens_per_second_per_gb"] = round(tps / peak_memory_gb, 2)

    return metrics


def detect_response_structure(text: str) -> dict[str, bool]:
    """Detect if response contains expected structural elements.

    For image cataloging tasks, we expect outputs to include captions,
    keywords, and/or descriptions. This detects their presence.

    This is a lightweight wrapper around compute_task_compliance() that
    adds section detection and returns only boolean presence indicators.

    Args:
        text: Generated text to analyze

    Returns:
        Dict indicating presence of each structural element
    """
    if not text:
        return {
            "has_caption": False,
            "has_keywords": False,
            "has_description": False,
            "has_sections": False,
        }

    # Reuse task compliance detection for the core elements
    compliance: dict[str, bool | float] = compute_task_compliance(text)

    # Add section detection (markdown headers)
    has_sections: bool = bool(re.search(r"^#{1,3}\s+\w+", text, re.MULTILINE))

    return {
        "has_caption": bool(compliance["has_caption"]),
        "has_keywords": bool(compliance["has_keywords"]),
        "has_description": bool(compliance["has_description"]),
        "has_sections": has_sections,
    }


def _get_quality_pattern_list(
    pattern_key: str,
    fallback: list[str],
    *,
    quality_thresholds: QualityThresholds | None = None,
) -> list[str]:
    """Return configured regex/pattern list for a key, falling back to defaults."""
    if quality_thresholds is None:
        quality_thresholds = QUALITY

    if not quality_thresholds.patterns:
        return fallback

    configured: list[str] | None = quality_thresholds.patterns.get(pattern_key)
    if not configured:
        return fallback

    # Guard against malformed YAML values while keeping detector behavior stable.
    valid: list[str] = [p for p in configured if isinstance(p, str)]
    return valid or fallback


def _contains_labeled_section(text_lower: str, label_patterns: list[str]) -> bool:
    """Return True if text contains any configured section label pattern."""
    for pattern in label_patterns:
        compiled = _compile_regex_for_detection(
            rf"\b(?:{pattern})\s*:",
            debug_context="section label",
        )
        if compiled and compiled.search(text_lower):
            return True
    return False


@lru_cache(maxsize=2048)
def _compile_regex_cached(pattern: str, flags: int) -> re.Pattern[str] | None:
    """Compile and cache regex patterns used in detector utilities."""
    try:
        return re.compile(pattern, flags)
    except re.error:
        return None


def _compile_regex_for_detection(
    pattern: str,
    *,
    debug_context: str,
    flags: int = 0,
) -> re.Pattern[str] | None:
    """Return compiled regex or log-and-skip invalid configured patterns."""
    compiled = _compile_regex_cached(pattern, flags)
    if compiled is None:
        logger.debug("Ignoring invalid %s regex: %s", debug_context, pattern)
    return compiled


def _extract_pattern_matches(
    text: str,
    patterns: list[str],
    *,
    debug_context: str,
    flags: int = 0,
    unique: bool = False,
) -> list[str]:
    """Return regex matches for configured patterns, ignoring invalid regex entries."""
    matches: list[str] = []
    seen: set[str] = set()

    for pattern in patterns:
        compiled = _compile_regex_for_detection(
            pattern,
            debug_context=debug_context,
            flags=flags,
        )
        if compiled is None:
            continue
        for match in compiled.finditer(text):
            value: str = match.group(0)
            if not value:
                continue
            if unique and value in seen:
                continue
            matches.append(value)
            seen.add(value)

    return matches


def _matches_any_pattern(text: str, patterns: list[str], *, debug_context: str) -> bool:
    """Return True if text matches at least one configured regex pattern."""
    for pattern in patterns:
        compiled = _compile_regex_for_detection(pattern, debug_context=debug_context)
        if compiled and compiled.search(text):
            return True
    return False


def _count_pattern_matches(text: str, patterns: list[str]) -> int:
    """Count total regex matches across all configured patterns."""
    total: int = 0
    for pattern in patterns:
        compiled = _compile_regex_for_detection(pattern, debug_context="pattern")
        if compiled is None:
            continue
        total += len(compiled.findall(text))
    return total


def compute_confidence_indicators(text: str) -> dict[str, float | int]:
    """Analyze text for confidence/certainty indicators.

    Hedge words indicate uncertainty; definitive language indicates confidence.
    The ratio helps assess how certain the model is about its descriptions.

    Args:
        text: Generated text to analyze

    Returns:
        Dict with:
        - hedge_count: Number of hedge words/phrases
        - definitive_count: Number of definitive statements
        - confidence_ratio: Ratio of definitive to (definitive + hedge)
    """
    if not text:
        return {"hedge_count": 0, "definitive_count": 0, "confidence_ratio": 0.0}

    text_lower: str = text.lower()

    hedge_patterns: list[str] = _get_quality_pattern_list(
        "confidence_hedge_patterns",
        [
            r"\bappears to\b",
            r"\bseems to\b",
            r"\blooks like\b",
            r"\bmight be\b",
            r"\bcould be\b",
            r"\bpossibly\b",
            r"\bperhaps\b",
            r"\bprobably\b",
            r"\blikely\b",
            r"\bmaybe\b",
            r"\bi think\b",
            r"\bi believe\b",
            r"\bit's unclear\b",
            r"\buncertain\b",
        ],
    )
    definitive_patterns: list[str] = _get_quality_pattern_list(
        "confidence_definitive_patterns",
        [
            r"\bis a\b",
            r"\bare \w+\b",
            r"\bshows\b",
            r"\bdepicts\b",
            r"\bfeatures\b",
            r"\bcontains\b",
            r"\bdefinitely\b",
            r"\bclearly\b",
            r"\bobviously\b",
        ],
    )

    hedge_count: int = _count_pattern_matches(text_lower, hedge_patterns)
    definitive_count: int = _count_pattern_matches(text_lower, definitive_patterns)

    total: int = hedge_count + definitive_count
    confidence_ratio: float = definitive_count / total if total > 0 else 0.5

    return {
        "hedge_count": hedge_count,
        "definitive_count": definitive_count,
        "confidence_ratio": round(confidence_ratio, 2),
    }


# =============================================================================
# CATALOGING-SPECIFIC QUALITY METRICS
# =============================================================================


def compute_information_gain(text: str, context: str | None) -> dict[str, float | int]:
    """Measure novel information in output beyond what was provided in context.

    For cataloging tasks, we want models to add value beyond just echoing the
    context hint. This measures how much new information the model contributes.

    Args:
        text: Generated text to analyze
        context: Original context/hint provided to the model

    Returns:
        Dict with:
        - context_words: Words from context
        - output_words: Words in output
        - novel_words: Words in output not in context
        - echo_ratio: Fraction of output that's just echoed context (lower = better)
        - information_gain: Fraction of output that's novel (higher = better)
    """
    if not text:
        return {
            "context_words": 0,
            "output_words": 0,
            "novel_words": 0,
            "echo_ratio": 0.0,
            "information_gain": 0.0,
        }

    # Extract meaningful words (lowercase, alpha only, 3+ chars)
    def extract_words(s: str) -> set[str]:
        return {w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", s)}

    output_words = extract_words(text)
    context_words = extract_words(context) if context else set()

    if not output_words:
        return {
            "context_words": len(context_words),
            "output_words": 0,
            "novel_words": 0,
            "echo_ratio": 0.0,
            "information_gain": 0.0,
        }

    # Words in output that came from context (echoed)
    # Words in output that came from context (echoed)
    echoed_words = output_words & context_words
    # Words in output that are novel (not in context)
    novel_words = output_words - context_words

    echo_ratio = len(echoed_words) / len(output_words) if output_words else 0.0
    information_gain = len(novel_words) / len(output_words) if output_words else 0.0

    # Penalised echo: only count nonvisual/instructional context reuse.
    # Models are expected to reuse visual hint terms (e.g. "church", "sunset")
    # — only structural or metadata terms count as problematic echo.
    nonvisual_context = {w for w in context_words if _is_nonvisual_context_term(w)}
    penalized_echo = output_words & nonvisual_context
    penalized_echo_ratio = len(penalized_echo) / len(output_words) if output_words else 0.0

    return {
        "context_words": len(context_words),
        "output_words": len(output_words),
        "novel_words": len(novel_words),
        "echo_ratio": round(echo_ratio, 2),
        "penalized_echo_ratio": round(penalized_echo_ratio, 2),
        "information_gain": round(information_gain, 2),
    }


def compute_task_compliance(text: str) -> dict[str, bool | float]:
    """Check if output follows the requested structure for cataloging tasks.

    When asked for "caption, description, and keywords", models should provide
    all three components. This measures compliance with that structure.

    Args:
        text: Generated text to analyze

    Returns:
        Dict with:
        - has_caption: Contains a caption or title
        - has_description: Contains descriptive text
        - has_keywords: Contains keywords or tags
        - compliance_score: 0-1 score based on components present
    """
    if not text:
        return {
            "has_caption": False,
            "has_description": False,
            "has_keywords": False,
            "compliance_score": 0.0,
        }

    text_lower = text.lower()

    caption_labels = _get_quality_pattern_list(
        "task_caption_labels",
        ["caption", "title"],
    )
    description_labels = _get_quality_pattern_list(
        "task_description_labels",
        ["description", "details?", "summary"],
    )
    keyword_labels = _get_quality_pattern_list(
        "task_keyword_labels",
        ["keywords?", "tags?"],
    )

    # Check for explicit labeled sections
    has_explicit_caption = _contains_labeled_section(text_lower, caption_labels)
    has_explicit_description = _contains_labeled_section(text_lower, description_labels)
    has_explicit_keywords = _contains_labeled_section(text_lower, keyword_labels)

    # Check for implicit structure (bullet lists for keywords, paragraphs for description)
    has_bullet_list = bool(re.search(r"^[-•*]\s+\w+", text, re.MULTILINE))
    has_paragraph = len(text.split()) > QUALITY.substantial_prose_words

    # Combine explicit and implicit signals
    has_caption = has_explicit_caption or (
        # First line could be a caption if short and followed by more text
        len(text.split("\n", maxsplit=1)[0].split()) <= QUALITY.max_caption_words
        and len(text.split("\n")) > 1
    )
    has_description = has_explicit_description or has_paragraph
    has_keywords = has_explicit_keywords or has_bullet_list

    # Extract strict constraint data
    (
        _,
        title_word_count,
        description_sentence_count,
        keyword_count,
        keyword_dup_ratio,
    ) = _analyze_catalog_contract(text)

    score = 0.0

    # Caption/Title scoring
    if has_caption:
        score += 0.16
        if (
            title_word_count is not None
            and QUALITY.min_title_words <= title_word_count <= QUALITY.max_title_words
        ):
            score += 0.17

    # Description scoring
    if has_description:
        score += 0.17
        if (
            description_sentence_count is not None
            and QUALITY.min_description_sentences
            <= description_sentence_count
            <= QUALITY.max_description_sentences
        ):
            score += 0.17

    # Keywords scoring
    if has_keywords:
        score += 0.16
        if (
            keyword_count is not None
            and QUALITY.min_keywords_count <= keyword_count <= QUALITY.max_keywords_count
        ):
            keyword_score = 0.17
            if (
                keyword_dup_ratio is not None
                and keyword_dup_ratio >= QUALITY.keyword_duplication_ratio_threshold
            ):
                keyword_score = 0.0
            score += keyword_score

    return {
        "has_caption": has_caption,
        "has_description": has_description,
        "has_keywords": has_keywords,
        "compliance_score": round(score, 2),
    }


def compute_visual_grounding(text: str, context: str | None) -> dict[str, float | int]:
    """Measure references to actual visual elements vs. just context regurgitation.

    Good cataloging descriptions should reference what's actually visible in the
    image - colors, objects, people, actions, spatial relationships - not just
    repeat location/date metadata from the context.

    Args:
        text: Generated text to analyze
        context: Original context provided (to distinguish visual from contextual)

    Returns:
        Dict with:
        - visual_terms: Count of visual description terms
        - spatial_terms: Count of spatial relationship terms
        - color_terms: Count of color references
        - grounding_score: 0-1 overall visual grounding score
    """
    if not text:
        return {
            "visual_terms": 0,
            "spatial_terms": 0,
            "color_terms": 0,
            "grounding_score": 0.0,
        }

    text_lower = text.lower()
    context_lower = (context or "").lower()

    # Visual object/element terms (things you can see)
    visual_patterns = _get_quality_pattern_list(
        "visual_grounding_visual_patterns",
        [
            r"\b(building|house|shop|store|street|road|car|vehicle|person|people|pedestrian)\b",
            r"\b(sign|window|door|roof|wall|brick|stone|glass)\b",
            r"\b(tree|sky|cloud|hill|mountain|grass|flower)\b",
            r"\b(light|lamp|shadow|reflection|glow)\b",
            r"\b(wearing|standing|sitting|walking|driving)\b",
        ],
    )

    # Spatial relationship terms
    spatial_patterns = _get_quality_pattern_list(
        "visual_grounding_spatial_patterns",
        [
            r"\b(left|right|center|middle|foreground|background)\b",
            r"\b(above|below|beside|behind|front|back)\b",
            r"\b(near|far|distant|close|adjacent)\b",
            r"\b(top|bottom|side|corner|edge)\b",
        ],
    )

    # Color terms
    color_patterns = _get_quality_pattern_list(
        "visual_grounding_color_patterns",
        [
            r"\b(red|blue|green|yellow|orange|purple|pink|brown|black|white|gray|grey)\b",
            r"\b(golden|silver|bronze|dark|light|bright|pale|vivid)\b",
            r"\b(warm|cool|muted|saturated)\b",
        ],
    )

    visual_count = _count_pattern_matches(text_lower, visual_patterns)
    spatial_count = _count_pattern_matches(text_lower, spatial_patterns)
    color_count = _count_pattern_matches(text_lower, color_patterns)

    # Penalize if visual terms are just from context (not novel observations)
    context_visual = _count_pattern_matches(context_lower, visual_patterns) if context else 0
    novel_visual = max(0, visual_count - context_visual)

    # Calculate grounding score (weighted combination)
    # More weight to novel visual observations
    word_count = len(text.split())
    if word_count == 0:
        grounding_score = 0.0
    else:
        # Normalize by output length, cap at 1.0
        raw_score = (novel_visual * 2 + spatial_count + color_count) / max(word_count / 10, 1)
        grounding_score = min(1.0, raw_score)

    return {
        "visual_terms": visual_count,
        "spatial_terms": spatial_count,
        "color_terms": color_count,
        "grounding_score": round(grounding_score, 2),
    }


def _bounded_range_score(value: int, *, lower: int, upper: int) -> float:
    """Score an integer against an inclusive target range on a 0-1 scale."""
    if value <= 0:
        return 0.0
    if lower <= value <= upper:
        return 1.0
    if value < lower:
        return min(value / max(lower, 1), 1.0)
    return min(upper / max(value, 1), 1.0)


def _extract_description_candidate(text: str) -> str:
    """Extract the best available prose field for description scoring."""
    if not text:
        return ""

    sections = _extract_catalog_sections(text)
    for section_name in ("description", "title"):
        candidate = sections.get(section_name, "").strip()
        if candidate:
            return candidate
    return text.strip()


def _extract_keyword_terms_for_scoring(text: str) -> list[str]:
    """Extract explicit keyword terms from catalog-style output."""
    if not text:
        return []

    sections = _extract_catalog_sections(text)
    keywords_text = sections.get("keywords", "").strip()
    if keywords_text:
        return _split_catalog_keywords(keywords_text)

    bullet_terms: list[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped.startswith(("-", "*", "•")):
            continue
        candidate = stripped.lstrip("-*• ").strip()
        if not candidate:
            continue
        if re.match(r"(?i)^(title|description|keywords)\s*:", candidate):
            continue
        bullet_terms.append(candidate)
    if bullet_terms:
        return _split_catalog_keywords("\n".join(bullet_terms))
    return []


def compute_description_quality(text: str, context: str | None) -> dict[str, float | int]:
    """Score how useful the prose is for image description work."""
    description_text = _extract_description_candidate(text)
    if not description_text or len(description_text.strip()) < QUALITY.min_useful_chars:
        return {
            "description_score": 0.0,
            "description_word_count": 0,
            "description_sentence_count": 0,
            "description_detail_coverage": 0.0,
            "description_grounding_score": 0.0,
            "description_specificity_score": 0.0,
        }

    word_count = len(re.findall(r"[A-Za-z0-9']+", description_text))
    sentence_count = _count_factual_sentences(description_text)
    grounding = compute_visual_grounding(description_text, context)
    _is_generic, specificity_score = _detect_generic_output(description_text)

    detail_hits = sum(
        int(count > 0)
        for count in (
            int(grounding["visual_terms"]),
            int(grounding["spatial_terms"]),
            int(grounding["color_terms"]),
        )
    )
    detail_coverage = detail_hits / 3
    length_score = min(word_count / max(QUALITY.substantial_prose_words, 1), 1.0)
    sentence_fit = _bounded_range_score(
        sentence_count,
        lower=QUALITY.min_description_sentences,
        upper=QUALITY.max_description_sentences,
    )

    description_score = (
        (
            float(grounding["grounding_score"])
            + detail_coverage
            + min(specificity_score / 100.0, 1.0)
            + length_score
            + sentence_fit
        )
        / 5
    ) * 100.0

    return {
        "description_score": round(description_score, 1),
        "description_word_count": word_count,
        "description_sentence_count": sentence_count,
        "description_detail_coverage": round(detail_coverage, 2),
        "description_grounding_score": float(grounding["grounding_score"]),
        "description_specificity_score": round(specificity_score, 1),
    }


def compute_keyword_quality(text: str, context: str | None) -> dict[str, float | int]:
    """Score how useful the keyword field is for image indexing work."""
    keyword_terms = _extract_keyword_terms_for_scoring(text)
    if not keyword_terms:
        return {
            "keyword_score": 0.0,
            "keyword_term_count": 0,
            "keyword_unique_terms": 0,
            "keyword_duplication_ratio": 0.0,
            "keyword_category_coverage": 0.0,
            "keyword_grounding_score": 0.0,
        }

    normalized_terms = [_normalize_phrase_for_matching(term) for term in keyword_terms]
    unique_pairs: list[tuple[str, str]] = []
    seen_terms: set[str] = set()
    for term, normalized in zip(keyword_terms, normalized_terms, strict=False):
        if not normalized or normalized in seen_terms:
            continue
        seen_terms.add(normalized)
        unique_pairs.append((term, normalized))

    total_terms = len(keyword_terms)
    unique_terms = len(unique_pairs)
    duplication_ratio = 1.0 - (unique_terms / max(total_terms, 1))
    unique_term_text = ", ".join(term for term, _normalized in unique_pairs)

    composition_patterns = _get_quality_pattern_list(
        "keyword_composition_patterns",
        [
            r"\b(close[- ]up|macro|wide angle|panoramic|aerial|overhead)\b",
            r"\b(high angle|low angle|symmetry|copy space|selective focus|bokeh)\b",
            r"\b(portrait|landscape format|minimalism|leading lines)\b",
        ],
    )
    generic_keyword_patterns = _get_quality_pattern_list(
        "keyword_generic_patterns",
        [
            r"\bimage\b",
            r"\bphoto(graph)?\b",
            r"\bpicture\b",
            r"\bvisual\b",
            r"\bscene\b",
            r"\bbeautiful\b",
            r"\bnice\b",
        ],
    )
    grounding = compute_visual_grounding(unique_term_text, context)
    composition_count = _count_pattern_matches(unique_term_text.casefold(), composition_patterns)
    category_hits = sum(
        (
            int(int(grounding["visual_terms"]) > 0),
            int(int(grounding["color_terms"]) > 0),
            int(int(grounding["spatial_terms"]) > 0 or composition_count > 0),
        ),
    )
    category_coverage = category_hits / 3
    count_fit = _bounded_range_score(
        total_terms,
        lower=QUALITY.min_keywords_count,
        upper=QUALITY.max_keywords_count,
    )
    minimum_multiword_term_tokens = 2
    uniqueness_score = max(0.0, 1.0 - duplication_ratio)
    generic_count = sum(
        1
        for term, _normalized in unique_pairs
        if any(re.search(pattern, term, re.IGNORECASE) for pattern in generic_keyword_patterns)
    )
    multiword_count = sum(
        1
        for term, _normalized in unique_pairs
        if len(re.findall(r"[A-Za-z0-9']+", term)) >= minimum_multiword_term_tokens
    )
    specificity_score = (
        max(unique_terms - generic_count, 0) / max(unique_terms, 1)
        + (multiword_count / max(unique_terms, 1))
    ) / 2

    keyword_score = (
        (
            count_fit
            + uniqueness_score
            + category_coverage
            + specificity_score
            + float(grounding["grounding_score"])
        )
        / 5
    ) * 100.0

    return {
        "keyword_score": round(keyword_score, 1),
        "keyword_term_count": total_terms,
        "keyword_unique_terms": unique_terms,
        "keyword_duplication_ratio": round(duplication_ratio, 2),
        "keyword_category_coverage": round(category_coverage, 2),
        "keyword_grounding_score": float(grounding["grounding_score"]),
    }


def compute_cataloging_utility(
    text: str,
    context: str | None,
    *,
    info_gain: dict[str, float | int] | None = None,
    task_compliance: dict[str, bool | float] | None = None,
    visual_grounding: dict[str, float | int] | None = None,
) -> dict[str, float | int | str]:
    """Compute overall cataloging utility score combining all metrics.

    This is the primary "is this output useful for cataloging?" metric.

    Args:
        text: Generated text to analyze
        context: Original context provided
        info_gain: Pre-computed information gain (computed if None)
        task_compliance: Pre-computed task compliance (computed if None)
        visual_grounding: Pre-computed visual grounding (computed if None)

    Returns:
        Dict with:
        - utility_score: 0-100 overall utility for cataloging
        - utility_grade: Letter grade (A-F)
        - primary_weakness: Main issue limiting utility
    """
    if not text or len(text.strip()) < QUALITY.min_useful_chars:
        return {
            "utility_score": 0.0,
            "utility_grade": "F",
            "primary_weakness": "Empty or minimal output",
            "description_score": 0.0,
            "keyword_score": 0.0,
            "description_word_count": 0,
            "keyword_term_count": 0,
            "keyword_unique_terms": 0,
            "keyword_duplication_ratio": 0.0,
            "keyword_category_coverage": 0.0,
        }

    # Compute sub-metrics if not provided
    if info_gain is None:
        info_gain = compute_information_gain(text, context)
    if task_compliance is None:
        task_compliance = compute_task_compliance(text)
    if visual_grounding is None:
        visual_grounding = compute_visual_grounding(text, context)

    # Extract key values
    information_gain_score = float(info_gain.get("information_gain", 0.0))
    echo_ratio = float(info_gain.get("penalized_echo_ratio", info_gain.get("echo_ratio", 0.0)))
    compliance_score = float(task_compliance.get("compliance_score", 0.0))
    grounding_score = float(visual_grounding.get("grounding_score", 0.0))
    description_quality = compute_description_quality(text, context)
    keyword_quality = compute_keyword_quality(text, context)
    description_score = float(description_quality.get("description_score", 0.0))
    keyword_score = float(keyword_quality.get("keyword_score", 0.0))

    # Compute echo penalty inline
    word_count = len(text.split())
    if echo_ratio > QUALITY.severe_echo_threshold:
        echo_penalty = QUALITY.severe_echo_penalty  # Severe penalty for mostly echoing
    elif echo_ratio > QUALITY.moderate_echo_threshold:
        echo_penalty = QUALITY.moderate_echo_penalty  # Moderate penalty
    else:
        echo_penalty = 1.0  # No penalty

    # Compute length factor inline
    if word_count < QUALITY.min_useful_words:
        length_factor, length_weakness = (
            QUALITY.very_short_length_factor,
            "Output too short to be useful",
        )
    elif word_count < QUALITY.short_output_words:
        length_factor, length_weakness = QUALITY.short_length_factor, "Output lacks detail"
    else:
        length_factor, length_weakness = 1.0, ""

    # Weighted combination (out of 100)
    raw_score = (
        information_gain_score * QUALITY.cataloging_weight_information_gain
        + compliance_score * QUALITY.cataloging_weight_compliance
        + grounding_score * QUALITY.cataloging_weight_grounding
        + min(word_count / 50, 1.0) * QUALITY.cataloging_weight_length
    )

    final_score = raw_score * echo_penalty * length_factor

    # Identify primary weakness using threshold checks
    weakness_checks = [
        (bool(length_weakness), length_weakness),
        (
            echo_ratio > QUALITY.moderate_echo_threshold,
            "Mostly echoes context without adding value",
        ),
        (
            keyword_score < QUALITY.grade_d_threshold,
            "Keywords are not specific or diverse enough",
        ),
        (
            description_score < QUALITY.grade_d_threshold,
            "Description lacks concrete visual detail",
        ),
        (grounding_score < QUALITY.low_grounding_threshold, "Lacks visual description of image"),
        (compliance_score < QUALITY.low_compliance_threshold, "Missing requested structure"),
        (information_gain_score < QUALITY.low_info_gain_threshold, "Limited novel information"),
    ]
    weakness = next((msg for cond, msg in weakness_checks if cond), "None identified")

    # Convert score to grade using threshold lookup
    grade_thresholds = [
        (QUALITY.grade_a_threshold, "A"),
        (QUALITY.grade_b_threshold, "B"),
        (QUALITY.grade_c_threshold, "C"),
        (QUALITY.grade_d_threshold, "D"),
    ]
    grade = next((g for thresh, g in grade_thresholds if final_score >= thresh), "F")

    return {
        "utility_score": round(final_score, 1),
        "utility_grade": grade,
        "primary_weakness": weakness,
        "description_score": round(description_score, 1),
        "keyword_score": round(keyword_score, 1),
        "description_word_count": int(description_quality.get("description_word_count", 0)),
        "keyword_term_count": int(keyword_quality.get("keyword_term_count", 0)),
        "keyword_unique_terms": int(keyword_quality.get("keyword_unique_terms", 0)),
        "keyword_duplication_ratio": float(
            keyword_quality.get("keyword_duplication_ratio", 0.0),
        ),
        "keyword_category_coverage": float(keyword_quality.get("keyword_category_coverage", 0.0)),
    }


def _extract_metadata_baseline_text(context: str | None) -> str | None:
    """Extract compact baseline text from prompt context metadata hints.

    The prompt context may include wrapper labels (for example ``Title hint:``).
    This helper strips wrappers so baseline utility is scored on metadata content.
    """
    if not context:
        return None

    label_prefixes: tuple[str, ...] = (
        "title hint:",
        "description hint:",
        "keyword hints:",
        "title:",
        "description:",
        "keywords:",
    )
    extracted_lines: list[str] = []
    for raw_line in context.splitlines():
        line = raw_line.strip().lstrip("-").strip()
        if not line:
            continue
        if line.lower().startswith("existing metadata hints"):
            continue
        line_lower = line.lower()
        if line_lower.startswith(("capture metadata:", "capture metadata hints:")):
            continue
        for prefix in label_prefixes:
            if line_lower.startswith(prefix):
                line = line[len(prefix) :].strip()
                break
        if line:
            extracted_lines.append(line)

    merged = " ".join(extracted_lines).strip()
    return merged or None


def _compute_metadata_baseline_utility(context: str | None) -> tuple[float, str] | None:
    """Compute baseline utility score/grade from existing image metadata context."""
    baseline_text = _extract_metadata_baseline_text(context)
    if not baseline_text:
        return None
    baseline_utility = compute_cataloging_utility(baseline_text, None)
    return (
        float(baseline_utility["utility_score"]),
        str(baseline_utility["utility_grade"]),
    )


def _build_metadata_scoring_inputs(
    metadata: MetadataDict | None,
) -> tuple[str | None, str | None, tuple[str, ...], tuple[str, ...]]:
    """Extract trusted reference fields and nonvisual metadata terms for scoring."""
    if not metadata:
        return None, None, (), ()

    raw_title = metadata.get("title")
    reference_title = (
        raw_title.strip() if isinstance(raw_title, str) and raw_title.strip() else None
    )

    raw_description = metadata.get("description")
    reference_description = (
        raw_description.strip()
        if isinstance(raw_description, str) and raw_description.strip()
        else None
    )

    raw_keywords = metadata.get("keywords") or ""
    keyword_terms = _split_catalog_keywords(raw_keywords) if isinstance(raw_keywords, str) else []
    trusted_keywords, keyword_nonvisual_terms = _partition_hint_terms(keyword_terms)

    nonvisual_terms: list[str] = list(keyword_nonvisual_terms)
    for free_text in filter(None, (reference_title, reference_description)):
        _trusted_terms, nonvisual_hint_terms = _partition_hint_terms(
            _extract_hint_signal_terms(free_text),
        )
        nonvisual_terms.extend(nonvisual_hint_terms)

    for field_name in ("date", "time", "gps"):
        raw_value = metadata.get(field_name)
        if not isinstance(raw_value, str):
            continue
        cleaned_value = raw_value.strip()
        if not cleaned_value:
            continue
        nonvisual_terms.append(cleaned_value)
        nonvisual_terms.extend(_extract_hint_signal_terms(cleaned_value))

    deduped_nonvisual_terms = _dedupe_preserve_order(nonvisual_terms)
    filtered_title = _strip_nonvisual_terms_from_text(
        reference_title or "", deduped_nonvisual_terms
    )
    filtered_description = _strip_nonvisual_terms_from_text(
        reference_description or "",
        deduped_nonvisual_terms,
    )

    return (
        filtered_title or None,
        filtered_description or None,
        tuple(_dedupe_preserve_order(trusted_keywords)),
        tuple(deduped_nonvisual_terms),
    )


def _score_metadata_title(
    generated_sections: dict[str, str],
    reference_title: str | None,
) -> tuple[float, set[str], set[str]]:
    """Score recall of trusted title terms in the generated title field."""
    if not reference_title:
        return 0.0, set(), set()

    generated_title = generated_sections.get("title", "").strip()
    reference_terms, _nonvisual_terms = _partition_hint_terms(
        _extract_hint_signal_terms(reference_title),
    )
    normalized_reference_terms = {
        _normalize_phrase_for_matching(term)
        for term in reference_terms
        if _normalize_phrase_for_matching(term)
    }
    if not normalized_reference_terms:
        normalized_reference = _normalize_phrase_for_matching(reference_title)
        if normalized_reference:
            normalized_reference_terms = {normalized_reference}

    if not generated_title or not normalized_reference_terms:
        return 0.0, set(), normalized_reference_terms

    normalized_generated = _normalize_phrase_for_matching(generated_title)
    normalized_reference_title = _normalize_phrase_for_matching(reference_title)
    if normalized_reference_title and normalized_reference_title == normalized_generated:
        return 100.0, normalized_reference_terms, set()

    matched_terms = {
        term
        for term in normalized_reference_terms
        if term and _context_term_present(term, normalized_generated)
    }
    recall = len(matched_terms) / len(normalized_reference_terms)
    return round(recall * 100.0, 1), matched_terms, normalized_reference_terms - matched_terms


def _score_metadata_description(
    generated_sections: dict[str, str],
    reference_terms: tuple[str, ...],
) -> tuple[float, set[str], set[str]]:
    """Score recall of salient metadata terms in the generated description field."""
    normalized_reference_terms = {
        _normalize_phrase_for_matching(term)
        for term in reference_terms
        if _normalize_phrase_for_matching(term)
    }
    if not normalized_reference_terms:
        return 0.0, set(), set()

    generated_description = generated_sections.get("description", "").strip()
    if not generated_description:
        return 0.0, set(), normalized_reference_terms

    normalized_generated = _normalize_phrase_for_matching(generated_description)
    matched_terms = {
        term
        for term in normalized_reference_terms
        if term and _context_term_present(term, normalized_generated)
    }
    recall = len(matched_terms) / len(normalized_reference_terms)
    return round(recall * 100.0, 1), matched_terms, normalized_reference_terms - matched_terms


def _score_metadata_keywords(
    generated_sections: dict[str, str],
    reference_keywords: tuple[str, ...],
) -> tuple[float, set[str], set[str]]:
    """Score F1 overlap between generated keywords and metadata keywords."""
    normalized_reference_terms = {
        _normalize_phrase_for_matching(term)
        for term in reference_keywords
        if _normalize_phrase_for_matching(term)
    }
    if not normalized_reference_terms:
        return 0.0, set(), set()

    generated_keyword_text = generated_sections.get("keywords", "").strip()
    if not generated_keyword_text:
        return 0.0, set(), normalized_reference_terms

    generated_keywords = _dedupe_preserve_order(_split_catalog_keywords(generated_keyword_text))
    normalized_generated_terms = {
        _normalize_phrase_for_matching(term)
        for term in generated_keywords
        if _normalize_phrase_for_matching(term)
    }
    if not normalized_generated_terms:
        return 0.0, set(), normalized_reference_terms

    matched_terms = normalized_reference_terms & normalized_generated_terms
    precision = len(matched_terms) / len(normalized_generated_terms)
    recall = len(matched_terms) / len(normalized_reference_terms)
    f1_score = (
        0.0 if (precision + recall) == 0.0 else (2 * precision * recall) / (precision + recall)
    )
    return round(f1_score * 100.0, 1), matched_terms, normalized_reference_terms - matched_terms


def _score_nonvisual_metadata_leakage(
    text: str,
    metadata: MetadataDict | None,
) -> tuple[float, tuple[str, ...]]:
    """Penalize reuse of nonvisual metadata such as date/time/GPS strings."""
    if not text or not metadata:
        return 0.0, ()

    _reference_title, _reference_description, _reference_keywords, nonvisual_terms = (
        _build_metadata_scoring_inputs(metadata)
    )
    if not nonvisual_terms:
        return 0.0, ()

    text_lower = text.casefold()
    normalized_text = _normalize_phrase_for_matching(text)
    hits = _dedupe_preserve_order(
        [
            term
            for term in nonvisual_terms
            if term.casefold() in text_lower or _context_term_present(term, normalized_text)
        ],
    )
    if not hits:
        return 0.0, ()

    scaled_penalty = (min(len(hits), 3) / 3) * QUALITY.metadata_agreement_nonvisual_penalty
    return round(scaled_penalty, 1), tuple(hits)


def compute_metadata_agreement(
    text: str,
    metadata: MetadataDict | None,
) -> MetadataAgreementMetrics:
    """Compute post-generation agreement against trusted image metadata."""
    if not text.strip() or not metadata:
        return MetadataAgreementMetrics()

    reference_title, reference_description, reference_keywords, _nonvisual_terms = (
        _build_metadata_scoring_inputs(metadata)
    )
    generated_sections = _extract_catalog_sections(text)

    reference_terms_display: list[str] = []
    for free_text in filter(None, (reference_title, reference_description)):
        trusted_terms, _nonvisual_hint_terms = _partition_hint_terms(
            _extract_hint_signal_terms(free_text),
        )
        reference_terms_display.extend(trusted_terms)
    reference_terms_display.extend(reference_keywords)
    reference_terms_display = _dedupe_preserve_order(reference_terms_display)

    title_score, title_matched, _title_missed = _score_metadata_title(
        generated_sections,
        reference_title,
    )
    description_score, description_matched, _description_missed = _score_metadata_description(
        generated_sections,
        tuple(reference_terms_display),
    )
    keyword_score, keyword_matched, _keyword_missed = _score_metadata_keywords(
        generated_sections,
        reference_keywords,
    )
    nonvisual_penalty, nonvisual_hits = _score_nonvisual_metadata_leakage(text, metadata)

    weighted_components: list[tuple[float, float]] = []
    if reference_title:
        weighted_components.append((title_score, QUALITY.metadata_agreement_weight_title))
    if reference_terms_display:
        weighted_components.append(
            (description_score, QUALITY.metadata_agreement_weight_description),
        )
    if reference_keywords:
        weighted_components.append((keyword_score, QUALITY.metadata_agreement_weight_keywords))

    weighted_score = 0.0
    if weighted_components:
        weight_total = sum(weight for _score, weight in weighted_components)
        weighted_score = sum(score * weight for score, weight in weighted_components) / weight_total

    matched_normalized_terms = title_matched | description_matched | keyword_matched
    matched_terms = tuple(
        term
        for term in reference_terms_display
        if _normalize_phrase_for_matching(term) in matched_normalized_terms
    )
    missed_terms = tuple(
        term
        for term in reference_terms_display
        if _normalize_phrase_for_matching(term) not in matched_normalized_terms
    )

    return MetadataAgreementMetrics(
        overall_score=round(max(0.0, min(100.0, weighted_score - nonvisual_penalty)), 1),
        title_score=title_score,
        description_score=description_score,
        keyword_score=keyword_score,
        nonvisual_penalty=nonvisual_penalty,
        matched_terms=matched_terms,
        missed_terms=missed_terms,
        nonvisual_hits=nonvisual_hits,
    )


@dataclass(frozen=True)
class GenerationQualityAnalysis:
    """Analysis results for generated text quality.

    Consolidates all quality checks into a single structured result.
    Each check detects a different failure mode:

    - Repetition: Model stuck in a loop outputting same tokens
    - Hallucination: Fabricated structures (tables, code) not in the image
    - Verbosity: Excessive meta-commentary instead of content
    - Formatting: HTML/Markdown artifacts breaking output display
    - Context ignorance: Output doesn't reference provided context
    - Refusal: Model declines to answer (capability or safety)
    - Generic output: Boilerplate without image-specific content
    - Language mixing: Unexpected language/script switches
    - Degeneration: Garbage characters, encoding corruption
    - Fabrication: Hallucinated specific details (dates, names, stats)
    - Harness issues: mlx-vlm integration bugs, not model quality problems
    """

    is_repetitive: bool
    repeated_token: str | None
    hallucination_issues: list[str]
    is_verbose: bool
    formatting_issues: list[str]
    has_excessive_bullets: bool
    bullet_count: int
    is_context_ignored: bool
    missing_context_terms: list[str]
    is_refusal: bool
    refusal_type: str | None
    is_generic: bool
    specificity_score: float
    has_language_mixing: bool
    language_mixing_issues: list[str]
    has_degeneration: bool
    degeneration_type: str | None
    has_fabrication: bool
    fabrication_issues: list[str]
    missing_sections: list[str] = dataclass_field(default_factory=list)
    title_word_count: int | None = None
    description_sentence_count: int | None = None
    keyword_count: int | None = None
    keyword_duplication_ratio: float | None = None
    has_reasoning_leak: bool = False
    reasoning_leak_markers: list[str] = dataclass_field(default_factory=list)
    has_context_echo: bool = False
    context_echo_ratio: float = 0.0
    # Harness issues indicate mlx-vlm integration bugs, not model quality problems
    has_harness_issue: bool = False
    harness_issue_type: str | None = None
    harness_issue_details: list[str] = dataclass_field(default_factory=list)
    # Lightweight metrics useful for JSONL/report triage
    word_count: int = 0
    unique_ratio: float = 0.0
    prompt_checks_ran: bool = False
    instruction_echo: bool = False
    metadata_borrowing: bool = False
    hint_relationship: str = "preserves_trusted_hints"
    verdict: str = "clean"
    owner: str = "model"
    user_bucket: str = "recommended"
    evidence: list[str] = dataclass_field(default_factory=list)
    likely_capped: bool = False
    requested_max_tokens: int | None = None
    prompt_tokens_total: int | None = None
    prompt_tokens_text_est: int | None = None
    prompt_tokens_nontext_est: int | None = None

    @property
    def has_title_length_violation(self) -> bool:
        """Return True when title word count violates configured bounds."""
        return self.title_word_count is not None and not (
            QUALITY.min_title_words <= self.title_word_count <= QUALITY.max_title_words
        )

    @property
    def has_description_sentence_violation(self) -> bool:
        """Return True when description sentence count violates configured bounds."""
        return self.description_sentence_count is not None and not (
            QUALITY.min_description_sentences
            <= self.description_sentence_count
            <= QUALITY.max_description_sentences
        )

    @property
    def has_keyword_count_violation(self) -> bool:
        """Return True when keyword count violates configured bounds."""
        return self.keyword_count is not None and not (
            QUALITY.min_keywords_count <= self.keyword_count <= QUALITY.max_keywords_count
        )

    @property
    def has_keyword_duplication_violation(self) -> bool:
        """Return True when keyword duplication exceeds configured threshold."""
        return (
            self.keyword_duplication_ratio is not None
            and self.keyword_duplication_ratio >= QUALITY.keyword_duplication_ratio_threshold
        )

    def has_any_issues(self) -> bool:
        """Return True if any quality issues were detected."""
        has_contract_violation = (
            bool(self.missing_sections)
            or self.has_title_length_violation
            or self.has_description_sentence_violation
            or self.has_keyword_count_violation
            or self.has_keyword_duplication_violation
        )
        return (
            self.is_repetitive
            or bool(self.hallucination_issues)
            or self.is_verbose
            or bool(self.formatting_issues)
            or self.has_excessive_bullets
            or self.is_context_ignored
            or self.is_refusal
            or self.is_generic
            or self.has_language_mixing
            or self.has_degeneration
            or self.has_fabrication
            or has_contract_violation
            or self.has_reasoning_leak
            or self.has_context_echo
            or self.has_harness_issue
            or self.instruction_echo
            or self.metadata_borrowing
            or self.verdict != "clean"
        )

    @property
    def issues(self) -> list[str]:
        """Return a list of all detected quality issues as human-readable strings."""
        issues_list: list[str] = []
        keyword_duplication_label = (
            f"Keyword duplication ({self.keyword_duplication_ratio:.0%} duplicated terms)"
            if self.keyword_duplication_ratio is not None
            else "Keyword duplication"
        )

        scalar_issues = [
            (
                self.is_repetitive,
                f"Repetitive output ({self.repeated_token})",
            ),
            (self.is_verbose, "Excessive verbosity"),
            (self.has_excessive_bullets, f"Excessive bullet points ({self.bullet_count})"),
            (
                self.is_context_ignored,
                f"Context ignored (missing: {', '.join(self.missing_context_terms)})",
            ),
            (self.is_refusal, f"Refusal detected ({self.refusal_type})"),
            (
                self.is_generic,
                f"Generic output (specificity: {self.specificity_score:.2f})",
            ),
            (self.has_degeneration, f"Output degeneration ({self.degeneration_type})"),
            (bool(self.missing_sections), f"Missing sections ({', '.join(self.missing_sections)})"),
            (
                self.has_title_length_violation,
                "Title length violation "
                f"({self.title_word_count} words; expected "
                f"{QUALITY.min_title_words}-{QUALITY.max_title_words})",
            ),
            (
                self.has_description_sentence_violation,
                "Description sentence violation "
                f"({self.description_sentence_count}; expected "
                f"{QUALITY.min_description_sentences}-{QUALITY.max_description_sentences})",
            ),
            (
                self.has_keyword_count_violation,
                "Keyword count violation "
                f"({self.keyword_count}; expected "
                f"{QUALITY.min_keywords_count}-{QUALITY.max_keywords_count})",
            ),
            (
                self.has_keyword_duplication_violation,
                keyword_duplication_label,
            ),
            (
                self.has_reasoning_leak,
                (
                    f"Reasoning leak ({', '.join(self.reasoning_leak_markers[:2])})"
                    if self.reasoning_leak_markers
                    else "Reasoning leak"
                ),
            ),
            (self.has_context_echo, f"Context echo ({self.context_echo_ratio:.0%} overlap)"),
            (self.instruction_echo, "Instruction echo"),
            (self.metadata_borrowing, "Nonvisual metadata borrowing"),
            (self.likely_capped, "Likely capped by max token budget"),
            (
                self.hint_relationship == "ignores_trusted_hints",
                "Ignores trusted hints",
            ),
            (
                self.hint_relationship == "degrades_trusted_hints",
                "Degrades trusted hints",
            ),
        ]
        issues_list.extend(label for condition, label in scalar_issues if condition)
        issues_list.extend(self.hallucination_issues)
        issues_list.extend(self.formatting_issues)
        issues_list.extend(self.language_mixing_issues)
        issues_list.extend(self.fabrication_issues)

        # Harness issues (prominently marked as integration problems)
        if self.has_harness_issue:
            harness_label = f"⚠️HARNESS:{self.harness_issue_type}"
            issues_list.insert(0, harness_label)  # Put at front for visibility
            issues_list.extend(self.harness_issue_details)
        if self.verdict == "cutoff":
            issues_list.insert(0, "⚠️REVIEW:cutoff")
        elif self.verdict == "context_budget":
            issues_list.insert(0, "⚠️REVIEW:context_budget")
        return issues_list


@dataclass(frozen=True)
class PromptQualitySignals:
    """Prompt-aware quality signals derived from trusted hints and contract checks."""

    prompt_bundle: TrustedHintBundle
    is_context_ignored: bool = False
    missing_context_terms: tuple[str, ...] = ()
    has_reasoning_leak: bool = False
    reasoning_leak_markers: tuple[str, ...] = ()
    has_context_echo: bool = False
    context_echo_ratio: float = 0.0
    instruction_echo: bool = False
    instruction_markers: tuple[str, ...] = ()
    metadata_borrowing: bool = False
    borrowed_metadata_terms: tuple[str, ...] = ()
    missing_sections: tuple[str, ...] = ()
    title_word_count: int | None = None
    description_sentence_count: int | None = None
    keyword_count: int | None = None
    keyword_duplication_ratio: float | None = None


@dataclass(frozen=True)
class HarnessQualitySignals:
    """Harness and integration signals derived from generated text."""

    has_harness_issue: bool
    harness_type: str | None
    harness_issues: tuple[str, ...]


def _collect_prompt_quality_signals(
    text: str,
    *,
    generated_tokens: int,
    prompt: str | None,
    context_marker: str,
) -> PromptQualitySignals:
    """Collect prompt-aware trusted-hint and contract signals."""
    has_reasoning_leak, reasoning_leak_markers = _detect_reasoning_leakage(text)
    instruction_echo, instruction_markers = _detect_instruction_echo(text)
    if not prompt:
        return PromptQualitySignals(
            prompt_bundle=TrustedHintBundle(),
            has_reasoning_leak=has_reasoning_leak,
            reasoning_leak_markers=tuple(reasoning_leak_markers),
            instruction_echo=instruction_echo,
            instruction_markers=tuple(instruction_markers),
        )

    prompt_bundle = _extract_trusted_hint_bundle(prompt, context_marker=context_marker)
    is_context_ignored, missing_context_terms = _detect_context_ignorance(
        text,
        prompt,
        context_marker=context_marker,
    )
    has_context_echo, context_echo_ratio = _detect_context_echo(
        text,
        prompt,
        context_marker=context_marker,
    )
    metadata_borrowing, borrowed_metadata_terms = _detect_metadata_borrowing(
        text,
        prompt_bundle,
    )
    missing_sections: list[str] = []
    title_word_count: int | None = None
    description_sentence_count: int | None = None
    keyword_count: int | None = None
    keyword_duplication_ratio: float | None = None
    if generated_tokens >= QUALITY.min_tokens_for_substantial and _prompt_requests_catalog_contract(
        prompt
    ):
        (
            missing_sections,
            title_word_count,
            description_sentence_count,
            keyword_count,
            keyword_duplication_ratio,
        ) = _analyze_catalog_contract(text)

    return PromptQualitySignals(
        prompt_bundle=prompt_bundle,
        is_context_ignored=is_context_ignored,
        missing_context_terms=tuple(missing_context_terms),
        has_reasoning_leak=has_reasoning_leak,
        reasoning_leak_markers=tuple(reasoning_leak_markers),
        has_context_echo=has_context_echo,
        context_echo_ratio=context_echo_ratio,
        instruction_echo=instruction_echo,
        instruction_markers=tuple(instruction_markers),
        metadata_borrowing=metadata_borrowing,
        borrowed_metadata_terms=tuple(borrowed_metadata_terms),
        missing_sections=tuple(missing_sections),
        title_word_count=title_word_count,
        description_sentence_count=description_sentence_count,
        keyword_count=keyword_count,
        keyword_duplication_ratio=keyword_duplication_ratio,
    )


def _collect_harness_quality_signals(
    text: str,
    *,
    generated_tokens: int,
    prompt_tokens: int | None,
    is_repetitive: bool,
    is_context_ignored: bool,
    is_refusal: bool,
) -> HarnessQualitySignals:
    """Collect harness and integration failure signals."""
    harness_issues: list[str] = []
    harness_type: str | None = None

    has_encoding_issue, encoding_type = _detect_token_encoding_issues(text)
    if has_encoding_issue and encoding_type:
        harness_type = harness_type or "encoding"
        harness_issues.append(f"token_encoding:{encoding_type}")

    has_token_leak, leaked_tokens = _detect_special_token_leakage(text)
    if has_token_leak:
        harness_type = harness_type or "stop_token"
        harness_issues.extend([f"token_leak:{tok}" for tok in leaked_tokens[:3]])

    has_minimal, minimal_type = _detect_minimal_output(
        text,
        generated_tokens,
        prompt_tokens=prompt_tokens,
    )
    if has_minimal and minimal_type:
        harness_type = harness_type or "prompt_template"
        harness_issues.append(f"output:{minimal_type}")

    has_long_context_breakdown, long_context_issue = _detect_long_context_breakdown(
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        text=text,
        is_repetitive=is_repetitive,
        is_context_ignored=is_context_ignored,
        is_refusal=is_refusal,
    )
    if has_long_context_breakdown and long_context_issue:
        if harness_type in {None, "prompt_template"}:
            harness_type = "long_context"
        harness_issues.append(long_context_issue)

    has_training_leak, leak_type = _detect_training_data_leak(text)
    if has_training_leak and leak_type:
        harness_type = harness_type or "generation_loop"
        harness_issues.append(f"training_leak:{leak_type}")

    return HarnessQualitySignals(
        has_harness_issue=bool(harness_issues),
        harness_type=harness_type,
        harness_issues=tuple(harness_issues),
    )


def analyze_generation_text(
    text: str,
    generated_tokens: int,
    prompt_tokens: int | None = None,
    prompt: str | None = None,
    requested_max_tokens: int | None = None,
    context_marker: str = "Context:",
) -> GenerationQualityAnalysis:
    """Analyze generated text for quality issues.

    Consolidates all quality detection logic into a single function to avoid
    duplication between preview and verbose output modes.

    Args:
        text: Generated text to analyze
        generated_tokens: Number of tokens generated
        prompt_tokens: Number of prompt/prefill tokens (if available)
        prompt: Optional prompt text for context ignorance detection
        requested_max_tokens: Requested max generation tokens for cutoff detection
        context_marker: Marker for context section in prompt

    Returns:
        GenerationQualityAnalysis with all detected issues
    """
    is_repetitive, repeated_token = _detect_repetitive_output(text)
    hallucination_issues = _detect_hallucination_patterns(text)
    is_verbose = _detect_excessive_verbosity(text, generated_tokens)
    formatting_issues = _detect_formatting_violations(text)
    has_excessive_bullets, bullet_count = _detect_excessive_bullets(text)
    prompt_signals = _collect_prompt_quality_signals(
        text,
        generated_tokens=generated_tokens,
        prompt=prompt,
        context_marker=context_marker,
    )

    # Refusal detection: model declines to answer
    is_refusal, refusal_type = _detect_refusal_patterns(text)

    # Generic output: boilerplate without image-specific content
    is_generic, specificity_score = _detect_generic_output(text)

    # Language mixing: unexpected language/script switches
    has_language_mixing, language_mixing_issues = _detect_language_mixing(text)

    # Output degeneration: rubbish characters, encoding corruption
    has_degeneration, degeneration_type = _detect_output_degeneration(text)

    # Fabrication: hallucinated specific details
    has_fabrication, fabrication_issues = _detect_fabricated_details(text)

    harness_signals = _collect_harness_quality_signals(
        text,
        generated_tokens=generated_tokens,
        prompt_tokens=prompt_tokens,
        is_repetitive=is_repetitive,
        is_context_ignored=prompt_signals.is_context_ignored,
        is_refusal=is_refusal,
    )

    _ttr, unique_words, total_words = compute_vocabulary_diversity(text)
    unique_ratio = unique_words / total_words if total_words else 0.0
    prompt_tokens_text_est = _estimate_prompt_tokens_from_text(prompt)
    prompt_tokens_nontext_est = (
        max(prompt_tokens - prompt_tokens_text_est, 0)
        if (prompt_tokens is not None and prompt_tokens_text_est is not None)
        else None
    )
    likely_capped, cutoff_reasons = _detect_likely_cutoff(
        text,
        generated_tokens=generated_tokens,
        requested_max_tokens=requested_max_tokens,
        is_repetitive=is_repetitive,
        missing_sections=prompt_signals.missing_sections,
    )
    utility_context = prompt_signals.prompt_bundle.trusted_text or None
    utility_grade = str(compute_cataloging_utility(text, utility_context)["utility_grade"])
    hint_relationship = "preserves_trusted_hints"
    hint_evidence: list[str] = []
    if prompt_signals.prompt_bundle.trusted_text:
        hint_relationship, hint_evidence = _classify_hint_relationship(
            text,
            prompt_signals.prompt_bundle,
        )
    verdict, review_evidence = _classify_review_verdict(
        has_harness_issue=harness_signals.has_harness_issue,
        harness_type=harness_signals.harness_type,
        likely_cutoff=likely_capped,
        cutoff_reasons=cutoff_reasons,
        prompt_tokens_total=prompt_tokens,
        prompt_tokens_text_est=prompt_tokens_text_est,
        prompt_tokens_nontext_est=prompt_tokens_nontext_est,
        missing_sections=prompt_signals.missing_sections,
        utility_grade=utility_grade,
        instruction_echo=prompt_signals.instruction_echo,
        metadata_borrowing=prompt_signals.metadata_borrowing,
        has_hallucination=bool(hallucination_issues),
    )
    # Promote clean+F to unknown_runtime_anomaly when output is near-empty,
    # signalling an unexplained failure that warrants manual triage.
    if (
        verdict == "clean"
        and utility_grade == "F"
        and len(text.strip()) < QUALITY.anomaly_min_output_chars
    ):
        verdict = "unknown_runtime_anomaly"
        review_evidence.append("utility:F")
    has_contract_issue = bool(prompt_signals.missing_sections) or any(
        (
            prompt_signals.title_word_count is not None
            and not (
                QUALITY.min_title_words
                <= prompt_signals.title_word_count
                <= QUALITY.max_title_words
            ),
            prompt_signals.description_sentence_count is not None
            and not (
                QUALITY.min_description_sentences
                <= prompt_signals.description_sentence_count
                <= QUALITY.max_description_sentences
            ),
            prompt_signals.keyword_count is not None
            and not (
                QUALITY.min_keywords_count
                <= prompt_signals.keyword_count
                <= QUALITY.max_keywords_count
            ),
            prompt_signals.keyword_duplication_ratio is not None
            and prompt_signals.keyword_duplication_ratio
            >= QUALITY.keyword_duplication_ratio_threshold,
        ),
    )
    owner = _classify_review_owner(
        harness_type=harness_signals.harness_type,
        failure_owner=None,
    )
    user_bucket = _classify_user_bucket(
        verdict=verdict,
        hint_relationship=hint_relationship,
        has_contract_issue=has_contract_issue,
        utility_grade=utility_grade,
    )
    evidence = _dedupe_preserve_order(
        [
            *review_evidence,
            *cutoff_reasons,
            *hint_evidence,
            *(["instruction_markers"] if prompt_signals.instruction_markers else []),
            *(["metadata_terms"] if prompt_signals.borrowed_metadata_terms else []),
            *(["reasoning_leak"] if prompt_signals.has_reasoning_leak else []),
            *(["context_echo"] if prompt_signals.has_context_echo else []),
            *(["refusal"] if is_refusal else []),
            *(["generic"] if is_generic else []),
            *(["degeneration"] if has_degeneration else []),
            *(["fabrication"] if has_fabrication else []),
        ],
    )

    return GenerationQualityAnalysis(
        is_repetitive=is_repetitive,
        repeated_token=repeated_token,
        hallucination_issues=hallucination_issues,
        is_verbose=is_verbose,
        formatting_issues=formatting_issues,
        has_excessive_bullets=has_excessive_bullets,
        bullet_count=bullet_count,
        is_context_ignored=prompt_signals.is_context_ignored,
        missing_context_terms=list(prompt_signals.missing_context_terms),
        is_refusal=is_refusal,
        refusal_type=refusal_type,
        is_generic=is_generic,
        specificity_score=specificity_score,
        has_language_mixing=has_language_mixing,
        language_mixing_issues=language_mixing_issues,
        has_degeneration=has_degeneration,
        degeneration_type=degeneration_type,
        has_fabrication=has_fabrication,
        fabrication_issues=fabrication_issues,
        missing_sections=list(prompt_signals.missing_sections),
        title_word_count=prompt_signals.title_word_count,
        description_sentence_count=prompt_signals.description_sentence_count,
        keyword_count=prompt_signals.keyword_count,
        keyword_duplication_ratio=prompt_signals.keyword_duplication_ratio,
        has_reasoning_leak=prompt_signals.has_reasoning_leak,
        reasoning_leak_markers=list(prompt_signals.reasoning_leak_markers),
        has_context_echo=prompt_signals.has_context_echo,
        context_echo_ratio=prompt_signals.context_echo_ratio,
        has_harness_issue=harness_signals.has_harness_issue,
        harness_issue_type=harness_signals.harness_type,
        harness_issue_details=list(harness_signals.harness_issues),
        word_count=total_words,
        unique_ratio=round(unique_ratio, 3),
        prompt_checks_ran=bool(prompt),
        instruction_echo=prompt_signals.instruction_echo,
        metadata_borrowing=prompt_signals.metadata_borrowing,
        hint_relationship=hint_relationship,
        verdict=verdict,
        owner=owner,
        user_bucket=user_bucket,
        evidence=evidence,
        likely_capped=likely_capped,
        requested_max_tokens=requested_max_tokens,
        prompt_tokens_total=prompt_tokens,
        prompt_tokens_text_est=prompt_tokens_text_est,
        prompt_tokens_nontext_est=prompt_tokens_nontext_est,
    )


def _analyze_text_quality(
    text: str,
    generated_tokens: int,
    *,
    prompt_tokens: int | None = None,
    prompt: str | None = None,
    requested_max_tokens: int | None = None,
    context_marker: str = "Context:",
) -> tuple[GenerationQualityAnalysis, str | None]:
    """Return structured quality analysis plus the normalized issue label string."""
    analysis = analyze_generation_text(
        text,
        generated_tokens,
        prompt_tokens=prompt_tokens,
        prompt=prompt,
        requested_max_tokens=requested_max_tokens,
        context_marker=context_marker,
    )
    return analysis, _build_quality_issues_string(analysis)


def local_now_str(fmt: str = LOCAL_TIMESTAMP_FORMAT) -> str:
    """Return localized current time as a formatted string.

    Centralizes timestamp formatting so report generators and version info
    stay consistent and makes future changes (e.g. adding UTC or ISO8601
    variants) trivial.
    """
    return datetime.now().astimezone().strftime(fmt)


# Field name patterns for format dispatch
_TIME_FIELDS: frozenset[str] = frozenset(
    {"total_time", "generation_time", "model_load_time"},
)
_BOOLEAN_FLAG_FIELDS: frozenset[str] = frozenset(
    {
        "is_repetitive",
        "is_verbose",
        "has_formatting_issues",
        "has_hallucination_issues",
        "has_excessive_bullets",
        "is_context_ignored",
    },
)


def _coerce_numeric_value(value: object) -> float | None:
    """Return a float for numeric or numeric-string values, else ``None``."""
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped.replace(",", ""))
        except ValueError:
            return None
    return None


def format_field_value(field_name: str, value: MetricValue) -> str:
    """Normalize values for report/log rendering using field-aware numeric rules.

    Handles memory/time/TPS conventions, preserves non-numeric strings as-is,
    and returns ``""`` for null values.
    """
    formatted_value: str
    if value is None:
        formatted_value = ""
    else:
        numeric_value = _coerce_numeric_value(value)
        if numeric_value is not None:
            if field_name.endswith("_memory"):
                formatted_value = _format_memory_value_gb(numeric_value)
            elif field_name.endswith("_tps"):
                formatted_value = _format_tps(numeric_value)
            elif field_name in _TIME_FIELDS:
                formatted_value = _format_time_seconds(numeric_value)
            elif field_name == "quality_score":
                formatted_value = f"{numeric_value:.1f}"
            elif field_name in _BOOLEAN_FLAG_FIELDS:
                formatted_value = "✓" if numeric_value else "-"
            else:
                formatted_value = fmt_num(numeric_value)
        elif isinstance(value, str):
            formatted_value = value
        else:
            formatted_value = str(value)

    return formatted_value


def is_numeric_value(val: object) -> bool:
    """Return True if val can be interpreted as a number."""
    return _coerce_numeric_value(val) is not None


@lru_cache(maxsize=128)
def is_numeric_field(field_name: str) -> bool:
    """Check if a field should be treated as numeric (right-aligned).

    Uses caching to avoid repeated string operations for frequently
    accessed field names during formatting operations.

    Args:
        field_name: Name of the metric field to check

    Returns:
        True if field contains numeric data
    """
    field_lower = field_name.lower()
    return (
        field_name in NUMERIC_FIELD_PATTERNS
        or any(keyword in field_lower for keyword in ("token", "tps", "memory", "time"))
        or field_lower.endswith("_tokens")
    )


# --- Console UI helpers (rules/separators) ---


def get_terminal_width(min_width: int = 60, max_width: int = 120) -> int:
    """Return a clamped terminal width for formatting.

    Uses shutil.get_terminal_size with a sensible fallback; clamps the
    value to avoid excessive lines on very wide terminals and poor display
    on very narrow ones.
    """
    # If an explicit override is set (via --width), prefer it and do not apply
    # per-call max_width limits; still enforce a minimal practical width.
    if WIDTH_OVERRIDE is not None and WIDTH_OVERRIDE > 0:
        return max(min_width, WIDTH_OVERRIDE)
    # Support environment-based override as well (useful in CI): MLX_VLM_WIDTH
    env_width = os.getenv("MLX_VLM_WIDTH")
    if env_width:
        try:
            return max(min_width, int(env_width))
        except ValueError:
            pass
    try:
        width = shutil.get_terminal_size(fallback=(FORMATTING.generation_wrap_width, 24)).columns
    except OSError:
        width = FORMATTING.generation_wrap_width
    return max(min_width, min(width, max_width))


def _log_wrapped_error(label: str, value: str) -> None:
    """Log error with simple formatting for readability."""
    width = get_terminal_width(max_width=100)

    # Label
    logger.error(label, extra={"style_hint": LogStyles.ERROR, "style_prefix": "ERROR"})

    # Content with wrapping and indentation
    cont_indent = "  "
    cont_avail = max(20, width - len(cont_indent))
    lines = value.splitlines() or [""]
    for original_line in lines:
        if not original_line.strip():
            continue
        wrapped = textwrap.wrap(
            original_line,
            width=cont_avail,
            break_long_words=False,
            break_on_hyphens=False,
            replace_whitespace=False,
        ) or [""]
        for wline in wrapped:
            formatted_line = f"{cont_indent}{wline}"
            logger.error(
                formatted_line,
                extra={"style_hint": LogStyles.DETAIL},
            )


def _apply_cli_output_preferences(args: argparse.Namespace) -> None:
    """Apply color and width preferences based on CLI flags.

    - Honors --no-color / --force-color to toggle Rich console styling
    - Applies --width via MLX_VLM_WIDTH env var for child processes too
    """
    if getattr(args, "no_color", False):
        _set_rich_color_enabled(enabled=False)
    elif getattr(args, "force_color", False):
        _set_rich_color_enabled(enabled=True)

    # Width override: prefer CLI value; store in env so subprocesses inherit it
    if getattr(args, "width", None) is not None:
        try:
            os.environ["MLX_VLM_WIDTH"] = str(int(args.width))
        except (TypeError, ValueError):
            # Invalid width -> remove override and fall back to detection
            os.environ.pop("MLX_VLM_WIDTH", None)
        else:
            if getattr(args, "verbose", False):
                logger.debug(
                    "Width override set to %s columns",
                    os.environ.get("MLX_VLM_WIDTH"),
                )


def log_rule(
    width: int = FORMATTING.generation_wrap_width,
    *,
    char: str = "─",
    style: str | None = None,
    bold: bool = False,
    level: int = logging.INFO,
    pre_newline: bool = False,
    post_newline: bool = False,
) -> None:
    """Log a horizontal rule line with optional Rich styling.

    Uses unicode box-drawing characters for better visual separation.
    Keeps a single place for styling separators to ensure consistency.
    """
    if pre_newline:
        logger.log(level, "")

    line = char * max(1, width)
    extra: dict[str, object] = {"style_hint": LogStyles.RULE}
    if bold:
        style = f"bold {style}".strip() if style and "bold" not in style else style or "bold"
    if style:
        extra["style_override"] = style
    logger.log(level, line, extra=extra)

    if post_newline:
        logger.log(level, "")


# --- Utility Functions ---


def get_library_versions() -> LibraryVersionDict:
    """Return versions of key libraries as a dictionary, using None for missing."""

    def _get_version(pkg_name: str, fallback: str | None = None) -> str | None:
        try:
            return version(pkg_name)
        except PackageNotFoundError:
            return fallback

    def _none_if_na(v: str | None) -> str | None:
        if v is None:
            return None
        s = v.strip() if isinstance(v, str) else str(v).strip()
        if not s or s == NOT_AVAILABLE or s.startswith("N/A"):
            return None
        return s

    # Get MLX version (prefer metadata, fallback to module attribute)
    mlx_ver = _get_version("mlx", getattr(mx, "__version__", None))

    # Get MLX-VLM version
    mlx_vlm_ver = _get_version("mlx-vlm", vlm_version if "vlm_version" in globals() else None)

    return {
        "numpy": _none_if_na(_get_version("numpy", numpy_version)),
        "mlx": _none_if_na(mlx_ver),
        "mlx-metal": _none_if_na(_get_version("mlx-metal")),
        "mlx-vlm": _none_if_na(mlx_vlm_ver),
        "mlx-lm": _none_if_na(_get_version("mlx-lm")),
        "huggingface-hub": _none_if_na(_get_version("huggingface-hub", hf_version)),
        "transformers": _none_if_na(_get_version("transformers")),
        "tokenizers": _none_if_na(_get_version("tokenizers")),
        "Pillow": _none_if_na(_get_version("Pillow", pillow_version)),
    }


def _version_components(version_text: str, *, width: int = 4) -> tuple[int, ...]:
    """Convert a version string to fallback numeric components.

    This is used only when PEP 440 parsing fails; prefer ``packaging.version``
    for normal comparisons so dev, rc, and local editable versions sort
    correctly.
    """
    numbers = [int(part) for part in re.findall(r"\d+", version_text)]
    if len(numbers) < width:
        numbers.extend([0] * (width - len(numbers)))
    return tuple(numbers[:width])


def _is_version_at_least(installed: str, minimum: str) -> bool:
    """Return whether ``installed`` satisfies ``minimum`` using PEP 440 semantics."""
    try:
        is_at_least: bool = Version(installed) >= Version(minimum)
    except InvalidVersion:
        return _version_components(installed) >= _version_components(minimum)
    else:
        return is_at_least


def _collect_upstream_requirements(
    versions: LibraryVersionDict,
) -> dict[str, tuple[str, set[str]]]:
    """Collect package floors implied by installed stacks and project policy."""
    requirements: dict[str, tuple[str, set[str]]] = {}

    def _record_requirement(package: str, minimum: str, source_stack: str) -> None:
        current = requirements.get(package)
        if current is None:
            requirements[package] = (minimum, {source_stack})
            return

        current_minimum, current_sources = current
        merged_sources = current_sources | {source_stack}
        if _is_version_at_least(minimum, current_minimum):
            requirements[package] = (minimum, merged_sources)
        else:
            requirements[package] = (current_minimum, merged_sources)

    for package_name, minimum_version in PROJECT_RUNTIME_STACK_MINIMUMS.items():
        _record_requirement(package_name, minimum_version, "check_models")

    if versions.get("mlx-vlm"):
        # mlx-vlm requirements.txt currently specifies:
        #   mlx>=0.31.2, mlx-lm>=0.31.3, transformers>=5.5.0
        for package_name, minimum_version in UPSTREAM_MLX_VLM_MINIMUMS.items():
            _record_requirement(package_name, minimum_version, "mlx-vlm")

    if versions.get("mlx-lm"):
        # mlx-lm setup.py currently specifies:
        #   mlx>=0.31.2, transformers>=5.0.0
        for package_name, minimum_version in UPSTREAM_MLX_LM_MINIMUMS.items():
            _record_requirement(package_name, minimum_version, "mlx-lm")

    return requirements


def _detect_upstream_version_issues(versions: LibraryVersionDict) -> list[str]:
    """Return compatibility issues against current package minimums."""
    issues: list[str] = []
    requirements = _collect_upstream_requirements(versions)

    for package, (minimum, sources) in sorted(requirements.items()):
        source_label = ", ".join(sorted(sources))
        installed = versions.get(package)
        if installed is None:
            issues.append(
                f"{package} is missing; {source_label} expects {package}>={minimum}.",
            )
            continue

        if not _is_version_at_least(installed, minimum):
            issues.append(
                f"{package}=={installed} is below minimum {minimum} required by {source_label}.",
            )

    return issues


def _has_mlx_vlm_load_image_path_bug(source_text: str) -> bool:
    """Detect the known unguarded ``startswith`` branch in mlx-vlm load_image()."""
    has_risky_branch = 'elif image_source.startswith(("http://", "https://"))' in source_text
    has_safe_guard = (
        'elif isinstance(image_source, str) and image_source.startswith(("http://", "https://"))'
        in source_text
    )
    return has_risky_branch and not has_safe_guard


def _resolve_distribution_source_file(distribution_name: str, relative_path: str) -> Path | None:
    """Locate an installed distribution file path without importing the package."""
    try:
        package_distribution = distribution(distribution_name)
    except PackageNotFoundError:
        return None

    direct_candidate = Path(str(package_distribution.locate_file(relative_path)))
    if direct_candidate.is_file():
        return direct_candidate

    normalized_target = relative_path.replace("\\", "/")
    for file_ref in package_distribution.files or []:
        file_path = str(file_ref).replace("\\", "/")
        if not file_path.endswith(normalized_target):
            continue

        candidate = Path(str(package_distribution.locate_file(file_ref)))
        if candidate.is_file():
            return candidate

    module_name = normalized_target.split("/", 1)[0]
    module_spec = find_spec(module_name)
    module_locations = (
        list(module_spec.submodule_search_locations)
        if module_spec and module_spec.submodule_search_locations
        else []
    )
    if module_locations:
        module_root = Path(module_locations[0])
        _, _, module_relative = normalized_target.partition("/")
        if module_relative:
            candidate = module_root / module_relative
            if candidate.is_file():
                return candidate

    return None


def _detect_mlx_vlm_load_image_issue() -> str | None:
    """Detect known mlx-vlm load_image Path/BytesIO branch bug from source."""
    source_path: Path | None = None
    if getattr(load_image, "__module__", "") == "mlx_vlm.utils":
        code_obj = getattr(load_image, "__code__", None)
        if code_obj is not None:
            source_path = Path(code_obj.co_filename)

    if source_path is None:
        source_path = _resolve_distribution_source_file("mlx-vlm", "mlx_vlm/utils.py")
    if source_path is None:
        return None

    try:
        source_text = source_path.read_text(encoding="utf-8")
    except OSError:
        return None

    if _has_mlx_vlm_load_image_path_bug(source_text):
        return (
            "mlx-vlm load_image() has an unguarded URL startswith() branch; "
            "Path/BytesIO inputs can raise AttributeError in upstream code."
        )

    return None


_RUNTIME_API_CALL_CONTRACTS: Final[dict[str, tuple[str, tuple[str, ...]]]] = {
    "load": (
        "mlx_vlm.utils.load",
        ("path_or_hf_repo", "adapter_path", "lazy", "revision", "trust_remote_code"),
    ),
    "apply_chat_template": (
        "mlx_vlm.prompt_utils.apply_chat_template",
        ("processor", "config", "prompt", "num_images"),
    ),
    "generate": (
        "mlx_vlm.generate.generate",
        (
            "model",
            "processor",
            "prompt",
            "image",
            "verbose",
            "temperature",
            "top_p",
            "repetition_penalty",
            "repetition_context_size",
            "max_kv_size",
            "kv_bits",
            "kv_quant_scheme",
            "kv_group_size",
            "quantized_kv_start",
            "max_tokens",
            "min_p",
            "top_k",
            "prefill_step_size",
            "resize_shape",
            "eos_tokens",
            "skip_special_tokens",
            "enable_thinking",
            "thinking_budget",
            "thinking_end_token",
            "thinking_start_token",
        ),
    ),
    "load_image": (
        "mlx_vlm.utils.load_image",
        ("image_source",),
    ),
}
_GENERATION_RESULT_REQUIRED_FIELDS: Final[tuple[str, ...]] = (
    "text",
    "prompt_tokens",
    "generation_tokens",
    "total_tokens",
    "prompt_tps",
    "generation_tps",
    "peak_memory",
)


def _get_callable_contract_issues(
    *,
    qualified_name: str,
    symbol_value: object,
    required_keyword_params: Sequence[str],
) -> list[str]:
    """Return contract issues for a callable surface used by check_models."""
    if symbol_value is _raise_mlx_vlm_missing:
        dependency_message = MISSING_DEPENDENCIES.get("mlx-vlm", ERROR_MLX_VLM_MISSING)
        return [
            f"{qualified_name} is still bound to the missing-dependency placeholder "
            f"({dependency_message}).",
        ]

    if not callable(symbol_value):
        return [f"{qualified_name} is not callable."]

    try:
        signature = inspect.signature(symbol_value)
    except (TypeError, ValueError) as err:
        return [f"{qualified_name} signature could not be inspected for API drift checks: {err}."]

    parameters = signature.parameters
    has_var_keyword = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    missing_keyword_params: list[str] = []
    positional_only_keyword_params: list[str] = []

    for parameter_name in required_keyword_params:
        parameter = parameters.get(parameter_name)
        if parameter is None:
            if not has_var_keyword:
                missing_keyword_params.append(parameter_name)
            continue
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            positional_only_keyword_params.append(parameter_name)

    issues: list[str] = []
    if missing_keyword_params:
        issues.append(
            f"{qualified_name} is missing required keyword parameter(s): "
            f"{', '.join(missing_keyword_params)}.",
        )
    if positional_only_keyword_params:
        issues.append(
            f"{qualified_name} exposes positional-only parameter(s) that check_models passes "
            f"by keyword: {', '.join(positional_only_keyword_params)}.",
        )
    return issues


def _resolve_generation_result_type() -> type[object] | None:
    """Resolve ``mlx_vlm.generate.GenerationResult`` without importing on module load."""
    if generate is _raise_mlx_vlm_missing:
        return None
    try:
        generate_module = __import__("mlx_vlm.generate", fromlist=["GenerationResult"])
    except ImportError:
        return None

    candidate = getattr(generate_module, "GenerationResult", None)
    return candidate if isinstance(candidate, type) else None


def _get_generation_result_contract_issues(result_type: type[object] | None) -> list[str]:
    """Return drift issues for the upstream ``GenerationResult`` shape we consume."""
    if result_type is None:
        if generate is _raise_mlx_vlm_missing:
            return []
        return [
            "mlx_vlm.generate.GenerationResult could not be imported for API drift checks.",
        ]

    field_names: set[str] = set()
    if is_dataclass(result_type):
        field_names.update(field.name for field in fields(result_type))

    raw_annotations = getattr(result_type, "__annotations__", None)
    if isinstance(raw_annotations, dict):
        field_names.update(
            field_name
            for field_name in raw_annotations
            if isinstance(field_name, str) and field_name.strip()
        )

    missing_fields = [
        field_name
        for field_name in _GENERATION_RESULT_REQUIRED_FIELDS
        if field_name not in field_names
    ]
    if not missing_fields:
        return []

    return [
        "mlx_vlm.generate.GenerationResult is missing required field(s): "
        + ", ".join(missing_fields)
        + "."
    ]


def _detect_runtime_api_drift_issues() -> tuple[str, ...]:
    """Return issues when installed MLX runtime call surfaces drift from our contract."""
    issues: list[str] = []
    for symbol_name, (
        qualified_name,
        required_keyword_params,
    ) in _RUNTIME_API_CALL_CONTRACTS.items():
        issues.extend(
            _get_callable_contract_issues(
                qualified_name=qualified_name,
                symbol_value=globals()[symbol_name],
                required_keyword_params=required_keyword_params,
            ),
        )

    issues.extend(_get_generation_result_contract_issues(_resolve_generation_result_type()))
    return tuple(dict.fromkeys(issues))


def _collect_preflight_package_issues(versions: LibraryVersionDict) -> list[str]:
    """Collect actionable dependency/runtime issues before model execution."""
    issues = _detect_upstream_version_issues(versions)

    load_image_issue = _detect_mlx_vlm_load_image_issue()
    if load_image_issue:
        issues.append(load_image_issue)

    issues.extend(_detect_runtime_api_drift_issues())

    return issues


def _get_available_fields(results: list[PerformanceResult]) -> list[str]:
    """Return ordered list of metric field names present across results.

    We skip heavy / long fields (``text``, ``token``, ``logprobs``) to keep
    summary tables concise. Timing fields from ``PerformanceResult`` are
    appended explicitly so they appear in a predictable order if present.
    """
    # Determine GenerationResult fields while excluding raw text/token payloads.
    gen_fields: list[str] = []
    for r in results:
        if r.generation is not None and is_dataclass(r.generation):
            gen_fields = [
                f.name for f in fields(r.generation) if f.name not in ("text", "token", "logprobs")
            ]
            break

    # Combine with PerformanceResult timing fields
    return gen_fields + PERFORMANCE_TIMING_FIELDS


def _get_field_value(result: PerformanceResult, field_name: str) -> MetricValue:
    """Get field value from either GenerationResult or PerformanceResult."""
    if field_name in PERFORMANCE_TIMING_FIELDS:
        return getattr(result, field_name, None)
    return getattr(result.generation, field_name, None) if result.generation else None


def _sort_results_by_time(results: list[PerformanceResult]) -> list[PerformanceResult]:
    """Return results ordered by effective generation time.

    Failed results are placed first (negative inf) to highlight errors,
    followed by successful results sorted by generation time (fastest first).
    """

    def get_time_value(result: PerformanceResult) -> float:
        """Extract time value for sorting, with fallback for failed results."""
        if not result.success:
            return float("-inf")  # Failed results go to the beginning

        # Use the generation_time field from PerformanceResult
        if result.generation_time is not None:
            return result.generation_time

        # Fallback: calculate time from GenerationResult tokens-per-second if available
        if result.generation is not None:
            g_tokens = _generation_int_metric(result.generation, "generation_tokens") or 0
            g_tps = _generation_float_metric(result.generation, "generation_tps") or 0.0
            if g_tps > 0 and g_tokens:
                return g_tokens / g_tps

        return float("inf")  # No timing data available

    return sorted(results, key=get_time_value)


# =============================================================================
# SYSTEM INFO & VERSION DETECTION (Hardware, OS, Dependencies)
# =============================================================================


def _first_system_profiler_entry(
    device_info: SystemProfilerDict | None,
    section_name: str,
) -> SystemProfilerEntry | None:
    """Return the first entry from a system_profiler section when present."""
    if device_info is None:
        return None
    entries = device_info.get(section_name)
    if not entries:
        return None
    return entries[0]


def _normalize_system_profiler_data(payload: object) -> SystemProfilerDict | None:
    """Normalize parsed ``system_profiler -json`` output to the subset used here."""
    raw_payload = _as_str_object_mapping(payload)
    if raw_payload is None:
        return None

    normalized: SystemProfilerDict = {}
    for key, raw_entries in raw_payload.items():
        if not isinstance(raw_entries, list):
            continue

        normalized_entries: list[SystemProfilerEntry] = []
        for raw_entry in raw_entries:
            entry_mapping = _as_str_object_mapping(raw_entry)
            if entry_mapping is not None:
                normalized_entries.append(dict(entry_mapping))
        normalized[key] = normalized_entries

    return normalized


def _mapping_first_text_value(mapping: Mapping[str, object], *keys: str) -> str | None:
    """Return the first non-empty string value found under ``keys``."""
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


@lru_cache(maxsize=1)
def get_device_info() -> SystemProfilerDict | None:
    """Return system_profiler display (GPU) info as dict or None on failure.

    Only invoked on macOS to enrich hardware section. Failures are swallowed
    (we log at debug) so version printing never hard-fails.
    """
    if platform.system() != "Darwin":  # system_profiler is macOS specific
        return None
    try:
        data = subprocess.check_output(
            ["/usr/sbin/system_profiler", "SPDisplaysDataType", "-json"],
            text=True,
            timeout=5,
        )
        return _normalize_system_profiler_data(json.loads(data))
    except (
        subprocess.SubprocessError,
        json.JSONDecodeError,
        FileNotFoundError,
        PermissionError,
    ) as err:
        logger.debug("Could not retrieve GPU information: %s", err)
        return None


def print_version_info(versions: LibraryVersionDict) -> None:
    """Print library versions and system / hardware info.

    Uses get_system_characteristics() to provide consistent output across
    CLI, HTML, and Markdown reports. Errors are swallowed so version
    printing never fails.
    """
    logger.info("--- Library Versions ---")
    version_rows = [[name, ver or ""] for name, ver in sorted(versions.items())]
    if version_rows:
        _log_rich_table(headers=["Library", "Version"], rows=version_rows)

    logger.info(
        "Generated: %s",
        local_now_str(),
    )

    # --- System / hardware information block ---
    try:
        system_info = get_system_characteristics()
        if system_info:
            logger.info("")  # spacer
            logger.info("--- System Information ---")
            _log_rich_table(
                headers=["Property", "Value"],
                rows=[[key, value] for key, value in system_info.items()],
            )
        else:
            logger.debug("No system information available.")
    except (OSError, RuntimeError, ValueError) as err:
        logger.debug("Skipping system info block: %s", err)


# =============================================================================
# IMAGE & EXIF METADATA PROCESSING (File handling, GPS, EXIF extraction)
# =============================================================================


# --- File Handling ---
def find_most_recent_file(folder: PathLike) -> Path | None:
    """Return the most recently modified image file in a folder, or None.

    Scans for regular image files (with supported extensions: .jpg, .jpeg, .png, .webp)
    excluding hidden files starting with '.', and returns the one with the most recent
    modification time.
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        logger.error("Provided path is not a directory: %s", folder_path)
        return None

    try:
        # Find all regular image files, excluding hidden files (starting with '.')
        regular_files = [
            f
            for f in folder_path.iterdir()
            if f.is_file()
            and not f.name.startswith(".")
            and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]

        # Return the most recently modified file, or None if no files found
        most_recent: Path | None = None
        if regular_files:
            most_recent = max(
                regular_files,
                key=lambda f: f.stat().st_mtime,
            )

    except FileNotFoundError:
        logger.exception("Directory not found: %s", folder_path)
        return None
    except PermissionError:
        logger.exception("Permission denied accessing folder: %s", folder_path)
        return None
    except OSError:
        logger.exception("OS error scanning folder %s", folder_path)
        return None

    # Log result and return
    if most_recent:
        logger.debug("Most recent image file found: %s", str(most_recent))
        return most_recent

    logger.debug("No image files found in directory: %s", folder_path)
    return None


def print_image_dimensions(image_path: PathLike) -> None:
    """Print the dimensions and megapixel count of an image file."""
    img_path = Path(image_path)
    try:
        with Image.open(img_path) as img:
            width, height = img.size
            total_pixels = width * height
            logger.info(
                "Image dimensions: %s (%.1f MPixels)",
                f"{width}x{height}",
                total_pixels / MEGAPIXEL_CONVERSION,
            )
    except (FileNotFoundError, UnidentifiedImageError):
        logger.exception("Error with image file %s", img_path)
    except OSError:
        logger.exception("Unexpected error reading image dimensions for %s", img_path)


# --- EXIF & Metadata Handling ---
def _process_exif_subifd(exif_raw: SupportsExifIfd) -> ExifDict:
    out: ExifDict = {}
    try:
        exif_ifd = exif_raw.get_ifd(ExifTags.IFD.Exif)
        if exif_ifd:
            for tag_id, value in exif_ifd.items():
                tag_name = TAGS.get(tag_id, str(tag_id)) if isinstance(tag_id, int) else str(tag_id)
                out[tag_name] = value
    except (KeyError, AttributeError, TypeError):
        logger.warning("Could not extract Exif SubIFD")
    return out


def get_exif_data(image_path: PathLike) -> ExifDict | None:
    """Return decoded EXIF structure or ``None`` if absent.

    Supports both local file paths and URLs. For URLs, downloads the image
    into memory using urllib.request and PIL's file-like object support.

    Multi-pass extraction strategy (kept explicit for robustness / debugging):
        1. IFD0 pass: baseline tags (camera vendor, dimensions, etc.). We *skip*
            pointers to sub directories (Exif / GPS) so we can handle them with
            targeted try/except blocks and continue even if one sub-IFD is corrupt.
        2. Exif SubIFD pass: exposure details, lens, ISO. Failure here should not
            abort the whole extraction—exceptions are logged and ignored.
        3. GPS IFD pass: converted into a nested mapping so later code can decide
            whether/how to stringify. We do not attempt immediate DMS conversion
            here (that happens downstream) to keep responsibilities separate.

    Rationale: real-world photographs often contain partially corrupt EXIF
    segments; failing soft ensures we still display whatever remains.
    """
    image_str = str(image_path)

    # Check if input is a URL (http/https only)
    parsed_url = urlparse(image_str)
    if parsed_url.scheme:
        scheme = parsed_url.scheme.lower()
        if scheme not in {"http", "https"}:
            msg = f"Unsupported URL scheme for image: {parsed_url.scheme}"
            raise ValueError(msg)
        is_url = True
    else:
        is_url = False
    try:
        if is_url:
            # Download URL into memory and open with PIL
            logger.debug("Downloading image from URL for EXIF extraction: %s", image_str)
            with urlopen(  # noqa: S310 - scheme is restricted to http/https above
                image_str,
                timeout=30,
            ) as response:
                img_data = io.BytesIO(response.read())
            img = Image.open(img_data)
        else:
            # Local file path
            img = Image.open(Path(image_path))

        with img:
            exif_raw: Any = img.getexif()
            if not exif_raw:
                logger.warning("No EXIF data found in %s", image_str)
                return None

            # Pass 1: IFD0 — baseline tags (skip SubIFD pointers handled below)
            _ifd0_skip = {ExifTags.Base.ExifOffset, ExifTags.Base.GPSInfo}
            exif_decoded: ExifDict = {
                TAGS.get(tag_id, str(tag_id)): value
                for tag_id, value in exif_raw.items()
                if tag_id not in _ifd0_skip
            }

            # Pass 2: Exif SubIFD (exposure, lens, ISO)
            exif_decoded.update(_process_exif_subifd(cast("SupportsExifIfd", exif_raw)))

            # Pass 3: GPS IFD
            try:
                gps_ifd: Mapping[object, ExifValue] | None = cast(
                    "SupportsExifIfd",
                    exif_raw,
                ).get_ifd(ExifTags.IFD.GPSInfo)
                if isinstance(gps_ifd, dict) and gps_ifd:
                    gps_decoded: GPSDict = {}
                    for gps_tag_id, gps_value in gps_ifd.items():
                        tag_id_int = (
                            gps_tag_id
                            if isinstance(gps_tag_id, int)
                            else int(gps_tag_id)
                            if isinstance(gps_tag_id, str)
                            else None
                        )
                        if tag_id_int is None:
                            gps_key = str(gps_tag_id)
                        else:
                            gps_key = GPSTAGS.get(tag_id_int, str(gps_tag_id))
                            if GPS is not None:
                                try:
                                    gps_key = GPS(tag_id_int).name
                                except ValueError:
                                    gps_key = GPSTAGS.get(tag_id_int, str(gps_tag_id))
                        gps_decoded[gps_key] = gps_value
                    exif_decoded["GPSInfo"] = gps_decoded
            except (KeyError, AttributeError, TypeError) as gps_err:
                logger.warning("Could not extract GPS IFD: %s", gps_err)

            return exif_decoded
    except (FileNotFoundError, UnidentifiedImageError):
        logger.exception("Error reading image: %s", image_str)
    except (OSError, ValueError, URLError) as e:
        logger.debug("Failed to extract EXIF from %s: %s", image_str, e)
    return None


def to_float(val: float | str | None) -> float | None:
    """Convert a value to float if possible, else return None."""
    if val is None:
        return None
    try:
        temp = float(val)
    except (TypeError, ValueError):
        return None
    else:
        return temp


def _convert_gps_coordinate(
    coord: tuple[Any, ...] | list[Any],
) -> tuple[float, float, float] | None:
    """Convert GPS EXIF coordinate to (degrees, minutes, seconds) tuple.

    GPS coordinates in EXIF can be:
    - (deg, min, sec) as floats/ints
    - ((deg_num, deg_den), (min_num, min_den), (sec_num, sec_den)) as rational tuples
    - (deg, decimal_minutes)

    This function normalizes these formats.

    Args:
        coord: Tuple or list of 2 or 3 values representing GPS coordinate

    Returns:
        Tuple of (degrees, minutes, seconds) as floats, or None if conversion fails
    """
    if not isinstance(coord, (tuple, list)):
        return None

    clen = len(coord)
    if clen not in (MIN_GPS_COORD_LEN, MED_GPS_COORD_LEN, MAX_GPS_COORD_LEN):
        return None

    # EXIF values have unpredictable types from camera vendors
    # Use object type to accept anything, then check with isinstance
    def _to_val(v: float | tuple[float, ...] | list[float] | str | object) -> float | None:
        if isinstance(v, (float, int)):
            return float(v)
        if isinstance(v, (tuple, list)) and len(v) == RATIONAL_TUPLE_LEN:
            try:
                # Cast to sequence for type checker - we've verified it's tuple/list
                seq = cast("Sequence[float]", v)
                num = seq[0]
                den = seq[1]
                return num / den if den != 0 else None
            except (ValueError, TypeError):
                return None
        return to_float(str(v))

    components = [_to_val(coord[i]) if i < clen else 0.0 for i in range(3)]

    if any(c is None for c in components[:clen]):
        return None

    return (components[0] or 0.0, components[1] or 0.0, components[2] or 0.0)


def _first_exif_date_value(exif_data: ExifDict) -> ExifValue | None:
    """Return the first populated EXIF date value using configured tag priority."""
    for tag in EXIF_DATE_TAGS:
        if value := exif_data.get(tag):
            return value
    return None


def _parse_exif_local_datetime(exif_date: ExifValue) -> datetime | None:
    """Parse an EXIF date value and convert it to local timezone."""
    exif_text = str(exif_date)
    parsed: datetime | None = None
    for fmt in DATE_FORMATS:
        with suppress(ValueError):
            parsed = datetime.strptime(exif_text, fmt).replace(tzinfo=UTC).astimezone()
            break
    return parsed


def _extract_file_mtime_local(
    img_path: PathLike,
    *,
    log_context: str = "",
) -> datetime | None:
    """Return file mtime as localized datetime, or ``None`` on filesystem errors."""
    try:
        return datetime.fromtimestamp(Path(img_path).stat().st_mtime, tz=UTC).astimezone()
    except OSError as err:
        logger.debug("Could not get file mtime%s: %s", log_context, err)
        return None


def _extract_primary_exif_local_datetime(
    exif_data: ExifDict,
    *,
    warning_message: str,
) -> tuple[datetime | None, str | None]:
    """Return parsed primary EXIF datetime plus the original raw value."""
    exif_date = _first_exif_date_value(exif_data)
    if not exif_date:
        return None, None

    raw_exif_date = str(exif_date)
    try:
        return _parse_exif_local_datetime(exif_date), raw_exif_date
    except (TypeError, UnicodeDecodeError) as err:
        logger.warning(warning_message, err)
        return None, raw_exif_date


def _extract_exif_date(img_path: PathLike, exif_data: ExifDict) -> str | None:
    parsed, raw_exif_date = _extract_primary_exif_local_datetime(
        exif_data,
        warning_message="Could not localize EXIF date: %s",
    )
    return _format_exif_datetime_or_mtime(
        parsed,
        raw_exif_date,
        img_path=img_path,
        fmt="%Y-%m-%d %H:%M:%S %Z",
        fallback_to_raw_exif=True,
    )


def _extract_exif_time(img_path: PathLike, exif_data: ExifDict) -> str | None:
    """Extract local capture time (``HH:MM:SS``) from EXIF or file mtime."""
    parsed, raw_exif_date = _extract_primary_exif_local_datetime(
        exif_data,
        warning_message="Could not extract time from EXIF date: %s",
    )
    return _format_exif_datetime_or_mtime(
        parsed,
        raw_exif_date,
        img_path=img_path,
        fmt="%H:%M:%S",
        log_context=" for time",
    )


def _format_exif_datetime_or_mtime(
    parsed: datetime | None,
    raw_exif_date: str | None,
    *,
    img_path: PathLike,
    fmt: str,
    fallback_to_raw_exif: bool = False,
    log_context: str = "",
) -> str | None:
    """Format parsed EXIF datetime or fall back to file mtime when unavailable."""
    if raw_exif_date is not None:
        if parsed is not None:
            return parsed.strftime(fmt)
        return raw_exif_date if fallback_to_raw_exif else None

    mtime_local = _extract_file_mtime_local(img_path, log_context=log_context)
    return mtime_local.strftime(fmt) if mtime_local is not None else None


def _decode_exif_string(value: bytes | str | None) -> str:
    """Robustly decode EXIF string values, handling encoding prefixes and fallbacks.

    Handles:
    - Byte strings with null terminators.
    - UserComment 8-byte encoding prefixes (ASCII, UNICODE, JIS).
    - Fallback from UTF-8 to Latin-1/CP1252.

    Args:
        value: The raw EXIF value (bytes or str)

    Returns:
        Decoded and sanitized string.
    """
    if not value:
        return ""
    if not isinstance(value, bytes):
        return (
            value.replace("\x00", "").strip()
            if isinstance(value, str)
            else str(value).replace("\x00", "").strip()
        )

    decoded: str = ""
    # Handle UserComment prefixes (fixed 8-byte header)
    if len(value) >= EXIF_USERCOMMENT_PREFIX_LEN:
        prefix = value[:EXIF_USERCOMMENT_PREFIX_LEN]
        data = value[EXIF_USERCOMMENT_PREFIX_LEN:]
        if prefix.startswith(b"ASCII\x00"):
            decoded = data.decode("ascii", errors="replace").replace("\x00", "").strip()
        elif prefix.startswith(b"UNICODE\x00"):
            # UNICODE prefix usually implies UTF-16.
            # We try utf-16 first (which handles BOM).
            for enc in ("utf-16", "utf-16-be", "utf-16-le"):
                try:
                    candidate = data.decode(enc).replace("\x00", "").strip()
                except (UnicodeDecodeError, LookupError):
                    continue
                else:
                    if not candidate:
                        continue
                    # Heuristic: if we have a lot of high-range characters
                    # it might be the wrong endianness.
                    if (
                        any(ord(c) > CJK_IDEOGRAPH_START for c in candidate)
                        and len(candidate) < MOJIBAKE_HEURISTIC_LEN
                    ):
                        continue
                    decoded = candidate
                    break
            else:
                # Fallback for when heuristic skips everything
                with suppress(UnicodeDecodeError, ValueError):
                    decoded = data.decode("utf-16", errors="replace").replace("\x00", "").strip()
        elif prefix.startswith(b"JIS\x00"):
            decoded = data.decode("shift-jis", errors="replace").replace("\x00", "").strip()

    if decoded:
        return decoded

    # Default decoding strategy
    try:
        # Try UTF-8 first
        return value.decode("utf-8").replace("\x00", "").strip()
    except UnicodeDecodeError:
        # Fallback to Latin-1
        return value.decode("latin-1", errors="replace").replace("\x00", "").strip()


def _extract_description(exif_data: ExifDict) -> str | None:
    description = exif_data.get("ImageDescription")
    if description is None:
        return None
    desc = _decode_exif_string(description)
    return desc or None


def _extract_gps_str(gps_info_raw: object | None) -> str | None:
    """Convert EXIF GPS mapping to decimal-degree text with cardinal suffixes.

    Accepts mixed key formats (numeric EXIF ids or tag strings) and returns
    ``"<lat>°N/S, <lon>°E/W"`` when required coordinates are present.
    """
    if not isinstance(gps_info_raw, Mapping):
        return None
    gps_info: GPSDict = {}
    for k, v in gps_info_raw.items():
        if isinstance(k, int):
            tag_name: str = GPSTAGS.get(k, str(k))
        else:
            tag_name = str(k)
        gps_info[tag_name] = v
    lat = gps_info.get("GPSLatitude")
    lat_ref = gps_info.get("GPSLatitudeRef")
    lon = gps_info.get("GPSLongitude")
    lon_ref = gps_info.get("GPSLongitudeRef")
    logger.debug(
        "Extracted GPS fields: lat=%r, lat_ref=%r, lon=%r, lon_ref=%r",
        lat,
        lat_ref,
        lon,
        lon_ref,
    )
    latitude = _convert_gps_coordinate(lat) if lat and lat_ref else None
    longitude = _convert_gps_coordinate(lon) if lon and lon_ref else None
    logger.debug("Converted GPS: latitude=%r, longitude=%r", latitude, longitude)
    if latitude is None or longitude is None:
        logger.debug("GPS conversion failed: latitude or longitude is None.")
        return None

    def dms_to_dd(dms: tuple[float, float, float], ref: str) -> tuple[float, str]:
        """Convert DMS (degrees, minutes, seconds) to unsigned decimal degrees.

        Returns unsigned decimal and normalized cardinal direction (N/S/E/W).
        Display convention: show absolute value with cardinal direction suffix.
        """
        deg, min_, sec = dms
        dd = deg + min_ / 60.0 + sec / 3600.0
        ref_upper = ref.upper()
        return (dd, ref_upper)

    try:
        lat_ref_str: str = _decode_exif_string(lat_ref)
        lon_ref_str: str = _decode_exif_string(lon_ref)
        lat_dd, lat_card = dms_to_dd(latitude, lat_ref_str)
        lon_dd, lon_card = dms_to_dd(longitude, lon_ref_str)
    except (ValueError, AttributeError, TypeError) as err:
        logger.debug("Failed to convert GPS DMS to decimal: %s", err)
        return None
    else:
        # Format with degree symbol and cardinal direction (standard GPS display)
        return f"{lat_dd:.6f}°{lat_card}, {lon_dd:.6f}°{lon_card}"


def _extract_iptc_metadata(image_path: PathLike) -> IPTCMetadata:
    """Extract IPTC/IIM metadata (keywords, caption) from an image.

    Uses Pillow's IptcImagePlugin to read standard IPTC records:
        - (2, 25): Keywords (multi-valued)
        - (2, 120): Caption/Abstract
    """
    try:
        with Image.open(Path(image_path)) as img:
            iptc: Mapping[tuple[int, int], object] | None = IptcImagePlugin.getiptcinfo(img)
            if not iptc:
                return {}

            result: IPTCMetadata = {}

            # Keywords (2, 25) — may be a single bytes or a list of bytes
            raw_keywords_value = iptc.get((2, 25), [])
            raw_keywords: list[object]
            if isinstance(raw_keywords_value, (bytes, bytearray)):
                raw_keywords = [bytes(raw_keywords_value)]
            elif isinstance(raw_keywords_value, list):
                raw_keywords = list(raw_keywords_value)
            else:
                raw_keywords = []
            keywords: list[str] = []
            for kw in raw_keywords:
                decoded = kw.decode("utf-8", errors="replace") if isinstance(kw, bytes) else str(kw)
                if decoded.strip():
                    keywords.append(decoded.strip())
            if keywords:
                result["iptc_keywords"] = keywords

            # IPTC caption/abstract record
            caption_raw = iptc.get((2, 120))
            if isinstance(caption_raw, (bytes, bytearray)):
                caption_str = bytes(caption_raw).decode("utf-8", errors="replace").strip()
                if caption_str:
                    result["iptc_caption"] = caption_str
            elif isinstance(caption_raw, str):
                caption_str = caption_raw.strip()
                if caption_str:
                    result["iptc_caption"] = caption_str

            return result
    except (OSError, ValueError, AttributeError):
        logger.debug("Failed to extract IPTC metadata from %s", image_path)
    return {}


def _is_str_object_mapping(value: object) -> TypeGuard[Mapping[str, object]]:
    """Return True when ``value`` is a mapping with string keys."""
    return isinstance(value, Mapping) and all(isinstance(key, str) for key in value)


def _as_str_object_mapping(value: object) -> Mapping[str, object] | None:
    """Return ``value`` as a string-keyed mapping when possible."""
    return value if _is_str_object_mapping(value) else None


def _require_str_object_mapping(value: object, error_message: str) -> Mapping[str, object]:
    """Return ``value`` as a string-keyed mapping or raise ``TypeError``."""
    mapping = _as_str_object_mapping(value)
    if mapping is None:
        raise TypeError(error_message)
    return mapping


def _nested_mapping_value(
    container: Mapping[str, object] | None,
    key: str,
) -> Mapping[str, object] | None:
    """Return a nested mapping value from a string-keyed metadata mapping."""
    if container is None:
        return None
    return _as_str_object_mapping(container.get(key, {}))


def _metadata_text_value(metadata: Mapping[str, object], key: str) -> str | None:
    """Return a string metadata value when present and correctly typed."""
    value = metadata.get(key)
    return value if isinstance(value, str) else None


def _metadata_keyword_list(metadata: Mapping[str, object], key: str) -> list[str]:
    """Return a string-list metadata value when present and correctly typed."""
    value = metadata.get(key)
    if not isinstance(value, list):
        return []

    keyword_items = [item for item in value if isinstance(item, str)]
    return keyword_items if len(keyword_items) == len(value) else []


def _xmp_alt_text(container: Mapping[str, object] | object, rdf_ns: str) -> str | None:
    """Extract a text value from an XMP rdf:Alt container."""
    container_mapping = _as_str_object_mapping(container)
    if container_mapping is None:
        return None
    alt = _nested_mapping_value(container_mapping, f"{rdf_ns}Alt")
    if alt is None:
        return None
    text = alt.get(f"{rdf_ns}li", "")
    text_mapping = _as_str_object_mapping(text)
    if text_mapping is not None:
        text = text_mapping.get("#text", "")
    return text.strip() if isinstance(text, str) and text.strip() else None


def _extract_xmp_metadata(image_path: PathLike) -> XMPMetadata:
    """Extract XMP metadata (dc:subject keywords, dc:title, dc:description).

    Uses Pillow's ``Image.getxmp()`` (8.2+).  The returned dict is deeply
    nested with XML namespace prefixes; this function navigates defensively.
    Requires the ``defusedxml`` package; returns empty if unavailable.
    """
    if not _defusedxml_available:
        logger.debug("Skipping XMP extraction — defusedxml not installed")
        return {}

    rdf_ns = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
    dc_ns = "{http://purl.org/dc/elements/1.1/}"

    try:
        with Image.open(Path(image_path)) as img:
            if not hasattr(img, "getxmp"):
                return {}
            xmp = _as_str_object_mapping(img.getxmp())
            if not xmp:
                return {}

            result: XMPMetadata = {}

            # Navigate: xmpmeta → RDF → Description
            desc_block = _nested_mapping_value(
                _nested_mapping_value(_nested_mapping_value(xmp, "xmpmeta"), f"{rdf_ns}RDF"),
                f"{rdf_ns}Description",
            )
            if desc_block is None:
                return {}

            # dc:subject → keywords list
            subject = desc_block.get(f"{dc_ns}subject", {})
            subject_mapping = _as_str_object_mapping(subject)
            if subject_mapping is not None:
                bag = _nested_mapping_value(subject_mapping, f"{rdf_ns}Bag")
                if bag is not None:
                    items = bag.get(f"{rdf_ns}li", [])
                    if isinstance(items, str):
                        items = [items]
                    if isinstance(items, list):
                        kw_list = [str(k).strip() for k in items if str(k).strip()]
                        if kw_list:
                            result["xmp_keywords"] = kw_list

            # XMP description (rdf:Alt language alternative)
            description = desc_block.get(f"{dc_ns}description", {})
            desc_text = _xmp_alt_text(description, rdf_ns)
            if desc_text:
                result["xmp_description"] = desc_text

            # XMP title (rdf:Alt language alternative)
            title = desc_block.get(f"{dc_ns}title", {})
            title_text = _xmp_alt_text(title, rdf_ns)
            if title_text:
                result["xmp_title"] = title_text

            return result
    except (OSError, ValueError, AttributeError, TypeError):
        logger.debug("Failed to extract XMP metadata from %s", image_path)
    return {}


def _extract_xp_keywords(exif_data: ExifDict) -> list[str]:
    """Extract Windows XP keywords from EXIF IFD0 (UTF-16LE encoded, semicolon-delimited)."""
    raw = exif_data.get("XPKeywords")
    if raw is None:
        return []
    if isinstance(raw, bytes):
        try:
            decoded = raw.decode("utf-16-le").rstrip("\x00")
        except UnicodeDecodeError:
            decoded = raw.decode("latin-1", errors="replace")
    elif isinstance(raw, str):
        decoded = raw
    else:
        return []
    return [kw.strip() for kw in decoded.split(";") if kw.strip()]


def _merge_keywords(*sources: list[str]) -> str | None:
    """Merge keyword lists from multiple sources, preserving first-seen order."""
    seen: set[str] = set()
    merged: list[str] = []
    for source in sources:
        for kw in source:
            lower = kw.lower()
            if lower not in seen:
                seen.add(lower)
                merged.append(kw)
    return ", ".join(merged) if merged else None


def extract_image_metadata(
    image_path: PathLike,
    *,
    exif_data: ExifDict | None = None,
) -> MetadataDict:
    """Derive high-level metadata (date, description, GPS, keywords, title, raw EXIF).

    Extracts from three metadata families:
        1. EXIF (IFD0 + SubIFD + GPS) via ``get_exif_data``
        2. IPTC/IIM keywords and caption via ``_extract_iptc_metadata``
        3. XMP dc:subject, dc:title, dc:description via ``_extract_xmp_metadata``
        4. Windows XP keywords from EXIF IFD0

    Keywords are merged (deduplicated, order-preserved) across all sources.
    Returns None for unavailable fields instead of sentinel strings.
    """
    metadata: MetadataDict = {}
    img_path = Path(image_path)
    if exif_data is None:
        exif_data = get_exif_data(img_path) or {}

    # IPTC + XMP extraction (separate image opens for header-only access)
    iptc = _extract_iptc_metadata(img_path)
    xmp = _extract_xmp_metadata(img_path)
    iptc_caption = _metadata_text_value(iptc, "iptc_caption")
    xmp_description = _metadata_text_value(xmp, "xmp_description")
    xmp_title = _metadata_text_value(xmp, "xmp_title")
    iptc_keywords = _metadata_keyword_list(iptc, "iptc_keywords")
    xmp_keywords = _metadata_keyword_list(xmp, "xmp_keywords")

    # Date, Time, GPS
    metadata["date"] = _extract_exif_date(img_path, exif_data)
    metadata["time"] = _extract_exif_time(img_path, exif_data)
    metadata["gps"] = _extract_gps_str(exif_data.get("GPSInfo"))

    # Description: prefer IPTC caption → XMP description → EXIF ImageDescription
    description = iptc_caption or xmp_description or _extract_description(exif_data)
    metadata["description"] = description

    # Title: from XMP dc:title (falls back to None)
    metadata["title"] = xmp_title

    # Keywords: merge IPTC → XMP → Windows XP, deduplicated
    metadata["keywords"] = _merge_keywords(
        iptc_keywords,
        xmp_keywords,
        _extract_xp_keywords(exif_data),
    )

    # Raw EXIF for reference
    metadata["exif"] = str(exif_data)
    return metadata


def exif_value_to_str(tag_str: str, value: object) -> str:
    """Convert an EXIF value to a string for display, sanitizing and truncating it."""

    def _sanitize(s: str) -> str:
        """Replace control characters with spaces to prevent breaking table alignment."""
        # This regex targets ASCII control characters, including tabs and newlines.
        return re.sub(r"[\x00-\x1F\x7F-\x9F]", " ", s).strip()

    processed_str: str
    if isinstance(value, bytes):
        # Try UTF-8 first (handles most modern metadata), fallback to latin-1 for legacy
        # This fixes garbled copyright symbols (© showing as Â©) and other Unicode chars
        try:
            processed_str = _sanitize(value.decode("utf-8"))
        except UnicodeDecodeError:
            # Fallback to latin-1 which can decode any byte sequence
            processed_str = _sanitize(value.decode("latin-1", errors="replace"))
        except AttributeError:
            return f"<bytes len={len(value)} un-decodable>"
    elif isinstance(value, tuple | list) and len(value) > MAX_TUPLE_LEN:
        return f"<tuple len={len(value)}>"
    elif isinstance(value, bytearray):
        return f"<bytearray len={len(value)}>"
    else:
        try:
            processed_str = _sanitize(str(value))
        except (TypeError, ValueError) as str_err:
            logger.debug(
                "Could not convert EXIF value for tag '%s' to string: %s",
                tag_str,
                str_err,
            )
            return f"<unrepresentable type: {type(value).__name__}>"

    # Truncate the final sanitized string if it's too long
    if len(processed_str) > MAX_STR_LEN:
        return processed_str[:STR_TRUNCATE_LEN] + "..."
    return processed_str


def filter_and_format_tags(
    exif: ExifDict,
    *,
    show_all: bool = False,
) -> list[tuple[str, str, bool]]:
    """Filter and format EXIF tags for pretty printing."""
    tags: list[tuple[str, str, bool]] = []
    for tag, value in exif.items():
        tag_str: str = str(tag)
        if tag_str == "GPSInfo" and isinstance(value, dict):
            continue
        if isinstance(value, dict):
            logger.debug(
                "Skipping dictionary value for EXIF tag '%s' in pretty print.",
                tag_str,
            )
            continue
        value_str: str = exif_value_to_str(tag_str, value)
        is_important: bool = tag_str in IMPORTANT_EXIF_TAGS
        if show_all or is_important:
            tags.append((tag_str, value_str, is_important))
    return tags


def pretty_print_exif(
    exif: ExifDict,
    *,
    show_all: bool = True,
    title: str = "EXIF Metadata Summary",
) -> None:
    """Render selected EXIF tags in a Rich table.

    Only simple presentation logic lives here; extraction, filtering and
    sanitizing occur earlier (see ``get_exif_data`` / ``filter_and_format_tags``).
    """
    if not exif:
        logger.info("No EXIF data available.")
        return

    tags_to_print: list[tuple[str, str, bool]] = filter_and_format_tags(
        exif,
        show_all=show_all,
    )
    if not tags_to_print:
        log_warning_note("No relevant EXIF tags found to display.")
        return

    table = Table(box=box.ROUNDED, show_lines=False, expand=False)
    table.add_column("Tag", style="bold blue", no_wrap=True)
    table.add_column("Value", overflow="fold")
    for tag_name, value_display, is_important_tag in tags_to_print:
        tag_cell = Text(tag_name, style="bold yellow" if is_important_tag else "")
        table.add_row(tag_cell, value_display)

    header_width: int = max(40, get_terminal_width(max_width=100))

    log_rule(header_width, char="=", style="blue", bold=True)
    logger.info(
        _display_align(title, header_width, alignment="center"),
        extra={"style_hint": LogStyles.HEADER, "style_width": header_width},
    )
    log_rule(header_width, char="=", style="blue", bold=True)

    _log_rich_renderable(table, width=header_width)
    log_rule(header_width, char="=", style="blue", bold=True)


def _format_table_field_value(
    field_name: str,
    res: PerformanceResult,
) -> str:
    """Format one report-table field, including the field-specific shortcuts.

    Args:
        field_name: Table field name. ``model_name`` bypasses generic
            formatting, ``output`` uses the shared preview builder, and
            ``quality_issues`` is truncated after field-aware formatting.
        res: Performance result supplying timing, generation, and issue data

    Returns:
        Formatted string value for the field
    """
    if field_name == "model_name":
        return res.model_name

    if field_name == "output":
        return _build_result_output_preview(
            res,
            max_chars=MAX_OUTPUT_PREVIEW_CHARS,
        )

    value = _get_field_value(res, field_name)
    if field_name == "quality_issues":
        return _truncate_quality_issues(
            format_field_value(field_name, value),
        )

    return format_field_value(field_name, value)


def _build_prepared_table_data(
    *,
    result_set: ResultSet,
    header_separator: str = "<br>",
    include_output: bool = True,
) -> PreparedTableData:
    """Build immutable cached table data from a sorted result set."""
    field_names = ["model_name", *result_set.get_fields()]
    if include_output:
        field_names.append("output")

    headers: list[str] = []
    for field_name in field_names:
        if field_name in FIELD_ABBREVIATIONS:
            line1, line2 = FIELD_ABBREVIATIONS[field_name]
            should_split = line2 and (
                header_separator == "\n"
                or len(line1) > HEADER_SPLIT_LENGTH
                or len(line2) > HEADER_SPLIT_LENGTH
            )
            if should_split:
                headers.append(f"{line1}{header_separator}{line2}")
            elif line2:
                headers.append(f"{line1} {line2}")
            else:
                headers.append(line1)
        else:
            headers.append(format_field_label(field_name))

    rows: list[tuple[str, ...]] = []
    for res in result_set.results:
        row = tuple(_format_table_field_value(field_name, res) for field_name in field_names)
        rows.append(row)

    return PreparedTableData(
        headers=tuple(headers),
        rows=tuple(rows),
        field_names=tuple(field_names),
    )


def _materialize_prepared_table_data(
    table_data: PreparedTableData,
) -> tuple[list[str], list[list[str]], list[str]]:
    """Return mutable copies of cached table data for renderer-specific edits."""
    headers = list(table_data.headers)
    rows = [list(row) for row in table_data.rows]
    field_names = list(table_data.field_names)
    return headers, rows, field_names


def _build_report_render_context(
    *,
    results: list[PerformanceResult],
    prompt: str,
    system_info: dict[str, str] | None = None,
    preflight_issues: Sequence[str] = (),
) -> ReportRenderContext:
    """Build shared derived report data once for all renderers."""
    resolved_results = [
        _populate_result_quality_analysis(
            result,
            prompt=prompt,
        )
        for result in results
    ]
    result_set: ResultSet = ResultSet(resolved_results)
    prompt_context = _extract_trusted_hint_bundle(prompt).trusted_text or None
    summary: ModelIssueSummary = analyze_model_issues(results, prompt_context)
    stats: PerformanceStats = compute_performance_statistics(results)
    resolved_system_info: dict[str, str] = (
        system_info if system_info is not None else get_system_characteristics()
    )
    table_data: PreparedTableData = _build_prepared_table_data(result_set=result_set)
    triage: ReportTriageContext = _build_report_triage_context(
        [result for result in result_set.results if result.success],
        prompt=prompt,
    )
    return ReportRenderContext(
        result_set=result_set,
        table_data=table_data,
        prompt_context=prompt_context,
        summary=summary,
        stats=stats,
        system_info=resolved_system_info,
        triage=triage,
        preflight_issues=tuple(issue for issue in preflight_issues),
    )


def _mark_failed_rows_in_html(
    html_table: str,
    sorted_results: Sequence[PerformanceResult],
) -> str:
    """Annotate HTML result rows with filter metadata and failure styling hooks.

    Only data rows are rewritten. Failed rows also tag the first data cell so the
    CSS contract stays aligned with the row-level status attributes used by the
    client-side filters.
    """
    row_pattern: re.Pattern[str] = re.compile(r"<tr>.*?</tr>", re.DOTALL)
    result_index: int = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal result_index
        row_html = match.group(0)
        if "<td" not in row_html:
            return row_html
        if result_index >= len(sorted_results):
            return row_html

        result = sorted_results[result_index]
        result_index += 1

        if result.success:
            return row_html.replace(
                "<tr>",
                '<tr class="success" data-status="success">',
                1,
            )

        error_stage = html.escape(result.error_stage or "unknown", quote=True)
        error_type = html.escape(result.error_type or "error", quote=True)
        error_package = html.escape(result.error_package or "unknown", quote=True)
        row_html = row_html.replace(
            "<tr>",
            f'<tr class="failed" data-status="failed" '
            f'data-error-stage="{error_stage}" data-error-type="{error_type}" '
            f'data-error-package="{error_package}">',
            1,
        )

        def mark_failed_cell(cell_match: re.Match[str]) -> str:
            return _append_html_class(cell_match.group(0), "failed")

        return re.sub(
            r"<td\b[^>]*>",
            mark_failed_cell,
            row_html,
            count=1,
        )

    return row_pattern.sub(repl, html_table)


def _append_html_class(opening_tag: str, class_name: str) -> str:
    """Append a CSS class to an opening HTML tag, preserving existing classes."""
    class_match = re.search(r'class="([^"]*)"', opening_tag)
    if class_match is None:
        return opening_tag[:-1] + f' class="{class_name}">'

    classes = class_match.group(1).split()
    if class_name in classes:
        return opening_tag

    classes.append(class_name)
    updated_classes = " ".join(classes)
    return opening_tag[: class_match.start(1)] + updated_classes + opening_tag[class_match.end(1) :]


def _add_html_table_alignment_classes(
    html_table: str,
    field_names: Sequence[str],
) -> str:
    """Apply alignment classes from the prepared field order to each table cell.

    The table is rewritten after HTML generation, so this pass works row by row
    and preserves any classes that earlier helpers already attached.
    """
    numeric_columns: frozenset[int] = frozenset(
        index for index, field_name in enumerate(field_names) if is_numeric_field(field_name)
    )
    row_pattern: re.Pattern[str] = re.compile(r"<tr>.*?</tr>", re.DOTALL)

    def _apply_to_row(row_html: str, *, tag_name: str) -> str:
        cell_pattern: re.Pattern[str] = re.compile(
            rf"<{tag_name}\b[^>]*>.*?</{tag_name}>",
            re.DOTALL,
        )
        cells = cell_pattern.findall(row_html)
        if not cells:
            return row_html

        def _with_alignment_class(
            cell_html: str,
            *,
            class_name: str,
        ) -> str:
            def replace_opening_tag(match: re.Match[str]) -> str:
                return _append_html_class(match.group(0), class_name)

            return re.sub(
                rf"<{tag_name}\b[^>]*>",
                replace_opening_tag,
                cell_html,
                count=1,
            )

        updated_cells: list[str] = []
        for index, cell_html in enumerate(cells):
            alignment_class = "numeric" if index in numeric_columns else "text"
            updated_cells.append(_with_alignment_class(cell_html, class_name=alignment_class))

        replacement_iter: Iterator[str] = iter(updated_cells)

        def replace_cell(_: re.Match[str]) -> str:
            return next(replacement_iter)

        return cell_pattern.sub(replace_cell, row_html)

    def repl(match: re.Match[str]) -> str:
        row_html = match.group(0)
        if "<th" in row_html:
            return _apply_to_row(row_html, tag_name="th")
        if "<td" in row_html:
            return _apply_to_row(row_html, tag_name="td")
        return row_html

    return row_pattern.sub(repl, html_table)


def _wrap_output_column_in_details(
    html_table: str,
    output_col_idx: int,
    sorted_results: Sequence[PerformanceResult],
) -> str:
    """Wrap the configured output cell in <details>/<summary> for expansion.

    Args:
        html_table: The HTML table string
        output_col_idx: The index of the output column (0-based)
        sorted_results: Results aligned to table row order for retrieving full text

    Returns:
        Modified HTML table with output column wrapped in details/summary tags
    """
    # Process the rendered HTML row by row so the preview stays in <summary>
    # and the fully escaped output remains available on expansion.
    lines: list[str] = html_table.split("\n")
    result_lines: list[str] = []
    row_idx = 0

    for original_line in lines:
        # Check if this is a data row (contains <td> tags)
        if "<td" in original_line and "</td>" in original_line:
            # Find all <td>...</td> cells in this row
            cells: list[str] = re.findall(r"<td[^>]*>.*?</td>", original_line)
            if len(cells) > output_col_idx:
                # Grab the configured output cell rather than assuming it is last.
                output_cell: str = cells[output_col_idx]

                # Extract the content between <td...> and </td>
                match: re.Match[str] | None = re.match(
                    r"(<td[^>]*>)(.*?)(</td>)",
                    output_cell,
                    re.DOTALL,
                )
                if match:
                    opening_tag: str
                    content: str
                    closing_tag: str
                    opening_tag, content, closing_tag = match.groups()

                    full_text = (
                        _full_output_report_text(sorted_results[row_idx])
                        if row_idx < len(sorted_results)
                        else html.unescape(content)
                    )

                    # Keep the preview in <summary> and reveal the full text on demand.
                    wrapped_content: str = (
                        f"<details><summary>{content}</summary>"
                        f"<div style='margin-top: 0.5em;'>{html.escape(full_text)}</div></details>"
                    )
                    new_cell: str = opening_tag + wrapped_content + closing_tag

                    # Replace the original output cell with the expandable version.
                    cells[output_col_idx] = new_cell

                    # Reconstruct the row with the updated cell sequence.
                    cell_iter: Iterator[str] = iter(cells)

                    def repl(_: re.Match[str], ci: Iterator[str] = cell_iter) -> str:
                        return next(ci)

                    updated_line: str = re.sub(
                        r"<td[^>]*>.*?</td>",
                        repl,
                        original_line,
                    )
                    result_lines.append(updated_line)
                    row_idx += 1
                else:
                    result_lines.append(original_line)
                    row_idx += 1
            else:
                result_lines.append(original_line)
                row_idx += 1
        else:
            result_lines.append(original_line)

    return "\n".join(result_lines)


def _initialize_metadata_baseline_tracking(
    summary: ModelIssueSummary,
    *,
    baseline_score: float | None,
    baseline_grade: str | None,
) -> tuple[list[str] | None, list[str] | None, list[str] | None]:
    """Initialize metadata-baseline tracking lists when a baseline is available."""
    if baseline_score is None or baseline_grade is None:
        return None, None, None

    improves_metadata: list[str] = []
    neutral_vs_metadata: list[str] = []
    worse_than_metadata: list[str] = []
    summary["metadata_baseline_score"] = baseline_score
    summary["metadata_baseline_grade"] = baseline_grade
    summary["cataloging_improves_metadata"] = improves_metadata
    summary["cataloging_neutral_vs_metadata"] = neutral_vs_metadata
    summary["cataloging_worse_than_metadata"] = worse_than_metadata
    return improves_metadata, neutral_vs_metadata, worse_than_metadata


def _is_cataloging_pick_eligible(
    result: PerformanceResult,
    analysis: GenerationQualityAnalysis | None,
) -> bool:
    """Return whether a result can appear in user-facing best-model picks."""
    if not result.success:
        return False
    if analysis is not None:
        if analysis.has_harness_issue or analysis.user_bucket == "avoid":
            return False
        if analysis.verdict in {"harness", "runtime_failure"}:
            return False

    labels = _extract_quality_issue_labels(result.quality_issues)
    return "harness" not in labels


def _finalize_cataloging_summary(
    summary: ModelIssueSummary,
    utility_scores: list[CatalogingScoreRecord],
    *,
    baseline_score: float | None,
) -> None:
    """Populate summary cataloging aggregates from per-model utility scores."""
    if not utility_scores:
        return

    best: CatalogingScoreRecord = max(utility_scores, key=lambda row: row[1])
    worst: CatalogingScoreRecord = min(utility_scores, key=lambda row: row[1])
    summary["cataloging_best"] = (best[0], best[1], best[2])
    summary["cataloging_worst"] = (worst[0], worst[1], worst[2])
    summary["cataloging_avg_score"] = sum(
        score for _model, score, _grade, _weakness, _delta in utility_scores
    ) / len(utility_scores)
    if baseline_score is None:
        return

    deltas: list[float] = [delta for _m, _s, _g, _w, delta in utility_scores if delta is not None]
    if deltas:
        summary["cataloging_avg_delta"] = sum(deltas) / len(deltas)


def analyze_model_issues(
    results: list[PerformanceResult],
    context: str | None = None,
) -> ModelIssueSummary:
    """Analyze results to identify common model issues and calculate performance highlights.

    Args:
        results: List of model performance results
        context: Optional context string (from prompt) for cataloging utility analysis
    """
    baseline: tuple[float, str] | None = _compute_metadata_baseline_utility(context)
    baseline_score: float | None = baseline[0] if baseline is not None else None
    baseline_grade: str | None = baseline[1] if baseline is not None else None

    failed_models: list[FailedModelIssue] = []
    repetitive_models: list[RepetitiveModelIssue] = []
    hallucination_models: list[HallucinationModelIssue] = []
    verbose_models: list[VerboseModelIssue] = []
    formatting_issues: list[FormattingModelIssue] = []
    excessive_bullets: list[ExcessiveBulletsIssue] = []
    cataloging_grades: dict[str, list[str]] = {}
    low_utility_models: list[LowUtilityModelIssue] = []
    utility_scores: list[CatalogingScoreRecord] = []
    recommendation_description_scores: list[tuple[str, float]] = []
    recommendation_keyword_scores: list[tuple[str, float]] = []

    summary: ModelIssueSummary = {
        "total_models": len(results),
        "failed_models": failed_models,
        "repetitive_models": repetitive_models,
        "hallucination_models": hallucination_models,
        "verbose_models": verbose_models,
        "formatting_issues": formatting_issues,
        "excessive_bullets": excessive_bullets,
        "cataloging_grades": cataloging_grades,
        "cataloging_best": None,
        "cataloging_worst": None,
        "cataloging_best_description": ("", 0.0),
        "cataloging_best_keywords": ("", 0.0),
        "cataloging_avg_score": 0.0,
        "cataloging_scores": utility_scores,
        "low_utility_models": low_utility_models,
    }

    improves_metadata: list[str] | None
    neutral_vs_metadata: list[str] | None
    worse_than_metadata: list[str] | None
    improves_metadata, neutral_vs_metadata, worse_than_metadata = (
        _initialize_metadata_baseline_tracking(
            summary,
            baseline_score=baseline_score,
            baseline_grade=baseline_grade,
        )
    )

    successful: list[PerformanceResult] = [r for r in results if r.success]
    _populate_summary_performance_highlights(summary, successful)
    runtime_analysis: RuntimeAnalysisSummary | None = _build_runtime_analysis_summary(results)
    if runtime_analysis is not None:
        summary["runtime_analysis"] = runtime_analysis

    for res in results:
        if not res.success:
            failed_models.append((res.model_name, res.error_stage, res.error_message))
            continue
        if not res.generation:
            continue

        text: str = getattr(res.generation, "text", "") or ""
        generation_tokens: int = getattr(res.generation, "generation_tokens", 0)
        prompt_tokens: int | None = getattr(res.generation, "prompt_tokens", None)

        analysis: GenerationQualityAnalysis | None = res.quality_analysis
        if analysis is None:
            analysis = analyze_generation_text(
                text,
                generation_tokens,
                prompt_tokens=prompt_tokens,
            )
        _append_quality_issue_entries(
            model_name=res.model_name,
            analysis=analysis,
            generation_tokens=generation_tokens,
            repetitive_models=repetitive_models,
            hallucination_models=hallucination_models,
            verbose_models=verbose_models,
            formatting_issues=formatting_issues,
            excessive_bullets=excessive_bullets,
        )

        utility = compute_cataloging_utility(text, context)
        score = float(utility["utility_score"])
        grade = str(utility["utility_grade"])
        weakness = str(utility["primary_weakness"])
        delta = score - baseline_score if baseline_score is not None else None
        utility_scores.append((res.model_name, score, grade, weakness, delta))
        cataloging_grades.setdefault(grade, []).append(res.model_name)
        description_score, keyword_score = (
            float(utility.get("description_score", 0.0)),
            float(utility.get("keyword_score", 0.0)),
        )
        if _is_cataloging_pick_eligible(res, analysis):
            recommendation_description_scores.append((res.model_name, description_score))
            recommendation_keyword_scores.append((res.model_name, keyword_score))

        if grade in ("D", "F"):
            low_utility_models.append((res.model_name, score, grade, weakness))

        _bucket_metadata_delta(
            model_name=res.model_name,
            delta=delta,
            improves_metadata=improves_metadata,
            neutral_vs_metadata=neutral_vs_metadata,
            worse_than_metadata=worse_than_metadata,
        )

    _finalize_cataloging_summary(
        summary,
        utility_scores,
        baseline_score=baseline_score,
    )
    if recommendation_description_scores:
        summary["cataloging_best_description"] = max(
            recommendation_description_scores,
            key=lambda item: (item[1], item[0]),
        )
    else:
        summary.pop("cataloging_best_description", None)
    if recommendation_keyword_scores:
        summary["cataloging_best_keywords"] = max(
            recommendation_keyword_scores,
            key=lambda item: (item[1], item[0]),
        )
    else:
        summary.pop("cataloging_best_keywords", None)

    return summary


def _populate_summary_performance_highlights(
    summary: ModelIssueSummary,
    successful: list[PerformanceResult],
) -> None:
    """Populate speed/memory highlights and aggregate resource stats."""
    if not successful:
        return

    fastest: PerformanceResult = max(
        successful,
        key=lambda r: getattr(r.generation, "generation_tps", 0) or 0,
    )
    fastest_tps: float = getattr(fastest.generation, "generation_tps", 0) or 0
    summary["fastest_model"] = (fastest.model_name, fastest_tps)

    most_efficient: PerformanceResult = min(
        successful,
        key=lambda r: getattr(r.generation, "peak_memory", float("inf")) or float("inf"),
    )
    efficient_mem: float = getattr(most_efficient.generation, "peak_memory", 0) or 0
    summary["most_efficient_model"] = (most_efficient.model_name, efficient_mem)

    fastest_load: PerformanceResult = min(
        successful,
        key=lambda r: getattr(r, "model_load_time", float("inf")) or float("inf"),
    )
    load_time: float = getattr(fastest_load, "model_load_time", 0) or 0
    summary["fastest_load_model"] = (fastest_load.model_name, load_time)

    total_tps: float = sum(getattr(r.generation, "generation_tps", 0) or 0 for r in successful)
    summary["average_tps"] = total_tps / len(successful)
    summary["successful_count"] = len(successful)

    total_mem: float = sum(getattr(r.generation, "peak_memory", 0) or 0 for r in successful)
    summary["total_peak_memory"] = total_mem
    summary["average_peak_memory"] = total_mem / len(successful)

    total_tokens: int = sum(
        (getattr(r.generation, "prompt_tokens", 0) or 0)
        + (getattr(r.generation, "generation_tokens", 0) or 0)
        for r in successful
    )
    summary["memory_efficiency"] = total_tokens / total_mem if total_mem > 0 else 0


def _append_quality_issue_entries(
    *,
    model_name: str,
    analysis: GenerationQualityAnalysis,
    generation_tokens: int,
    repetitive_models: list[RepetitiveModelIssue],
    hallucination_models: list[HallucinationModelIssue],
    verbose_models: list[VerboseModelIssue],
    formatting_issues: list[FormattingModelIssue],
    excessive_bullets: list[ExcessiveBulletsIssue],
) -> None:
    """Append generation quality flags into per-run issue buckets."""
    if analysis.is_repetitive:
        repetitive_models.append((model_name, analysis.repeated_token))
    if analysis.hallucination_issues:
        hallucination_models.append((model_name, analysis.hallucination_issues))
    if analysis.is_verbose:
        verbose_models.append((model_name, generation_tokens))
    if analysis.formatting_issues:
        formatting_issues.append((model_name, analysis.formatting_issues))
    if analysis.has_excessive_bullets:
        excessive_bullets.append((model_name, analysis.bullet_count))


def _bucket_metadata_delta(
    *,
    model_name: str,
    delta: float | None,
    improves_metadata: list[str] | None,
    neutral_vs_metadata: list[str] | None,
    worse_than_metadata: list[str] | None,
) -> None:
    """Bucket utility delta vs metadata baseline when baseline buckets are active."""
    if (
        delta is None
        or improves_metadata is None
        or neutral_vs_metadata is None
        or worse_than_metadata is None
    ):
        return
    if delta > UTILITY_DELTA_NEUTRAL_BAND:
        improves_metadata.append(model_name)
    elif delta < -UTILITY_DELTA_NEUTRAL_BAND:
        worse_than_metadata.append(model_name)
    else:
        neutral_vs_metadata.append(model_name)


_RUNTIME_PHASE_KEYS: Final[tuple[RuntimePhaseName, ...]] = (
    "model_load",
    "prompt_prep",
    "upstream_prefill_first_token",
    "post_prefill_decode",
    "decode",
    "cleanup",
)

_RUNTIME_PHASE_LABELS: Final[dict[RuntimePhaseName, str]] = {
    "model_load": "model load",
    "prompt_prep": "local prompt prep",
    "upstream_prefill_first_token": "upstream prefill / first-token",
    "post_prefill_decode": "post-prefill decode",
    "decode": "generation total (unsplit)",
    "cleanup": "cleanup",
}

_RUNTIME_PHASE_ACTIONS: Final[dict[RuntimePhaseName, tuple[str, str]]] = {
    "upstream_prefill_first_token": (
        (
            "Most measured runtime is spent inside upstream generation before the first "
            "token is available."
        ),
        (
            "Inspect prompt/image token accounting, dynamic-resolution image burden, "
            "prefill step sizing, and cache/prefill behavior."
        ),
    ),
    "post_prefill_decode": (
        "Most measured runtime is spent generating after the first token is available.",
        (
            "Prioritize early-stop policies, lower long-tail token budgets, "
            "or upstream decode-path work."
        ),
    ),
    "decode": (
        (
            "Most measured runtime is spent inside generation, but first-token timing "
            "was not available to split prefill from decode."
        ),
        (
            "Prioritize early-stop policies, lower long-tail token budgets, "
            "or rerun with prompt throughput metrics available to isolate prefill."
        ),
    ),
    "prompt_prep": (
        (
            "Local prompt preparation is consuming a large share of runtime, which often "
            "means expensive chat-template, preflight, or multimodal preprocessing work."
        ),
        "Inspect chat-template rendering, context blocks, image preprocessing, and validators.",
    ),
    "model_load": (
        "Cold model load time is a major share of runtime for this cohort.",
        "Consider staged runs, model reuse, or narrowing the model set before reruns.",
    ),
    "cleanup": (
        "Post-run synchronization or cache cleanup is taking a non-trivial share of time.",
        "Inspect cleanup policy, synchronization frequency, and cache-reset behavior.",
    ),
}


def _positive_runtime_duration(value: float | None) -> float | None:
    """Return a finite positive runtime duration, or None when unavailable."""
    if not isinstance(value, int | float):
        return None
    duration = float(value)
    if not math.isfinite(duration) or duration <= 0.0:
        return None
    return duration


def _runtime_upstream_prefill_first_token_time_s(
    runtime: RuntimeDiagnostics | None,
) -> float | None:
    """Return the upstream prefill/first-token proxy from mlx-vlm prompt timing."""
    if runtime is None:
        return None
    return _positive_runtime_duration(runtime.first_token_latency_s)


def _runtime_post_prefill_decode_time_s(runtime: RuntimeDiagnostics | None) -> float | None:
    """Return generation time after first token when a split signal is available."""
    if runtime is None:
        return None
    decode_time = _positive_runtime_duration(runtime.decode_time_s)
    upstream_prefill_time = _runtime_upstream_prefill_first_token_time_s(runtime)
    if decode_time is None or upstream_prefill_time is None:
        return None
    post_prefill_decode_time = max(decode_time - upstream_prefill_time, 0.0)
    return post_prefill_decode_time if post_prefill_decode_time > 0.0 else None


def _runtime_phase_durations(runtime: RuntimeDiagnostics | None) -> dict[RuntimePhaseName, float]:
    """Return normalized non-negative phase durations for one run."""
    if runtime is None:
        return {}

    phase_durations: dict[RuntimePhaseName, float] = {}
    base_phase_values: tuple[tuple[RuntimePhaseName, float | None], ...] = (
        ("model_load", runtime.model_load_time_s),
        ("prompt_prep", runtime.prompt_prep_time_s),
    )
    for phase, value in base_phase_values:
        duration = _positive_runtime_duration(value)
        if duration is not None:
            phase_durations[phase] = duration

    upstream_prefill_time = _runtime_upstream_prefill_first_token_time_s(runtime)
    post_prefill_decode_time = _runtime_post_prefill_decode_time_s(runtime)
    decode_time = _positive_runtime_duration(runtime.decode_time_s)
    if upstream_prefill_time is not None:
        phase_durations["upstream_prefill_first_token"] = upstream_prefill_time
        if post_prefill_decode_time is not None:
            phase_durations["post_prefill_decode"] = post_prefill_decode_time
    elif decode_time is not None:
        phase_durations["decode"] = decode_time

    cleanup_time = _positive_runtime_duration(runtime.cleanup_time_s)
    if cleanup_time is not None:
        phase_durations["cleanup"] = cleanup_time
    return phase_durations


def _build_runtime_analysis_summary(
    results: Sequence[PerformanceResult],
) -> RuntimeAnalysisSummary | None:
    """Build a rule-based runtime interpretation from per-result phase timings."""
    phase_totals: dict[RuntimePhaseName, float] = dict.fromkeys(_RUNTIME_PHASE_KEYS, 0.0)
    dominant_counts: Counter[RuntimePhaseName] = Counter[RuntimePhaseName]()
    termination_counts: Counter[str] = Counter[str]()
    measured_models: int = 0
    generation_total: float = 0.0
    generation_models: int = 0
    prefill_split_models: int = 0
    validation_total: float = 0.0
    validation_models: int = 0
    first_token_latencies: list[float] = []

    for result in results:
        runtime: RuntimeDiagnostics | None = result.runtime_diagnostics
        phase_durations: dict[RuntimePhaseName, float] = _runtime_phase_durations(runtime)
        if phase_durations:
            measured_models += 1
            run_dominant_phase: RuntimePhaseName = max(
                phase_durations,
                key=phase_durations.__getitem__,
            )
            dominant_counts[run_dominant_phase] += 1
            for phase, duration in phase_durations.items():
                phase_totals[phase] += duration
        if runtime is not None:
            decode_time: float | None = _positive_runtime_duration(runtime.decode_time_s)
            if decode_time is not None:
                generation_total += decode_time
                generation_models += 1
            if _runtime_upstream_prefill_first_token_time_s(runtime) is not None:
                prefill_split_models += 1
            validation_time: float | None = runtime.input_validation_time_s
            validation_duration = _positive_runtime_duration(validation_time)
            if validation_duration is not None:
                validation_total += validation_duration
                validation_models += 1
            first_token_latency: float | None = runtime.first_token_latency_s
            first_token_duration = _positive_runtime_duration(first_token_latency)
            if first_token_duration is not None:
                first_token_latencies.append(first_token_duration)
        stop_reason: str | None = runtime.stop_reason if runtime is not None else None
        if stop_reason:
            termination_counts[stop_reason] += 1

    total_measured: float = sum(phase_totals.values())
    if measured_models == 0 or total_measured <= 0.0:
        return None

    dominant_phase: RuntimePhaseName = max(phase_totals, key=phase_totals.__getitem__)
    dominant_phase_share: float = phase_totals[dominant_phase] / total_measured
    interpretation: str
    next_action: str
    interpretation, next_action = _RUNTIME_PHASE_ACTIONS.get(
        dominant_phase,
        (
            "Runtime is distributed across multiple phases without a clear single bottleneck.",
            "Inspect per-phase distributions before changing benchmark policy.",
        ),
    )

    if termination_counts.get("timeout", 0) > 0:
        interpretation += (
            f" Timeouts also affected {termination_counts['timeout']} model(s), "
            "so some totals may be dominated by interrupted runs."
        )

    first_token_latency_avg: float | None = (
        sum(first_token_latencies) / len(first_token_latencies) if first_token_latencies else None
    )
    dominant_count_map: dict[RuntimePhaseName, int] = {
        phase: dominant_counts[phase] for phase in _RUNTIME_PHASE_KEYS if dominant_counts[phase] > 0
    }

    runtime_summary: RuntimeAnalysisSummary = {
        "dominant_phase": dominant_phase,
        "dominant_phase_share": dominant_phase_share,
        "dominant_phase_count": dominant_counts.get(dominant_phase, 0),
        "measured_models": measured_models,
        "interpretation": interpretation,
        "next_action": next_action,
        "phase_totals": phase_totals,
        "dominant_counts": dominant_count_map,
        "termination_counts": dict(termination_counts),
        "generation_total": generation_total,
        "generation_models": generation_models,
        "prefill_split_models": prefill_split_models,
        "validation_total": validation_total,
        "validation_models": validation_models,
        "first_token_latency_avg": first_token_latency_avg,
        "first_token_latency_min": min(first_token_latencies) if first_token_latencies else None,
        "first_token_latency_max": max(first_token_latencies) if first_token_latencies else None,
        "first_token_latency_models": len(first_token_latencies),
    }
    return runtime_summary


def _format_runtime_generation_total_line(
    runtime_analysis: RuntimeAnalysisSummary,
) -> str | None:
    """Build the total upstream generation timing line for runtime summaries."""
    generation_total: float = runtime_analysis["generation_total"]
    generation_models: int = runtime_analysis["generation_models"]
    if generation_models <= 0 or generation_total <= 0.0:
        return None

    prefill_split_models: int = runtime_analysis["prefill_split_models"]
    if prefill_split_models > 0:
        split_note = (
            f"; upstream prefill / first-token split available for "
            f"{prefill_split_models}/{generation_models} model(s)"
        )
    else:
        split_note = "; upstream prefill / first-token split unavailable"
    return (
        "- **Generation total:** "
        f"{format_overall_runtime(generation_total)} across {generation_models} model(s)"
        f"{split_note}."
    )


def _format_runtime_phase_totals_line(
    runtime_analysis: RuntimeAnalysisSummary,
    *,
    trailing_period: bool = True,
) -> str | None:
    """Build the aggregate phase totals line shared by report renderers."""
    phase_totals: dict[RuntimePhaseName, float] = runtime_analysis["phase_totals"]
    phase_summary = ", ".join(
        f"{_RUNTIME_PHASE_LABELS.get(phase, phase)}={format_overall_runtime(duration)}"
        for phase, duration in phase_totals.items()
        if duration > 0.0
    )
    if not phase_summary:
        return None
    suffix = "." if trailing_period else ""
    return f"- **Phase totals:** {phase_summary}{suffix}"


def _format_runtime_timing_snapshot_lines(runtime_analysis: RuntimeAnalysisSummary) -> list[str]:
    """Build concise aggregate timing bullets for optional runtime signals."""
    lines: list[str] = []

    validation_models: int = runtime_analysis["validation_models"]
    validation_total: float = runtime_analysis["validation_total"]
    if validation_models > 0 and validation_total > 0.0:
        validation_avg: float = validation_total / validation_models
        lines.append(
            "- **Validation overhead:** "
            f"{format_overall_runtime(validation_total)} total "
            f"(avg {format_overall_runtime(validation_avg)} across "
            f"{validation_models} model(s)).",
        )

    first_token_models: int = runtime_analysis["first_token_latency_models"]
    first_token_avg: float | None = runtime_analysis["first_token_latency_avg"]
    first_token_min: float | None = runtime_analysis["first_token_latency_min"]
    first_token_max: float | None = runtime_analysis["first_token_latency_max"]
    if (
        first_token_models > 0
        and first_token_avg is not None
        and first_token_min is not None
        and first_token_max is not None
    ):
        lines.append(
            "- **Upstream prefill / first-token latency:** "
            f"Avg {format_overall_runtime(first_token_avg)} | "
            f"Min {format_overall_runtime(first_token_min)} | "
            f"Max {format_overall_runtime(first_token_max)} "
            f"across {first_token_models} model(s).",
        )

    return lines


def _format_runtime_analysis_lines(runtime_analysis: RuntimeAnalysisSummary) -> list[str]:
    """Build concise Markdown bullets explaining runtime timing implications."""
    dominant_phase: RuntimePhaseName = runtime_analysis["dominant_phase"]
    dominant_label: str = _RUNTIME_PHASE_LABELS.get(dominant_phase, dominant_phase)
    dominant_share: float = runtime_analysis["dominant_phase_share"]
    dominant_count: int = runtime_analysis["dominant_phase_count"]
    measured_models: int = runtime_analysis["measured_models"]
    termination_counts: dict[str, int] = runtime_analysis["termination_counts"]
    termination_summary: str = ", ".join(
        f"{name}={count}" for name, count in sorted(termination_counts.items())
    )

    lines: list[str] = [
        (
            f"- **Runtime pattern:** {dominant_label} dominates measured phase time "
            f"({dominant_share:.0%}; {dominant_count}/{measured_models} measured model(s))."
        ),
        f"- **What this likely means:** {runtime_analysis['interpretation']}",
        f"- **Suggested next action:** {runtime_analysis['next_action']}",
    ]
    phase_totals_line = _format_runtime_phase_totals_line(runtime_analysis)
    if phase_totals_line is not None:
        lines.insert(1, phase_totals_line)
    generation_total_line = _format_runtime_generation_total_line(runtime_analysis)
    if generation_total_line is not None:
        lines.insert(2, generation_total_line)
    if termination_summary:
        lines.append(f"- **Termination reasons:** {termination_summary}.")
    return lines


def compute_performance_statistics(results: list[PerformanceResult]) -> PerformanceStats:
    """Compute performance statistics (min, max, avg) for successful runs.

    Uses single-pass aggregation to build stats for all fields at once
    reducing overhead from repeated filtering and type conversions.
    """
    stats: PerformanceStats = {}
    successful_results: list[PerformanceResult] = [r for r in results if r.success and r.generation]
    if not successful_results:
        return stats

    fields_to_stat: list[str] = [
        "generation_tps",
        "peak_memory",
        "total_time",
        "generation_time",
        "model_load_time",
    ]

    # Single-pass aggregation: build value lists for all fields at once
    field_values: dict[str, list[float]] = {field: [] for field in fields_to_stat}

    for res in successful_results:
        for field in fields_to_stat:
            value: MetricValue = _get_field_value(res, field)
            numeric_value: float | None = _coerce_numeric_value(value)
            if numeric_value is not None:
                field_values[field].append(numeric_value)

    # Compute min/max/avg for fields with data
    for field, values in field_values.items():
        if values:
            stats[field] = NumericFieldStats(
                min=min(values),
                max=max(values),
                avg=sum(values) / len(values),
            )

    return stats


def _collect_top_performer_metrics(summary: ModelIssueSummary) -> list[TopPerformerMetric]:
    """Collect top-performer rows shared by HTML/Markdown renderers."""
    metrics: list[TopPerformerMetric] = []

    fastest_model = summary.get("fastest_model")
    if fastest_model is not None:
        metrics.append(("Fastest", fastest_model[0], fastest_model[1], "{:.1f} tps"))

    most_efficient_model = summary.get("most_efficient_model")
    if most_efficient_model is not None:
        metrics.append(
            ("💾 Most efficient", most_efficient_model[0], most_efficient_model[1], "{:.1f} GB"),
        )

    fastest_load_model = summary.get("fastest_load_model")
    if fastest_load_model is not None:
        metrics.append(("⚡ Fastest load", fastest_load_model[0], fastest_load_model[1], "{:.2f}s"))

    return metrics


def _collect_resource_usage_metrics(summary: ModelIssueSummary) -> list[ResourceUsageMetric]:
    """Collect aggregate resource rows shared by HTML/Markdown renderers."""
    metrics: list[ResourceUsageMetric] = []

    total_peak_memory = summary.get("total_peak_memory")
    if total_peak_memory is not None:
        metrics.append(("Total peak memory", total_peak_memory, "{:.1f} GB"))

    average_peak_memory = summary.get("average_peak_memory")
    if average_peak_memory is not None:
        metrics.append(("Average peak memory", average_peak_memory, "{:.1f} GB"))

    memory_efficiency = summary.get("memory_efficiency")
    if memory_efficiency is not None:
        metrics.append(("Memory efficiency", memory_efficiency, "{:.0f} tokens/GB"))

    return metrics


def _collect_aggregate_statistics_rows(stats: PerformanceStats) -> list[AggregateStatRow]:
    """Collect preformatted aggregate statistics rows shared by HTML/Markdown summaries."""
    rows: list[AggregateStatRow] = []
    for field, data in stats.items():
        rows.append(
            (
                format_field_label(field),
                format_field_value(field, data["avg"]),
                format_field_value(field, data["min"]),
                format_field_value(field, data["max"]),
            ),
        )
    return rows


def _format_top_performers(
    summary: ModelIssueSummary,
    *,
    html_output: bool,
) -> list[str]:
    parts: list[str] = []
    top_metrics = _collect_top_performer_metrics(summary)
    average_tps = summary.get("average_tps")
    successful_count = summary.get("successful_count")
    if top_metrics or (average_tps is not None and successful_count is not None):
        if html_output:
            parts.append("<h3>🏆 Performance Highlights</h3><ul>")
        else:
            _append_markdown_section(parts, title="## 🏆 Performance Highlights")
        for label, model, value, fmt in top_metrics:
            if html_output:
                parts.append(
                    f"<li><b>{label}:</b> <code>{html.escape(model)}</code> "
                    f"({fmt.format(value)})</li>",
                )
            else:
                parts.append(f"- **{label}:** `{model}` ({fmt.format(value)})")
        if average_tps is not None and successful_count is not None:
            if html_output:
                parts.append(
                    f"<li><b>📊 Average TPS:</b> {average_tps:.1f} "
                    f"across {successful_count} models</li>",
                )
            else:
                parts.append(
                    f"- **📊 Average TPS:** {average_tps:.1f} across {successful_count} models",
                )
        parts.append("</ul>" if html_output else "")

    resource_metrics = _collect_resource_usage_metrics(summary)
    if resource_metrics:
        if html_output:
            parts.append("<h3>📈 Resource Usage</h3><ul>")
        else:
            _append_markdown_section(parts, title="## 📈 Resource Usage")
        for label, value, fmt in resource_metrics:
            if html_output:
                parts.append(f"<li><b>{label}:</b> {fmt.format(value)}</li>")
            else:
                parts.append(f"- **{label}:** {fmt.format(value)}")
        parts.append("</ul>" if html_output else "")

    return parts


def _collect_quality_issue_sections(summary: ModelIssueSummary) -> list[QualityIssueSection]:
    """Collect quality issue sections shared by HTML/Markdown summary renderers."""
    sections: list[QualityIssueSection] = []

    failed_models = summary.get("failed_models", [])
    if failed_models:
        sections.append(
            (
                "❌ Failed Models",
                "metric-bad",
                [(model, stage or "Unknown") for model, stage, _ in failed_models],
            ),
        )

    repetitive_models = summary.get("repetitive_models", [])
    if repetitive_models:
        sections.append(
            (
                "🔄 Repetitive Output",
                "metric-warn",
                [(model, f"token: {token or '?'}") for model, token in repetitive_models],
            ),
        )

    hallucination_models = summary.get("hallucination_models", [])
    if hallucination_models:
        sections.append(
            (
                "👻 Hallucinations",
                "metric-warn",
                [(model, None) for model, _ in hallucination_models],
            ),
        )

    formatting_issues = summary.get("formatting_issues", [])
    if formatting_issues:
        sections.append(
            (
                "📝 Formatting Issues",
                "metric-warn",
                [(model, None) for model, _ in formatting_issues],
            ),
        )

    return sections


def _format_quality_detail(detail: str, *, html_output: bool) -> str:
    """Format detail text for HTML/Markdown quality rows while preserving token styling."""
    if detail.startswith("token: "):
        token_text = detail.removeprefix("token: ")
        if html_output:
            return f"token: <code>{html.escape(token_text)}</code>"
        return f"token: `{token_text}`"
    if html_output:
        return html.escape(detail)
    return f"`{detail}`"


def _format_quality_issues(
    summary: ModelIssueSummary,
    *,
    html_output: bool,
) -> list[str]:
    sections = _collect_quality_issue_sections(summary)
    if not sections:
        return []

    quality_parts: list[str] = []
    for title, css_class, entries in sections:
        if html_output:
            quality_parts.append(f"<li><b class='{css_class}'>{title} ({len(entries)}):</b><ul>")
        else:
            quality_parts.append(f"- **{title} ({len(entries)}):**")
        for model, detail in entries:
            if detail is None:
                quality_parts.append(
                    f"<li><code>{html.escape(model)}</code></li>"
                    if html_output
                    else f"  - `{model}`",
                )
            else:
                detail_text = _format_quality_detail(detail, html_output=html_output)
                if html_output:
                    quality_parts.append(
                        f"<li><code>{html.escape(model)}</code> ({detail_text})</li>",
                    )
                else:
                    quality_parts.append(f"  - `{model}` ({detail_text})")
        if html_output:
            quality_parts.append("</ul></li>")

    if html_output:
        return ["<h3>⚠️ Quality Issues</h3><ul>", *quality_parts, "</ul>"]

    parts: list[str] = []
    _append_markdown_section(parts, title="## ⚠️ Quality Issues")
    parts.extend(quality_parts)
    parts.append("")
    return parts


def _cataloging_grade_distribution_items(summary: ModelIssueSummary) -> list[str]:
    """Build grade-distribution labels shared by HTML/Markdown renderers."""
    grades = summary.get("cataloging_grades", {})
    if not grades:
        return []
    grade_counts: list[str] = []
    for grade in ["A", "B", "C", "D", "F"]:
        count = len(grades.get(grade, []))
        if count > 0:
            emoji = GRADE_EMOJIS.get(grade, "")
            grade_counts.append(f"{emoji} {grade}: {count}")
    return grade_counts


def _cataloging_vs_metadata_breakdown(
    summary: ModelIssueSummary,
) -> tuple[float, str, float, int, int, int] | None:
    """Return baseline/delta summary when metadata baseline is available."""
    baseline_score = summary.get("metadata_baseline_score")
    baseline_grade = summary.get("metadata_baseline_grade")
    if baseline_score is None or baseline_grade is None:
        return None
    better = len(summary.get("cataloging_improves_metadata", []))
    neutral = len(summary.get("cataloging_neutral_vs_metadata", []))
    worse = len(summary.get("cataloging_worse_than_metadata", []))
    avg_delta = summary.get("cataloging_avg_delta", 0.0)
    return baseline_score, baseline_grade, avg_delta, better, neutral, worse


def _collect_cataloging_summary_data(summary: ModelIssueSummary) -> CatalogingSummaryData | None:
    """Collect shared cataloging summary data for HTML and Markdown renderers."""
    best_entry = summary.get("cataloging_best")
    if best_entry is None:
        return None

    average_score = summary.get("cataloging_avg_score", 0.0)
    return CatalogingSummaryData(
        grade_counts=tuple(_cataloging_grade_distribution_items(summary)),
        average_score=average_score if average_score > 0 else None,
        metadata_breakdown=_cataloging_vs_metadata_breakdown(summary),
        best_entry=best_entry,
        worst_entry=summary.get("cataloging_worst"),
        best_description_entry=summary.get("cataloging_best_description"),
        best_keyword_entry=summary.get("cataloging_best_keywords"),
        low_utility_models=tuple(summary.get("low_utility_models", [])),
    )


def _format_cataloging_summary(  # noqa: C901, PLR0912, PLR0915 — dual-format renderer
    data: CatalogingSummaryData,
    *,
    html_output: bool,
) -> list[str]:
    """Format cataloging utility summary as HTML or Markdown text."""
    parts: list[str] = []

    if html_output:
        parts.append("<h3>📚 Cataloging Utility Summary</h3>")
    else:
        _append_markdown_section(parts, title="## 📚 Cataloging Utility Summary")

    if data.grade_counts:
        grade_line = f"{' | '.join(data.grade_counts)}"
        if html_output:
            parts.append(f"<p><b>Grade Distribution:</b> {grade_line}</p>")
        else:
            parts.append(f"**Grade Distribution:** {grade_line}")
            parts.append("")

    if data.average_score is not None:
        score_text = f"{data.average_score:.0f}/100"
        if html_output:
            parts.append(f"<p><b>Average Utility Score:</b> {score_text}</p>")
        else:
            parts.append(f"**Average Utility Score:** {score_text}")
            parts.append("")

    if data.metadata_breakdown is not None:
        baseline_score, baseline_grade, avg_delta, better, neutral, worse = data.metadata_breakdown
        baseline_emoji = GRADE_EMOJIS.get(baseline_grade, "❌")
        baseline_text = f"{baseline_emoji} {baseline_grade} ({baseline_score:.0f}/100)"
        delta_text = (
            f"Avg Δ {avg_delta:+.0f} | Better: {better}, Neutral: {neutral}, Worse: {worse}"
        )
        if html_output:
            parts.append(f"<p><b>Existing Metadata Baseline:</b> {baseline_text}</p>")
            parts.append(f"<p><b>Vs Existing Metadata:</b> {delta_text}</p>")
        else:
            parts.append(f"**Existing Metadata Baseline:** {baseline_text}")
            parts.append(f"**Vs Existing Metadata:** {delta_text}")
            parts.append("")

    if html_output:
        parts.append("<ul>")
    if data.best_entry is not None:
        model, score, grade = data.best_entry
        emoji = GRADE_EMOJIS.get(grade, "")
        score_str = f"{score:.0f}/100"
        if html_output:
            parts.append(
                f"<li><b>Best for cataloging:</b> <code>{html.escape(model)}</code> "
                f"({emoji} {grade}, {score_str})</li>",
            )
        else:
            parts.append(
                f"- **Best for cataloging:** `{model}` ({emoji} {grade}, {score_str})",
            )
    if data.best_description_entry is not None:
        model, score = data.best_description_entry
        score_str = f"{score:.0f}/100"
        if html_output:
            parts.append(
                f"<li><b>Best descriptions:</b> <code>{html.escape(model)}</code> ({score_str})</li>",
            )
        else:
            parts.append(f"- **Best descriptions:** `{model}` ({score_str})")
    if data.best_keyword_entry is not None:
        model, score = data.best_keyword_entry
        score_str = f"{score:.0f}/100"
        if html_output:
            parts.append(
                f"<li><b>Best keywording:</b> <code>{html.escape(model)}</code> ({score_str})</li>",
            )
        else:
            parts.append(f"- **Best keywording:** `{model}` ({score_str})")
    if data.worst_entry is not None:
        model, score, grade = data.worst_entry
        emoji = GRADE_EMOJIS.get(grade, "")
        score_str = f"{score:.0f}/100"
        if html_output:
            parts.append(
                f"<li><b>Worst for cataloging:</b> <code>{html.escape(model)}</code> "
                f"({emoji} {grade}, {score_str})</li>",
            )
        else:
            parts.append(
                f"- **Worst for cataloging:** `{model}` ({emoji} {grade}, {score_str})",
            )
    if html_output:
        parts.append("</ul>")
    else:
        parts.append("")

    if data.low_utility_models:
        count = len(data.low_utility_models)
        if html_output:
            parts.append(
                f"<p><b class='metric-warn'>⚠️ {count} models with low utility (D/F):</b></p>",
            )
            parts.append("<ul>")
        else:
            parts.append(f"### ⚠️ {count} Models with Low Utility (D/F)")
            parts.append("")
        for model, score, grade, weakness in data.low_utility_models:
            emoji = GRADE_EMOJIS.get(grade, "")
            if html_output:
                parts.append(
                    f"<li><code>{html.escape(model)}</code>: "
                    f"{emoji} {grade} ({score:.0f}/100) "
                    f"- {html.escape(weakness)}</li>",
                )
            else:
                parts.append(
                    f"- `{model}`: {emoji} {grade} ({score:.0f}/100) - {weakness}",
                )
        if html_output:
            parts.append("</ul>")
        else:
            parts.append("")

    return parts


def _format_aggregate_statistics(
    stats: PerformanceStats,
    *,
    html_output: bool,
) -> list[str]:
    rows = _collect_aggregate_statistics_rows(stats)
    if not rows:
        return []

    parts: list[str] = []
    if html_output:
        parts.append("<h3>📊 Aggregate Statistics (Successful Runs)</h3><ul>")
    else:
        _append_markdown_section(parts, title="## 📊 Aggregate Statistics (Successful Runs)")

    for label, average_value, minimum_value, maximum_value in rows:
        if html_output:
            parts.append(
                f"<li><b>{label}</b>: Avg: {average_value} | "
                f"Min: {minimum_value} | Max: {maximum_value}</li>",
            )
        else:
            parts.append(
                f"- **{label}**: Avg: {average_value} | "
                f"Min: {minimum_value} | Max: {maximum_value}",
            )
    parts.append("</ul>" if html_output else "")
    return parts


def _format_issues_summary_parts(
    summary: ModelIssueSummary,
    stats: PerformanceStats,
    *,
    html_output: bool,
) -> list[str]:
    parts: list[str] = []
    cataloging_data = _collect_cataloging_summary_data(summary)
    parts.extend(_format_top_performers(summary, html_output=html_output))
    if cataloging_data is not None:
        parts.extend(_format_cataloging_summary(cataloging_data, html_output=html_output))
    parts.extend(_format_quality_issues(summary, html_output=html_output))
    parts.extend(
        _format_aggregate_statistics(
            stats,
            html_output=html_output,
        ),
    )
    return parts


def format_issues_summary_html(summary: ModelIssueSummary, stats: PerformanceStats) -> str:
    """Format the issues and statistics summary as an HTML string."""
    return "".join(_format_issues_summary_parts(summary, stats, html_output=True))


def _relative_markdown_artifact_path(*, report_filename: Path, artifact_filename: Path) -> str:
    """Return a relative path for links between Markdown artifacts."""
    try:
        return os.path.relpath(artifact_filename, start=report_filename.parent)
    except ValueError:
        return str(artifact_filename)


def _gallery_model_anchor(model_name: str) -> str:
    """Build a stable anchor for model sections in the gallery artifact."""
    normalized_name = model_name.lower().replace("/", " ")
    normalized = re.sub(r"[^a-z0-9\s-]", "", normalized_name)
    collapsed = re.sub(r"[-\s]+", "-", normalized).strip("-")
    return f"model-{collapsed or 'entry'}"


def _format_gallery_model_link(
    model_name: str,
    *,
    gallery_relative_path: str | None = None,
) -> str:
    """Format a Markdown link to a model entry in the gallery artifact."""
    target = f"#{_gallery_model_anchor(model_name)}"
    if gallery_relative_path is not None:
        target = f"{gallery_relative_path.replace(' ', '%20')}{target}"
    return f"[`{model_name}`]({target})"


def _collect_report_component_rows(
    *,
    versions: LibraryVersionDict,
    system_info: Mapping[str, str],
    library_names: Sequence[str] | None = None,
    system_keys: Sequence[str] | None = None,
) -> list[tuple[str, str]]:
    """Collect shared library/system rows for human-facing report artifacts."""
    rows: list[tuple[str, str]] = []
    selected_library_names = tuple(library_names) if library_names is not None else tuple(versions)
    selected_system_keys = tuple(system_keys) if system_keys is not None else tuple(system_info)

    for library_name in selected_library_names:
        version_value = versions.get(library_name)
        if version_value:
            rows.append((library_name, version_value))

    for system_key in selected_system_keys:
        system_value = system_info.get(system_key)
        if system_value:
            rows.append((system_key, system_value))

    return rows


def _cataloging_score_index(
    summary: ModelIssueSummary,
) -> dict[str, tuple[float, str, str, float | None]]:
    """Index cached cataloging scores by model for shared report helpers."""
    return {
        model_name: (score, grade, weakness, delta)
        for model_name, score, grade, weakness, delta in summary.get("cataloging_scores", [])
    }


def _quality_analysis_for_result(res: PerformanceResult) -> GenerationQualityAnalysis | None:
    """Return cached quality analysis for a result when present."""
    if res.quality_analysis is not None:
        return res.quality_analysis
    generation = res.generation
    if generation is None:
        return None
    generation_analysis = getattr(generation, "quality_analysis", None)
    return (
        generation_analysis if isinstance(generation_analysis, GenerationQualityAnalysis) else None
    )


def _build_jsonl_review_record(result: PerformanceResult) -> JsonlReviewRecord | None:
    """Build the canonical automated review payload for one result."""
    analysis = _quality_analysis_for_result(result)
    generation = result.generation
    generation_tokens = 0
    if generation is not None:
        generation_tokens = getattr(generation, "generation_tokens", 0) or 0
    prompt_tokens_total = getattr(generation, "prompt_tokens", None) if generation else None
    prompt_output_ratio = (
        generation_tokens / prompt_tokens_total
        if prompt_tokens_total is not None and prompt_tokens_total > 0
        else None
    )

    if analysis is not None:
        requested_max_tokens = analysis.requested_max_tokens or result.requested_max_tokens
        nontext_prompt_ratio = (
            analysis.prompt_tokens_nontext_est / analysis.prompt_tokens_total
            if analysis.prompt_tokens_total is not None
            and analysis.prompt_tokens_total > 0
            and analysis.prompt_tokens_nontext_est is not None
            else None
        )
        return {
            "verdict": analysis.verdict,
            "hint_relationship": analysis.hint_relationship,
            "instruction_echo": analysis.instruction_echo,
            "metadata_borrowing": analysis.metadata_borrowing,
            "likely_capped": analysis.likely_capped,
            "owner": analysis.owner,
            "user_bucket": analysis.user_bucket,
            "evidence": list(analysis.evidence),
            "requested_max_tokens": requested_max_tokens,
            "hit_max_tokens": bool(
                requested_max_tokens is not None and generation_tokens >= requested_max_tokens
            ),
            "prompt_tokens_total": analysis.prompt_tokens_total,
            "prompt_tokens_text_est": analysis.prompt_tokens_text_est,
            "prompt_tokens_nontext_est": analysis.prompt_tokens_nontext_est,
            "prompt_output_ratio": prompt_output_ratio,
            "nontext_prompt_ratio": nontext_prompt_ratio,
            "missing_terms": list(analysis.missing_context_terms),
            "missing_sections": list(analysis.missing_sections),
            "harness_details": list(analysis.harness_issue_details),
        }

    if result.success:
        return None

    requested_max_tokens = result.requested_max_tokens
    evidence: list[str] = _failure_review_evidence(result)
    return {
        "verdict": "runtime_failure",
        "hint_relationship": "preserves_trusted_hints",
        "instruction_echo": False,
        "metadata_borrowing": False,
        "likely_capped": bool(
            requested_max_tokens is not None and generation_tokens >= requested_max_tokens
        ),
        "owner": _classify_review_owner(
            harness_type=None,
            failure_owner=result.error_package,
        ),
        "user_bucket": "avoid",
        "evidence": _dedupe_preserve_order(evidence) or ["runtime_failure"],
        "requested_max_tokens": requested_max_tokens,
        "hit_max_tokens": bool(
            requested_max_tokens is not None and generation_tokens >= requested_max_tokens
        ),
        "prompt_tokens_total": prompt_tokens_total,
        "prompt_tokens_text_est": None,
        "prompt_tokens_nontext_est": None,
        "prompt_output_ratio": prompt_output_ratio,
        "nontext_prompt_ratio": None,
        "missing_terms": [],
        "missing_sections": [],
        "harness_details": [],
    }


_HUGGINGFACE_HUB_CONNECTIVITY_NEEDLES: Final[tuple[str, ...]] = (
    "server disconnected without sending a response",
    "remoteprotocolerror",
    "connection refused",
    "connection reset",
    "connection aborted",
    "connection error",
    "network is unreachable",
    "temporary failure in name resolution",
    "nodename nor servname provided",
    "failed to establish a new connection",
    "read timeout",
    "connect timeout",
    "timed out",
    "503 service unavailable",
    "502 bad gateway",
    "504 gateway timeout",
)

HEAVY_NON_TEXT_PROMPT_RATIO_THRESHOLD: Final[float] = 0.5


def _is_huggingface_hub_connectivity_failure(result: PerformanceResult) -> bool:
    """Return whether a failed result likely reflects a transient Hub connectivity issue."""
    if (result.error_package or "").casefold() != "huggingface-hub":
        return False

    combined = " ".join(
        part.casefold()
        for part in (
            result.error_message,
            result.error_traceback,
            result.captured_output_on_fail,
        )
        if part
    )
    return any(needle in combined for needle in _HUGGINGFACE_HUB_CONNECTIVITY_NEEDLES)


def _failure_review_evidence(result: PerformanceResult) -> list[str]:
    """Build compact evidence labels for failed-result review payloads."""
    evidence: list[str] = []
    if result.error_stage:
        evidence.append(re.sub(r"[^a-z0-9]+", "_", result.error_stage.casefold()).strip("_"))
    if result.error_code:
        evidence.append(result.error_code.casefold())
    if _is_huggingface_hub_connectivity_failure(result):
        evidence.append("hub_connectivity")
    return _dedupe_preserve_order(evidence) or ["runtime_failure"]


def _review_issue_subtype(
    result: PerformanceResult,
    review: JsonlReviewRecord,
    analysis: GenerationQualityAnalysis | None,
) -> str | None:
    """Return the most actionable subtype label for maintainer triage surfaces."""
    subtype: str | None = None
    if analysis is not None and analysis.harness_issue_type:
        subtype = analysis.harness_issue_type
    elif result.error_code:
        subtype = result.error_code
    elif review["hit_max_tokens"]:
        subtype = "token_cap"
    elif review["missing_sections"]:
        subtype = "contract_mismatch"
    elif review["missing_terms"]:
        subtype = "context_gap"
    elif review["evidence"]:
        subtype = review["evidence"][0]
    return subtype


def _review_confidence(
    result: PerformanceResult,
    review: JsonlReviewRecord,
    analysis: GenerationQualityAnalysis | None,
) -> MaintainerConfidence:
    """Return confidence level for maintainer-oriented triage routing."""
    if not result.success and (
        result.error_code is not None
        or result.error_traceback is not None
        or result.error_package is not None
    ):
        return "high"
    if analysis is not None:
        if analysis.has_harness_issue and analysis.harness_issue_type is not None:
            return "high"
        if review["verdict"] in {"cutoff", "context_budget"} and (
            review["hit_max_tokens"] or review["nontext_prompt_ratio"] is not None
        ):
            return "high"
        if analysis.evidence or analysis.missing_sections or analysis.missing_context_terms:
            return "medium"
    if review["evidence"] or review["missing_sections"] or review["missing_terms"]:
        return "medium"
    return "low"


def _failure_context_text(result: PerformanceResult) -> str:
    """Return combined failure text for issue/routing heuristics."""
    return " ".join(
        part
        for part in (
            result.root_error_message,
            result.error_message,
            result.error_stage,
            result.error_code,
            result.error_traceback,
            result.captured_output_on_fail,
        )
        if part
    )


_RUNTIME_FAILURE_PROBLEM_LABELS: Final[dict[str, str]] = {
    "unsupported_granite": "Unsupported Granite model type/import path",
    "weight_config_mismatch": "Weight/config mismatch during model load",
    "missing_image_processor": "Processor config is missing image processor",
    "missing_chat_template": "Missing chat template in model/tokenizer config",
    "api_mismatch": "Upstream API mismatch in processor/generation call",
    "missing_module": "Missing module/import during model load",
}

_RUNTIME_FAILURE_NEXT_ACTIONS: Final[dict[str, str]] = {
    "unsupported_granite": (
        "Check mlx-vlm model-type registration/import handling for Granite; confirm the "
        "loader either supports this architecture or fails before generation with a clear "
        "unsupported-model message."
    ),
    "weight_config_mismatch": (
        "Compare checkpoint keys with the selected model class/config, especially projector "
        "scale/bias parameters and quantized weight naming, before judging model quality."
    ),
    "missing_image_processor": (
        "Inspect the model repo processor/preprocessor config and AutoProcessor mapping; "
        "the multimodal processor is missing or not exposing the image processor expected "
        "by mlx-vlm."
    ),
    "missing_chat_template": (
        "Inspect tokenizer chat_template and EOS defaults in the model repo before changing "
        "generation logic."
    ),
    "api_mismatch": (
        "Check installed transformers/mlx-vlm API compatibility and forwarded processor or "
        "generation kwargs."
    ),
    "missing_module": (
        "Inspect the import path and installed package version that owns the missing module "
        "before treating this as a model failure."
    ),
}


def _runtime_failure_diagnostic_kind(result: PerformanceResult | None) -> str | None:
    """Return a compact runtime-failure kind from structured root-cause fields."""
    kind: str | None = None
    if result is None:
        return kind

    failure_text = _failure_context_text(result)
    normalized = failure_text.casefold()
    error_code = (result.error_code or "").upper()
    error_stage = (result.error_stage or "").casefold()

    if "granite" in normalized and "not supported" in normalized:
        kind = "unsupported_granite"
    elif (
        "WEIGHT_MISMATCH" in error_code
        or "weight mismatch" in error_stage
        or "parameters not in model" in normalized
        or ("missing" in normalized and "parameters" in normalized)
        or ("received" in normalized and "parameters" in normalized)
    ):
        kind = "weight_config_mismatch"
    elif (
        "image_processor" in normalized
        or "image processor" in normalized
        or "PROCESSOR" in error_code
        or "processor" in error_stage
    ):
        kind = "missing_image_processor"
    elif "chat_template is not set" in normalized or "no template argument" in normalized:
        kind = "missing_chat_template"
    elif "unexpected keyword argument" in normalized or "got an unexpected keyword" in normalized:
        kind = "api_mismatch"
    elif "no module named" in normalized:
        kind = "missing_module"
    return kind


def _review_runtime_failure_next_action(result: PerformanceResult) -> str | None:
    """Return failure-specific maintainer guidance from structured root-cause fields."""
    kind = _runtime_failure_diagnostic_kind(result)
    return _RUNTIME_FAILURE_NEXT_ACTIONS.get(kind or "")


def _review_next_action_for_result(
    result: PerformanceResult,
    review: JsonlReviewRecord,
) -> str:
    """Return one actionable next-step line, using result details when available."""
    if review["verdict"] == "runtime_failure" and (
        action := _review_runtime_failure_next_action(result)
    ):
        return action
    return _review_next_action_text(review)


def _review_maintainer_next_action(
    result: PerformanceResult,
    review: JsonlReviewRecord,
) -> str:
    """Return a maintainer-facing next action, avoiding noisy advice for clean rows."""
    if review["verdict"] == "clean" and review["user_bucket"] == "recommended":
        return "No immediate maintainer action."
    return _review_next_action_for_result(result, review)


def _build_jsonl_maintainer_triage_record(
    result: PerformanceResult,
    review: JsonlReviewRecord,
    *,
    issue_cluster: IssueCluster | None = None,
) -> JsonlMaintainerTriageRecord:
    """Build an action-oriented maintainer triage payload for one result."""
    analysis = _quality_analysis_for_result(result)
    triage: JsonlMaintainerTriageRecord = {
        "suspected_owner": review["owner"],
        "confidence": _review_confidence(result, review, analysis),
        "issue_kind": review["verdict"],
        "summary": _review_focus_text(review, analysis),
        "next_action": _review_maintainer_next_action(result, review),
        "user_bucket": review["user_bucket"],
        "evidence": list(review["evidence"]),
        "harness_details": list(review["harness_details"]),
        "hit_max_tokens": review["hit_max_tokens"],
    }
    issue_subtype = _review_issue_subtype(result, review, analysis)
    stop_reason = result.runtime_diagnostics.stop_reason if result.runtime_diagnostics else None

    if issue_subtype is not None:
        triage["issue_subtype"] = issue_subtype
    if result.error_stage is not None:
        triage["error_stage"] = result.error_stage
    if result.error_code is not None:
        triage["error_code"] = result.error_code
    if stop_reason is not None:
        triage["stop_reason"] = stop_reason
    if review["requested_max_tokens"] is not None:
        triage["requested_max_tokens"] = review["requested_max_tokens"]
    if review["prompt_tokens_total"] is not None:
        triage["prompt_tokens_total"] = review["prompt_tokens_total"]
    if review["prompt_output_ratio"] is not None:
        triage["prompt_output_ratio"] = review["prompt_output_ratio"]
    if review["nontext_prompt_ratio"] is not None:
        triage["nontext_prompt_ratio"] = review["nontext_prompt_ratio"]
    if issue_cluster is not None:
        triage["issue_cluster_id"] = issue_cluster.cluster_id
        triage["issue_cluster_path"] = _issue_cluster_path(issue_cluster)
        triage["acceptance_signal"] = issue_cluster.acceptance_signal
    return triage


def _maintainer_triage_for_result(
    result: PerformanceResult,
) -> JsonlMaintainerTriageRecord | None:
    """Return cached maintainer triage payload for a result when available."""
    review = _build_jsonl_review_record(result)
    if review is None:
        return None
    return _build_jsonl_maintainer_triage_record(result, review)


def _maintainer_triage_evidence_text(triage: JsonlMaintainerTriageRecord) -> str | None:
    """Return human-readable evidence text for maintainer triage surfaces."""
    items = [
        description
        for detail in triage["harness_details"][:2]
        if (description := _describe_harness_detail(detail))
    ]
    if not items:
        items = [_humanize_review_evidence_label(label) for label in triage["evidence"][:3]]
    unique_items = _dedupe_preserve_order(items)
    return " | ".join(unique_items) if unique_items else None


def _maintainer_triage_context_text(triage: JsonlMaintainerTriageRecord) -> str | None:
    """Return prompt/output context from triage when available."""
    context_parts: list[str] = []
    prompt_tokens_total = triage.get("prompt_tokens_total")
    if prompt_tokens_total is not None:
        context_parts.append(f"prompt={fmt_num(prompt_tokens_total)}")
    prompt_output_ratio = triage.get("prompt_output_ratio")
    if prompt_output_ratio is not None:
        context_parts.append(f"output/prompt={prompt_output_ratio:.2%}")
    nontext_prompt_ratio = triage.get("nontext_prompt_ratio")
    if nontext_prompt_ratio is not None:
        context_parts.append(f"nontext burden={nontext_prompt_ratio:.0%}")
    stop_reason = triage.get("stop_reason")
    if stop_reason is not None:
        context_parts.append(f"stop={stop_reason}")
    if triage["hit_max_tokens"]:
        requested_max_tokens = triage.get("requested_max_tokens")
        if requested_max_tokens is not None:
            context_parts.append(f"hit token cap ({requested_max_tokens})")
        else:
            context_parts.append("hit token cap")
    return " | ".join(context_parts) if context_parts else None


def _build_maintainer_triage_rows(result: PerformanceResult) -> list[tuple[str, str]]:
    """Build compact human-readable maintainer triage rows for Markdown reports."""
    triage = _maintainer_triage_for_result(result)
    if triage is None:
        return []

    classification = triage["issue_kind"]
    issue_subtype = triage.get("issue_subtype")
    if issue_subtype:
        classification = f"{classification} | {issue_subtype}"

    rows: list[tuple[str, str]] = [
        (
            "Likely owner",
            f"{triage['suspected_owner']} | confidence={triage['confidence']}",
        ),
        ("Classification", classification),
    ]
    if triage["summary"] and triage["summary"] != "no flagged signals":
        rows.append(("Summary", triage["summary"]))
    if evidence_text := _maintainer_triage_evidence_text(triage):
        rows.append(("Evidence", evidence_text))
    if context_text := _maintainer_triage_context_text(triage):
        rows.append(("Token context", context_text))
    rows.append(("Next action", triage["next_action"]))
    return rows


def _append_maintainer_triage_markdown(
    parts: list[str],
    *,
    result: PerformanceResult,
    escaper: DiagnosticsEscaper | MarkdownPipeEscaper,
    heading: str | None = "**Maintainer triage:**",
) -> None:
    """Append a compact maintainer triage block to a Markdown artifact."""
    rows = _build_maintainer_triage_rows(result)
    if not rows:
        return
    escaped_rows = [(label, escaper.escape(value)) for label, value in rows]
    _append_markdown_row_block(parts, rows=escaped_rows, heading=heading)


def _review_hint_text(
    review: JsonlReviewRecord,
    analysis: GenerationQualityAnalysis | None,
) -> str:
    """Return a concise trusted-hint summary for review surfaces."""
    if analysis is None:
        return "not evaluated"
    parts = [review["hint_relationship"].replace("_", " ")]
    if analysis.is_context_ignored and analysis.missing_context_terms:
        parts.append("missing terms: " + ", ".join(analysis.missing_context_terms))
    if analysis.metadata_borrowing:
        parts.append("nonvisual metadata reused")
    return " | ".join(parts)


def _review_contract_text(analysis: GenerationQualityAnalysis | None) -> str:
    """Return a compact contract-compliance summary."""
    if analysis is None:
        return "not evaluated"

    issues: list[str] = []
    if analysis.missing_sections:
        issues.append("missing: " + ", ".join(analysis.missing_sections))
    if analysis.has_title_length_violation:
        issues.append(f"title words={analysis.title_word_count}")
    if analysis.has_description_sentence_violation:
        issues.append(f"description sentences={analysis.description_sentence_count}")
    if analysis.has_keyword_count_violation:
        issues.append(f"keywords={analysis.keyword_count}")
    if analysis.has_keyword_duplication_violation:
        issues.append(f"keyword duplication={analysis.keyword_duplication_ratio:.2f}")
    return "ok" if not issues else " | ".join(issues)


def _review_utility_text(
    review: JsonlReviewRecord,
    analysis: GenerationQualityAnalysis | None,
) -> str:
    """Return a concise informational-utility summary."""
    parts = [f"user={review['user_bucket']}"]
    if analysis is not None:
        parts.append(review["hint_relationship"].replace("_", " "))
        if analysis.instruction_echo:
            parts.append("instruction echo")
        if analysis.metadata_borrowing:
            parts.append("metadata borrowing")
        if analysis.is_generic:
            parts.append("generic")
        if analysis.has_context_echo:
            parts.append("context echo")
    return " | ".join(parts)


def _review_stack_owner_text(
    result: PerformanceResult,
    review: JsonlReviewRecord,
    analysis: GenerationQualityAnalysis | None,
) -> str:
    """Return a compact stack-integrity and ownership summary."""
    parts = [f"owner={review['owner']}"]
    if analysis is not None and analysis.has_harness_issue:
        parts.append(f"harness={analysis.harness_issue_type or 'yes'}")
    if result.error_package:
        parts.append(f"package={result.error_package}")
    if result.error_stage:
        parts.append(f"stage={result.error_stage}")
    if result.error_code:
        parts.append(f"code={result.error_code}")
    return " | ".join(parts)


def _review_token_accounting_text(result: PerformanceResult, review: JsonlReviewRecord) -> str:
    """Return prompt/output token accounting for review surfaces."""
    generation_tokens = (
        getattr(result.generation, "generation_tokens", None)
        if result.generation is not None
        else None
    )
    stop_reason = result.runtime_diagnostics.stop_reason if result.runtime_diagnostics else None
    parts = [
        f"prompt={review['prompt_tokens_total'] if review['prompt_tokens_total'] is not None else 'n/a'}",
        (
            f"text_est={review['prompt_tokens_text_est']}"
            if review["prompt_tokens_text_est"] is not None
            else "text_est=n/a"
        ),
        (
            f"nontext_est={review['prompt_tokens_nontext_est']}"
            if review["prompt_tokens_nontext_est"] is not None
            else "nontext_est=n/a"
        ),
        f"gen={generation_tokens if generation_tokens is not None else 'n/a'}",
        f"max={review['requested_max_tokens'] if review['requested_max_tokens'] is not None else 'n/a'}",
    ]
    if stop_reason:
        parts.append(f"stop={stop_reason}")
    return " | ".join(parts)


def _humanize_review_evidence_label(label: str) -> str:
    """Return a human-readable fallback label for compact review evidence."""
    return label.replace("_", " ")


def _review_analysis_focus_parts(analysis: GenerationQualityAnalysis) -> list[str]:
    """Return compact focus fragments sourced directly from quality analysis."""
    parts: list[str] = []
    if (
        analysis.has_keyword_duplication_violation
        and analysis.keyword_duplication_ratio is not None
    ):
        parts.append(f"keyword duplication={analysis.keyword_duplication_ratio:.0%}")
    elif analysis.has_keyword_count_violation and analysis.keyword_count is not None:
        parts.append(f"keywords={analysis.keyword_count}")

    if analysis.has_context_echo and analysis.context_echo_ratio > 0:
        parts.append(f"context echo={analysis.context_echo_ratio:.0%}")
    if analysis.metadata_borrowing:
        parts.append("nonvisual metadata reused")
    if analysis.has_reasoning_leak:
        parts.append("reasoning leak")
    if analysis.formatting_issues:
        formatting_preview = html.escape(
            _collapse_preview_whitespace(analysis.formatting_issues[0]),
            quote=False,
        )
        parts.append(f"formatting={formatting_preview}")
    if analysis.has_excessive_bullets:
        parts.append(f"excessive bullets={analysis.bullet_count}")
    if analysis.has_degeneration and analysis.degeneration_type is not None:
        parts.append(f"degeneration={analysis.degeneration_type}")
    if analysis.is_repetitive and analysis.repeated_token is not None:
        safe_token = analysis.repeated_token.replace("*", r"\*")
        parts.append(f"repetitive token={safe_token}")
    return parts


def _review_focus_text(
    review: JsonlReviewRecord,
    analysis: GenerationQualityAnalysis | None,
) -> str:
    """Return compact evidence text tuned for human review surfaces."""
    parts: list[str] = []

    harness_descriptions = [
        description
        for detail in review["harness_details"][:2]
        if (description := _describe_harness_detail(detail))
    ]
    parts.extend(harness_descriptions)

    if review["hit_max_tokens"] and review["requested_max_tokens"] is not None:
        parts.append(f"hit token cap ({review['requested_max_tokens']})")

    if review["prompt_output_ratio"] is not None and review["verdict"] in {
        "cutoff",
        "context_budget",
    }:
        parts.append(f"output/prompt={review['prompt_output_ratio']:.2%}")

    if (
        review["nontext_prompt_ratio"] is not None
        and review["nontext_prompt_ratio"] >= HEAVY_NON_TEXT_PROMPT_RATIO_THRESHOLD
    ):
        parts.append(f"nontext prompt burden={review['nontext_prompt_ratio']:.0%}")

    if review["missing_sections"]:
        parts.append("missing sections: " + ", ".join(review["missing_sections"]))

    if review["missing_terms"]:
        parts.append("missing terms: " + ", ".join(review["missing_terms"]))

    if analysis is not None:
        parts.extend(_review_analysis_focus_parts(analysis))

    if not parts and review["evidence"]:
        parts.extend(_humanize_review_evidence_label(label) for label in review["evidence"][:3])

    return " | ".join(_dedupe_preserve_order(parts[:4])) or "no flagged signals"


def _review_cutoff_next_action(review: JsonlReviewRecord) -> str:
    """Return evidence-specific next action for cutoff verdicts."""
    action = (
        "Inspect stop behavior and tail quality before treating this as a model-quality failure."
    )
    if review["hit_max_tokens"] and review["missing_sections"]:
        action = (
            "Raise the token cap or trim prompt burden first; generation hit the limit "
            f"while {', '.join(review['missing_sections'])} remained incomplete."
        )
    elif review["hit_max_tokens"] and review["prompt_output_ratio"] is not None:
        action = (
            "Treat this as cap-limited output first; generation exhausted the token budget "
            f"with output/prompt={review['prompt_output_ratio']:.2%}."
        )
    return action


def _review_owner_specific_next_action(review: JsonlReviewRecord) -> str | None:
    """Return owner-tuned action text when evidence allows a more specific hint."""
    owner = review["owner"]
    harness_details = tuple(review["harness_details"])
    action: str | None = None

    if owner == "mlx-vlm":
        if any(detail.startswith("token_leak:") for detail in harness_details):
            action = "Inspect EOS/stop-token stripping; control tokens are leaking into user-facing text."
        elif any(detail.startswith("token_encoding:") for detail in harness_details):
            action = "Inspect decode cleanup; tokenizer markers are leaking into user-facing text."
        elif any(detail.startswith("training_leak:") for detail in harness_details):
            action = (
                "Inspect continuation and stop handling; generation is drifting into template text."
            )
    elif (
        owner == "mlx"
        and review["nontext_prompt_ratio"] is not None
        and review["nontext_prompt_ratio"] >= HEAVY_NON_TEXT_PROMPT_RATIO_THRESHOLD
    ):
        action = "Inspect long-context cache behavior under heavy image-token burden."
    elif owner == "model-config" and review["missing_sections"]:
        action = (
            "Check chat-template and EOS defaults first; the output shape is not matching the "
            "requested contract."
        )
    elif owner == "model":
        if review["missing_sections"]:
            action = (
                "Treat as a model limitation for this prompt; the requested output contract is "
                "not being met."
            )
        elif review["missing_terms"]:
            action = (
                "Treat as a model limitation for this prompt; trusted hint coverage is still weak."
            )

    return action


def _review_next_action_text(review: JsonlReviewRecord) -> str:
    """Return one actionable next-step line for developers or users."""
    action: str | None = None
    if review["verdict"] == "cutoff":
        action = _review_cutoff_next_action(review)
    elif review["verdict"] == "context_budget":
        action = (
            "Treat this as a prompt-budget issue first; nontext prompt burden is "
            f"{review['nontext_prompt_ratio']:.0%} and the output stays weak under that load."
            if review["nontext_prompt_ratio"] is not None
            else "Reduce prompt/image burden or inspect long-context handling before judging quality."
        )
    elif review["owner"] == "huggingface-hub":
        action = (
            "Check whether Hugging Face was reachable; this may be a transient Hub/network "
            "outage or disconnect rather than a model defect."
            if "hub_connectivity" in review["evidence"]
            else "Check cache/revision availability and network/auth state before blaming the model."
        )
    else:
        action = _review_owner_specific_next_action(review)

    if action is not None:
        return action

    owner_actions: dict[str, str] = {
        "mlx-vlm": "Inspect prompt-template, stop-token, and decode post-processing behavior.",
        "mlx": "Inspect KV/cache behavior, memory pressure, and long-context execution.",
        "mlx-lm": "Inspect tokenizer/generation stack shared with mlx-lm.",
        "transformers": "Inspect upstream template/tokenizer/config compatibility.",
        "huggingface-hub": (
            "Check cache/revision availability and network/auth state before blaming the model."
        ),
        "model-config": "Inspect model repo config, chat template, and EOS settings.",
        "model": "Treat as a model-quality limitation for this prompt and image.",
    }
    return owner_actions.get(review["owner"], "Inspect the canonical log and diagnostics output.")


def _build_review_payload(result: PerformanceResult) -> ReviewPayload | None:
    """Build the shared review payload once for downstream report renderers."""
    review = _build_jsonl_review_record(result)
    if review is None:
        return None

    analysis = _quality_analysis_for_result(result)
    return ReviewPayload(
        review=review,
        analysis=analysis,
        key_signals=_review_focus_text(review, analysis),
        next_action=_review_next_action_for_result(result, review),
    )


def _build_review_block_rows(result: PerformanceResult) -> list[tuple[str, str]]:
    """Build ordered canonical review rows shared by log and Markdown outputs."""
    payload = _build_review_payload(result)
    if payload is None:
        return []

    review = payload.review
    analysis = payload.analysis
    return [
        ("Verdict", f"{review['verdict']} | user={review['user_bucket']}"),
        ("Why", payload.key_signals),
        ("Trusted hints", _review_hint_text(review, analysis)),
        ("Contract", _review_contract_text(analysis)),
        ("Utility", _review_utility_text(review, analysis)),
        ("Stack / owner", _review_stack_owner_text(result, review, analysis)),
        ("Token accounting", _review_token_accounting_text(result, review)),
        ("Next action", payload.next_action),
    ]


def _markdown_review_text(text: str) -> str:
    """Return gallery-friendly text for compact review fragments."""
    return text.replace(" | ", "; ")


def _humanize_review_bucket_text(bucket: str) -> str:
    """Return user-facing phrasing for gallery recommendations."""
    bucket_labels = {
        "recommended": "recommended",
        "caveat": "use with caveats",
        "avoid": "avoid for now",
    }
    return bucket_labels.get(bucket, bucket.replace("_", " "))


def _humanize_review_verdict_text(verdict: str) -> str:
    """Return human-readable verdict text for gallery surfaces."""
    return verdict.replace("_", " ")


def _markdown_review_stack_owner_text(
    result: PerformanceResult,
    review: JsonlReviewRecord,
    analysis: GenerationQualityAnalysis | None,
) -> str:
    """Return gallery-friendly maintainer-routing text."""
    parts = [f"likely owner `{review['owner']}`"]
    if analysis is not None and analysis.has_harness_issue:
        parts.append(f"harness signal `{analysis.harness_issue_type or 'present'}`")
    if result.error_package:
        parts.append(f"reported package `{result.error_package}`")
    if result.error_stage:
        parts.append(f"failure stage `{result.error_stage}`")
    if result.error_code:
        parts.append(f"diagnostic code `{result.error_code}`")
    return "; ".join(parts)


def _markdown_review_token_accounting_text(
    result: PerformanceResult,
    review: JsonlReviewRecord,
) -> str:
    """Return gallery-friendly token summary text."""

    def _format_token_count(value: int | None) -> str:
        return f"{value} tok" if value is not None else "n/a"

    generation_tokens = (
        getattr(result.generation, "generation_tokens", None)
        if result.generation is not None
        else None
    )
    stop_reason = result.runtime_diagnostics.stop_reason if result.runtime_diagnostics else None
    parts = [
        f"prompt {_format_token_count(review['prompt_tokens_total'])}",
        f"estimated text {_format_token_count(review['prompt_tokens_text_est'])}",
        f"estimated non-text {_format_token_count(review['prompt_tokens_nontext_est'])}",
        f"generated {_format_token_count(generation_tokens)}",
        f"requested max {_format_token_count(review['requested_max_tokens'])}",
    ]
    if stop_reason:
        parts.append(f"stop reason {stop_reason}")
    return "; ".join(parts)


def _build_markdown_review_block_rows(result: PerformanceResult) -> list[tuple[str, str]]:
    """Build clearer review rows for gallery-style Markdown artifacts."""
    payload = _build_review_payload(result)
    if payload is None:
        return []

    review = payload.review
    analysis = payload.analysis
    return [
        (
            "Recommendation",
            (
                f"{_humanize_review_bucket_text(review['user_bucket'])}; "
                f"review verdict: {_humanize_review_verdict_text(review['verdict'])}"
            ),
        ),
        ("Owner", _markdown_review_stack_owner_text(result, review, analysis)),
        ("Next step", payload.next_action),
        ("Key signals", _markdown_review_text(payload.key_signals)),
        ("Tokens", _markdown_review_token_accounting_text(result, review)),
    ]


def _summarize_context_ignored_signal(qa: GenerationQualityAnalysis) -> str | None:
    """Return contextual-miss signal text when available."""
    if not qa.is_context_ignored:
        return None
    if qa.missing_context_terms:
        return (
            "Model output may not follow prompt or image contents "
            f"(missing: {', '.join(qa.missing_context_terms)})."
        )
    return "Model output may not follow prompt or image contents."


def _summarize_repetition_signal(qa: GenerationQualityAnalysis) -> str | None:
    """Return repetition signal text when available."""
    if not qa.is_repetitive:
        return None
    if qa.repeated_token is not None:
        return (
            "Output became repetitive, indicating possible generation instability "
            f"(token: {qa.repeated_token})."
        )
    return "Output became repetitive, indicating possible generation instability."


def _summarize_degeneration_signal(qa: GenerationQualityAnalysis) -> str | None:
    """Return degeneration signal text when available."""
    if not qa.has_degeneration:
        return None
    if qa.degeneration_type is not None:
        return f"Output contains corrupted or malformed text segments ({qa.degeneration_type})."
    return "Output contains corrupted or malformed text segments."


def _summarize_language_signal(qa: GenerationQualityAnalysis) -> str | None:
    """Return language-mixing signal text when available."""
    if not qa.has_language_mixing:
        return None
    if qa.language_mixing_issues:
        return (
            "Output switched language/script unexpectedly "
            f"({', '.join(qa.language_mixing_issues)})."
        )
    return "Output switched language/script unexpectedly."


def _summarize_refusal_signal(qa: GenerationQualityAnalysis) -> str | None:
    """Return refusal signal text when available."""
    if not qa.is_refusal:
        return None
    if qa.refusal_type is not None:
        return f"Model refused or deflected the requested task ({qa.refusal_type})."
    return "Model refused or deflected the requested task."


def _summarize_formatting_signal(qa: GenerationQualityAnalysis) -> str | None:
    """Return formatting signal text when available."""
    if not qa.formatting_issues:
        return None
    return (
        "Output formatting deviated from the requested structure. "
        f"Details: {'; '.join(qa.formatting_issues[:2])}."
    )


def _summarize_missing_sections_signal(qa: GenerationQualityAnalysis) -> str | None:
    """Return missing-section signal text when available."""
    if not qa.missing_sections:
        return None
    return (
        "Output omitted required Title/Description/Keywords sections "
        f"({', '.join(qa.missing_sections)})."
    )


def _summarize_reasoning_leak_signal(qa: GenerationQualityAnalysis) -> str | None:
    """Return reasoning-leak signal text when available."""
    if not qa.has_reasoning_leak:
        return None
    if qa.reasoning_leak_markers:
        return (
            "Output leaked reasoning or prompt-template text "
            f"({', '.join(qa.reasoning_leak_markers[:2])})."
        )
    return "Output leaked reasoning or prompt-template text."


def _summarize_context_echo_signal(qa: GenerationQualityAnalysis) -> str | None:
    """Return context-echo signal text when available."""
    if not qa.has_context_echo:
        return None
    return f"Output appears to copy prompt context verbatim ({qa.context_echo_ratio:.0%} overlap)."


def _failure_review_text(result: PerformanceResult) -> str:
    """Return the full failure text preserved in canonical log output."""
    if result.captured_output_on_fail:
        return result.captured_output_on_fail
    if result.error_traceback:
        return result.error_traceback
    return result.error_message or "No captured failure output."


def _log_canonical_model_review(result: PerformanceResult) -> None:
    """Emit a full-fidelity per-model review block to the file log."""
    logger.debug("")
    logger.debug("=== CANONICAL REVIEW: %s ===", result.model_name)
    for label, value in _build_review_block_rows(result):
        logger.debug("%s: %s", label, value)
    if result.success and result.generation is not None:
        logger.debug("Full output:\n%s", getattr(result.generation, "text", "") or "")
    else:
        logger.debug("Full captured failure output:\n%s", _failure_review_text(result))
    logger.debug("=== END CANONICAL REVIEW: %s ===", result.model_name)


def _log_canonical_run_review_summary(results: Sequence[PerformanceResult]) -> None:
    """Emit a grouped run-level review summary for the canonical log."""
    owner_map: dict[str, list[str]] = {}
    bucket_map: dict[str, list[str]] = {}
    evidence_counts: Counter[str] = Counter()

    for result in results:
        review = _build_jsonl_review_record(result)
        if review is None:
            continue
        owner_map.setdefault(review["owner"], []).append(
            f"{result.model_name} ({review['verdict']})",
        )
        bucket_map.setdefault(review["user_bucket"], []).append(result.model_name)
        evidence_counts.update(review["evidence"])

    if not owner_map and not bucket_map:
        return

    logger.debug("")
    logger.debug("=== RUN REVIEW SUMMARY ===")
    logger.debug("Maintainer queue by owner:")
    for owner in sorted(owner_map):
        logger.debug("  %s: %s", owner, ", ".join(owner_map[owner]))
    logger.debug("User buckets:")
    for bucket in sorted(bucket_map):
        logger.debug("  %s: %s", bucket, ", ".join(bucket_map[bucket]))
    if evidence_counts:
        logger.debug(
            "Top recurring verdict causes: %s",
            ", ".join(f"{label}={count}" for label, count in evidence_counts.most_common(8)),
        )
    logger.debug("=== END RUN REVIEW SUMMARY ===")


def _recommendation_candidate_rows(
    report_context: ReportRenderContext,
) -> list[UtilityTriageRow]:
    """Return rows safe enough to use for user-facing model picks."""
    rows = [
        row
        for row in report_context.triage.utility_rows
        if _is_cataloging_pick_eligible(row.result, _quality_analysis_for_result(row.result))
    ]
    actionable_rows = [
        row
        for row in rows
        if row.grade in {"A", "B", "C"} and not (row.labels & QUALITY_BREAKING_LABELS)
    ]
    return actionable_rows or rows


def _select_recommended_models(
    report_context: ReportRenderContext,
) -> list[tuple[str, PerformanceResult, tuple[float, str, str, float | None] | None]]:
    """Select compact recommendation targets for summary and gallery navigation."""
    successful_results = list(report_context.result_set.successful)
    if not successful_results:
        return []

    results_by_name = {result.model_name: result for result in successful_results}
    score_index = _cataloging_score_index(report_context.summary)
    candidate_rows = _recommendation_candidate_rows(report_context)
    recommendations: list[
        tuple[str, PerformanceResult, tuple[float, str, str, float | None] | None]
    ] = []

    def _append_ranked_row(
        label: str,
        key: Callable[[UtilityTriageRow], tuple[float, float, float, str]],
    ) -> None:
        if not candidate_rows:
            return
        row = max(candidate_rows, key=key)
        recommendations.append(
            (
                label,
                row.result,
                score_index.get(row.result.model_name),
            ),
        )

    _append_ranked_row(
        "Best end-to-end cataloging",
        lambda row: (
            row.score,
            row.description_score + row.keyword_score,
            getattr(row.result.generation, "generation_tps", 0.0) or 0.0,
            row.result.model_name,
        ),
    )
    _append_ranked_row(
        "Best descriptions",
        lambda row: (
            row.description_score,
            row.score,
            getattr(row.result.generation, "generation_tps", 0.0) or 0.0,
            row.result.model_name,
        ),
    )
    _append_ranked_row(
        "Best keywording",
        lambda row: (
            row.keyword_score,
            row.score,
            getattr(row.result.generation, "generation_tps", 0.0) or 0.0,
            row.result.model_name,
        ),
    )

    fastest_model = report_context.summary.get("fastest_model")
    if fastest_model is not None:
        model_name = fastest_model[0]
        if model_name in results_by_name:
            recommendations.append(
                ("Fastest generation", results_by_name[model_name], score_index.get(model_name)),
            )

    efficient_model = report_context.summary.get("most_efficient_model")
    if efficient_model is not None:
        model_name = efficient_model[0]
        if model_name in results_by_name:
            recommendations.append(
                (
                    "Lowest memory footprint",
                    results_by_name[model_name],
                    score_index.get(model_name),
                ),
            )

    balance_candidates: list[
        tuple[float, float, float, PerformanceResult, tuple[float, str, str, float | None] | None]
    ] = []
    for row in candidate_rows:
        result = row.result
        score_data = score_index.get(result.model_name)
        generation_tps = getattr(result.generation, "generation_tps", 0.0) or 0.0
        peak_memory = getattr(result.generation, "peak_memory", float("inf")) or float("inf")
        if not row.labels:
            balance_candidates.append((row.score, generation_tps, -peak_memory, result, score_data))

    if not balance_candidates:
        for row in candidate_rows:
            result = row.result
            score_data = score_index.get(result.model_name)
            generation_tps = getattr(result.generation, "generation_tps", 0.0) or 0.0
            peak_memory = getattr(result.generation, "peak_memory", float("inf")) or float("inf")
            balance_candidates.append((row.score, generation_tps, -peak_memory, result, score_data))

    if balance_candidates:
        _score, _tps, _neg_peak, result, score_data = max(
            balance_candidates,
            key=lambda item: (item[0], item[1], item[2], item[3].model_name),
        )
        recommendations.append(("Best balance", result, score_data))

    deduped: list[tuple[str, PerformanceResult, tuple[float, str, str, float | None] | None]] = []
    seen_labels: set[str] = set()
    for label, result, score_data in recommendations:
        if label in seen_labels:
            continue
        seen_labels.add(label)
        deduped.append((label, result, score_data))
    return deduped


def _preview_model_references(
    model_names: Sequence[str],
    *,
    gallery_relative_path: str | None = None,
    max_items: int = 4,
) -> str:
    """Format a compact preview of model names, optionally linking to the gallery."""
    unique_names = list(dict.fromkeys(model_names))
    rendered_names = [
        _format_gallery_model_link(
            model_name,
            gallery_relative_path=gallery_relative_path,
        )
        if gallery_relative_path is not None
        else f"`{model_name}`"
        for model_name in unique_names[:max_items]
    ]
    if len(unique_names) > max_items:
        rendered_names.append(f"+{len(unique_names) - max_items} more")
    return ", ".join(rendered_names)


def _build_markdown_recommended_models(
    report_context: ReportRenderContext,
    *,
    gallery_relative_path: str | None = None,
) -> list[str]:
    """Build a recommendation section for the Markdown summary report."""
    recommendations = _select_recommended_models(report_context)
    if not recommendations:
        return []

    triage_by_name = {row.result.model_name: row for row in report_context.triage.utility_rows}
    parts: list[str] = []
    _append_markdown_section(
        parts,
        title="## ✅ Recommended Models",
        body_lines=[
            "Quick picks based on end-to-end utility plus description and keyword strength.",
        ],
    )
    for label, result, score_data in recommendations:
        model_ref = (
            _format_gallery_model_link(
                result.model_name,
                gallery_relative_path=gallery_relative_path,
            )
            if gallery_relative_path is not None
            else f"`{result.model_name}`"
        )
        rationale: list[str] = []
        if score_data is not None:
            score, grade, _weakness, _delta = score_data
            rationale.append(f"Utility {grade} {score:.0f}/100")
        triage_row = triage_by_name.get(result.model_name)
        if triage_row is not None:
            rationale.append(f"Description {triage_row.description_score:.0f}")
            rationale.append(f"Keywords {triage_row.keyword_score:.0f}")
        generation_tps = format_field_value(
            "generation_tps",
            cast("MetricValue", getattr(result.generation, "generation_tps", None)),
        )
        if generation_tps:
            rationale.append(f"Speed {generation_tps} TPS")
        peak_memory = format_field_value(
            "peak_memory",
            cast("MetricValue", getattr(result.generation, "peak_memory", None)),
        )
        if peak_memory:
            rationale.append(f"Memory {peak_memory}")
        review = _build_jsonl_review_record(result)
        analysis = _quality_analysis_for_result(result)
        if review is not None:
            focus = _review_focus_text(review, analysis)
            if focus != "no flagged signals":
                rationale.append(f"Caveat {_markdown_review_text(focus)}")
        _append_markdown_labeled_value(
            parts,
            label=label,
            value=f"{model_ref} ({' | '.join(rationale)})",
            bullet=True,
        )
    parts.append("")
    return parts


def _build_markdown_quality_breakdown(
    report_context: ReportRenderContext,
    *,
    gallery_relative_path: str | None = None,
) -> list[str]:
    """Build a compact quality-pattern breakdown for the Markdown summary report."""
    summary = report_context.summary
    sections = _collect_quality_issue_sections(summary)
    low_utility_models = summary.get("low_utility_models", [])
    if not sections and not low_utility_models:
        return []

    parts: list[str] = []
    _append_markdown_section(
        parts,
        title="## 🔍 Quality Pattern Breakdown",
        body_lines=[
            (
                "Common weaknesses and failure patterns from this run, linked to the "
                "gallery when available."
            ),
        ],
    )
    for title, _css_class, entries in sections:
        models = [model for model, _detail in entries]
        preview = _preview_model_references(models, gallery_relative_path=gallery_relative_path)
        details = [detail for _model, detail in entries if detail]
        detail_text = (
            f" Example: {_format_quality_detail(details[0], html_output=False)}." if details else ""
        )
        _append_markdown_labeled_value(
            parts,
            label=f"{title} ({len(entries)})",
            value=f"{preview}.{detail_text}",
            bullet=True,
        )

    if low_utility_models:
        model_names = [model_name for model_name, _score, _grade, _weakness in low_utility_models]
        weakest_model = low_utility_models[0]
        model_preview = _preview_model_references(
            model_names,
            gallery_relative_path=gallery_relative_path,
        )
        _append_markdown_labeled_value(
            parts,
            label=f"Low-utility outputs ({len(low_utility_models)})",
            value=(
                f"{model_preview}. Common weakness: {html.escape(weakest_model[3], quote=False)}."
            ),
            bullet=True,
        )

    parts.append("")
    return parts


def _build_markdown_gallery_navigation(report_context: ReportRenderContext) -> list[str]:
    """Build a quick-navigation section for the standalone gallery artifact."""
    recommended = _select_recommended_models(report_context)
    failed_models = [
        model_name
        for model_name, _stage, _message in report_context.summary.get(
            "failed_models",
            [],
        )
    ]
    low_utility_models = [
        model_name
        for model_name, _score, _grade, _weakness in report_context.summary.get(
            "low_utility_models",
            [],
        )
    ]
    if not recommended and not failed_models and not low_utility_models:
        return []

    parts: list[str] = []
    _append_markdown_section(parts, title="## Quick Navigation")
    for label, result, _score_data in recommended:
        _append_markdown_labeled_value(
            parts,
            label=label,
            value=_format_gallery_model_link(result.model_name),
            bullet=True,
        )
    if failed_models:
        failed_preview = _preview_model_references(failed_models, max_items=6)
        _append_markdown_labeled_value(
            parts,
            label="Failed models",
            value=failed_preview,
            bullet=True,
        )
    if low_utility_models:
        low_utility_preview = _preview_model_references(low_utility_models, max_items=6)
        _append_markdown_labeled_value(
            parts,
            label="D/F utility models",
            value=low_utility_preview,
            bullet=True,
        )
    parts.append("")
    return parts


# =============================================================================
# DIAGNOSTICS REPORT — Upstream issue filing aid
# =============================================================================


@dataclass(frozen=True)
class DiagnosticsConfig:
    """Centralized configuration for diagnostics report behavior."""

    traceback_tail_lines: int = 6  # Lines to keep from traceback tail
    output_snippet_len: int = 200  # Max chars for sample output
    recent_run_window: int = 3  # Runs used for reproducibility signal
    history_max_records: int = 100  # History rows loaded for context


DIAGNOSTICS: Final[DiagnosticsConfig] = DiagnosticsConfig()

_DIAGNOSTICS_CAPTURE_NOISE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(
        r"^\s*(?:Downloading|Fetching|Resolving|Loading checkpoint shards)\b.*$", re.IGNORECASE
    ),
    re.compile(r"^\s*(?:Download complete|Download finished)\b.*$", re.IGNORECASE),
    re.compile(r"^\s*(?:\d+%|\d+/\d+)\s*\|.*\|.*$"),
    re.compile(r"^\s*[A-Za-z][A-Za-z0-9 _/:-]{0,48}:\s*\d+%\s*$"),
)

# System info keys to include in the environment table (order matters)
_DIAGNOSTICS_SYSTEM_KEYS: Final[tuple[str, ...]] = (
    "Python Version",
    "OS",
    "macOS Version",
    "GPU/Chip",
    "GPU Cores",
    "Metal Support",
    "RAM",
)

# Library names to include in the environment table (order matters)
_DIAGNOSTICS_LIB_NAMES: Final[tuple[str, ...]] = (
    "mlx-vlm",
    "mlx",
    "mlx-lm",
    "transformers",
    "tokenizers",
    "huggingface-hub",
)

_PORTABLE_DEPENDENCY_PROBE_CMD: Final[str] = (
    "python -m pip show mlx mlx-vlm mlx-lm transformers huggingface-hub tokenizers"
)
_PORTABLE_PROBE_IMAGE_PATH: Final[str] = "./check_models_portable_probe.png"

ISSUE_EVIDENCE_MODEL_LIMIT: Final[int] = 3


@dataclass(frozen=True)
class FailureHistoryContext:
    """History-derived context for a single failing model."""

    first_failure_timestamp: str = "unknown"
    recent_failures: int = 0
    recent_considered: int = 0


@dataclass(frozen=True)
class DiagnosticsContext:
    """Shared diagnostics context built once and consumed by renderers."""

    regressions: frozenset[str] = frozenset()
    recoveries: frozenset[str] = frozenset()
    new_models: frozenset[str] = frozenset()
    missing_models: frozenset[str] = frozenset()
    failure_history: dict[str, FailureHistoryContext] = dataclass_field(default_factory=dict)


@dataclass(frozen=True)
class DiagnosticsSnapshot:
    """Cached diagnostics classification derived from a completed run."""

    failed: tuple[PerformanceResult, ...] = ()
    harness_results: tuple[tuple[PerformanceResult, str], ...] = ()
    stack_signals: tuple[tuple[PerformanceResult, str, str], ...] = ()
    unflagged_successful: tuple[PerformanceResult, ...] = ()
    preflight_issues: tuple[str, ...] = ()
    failure_clusters: tuple[tuple[str, tuple[PerformanceResult, ...]], ...] = ()


@dataclass(frozen=True)
class IssueCluster:
    """Root-cause issue draft cluster for maintainer-facing artifacts."""

    cluster_id: str
    issue_filename: str
    owner: str
    issue_kind: str
    issue_subtype: str
    symptom_family: str
    symptom: str
    acceptance_signal: str
    source: str
    sort_rank: int
    results: tuple[PerformanceResult, ...] = ()
    stack_signals: tuple[tuple[PerformanceResult, str, str], ...] = ()


@dataclass
class _IssueClusterBuildState:
    """Mutable accumulator used while building deterministic issue clusters."""

    owner: str
    issue_kind: str
    issue_subtype: str
    symptom_family: str
    symptom: str
    source: str
    sort_rank: int
    results: list[PerformanceResult] = dataclass_field(default_factory=list)
    stack_signals: list[tuple[PerformanceResult, str, str]] = dataclass_field(default_factory=list)


@dataclass(frozen=True)
class PreparedTableData:
    """Immutable cached table data shared across report renderers."""

    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    field_names: tuple[str, ...]


@dataclass(frozen=True)
class ReportTriageContext:
    """Shared quality/utility triage context reused across report renderers."""

    quality_counts: tuple[tuple[str, int], ...] = ()
    clean_count: int = 0
    utility_rows: tuple[UtilityTriageRow, ...] = ()
    useful_rows: tuple[UtilityTriageRow, ...] = ()
    watchlist_rows: tuple[tuple[UtilityTriageRow, str], ...] = ()
    baseline_score: float | None = None
    baseline_grade: str | None = None


@dataclass(frozen=True)
class ReportRenderContext:
    """Shared cached context for HTML/Markdown/TSV report generation."""

    result_set: ResultSet
    table_data: PreparedTableData
    prompt_context: str | None
    summary: ModelIssueSummary
    stats: PerformanceStats
    system_info: dict[str, str]
    triage: ReportTriageContext
    preflight_issues: tuple[str, ...] = ()


def _append_markdown_code_block(
    parts: list[str],
    content: str,
    *,
    language: str = "text",
) -> None:
    """Append a fenced code block with consistent spacing.

    Uses a fence longer than any backtick run in ``content`` so nested
    Markdown/code fences remain valid when embedded.
    """
    max_backtick_run = max((len(m.group(0)) for m in re.finditer(r"`+", content)), default=0)
    fence = "`" * max(3, max_backtick_run + 1)
    if not parts or parts[-1] != "":
        parts.append("")
    parts.append(f"{fence}{language}")
    parts.append(content)
    parts.append(fence)
    if not parts or parts[-1] != "":
        parts.append("")


def _is_markdown_blockquote_label_line(text: str) -> bool:
    """Return whether escaped blockquote text is just a short label line."""
    emphasis_wrapper = "&#42;&#42;"
    normalized = text
    if normalized.startswith(emphasis_wrapper) and normalized.endswith(emphasis_wrapper):
        normalized = normalized[len(emphasis_wrapper) : -len(emphasis_wrapper)]
    return re.fullmatch(r"[A-Za-z][A-Za-z0-9 \"'()/,&.-]{0,80}:", normalized) is not None


def _neutralize_markdown_blockquote_prefix(text: str) -> str:
    """Render leading Markdown control syntax as plain text in blockquotes."""
    replacement = text
    if text.startswith("[!"):
        replacement = f"&#91;{text[1:]}"
    else:
        setext_match = re.fullmatch(r"([=-])\1{2,}\s*", text)
        if setext_match is not None:
            underline_entity = "&#61;" if setext_match.group(1) == "=" else "&#45;"
            replacement = f"{underline_entity}{text[1:]}"
        else:
            bullet_entities: dict[str, str] = {
                "*": "&#42;",
                "+": "&#43;",
                "-": "&#45;",
            }
            lone_bullet_match = re.fullmatch(r"([*+-])", text)
            if lone_bullet_match is not None:
                replacement = bullet_entities[lone_bullet_match.group(1)]
            else:
                ordered_match = re.match(r"^(\d+)([.)])(\s)", text)
                if ordered_match is not None:
                    punctuation_entity = "&#46;" if ordered_match.group(2) == "." else "&#41;"
                    replacement = (
                        f"{ordered_match.group(1)}{punctuation_entity}{ordered_match.group(3)}"
                        f"{text[ordered_match.end() :]}"
                    )
                else:
                    bullet_match = re.match(r"^([*+-])(\s)", text)
                    if bullet_match is not None:
                        replacement = (
                            f"{bullet_entities[bullet_match.group(1)]}{bullet_match.group(2)}"
                            f"{text[bullet_match.end() :]}"
                        )
                    else:
                        prefix_entities: dict[str, str] = {
                            "#": "&#35;",
                            ">": "&gt;",
                            "`": "&#96;",
                        }
                        leading_char = text[:1]
                        if leading_char in prefix_entities:
                            replacement = f"{prefix_entities[leading_char]}{text[1:]}"
                        elif _is_markdown_blockquote_label_line(text):
                            replacement = f"&#8203;{text}"

    return replacement


def _escape_markdown_blockquote_line(text: str) -> str:
    """Escape structural Markdown syntax for wrapped blockquote text."""
    escaped: str = HTML_ESCAPER.escape(_wrap_bare_urls(text)).replace("__", r"\_\_")
    escaped = escaped.replace("*", "&#42;")
    escaped = re.sub(r"&(?!lt;|gt;|amp;|#)", "&amp;", escaped)
    return _neutralize_markdown_blockquote_prefix(escaped)


def _append_markdown_wrapped_blockquote(
    parts: list[str],
    content: str,
    *,
    width: int = FORMATTING.markdown_wrap_width - 2,
) -> None:
    """Append a wrapping plain blockquote for human-facing text blocks.

    This path is for prompt/model-output readability. Technical logs and
    reproducibility commands should continue to use fenced code blocks.
    """
    if not parts or parts[-1] != "":
        parts.append("")
    parts.append("<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->")
    blockquote_lines: list[str] = [">"]

    normalized: str = content.replace("\r\n", "\n").replace("\r", "\n")
    for raw_line in normalized.split("\n"):
        if raw_line == "":
            blockquote_lines.append(">")
            continue
        wrapped_lines: list[str] = textwrap.wrap(
            raw_line,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
            drop_whitespace=False,
        )
        if not wrapped_lines:
            blockquote_lines.append(">")
            continue
        for wrapped_line in wrapped_lines:
            clean_line = wrapped_line.lstrip().rstrip(" \t\u00a0\u200b\u200c\u200d\u2060\ufeff")
            blockquote_lines.append(
                ">" if clean_line == "" else f"> {_escape_markdown_blockquote_line(clean_line)}"
            )

    parts.extend(blockquote_lines)
    parts.append("<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->")
    if not parts or parts[-1] != "":
        parts.append("")


def _append_markdown_image_metadata_section(
    parts: list[str],
    metadata: MetadataDict | None,
) -> None:
    """Append selected image metadata fields for Markdown gallery output."""
    if not metadata:
        return

    metadata_fields: tuple[tuple[str, str | None], ...] = (
        ("Title", metadata.get("title")),
        ("Description", metadata.get("description")),
        ("Keywords", metadata.get("keywords")),
        ("Date", metadata.get("date")),
        ("Time", metadata.get("time")),
        ("GPS", metadata.get("gps")),
    )
    populated_fields: list[tuple[str, str]] = [
        (label, value) for label, value in metadata_fields if value is not None
    ]
    if not populated_fields:
        return

    parts.append("## Image Metadata")
    parts.append("")
    for label, value in populated_fields:
        normalized_value = value.replace("\r\n", "\n").replace("\r", "\n")
        paragraphs = [
            paragraph.strip() for paragraph in normalized_value.split("\n\n") if paragraph.strip()
        ]
        if not paragraphs:
            continue

        first_paragraph_lines = [
            line.strip() for line in paragraphs[0].splitlines() if line.strip()
        ]
        first_line = first_paragraph_lines[0] if first_paragraph_lines else ""
        _append_markdown_labeled_value(parts, label=label, value=first_line, bullet=True)

        remaining_lines = first_paragraph_lines[1:]
        if remaining_lines:
            parts.append("")
            parts.extend(f"    {line}" for line in remaining_lines)

        for paragraph in paragraphs[1:]:
            paragraph_lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
            if not paragraph_lines:
                continue
            parts.append("")
            parts.extend(f"    {line}" for line in paragraph_lines)
    parts.append("")


def _write_markdown_artifact(
    filename: Path,
    markdown_lines: Sequence[str],
    *,
    artifact_name: str,
) -> None:
    """Normalize and write Markdown artifact content with consistent error handling."""
    markdown_content: str = normalize_markdown_trailing_spaces("\n".join(markdown_lines)) + "\n"

    try:
        with filename.open("w", encoding="utf-8") as f:
            f.write(markdown_content)
        # Logging handled in finalize_execution
    except OSError:
        logger.exception("Failed to write %s to file %s.", artifact_name, str(filename))
    except ValueError:
        logger.exception(
            "A value error occurred while writing %s %s",
            artifact_name,
            str(filename),
        )


def _append_markdown_section(
    parts: list[str],
    *,
    title: str,
    body_lines: list[str] | None = None,
) -> None:
    """Append a Markdown section heading with optional body lines."""
    parts.append(title)
    parts.append("")
    if body_lines:
        for body_line in body_lines:
            if body_line == "":
                parts.append("")
                continue
            if body_line.startswith(("- ", "* ", "|", "<", "#", "```")):
                parts.append(body_line)
                continue
            if re.match(r"\d+\. ", body_line):
                parts.append(body_line)
                continue
            parts.extend(_wrap_markdown_text(body_line))
        parts.append("")


def _append_markdown_details_block(
    parts: list[str],
    *,
    summary: str,
    body_lines: Sequence[str],
) -> None:
    """Append an HTML <details> block for collapsed technical content."""
    normalized_body_lines = list(body_lines)
    while normalized_body_lines and normalized_body_lines[0] == "":
        normalized_body_lines.pop(0)
    while normalized_body_lines and normalized_body_lines[-1] == "":
        normalized_body_lines.pop()

    parts.append("<details>")
    parts.append(f"<summary>{html.escape(summary, quote=False)}</summary>")
    parts.append("")
    parts.extend(normalized_body_lines)
    if normalized_body_lines and re.fullmatch(r"`{3,}", normalized_body_lines[-1]):
        parts.append("")
    parts.append("</details>")
    parts.append("")


def _begin_diagnostics_section(
    *,
    title: str | None = None,
    body_lines: list[str] | None = None,
) -> list[str]:
    """Return a diagnostics section buffer with the standard divider and optional heading."""
    parts: list[str] = ["---", ""]
    if title is not None:
        _append_markdown_section(parts, title=title, body_lines=body_lines)
    return parts


_LOCAL_RUNNER_TRACEBACK_FRAME_RE: Final[re.Pattern[str]] = re.compile(
    r'^\s*File ".*(?:^|/)check_models\.py", line \d+, in ',
)


def _normalize_traceback_for_report(traceback_str: str | None) -> str | None:
    """Normalize traceback text for diagnostics output."""
    if not traceback_str:
        return None

    lines = traceback_str.splitlines()
    kept: list[str] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if _LOCAL_RUNNER_TRACEBACK_FRAME_RE.search(line):
            idx += 1
            while idx < len(lines):
                follow = lines[idx]
                if follow.lstrip().startswith('File "') or not follow.startswith(" "):
                    break
                idx += 1
            continue
        kept.append(line)
        idx += 1

    sanitized = "\n".join(kept).strip()
    return sanitized or None


def _format_traceback_tail(traceback_str: str | None) -> str | None:
    """Extract the last meaningful lines from a full traceback.

    Strips blank lines and returns the tail suitable for inclusion in issue
    reports.  Returns None when no useful info can be extracted.
    """
    normalized = _normalize_traceback_for_report(traceback_str)
    if normalized is None:
        return None
    lines = [ln for ln in normalized.splitlines() if ln.strip()]
    if not lines:
        return None
    tail = lines[-DIAGNOSTICS.traceback_tail_lines :]
    return "\n".join(tail)


def _build_failure_history_context(
    *,
    failed_models: set[str],
    history_records: list[HistoryRunRecord],
) -> dict[str, FailureHistoryContext]:
    """Compute first-seen and recent reproducibility stats for each failed model."""
    context: dict[str, FailureHistoryContext] = {}

    for model in failed_models:
        first_failure_timestamp = "unknown"
        for record in history_records:
            model_results = record.get("model_results", {})
            if not isinstance(model_results, dict):
                continue
            info = model_results.get(model)
            if isinstance(info, dict) and info.get("success") is False:
                timestamp = record.get("timestamp")
                if isinstance(timestamp, str) and timestamp:
                    first_failure_timestamp = timestamp
                break

        recent_considered = 0
        recent_failures = 0
        for record in reversed(history_records):
            model_results = record.get("model_results", {})
            if not isinstance(model_results, dict):
                continue
            info = model_results.get(model)
            if not isinstance(info, dict):
                continue
            success = info.get("success")
            if success not in {True, False}:
                continue
            recent_considered += 1
            if success is False:
                recent_failures += 1
            if recent_considered >= DIAGNOSTICS.recent_run_window:
                break

        context[model] = FailureHistoryContext(
            first_failure_timestamp=first_failure_timestamp,
            recent_failures=recent_failures,
            recent_considered=recent_considered,
        )

    return context


def _build_diagnostics_context(
    *,
    failed_models: set[str],
    history_records: list[HistoryRunRecord],
    comparison: HistoryComparisonSummary | None,
) -> DiagnosticsContext:
    """Build immutable diagnostics context from history + comparison data."""
    history_context = _build_failure_history_context(
        failed_models=failed_models,
        history_records=history_records,
    )
    if comparison is None:
        return DiagnosticsContext(failure_history=history_context)

    return DiagnosticsContext(
        regressions=frozenset(comparison["regressions"]),
        recoveries=frozenset(comparison["recoveries"]),
        new_models=frozenset(comparison["new_models"]),
        missing_models=frozenset(comparison["missing_models"]),
        failure_history=history_context,
    )


def _build_diagnostics_snapshot(
    *,
    results: list[PerformanceResult],
    prompt: str | None = None,
    preflight_issues: Sequence[str] = (),
) -> DiagnosticsSnapshot:
    """Build reusable diagnostics classification data from one completed run."""
    resolved_results = [
        _populate_result_quality_analysis(result, prompt=prompt) for result in results
    ]
    failed = tuple(r for r in resolved_results if not r.success)
    successful = [r for r in resolved_results if r.success]
    harness_detected = _collect_harness_results(successful)
    stack_detected = _collect_stack_issue_signals(
        successful,
        preflight_issues=preflight_issues,
    )
    harness_results, stack_signals, unflagged_successful = _partition_success_diagnostics(
        successful=successful,
        harness_results=harness_detected,
        stack_signals=stack_detected,
    )
    failure_clusters: tuple[tuple[str, tuple[PerformanceResult, ...]], ...] = ()
    if failed:
        failure_clusters = tuple(
            (signature, tuple(cluster_results))
            for signature, cluster_results in sorted(
                _cluster_failures_by_pattern(resolved_results).items(),
                key=lambda kv: -len(kv[1]),
            )
        )
    return DiagnosticsSnapshot(
        failed=failed,
        harness_results=tuple(harness_results),
        stack_signals=tuple(stack_signals),
        unflagged_successful=tuple(unflagged_successful),
        preflight_issues=tuple(preflight_issues),
        failure_clusters=failure_clusters,
    )


def _snapshot_has_maintainer_signals(snapshot: DiagnosticsSnapshot) -> bool:
    """Return True when a run produced maintainer-facing diagnostics."""
    return bool(
        snapshot.failed
        or snapshot.harness_results
        or snapshot.stack_signals
        or snapshot.preflight_issues,
    )


def _issue_component_slug(value: str) -> str:
    """Return a stable slug component for issue cluster IDs and filenames."""
    normalized = value.strip().casefold().replace("_", "-")
    normalized = re.sub(r"[^a-z0-9-]+", "-", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
    return normalized or "unknown"


def _issue_cluster_path(cluster: IssueCluster) -> str:
    """Return the JSONL/report-relative path for a generated issue draft."""
    return str(Path("issues") / cluster.issue_filename)


def _normalize_issue_symptom_family(value: str) -> str:
    """Collapse free-form symptoms into deterministic issue-clustering families."""
    normalized = re.sub(r"\b0x[0-9a-f]+\b", "0xADDR", value.casefold())
    normalized = re.sub(r"\b\d+(?:\.\d+)?%?\b", "N", normalized)
    normalized = re.sub(r"[^a-z0-9_<>/| -]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized[:120] or "unknown"


def _harness_symptom_family(
    issue_subtype: str,
    analysis: GenerationQualityAnalysis | None,
) -> str:
    """Return a normalized symptom family for harness issue grouping."""
    subtype = issue_subtype.strip().casefold() or "harness"
    if subtype in {"stop_token", "encoding", "prompt_template", "long_context"}:
        return subtype
    details = tuple(analysis.harness_issue_details if analysis is not None else ())
    if any(detail.startswith("token_leak:") for detail in details):
        return "stop_token"
    if any(detail.startswith("token_encoding:") for detail in details):
        return "encoding"
    if any(detail.startswith("output:") for detail in details):
        return "prompt_template"
    if any(detail.startswith("long_context_") for detail in details):
        return "long_context"
    return subtype


def _stack_signal_subtype(symptom: str) -> str:
    """Return a stable issue subtype for successful-run stack anomalies."""
    normalized = symptom.casefold()
    if "long-context" in normalized or "long prompt" in normalized or "context" in normalized:
        return "long_context"
    if "empty output" in normalized:
        return "empty_output"
    return "stack_signal"


def _issue_acceptance_signal(issue_subtype: str, issue_kind: str) -> str:
    """Return a concrete fixed-state signal for a clustered issue."""
    subtype = issue_subtype.casefold()
    subtype_signals = {
        "stop_token": (
            "Affected reruns contain no leaked stop/control tokens and terminate cleanly "
            "before the configured max-token cap when the response is complete."
        ),
        "encoding": "Affected reruns contain no leaked BPE, byte-level, or tokenizer marker text.",
        "prompt_template": (
            "Affected reruns produce the requested sections without empty/filler output, "
            "template leakage, or image-placeholder mismatch symptoms."
        ),
        "empty_output": (
            "Affected reruns return non-empty generated text for the same prompt and image."
        ),
    }
    if subtype in subtype_signals:
        return subtype_signals[subtype]
    if subtype in {"long_context", "context_budget", "token_cap"}:
        return (
            "A same-command rerun and a reduced image/text burden rerun show consistent "
            "prompt-token accounting and no long-context collapse."
        )
    if issue_kind == "runtime_failure" or subtype.endswith("_runtime"):
        return (
            "Affected reruns complete model load and generation, or fail with a narrower "
            "configuration/compatibility error that points to the owning layer."
        )
    return "Affected reruns no longer emit this maintainer-triage issue cluster."


def _issue_cluster_results(cluster: IssueCluster) -> tuple[PerformanceResult, ...]:
    """Return affected results in stable model-name order, deduping stack rows."""
    results_by_model: dict[str, PerformanceResult] = {
        result.model_name: result for result in cluster.results
    }
    for result, _symptom, _owner in cluster.stack_signals:
        results_by_model.setdefault(result.model_name, result)
    return tuple(results_by_model[name] for name in sorted(results_by_model))


def _issue_cluster_model_names(cluster: IssueCluster) -> list[str]:
    """Return affected model names for an issue cluster."""
    return [result.model_name for result in _issue_cluster_results(cluster)]


def _issue_cluster_representative(cluster: IssueCluster) -> PerformanceResult | None:
    """Return the deterministic representative result for a cluster."""
    results = _issue_cluster_results(cluster)
    return results[0] if results else None


def _issue_cluster_model_count(cluster: IssueCluster) -> int:
    """Return affected model count for an issue cluster."""
    return len(_issue_cluster_results(cluster))


def _issue_harness_symptom(
    issue_subtype: str,
    analysis: GenerationQualityAnalysis | None,
) -> str:
    """Return the issue title/body symptom for a harness issue cluster."""
    if analysis is not None:
        for detail in analysis.harness_issue_details:
            if description := _describe_harness_detail(detail):
                return description
    return _HARNESS_TYPE_DESCRIPTIONS.get(
        issue_subtype,
        "Output indicates a likely integration/runtime issue.",
    )


def _issue_state_for_key(
    states: dict[tuple[str, str, str, str], _IssueClusterBuildState],
    *,
    owner: str,
    issue_kind: str,
    issue_subtype: str,
    symptom_family: str,
    symptom: str,
    source: str,
    sort_rank: int,
) -> _IssueClusterBuildState:
    """Fetch or create a mutable issue-cluster accumulator."""
    key = (owner, issue_kind, issue_subtype, symptom_family)
    if key not in states:
        states[key] = _IssueClusterBuildState(
            owner=owner,
            issue_kind=issue_kind,
            issue_subtype=issue_subtype,
            symptom_family=symptom_family,
            symptom=symptom,
            source=source,
            sort_rank=sort_rank,
        )
    return states[key]


def _build_issue_clusters(snapshot: DiagnosticsSnapshot) -> tuple[IssueCluster, ...]:
    """Build deterministic root-cause clusters for generated issue drafts."""
    states: dict[tuple[str, str, str, str], _IssueClusterBuildState] = {}

    for _signature, cluster_results in snapshot.failure_clusters:
        if not cluster_results:
            continue
        sorted_results = sorted(cluster_results, key=lambda result: result.model_name)
        representative = sorted_results[0]
        owner = _diagnostics_owner_label(representative.error_package or "unknown")
        subtype = representative.error_code or "runtime_failure"
        representative_error = representative.root_error_message or representative.error_message
        symptom = _truncate_text_preview(
            _simplify_failure_message(
                representative_error,
                model_name=representative.model_name,
            ),
            max_chars=120,
        )
        state = _issue_state_for_key(
            states,
            owner=owner,
            issue_kind="runtime_failure",
            issue_subtype=subtype,
            symptom_family=_normalize_issue_symptom_family(
                f"{subtype} {representative_error or symptom}"
            ),
            symptom=symptom,
            source="failure",
            sort_rank=0,
        )
        state.results.extend(sorted_results)

    for result, _sample_output in snapshot.harness_results:
        analysis = _quality_analysis_for_result(result)
        review = _build_jsonl_review_record(result)
        subtype = (
            analysis.harness_issue_type
            if analysis is not None and analysis.harness_issue_type
            else "harness"
        )
        issue_kind = review["verdict"] if review is not None else "harness"
        owner = _infer_harness_issue_owner(result)
        source = (
            "context_budget"
            if subtype == "long_context" or issue_kind == "context_budget"
            else "harness"
        )
        state = _issue_state_for_key(
            states,
            owner=owner,
            issue_kind=issue_kind,
            issue_subtype=subtype,
            symptom_family=_harness_symptom_family(subtype, analysis),
            symptom=_issue_harness_symptom(subtype, analysis),
            source=source,
            sort_rank=2 if source == "context_budget" else 1,
        )
        state.results.append(result)

    for result, symptom, package_hint in snapshot.stack_signals:
        subtype = _stack_signal_subtype(symptom)
        state = _issue_state_for_key(
            states,
            owner=_diagnostics_owner_label(package_hint),
            issue_kind="stack_signal",
            issue_subtype=subtype,
            symptom_family=_normalize_issue_symptom_family(subtype),
            symptom=symptom,
            source="stack_signal",
            sort_rank=3,
        )
        state.stack_signals.append((result, symptom, package_hint))

    sorted_states = sorted(
        states.values(),
        key=lambda state: (
            state.sort_rank,
            state.owner,
            state.issue_subtype,
            state.symptom_family,
            min(
                [result.model_name for result in state.results]
                + [result.model_name for result, _symptom, _owner in state.stack_signals],
                default="",
            ),
        ),
    )

    ordinals: Counter[tuple[str, str]] = Counter()
    clusters: list[IssueCluster] = []
    for index, state in enumerate(sorted_states, start=1):
        owner_slug = _issue_component_slug(state.owner)
        subtype_slug = _issue_component_slug(state.issue_subtype)
        ordinal_key = (owner_slug, subtype_slug)
        ordinals[ordinal_key] += 1
        cluster_id = f"{owner_slug}_{subtype_slug}_{ordinals[ordinal_key]:03d}"
        issue_filename = f"issue_{index:03d}_{cluster_id}.md"
        clusters.append(
            IssueCluster(
                cluster_id=cluster_id,
                issue_filename=issue_filename,
                owner=state.owner,
                issue_kind=state.issue_kind,
                issue_subtype=state.issue_subtype,
                symptom_family=state.symptom_family,
                symptom=state.symptom,
                acceptance_signal=_issue_acceptance_signal(
                    state.issue_subtype,
                    state.issue_kind,
                ),
                source=state.source,
                sort_rank=state.sort_rank,
                results=tuple(sorted(state.results, key=lambda result: result.model_name)),
                stack_signals=tuple(sorted(state.stack_signals, key=lambda row: row[0].model_name)),
            )
        )

    return tuple(clusters)


def _issue_cluster_map_for_results(
    results: Sequence[PerformanceResult],
    *,
    prompt: str | None,
) -> dict[str, IssueCluster]:
    """Return model-name to issue-cluster mapping for JSONL/repro metadata."""
    snapshot = _build_diagnostics_snapshot(
        results=list(results),
        prompt=prompt,
    )
    cluster_by_model: dict[str, IssueCluster] = {}
    for cluster in _build_issue_clusters(snapshot):
        for result in _issue_cluster_results(cluster):
            cluster_by_model.setdefault(result.model_name, cluster)
    return cluster_by_model


def _maintainer_owner_counts(snapshot: DiagnosticsSnapshot) -> list[tuple[str, int]]:
    """Summarize likely owner buckets for the diagnostics snapshot."""
    owner_counts: Counter[str] = Counter()

    for _signature, cluster_results in snapshot.failure_clusters:
        representative = cluster_results[0]
        owner = _diagnostics_owner_label(representative.error_package or "unknown")
        owner_counts[owner] += len(cluster_results)

    for res, _text in snapshot.harness_results:
        owner_counts[_diagnostics_owner_label(_infer_harness_issue_owner(res))] += 1

    for _res, _symptom, package_hint in snapshot.stack_signals:
        owner_counts[_diagnostics_owner_label(package_hint)] += 1

    for issue in snapshot.preflight_issues:
        owner_key = _guess_preflight_issue_package(issue)
        owner_counts[_diagnostics_owner_label(owner_key)] += 1

    return sorted(owner_counts.items(), key=lambda item: (-item[1], item[0]))


def _log_maintainer_summary(
    *,
    artifacts: DiagnosticsArtifacts,
    diagnostics_path: Path,
) -> None:
    """Emit a concise maintainer-facing summary sourced from diagnostics data."""
    snapshot = artifacts.snapshot
    log_blank()
    print_cli_section("Maintainer Summary")

    if not _snapshot_has_maintainer_signals(snapshot):
        logger.info("No maintainer-facing diagnostics signals detected.")
        return

    logger.info(
        "Diagnostics signals: failures=%d, harness=%d, stack=%d, preflight=%d",
        len(snapshot.failed),
        len(snapshot.harness_results),
        len(snapshot.stack_signals),
        len(snapshot.preflight_issues),
    )

    owner_counts = _maintainer_owner_counts(snapshot)
    if owner_counts:
        logger.info(
            "Likely owners: %s",
            "; ".join(f"{owner}={count}" for owner, count in owner_counts[:3]),
        )

    for index, (_signature, cluster_results) in enumerate(snapshot.failure_clusters[:3], start=1):
        representative = cluster_results[0]
        owner_key = representative.error_package or "unknown"
        owner = _diagnostics_owner_label(owner_key)
        issue = _truncate_text_preview(
            _simplify_failure_message(
                representative.error_message,
                model_name=representative.model_name,
            ),
            max_chars=96,
        )
        logger.info(
            "Issue cluster %d: %d model(s) | owner=%s | issue=%s",
            index,
            len(cluster_results),
            owner,
            issue,
        )

    for owner_key, owner_results in _group_harness_results_by_owner(snapshot.harness_results):
        owner = _diagnostics_owner_label(owner_key)
        logger.info(
            "Harness/runtime anomalies: %d model(s) likely owned by %s. Next: %s",
            len(owner_results),
            owner,
            _diagnostics_next_action(owner_key),
        )
    for owner_key, owner_signals in _group_stack_signals_by_owner(snapshot.stack_signals):
        owner = _diagnostics_owner_label(owner_key)
        logger.info(
            "Long-context or stack-signal anomalies: %d model(s) likely owned by %s. Next: %s",
            len(owner_signals),
            owner,
            _diagnostics_next_action(owner_key),
        )
    for owner_key, owner_issues in _group_preflight_issues_by_owner(snapshot.preflight_issues):
        owner = _diagnostics_owner_label(owner_key)
        logger.info(
            "Preflight compatibility warnings: %d issue(s) likely owned by %s. Next: %s",
            len(owner_issues),
            owner,
            _diagnostics_next_action(owner_key),
        )

    if artifacts.diagnostics_written:
        log_file_path(diagnostics_path, label="   Diagnostics:  ")
    if artifacts.repro_bundles:
        logger.info(
            "Repro bundles available for %d issue-linked model(s).",
            len(artifacts.repro_bundles),
        )


def _format_recent_repro_ratio(history_info: FailureHistoryContext | None) -> str:
    """Format reproducibility ratio string such as ``2/3 recent runs failed``."""
    if not history_info:
        return "n/a"
    recent_failures = history_info.recent_failures
    recent_considered = history_info.recent_considered
    if recent_considered <= 0:
        return "n/a"
    return f"{recent_failures}/{recent_considered} recent runs failed"


def _sanitize_capture_for_diagnostics(captured_output: str | None) -> str | None:
    """Strip low-signal terminal noise while preserving actionable stderr/stdout."""
    if not captured_output:
        return None

    sanitized = ANSI_ESCAPE_RE.sub("", captured_output)
    sanitized = sanitized.replace("\r\n", "\n").replace("\r", "\n").replace("\x08", "")
    sanitized = "".join(ch for ch in sanitized if ch in {"\n", "\t"} or ch.isprintable())

    kept_lines: list[str] = []
    for raw_line in sanitized.splitlines():
        stripped = raw_line.strip()
        if stripped and any(
            pattern.search(stripped) for pattern in _DIAGNOSTICS_CAPTURE_NOISE_PATTERNS
        ):
            continue
        kept_lines.append(raw_line.rstrip())

    compact = "\n".join(kept_lines)
    compact = re.sub(r"\n{3,}", "\n\n", compact).strip()
    return compact or None


def _diagnostics_detailed_trace_logs_section(
    cluster_results: list[PerformanceResult],
) -> list[str]:
    """Build one collapsed block containing traceback/log details per model."""
    model_logs: list[tuple[str, str | None, str | None]] = []
    for result in cluster_results:
        traceback_text = _format_traceback_tail(result.error_traceback)
        captured_text = _sanitize_capture_for_diagnostics(result.captured_output_on_fail)
        if traceback_text is None and captured_text is None:
            continue
        model_logs.append((result.model_name, traceback_text, captured_text))

    if not model_logs:
        return []

    body_lines: list[str] = []
    for model_name, traceback_text, captured_text in model_logs:
        body_lines.append(f"#### `{DIAGNOSTICS_ESCAPER.escape(model_name)}`")
        body_lines.append("")
        if traceback_text is not None:
            body_lines.append("Traceback tail:")
            _append_markdown_code_block(body_lines, traceback_text, language="text")
        if captured_text is not None:
            body_lines.append("Captured stdout/stderr:")
            _append_markdown_code_block(body_lines, captured_text, language="text")

    scope = "affected model" if len(model_logs) == 1 else "affected models"
    parts: list[str] = []
    _append_markdown_details_block(
        parts,
        summary=f"Detailed trace logs ({scope})",
        body_lines=body_lines,
    )
    return parts


def _cluster_failures_by_pattern(
    results: list[PerformanceResult],
) -> dict[str, list[PerformanceResult]]:
    """Group failed results by canonical signature.

    Unlike ``_bucket_failures_by_error`` (which produces ``{signature: [model_name]}``
    for console logging), this returns the full ``PerformanceResult`` objects so the
    diagnostics report can access tracebacks, packages, and other fields.

    Clustering heuristic:
        1. Prefer explicit ``error_signature`` from runtime failure metadata.
        2. Fall back to deterministic signature built from error code + normalized
           message/traceback.
        3. After initial clustering, merge clusters that share the same
           message-only signature (error code + normalized message, ignoring
           traceback differences).  This collapses e.g. ``broadcast_shapes``
           failures that differ only in stack trace details.

    Returns:
        Mapping from canonical signature to the list of results sharing it.
    """
    clusters: dict[str, list[PerformanceResult]] = {}
    sig_to_msg_key: dict[str, str] = {}
    failed = [r for r in results if not r.success]

    for res in failed:
        stage = res.error_stage or _classify_error(res.error_message or "Unknown error")
        package = res.error_package or _attribute_error_to_package(
            res.error_message or "Unknown error",
            res.error_traceback,
        )
        code = res.error_code or _build_canonical_error_code(
            error_stage=stage,
            error_package=package,
            failure_phase=res.failure_phase,
        )
        signature = res.error_signature or _build_error_signature(
            error_code=code,
            error_message=res.error_message,
            error_traceback=res.error_traceback,
        )

        # Track a message-only key for the merge pass
        msg_key = f"{code}|{_normalize_error_core_message(res.error_message or 'Unknown error')}"
        sig_to_msg_key[signature] = msg_key
        clusters.setdefault(signature, []).append(res)

    # Merge pass: collapse clusters that share the same message-only key
    msg_key_to_canonical: dict[str, str] = {}
    merged: dict[str, list[PerformanceResult]] = {}
    for sig, results_list in clusters.items():
        msg_key = sig_to_msg_key[sig]
        if msg_key not in msg_key_to_canonical:
            msg_key_to_canonical[msg_key] = sig
            merged[sig] = results_list.copy()
        else:
            canonical_sig = msg_key_to_canonical[msg_key]
            merged[canonical_sig].extend(results_list)

    return merged


def _collect_harness_results(
    successful: list[PerformanceResult],
) -> list[tuple[PerformanceResult, str]]:
    """Collect successful models that have harness/integration issues."""
    harness_results: list[tuple[PerformanceResult, str]] = []
    for res in successful:
        qa = res.quality_analysis
        if qa is None and res.generation is not None:
            # Backward compatibility for callers that attach analysis to generation.
            qa = getattr(res.generation, "quality_analysis", None)
        if qa and getattr(qa, "has_harness_issue", False):
            text = getattr(res.generation, "text", "") if res.generation else ""
            harness_results.append((res, text))
    return harness_results


def _collect_stack_issue_signals(
    successful: list[PerformanceResult],
    *,
    preflight_issues: Sequence[str] = (),
) -> list[tuple[PerformanceResult, str, str]]:
    """Collect likely upstream stack issues from successful-but-suspicious runs."""
    signals: list[tuple[PerformanceResult, str, str]] = []

    for res in successful:
        if not res.generation:
            continue

        gen = res.generation
        qa = res.quality_analysis or getattr(gen, "quality_analysis", None)
        text = str(getattr(gen, "text", "") or "")
        prompt_tokens = int(getattr(gen, "prompt_tokens", 0) or 0)
        generated_tokens = int(getattr(gen, "generation_tokens", 0) or 0)
        ratio = (generated_tokens / prompt_tokens) if prompt_tokens > 0 else 0.0

        if not text.strip() or generated_tokens == 0:
            symptom = "Empty output despite successful run"
            signals.append(
                (
                    res,
                    symptom,
                    _infer_stack_signal_owner(symptom, preflight_issues=preflight_issues),
                ),
            )
            continue

        if (
            prompt_tokens >= QUALITY.long_prompt_tokens_threshold
            and generated_tokens < QUALITY.min_output_tokens_for_ratio
            and ratio < QUALITY.min_output_ratio
        ):
            symptom = f"Long-context low output ratio ({ratio:.1%})"
            signals.append(
                (
                    res,
                    symptom,
                    _infer_stack_signal_owner(symptom, preflight_issues=preflight_issues),
                ),
            )
            continue

        if (
            prompt_tokens >= QUALITY.severe_prompt_tokens_threshold
            and qa is not None
            and (
                qa.is_repetitive
                or qa.is_context_ignored
                or qa.has_context_echo
                or qa.has_degeneration
            )
        ):
            if qa.is_repetitive:
                symptom = "Repetition under long prompt length"
            elif qa.is_context_ignored:
                symptom = "Context dropped under long prompt length"
            elif qa.has_context_echo:
                symptom = "Context echo under long prompt length"
            else:
                degeneration_type = qa.degeneration_type or "degradation"
                symptom = f"Output degeneration under long prompt length ({degeneration_type})"
            signals.append(
                (
                    res,
                    symptom,
                    _infer_stack_signal_owner(symptom, preflight_issues=preflight_issues),
                ),
            )

    return signals


def _diagnostics_header(
    *,
    total: int,
    n_failed: int,
    n_harness: int,
    n_preflight: int,
    n_success: int,
    versions: LibraryVersionDict,
    image_path: Path | None,
) -> list[str]:
    """Build the compact title block for the diagnostics report."""
    parts: list[str] = []
    version = versions.get("mlx-vlm") or "unknown"
    parts.append(
        f"# Diagnostics Report — {n_failed} failure(s), "
        f"{n_harness} harness issue(s) (mlx-vlm {version})",
    )
    parts.append("")
    parts.append(
        f"**Run summary:** {total} locally-cached VLM model(s) checked; "
        f"{n_failed} hard failure(s), {n_harness} harness/integration issue(s), "
        f"{n_preflight} preflight warning(s), {n_success} successful run(s)."
    )
    parts.append("")

    if image_path and image_path.exists():
        try:
            size_mb = image_path.stat().st_size / (1024 * 1024)
            parts.append(
                f"Test image: `{image_path.name}` ({size_mb:.1f} MB).",
            )
            parts.append("")
        except OSError:
            pass

    return parts


def _diagnostics_environment_section(
    *,
    versions: LibraryVersionDict,
    system_info: dict[str, str],
) -> list[str]:
    """Build environment details section for diagnostics report footer context."""
    parts: list[str] = _begin_diagnostics_section(title="## Environment")
    table_rows = [
        (component, DIAGNOSTICS_ESCAPER.escape(value))
        for component, value in _collect_report_component_rows(
            versions=versions,
            system_info=system_info,
            library_names=_DIAGNOSTICS_LIB_NAMES,
            system_keys=_DIAGNOSTICS_SYSTEM_KEYS,
        )
    ]
    parts.extend(
        tabulate(
            table_rows,
            headers=["Component", "Version"],
            tablefmt="github",
        ).splitlines()
    )
    parts.append("")
    return parts


def _build_environment_failure_diagnostics(
    *,
    error_message: str,
    versions: LibraryVersionDict,
    system_info: dict[str, str],
) -> list[str]:
    """Build diagnostics content for environment preflight failures (no models run)."""
    parts: list[str] = [
        "# Diagnostics Report — environment preflight failure",
        "",
    ]
    _append_markdown_section(
        parts,
        title="## Summary",
        body_lines=[
            (
                "Run aborted before any model execution because core runtime "
                "dependencies were unavailable."
            ),
            f"**Error:** {DIAGNOSTICS_ESCAPER.escape(error_message)}",
        ],
    )
    parts.extend(
        _diagnostics_environment_section(
            versions=versions,
            system_info=system_info,
        ),
    )
    return parts


def _issue_target_for_package(
    package: str,
    *,
    model_name: str,
) -> tuple[str, str]:
    """Map failure package attribution to a likely upstream issue tracker."""
    target_map: dict[str, tuple[str, str]] = {
        "mlx-vlm": ("mlx-vlm", "https://github.com/Blaizzy/mlx-vlm/issues/new"),
        "mlx": ("mlx", "https://github.com/ml-explore/mlx/issues/new"),
        "mlx-lm": ("mlx-lm", "https://github.com/ml-explore/mlx-lm/issues/new"),
        ("transformers"): (
            "transformers",
            "https://github.com/huggingface/transformers/issues/new",
        ),
        ("huggingface-hub"): (
            "huggingface_hub",
            "https://github.com/huggingface/huggingface_hub/issues/new",
        ),
    }
    if package in target_map:
        return target_map[package]
    if package == "model-config" and "/" in model_name:
        return (
            "model repository",
            f"https://huggingface.co/{quote(model_name, safe='/')}",
        )
    return ("mlx-vlm", "https://github.com/Blaizzy/mlx-vlm/issues/new")


def _guess_preflight_issue_package(issue: str) -> str:
    """Infer likely owner package from a preflight warning string."""
    normalized = issue.lower()
    if "mlx-vlm" in normalized:
        return "mlx-vlm"
    if "mlx-lm" in normalized:
        return "mlx-lm"
    if "transformers" in normalized:
        return "transformers"
    if "huggingface" in normalized:
        return "huggingface-hub"
    if "mlx" in normalized:
        return "mlx"
    return "unknown"


def _infer_stack_signal_owner(
    symptom: str,
    *,
    preflight_issues: Sequence[str] = (),
) -> str:
    """Infer the most likely owner for a successful-run stack anomaly."""
    preflight_packages = {
        _guess_preflight_issue_package(issue)
        for issue in preflight_issues
        if _guess_preflight_issue_package(issue) != "unknown"
    }
    symptom_lower = symptom.casefold()

    owner = "mlx-vlm / mlx"

    if "empty output" in symptom_lower:
        owner = "mlx-vlm"

    package_priority = (
        ("huggingface-hub", "huggingface-hub"),
        ("transformers", "transformers"),
        ("mlx-lm", "mlx-lm"),
        ("mlx", "mlx"),
        ("mlx-vlm", "mlx-vlm"),
    )
    for package_name, package_owner in package_priority:
        if package_name in preflight_packages:
            owner = package_owner

    if "mlx-vlm" in preflight_packages and "mlx" in preflight_packages:
        owner = "mlx-vlm / mlx"
    if "transformers" in preflight_packages and any(
        marker in symptom_lower
        for marker in ("context", "prompt", "echo", "degeneration", "long-context")
    ):
        owner = "transformers / mlx-vlm"

    return owner


def _group_stack_signals_by_owner(
    stack_signals: Sequence[tuple[PerformanceResult, str, str]],
) -> list[tuple[str, list[tuple[PerformanceResult, str, str]]]]:
    """Group stack-signal rows by inferred owner for maintainer triage."""
    grouped: dict[str, list[tuple[PerformanceResult, str, str]]] = {}
    for result in stack_signals:
        owner = result[2]
        grouped.setdefault(owner, []).append(result)
    return sorted(grouped.items(), key=lambda item: (item[0], len(item[1])))


def _group_preflight_issues_by_owner(
    preflight_issues: Sequence[str],
) -> list[tuple[str, list[str]]]:
    """Group preflight warnings by inferred owner for maintainer triage."""
    grouped: dict[str, list[str]] = {}
    for issue in preflight_issues:
        owner = _guess_preflight_issue_package(issue)
        grouped.setdefault(owner, []).append(issue)
    return sorted(grouped.items(), key=lambda item: (item[0], len(item[1])))


def _diagnostics_preflight_section(preflight_issues: Sequence[str]) -> list[str]:
    """Build diagnostics section for compatibility warnings seen during preflight."""
    if not preflight_issues:
        return []

    parts = _begin_diagnostics_section(
        title=f"## Preflight Compatibility Warnings ({len(preflight_issues)} issue(s))",
        body_lines=[
            "These warnings were detected before inference. They are informational by "
            "default and do not invalidate successful runs on their own.",
            "Keep running if outputs look healthy. Escalate only when the warnings line "
            "up with API mismatches, startup hangs, or runtime crashes.",
            "Do not treat these warnings alone as a reason to assume the benchmark "
            "results are bad.",
        ],
    )

    for package, owner_issues in _group_preflight_issues_by_owner(preflight_issues):
        target_name, target_url = _issue_target_for_package(
            package,
            model_name="unknown/model",
        )
        parts.append(f"### `{DIAGNOSTICS_ESCAPER.escape(_diagnostics_owner_label(package))}`")
        parts.append("")
        _append_markdown_row_block(
            parts,
            rows=_build_preflight_overview_rows(
                package=package,
                owner_issues=owner_issues,
                target_name=target_name,
                target_url=target_url,
                escaper=DIAGNOSTICS_ESCAPER,
            ),
        )
        parts.append("Warnings:")
        parts.append("")
        for issue in owner_issues:
            escaped_issue = DIAGNOSTICS_ESCAPER.escape(issue)
            parts.append(f"- `{escaped_issue}`")
        parts.append("")
    parts.append("")
    return parts


_DIAGNOSTICS_COMPONENT_LABELS: Final[dict[str, str]] = {
    "mlx-vlm": "mlx-vlm",
    "mlx": "mlx",
    "mlx-lm": "mlx-lm",
    "transformers": "transformers",
    "huggingface-hub": "huggingface_hub",
    "model-config": "model configuration/repository",
    "unknown": "unknown component",
}

_DIAGNOSTICS_OWNER_ACTIONS: Final[dict[str, str]] = {
    "mlx-vlm": "check processor/chat-template wiring and generation kwargs.",
    "mlx": "check tensor/cache behavior and memory pressure handling.",
    "mlx-lm": "verify tokenizer/runtime compatibility and quantization settings.",
    "transformers": "verify API compatibility and pinned version floor.",
    "huggingface-hub": (
        "check cache/revision availability and network/auth state; Hub disconnects may be "
        "transient outages rather than model defects."
    ),
    "model-config": "verify model config, tokenizer files, and revision alignment.",
    "unknown": "capture traceback + env fingerprint, then triage manually.",
}

_HARNESS_TYPE_DESCRIPTIONS: Final[dict[str, str]] = {
    "encoding": (
        "Decoded output contains tokenizer artifacts that should not appear in user-facing text."
    ),
    "stop_token": (
        "Generation appears to continue through stop/control tokens instead of ending cleanly."
    ),
    "prompt_template": "Output shape suggests a prompt-template or stop-condition mismatch.",
    "long_context": "Behavior degrades under long prompt context.",
    "generation_loop": "Output appears to drift into instruction/training-template text.",
}

_TRAINING_LEAK_LABELS: Final[dict[str, str]] = {
    "instruction_header": "instruction headers mid-output",
    "task_header": "task/question headers mid-output",
    "write_prompt": "new writing prompts mid-output",
    "user_turn": "new user-turn delimiters mid-output",
    "code_example": "example-code templates mid-output",
    "qa_pair": "Q/A template patterns mid-output",
}

_REVIEW_OWNER_BY_FAILURE_NEEDLE: Final[tuple[tuple[str, str], ...]] = (
    ("transformers", "transformers"),
    ("huggingface-hub", "huggingface-hub"),
    ("mlx-lm", "mlx-lm"),
    ("mlx-vlm", "mlx-vlm"),
    ("model-config", "model-config"),
    ("mlx", "mlx"),
)

_REVIEW_OWNER_BY_HARNESS_TYPE: Final[dict[str, str]] = {
    "prompt_template": "model-config",
    "stop_token": "mlx-vlm",
    "encoding": "mlx-vlm",
    "generation_loop": "mlx-vlm",
    "long_context": "mlx",
}

_DIAGNOSTICS_OWNER_PART_ACTIONS: Final[tuple[tuple[frozenset[str], str], ...]] = (
    (
        frozenset({"model-config", "mlx-vlm"}),
        "validate chat-template/config expectations and mlx-vlm prompt formatting for this model.",
    ),
    (frozenset({"model-config"}), _DIAGNOSTICS_OWNER_ACTIONS["model-config"]),
    (frozenset({"transformers"}), _DIAGNOSTICS_OWNER_ACTIONS["transformers"]),
    (
        frozenset({"mlx-vlm", "mlx"}),
        "validate long-context handling and stop-token behavior across mlx-vlm + mlx runtime.",
    ),
    (
        frozenset({"mlx-vlm", "mlx-lm"}),
        "validate generation-loop handoff and template continuation behavior across mlx-vlm + mlx-lm.",
    ),
    (frozenset({"huggingface-hub"}), _DIAGNOSTICS_OWNER_ACTIONS["huggingface-hub"]),
    (frozenset({"mlx-vlm"}), _DIAGNOSTICS_OWNER_ACTIONS["mlx-vlm"]),
    (frozenset({"mlx-lm"}), _DIAGNOSTICS_OWNER_ACTIONS["mlx-lm"]),
    (frozenset({"mlx"}), _DIAGNOSTICS_OWNER_ACTIONS["mlx"]),
)

_HARNESS_OWNER_BY_TYPE: Final[dict[str, str]] = {
    "prompt_template": "model-config / mlx-vlm",
    "long_context": "mlx-vlm / mlx",
}


def _diagnostics_owner_label(owner_key: str) -> str:
    """Return readable owner label for diagnostics rows/sections."""
    return _DIAGNOSTICS_COMPONENT_LABELS.get(owner_key, owner_key)


def _has_prefixed_detail(details: Sequence[str], prefix: str) -> bool:
    """Return True when any detail entry starts with the requested prefix."""
    return any(detail.startswith(prefix) for detail in details)


def _diagnostics_next_action(owner_key: str) -> str:
    """Return a short owner-specific next action hint."""
    if action := _DIAGNOSTICS_OWNER_ACTIONS.get(owner_key):
        return action

    owner_parts = {part.strip() for part in owner_key.split("/") if part.strip()}
    for required_parts, action in _DIAGNOSTICS_OWNER_PART_ACTIONS:
        if required_parts.issubset(owner_parts):
            return action
    return _DIAGNOSTICS_OWNER_ACTIONS["unknown"]


def _failure_impact_text(result: PerformanceResult) -> str:
    """Return a short explanation of why a hard failure matters."""
    details: list[str] = []
    if result.error_stage:
        details.append(f"stage `{result.error_stage}`")
    if result.failure_phase:
        details.append(f"phase `{result.failure_phase}`")
    if result.error_code:
        details.append(f"code `{result.error_code}`")
    if result.error_type:
        details.append(f"type `{result.error_type}`")
    if details:
        return "This prevented a complete model response; " + "; ".join(details) + "."
    return "This prevented a complete model response and should be triaged before judging model quality."


def _build_failure_overview_rows(
    result: PerformanceResult,
    *,
    escaper: DiagnosticsEscaper | MarkdownPipeEscaper,
    affected_models: Sequence[str] = (),
) -> list[tuple[str, str]]:
    """Build the shared failure overview rows for diagnostics and issue output."""
    owner_key = result.error_package or "unknown"
    rows: list[tuple[str, str]] = [
        (
            "Observed",
            escaper.escape(
                _simplify_failure_message(
                    result.error_message,
                    model_name=result.model_name,
                ),
            ),
        ),
        ("Likely owner", f"`{escaper.escape(_diagnostics_owner_label(owner_key))}`"),
        ("Why it matters", escaper.escape(_failure_impact_text(result))),
        ("Suggested next step", escaper.escape(_diagnostics_next_action(owner_key))),
    ]
    if affected_models:
        rows.append(
            (
                "Affected models",
                ", ".join(f"`{escaper.escape(model_name)}`" for model_name in affected_models),
            ),
        )
    return rows


def _build_harness_overview_rows(
    *,
    harness_summary: str,
    likely_owner: str,
    prompt_tokens: int,
    generated_tokens: int,
    ratio_text: str,
    escaper: DiagnosticsEscaper | MarkdownPipeEscaper,
) -> list[tuple[str, str]]:
    """Build the shared harness-at-a-glance rows for diagnostics and issue output."""
    token_summary = (
        f"prompt={fmt_num(prompt_tokens)}, output={fmt_num(generated_tokens)}, "
        f"output/prompt={ratio_text}"
    )
    return [
        ("Observed", escaper.escape(harness_summary)),
        ("Likely owner", f"`{escaper.escape(likely_owner)}`"),
        (
            "Why it matters",
            escaper.escape(
                "The run completed, but the output pattern points to stack/runtime behavior "
                "rather than a clean model-quality limitation."
            ),
        ),
        ("Suggested next step", escaper.escape(_diagnostics_next_action(likely_owner))),
        ("Token summary", escaper.escape(token_summary)),
    ]


def _build_preflight_overview_rows(
    *,
    package: str,
    owner_issues: Sequence[str],
    target_name: str,
    target_url: str,
    escaper: DiagnosticsEscaper | MarkdownPipeEscaper,
) -> list[tuple[str, str]]:
    """Build shared overview rows for grouped preflight compatibility warnings."""
    preview = _truncate_text_preview(owner_issues[0], max_chars=140) if owner_issues else "unknown"
    rows: list[tuple[str, str]] = [
        ("Observed", escaper.escape(preview)),
        ("Likely owner", f"`{escaper.escape(_diagnostics_owner_label(package))}`"),
        (
            "Why it matters",
            escaper.escape(
                "These warnings are informational until they line up with API mismatches, "
                "startup hangs, or runtime crashes in the same run."
            ),
        ),
        ("Suggested tracker", f"`{escaper.escape(target_name)}` (<{target_url}>)"),
        ("Suggested next step", escaper.escape(_diagnostics_next_action(package))),
    ]
    if len(owner_issues) > 1:
        rows.append(
            ("Additional warnings", escaper.escape(f"{len(owner_issues) - 1} more listed below"))
        )
    return rows


def _infer_harness_issue_owner(result: PerformanceResult) -> str:
    """Infer the most likely owner for a successful-run harness issue."""
    gen = result.generation
    qa = result.quality_analysis or (getattr(gen, "quality_analysis", None) if gen else None)
    if qa is None:
        return "mlx-vlm"

    harness_type = (qa.harness_issue_type or "").strip().lower()
    harness_details = tuple((detail or "").strip().lower() for detail in qa.harness_issue_details)

    if inferred_owner := _HARNESS_OWNER_BY_TYPE.get(harness_type):
        return inferred_owner
    if harness_type == "generation_loop" and _has_prefixed_detail(
        harness_details,
        "training_leak:",
    ):
        return "mlx-vlm / mlx-lm"
    if _has_prefixed_detail(harness_details, "output:"):
        return "model-config / mlx-vlm"
    return "mlx-vlm"


def _group_harness_results_by_owner(
    harness_results: Sequence[tuple[PerformanceResult, str]],
) -> list[tuple[str, list[tuple[PerformanceResult, str]]]]:
    """Group harness-result rows by inferred owner for maintainer triage."""
    grouped: dict[str, list[tuple[PerformanceResult, str]]] = {}
    for result in harness_results:
        owner = _infer_harness_issue_owner(result[0])
        grouped.setdefault(owner, []).append(result)
    return sorted(grouped.items(), key=lambda item: (item[0], len(item[1])))


def _dedupe_preserve_order(items: Sequence[str]) -> list[str]:
    """Return unique items while preserving first-seen order."""
    unique_items: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique_items.append(item)
    return unique_items


def _simplify_failure_message(error_message: str | None, *, model_name: str) -> str:
    """Strip local wrapper prefixes to keep issue text maintainer-focused."""
    if not error_message:
        return "Unknown runtime failure."

    message = error_message.split("\n")[0].strip()
    prefixes = (
        f"Model generation failed for {model_name}: ",
        f"Model preflight failed for {model_name}: ",
    )
    for prefix in prefixes:
        if message.startswith(prefix):
            return message.removeprefix(prefix).strip() or message
    return message


def _describe_token_encoding_detail(token_issue: str) -> str:
    """Describe token-encoding harness anomalies in plain language."""
    if match := re.fullmatch(r"bpe_space_leak\((\d+)\)", token_issue):
        count = match.group(1)
        return (
            "Tokenizer space-marker artifacts (for example Ġ) appeared in output "
            f"(about {count} occurrences)."
        )
    if match := re.fullmatch(r"bpe_newline_leak\((\d+)\)", token_issue):
        count = match.group(1)
        return (
            "Tokenizer newline-marker artifacts (for example Ċ) appeared in output "
            f"(about {count} occurrences)."
        )
    if token_issue.startswith("bpe_byte_leak("):
        return "Byte-level tokenizer artifacts appeared in generated text."
    return "Tokenizer decoding artifacts appeared in generated text."


def _describe_output_detail(output_detail: str) -> str | None:
    """Describe output-length harness anomalies in plain language."""
    if output_detail == "zero_tokens":
        return "Model returned zero output tokens."
    if match := re.fullmatch(r"truncated\((\d+)tok\)", output_detail):
        return f"Output appears truncated to about {match.group(1)} tokens."
    if match := re.fullmatch(r"filler_response\((\d+)tok\)", output_detail):
        return f"Output was a short generic filler response (about {match.group(1)} tokens)."
    if match := re.fullmatch(r"output_ratio\(([^)]+)\)", output_detail):
        ratio_text = match.group(1)
        return (
            "Output is very short relative to prompt size "
            f"({ratio_text}), suggesting possible early-stop or prompt-handling issues."
        )
    return None


def _describe_long_context_detail(detail: str) -> str | None:
    """Describe long-context harness anomalies in plain language."""
    if match := re.fullmatch(r"long_context_empty\((\d+)tok\)", detail):
        return f"At long prompt length ({match.group(1)} tokens), generation returned empty output."
    if match := re.fullmatch(r"long_context_low_ratio\(([^;]+);(\d+)->(\d+)\)", detail):
        ratio_text, prompt_tok, output_tok = match.groups()
        return (
            "At long prompt length "
            f"({prompt_tok} tokens), output stayed unusually short "
            f"({output_tok} tokens; ratio {ratio_text})."
        )
    if match := re.fullmatch(r"long_context_repetition\((\d+)tok\)", detail):
        return f"At long prompt length ({match.group(1)} tokens), output became repetitive."
    if match := re.fullmatch(r"long_context_context_drop\((\d+)tok\)", detail):
        return (
            "At long prompt length "
            f"({match.group(1)} tokens), output may stop following prompt/image context."
        )
    return None


def _describe_harness_detail(detail: str) -> str | None:
    """Translate internal harness detail tokens into maintainer-friendly prose."""
    description: str | None = None
    if detail.startswith("token_leak:"):
        token = detail.removeprefix("token_leak:")
        description = (
            f"Special control token {html.escape(token, quote=False)} appeared in generated text."
        )
    elif detail.startswith("token_encoding:"):
        description = _describe_token_encoding_detail(detail.removeprefix("token_encoding:"))
    elif detail.startswith("output:"):
        description = _describe_output_detail(detail.removeprefix("output:"))
    elif detail.startswith("long_context_"):
        description = _describe_long_context_detail(detail)
    elif detail.startswith("training_leak:"):
        leak_type = detail.removeprefix("training_leak:")
        leak_label = _TRAINING_LEAK_LABELS.get(leak_type, "instruction/template text")
        description = f"Generated text appears to continue into {leak_label}."
    return description


def _summarize_quality_signals(qa: GenerationQualityAnalysis | None) -> list[str]:
    """Convert additional quality flags into self-explanatory prose."""
    if qa is None:
        return []

    signal_builders = (
        _summarize_context_ignored_signal,
        _summarize_repetition_signal,
        _summarize_degeneration_signal,
        _summarize_language_signal,
        _summarize_refusal_signal,
        _summarize_formatting_signal,
        _summarize_missing_sections_signal,
        _summarize_reasoning_leak_signal,
        _summarize_context_echo_signal,
    )
    messages = [message for builder in signal_builders if (message := builder(qa)) is not None]
    return _dedupe_preserve_order(messages)


def _build_cluster_filing_guidance(
    *,
    representative: PerformanceResult,
) -> list[str]:
    """Build concise filing guidance without repeating the full repro command."""
    return [
        "- Exact model-specific repro command appears below in the "
        "`Reproducibility` section under `Target specific failing models`.",
        f"- Representative failing model: `{DIAGNOSTICS_ESCAPER.escape(representative.model_name)}`",
    ]


def _diagnostics_failure_clusters(
    failure_clusters: list[tuple[str, list[PerformanceResult]]],
    *,
    diagnostics_context: DiagnosticsContext,
) -> list[str]:
    """Build grouped failure sections of the diagnostics report."""
    if not failure_clusters:
        return []

    parts = _begin_diagnostics_section()
    for idx, (_cluster_signature, cluster_results) in enumerate(failure_clusters, 1):
        rep = cluster_results[0]
        n = len(cluster_results)
        model_word = "model" if n == 1 else "models"

        parts.append(
            f"## {idx}. Failure affecting {n} {model_word}",
        )
        parts.append("")
        _append_markdown_row_block(
            parts,
            rows=_build_failure_overview_rows(
                rep,
                escaper=DIAGNOSTICS_ESCAPER,
                affected_models=[r.model_name for r in cluster_results],
            ),
        )
        _append_maintainer_triage_markdown(
            parts,
            result=rep,
            escaper=DIAGNOSTICS_ESCAPER,
            heading="**Representative maintainer triage:**",
        )

        # Maintainer-facing table (human-readable, no local diagnostic codes).
        table_rows: list[tuple[str, ...]] = []
        for r in cluster_results:
            model = DIAGNOSTICS_ESCAPER.escape(r.model_name)
            short_error = DIAGNOSTICS_ESCAPER.escape(
                _simplify_failure_message(r.error_message, model_name=r.model_name),
            )
            history_info = diagnostics_context.failure_history.get(r.model_name)
            first_seen = DIAGNOSTICS_ESCAPER.escape(
                history_info.first_failure_timestamp if history_info else "unknown",
            )
            recent_repro = DIAGNOSTICS_ESCAPER.escape(_format_recent_repro_ratio(history_info))
            table_rows.append((f"`{model}`", short_error, first_seen, recent_repro))
        parts.extend(
            tabulate(
                table_rows,
                headers=["Model", "Observed Behavior", "First Seen Failing", "Recent Repro"],
                tablefmt="github",
            ).splitlines()
        )
        parts.append("")

        _append_markdown_section(parts, title="### To reproduce")
        filing_guidance = _build_cluster_filing_guidance(
            representative=rep,
        )
        parts.extend(filing_guidance)
        parts.append("")

        output_entry: tuple[str, str] | None = None
        for result in cluster_results:
            generated_text = (
                str(getattr(result.generation, "text", "") or "").strip()
                if result.generation is not None
                else ""
            )
            if generated_text:
                output_entry = (result.model_name, generated_text)
                break

        if output_entry is not None:
            output_model, output_text = output_entry
            parts.append(
                f"**Observed model output (`{DIAGNOSTICS_ESCAPER.escape(output_model)}`):**",
            )
            _append_markdown_code_block(
                parts,
                _truncate_text_preview(
                    output_text,
                    max_chars=DIAGNOSTICS.output_snippet_len,
                ),
                language="text",
            )

        parts.extend(_diagnostics_detailed_trace_logs_section(cluster_results))

    return parts


def _diagnostics_harness_section(
    harness_results: list[tuple[PerformanceResult, str]],
) -> list[str]:
    """Build the harness/integration issues section of the diagnostics report."""
    if not harness_results:
        return []

    parts = _begin_diagnostics_section(
        title=f"## Harness/Integration Issues ({len(harness_results)} model(s))",
        body_lines=[
            f"{len(harness_results)} model(s) show potential harness/integration issues; "
            "see per-model breakdown below.",
            "These models completed successfully but show integration problems "
            "(for example stop-token leakage, decoding artifacts, or long-context "
            "breakdown) that likely point to stack/runtime behavior rather than "
            "inherent model quality limits.",
        ],
    )

    for res, text in harness_results:
        gen = res.generation
        qa = res.quality_analysis or (getattr(gen, "quality_analysis", None) if gen else None)
        harness_type = getattr(qa, "harness_issue_type", "unknown") if qa else "unknown"
        harness_details = getattr(qa, "harness_issue_details", []) if qa else []
        prompt_tokens = int(getattr(gen, "prompt_tokens", 0) or 0) if gen else 0
        generated_tokens = int(getattr(gen, "generation_tokens", 0) or 0) if gen else 0
        ratio_text = f"{(generated_tokens / prompt_tokens):.2%}" if prompt_tokens > 0 else "n/a"

        likely_package = _infer_harness_issue_owner(res)

        _append_markdown_section(parts, title=f"### `{res.model_name}`")
        harness_summary = _HARNESS_TYPE_DESCRIPTIONS.get(
            harness_type or "",
            "Output indicates a likely integration issue.",
        )
        _append_markdown_row_block(
            parts,
            rows=_build_harness_overview_rows(
                harness_summary=harness_summary,
                likely_owner=likely_package,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                ratio_text=ratio_text,
                escaper=DIAGNOSTICS_ESCAPER,
            ),
        )
        _append_maintainer_triage_markdown(
            parts,
            result=res,
            escaper=DIAGNOSTICS_ESCAPER,
        )
        observations = [
            desc for detail in harness_details if (desc := _describe_harness_detail(detail))
        ]
        observations.extend(_summarize_quality_signals(qa))
        unique_observations = _dedupe_preserve_order(observations)
        if unique_observations:
            parts.append("**Why this appears to be an integration/runtime issue:**")
            parts.append("")
            parts.extend(f"- {observation}" for observation in unique_observations)
            parts.append("")
        snippet_source = text.strip() or "<empty output>"
        snippet = snippet_source[: DIAGNOSTICS.output_snippet_len]
        if len(snippet_source) > DIAGNOSTICS.output_snippet_len:
            snippet += "..."
        parts.append("**Sample output:**")
        _append_markdown_code_block(parts, snippet, language="text")

    return parts


def _diagnostics_stack_signal_section(
    stack_signals: list[tuple[PerformanceResult, str, str]],
) -> list[str]:
    """Build a section for likely stack issues observed in successful runs (now sub-section)."""
    if not stack_signals:
        return []

    parts: list[str] = _begin_diagnostics_section(
        title=(
            f"### Long-Context Degradation / Potential Stack Issues ({len(stack_signals)} model(s))"
        ),
        body_lines=[
            f"{len(stack_signals)} model(s) show long-context degradation or stack anomalies; "
            "see table below.",
            "These models technically succeeded, but token/output patterns suggest likely "
            "integration/runtime issues worth checking upstream.",
        ],
    )

    rows: list[tuple[str, ...]] = []
    for res, symptom, package_hint in stack_signals:
        gen = res.generation
        prompt_tokens = int(getattr(gen, "prompt_tokens", 0) or 0) if gen else 0
        generated_tokens = int(getattr(gen, "generation_tokens", 0) or 0) if gen else 0
        ratio = f"{(generated_tokens / prompt_tokens):.2%}" if prompt_tokens > 0 else "n/a"
        rows.append(
            (
                f"`{DIAGNOSTICS_ESCAPER.escape(res.model_name)}`",
                fmt_num(prompt_tokens),
                fmt_num(generated_tokens),
                ratio,
                DIAGNOSTICS_ESCAPER.escape(symptom),
                f"`{DIAGNOSTICS_ESCAPER.escape(package_hint)}`",
            )
        )

    parts.extend(
        tabulate(
            rows,
            headers=["Model", "Prompt Tok", "Output Tok", "Output/Prompt", "Symptom", "Owner"],
            tablefmt="github",
        ).splitlines()
    )
    parts.append("")
    return parts


_ISSUE_QUEUE_HEADERS: Final[tuple[str, ...]] = (
    "Target",
    "Problem",
    "Affected Models",
    "Issue Draft",
    "Evidence Bundle",
    "Fixed When",
)

_ISSUE_SUBTYPE_DISPLAY_LABELS: Final[dict[str, str]] = {
    "context_budget": "Context-budget pressure",
    "empty_output": "Empty output",
    "encoding": "Tokenizer / decoding artifact",
    "generation_loop": "Generation-loop drift",
    "harness": "Harness / integration",
    "long_context": "Long-context collapse",
    "prompt_template": "Prompt-template / image-placeholder mismatch",
    "runtime_failure": "Runtime failure",
    "stack_signal": "Stack anomaly",
    "stop_token": "Stop-token leakage",
    "token_cap": "Token cap / cutoff",
}

_ERROR_CODE_PACKAGE_DISPLAY_LABELS: Final[dict[str, str]] = {
    "HUGGINGFACE_HUB": "Hugging Face Hub",
    "MLX": "MLX",
    "MLX_LM": "mlx-lm",
    "MLX_VLM": "mlx-vlm",
    "MODEL_CONFIG": "Model config",
    "TRANSFORMERS": "Transformers",
    "UNKNOWN": "Unknown owner",
}

_ERROR_CODE_STAGE_DISPLAY_LABELS: Final[dict[str, str]] = {
    "API_DRIFT": "API drift",
    "API_MISMATCH": "API mismatch",
    "CONFIG_MISSING": "missing config",
    "ERROR": "runtime error",
    "LIB_VERSION": "library-version mismatch",
    "MISSING_DEP": "missing dependency",
    "MODEL": "model error",
    "NO_CHAT_TEMPLATE": "missing chat template",
    "OOM": "out-of-memory",
    "PROCESSOR": "processor error",
    "TIMEOUT": "timeout",
    "TOKENIZER": "tokenizer error",
    "TYPE_CAST": "type-cast error",
    "WEIGHT_MISMATCH": "weight/config mismatch",
}

_ISSUE_SUBTYPE_TOKEN_LABELS: Final[dict[str, str]] = {
    "api": "API",
    "bpe": "BPE",
    "eos": "EOS",
    "hf": "HF",
    "id": "ID",
    "jsonl": "JSONL",
    "kv": "KV",
    "lm": "LM",
    "mlx": "MLX",
    "oom": "OOM",
    "tps": "TPS",
    "vlm": "VLM",
}

_ISSUE_FILING_TARGET_LABELS: Final[dict[str, str]] = {
    "model-config / mlx-vlm": "model repo first; mlx-vlm if template handling disagrees",
    "model configuration/repository": "model repository",
    "mlx-vlm / mlx": "mlx-vlm first; MLX if cache/runtime reproduces",
    "mlx-vlm / mlx-lm": "mlx-vlm first; mlx-lm if shared generation reproduces",
    "transformers / mlx-vlm": "transformers first; mlx-vlm integration check",
}


def _humanize_issue_subtype_token(value: str) -> str:
    """Return a readable label for an issue-subtype token or slug."""
    tokens = [token for token in re.split(r"[_\-\s/]+", value.strip()) if token]
    words: list[str] = []
    for token in tokens:
        normalized = token.casefold()
        words.append(_ISSUE_SUBTYPE_TOKEN_LABELS.get(normalized, normalized))
    if not words:
        return "Unknown issue type"
    if words[0].islower():
        words[0] = words[0].capitalize()
    return " ".join(words)


def _issue_error_code_display_label(issue_subtype: str) -> str | None:
    """Render canonical runtime error codes as maintainer-readable labels."""
    normalized = issue_subtype.strip().upper().replace("-", "_")
    if not normalized:
        return None

    package_tokens: tuple[str, ...] = tuple(
        sorted(set(_PACKAGE_CODE_MAP.values()), key=str.__len__, reverse=True)
    )
    package_token = next(
        (
            token
            for token in package_tokens
            if normalized == token or normalized.startswith(f"{token}_")
        ),
        None,
    )
    if package_token is None:
        return None

    remainder = normalized.removeprefix(package_token).removeprefix("_")
    if not remainder:
        return _ERROR_CODE_PACKAGE_DISPLAY_LABELS.get(
            package_token,
            _humanize_issue_subtype_token(package_token),
        )

    stage_tokens: tuple[str, ...] = tuple(
        sorted(set(_STAGE_CODE_MAP.values()), key=str.__len__, reverse=True)
    )
    stage_token = next(
        (token for token in stage_tokens if remainder == token or remainder.endswith(f"_{token}")),
        None,
    )
    if stage_token is None:
        return None

    phase_token = remainder.removesuffix(stage_token).removesuffix("_")
    owner_label = _ERROR_CODE_PACKAGE_DISPLAY_LABELS.get(
        package_token,
        _humanize_issue_subtype_token(package_token),
    )
    stage_label = _ERROR_CODE_STAGE_DISPLAY_LABELS.get(
        stage_token,
        _humanize_issue_subtype_token(stage_token),
    )
    if not phase_token:
        return f"{owner_label}: {stage_label}"
    return f"{owner_label}: {_humanize_issue_subtype_token(phase_token)} / {stage_label}"


def _issue_subtype_display_label(issue_subtype: str) -> str:
    """Return a maintainer-readable label for an issue subtype."""
    normalized = issue_subtype.strip().casefold().replace("-", "_")
    if label := _ISSUE_SUBTYPE_DISPLAY_LABELS.get(normalized):
        return label
    if label := _issue_error_code_display_label(issue_subtype):
        return label
    return _humanize_issue_subtype_token(issue_subtype)


def _issue_filing_target_label(owner: str) -> str:
    """Return the practical filing target shown in maintainer-facing queues."""
    normalized = " / ".join(part.strip() for part in owner.split("/") if part.strip())
    return _ISSUE_FILING_TARGET_LABELS.get(normalized, normalized or "unknown component")


def _issue_queue_target_cell(
    cluster: IssueCluster,
    *,
    escape_text: Callable[[str], str],
) -> str:
    """Render the queue target without forcing composite ownership into a raw code."""
    target = _issue_filing_target_label(cluster.owner)
    if target == cluster.owner:
        return f"`{escape_text(target)}`"
    return escape_text(target)


def _runtime_failure_problem_summary(result: PerformanceResult | None) -> str | None:
    """Return a concise problem label for a hard failure when evidence is specific."""
    kind = _runtime_failure_diagnostic_kind(result)
    return _RUNTIME_FAILURE_PROBLEM_LABELS.get(kind or "")


def _issue_problem_summary(cluster: IssueCluster) -> str:
    """Return the filing-facing problem summary for an issue cluster."""
    representative = _issue_cluster_representative(cluster)
    summary: str | None = None
    if cluster.issue_kind == "runtime_failure" and (
        runtime_summary := _runtime_failure_problem_summary(representative)
    ):
        summary = runtime_summary

    subtype = cluster.issue_subtype.casefold()
    if summary is None:
        subtype_summaries = {
            "encoding": "Tokenizer decode leaked BPE/byte markers",
            "stop_token": "Stop/control tokens leaked into generated text",
            "prompt_template": "Prompt/template output shape mismatch",
            "empty_output": "Generation returned empty output",
        }
        summary = subtype_summaries.get(subtype)
    if summary is None and subtype in {"long_context", "context_budget", "token_cap"}:
        summary = "Long-context generation collapsed or became too short"
    if summary is None:
        label = _issue_subtype_display_label(cluster.issue_subtype)
        symptom = _sanitize_bpe_display(cluster.symptom, max_len=92).strip().rstrip(".")
        if (
            not symptom
            or symptom.casefold() == label.casefold()
            or symptom.casefold() in label.casefold()
        ):
            summary = label
        else:
            summary = f"{label}: {symptom}"
    return summary


def _issue_queue_problem_cell(
    cluster: IssueCluster,
    *,
    escape_text: Callable[[str], str],
) -> str:
    """Render the queue problem as prose, not an internal subtype code."""
    return escape_text(_issue_problem_summary(cluster))


def _issue_queue_affected_cell(
    cluster: IssueCluster,
    *,
    escape_text: Callable[[str], str],
) -> str:
    """Render affected-model count plus a representative without widening the queue."""
    representative = _issue_cluster_representative(cluster)
    if representative is None:
        return "0"
    count = _issue_cluster_model_count(cluster)
    model = f"`{escape_text(representative.model_name)}`"
    if count == 1:
        return f"1: {model}"
    return f"{count}: {model} (+{count - 1})"


def _issue_queue_fixed_when(cluster: IssueCluster) -> str:
    """Return a compact acceptance signal for the queue table."""
    subtype = cluster.issue_subtype.casefold()
    if subtype == "stop_token":
        return "No leaked stop/control tokens."
    if subtype == "encoding":
        return "No BPE/byte markers in output."
    if subtype == "prompt_template":
        return "Requested sections render without template leakage."
    if subtype in {"long_context", "context_budget", "token_cap"}:
        return "Full and reduced reruns avoid context collapse."
    if cluster.issue_kind == "runtime_failure":
        return "Load/generation completes or fails with a narrower owner."
    return _truncate_text_preview(cluster.acceptance_signal, max_chars=88)


def _issue_cluster_bundle_link(
    cluster: IssueCluster,
    repro_bundles: Mapping[str, Path],
    *,
    relative_prefix: str = "../repro_bundles",
) -> str:
    """Return a short queue link to the first available repro bundle."""
    bundle_paths = [
        bundle
        for model_name in _issue_cluster_model_names(cluster)
        if (bundle := repro_bundles.get(model_name)) is not None
    ]
    if not bundle_paths:
        return "-"
    label = "repro JSON" if len(bundle_paths) == 1 else f"{len(bundle_paths)} repro JSONs"
    return f"[{label}]({relative_prefix}/{quote(bundle_paths[0].name)})"


def _render_issue_queue_table(
    clusters: Sequence[IssueCluster],
    *,
    escape_text: Callable[[str], str],
    issue_link_for_cluster: Callable[[IssueCluster], str],
    evidence_link_for_cluster: Callable[[IssueCluster], str],
) -> list[str]:
    """Render the shared issue-queue table used across Markdown artifacts."""
    rows = [
        (
            _issue_queue_target_cell(cluster, escape_text=escape_text),
            _issue_queue_problem_cell(cluster, escape_text=escape_text),
            _issue_queue_affected_cell(cluster, escape_text=escape_text),
            issue_link_for_cluster(cluster),
            evidence_link_for_cluster(cluster),
            escape_text(_issue_queue_fixed_when(cluster)),
        )
        for cluster in clusters
    ]
    return tabulate(rows, headers=_ISSUE_QUEUE_HEADERS, tablefmt="github").splitlines()


def _diagnostics_issue_queue_section(
    snapshot: DiagnosticsSnapshot,
    *,
    repro_bundles: Mapping[str, Path],
) -> list[str]:
    """Build the issue-queue block near the top of diagnostics output."""
    clusters = _build_issue_clusters(snapshot)
    parts = _begin_diagnostics_section(
        title="## Issue Queue",
        body_lines=[
            "Root-cause issue drafts are generated in "
            "[issues/index.md](../issues/index.md). Each row is intended to become one "
            "focused upstream GitHub issue.",
        ],
    )

    if not clusters:
        parts.append("No issue drafts were generated for this run.")
        parts.append("")
        return parts

    parts.extend(
        _render_issue_queue_table(
            clusters,
            escape_text=DIAGNOSTICS_ESCAPER.escape,
            issue_link_for_cluster=lambda cluster: (
                f"[issue draft](../issues/{quote(cluster.issue_filename)})"
            ),
            evidence_link_for_cluster=lambda cluster: _issue_cluster_bundle_link(
                cluster,
                repro_bundles,
            ),
        )
    )
    parts.append("")
    return parts


def _diagnostics_upstream_filing_notes_section(snapshot: DiagnosticsSnapshot) -> list[str]:
    """Build short filing guidance before the detailed diagnostics appendix."""
    clusters = _build_issue_clusters(snapshot)
    if not clusters:
        return []

    owner_counts = Counter(cluster.owner for cluster in clusters)
    owner_summary = ", ".join(
        f"`{DIAGNOSTICS_ESCAPER.escape(owner)}`={count}"
        for owner, count in sorted(owner_counts.items())
    )
    parts = _begin_diagnostics_section(
        title="## Upstream Filing Notes",
        body_lines=[
            "File one upstream issue per row above. The linked issue drafts are the "
            "pasteable bodies; this diagnostics file is the run-level queue and appendix.",
        ],
    )
    parts.append(f"- **Issue drafts:** {len(clusters)} root-cause cluster(s).")
    parts.append(f"- **Suggested targets:** {owner_summary}.")
    parts.append(
        "- **Standalone evidence:** each issue draft includes minimal inline evidence plus "
        "an exact cluster rerun command before any appendix detail.",
    )
    parts.append(
        "- **Supporting files:** `repro JSON` links point to local repro bundles with prompt, "
        "environment, and generated-output context. Attach or publish those JSON files when "
        "filing upstream; GitHub will not resolve local artifact paths from a pasted issue body.",
    )
    parts.append("")
    return parts


def _diagnostics_appendix_intro_section() -> list[str]:
    """Separate pasteable filing guidance from verbose run-level detail."""
    return [
        "## Appendix",
        "",
        "The remaining sections keep full run evidence for audit/debugging. They are not "
        "intended to be pasted wholesale into an upstream issue.",
        "",
    ]


def _diagnostics_history_section(
    *,
    failed: list[PerformanceResult],
    previous_history: HistoryRunRecord | None,
    diagnostics_context: DiagnosticsContext,
) -> list[str]:
    """Build regression/recovery and first-seen context using run history."""
    if not failed:
        return []

    failed_models = {r.model_name for r in failed}

    raw_prev_model_results = previous_history.get("model_results", {}) if previous_history else {}
    prev_model_results: dict[str, HistoryModelResultRecord] = (
        raw_prev_model_results if isinstance(raw_prev_model_results, dict) else {}
    )

    parts: list[str] = _begin_diagnostics_section(
        title="## History Context",
        body_lines=[
            "Recent reproducibility is measured from history "
            f"(up to last {DIAGNOSTICS.recent_run_window} runs where each model appears).",
        ],
    )

    if previous_history:
        reg_now = sorted(
            model for model in diagnostics_context.regressions if model in failed_models
        )
        if reg_now:
            parts.append(
                "**Regressions since previous run:** " + ", ".join(f"`{m}`" for m in reg_now),
            )
        else:
            parts.append("**Regressions since previous run:** none")

        if diagnostics_context.recoveries:
            recovery_list = sorted(diagnostics_context.recoveries)
            parts.append(
                "**Recoveries since previous run:** " + ", ".join(f"`{m}`" for m in recovery_list),
            )
        else:
            parts.append("**Recoveries since previous run:** none")
        parts.append("")
    else:
        parts.append("No prior history baseline available for regression/recovery status.")
        parts.append("")

    table_rows: list[tuple[str, ...]] = []
    for model in sorted(failed_models):
        if model in diagnostics_context.regressions:
            status = "new regression"
        elif model in diagnostics_context.new_models:
            status = "new model failing"
        else:
            prev_info = prev_model_results.get(model)
            prev_success = prev_info["success"] if prev_info is not None else None
            status = "still failing" if prev_success is False else "failing"

        info = diagnostics_context.failure_history.get(model)
        first_seen = DIAGNOSTICS_ESCAPER.escape(
            info.first_failure_timestamp if info else "unknown",
        )
        recent_repro = DIAGNOSTICS_ESCAPER.escape(_format_recent_repro_ratio(info))

        esc_model = DIAGNOSTICS_ESCAPER.escape(model)
        esc_status = DIAGNOSTICS_ESCAPER.escape(status)
        table_rows.append((f"`{esc_model}`", esc_status, first_seen, recent_repro))
    parts.extend(
        tabulate(
            table_rows,
            headers=["Model", "Status vs Previous Run", "First Seen Failing", "Recent Repro"],
            tablefmt="github",
        ).splitlines()
    )
    parts.append("")

    return parts


def _partition_success_diagnostics(
    *,
    successful: list[PerformanceResult],
    harness_results: list[tuple[PerformanceResult, str]],
    stack_signals: list[tuple[PerformanceResult, str, str]],
) -> tuple[
    list[tuple[PerformanceResult, str]],
    list[tuple[PerformanceResult, str, str]],
    list[PerformanceResult],
]:
    """Partition successful runs into exclusive diagnostics buckets.

    Precedence is stable and explicit: harness -> stack -> unflagged summary.
    """
    harness_unique: list[tuple[PerformanceResult, str]] = []
    harness_seen: set[str] = set()
    for res, text in harness_results:
        if res.model_name in harness_seen:
            continue
        harness_seen.add(res.model_name)
        harness_unique.append((res, text))

    stack_unique: list[tuple[PerformanceResult, str, str]] = []
    stack_seen: set[str] = set()
    for res, symptom, owner in stack_signals:
        if res.model_name in harness_seen or res.model_name in stack_seen:
            continue
        stack_seen.add(res.model_name)
        stack_unique.append((res, symptom, owner))

    unflagged_successful = [
        res
        for res in successful
        if res.model_name not in harness_seen and res.model_name not in stack_seen
    ]

    return harness_unique, stack_unique, unflagged_successful


def _diagnostics_coverage_and_runtime_section(
    *,
    results: list[PerformanceResult],
    failed: list[PerformanceResult],
    harness_results: list[tuple[PerformanceResult, str]],
    stack_signals: list[tuple[PerformanceResult, str, str]],
    unflagged_successful: list[PerformanceResult],
) -> list[str]:
    """Build coverage verification and aggregate runtime metrics section."""
    if not results:
        return []

    detailed_names = [
        *[res.model_name for res in failed],
        *[res.model_name for res, _ in harness_results],
        *[res.model_name for res, _symptom, _owner in stack_signals],
    ]
    summary_names = [res.model_name for res in unflagged_successful]
    listed_names = [*detailed_names, *summary_names]

    name_counts = Counter(listed_names)
    duplicate_names = sorted(name for name, count in name_counts.items() if count > 1)
    missing_names = sorted({res.model_name for res in results} - set(listed_names))

    coverage_ok = not duplicate_names and not missing_names and len(listed_names) == len(results)
    coverage_text = (
        "✅ Complete (each model appears exactly once)."
        if coverage_ok
        else "⚠️ Incomplete (duplicates or missing model entries detected)."
    )

    runtime_per_model: list[float] = []
    missing_timing_models = 0
    for res in results:
        if isinstance(res.total_time, int | float) and float(res.total_time) >= 0.0:
            runtime_per_model.append(float(res.total_time))
            continue

        generation_time = (
            float(res.generation_time)
            if isinstance(res.generation_time, int | float) and float(res.generation_time) >= 0.0
            else None
        )
        model_load_time = (
            float(res.model_load_time)
            if isinstance(res.model_load_time, int | float) and float(res.model_load_time) >= 0.0
            else None
        )

        if generation_time is not None or model_load_time is not None:
            runtime_per_model.append((generation_time or 0.0) + (model_load_time or 0.0))
        else:
            missing_timing_models += 1
            runtime_per_model.append(0.0)

    total_runtime = sum(runtime_per_model)
    avg_runtime = total_runtime / len(runtime_per_model)
    runtime_analysis = _build_runtime_analysis_summary(results)

    parts: list[str] = _begin_diagnostics_section(title="## Coverage & Runtime Metrics")
    parts.append(f"- **Detailed diagnostics models:** {len(detailed_names)}")
    parts.append(f"- **Summary diagnostics models:** {len(summary_names)}")
    parts.append(f"- **Coverage check:** {coverage_text}")

    if duplicate_names:
        duplicate_text = ", ".join(
            f"`{DIAGNOSTICS_ESCAPER.escape(name)}`" for name in duplicate_names
        )
        parts.append(f"- **Duplicates:** {duplicate_text}")
    if missing_names:
        missing_text = ", ".join(f"`{DIAGNOSTICS_ESCAPER.escape(name)}`" for name in missing_names)
        parts.append(f"- **Missing from diagnostics sections:** {missing_text}")

    parts.append(
        "- **Total model runtime (sum):** "
        f"{format_overall_runtime(total_runtime)} ({total_runtime:.2f}s)",
    )
    parts.append(
        "- **Average runtime per model:** "
        f"{format_overall_runtime(avg_runtime)} ({avg_runtime:.2f}s)",
    )
    if missing_timing_models:
        parts.append(
            "- **Runtime note:** "
            f"{missing_timing_models} model(s) had missing timing fields "
            "and were counted as 0.00s.",
        )
    if runtime_analysis is not None:
        phase_label = _RUNTIME_PHASE_LABELS[runtime_analysis["dominant_phase"]]
        phase_share = runtime_analysis["dominant_phase_share"]
        phase_count = runtime_analysis["dominant_phase_count"]
        measured_models = runtime_analysis["measured_models"]
        parts.append(
            "- **Dominant runtime phase:** "
            f"{phase_label} dominated {phase_count}/{measured_models} measured model runs "
            f"({phase_share:.0%} of tracked runtime).",
        )
        phase_totals_line = _format_runtime_phase_totals_line(
            runtime_analysis,
            trailing_period=False,
        )
        if phase_totals_line is not None:
            parts.append(phase_totals_line)
        generation_total_line = _format_runtime_generation_total_line(runtime_analysis)
        if generation_total_line is not None:
            parts.append(generation_total_line)
        termination_counts = runtime_analysis["termination_counts"]
        if termination_counts:
            termination_summary = ", ".join(
                f"{reason}={count}" for reason, count in sorted(termination_counts.items())
            )
            parts.append(f"- **Observed stop reasons:** {termination_summary}")
        parts.extend(_format_runtime_timing_snapshot_lines(runtime_analysis))
        parts.append(f"- **What this likely means:** {runtime_analysis['interpretation']}")
        parts.append(f"- **Suggested next action:** {runtime_analysis['next_action']}")

    parts.append("")
    return parts


def _diagnostics_unflagged_success_section(
    *,
    unflagged_successful: list[PerformanceResult],
) -> list[str]:
    """Build a near-end section listing successful models with no diagnostics flags."""

    def _quality_warning_summary(analysis: GenerationQualityAnalysis) -> str:
        """Build a concise one-line quality warning summary from existing analysis data."""
        signals = _summarize_quality_signals(analysis)
        if signals:
            return _truncate_text_preview(signals[0], max_chars=120)

        for issue in analysis.issues:
            if issue.startswith("⚠️HARNESS"):
                continue
            return _truncate_text_preview(issue, max_chars=120)

        return "Quality warnings detected by analysis."

    if not unflagged_successful:
        return []

    clean_models: list[str] = []
    quality_warning_models: list[tuple[str, str]] = []
    no_analysis_models: list[str] = []
    prompt_incomplete_models: list[str] = []

    for res in sorted(unflagged_successful, key=lambda row: row.model_name):
        qa = res.quality_analysis
        if qa is None and res.generation is not None:
            qa = getattr(res.generation, "quality_analysis", None)

        if qa is None:
            no_analysis_models.append(res.model_name)
            continue

        if not qa.prompt_checks_ran:
            prompt_incomplete_models.append(res.model_name)
            continue

        if qa.has_any_issues():
            quality_warning_models.append((res.model_name, _quality_warning_summary(qa)))
        else:
            clean_models.append(res.model_name)

    parts: list[str] = _begin_diagnostics_section(
        title=f"## Models Not Flagged ({len(unflagged_successful)} model(s))",
        body_lines=[
            "These models completed without diagnostics flags "
            "(no hard failure, harness warning, or stack-signal anomaly).",
        ],
    )

    if clean_models:
        parts.append(f"### Clean output ({len(clean_models)} model(s))")
        parts.append("")
        parts.extend(f"- `{DIAGNOSTICS_ESCAPER.escape(model)}`" for model in clean_models)
        parts.append("")

    if quality_warning_models:
        parts.append(
            f"### Ran, but with quality warnings ({len(quality_warning_models)} model(s))",
        )
        parts.append("")
        parts.extend(
            (f"- `{DIAGNOSTICS_ESCAPER.escape(model)}`: {DIAGNOSTICS_ESCAPER.escape(summary)}")
            for model, summary in quality_warning_models
        )
        parts.append("")

    if prompt_incomplete_models:
        parts.append(
            "### Passed (prompt-dependent quality checks unavailable) "
            f"({len(prompt_incomplete_models)} model(s))",
        )
        parts.append("")
        parts.append(
            "These outputs lack the original prompt context, so context-echo and "
            "catalog-contract checks could not be rerun.",
        )
        parts.append("")
        parts.extend(
            f"- `{DIAGNOSTICS_ESCAPER.escape(model)}`" for model in prompt_incomplete_models
        )
        parts.append("")

    if no_analysis_models:
        parts.append(
            f"### Passed (quality analysis unavailable) ({len(no_analysis_models)} model(s))",
        )
        parts.append("")
        parts.extend(f"- `{DIAGNOSTICS_ESCAPER.escape(model)}`" for model in no_analysis_models)
        parts.append("")

    return parts


def _append_repro_input_tokens(
    *,
    tokens: list[str],
    image_path: Path | None,
    run_args: argparse.Namespace | None,
) -> None:
    """Append image or folder selection flags for repro commands."""
    if image_path is not None:
        tokens.extend(["--image", str(image_path)])
        return

    if run_args is None:
        return

    run_image = getattr(run_args, "image", None)
    run_folder = getattr(run_args, "folder", None)
    if run_image is not None:
        tokens.extend(["--image", str(run_image)])
        return
    if run_folder is not None:
        tokens.extend(["--folder", str(run_folder)])


def _append_repro_selection_tokens(
    *,
    tokens: list[str],
    run_args: argparse.Namespace,
) -> None:
    """Append model selection flags for repro commands."""
    models = getattr(run_args, "models", None)
    if models:
        tokens.extend(["--models", *[str(model) for model in models]])

    exclude = getattr(run_args, "exclude", None)
    if exclude:
        tokens.extend(["--exclude", *[str(model) for model in exclude]])


def _append_repro_flags(
    *,
    tokens: list[str],
    run_args: argparse.Namespace,
    flag_map: Sequence[tuple[str, str]],
) -> None:
    """Append enabled boolean flags for repro commands."""
    for attr_name, flag in flag_map:
        if bool(getattr(run_args, attr_name, False)):
            tokens.append(flag)


def _append_repro_extended_generate_args(
    *,
    tokens: list[str],
    run_args: argparse.Namespace,
) -> None:
    """Append optional non-scalar generate kwargs for repro commands."""
    resize_shape = getattr(run_args, "resize_shape", None)
    if resize_shape is not None:
        tokens.extend(["--resize-shape", *[str(value) for value in resize_shape]])

    eos_tokens = getattr(run_args, "eos_tokens", None)
    if eos_tokens:
        tokens.extend(["--eos-tokens", *[str(token) for token in eos_tokens]])

    processor_kwargs = getattr(run_args, "processor_kwargs", None)
    if processor_kwargs:
        tokens.extend(["--processor-kwargs", json.dumps(processor_kwargs, sort_keys=True)])


def _thinking_end_token_for_repro(run_args: argparse.Namespace) -> str | None:
    """Return a non-default thinking end token for repro commands."""
    if not bool(getattr(run_args, "enable_thinking", False)):
        return None

    thinking_end_token = getattr(run_args, "thinking_end_token", DEFAULT_THINKING_END_MARKER)
    if thinking_end_token == DEFAULT_THINKING_END_MARKER:
        return None
    return str(thinking_end_token)


def _build_repro_command_tokens(
    *,
    image_path: Path | None,
    run_args: argparse.Namespace | None,
    include_selection: bool,
    include_input: bool = True,
) -> list[str]:
    """Build CLI command tokens for diagnostics reproducibility snippets."""

    def _append_pairs(
        pairs: Sequence[tuple[str, str | int | float | Path | None]],
    ) -> None:
        for flag, value in pairs:
            if value is not None:
                tokens.extend([flag, str(value)])

    tokens = ["python", "-m", "check_models"]
    if include_input:
        _append_repro_input_tokens(tokens=tokens, image_path=image_path, run_args=run_args)

    if run_args is None:
        return tokens

    if include_selection:
        _append_repro_selection_tokens(tokens=tokens, run_args=run_args)

    trust_remote_code = bool(getattr(run_args, "trust_remote_code", True))
    tokens.append("--trust-remote-code" if trust_remote_code else "--no-trust-remote-code")

    _append_pairs(
        (
            ("--revision", getattr(run_args, "revision", None)),
            ("--adapter-path", getattr(run_args, "adapter_path", None)),
            ("--prompt", getattr(run_args, "prompt", None)),
        ),
    )
    _append_repro_flags(
        tokens=tokens,
        run_args=run_args,
        flag_map=(
            ("detailed_metrics", "--detailed-metrics"),
            ("lazy_load", "--lazy-load"),
            ("skip_special_tokens", "--skip-special-tokens"),
            ("enable_thinking", "--enable-thinking"),
        ),
    )
    tokens.extend(["--max-tokens", str(getattr(run_args, "max_tokens", DEFAULT_MAX_TOKENS))])
    tokens.extend(["--temperature", str(getattr(run_args, "temperature", DEFAULT_TEMPERATURE))])
    tokens.extend(["--top-p", str(getattr(run_args, "top_p", 1.0))])

    kv_group_size = getattr(run_args, "kv_group_size", None)
    quantized_kv_start = getattr(run_args, "quantized_kv_start", None)
    _append_repro_extended_generate_args(tokens=tokens, run_args=run_args)
    _append_pairs(
        (
            ("--repetition-penalty", getattr(run_args, "repetition_penalty", None)),
            ("--repetition-context-size", getattr(run_args, "repetition_context_size", None)),
            (
                "--min-p",
                getattr(run_args, "min_p", None)
                if getattr(run_args, "min_p", 0.0) not in {None, 0.0}
                else None,
            ),
            (
                "--top-k",
                getattr(run_args, "top_k", None)
                if getattr(run_args, "top_k", 0) not in {None, 0}
                else None,
            ),
            ("--max-kv-size", getattr(run_args, "max_kv_size", None)),
            ("--kv-bits", getattr(run_args, "kv_bits", None)),
            (
                "--kv-quant-scheme",
                getattr(run_args, "kv_quant_scheme", DEFAULT_KV_QUANT_SCHEME)
                if getattr(run_args, "kv_quant_scheme", DEFAULT_KV_QUANT_SCHEME)
                != DEFAULT_KV_QUANT_SCHEME
                else None,
            ),
            ("--prefill-step-size", getattr(run_args, "prefill_step_size", None)),
            ("--thinking-budget", getattr(run_args, "thinking_budget", None)),
            ("--thinking-start-token", getattr(run_args, "thinking_start_token", None)),
            (
                "--kv-group-size",
                kv_group_size if kv_group_size not in {None, 64} else None,
            ),
            (
                "--quantized-kv-start",
                quantized_kv_start
                if quantized_kv_start not in {None, DEFAULT_QUANTIZED_KV_START}
                else None,
            ),
            ("--thinking-end-token", _thinking_end_token_for_repro(run_args)),
            ("--timeout", getattr(run_args, "timeout", DEFAULT_TIMEOUT)),
        ),
    )

    _append_repro_flags(
        tokens=tokens,
        run_args=run_args,
        flag_map=(
            ("verbose", "--verbose"),
            ("no_color", "--no-color"),
            ("force_color", "--force-color"),
        ),
    )

    context_marker = getattr(run_args, "context_marker", None)
    _append_pairs(
        (
            ("--width", getattr(run_args, "width", None)),
            ("--quality-config", getattr(run_args, "quality_config", None)),
            (
                "--context-marker",
                context_marker if context_marker and context_marker != "Context:" else None,
            ),
        ),
    )

    return tokens


_REPRO_ENV_KEYS: Final[tuple[str, ...]] = (
    "HF_HOME",
    "HF_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "TOKENIZERS_PARALLELISM",
    "MLX_VLM_WIDTH",
    "PYTHONPATH",
)


def _jsonify_cli_value(value: object) -> object:
    """Convert argparse values into JSON-safe primitives."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_jsonify_cli_value(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonify_cli_value(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _jsonify_cli_value(v) for k, v in value.items()}
    return str(value)


def _sha256_text(value: str) -> str:
    """Return SHA256 hash of UTF-8 text."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str | None:
    """Return SHA256 hash for file contents, or None if unreadable."""
    if not path.exists() or not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()
    except OSError:
        return None


def _sanitize_bundle_filename(model_name: str) -> str:
    """Create a filesystem-safe basename from model identifier."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("._")
    return cleaned or "model"


def _collect_repro_env() -> dict[str, str]:
    """Collect relevant environment variables for reproducibility bundles."""
    env: dict[str, str] = {}
    for key in _REPRO_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            env[key] = value
    return env


def export_failure_repro_bundles(
    *,
    results: list[PerformanceResult],
    output_dir: Path,
    run_args: argparse.Namespace,
    versions: LibraryVersionDict,
    system_info: dict[str, str],
    prompt: str,
    image_path: Path | None,
) -> dict[str, Path]:
    """Write reproducibility bundles for failed and issue-clustered results."""
    bundles: dict[str, Path] = {}
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    prompt_hash = _sha256_text(prompt)
    image_hash = _sha256_file(image_path) if image_path is not None else None
    image_ref_hash = _sha256_text(str(image_path)) if image_path is not None else None
    env_vars = _collect_repro_env()
    serialized_args = {
        key: _jsonify_cli_value(value) for key, value in sorted(vars(run_args).items())
    }
    preflight_issues = list(_get_run_preflight_issues(run_args))
    issue_cluster_by_model = _issue_cluster_map_for_results(results, prompt=prompt)
    bundled_results = [
        result
        for result in results
        if not result.success or result.model_name in issue_cluster_by_model
    ]
    if not bundled_results:
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)

    for index, result in enumerate(bundled_results, start=1):
        safe_model = _sanitize_bundle_filename(result.model_name)
        issue_cluster = issue_cluster_by_model.get(result.model_name)
        signature_source = result.error_signature or (
            issue_cluster.cluster_id if issue_cluster is not None else "unknown"
        )
        signature_token = re.sub(r"[^A-Za-z0-9_]+", "_", signature_source)
        signature_token = signature_token[:40] if signature_token else "unknown"
        bundle_name = f"{timestamp}_{index:03d}_{safe_model}_{signature_token}.json"
        bundle_path = output_dir / bundle_name

        rerun_tokens = _build_repro_command_tokens(
            image_path=image_path,
            run_args=run_args,
            include_selection=False,
        )
        rerun_command = shlex_join([*rerun_tokens, "--models", result.model_name])
        cluster_repro_command = (
            shlex_join(
                [
                    *rerun_tokens,
                    "--models",
                    *_issue_cluster_model_names(issue_cluster),
                ]
            )
            if issue_cluster is not None
            else None
        )
        repro_payload: dict[str, object] = {
            "rerun_command": rerun_command,
            "seed": serialized_args.get("seed"),
            "prompt_hash_sha256": prompt_hash,
            "prompt_preview": _build_prompt_preview(prompt, max_chars=400),
            "image_path": str(image_path) if image_path is not None else None,
            "image_sha256": image_hash,
            "image_ref_sha256": image_ref_hash,
            "args": serialized_args,
            "env_vars": env_vars,
        }
        prompt_diagnostics = _prompt_diagnostics_to_json(result.prompt_diagnostics)
        if prompt_diagnostics:
            repro_payload["prompt_diagnostics"] = prompt_diagnostics

        bundle_payload: dict[str, object] = {
            "schema_version": "1.0",
            "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "model": result.model_name,
            "failure": {
                "phase": result.failure_phase,
                "stage": result.error_stage,
                "code": result.error_code,
                "signature": result.error_signature,
                "type": result.error_type,
                "package": result.error_package,
                "message": result.error_message,
                "traceback": result.error_traceback,
                "captured_output": result.captured_output_on_fail,
            },
            "result": {
                "success": result.success,
                "quality_issues": _parse_quality_issues_to_list(result.quality_issues),
                "maintainer_triage": _maintainer_triage_for_result(result),
                "generated_text_preview": _build_prompt_preview(
                    _generation_text_value(result.generation),
                    max_chars=400,
                )
                if result.generation is not None
                else None,
            },
            "repro": repro_payload,
            "environment": {
                "platform": platform.platform(),
                "system_info": system_info,
                "library_versions": versions,
                "preflight_issues": preflight_issues,
            },
        }
        if issue_cluster is not None:
            bundle_payload["issue_cluster_id"] = issue_cluster.cluster_id
            if cluster_repro_command is not None:
                repro_payload["cluster_repro_command"] = cluster_repro_command

        try:
            bundle_path.write_text(
                json.dumps(bundle_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except OSError:
            logger.exception("Failed to write repro bundle for %s", result.model_name)
            continue
        bundles[result.model_name] = bundle_path

    return bundles


def _diagnostics_footer(
    failed: list[PerformanceResult],
    prompt: str,
    *,
    diagnostics_snapshot: DiagnosticsSnapshot,
    image_path: Path | None,
    run_args: argparse.Namespace | None,
) -> list[str]:
    def _portable_probe_model_names(snapshot: DiagnosticsSnapshot) -> list[str]:
        names = [result.model_name for result in snapshot.failed]
        names.extend(result.model_name for result, _sample in snapshot.harness_results)
        names.extend(result.model_name for result, _symptom, _owner in snapshot.stack_signals)
        return _dedupe_preserve_order(names)

    def _portable_load_probe_script(model_names: Sequence[str]) -> str:
        load_kwargs = {
            "adapter_path": getattr(run_args, "adapter_path", None) if run_args else None,
            "lazy": bool(getattr(run_args, "lazy_load", False)) if run_args else False,
            "revision": getattr(run_args, "revision", None) if run_args else None,
            "trust_remote_code": bool(getattr(run_args, "trust_remote_code", True))
            if run_args
            else True,
        }
        return "\n".join(
            (
                "python - <<'PY'",
                "from mlx_vlm.utils import load  # mlx_vlm.utils.load",
                "",
                f"MODELS = {list(model_names)!r}",
                f"LOAD_KWARGS = {load_kwargs!r}",
                "",
                "for model_id in MODELS:",
                '    print(f"== {model_id} ==")',
                "    try:",
                "        model, processor = load(model_id, **LOAD_KWARGS)",
                "    except Exception as exc:",
                '        print(f"load FAIL: {type(exc).__name__}: {exc}")',
                "        continue",
                "",
                "    config = getattr(model, 'config', None)",
                "    tokenizer = getattr(processor, 'tokenizer', None)",
                "    image_processor = getattr(processor, 'image_processor', None)",
                '    print("load OK")',
                '    print(f"model_class={type(model).__name__}")',
                "    config_class = type(config).__name__ if config is not None else 'None'",
                '    print(f"config_class={config_class}")',
                "    model_type = getattr(config, 'model_type', None)",
                '    print(f"model_type={model_type}")',
                '    print(f"processor_class={type(processor).__name__}")',
                "    tokenizer_class = type(tokenizer).__name__ if tokenizer is not None else 'None'",
                '    print(f"tokenizer_class={tokenizer_class}")',
                '    print(f"has_image_processor={image_processor is not None}")',
                "    eos_token = getattr(tokenizer, 'eos_token', None)",
                "    eos_token_id = getattr(tokenizer, 'eos_token_id', None)",
                '    print(f"eos_token={eos_token}")',
                '    print(f"eos_token_id={eos_token_id}")',
                "PY",
            )
        )

    def _portable_generation_probe_script(model_names: Sequence[str]) -> str:
        tokens = _build_repro_command_tokens(
            image_path=None,
            run_args=run_args,
            include_selection=False,
            include_input=False,
        )
        if run_args is None or getattr(run_args, "prompt", None) is None:
            tokens.extend(["--prompt", prompt])
        tokens.extend(["--image", _PORTABLE_PROBE_IMAGE_PATH, "--models", *model_names])
        rerun_command = shlex_join(tokens)
        return "\n".join(
            (
                "python - <<'PY'",
                "from PIL import Image",
                "",
                "size = 32",
                "image = Image.new('RGB', (size, size))",
                "pixels = [",
                "    ((x * 7) % 256, (y * 11) % 256, ((x + y) * 13) % 256)",
                "    for y in range(size)",
                "    for x in range(size)",
                "]",
                "image.putdata(pixels)",
                f"image.save('{_PORTABLE_PROBE_IMAGE_PATH}')",
                f"print('wrote {_PORTABLE_PROBE_IMAGE_PATH}')",
                "PY",
                "",
                rerun_command,
            )
        )

    all_run_tokens = _build_repro_command_tokens(
        image_path=image_path,
        run_args=run_args,
        include_selection=True,
    )
    all_run_command = shlex_join(all_run_tokens)

    all_models_command = (
        "# Install check_models benchmarking tool\n"
        'pip install -e "src/[dev]"\n'
        "\n"
        "# Re-run with the same CLI arguments\n"
        f"{all_run_command}"
    )
    portable_probe_models = _portable_probe_model_names(diagnostics_snapshot)
    parts: list[str] = []
    _append_markdown_section(parts, title="## Reproducibility")
    _append_markdown_code_block(parts, all_models_command, language="bash")

    portable_body_lines = [
        "These probes are not a substitute for the original repro bundle. They help "
        "upstream maintainers separate package/import problems, model repository/config/load "
        "problems, and image-dependent generation problems without needing the original local image.",
    ]
    if not portable_probe_models:
        portable_body_lines.append(
            "This run did not queue any model-specific failure, harness, or stack-signal probes, "
            "so only the environment sanity step is shown below."
        )
    _append_markdown_section(
        parts,
        title="### Portable upstream probes (no local image required)",
        body_lines=portable_body_lines,
    )

    portable_probe_blocks = [
        (
            "# 1) Environment sanity: package versions + import smoke test\n"
            f"{_PORTABLE_DEPENDENCY_PROBE_CMD}\n"
            "\n"
            "python - <<'PY'\n"
            "import importlib\n"
            "packages = ('mlx', 'mlx_vlm', 'mlx_lm', 'transformers', 'huggingface_hub', 'tokenizers')\n"
            "for name in packages:\n"
            "    try:\n"
            "        mod = importlib.import_module(name)\n"
            "        version = getattr(mod, '__version__', 'unknown')\n"
            "        print(f'{name} OK {version}')\n"
            "    except Exception as exc:\n"
            "        print(f'{name} FAIL {type(exc).__name__}: {exc}')\n"
            "PY"
        )
    ]
    if portable_probe_models:
        portable_probe_blocks.append(
            "\n".join(
                (
                    "# 2) Model load/config probe: no original image required",
                    _portable_load_probe_script(portable_probe_models),
                )
            )
        )
        portable_probe_blocks.append(
            "\n".join(
                (
                    "# 3) Synthetic-image generation probe: separates image-dependent failures",
                    _portable_generation_probe_script(portable_probe_models),
                )
            )
        )
    _append_markdown_code_block(parts, "\n\n".join(portable_probe_blocks), language="bash")

    if failed:
        _append_markdown_section(
            parts,
            title="### Target specific failing models",
            body_lines=[
                (
                    "**Note:** A comprehensive JSON reproduction bundle including system info "
                    "and the exact prompt trace has been exported to "
                    "[repro_bundles/]"
                    "(https://github.com/jrp2014/check_models/tree/main/src/output/repro_bundles) "
                    "for each failing model."
                ),
            ],
        )
        target_base_tokens = _build_repro_command_tokens(
            image_path=image_path,
            run_args=run_args,
            include_selection=False,
        )
        failed_models = sorted({r.model_name for r in failed})
        target_model_commands = "\n".join(
            shlex_join([*target_base_tokens, "--models", model_name])
            for model_name in failed_models
        )
        _append_markdown_code_block(parts, target_model_commands, language="bash")

    _append_markdown_section(parts, title="### Prompt Used")
    _append_markdown_code_block(parts, prompt, language="text")
    _append_markdown_section(parts, title="### Run details")
    image_detail = str(image_path) if image_path is not None else "not specified"
    parts.append(f"- Input image: `{DIAGNOSTICS_ESCAPER.escape(image_detail)}`")
    if run_args is not None:
        max_tokens = getattr(run_args, "max_tokens", DEFAULT_MAX_TOKENS)
        temperature = getattr(run_args, "temperature", DEFAULT_TEMPERATURE)
        top_p = getattr(run_args, "top_p", 1.0)
        parts.append(
            f"- Generation settings: max_tokens={max_tokens}, "
            f"temperature={temperature}, top_p={top_p}",
        )
    parts.append("")
    parts.append(
        f"_Report generated on {local_now_str()} by "
        "[check_models](https://github.com/jrp2014/check_models)._",
    )

    return parts


def generate_diagnostics_report(
    *,
    results: list[PerformanceResult],
    filename: Path,
    versions: LibraryVersionDict,
    system_info: dict[str, str],
    prompt: str,
    image_path: Path | None = None,
    run_args: argparse.Namespace | None = None,
    history: DiagnosticsHistoryInputs | None = None,
    repro_bundles: Mapping[str, Path] | None = None,
    diagnostics_snapshot: DiagnosticsSnapshot | None = None,
) -> bool:
    """Generate a Markdown diagnostics report structured for upstream issue filing.

    The report clusters failures by root-cause pattern, includes full error
    messages and traceback excerpts, and highlights harness/encoding issues
    from successful models in a self-contained format suitable for direct
    upstream issue filing against mlx-vlm, mlx, or transformers.

    Args:
        results: All PerformanceResult objects from the run.
        filename: Output path for the diagnostics .md file.
        versions: Library version mapping for the environment table.
        system_info: System characteristics dict from get_system_characteristics().
        prompt: The prompt that was used.
        image_path: Path to the test image (for reproducibility section).
        run_args: Optional parsed CLI args from this run for exact repro commands.
        history: Optional history inputs for first-seen/repro/regression context.
        repro_bundles: Optional model->bundle path mapping from repro export.
        diagnostics_snapshot: Optional cached diagnostics classification for this run.

    Returns:
        True if the report was written (i.e. there was something to report),
        False if skipped because there were no failures or harness issues.
    """
    if diagnostics_snapshot is None:
        diagnostics_snapshot = _build_diagnostics_snapshot(
            results=results,
            prompt=prompt,
            preflight_issues=history.preflight_issues if history is not None else (),
        )

    failed = list(diagnostics_snapshot.failed)
    harness_results = list(diagnostics_snapshot.harness_results)
    stack_signals = list(diagnostics_snapshot.stack_signals)
    unflagged_successful = list(diagnostics_snapshot.unflagged_successful)
    preflight_issues = list(diagnostics_snapshot.preflight_issues)
    successful_count = len(results) - len(failed)
    resolved_repro_bundles = repro_bundles or {}

    if not failed and not harness_results and not stack_signals and not preflight_issues:
        return False

    failure_clusters = [
        (signature, list(cluster_results))
        for signature, cluster_results in diagnostics_snapshot.failure_clusters
    ]

    history_path = history.history_path if history is not None else None
    previous_history = history.previous_history if history is not None else None
    current_history = history.current_history if history is not None else None

    history_records = _load_history_run_records(history_path)
    failed_models = {r.model_name for r in failed}
    comparison = (
        compare_history_records(previous_history, current_history)
        if previous_history and current_history
        else None
    )
    diagnostics_context = _build_diagnostics_context(
        failed_models=failed_models,
        history_records=history_records,
        comparison=comparison,
    )

    parts: list[str] = _diagnostics_header(
        total=len(results),
        n_failed=len(failed),
        n_harness=len(harness_results),
        n_preflight=len(preflight_issues),
        n_success=successful_count,
        versions=versions,
        image_path=image_path,
    )
    parts.extend(
        _diagnostics_issue_queue_section(
            diagnostics_snapshot,
            repro_bundles=resolved_repro_bundles,
        )
    )
    parts.extend(_diagnostics_upstream_filing_notes_section(diagnostics_snapshot))
    parts.extend(_diagnostics_appendix_intro_section())
    parts.extend(
        _diagnostics_failure_clusters(
            failure_clusters,
            diagnostics_context=diagnostics_context,
        ),
    )
    parts.extend(_diagnostics_preflight_section(preflight_issues))
    parts.extend(_diagnostics_harness_section(harness_results))

    # Render stack signals merged into the harness/integration section
    parts.extend(_diagnostics_stack_signal_section(stack_signals))

    parts.extend(
        _diagnostics_history_section(
            failed=failed,
            previous_history=previous_history,
            diagnostics_context=diagnostics_context,
        ),
    )
    parts.extend(
        _diagnostics_coverage_and_runtime_section(
            results=results,
            failed=failed,
            harness_results=harness_results,
            stack_signals=stack_signals,
            unflagged_successful=unflagged_successful,
        ),
    )
    parts.extend(
        _diagnostics_unflagged_success_section(
            unflagged_successful=unflagged_successful,
        ),
    )
    parts.extend(
        _diagnostics_environment_section(
            versions=versions,
            system_info=system_info,
        ),
    )
    parts.extend(
        _diagnostics_footer(
            failed,
            prompt,
            diagnostics_snapshot=diagnostics_snapshot,
            image_path=image_path,
            run_args=run_args,
        ),
    )

    try:
        filename.parent.mkdir(parents=True, exist_ok=True)
        filename.write_text("\n".join(parts) + "\n", encoding="utf-8")
    except OSError:
        logger.exception("Failed to write diagnostics report to %s", filename)
        return False
    else:
        return True


def format_issues_summary_text(summary: ModelIssueSummary, stats: PerformanceStats) -> str:
    """Format the issues and statistics summary as a Markdown string."""
    return "\n".join(_format_issues_summary_parts(summary, stats, html_output=False))


def _build_full_html_document(
    *,
    html_table: str,
    versions: LibraryVersionDict,
    prompt: str,
    total_runtime_seconds: float,
    action_snapshot_html: str,
    issues_summary_html: str,
    review_priorities_html: str,
    failures_by_package_html: str,
    system_info: dict[str, str],
    image_path: Path | None = None,
) -> str:
    """Build the full self-contained HTML document from components with optional embedded image."""
    css = """
    <style>
        body { font-family: sans-serif; margin: 2em; }
        table { border-collapse: collapse; margin-top: 1em; width: 100%; }
        th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; font-weight: bold; }
        .numeric { text-align: right; }
        .failed { background-color: #ffdddd; }
        tr.hidden { display: none; }
        .summary {
            margin-top: 2em; padding: 1em; border: 1px solid #eee;
            background-color: #f9f9f9;
        }
        .summary h3 { margin-top: 1.2em; }
        .summary h3:first-child { margin-top: 0; }
        .summary ul { margin-top: 0.4em; }
        .embedded-image {
            max-width: 600px; margin: 1em 0; border: 1px solid #ccc;
            border-radius: 4px;
        }
        details { cursor: pointer; max-width: 800px; }
        details summary {
            font-weight: normal;
            color: #0066cc;
            padding: 0.25em;
            user-select: none;
        }
        details summary:hover { background-color: #f0f0f0; }
        details[open] summary { color: #004499; font-weight: bold; }
        details > div {
            margin-top: 0.5em;
            padding: 0.5em;
            border-left: 3px solid #0066cc;
            background-color: #f8f8f8;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .filter-controls {
            margin: 1em 0;
            padding: 1em;
            background-color: #f5f5f5;
            border-radius: 4px;
            border: 1px solid #ddd;
        }
        .filter-btn {
            padding: 0.5em 1em;
            margin: 0.25em;
            border: 1px solid #999;
            background-color: #fff;
            border-radius: 3px;
            cursor: pointer;
            font-size: 0.9em;
        }
        .filter-btn:hover {
            background-color: #e8e8e8;
        }
        .filter-btn.active {
            background-color: #0066cc;
            color: white;
            border-color: #0055aa;
        }
        .filter-info {
            margin-top: 0.5em;
            font-size: 0.9em;
            color: #666;
        }
    </style>
    """
    sys_info_html = "<ul>"
    for k, v in system_info.items():
        sys_info_html += f"<li><b>{html.escape(k)}:</b> {html.escape(v)}</li>"
    sys_info_html += "</ul>"

    # Build filter controls with JavaScript
    filter_controls = """
    <div class="filter-controls">
        <div>
            <strong>Filter Results:</strong>
            <button class="filter-btn active" onclick="filterTable('all')">All</button>
            <button class="filter-btn" onclick="filterTable('success')">Success Only</button>
            <button class="filter-btn" onclick="filterTable('failed')">Failed Only</button>
            <button class="filter-btn" onclick="filterTable('load')">Load Errors</button>
            <button class="filter-btn" onclick="filterTable('generation')">
                Generation Errors
            </button>
            <button class="filter-btn" onclick="filterTable('timeout')">Timeouts</button>
        </div>
        <div class="filter-info" id="filter-info">Showing all rows</div>
    </div>

    <script>
    function filterTable(filterType) {
        const table = document.querySelector('table');
        const rows = table.querySelectorAll('tr[data-status]');
        let visibleCount = 0;

        // Update button states
        document.querySelectorAll('.filter-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        event.target.classList.add('active');

        // Apply filter
        rows.forEach(row => {
            let show = false;

            if (filterType === 'all') {
                show = true;
            } else if (filterType === 'success') {
                show = row.dataset.status === 'success';
            } else if (filterType === 'failed') {
                show = row.dataset.status === 'failed';
            } else {
                // Filter by error stage
                show = row.dataset.errorStage === filterType;
            }

            if (show) {
                row.classList.remove('hidden');
                visibleCount++;
            } else {
                row.classList.add('hidden');
            }
        });

        // Update info text
        const totalRows = rows.length;
        const filterInfo = document.getElementById('filter-info');
        if (filterType === 'all') {
            filterInfo.textContent = `Showing all ${totalRows} rows`;
        } else {
            filterInfo.textContent = `Showing ${visibleCount} of ${totalRows} rows`;
        }
    }
    </script>
    """

    versions_html = "<ul>"
    for name, ver in sorted(versions.items()):
        ver_str = "" if ver is None else ver
        versions_html += (
            f"<li><code>{html.escape(name)}</code>: <code>{html.escape(ver_str)}</code></li>"
        )
    versions_html += "</ul>"

    # Embed image if provided
    image_html = ""
    if image_path and image_path.exists():
        try:
            # Open and resize image if needed
            with Image.open(image_path) as img_original:
                # Resize if larger than 1024px in either dimension
                max_size = 1024
                # Explicitly type as generic Image to handle both ImageFile and resized Image
                img_to_save: PILImage = img_original
                if img_original.width > max_size or img_original.height > max_size:
                    # Calculate new dimensions maintaining aspect ratio
                    ratio = min(max_size / img_original.width, max_size / img_original.height)
                    new_width = int(img_original.width * ratio)
                    new_height = int(img_original.height * ratio)
                    img_to_save = img_original.resize(
                        (new_width, new_height),
                        Image.Resampling.LANCZOS,
                    )

                # Convert to bytes
                img_buffer = io.BytesIO()
                # Determine format from extension
                ext = image_path.suffix.lower()
                img_format = {
                    ".jpg": "JPEG",
                    ".jpeg": "JPEG",
                    ".png": "PNG",
                    ".gif": "GIF",
                    ".webp": "WEBP",
                }.get(ext, "JPEG")
                img_to_save.save(img_buffer, format=img_format)
                img_data = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

                # Determine MIME type
                mime_type = {
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".png": "image/png",
                    ".gif": "image/gif",
                    ".webp": "image/webp",
                }.get(ext, "image/jpeg")
                image_html = (
                    f"<h2>Test Image</h2>"
                    f'<img src="data:{mime_type};base64,{img_data}" '
                    f'class="embedded-image" '
                    f'alt="Test image used for model evaluation" />'
                )
        except (OSError, ValueError):
            logger.warning("Failed to embed image: %s", image_path)

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Model Performance Report</title>
        {css}
    </head>
    <body>
        <h1>Model Performance Report</h1>
        <p><em>Generated on {local_now_str()}</em></p>
        {image_html}
        <div class="summary">
            <h2>Summary</h2>
            {action_snapshot_html}
            {issues_summary_html}
            {review_priorities_html}
            {failures_by_package_html}
        </div>
        <h2>Prompt</h2>
        <pre>{html.escape(prompt)}</pre>
        <h2>Results</h2>
        <p><strong>Overall runtime:</strong> {format_overall_runtime(total_runtime_seconds)}</p>
        {filter_controls}
        {html_table}
        <h2>System Information</h2>
        {sys_info_html}
        <h2>Library Versions</h2>
        {versions_html}
    </body>
    </html>
    """


def print_model_stats(results: list[PerformanceResult]) -> None:
    """Print model performance statistics in a formatted table."""
    if not results:
        logger.info("No results to display.")
        return

    table_data = _build_prepared_table_data(
        result_set=ResultSet(results),
        header_separator="\n",
        include_output=False,
    )
    headers, rows, field_names = _materialize_prepared_table_data(table_data)
    if not headers or not rows:
        logger.info("No data to display in stats table.")
        return

    _log_rich_table(
        headers=headers,
        rows=rows,
        justifications=["right" if is_numeric_field(field) else "left" for field in field_names],
        max_widths=[20 if field == "quality_issues" else None for field in field_names],
        indent="",
    )


# =============================================================================
# REPORT GENERATION (HTML, Markdown)
# =============================================================================


def generate_html_report(
    results: list[PerformanceResult],
    filename: Path,
    versions: LibraryVersionDict,
    prompt: str,
    total_runtime_seconds: float,
    image_path: Path | None = None,
    report_context: ReportRenderContext | None = None,
) -> None:
    """Write a self-contained HTML summary with aligned table and embedded image.

    Args:
        results: Run results to render.
        filename: Output HTML file path.
        versions: Installed library versions shown in the report.
        prompt: Prompt used for the run.
        total_runtime_seconds: Total wall-clock runtime for the full run.
        image_path: Optional input image path for the report header.
        report_context: Optional cached shared report context built in finalization.
    """
    if not results:
        log_warning_note("No results to generate HTML report.")
        return

    if report_context is None:
        report_context = _build_report_render_context(results=results, prompt=prompt)

    headers, rows, field_names = _materialize_prepared_table_data(report_context.table_data)

    if not headers or not rows:
        log_warning_note("No table data to generate HTML report.")
        return

    html_table = tabulate(
        rows,
        headers=headers,
        tablefmt="html",
        colalign=[
            "left",
            *["right" if is_numeric_field(field) else "left" for field in field_names[1:]],
        ],
    )

    # Add CSS classes for alignment and styling
    html_table = _add_html_table_alignment_classes(html_table, field_names)

    # Mark failed rows using the already-sorted cached context
    html_table = _mark_failed_rows_in_html(html_table, report_context.result_set.results)

    # Wrap output column (last column) in <details> for expandability
    output_col_idx = len(field_names) - 1  # output is last column
    html_table = _wrap_output_column_in_details(
        html_table,
        output_col_idx,
        report_context.result_set.results,
    )

    issues_summary_html = format_issues_summary_html(
        report_context.summary,
        report_context.stats,
    )
    action_snapshot_html = "".join(
        _format_action_snapshot_parts(
            results,
            report_context,
            html_output=True,
        ),
    )
    review_priorities_html = "".join(
        _format_review_priorities_parts(
            report_context,
            html_output=True,
        ),
    )
    failures_by_package_html = "".join(
        _format_failures_by_package_parts(
            results,
            html_output=True,
        ),
    )

    # Build the full HTML document
    html_content = _build_full_html_document(
        html_table=html_table,
        versions=versions,
        prompt=prompt,
        total_runtime_seconds=total_runtime_seconds,
        action_snapshot_html=action_snapshot_html,
        issues_summary_html=issues_summary_html,
        review_priorities_html=review_priorities_html,
        failures_by_package_html=failures_by_package_html,
        system_info=report_context.system_info,
        image_path=image_path,
    )

    try:
        with filename.open("w", encoding="utf-8") as f:
            f.write(html_content)
        # Logging handled in finalize_execution
    except OSError:
        logger.exception("Failed to write HTML report to %s", filename)


def _process_markdown_rows(
    rows: list[list[str]],
    sorted_results: Sequence[PerformanceResult],
) -> None:
    """Process table rows for Markdown: escape content and format model names."""
    for i in range(len(rows)):
        # Wrap model name in backticks to preserve underscores and special chars
        if rows[i][0]:
            rows[i][0] = f"`{rows[i][0]}`"

        last_col_idx = len(rows[i]) - 1
        if last_col_idx < 0:
            continue
        # If corresponding result failed, treat as diagnostics and escape more aggressively
        is_failure = i < len(sorted_results) and not sorted_results[i].success
        if is_failure:
            rows[i][last_col_idx] = DIAGNOSTICS_ESCAPER.escape(rows[i][last_col_idx])
        else:
            # Minimal structural escaping only (protect pipes/HTML-like tags, preserve output
            # as-is otherwise)
            rows[i][last_col_idx] = MARKDOWN_ESCAPER.escape(rows[i][last_col_idx])


def _generate_model_gallery_section(
    report_context: ReportRenderContext | list[PerformanceResult],
) -> list[str]:
    """Generate the Model Gallery section for the Markdown report."""
    md: list[str] = []
    md.append("## Model Gallery")
    md.append("")
    md.append("Full generated output by model:")
    md.append("")
    md.append("<!-- markdownlint-disable MD033 MD034 -->")
    md.append("")

    sorted_results = (
        report_context.result_set.results
        if isinstance(report_context, ReportRenderContext)
        else ResultSet(report_context).results
    )
    summary = report_context.summary if isinstance(report_context, ReportRenderContext) else None
    useful_model_names: set[str] = set()
    watchlist_by_model: dict[str, str] = {}
    if isinstance(report_context, ReportRenderContext):
        useful_model_names = {row.result.model_name for row in report_context.triage.useful_rows}
        watchlist_by_model = {
            row.result.model_name: reason for row, reason in report_context.triage.watchlist_rows
        }
    for res in sorted_results:
        icon = "✅" if res.success else "❌"
        md.append(f'<a id="{_gallery_model_anchor(res.model_name)}"></a>')
        md.append("")
        md.append(f"### {icon} {res.model_name}")
        md.append("")
        if not res.success:
            block_lines = _build_gallery_error_block_lines(res)
        else:
            block_lines = _build_gallery_success_block_lines(
                res,
                summary=summary,
                useful_now=res.model_name in useful_model_names,
                watchlist_reason=watchlist_by_model.get(res.model_name),
            )
        while block_lines and block_lines[-1] == "":
            block_lines.pop()
        md.extend(block_lines)
        md.append("")
        md.append("---")
        md.append("")

    md.append("<!-- markdownlint-enable MD033 MD034 -->")
    md.append("")
    return md


def _generate_markdown_table_section(report_context: ReportRenderContext) -> list[str]:
    """Generate the metrics table section for the Markdown report."""
    headers, rows, field_names = _materialize_prepared_table_data(report_context.table_data)

    # For Markdown, we need to process headers to remove HTML breaks and use simpler formatting
    markdown_headers: list[str] = []

    # Remove "Output" column from table data for Markdown report
    # We will show it in a separate "Model Gallery" section instead
    output_col_idx = -1
    if "output" in field_names:
        output_col_idx = field_names.index("output")
        headers.pop(output_col_idx)
        # We don't pop from field_names yet as we might need it for alignment,
        # but we must remove it from rows
        for row in rows:
            if len(row) > output_col_idx:
                row.pop(output_col_idx)
        # Now remove from field_names
        field_names.pop(output_col_idx)

    for header in headers:
        # Replace <br> with space for Markdown compatibility
        clean_header = header.replace("<br>", " ")
        markdown_headers.append(clean_header)

    # Escape Markdown only for diagnostics (failed rows). Keep successful model output
    # unchanged. This preserves model formatting (including *, _, `, etc.) while
    # avoiding table breakage from diagnostics.
    _process_markdown_rows(rows, report_context.result_set.results)

    colalign = ["left"] + [
        "right" if is_numeric_field(field_name) else "left" for field_name in field_names[1:]
    ]

    markdown_table = tabulate(
        rows,
        headers=markdown_headers,
        tablefmt="pipe",  # Use 'pipe' format for proper GitHub alignment with colons
        colalign=colalign,
    )

    markdown_table = normalize_markdown_trailing_spaces(markdown_table)

    md: list[str] = []
    # Surround the table with markdownlint rule guards; the table can be wide and may
    # contain HTML breaks and model-generated emphasis styles
    md.append("<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->")
    md.append("")
    md.append(markdown_table)
    md.append("")
    md.append("<!-- markdownlint-enable MD013 MD033 MD034 MD037 MD049 -->")
    md.append("")
    return md


def _append_markdown_gallery_note(
    md: list[str],
    *,
    report_filename: Path,
    gallery_filename: Path | None,
    review_filename: Path | None = None,
    log_filename: Path | None = None,
) -> None:
    """Append relative links to companion artifacts when present."""
    artifact_lines: list[tuple[str, str]] = []
    if gallery_filename is not None:
        relative_gallery_path = _relative_markdown_artifact_path(
            report_filename=report_filename,
            artifact_filename=gallery_filename,
        )
        artifact_lines.append(
            (
                "Standalone output gallery",
                f"[{relative_gallery_path}]({relative_gallery_path.replace(' ', '%20')})",
            ),
        )
    if review_filename is not None:
        relative_review_path = _relative_markdown_artifact_path(
            report_filename=report_filename,
            artifact_filename=review_filename,
        )
        artifact_lines.append(
            (
                "Automated review digest",
                f"[{relative_review_path}]({relative_review_path.replace(' ', '%20')})",
            ),
        )
    if log_filename is not None:
        relative_log_path = _relative_markdown_artifact_path(
            report_filename=report_filename,
            artifact_filename=log_filename,
        )
        artifact_lines.append(
            (
                "Canonical run log",
                f"[{relative_log_path}]({relative_log_path.replace(' ', '%20')})",
            ),
        )

    if not artifact_lines:
        return

    md.append(_markdown_emphasis("Review artifacts:"))
    md.append("")
    for label, link in artifact_lines:
        _append_markdown_labeled_value(md, label=label, value=link, bullet=True)
    md.append("")


def generate_markdown_report(
    results: list[PerformanceResult],
    filename: Path,
    versions: LibraryVersionDict,
    prompt: str,
    total_runtime_seconds: float,
    report_context: ReportRenderContext | None = None,
    gallery_filename: Path | None = None,
    review_filename: Path | None = None,
    log_filename: Path | None = None,
) -> None:
    """Write a GitHub-friendly Markdown summary with aligned pipe table.

    Args:
        results: Run results to render.
        filename: Output Markdown file path.
        versions: Installed library versions shown in the report.
        prompt: Prompt used for the run.
        total_runtime_seconds: Total wall-clock runtime for the full run.
        report_context: Optional cached shared report context built in finalization.
        gallery_filename: Optional standalone gallery artifact path to link from results.md.
        review_filename: Optional automated review digest path to link from results.md.
        log_filename: Optional canonical log path to link from results.md.
    """
    if not results:
        log_warning_note("No results to generate Markdown report.")
        return

    if report_context is None:
        report_context = _build_report_render_context(results=results, prompt=prompt)

    headers, rows, _ = _materialize_prepared_table_data(report_context.table_data)

    if not headers or not rows:
        log_warning_note("No table data to generate Markdown report.")
        return

    issues_text: str = format_issues_summary_text(
        report_context.summary,
        report_context.stats,
    )

    gallery_relative_path = (
        _relative_markdown_artifact_path(
            report_filename=filename,
            artifact_filename=gallery_filename,
        )
        if gallery_filename is not None
        else None
    )

    # Build the complete markdown content
    md: list[str] = []
    md.append("# Model Performance Results")
    md.append("")
    md.append(f"_Generated on {local_now_str()}_")
    md.append("")
    md.extend(
        _format_action_snapshot_parts(
            results,
            report_context,
            html_output=False,
        ),
    )
    # Add issues summary before prompt
    if issues_text:
        md.append(issues_text)
    md.extend(
        _build_markdown_recommended_models(
            report_context,
            gallery_relative_path=gallery_relative_path,
        ),
    )
    md.extend(
        _build_markdown_quality_breakdown(
            report_context,
            gallery_relative_path=gallery_relative_path,
        ),
    )

    # Add failures-by-package section for actionable reporting
    failures_by_pkg = _format_failures_by_package_parts(
        results,
        html_output=False,
    )
    if failures_by_pkg:
        md.extend(failures_by_pkg)

    md.append(_markdown_emphasis("Prompt used:"))
    _append_markdown_wrapped_blockquote(md, prompt)
    md.extend(
        _wrap_markdown_text(
            "_Note:_ Results sorted: errors first, then by generation time (fastest to slowest).",
        ),
    )
    md.append("")
    md.append(
        _markdown_emphasis("Overall runtime:") + f" {format_overall_runtime(total_runtime_seconds)}"
    )
    md.append("")

    # Generate table section
    table_md: list[str] = _generate_markdown_table_section(report_context)
    md.extend(table_md)

    _append_markdown_gallery_note(
        md,
        report_filename=filename,
        gallery_filename=gallery_filename,
        review_filename=review_filename,
        log_filename=log_filename,
    )

    md.append("---")

    # Add system/hardware information if available
    if report_context.system_info:
        md.append("")
        md.append("## System/Hardware Information")
        md.append("")
        for name, value in _collect_report_component_rows(
            versions={},
            system_info=report_context.system_info,
        ):
            _append_markdown_labeled_value(md, label=name, value=value, bullet=True)
        md.append("")

    md.append("## Library Versions")
    md.append("")
    for name, value in _collect_report_component_rows(versions=versions, system_info={}):
        md.append(f"- `{name}`: `{value}`")
    md.append("")
    md.append(f"_Report generated on: {local_now_str()}_")

    _write_markdown_artifact(filename, md, artifact_name="Markdown report")


def generate_markdown_gallery_report(
    results: list[PerformanceResult],
    filename: Path,
    prompt: str,
    metadata: MetadataDict | None = None,
    report_context: ReportRenderContext | None = None,
) -> None:
    """Write a review-focused markdown artifact with metadata, prompt, and full outputs."""
    if not results:
        log_warning_note("No results to generate Markdown gallery report.")
        return

    if report_context is None:
        report_context = _build_report_render_context(results=results, prompt=prompt)

    md: list[str] = []
    md.append("# Model Output Gallery")
    md.append("")
    md.append(f"_Generated on {local_now_str()}_")
    md.append("")
    md.extend(
        _wrap_markdown_text(
            "A review-friendly artifact with image metadata, the source prompt, and full "
            "generated output for each model.",
        ),
    )
    md.append("")
    md.append("_Action Snapshot: see [results.md](results.md) for the full summary._")
    md.append("")
    md.extend(_format_review_priorities_parts(report_context, html_output=False))
    md.extend(_format_failures_by_package_parts(results, html_output=False))
    _append_markdown_image_metadata_section(md, metadata)
    md.append("## Prompt")
    _append_markdown_wrapped_blockquote(md, prompt)
    md.extend(_build_markdown_gallery_navigation(report_context))
    md.extend(_generate_model_gallery_section(report_context))

    _write_markdown_artifact(filename, md, artifact_name="Markdown gallery report")


def _group_review_results(
    results: Sequence[PerformanceResult],
    *,
    key_name: Literal["owner", "user_bucket"],
) -> dict[str, list[PerformanceResult]]:
    """Group results by one field from the canonical review payload."""
    grouped: dict[str, list[PerformanceResult]] = {}
    for result in results:
        review = _build_jsonl_review_record(result)
        if review is None:
            continue
        key = review["owner"] if key_name == "owner" else review["user_bucket"]
        grouped.setdefault(key, []).append(result)
    return grouped


def _is_review_escalation(result: PerformanceResult) -> bool:
    """Return True for review rows that need maintainer attention."""
    review = _build_jsonl_review_record(result)
    if review is None:
        return False
    if review["verdict"] in {
        "runtime_failure",
        "harness",
        "cutoff",
        "cutoff_degraded",
        "context_budget",
    }:
        return True

    analysis = _quality_analysis_for_result(result)
    if analysis is None:
        return result.error_package is not None
    if analysis.has_harness_issue or analysis.has_reasoning_leak or analysis.has_degeneration:
        return True
    if analysis.formatting_issues or analysis.has_excessive_bullets or analysis.has_context_echo:
        return True
    return any(
        "long_context" in evidence or evidence in {"reasoning_leak", "formatting"}
        for evidence in review["evidence"]
    )


def _group_review_escalations_by_owner(
    results: Sequence[PerformanceResult],
) -> dict[str, list[PerformanceResult]]:
    """Group only actionable review escalations by likely owner."""
    return _group_review_results(
        [result for result in results if _is_review_escalation(result)],
        key_name="owner",
    )


def _append_review_owner_queue(
    md: list[str],
    owner_groups: Mapping[str, Sequence[PerformanceResult]],
) -> None:
    """Append the maintainer ownership section for the review digest."""
    md.extend(
        [
            "## Maintainer Queue",
            "",
            "Owner-grouped escalations only; clean recommended models stay in user buckets.",
            "",
        ]
    )
    if not owner_groups:
        md.extend(["- No owner-specific escalations were produced.", ""])
        return

    for owner in sorted(owner_groups):
        md.extend([f"### `{owner}`", ""])
        rows: list[tuple[str, ...]] = []
        for result in owner_groups[owner]:
            review = _build_jsonl_review_record(result)
            if review is None:
                continue
            analysis = _quality_analysis_for_result(result)
            rows.append(
                (
                    f"`{MARKDOWN_ESCAPER.escape(result.model_name)}`",
                    f"`{MARKDOWN_ESCAPER.escape(review['verdict'])}`",
                    MARKDOWN_ESCAPER.escape(_review_focus_text(review, analysis)),
                    MARKDOWN_ESCAPER.escape(_review_next_action_text(review)),
                )
            )
        if rows:
            md.extend(
                tabulate(
                    rows,
                    headers=["Model", "Verdict", "Evidence", "Next Action"],
                    tablefmt="github",
                ).splitlines()
            )
            md.append("")
        else:
            md.extend(["- No owner-specific escalations were produced.", ""])


def _append_review_user_buckets(
    md: list[str],
    bucket_groups: Mapping[str, Sequence[PerformanceResult]],
) -> None:
    """Append user-facing recommendation buckets for the review digest."""
    md.extend(
        [
            "## User Buckets",
            "",
            "User-first summary grouped by recommendation bucket.",
            "",
        ]
    )
    for bucket in ("recommended", "caveat", "needs_triage", "avoid"):
        md.extend([f"### `{bucket}`", ""])
        bucket_results = bucket_groups.get(bucket, [])
        if not bucket_results:
            if bucket == "recommended":
                md.extend(
                    [
                        "- None — no models produced clean output meeting all quality thresholds for this prompt.",
                        "",
                    ]
                )
            else:
                md.extend(["- None.", ""])
            continue
        rows: list[tuple[str, ...]] = []
        for result in bucket_results:
            review = _build_jsonl_review_record(result)
            if review is None:
                continue
            analysis = _quality_analysis_for_result(result)
            rows.append(
                (
                    f"`{MARKDOWN_ESCAPER.escape(result.model_name)}`",
                    f"`{MARKDOWN_ESCAPER.escape(review['verdict'])}`",
                    MARKDOWN_ESCAPER.escape(_review_hint_text(review, analysis)),
                    MARKDOWN_ESCAPER.escape(_review_focus_text(review, analysis)),
                )
            )
        md.extend(
            tabulate(
                rows,
                headers=["Model", "Verdict", "Hint Handling", "Key Evidence"],
                tablefmt="github",
            ).splitlines()
        )
        md.append("")


def _append_review_model_verdicts(
    md: list[str],
    results: Sequence[PerformanceResult],
) -> None:
    """Append detailed per-model canonical review rows for the digest."""
    md.extend(["## Model Verdicts", ""])
    for result in results:
        review = _build_jsonl_review_record(result)
        if review is None:
            continue
        md.extend([f"### `{result.model_name}`", ""])
        _append_markdown_review_block(md, res=result)
        md.append("")


def _append_review_issue_queue(
    md: list[str],
    *,
    report_context: ReportRenderContext,
    report_filename: Path,
) -> None:
    """Append a compact issue-queue pointer to the review digest."""
    snapshot = _build_diagnostics_snapshot(
        results=list(report_context.result_set.results),
        prompt=report_context.prompt_context,
        preflight_issues=report_context.preflight_issues,
    )
    clusters = _build_issue_clusters(snapshot)
    if not clusters:
        return

    output_root = (
        report_filename.parent.parent
        if report_filename.parent.name == "reports"
        else report_filename.parent
    )
    issue_index = output_root / "issues" / "index.md"
    issue_index_rel = _relative_markdown_artifact_path(
        report_filename=report_filename,
        artifact_filename=issue_index,
    ).replace(" ", "%20")
    md.extend(
        [
            "## Maintainer Escalations",
            "",
            f"Focused upstream issue drafts are queued in [issues/index.md]({issue_index_rel}).",
            "",
        ]
    )

    def _review_issue_link(cluster: IssueCluster) -> str:
        issue_path = output_root / "issues" / cluster.issue_filename
        issue_rel = _relative_markdown_artifact_path(
            report_filename=report_filename,
            artifact_filename=issue_path,
        ).replace(" ", "%20")
        return f"[issue draft]({issue_rel})"

    md.extend(
        _render_issue_queue_table(
            clusters,
            escape_text=MARKDOWN_ESCAPER.escape,
            issue_link_for_cluster=_review_issue_link,
            evidence_link_for_cluster=lambda _cluster: "-",
        )
    )
    md.append("")


def generate_review_report(
    results: list[PerformanceResult],
    filename: Path,
    *,
    prompt: str,
    report_context: ReportRenderContext | None = None,
    log_filename: Path | None = None,
    gallery_filename: Path | None = None,
) -> None:
    """Write a short Markdown digest of automated verdicts and action buckets."""
    if not results:
        log_warning_note("No results to generate review report.")
        return

    if report_context is None:
        report_context = _build_report_render_context(results=results, prompt=prompt)

    sorted_results = list(report_context.result_set.results)
    bucket_groups = _group_review_results(sorted_results, key_name="user_bucket")

    md: list[str] = [
        "# Automated Review Digest",
        "",
        f"_Generated on {local_now_str()}_",
        "",
        (
            "Trusted-hint review uses only prompt title/description/keyword hints for utility "
            "comparison. Capture metadata, GPS, timestamps, source labels, and location labels "
            "are treated as nonvisual metadata and are not required visual evidence."
        ),
        "",
    ]

    if log_filename is not None or gallery_filename is not None:
        _append_markdown_gallery_note(
            md,
            report_filename=filename,
            gallery_filename=gallery_filename,
            log_filename=log_filename,
        )

    md.extend(_format_review_priorities_parts(report_context, html_output=False))
    _append_review_user_buckets(md, bucket_groups)
    _append_review_issue_queue(md, report_context=report_context, report_filename=filename)
    _append_review_model_verdicts(md, sorted_results)

    _write_markdown_artifact(filename, md, artifact_name="Review report")


def generate_tsv_report(
    results: list[PerformanceResult],
    filename: Path,
    report_context: ReportRenderContext | None = None,
) -> None:
    """Write a TSV (tab-separated values) file of the core results table.

    A ``# generated_at: <timestamp>`` comment line is written first so
    downstream consumers know when the data was produced.  For failed models
    two extra columns (``error_type``, ``error_package``) are appended to
    aid programmatic triage.

    Args:
        results: List of PerformanceResult objects.
        filename: Path where the TSV file will be written.
        report_context: Optional cached shared report context built in finalization.
    """
    if not results:
        log_warning_note("No results to generate TSV report.")
        return

    if report_context is None:
        result_set = ResultSet(results)
        table_data = _build_prepared_table_data(result_set=result_set)
        headers, rows, _ = _materialize_prepared_table_data(table_data)
        sorted_results = list(result_set.results)
    else:
        headers, rows, _ = _materialize_prepared_table_data(report_context.table_data)
        sorted_results = list(report_context.result_set.results)

    if not headers or not rows:
        log_warning_note("No table data to generate TSV report.")
        return

    def escape_tsv_value(value: str) -> str:
        r"""Escape a value for TSV format.

        Replaces tabs with spaces and newlines with literal \n to preserve
        the tabular structure. This ensures each record stays on one line.
        """
        # Replace actual tabs with spaces
        value = value.replace("\t", "    ")
        # Replace newlines with escaped newline sequence
        value = value.replace("\n", "\\n")
        # Remove carriage returns
        return value.replace("\r", "")

    # Clean headers: remove HTML tags and escape for TSV
    clean_headers: list[str] = []
    for header in headers:
        # Remove <br> tags and other HTML
        clean_header = header.replace("<br>", " ").strip()
        clean_header = re.sub(r"<[^>]+>", "", clean_header)
        clean_headers.append(escape_tsv_value(clean_header))

    clean_headers.extend(["error_type", "error_package"])

    clean_rows: list[list[str]] = []
    for row, res in zip(rows, sorted_results, strict=False):
        clean_row = [escape_tsv_value(cell) for cell in row]
        clean_row.append(escape_tsv_value(res.error_type or ""))
        clean_row.append(escape_tsv_value(res.error_package or ""))
        clean_rows.append(
            [
                c if len(c) <= MAX_TSV_CELL_CHARS else c[: MAX_TSV_CELL_CHARS - 3].rstrip() + "..."
                for c in clean_row
            ]
        )

    tsv_content = tabulate(
        clean_rows,
        headers=clean_headers,
        tablefmt="tsv",
    )

    try:
        with filename.open("w", encoding="utf-8") as f:
            f.write(f"# generated_at: {local_now_str()}\n")
            f.write(tsv_content)
            if not tsv_content.endswith("\n"):
                f.write("\n")
        # Logging handled in finalize_execution
    except OSError:
        logger.exception("Failed to write TSV report to %s", filename)


def _wrap_bare_urls(text: str) -> str:
    """Wrap bare URLs in angle brackets to satisfy markdownlint MD034.

    URLs are wrapped as <URL> which tells markdown processors they are autolinks.
    This prevents MD034 "Bare URL used" warnings.
    """
    # Match http:// or https:// URLs that aren't already in angle brackets or markdown links
    # Pattern: not preceded by [ or <, then URL, not followed by ] or >
    url_pattern = re.compile(
        r"(?<![\[<])"  # Negative lookbehind: not [ or <
        r"(https?://[^\s\)>\]]+)"  # URL (not followed by space, ), >, or ])
        r"(?![\]>])",  # Negative lookahead: not ] or >
    )
    return url_pattern.sub(r"<\1>", text)


def normalize_markdown_trailing_spaces(md_text: str) -> str:
    """Normalize trailing spaces on each line in Markdown text.

    Rules:
    - Keep exactly FORMATTING.markdown_hard_break_spaces trailing spaces (Markdown hard line break).
    - Strip any other trailing horizontal whitespace or invisible format characters,
      including non-breaking spaces and trailing BOM/zero-width markers, to avoid
      accidental MD009-style endings.
    """
    out_lines: list[str] = []
    for ln in md_text.splitlines():
        stripped = ln.rstrip(" \t\u00a0\u200b\u200c\u200d\u2060\ufeff")
        trailing = ln[len(stripped) :]
        if not trailing:
            out_lines.append(ln)
            continue
        if trailing == (" " * FORMATTING.markdown_hard_break_spaces):
            out_lines.append(ln)
        else:
            out_lines.append(stripped)
    return "\n".join(out_lines)


@lru_cache(maxsize=1)
def get_system_info() -> tuple[str, str | None]:
    """Get system architecture and GPU information.

    Cached since system info doesn't change during execution.

    Returns:
        Tuple of (architecture_string, optional_gpu_info)
    """
    arch: str = platform.machine()
    gpu_info: str | None = None
    if platform.system() == "Darwin":
        first_display = _first_system_profiler_entry(get_device_info(), "SPDisplaysDataType")
        if first_display is not None:
            gpu_info = _mapping_first_text_value(first_display, "sppci_model", "_name")
    return arch, gpu_info


def _get_macos_toolchain_info() -> dict[str, str]:
    """Get macOS developer toolchain info (Xcode, SDK, CLTools).

    Returns dict with available toolchain version info. Safe to call on any OS
    (returns empty dict on non-macOS).
    """
    info: dict[str, str] = {}
    if platform.system() != "Darwin":
        return info

    # Get SDK version (useful for Metal/framework compatibility)
    try:
        sdk_result = subprocess.run(
            ["/usr/bin/xcrun", "--show-sdk-version"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if sdk_result.returncode == 0 and sdk_result.stdout.strip():
            info["SDK Version"] = sdk_result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    # Get Xcode version (important for Metal shader compilation)
    try:
        xcode_result = subprocess.run(
            ["/usr/bin/xcodebuild", "-version"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if xcode_result.returncode == 0 and xcode_result.stdout.strip():
            # Output format: "Xcode 15.4\nBuild version 15F31d"
            lines = xcode_result.stdout.strip().split("\n")
            if lines:
                info["Xcode Version"] = lines[0].replace("Xcode ", "").strip()
                if len(lines) > 1 and "Build version" in lines[1]:
                    info["Xcode Build"] = lines[1].replace("Build version ", "").strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    # Get Metal SDK path (useful for debugging Metal issues)
    try:
        sdk_path_result = subprocess.run(
            ["/usr/bin/xcrun", "--show-sdk-path"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if sdk_path_result.returncode == 0 and sdk_path_result.stdout.strip():
            info["Metal SDK"] = Path(sdk_path_result.stdout.strip()).name
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    # Get Command Line Tools version if Xcode not available
    if "Xcode Version" not in info:
        try:
            clt_result = subprocess.run(
                ["/usr/bin/pkgutil", "--pkg-info", "com.apple.pkg.CLTools_Executables"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if clt_result.returncode == 0:
                for line in clt_result.stdout.split("\n"):
                    if line.startswith("version:"):
                        info["CLTools Version"] = line.split(":", 1)[1].strip()
                        break
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

    return info


def _get_apple_silicon_info() -> dict[str, str]:
    """Get Apple Silicon GPU info from system_profiler.

    Returns dict with GPU cores and Metal support info. Safe to call on any
    architecture (returns empty dict on non-arm64).
    """
    info: dict[str, str] = {}
    if platform.machine() != "arm64":
        return info

    first = _first_system_profiler_entry(get_device_info(), "SPDisplaysDataType")
    if first is None:
        return info

    gpu_cores = first.get("sppci_cores")
    if gpu_cores is not None:
        info["GPU Cores"] = str(gpu_cores)

    metal_family = _mapping_first_text_value(first, "spdisplays_mtlgpufamilysupport")
    if metal_family is not None:
        info["Metal Support"] = metal_family.replace("spdisplays_metal", "Metal ")

    return info


def get_system_characteristics() -> dict[str, str]:
    """Gather system/hardware characteristics for inclusion in reports.

    Returns a dict with human-readable hardware info (OS, chip, RAM, cores, etc).
    Safe to call even if psutil or system_profiler unavailable.
    """
    info: dict[str, str] = {}

    try:
        # Basic platform info
        info["OS"] = f"{platform.system()} {platform.release()}"
        if platform.system() == "Darwin":
            info["macOS Version"] = platform.mac_ver()[0]
            info.update(_get_macos_toolchain_info())

        info["Python Version"] = sys.version.split()[0]
        info["Architecture"] = platform.machine()

        # Get GPU info
        _, gpu_name = get_system_info()
        if gpu_name:
            info["GPU/Chip"] = gpu_name

        # Get detailed Apple Silicon info
        info.update(_get_apple_silicon_info())

        # Get memory and CPU info if psutil available
        if psutil is not None:
            ram_gb = psutil.virtual_memory().total / (1024**3)
            info["RAM"] = f"{ram_gb:.1f} GB"

            physical_cores = psutil.cpu_count(logical=False)
            if physical_cores:
                info["CPU Cores (Physical)"] = str(physical_cores)

            logical_cores = psutil.cpu_count(logical=True)
            if logical_cores:
                info["CPU Cores (Logical)"] = str(logical_cores)

    except (OSError, RuntimeError, ValueError) as err:
        logger.debug("Error gathering system characteristics: %s", err)

    return info


def collect_runtime_fingerprint() -> dict[str, RuntimeProbeResult]:
    """Probe runtime capabilities once per run and record per-probe status.

    Each probe records ``ok``, ``unavailable``, ``errored``, or ``timed_out``
    (guardrail G2: never silently omit a probe).
    """
    probes: dict[str, RuntimeProbeResult] = {}

    # Metal GPU availability
    try:
        if mx is not None:
            mx.metal.is_available()
            probes["metal_gpu"] = RuntimeProbeResult(status="ok")
        else:
            probes["metal_gpu"] = RuntimeProbeResult(
                status="unavailable", detail="mlx not imported"
            )
    except Exception as exc:  # noqa: BLE001
        probes["metal_gpu"] = RuntimeProbeResult(status="errored", detail=str(exc)[:120])

    # MLX framework version
    try:
        if mx is not None:
            probes["mlx_framework"] = RuntimeProbeResult(
                status="ok", detail=getattr(mx, "__version__", "unknown")
            )
        else:
            probes["mlx_framework"] = RuntimeProbeResult(status="unavailable")
    except Exception as exc:  # noqa: BLE001
        probes["mlx_framework"] = RuntimeProbeResult(status="errored", detail=str(exc)[:120])

    # mlx-vlm importability
    try:
        if "mlx-vlm" not in MISSING_DEPENDENCIES:
            probes["mlx_vlm"] = RuntimeProbeResult(status="ok")
        else:
            probes["mlx_vlm"] = RuntimeProbeResult(
                status="unavailable", detail=MISSING_DEPENDENCIES.get("mlx-vlm", "not imported")
            )
    except Exception as exc:  # noqa: BLE001
        probes["mlx_vlm"] = RuntimeProbeResult(status="errored", detail=str(exc)[:120])

    # GPU memory query
    try:
        if mx is not None and hasattr(mx, "get_active_memory"):
            active = mx.get_active_memory()
            probes["gpu_memory"] = RuntimeProbeResult(
                status="ok", detail=f"active={active / (1024**3):.2f}GB"
            )
        else:
            probes["gpu_memory"] = RuntimeProbeResult(status="unavailable")
    except Exception as exc:  # noqa: BLE001
        probes["gpu_memory"] = RuntimeProbeResult(status="errored", detail=str(exc)[:120])

    return probes


# --- Model Processing Core ---
def validate_inputs(
    image_path: PathLike,
    temperature: float = 0.0,
) -> None:
    """Validate input paths and parameters with comprehensive checks."""
    img_path: Path = Path(image_path)

    # Early validation for performance
    if not img_path.exists():
        msg = f"Image not found: {img_path}"
        raise FileNotFoundError(msg)
    if not img_path.is_file():
        msg = f"Not a file: {img_path}"
        raise ValueError(msg)

    # Check file extension (case-insensitive for robustness)
    if img_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        msg = f"Unsupported image format: {img_path.suffix}"
        raise ValueError(msg)

    validate_temperature(temp=temperature)


def validate_temperature(*, temp: float) -> None:
    """Validate temperature parameter is within acceptable range."""
    if temp < 0.0:
        msg: str = f"Temperature must be non-negative, got {temp}"
        raise ValueError(msg)
    if temp > MAX_REASONABLE_TEMPERATURE:
        logger.warning(
            "Temperature %.2f is unusually high (>%.1f). Output may be very random.",
            temp,
            MAX_REASONABLE_TEMPERATURE,
        )


def validate_sampling_params(
    *,  # Force all parameters to be keyword-only for clarity
    top_p: float,
    min_p: float,
    top_k: int,
    repetition_penalty: float | None,
) -> None:
    """Validate sampling parameters are within acceptable ranges."""
    if not 0.0 <= top_p <= 1.0:
        msg = f"top_p must be between 0.0 and 1.0, got {top_p}"
        raise ValueError(msg)

    if not 0.0 <= min_p <= 1.0:
        msg = f"min_p must be between 0.0 and 1.0, got {min_p}"
        raise ValueError(msg)

    if top_k < 0:
        msg = f"top_k must be >= 0, got {top_k}"
        raise ValueError(msg)

    if repetition_penalty is not None and repetition_penalty < 1.0:
        msg = f"repetition_penalty must be >= 1.0 if specified, got {repetition_penalty}"
        raise ValueError(msg)


def validate_kv_params(
    *,  # Force all parameters to be keyword-only for clarity
    max_kv_size: int | None,
    kv_bits: float | None,
    kv_quant_scheme: Literal["uniform", "turboquant"] = DEFAULT_KV_QUANT_SCHEME,
) -> None:
    """Validate KV cache parameters are within acceptable ranges."""
    if max_kv_size is not None and max_kv_size <= 0:
        msg = f"max_kv_size must be > 0 if specified, got {max_kv_size}"
        raise ValueError(msg)

    if kv_bits is None:
        return

    rounded_half = round(kv_bits * 2) / 2
    if kv_bits < 1:
        msg = f"kv_bits must be >= 1 if specified, got {kv_bits:g}"
        raise ValueError(msg)

    if not math.isclose(kv_bits, rounded_half, abs_tol=1e-6):
        msg = f"kv_bits must be an integer or .5 increment if specified, got {kv_bits:g}"
        raise ValueError(msg)

    is_integer_bits = math.isclose(kv_bits, round(kv_bits), abs_tol=1e-6)
    if kv_quant_scheme == "uniform" and is_integer_bits:
        int_bits = round(kv_bits)
        if int_bits not in UNIFORM_KV_BITS:
            msg = (
                f"uniform kv_bits must be one of 2, 3, 4, 5, 6, or 8 if specified, got {kv_bits:g}"
            )
            raise ValueError(msg)


_RESERVED_PROCESSOR_KWARG_KEYS: Final[frozenset[str]] = frozenset(
    {
        "audio",
        "eos_tokens",
        "image",
        "kv_bits",
        "kv_group_size",
        "kv_quant_scheme",
        "max_kv_size",
        "max_tokens",
        "min_p",
        "prefill_step_size",
        "processor",
        "prompt",
        "quantized_kv_start",
        "repetition_context_size",
        "repetition_penalty",
        "resize_shape",
        "skip_special_tokens",
        "enable_thinking",
        "thinking_budget",
        "thinking_end_token",
        "thinking_start_token",
        "temperature",
        "top_k",
        "top_p",
        "verbose",
    },
)


def _parse_processor_kwargs_arg(value: str) -> dict[str, JsonLike]:
    """Parse ``--processor-kwargs`` as a JSON object."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        msg = f"processor_kwargs must be valid JSON: {exc.msg}"
        raise argparse.ArgumentTypeError(msg) from exc

    if not isinstance(parsed, dict):
        msg = "processor_kwargs must be a JSON object"
        raise argparse.ArgumentTypeError(msg)
    if not all(isinstance(key, str) for key in parsed):
        msg = "processor_kwargs keys must all be strings"
        raise argparse.ArgumentTypeError(msg)
    return cast("dict[str, JsonLike]", parsed)


def _normalize_resize_shape(raw_shape: Sequence[int] | None) -> tuple[int, int] | None:
    """Normalize CLI resize shape values to a ``(height, width)`` tuple."""
    if raw_shape is None:
        return None
    if len(raw_shape) not in (1, 2):
        msg = f"resize_shape must contain 1 or 2 integers, got {len(raw_shape)}"
        raise ValueError(msg)
    if any(size <= 0 for size in raw_shape):
        msg = f"resize_shape values must be > 0, got {list(raw_shape)}"
        raise ValueError(msg)
    if len(raw_shape) == 1:
        return (raw_shape[0], raw_shape[0])
    return (raw_shape[0], raw_shape[1])


def _decode_cli_eos_tokens(raw_tokens: Sequence[str] | None) -> tuple[str, ...] | None:
    """Decode CLI EOS tokens, supporting escaped characters like upstream mlx-vlm."""
    if raw_tokens is None:
        return None

    decoded_tokens: list[str] = []
    for token in raw_tokens:
        try:
            decoded_tokens.append(codecs.decode(token, "unicode_escape"))
        except (UnicodeDecodeError, UnicodeError):
            decoded_tokens.append(token)
    return tuple(decoded_tokens)


def _validate_processor_kwargs(
    processor_kwargs: Mapping[str, JsonLike] | None,
) -> dict[str, JsonLike] | None:
    """Reject processor kwargs that would collide with dedicated CLI flags."""
    if processor_kwargs is None:
        return None

    overlap = sorted(set(processor_kwargs).intersection(_RESERVED_PROCESSOR_KWARG_KEYS))
    if overlap:
        overlap_str = ", ".join(overlap)
        msg = f"processor_kwargs cannot override dedicated CLI arguments: {overlap_str}"
        raise ValueError(msg)
    return dict(processor_kwargs)


def _validate_thinking_params(args: argparse.Namespace) -> None:
    """Validate opt-in thinking-mode arguments."""
    enable_thinking = bool(getattr(args, "enable_thinking", False))
    thinking_budget = getattr(args, "thinking_budget", None)
    thinking_start_token = getattr(args, "thinking_start_token", None)
    thinking_end_token = getattr(args, "thinking_end_token", DEFAULT_THINKING_END_MARKER)
    thinking_requested = bool(
        enable_thinking or thinking_budget is not None or thinking_start_token is not None,
    )

    if thinking_budget is not None and thinking_budget <= 0:
        msg = f"thinking_budget must be > 0 if specified, got {thinking_budget}"
        raise ValueError(msg)

    if thinking_requested and not enable_thinking:
        msg = "thinking_budget and thinking token flags require --enable-thinking"
        raise ValueError(msg)

    if enable_thinking and not thinking_end_token:
        msg = "thinking_end_token must be non-empty when thinking mode is enabled"
        raise ValueError(msg)


def validate_cli_arguments(args: argparse.Namespace) -> None:
    """Validate all CLI arguments before processing begins.

    This performs early validation to fail fast on invalid inputs
    before any expensive operations (model loading, image processing, etc.).
    """
    # Validate temperature
    validate_temperature(temp=args.temperature)

    # Validate max_tokens
    if args.max_tokens <= 0:
        msg = f"max_tokens must be > 0, got {args.max_tokens}"
        raise ValueError(msg)

    # Validate sampling parameters
    validate_sampling_params(
        top_p=args.top_p,
        min_p=args.min_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )

    # Validate KV cache parameters
    validate_kv_params(
        max_kv_size=args.max_kv_size,
        kv_bits=args.kv_bits,
        kv_quant_scheme=args.kv_quant_scheme,
    )

    args.resize_shape = _normalize_resize_shape(getattr(args, "resize_shape", None))
    args.eos_tokens = _decode_cli_eos_tokens(getattr(args, "eos_tokens", None))
    args.processor_kwargs = _validate_processor_kwargs(
        getattr(args, "processor_kwargs", None),
    )
    _validate_thinking_params(args)

    if bool(getattr(args, "detailed_metrics", False)) and not bool(getattr(args, "verbose", False)):
        logger.warning(
            "--detailed-metrics has no effect unless --verbose is also set; "
            "continuing with compact metrics output.",
        )


def _build_generate_extra_kwargs(params: ProcessImageParams) -> GenerateExtraKwargs:
    """Collect optional generate kwargs for benchmark runs."""
    extra_kwargs: GenerateExtraKwargs = {}
    if params.min_p > 0.0:
        extra_kwargs["min_p"] = params.min_p
    if params.top_k > 0:
        extra_kwargs["top_k"] = params.top_k
    if params.prefill_step_size is not None:
        extra_kwargs["prefill_step_size"] = params.prefill_step_size
    if params.resize_shape is not None:
        extra_kwargs["resize_shape"] = params.resize_shape
    if params.eos_tokens is not None:
        extra_kwargs["eos_tokens"] = list(params.eos_tokens)
    if params.skip_special_tokens:
        extra_kwargs["skip_special_tokens"] = True
    if params.enable_thinking:
        extra_kwargs["enable_thinking"] = True
        extra_kwargs["thinking_end_token"] = params.thinking_end_token
        if params.thinking_budget is not None:
            extra_kwargs["thinking_budget"] = params.thinking_budget
        if params.thinking_start_token is not None:
            extra_kwargs["thinking_start_token"] = params.thinking_start_token
    return extra_kwargs


_IMAGE_PLACEHOLDER_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"<\|?image(?:_\d+)?\|?>", re.IGNORECASE),
    re.compile(r"<start_of_image>", re.IGNORECASE),
    re.compile(r"<image_soft_token>", re.IGNORECASE),
)
_MAX_PROMPT_DIAGNOSTIC_ITEMS: Final[int] = 32
_RENDERED_PROMPT_PREVIEW_CHARS: Final[int] = 800


def _prompt_diag_json_value(value: object) -> JsonLike:
    """Convert prompt-diagnostic values into bounded JSON-compatible data."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _prompt_diag_json_value(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_prompt_diag_json_value(item) for item in value]
    return str(value)


def _qualified_class_name(value: object | None) -> str | None:
    """Return a stable module-qualified class name for diagnostics."""
    if value is None:
        return None
    cls = value.__class__
    module = getattr(cls, "__module__", "")
    qualname = getattr(cls, "__qualname__", cls.__name__)
    return f"{module}.{qualname}" if module else qualname


def _count_image_placeholders(rendered_prompt: str) -> int:
    """Count common multimodal image placeholder spellings in rendered templates."""
    return sum(len(pattern.findall(rendered_prompt)) for pattern in _IMAGE_PLACEHOLDER_PATTERNS)


def _bounded_json_sequence(value: object) -> tuple[JsonLike, ...]:
    """Return a bounded tuple of JSON-compatible diagnostic values."""
    if value is None or isinstance(value, str | bytes | bytearray):
        return ()
    if not isinstance(value, Sequence):
        return ()
    return tuple(_prompt_diag_json_value(item) for item in value[:_MAX_PROMPT_DIAGNOSTIC_ITEMS])


def _bounded_string_sequence(value: object) -> tuple[str, ...]:
    """Return a bounded tuple of stringified diagnostic values."""
    if value is None or isinstance(value, str | bytes | bytearray):
        return ()
    if not isinstance(value, Sequence):
        return ()
    return tuple(str(item) for item in value[:_MAX_PROMPT_DIAGNOSTIC_ITEMS])


def _generation_kwargs_for_prompt_diagnostics(
    *,
    params: ProcessImageParams,
    extra_kwargs: Mapping[str, object],
    processor_passthrough_kwargs: Mapping[str, object],
) -> dict[str, JsonLike]:
    """Return the generation kwargs forwarded to mlx-vlm in JSON-safe form."""
    kwargs: dict[str, object] = {
        "temperature": params.temperature,
        "top_p": params.top_p,
        "repetition_penalty": params.repetition_penalty,
        "repetition_context_size": params.repetition_context_size,
        "max_kv_size": params.max_kv_size,
        "kv_bits": params.kv_bits,
        "kv_quant_scheme": params.kv_quant_scheme,
        "kv_group_size": params.kv_group_size,
        "quantized_kv_start": params.quantized_kv_start,
        "max_tokens": params.max_tokens,
    }
    kwargs.update(extra_kwargs)
    if processor_passthrough_kwargs:
        kwargs["processor_kwargs"] = dict(processor_passthrough_kwargs)
    return {key: _prompt_diag_json_value(value) for key, value in sorted(kwargs.items())}


def _build_prompt_diagnostics(
    *,
    params: ProcessImageParams,
    processor: ProcessorMixin,
    config: PreTrainedConfig | Mapping[str, object] | None,
    formatted_prompt: str,
    extra_kwargs: Mapping[str, object],
    processor_passthrough_kwargs: Mapping[str, object],
) -> PromptDiagnostics:
    """Collect bounded prompt/template diagnostics for JSONL and repro bundles."""
    tokenizer = _extract_processor_tokenizer(processor)
    model_type_raw = _get_config_value(config, "model_type")
    model_type = str(model_type_raw) if model_type_raw is not None else None
    eos_token = getattr(tokenizer, "eos_token", None)
    return PromptDiagnostics(
        model_type=model_type,
        processor_class=_qualified_class_name(processor),
        tokenizer_class=_qualified_class_name(tokenizer),
        rendered_prompt_hash_sha256=_sha256_text(formatted_prompt),
        rendered_prompt_preview=_build_prompt_preview(
            formatted_prompt,
            max_chars=_RENDERED_PROMPT_PREVIEW_CHARS,
        ),
        rendered_prompt_chars=len(formatted_prompt),
        image_placeholder_count=_count_image_placeholders(formatted_prompt),
        eos_token_id=_prompt_diag_json_value(getattr(tokenizer, "eos_token_id", None)),
        eos_token=str(eos_token) if eos_token is not None else None,
        special_token_ids=_bounded_json_sequence(getattr(tokenizer, "all_special_ids", None)),
        special_tokens=_bounded_string_sequence(getattr(tokenizer, "all_special_tokens", None)),
        generate_kwargs=_generation_kwargs_for_prompt_diagnostics(
            params=params,
            extra_kwargs=extra_kwargs,
            processor_passthrough_kwargs=processor_passthrough_kwargs,
        ),
    )


def _prompt_diagnostics_to_json(diagnostics: PromptDiagnostics | None) -> dict[str, JsonLike]:
    """Serialize prompt diagnostics as optional JSONL/repro metadata."""
    if diagnostics is None:
        return {}
    payload: dict[str, JsonLike] = {}
    for key in (
        "model_type",
        "processor_class",
        "tokenizer_class",
        "rendered_prompt_hash_sha256",
        "rendered_prompt_preview",
        "rendered_prompt_chars",
        "image_placeholder_count",
        "eos_token_id",
        "eos_token",
    ):
        value = getattr(diagnostics, key)
        if value is not None:
            payload[key] = _prompt_diag_json_value(value)
    if diagnostics.special_token_ids:
        payload["special_token_ids"] = list(diagnostics.special_token_ids)
    if diagnostics.special_tokens:
        payload["special_tokens"] = [
            _prompt_diag_json_value(item) for item in diagnostics.special_tokens
        ]
    if diagnostics.generate_kwargs:
        payload["generate_kwargs"] = dict(diagnostics.generate_kwargs)
    return payload


def validate_image_accessible(*, image_path: str | Path) -> None:
    """Validate image file is accessible and supported.

    Uses mlx_vlm's load_image() which supports both local file paths and URLs.
    This enables --image https://... usage following mlx-vlm best practices.
    """
    try:
        with TimeoutManager(seconds=IMAGE_OPEN_TIMEOUT):
            # load_image() from mlx_vlm.utils handles both file paths and URLs
            # Returns PIL.Image.Image, verifying the image is accessible and valid
            # Convert Path to str since load_image expects str
            _ = load_image(str(image_path))
    except RuntimeError as err:
        if str(err) != ERROR_MLX_VLM_MISSING:
            msg = f"Error accessing image {image_path}: {err}"
            raise OSError(msg) from err

        # mlx-vlm is unavailable: use Pillow-only validation for local files so
        # direct function callers and unit tests can still validate image inputs.
        try:
            with TimeoutManager(seconds=IMAGE_OPEN_TIMEOUT), Image.open(image_path) as img:
                img.verify()
        except TimeoutError as timeout_err:
            msg = f"Timeout while reading image: {image_path}"
            raise OSError(msg) from timeout_err
        except UnidentifiedImageError as image_err:
            msg = f"File is not a recognized image format: {image_path}"
            raise ValueError(msg) from image_err
        except (OSError, ValueError) as image_err:
            msg = f"Error accessing image {image_path}: {image_err}"
            raise OSError(msg) from image_err
    except TimeoutError as err:
        msg = f"Timeout while reading image: {image_path}"
        raise OSError(msg) from err
    except UnidentifiedImageError as err:
        msg = f"File is not a recognized image format: {image_path}"
        raise ValueError(msg) from err
    except (OSError, ValueError) as err:
        msg = f"Error accessing image {image_path}: {err}"
        raise OSError(msg) from err


@dataclass
class HFCacheScanState:
    """Mutable cache state for Hugging Face cache scans within one process."""

    attempted: bool = False
    info: HFCacheInfo | None = None
    error: OSError | ValueError | HFValidationError | None = None


_HF_CACHE_SCAN_STATE = HFCacheScanState()


def _get_hf_cache_info_cached(*, refresh: bool = False) -> HFCacheInfo:
    """Return cached HF cache info for this run, with optional refresh."""
    state = _HF_CACHE_SCAN_STATE

    if refresh:
        state.attempted = False
        state.info = None
        state.error = None

    if state.attempted:
        if state.info is not None:
            return state.info
        cached_error = state.error
        if cached_error is None:
            msg = "Hugging Face cache scan previously failed with unknown error."
            raise OSError(msg)
        raise cached_error

    try:
        cache_info = scan_cache_dir()
    except (HFValidationError, FileNotFoundError, OSError, ValueError) as err:
        state.attempted = True
        state.info = None
        state.error = err
        raise

    state.attempted = True
    state.info = cache_info
    state.error = None
    return cache_info


def _check_hf_cache_integrity(model_identifier: str) -> None:
    """Check HuggingFace cache integrity for a model and log diagnostics.

    When a model fails to load, this helps distinguish between:
    - Code bugs (wrong parameters, incompatible model)
    - Environment bugs (corrupted cache, incomplete download)

    Args:
        model_identifier: HuggingFace model identifier
            (e.g., "mlx-community/Qwen2-VL-2B-Instruct-4bit")
    """
    min_cache_size_mb = 1  # Less than 1MB is suspicious for any model
    try:
        cache_info = _get_hf_cache_info_cached()
        # Find the specific repo in cache
        repo_found = False
        for repo in cache_info.repos:
            if model_identifier in repo.repo_id:
                repo_found = True
                logger.debug(
                    "HF Cache Info for %s: size=%s MB, files=%d",
                    repo.repo_id,
                    f"{repo.size_on_disk / (1024**2):.1f}",
                    repo.nb_files,
                )
                # Check for missing or corrupt files
                if repo.nb_files == 0:
                    logger.warning(
                        "⚠️  Cache Warning: Model %s has 0 files "
                        "(incomplete download or corruption)",
                        model_identifier,
                    )
                elif repo.size_on_disk < min_cache_size_mb * (1024**2):
                    logger.warning(
                        "⚠️  Cache Warning: Model %s cache is suspiciously small (%s MB)",
                        model_identifier,
                        f"{repo.size_on_disk / (1024**2):.1f}",
                    )
                break

        if not repo_found:
            logger.debug(
                "Model %s not found in HF cache (may need to download)",
                model_identifier,
            )
    except (OSError, ValueError, HFValidationError) as cache_err:
        logger.debug("Could not check HF cache integrity: %s", cache_err)


_FAILURE_PHASE_ATTR: Final[str] = "_check_models_failure_phase"
_PROMPT_DIAGNOSTICS_ATTR: Final[str] = "_check_models_prompt_diagnostics"

_STAGE_CODE_MAP: Final[dict[str, str]] = {
    "OOM": "OOM",
    "Timeout": "TIMEOUT",
    "Missing Dep": "MISSING_DEP",
    "Lib Version": "LIB_VERSION",
    "API Drift": "API_DRIFT",
    "API Mismatch": "API_MISMATCH",
    "Config Missing": "CONFIG_MISSING",
    "No Chat Template": "NO_CHAT_TEMPLATE",
    "Weight Mismatch": "WEIGHT_MISMATCH",
    "Type Cast Error": "TYPE_CAST",
    "Processor Error": "PROCESSOR",
    "Tokenizer Error": "TOKENIZER",
    "Model Error": "MODEL",
    "Error": "ERROR",
}
_PACKAGE_CODE_MAP: Final[dict[str, str]] = {
    "mlx": "MLX",
    "mlx-vlm": "MLX_VLM",
    "mlx-lm": "MLX_LM",
    "transformers": "TRANSFORMERS",
    "huggingface-hub": "HUGGINGFACE_HUB",
    "model-config": "MODEL_CONFIG",
    "unknown": "UNKNOWN",
}

_PACKAGE_OWNER_HINTS: Final[dict[str, str]] = {
    "mlx": "mlx core/runtime (memory, tensor ops, Metal backend)",
    "mlx-vlm": "mlx-vlm generation/integration path",
    "mlx-lm": "mlx-lm text-generation/runtime path",
    "transformers": "transformers API/processor/tokenizer compatibility",
    "huggingface-hub": "huggingface-hub fetch/cache/model-file resolution",
    "model-config": "model repository/config artifacts (not core runtime)",
    "unknown": "unable to attribute confidently (inspect traceback/signature)",
}
_FAILURE_PHASE_HINTS: Final[dict[str, str]] = {
    "import": "runtime symbol/import checks",
    "model_load": "model + weights loading",
    "tokenizer_load": "tokenizer extraction/loading",
    "processor_load": "processor/image-processor initialization",
    "model_preflight": "model artifact/config preflight checks",
    "prefill": "prompt templating/tokenization",
    "decode": "decode/generation call (`mlx_vlm.generate`)",
}
_ERROR_STAGE_HINTS: Final[dict[str, str]] = {
    "OOM": "out-of-memory pressure in backend runtime",
    "Timeout": "operation exceeded configured timeout",
    "Missing Dep": "missing required package(s) or extras",
    "Lib Version": "incompatible library versions/import surface changed",
    "API Drift": "required upstream runtime contract changed (missing symbol/signature/result fields)",
    "API Mismatch": "upstream function signature changed",
    "Config Missing": "required model config/artifact missing",
    "No Chat Template": "chat template unavailable in processor/tokenizer",
    "Weight Mismatch": "model weights/config do not match expected architecture",
    "Type Cast Error": "MLX type/shape/runtime cast failure",
    "Processor Error": "processor construction/processor config incompatibility",
    "Tokenizer Error": "tokenizer class or tokenizer assets mismatch",
    "Model Error": "model runtime failure in generation path",
    "Error": "unclassified failure; inspect traceback/signature",
}


def _normalise_failure_phase(phase: str | None) -> str | None:
    """Return a normalized failure phase label or ``None`` when unavailable."""
    if phase is None:
        return None
    return phase.strip().lower().replace("-", "_")


def _tag_exception_failure_phase[E: BaseException](error: E, phase: str) -> E:
    """Annotate an exception with the current failing phase."""
    normalized = _normalise_failure_phase(phase)
    if normalized is not None:
        setattr(cast("Any", error), _FAILURE_PHASE_ATTR, normalized)
    return error


def _tag_exception_prompt_diagnostics[E: BaseException](
    error: E,
    diagnostics: PromptDiagnostics | None,
) -> E:
    """Attach rendered prompt diagnostics to an exception when available."""
    if diagnostics is not None:
        setattr(cast("Any", error), _PROMPT_DIAGNOSTICS_ATTR, diagnostics)
    return error


def _object_prompt_diagnostics(value: object | None) -> PromptDiagnostics | None:
    """Return prompt diagnostics attached to an arbitrary runtime object."""
    diagnostics = getattr(value, _PROMPT_DIAGNOSTICS_ATTR, None)
    return diagnostics if isinstance(diagnostics, PromptDiagnostics) else None


def _extract_failure_phase(
    error: BaseException,
    *,
    fallback: str | None = None,
) -> str | None:
    """Extract the first known failure phase from an exception chain."""
    cur: BaseException | None = error
    while cur is not None:
        tagged = getattr(cur, _FAILURE_PHASE_ATTR, None)
        if isinstance(tagged, str) and tagged.strip():
            return _normalise_failure_phase(tagged)
        cur = cur.__cause__
    return _normalise_failure_phase(fallback)


def _exception_chain(error: BaseException) -> Iterator[BaseException]:
    """Yield an exception and its causal/context chain without looping forever."""
    seen: set[int] = set()
    cur: BaseException | None = error
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        yield cur
        if cur.__cause__ is not None:
            cur = cur.__cause__
        elif not cur.__suppress_context__:
            cur = cur.__context__
        else:
            cur = None


def _root_cause_exception(error: BaseException) -> BaseException:
    """Return the deepest available exception in a cause/context chain."""
    root = error
    for candidate in _exception_chain(error):
        root = candidate
    return root


def _extract_exception_prompt_diagnostics(error: BaseException) -> PromptDiagnostics | None:
    """Return prompt diagnostics attached anywhere in an exception chain."""
    for item in _exception_chain(error):
        if diagnostics := _object_prompt_diagnostics(item):
            return diagnostics
    return None


def _sanitize_error_token(value: str | None, *, default: str) -> str:
    """Convert free-form labels into stable uppercase token segments."""
    if value is None or not value.strip():
        return default
    token = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().upper()).strip("_")
    return token or default


def _build_failure_action_hint(
    *,
    error_package: str | None,
    failure_phase: str | None,
    error_stage: str | None,
) -> str:
    """Build an owner/component/cause hint for failed-model summary lines."""
    package_key = (error_package or "unknown").strip().lower() or "unknown"
    owner_hint = _PACKAGE_OWNER_HINTS.get(
        package_key,
        f"{package_key} (upstream owner not in built-in hint map)",
    )

    phase_key = _normalise_failure_phase(failure_phase)
    component_hint = _FAILURE_PHASE_HINTS.get(
        phase_key or "",
        f"{phase_key or 'unknown phase'}",
    )

    stage_key = error_stage or "Error"
    cause_hint = _ERROR_STAGE_HINTS.get(
        stage_key,
        _ERROR_STAGE_HINTS["Error"],
    )

    return f"owner≈{owner_hint} | component={component_hint} | likely={cause_hint}"


def _normalize_error_core_message(error_msg: str) -> str:
    """Normalize model-specific and numeric variations from top-level error text."""
    core = error_msg.split("\n", maxsplit=1)[0].strip() if error_msg else "Unknown error"
    wrapper_match = re.match(
        r"Model (?:generation|loading|preflight) failed(?:\s+for\s+\S+)?:\s*(.+)",
        core,
    )
    if wrapper_match:
        core = wrapper_match.group(1)
    return re.sub(r"\(N(?:,N)+\)", "(N,...)", re.sub(r"\d+", "N", core))


def _normalize_traceback_signature(traceback_str: str | None) -> str:
    """Return a stable traceback fingerprint fragment for signature hashing."""
    if not traceback_str:
        return ""

    normalized_lines: list[str] = []
    for raw_line in traceback_str.strip().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("File "):
            line = re.sub(r'File ".*?([^/\\"]+)"', r'File "\1"', line)
            line = re.sub(r"line \d+", "line N", line)
        line = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", line)
        line = re.sub(r"\d+", "N", line)
        normalized_lines.append(line)

    if not normalized_lines:
        return ""
    return " | ".join(normalized_lines[-4:])


def _build_canonical_error_code(
    *,
    error_stage: str | None,
    error_package: str | None,
    failure_phase: str | None,
) -> str:
    """Build stable machine-readable error code for cross-run clustering."""
    package_token = _PACKAGE_CODE_MAP.get(
        (error_package or "unknown").lower(),
        _sanitize_error_token(error_package, default="UNKNOWN"),
    )
    phase_token = _sanitize_error_token(_normalise_failure_phase(failure_phase), default="UNKNOWN")
    stage_token = _STAGE_CODE_MAP.get(
        error_stage or "",
        _sanitize_error_token(error_stage, default="ERROR"),
    )
    return f"{package_token}_{phase_token}_{stage_token}"


def _build_error_signature(
    *,
    error_code: str,
    error_message: str | None,
    error_traceback: str | None,
) -> str:
    """Build stable signature hash for clustering related failures."""
    canonical_message = _normalize_error_core_message(error_message or "Unknown error")
    traceback_sig = _normalize_traceback_signature(error_traceback)
    payload = f"{error_code}|{canonical_message}|{traceback_sig}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"{error_code}:{digest}"


def _classify_error(error_msg: str) -> str:
    """Classify error message into a short, readable status code.

    Categories (in order of precedence):
        - OOM: Metal memory allocation failures
        - Timeout: Operation timed out
        - Missing Dep: Missing pip packages
        - Lib Version: Import errors due to version mismatches
        - API Drift: Required runtime surface missing/reshaped before invocation
        - API Mismatch: Unexpected keyword arguments (transformers/mlx-vlm API changes)
        - Config Missing: Model repository missing required config files
        - No Chat Template: Tokenizer/processor lacks chat template
        - Weight Mismatch: Model weights don't match expected parameters
        - Type Cast Error: MLX core type/cast errors (std::bad_cast)
        - Processor Error: Image processor instantiation failures
        - Tokenizer Error: Tokenizer class/loading failures
        - Model Error: Generic model loading/config issues
        - Error: Unclassified errors
    """
    msg_lower = error_msg.lower()

    # (Error Type, List of lowercase keywords that trigger this type)
    # Order matters: first match wins
    error_definitions = [
        # Critical infrastructure errors
        ("OOM", ["metal::malloc", "maximum allowed buffer size"]),
        ("Timeout", ["timeout"]),
        # Dependency/version errors
        (
            "Missing Dep",
            ["requires", "packages", "pip install"],
        ),  # All must be present logic handled below
        ("Lib Version", ["cannot import name", "importerror"]),
        (
            "API Drift",
            [
                "generation runtime api drift",
                "missing-dependency placeholder",
                "missing required keyword parameter",
                "positional-only parameter",
                "required field(s)",
                "signature could not be inspected for api drift checks",
            ],
        ),
        # API compatibility errors
        ("API Mismatch", ["unexpected keyword argument", "got an unexpected keyword"]),
        # Model configuration/file errors
        (
            "Config Missing",
            ["does not appear to have a file named", "missing required file", "config is missing"],
        ),
        ("No Chat Template", ["chat_template is not set", "no template argument was passed"]),
        # Weight/parameter errors
        ("Weight Mismatch", ["missing", "parameters"]),
        # MLX core errors
        ("Type Cast Error", ["std::bad_cast"]),
        # Processor/tokenizer errors
        ("Processor Error", ["imageprocessor", "image_processor"]),
        ("Tokenizer Error", ["tokenizer class", "does not exist"]),
        # Generic model errors
        ("Model Error", ["model", "loading", "failed"]),
    ]

    for error_type, patterns in error_definitions:
        # Special case for multi-keyword AND logic (Missing Dep, Model Error)
        if error_type == "Missing Dep":
            if all(p in msg_lower for p in patterns):
                return error_type
            continue

        if error_type == "Model Error":
            # "model" AND ("loading" OR "failed")
            if "model" in msg_lower and ("loading" in msg_lower or "failed" in msg_lower):
                return error_type
            continue

        if error_type == "Tokenizer Error":
            # "tokenizer class" AND "does not exist"
            if all(p in msg_lower for p in patterns):
                return error_type
            continue

        if error_type == "Weight Mismatch":
            # "missing" AND "parameters"
            if all(p in msg_lower for p in patterns):
                return error_type
            continue

        # Default OR logic for other patterns
        if any(p in msg_lower for p in patterns):
            return error_type

    return "Error"


def _attribute_error_to_package(error_msg: str, traceback_str: str | None = None) -> str:
    """Heuristically attribute an error to the most likely owning package.

    Uses ordered pattern precedence across message/traceback text so diagnostics
    can route issue reports to the right upstream project.
    """
    msg_lower = error_msg.lower()
    tb_lower = (traceback_str or "").lower()
    combined = msg_lower + " " + tb_lower

    # (Package Name, List of unique identification patterns)
    # Order matters: matches earlier in list take precedence
    package_definitions = [
        (
            "mlx",
            [
                "metal::malloc",
                "maximum allowed buffer size",
                "std::bad_cast",
                "mlx/core/",
                "mlx/nn/",
                "mlx/python/mlx/",
            ],
        ),
        (
            "mlx-vlm",
            [
                "mlx_vlm/",
                "mlx_vlm.",
                "mlx-vlm/",
                "apply_chat_template",
                "generationresult",
                "load_image",
                "runtime api drift",
                "missing-dependency placeholder",
            ],
        ),
        (
            "mlx-lm",
            [
                "mlx_lm/",
                "mlx-lm/",
            ],
        ),
        (
            "transformers",
            [
                "transformers/",
                "cannot import name",
                "importerror",
                "unexpected keyword argument",
                "tokenizer class",
                "processing_utils",
                "tokenization_",
                "image_processing_",
                "_batch_encode_plus",
            ],
        ),
        (
            "huggingface-hub",
            [
                "huggingface_hub",
                "does not appear to have a file",
                "hfvalidationerror",
            ],
        ),
        (
            "model-config",
            [
                "chat_template is not set",
                "no template argument",
                "config.json",
                "missing required file",
                "model preflight failed",
                "tokenizer artifacts missing",
            ],
        ),
    ]

    def _matching_packages(text: str) -> list[str]:
        matches: list[str] = []
        for package, patterns in package_definitions:
            if any(pattern in text for pattern in patterns):
                matches.append(package)
        return matches

    message_matches = _matching_packages(msg_lower)
    if message_matches:
        return message_matches[0]

    traceback_matches = _matching_packages(tb_lower)
    if traceback_matches:
        chained_exception_markers = (
            "during handling of the above exception",
            "the above exception was the direct cause of the following exception",
        )
        if any(marker in tb_lower for marker in chained_exception_markers):
            wrapped_traceback_preference = (
                "transformers",
                "huggingface-hub",
                "model-config",
                "mlx-lm",
                "mlx",
                "mlx-vlm",
            )
            for package in wrapped_traceback_preference:
                if package in traceback_matches:
                    return package

    for package, patterns in package_definitions:
        if any(pattern in combined for pattern in patterns):
            return package

    # Special case for compound logic check which didn't fit the loop
    if "missing" in combined and "parameters" in combined:
        return "model-config"

    return "unknown"


def _load_model(
    params: ProcessImageParams,
) -> tuple[nn.Module, ProcessorMixin, PreTrainedConfig | Mapping[str, object] | None]:
    """Load model from HuggingFace Hub or local path.

    Args:
        params: The parameters for image processing, including model identifier.

    Returns:
        Tuple of ``(model, processor, config)`` where ``processor`` is an
        ``transformers.ProcessorMixin`` and ``config`` may be ``None``.
    """
    model, processor = load(
        path_or_hf_repo=params.model_identifier,
        adapter_path=params.adapter_path,
        lazy=params.lazy,
        revision=params.revision,
        trust_remote_code=params.trust_remote_code,
    )
    config = cast(
        "PreTrainedConfig | Mapping[str, object] | None",
        getattr(model, "config", None),
    )
    return model, processor, config


def _set_failure_phase(
    phase_callback: Callable[[str], None] | None,
    phase: str,
) -> str:
    """Update the active execution phase and notify optional callback."""
    normalized = _normalise_failure_phase(phase) or phase
    if phase_callback is not None:
        phase_callback(normalized)
    return normalized


def _has_text_decoder_api(candidate: object) -> TypeGuard[SupportsTextDecoder]:
    """Return whether a candidate exposes the decode API we rely on."""
    return callable(getattr(candidate, "decode", None)) and callable(
        getattr(candidate, "batch_decode", None),
    )


def _extract_processor_tokenizer(
    processor: ProcessorMixin,
) -> PreTrainedTokenizerBase | SupportsTextDecoder | None:
    """Best-effort extraction of tokenizer from a loaded processor."""
    tokenizer = cast("object | None", getattr(processor, "tokenizer", None))
    if tokenizer is not None and _has_text_decoder_api(tokenizer):
        return tokenizer
    if _has_text_decoder_api(processor):
        return processor
    return None


def _resolve_model_snapshot_path(model_identifier: str) -> Path | None:
    """Resolve local snapshot path for a model identifier when available."""
    if model_identifier.startswith(("/", "./", "../")):
        path = Path(model_identifier)
        return path.resolve() if path.is_dir() else None

    try:
        cache_info = _get_hf_cache_info_cached()
    except (OSError, ValueError, FileNotFoundError, HFValidationError):
        return None

    for repo in cache_info.repos:
        if repo.repo_id != model_identifier:
            continue
        revisions = list(repo.revisions)
        if not revisions:
            repo_path = getattr(repo, "repo_path", None)
            return Path(repo_path) if repo_path else None
        revisions.sort(
            key=lambda revision: float(getattr(revision, "last_modified", 0.0) or 0.0),
            reverse=True,
        )
        snapshot_path = getattr(revisions[0], "snapshot_path", None)
        return Path(snapshot_path) if snapshot_path else None
    return None


def _get_config_value(config: object | None, key: str) -> object | None:
    """Read config values from dict-like or object-like config containers."""
    if config is None:
        return None
    if isinstance(config, Mapping):
        config_map = cast("Mapping[str, object]", config)
        return config_map.get(key)
    return getattr(config, key, None)


def _raise_preflight_error(message: str, *, phase: str) -> NoReturn:
    """Raise a preflight ValueError annotated with the failing phase."""
    raise _tag_exception_failure_phase(ValueError(message), phase)


def _validate_model_artifact_layout(
    *,
    model_identifier: str,
    snapshot_path: Path | None,
    tokenizer: PreTrainedTokenizerBase | SupportsTextDecoder | None,
) -> None:
    """Validate local model artifact structure and emit actionable warnings.

    These checks are intentionally non-fatal because many older/community repos
    rely on legacy file layouts that still run correctly with mlx-vlm.
    """
    if snapshot_path is None or not snapshot_path.exists():
        logger.debug(
            "Skipping file-layout preflight for %s (snapshot unavailable).",
            model_identifier,
        )
        return

    if tokenizer is not None:
        tokenizer_candidates = (
            "tokenizer_config.json",
            "tokenizer.json",
            "tokenizer.model",
            "vocab.json",
        )
        if not any((snapshot_path / name).exists() for name in tokenizer_candidates):
            logger.debug(
                "Legacy snapshot note for %s: tokenizer artifacts missing from snapshot (%s).",
                model_identifier,
                ", ".join(tokenizer_candidates),
            )

    processor_candidates = ("preprocessor_config.json", "processor_config.json")
    if not any((snapshot_path / name).exists() for name in processor_candidates):
        logger.debug(
            "Legacy snapshot note for %s: processor config missing from snapshot (%s).",
            model_identifier,
            ", ".join(processor_candidates),
        )


def _run_model_preflight_validators(
    *,
    model_identifier: str,
    processor: ProcessorMixin,
    config: PreTrainedConfig | Mapping[str, object] | None,
    phase_callback: Callable[[str], None] | None = None,
) -> None:
    """Run preflight validators before invoking generation."""
    _set_failure_phase(phase_callback, "tokenizer_load")
    tokenizer = _extract_processor_tokenizer(processor)
    if tokenizer is None:
        _raise_preflight_error(
            "Could not resolve tokenizer from loaded processor.",
            phase="tokenizer_load",
        )

    _set_failure_phase(phase_callback, "processor_load")
    if not callable(processor):
        _raise_preflight_error(
            "Loaded processor is not callable.",
            phase="processor_load",
        )
    if getattr(processor, "image_processor", None) is None:
        _raise_preflight_error(
            "Loaded processor has no image_processor; expected multimodal processor.",
            phase="processor_load",
        )

    _set_failure_phase(phase_callback, "model_preflight")
    model_type = _get_config_value(config, "model_type")
    if not isinstance(model_type, str) or not model_type.strip():
        logger.warning(
            "Preflight warning for %s: model config missing model_type.",
            model_identifier,
        )

    snapshot_path = _resolve_model_snapshot_path(model_identifier)
    _validate_model_artifact_layout(
        model_identifier=model_identifier,
        snapshot_path=snapshot_path,
        tokenizer=tokenizer,
    )


def _ensure_generation_runtime_symbols() -> None:
    """Validate core generation callables before model execution."""
    runtime_api_issues = list(_detect_runtime_api_drift_issues())
    if not runtime_api_issues:
        return

    msg = "Generation runtime API drift: " + "; ".join(runtime_api_issues)
    raise _tag_exception_failure_phase(RuntimeError(msg), "import")


def _is_mlx_vlm_bpe_detokenizer_decode_failure(error: BaseException) -> bool:
    """Return whether an error matches the upstream mlx-vlm UTF-8 detokenizer bug."""
    if not isinstance(error, UnicodeDecodeError):
        return False
    if error.encoding.strip().lower() != "utf-8":
        return False

    for frame in traceback.extract_tb(error.__traceback__):
        normalized_filename = frame.filename.replace("\\", "/")
        if (
            normalized_filename.endswith("/mlx_vlm/tokenizer_utils.py")
            and frame.name == "add_token"
        ):
            return True
    return False


@contextmanager
def _temporary_mlx_vlm_lossy_bpe_detokenizer_patch() -> Generator[None]:
    """Temporarily ignore undecodable bytes in mlx-vlm BPE streaming detokenization."""
    space_byte: Final[int] = 32
    try:
        tokenizer_utils = __import__(
            "mlx_vlm.tokenizer_utils",
            fromlist=["BPEStreamingDetokenizer"],
        )
    except ImportError:
        yield
        return

    detokenizer_cls = getattr(tokenizer_utils, "BPEStreamingDetokenizer", None)
    remove_space = getattr(tokenizer_utils, "_remove_space", None)
    original_add_token = getattr(detokenizer_cls, "add_token", None)
    if not isinstance(detokenizer_cls, type) or not callable(original_add_token):
        yield
        return
    detokenizer_type = cast("Any", detokenizer_cls)

    def _lossy_add_token(
        self: object,
        token: int,
        skip_special_token_ids: Sequence[int] = (),
    ) -> None:
        detokenizer = cast("Any", self)
        if token in skip_special_token_ids:
            return

        tokenmap = getattr(detokenizer, "tokenmap", None)
        byte_decoder = getattr(detokenizer, "_byte_decoder", None)
        pending = getattr(detokenizer, "_unflushed", None)
        accumulated_text = getattr(detokenizer, "text", None)
        trim_space = getattr(detokenizer, "trim_space", False)
        if (
            not isinstance(tokenmap, list)
            or not isinstance(byte_decoder, Mapping)
            or not isinstance(pending, str)
            or not isinstance(accumulated_text, str)
        ):
            original_add_token(self, token, skip_special_token_ids)
            return

        try:
            value = tokenmap[token]
        except (IndexError, TypeError):
            original_add_token(self, token, skip_special_token_ids)
            return
        if not isinstance(value, str) or not value:
            original_add_token(self, token, skip_special_token_ids)
            return

        try:
            leading_byte = byte_decoder[value[0]]
        except KeyError:
            original_add_token(self, token, skip_special_token_ids)
            return

        if leading_byte != space_byte:
            object.__setattr__(detokenizer, "_unflushed", pending + value)
            return

        current_text = bytearray(byte_decoder[ch] for ch in pending if ch in byte_decoder).decode(
            "utf-8", errors="ignore"
        )
        if accumulated_text or not trim_space:
            next_text = accumulated_text + current_text
        elif callable(remove_space):
            next_text = accumulated_text + cast("str", remove_space(current_text))
        else:
            next_text = accumulated_text + current_text.lstrip()

        detokenizer.text = next_text
        object.__setattr__(detokenizer, "_unflushed", value)

    detokenizer_type.add_token = _lossy_add_token
    try:
        yield
    finally:
        detokenizer_type.add_token = original_add_token


def _run_generation_with_retry_workaround(
    *,
    params: ProcessImageParams,
    generate_once: Callable[[], GenerationResult | SupportsGenerationResult],
) -> GenerationResult | SupportsGenerationResult:
    """Run generation once, retrying only for the known upstream detokenizer bug."""
    try:
        return generate_once()
    except TimeoutError as gen_to_err:
        msg = f"Generation timed out for model {params.model_identifier}: {gen_to_err}"
        raise _tag_exception_failure_phase(TimeoutError(msg), "decode") from gen_to_err
    except (OSError, ValueError) as gen_known_err:
        if not _is_mlx_vlm_bpe_detokenizer_decode_failure(gen_known_err):
            msg = f"Model generation failed for {params.model_identifier}: {gen_known_err}"
            logger.exception("Generation error for %s", params.model_identifier)
            raise _tag_exception_failure_phase(ValueError(msg), "decode") from gen_known_err

        logger.warning(
            "Generation hit upstream mlx-vlm UTF-8 detokenizer failure for %s; "
            "retrying once with lossy BPE decode fallback.",
            params.model_identifier,
        )
        try:
            with _temporary_mlx_vlm_lossy_bpe_detokenizer_patch():
                return generate_once()
        except TimeoutError as retry_timeout_err:
            msg = f"Generation timed out for model {params.model_identifier}: {retry_timeout_err}"
            raise _tag_exception_failure_phase(TimeoutError(msg), "decode") from retry_timeout_err
        except (OSError, ValueError) as retry_known_err:
            msg = f"Model generation failed for {params.model_identifier}: {retry_known_err}"
            logger.exception("Generation error for %s", params.model_identifier)
            raise _tag_exception_failure_phase(ValueError(msg), "decode") from retry_known_err
        except (RuntimeError, TypeError, AttributeError, KeyError) as retry_err:
            msg = (
                f"Model runtime error during generation for {params.model_identifier}: {retry_err}"
            )
            logger.exception("Runtime error for %s", params.model_identifier)
            raise _tag_exception_failure_phase(ValueError(msg), "decode") from retry_err
    except (RuntimeError, TypeError, AttributeError, KeyError) as gen_err:
        msg = f"Model runtime error during generation for {params.model_identifier}: {gen_err}"
        logger.exception("Runtime error for %s", params.model_identifier)
        raise _tag_exception_failure_phase(ValueError(msg), "decode") from gen_err


def _build_runtime_diagnostics(
    phase_timer: PhaseTimer,
    *,
    stop_reason: str | None,
    first_token_latency_s: float | None = None,
) -> RuntimeDiagnostics:
    """Build immutable runtime diagnostics from a phase timer snapshot."""
    return RuntimeDiagnostics(
        input_validation_time_s=phase_timer.duration("input_validation"),
        model_load_time_s=phase_timer.duration("model_load"),
        prompt_prep_time_s=phase_timer.duration("prompt_prep"),
        decode_time_s=phase_timer.duration("decode"),
        cleanup_time_s=phase_timer.duration("cleanup"),
        first_token_latency_s=first_token_latency_s,
        stop_reason=stop_reason,
    )


def _prepare_generation_prompt(
    *,
    params: ProcessImageParams,
    processor: ProcessorMixin,
    config: PreTrainedConfig | Mapping[str, object] | None,
    phase_callback: Callable[[str], None] | None,
    phase_timer: PhaseTimer | None,
) -> str | list[object]:
    """Run preflight checks and build the prompt payload for generation."""
    try:
        chat_template_kwargs: ChatTemplateKwargs = (
            {"enable_thinking": True} if params.enable_thinking else {}
        )
        if phase_timer is not None:
            with phase_timer.track("prompt_prep"):
                _run_model_preflight_validators(
                    model_identifier=params.model_identifier,
                    processor=processor,
                    config=config,
                    phase_callback=phase_callback,
                )
                _set_failure_phase(phase_callback, "prefill")
                return cast(
                    "str | list[object]",
                    apply_chat_template(
                        processor=processor,
                        config=config,
                        prompt=params.prompt,
                        num_images=1,
                        **chat_template_kwargs,
                    ),
                )

        _run_model_preflight_validators(
            model_identifier=params.model_identifier,
            processor=processor,
            config=config,
            phase_callback=phase_callback,
        )
        _set_failure_phase(phase_callback, "prefill")
        return cast(
            "str | list[object]",
            apply_chat_template(
                processor=processor,
                config=config,
                prompt=params.prompt,
                num_images=1,
                **chat_template_kwargs,
            ),
        )
    except ValueError as preflight_err:
        message = f"Model preflight failed for {params.model_identifier}: {preflight_err}"
        logger.exception("Model preflight validation failed for %s", params.model_identifier)
        phase = (
            _extract_failure_phase(preflight_err, fallback="model_preflight") or "model_preflight"
        )
        raise _tag_exception_failure_phase(ValueError(message), phase) from preflight_err
    except (OSError, RuntimeError, TypeError, AttributeError, KeyError) as prefill_err:
        msg = f"Prompt prefill failed for {params.model_identifier}: {prefill_err}"
        logger.exception("Prompt prefill failed for %s", params.model_identifier)
        raise _tag_exception_failure_phase(ValueError(msg), "prefill") from prefill_err


def _cleanup_runtime_resources(*, synchronize_first: bool = True) -> None:
    """Clear runtime resources after each model run.

    Successful generation paths already synchronize before sampling timing and
    memory, so callers can skip the extra barrier when no further MLX work ran.
    Failure paths should still request the pre-cleanup synchronize to avoid
    clearing caches while kernels may still be in flight.
    """

    def _run_cleanup_step(step_name: str, func: Callable[[], object] | None) -> None:
        if func is None:
            return
        try:
            func()
        except (AttributeError, OSError, RuntimeError, SystemError, TypeError, ValueError):
            logger.debug("Ignoring cleanup failure in %s", step_name, exc_info=True)

    if synchronize_first:
        synchronize_fn = cast("Callable[[], object] | None", getattr(mx, "synchronize", None))
        _run_cleanup_step("mx.synchronize", synchronize_fn)
    gc.collect()

    clear_cache_fn = cast("Callable[[], object] | None", getattr(mx, "clear_cache", None))
    _run_cleanup_step("mx.clear_cache", clear_cache_fn)

    reset_peak_memory_fn = cast(
        "Callable[[], object] | None",
        getattr(mx, "reset_peak_memory", None),
    )
    _run_cleanup_step("mx.reset_peak_memory", reset_peak_memory_fn)


def _generate_with_processor_passthrough(
    *,
    generate_fn: Callable[..., GenerationResult],
    model: nn.Module,
    processor: ProcessorMixin,
    params: ProcessImageParams,
    formatted_prompt: str,
    extra_kwargs: GenerateExtraKwargs,
) -> GenerationResult:
    """Call upstream generate() with user-provided passthrough kwargs.

    This branch is intentionally dynamic because ``processor_kwargs`` is a
    user-supplied JSON object whose keys depend on the active upstream model.
    """
    processor_kwargs = params.processor_kwargs or {}
    return generate_fn(
        model=model,
        processor=processor,
        prompt=formatted_prompt,
        image=str(params.image_path),
        verbose=params.verbose,
        temperature=params.temperature,
        top_p=params.top_p,
        repetition_penalty=params.repetition_penalty,
        repetition_context_size=params.repetition_context_size,
        max_kv_size=params.max_kv_size,
        kv_bits=params.kv_bits,
        kv_quant_scheme=params.kv_quant_scheme,
        kv_group_size=params.kv_group_size,
        quantized_kv_start=params.quantized_kv_start,
        max_tokens=params.max_tokens,
        **processor_kwargs,
        **extra_kwargs,
    )


def _attach_generation_runtime_metrics(
    output: GenerationResult | SupportsGenerationResult,
    *,
    duration: float,
) -> SupportsGenerationResult:
    """Attach local timing and allocator snapshot metrics to a generation result."""
    get_active_memory_fn = getattr(mx, "get_active_memory", None)
    get_cache_memory_fn = getattr(mx, "get_cache_memory", None)

    active_mem_raw = get_active_memory_fn() if callable(get_active_memory_fn) else 0.0
    cache_mem_raw = get_cache_memory_fn() if callable(get_cache_memory_fn) else 0.0

    active_mem_bytes = float(active_mem_raw) if isinstance(active_mem_raw, int | float) else 0.0
    cache_mem_bytes = float(cache_mem_raw) if isinstance(cache_mem_raw, int | float) else 0.0

    result = cast("SupportsGenerationResult", output)
    result.time = duration
    result.active_memory = active_mem_bytes / (1024**3)  # Convert to GB
    result.cache_memory = cache_mem_bytes / (1024**3)  # Convert to GB
    return result


def _derive_first_token_latency_s(
    output: GenerationResult | SupportsGenerationResult,
) -> float | None:
    """Derive first-token latency from upstream prompt throughput metrics.

    Installed mlx-vlm computes ``prompt_tps`` as ``prompt_tokens / prompt_time``
    inside ``stream_generate()``, where ``prompt_time`` is the elapsed time until
    the first token becomes available. That makes ``prompt_tokens / prompt_tps``
    a stable way to recover first-token latency from the final ``GenerationResult``.
    """
    prompt_tokens = output.prompt_tokens
    prompt_tps = output.prompt_tps
    if not isinstance(prompt_tokens, int | float) or prompt_tokens <= 0:
        return None
    if not isinstance(prompt_tps, int | float) or prompt_tps <= 0:
        return None
    return float(prompt_tokens) / float(prompt_tps)


def _finalize_process_result(
    *,
    result_payload: PerformanceResult | None,
    params: ProcessImageParams,
    phase_timer: PhaseTimer,
    stop_reason: str | None,
    current_phase: str,
    total_start_time: float,
) -> PerformanceResult:
    """Attach final runtime diagnostics after cleanup has completed."""
    runtime_diagnostics = _build_runtime_diagnostics(
        phase_timer,
        stop_reason=(
            result_payload.runtime_diagnostics.stop_reason
            if result_payload is not None and result_payload.runtime_diagnostics is not None
            else stop_reason
        ),
        first_token_latency_s=(
            result_payload.runtime_diagnostics.first_token_latency_s
            if result_payload is not None and result_payload.runtime_diagnostics is not None
            else None
        ),
    )
    if result_payload is None:
        fallback_error = RuntimeError(
            "process_image_with_model completed without producing a result",
        )
        return _build_failure_result(
            model_name=params.model_identifier,
            error=fallback_error,
            captured_output=None,
            failure_phase=current_phase,
            generation_time=phase_timer.duration("decode"),
            model_load_time=phase_timer.duration("model_load"),
            total_time=time.perf_counter() - total_start_time,
            runtime_diagnostics=runtime_diagnostics,
            requested_max_tokens=params.max_tokens,
        )
    return replace(result_payload, runtime_diagnostics=runtime_diagnostics)


def _run_model_generation(
    params: ProcessImageParams,
    timer: TimingStrategy | None = None,
    phase_callback: Callable[[str], None] | None = None,
    phase_timer: PhaseTimer | None = None,
) -> GenerationResult | SupportsGenerationResult:
    """Load model + processor, apply chat template, run generation, time it.

    We keep all loading + formatting + generation steps together because they
    form a tightly coupled sequence (tokenizer/model/config interplay varies by
    repo). Errors are wrapped with traceback context so upstream summaries can
    show concise messages while verbose logs retain full detail.

    Args:
        params: The parameters for the image processing.
        timer: Optional timing strategy. If None, uses PerfCounterTimer.
        phase_callback: Optional callback invoked when execution enters a new phase.
        phase_timer: Optional per-phase timer that records load, prep, and decode durations.
    """
    model: nn.Module
    processor: ProcessorMixin

    _set_failure_phase(phase_callback, "import")
    _ensure_generation_runtime_symbols()

    # Load model from HuggingFace Hub - this handles automatic download/caching
    # and converts weights to MLX format for Apple Silicon optimization
    _set_failure_phase(phase_callback, "model_load")
    try:
        if phase_timer is not None:
            with phase_timer.track("model_load"):
                model, processor, config = _load_model(params)
        else:
            model, processor, config = _load_model(params)
    except Exception as load_err:
        # Capture model loading errors and run cache diagnostics to distinguish
        # code bugs from environment issues (corrupted cache, incomplete download)
        error_details = f"Model loading failed: {load_err}"
        logger.exception("Failed to load model %s", params.model_identifier)
        _check_hf_cache_integrity(params.model_identifier)

        raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err

    formatted_prompt = _prepare_generation_prompt(
        params=params,
        processor=processor,
        config=config,
        phase_callback=phase_callback,
        phase_timer=phase_timer,
    )
    # Handle list return from apply_chat_template
    if isinstance(formatted_prompt, list):
        formatted_prompt = "\n".join(str(m) for m in formatted_prompt)

    # Time the generation process manually since MLX VLM doesn't include timing
    # Use injected timer or default to PerfCounterTimer
    if timer is None:
        timer = PerfCounterTimer()

    extra_kwargs = _build_generate_extra_kwargs(params)
    processor_passthrough_kwargs = params.processor_kwargs or {}
    prompt_diagnostics = _build_prompt_diagnostics(
        params=params,
        processor=processor,
        config=config,
        formatted_prompt=formatted_prompt,
        extra_kwargs=extra_kwargs,
        processor_passthrough_kwargs=processor_passthrough_kwargs,
    )
    strict_generate = cast("StrictGenerateCallable", generate)

    def _generate_once() -> GenerationResult | SupportsGenerationResult:
        if processor_passthrough_kwargs:
            return _generate_with_processor_passthrough(
                generate_fn=generate,
                model=model,
                processor=processor,
                params=params,
                formatted_prompt=formatted_prompt,
                extra_kwargs=extra_kwargs,
            )
        return strict_generate(
            model=model,
            processor=processor,
            prompt=formatted_prompt,
            image=str(params.image_path),
            verbose=params.verbose,
            temperature=params.temperature,
            top_p=params.top_p,
            repetition_penalty=params.repetition_penalty,
            repetition_context_size=params.repetition_context_size,
            max_kv_size=params.max_kv_size,
            kv_bits=params.kv_bits,
            kv_quant_scheme=params.kv_quant_scheme,
            kv_group_size=params.kv_group_size,
            quantized_kv_start=params.quantized_kv_start,
            max_tokens=params.max_tokens,
            **extra_kwargs,
        )

    timer.start()
    if phase_timer is not None:
        phase_timer.start("decode")
    _set_failure_phase(phase_callback, "decode")
    try:
        output = _run_generation_with_retry_workaround(
            params=params,
            generate_once=_generate_once,
        )
    except (TimeoutError, OSError, ValueError, RuntimeError) as generation_err:
        _tag_exception_prompt_diagnostics(generation_err, prompt_diagnostics)
        raise
    finally:
        if phase_timer is not None:
            phase_timer.stop("decode")

    # Force GPU synchronization to ensures timing includes all pending compute (MLX is lazy)
    mx.synchronize()
    duration = timer.stop()

    # Capture local timing plus active/cache memory snapshots while model state is intact.
    result = _attach_generation_runtime_metrics(output, duration=duration)
    setattr(cast("Any", result), _PROMPT_DIAGNOSTICS_ATTR, prompt_diagnostics)
    return result


def _build_failure_result(
    *,
    model_name: str,
    error: TimeoutError | OSError | ValueError | RuntimeError,
    captured_output: str | None,
    quality_issues: str | None = None,
    quality_analysis: GenerationQualityAnalysis | None = None,
    failure_phase: str | None = None,
    generation_time: float | None = None,
    model_load_time: float | None = None,
    total_time: float | None = None,
    runtime_diagnostics: RuntimeDiagnostics | None = None,
    requested_max_tokens: int | None = None,
    prompt_diagnostics: PromptDiagnostics | None = None,
) -> PerformanceResult:
    """Build a standardized failure result payload for a model run."""
    error_msg = str(error)
    tb_str = traceback.format_exc()
    root_error = _root_cause_exception(error)
    root_error_msg = str(root_error)
    classification_text = " ".join(part for part in (root_error_msg, error_msg) if part)
    resolved_phase = _extract_failure_phase(error, fallback=failure_phase)
    classified_stage = _classify_error(classification_text or error_msg)
    error_package = _attribute_error_to_package(classification_text or error_msg, tb_str)
    error_code = _build_canonical_error_code(
        error_stage=classified_stage,
        error_package=error_package,
        failure_phase=resolved_phase,
    )
    error_signature = _build_error_signature(
        error_code=error_code,
        error_message=root_error_msg or error_msg,
        error_traceback=tb_str,
    )
    resolved_prompt_diagnostics = prompt_diagnostics or _extract_exception_prompt_diagnostics(error)
    return PerformanceResult(
        model_name=model_name,
        generation=None,
        success=False,
        failure_phase=resolved_phase,
        error_stage=classified_stage,
        error_code=error_code,
        error_signature=error_signature,
        error_message=error_msg,
        captured_output_on_fail=captured_output,
        error_type=type(error).__name__,
        root_error_type=type(root_error).__name__,
        root_error_module=type(root_error).__module__,
        root_error_message=root_error_msg,
        quality_issues=quality_issues,
        quality_analysis=quality_analysis,
        error_package=error_package,
        error_traceback=tb_str,
        generation_time=generation_time,
        model_load_time=model_load_time,
        total_time=total_time,
        runtime_diagnostics=runtime_diagnostics,
        requested_max_tokens=requested_max_tokens,
        prompt_diagnostics=resolved_prompt_diagnostics,
    )


def process_image_with_model(params: ProcessImageParams) -> PerformanceResult:
    """Process an image with a Vision Language Model, managing stats and errors."""
    arch, gpu_info = get_system_info()
    stdout_capture = _TeeCaptureStream(sys.stdout)
    stderr_capture = _TeeCaptureStream(sys.stderr)

    # Track overall timing
    total_start_time = time.perf_counter()
    current_phase: str = "input_validation"
    phase_timer = PhaseTimer()
    result_payload: PerformanceResult | None = None
    stop_reason: str | None = None
    cleanup_requires_sync = True

    def _update_phase(phase: str) -> None:
        nonlocal current_phase
        current_phase = _normalise_failure_phase(phase) or phase

    try:
        _update_phase("input_validation")
        with phase_timer.track("input_validation"):
            validate_temperature(temp=params.temperature)
            validate_image_accessible(image_path=params.image_path)
        logger.debug(
            "System: %s, GPU: %s",
            arch,
            gpu_info if gpu_info is not None else "",
        )
        if params.verbose:
            logger.debug(
                "[verbose passthrough start] mlx-vlm.generate output for %s",
                params.model_identifier,
            )

        with (
            redirect_stdout(cast("TextIO", stdout_capture)),
            redirect_stderr(cast("TextIO", stderr_capture)),
            TimeoutManager(params.timeout),
        ):
            output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                params=params,
                phase_callback=_update_phase,
                phase_timer=phase_timer,
            )
            cleanup_requires_sync = False
        if params.verbose:
            logger.debug(
                "[verbose passthrough end] mlx-vlm.generate output for %s",
                params.model_identifier,
            )

        generation_time = getattr(output, "time", None) or phase_timer.duration("decode")
        total_time = time.perf_counter() - total_start_time
        model_load_time = phase_timer.duration("model_load")
        first_token_latency_s = _derive_first_token_latency_s(output)

        # Read memory metrics from GenerationResult (captured inside _run_model_generation)
        active_mem_gb = getattr(output, "active_memory", None) or 0.0
        cache_mem_gb = getattr(output, "cache_memory", None) or 0.0

        stop_reason = "completed"

        result_payload = PerformanceResult(
            model_name=params.model_identifier,
            generation=output,
            success=True,
            generation_time=generation_time,
            model_load_time=model_load_time,
            total_time=total_time,
            active_memory=active_mem_gb if active_mem_gb > 0 else None,
            cache_memory=cache_mem_gb if cache_mem_gb > 0 else None,
            runtime_diagnostics=_build_runtime_diagnostics(
                phase_timer,
                first_token_latency_s=first_token_latency_s,
                stop_reason=stop_reason,
            ),
            requested_max_tokens=params.max_tokens,
            prompt_diagnostics=_object_prompt_diagnostics(output),
        )
        result_payload = _populate_result_quality_analysis(
            result_payload,
            prompt=params.prompt,
            requested_max_tokens=params.max_tokens,
            context_marker=params.context_marker,
        )
    except (TimeoutError, OSError, ValueError, RuntimeError) as e:
        captured_sections: list[str] = []
        stdout_clean = stdout_capture.getvalue().strip()
        stderr_clean = stderr_capture.getvalue().strip()
        failure_quality_analysis: GenerationQualityAnalysis | None = None
        failure_quality_issues: str | None = None
        if stdout_clean:
            captured_sections.append("=== STDOUT ===\n" + stdout_clean)
            if (
                len(stdout_clean) >= QUALITY.min_text_length
                or len(stdout_clean.split()) >= QUALITY.min_token_count
            ):
                failure_quality_analysis, failure_quality_issues = _analyze_text_quality(
                    stdout_clean,
                    max(len(stdout_clean.split()), 1),
                    prompt=params.prompt,
                    requested_max_tokens=params.max_tokens,
                    context_marker=params.context_marker,
                )
        if stderr_clean:
            captured_sections.append("=== STDERR ===\n" + stderr_clean)
        captured_output = "\n\n".join(captured_sections) if captured_sections else None
        stop_reason = "timeout" if isinstance(e, TimeoutError) else "exception"
        result_payload = _build_failure_result(
            model_name=params.model_identifier,
            error=e,
            captured_output=captured_output,
            quality_issues=failure_quality_issues,
            quality_analysis=failure_quality_analysis,
            failure_phase=current_phase,
            generation_time=phase_timer.duration("decode"),
            model_load_time=phase_timer.duration("model_load"),
            total_time=time.perf_counter() - total_start_time,
            runtime_diagnostics=_build_runtime_diagnostics(
                phase_timer,
                stop_reason=stop_reason,
            ),
            requested_max_tokens=params.max_tokens,
        )
    finally:
        _update_phase("cleanup")
        with phase_timer.track("cleanup"):
            _cleanup_runtime_resources(synchronize_first=cleanup_requires_sync)
        logger.debug("Cleaned up resources for model %s", params.model_identifier)

    return _finalize_process_result(
        result_payload=result_payload,
        params=params,
        phase_timer=phase_timer,
        stop_reason=stop_reason or "exception",
        current_phase=current_phase,
        total_start_time=total_start_time,
    )


# --- Main Execution Helper Functions ---


def print_cli_header(title: str) -> None:
    """Print a formatted CLI header with the given title."""
    width = get_terminal_width(max_width=100)
    log_rule(width, char="=", style="blue", bold=True)
    logger.info(
        title,
        extra={"style_hint": LogStyles.HEADER, "style_width": width},
    )
    log_rule(width, char="=", style="blue", bold=True)


def print_cli_section(title: str, *, show_rule: bool = True) -> None:
    """Print a formatted CLI section header with visual prefix."""
    width = get_terminal_width(max_width=100)
    if show_rule:
        log_rule(width, char="─", style="blue")
    logger.info(
        title,
        extra={
            "style_hint": LogStyles.SECTION,
            "style_uppercase": "\x1b[" not in title,
        },
    )


def print_cli_error(msg: str) -> None:
    """Print a formatted CLI error message."""
    logger.error(msg, extra={"style_hint": LogStyles.ERROR})


def exit_with_cli_error(
    msg: str,
    *,
    exit_code: int = 1,
    suppress_cause: bool = False,
    cause: BaseException | None = None,
) -> NoReturn:
    """Log a CLI-friendly error message and terminate the program."""
    print_cli_error(msg)
    if suppress_cause:
        raise SystemExit(exit_code) from None
    if cause is not None:
        raise SystemExit(exit_code) from cause
    raise SystemExit(exit_code)


def log_success(msg: str, *, prefix: str = "✓") -> None:
    """Log a success message with green styling and optional prefix."""
    formatted_msg = f"{prefix} {msg}" if prefix else msg
    logger.info(formatted_msg, extra={"style_hint": LogStyles.SUCCESS})


def log_warning_note(msg: str, *, prefix: str = "⚠️") -> None:
    """Log a warning note (non-error condition worth noting)."""
    formatted_msg = f"{prefix}  {msg}" if prefix else msg
    logger.warning(formatted_msg, extra={"style_hint": LogStyles.WARNING})


def log_failure(msg: str, *, prefix: str = "✗") -> None:
    """Log a failure message with red styling and optional prefix."""
    formatted_msg = f"{prefix} {msg}" if prefix else msg
    logger.error(formatted_msg, extra={"style_hint": LogStyles.ERROR})


def log_metric_label(label: str, *, emoji: str = "", indent: str = "") -> None:
    """Log a metric category label (e.g., '🔢 Tokens:') with consistent styling."""
    formatted = f"{indent}{label}"
    logger.info(
        formatted,
        extra={"style_hint": LogStyles.METRIC_LABEL, "style_emoji": emoji},
    )


def log_generated_text(
    text: str,
    *,
    wrap: bool = True,
    indent: str = "",
    max_lines: int | None = None,
) -> None:
    """Log generated model output with cyan styling and optional wrapping.

    Preserves original line breaks in the text. Each line is wrapped independently
    to terminal width if needed.

    Args:
        text: The generated text to display
        wrap: Whether to wrap long lines to terminal width
        indent: Indentation prefix for each line
        max_lines: Maximum number of lines to log (truncates if exceeded)
    """
    lines = text.splitlines()
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        lines.append("... (truncated)")

    if wrap:
        width = get_terminal_width(max_width=100)
        avail_width = max(20, width - len(indent))

        for line in lines:
            if not line.strip():
                logger.info("", extra={"style_hint": LogStyles.GENERATED_TEXT})
                continue

            if len(line) <= avail_width:
                formatted = f"{indent}{line}"
                logger.info(formatted, extra={"style_hint": LogStyles.GENERATED_TEXT})
            else:
                wrapped = textwrap.wrap(
                    line,
                    width=avail_width,
                    break_long_words=False,
                    break_on_hyphens=False,
                ) or [line]
                for wrapped_line in wrapped:
                    formatted = f"{indent}{wrapped_line}"
                    logger.info(formatted, extra={"style_hint": LogStyles.GENERATED_TEXT})
    else:
        for line in lines:
            formatted = f"{indent}{line}"
            logger.info(formatted, extra={"style_hint": LogStyles.GENERATED_TEXT})


def log_model_name(name: str, *, label: str = "") -> None:
    """Log a model identifier with magenta highlight.

    Args:
        name: The model identifier/name
        label: Optional label prefix (e.g., 'Model:')
    """
    if label:
        # Split formatting: plain label + styled name
        msg = "%s %s"
        logger.info(
            msg,
            label,
            name,
            extra={"style_hint": LogStyles.MODEL_NAME},
        )
    else:
        logger.info(name, extra={"style_hint": LogStyles.MODEL_NAME})


def log_file_path(path: Path | str, *, label: str = "", style: str = "cyan") -> None:
    """Log a file path with highlighting.

    Args:
        path: The file path to display
        label: Optional label prefix (e.g., '   HTML:')
        style: Rich style to use for the path
    """
    path_str = str(path)
    if label:
        msg = "%s %s"
        logger.info(
            msg,
            label,
            path_str,
            extra={"style_hint": LogStyles.FILE_PATH, "style_override": style},
        )
    else:
        logger.info(
            path_str,
            extra={"style_hint": LogStyles.FILE_PATH, "style_override": style},
        )


def log_blank(count: int = 1) -> None:
    """Log blank lines for spacing.

    Args:
        count: Number of blank lines to emit
    """
    for _ in range(count):
        logger.info("")


def _render_rich_lines(renderable: ConsoleRenderable, *, width: int | None = None) -> list[str]:
    """Render a Rich object to plain text lines for logger/file-log parity."""
    stream = io.StringIO()
    render_console = Console(
        file=stream,
        width=width or get_terminal_width(max_width=120),
        no_color=True,
        force_terminal=False,
        markup=False,
        highlight=False,
        emoji=True,
    )
    render_console.print(renderable)
    return stream.getvalue().splitlines()


def _log_rich_renderable(
    renderable: ConsoleRenderable,
    *,
    indent: str = "",
    width: int | None = None,
) -> None:
    """Log a Rich-rendered table/panel while keeping persisted logs plain."""
    for line in _render_rich_lines(renderable, width=width):
        logger.info("%s%s", indent, line)


type RichJustify = Literal["left", "center", "right"]


def _log_rich_table(
    *,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    justifications: Sequence[RichJustify] | None = None,
    max_widths: Sequence[int | None] | None = None,
    no_wrap_headers: frozenset[str] = frozenset(),
    indent: str = "   ",
) -> None:
    """Log a Rich table from plain rows with consistent console/file behavior."""
    table = Table(box=box.ROUNDED, show_lines=False, expand=False)
    for index, header in enumerate(headers):
        justify = (
            justifications[index] if justifications and index < len(justifications) else "left"
        )
        max_width = max_widths[index] if max_widths and index < len(max_widths) else None
        table.add_column(
            header,
            justify=justify,
            no_wrap=header in no_wrap_headers,
            max_width=max_width,
            overflow="fold",
        )
    for row in rows:
        table.add_row(*row)
    _log_rich_renderable(table, indent=indent)


type MetricTreeRow = tuple[str, str]


def _metric_tree_label(label: str, value: str) -> Text:
    """Build one Rich tree node for a key/value metric row."""
    text = Text()
    if label:
        text.append(_display_align(label, 12, alignment="left"), style="bold")
        text.append(" ")
    text.append(value)
    return text


def _log_metric_tree(
    title: str,
    rows: Sequence[MetricTreeRow],
    *,
    emoji: str = "",
    indent: str = "  ",
) -> None:
    """Log a Rich tree for grouped CLI metrics."""
    if not rows:
        return
    title_text = Text(f"{emoji} {title}" if emoji else title, style="bold white")
    tree = Tree(title_text, guide_style="dim")
    for label, value in rows:
        tree.add(_metric_tree_label(label, value))
    _log_rich_renderable(tree, indent=indent)


def _summary_parts(res: PerformanceResult, model_short: str) -> list[str]:
    """Assemble key=value summary segments for per-run triage."""
    parts: list[str] = [
        f"model={model_short}",
        f"status={'OK' if res.success else 'FAIL'}",
    ]
    issue_labels = sorted(_extract_quality_issue_labels(res.quality_issues))
    if res.success:
        if issue_labels:
            parts.append(f"quality={'+'.join(issue_labels)}")
        else:
            parts.append("quality=clean")
    elif issue_labels:
        parts.append(f"quality={'+'.join(issue_labels)}")
    if res.error_stage:
        parts.append(f"stage={res.error_stage}")
    if res.failure_phase:
        parts.append(f"phase={res.failure_phase}")
    if res.error_code:
        parts.append(f"code={res.error_code}")
    if res.error_package:
        parts.append(f"package={res.error_package}")
    if res.error_type:
        parts.append(f"type={res.error_type}")
    return parts


def _preview_generation(
    gen: StoredGenerationResult | None,
    *,
    analysis: GenerationQualityAnalysis | None = None,
    prompt: str | None = None,
    context_marker: str = "Context:",
) -> None:
    if not gen:
        return
    text_val = _generation_text_value(gen)
    gen_tokens = _generation_int_metric(gen, "generation_tokens") or 0
    prompt_tokens = _generation_int_metric(gen, "prompt_tokens")
    if analysis is None:
        analysis = analyze_generation_text(
            text_val,
            gen_tokens,
            prompt_tokens=prompt_tokens,
            prompt=prompt,
            context_marker=context_marker,
        )

    if not text_val:
        if analysis.has_harness_issue:
            details = ", ".join(analysis.harness_issue_details[:2])
            log_warning_note(
                f"Likely harness issue ({analysis.harness_issue_type}): {details}",
            )
        log_metric_label("Generated Text:", emoji="📝")
        logger.info(
            "<empty>",
            extra={"style_hint": LogStyles.GENERATED_TEXT},
        )
        return

    # Show brief inline warnings for quality issues
    if analysis.is_repetitive and analysis.repeated_token:
        log_warning_note(f"Repetitive: '{analysis.repeated_token}'")
    if analysis.hallucination_issues:
        issues_preview = ", ".join(analysis.hallucination_issues[:2])
        log_warning_note(issues_preview)
    if analysis.is_verbose:
        log_warning_note(f"Verbose ({gen_tokens} tokens)")
    if analysis.formatting_issues:
        log_warning_note(analysis.formatting_issues[0])
    if analysis.is_context_ignored and analysis.missing_context_terms:
        missing = ", ".join(analysis.missing_context_terms[:3])
        log_warning_note(f"Context ignored (missing: {missing})")
    if analysis.missing_sections:
        missing = ", ".join(analysis.missing_sections)
        log_warning_note(f"Missing sections: {missing}")
    if analysis.has_reasoning_leak:
        log_warning_note("Reasoning/prompt text leaked into output")
    if analysis.has_context_echo:
        log_warning_note(f"Context echo ({analysis.context_echo_ratio:.0%} overlap)")
    if analysis.has_harness_issue:
        details = ", ".join(analysis.harness_issue_details[:2])
        log_warning_note(f"Harness issue ({analysis.harness_issue_type}): {details}")

    # Show full output in trace (truncated in summary table)
    log_metric_label("Generated Text:", emoji="📝")
    log_generated_text(text_val, wrap=True)


def _log_verbose_success_details_mode(
    res: PerformanceResult,
    *,
    detailed: bool,
    analysis: GenerationQualityAnalysis | None = None,
    prompt: str | None = None,
    context_marker: str = "Context:",
) -> None:
    """Emit verbose block using either compact or detailed metrics style with visual hierarchy."""
    if not res.generation:
        return

    # Generated text with emoji prefix for easy scanning
    gen_text = getattr(res.generation, "text", None) or ""
    gen_tokens = getattr(res.generation, "generation_tokens", 0)

    if analysis is None:
        prompt_tokens = getattr(res.generation, "prompt_tokens", None)
        analysis = analyze_generation_text(
            gen_text,
            gen_tokens,
            prompt_tokens=prompt_tokens,
            prompt=prompt,
            context_marker=context_marker,
        )

    log_blank()
    log_metric_label("Generated Text:", emoji="📝")

    # Warn about quality issues
    if analysis.is_repetitive and analysis.repeated_token:
        warning_msg = (
            f"WARNING: Output appears to be rubbish (repetitive: '{analysis.repeated_token}')"
        )
        log_warning_note(warning_msg)

    if analysis.hallucination_issues:
        for issue in analysis.hallucination_issues:
            log_warning_note(issue, prefix="⚠️  Note:")

    if analysis.is_verbose:
        log_warning_note(
            f"Note: Output is excessively verbose ({gen_tokens} tokens)",
            prefix="⚠️",
        )

    if analysis.formatting_issues:
        for issue in analysis.formatting_issues[:2]:  # Show first 2 issues
            log_warning_note(issue, prefix="⚠️  Note:")

    if analysis.is_context_ignored and analysis.missing_context_terms:
        missing = ", ".join(analysis.missing_context_terms)
        log_warning_note(
            f"Note: Output ignored key context (missing: {missing})",
            prefix="⚠️",
        )
    if analysis.has_harness_issue:
        details = ", ".join(analysis.harness_issue_details[:3])
        log_warning_note(
            f"Likely harness issue ({analysis.harness_issue_type}): {details}",
            prefix="⚠️",
        )

    if gen_text:
        log_generated_text(gen_text, wrap=True, indent="   ")
    else:
        logger.info("   <empty>", extra={"style_hint": LogStyles.GENERATED_TEXT})

    if detailed:
        log_blank()
        log_metric_label("Performance Metrics:", emoji="📊")
        _log_token_summary(res)
        _log_detailed_timings(res)
        log_blank()
        _log_perf_block(res)
        log_blank()
        _log_additional_diagnostics(res, gen_text, prompt=prompt)
    else:
        _log_compact_metrics(res)


def _log_token_summary(res: PerformanceResult) -> None:
    """Log tokens and generation TPS with tree structure for visual hierarchy."""
    if res.generation is None:
        return

    p_tokens = _generation_int_metric(res.generation, "prompt_tokens") or 0
    g_tokens = _generation_int_metric(res.generation, "generation_tokens") or 0
    tot_tokens = (p_tokens or 0) + (g_tokens or 0)
    gen_tps = _generation_float_metric(res.generation, "generation_tps") or 0.0
    prompt_tps = _generation_float_metric(res.generation, "prompt_tps") or 0.0

    _log_metric_tree(
        "Tokens:",
        (
            ("Prompt:", f"{fmt_num(p_tokens):>8} @ {fmt_num(prompt_tps)} tok/s"),
            ("Generated:", f"{fmt_num(g_tokens):>8} @ {fmt_num(gen_tps)} tok/s"),
            ("Total:", f"{fmt_num(tot_tokens):>8}"),
        ),
        emoji="🔢",
    )


def _log_detailed_timings(res: PerformanceResult) -> None:
    """Log detailed runtime timings and termination metadata with tree structure."""
    total_time_val = getattr(res, "total_time", None)
    generation_time_val = getattr(res, "generation_time", None)
    model_load_time_val = getattr(res, "model_load_time", None)
    runtime = res.runtime_diagnostics

    if not total_time_val or total_time_val <= 0:
        return
    total_time_seconds = float(total_time_val)

    tt_val = format_field_value("total_time", total_time_val)
    tt_disp = tt_val if isinstance(tt_val, str) else _format_time_seconds(total_time_val)
    entries: list[tuple[str, str]] = [("Total:", f"{tt_disp:>8}")]

    def _append_phase_entry(
        *,
        label: str,
        value: float | None,
        field_name: str,
        include_pct: bool = True,
    ) -> None:
        if value is None or value <= 0:
            return
        formatted = format_field_value(field_name, value)
        display = formatted if isinstance(formatted, str) else _format_time_seconds(value)
        if include_pct:
            pct = value / total_time_seconds * 100
            entries.append((label, f"{display:>8} ({pct:>3.0f}%)"))
            return
        entries.append((label, f"{display:>8}"))

    _append_phase_entry(
        label="Generation:",
        value=generation_time_val,
        field_name="generation_time",
    )
    _append_phase_entry(
        label="Load:",
        value=model_load_time_val,
        field_name="model_load_time",
    )

    if runtime is not None:
        _append_phase_entry(
            label="Validation:",
            value=runtime.input_validation_time_s,
            field_name="total_time",
        )
        _append_phase_entry(
            label="Prompt prep:",
            value=runtime.prompt_prep_time_s,
            field_name="total_time",
        )
        _append_phase_entry(
            label="Cleanup:",
            value=runtime.cleanup_time_s,
            field_name="total_time",
        )
        _append_phase_entry(
            label="First token:",
            value=runtime.first_token_latency_s,
            field_name="total_time",
            include_pct=False,
        )
        if runtime.stop_reason:
            entries.append(("Stop reason:", runtime.stop_reason))

    _log_metric_tree("Timing:", entries, emoji="⏱")


def _log_perf_block(res: PerformanceResult) -> None:
    """Log inner performance metrics (memory) with tree structure and emoji."""
    active_mem = getattr(res.generation, "active_memory", 0.0) or 0.0
    cached_mem = getattr(res.generation, "cache_memory", None)
    if not isinstance(cached_mem, int | float):
        cached_mem = getattr(res.generation, "cached_memory", 0.0)
    cached_mem = float(cached_mem or 0.0)
    peak_mem = getattr(res.generation, "peak_memory", 0.0) or 0.0

    # Only show memory section if at least one value is present
    if active_mem <= 0 and cached_mem <= 0 and peak_mem <= 0:
        return

    entries: list[MetricTreeRow] = []

    def _append_mem(label: str, field: str, raw_val: float) -> None:
        if raw_val <= 0:
            return
        formatted = format_field_value(field, raw_val)
        unit = "GB"
        text = formatted if formatted.endswith(unit) else f"{formatted} GB"
        entries.append((label, f"{text:>8}"))

    _append_mem("Active Δ:", "active_memory", active_mem)
    _append_mem("Cache Δ:", "cache_memory", cached_mem)
    _append_mem("Peak:", "peak_memory", peak_mem)
    _log_metric_tree("Memory:", entries, emoji="💾")


def _log_output_analysis(
    gen_text: str,
    gen_tokens: int,
    generation_time: float,
    peak_mem: float,
) -> None:
    """Log output analysis section: vocabulary, efficiency, structure, confidence."""
    entries: list[MetricTreeRow] = []

    # Vocabulary diversity
    ttr, unique_words, total_words = compute_vocabulary_diversity(gen_text)
    entries.append(
        (
            "Vocabulary:",
            f"TTR={ttr:.2f} ({unique_words}/{total_words} unique words)",
        )
    )

    # Efficiency metrics
    efficiency = compute_efficiency_metrics(gen_tokens, generation_time, peak_mem)
    if efficiency["tokens_per_second_per_gb"]:
        entries.append(
            (
                "Efficiency:",
                f"{efficiency['tokens_per_second_per_gb']:.1f} tok/s/GB",
            )
        )

    # Response structure
    structure = detect_response_structure(gen_text)
    structure_parts: list[str] = []
    if structure["has_caption"]:
        structure_parts.append("caption")
    if structure["has_keywords"]:
        structure_parts.append("keywords")
    if structure["has_description"]:
        structure_parts.append("description")
    if structure["has_sections"]:
        structure_parts.append("sections")

    structure_str = ", ".join(structure_parts) if structure_parts else "unstructured"
    entries.append(("Structure:", structure_str))

    # Confidence indicators
    confidence = compute_confidence_indicators(gen_text)
    conf_ratio = confidence["confidence_ratio"]
    if conf_ratio > QUALITY.high_confidence_threshold:
        conf_label = "high"
    elif conf_ratio > QUALITY.medium_confidence_threshold:
        conf_label = "medium"
    else:
        conf_label = "low"
    entries.append(("Confidence:", f"{conf_label} ({conf_ratio:.0%})"))
    _log_metric_tree("Output Analysis:", entries, emoji="🔍")


def _get_grade_display(grade: str) -> str:
    """Return emoji-decorated grade display string."""
    emoji = GRADE_EMOJIS.get(grade, "❌")
    return f"{emoji} {grade}"


def _log_cataloging_utility(gen_text: str, context: str | None) -> None:
    """Log cataloging utility metrics section."""
    entries: list[MetricTreeRow] = []

    # Information gain
    info_gain = compute_information_gain(gen_text, context)
    echo_ratio = info_gain["echo_ratio"]
    entries.append(
        (
            "Info Gain:",
            f"{info_gain['information_gain']:.0%} novel "
            f"({info_gain['novel_words']}/{info_gain['output_words']} words)",
        )
    )
    if echo_ratio > QUALITY.moderate_echo_threshold:
        entries.append(("Context echo:", f"⚠️  {echo_ratio:.0%} echoed from context"))

    # Task compliance
    compliance = compute_task_compliance(gen_text)
    compliance_parts = [
        "✓ caption" if compliance["has_caption"] else "✗ caption",
        "✓ desc" if compliance["has_description"] else "✗ desc",
        "✓ keywords" if compliance["has_keywords"] else "✗ keywords",
    ]
    entries.append(
        (
            "Compliance:",
            f"{', '.join(compliance_parts)} ({compliance['compliance_score']:.0%})",
        )
    )

    description = compute_description_quality(gen_text, context)
    entries.append(
        (
            "Description:",
            f"{description['description_score']:.0f}/100 "
            f"({description['description_word_count']} words, "
            f"{description['description_sentence_count']} sentences, "
            f"grounding={description['description_grounding_score']:.0%})",
        )
    )

    keywords = compute_keyword_quality(gen_text, context)
    entries.append(
        (
            "Keywords:",
            f"{keywords['keyword_score']:.0f}/100 "
            f"({keywords['keyword_term_count']} terms, "
            f"{keywords['keyword_unique_terms']} unique, "
            f"coverage={keywords['keyword_category_coverage']:.0%})",
        )
    )

    grounding = compute_visual_grounding(gen_text, context)
    utility = compute_cataloging_utility(
        gen_text,
        context,
        info_gain=info_gain,
        task_compliance=compliance,
        visual_grounding=grounding,
    )
    grade = str(utility["utility_grade"])
    grade_display = _get_grade_display(grade)
    entries.append(
        (
            "UTILITY:",
            f"{grade_display} ({utility['utility_score']:.0f}/100) - {utility['primary_weakness']}",
        )
    )
    _log_metric_tree("Cataloging Utility:", entries, emoji="📚")


def _log_additional_diagnostics(
    res: PerformanceResult,
    gen_text: str,
    *,
    prompt: str | None = None,
) -> None:
    """Log additional output diagnostics for detailed metrics mode.

    Displays:
    - Vocabulary diversity (type-token ratio)
    - Efficiency metrics (tokens per GB)
    - Response structure indicators
    - Confidence indicators
    - Cataloging utility metrics (information gain, task compliance, visual grounding)
    """
    if not gen_text:
        return

    gen_tokens = getattr(res.generation, "generation_tokens", 0) or 0
    generation_time = res.generation_time or 0.0
    peak_mem = getattr(res.generation, "peak_memory", 0.0) or 0.0

    _log_output_analysis(gen_text, gen_tokens, generation_time, peak_mem)

    log_blank()
    context = _extract_trusted_hint_bundle(prompt).trusted_text or None if prompt else None
    _log_cataloging_utility(gen_text, context)


def _log_compact_metrics(res: PerformanceResult) -> None:
    """Emit two-line metrics for improved scannability.

    Example output:
        📊 Timing: 5.41s total (gen=4.53s, load=0.88s) | Memory: 5.5GB peak
           Tokens: 1,759 (1,442 prompt + 317 gen) | Speed: 114 gen/s, 1,231 prompt/s
    """
    if not res.generation:
        return

    log_blank()  # Breathing room
    gen = res.generation

    # Extract values
    total_time = getattr(res, "total_time", None)
    gen_time = getattr(res, "generation_time", None)
    load_time = getattr(res, "model_load_time", None)
    runtime = res.runtime_diagnostics
    peak_mem = _generation_float_metric(gen, "peak_memory") or 0.0
    prompt_tokens = _generation_int_metric(gen, "prompt_tokens") or 0
    gen_tokens = _generation_int_metric(gen, "generation_tokens") or 0
    gen_tps = _generation_float_metric(gen, "generation_tps") or 0.0
    prompt_tps = _generation_float_metric(gen, "prompt_tps") or 0.0

    # Line 1: Timing and Memory
    timing_parts: list[str] = []
    if total_time is not None:
        sub_parts: list[str] = []
        if gen_time is not None:
            sub_parts.append(f"gen={_format_time_seconds(gen_time)}")
        if load_time is not None:
            sub_parts.append(f"load={_format_time_seconds(load_time)}")
        if runtime is not None:
            prompt_prep_time = runtime.prompt_prep_time_s
            if prompt_prep_time is not None and prompt_prep_time > 0:
                sub_parts.append(f"prep={_format_time_seconds(prompt_prep_time)}")
            first_token_latency = runtime.first_token_latency_s
            if first_token_latency is not None and first_token_latency > 0:
                sub_parts.append(f"first={_format_time_seconds(first_token_latency)}")
            if runtime.stop_reason and runtime.stop_reason != "completed":
                sub_parts.append(f"stop={runtime.stop_reason}")
        breakdown = f" ({', '.join(sub_parts)})" if sub_parts else ""
        timing_parts.append(f"{_format_time_seconds(total_time)} total{breakdown}")

    mem_part = ""
    if peak_mem > 0:
        mem_fmt = format_field_value("peak_memory", peak_mem)
        mem_str = f"{mem_fmt}GB" if not mem_fmt.endswith("GB") else mem_fmt
        mem_part = f" | Memory: {mem_str} peak"

    line1 = f"📊 Timing: {timing_parts[0] if timing_parts else NOT_AVAILABLE}{mem_part}"
    logger.info(line1, extra={"style_hint": LogStyles.METRIC_LABEL})

    # Line 2: Tokens and Speed
    all_tokens = prompt_tokens + gen_tokens
    tokens_part = ""
    if all_tokens:
        tokens_part = (
            f"{fmt_num(all_tokens)} ({fmt_num(prompt_tokens)} prompt + {fmt_num(gen_tokens)} gen)"
        )

    speed_parts: list[str] = []
    if gen_tps:
        speed_parts.append(f"{fmt_num(gen_tps)} gen/s")
    if prompt_tps:
        speed_parts.append(f"{fmt_num(prompt_tps)} prompt/s")
    speed_part = f" | Speed: {', '.join(speed_parts)}" if speed_parts else ""

    if tokens_part or speed_part:
        line2 = f"   Tokens: {tokens_part}{speed_part}"
        logger.info(line2, extra={"style_hint": LogStyles.METRIC_LABEL})

    if (
        prompt_tokens >= QUALITY.long_prompt_tokens_threshold
        and gen_tokens < QUALITY.min_output_tokens_for_ratio
    ):
        ratio = gen_tokens / max(prompt_tokens, 1)
        log_warning_note(
            "Potential long-context degradation: "
            f"prompt={fmt_num(prompt_tokens)} tok, "
            f"output={fmt_num(gen_tokens)} tok ({ratio:.1%})",
        )


def log_metrics_legend(*, detailed: bool) -> None:
    """Emit a one-time legend at the beginning of processing for clarity."""
    log_blank()
    mode_line = (
        "Detailed mode: separate lines for timing, memory, tokens, TPS"
        if detailed
        else "Compact mode: tokens(total/prompt/gen) format with aligned keys"
    )
    panel = Panel(
        f"{mode_line}\nWarnings are shown for repetitive or hallucinated output.",
        title="📖 Metrics Legend",
        border_style="blue",
        box=box.ROUNDED,
    )
    _log_rich_renderable(panel, width=get_terminal_width(max_width=100))
    log_blank()


def _log_failure_details(result: PerformanceResult) -> None:
    """Emit actionable failure details for issue reporting."""
    if result.error_message:
        logger.info("Error: %s", result.error_message)

    if result.error_package:
        logger.info(
            "Error package: %s",
            result.error_package,
            extra={"style_hint": LogStyles.DETAIL},
        )
    if result.error_type:
        logger.info(
            "Error type: %s",
            result.error_type,
            extra={"style_hint": LogStyles.DETAIL},
        )

    traceback_tail = _format_traceback_tail(result.error_traceback)
    if traceback_tail:
        _log_wrapped_error("Traceback tail:", traceback_tail)

    captured_output = (result.captured_output_on_fail or "").strip()
    if captured_output:
        preview = _truncate_text_preview(
            captured_output,
            max_chars=MAX_CAPTURED_OUTPUT_LOG_CHARS,
        )
        _log_wrapped_error("Captured output:", preview)


def print_model_result(
    result: PerformanceResult,
    *,
    verbose: bool = False,
    detailed_metrics: bool = False,
    run_index: int | None = None,
    total_runs: int | None = None,
    prompt: str | None = None,
    context_marker: str = "Context:",
) -> None:
    """Print a concise summary + optional verbose block for a model result."""
    run_prefix = "" if run_index is None else f"[RUN {run_index}/{total_runs}] "
    summary = run_prefix + "SUMMARY " + " ".join(_summary_parts(result, result.model_name))
    # Wrap summary to terminal width for readability
    width = get_terminal_width(max_width=100)
    for line in textwrap.wrap(summary, width=width, break_long_words=False, break_on_hyphens=False):
        if result.success:
            log_success(line, prefix="")
        else:
            log_failure(line, prefix="")
    if result.success and not verbose:  # quick exit with preview only
        _preview_generation(
            result.generation,
            analysis=result.quality_analysis,
            prompt=prompt,
            context_marker=context_marker,
        )
        return
    # For failures, show detailed error info; for success, show generation details
    if not result.success:
        log_blank()  # Single blank before error details
        _log_failure_details(result)
        return
    if result.generation and verbose:
        _log_verbose_success_details_mode(
            result,
            detailed=detailed_metrics,
            analysis=result.quality_analysis,
            prompt=prompt,
            context_marker=context_marker,
        )


def print_cli_separator() -> None:
    """Print a visually distinct separator line using unicode box-drawing characters."""
    width = get_terminal_width(max_width=100)
    log_rule(width, char="─", style="blue")


def _dump_environment_to_log(output_path: Path) -> None:
    """Dump complete Python environment to separate file for debugging/reproducibility.

    Captures output from pip freeze (and conda list if in conda environment)
    to provide complete package manifest for issue reproduction.

    Args:
        output_path: Path where the environment log should be written
    """
    try:
        # Detect if we're in a conda environment
        conda_env = os.environ.get("CONDA_DEFAULT_ENV")
        # Ensure output directory exists
        env_log_path = output_path.resolve()
        env_log_path.parent.mkdir(parents=True, exist_ok=True)

        with env_log_path.open("w", encoding="utf-8") as env_file:
            env_file.write("=" * 80 + "\n")
            env_file.write(f"FULL ENVIRONMENT DUMP - {local_now_str()}\n")
            env_file.write("=" * 80 + "\n\n")

            # Use importlib.metadata (standard library) instead of subprocess calling pip/conda
            # to avoid S603 security lints and provide faster, more reliable dumping.
            try:
                # Some environments can have incomplete distribution metadata missing .name
                def _get_name(d: Distribution) -> str:
                    return getattr(d, "name", "") or ""

                dists = sorted(
                    distributions(),
                    key=lambda d: _get_name(d).lower(),
                )
                env_file.write("--- Python Packages (via importlib.metadata) ---\n")
                env_file.write(f"Total packages: {len(dists)}\n\n")
                for d in dists:
                    env_file.write(f"{d.name}=={d.version}\n")
                env_file.write("\n")
            except (OSError, ValueError, RuntimeError) as dist_err:
                env_file.write(f"Could not gather package list: {dist_err}\n")

            # Log conda environment name if applicable
            if conda_env:
                env_file.write(f"Conda Environment: {conda_env}\n")

            env_file.write("=" * 80 + "\n")
            env_file.write(f"Environment dump completed at {local_now_str()}\n")
            env_file.write("=" * 80 + "\n")

        # Log single line pointing to the file
        log_metric_label("Full environment dump written to:", emoji="📝")
        log_file_path(str(env_log_path))
        logger.debug("Environment details saved for reproducibility")

    except (OSError, FileNotFoundError, subprocess.SubprocessError) as e:
        logger.warning("Failed to dump environment info: %s", e)


def setup_environment(args: argparse.Namespace) -> LibraryVersionDict:
    """Configure logging, collect versions, print warnings."""
    # Apply CLI output preferences before constructing the Rich console handler.
    _apply_cli_output_preferences(args)

    # Set DEBUG if verbose, else INFO
    console_log_level: int = logging.DEBUG if args.verbose else logging.INFO
    # Remove all handlers and add console + file handlers
    logger.handlers.clear()

    # Rich console handler handles terminal width, color/no-color mode, and tracebacks.
    console_handler = _make_console_log_handler(
        level=console_log_level,
        verbose=bool(args.verbose),
        width=get_terminal_width(max_width=120),
    )
    logger.addHandler(console_handler)

    # File handler - write to specified log file (overwritten each run)
    log_file: Path = args.output_log.resolve()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler: logging.FileHandler = logging.FileHandler(
        log_file,
        mode="w",
        encoding="utf-8",
    )
    # File gets full timestamp + level always (no colors in file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter: logging.Formatter = FileSafeFormatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt=LOCAL_TIMESTAMP_FORMAT,
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Logger captures everything so file handler gets debug info
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # Prevent double logging

    if args.verbose:
        logger.debug("Verbose/debug mode enabled.")

    # Dump full environment to log file for reproducibility (after logging setup)
    _dump_environment_to_log(args.output_env)

    # Note extra framework installs that are not part of the normal MLX path.
    st_present = bool(find_spec("sentence_transformers"))
    if st_present:
        logger.debug(
            "Detected 'sentence-transformers'. It's not used here by default and may "
            "import heavy backends.",
        )

    library_versions: LibraryVersionDict = get_library_versions()
    preflight_issues = _collect_preflight_package_issues(library_versions)
    _set_run_preflight_issues(args, preflight_issues)
    if preflight_issues:
        logger.warning("Detected upstream package compatibility risks:")
        for issue in preflight_issues:
            logger.warning("  - %s", issue)

    if args.verbose:
        print_version_info(library_versions)

    if args.trust_remote_code:
        print_cli_separator()
        log_warning_note("SECURITY WARNING: --trust-remote-code is enabled.")
        log_warning_note("This allows execution of remote code and may pose security risks.")

    return library_versions


def _set_run_preflight_issues(args: argparse.Namespace, issues: Sequence[str]) -> None:
    """Store preflight warning strings on argparse namespace for later reporting."""
    setattr(args, _PREFLIGHT_ISSUES_ARG_ATTR, tuple(issue for issue in issues))


def _get_run_preflight_issues(args: argparse.Namespace | None) -> tuple[str, ...]:
    """Read preflight warning strings from argparse namespace."""
    if args is None:
        return ()
    raw = getattr(args, _PREFLIGHT_ISSUES_ARG_ATTR, ())
    if isinstance(raw, tuple):
        return tuple(str(item) for item in raw)
    if isinstance(raw, list):
        return tuple(str(item) for item in raw)
    return ()


def _raise_for_missing_runtime_dependencies() -> None:
    """Raise a fatal runtime error when core inference dependencies are unavailable."""
    transformers_version = get_library_versions().get("transformers")
    if transformers_version is None:
        MISSING_DEPENDENCIES.setdefault(
            "transformers",
            (
                "Core dependency missing: transformers. "
                f"Please install transformers>={PROJECT_MIN_TRANSFORMERS_VERSION}."
            ),
        )
    elif not _is_version_at_least(transformers_version, PROJECT_MIN_TRANSFORMERS_VERSION):
        MISSING_DEPENDENCIES["transformers"] = (
            f"Core dependency too old: transformers=={transformers_version}. "
            f"Need transformers>={PROJECT_MIN_TRANSFORMERS_VERSION}."
        )

    if not MISSING_DEPENDENCIES:
        return

    for dependency_name, message in sorted(MISSING_DEPENDENCIES.items()):
        logger.critical("[%s] %s", dependency_name, message)
    missing_list = ", ".join(sorted(MISSING_DEPENDENCIES))
    error_message = (
        f"Required runtime dependencies unavailable: {missing_list}. "
        "Install/repair these packages before running model checks."
    )
    raise RuntimeError(error_message)


def find_and_validate_image(args: argparse.Namespace) -> Path:
    """Find and validate the image file to process from arguments."""
    if getattr(args, "image", None) is not None:
        img_path: Path = args.image.resolve()
        log_file_path(str(img_path), label="Image File:     ")
        try:
            with Image.open(img_path) as img:
                img.verify()
            print_image_dimensions(img_path)
        except (
            FileNotFoundError,
            UnidentifiedImageError,
            OSError,
        ) as img_err:
            exit_with_cli_error(
                f"Cannot open or verify image {img_path}: {img_err}. Exiting.",
                suppress_cause=True,
            )
        else:
            return img_path
    else:
        folder_path: Path = args.folder.resolve()
        log_file_path(str(folder_path), label="Scanning folder:")
        if not folder_path.is_dir():
            exit_with_cli_error(f"Folder '{folder_path}' does not exist. Exiting.")
        most_recent_path: Path | None = find_most_recent_file(folder_path)
        if most_recent_path is None:
            exit_with_cli_error(
                f"Could not find the most recent image file in {folder_path}. Exiting.",
            )
            raise SystemExit  # pragma: no cover
        resolved_image_path: Path = most_recent_path.resolve()
        log_file_path(str(resolved_image_path), label="Image File:     ")
        try:
            with Image.open(resolved_image_path) as img:
                img.verify()
            print_image_dimensions(resolved_image_path)
        except (
            FileNotFoundError,
            UnidentifiedImageError,
            OSError,
        ) as img_err:
            exit_with_cli_error(
                f"Cannot open or verify image {resolved_image_path}: {img_err}. Exiting.",
                suppress_cause=True,
            )
        else:
            return resolved_image_path


def handle_metadata(image_path: Path, args: argparse.Namespace) -> MetadataDict:
    """Extract, print, and return image metadata."""
    print_cli_section("Image Metadata")

    # Single EXIF extraction shared by metadata and verbose pretty-print
    exif_data: ExifDict | None = get_exif_data(image_path)
    metadata: MetadataDict = extract_image_metadata(image_path, exif_data=exif_data)

    # Display key metadata (fallback to empty at presentation time)
    logger.info("Date: %s", metadata.get("date") or "")
    logger.info("Description: %s", metadata.get("description") or "")
    logger.info("GPS Location: %s", metadata.get("gps") or "")
    if metadata.get("title"):
        logger.info("Title: %s", metadata["title"])
    if metadata.get("keywords"):
        logger.info("Keywords: %s", metadata["keywords"])

    if args.verbose:
        # Reuse already-extracted EXIF data (no second image open)
        if exif_data:
            pretty_print_exif(exif_data, show_all=True)
        else:
            logger.warning("No detailed EXIF data could be extracted.")
    return metadata


def _compact_prompt_text(value: str, *, max_chars: int) -> str:
    """Normalize whitespace and clip prompt context fields to a safe size."""
    compact = re.sub(r"\s+", " ", value).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _summarize_prompt_keywords(raw_keywords: str) -> str:
    """Return a deduplicated, size-limited keyword hint string for the prompt."""
    deduped_items: list[str] = []
    seen: set[str] = set()
    for item in raw_keywords.split(","):
        compact_item = _compact_prompt_text(
            item,
            max_chars=QUALITY.prompt_keyword_item_max_chars,
        )
        if not compact_item:
            continue
        key = compact_item.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped_items.append(compact_item)
        if len(deduped_items) >= QUALITY.prompt_keyword_max_items:
            break
    return ", ".join(deduped_items)


def _build_cataloguing_prompt(metadata: MetadataDict) -> str:
    """Build a structured prompt optimised for stock-photo cataloguing.

    Keeps instructions concise while preserving metadata grounding from
    IPTC/XMP/EXIF fields. The ``Context:`` marker is retained for
    context-ignorance diagnostics.
    """
    parts: list[str] = [
        "Analyze this image for cataloguing metadata, using British English.",
        "",
        (
            "Use only details that are clearly and definitely visible in the image. "
            "If a detail is uncertain, ambiguous, partially obscured, too small to "
            "verify, or not directly visible, leave it out. Do not guess."
        ),
        "",
        (
            "Treat the metadata hints below as a draft catalog record. Keep only details "
            "that are clearly confirmed by the image, correct anything contradicted by "
            "the image, and add important visible details that are definitely present."
        ),
        "",
        "Return exactly these three sections, and nothing else:",
        "",
        "Title:",
        "- 5-10 words, concrete and factual, limited to clearly visible content.",
        "- Output only the title text after the label.",
        "- Do not repeat or paraphrase these instructions in the title.",
        "",
        "Description:",
        (
            "- 1-2 factual sentences describing the main visible subject, "
            "setting, lighting, action, and other distinctive visible details. Omit "
            "anything uncertain or inferred."
        ),
        "- Output only the description text after the label.",
        "",
        "Keywords:",
        (
            "- 10-18 unique comma-separated terms based only on clearly visible "
            "subjects, setting, colors, composition, and style. Omit uncertain tags "
            "rather than guessing."
        ),
        "- Output only the keyword list after the label.",
        "",
        "Rules:",
        "- Include only details that are definitely visible in the image.",
        "- Reuse metadata terms only when they are clearly supported by the image.",
        "- If metadata and image disagree, follow the image.",
        "- Prefer omission to speculation.",
        "- Do not copy prompt instructions into the Title, Description, or Keywords fields.",
        (
            "- Do not infer identity, location, event, brand, species, time period, "
            "or intent unless visually obvious."
        ),
        "- Do not output reasoning, notes, hedging, or extra sections.",
    ]

    # --- Context block (uses the "Context:" marker for quality analysis) ---
    desc = metadata.get("description")
    title = metadata.get("title")
    existing_kw = metadata.get("keywords")
    has_context = desc or title or existing_kw
    if has_context:
        parts.append("")
        parts.append(
            "Context: Existing metadata hints (high confidence; use only when visually confirmed):",
        )
        if title:
            title_hint = _compact_prompt_text(title, max_chars=QUALITY.prompt_title_max_chars)
            parts.append(f"- Title hint: {title_hint}")
        if desc:
            desc_hint = _compact_prompt_text(
                desc,
                max_chars=QUALITY.prompt_description_max_chars,
            )
            parts.append(f"- Description hint: {desc_hint}")
        if existing_kw:
            keyword_hint = _summarize_prompt_keywords(existing_kw)
            if keyword_hint:
                parts.append(f"- Keyword hints: {keyword_hint}")

    # Date / time / GPS metadata
    date_val = metadata.get("date")
    time_val = metadata.get("time")
    gps_val = metadata.get("gps")
    if date_val or gps_val:
        meta_fragments: list[str] = []
        if date_val:
            s = f"Taken on {date_val}"
            if time_val:
                s += f" (at {time_val} local time)"
            meta_fragments.append(s)
        if gps_val:
            meta_fragments.append(f"GPS: {gps_val}")
        if has_context:
            parts.append("- Capture metadata: " + ". ".join(meta_fragments) + ".")
        else:
            parts.append("")
            parts.append("Capture metadata hints: " + ". ".join(meta_fragments) + ".")

    return "\n".join(parts)


def prepare_prompt(args: argparse.Namespace, metadata: MetadataDict) -> str:
    """Prepare the prompt for the VLM, using user input or generating from metadata."""
    print_cli_section("Prompt Configuration")
    max_display_len = 200

    eval_mode = getattr(args, "eval_mode", "stress")

    prompt: str
    if args.prompt:
        prompt = args.prompt
        logger.info("Using user-provided prompt from --prompt.")
        logger.info(
            "User-provided prompt (--prompt): %s",
            _build_prompt_preview(prompt, max_chars=max_display_len),
        )
    elif eval_mode == "triage":
        prompt = "Describe this image briefly."
        logger.info("Using triage-mode prompt (minimal context).")
    else:
        logger.info("Generating default prompt based on image metadata.")
        prompt = _build_cataloguing_prompt(metadata)
        logger.debug("Using generated prompt based on metadata.")
        logger.info(
            "Final prompt: %s",
            _build_prompt_preview(prompt, max_chars=max_display_len),
        )
    logger.debug("Full prompt:\n%s", prompt)
    logger.info("Prompt length: %d characters", len(prompt))
    return prompt


def get_cached_model_ids() -> list[str]:
    """Return a list of model IDs found in the Hugging Face cache."""
    try:
        cache_info = _get_hf_cache_info_cached()
    except HFValidationError:
        logger.warning("Hugging Face cache directory invalid.")
        return []
    except FileNotFoundError:
        logger.warning("Hugging Face cache directory not found.")
        return []
    except (OSError, ValueError) as e:
        logger.warning("Unexpected error scanning Hugging Face cache: %s", e)
        return []
    else:
        return sorted([repo.repo_id for repo in cache_info.repos])


def validate_model_identifier(model_id: str) -> None:
    """Validate model ID is well-formed (org/name format or existing local path).

    Args:
        model_id: Model identifier to validate

    Raises:
        ValueError: If model_id is empty, malformed, or local path doesn't exist

    """
    if not model_id or not model_id.strip():
        msg = "Model identifier cannot be empty"
        raise ValueError(msg)

    # Check if it's a local path (starts with / or ./ or ../)
    if model_id.startswith(("/", "./", "../")):
        model_path = Path(model_id)
        if not model_path.exists():
            msg = f"Local model path does not exist: {model_id}"
            raise ValueError(msg)
        if not model_path.is_dir():
            msg = f"Local model path is not a directory: {model_id}"
            raise ValueError(msg)
    elif " " in model_id:
        # Hub identifier: basic sanity checks
        msg = f"Model identifier contains spaces: '{model_id}'"
        raise ValueError(msg)
        # Optionally check for org/name format (though single names are valid too)
        # For now, just ensure it's not obviously malformed


def validate_and_warn_model_selection(args: argparse.Namespace) -> None:
    """Validate exclusions against local cache membership and current selection.

    Args:
        args: Parsed command line arguments

    """
    if not args.exclude:
        return  # No exclusions to validate

    cached_models = set(get_cached_model_ids())
    excluded_models: set[str] = set(args.exclude)
    ineffective_exclusions: set[str] = excluded_models - cached_models

    if ineffective_exclusions:
        ineffective_list = sorted(ineffective_exclusions)
        logger.warning(
            "The following excluded models are not in the local cache and will have no effect: %s",
            ", ".join(ineffective_list),
        )

    if not args.verbose:
        return

    if args.models:
        effective_exclusions = excluded_models & set(args.models)
        if effective_exclusions:
            logger.info(
                "Effective exclusions for this run (selected models that will be filtered out): %s",
                ", ".join(sorted(effective_exclusions)),
            )
        return

    effective_exclusions = excluded_models & cached_models
    if effective_exclusions:
        logger.info(
            "Effective exclusions (models that will be filtered out): %s",
            ", ".join(sorted(effective_exclusions)),
        )


def apply_exclusions(
    model_list: list[str],
    exclude_list: list[str],
    context: str,
) -> list[str]:
    """Apply exclusion list to models and log results.

    Args:
        model_list: List of model identifiers to filter
        exclude_list: List of models to exclude
        context: Description for logging (e.g., "explicit list", "cached models")

    Returns:
        Filtered list with excluded models removed

    """
    if not exclude_list:
        return model_list

    excluded_set = set(exclude_list)
    original_count = len(model_list)
    filtered = [model for model in model_list if model not in excluded_set]
    excluded_count = original_count - len(filtered)

    if excluded_count > 0:
        logger.info(
            "Excluded %d model(s) from %s. Remaining: %d model(s)",
            excluded_count,
            context,
            len(filtered),
        )

    return filtered


def process_models(
    args: argparse.Namespace,
    image_path: Path,
    *,  # Force keyword-only arguments for clarity
    prompt: str,
    metadata: MetadataDict | None = None,
) -> list[PerformanceResult]:
    """Resolve the definitive model list and execute each model run.

    Selection logic:
        * If --models provided: start with that list; optionally filter via --exclude.
        * Else: enumerate cached model repo IDs and apply --exclude.
    Each resolved identifier is processed sequentially (future work: parallel
    scheduling once thread/process safety of underlying libs confirmed).
    """
    # Validate model selection and warn about ineffective exclusions
    validate_and_warn_model_selection(args)

    model_identifiers: list[str]
    if args.models:
        # Case 1: Explicit models specified - apply exclusions to this list
        model_identifiers = args.models
        logger.info("Processing specified models: %s", ", ".join(model_identifiers))
        model_identifiers = apply_exclusions(
            model_identifiers,
            args.exclude or [],
            "explicit list",
        )
    else:
        # Case 2: No explicit models - scan cache and apply exclusions
        logger.info("Scanning cache for models to process...")
        model_identifiers = get_cached_model_ids()
        if not model_identifiers:
            exit_with_cli_error(
                "No models found in the local Hugging Face cache. "
                "Download a model (e.g., `huggingface-cli download mlx-community/<model>`) "
                "or pass explicit IDs with --models.",
            )
        model_identifiers = apply_exclusions(
            model_identifiers,
            args.exclude or [],
            "cached models",
        )

    results: list[PerformanceResult] = []
    if not model_identifiers:
        msg = "No models specified or found in cache."
        if not args.models:
            msg += " Ensure models are downloaded and cache is accessible."
        exit_with_cli_error(msg)
    else:
        logger.info("Processing %d model(s)...", len(model_identifiers))

        # Validate all model identifiers before processing
        # (Note: Sampling/KV params already validated in validate_cli_arguments)
        for model_id in model_identifiers:
            try:
                validate_model_identifier(model_id)
            except ValueError:
                logger.exception("Invalid model identifier '%s'", model_id)
                raise

    # Emit legend once if verbose
    if args.verbose:
        log_metrics_legend(detailed=args.detailed_metrics)

    for idx, model_id in enumerate(model_identifiers, start=1):
        print_cli_separator()
        log_blank()  # Add visual separation between model runs
        # Use full model ID (e.g. "mlx-community/Qwen2-VL-2B-Instruct") instead of just the name
        model_label = model_id
        run_label = f"[{idx}/{len(model_identifiers)}]"
        # Compact logging for model header
        log_model_name(model_label, label=f"Processing Model {run_label}:")

        is_vlm_verbose: bool = args.verbose
        params = ProcessImageParams(
            model_identifier=model_id,
            image_path=image_path,
            prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            verbose=is_vlm_verbose,
            trust_remote_code=args.trust_remote_code,
            top_p=args.top_p,
            min_p=args.min_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            repetition_context_size=args.repetition_context_size,
            lazy=args.lazy_load,
            max_kv_size=args.max_kv_size,
            kv_bits=args.kv_bits,
            kv_quant_scheme=args.kv_quant_scheme,
            kv_group_size=args.kv_group_size,
            quantized_kv_start=args.quantized_kv_start,
            revision=args.revision,
            adapter_path=args.adapter_path,
            prefill_step_size=args.prefill_step_size,
            resize_shape=args.resize_shape,
            eos_tokens=args.eos_tokens,
            skip_special_tokens=args.skip_special_tokens,
            processor_kwargs=args.processor_kwargs,
            enable_thinking=args.enable_thinking,
            thinking_budget=args.thinking_budget,
            thinking_start_token=args.thinking_start_token,
            thinking_end_token=args.thinking_end_token,
            context_marker=args.context_marker,
        )
        result: PerformanceResult = process_image_with_model(params)

        # Calculate and log quality score for successful generations.
        if result.success and result.generation:
            result = _populate_result_quality_analysis(
                result,
                prompt=prompt,
                metadata=metadata,
                requested_max_tokens=args.max_tokens,
                context_marker=args.context_marker,
            )
            analysis = result.quality_analysis
            if analysis is None:
                msg = f"Quality analysis missing for successful result {result.model_name}"
                raise RuntimeError(msg)
            # Log quality analysis results at DEBUG level
            logger.debug(
                "Quality analysis for %s: %s",
                result.model_name,
                _format_quality_analysis_for_log(analysis),
            )
            if result.quality_issues:
                logger.info(
                    "Quality issues detected for %s: %s",
                    result.model_name,
                    result.quality_issues,
                )

        results.append(result)
        _log_canonical_model_review(result)

        print_model_result(
            result,
            verbose=args.verbose,
            detailed_metrics=getattr(args, "detailed_metrics", False),
            run_index=idx,
            total_runs=len(model_identifiers),
            prompt=prompt,
            context_marker=args.context_marker,
        )
    return results


def _format_quality_analysis_for_log(analysis: GenerationQualityAnalysis) -> str:
    """Serialize quality-analysis flags into a compact log string.

    Includes only active flags plus lightweight counters to keep diagnostics
    readable in one line.
    """
    repetitive_part = (
        f"repetitive=True (token={analysis.repeated_token})"
        if analysis.is_repetitive and analysis.repeated_token
        else ("repetitive=True" if analysis.is_repetitive else None)
    )
    refusal_part = (
        f"refusal=True (type={analysis.refusal_type})"
        if analysis.is_refusal and analysis.refusal_type
        else ("refusal=True" if analysis.is_refusal else None)
    )
    harness_details = (
        ",".join(analysis.harness_issue_details[:2]) if analysis.harness_issue_details else ""
    )
    harness_part = (
        f"harness=True ({analysis.harness_issue_type}; {harness_details})"
        if analysis.has_harness_issue
        else None
    )
    reasoning_marker = (
        analysis.reasoning_leak_markers[0] if analysis.reasoning_leak_markers else "marker"
    )

    parts_raw: list[str | None] = [
        repetitive_part,
        refusal_part,
        "language_mixing=True" if analysis.has_language_mixing else None,
        "hallucination=True" if analysis.hallucination_issues else None,
        (f"generic=True (score={analysis.specificity_score:.1f})" if analysis.is_generic else None),
        "verbose=True" if analysis.is_verbose else None,
        "formatting_issues=True" if analysis.formatting_issues else None,
        (
            f"excessive_bullets=True (count={analysis.bullet_count})"
            if analysis.has_excessive_bullets
            else None
        ),
        "context_ignored=True" if analysis.is_context_ignored else None,
        (
            f"degeneration=True ({analysis.degeneration_type})"
            if analysis.has_degeneration
            else None
        ),
        "fabrication=True" if analysis.has_fabrication else None,
        (
            f"missing_sections={'+'.join(analysis.missing_sections)}"
            if analysis.missing_sections
            else None
        ),
        (
            f"title_words={analysis.title_word_count}"
            if analysis.title_word_count is not None
            else None
        ),
        (
            f"description_sentences={analysis.description_sentence_count}"
            if analysis.description_sentence_count is not None
            else None
        ),
        f"keywords={analysis.keyword_count}" if analysis.keyword_count is not None else None,
        (
            f"keyword_dup={analysis.keyword_duplication_ratio:.2f}"
            if analysis.keyword_duplication_ratio is not None
            else None
        ),
        f"reasoning_leak=True ({reasoning_marker})" if analysis.has_reasoning_leak else None,
        f"context_echo=True ({analysis.context_echo_ratio:.2f})"
        if analysis.has_context_echo
        else None,
        "instruction_echo=True" if analysis.instruction_echo else None,
        "metadata_borrowing=True" if analysis.metadata_borrowing else None,
        (
            f"hint_relationship={analysis.hint_relationship}"
            if analysis.hint_relationship != "preserves_trusted_hints"
            else None
        ),
        f"verdict={analysis.verdict}" if analysis.verdict != "clean" else None,
        f"user_bucket={analysis.user_bucket}" if analysis.user_bucket != "recommended" else None,
        "likely_capped=True" if analysis.likely_capped else None,
        harness_part,
    ]
    parts = [part for part in parts_raw if part is not None]
    parts.append(f"words={analysis.word_count}")

    return ", ".join(parts) if parts else "no issues detected"


def _build_quality_issues_string(analysis: GenerationQualityAnalysis) -> str | None:
    """Build prioritized, comma-separated issue labels from analysis flags.

    Ordering is intentional for triage severity: harness/integration issues
    first, then critical model quality issues, then lower-severity formatting.
    """
    issues: list[str] = []

    # HIGHEST PRIORITY: Harness/integration issues (mlx-vlm bugs, not model quality)
    # These get special prefix to clearly mark them as actionable infrastructure issues
    if analysis.has_harness_issue:
        harness_label = (
            f"⚠️harness({analysis.harness_issue_type})"
            if analysis.harness_issue_type
            else "⚠️harness"
        )
        issues.append(harness_label)
        if analysis.harness_issue_type == "long_context":
            issues.append("long-context")

    # Critical model quality issues
    refusal_label = f"refusal({analysis.refusal_type})" if analysis.refusal_type else "refusal"
    repetitive_label = (
        f"repetitive({analysis.repeated_token})" if analysis.repeated_token else "repetitive"
    )
    keyword_duplication_label = (
        f"keyword-duplication({analysis.keyword_duplication_ratio:.2f})"
        if analysis.keyword_duplication_ratio is not None
        else "keyword-duplication"
    )

    issue_candidates = [
        (analysis.is_refusal, refusal_label),
        (analysis.is_repetitive, repetitive_label),
        (analysis.has_language_mixing, "lang_mixing"),
        (bool(analysis.hallucination_issues), "hallucination"),
        (analysis.has_degeneration, "degeneration"),
        (analysis.has_fabrication, "fabrication"),
        (
            bool(analysis.missing_sections),
            f"missing-sections({'+'.join(analysis.missing_sections)})",
        ),
        (
            analysis.has_title_length_violation,
            f"title-length({analysis.title_word_count})",
        ),
        (
            analysis.has_description_sentence_violation,
            f"description-sentences({analysis.description_sentence_count})",
        ),
        (
            analysis.has_keyword_count_violation,
            f"keyword-count({analysis.keyword_count})",
        ),
        (
            analysis.has_keyword_duplication_violation,
            keyword_duplication_label,
        ),
        (analysis.has_reasoning_leak, "reasoning-leak"),
        (analysis.has_context_echo, f"context-echo({analysis.context_echo_ratio:.2f})"),
        (analysis.instruction_echo, "instruction-echo"),
        (analysis.metadata_borrowing, "metadata-borrowing"),
        (
            analysis.hint_relationship == "ignores_trusted_hints",
            "trusted-hints-ignored",
        ),
        (
            analysis.hint_relationship == "degrades_trusted_hints",
            "trusted-hints-degraded",
        ),
        (analysis.verdict == "cutoff_degraded", "cutoff"),
        (analysis.verdict == "token_cap", "token-cap"),
        (analysis.verdict == "context_budget", "context-budget"),
        (analysis.is_generic, f"generic({analysis.specificity_score:.0f})"),
        (analysis.is_verbose, "verbose"),
        (bool(analysis.formatting_issues), "formatting"),
        (analysis.has_excessive_bullets, f"bullets({analysis.bullet_count})"),
        (analysis.is_context_ignored, "context-ignored"),
    ]
    issues.extend(label for condition, label in issue_candidates if condition)

    return ", ".join(issues) if issues else None


QUALITY_ISSUE_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "harness": re.compile(r"⚠️?harness", re.IGNORECASE),
    "long_context": re.compile(r"long[-_]context", re.IGNORECASE),
    "refusal": re.compile(r"\brefusal\b", re.IGNORECASE),
    "repetitive": re.compile(r"\brepetitive\b", re.IGNORECASE),
    "lang_mixing": re.compile(r"\blang_mixing\b", re.IGNORECASE),
    "hallucination": re.compile(r"\bhallucination\b", re.IGNORECASE),
    "degeneration": re.compile(r"\bdegeneration\b", re.IGNORECASE),
    "fabrication": re.compile(r"\bfabrication\b", re.IGNORECASE),
    "missing_sections": re.compile(r"\bmissing-sections\b", re.IGNORECASE),
    "title_length": re.compile(r"\btitle-length\b", re.IGNORECASE),
    "description_length": re.compile(r"\bdescription-sentences\b", re.IGNORECASE),
    "keyword_count": re.compile(r"\bkeyword-count\b", re.IGNORECASE),
    "keyword_duplication": re.compile(r"\bkeyword-duplication\b", re.IGNORECASE),
    "reasoning_leak": re.compile(r"\breasoning-leak\b", re.IGNORECASE),
    "context_echo": re.compile(r"\bcontext-echo\b", re.IGNORECASE),
    "instruction_echo": re.compile(r"\binstruction-echo\b", re.IGNORECASE),
    "metadata_borrowing": re.compile(r"\bmetadata-borrowing\b", re.IGNORECASE),
    "trusted_hint_ignored": re.compile(r"\btrusted-hints-ignored\b", re.IGNORECASE),
    "trusted_hint_degraded": re.compile(r"\btrusted-hints-degraded\b", re.IGNORECASE),
    "cutoff": re.compile(r"\bcutoff\b", re.IGNORECASE),
    "token_cap": re.compile(r"\btoken-cap\b", re.IGNORECASE),
    "context_budget": re.compile(r"\bcontext-budget\b", re.IGNORECASE),
    "generic": re.compile(r"\bgeneric\b", re.IGNORECASE),
    "verbose": re.compile(r"\bverbose\b", re.IGNORECASE),
    "formatting": re.compile(r"\bformatting\b", re.IGNORECASE),
    "bullets": re.compile(r"\bbullets?\b", re.IGNORECASE),
    "context_ignored": re.compile(r"context-ignored", re.IGNORECASE),
}

QUALITY_BREAKING_LABELS: Final[frozenset[str]] = frozenset(
    {
        "harness",
        "long_context",
        "refusal",
        "repetitive",
        "hallucination",
        "degeneration",
        "context_ignored",
        "missing_sections",
        "reasoning_leak",
        "context_echo",
        "instruction_echo",
        "metadata_borrowing",
        "cutoff",
        "trusted_hint_degraded",
        "runtime_failure",
        "unknown_runtime_anomaly",
    },
)


def _truncate_text_preview(text: str, *, max_chars: int) -> str:
    """Trim text previews to a fixed character budget."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _collapse_preview_whitespace(text: str) -> str:
    """Flatten whitespace for compact table/report previews."""
    return re.sub(r"\s+", " ", text).strip()


def _populate_result_quality_analysis(
    result: PerformanceResult,
    *,
    prompt: str | None = None,
    metadata: MetadataDict | None = None,
    requested_max_tokens: int | None = None,
    context_marker: str = "Context:",
) -> PerformanceResult:
    """Attach structured quality analysis to successful results as soon as they exist."""
    if not result.success or result.generation is None:
        return result

    text = str(getattr(result.generation, "text", ""))
    needs_metadata_refresh = bool(metadata) and result.metadata_agreement is None

    cached_analysis = _quality_analysis_for_result(result)
    if cached_analysis is not None:
        cached_quality_issues = result.quality_issues or _build_quality_issues_string(
            cached_analysis,
        )
        needs_prompt_refresh = bool(prompt) and not cached_analysis.prompt_checks_ran
        if (
            not needs_prompt_refresh
            and not needs_metadata_refresh
            and result.quality_analysis is not None
            and result.quality_issues == cached_quality_issues
        ):
            return result
        if not needs_prompt_refresh:
            metadata_agreement = result.metadata_agreement
            if needs_metadata_refresh:
                metadata_agreement = compute_metadata_agreement(text, metadata)
            return replace(
                result,
                quality_analysis=cached_analysis,
                quality_issues=cached_quality_issues,
                metadata_agreement=metadata_agreement,
            )

    generated_tokens = getattr(result.generation, "generation_tokens", 0)
    prompt_tokens = getattr(result.generation, "prompt_tokens", None)
    resolved_requested_max_tokens = (
        requested_max_tokens if requested_max_tokens is not None else result.requested_max_tokens
    )
    analysis, quality_issues = _analyze_text_quality(
        text,
        generated_tokens,
        prompt_tokens=prompt_tokens,
        prompt=prompt,
        requested_max_tokens=resolved_requested_max_tokens,
        context_marker=context_marker,
    )
    if cached_analysis is not None:
        cached_quality_issues = result.quality_issues or _build_quality_issues_string(
            cached_analysis,
        )
        if result.quality_issues and result.quality_issues != cached_quality_issues:
            quality_issues = result.quality_issues

    metadata_agreement = result.metadata_agreement
    if needs_metadata_refresh:
        metadata_agreement = compute_metadata_agreement(text, metadata)

    return replace(
        result,
        quality_analysis=analysis,
        quality_issues=result.quality_issues or quality_issues,
        metadata_agreement=metadata_agreement,
    )


def _build_result_output_cues(result: PerformanceResult) -> list[str]:
    """Return compact issue cues that explain why an output is suspicious."""
    if not result.success:
        failure_cues: list[str] = []
        if result.error_package:
            failure_cues.append(result.error_package)
        if result.error_stage:
            failure_cues.append(result.error_stage.lower().replace(" ", "-"))
        return _dedupe_preserve_order(failure_cues)[:OUTPUT_PREVIEW_CUE_LIMIT]

    quality_labels = _extract_quality_issue_labels(result.quality_issues)
    analysis = _quality_analysis_for_result(result)
    cues: list[str] = []

    if (analysis is not None and analysis.has_harness_issue) or "harness" in quality_labels:
        harness_type = (
            analysis.harness_issue_type.replace("_", "-")
            if analysis and analysis.harness_issue_type
            else ""
        )
        cues.append(f"harness:{harness_type}" if harness_type else "harness")
    candidate_cues = [
        (
            (analysis is not None and analysis.is_repetitive) or "repetitive" in quality_labels,
            "repetitive",
        ),
        (
            (analysis is not None and analysis.has_context_echo)
            or "context_echo" in quality_labels,
            "context-echo",
        ),
        (
            (analysis is not None and analysis.instruction_echo)
            or "instruction_echo" in quality_labels,
            "instruction-echo",
        ),
        (
            (analysis is not None and analysis.metadata_borrowing)
            or "metadata_borrowing" in quality_labels,
            "metadata-borrowing",
        ),
        (
            (analysis is not None and analysis.verdict == "cutoff") or "cutoff" in quality_labels,
            "cutoff",
        ),
        (
            (analysis is not None and analysis.verdict == "context_budget")
            or "context_budget" in quality_labels,
            "context-budget",
        ),
        (
            (analysis is not None and analysis.has_reasoning_leak)
            or "reasoning_leak" in quality_labels,
            "reasoning-leak",
        ),
        (
            (analysis is not None and analysis.has_degeneration)
            or "degeneration" in quality_labels,
            "degeneration",
        ),
        (
            (analysis is not None and analysis.is_context_ignored)
            or "context_ignored" in quality_labels,
            "context-ignored",
        ),
        ((analysis is not None and analysis.is_refusal) or "refusal" in quality_labels, "refusal"),
        (
            (analysis is not None and analysis.missing_sections)
            or "missing_sections" in quality_labels,
            "missing-sections",
        ),
        (
            (analysis is not None and analysis.formatting_issues) or "formatting" in quality_labels,
            "formatting",
        ),
        ((analysis is not None and analysis.is_generic) or "generic" in quality_labels, "generic"),
    ]
    cues.extend(label for condition, label in candidate_cues if condition)

    return _dedupe_preserve_order(cues)[:OUTPUT_PREVIEW_CUE_LIMIT]


def _build_head_tail_preview(text: str, *, max_chars: int) -> str:
    """Build a compact preview that exposes both the start and end of long output."""
    if len(text) <= max_chars:
        return text

    separator = " ... [tail] "
    min_total = OUTPUT_PREVIEW_MIN_HEAD_CHARS + OUTPUT_PREVIEW_MIN_TAIL_CHARS + len(separator)
    if max_chars <= min_total:
        return _truncate_text_preview(text, max_chars=max_chars)

    head_budget = max(OUTPUT_PREVIEW_MIN_HEAD_CHARS, int(max_chars * 0.65))
    head_budget = min(head_budget, max_chars - OUTPUT_PREVIEW_MIN_TAIL_CHARS - len(separator))
    tail_budget = max_chars - head_budget - len(separator)

    head = text[:head_budget].rstrip()
    tail = text[-tail_budget:].lstrip()
    if not tail or tail in head:
        return _truncate_text_preview(text, max_chars=max_chars)
    return f"{head}{separator}{tail}"


def _build_output_preview_text(
    text: str,
    *,
    cues: Sequence[str] = (),
    max_chars: int,
) -> str:
    """Build a deterministic compact preview for Markdown/HTML/TSV tables."""
    normalized_text = _collapse_preview_whitespace(text)
    if not normalized_text:
        return ""

    cue_text = ""
    if cues:
        cue_text = f"[{'; '.join(cues[:OUTPUT_PREVIEW_CUE_LIMIT])}] "

    available_chars = max_chars - len(cue_text)
    if available_chars <= OUTPUT_PREVIEW_MIN_BODY_CHARS:
        return _truncate_text_preview(cue_text + normalized_text, max_chars=max_chars)

    preview_body = _build_head_tail_preview(normalized_text, max_chars=available_chars)
    return cue_text + preview_body


def _full_output_report_text(result: PerformanceResult) -> str:
    """Return the full text shown when expanding output details."""
    if result.success and result.generation is not None:
        return str(getattr(result.generation, "text", ""))
    if result.error_message:
        error_stage = result.error_stage or "Error"
        return f"Error: {error_stage} - {result.error_message}"
    return "Unknown error"


def _build_result_output_preview(
    result: PerformanceResult,
    *,
    max_chars: int = MAX_OUTPUT_PREVIEW_CHARS,
) -> str:
    """Build a shared skim-first preview for compact report surfaces."""
    full_text = _full_output_report_text(result)
    cues = _build_result_output_cues(result)

    if result.success:
        preview_source = _truncate_repetitive_output(full_text)
        lines = preview_source.splitlines()
        if len(lines) > MAX_OUTPUT_LINES:
            preview_source = "\n".join(lines[:MAX_OUTPUT_LINES]) + "\n..."
        return _build_output_preview_text(
            preview_source,
            cues=cues,
            max_chars=max_chars,
        )

    return _build_output_preview_text(
        full_text,
        cues=cues,
        max_chars=max_chars,
    )


def _extract_quality_issue_labels(quality_issues: str | None) -> set[str]:
    """Extract normalized quality labels from a free-form issue string."""
    if not quality_issues:
        return set()
    labels: set[str] = set()
    for label, pattern in QUALITY_ISSUE_PATTERNS.items():
        if pattern.search(quality_issues):
            labels.add(label)
    return labels


def _parse_quality_issues_to_list(quality_issues: str | None) -> list[str]:
    """Split stored comma-separated quality issue text into normalized items."""
    if not quality_issues:
        return []
    return [
        issue.strip() for issue in re.split(r",\s*(?![^()]*\))", quality_issues) if issue.strip()
    ]


def _truncate_quality_issues(
    quality_issues: str | None,
    max_len: int = MAX_QUALITY_ISSUES_LEN,
) -> str:
    """Truncate quality issue text for narrow table cells.

    Prefers whole parsed issue labels before falling back to hard clipping.
    """
    if not quality_issues:
        return ""
    if len(quality_issues) <= max_len:
        return quality_issues

    issues = _parse_quality_issues_to_list(quality_issues)
    if not issues:
        return quality_issues[: max_len - 3].rstrip() + "..."

    kept: list[str] = []
    for issue in issues:
        candidate = ", ".join([*kept, issue])
        if len(candidate) <= max_len:
            kept.append(issue)
            continue
        if kept:
            return ", ".join(kept) + ", ..."
        return issue[: max_len - 3].rstrip() + "..."

    return ", ".join(kept)


def _format_counter_items(counter: Counter[str], *, max_items: int = 6) -> str:
    """Format a counter as ``label=count`` pairs ordered by frequency."""
    if not counter:
        return "none"
    return ", ".join(f"{label}={count}" for label, count in counter.most_common(max_items))


@dataclass(frozen=True)
class UtilityTriageRow:
    """Cataloging triage row for summary ranking/logging."""

    result: PerformanceResult
    score: float
    description_score: float
    keyword_score: float
    grade: str
    weakness: str
    delta_vs_metadata: float | None
    labels: frozenset[str]


def _short_model_label(model_name: str, *, max_len: int = SUMMARY_MODEL_LABEL_MAX) -> str:
    """Return a compact model label suitable for narrow summary tables/charts."""
    label = model_name.rsplit("/", maxsplit=1)[-1]
    if len(label) <= max_len:
        return label
    return label[: max_len - 3] + "..."


def _format_float_or_dash(value: float | None, *, digits: int = 2) -> str:
    """Format a float value with fixed precision, or ``-`` when missing."""
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _log_rich_metric_chart(
    title: str,
    entries: Sequence[tuple[str, float]],
    *,
    unit: str = "",
    digits: int = 2,
    max_rows: int = SUMMARY_CHART_MAX_ROWS,
) -> None:
    """Log a compact Rich bar chart for ranked model metrics."""
    if not entries:
        return
    ranked = sorted(entries, key=lambda item: item[1], reverse=True)[:max_rows]
    max_value = max(value for _label, value in ranked)
    if max_value <= 0:
        return

    logger.info("%s", title)
    chart = Table.grid(padding=(0, 1))
    chart.add_column(justify="left", no_wrap=True)
    chart.add_column(ratio=1)
    chart.add_column(justify="right", no_wrap=True)
    for label, value in ranked:
        chart.add_row(
            _short_model_label(label),
            Bar(max_value, 0, value, width=SUMMARY_CHART_WIDTH),
            f"{_format_float_or_dash(value, digits=digits)}{unit}",
        )
    _log_rich_renderable(chart, indent="   ")


def _log_model_comparison_table_and_charts(results: list[PerformanceResult]) -> None:
    """Log per-model comparison table and compact Rich charts for this run."""
    if not results:
        return

    def _sort_key(result: PerformanceResult) -> tuple[int, float, str]:
        tps = float(getattr(result.generation, "generation_tps", 0.0) or 0.0)
        return (0 if result.success else 1, -tps, result.model_name)

    sorted_results = sorted(results, key=_sort_key)
    rows: list[list[str]] = []
    tps_entries: list[tuple[str, float]] = []
    total_time_entries: list[tuple[str, float]] = []

    for idx, res in enumerate(sorted_results, start=1):
        model_label = _short_model_label(res.model_name)
        if res.success and res.generation is not None:
            tps = float(getattr(res.generation, "generation_tps", 0.0) or 0.0)
            peak_mem = float(getattr(res.generation, "peak_memory", 0.0) or 0.0)
            notes = _truncate_quality_issues(res.quality_issues, max_len=34) or "clean"
            rows.append(
                [
                    str(idx),
                    model_label,
                    "OK",
                    _format_float_or_dash(tps, digits=1),
                    _format_float_or_dash(res.total_time, digits=2),
                    _format_float_or_dash(res.model_load_time, digits=2),
                    _format_float_or_dash(peak_mem if peak_mem > 0 else None, digits=2),
                    notes,
                ],
            )
            if tps > 0:
                tps_entries.append((res.model_name, tps))
            if res.total_time is not None and res.total_time > 0:
                total_time_entries.append((res.model_name, res.total_time))
        else:
            error_note = res.error_code or res.error_stage or res.error_message or "failure"
            error_note = _truncate_text_preview(error_note, max_chars=34)
            rows.append(
                [
                    str(idx),
                    model_label,
                    "FAIL",
                    "-",
                    _format_float_or_dash(res.total_time, digits=2),
                    _format_float_or_dash(res.model_load_time, digits=2),
                    "-",
                    error_note,
                ],
            )

    logger.info("📋 Model Comparison (current run):")
    headers = ["#", "Model", "Status", "TPS", "Total(s)", "Load(s)", "PeakGB", "Notes"]
    _log_rich_table(
        headers=headers,
        rows=rows,
        justifications=("right", "left", "left", "right", "right", "right", "right", "left"),
        no_wrap_headers=frozenset({"#", "Status", "TPS"}),
    )

    if tps_entries:
        log_blank()
        _log_rich_metric_chart("📊 TPS comparison chart:", tps_entries, unit=" tps", digits=1)
    if len(total_time_entries) >= MIN_MODELS_FOR_EFFICIENCY_CHART:
        inverted = [(name, 1.0 / value) for name, value in total_time_entries if value > 0]
        if inverted:
            _log_rich_metric_chart(
                "⏱ Efficiency chart (higher is faster overall):",
                inverted,
                unit=" 1/s",
                digits=3,
            )

    failed = [res for res in sorted_results if not res.success]
    if failed:
        stage_counts = Counter(res.error_stage or "Unknown" for res in failed)
        _log_rich_metric_chart(
            "❌ Failure stage frequency:",
            [(stage, float(count)) for stage, count in stage_counts.items()],
            unit=" x",
            digits=0,
        )


def _log_performance_highlights(successful: list[PerformanceResult]) -> None:
    """Log speed and memory highlights for successful runs."""
    if not successful:
        return

    summary: ModelIssueSummary = {}
    _populate_summary_performance_highlights(summary, successful)

    fastest_model = summary.get("fastest_model", ("<unknown>", 0.0))
    most_efficient_model = summary.get("most_efficient_model", ("<unknown>", 0.0))
    fastest_load_model = summary.get("fastest_load_model", ("<unknown>", 0.0))
    average_tps = summary.get("average_tps", 0.0)
    successful_count = summary.get("successful_count", 0)
    total_peak_memory = summary.get("total_peak_memory", 0.0)
    average_peak_memory = summary.get("average_peak_memory", 0.0)
    memory_efficiency = summary.get("memory_efficiency", 0.0)

    logger.info("🏆 Performance Highlights:")
    logger.info("   Fastest: %s (%.1f tps)", fastest_model[0], fastest_model[1])
    logger.info(
        "   💾 Most efficient: %s (%.1f GB)",
        most_efficient_model[0],
        most_efficient_model[1],
    )
    logger.info("   ⚡ Fastest load: %s (%.2fs)", fastest_load_model[0], fastest_load_model[1])
    logger.info("   📊 Average TPS: %.1f across %d models", average_tps, successful_count)

    log_blank()
    logger.info("📈 Resource Usage:")
    logger.info("   Total peak memory: %.1f GB", total_peak_memory)
    logger.info("   Average peak memory: %.1f GB", average_peak_memory)
    logger.info("   Memory efficiency: %.0f tokens/GB", memory_efficiency)


def _collect_quality_and_utility_rows(
    successful: list[PerformanceResult],
    *,
    prompt: str | None,
) -> tuple[Counter[str], int, list[UtilityTriageRow], float | None, str | None]:
    """Collect quality counts and cataloging utility rows."""
    quality_counts: Counter[str] = Counter()
    clean_count = 0
    context = _extract_trusted_hint_bundle(prompt).trusted_text or None if prompt else None
    baseline = _compute_metadata_baseline_utility(context)
    baseline_score = baseline[0] if baseline is not None else None
    baseline_grade = baseline[1] if baseline is not None else None
    rows: list[UtilityTriageRow] = []

    for res in successful:
        labels = frozenset(_extract_quality_issue_labels(res.quality_issues))
        if labels:
            for label in labels:
                quality_counts[label] += 1
        else:
            clean_count += 1

        if not res.generation:
            continue
        text = str(getattr(res.generation, "text", "") or "")
        utility = compute_cataloging_utility(text, context)
        score = float(utility["utility_score"])
        grade = str(utility["utility_grade"])
        weakness = str(utility["primary_weakness"])
        delta = score - baseline_score if baseline_score is not None else None
        rows.append(
            UtilityTriageRow(
                result=res,
                score=score,
                description_score=float(utility.get("description_score", 0.0)),
                keyword_score=float(utility.get("keyword_score", 0.0)),
                grade=grade,
                weakness=weakness,
                delta_vs_metadata=delta,
                labels=labels,
            ),
        )

    return quality_counts, clean_count, rows, baseline_score, baseline_grade


def _log_quality_signal_summary(
    quality_counts: Counter[str],
    *,
    clean_count: int,
    successful_count: int,
) -> None:
    """Log quality issue frequency among successful models."""
    logger.info("🧪 Quality Signal Frequency:")
    logger.info("   %s", _format_counter_items(quality_counts))
    logger.info("   Clean outputs: %d/%d", clean_count, successful_count)


def _build_report_triage_context(
    successful: Sequence[PerformanceResult],
    *,
    prompt: str | None,
) -> ReportTriageContext:
    """Build cached quality/utility triage data for report renderers."""
    quality_counts, clean_count, utility_rows, baseline_score, baseline_grade = (
        _collect_quality_and_utility_rows(
            list(successful),
            prompt=prompt,
        )
    )
    useful_rows = _select_useful_rows(utility_rows)
    watchlist_rows = _select_watchlist_rows(utility_rows)
    sorted_quality_counts = tuple(quality_counts.most_common())
    return ReportTriageContext(
        quality_counts=sorted_quality_counts,
        clean_count=clean_count,
        utility_rows=tuple(utility_rows),
        useful_rows=tuple(useful_rows),
        watchlist_rows=tuple(watchlist_rows),
        baseline_score=baseline_score,
        baseline_grade=baseline_grade,
    )


def _grade_display_parts(grade: str, score: float) -> str:
    """Return a compact grade/score label for report sections."""
    emoji = GRADE_EMOJIS.get(grade, "")
    return f"{emoji} {grade} ({score:.0f}/100)"


def _humanize_watchlist_reason(reason: str) -> str:
    """Convert internal watchlist reason labels into reviewer-facing text."""
    if reason.startswith("worse-than-metadata"):
        return "worse than metadata baseline"
    return re.sub(
        r"\s+",
        " ",
        reason.replace("_", " ").replace("-", " ").replace(",", ", "),
    ).strip()


def _format_review_shortlist_line(
    row: UtilityTriageRow,
    *,
    include_reason: str | None = None,
    html_output: bool = False,
) -> str:
    """Format one useful/watchlist row for shared report sections."""
    details: list[str] = [_grade_display_parts(row.grade, row.score)]
    details.append(f"Desc {row.description_score:.0f}")
    details.append(f"Keywords {row.keyword_score:.0f}")
    if row.delta_vs_metadata is not None:
        details.append(f"Δ{row.delta_vs_metadata:+.0f}")
    tps = getattr(row.result.generation, "generation_tps", None)
    if isinstance(tps, int | float) and float(tps) > 0.0:
        details.append(f"{float(tps):.1f} tps")
    if include_reason is not None:
        details.append(_humanize_watchlist_reason(include_reason))
    if html_output:
        return (
            f"<code>{html.escape(row.result.model_name)}</code>: {html.escape(' | '.join(details))}"
        )
    return f"`{row.result.model_name}`: {' | '.join(details)}"


def _format_review_priorities_parts(
    report_context: ReportRenderContext,
    *,
    html_output: bool,
) -> list[str]:
    """Render shared reviewer-oriented shortlists for HTML/Markdown reports."""
    triage = report_context.triage
    useful_rows = list(triage.useful_rows)
    watchlist_rows = list(triage.watchlist_rows)
    if not useful_rows and not watchlist_rows:
        return []

    parts: list[str] = []
    if html_output:
        parts.append("<h3>🧭 Review Shortlist</h3>")
    else:
        _append_markdown_section(parts, title="## 🧭 Review Shortlist")

    if useful_rows:
        if html_output:
            parts.append("<p><b>Strong candidates:</b></p><ul>")
            parts.extend(
                f"<li>{_format_review_shortlist_line(row, html_output=True)}</li>"
                for row in useful_rows[:MAX_TRIAGE_MODELS]
            )
            parts.append("</ul>")
        else:
            parts.append("### Strong Candidates")
            parts.append("")
            parts.extend(
                f"- {_format_review_shortlist_line(row)}" for row in useful_rows[:MAX_TRIAGE_MODELS]
            )
            parts.append("")

    if watchlist_rows:
        if html_output:
            parts.append("<p><b>Watchlist:</b></p><ul>")
            parts.extend(
                "<li>"
                f"{_format_review_shortlist_line(row, include_reason=reason, html_output=True)}"
                "</li>"
                for row, reason in watchlist_rows[:MAX_TRIAGE_MODELS]
            )
            parts.append("</ul>")
        else:
            parts.append("### Watchlist")
            parts.append("")
            parts.extend(
                f"- {_format_review_shortlist_line(row, include_reason=reason)}"
                for row, reason in watchlist_rows[:MAX_TRIAGE_MODELS]
            )
            parts.append("")

    return parts


def _build_action_snapshot_stanzas(
    results: list[PerformanceResult],
    report_context: ReportRenderContext,
) -> list[ReportStanza]:
    """Build shared action-snapshot stanzas for Markdown and HTML reports."""
    summary = report_context.summary
    triage = report_context.triage
    preflight_issues = report_context.preflight_issues
    failed: list[PerformanceResult] = [res for res in results if not res.success]
    quality_counts = Counter(dict(triage.quality_counts))
    harness_success_count = sum("harness" in row.labels for row in triage.utility_rows)
    successful_count = len(triage.utility_rows)
    runtime_analysis = summary.get("runtime_analysis")

    failure_rows = _action_snapshot_failure_items(
        failed,
        triage=triage,
        harness_success_count=harness_success_count,
        successful_count=successful_count,
    )
    if preflight_issues:
        failure_rows.extend(
            [
                (
                    "Preflight compatibility",
                    f"{len(preflight_issues)} informational warning(s); "
                    "do not treat these alone as run failures.",
                ),
                (
                    "Escalate only if",
                    "they line up with API mismatches, startup hangs, or backend/runtime crashes.",
                ),
            ]
        )

    quality_rows: list[tuple[str, str | None]] = []
    if triage.baseline_score is not None and triage.baseline_grade is not None:
        better = len(summary.get("cataloging_improves_metadata", []))
        neutral = len(summary.get("cataloging_neutral_vs_metadata", []))
        worse = len(summary.get("cataloging_worse_than_metadata", []))
        quality_rows.append(
            (
                "Vs existing metadata",
                f"better={better}, neutral={neutral}, worse={worse} "
                f"(baseline {triage.baseline_grade} {triage.baseline_score:.0f}/100).",
            )
        )
    quality_rows.append(("Quality signal frequency", f"{_format_counter_items(quality_counts)}."))
    termination_summary = ", ".join(
        f"{reason}={count}"
        for reason, count in sorted(
            Counter("completed" if res.success else "exception" for res in results).items(),
        )
    )
    if runtime_analysis is None:
        quality_rows.append(("Termination reasons", f"{termination_summary}."))

    runtime_rows: list[tuple[str, str | None]] = []
    if runtime_analysis is not None:
        runtime_lines = [
            *_format_runtime_analysis_lines(runtime_analysis),
            *_format_runtime_timing_snapshot_lines(runtime_analysis),
        ]
        for line in runtime_lines:
            if not line.startswith("- **") or ":** " not in line:
                runtime_rows.append(("Runtime note", line.removeprefix("- ")))
                continue
            label, value = line.removeprefix("- **").split(":** ", maxsplit=1)
            runtime_rows.append((label, value))

    stanzas = [
        _build_report_stanza("Failures & Triage", failure_rows),
        _build_report_stanza("Quality & Metadata", quality_rows),
        _build_report_stanza("Runtime", runtime_rows),
    ]
    return [stanza for stanza in stanzas if stanza.rows]


def _format_action_snapshot_parts(
    results: list[PerformanceResult],
    report_context: ReportRenderContext,
    *,
    html_output: bool,
) -> list[str]:
    """Build a compact shared triage block for maintainers and reviewers.

    The snapshot is split into three scannable groups:
    **Failures & Triage** — hard failures and review queues.
    **Quality & Metadata** — signal frequencies and baseline comparison.
    **Runtime** — phase timing and decode dominance.
    """
    stanzas = _build_action_snapshot_stanzas(results, report_context)

    if html_output:
        parts: list[str] = ["<h3>🎯 Action Snapshot</h3>"]
        for stanza in stanzas:
            parts.extend(_render_html_stanza(stanza))
        return parts

    parts = ["## 🎯 Action Snapshot", ""]
    for stanza in stanzas:
        _append_markdown_stanza(parts, stanza)
    return parts


def _action_snapshot_failure_items(
    failed: list[PerformanceResult],
    *,
    triage: ReportTriageContext,
    harness_success_count: int,
    successful_count: int,
) -> list[tuple[str, str]]:
    """Build label/value pairs for the Failures & Triage snapshot group."""
    items: list[tuple[str, str]] = []
    if failed:
        owners = Counter(res.error_package or "unknown" for res in failed)
        owner_summary = ", ".join(f"{owner}={count}" for owner, count in owners.most_common(3))
        items.append(
            ("Framework/runtime failures", f"{len(failed)} (top owners: {owner_summary}).")
        )
        items.append(
            (
                "Next action",
                "review failure ownership below and use diagnostics.md for filing.",
            )
        )
    else:
        items.append(("Framework/runtime failures", "none."))

    items.append(
        (
            "Maintainer signals",
            f"harness-risk successes={harness_success_count}, "
            f"clean outputs={triage.clean_count}/{successful_count}.",
        )
    )
    useful_text = (
        f"{len(triage.useful_rows)} clean A/B model(s) worth first review."
        if triage.useful_rows
        else "none (no clean A/B shortlist for this run)."
    )
    items.append(("Useful now", useful_text))
    watchlist_text = (
        f"{len(triage.watchlist_rows)} model(s) with breaking or lower-value output."
        if triage.watchlist_rows
        else "none."
    )
    items.append(("Review watchlist", watchlist_text))
    return items


def _group_failures_by_package(
    results: Sequence[PerformanceResult],
) -> list[tuple[str, list[PerformanceResult]]]:
    """Group failures by originating package ordered by highest count first."""
    failed = [result for result in results if not result.success]
    by_package: dict[str, list[PerformanceResult]] = {}
    for result in failed:
        package = result.error_package or "unknown"
        by_package.setdefault(package, []).append(result)
    return sorted(by_package.items(), key=lambda item: -len(item[1]))


def _format_failures_by_package_parts(
    results: list[PerformanceResult],
    *,
    html_output: bool,
) -> list[str]:
    """Render shared failure-ownership sections for HTML/Markdown reports."""
    sorted_packages = _group_failures_by_package(results)
    if not sorted_packages:
        return []

    if html_output:
        parts: list[str] = [
            "<h3>🚨 Failures by Package</h3>",
            "<table>",
            (
                "<thead><tr><th>Package</th><th>Failures</th><th>Error Types</th>"
                "<th>Affected Models</th></tr></thead>"
            ),
            "<tbody>",
        ]
        for package, failures in sorted_packages:
            error_types = ", ".join(
                sorted({result.error_stage or "unknown" for result in failures}),
            )
            models = ", ".join(result.model_name for result in failures)
            parts.append(
                "<tr>"
                f"<td><code>{html.escape(package)}</code></td>"
                f"<td>{len(failures)}</td>"
                f"<td>{html.escape(error_types)}</td>"
                f"<td>{html.escape(models)}</td>"
                "</tr>",
            )
        parts.append("</tbody></table>")
        parts.append("<p><b>Actionable Items by Package</b></p>")
        for package, failures in sorted_packages:
            parts.append(f"<h4>{html.escape(package)}</h4><ul>")
            for result in failures:
                error_message = result.error_message or ""
                if len(error_message) > ERROR_MESSAGE_TRUNCATE_LEN:
                    error_message = error_message[: ERROR_MESSAGE_TRUNCATE_LEN - 3] + "..."
                issue_parts = [html.escape(result.model_name)]
                if result.error_stage:
                    issue_parts.append(html.escape(result.error_stage))
                if result.error_type:
                    issue_parts.append(html.escape(result.error_type))
                if error_message:
                    issue_parts.append(html.escape(error_message))
                parts.append(f"<li>{' | '.join(issue_parts)}</li>")
            parts.append("</ul>")
        return parts

    parts = ["## 🚨 Failures by Package (Actionable)", ""]
    rows: list[tuple[str, str, str, str]] = []
    for package, failures in sorted_packages:
        markdown_error_types = sorted({result.error_stage or "unknown" for result in failures})
        markdown_models = [f"`{result.model_name}`" for result in failures]
        rows.append(
            (
                f"`{package}`",
                str(len(failures)),
                ", ".join(markdown_error_types),
                ", ".join(markdown_models),
            )
        )
    parts.extend(
        tabulate(
            rows,
            headers=("Package", "Failures", "Error Types", "Affected Models"),
            tablefmt="github",
        ).splitlines()
    )

    parts.append("")
    parts.append("### Actionable Items by Package")
    parts.append("")

    for package, failures in sorted_packages:
        parts.append(f"#### {package}")
        parts.append("")
        for result in failures:
            parts.extend(
                _wrap_markdown_text(
                    f"{result.model_name} ({result.error_stage})",
                    initial_indent="- ",
                    subsequent_indent="  ",
                )
            )
            error_message = result.error_message or ""
            if len(error_message) > ERROR_MESSAGE_TRUNCATE_LEN:
                error_message = error_message[: ERROR_MESSAGE_TRUNCATE_LEN - 3] + "..."
            escaped_message = DIAGNOSTICS_ESCAPER.escape(error_message)
            parts.append(f"  - Error: `{escaped_message}`")
            if result.error_type:
                parts.append(f"  - Type: `{result.error_type}`")
        parts.append("")

    return parts


def _select_useful_rows(rows: list[UtilityTriageRow]) -> list[UtilityTriageRow]:
    """Return shortlist of high-utility, non-breaking models."""
    useful_rows = [
        row
        for row in rows
        if row.grade in {"A", "B"}
        and not (row.labels & QUALITY_BREAKING_LABELS)
        and (row.delta_vs_metadata is None or row.delta_vs_metadata > 0)
    ]
    useful_rows.sort(
        key=lambda row: (
            row.delta_vs_metadata if row.delta_vs_metadata is not None else float("-inf"),
            row.score,
            getattr(row.result.generation, "generation_tps", 0) or 0,
        ),
        reverse=True,
    )
    return useful_rows


def _select_watchlist_rows(rows: list[UtilityTriageRow]) -> list[tuple[UtilityTriageRow, str]]:
    """Return successful-but-risky models with reason labels."""
    watchlist_rows: list[tuple[UtilityTriageRow, str]] = []
    for row in rows:
        breaking = sorted(row.labels & QUALITY_BREAKING_LABELS)
        if breaking:
            watchlist_rows.append((row, ",".join(breaking)))
            continue
        if (
            row.delta_vs_metadata is not None
            and row.delta_vs_metadata < -UTILITY_DELTA_NEUTRAL_BAND
        ):
            watchlist_rows.append((row, f"worse-than-metadata(Δ{row.delta_vs_metadata:+.0f})"))
            continue
        if row.grade in {"D", "F"}:
            watchlist_rows.append((row, row.weakness))

    watchlist_rows.sort(
        key=lambda item: (
            0 if "harness" in item[0].labels else 1,
            0
            if (
                item[0].delta_vs_metadata is not None
                and item[0].delta_vs_metadata < -UTILITY_DELTA_NEUTRAL_BAND
            )
            else 1,
            {"F": 0, "D": 1, "C": 2, "B": 3, "A": 4}.get(item[0].grade, 2),
            item[0].score,
        ),
    )
    return watchlist_rows


def _log_utility_triage(
    rows: list[UtilityTriageRow],
    *,
    baseline_score: float | None = None,
    baseline_grade: str | None = None,
) -> None:
    """Log utility grade distribution and shortlists."""
    if not rows:
        return

    grade_counts: Counter[str] = Counter(row.grade for row in rows)
    avg_utility = sum(row.score for row in rows) / len(rows)
    n_ab = grade_counts.get("A", 0) + grade_counts.get("B", 0)
    n_c = grade_counts.get("C", 0)
    n_df = grade_counts.get("D", 0) + grade_counts.get("F", 0)

    log_blank()
    logger.info("📚 Cataloging Utility Snapshot:")
    logger.info(
        "   Avg score: %.0f/100 | A/B=%d, C=%d, D/F=%d",
        avg_utility,
        n_ab,
        n_c,
        n_df,
    )
    if baseline_score is not None:
        baseline_display = _get_grade_display(baseline_grade or "F")
        deltas = [row.delta_vs_metadata for row in rows if row.delta_vs_metadata is not None]
        better = sum(delta > UTILITY_DELTA_NEUTRAL_BAND for delta in deltas)
        worse = sum(delta < -UTILITY_DELTA_NEUTRAL_BAND for delta in deltas)
        neutral = len(deltas) - better - worse
        avg_delta = sum(deltas) / len(deltas) if deltas else 0.0
        logger.info("   Metadata baseline: %s %.0f/100", baseline_display, baseline_score)
        logger.info(
            "   Vs metadata: Avg Δ %+.0f | better=%d, neutral=%d, worse=%d",
            avg_delta,
            better,
            neutral,
            worse,
        )

    candidate_rows = [
        row
        for row in rows
        if _is_cataloging_pick_eligible(row.result, _quality_analysis_for_result(row.result))
        and row.grade in {"A", "B", "C"}
        and not (row.labels & QUALITY_BREAKING_LABELS)
    ] or [
        row
        for row in rows
        if _is_cataloging_pick_eligible(row.result, _quality_analysis_for_result(row.result))
    ]
    if candidate_rows:
        best_description = max(
            candidate_rows,
            key=lambda row: (
                row.description_score,
                row.score,
                getattr(row.result.generation, "generation_tps", 0) or 0,
                row.result.model_name,
            ),
        )
        best_keywords = max(
            candidate_rows,
            key=lambda row: (
                row.keyword_score,
                row.score,
                getattr(row.result.generation, "generation_tps", 0) or 0,
                row.result.model_name,
            ),
        )
        logger.info(
            "   Best description: %s (%.0f/100)",
            best_description.result.model_name,
            best_description.description_score,
        )
        logger.info(
            "   Best keywording: %s (%.0f/100)",
            best_keywords.result.model_name,
            best_keywords.keyword_score,
        )

    useful_rows = _select_useful_rows(rows)
    if useful_rows:
        logger.info("   Useful now (top %d):", min(MAX_TRIAGE_MODELS, len(useful_rows)))
        for row in useful_rows[:MAX_TRIAGE_MODELS]:
            tps = getattr(row.result.generation, "generation_tps", 0) or 0
            if row.delta_vs_metadata is None:
                logger.info(
                    "   - %s: %s %.0f/100 (%.1f tps)",
                    row.result.model_name,
                    row.grade,
                    row.score,
                    tps,
                )
            else:
                logger.info(
                    "   - %s: %s %.0f/100 (Δ%+.0f, %.1f tps)",
                    row.result.model_name,
                    row.grade,
                    row.score,
                    row.delta_vs_metadata,
                    tps,
                )
    else:
        logger.info("   Useful now: none (no clean A/B outputs)")

    watchlist_rows = _select_watchlist_rows(rows)
    if watchlist_rows:
        logger.info("   Watchlist (top %d):", min(MAX_TRIAGE_MODELS, len(watchlist_rows)))
        for row, reason in watchlist_rows[:MAX_TRIAGE_MODELS]:
            logger.info(
                "   - %s: %s %.0f/100 (%s)",
                row.result.model_name,
                row.grade,
                row.score,
                reason,
            )


def _log_failed_models_summary(failed: list[PerformanceResult]) -> None:
    """Log failed models and failure distribution for actionable triage."""
    logger.info("❌ Failed Models (%d):", len(failed))
    for res in failed:
        error_pkg = f" -> {res.error_package}" if res.error_package else ""
        phase_suffix = f" [{res.failure_phase}]" if res.failure_phase else ""
        code_suffix = f" {{{res.error_code}}}" if res.error_code else ""
        logger.info(
            "  - %s (%s%s)%s%s",
            res.model_name,
            res.error_stage or "Unknown",
            error_pkg,
            phase_suffix,
            code_suffix,
            extra={"style_hint": LogStyles.ERROR},
        )
        action_hint = _build_failure_action_hint(
            error_package=res.error_package,
            failure_phase=res.failure_phase,
            error_stage=res.error_stage,
        )
        logger.info(
            "    -> %s",
            action_hint,
            extra={"style_hint": LogStyles.WARNING},
        )
        if res.error_message:
            symptom = _truncate_text_preview(
                _normalize_error_core_message(res.error_message),
                max_chars=160,
            )
            logger.info(
                "    -> symptom: %s",
                symptom,
                extra={"style_hint": LogStyles.WARNING},
            )

    package_names: list[str] = [res.error_package or "unknown" for res in failed]
    stage_names: list[str] = [res.error_stage or "Unknown" for res in failed]
    pkg_counts: Counter[str] = Counter(package_names)
    stage_counts: Counter[str] = Counter(stage_names)
    logger.info("   By package: %s", _format_counter_items(pkg_counts))
    logger.info("   By stage: %s", _format_counter_items(stage_counts))


def _log_successful_models_list(successful: list[PerformanceResult]) -> None:
    """Log successful models sorted by generation throughput."""
    logger.info("✅ Successful Models (%d):", len(successful))
    sorted_success = sorted(
        successful,
        key=lambda r: getattr(r.generation, "generation_tps", 0) or 0,
        reverse=True,
    )
    for res in sorted_success:
        tps = getattr(res.generation, "generation_tps", 0) or 0
        active_mem = res.active_memory or 0.0
        cache_mem = res.cache_memory or 0.0
        mem_info = ""
        if active_mem > 0 or cache_mem > 0:
            mem_info = f" (Active: {active_mem:.1f}GB, Cache: {cache_mem:.1f}GB)"
        logger.info(
            "  - %s: %.1f tps%s",
            res.model_name,
            tps,
            mem_info,
            extra={"style_hint": LogStyles.SUCCESS},
        )


def log_summary(
    results: list[PerformanceResult],
    *,
    prompt: str | None = None,
) -> None:
    """Log run summary focused on diagnostics, quality, and model triage."""
    if not results:
        return

    log_blank()
    log_rule(style="blue", bold=True)
    logger.info("Results Summary", extra={"style_hint": LogStyles.HEADER})
    log_rule(style="blue", bold=True)

    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    if successful:
        _log_performance_highlights(successful)
        log_blank()
        (
            quality_counts,
            clean_count,
            utility_rows,
            baseline_score,
            baseline_grade,
        ) = _collect_quality_and_utility_rows(
            successful,
            prompt=prompt,
        )
        _log_quality_signal_summary(
            quality_counts,
            clean_count=clean_count,
            successful_count=len(successful),
        )
        _log_utility_triage(
            utility_rows,
            baseline_score=baseline_score,
            baseline_grade=baseline_grade,
        )
        log_blank()

    if failed:
        _log_failed_models_summary(failed)
        log_blank()

    _log_model_comparison_table_and_charts(results)
    log_blank()

    if successful:
        _log_successful_models_list(successful)
    _log_canonical_run_review_summary(results)


def _history_path_for_jsonl(jsonl_path: Path) -> Path:
    """Derive history JSONL path from the main JSONL report path."""
    return jsonl_path.with_name(f"{jsonl_path.stem}.history.jsonl")


def _load_latest_history_record(history_path: Path) -> HistoryRunRecord | None:
    """Load the most recent history record from an append-only JSONL file."""
    if not history_path.exists():
        return None

    try:
        lines = history_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        logger.warning("Failed to read history file %s", history_path)
        return None

    for line in reversed(lines):
        record_line = line.strip()
        if not record_line:
            continue
        try:
            record = json.loads(record_line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and record.get("_type") == "run":
            return cast("HistoryRunRecord", record)
    return None


def _load_history_run_records(
    history_path: Path | None,
    *,
    max_records: int = DIAGNOSTICS.history_max_records,
) -> list[HistoryRunRecord]:
    """Load up to ``max_records`` run entries from append-only history JSONL."""
    if history_path is None or not history_path.exists():
        return []

    try:
        lines = history_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        logger.warning("Failed to read history file %s", history_path)
        return []

    records: list[HistoryRunRecord] = []
    for line in lines[-max_records:]:
        text = line.strip()
        if not text:
            continue
        try:
            record = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and record.get("_type") == "run":
            records.append(cast("HistoryRunRecord", record))
    return records


def _history_bucket_rank(bucket: object) -> int:
    """Return severity rank for history user-bucket comparisons."""
    if not isinstance(bucket, str):
        return 0
    ranks = {
        "recommended": 0,
        "caveat": 1,
        "caution": 1,
        "needs_triage": 2,
        "avoid": 3,
    }
    return ranks.get(bucket, 0)


def compare_history_records(
    previous: HistoryRunRecord | None,
    current: HistoryRunRecord,
) -> HistoryComparisonSummary:
    """Compare two history records and return status plus quality transitions."""
    prev_models: dict[str, HistoryModelResultRecord] = (
        previous.get("model_results", {}) if previous else {}
    )
    curr_models: dict[str, HistoryModelResultRecord] = (
        current.get("model_results", {}) if current else {}
    )
    if not isinstance(prev_models, dict):
        prev_models = {}
    if not isinstance(curr_models, dict):
        curr_models = {}

    prev_success = {model for model, info in prev_models.items() if info.get("success") is True}
    prev_failed = {model for model, info in prev_models.items() if info.get("success") is False}
    curr_success = {model for model, info in curr_models.items() if info.get("success") is True}
    curr_failed = {model for model, info in curr_models.items() if info.get("success") is False}

    regressions = sorted(prev_success & curr_failed)
    recoveries = sorted(prev_failed & curr_success)
    new_models = sorted(set(curr_models) - set(prev_models))
    missing_models = sorted(set(prev_models) - set(curr_models))

    quality_regressions: list[str] = []
    quality_recoveries: list[str] = []
    harness_regressions: list[str] = []
    harness_recoveries: list[str] = []
    owner_changes: list[str] = []

    stable_models = sorted(
        (set(prev_models) & set(curr_models))
        - set(regressions)
        - set(recoveries)
        - set(new_models)
        - set(missing_models)
    )
    for model_name in stable_models:
        prev_info = prev_models[model_name]
        curr_info = curr_models[model_name]
        if prev_info.get("success") is True and curr_info.get("success") is True:
            prev_bucket = _history_bucket_rank(prev_info.get("review_user_bucket"))
            curr_bucket = _history_bucket_rank(curr_info.get("review_user_bucket"))
            if curr_bucket > prev_bucket:
                quality_regressions.append(model_name)
            elif curr_bucket < prev_bucket:
                quality_recoveries.append(model_name)

            prev_harness = bool(prev_info.get("harness_issue_type"))
            curr_harness = bool(curr_info.get("harness_issue_type"))
            if not prev_harness and curr_harness:
                harness_regressions.append(model_name)
            elif prev_harness and not curr_harness:
                harness_recoveries.append(model_name)

        prev_owner = prev_info.get("review_owner")
        curr_owner = curr_info.get("review_owner")
        if prev_owner and curr_owner and prev_owner != curr_owner:
            owner_changes.append(model_name)

    return {
        "regressions": regressions,
        "recoveries": recoveries,
        "new_models": new_models,
        "missing_models": missing_models,
        "quality_regressions": quality_regressions,
        "quality_recoveries": quality_recoveries,
        "harness_regressions": harness_regressions,
        "harness_recoveries": harness_recoveries,
        "owner_changes": owner_changes,
    }


def _build_prompt_preview(prompt: str, *, max_chars: int = 200) -> str:
    """Return prompt preview truncated to ``max_chars``."""
    return prompt if len(prompt) <= max_chars else f"{prompt[:max_chars]}..."


def _history_model_result_from_result(result: PerformanceResult) -> HistoryModelResultRecord:
    """Build a typed per-model history row from runtime result."""
    record: HistoryModelResultRecord = {
        "success": result.success,
        "failure_phase": result.failure_phase,
        "error_stage": result.error_stage,
        "error_type": result.error_type,
        "error_package": result.error_package,
        "error_code": result.error_code,
        "error_signature": result.error_signature,
    }
    review = _build_jsonl_review_record(result)
    analysis = _quality_analysis_for_result(result)
    if review is not None:
        record["review_verdict"] = review["verdict"]
        record["review_owner"] = review["owner"]
        record["review_user_bucket"] = review["user_bucket"]
        record["hit_max_tokens"] = review["hit_max_tokens"]
        issue_subtype = _review_issue_subtype(result, review, analysis)
        if issue_subtype is not None:
            record["review_issue_subtype"] = issue_subtype
        if review["prompt_output_ratio"] is not None:
            record["prompt_output_ratio"] = review["prompt_output_ratio"]
        if review["nontext_prompt_ratio"] is not None:
            record["nontext_prompt_ratio"] = review["nontext_prompt_ratio"]
    if analysis is not None and analysis.harness_issue_type is not None:
        record["harness_issue_type"] = analysis.harness_issue_type
    stop_reason = result.runtime_diagnostics.stop_reason if result.runtime_diagnostics else None
    if stop_reason is not None:
        record["stop_reason"] = stop_reason
    return record


def _build_history_run_record(
    *,
    results: list[PerformanceResult],
    prompt: str,
    system_info: dict[str, str],
    library_versions: LibraryVersionDict,
    image_path: Path | None,
    runtime_fingerprint: dict[str, RuntimeProbeResult] | None = None,
) -> HistoryRunRecord:
    """Build typed run-history record payload prior to append."""
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    model_results = {
        result.model_name: _history_model_result_from_result(result) for result in results
    }
    record: HistoryRunRecord = {
        "_type": "run",
        "format_version": "1.0",
        "timestamp": local_now_str(),
        "prompt_hash": prompt_hash,
        "prompt_preview": _build_prompt_preview(prompt),
        "image_path": str(image_path) if image_path is not None else None,
        "model_results": model_results,
        "system": system_info,
        "library_versions": library_versions,
    }
    if runtime_fingerprint is not None:
        record["runtime_fingerprint"] = runtime_fingerprint
    return record


def append_history_record(
    *,
    history_path: Path,
    results: list[PerformanceResult],
    prompt: str,
    system_info: dict[str, str],
    library_versions: LibraryVersionDict,
    image_path: Path | None = None,
    runtime_fingerprint: dict[str, RuntimeProbeResult] | None = None,
) -> HistoryRunRecord:
    """Append a per-run history record for tracking regressions/recoveries."""
    record = _build_history_run_record(
        results=results,
        prompt=prompt,
        system_info=system_info,
        library_versions=library_versions,
        image_path=image_path,
        runtime_fingerprint=runtime_fingerprint,
    )

    try:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        logger.exception("Failed to append history record to %s", history_path)

    return record


def _build_jsonl_metadata_record(
    *,
    prompt: str,
    system_info: dict[str, str],
    library_versions: LibraryVersionDict | None = None,
    runtime_fingerprint: dict[str, RuntimeProbeResult] | None = None,
) -> JsonlMetadataRecord:
    """Build shared metadata header row for JSONL results."""
    record: JsonlMetadataRecord = {
        "_type": "metadata",
        "format_version": "2.0",
        "prompt": prompt,
        "system": system_info,
        "timestamp": local_now_str(),
    }
    if library_versions is not None:
        record["library_versions"] = library_versions
    if runtime_fingerprint is not None:
        record["runtime_fingerprint"] = runtime_fingerprint
    return record


def _build_jsonl_result_record_base(result: PerformanceResult) -> JsonlResultRecord:
    """Build base per-model JSONL row with failure-safe defaults."""
    runtime = result.runtime_diagnostics
    record: JsonlResultRecord = {
        "_type": "result",
        "model": result.model_name,
        "success": result.success,
        "failure_phase": result.failure_phase,
        "error_stage": result.error_stage,
        "error_code": result.error_code,
        "error_signature": result.error_signature,
        "error_message": result.error_message,
        "captured_output_on_fail": result.captured_output_on_fail,
        "error_type": result.error_type,
        "error_package": result.error_package,
        "error_traceback": result.error_traceback,
        "quality_issues": _parse_quality_issues_to_list(result.quality_issues),
        "timestamp": local_now_str(),
        "metrics": {},
        "timing": {
            "generation_time_s": result.generation_time,
            "model_load_time_s": result.model_load_time,
            "total_time_s": result.total_time,
            "input_validation_time_s": (
                runtime.input_validation_time_s if runtime is not None else None
            ),
            "prompt_prep_time_s": runtime.prompt_prep_time_s if runtime is not None else None,
            "cleanup_time_s": runtime.cleanup_time_s if runtime is not None else None,
            "first_token_latency_s": (
                runtime.first_token_latency_s if runtime is not None else None
            ),
            "stop_reason": runtime.stop_reason if runtime is not None else None,
        },
    }
    # Attach structured signature components for failed results to enable
    # precise cross-run clustering without re-parsing messages.
    if not result.success and result.error_message is not None:
        components: JsonlSignatureComponents = {}
        if result.error_code is not None:
            components["error_code"] = result.error_code
        components["normalized_message"] = _normalize_error_core_message(
            result.root_error_message or result.error_message
        )
        if result.error_traceback is not None:
            components["traceback_signature"] = _normalize_traceback_signature(
                result.error_traceback
            )
        record["signature_components"] = components
    # Attach differential rerun evidence when available (G3: secondary only)
    if result.rerun_evidence is not None:
        ev = result.rerun_evidence
        summary: JsonlRerunSummary = {"rerun_success": ev.rerun_success}
        if ev.rerun_verdict is not None:
            summary["rerun_verdict"] = ev.rerun_verdict
        if ev.rerun_error_code is not None:
            summary["rerun_error_code"] = ev.rerun_error_code
        if ev.rerun_generated_chars is not None:
            summary["rerun_generated_chars"] = ev.rerun_generated_chars
        if ev.rerun_generation_time is not None:
            summary["rerun_generation_time"] = ev.rerun_generation_time
        if ev.rerun_prompt is not None:
            summary["rerun_prompt"] = ev.rerun_prompt
        record["rerun_summary"] = summary
    if result.root_error_type is not None:
        record["root_error_type"] = result.root_error_type
    if result.root_error_module is not None:
        record["root_error_module"] = result.root_error_module
    if result.root_error_message is not None:
        record["root_error_message"] = result.root_error_message
    prompt_diagnostics = _prompt_diagnostics_to_json(result.prompt_diagnostics)
    if prompt_diagnostics:
        record["prompt_diagnostics"] = prompt_diagnostics
    return record


def _build_jsonl_quality_analysis_record(
    quality_analysis: object,
) -> JsonlQualityAnalysisRecord | None:
    """Build JSONL quality-analysis payload from generation analysis object."""
    if quality_analysis is None:
        return None
    return {
        "issues": list(getattr(quality_analysis, "issues", [])),
        "metrics": {
            "word_count": int(getattr(quality_analysis, "word_count", 0)),
            "unique_ratio": float(getattr(quality_analysis, "unique_ratio", 0.0)),
            "bullet_count": int(getattr(quality_analysis, "bullet_count", 0)),
        },
    }


def _build_jsonl_metadata_agreement_record(
    metadata_agreement: MetadataAgreementMetrics | None,
) -> JsonlMetadataAgreementRecord | None:
    """Build JSONL metadata-agreement payload from result-level benchmark data."""
    if metadata_agreement is None:
        return None
    return {
        "overall_score": metadata_agreement.overall_score,
        "title_score": metadata_agreement.title_score,
        "description_score": metadata_agreement.description_score,
        "keyword_score": metadata_agreement.keyword_score,
        "nonvisual_penalty": metadata_agreement.nonvisual_penalty,
        "matched_terms": list(metadata_agreement.matched_terms),
        "missed_terms": list(metadata_agreement.missed_terms),
        "nonvisual_hits": list(metadata_agreement.nonvisual_hits),
    }


def _populate_jsonl_result_generation_data(
    record: JsonlResultRecord,
    result: PerformanceResult,
) -> None:
    """Attach success-only generation metrics/text/quality data to JSONL row."""
    if not result.generation:
        return

    generation = result.generation
    record["metrics"] = {
        "prompt_tokens": getattr(generation, "prompt_tokens", 0),
        "generation_tokens": getattr(generation, "generation_tokens", 0),
        "total_tokens": getattr(generation, "total_tokens", 0),
        "prompt_tps": getattr(generation, "prompt_tps", 0.0),
        "generation_tps": getattr(generation, "generation_tps", 0.0),
        "peak_memory_gb": getattr(generation, "peak_memory", 0.0),
        "active_memory_gb": result.active_memory or 0.0,
        "cache_memory_gb": result.cache_memory or 0.0,
    }

    text = getattr(generation, "text", None)
    if text is not None:
        record["generated_text"] = text

    quality_payload = _build_jsonl_quality_analysis_record(result.quality_analysis)
    if quality_payload:
        record["quality_analysis"] = quality_payload

    metadata_agreement_payload = _build_jsonl_metadata_agreement_record(result.metadata_agreement)
    if metadata_agreement_payload:
        record["metadata_agreement"] = metadata_agreement_payload


def save_jsonl_report(
    results: list[PerformanceResult],
    filename: Path,
    prompt: str,
    system_info: dict[str, str],
    *,
    library_versions: LibraryVersionDict | None = None,
    runtime_fingerprint: dict[str, RuntimeProbeResult] | None = None,
) -> None:
    """Save results to a JSONL file for programmatic analysis and AI issue generation.

    The JSONL format includes all diagnostic information needed to generate
    actionable GitHub issue reports, including:
    - Full error tracebacks for debugging
    - Error type classification for bucketing
    - Package attribution for directing reports
    - Quality analysis for successful models
    - Timing metrics for performance analysis
    - Shared library/runtime metadata for reproducible maintainer triage
    - Per-result maintainer triage hints for actionable follow-up

    Format (v2.0): First line is a metadata header containing prompt,
    system_info, and shared runtime context. Per-model result lines follow.
    """
    issue_cluster_by_model = _issue_cluster_map_for_results(results, prompt=prompt)
    try:
        with filename.open("w", encoding="utf-8") as f:
            # Write shared metadata header (avoids repeating prompt/system per row)
            header = _build_jsonl_metadata_record(
                prompt=prompt,
                system_info=system_info,
                library_versions=library_versions,
                runtime_fingerprint=runtime_fingerprint,
            )
            f.write(json.dumps(header) + "\n")

            for result in results:
                record = _build_jsonl_result_record_base(result)
                _populate_jsonl_result_generation_data(record, result)
                review_payload = _build_jsonl_review_record(result)
                if review_payload:
                    record["review"] = review_payload
                    record["maintainer_triage"] = _build_jsonl_maintainer_triage_record(
                        result,
                        review_payload,
                        issue_cluster=issue_cluster_by_model.get(result.model_name),
                    )
                f.write(json.dumps(record) + "\n")
        # Logging handled in finalize_execution
    except OSError:
        logger.exception("Failed to write JSONL report to %s", filename)


def _write_report_failure_jsonl(
    *,
    filename: Path,
    failed_report: str,
    error: Exception,
) -> None:
    """Write a minimal JSONL error record when report generation fails."""
    record = {
        "status": "error",
        "error_stage": "report_generation",
        "failed_report": failed_report,
        "error_message": str(error),
        "timestamp": local_now_str(),
    }
    try:
        filename.parent.mkdir(parents=True, exist_ok=True)
        with filename.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        logger.exception("Failed to write report failure JSONL to %s", filename)


def _history_model_results(record: HistoryRunRecord | None) -> dict[str, HistoryModelResultRecord]:
    """Return normalized model-results map from a history record."""
    if record is None:
        return {}
    raw = record.get("model_results", {})
    if not isinstance(raw, dict):
        return {}
    return raw


def _history_counts(record: HistoryRunRecord | None) -> tuple[int, int, int, float]:
    """Return (total, success, failed, success_rate_pct) for a history record."""
    models = _history_model_results(record)
    total = len(models)
    success = sum(1 for info in models.values() if info.get("success") is True)
    failed = sum(1 for info in models.values() if info.get("success") is False)
    success_rate = (success / total * 100.0) if total else 0.0
    return total, success, failed, success_rate


def _fmt_delta_int(current: int, previous: int | None) -> str:
    """Format signed integer delta for run-over-run summary tables."""
    if previous is None:
        return "-"
    delta = current - previous
    return "0" if delta == 0 else f"{delta:+d}"


def _fmt_delta_pct(current: float, previous: float | None) -> str:
    """Format signed percentage-point delta for run-over-run summary tables."""
    if previous is None:
        return "-"
    delta = current - previous
    return "0.0pp" if abs(delta) < FLOAT_ZERO_EPSILON else f"{delta:+.1f}pp"


def _short_hash(value: str | None, *, size: int = 10) -> str:
    """Return short hash preview for compact context tables."""
    if not value:
        return "-"
    return value[:size]


def _format_library_versions_for_history(record: HistoryRunRecord | None) -> str:
    """Return compact comma-separated library version snapshot for history rows."""
    if record is None:
        return "-"
    versions = record.get("library_versions", {})
    if not isinstance(versions, dict) or not versions:
        return "-"

    parts: list[str] = []
    for name in sorted(versions):
        version = versions.get(name)
        if version:
            parts.append(f"{name}={version}")
    if not parts:
        return "-"

    combined = ", ".join(parts)
    return _truncate_text_preview(combined, max_chars=80)


def _model_status_from_history(info: HistoryModelResultRecord | None) -> str:
    """Render ``OK``/``FAIL``/``-`` for per-model history transition tables."""
    if info is None:
        return "-"
    return "OK" if info.get("success") is True else "FAIL"


def _history_summary_for_comparison(
    previous: HistoryRunRecord | None,
    current: HistoryRunRecord,
) -> HistoryComparisonSummary:
    """Return transition summary and log context-change warnings when needed."""
    if previous is None:
        return {
            "regressions": [],
            "recoveries": [],
            "new_models": [],
            "missing_models": [],
            "quality_regressions": [],
            "quality_recoveries": [],
            "harness_regressions": [],
            "harness_recoveries": [],
            "owner_changes": [],
        }

    prev_prompt_hash = previous.get("prompt_hash")
    curr_prompt_hash = current.get("prompt_hash")
    if prev_prompt_hash and curr_prompt_hash and prev_prompt_hash != curr_prompt_hash:
        log_warning_note(
            "Prompt differs from previous run; regressions/recoveries may be noisy.",
        )
    prev_image = previous.get("image_path")
    curr_image = current.get("image_path")
    if prev_image and curr_image and prev_image != curr_image:
        log_warning_note(
            "Image path differs from previous run; regressions/recoveries may be noisy.",
        )

    return compare_history_records(previous, current)


def _history_summary_rows(
    *,
    previous: HistoryRunRecord | None,
    current: HistoryRunRecord,
    summary: HistoryComparisonSummary,
) -> list[list[str]]:
    """Build run-over-run summary rows for history comparison output."""
    regressions = summary["regressions"]
    recoveries = summary["recoveries"]
    new_models = summary["new_models"]
    missing_models = summary["missing_models"]
    quality_regressions = summary["quality_regressions"]
    quality_recoveries = summary["quality_recoveries"]
    harness_regressions = summary["harness_regressions"]
    harness_recoveries = summary["harness_recoveries"]
    owner_changes = summary["owner_changes"]

    prev_total, prev_success, prev_failed, prev_success_rate = _history_counts(previous)
    curr_total, curr_success, curr_failed, curr_success_rate = _history_counts(current)

    previous_values: tuple[str, str, str, str]
    if previous is None:
        previous_values = ("-", "-", "-", "-")
    else:
        previous_values = (
            str(prev_total),
            str(prev_success),
            str(prev_failed),
            f"{prev_success_rate:.1f}%",
        )

    return [
        [
            "Models tested",
            previous_values[0],
            str(curr_total),
            _fmt_delta_int(curr_total, prev_total if previous is not None else None),
        ],
        [
            "Successful",
            previous_values[1],
            str(curr_success),
            _fmt_delta_int(curr_success, prev_success if previous is not None else None),
        ],
        [
            "Failed",
            previous_values[2],
            str(curr_failed),
            _fmt_delta_int(curr_failed, prev_failed if previous is not None else None),
        ],
        [
            "Success rate",
            previous_values[3],
            f"{curr_success_rate:.1f}%",
            _fmt_delta_pct(curr_success_rate, prev_success_rate if previous is not None else None),
        ],
        ["Regressions", "-", str(len(regressions)), "-"],
        ["Recoveries", "-", str(len(recoveries)), "-"],
        ["Quality regressions", "-", str(len(quality_regressions)), "-"],
        ["Quality recoveries", "-", str(len(quality_recoveries)), "-"],
        ["Harness regressions", "-", str(len(harness_regressions)), "-"],
        ["Harness recoveries", "-", str(len(harness_recoveries)), "-"],
        ["Owner changes", "-", str(len(owner_changes)), "-"],
        ["New models", "-", str(len(new_models)), "-"],
        ["Missing models", "-", str(len(missing_models)), "-"],
    ]


def _history_context_rows(
    previous: HistoryRunRecord | None,
    current: HistoryRunRecord,
) -> list[list[str]]:
    """Build compact context rows (prompt/image/version) for history comparisons."""
    previous_image = (
        _truncate_text_preview(previous.get("image_path") or "-", max_chars=40)
        if previous is not None
        else "-"
    )
    return [
        [
            "Prompt hash",
            _short_hash(previous.get("prompt_hash") if previous else None),
            _short_hash(current.get("prompt_hash")),
        ],
        [
            "Image path",
            previous_image,
            _truncate_text_preview(current.get("image_path") or "-", max_chars=40),
        ],
        [
            "Library versions",
            _format_library_versions_for_history(previous),
            _format_library_versions_for_history(current),
        ],
    ]


def _log_history_transition_chart(
    *,
    summary: HistoryComparisonSummary,
    current: HistoryRunRecord,
) -> None:
    """Log transition chart (or current status chart when no transitions)."""
    regressions = summary["regressions"]
    recoveries = summary["recoveries"]
    new_models = summary["new_models"]
    missing_models = summary["missing_models"]
    quality_regressions = summary["quality_regressions"]
    quality_recoveries = summary["quality_recoveries"]
    harness_regressions = summary["harness_regressions"]
    harness_recoveries = summary["harness_recoveries"]
    owner_changes = summary["owner_changes"]
    transition_entries = [
        ("regressions", float(len(regressions))),
        ("recoveries", float(len(recoveries))),
        ("quality regressions", float(len(quality_regressions))),
        ("quality recoveries", float(len(quality_recoveries))),
        ("harness regressions", float(len(harness_regressions))),
        ("harness recoveries", float(len(harness_recoveries))),
        ("owner changes", float(len(owner_changes))),
        ("new", float(len(new_models))),
        ("missing", float(len(missing_models))),
    ]
    if any(value > 0 for _label, value in transition_entries):
        _log_rich_metric_chart(
            "🔁 Status transition counts:",
            transition_entries,
            unit=" x",
            digits=0,
            max_rows=len(transition_entries),
        )
        return

    _total, curr_success, curr_failed, _success_rate = _history_counts(current)
    _log_rich_metric_chart(
        "✅ Current run status counts:",
        [("success", float(curr_success)), ("failed", float(curr_failed))],
        unit=" x",
        digits=0,
        max_rows=2,
    )


def _history_transition_detail_text(
    prev_info: HistoryModelResultRecord | None,
    curr_info: HistoryModelResultRecord | None,
) -> str:
    """Return compact detail text for history transition rows."""
    source = curr_info if curr_info is not None else prev_info
    if source is None:
        return "-"

    parts: list[str] = []
    if prev_info is not None and curr_info is not None:
        prev_bucket = prev_info.get("review_user_bucket")
        curr_bucket = curr_info.get("review_user_bucket")
        if prev_bucket != curr_bucket and (prev_bucket or curr_bucket):
            parts.append(f"bucket={prev_bucket or '-'}->{curr_bucket or '-'}")

        prev_verdict = prev_info.get("review_verdict")
        curr_verdict = curr_info.get("review_verdict")
        if prev_verdict != curr_verdict and (prev_verdict or curr_verdict):
            parts.append(f"verdict={prev_verdict or '-'}->{curr_verdict or '-'}")

        prev_harness = prev_info.get("harness_issue_type")
        curr_harness = curr_info.get("harness_issue_type")
        if prev_harness != curr_harness and (prev_harness or curr_harness):
            parts.append(f"harness={prev_harness or '-'}->{curr_harness or '-'}")

        prev_owner = prev_info.get("review_owner")
        curr_owner = curr_info.get("review_owner")
        if prev_owner != curr_owner and (prev_owner or curr_owner):
            parts.append(f"owner={prev_owner or '-'}->{curr_owner or '-'}")

    if not parts:
        harness_issue = source.get("harness_issue_type")
        review_verdict = source.get("review_verdict")
        error_stage = source.get("error_stage")
        if harness_issue:
            parts.append(f"harness={harness_issue}")
        elif review_verdict:
            parts.append(f"verdict={review_verdict}")
        elif error_stage:
            parts.append(error_stage)

    return " | ".join(parts[:2]) or "-"


def _history_transition_rows(
    *,
    previous: HistoryRunRecord | None,
    current: HistoryRunRecord,
    summary: HistoryComparisonSummary,
) -> list[list[str]]:
    """Build per-model transition rows for regressions/recoveries/new/missing models."""
    regressions = summary["regressions"]
    recoveries = summary["recoveries"]
    new_models = summary["new_models"]
    missing_models = summary["missing_models"]
    quality_regressions = summary["quality_regressions"]
    quality_recoveries = summary["quality_recoveries"]
    harness_regressions = summary["harness_regressions"]
    harness_recoveries = summary["harness_recoveries"]
    owner_changes = summary["owner_changes"]
    changed_models = sorted(
        {
            *regressions,
            *recoveries,
            *new_models,
            *missing_models,
            *quality_regressions,
            *quality_recoveries,
            *harness_regressions,
            *harness_recoveries,
            *owner_changes,
        }
    )
    if not changed_models:
        return []

    prev_models = _history_model_results(previous)
    curr_models = _history_model_results(current)
    transition_rows: list[list[str]] = []
    for model_name in changed_models:
        if model_name in regressions:
            transition = "Regression"
        elif model_name in recoveries:
            transition = "Recovery"
        elif model_name in harness_regressions:
            transition = "Harness regression"
        elif model_name in harness_recoveries:
            transition = "Harness recovery"
        elif model_name in quality_regressions:
            transition = "Quality regression"
        elif model_name in quality_recoveries:
            transition = "Quality recovery"
        elif model_name in owner_changes:
            transition = "Owner changed"
        elif model_name in new_models:
            transition = "New"
        elif model_name in missing_models:
            transition = "Missing"
        else:
            transition = "Changed"

        prev_info = prev_models.get(model_name)
        curr_info = curr_models.get(model_name)
        transition_rows.append(
            [
                _short_model_label(model_name),
                transition,
                _model_status_from_history(prev_info),
                _model_status_from_history(curr_info),
                _history_transition_detail_text(prev_info, curr_info),
            ],
        )
    return transition_rows


def _log_history_comparison(
    previous: HistoryRunRecord | None,
    current: HistoryRunRecord,
) -> None:
    """Print the History Comparison CLI section (regressions/recoveries)."""
    print_cli_section("History Comparison")
    if previous is None:
        logger.info("No prior history run available. Baseline created.")
    summary = _history_summary_for_comparison(previous, current)
    rows = _history_summary_rows(previous=previous, current=current, summary=summary)
    logger.info("📚 Run-over-run comparison:")
    _log_rich_table(headers=["Metric", "Previous", "Current", "Delta"], rows=rows)

    context_rows = _history_context_rows(previous, current)
    logger.info("🔎 Comparison context:")
    _log_rich_table(headers=["Context", "Previous", "Current"], rows=context_rows)

    _log_history_transition_chart(summary=summary, current=current)
    transition_rows = _history_transition_rows(previous=previous, current=current, summary=summary)
    if transition_rows:
        logger.info("🧾 Detailed model transitions:")
        _log_rich_table(
            headers=["Model", "Transition", "Prev", "Current", "Signal"],
            rows=transition_rows,
            no_wrap_headers=frozenset({"Prev", "Current"}),
        )


def _issue_title(cluster: IssueCluster) -> str:
    """Build the GitHub issue draft title for a cluster."""
    symptom = _sanitize_bpe_display(_issue_problem_summary(cluster), max_len=96).strip().rstrip(".")
    if not symptom:
        symptom = _issue_subtype_display_label(cluster.issue_subtype)
    owner_label = cluster.owner.replace("[", r"\[").replace("]", r"\]")
    subtype_label = (
        _issue_subtype_display_label(cluster.issue_subtype).replace("[", r"\[").replace("]", r"\]")
    )
    return (
        f"\\[{owner_label}\\]\\[{subtype_label}\\] "
        f"{symptom} affecting {_issue_cluster_model_count(cluster)} model(s)"
    )


def _issue_summary_section(cluster: IssueCluster) -> list[str]:
    """Build the action-oriented Summary section for one issue cluster."""
    parts = ["", "## Summary", ""]
    model_count = _issue_cluster_model_count(cluster)
    parts.append(
        f"{model_count} model(s) show **{MARKDOWN_ESCAPER.escape(_issue_subtype_display_label(cluster.issue_subtype))}** "
        f"that should be filed against {MARKDOWN_ESCAPER.escape(_issue_filing_target_label(cluster.owner))}."
    )
    parts.append("")
    parts.append(
        f"- **Observed problem:** {MARKDOWN_ESCAPER.escape(_issue_problem_summary(cluster))}"
    )
    parts.append(
        f"- **Target:** {MARKDOWN_ESCAPER.escape(_issue_filing_target_label(cluster.owner))}"
    )
    parts.append(f"- **Raw owner hint:** `{MARKDOWN_ESCAPER.escape(cluster.owner)}`")
    parts.append(f"- **Affected models:** {model_count}")
    parts.append(f"- **Fixed when:** {MARKDOWN_ESCAPER.escape(_issue_queue_fixed_when(cluster))}")
    parts.append(f"- **Issue kind:** `{MARKDOWN_ESCAPER.escape(cluster.issue_kind)}`")
    parts.append(
        f"- **Raw cluster:** `{MARKDOWN_ESCAPER.escape(cluster.cluster_id)}` "
        f"(`{MARKDOWN_ESCAPER.escape(cluster.issue_subtype)}`)"
    )
    parts.append("")
    return parts


def _issue_bundle_link(
    model_name: str,
    repro_bundles: Mapping[str, Path],
    *,
    label: str = "repro JSON",
) -> str:
    """Return the issue-draft repro-bundle link for a model, if available."""
    bundle = repro_bundles.get(model_name)
    if bundle is None:
        return ""
    return f"[{MARKDOWN_ESCAPER.escape(label)}](../repro_bundles/{quote(bundle.name)})"


def _issue_model_signal(
    result: PerformanceResult,
    *,
    stack_symptoms: Mapping[str, str],
) -> str:
    """Return the strongest per-model signal for an affected-model row."""
    if result.model_name in stack_symptoms:
        return stack_symptoms[result.model_name]
    triage = _maintainer_triage_for_result(result)
    if triage is not None:
        if evidence_text := _maintainer_triage_evidence_text(triage):
            return evidence_text
        if triage["summary"] and triage["summary"] != "no flagged signals":
            return triage["summary"]
        issue_subtype = triage.get("issue_subtype")
        return issue_subtype or triage["issue_kind"]
    return result.error_message or "clustered issue signal"


def _issue_affected_models_section(
    cluster: IssueCluster,
    *,
    repro_bundles: Mapping[str, Path],
) -> list[str]:
    """Build the Affected Models table for one issue cluster."""
    stack_symptoms = {
        result.model_name: symptom for result, symptom, _owner in cluster.stack_signals
    }
    rows: list[tuple[str, ...]] = []
    for result in _issue_cluster_results(cluster):
        triage = _maintainer_triage_for_result(result)
        context = _maintainer_triage_context_text(triage) if triage is not None else None
        rows.append(
            (
                f"`{MARKDOWN_ESCAPER.escape(result.model_name)}`",
                MARKDOWN_ESCAPER.escape(_issue_model_signal(result, stack_symptoms=stack_symptoms)),
                MARKDOWN_ESCAPER.escape(context or ""),
                _issue_bundle_link(result.model_name, repro_bundles),
            )
        )

    parts = ["", "## Affected Models", ""]
    parts.extend(
        tabulate(
            rows,
            headers=["Model", "Representative Signal", "Token Context", "Repro JSON"],
            tablefmt="github",
        ).splitlines()
    )
    parts.append("")
    return parts


def _issue_failure_evidence(result: PerformanceResult) -> list[str]:
    """Build failure-specific evidence for one affected model."""
    parts = [f"### `{MARKDOWN_ESCAPER.escape(result.model_name)}`", ""]
    if result.error_message:
        parts.append("Observed error:")
        _append_markdown_code_block(parts, result.error_message, language="text")
    if result.root_error_type and result.root_error_message:
        root_label = result.root_error_type
        if result.root_error_module:
            root_label = f"{result.root_error_module}.{root_label}"
        root_text = f"{root_label}: {result.root_error_message}"
        if (
            result.root_error_type != result.error_type
            or result.root_error_message != result.error_message
        ):
            parts.append("Root exception:")
            _append_markdown_code_block(parts, root_text, language="text")
    if trace_tail := _format_traceback_tail(result.error_traceback):
        parts.append("Traceback tail:")
        _append_markdown_code_block(parts, trace_tail, language="text")
    return parts


def _issue_harness_evidence(result: PerformanceResult) -> list[str]:
    """Build harness-specific evidence for one affected model."""
    parts = [f"### `{MARKDOWN_ESCAPER.escape(result.model_name)}`", ""]
    analysis = _quality_analysis_for_result(result)
    observations = [
        desc
        for detail in (analysis.harness_issue_details if analysis is not None else [])
        if (desc := _describe_harness_detail(detail))
    ]
    observations.extend(_summarize_quality_signals(analysis))
    unique_observations = _dedupe_preserve_order(observations)
    if unique_observations:
        parts.append("Observed signals:")
        parts.append("")
        parts.extend(f"- {MARKDOWN_ESCAPER.escape(item)}" for item in unique_observations)
        parts.append("")

    sample_output = (
        str(getattr(result.generation, "text", "") or "").strip()
        if result.generation is not None
        else ""
    )
    if sample_output:
        parts.append("Sample output:")
        _append_markdown_code_block(
            parts,
            _truncate_text_preview(sample_output, max_chars=DIAGNOSTICS.output_snippet_len),
            language="text",
        )
    return parts


def _issue_stack_evidence(cluster: IssueCluster) -> list[str]:
    """Build stack-signal evidence for clustered successful-run anomalies."""
    if not cluster.stack_signals:
        return []
    rows: list[tuple[str, ...]] = []
    for result, symptom, owner in cluster.stack_signals:
        generation = result.generation
        prompt_tokens = int(getattr(generation, "prompt_tokens", 0) or 0) if generation else 0
        generated_tokens = (
            int(getattr(generation, "generation_tokens", 0) or 0) if generation else 0
        )
        ratio = f"{(generated_tokens / prompt_tokens):.2%}" if prompt_tokens > 0 else "n/a"
        rows.append(
            (
                f"`{MARKDOWN_ESCAPER.escape(result.model_name)}`",
                fmt_num(prompt_tokens),
                fmt_num(generated_tokens),
                ratio,
                MARKDOWN_ESCAPER.escape(symptom),
                f"`{MARKDOWN_ESCAPER.escape(owner)}`",
            )
        )
    parts = ["### Stack Signals", ""]
    parts.extend(
        tabulate(
            rows,
            headers=["Model", "Prompt Tok", "Output Tok", "Output/Prompt", "Symptom", "Owner"],
            tablefmt="github",
        ).splitlines()
    )
    parts.append("")
    return parts


def _issue_minimal_evidence_for_result(
    result: PerformanceResult,
    *,
    stack_symptoms: Mapping[str, str],
) -> list[str]:
    """Return a few standalone evidence bullets for the pasteable issue body."""
    bullets: list[str] = []
    if result.model_name in stack_symptoms:
        bullets.append(
            f"`{MARKDOWN_ESCAPER.escape(result.model_name)}`: "
            f"{MARKDOWN_ESCAPER.escape(stack_symptoms[result.model_name])}"
        )
    elif not result.success:
        observed = _simplify_failure_message(result.error_message, model_name=result.model_name)
        bullets.append(
            f"`{MARKDOWN_ESCAPER.escape(result.model_name)}` fails with: "
            f"{MARKDOWN_ESCAPER.escape(observed)}"
        )
        if result.root_error_type and result.root_error_message:
            root_label = result.root_error_type
            if result.root_error_module:
                root_label = f"{result.root_error_module}.{root_label}"
            root_text = _truncate_text_preview(result.root_error_message, max_chars=180)
            bullets.append(
                f"Root exception: `{MARKDOWN_ESCAPER.escape(root_label)}`: "
                f"{MARKDOWN_ESCAPER.escape(root_text)}"
            )
    else:
        analysis = _quality_analysis_for_result(result)
        observations = [
            desc
            for detail in (analysis.harness_issue_details if analysis is not None else [])
            if (desc := _describe_harness_detail(detail))
        ]
        observations.extend(_summarize_quality_signals(analysis))
        bullets.extend(
            (
                f"`{MARKDOWN_ESCAPER.escape(result.model_name)}`: "
                f"{MARKDOWN_ESCAPER.escape(observation)}"
            )
            for observation in _dedupe_preserve_order(observations)[:2]
        )

    generation_text = (
        str(getattr(result.generation, "text", "") or "").strip()
        if result.generation is not None
        else ""
    )
    if generation_text:
        preview = " ".join(_truncate_text_preview(generation_text, max_chars=320).split())
        bullets.append(f"Output excerpt: `{MARKDOWN_ESCAPER.escape(preview)}`")
    return bullets


def _issue_minimal_evidence_section(cluster: IssueCluster) -> list[str]:
    """Build compact inline evidence before repro commands."""
    parts = ["", "## Minimal Evidence", ""]
    stack_symptoms = {
        result.model_name: symptom for result, symptom, _owner in cluster.stack_signals
    }
    evidence_lines: list[str] = []
    for result in _issue_cluster_results(cluster)[:2]:
        evidence_lines.extend(
            _issue_minimal_evidence_for_result(result, stack_symptoms=stack_symptoms)
        )
    if evidence_lines:
        parts.extend(f"- {line}" for line in evidence_lines[:4])
    else:
        parts.append(f"- {MARKDOWN_ESCAPER.escape(cluster.symptom)}")
    parts.append("")
    return parts


def _issue_evidence_section(cluster: IssueCluster) -> list[str]:
    """Build the Evidence section for one issue cluster."""
    parts = ["", "## Appendix: Detailed Evidence", ""]
    if cluster.stack_signals:
        parts.extend(_issue_stack_evidence(cluster))

    for result in _issue_cluster_results(cluster)[:ISSUE_EVIDENCE_MODEL_LIMIT]:
        if not result.success:
            parts.extend(_issue_failure_evidence(result))
        else:
            parts.extend(_issue_harness_evidence(result))

    if _issue_cluster_model_count(cluster) > ISSUE_EVIDENCE_MODEL_LIMIT:
        parts.append("_Additional affected models are listed in the Affected Models table above._")
        parts.append("")
    return parts


def _issue_likely_root_cause_section(cluster: IssueCluster) -> list[str]:
    """Build the likely-root-cause section for one issue cluster."""
    representative = _issue_cluster_representative(cluster)
    triage = _maintainer_triage_for_result(representative) if representative is not None else None
    confidence = triage["confidence"] if triage is not None else "medium"
    next_action = (
        triage["next_action"] if triage is not None else _diagnostics_next_action(cluster.owner)
    )
    why = (
        triage["summary"]
        if triage is not None and triage["summary"] != "no flagged signals"
        else cluster.symptom
    )
    rows = [
        ("Filing target", MARKDOWN_ESCAPER.escape(_issue_filing_target_label(cluster.owner))),
        ("Likely owner", f"`{MARKDOWN_ESCAPER.escape(cluster.owner)}`"),
        ("Confidence", MARKDOWN_ESCAPER.escape(confidence)),
        ("Issue kind", f"`{MARKDOWN_ESCAPER.escape(cluster.issue_kind)}`"),
        ("Issue subtype", f"`{MARKDOWN_ESCAPER.escape(cluster.issue_subtype)}`"),
        ("Why this classification is credible", MARKDOWN_ESCAPER.escape(why)),
        ("Suggested next action", MARKDOWN_ESCAPER.escape(next_action)),
    ]
    parts = ["", "## Likely Root Cause", ""]
    _append_markdown_row_block(parts, rows=rows)
    return parts


def _issue_repro_section(
    cluster: IssueCluster,
    *,
    repro_bundles: Mapping[str, Path],
    run_args: argparse.Namespace | None,
) -> list[str]:
    """Build cluster-first repro commands and bundle links for one issue."""
    models = _issue_cluster_model_names(cluster)
    base_tokens = _build_repro_command_tokens(
        image_path=None,
        run_args=run_args,
        include_selection=False,
    )
    cluster_command = shlex_join([*base_tokens, "--models", *models])

    parts = ["", "## Repro Commands", ""]
    parts.append("Cluster rerun:")
    _append_markdown_code_block(parts, cluster_command, language="bash")

    bundle_lines = [
        f"- `{MARKDOWN_ESCAPER.escape(model)}`: {_issue_bundle_link(model, repro_bundles)}"
        for model in models
        if model in repro_bundles
    ]
    if bundle_lines:
        parts.append("Repro bundles:")
        parts.append("")
        parts.extend(bundle_lines)
        parts.append(
            "- Note: these are local artifact links; attach or publish the JSON when filing "
            "upstream."
        )
        parts.append("")

    if cluster.source == "failure" and len(models) > 1:
        per_model_commands = "\n".join(
            shlex_join([*base_tokens, "--models", model]) for model in models
        )
        parts.append("Per-model failure reruns:")
        _append_markdown_code_block(parts, per_model_commands, language="bash")
    return parts


def _issue_fix_checklist_items(cluster: IssueCluster) -> list[str]:
    """Return subtype-specific fix checklist items for one issue cluster."""
    subtype = cluster.issue_subtype.casefold()
    checklist: list[str] | None = None
    if subtype == "stop_token":
        checklist = [
            "Inspect model EOS token IDs and tokenizer special-token mappings.",
            "Verify mlx-vlm stop criteria receive all configured EOS/stop tokens.",
            "Check `skip_special_tokens` handling during decode.",
            "Strip generated control tokens such as `<|end|>` and `</think>` only after "
            "confirming generation stopped at the right boundary.",
        ]
    elif subtype == "encoding":
        checklist = [
            "Inspect tokenizer decode cleanup for byte-level/BPE marker leakage.",
            "Compare `decode` and `batch_decode` behavior with `skip_special_tokens=True`.",
            "Verify processor/tokenizer config does not require model-specific cleanup flags.",
        ]
    elif subtype == "prompt_template":
        checklist = [
            "Inspect chat template selection and rendered message roles.",
            "Verify image placeholder count and order match the processor config.",
            "Check EOS defaults and whether the template expects explicit assistant prefixes.",
        ]
    elif subtype in {"long_context", "context_budget", "token_cap"}:
        checklist = [
            "Rerun with reduced image/text burden and compare output recovery.",
            "Compare prompt-token accounting with text-only and image+text prompts.",
            "Inspect cache allocation, prefill step size, and long-context generation behavior.",
        ]
    elif cluster.issue_kind == "runtime_failure":
        representative = _issue_cluster_representative(cluster)
        runtime_kind = _runtime_failure_diagnostic_kind(representative)
        if runtime_kind == "unsupported_granite":
            checklist = [
                "Inspect mlx-vlm model-type registry and architecture import path for Granite.",
                "Confirm whether this model type should be supported by the installed mlx-vlm version.",
                "If unsupported, fail before generation with a clear architecture-specific message.",
                "If supported, add a focused load/regression check for this model family.",
            ]
        elif runtime_kind == "weight_config_mismatch":
            checklist = [
                "Compare checkpoint keys with the selected model class and model config.",
                "Inspect missing/unexpected projector, scale, bias, and quantized-weight parameter names.",
                "Verify the model repo revision matches the mlx-vlm/mlx loader expectations.",
                "Reproduce after upgrading/downgrading mlx-vlm and mlx to isolate version compatibility.",
            ]
        elif runtime_kind == "missing_image_processor":
            checklist = [
                "Inspect `preprocessor_config.json`, `processor_config.json`, and AutoProcessor mapping.",
                "Verify the loaded processor exposes the image processor expected by mlx-vlm.",
                "Check whether the model repo needs processor files or mlx-vlm needs a fallback path.",
                "Reproduce with the single affected model before judging output quality.",
            ]
        else:
            checklist = [
                "Inspect the exported error package, load phase, and traceback owner.",
                "Check model config, tokenizer files, and weight shape compatibility.",
                "Compare against installed mlx, mlx-vlm, mlx-lm, transformers, and tokenizers versions.",
                "Reproduce with the single affected model before judging output quality.",
            ]
    if checklist is None:
        checklist = [
            "Reproduce with the cluster command.",
            "Confirm owner attribution against traceback, tokenizer, processor, and runtime evidence.",
            "Add a focused regression check for the affected symptom family.",
        ]
    return checklist


def _issue_fix_checklist_section(cluster: IssueCluster) -> list[str]:
    """Build the Fix Checklist section for one issue cluster."""
    parts = ["", "## Fix Checklist", ""]
    parts.extend(f"- [ ] {item}" for item in _issue_fix_checklist_items(cluster))
    parts.append("")
    return parts


def _issue_acceptance_section(cluster: IssueCluster) -> list[str]:
    """Build the expected fixed-state section for one issue cluster."""
    parts = ["", "## Expected Fix Signal", ""]
    parts.append(f"- [ ] {MARKDOWN_ESCAPER.escape(cluster.acceptance_signal)}")
    parts.append(
        "- [ ] The cluster rerun no longer produces this "
        f"`{MARKDOWN_ESCAPER.escape(cluster.cluster_id)}` maintainer-triage cluster."
    )
    parts.append("")
    return parts


def _issue_environment_section(
    *,
    versions: LibraryVersionDict,
    system_info: dict[str, str],
) -> list[str]:
    """Build a standalone Environment section for issue drafts."""
    rows = [
        (component, MARKDOWN_ESCAPER.escape(value))
        for component, value in _collect_report_component_rows(
            versions=versions,
            system_info=system_info,
            library_names=_DIAGNOSTICS_LIB_NAMES,
            system_keys=_DIAGNOSTICS_SYSTEM_KEYS,
        )
    ]
    parts = ["", "## Appendix: Environment", ""]
    if rows:
        parts.extend(
            tabulate(rows, headers=["Component", "Version"], tablefmt="github").splitlines()
        )
    else:
        parts.append("- Environment details were not available.")
    parts.append("")
    return parts


def _build_issue_markdown(
    cluster: IssueCluster,
    *,
    versions: LibraryVersionDict,
    system_info: dict[str, str],
    repro_bundles: Mapping[str, Path],
    run_args: argparse.Namespace | None,
) -> list[str]:
    """Build a complete clustered GitHub issue draft."""
    parts = [f"# {_issue_title(cluster)}"]
    parts.extend(_issue_summary_section(cluster))
    parts.extend(_issue_affected_models_section(cluster, repro_bundles=repro_bundles))
    parts.extend(_issue_minimal_evidence_section(cluster))
    parts.extend(
        _issue_repro_section(
            cluster,
            repro_bundles=repro_bundles,
            run_args=run_args,
        )
    )
    parts.extend(_issue_acceptance_section(cluster))
    parts.extend(_issue_fix_checklist_section(cluster))
    parts.extend(_issue_likely_root_cause_section(cluster))
    parts.extend(_issue_environment_section(versions=versions, system_info=system_info))
    parts.extend(_issue_evidence_section(cluster))
    return parts


def _write_issue_index(
    *,
    issues_dir: Path,
    clusters: Sequence[IssueCluster],
    repro_bundles: Mapping[str, Path],
) -> None:
    """Write the issue queue index for the current run."""
    parts = ["# Issue Queue", ""]
    if not clusters:
        parts.extend(["No issue drafts were generated for this run.", ""])
        issues_dir.joinpath("index.md").write_text("\n".join(parts), encoding="utf-8")
        return

    parts.extend(
        _render_issue_queue_table(
            clusters,
            escape_text=MARKDOWN_ESCAPER.escape,
            issue_link_for_cluster=lambda cluster: (
                f"[issue draft]({quote(cluster.issue_filename)})"
            ),
            evidence_link_for_cluster=lambda cluster: _issue_cluster_bundle_link(
                cluster,
                repro_bundles,
            ),
        )
    )
    parts.append("")
    issues_dir.joinpath("index.md").write_text("\n".join(parts) + "\n", encoding="utf-8")


def _generate_github_issue_reports(
    *,
    diagnostics_snapshot: DiagnosticsSnapshot,
    output_dir: Path,
    versions: LibraryVersionDict,
    system_info: dict[str, str],
    repro_bundles: Mapping[str, Path],
    run_args: argparse.Namespace | None,
) -> Mapping[str, Path]:
    """Generate clustered GitHub issue drafts and the issue queue index."""
    issues_dir = output_dir / "issues"
    if not issues_dir.exists():
        issues_dir.mkdir(parents=True, exist_ok=True)

    for stale in issues_dir.glob("issue_*.md"):
        stale.unlink()
    stale_index = issues_dir / "index.md"
    if stale_index.exists():
        stale_index.unlink()

    clusters = _build_issue_clusters(diagnostics_snapshot)
    generated_reports: dict[str, Path] = {}
    for cluster in clusters:
        issue_path = issues_dir / cluster.issue_filename
        issue_path.write_text(
            "\n".join(
                _build_issue_markdown(
                    cluster,
                    versions=versions,
                    system_info=system_info,
                    repro_bundles=repro_bundles,
                    run_args=run_args,
                )
            )
            + "\n",
            encoding="utf-8",
        )
        generated_reports[cluster.cluster_id] = issue_path

    _write_issue_index(issues_dir=issues_dir, clusters=clusters, repro_bundles=repro_bundles)

    return generated_reports


def _write_diagnostics_and_repro_artifacts(
    *,
    args: argparse.Namespace,
    results: list[PerformanceResult],
    library_versions: LibraryVersionDict,
    system_info: dict[str, str],
    prompt: str,
    image_path: Path | None,
    diagnostics_path: Path,
    history_path: Path,
    previous_history: HistoryRunRecord | None,
    current_history: HistoryRunRecord | None,
) -> DiagnosticsArtifacts:
    """Export repro bundles and diagnostics markdown after history append."""
    preflight_issues = _get_run_preflight_issues(args)
    diagnostics_snapshot = _build_diagnostics_snapshot(
        results=results,
        prompt=prompt,
        preflight_issues=preflight_issues,
    )
    repro_bundles = export_failure_repro_bundles(
        results=results,
        output_dir=diagnostics_path.parent.parent / "repro_bundles",
        run_args=args,
        versions=library_versions,
        system_info=system_info,
        prompt=prompt,
        image_path=image_path,
    )
    if repro_bundles:
        logger.info("Repro bundles written for %d issue-linked model(s).", len(repro_bundles))
        for model_name, bundle_path in sorted(repro_bundles.items()):
            log_file_path(bundle_path, label=f"   Repro Bundle ({model_name}):")

    diagnostics_written = generate_diagnostics_report(
        results=results,
        filename=diagnostics_path,
        versions=library_versions,
        system_info=system_info,
        prompt=prompt,
        image_path=image_path,
        run_args=args,
        history=DiagnosticsHistoryInputs(
            history_path=history_path,
            previous_history=previous_history,
            current_history=current_history,
            preflight_issues=preflight_issues,
        ),
        repro_bundles=repro_bundles,
        diagnostics_snapshot=diagnostics_snapshot,
    )
    issue_reports = _generate_github_issue_reports(
        diagnostics_snapshot=diagnostics_snapshot,
        output_dir=diagnostics_path.parent.parent,  # Diagnostics is in output/reports/ now
        versions=library_versions,
        system_info=system_info,
        repro_bundles=repro_bundles,
        run_args=args,
    )
    if issue_reports:
        logger.info("GitHub Issue reports generated for %d issue(s).", len(issue_reports))
        for issue_key, issue_path in sorted(issue_reports.items()):
            log_file_path(issue_path, label=f"   Issue Report ({issue_key}):")

    return DiagnosticsArtifacts(
        snapshot=diagnostics_snapshot,
        diagnostics_written=diagnostics_written,
        repro_bundles=repro_bundles,
        issue_reports=issue_reports,
    )


def _write_environment_failure_diagnostics(
    *,
    args: argparse.Namespace,
    library_versions: LibraryVersionDict,
    error_message: str,
) -> None:
    """Write a minimal diagnostics report when runtime deps are missing."""
    diagnostics_path: Path = args.output_diagnostics.resolve()
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    system_info = get_system_characteristics()
    parts = _build_environment_failure_diagnostics(
        error_message=error_message,
        versions=library_versions,
        system_info=system_info,
    )
    try:
        diagnostics_path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    except OSError:
        logger.exception("Failed to write environment diagnostics report to %s", diagnostics_path)
    else:
        log_file_path(diagnostics_path, label="   Diagnostics:  ")


def _generate_reports_and_log_outputs(
    inputs: ReportGenerationInputs,
) -> None:
    """Generate reports and log the emitted artifact paths."""
    gallery_output_path = inputs.args.output_gallery_markdown.resolve()
    review_output_path = inputs.review_output_path
    report_jobs: tuple[tuple[str, Callable[[], None]], ...] = (
        (
            "html",
            lambda: generate_html_report(
                results=inputs.results,
                filename=inputs.args.output_html,
                versions=inputs.library_versions,
                prompt=inputs.prompt,
                total_runtime_seconds=inputs.overall_time,
                image_path=inputs.image_path,
                report_context=inputs.report_context,
            ),
        ),
        (
            "markdown",
            lambda: generate_markdown_report(
                results=inputs.results,
                filename=inputs.args.output_markdown,
                versions=inputs.library_versions,
                prompt=inputs.prompt,
                total_runtime_seconds=inputs.overall_time,
                report_context=inputs.report_context,
                gallery_filename=gallery_output_path,
                review_filename=review_output_path,
                log_filename=inputs.log_output_path,
            ),
        ),
        (
            "markdown_gallery",
            lambda: generate_markdown_gallery_report(
                results=inputs.results,
                filename=gallery_output_path,
                prompt=inputs.prompt,
                metadata=inputs.metadata,
                report_context=inputs.report_context,
            ),
        ),
        (
            "review",
            lambda: generate_review_report(
                results=inputs.results,
                filename=review_output_path,
                prompt=inputs.prompt,
                report_context=inputs.report_context,
                log_filename=inputs.log_output_path,
                gallery_filename=gallery_output_path,
            ),
        ),
    )

    try:
        for report_name, report_job in report_jobs:
            try:
                report_job()
            except (OSError, ValueError) as err:
                logger.exception("Failed to generate %s report.", report_name)
                _write_report_failure_jsonl(
                    filename=inputs.jsonl_output_path,
                    failed_report=report_name,
                    error=err,
                )

        generate_tsv_report(
            results=inputs.results,
            filename=inputs.args.output_tsv,
            report_context=inputs.report_context,
        )
        save_jsonl_report(
            inputs.results,
            inputs.args.output_jsonl,
            prompt=inputs.prompt,
            system_info=inputs.system_info,
            library_versions=inputs.library_versions,
            runtime_fingerprint=inputs.runtime_fingerprint,
        )

        logger.info("")
        log_success("Reports successfully generated:", prefix="📊")
        report_paths = (
            (inputs.args.output_html, "   HTML Report:"),
            (inputs.args.output_markdown, "   Markdown Report:"),
            (inputs.args.output_gallery_markdown, "   Gallery Report: "),
            (inputs.args.output_review, "   Review Report:  "),
            (inputs.args.output_tsv, "   TSV Report:   "),
            (inputs.args.output_jsonl, "   JSONL Report: "),
        )
        for path, label in report_paths:
            log_file_path(path, label=label)

        log_file_path(inputs.log_output_path, label="   Log File:")
        if inputs.env_output_path.exists():
            log_file_path(inputs.env_output_path, label="   Environment:")
    except (OSError, ValueError):
        logger.exception("Failed to generate reports.")


def _prune_repro_bundles(
    repro_dir: Path,
    max_age_days: int,
    *,
    max_runs: int = 10,
) -> int:
    """Remove old repro bundles by age and run count. Returns count removed.

    Bundles are JSON files named ``<timestamp>_<seq>_<model>_<sig>.json``.
    The timestamp prefix groups files into runs.  We keep the newest
    *max_runs* run groups and also remove any individual file older than
    *max_age_days*.
    """
    if not repro_dir.is_dir():
        return 0

    entries = sorted(repro_dir.iterdir(), reverse=True)
    if not entries:
        return 0

    age_cutoff = time.time() - max_age_days * 86400 if max_age_days > 0 else 0.0

    # Identify run-group timestamps from file names (first 16 chars like
    # ``20260417T234549Z``).
    seen_runs: set[str] = set()
    run_order: list[str] = []
    for entry in entries:
        if entry.is_dir() or not entry.name.endswith(".json"):
            continue
        run_prefix = entry.name[:16]
        if run_prefix not in seen_runs:
            seen_runs.add(run_prefix)
            run_order.append(run_prefix)

    keep_prefixes = set(run_order[:max_runs]) if max_runs > 0 else seen_runs

    removed = 0
    for entry in entries:
        if entry.is_dir():
            try:
                if not any(entry.iterdir()):
                    entry.rmdir()
            except OSError:
                pass
            continue
        if not entry.name.endswith(".json"):
            continue
        run_prefix = entry.name[:16]
        prune = run_prefix not in keep_prefixes
        if not prune and age_cutoff > 0:
            try:
                prune = entry.stat().st_mtime < age_cutoff
            except OSError:
                continue
        if prune:
            try:
                entry.unlink()
                removed += 1
            except OSError:
                logger.debug("Failed to remove old repro bundle: %s", entry)
    return removed


def _clean_stale_toplevel_reports(output_dir: Path, reports_dir: Path) -> int:
    """Remove stale top-level report copies superseded by reports/ versions.

    Earlier versions of the tool wrote reports directly into ``output/``.
    Now they live under ``output/reports/``.  This removes any leftover
    top-level duplicates so only the canonical copies remain.
    """
    stale_names = (
        "results.md",
        "results.html",
        "results.tsv",
        "diagnostics.md",
        "model_gallery.md",
        "review.md",
    )
    removed = 0
    for name in stale_names:
        stale = output_dir / name
        canonical = reports_dir / name
        if stale.is_file() and canonical.is_file():
            try:
                stale.unlink()
                removed += 1
            except OSError:
                logger.debug("Failed to remove stale top-level report: %s", stale)
    return removed


# ---------- Phase 5: Automatic Differential Reruns ----------

RERUN_TRIAGE_PROMPT: Final[str] = "Describe this image briefly."
RERUN_TRIAGE_MAX_TOKENS: Final[int] = 100
RERUN_TRIAGE_TIMEOUT: Final[float] = 60.0

_RERUN_ELIGIBLE_VERDICTS: Final[frozenset[str]] = frozenset(
    {
        "unknown_runtime_anomaly",
        "runtime_failure",
    }
)


def _select_rerun_candidates(
    results: list[PerformanceResult],
) -> list[PerformanceResult]:
    """Select models that merit a differential rerun for secondary evidence.

    Eligible models are those with verdicts indicating uncertain or transient
    failures.  Successful models and models with deterministic failures
    (e.g. ``harness``, ``model_shortcoming``) are never rerun.
    """
    candidates: list[PerformanceResult] = []
    for result in results:
        review = result.quality_analysis
        verdict = getattr(review, "verdict", None) if review else None
        # Also include failed models with no verdict (pure runtime failures)
        if (not result.success and verdict is None) or verdict in _RERUN_ELIGIBLE_VERDICTS:
            candidates.append(result)
    return candidates


def _build_rerun_evidence(
    rerun_result: PerformanceResult,
    *,
    rerun_prompt: str,
) -> RerunEvidence:
    """Build a RerunEvidence summary from a rerun PerformanceResult."""
    gen_chars: int | None = None
    if rerun_result.generation is not None:
        text = getattr(rerun_result.generation, "text", None)
        if text is not None:
            gen_chars = len(text)
    return RerunEvidence(
        rerun_success=rerun_result.success,
        rerun_verdict=None,  # Not classified — secondary evidence only
        rerun_error_code=rerun_result.error_code,
        rerun_generated_chars=gen_chars,
        rerun_generation_time=rerun_result.generation_time,
        rerun_prompt=rerun_prompt,
    )


def _run_differential_reruns(
    candidates: list[PerformanceResult],
    args: argparse.Namespace,
    image_path: Path,
) -> list[PerformanceResult]:
    """Rerun candidate models with a simple triage prompt for secondary evidence.

    Returns the original results list with ``rerun_evidence`` populated on
    candidates that were rerun.  First-pass data is never overwritten (G3).
    """
    if not candidates:
        return []

    logger.info(
        "Running differential reruns for %d triage candidate(s)...",
        len(candidates),
    )
    updated: list[PerformanceResult] = []
    for idx, result in enumerate(candidates, start=1):
        logger.info(
            "  Rerun %d/%d: %s",
            idx,
            len(candidates),
            result.model_name,
        )
        params = ProcessImageParams(
            model_identifier=result.model_name,
            image_path=image_path,
            prompt=RERUN_TRIAGE_PROMPT,
            max_tokens=RERUN_TRIAGE_MAX_TOKENS,
            temperature=0.0,
            timeout=RERUN_TRIAGE_TIMEOUT,
            verbose=False,
            trust_remote_code=getattr(args, "trust_remote_code", True),
            top_p=getattr(args, "top_p", 1.0),
            min_p=getattr(args, "min_p", 0.0),
            top_k=getattr(args, "top_k", 0),
            repetition_penalty=getattr(args, "repetition_penalty", None),
            repetition_context_size=getattr(args, "repetition_context_size", 20),
            lazy=getattr(args, "lazy_load", False),
            max_kv_size=getattr(args, "max_kv_size", None),
            kv_bits=getattr(args, "kv_bits", None),
            kv_quant_scheme=cast(
                'Literal["uniform", "turboquant"]',
                getattr(args, "kv_quant_scheme", DEFAULT_KV_QUANT_SCHEME),
            ),
            kv_group_size=getattr(args, "kv_group_size", 64),
            quantized_kv_start=getattr(
                args,
                "quantized_kv_start",
                DEFAULT_QUANTIZED_KV_START,
            ),
        )
        rerun_result = process_image_with_model(params)
        evidence = _build_rerun_evidence(rerun_result, rerun_prompt=RERUN_TRIAGE_PROMPT)
        updated_result = replace(result, rerun_evidence=evidence)
        logger.info(
            "  Rerun %s: %s (chars=%s)",
            result.model_name,
            "ok" if evidence.rerun_success else "fail",
            evidence.rerun_generated_chars,
        )
        updated.append(updated_result)
    return updated


def finalize_execution(
    *,
    args: argparse.Namespace,
    results: list[PerformanceResult],
    library_versions: LibraryVersionDict,
    overall_start_time: float,
    prompt: str,
    image_path: Path | None = None,
    metadata: MetadataDict | None = None,
) -> None:
    """Output summary statistics, generate reports, and display timing information."""
    overall_time: float = time.perf_counter() - overall_start_time
    if results:
        print_cli_section("Performance Summary")
        print_model_stats(results)

        # Log summary with failure bucketing for diagnostics
        log_summary(results, prompt=prompt)

        # Gather system characteristics for reports
        system_info = get_system_characteristics()
        runtime_fingerprint = collect_runtime_fingerprint()
        report_context = _build_report_render_context(
            results=results,
            prompt=prompt,
            system_info=system_info,
            preflight_issues=_get_run_preflight_issues(args),
        )

        # Prepare output paths
        tsv_output_path: Path = args.output_tsv.resolve()
        jsonl_output_path: Path = args.output_jsonl.resolve()
        diagnostics_path: Path = args.output_diagnostics.resolve()
        log_output_path: Path = args.output_log.resolve()
        env_output_path: Path = args.output_env.resolve()
        history_path = _history_path_for_jsonl(jsonl_output_path)
        previous_history = _load_latest_history_record(history_path)

        for output_path in (
            args.output_html.resolve(),
            args.output_markdown.resolve(),
            args.output_gallery_markdown.resolve(),
            args.output_review.resolve(),
            tsv_output_path,
            jsonl_output_path,
        ):
            output_path.parent.mkdir(parents=True, exist_ok=True)

        _generate_reports_and_log_outputs(
            ReportGenerationInputs(
                args=args,
                results=results,
                library_versions=library_versions,
                prompt=prompt,
                metadata=metadata,
                overall_time=overall_time,
                image_path=image_path,
                system_info=system_info,
                report_context=report_context,
                jsonl_output_path=jsonl_output_path,
                log_output_path=log_output_path,
                env_output_path=env_output_path,
                review_output_path=args.output_review.resolve(),
                runtime_fingerprint=runtime_fingerprint,
            ),
        )

        # Append run history (append-only) and compare with previous run
        current_history = append_history_record(
            history_path=history_path,
            results=results,
            prompt=prompt,
            system_info=system_info,
            library_versions=library_versions,
            image_path=image_path,
            runtime_fingerprint=runtime_fingerprint,
        )

        log_file_path(history_path, label="   History:     ")

        _log_history_comparison(previous_history, current_history)

        # Generate diagnostics report after history append so regression/retry
        # context in diagnostics.md reflects this run.
        diagnostics_artifacts = _write_diagnostics_and_repro_artifacts(
            args=args,
            results=results,
            library_versions=library_versions,
            system_info=system_info,
            prompt=prompt,
            image_path=image_path,
            diagnostics_path=diagnostics_path,
            history_path=history_path,
            previous_history=previous_history,
            current_history=current_history,
        )
        _log_maintainer_summary(
            artifacts=diagnostics_artifacts,
            diagnostics_path=diagnostics_path,
        )

        # Prune old repro bundles
        prune_days = getattr(args, "prune_repro_days", 90)
        if prune_days > 0:
            repro_dir = diagnostics_path.parent.parent / "repro_bundles"
            pruned = _prune_repro_bundles(repro_dir, prune_days)
            if pruned:
                logger.info("Pruned %d repro bundle(s) older than %d days.", pruned, prune_days)

        # Remove stale top-level report copies superseded by reports/ subdirectory
        reports_dir = diagnostics_path.parent
        output_dir = reports_dir.parent
        stale_removed = _clean_stale_toplevel_reports(output_dir, reports_dir)
        if stale_removed:
            logger.info(
                "Removed %d stale top-level report(s) superseded by reports/.", stale_removed
            )
    else:
        log_warning_note("No models processed. No performance summary generated.")
        logger.info("Skipping report generation as no models were processed.")

    print_cli_section("Final Summary")
    log_blank()
    logger.info(
        "⏱  Overall runtime: %s",
        format_overall_runtime(overall_time),
        extra={"style_hint": LogStyles.METRIC_LABEL},
    )
    print_version_info(library_versions)


# =============================================================================
# MAIN ORCHESTRATION & CLI (Argument parsing, execution flow)
# =============================================================================


def main(args: argparse.Namespace) -> None:
    """Run CLI execution for MLX VLM model check."""
    overall_start_time: float = time.perf_counter()
    library_versions: LibraryVersionDict | None = None
    try:
        library_versions = setup_environment(args)
        # Validate all CLI arguments early to fail fast (after logging setup)
        validate_cli_arguments(args)

        # Apply eval-mode overrides (only when user hasn't explicitly set values)
        eval_mode = getattr(args, "eval_mode", "stress")
        if eval_mode == "triage" and args.max_tokens == DEFAULT_MAX_TOKENS:
            args.max_tokens = 200
        elif eval_mode == "quality" and args.max_tokens == DEFAULT_MAX_TOKENS:
            args.max_tokens = 1000

        print_cli_header("MLX Vision Language Model Check")

        image_path = find_and_validate_image(args)

        metadata = handle_metadata(image_path, args)

        prompt = prepare_prompt(args, metadata)

        # Handle dry-run mode: show what would be run and exit
        if getattr(args, "dry_run", False):
            _handle_dry_run(args, image_path, prompt, library_versions)
            return

        # Hard-fail before any model execution when core runtime deps are unavailable.
        _raise_for_missing_runtime_dependencies()

        results = process_models(args, image_path, prompt=prompt, metadata=metadata)

        # Phase 5: Differential reruns for triage-worthy models
        if getattr(args, "rerun_triage", False):
            candidates = _select_rerun_candidates(results)
            if candidates:
                updated = _run_differential_reruns(candidates, args, image_path)
                # Merge rerun evidence back into results list
                rerun_map = {r.model_name: r for r in updated}
                results = [rerun_map.get(r.model_name, r) for r in results]

        finalize_execution(
            args=args,
            results=results,
            library_versions=library_versions,
            overall_start_time=overall_start_time,
            prompt=prompt,
            image_path=image_path,
            metadata=metadata,
        )
    except KeyboardInterrupt:
        logger.warning("Execution interrupted by user.")
        sys.exit(130)
    except SystemExit:
        raise
    except RuntimeError as main_err:
        message = str(main_err)
        if message.startswith("Required runtime dependencies unavailable:"):
            log_failure("Required runtime dependencies unavailable; no models were run.")
            log_warning_note(
                "Install/repair the missing packages and re-run. Environment details and a "
                "minimal diagnostics report have been written for triage.",
            )
            _write_environment_failure_diagnostics(
                args=args,
                library_versions=library_versions or get_library_versions(),
                error_message=message,
            )
        else:
            logger.critical("Fatal error in main execution: %s", main_err, exc_info=True)
        sys.exit(1)
    except (OSError, ValueError) as main_err:
        logger.critical("Fatal error in main execution: %s", main_err, exc_info=True)
        sys.exit(1)


def _handle_dry_run(
    args: argparse.Namespace,
    image_path: Path,
    prompt: str,
    library_versions: LibraryVersionDict,
) -> None:
    """Handle --dry-run mode: display what would be run without invoking models.

    Args:
        args: Parsed command line arguments
        image_path: Resolved image path
        prompt: Generated or user-provided prompt
        library_versions: Dictionary of library versions
    """
    print_cli_section("Dry Run Mode")
    logger.info("🔍 Validating configuration without running models...")
    log_blank()

    # Image info
    logger.info("📷 Image: %s", image_path)
    if image_path.exists():
        size_mb = image_path.stat().st_size / (1024 * 1024)
        logger.info("   Size: %.2f MB", size_mb)
    log_blank()

    # Prompt info
    logger.info("💬 Prompt:")
    # Wrap prompt for readability
    wrapped = textwrap.wrap(prompt, width=90)
    max_lines = FORMATTING.max_prompt_preview_lines
    for line in wrapped[:max_lines]:
        logger.info("   %s", line)
    if len(wrapped) > max_lines:
        logger.info("   ... (%d more lines)", len(wrapped) - max_lines)
    log_blank()

    # Discover models
    if args.models:
        model_identifiers = args.models
        logger.info("📦 Models specified explicitly:")
    else:
        model_identifiers = get_cached_model_ids()
        logger.info("📦 Models discovered in cache:")

    # Apply exclusions
    excluded = set(args.exclude or [])
    if excluded:
        before_count = len(model_identifiers)
        model_identifiers = [m for m in model_identifiers if m not in excluded]
        logger.info("   (Excluded %d models via --exclude)", before_count - len(model_identifiers))

    if not model_identifiers:
        logger.warning("   ⚠️  No models to process!")
    else:
        for idx, model_id in enumerate(model_identifiers, start=1):
            logger.info("   %2d. %s", idx, model_id)

    log_blank()
    logger.info("📊 Would process %d model(s)", len(model_identifiers))
    log_blank()

    # Library versions
    print_version_info(library_versions)

    log_blank()
    log_success("Dry run complete. No models were invoked.", prefix="✅")


def main_cli() -> None:
    """CLI entry point for the MLX VLM checker script."""
    parser = _build_cli_parser()

    # Parse arguments
    args: argparse.Namespace = parser.parse_args()

    # If neither --folder nor --image is specified, assume default folder
    if getattr(args, "folder", None) is None and getattr(args, "image", None) is None:
        args.folder = DEFAULT_FOLDER
        logger.info(
            "No --folder or --image specified. Assuming default folder: %s",
            DEFAULT_FOLDER,
        )
        print_cli_section("No image or folder specified")
        logger.info(
            "Assuming default folder: %s. To override, specify --folder or --image.",
            DEFAULT_FOLDER,
        )
        if not DEFAULT_FOLDER.exists():
            logger.warning(
                "Default folder does not exist: %s — create it or use --folder / --image.",
                DEFAULT_FOLDER,
            )

    # Print all command-line arguments if verbose is set
    if getattr(args, "verbose", False):
        print_cli_section("Command Line Parameters")
        for arg_name, arg_value in sorted(vars(args).items()):
            logger.info("  %s: %s", arg_name, arg_value)

    # Load quality configuration
    load_quality_config(getattr(args, "quality_config", None))

    main(args)


class _ConditionalDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Show argparse defaults only when they are real user-facing defaults."""

    def _get_help_string(self, action: argparse.Action) -> str:
        help_text = action.help or ""
        if action.default in (None, argparse.SUPPRESS):
            return help_text
        return super()._get_help_string(action) or help_text


def _build_cli_parser() -> argparse.ArgumentParser:
    """Build and return the command-line parser for the CLI entry point."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="MLX VLM Model Checker",
        formatter_class=_ConditionalDefaultsHelpFormatter,
    )

    def _output_help(label: str) -> str:
        return f"Output {label} report filename."

    # Add arguments (separated for clarity)
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "-f",
        "--folder",
        type=Path,
        default=None,
        help=(
            "Folder to scan. Requires a path when provided. The most recently modified "
            "image file in the folder will be used. If both --folder and --image are "
            f"omitted, the most recently modified image in {DEFAULT_FOLDER} will be used."
        ),
    )
    group.add_argument(
        "-i",
        "--image",
        type=Path,
        default=None,
        help=("Path to a specific image file to process directly. Requires a path when provided."),
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=DEFAULT_HTML_OUTPUT,
        help=_output_help("HTML"),
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=DEFAULT_MD_OUTPUT,
        help=_output_help("GitHub Markdown"),
    )
    parser.add_argument(
        "--output-gallery-markdown",
        type=Path,
        default=DEFAULT_GALLERY_MD_OUTPUT,
        help=("Output GitHub Markdown gallery filename for the standalone review artifact."),
    )
    parser.add_argument(
        "--output-review",
        type=Path,
        default=DEFAULT_REVIEW_MD_OUTPUT,
        help="Output Markdown review digest filename.",
    )
    parser.add_argument(
        "--output-tsv",
        type=Path,
        default=DEFAULT_TSV_OUTPUT,
        help=_output_help("TSV (tab-separated values)"),
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=DEFAULT_JSONL_OUTPUT,
        help=_output_help("JSONL"),
    )
    parser.add_argument(
        "--output-log",
        type=Path,
        default=DEFAULT_LOG_OUTPUT,
        help=(
            "Command line output log filename (overwritten each run). "
            "Use different path for tests/debug runs."
        ),
    )
    parser.add_argument(
        "--output-env",
        type=Path,
        default=DEFAULT_ENV_OUTPUT,
        help="Environment log filename (pip freeze, conda list for reproducibility).",
    )
    parser.add_argument(
        "--output-diagnostics",
        type=Path,
        default=DEFAULT_DIAGNOSTICS_OUTPUT,
        help=(
            "Diagnostics report filename (Markdown). "
            "Generated when failures or harness issues are detected. "
            "Structured for filing upstream issues against mlx-vlm / mlx / transformers."
        ),
    )
    parser.add_argument(
        "-m",
        "--models",
        action="extend",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Specify models by ID/path. Overrides cache scan. "
            "May be provided multiple times; model lists accumulate."
        ),
    )
    parser.add_argument(
        "-e",
        "--exclude",
        action="extend",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Exclude models by ID/path from the model list "
            "(works with both explicit --models and cache scan). "
            "May be provided multiple times; exclusions accumulate."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow custom code from Hub models (SECURITY RISK). "
            "Use --no-trust-remote-code to disable."
        ),
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Model revision (branch, tag, or commit hash) for version pinning.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Path to LoRA adapter weights to apply on top of the base model.",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default=None,
        help=(
            "Prompt text to send to the model. Requires text when provided. If omitted, "
            "an automatic metadata-verification prompt is used."
        ),
    )
    parser.add_argument(
        "--resize-shape",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Resize image input before processor handling. "
            "Provide 1 integer for square resize or 2 for height width after a "
            "single flag occurrence."
        ),
    )
    parser.add_argument(
        "--eos-tokens",
        action="extend",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Additional EOS tokens to stop on. Supports escaped values like \\n. "
            "May be provided multiple times; token lists accumulate."
        ),
    )
    parser.add_argument(
        "--skip-special-tokens",
        action="store_true",
        default=False,
        help="Skip tokenizer special tokens in the detokenized output.",
    )
    parser.add_argument(
        "--processor-kwargs",
        type=_parse_processor_kwargs_arg,
        default=None,
        help=(
            "Extra processor kwargs as a JSON object. "
            'Example: --processor-kwargs \'{"cropping": false, "max_patches": 3}\''
        ),
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=False,
        help="Enable thinking mode in the upstream chat template and generation flow.",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=None,
        help="Maximum number of thinking tokens before forcing the end token.",
    )
    parser.add_argument(
        "--thinking-start-token",
        type=str,
        default=None,
        help="Token marking the start of a thinking block, such as <think>.",
    )
    parser.add_argument(
        "--thinking-end-token",
        type=str,
        default=DEFAULT_THINKING_END_MARKER,
        help="Token marking the end of a thinking block when thinking mode is enabled.",
    )

    parser.add_argument(
        "-d",
        "--detailed-metrics",
        action="store_true",
        default=False,
        help=("Show expanded multi-line metrics block; ignored unless --verbose is also set."),
    )
    parser.add_argument(
        "--eval-mode",
        choices=["stress", "triage", "quality"],
        default="stress",
        help=(
            "Evaluation lane: 'stress' (default) = full cataloguing prompt with metadata, "
            "500 token cap; 'triage' = short prompt, minimal context, 200 token cap, "
            "pass/fail only; 'quality' = cataloguing prompt with generous 1000 token cap, "
            "scored on output quality."
        ),
    )
    parser.add_argument(
        "-x",
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Max new tokens to generate.",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus sampling parameter (0.0-1.0). Lower values = more focused output.",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.0,
        help="Minimum-probability sampling floor (0.0-1.0). 0.0 disables min-p filtering.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-k sampling limit. 0 disables top-k filtering.",
    )
    parser.add_argument(
        "-r",
        "--repetition-penalty",
        type=float,
        default=None,
        help="Penalize repeated tokens (>1.0 discourages repetition). None = no penalty.",
    )
    parser.add_argument(
        "--repetition-context-size",
        type=int,
        default=20,
        help="Context window size for repetition penalty.",
    )
    parser.add_argument(
        "-L",
        "--lazy-load",
        action="store_true",
        default=False,
        help="Use lazy loading for models (loads weights on-demand, reduces peak memory).",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        default=None,
        help="Maximum KV cache size (limits memory for long sequences). None = no limit.",
    )
    parser.add_argument(
        "-b",
        "--kv-bits",
        type=float,
        default=None,
        help=(
            "Quantize KV cache to N bits. Uniform supports 2, 3, 4, 5, 6, "
            "or 8; fractional values such as 3.5 use TurboQuant automatically."
        ),
    )
    parser.add_argument(
        "--kv-quant-scheme",
        choices=["uniform", "turboquant"],
        default=DEFAULT_KV_QUANT_SCHEME,
        help=(
            "KV cache quantization backend. Default: uniform. "
            "Use turboquant to match upstream TurboQuant cache handling."
        ),
    )
    parser.add_argument(
        "-g",
        "--kv-group-size",
        type=int,
        default=64,
        help="Quantization group size for KV cache.",
    )
    parser.add_argument(
        "--quantized-kv-start",
        type=int,
        default=DEFAULT_QUANTIZED_KV_START,
        help=(
            "Start position for KV cache quantization. "
            f"Default: {DEFAULT_QUANTIZED_KV_START} (matches mlx-vlm upstream)."
        ),
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=4096,
        help="Step size for prompt prefill. Default: 4096 (faster than mlx-lm default).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose and debug output (DEBUG logging).",
    )
    parser.add_argument(
        "-T",
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="Timeout in seconds for model operations.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors in output.",
    )
    parser.add_argument(
        "--force-color",
        action="store_true",
        help="Force enable ANSI colors even if not a TTY.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help=(
            "Force a specific CLI output width (columns) for separators and text wrapping. "
            "Overrides terminal detection. Also supported via MLX_VLM_WIDTH env var."
        ),
    )
    parser.add_argument(
        "-c",
        "--quality-config",
        type=Path,
        default=None,
        help="Path to custom quality configuration YAML file.",
    )
    parser.add_argument(
        "--context-marker",
        type=str,
        default="Context:",
        help="Marker used to identify context section in prompt.",
    )
    parser.add_argument(
        "--prune-repro-days",
        type=int,
        default=90,
        help="Delete repro bundles older than N days (default: 90). Set 0 to disable pruning.",
    )
    parser.add_argument(
        "--rerun-triage",
        action="store_true",
        default=False,
        help=(
            "After the first pass, rerun triage-worthy models (runtime failures and "
            "unknown anomalies) with a simple prompt to gather secondary evidence. "
            "First-pass results are never overwritten."
        ),
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Validate arguments and show what would be run without invoking any models. "
            "Lists discovered models, the generated prompt, and image path then exits."
        ),
    )
    return parser


if __name__ == "__main__":
    main_cli()
