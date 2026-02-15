#!/usr/bin/env python3
"""Image analysis and caption generation using MLX Vision Language Models."""

from __future__ import annotations

import argparse
import base64
import contextlib
import dataclasses
import gc
import hashlib
import html
import importlib.metadata
import importlib.util as importlib_util
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
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
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
    Self,
    TextIO,
    TypedDict,
    cast,
    runtime_checkable,
)

import requests
import yaml
from huggingface_hub import HFCacheInfo, scan_cache_dir
from huggingface_hub import __version__ as hf_version
from huggingface_hub.errors import HFValidationError
from tabulate import tabulate
from tzlocal import get_localzone

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

if TYPE_CHECKING:
    import types
    from collections.abc import Iterator

    from mlx.nn import Module
    from mlx_vlm.generate import GenerationResult
    from PIL.Image import Image as PILImage
    from transformers.tokenization_python import PythonBackend
    from transformers.tokenization_utils_tokenizers import TokenizersBackend

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
    "log_metric_tree",
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
    "Core dependency missing: mlx-lm. Please install it (`pip install mlx-lm`)."
)
ERROR_MLX_VLM_RUNTIME_INIT: Final[str] = (
    "Core dependency initialization failed: mlx-vlm could not be imported safely."
)
MLX_IMPORT_PROBE_TIMEOUT_SECONDS: Final[float] = 8.0


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

    # Layout constants
    min_separator_chars: int = 50
    default_decimal_places: int = 2
    markdown_hard_break_spaces: int = 2
    generation_wrap_width: int = 100


@dataclass
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
    min_useful_chars: int = 10  # Minimum chars for useful output
    severe_echo_threshold: float = 0.8  # Echo ratio triggering severe penalty
    moderate_echo_threshold: float = 0.5  # Echo ratio triggering moderate penalty
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

    # Patterns (loaded from config)
    patterns: dict[str, list[str]] | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> QualityThresholds:
        """Create instance from configuration dictionary."""
        thresholds = config.get("thresholds", {})
        patterns = config.get("patterns", {})

        # Filter valid fields for the dataclass
        valid_fields = {f.name for f in dataclasses.fields(cls) if f.name != "patterns"}
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

        return cls(**filtered_thresholds, patterns=patterns)


# Instantiate singletons for runtime use
FORMATTING = FormattingThresholds()
# Default QUALITY instance (will be updated if config is loaded)
QUALITY = QualityThresholds()


def load_quality_config(config_path: Path | None = None) -> None:
    """Load quality configuration from file and update global QUALITY instance.

    If no config_path is provided, looks for 'quality_config.yaml' in the
    same directory as this script.
    """
    # Default path relative to script location (robust to CWD changes)
    if config_path is None:
        # Use resolve() to handle symlinks and ensure absolute path
        script_dir = Path(__file__).resolve().parent
        default_path = script_dir / "quality_config.yaml"
        if default_path.exists():
            config_path = default_path

    if config_path and config_path.exists():
        try:
            with config_path.open("r") as f:
                config = yaml.safe_load(f)
                if config:
                    new_quality = QualityThresholds.from_config(config)
                    # Update existing global instance in-place to avoid 'global' keyword
                    # and ensure all references see the update.
                    for field in dataclasses.fields(QualityThresholds):
                        setattr(QUALITY, field.name, getattr(new_quality, field.name))
                    logger.debug("Loaded quality configuration from %s", config_path)
        except (OSError, yaml.YAMLError) as e:
            logger.warning("Failed to load quality config from %s: %s", config_path, e)
    elif config_path:
        logger.warning("Quality config file not found: %s", config_path)


# Consolidating constants to FORMATTING and other locations
# FORMATTING.min_separator_chars -> FORMATTING.min_separator_chars
# FORMATTING.default_decimal_places -> FORMATTING.default_decimal_places
# GRADE_EMOJIS -> moved below
# FORMATTING.markdown_hard_break_spaces -> FORMATTING.markdown_hard_break_spaces
# IMAGE_OPEN_TIMEOUT -> moved below
# FORMATTING.generation_wrap_width -> FORMATTING.generation_wrap_width
# SUPPORTED_IMAGE_EXTENSIONS -> moved to validation section constants


_temp_logger = logging.getLogger(LOGGER_NAME)


def _truncate_probe_output(text: str, *, max_chars: int = 220) -> str:
    """Collapse probe output into a compact single-line message."""
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return f"{compact[: max_chars - 3]}..."


def _probe_import_runtime(
    *,
    import_target: str,
    error_prefix: str,
    detect_metal_nsrange: bool = False,
) -> str | None:
    """Run a subprocess import probe and return an actionable error message when it fails."""
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

    output_excerpt = _truncate_probe_output(combined_output) if combined_output else "no output"
    return (
        f"{error_prefix} Import probe exited with code "
        f"{probe_result.returncode}. Probe output: {output_excerpt}"
    )


def _probe_mlx_import_runtime() -> str | None:
    """Return None when mlx.core import appears safe; else an actionable message."""
    return _probe_import_runtime(
        import_target="mlx.core",
        error_prefix=ERROR_MLX_RUNTIME_INIT,
        detect_metal_nsrange=True,
    )


def _probe_mlx_vlm_import_runtime() -> str | None:
    """Return None when mlx_vlm import appears safe; else an actionable message."""
    return _probe_import_runtime(
        import_target="mlx_vlm",
        error_prefix=ERROR_MLX_VLM_RUNTIME_INIT,
    )


mx: Any = cast("Any", None)
mlx_probe_error = _probe_mlx_import_runtime()
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

ExifTags: Any
GPSTAGS: Mapping[Any, Any]
TAGS: Mapping[Any, Any]
UnidentifiedImageError: type[Exception]
GPS: Any  # Type annotation for GPS enum (defined below based on Pillow availability)

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

    ExifTags = _ExifTagsUnavailable()
    Image = cast("Any", _ImageUnavailable())
    UnidentifiedImageError = _PILUnavailableError
    GPS = cast("Any", None)  # GPS enum unavailable when Pillow missing
    GPSTAGS = {}
    TAGS = {}
    MISSING_DEPENDENCIES["Pillow"] = ERROR_PILLOW_MISSING
else:
    from PIL import ExifTags as PIL_ExifTags
    from PIL import IptcImagePlugin
    from PIL.ExifTags import GPS as PIL_GPS
    from PIL.ExifTags import GPSTAGS as PIL_GPSTAGS
    from PIL.ExifTags import TAGS as PIL_TAGS

    pillow_version = Image.__version__ if hasattr(Image, "__version__") else NOT_AVAILABLE
    ExifTags = PIL_ExifTags
    GPS = PIL_GPS
    GPSTAGS = PIL_GPSTAGS
    TAGS = PIL_TAGS

# defusedxml is required by Pillow's Image.getxmp() for safe XMP/XML parsing.
# Pulled in transitively via Pillow[xmp] in pyproject.toml, but guard here
# so _extract_xmp_metadata() degrades gracefully with a clear message.
_defusedxml_available: bool
try:
    import defusedxml.ElementTree  # noqa: F401 — imported for availability check

    _defusedxml_available = True
except ImportError:
    _defusedxml_available = False
    _temp_logger.warning(
        "defusedxml not installed — XMP metadata extraction will be disabled. "
        "Install with: pip install 'Pillow[xmp]'",
    )

try:
    import numpy as np

    numpy_version: str = getattr(np, "__version__", NOT_AVAILABLE)
except ImportError:
    numpy_version = NOT_AVAILABLE

vlm_version: str


@dataclass
class _GenerationResultFallback:
    """Fallback structure used when mlx-vlm is unavailable."""

    text: str | None = None
    prompt_tokens: int | None = None
    generation_tokens: int | None = None


def _raise_mlx_vlm_missing(*_args: object, **_kwargs: object) -> NoReturn:
    """Raise a consistent runtime error when mlx-vlm is unavailable."""
    raise RuntimeError(ERROR_MLX_VLM_MISSING)


def _configure_mlx_vlm_fallback(error_message: str) -> None:
    """Set runtime fallbacks when mlx-vlm cannot be imported."""
    # Use Any for fallback functions to avoid type conflicts with stub signatures
    globals()["generate"] = cast("Any", _raise_mlx_vlm_missing)
    globals()["apply_chat_template"] = cast("Any", _raise_mlx_vlm_missing)
    globals()["load"] = cast("Any", _raise_mlx_vlm_missing)
    globals()["load_image"] = cast("Any", _raise_mlx_vlm_missing)
    MISSING_DEPENDENCIES["mlx-vlm"] = error_message


mlx_vlm_probe_error = _probe_mlx_vlm_import_runtime()
if mlx_vlm_probe_error is None:
    try:
        from mlx_vlm.generate import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load, load_image
        from mlx_vlm.version import __version__ as _mlx_vlm_version

        vlm_version = _mlx_vlm_version
    except ImportError:
        vlm_version = NOT_AVAILABLE
        _configure_mlx_vlm_fallback(ERROR_MLX_VLM_MISSING)
else:
    vlm_version = NOT_AVAILABLE
    _configure_mlx_vlm_fallback(mlx_vlm_probe_error)

try:
    importlib.metadata.version("mlx-lm")
except importlib.metadata.PackageNotFoundError:
    MISSING_DEPENDENCIES["mlx-lm"] = ERROR_MLX_LM_MISSING

_transformers_guard_enabled: bool = os.getenv("MLX_VLM_ALLOW_TF", "0") != "1"
if _transformers_guard_enabled:
    # Prevent Transformers from importing heavy backends that can hang on macOS/ARM
    # when they are present in the environment but not needed for MLX workflows.
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
    os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")


# =============================================================================
# TYPE ALIASES & PROTOCOLS
# =============================================================================

type ExifValue = Any  # Pillow yields varied scalar / tuple EXIF types; keep permissive
type ExifDict = dict[str | int, ExifValue]
type MetadataDict = dict[str, str | None]
type PathLike = str | Path
type GPSTupleElement = int | float
type GPSTuple = tuple[GPSTupleElement, GPSTupleElement, GPSTupleElement]
type GPSDict = dict[str, ExifValue]  # GPS EXIF data structure
type SystemProfilerDict = dict[str, list[dict[str, Any]]]  # macOS system_profiler JSON structure
type LibraryVersionDict = dict[str, str | None]  # Library name to version mapping (optional values)
type MetricValue = int | float | str | bool | None  # Common scalar metric variants for metrics


class HistoryModelResultRecord(TypedDict):
    """Per-model success/failure status stored in run history."""

    success: bool
    error_stage: str | None
    error_type: str | None
    error_package: str | None
    failure_phase: NotRequired[str | None]
    error_code: NotRequired[str | None]
    error_signature: NotRequired[str | None]


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


class JsonlTimingRecord(TypedDict):
    """Timing fields emitted per result row in ``results.jsonl``."""

    generation_time_s: float | None
    model_load_time_s: float | None
    total_time_s: float | None


class JsonlMetricsRecord(TypedDict, total=False):
    """Performance metrics emitted for successful generations."""

    prompt_tokens: int
    generation_tokens: int
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


class JsonlMetadataRecord(TypedDict):
    """Shared header row for ``results.jsonl`` format."""

    _type: Literal["metadata"]
    format_version: str
    prompt: str
    system: dict[str, str]
    timestamp: str


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
    error_package: str | None
    error_traceback: str | None
    quality_issues: list[str]
    timestamp: str
    metrics: JsonlMetricsRecord
    timing: JsonlTimingRecord
    generated_text: NotRequired[str]
    quality_analysis: NotRequired[JsonlQualityAnalysisRecord]


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
    cataloging_avg_score: float
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
class DiagnosticsHistoryInputs:
    """Optional history inputs for diagnostics report generation."""

    history_path: Path | None = None
    previous_history: HistoryRunRecord | None = None
    current_history: HistoryRunRecord | None = None
    preflight_issues: tuple[str, ...] = ()


@runtime_checkable
class SupportsGenerationResult(Protocol):  # Minimal attributes we read from GenerationResult
    """Structural subset of GenerationResult accessed by this script.

    Using a Protocol keeps typing resilient to upstream changes in the
    concrete GenerationResult while still giving linters strong guarantees
    about the attributes actually consumed here.

    Note: `time`, `active_memory`, and `cache_memory` attributes are added
    dynamically by our code after generation.
    """

    text: str | None
    prompt_tokens: int | None
    generation_tokens: int | None
    time: float | None  # Dynamically added timing attribute
    active_memory: float | None  # Dynamically added active memory (GB)
    cache_memory: float | None  # Dynamically added cache memory (GB)


class SupportsExifIfd(Protocol):
    """Minimal interface for EXIF objects providing nested IFD access."""

    def get_ifd(self, tag: object) -> Mapping[object, Mapping[object, object]] | None:
        """Retrieve a nested IFD mapping by tag identifier."""


# =============================================================================
# APPLICATION CONSTANTS & DEFAULTS
# =============================================================================

# These constants define default values for various parameters used in the script.
DEFAULT_MAX_TOKENS: Final[int] = 500
DEFAULT_FOLDER: Final[Path] = Path.home() / "Pictures" / "Processed"
# Output paths relative to script's directory (not CWD) for consistency
_SCRIPT_DIR = Path(__file__).parent
DEFAULT_HTML_OUTPUT: Final[Path] = _SCRIPT_DIR / "output" / "results.html"
DEFAULT_MD_OUTPUT: Final[Path] = _SCRIPT_DIR / "output" / "results.md"
DEFAULT_TSV_OUTPUT: Final[Path] = _SCRIPT_DIR / "output" / "results.tsv"
DEFAULT_LOG_OUTPUT: Final[Path] = _SCRIPT_DIR / "output" / "check_models.log"
DEFAULT_JSONL_OUTPUT: Final[Path] = _SCRIPT_DIR / "output" / "results.jsonl"
DEFAULT_ENV_OUTPUT: Final[Path] = _SCRIPT_DIR / "output" / "environment.log"
DEFAULT_DIAGNOSTICS_OUTPUT: Final[Path] = _SCRIPT_DIR / "output" / "diagnostics.md"
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
        active_memory: GPU memory in use (GB), from mx.metal.get_active_memory()
        cache_memory: GPU memory in cache (GB), from mx.metal.get_cache_memory()
        error_package: Which package raised the error (mlx, mlx-vlm, transformers)
        error_traceback: Full traceback for actionable error reports
    """

    model_name: str
    generation: GenerationResult | SupportsGenerationResult | None
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
    quality_issues: str | None = None
    quality_analysis: GenerationQualityAnalysis | None = None
    active_memory: float | None = None
    cache_memory: float | None = None
    error_package: str | None = None
    error_traceback: str | None = None


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


def _merge_captured_output(stdout_text: str, stderr_text: str) -> str | None:
    """Merge captured stdout/stderr into a single diagnostics-friendly block."""
    sections: list[str] = []

    stdout_clean = stdout_text.strip()
    stderr_clean = stderr_text.strip()

    if stdout_clean:
        sections.append("=== STDOUT ===\n" + stdout_clean)
    if stderr_clean:
        sections.append("=== STDERR ===\n" + stderr_clean)

    if not sections:
        return None
    return "\n\n".join(sections)


# Gallery rendering helpers (outside class)
def _gallery_render_error(res: PerformanceResult) -> list[str]:
    out = []
    out.append(f"**Status:** Failed ({res.error_stage})")
    error_msg = str(res.error_message)
    error_msg = re.sub(r"(?<![<(])(https?://[^\s>)]+)", r"<\1>", error_msg)
    max_inline_length = 80
    if len(error_msg) > max_inline_length:
        wrapped_lines = textwrap.wrap(
            error_msg,
            width=76,
            break_long_words=False,
            break_on_hyphens=False,
        )
        out.append("**Error:**")
        out.append("")
        out.extend(f"> {line}" for line in wrapped_lines)
    else:
        out.append(f"**Error:** {error_msg}")
    if res.error_type:
        out.append(f"**Type:** `{res.error_type}`")
    if res.failure_phase:
        out.append(f"**Phase:** `{res.failure_phase}`")
    if res.error_code:
        out.append(f"**Code:** `{res.error_code}`")
    if res.error_package:
        out.append(f"**Package:** `{res.error_package}`")
    if res.error_traceback:
        out.append("")
        out.append("<details>")
        out.append("<summary>Full Traceback (click to expand)</summary>")
        out.append("")
        out.append("```python")
        out.append(res.error_traceback.rstrip())
        out.append("```")
        out.append("")
        out.append("</details>")
    return out


def _gallery_render_success(res: PerformanceResult) -> list[str]:
    out = []
    gen = res.generation
    if gen:
        tps = getattr(gen, "generation_tps", 0)
        tokens = getattr(gen, "generation_tokens", 0)
        out.append(f"**Metrics:** {fmt_num(tps)} TPS | {tokens} tokens")
    out.append("")
    out.append("```text")
    text = str(getattr(res.generation, "text", "")) if res.generation else ""
    out.append(text)
    out.append("```")
    if res.generation:
        analysis = getattr(res.generation, "quality_analysis", None)
        if analysis and analysis.issues:
            out.append("")
            out.append("⚠️ **Quality Warnings:**")
            out.extend(f"- {html.escape(issue)}" for issue in analysis.issues)
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

    __slots__ = ("_fields", "_results")  # Sorted for lint consistency

    def __init__(self, results: list[PerformanceResult]) -> None:
        """Initialize and sort results.

        A shallow copy of ``results`` is taken to guard against external
        mutation after construction.
        """
        self._results = _sort_results_by_time(list(results))
        self._fields: list[str] | None = None

    # Public API -----------------------------------------------------
    @property
    def results(self) -> list[PerformanceResult]:  # Sorted
        """Return results sorted by generation time (fastest first)."""
        return self._results

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
    repetition_penalty: float | None
    repetition_context_size: int
    lazy: bool
    max_kv_size: int | None
    kv_bits: int | None
    kv_group_size: int
    quantized_kv_start: int
    revision: str | None = None
    adapter_path: str | None = None
    prefill_step_size: int | None = None
    context_marker: str = "Context:"


# =============================================================================
# INFRASTRUCTURE: Timeouts, Colors, Logging
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
class TimeoutManager(contextlib.ContextDecorator):
    """Manage a timeout context for code execution (UNIX only)."""

    def __init__(self, seconds: float) -> None:
        """Initialize a timeout manager with a timeout duration.

        Args:
            seconds: The timeout duration in seconds.

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
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
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


# =============================================================================
# UTILITY CLASSES (Colors, Logging, Timeout Management)
# =============================================================================


class Colors:
    """ANSI color codes for terminal output."""

    RESET: Final[str] = "\033[0m"
    BOLD: Final[str] = "\033[1m"
    RED: Final[str] = "\033[91m"
    GREEN: Final[str] = "\033[92m"
    YELLOW: Final[str] = "\033[93m"
    BLUE: Final[str] = "\033[94m"
    MAGENTA: Final[str] = "\033[95m"
    CYAN: Final[str] = "\033[96m"
    WHITE: Final[str] = "\033[97m"
    GRAY: Final[str] = "\033[90m"
    # Honor NO_COLOR / FORCE_COLOR conventions while defaulting to TTY detection
    _enabled: ClassVar[bool] = (
        True
        if os.getenv("FORCE_COLOR", "").lower() in {"1", "true", "yes"}
        else (sys.stderr.isatty() and os.getenv("NO_COLOR") is None)
    )
    _ansi_escape_re: ClassVar[re.Pattern[str]] = ANSI_ESCAPE_RE

    @staticmethod
    def colored(text: str, *color_codes: str) -> str:
        """Return text wrapped in ANSI color codes if enabled."""
        text_str: str = str(text)
        # Filter out None values from color_codes
        filtered_codes: list[str] = [c for c in color_codes if isinstance(c, str)]
        if not Colors._enabled or not filtered_codes:
            return text_str
        color_seq: str = "".join(filtered_codes)
        return f"{color_seq}{text_str}{Colors.RESET}"

    @staticmethod
    def set_enabled(*, enabled: bool) -> None:
        """Globally enable/disable ANSI colors for this process."""
        Colors._enabled = bool(enabled)

    @staticmethod
    def visual_len(text: str) -> int:
        """Return the visual length of text, ignoring ANSI codes."""
        # Remove ANSI codes for accurate width
        return len(Colors._ansi_escape_re.sub("", text))


class LogStyles:
    """String constants describing structured log presentation styles."""

    HEADER: ClassVar[str] = "header"
    SECTION: ClassVar[str] = "section"
    RULE: ClassVar[str] = "rule"
    ERROR: ClassVar[str] = "error"
    SUCCESS: ClassVar[str] = "success"
    WARNING: ClassVar[str] = "warning"
    DETAIL: ClassVar[str] = "detail"
    # New styles for consistent output formatting
    METRIC_LABEL: ClassVar[str] = "metric_label"  # Bold headers for metrics
    METRIC_VALUE: ClassVar[str] = "metric_value"  # Formatted metric values
    GENERATED_TEXT: ClassVar[str] = "generated_text"  # Cyan model output
    FILE_PATH: ClassVar[str] = "file_path"  # Highlighted file paths
    MODEL_NAME: ClassVar[str] = "model_name"  # Magenta model identifiers


class ColoredFormatter(logging.Formatter):
    """A logging formatter that applies color to log messages based on their level and content."""

    LEVEL_COLORS: ClassVar[dict[int, str]] = {
        logging.DEBUG: Colors.GRAY,
        logging.INFO: "",
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.RED + Colors.BOLD,
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with context-aware colors."""
        style_hint: str | None = getattr(record, "style_hint", None)
        original_msg: object = record.msg
        original_args: tuple[object, ...] | Mapping[str, object] | None = record.args

        if style_hint:
            raw_message: str = record.getMessage()
            structured: str = self._format_structured_message(style_hint, raw_message, record)
            record.msg = structured
            record.args = ()

        try:
            msg: str = super().format(record)
        finally:
            record.msg = original_msg
            record.args = original_args

        # If there's a style_hint, it already applied styling, so return as-is
        if style_hint:
            return msg

        # For messages without style_hint, apply level-based colors
        level_color: str = self.LEVEL_COLORS.get(record.levelno, "")

        if record.levelno == logging.INFO:
            return self._format_info_message(msg)

        # Apply level color (ERROR=red, WARNING=yellow, etc.)
        if level_color:
            return Colors.colored(msg, level_color)

        return msg

    def _format_structured_message(
        self,
        style_hint: str,
        raw_message: str,
        record: logging.LogRecord,
    ) -> str:
        """Return a message string styled according to the provided hint."""
        handlers: dict[str, Callable[[str, logging.LogRecord], str]] = {
            LogStyles.RULE: self._style_rule,
            LogStyles.HEADER: self._style_header,
            LogStyles.SECTION: self._style_section,
            LogStyles.ERROR: self._style_error,
            LogStyles.SUCCESS: self._style_success,
            LogStyles.WARNING: self._style_warning,
            LogStyles.DETAIL: self._style_detail,
            LogStyles.METRIC_LABEL: self._style_metric_label,
            LogStyles.METRIC_VALUE: self._style_metric_value,
            LogStyles.GENERATED_TEXT: self._style_generated_text,
            LogStyles.FILE_PATH: self._style_file_path,
            LogStyles.MODEL_NAME: self._style_model_name,
        }

        handler = handlers.get(style_hint)
        if handler is None:
            return raw_message

        return handler(raw_message, record)

    def _style_rule(self, raw_message: str, record: logging.LogRecord) -> str:
        styles: list[str] = []
        if bool(getattr(record, "style_bold", False)):
            styles.append(Colors.BOLD)
        rule_color = getattr(record, "style_color", None)
        if isinstance(rule_color, str) and rule_color:
            styles.append(rule_color)
        return Colors.colored(raw_message, *styles) if styles else raw_message

    def _style_header(self, raw_message: str, record: logging.LogRecord) -> str:
        width = int(getattr(record, "style_width", max(len(raw_message), 1)))
        centered = raw_message.center(width)
        return Colors.colored(centered, Colors.BOLD, Colors.MAGENTA)

    def _style_section(self, raw_message: str, record: logging.LogRecord) -> str:
        uppercase_enabled = bool(getattr(record, "style_uppercase", True))
        has_ansi = "\x1b[" in raw_message
        uppercase = uppercase_enabled and not has_ansi
        safe_title = raw_message.upper() if uppercase else raw_message
        title_colored = Colors.colored(safe_title, Colors.BOLD, Colors.MAGENTA)
        prefix = getattr(record, "style_prefix", "▶")
        return f"{prefix} [ {title_colored} ]"

    def _style_error(self, raw_message: str, _record: logging.LogRecord) -> str:
        # Don't add "ERROR:" prefix since the log level already shows "ERROR"
        # Just apply error styling (red, bold) to the message
        return Colors.colored(raw_message, Colors.BOLD, Colors.RED)

    def _style_success(self, raw_message: str, _record: logging.LogRecord) -> str:
        return Colors.colored(raw_message, Colors.BOLD, Colors.GREEN)

    def _style_warning(self, raw_message: str, _record: logging.LogRecord) -> str:
        return Colors.colored(raw_message, Colors.BOLD, Colors.YELLOW)

    def _style_detail(self, raw_message: str, record: logging.LogRecord) -> str:
        detail_styles: list[str] = []
        if bool(getattr(record, "style_bold", False)):
            detail_styles.append(Colors.BOLD)
        detail_color = getattr(record, "style_color", Colors.CYAN)
        if isinstance(detail_color, str) and detail_color:
            detail_styles.append(detail_color)
        return Colors.colored(raw_message, *detail_styles) if detail_styles else raw_message

    def _style_metric_label(self, raw_message: str, record: logging.LogRecord) -> str:
        """Style metric category labels (e.g., 'Tokens:', 'Memory:')."""
        emoji = getattr(record, "style_emoji", "")
        label = f"{emoji} {raw_message}" if emoji else raw_message
        color = getattr(record, "style_color", Colors.WHITE)
        return Colors.colored(label, Colors.BOLD, color)

    def _style_metric_value(self, raw_message: str, record: logging.LogRecord) -> str:
        """Style metric values with optional color."""
        color = getattr(record, "style_color", Colors.WHITE)
        return Colors.colored(raw_message, color) if color else raw_message

    def _style_generated_text(self, raw_message: str, _record: logging.LogRecord) -> str:
        """Style generated model output in cyan."""
        return Colors.colored(raw_message, Colors.CYAN)

    def _style_file_path(self, raw_message: str, record: logging.LogRecord) -> str:
        """Style file paths with highlighting."""
        color = getattr(record, "style_color", Colors.CYAN)
        return Colors.colored(raw_message, color)

    def _style_model_name(self, raw_message: str, _record: logging.LogRecord) -> str:
        """Style model identifiers in magenta."""
        return Colors.colored(raw_message, Colors.MAGENTA)

    def _format_info_message(self, msg: str) -> str:
        """Apply context-aware formatting to INFO messages for better visual hierarchy."""
        stripped: str = msg.strip()

        # Define format patterns with their corresponding colors (priority ordered)
        format_patterns: list[tuple[Callable[[str, str], bool], tuple[str, ...]]] = [
            # Section separators (highest priority)
            (
                lambda s, _: (
                    s.startswith(("===", "---"))
                    or (
                        len(s) > FORMATTING.min_separator_chars
                        and s.count("=") > FORMATTING.min_separator_chars
                    )
                    or (
                        len(s) > FORMATTING.min_separator_chars
                        and s.count("-") > FORMATTING.min_separator_chars
                    )
                ),
                (Colors.BOLD, Colors.BLUE),
            ),
            # Section headers in brackets
            (lambda s, _: s.startswith("[ ") and s.endswith(" ]"), (Colors.BOLD, Colors.MAGENTA)),
            # Success indicators
            (lambda s, m: "SUCCESS:" in m or s.startswith("✓"), (Colors.BOLD, Colors.GREEN)),
            # Failure indicators
            (
                lambda s, m: any(x in m for x in ["FAILED:", "ERROR:"]) or s.startswith("✗"),
                (Colors.BOLD, Colors.RED),
            ),
            # Generated text highlighting
            (lambda _, m: "Generated Text:" in m, (Colors.CYAN,)),
            # Performance metrics
            (
                lambda _, m: any(
                    metric in m
                    for metric in [
                        "Tokens:",
                        "TPS:",
                        "Time:",
                        "Memory (",  # matches verbose memory lines
                        "Memory:",
                    ]
                ),
                (Colors.WHITE,),
            ),
            # File operations
            (
                lambda _, m: any(x in m for x in ["HTML report saved", "Markdown report saved"]),
                (Colors.BOLD, Colors.GREEN),
            ),
            # Processing status
            (lambda s, _: s.startswith("Processing"), (Colors.YELLOW,)),
            # Library versions section
            (
                lambda _, m: (
                    "Library Versions" in m
                    or (
                        m.count(":") == 1
                        and any(lib in m.lower() for lib in ["mlx", "pillow", "transformers"])
                    )
                ),
                (Colors.CYAN,),
            ),
        ]

        # Apply first matching pattern
        for pattern_check, colors in format_patterns:
            if pattern_check(stripped, msg):
                return Colors.colored(msg, *colors)

        # Default INFO messages without color to reduce noise
        return msg


class FileSafeFormatter(logging.Formatter):
    """Formatter for file logs that strips ANSI/control formatting codes."""

    def format(self, record: logging.LogRecord) -> str:
        raw = super().format(record)
        return ANSI_ESCAPE_RE.sub("", raw)


# Configure logging to use ColoredFormatter
handler: logging.StreamHandler[Any] = logging.StreamHandler(sys.stderr)
formatter: ColoredFormatter = ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
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
ERROR_MESSAGE_PREVIEW_LEN: Final[int] = 40  # Max chars to show from error in summary line
ERROR_MESSAGE_TRUNCATE_LEN: Final[int] = 120  # Max chars for error messages in actionable reports
MAX_QUALITY_ISSUES_LEN: Final[int] = 30  # Max chars for quality issues in Markdown tables
MAX_OUTPUT_LINES: Final[int] = 3  # Max lines to show in summary table cells
MAX_OUTPUT_PREVIEW_CHARS: Final[int] = 280  # Max chars for output previews in summary tables
MAX_CAPTURED_OUTPUT_LOG_CHARS: Final[int] = 1200  # Max chars of captured stdout/stderr in logs
MAX_TRIAGE_MODELS: Final[int] = 5  # Max model rows shown in triage subsections
SUMMARY_CHART_WIDTH: Final[int] = 24  # Character width for compact ASCII summary bars
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


@runtime_checkable
class EscapeStrategy(Protocol):
    """Protocol for text escaping strategies.

    Enables unified handling of different output format escaping needs
    (HTML, Markdown) with consistent interface.
    """

    def escape(self, text: str) -> str:
        """Escape text according to strategy rules."""
        ...


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
    """Format overall runtime with adaptive precision based on duration.

    For durations < 3600s return ``"{seconds:.2f}s"``.
    For durations >= 3600s return ``"HH:MM:SS ({seconds:.2f}s)"``.

    Args:
        total_seconds: Total elapsed time in seconds

    Returns:
        Formatted runtime string with seconds or HH:MM:SS format

    Examples:
        >>> # Short durations show seconds only
        >>> format_overall_runtime(42.5)
        '42.50s'

        >>> # Medium durations still use seconds
        >>> format_overall_runtime(1234.56)
        '1234.56s'

        >>> # Long durations show HH:MM:SS plus seconds
        >>> format_overall_runtime(7384.25)
        '02:03:04 (7384.25s)'

        >>> # Very long durations
        >>> format_overall_runtime(36125.78)
        '10:02:05 (36125.78s)'

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

    # Check n-grams of length 4 to 10
    for n in range(min_phrase_len, min(11, n_words // 2)):
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
    r"""Detect patterns that suggest model hallucination or non-stopping behavior.

    Looks for:
    - Markdown tables in non-table contexts (like captions)
    - Questions appearing in generated descriptions
    - Multiple choice answer patterns (A), B), C), D))
    - Unrelated mathematical or quiz content

    Args:
        text: Generated text to check

    Returns:
        List of detected issue descriptions (empty if clean)

    Examples:
        >>> _detect_hallucination_patterns("Caption: Nice photo\n\n| Grade | Count |")
        ['Contains unexpected table']

        >>> _detect_hallucination_patterns("A) 42\nB) 43\nC) 44")
        ['Contains multiple choice pattern']
    """
    issues: list[str] = []

    if not text:
        return issues

    text_lower = text.lower()

    # Check for markdown tables (pipe-delimited)
    if "|" in text and text.count("|") >= QUALITY.min_pipes_for_table:
        # Likely a table if we see multiple pipes
        lines_with_pipes = [line for line in text.split("\n") if "|" in line]
        if len(lines_with_pipes) >= QUALITY.min_table_rows:
            issues.append("Contains unexpected table")

    # Check for multiple choice patterns
    mc_pattern = re.compile(r"^[A-D]\)", re.MULTILINE)
    mc_matches = mc_pattern.findall(text)
    if len(mc_matches) >= QUALITY.min_mc_answers:
        issues.append("Contains multiple choice pattern")

    # Check for quiz/test questions
    question_indicators = (
        QUALITY.patterns.get("hallucination_question_indicators", [])
        if QUALITY.patterns
        else ["what is", "how many", "based on the chart", "calculate"]
    )
    has_question = any(indicator in text_lower for indicator in question_indicators)
    if has_question and len(text) > QUALITY.substantial_text_length:
        issues.append("Contains question/quiz content")

    # Check for unrelated educational content keywords
    edu_keywords = (
        QUALITY.patterns.get("hallucination_edu_keywords", [])
        if QUALITY.patterns
        else ["grade level", "students with adhd", "test scores", "homework"]
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

    text_lower = text.lower()

    # Check for meta-commentary patterns
    meta_patterns = (
        QUALITY.patterns.get("meta_commentary", [])
        if QUALITY.patterns
        else [
            "the image depicts",
            "the image shows",
            "the photograph captures",
            "this image features",
            "in conclusion",
            "### analysis",
            "### conclusion",
            "based on the image",
        ]
    )

    meta_count = sum(1 for pattern in meta_patterns if pattern in text_lower)

    # Check for excessive sectioning
    section_headers = text.count("###") + text.count("## ")

    # Verbose if has meta-commentary + sections or just too many sections
    return meta_count >= QUALITY.min_meta_patterns or section_headers >= QUALITY.min_section_headers


def _detect_formatting_violations(text: str) -> list[str]:
    """Detect formatting issues in generated output.

    Looks for:
    - Unknown/unexpected tags (not simple <br>) that may interfere with output rendering
    - Excessive markdown headers/structure

    Note: Bullet lists are checked separately by _detect_excessive_bullets()
    since they may be appropriate depending on the prompt.

    Args:
        text: Generated text to check

    Returns:
        List of detected formatting issues (excluding bullets)
    """
    issues: list[str] = []

    if not text:
        return issues

    # Check for tags (beyond simple breaks) that may interfere with rendering
    html_tags = re.findall(r"<(?!br>|/br>)[a-z]+[^>]*>", text, re.IGNORECASE)
    if html_tags:
        # Report the raw tags (escaping handled by reporters)
        tags_preview = ", ".join(set(html_tags[:3]))
        issues.append(f"Unknown tags: {tags_preview}")

    # Check for excessive markdown structure
    header_count = text.count("\n##") + text.count("\n###")
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
    pattern = re.escape(repeated_token)
    match = re.search(rf"({pattern}(?:\s*{pattern}){{10,}})", text)

    if match:
        # Count total repetitions in the matched section
        repetitions = match.group(0).count(repeated_token)
        # Show first few occurrences + count + ellipsis
        truncated_section = (
            f"{repeated_token} {repeated_token} {repeated_token} "
            f"... [{repetitions} total repetitions] ..."
        )
        return text.replace(match.group(0), truncated_section)

    return text


def _detect_excessive_bullets(text: str) -> tuple[bool, int]:
    """Detect if output contains excessive bullet points.

    Bullet lists may be appropriate depending on the prompt (e.g., if the
    prompt asks "list the items in this image"), so this is separated from
    other formatting issues.

    Args:
        text: Generated text to check

    Returns:
        Tuple of (has_excessive_bullets, bullet_count)
    """
    if not text:
        return False, 0

    bullet_prefixes = ("- ", "* ", "• ")
    bullet_lines = [line for line in text.split("\n") if line.strip().startswith(bullet_prefixes)]
    bullet_count = len(bullet_lines)

    # Use config threshold if available, otherwise default to 15 (lowered for cataloging)
    threshold = QUALITY.max_bullets or 15
    return bullet_count > threshold, bullet_count


def _detect_context_ignorance(
    text: str,
    prompt: str,
    context_marker: str = "Context:",
) -> tuple[bool, list[str]]:
    """Detect if the generated text ignores key context from the prompt.

    Extracts proper nouns and key contextual terms from the prompt (e.g., from
    "Context:" sections) and checks if they appear in the generated text.

    Args:
        text: Generated text to check
        prompt: Original prompt text containing context
        context_marker: The marker used to identify the context section (default: "Context:")

    Returns:
        Tuple of (is_context_ignored, missing_context_terms)
    """
    if not text or not prompt:
        return False, []

    # Extract context section if present
    # Escape the marker for regex safety
    marker_pattern = re.escape(context_marker)
    context_match = re.search(
        rf"{marker_pattern}\s*(.+?)(?:\n\n|\Z)",
        prompt,
        re.DOTALL | re.IGNORECASE,
    )
    if not context_match:
        # No explicit context section, so can't check
        return False, []

    context_text = context_match.group(1)

    # Extract potential proper nouns and key terms from context
    # Look for capitalized words that aren't common words
    common_words = {
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
        "context",
        "image",
        "photo",
        "picture",
    }

    # Find capitalized words (potential proper nouns)
    potential_terms = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", context_text)

    # Filter out common words and keep unique terms
    key_terms = [
        term
        for term in set(potential_terms)
        if term.lower() not in common_words and len(term) > QUALITY.min_context_term_length
    ]

    # Check if these terms appear in the generated text (case-insensitive)
    missing_terms = [term for term in key_terms if term.lower() not in text.lower()]

    # Only flag as "ignored" if we found key terms and most are missing
    # Use thresholds from configuration
    is_ignored = (
        len(missing_terms) > 0
        and len(key_terms) >= QUALITY.min_key_terms_threshold
        and len(missing_terms) >= len(key_terms) * QUALITY.min_missing_ratio
    )

    return is_ignored, missing_terms


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

    text_lower = text.lower()

    # Refusal patterns
    refusal_patterns = []

    if QUALITY.patterns:
        if "refusal_explicit" in QUALITY.patterns:
            refusal_patterns.append(("explicit_refusal", QUALITY.patterns["refusal_explicit"]))
        if "refusal_uncertainty" in QUALITY.patterns:
            refusal_patterns.append(("uncertainty", QUALITY.patterns["refusal_uncertainty"]))
        if "refusal_insufficient_info" in QUALITY.patterns:
            refusal_patterns.append(
                ("insufficient_info", QUALITY.patterns["refusal_insufficient_info"]),
            )

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

    text_lower = text.lower()
    word_count = len(text.split())

    if word_count == 0:
        return False, 0.0

    # Count filler/hedge words
    filler_words = (
        QUALITY.patterns.get("filler_words", [])
        if QUALITY.patterns
        else [
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
        ]
    )
    filler_count = sum(text_lower.count(filler) for filler in filler_words)

    # Calculate filler ratio
    filler_ratio = filler_count / word_count

    # Check for specific details (numbers, measurements, colors, names)
    has_numbers = bool(re.search(r"\d+", text))
    has_specific_colors = bool(
        re.search(
            r"\b(red|blue|green|yellow|orange|purple|pink|brown|black|white|gray|grey)\b",
            text_lower,
        ),
    )
    has_proper_nouns = bool(re.search(r"\b[A-Z][a-z]+", text))

    specificity_indicators = sum([has_numbers, has_specific_colors, has_proper_nouns])

    # Generic if high filler ratio and low specificity
    is_generic = (
        filler_ratio > QUALITY.generic_filler_threshold
        and specificity_indicators < QUALITY.min_specificity_indicators
    )

    # Specificity score: higher = more specific (0-100)
    specificity_score = max(0.0, 100 - (filler_ratio * 200) + (specificity_indicators * 20))

    return is_generic, round(specificity_score, 1)


def _detect_language_mixing(
    text: str,
    quality_thresholds: QualityThresholds = QUALITY,
) -> tuple[bool, list[str]]:
    """Detect unexpected language switches or code/tokenizer artifacts.

    Catches technical artifacts that shouldn't appear in natural language output.

    Args:
        text: Generated text to check
        quality_thresholds: Configuration object containing patterns and thresholds

    Returns:
        Tuple of (has_mixing, list of detected issues)
    """
    if not text:
        return False, []

    issues: list[str] = []

    # Check for common tokenizer artifacts
    if quality_thresholds.patterns:
        tokenizer_artifacts = quality_thresholds.patterns.get("tokenizer_artifacts", [])
    else:
        tokenizer_artifacts = [
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
        ]

    for artifact in tokenizer_artifacts:
        if re.search(artifact, text, re.IGNORECASE):
            issues.append("tokenizer_artifact")
            break

    # Check for code snippets (function calls, variable assignments)
    if quality_thresholds.patterns:
        code_patterns = quality_thresholds.patterns.get("code_patterns", [])
    else:
        code_patterns = [
            r"\bdef\s+\w+\(",  # Python function def
            r"\bfunction\s+\w+\(",  # JavaScript function
            r"\bclass\s+\w+",  # Class definition
            r"\bimport\s+\w+",  # Import statement
            r"\breturn\s+",  # Return statement
        ]

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
    tail_length = min(200, len(text) // 3)
    tail = text[-tail_length:]
    result: str | None = None

    # 1. Detect repeated punctuation/special char sequences at end
    # e.g., "......" or "?????" or "!!!!!" or "-----"
    punct_repeat = re.search(r"([.?!,;:\-_=+*#]{3,})\s*$", tail)
    if punct_repeat:
        result = f"repeated_punctuation: '{punct_repeat.group(1)[:10]}...'"

    # 2. Detect incomplete sentence (ends mid-word or with lowercase without punctuation)
    if result is None:
        stripped = text.rstrip()
        if stripped:
            last_char = stripped[-1]
            # Normal endings: . ! ? ) " ' ] }
            normal_endings = ".!?)]}'\"}"
            if last_char not in normal_endings:
                last_word_match = re.search(r"\b(\w+)$", stripped)
                if last_word_match:
                    last_word = last_word_match.group(1)
                    if len(last_word) <= QUALITY.min_cutoff_word_length and last_word.islower():
                        result = f"incomplete_sentence: ends with '{last_word}'"

    # 3. Detect Unicode rubbish/control characters (excluding normal whitespace)
    if result is None:
        control_chars = re.findall(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", tail)
        if len(control_chars) > QUALITY.max_control_chars:
            result = f"control_characters: {len(control_chars)} found"

    # 4. Detect repeated newline patterns (degenerate spacing)
    if result is None and "\n\n\n\n\n\n" in tail:
        result = "excessive_newlines"

    # 5. Detect character-level repetition at the end
    if result is None:
        char_repeat = re.search(r"(.{1,3})\1{5,}\s*$", tail)
        if char_repeat:
            pattern = char_repeat.group(1)
            result = f"character_loop: '{pattern}' repeated"

    # 6. Detect sudden encoding shift
    if result is None and len(text) > tail_length * 2:
        head = text[:-tail_length]
        ascii_max = 127  # Standard ASCII range
        head_non_ascii = len([c for c in head if ord(c) > ascii_max]) / max(len(head), 1)
        tail_non_ascii = len([c for c in tail if ord(c) > ascii_max]) / max(len(tail), 1)
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
    url_patterns = _get_quality_pattern_list(
        "fabrication_url_patterns",
        [r"https?://[^\s<>\"']+"],
    )
    urls = _extract_pattern_matches(
        text,
        url_patterns,
        debug_context="fabrication URL",
        unique=True,
    )

    suspicious_url_keywords = _get_quality_pattern_list(
        "fabrication_suspicious_url_keywords",
        ["example.com", "placeholder", "xxx", "fake"],
    )
    long_url_path_patterns = _get_quality_pattern_list(
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
    precise_stat_patterns = _get_quality_pattern_list(
        "fabrication_precise_stat_patterns",
        [r"\b(\d{1,3}(?:,\d{3})*\.\d{3,})\s*%?"],
    )
    precise_stats = _extract_pattern_matches(
        text,
        precise_stat_patterns,
        debug_context="fabrication precise stat",
    )
    if len(precise_stats) >= QUALITY.min_precise_stats:
        issues.append(f"suspicious_precision: {len(precise_stats)} overly precise numbers")

    # 3. Detect future dates (model can't know the future)
    # Years 2030+ are definitely future
    future_year_patterns = _get_quality_pattern_list(
        "fabrication_future_year_patterns",
        [r"\b(20[3-9]\d|2[1-9]\d{2})\b"],
    )
    future_years = _extract_pattern_matches(
        text,
        future_year_patterns,
        debug_context="fabrication future year",
        unique=True,
    )
    if future_years:
        issues.append(f"future_date: {', '.join(future_years[:3])}")

    # 4. Detect citations to non-existent sources (common hallucination)
    # Patterns like "according to Smith et al. (2024)" or "(Johnson, 2025)"
    citation_patterns = _get_quality_pattern_list(
        "fabrication_citation_patterns",
        [r"\(([A-Z][a-z]+(?:\s+et\s+al\.?)?,?\s*\d{4})\)"],
    )
    fake_citations = _extract_pattern_matches(
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
        count = text.count("\u0120")
        return True, f"bpe_space_leak({count})"

    # Check for Ċ (U+010A) - BPE newline marker leak
    if "\u010a" in text:
        count = text.count("\u010a")
        return True, f"bpe_newline_leak({count})"

    # Check for other common tokenizer artifacts that shouldn't be visible
    # These are byte-level BPE artifacts
    bpe_artifacts = [
        ("\u0100", "byte_0"),  # Byte fallback prefix
        ("\u0101", "byte_1"),
        ("\u0102", "byte_2"),
    ]
    for artifact, name in bpe_artifacts:
        if artifact in text and text.count(artifact) > QUALITY.min_bpe_artifact_count:
            return True, f"bpe_byte_leak({name})"

    return False, None


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

    # Common special tokens that should never appear in output
    special_token_patterns = [
        # End tokens
        (r"<\|end\|>", "<|end|>"),
        (r"<\|endoftext\|>", "<|endoftext|>"),
        (r"<\|eot_id\|>", "<|eot_id|>"),
        (r"<\|im_end\|>", "<|im_end|>"),
        (r"<\|assistant\|>", "<|assistant|>"),
        (r"<end_of_turn>", "<end_of_turn>"),
        (r"</s>(?!\w)", "</s>"),  # Not followed by word char (avoid </span>)
        (r"<s>(?!\w)", "<s>"),
        # Instruction markers
        (r"# INSTRUCTION", "# INSTRUCTION"),
        (r"# SOLUTION", "# SOLUTION"),
        (r"\[INST\]", "[INST]"),
        (r"\[/INST\]", "[/INST]"),
        # Thinking markers (some models expose these)
        (r"<\|think\|>", "<|think|>"),
        (r"</think>", "</think>"),
        # Other control tokens
        (r"<\|pad\|>", "<|pad|>"),
        (r"<\|unk\|>", "<|unk|>"),
        (r"\[PAD\]", "[PAD]"),
        (r"\[CLS\]", "[CLS]"),
        (r"\[SEP\]", "[SEP]"),
    ]

    for pattern, token_name in special_token_patterns:
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
        text_stripped = text.strip()
        word_count = len(text_stripped.split())

        # Single sentence filler responses
        filler_responses = [
            "the image is a photograph",
            "the image is in the public domain",
            "i cannot",
            "i can't",
            "this image shows",
        ]
        text_lower = text_stripped.lower()
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
        ratio = generated_tokens / prompt_tokens
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

    safe_prompt_tokens = max(prompt_tokens, 1)
    ratio = generated_tokens / safe_prompt_tokens
    text_empty = not text.strip()

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
    training_leak_pattern_groups = [
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
    check_portion = text[len(text) // 3 :]

    for patterns, leak_type in training_leak_pattern_groups:
        for pattern in patterns:
            try:
                if re.search(pattern, check_portion, re.DOTALL):
                    return True, leak_type
            except re.error:
                logger.debug("Ignoring invalid training leak regex: %s", pattern)

    return False, None


def compute_vocabulary_diversity(text: str) -> tuple[float, int, int]:
    """Compute vocabulary diversity metrics for generated text.

    Type-token ratio (TTR) measures lexical diversity - the ratio of unique
    words to total words. Higher values indicate more varied vocabulary.

    Args:
        text: Generated text to analyze

    Returns:
        Tuple of (type_token_ratio, unique_words, total_words)
        Returns (0.0, 0, 0) for empty text

    Examples:
        >>> compute_vocabulary_diversity("The cat sat on the mat")
        (0.83, 5, 6)  # 5 unique words out of 6 total

        >>> compute_vocabulary_diversity("yes yes yes yes")
        (0.25, 1, 4)  # Low diversity - repetitive
    """
    if not text:
        return 0.0, 0, 0

    # Normalize: lowercase, extract word tokens only
    words = re.findall(r"\b[a-z]+\b", text.lower())
    total_words = len(words)

    if total_words == 0:
        return 0.0, 0, 0

    unique_words = len(set(words))
    ttr = unique_words / total_words

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
            tps = tokens_generated / generation_time
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
    compliance = compute_task_compliance(text)

    # Add section detection (markdown headers)
    has_sections = bool(re.search(r"^#{1,3}\s+\w+", text, re.MULTILINE))

    return {
        "has_caption": bool(compliance["has_caption"]),
        "has_keywords": bool(compliance["has_keywords"]),
        "has_description": bool(compliance["has_description"]),
        "has_sections": has_sections,
    }


def _get_quality_pattern_list(pattern_key: str, fallback: list[str]) -> list[str]:
    """Return configured regex/pattern list for a key, falling back to defaults."""
    if not QUALITY.patterns:
        return fallback

    configured = QUALITY.patterns.get(pattern_key)
    if not configured:
        return fallback

    # Guard against malformed YAML values while keeping detector behavior stable.
    valid = [p for p in configured if isinstance(p, str)]
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
            value = match.group(0)
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
    total = 0
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

    text_lower = text.lower()

    hedge_patterns = _get_quality_pattern_list(
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
    definitive_patterns = _get_quality_pattern_list(
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

    hedge_count = _count_pattern_matches(text_lower, hedge_patterns)
    definitive_count = _count_pattern_matches(text_lower, definitive_patterns)

    total = hedge_count + definitive_count
    confidence_ratio = definitive_count / total if total > 0 else 0.5

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
    echoed_words = output_words & context_words
    # Words in output that are novel (not in context)
    novel_words = output_words - context_words

    echo_ratio = len(echoed_words) / len(output_words) if output_words else 0.0
    information_gain = len(novel_words) / len(output_words) if output_words else 0.0

    return {
        "context_words": len(context_words),
        "output_words": len(output_words),
        "novel_words": len(novel_words),
        "echo_ratio": round(echo_ratio, 2),
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

    # Score: 1/3 for each component
    score = (
        (0.33 if has_caption else 0.0)
        + (0.34 if has_description else 0.0)
        + (0.33 if has_keywords else 0.0)
    )

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


def compute_cataloging_utility(
    text: str,
    context: str | None,
    *,
    info_gain: dict[str, float | int] | None = None,
    task_compliance: dict[str, bool | float] | None = None,
    visual_grounding: dict[str, float | int] | None = None,
) -> dict[str, float | str]:
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
    echo_ratio = float(info_gain.get("echo_ratio", 0.0))
    compliance_score = float(task_compliance.get("compliance_score", 0.0))
    grounding_score = float(visual_grounding.get("grounding_score", 0.0))

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
        "capture metadata:",
        "capture metadata hints:",
    )
    extracted_lines: list[str] = []
    for raw_line in context.splitlines():
        line = raw_line.strip().lstrip("-").strip()
        if not line:
            continue
        if line.lower().startswith("existing metadata hints"):
            continue
        line_lower = line.lower()
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


def _compute_utility_snapshot(
    text: str,
    context: str | None,
    *,
    baseline_score: float | None = None,
) -> tuple[float, str, str, float | None]:
    """Compute utility score/grade/weakness and optional delta vs metadata baseline."""
    utility = compute_cataloging_utility(text, context)
    score = float(utility["utility_score"])
    grade = str(utility["utility_grade"])
    weakness = str(utility["primary_weakness"])
    delta = score - baseline_score if baseline_score is not None else None
    return score, grade, weakness, delta


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
    # Harness issues indicate mlx-vlm integration bugs, not model quality problems
    has_harness_issue: bool
    harness_issue_type: str | None
    harness_issue_details: list[str]
    # Lightweight metrics useful for JSONL/report triage
    word_count: int = 0
    unique_ratio: float = 0.0

    def has_any_issues(self) -> bool:
        """Return True if any quality issues were detected."""
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
            or self.has_harness_issue
        )

    def has_harness_issues_only(self) -> bool:
        """Return True if only harness issues detected (no model quality issues).

        Useful for filtering reports - if only harness issues, the model may
        work fine with a different integration approach.
        """
        if not self.has_harness_issue:
            return False
        # Check if there are any non-harness issues
        return not (
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
        )

    @property
    def issues(self) -> list[str]:
        """Return a list of all detected quality issues as human-readable strings."""
        issues_list = []
        if self.is_repetitive:
            issues_list.append(f"Repetitive output ({self.repeated_token})")
        issues_list.extend(self.hallucination_issues)
        if self.is_verbose:
            issues_list.append("Excessive verbosity")
        issues_list.extend(self.formatting_issues)
        if self.has_excessive_bullets:
            issues_list.append(f"Excessive bullet points ({self.bullet_count})")
        if self.is_context_ignored:
            issues_list.append(
                f"Context ignored (missing: {', '.join(self.missing_context_terms)})",
            )
        if self.is_refusal:
            issues_list.append(f"Refusal detected ({self.refusal_type})")
        if self.is_generic:
            issues_list.append(f"Generic output (specificity: {self.specificity_score:.2f})")
        if self.has_language_mixing:
            issues_list.extend(self.language_mixing_issues)
        if self.has_degeneration:
            issues_list.append(f"Output degeneration ({self.degeneration_type})")
        if self.has_fabrication:
            issues_list.extend(self.fabrication_issues)
        # Harness issues (prominently marked as integration problems)
        if self.has_harness_issue:
            harness_label = f"⚠️HARNESS:{self.harness_issue_type}"
            issues_list.insert(0, harness_label)  # Put at front for visibility
            issues_list.extend(self.harness_issue_details)
        return issues_list


def analyze_generation_text(
    text: str,
    generated_tokens: int,
    prompt_tokens: int | None = None,
    prompt: str | None = None,
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
        context_marker: Marker for context section in prompt

    Returns:
        GenerationQualityAnalysis with all detected issues
    """
    is_repetitive, repeated_token = _detect_repetitive_output(text)
    hallucination_issues = _detect_hallucination_patterns(text)
    is_verbose = _detect_excessive_verbosity(text, generated_tokens)
    formatting_issues = _detect_formatting_violations(text)
    has_excessive_bullets, bullet_count = _detect_excessive_bullets(text)

    # Context ignorance: output doesn't reference key terms from prompt
    is_context_ignored = False
    missing_context_terms: list[str] = []
    if prompt:
        is_context_ignored, missing_context_terms = _detect_context_ignorance(
            text,
            prompt,
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

    # Harness/integration issues: mlx-vlm bugs, not model quality problems
    harness_issues: list[str] = []
    harness_type: str | None = None

    # Check for token encoding issues (BPE leak)
    has_encoding_issue, encoding_type = _detect_token_encoding_issues(text)
    if has_encoding_issue and encoding_type:
        harness_type = harness_type or "encoding"
        harness_issues.append(f"token_encoding:{encoding_type}")

    # Check for special token leakage
    has_token_leak, leaked_tokens = _detect_special_token_leakage(text)
    if has_token_leak:
        harness_type = harness_type or "stop_token"
        harness_issues.extend([f"token_leak:{tok}" for tok in leaked_tokens[:3]])

    # Check for minimal/zero output (prompt template issue)
    has_minimal, minimal_type = _detect_minimal_output(
        text,
        generated_tokens,
        prompt_tokens=prompt_tokens,
    )
    if has_minimal and minimal_type:
        harness_type = harness_type or "prompt_template"
        harness_issues.append(f"output:{minimal_type}")

    # Check for long-context degradation (high prompt token count with weak output)
    has_long_context_breakdown, long_context_issue = _detect_long_context_breakdown(
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        text=text,
        is_repetitive=is_repetitive,
        is_context_ignored=is_context_ignored,
        is_refusal=is_refusal,
    )
    if has_long_context_breakdown and long_context_issue:
        # Prefer explicit long-context classification over generic prompt-template label.
        if harness_type in {None, "prompt_template"}:
            harness_type = "long_context"
        harness_issues.append(long_context_issue)

    # Check for training data leakage
    has_training_leak, leak_type = _detect_training_data_leak(text)
    if has_training_leak and leak_type:
        harness_type = harness_type or "generation_loop"
        harness_issues.append(f"training_leak:{leak_type}")

    _ttr, unique_words, total_words = compute_vocabulary_diversity(text)
    unique_ratio = unique_words / total_words if total_words else 0.0

    has_harness_issue = bool(harness_issues)

    return GenerationQualityAnalysis(
        is_repetitive=is_repetitive,
        repeated_token=repeated_token,
        hallucination_issues=hallucination_issues,
        is_verbose=is_verbose,
        formatting_issues=formatting_issues,
        has_excessive_bullets=has_excessive_bullets,
        bullet_count=bullet_count,
        is_context_ignored=is_context_ignored,
        missing_context_terms=missing_context_terms,
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
        has_harness_issue=has_harness_issue,
        harness_issue_type=harness_type,
        harness_issue_details=harness_issues,
        word_count=total_words,
        unique_ratio=round(unique_ratio, 3),
    )


def local_now_str(fmt: str = "%Y-%m-%d %H:%M:%S %Z") -> str:
    """Return localized current time as a formatted string.

    Centralizes timestamp formatting so report generators and version info
    stay consistent and makes future changes (e.g. adding UTC or ISO8601
    variants) trivial.
    """
    return datetime.now(get_localzone()).strftime(fmt)


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


def _format_numeric_by_field(field_name: str, num: float) -> str:
    """Format a numeric value based on field name conventions.

    Dispatches to appropriate formatter based on field suffix or exact name.
    """
    # Check suffix-based patterns first
    if field_name.endswith("_memory"):
        return _format_memory_value_gb(num)
    if field_name.endswith("_tps"):
        return _format_tps(num)

    # Check exact field name matches via lookup
    if field_name in _TIME_FIELDS:
        return _format_time_seconds(num)
    if field_name == "quality_score":
        return f"{num:.1f}"
    if field_name in _BOOLEAN_FLAG_FIELDS:
        return "✓" if num else "-"

    # Default numeric formatting
    return fmt_num(num)


def format_field_value(field_name: str, value: MetricValue) -> str:
    """Normalize and format field values for display.

    Rules:
        - Memory fields ("*_memory"): mixed sources (mlx returns bytes; mlx-vlm returns
            decimal GB). Heuristic: if raw value > MEM_BYTES_TO_GB_THRESHOLD treat as bytes,
            else assume already GB. Formatting thresholds:
                >= 10 GB : integer with commas
                >= 1 GB  : one decimal place
                < 1 GB   : two decimals
    - Time fields: seconds with 2 decimals + trailing 's'.
    - TPS fields: adaptive precision (integer / 1 decimal / 3 sig figs).
    - Other numerics: general fmt_num; non-numerics: str(value) or ''.

    Args:
        field_name: Name of the metric field (used for format detection)
        value: Numeric or string value to format

    Returns:
        Formatted string representation of the value

    Examples:
        >>> # Memory formatting (bytes to GB conversion)
        >>> format_field_value("peak_memory", 15_728_640_000.0)
        '15 GB'

        >>> # Time formatting (seconds with 2 decimals)
        >>> format_field_value("generation_time", 3.14159)
        '3.14s'

        >>> # TPS formatting (adaptive precision)
        >>> format_field_value("generation_tps", 42.567)
        '42.6'

        >>> # Non-numeric values returned as strings
        >>> format_field_value("model_name", "qwen2-vl-2b-instruct-4bit")
        'qwen2-vl-2b-instruct-4bit'

        >>> # None values return empty string
        >>> format_field_value("any_field", None)
        ''

    """
    if value is None:
        return ""

    # Handle numeric types directly
    if isinstance(value, int | float):
        return _format_numeric_by_field(field_name, float(value))

    # Handle string values - try to parse as numeric
    if isinstance(value, str) and value:
        s: str = value.strip().replace(",", "")
        try:
            f = float(s)
        except ValueError:
            return value  # Return non-numeric strings as-is
        return _format_numeric_by_field(field_name, f)

    return str(value)


def is_numeric_value(val: object) -> bool:
    """Return True if val can be interpreted as a number."""
    if isinstance(val, int | float):
        return True
    if isinstance(val, str):
        s = val.strip().replace(",", "")
        try:
            float(s)
        except ValueError:
            return False
        else:
            return True
    return False


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
        return max(min_width, int(WIDTH_OVERRIDE))
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

    - Honors --no-color / --force-color to toggle ANSI colors
    - Applies --width via MLX_VLM_WIDTH env var for child processes too
    """
    # Color controls
    if getattr(args, "no_color", False):
        Colors.set_enabled(enabled=False)
    elif getattr(args, "force_color", False):
        Colors.set_enabled(enabled=True)

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
    char: str = "─",  # Unicode box-drawing character (was "-")
    color: str | None = None,
    bold: bool = False,
    level: int = logging.INFO,
    pre_newline: bool = False,
    post_newline: bool = False,
) -> None:
    """Log a horizontal rule line with optional color and bold.

    Uses unicode box-drawing characters for better visual separation.
    Keeps a single place for styling separators to ensure consistency.
    """
    if pre_newline:
        logger.log(level, "")

    line = char * max(1, width)
    extra: dict[str, object] = {"style_hint": LogStyles.RULE}
    if color:
        extra["style_color"] = color
    if bold:
        extra["style_bold"] = True
    logger.log(level, line, extra=extra)

    if post_newline:
        logger.log(level, "")


# --- Utility Functions ---
# Ensure _pad_text is defined only once at module level and used everywhere


def get_library_versions() -> LibraryVersionDict:
    """Return versions of key libraries as a dictionary, using None for missing."""

    def _get_version(pkg_name: str, fallback: str | None = None) -> str | None:
        try:
            return importlib.metadata.version(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            return fallback

    def _none_if_na(v: str | None) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
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
    """Convert a version string to comparable numeric components.

    Keeps only numeric segments so strings like ``0.30.7.dev20260214`` compare
    sensibly against floor versions such as ``0.30.5``.
    """
    numbers = [int(part) for part in re.findall(r"\d+", version_text)]
    if len(numbers) < width:
        numbers.extend([0] * (width - len(numbers)))
    return tuple(numbers[:width])


def _is_version_at_least(installed: str, minimum: str) -> bool:
    """Return whether ``installed`` satisfies ``minimum`` using numeric comparison."""
    return _version_components(installed) >= _version_components(minimum)


def _collect_upstream_requirements(
    versions: LibraryVersionDict,
) -> dict[str, tuple[str, set[str]]]:
    """Collect package floor versions implied by installed upstream stacks."""
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

    if versions.get("mlx-vlm"):
        _record_requirement("mlx", "0.30.0", "mlx-vlm")
        _record_requirement("mlx-lm", "0.30.5", "mlx-vlm")
        _record_requirement("transformers", "5.1.0", "mlx-vlm")

    if versions.get("mlx-lm"):
        _record_requirement("mlx", "0.30.4", "mlx-lm")
        _record_requirement("transformers", "5.0.0", "mlx-lm")

    return requirements


def _detect_upstream_version_issues(versions: LibraryVersionDict) -> list[str]:
    """Return compatibility issues against current upstream package minimums."""
    issues: list[str] = []
    requirements = _collect_upstream_requirements(versions)

    for package, (minimum, sources) in sorted(requirements.items()):
        source_label = ", ".join(sorted(sources))
        installed = versions.get(package)
        if installed is None:
            issues.append(
                f"{package} is missing; upstream {source_label} expects {package}>={minimum}.",
            )
            continue

        if not _is_version_at_least(installed, minimum):
            issues.append(
                f"{package}=={installed} is below upstream minimum {minimum} "
                f"required by {source_label}.",
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
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None

    direct_candidate = Path(str(distribution.locate_file(relative_path)))
    if direct_candidate.is_file():
        return direct_candidate

    normalized_target = relative_path.replace("\\", "/")
    for file_ref in distribution.files or []:
        file_path = str(file_ref).replace("\\", "/")
        if not file_path.endswith(normalized_target):
            continue

        candidate = Path(str(distribution.locate_file(file_ref)))
        if candidate.is_file():
            return candidate

    module_name = normalized_target.split("/", 1)[0]
    module_spec = importlib_util.find_spec(module_name)
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


def _has_transformers_backend_guard_names(import_utils_source: str) -> bool:
    """Return whether transformers source still references TRANSFORMERS_NO_* vars."""
    guard_names = ("TRANSFORMERS_NO_TF", "TRANSFORMERS_NO_FLAX", "TRANSFORMERS_NO_JAX")
    return any(name in import_utils_source for name in guard_names)


def _detect_transformers_env_guard_issue() -> str | None:
    """Detect whether transformers still honors TRANSFORMERS_NO_* guard vars."""
    if not _transformers_guard_enabled:
        return None

    import_utils_spec = importlib_util.find_spec("transformers.utils.import_utils")
    import_utils_origin = getattr(import_utils_spec, "origin", None) if import_utils_spec else None
    if not import_utils_origin:
        return None

    try:
        import_utils_source = Path(import_utils_origin).read_text(encoding="utf-8")
    except OSError:
        return None

    if _has_transformers_backend_guard_names(import_utils_source):
        return None

    return (
        "transformers import utils no longer reference TRANSFORMERS_NO_TF/FLAX/JAX; "
        "check_models backend guard env vars may be ignored with this version."
    )


def _collect_preflight_package_issues(versions: LibraryVersionDict) -> list[str]:
    """Collect actionable dependency/runtime issues before model execution."""
    issues = _detect_upstream_version_issues(versions)

    load_image_issue = _detect_mlx_vlm_load_image_issue()
    if load_image_issue:
        issues.append(load_image_issue)

    guard_issue = _detect_transformers_env_guard_issue()
    if guard_issue:
        issues.append(guard_issue)

    return issues


def _get_available_fields(results: list[PerformanceResult]) -> list[str]:
    """Return ordered list of metric field names present across results.

    We skip heavy / long fields (``text``, ``logprobs``) to keep summary tables
    concise. Timing fields from ``PerformanceResult`` are appended explicitly so
    they appear in a predictable order if present.
    """
    # Determine GenerationResult fields (excluding 'text' and 'logprobs')
    gen_fields: list[str] = []
    for r in results:
        if r.generation is not None and dataclasses.is_dataclass(r.generation):
            gen_fields = [
                f.name
                for f in dataclasses.fields(r.generation)
                if f.name not in ("text", "logprobs")
            ]
            break

    # Combine with PerformanceResult timing fields
    return gen_fields + PERFORMANCE_TIMING_FIELDS


def _get_field_value(result: PerformanceResult, field_name: str) -> MetricValue:
    """Get field value from either GenerationResult or PerformanceResult."""
    if field_name in PERFORMANCE_TIMING_FIELDS:
        return getattr(result, field_name, None)
    return getattr(result.generation, field_name, None) if result.generation else None


# Helper function to sort results by generation time (lowest to highest)
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
            return float(result.generation_time)

        # Fallback: calculate time from GenerationResult tokens-per-second if available
        if (
            result.generation
            and hasattr(result.generation, "generation_tokens")
            and hasattr(result.generation, "generation_tps")
        ):
            g_tokens = getattr(result.generation, "generation_tokens", 0) or 0
            g_tps = getattr(result.generation, "generation_tps", 0.0) or 0.0
            if g_tps > 0 and g_tokens:
                return float(g_tokens / g_tps)

        return float("inf")  # No timing data available

    return sorted(results, key=get_time_value)


# =============================================================================
# SYSTEM INFO & VERSION DETECTION (Hardware, OS, Dependencies)
# =============================================================================


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
        return cast("SystemProfilerDict", json.loads(data))
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
    max_len: int = max(len(k) for k in versions) + 1 if versions else 10
    for name, ver in sorted(versions.items()):
        name_padded: str = name.ljust(max_len)
        logger.info("%s: %s", name_padded, ver or "")

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
            # Calculate max key length for alignment
            max_key_len = max(len(k) for k in system_info) if system_info else 10
            for key, value in system_info.items():
                key_padded = key.ljust(max_key_len)
                logger.info("%s: %s", key_padded, value)
        else:
            logger.debug("No system information available.")
    except (OSError, RuntimeError, ValueError) as err:
        logger.debug("Skipping system info block: %s", err)


# =============================================================================
# IMAGE & EXIF METADATA PROCESSING (File handling, GPS, EXIF extraction)
# =============================================================================


# --- File Handling ---
# Simplified the `find_most_recent_file` function by using `max` with a generator.
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


# Improved error handling in `print_image_dimensions`.
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
def _process_ifd0(exif_raw: Mapping[int, Any]) -> ExifDict:
    exif_decoded: ExifDict = {}
    for tag_id, value in exif_raw.items():
        # Skip SubIFD pointers, we'll handle them separately
        if tag_id in (ExifTags.Base.ExifOffset, ExifTags.Base.GPSInfo):
            continue
        # tag_id is int per signature, no cast needed
        tag_name: str = TAGS.get(tag_id, str(tag_id))
        exif_decoded[tag_name] = value
    return exif_decoded


def _process_exif_subifd(exif_raw: SupportsExifIfd) -> ExifDict:
    out: ExifDict = {}
    try:
        exif_ifd: Any = exif_raw.get_ifd(ExifTags.IFD.Exif)
        if exif_ifd:
            out.update({TAGS.get(tag_id, str(tag_id)): value for tag_id, value in exif_ifd.items()})
    except (KeyError, AttributeError, TypeError):
        logger.warning("Could not extract Exif SubIFD")
    return out


def _process_gps_ifd(exif_raw: SupportsExifIfd) -> GPSDict | None:
    try:
        gps_ifd: Any = exif_raw.get_ifd(ExifTags.IFD.GPSInfo)
        if isinstance(gps_ifd, dict) and gps_ifd:
            gps_decoded: GPSDict = {}
            for gps_tag_id, gps_value in gps_ifd.items():
                # Use modern Pillow GPS enum (10.0+) for type-safe tag name resolution
                try:
                    gps_key = GPS(int(gps_tag_id)).name
                except (ValueError, TypeError):
                    # Fallback to dict lookup for unknown tags
                    gps_key = GPSTAGS.get(int(gps_tag_id), str(gps_tag_id))
                gps_decoded[str(gps_key)] = gps_value
            return gps_decoded
    except (KeyError, AttributeError, TypeError) as gps_err:
        logger.warning("Could not extract GPS IFD: %s", gps_err)
    return None


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
    parsed_url = urllib.parse.urlparse(image_str)
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
            # Use requests for better security audit and timeout handling
            response = requests.get(image_str, timeout=30)
            response.raise_for_status()
            img_data = io.BytesIO(response.content)
            img = Image.open(img_data)
        else:
            # Local file path
            img = Image.open(Path(image_path))

        with img:
            exif_raw: Any = img.getexif()
            if not exif_raw:
                logger.warning("No EXIF data found in %s", image_str)
                return None

            # exif_raw is Any (from img.getexif logic), but we expect it to match our protocols
            # casts removed as redundant since Any matches everything,
            # and passing Any to these functions is valid if unsafe.
            # Using cast() here was just visual noise.
            # Cast to SupportsExifIfd because standard PIL stubs don't yet see
            # get_ifd() on Exif class or treat it as a union that doesn't fully satisfy the
            # protocol.
            exif_decoded: ExifDict = _process_ifd0(exif_raw)
            exif_decoded.update(_process_exif_subifd(cast("SupportsExifIfd", exif_raw)))
            gps_decoded = _process_gps_ifd(cast("SupportsExifIfd", exif_raw))
            if gps_decoded:
                exif_decoded["GPSInfo"] = gps_decoded
            return exif_decoded
    except (FileNotFoundError, UnidentifiedImageError):
        logger.exception("Error reading image: %s", image_str)
    except (OSError, ValueError, urllib.error.URLError) as e:
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


# Reduce return count and use named constants
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
                num = float(seq[0])
                den = float(seq[1])
                return num / den if den != 0 else None
            except (ValueError, TypeError):
                return None
        return to_float(str(v))

    components = [_to_val(coord[i]) if i < clen else 0.0 for i in range(3)]

    if any(c is None for c in components[:clen]):
        return None

    return (components[0] or 0.0, components[1] or 0.0, components[2] or 0.0)


def _extract_exif_date(img_path: PathLike, exif_data: ExifDict) -> str | None:
    # Try EXIF date tags in priority order (using tuple unpacking for cleaner code)
    for tag in EXIF_DATE_TAGS:
        if exif_date := exif_data.get(tag):
            break
    else:
        exif_date = None

    if exif_date:
        # Try to parse with known formats
        try:
            local_tz = get_localzone()
            for fmt in DATE_FORMATS:
                try:
                    dt = datetime.strptime(str(exif_date), fmt).replace(tzinfo=UTC)
                    return dt.astimezone(local_tz).strftime("%Y-%m-%d %H:%M:%S %Z")
                except ValueError:
                    continue
            # If no format matched, return raw date string
            return str(exif_date)
        except (TypeError, UnicodeDecodeError) as err:
            logger.warning("Could not localize EXIF date: %s", err)
            return str(exif_date)

    # Fallback to filesystem mtime
    try:
        local_tz = get_localzone()
        return datetime.fromtimestamp(
            Path(img_path).stat().st_mtime,
            tz=local_tz,
        ).strftime("%Y-%m-%d %H:%M:%S %Z")
    except OSError as err:
        logger.debug("Could not get file mtime: %s", err)
        return None


def _extract_exif_time(img_path: PathLike, exif_data: ExifDict) -> str | None:
    """Extract just the local time portion from EXIF data.

    Returns time in HH:MM:SS format (24-hour), or None if unavailable.
    """
    # Try EXIF date tags in priority order
    for tag in EXIF_DATE_TAGS:
        if exif_date := exif_data.get(tag):
            break
    else:
        exif_date = None

    if exif_date:
        # Try to parse with known formats
        try:
            local_tz = get_localzone()
            for fmt in DATE_FORMATS:
                try:
                    dt = datetime.strptime(str(exif_date), fmt).replace(tzinfo=UTC)
                    return dt.astimezone(local_tz).strftime("%H:%M:%S")
                except ValueError:
                    continue
            # If no format matched, cannot extract time
        except (TypeError, UnicodeDecodeError) as err:
            logger.warning("Could not extract time from EXIF date: %s", err)
        return None

    # Fallback to filesystem mtime
    try:
        local_tz = get_localzone()
        return datetime.fromtimestamp(
            Path(img_path).stat().st_mtime,
            tz=local_tz,
        ).strftime("%H:%M:%S")
    except OSError as err:
        logger.debug("Could not get file mtime for time: %s", err)
        return None


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
        return str(value).replace("\x00", "").strip()

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
                with contextlib.suppress(UnicodeDecodeError, ValueError):
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


def _extract_gps_str(gps_info_raw: Mapping[Any, Any] | None) -> str | None:
    """Extract formatted GPS string from EXIF GPS info dictionary.

    Converts raw EXIF GPS data (DMS format) into human-readable decimal degrees
    with cardinal directions. Handles byte decoding and various EXIF tag formats.

    Args:
        gps_info_raw: Raw GPS info dict from EXIF with numeric or string keys

    Returns:
        Formatted GPS string like "37.775139°N, 122.418336°W" or None if invalid

    Examples:
        >>> # Standard GPS EXIF data with DMS coordinates
        >>> gps_data = {
        ...     1: b'N',  # GPSLatitudeRef
        ...     2: (37.0, 46.0, 30.5),  # GPSLatitude
        ...     3: b'W',  # GPSLongitudeRef
        ...     4: (122.0, 25.0, 6.0)  # GPSLongitude
        ... }
        >>> _extract_gps_str(gps_data)
        '37.775139°N, 122.418336°W'

        >>> # Missing required fields returns None
        >>> _extract_gps_str({1: b'N', 2: (37.0, 46.0, 30.5)})
        None

        >>> # Invalid input returns None
        >>> _extract_gps_str(None)
        None

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


def _extract_iptc_metadata(image_path: PathLike) -> dict[str, Any]:
    """Extract IPTC/IIM metadata (keywords, caption) from an image.

    Uses Pillow's IptcImagePlugin to read standard IPTC records:
        - (2, 25): Keywords (multi-valued)
        - (2, 120): Caption/Abstract
    """
    try:
        with Image.open(Path(image_path)) as img:
            iptc: dict[tuple[int, int], Any] | None = IptcImagePlugin.getiptcinfo(img)
            if not iptc:
                return {}

            result: dict[str, Any] = {}

            # Keywords (2, 25) — may be a single bytes or a list of bytes
            raw_keywords = iptc.get((2, 25), [])
            if isinstance(raw_keywords, bytes):
                raw_keywords = [raw_keywords]
            keywords: list[str] = []
            for kw in raw_keywords:
                decoded = kw.decode("utf-8", errors="replace") if isinstance(kw, bytes) else str(kw)
                if decoded.strip():
                    keywords.append(decoded.strip())
            if keywords:
                result["iptc_keywords"] = keywords

            # IPTC caption/abstract record
            caption_raw = iptc.get((2, 120))
            if isinstance(caption_raw, bytes):
                caption_str = caption_raw.decode("utf-8", errors="replace").strip()
                if caption_str:
                    result["iptc_caption"] = caption_str
            elif isinstance(caption_raw, str) and caption_raw.strip():
                result["iptc_caption"] = caption_raw.strip()

            return result
    except (OSError, ValueError, AttributeError):
        logger.debug("Failed to extract IPTC metadata from %s", image_path)
    return {}


def _xmp_alt_text(container: dict[str, Any], rdf_ns: str) -> str | None:
    """Extract a text value from an XMP rdf:Alt container."""
    if not isinstance(container, dict):
        return None
    alt = container.get(f"{rdf_ns}Alt", {})
    if not isinstance(alt, dict):
        return None
    text = alt.get(f"{rdf_ns}li", "")
    if isinstance(text, dict):
        text = text.get("#text", "")
    return text.strip() if isinstance(text, str) and text.strip() else None


def _extract_xmp_metadata(image_path: PathLike) -> dict[str, Any]:
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
            xmp: dict[str, Any] = img.getxmp()
            if not xmp:
                return {}

            result: dict[str, Any] = {}

            # Navigate: xmpmeta → RDF → Description
            desc_block = (
                xmp.get("xmpmeta", {}).get(f"{rdf_ns}RDF", {}).get(f"{rdf_ns}Description", {})
            )
            if not isinstance(desc_block, dict):
                return {}

            # dc:subject → keywords list
            subject = desc_block.get(f"{dc_ns}subject", {})
            if isinstance(subject, dict):
                bag = subject.get(f"{rdf_ns}Bag", {})
                if isinstance(bag, dict):
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

    # Date, Time, GPS
    metadata["date"] = _extract_exif_date(img_path, exif_data)
    metadata["time"] = _extract_exif_time(img_path, exif_data)
    metadata["gps"] = _extract_gps_str(exif_data.get("GPSInfo"))

    # Description: prefer IPTC caption → XMP description → EXIF ImageDescription
    description = (
        iptc.get("iptc_caption") or xmp.get("xmp_description") or _extract_description(exif_data)
    )
    metadata["description"] = description

    # Title: from XMP dc:title (falls back to None)
    metadata["title"] = xmp.get("xmp_title")

    # Keywords: merge IPTC → XMP → Windows XP, deduplicated
    metadata["keywords"] = _merge_keywords(
        iptc.get("iptc_keywords", []),
        xmp.get("xmp_keywords", []),
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
    """Render selected EXIF tags in a colored table.

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

    # Prepare data for tabulate with colors
    header_color: str = Colors.BLUE
    important_color: str = Colors.YELLOW

    # Create colored headers
    headers: list[str] = [
        Colors.colored("Tag", Colors.BOLD, header_color),
        Colors.colored("Value", Colors.BOLD, header_color),
    ]

    # Create table rows with appropriate coloring
    rows: list[list[str]] = []
    for tag_name, value_display, is_important_tag in tags_to_print:
        tag_display: str = (
            Colors.colored(tag_name, Colors.BOLD, important_color) if is_important_tag else tag_name
        )
        rows.append([tag_display, value_display])

    # Generate table using tabulate with outline format for clean borders without row separators
    table: str = tabulate(
        rows,
        headers=headers,
        tablefmt="fancy_grid",
        colalign=["left", "left"],
    )

    # Print title and table with decorative separators
    table_lines: list[str] = table.split("\n")
    # Use a consistent terminal-based width for header rules to avoid ragged lines
    # Use a clamped terminal width by default; if --width is set, get_terminal_width
    # will return the explicit override and ignore the max clamp.
    header_width: int = max(40, get_terminal_width(max_width=100))

    # Print the title with consistent rule width
    log_rule(header_width, char="=", color=Colors.BLUE, bold=True)
    logger.info(
        title.center(header_width),
        extra={"style_hint": LogStyles.HEADER, "style_width": header_width},
    )
    log_rule(header_width, char="=", color=Colors.BLUE, bold=True)

    # Print the tabulated table
    for line in table_lines:
        logger.info(line)
    log_rule(header_width, char="=", color=Colors.BLUE, bold=True)


def _format_table_field_value(
    field_name: str,
    res: PerformanceResult,
) -> str:
    """Format a single field value for table display.

    Args:
        field_name: Name of the field to format
        res: Performance result containing the data

    Returns:
        Formatted string value for the field
    """
    if field_name == "model_name":
        return res.model_name

    if field_name == "output":
        if res.success and res.generation:
            text = str(getattr(res.generation, "text", ""))
            # Truncate repetitive output for readability
            text = _truncate_repetitive_output(text)
            # Truncate to [MAX_OUTPUT_LINES] lines for table display (full text shown in main trace)
            lines = text.splitlines()
            if (
                len(lines) > MAX_OUTPUT_LINES
            ):  # This constant should be part of the quality issues config
                text = "\n".join(lines[:MAX_OUTPUT_LINES]) + "\n..."
            return _truncate_text_preview(text, max_chars=MAX_OUTPUT_PREVIEW_CHARS)
        error_text = (
            f"Error: {res.error_stage} - {res.error_message}"
            if res.error_message
            else "Unknown error"
        )
        return _truncate_text_preview(error_text, max_chars=MAX_OUTPUT_PREVIEW_CHARS)

    if field_name == "quality_issues":
        # Truncate quality issues for Markdown table display
        value = _get_field_value(res, field_name)
        formatted_value = format_field_value(field_name, value)
        return _truncate_quality_issues(formatted_value)

    # Default: format the field value normally
    value = _get_field_value(res, field_name)
    return format_field_value(field_name, value)


def _prepare_table_data(
    results: list[PerformanceResult],
    header_separator: str = "<br>",
    *,
    include_output: bool = True,
) -> tuple[list[str], list[list[str]], list[str]]:
    """Prepare headers, rows, and field names for reports.

    Args:
        results: List of PerformanceResult objects.
        header_separator: String to use for separating header lines (default: "<br>").
        include_output: Whether to include the output preview column.

    Returns:
        A tuple containing:
        - list[str]: Headers for the table.
        - list[list[str]]: Rows of data for the table.
        - list[str]: The names of the fields.
    """
    if not results:
        return [], [], []

    result_set = ResultSet(results)
    field_names = ["model_name", *result_set.get_fields()]
    if include_output:
        field_names.append("output")
    sorted_results = result_set.results

    # Create headers
    headers = []
    for field_name in field_names:
        if field_name in FIELD_ABBREVIATIONS:
            line1, line2 = FIELD_ABBREVIATIONS[field_name]
            # Split long headers for better readability in reports
            # Only add separator if we actually have two parts to show
            # Force split if using newline separator (CLI), otherwise check length threshold
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

    # Create rows
    rows: list[list[str]] = []
    for res in sorted_results:
        row = [_format_table_field_value(field_name, res) for field_name in field_names]
        rows.append(row)

    return headers, rows, field_names


def _mark_failed_rows_in_html(html_table: str, results: list[PerformanceResult]) -> str:
    """Add data attributes and classes to rows for filtering in the HTML table."""
    sorted_results = _sort_results_by_time(results)
    table_rows = html_table.split("<tr>")
    # Keep preamble and header row (index 0 and 1)
    new_table_rows = [table_rows[0], table_rows[1]]

    for i, res in enumerate(sorted_results):
        # Data rows start at index 2
        if i + 2 < len(table_rows):
            row_html = table_rows[i + 2]

            # Add data attributes for filtering
            if not res.success:
                # Determine error category
                error_stage = res.error_stage or "unknown"
                error_type = res.error_type or "error"
                error_package = res.error_package or "unknown"

                # Add both class and data attributes for flexible filtering
                row_html = row_html.replace(
                    "<tr>",
                    f'<tr class="failed" data-status="failed" '
                    f'data-error-stage="{error_stage}" data-error-type="{error_type}" '
                    f'data-error-package="{error_package}">',
                    1,
                )
                # Also mark first td with failed class for background color
                row_html = row_html.replace("<td", '<td class="failed"', 1)
            else:
                row_html = row_html.replace(
                    "<tr>",
                    '<tr class="success" data-status="success">',
                    1,
                )

            new_table_rows.append(row_html)

    return "<tr>".join(new_table_rows)


def _wrap_output_column_in_details(html_table: str, output_col_idx: int) -> str:
    """Wrap the output column content in <details>/<summary> for expandability.

    Args:
        html_table: The HTML table string
        output_col_idx: The index of the output column (0-based)

    Returns:
        Modified HTML table with output column wrapped in details/summary tags
    """
    preview_length = 100

    # Pattern to match table cells in data rows (not header)
    # We'll process each row and wrap the last td content
    lines = html_table.split("\n")
    result_lines: list[str] = []

    for original_line in lines:
        # Check if this is a data row (contains <td> tags)
        if "<td" in original_line and "</td>" in original_line:
            # Find all <td>...</td> cells in this row
            cells = re.findall(r"<td[^>]*>.*?</td>", original_line)
            if len(cells) > output_col_idx:
                # Get the last cell (output column)
                output_cell = cells[output_col_idx]

                # Extract the content between <td...> and </td>
                match = re.match(r"(<td[^>]*>)(.*?)(</td>)", output_cell, re.DOTALL)
                if match:
                    opening_tag, content, closing_tag = match.groups()

                    # Create preview (first N chars of actual text)
                    # Content is already HTML-escaped by tabulate, so unescape to get real text
                    # for accurate character counting (not entity counting)
                    text_content = html.unescape(content)
                    preview_text = text_content[:preview_length]
                    if len(text_content) > preview_length:
                        preview_text += "..."

                    # Wrap in details/summary
                    # Escape the preview text for HTML (it was unescaped above for char counting)
                    # The full content is already escaped by tabulate
                    wrapped_content = (
                        f"<details><summary>{html.escape(preview_text)}</summary>"
                        f"<div style='margin-top: 0.5em;'>{content}</div></details>"
                    )
                    new_cell = opening_tag + wrapped_content + closing_tag

                    # Replace the old cell with the new one
                    cells[output_col_idx] = new_cell

                    # Reconstruct the line with updated cells
                    cell_iter = iter(cells)

                    def repl(_: re.Match[str], ci: Iterator[str] = cell_iter) -> str:
                        return next(ci)

                    updated_line = re.sub(
                        r"<td[^>]*>.*?</td>",
                        repl,
                        original_line,
                    )
                    result_lines.append(updated_line)
                else:
                    result_lines.append(original_line)
            else:
                result_lines.append(original_line)
        else:
            result_lines.append(original_line)

    return "\n".join(result_lines)


def analyze_model_issues(
    results: list[PerformanceResult],
    context: str | None = None,
) -> ModelIssueSummary:
    """Analyze results to identify common model issues and calculate performance highlights.

    Args:
        results: List of model performance results
        context: Optional context string (from prompt) for cataloging utility analysis
    """
    baseline = _compute_metadata_baseline_utility(context)
    baseline_score = baseline[0] if baseline is not None else None
    baseline_grade = baseline[1] if baseline is not None else None

    failed_models: list[FailedModelIssue] = []
    repetitive_models: list[RepetitiveModelIssue] = []
    hallucination_models: list[HallucinationModelIssue] = []
    verbose_models: list[VerboseModelIssue] = []
    formatting_issues: list[FormattingModelIssue] = []
    excessive_bullets: list[ExcessiveBulletsIssue] = []
    cataloging_grades: dict[str, list[str]] = {}
    low_utility_models: list[LowUtilityModelIssue] = []
    utility_scores: list[CatalogingScoreRecord] = []

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
        "cataloging_avg_score": 0.0,
        "low_utility_models": low_utility_models,
    }

    improves_metadata: list[str] | None = None
    neutral_vs_metadata: list[str] | None = None
    worse_than_metadata: list[str] | None = None
    if baseline_score is not None and baseline_grade is not None:
        improves_metadata = []
        neutral_vs_metadata = []
        worse_than_metadata = []
        summary["metadata_baseline_score"] = baseline_score
        summary["metadata_baseline_grade"] = baseline_grade
        summary["cataloging_improves_metadata"] = improves_metadata
        summary["cataloging_neutral_vs_metadata"] = neutral_vs_metadata
        summary["cataloging_worse_than_metadata"] = worse_than_metadata

    successful = [r for r in results if r.success]
    _populate_summary_performance_highlights(summary, successful)

    for res in results:
        if not res.success:
            failed_models.append((res.model_name, res.error_stage, res.error_message))
            continue
        if not res.generation:
            continue

        text = getattr(res.generation, "text", "") or ""
        generation_tokens = getattr(res.generation, "generation_tokens", 0)
        prompt_tokens = getattr(res.generation, "prompt_tokens", None)

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

        score, grade, weakness, delta = _compute_utility_snapshot(
            text,
            context,
            baseline_score=baseline_score,
        )
        utility_scores.append((res.model_name, score, grade, weakness, delta))
        cataloging_grades.setdefault(grade, []).append(res.model_name)

        if grade in ("D", "F"):
            low_utility_models.append((res.model_name, score, grade, weakness))

        _bucket_metadata_delta(
            model_name=res.model_name,
            delta=delta,
            improves_metadata=improves_metadata,
            neutral_vs_metadata=neutral_vs_metadata,
            worse_than_metadata=worse_than_metadata,
        )

    if utility_scores:
        best = max(utility_scores, key=lambda row: row[1])
        worst = min(utility_scores, key=lambda row: row[1])
        summary["cataloging_best"] = (best[0], best[1], best[2])
        summary["cataloging_worst"] = (worst[0], worst[1], worst[2])
        summary["cataloging_avg_score"] = sum(
            score for _model, score, _grade, _weakness, _delta in utility_scores
        ) / len(utility_scores)
        if baseline_score is not None:
            deltas = [delta for _m, _s, _g, _w, delta in utility_scores if delta is not None]
            if deltas:
                summary["cataloging_avg_delta"] = sum(deltas) / len(deltas)

    return summary


def _populate_summary_performance_highlights(
    summary: ModelIssueSummary,
    successful: list[PerformanceResult],
) -> None:
    """Populate speed/memory highlights and aggregate resource stats."""
    if not successful:
        return

    fastest = max(successful, key=lambda r: getattr(r.generation, "generation_tps", 0) or 0)
    fastest_tps = getattr(fastest.generation, "generation_tps", 0) or 0
    summary["fastest_model"] = (fastest.model_name, fastest_tps)

    most_efficient = min(
        successful,
        key=lambda r: getattr(r.generation, "peak_memory", float("inf")) or float("inf"),
    )
    efficient_mem = getattr(most_efficient.generation, "peak_memory", 0) or 0
    summary["most_efficient_model"] = (most_efficient.model_name, efficient_mem)

    fastest_load = min(
        successful,
        key=lambda r: getattr(r, "model_load_time", float("inf")) or float("inf"),
    )
    load_time = getattr(fastest_load, "model_load_time", 0) or 0
    summary["fastest_load_model"] = (fastest_load.model_name, load_time)

    total_tps = sum(getattr(r.generation, "generation_tps", 0) or 0 for r in successful)
    summary["average_tps"] = total_tps / len(successful)
    summary["successful_count"] = len(successful)

    total_mem = sum(getattr(r.generation, "peak_memory", 0) or 0 for r in successful)
    summary["total_peak_memory"] = total_mem
    summary["average_peak_memory"] = total_mem / len(successful)

    total_tokens = sum(
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


def compute_performance_statistics(results: list[PerformanceResult]) -> PerformanceStats:
    """Compute performance statistics (min, max, avg) for successful runs.

    Uses single-pass aggregation to build stats for all fields at once
    reducing overhead from repeated filtering and type conversions.
    """
    stats: PerformanceStats = {}
    successful_results = [r for r in results if r.success and r.generation]
    if not successful_results:
        return stats

    fields_to_stat = [
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
            value = _get_field_value(res, field)
            if value is not None and is_numeric_value(value):
                # Convert to float once
                try:
                    field_values[field].append(float(value))
                except (ValueError, TypeError):
                    continue

    # Compute min/max/avg for fields with data
    for field, values in field_values.items():
        if values:
            stats[field] = {
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
            }

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
            ("💾 Most efficient", most_efficient_model[0], most_efficient_model[1], "{:.1f} GB")
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


def _format_top_performers_html(summary: ModelIssueSummary) -> list[str]:
    parts = []
    top_metrics = _collect_top_performer_metrics(summary)
    average_tps = summary.get("average_tps")
    successful_count = summary.get("successful_count")
    if top_metrics or (average_tps is not None and successful_count is not None):
        parts.append("<h3>🏆 Performance Highlights</h3><ul>")
        for label, model, value, fmt in top_metrics:
            parts.append(
                f"<li><b>{label}:</b> <code>{html.escape(model)}</code> ({fmt.format(value)})</li>",
            )
        if average_tps is not None and successful_count is not None:
            parts.append(
                f"<li><b>📊 Average TPS:</b> {average_tps:.1f} "
                f"across {successful_count} models</li>",
            )

        parts.append("</ul>")

    resource_metrics = _collect_resource_usage_metrics(summary)
    if resource_metrics:
        parts.append("<h3>📈 Resource Usage</h3><ul>")
        for label, value, fmt in resource_metrics:
            parts.append(
                f"<li><b>{label}:</b> {fmt.format(value)}</li>",
            )
        parts.append("</ul>")

    return parts


def _format_quality_issues_html(summary: ModelIssueSummary) -> list[str]:
    quality_parts = []

    failed_models = summary.get("failed_models", [])
    if failed_models:
        quality_parts.append(
            f"<li><b class='metric-bad'>❌ Failed Models ({len(failed_models)}):</b><ul>",
        )
        quality_parts.extend(
            [
                f"<li><code>{html.escape(model)}</code> ({html.escape(stage or 'Unknown')})</li>"
                for model, stage, _ in failed_models
            ],
        )
        quality_parts.append("</ul></li>")

    repetitive_models = summary.get("repetitive_models", [])
    if repetitive_models:
        quality_parts.append(
            f"<li><b class='metric-warn'>🔄 Repetitive Output ({len(repetitive_models)}):</b><ul>",
        )
        quality_parts.extend(
            [
                f"<li><code>{html.escape(model)}</code> "
                f"(token: <code>{html.escape(token or '?')}</code>)</li>"
                for model, token in repetitive_models
            ],
        )
        quality_parts.append("</ul></li>")

    hallucination_models = summary.get("hallucination_models", [])
    if hallucination_models:
        quality_parts.append(
            f"<li><b class='metric-warn'>👻 Hallucinations ({len(hallucination_models)}):</b><ul>",
        )
        quality_parts.extend(
            [f"<li><code>{html.escape(model)}</code></li>" for model, _ in hallucination_models],
        )
        quality_parts.append("</ul></li>")

    formatting_issues = summary.get("formatting_issues", [])
    if formatting_issues:
        quality_parts.append(
            f"<li><b class='metric-warn'>📝 Formatting Issues ({len(formatting_issues)}):</b><ul>",
        )
        quality_parts.extend(
            [f"<li><code>{html.escape(model)}</code></li>" for model, _ in formatting_issues],
        )
        quality_parts.append("</ul></li>")

    parts = []
    if quality_parts:
        parts.append("<h3>⚠️ Quality Issues</h3><ul>")
        parts.extend(quality_parts)
        parts.append("</ul>")

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


def _format_cataloging_summary_html(summary: ModelIssueSummary) -> list[str]:
    """Format cataloging utility summary as HTML."""
    parts: list[str] = []

    # Only show if we have cataloging data
    if not summary.get("cataloging_best"):
        return parts

    parts.append("<h3>📚 Cataloging Utility Summary</h3>")

    # Grade distribution overview
    grade_counts = _cataloging_grade_distribution_items(summary)
    if grade_counts:
        parts.append(f"<p><b>Grade Distribution:</b> {' | '.join(grade_counts)}</p>")

    # Average score
    avg_score = summary.get("cataloging_avg_score", 0)
    if avg_score > 0:
        parts.append(f"<p><b>Average Utility Score:</b> {avg_score:.0f}/100</p>")
    metadata_breakdown = _cataloging_vs_metadata_breakdown(summary)
    if metadata_breakdown is not None:
        baseline_score, baseline_grade, avg_delta, better, neutral, worse = metadata_breakdown
        baseline_emoji = GRADE_EMOJIS.get(baseline_grade, "❌")
        parts.append(
            "<p><b>Existing Metadata Baseline:</b> "
            f"{baseline_emoji} {baseline_grade} ({baseline_score:.0f}/100)</p>",
        )
        parts.append(
            "<p><b>Vs Existing Metadata:</b> "
            f"Avg Δ {avg_delta:+.0f} | Better: {better}, Neutral: {neutral}, Worse: {worse}</p>",
        )

    # Best and worst performers
    parts.append("<ul>")
    best_entry = summary.get("cataloging_best")
    if best_entry is not None:
        model, score, grade = best_entry
        emoji = GRADE_EMOJIS.get(grade, "")
        parts.append(
            f"<li><b>Best for cataloging:</b> <code>{html.escape(model)}</code> "
            f"({emoji} {grade}, {score:.0f}/100)</li>",
        )
    worst_entry = summary.get("cataloging_worst")
    if worst_entry is not None:
        model, score, grade = worst_entry
        emoji = GRADE_EMOJIS.get(grade, "")
        parts.append(
            f"<li><b>Worst for cataloging:</b> <code>{html.escape(model)}</code> "
            f"({emoji} {grade}, {score:.0f}/100)</li>",
        )
    parts.append("</ul>")

    # Low utility warnings
    low_utility = summary.get("low_utility_models", [])
    if low_utility:
        parts.append(
            f"<p><b class='metric-warn'>⚠️ {len(low_utility)} models "
            "with low utility (D/F):</b></p>",
        )
        parts.append("<ul>")
        for model, score, grade, weakness in low_utility:
            emoji = GRADE_EMOJIS.get(grade, "")
            parts.append(
                f"<li><code>{html.escape(model)}</code>: {emoji} {grade} ({score:.0f}/100) "
                f"- {html.escape(weakness)}</li>",
            )
        parts.append("</ul>")

    return parts


def _format_cataloging_summary_text(summary: ModelIssueSummary) -> list[str]:
    """Format cataloging utility summary as Markdown text."""
    parts: list[str] = []

    # Only show if we have cataloging data
    if not summary.get("cataloging_best"):
        return parts

    parts.append("## 📚 Cataloging Utility Summary")
    parts.append("")

    # Grade distribution overview
    grade_counts = _cataloging_grade_distribution_items(summary)
    if grade_counts:
        parts.append(f"**Grade Distribution:** {' | '.join(grade_counts)}")
        parts.append("")

    # Average score
    avg_score = summary.get("cataloging_avg_score", 0)
    if avg_score > 0:
        parts.append(f"**Average Utility Score:** {avg_score:.0f}/100")
        parts.append("")
    metadata_breakdown = _cataloging_vs_metadata_breakdown(summary)
    if metadata_breakdown is not None:
        baseline_score, baseline_grade, avg_delta, better, neutral, worse = metadata_breakdown
        baseline_emoji = GRADE_EMOJIS.get(baseline_grade, "❌")
        parts.append(
            f"**Existing Metadata Baseline:** {baseline_emoji} {baseline_grade} "
            f"({baseline_score:.0f}/100)",
        )
        parts.append(
            f"**Vs Existing Metadata:** Avg Δ {avg_delta:+.0f} | "
            f"Better: {better}, Neutral: {neutral}, Worse: {worse}",
        )
        parts.append("")

    # Best and worst performers
    best_entry = summary.get("cataloging_best")
    if best_entry is not None:
        model, score, grade = best_entry
        emoji = GRADE_EMOJIS.get(grade, "")
        parts.append(f"- **Best for cataloging:** `{model}` ({emoji} {grade}, {score:.0f}/100)")
    worst_entry = summary.get("cataloging_worst")
    if worst_entry is not None:
        model, score, grade = worst_entry
        emoji = GRADE_EMOJIS.get(grade, "")
        parts.append(f"- **Worst for cataloging:** `{model}` ({emoji} {grade}, {score:.0f}/100)")
    parts.append("")

    # Low utility warnings
    low_utility = summary.get("low_utility_models", [])
    if low_utility:
        parts.append(f"### ⚠️ {len(low_utility)} Models with Low Utility (D/F)")
        parts.append("")
        for model, score, grade, weakness in low_utility:
            emoji = GRADE_EMOJIS.get(grade, "")
            parts.append(f"- `{model}`: {emoji} {grade} ({score:.0f}/100) - {weakness}")
        parts.append("")

    return parts


def format_issues_summary_html(summary: ModelIssueSummary, stats: PerformanceStats) -> str:
    """Format the issues and statistics summary as an HTML string."""
    parts = []
    parts.extend(_format_top_performers_html(summary))
    parts.extend(_format_cataloging_summary_html(summary))
    parts.extend(_format_quality_issues_html(summary))

    # General Stats
    if stats:
        parts.append("<h3>📊 Aggregate Statistics (Successful Runs)</h3><ul>")
        for field, data in stats.items():
            parts.append(
                f"<li><b>{format_field_label(field)}</b>: "
                f"Avg: {format_field_value(field, data['avg'])} | "
                f"Min: {format_field_value(field, data['min'])} | "
                f"Max: {format_field_value(field, data['max'])}</li>",
            )
        parts.append("</ul>")

    return "".join(parts)


def _format_top_performers_text(summary: ModelIssueSummary) -> list[str]:
    parts = []
    top_metrics = _collect_top_performer_metrics(summary)
    average_tps = summary.get("average_tps")
    successful_count = summary.get("successful_count")
    if top_metrics or (average_tps is not None and successful_count is not None):
        parts.append("## 🏆 Performance Highlights")
        parts.append("")  # Blank line after heading (MD022)

        for label, model, value, fmt in top_metrics:
            parts.append(f"- **{label}:** `{model}` ({fmt.format(value)})")
        if average_tps is not None and successful_count is not None:
            parts.append(
                f"- **📊 Average TPS:** {average_tps:.1f} across {successful_count} models",
            )

        parts.append("")  # Blank line after list (MD032)

    resource_metrics = _collect_resource_usage_metrics(summary)
    if resource_metrics:
        parts.append("## 📈 Resource Usage")
        parts.append("")  # Blank line after heading (MD022)

        for label, value, fmt in resource_metrics:
            parts.append(f"- **{label}:** {fmt.format(value)}")

        parts.append("")  # Blank line after list (MD032)

    return parts


def _format_failures_by_package_text(results: list[PerformanceResult]) -> list[str]:
    """Generate a breakdown of failures organized by responsible package for actionable reporting.

    This helps framework maintainers quickly identify which issues belong to them.
    """
    parts: list[str] = []
    failed = [r for r in results if not r.success]
    if not failed:
        return parts

    # Group failures by package
    by_package: dict[str, list[PerformanceResult]] = {}
    for res in failed:
        pkg = res.error_package or "unknown"
        by_package.setdefault(pkg, []).append(res)

    # Sort packages by failure count (descending) for priority
    sorted_packages = sorted(by_package.items(), key=lambda x: -len(x[1]))

    parts.append("## 🚨 Failures by Package (Actionable)")
    parts.append("")
    # Disable MD060 (table column style) as this table may not be perfectly aligned
    parts.append("<!-- markdownlint-disable MD060 -->")
    parts.append("")
    parts.append("| Package | Failures | Error Types | Affected Models |")
    parts.append("|---------|----------|-------------|-----------------|")

    for pkg, failures in sorted_packages:
        error_types = sorted({r.error_stage or "unknown" for r in failures})
        models = [f"`{r.model_name}`" for r in failures]
        parts.append(
            f"| `{pkg}` | {len(failures)} | {', '.join(error_types)} | {', '.join(models)} |",
        )

    parts.append("")
    parts.append("<!-- markdownlint-enable MD060 -->")
    parts.append("")

    # Generate per-package actionable sections
    parts.append("### Actionable Items by Package")
    parts.append("")

    for pkg, failures in sorted_packages:
        parts.append(f"#### {pkg}")
        parts.append("")
        for res in failures:
            parts.append(f"- **{res.model_name}** ({res.error_stage})")
            # Add truncated error message
            error_msg = res.error_message or ""
            if len(error_msg) > ERROR_MESSAGE_TRUNCATE_LEN:
                error_msg = error_msg[: ERROR_MESSAGE_TRUNCATE_LEN - 3] + "..."
            parts.append(f"  - Error: `{error_msg}`")
            if res.error_type:
                parts.append(f"  - Type: `{res.error_type}`")
        parts.append("")

    return parts


# =============================================================================
# DIAGNOSTICS REPORT — Upstream issue filing aid
# =============================================================================

# Priority thresholds for diagnostics report classification
_DIAGNOSTICS_HIGH_COUNT: Final[int] = 2  # ≥ N models = High priority cluster
_DIAGNOSTICS_TRACEBACK_TAIL_LINES: Final[int] = 6  # Lines to keep from traceback tail
_DIAGNOSTICS_OUTPUT_SNIPPET_LEN: Final[int] = 200  # Max chars for sample output
_DIAGNOSTICS_RECENT_RUN_WINDOW: Final[int] = 3  # Runs used for reproducibility signal

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
    failure_history: dict[str, FailureHistoryContext] = dataclasses.field(default_factory=dict)


def _append_markdown_code_block(
    parts: list[str],
    content: str,
    *,
    language: str = "text",
) -> None:
    """Append a fenced code block with consistent spacing."""
    parts.append("")
    parts.append(f"```{language}")
    parts.append(content)
    parts.append("```")
    parts.append("")


def _append_markdown_table(
    parts: list[str],
    *,
    header: str,
    separator: str,
    rows: list[str],
) -> None:
    """Append a Markdown table with a blank line after it."""
    parts.append(header)
    parts.append(separator)
    parts.extend(rows)
    parts.append("")


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
        parts.extend(body_lines)
        parts.append("")


def _render_collapsible_model_blocks(
    *,
    summary: str,
    entries: list[tuple[str, str]],
) -> list[str]:
    """Render model-specific text blocks inside a collapsible details section."""
    if not entries:
        return []

    parts = [f"<details><summary>{summary}</summary>", ""]
    for model_name, block_text in entries:
        safe_model = DIAGNOSTICS_ESCAPER.escape(model_name)
        parts.append(f"### `{safe_model}`")
        _append_markdown_code_block(parts, block_text, language="text")
    parts.append("</details>")
    parts.append("")
    return parts


def _format_traceback_tail(traceback_str: str | None) -> str | None:
    """Extract the last meaningful lines from a full traceback.

    Strips blank lines and returns the tail suitable for inclusion in issue
    reports.  Returns None when no useful info can be extracted.
    """
    if not traceback_str:
        return None
    lines = [ln for ln in traceback_str.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    tail = lines[-_DIAGNOSTICS_TRACEBACK_TAIL_LINES:]
    return "\n".join(tail)


def _format_traceback_full(traceback_str: str | None) -> str | None:
    """Return full traceback text (trimmed), or None when missing."""
    if not traceback_str:
        return None
    full = traceback_str.strip()
    return full or None


def _sanitize_captured_output_for_report(captured_text: str) -> str:
    """Normalize captured stdout/stderr for Markdown diagnostics readability."""
    if not captured_text:
        return ""

    text = ANSI_ESCAPE_RE.sub("", captured_text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x08", "")
    text = "".join(ch for ch in text if ch in {"\n", "\t"} or ch.isprintable())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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
            if recent_considered >= _DIAGNOSTICS_RECENT_RUN_WINDOW:
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
    comparison: dict[str, list[str]] | None,
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


def _format_recent_repro_ratio(history_info: FailureHistoryContext | None) -> str:
    """Format reproducibility ratio string such as ``2/3 recent runs failed``."""
    if not history_info:
        return "n/a"
    recent_failures = history_info.recent_failures
    recent_considered = history_info.recent_considered
    if recent_considered <= 0:
        return "n/a"
    return f"{recent_failures}/{recent_considered} recent runs failed"


def _diagnostics_full_tracebacks_section(cluster_results: list[PerformanceResult]) -> list[str]:
    """Build collapsed full traceback blocks for all models in a cluster."""
    traceback_entries: list[tuple[str, str]] = []
    for result in cluster_results:
        traceback_text = _format_traceback_full(result.error_traceback)
        if traceback_text is None:
            continue
        traceback_entries.append((result.model_name, traceback_text))

    return _render_collapsible_model_blocks(
        summary="Full tracebacks (all models in this cluster)",
        entries=traceback_entries,
    )


def _diagnostics_captured_output_section(cluster_results: list[PerformanceResult]) -> list[str]:
    """Build collapsed captured stdout/stderr blocks for models in a cluster."""
    with_output = [
        (
            r.model_name,
            _sanitize_captured_output_for_report(r.captured_output_on_fail or ""),
        )
        for r in cluster_results
    ]
    with_output = [(name, out) for name, out in with_output if out]
    return _render_collapsible_model_blocks(
        summary="Captured stdout/stderr (all models in this cluster)",
        entries=with_output,
    )


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

    Returns:
        Mapping from canonical signature to the list of results sharing it.
    """
    clusters: dict[str, list[PerformanceResult]] = {}
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

        clusters.setdefault(signature, []).append(res)

    return clusters


def _diagnostics_priority(
    cluster_size: int,
    error_stage: str | None,
) -> str:
    """Assign a priority label for an error cluster in the diagnostics report."""
    if cluster_size >= _DIAGNOSTICS_HIGH_COUNT:
        return "High"
    if error_stage in {"Weight Mismatch", "Config Missing"}:
        return "Low"
    return "Medium"


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
            signals.append((res, "Empty output despite successful run", "mlx-vlm"))
            continue

        if (
            prompt_tokens >= QUALITY.long_prompt_tokens_threshold
            and generated_tokens < QUALITY.min_output_tokens_for_ratio
            and ratio < QUALITY.min_output_ratio
        ):
            signals.append(
                (
                    res,
                    f"Long-context low output ratio ({ratio:.1%})",
                    "mlx-vlm / mlx",
                ),
            )
            continue

        if (
            prompt_tokens >= QUALITY.severe_prompt_tokens_threshold
            and qa is not None
            and (qa.is_repetitive or qa.is_context_ignored)
        ):
            symptom = (
                "Repetition under extreme prompt length"
                if qa.is_repetitive
                else "Context dropped under extreme prompt length"
            )
            signals.append((res, symptom, "mlx-vlm / mlx"))

    return signals


def _diagnostics_header(
    *,
    total: int,
    n_failed: int,
    n_harness: int,
    n_preflight: int,
    n_success: int,
    versions: LibraryVersionDict,
    system_info: dict[str, str],
    image_path: Path | None,
) -> list[str]:
    """Build title, summary, and environment sections of the diagnostics report."""
    parts: list[str] = []
    version = versions.get("mlx-vlm") or "unknown"
    parts.append(
        f"# Diagnostics Report — {n_failed} failure(s), "
        f"{n_harness} harness issue(s) (mlx-vlm {version})",
    )
    parts.append("")
    _append_markdown_section(
        parts,
        title="## Summary",
        body_lines=[
            (
                f"Automated benchmarking of **{total} locally-cached VLM models** "
                f"found **{n_failed} hard failure(s)** and "
                f"**{n_harness} harness/integration issue(s)** "
                f"plus **{n_preflight} preflight compatibility warning(s)** "
                f"in successful models. {n_success} of {total} models succeeded."
            ),
        ],
    )

    if image_path and image_path.exists():
        try:
            size_mb = image_path.stat().st_size / (1024 * 1024)
            parts.append(
                f"Test image: `{image_path.name}` ({size_mb:.1f} MB).",
            )
            parts.append("")
        except OSError:
            pass

    # Environment table
    _append_markdown_section(parts, title="## Environment")
    table_rows: list[str] = []
    for lib in _DIAGNOSTICS_LIB_NAMES:
        ver = versions.get(lib, "")
        if ver:
            table_rows.append(f"| {lib} | {DIAGNOSTICS_ESCAPER.escape(str(ver))} |")
    for key in _DIAGNOSTICS_SYSTEM_KEYS:
        val = system_info.get(key, "")
        if val:
            table_rows.append(f"| {key} | {DIAGNOSTICS_ESCAPER.escape(str(val))} |")
    _append_markdown_table(
        parts,
        header="| Component | Version |",
        separator="| --------- | ------- |",
        rows=table_rows,
    )
    return parts


def _build_environment_fingerprint(
    *,
    versions: LibraryVersionDict,
    system_info: dict[str, str],
) -> str:
    """Build compact environment fingerprint for issue templates."""
    tokens: list[str] = []
    for label, value in (
        ("python", system_info.get("Python Version")),
        ("platform", system_info.get("Platform")),
        ("chip", system_info.get("GPU/Chip")),
        ("mlx", versions.get("mlx")),
        ("mlx-vlm", versions.get("mlx-vlm")),
        ("mlx-lm", versions.get("mlx-lm")),
        ("transformers", versions.get("transformers")),
    ):
        if value:
            tokens.append(f"{label}={value}")
    return "; ".join(tokens) if tokens else "unknown"


def _issue_target_for_package(
    package: str,
    *,
    model_name: str,
) -> tuple[str, str]:
    """Map failure package attribution to a likely upstream issue tracker."""
    target_map: dict[str, tuple[str, str]] = {
        "mlx-vlm": ("mlx-vlm", "https://github.com/ml-explore/mlx-vlm/issues/new"),
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
            f"https://huggingface.co/{urllib.parse.quote(model_name, safe='/')}",
        )
    return ("check_models", "https://github.com/jrp2014/check_models/issues/new")


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


def _diagnostics_preflight_section(preflight_issues: Sequence[str]) -> list[str]:
    """Build diagnostics section for compatibility warnings seen during preflight."""
    if not preflight_issues:
        return []

    parts: list[str] = ["---", ""]
    _append_markdown_section(
        parts,
        title=f"## Preflight Compatibility Warnings ({len(preflight_issues)} issue(s))",
        body_lines=[
            "These warnings were detected before inference. They are non-fatal but "
            "should be tracked as potential upstream compatibility issues.",
        ],
    )

    for issue in preflight_issues:
        package = _guess_preflight_issue_package(issue)
        target_name, target_url = _issue_target_for_package(
            package,
            model_name="unknown/model",
        )
        escaped_issue = DIAGNOSTICS_ESCAPER.escape(issue)
        parts.append(f"- `{escaped_issue}`")
        parts.append(
            f"  - Likely package: `{package}`; suggested tracker: `{target_name}` ({target_url})",
        )
    parts.append("")
    return parts


def _build_cluster_issue_template(
    *,
    representative: PerformanceResult,
    cluster_results: list[PerformanceResult],
    versions: LibraryVersionDict,
    system_info: dict[str, str],
    image_path: Path | None,
    run_args: argparse.Namespace | None,
    repro_bundle_path: Path | None,
) -> str:
    """Build a copy/paste-ready issue body for one failure cluster."""
    pkg = representative.error_package or "unknown"
    target_name, target_url = _issue_target_for_package(
        pkg,
        model_name=representative.model_name,
    )
    stage = representative.error_stage or _classify_error(representative.error_message or "")
    code = representative.error_code or _build_canonical_error_code(
        error_stage=stage,
        error_package=pkg,
        failure_phase=representative.failure_phase,
    )
    signature = representative.error_signature or _build_error_signature(
        error_code=code,
        error_message=representative.error_message,
        error_traceback=representative.error_traceback,
    )
    failure_phase = representative.failure_phase or "unknown"
    env_fingerprint = _build_environment_fingerprint(versions=versions, system_info=system_info)
    repro_tokens = _build_repro_command_tokens(
        image_path=image_path,
        run_args=run_args,
        include_selection=False,
    )
    repro_command = shlex_join([*repro_tokens, "--models", representative.model_name])
    affected_models = ", ".join(sorted({r.model_name for r in cluster_results}))
    error_head = (representative.error_message or "Unknown error").splitlines()[0].strip()
    traceback_tail = _format_traceback_tail(representative.error_traceback) or "n/a"
    bundle_text = str(repro_bundle_path) if repro_bundle_path is not None else "n/a"

    lines = [
        "### Summary",
        error_head,
        "",
        "### Classification",
        f"- Package attribution: `{pkg}`",
        f"- Failure phase: `{failure_phase}`",
        f"- Error stage: `{stage}`",
        f"- Canonical code: `{code}`",
        f"- Signature: `{signature}`",
        "",
        "### Affected Models",
        f"{affected_models}",
        "",
        "### Minimal Reproduction",
        "```bash",
        repro_command,
        "```",
        "",
        "### Environment Fingerprint",
        f"`{env_fingerprint}`",
        "",
        "### Repro Bundle",
        f"`{bundle_text}`",
        "",
        "### Traceback Tail",
        "```text",
        traceback_tail,
        "```",
        "",
        "### Suggested Tracker",
        f"- `{target_name}`: {target_url}",
    ]
    return "\n".join(lines)


def _diagnostics_failure_clusters(
    results: list[PerformanceResult],
    *,
    diagnostics_context: DiagnosticsContext,
    versions: LibraryVersionDict,
    system_info: dict[str, str],
    image_path: Path | None,
    run_args: argparse.Namespace | None,
    repro_bundles: Mapping[str, Path] | None,
) -> list[str]:
    """Build the failure-cluster sections of the diagnostics report."""
    failed = [r for r in results if not r.success]
    if not failed:
        return []

    clusters = _cluster_failures_by_pattern(results)
    sorted_clusters = sorted(clusters.items(), key=lambda kv: -len(kv[1]))

    parts: list[str] = ["---", ""]
    for idx, (cluster_signature, cluster_results) in enumerate(sorted_clusters, 1):
        stage = cluster_results[0].error_stage or "Error"
        pkg = cluster_results[0].error_package or "unknown"
        n = len(cluster_results)
        priority = _diagnostics_priority(n, stage)
        rep = cluster_results[0]
        rep_phase = rep.failure_phase or "unknown"
        rep_code = rep.error_code or _build_canonical_error_code(
            error_stage=stage,
            error_package=pkg,
            failure_phase=rep_phase,
        )

        parts.append(
            f"## {idx}. {stage} — {n} model(s) [`{pkg}`] (Priority: {priority})",
        )
        parts.append("")

        full_msg = (rep.error_message or "").split("\n")[0].strip()
        parts.append(f"**Error:** `{full_msg}`")
        parts.append(f"**Failure phase:** `{rep_phase}`")
        parts.append(f"**Canonical code:** `{rep_code}`")
        parts.append(f"**Signature:** `{cluster_signature}`")
        parts.append("")

        # Affected models table
        table_rows: list[str] = []
        for r in cluster_results:
            model = DIAGNOSTICS_ESCAPER.escape(r.model_name)
            stage = DIAGNOSTICS_ESCAPER.escape(r.error_stage or "")
            pkg = DIAGNOSTICS_ESCAPER.escape(r.error_package or "unknown")
            phase = DIAGNOSTICS_ESCAPER.escape(r.failure_phase or "unknown")
            code = DIAGNOSTICS_ESCAPER.escape(r.error_code or rep_code)
            regression = "yes" if r.model_name in diagnostics_context.regressions else "no"
            history_info = diagnostics_context.failure_history.get(r.model_name)
            first_seen = DIAGNOSTICS_ESCAPER.escape(
                history_info.first_failure_timestamp if history_info else "unknown",
            )
            recent_repro = DIAGNOSTICS_ESCAPER.escape(_format_recent_repro_ratio(history_info))
            table_rows.append(
                f"| `{model}` | {phase} | {stage} | {pkg} | `{code}` | "
                f"{regression} | {first_seen} | {recent_repro} |",
            )
        _append_markdown_table(
            parts,
            header=(
                "| Model | Failure Phase | Error Stage | Package | Code | "
                "Regression vs Prev | First Seen Failing | Recent Repro |"
            ),
            separator=(
                "| ----- | ------------- | ----------- | ------- | ---- | "
                "------------------ | ------------------ | ------------ |"
            ),
            rows=table_rows,
        )

        # Per-model error messages (only when they differ across the cluster)
        msgs = {
            r.model_name: (r.error_message or "").split("\n")[0].strip() for r in cluster_results
        }
        if len(set(msgs.values())) > 1:
            parts.append("**Per-model error messages:**")
            parts.append("")
            parts.extend(f"- `{model}`: `{msg}`" for model, msg in msgs.items())
            parts.append("")

        # Traceback excerpt from the representative model
        tb_tail = _format_traceback_tail(rep.error_traceback)
        if tb_tail:
            parts.append("**Traceback (tail):**")
            _append_markdown_code_block(parts, tb_tail, language="text")

        parts.append("### Issue Template")
        parts.append("")
        bundle_path = (
            repro_bundles.get(rep.model_name)
            if repro_bundles is not None and rep.model_name in repro_bundles
            else None
        )
        issue_template = _build_cluster_issue_template(
            representative=rep,
            cluster_results=cluster_results,
            versions=versions,
            system_info=system_info,
            image_path=image_path,
            run_args=run_args,
            repro_bundle_path=bundle_path,
        )
        parts.append("<details><summary>Copy/paste GitHub issue template</summary>")
        parts.append("")
        _append_markdown_code_block(parts, issue_template, language="markdown")
        parts.append("</details>")
        parts.append("")

        parts.extend(_diagnostics_full_tracebacks_section(cluster_results))
        parts.extend(_diagnostics_captured_output_section(cluster_results))

    return parts


def _diagnostics_harness_section(
    harness_results: list[tuple[PerformanceResult, str]],
) -> list[str]:
    """Build the harness/integration issues section of the diagnostics report."""
    if not harness_results:
        return []

    parts: list[str] = ["---", ""]
    _append_markdown_section(
        parts,
        title=f"## Harness/Integration Issues ({len(harness_results)} model(s))",
        body_lines=[
            "These models completed successfully but show integration problems "
            "(including empty output, encoding corruption, stop-token leakage, "
            "or prompt-template/long-context issues) that indicate stack bugs "
            "rather than inherent model quality limits.",
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

        likely_package = "mlx-vlm"
        if harness_type == "long_context":
            likely_package = "mlx-vlm / mlx"

        parts.append(f"### `{res.model_name}` — {harness_type}")
        parts.append("")
        parts.append(
            f"**Tokens:** prompt={fmt_num(prompt_tokens)}, "
            f"generated={fmt_num(generated_tokens)}, ratio={ratio_text}",
        )
        parts.append(f"**Likely package:** `{likely_package}`")
        if res.quality_issues:
            parts.append(f"**Quality flags:** {res.quality_issues}")
        parts.append("")
        if harness_details:
            parts.append("**Details:** " + ", ".join(harness_details))
            parts.append("")
        snippet_source = text.strip() or "<empty output>"
        snippet = snippet_source[:_DIAGNOSTICS_OUTPUT_SNIPPET_LEN]
        if len(snippet_source) > _DIAGNOSTICS_OUTPUT_SNIPPET_LEN:
            snippet += "..."
        parts.append("**Sample output:**")
        _append_markdown_code_block(parts, snippet, language="text")

    return parts


def _diagnostics_stack_signal_section(
    stack_signals: list[tuple[PerformanceResult, str, str]],
) -> list[str]:
    """Build a section for likely stack issues observed in successful runs."""
    if not stack_signals:
        return []

    parts: list[str] = ["---", ""]
    _append_markdown_section(
        parts,
        title=f"## Potential Stack Issues ({len(stack_signals)} model(s))",
        body_lines=[
            "These models technically succeeded, but token/output patterns suggest likely "
            "integration/runtime issues worth checking upstream.",
        ],
    )

    rows: list[str] = []
    for res, symptom, package_hint in stack_signals:
        gen = res.generation
        prompt_tokens = int(getattr(gen, "prompt_tokens", 0) or 0) if gen else 0
        generated_tokens = int(getattr(gen, "generation_tokens", 0) or 0) if gen else 0
        ratio = f"{(generated_tokens / prompt_tokens):.2%}" if prompt_tokens > 0 else "n/a"
        quality_flags = DIAGNOSTICS_ESCAPER.escape(res.quality_issues or "")
        rows.append(
            "| "
            f"`{DIAGNOSTICS_ESCAPER.escape(res.model_name)}` | "
            f"{fmt_num(prompt_tokens)} | "
            f"{fmt_num(generated_tokens)} | "
            f"{ratio} | "
            f"{DIAGNOSTICS_ESCAPER.escape(symptom)} | "
            f"`{DIAGNOSTICS_ESCAPER.escape(package_hint)}` | "
            f"{quality_flags or '-'} |",
        )

    _append_markdown_table(
        parts,
        header=(
            "| Model | Prompt Tok | Output Tok | Output/Prompt | "
            "Symptom | Likely Package | Quality Flags |"
        ),
        separator=(
            "| ----- | ---------- | ---------- | ------------- | "
            "------- | -------------- | ------------- |"
        ),
        rows=rows,
    )
    return parts


def _diagnostics_priority_table(
    results: list[PerformanceResult],
    harness_results: list[tuple[PerformanceResult, str]],
    preflight_issues: Sequence[str],
) -> list[str]:
    """Build the priority summary table for the diagnostics report."""
    parts: list[str] = ["---", ""]
    _append_markdown_section(parts, title="## Priority Summary")
    table_rows: list[str] = []

    failed = [r for r in results if not r.success]
    if failed:
        clusters = _cluster_failures_by_pattern(results)
        sorted_clusters = sorted(clusters.items(), key=lambda kv: -len(kv[1]))
        for _pattern, cluster_results in sorted_clusters:
            stage = cluster_results[0].error_stage or "Error"
            pkg = cluster_results[0].error_package or "unknown"
            n = len(cluster_results)
            priority = _diagnostics_priority(n, stage)
            names = ", ".join(r.model_name.split("/")[-1] for r in cluster_results)
            # Escape fields
            esc_stage = DIAGNOSTICS_ESCAPER.escape(stage)
            esc_pkg = DIAGNOSTICS_ESCAPER.escape(pkg)
            esc_names = DIAGNOSTICS_ESCAPER.escape(names)
            table_rows.append(f"| **{priority}** | {esc_stage} | {n} ({esc_names}) | {esc_pkg} |")

    if harness_results:
        names = ", ".join(r.model_name.split("/")[-1] for r, _ in harness_results)
        esc_names = DIAGNOSTICS_ESCAPER.escape(names)
        n = len(harness_results)
        table_rows.append(
            f"| **Medium** | Harness/integration | {n} ({esc_names}) | mlx-vlm |",
        )
    if preflight_issues:
        package_names = sorted(
            {_guess_preflight_issue_package(issue) for issue in preflight_issues},
        )
        package_summary = DIAGNOSTICS_ESCAPER.escape(", ".join(package_names))
        n = len(preflight_issues)
        table_rows.append(
            f"| **Medium** | Preflight compatibility warning | {n} issue(s) | "
            f"{package_summary or 'unknown'} |",
        )
    _append_markdown_table(
        parts,
        header="| Priority | Issue | Models Affected | Package |",
        separator="| -------- | ----- | --------------- | ------- |",
        rows=table_rows,
    )
    return parts


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

    parts: list[str] = ["---", ""]
    _append_markdown_section(
        parts,
        title="## History Context",
        body_lines=[
            "Recent reproducibility is measured from history "
            f"(up to last {_DIAGNOSTICS_RECENT_RUN_WINDOW} runs where each model appears).",
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

    table_rows: list[str] = []
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
        table_rows.append(f"| `{esc_model}` | {esc_status} | {first_seen} | {recent_repro} |")
    _append_markdown_table(
        parts,
        header="| Model | Status vs Previous Run | First Seen Failing | Recent Repro |",
        separator="| ----- | ---------------------- | ------------------ | ------------ |",
        rows=table_rows,
    )

    return parts


def _append_repro_input_tokens(
    tokens: list[str],
    *,
    image_path: Path | None,
    run_args: argparse.Namespace | None,
) -> None:
    """Append image/folder input tokens to reproducibility command."""
    if image_path is not None:
        tokens.extend(["--image", str(image_path)])
        return

    if run_args is None:
        return

    run_image = getattr(run_args, "image", None)
    run_folder = getattr(run_args, "folder", None)
    if run_image is not None:
        tokens.extend(["--image", str(run_image)])
    elif run_folder is not None:
        tokens.extend(["--folder", str(run_folder)])


def _append_repro_selection_tokens(tokens: list[str], run_args: argparse.Namespace) -> None:
    """Append model-selection tokens to reproducibility command."""
    models = getattr(run_args, "models", None)
    if models:
        tokens.extend(["--models", *[str(model) for model in models]])

    exclude = getattr(run_args, "exclude", None)
    if exclude:
        tokens.extend(["--exclude", *[str(model) for model in exclude]])


def _append_repro_optional_value_flags(
    tokens: list[str],
    flag_values: Sequence[tuple[str, str | int | float | Path | None]],
) -> None:
    """Append `--flag value` pairs when values are present."""
    for flag, value in flag_values:
        if value is None:
            continue
        tokens.extend([flag, str(value)])


def _append_repro_bool_flags(
    tokens: list[str],
    run_args: argparse.Namespace,
    flag_map: Sequence[tuple[str, str]],
) -> None:
    """Append boolean flags when the corresponding args attribute is truthy."""
    for attr_name, flag in flag_map:
        if bool(getattr(run_args, attr_name, False)):
            tokens.append(flag)


def _append_repro_runtime_tokens(tokens: list[str], run_args: argparse.Namespace) -> None:
    """Append generation/runtime-affecting flags for reproducibility."""
    trust_remote_code = bool(getattr(run_args, "trust_remote_code", True))
    tokens.append("--trust-remote-code" if trust_remote_code else "--no-trust-remote-code")

    _append_repro_optional_value_flags(
        tokens,
        [
            ("--revision", getattr(run_args, "revision", None)),
            ("--adapter-path", getattr(run_args, "adapter_path", None)),
            ("--prompt", getattr(run_args, "prompt", None)),
        ],
    )

    _append_repro_bool_flags(
        tokens,
        run_args,
        [
            ("detailed_metrics", "--detailed-metrics"),
            ("lazy_load", "--lazy-load"),
        ],
    )

    tokens.extend(["--max-tokens", str(getattr(run_args, "max_tokens", DEFAULT_MAX_TOKENS))])
    tokens.extend(["--temperature", str(getattr(run_args, "temperature", DEFAULT_TEMPERATURE))])
    tokens.extend(["--top-p", str(getattr(run_args, "top_p", 1.0))])

    _append_repro_optional_value_flags(
        tokens,
        [
            ("--repetition-penalty", getattr(run_args, "repetition_penalty", None)),
            ("--repetition-context-size", getattr(run_args, "repetition_context_size", None)),
            ("--max-kv-size", getattr(run_args, "max_kv_size", None)),
            ("--kv-bits", getattr(run_args, "kv_bits", None)),
            ("--prefill-step-size", getattr(run_args, "prefill_step_size", None)),
        ],
    )

    kv_group_size = getattr(run_args, "kv_group_size", None)
    if kv_group_size not in {None, 64}:
        tokens.extend(["--kv-group-size", str(kv_group_size)])

    quantized_kv_start = getattr(run_args, "quantized_kv_start", None)
    if quantized_kv_start not in {None, 0}:
        tokens.extend(["--quantized-kv-start", str(quantized_kv_start)])

    tokens.extend(["--timeout", str(getattr(run_args, "timeout", DEFAULT_TIMEOUT))])


def _append_repro_display_tokens(tokens: list[str], run_args: argparse.Namespace) -> None:
    """Append display/config flags when relevant."""
    _append_repro_bool_flags(
        tokens,
        run_args,
        [
            ("verbose", "--verbose"),
            ("no_color", "--no-color"),
            ("force_color", "--force-color"),
        ],
    )

    _append_repro_optional_value_flags(
        tokens,
        [
            ("--width", getattr(run_args, "width", None)),
            ("--quality-config", getattr(run_args, "quality_config", None)),
        ],
    )

    context_marker = getattr(run_args, "context_marker", None)
    if context_marker and context_marker != "Context:":
        tokens.extend(["--context-marker", str(context_marker)])


def _build_repro_command_tokens(
    *,
    image_path: Path | None,
    run_args: argparse.Namespace | None,
    include_selection: bool,
) -> list[str]:
    """Build CLI command tokens for diagnostics reproducibility snippets."""
    tokens = ["python", "-m", "check_models"]

    _append_repro_input_tokens(tokens, image_path=image_path, run_args=run_args)

    if run_args is None:
        return tokens

    if include_selection:
        _append_repro_selection_tokens(tokens, run_args)

    _append_repro_runtime_tokens(tokens, run_args)
    _append_repro_display_tokens(tokens, run_args)

    return tokens


_REPRO_ENV_KEYS: Final[tuple[str, ...]] = (
    "HF_HOME",
    "HF_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "TOKENIZERS_PARALLELISM",
    "TRANSFORMERS_NO_TF",
    "TRANSFORMERS_NO_FLAX",
    "TRANSFORMERS_NO_JAX",
    "MLX_VLM_ALLOW_TF",
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
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
    except OSError:
        return None
    return hasher.hexdigest()


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
    """Write per-model reproducibility bundles for each failed result."""
    failed = [res for res in results if not res.success]
    if not failed:
        return {}

    bundles: dict[str, Path] = {}
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_hash = _sha256_text(prompt)
    image_hash = _sha256_file(image_path) if image_path is not None else None
    image_ref_hash = _sha256_text(str(image_path)) if image_path is not None else None
    env_vars = _collect_repro_env()
    serialized_args = {
        key: _jsonify_cli_value(value) for key, value in sorted(vars(run_args).items())
    }
    preflight_issues = list(_get_run_preflight_issues(run_args))

    for index, result in enumerate(failed, start=1):
        safe_model = _sanitize_bundle_filename(result.model_name)
        signature_token = re.sub(r"[^A-Za-z0-9_]+", "_", result.error_signature or "unknown")
        signature_token = signature_token[:40] if signature_token else "unknown"
        bundle_name = f"{timestamp}_{index:03d}_{safe_model}_{signature_token}.json"
        bundle_path = output_dir / bundle_name

        rerun_tokens = _build_repro_command_tokens(
            image_path=image_path,
            run_args=run_args,
            include_selection=False,
        )
        rerun_command = shlex_join([*rerun_tokens, "--models", result.model_name])

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
            "repro": {
                "rerun_command": rerun_command,
                "seed": serialized_args.get("seed"),
                "prompt_hash_sha256": prompt_hash,
                "prompt_preview": _build_prompt_preview(prompt, max_chars=400),
                "image_path": str(image_path) if image_path is not None else None,
                "image_sha256": image_hash,
                "image_ref_sha256": image_ref_hash,
                "args": serialized_args,
                "env_vars": env_vars,
            },
            "environment": {
                "platform": platform.platform(),
                "system_info": system_info,
                "library_versions": versions,
                "preflight_issues": preflight_issues,
            },
        }

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
    image_path: Path | None,
    run_args: argparse.Namespace | None,
) -> list[str]:
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
    parts: list[str] = []
    _append_markdown_section(parts, title="## Reproducibility")
    _append_markdown_code_block(parts, all_models_command, language="bash")

    if failed:
        parts.append("")
        parts.append("### Target specific failing models")
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

    parts.append("<details><summary>Prompt used (click to expand)</summary>")
    _append_markdown_code_block(parts, prompt, language="text")
    parts.append("</details>")
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
) -> bool:
    """Generate a Markdown diagnostics report structured for upstream issue filing.

    The report clusters failures by root-cause pattern, includes full error
    messages and traceback excerpts, and highlights harness/encoding issues
    from successful models — everything needed to copy-paste into a GitHub
    issue against mlx-vlm, mlx, or transformers.

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

    Returns:
        True if the report was written (i.e. there was something to report),
        False if skipped because there were no failures or harness issues.
    """
    failed = [r for r in results if not r.success]
    successful = [r for r in results if r.success]
    harness_results = _collect_harness_results(successful)
    stack_signals = _collect_stack_issue_signals(successful)
    preflight_issues = list(history.preflight_issues) if history is not None else []

    if not failed and not harness_results and not stack_signals and not preflight_issues:
        return False

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
        n_success=len(successful),
        versions=versions,
        system_info=system_info,
        image_path=image_path,
    )
    parts.extend(
        _diagnostics_failure_clusters(
            results,
            diagnostics_context=diagnostics_context,
            versions=versions,
            system_info=system_info,
            image_path=image_path,
            run_args=run_args,
            repro_bundles=repro_bundles,
        ),
    )
    parts.extend(_diagnostics_preflight_section(preflight_issues))
    parts.extend(_diagnostics_harness_section(harness_results))
    parts.extend(_diagnostics_stack_signal_section(stack_signals))
    parts.extend(
        _diagnostics_history_section(
            failed=failed,
            previous_history=previous_history,
            diagnostics_context=diagnostics_context,
        ),
    )
    parts.extend(_diagnostics_priority_table(results, harness_results, preflight_issues))
    parts.extend(
        _diagnostics_footer(
            failed,
            prompt,
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


def _format_quality_issues_text(summary: ModelIssueSummary) -> list[str]:
    parts = []
    quality_parts = []

    failed_models = summary.get("failed_models", [])
    if failed_models:
        quality_parts.append(f"- **❌ Failed Models ({len(failed_models)}):**")
        for model, stage, _ in failed_models:
            stage_text = stage or "Unknown"
            quality_parts.append(f"  - `{model}` (`{stage_text}`)")

    repetitive_models = summary.get("repetitive_models", [])
    if repetitive_models:
        quality_parts.append(
            f"- **🔄 Repetitive Output ({len(repetitive_models)}):**",
        )
        for model, token in repetitive_models:
            quality_parts.append(f"  - `{model}` (token: `{token or '?'}`)")

    hallucination_models = summary.get("hallucination_models", [])
    if hallucination_models:
        quality_parts.append(
            f"- **👻 Hallucinations ({len(hallucination_models)}):**",
        )
        quality_parts.extend(
            [f"  - `{model}`" for model, _ in hallucination_models],
        )

    formatting_issues = summary.get("formatting_issues", [])
    if formatting_issues:
        quality_parts.append(
            f"- **📝 Formatting Issues ({len(formatting_issues)}):**",
        )
        quality_parts.extend(
            [f"  - `{model}`" for model, _ in formatting_issues],
        )

    if quality_parts:
        parts.append("## ⚠️ Quality Issues")
        parts.append("")  # Blank line after heading (MD022)
        parts.extend(quality_parts)
        parts.append("")

    return parts


def format_issues_summary_text(summary: ModelIssueSummary, stats: PerformanceStats) -> str:
    """Format the issues and statistics summary as a Markdown string."""
    parts = []

    parts.extend(_format_top_performers_text(summary))
    parts.extend(_format_cataloging_summary_text(summary))
    parts.extend(_format_quality_issues_text(summary))

    # General Stats
    if stats:
        parts.append("## 📊 Aggregate Statistics (Successful Runs)")
        parts.append("")  # Blank line after heading (MD022)
        for field, data in stats.items():
            parts.append(
                f"- **{format_field_label(field)}**: "
                f"Avg: {format_field_value(field, data['avg'])} | "
                f"Min: {format_field_value(field, data['min'])} | "
                f"Max: {format_field_value(field, data['max'])}",
            )
        parts.append("")  # Blank line after list (MD032)

    return "\n".join(parts)


def _build_full_html_document(
    *,
    html_table: str,
    versions: LibraryVersionDict,
    prompt: str,
    total_runtime_seconds: float,
    issues_summary_html: str,
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
            {issues_summary_html}
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

    headers, rows, field_names = _prepare_table_data(
        results,
        header_separator="\n",
        include_output=False,
    )
    if not headers or not rows:
        logger.info("No data to display in stats table.")
        return

    # Use tabulate with 'simple' format for clean plain text output
    # Determine column alignment (numeric fields right-aligned)
    colalign = ["right" if is_numeric_field(field) else "left" for field in field_names]

    # Set max widths for specific columns to keep table compact
    # Quality Issues: 20 chars, others: no limit
    maxcolwidths: list[int | None] = []
    for field_name in field_names:
        if field_name == "quality_issues":
            maxcolwidths.append(20)
        else:
            maxcolwidths.append(None)

    # Generate table using tabulate
    table_text = tabulate(
        rows,
        headers=headers,
        tablefmt="plain",
        colalign=colalign,
        maxcolwidths=maxcolwidths,
    )

    # Log the table line by line
    for line in table_text.split("\n"):
        if not line.strip():
            continue
        logger.info(line)


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
) -> None:
    """Write a self-contained HTML summary with aligned table and embedded image."""
    if not results:
        log_warning_note("No results to generate HTML report.")
        return

    headers, rows, field_names = _prepare_table_data(results)

    if not headers or not rows:
        log_warning_note("No table data to generate HTML report.")
        return

    # Generate HTML table using tabulate
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
    html_table = html_table.replace("<td>", '<td class="text">').replace(
        "<th>",
        '<th class="text">',
    )
    for field in field_names:
        if is_numeric_field(field):
            idx = field_names.index(field)
            html_table = html_table.replace(
                f"<td>{rows[0][idx]}",
                f'<td class="numeric">{rows[0][idx]}',
                1,
            )
            html_table = html_table.replace(
                f"<th>{headers[idx]}",
                f'<th class="numeric">{headers[idx]}',
                1,
            )

    # Mark failed rows
    html_table = _mark_failed_rows_in_html(html_table, results)

    # Wrap output column (last column) in <details> for expandability
    output_col_idx = len(field_names) - 1  # output is last column
    html_table = _wrap_output_column_in_details(html_table, output_col_idx)

    # Extract context from prompt for cataloging utility analysis
    context = _extract_context_from_prompt(prompt)

    # Analyze model issues and generate summary
    summary = analyze_model_issues(results, context)
    stats = compute_performance_statistics(results)
    issues_summary_html = format_issues_summary_html(summary, stats)

    # Gather system characteristics for the report
    system_info = get_system_characteristics()

    # Build the full HTML document
    html_content = _build_full_html_document(
        html_table=html_table,
        versions=versions,
        prompt=prompt,
        total_runtime_seconds=total_runtime_seconds,
        issues_summary_html=issues_summary_html,
        system_info=system_info,
        image_path=image_path,
    )

    try:
        with filename.open("w", encoding="utf-8") as f:
            f.write(html_content)
        # Logging handled in finalize_execution
    except OSError:
        logger.exception("Failed to write HTML report to %s", filename)


def _process_markdown_rows(rows: list[list[str]], results: list[PerformanceResult]) -> None:
    """Process table rows for Markdown: escape content and format model names."""
    sorted_results_for_flags = _sort_results_by_time(results)
    for i in range(len(rows)):
        # Wrap model name in backticks to preserve underscores and special chars
        if rows[i][0]:
            rows[i][0] = f"`{rows[i][0]}`"

        last_col_idx = len(rows[i]) - 1
        if last_col_idx < 0:
            continue
        # If corresponding result failed, treat as diagnostics and escape more aggressively
        is_failure = i < len(sorted_results_for_flags) and not sorted_results_for_flags[i].success
        if is_failure:
            rows[i][last_col_idx] = _escape_markdown_diagnostics(rows[i][last_col_idx])
        else:
            # Minimal structural escaping only (protect pipes/HTML-like tags, preserve output
            # as-is otherwise)
            rows[i][last_col_idx] = _escape_markdown_in_text(rows[i][last_col_idx])


def _generate_model_gallery_section(results: list[PerformanceResult]) -> list[str]:
    """Generate the Model Gallery section for the Markdown report."""
    md: list[str] = []
    md.append("## Model Gallery")
    md.append("")
    md.append("Full output from each model:")
    md.append("")
    md.append("")
    md.append("<!-- markdownlint-disable MD013 MD033 -->")
    md.append("")

    sorted_results = _sort_results_by_time(results)
    for res in sorted_results:
        icon = "✅" if res.success else "❌"
        md.append(f"### {icon} {res.model_name}")
        md.append("")
        if not res.success:
            md.extend(_gallery_render_error(res))
        else:
            md.extend(_gallery_render_success(res))
        md.append("")
        md.append("---")
        md.append("")

    md.append("<!-- markdownlint-enable MD013 MD033 -->")
    md.append("")
    return md


def _generate_markdown_table_section(results: list[PerformanceResult]) -> list[str]:
    """Generate the metrics table section for the Markdown report."""
    headers, rows, field_names = _prepare_table_data(results)

    # For Markdown, we need to process headers to remove HTML breaks and use simpler formatting
    markdown_headers = []

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
    _process_markdown_rows(rows, results)

    # Determine column alignment using original field names
    colalign = ["left"] + [
        "right" if is_numeric_field(field_name) else "left" for field_name in field_names[1:]
    ]

    # Generate Markdown table using tabulate with proper GitHub alignment syntax
    markdown_table = tabulate(
        rows,
        headers=markdown_headers,
        tablefmt="pipe",  # Use 'pipe' format for proper GitHub alignment with colons
        colalign=colalign,
    )

    # Normalize trailing spaces per line using shared helper
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


def generate_markdown_report(
    results: list[PerformanceResult],
    filename: Path,
    versions: LibraryVersionDict,
    prompt: str,
    total_runtime_seconds: float,
) -> None:
    """Write a GitHub-friendly Markdown summary with aligned pipe table."""
    if not results:
        log_warning_note("No results to generate Markdown report.")
        return

    # Get table data using our helper function
    headers, rows, _ = _prepare_table_data(results)

    if not headers or not rows:
        log_warning_note("No table data to generate Markdown report.")
        return

    # Extract context from prompt for cataloging utility analysis
    context = _extract_context_from_prompt(prompt)

    # Analyze model issues and generate summary
    summary = analyze_model_issues(results, context)
    stats = compute_performance_statistics(results)
    issues_text = format_issues_summary_text(summary, stats)

    # Gather system characteristics for the report
    system_info = get_system_characteristics()

    # Build the complete markdown content
    md: list[str] = []
    md.append("# Model Performance Results")
    md.append("")
    md.append(f"_Generated on {local_now_str()}_")
    md.append("")
    # Add issues summary before prompt
    if issues_text:
        md.append(issues_text)

    # Add failures-by-package section for actionable reporting
    failures_by_pkg = _format_failures_by_package_text(results)
    if failures_by_pkg:
        md.extend(failures_by_pkg)

    # Embed prompt in a blockquote with a fenced code block to avoid
    # MD032 (lists-need-blank-lines) when the prompt contains list items.
    md.append("> **Prompt used:**")
    md.append(">")
    md.append("")
    md.append("> ```text")
    md.extend(f"> {prompt_line}" for prompt_line in prompt.split("\n"))
    md.append("> ```")
    md.append("")
    md.append(
        "**Note:** Results sorted: errors first, then by generation time (fastest to slowest).",
    )
    md.append("")
    md.append(f"**Overall runtime:** {format_overall_runtime(total_runtime_seconds)}")
    md.append("")

    # Generate table section
    table_md = _generate_markdown_table_section(results)
    md.extend(table_md)

    # --- Model Gallery Section ---
    md.extend(_generate_model_gallery_section(results))

    md.append("---")

    # Add system/hardware information if available
    if system_info:
        md.append("")
        md.append("## System/Hardware Information")
        md.append("")
        for name, value in system_info.items():
            md.append(f"- **{name}**: {value}")
        md.append("")

    md.append("## Library Versions")
    md.append("")
    for name, ver in sorted(versions.items()):
        ver_str = "" if ver is None else ver
        md.append(f"- `{name}`: `{ver_str}`")
    md.append("")
    md.append(f"_Report generated on: {local_now_str()}_")

    # Join and normalize trailing spaces across the entire Markdown document
    # Ensure file ends with single newline (MD047 requirement)
    markdown_content = normalize_markdown_trailing_spaces("\n".join(md)) + "\n"

    try:
        with filename.open("w", encoding="utf-8") as f:
            f.write(markdown_content)
        # Logging handled in finalize_execution
    except OSError:
        logger.exception(
            "Failed to write Markdown report to file %s.",
            str(filename),
        )
    except ValueError:
        logger.exception(
            "A value error occurred while writing Markdown report %s",
            str(filename),
        )


def generate_tsv_report(
    results: list[PerformanceResult],
    filename: Path,
) -> None:
    """Write a TSV (tab-separated values) file of the core results table.

    A ``# generated_at: <timestamp>`` comment line is written first so
    downstream consumers know when the data was produced.  For failed models
    two extra columns (``error_type``, ``error_package``) are appended to
    aid programmatic triage.

    Args:
        results: List of PerformanceResult objects.
        filename: Path where the TSV file will be written.
    """
    if not results:
        log_warning_note("No results to generate TSV report.")
        return

    headers, rows, _ = _prepare_table_data(results)

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
    clean_headers = []
    for header in headers:
        # Remove <br> tags and other HTML
        clean_header = header.replace("<br>", " ").strip()
        clean_header = re.sub(r"<[^>]+>", "", clean_header)
        clean_headers.append(escape_tsv_value(clean_header))

    # Append error diagnostic columns
    clean_headers.extend(["error_type", "error_package"])

    # Clean and escape row data, appending error columns per result
    sorted_results = _sort_results_by_time(list(results))
    clean_rows = []
    for row, res in zip(rows, sorted_results, strict=False):
        clean_row = [escape_tsv_value(str(cell)) for cell in row]
        clean_row.append(escape_tsv_value(res.error_type or ""))
        clean_row.append(escape_tsv_value(res.error_package or ""))
        clean_rows.append(clean_row)

    # Generate TSV using tabulate with tsv format
    tsv_content = tabulate(
        clean_rows,
        headers=clean_headers,
        tablefmt="tsv",
    )

    try:
        with filename.open("w", encoding="utf-8") as f:
            # Metadata comment line (parsers can skip lines starting with #)
            f.write(f"# generated_at: {local_now_str()}\n")
            f.write(tsv_content)
            # Ensure file ends with newline
            if not tsv_content.endswith("\n"):
                f.write("\n")
        # Logging handled in finalize_execution
    except OSError:
        logger.exception("Failed to write TSV report to %s", filename)


def _escape_markdown_in_text(text: str) -> str:
    """Escape structural elements while preserving model-generated markdown.

    Strategy:
    - Escape pipes (|) to prevent breaking the outer table structure
    - Convert newlines to <br> to preserve multi-line output in table cells
    - Escape tag-like sequences (e.g., <s>, </s>) that aren't recognized GitHub formatting
    - PRESERVE model-generated markdown: **bold**, *italic*, `code`, etc. (GitHub renders these)
    - Wrap bare URLs in angle brackets for MD034 compliance

    This allows models to produce markdown formatting that GitHub interprets correctly,
    while preventing output from breaking the report table structure.
    """
    return MARKDOWN_ESCAPER.escape(text)


def _escape_markdown_diagnostics(text: str) -> str:
    """Escape diagnostics text for Markdown tables - minimal approach.

    Error messages are already in table cells, so we only need to:
    - Escape pipes (|) to prevent breaking table structure
    - Convert newlines to <br> for multi-line preservation
    - Escape HTML-like tags that could be misinterpreted

    We do NOT escape *, _, `, ~ as these rarely break tables and
    escaping them makes Python tracebacks harder to read.
    """
    return DIAGNOSTICS_ESCAPER.escape(text)


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
    - Strip any other count of trailing spaces to avoid accidental single-space endings.
    """
    out_lines: list[str] = []
    for ln in md_text.splitlines():
        m = re.search(r"( +)$", ln)
        if not m:
            out_lines.append(ln)
            continue
        spaces = len(m.group(1))
        if spaces == FORMATTING.markdown_hard_break_spaces:
            out_lines.append(ln)
        else:
            out_lines.append(ln[:-spaces])
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
    try:
        # Try to get GPU info on macOS using full path and JSON output for robustness
        # Matches mlx-vlm/tests/test_smoke.py implementation
        if platform.system() == "Darwin":
            result: subprocess.CompletedProcess[str] = subprocess.run(
                ["/usr/sbin/system_profiler", "SPDisplaysDataType", "-json"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0:
                try:
                    data = json.loads(result.stdout)
                    # Navigate the JSON structure: SPDisplaysDataType -> [0] -> sppci_model
                    # Note: keys can vary slightly, but 'sppci_model' or '_name' are common
                    displays = data.get("SPDisplaysDataType", [])
                    if displays:
                        # Try commonly used keys for GPU name
                        gpu_info = displays[0].get("sppci_model") or displays[0].get("_name")
                except json.JSONDecodeError:
                    logger.warning("Failed to parse system_profiler JSON output")
    except (subprocess.SubprocessError, TimeoutError) as e:
        logger.debug("Could not get GPU info: %s", e)
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

    device_info = get_device_info() or {}
    displays = device_info.get("SPDisplaysDataType") or []
    if not displays or not isinstance(displays, list):
        return info

    first = displays[0]
    if not isinstance(first, dict):
        return info

    gpu_cores = first.get("sppci_cores")
    if gpu_cores is not None:
        info["GPU Cores"] = str(gpu_cores)

    metal_family = first.get("spdisplays_mtlgpufamilysupport")
    if metal_family:
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
    repetition_penalty: float | None,
) -> None:
    """Validate sampling parameters are within acceptable ranges."""
    if not 0.0 <= top_p <= 1.0:
        msg = f"top_p must be between 0.0 and 1.0, got {top_p}"
        raise ValueError(msg)

    if repetition_penalty is not None and repetition_penalty < 1.0:
        msg = f"repetition_penalty must be >= 1.0 if specified, got {repetition_penalty}"
        raise ValueError(msg)


def validate_kv_params(
    *,  # Force all parameters to be keyword-only for clarity
    max_kv_size: int | None,
    kv_bits: int | None,
) -> None:
    """Validate KV cache parameters are within acceptable ranges."""
    if max_kv_size is not None and max_kv_size <= 0:
        msg = f"max_kv_size must be > 0 if specified, got {max_kv_size}"
        raise ValueError(msg)

    if kv_bits is not None and kv_bits not in (4, 8):
        msg = f"kv_bits must be 4 or 8 if specified, got {kv_bits}"
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
        repetition_penalty=args.repetition_penalty,
    )

    # Validate KV cache parameters
    validate_kv_params(
        max_kv_size=args.max_kv_size,
        kv_bits=args.kv_bits,
    )


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
        cache_info: HFCacheInfo = scan_cache_dir()
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
    except (OSError, HFValidationError) as cache_err:
        logger.debug("Could not check HF cache integrity: %s", cache_err)


_FAILURE_PHASE_ATTR: Final[str] = "_check_models_failure_phase"

_STAGE_CODE_MAP: Final[dict[str, str]] = {
    "OOM": "OOM",
    "Timeout": "TIMEOUT",
    "Missing Dep": "MISSING_DEP",
    "Lib Version": "LIB_VERSION",
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


def _sanitize_error_token(value: str | None, *, default: str) -> str:
    """Convert free-form labels into stable uppercase token segments."""
    if value is None or not value.strip():
        return default
    token = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().upper()).strip("_")
    return token or default


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
    """Determine which package most likely caused the error.

    Analyzes the error message and optional traceback to identify the
    originating package. This helps direct bug reports to the correct
    repository (mlx, mlx-vlm, mlx-lm, or transformers).

    Args:
        error_msg: The error message string
        traceback_str: Optional full traceback string for deeper analysis

    Returns:
        Package name: 'mlx', 'mlx-vlm', 'mlx-lm', 'transformers', 'huggingface-hub',
                      'model-config', or 'unknown'

    Examples:
        >>> _attribute_error_to_package("[metal::malloc] Attempting to allocate...")
        'mlx'

        >>> _attribute_error_to_package("cannot import name '_validate_images_text_input_order'")
        'transformers'

        >>> _attribute_error_to_package(
        ...     "Model loading failed",
        ...     "File '/path/mlx_vlm/utils.py', line 245..."
        ... )
        'mlx-vlm'
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
                "mlx-vlm/",
                "apply_chat_template",
                "load_image",
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

    for package, patterns in package_definitions:
        if any(pattern in combined for pattern in patterns):
            return package

    # Special case for compound logic check which didn't fit the loop
    if "missing" in combined and "parameters" in combined:
        return "model-config"

    return "unknown"


def _load_model(
    params: ProcessImageParams,
) -> tuple[Module, PythonBackend | TokenizersBackend, Any | None]:
    """Load model from HuggingFace Hub or local path.

    Args:
        params: The parameters for image processing, including model identifier.

    Returns:
        Tuple of (model, processor, config) where config may be None.
    """
    model, processor = load(
        path_or_hf_repo=params.model_identifier,
        adapter_path=params.adapter_path,
        lazy=params.lazy,
        revision=params.revision,
        trust_remote_code=params.trust_remote_code,
    )
    # Note: mlx-vlm.utils.load() is type-hinted to return Union[PreTrainedTokenizer, ...]
    # but at runtime it returns a Processor object (from AutoProcessor).

    return model, processor, getattr(model, "config", None)


def _set_failure_phase(
    phase_callback: Callable[[str], None] | None,
    phase: str,
) -> str:
    """Update the active execution phase and notify optional callback."""
    normalized = _normalise_failure_phase(phase) or phase
    if phase_callback is not None:
        phase_callback(normalized)
    return normalized


def _extract_processor_tokenizer(processor: object) -> object | None:
    """Best-effort extraction of tokenizer from an AutoProcessor-like object."""
    tokenizer = cast("object | None", getattr(processor, "tokenizer", None))
    if tokenizer is not None:
        return tokenizer
    if hasattr(processor, "encode") and hasattr(processor, "decode"):
        return processor
    return None


def _resolve_model_snapshot_path(model_identifier: str) -> Path | None:
    """Resolve local snapshot path for a model identifier when available."""
    if model_identifier.startswith(("/", "./", "../")):
        path = Path(model_identifier)
        return path.resolve() if path.is_dir() else None

    try:
        cache_info: HFCacheInfo = scan_cache_dir()
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
    tokenizer: object | None,
    processor: object,
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

    if not (snapshot_path / "config.json").exists():
        logger.warning(
            "Preflight warning for %s: snapshot missing config.json (%s)",
            model_identifier,
            snapshot_path,
        )

    if getattr(processor, "image_processor", None) is None:
        logger.warning(
            "Preflight warning for %s: loaded processor has no image_processor.",
            model_identifier,
        )

    if tokenizer is not None:
        tokenizer_candidates = (
            "tokenizer_config.json",
            "tokenizer.json",
            "tokenizer.model",
            "vocab.json",
        )
        if not any((snapshot_path / name).exists() for name in tokenizer_candidates):
            logger.warning(
                "Preflight warning for %s: tokenizer artifacts missing from snapshot (%s).",
                model_identifier,
                ", ".join(tokenizer_candidates),
            )

    processor_candidates = ("preprocessor_config.json", "processor_config.json")
    if not any((snapshot_path / name).exists() for name in processor_candidates):
        logger.warning(
            "Preflight warning for %s: processor config missing from snapshot (%s).",
            model_identifier,
            ", ".join(processor_candidates),
        )


def _run_model_preflight_validators(
    *,
    model_identifier: str,
    processor: object,
    config: object | None,
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
    if not (hasattr(tokenizer, "decode") or hasattr(tokenizer, "batch_decode")):
        _raise_preflight_error(
            "Resolved tokenizer does not expose decode/batch_decode.",
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
        processor=processor,
    )


def _ensure_generation_runtime_symbols() -> None:
    """Validate core generation callables before model execution."""
    missing: list[str] = []
    for symbol_name, symbol_value in (
        ("load", load),
        ("apply_chat_template", apply_chat_template),
        ("generate", generate),
    ):
        if not callable(symbol_value):
            missing.append(symbol_name)
    if missing:
        msg = f"Generation runtime unavailable: missing callables ({', '.join(missing)})."
        raise _tag_exception_failure_phase(RuntimeError(msg), "import")


def _run_model_generation(
    params: ProcessImageParams,
    timer: TimingStrategy | None = None,
    phase_callback: Callable[[str], None] | None = None,
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
    """
    model: Module
    processor: Any

    _set_failure_phase(phase_callback, "import")
    _ensure_generation_runtime_symbols()

    # Load model from HuggingFace Hub - this handles automatic download/caching
    # and converts weights to MLX format for Apple Silicon optimization
    _set_failure_phase(phase_callback, "model_load")
    try:
        model, processor, config = _load_model(params)
    except Exception as load_err:
        # Capture model loading errors and run cache diagnostics to distinguish
        # code bugs from environment issues (corrupted cache, incomplete download)
        error_details = f"Model loading failed: {load_err}"
        logger.exception("Failed to load model %s", params.model_identifier)
        _check_hf_cache_integrity(params.model_identifier)

        raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err

    try:
        _run_model_preflight_validators(
            model_identifier=params.model_identifier,
            processor=processor,
            config=config,
            phase_callback=phase_callback,
        )
    except ValueError as preflight_err:
        message = f"Model preflight failed for {params.model_identifier}: {preflight_err}"
        logger.exception("Model preflight validation failed for %s", params.model_identifier)
        phase = (
            _extract_failure_phase(preflight_err, fallback="model_preflight") or "model_preflight"
        )
        raise _tag_exception_failure_phase(ValueError(message), phase) from preflight_err

    # Apply model-specific chat template - each model has its own conversation format
    # (e.g., Llama uses <|begin_of_text|>, Phi-3 uses <|user|>, etc.)
    _set_failure_phase(phase_callback, "prefill")
    try:
        formatted_prompt: str | list[Any] = apply_chat_template(
            processor=processor,
            config=config,
            prompt=params.prompt,
            num_images=1,
        )
    except (OSError, ValueError, RuntimeError, TypeError, AttributeError, KeyError) as prefill_err:
        msg = f"Prompt prefill failed for {params.model_identifier}: {prefill_err}"
        logger.exception("Prompt prefill failed for %s", params.model_identifier)
        raise _tag_exception_failure_phase(ValueError(msg), "prefill") from prefill_err
    # Handle list return from apply_chat_template
    if isinstance(formatted_prompt, list):
        formatted_prompt = "\n".join(str(m) for m in formatted_prompt)

    # Time the generation process manually since MLX VLM doesn't include timing
    # Use injected timer or default to PerfCounterTimer
    if timer is None:
        timer = PerfCounterTimer()

    timer.start()
    _set_failure_phase(phase_callback, "decode")
    try:
        # Build optional kwargs for generate() — only pass when explicitly set.
        # Note: we use a separate dict + **unpacking instead of a conditional
        # inline expression because pyrefly cannot type-check
        # ``**({k: v} if cond else {})`` against generate()'s **kwargs signature
        # (it incorrectly infers the value type as incompatible with other kw params).
        extra_kwargs: dict[str, Any] = {}
        if params.prefill_step_size is not None:
            extra_kwargs["prefill_step_size"] = params.prefill_step_size

        output: GenerationResult | SupportsGenerationResult = generate(
            model=model,
            processor=cast("PythonBackend", processor),
            prompt=formatted_prompt,
            image=str(params.image_path),
            verbose=params.verbose,
            temperature=params.temperature,
            top_p=params.top_p,
            repetition_penalty=params.repetition_penalty,
            repetition_context_size=params.repetition_context_size,
            max_kv_size=params.max_kv_size,
            kv_bits=params.kv_bits,
            kv_group_size=params.kv_group_size,
            quantized_kv_start=params.quantized_kv_start,
            max_tokens=params.max_tokens,
            **extra_kwargs,
        )
    except TimeoutError as gen_to_err:
        msg = f"Generation timed out for model {params.model_identifier}: {gen_to_err}"
        # Re-raise to be handled by outer TimeoutError branch
        raise _tag_exception_failure_phase(TimeoutError(msg), "decode") from gen_to_err
    except (OSError, ValueError) as gen_known_err:
        # Known I/O or validation-style issues
        msg = f"Model generation failed for {params.model_identifier}: {gen_known_err}"
        logger.exception("Generation error for %s", params.model_identifier)
        raise _tag_exception_failure_phase(ValueError(msg), "decode") from gen_known_err
    except (RuntimeError, TypeError, AttributeError, KeyError) as gen_err:
        # Model-specific runtime errors (weights, config, tensor ops, missing attributes)
        msg = f"Model runtime error during generation for {params.model_identifier}: {gen_err}"
        logger.exception("Runtime error for %s", params.model_identifier)
        raise _tag_exception_failure_phase(ValueError(msg), "decode") from gen_err

    # Force GPU synchronization to ensures timing includes all pending compute (MLX is lazy)
    mx.synchronize()
    duration = timer.stop()

    # Capture memory metrics immediately after generation while model is still active
    # This must happen before mx.eval() which can change memory state
    active_mem_bytes = mx.get_active_memory()
    cache_mem_bytes = mx.get_cache_memory()

    # Add timing and memory to the GenerationResult object dynamically
    # Cast to our Protocol which includes the time attribute we're adding
    result = cast("SupportsGenerationResult", output)
    result.time = duration
    result.active_memory = active_mem_bytes / (1024**3)  # Convert to GB
    result.cache_memory = cache_mem_bytes / (1024**3)  # Convert to GB

    mx.eval(model.parameters())
    return result


def _build_failure_result(
    *,
    model_name: str,
    error: TimeoutError | OSError | ValueError | RuntimeError,
    captured_output: str | None,
    failure_phase: str | None = None,
) -> PerformanceResult:
    """Build a standardized failure result payload for a model run."""
    error_msg = str(error)
    tb_str = traceback.format_exc()
    resolved_phase = _extract_failure_phase(error, fallback=failure_phase)
    classified_stage = _classify_error(error_msg)
    error_package = _attribute_error_to_package(error_msg, tb_str)
    error_code = _build_canonical_error_code(
        error_stage=classified_stage,
        error_package=error_package,
        failure_phase=resolved_phase,
    )
    error_signature = _build_error_signature(
        error_code=error_code,
        error_message=error_msg,
        error_traceback=tb_str,
    )
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
        error_package=error_package,
        error_traceback=tb_str,
        generation_time=None,
        model_load_time=None,
        total_time=None,
    )


def process_image_with_model(params: ProcessImageParams) -> PerformanceResult:
    """Process an image with a Vision Language Model, managing stats and errors."""
    model: Module | None = None
    processor: PythonBackend | TokenizersBackend | None = None
    arch, gpu_info = get_system_info()
    stdout_capture = _TeeCaptureStream(sys.stdout)
    stderr_capture = _TeeCaptureStream(sys.stderr)

    # Track overall timing
    total_start_time = time.perf_counter()
    model_load_time: float | None = None
    generation_time: float | None = None
    current_phase: str = "input_validation"

    def _update_phase(phase: str) -> None:
        nonlocal current_phase
        current_phase = _normalise_failure_phase(phase) or phase

    try:
        _update_phase("input_validation")
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
            contextlib.redirect_stdout(cast("TextIO", stdout_capture)),
            contextlib.redirect_stderr(cast("TextIO", stderr_capture)),
            TimeoutManager(params.timeout),
        ):
            output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                params=params,
                phase_callback=_update_phase,
            )
        if params.verbose:
            logger.debug(
                "[verbose passthrough end] mlx-vlm.generate output for %s",
                params.model_identifier,
            )

        # Extract timing from GenerationResult if available
        generation_time = getattr(output, "time", None)
        total_end_time = time.perf_counter()
        total_time = total_end_time - total_start_time

        # Estimate model load time (total - generation time)
        if generation_time is not None:
            model_load_time = max(0.0, total_time - generation_time)

        # Read memory metrics from GenerationResult (captured inside _run_model_generation)
        active_mem_gb = getattr(output, "active_memory", None) or 0.0
        cache_mem_gb = getattr(output, "cache_memory", None) or 0.0

        return PerformanceResult(
            model_name=params.model_identifier,
            generation=output,
            success=True,
            generation_time=generation_time,
            model_load_time=model_load_time,
            total_time=total_time,
            active_memory=active_mem_gb if active_mem_gb > 0 else None,
            cache_memory=cache_mem_gb if cache_mem_gb > 0 else None,
        )
    except (TimeoutError, OSError, ValueError, RuntimeError) as e:
        captured_output = _merge_captured_output(
            stdout_capture.getvalue(),
            stderr_capture.getvalue(),
        )
        return _build_failure_result(
            model_name=params.model_identifier,
            error=e,
            captured_output=captured_output,
            failure_phase=current_phase,
        )
    finally:
        _update_phase("cleanup")
        # Aggressive cleanup matching mlx-vlm/tests/test_smoke.py
        if model is not None:
            del model
        if processor is not None:
            del processor

        # Force synchronization and garbage collection when MLX runtime is available.
        synchronize_fn = getattr(mx, "synchronize", None)
        if callable(synchronize_fn):
            synchronize_fn()
        gc.collect()

        # Clear both Metal and MLX caches for thorough GPU memory cleanup.
        clear_cache_fn = getattr(mx, "clear_cache", None)
        if callable(clear_cache_fn):
            clear_cache_fn()

        reset_peak_memory_fn = getattr(mx, "reset_peak_memory", None)
        if callable(reset_peak_memory_fn):
            reset_peak_memory_fn()
        logger.debug("Cleaned up resources for model %s", params.model_identifier)


# --- Main Execution Helper Functions ---


def print_cli_header(title: str) -> None:
    """Print a formatted CLI header with the given title."""
    width = get_terminal_width(max_width=100)
    log_rule(width, char="=", color=Colors.BLUE, bold=True)
    logger.info(
        title,
        extra={"style_hint": LogStyles.HEADER, "style_width": width},
    )
    log_rule(width, char="=", color=Colors.BLUE, bold=True)


def print_cli_section(title: str, *, show_rule: bool = True) -> None:
    """Print a formatted CLI section header with visual prefix."""
    width = get_terminal_width(max_width=100)
    if show_rule:
        log_rule(width, char="─", color=Colors.BLUE, bold=False)
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


# --- New Structured Logging Helpers (Consistent Output Formatting) ---


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


def log_metric_tree(prefix: str, label: str, value: str, *, indent: str = "") -> None:
    """Log a tree-structured metric line (e.g., '├─ Total: 1,234').

    Args:
        prefix: Tree prefix characters (├─, └─, etc.)
        label: Metric label (e.g., 'Total:', 'Prompt:')
        value: Formatted value to display
        indent: Additional indentation before the prefix
    """
    # Example output: "     ├─ Total:      1,234 tok/s"
    formatted = f"{indent}{prefix} {label.ljust(11)} {value}"
    logger.info(formatted, extra={"style_hint": LogStyles.METRIC_VALUE})


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

        # Process each line independently to preserve line breaks
        for line in lines:
            if not line.strip():
                # Preserve blank lines
                logger.info("", extra={"style_hint": LogStyles.GENERATED_TEXT})
                continue

            # Wrap only if line exceeds available width
            if len(line) <= avail_width:
                formatted = f"{indent}{line}"
                logger.info(formatted, extra={"style_hint": LogStyles.GENERATED_TEXT})
            else:
                # Wrap this line while preserving its content
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
        # No wrapping - output each line as-is
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


def log_file_path(path: Path | str, *, label: str = "", color: str = Colors.CYAN) -> None:
    """Log a file path with highlighting.

    Args:
        path: The file path to display
        label: Optional label prefix (e.g., '   HTML:')
        color: Color to use for the path
    """
    path_str = str(path)
    if label:
        # Example output: "   HTML:     /path/to/file.html"
        msg = "%s %s"
        logger.info(
            msg,
            label,
            path_str,
            extra={"style_hint": LogStyles.FILE_PATH, "style_color": color},
        )
    else:
        logger.info(
            path_str,
            extra={"style_hint": LogStyles.FILE_PATH, "style_color": color},
        )


def log_blank(count: int = 1) -> None:
    """Log blank lines for spacing (replaces logger.info("")).

    Args:
        count: Number of blank lines to emit
    """
    for _ in range(count):
        logger.info("")


def _summary_parts(res: PerformanceResult, model_short: str) -> list[str]:
    """Assemble key=value summary segments for per-run triage."""
    parts: list[str] = [
        f"model={model_short}",
        f"status={'OK' if res.success else 'FAIL'}",
    ]
    if res.success:
        issue_labels = sorted(_extract_quality_issue_labels(res.quality_issues))
        if issue_labels:
            parts.append(f"quality={'+'.join(issue_labels)}")
        else:
            parts.append("quality=clean")
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
    gen: GenerationResult | SupportsGenerationResult | None,
    *,
    prompt: str | None = None,
    context_marker: str = "Context:",
) -> None:
    if not gen:
        return
    text_val = str(getattr(gen, "text", ""))
    gen_tokens = getattr(gen, "generation_tokens", 0)
    prompt_tokens = getattr(gen, "prompt_tokens", None)
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
    if analysis.has_harness_issue:
        details = ", ".join(analysis.harness_issue_details[:2])
        log_warning_note(f"Harness issue ({analysis.harness_issue_type}): {details}")

    # Show full output in trace (truncated in summary table)
    log_generated_text(text_val, wrap=True)


def _log_verbose_success_details_mode(
    res: PerformanceResult,
    *,
    detailed: bool,
    prompt: str | None = None,
    context_marker: str = "Context:",
) -> None:
    """Emit verbose block using either compact or detailed metrics style with visual hierarchy."""
    if not res.generation:
        return

    # Generated text with emoji prefix for easy scanning
    gen_text = getattr(res.generation, "text", None) or ""

    gen_tokens = getattr(res.generation, "generation_tokens", 0)
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
    p_tokens = getattr(res.generation, "prompt_tokens", 0)
    g_tokens = getattr(res.generation, "generation_tokens", 0)
    tot_tokens = (p_tokens or 0) + (g_tokens or 0)
    gen_tps = getattr(res.generation, "generation_tps", 0.0) or 0.0
    prompt_tps = getattr(res.generation, "prompt_tps", 0.0) or 0.0

    log_metric_label("Tokens:", emoji="🔢", indent="  ")
    log_metric_tree(
        "├─",
        "Prompt:",
        f"{fmt_num(p_tokens):>8} @ {fmt_num(prompt_tps)} tok/s",
        indent="     ",
    )
    log_metric_tree(
        "├─",
        "Generated:",
        f"{fmt_num(g_tokens):>8} @ {fmt_num(gen_tps)} tok/s",
        indent="     ",
    )
    log_metric_tree(
        "└─",
        "Total:",
        f"{fmt_num(tot_tokens):>8}",
        indent="     ",
    )


def _log_detailed_timings(res: PerformanceResult) -> None:
    """Log total, generation, and model load times with tree structure."""
    total_time_val = getattr(res, "total_time", None)
    generation_time_val = getattr(res, "generation_time", None)
    model_load_time_val = getattr(res, "model_load_time", None)

    if not total_time_val or total_time_val <= 0:
        return

    log_metric_label("Timing:", emoji="⏱", indent="  ")

    tt_val = format_field_value("total_time", total_time_val)
    tt_disp = tt_val if isinstance(tt_val, str) else _format_time_seconds(total_time_val)
    log_metric_tree("├─", "Total:", f"{tt_disp:>8}", indent="     ")

    if generation_time_val and generation_time_val > 0:
        gt_val = format_field_value("generation_time", generation_time_val)
        gt_disp = gt_val if isinstance(gt_val, str) else _format_time_seconds(generation_time_val)
        pct = (generation_time_val / total_time_val * 100) if total_time_val > 0 else 0
        log_metric_tree(
            "├─",
            "Generation:",
            f"{gt_disp:>8} ({pct:>3.0f}%)",
            indent="     ",
        )

    if model_load_time_val and model_load_time_val > 0:
        ml_val = format_field_value("model_load_time", model_load_time_val)
        ml_disp = ml_val if isinstance(ml_val, str) else _format_time_seconds(model_load_time_val)
        pct = (model_load_time_val / total_time_val * 100) if total_time_val > 0 else 0
        log_metric_tree(
            "└─",
            "Load:",
            f"{ml_disp:>8} ({pct:>3.0f}%)",
            indent="     ",
        )


def _log_perf_block(res: PerformanceResult) -> None:
    """Log inner performance metrics (memory) with tree structure and emoji."""
    active_mem = getattr(res.generation, "active_memory", 0.0) or 0.0
    cached_mem = getattr(res.generation, "cached_memory", 0.0) or 0.0
    peak_mem = getattr(res.generation, "peak_memory", 0.0) or 0.0

    # Only show memory section if at least one value is present
    if active_mem <= 0 and cached_mem <= 0 and peak_mem <= 0:
        return

    log_metric_label("Memory:", emoji="💾", indent="  ")

    def _log_mem(prefix: str, label: str, field: str, raw_val: float) -> None:
        if raw_val <= 0:
            return
        formatted = format_field_value(field, raw_val)
        unit = "GB"
        text = str(formatted) if str(formatted).endswith(unit) else f"{formatted} GB"
        log_metric_tree(prefix, label, f"{text:>8}", indent="     ")

    _log_mem("├─", "Active Δ:", "active_memory", active_mem)
    _log_mem("├─", "Cache Δ:", "cached_memory", cached_mem)
    _log_mem("└─", "Peak:", "peak_memory", peak_mem)


def _log_output_analysis(
    gen_text: str,
    gen_tokens: int,
    generation_time: float,
    peak_mem: float,
) -> None:
    """Log output analysis section: vocabulary, efficiency, structure, confidence."""
    log_metric_label("Output Analysis:", emoji="🔍", indent="  ")

    # Vocabulary diversity
    ttr, unique_words, total_words = compute_vocabulary_diversity(gen_text)
    log_metric_tree(
        "├─",
        "Vocabulary:",
        f"TTR={ttr:.2f} ({unique_words}/{total_words} unique words)",
        indent="     ",
    )

    # Efficiency metrics
    efficiency = compute_efficiency_metrics(gen_tokens, generation_time, peak_mem)
    if efficiency["tokens_per_second_per_gb"]:
        log_metric_tree(
            "├─",
            "Efficiency:",
            f"{efficiency['tokens_per_second_per_gb']:.1f} tok/s/GB",
            indent="     ",
        )

    # Response structure
    structure = detect_response_structure(gen_text)
    structure_parts = []
    if structure["has_caption"]:
        structure_parts.append("caption")
    if structure["has_keywords"]:
        structure_parts.append("keywords")
    if structure["has_description"]:
        structure_parts.append("description")
    if structure["has_sections"]:
        structure_parts.append("sections")

    structure_str = ", ".join(structure_parts) if structure_parts else "unstructured"
    log_metric_tree(
        "├─",
        "Structure:",
        structure_str,
        indent="     ",
    )

    # Confidence indicators
    confidence = compute_confidence_indicators(gen_text)
    conf_ratio = confidence["confidence_ratio"]
    if conf_ratio > QUALITY.high_confidence_threshold:
        conf_label = "high"
    elif conf_ratio > QUALITY.medium_confidence_threshold:
        conf_label = "medium"
    else:
        conf_label = "low"
    log_metric_tree(
        "└─",
        "Confidence:",
        f"{conf_label} ({conf_ratio:.0%})",
        indent="     ",
    )


def _get_grade_display(grade: str) -> str:
    """Return emoji-decorated grade display string."""
    emoji = GRADE_EMOJIS.get(grade, "❌")
    return f"{emoji} {grade}"


def _log_cataloging_utility(gen_text: str, context: str | None) -> None:
    """Log cataloging utility metrics section."""
    log_metric_label("Cataloging Utility:", emoji="📚", indent="  ")

    # Information gain
    info_gain = compute_information_gain(gen_text, context)
    echo_ratio = info_gain["echo_ratio"]
    log_metric_tree(
        "├─",
        "Info Gain:",
        f"{info_gain['information_gain']:.0%} novel "
        f"({info_gain['novel_words']}/{info_gain['output_words']} words)",
        indent="     ",
    )
    if echo_ratio > QUALITY.moderate_echo_threshold:
        log_metric_tree(
            "│ ",
            "",
            f"⚠️  {echo_ratio:.0%} echoed from context",
            indent="     ",
        )

    # Task compliance
    compliance = compute_task_compliance(gen_text)
    compliance_parts = [
        "✓ caption" if compliance["has_caption"] else "✗ caption",
        "✓ desc" if compliance["has_description"] else "✗ desc",
        "✓ keywords" if compliance["has_keywords"] else "✗ keywords",
    ]
    log_metric_tree(
        "├─",
        "Compliance:",
        f"{', '.join(compliance_parts)} ({compliance['compliance_score']:.0%})",
        indent="     ",
    )

    # Visual grounding
    grounding = compute_visual_grounding(gen_text, context)
    grounding_detail = (
        f"{grounding['visual_terms']} visual, "
        f"{grounding['spatial_terms']} spatial, "
        f"{grounding['color_terms']} color"
    )
    log_metric_tree(
        "├─",
        "Grounding:",
        f"{grounding['grounding_score']:.0%} ({grounding_detail})",
        indent="     ",
    )

    # Overall utility
    utility = compute_cataloging_utility(
        gen_text,
        context,
        info_gain=info_gain,
        task_compliance=compliance,
        visual_grounding=grounding,
    )
    grade = str(utility["utility_grade"])
    grade_display = _get_grade_display(grade)
    log_metric_tree(
        "└─",
        "UTILITY:",
        f"{grade_display} ({utility['utility_score']:.0f}/100) - {utility['primary_weakness']}",
        indent="     ",
    )


def _extract_context_from_prompt(prompt: str | None) -> str | None:
    """Extract the context section from a prompt string."""
    if not prompt:
        return None
    context_match = re.search(r"Context:\s*(.+?)(?:\n\n|$)", prompt, re.DOTALL)
    return context_match.group(1) if context_match else None


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
    context = _extract_context_from_prompt(prompt)
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
    peak_mem = getattr(gen, "peak_memory", None) or 0.0
    prompt_tokens = getattr(gen, "prompt_tokens", 0) or 0
    gen_tokens = getattr(gen, "generation_tokens", 0) or 0
    gen_tps = getattr(gen, "generation_tps", 0.0) or 0.0
    prompt_tps = getattr(gen, "prompt_tps", 0.0) or 0.0

    # Line 1: Timing and Memory
    timing_parts: list[str] = []
    if total_time is not None:
        sub_parts: list[str] = []
        if gen_time is not None:
            sub_parts.append(f"gen={_format_time_seconds(gen_time)}")
        if load_time is not None:
            sub_parts.append(f"load={_format_time_seconds(load_time)}")
        breakdown = f" ({', '.join(sub_parts)})" if sub_parts else ""
        timing_parts.append(f"{_format_time_seconds(total_time)} total{breakdown}")

    mem_part = ""
    if peak_mem > 0:
        mem_fmt = format_field_value("peak_memory", peak_mem)
        mem_str = f"{mem_fmt}GB" if not str(mem_fmt).endswith("GB") else str(mem_fmt)
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
    width = get_terminal_width(max_width=100)
    top_line = f"╔{'═' * (width - 2)}╗"
    header_line = f"║ 📖 METRICS LEGEND{' ' * (width - 20)}║"
    bottom_line = f"╚{'═' * (width - 2)}╝"

    logger.info(
        "%s",
        top_line,
        extra={"style_hint": LogStyles.RULE},
    )
    logger.info(
        "%s",
        header_line,
        extra={"style_hint": LogStyles.HEADER},
    )
    logger.info(
        "%s",
        bottom_line,
        extra={"style_hint": LogStyles.RULE},
    )
    if detailed:
        logger.info(
            "  • Detailed mode: separate lines for timing, memory, tokens, TPS",
            extra={"style_hint": LogStyles.DETAIL},
        )
    else:
        logger.info(
            "  • Compact mode: tokens(total/prompt/gen) format with aligned keys",
            extra={"style_hint": LogStyles.DETAIL},
        )
    logger.info(
        "  • ⚠️  warnings shown for repetitive or hallucinated output",
        extra={"style_hint": LogStyles.DETAIL},
    )
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
        _preview_generation(result.generation, prompt=prompt, context_marker=context_marker)
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
            prompt=prompt,
            context_marker=context_marker,
        )


def print_cli_separator() -> None:
    """Print a visually distinct separator line using unicode box-drawing characters."""
    width = get_terminal_width(max_width=100)
    log_rule(width, char="─", color=Colors.BLUE, bold=False)


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
                dists = sorted(importlib.metadata.distributions(), key=lambda d: d.name.lower())
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
    # Set DEBUG if verbose, else INFO
    console_log_level: int = logging.DEBUG if args.verbose else logging.INFO
    # Remove all handlers and add console + file handlers
    logger.handlers.clear()

    # Console handler with colored output
    console_handler: logging.StreamHandler[Any] = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(console_log_level)
    # Include timestamp for better traceability, level in verbose mode
    if args.verbose:
        fmt = "%(asctime)s - %(levelname)s - %(message)s"
    else:
        fmt = "%(asctime)s - %(message)s"
    console_formatter: ColoredFormatter = ColoredFormatter(fmt)
    console_handler.setFormatter(console_formatter)
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

    # Apply CLI output preferences (color + width)
    _apply_cli_output_preferences(args)

    # Warn if TensorFlow or sentence-transformers are present
    tf_present = bool(importlib_util.find_spec("tensorflow"))
    st_present = bool(importlib_util.find_spec("sentence_transformers"))
    guard_on = os.getenv("MLX_VLM_ALLOW_TF", "0") != "1"
    if guard_on and tf_present:
        logger.info(
            "TensorFlow detected but disabled (set MLX_VLM_ALLOW_TF=1 to opt in)",
        )
    if st_present:
        logger.warning(
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
    setattr(args, _PREFLIGHT_ISSUES_ARG_ATTR, tuple(str(issue) for issue in issues))


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
        "Analyze this image for cataloguing metadata.",
        "",
        "Return exactly these three sections:",
        "",
        "Title: 6-12 words, descriptive and concrete.",
        "",
        "Description: 1-2 factual sentences covering key subjects, setting, and action.",
        "",
        "Keywords: 15-30 comma-separated terms, ordered most specific to most general.",
        "Use concise, image-grounded wording and avoid speculation.",
    ]

    # --- Context block (uses the "Context:" marker for quality analysis) ---
    desc = metadata.get("description")
    title = metadata.get("title")
    existing_kw = metadata.get("keywords")
    has_context = desc or title or existing_kw
    if has_context:
        parts.append("")
        parts.append("Context: Existing metadata hints (use only if visually consistent):")
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

    parts.append("")
    parts.append(
        "Prioritize what is visibly present. If context conflicts with the image, trust the image.",
    )

    return "\n".join(parts)


def prepare_prompt(args: argparse.Namespace, metadata: MetadataDict) -> str:
    """Prepare the prompt for the VLM, using user input or generating from metadata."""
    print_cli_section("Prompt Configuration")

    prompt: str
    if args.prompt:
        prompt = args.prompt
        logger.info("Using user-provided prompt.")
    else:
        logger.info("Generating default prompt based on image metadata.")
        prompt = _build_cataloguing_prompt(metadata)
        logger.debug("Using generated prompt based on metadata.")

    # Truncate long prompts for display
    max_display_len = 200
    if len(prompt) > max_display_len:
        prompt_display = prompt[:max_display_len] + "..."
        logger.info("Final prompt: %s", prompt_display)
    else:
        logger.info("Final prompt: %s", prompt)
    logger.info("Prompt length: %d characters", len(prompt))
    return prompt


def get_cached_model_ids() -> list[str]:
    """Return a list of model IDs found in the Hugging Face cache."""
    try:
        cache_info: HFCacheInfo = scan_cache_dir()
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
    """Validate model selection and warn about ineffective exclusions.

    Args:
        args: Parsed command line arguments

    """
    if not args.exclude:
        return  # No exclusions to validate

    # Get available models for validation
    available_models: set[str] = set()
    if args.models:
        # When explicit models are specified, available = explicit models
        available_models = set(args.models)
        context_msg: str = "explicitly specified models"
    else:
        # When no models specified, available = cached models
        available_models = set(get_cached_model_ids())
        context_msg = "locally cached models"

    # Check for ineffective exclusions (models to exclude that aren't available)
    excluded_models: set[str] = set(args.exclude)
    ineffective_exclusions: set[str] = excluded_models - available_models

    if ineffective_exclusions:
        ineffective_list = sorted(ineffective_exclusions)
        logger.warning(
            "The following excluded models are not in the %s and will have no effect: %s",
            context_msg,
            ", ".join(ineffective_list),
        )
        if args.verbose:
            effective_exclusions = excluded_models & available_models
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
        progress = f"[{idx}/{len(model_identifiers)}]"
        # Compact logging for model header
        log_model_name(model_label, label=f"Processing Model {progress}:")

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
            repetition_penalty=args.repetition_penalty,
            repetition_context_size=args.repetition_context_size,
            lazy=args.lazy_load,
            max_kv_size=args.max_kv_size,
            kv_bits=args.kv_bits,
            kv_group_size=args.kv_group_size,
            quantized_kv_start=args.quantized_kv_start,
            revision=args.revision,
            adapter_path=args.adapter_path,
            prefill_step_size=args.prefill_step_size,
            context_marker=args.context_marker,
        )
        result: PerformanceResult = process_image_with_model(params)

        # Calculate quality score for successful generations
        if result.success and result.generation:
            gen_text = str(getattr(result.generation, "text", ""))
            gen_tokens = getattr(result.generation, "generation_tokens", 0)
            prompt_tokens = getattr(result.generation, "prompt_tokens", None)

            # Perform quality analysis for all successful runs, including empty output.
            analysis = analyze_generation_text(
                gen_text,
                gen_tokens,
                prompt_tokens=prompt_tokens,
                prompt=prompt,
                context_marker=args.context_marker,
            )
            # Log quality analysis results at DEBUG level
            logger.debug(
                "Quality analysis for %s: %s",
                result.model_name,
                _format_quality_analysis_for_log(analysis),
            )
            # Build consolidated quality issues string using helper
            quality_issues_str = _build_quality_issues_string(analysis)
            if quality_issues_str:
                logger.info(
                    "Quality issues detected for %s: %s",
                    result.model_name,
                    quality_issues_str,
                )

            # Update result with quality metrics
            result = dataclasses.replace(
                result,
                quality_issues=quality_issues_str,
                quality_analysis=analysis,
            )

        results.append(result)

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
    """Format quality analysis for structured logging.

    Args:
        analysis: GenerationQualityAnalysis with detected metrics

    Returns:
        Formatted string with all quality metrics for logging

    Examples:
        >>> analysis = GenerationQualityAnalysis(
        ...     is_repetitive=True, repeated_token="<s>",
        ...     is_verbose=True, ...
        ... )
        >>> _format_quality_analysis_for_log(analysis)
        'repetitive=True(token=<s>), verbose=True, generic=False, ...'
    """
    parts = []
    if analysis.is_repetitive:
        token_info = f" (token={analysis.repeated_token})" if analysis.repeated_token else ""
        parts.append(f"repetitive=True{token_info}")
    if analysis.is_refusal:
        refusal_info = f" (type={analysis.refusal_type})" if analysis.refusal_type else ""
        parts.append(f"refusal=True{refusal_info}")
    if analysis.has_language_mixing:
        parts.append("language_mixing=True")
    if analysis.hallucination_issues:
        parts.append("hallucination=True")
    if analysis.is_generic:
        parts.append(f"generic=True (score={analysis.specificity_score:.1f})")
    if analysis.is_verbose:
        parts.append("verbose=True")
    if analysis.formatting_issues:
        parts.append("formatting_issues=True")
    if analysis.has_excessive_bullets:
        parts.append(f"excessive_bullets=True (count={analysis.bullet_count})")
    if analysis.is_context_ignored:
        parts.append("context_ignored=True")
    if analysis.has_degeneration:
        parts.append(f"degeneration=True ({analysis.degeneration_type})")
    if analysis.has_fabrication:
        parts.append("fabrication=True")
    if analysis.has_harness_issue:
        details = (
            ",".join(analysis.harness_issue_details[:2]) if analysis.harness_issue_details else ""
        )
        parts.append(f"harness=True ({analysis.harness_issue_type}; {details})")
    parts.append(f"words={analysis.word_count}")

    return ", ".join(parts) if parts else "no issues detected"


def _build_quality_issues_string(analysis: GenerationQualityAnalysis) -> str | None:
    """Build consolidated quality issues string from analysis results.

    Prioritizes critical issues first:
    HARNESS ISSUES (integration bugs) → refusal → repetitive → lang_mixing →
    hallucination → generic → verbose → formatting → bullets → context-ignored.

    Harness issues are prefixed with ⚠️ to clearly distinguish them from
    model quality issues. These indicate bugs in mlx-vlm or model integration
    that should be reported upstream.

    Args:
        analysis: GenerationQualityAnalysis with detected issues

    Returns:
        Comma-separated issues string or None if no issues detected

    Examples:
        >>> analysis = GenerationQualityAnalysis(
        ...     is_repetitive=True, repeated_token="<s>",
        ...     is_verbose=True, ...
        ... )
        >>> _build_quality_issues_string(analysis)
        'repetitive(<s>), verbose'
    """
    issues = []

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
    if analysis.is_refusal:
        refusal_label = f"refusal({analysis.refusal_type})" if analysis.refusal_type else "refusal"
        issues.append(refusal_label)

    if analysis.is_repetitive:
        rep_label = (
            f"repetitive({analysis.repeated_token})" if analysis.repeated_token else "repetitive"
        )
        issues.append(rep_label)

    if analysis.has_language_mixing:
        issues.append("lang_mixing")

    if analysis.hallucination_issues:
        issues.append("hallucination")

    if analysis.has_degeneration:
        issues.append("degeneration")

    if analysis.has_fabrication:
        issues.append("fabrication")

    if analysis.is_generic:
        issues.append(f"generic({analysis.specificity_score:.0f})")

    if analysis.is_verbose:
        issues.append("verbose")

    if analysis.formatting_issues:
        issues.append("formatting")

    if analysis.has_excessive_bullets:
        issues.append(f"bullets({analysis.bullet_count})")

    if analysis.is_context_ignored:
        issues.append("context-ignored")

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
    },
)


def _truncate_text_preview(text: str, *, max_chars: int) -> str:
    """Trim text previews to a fixed character budget."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


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
    """Parse quality issues string into a list of individual issues.

    Args:
        quality_issues: Comma-separated quality issues string or None

    Returns:
        List of individual quality issue strings, or empty list if None

    Examples:
        >>> _parse_quality_issues_to_list("repetitive(<s>), verbose")
        ['repetitive(<s>)', 'verbose']
        >>> _parse_quality_issues_to_list(None)
        []
    """
    if not quality_issues:
        return []
    return [issue.strip() for issue in quality_issues.split(",")]


def _truncate_quality_issues(
    quality_issues: str | None,
    max_len: int = MAX_QUALITY_ISSUES_LEN,
) -> str:
    """Truncate quality issues string for display in Markdown tables.

    Args:
        quality_issues: Comma-separated quality issues string or None
        max_len: Maximum length for the truncated string

    Returns:
        Truncated quality issues string with ellipsis if needed, or empty string if None

    Examples:
        >>> _truncate_quality_issues("repetitive(<s>), verbose, formatting", 20)
        'repetitive(<s>), ...'
        >>> _truncate_quality_issues("short", 20)
        'short'
    """
    if not quality_issues:
        return ""
    if len(quality_issues) <= max_len:
        return quality_issues
    # Try to truncate at a comma boundary
    truncated = quality_issues[:max_len]
    last_comma = truncated.rfind(",")
    if last_comma > 0:
        return truncated[:last_comma] + ", ..."
    return truncated.rstrip() + "..."


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


def _ascii_bar(value: float, *, max_value: float, width: int = SUMMARY_CHART_WIDTH) -> str:
    """Render a compact ASCII bar for a positive numeric value."""
    if max_value <= 0 or value <= 0:
        return "." * width
    filled = max(1, round((value / max_value) * width))
    filled = min(width, filled)
    return "#" * filled + "." * (width - filled)


def _log_ascii_metric_chart(
    title: str,
    entries: Sequence[tuple[str, float]],
    *,
    unit: str = "",
    digits: int = 2,
    max_rows: int = SUMMARY_CHART_MAX_ROWS,
) -> None:
    """Log a compact ASCII chart for ranked model metrics."""
    if not entries:
        return
    ranked = sorted(entries, key=lambda item: item[1], reverse=True)[:max_rows]
    max_value = max(value for _label, value in ranked)
    if max_value <= 0:
        return

    logger.info("%s", title)
    for label, value in ranked:
        bar = _ascii_bar(value, max_value=max_value)
        logger.info(
            "   %-*s |%s| %s%s",
            SUMMARY_MODEL_LABEL_MAX,
            _short_model_label(label),
            bar,
            _format_float_or_dash(value, digits=digits),
            unit,
        )


def _log_model_comparison_table_and_charts(results: list[PerformanceResult]) -> None:
    """Log tabulated per-model comparison and compact ASCII charts for this run."""
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

    headers = ["#", "Model", "Status", "TPS", "Total(s)", "Load(s)", "PeakGB", "Notes"]
    table_text = tabulate(rows, headers=headers, tablefmt="github", disable_numparse=True)
    logger.info("📋 Model Comparison (current run):")
    for line in table_text.splitlines():
        logger.info("   %s", line)

    if tps_entries:
        log_blank()
        _log_ascii_metric_chart("📊 TPS comparison chart:", tps_entries, unit=" tps", digits=1)
    if len(total_time_entries) >= MIN_MODELS_FOR_EFFICIENCY_CHART:
        inverted = [(name, 1.0 / value) for name, value in total_time_entries if value > 0]
        if inverted:
            _log_ascii_metric_chart(
                "⏱ Efficiency chart (higher is faster overall):",
                inverted,
                unit=" 1/s",
                digits=3,
            )

    failed = [res for res in sorted_results if not res.success]
    if failed:
        stage_counts = Counter(res.error_stage or "Unknown" for res in failed)
        _log_ascii_metric_chart(
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
    context = _extract_context_from_prompt(prompt)
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
        score, grade, weakness, delta = _compute_utility_snapshot(
            text,
            context,
            baseline_score=baseline_score,
        )
        rows.append(
            UtilityTriageRow(
                result=res,
                score=score,
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
    log_rule(color=Colors.BLUE, bold=True)
    logger.info("Results Summary", extra={"style_hint": LogStyles.HEADER})
    log_rule(color=Colors.BLUE, bold=True)

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
    max_records: int = 100,
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


def compare_history_records(
    previous: HistoryRunRecord | None,
    current: HistoryRunRecord,
) -> dict[str, list[str]]:
    """Compare two history records and return regressions/recoveries."""
    prev_models = previous.get("model_results", {}) if previous else {}
    curr_models = current.get("model_results", {}) if current else {}
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

    return {
        "regressions": regressions,
        "recoveries": recoveries,
        "new_models": new_models,
        "missing_models": missing_models,
    }


def _build_prompt_preview(prompt: str, *, max_chars: int = 200) -> str:
    """Return prompt preview truncated to ``max_chars``."""
    return prompt if len(prompt) <= max_chars else f"{prompt[:max_chars]}..."


def _history_model_result_from_result(result: PerformanceResult) -> HistoryModelResultRecord:
    """Build a typed per-model history row from runtime result."""
    return {
        "success": result.success,
        "failure_phase": result.failure_phase,
        "error_stage": result.error_stage,
        "error_type": result.error_type,
        "error_package": result.error_package,
        "error_code": result.error_code,
        "error_signature": result.error_signature,
    }


def _build_history_run_record(
    *,
    results: list[PerformanceResult],
    prompt: str,
    system_info: dict[str, str],
    library_versions: LibraryVersionDict,
    image_path: Path | None,
) -> HistoryRunRecord:
    """Build typed run-history record payload prior to append."""
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    model_results = {
        result.model_name: _history_model_result_from_result(result) for result in results
    }
    return {
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


def append_history_record(
    *,
    history_path: Path,
    results: list[PerformanceResult],
    prompt: str,
    system_info: dict[str, str],
    library_versions: LibraryVersionDict,
    image_path: Path | None = None,
) -> HistoryRunRecord:
    """Append a per-run history record for tracking regressions/recoveries."""
    record = _build_history_run_record(
        results=results,
        prompt=prompt,
        system_info=system_info,
        library_versions=library_versions,
        image_path=image_path,
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
) -> JsonlMetadataRecord:
    """Build shared metadata header row for JSONL results."""
    return {
        "_type": "metadata",
        "format_version": "1.2",
        "prompt": prompt,
        "system": system_info,
        "timestamp": local_now_str(),
    }


def _build_jsonl_result_record_base(result: PerformanceResult) -> JsonlResultRecord:
    """Build base per-model JSONL row with failure-safe defaults."""
    return {
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
        },
    }


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


def save_jsonl_report(
    results: list[PerformanceResult],
    filename: Path,
    prompt: str,
    system_info: dict[str, str],
) -> None:
    """Save results to a JSONL file for programmatic analysis and AI issue generation.

    The JSONL format includes all diagnostic information needed to generate
    actionable GitHub issue reports, including:
    - Full error tracebacks for debugging
    - Error type classification for bucketing
    - Package attribution for directing reports
    - Quality analysis for successful models
    - Timing metrics for performance analysis

    Format (v1.2): First line is a metadata header containing prompt and
    system_info (shared across all rows). Per-model result lines follow.
    """
    try:
        with filename.open("w", encoding="utf-8") as f:
            # Write shared metadata header (avoids repeating prompt/system per row)
            header = _build_jsonl_metadata_record(prompt=prompt, system_info=system_info)
            f.write(json.dumps(header) + "\n")

            for result in results:
                record = _build_jsonl_result_record_base(result)
                _populate_jsonl_result_generation_data(record, result)
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
) -> dict[str, list[str]]:
    """Return transition summary and log context-change warnings when needed."""
    if previous is None:
        return {
            "regressions": [],
            "recoveries": [],
            "new_models": [],
            "missing_models": [],
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
    summary: dict[str, list[str]],
) -> list[list[str]]:
    """Build run-over-run summary rows for history comparison output."""
    regressions = summary["regressions"]
    recoveries = summary["recoveries"]
    new_models = summary["new_models"]
    missing_models = summary["missing_models"]

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
        ["New models", "-", str(len(new_models)), "-"],
        ["Missing models", "-", str(len(missing_models)), "-"],
    ]


def _history_context_rows(
    previous: HistoryRunRecord | None,
    current: HistoryRunRecord,
) -> list[list[str]]:
    """Build compact context rows (prompt/image/version) for history comparisons."""
    previous_image = (
        _truncate_text_preview(str(previous.get("image_path") or "-"), max_chars=40)
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
            _truncate_text_preview(str(current.get("image_path") or "-"), max_chars=40),
        ],
        [
            "Library versions",
            _format_library_versions_for_history(previous),
            _format_library_versions_for_history(current),
        ],
    ]


def _log_history_transition_chart(
    *,
    summary: dict[str, list[str]],
    current: HistoryRunRecord,
) -> None:
    """Log transition chart (or current status chart when no transitions)."""
    regressions = summary["regressions"]
    recoveries = summary["recoveries"]
    new_models = summary["new_models"]
    missing_models = summary["missing_models"]
    transition_entries = [
        ("regressions", float(len(regressions))),
        ("recoveries", float(len(recoveries))),
        ("new", float(len(new_models))),
        ("missing", float(len(missing_models))),
    ]
    if any(value > 0 for _label, value in transition_entries):
        _log_ascii_metric_chart(
            "🔁 Status transition counts:",
            transition_entries,
            unit=" x",
            digits=0,
            max_rows=4,
        )
        return

    _total, curr_success, curr_failed, _success_rate = _history_counts(current)
    _log_ascii_metric_chart(
        "✅ Current run status counts:",
        [("success", float(curr_success)), ("failed", float(curr_failed))],
        unit=" x",
        digits=0,
        max_rows=2,
    )


def _history_transition_rows(
    *,
    previous: HistoryRunRecord | None,
    current: HistoryRunRecord,
    summary: dict[str, list[str]],
) -> list[list[str]]:
    """Build per-model transition rows for regressions/recoveries/new/missing models."""
    regressions = summary["regressions"]
    recoveries = summary["recoveries"]
    new_models = summary["new_models"]
    missing_models = summary["missing_models"]
    changed_models = sorted({*regressions, *recoveries, *new_models, *missing_models})
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
        elif model_name in new_models:
            transition = "New"
        elif model_name in missing_models:
            transition = "Missing"
        else:
            transition = "Changed"

        prev_info = prev_models.get(model_name)
        curr_info = curr_models.get(model_name)
        stage_source = curr_info if curr_info is not None else prev_info
        stage_text = stage_source.get("error_stage") if stage_source is not None else None
        transition_rows.append(
            [
                _short_model_label(model_name),
                transition,
                _model_status_from_history(prev_info),
                _model_status_from_history(curr_info),
                stage_text or "-",
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
    summary_table = tabulate(
        rows,
        headers=["Metric", "Previous", "Current", "Delta"],
        tablefmt="github",
        disable_numparse=True,
    )
    logger.info("📚 Run-over-run comparison:")
    for line in summary_table.splitlines():
        logger.info("   %s", line)

    context_rows = _history_context_rows(previous, current)
    context_table = tabulate(
        context_rows,
        headers=["Context", "Previous", "Current"],
        tablefmt="github",
        disable_numparse=True,
    )
    logger.info("🔎 Comparison context:")
    for line in context_table.splitlines():
        logger.info("   %s", line)

    _log_history_transition_chart(summary=summary, current=current)
    transition_rows = _history_transition_rows(previous=previous, current=current, summary=summary)
    if transition_rows:
        transitions_table = tabulate(
            transition_rows,
            headers=["Model", "Transition", "Prev", "Current", "Error Stage"],
            tablefmt="github",
            disable_numparse=True,
        )
        logger.info("🧾 Detailed model transitions:")
        for line in transitions_table.splitlines():
            logger.info("   %s", line)


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
) -> None:
    """Export repro bundles and diagnostics markdown after history append."""
    repro_bundles = export_failure_repro_bundles(
        results=results,
        output_dir=diagnostics_path.parent / "repro_bundles",
        run_args=args,
        versions=library_versions,
        system_info=system_info,
        prompt=prompt,
        image_path=image_path,
    )
    if repro_bundles:
        logger.info("Repro bundles written for %d failed model(s).", len(repro_bundles))
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
            preflight_issues=_get_run_preflight_issues(args),
        ),
        repro_bundles=repro_bundles,
    )
    if diagnostics_written:
        log_file_path(diagnostics_path, label="   Diagnostics:  ")


def finalize_execution(
    *,
    args: argparse.Namespace,
    results: list[PerformanceResult],
    library_versions: LibraryVersionDict,
    overall_start_time: float,
    prompt: str,
    image_path: Path | None = None,
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

        # Prepare output paths
        html_output_path: Path = args.output_html.resolve()
        md_output_path: Path = args.output_markdown.resolve()
        tsv_output_path: Path = args.output_tsv.resolve()
        jsonl_output_path: Path = args.output_jsonl.resolve()
        diagnostics_path: Path = args.output_diagnostics.resolve()
        log_output_path: Path = args.output_log.resolve()
        env_output_path: Path = args.output_env.resolve()
        history_path = _history_path_for_jsonl(jsonl_output_path)
        previous_history = _load_latest_history_record(history_path)

        html_output_path.parent.mkdir(parents=True, exist_ok=True)
        md_output_path.parent.mkdir(parents=True, exist_ok=True)
        tsv_output_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_output_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate reports
        try:
            try:
                generate_html_report(
                    results=results,
                    filename=args.output_html,
                    versions=library_versions,
                    prompt=prompt,
                    total_runtime_seconds=overall_time,
                    image_path=image_path,
                )
            except (OSError, ValueError) as err:
                logger.exception("Failed to generate HTML report.")
                _write_report_failure_jsonl(
                    filename=jsonl_output_path,
                    failed_report="html",
                    error=err,
                )

            try:
                generate_markdown_report(
                    results=results,
                    filename=args.output_markdown,
                    versions=library_versions,
                    prompt=prompt,
                    total_runtime_seconds=overall_time,
                )
            except (OSError, ValueError) as err:
                logger.exception("Failed to generate Markdown report.")
                _write_report_failure_jsonl(
                    filename=jsonl_output_path,
                    failed_report="markdown",
                    error=err,
                )
            generate_tsv_report(
                results=results,
                filename=args.output_tsv,
            )
            # New: Save JSONL report
            save_jsonl_report(
                results,
                args.output_jsonl,
                prompt=prompt,
                system_info=system_info,
            )

            # Log file locations
            logger.info("")
            log_success("Reports successfully generated:", prefix="📊")
            log_file_path(args.output_html, label="   HTML Report:")
            log_file_path(
                args.output_markdown,
                label="   Markdown Report:",
            )
            log_file_path(args.output_tsv, label="   TSV Report:   ")
            log_file_path(args.output_jsonl, label="   JSONL Report: ")

            log_file_path(log_output_path, label="   Log File:")
            # Include environment.log in the output file listing
            if env_output_path.exists():
                log_file_path(env_output_path, label="   Environment:")
        except (OSError, ValueError):
            logger.exception("Failed to generate reports.")

        # Append run history (append-only) and compare with previous run
        current_history = append_history_record(
            history_path=history_path,
            results=results,
            prompt=prompt,
            system_info=system_info,
            library_versions=library_versions,
            image_path=image_path,
        )

        log_file_path(history_path, label="   History:     ")

        _log_history_comparison(previous_history, current_history)

        # Generate diagnostics report after history append so regression/retry
        # context in diagnostics.md reflects this run.
        _write_diagnostics_and_repro_artifacts(
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
    try:
        library_versions = setup_environment(args)
        # Validate all CLI arguments early to fail fast (after logging setup)
        validate_cli_arguments(args)
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

        results = process_models(args, image_path, prompt=prompt)

        finalize_execution(
            args=args,
            results=results,
            library_versions=library_versions,
            overall_start_time=overall_start_time,
            prompt=prompt,
            image_path=image_path,
        )
    except KeyboardInterrupt:
        logger.warning("Execution interrupted by user.")
        sys.exit(130)
    except SystemExit:
        raise
    except (OSError, ValueError, RuntimeError) as main_err:
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
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="MLX VLM Model Checker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
            "Folder to scan. The most recently modified image file in the folder will be used. "
            "If neither --folder nor --image is specified, the default folder will be used."
        ),
    )
    group.add_argument(
        "-i",
        "--image",
        type=Path,
        default=None,
        help="Path to a specific image file to process directly.",
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
        nargs="+",
        type=str,
        default=None,
        help="Specify models by ID/path. Overrides cache scan.",
    )
    parser.add_argument(
        "-e",
        "--exclude",
        nargs="+",
        type=str,
        default=None,
        help="Exclude models by ID/path from the model list "
        "(works with both explicit --models and cache scan).",
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
        help="Prompt.",
    )

    parser.add_argument(
        "-d",
        "--detailed-metrics",
        action="store_true",
        default=False,
        help="Show expanded multi-line metrics block (verbose mode only).",
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
        type=int,
        default=None,
        choices=[4, 8],
        help="Quantize KV cache to N bits (4 or 8). Saves memory with small quality trade-off.",
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
        default=0,
        help="Start position for KV cache quantization. 0 = from beginning.",
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=None,
        help="Step size for prompt prefill. None = use model default.",
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
        "-n",
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Validate arguments and show what would be run without invoking any models. "
            "Lists discovered models, the generated prompt, and image path then exits."
        ),
    )

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


if __name__ == "__main__":
    main_cli()
