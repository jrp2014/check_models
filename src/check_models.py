#!/usr/bin/env python3
"""Image analysis and caption generation using MLX Vision Language Models."""

from __future__ import annotations

import argparse
import base64
import contextlib
import dataclasses
import gc
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
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Final,
    NoReturn,
    Protocol,
    Self,
    cast,
    runtime_checkable,
)

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
    from transformers.tokenization_python import PythonBackend
    from transformers.tokenization_utils_tokenizers import TokenizersBackend

# Public API (PEP 8 / PEP 561 best practice)
__all__ = [
    "GenerationQualityAnalysis",
    "PerformanceResult",
    "ProcessImageParams",
    "ResultSet",
    "analyze_generation_text",
    "exit_with_cli_error",
    "extract_image_metadata",
    "find_most_recent_file",
    "format_field_label",
    "format_field_value",
    "format_overall_runtime",
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
ERROR_PILLOW_MISSING: Final[str] = (
    "Error: Pillow not found. Please install it (`pip install Pillow`)."
)
ERROR_MLX_VLM_MISSING: Final[str] = (
    "Error: mlx-vlm not found. Please install it (`pip install mlx-vlm`)."
)


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

    # Bullet point detection
    max_bullets: int = 10

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

try:
    import mlx.core as mx
except ImportError:
    mx = cast("Any", None)
    MISSING_DEPENDENCIES["mlx"] = ERROR_MLX_MISSING

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
    from PIL.ExifTags import GPS as PIL_GPS
    from PIL.ExifTags import GPSTAGS as PIL_GPSTAGS
    from PIL.ExifTags import TAGS as PIL_TAGS

    pillow_version = Image.__version__ if hasattr(Image, "__version__") else NOT_AVAILABLE
    ExifTags = PIL_ExifTags
    GPS = PIL_GPS
    GPSTAGS = PIL_GPSTAGS
    TAGS = PIL_TAGS

try:
    import numpy as np

    numpy_version: str = getattr(np, "__version__", NOT_AVAILABLE)
except ImportError:
    numpy_version = NOT_AVAILABLE

vlm_version: str

try:
    from mlx_vlm.generate import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load, load_image
    from mlx_vlm.version import __version__ as _mlx_vlm_version

    vlm_version = _mlx_vlm_version
except ImportError:
    vlm_version = NOT_AVAILABLE

    @dataclass
    class _GenerationResultFallback:
        """Fallback structure used when mlx-vlm is unavailable."""

        text: str | None = None
        prompt_tokens: int | None = None
        generation_tokens: int | None = None

    def _raise_mlx_vlm_missing(*_args: object, **_kwargs: object) -> NoReturn:
        """Raise a consistent runtime error when mlx-vlm is unavailable."""
        raise RuntimeError(ERROR_MLX_VLM_MISSING)

    # Use Any for fallback functions to avoid type conflicts with stub signatures
    generate = cast("Any", _raise_mlx_vlm_missing)
    apply_chat_template = cast("Any", _raise_mlx_vlm_missing)
    load = cast("Any", _raise_mlx_vlm_missing)
    load_image = cast("Any", _raise_mlx_vlm_missing)

    MISSING_DEPENDENCIES["mlx-vlm"] = ERROR_MLX_VLM_MISSING
try:
    import mlx_lm

    mlx_lm_version: str = getattr(mlx_lm, "__version__", NOT_AVAILABLE)
except ImportError:
    mlx_lm_version = NOT_AVAILABLE
except AttributeError:
    mlx_lm_version = f"{NOT_AVAILABLE} (module found, no version attr)"
_transformers_guard_enabled: bool = os.getenv("MLX_VLM_ALLOW_TF", "0") != "1"
if _transformers_guard_enabled:
    # Prevent Transformers from importing heavy backends that can hang on macOS/ARM
    # when they are present in the environment but not needed for MLX workflows.
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
    os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")

try:
    import transformers

    transformers_version: str = transformers.__version__

except ImportError:
    transformers_version = NOT_AVAILABLE

try:
    import tokenizers

    tokenizers_version: str = getattr(tokenizers, "__version__", NOT_AVAILABLE)
except ImportError:
    tokenizers_version = NOT_AVAILABLE


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

DEFAULT_TEMPERATURE: Final[float] = 0.1
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
    """Encapsulates a GenerationResult and execution metadata for a model run."""

    model_name: str
    generation: GenerationResult | SupportsGenerationResult | None
    success: bool
    error_stage: str | None = None
    error_message: str | None = None
    captured_output_on_fail: str | None = None
    generation_time: float | None = None  # Time taken for generation in seconds
    model_load_time: float | None = None  # Time taken to load the model in seconds
    total_time: float | None = None  # Total time including model loading
    # MOD: Added error_type to preserve original exception type for better error bucketing
    error_type: str | None = None  # Original exception type name (e.g., "TypeError", "ImportError")
    # MOD: Consolidated quality analysis into single field
    quality_issues: str | None = None  # Detected issues (e.g., "repetitive, verbose")
    # MOD: Added detailed memory profiling
    active_memory: float | None = None  # Active GPU memory in GB
    cache_memory: float | None = None  # Cached GPU memory in GB
    # MOD: Added package attribution to identify which package caused the error
    error_package: str | None = None  # Package that caused error (mlx, mlx-vlm, transformers, etc.)
    # MOD: Added full traceback for actionable GitHub issue reports
    error_traceback: str | None = None  # Full traceback for error diagnosis


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
    _ansi_escape_re: ClassVar[re.Pattern[str]] = re.compile(
        r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])",
    )

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
                lambda _, m: "Library Versions" in m
                or (
                    m.count(":") == 1
                    and any(lib in m.lower() for lib in ["mlx", "pillow", "transformers"])
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
MAX_TUPLE_LEN: Final[int] = 10
MAX_STR_LEN: Final[int] = 60
STR_TRUNCATE_LEN: Final[int] = 57
BASE_NAME_MAX_WIDTH: Final[int] = 45
COL_WIDTH: Final[int] = 12
MIN_NAME_COL_WIDTH: Final[int] = len("Model")

# Field display configuration: maps field names to (label, unit) tuples
# This centralizes all field metadata in one place for consistency
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
    "quality_issues": ("Quality", "Issues"),  # MOD: Consolidated quality analysis
    "active_memory": ("Active", "Mem (GB)"),  # MOD: Active GPU memory
    "cache_memory": ("Cache", "Mem (GB)"),  # MOD: Cached GPU memory
    "error_package": ("Error", "Package"),  # MOD: Package attribution
}

# Threshold for splitting long header text into multiple lines
HEADER_SPLIT_LENGTH = 10
ERROR_MESSAGE_PREVIEW_LEN: Final[int] = 40  # Max chars to show from error in summary line
ERROR_MESSAGE_TRUNCATE_LEN: Final[int] = 120  # Max chars for error messages in actionable reports
MAX_QUALITY_ISSUES_LEN: Final[int] = 30  # Max chars for quality issues in Markdown tables

# Numeric fields are automatically derived from FIELD_ABBREVIATIONS for consistency
# Exclude non-numeric fields explicitly
NUMERIC_FIELD_PATTERNS: Final[frozenset[str]] = frozenset(
    k for k in FIELD_ABBREVIATIONS if k not in {"model_name", "quality_issues"}
)

# Console table formatting constants
MAX_MODEL_NAME_LENGTH = 20  # Allows "microsoft/phi-3-vision" without truncation
MAX_OUTPUT_LENGTH = 28
MAX_OUTPUT_LINES = 3  # Max lines to show in summary table

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
    threshold = QUALITY.max_bullets if QUALITY.max_bullets else 15
    return bullet_count > threshold, bullet_count


def _detect_markdown_formatting(text: str) -> bool:
    """Detect if output contains markdown formatting elements.

    Args:
        text: Generated text to check

    Returns:
        True if markdown formatting is detected
    """
    if not text:
        return False

    # Common markdown patterns
    markdown_indicators = [
        r"^#{1,6}\s",  # Headers (# Header, ## Header, etc.)
        r"\*\*[^*]+\*\*",  # Bold (**text**)
        r"\*[^*]+\*",  # Italic (*text*)
        r"__[^_]+__",  # Bold (__text__)
        r"_[^_]+_",  # Italic (_text_)
        r"`[^`]+`",  # Inline code (`code`)
        r"```",  # Code blocks
        r"^\s*[-*+]\s",  # Unordered lists
        r"^\s*\d+\.\s",  # Ordered lists
        r"\[([^\]]+)\]\(([^)]+)\)",  # Links [text](url)
    ]

    return any(re.search(pattern, text, re.MULTILINE) for pattern in markdown_indicators)


# MOD: Added context ignorance detection
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
    """Detect end-of-output degeneration (garbage/nonsense at the end).

    LLMs sometimes fail to stop properly and produce:
    - Repeated special characters or punctuation
    - Incomplete sentences cut off mid-word
    - Unicode garbage or control characters
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

    # 3. Detect Unicode garbage/control characters (excluding normal whitespace)
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
    urls = re.findall(r"https?://[^\s<>\"']+", text)
    for url in urls:
        # Fabricated URLs often have suspicious patterns
        if any(
            suspicious in url.lower()
            for suspicious in ["example.com", "placeholder", "xxx", "fake"]
        ):
            issues.append(f"suspicious_url: {url[:50]}")
        # Very long URLs with random-looking paths
        elif len(url) > QUALITY.max_url_length and re.search(r"/[a-z0-9]{20,}/", url.lower()):
            issues.append(f"fabricated_url: {url[:50]}...")

    # 2. Detect invented precise statistics (suspiciously specific numbers)
    # e.g., "exactly 73.847%" or "precisely 14,523 items"
    precise_stats = re.findall(r"\b(\d{1,3}(?:,\d{3})*\.\d{3,})\s*%?", text)
    if len(precise_stats) >= QUALITY.min_precise_stats:
        issues.append(f"suspicious_precision: {len(precise_stats)} overly precise numbers")

    # 3. Detect future dates (model can't know the future)
    # Years 2030+ are definitely future
    future_years = re.findall(r"\b(20[3-9]\d|2[1-9]\d{2})\b", text)
    if future_years:
        issues.append(f"future_date: {', '.join(future_years[:3])}")

    # 4. Detect citations to non-existent sources (common hallucination)
    # Patterns like "according to Smith et al. (2024)" or "(Johnson, 2025)"
    fake_citations = re.findall(r"\(([A-Z][a-z]+(?:\s+et\s+al\.?)?,?\s*\d{4})\)", text)
    if fake_citations:
        issues.append(f"unverifiable_citation: {', '.join(fake_citations[:2])}")

    return bool(issues), issues


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

    # Hedge words/phrases indicating uncertainty
    hedge_patterns = [
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
    ]

    # Definitive patterns indicating confidence
    definitive_patterns = [
        r"\bis a\b",
        r"\bare \w+\b",
        r"\bshows\b",
        r"\bdepicts\b",
        r"\bfeatures\b",
        r"\bcontains\b",
        r"\bdefinitely\b",
        r"\bclearly\b",
        r"\bobviously\b",
    ]

    hedge_count = sum(len(re.findall(p, text_lower)) for p in hedge_patterns)
    definitive_count = sum(len(re.findall(p, text_lower)) for p in definitive_patterns)

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

    # Check for explicit labeled sections
    has_explicit_caption = bool(re.search(r"\b(caption|title)\s*:", text_lower))
    has_explicit_description = bool(re.search(r"\b(description|details?|summary)\s*:", text_lower))
    has_explicit_keywords = bool(re.search(r"\b(keywords?|tags?)\s*:", text_lower))

    # Check for implicit structure (bullet lists for keywords, paragraphs for description)
    has_bullet_list = bool(re.search(r"^[-•*]\s+\w+", text, re.MULTILINE))
    has_paragraph = len(text.split()) > QUALITY.substantial_prose_words

    # Combine explicit and implicit signals
    has_caption = has_explicit_caption or (
        # First line could be a caption if short and followed by more text
        len(text.split("\n")[0].split()) <= QUALITY.max_caption_words and len(text.split("\n")) > 1
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
    visual_patterns = [
        r"\b(building|house|shop|store|street|road|car|vehicle|person|people|pedestrian)\b",
        r"\b(sign|window|door|roof|wall|brick|stone|glass)\b",
        r"\b(tree|sky|cloud|hill|mountain|grass|flower)\b",
        r"\b(light|lamp|shadow|reflection|glow)\b",
        r"\b(wearing|standing|sitting|walking|driving)\b",
    ]

    # Spatial relationship terms
    spatial_patterns = [
        r"\b(left|right|center|middle|foreground|background)\b",
        r"\b(above|below|beside|behind|front|back)\b",
        r"\b(near|far|distant|close|adjacent)\b",
        r"\b(top|bottom|side|corner|edge)\b",
    ]

    # Color terms
    color_patterns = [
        r"\b(red|blue|green|yellow|orange|purple|pink|brown|black|white|gray|grey)\b",
        r"\b(golden|silver|bronze|dark|light|bright|pale|vivid)\b",
        r"\b(warm|cool|muted|saturated)\b",
    ]

    def count_matches(patterns: list[str], text: str) -> int:
        return sum(len(re.findall(p, text)) for p in patterns)

    visual_count = count_matches(visual_patterns, text_lower)
    spatial_count = count_matches(spatial_patterns, text_lower)
    color_count = count_matches(color_patterns, text_lower)

    # Penalize if visual terms are just from context (not novel observations)
    context_visual = count_matches(visual_patterns, context_lower) if context else 0
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


def _compute_echo_penalty(echo_ratio: float) -> float:
    """Compute penalty factor based on context echo ratio."""
    if echo_ratio > QUALITY.severe_echo_threshold:
        return 0.5  # Severe penalty for mostly echoing
    if echo_ratio > QUALITY.moderate_echo_threshold:
        return 0.8  # Moderate penalty
    return 1.0  # No penalty


def _compute_length_factor(word_count: int) -> tuple[float, str]:
    """Compute length factor and weakness message based on word count."""
    if word_count < QUALITY.min_useful_words:
        return 0.2, "Output too short to be useful"
    if word_count < QUALITY.short_output_words:
        return 0.6, "Output lacks detail"
    return 1.0, ""


def _identify_primary_weakness(
    echo_ratio: float,
    grounding_score: float,
    compliance_score: float,
    information_gain_score: float,
) -> str:
    """Identify the primary weakness limiting cataloging utility."""
    if echo_ratio > QUALITY.moderate_echo_threshold:
        return "Mostly echoes context without adding value"
    if grounding_score < QUALITY.low_grounding_threshold:
        return "Lacks visual description of image content"
    if compliance_score < QUALITY.low_compliance_threshold:
        return "Missing requested structure (caption/description/keywords)"
    if information_gain_score < QUALITY.low_info_gain_threshold:
        return "Limited novel information"
    return "None identified"


def _score_to_grade(score: float) -> str:
    """Convert numeric score to letter grade."""
    if score >= QUALITY.grade_a_threshold:
        return "A"
    if score >= QUALITY.grade_b_threshold:
        return "B"
    if score >= QUALITY.grade_c_threshold:
        return "C"
    if score >= QUALITY.grade_d_threshold:
        return "D"
    return "F"


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

    # Compute penalties and factors
    echo_penalty = _compute_echo_penalty(echo_ratio)
    word_count = len(text.split())
    length_factor, length_weakness = _compute_length_factor(word_count)

    # Weighted combination (out of 100)
    raw_score = (
        information_gain_score * 25  # 25 points for adding new info
        + compliance_score * 30  # 30 points for following structure
        + grounding_score * 30  # 30 points for visual descriptions
        + min(word_count / 50, 1.0) * 15  # 15 points for reasonable length
    )

    final_score = raw_score * echo_penalty * length_factor

    # Determine weakness
    weakness = length_weakness or _identify_primary_weakness(
        echo_ratio,
        grounding_score,
        compliance_score,
        information_gain_score,
    )

    return {
        "utility_score": round(final_score, 1),
        "utility_grade": _score_to_grade(final_score),
        "primary_weakness": weakness,
    }


@dataclass(frozen=True)
class GenerationQualityAnalysis:
    """Analysis results for generated text quality.

    Consolidates all quality checks (repetition, hallucination, verbosity,
    formatting) into a single structured result to avoid duplicate analysis.
    """

    is_repetitive: bool
    repeated_token: str | None
    hallucination_issues: list[str]
    is_verbose: bool
    formatting_issues: list[str]
    has_excessive_bullets: bool
    bullet_count: int
    # MOD: Added context ignorance detection
    is_context_ignored: bool
    missing_context_terms: list[str]
    # MOD: Added refusal/uncertainty detection
    is_refusal: bool
    refusal_type: str | None
    # MOD: Added generic output detection
    is_generic: bool
    specificity_score: float
    # MOD: Added language/code mixing detection
    has_language_mixing: bool
    language_mixing_issues: list[str]
    # MOD: Added output degeneration detection (garbage/nonsense)
    has_degeneration: bool
    degeneration_type: str | None
    # MOD: Added fabrication detection (hallucinated details)
    has_fabrication: bool
    fabrication_issues: list[str]

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
        return issues_list


def analyze_generation_text(
    text: str,
    generated_tokens: int,
    prompt: str | None = None,
    context_marker: str = "Context:",
) -> GenerationQualityAnalysis:
    """Analyze generated text for quality issues.

    Consolidates all quality detection logic into a single function to avoid
    duplication between preview and verbose output modes.

    Args:
        text: Generated text to analyze
        generated_tokens: Number of tokens generated
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

    # MOD: Added context ignorance detection
    is_context_ignored = False
    missing_context_terms: list[str] = []
    if prompt:
        is_context_ignored, missing_context_terms = _detect_context_ignorance(
            text,
            prompt,
            context_marker=context_marker,
        )

    # MOD: Added refusal/uncertainty detection
    is_refusal, refusal_type = _detect_refusal_patterns(text)

    # MOD: Added generic output detection
    is_generic, specificity_score = _detect_generic_output(text)

    # MOD: Added language/code mixing detection
    has_language_mixing, language_mixing_issues = _detect_language_mixing(text)

    # MOD: Added output degeneration detection (garbage at end)
    has_degeneration, degeneration_type = _detect_output_degeneration(text)

    # MOD: Added fabrication detection (hallucinated specifics)
    has_fabrication, fabrication_issues = _detect_fabricated_details(text)

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
        "mlx-vlm": _none_if_na(mlx_vlm_ver),
        "mlx-lm": _none_if_na(_get_version("mlx-lm")),
        "huggingface-hub": _none_if_na(_get_version("huggingface-hub", hf_version)),
        "transformers": _none_if_na(_get_version("transformers")),
        "tokenizers": _none_if_na(_get_version("tokenizers")),
        "Pillow": _none_if_na(_get_version("Pillow", pillow_version)),
    }


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


def _is_performance_timing_field(field_name: str) -> bool:  # kept for clarity
    """Return True if ``field_name`` is one of the timing fields we expose.

    Separated into a helper for readability and potential future extension
    (e.g. alias mapping or dynamic timing metrics).
    """
    return field_name in PERFORMANCE_TIMING_FIELDS


def _get_field_value(result: PerformanceResult, field_name: str) -> MetricValue:
    """Get field value from either GenerationResult or PerformanceResult."""
    if _is_performance_timing_field(field_name):
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
            # URL scheme validated above (http/https only)
            with urllib.request.urlopen(image_str, timeout=30) as response:  # noqa: S310
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


def to_float(val: float | str) -> float | None:
    """Convert a value to float if possible, else return None."""
    try:
        temp = float(val)
    except (TypeError, ValueError):
        return None
    else:
        return temp


# Reduce return count and use named constants
def _convert_gps_coordinate(
    coord: tuple[float | str, ...] | list[float | str],
) -> tuple[float, float, float] | None:
    """Convert GPS EXIF coordinate to (degrees, minutes, seconds) tuple.

    GPS coordinates in EXIF are stored as tuples of (degrees, minutes, seconds)
    or (degrees, decimal_minutes). This function normalizes both formats.

    Args:
        coord: Tuple or list of 2 or 3 numeric values representing GPS coordinate

    Returns:
        Tuple of (degrees, minutes, seconds) as floats, or None if conversion fails

    Examples:
        >>> # Standard DMS format (degrees, minutes, seconds)
        >>> _convert_gps_coordinate((37.0, 46.0, 30.5))
        (37.0, 46.0, 30.5)

        >>> # Decimal minutes format (degrees, decimal_minutes)
        >>> _convert_gps_coordinate((37.0, 46.508333))
        (37.0, 46.508333, 0.0)

        >>> # Invalid format returns None
        >>> _convert_gps_coordinate((37.0,))
        None

    """
    clen = len(coord)
    if clen not in (MIN_GPS_COORD_LEN, MED_GPS_COORD_LEN, MAX_GPS_COORD_LEN):
        return None

    # Convert available components to float, padding with 0.0 for missing values
    components = [to_float(coord[i]) if i < clen else 0.0 for i in range(3)]

    # Return None if any required component failed to convert
    if any(c is None for c in components[:clen]):
        return None

    # All conversions succeeded, return tuple with defaults for missing values
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


def _extract_description(exif_data: ExifDict) -> str | None:
    description = exif_data.get("ImageDescription")
    if description is None:
        return None
    if isinstance(description, bytes):
        try:
            # Try strict UTF-8 first
            desc = description.decode("utf-8").replace("\x00", "").strip()
        except UnicodeDecodeError:
            # Fallback to Latin-1 (CP1252 subset maps 1:1, usually correct for legacy EXIF)
            # This handles "copyright" symbols (\xa9) correctly where utf-8 would fail
            try:
                desc = description.decode("latin-1").replace("\x00", "").strip()
            except (UnicodeDecodeError, AttributeError) as err:
                desc = str(description)
                logger.debug("Failed to decode description with fallbacks: %s", err)
    else:
        desc = str(description).replace("\x00", "").strip()
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
        lat_ref_str: str = (
            (
                lat_ref.decode("ascii", errors="replace")
                if isinstance(lat_ref, bytes)
                else str(lat_ref)
            )
            .replace("\x00", "")
            .strip()
        )
        lon_ref_str: str = (
            (
                lon_ref.decode("ascii", errors="replace")
                if isinstance(lon_ref, bytes)
                else str(lon_ref)
            )
            .replace("\x00", "")
            .strip()
        )
        lat_dd, lat_card = dms_to_dd(latitude, lat_ref_str)
        lon_dd, lon_card = dms_to_dd(longitude, lon_ref_str)
    except (ValueError, AttributeError, TypeError) as err:
        logger.debug("Failed to convert GPS DMS to decimal: %s", err)
        return None
    else:
        # Format with degree symbol and cardinal direction (standard GPS display)
        return f"{lat_dd:.6f}°{lat_card}, {lon_dd:.6f}°{lon_card}"


def extract_image_metadata(image_path: PathLike) -> MetadataDict:
    """Derive high-level metadata (date, description, GPS string, raw EXIF).

    Returns None for unavailable date/description/gps instead of sentinel strings.
    """
    metadata: MetadataDict = {}
    img_path = Path(image_path)
    exif_data = get_exif_data(img_path) or {}

    # Date, Time, Description, GPS
    metadata["date"] = _extract_exif_date(img_path, exif_data)
    metadata["time"] = _extract_exif_time(img_path, exif_data)
    metadata["description"] = _extract_description(exif_data)
    metadata["gps"] = _extract_gps_str(exif_data.get("GPSInfo"))

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
            return text
        return (
            f"Error: {res.error_stage} - {res.error_message}"
            if res.error_message
            else "Unknown error"
        )

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
) -> tuple[list[str], list[list[str]], list[str]]:
    """Prepare headers, rows, and field names for reports.

    Args:
        results: List of PerformanceResult objects.
        header_separator: String to use for separating header lines (default: "<br>").

    Returns:
        A tuple containing:
        - list[str]: Headers for the table.
        - list[list[str]]: Rows of data for the table.
        - list[str]: The names of the fields.
    """
    if not results:
        return [], [], []

    result_set = ResultSet(results)
    field_names = ["model_name", *result_set.get_fields(), "output"]
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
) -> dict[str, Any]:
    """Analyze results to identify common model issues and calculate performance highlights.

    Args:
        results: List of model performance results
        context: Optional context string (from prompt) for cataloging utility analysis
    """
    summary: dict[str, Any] = {
        "total_models": len(results),
        "failed_models": [],
        "repetitive_models": [],
        "hallucination_models": [],
        "verbose_models": [],
        "formatting_issues": [],
        "excessive_bullets": [],
        # Cataloging utility summary
        "cataloging_grades": {},  # grade -> list of model names
        "cataloging_best": None,  # (model_name, score, grade)
        "cataloging_worst": None,  # (model_name, score, grade)
        "cataloging_avg_score": 0.0,
        "low_utility_models": [],  # Models with grade D or F
    }

    # Separate successful and failed results
    successful = [r for r in results if r.success]

    # Calculate performance highlights for successful models
    if successful:
        # Fastest model (generation TPS)
        fastest = max(successful, key=lambda r: getattr(r.generation, "generation_tps", 0) or 0)
        fastest_tps = getattr(fastest.generation, "generation_tps", 0) or 0
        summary["fastest_model"] = (fastest.model_name, fastest_tps)

        # Most memory efficient (lowest peak memory)
        most_efficient = min(
            successful,
            key=lambda r: getattr(r.generation, "peak_memory", float("inf")) or float("inf"),
        )
        efficient_mem = getattr(most_efficient.generation, "peak_memory", 0) or 0
        summary["most_efficient_model"] = (most_efficient.model_name, efficient_mem)

        # Fastest to load
        fastest_load = min(
            successful,
            key=lambda r: getattr(r, "load_time", float("inf")) or float("inf"),
        )
        load_time = getattr(fastest_load, "load_time", 0) or 0
        summary["fastest_load_model"] = (fastest_load.model_name, load_time)

        # Average TPS
        total_tps = sum(getattr(r.generation, "generation_tps", 0) or 0 for r in successful)
        avg_tps = total_tps / len(successful) if successful else 0
        summary["average_tps"] = avg_tps
        summary["successful_count"] = len(successful)

        # Resource usage summary
        total_mem = sum(getattr(r.generation, "peak_memory", 0) or 0 for r in successful)
        avg_mem = total_mem / len(successful) if successful else 0
        summary["total_peak_memory"] = total_mem
        summary["average_peak_memory"] = avg_mem

        # Memory efficiency (tokens per GB)
        total_tokens = sum(
            (getattr(r.generation, "prompt_tokens", 0) or 0)
            + (getattr(r.generation, "generation_tokens", 0) or 0)
            for r in successful
        )
        tokens_per_gb = total_tokens / total_mem if total_mem > 0 else 0
        summary["memory_efficiency"] = tokens_per_gb

    # Track cataloging utility scores for aggregation
    utility_scores: list[tuple[str, float, str, str]] = []  # (model, score, grade, weakness)

    for res in results:
        if not res.success:
            summary["failed_models"].append(
                (res.model_name, res.error_stage, res.error_message),
            )
            continue

        if res.generation and hasattr(res.generation, "text"):
            text = getattr(res.generation, "text", "") or ""
            gen_tokens = getattr(res.generation, "generation_tokens", 0)

            # Use consolidated quality analysis utility
            analysis = analyze_generation_text(text, gen_tokens)

            if analysis.is_repetitive:
                summary["repetitive_models"].append((res.model_name, analysis.repeated_token))
            if analysis.hallucination_issues:
                summary["hallucination_models"].append(
                    (res.model_name, analysis.hallucination_issues),
                )
            if analysis.is_verbose:
                summary["verbose_models"].append((res.model_name, gen_tokens))
            if analysis.formatting_issues:
                summary["formatting_issues"].append(
                    (res.model_name, analysis.formatting_issues),
                )
            if analysis.has_excessive_bullets:
                summary["excessive_bullets"].append((res.model_name, analysis.bullet_count))

            # Compute cataloging utility
            utility = compute_cataloging_utility(text, context)
            score = float(utility["utility_score"])
            grade = str(utility["utility_grade"])
            weakness = str(utility["primary_weakness"])
            utility_scores.append((res.model_name, score, grade, weakness))

            # Group by grade
            summary["cataloging_grades"].setdefault(grade, []).append(res.model_name)

            # Track low utility models (D or F)
            if grade in ("D", "F"):
                summary["low_utility_models"].append((res.model_name, score, grade, weakness))

    # Compute cataloging summary statistics
    if utility_scores:
        # Best and worst
        best = max(utility_scores, key=lambda x: x[1])
        worst = min(utility_scores, key=lambda x: x[1])
        summary["cataloging_best"] = (best[0], best[1], best[2])
        summary["cataloging_worst"] = (worst[0], worst[1], worst[2])

        # Average score
        avg_score = sum(s[1] for s in utility_scores) / len(utility_scores)
        summary["cataloging_avg_score"] = avg_score

    return summary


def compute_performance_statistics(results: list[PerformanceResult]) -> dict[str, Any]:
    """Compute performance statistics (min, max, avg) for successful runs.

    Uses single-pass aggregation to build stats for all fields at once
    reducing overhead from repeated filtering and type conversions.
    """
    stats: dict[str, Any] = {}
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


def _format_top_performers_html(summary: dict[str, Any]) -> list[str]:
    parts = []

    # Performance Highlights
    if any(
        k in summary
        for k in ["fastest_model", "most_efficient_model", "fastest_load_model", "average_tps"]
    ):
        parts.append("<h3>🏆 Performance Highlights</h3><ul>")

        if "fastest_model" in summary:
            model, tps = summary["fastest_model"]
            parts.append(
                f"<li><b>Fastest:</b> <code>{html.escape(model)}</code> ({tps:.1f} tps)</li>",
            )

        if "most_efficient_model" in summary:
            model, mem = summary["most_efficient_model"]
            parts.append(
                f"<li><b>💾 Most efficient:</b> "
                f"<code>{html.escape(model)}</code> ({mem:.1f} GB)</li>",
            )

        if "fastest_load_model" in summary:
            model, load_time = summary["fastest_load_model"]
            parts.append(
                f"<li><b>⚡ Fastest load:</b> "
                f"<code>{html.escape(model)}</code> ({load_time:.2f}s)</li>",
            )

        if "average_tps" in summary and "successful_count" in summary:
            parts.append(
                f"<li><b>📊 Average TPS:</b> {summary['average_tps']:.1f} "
                f"across {summary['successful_count']} models</li>",
            )

        parts.append("</ul>")

    # Resource Usage Summary
    if any(k in summary for k in ["total_peak_memory", "average_peak_memory", "memory_efficiency"]):
        parts.append("<h3>📈 Resource Usage</h3><ul>")

        if "total_peak_memory" in summary:
            parts.append(
                f"<li><b>Total peak memory:</b> {summary['total_peak_memory']:.1f} GB</li>",
            )

        if "average_peak_memory" in summary:
            parts.append(
                f"<li><b>Average peak memory:</b> {summary['average_peak_memory']:.1f} GB</li>",
            )

        if "memory_efficiency" in summary:
            parts.append(
                f"<li><b>Memory efficiency:</b> {summary['memory_efficiency']:.0f} tokens/GB</li>",
            )

        parts.append("</ul>")

    return parts


def _format_quality_issues_html(summary: dict[str, Any]) -> list[str]:
    quality_parts = []

    if summary.get("failed_models"):
        quality_parts.append(
            f"<li><b class='metric-bad'>❌ Failed Models "
            f"({len(summary['failed_models'])}):</b><ul>",
        )
        quality_parts.extend(
            [
                f"<li><code>{html.escape(model)}</code> ({html.escape(stage)})</li>"
                for model, stage, _ in summary["failed_models"]
            ],
        )
        quality_parts.append("</ul></li>")

    if summary.get("context_ignored"):
        quality_parts.append(
            f"<li><b class='metric-warn'>⚠️ Context Ignored "
            f"({len(summary['context_ignored'])}):</b><ul>",
        )
        quality_parts.extend(
            [f"<li><code>{html.escape(model)}</code></li>" for model in summary["context_ignored"]],
        )
        quality_parts.append("</ul></li>")

    if summary.get("repetitive_models"):
        quality_parts.append(
            f"<li><b class='metric-warn'>🔄 Repetitive Output "
            f"({len(summary['repetitive_models'])}):</b><ul>",
        )
        quality_parts.extend(
            [
                f"<li><code>{html.escape(model)}</code> "
                f"(token: <code>{html.escape(token)}</code>)</li>"
                for model, token in summary["repetitive_models"]
            ],
        )
        quality_parts.append("</ul></li>")

    if summary.get("hallucination_models"):
        quality_parts.append(
            f"<li><b class='metric-warn'>👻 Hallucinations "
            f"({len(summary['hallucination_models'])}):</b><ul>",
        )
        quality_parts.extend(
            [
                f"<li><code>{html.escape(model)}</code></li>"
                for model, _ in summary["hallucination_models"]
            ],
        )
        quality_parts.append("</ul></li>")

    if summary.get("formatting_issues"):
        quality_parts.append(
            f"<li><b class='metric-warn'>📝 Formatting Issues "
            f"({len(summary['formatting_issues'])}):</b><ul>",
        )
        quality_parts.extend(
            [
                f"<li><code>{html.escape(model)}</code></li>"
                for model, _ in summary["formatting_issues"]
            ],
        )
        quality_parts.append("</ul></li>")

    parts = []
    if quality_parts:
        parts.append("<h3>⚠️ Quality Issues</h3><ul>")
        parts.extend(quality_parts)
        parts.append("</ul>")

    return parts


def _format_cataloging_summary_html(summary: dict[str, Any]) -> list[str]:
    """Format cataloging utility summary as HTML."""
    parts: list[str] = []

    # Only show if we have cataloging data
    if not summary.get("cataloging_best"):
        return parts

    parts.append("<h3>📚 Cataloging Utility Summary</h3>")

    # Grade distribution overview
    grades = summary.get("cataloging_grades", {})
    if grades:
        grade_counts = []
        for grade in ["A", "B", "C", "D", "F"]:
            count = len(grades.get(grade, []))
            if count > 0:
                emoji = GRADE_EMOJIS.get(grade, "")
                grade_counts.append(f"{emoji} {grade}: {count}")
        if grade_counts:
            parts.append(f"<p><b>Grade Distribution:</b> {' | '.join(grade_counts)}</p>")

    # Average score
    avg_score = summary.get("cataloging_avg_score", 0)
    if avg_score > 0:
        parts.append(f"<p><b>Average Utility Score:</b> {avg_score:.0f}/100</p>")

    # Best and worst performers
    parts.append("<ul>")
    if summary.get("cataloging_best"):
        model, score, grade = summary["cataloging_best"]
        emoji = GRADE_EMOJIS.get(grade, "")
        parts.append(
            f"<li><b>Best for cataloging:</b> <code>{html.escape(model)}</code> "
            f"({emoji} {grade}, {score:.0f}/100)</li>",
        )
    if summary.get("cataloging_worst"):
        model, score, grade = summary["cataloging_worst"]
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


def _format_cataloging_summary_text(summary: dict[str, Any]) -> list[str]:
    """Format cataloging utility summary as Markdown text."""
    parts: list[str] = []

    # Only show if we have cataloging data
    if not summary.get("cataloging_best"):
        return parts

    parts.append("## 📚 Cataloging Utility Summary")
    parts.append("")

    # Grade distribution overview
    grades = summary.get("cataloging_grades", {})
    if grades:
        grade_counts = []
        for grade in ["A", "B", "C", "D", "F"]:
            count = len(grades.get(grade, []))
            if count > 0:
                emoji = GRADE_EMOJIS.get(grade, "")
                grade_counts.append(f"{emoji} {grade}: {count}")
        if grade_counts:
            parts.append(f"**Grade Distribution:** {' | '.join(grade_counts)}")
            parts.append("")

    # Average score
    avg_score = summary.get("cataloging_avg_score", 0)
    if avg_score > 0:
        parts.append(f"**Average Utility Score:** {avg_score:.0f}/100")
        parts.append("")

    # Best and worst performers
    if summary.get("cataloging_best"):
        model, score, grade = summary["cataloging_best"]
        emoji = GRADE_EMOJIS.get(grade, "")
        parts.append(f"- **Best for cataloging:** `{model}` ({emoji} {grade}, {score:.0f}/100)")
    if summary.get("cataloging_worst"):
        model, score, grade = summary["cataloging_worst"]
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


def format_issues_summary_html(summary: dict[str, Any], stats: dict[str, Any]) -> str:
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


def _format_top_performers_text(summary: dict[str, Any]) -> list[str]:
    parts = []

    # Performance Highlights
    if any(
        k in summary
        for k in ["fastest_model", "most_efficient_model", "fastest_load_model", "average_tps"]
    ):
        parts.append("## 🏆 Performance Highlights")
        parts.append("")  # Blank line after heading (MD022)

        if "fastest_model" in summary:
            model, tps = summary["fastest_model"]
            parts.append(f"- **Fastest:** `{model}` ({tps:.1f} tps)")

        if "most_efficient_model" in summary:
            model, mem = summary["most_efficient_model"]
            parts.append(f"- **💾 Most efficient:** `{model}` ({mem:.1f} GB)")

        if "fastest_load_model" in summary:
            model, load_time = summary["fastest_load_model"]
            parts.append(f"- **⚡ Fastest load:** `{model}` ({load_time:.2f}s)")

        if "average_tps" in summary and "successful_count" in summary:
            parts.append(
                f"- **📊 Average TPS:** {summary['average_tps']:.1f} "
                f"across {summary['successful_count']} models",
            )

        parts.append("")  # Blank line after list (MD032)

    # Resource Usage Summary
    if any(k in summary for k in ["total_peak_memory", "average_peak_memory", "memory_efficiency"]):
        parts.append("## 📈 Resource Usage")
        parts.append("")  # Blank line after heading (MD022)

        if "total_peak_memory" in summary:
            parts.append(f"- **Total peak memory:** {summary['total_peak_memory']:.1f} GB")

        if "average_peak_memory" in summary:
            parts.append(f"- **Average peak memory:** {summary['average_peak_memory']:.1f} GB")

        if "memory_efficiency" in summary:
            parts.append(f"- **Memory efficiency:** {summary['memory_efficiency']:.0f} tokens/GB")

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


def _format_quality_issues_text(summary: dict[str, Any]) -> list[str]:
    parts = []
    quality_parts = []

    if summary.get("failed_models"):
        quality_parts.append(f"- **❌ Failed Models ({len(summary['failed_models'])}):**")
        for model, stage, _ in summary["failed_models"]:
            quality_parts.append(f"  - `{model}` (`{stage}`)")

    if summary.get("context_ignored"):
        quality_parts.append(f"- **⚠️ Context Ignored ({len(summary['context_ignored'])}):**")
        quality_parts.extend([f"  - `{model}`" for model in summary["context_ignored"]])

    if summary.get("repetitive_models"):
        quality_parts.append(
            f"- **🔄 Repetitive Output ({len(summary['repetitive_models'])}):**",
        )
        for model, token in summary["repetitive_models"]:
            quality_parts.append(f"  - `{model}` (token: `{token}`)")

    if summary.get("hallucination_models"):
        quality_parts.append(
            f"- **👻 Hallucinations ({len(summary['hallucination_models'])}):**",
        )
        quality_parts.extend(
            [f"  - `{model}`" for model, _ in summary["hallucination_models"]],
        )

    if summary.get("formatting_issues"):
        quality_parts.append(
            f"- **📝 Formatting Issues ({len(summary['formatting_issues'])}):**",
        )
        quality_parts.extend(
            [f"  - `{model}`" for model, _ in summary["formatting_issues"]],
        )

    if quality_parts:
        parts.append("## ⚠️ Quality Issues")
        parts.append("")  # Blank line after heading (MD022)
        parts.extend(quality_parts)
        parts.append("")

    return parts


def format_issues_summary_text(summary: dict[str, Any], stats: dict[str, Any]) -> str:
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
                img_to_save: Image.Image = img_original
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

    headers, rows, field_names = _prepare_table_data(results, header_separator="\n")
    if not headers or not rows:
        logger.info("No data to display in stats table.")
        return

    # Use tabulate with 'simple' format for clean plain text output
    # Determine column alignment (numeric fields right-aligned)
    colalign = ["right" if is_numeric_field(field) else "left" for field in field_names]

    # Set max widths for specific columns to keep table compact
    # Quality Issues: 20 chars, Output: 50 chars, others: no limit
    maxcolwidths: list[int | None] = []
    for field_name in field_names:
        if field_name == "quality_issues":
            maxcolwidths.append(20)
        elif field_name == "output":
            maxcolwidths.append(50)
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
    # Disable line-length and inline-html linting - model outputs can be long and
    # error tracebacks use <details> for collapsible sections
    md.append("<!-- markdownlint-disable MD013 MD033 -->")
    md.append("")

    sorted_results = _sort_results_by_time(results)
    for res in sorted_results:
        # Header with status icon
        icon = "✅" if res.success else "❌"
        md.append(f"### {icon} {res.model_name}")
        md.append("")

        if not res.success:
            md.append(f"**Status:** Failed ({res.error_stage})")
            # Wrap error messages to avoid excessive line length
            error_msg = str(res.error_message)
            max_inline_length = 80
            if len(error_msg) > max_inline_length:
                # Escape markdown characters and wrap at word boundaries
                wrapped_lines = textwrap.wrap(
                    error_msg,
                    width=76,  # Leave room for "> " prefix
                    break_long_words=False,
                    break_on_hyphens=False,
                )
                md.append("**Error:**")
                md.append("")
                md.extend(f"> {line}" for line in wrapped_lines)
            else:
                # Short errors can stay inline
                md.append(f"**Error:** {error_msg}")
            if res.error_type:
                md.append(f"**Type:** `{res.error_type}`")
            if res.error_package:
                md.append(f"**Package:** `{res.error_package}`")
            # Include full traceback in collapsible section for GitHub issue reports
            if res.error_traceback:
                md.append("")
                md.append("<details>")
                md.append("<summary>Full Traceback (click to expand)</summary>")
                md.append("")
                md.append("```python")
                md.append(res.error_traceback.rstrip())
                md.append("```")
                md.append("")
                md.append("</details>")
        else:
            # Show metrics summary line
            gen = res.generation
            if gen:
                tps = getattr(gen, "generation_tps", 0)
                tokens = getattr(gen, "generation_tokens", 0)
                md.append(
                    f"**Metrics:** {fmt_num(tps)} TPS | {tokens} tokens",
                )

            md.append("")
            md.append("```text")
            text = str(getattr(res.generation, "text", "")) if res.generation else ""
            md.append(text)
            md.append("```")

            # Show quality warnings if any
            if res.generation:
                analysis = getattr(res.generation, "quality_analysis", None)
                if analysis and analysis.issues:
                    md.append("")
                    md.append("⚠️ **Quality Warnings:**")
                    # Escape issues to prevent tags like <end_of_utterance> from being hidden
                    md.extend(f"- {html.escape(issue)}" for issue in analysis.issues)

        md.append("")
        md.append("---")
        md.append("")

    # Re-enable linting after gallery section
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

    md.append("> **Prompt used:**\n>\n> " + prompt.replace("\n", "\n> "))
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

    This outputs only the data table without metadata, properly escaping tabs and newlines
    in field values to maintain TSV format integrity.

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

    # Clean and escape row data
    clean_rows = []
    for row in rows:
        clean_row = [escape_tsv_value(str(cell)) for cell in row]
        clean_rows.append(clean_row)

    # Generate TSV using tabulate with tsv format
    tsv_content = tabulate(
        clean_rows,
        headers=clean_headers,
        tablefmt="tsv",
    )

    try:
        with filename.open("w", encoding="utf-8") as f:
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
                pass  # SDK version is nice-to-have, not critical
        info["Python Version"] = sys.version.split()[0]
        info["Architecture"] = platform.machine()

        # Get GPU info
        _, gpu_name = get_system_info()
        if gpu_name:
            info["GPU/Chip"] = gpu_name

        # Get detailed device info if on Apple Silicon
        if platform.machine() == "arm64":
            device_info = get_device_info() or {}
            displays = device_info.get("SPDisplaysDataType") or []
            if displays and isinstance(displays, list):
                first = displays[0]
                if isinstance(first, dict):
                    gpu_cores = first.get("sppci_cores")
                    if gpu_cores is not None:
                        info["GPU Cores"] = str(gpu_cores)

                    # Get Metal family version if available
                    metal_family = first.get("spdisplays_mtlgpufamilysupport")
                    if metal_family:
                        # Convert "spdisplays_metal4" to "Metal 4"
                        metal_ver = metal_family.replace("spdisplays_metal", "Metal ")
                        info["Metal Support"] = metal_ver

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
    except TimeoutError as err:
        msg = f"Timeout while reading image: {image_path}"
        raise OSError(msg) from err
    except UnidentifiedImageError as err:
        msg = f"File is not a recognized image format: {image_path}"
        raise ValueError(msg) from err
    except (OSError, ValueError) as err:
        msg = f"Error accessing image {image_path}: {err}"
        raise OSError(msg) from err


# MOD: Added HF cache integrity diagnostic helper
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
        ("Config Missing", ["does not appear to have a file named"]),
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
        lazy=params.lazy,
        trust_remote_code=params.trust_remote_code,
    )
    # Note: mlx-vlm.utils.load() is type-hinted to return Union[PreTrainedTokenizer, ...]
    # but at runtime it returns a Processor object (from AutoProcessor).

    return model, processor, getattr(model, "config", None)


def _run_model_generation(
    params: ProcessImageParams,
    timer: TimingStrategy | None = None,
) -> GenerationResult | SupportsGenerationResult:
    """Load model + processor, apply chat template, run generation, time it.

    We keep all loading + formatting + generation steps together because they
    form a tightly coupled sequence (tokenizer/model/config interplay varies by
    repo). Errors are wrapped with traceback context so upstream summaries can
    show concise messages while verbose logs retain full detail.

    Args:
        params: The parameters for the image processing.
        timer: Optional timing strategy. If None, uses PerfCounterTimer.
    """
    model: Module
    processor: Any

    # Load model from HuggingFace Hub - this handles automatic download/caching
    # and converts weights to MLX format for Apple Silicon optimization
    try:
        model, processor, config = _load_model(params)
    except Exception as load_err:
        # Capture any model loading errors (config issues, missing files, etc.)
        # MOD: Enhanced error handling with cache integrity check
        error_details = f"Model loading failed: {load_err}"
        logger.exception("Failed to load model %s", params.model_identifier)

        # MOD: HF cache integrity check on load failure
        _check_hf_cache_integrity(params.model_identifier)

        raise ValueError(error_details) from load_err

    # Apply model-specific chat template - each model has its own conversation format
    # (e.g., Llama uses <|begin_of_text|>, Phi-3 uses <|user|>, etc.)
    formatted_prompt: str | list[Any] = apply_chat_template(
        processor=processor,
        config=config,
        prompt=params.prompt,
        num_images=1,
    )
    # Handle list return from apply_chat_template
    if isinstance(formatted_prompt, list):
        formatted_prompt = "\n".join(str(m) for m in formatted_prompt)

    # Time the generation process manually since MLX VLM doesn't include timing
    # Use injected timer or default to PerfCounterTimer
    if timer is None:
        timer = PerfCounterTimer()

    timer.start()
    try:
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
        )
    except TimeoutError as gen_to_err:
        msg = f"Generation timed out for model {params.model_identifier}: {gen_to_err}"
        # Re-raise to be handled by outer TimeoutError branch
        raise TimeoutError(msg) from gen_to_err
    except (OSError, ValueError) as gen_known_err:
        # Known I/O or validation-style issues
        msg = f"Model generation failed for {params.model_identifier}: {gen_known_err}"
        logger.exception("Generation error for %s", params.model_identifier)
        raise ValueError(msg) from gen_known_err
    except (RuntimeError, TypeError, AttributeError, KeyError) as gen_err:
        # Model-specific runtime errors (weights, config, tensor ops, missing attributes)
        msg = f"Model runtime error during generation for {params.model_identifier}: {gen_err}"
        logger.exception("Runtime error for %s", params.model_identifier)
        raise ValueError(msg) from gen_err

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


def process_image_with_model(params: ProcessImageParams) -> PerformanceResult:
    """Process an image with a Vision Language Model, managing stats and errors."""
    model: Module | None = None
    processor: PythonBackend | TokenizersBackend | None = None
    arch, gpu_info = get_system_info()

    # Track overall timing
    total_start_time = time.perf_counter()
    model_load_time: float | None = None
    generation_time: float | None = None

    try:
        validate_temperature(temp=params.temperature)
        validate_image_accessible(image_path=params.image_path)
        logger.debug(
            "System: %s, GPU: %s",
            arch,
            gpu_info if gpu_info is not None else "",
        )

        with TimeoutManager(params.timeout):
            output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                params=params,
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
    except (TimeoutError, OSError, ValueError) as e:
        # Full traceback already logged at ERROR level in _run_model_generation
        error_msg = str(e)
        tb_str = traceback.format_exc()
        classified_stage = _classify_error(error_msg)
        error_package = _attribute_error_to_package(error_msg, tb_str)
        return PerformanceResult(
            model_name=params.model_identifier,
            generation=None,
            success=False,
            # Use classified error as stage/status, or exception name if unknown
            error_stage=classified_stage,
            error_message=error_msg,
            error_type=type(e).__name__,
            error_package=error_package,
            error_traceback=tb_str,
            generation_time=None,
            model_load_time=None,
            total_time=None,
        )
    finally:
        # Aggressive cleanup matching mlx-vlm/tests/test_smoke.py
        if model is not None:
            del model
        if processor is not None:
            del processor

        # Force synchronization and garbage collection
        mx.synchronize()
        gc.collect()

        # Clear both Metal and MLX caches for thorough GPU memory cleanup
        mx.clear_cache()
        mx.reset_peak_memory()
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
    """Assemble key=value summary segments (status only, metrics shown separately)."""
    parts: list[str] = [
        f"model={model_short}",
        f"status={'OK' if res.success else 'FAIL'}",
    ]
    if res.error_stage:
        parts.append(f"stage={res.error_stage}")
    # Metrics are shown in the dedicated metrics section below, not in summary
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
    if not text_val:
        logger.info(
            "<empty>",
            extra={"style_hint": LogStyles.GENERATED_TEXT},
        )
        return

    # MOD: Analyze quality using consolidated utility with optional prompt
    gen_tokens = getattr(gen, "generation_tokens", 0)
    analysis = analyze_generation_text(
        text_val,
        gen_tokens,
        prompt=prompt,
        context_marker=context_marker,
    )
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
    # MOD: Added context ignorance warning
    if analysis.is_context_ignored and analysis.missing_context_terms:
        missing = ", ".join(analysis.missing_context_terms[:3])
        log_warning_note(f"Context ignored (missing: {missing})")

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

    # Add breathing room
    log_blank()

    # Generated text with emoji prefix for easy scanning
    gen_text = getattr(res.generation, "text", None) or ""

    # MOD: Analyze quality using consolidated utility with optional prompt
    gen_tokens = getattr(res.generation, "generation_tokens", 0)
    analysis = analyze_generation_text(
        gen_text,
        gen_tokens,
        prompt=prompt,
        context_marker=context_marker,
    )

    log_blank()  # Extra spacing before generated text section
    log_metric_label("Generated Text:", emoji="📝")

    # Warn about quality issues
    if analysis.is_repetitive and analysis.repeated_token:
        warning_msg = (
            f"WARNING: Output appears to be garbage (repetitive: '{analysis.repeated_token}')"
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

    # MOD: Added context ignorance warning
    if analysis.is_context_ignored and analysis.missing_context_terms:
        missing = ", ".join(analysis.missing_context_terms)
        log_warning_note(
            f"Note: Output ignored key context (missing: {missing})",
            prefix="⚠️",
        )

    log_generated_text(gen_text, wrap=True, indent="   ")

    log_blank()  # Breathing room

    if detailed:
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


def _build_compact_metric_parts(
    res: PerformanceResult,
    gen: GenerationResult | SupportsGenerationResult,
) -> list[str]:
    """Return list of metric segments for compact metrics line."""
    total_time_val = getattr(res, "total_time", None)
    generation_time_val = getattr(res, "generation_time", None)
    load_time_val = getattr(res, "model_load_time", None)
    peak_mem = getattr(gen, "peak_memory", None) or 0.0
    prompt_tokens = getattr(gen, "prompt_tokens", 0) or 0
    gen_tokens = getattr(gen, "generation_tokens", 0) or 0
    all_tokens = prompt_tokens + gen_tokens
    gen_tps = getattr(gen, "generation_tps", 0.0) or 0.0
    prompt_tps = getattr(gen, "prompt_tps", 0.0) or 0.0

    parts: list[str] = []
    if total_time_val is not None:
        parts.append(f"total={_format_time_seconds(total_time_val)}")
    if generation_time_val is not None:
        parts.append(f"gen={_format_time_seconds(generation_time_val)}")
    if load_time_val is not None:
        parts.append(f"load={_format_time_seconds(load_time_val)}")
    if peak_mem > 0:
        mem_fmt = format_field_value("peak_memory", peak_mem)
        mem_str = f"{mem_fmt}GB" if not str(mem_fmt).endswith("GB") else str(mem_fmt)
        parts.append(f"peak_mem={mem_str}")
    if all_tokens:
        parts.append(
            "tokens(total/prompt/gen)="
            f"{fmt_num(all_tokens)}/{fmt_num(prompt_tokens)}/{fmt_num(gen_tokens)}",
        )
    if gen_tps:
        parts.append(f"gen_tps={fmt_num(gen_tps)}")
    if prompt_tps:
        parts.append(f"prompt_tps={fmt_num(prompt_tps)}")
    return parts


def _align_metric_parts(parts: list[str]) -> list[str]:
    """Return parts with compact key=value formatting.

    No padding/alignment to keep output compact and readable.
    """
    # Simply return parts as-is since they're already in key=value format
    return parts


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


def print_model_result(
    result: PerformanceResult,
    *,
    verbose: bool = False,
    detailed_metrics: bool = False,
    run_index: int | None = None,
    total_runs: int | None = None,
    prompt: str | None = None,  # MOD: Added for context ignorance detection
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
        # Stage is already in summary, so we only log the full error here
        if result.error_message:
            logger.info("Error: %s", result.error_message)
        if result.captured_output_on_fail:
            _log_wrapped_error("Output:", str(result.captured_output_on_fail))
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


def _resolve_safe_command(command: list[str]) -> list[str] | None:
    """Validate and resolve a command for subprocess execution.

    Ensures the executable is an absolute, existing file and prevents NUL bytes.
    Returns the resolved command list or None if validation fails.
    """
    if not command or not all(isinstance(arg, str) and arg for arg in command):
        return None
    if any("\x00" in arg for arg in command):
        return None

    exec_path = Path(command[0])
    if not exec_path.is_absolute():
        resolved = shutil.which(command[0])
        if not resolved:
            return None
        exec_path = Path(resolved)

    try:
        exec_path = exec_path.resolve()
    except OSError:
        return None

    if not exec_path.is_file():
        return None

    return [str(exec_path), *command[1:]]


# MOD: Write environment dump to separate file to reduce log clutter
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
        is_conda = conda_env is not None

        # Ensure output directory exists
        env_log_path = output_path.resolve()
        env_log_path.parent.mkdir(parents=True, exist_ok=True)

        with env_log_path.open("w", encoding="utf-8") as env_file:
            env_file.write("=" * 80 + "\n")
            env_file.write(f"FULL ENVIRONMENT DUMP - {local_now_str()}\n")
            env_file.write("=" * 80 + "\n\n")

            # Try pip freeze first (works in both conda and venv)
            try:
                pip_cmd = _resolve_safe_command([sys.executable, "-m", "pip", "freeze"])
                if pip_cmd:
                    # Command is hardcoded above, validated by _resolve_safe_command
                    pip_result = subprocess.run(  # noqa: S603
                        pip_cmd,
                        capture_output=True,
                        text=True,
                        timeout=30,
                        check=False,
                    )
                    if pip_result.returncode == 0:
                        env_file.write("--- pip freeze ---\n")
                        env_file.write(pip_result.stdout)
                        env_file.write("\n")
                    else:
                        env_file.write(f"pip freeze failed: {pip_result.stderr}\n")
                else:
                    env_file.write("pip freeze skipped: invalid executable\n")
            except (subprocess.SubprocessError, FileNotFoundError) as pip_err:
                env_file.write(f"Could not run pip freeze: {pip_err}\n")

            # If in conda, also capture conda list
            if is_conda:
                try:
                    conda_path = shutil.which("conda")
                    if conda_path:
                        conda_cmd = _resolve_safe_command([conda_path, "list"])
                        if conda_cmd:
                            # Command is hardcoded above, validated by _resolve_safe_command
                            conda_result = subprocess.run(  # noqa: S603
                                conda_cmd,
                                capture_output=True,
                                text=True,
                                timeout=30,
                                check=False,
                            )
                            if conda_result.returncode == 0:
                                env_file.write(f"--- conda list (env: {conda_env}) ---\n")
                                env_file.write(conda_result.stdout)
                                env_file.write("\n")
                            else:
                                env_file.write(f"conda list failed: {conda_result.stderr}\n")
                        else:
                            env_file.write("conda list skipped: invalid executable\n")
                    else:
                        env_file.write("conda command not found in PATH\n")
                except (subprocess.SubprocessError, FileNotFoundError) as conda_err:
                    env_file.write(f"Could not run conda list: {conda_err}\n")

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
    if MISSING_DEPENDENCIES:
        for message in MISSING_DEPENDENCIES.values():
            logger.critical("%s", message)
        missing_list = ", ".join(sorted(MISSING_DEPENDENCIES))
        error_message = (
            f"Missing required runtime dependencies: {missing_list}. "
            "Install the missing packages or adjust optional features."
        )
        raise RuntimeError(error_message)

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
    file_formatter: logging.Formatter = logging.Formatter(
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
    if args.verbose:
        print_version_info(library_versions)

    if args.trust_remote_code:
        print_cli_separator()
        log_warning_note("SECURITY WARNING: --trust-remote-code is enabled.")
        log_warning_note("This allows execution of remote code and may pose security risks.")

    return library_versions


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

    metadata: MetadataDict = extract_image_metadata(image_path)

    # Display key metadata (fallback to N/A only at presentation time)
    logger.info("Date: %s", metadata.get("date") if metadata.get("date") is not None else "")
    logger.info(
        "Description: %s",
        metadata.get("description") if metadata.get("description") is not None else "",
    )
    logger.info(
        "GPS Location: %s",
        metadata.get("gps") if metadata.get("gps") is not None else "",
    )

    if args.verbose:
        # Note: pretty_print_exif has its own header separators, no need for one here
        exif_data: ExifDict | None = get_exif_data(image_path)
        if exif_data:
            pretty_print_exif(exif_data, show_all=True)
        else:
            logger.warning("No detailed EXIF data could be extracted.")
    return metadata


def prepare_prompt(args: argparse.Namespace, metadata: MetadataDict) -> str:
    """Prepare the prompt for the VLM, using user input or generating from metadata."""
    print_cli_section("Prompt Configuration")

    prompt: str
    if args.prompt:
        prompt = args.prompt
        logger.info("Using user-provided prompt.")
    else:
        logger.info("Generating default prompt based on image metadata.")
        desc = metadata.get("description")
        date_val = metadata.get("date")
        time_val = metadata.get("time")
        gps_val = metadata.get("gps")
        prompt_parts: list[str] = [
            (
                "Provide a factual caption, description, and keywords suitable for "
                "cataloguing, or searching for, the image."
            ),
            (f"\n\nContext: The image relates to '{desc}'" if desc else ""),
            (f"\n\nThe photo was taken around {date_val}" if date_val else ""),
            (f" at local time {time_val}" if time_val else ""),
            (f" from GPS {gps_val}" if gps_val else ""),
            (
                ". Focus on visual content, drawing on any available "
                "contextual information for specificity. Do not speculate."
            ),
        ]
        prompt = " ".join(filter(None, prompt_parts)).strip()
        logger.debug("Using generated prompt based on metadata.")

    # Truncate long prompts for display
    max_display_len = 200
    if len(prompt) > max_display_len:
        prompt_display = prompt[:max_display_len] + "..."
        logger.info("Final prompt: %s", prompt_display)
    else:
        logger.info("Final prompt: %s", prompt)
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


def _apply_exclusions(
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
        model_identifiers = _apply_exclusions(
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
        model_identifiers = _apply_exclusions(
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
            context_marker=args.context_marker,
        )
        result: PerformanceResult = process_image_with_model(params)

        # MOD: Calculate quality score and analysis only for successful generations
        if result.success and result.generation:
            gen_text = str(getattr(result.generation, "text", ""))
            gen_tokens = getattr(result.generation, "generation_tokens", 0)
            if gen_text:
                # Perform quality analysis (only for successful runs)
                analysis = analyze_generation_text(
                    gen_text,
                    gen_tokens,
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
                )

        results.append(result)

        # MOD: Structured output for this run with prompt for evaluation
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

    return ", ".join(parts) if parts else "no issues detected"


def _build_quality_issues_string(analysis: GenerationQualityAnalysis) -> str | None:
    """Build consolidated quality issues string from analysis results.

    Prioritizes critical issues first (refusal → repetitive → lang_mixing →
    hallucination → generic → verbose → formatting → bullets → context-ignored).

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

    # Critical issues first
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


# MOD: Added error bucketing diagnostic helper
def _bucket_failures_by_error(
    results: list[PerformanceResult],
) -> dict[str, list[str]]:
    """Group failed models by their root cause exception type and message pattern.

    This transforms a flat list of failures into categorized buckets, making it
    easier to identify systematic issues (e.g., all models failing with the same
    missing parameter error).

    Args:
        results: List of all PerformanceResult objects

    Returns:
        Dictionary mapping error signature to list of failed model names.
        Format: {"TypeError: unexpected keyword 'temperature'": ["model1", "model2"], ...}
    """
    buckets: dict[str, list[str]] = {}
    failed_results = [r for r in results if not r.success]

    min_wrapped_parts = 3  # Minimum parts to identify wrapped exception
    max_error_msg_len = 120  # Truncate long error messages for readability

    for res in failed_results:
        # Build error signature from error_type and core message
        error_type = res.error_type or "UnknownError"
        error_msg = res.error_message or "No error message"

        # Extract the core exception message (first line, truncated)
        # Skip traceback noise and focus on the actual error
        core_msg = error_msg.split("\n")[0].strip()

        # Try to extract the root cause from wrapped exceptions
        # Look for patterns like "ValueError: Model loading failed: ImportError: ..."
        if ": " in core_msg:
            parts = core_msg.split(": ")
            # If this looks like a wrapped exception, extract the inner one
            if len(parts) >= min_wrapped_parts and parts[1].startswith("Model"):
                # Example pattern: ValueError, Model loading failed, ImportError, cannot import
                # We want to extract: ImportError: cannot import
                inner_parts = parts[2:]
                core_msg = ": ".join(inner_parts)
                # Update error_type to the actual root cause
                if inner_parts and inner_parts[0] in {
                    "TypeError",
                    "ImportError",
                    "ValueError",
                    "AttributeError",
                    "KeyError",
                    "OSError",
                    "RuntimeError",
                }:
                    error_type = inner_parts[0]

        # Truncate very long messages but keep key info
        if len(core_msg) > max_error_msg_len:
            core_msg = core_msg[: max_error_msg_len - 3] + "..."

        # Create signature
        signature = f"{error_type}: {core_msg}"

        # Add model to this bucket
        if signature not in buckets:
            buckets[signature] = []
        buckets[signature].append(res.model_name)

    return buckets


def log_failure_summary(buckets: dict[str, list[str]]) -> None:
    """Log error buckets as a human-readable failure summary.

    Args:
        buckets: Dictionary from _bucket_failures_by_error
    """
    if not buckets:
        return

    log_rule(80, char="=", bold=True, pre_newline=True)
    logger.info("FAILURE SUMMARY - Models Grouped by Root Cause")
    log_rule(80, char="=", bold=True, post_newline=True)

    # Sort buckets by number of affected models (most common errors first)
    sorted_buckets = sorted(
        buckets.items(),
        key=lambda item: len(item[1]),
        reverse=True,
    )

    for signature, models in sorted_buckets:
        logger.info("%s", signature)
        logger.info("  Affected models (%d):", len(models))
        for model in models:
            logger.info("    • %s", model)
        log_blank()  # Blank line between buckets

    log_rule(80, char="=", post_newline=True)


def log_summary(results: list[PerformanceResult]) -> None:
    """Log a text-based summary of the execution results to the log file."""
    if not results:
        return

    log_blank()
    log_rule(color=Colors.BLUE, bold=True)
    logger.info("Results Summary", extra={"style_hint": LogStyles.HEADER})
    log_rule(color=Colors.BLUE, bold=True)

    # Success stats
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    # Performance Highlights (only if we have successful models)
    if successful:
        logger.info("🏆 Performance Highlights:")

        # Fastest model (gen TPS)
        fastest = max(successful, key=lambda r: getattr(r.generation, "generation_tps", 0) or 0)
        fastest_tps = getattr(fastest.generation, "generation_tps", 0) or 0
        logger.info("   Fastest: %s (%.1f tps)", fastest.model_name, fastest_tps)

        # Most memory efficient (lowest peak memory)
        most_efficient = min(
            successful,
            key=lambda r: getattr(r.generation, "peak_memory", float("inf")) or float("inf"),
        )
        efficient_mem = getattr(most_efficient.generation, "peak_memory", 0) or 0
        logger.info("   💾 Most efficient: %s (%.1f GB)", most_efficient.model_name, efficient_mem)

        # Fastest to load
        fastest_load = min(
            successful,
            key=lambda r: getattr(r, "load_time", float("inf")) or float("inf"),
        )
        load_time = getattr(fastest_load, "load_time", 0) or 0
        logger.info("   ⚡ Fastest load: %s (%.2fs)", fastest_load.model_name, load_time)

        # Average TPS
        total_tps = sum(getattr(r.generation, "generation_tps", 0) or 0 for r in successful)
        avg_tps = total_tps / len(successful) if successful else 0
        logger.info("   📊 Average TPS: %.1f across %d models", avg_tps, len(successful))

        log_blank()

        # Resource Usage Summary
        logger.info("📈 Resource Usage:")

        # Total and average memory
        total_mem = sum(getattr(r.generation, "peak_memory", 0) or 0 for r in successful)
        avg_mem = total_mem / len(successful) if successful else 0
        logger.info("   Total peak memory: %.1f GB", total_mem)
        logger.info("   Average peak memory: %.1f GB", avg_mem)

        # Memory efficiency (tokens per GB)
        total_tokens = sum(
            (getattr(r.generation, "prompt_tokens", 0) or 0)
            + (getattr(r.generation, "generation_tokens", 0) or 0)
            for r in successful
        )
        tokens_per_gb = total_tokens / total_mem if total_mem > 0 else 0
        logger.info("   Memory efficiency: %.0f tokens/GB", tokens_per_gb)

        log_blank()

    # Failed models
    if failed:
        logger.info("❌ Failed Models (%d):", len(failed))
        for res in failed:
            error_pkg = f" -> {res.error_package}" if res.error_package else ""
            logger.info(
                "  - %s (%s%s)",
                res.model_name,
                res.error_stage or "Unknown",
                error_pkg,
                extra={"style_hint": LogStyles.ERROR},
            )
        log_blank()

    # Successful models list
    if successful:
        logger.info("✅ Successful Models (%d):", len(successful))
        # Sort by generation tps (descending)
        sorted_success = sorted(
            successful,
            key=lambda r: getattr(r.generation, "generation_tps", 0) or 0,
            reverse=True,
        )
        for res in sorted_success:
            tps = getattr(res.generation, "generation_tps", 0) or 0
            active_mem = res.active_memory if res.active_memory else 0.0
            cache_mem = res.cache_memory if res.cache_memory else 0.0

            # Format memory info only if we have data
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
    - System info and Prompt context (replicated for stateless processing)
    """
    try:
        with filename.open("w", encoding="utf-8") as f:
            for res in results:
                record: dict[str, object] = {
                    "model": res.model_name,
                    "success": res.success,
                    "error_stage": res.error_stage,
                    "error_message": res.error_message,
                    "error_type": res.error_type,  # Original exception type for bucketing
                    "error_package": res.error_package,
                    "error_traceback": res.error_traceback,  # Full traceback for diagnostics
                    "quality_issues": _parse_quality_issues_to_list(res.quality_issues),
                    "context": {
                        "prompt": prompt,
                        "system": system_info,
                        "timestamp": local_now_str(),
                    },
                    "metrics": {},
                    "timing": {},
                }
                # Add timing metrics (always available, even for failures)
                record["timing"] = {
                    "generation_time_s": res.generation_time,
                    "model_load_time_s": res.model_load_time,
                    "total_time_s": res.total_time,
                }
                if res.generation:
                    gen = res.generation
                    record["metrics"] = {
                        "prompt_tokens": getattr(gen, "prompt_tokens", 0),
                        "generation_tokens": getattr(gen, "generation_tokens", 0),
                        "generation_tps": getattr(gen, "generation_tps", 0.0),
                        "peak_memory_gb": getattr(gen, "peak_memory", 0.0),
                        "active_memory_gb": res.active_memory if res.active_memory else 0.0,
                        "cache_memory_gb": res.cache_memory if res.cache_memory else 0.0,
                    }
                    # Include generated text for quality analysis context
                    text = getattr(gen, "text", None)
                    if text:
                        record["generated_text"] = text
                    # Include quality analysis details if available
                    quality_analysis = getattr(gen, "quality_analysis", None)
                    if quality_analysis:
                        record["quality_analysis"] = {
                            "issues": getattr(quality_analysis, "issues", []),
                            "metrics": {
                                "word_count": getattr(quality_analysis, "word_count", 0),
                                "unique_ratio": getattr(quality_analysis, "unique_ratio", 0.0),
                                "bullet_count": getattr(quality_analysis, "bullet_count", 0),
                            },
                        }
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

        # MOD: Added failure bucketing summary for better diagnostics
        # (Already handled by print_model_stats and results.md, but we add log summary now)
        log_summary(results)

        # Gather system characteristics for reports
        system_info = get_system_characteristics()

        # Generate reports
        try:
            html_output_path: Path = args.output_html.resolve()
            md_output_path: Path = args.output_markdown.resolve()
            tsv_output_path: Path = args.output_tsv.resolve()
            jsonl_output_path: Path = args.output_jsonl.resolve()
            html_output_path.parent.mkdir(parents=True, exist_ok=True)
            md_output_path.parent.mkdir(parents=True, exist_ok=True)
            tsv_output_path.parent.mkdir(parents=True, exist_ok=True)
            jsonl_output_path.parent.mkdir(parents=True, exist_ok=True)

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

            log_file_path(DEFAULT_LOG_OUTPUT, label="   Log File:")
            # Include environment.log in the output file listing
            env_log = DEFAULT_ENV_OUTPUT.resolve()
            if env_log.exists():
                log_file_path(env_log, label="   Environment:")
        except (OSError, ValueError):
            logger.exception("Failed to generate reports.")
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
        # Validate all CLI arguments early to fail fast
        validate_cli_arguments(args)

        library_versions = setup_environment(args)
        print_cli_header("MLX Vision Language Model Check")

        image_path = find_and_validate_image(args)

        metadata = handle_metadata(image_path, args)

        prompt = prepare_prompt(args, metadata)

        # Handle dry-run mode: show what would be run and exit
        if getattr(args, "dry_run", False):
            _handle_dry_run(args, image_path, prompt, library_versions)
            return

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
        logger.exception("Execution interrupted by user.")
        sys.exit(1)
    except SystemExit:
        logger.exception("Execution halted (SystemExit raised).")
        raise
    except (OSError, ValueError, RuntimeError) as main_err:
        logger.critical("Fatal error in main execution: %s", main_err)
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
