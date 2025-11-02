#!/usr/bin/env python3
"""Image analysis and caption generation using MLX Vision Language Models."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import html
import importlib.util as importlib_util
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
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Final,
    NamedTuple,
    NoReturn,
    Protocol,
    Self,
    cast,
    runtime_checkable,
)

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

psutil: Any | None = psutil_mod

if TYPE_CHECKING:
    import types
    from collections.abc import Iterator

    from mlx.nn import Module
    from mlx_vlm.generate import GenerationResult
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

# Public API (PEP 8 / PEP 561 best practice)
__all__ = [
    "PerformanceResult",
    "ProcessImageParams",
    "ResultSet",
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
    "get_system_info",
    "get_terminal_width",
    "is_numeric_field",
    "is_numeric_value",
    "main",
    "main_cli",
    "pretty_print_exif",
    "print_cli_error",
    "print_cli_header",
    "print_cli_section",
    "print_model_result",
    "print_model_stats",
    "print_version_info",
    "process_image_with_model",
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


MIN_SEPARATOR_CHARS: Final[int] = 50
DEFAULT_DECIMAL_PLACES: Final[int] = 2
LARGE_NUMBER_THRESHOLD: Final[float] = 100.0
MEDIUM_NUMBER_THRESHOLD: Final[int] = 10
THOUSAND_THRESHOLD: Final[int] = 1000
MEMORY_GB_INTEGER_THRESHOLD: Final[float] = 10.0  # >= this many GB show as integer (no decimals)
MARKDOWN_HARD_BREAK_SPACES: Final[int] = 2  # Preserve exactly two trailing spaces for hard breaks
HOUR_THRESHOLD_SECONDS: Final[int] = 3600  # Threshold for displaying HH:MM:SS runtime
IMAGE_OPEN_TIMEOUT: Final[float] = 5.0  # Timeout for opening/verifying image files
GENERATION_WRAP_WIDTH: Final[int] = 80  # Console output wrapping width for generated text

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

vlm_version: str

try:
    from mlx_vlm.generate import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load
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

    MISSING_DEPENDENCIES["mlx-vlm"] = ERROR_MLX_VLM_MISSING
try:
    import mlx_lm

    mlx_lm_version: str = getattr(mlx_lm, "__version__", NOT_AVAILABLE)
except ImportError:
    mlx_lm_version = NOT_AVAILABLE
except AttributeError:
    mlx_lm_version = "N/A (module found, no version attr)"
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

    Note: `time` attribute is added dynamically by our code after generation.
    """

    text: str | None
    prompt_tokens: int | None
    generation_tokens: int | None
    time: float | None  # Dynamically added timing attribute


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
DEFAULT_LOG_OUTPUT: Final[Path] = _SCRIPT_DIR / "output" / "check_models.log"
DEFAULT_TEMPERATURE: Final[float] = 0.1
DEFAULT_TIMEOUT: Final[float] = 300.0  # Default timeout in seconds
MAX_REASONABLE_TEMPERATURE: Final[float] = 2.0  # Warn if temperature exceeds this

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

    def fields(self) -> list[str]:
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


class ProcessImageParams(NamedTuple):
    """Parameters for processing an image with a model.

    Centralizes all parameters needed for model inference into a single
    immutable structure. This approach:
        - Reduces function signature complexity (single param vs many)
        - Makes parameter passing explicit and type-safe
        - Simplifies testing (one object to mock/construct)
        - Documents expected inputs clearly

    Note: Using NamedTuple instead of dataclass for lightweight immutability
    and automatic tuple unpacking support if needed.
    """

    model_identifier: str
    image_path: Path
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


# =============================================================================
# INFRASTRUCTURE: Timeouts, Colors, Logging
# =============================================================================


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

        if style_hint:
            return msg

        level_color: str = self.LEVEL_COLORS.get(record.levelno, "")

        if record.levelno == logging.INFO:
            return self._format_info_message(msg)

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

    def _style_error(self, raw_message: str, record: logging.LogRecord) -> str:
        prefix = getattr(record, "style_prefix", "ERROR")
        styled_prefix = Colors.colored(f"{prefix}:", Colors.BOLD, Colors.RED)
        return f"{styled_prefix} {raw_message}"

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

    def _format_info_message(self, msg: str) -> str:
        """Apply context-aware formatting to INFO messages for better visual hierarchy."""
        stripped: str = msg.strip()

        # Define format patterns with their corresponding colors (priority ordered)
        format_patterns: list[tuple[Callable[[str, str], bool], tuple[str, ...]]] = [
            # Section separators (highest priority)
            (
                lambda s, _: (
                    s.startswith(("===", "---"))
                    or (len(s) > MIN_SEPARATOR_CHARS and s.count("=") > MIN_SEPARATOR_CHARS)
                    or (len(s) > MIN_SEPARATOR_CHARS and s.count("-") > MIN_SEPARATOR_CHARS)
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

# Removed unused constants (DEFAULT_TIMEOUT_LONG, MB_CONVERSION, GB_CONVERSION, DISPLAY_WRAP_WIDTH)
# to reduce surface area; they had no runtime references.
DECIMAL_GB: Final[float] = 1_000_000_000.0  # Decimal GB (mlx-vlm already divides by 1e9)
MEM_BYTES_TO_GB_THRESHOLD: Final[float] = 1_000_000.0  # > ~1MB treat as raw bytes from mlx
MEGAPIXEL_CONVERSION: Final[float] = 1_000_000.0  # Convert pixels to megapixels

# EXIF/GPS coordinate standards: camera hardware stores GPS as degrees-minutes-seconds tuples
MAX_GPS_COORD_LEN: Final[int] = 3  # Full GPS coordinate tuple (degrees, minutes, seconds)
MED_GPS_COORD_LEN: Final[int] = 2  # GPS coordinate with 2 elements (degrees, minutes)
MIN_GPS_COORD_LEN: Final[int] = 1  # GPS coordinate with 1 element (degrees only)
MAX_TUPLE_LEN: Final[int] = 10
MAX_STR_LEN: Final[int] = 60
STR_TRUNCATE_LEN: Final[int] = 57
BASE_NAME_MAX_WIDTH: Final[int] = 45
COL_WIDTH: Final[int] = 12
MIN_NAME_COL_WIDTH: Final[int] = len("Model")

# Shared field metadata for reports
FIELD_UNITS: Final[dict[str, str]] = {
    "tokens": "(count)",
    "prompt_tokens": "(count)",
    "generation_tokens": "(count)",
    "prompt_tps": "(t/s)",
    "generation_tps": "(t/s)",
    "peak_memory": "(GB)",
}

FIELD_ABBREVIATIONS: Final[dict[str, tuple[str, str]]] = {
    "tokens": ("Tokens", "(ct)"),
    "prompt_tokens": ("Prompt", "(ct)"),
    "generation_tokens": ("Gen", "(ct)"),
    "total_tokens": ("Total", "Tokens"),
    "prompt_tps": ("Prompt", "(t/s)"),
    "generation_tps": ("Gen", "(t/s)"),
    "peak_memory": ("Peak", "(GB)"),
    "generation_time": ("Generation", "(s)"),
    "model_load_time": ("Load", "(s)"),
    "total_time": ("Total", "(s)"),
}

# Threshold for splitting long header text into multiple lines
HEADER_SPLIT_LENGTH = 10
ERROR_MESSAGE_PREVIEW_LEN: Final[int] = 40  # Max chars to show from error in summary line

# Fields that should be right-aligned (numeric fields)
NUMERIC_FIELD_PATTERNS: Final[frozenset[str]] = frozenset(
    {
        "tokens",
        "prompt_tokens",
        "generation_tokens",
        "prompt_tps",
        "generation_tps",
        "peak_memory",
        "generation_time",
        "model_load_time",
        "total_time",
    },
)

# Console table formatting constants
MAX_MODEL_NAME_LENGTH = 20  # Allows "microsoft/phi-3-vision" without truncation
MAX_OUTPUT_LENGTH = 28

# PerformanceResult timing fields - centralized definition
PERFORMANCE_TIMING_FIELDS: Final[list[str]] = ["generation_time", "model_load_time", "total_time"]


# =============================================================================
# FORMATTING UTILITIES (Numbers, Memory, Time, Tokens/sec, Field Values)
# =============================================================================


def fmt_num(val: float | str) -> str:
    """Format numbers consistently with thousands separators across all output formats."""
    try:
        fval = float(val)
        if abs(fval) >= LARGE_NUMBER_THRESHOLD:
            return f"{fval:,.0f}"
        # For integers or whole numbers, use comma separator if >= THOUSAND_THRESHOLD
        if fval == int(fval) and abs(fval) >= THOUSAND_THRESHOLD:
            return f"{int(fval):,}"
        if abs(fval) > 0:
            return f"{fval:.3g}"
        return str(val)
    except (ValueError, TypeError, OverflowError):
        return str(val)


def format_field_label(field_name: str) -> str:
    """Return a human-friendly label for a metric field name."""
    return field_name.replace("_", " ").title()


# Allowlist of inline HTML tags we preserve in Markdown output
# Keep <br> for line breaks; do NOT include 's' to avoid interpreting <s> tokens
# from model output as strikethrough.
allowed_inline_tags = {"br", "b", "strong", "i", "em", "code"}


def _escape_html_tags_selective(text: str) -> str:
    """Escape HTML-like tags except GitHub-allowed safe tags.

    Helper for markdown escaping functions. Neutralizes potentially unsafe HTML
    while preserving common formatting tags that GitHub recognizes.
    """
    tag_pattern = re.compile(r"</?[A-Za-z][A-Za-z0-9:-]*(?:\s+[^<>]*?)?>")

    def _escape_html_like(m: re.Match[str]) -> str:
        token = m.group(0)
        inner = token[1:-1].strip()
        if not inner:
            return token.replace("<", "&lt;").replace(">", "&gt;")
        core = inner.lstrip("/").split(None, 1)[0].rstrip("/").lower()
        if core in allowed_inline_tags:
            return token  # Keep recognized safe tag
        return token.replace("<", "&lt;").replace(">", "&gt;")

    return tag_pattern.sub(_escape_html_like, text)


def _format_memory_value_gb(num: float) -> str:
    """Format mixed-source memory value as GB string.

    Accepts raw bytes (mlx) or decimal GB (mlx-vlm). Returns a string without unit.
    """
    if num <= 0:
        return "0"
    gb: float = (num / DECIMAL_GB) if num > MEM_BYTES_TO_GB_THRESHOLD else num
    if gb >= MEMORY_GB_INTEGER_THRESHOLD:
        return f"{gb:,.0f}"
    if gb >= 1:
        return f"{gb:,.1f}"
    return f"{gb:.2f}"


def _format_time_seconds(num: float) -> str:
    """Format seconds with two decimals and trailing 's'."""
    return f"{num:.2f}s"


def _format_tps(num: float) -> str:
    """Format tokens-per-second with adaptive precision."""
    if abs(num) >= LARGE_NUMBER_THRESHOLD:
        return f"{num:,.0f}"
    if abs(num) >= MEDIUM_NUMBER_THRESHOLD:
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
    if total_seconds >= HOUR_THRESHOLD_SECONDS:
        return f"{_format_hms(total_seconds)} ({total_seconds:.2f}s)"
    return f"{total_seconds:.2f}s"


def _detect_repetitive_output(text: str, threshold: float = 0.8) -> tuple[bool, str | None]:
    """Detect if generated text is highly repetitive (like '<s>' repeated 500 times).

    Args:
        text: Generated text to check
        threshold: Fraction of text that must be repetitive to flag (default 0.8)

    Returns:
        Tuple of (is_repetitive, repeated_pattern)
        - is_repetitive: True if text exceeds repetition threshold
        - repeated_pattern: The repeated token/pattern if detected, None otherwise

    Examples:
        >>> _detect_repetitive_output("<s>" * 400)
        (True, '<s>')

        >>> _detect_repetitive_output("This is normal varied text")
        (False, None)
    """
    min_text_len = 10
    min_tokens = 5

    if not text or len(text) < min_text_len:
        return False, None

    # Split into tokens (words/symbols)
    tokens = text.split()
    if len(tokens) < min_tokens:
        return False, None

    # Count token frequency
    token_counts = Counter(tokens)

    # Find most common token
    if not token_counts:
        return False, None

    most_common_token, count = token_counts.most_common(1)[0]
    repetition_ratio = count / len(tokens)

    if repetition_ratio >= threshold:
        return True, most_common_token

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
    min_pipes_for_table = 4
    min_table_rows = 2
    min_mc_answers = 3
    substantial_text_len = 200

    issues: list[str] = []

    if not text:
        return issues

    text_lower = text.lower()

    # Check for markdown tables (pipe-delimited)
    if "|" in text and text.count("|") >= min_pipes_for_table:
        # Likely a table if we see multiple pipes
        lines_with_pipes = [line for line in text.split("\n") if "|" in line]
        if len(lines_with_pipes) >= min_table_rows:
            issues.append("Contains unexpected table")

    # Check for multiple choice patterns
    mc_pattern = re.compile(r"^[A-D]\)", re.MULTILINE)
    mc_matches = mc_pattern.findall(text)
    if len(mc_matches) >= min_mc_answers:
        issues.append("Contains multiple choice pattern")

    # Check for quiz/test questions
    question_indicators = ["what is", "how many", "based on the chart", "calculate"]
    has_question = any(indicator in text_lower for indicator in question_indicators)
    if has_question and len(text) > substantial_text_len:
        issues.append("Contains question/quiz content")

    # Check for unrelated educational content keywords
    edu_keywords = ["grade level", "students with adhd", "test scores", "homework"]
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
    verbose_threshold = 300
    min_meta_count = 2
    min_section_headers = 3

    if generated_tokens < verbose_threshold:
        return False

    text_lower = text.lower()

    # Check for meta-commentary patterns
    meta_patterns = [
        "the image depicts",
        "the image shows",
        "the photograph captures",
        "this image features",
        "in conclusion",
        "### analysis",
        "### conclusion",
        "based on the image",
    ]

    meta_count = sum(1 for pattern in meta_patterns if pattern in text_lower)

    # Check for excessive sectioning
    section_headers = text.count("###") + text.count("## ")

    # Verbose if has meta-commentary + sections or just too many sections
    return meta_count >= min_meta_count or section_headers >= min_section_headers


def _detect_formatting_violations(text: str) -> list[str]:
    """Detect formatting issues in generated output.

    Looks for:
    - HTML tags when not expected
    - Excessive markdown headers/structure
    - Bullet lists when simple text requested

    Args:
        text: Generated text to check

    Returns:
        List of detected formatting issues
    """
    max_headers = 5
    max_bullets = 15

    issues: list[str] = []

    if not text:
        return issues

    # Check for HTML tags (beyond simple breaks)
    html_tags = re.findall(r"<(?!br>|/br>)[a-z]+[^>]*>", text, re.IGNORECASE)
    if html_tags:
        issues.append(f"Contains HTML tags: {', '.join(set(html_tags[:3]))}")

    # Check for excessive markdown structure
    header_count = text.count("\n##") + text.count("\n###")
    if header_count > max_headers:
        issues.append(f"Excessive markdown headers ({header_count})")

    # Check for bullet lists when might not be appropriate
    bullet_prefixes = ("- ", "* ", "• ")
    bullet_lines = [
        line for line in text.split("\n") if line.strip().startswith(bullet_prefixes)
    ]
    if len(bullet_lines) > max_bullets:
        issues.append(f"Excessive bullet points ({len(bullet_lines)})")

    return issues


def local_now_str(fmt: str = "%Y-%m-%d %H:%M:%S %Z") -> str:
    """Return localized current time as a formatted string.

    Centralizes timestamp formatting so report generators and version info
    stay consistent and makes future changes (e.g. adding UTC or ISO8601
    variants) trivial.
    """
    return datetime.now(get_localzone()).strftime(fmt)


def format_field_value(field_name: str, value: MetricValue) -> str:  # noqa: PLR0911 - dispatcher with 8 return branches (memory/time/tps/numeric/string)
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
    if isinstance(value, (int, float)):
        num = float(value)
        if field_name.endswith("_memory"):
            return _format_memory_value_gb(num)
        if field_name.endswith("_tps"):
            return _format_tps(num)
        if field_name in {"total_time", "generation_time", "model_load_time"}:
            return _format_time_seconds(num)
        return fmt_num(num)
    if isinstance(value, str) and value:
        s: str = value.strip().replace(",", "")
        try:
            f = float(s)
        except ValueError:
            return value
        return format_field_value(field_name, f)
    return str(value)


def is_numeric_value(val: object) -> bool:
    """Return True if val can be interpreted as a number."""
    if isinstance(val, (int, float)):
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


def is_numeric_field(field_name: str) -> bool:
    """Check if a field should be treated as numeric (right-aligned)."""
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
        width = shutil.get_terminal_size(fallback=(GENERATION_WRAP_WIDTH, 24)).columns
    except OSError:
        width = GENERATION_WRAP_WIDTH
    return max(min_width, min(width, max_width))


def _log_wrapped_label_value(
    label: str,
    value: str,
    *,
    color: str = "",
    indent: int = 2,
) -> None:
    """Log a potentially long label/value pair wrapped to terminal width.

    The first line includes the label; subsequent lines are aligned under the value.
    Groups output into paragraphs to reduce log line count.
    """
    width = get_terminal_width(max_width=100)
    prefix = (" " * indent) + label
    first_avail = max(20, width - Colors.visual_len(prefix) - 1)
    cont_indent = " " * (indent + 2)
    cont_avail = max(20, width - len(cont_indent) - 1)

    # Preserve user newlines: wrap each input line independently
    # Group into paragraphs separated by blank lines
    lines = value.splitlines() or [""]
    output_lines: list[str] = []

    for li, original_line in enumerate(lines):
        # Skip empty lines to reduce noise
        if not original_line.strip():
            continue

        wrapped = textwrap.wrap(
            original_line,
            width=first_avail if li == 0 else cont_avail,
            break_long_words=False,
            break_on_hyphens=False,
            replace_whitespace=False,
        ) or [""]
        for wi, wline in enumerate(wrapped):
            if li == 0 and wi == 0:
                output_lines.append(f"{prefix} {Colors.colored(wline, color)}")
            else:
                output_lines.append(f"{cont_indent}{Colors.colored(wline, color)}")

    # Output as single multi-line log statement
    if output_lines:
        logger.info("\n".join(output_lines))


def _log_wrapped_error(label: str, value: str) -> None:
    """Log error with simple formatting for readability."""
    width = get_terminal_width(max_width=100)

    # Label
    logger.error(Colors.colored(f"{label}", Colors.RED, Colors.BOLD))

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
            logger.error(Colors.colored(f"{cont_indent}{wline}", Colors.RED))


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
    width: int = GENERATION_WRAP_WIDTH,
    *,
    char: str = "─",  # Unicode box-drawing character (was "-")
    color: str | None = None,
    bold: bool = False,
) -> None:
    """Log a horizontal rule line with optional color and bold.

    Uses unicode box-drawing characters for better visual separation.
    Keeps a single place for styling separators to ensure consistency.
    """
    line = char * max(1, width)
    extra: dict[str, object] = {"style_hint": LogStyles.RULE}
    if color:
        extra["style_color"] = color
    if bold:
        extra["style_bold"] = True
    logger.info(line, extra=extra)


# --- Utility Functions ---
# Ensure _pad_text is defined only once at module level and used everywhere


def _pad_text(text: str, width: int, *, right_align: bool = False) -> str:
    """Pad text to a given width, optionally right-aligning."""
    pad_len: int = width - Colors.visual_len(text)
    pad_len = max(pad_len, 0)
    pad_str: str = " " * pad_len
    return (pad_str + text) if right_align else (text + pad_str)


def get_library_versions() -> LibraryVersionDict:
    """Return versions of key libraries as a dictionary, using None for missing."""

    def _none_if_na(v: object) -> str | None:
        s = str(v) if v is not None else ""
        s_norm = s.strip()
        if not s_norm:
            return None
        if s_norm == NOT_AVAILABLE or s_norm.startswith("N/A"):
            return None
        return s_norm

    return {
        "mlx": _none_if_na(getattr(mx, "__version__", None)),
        "mlx-vlm": _none_if_na(vlm_version) if "vlm_version" in globals() else None,
        "mlx-lm": _none_if_na(mlx_lm_version),
        "huggingface-hub": _none_if_na(hf_version),
        "transformers": _none_if_na(transformers_version),
        "tokenizers": _none_if_na(tokenizers_version),
        "Pillow": _none_if_na(pillow_version),
    }


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
    """Print library versions and optionally system / hardware info.

    If running on Apple Silicon (arm64 macOS) we append a concise hardware
    block (GPU name, RAM, physical CPU cores, GPU cores). Otherwise we note
    that the extended block is skipped. Errors are swallowed so version
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

    # --- Optional system / hardware block (Apple Silicon focus) ---
    try:
        is_arm64: bool = platform.machine() == "arm64"
        if not is_arm64:
            logger.info("Not running on Apple Silicon (system info block skipped).")
            return

        device_info = get_device_info() or {}
        displays = device_info.get("SPDisplaysDataType") or []
        gpu_name: str | None = None
        gpu_cores: str | int | None = None
        if displays and isinstance(displays, list):
            first = displays[0]
            if isinstance(first, dict):
                gpu_name = first.get("_name")
                gpu_cores = first.get("sppci_cores")

        if psutil is None:  # safety if optional import failed
            logger.debug("psutil not available; skipping extended system info block.")
            return

        ram_gb = psutil.virtual_memory().total / (1024**3)
        physical_cores = psutil.cpu_count(logical=False) or 0

        sys_lines = [
            "",  # spacer
            "--- System Information ---",
            f"macOS:        v{platform.mac_ver()[0]}",
            f"Python:       v{sys.version.split()[0]}",
            "",
            "Hardware:",
            f"• Chip:        {gpu_name or 'Unknown'}",
            f"• RAM:         {ram_gb:.1f} GB",
            f"• CPU Cores:   {physical_cores}",
            f"• GPU Cores:   {gpu_cores if gpu_cores is not None else 'Unknown'}",
        ]
        for line in sys_lines:
            logger.info(line)
    except Exception as err:  # noqa: BLE001 - system info is non-critical, intentionally catch all failure modes
        logger.debug("Skipping system info block: %s", err)


# =============================================================================
# IMAGE & EXIF METADATA PROCESSING (File handling, GPS, EXIF extraction)
# =============================================================================


# --- File Handling ---
# Simplified the `find_most_recent_file` function by using `max` with a generator.
def find_most_recent_file(folder: PathLike) -> Path | None:
    """Return the most recently modified file in a folder, or None.

    Scans for regular files (excluding hidden files starting with '.') and
    returns the one with the most recent modification time.
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        logger.error("Provided path is not a directory: %s", folder_path)
        return None

    try:
        # Find all regular files, excluding hidden files (starting with '.')
        regular_files = [
            f for f in folder_path.iterdir() if f.is_file() and not f.name.startswith(".")
        ]

        # Return the most recently modified file, or None if no files found
        most_recent: Path | None = max(
            regular_files,
            key=lambda f: f.stat().st_mtime,
            default=None,
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
        logger.debug("Most recent file found: %s", str(most_recent))
        return most_recent

    logger.debug("No files found in directory: %s", folder_path)
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
def _process_ifd0(exif_raw: Mapping[object, object]) -> ExifDict:
    exif_decoded: ExifDict = {}
    for tag_id, value in exif_raw.items():
        # Skip SubIFD pointers, we'll handle them separately
        if tag_id in (ExifTags.Base.ExifOffset, ExifTags.Base.GPSInfo):
            continue
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
    img_path = Path(image_path)
    try:
        with Image.open(img_path) as img:
            exif_raw: Any = img.getexif()
            if not exif_raw:
                logger.debug("No EXIF data found in %s", img_path)
                return None
            exif_decoded: ExifDict = _process_ifd0(exif_raw)
            exif_decoded.update(_process_exif_subifd(exif_raw))
            gps_decoded = _process_gps_ifd(exif_raw)
            if gps_decoded:
                exif_decoded["GPSInfo"] = gps_decoded
            return exif_decoded
    except (FileNotFoundError, UnidentifiedImageError):
        logger.exception(
            Colors.colored(f"Error reading image file: {img_path}", Colors.YELLOW),
        )
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


def _extract_description(exif_data: ExifDict) -> str | None:
    description = exif_data.get("ImageDescription")
    if description is None:
        return None
    if isinstance(description, bytes):
        try:
            desc = description.decode("utf-8", errors="replace").strip()
        except UnicodeDecodeError as err:
            desc = str(description)
            logger.debug("Failed to decode description: %s", err)
    else:
        desc = str(description).strip()
    return desc or None


def _extract_gps_str(gps_info_raw: Mapping[object, Any] | None) -> str | None:
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
            lat_ref.decode("ascii", errors="replace")
            if isinstance(lat_ref, bytes)
            else str(lat_ref)
        )
        lon_ref_str: str = (
            lon_ref.decode("ascii", errors="replace")
            if isinstance(lon_ref, bytes)
            else str(lon_ref)
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

    # Date, Description, GPS
    metadata["date"] = _extract_exif_date(img_path, exif_data)
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
        # Use latin-1 as a robust decoder. It can decode any byte, preventing errors,
        # though it may not render all international characters correctly.
        # This prioritizes stability and alignment over perfect character representation.
        try:
            processed_str = _sanitize(value.decode("latin-1", errors="replace"))
        except (UnicodeDecodeError, AttributeError):
            return f"<bytes len={len(value)} un-decodable>"
    elif isinstance(value, (tuple, list)) and len(value) > MAX_TUPLE_LEN:
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
        logger.warning(
            Colors.colored("No relevant EXIF tags found to display.", Colors.YELLOW),
        )
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
    logger.info(Colors.colored(f"{title.center(header_width)}", Colors.BOLD, Colors.MAGENTA))
    log_rule(header_width, char="=", color=Colors.BLUE, bold=True)

    # Print the tabulated table
    for line in table_lines:
        logger.info(line)
    log_rule(header_width, char="=", color=Colors.BLUE, bold=True)


# =============================================================================
# TABLE & REPORT GENERATION (CLI Tables, Field Formatting, Aggregation)
# =============================================================================


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
                f.name for f in fields(r.generation) if f.name not in ("text", "logprobs")
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


# Additional constants for multi-line formatting
MAX_SHORT_NAME_LENGTH: int = 16  # Increased from 12 to show more of model names
MAX_LONG_NAME_LENGTH: int = 32  # Increased from 24 for longer model names
MAX_SIMPLE_NAME_LENGTH: int = 24  # Increased from 20
MAX_OUTPUT_LINE_LENGTH: int = 60  # Increased from 35 to show more error details
MAX_OUTPUT_TOTAL_LENGTH: int = 120  # Increased from 70 for complete error messages


def _format_model_name_multiline(model_name: str) -> str:
    """Format model name for multi-line display in console table."""
    if "/" in model_name:
        parts = model_name.split("/")
        if len(parts[-1]) > MAX_SHORT_NAME_LENGTH:  # If final part is long, break it
            final_part = parts[-1]
            if len(final_part) > MAX_LONG_NAME_LENGTH:  # Very long, truncate with ellipsis
                # Use MAX_LONG_NAME_LENGTH - 3 for the ellipsis
                return f"{'/'.join(parts[:-1])}/\n{final_part[: MAX_LONG_NAME_LENGTH - 3]}..."
            # Moderate length, wrap
            mid_point = len(final_part) // 2
            return f"{'/'.join(parts[:-1])}/\n{final_part[:mid_point]}-\n{final_part[mid_point:]}"
        return model_name
    if len(model_name) > MAX_SIMPLE_NAME_LENGTH:
        # No slash, just split if too long
        mid_point = len(model_name) // 2
        return f"{model_name[:mid_point]}\n{model_name[mid_point:]}"
    return model_name


def _format_output_multiline(output_text: str) -> str:
    """Format output text for multi-line display in console table."""
    # Clean newlines
    output_text = re.sub(r"[\n\r]", " ", output_text)

    if len(output_text) <= MAX_OUTPUT_LINE_LENGTH:
        return output_text

    # Find a good break point around the middle
    mid_point = min(MAX_OUTPUT_LINE_LENGTH, len(output_text) // 2)
    # Try to break at word boundary near mid point
    space_pos = output_text.find(" ", mid_point)
    if space_pos != -1 and space_pos < len(output_text) * 0.7:  # Break at space if reasonable
        line1 = output_text[:space_pos]
        line2 = output_text[space_pos + 1 :]
        if len(line2) > MAX_OUTPUT_LINE_LENGTH:  # Second line too long
            line2 = line2[: MAX_OUTPUT_LINE_LENGTH - 3] + "..."
        return f"{line1}\n{line2}"
    # No good break point, just split at reasonable length
    truncated = "..." if len(output_text) > MAX_OUTPUT_TOTAL_LENGTH else ""
    return (
        f"{output_text[:MAX_OUTPUT_LINE_LENGTH]}\n"
        f"{output_text[MAX_OUTPUT_LINE_LENGTH:MAX_OUTPUT_TOTAL_LENGTH]}"
        f"{truncated}"
    )


def _build_stat_headers(all_fields: list[str]) -> tuple[list[str], list[str]]:
    """Return (headers, field_names) for the stats table.

    headers contain multi-line abbreviated labels. field_names maintains the
    underlying attribute order for alignment & width rules.
    """
    headers: list[str] = ["Model"]
    field_names: list[str] = ["model"]
    for field in all_fields:
        if field in FIELD_ABBREVIATIONS:
            line1, line2 = FIELD_ABBREVIATIONS[field]
            label = f"{line1}\n{line2}"
        else:
            base_label = format_field_label(field)
            label = f"{base_label}\n{FIELD_UNITS[field]}" if field in FIELD_UNITS else base_label
        headers.append(label)
        field_names.append(field)
    headers.append("Output")
    field_names.append("output")
    return headers, field_names


Numberish = int | float | str


def _format_numeric_display(val: Numberish) -> str:
    """Format numeric-ish values with commas / precision heuristics.

    Non-numeric inputs should be pre-converted to str before calling.
    This is a thin wrapper around fmt_num for consistency.
    """
    return fmt_num(val)


def _build_stat_row(result: PerformanceResult, all_fields: list[str]) -> list[str]:
    """Build a single table row for the stats table."""
    model_display = _format_model_name_multiline(str(result.model_name))
    model_colored = (
        Colors.colored(model_display, Colors.RED)
        if not result.success
        else Colors.colored(model_display, Colors.MAGENTA)
    )
    row: list[str] = [model_colored]
    for field in all_fields:
        raw_val = _get_field_value(result, field)
        formatted = format_field_value(field, raw_val)
        # Ensure we only pass supported types
        fmt_input: Numberish = (
            formatted if isinstance(formatted, (int, float, str)) else str(formatted)
        )
        row.append(_format_numeric_display(fmt_input))
    if result.success and result.generation:
        output_text = str(getattr(result.generation, "text", ""))
    else:
        output_text = result.error_message or result.captured_output_on_fail or "-"
    row.append(_format_output_multiline(output_text))
    return row


def _compute_column_widths(field_names: list[str]) -> list[int | None]:
    """Compute per-column width hints based on field naming heuristics.

    Returns a list with explicit widths for critical columns (model, output)
    and None for numeric columns to enable tabulate's intelligent auto-sizing.
    """
    widths: list[int | None] = []
    term_w = get_terminal_width()
    # Allocate generous space for output/error messages - prioritize readability
    out_w = max(60, min(120, int(term_w * 0.6)))
    for idx, name in enumerate(field_names):
        if idx == 0:  # Model column - keep explicit width for consistency
            widths.append(35)
        elif name == "output":
            widths.append(out_w)
        else:
            # Let tabulate auto-size numeric and other columns intelligently
            widths.append(None)
    return widths


@dataclass
class ModelIssuesSummary:
    """Summary of model performance issues categorized by type."""

    crashed: list[str]  # Models that failed with errors/OOM
    repetitive: list[str]  # Models with highly repetitive output
    succinct: list[str]  # Models with very short output relative to prompt
    hallucinated: list[str]  # Models with detected hallucination patterns
    verbose: list[str]  # Models with excessively verbose/over-structured output
    formatting_issues: list[str]  # Models with HTML/markdown formatting violations
    successful: list[str]  # Models that completed successfully without issues


def analyze_model_issues(results: list[PerformanceResult]) -> ModelIssuesSummary:
    """Analyze results and categorize models by performance issues.

    Args:
        results: List of model performance results

    Returns:
        ModelIssuesSummary with categorized model lists
    """
    crashed: list[str] = []
    repetitive: list[str] = []
    succinct: list[str] = []
    hallucinated: list[str] = []
    verbose: list[str] = []
    formatting_issues: list[str] = []
    successful: list[str] = []

    succinct_threshold = 0.3  # Output < 30% of prompt length

    for result in results:
        model_name = result.model_name.split("/")[-1]

        # Check for crashes/failures
        if not result.success:
            crashed.append(model_name)
            continue

        # Analyze successful models
        gen_text = str(getattr(result.generation, "text", "")) if result.generation else ""
        gen_tokens = (
            getattr(result.generation, "generation_tokens", 0) if result.generation else 0
        )

        # Check for repetitive output
        is_repetitive, repeated_token = _detect_repetitive_output(gen_text)
        if is_repetitive:
            repetitive.append(f"{model_name} ({repeated_token})")
            continue

        # Check for hallucinations
        hallucination_issues = _detect_hallucination_patterns(gen_text)
        if hallucination_issues:
            issue_desc = ", ".join(hallucination_issues[:2])  # First 2 issues
            hallucinated.append(f"{model_name} ({issue_desc})")
            continue

        # Check for excessive verbosity
        if _detect_excessive_verbosity(gen_text, gen_tokens):
            verbose.append(f"{model_name} ({gen_tokens} tokens)")
            continue

        # Check for formatting violations
        format_issues = _detect_formatting_violations(gen_text)
        if format_issues:
            issue_desc = format_issues[0]  # First issue
            formatting_issues.append(f"{model_name} ({issue_desc})")
            continue

        # Check for overly succinct output
        prompt_tokens = getattr(result.generation, "prompt_tokens", 0) if result.generation else 0

        if prompt_tokens > 0 and gen_tokens < (prompt_tokens * succinct_threshold):
            succinct.append(f"{model_name} ({gen_tokens} tokens)")
            continue

        # If none of the above, it's successful
        successful.append(model_name)

    return ModelIssuesSummary(
        crashed=crashed,
        repetitive=repetitive,
        succinct=succinct,
        hallucinated=hallucinated,
        verbose=verbose,
        formatting_issues=formatting_issues,
        successful=successful,
    )


def format_issues_summary_text(summary: ModelIssuesSummary) -> str:
    """Format issues summary as plain text for CLI and Markdown.

    Args:
        summary: Analyzed model issues summary

    Returns:
        Formatted text summary
    """
    max_shown = 5
    lines: list[str] = []
    total = (
        len(summary.crashed)
        + len(summary.repetitive)
        + len(summary.succinct)
        + len(summary.hallucinated)
        + len(summary.verbose)
        + len(summary.formatting_issues)
        + len(summary.successful)
    )

    lines.append("## 📊 Model Performance Summary")
    lines.append("")
    lines.append(f"**Total Models Evaluated:** {total}")
    lines.append("")

    if summary.successful:
        lines.append(f"✅ **Successful ({len(summary.successful)}):**")
        lines.append("   Clean output, no issues detected")
        lines.extend([f"  • {model}" for model in summary.successful[:max_shown]])
        if len(summary.successful) > max_shown:
            lines.append(f"  • ...and {len(summary.successful) - max_shown} more")
        lines.append("")

    if summary.crashed:
        lines.append(f"❌ **Crashed/Failed ({len(summary.crashed)}):**")
        lines.append("   Errors, OOM, or other failures")
        lines.extend([f"  • {model}" for model in summary.crashed])
        lines.append("")

    if summary.repetitive:
        lines.append(f"🔁 **Repetitive Output ({len(summary.repetitive)}):**")
        lines.append("   Generated highly repetitive or garbage tokens")
        lines.extend([f"  • {model}" for model in summary.repetitive])
        lines.append("")

    if summary.hallucinated:
        lines.append(f"🤔 **Possible Hallucinations ({len(summary.hallucinated)}):**")
        lines.append("   Detected unexpected content patterns")
        lines.extend([f"  • {model}" for model in summary.hallucinated])
        lines.append("")

    if summary.verbose:
        lines.append(f"📚 **Excessively Verbose ({len(summary.verbose)}):**")
        lines.append("   Over-structured output with meta-commentary")
        lines.extend([f"  • {model}" for model in summary.verbose])
        lines.append("")

    if summary.formatting_issues:
        lines.append(f"🎨 **Formatting Issues ({len(summary.formatting_issues)}):**")
        lines.append("   Unexpected HTML/markdown structure")
        lines.extend([f"  • {model}" for model in summary.formatting_issues])
        lines.append("")

    if summary.succinct:
        lines.append(f"📝 **Very Succinct ({len(summary.succinct)}):**")
        lines.append("   Generated output much shorter than prompt")
        lines.extend([f"  • {model}" for model in summary.succinct])
        lines.append("")

    return "\n".join(lines)


def format_issues_summary_html(summary: ModelIssuesSummary) -> str:
    """Format issues summary as HTML for reports.

    Args:
        summary: Analyzed model issues summary

    Returns:
        Formatted HTML summary
    """
    max_shown = 5
    lines: list[str] = []
    total = (
        len(summary.crashed)
        + len(summary.repetitive)
        + len(summary.succinct)
        + len(summary.hallucinated)
        + len(summary.verbose)
        + len(summary.formatting_issues)
        + len(summary.successful)
    )

    lines.append("<div class='issues-summary'>")
    lines.append("<h3>📊 Model Performance Summary</h3>")
    lines.append(f"<p><strong>Total Models Evaluated:</strong> {total}</p>")

    if summary.successful:
        lines.append(f"<h4>✅ Successful ({len(summary.successful)})</h4>")
        lines.append("<p>Clean output, no issues detected</p>")
        lines.append("<ul>")
        lines.extend([f"<li>{html.escape(m)}</li>" for m in summary.successful[:max_shown]])
        if len(summary.successful) > max_shown:
            remaining = len(summary.successful) - max_shown
            lines.append(f"<li><em>...and {remaining} more</em></li>")
        lines.append("</ul>")

    if summary.crashed:
        lines.append(
            f"<h4 style='color: #d32f2f;'>❌ Crashed/Failed ({len(summary.crashed)})</h4>",
        )
        lines.append("<p>Errors, OOM, or other failures</p>")
        lines.append("<ul>")
        lines.extend([f"<li>{html.escape(m)}</li>" for m in summary.crashed])
        lines.append("</ul>")

    if summary.repetitive:
        lines.append(
            f"<h4 style='color: #ff9800;'>🔁 Repetitive Output "
            f"({len(summary.repetitive)})</h4>",
        )
        lines.append("<p>Generated highly repetitive or garbage tokens</p>")
        lines.append("<ul>")
        lines.extend([f"<li>{html.escape(m)}</li>" for m in summary.repetitive])
        lines.append("</ul>")

    if summary.hallucinated:
        lines.append(
            f"<h4 style='color: #ff9800;'>🤔 Possible Hallucinations "
            f"({len(summary.hallucinated)})</h4>",
        )
        lines.append("<p>Detected unexpected content patterns</p>")
        lines.append("<ul>")
        lines.extend([f"<li>{html.escape(m)}</li>" for m in summary.hallucinated])
        lines.append("</ul>")

    if summary.verbose:
        lines.append(
            f"<h4 style='color: #9c27b0;'>📚 Excessively Verbose "
            f"({len(summary.verbose)})</h4>",
        )
        lines.append("<p>Over-structured output with meta-commentary</p>")
        lines.append("<ul>")
        lines.extend([f"<li>{html.escape(m)}</li>" for m in summary.verbose])
        lines.append("</ul>")

    if summary.formatting_issues:
        lines.append(
            f"<h4 style='color: #795548;'>🎨 Formatting Issues "
            f"({len(summary.formatting_issues)})</h4>",
        )
        lines.append("<p>Unexpected HTML/markdown structure</p>")
        lines.append("<ul>")
        lines.extend([f"<li>{html.escape(m)}</li>" for m in summary.formatting_issues])
        lines.append("</ul>")

    if summary.succinct:
        lines.append(
            f"<h4 style='color: #2196f3;'>📝 Very Succinct "
            f"({len(summary.succinct)})</h4>",
        )
        lines.append("<p>Generated output much shorter than prompt</p>")
        lines.append("<ul>")
        lines.extend([f"<li>{html.escape(m)}</li>" for m in summary.succinct])
        lines.append("</ul>")

    lines.append("</div>")
    lines.append("<hr>")

    return "\n".join(lines)


def print_model_stats(results: list[PerformanceResult]) -> None:
    """Emit an enhanced table of per-model metrics with visual indicators and summary.

    (Refactored) The original monolithic implementation has been decomposed
    into small helpers: header building, row construction, column width
    calculation. Now enhanced with unicode borders, success/fail indicators,
    and summary statistics.
    """
    if not results:
        logger.info("No model results to display.")
        return

    rs = ResultSet(results)
    all_fields = rs.fields() or []
    headers, field_names = _build_stat_headers(all_fields)

    # Add visual indicators to rows
    success_count = sum(1 for r in rs.results if r.success)
    fail_count = len(rs.results) - success_count

    rows = []
    for r in rs.results:
        row_data = _build_stat_row(r, all_fields)
        # Prepend success/fail indicator to model name
        indicator = "✓" if r.success else "✗"
        row_data[0] = f"{indicator} {row_data[0]}"
        rows.append(row_data)

    colalign: list[str] = ["left"] + [
        "right" if is_numeric_field(fname) else "left" for fname in field_names[1:]
    ]
    widths = _compute_column_widths(field_names)
    table = tabulate(
        rows,
        headers=headers,
        tablefmt="plain",
        colalign=colalign,
        maxcolwidths=widths,
    )
    lines = table.split("\n")

    # Use terminal width for framing rules
    max_width = get_terminal_width(max_width=100)

    # Top border with title
    logger.info("")
    log_rule(max_width, char="═", color=Colors.BLUE, bold=True)
    title = "PERFORMANCE SUMMARY"
    padding = (max_width - len(title)) // 2
    logger.info(Colors.colored(" " * padding + title, Colors.BOLD, Colors.CYAN))
    log_rule(max_width, char="═", color=Colors.BLUE, bold=True)

    # Table content
    for line in lines:
        logger.info(line)

    # Bottom border with summary
    log_rule(max_width, char="─", color=Colors.BLUE, bold=False)
    summary = f"Total Models: {len(rs.results)}  │  "
    summary += Colors.colored(f"✓ Pass: {success_count}", Colors.GREEN) + "  "
    if fail_count > 0:
        summary += Colors.colored(f"✗ Fail: {fail_count}", Colors.RED)
    logger.info(summary)
    log_rule(max_width, char="═", color=Colors.BLUE, bold=True)

    logger.info("Results sorted: errors first, then by generation time (fastest to slowest).")


def _prepare_table_data(
    results: list[PerformanceResult],
) -> tuple[list[str], list[list[str]], list[str]]:
    """Normalize model results into (headers, rows, field_names) for reports.

    Separation of concerns: this function handles data shaping only; rendering
    (HTML / Markdown / console) is delegated so that alternate output formats
    can reuse the same pre-processed structure.
    """
    if not results:
        return [], [], []

    rs = ResultSet(results)
    results = rs.results
    all_fields = rs.fields()

    headers = ["Model"]
    field_names = ["model"]  # Track original field names for alignment

    # Create more compact, multi-line friendly headers with consistent formatting
    compact_headers = {
        "tokens": "Total<br>Tokens",
        "prompt_tokens": "Prompt<br>Tokens",
        "generation_tokens": "Generated<br>Tokens",
        "prompt_tps": "Prompt<br>Speed<br>(t/s)",
        "generation_tps": "Generation<br>Speed<br>(t/s)",
        "peak_memory": "Peak<br>Memory<br>(GB)",
        "active_memory": "Active<br>Memory<br>(GB)",
        "cached_memory": "Cached<br>Memory<br>(GB)",
        "generation_time": "Generation<br>Time<br>(s)",
        "model_load_time": "Model<br>Load<br>(s)",
        "total_time": "Total<br>Time<br>(s)",
    }

    for f in all_fields:
        if f in compact_headers:
            headers.append(compact_headers[f])
        else:
            # Fallback to formatted label with units
            label = format_field_label(f)
            if f in FIELD_UNITS:
                # Split long headers into multiple lines
                parts = label.split()
                if len(parts) > 1 and len(label) > HEADER_SPLIT_LENGTH:
                    label = "<br>".join(parts) + f"<br>{FIELD_UNITS[f]}"
                else:
                    label += f" {FIELD_UNITS[f]}"
            headers.append(label)
        field_names.append(f)

    headers.append("Output /<br>Diagnostics")
    field_names.append("output")

    # Build table rows
    rows = []
    for r in results:
        row = [str(r.model_name)]

        # Add generation fields and performance timing fields
        for f in all_fields:
            val = _get_field_value(r, f)
            val = format_field_value(f, val)
            row.append(str(val))

        # Add output/diagnostic column (empty string as unified fallback)
        if r.success and r.generation:
            out_val = str(getattr(r.generation, "text", ""))
        else:
            out_val = r.error_message or r.captured_output_on_fail or ""
        row.append(out_val)

        rows.append(row)

    return headers, rows, field_names


# --- HTML Report Generation ---
def _mark_failed_rows_in_html(html_table: str, results: list[PerformanceResult]) -> str:
    """Add class="failed-row" to <tr> elements whose corresponding result failed."""
    sorted_results_for_flags = _sort_results_by_time(results)
    failed_set = {idx for idx, r in enumerate(sorted_results_for_flags) if not r.success}
    if not failed_set or "<tbody>" not in html_table:
        return html_table
    tbody_start = html_table.find("<tbody>")
    tbody_end = html_table.find("</tbody>", tbody_start)
    if tbody_start == -1 or tbody_end == -1:
        return html_table
    body_html = html_table[tbody_start:tbody_end]
    row_index = -1

    def _row_replacer(match: re.Match[str]) -> str:
        nonlocal row_index
        row_index += 1
        return '<tr class="failed-row">' if row_index in failed_set else match.group(0)

    body_html = re.sub(r"<tr>", _row_replacer, body_html)
    return html_table[:tbody_start] + body_html + html_table[tbody_end:]


def _build_full_html_document(
    *,
    html_table: str,
    versions: LibraryVersionDict,
    prompt: str,
    total_runtime_seconds: float,
    issues_summary_html: str = "",
) -> str:
    local_tz = get_localzone()
    # Build complete HTML document
    html_content = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>MLX VLM Performance Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px; background-color: #f8f9fa; color: #212529; line-height: 1.6;
        }}
        h1 {{
            color: #495057; text-align: center; margin-bottom: 15px;
            border-bottom: 3px solid #007bff; padding-bottom: 15px; font-size: 2.2em;
        }}
        h2 {{
            color: #495057; margin-top: 30px; margin-bottom: 15px;
            border-bottom: 2px solid #6c757d; padding-bottom: 8px; font-size: 1.4em;
        }}
        .prompt-section {{
            background-color: #e3f2fd; border-left: 4px solid #2196f3; padding: 20px;
            margin: 25px 0; border-radius: 6px;
        }}
        .prompt-section h3 {{
            color: #1976d2; margin-top: 0; margin-bottom: 10px;
            font-size: 1.1em;
        }}
        .meta-info {{ color: #6c757d; font-style: italic; margin: 15px 0; text-align: center; }}
        table {{
            border-collapse: collapse; width: 95%; margin: 30px auto; background-color: #fff;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1); border-radius: 8px; overflow: hidden;
        }}
        th, td {{ border: 1px solid #dee2e6; padding: 8px 12px; vertical-align: top; }}
        thead th, tbody td {{ vertical-align: top; }}
        th {{
            background: linear-gradient(135deg, #e9ecef 0%, #f8f9fa 100%);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-weight: 600;
            color: #495057; text-shadow: 0 1px 0 white; font-size: 14px; text-align: center;
        }}
        th.numeric {{ text-align: right; }}
        th.text {{ text-align: left; }}
        tr:nth-child(even):not(.failed-row) {{ background-color: #f8f9fa; }}
        tr:hover:not(.failed-row) {{
            background-color: #e3f2fd; transition: background-color 0.2s;
        }}
        tr.failed-row {{ background-color: #f8d7da !important; color: #721c24; }}
        tr.failed-row:hover {{ background-color: #f5c6cb !important; }}
        .model-name {{
            font-family: 'Courier New', Courier, monospace; font-weight: 500;
            text-align: left; color: #0d6efd;
        }}
        .error-message {{ font-weight: bold; color: #721c24; }}
        td.numeric {{ text-align: right; font-family: 'Courier New', monospace; }}
        td.text {{ text-align: left; }}
        caption {{ font-style: italic; color: #6c757d; margin-bottom: 10px; }}
        footer {{ margin-top: 40px; padding-top: 20px; border-top: 2px solid #dee2e6; }}
        footer h2 {{ color: #495057; }}
        footer ul {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
        footer code {{
            background-color: #e9ecef; padding: 2px 4px; border-radius: 3px;
            color: #d63384;
        }}
    </style>
</head>
<body>
    <h1>MLX Vision Language Model Performance Report</h1>
    {issues_summary_html}
    <div class=\"prompt-section\">
        <h3>📝 Test Prompt</h3>
        <div>{html.escape(prompt).replace("\n", "<br>")}</div>
    </div>
    <h2>📊 Performance Results</h2>
    <div class=\"meta-info\">
        Performance metrics and output for Vision Language Model processing<br>
        Results sorted: errors first, then by generation time (fastest to slowest) • Generated on
        {datetime.now(local_tz).strftime("%Y-%m-%d %H:%M:%S %Z")} • Failures shown but
        excluded from averages<br>
        Overall runtime: {format_overall_runtime(total_runtime_seconds)}
    </div>
    {html_table}
    <footer>
        <h2>🔧 System Information</h2>
        <ul>\n"""

    for name, ver in sorted(versions.items()):
        ver_str = "" if ver is None else ver
        html_content += (
            f"            <li><code>{html.escape(name)}</code>: "
            f"<code>{html.escape(ver_str)}</code></li>\n"
        )

    generated_ts = datetime.now(local_tz).strftime("%Y-%m-%d %H:%M:%S %Z")
    html_content += (
        "        </ul>\n"
        f"        <p><em>Report generated: {generated_ts}</em></p>\n"
        "    </footer>\n"
        "</body>\n"
        "</html>"
    )
    return html_content


# =============================================================================
# HTML REPORT GENERATION (HTML formatting, styling, templates)
# =============================================================================


def generate_html_report(
    results: list[PerformanceResult],
    filename: Path,
    versions: LibraryVersionDict,
    prompt: str,
    total_runtime_seconds: float,
) -> None:
    """Write an HTML report (standalone) including metrics table and context.

    Inline CSS is used deliberately to keep the artifact portable (single file
    shareable via email / chat) without additional assets. If styling grows
    further, moving to a template system + external stylesheet would be the
    next step.
    """
    if not results:
        logger.warning(
            Colors.colored("No results to generate HTML report.", Colors.YELLOW),
        )
        return

    # Get table data using our helper function
    headers, rows, field_names = _prepare_table_data(results)

    if not headers or not rows:
        logger.warning("No table data to generate HTML report.")
        return

    # Escape HTML characters for safety (all user-controlled content: model names, output)
    for i in range(len(rows)):
        for j in range(len(rows[i])):
            # Escape first column (model name) and last column (output text)
            if j == 0 or j == len(rows[i]) - 1:
                rows[i][j] = html.escape(str(rows[i][j]))

    # Determine column alignment using original field names
    colalign = ["left"] + [
        "right" if is_numeric_field(field_name) else "left" for field_name in field_names[1:]
    ]

    # Generate HTML table using tabulate
    html_table = tabulate(
        rows,
        headers=headers,
        tablefmt="unsafehtml",
        colalign=colalign,
    )
    # Analyze model issues and generate summary
    summary = analyze_model_issues(results)
    issues_html = format_issues_summary_html(summary)

    # Mark failed rows and build the final document
    html_table = _mark_failed_rows_in_html(html_table, results)
    html_content = _build_full_html_document(
        html_table=html_table,
        versions=versions,
        prompt=prompt,
        total_runtime_seconds=total_runtime_seconds,
        issues_summary_html=issues_html,
    )

    try:
        with filename.open("w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(
            "HTML report saved to: %s",
            Colors.colored(str(filename.resolve()), Colors.GREEN),
        )
    except OSError:
        logger.exception(
            "Failed to write HTML report to file %s.",
            str(filename),
        )
    except ValueError:
        logger.exception(
            "A value error occurred while writing HTML report %s",
            str(filename),
        )


# =============================================================================
# MARKDOWN REPORT GENERATION (Markdown tables, formatting)
# =============================================================================


def generate_markdown_report(
    results: list[PerformanceResult],
    filename: Path,
    versions: LibraryVersionDict,
    prompt: str,
    total_runtime_seconds: float,
) -> None:
    """Write a GitHub-friendly Markdown summary with aligned pipe table."""
    if not results:
        logger.warning(
            Colors.colored("No results to generate Markdown report.", Colors.YELLOW),
        )
        return

    # Get table data using our helper function
    headers, rows, field_names = _prepare_table_data(results)

    if not headers or not rows:
        logger.warning("No table data to generate Markdown report.")
        return

    # For Markdown, we need to process headers to remove HTML breaks and use simpler formatting
    markdown_headers = []
    for header in headers:
        # Replace <br> with space for Markdown compatibility
        clean_header = header.replace("<br>", " ")
        markdown_headers.append(clean_header)

    # Escape Markdown only for diagnostics (failed rows). Keep successful model output
    # unchanged. This preserves model formatting (including *, _, `, etc.) while
    # avoiding table breakage from diagnostics.
    sorted_results_for_flags = _sort_results_by_time(results)
    for i in range(len(rows)):
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

    # Analyze model issues and generate summary
    summary = analyze_model_issues(results)
    issues_text = format_issues_summary_text(summary)

    # Build the complete markdown content
    md: list[str] = []
    md.append("# Model Performance Results\n")
    md.append(f"_Generated on {local_now_str()}_\n")
    md.append("")
    # Add issues summary before prompt
    if issues_text:
        md.append(issues_text)
        md.append("")
    md.append("> **Prompt used:**\n>\n> " + prompt.replace("\n", "\n> ") + "\n")
    md.append("")
    note = "**Note:** Results sorted: errors first, then by generation time (fastest to slowest).\n"
    md.append(note)
    md.append(f"**Overall runtime:** {format_overall_runtime(total_runtime_seconds)}\n")
    md.append("")
    # Surround the table with markdownlint rule guards; the table can be wide and may
    # contain HTML breaks and model-generated emphasis styles
    md.append("<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->")
    md.append(markdown_table)
    md.append("<!-- markdownlint-enable MD013 MD033 MD034 MD037 MD049 -->")
    md.append("\n---\n")
    md.append("## Library Versions\n")
    for name, ver in sorted(versions.items()):
        ver_str = "" if ver is None else ver
        md.append(f"- `{name}`: `{ver_str}`")
    md.append(f"\n_Report generated on: {local_now_str()}_")

    # Join and normalize trailing spaces across the entire Markdown document
    # Ensure file ends with single newline (MD047 requirement)
    markdown_content = normalize_markdown_trailing_spaces("\n".join(md)) + "\n"

    try:
        with filename.open("w", encoding="utf-8") as f:
            f.write(markdown_content)
        logger.info(
            "Markdown report saved to: %s",
            Colors.colored(str(filename.resolve()), Colors.GREEN),
        )
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


def _escape_markdown_in_text(text: str) -> str:
    """Escape only structural Markdown chars (currently pipe only).

    Minimal escaping avoids noisy backslashes while preserving readability.
    Newlines are replaced with ``<br>`` so multi-line outputs don't break the
    pipe table layout. Bare URLs are wrapped in angle brackets for MD034 compliance.
    """
    # First, convert newlines to HTML <br> tags to preserve line structure
    # Handle different newline formats consistently
    result = text.replace("\r\n", "<br>").replace("\r", "<br>").replace("\n", "<br>")

    # Clean up multiple consecutive <br> tags and normalize spacing
    result = re.sub(r"(<br>\s*){2,}", "<br><br>", result)  # Max 2 consecutive line breaks
    result = re.sub(r"\s+", " ", result).strip()  # Normalize other whitespace

    # Wrap bare URLs in angle brackets (MD034 compliance)
    result = _wrap_bare_urls(result)

    # Escape only the critical markdown characters that break table formatting, PLUS
    # neutralize raw HTML tag markers (<tag>) which GitHub may treat as HTML
    # (e.g. <s> strike-through).
    # (Subset chosen for formatting/readability; extend if needed.)
    escape_pairs = [
        ("|", "\\|"),  # Pipe - CRITICAL: breaks table column structure
    ]
    for char, escaped in escape_pairs:
        result = result.replace(char, escaped)

    # HTML-like angle bracket handling: escape < and > only when they appear to form a tag
    result = _escape_html_tags_selective(result)

    # Escape bare ampersands that could start entities; avoid double-escaping existing ones.
    return re.sub(r"&(?!lt;|gt;|amp;|#)", "&amp;", result)


def _escape_markdown_diagnostics(text: str) -> str:
    """Escape diagnostics text for Markdown tables more defensively.

    Behavior:
    - Convert newlines to <br> to keep table rows intact.
    - Wrap bare URLs in angle brackets to satisfy markdownlint MD034.
    - Escape characters that commonly trigger Markdown formatting in diagnostics: *, _, `, ~, and |.
    - Neutralize HTML-like tags except for a safe inline subset defined in allowed_inline_tags.
    - Do NOT collapse general whitespace, to avoid losing error message detail.
    """
    # Convert newlines to <br> but otherwise keep spacing as-is
    result = text.replace("\r\n", "<br>").replace("\r", "<br>").replace("\n", "<br>")

    # Limit excessive consecutive <br> while preserving intentional blank lines
    result = re.sub(r"(<br>\s*){3,}", "<br><br>", result)

    # Wrap bare URLs in angle brackets (MD034 compliance)
    result = _wrap_bare_urls(result)

    # Escape characters with special meaning in Markdown and table structure
    escape_map = {
        "|": "\\|",
        "*": "\\*",
        "_": "\\_",
        "`": "\\`",
        "~": "\\~",
    }
    for ch, repl in escape_map.items():
        result = result.replace(ch, repl)

    # Neutralize HTML-like tags except a safe allowlist
    result = _escape_html_tags_selective(result)

    # Escape bare ampersands (avoid starting entities)
    return re.sub(r"&(?!lt;|gt;|amp;|#)", "&amp;", result)


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
    - Keep exactly MARKDOWN_HARD_BREAK_SPACES trailing spaces (Markdown hard line break).
    - Strip any other count of trailing spaces to avoid accidental single-space endings.
    """
    out_lines: list[str] = []
    for ln in md_text.splitlines():
        m = re.search(r"( +)$", ln)
        if not m:
            out_lines.append(ln)
            continue
        spaces = len(m.group(1))
        if spaces == MARKDOWN_HARD_BREAK_SPACES:
            out_lines.append(ln)
        else:
            out_lines.append(ln[:-spaces])
    return "\n".join(out_lines)


def get_system_info() -> tuple[str, str | None]:
    """Get system architecture and GPU information."""
    arch: str = platform.machine()
    gpu_info: str | None = None
    try:
        # Try to get GPU info on macOS using full path for security
        if platform.system() == "Darwin":
            result: subprocess.CompletedProcess[str] = subprocess.run(
                ["/usr/sbin/system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
                timeout=5,  # Increased timeout for robustness
                check=False,
            )
            if result.returncode == 0:
                # --- Extract GPS info from system_profiler output
                gpu_lines: list[str] = [
                    line for line in result.stdout.split("\n") if "Chipset Model:" in line
                ]
                if gpu_lines:
                    gpu_info = gpu_lines[0].split("Chipset Model:")[-1].strip()
    except (subprocess.SubprocessError, TimeoutError) as e:
        logger.debug("Could not get GPU info: %s", e)
    return arch, gpu_info


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
    supported_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    if img_path.suffix.lower() not in supported_extensions:
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


def validate_image_accessible(*, image_path: PathLike) -> None:
    """Validate image file is accessible and supported."""
    img_path = Path(image_path)
    try:
        with (
            TimeoutManager(seconds=IMAGE_OPEN_TIMEOUT),
            Image.open(img_path),
        ):
            pass
    except TimeoutError as err:
        msg = f"Timeout while reading image: {img_path}"
        raise OSError(msg) from err
    except UnidentifiedImageError as err:
        msg = f"File is not a recognized image format: {img_path}"
        raise ValueError(msg) from err
    except (OSError, ValueError) as err:
        msg = f"Error accessing image {img_path}: {err}"
        raise OSError(msg) from err


def _run_model_generation(
    params: ProcessImageParams,
) -> GenerationResult | SupportsGenerationResult:
    """Load model + processor, apply chat template, run generation, time it.

    We keep all loading + formatting + generation steps together because they
    form a tightly coupled sequence (tokenizer/model/config interplay varies by
    repo). Errors are wrapped with traceback context so upstream summaries can
    show concise messages while verbose logs retain full detail.
    """
    model: Module
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast

    # Load model from HuggingFace Hub - this handles automatic download/caching
    # and converts weights to MLX format for Apple Silicon optimization
    try:
        model, tokenizer = load(
            path_or_hf_repo=params.model_identifier,
            lazy=params.lazy,
            trust_remote_code=params.trust_remote_code,
        )
        config: Any | None = getattr(model, "config", None)
    except Exception as load_err:
        # Capture any model loading errors (config issues, missing files, etc.)
        error_details = f"Model loading failed: {load_err}\n{traceback.format_exc()}"
        logger.exception("Failed to load model %s", params.model_identifier)
        raise ValueError(error_details) from load_err

    # Apply model-specific chat template - each model has its own conversation format
    # (e.g., Llama uses <|begin_of_text|>, Phi-3 uses <|user|>, etc.)
    formatted_prompt: str | list[Any] = apply_chat_template(
        processor=tokenizer,
        config=config,
        prompt=params.prompt,
        num_images=1,
    )
    # Handle list return from apply_chat_template
    if isinstance(formatted_prompt, list):
        formatted_prompt = "\n".join(str(m) for m in formatted_prompt)

    # Time the generation process manually since MLX VLM doesn't include timing
    start_time = time.perf_counter()
    try:
        output: GenerationResult | SupportsGenerationResult = generate(
            model=model,
            processor=cast("PreTrainedTokenizer", tokenizer),  # Cast to expected type
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
            trust_remote_code=params.trust_remote_code,
            max_tokens=params.max_tokens,
        )
    except TimeoutError as gen_to_err:
        msg = f"Generation timed out for model {params.model_identifier}: {gen_to_err}"
        # Re-raise to be handled by outer TimeoutError branch
        raise TimeoutError(msg) from gen_to_err
    except (OSError, ValueError) as gen_known_err:
        # Known I/O or validation-style issues
        msg = (
            f"Model generation failed for {params.model_identifier}: {gen_known_err}\n"
            f"{traceback.format_exc()}"
        )
        raise ValueError(msg) from gen_known_err
    except (RuntimeError, TypeError, AttributeError, KeyError) as gen_err:
        # Model-specific runtime errors (weights, config, tensor ops, missing attributes)
        msg = (
            f"Model runtime error during generation for {params.model_identifier}: {gen_err}\n"
            f"{traceback.format_exc()}"
        )
        raise ValueError(msg) from gen_err
    end_time = time.perf_counter()

    # Add timing to the GenerationResult object dynamically
    # Cast to our Protocol which includes the time attribute we're adding
    cast("SupportsGenerationResult", output).time = end_time - start_time

    mx.eval(model.parameters())
    return output


def process_image_with_model(params: ProcessImageParams) -> PerformanceResult:
    """Process an image with a Vision Language Model, managing stats and errors."""
    logger.info(
        "Processing '%s' with model: %s",
        Colors.colored(str(getattr(params.image_path, "name", params.image_path)), Colors.MAGENTA),
        Colors.colored(params.model_identifier, Colors.MAGENTA),
    )
    model: Module | None = None
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None
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

        return PerformanceResult(
            model_name=params.model_identifier,
            generation=output,
            success=True,
            generation_time=generation_time,
            model_load_time=model_load_time,
            total_time=total_time,
        )
    except TimeoutError as e:
        logger.exception("Timeout during model processing")
        return PerformanceResult(
            model_name=params.model_identifier,
            generation=None,
            success=False,
            error_stage="timeout",
            error_message=str(e),
            generation_time=None,
            model_load_time=None,
            total_time=None,
        )
    except (OSError, ValueError) as e:
        logger.exception("Model processing error")
        return PerformanceResult(
            model_name=params.model_identifier,
            generation=None,
            success=False,
            error_stage="processing",
            error_message=str(e),
            generation_time=None,
            model_load_time=None,
            total_time=None,
        )
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
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


def print_cli_section(title: str) -> None:
    """Print a formatted CLI section header with visual prefix."""
    width = get_terminal_width(max_width=100)
    logger.info(
        title,
        extra={
            "style_hint": LogStyles.SECTION,
            "style_uppercase": "\x1b[" not in title,
        },
    )
    log_rule(width, char="─", color=Colors.BLUE, bold=False)


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


def _summary_parts(res: PerformanceResult, model_short: str) -> list[str]:
    """Assemble key=value summary segments (reduced branching)."""
    gen = res.generation
    parts: list[str] = [
        f"model={model_short}",
        f"status={'OK' if res.success else 'FAIL'}",
    ]
    if gen:
        p_tokens = getattr(gen, "prompt_tokens", 0) or 0
        g_tokens = getattr(gen, "generation_tokens", 0) or 0
        tot_tokens = p_tokens + g_tokens
        gen_tps = getattr(gen, "generation_tps", 0.0) or 0.0
        peak_mem = getattr(gen, "peak_memory", 0.0) or 0.0
        total_time_val = getattr(res, "total_time", None)
        for present, label in (
            (tot_tokens, f"tokens={tot_tokens}"),
            (p_tokens, f"prompt={p_tokens}"),
            (g_tokens, f"gen={g_tokens}"),
        ):
            if present:
                parts.append(label)
        if gen_tps:
            parts.append(f"gen_tps={fmt_num(gen_tps)}")
        if peak_mem:
            peak_str = format_field_value("peak_memory", peak_mem)
            parts.append(f"peak_mem={peak_str}GB")
        if total_time_val:
            tt_val = format_field_value("total_time", total_time_val)
            time_str = (
                f"total_time={tt_val}"
                if isinstance(tt_val, str)
                else f"total_time={_format_time_seconds(total_time_val)}"
            )
            parts.append(time_str)
    if res.error_stage:
        parts.append(f"stage={res.error_stage}")
    if res.error_message:
        clean_err = re.sub(r"\s+", " ", str(res.error_message))
        preview = clean_err[:ERROR_MESSAGE_PREVIEW_LEN].rstrip()
        if len(clean_err) > ERROR_MESSAGE_PREVIEW_LEN:
            preview += "…"
        parts.append(f"error={preview}")
    return parts


def _preview_generation(gen: GenerationResult | SupportsGenerationResult | None) -> None:
    if not gen:
        return
    text_val = str(getattr(gen, "text", ""))
    if not text_val:
        logger.info("%s", Colors.colored("<empty>", Colors.CYAN))
        return

    # Check for quality issues (non-verbose mode gets brief inline warnings)
    is_repetitive, repeated_token = _detect_repetitive_output(text_val)
    hallucination_issues = _detect_hallucination_patterns(text_val)
    gen_tokens = getattr(gen, "generation_tokens", 0)
    is_verbose = _detect_excessive_verbosity(text_val, gen_tokens)
    format_issues = _detect_formatting_violations(text_val)

    if is_repetitive and repeated_token:
        logger.warning(Colors.colored(f"⚠️  Repetitive: '{repeated_token}'", Colors.YELLOW))
    if hallucination_issues:
        logger.warning(Colors.colored(f"⚠️  {', '.join(hallucination_issues[:2])}", Colors.YELLOW))
    if is_verbose:
        logger.warning(Colors.colored(f"⚠️  Verbose ({gen_tokens} tokens)", Colors.YELLOW))
    if format_issues:
        logger.warning(Colors.colored(f"⚠️  {format_issues[0]}", Colors.YELLOW))

    width = get_terminal_width(max_width=100)
    for original_line in text_val.splitlines():
        if not original_line.strip():
            continue
        wrapped = textwrap.wrap(
            original_line,
            width=width,
            replace_whitespace=False,
            drop_whitespace=True,
            break_long_words=False,
            break_on_hyphens=False,
        ) or [""]
        for wline in wrapped:
            logger.info("%s", Colors.colored(wline.lstrip(), Colors.CYAN))


def _log_verbose_success_details_mode(res: PerformanceResult, *, detailed: bool) -> None:
    """Emit verbose block using either compact or detailed metrics style with visual hierarchy."""
    if not res.generation:
        return

    # Add breathing room
    logger.info("")

    # Generated text with emoji prefix for easy scanning
    gen_text = getattr(res.generation, "text", None) or ""

    # Check for quality issues in generated text
    is_repetitive, repeated_token = _detect_repetitive_output(gen_text)
    hallucination_issues = _detect_hallucination_patterns(gen_text)
    gen_tokens = getattr(res.generation, "generation_tokens", 0)
    is_verbose = _detect_excessive_verbosity(gen_text, gen_tokens)
    format_issues = _detect_formatting_violations(gen_text)

    logger.info("📝 %s", Colors.colored("Generated Text:", Colors.BOLD, Colors.CYAN))

    # Warn about repetitive output
    if is_repetitive and repeated_token:
        warning_msg = f"⚠️  WARNING: Output appears to be garbage (repetitive: '{repeated_token}')"
        logger.warning(Colors.colored(warning_msg, Colors.YELLOW, Colors.BOLD))

    # Warn about hallucination patterns
    if hallucination_issues:
        for issue in hallucination_issues:
            logger.warning(Colors.colored(f"⚠️  Note: {issue}", Colors.YELLOW))

    # Warn about excessive verbosity
    if is_verbose:
        logger.warning(
            Colors.colored(
                f"⚠️  Note: Output is excessively verbose ({gen_tokens} tokens)",
                Colors.YELLOW,
            ),
        )

    # Warn about formatting issues
    if format_issues:
        for issue in format_issues[:2]:  # Show first 2 issues
            logger.warning(Colors.colored(f"⚠️  Note: {issue}", Colors.YELLOW))

    _log_wrapped_label_value("   ", gen_text, color=Colors.CYAN)

    logger.info("")  # Breathing room

    if detailed:
        logger.info("📊 %s", Colors.colored("Performance Metrics:", Colors.BOLD, Colors.WHITE))
        _log_token_summary(res)
        _log_detailed_timings(res)
        logger.info("")
        _log_perf_block(res)
    else:
        _log_compact_metrics(res)


def _log_token_summary(res: PerformanceResult) -> None:
    """Log tokens and generation TPS with tree structure for visual hierarchy."""
    p_tokens = getattr(res.generation, "prompt_tokens", 0)
    g_tokens = getattr(res.generation, "generation_tokens", 0)
    tot_tokens = (p_tokens or 0) + (g_tokens or 0)
    gen_tps = getattr(res.generation, "generation_tps", 0.0) or 0.0
    prompt_tps = getattr(res.generation, "prompt_tps", 0.0) or 0.0

    logger.info("  🔢 %s", Colors.colored("Tokens:", Colors.BOLD, Colors.WHITE))
    logger.info(
        "     ├─ Prompt:     %s",
        Colors.colored(f"{fmt_num(p_tokens):>8} @ {fmt_num(prompt_tps)} tok/s", Colors.WHITE),
    )
    logger.info(
        "     ├─ Generated:  %s",
        Colors.colored(f"{fmt_num(g_tokens):>8} @ {fmt_num(gen_tps)} tok/s", Colors.WHITE),
    )
    logger.info("     └─ Total:      %s", Colors.colored(f"{fmt_num(tot_tokens):>8}", Colors.WHITE))


def _log_detailed_timings(res: PerformanceResult) -> None:
    """Log total, generation, and model load times with tree structure."""
    total_time_val = getattr(res, "total_time", None)
    generation_time_val = getattr(res, "generation_time", None)
    model_load_time_val = getattr(res, "model_load_time", None)

    if not total_time_val or total_time_val <= 0:
        return

    logger.info("  ⏱  %s", Colors.colored("Timing:", Colors.BOLD, Colors.WHITE))

    tt_val = format_field_value("total_time", total_time_val)
    tt_disp = tt_val if isinstance(tt_val, str) else _format_time_seconds(total_time_val)
    logger.info("     ├─ Total:      %s", Colors.colored(f"{tt_disp:>8}", Colors.WHITE))

    if generation_time_val and generation_time_val > 0:
        gt_val = format_field_value("generation_time", generation_time_val)
        gt_disp = gt_val if isinstance(gt_val, str) else _format_time_seconds(generation_time_val)
        pct = (generation_time_val / total_time_val * 100) if total_time_val > 0 else 0
        logger.info(
            "     ├─ Generation: %s",
            Colors.colored(f"{gt_disp:>8} ({pct:>3.0f}%)", Colors.WHITE),
        )

    if model_load_time_val and model_load_time_val > 0:
        ml_val = format_field_value("model_load_time", model_load_time_val)
        ml_disp = ml_val if isinstance(ml_val, str) else _format_time_seconds(model_load_time_val)
        pct = (model_load_time_val / total_time_val * 100) if total_time_val > 0 else 0
        logger.info(
            "     └─ Load:       %s",
            Colors.colored(f"{ml_disp:>8} ({pct:>3.0f}%)", Colors.WHITE),
        )


def _log_perf_block(res: PerformanceResult) -> None:
    """Log inner performance metrics (memory) with tree structure and emoji."""
    active_mem = getattr(res.generation, "active_memory", 0.0) or 0.0
    cached_mem = getattr(res.generation, "cached_memory", 0.0) or 0.0
    peak_mem = getattr(res.generation, "peak_memory", 0.0) or 0.0

    # Only show memory section if at least one value is present
    if active_mem <= 0 and cached_mem <= 0 and peak_mem <= 0:
        return

    logger.info("  💾 %s", Colors.colored("Memory:", Colors.BOLD, Colors.WHITE))

    def _log_mem(prefix: str, label: str, field: str, raw_val: float) -> None:
        if raw_val <= 0:
            return
        formatted = format_field_value(field, raw_val)
        unit = "GB"
        text = str(formatted) if str(formatted).endswith(unit) else f"{formatted} GB"
        logger.info(
            "     %s %s %s",
            prefix,
            label.ljust(11),
            Colors.colored(f"{text:>8}", Colors.WHITE),
        )

    _log_mem("├─", "Active Δ:", "active_memory", active_mem)
    _log_mem("├─", "Cache Δ:", "cached_memory", cached_mem)
    _log_mem("└─", "Peak:", "peak_memory", peak_mem)


def _log_compact_metrics(res: PerformanceResult) -> None:
    """Emit grouped metrics lines for better readability while maintaining parsability.

    Example output (single aligned line):
        📊 Metrics: total=4.13s  gen=3.03s  load=1.09s  peak_mem=5.5GB
                    tokens(total/prompt/gen)=1,637/1,488/149  gen_tps=114  prompt_tps=421
    """
    if not res.generation:
        return

    logger.info("")  # Breathing room
    parts = _build_compact_metric_parts(res, res.generation)
    if not parts:
        return
    aligned = _align_metric_parts(parts)
    logger.info(
        "📊 %s %s",
        Colors.colored("Metrics:", Colors.BOLD, Colors.WHITE),
        Colors.colored("  ".join(aligned), Colors.WHITE),
    )


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
    logger.info("")
    logger.info(Colors.colored("📖 Metrics Format:", Colors.BOLD, Colors.CYAN))
    if detailed:
        logger.info(
            Colors.colored(
                "  • Detailed mode: separate lines for timing, memory, tokens, TPS",
                Colors.GRAY,
            ),
        )
    else:
        logger.info(
            Colors.colored(
                "  • Compact mode: tokens(total/prompt/gen) format with aligned keys",
                Colors.GRAY,
            ),
        )
    logger.info(
        Colors.colored(
            "  • ⚠️  warnings shown for repetitive or hallucinated output",
            Colors.GRAY,
        ),
    )
    logger.info(
        Colors.colored(
            "  • Note: Streaming early-stop for repetitive output is not yet implemented",
            Colors.GRAY,
        ),
    )
    logger.info("")


def print_model_result(
    result: PerformanceResult,
    *,
    verbose: bool = False,
    detailed_metrics: bool = False,
    run_index: int | None = None,
    total_runs: int | None = None,
) -> None:
    """Print a concise summary + optional verbose block for a model result."""
    model_short = result.model_name.split("/")[-1]
    run_prefix = "" if run_index is None else f"[RUN {run_index}/{total_runs}] "
    summary = run_prefix + "SUMMARY " + " ".join(_summary_parts(result, model_short))
    log_fn = logger.info if result.success else logger.error
    color = Colors.GREEN if result.success else Colors.RED
    # Wrap summary to terminal width for readability
    width = get_terminal_width(max_width=100)
    for line in textwrap.wrap(summary, width=width, break_long_words=False, break_on_hyphens=False):
        log_fn(Colors.colored(line, color))
    if result.success and not verbose:  # quick exit with preview only
        _preview_generation(result.generation)
        return
    header_label = "✓ SUCCESS" if result.success else "✗ FAILED"
    header_color = Colors.GREEN if result.success else Colors.RED
    header = (
        f"{header_label}: "
        f"{Colors.colored(model_short, Colors.MAGENTA if result.success else Colors.RED)}"
    )
    log_fn(Colors.colored(header, Colors.BOLD, header_color))
    if not result.success:
        if result.error_stage:
            _log_wrapped_error("Stage:", str(result.error_stage))
        if result.error_message:
            _log_wrapped_error("Error:", str(result.error_message))
        if result.captured_output_on_fail:
            _log_wrapped_error("Output:", str(result.captured_output_on_fail))
        return
    if result.generation and verbose:
        _log_verbose_success_details_mode(result, detailed=detailed_metrics)


def print_cli_separator() -> None:
    """Print a visually distinct separator line using unicode box-drawing characters."""
    width = get_terminal_width(max_width=100)
    log_rule(width, char="─", color=Colors.BLUE, bold=False)


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

    # Set DEBUG if verbose, else WARNING
    log_level: int = logging.DEBUG if args.verbose else logging.INFO
    # Remove all handlers and add console + file handlers
    logger.handlers.clear()

    # Console handler with colored output
    console_handler: logging.StreamHandler[Any] = logging.StreamHandler(sys.stderr)
    # Include timestamp for better traceability, level in verbose mode
    if args.verbose:
        fmt = "%(asctime)s - %(levelname)s - %(message)s"
    else:
        fmt = "%(asctime)s - %(message)s"
    console_formatter: ColoredFormatter = ColoredFormatter(fmt)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler - write to specified log file (overwrite each run)
    log_file: Path = args.output_log.resolve()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler: logging.FileHandler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    # File gets full timestamp + level always (no colors in file)
    file_formatter: logging.Formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logger.setLevel(log_level)
    logger.propagate = False  # Prevent double logging

    if args.verbose:
        logger.debug("Verbose/debug mode enabled.")

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
        logger.warning("SECURITY WARNING: --trust-remote-code is enabled.")
        logger.warning("This allows execution of remote code and may pose security risks.")

    return library_versions


def find_and_validate_image(args: argparse.Namespace) -> Path:
    """Find and validate the image file to process from arguments."""
    folder_path: Path = args.folder.resolve()
    print_cli_section(
        f"Scanning folder: {Colors.colored(str(folder_path), Colors.MAGENTA)}",
    )

    if args.folder == DEFAULT_FOLDER and not DEFAULT_FOLDER.is_dir():
        print_cli_error(f"Default folder '{DEFAULT_FOLDER}' does not exist.")

    image_path: Path | None = find_most_recent_file(folder_path)
    if image_path is None:
        exit_with_cli_error(
            f"Could not find the most recent image file in {folder_path}. Exiting.",
        )

    # At this point image_path is guaranteed to be Path (not None) due to exit above
    resolved_image_path: Path = image_path.resolve()
    print_cli_section(f"Image File: {Colors.colored(str(resolved_image_path), Colors.MAGENTA)}")

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
        print_cli_separator()
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
        gps_val = metadata.get("gps")
        prompt_parts: list[str] = [
            (
                "Provide a factual caption, description, and keywords suitable for "
                "cataloguing, or searching for, the image."
            ),
            (f"\n\nContext: The image relates to '{desc}'" if desc else ""),
            (f"\n\nThe photo was taken around {date_val}" if date_val else ""),
            (f"from GPS {gps_val}" if gps_val else ""),
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
    *,  # Force prompt to be keyword-only for clarity
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
        model_identifiers = _apply_exclusions(
            model_identifiers,
            args.exclude or [],
            "cached models",
        )

    results: list[PerformanceResult] = []
    if not model_identifiers:
        logger.error("No models specified or found in cache.")
        if not args.models:
            logger.error("Ensure models are downloaded and cache is accessible.")
    else:
        logger.info("Processing %d model(s)...", len(model_identifiers))

        # Validate parameters before processing any models
        validate_sampling_params(
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        validate_kv_params(
            max_kv_size=args.max_kv_size,
            kv_bits=args.kv_bits,
        )

        # Validate all model identifiers before processing
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
        model_label = model_id.split("/")[-1]
        progress = f"[{idx}/{len(model_identifiers)}]"
        print_cli_section(
            f"Processing Model {progress}: {Colors.colored(model_label, Colors.MAGENTA)}",
        )

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
        )
        result: PerformanceResult = process_image_with_model(params)
        results.append(result)

        # Structured output for this run
        print_model_result(
            result,
            verbose=args.verbose,
            detailed_metrics=getattr(args, "detailed_metrics", False),
            run_index=idx,
            total_runs=len(model_identifiers),
        )
    return results


def finalize_execution(
    *,
    args: argparse.Namespace,
    results: list[PerformanceResult],
    library_versions: LibraryVersionDict,
    overall_start_time: float,
    prompt: str,
) -> None:
    """Output summary statistics, generate reports, and display timing information."""
    overall_time: float = time.perf_counter() - overall_start_time
    if results:
        print_cli_section("Performance Summary")
        print_model_stats(results)

        # Add model issues summary
        summary = analyze_model_issues(results)
        issues_text = format_issues_summary_text(summary)
        if issues_text:
            logger.info("")
            logger.info(issues_text)
            logger.info("")

        print_cli_section("Report Generation")
        try:
            html_output_path: Path = args.output_html.resolve()
            md_output_path: Path = args.output_markdown.resolve()
            html_output_path.parent.mkdir(parents=True, exist_ok=True)
            md_output_path.parent.mkdir(parents=True, exist_ok=True)

            generate_html_report(
                results=results,
                filename=html_output_path,
                versions=library_versions,
                prompt=prompt,
                total_runtime_seconds=overall_time,
            )
            generate_markdown_report(
                results=results,
                filename=md_output_path,
                versions=library_versions,
                prompt=prompt,
                total_runtime_seconds=overall_time,
            )

            logger.info("")
            logger.info(
                "📊 %s",
                Colors.colored("Reports successfully generated:", Colors.BOLD, Colors.GREEN),
            )
            logger.info("   HTML:     %s", Colors.colored(str(html_output_path), Colors.CYAN))
            logger.info("   Markdown: %s", Colors.colored(str(md_output_path), Colors.CYAN))
            log_output = args.output_log.resolve()
            logger.info("   Log:      %s", Colors.colored(str(log_output), Colors.CYAN))
        except (OSError, ValueError):
            logger.exception("Failed to generate reports.")
    else:
        logger.warning("No models processed. No performance summary generated.")
        logger.info("Skipping report generation as no models were processed.")

    print_cli_section("Execution Summary")
    logger.info("")
    logger.info(
        "⏱  Overall runtime: %s",
        Colors.colored(format_overall_runtime(overall_time), Colors.BOLD, Colors.WHITE),
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
        print_cli_header("MLX Vision Language Model Check")

        image_path = find_and_validate_image(args)

        metadata = handle_metadata(image_path, args)

        prompt = prepare_prompt(args, metadata)

        results = process_models(args, image_path, prompt=prompt)

        finalize_execution(
            args=args,
            results=results,
            library_versions=library_versions,
            overall_start_time=overall_start_time,
            prompt=prompt,
        )
    except (KeyboardInterrupt, SystemExit):
        logger.exception("Execution interrupted by user.")
        sys.exit(1)
    except (OSError, ValueError) as main_err:
        logger.critical("Fatal error in main execution: %s", main_err)
        sys.exit(1)


def main_cli() -> None:
    """CLI entry point for the MLX VLM checker script."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="MLX VLM Model Checker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Add arguments (separated for clarity)
    parser.add_argument(
        "-f",
        "--folder",
        type=Path,
        default=DEFAULT_FOLDER,
        help="Folder to scan.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=DEFAULT_HTML_OUTPUT,
        help="Output HTML report file.",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=DEFAULT_MD_OUTPUT,
        help="Output Github Markdown report file.",
    )
    parser.add_argument(
        "--output-log",
        type=Path,
        default=DEFAULT_LOG_OUTPUT,
        help="Output log file (overwritten each run).",
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
        action="store_true",
        default=True,
        help="Allow custom code from Hub models (SECURITY RISK).",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default=None,
        help="Prompt.",
    )
    metrics_group = parser.add_mutually_exclusive_group()
    metrics_group.add_argument(
        "--detailed-metrics",
        action="store_true",
        default=False,
        help=(
            "Use expanded multi-line metrics block instead of compact aligned single-line metrics. "
            "Only applies when --verbose is set."
        ),
    )
    metrics_group.add_argument(
        "--compact-metrics",
        action="store_true",
        default=False,
        help=(
            "Force compact metrics (explicit; default behavior). "
            "Useful if environment sets a different default."
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
        "--repetition-penalty",
        type=float,
        default=None,
        help="Penalize repeated tokens (>1.0 discourages repetition). Default: None (disabled).",
    )
    parser.add_argument(
        "--repetition-context-size",
        type=int,
        default=20,
        help="Context window size for repetition penalty. Default: 20 tokens.",
    )
    parser.add_argument(
        "--lazy-load",
        action="store_true",
        default=False,
        help="Use lazy loading for models (loads weights on-demand, reduces peak memory).",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        default=None,
        help="Maximum KV cache size (limits memory for long sequences). Default: None (unlimited).",
    )
    parser.add_argument(
        "--kv-bits",
        type=int,
        default=None,
        choices=[4, 8],
        help="Quantize KV cache to N bits (4 or 8). Saves memory with small quality trade-off.",
    )
    parser.add_argument(
        "--kv-group-size",
        type=int,
        default=64,
        help="Quantization group size for KV cache. Default: 64.",
    )
    parser.add_argument(
        "--quantized-kv-start",
        type=int,
        default=0,
        help="Start position for KV cache quantization. Default: 0 (from beginning).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose and debug output (DEBUG logging).",
    )
    parser.add_argument(
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

    # Parse arguments
    args: argparse.Namespace = parser.parse_args()
    # Print all command-line arguments if verbose is set
    if getattr(args, "verbose", False):
        print_cli_section("Command Line Parameters")
        for arg_name, arg_value in sorted(vars(args).items()):
            logger.info("  %s: %s", arg_name, arg_value)
    # Normalize: if neither flag set, compact is default; if both prevented by group.
    if getattr(args, "compact_metrics", False):
        args.detailed_metrics = False
    main(args)


if __name__ == "__main__":
    main_cli()
