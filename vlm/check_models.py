#!/usr/bin/env python3
"""Image analysis and caption generation using MLX Vision Language Models."""

from __future__ import annotations

import argparse
import contextlib
import functools
import html
import logging
import platform
import re
import signal
import subprocess
import sys
import time
import traceback
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
    Self,
    TypeVar,
)

from huggingface_hub import HFCacheInfo, scan_cache_dir
from huggingface_hub import __version__ as hf_version
from huggingface_hub.errors import HFValidationError
from tabulate import tabulate
from tzlocal import get_localzone

if TYPE_CHECKING:
    import types
    from collections.abc import Callable
    from zoneinfo import ZoneInfo

LOGGER_NAME: Final[str] = "mlx-vlm-check"
NOT_AVAILABLE: Final[str] = "N/A"

ERROR_MLX_MISSING: Final[str] = "Core dependency missing: mlx. Please install it."
ERROR_PILLOW_MISSING: Final[str] = (
    "Error: Pillow not found. Please install it (`pip install Pillow`)."
)
ERROR_MLX_VLM_MISSING: Final[str] = (
    "Error: mlx-vlm not found. Please install it (`pip install mlx-vlm`)."
)

DEFAULT_TIMEOUT_LONG: Final[float] = 300.0

MIN_SEPARATOR_CHARS: Final[int] = 50
DEFAULT_DECIMAL_PLACES: Final[int] = 2
LARGE_NUMBER_THRESHOLD: Final[float] = 100.0
MEDIUM_NUMBER_THRESHOLD: Final[int] = 10
THOUSAND_THRESHOLD: Final[int] = 1000

_temp_logger = logging.getLogger(LOGGER_NAME)

try:
    import mlx.core as mx
except ImportError:
    _temp_logger.exception(ERROR_MLX_MISSING)
    sys.exit(1)

try:
    from PIL import ExifTags, Image, UnidentifiedImageError
    from PIL.ExifTags import GPSTAGS, TAGS

    pillow_version: str = Image.__version__ if hasattr(Image, "__version__") else NOT_AVAILABLE
except ImportError:
    _temp_logger.critical(ERROR_PILLOW_MISSING)
    pillow_version = NOT_AVAILABLE
    sys.exit(1)

try:
    from mlx_vlm.generate import GenerationResult, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load
    from mlx_vlm.version import __version__ as vlm_version
except ImportError:
    _temp_logger.critical(ERROR_MLX_VLM_MISSING)
    sys.exit(1)
try:
    import mlx_lm

    mlx_lm_version: str = getattr(mlx_lm, "__version__", NOT_AVAILABLE)
except ImportError:
    mlx_lm_version = NOT_AVAILABLE
except AttributeError:
    mlx_lm_version = "N/A (module found, no version attr)"
try:
    import transformers

    transformers_version: str = transformers.__version__
except ImportError:
    transformers_version = NOT_AVAILABLE


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
        self.timer: Any = None

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
                    signal.alarm(int(self.seconds))
                except ValueError as e:
                    # Signal handling restricted (threading/subprocess environment)
                    logger.warning(
                        "Could not set SIGALRM for timeout: %s. Timeout disabled.",
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
    _enabled: ClassVar[bool] = sys.stderr.isatty()
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
    def visual_len(text: str) -> int:
        """Return the visual length of text, ignoring ANSI codes."""
        # Remove ANSI codes for accurate width
        return len(Colors._ansi_escape_re.sub("", text))


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
        msg: str = super().format(record)

        # Apply level-based coloring first
        level_color: str = self.LEVEL_COLORS.get(record.levelno, "")

        if record.levelno == logging.INFO:
            return self._format_info_message(msg)

        if level_color:
            return Colors.colored(msg, level_color)

        return msg

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
                lambda _, m: any(metric in m for metric in ["Tokens:", "TPS:", "Time:", "Memory:"]),
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

MB_CONVERSION: Final[float] = 1024 * 1024

# EXIF/GPS coordinate standards: camera hardware stores GPS as degrees-minutes-seconds tuples
DMS_LEN: Final[int] = 3  # Full DMS: (degrees, minutes, seconds)
DM_LEN: Final[int] = 2  # Decimal minutes: (degrees, decimal_minutes)
MAX_DEGREES: Final[int] = 180
MAX_MINUTES: Final[int] = 60
MAX_SECONDS: Final[int] = 60
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
    "peak_memory": "(MB)",
    "time": "(s)",
    "duration": "(s)",
}

FIELD_ABBREVIATIONS: Final[dict[str, tuple[str, str]]] = {
    "tokens": ("Token", "(ct)"),
    "prompt_tokens": ("Prompt", "(ct)"),
    "generation_tokens": ("Gen", "(ct)"),
    "total_tokens": ("Total", "Tokens"),
    "prompt_tps": ("Prompt", "(t/s)"),
    "generation_tps": ("Gen", "(t/s)"),
    "peak_memory": ("Peak", "(MB)"),
    "time": ("Time", "(s)"),
    "duration": ("Dur", "(s)"),
    "elapsed_time": ("Elapsed", "(s)"),
    "model_load_time": ("Load", "(s)"),
    "total_time": ("Total", "(s)"),
}

# Threshold for splitting long header text into multiple lines
HEADER_SPLIT_LENGTH = 10

# Fields that should be right-aligned (numeric fields)
NUMERIC_FIELD_PATTERNS: Final[frozenset[str]] = frozenset(
    {
        "tokens",
        "prompt_tokens",
        "generation_tokens",
        "prompt_tps",
        "generation_tps",
        "peak_memory",
        "time",
        "duration",
        "elapsed_time",
        "model_load_time",
        "total_time",
    },
)

# Console table formatting constants
MAX_MODEL_NAME_LENGTH = 14
MAX_OUTPUT_LENGTH = 28

# PerformanceResult timing fields - centralized definition
PERFORMANCE_TIMING_FIELDS: Final[list[str]] = ["elapsed_time", "model_load_time", "total_time"]


def fmt_num(val: float | str) -> str:
    """Format numbers consistently across all output formats."""
    try:
        fval = float(val)
        if abs(fval) >= LARGE_NUMBER_THRESHOLD:
            return f"{fval:,.0f}"
        if abs(fval) >= 1:
            return f"{fval:.3g}"
        if abs(fval) > 0:
            return f"{fval:.3g}"
        return str(val)
    except (ValueError, TypeError, OverflowError):
        return str(val)


def format_field_label(field_name: str) -> str:
    """Convert field_name to a formatted display label."""
    return field_name.replace("_", " ").title()


def format_field_value(field_name: str, value: object) -> object:
    """Format a field value for display, applying unit conversions as needed."""
    if field_name == "peak_memory" and isinstance(value, (int, float)):
        # MLX VLM returns memory in GB, convert to MB for display consistency
        # If value is very small (< 1), assume it's already in MB; if large (> 100), assume bytes
        if value < 1.0:
            # Assume already in MB
            return f"{value:,.0f}"
        if value > LARGE_NUMBER_THRESHOLD:
            # Assume in bytes, convert to MB
            mb_value = value / MB_CONVERSION
            return f"{mb_value:,.0f}"
        # Assume in GB, convert to MB
        mb_value = value * 1024
        return f"{mb_value:,.0f}"
    if field_name.endswith("_tps") and isinstance(value, (int, float)):
        # Format TPS values to 1 decimal place or 3 significant figures with comma separators
        if abs(value) >= LARGE_NUMBER_THRESHOLD:
            return f"{value:,.0f}"  # No decimal for large values (≥100)
        if abs(value) >= MEDIUM_NUMBER_THRESHOLD:
            return f"{value:,.1f}"  # 1 decimal place for medium values (10-99.9)
        return f"{value:.2g}"  # Up to 2 significant figures for small values (<10)
    return value


def is_numeric_value(val: float | str | object) -> bool:
    """Check if a value is numeric or a string representing a numeric value."""
    return isinstance(val, (int, float)) or (
        isinstance(val, str) and val.replace(".", "", 1).isdigit()
    )


def is_numeric_field(field_name: str) -> bool:
    """Check if a field should be treated as numeric (right-aligned)."""
    field_lower = field_name.lower()
    return (
        field_name in NUMERIC_FIELD_PATTERNS
        or any(keyword in field_lower for keyword in ("token", "tps", "memory", "time"))
        or field_lower.endswith("_tokens")
    )


# Alias for backward compatibility
is_numeric_string = is_numeric_value


# --- Utility Functions ---
# Ensure _pad_text is defined only once at module level and used everywhere


def _pad_text(text: str, width: int, *, right_align: bool = False) -> str:
    """Pad text to a given width, optionally right-aligning."""
    pad_len: int = width - Colors.visual_len(text)
    pad_len = max(pad_len, 0)
    pad_str: str = " " * pad_len
    return (pad_str + text) if right_align else (text + pad_str)


def get_library_versions() -> dict[str, str]:
    """Return versions of key libraries as a dictionary."""
    return {
        "mlx": getattr(mx, "__version__", "N/A"),
        "mlx-vlm": vlm_version if "vlm_version" in globals() else "N/A",
        "mlx-lm": mlx_lm_version,
        "huggingface-hub": hf_version,
        "transformers": transformers_version,
        "Pillow": pillow_version,
    }


def print_version_info(versions: dict[str, str]) -> None:
    """Print library versions and generation date to the console."""
    logger.info("--- Library Versions ---")
    max_len: int = max(len(k) for k in versions) + 1 if versions else 10
    for name, ver in sorted(versions.items()):
        name_padded: str = name.ljust(max_len)
        logger.info("%s: %s", name_padded, ver)
    logger.info(
        "Generated: %s",
        datetime.now(get_localzone()).strftime("%Y-%m-%d %H:%M:%S %Z"),
    )


# Type aliases and definitions
T = TypeVar("T")
ExifValue = Any
ExifDict = dict[str | int, ExifValue]
MetadataDict = dict[str, str]
PathLike = str | Path
GPSTupleElement = int | float
GPSTuple = tuple[GPSTupleElement, GPSTupleElement, GPSTupleElement]

# Constants - Defaults
# These constants define default values for various parameters used in the script.
DEFAULT_MAX_TOKENS: Final[int] = 500
DEFAULT_FOLDER: Final[Path] = Path.home() / "Pictures" / "Processed"
DEFAULT_HTML_OUTPUT: Final[Path] = Path("results.html")
DEFAULT_MD_OUTPUT: Final[Path] = Path("results.md")
DEFAULT_TEMPERATURE: Final[float] = 0.1
DEFAULT_TIMEOUT: Final[float] = 300.0  # Default timeout in seconds

# Constants - EXIF
EXIF_IMAGE_DESCRIPTION_TAG: Final[int] = 270  # Standard EXIF tag ID for ImageDescription
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
GPS_LAT_REF_TAG: Final[int] = 1
GPS_LAT_TAG: Final[int] = 2
GPS_LON_REF_TAG: Final[int] = 3
GPS_LON_TAG: Final[int] = 4
GPS_INFO_TAG_ID: Final[int] = 34853  # Standard EXIF tag ID for GPS IFD


# Type definitions


@dataclass(frozen=True)
class PerformanceResult:
    """Encapsulates a GenerationResult and execution metadata for a model run."""

    model_name: str
    generation: GenerationResult | None
    success: bool
    error_stage: str | None = None
    error_message: str | None = None
    captured_output_on_fail: str | None = None
    elapsed_time: float | None = None  # Time taken for generation in seconds
    model_load_time: float | None = None  # Time taken to load the model in seconds
    total_time: float | None = None  # Total time including model loading


# --- File Handling ---
# Simplified the `find_most_recent_file` function by using `max` with a generator.
def find_most_recent_file(folder: Path | str) -> Path | None:
    """Return the most recently modified file in a folder, or None."""
    folder_path = Path(folder)  # Path() handles both str and Path objects
    if not folder_path.is_dir():
        logger.error("Provided path is not a directory: %s", folder_path)
        return None
    try:
        most_recent: Path | None = max(
            (f for f in folder_path.iterdir() if f.is_file() and not f.name.startswith(".")),
            key=lambda f: f.stat().st_mtime,
            default=None,
        )
        if most_recent:
            logger.debug("Most recent file found: %s", str(most_recent))
            return most_recent
    except FileNotFoundError:
        logger.exception("Directory not found: %s", folder_path)
    except PermissionError:
        logger.exception("Permission denied accessing folder: %s", folder_path)
    except OSError:
        logger.exception("OS error scanning folder %s", folder_path)
    logger.debug("No files found in directory: %s", folder_path)
    return None


# Improved error handling in `print_image_dimensions`.
def print_image_dimensions(image_path: Path | str) -> None:
    """Print the dimensions and megapixel count of an image file."""
    img_path_str: str = str(image_path)
    try:
        with Image.open(img_path_str) as img:
            width, height = img.size
            total_pixels = width * height
            logger.info(
                "Image dimensions: %s (%.1f MPixels)",
                f"{width}x{height}",
                total_pixels / 1_000_000,
            )
    except (FileNotFoundError, UnidentifiedImageError):
        logger.exception("Error with image file %s", img_path_str)
    except OSError:
        logger.exception("Unexpected error reading image dimensions for %s", img_path_str)


# --- EXIF & Metadata Handling ---
@functools.lru_cache(maxsize=128)
def get_exif_data(image_path: PathLike) -> ExifDict | None:
    """Extract EXIF data from an image file.

    EXIF data is stored in Image File Directories (IFDs) with a hierarchical structure:
    - IFD0: Main image metadata (camera, dimensions, etc.)
    - Exif SubIFD: Camera settings (exposure, ISO, etc.)
    - GPS SubIFD: Location data stored as degrees/minutes/seconds tuples

    The PIL library provides access to raw numeric tag IDs which we convert to
    human-readable names using TAGS/GPSTAGS lookup tables.

    Args:
        image_path: Path to the image file to process

    Returns:
        Dictionary of EXIF tags and values, or None if no EXIF data found

    """
    img_path_str: str = str(image_path)
    try:
        with Image.open(img_path_str) as img:
            exif_raw: Any = img.getexif()
            if not exif_raw:
                logger.debug("No EXIF data found in %s", img_path_str)
                return None
            exif_decoded: ExifDict = {}
            # First pass: Process IFD0 (main image directory) tags
            for tag_id, value in exif_raw.items():
                # Skip SubIFD pointers, we'll handle them separately
                if tag_id in (ExifTags.Base.ExifOffset, ExifTags.Base.GPSInfo):
                    continue
                tag_name: str = TAGS.get(tag_id, str(tag_id))
                exif_decoded[tag_name] = value
            # Second pass: Process Exif SubIFD (if available)
            try:
                exif_ifd: Any = exif_raw.get_ifd(ExifTags.IFD.Exif)
                if exif_ifd:
                    exif_decoded.update(
                        {
                            TAGS.get(tag_id, str(tag_id)): value
                            for tag_id, value in exif_ifd.items()
                        },
                    )
            except (KeyError, AttributeError, TypeError):
                logger.exception("Could not extract Exif SubIFD")
            # Third pass: Process GPS IFD (if available)
            try:
                gps_ifd: Any = exif_raw.get_ifd(ExifTags.IFD.GPSInfo)
                if isinstance(gps_ifd, dict) and gps_ifd:
                    gps_decoded: dict[str, Any] = {}
                    for gps_tag_id, gps_value in gps_ifd.items():
                        try:
                            gps_key = GPSTAGS.get(int(gps_tag_id), str(gps_tag_id))
                        except (KeyError, ValueError, TypeError):
                            gps_key = str(gps_tag_id)
                        gps_decoded[str(gps_key)] = gps_value
                    exif_decoded["GPSInfo"] = gps_decoded
            except (KeyError, AttributeError, TypeError) as gps_err:
                logger.warning("Could not extract GPS IFD: %s", gps_err)
            return exif_decoded
    except (FileNotFoundError, UnidentifiedImageError):
        logger.exception(
            Colors.colored(f"Error reading image file: {img_path_str}", Colors.YELLOW),
        )
    except (OSError, ValueError, TypeError):
        logger.exception(
            Colors.colored(
                f"Unexpected error reading EXIF from: {img_path_str}",
                Colors.YELLOW,
            ),
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


MAX_GPS_COORD_LEN: Final[int] = 3
MED_GPS_COORD_LEN: Final[int] = 2
MIN_GPS_COORD_LEN: Final[int] = 1


# Reduce return count and use named constants
def _convert_gps_coordinate(
    coord: tuple[float | str, ...] | list[float | str],
) -> tuple[float, float, float] | None:
    """Convert GPS EXIF coordinate to (degrees, minutes, seconds) tuple."""
    clen: int = len(coord)
    if clen == MAX_GPS_COORD_LEN:
        deg = to_float(val=coord[0])
        min_ = to_float(val=coord[1])
        sec = to_float(val=coord[2])
        if deg is not None and min_ is not None and sec is not None:
            return (deg, min_, sec)
    elif clen == MED_GPS_COORD_LEN:
        deg = to_float(val=coord[0])
        min_ = to_float(val=coord[1])
        if deg is not None and min_ is not None:
            return (deg, min_, 0.0)
    elif clen == MIN_GPS_COORD_LEN:
        deg = to_float(val=coord[0])
        if deg is not None:
            return (deg, 0.0, 0.0)
    return None


def extract_image_metadata(image_path: Path | str) -> MetadataDict:
    """Extract key metadata (date, GPS, EXIF tags) from an image file."""
    metadata: MetadataDict = {}
    img_path_str: str = str(image_path)
    exif_data = get_exif_data(img_path_str) or {}

    date = (
        exif_data.get("DateTimeOriginal")
        or exif_data.get("CreateDate")
        or exif_data.get("DateTime")
        or "No date recorded"
    )
    if not date:
        try:
            local_tz = get_localzone()
            date = datetime.fromtimestamp(
                Path(img_path_str).stat().st_mtime,
                tz=local_tz,
            ).strftime("%Y-%m-%d %H:%M:%S %Z")
        except OSError as err:
            date = "Unknown date"
            logger.debug("Could not get file mtime: %s", err)
    else:
        # If EXIF date is present, try to parse and localize it
        try:
            for fmt in DATE_FORMATS:
                try:
                    dt: datetime = datetime.strptime(str(date), fmt).replace(
                        tzinfo=UTC,
                    )
                    local_tz: ZoneInfo = get_localzone()
                    date: str = dt.astimezone(local_tz).strftime("%Y-%m-%d %H:%M:%S %Z")
                    break
                except ValueError:
                    continue
        except (ValueError, TypeError, UnicodeDecodeError) as err:
            logger.warning("Could not localize EXIF date: %s", err)
            date = str(date)
    metadata["date"] = str(date)

    # --- Description extraction ---
    description = exif_data.get("ImageDescription")
    desc_str = "N/A"
    if description is not None:
        if isinstance(description, bytes):
            try:
                desc_str = description.decode("utf-8", errors="replace").strip()
            except UnicodeDecodeError as err:
                desc_str = str(description)
                logger.debug("Failed to decode description: %s", err)
        else:
            desc_str = str(description).strip()
        if not desc_str:
            desc_str = "N/A"
    metadata["description"] = desc_str

    # --- GPS extraction helper ---
    def _extract_gps_str(gps_info_raw: object) -> str:
        if not isinstance(gps_info_raw, dict):
            return "N/A"
        gps_info: dict[str, Any] = {}
        for k, v in gps_info_raw.items():
            tag_name: str = GPSTAGS.get(int(k), str(k)) if isinstance(k, int) else str(k)
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
        logger.debug(
            "Converted GPS: latitude=%r, longitude=%r",
            latitude,
            longitude,
        )
        if latitude is None or longitude is None:
            logger.debug("GPS conversion failed: latitude or longitude is None.")
            return "Unknown location"

        def dms_to_dd(dms: tuple[float, float, float], ref: str) -> tuple[float, str]:
            # Ensure all elements are valid floats
            deg, min_, sec = dms
            if deg is None or min_ is None or sec is None:
                msg = "Invalid DMS tuple: contains None"
                raise ValueError(msg)
            dd = deg + min_ / 60.0 + sec / 3600.0
            ref_upper = ref.upper()
            sign = -1 if ref_upper in ("S", "W") else 1
            return (dd * sign, ref_upper)

        try:
            lat_ref_str: str = lat_ref.decode() if isinstance(lat_ref, bytes) else str(lat_ref)
            lon_ref_str: str = lon_ref.decode() if isinstance(lon_ref, bytes) else str(lon_ref)
            lat_dd, lat_card = dms_to_dd(latitude, lat_ref_str)
            lon_dd, lon_card = dms_to_dd(longitude, lon_ref_str)
            lat_dd = -abs(lat_dd) if lat_card == "S" else abs(lat_dd)
            lon_dd = -abs(lon_dd) if lon_card == "W" else abs(lon_dd)
            return f"{abs(lat_dd):.6f} {lat_card}, {abs(lon_dd):.6f} {lon_card}"
        except (ValueError, AttributeError, TypeError) as err:
            logger.debug("Failed to convert GPS DMS to decimal: %s", err)
            return "Unknown location"

    # --- End GPS extraction helper ---

    metadata["gps"] = _extract_gps_str(exif_data.get("GPSInfo"))
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
    """Print key EXIF data in a formatted table with colors and a title using tabulate."""
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
        tablefmt="outline",
        colalign=["left", "left"],
    )

    # Print title and table with decorative separators
    table_lines: list[str] = table.split("\n")
    table_width: int = max(Colors.visual_len(line) for line in table_lines) if table_lines else 80

    # Print title above the table, visually separated
    logger.info(
        Colors.colored("=" * table_width, Colors.BOLD, Colors.BLUE),
    )
    logger.info(
        Colors.colored(f"{title.center(table_width)}", Colors.BOLD, Colors.MAGENTA),
    )
    logger.info("")
    logger.info(
        Colors.colored("=" * table_width, Colors.BOLD, Colors.BLUE),
    )

    # Print the tabulated table
    for line in table_lines:
        logger.info(line)

    logger.info(
        Colors.colored("=" * table_width + "\n", Colors.BOLD, Colors.BLUE),
    )


def _get_available_fields(results: list[PerformanceResult]) -> list[str]:
    """Extract available GenerationResult fields and add PerformanceResult timing fields.

    Returns combined list of fields excluding 'text' and 'logprobs'.
    """
    # Determine GenerationResult fields (excluding 'text' and 'logprobs')
    gen_fields: list[str] = []
    for r in results:
        if r.generation is not None:
            gen_fields = [
                f.name for f in fields(r.generation) if f.name not in ("text", "logprobs")
            ]
            break

    # Combine with PerformanceResult timing fields
    return gen_fields + PERFORMANCE_TIMING_FIELDS


def _is_performance_timing_field(field_name: str) -> bool:
    """Check if a field is a PerformanceResult timing field."""
    return field_name in PERFORMANCE_TIMING_FIELDS


def _get_field_value(result: PerformanceResult, field_name: str) -> object:
    """Get field value from either GenerationResult or PerformanceResult."""
    if _is_performance_timing_field(field_name):
        # Get from PerformanceResult
        return getattr(result, field_name, "-")
    # Get from GenerationResult
    return getattr(result.generation, field_name, "-") if result.generation else "-"


# Helper function to sort results by elapsed time (lowest to highest)
def _sort_results_by_time(results: list[PerformanceResult]) -> list[PerformanceResult]:
    """Sort results by elapsed time (lowest to highest).

    Failed results (no timing data) are placed at the end.
    """

    def get_time_value(result: PerformanceResult) -> float:
        """Extract time value for sorting, with fallback for failed results."""
        if not result.success:
            return float("inf")  # Failed results go to the end

        # Use the elapsed_time field from PerformanceResult
        if result.elapsed_time is not None:
            return float(result.elapsed_time)

        # Fallback: calculate time from GenerationResult tokens-per-second if available
        if (
            result.generation
            and hasattr(result.generation, "generation_tokens")
            and hasattr(result.generation, "generation_tps")
            and result.generation.generation_tps > 0
        ):
            estimated_time = result.generation.generation_tokens / result.generation.generation_tps
            return float(estimated_time)

        return float("inf")  # No timing data available

    return sorted(results, key=get_time_value)


# Additional constants for multi-line formatting
MAX_SHORT_NAME_LENGTH: int = 12
MAX_LONG_NAME_LENGTH: int = 24
MAX_SIMPLE_NAME_LENGTH: int = 20
MAX_OUTPUT_LINE_LENGTH: int = 35
MAX_OUTPUT_TOTAL_LENGTH: int = 70


def _format_model_name_multiline(model_name: str) -> str:
    """Format model name for multi-line display in console table."""
    if "/" in model_name:
        parts = model_name.split("/")
        if len(parts[-1]) > MAX_SHORT_NAME_LENGTH:  # If final part is long, break it
            final_part = parts[-1]
            if len(final_part) > MAX_LONG_NAME_LENGTH:  # Very long, truncate
                return f"{'/'.join(parts[:-1])}/\n{final_part[:21]}..."
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
        if len(line2) > MAX_OUTPUT_LINE_LENGTH:  # Second line still too long
            line2 = line2[:32] + "..."
        return f"{line1}\n{line2}"
    # No good break point, just split at reasonable length
    truncated = "..." if len(output_text) > MAX_OUTPUT_TOTAL_LENGTH else ""
    return (
        f"{output_text[:MAX_OUTPUT_LINE_LENGTH]}\n"
        f"{output_text[MAX_OUTPUT_LINE_LENGTH:MAX_OUTPUT_TOTAL_LENGTH]}"
        f"{truncated}"
    )


def print_model_stats(results: list[PerformanceResult]) -> None:
    """Print a compact, aligned table of model performance metrics using tabulate.

    Displays GenerationResult fields with units and formatted numbers.
    Results are sorted by elapsed time (lowest to highest).
    """
    if not results:
        logger.info("No model results to display.")
        return

    # Sort results by elapsed time (lowest to highest)
    results = _sort_results_by_time(results)

    all_fields = _get_available_fields(results)

    if not all_fields:
        all_fields = []

    # Build compact headers with multi-line format
    headers: list[str] = ["Model"]
    field_names: list[str] = ["model"]  # Track original field names for alignment

    for f in all_fields:
        # Use multi-line headers to save space
        if f in FIELD_ABBREVIATIONS:
            line1, line2 = FIELD_ABBREVIATIONS[f]
            label = f"{line1}\n{line2}"
        else:
            # For fields not in abbreviations, create compact multi-line format
            base_label = format_field_label(f)
            if f in FIELD_UNITS:
                unit = FIELD_UNITS[f]
                label = f"{base_label}\n{unit}"
            else:
                label = base_label
        headers.append(label)
        field_names.append(f)

    headers.append("Output")
    field_names.append("output")

    # Build table rows with multi-line formatting for better space utilization
    rows: list[list[str]] = []
    for r in results:
        # Format model name with multi-line support
        model_display = _format_model_name_multiline(str(r.model_name))
        row: list[str] = [model_display]

        # Add generation fields and performance timing fields
        for f in all_fields:
            val = _get_field_value(r, f)
            val = format_field_value(f, val)
            # Format numbers with commas for console display
            is_numeric = isinstance(val, (int, float)) or is_numeric_string(val)
            if is_numeric and isinstance(val, (int, float, str)):
                # Use comma formatting for readability
                try:
                    num_val = float(val)
                    if abs(num_val) >= THOUSAND_THRESHOLD:
                        val = f"{num_val:,.0f}"  # With commas, whole numbers for large values
                    elif abs(num_val) >= 1:
                        val = f"{num_val:.3g}"
                    else:
                        val = f"{num_val:.2g}"
                except (ValueError, TypeError):
                    val = str(val)
            row.append(str(val))

        # Add output/diagnostic column with multi-line support
        if r.success and r.generation:
            out_val: str = str(getattr(r.generation, "text", ""))
        else:
            out_val = r.error_message or r.captured_output_on_fail or "-"

        # Format output for multi-line display
        out_val = _format_output_multiline(out_val)
        row.append(out_val)
        rows.append(row)

    # Determine column alignment using original field names
    colalign: list[str] = ["left"] + [
        "right" if is_numeric_field(field_name) else "left" for field_name in field_names[1:]
    ]

    # Generate compact table using plain format with multi-line headers
    # Use optimized widths for better vertical space utilization
    table = tabulate(
        rows,
        headers=headers,
        tablefmt="plain",
        colalign=colalign,
        maxcolwidths=[20, 6, 6, 6, 6, 7, 7, 6, 45],  # Allow longer model names and output
    )

    # Print the table with surrounding decorations
    table_lines = table.split("\n")
    max_width = max(len(line) for line in table_lines) if table_lines else 80

    logger.info("=" * max_width)
    for line in table_lines:
        logger.info(line)
    logger.info("=" * max_width)
    logger.info("Results sorted by elapsed time (fastest to slowest).")


def _prepare_table_data(
    results: list[PerformanceResult],
) -> tuple[list[str], list[list[str]], list[str]]:
    """Prepare table data for both HTML and Markdown reports using tabulate.

    Results are sorted by elapsed time (lowest to highest).

    Returns:
        Tuple of (headers, rows, field_names) where headers is a list of column names,
        rows is a list of row data lists, and field_names tracks the original field names
        for alignment purposes.

    """
    if not results:
        return [], [], []

    # Sort results by elapsed time (lowest to highest)
    results = _sort_results_by_time(results)

    all_fields = _get_available_fields(results)

    headers = ["Model"]
    field_names = ["model"]  # Track original field names for alignment

    # Create more compact, multi-line friendly headers with consistent formatting
    compact_headers = {
        "tokens": "Total<br>Tokens",
        "prompt_tokens": "Prompt<br>Tokens",
        "generation_tokens": "Generated<br>Tokens",
        "prompt_tps": "Prompt<br>Speed<br>(t/s)",
        "generation_tps": "Generation<br>Speed<br>(t/s)",
        "peak_memory": "Peak<br>Memory<br>(MB)",
        "active_memory": "Active<br>Memory<br>(MB)",
        "cached_memory": "Cached<br>Memory<br>(MB)",
        "time": "Total<br>Time<br>(s)",
        "duration": "Duration<br>(s)",
        "elapsed_time": "Elapsed<br>Time<br>(s)",
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
            # Format numbers
            is_numeric = isinstance(val, (int, float)) or is_numeric_string(val)
            if is_numeric and isinstance(val, (int, float, str)):
                val = fmt_num(val)
            row.append(str(val))

        # Add output/diagnostic column
        if r.success and r.generation:
            out_val = str(getattr(r.generation, "text", ""))
        else:
            out_val = r.error_message or r.captured_output_on_fail or "-"
        row.append(out_val)

        rows.append(row)

    return headers, rows, field_names


# --- HTML Report Generation ---
def generate_html_report(
    results: list[PerformanceResult],
    filename: Path,
    versions: dict[str, str],
    prompt: str,
) -> None:
    """Generate an HTML report with performance metrics and output using tabulate."""
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

    # Escape HTML characters in output text for HTML safety
    for i in range(len(rows)):
        for j in range(len(rows[i])):
            if j == len(rows[i]) - 1:  # Only escape in the last column (output column)
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

    # Use the shared FIELD_UNITS constant
    local_tz = get_localzone()

    # Build complete HTML document
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MLX VLM Performance Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f8f9fa; color: #212529; line-height: 1.6; }}
        h1 {{ color: #495057; text-align: center; margin-bottom: 15px; border-bottom: 3px solid #007bff; padding-bottom: 15px; font-size: 2.2em; }}
        h2 {{ color: #495057; margin-top: 30px; margin-bottom: 15px; border-bottom: 2px solid #6c757d; padding-bottom: 8px; font-size: 1.4em; }}
        .prompt-section {{ background-color: #e3f2fd; border-left: 4px solid #2196f3; padding: 20px; margin: 25px 0; border-radius: 6px; }}
        .prompt-section h3 {{ color: #1976d2; margin-top: 0; margin-bottom: 10px; font-size: 1.1em; }}
        .meta-info {{ color: #6c757d; font-style: italic; margin: 15px 0; text-align: center; }}
        table {{ border-collapse: collapse; width: 95%; margin: 30px auto; background-color: #fff; box-shadow: 0 2px 8px rgba(0,0,0,0.1); border-radius: 8px; overflow: hidden; }}
        th, td {{ border: 1px solid #dee2e6; padding: 8px 12px; vertical-align: top; }}
        th {{ background: linear-gradient(135deg, #e9ecef 0%, #f8f9fa 100%); font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-weight: 600; color: #495057; text-shadow: 0 1px 0 white; font-size: 14px; text-align: center; }}
        th.numeric {{ text-align: right; }}
        th.text {{ text-align: left; }}
        tr:nth-child(even):not(.failed-row) {{ background-color: #f8f9fa; }}
        tr:hover:not(.failed-row) {{ background-color: #e3f2fd; transition: background-color 0.2s; }}
        tr.failed-row {{ background-color: #f8d7da !important; color: #721c24; }}
        tr.failed-row:hover {{ background-color: #f5c6cb !important; }}
        .model-name {{ font-family: 'Courier New', Courier, monospace; font-weight: 500; text-align: left; color: #0d6efd; }}
        .error-message {{ font-weight: bold; color: #721c24; }}
        td.numeric {{ text-align: right; font-family: 'Courier New', monospace; }}
        td.text {{ text-align: left; }}
        caption {{ font-style: italic; color: #6c757d; margin-bottom: 10px; }}
        footer {{ margin-top: 40px; padding-top: 20px; border-top: 2px solid #dee2e6; }}
        footer h2 {{ color: #495057; }}
        footer ul {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
        footer code {{ background-color: #e9ecef; padding: 2px 4px; border-radius: 3px; color: #d63384; }}
    </style>
</head>
<body>
    <h1>MLX Vision Language Model Performance Report</h1>

    <div class="prompt-section">
        <h3>📝 Test Prompt</h3>
        <div>{html.escape(prompt).replace("\n", "<br>")}</div>
    </div>

    <h2>📊 Performance Results</h2>
    <div class="meta-info">
        Performance metrics and output for Vision Language Model processing<br>
        Results sorted by elapsed time (fastest to slowest) • Generated on {datetime.now(local_tz).strftime("%Y-%m-%d %H:%M:%S %Z")} • Failures shown but excluded from averages
    </div>

    {html_table}

    <footer>
        <h2>🔧 System Information</h2>
        <ul>
"""

    # Add library versions
    for name, ver in sorted(versions.items()):
        html_content += (
            f"            <li><code>{html.escape(name)}</code>: "
            f"<code>{html.escape(ver)}</code></li>\n"
        )

    html_content += f"""        </ul>
        <p><em>Report generated: {datetime.now(local_tz).strftime("%Y-%m-%d %H:%M:%S %Z")}</em></p>
    </footer>
</body>
</html>"""

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


def generate_markdown_report(
    results: list[PerformanceResult],
    filename: Path,
    versions: dict[str, str],
    prompt: str,
) -> None:
    """Generate a Markdown report with performance metrics and output using tabulate."""
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

    # Escape markdown characters in output text for Markdown safety
    for i in range(len(rows)):
        for j in range(len(rows[i])):
            if j == len(rows[i]) - 1:  # Only escape in the last column (output column)
                rows[i][j] = _escape_markdown_in_text(rows[i][j])

    # Determine column alignment using original field names
    colalign = ["left"] + [
        "right" if is_numeric_field(field_name) else "left" for field_name in field_names[1:]
    ]

    # Generate Markdown table using tabulate
    markdown_table = tabulate(
        rows,
        headers=markdown_headers,
        tablefmt="github",
        colalign=colalign,
    )

    # Build the complete markdown content
    md: list[str] = []
    md.append("# Model Performance Results\n")
    md.append(f"_Generated on {datetime.now(get_localzone()).strftime('%Y-%m-%d %H:%M:%S %Z')}_\n")
    md.append("")
    md.append("> **Prompt used:**\n>\n> " + prompt.replace("\n", "\n> ") + "\n")
    md.append("")
    md.append("**Note:** Results are sorted by elapsed time (fastest to slowest).\n")
    md.append("")
    md.append(markdown_table)
    md.append("\n---\n")
    md.append("**Library Versions:**\n")
    for name, ver in sorted(versions.items()):
        md.append(f"- `{name}`: `{ver}`")
    local_tz = get_localzone()
    md.append(
        f"\n_Report generated on: {datetime.now(local_tz).strftime('%Y-%m-%d %H:%M:%S %Z')}_\n",
    )

    try:
        with filename.open("w", encoding="utf-8") as f:
            f.write("\n".join(md))
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
    """Escape markdown characters in text to prevent formatting issues in markdown tables."""
    if not isinstance(text, str):
        return str(text)

    # Escape common markdown characters that could break table formatting
    # Order matters - escape backslashes first
    escape_chars = [
        ("\\", "\\\\"),  # Backslash
        ("`", "\\`"),  # Backticks
        ("*", "\\*"),  # Asterisks
        ("_", "\\_"),  # Underscores
        ("#", "\\#"),  # Headers
        ("|", "\\|"),  # Pipe (table separator)
        ("[", "\\["),  # Square brackets
        ("]", "\\]"),  # Square brackets
    ]

    result = text
    for char, escaped in escape_chars:
        result = result.replace(char, escaped)

    return result


def get_system_info() -> tuple[str, str]:
    """Get system architecture and GPU information."""
    arch: str = platform.machine()
    gpu_info: str = "Unknown"
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


def validate_temperature(temp: float) -> None:
    """Validate temperature parameter is within acceptable range."""
    if not isinstance(temp, (int, float)):
        msg: str = f"Temperature must be a number, got {type(temp)}"
        raise TypeError(msg)
    if not 0.0 <= temp <= 1.0:
        msg: str = f"Temperature must be between 0 and 1, got {temp}"
        raise ValueError(msg)


def validate_image_accessible(image_path: PathLike) -> None:
    """Validate image file is accessible and supported."""
    img_path_str: str = str(image_path)
    try:
        with (
            TimeoutManager(seconds=5),
            Image.open(img_path_str),
        ):
            pass
    except TimeoutError as err:
        msg: str = f"Timeout while reading image: {img_path_str}"
        raise OSError(msg) from err
    except UnidentifiedImageError as err:
        msg: str = f"File is not a recognized image format: {img_path_str}"
        raise ValueError(msg) from err
    except (OSError, ValueError) as err:
        msg: str = f"Error accessing image {img_path_str}: {err}"
        raise OSError(msg) from err


class ModelGenParams(NamedTuple):
    """Parameters for model generation."""

    model_path: str
    prompt: str
    max_tokens: int
    temperature: float
    trust_remote_code: bool


def _run_model_generation(
    params: ModelGenParams,
    image_path: Path,
    *,
    verbose: bool,
) -> GenerationResult:
    """Load HuggingFace model, format prompt and run MLX VLM generation.

    This function interfaces between HuggingFace's model repository system and
    MLX's optimized inference engine. The model loading process:
    1. Downloads model weights/config from HuggingFace Hub if not cached locally
    2. Loads the model into MLX's memory-efficient format
    3. Applies the model's specific chat template format to the user prompt
    4. Runs inference using MLX's Metal Performance Shaders acceleration

    Raise exceptions on failure.
    """
    model: object
    tokenizer: object

    # Load model from HuggingFace Hub - this handles automatic download/caching
    # and converts weights to MLX format for Apple Silicon optimization
    try:
        model, tokenizer = load(
            path_or_hf_repo=params.model_path,
            trust_remote_code=params.trust_remote_code,
        )
        config = model.config
    except Exception as load_err:
        # Capture any model loading errors (config issues, missing files, etc.)
        error_details = (
            f"Model loading failed: {load_err}\n\nFull traceback:\n{traceback.format_exc()}"
        )
        logger.exception("Failed to load model %s", params.model_path)
        raise ValueError(error_details) from load_err

    # Apply model-specific chat template - each model has its own conversation format
    # (e.g., Llama uses <|begin_of_text|>, Phi-3 uses <|user|>, etc.)
    formatted_prompt = apply_chat_template(
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
    output: GenerationResult = generate(
        model=model,
        processor=tokenizer,  # type: ignore[arg-type] # MLX VLM accepts both tokenizer types
        prompt=formatted_prompt,
        image=str(image_path),
        verbose=verbose,
        temperature=params.temperature,
        trust_remote_code=params.trust_remote_code,
        max_tokens=params.max_tokens,
    )
    end_time = time.perf_counter()

    # Add timing to the GenerationResult object
    # Since we can't modify the dataclass, we'll add it as an attribute
    output.time = end_time - start_time  # type: ignore[attr-defined]

    mx.eval(model.parameters())
    return output


class ProcessImageParams(NamedTuple):
    """Parameters for processing an image with a VLM.

    Attributes:
        model_identifier: Model path or identifier.
        image_path: Path to the image file.
        prompt: Prompt string for the model.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        timeout: Timeout in seconds.
        verbose: Verbose/debug flag.
        trust_remote_code: Allow remote code execution.

    """

    model_identifier: str
    image_path: Path
    prompt: str
    max_tokens: int
    temperature: float
    timeout: float
    verbose: bool
    trust_remote_code: bool


def process_image_with_model(params: ProcessImageParams) -> PerformanceResult:
    """Process an image with a Vision Language Model, managing stats and errors."""
    logger.info(
        "Processing '%s' with model: %s",
        str(getattr(params.image_path, "name", params.image_path)),
        params.model_identifier,
    )
    model: object | None = None
    tokenizer: object | None = None
    arch, gpu_info = get_system_info()
    try:
        validate_temperature(temp=params.temperature)
        validate_image_accessible(image_path=params.image_path)
        logger.debug("System: %s, GPU: %s", arch, gpu_info)
        with TimeoutManager(params.timeout):
            gen_params: ModelGenParams = ModelGenParams(
                model_path=params.model_identifier,
                prompt=params.prompt,
                max_tokens=params.max_tokens,
                temperature=params.temperature,
                trust_remote_code=params.trust_remote_code,
            )
            output: GenerationResult = _run_model_generation(
                params=gen_params,
                image_path=params.image_path,
                verbose=params.verbose,
            )
        return PerformanceResult(
            model_name=params.model_identifier,
            generation=output,
            success=True,
        )
    except TimeoutError as e:
        logger.exception("Timeout during model processing")
        return PerformanceResult(
            model_name=params.model_identifier,
            generation=None,
            success=False,
            error_stage="timeout",
            error_message=str(e),
        )
    except (OSError, ValueError) as e:
        logger.exception("Model processing error")
        return PerformanceResult(
            model_name=params.model_identifier,
            generation=None,
            success=False,
            error_stage="processing",
            error_message=str(e),
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
    separator_line: str = "=" * 80
    logger.info(separator_line)
    logger.info("%s", title.center(80))
    logger.info(separator_line)


def print_cli_section(title: str) -> None:
    """Print a formatted CLI section header."""
    separator_line: str = "-" * 60
    logger.info(separator_line)
    logger.info("[ %s ]", title.upper())
    logger.info(separator_line)


def print_cli_error(msg: str) -> None:
    """Print a formatted CLI error message."""
    logger.error("ERROR: %s", msg)


def print_model_result(result: PerformanceResult, *, verbose: bool = False) -> None:
    """Print model processing result with enhanced visual formatting and contextual colors."""
    model_short_name: str = result.model_name.split("/")[-1]

    if result.success:
        # Success header with green coloring
        success_msg: str = f"✓ SUCCESS: {model_short_name}"
        sys.stderr.write(Colors.colored(success_msg, Colors.BOLD, Colors.GREEN) + "\n")

        if result.generation:
            # Generated text with cyan highlighting for visibility
            gen_text: str = getattr(result.generation, "text", "N/A")
            gen_msg: str = f"  Generated Text: {Colors.colored(gen_text, Colors.CYAN)}"
            sys.stderr.write(gen_msg + "\n")

            # Performance metrics with white color for better visibility
            prompt_tokens: int = getattr(result.generation, "prompt_tokens", 0)
            generation_tokens: int = getattr(result.generation, "generation_tokens", 0)
            total_tokens: int = prompt_tokens + generation_tokens

            tokens_msg: str = f"  Tokens: {Colors.colored(fmt_num(total_tokens), Colors.WHITE)}"
            sys.stderr.write(tokens_msg + "\n")

            generation_tps: float = getattr(result.generation, "generation_tps", 0.0)
            tps_msg: str = (
                f"  Generation TPS: {Colors.colored(fmt_num(generation_tps), Colors.WHITE)}"
            )
            sys.stderr.write(tps_msg + "\n")

        if verbose and result.generation:
            # Section header in magenta
            header_msg = Colors.colored("  Performance Metrics:", Colors.BOLD, Colors.MAGENTA)
            sys.stderr.write(header_msg + "\n")

            # Debug: show all available fields in GenerationResult
            available_fields = [f.name for f in fields(result.generation)]
            logger.debug("  Available GenerationResult fields: %s", available_fields)

            # Time metric in white - try both 'time' and 'duration' fields
            time_val = getattr(result.generation, "time", None)
            if time_val is None:
                time_val = getattr(result.generation, "duration", 0.0)
            time_msg = f"    Time: {Colors.colored(f'{time_val:.2f}s', Colors.WHITE)}"
            sys.stderr.write(time_msg + "\n")

            # Memory fields with white coloring for visibility
            active_mem = getattr(result.generation, "active_memory", 0.0)
            cached_mem = getattr(result.generation, "cached_memory", 0.0)
            peak_mem = getattr(result.generation, "peak_memory", 0.0)

            if active_mem > 0:
                formatted_active = format_field_value("active_memory", active_mem)
                if isinstance(formatted_active, str):
                    active_msg = f"    Memory (Active Δ): {Colors.colored(f'{formatted_active} MB', Colors.WHITE)}"
                else:
                    active_msg = f"    Memory (Active Δ): {Colors.colored(f'{fmt_num(active_mem / MB_CONVERSION)} MB', Colors.WHITE)}"
                sys.stderr.write(active_msg + "\n")

            if cached_mem > 0:
                formatted_cached = format_field_value("cached_memory", cached_mem)
                if isinstance(formatted_cached, str):
                    cached_msg = f"    Memory (Cache Δ): {Colors.colored(f'{formatted_cached} MB', Colors.WHITE)}"
                else:
                    cached_msg = f"    Memory (Cache Δ): {Colors.colored(f'{fmt_num(cached_mem / MB_CONVERSION)} MB', Colors.WHITE)}"
                sys.stderr.write(cached_msg + "\n")

            if peak_mem > 0:
                formatted_peak = format_field_value("peak_memory", peak_mem)
                if isinstance(formatted_peak, str):
                    peak_msg = (
                        f"    Memory (Peak): {Colors.colored(f'{formatted_peak} MB', Colors.WHITE)}"
                    )
                else:
                    peak_msg = f"    Memory (Peak): {Colors.colored(f'{fmt_num(peak_mem / MB_CONVERSION)} MB', Colors.WHITE)}"
                sys.stderr.write(peak_msg + "\n")

            # Token details in white
            prompt_tokens_val = getattr(result.generation, "prompt_tokens", 0)
            prompt_msg = (
                f"    Prompt Tokens: {Colors.colored(fmt_num(prompt_tokens_val), Colors.WHITE)}"
            )
            sys.stderr.write(prompt_msg + "\n")

            generation_tokens_val = getattr(result.generation, "generation_tokens", 0)
            gen_tokens_msg = f"    Generation Tokens: {Colors.colored(fmt_num(generation_tokens_val), Colors.WHITE)}"
            sys.stderr.write(gen_tokens_msg + "\n")

            prompt_tps = getattr(result.generation, "prompt_tps", 0.0)
            prompt_tps_msg = f"    Prompt TPS: {Colors.colored(fmt_num(prompt_tps), Colors.WHITE)}"
            sys.stderr.write(prompt_tps_msg + "\n")
    else:
        # Failure header with red coloring
        failure_msg = f"✗ FAILED: {model_short_name}"
        sys.stderr.write(Colors.colored(failure_msg, Colors.BOLD, Colors.RED) + "\n")

        if result.error_stage:
            stage_msg = f"  Stage: {Colors.colored(result.error_stage, Colors.RED)}"
            sys.stderr.write(stage_msg + "\n")

        if result.error_message:
            error_msg = f"  Error: {Colors.colored(result.error_message, Colors.RED)}"
            sys.stderr.write(error_msg + "\n")

        if result.captured_output_on_fail:
            output_msg = f"  Output: {Colors.colored(result.captured_output_on_fail, Colors.RED)}"
            sys.stderr.write(output_msg + "\n")


def print_cli_separator() -> None:
    """Print a simple separator line."""
    logger.info("")


def setup_environment(args: argparse.Namespace) -> dict[str, str]:
    """Configure logging, collect versions, print warnings."""
    # Set DEBUG if verbose, else WARNING
    log_level: int = logging.DEBUG if args.verbose else logging.INFO
    # Remove all handlers and add only one
    logger.handlers.clear()
    handler: logging.StreamHandler[Any] = logging.StreamHandler(sys.stderr)
    formatter: ColoredFormatter = ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(log_level)
    logger.propagate = False  # Prevent double logging

    if args.verbose:
        logger.debug("Verbose/debug mode enabled.")

    library_versions: dict[str, str] = get_library_versions()
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
    print_cli_section(f"Scanning folder: {folder_path}")

    if args.folder == DEFAULT_FOLDER and not DEFAULT_FOLDER.is_dir():
        print_cli_error(f"Default folder '{DEFAULT_FOLDER}' does not exist.")

    image_path: Path | None = find_most_recent_file(folder_path)
    if image_path is None:
        print_cli_error(
            f"Could not find the most recent image file in {folder_path}. Exiting.",
        )
        sys.exit(1)

    resolved_image_path: Path = image_path.resolve()
    print_cli_section(f"Image File: {resolved_image_path.name}")
    logger.info("Full path: %s", str(resolved_image_path))

    try:
        with Image.open(resolved_image_path) as img:
            img.verify()
        print_image_dimensions(resolved_image_path)
    except (
        FileNotFoundError,
        UnidentifiedImageError,
        OSError,
    ) as img_err:
        print_cli_error(
            f"Cannot open or verify image {resolved_image_path}: {img_err}. Exiting.",
        )
        sys.exit(1)
    else:
        return resolved_image_path


def handle_metadata(image_path: Path, args: argparse.Namespace) -> MetadataDict:
    """Extract, print, and return image metadata."""
    print_cli_section("Image Metadata")

    metadata: MetadataDict = extract_image_metadata(image_path)

    # Display key metadata in a clean format
    logger.info("Date: %s", metadata.get("date", "N/A"))
    logger.info("Description: %s", metadata.get("description", "N/A"))
    logger.info("GPS Location: %s", metadata.get("gps", "N/A"))

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
        prompt_parts: list[str] = [
            (
                "Provide a factual caption, description, and keywords suitable for "
                "cataloguing, or searching for, the image."
            ),
            (
                f"\n\nContext: The image relates to '{metadata.get('description', '')}'"
                if metadata.get("description") and metadata["description"] != "N/A"
                else ""
            ),
            (
                f"\n\nThe photo was taken around {metadata.get('date', '')}"
                if metadata.get("date") and metadata["date"] != "Unknown date"
                else ""
            ),
            (
                f"near GPS {metadata.get('gps', '')}"
                if metadata.get("gps") and metadata["gps"] != "Unknown location"
                else ""
            ),
            (
                ". Focus on visual content, drawing on any available "
                "contextual information for specificity. Do not speculate."
            ),
        ]
        prompt = " ".join(filter(None, prompt_parts)).strip()
        logger.debug("Using generated prompt based on metadata.")

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


def process_models(
    args: argparse.Namespace,
    image_path: Path,
    prompt: str,
) -> list[PerformanceResult]:
    """Process images with the specified models or scan cache for available models.

    Returns a list of performance results with outputs and performance metrics.
    """
    # Validate model selection and warn about ineffective exclusions
    validate_and_warn_model_selection(args)

    model_identifiers: list[str]
    if args.models:
        # Case 1: Explicit models specified - apply exclusions to this list
        model_identifiers = args.models
        logger.info("Processing specified models: %s", ", ".join(model_identifiers))

        # Apply exclusions to explicit model list if specified
        if args.exclude:
            excluded_set = set(args.exclude)
            original_count = len(model_identifiers)

            # Filter out excluded models from explicit list
            model_identifiers = [model for model in model_identifiers if model not in excluded_set]

            excluded_count = original_count - len(model_identifiers)
            if excluded_count > 0:
                logger.info(
                    "Excluded %d model(s) from explicit list. Remaining: %d model(s)",
                    excluded_count,
                    len(model_identifiers),
                )
    else:
        # Case 2: No explicit models - scan cache and apply exclusions
        logger.info("Scanning cache for models to process...")
        model_identifiers = get_cached_model_ids()

        # Apply exclusions if specified and not using explicit model list
        if args.exclude and model_identifiers:
            excluded_set = set(args.exclude)
            original_count = len(model_identifiers)

            # Filter out excluded models (exact match)
            model_identifiers = [model for model in model_identifiers if model not in excluded_set]

            excluded_count = original_count - len(model_identifiers)
            if excluded_count > 0:
                logger.info(
                    "Excluded %d model(s) from cached models. Remaining: %d model(s)",
                    excluded_count,
                    len(model_identifiers),
                )

    results: list[PerformanceResult] = []
    if not model_identifiers:
        logger.error("No models specified or found in cache.")
        if not args.models:
            logger.error("Ensure models are downloaded and cache is accessible.")
    else:
        logger.info("Processing %d model(s)...", len(model_identifiers))
        for model_id in model_identifiers:
            print_cli_separator()
            print_cli_section(f"Processing Model: {model_id.split('/')[-1]}")

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
            )
            result: PerformanceResult = process_image_with_model(params)
            results.append(result)

            # Use the new structured output function
            print_model_result(result, verbose=args.verbose)
    return results


def finalize_execution(
    args: argparse.Namespace,
    results: list[PerformanceResult],
    library_versions: dict[str, str],
    overall_start_time: float,
    prompt: str,
) -> None:
    """Output summary statistics, generate reports, and display timing information."""
    if results:
        print_cli_section("Performance Summary")
        print_model_stats(results)

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
            )
            generate_markdown_report(
                results=results,
                filename=md_output_path,
                versions=library_versions,
                prompt=prompt,
            )

            logger.info("Reports successfully generated:")
            logger.info("  HTML: %s", str(html_output_path))
            logger.info("  Markdown: %s", str(md_output_path))
        except (OSError, ValueError):
            logger.exception("Failed to generate reports.")
    else:
        logger.warning("No models processed. No performance summary generated.")
        logger.info("Skipping report generation as no models were processed.")

    print_cli_section("Execution Summary")
    print_version_info(library_versions)
    overall_time: float = time.perf_counter() - overall_start_time
    logger.info("Total execution time: %.2f seconds", overall_time)


def main(args: argparse.Namespace) -> None:
    """Run CLI execution for MLX VLM model check."""
    overall_start_time: float = time.perf_counter()
    try:
        library_versions = setup_environment(args)
        print_cli_header("MLX Vision Language Model Check")
        image_path = find_and_validate_image(args)
        metadata = handle_metadata(image_path, args)
        prompt = prepare_prompt(args, metadata)
        results = process_models(args, image_path, prompt)
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

    # Parse arguments
    args: argparse.Namespace = parser.parse_args()
    # Print all command-line arguments if verbose is set
    if getattr(args, "verbose", False):
        print_cli_section("Command Line Parameters")
        for arg_name, arg_value in sorted(vars(args).items()):
            logger.info("  %s: %s", arg_name, arg_value)
    main(args)


if __name__ == "__main__":
    main_cli()
