#!/usr/bin/env python3
"""Image analysis and caption generation using MLX Vision Language Models."""

from __future__ import annotations

# Standard library imports
import argparse
import contextlib
import html
import logging
import platform
import re  # For ANSI code stripping
import signal
import subprocess
import sys
import time
import traceback
import types  # Added for TracebackType
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from re import Pattern
from typing import (
    Any,
    Callable,
    ClassVar,
    Final,
    NamedTuple,
    NoReturn,
    TextIO,
    TypeVar,
)

from huggingface_hub import HFCacheInfo, scan_cache_dir
from huggingface_hub import __version__ as hf_version
from huggingface_hub.errors import HFValidationError

# Third-party imports
try:
    import mlx.core as mx
except ImportError:
    print("Core dependency missing: mlx. Please install it.", file=sys.stderr)
    sys.exit(1)

try:
    from PIL import ExifTags, Image, UnidentifiedImageError
    from PIL.ExifTags import GPSTAGS, TAGS

    pillow_version = Image.__version__ if hasattr(Image, "__version__") else "N/A"
except ImportError:
    print(
        "Error: Pillow not found. Please install it (`pip install Pillow`).",
        file=sys.stderr,
    )
    pillow_version = "N/A"
    sys.exit(1)

# Local application/library specific imports
try:
    from mlx_vlm import __version__ as vlm_version
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
except ImportError:
    print(
        "Error: mlx-vlm not found. Please install it (`pip install mlx-vlm`).",
        file=sys.stderr,
    )
    sys.exit(1)

# Optional imports for version reporting
try:
    # Import the module first
    from mlx_lm import _version as mlx_lm_version_module

    # Then try to get its __version__ attribute, ensuring it's a string
    mlx_lm_version = str(getattr(mlx_lm_version_module, "__version__", "N/A"))
except ImportError:
    # If the module itself cannot be imported
    mlx_lm_version = "N/A"
except AttributeError:
    # If the module is imported but lacks a __version__ attribute
    mlx_lm_version = "N/A (module found, no version attr)"
try:
    import transformers

    transformers_version = transformers.__version__
    # Import specific tokenizer types
    from transformers.tokenization_utils import PreTrainedTokenizer
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
except ImportError:
    transformers_version = "N/A"
    # Define dummy types if transformers is not installed to avoid NameErrors later
    # Although the script exits earlier if transformers is missing for mlx_vlm
    PreTrainedTokenizer = type("PreTrainedTokenizer", (), {})
    PreTrainedTokenizerFast = type("PreTrainedTokenizerFast", (), {})


# Custom timeout context manager (for Python < 3.11)
# Note: This implementation relies on signal.SIGALRM and will not work on Windows.
class TimeoutManager(contextlib.ContextDecorator):
    """Manage a timeout context for code execution (UNIX only)."""

    def __init__(self, seconds: float) -> None:
        """Initialize a timeout manager with a timeout duration.

        Args:
            seconds: The timeout duration in seconds.
        """
        self.seconds: float = seconds
        # Accommodate signal.SIG_DFL, signal.SIG_IGN (integers)
        self.timer: Callable[[int, types.FrameType | None], Any] | int | None = None

    def _timeout_handler(
        self,
        _signum: int,
        _frame: types.FrameType | None,
    ) -> NoReturn:  # Use FrameType
        msg = f"Operation timed out after {self.seconds} seconds"
        raise TimeoutError(msg)

    def __enter__(self) -> TimeoutManager:
        """Enter the timeout context manager."""
        # Check if SIGALRM is available (won't be on Windows)
        if hasattr(signal, "SIGALRM"):
            if self.seconds > 0:
                try:
                    self.timer = signal.signal(
                        signal.SIGALRM,
                        self._timeout_handler,
                    )
                    signal.alarm(int(self.seconds))
                except ValueError as e:
                    # Running in a thread or environment where signals are restricted
                    logger.warning(
                        "Could not set SIGALRM for timeout: %s. Timeout disabled.",
                        e,
                    )
                    self.seconds = 0  # Disable timeout functionality
        elif self.seconds > 0:
            logger.warning(
                "Timeout functionality requires signal.SIGALRM, "
                "not available on this platform (e.g., Windows). "
                "Timeout disabled.",
            )
            self.seconds = 0  # Disable timeout functionality
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Exit the timeout context manager and clear the alarm."""
        # Only try to reset the alarm if it was successfully set
        if hasattr(signal, "SIGALRM") and self.seconds > 0 and self.timer is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self.timer)


# Configure logging
logger = logging.getLogger(__name__)
# BasicConfig called in main()


# --- ANSI Color Codes for Console Output ---
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
    _ansi_escape_re: ClassVar[Pattern[str]] = re.compile(
        r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])",
    )

    @staticmethod
    def colored(text: str, *colors: str) -> str:
        """Return text wrapped in ANSI color codes if enabled."""
        if not Colors._enabled or not colors:
            return text
        color_seq = "".join(colors)
        return f"{color_seq}{text}{Colors.RESET}"

    @staticmethod
    def visual_len(text: str) -> int:
        """Return the visual length of text, ignoring ANSI codes."""
        # Remove ANSI codes for accurate width
        return len(Colors._ansi_escape_re.sub("", text))


# --- Colored Logging Formatter ---
class ColoredFormatter(logging.Formatter):
    """A logging formatter that applies color to log messages based on their level."""

    LEVEL_COLORS: ClassVar[dict[int, str]] = {
        logging.DEBUG: Colors.CYAN,
        logging.INFO: Colors.GREEN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.RED + Colors.BOLD,
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with color based on its level."""
        color = self.LEVEL_COLORS.get(record.levelno, "")
        msg = super().format(record)
        if color:
            msg = Colors.colored(msg, color)
        return msg


# Configure logging to use ColoredFormatter
handler = logging.StreamHandler(sys.stderr)
formatter = ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.handlers.clear()
logger.addHandler(handler)

# Constants
MB_CONVERSION: Final[float] = 1024 * 1024

# Magic value constants
DMS_LEN: Final[int] = 3  # Degrees, Minutes, Seconds
DM_LEN: Final[int] = 2  # Degrees, Decimal Minutes
MAX_DEGREES: Final[int] = 180
MAX_MINUTES: Final[int] = 60
MAX_SECONDS: Final[int] = 60
MAX_TUPLE_LEN: Final[int] = 10
MAX_STR_LEN: Final[int] = 60
STR_TRUNCATE_LEN: Final[int] = 57
BASE_NAME_MAX_WIDTH: Final[int] = 45
COL_WIDTH: Final[int] = 12
MIN_NAME_COL_WIDTH: Final[int] = len("Model")


# --- Utility Functions ---
def _pad_text(text: str, width: int, right_align: bool = False) -> str:
    """Pad text to a specific visual width, accounting for ANSI codes."""
    pad_len = max(0, width - Colors.visual_len(text))
    padding = " " * pad_len
    return f"{padding}{text}" if right_align else f"{text}{padding}"


# --- Version Info ---
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
    max_len = max(len(k) for k in versions) + 1 if versions else 10
    for name, ver in sorted(versions.items()):
        status_color = Colors.GREEN if ver != "N/A" else Colors.YELLOW
        name_padded = name.ljust(max_len)
        logger.info("%s: %s", name_padded, Colors.colored(ver, status_color))
    logger.info("Generated: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


# --- Status Tag Helper ---
def status_tag(status: str) -> str:
    """Return a colored status tag for a given status string."""
    s = status.upper()
    if s == "SUCCESS":
        return Colors.colored("SUCCESS", Colors.BOLD, Colors.GREEN)
    if s == "FAIL":
        return Colors.colored("FAIL", Colors.BOLD, Colors.RED)
    if s == "WARNING":
        return Colors.colored("WARNING", Colors.BOLD, Colors.YELLOW)
    if s == "INFO":
        return Colors.colored("INFO", Colors.BOLD, Colors.BLUE)
    return Colors.colored(s, Colors.BOLD, Colors.MAGENTA)


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
EXIF_IMAGE_DESCRIPTION_TAG: Final[int] = (
    270  # Standard EXIF tag ID for ImageDescription
)
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
class MemoryStats(NamedTuple):
    """Represent memory statistics (deltas or peak values)."""

    active: float
    cached: float
    peak: float
    time: float

    @staticmethod
    def zero() -> MemoryStats:
        """Return a MemoryStats instance with all values zeroed."""
        return MemoryStats(0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class ModelResult:
    """Represent the result of processing a model, including failures."""

    model_name: str
    success: bool
    output: str | None = None
    stats: MemoryStats = field(default_factory=MemoryStats.zero)
    error_stage: str | None = None
    error_message: str | None = None
    captured_output_on_fail: str | None = None


# --- File Handling ---
# Simplified the `find_most_recent_file` function by using `max` with a generator.
def find_most_recent_file(folder: Path) -> Path | None:
    """Return the most recently modified file in a folder, or None."""
    if not folder.is_dir():
        logger.error(
            "Provided path is not a directory: %s",
            folder,
        )
        return None
    try:
        most_recent = max(
            (f for f in folder.iterdir() if f.is_file() and not f.name.startswith(".")),
            key=lambda f: f.stat().st_mtime,
            default=None,
        )
        if most_recent and logger.isEnabledFor(logging.DEBUG):
            logger.debug("Most recent file found: %s", most_recent)
        else:
            return most_recent
    except PermissionError:
        logger.exception(
            "Permission denied accessing folder: %s",
            folder,
        )
    except OSError:
        logger.exception(
            "OS error scanning folder %s",
            folder,
        )
    return None


# Improved error handling in `print_image_dimensions`.
def print_image_dimensions(image_path: Path) -> None:
    """Print the dimensions and megapixel count of an image file."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            mpx: float = (width * height) / 1_000_000
            logger.info(
                "Image dimensions: %s (%s MPixels)",
                Colors.colored(f"{width}x{height}", Colors.CYAN),
                Colors.colored(f"{mpx:.1f}", Colors.CYAN),
            )
    except (FileNotFoundError, UnidentifiedImageError):
        logger.exception(
            "Error with image file %s",
            image_path,
        )
    except Exception:
        logger.exception(
            "Unexpected error reading image dimensions for %s",
            image_path,
        )


# --- EXIF & Metadata Handling ---
@lru_cache(maxsize=128)
def get_exif_data(image_path: PathLike) -> ExifDict | None:
    """Extract EXIF data from an image file and return as a dictionary."""
    try:
        with Image.open(image_path) as img:
            exif_raw: Any = img.getexif()
            if not exif_raw:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("No EXIF data found in %s", image_path)
                return None
            exif_decoded: ExifDict = {}
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Raw EXIF data for %s: %s", image_path, exif_raw)
            # First pass: Process IFD0 (main image directory) tags
            for tag_id, value in exif_raw.items():
                # Skip SubIFD pointers, we'll handle them separately
                if tag_id in (ExifTags.Base.ExifOffset, ExifTags.Base.GPSInfo):
                    continue
                tag_name: str = TAGS.get(tag_id, str(tag_id))
                exif_decoded[tag_name] = value
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "IFD0 tags decoded for %s: %s",
                    image_path,
                    exif_decoded,
                )
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
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "Exif SubIFD merged for %s: %s",
                            image_path,
                            exif_decoded,
                        )
            except Exception:
                logger.exception("Could not extract Exif SubIFD")
            # Third pass: Process GPS IFD (if available)
            try:
                gps_ifd: Any = exif_raw.get_ifd(ExifTags.IFD.GPSInfo)
                if isinstance(gps_ifd, dict) and gps_ifd:
                    gps_decoded: dict[str, Any] = {
                        GPSTAGS.get(gps_tag_id, str(gps_tag_id)): gps_value
                        for gps_tag_id, gps_value in gps_ifd.items()
                    }
                    exif_decoded["GPSInfo"] = gps_decoded
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "GPS IFD merged for %s: %s",
                            image_path,
                            exif_decoded,
                        )
            except Exception as e:
                logger.warning("Could not extract GPS IFD: %s", e)
            return exif_decoded
    except (FileNotFoundError, UnidentifiedImageError):
        logger.exception("Error reading image file")
    except Exception:
        logger.exception("Unexpected error reading EXIF")
    return None


def _format_exif_date(date_str_input: str | bytes | float | None) -> str | None:
    """Return the EXIF date value as a string, or None if not available."""
    if date_str_input is None:
        return None
    if isinstance(date_str_input, str):
        return date_str_input.strip()
    try:
        # Attempt to convert non-string types to string
        return str(date_str_input).strip()
    except Exception:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Could not convert potential date value '%s' to string.",
                date_str_input,
            )
        return None


def _convert_gps_coordinate(
    ref: str | bytes | None,
    coord: float
    | tuple[float | int | object, ...]
    | list[float | int | object]
    | object,
) -> float | None:
    """Convert GPS coordinate and reference to decimal degrees, or None."""
    if not ref or not coord:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Missing GPS reference or coordinate.")
        return None

    try:
        # --- Reference Handling ---
        ref_str: str
        if isinstance(ref, bytes):
            try:
                ref_str = ref.decode("ascii")
            except UnicodeDecodeError:
                logger.warning("Invalid GPS reference encoding: %r", ref)
                return None
        elif isinstance(ref, str):
            ref_str = ref
        else:
            logger.warning(
                "Unexpected GPS reference type: %s",
                type(ref).__name__,
            )
            return None

        ref_upper = ref_str.strip().upper()
        if ref_upper not in ["N", "S", "E", "W"]:
            logger.warning("Unexpected GPS reference value: %s", ref_str)
            return None

        # --- Coordinate Value Handling ---
        degrees: float | None = None
        minutes: float | None = None
        seconds: float | None = 0.0  # Default seconds to 0

        def to_float(val: float | str | object) -> float | None:
            """Safely convert EXIF value (potentially Ratio) to float."""
            if hasattr(val, "numerator") and hasattr(
                val,
                "denominator",
            ):  # Check for Ratio-like object
                try:
                    # Ensure numerator/denominator are numeric before division
                    num = float(val.numerator)
                    den = float(val.denominator)
                    if den == 0:
                        logger.warning(
                            "Invalid Ratio in GPS (denominator is zero): %s",
                            val,
                        )
                        return None
                    return num / den
                except (
                    ValueError,
                    TypeError,
                    AttributeError,
                    ZeroDivisionError,
                ) as e:
                    logger.warning("Malformed Ratio in GPS: %s (%s)", val, e)
                    return None
            try:
                # Handle direct numeric types or strings representing numbers
                return float(val)
            except (ValueError, TypeError):
                logger.warning(
                    "Could not convert GPS value to float: %r (type: %s)",
                    val,
                    type(val).__name__,
                )
                return None

        # Check if coord is a sequence (tuple or list)
        if isinstance(coord, (tuple, list)):
            if len(coord) == DMS_LEN:  # Assume Degrees, Minutes, Seconds (DMS)
                degrees = to_float(coord[0])
                minutes = to_float(coord[1])
                seconds = to_float(coord[2])
            elif len(coord) == DM_LEN:  # Assume Degrees, Decimal Minutes (DM)
                degrees = to_float(coord[0])
                minutes = to_float(coord[1])
            elif len(coord) == 1:  # Assume Decimal Degrees in a sequence
                degrees = to_float(coord[0])
                minutes = 0.0  # Set minutes explicitly
            else:
                logger.warning(
                    "Unexpected GPS coordinate sequence length: %d for %s",
                    len(coord),
                    coord,
                )
                return None
        else:
            # Assume direct Decimal Degrees if not a sequence
            degrees = to_float(coord)
            minutes = 0.0  # Set minutes explicitly

        # --- Validation and Calculation ---
        if degrees is None or minutes is None or seconds is None:
            logger.warning(
                "Failed to extract valid numeric values from GPS coordinate: %s",
                coord,
            )
            return None

        # Validate ranges
        # Allow slightly wider range for degrees initially, sign applied later
        if not (
            0 <= abs(degrees) <= MAX_DEGREES
            and 0 <= minutes < MAX_MINUTES
            and 0 <= seconds < MAX_SECONDS
        ):
            logger.warning(
                "GPS values out of range: Deg=%s, Min=%s, Sec=%s",
                degrees,
                minutes,
                seconds,
            )
            return None

        # Calculate decimal degrees
        decimal = abs(degrees) + (minutes / 60.0) + (seconds / 3600.0)

        # Apply sign based on reference
        return -decimal if ref_upper in ["S", "W"] else decimal

    except Exception as e:
        # Catch-all for unexpected errors during conversion
        logger.exception(
            "Unexpected GPS conversion error for coord=%r, ref=%r: %s: %s",
            coord,
            ref,
            type(e).__name__,
            e,
            exc_info=logger.level <= logging.DEBUG,
        )
        return None


def _extract_exif_date(
    exif_data: ExifDict | None,
) -> tuple[str | None, str | None, str | None]:
    """Extract the best date string from EXIF data."""
    if not exif_data:
        return None, None, None

    # Prioritize DateTimeOriginal
    dt_original = exif_data.get("DateTimeOriginal")
    if dt_original:
        raw_date_str = _format_exif_date(dt_original)  # Get raw string
        if raw_date_str:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Using raw DateTimeOriginal from EXIF: '%s'",
                    raw_date_str,
                )
            return raw_date_str, "EXIF (DateTimeOriginal)", "DateTimeOriginal"

    # Fallback to other date tags
    for tag in EXIF_DATE_TAGS:
        if tag == "DateTimeOriginal":
            continue  # Already checked
        if tag in exif_data:
            raw_date_str = _format_exif_date(exif_data[tag])  # Get raw string
            if raw_date_str:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Using raw date from %s: '%s'", tag, raw_date_str)
                return raw_date_str, f"EXIF ({tag})", tag

    return None, None, None  # No valid EXIF date found


def _extract_exif_description(exif_data: ExifDict | None) -> str | None:
    """Extract the image description from EXIF data, or None."""
    if not exif_data:
        return None

    description = exif_data.get("ImageDescription", None)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Raw EXIF ImageDescription: '%s'", description)

    if description is not None:
        try:
            desc_str = str(description).strip()
            return (
                desc_str if desc_str else None
            )  # Return None if empty after stripping
        except Exception as desc_err:
            logger.warning(
                "Could not convert ImageDescription value '%s' to string: %s",
                description,
                desc_err,
            )
            return None
    return None


def _extract_gps_coordinates(
    exif_data: ExifDict | None,
    image_path_name: str,
) -> str | None:
    """Extract and format GPS coordinates from EXIF data, or None."""
    if (
        not exif_data
        or "GPSInfo" not in exif_data
        or not isinstance(exif_data["GPSInfo"], dict)
    ):
        return None

    gps_info = exif_data["GPSInfo"]
    lat = gps_info.get("GPSLatitude")
    lat_ref = gps_info.get("GPSLatitudeRef")
    lon = gps_info.get("GPSLongitude")
    lon_ref = gps_info.get("GPSLongitudeRef")

    if not (lat and lat_ref and lon and lon_ref):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Incomplete GPS tags found for %s", image_path_name)
        return None

    try:
        latitude = _convert_gps_coordinate(lat_ref, lat)
        longitude = _convert_gps_coordinate(lon_ref, lon)

        if latitude is not None and longitude is not None:
            # ISO 6709:2022 format: ±DD.DDDDDD±DDD.DDDDDD/
            lat_str = f"{latitude:+.6f}"
            lon_str = f"{longitude:+.6f}"
            gps_str = f"{lat_str}{lon_str}/"
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Extracted GPS (ISO 6709): %s for %s",
                    gps_str,
                    image_path_name,
                )
            return gps_str
        else:
            logger.warning(
                "Failed to convert GPS coordinates for %s. Lat: %s, Lon: %s",
                image_path_name,
                lat,
                lon,
            )
            return None
    except Exception as e:
        logger.warning(
            "Error processing GPS coordinates for %s: %s",
            image_path_name,
            e,
        )
        return None


def extract_image_metadata(
    image_path: Path,
    debug: bool = False,
) -> MetadataDict:
    """Extract key metadata (date, GPS, EXIF tags) from an image file."""
    metadata: MetadataDict = {}
    exif_data: ExifDict | None = get_exif_data(image_path)

    if debug:
        logger.debug(
            "Debug mode enabled for metadata extraction of %s",
            image_path.name,
        )
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Raw EXIF data for %s: %s", image_path.name, exif_data)

    # 1. Extract Date String (using helper)
    raw_exif_date, date_source, date_tag = _extract_exif_date(exif_data)
    if raw_exif_date:
        metadata["date"] = raw_exif_date  # Store the raw string
        metadata["date_source"] = date_source
        metadata["date_tag"] = date_tag
    else:
        # Fallback to file modification time (formatted as ISO 8601 string)
        try:
            mtime = Path(image_path).stat().st_mtime
            # Format as YYYY-MM-DD HH:MM:SS for consistency
            metadata["date"] = datetime.fromtimestamp(mtime).strftime(
                "%Y-%m-%d %H:%M:%S",
            )
            metadata["date_source"] = "File Modification Time"
            metadata["date_tag"] = "mtime"
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Using file modification time: %s",
                    metadata["date"],
                )
        except OSError as e:
            logger.warning(
                "Could not get modification time for %s: %s",
                image_path.name,
                e,
            )
            metadata["date"] = "Unknown"
            metadata["date_source"] = "Unavailable"
            metadata["date_tag"] = "None"

    # 2. Extract Description (using helper)
    description = _extract_exif_description(exif_data)
    metadata["description"] = description if description is not None else "N/A"

    # 3. Extract GPS (using helper)
    gps_coords = _extract_gps_coordinates(exif_data, image_path.name)
    if gps_coords:
        metadata["gps"] = gps_coords
    # No need for an else clause, if gps_coords is None, 'gps' key won't be in metadata

    # 4. Add other important EXIF tags
    if exif_data:
        for tag in IMPORTANT_EXIF_TAGS:
            # Avoid overwriting the date/desc we just determined
            if (
                tag not in EXIF_DATE_TAGS
                and tag != "ImageDescription"
                and tag in exif_data
            ):
                value = exif_data[tag]
                # Simple conversion for common types, avoid complex objects
                if isinstance(value, (str, int, float)):
                    metadata[tag] = str(value).strip()
                elif isinstance(value, bytes):
                    # Attempt to decode bytes, fallback to repr
                    try:
                        metadata[tag] = value.decode(
                            "utf-8",
                            errors="replace",
                        ).strip()
                    except UnicodeDecodeError:
                        metadata[tag] = repr(value)
                elif isinstance(value, tuple):
                    metadata[tag] = ", ".join(map(str, value))
                # Add other simple types if needed, but avoid deep structures

    return metadata


def pretty_print_exif(exif: ExifDict, verbose: bool = False) -> None:
    """Print key EXIF data in a formatted table with colors."""
    if not exif:
        logger.info("No EXIF data available.")
        return

    logger.info("\n--- %s ---", Colors.colored("Key EXIF Data", Colors.CYAN))
    tags_to_print: list[tuple[str, str, bool]] = []
    for tag, value in exif.items():
        tag_str = str(tag)
        if tag_str == "GPSInfo" and isinstance(value, dict):
            continue
        if isinstance(value, dict):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Skipping dictionary value for EXIF tag '%s' in pretty print.",
                    tag_str,
                )
            continue
        value_str: str
        if isinstance(value, bytes):
            try:
                decoded_str = value.decode("utf-8", errors="replace")
                value_str = (
                    decoded_str[:STR_TRUNCATE_LEN] + "..."
                    if len(decoded_str) > MAX_STR_LEN
                    else decoded_str
                )
            except Exception:
                value_str = f"<bytes len={len(value)}>"
        elif isinstance(value, tuple) and len(value) > MAX_TUPLE_LEN:
            value_str = f"<tuple len={len(value)}>"
        elif isinstance(value, bytearray):
            value_str = f"<bytearray len={len(value)}>"
        else:
            try:
                value_str = str(value)
                if len(value_str) > MAX_STR_LEN:
                    value_str = value_str[:STR_TRUNCATE_LEN] + "..."
            except Exception as str_err:
                # Log the specific error during string conversion
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Could not convert EXIF value for tag '%s' to string: %s",
                        tag_str,
                        str_err,
                    )
                value_str = f"<unrepresentable type: {type(value).__name__}>"

        is_important = tag_str in IMPORTANT_EXIF_TAGS
        if verbose or is_important:
            tags_to_print.append((tag_str, value_str, is_important))

    if not tags_to_print:
        logging.warning("No relevant EXIF tags found to display.")
        return
    tags_to_print.sort(key=lambda x: x[0])
    max_tag_len = (
        max(Colors.visual_len(t[0]) for t in tags_to_print) if tags_to_print else 20
    )
    max_val_len = (
        max(Colors.visual_len(t[1]) for t in tags_to_print) if tags_to_print else 40
    )
    min_width = 10
    max_tag_len = max(max_tag_len, min_width)
    max_val_len = max(max_val_len, min_width + 5)
    header_color = Colors.BLUE
    border_color = Colors.BLUE
    important_color = Colors.YELLOW
    # Use the extracted helper function
    pad = _pad_text
    print(
        Colors.colored(
            f"╔{'═' * (max_tag_len + 2)}╤{'═' * (max_val_len + 2)}╗",
            border_color,
        ),
    )
    print(
        f"{Colors.colored('║', border_color)} {pad(Colors.colored('Tag', header_color), max_tag_len)} {Colors.colored('│', border_color)} {pad(Colors.colored('Value', header_color), max_val_len)} {Colors.colored('║', border_color)}",
    )
    print(
        Colors.colored(
            f"╠{'═' * (max_tag_len + 2)}╪{'═' * (max_val_len + 2)}╣",
            border_color,
        ),
    )
    for tag_name, value_display, is_important_tag in tags_to_print:
        tag_display = (
            Colors.colored(tag_name, Colors.BOLD + important_color)
            if is_important_tag
            else tag_name
        )
        print(
            f"{Colors.colored('║', border_color)} {pad(tag_display, max_tag_len)} {Colors.colored('│', border_color)} {pad(value_display, max_val_len)} {Colors.colored('║', border_color)}",
        )
    print(
        Colors.colored(
            f"╚{'═' * (max_tag_len + 2)}╧{'═' * (max_val_len + 2)}╝",
            border_color,
        ),
    )


# --- Model Handling ---
def get_cached_model_ids() -> list[str]:
    """Return a list of model repo IDs from the Hugging Face cache."""
    if scan_cache_dir is None:
        logger.error(
            "huggingface_hub library not found. Cannot scan Hugging Face cache."
        )
        return []
    try:
        # Use CacheInfo (public class)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Scanning Hugging Face cache directory...")
        cache_info: HFCacheInfo = scan_cache_dir()
        model_ids = sorted([repo.repo_id for repo in cache_info.repos])
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Found %d potential models in Hugging Face cache: %s",
                len(model_ids),
                model_ids,
            )
        else:
            return model_ids
    except HFValidationError:
        logger.exception("Hugging Face cache directory invalid.")
        return []
    except FileNotFoundError:
        logger.exception("Hugging Face cache directory not found.")
        return []
    except Exception:
        logger.exception(
            "Unexpected error scanning Hugging Face cache.",
            exc_info=logger.level <= logging.DEBUG,
        )
        return []


def print_model_stats(results: list[ModelResult]) -> None:
    """Print a table summarizing model performance statistics."""
    if not results:
        logger.info("No model results to display.")
        return

    results.sort(
        key=lambda x: (
            not x.success,
            x.stats.time if x.success else float("inf"),
        ),
    )
    BASE_NAME_MAX_WIDTH: Final[int] = 45
    COL_WIDTH: Final[int] = 12
    MIN_NAME_COL_WIDTH: Final[int] = len("Model")
    # Color mapping
    # Each color maps to a logging level or UI element color
    COLORS: Final[types.SimpleNamespace] = types.SimpleNamespace(
        HEADER=Colors.CYAN,  # Match logger headers
        BORDER=Colors.BLUE,  # Keep borders blue for contrast
        SUMMARY=Colors.GREEN,  # Match SUCCESS logger color
        FAIL=Colors.RED,  # Match ERROR logger color
        FAIL_TEXT=Colors.RED,  # Match ERROR text color
        SUCCESS=Colors.GREEN,  # Match SUCCESS logger color
        MODEL=Colors.MAGENTA,  # Match model identifier color in logs
        VARIABLE=Colors.CYAN,  # Match variable/value colors in logs
        DIAG=Colors.YELLOW,  # Match WARNING diagnostic colors
    )

    def format_model_name(result: ModelResult) -> tuple[str, int]:
        base_name = result.model_name.split("/")[-1]
        display_name = base_name[:BASE_NAME_MAX_WIDTH] + (
            "..." if len(base_name) > BASE_NAME_MAX_WIDTH else ""
        )
        if not result.success:
            fail_suffix = f" [FAIL: {result.error_stage or '?'}]"
            display_name = Colors.colored(
                display_name,
                COLORS.MODEL,
            ) + Colors.colored(fail_suffix, Colors.BOLD, COLORS.FAIL)
        else:
            display_name = Colors.colored(display_name, COLORS.MODEL)
        return display_name, Colors.visual_len(display_name)

    name_displays = [format_model_name(r) for r in results]
    max_display_len = max(
        (length for _, length in name_displays),
        default=MIN_NAME_COL_WIDTH,
    )
    name_col_width = max(max_display_len, MIN_NAME_COL_WIDTH)

    def h_line(char: str) -> str:
        return Colors.colored(
            f"╔{'═' * (name_col_width + 2)}╤{'═' * (COL_WIDTH + 2)}╤"
            f"{'═' * (COL_WIDTH + 2)}╤{'═' * (COL_WIDTH + 2)}╤{'═' * (COL_WIDTH + 2)}╗"
            if char == "═"
            else f"╚{'═' * (name_col_width + 2)}╧{'═' * (COL_WIDTH + 2)}╧"
            f"{'═' * (COL_WIDTH + 2)}╧{'═' * (COL_WIDTH + 2)}╧{'═' * (COL_WIDTH + 2)}╝",
            COLORS.BORDER,
        )

    print("\n" + h_line("═"))
    headers = ["Model", "Active Δ", "Cache Δ", "Peak Mem", "Time"]
    header_row = Colors.colored(
        f"║ {_pad_text(Colors.colored(headers[0], COLORS.HEADER, Colors.BOLD), name_col_width)} │ "
        + " │ ".join(
            _pad_text(
                Colors.colored(h, COLORS.HEADER, Colors.BOLD),
                COL_WIDTH,
                right_align=True,
            )
            for h in headers[1:]
        )
        + " ║",
        COLORS.BORDER,
    )
    print(header_row)
    print(
        Colors.colored(
            f"╠{'═' * (name_col_width + 2)}╪{'═' * (COL_WIDTH + 2)}╪"
            f"{'═' * (COL_WIDTH + 2)}╪{'═' * (COL_WIDTH + 2)}╪{'═' * (COL_WIDTH + 2)}╣",
            COLORS.BORDER,
        ),
    )
    successful_results = []
    for result, (display_name, _) in zip(results, name_displays):
        if result.success:
            successful_results.append(result)
            stats = [
                Colors.colored(
                    "%s MB",
                    COLORS.VARIABLE,
                )
                % f"{result.stats.active:,.0f}",
                Colors.colored(
                    "%s MB",
                    COLORS.VARIABLE,
                )
                % f"{result.stats.cached:,.0f}",
                Colors.colored("%s MB", COLORS.VARIABLE) % f"{result.stats.peak:,.0f}",
                Colors.colored("%s s", COLORS.VARIABLE) % f"{result.stats.time:.2f}",
            ]
        else:
            stats = [Colors.colored("-", COLORS.FAIL_TEXT)] * 4
        row = Colors.colored(
            f"║ {_pad_text(display_name, name_col_width)} │ "
            + " │ ".join(_pad_text(stat, COL_WIDTH, right_align=True) for stat in stats)
            + " ║",
            COLORS.BORDER,
        )
        print(row)
    if successful_results:
        print(
            Colors.colored(
                f"╠{'═' * (name_col_width + 2)}╪{'═' * (COL_WIDTH + 2)}╪"
                f"{'═' * (COL_WIDTH + 2)}╪{'═' * (COL_WIDTH + 2)}╪{'═' * (COL_WIDTH + 2)}╣",
                COLORS.BORDER,
            ),
        )
        avg_stats = [
            sum(r.stats.active for r in successful_results) / len(successful_results),
            sum(r.stats.cached for r in successful_results) / len(successful_results),
            max(r.stats.peak for r in successful_results),
            sum(r.stats.time for r in successful_results) / len(successful_results),
        ]
        summary_stats = [
            f"{avg_stats[0]:,.0f} MB",
            f"{avg_stats[1]:,.0f} MB",
            f"{avg_stats[2]:,.0f} MB",
            f"{avg_stats[3]:.2f} s",
        ]
        summary_title = Colors.colored(
            f"AVG/PEAK ({len(successful_results)} Success)",
            COLORS.SUMMARY,
            Colors.BOLD,
        )
        summary_row = Colors.colored(
            f"║ {_pad_text(summary_title, name_col_width)} │ "
            + " │ ".join(
                _pad_text(
                    Colors.colored(stat, COLORS.SUMMARY, Colors.BOLD),
                    COL_WIDTH,
                    right_align=True,
                )
                for stat in summary_stats
            )
            + " ║",
            COLORS.BORDER,
        )
        print(summary_row)
    print(h_line("╝"))
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Displayed stats for %d models (%d successful)",
            len(results),
            len(successful_results),
        )


# --- HTML Report Generation ---
def generate_html_report(
    results: list[ModelResult],
    filename: Path,
    versions: dict[str, str],
    prompt: str,
) -> None:
    """Generate an HTML file with model stats, outputs, errors, and versions."""
    if not results:
        logger.warning("No results to generate HTML report.")
        return

    results.sort(
        key=lambda x: (not x.success, x.stats.time if x.success else 0),
    )

    html_start = (
        """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Performance Results</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f8f9fa; color: #212529; line-height: 1.6; }
        h1 { text-align: center; color: #343a40; border-bottom: 2px solid #dee2e6; padding-bottom: 10px; margin-bottom: 30px; }
        .prompt-block { background: #e9ecef; border-left: 4px solid #007bff; padding: 12px 18px; margin: 20px auto 30px auto; max-width: 900px; font-size: 1.05em; font-family: 'Fira Mono', 'Consolas', monospace; color: #343a40; }
        table { border-collapse: collapse; width: 95%; margin: 30px auto; box-shadow: 0 4px 8px rgba(0,0,0,0.1); background-color: #ffffff; }
        th, td { border: 1px solid #dee2e6; padding: 12px 15px; text-align: left; vertical-align: top; }
        th { background-color: #e9ecef; font-weight: 600; color: #495057; position: sticky; top: 0; z-index: 1; }
        tr:nth-child(even):not(.failed-row) { background-color: #f8f9fa; }
        tr:not(.failed-row):hover { background-color: #e2e6ea; }
        td.numeric, th.numeric { text-align: right; font-variant-numeric: tabular-nums; }
        .summary td { font-weight: bold; background-color: #d1ecf1; color: #0c5460; border-color: #bee5eb; }
        caption { caption-side: bottom; padding: 15px; font-style: italic; color: #6c757d; text-align: center; }
        .model-name { font-family: 'Courier New', Courier, monospace; font-weight: 500; }
        .model-output, .captured-output pre { white-space: pre-wrap; word-wrap: break-word; max-width: 600px; font-size: 0.9em; }
        .captured-output { background-color: #fff3cd; border-left: 3px solid #ffeeba; padding: 8px; margin-top: 10px; color: #856404; }
        .captured-output strong { color: #856404; }
        .captured-output pre { margin: 0; padding: 5px; background-color: transparent; border: none; color: #856404; }
        tr.failed-row { background-color: #f8d7da !important; color: #721c24; }
        tr.failed-row:hover { background-color: #f5c6cb !important; }
        tr.failed-row .model-output { font-style: italic; color: #721c24; }
        tr.failed-row td.numeric { color: #721c24; font-style: italic; }
        .error-message { font-weight: bold; display: block; color: #721c24; }
        footer { margin-top: 30px; text-align: center; font-size: 0.85em; color: #6c757d; }
        footer h2 { font-size: 1.1em; color: #495057; margin-bottom: 10px;}
        footer ul { list-style: none; padding: 0; margin: 0 0 10px 0; }
        footer li { display: inline-block; margin: 0 10px; }
        footer code { background-color: #e9ecef; padding: 2px 4px; border-radius: 3px; }
        footer p { margin-top: 5px; }
    </style>
</head>
<body>
    <h1>Model Performance Summary</h1>
    <div class="prompt-block"><strong>Prompt used:</strong><br>"""
        + html.escape(prompt).replace("\n", "<br>")
        + """</div>
    <table>
        <caption>Performance metrics and output/errors for Vision Language Model processing. Generated on """
        + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        + """. Failures shown but excluded from averages.</caption>
        <thead>
            <tr>
                <th>Model</th>
                <th class="numeric">Active Δ (MB)</th> <th class="numeric">Cache Δ (MB)</th>
                <th class="numeric">Peak Mem (MB)</th> <th class="numeric">Time (s)</th>
                <th>Result / Error / Captured Output</th>
            </tr>
        </thead>
        <tbody>
"""
    )
    html_rows: str = ""
    successful_results: list[ModelResult] = [r for r in results if r.success]

    for result in results:
        model_disp_name = html.escape(result.model_name)
        row_class = ""
        result_content = ""
        stats_cells = ""

        if result.success:
            escaped_output = html.escape(result.output or "")
            # Highlight model output in HTML
            result_content = (
                f'<div class="model-output"><strong>{escaped_output}</strong></div>'
            )
            stats_cells = f"""
                <td class="numeric">{result.stats.active:,.0f}</td>
                <td class="numeric">{result.stats.cached:,.0f}</td>
                <td class="numeric">{result.stats.peak:,.0f}</td>
                <td class="numeric">{result.stats.time:.2f}</td>
            """
        else:
            row_class = ' class="failed-row"'
            error_message = html.escape(result.error_message or "Unknown error")
            result_content = f'<span class="error-message">{error_message}</span>'
            if result.captured_output_on_fail:
                captured_output = html.escape(result.captured_output_on_fail)
                result_content += f'<div class="captured-output"><strong>Captured Output:</strong><pre>{captured_output}</pre></div>'
            stats_cells = """
                <td class="numeric">-</td>
                <td class="numeric">-</td>
                <td class="numeric">-</td>
                <td class="numeric">-</td>
            """

        html_rows += f"""
            <!-- Data row for model: {model_disp_name} -->
            <tr{row_class}>
                <td class="model-name">{model_disp_name}</td>
                {stats_cells}
                <td>{result_content}</td>
            </tr>
"""

    html_summary_row: str = ""
    if successful_results:
        avg_active = sum(r.stats.active for r in successful_results) / len(
            successful_results,
        )
        avg_cache = sum(r.stats.cached for r in successful_results) / len(
            successful_results,
        )
        max_peak = max(r.stats.peak for r in successful_results)
        avg_time = sum(r.stats.time for r in successful_results) / len(
            successful_results,
        )
        summary_title = f"AVG/PEAK ({len(successful_results)} Success)"
        # Format summary stats as comma-separated integers
        html_summary_row = f"""
            <!-- Summary Row (Based on Successful Runs) -->
            <tr class="summary">
                <td>{summary_title}</td>
                <td class="numeric">{avg_active:,.0f}</td>
                <td class="numeric">{avg_cache:,.0f}</td>
                <td class="numeric">{max_peak:,.0f}</td>
                <td class="numeric">{avg_time:.2f}</td>
                <td></td>
            </tr>
"""

    # --- Add Version Info Footer ---
    html_footer: str = "<footer>\n<h2>Library Versions</h2>\n<ul>\n"
    # Use sorted items for consistent order in HTML
    for name, ver in sorted(versions.items()):
        status_style = "color: green;" if ver != "N/A" else "color: orange;"
        html_footer += f'<li>{html.escape(name)}: <code style="{status_style}">{html.escape(ver)}</code></li>\n'
    html_footer += "</ul>\n"
    # Add date to footer
    html_footer += (
        f"<p>Report generated on: {datetime.now().strftime('%Y-%m-%d')}</p>\n</footer>"
    )
    # -----------------------------

    html_end = f"""
        </tbody>
    </table>
    <!-- End of Table -->
    {html_footer}
</body>
</html>
"""
    html_content: str = html_start + html_rows + html_summary_row + html_end

    try:
        # *** Restore TextIO type hint for file handle 'f' ***
        f: TextIO
        with Path.open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(
            "HTML report saved to: %s",
            Colors.colored(str(filename.resolve()), Colors.GREEN),
        )
    except OSError:
        logger.exception(
            "Failed to write HTML report to file.",
        )
    except Exception:
        logger.exception(
            "An unexpected error occurred while writing HTML report.",
            exc_info=logger.level <= logging.DEBUG,
        )


def generate_markdown_report(
    results: list[ModelResult],
    filename: Path,
    versions: dict[str, str],
    prompt: str,
) -> None:
    """Generate a Markdown file with model stats, output/errors, failures, and versions. All table cells are left- and top-aligned."""
    if not results:
        logger.warning("No results to generate Markdown report.")
        return

    results.sort(
        key=lambda x: (not x.success, x.stats.time if x.success else 0),
    )
    successful_results: list[ModelResult] = [r for r in results if r.success]

    # Table header
    md: list[str] = []
    # Add a style block to force top alignment for all table cells (works in GitHub and VS Code preview)
    md.append(
        "<style>table td, table th { vertical-align: top !important; text-align: left !important; }</style>",
    )
    md.append("# Model Performance Results\n")
    md.append(
        f"_Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n",
    )
    md.append("")
    md.append(
        "> **Prompt used:**\n>\n> " + prompt.replace("\n", "\n> ") + "\n",
    )
    md.append("")
    md.append(
        "| Model | Active Δ (MB) | Cache Δ (MB) | Peak Mem (MB) | Time (s) | Output / Error / Diagnostics |",
    )
    # All columns left-aligned:
    md.append(
        "|:------|:--------------|:-------------|:--------------|:---------|:-----------------------------|",
    )

    for result in results:
        model_disp_name: str = f"`{result.model_name}`"
        if result.success:
            stats = [
                f"{result.stats.active:,.0f}",
                f"{result.stats.cached:,.0f}",
                f"{result.stats.peak:,.0f}",
                f"{result.stats.time:.2f}",
            ]
            # Replace newlines with <br> in output
            output_text: str = (result.output or "").replace("\n", "<br>")
            output_md: str = output_text
        else:
            stats = ["-", "-", "-", "-"]
            error_msg: str = result.error_message or "Unknown error"
            # Replace newlines in error message with <br>
            error_text: str = error_msg.replace("\n", "<br>")
            output_md: str = f"**ERROR:** {error_text}"
            if result.captured_output_on_fail:
                # Replace newlines in captured output with <br>
                captured: str = result.captured_output_on_fail.replace(
                    "\n",
                    "<br>",
                )
                output_md += f"<br>**Captured Output:** {captured}"

        # Escape any pipe characters in the output to preserve table formatting
        output_md = output_md.replace("|", "\\|")
        md.append(
            f"| {model_disp_name} | {stats[0]} | {stats[1]} | {stats[2]} | {stats[3]} | {output_md} |",
        )

    # Summary row (no changes needed here, as it doesn't contain multiline content)
    if successful_results:
        avg_active = sum(r.stats.active for r in successful_results) / len(
            successful_results,
        )
        avg_cache = sum(r.stats.cached for r in successful_results) / len(
            successful_results,
        )
        max_peak = max(r.stats.peak for r in successful_results)
        avg_time = sum(r.stats.time for r in successful_results) / len(
            successful_results,
        )
        summary_title = f"**AVG/PEAK ({len(successful_results)} Success)**"
        summary_stats = [
            f"{avg_active:,.0f}",
            f"{avg_cache:,.0f}",
            f"{max_peak:,.0f}",
            f"{avg_time:.2f}",
        ]
        md.append(
            f"| {summary_title} | {summary_stats[0]} | {summary_stats[1]} | {summary_stats[2]} | {summary_stats[3]} |  |",
        )

    # Version info
    md.append("\n---\n")
    md.append("**Library Versions:**\n")
    for name, ver in sorted(versions.items()):
        md.append(f"- `{name}`: `{ver}`")
    md.append(
        f"\n_Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n",
    )

    try:
        with Path.open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(md))
        logger.info(
            "Markdown report saved to: %s",
            Colors.colored(str(filename.resolve()), Colors.GREEN),
        )
    except OSError:
        logger.exception(
            "Failed to write Markdown report to file.",
        )
    except Exception:
        logger.exception(
            "An unexpected error occurred while writing Markdown report.",
        )


def get_system_info() -> tuple[str, str]:
    """Get system architecture and GPU information."""
    arch: str = platform.machine()
    gpu_info: str = "Unknown"
    try:
        # Try to get GPU info on macOS
        if platform.system() == "Darwin":
            result: subprocess.CompletedProcess[str] = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if result.returncode == 0:
                # Extract GPU info from system_profiler output
                gpu_lines: list[str] = [
                    line
                    for line in result.stdout.split("\n")
                    if "Chipset Model:" in line
                ]
                if gpu_lines:
                    gpu_info = gpu_lines[0].split("Chipset Model:")[-1].strip()
    except (subprocess.SubprocessError, TimeoutError):
        pass
    return arch, gpu_info


# --- Model Processing Core ---
def validate_inputs(
    image_path: PathLike,
    temperature: float = 0.0,
) -> None:
    """Validate input paths and parameters."""
    img_path: Path = Path(image_path)
    if not img_path.exists():
        msg = f"Image not found: {img_path}"
        raise FileNotFoundError(msg)
    if not img_path.is_file():
        msg = f"Not a file: {img_path}"
        raise ValueError(msg)
    if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        msg = f"Unsupported image format: {img_path.suffix}"
        raise ValueError(msg)

    validate_temperature(temperature)


def validate_temperature(temp: float) -> None:
    """Validate temperature parameter is within acceptable range."""
    if not isinstance(temp, (int, float)):
        msg = f"Temperature must be a number, got {type(temp)}"
        raise TypeError(msg)
    if not 0.0 <= temp <= 1.0:
        msg = f"Temperature must be between 0 and 1, got {temp}"
        raise ValueError(msg)


def validate_image_accessible(image_path: Path) -> None:
    """Validate image file is accessible and supported."""
    try:
        with (
            TimeoutManager(5),
            Image.open(image_path) as img,
        ):  # 5 second timeout for read test
            img.verify()
    except TimeoutError:
        msg = f"Timeout while reading image: {image_path}"
        raise OSError(msg)
    except UnidentifiedImageError:
        msg = f"File is not a recognized image format: {image_path}"
        raise ValueError(msg)
    except Exception as e:
        msg = f"Error accessing image {image_path}: {e}"
        raise OSError(msg)


def _run_model_generation(
    model_identifier: str,
    image_path: Path,
    prompt: str,
    max_tokens: int,
    verbose: bool,
    trust_remote_code: bool,
    temperature: float,
) -> tuple[
    str,
    Any,
    Any,
]:  # Returns (output, model, tokenizer) - model/tokenizer needed for cleanup
    """Load model, format prompt and run generation.
    Raise exceptions on failure.
    """
    model = tokenizer = None  # Ensure they are defined in this scope
    try:
        # Load model and tokenizer
        model, tokenizer = load(
            model_identifier,
            trust_remote_code=trust_remote_code,
        )
        config: dict[str, Any] = load_config(
            model_identifier,
            trust_remote_code=trust_remote_code,
        )

        # Prepare prompt
        formatted_prompt: str = apply_chat_template(
            tokenizer,
            config,
            prompt,
            num_images=1,
        )

        # Generate output
        # output: Optional[str] = generate(
        output, _stats = generate(
            model=model,
            processor=tokenizer,  # Type checking handled by function signature
            prompt=formatted_prompt,
            image=image_path.as_posix(),
            max_tokens=max_tokens,
            verbose=verbose,
            temp=temperature,
        )

        # Ensure all computations involving the model are done before measuring memory/time
        mx.eval(
            model.parameters(),
        )  # Evaluate model parameters if needed after generation

        return (
            output if output is not None else "[No model output]",
            model,
            tokenizer,
        )

    except Exception:
        # If loading failed, model might be None. If generation failed, model is likely loaded.
        # Re-raise the exception to be caught by the outer function, which handles ModelResult creation.
        # We pass model/tokenizer back mainly for the finally block in the caller.
        raise  # Re-raise the original exception


def process_image_with_model(
    model_identifier: str,
    image_path: Path,
    prompt: str,
    max_tokens: int,
    temperature: float = DEFAULT_TEMPERATURE,
    timeout: float = DEFAULT_TIMEOUT,
    verbose: bool = False,
    trust_remote_code: bool = False,
) -> ModelResult:
    """Process an image with a Vision Language Model, managing stats and errors."""
    logger.info(
        "Processing '%s' with model: %s",
        image_path.name,
        Colors.colored(model_identifier, Colors.MAGENTA),
    )

    model = tokenizer = None  # Initialize here for the finally block
    arch, gpu_info = get_system_info()
    start_time: float = 0.0
    initial_mem: float = 0.0
    initial_cache: float = 0.0
    output: str | None = None
    error_stage: str = "initialization"

    try:
        validate_temperature(temperature)
        validate_image_accessible(image_path)
        error_stage = "validation"

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"System: {arch}, GPU: {gpu_info}")

        # --- Capture initial state BEFORE model operations ---
        initial_mem = mx.get_active_memory() / MB_CONVERSION
        initial_cache = mx.get_cache_memory() / MB_CONVERSION
        start_time = time.perf_counter()
        # -----------------------------------------------------

        with TimeoutManager(timeout):
            error_stage = "load/generate"  # Stage if timeout occurs here
            # Call the internal generation function
            output, model, tokenizer = _run_model_generation(
                model_identifier=model_identifier,
                image_path=image_path,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=verbose,
                trust_remote_code=trust_remote_code,
                temperature=temperature,
            )
            error_stage = "post-generate"  # Stage after successful generation

        # --- Capture final state AFTER model operations ---
        end_time = time.perf_counter()
        final_active_mem = mx.get_active_memory() / MB_CONVERSION
        final_cache_mem = mx.get_cache_memory() / MB_CONVERSION
        peak_mem = mx.get_peak_memory() / MB_CONVERSION
        # --------------------------------------------------

        final_stats = MemoryStats(
            active=final_active_mem - initial_mem,
            cached=final_cache_mem - initial_cache,
            peak=peak_mem,
            time=end_time - start_time,
        )

        return ModelResult(
            model_name=model_identifier,
            success=True,
            output=output,  # Use the output from _run_model_generation
            stats=final_stats,
        )

    except TimeoutError:
        # Timeout specifically handled
        logger.exception(
            "Timeout (%ss) during '%s' for model %s",
            timeout,
            error_stage,
            model_identifier,
        )
        return ModelResult(
            model_name=model_identifier,
            success=False,
            error_stage="timeout",
            error_message=f"Operation timed out after {timeout} seconds during {error_stage}",
        )
    except Exception as e:
        # Determine stage more accurately based on where the exception occurred
        # If model is None here, it likely failed during load within _run_model_generation
        if model is None and error_stage == "load/generate":
            error_stage = "model_load"

        logger.exception(
            "Failed during '%s' for model %s: %s: %s",
            error_stage,
            model_identifier,
            type(e).__name__,
            e,
        )
        if logger.isEnabledFor(logging.DEBUG):
            traceback.print_exc()

        return ModelResult(
            model_name=model_identifier,
            success=False,
            error_stage=error_stage,
            error_message=str(e),
            # captured_output_on_fail might be added here if needed, but requires more complex handling
        )
    finally:
        # Ensure cleanup happens regardless of success/failure
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        # Clear cache and reset peak memory after each model run
        mx.clear_cache()
        mx.reset_peak_memory()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Cleaned up resources for model {model_identifier}")


# --- Main Execution Helper Functions ---


def print_cli_header(title: str) -> None:
    """Print a formatted CLI header with the given title."""
    logger.info(f"\n{'=' * 80}")
    logger.info(f"{title.center(80)}")
    logger.info(f"{'=' * 80}\n")


def print_cli_section(title: str) -> None:
    """Print a formatted CLI section header."""
    logging.info(f"\n--- {title} ---")


def print_cli_error(msg: str) -> None:
    """Print a formatted CLI error message."""
    logger.error(f"Error: {msg}")


def setup_environment(args: argparse.Namespace) -> dict[str, str]:
    """Configure logging, collect versions, print warnings."""
    log_level: int = (
        logging.DEBUG
        if args.debug
        else (logging.VERBOSE if args.verbose else logging.INFO)
    )
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
        force=True,
    )
    if args.debug:
        logger.debug("Debug mode enabled.")
    elif args.verbose:
        logger.info("Verbose mode enabled.")

    library_versions: dict[str, str] = get_library_versions()
    if args.debug:  # Only print versions in debug mode initially
        print_version_info(library_versions)

    if args.trust_remote_code:
        logger.warning(
            "--- SECURITY WARNING ---"
        )
        logger.warning(
            "`--trust-remote-code` is enabled."
        )
        logger.warning(
            "-----------------------"
        )

    return library_versions


def find_and_validate_image(args: argparse.Namespace) -> Path:
    """Find and validate the image file to process from arguments."""
    folder_path: Path = args.folder.resolve()
    print_cli_section(f"Scanning folder: {folder_path}")
    if args.folder == DEFAULT_FOLDER and not DEFAULT_FOLDER.is_dir():
        print_cli_error(f"Default folder '{DEFAULT_FOLDER}' does not exist.")
    image_path: Path | None = find_most_recent_file(folder_path)
    if not image_path:
        print_cli_error(f"Could not find a suitable image file in {folder_path}. Exiting.")
        sys.exit(1)
    resolved_image_path: Path = image_path.resolve()
    print_cli_section(f"Processing file: {resolved_image_path.name}")
    logger.info("Located at: %s", Colors.colored(resolved_image_path, Colors.BLUE))
    try:
        with Image.open(resolved_image_path) as img:
            img.verify()
        print_image_dimensions(resolved_image_path)
        return resolved_image_path
    except (
        FileNotFoundError,
        UnidentifiedImageError,
        OSError,
        Exception,
    ) as img_err:
        print_cli_error(f"Cannot open or verify image {resolved_image_path}: {img_err}. Exiting.")
        sys.exit(1)


def handle_metadata(image_path: Path, args: argparse.Namespace) -> MetadataDict:
    """Extract, print, and return image metadata."""
    metadata: MetadataDict = extract_image_metadata(
        image_path,
        debug=args.debug,
    )
    logger.info("  Date: %s", Colors.colored(metadata.get("date", "N/A"), Colors.CYAN))
    logger.info(
        "  Desc: %s",
        Colors.colored(metadata.get("description", "N/A"), Colors.CYAN),
    )
    logger.info("  GPS:  %s", Colors.colored(metadata.get("gps", "N/A"), Colors.CYAN))

    if args.verbose or args.debug:
        exif_data: ExifDict | None = get_exif_data(image_path)
        if exif_data:
            pretty_print_exif(exif_data, verbose=True)
        else:
            logger.warning("\nNo detailed EXIF data could be extracted.")
    return metadata


def prepare_prompt(args: argparse.Namespace, metadata: MetadataDict) -> str:
    """Prepare the prompt for the VLM, using user input or generating from metadata."""
    prompt: str
    if args.prompt:
        prompt = args.prompt
        logger.info("Using user-provided prompt.")
    else:
        logger.info("Generating default prompt based on image metadata.")
        prompt_parts: list[str] = [
            "Provide a factual caption, description, and keywords suitable for cataloguing, or searching for, the image.",
            (
                f"Context: Relates to '{metadata.get('description', '')}'"
                if metadata.get("description") and metadata["description"] != "N/A"
                else ""
            ),
            (
                f"taken around {metadata.get('date', '')}"
                if metadata.get("date") and metadata["date"] != "Unknown date"
                else ""
            ),
            (
                f"near GPS {metadata.get('gps', '')}"
                if metadata.get("gps") and metadata["gps"] != "Unknown location"
                else ""
            ),
            "Focus on visual content. Avoid repeating the context unless it is visible.",
        ]
        prompt = " ".join(filter(None, prompt_parts)).strip()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Using generated prompt based on metadata.")
    logger.info(
        "\n%s\n%s\n%s",
        Colors.colored("--- Using Prompt ---", Colors.CYAN),
        prompt,
        Colors.colored("-" * 40, Colors.BLUE),
    )
    return prompt


def process_models(
    args: argparse.Namespace,
    image_path: Path,
    prompt: str,
) -> list[ModelResult]:
    """Process images with the specified models or scan cache for available models.

    Returns a list of model results with outputs and performance metrics.
    """
    model_identifiers: list[str]
    if args.models:
        model_identifiers = args.models
        logger.info("Processing specified models: %s", ", ".join(model_identifiers))
    else:
        logger.info("Scanning cache for models to process...")
        model_identifiers = get_cached_model_ids()

    results: list[ModelResult] = []
    if not model_identifiers:
        logger.error("No models specified or found in cache.")
        if not args.models:
            logger.error("Ensure models are downloaded and cache is accessible.")
    else:
        logger.info("Processing %d model(s)...", len(model_identifiers))
        separator = Colors.colored(f"\n{'-' * 40}\n", Colors.BLUE)
        for model_id in model_identifiers:
            logger.info(separator)
            is_vlm_verbose: bool = args.verbose or args.debug
            result: ModelResult = process_image_with_model(
                model_identifier=model_id,
                image_path=image_path,
                prompt=prompt,
                max_tokens=args.max_tokens,
                verbose=is_vlm_verbose,
                trust_remote_code=args.trust_remote_code,
                temperature=args.temperature,
                timeout=args.timeout,
            )
            results.append(result)
            model_short_name: str = model_id.split("/")[-1]

            if result.success:
                logger.info(f"[SUCCESS] {model_short_name}")

                if result.output:
                    # Log output with proper formatting
                    logger.info(
                        "%s\n%s",
                        Colors.colored("Output:", Colors.CYAN),
                        Colors.colored(result.output, Colors.CYAN + Colors.BOLD),
                    )
                if args.verbose or args.debug:
                    logger.info("Time taken: %.2f s", result.stats.time)
            else:
                # Format error messages
                logger.error(
                    "[FAIL] %s (Stage: %s)", model_short_name, result.error_stage
                )
                logger.error(
                    "Model %s failed during '%s'.",
                    model_short_name,
                    result.error_stage,
                )
                if args.verbose or args.debug:
                    logger.error("Reason: %s", result.error_message)
    return results


def finalize_execution(
    args: argparse.Namespace,
    results: list[ModelResult],
    library_versions: dict[str, str],
    overall_start_time: float,
    prompt: str,
) -> None:
    """Output summary statistics, generate reports, and display timing information."""
    if results:
        # Print summary statistics
        separator = Colors.colored("=" * 80, Colors.BLUE)
        logger.info("\n%s\n", separator)
        print_model_stats(results)

        try:
            html_output_path: Path = args.output_html.resolve()
            md_output_path: Path = args.output_markdown.resolve()

            # Create parent directories if they don't exist
            html_output_path.parent.mkdir(parents=True, exist_ok=True)
            md_output_path.parent.mkdir(parents=True, exist_ok=True)

            # Generate reports
            generate_html_report(results, html_output_path, library_versions, prompt)
            generate_markdown_report(results, md_output_path, library_versions, prompt)

            # Log paths to generated files
            logger.info("Reports generated:")
            logger.info(
                "HTML:     %s",
                Colors.colored(str(html_output_path), Colors.GREEN),
            )
            logger.info(
                "Markdown: %s",
                Colors.colored(str(md_output_path), Colors.GREEN),
            )
        except Exception:
            logger.exception("Failed to generate reports")
    else:
        logger.warning(
            "\nNo models processed. No performance summary generated."
        )
        logger.info("Skipping report generation as no models were processed.")

    # Print version information
    print_version_info(library_versions)

    # Calculate and log total time
    overall_time: float = time.perf_counter() - overall_start_time
    time_msg = Colors.colored("%s seconds", Colors.GREEN) % f"{overall_time:.2f}"
    logger.info("\nTotal execution time: %s", time_msg)


# --- Main Execution ---
def main(args: argparse.Namespace) -> None:
    """Main function to orchestrate image analysis."""
    overall_start_time: float = time.perf_counter()
    print_cli_header("MLX Vision Language Model Image Analysis")
    library_versions: dict[str, str] = setup_environment(args)
    resolved_image_path: Path = find_and_validate_image(args)
    metadata: MetadataDict = handle_metadata(resolved_image_path, args)
    prompt: str = prepare_prompt(args, metadata)
    results: list[ModelResult] = process_models(
        args,
        resolved_image_path,
        prompt,
    )
    finalize_execution(args, results, library_versions, overall_start_time, prompt)


if __name__ == "__main__":
    # Setup Argument Parser
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Analyze image with MLX VLMs.",
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
        "--models",
        nargs="+",
        type=str,
        default=None,
        help="Specify models by ID/path. Overrides cache scan.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow custom code from Hub models (SECURITY RISK).",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt.",
    )
    parser.add_argument(
        "-m",
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
        help=f"Sampling temperature.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output (INFO logging).",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug logging (DEBUG logging).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"Timeout in seconds for model operations.",
    )

    # Parse arguments
    parsed_args: argparse.Namespace = parser.parse_args()

    # --- Main Execution ---
    try:
        main(parsed_args)
    except Exception as main_err:
        # Log final unhandled exceptions with color
        # Use logger.critical for severe errors causing exit
        logger.critical(
            Colors.colored(
                "An unexpected error occurred during main execution: %s",
                Colors.RED,
            ),
            main_err,
            exc_info=True,
        )
        sys.exit(1)  # Exit with error status
