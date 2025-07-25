#!/usr/bin/env python3
"""Image analysis and caption generation using MLX Vision Language Models."""

from __future__ import annotations

# Standard library imports
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
import types
from dataclasses import asdict, dataclass, fields
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

if TYPE_CHECKING:
    from zoneinfo import ZoneInfo

# Third-party imports
from huggingface_hub import HFCacheInfo, scan_cache_dir
from huggingface_hub import __version__ as hf_version
from huggingface_hub.errors import HFValidationError
from tzlocal import get_localzone

try:
    import mlx.core as mx
except ImportError:
    # Create a temporary logger since the main one isn't configured yet
    _temp_logger = logging.getLogger("mlx-vlm-check")
    _temp_logger.exception("Core dependency missing: mlx. Please install it.")
    sys.exit(1)

try:
    from PIL import ExifTags, Image, UnidentifiedImageError
    from PIL.ExifTags import GPSTAGS, TAGS

    pillow_version: str = Image.__version__ if hasattr(Image, "__version__") else "N/A"
except ImportError:
    # Create a temporary logger since the main one isn't configured yet
    _temp_logger = logging.getLogger("mlx-vlm-check")
    _temp_logger.critical(
        "Error: Pillow not found. Please install it (`pip install Pillow`).",
    )
    pillow_version = "N/A"
    sys.exit(1)

# Local application/library specific imports
try:
    from mlx_vlm.generate import GenerationResult, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load
    from mlx_vlm.version import __version__ as vlm_version
except ImportError:
    # Create a temporary logger since the main one isn't configured yet
    _temp_logger = logging.getLogger("mlx-vlm-check")
    _temp_logger.critical(
        "Error: mlx-vlm not found. Please install it (`pip install mlx-vlm`).",
    )
    sys.exit(1)

# Optional imports for version reporting
try:
    import mlx_lm

    mlx_lm_version: str = str(getattr(mlx_lm, "__version__", "N/A"))
except ImportError:
    mlx_lm_version = "N/A"
except AttributeError:
    mlx_lm_version = "N/A (module found, no version attr)"
try:
    import transformers

    transformers_version: str = transformers.__version__
except ImportError:
    transformers_version = "N/A"


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


# Configure logging - Single logger instance
logger: logging.Logger = logging.getLogger("mlx-vlm-check")


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
        color: str = self.LEVEL_COLORS.get(record.levelno, "")
        msg: str = super().format(record)
        if color:
            msg = Colors.colored(msg, color)
        return msg


# Configure logging to use ColoredFormatter
handler: logging.StreamHandler[Any] = logging.StreamHandler(sys.stderr)
formatter: ColoredFormatter = ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.handlers.clear()
logger.addHandler(handler)

# Constants
MB_CONVERSION: Final[float] = 1024 * 1024
LARGE_NUMBER_THRESHOLD: Final[float] = 1000.0

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
}

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
    },
)


# --- Shared Utility Functions ---
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


def is_numeric_field(field_name: str) -> bool:
    """Check if a field should be treated as numeric (right-aligned)."""
    field_lower = field_name.lower()
    return (
        field_name in NUMERIC_FIELD_PATTERNS
        or any(keyword in field_lower for keyword in ("token", "tps", "memory", "time"))
        or field_lower.endswith("_tokens")
    )


def is_numeric_value(val: float | str | object) -> bool:
    """Check if a value should be formatted as numeric."""
    return isinstance(val, (int, float)) or (
        isinstance(val, str) and val.replace(".", "", 1).isdigit()
    )


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
        status_color: str = Colors.GREEN if ver != "N/A" else Colors.YELLOW
        name_padded: str = name.ljust(max_len)
        logger.info("%s: %s", name_padded, Colors.colored(ver, status_color))
    logger.info(
        "Generated: %s",
        datetime.now(get_localzone()).strftime("%Y-%m-%d %H:%M:%S %Z"),
    )


def status_tag(status: str) -> str:
    """Return a colored status tag for a given status string."""
    s: str = status.upper()
    # Use dictionary lookup for better performance and reduced branching
    status_colors = {
        "SUCCESS": (Colors.BOLD, Colors.GREEN),
        "FAIL": (Colors.BOLD, Colors.RED),
        "WARNING": (Colors.BOLD, Colors.YELLOW),
        "INFO": (Colors.BOLD, Colors.BLUE),
    }
    colors = status_colors.get(s, (Colors.BOLD, Colors.MAGENTA))
    return Colors.colored(s, *colors)


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
class PerformanceResult:
    """Encapsulates a GenerationResult, timing, and performance metrics for a model run."""

    model_name: str
    generation: GenerationResult | None
    success: bool
    error_stage: str | None = None
    error_message: str | None = None
    captured_output_on_fail: str | None = None
    active_mb: float = 0.0
    cached_mb: float = 0.0
    peak_mb: float = 0.0
    time_s: float = 0.0


# --- File Handling ---
# Simplified the `find_most_recent_file` function by using `max` with a generator.
def find_most_recent_file(folder: Path | str) -> Path | None:
    """Return the most recently modified file in a folder, or None."""
    folder_str = str(folder)
    folder_path = Path(folder_str)
    if not folder_path.is_dir():
        logger.error(
            "Provided path is not a directory: %s",
            folder_str,
        )
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
        logger.exception(
            Colors.colored(f"Directory not found: {folder_str}", Colors.YELLOW),
        )
    except PermissionError:
        logger.exception(
            Colors.colored(
                f"Permission denied accessing folder: {folder_str}",
                Colors.YELLOW,
            ),
        )
    except OSError:
        logger.exception(
            Colors.colored(f"OS error scanning folder {folder_str}", Colors.YELLOW),
        )
    logger.debug("No files found in directory: %s", folder_str)
    return None


# Improved error handling in `print_image_dimensions`.
def print_image_dimensions(image_path: Path | str) -> None:
    """Print the dimensions and megapixel count of an image file."""
    img_path_str: str = str(image_path)
    try:
        with Image.open(img_path_str) as img:
            width, height = img.size
            mpx: float = (width * height) / 1_000_000
            logger.info(
                "Image dimensions: %s (%s MPixels)",
                f"{width}x{height}",
                f"{mpx:.1f}",
            )
    except (FileNotFoundError, UnidentifiedImageError):
        logger.exception(
            Colors.colored(f"Error with image file {img_path_str}", Colors.YELLOW),
        )
    except OSError:
        logger.exception(
            Colors.colored(
                f"Unexpected error reading image dimensions for {img_path_str}",
                Colors.YELLOW,
            ),
        )


# --- EXIF & Metadata Handling ---
@functools.lru_cache(maxsize=128)
def get_exif_data(image_path: PathLike) -> ExifDict | None:
    """Extract EXIF data from an image file and return as a dictionary."""
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
                logger.warning(
                    Colors.colored(
                        f"Could not extract GPS IFD: {gps_err}",
                        Colors.YELLOW,
                    ),
                )
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

    # --- Date extraction ---
    date: str = (
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
            logger.warning(
                Colors.colored(f"Could not localize EXIF date: {err}", Colors.YELLOW),
            )
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
                logger.debug(
                    Colors.colored(
                        f"Failed to decode description: {err}",
                        Colors.YELLOW,
                    ),
                )
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
            tag_name = GPSTAGS.get(int(k), str(k)) if isinstance(k, int) else str(k)
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
            logger.debug(
                Colors.colored(
                    "GPS conversion failed: latitude or longitude is None.",
                    Colors.YELLOW,
                ),
            )
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
            lat_ref_str = lat_ref.decode() if isinstance(lat_ref, bytes) else str(lat_ref)
            lon_ref_str = lon_ref.decode() if isinstance(lon_ref, bytes) else str(lon_ref)
            lat_dd, lat_card = dms_to_dd(latitude, lat_ref_str)
            lon_dd, lon_card = dms_to_dd(longitude, lon_ref_str)
            lat_dd = -abs(lat_dd) if lat_card == "S" else abs(lat_dd)
            lon_dd = -abs(lon_dd) if lon_card == "W" else abs(lon_dd)
            return f"{abs(lat_dd):.6f} {lat_card}, {abs(lon_dd):.6f} {lon_card}"
        except (ValueError, AttributeError, TypeError) as err:
            logger.debug(
                Colors.colored(
                    f"Failed to convert GPS DMS to decimal: {err}",
                    Colors.YELLOW,
                ),
            )
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
        tag_str = str(tag)
        if tag_str == "GPSInfo" and isinstance(value, dict):
            continue
        if isinstance(value, dict):
            logger.debug(
                "Skipping dictionary value for EXIF tag '%s' in pretty print.",
                tag_str,
            )
            continue
        value_str = exif_value_to_str(tag_str, value)
        is_important = tag_str in IMPORTANT_EXIF_TAGS
        if show_all or is_important:
            tags.append((tag_str, value_str, is_important))
    return tags


def pretty_print_exif(
    exif: ExifDict,
    *,
    show_all: bool = True,
    title: str = "EXIF Metadata Summary",
) -> None:
    """Print key EXIF data in a formatted table with colors and a title."""
    if not exif:
        logger.info(Colors.colored("No EXIF data available.", Colors.YELLOW))
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
    max_tag_len = max(Colors.visual_len(t[0]) for t in tags_to_print) if tags_to_print else 20
    max_val_len = max(Colors.visual_len(t[1]) for t in tags_to_print) if tags_to_print else 40
    min_width = 10
    max_tag_len = max(max_tag_len, min_width)
    max_val_len = max(max_val_len, min_width + 5)
    header_color = Colors.BLUE
    border_color = Colors.BLUE
    important_color = Colors.YELLOW

    table_width = max_tag_len + max_val_len + 7

    # Print title above the table, visually separated (no leading newline)
    logger.info(
        Colors.colored("=" * table_width, Colors.BOLD, Colors.BLUE),
    )
    # Print the title in a more visually distinct color and with extra spacing
    logger.info(
        Colors.colored(f"{title.center(table_width)}", Colors.BOLD, Colors.MAGENTA),
    )
    # Add a blank line for separation
    logger.info("")
    logger.info(
        Colors.colored("=" * table_width, Colors.BOLD, Colors.BLUE),
    )

    logger.info(
        Colors.colored(
            "╔{}╤{}╗".format("═" * (max_tag_len + 2), "═" * (max_val_len + 2)),
            border_color,
        ),
    )
    tag_header = _pad_text(Colors.colored("Tag", header_color), max_tag_len)
    value_header = _pad_text(Colors.colored("Value", header_color), max_val_len)
    logger.info(
        "%s %s %s %s %s",
        Colors.colored("║", border_color),
        tag_header,
        Colors.colored("│", border_color),
        value_header,
        Colors.colored("║", border_color),
    )
    logger.info(
        Colors.colored(
            "╠{}╪{}╣".format("═" * (max_tag_len + 2), "═" * (max_val_len + 2)),
            border_color,
        ),
    )
    for tag_name, value_display, is_important_tag in tags_to_print:
        tag_display = (
            Colors.colored(tag_name, Colors.BOLD + important_color)
            if is_important_tag
            else tag_name
        )
        padded_tag = _pad_text(tag_display, max_tag_len)
        padded_value = _pad_text(value_display, max_val_len)
        logger.info(
            "%s %s %s %s %s",
            Colors.colored("║", border_color),
            padded_tag,
            Colors.colored("│", border_color),
            padded_value,
            Colors.colored("║", border_color),
        )
    logger.info(
        Colors.colored(
            "╚{}╧{}╝".format("═" * (max_tag_len + 2), "═" * (max_val_len + 2)),
            border_color,
        ),
    )
    logger.info(
        Colors.colored("=" * table_width + "\n", Colors.BOLD, Colors.BLUE),
    )


def print_model_stats(results: list[PerformanceResult]) -> None:
    """Print a visually compact, perfectly aligned table summarizing model GenerationResult fields and text output/diagnostics, with units and formatted numbers, fitting within 100 characters."""
    if not results:
        logger.info(Colors.colored("No model results to display.", Colors.YELLOW))
        return

    # Determine GenerationResult fields (excluding 'text' and 'logprobs')
    gen_fields = []
    for r in results:
        if r.generation is not None:
            gen_fields = [
                f.name for f in fields(r.generation) if f.name not in ("text", "logprobs")
            ]
            break
    if not gen_fields:
        gen_fields = []

    # Abbreviated headers and units for CLI
    field_abbr = FIELD_ABBREVIATIONS

    # Build headers (2 lines: label and unit)
    header_line1 = ["Model"]
    header_line2 = [""]
    for h in gen_fields:
        abbr, unit = field_abbr.get(h, (h.replace("_", " ").title(), ""))
        header_line1.append(abbr)
        header_line2.append(unit)
    # Use a short header for output column
    output_header = "Output"
    header_line1.append(output_header)
    header_line2.append("")
    ncols = len(header_line1)
    while len(header_line2) < ncols:
        header_line2.append("")

    # Set minimal and maximal column widths
    min_col = 6
    max_output_col = max(len(output_header), 18)  # At least as wide as header
    col_widths = [max(min_col, len(header_line1[0]), len(header_line2[0]))]
    # Use list comprehension for better performance
    col_widths.extend([
        max(min_col, len(header_line1[i]), len(header_line2[i]))
        for i in range(1, ncols - 1)
    ])
    col_widths.append(max_output_col)

    # Update col_widths based on data (but never exceed max_output_col for output)
    for r in results:
        col_widths[0] = max(col_widths[0], min(25, len(str(r.model_name))))
        for i, f in enumerate(gen_fields):
            val = getattr(r.generation, f, "-") if r.generation else "-"
            if isinstance(val, (int, float)) or (
                isinstance(val, str) and val.replace(".", "", 1).isdigit()
            ):
                val = fmt_num(val)
            col_widths[i + 1] = max(col_widths[i + 1], min(12, len(str(val))))
    # Calculate total width and shrink if needed
    max_width = 100
    total = sum(col_widths) + 3 * (ncols - 1)
    # Shrink output col first
    if total > max_width:
        excess = total - max_width
        if col_widths[-1] > len(output_header):
            shrink = min(excess, col_widths[-1] - len(output_header))
            col_widths[-1] -= shrink
            total -= shrink
    # Shrink other cols if still too wide
    for i in range(1, ncols - 1):
        if total <= max_width:
            break
        if col_widths[i] > min_col:
            shrink = min(total - max_width, col_widths[i] - min_col)
            col_widths[i] -= shrink
            total -= shrink
    # Final adjustment: pad columns to fill exactly max_width
    total = sum(col_widths) + 3 * (ncols - 1)
    if total < max_width:
        col_widths[-1] += max_width - total

    # Determine column alignment: numeric fields right-aligned, text fields left-aligned
    col_alignments = ["left"]  # Model name is always left-aligned
    for f in gen_fields:
        # Use the shared is_numeric_field function for consistency
        if is_numeric_field(f):
            col_alignments.append("right")
        else:
            col_alignments.append("left")
    col_alignments.append("left")  # Output column is left-aligned

    # Print header (no extra blank lines) - use logger.info() for consistency
    logger.info("=" * max_width)
    # Align headers according to column type
    header_row1 = []
    header_row2 = []
    for i in range(ncols):
        if col_alignments[i] == "right":
            header_row1.append(header_line1[i].rjust(col_widths[i]))
            header_row2.append(header_line2[i].rjust(col_widths[i]))
        else:
            header_row1.append(header_line1[i].ljust(col_widths[i]))
            header_row2.append(header_line2[i].ljust(col_widths[i]))
    logger.info(" | ".join(header_row1))
    logger.info(" | ".join(header_row2))
    logger.info("-+-".join("-" * w for w in col_widths))
    # Print rows
    for r in results:
        row = [str(r.model_name)[: col_widths[0]].ljust(col_widths[0])]
        for i, f in enumerate(gen_fields):
            val = getattr(r.generation, f, "-") if r.generation else "-"
            is_numeric = isinstance(val, (int, float)) or (
                isinstance(val, str) and val.replace(".", "", 1).isdigit()
            )
            if is_numeric:
                val = fmt_num(val)
            sval = str(val)[: col_widths[i + 1]]
            # Align based on column type
            if col_alignments[i + 1] == "right":
                row.append(sval.rjust(col_widths[i + 1]))
            else:
                row.append(sval.ljust(col_widths[i + 1]))
        if r.success and r.generation:
            out_val = str(getattr(r.generation, "text", ""))
        else:
            out_val = r.error_message or r.captured_output_on_fail or "-"
        out_val = out_val.replace("\n", " ").replace("\r", " ")
        if len(out_val) > col_widths[-1]:
            out_val = out_val[: col_widths[-1] - 3] + "..."
        row.append(out_val.ljust(col_widths[-1]))
        logger.info(" | ".join(row))
    logger.info("=" * max_width)


# --- HTML Report Generation ---
def generate_html_report(
    results: list[PerformanceResult],
    filename: Path,
    versions: dict[str, str],
    prompt: str,
) -> None:
    """Generate an HTML file with GenerationResult fields as columns and text/diagnostics as last column, with units and formatted numbers."""
    if not results:
        logger.warning(
            Colors.colored("No results to generate HTML report.", Colors.YELLOW),
        )
        return

    # Determine GenerationResult fields (excluding 'text')
    gen_fields = []
    for r in results:
        if r.generation is not None:
            gen_fields = [f.name for f in fields(r.generation) if f.name != "text"]
            break
    if not gen_fields:
        gen_fields = []

    # Use the shared FIELD_UNITS constant
    local_tz = get_localzone()
    html_start = (
        """
<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>Model Performance Results</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f8f9fa; color: #212529; line-height: 1.6; }
        table { border-collapse: collapse; width: 95%; margin: 30px auto; background-color: #fff; }
        th, td { border: 1px solid #dee2e6; padding: 8px 12px; vertical-align: top; }
        th { background-color: #e9ecef; font-weight: 600; color: #495057; }
        tr:nth-child(even):not(.failed-row) { background-color: #f8f9fa; }
        tr.failed-row { background-color: #f8d7da !important; color: #721c24; }
        .model-name { font-family: 'Courier New', Courier, monospace; font-weight: 500; text-align: left; }
        .error-message { font-weight: bold; color: #721c24; }
        .numeric { text-align: right; }
        .text { text-align: left; }
    </style>
</head>
<body>
    <h1>Model Performance Summary</h1>
    <div class=\"prompt-block\"><strong>Prompt used:</strong><br>"""
        + html.escape(prompt).replace("\n", "<br>")
        + """</div>
    <table>
        <caption>Performance metrics and output/errors for Vision Language Model processing. Generated on """
        + datetime.now(local_tz).strftime("%Y-%m-%d %H:%M:%S %Z")
        + """. Failures shown but excluded from averages.</caption>
        <thead>
            <tr>
                <th class="text">Model</th>
"""
    )
    for f in gen_fields:
        label = f.replace("_", " ").title()
        if f in FIELD_UNITS:
            label += f" {FIELD_UNITS[f]}"
        # Determine alignment class based on field type
        # Include patterns for token counts, rates, memory, and timing fields
        alignment_class = (
            "numeric"
            if (
                f
                in (
                    "tokens",
                    "prompt_tokens",
                    "generation_tokens",
                    "prompt_tps",
                    "generation_tps",
                    "peak_memory",
                    "time",
                    "duration",
                )
                or "token" in f.lower()
                or "tps" in f.lower()
                or "memory" in f.lower()
                or "time" in f.lower()
                or f.lower().endswith("_tokens")
            )
            else "text"
        )
        html_start += f'<th class="{alignment_class}">{html.escape(label)}</th>\n'
    html_start += '<th class="text">Output / Diagnostics</th>\n</tr>\n</thead>\n<tbody>\n'

    html_rows: str = ""
    for r in results:
        row_class = ' class="failed-row"' if not r.success else ""
        html_rows += f'<tr{row_class}><td class="model-name">{html.escape(str(r.model_name))}</td>'
        for f in gen_fields:
            val = getattr(r.generation, f, "-") if r.generation else "-"
            is_numeric = isinstance(val, (int, float)) or (
                isinstance(val, str) and val.replace(".", "", 1).isdigit()
            )
            if is_numeric:
                val = fmt_num(val)
            # Use the shared is_numeric_field function for consistency
            alignment_class = "numeric" if is_numeric_field(f) else "text"
            html_rows += f'<td class="{alignment_class}">{html.escape(str(val))}</td>'
        if r.success and r.generation:
            out_val = str(getattr(r.generation, "text", ""))
        else:
            out_val = r.error_message or r.captured_output_on_fail or "-"
        html_rows += f'<td class="text">{html.escape(out_val)}</td></tr>\n'

    html_footer: str = "<footer>\n<h2>Library Versions</h2>\n<ul>\n"
    for name, ver in sorted(versions.items()):
        html_footer += (
            f"<li><code>{html.escape(name)}</code>: <code>{html.escape(ver)}</code></li>\n"
        )
    html_footer += (
        "</ul>\n<p>Report generated on: "
        + datetime.now(local_tz).strftime("%Y-%m-%d %H:%M:%S %Z")
        + "</p>\n</footer>"
    )

    html_end = f"""
        </tbody>
    </table>
    <!-- End of Table -->
    {html_footer}
</body>
</html>
"""
    html_content: str = html_start + html_rows + html_end

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


def _format_markdown_table_row(
    model_disp_name: str,
    stats: list[str],
    output_md: str,
) -> str:
    return (
        f"| {model_disp_name} | {stats[0]} | {stats[1]} | {stats[2]} | {stats[3]} | {output_md} |"
    )


def generate_markdown_report(
    results: list[PerformanceResult],
    filename: Path,
    versions: dict[str, str],
    prompt: str,
) -> None:
    """Generate a Markdown file with GenerationResult fields as columns and text/diagnostics as last column, with units and formatted numbers."""
    if not results:
        logger.warning(
            Colors.colored("No results to generate Markdown report.", Colors.YELLOW),
        )
        return

    # Determine GenerationResult fields (excluding 'text')
    gen_fields = []
    for r in results:
        if r.generation is not None:
            gen_fields = [f.name for f in fields(r.generation) if f.name != "text"]
            break
    if not gen_fields:
        gen_fields = []

    # Use the shared FIELD_UNITS constant
    md: list[str] = []
    md.append("# Model Performance Results\n")
    md.append(f"_Generated on {datetime.now(get_localzone()).strftime('%Y-%m-%d %H:%M:%S %Z')}_\n")
    md.append("")
    md.append("> **Prompt used:**\n>\n> " + prompt.replace("\n", "\n> ") + "\n")
    md.append("")
    # Build header row with units
    header_row = ["Model"]
    alignment_row = [":-"]  # Model column is left-aligned
    for h in gen_fields:
        label = h.replace("_", " ").title()
        if h in FIELD_UNITS:
            label += f" {FIELD_UNITS[h]}"
        header_row.append(label)
        # Use the shared is_numeric_field function for consistency
        if is_numeric_field(h):
            alignment_row.append("-:")  # Right-aligned for numeric fields
        else:
            alignment_row.append(":-")  # Left-aligned for text fields
    header_row.append("Output / Diagnostics")
    alignment_row.append(":-")  # Output column is left-aligned

    md.append("| " + " | ".join(header_row) + " |")
    md.append("|" + "|".join(alignment_row) + "|")

    for r in results:
        row = [f"`{r.model_name}`"]
        for f in gen_fields:
            val = getattr(r.generation, f, "-") if r.generation else "-"
            # Format numbers
            if isinstance(val, (int, float)) or (
                isinstance(val, str) and val.replace(".", "", 1).isdigit()
            ):
                val = fmt_num(val)
            row.append(str(val))
        if r.success and r.generation:
            out_val = str(getattr(r.generation, "text", ""))
        else:
            out_val = r.error_message or r.captured_output_on_fail or "-"
        row.append(out_val)
        md.append("| " + " | ".join(row) + " |")

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
                    line for line in result.stdout.split("\n") if "Chipset Model:" in line
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
    img_path: Path = Path(str(image_path))
    if not img_path.exists():
        msg: str = f"Image not found: {img_path}"
        raise FileNotFoundError(msg)
    if not img_path.is_file():
        msg: str = f"Not a file: {img_path}"
        raise ValueError(msg)
    if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        msg: str = f"Unsupported image format: {img_path.suffix}"
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
    else:
        return


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
    """Load model, format prompt and run generation.

    Raise exceptions on failure.
    """
    model: object
    tokenizer: object

    model, tokenizer = load(
        path_or_hf_repo=params.model_path,
        trust_remote_code=params.trust_remote_code,
    )
    config = model.config

    formatted_prompt = apply_chat_template(
        processor=tokenizer,
        config=config,
        prompt=params.prompt,
        num_images=1,
    )
    # Handle list return from apply_chat_template
    if isinstance(formatted_prompt, list):
        formatted_prompt = "\n".join(str(m) for m in formatted_prompt)

    output: GenerationResult = generate(
        model=model,
        processor=tokenizer,
        prompt=formatted_prompt,
        image=str(image_path),
        verbose=verbose,
        temperature=params.temperature,
        trust_remote_code=params.trust_remote_code,
        max_tokens=params.max_tokens,
    )
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
        Colors.colored(
            "Processing '%s' with model: %s",
            Colors.BOLD,
            Colors.MAGENTA,
        ),
        str(getattr(params.image_path, "name", params.image_path)),
        params.model_identifier,
    )
    model: object | None = None
    tokenizer: object | None = None
    arch, gpu_info = get_system_info()
    start_time: float = 0.0
    initial_mem: float = 0.0
    initial_cache: float = 0.0
    try:
        validate_temperature(temp=params.temperature)
        validate_image_accessible(image_path=params.image_path)
        logger.debug("System: %s, GPU: %s", arch, gpu_info)
        initial_mem = mx.get_active_memory() / MB_CONVERSION  # type: ignore[attr-defined]
        initial_cache = mx.get_cache_memory() / MB_CONVERSION  # type: ignore[attr-defined]
        start_time = time.perf_counter()
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
        end_time: float = time.perf_counter()
        final_active_mem: float = mx.get_active_memory() / MB_CONVERSION  # type: ignore[attr-defined]
        final_cache_mem: float = mx.get_cache_memory() / MB_CONVERSION  # type: ignore[attr-defined]
        peak_mem: float = mx.get_peak_memory() / MB_CONVERSION  # type: ignore[attr-defined]
        return PerformanceResult(
            model_name=params.model_identifier,
            generation=output,
            success=True,
            active_mb=final_active_mem - initial_mem,
            cached_mb=final_cache_mem - initial_cache,
            peak_mb=peak_mem,
            time_s=end_time - start_time,
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
        mx.clear_cache()  # type: ignore[attr-defined]
        mx.reset_peak_memory()  # type: ignore[attr-defined]
        logger.debug("Cleaned up resources for model %s", params.model_identifier)


# --- Main Execution Helper Functions ---


def print_cli_header(title: str) -> None:
    """Print a formatted CLI header with the given title."""
    logger.info(Colors.colored("=" * 100, Colors.BOLD, Colors.BLUE))
    logger.info(Colors.colored(title.center(100), Colors.BOLD, Colors.MAGENTA))
    logger.info(Colors.colored("=" * 100, Colors.BOLD, Colors.BLUE))


def print_cli_section(title: str) -> None:
    """Print a formatted CLI section header."""
    logger.info(Colors.colored("--- %s ---", Colors.BOLD, Colors.MAGENTA), title)


def print_cli_error(msg: str) -> None:
    """Print a formatted CLI error message."""
    logger.error(Colors.colored("Error: %s", Colors.BOLD, Colors.CYAN), msg)


def setup_environment(args: argparse.Namespace) -> dict[str, str]:
    """Configure logging, collect versions, print warnings."""
    # Set DEBUG if verbose, else WARNING
    log_level: int = logging.DEBUG if args.verbose else logging.INFO
    # Remove all handlers and add only one
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    formatter = ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s")
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
        logger.warning(
            "--- SECURITY WARNING ---",
        )
        logger.warning(
            "`--trust-remote-code` is enabled.",
        )
        logger.warning(
            "------------------------",
        )

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
    print_cli_section(f"Processing file: {resolved_image_path.name}")
    logger.info("Image: %s", str(resolved_image_path))
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
    metadata: MetadataDict = extract_image_metadata(
        image_path,
    )
    logger.info("  Date: %s", Colors.colored(metadata.get("date", "N/A"), Colors.CYAN))
    logger.info(
        "  Desc: %s",
        Colors.colored(metadata.get("description", "N/A"), Colors.CYAN),
    )
    logger.info("  GPS:  %s", Colors.colored(metadata.get("gps", "N/A"), Colors.CYAN))

    if args.verbose:
        exif_data: ExifDict | None = get_exif_data(image_path)
        if exif_data:
            pretty_print_exif(exif_data, show_all=True)
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
            (
                "Provide a factual caption, description, and keywords suitable for "
                "cataloguing, or searching for, the image."
            ),
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
            (
                ". Focus on visual content. Avoid repeating the context unless it is "
                "visible. Do not speculate."
            ),
        ]
        prompt = " ".join(filter(None, prompt_parts)).strip()
        logger.debug("Using generated prompt based on metadata.")
    logger.info(
        "%s\n%s\n%s",
        "--- Using Prompt ---",
        prompt,
        "-" * 40,
    )
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


def process_models(
    args: argparse.Namespace,
    image_path: Path,
    prompt: str,
) -> list[PerformanceResult]:
    """Process images with the specified models or scan cache for available models.

    Returns a list of performance results with outputs and performance metrics.
    """
    model_identifiers: list[str]
    if args.models:
        model_identifiers = args.models
        logger.info("Processing specified models: %s", ", ".join(model_identifiers))
    else:
        logger.info("Scanning cache for models to process...")
        model_identifiers = get_cached_model_ids()

    results: list[PerformanceResult] = []
    if not model_identifiers:
        logger.error("No models specified or found in cache.")
        if not args.models:
            logger.error("Ensure models are downloaded and cache is accessible.")
    else:
        logger.info("Processing %d model(s)...", len(model_identifiers))
        separator = "\n" + ("-" * 40) + "\n"
        for model_id in model_identifiers:
            logger.info(separator)
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
            model_short_name: str = model_id.split("/")[-1]
            if result.success:
                logger.info(
                    Colors.colored(
                        f"[SUCCESS] {model_short_name}",
                        Colors.BOLD,
                        Colors.GREEN,
                    ),
                )
                if result.generation:
                    logger.info(
                        Colors.colored("Output fields:", Colors.BOLD, Colors.GREEN),
                    )
                    for k, v in asdict(result.generation).items():
                        logger.info("  %s: %s", k, v)
                if args.verbose:
                    logger.info(
                        Colors.colored(
                            "Statistics:\nTime taken: %.2fs\nActive Δ: %.1f MB\nCache Δ: %.1f MB\nPeak Mem: %.1f MB\n",
                            Colors.BOLD,
                            Colors.CYAN,
                        ),
                        result.time_s,
                        result.active_mb,
                        result.cached_mb,
                        result.peak_mb,
                    )
            else:
                logger.error(
                    Colors.colored(
                        f"[FAIL] {model_short_name} (Stage: {result.error_stage})",
                        Colors.BOLD,
                        Colors.RED,
                    ),
                )
                logger.error(
                    Colors.colored("Reason: %s", Colors.BOLD, Colors.RED),
                    result.error_message,
                )
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
        logger.info("=" * 100)
        print_model_stats(results)
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
            logger.info("Reports generated:")
            logger.info("HTML:     %s", str(html_output_path))
            logger.info("Markdown: %s", str(md_output_path))
        except (OSError, ValueError):
            logger.exception("Failed to generate reports.")
    else:
        logger.warning("No models processed. No performance summary generated.")
        logger.info("Skipping report generation as no models were processed.")
    print_version_info(library_versions)
    overall_time: float = time.perf_counter() - overall_start_time
    time_msg = "{} seconds".format(f"{overall_time:.2f}")
    logger.info("Total execution time: %s", time_msg)


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


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="MLX VLM Model Checker")
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
        logger.info(
            Colors.colored(
                "--- Command Line Parameters ---",
                Colors.BOLD,
                Colors.BLUE,
            ),
        )
        for arg_name, arg_value in sorted(vars(args).items()):
            logger.info(
                "%s: %s",
                Colors.colored(arg_name, Colors.BOLD, Colors.CYAN),
                Colors.colored(str(arg_value), Colors.GREEN),
            )
        logger.info(
            Colors.colored("--- End Parameters ---", Colors.BOLD, Colors.BLUE),
        )
    main(args)
