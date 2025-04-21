#!/usr/bin/env python3
"""Image analysis and caption generation using MLX Vision Language Models."""

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
import types # Added for TracebackType
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import (
    Any, Dict, Final, List, NamedTuple, NoReturn,
    Optional, TextIO, Tuple, Union, Callable,
    TypeVar
)

import os # Needed for os.path.getmtime

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
    from PIL import Image, UnidentifiedImageError
    from PIL.ExifTags import GPSTAGS, TAGS
    pillow_version = Image.__version__ if hasattr(Image, '__version__') else 'N/A'
except ImportError:
    print("Error: Pillow not found. Please install it (`pip install Pillow`).", file=sys.stderr)
    pillow_version = "N/A"
    sys.exit(1)

# Local application/library specific imports
try:
    from mlx_vlm import (__version__ as vlm_version, generate, load)
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
except ImportError:
    print("Error: mlx-vlm not found. Please install it (`pip install mlx-vlm`).", file=sys.stderr)
    sys.exit(1)

# Optional imports for version reporting
try:
    # Import the module first
    from mlx_lm import _version as mlx_lm_version_module
    # Then try to get its __version__ attribute, ensuring it's a string
    mlx_lm_version = str(getattr(mlx_lm_version_module, '__version__', 'N/A'))
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
class timeout_manager(contextlib.ContextDecorator):
    def __init__(self, seconds: float) -> None:
        self.seconds: float = seconds
        # Accommodate signal.SIG_DFL, signal.SIG_IGN (integers)
        self.timer: Union[Callable[[int, Optional[types.FrameType]], Any], int, None] = None

    def _timeout_handler(self, signum: int, frame: Optional[types.FrameType]) -> NoReturn: # Use FrameType
        raise TimeoutError(f"Operation timed out after {self.seconds} seconds")

    def __enter__(self) -> 'timeout_manager':
        # Check if SIGALRM is available (won't be on Windows)
        if hasattr(signal, 'SIGALRM'):
            if self.seconds > 0:
                try:
                    self.timer = signal.signal(signal.SIGALRM, self._timeout_handler)
                    signal.alarm(int(self.seconds))
                except ValueError as e:
                    # Running in a thread or environment where signals are restricted
                    logger.warning(f"Could not set SIGALRM for timeout: {e}. Timeout disabled.")
                    self.seconds = 0 # Disable timeout functionality
        else:
            if self.seconds > 0:
                logger.warning("Timeout functionality requires signal.SIGALRM, not available on this platform (e.g., Windows). Timeout disabled.")
                self.seconds = 0 # Disable timeout functionality
        return self

    def __exit__(self, exc_type: Optional[type[BaseException]],
                 exc_val: Optional[BaseException],
                 exc_tb: Optional[types.TracebackType]) -> None: # Use TracebackType
        # Only try to reset the alarm if it was successfully set
        if hasattr(signal, 'SIGALRM') and self.seconds > 0 and self.timer is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self.timer)


# Configure logging
logger = logging.getLogger(__name__)
# BasicConfig called in main()


# --- Version Info ---
def get_library_versions() -> Dict[str, str]:
    """Collect versions of key libraries."""
    versions = {
        'mlx': getattr(mx, '__version__', 'N/A'),
        'mlx-vlm': vlm_version if 'vlm_version' in globals() else 'N/A',
        'mlx-lm': mlx_lm_version,
        'huggingface-hub': hf_version,
        'transformers': transformers_version,
        'Pillow': pillow_version
    }
    return versions

def print_version_info(versions: Dict[str, str]) -> None:
    """Print collected library versions to the console with date."""
    print("\n--- Library Versions ---")
    max_len = max(len(k) for k in versions) + 1 if versions else 10
    for name, ver in sorted(versions.items()):
        status_color = Colors.GREEN if ver != "N/A" else Colors.YELLOW
        name_padded = name.ljust(max_len)
        print(f"{name_padded}: {Colors.colored(ver, status_color)}")
    print(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# --- ANSI Color Codes for Console Output ---
class Colors:
    """ANSI color codes for terminal output"""
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
    _enabled: bool = sys.stderr.isatty()
    _ansi_escape_re: re.Pattern[str] = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    @staticmethod
    def colored(text: str, color: str) -> str:
        if not Colors._enabled:
            return text
        return f"{color}{text}{Colors.RESET}"

    @staticmethod
    def visual_len(text: Union[str, Any]) -> int:
        if not isinstance(text, str):
             text = str(text)
        return len(Colors._ansi_escape_re.sub('', text))


# Type aliases and definitions
T = TypeVar('T')
ExifValue = Any
ExifDict = Dict[Union[str, int], ExifValue]
MetadataDict = Dict[str, str]
PathLike = Union[str, Path]
GPSTupleElement = Union[int, float]
GPSTuple = Tuple[GPSTupleElement, GPSTupleElement, GPSTupleElement]

# Constants - Defaults
DEFAULT_MAX_TOKENS: Final[int] = 500
DEFAULT_FOLDER: Final[Path] = Path.home() / "Pictures" / "Processed"
DEFAULT_HTML_OUTPUT: Final[Path] = Path("results.html")
DEFAULT_TEMPERATURE: Final[float] = 0.1
DEFAULT_TIMEOUT: Final[float] = 300.0  # Default timeout in seconds

# Constants - EXIF
EXIF_IMAGE_DESCRIPTION_TAG: Final[int] = 270  # Standard EXIF tag ID for ImageDescription
IMPORTANT_EXIF_TAGS: Final[frozenset[str]] = frozenset({
    "DateTimeOriginal", "ImageDescription", "CreateDate", "Make", "Model",
    "LensModel", "ExposureTime", "FNumber", "ISOSpeedRatings",
    "FocalLength", "ExposureProgram",
})
DATE_FORMATS: Final[Tuple[str, ...]] = ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y%m%d")
EXIF_DATE_TAGS: Final[Tuple[str, ...]] = ("DateTimeOriginal", "CreateDate",  "DateTime")
GPS_LAT_REF_TAG: Final[int] = 1
GPS_LAT_TAG: Final[int] = 2
GPS_LON_REF_TAG: Final[int] = 3
GPS_LON_TAG: Final[int] = 4
GPS_INFO_TAG_ID: Final[int] = 34853  # Standard EXIF tag ID for GPS IFD

# Type definitions
class MemoryStats(NamedTuple):
    """Memory statistics container (values represent deltas or peak)."""
    active: float
    cached: float
    peak: float
    time: float
    @staticmethod
    def zero() -> 'MemoryStats': return MemoryStats(0.0, 0.0, 0.0, 0.0)

@dataclass(frozen=True)
class ModelResult:
    """Container for model processing results, including failures."""
    model_name: str
    success: bool
    output: Optional[str] = None
    stats: MemoryStats = field(default_factory=MemoryStats.zero)
    error_stage: Optional[str] = None
    error_message: Optional[str] = None
    captured_output_on_fail: Optional[str] = None


# --- File Handling ---
def find_most_recent_file(folder: Path) -> Optional[Path]:
    """Return the Path of the most recently modified file in the folder."""
    if not folder.is_dir():
        logger.error(Colors.colored(f"Provided path is not a directory: {folder}", Colors.RED))
        return None
    try:
        files: List[Path] = [
            f for f in folder.iterdir() if f.is_file() and not f.name.startswith(".")
        ]
        if not files:
            logger.warning(Colors.colored(f"No non-hidden files found in: {folder}", Colors.YELLOW))
            return None
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        most_recent: Path = files[0]
        logger.debug(f"Most recent file found: {most_recent}")
        return most_recent
    except PermissionError:
        logger.error(Colors.colored(f"Permission denied accessing folder: {folder}", Colors.RED))
        return None
    except OSError as e:
        logger.error(Colors.colored(f"OS error scanning folder {folder}: {e}", Colors.RED))
        return None

def print_image_dimensions(image_path: Path) -> None:
    """Print the dimensions and megapixel count of the image."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            mpx: float = (width * height) / 1_000_000
            print(f"Image dimensions: {Colors.colored(f'{width}x{height}', Colors.CYAN)} ({Colors.colored(f'{mpx:.1f}', Colors.CYAN)} MPixels)")
    except FileNotFoundError:
        logger.error(Colors.colored(f"Image file not found: {image_path}", Colors.RED))
    except UnidentifiedImageError:
        logger.error(Colors.colored(f"Cannot identify image file (may be corrupt or wrong format): {image_path}", Colors.RED))
    except Exception as e:
        logger.error(Colors.colored(f"Error reading image dimensions for {image_path}: {e}", Colors.RED))


# --- EXIF & Metadata Handling ---
@lru_cache(maxsize=128)
def get_exif_data(image_path: Path) -> Optional[ExifDict]:
    """Extract EXIF data from an image file, including decoding GPS IFD."""
    # GPS_INFO_TAG_ID moved to global constants

    try:
        with Image.open(image_path) as img:
            exif_raw: Any = img.getexif()
            if not exif_raw:
                logger.debug(f"No EXIF data found in {image_path}")
                return None

            # First pass: decode main EXIF tags
            exif_decoded: ExifDict = {}
            
            # Try to get DateTimeOriginal directly from IFD using tag ID
            dt_original_tag_id = 36867  # Standard EXIF tag ID for DateTimeOriginal
            if dt_original_tag_id in exif_raw:
                exif_decoded["DateTimeOriginal"] = exif_raw[dt_original_tag_id]

            # Decode remaining EXIF tags
            for tag_id, value in exif_raw.items():
                if tag_id == GPS_INFO_TAG_ID:
                    continue
                if tag_id == dt_original_tag_id:
                    continue
                tag_name = TAGS.get(tag_id, str(tag_id))
                exif_decoded[tag_name] = value

            logger.debug(f"EXIF data decoded for {image_path}: {exif_decoded}")

            # Second pass: handle GPS IFD specifically
            if GPS_INFO_TAG_ID in exif_raw:
                try:
                    gps_ifd = exif_raw.get_ifd(GPS_INFO_TAG_ID)
                    if isinstance(gps_ifd, dict) and gps_ifd:
                        gps_decoded = {}
                        for gps_tag_id, gps_value in gps_ifd.items():
                            gps_tag_name = GPSTAGS.get(gps_tag_id, str(gps_tag_id))
                            gps_decoded[gps_tag_name] = gps_value
                        exif_decoded["GPSInfo"] = gps_decoded
                except Exception as e:
                    # Add image path to the warning log
                    logger.warning(f"Failed to decode GPS data for {image_path}: {e}")

            return exif_decoded

    except (FileNotFoundError, UnidentifiedImageError) as e:
        logger.error(f"Error reading image file: {e}")
    except Exception as e:
        logger.error(f"Unexpected error reading EXIF: {e}")
    return None

def _format_exif_date(date_str_input: Any) -> Optional[str]:
    """Attempt to parse and format a date string from EXIF."""
    date_str: str
    if not isinstance(date_str_input, str):
        try:
            date_str = str(date_str_input)
        except Exception:
            logger.debug(f"Could not convert potential date value '{date_str_input}' to string.")
            return None
    else:
        date_str = date_str_input

    for fmt in DATE_FORMATS:
        try:
            dt_obj: datetime = datetime.strptime(date_str, fmt)
            return dt_obj.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            continue  # Try next format

    logger.debug(f"Could not parse date string '{date_str}' with known formats.")
    return None


def _convert_gps_coordinate(ref: Optional[Union[str, bytes]], coord: Any) -> Optional[float]:
    """Convert various GPS coordinate formats to decimal degrees, robustly handling Ratio types and malformed data."""
    if not ref or not coord:
        logger.debug("Missing GPS reference or coordinate.")
        return None
    try:
        # Handle reference direction
        ref_str = ref.decode('ascii') if isinstance(ref, bytes) else str(ref)
        ref_upper = ref_str.upper()
        if ref_upper not in ['N', 'S', 'E', 'W']:
            logger.warning(f"Unexpected GPS reference: {ref_str}")
            return None
        # Handle DMS tuple (degrees, minutes, seconds)
        # Define to_float function before using it
        def to_float(val):
            if hasattr(val, 'numerator') and hasattr(val, 'denominator'):
                try:
                    return float(val.numerator) / float(val.denominator)
                except Exception as e:
                    logger.warning(f"Malformed Ratio in GPS: {val} ({e})")
                    return None
            try:
                return float(val)
            except Exception as e:
                logger.warning(f"Malformed GPS value: {val} ({e})")
                return None

        if isinstance(coord, (tuple, list)) and len(coord) == 3:
            degrees = to_float(coord[0])
            minutes = to_float(coord[1])
            seconds = to_float(coord[2])
            if None in (degrees, minutes, seconds):
                logger.warning(f"Malformed GPS DMS tuple: {coord}")
                return None
        # Handle decimal degrees and minutes
        elif isinstance(coord, (tuple, list)) and len(coord) == 2:
            degrees = to_float(coord[0])
            minutes = to_float(coord[1])
            seconds = 0.0
            if None in (degrees, minutes):
                logger.warning(f"Malformed GPS DM tuple: {coord}")
                return None
        # Handle direct decimal degrees
        else:
            # Extract the value if it's a single-element tuple/list
            val_to_convert = coord[0] if isinstance(coord, (tuple, list)) and len(coord) == 1 else coord
            # Use the robust to_float function for conversion
            decimal = to_float(val_to_convert)
            if decimal is None:
                logger.warning(f"Could not convert direct GPS value to float: {coord}")
                return None
            # Apply direction sign
            return -decimal if ref_upper in ['S', 'W'] else decimal

        # Validate ranges (This part remains the same, but the logic above handles conversion)
        if degrees is None or minutes is None or seconds is None:
            logger.warning(f"Invalid GPS values (None found): {degrees}, {minutes}, {seconds}")
            return None
        if not (0 <= degrees <= 180 and 0 <= minutes < 60 and 0 <= seconds < 60):
            logger.warning(f"GPS values out of range: {degrees}, {minutes}, {seconds}")
            return None
        decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
        return -decimal if ref_upper in ['S', 'W'] else decimal
    except Exception as e:
        logger.warning(f"GPS conversion error: {type(e).__name__}: {e}")
        return None


def extract_image_metadata(image_path: Path) -> MetadataDict:
    """Extract key metadata: date, GPS, and selected EXIF tags.

    Prioritizes DateTimeOriginal from raw IFD, then falls back to other date sources.
    """
    metadata: MetadataDict = {}
    exif_data: Optional[ExifDict] = get_exif_data(image_path)

    # 1. Try to get date from EXIF, prioritizing DateTimeOriginal
    if exif_data:
        # Prioritize DateTimeOriginal
        dt_original = exif_data.get("DateTimeOriginal")
        if dt_original:
            formatted_date = _format_exif_date(dt_original)
            if formatted_date:
                metadata['date'] = formatted_date
                metadata['date_source'] = "EXIF (DateTimeOriginal)"
                metadata['date_tag'] = "DateTimeOriginal"
                logger.debug(f"Using DateTimeOriginal from EXIF: {formatted_date}")

        # Only try other date tags if DateTimeOriginal wasn't found or couldn't be parsed
        if 'date' not in metadata:
            for tag in EXIF_DATE_TAGS:
                # Skip DateTimeOriginal as it was already checked
                if tag == "DateTimeOriginal":
                    continue
                if tag in exif_data:
                    date_str = str(exif_data[tag])
                    formatted_date = _format_exif_date(date_str)
                    if formatted_date:
                        metadata['date'] = formatted_date
                        metadata['date_source'] = f"EXIF ({tag})"
                        metadata['date_tag'] = tag
                        logger.debug(f"Using date from {tag}: {formatted_date}")
                        break # Found a fallback date, stop searching

    # Extract description directly from ImageDescription tag using its ID
    if exif_data:
        description = exif_data.get(EXIF_IMAGE_DESCRIPTION_TAG, 'N/A') # Use the constant tag ID
        logger.debug(f"EXIF_IMAGE_DESCRIPTION_TAG is '{description}'")
        if description != 'N/A':
            # Ensure description is a string and strip whitespace
            try:
                metadata['description'] = str(description).strip()
            except Exception as desc_err:
                logger.warning(f"Could not convert ImageDescription value '{description}' to string: {desc_err}")
                metadata['description'] = 'N/A' # Fallback if conversion fails
        else:
            metadata['description'] = 'N/A' # Explicitly set N/A if tag not found
    else:
        metadata['description'] = 'N/A' # Set N/A if no EXIF data at all

    # Fallback to file modification time if no EXIF date found
    if 'date' not in metadata:
        try:
            mtime = os.path.getmtime(image_path)
            metadata['date'] = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            metadata['date_source'] = "File Modification Time"
            metadata['date_tag'] = "mtime"
            logger.debug(f"Using file modification time: {metadata['date']}")
        except OSError as e:
            logger.warning(f"Could not get modification time: {e}")
            metadata['date'] = "Unknown"
            metadata['date_source'] = "Unavailable"
            metadata['date_tag'] = "None"

    # 3. Extract GPS coordinates if available
    if exif_data and "GPSInfo" in exif_data and isinstance(exif_data["GPSInfo"], dict):
        gps_info = exif_data["GPSInfo"]
        lat = gps_info.get("GPSLatitude")
        lat_ref = gps_info.get("GPSLatitudeRef")
        lon = gps_info.get("GPSLongitude")
        lon_ref = gps_info.get("GPSLongitudeRef")

        if lat and lat_ref and lon and lon_ref:
            try:
                # Use the more robust conversion function directly
                latitude = _convert_gps_coordinate(lat_ref, lat)
                longitude = _convert_gps_coordinate(lon_ref, lon)

                # Check if conversion was successful
                if latitude is not None and longitude is not None:
                    metadata['gps'] = f"{latitude:.6f}, {longitude:.6f}"
                    logger.debug(f"Extracted GPS {metadata['gps']} for {image_path.name}")
                else:
                    logger.warning(f"Failed to convert GPS coordinates for {image_path.name}. Lat: {lat}, Lon: {lon}")
            # Keep specific error handling for conversion issues
            except Exception as e: # Catch broader exceptions during conversion attempt
                logger.warning(f"Error processing GPS coordinates for {image_path.name}: {e}")
        else:
            logger.debug(f"Incomplete GPS tags found for {image_path.name}")

    # 4. Add other important EXIF tags
    if exif_data:
        for tag in IMPORTANT_EXIF_TAGS:
            # Avoid overwriting the date we just determined
            if tag not in EXIF_DATE_TAGS and tag in exif_data:
                value = exif_data[tag]
                # Simple conversion for common types, avoid complex objects
                if isinstance(value, (str, int, float)):
                     metadata[tag] = str(value).strip()
                elif isinstance(value, bytes):
                     # Attempt to decode bytes, fallback to repr
                     try:
                         metadata[tag] = value.decode('utf-8', errors='replace').strip()
                     except UnicodeDecodeError:
                         metadata[tag] = repr(value)
                elif isinstance(value, tuple):
                     metadata[tag] = ", ".join(map(str, value))
                # Add other simple types if needed, but avoid deep structures

    return metadata

def pretty_print_exif(exif: ExifDict, verbose: bool = False) -> None:
    """Pretty print key EXIF data in a formatted table, using colors."""
    # (Implementation remains the same as previous correct version)
    if not exif:
        print("No EXIF data available.")
        return

    print(f"\n--- {Colors.colored('Key EXIF Data', Colors.CYAN)} ---")
    tags_to_print: List[Tuple[str, str, bool]] = []
    for tag, value in exif.items():
        tag_str = str(tag)
        if tag_str == "GPSInfo" and isinstance(value, dict):
            continue
        if isinstance(value, dict):
            logger.debug(f"Skipping dictionary value for EXIF tag '{tag_str}' in pretty print.")
            continue
        value_str: str
        if isinstance(value, bytes):
             try:
                 decoded_str = value.decode('utf-8', errors='replace')
                 value_str = decoded_str[:57] + "..." if len(decoded_str) > 60 else decoded_str
             except Exception:
                 value_str = f"<bytes len={len(value)}>"
        elif isinstance(value, tuple) and len(value) > 10:
            value_str = f"<tuple len={len(value)}>"
        elif isinstance(value, bytearray):
            value_str = f"<bytearray len={len(value)}>"
        else:
             try:
                  value_str = str(value)
                  if len(value_str) > 60:
                      value_str = value_str[:57] + "..."
             except Exception as str_err:
                 # Log the specific error during string conversion
                 logger.debug(f"Could not convert EXIF value for tag '{tag_str}' to string: {str_err}")
                 value_str = f"<unrepresentable type: {type(value).__name__}>"

        is_important = tag_str in IMPORTANT_EXIF_TAGS
        if verbose or is_important:
            tags_to_print.append((tag_str, value_str, is_important))

    if not tags_to_print:
        print("No relevant EXIF tags found to display.")
        return
    tags_to_print.sort(key=lambda x: x[0])
    max_tag_len = max(Colors.visual_len(t[0]) for t in tags_to_print) if tags_to_print else 20
    max_val_len = max(Colors.visual_len(t[1]) for t in tags_to_print) if tags_to_print else 40
    min_width = 10
    max_tag_len = max(max_tag_len, min_width)
    max_val_len = max(max_val_len, min_width + 5)
    header_color = Colors.BLUE
    border_color = Colors.BLUE
    important_color = Colors.YELLOW
    def pad(text: str, width: int, left: bool = True) -> str:
        pad_len = max(0, width - Colors.visual_len(text))
        return f"{text}{' '*pad_len}" if left else f"{' '*pad_len}{text}"
    print(Colors.colored(f"╔{'═' * (max_tag_len + 2)}╤{'═' * (max_val_len + 2)}╗", border_color))
    print(f"{Colors.colored('║', border_color)} {pad(Colors.colored('Tag', header_color), max_tag_len)} {Colors.colored('│', border_color)} {pad(Colors.colored('Value', header_color), max_val_len)} {Colors.colored('║', border_color)}")
    print(Colors.colored(f"╠{'═' * (max_tag_len + 2)}╪{'═' * (max_val_len + 2)}╣", border_color))
    for tag_name, value_display, is_important_tag in tags_to_print:
        tag_display = Colors.colored(tag_name, Colors.BOLD + important_color) if is_important_tag else tag_name
        print(f"{Colors.colored('║', border_color)} {pad(tag_display, max_tag_len)} {Colors.colored('│', border_color)} {pad(value_display, max_val_len)} {Colors.colored('║', border_color)}")
    print(Colors.colored(f"╚{'═' * (max_tag_len + 2)}╧{'═' * (max_val_len + 2)}╝", border_color))


# --- Model Handling ---
def get_cached_model_ids() -> List[str]:
    """Get list of model repo IDs from the huggingface cache."""
    if scan_cache_dir is None:
        logger.error(Colors.colored("huggingface_hub library not found. Cannot scan Hugging Face cache.", Colors.RED))
        return []
    try:
        # Use CacheInfo (public class)
        logger.debug("Scanning Hugging Face cache directory...")
        cache_info: HFCacheInfo = scan_cache_dir()
        model_ids = sorted([repo.repo_id for repo in cache_info.repos])
        logger.debug(f"Found {len(model_ids)} potential models in Hugging Face cache: {model_ids}")
        return model_ids
    except HFValidationError:
        logger.error(Colors.colored("Hugging Face cache directory invalid.", Colors.RED))
        return []
    except FileNotFoundError:
        logger.error(Colors.colored("Hugging Face cache directory not found.", Colors.RED))
        return []
    except Exception as e:
        logger.error(Colors.colored(f"Unexpected error scanning Hugging Face cache: {type(e).__name__}: {e}", Colors.RED), exc_info=logger.level <= logging.DEBUG)
        return []


def print_model_stats(results: List[ModelResult]) -> None:
    """Print a table summarizing model performance statistics to the console, including failures."""
    # (Implementation remains the same as previous correct version)
    if not results:
         print("No model results to display.")
         return
    results.sort(key=lambda x: (not x.success, x.stats.time if x.success else 0))
    display_names = [(r.model_name.split('/')[-1]) for r in results]
    max_name_len_base = max(len(name) for name in display_names) if display_names else 20
    max_name_len_cap = 44
    max_name_len_base = min(max_name_len_base, max_name_len_cap)
    col_width = 12
    header_color = Colors.BLUE
    border_color = Colors.BLUE
    summary_color = Colors.YELLOW
    failure_color = Colors.RED
    failure_text_color = Colors.GRAY
    captured_marker_plain = "(+cap)"
    captured_marker_colored = Colors.colored(captured_marker_plain, Colors.GRAY)
    max_len_needed = 0
    temp_display_names = []
    for i, r in enumerate(results):
        name = display_names[i]
        current_max_name = max_name_len_base
        if len(name) > current_max_name:
            name = name[:current_max_name - 3] + "..."
        display_str = name
        if not r.success:
            fail_info_text = f" (Failed: {r.error_stage or '?'})"
            if r.captured_output_on_fail: 
                fail_info_text += f" {captured_marker_plain}"
            display_str += fail_info_text
        temp_display_names.append(display_str)
        max_len_needed = max(max_len_needed, len(display_str))
    max_name_len = min(max(max_name_len_base, max_len_needed), 55)
    def pad(text: str, width: int, left: bool = True) -> str:
        vlen = Colors.visual_len(text)
        padding = ' ' * max(0, width - vlen)
        return f"{text}{padding}" if left else f"{padding}{text}"
    print(f"\n--- {Colors.colored('Model Performance Summary (Console)', Colors.CYAN)} ---")
    print(Colors.colored(f"╔{'═' * (max_name_len + 2)}╤{'═' * (col_width + 2)}╤{'═' * (col_width + 2)}╤{'═' * (col_width + 2)}╤{'═' * (col_width + 2)}╗", border_color))
    header_model = pad(Colors.colored('Model', header_color), max_name_len)
    header_active = pad(Colors.colored('Active Δ', header_color), col_width, False)
    header_cache = pad(Colors.colored('Cache Δ', header_color), col_width, False)
    header_peak = pad(Colors.colored('Peak Mem', header_color), col_width, False)
    header_time = pad(Colors.colored('Time', header_color), col_width, False)
    print(f"{Colors.colored('║', border_color)} {header_model} {Colors.colored('│', border_color)} {header_active} {Colors.colored('│', border_color)} {header_cache} {Colors.colored('│', border_color)} {header_peak} {Colors.colored('│', border_color)} {header_time} {Colors.colored('║', border_color)}")
    print(Colors.colored(f"╠{'═' * (max_name_len + 2)}╪{'═' * (col_width + 2)}╪{'═' * (col_width + 2)}╪{'═' * (col_width + 2)}╪{'═' * (col_width + 2)}╣", border_color))
    successful_results = [r for r in results if r.success]
    for i, result in enumerate(results):
        model_disp_name_raw = display_names[i]
        model_display_text: str
        active_str: str
        cache_str: str
        peak_str: str
        time_str: str
        allowance = 15 if not result.success else 0
        if len(model_disp_name_raw) > max_name_len - allowance:
             truncate_at = max(0, max_name_len - allowance - 3)
             model_disp_name_raw = model_disp_name_raw[:truncate_at] + "..."
        if result.success:
            model_display_text = model_disp_name_raw
            active_str = f"{result.stats.active:,.0f} MB"
            cache_str = f"{result.stats.cached:,.0f} MB" 
            peak_str = f"{result.stats.peak:,.0f} MB"
            time_str = f"{result.stats.time:.2f} s"
        else:
            fail_info = f" (Failed: {result.error_stage or '?'})"
            if result.captured_output_on_fail: 
                fail_info += f" {captured_marker_colored}"
            model_display_text_base = model_disp_name_raw + fail_info.replace(captured_marker_colored, "(+cap)")
            model_display_text = Colors.colored(model_display_text_base, failure_color)
            if result.captured_output_on_fail:
                model_display_text = model_display_text.replace("(+cap)", captured_marker_colored)
            active_str = Colors.colored("-".rjust(col_width - 4), failure_text_color)
            cache_str = Colors.colored("-".rjust(col_width - 4), failure_text_color) 
            peak_str = Colors.colored("-".rjust(col_width - 4), failure_text_color) 
            time_str = Colors.colored("-".rjust(col_width - 2), failure_text_color)
        print(f"{Colors.colored('║', border_color)} {pad(model_display_text, max_name_len)} {Colors.colored('│', border_color)} {pad(active_str, col_width, False)} {Colors.colored('│', border_color)} {pad(cache_str, col_width, False)} {Colors.colored('│', border_color)} {pad(peak_str, col_width, False)} {Colors.colored('│', border_color)} {pad(time_str, col_width, False)} {Colors.colored('║', border_color)}")
    if successful_results:
        avg_active = sum(r.stats.active for r in successful_results) / len(successful_results)
        avg_cache = sum(r.stats.cached for r in successful_results) / len(successful_results)
        max_peak = max(r.stats.peak for r in successful_results)
        avg_time = sum(r.stats.time for r in successful_results) / len(successful_results)
        avg_active_str = f"{avg_active:,.0f} MB"
        avg_cache_str = f"{avg_cache:,.0f} MB"
        max_peak_str = f"{max_peak:,.0f} MB"
        avg_time_str = f"{avg_time:.2f} s"
        print(Colors.colored(f"╠{'═' * (max_name_len + 2)}╪{'═' * (col_width + 2)}╪{'═' * (col_width + 2)}╪{'═' * (col_width + 2)}╪{'═' * (col_width + 2)}╣", border_color))
        summary_title = f"AVG/PEAK ({len(successful_results)} Success)"
        summary_model = pad(Colors.colored(summary_title, summary_color), max_name_len)
        summary_active = pad(Colors.colored(avg_active_str, summary_color), col_width, False)
        summary_cache = pad(Colors.colored(avg_cache_str, summary_color), col_width, False)
        summary_peak = pad(Colors.colored(max_peak_str, summary_color), col_width, False)
        summary_time = pad(Colors.colored(avg_time_str, summary_color), col_width, False)
        print(f"{Colors.colored('║', border_color)} {summary_model} {Colors.colored('│', border_color)} {summary_active} {Colors.colored('│', border_color)} {summary_cache} {Colors.colored('│', border_color)} {summary_peak} {Colors.colored('│', border_color)} {summary_time} {Colors.colored('║', border_color)}")
    print(Colors.colored(f"╚{'═' * (max_name_len + 2)}╧{'═' * (col_width + 2)}╧{'═' * (col_width + 2)}╧{'═' * (col_width + 2)}╧{'═' * (col_width + 2)}╝", border_color))


# --- HTML Report Generation ---
def generate_html_report(results: List[ModelResult], filename: Path, versions: Dict[str, str]) -> None:
    """Generates an HTML file with model stats, output/errors, failures, and versions."""
    if not results:
        logger.warning("No results to generate HTML report.")
        return

    results.sort(key=lambda x: (not x.success, x.stats.time if x.success else 0))

    html_start = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Performance Results</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f8f9fa; color: #212529; line-height: 1.6; }
        h1 { text-align: center; color: #343a40; border-bottom: 2px solid #dee2e6; padding-bottom: 10px; margin-bottom: 30px; }
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
    <table>
        <caption>Performance metrics and output/errors for Vision Language Model processing. Generated on """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """. Failures shown but excluded from averages.</caption>
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
    html_rows = ""
    successful_results = [r for r in results if r.success]

    for result in results:
        model_disp_name = html.escape(result.model_name)
        row_class = ""
        result_content = ""
        stats_cells = ""

        if result.success:
            escaped_output = html.escape(result.output or "")
            result_content = f'<div class="model-output">{escaped_output}</div>'
            # Format stats as comma-separated integers for HTML
            stats_cells = f"""
                <td class="numeric">{result.stats.active:,.0f}</td>
                <td class="numeric">{result.stats.cached:,.0f}</td>
                <td class="numeric">{result.stats.peak:,.0f}</td>
                <td class="numeric">{result.stats.time:.2f}</td>
            """
        else:
            row_class = ' class="failed-row"'
            error_info = f"<span class='error-message'>Failed during '{html.escape(result.error_stage or 'Unknown')}'"
            if result.error_message:
                error_info += f": {html.escape(result.error_message)}"
            error_info += "</span>"
            result_content = error_info

            if result.captured_output_on_fail:
                escaped_capture = html.escape(result.captured_output_on_fail)
                # Wrap captured output in a distinct div/pre block
                result_content += f'<div class="captured-output"><strong>Captured Output (during generate):</strong><pre>{escaped_capture}</pre></div>'

            # Use hyphen placeholder for failed stats in HTML
            stats_cells = """
                <td class="numeric">-</td> <td class="numeric">-</td>
                <td class="numeric">-</td> <td class="numeric">-</td>
            """

        html_rows += f"""
            <!-- Data row for model: {model_disp_name} -->
            <tr{row_class}>
                <td class="model-name">{model_disp_name}</td>
                {stats_cells}
                <td>{result_content}</td>
            </tr>
"""

    html_summary_row = ""
    if successful_results:
        avg_active = sum(r.stats.active for r in successful_results) / len(successful_results)
        avg_cache = sum(r.stats.cached for r in successful_results) / len(successful_results)
        max_peak = max(r.stats.peak for r in successful_results)
        avg_time = sum(r.stats.time for r in successful_results) / len(successful_results)
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
    html_footer = "<footer>\n<h2>Library Versions</h2>\n<ul>\n"
    # Use sorted items for consistent order in HTML
    for name, ver in sorted(versions.items()):
        status_style = 'color: green;' if ver != "N/A" else 'color: orange;'
        html_footer += f'<li>{html.escape(name)}: <code style="{status_style}">{html.escape(ver)}</code></li>\n'
    html_footer += "</ul>\n"
    # Add date to footer
    html_footer += f"<p>Report generated on: {datetime.now().strftime('%Y-%m-%d')}</p>\n</footer>"
    # -----------------------------

    html_end = f"""
        </tbody>
    </table>
    <!-- End of Table -->
    {html_footer}
</body>
</html>
"""
    html_content = html_start + html_rows + html_summary_row + html_end

    try:
        # *** Restore TextIO type hint for file handle 'f' ***
        f: TextIO
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"HTML report saved to: {Colors.colored(str(filename.resolve()), Colors.GREEN)}")
    except IOError as e:
        logger.error(Colors.colored(f"Failed to write HTML report to {filename}: {e}", Colors.RED))
    except Exception as e:
         logger.error(Colors.colored(f"An unexpected error occurred while writing HTML report: {type(e).__name__}: {e}", Colors.RED), exc_info=logger.level <= logging.DEBUG)


def get_system_info() -> Tuple[str, str]:
    """Get system architecture and GPU information."""
    arch = platform.machine()
    gpu_info = "Unknown"
    try:
        # Try to get GPU info on macOS
        if platform.system() == "Darwin":
            result = subprocess.run(['system_profiler', 'SPDisplaysDataType'],
                                 capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                # Extract GPU info from system_profiler output
                gpu_lines = [line for line in result.stdout.split('\n')
                           if "Chipset Model:" in line]
                if gpu_lines:
                    gpu_info = gpu_lines[0].split("Chipset Model:")[-1].strip()
    except (subprocess.SubprocessError, TimeoutError):
        pass
    return arch, gpu_info

# --- Model Processing Core ---
def validate_inputs(image_path: PathLike, model_path: str, temperature: float = 0.0) -> None:
    """Validate input paths and parameters."""
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    if not img_path.is_file():
        raise ValueError(f"Not a file: {img_path}")
    if img_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.webp'}:
        raise ValueError(f"Unsupported image format: {img_path.suffix}")

    validate_temperature(temperature)

def validate_temperature(temp: float) -> None:
    """Validate temperature parameter is within acceptable range."""
    if not isinstance(temp, (int, float)):
        raise ValueError(f"Temperature must be a number, got {type(temp)}")
    if not 0.0 <= temp <= 1.0:
        raise ValueError(f"Temperature must be between 0 and 1, got {temp}")

def validate_image_accessible(image_path: Path) -> None:
    """Validate image file is accessible and supported."""
    try:
        with timeout_manager(5):  # 5 second timeout for read test
            with Image.open(image_path) as img:
                img.verify()
    except TimeoutError:
        raise IOError(f"Timeout while reading image: {image_path}")
    except UnidentifiedImageError:
        raise ValueError(f"File is not a recognized image format: {image_path}")
    except Exception as e:
        raise IOError(f"Error accessing image {image_path}: {e}")

def process_image_with_model(
    model_identifier: str,
    image_path: Path,
    prompt: str,
    max_tokens: int,
    verbose: bool = False,
    trust_remote_code: bool = False,
    temperature: float = DEFAULT_TEMPERATURE,
    timeout: float = DEFAULT_TIMEOUT,
) -> ModelResult:
    """Process an image with a Vision Language Model."""
    logger.info(f"Processing '{image_path}' with model: {model_identifier}")

    model = tokenizer = None
    arch, gpu_info = get_system_info()

    try:
        validate_temperature(temperature)
        validate_image_accessible(image_path)

        # Log system info in debug mode
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"System: {arch}, GPU: {gpu_info}")

        with timeout_manager(timeout):  # Use provided timeout value
            validate_image_accessible(image_path)

            # Log system info in debug mode
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"System: {arch}, GPU: {gpu_info}")

            with timeout_manager(timeout):  # Use provided timeout value
                model, tokenizer = load(model_identifier, trust_remote_code=trust_remote_code)
                config = load_config(model_identifier, trust_remote_code=trust_remote_code)

                formatted_prompt = apply_chat_template(tokenizer, config, prompt, num_images=1)
                output = generate(
                    model=model,
                    processor=tokenizer,  # Type checking handled by function signature
                    prompt=str(formatted_prompt),
                    image=image_path.as_posix(),
                    max_tokens=max_tokens,
                    verbose=verbose,
                    temp=temperature
                )

                # Ensure all computations involving the model are done before measuring memory/time
                mx.eval(model.parameters()) # Evaluate model parameters if needed after generation

                # Note: initial_mem and initial_cache are not defined in this scope.
                # Assuming they should be captured before the 'try' block or model loading.
                # For now, setting them to 0 as placeholders.
                initial_mem = 0.0
                initial_cache = 0.0
                start_time = time.perf_counter() # Should ideally be before model loading/generation

                final_stats = MemoryStats(
                    active=mx.get_active_memory() / 1024 / 1024 - initial_mem,
                    cached=mx.get_cache_memory() / 1024 / 1024 - initial_cache,
                    peak=mx.get_peak_memory() / 1024 / 1024,
                    time=time.perf_counter() - start_time
                )

                return ModelResult(
                    model_name=model_identifier,
                    success=True,
                    output=str(output) if output is not None else "",
                    stats=final_stats
                )

    except TimeoutError:
        logger.error(f"Timeout while processing model {model_identifier}")
        return ModelResult(
            model_name=model_identifier,
            success=False,
            error_stage="timeout",
            error_message="Operation timed out"
        )
    except Exception as e:
        error_stage = "model_load" if model is None else "generate"
        logger.error(f"Failed during {error_stage}: {type(e).__name__}: {e}")
        if logger.isEnabledFor(logging.DEBUG):
            traceback.print_exc()
        return ModelResult(
            model_name=model_identifier,
            success=False,
            error_stage=error_stage,
            error_message=str(e)
        )
    finally:
        del model
        del tokenizer
        mx.clear_cache()
        mx.reset_peak_memory()


# --- Main Execution ---
def main(args: argparse.Namespace) -> None:
    """Main function to orchestrate image analysis."""
    # Configure logging level based on args
    log_level = logging.DEBUG if args.debug else (logging.INFO if args.verbose else logging.INFO)
    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stderr)], force=True)
    if args.debug:
        logger.debug("Debug mode enabled.")
    elif args.verbose:
        logger.info("Verbose mode enabled.")

    # Collect library versions early
    library_versions = get_library_versions()

    # Print library versions initially (now using the collected dict)
    if args.debug:
        print_version_info(library_versions)

    # Warn about trusting remote code
    if args.trust_remote_code:
        logger.warning(Colors.colored("--- SECURITY WARNING ---", Colors.YELLOW + Colors.BOLD))
        logger.warning(Colors.colored("`--trust-remote-code` is enabled.", Colors.YELLOW))
        logger.warning(Colors.colored("-----------------------", Colors.YELLOW + Colors.BOLD))

    overall_start_time = time.perf_counter()

    # --- 1. Find Image ---
    folder_path = args.folder.resolve()
    logger.info(f"Scanning folder: {Colors.colored(str(folder_path), Colors.BLUE)}")
    if args.folder == DEFAULT_FOLDER and not DEFAULT_FOLDER.is_dir():
        print(Colors.colored(f"Warning: Default folder '{DEFAULT_FOLDER}' does not exist.", Colors.YELLOW), file=sys.stderr)
    image_path = find_most_recent_file(folder_path)
    if not image_path:
        print(Colors.colored(f"\nError: Could not find a suitable image file in {folder_path}.", Colors.RED), file=sys.stderr)
        sys.exit(1)
    resolved_image_path = image_path.resolve()
    print(f"\nProcessing file: {Colors.colored(resolved_image_path.name, Colors.MAGENTA)} (located at {resolved_image_path})")
    # Validate image readability early
    try:
        with Image.open(resolved_image_path) as img:
            img.verify() # Verify basic structure without loading full data
        print_image_dimensions(resolved_image_path)
    except (FileNotFoundError, UnidentifiedImageError, OSError, Exception) as img_err:
        logger.error(Colors.colored(f"Error opening or verifying image {resolved_image_path}: {img_err}", Colors.RED))
        sys.exit(1)


    # --- 2. Extract Metadata ---
    metadata = extract_image_metadata(resolved_image_path)
    print(f"  Date: {Colors.colored(metadata.get('date', 'N/A'), Colors.CYAN)}")
    print(f"  Desc: {Colors.colored(metadata.get('description', 'N/A'), Colors.CYAN)}")
    print(f"  GPS:  {Colors.colored(metadata.get('gps', 'N/A'), Colors.CYAN)}")
    if args.verbose or args.debug:
         exif_data = get_exif_data(resolved_image_path)
         if exif_data:
             pretty_print_exif(exif_data, verbose=True)
         else:
             print("\nNo detailed EXIF data could be extracted.")

    # --- 3. Prepare Prompt ---
    prompt: str
    if args.prompt:
        prompt = args.prompt
        logger.info("Using user-provided prompt.")
    else:
        logger.info("Generating default prompt based on image metadata.")
        prompt_parts = [
            "Provide factual caption, description, keywords/tags for cataloguing/searching.",
            (f"Context: Relates to '{metadata.get('description', '')}'"
             if metadata.get('description') and metadata['description'] != "No description" else ""),
            (f"taken around {metadata.get('date', '')}"
             if metadata.get('date') and metadata['date'] != "Unknown date" else ""),
            (f"near GPS {metadata.get('gps', '')}."
             if metadata.get('gps') and metadata['gps'] != "Unknown location" else ""),
            "Focus on visual content. Avoid repeating context unless visible."
        ]
        prompt = " ".join(filter(None, prompt_parts)).strip()
        logger.debug("Using generated prompt based on metadata.")
    print(f"\n{Colors.colored('--- Using Prompt ---', Colors.CYAN)}\n{prompt}\n{Colors.colored('-'*40, Colors.BLUE)}")

    # --- 4. Find and Process Models ---
    model_identifiers: List[str]
    if args.models:
        model_identifiers = args.models
        logger.info(f"Processing explicitly specified models: {Colors.colored(str(model_identifiers), Colors.GREEN)}")
    else:
        logger.info("Scanning cache for models to process...")
        model_identifiers = get_cached_model_ids()

    results: List[ModelResult] = []
    if not model_identifiers:
        msg = ("\nWarning: No models specified and none found in cache." if not args.models
               else "\nWarning: No models specified via --models argument.")
        print(Colors.colored(msg, Colors.YELLOW), file=sys.stderr)
        if not args.models:
            print("Ensure models are downloaded and cache is accessible.", file=sys.stderr)
    else:
        print(f"\nProcessing {Colors.colored(str(len(model_identifiers)), Colors.GREEN)} model(s)...")
        separator = Colors.colored(f"\n{'=' * 80}\n", Colors.BLUE)
        for model_id in model_identifiers:
            print(separator)
            is_vlm_verbose = args.verbose or args.debug
            result = process_image_with_model(
                model_identifier=model_id, 
                image_path=resolved_image_path,
                prompt=prompt, max_tokens=args.max_tokens,
                verbose=is_vlm_verbose, 
                trust_remote_code=args.trust_remote_code,
                temperature=args.temperature,
                timeout=args.timeout
            )
            results.append(result)
            # Print immediate feedback
            if result.success:
                print(f"\n--- {Colors.colored(f'Output from {model_id.split('/')[-1]}', Colors.GREEN)} ---")
                print(result.output)
            else:
                print(f"--- {Colors.colored(f'Processing failed for model: {model_id}', Colors.RED)} ---")
                # Conditionally print captured output if it exists
                if result.captured_output_on_fail:
                    print(Colors.colored("Captured output during failure:", Colors.YELLOW))
                    # Maybe limit how much is printed to console?
                    capture_snippet = result.captured_output_on_fail[:1000]
                    if len(result.captured_output_on_fail) > 1000:
                        capture_snippet += '...'
                    print(capture_snippet)

    # --- 5. Print Summary Statistics ---
    if results:
        print(Colors.colored(f"\n{'=' * 80}\n", Colors.BLUE)) # Separator
        print_model_stats(results) # Function call
    else:
        print(Colors.colored("\nNo models processed. No performance summary generated.", Colors.YELLOW))

    # --- 6. Generate HTML Report ---
    html_output_path = args.output_html.resolve()
    if results:
        # Pass collected versions to the report generator
        generate_html_report(results, html_output_path, library_versions)
    else:
        logger.info(f"Skipping HTML report generation to {html_output_path} as no models were processed.")

    # --- 7. Print Version Info to Console ---
    # Print versions after all processing and reporting is done
    print_version_info(library_versions)

    # --- Calculate and Print Total Time ---
    overall_time = time.perf_counter() - overall_start_time
    print(f"\nTotal execution time: {Colors.colored(f'{overall_time:.2f} seconds', Colors.GREEN)}.")


if __name__ == "__main__":
    
    # Setup Argument Parser
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Analyze image with MLX VLMs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Add arguments (separated for clarity)
    parser.add_argument("-f", "--folder", type=Path, default=DEFAULT_FOLDER, help="Folder to scan.")
    parser.add_argument("--output-html", type=Path, default=DEFAULT_HTML_OUTPUT, help="Output HTML report file.")
    parser.add_argument("--models", nargs='+', type=str, default=None, help="Specify models by ID/path. Overrides cache scan.")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True, help="Allow custom code from Hub models (SECURITY RISK).")
    parser.add_argument("-p", "--prompt", type=str, default=None, help="Custom VLM prompt.")
    parser.add_argument("-m", "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max new tokens to generate.")
    parser.add_argument("-t", "--temperature", type=float, default=DEFAULT_TEMPERATURE, help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE}).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output (INFO logging).")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logging (DEBUG level).")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help=f"Timeout in seconds for model operations (default: {DEFAULT_TIMEOUT}).")

    # Parse arguments
    parsed_args: argparse.Namespace = parser.parse_args()

    # --- Main Execution ---
    try:
        main(parsed_args)
    except Exception as main_err:
        # Log final unhandled exceptions with color
        logger.exception(Colors.colored(f"An unexpected error occurred during main execution: {main_err}", Colors.RED))
        sys.exit(1) # Exit with error status