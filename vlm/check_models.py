#!/usr/bin/env python3
"""Image analysis and caption generation using MLX Vision Language Models."""

# Standard library imports
import argparse
import contextlib
import html
import io
import logging
import re  # Added for ANSI code stripping
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import ( # Removed IO
    Any, Dict, Final, List, NamedTuple,
    Optional, TextIO, Tuple, Union
)

# Third-party imports
import mlx.core as mx
from huggingface_hub import HFCacheInfo, scan_cache_dir
from huggingface_hub.errors import HFValidationError
from PIL import Image, UnidentifiedImageError
from PIL.ExifTags import GPSTAGS, TAGS

# Local application/library specific imports
from mlx_vlm import (__version__ as vlm_version, generate, load)
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

# Configure logging
logger = logging.getLogger(__name__)
# BasicConfig called in main()


# --- ANSI Color Codes for Console Output ---
class Colors:
    """ANSI color codes for terminal output"""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"

    _enabled = sys.stderr.isatty()
    # Regex to remove ANSI escape codes
    _ansi_escape_re = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    @staticmethod
    def colored(text: str, color: str) -> str:
        """Wrap text in color codes if outputting to a TTY."""
        if not Colors._enabled:
            return text
        return f"{color}{text}{Colors.RESET}"

    @staticmethod
    def visual_len(text: str) -> int:
        """Calculate the visual length of a string (stripping ANSI codes)."""
        if not isinstance(text, str): # Handle non-string inputs gracefully
             text = str(text)
        return len(Colors._ansi_escape_re.sub('', text))


# Type aliases and definitions
ExifValue = Any
ExifDict = Dict[Union[str, int], ExifValue]
MetadataDict = Dict[str, str]
PathLike = Union[str, Path]

# Constants - Defaults
DEFAULT_MAX_TOKENS: Final[int] = 500
DEFAULT_FOLDER: Final[Path] = Path.home() / "Pictures" / "Processed"
DEFAULT_HTML_OUTPUT: Final[Path] = Path("results.html")
DEFAULT_TEMPERATURE: Final[float] = 0.0

# Constants - EXIF
IMPORTANT_EXIF_TAGS: Final[frozenset[str]] = frozenset({
    "DateTimeOriginal", "ImageDescription", "CreateDate", "Make", "Model",
    "LensModel", "ExposureTime", "FNumber", "ISOSpeedRatings",
    "FocalLength", "ExposureProgram",
})
DATE_FORMATS: Final[Tuple[str, ...]] = (
    "%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y%m%d"
)
EXIF_DATE_TAGS: Final[Tuple[str, ...]] = (
    "DateTimeOriginal", "CreateDate", "DateTime"
)
GPS_LAT_REF_TAG: Final[int] = 1
GPS_LAT_TAG: Final[int] = 2
GPS_LON_REF_TAG: Final[int] = 3
GPS_LON_TAG: Final[int] = 4


# Type definitions
class MemoryStats(NamedTuple):
    """Memory statistics container (values represent deltas or peak)."""
    active: float
    cached: float
    peak: float
    time: float

    @staticmethod
    def zero() -> 'MemoryStats':
        """Create a zero-initialized MemoryStats object."""
        return MemoryStats(active=0.0, cached=0.0, peak=0.0, time=0.0)


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
        # Sort by modification time, newest first
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
    """Extract EXIF data from an image file."""
    try:
        img: Image.Image
        with Image.open(image_path) as img:
            exif_raw: Any = img.getexif()
            if not exif_raw:
                logger.debug(f"No EXIF data found in {image_path}")
                return None

            exif_decoded: ExifDict = {
                TAGS.get(tag_id, tag_id): value
                for tag_id, value in exif_raw.items()
            }
            try:
                 gps_ifd: Optional[Dict[int, Any]] = exif_raw.get_ifd(Image.Exif.IFD.GPSInfo)
                 if gps_ifd:
                     exif_decoded["GPSInfo"] = {
                         GPSTAGS.get(gps_tag_id, gps_tag_id): gps_value
                         for gps_tag_id, gps_value in gps_ifd.items()
                     }
            except KeyError:
                 logger.debug(f"GPSInfo IFD (key {Image.Exif.IFD.GPSInfo}) not found in {image_path}")
            except Exception as e_gps:
                 logger.warning(f"Could not read GPS IFD for {image_path}: {type(e_gps).__name__} {e_gps}")

            logger.debug(f"Successfully extracted EXIF for {image_path}")
            return exif_decoded

    except FileNotFoundError:
        logger.error(Colors.colored(f"Image file not found for EXIF extraction: {image_path}", Colors.RED))
        return None
    except UnidentifiedImageError:
        logger.error(Colors.colored(f"Cannot identify image file for EXIF: {image_path}", Colors.RED))
        return None
    except AttributeError as e:
        logger.warning(Colors.colored(f"PIL lacks getexif/get_ifd method or image has no EXIF support ({e}): {image_path}", Colors.YELLOW))
        return None
    except Exception as e:
        logger.error(Colors.colored(f"Unexpected error reading EXIF from {image_path}: {e}", Colors.RED), exc_info=logger.level <= logging.DEBUG)
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


GPSTupleElement = Any
GPSTuple = Tuple[GPSTupleElement, GPSTupleElement, GPSTupleElement]


def _convert_gps_coordinate(ref: Optional[Union[str, bytes]], coord_tuple: Optional[GPSTuple]) -> Optional[float]:
    """Convert EXIF GPS coordinate tuple to decimal degrees."""
    if ref is None or coord_tuple is None or len(coord_tuple) != 3:
        logger.debug(f"Invalid input for GPS conversion: ref={ref}, coord_tuple={coord_tuple}")
        return None
    try:
        ref_str: str
        if isinstance(ref, bytes):
            ref_str = ref.decode('ascii')
        elif isinstance(ref, str):
            ref_str = ref
        else:
            logger.warning(f"Unexpected type for GPS reference: {type(ref)}")
            return None

        ref_upper: str = ref_str.upper()
        if ref_upper not in ['N', 'S', 'E', 'W']:
            logger.warning(f"Invalid GPS reference value: {ref_str}")
            return None

        def get_float_val(item: GPSTupleElement) -> float:
             """Safely convert GPS coordinate component to float."""
             if hasattr(item, 'numerator') and hasattr(item, 'denominator'):
                 num = item.numerator
                 den = item.denominator
                 if den == 0:
                     logger.warning("GPS coordinate component has zero denominator.")
                     return float(num)
                 return float(num) / float(den)
             elif isinstance(item, (int, float)):
                 return float(item)
             else:
                 try:
                     return float(item)
                 except (TypeError, ValueError) as e_conv:
                     logger.warning(f"Could not convert GPS component '{item}' to float: {e_conv}")
                     # Raise error to be caught by outer try-except
                     raise ValueError("Invalid GPS component type") from e_conv

        degrees = get_float_val(coord_tuple[0])
        minutes = get_float_val(coord_tuple[1])
        seconds = get_float_val(coord_tuple[2])
        decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)

        if ref_upper in ['S', 'W']:
            decimal = -decimal
        return decimal

    except ValueError as ve:
        # Catch specific error from get_float_val
        logger.warning(Colors.colored(f"Failed GPS conversion due to invalid component type: {ve}", Colors.YELLOW))
        return None
    except (AttributeError, IndexError, TypeError, ZeroDivisionError, UnicodeDecodeError) as e:
        # Catch other potential errors during processing
        logger.warning(Colors.colored(f"Error converting GPS coordinate component: {type(e).__name__} {e} (Ref: {ref}, Coords: {coord_tuple})", Colors.YELLOW))
        return None


def extract_image_metadata(image_path: Path) -> MetadataDict:
    """Extract key metadata (Date, Description, GPS) from image EXIF."""
    metadata: MetadataDict = {"date": "Unknown date", "description": "No description", "gps": "Unknown location"}
    exif = get_exif_data(image_path)
    if not exif:
        logger.debug(f"No EXIF data found for {image_path}, returning default metadata.")
        return metadata

    # Extract Date
    found_date = False
    for tag in EXIF_DATE_TAGS:
        date_val = exif.get(tag)
        if date_val is not None:
            formatted_date = _format_exif_date(date_val)
            if formatted_date:
                metadata["date"] = formatted_date
                logger.debug(f"Found and formatted date from tag '{tag}': {formatted_date}")
                found_date = True
                break # Found the first valid date
            else:
                # Continue loop if formatting failed for this tag
                logger.debug(f"Date tag '{tag}' ('{date_val}') found but could not be formatted.")
    if not found_date:
        logger.debug(f"Could not find or format a suitable date tag in {EXIF_DATE_TAGS}")

    # Extract Description
    desc_val = exif.get("ImageDescription")
    if desc_val is not None:
        try:
            desc_str: str
            if isinstance(desc_val, bytes):
                desc_str = desc_val.decode('utf-8', errors='replace')
            else:
                desc_str = str(desc_val)
            metadata["description"] = desc_str.strip()
            logger.debug(f"Found description: '{metadata['description']}'")
        except Exception as e:
            logger.warning(f"Could not convert description value '{desc_val}' to string: {e}")

    # Extract GPS
    gps_info_val = exif.get("GPSInfo")
    if isinstance(gps_info_val, dict):
        logger.debug(f"Raw GPS Info dictionary: {gps_info_val}")
        lat_ref = gps_info_val.get("GPSLatitudeRef")
        lat_coord_raw = gps_info_val.get("GPSLatitude")
        lon_ref = gps_info_val.get("GPSLongitudeRef")
        lon_coord_raw = gps_info_val.get("GPSLongitude")

        lat_coord = lat_coord_raw if isinstance(lat_coord_raw, tuple) and len(lat_coord_raw) == 3 else None
        lon_coord = lon_coord_raw if isinstance(lon_coord_raw, tuple) and len(lon_coord_raw) == 3 else None

        lat = _convert_gps_coordinate(lat_ref, lat_coord)
        lon = _convert_gps_coordinate(lon_ref, lon_coord)

        if lat is not None and lon is not None:
            metadata["gps"] = f"{lat:+.6f}, {lon:+.6f}"
            logger.debug(f"Calculated GPS Coordinates: {metadata['gps']}")
        else:
            logger.debug("Could not calculate valid GPS coordinates from available tags.")
            if lat_coord is None or lon_coord is None:
                 logger.debug(f"GPS coordinate tuples were invalid or missing. Lat raw: {lat_coord_raw}, Lon raw: {lon_coord_raw}")
    elif gps_info_val is not None:
        # Value exists but isn't the expected dict
        logger.warning(f"GPSInfo tag found but was not a dictionary (type: {type(gps_info_val)}). Value: {gps_info_val}")
    else:
        # Tag is missing entirely
        logger.debug("No GPSInfo tag found in EXIF data.")
    return metadata


def pretty_print_exif(exif: ExifDict, verbose: bool = False) -> None:
    """Pretty print key EXIF data in a formatted table, using colors."""
    if not exif:
        print("No EXIF data available.")
        return

    print(f"\n--- {Colors.colored('Key EXIF Data', Colors.CYAN)} ---")
    tags_to_print: List[Tuple[str, str, bool]] = []
    for tag, value in exif.items():
        tag_str = str(tag)
        # Skip complex structures handled elsewhere or not printable
        if tag_str == "GPSInfo" and isinstance(value, dict):
            continue
        if isinstance(value, dict):
            logger.debug(f"Skipping dictionary value for EXIF tag '{tag_str}' in pretty print.")
            continue

        value_str: str
        if isinstance(value, bytes):
             try:
                 decoded_str = value.decode('utf-8', errors='replace')
                 # Truncate long decoded strings
                 if len(decoded_str) > 60:
                      value_str = decoded_str[:57] + "..."
                 else:
                     value_str = decoded_str
             except Exception:
                 # Fallback for non-decodable bytes
                 value_str = f"<bytes len={len(value)}>"
        elif isinstance(value, tuple) and len(value) > 10:
             # Indicate long tuples without printing content
             value_str = f"<tuple len={len(value)}>"
        else:
             try:
                  value_str = str(value)
                  # Truncate other long string representations
                  if len(value_str) > 60:
                      value_str = value_str[:57] + "..."
             except Exception:
                  # Fallback for unrepresentable types
                  value_str = f"<unrepresentable type: {type(value).__name__}>"

        is_important = tag_str in IMPORTANT_EXIF_TAGS
        if verbose or is_important:
            tags_to_print.append((tag_str, value_str, is_important))

    if not tags_to_print:
        print("No relevant EXIF tags found to display based on current settings.")
        return

    tags_to_print.sort(key=lambda x: x[0])
    # Determine column widths dynamically based on visual length
    max_tag_len = max(Colors.visual_len(t[0]) for t in tags_to_print) if tags_to_print else 20
    max_val_len = max(Colors.visual_len(t[1]) for t in tags_to_print) if tags_to_print else 40
    min_width = 10
    max_tag_len = max(max_tag_len, min_width)
    max_val_len = max(max_val_len, min_width + 5)

    # Define colors
    header_color = Colors.BLUE
    border_color = Colors.BLUE
    important_color = Colors.YELLOW

    # Helper for padding colored strings correctly
    def pad(text: str, width: int, is_left_justified: bool = True) -> str:
        vlen = Colors.visual_len(text)
        padding = ' ' * max(0, width - vlen)
        if is_left_justified:
            return text + padding
        else:
            return padding + text

    # Print table header
    print(Colors.colored(f"╔{'═' * (max_tag_len + 2)}╤{'═' * (max_val_len + 2)}╗", border_color))
    print(f"{Colors.colored('║', border_color)} {pad(Colors.colored('Tag', header_color), max_tag_len)} {Colors.colored('│', border_color)} {pad(Colors.colored('Value', header_color), max_val_len)} {Colors.colored('║', border_color)}")
    print(Colors.colored(f"╠{'═' * (max_tag_len + 2)}╪{'═' * (max_val_len + 2)}╣", border_color))
    # Print table rows
    for tag_name, value_display, is_important_tag in tags_to_print:
        tag_display: str
        if is_important_tag:
            # Apply bold formatting
            tag_display = Colors.colored(tag_name, Colors.BOLD + important_color)
        else:
            tag_display = tag_name
        # Print the formatted row using the padding helper
        print(f"{Colors.colored('║', border_color)} {pad(tag_display, max_tag_len)} {Colors.colored('│', border_color)} {pad(value_display, max_val_len)} {Colors.colored('║', border_color)}")
    # Print table footer
    print(Colors.colored(f"╚{'═' * (max_tag_len + 2)}╧{'═' * (max_val_len + 2)}╝", border_color))


# --- Model Handling ---
def get_cached_model_ids() -> List[str]:
    """Get list of model repo IDs from the huggingface cache."""
    try:
        cache_info: HFCacheInfo = scan_cache_dir()
        # Sort models for consistent processing order
        model_ids: List[str] = sorted([repo.repo_id for repo in cache_info.repos])
        logger.debug(f"Found {len(model_ids)} potential models in cache: {model_ids}")
        return model_ids
    except HFValidationError:
        logger.error(Colors.colored("HF cache directory invalid.", Colors.RED))
        return []
    except FileNotFoundError:
        logger.error(Colors.colored("HF cache directory not found.", Colors.RED))
        return []
    except Exception as e:
        logger.error(Colors.colored(f"Unexpected error scanning HF cache: {type(e).__name__}: {e}", Colors.RED), exc_info=logger.level <= logging.DEBUG)
        return []


def print_model_stats(results: List[ModelResult]) -> None:
    """Print a table summarizing model performance statistics to the console, including failures."""
    if not results:
        print("No model results to display.")
        return

    # Sort results primarily by success, then by time for successful runs
    results.sort(key=lambda x: (not x.success, x.stats.time if x.success else 0))
    # Get base display names
    display_names = [(r.model_name.split('/')[-1]) for r in results]

    # *** Fix: Initialize max_name_len_base before use ***
    max_name_len_base = max(len(name) for name in display_names) if display_names else 20
    max_name_len_cap = 44 # Cap base length reasonably
    max_name_len_base = min(max_name_len_base, max_name_len_cap) # Apply cap

    col_width = 12 # Increased slightly for comma formatting
    # Define colors
    header_color = Colors.BLUE
    border_color = Colors.BLUE
    summary_color = Colors.YELLOW
    failure_color = Colors.RED
    failure_text_color = Colors.GRAY
    captured_marker_plain = "(+cap)"
    captured_marker_colored = Colors.colored(captured_marker_plain, Colors.GRAY) # Marker for captured output

    # Adjust max_name_len dynamically based on actual content including failure text
    max_len_needed = 0
    temp_display_names = [] # Store potentially modified names for width calculation
    for i, r in enumerate(results):
        name = display_names[i]
        # Apply base truncation before calculating potential length
        current_max_name = max_name_len_base # Use initialized value
        if len(name) > current_max_name:
            name = name[:current_max_name - 3] + "..."

        display_str = name # Start with (potentially truncated) name
        if not r.success:
            # Calculate length of appended failure info text (without ANSI codes)
            fail_info_text = f" (Failed: {r.error_stage or '?'})"
            if r.captured_output_on_fail:
                fail_info_text += f" {captured_marker_plain}" # Use plain marker for length calc
            display_str += fail_info_text
        temp_display_names.append(display_str)
        max_len_needed = max(max_len_needed, len(display_str))

    # Final column width for name, capped reasonably
    max_name_len = min(max(max_name_len_base, max_len_needed), 55)

    # Helper for padding colored strings correctly using visual length
    def pad(text: str, width: int, is_left_justified: bool = True) -> str:
        vlen = Colors.visual_len(text)
        padding = ' ' * max(0, width - vlen)
        if is_left_justified:
            return text + padding
        else:
            return padding + text

    print(f"\n--- {Colors.colored('Model Performance Summary (Console)', Colors.CYAN)} ---")
    # Adjust table width based on final max_name_len
    print(Colors.colored(f"╔{'═' * (max_name_len + 2)}╤{'═' * (col_width + 2)}╤{'═' * (col_width + 2)}╤{'═' * (col_width + 2)}╤{'═' * (col_width + 2)}╗", border_color))

    # Print Header Row using padding helper
    header_model = pad(Colors.colored('Model', header_color), max_name_len)
    header_active = pad(Colors.colored('Active Δ', header_color), col_width, False)
    header_cache = pad(Colors.colored('Cache Δ', header_color), col_width, False)
    header_peak = pad(Colors.colored('Peak Mem', header_color), col_width, False)
    header_time = pad(Colors.colored('Time', header_color), col_width, False)
    print(f"{Colors.colored('║', border_color)} {header_model} {Colors.colored('│', border_color)} {header_active} {Colors.colored('│', border_color)} {header_cache} {Colors.colored('│', border_color)} {header_peak} {Colors.colored('│', border_color)} {header_time} {Colors.colored('║', border_color)}")
    print(Colors.colored(f"╠{'═' * (max_name_len + 2)}╪{'═' * (col_width + 2)}╪{'═' * (col_width + 2)}╪{'═' * (col_width + 2)}╪{'═' * (col_width + 2)}╣", border_color))

    successful_results = [r for r in results if r.success]

    # Print Data Rows
    for i, result in enumerate(results):
        model_disp_name_raw = display_names[i] # Base short name
        model_display_text: str
        active_str: str
        cache_str: str
        peak_str: str
        time_str: str

        # Apply base truncation based on adjusted max_name_len allowance
        allowance = 15 if not result.success else 0
        if len(model_disp_name_raw) > max_name_len - allowance:
             # Ensure truncate_at is non-negative
             truncate_at = max(0, max_name_len - allowance - 3)
             model_disp_name_raw = model_disp_name_raw[:truncate_at] + "..."

        if result.success:
            model_display_text = model_disp_name_raw
            # Format stats as comma-separated integers
            active_str = f"{result.stats.active:,.0f} MB"
            cache_str = f"{result.stats.cached:,.0f} MB"
            peak_str = f"{result.stats.peak:,.0f} MB"
            time_str = f"{result.stats.time:.2f} s"
        else:
            fail_info = f" (Failed: {result.error_stage or '?'})"
            if result.captured_output_on_fail:
                fail_info += f" {captured_marker_colored}" # Use colored marker here
            # Color the fail info part
            model_display_text = model_disp_name_raw + Colors.colored(fail_info.replace(captured_marker_colored, "(+cap)"), failure_color)
            if result.captured_output_on_fail: # Add back colored marker
                 model_display_text = model_display_text.replace("(+cap)", captured_marker_colored)

            # Use dimmed placeholder for stats
            active_str = Colors.colored("-".rjust(col_width - 1), failure_text_color)
            cache_str = Colors.colored("-".rjust(col_width - 1), failure_text_color)
            peak_str = Colors.colored("-".rjust(col_width - 1), failure_text_color)
            time_str = Colors.colored("-".rjust(col_width - 1), failure_text_color)

        # Print row using padding helper for alignment
        print(
             f"{Colors.colored('║', border_color)} {pad(model_display_text, max_name_len)} {Colors.colored('│', border_color)} "
             f"{pad(active_str, col_width, False)} {Colors.colored('│', border_color)} "
             f"{pad(cache_str, col_width, False)} {Colors.colored('│', border_color)} "
             f"{pad(peak_str, col_width, False)} {Colors.colored('│', border_color)} "
             f"{pad(time_str, col_width, False)} {Colors.colored('║', border_color)}"
        )

    # Print average/summary row only if there were successful runs to average
    if successful_results: # Check if list is not empty
        avg_active = sum(r.stats.active for r in successful_results) / len(successful_results)
        avg_cache = sum(r.stats.cached for r in successful_results) / len(successful_results)
        max_peak = max(r.stats.peak for r in successful_results)
        avg_time = sum(r.stats.time for r in successful_results) / len(successful_results)
        # Format averages as comma-separated integers
        avg_active_str = f"{avg_active:,.0f} MB"
        avg_cache_str = f"{avg_cache:,.0f} MB"
        max_peak_str = f"{max_peak:,.0f} MB"
        avg_time_str = f"{avg_time:.2f} s"

        print(Colors.colored(f"╠{'═' * (max_name_len + 2)}╪{'═' * (col_width + 2)}╪{'═' * (col_width + 2)}╪{'═' * (col_width + 2)}╪{'═' * (col_width + 2)}╣", border_color))
        summary_title = f"AVG/PEAK ({len(successful_results)} Success)"
        # Color each part of the summary row individually
        summary_model = pad(Colors.colored(summary_title, summary_color), max_name_len)
        summary_active = pad(Colors.colored(avg_active_str, summary_color), col_width, False)
        summary_cache = pad(Colors.colored(avg_cache_str, summary_color), col_width, False)
        summary_peak = pad(Colors.colored(max_peak_str, summary_color), col_width, False)
        summary_time = pad(Colors.colored(avg_time_str, summary_color), col_width, False)

        print(f"{Colors.colored('║', border_color)} {summary_model} {Colors.colored('│', border_color)} {summary_active} {Colors.colored('│', border_color)} {summary_cache} {Colors.colored('│', border_color)} {summary_peak} {Colors.colored('│', border_color)} {summary_time} {Colors.colored('║', border_color)}")

    # Print Footer
    print(Colors.colored(f"╚{'═' * (max_name_len + 2)}╧{'═' * (col_width + 2)}╧{'═' * (col_width + 2)}╧{'═' * (col_width + 2)}╧{'═' * (col_width + 2)}╝", border_color))


# --- HTML Report Generation ---
def generate_html_report(results: List[ModelResult], filename: Path) -> None:
    """Generates an HTML file with model stats and output/errors, including failures."""
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
    # Only show summary if there was at least one success
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

    html_end = """
        </tbody>
    </table>
    <!-- End of Table -->
</body>
</html>
"""
    html_content = html_start + html_rows + html_summary_row + html_end

    try:
        f: TextIO[str] 
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"HTML report saved to: {Colors.colored(str(filename.resolve()), Colors.GREEN)}")
    except IOError as e:
        logger.error(Colors.colored(f"Failed to write HTML report to {filename}: {e}", Colors.RED))
    except Exception as e:
         logger.error(Colors.colored(f"An unexpected error occurred while writing HTML report: {type(e).__name__}: {e}", Colors.RED), exc_info=logger.level <= logging.DEBUG)


# --- Model Processing Core ---
def process_image_with_model(
    model_identifier: str,
    image_path: Path,
    prompt: str,
    max_tokens: int,
    vlm_verbose: bool,
    trust_remote_code: bool,
    temperature: float
) -> ModelResult:
    """Loads/runs VLM, tracks performance. Returns result object even on failure."""
    model_short_name = model_identifier.split('/')[-1]
    logger.info(f"Processing '{image_path.name}' with model: {Colors.colored(model_short_name, Colors.MAGENTA)}")

    model: Optional[Any] = None
    tokenizer: Optional[Any] = None
    config: Optional[Any] = None
    output: Optional[str] = None
    error_stage: Optional[str] = None
    error_message_str: Optional[str] = None
    captured_output_str: Optional[str] = None

    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()

    try:
        # Init & Measurement Start
        mx.clear_cache()
        mx.reset_peak_memory()
        initial_active_mb = mx.get_active_memory() / 1e6
        initial_cache_mb = mx.get_cache_memory() / 1e6
        start_time = time.perf_counter()

        # Stage 1: Load
        error_stage = 'load'
        load_start_time = time.perf_counter()
        logger.debug(f"Loading {model_short_name}...")
        model, tokenizer = load(model_identifier, trust_remote_code=trust_remote_code)
        config = load_config(model_identifier, trust_remote_code=trust_remote_code)
        logger.debug(f"Loaded {model_short_name} in {time.perf_counter() - load_start_time:.2f}s")

        # Stage 2: Prompt
        error_stage = 'prompt'
        logger.debug(f"Applying chat template for {model_short_name}...")
        formatted_prompt: str = apply_chat_template(tokenizer, config, prompt, num_images=1)
        logger.debug(f"Using formatted prompt: {formatted_prompt[:100]}...")

        # Stage 3: Generate (with capture)
        error_stage = 'generate'
        gen_start_time = time.perf_counter()
        logger.debug(f"Generating text with {model_short_name} (temp={temperature})...")
        with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
            output = generate(
                model=model, processor=tokenizer, prompt=formatted_prompt,
                image_paths=[str(image_path)], max_tokens=max_tokens,
                verbose=vlm_verbose, temp=temperature
            )
        logger.debug(f"Generated output in {time.perf_counter() - gen_start_time:.2f}s")

        # Finalize Measurements
        error_stage = 'measure'
        # Ensure model parameters are evaluated if model loaded successfully
        if model is not None and hasattr(model, 'parameters'):
             mx.eval(model.parameters()) # Ensure calculations finish
        end_time = time.perf_counter()
        final_active_mb = mx.get_active_memory() / 1e6
        final_cache_mb = mx.get_cache_memory() / 1e6
        peak_mb = mx.get_peak_memory() / 1e6
        elapsed_time = end_time - start_time
        active_delta_mb = final_active_mb - initial_active_mb
        cache_delta_mb = final_cache_mb - initial_cache_mb

        final_stats = MemoryStats(active=active_delta_mb, cached=cache_delta_mb, peak=peak_mb, time=elapsed_time)
        error_stage = None # Success

        # Format peak mem in log message
        logger.info(f"Finished processing {Colors.colored(model_short_name, Colors.MAGENTA)} in {Colors.colored(f'{elapsed_time:.1f}s', Colors.GREEN)}. Peak Mem: {Colors.colored(f'{peak_mb:,.0f}MB', Colors.CYAN)}")
        return ModelResult(model_name=model_identifier, success=True, output=str(output or ""), stats=final_stats)

    except Exception as e:
        failure_location = f" during {error_stage}" if error_stage else ""
        error_message_str = f"{type(e).__name__}: {e}"
        logger.error(Colors.colored(f"Failed to process model {model_identifier}{failure_location}: {error_message_str}", Colors.RED))
        if logger.isEnabledFor(logging.DEBUG):
            traceback.print_exc()

        # Capture output from buffers AFTER exception if failure was during generate
        if error_stage == 'generate':
            stdout_val = captured_stdout.getvalue()
            stderr_val = captured_stderr.getvalue()
            combined_capture = ""
            if stdout_val:
                combined_capture += f"--- Captured STDOUT ---\n{stdout_val.strip()}\n"
            if stderr_val:
                combined_capture += f"--- Captured STDERR ---\n{stderr_val.strip()}\n"
            # Assign only if there was actual captured output
            captured_output_str = combined_capture.strip() if combined_capture.strip() else None
            if captured_output_str:
                logger.warning(Colors.colored(f"Captured partial output during failed generation for {model_identifier}:", Colors.YELLOW))

        # Return failed result object
        return ModelResult(
            model_name=model_identifier, success=False,
            error_stage=error_stage, error_message=error_message_str,
            captured_output_on_fail=captured_output_str
            # Stats default to zero via dataclass field factory
        )
    finally:
        # Ensure cleanup even if errors occur
        # Check if variables exist before deleting
        if 'model' in locals() and model is not None:
            del model
        if 'tokenizer' in locals() and tokenizer is not None:
            del tokenizer
        if 'config' in locals() and config is not None:
            del config
        mx.clear_cache()


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

    # Print library versions
    try:
         print(f"MLX version: {Colors.colored(mx.__version__, Colors.GREEN)}")
         print(f"MLX-VLM version: {Colors.colored(vlm_version, Colors.GREEN)}\n")
         logger.debug(f"Default MLX device: {mx.default_device()}")
    except Exception as version_err:
        logger.warning(Colors.colored(f"Could not retrieve library version info: {version_err}", Colors.YELLOW))

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
    print_image_dimensions(resolved_image_path)

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
                model_identifier=model_id, image_path=resolved_image_path,
                prompt=prompt, max_tokens=args.max_tokens,
                vlm_verbose=is_vlm_verbose, trust_remote_code=args.trust_remote_code,
                temperature=args.temperature
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
        generate_html_report(results, html_output_path) # Function call
    else:
        logger.info(f"Skipping HTML report generation to {html_output_path} as no models were processed.")

    # --- Calculate and Print Total Time ---
    overall_time = time.perf_counter() - overall_start_time
    print(f"\nTotal execution time: {Colors.colored(f'{overall_time:.2f} seconds', Colors.GREEN)}.")


if __name__ == "__main__":
    # Setup Argument Parser
    parser = argparse.ArgumentParser(
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

    # Parse arguments
    parsed_args = parser.parse_args()

    # --- Main Execution ---
    try:
        main(parsed_args)
    except Exception as main_err:
        # Log final unhandled exceptions with color
        logger.exception(Colors.colored(f"An unexpected error occurred during main execution: {main_err}", Colors.RED))
        sys.exit(1) # Exit with error status