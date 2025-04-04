#!/usr/bin/env python3
"""Image analysis and caption generation using MLX Vision Language Models."""

# Standard library imports
import argparse
import html  # Import the html module for escaping
import logging
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import (
    Any, Dict, Final, IO, List, NamedTuple,
    Optional, Tuple, Union
)

# Third-party imports
import mlx.core as mx
from PIL import Image, UnidentifiedImageError
from PIL.ExifTags import GPSTAGS, TAGS
from mlx_vlm import (load, generate, __version__ as vlm_version)
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from huggingface_hub import scan_cache_dir, HFCacheInfo
from huggingface_hub.errors import HFValidationError

# Configure logging
logger = logging.getLogger(__name__)
# Note: BasicConfig setup should happen *before* any logging calls
# if module-level logging is done, but here it's fine as logger
# is instantiated first and configured later or in main.
# We will ensure configuration in main or before first use.


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

    # Check if stderr is a TTY once, assuming logs go to stderr
    _enabled = sys.stderr.isatty()

    @staticmethod
    def colored(text: str, color: str) -> str:
        """Wrap text in color codes if outputting to a TTY."""
        if not Colors._enabled:
            return text
        # Ensure reset is always appended for safety
        return f"{color}{text}{Colors.RESET}"

# Type aliases and definitions
ExifValue = Any # EXIF values can be complex (ints, strings, bytes, tuples, Ratio)
ExifDict = Dict[Union[str, int], ExifValue]
MetadataDict = Dict[str, str]         # Metadata results dictionary (string keys, string values)
PathLike = Union[str, Path]           # For user input (can be str or Path)

# Constants - Defaults
DEFAULT_MAX_TOKENS: Final[int] = 500
DEFAULT_FOLDER: Final[Path] = Path.home() / "Pictures" / "Processed"
DEFAULT_HTML_OUTPUT: Final[Path] = Path("results.html")
DEFAULT_TEMPERATURE: Final[float] = 0.0 # Default sampling temperature

# Constants - EXIF
IMPORTANT_EXIF_TAGS: Final[frozenset[str]] = frozenset({
    "DateTimeOriginal", "ImageDescription", "CreateDate",
    "Make", "Model", "LensModel", "ExposureTime",
    "FNumber", "ISOSpeedRatings", "FocalLength", "ExposureProgram",
})

DATE_FORMATS: Final[Tuple[str, ...]] = (
    "%Y:%m:%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y%m%d",
)

EXIF_DATE_TAGS: Final[Tuple[str, ...]] = (
    "DateTimeOriginal",
    "CreateDate",
    "DateTime",
)

# GPS Coordinate Tags
GPS_LAT_REF_TAG: Final[int] = 1
GPS_LAT_TAG: Final[int] = 2
GPS_LON_REF_TAG: Final[int] = 3
GPS_LON_TAG: Final[int] = 4

# Type definitions

class MemoryStats(NamedTuple):
    """Memory statistics container (values represent deltas or peak)."""
    active: float    # Change in active memory (MB) during processing
    cached: float    # Change in cached memory (MB) during processing
    peak: float      # Peak memory usage (MB) during processing
    time: float      # Processing time (seconds)

    @staticmethod
    def zero() -> 'MemoryStats':
        """Create a zero-initialized MemoryStats object."""
        return MemoryStats(active=0.0, cached=0.0, peak=0.0, time=0.0)

@dataclass(frozen=True)
class ModelResult:
    """Container for model processing results."""
    model_name: str
    output: str
    stats: MemoryStats
    error_stage: Optional[str] = None # Track stage where error occurred ('load', 'prompt', 'generate')

# --- Memory Management (Integrated into process_image_with_model) ---

# --- File Handling ---

def find_most_recent_file(folder: Path) -> Optional[Path]:
    """Return the Path of the most recently modified file in the folder."""
    if not folder.is_dir():
        logger.error(Colors.colored(f"Provided path is not a directory: {folder}", Colors.RED))
        return None
    try:
        files: List[Path] = [
            f for f in folder.iterdir()
            if f.is_file() and not f.name.startswith(".")
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
    """Extract EXIF data from an image file."""
    # Docstring remains mostly the same, implementation unchanged
    try:
        img: Image.Image
        with Image.open(image_path) as img:
            exif_raw: Any = img.getexif()
            if not exif_raw:
                logger.debug(f"No EXIF data found in {image_path}")
                return None

            exif_decoded: ExifDict = {}
            for tag_id, value in exif_raw.items():
                tag_name: Union[str, int] = TAGS.get(tag_id, tag_id)
                exif_decoded[tag_name] = value

            try:
                 gps_ifd: Optional[Dict[int, Any]] = exif_raw.get_ifd(Image.Exif.IFD.GPSInfo)
                 if gps_ifd:
                    gps_decoded: Dict[Union[str, int], Any] = {}
                    for gps_tag_id, gps_value in gps_ifd.items():
                         gps_tag_name: Union[str, int] = GPSTAGS.get(gps_tag_id, gps_tag_id)
                         gps_decoded[gps_tag_name] = gps_value
                    exif_decoded["GPSInfo"] = gps_decoded
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
    # Implementation unchanged
    if not isinstance(date_str_input, str):
        try:
            date_str: str = str(date_str_input)
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
            continue
    logger.debug(f"Could not parse date string '{date_str}' with known formats.")
    return None

GPSTupleElement = Any
GPSTuple = Tuple[GPSTupleElement, GPSTupleElement, GPSTupleElement]

def _convert_gps_coordinate(
    ref: Optional[Union[str, bytes]],
    coord_tuple: Optional[GPSTuple]
) -> Optional[float]:
    """
    Convert EXIF GPS coordinate tuple (Degrees, Minutes, Seconds) to decimal degrees.

    Handles raw EXIF GPS tags which might include byte strings for reference
    and tuples containing numeric types or PIL Rational objects for coordinates.

    Args:
        ref: The reference direction ('N', 'S', 'E', 'W'), possibly as bytes.
        coord_tuple: A tuple containing three elements representing
                     Degrees, Minutes, and Seconds. Elements can be int, float,
                     or rational-like objects (having .numerator, .denominator).

    Returns:
        The coordinate as a signed float (decimal degrees), or None if conversion fails
        due to invalid input format, invalid reference, or calculation errors.
        South latitudes and West longitudes are returned as negative values.

    Raises:
        ValueError: Internally raised if a coordinate component cannot be
                    converted to float (e.g., non-numeric type without
                    numerator/denominator), caught and logged as a warning.
    """
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
                      return float(num) # Or raise error? For now, return numerator.
                 return float(num) / float(den)
             elif isinstance(item, (int, float)):
                 return float(item)
             else:
                  # Try direct conversion, raise if it fails fundamentally
                  try:
                       return float(item)
                  except (TypeError, ValueError) as e_conv:
                       logger.warning(f"Could not convert GPS component '{item}' to float: {e_conv}")
                       raise ValueError("Invalid GPS component type") from e_conv

        degrees: float = get_float_val(coord_tuple[0])
        minutes: float = get_float_val(coord_tuple[1])
        seconds: float = get_float_val(coord_tuple[2])
        decimal: float = degrees + (minutes / 60.0) + (seconds / 3600.0)

        if ref_upper in ['S', 'W']:
            decimal = -decimal
        return decimal

    except ValueError as ve: # Catch the specific error from get_float_val
        logger.warning(Colors.colored(f"Failed GPS conversion due to invalid component type: {ve}", Colors.YELLOW))
        return None
    except (AttributeError, IndexError, TypeError, ZeroDivisionError, UnicodeDecodeError) as e:
        # Catch other potential errors during processing
        logger.warning(Colors.colored(f"Error converting GPS coordinate component: {type(e).__name__} {e} (Ref: {ref}, Coords: {coord_tuple})", Colors.YELLOW))
        return None

def extract_image_metadata(image_path: Path) -> MetadataDict:
    """Extract key metadata (Date, Description, GPS) from image EXIF."""
    # Implementation unchanged
    metadata: MetadataDict = {"date": "Unknown date", "description": "No description", "gps": "Unknown location"}
    exif: Optional[ExifDict] = get_exif_data(image_path)
    if not exif:
        logger.debug(f"No EXIF data found for {image_path}, returning default metadata.")
        return metadata

    found_date: bool = False
    for tag in EXIF_DATE_TAGS:
        date_val: ExifValue = exif.get(tag)
        if date_val is not None:
            formatted_date: Optional[str] = _format_exif_date(date_val)
            if formatted_date:
                metadata["date"] = formatted_date
                logger.debug(f"Found and formatted date from tag '{tag}': {formatted_date}")
                found_date = True
                break
            else:
                 logger.debug(f"Date tag '{tag}' ('{date_val}') found but could not be formatted.")
    if not found_date:
         logger.debug(f"Could not find or format a suitable date tag in {EXIF_DATE_TAGS}")

    desc_val: ExifValue = exif.get("ImageDescription")
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

    gps_info_val: ExifValue = exif.get("GPSInfo")
    if isinstance(gps_info_val, dict):
        logger.debug(f"Raw GPS Info dictionary: {gps_info_val}")
        lat_ref: Optional[Union[str, bytes]] = gps_info_val.get("GPSLatitudeRef")
        lat_coord_raw: Any = gps_info_val.get("GPSLatitude")
        lon_ref: Optional[Union[str, bytes]] = gps_info_val.get("GPSLongitudeRef")
        lon_coord_raw: Any = gps_info_val.get("GPSLongitude")

        lat_coord: Optional[GPSTuple] = lat_coord_raw if isinstance(lat_coord_raw, tuple) and len(lat_coord_raw) == 3 else None
        lon_coord: Optional[GPSTuple] = lon_coord_raw if isinstance(lon_coord_raw, tuple) and len(lon_coord_raw) == 3 else None

        lat: Optional[float] = _convert_gps_coordinate(lat_ref, lat_coord)
        lon: Optional[float] = _convert_gps_coordinate(lon_ref, lon_coord)

        if lat is not None and lon is not None:
            metadata["gps"] = f"{lat:+.6f}, {lon:+.6f}"
            logger.debug(f"Calculated GPS Coordinates: {metadata['gps']}")
        else:
            logger.debug("Could not calculate valid GPS coordinates from available tags.")
            if lat_coord is None or lon_coord is None:
                 logger.debug(f"GPS coordinate tuples were invalid or missing. Lat raw: {lat_coord_raw}, Lon raw: {lon_coord_raw}")

    elif gps_info_val is not None:
         logger.warning(f"GPSInfo tag found but was not a dictionary (type: {type(gps_info_val)}). Value: {gps_info_val}")
    else:
         logger.debug("No GPSInfo tag found in EXIF data.")

    return metadata

def pretty_print_exif(exif: ExifDict, verbose: bool = False) -> None:
    """Pretty print key EXIF data in a formatted table, using colors."""
    # Implementation unchanged
    if not exif:
        print("No EXIF data available.")
        return

    print(f"\n--- {Colors.colored('Key EXIF Data', Colors.CYAN)} ---")
    tags_to_print: List[Tuple[str, str, bool]] = []
    for tag, value in exif.items():
        tag_str: str = str(tag)

        if tag_str == "GPSInfo" and isinstance(value, dict):
            continue
        if isinstance(value, dict):
             logger.debug(f"Skipping dictionary value for EXIF tag '{tag_str}' in pretty print.")
             continue

        value_str: str
        if isinstance(value, bytes):
             try:
                 decoded_str: str = value.decode('utf-8', errors='replace')
                 value_str = decoded_str[:57] + "..." if len(decoded_str) > 60 else decoded_str
             except Exception:
                 value_str = f"<bytes len={len(value)}>"
        elif isinstance(value, tuple) and len(value) > 10:
             value_str = f"<tuple len={len(value)}>"
        else:
             try:
                  value_str = str(value)
             except Exception:
                  value_str = f"<unrepresentable type: {type(value).__name__}>"

        if len(value_str) > 60:
            value_str = value_str[:57] + "..."

        is_important: bool = tag_str in IMPORTANT_EXIF_TAGS
        if verbose or is_important:
             tags_to_print.append((tag_str, value_str, is_important))

    if not tags_to_print:
        print("No relevant EXIF tags found to display based on current settings.")
        return

    tags_to_print.sort(key=lambda x: x[0])

    max_tag_len: int = max(len(t[0]) for t in tags_to_print) if tags_to_print else 20
    max_val_len: int = max(len(t[1]) for t in tags_to_print) if tags_to_print else 40
    min_width: int = 10
    max_tag_len = max(max_tag_len, min_width)
    max_val_len = max(max_val_len, min_width + 5)

    bold_len: int = len(Colors.BOLD) + len(Colors.RESET) # Length adjustment needed for bolded text

    header_color = Colors.BLUE
    border_color = Colors.BLUE
    important_color = Colors.YELLOW

    print(Colors.colored(f"╔{'═' * (max_tag_len + 2)}╤{'═' * (max_val_len + 2)}╗", border_color))
    print(f"{Colors.colored('║', border_color)} {Colors.colored('Tag'.ljust(max_tag_len), header_color)} {Colors.colored('│', border_color)} {Colors.colored('Value'.ljust(max_val_len), header_color)} {Colors.colored('║', border_color)}")
    print(Colors.colored(f"╠{'═' * (max_tag_len + 2)}╪{'═' * (max_val_len + 2)}╣", border_color))

    for tag_name, value_display, is_important_tag in tags_to_print:
        tag_display: str
        tag_padding: int
        if is_important_tag:
            tag_display = Colors.colored(tag_name, Colors.BOLD + important_color)
            tag_padding = max_tag_len + bold_len
        else:
            tag_display = tag_name
            tag_padding = max_tag_len

        print(f"{Colors.colored('║', border_color)} {tag_display.ljust(tag_padding)} {Colors.colored('│', border_color)} {value_display.ljust(max_val_len)} {Colors.colored('║', border_color)}")

    print(Colors.colored(f"╚{'═' * (max_tag_len + 2)}╧{'═' * (max_val_len + 2)}╝", border_color))


# --- Model Handling ---

def get_cached_model_ids() -> List[str]:
    """Get list of model repo IDs from the huggingface cache."""
    # Implementation unchanged
    try:
        cache_info: HFCacheInfo = scan_cache_dir()
        model_ids: List[str] = sorted([repo.repo_id for repo in cache_info.repos])
        logger.debug(f"Found {len(model_ids)} potential models in cache: {model_ids}")
        return model_ids
    except HFValidationError:
         logger.error(Colors.colored("Hugging Face cache directory seems invalid or inaccessible.", Colors.RED))
         return []
    except FileNotFoundError:
        logger.error(Colors.colored("Hugging Face cache directory not found. Is huggingface_hub configured?", Colors.RED))
        return []
    except Exception as e:
        logger.error(Colors.colored(f"Unexpected error scanning Hugging Face cache: {type(e).__name__}: {e}", Colors.RED), exc_info=logger.level <= logging.DEBUG)
        return []


def print_model_stats(results: List[ModelResult]) -> None:
    """Print a table summarizing model performance statistics to the console."""
    # Implementation unchanged
    if not results:
        print("No model results to display.")
        return

    results.sort(key=lambda x: x.stats.time)
    display_names: List[str] = [r.model_name.split('/')[-1] for r in results]
    max_name_len: int = max(len(name) for name in display_names) if display_names else 20
    max_name_len = min(max_name_len, 44)
    col_width: int = 11

    header_color = Colors.BLUE
    border_color = Colors.BLUE
    summary_color = Colors.YELLOW

    print(f"\n--- {Colors.colored('Model Performance Summary (Console)', Colors.CYAN)} ---")
    print(Colors.colored(f"╔{'═' * max_name_len}═╦{'═' * col_width}═╦{'═' * col_width}═╦{'═' * col_width}═╦{'═' * col_width}═╗", border_color))
    title_line: str = (f"║ {Colors.colored('Model'.ljust(max_name_len), header_color)} │ "
                      f"{Colors.colored('Active Δ'.rjust(col_width), header_color)} │ "
                      f"{Colors.colored('Cache Δ'.rjust(col_width), header_color)} │ "
                      f"{Colors.colored('Peak Mem'.rjust(col_width), header_color)} │ "
                      f"{Colors.colored('Time'.rjust(col_width), header_color)} ║")
    print(f"{Colors.colored('║', border_color)}{title_line[1:-1]}{Colors.colored('║', border_color)}")
    print(Colors.colored(f"╠{'═' * max_name_len}═╬{'═' * col_width}═╬{'═' * col_width}═╬{'═' * col_width}═╬{'═' * col_width}═╣", border_color))

    for i, result in enumerate(results):
        model_disp_name: str = display_names[i]
        if len(model_disp_name) > max_name_len:
            model_disp_name = model_disp_name[:max_name_len-3] + "..."

        active_str: str = f"{result.stats.active:,.1f} MB"
        cache_str: str = f"{result.stats.cached:,.1f} MB"
        peak_str: str = f"{result.stats.peak:,.1f} MB"
        time_str: str = f"{result.stats.time:.2f} s"

        print(
            f"{Colors.colored('║', border_color)} {model_disp_name.ljust(max_name_len)} │ "
            f"{active_str.rjust(col_width)} │ "
            f"{cache_str.rjust(col_width)} │ "
            f"{peak_str.rjust(col_width)} │ "
            f"{time_str.rjust(col_width)} {Colors.colored('║', border_color)}"
        )

    if len(results) > 1:
        avg_active: float = sum(r.stats.active for r in results) / len(results)
        avg_cache: float = sum(r.stats.cached for r in results) / len(results)
        max_peak: float = max(r.stats.peak for r in results)
        avg_time: float = sum(r.stats.time for r in results) / len(results)

        avg_active_str: str = f"{avg_active:.1f} MB"
        avg_cache_str: str = f"{avg_cache:.1f} MB"
        max_peak_str: str = f"{max_peak:.1f} MB"
        avg_time_str: str = f"{avg_time:.2f} s"

        print(Colors.colored(f"╠{'═' * max_name_len}═╬{'═' * col_width}═╬{'═' * col_width}═╬{'═' * col_width}═╬{'═' * col_width}═╣", border_color))
        summary_line_content: str = (f"║ {Colors.colored('AVERAGE / MAX PEAK'.ljust(max_name_len), summary_color)} │ "
                      f"{Colors.colored(avg_active_str.rjust(col_width), summary_color)} │ "
                      f"{Colors.colored(avg_cache_str.rjust(col_width), summary_color)} │ "
                      f"{Colors.colored(max_peak_str.rjust(col_width), summary_color)} │ "
                      f"{Colors.colored(avg_time_str.rjust(col_width), summary_color)} ║")
        print(f"{Colors.colored('║', border_color)}{summary_line_content[1:-1]}{Colors.colored('║', border_color)}")

    print(Colors.colored(f"╚{'═' * max_name_len}═╩{'═' * col_width}═╩{'═' * col_width}═╩{'═' * col_width}═╩{'═' * col_width}═╝", border_color))


# --- HTML Report Generation ---

def generate_html_report(results: List[ModelResult], filename: Path) -> None:
    """Generates an HTML file with a table of model performance statistics and output."""
    # Implementation unchanged (already corrected in previous step)
    if not results:
        logger.warning("No results to generate HTML report.")
        return

    results.sort(key=lambda x: x.stats.time)

    html_start: str = """
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
        th, td { border: 1px solid #dee2e6; padding: 12px 15px; text-align: left; vertical-align: top; } /* Align top for output cell */
        th { background-color: #e9ecef; font-weight: 600; color: #495057; position: sticky; top: 0; z-index: 1; } /* Sticky header */
        tr:nth-child(even) { background-color: #f8f9fa; }
        tr:hover { background-color: #e2e6ea; }
        td.numeric, th.numeric { text-align: right; font-variant-numeric: tabular-nums; }
        .summary td { font-weight: bold; background-color: #d1ecf1; color: #0c5460; border-color: #bee5eb; }
        caption { caption-side: bottom; padding: 15px; font-style: italic; color: #6c757d; text-align: center; }
        .model-name { font-family: 'Courier New', Courier, monospace; font-weight: 500; }
        .model-output { white-space: pre-wrap; word-wrap: break-word; max-width: 600px; font-size: 0.9em; } /* Wrap output, limit width, slightly smaller font */
    </style>
</head>
<body>
    <h1>Model Performance Summary</h1>
    <table>
        <caption>Performance metrics and output for Vision Language Model processing. Generated on """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """. Active/Cache Δ = Change in MB. Peak Mem = Peak usage in MB. Time in seconds.</caption>
        <!-- Table Header -->
        <thead>
            <tr>
                <th>Model</th>
                <th class="numeric">Active Δ (MB)</th>
                <th class="numeric">Cache Δ (MB)</th>
                <th class="numeric">Peak Mem (MB)</th>
                <th class="numeric">Time (s)</th>
                <th>Generated Output</th> <!-- Column for model output -->
            </tr>
        </thead>
        <!-- Table Body: Data rows will be inserted here -->
        <tbody>
"""
    html_rows: str = ""
    for result in results:
        model_disp_name: str = html.escape(result.model_name)
        escaped_output: str = html.escape(result.output)
        html_rows += f"""
            <!-- Data row for model: {model_disp_name} -->
            <tr>
                <td class="model-name">{model_disp_name}</td>
                <td class="numeric">{result.stats.active:,.1f}</td>
                <td class="numeric">{result.stats.cached:,.1f}</td>
                <td class="numeric">{result.stats.peak:,.1f}</td>
                <td class="numeric">{result.stats.time:.2f}</td>
                <td class="model-output">{escaped_output}</td> <!-- Escaped model output -->
            </tr>
"""
    html_summary_row: str = ""
    if len(results) > 1:
        avg_active: float = sum(r.stats.active for r in results) / len(results)
        avg_cache: float = sum(r.stats.cached for r in results) / len(results)
        max_peak: float = max(r.stats.peak for r in results)
        avg_time: float = sum(r.stats.time for r in results) / len(results)
        html_summary_row = f"""
            <!-- Summary Row -->
            <tr class="summary">
                <td>AVERAGE / MAX PEAK</td>
                <td class="numeric">{avg_active:.1f}</td>
                <td class="numeric">{avg_cache:.1f}</td>
                <td class="numeric">{max_peak:.1f}</td>
                <td class="numeric">{avg_time:.2f}</td>
                <td></td> <!-- Empty cell for output column in summary -->
            </tr>
"""

    html_end: str = """
        </tbody>
    </table>
    <!-- End of Table -->
</body>
</html>
"""
    html_content: str = html_start + html_rows + html_summary_row + html_end

    try:
        f: IO[str]
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
    trust_remote_code: bool, # New parameter
    temperature: float       # New parameter
) -> Optional[ModelResult]:
    """
    Loads and runs a Vision Language Model (VLM) for image analysis, tracking performance.

    Handles loading the model and tokenizer, preparing the prompt, generating text
    based on the image and prompt, and measuring memory usage (delta active, delta cache,
    peak) and execution time specific to this model's processing run.

    Args:
        model_identifier: The Hugging Face identifier or local path for the VLM.
        image_path: Path to the input image file.
        prompt: The text prompt to use with the VLM.
        max_tokens: Maximum number of new tokens to generate.
        vlm_verbose: If True, enable verbose output from the `mlx_vlm.generate` function.
        trust_remote_code: If True, allow loading models with custom code from Hugging Face Hub.
                           This is required for many VLMs but carries security risks.
        temperature: Sampling temperature for text generation (0.0 for deterministic).

    Returns:
        A ModelResult object containing the model name, generated output string,
        and MemoryStats, if processing succeeds.
        Returns None if any stage (loading, prompt preparation, generation) fails.
        Error details and stage are logged.
    """
    model_short_name = model_identifier.split('/')[-1]
    logger.info(f"Processing '{image_path.name}' with model: {Colors.colored(model_short_name, Colors.MAGENTA)}")

    model: Any = None
    tokenizer: Any = None
    config: Any = None
    output: Optional[str] = None
    error_stage: Optional[str] = None # Track failure stage

    try:
        # --- Memory & Time Tracking Initialization ---
        mx.clear_cache()
        mx.reset_peak_memory()
        initial_active_mb = mx.get_active_memory() / 1e6
        initial_cache_mb = mx.get_cache_memory() / 1e6
        start_time = time.perf_counter()

        # --- Stage 1: Load Model and Config ---
        error_stage = 'load'
        load_start_time = time.perf_counter()
        logger.debug(f"Loading {model_short_name}...")
        # Use trust_remote_code flag here
        model, tokenizer = load(model_identifier, trust_remote_code=trust_remote_code)
        config = load_config(model_identifier, trust_remote_code=trust_remote_code)
        logger.debug(f"Loaded {model_short_name} in {time.perf_counter() - load_start_time:.2f}s")

        # --- Stage 2: Prepare Prompt ---
        error_stage = 'prompt'
        logger.debug(f"Applying chat template for {model_short_name}...")
        formatted_prompt: str = apply_chat_template(tokenizer, config, prompt, num_images=1)
        logger.debug(f"Using formatted prompt for {model_short_name} (first 100 chars): {formatted_prompt[:100]}...")

        # --- Stage 3: Generate Text ---
        error_stage = 'generate'
        gen_start_time = time.perf_counter()
        logger.debug(f"Generating text with {model_short_name} (temp={temperature})...")
        output = generate(
            model=model,
            processor=tokenizer,
            prompt=formatted_prompt,
            image_paths=[str(image_path)],
            max_tokens=max_tokens,
            verbose=vlm_verbose,
            temp=temperature # Use temperature parameter
        )
        logger.debug(f"Generated output from {model_short_name} in {time.perf_counter() - gen_start_time:.2f}s")

        # --- Finalize Measurements ---
        error_stage = 'measure' # Should ideally not fail here, but track anyway
        mx.eval(model.parameters()) # Ensure calculations finish

        end_time = time.perf_counter()
        final_active_mb = mx.get_active_memory() / 1e6
        final_cache_mb = mx.get_cache_memory() / 1e6
        peak_mb = mx.get_peak_memory() / 1e6

        elapsed_time = end_time - start_time
        active_delta_mb = final_active_mb - initial_active_mb
        cache_delta_mb = final_cache_mb - initial_cache_mb

        final_stats = MemoryStats(
            active=active_delta_mb,
            cached=cache_delta_mb,
            peak=peak_mb,
            time=elapsed_time
        )
        error_stage = None # Success

        logger.info(f"Finished processing {Colors.colored(model_short_name, Colors.MAGENTA)} in {Colors.colored(f'{elapsed_time:.1f}s', Colors.GREEN)}. Peak Mem: {Colors.colored(f'{peak_mb:.1f}MB', Colors.CYAN)}")
        return ModelResult(
            model_name=model_identifier,
            output=str(output) if output is not None else "",
            stats=final_stats
        )

    except Exception as e:
        # Log failure with stage information
        failure_location = f" during {error_stage}" if error_stage else ""
        logger.error(Colors.colored(f"Failed to process model {model_identifier}{failure_location}: {type(e).__name__}: {e}", Colors.RED))
        if logger.isEnabledFor(logging.DEBUG):
            traceback.print_exc()
        # Return None to indicate failure for this model
        return None
    finally:
        # --- Cleanup ---
        del model, tokenizer, config # Use tuple deletion
        mx.clear_cache()
        # Peak memory is reset at the start of the next call


# --- Main Execution ---

def main(args: argparse.Namespace) -> None:
    """Main function to orchestrate image analysis."""
    # Configure logging level based on args
    # Needs to be done before any logging happens if using module level logger
    logging.basicConfig(
        level=logging.DEBUG if args.debug else (logging.INFO if args.verbose else logging.INFO),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stderr)],
        force=True # Force reconfig if already configured by libraries
    )

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
        logger.warning(Colors.colored("`--trust-remote-code` is enabled. Models from the Hub can execute arbitrary code on your machine.", Colors.YELLOW))
        logger.warning(Colors.colored("Only use this option with models from sources you trust.", Colors.YELLOW))
        logger.warning(Colors.colored("-----------------------", Colors.YELLOW + Colors.BOLD))


    overall_start_time: float = time.perf_counter()

    # --- 1. Find Image ---
    folder_path: Path = args.folder.resolve()
    logger.info(f"Scanning folder: {Colors.colored(str(folder_path), Colors.BLUE)}")
    if args.folder == DEFAULT_FOLDER and not DEFAULT_FOLDER.is_dir():
         print(Colors.colored(f"Warning: Default folder '{DEFAULT_FOLDER}' does not exist.", Colors.YELLOW), file=sys.stderr)

    image_path: Optional[Path] = find_most_recent_file(folder_path)
    if not image_path:
        print(Colors.colored(f"\nError: Could not find a suitable image file in {folder_path}.", Colors.RED), file=sys.stderr)
        sys.exit(1)

    resolved_image_path: Path = image_path.resolve()
    print(f"\nProcessing file: {Colors.colored(resolved_image_path.name, Colors.MAGENTA)} (located at {resolved_image_path})")
    print_image_dimensions(resolved_image_path)

    # --- 2. Extract Metadata ---
    metadata: MetadataDict = extract_image_metadata(resolved_image_path)
    print(f"  Date: {Colors.colored(metadata.get('date', 'N/A'), Colors.CYAN)}")
    print(f"  Desc: {Colors.colored(metadata.get('description', 'N/A'), Colors.CYAN)}")
    print(f"  GPS:  {Colors.colored(metadata.get('gps', 'N/A'), Colors.CYAN)}")

    if args.verbose or args.debug:
         exif_data: Optional[ExifDict] = get_exif_data(resolved_image_path)
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
        prompt_parts: List[str] = [
            "Provide a factual caption, a brief description, and relevant comma-separated keywords/tags for this image.",
            "The goal is easy cataloguing and searching.",
            f"Context: The picture might relate to '{metadata['description']}'" if metadata.get('description') and metadata['description'] != "No description" else "",
            f"taken around {metadata['date']}" if metadata.get('date') and metadata['date'] != "Unknown date" else "",
            f"near GPS coordinates {metadata['gps']}." if metadata.get('gps') and metadata['gps'] != "Unknown location" else "",
            "Focus on the visual content. Avoid repeating the date or exact GPS coordinates unless visually evident in the image (e.g., a sign)."
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
        msg = ("\nWarning: No models specified via --models and none found in Hugging Face cache." if not args.models
               else "\nWarning: No models specified via --models argument.")
        print(Colors.colored(msg, Colors.YELLOW), file=sys.stderr)
        if not args.models:
            print("Please ensure models are downloaded and the cache is accessible.", file=sys.stderr)
    else:
        print(f"\nProcessing {Colors.colored(str(len(model_identifiers)), Colors.GREEN)} model(s)...")
        separator: str = Colors.colored(f"\n{'=' * 80}\n", Colors.BLUE)

        for model_id in model_identifiers:
            print(separator)
            is_vlm_verbose: bool = args.verbose or args.debug
            result: Optional[ModelResult] = process_image_with_model(
                model_identifier=model_id,
                image_path=resolved_image_path,
                prompt=prompt,
                max_tokens=args.max_tokens,
                vlm_verbose=is_vlm_verbose,
                trust_remote_code=args.trust_remote_code, # Pass flag
                temperature=args.temperature             # Pass temperature
            )
            if result:
                print(f"\n--- {Colors.colored(f'Output from {model_id.split('/')[-1]}', Colors.GREEN)} ---")
                print(result.output)
                results.append(result)
            else:
                # Failure message now includes stage if known
                # The detailed logging happens inside process_image_with_model
                print(f"--- {Colors.colored(f'Failed to process model: {model_id}', Colors.RED)} (Check logs above for details/stage) ---")


    # --- 5. Print Summary Statistics ---
    if results:
        print(Colors.colored(f"\n{'=' * 80}\n", Colors.BLUE))
        print_model_stats(results)
    else:
        print(Colors.colored("\nNo models were successfully processed. No performance summary generated.", Colors.YELLOW))

    # --- 6. Generate HTML Report ---
    html_output_path: Path = args.output_html.resolve()
    if results:
        generate_html_report(results, html_output_path)
    else:
        logger.info(f"Skipping HTML report generation to {html_output_path} as no models were processed.")

    # --- Calculate and Print Total Time ---
    overall_time: float = time.perf_counter() - overall_start_time
    print(f"\nTotal execution time: {Colors.colored(f'{overall_time:.2f} seconds', Colors.GREEN)}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze the most recent image in a folder using cached MLX Vision Language Models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Input/Output Arguments
    parser.add_argument("-f", "--folder", type=Path, default=DEFAULT_FOLDER, help="Folder to scan for the most recently modified image.")
    parser.add_argument("--output-html", type=Path, default=DEFAULT_HTML_OUTPUT, help="Output HTML file path for model performance statistics.")

    # Model Selection Arguments
    parser.add_argument("--models", nargs='+', type=str, default=None, help="Specify specific model IDs (repo_id or path) to run. If omitted, scans Hugging Face cache.")
    # Use BooleanOptionalAction for a --flag / --no-flag style argument
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True, help="Allow loading models with custom code from HuggingFace Hub (SECURITY RISK - use only trusted models).")

    # Generation Arguments
    parser.add_argument("-p", "--prompt", type=str, default=None, help="Custom prompt for the VLM. If omitted, a default prompt is generated.")
    parser.add_argument("-m", "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Maximum number of new tokens for the VLM to generate.")
    parser.add_argument("-t", "--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature for generation (0.0 for deterministic, default is {DEFAULT_TEMPERATURE}).")

    # Control Arguments
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output (INFO logging), VLM steps, detailed EXIF.")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logging (DEBUG level), implies verbose output.")

    parsed_args: argparse.Namespace = parser.parse_args()

    # --- Main Execution ---
    try:
        main(parsed_args)
    except Exception as main_err:
         # Log final unhandled exceptions with color
         logger.exception(Colors.colored(f"An unexpected error occurred during main execution: {main_err}", Colors.RED))
         sys.exit(1)
