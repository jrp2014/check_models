#!/usr/bin/env python3
"""Image analysis and caption generation using MLX Vision Language Models."""

# Standard library imports
import argparse
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import (Any, Callable, Dict, Final, List, Optional, NamedTuple,
                    Tuple, Union, IO) # Added IO, Text

# Third-party imports
try:
    import mlx.core as mx
    from PIL import Image, UnidentifiedImageError
    # Import specific EXIF related types if available/needed, otherwise use Any
    from PIL.ExifTags import GPSTAGS, TAGS # Keep these
    from mlx_vlm import (load, generate, __version__ as vlm_version)
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
    # Import huggingface_hub library
    from huggingface_hub import scan_cache_dir
    # Import specific types if needed, e.g., CacheInfo, CachedRepoInfo
    from huggingface_hub import HFCacheInfo
    from huggingface_hub.errors import HFValidationError # For error handling
except ImportError as e:
    print(f"Error importing required libraries: {e}", file=sys.stderr)
    print("Please ensure you have installed mlx, mlx-lm, Pillow, huggingface_hub, and mlx-vlm.", file=sys.stderr)
    sys.exit(1)


# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)

# Type aliases and definitions
# EXIF keys can be int (from raw exif) or str (after decoding)
ExifValue = Any # EXIF values can be complex (ints, strings, bytes, tuples, Ratio)
ExifDict = Dict[Union[str, int], ExifValue]
MetadataDict = Dict[str, str]         # Metadata results dictionary (string keys, string values)
PathLike = Union[str, Path]           # For user input (can be str or Path)

# Constants - Defaults
DEFAULT_MAX_TOKENS: Final[int] = 500
DEFAULT_FOLDER: Final[Path] = Path.home() / "Pictures" / "Processed" # More robust default
DEFAULT_HTML_OUTPUT: Final[Path] = Path("results.html")

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

# GPS Coordinate Tags (Integer values from Exif standard)
# These constants are not directly used in type hints but define keys
GPS_LAT_REF_TAG: Final[int] = 1
GPS_LAT_TAG: Final[int] = 2
GPS_LON_REF_TAG: Final[int] = 3
GPS_LON_TAG: Final[int] = 4

# Type definitions

class MemoryStats(NamedTuple):
    """Memory statistics container."""
    active: float    # Current active memory in MB
    cached: float    # Current cached memory in MB
    peak: float      # Peak memory usage in MB
    time: float      # Processing time in seconds

    @staticmethod
    def zero() -> 'MemoryStats':
        """Create a zero-initialized MemoryStats object."""
        return MemoryStats(active=0.0, cached=0.0, peak=0.0, time=0.0)

    @staticmethod
    def from_current() -> 'MemoryStats':
        """Create MemoryStats from current MLX memory state."""
        return MemoryStats(
            active=mx.get_active_memory() / 1024 / 1024,
            cached=mx.get_cache_memory() / 1024 / 1024,
            peak=mx.get_peak_memory() / 1024 / 1024,
            time=0.0
        )

    def with_time(self, elapsed: float) -> 'MemoryStats':
        """Return new MemoryStats with updated time."""
        return MemoryStats(
            active=self.active,
            cached=self.cached,
            peak=self.peak,
            time=elapsed
        )

@dataclass(frozen=True)
class ModelResult:
    """Container for model processing results."""
    model_name: str
    output: str
    stats: MemoryStats

# --- Memory Management ---

def track_memory_and_time(func: Callable[[], Any]) -> Tuple[Any, MemoryStats]:
    """
    Tracks MLX memory usage and timing for a given function execution.

    Args:
        func: The function to execute and measure. It should take no arguments.

    Returns:
        A tuple containing the result of the function and a MemoryStats object.
        Returns (None, MemoryStats.zero()) if the function raises an exception.
    """
    # Clear cache and reset peak memory *before* the operation
    mx.clear_cache()
    mx.reset_peak_memory()
    # Get initial memory state
    initial_stats: MemoryStats = MemoryStats.from_current()
    start_time: float = time.perf_counter()

    result: Any = None
    stats: MemoryStats = MemoryStats.zero()

    try:
        result = func() # Execute the code block

        # Ensure all computations are done for accurate memory measurement if result involves mx arrays
        # This part is heuristic; checks common MLX object patterns
        items_to_eval: List[Any] = []
        if isinstance(result, tuple):
            items_to_eval.extend(list(result))
        else:
            items_to_eval.append(result)

        for item in items_to_eval:
            if hasattr(item, 'parameters') and callable(getattr(item, 'parameters')):
                 # Assuming parameters() returns something mx.eval can handle (like dict/list of arrays)
                 mx.eval(item.parameters())
            elif isinstance(item, (list, dict)): # Eval if result is a collection of arrays
                 mx.eval(item)
            elif hasattr(item, 'tolist'): # Simple check for mx.array like objects
                 mx.eval(item)

    except Exception as e:
        log.error(f"Exception during tracked execution: {e}", exc_info=log.level <= logging.DEBUG)
        # Return None result and zero stats on error
        return None, MemoryStats.zero()
    finally:
        # Calculate deltas and final stats *after* the operation
        end_time: float = time.perf_counter()
        elapsed_time: float = end_time - start_time
        final_stats: MemoryStats = MemoryStats.from_current().with_time(elapsed_time)

        stats = MemoryStats(
            active=final_stats.active - initial_stats.active,
            cached=final_stats.cached - initial_stats.cached,
            peak=final_stats.peak,
            time=final_stats.time
        )

        log.debug(
            f"Memory Tracking: Active Δ: {stats.active:.1f} MB, "
            f"Cache Δ: {stats.cached:.1f} MB, "
            f"Peak: {stats.peak:.1f} MB, "
            f"Time: {stats.time:.2f}s"
        )
        # Clean up *after* measurements are taken
        mx.clear_cache()
        # Peak memory is reset at the start, no need to reset here

    return result, stats


# --- File Handling ---

def find_most_recent_file(folder: Path) -> Optional[Path]:
    """Return the Path of the most recently modified file in the folder."""
    if not folder.is_dir():
        log.error(f"Provided path is not a directory: {folder}")
        return None
    try:
        # Filter for files, excluding hidden ones
        files: List[Path] = [
            f for f in folder.iterdir()
            if f.is_file() and not f.name.startswith(".")
        ]
        if not files:
            log.warning(f"No non-hidden files found in: {folder}")
            return None
        # Sort by modification time, newest first
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        most_recent: Path = files[0]
        log.debug(f"Most recent file found: {most_recent}")
        return most_recent
    except PermissionError:
        log.error(f"Permission denied accessing folder: {folder}")
        return None
    except OSError as e:
        log.error(f"OS error scanning folder {folder}: {e}")
        return None

def print_image_dimensions(image_path: Path) -> None:
    """Print the dimensions and megapixel count of the image."""
    try:
        with Image.open(image_path) as img:
            width: int
            height: int
            width, height = img.size
            mpx: float = (width * height) / 1_000_000
            print(f"Image dimensions: {width}x{height} ({mpx:.1f} MPixels)")
    except FileNotFoundError:
        log.error(f"Image file not found: {image_path}")
    except UnidentifiedImageError:
        log.error(f"Cannot identify image file (may be corrupt or wrong format): {image_path}")
    except Exception as e:
        log.error(f"Error reading image dimensions for {image_path}: {e}")

# --- EXIF & Metadata Handling ---

@lru_cache(maxsize=128) # Cache EXIF data per image path
def get_exif_data(image_path: Path) -> Optional[ExifDict]:
    """Extract EXIF data from an image file."""
    try:
        # Type hint for the image object
        img: Image.Image
        with Image.open(image_path) as img:
            # Use getexif() which is preferred over _getexif()
            # The return type of getexif() is complex, often Image.Exif
            exif_raw: Any = img.getexif() # Use Any or a more specific type if PIL provides one easily
            if not exif_raw:
                log.debug(f"No EXIF data found in {image_path}")
                return None

            # Decode tags using PIL.ExifTags.TAGS
            exif_decoded: ExifDict = {}
            tag_id: int
            value: ExifValue
            for tag_id, value in exif_raw.items():
                # TAGS maps int -> str
                tag_name: Union[str, int] = TAGS.get(tag_id, tag_id)
                exif_decoded[tag_name] = value

            # Decode GPS tags specifically
            try:
                 # get_ifd returns an IFD object (dict-like)
                 # Use the correct constant: Image.Exif.IFD.GPSInfo (34853)
                 gps_ifd: Optional[Dict[int, Any]] = exif_raw.get_ifd(Image.Exif.IFD.GPSInfo)
                 if gps_ifd:
                    # GPSTAGS maps int -> str
                    gps_decoded: Dict[Union[str, int], Any] = {}
                    gps_tag_id: int
                    gps_value: Any
                    for gps_tag_id, gps_value in gps_ifd.items():
                         gps_tag_name: Union[str, int] = GPSTAGS.get(gps_tag_id, gps_tag_id)
                         gps_decoded[gps_tag_name] = gps_value
                    # Store decoded GPS info under a consistent string key "GPSInfo"
                    exif_decoded["GPSInfo"] = gps_decoded
            except KeyError: # Handle case where GPS IFD doesn't exist
                 log.debug(f"GPSInfo IFD (key {Image.Exif.IFD.GPSInfo}) not found in {image_path}")
            except Exception as e_gps: # Catch other potential errors reading GPS IFD
                 log.warning(f"Could not read GPS IFD for {image_path}: {type(e_gps).__name__} {e_gps}")


            log.debug(f"Successfully extracted EXIF for {image_path}")
            return exif_decoded

    except FileNotFoundError:
        log.error(f"Image file not found for EXIF extraction: {image_path}")
        return None
    except UnidentifiedImageError:
        log.error(f"Cannot identify image file for EXIF: {image_path}")
        return None
    except AttributeError as e:
        # Can happen if PIL version is very old or image format lacks EXIF support
        log.warning(f"PIL lacks getexif/get_ifd method or image has no EXIF support ({e}): {image_path}")
        return None
    except Exception as e:
        log.error(f"Unexpected error reading EXIF from {image_path}: {e}", exc_info=log.level <= logging.DEBUG)
        return None

def _format_exif_date(date_str_input: Any) -> Optional[str]:
    """Attempt to parse and format a date string from EXIF. Handles non-string input."""
    if not isinstance(date_str_input, str):
        try:
            date_str: str = str(date_str_input)
        except Exception:
             log.debug(f"Could not convert potential date value '{date_str_input}' to string.")
             return None
    else:
        date_str = date_str_input

    fmt: str
    for fmt in DATE_FORMATS:
        try:
            dt_obj: datetime = datetime.strptime(date_str, fmt)
            return dt_obj.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError): # Catch errors during parsing
            continue # Try next format
    log.debug(f"Could not parse date string '{date_str}' with known formats.")
    return None

# Helper type for GPS coordinate tuple elements (can be complex, e.g., Ratio)
GPSTupleElement = Any # Using Any as PIL.TiffImagePlugin.Ratio is not easily typed
GPSTuple = Tuple[GPSTupleElement, GPSTupleElement, GPSTupleElement]

def _convert_gps_coordinate(
    ref: Optional[Union[str, bytes]],
    coord_tuple: Optional[GPSTuple] # Use the alias, expect 3 elements
) -> Optional[float]:
    """Convert EXIF GPS coordinate tuple (Degrees, Minutes, Seconds) to decimal degrees."""
    # Basic validation
    if ref is None or coord_tuple is None or len(coord_tuple) != 3:
        log.debug(f"Invalid input for GPS conversion: ref={ref}, coord_tuple={coord_tuple}")
        return None

    try:
        # Decode reference if bytes (e.g., b'N')
        ref_str: str
        if isinstance(ref, bytes):
            ref_str = ref.decode('ascii')
        elif isinstance(ref, str):
            ref_str = ref
        else:
            log.warning(f"Unexpected type for GPS reference: {type(ref)}")
            return None
        ref_upper: str = ref_str.upper()
        if ref_upper not in ['N', 'S', 'E', 'W']:
             log.warning(f"Invalid GPS reference value: {ref_str}")
             return None

        # Helper to extract float value from Ratio or numeric types
        def get_float_val(item: GPSTupleElement) -> float:
             # Check for PIL Ratio type (older PIL) or numbers.Rational (newer PIL/standard lib)
             # Check numerator/denominator attributes for rational-like objects
             if hasattr(item, 'numerator') and hasattr(item, 'denominator'):
                 num = item.numerator
                 den = item.denominator
                 if den == 0:
                      log.warning("GPS coordinate component has zero denominator.")
                      # Fallback or raise? Let's try returning numerator as float
                      return float(num)
                 return float(num) / float(den)
             # Handle direct float/int/numeric types
             elif isinstance(item, (int, float)):
                 return float(item)
             else:
                  # Attempt conversion, log warning if fails
                  try:
                       return float(item)
                  except (TypeError, ValueError) as e_conv:
                       log.warning(f"Could not convert GPS component '{item}' to float: {e_conv}")
                       raise ValueError("Invalid GPS component type") from e_conv


        # Extract degrees, minutes, seconds
        degrees: float = get_float_val(coord_tuple[0])
        minutes: float = get_float_val(coord_tuple[1])
        seconds: float = get_float_val(coord_tuple[2])

        # Calculate decimal degrees
        decimal: float = degrees + (minutes / 60.0) + (seconds / 3600.0)

        # Adjust sign based on reference (S/W are negative)
        if ref_upper in ['S', 'W']:
            decimal = -decimal

        return decimal

    except (AttributeError, IndexError, TypeError, ValueError, ZeroDivisionError, UnicodeDecodeError) as e:
        # Catch errors during conversion or calculation
        log.warning(f"Error converting GPS coordinate component: {e} (Ref: {ref}, Coords: {coord_tuple})")
        return None

def extract_image_metadata(image_path: Path) -> MetadataDict:
    """Extract key metadata (Date, Description, GPS) from image EXIF."""
    # Initialize with default values
    metadata: MetadataDict = {
        "date": "Unknown date",
        "description": "No description",
        "gps": "Unknown location",
    }
    # Get EXIF data using the cached function
    exif: Optional[ExifDict] = get_exif_data(image_path)
    if not exif:
        log.debug(f"No EXIF data found for {image_path}, returning default metadata.")
        return metadata # Return defaults if no EXIF

    # 1. Extract Date
    found_date: bool = False
    tag: str
    for tag in EXIF_DATE_TAGS:
        date_val: ExifValue = exif.get(tag)
        if date_val is not None: # Check explicitly for None
            formatted_date: Optional[str] = _format_exif_date(date_val) # Pass the raw value
            if formatted_date:
                metadata["date"] = formatted_date
                log.debug(f"Found and formatted date from tag '{tag}': {formatted_date}")
                found_date = True
                break # Stop after finding the first valid date
            else:
                 log.debug(f"Date tag '{tag}' ('{date_val}') found but could not be formatted.")
    if not found_date:
         log.debug(f"Could not find or format a suitable date tag in {EXIF_DATE_TAGS}")

    # 2. Extract Description
    desc_val: ExifValue = exif.get("ImageDescription")
    if desc_val is not None:
        try:
            desc_str: str
            # Decode if bytes (sometimes happens)
            if isinstance(desc_val, bytes):
                 # Use 'replace' for invalid bytes, or 'ignore'
                 desc_str = desc_val.decode('utf-8', errors='replace')
            else:
                 desc_str = str(desc_val) # Convert other types to string
            metadata["description"] = desc_str.strip()
            log.debug(f"Found description: '{metadata['description']}'")
        except Exception as e:
            log.warning(f"Could not convert description value '{desc_val}' to string: {e}")

    # 3. Extract GPS
    # GPSInfo should have been decoded into a dict by get_exif_data
    gps_info_val: ExifValue = exif.get("GPSInfo")
    if isinstance(gps_info_val, dict): # Check if it's the decoded dictionary
        log.debug(f"Raw GPS Info dictionary: {gps_info_val}")
        # Explicitly type hint the retrieved values
        lat_ref: Optional[Union[str, bytes]] = gps_info_val.get("GPSLatitudeRef")
        # Assume GPSLatitude/Longitude are tuples if they exist
        lat_coord_raw: Any = gps_info_val.get("GPSLatitude")
        lon_ref: Optional[Union[str, bytes]] = gps_info_val.get("GPSLongitudeRef")
        lon_coord_raw: Any = gps_info_val.get("GPSLongitude")

        # Validate coordinate tuples before passing
        lat_coord: Optional[GPSTuple] = lat_coord_raw if isinstance(lat_coord_raw, tuple) and len(lat_coord_raw) == 3 else None
        lon_coord: Optional[GPSTuple] = lon_coord_raw if isinstance(lon_coord_raw, tuple) and len(lon_coord_raw) == 3 else None

        # Convert coordinates
        lat: Optional[float] = _convert_gps_coordinate(lat_ref, lat_coord)
        lon: Optional[float] = _convert_gps_coordinate(lon_ref, lon_coord)

        if lat is not None and lon is not None:
            # Format to standard GPS string (latitude, longitude)
            metadata["gps"] = f"{lat:+.6f}, {lon:+.6f}"
            log.debug(f"Calculated GPS Coordinates: {metadata['gps']}")
        else:
            log.debug("Could not calculate valid GPS coordinates from available tags.")
            if lat_coord is None or lon_coord is None:
                 log.debug(f"GPS coordinate tuples were invalid or missing. Lat raw: {lat_coord_raw}, Lon raw: {lon_coord_raw}")

    elif gps_info_val is not None:
        # Log if GPSInfo exists but isn't the expected dictionary
         log.warning(f"GPSInfo tag found but was not a dictionary (type: {type(gps_info_val)}). Value: {gps_info_val}")
    else:
         # Log if GPSInfo tag is completely missing
         log.debug("No GPSInfo tag found in EXIF data.")

    return metadata

def pretty_print_exif(exif: ExifDict, verbose: bool = False) -> None:
    """Pretty print key EXIF data in a formatted table."""
    if not exif:
        print("No EXIF data available.")
        return

    print("\n--- Key EXIF Data ---")
    # Filter and sort tags
    # List of tuples: (tag_name_str, value_str, is_important_bool)
    tags_to_print: List[Tuple[str, str, bool]] = []
    tag: Union[str, int]
    value: ExifValue
    for tag, value in exif.items():
        tag_str: str = str(tag) # Ensure tag is string for checks and printing

        # Skip complex structures we handle elsewhere or don't want to print directly
        if tag_str == "GPSInfo" and isinstance(value, dict): # Skip the decoded GPS dict
             continue
        if isinstance(value, dict): # Skip other unexpected dictionaries
             log.debug(f"Skipping dictionary value for EXIF tag '{tag_str}' in pretty print.")
             continue

        # Format the value nicely for printing
        value_str: str
        if isinstance(value, bytes):
             try:
                 # Try decoding bytes, replace errors
                 decoded_str: str = value.decode('utf-8', errors='replace')
                 if len(decoded_str) > 60: # Limit length even if decodable
                      value_str = decoded_str[:57] + "..."
                 else:
                      value_str = decoded_str
             except Exception: # Catch potential decoding issues not handled by 'replace'
                 value_str = f"<bytes len={len(value)}>"
        elif isinstance(value, tuple) and len(value) > 10: # Skip long tuples likely raw binary data
             value_str = f"<tuple len={len(value)}>"
        else:
             # Convert other types to string safely
             try:
                  value_str = str(value)
             except Exception:
                  value_str = f"<unrepresentable type: {type(value).__name__}>"

        # Truncate long strings
        if len(value_str) > 60:
            value_str = value_str[:57] + "..."

        # Check if the tag is considered important
        is_important: bool = tag_str in IMPORTANT_EXIF_TAGS

        # Add to list if verbose mode is on OR if the tag is important
        if verbose or is_important:
             tags_to_print.append((tag_str, value_str, is_important))

    if not tags_to_print:
        print("No relevant EXIF tags found to display based on current settings.")
        return

    # Sort alphabetically by tag name
    tags_to_print.sort(key=lambda x: x[0])

    # Determine column widths dynamically
    max_tag_len: int = max(len(t[0]) for t in tags_to_print) if tags_to_print else 20
    max_val_len: int = max(len(t[1]) for t in tags_to_print) if tags_to_print else 40
    # Ensure minimum widths
    min_width: int = 10
    max_tag_len = max(max_tag_len, min_width)
    max_val_len = max(max_val_len, min_width + 5) # Value column slightly wider min

    # ANSI escape codes for bold text (if used)
    bold_start: str = "\033[1m"
    bold_end: str = "\033[0m"
    bold_len: int = len(bold_start) + len(bold_end) # Length adjustment needed for bolded text

    # --- Print Table ---
    # Helper function for creating table lines
    def print_table_line(left: str = "═", middle: str = "╤", right: str = "═") -> None:
        print(f"║{left * (max_tag_len + 2)}{middle}{right * (max_val_len + 2)}║")

    def print_table_separator(left: str = "═", middle: str = "╪", right: str = "═") -> None:
        print(f"╠{left * (max_tag_len + 2)}{middle}{right * (max_val_len + 2)}╣")

    # Print table header
    print(f"╔{'═' * (max_tag_len + 2)}╤{'═' * (max_val_len + 2)}╗")
    print(f"║ {'Tag'.ljust(max_tag_len)} │ {'Value'.ljust(max_val_len)} ║")
    print(f"╠{'═' * (max_tag_len + 2)}╪{'═' * (max_val_len + 2)}╣")

    # Print rows
    tag_name: str
    value_display: str
    is_important_tag: bool
    for tag_name, value_display, is_important_tag in tags_to_print:
        # Handle potential bolding and adjust padding accordingly
        # Note: Simple ljust might misalign if terminal doesn't handle ANSI codes well.
        # This version attempts to compensate, but might not be perfect everywhere.
        tag_display: str
        if is_important_tag:
            # Apply bold formatting
            tag_display = f"{bold_start}{tag_name}{bold_end}"
            # Adjust ljust width by the length of the ANSI codes
            tag_padding: int = max_tag_len + bold_len
        else:
            tag_display = tag_name
            tag_padding = max_tag_len

        # Print the formatted row
        print(f"║ {tag_display.ljust(tag_padding)} │ {value_display.ljust(max_val_len)} ║")


    # Print table footer
    print(f"╚{'═' * (max_tag_len + 2)}╧{'═' * (max_val_len + 2)}╝")


# --- Model Handling ---

def get_cached_model_ids() -> List[str]:
    """Get list of model repo IDs from the huggingface cache."""
    try:
        # scan_cache_dir returns CacheInfo object
        cache_info: HFCacheInfo = scan_cache_dir()
        # CacheInfo.repos is a list of RepoInfo objects
        model_ids: List[str] = sorted(repo.repo_id for repo in cache_info.repos)
        log.debug(f"Found {len(model_ids)} potential models in cache: {model_ids}")
        # Optional: Add filtering here based on repo properties if needed
        # e.g., filter by repo_type if available and relevant
        return model_ids
    except HFValidationError:
         log.error("Hugging Face cache directory seems invalid or inaccessible.")
         return []
    except FileNotFoundError:
        log.error("Hugging Face cache directory not found. Is huggingface_hub configured?")
        return []
    except Exception as e:
        log.error(f"Unexpected error scanning Hugging Face cache: {type(e).__name__}: {e}", exc_info=log.level <= logging.DEBUG)
        return []


def print_model_stats(results: List[ModelResult]) -> None:
    """Print a table summarizing model performance statistics to the console."""
    if not results:
        print("No model results to display.")
        return

    # Sort by processing time (fastest first)
    results.sort(key=lambda x: x.stats.time)

    # Determine max model name length for alignment
    # Use the last part of the model ID for display name
    display_names: List[str] = [r.model_name.split('/')[-1] for r in results]
    max_name_len: int = max(len(name) for name in display_names) if display_names else 20
    # Cap max width to prevent overly wide tables
    max_name_len = min(max_name_len, 44)

    col_width: int = 11 # Width for numeric columns MB/s

    # --- Print Header ---
    header_line: str = f"╔═{'═' * max_name_len}═╦═{'═' * col_width}═╦═{'═' * col_width}═╦═{'═' * col_width}═╦═{'═' * col_width}═╗"
    title_line: str = (f"║ {'Model'.ljust(max_name_len)} │ "
                      f"{'Active Δ'.rjust(col_width)} │ "
                      f"{'Cache Δ'.rjust(col_width)} │ "
                      f"{'Peak Mem'.rjust(col_width)} │ "
                      f"{'Time'.rjust(col_width)} ║")
    separator_line: str = f"╠═{'═' * max_name_len}═╬═{'═' * col_width}═╬═{'═' * col_width}═╬═{'═' * col_width}═╬═{'═' * col_width}═╣"

    print("\n--- Model Performance Summary (Console) ---")
    print(header_line)
    print(title_line)
    print(separator_line)

    # --- Print Rows for each model ---
    result: ModelResult
    for i, result in enumerate(results):
        # Truncate display name if needed
        model_disp_name: str = display_names[i]
        if len(model_disp_name) > max_name_len:
            model_disp_name = model_disp_name[:max_name_len-3] + "..."

        # Format statistics strings
        active_str: str = f"{result.stats.active:,.1f} MB"
        cache_str: str = f"{result.stats.cached:,.1f} MB"
        peak_str: str = f"{result.stats.peak:,.1f} MB"
        time_str: str = f"{result.stats.time:.2f} s"

        # Print the row
        print(
            f"║ {model_disp_name.ljust(max_name_len)} │ "
            f"{active_str.rjust(col_width)} │ "
            f"{cache_str.rjust(col_width)} │ "
            f"{peak_str.rjust(col_width)} │ "
            f"{time_str.rjust(col_width)} ║"
        )

    # --- Print average/summary row if multiple results ---
    if len(results) > 1:
        # Calculate averages and maximum peak
        avg_active: float = sum(r.stats.active for r in results) / len(results)
        avg_cache: float = sum(r.stats.cached for r in results) / len(results)
        max_peak: float = max(r.stats.peak for r in results) # Peak is max across runs
        avg_time: float = sum(r.stats.time for r in results) / len(results)

        # Format summary strings
        avg_active_str: str = f"{avg_active:.1f} MB"
        avg_cache_str: str = f"{avg_cache:.1f} MB"
        max_peak_str: str = f"{max_peak:.1f} MB"
        avg_time_str: str = f"{avg_time:.2f} s"

        # Print separator and summary row
        print(separator_line) # Use the main separator line style
        print(
            f"║ {'AVERAGE / MAX PEAK'.ljust(max_name_len)} │ "
            f"{avg_active_str.rjust(col_width)} │ "
            f"{avg_cache_str.rjust(col_width)} │ "
            f"{max_peak_str.rjust(col_width)} │ "
            f"{avg_time_str.rjust(col_width)} ║"
        )

    # --- Print Footer ---
    footer_line: str = f"╚═{'═' * max_name_len}═╩═{'═' * col_width}═╩═{'═' * col_width}═╩═{'═' * col_width}═╩═{'═' * col_width}═╝"
    print(footer_line)


# --- HTML Report Generation ---

def generate_html_report(results: List[ModelResult], filename: Path) -> None:
    """Generates an HTML file with a table of model performance statistics."""
    if not results:
        log.warning("No results to generate HTML report.")
        return

    # Sort by processing time (fastest first) for consistency with console
    results.sort(key=lambda x: x.stats.time)

    # --- HTML Structure ---
    # Using f-strings for assembly, ensure proper escaping if user input were involved (not the case here)
    html_start: str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Performance Results</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f8f9fa; color: #212529; }
        h1 { text-align: center; color: #343a40; border-bottom: 2px solid #dee2e6; padding-bottom: 10px; margin-bottom: 30px; }
        table { border-collapse: collapse; width: 95%; margin: 30px auto; box-shadow: 0 4px 8px rgba(0,0,0,0.1); background-color: #ffffff; }
        th, td { border: 1px solid #dee2e6; padding: 12px 15px; text-align: left; }
        th { background-color: #e9ecef; font-weight: 600; color: #495057; }
        tr:nth-child(even) { background-color: #f8f9fa; }
        tr:hover { background-color: #e2e6ea; }
        td.numeric, th.numeric { text-align: right; font-variant-numeric: tabular-nums; }
        .summary td { font-weight: bold; background-color: #d1ecf1; color: #0c5460; border-color: #bee5eb; }
        caption { caption-side: bottom; padding: 15px; font-style: italic; color: #6c757d; text-align: center; }
        .model-name { font-family: 'Courier New', Courier, monospace; } /* Monospace for model IDs */
    </style>
</head>
<body>
    <h1>Model Performance Summary</h1>
    <table>
        <caption>Performance metrics for Vision Language Model processing. Generated on """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """. Active/Cache Δ = Change in MB. Peak Mem = Peak usage in MB. Time in seconds.</caption>
        <thead>
            <tr>
                <th>Model</th>
                <th class="numeric">Active Δ (MB)</th>
                <th class="numeric">Cache Δ (MB)</th>
                <th class="numeric">Peak Mem (MB)</th>
                <th class="numeric">Time (s)</th>
            </tr>
        </thead>
        <tbody>
"""
    html_rows: str = ""
    result: ModelResult
    for result in results:
        # Use full model name in HTML report for clarity
        model_disp_name: str = result.model_name
        html_rows += f"""
            <tr>
                <td class="model-name">{model_disp_name}</td>
                <td class="numeric">{result.stats.active:,.1f}</td>
                <td class="numeric">{result.stats.cached:,.1f}</td>
                <td class="numeric">{result.stats.peak:,.1f}</td>
                <td class="numeric">{result.stats.time:.2f}</td>
            </tr>
"""
    html_summary_row: str = ""
    # Add summary row if multiple results
    if len(results) > 1:
        avg_active: float = sum(r.stats.active for r in results) / len(results)
        avg_cache: float = sum(r.stats.cached for r in results) / len(results)
        max_peak: float = max(r.stats.peak for r in results)
        avg_time: float = sum(r.stats.time for r in results) / len(results)
        html_summary_row = f"""
            <tr class="summary">
                <td>AVERAGE / MAX PEAK</td>
                <td class="numeric">{avg_active:.1f}</td>
                <td class="numeric">{avg_cache:.1f}</td>
                <td class="numeric">{max_peak:.1f}</td>
                <td class="numeric">{avg_time:.2f}</td>
            </tr>
"""

    html_end: str = """
        </tbody>
    </table>
</body>
</html>
"""
    # Combine parts
    html_content: str = html_start + html_rows + html_summary_row + html_end

    # Write to file
    try:
        # Explicitly type the file handle
        f: IO[str]
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)
        log.info(f"HTML report saved to: {filename.resolve()}")
    except IOError as e:
        log.error(f"Failed to write HTML report to {filename}: {e}")
    except Exception as e:
         # Catch any other unexpected errors during file writing
         log.error(f"An unexpected error occurred while writing HTML report: {type(e).__name__}: {e}", exc_info=log.level <= logging.DEBUG)


# --- Model Processing Core ---

def process_image_with_model(
    model_identifier: str,
    image_path: Path,
    prompt: str,
    max_tokens: int,
    vlm_verbose: bool, # Verbosity for VLM generate function
) -> Optional[ModelResult]:
    """Loads a VLM, generates text for an image, and tracks resources."""
    log.info(f"Processing '{image_path.name}' with model: {model_identifier}")

    # Use Any for model, tokenizer, config as specific types might be internal/complex
    model_obj: Any = None
    tokenizer_obj: Any = None
    config_obj: Any = None

    # Define the core processing logic as a function to pass to the tracker
    # This function should return the generated output string or raise an exception
    def core_processing_logic() -> str: # Return type is string on success
        nonlocal model_obj, tokenizer_obj, config_obj # Allow modification

        # --- Load Model ---
        load_start_time: float = time.perf_counter()
        try:
            # Use trust_remote_code=True cautiously, often needed for VLMs from HF Hub
            # load() typically returns (model, tokenizer)
            model_obj, tokenizer_obj = load(model_identifier, trust_remote_code=True)
            # load_config() returns the model config dictionary
            config_obj = load_config(model_identifier, trust_remote_code=True)
            log.debug(f"Model '{model_identifier}' loaded in {time.perf_counter() - load_start_time:.2f}s")
        except Exception as load_err:
             # Log the specific error during loading
             log.error(f"Failed to load model/tokenizer/config for {model_identifier}: {type(load_err).__name__}: {load_err}", exc_info=log.level <= logging.DEBUG)
             raise RuntimeError(f"Model loading failed for {model_identifier}") from load_err # Re-raise standardized error

        # --- Prepare Prompt ---
        try:
            # apply_chat_template expects tokenizer, config, prompt, num_images
            # It returns the formatted prompt string
            chat_prompt: str = apply_chat_template(tokenizer_obj, config_obj, prompt, num_images=1)
            log.debug(f"Applied chat template for {model_identifier}: {chat_prompt[:100]}...") # Log snippet
        except Exception as prompt_err:
            log.error(f"Failed to apply chat template for {model_identifier}: {type(prompt_err).__name__}: {prompt_err}", exc_info=log.level <= logging.DEBUG)
            raise RuntimeError(f"Prompt templating failed for {model_identifier}") from prompt_err # Re-raise

        # --- Generate Text ---
        gen_start_time: float = time.perf_counter()
        try:
            # generate() returns the generated text as a string
            generated_output: str = generate(
                model=model_obj,
                processor=tokenizer_obj, # 'processor' is the term used in mlx-vlm for tokenizer/processor
                prompt=chat_prompt,
                image_paths=[str(image_path)], # generate expects list of strings
                max_tokens=max_tokens,
                verbose=vlm_verbose, # Pass verbose flag to generate
                temp=0.0 # Deterministic output
            )
            log.debug(f"Generation completed in {time.perf_counter() - gen_start_time:.2f}s for {model_identifier}")
            # Ensure the output is a string, handle potential non-string returns defensively
            return str(generated_output) if generated_output is not None else ""
        except Exception as gen_err:
            log.error(f"Generation failed for {model_identifier}: {type(gen_err).__name__}: {gen_err}", exc_info=log.level <= logging.DEBUG)
            raise RuntimeError(f"Text generation failed for {model_identifier}") from gen_err # Re-raise

    # --- Execute and Track ---
    # track_memory_and_time returns Tuple[Any, MemoryStats]
    # We expect the 'Any' part to be 'str' if core_processing_logic succeeds, None otherwise
    tracked_result: Optional[str] # Explicitly type the expected result
    stats: MemoryStats
    tracked_result, stats = track_memory_and_time(core_processing_logic)

    # Cleanup loaded objects explicitly after tracking (helps free memory sooner, maybe)
    # Use del statement
    try:
         if model_obj is not None:
           del model_obj
         if tokenizer_obj is not None:
           del tokenizer_obj
         if config_obj is not None:
           del config_obj
    except NameError:
         pass # Should not happen if logic flows correctly, but safe to ignore

    # Check if the tracking or the core logic failed
    # tracked_result will be None if core_processing_logic raised an exception caught by track_memory_and_time
    if tracked_result is None:
         # Error messages should have been logged within core_processing_logic or track_memory_and_time
         log.error(f"Processing failed for model {model_identifier} (check previous logs for details).")
         return None # Indicate failure

    # If core_processing_logic succeeded, tracked_result contains the generated text
    log.info(f"Finished processing '{model_identifier}' in {stats.time:.1f}s. Peak Mem: {stats.peak:.1f}MB")
    # Create and return the ModelResult object
    return ModelResult(model_name=model_identifier, output=tracked_result, stats=stats)

# --- Main Execution ---

def main(args: argparse.Namespace) -> None:
    """Main function to orchestrate image analysis."""
    # Setup logging level based on args
    if args.debug:
        log.setLevel(logging.DEBUG)
        log.debug("Debug mode enabled.")
    elif args.verbose: # If only verbose is set, ensure INFO level
        log.setLevel(logging.INFO)
        log.info("Verbose mode enabled.")
    else:
        log.setLevel(logging.INFO) # Default level

    # Print library versions if possible
    try:
         # Use f-string for cleaner output
         print(f"MLX version: {mx.__version__}")
         print(f"MLX-VLM version: {vlm_version}\n")
         log.debug(f"Default MLX device: {mx.default_device()}")
         # Optionally, add huggingface_hub version
         # import huggingface_hub
         # print(f"Hugging Face Hub version: {huggingface_hub.__version__}")
    except NameError: # mx or vlm_version might not be defined if import failed
         log.warning("Could not retrieve MLX/VLM version info (NameError).")
    except AttributeError: # mx might not have __version__ in older versions
         log.warning("Could not retrieve MLX version info (AttributeError).")

    # Record overall start time
    overall_start_time: float = time.perf_counter()

    # --- 1. Validate folder and find the image ---
    # args.folder should be Path type from argparse setup
    folder_path: Path = args.folder.resolve() # Use absolute path
    log.info(f"Scanning folder: {folder_path}")

    # Ensure default folder exists if it's the one being used
    if args.folder == DEFAULT_FOLDER and not DEFAULT_FOLDER.is_dir():
         print(f"Warning: Default folder '{DEFAULT_FOLDER}' does not exist.", file=sys.stderr)
         print(f"Please create it or specify a different folder using the -f option.", file=sys.stderr)
         # Decide if this should be fatal
         # sys.exit(1) # Make it non-fatal for now, find_most_recent_file will handle empty/missing dir

    image_path: Optional[Path] = find_most_recent_file(folder_path)

    if not image_path:
        print(f"\nError: Could not find a suitable image file in {folder_path}.", file=sys.stderr)
        print("Please ensure the folder contains image files and is accessible.", file=sys.stderr)
        sys.exit(1)

    # Use resolved path for clarity in output
    resolved_image_path: Path = image_path.resolve()
    print(f"\nProcessing file: {resolved_image_path.name} (located at {resolved_image_path})")
    print_image_dimensions(resolved_image_path)

    # --- 2. Extract Metadata ---
    metadata: MetadataDict = extract_image_metadata(resolved_image_path)
    print(f"  Date: {metadata.get('date', 'N/A')}") # Use .get for safety
    print(f"  Desc: {metadata.get('description', 'N/A')}")
    print(f"  GPS:  {metadata.get('gps', 'N/A')}")

    # Show detailed EXIF if verbose or debug
    if args.verbose or args.debug:
         exif_data: Optional[ExifDict] = get_exif_data(resolved_image_path)
         if exif_data:
              pretty_print_exif(exif_data, verbose=True) # Pass verbose=True to show all details
         else:
              print("\nNo detailed EXIF data could be extracted.")

    # --- 3. Prepare Prompt ---
    prompt: str
    if args.prompt:
        prompt = args.prompt
        log.info("Using user-provided prompt.")
    else:
        log.info("Generating default prompt based on image metadata.")
        # Construct default prompt using extracted metadata, filter empty parts
        prompt_parts: List[str] = [
            "Provide a factual caption, a brief description, and relevant comma-separated keywords/tags for this image.",
            "The goal is easy cataloguing and searching.",
            # Add context only if available and differs from the default
            f"Context: The picture might relate to '{metadata['description']}'" if metadata.get('description') and metadata['description'] != "No description" else "",
            f"taken around {metadata['date']}" if metadata.get('date') and metadata['date'] != "Unknown date" else "",
            f"near GPS coordinates {metadata['gps']}." if metadata.get('gps') and metadata['gps'] != "Unknown location" else "",
            "Focus on the visual content. Avoid repeating the date or exact GPS coordinates unless visually evident in the image (e.g., a sign)."
        ]
        # Join non-empty parts with a space, remove leading/trailing whitespace
        prompt = " ".join(filter(None, prompt_parts)).strip()
        log.debug("Using generated prompt based on metadata.")

    print(f"\nUsing Prompt:\n{prompt}\n{'-'*40}")

    # --- 4. Find and Process Models ---
    model_identifiers: List[str] = get_cached_model_ids()
    results: List[ModelResult] = [] # Initialize list to store results

    if not model_identifiers:
        print("\nWarning: No models found in Hugging Face cache via huggingface_hub.", file=sys.stderr)
        print("Please ensure models are downloaded (e.g., using huggingface-cli download) and the cache is accessible.", file=sys.stderr)
        # Continue execution even if no models found, summary will be empty
    else:
        print(f"\nFound {len(model_identifiers)} potential models in cache. Processing...")
        separator: str = f"\n{'=' * 80}\n" # Separator for console output between models
        model_id: str
        for model_id in model_identifiers:
            print(separator) # Print separator before starting processing each model
            # Determine VLM verbosity: enable if main script is verbose or debug
            is_vlm_verbose: bool = args.verbose or args.debug
            # Process the image with the current model
            result: Optional[ModelResult] = process_image_with_model(
                model_identifier=model_id,
                image_path=resolved_image_path, # Use resolved path
                prompt=prompt,
                max_tokens=args.max_tokens,
                vlm_verbose=is_vlm_verbose
            )
            if result:
                # Successfully processed
                print(f"\n--- Output from {model_id.split('/')[-1]} ---")
                print(result.output)
                results.append(result)
            else:
                # Processing failed, error logged within process_image_with_model
                print(f"--- Failed to process model: {model_id} (check logs above for details) ---")

    # --- 5. Print Summary Statistics to Console ---
    if results:
        print(f"\n{'=' * 80}\n") # Separator before summary
        print_model_stats(results)
    else:
        print("\nNo models were successfully processed. No performance summary generated.")

    # --- 6. Generate HTML Report ---
    # args.output_html should be Path type
    html_output_path: Path = args.output_html.resolve() # Resolve to absolute path
    if results:
        generate_html_report(results, html_output_path)
    else:
        log.info(f"Skipping HTML report generation to {html_output_path} as no models were successfully processed.")

    # --- Calculate and Print Total Execution Time ---
    overall_time: float = time.perf_counter() - overall_start_time
    print(f"\nTotal execution time: {overall_time:.2f} seconds.")


if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Analyze the most recent image in a folder using cached MLX Vision Language Models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
    )
    parser.add_argument(
        "-f", "--folder",
        type=Path, # Use Path type directly for validation
        default=DEFAULT_FOLDER, # Use the Path constant default
        help="Folder to scan for the most recently modified image.",
    )
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        default=None,
        help="Custom prompt for the VLM. If omitted, a default prompt using image metadata is generated.",
    )
    parser.add_argument(
        "-m", "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of new tokens for the VLM to generate.",
    )
    parser.add_argument(
        "--output-html",
        type=Path, # Use Path type for validation
        default=DEFAULT_HTML_OUTPUT, # Use the Path constant default
        help="Output HTML file path for model performance statistics.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output (INFO level logging), including VLM generation steps and detailed EXIF data.",
    )
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug logging (DEBUG level) for detailed diagnostics (implies verbose VLM/EXIF output).",
    )

    # Parse arguments into a Namespace object
    parsed_args: argparse.Namespace = parser.parse_args()

    # --- Post-parsing validation or setup ---
    # (Optional: Add any further validation of combined arguments if needed)

    # Call the main function with the parsed arguments
    try:
        main(parsed_args)
    except Exception as main_err:
         # Catch unexpected errors in main execution flow
         log.exception(f"An unexpected error occurred during main execution: {main_err}")
         sys.exit(1) # Exit with error status

