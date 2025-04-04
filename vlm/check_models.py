#!/usr/bin/env python3
"""Image analysis and caption generation using MLX Vision Language Models."""

# Standard library imports
import argparse
import logging
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import (Any, Dict, Final, Generator, List, Optional, NamedTuple,
                    Tuple, Union)

# Third-party imports
try:
    import mlx.core as mx
    from PIL import Image, UnidentifiedImageError
    from PIL.ExifTags import GPSTAGS, TAGS
    from mlx_vlm import (load, generate, __version__ as vlm_version)
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
except ImportError as e:
    print(f"Error importing required libraries: {e}", file=sys.stderr)
    print("Please ensure you have installed mlx, mlx-lm, Pillow, and huggingface_hub.", file=sys.stderr)
    sys.exit(1)


# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)] # Log errors/debug to stderr
)
log = logging.getLogger(__name__)

# Type aliases and definitions
ExifDict = Dict[Union[str, int], Any] # EXIF keys can be int before mapping
MetadataDict = Dict[str, str]         # Metadata results dictionary
PathLike = Union[str, Path]           # For user input (can be str or Path)

# Constants - Defaults
DEFAULT_MAX_TOKENS: Final[int] = 500
DEFAULT_FOLDER: Final[Path] = Path.home() / "Pictures" / "Processed" # More robust default

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
GPS_LAT_REF_TAG: Final[int] = 1
GPS_LAT_TAG: Final[int] = 2
GPS_LON_REF_TAG: Final[int] = 3
GPS_LON_TAG: Final[int] = 4

# Type definitions

class MemoryStats(NamedTuple):
    """Memory statistics container."""
    active_delta: float  # Change in active memory (MB)
    cache_delta: float   # Change in cache memory (MB)
    peak: float          # Peak memory usage (MB) during the operation
    time: float          # Processing time (seconds)

    @staticmethod
    def zero() -> 'MemoryStats':
        """Create a zero-initialized MemoryStats object."""
        return MemoryStats(active_delta=0.0, cache_delta=0.0, peak=0.0, time=0.0)

@dataclass(frozen=True)
class ModelResult:
    """Container for model processing results."""
    model_name: str
    output: str
    stats: MemoryStats

# Memory management
@contextmanager
def memory_tracker() -> Generator[None, None, None]:
    """Context manager to track MLX memory usage and timing."""
    # Clear cache and reset peak memory *before* the operation
    mx.clear_cache()
    mx.reset_peak_memory()
    # Get initial memory state
    initial_active = mx.get_active_memory()
    initial_cache = mx.get_cache_memory()
    start_time = time.perf_counter()

    try:
        yield # Execute the code block within the 'with' statement
    finally:
        # Calculate deltas and final stats *after* the operation
        # Note: peak memory is reset at the start, so get_peak_memory gives the peak *during* the block
        peak_mem = mx.get_peak_memory()
        end_time = time.perf_counter()
        active_delta = mx.get_active_memory() - initial_active
        cache_delta = mx.get_cache_memory() - initial_cache
        elapsed_time = end_time - start_time

        log.debug(
            f"Memory Tracking: Active Delta: {active_delta / 1e6:.1f} MB, "
            f"Cache Delta: {cache_delta / 1e6:.1f} MB, "
            f"Peak: {peak_mem / 1e6:.1f} MB, "
            f"Time: {elapsed_time:.2f}s"
        )
        # Clean up *after* measurements are taken
        mx.clear_cache()
        # Peak memory is already reset at the start for the next cycle


# --- File Handling ---

def find_most_recent_file(folder: Path) -> Optional[Path]:
    """Return the Path of the most recently modified file in the folder."""
    if not folder.is_dir():
        log.error(f"Provided path is not a directory: {folder}")
        return None
    try:
        files = [
            f for f in folder.iterdir()
            if not f.name.startswith(".") and f.is_file()
        ]
        if not files:
            log.warning(f"No non-hidden files found in: {folder}")
            return None
        # Sort by modification time, newest first
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        log.debug(f"Most recent file found: {files[0]}")
        return files[0]
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
        with Image.open(image_path) as img:
            # Use getexif() which is preferred over _getexif()
            exif_raw = img.getexif()
            if not exif_raw:
                log.debug(f"No EXIF data found in {image_path}")
                return None

            # Decode tags using PIL.ExifTags.TAGS
            exif_decoded: ExifDict = {}
            for tag_id, value in exif_raw.items():
                tag_name = TAGS.get(tag_id, tag_id)
                exif_decoded[tag_name] = value
            
            # Decode GPS tags specifically
            if gps_ifd := exif_raw.get_ifd(Image.Exif.IFD.GPSInfo): # Correct way to get GPS IFD
                gps_decoded : Dict[Union[str, int], Any]= {}
                for gps_tag_id, gps_value in gps_ifd.items():
                     gps_tag_name = GPSTAGS.get(gps_tag_id, gps_tag_id)
                     gps_decoded[gps_tag_name] = gps_value
                exif_decoded["GPSInfo"] = gps_decoded # Store decoded GPS info under "GPSInfo" key

            log.debug(f"Successfully extracted EXIF for {image_path}")
            return exif_decoded

    except FileNotFoundError:
        log.error(f"Image file not found for EXIF extraction: {image_path}")
        return None
    except UnidentifiedImageError:
        log.error(f"Cannot identify image file for EXIF: {image_path}")
        return None
    except AttributeError:
        log.warning(f"PIL version might lack getexif method or image has no EXIF support: {image_path}")
        return None
    except Exception as e:
        log.error(f"Unexpected error reading EXIF from {image_path}: {e}", exc_info=log.level == logging.DEBUG)
        return None

def _format_exif_date(date_str: str) -> Optional[str]:
    """Attempt to parse and format a date string from EXIF."""
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            continue
    log.debug(f"Could not parse date string '{date_str}' with known formats.")
    return None

def _convert_gps_coordinate(
    ref: Optional[Union[str, bytes]],
    coord_tuple: Optional[Tuple[Any, ...]] # Can contain Ratio or float
) -> Optional[float]:
    """Convert EXIF GPS coordinate tuple (Degrees, Minutes, Seconds) to decimal degrees."""
    if ref is None or coord_tuple is None or len(coord_tuple) != 3:
        return None

    try:
        # Decode reference if bytes (e.g., b'N')
        if isinstance(ref, bytes):
            ref = ref.decode('ascii')
        ref = ref.upper()

        # Helper to extract value from Ratio or float
        def get_val(item):
            if hasattr(item, 'real') and hasattr(item, 'imag'): # Check for Ratio object more safely
                 # Handle PIL's Ratio type or Fraction
                 return item.real / item.imag if item.imag != 0 else float(item.real)
            return float(item) # Assume float otherwise

        degrees = get_val(coord_tuple[0])
        minutes = get_val(coord_tuple[1])
        seconds = get_val(coord_tuple[2])

        decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)

        # Adjust sign based on reference (S/W are negative)
        if ref in ['S', 'W']:
            decimal = -decimal

        return decimal

    except (AttributeError, IndexError, TypeError, ValueError, ZeroDivisionError, UnicodeDecodeError) as e:
        log.warning(f"Error converting GPS coordinate component: {e} (Ref: {ref}, Coords: {coord_tuple})")
        return None

def extract_image_metadata(image_path: Path) -> MetadataDict:
    """Extract key metadata (Date, Description, GPS) from image EXIF."""
    metadata: MetadataDict = {
        "date": "Unknown date",
        "description": "No description",
        "gps": "Unknown location", # Changed key name for clarity
    }
    exif = get_exif_data(image_path)
    if not exif:
        return metadata # Return defaults if no EXIF

    # 1. Extract Date
    for tag in EXIF_DATE_TAGS:
        if date_val := exif.get(tag):
            if isinstance(date_val, str):
                if formatted_date := _format_exif_date(date_val):
                    metadata["date"] = formatted_date
                    log.debug(f"Found and formatted date from tag '{tag}': {formatted_date}")
                    break # Stop after finding the first valid date
            else:
                 log.debug(f"Date tag '{tag}' has unexpected type: {type(date_val)}")
    if metadata["date"] == "Unknown date":
         log.debug(f"Could not find a suitable date tag in {EXIF_DATE_TAGS}")


    # 2. Extract Description
    if desc_val := exif.get("ImageDescription"):
        try:
            metadata["description"] = str(desc_val).strip()
            log.debug(f"Found description: {metadata['description']}")
        except Exception as e:
            log.warning(f"Could not convert description to string: {e}")

    # 3. Extract GPS
    if gps_info := exif.get("GPSInfo"):
         if isinstance(gps_info, dict): # Ensure it's the decoded dict we expect
            log.debug(f"Raw GPS Info: {gps_info}")
            lat_ref = gps_info.get("GPSLatitudeRef")
            lat_coord = gps_info.get("GPSLatitude")
            lon_ref = gps_info.get("GPSLongitudeRef")
            lon_coord = gps_info.get("GPSLongitude")

            lat = _convert_gps_coordinate(lat_ref, lat_coord)
            lon = _convert_gps_coordinate(lon_ref, lon_coord)

            if lat is not None and lon is not None:
                metadata["gps"] = f"{lat:+.6f}, {lon:+.6f}" # Standard GPS format
                log.debug(f"Calculated GPS Coordinates: {metadata['gps']}")
            else:
                log.debug("Could not calculate valid GPS coordinates from available tags.")
         else:
              log.warning("GPSInfo tag found but was not a dictionary.")
    else:
         log.debug("No GPSInfo tag found in EXIF data.")


    return metadata

def pretty_print_exif(exif: ExifDict, verbose: bool = False) -> None:
    """Pretty print key EXIF data in a formatted table."""
    if not exif:
        print("No EXIF data available.")
        return

    print("\n--- Key EXIF Data ---")
    # Filter and sort tags
    tags_to_print = []
    for tag, value in exif.items():
        # Skip binary data, complex structures (like GPSInfo dict itself), etc.
        if isinstance(value, bytes) or tag == "GPSInfo":
             continue
        if isinstance(value, tuple) and len(value) > 10: # Skip long tuples likely raw data
             continue
        
        is_important = tag in IMPORTANT_EXIF_TAGS
        if verbose or is_important:
             # Format value nicely
             value_str = str(value)
             if len(value_str) > 60:
                 value_str = value_str[:57] + "..."
             tags_to_print.append((str(tag), value_str, is_important))

    if not tags_to_print:
        print("No relevant EXIF tags found to display.")
        return

    tags_to_print.sort(key=lambda x: x[0]) # Sort alphabetically by tag name

    # Determine column widths
    max_tag_len = max(len(t[0]) for t in tags_to_print) if tags_to_print else 20
    max_val_len = max(len(t[1]) for t in tags_to_print) if tags_to_print else 40
    max_tag_len = max(max_tag_len, 10) # Min width
    max_val_len = max(max_val_len, 15) # Min width

    # Print table header
    header = f"╔═{'═' * max_tag_len}═╤═{'═' * max_val_len}═╗"
    print(header)
    print(f"║ {'Tag'.ljust(max_tag_len)} │ {'Value'.ljust(max_val_len)} ║")
    print(f"╠═{'═' * max_tag_len}═╪═{'═' * max_val_len}═╣")

    # Print rows
    for tag, value_str, is_important in tags_to_print:
        tag_disp = f"\033[1m{tag}\033[0m" if is_important else tag # Bold important tags
        print(f"║ {tag_disp.ljust(max_tag_len + (9 if is_important else 0))} │ {value_str.ljust(max_val_len)} ║") # Adjust for bold escape codes

    # Print table footer
    footer = f"╚═{'═' * max_tag_len}═╧═{'═' * max_val_len}═╝"
    print(footer)


# --- Model Handling ---

def find_huggingface_cli() -> Optional[str]:
    """Locate the huggingface-cli executable."""
    cli_path = shutil.which("huggingface-cli")
    if cli_path:
        log.debug(f"Found huggingface-cli at: {cli_path}")
        return cli_path
    else:
        log.error("huggingface-cli command not found in PATH. Cannot scan cache.")
        log.error("Please ensure the Hugging Face CLI is installed and accessible.")
        return None

def get_cached_model_paths() -> List[str]:
    """Get list of model paths from the huggingface cache using CLI."""
    hf_cli_path = find_huggingface_cli()
    if not hf_cli_path:
        return []

    cmd: List[str] = [hf_cli_path, "scan-cache"]
    log.debug(f"Running command: {' '.join(cmd)}")
    try:
        # Run command, capture stdout, redirect stderr to PIPE to check for errors
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Process stdout
        lines = result.stdout.strip().split("\n")
        model_paths = []
        # Expected format has a header, then repo lines, then summary lines
        # We need to parse the lines listing repositories
        parsing_repos = False
        for line in lines:
            line = line.strip()
            if line.startswith("REPO ID"): # Header for repo list
                parsing_repos = True
                continue
            if line.startswith("Done"): # End of relevant output
                parsing_repos = False
                break
            if parsing_repos and line and not line.startswith("-"):
                # Extract the first part (repo ID)
                parts = line.split()
                if parts:
                    model_paths.append(parts[0])

        log.debug(f"Found {len(model_paths)} potential models in cache.")
        return model_paths

    except FileNotFoundError:
        log.error(f"huggingface-cli executable not found at: {hf_cli_path}")
        return []
    except subprocess.CalledProcessError as e:
        log.error(f"Error running huggingface-cli scan-cache: {e}")
        log.error(f"Stderr: {e.stderr}")
        return []
    except Exception as e:
        log.error(f"Unexpected error getting cached models: {e}", exc_info=log.level == logging.DEBUG)
        return []


def print_model_stats(results: List[ModelResult]) -> None:
    """Print a table summarizing model performance statistics."""
    if not results:
        return

    # Sort by processing time (fastest first)
    results.sort(key=lambda x: x.stats.time)

    # Determine max model name length for alignment
    max_name_len = max(len(r.model_name.split('/')[-1]) for r in results) if results else 20
    max_name_len = min(max_name_len, 44) # Cap max width

    col_width = 11 # Width for numeric columns MB/s

    # Print header
    header = (
        f"╔═{'═' * max_name_len}═╦═{'═' * col_width}═╦═{'═' * col_width}═╦═{'═' * col_width}═╦═{'═' * col_width}═╗\n"
        f"║ {'Model'.ljust(max_name_len)} │ {'Active Δ'.rjust(col_width)} │ {'Cache Δ'.rjust(col_width)} │ {'Peak Mem'.rjust(col_width)} │ {'Time'.rjust(col_width)} ║\n"
        f"╠═{'═' * max_name_len}═╬═{'═' * col_width}═╬═{'═' * col_width}═╬═{'═' * col_width}═╬═{'═' * col_width}═╣"
    )
    print("\n--- Model Performance Summary ---")
    print(header)

    # Print rows for each model
    for result in results:
        model_disp_name = result.model_name.split('/')[-1]
        if len(model_disp_name) > max_name_len:
            model_disp_name = model_disp_name[:max_name_len-3] + "..."

        active_str = f"{result.stats.active_delta:,.0f} MB"
        cache_str = f"{result.stats.cache_delta:,.0f} MB"
        peak_str = f"{result.stats.peak:,.0f} MB"
        time_str = f"{result.stats.time:.1f} s"

        print(
            f"║ {model_disp_name.ljust(max_name_len)} │ "
            f"{active_str.rjust(col_width)} │ "
            f"{cache_str.rjust(col_width)} │ "
            f"{peak_str.rjust(col_width)} │ "
            f"{time_str.rjust(col_width)} ║"
        )

    # Print average/summary row if multiple results
    if len(results) > 1:
        avg_active = sum(r.stats.active_delta for r in results) / len(results)
        avg_cache = sum(r.stats.cache_delta for r in results) / len(results)
        max_peak = max(r.stats.peak for r in results) # Peak is max across runs
        avg_time = sum(r.stats.time for r in results) / len(results)

        avg_active_str = f"{avg_active:,.0f} MB"
        avg_cache_str = f"{avg_cache:,.0f} MB"
        max_peak_str = f"{max_peak:,.0f} MB"
        avg_time_str = f"{avg_time:.1f} s"

        print(f"╠═{'═' * max_name_len}═╬═{'═' * col_width}═╬═{'═' * col_width}═╬═{'═' * col_width}═╬═{'═' * col_width}═╣")
        print(
            f"║ {'AVERAGE / MAX PEAK'.ljust(max_name_len)} │ "
            f"{avg_active_str.rjust(col_width)} │ "
            f"{avg_cache_str.rjust(col_width)} │ "
            f"{max_peak_str.rjust(col_width)} │ "
            f"{avg_time_str.rjust(col_width)} ║"
        )

    # Print footer
    footer = f"╚═{'═' * max_name_len}═╩═{'═' * col_width}═╩═{'═' * col_width}═╩═{'═' * col_width}═╩═{'═' * col_width}═╝"
    print(footer)


def process_image_with_model(
    model_identifier: str,
    image_path: Path,
    prompt: str,
    max_tokens: int,
    vlm_verbose: bool, # Verbosity for VLM generate function
) -> Optional[ModelResult]:
    """Loads a VLM, generates text for an image, and tracks resources."""
    log.info(f"Processing {image_path} with model: {model_identifier}")
    model = tokenizer = config = None # Ensure they are defined for potential cleanup

    # Get initial memory state *before* loading the model
    # These are used to calculate the *delta* caused by this specific model run
    initial_active = mx.get_active_memory()
    initial_cache = mx.get_cache_memory()
    # Note: Peak memory is reset *within* the memory_tracker context

    start_process_time = time.perf_counter()

    try:
        # Use memory_tracker to handle MLX state and timing
        with memory_tracker():
            # --- Load Model ---
            load_start_time = time.perf_counter()
            # trust_remote_code=True is often needed for VLMs from HF Hub
            model, tokenizer = load(model_identifier, trust_remote_code=True)
            config = load_config(model_identifier, trust_remote_code=True)
            log.debug(f"Model '{model_identifier}' loaded in {time.perf_counter() - load_start_time:.2f}s")

            # --- Prepare Prompt ---
            # Assuming single image input for this script
            chat_prompt = apply_chat_template(tokenizer, config, prompt, num_images=1)
            log.debug(f"Applied chat template: {chat_prompt}")


            # --- Generate Text ---
            gen_start_time = time.perf_counter()
            output = generate(
                model=model,
                processor=tokenizer,
                prompt=chat_prompt,
                image_paths=[str(image_path)], # generate expects list of strings
                max_tokens=max_tokens,
                verbose=vlm_verbose, # Pass verbose flag to generate
                # temp=0.0 # Could add temperature control if needed
            )
            log.debug(f"Generation completed in {time.perf_counter() - gen_start_time:.2f}s")

            # Ensure all computations are done for accurate memory measurement
            mx.eval(model.parameters()) # Evaluate lazy arrays

        # --- Measure Stats (after context manager exits) ---
        # Note: peak memory is reset at the start of memory_tracker context
        final_peak = mx.get_peak_memory()
        final_active = mx.get_active_memory()
        final_cache = mx.get_cache_memory()
        end_process_time = time.perf_counter()

        # Calculate deltas relative to state *before* loading this model
        active_delta_mb = (final_active - initial_active) / 1e6
        cache_delta_mb = (final_cache - initial_cache) / 1e6
        peak_mb = final_peak / 1e6
        total_time = end_process_time - start_process_time

        stats = MemoryStats(
            active_delta=active_delta_mb,
            cache_delta=cache_delta_mb,
            peak=peak_mb,
            time=total_time
        )

        log.info(f"Finished processing '{model_identifier}' in {total_time:.1f}s. Peak Mem: {peak_mb:.0f}MB")
        return ModelResult(model_name=model_identifier, output=output, stats=stats)

    except FileNotFoundError:
        log.error(f"Model files not found for identifier: {model_identifier}. Check path or HuggingFace cache.")
        return None
    except Exception as e:
        log.error(f"Error processing model {model_identifier}: {type(e).__name__}: {e}", exc_info=log.level <= logging.DEBUG)
        return None
    finally:
        # Explicitly delete model and tokenizer to potentially free GPU memory sooner
        # Python's garbage collector would eventually do this, but this can help
        # especially in loops processing multiple large models.
        del model
        del tokenizer
        del config
        # mx state (cache, peak) is managed by the memory_tracker context manager

# --- Main Execution ---

def main(args: argparse.Namespace) -> None:
    """Main function to orchestrate image analysis."""
    if args.debug:
        log.setLevel(logging.DEBUG)
        log.debug("Debug mode enabled.")
        try:
             print(f"MLX version: {mx.__version__}")
             print(f"MLX-VLM version: {vlm_version}\n")
             log.debug(f"Default device: {mx.default_device()}")
        except NameError:
             log.warning("Could not retrieve MLX/VLM version info.") # Handle if import failed earlier but wasn't fatal

    overall_start_time = time.perf_counter()

    # 1. Validate folder and find the image
    folder_path = Path(args.folder).resolve() # Use absolute path
    log.debug(f"Scanning folder: {folder_path}")
    image_path = find_most_recent_file(folder_path)

    if not image_path:
        print(f"Error: Could not find a suitable image file in {folder_path}.", file=sys.stderr)
        sys.exit(1)

    print(f"\nProcessing file: {image_path.name}")
    print_image_dimensions(image_path)

    # 2. Extract Metadata
    metadata = extract_image_metadata(image_path)
    print(f"  Date: {metadata['date']}")
    print(f"  Desc: {metadata['description']}")
    print(f"  GPS:  {metadata['gps']}")

    if args.verbose:
         # Optionally print detailed EXIF if verbose
         if exif_data := get_exif_data(image_path):
              pretty_print_exif(exif_data, verbose=True)


    # 3. Prepare Prompt
    if args.prompt:
        prompt = args.prompt
        log.debug("Using user-provided prompt.")
    else:
        # Construct default prompt using extracted metadata
        prompt_parts = [
            "Provide a factual caption, a brief description, and relevant comma-separated keywords/tags for this image.",
            "The goal is easy cataloguing and searching.",
            f"Context: The picture was likely taken near '{metadata['description']}'" if metadata['description'] != "No description" else "",
            f"around {metadata['date']}" if metadata['date'] != "Unknown date" else "",
            f"near GPS coordinates {metadata['gps']}." if metadata['gps'] != "Unknown location" else "",
            "Focus on the visual content. Do not repeat the date or exact GPS coordinates in your response unless visually apparent (e.g., a sign showing coordinates)."
        ]
        prompt = " ".join(filter(None, prompt_parts)) # Join non-empty parts
        log.debug("Using generated prompt based on metadata.")

    print(f"\nUsing Prompt:\n{prompt}\n{'-'*40}")


    # 4. Find and Process Models
    model_identifiers = get_cached_model_paths()
    if not model_identifiers:
        print("Error: No models found in Hugging Face cache via `huggingface-cli scan-cache`.", file=sys.stderr)
        print("Please ensure models are downloaded or the cache is accessible.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(model_identifiers)} potential models in cache. Processing...")

    results: List[ModelResult] = []
    separator = f"\n{'=' * 80}\n"

    for model_id in model_identifiers:
        print(separator)
        result = process_image_with_model(
            model_identifier=model_id,
            image_path=image_path,
            prompt=prompt,
            max_tokens=args.max_tokens,
            vlm_verbose=args.verbose # Use main verbose flag for VLM output too
        )
        if result:
            print(f"\n--- Output from {model_id.split('/')[-1]} ---")
            print(result.output)
            results.append(result)
        else:
            print(f"--- Failed to process model: {model_id} ---")


    # 5. Print Summary Statistics
    if results:
        print(separator)
        print_model_stats(results)
    else:
        print("\nNo models were successfully processed.")

    overall_time = time.perf_counter() - overall_start_time
    print(f"\nTotal execution time: {overall_time:.1f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze the most recent image in a folder using MLX VLMs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
    )
    parser.add_argument(
        "-f", "--folder",
        type=Path,
        default=str(DEFAULT_FOLDER), # Default needs to be string for argparse
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
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output, including VLM generation steps and detailed EXIF data.",
    )
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug logging for detailed diagnostics (implies verbose for some output).",
    )

    parsed_args = parser.parse_args()

    # Ensure default folder exists if used, or provide a helpful message
    default_folder_path = Path(DEFAULT_FOLDER)
    if parsed_args.folder == str(DEFAULT_FOLDER) and not default_folder_path.is_dir():
         print(f"Warning: Default folder '{DEFAULT_FOLDER}' does not exist. Please create it or specify a folder with -f.", file=sys.stderr)
         # Decide if this should be a fatal error or just a warning
         # sys.exit(1) # Uncomment to make it fatal

    main(parsed_args)
