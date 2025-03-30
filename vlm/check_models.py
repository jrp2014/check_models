#!/usr/bin/env python3
"""Image analysis and caption generation using MLX Vision Language Models."""

# Standard library imports
from argparse import ArgumentParser, Namespace
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import (
    Any, Dict, Final, Generator, List, Optional, NamedTuple, TypedDict,
    Union
)

# Third-party imports
import mlx.core as mx
from PIL import Image, UnidentifiedImageError
from PIL.ExifTags import GPSTAGS, TAGS

# Local imports
from mlx_vlm import (
    load, generate, __version__ as vlm_version
)
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

# Type aliases and definitions
ExifDict = Dict[str, Any]  # EXIF data dictionary
MetadataDict = Dict[str, str]  # Metadata results dictionary
PathLike = Union[str, Path]

# Constants - Defaults
DEFAULT_MAX_TOKENS: Final[int] = 500
DEFAULT_FOLDER: Final[Path] = Path("/Users/jrp/Pictures/Processed")

# Constants - EXIF and IPTC
IMPORTANT_EXIF_TAGS: Final[frozenset[str]] = frozenset({
    "DateTimeOriginal", "ImageDescription", "CreateDate",
    "Make", "Model", "LensModel", "ExposureTime",
    "FNumber", "ISOSpeedRatings", "FocalLength", "ExposureProgram",
})

DATE_FORMATS: Final[tuple[str, ...]] = (
    "%Y:%m:%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y%m%d",
)

EXIF_DATE_TAGS: Final[tuple[str, ...]] = (
    "DateTimeOriginal",
    "CreateDate",
    "DateTime"
)

IPTC_TAGS: Final[dict[str, int]] = {
    "LOCATION": 0x5A,
    "DATECREATED": 0x37,
    "CITY": 0x5A,
    "COUNTRY": 0x65,
}

# Type definitions

class MemoryStats(NamedTuple):
    """Memory statistics container."""
    active: float    # Active memory in MB
    cached: float    # Cache memory in MB
    peak: float      # Peak memory in MB
    time: float      # Processing time in seconds

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

class ModelConfig(TypedDict):
    """Type hints for model configuration."""
    chat_template: str
    image_token: str
    eos_token: str

# Memory management
@contextmanager
def memory_tracker(verbose: bool = False) -> Generator[None, None, None]:
    """Context manager to track MLX memory usage and timing."""
    mx.clear_cache()
    mx.reset_peak_memory()
    yield

# File handling functions
def get_most_recently_modified_file(folder: PathLike) -> Optional[PathLike]:
    """Return the name of the most recently modified file in the given folder, excluding hidden files."""
    try:
        folder_path: Path = Path(folder)
        if not folder_path.is_dir():
            return None
        files: List[str] = sorted(
            (f for f in folder_path.iterdir() if not f.name.startswith(".")),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        return files[0] if files else None
    except Exception as e:
        print(f"Error while getting the most recently modified file: {e}")
        return None


def print_image_dimensions(image_path: PathLike, debug: bool = False) -> None:
    """Print the dimensions of the image given as a path string."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            mpx: float = width * height / 1_000_000
            if debug:
                print(f"Image dimensions: {width}x{height} ({mpx:.1f} MPixels)")
    except Exception as e:
        print(f"Error while getting image dimensions: {e}")


def _map_key(k: int) -> str:
    try:
        return TAGS[k]
    except KeyError:
        return GPSTAGS.get(k, k)


@lru_cache(maxsize=128)
def get_exif(image_path: PathLike) -> Optional[ExifDict]:
    """Get EXIF data from image with caching."""
    try:
        with Image.open(image_path) as img:
            if not hasattr(img, '_getexif') or not (exif := img._getexif()):
                return None
            return {
                TAGS.get(tag_id, tag_id): value
                for tag_id, value in exif.items()
            }
    except (UnidentifiedImageError, OSError) as e:
        print(f"Error reading image file: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error reading EXIF data: {e}")
        return None


def format_date(date_str: str) -> str:
    """Format date string in a consistent way."""
    try:
        for fmt in DATE_FORMATS:
            try:
                return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
        return date_str
    except Exception:
        return "Unknown date"


def extract_image_metadata(
    image_path: PathLike, 
    verbose: bool = False,
    debug: bool = False,
) -> MetadataDict:
    """Extract and format image metadata with improved error handling."""
    metadata: MetadataDict = {
        "GPS": "Unknown location",
        "date": "Unknown date",
        "description": "No description",
    }

    try:
        exif = get_exif(image_path)
        if not exif:
            return metadata

        # Extract description
        if desc := exif.get("ImageDescription"):
            metadata["description"] = str(desc).strip(", ")

        # Handle GPS data
        if gps_info := exif.get("GPSInfo"):
            try:
                if debug:
                    print("\nRaw GPS Info:")
                    for k, v in gps_info.items():
                        tag = GPSTAGS.get(k, k)
                        print(f"  {tag}: {v}")

                # Helper function for GPS coordinate conversion
                def get_coordinate(ref_tag: int, coord_tag: int) -> Optional[float]:
                    try:
                        ref = gps_info.get(ref_tag, 'N' if ref_tag == 1 else 'E')
                        if isinstance(ref, bytes):
                            ref = ref.decode()
                        
                        coords = gps_info.get(coord_tag)
                        if not coords or len(coords) != 3:
                            return None

                        degrees = coords[0][0] / coords[0][1] if isinstance(coords[0], tuple) else float(coords[0])
                        minutes = coords[1][0] / coords[1][1] if isinstance(coords[1], tuple) else float(coords[1])
                        seconds = coords[2][0] / coords[2][1] if isinstance(coords[2], tuple) else float(coords[2])

                        decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
                        return -decimal if ref in ['S', 'W'] else decimal
                    except (IndexError, TypeError, ZeroDivisionError) as e:
                        if debug:
                            print(f"Error converting coordinate: {e}")
                        return None

                # Extract coordinates
                lat = get_coordinate(1, 2)  # 1=GPSLatitudeRef, 2=GPSLatitude
                lon = get_coordinate(3, 4)  # 3=GPSLongitudeRef, 4=GPSLongitude

                if debug:
                    print(f"Extracted coordinates - Latitude: {lat}, Longitude: {lon}")

                if lat is not None and lon is not None:
                    metadata["GPS"] = f"{lat:+.6f}°, {lon:+.6f}°"
                    if debug:
                        print(f"Final GPS string: {metadata['GPS']}")

            except Exception as exc:
                print(f"GPS parsing error: {exc}")
                if debug:
                    import traceback
                    traceback.print_exc()

        # Extract and format date
        for tag in EXIF_DATE_TAGS:
            if date_str := exif.get(tag):
                metadata["date"] = format_date(str(date_str))
                break

        return metadata

    except Exception as exc:
        print(f"Error extracting metadata: {exc}")
        if debug:
            import traceback
            traceback.print_exc()
        return metadata


def pretty_print_exif(exif: ExifDict, verbose: bool = False) -> None:
    """Pretty print EXIF data in a formatted table."""
    if not exif:
        print("No EXIF data found")
        return

    print("\n╔════════════════════════════════╦═══════════════════════════════════════════════╗")
    print("║ EXIF Tag                       ║ Value                                         ║")
    print("╠════════════════════════════════╬═══════════════════════════════════════════════╣")

    for tag, value in sorted(exif.items()):
        # Skip binary data and GPS info (handled separately)
        if isinstance(value, bytes) or tag == "GPSInfo":
            continue

        # Skip if not important and not verbose
        if not verbose and tag not in IMPORTANT_EXIF_TAGS:
            continue

        # Format the tag and value strings
        tag_str = str(tag)[:30]
        value_str = str(value)[:45]

        # Highlight important tags
        if tag in IMPORTANT_EXIF_TAGS:
            print(f"║ \033[1m{tag_str:<30}\033[0m ║ {value_str:<45} ║")
        else:
            print(f"║ {tag_str:<30} ║ {value_str:<45} ║")

    print("╚════════════════════════════════╩═══════════════════════════════════════════════╝")


# Model handling functions
def get_model_names() -> List[str]:
    """Get list of available models from huggingface cache."""
    try:
        cmd: List[str] = [
            "/opt/homebrew/Caskroom/miniconda/base/envs/mlx/bin/huggingface-cli",
            "scan-cache"
        ]
        output: bytes = subprocess.check_output(cmd, stderr=subprocess.PIPE)
        return [
            line.split()[0] 
            for line in output.decode().split("\n")[2:-4] 
            if line.strip()
        ]
    except subprocess.CalledProcessError as e:
        print(f"Error accessing huggingface cache: {e}")
        return []


def print_model_stats(results: List[ModelResult]) -> None:
    """Print a table of model statistics."""
    if not results:
        return

    print("\n╔══════════════════════════════════════════════╦═══════════╦═══════════╦═══════════╦═══════════╗")
    print(  "║ Model                                        ║  Active   ║   Cache   ║   Peak    ║   Time    ║")
    print(  "╠══════════════════════════════════════════════╬═══════════╬═══════════╬═══════════╬═══════════╣")
    
    for result in sorted(results, key=lambda x: x.stats.time):
        model_name = result.model_name.split('/')[-1][:40]
        print(
            f"║ {model_name:<44} ║"
            f"{result.stats.active:>8,.0f}MB ║"
            f"{result.stats.cached:>8,.0f}MB ║"
            f"{result.stats.peak:>8,.0f}MB ║"
            f"{result.stats.time:>8.1f}s  ║"
        )
    
    if len(results) > 1:
        avg_stats = MemoryStats(
            active=sum(r.stats.active for r in results) / len(results),
            cached=sum(r.stats.cached for r in results) / len(results),
            peak=max(r.stats.peak for r in results),
            time=sum(r.stats.time for r in results) / len(results)
        )
        print("╠══════════════════════════════════════════════╬═══════════╬═══════════╬═══════════╬═══════════╣")
        print(
            f"║ AVERAGE                                     ║"
            f"{avg_stats.active:>8,.0f}MB ║"
            f"{avg_stats.cached:>8,.0f}MB ║"
            f"{avg_stats.peak:>8,.0f}MB ║"
            f"{avg_stats.time:>8.1f}s  ║"
        )
    
    print("╚══════════════════════════════════════════════╩═══════════╩═══════════╩═══════════╩═══════════╝")

def process_model(
    model_path: PathLike, 
    image_path: PathLike, 
    prompt: str, 
    verbose: bool, 
    debug: bool = False,
    max_tokens: int = DEFAULT_MAX_TOKENS
) -> Optional[ModelResult]:
    """Process model with MLX memory tracking."""
    model = tokenizer = None
    try:
        initial_stats = MemoryStats(
            active=mx.get_active_memory() / 1024 / 1024,
            cached=mx.get_cache_memory() / 1024 / 1024,
            peak=mx.get_peak_memory() / 1024 / 1024,
            time=time.perf_counter()
        )

        with memory_tracker():
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            model, tokenizer = load(model_path, trust_remote_code=True)
            config = load_config(model_path, trust_remote_code=True)
            output = generate(
                model, 
                tokenizer, 
                apply_chat_template(tokenizer, config, prompt, num_images=1),
                [image_path],
                verbose=verbose, 
                max_tokens=max_tokens
            )

            mx.eval(model.parameters())
            final_stats = MemoryStats(
                active=mx.get_active_memory() / 1024 / 1024 - initial_stats.active,
                cached=mx.get_cache_memory() / 1024 / 1024 - initial_stats.cached,
                peak=mx.get_peak_memory() / 1024 / 1024,
                time=time.perf_counter() - initial_stats.time
            )

            if debug:
                print(f"\nModel memory: {final_stats.active:.0f}MB active, "
                      f"{final_stats.peak:.0f}MB peak, {final_stats.time:.1f}s")

            return ModelResult(model_name=model_path, output=output, stats=final_stats)
    except Exception as exc:
        print(f"Error processing model: {type(exc).__name__}: {exc}")
        if debug:
            import traceback
            traceback.print_exc()
    finally:
        del model, tokenizer
        mx.clear_cache()
        mx.reset_peak_memory()
    return None

def main(
    folder: PathLike,
    prompt: Optional[str] = None,
    verbose: bool = False,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    debug: bool = False,
) -> None:
    """Process images with MLX vision language models."""
    results: List[ModelResult] = []
    start_time: float = time.perf_counter()

    # Validate inputs
    if not (folder_path := Path(folder)).is_dir():
        print(f"Error: {folder} is not a valid directory")
        return

    if not (image_path := get_most_recently_modified_file(folder_path)):
        print("No files found in the folder.")
        return

    print(f"\nProcessing file: {image_path}")
    print_image_dimensions(image_path, debug=debug)

    if not (metadata := extract_image_metadata(image_path, verbose, debug)):
        print("Error: Could not extract metadata from image")
        return

    # Generate prompt if none provided
    actual_prompt: str = prompt or (
        f"Provide a factual caption, description and comma-separated "
        f"keywords or tags for this image so that it can be catalogued "
        f"and searched for easily. The picture was taken in "
        f"{metadata['description']} on {metadata['date']}"
        + (f" from the GPS location {metadata['GPS']}. "
           "Do not include this GPS location or the date in your response."
           if metadata['GPS'] != "Unknown location" else "")
    )

    # Process all models
    if not (models := get_model_names()):
        print("No models found in huggingface cache")
        return

    for model_path in models:
        print(f"\n{80 * 'v'}\n\033[1mProcessing {model_path}\033[0m")

        if result := process_model(
            model_path, 
            str(image_path), 
            actual_prompt, 
            verbose, 
            debug, 
            max_tokens
        ):
            print(f"\nModel output:\n{result.output}")
            results.append(result)
            if debug:
                print_model_stats([result])

    if results:
        print_model_stats(results)
    print(f"\nTotal execution time: {time.perf_counter() - start_time:.1f}s")

if __name__ == "__main__":
    print(f"\nMLX version: {mx.__version__}")
    print(f"MLX-VLM version: {vlm_version}\n")

    parser: ArgumentParser = ArgumentParser(
        description="Describe, caption and keyword the most recently modified image in a folder"
    )
    parser.add_argument(
        "-f",
        "--folder",
        type=PathLike,
        default=DEFAULT_FOLDER,
        help="The folder to scan for the most recently modified file",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="Custom prompt to use for image analysis",
    )
    parser.add_argument(
        "-m",
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum number of tokens to generate (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug output",
    )

    args: Namespace = parser.parse_args()
    main(args.folder, args.prompt, args.verbose, args.max_tokens, args.debug)
