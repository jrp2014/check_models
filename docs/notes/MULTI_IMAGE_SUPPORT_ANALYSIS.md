# Multi-Image Support Analysis

**Date**: 19 October 2025  
**Status**: Design Recommendation

## Overview

Analysis of adding multi-image support to `check_models.py` while maintaining backwards compatibility and intuitive CLI design.

## Current State

### Image Handling

- **Current behavior**: Single image from `--folder` (or default folder)
- **Selection logic**: Most recent file by modification time
- **Usage pattern**: `--folder /path/to/images` → finds newest image

### Existing MLX-VLM Support

The underlying `mlx-vlm` library **already supports multiple images**:

```python
# From typings/mlx_vlm/generate.pyi
def generate(
    model: nn.Module,
    processor: PreTrainedTokenizer,
    prompt: str,
    image: str | list[str] | None = None,  # ← Already supports list!
    audio: str | list[str] | None = None,
    verbose: bool = False,
    **kwargs
) -> GenerationResult: ...
```

### Current Code Path

```text
main()
  ↓
find_and_validate_image(args) → Path  # Single image
  ↓
process_models(args, image_path, prompt)
  ↓
process_image_with_model(params)  # params.image_path: Path
  ↓
generate(model, processor, prompt, image=str(image_path), ...)
```

## Design Recommendation

### Proposed CLI Interface

Add a new `--images` argument that coexists with `--folder`:

```bash
# Default behavior (unchanged) - single most recent image
python check_models.py --folder ~/Pictures

# Multiple specific images
python check_models.py --images img1.jpg img2.jpg img3.jpg

# Mix folder + specific images (folder ignored if --images provided)
python check_models.py --images /path/to/img1.jpg /path/to/img2.jpg

# Glob pattern support (future enhancement)
python check_models.py --images ~/Pictures/*.jpg
```

### Argument Definition

```python
parser.add_argument(
    "-i",
    "--images",
    nargs="+",
    type=Path,
    default=None,
    help=(
        "One or more images to process. When specified, overrides --folder. "
        "Models will process all images together (multi-image input). "
        "If omitted, defaults to the most recent image in --folder."
    ),
)
```

### Priority Logic

1. **If `--images` provided**: Use those images (ignore `--folder`)
2. **Else**: Use `--folder` behavior (find most recent single image)

### Handling Conflicting Arguments

**Question**: What if user specifies both `--folder` and `--images`?

Example:

```bash
python check_models.py --folder ~/Pictures --images img1.jpg img2.jpg
```

#### Option 1: Silent Override (Recommended)

`--images` takes precedence; `--folder` is silently ignored.

✅ **Pros**:

- Simple implementation
- Consistent with other CLI tools (e.g., `find --path X --name Y` - last wins)
- No error for harmless redundancy
- Works well with shell aliases/scripts that set default `--folder`

❌ **Cons**:

- User might not realize `--folder` was ignored
- Could hide mistakes in scripts

**Implementation**:

```python
def resolve_images(args: argparse.Namespace) -> list[Path]:
    """Resolve image paths based on CLI arguments.
    
    Priority:
        1. If --images provided: validate and return those paths
        2. Else: find most recent image in --folder (backwards compatible)
    """
    if args.images:
        # --images takes precedence, --folder is ignored
        if args.folder != DEFAULT_FOLDER:
            logger.warning(
                "--images specified; ignoring --folder argument (%s)",
                args.folder
            )
        # ... validate and return images ...
```

#### Option 2: Error on Conflict (Strict)

Raise an error if both provided; force user to choose.

✅ **Pros**:

- Explicit behavior, no ambiguity
- Forces user to think about what they want
- Catches script errors early

❌ **Cons**:

- More restrictive
- Annoying if user has `--folder` in an alias/config
- Extra validation code

**Implementation**:

```python
def resolve_images(args: argparse.Namespace) -> list[Path]:
    """Resolve image paths based on CLI arguments."""
    if args.images and args.folder != DEFAULT_FOLDER:
        print_cli_error(
            "Cannot specify both --folder and --images. "
            "Please choose one image source."
        )
        sys.exit(1)
    
    if args.images:
        # ... validate and return images ...
```

#### Option 3: Mutually Exclusive Group (Enforced by argparse)

Use `add_mutually_exclusive_group()` to prevent both being specified.

✅ **Pros**:

- argparse handles validation automatically
- Clear error message from argparse
- Explicit in `--help` output

❌ **Cons**:

- Both arguments must be in the same group
- Can't have default folder behavior AND explicit images
- Breaks backwards compatibility (existing scripts with `--folder` might fail)
- **Not viable** because `--folder` has a default value (DEFAULT_FOLDER)

**Implementation** (not recommended):

```python
image_group = parser.add_mutually_exclusive_group()
image_group.add_argument("--folder", ...)  # Can't have default!
image_group.add_argument("--images", ...)
```

**Problem**: If neither specified, need a default image source. But mutually exclusive groups don't work well with defaults.

#### Option 4: Append Behavior (Advanced)

`--images` adds to images from `--folder`.

✅ **Pros**:

- Flexible: can combine folder + explicit images
- Power-user feature

❌ **Cons**:

- Complex behavior, hard to explain
- Unexpected ordering issues
- Which image is "primary" for metadata?
- Probably too clever

**Example** (not recommended):

```bash
# Process most recent from ~/Pictures PLUS specific images
python check_models.py --folder ~/Pictures --images extra1.jpg extra2.jpg
# Result: [~/Pictures/recent.jpg, extra1.jpg, extra2.jpg]
```

#### Recommendation: Option 1 (Silent Override with Warning)

**Rationale**:

1. **Least Surprising**: Explicit arguments (`--images`) naturally override implicit behavior (`--folder`)
2. **Backwards Compatible**: Existing scripts work unchanged
3. **Shell-Friendly**: Works with aliases like `alias check='python check_models.py --folder ~/default'`
4. **Informative**: Warning logged when `--folder` ignored (visible in verbose mode)
5. **Standard Practice**: Similar to git, npm, docker (later args override earlier)

**Recommended Implementation**:

```python
def resolve_images(args: argparse.Namespace) -> list[Path]:
    """Resolve image paths based on CLI arguments.
    
    Priority:
        1. If --images provided: validate and return those paths (overrides --folder)
        2. Else: find most recent image in --folder (backwards compatible)
    
    Returns:
        List of validated image paths (may contain single item)
    
    Raises:
        SystemExit: If images not found or invalid
    """
    if args.images:
        # User specified explicit images - this takes precedence
        
        # Warn if --folder was also specified (likely unintentional)
        if args.folder != DEFAULT_FOLDER:
            logger.warning(
                "Both --images and --folder specified. "
                "Using --images and ignoring --folder (%s)",
                args.folder,
            )
        
        image_paths = [img.resolve() for img in args.images]
        print_cli_section(
            f"Processing {len(image_paths)} specified image(s)",
        )
        
        # Validate all images exist and are readable
        for img_path in image_paths:
            if not img_path.exists():
                print_cli_error(f"Image not found: {img_path}")
                sys.exit(1)
            
            try:
                with Image.open(img_path) as img:
                    img.verify()
            except (UnidentifiedImageError, OSError) as err:
                print_cli_error(f"Invalid image {img_path}: {err}")
                sys.exit(1)
        
        return image_paths
    
    else:
        # Legacy behavior: find most recent in folder
        # ... existing find_most_recent_file logic ...
```

### Key Design Decisions

#### ✅ Advantages

1. **Backwards Compatible**: Default behavior unchanged (most recent from folder)
2. **Clear Intent**: `--images` (plural) signals multi-image capability
3. **Intuitive**: Similar to `--models` (multiple values with `nargs="+"`)
4. **Library Support**: MLX-VLM already accepts `list[str]` for images
5. **Non-Ambiguous**: Explicit list vs. folder scanning are separate concerns

#### ⚠️ Considerations

1. **Prompt Design**: User must craft prompts aware of multi-image context
   - Example: "Compare these images" vs. "Describe this image"
2. **Model Compatibility**: Not all VLMs support multiple images
   - Some models might fail or ignore extra images
   - Document model-specific limitations
3. **Performance**: Multiple images increase memory/compute
4. **Validation**: Need to verify all images exist and are valid

## Implementation Plan

### Phase 1: Basic Multi-Image Support

#### 1.1 Add CLI Argument

```python
# In main() argument parser section
parser.add_argument(
    "-i",
    "--images",
    nargs="+",
    type=Path,
    default=None,
    help=(
        "One or more images to process. When specified, overrides --folder. "
        "Models will process all images together (multi-image input). "
        "If omitted, defaults to the most recent image in --folder."
    ),
)
```

#### 1.2 Refactor Image Resolution

Replace `find_and_validate_image()` with new function:

```python
def resolve_images(args: argparse.Namespace) -> list[Path]:
    """Resolve image paths based on CLI arguments.
    
    Priority:
        1. If --images provided: validate and return those paths
        2. Else: find most recent image in --folder (backwards compatible)
    
    Returns:
        List of validated image paths (may contain single item)
    
    Raises:
        SystemExit: If images not found or invalid
    """
    if args.images:
        # User specified explicit images
        image_paths = [img.resolve() for img in args.images]
        print_cli_section(
            f"Processing {len(image_paths)} specified image(s)",
        )
        
        # Validate all images exist and are readable
        for img_path in image_paths:
            if not img_path.exists():
                print_cli_error(f"Image not found: {img_path}")
                sys.exit(1)
            
            try:
                with Image.open(img_path) as img:
                    img.verify()
            except (UnidentifiedImageError, OSError) as err:
                print_cli_error(f"Invalid image {img_path}: {err}")
                sys.exit(1)
        
        return image_paths
    
    else:
        # Legacy behavior: find most recent in folder
        folder_path = args.folder.resolve()
        print_cli_section(
            f"Scanning folder: {Colors.colored(str(folder_path), Colors.MAGENTA)}",
        )
        
        image_path = find_most_recent_file(folder_path)
        if image_path is None:
            print_cli_error(
                f"Could not find the most recent image file in {folder_path}. Exiting.",
            )
            sys.exit(1)
        
        resolved_image_path = image_path.resolve()
        print_cli_section(
            f"Image File: {Colors.colored(resolved_image_path.name, Colors.MAGENTA)}",
        )
        
        # Validate single image
        try:
            with Image.open(resolved_image_path) as img:
                img.verify()
            print_image_dimensions(resolved_image_path)
        except (UnidentifiedImageError, OSError) as err:
            print_cli_error(f"Cannot verify image {resolved_image_path}: {err}")
            sys.exit(1)
        
        return [resolved_image_path]  # Return as list for consistency
```

#### 1.3 Update ProcessImageParams

```python
class ProcessImageParams(NamedTuple):
    """Parameters for processing images with a VLM.

    Attributes:
        model_identifier: Model path or identifier.
        image_paths: Path(s) to the image file(s).  # Changed from image_path
        prompt: Prompt string for the model.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        timeout: Timeout in seconds.
        verbose: Verbose/debug flag.
        trust_remote_code: Allow remote code execution.
    """

    model_identifier: str
    image_paths: list[Path]  # Changed from image_path: Path
    prompt: str
    max_tokens: int
    temperature: float
    timeout: float
    verbose: bool
    trust_remote_code: bool
```

#### 1.4 Update Model Processing

```python
def _format_image_for_generate(image_paths: list[Path]) -> str | list[str]:
    """Convert Path list to format expected by mlx_vlm.generate.
    
    Args:
        image_paths: List of image Path objects
    
    Returns:
        Single string if one image, list[str] if multiple
    """
    str_paths = [str(p) for p in image_paths]
    return str_paths[0] if len(str_paths) == 1 else str_paths


def _generate_vlm_output(...) -> GenerationResult | SupportsGenerationResult:
    """Execute VLM generation with proper image format."""
    # ... existing setup code ...
    
    # Format images for generate function
    image_arg = _format_image_for_generate(image_path)  # Now accepts list
    
    formatted_prompt = apply_chat_template(
        processor=tokenizer,
        config=config,
        prompt=params.prompt,
        num_images=len(params.image_paths),  # Pass actual count
    )
    
    # ... existing formatting code ...
    
    output: GenerationResult | SupportsGenerationResult = generate(
        model=model,
        processor=tokenizer,
        prompt=formatted_prompt,
        image=image_arg,  # Now str | list[str]
        verbose=verbose,
        temperature=params.temperature,
        trust_remote_code=params.trust_remote_code,
        max_tokens=params.max_tokens,
    )
    
    # ... rest unchanged ...
```

#### 1.5 Update main() Flow

```python
def main(args: argparse.Namespace) -> None:
    """Run CLI execution for MLX VLM model check."""
    overall_start_time = time.perf_counter()
    try:
        library_versions = setup_environment(args)
        print_cli_header("MLX Vision Language Model Check")

        # NEW: Resolve image paths (single or multiple)
        image_paths = resolve_images(args)

        # Handle metadata for first image (or aggregate logic TBD)
        metadata = handle_metadata(image_paths[0], args)

        prompt = prepare_prompt(args, metadata)

        # Pass image list to processing
        results = process_models(args, image_paths, prompt)

        finalize_execution(
            args=args,
            results=results,
            library_versions=library_versions,
            overall_start_time=overall_start_time,
            prompt=prompt,
        )
    except (KeyboardInterrupt, SystemExit):
        # ... unchanged ...
```

### Phase 2: Enhancements (Future)

#### 2.1 Metadata Handling

- **Current**: Extract EXIF from single image
- **Multi-image**: Aggregate or display per-image metadata
- **Options**:
  - Only use first image metadata (simplest)
  - Display all metadata in verbose mode
  - Allow `--primary-image` to select which image for metadata

#### 2.2 Glob Pattern Support

```python
import glob

if args.images:
    # Expand globs
    expanded_images = []
    for pattern in args.images:
        if '*' in str(pattern) or '?' in str(pattern):
            expanded_images.extend(glob.glob(str(pattern)))
        else:
            expanded_images.append(pattern)
    image_paths = [Path(p).resolve() for p in expanded_images]
```

#### 2.3 Dimension Reporting

```python
def print_multi_image_dimensions(image_paths: list[Path]) -> None:
    """Print dimensions for multiple images."""
    if len(image_paths) == 1:
        print_image_dimensions(image_paths[0])
    else:
        for idx, img_path in enumerate(image_paths, 1):
            with Image.open(img_path) as img:
                width, height = img.size
                print(f"  Image {idx}: {width}×{height} ({img_path.name})")
```

#### 2.4 Model Capability Detection

```python
# Document which models support multi-image
MULTI_IMAGE_MODELS = {
    "qnguyen3/nanoLLaVA",  # Example - verify actual support
    # ... others ...
}

def validate_multi_image_support(model_id: str, num_images: int) -> None:
    """Warn if model might not support multiple images."""
    if num_images > 1 and model_id not in MULTI_IMAGE_MODELS:
        logger.warning(
            "Model %s may not support multiple images. "
            "It might ignore extra images or fail.",
            model_id
        )
```

## Testing Strategy

### Test Cases

1. **Backwards Compatibility**

   ```bash
   python check_models.py --folder ~/Pictures
   # Should select most recent image (no change)
   ```

2. **Single Explicit Image**

   ```bash
   python check_models.py --images test.jpg
   # Should work identically to folder with single image
   ```

3. **Multiple Images**

   ```bash
   python check_models.py --images img1.jpg img2.jpg
   # Should pass both to generate()
   ```

4. **Invalid Image**

   ```bash
   python check_models.py --images missing.jpg
   # Should error with clear message
   ```

5. **Mixed Absolute/Relative Paths**

   ```bash
   python check_models.py --images ./local.jpg /abs/path/remote.jpg
   # Should resolve and validate both
   ```

6. **Both --folder and --images (Conflict)**

   ```bash
   python check_models.py --folder ~/Pictures --images test.jpg
   # Should use test.jpg and log warning about ignoring --folder
   # Exit code: 0 (success)
   ```

7. **Default Folder + Explicit Images**

   ```bash
   python check_models.py --images test.jpg
   # Should use test.jpg (no warning since --folder not explicitly set)
   # Exit code: 0 (success)
   ```

### Unit Tests

```python
# tests/test_multi_image_support.py

def test_resolve_images_with_explicit_images(tmp_path):
    """Test --images argument with valid images."""
    img1 = tmp_path / "test1.jpg"
    img2 = tmp_path / "test2.jpg"
    create_test_image(img1)
    create_test_image(img2)
    
    args = argparse.Namespace(images=[img1, img2], folder=None)
    result = resolve_images(args)
    
    assert len(result) == 2
    assert result[0] == img1.resolve()
    assert result[1] == img2.resolve()


def test_resolve_images_backwards_compat(tmp_path):
    """Test folder behavior when --images not provided."""
    # Create test images with different timestamps
    old_img = tmp_path / "old.jpg"
    new_img = tmp_path / "new.jpg"
    create_test_image(old_img)
    time.sleep(0.01)
    create_test_image(new_img)
    
    args = argparse.Namespace(images=None, folder=tmp_path)
    result = resolve_images(args)
    
    assert len(result) == 1
    assert result[0] == new_img.resolve()  # Most recent


def test_format_image_for_generate():
    """Test image format conversion for generate()."""
    single = [Path("test.jpg")]
    assert _format_image_for_generate(single) == "test.jpg"
    
    multiple = [Path("a.jpg"), Path("b.jpg")]
    assert _format_image_for_generate(multiple) == ["a.jpg", "b.jpg"]


def test_resolve_images_conflict_warning(tmp_path, caplog):
    """Test warning when both --folder and --images specified."""
    img = tmp_path / "explicit.jpg"
    create_test_image(img)
    
    custom_folder = tmp_path / "custom"
    custom_folder.mkdir()
    
    args = argparse.Namespace(
        images=[img],
        folder=custom_folder  # Non-default folder
    )
    
    with caplog.at_level(logging.WARNING):
        result = resolve_images(args)
    
    # Should use --images
    assert len(result) == 1
    assert result[0] == img.resolve()
    
    # Should log warning about ignoring --folder
    assert "ignoring --folder" in caplog.text.lower()


def test_resolve_images_no_warning_with_default_folder(tmp_path):
    """Test no warning when --images used with default folder."""
    img = tmp_path / "explicit.jpg"
    create_test_image(img)
    
    args = argparse.Namespace(
        images=[img],
        folder=DEFAULT_FOLDER  # Default folder
    )
    
    with caplog.at_level(logging.WARNING):
        result = resolve_images(args)
    
    # Should use --images without warning
    assert len(result) == 1
    assert result[0] == img.resolve()
    assert len(caplog.records) == 0  # No warnings
```

## Documentation Updates

### README.md

```markdown
### Image Input

By default, `check_models.py` processes the most recent image in the specified folder:

```bash
python check_models.py --folder ~/Pictures
```

To process specific images (including multiple images for multi-image VLMs):

```bash
# Single specific image
python check_models.py --images photo.jpg

# Multiple images (for models that support multi-image input)
python check_models.py --images img1.jpg img2.jpg img3.jpg
```

**Note**: Not all vision-language models support multiple images. Models that don't support this feature may ignore extra images or produce errors.

### Help Text

The argparse help will automatically show:

```text
  -i IMAGES [IMAGES ...], --images IMAGES [IMAGES ...]
                        One or more images to process. When specified,
                        overrides --folder. Models will process all images
                        together (multi-image input). If omitted, defaults to
                        the most recent image in --folder. (default: None)
```

## Summary

### Recommended Approach

**Add `--images` argument with `nargs="+"`** for explicit multi-image support:

✅ **Pros**:

- Backwards compatible (default behavior unchanged)
- Clear, intuitive interface
- Leverages existing MLX-VLM multi-image support
- Easy to implement (~200 lines of refactoring)
- Consistent with `--models` pattern
- Handles `--folder` + `--images` gracefully (with warning)

❌ **Minimal Cons**:

- User must understand model-specific limitations
- Need to document which models support multiple images
- Warning message needed when both flags specified (low cost)

### Conflict Resolution Strategy

**If both `--folder` and `--images` specified**:

- `--images` takes precedence (explicit > implicit)
- Log warning: "Using --images and ignoring --folder"
- No error raised (shell-friendly, works with aliases)
- Only warn if `--folder` differs from default

**Rationale**: Standard CLI practice (git, docker, npm use "last wins" / explicit override)

### Alternative Approaches (Not Recommended)

1. **Folder glob pattern**: `--folder "*.jpg"`
   - ❌ Confuses folder scanning vs. file selection
   - ❌ Shell expansion issues

2. **Multiple folder flags**: `--folder dir1 --folder dir2`
   - ❌ Still unclear how many images selected
   - ❌ Doesn't support mixing folders and files

3. **Auto-detect all images in folder**:
   - ❌ Breaks backwards compatibility
   - ❌ Performance impact (many images = slow)
   - ❌ Unclear default behavior

## Next Steps

1. ✅ Review this design document
2. ⏭️ Implement Phase 1 (basic multi-image support)
3. ⏭️ Add unit tests for new functionality
4. ⏭️ Update documentation (README, help text)
5. ⏭️ Test with models known to support multiple images
6. ⏭️ Consider Phase 2 enhancements based on usage patterns
