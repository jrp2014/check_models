# MLX Vision Language Model Checker (`check_models.py`)

`check_models.py` is a focused benchmarking and inspection tool for MLX-compatible Vision Language Models (VLMs) on Apple Silicon. It loads one or more local / cached models, optionally derives a prompt from image metadata (EXIF + GPS), runs generation, and reports performance (tokens, speed, timings, memory) plus outputs in colorized CLI, HTML, and Markdown formats.

## Who is this for?

- Users (including MLX/MLX‑VLM/MLX‑LM developers) who want to run models on their own images and quickly see outputs and metrics: see the TL;DR below.
- Contributors who want to work on `check_models.py` itself and keep quality gates green: see the Contributors section near the end.

## TL;DR for Users

Defaults assume you have Hugging Face models in your local cache. The tool will discover cached VLMs automatically (unless you pass `--models`), carefully read EXIF metadata (including GPS/time when present) to enrich prompts, and write reports to `output/results.html` and `output/results.md` by default.

Quickest start on Apple Silicon (Python 3.13):

```bash
# From repository root
make install

# Run against a folder of images (change the path to your images)
python -m check_models --folder ~/Pictures/Processed --prompt "Describe this image."

# Try specific models
python -m check_models --folder ~/Pictures/Processed --models mlx-community/nanoLLaVA mlx-community/llava-1.5-7b-hf

# Exclude a model from auto-discovered cache
python -m check_models --folder ~/Pictures/Processed --exclude "microsoft/Phi-3-vision-128k-instruct"
```

Prefer working directly in `src/`?

```bash
cd src
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Optional extras for system metrics & tokenizers
pip install -e ".[extras]"

# Optional: Install PyTorch (needed for some models like Phi-3-vision)
pip install -e ".[torch]"

# Run directly
python check_models.py -f ~/Pictures/Processed -p "Describe this image."
**Python**: 3.13+ (project is configured and tested on Python 3.13)

Key defaults and parameters:

- Models: discovered from Hugging Face cache. Use `--models` for explicit IDs, `--exclude` to filter.
- Images: `-f/--folder` points to your images; default is `~/Pictures/Processed`.
- **Folder behavior**: When you pass a folder path, the script automatically selects the **most recently modified image file** in that folder (hidden files are ignored).
- Reports: By default, `output/results.html` and `output/results.md` are created; override via `--output-html` and `--output-markdown`.
- Prompting: If `--prompt` isn't provided, the tool can compose a metadata‑aware prompt from EXIF data when available (camera, time, GPS).
- Runtime: `--timeout 300`, `--max-tokens 500`, `--temperature 0.1` by default.
- Sampling: `--top-p 1.0` (nucleus sampling), `--repetition-penalty` (disabled by default).
- Memory: `--lazy-load` (off by default), `--kv-bits` and `--max-kv-size` for KV cache optimization.
- Security: `--trust-remote-code=True` by default for Hub models; only use with trusted sources.

## Capabilities

- **Model Discovery**: Auto-discovers locally cached MLX VLMs from Hugging Face cache or processes explicit model list with `--models`
- **Selection Control**: Use `--exclude` to filter models from cache scan or explicit list
- **Folder Mode**: Automatically selects most recently modified image from specified folder
- **Metadata Extraction**: Robust EXIF + GPS parsing with fail-soft multi-pass strategy for partially corrupted metadata
- **Smart Prompting**: Generates metadata-aware prompts (image description, GPS, date) when `--prompt` not provided
- **Performance Metrics**:
  - Timing: generation_time, model_load_time, total_time
  - Tokens: total, prompt, generated with tokens/sec
  - Memory: peak, active delta, cached delta (GB)
- **Structured Logging**: Formatter-driven styling with LogStyles for consistent CLI output
- **Multiple Output Formats**:
  - **CLI**: Colorized with compact or detailed metrics modes
  - **HTML**: Standalone report with inline CSS, failed row highlighting
  - **Markdown**: GitHub-compatible with pipe tables
- **Error Handling**: Per-model isolation with detailed diagnostics; graceful timeout/failure handling
- **Machine Parsable**: SUMMARY lines with `key=value` format for automation
- **Visual Hierarchy**: Emoji prefixes, tree-structured metrics, wrapped text output

## Feature Highlights

| Area | Notes |
|------|-------|
| Model discovery | Scans Hugging Face cache; explicit `--models` overrides. |
| Selection control | `--exclude` works with cache scan or explicit list. |
| Prompting | `--prompt` overrides; otherwise metadata‑informed (image description, GPS, date). |
| Performance | generation_time, model_load_time, total_time, token counts, TPS, peak memory. |
| Reporting | CLI (color), HTML (standalone), Markdown (GitHub). |
| Robustness | Per‑model isolation; failures logged; SUMMARY lines for automation. |
| Timeout | Signal‑based (UNIX) manager; configurable per run. |
| Output preview | Non‑verbose mode still shows wrapped generated text (80 cols). |
| Metrics modes | Compact (default) or expanded with `--detailed-metrics`. |

## Installation and Environment Setup

### Using `pyproject.toml` (recommended)

This project includes a `pyproject.toml` file that defines all dependencies and can be used with modern Python package managers.

### With uv (fastest)

```bash
# Install uv if you haven't already
pip install uv

# Create environment and install dependencies
uv venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate     # On Windows
uv pip install -e .  # run inside the src/ directory (root-level Makefile is a shim)

# For development with additional tools
uv pip install -e ".[dev]"
```

### With pip

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate     # On Windows

# Install the package
pip install -e .  # run inside src/

# For development
pip install -e ".[dev]"
```

### With conda

```bash
# Create conda environment
conda create -n mlx-vlm python=3.13
conda activate mlx-vlm

# Install dependencies
pip install -e .  # run inside src/

# Optional: Install PyTorch and other extras
pip install -e ".[torch,extras]"
```

### Automated Setup (Recommended)

For the easiest setup, use the provided shell script that automates the entire conda environment creation:

```bash
# Make script executable (if needed)
chmod +x setup_conda_env.sh

# Create environment with default name 'mlx-vlm'
./setup_conda_env.sh

# Or create with custom name
./setup_conda_env.sh my-custom-env-name

# View help
./setup_conda_env.sh --help
```

The script will:

- Check for macOS and Apple Silicon compatibility
- Create a conda environment with Python 3.13
- Install all required and optional dependencies
- Install the package in development mode
- Verify the installation
- Provide usage instructions

Torch support:

- During setup, you'll be prompted to install the optional PyTorch stack (torch, torchvision, torchaudio). Choose 'y' if you plan to use models that rely on Torch.
- You can also install later via the project extra: `pip install -e ".[torch]"`.
- The update helper script supports installing Torch too: run with `INSTALL_TORCH=1` to include it.

## Notes on Metrics and Output Formatting

- Memory units: All memory metrics are displayed in GB. Sources differ: MLX reports bytes; mlx‑vlm reports decimal GB (bytes/1e9). The tool detects and normalizes both to GB for consistent display.
- Markdown escaping: The final output column preserves common GitHub‑supported tags (e.g., `<br>`) and escapes others so special tokens like `<s>` render literally.
- Compact vs detailed metrics (verbose mode): By default, verbose output shows a single aligned line beginning with `Metrics:`. Enable classic multi‑line breakdown with `--detailed-metrics`.
- Token triple legend: `tokens(total/prompt/gen)=T/P/G` corresponds to total tokens = prompt tokens + generated tokens.

## Environment Variable Overrides

Several behaviors can be customized via environment variables (useful for CI/automation):

| Variable | Purpose | Default | Example |
|----------|---------|---------|---------|
| `MLX_VLM_WIDTH` | Force CLI output width (columns) | Auto-detect terminal | `MLX_VLM_WIDTH=120` |
| `NO_COLOR` | Disable ANSI colors in output | Not set (colors enabled) | `NO_COLOR=1` |
| `FORCE_COLOR` | Force ANSI colors even in non-TTY | Not set | `FORCE_COLOR=1` |
| `TRANSFORMERS_NO_TF` | Block TensorFlow loading | `1` (blocked) | `TRANSFORMERS_NO_TF=0` to allow |
| `MLX_VLM_ALLOW_TF` | Override TensorFlow blocking | Not set (blocked) | `MLX_VLM_ALLOW_TF=1` to allow |
| `TOKENIZERS_PARALLELISM` | Disable tokenizer parallelism warnings | `false` | `TOKENIZERS_PARALLELISM=true` |
| `MLX_METAL_JIT` | Metal kernel compilation mode | `OFF` (pre-built) | `MLX_METAL_JIT=ON` for runtime JIT |

**Examples**:

```bash
# Force wider output for CI logs
MLX_VLM_WIDTH=120 python check_models.py --folder ~/Pictures

# Disable colors for log file capture
NO_COLOR=1 python check_models.py > output.log 2>&1

# Allow TensorFlow for specific model (may crash on Apple Silicon)
MLX_VLM_ALLOW_TF=1 python check_models.py --models model-needing-tf
```

## Git Hygiene and Caches

This repo excludes ephemeral caches and local environments via `.gitignore`. Common exclusions include `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`, `.venv/`, and editor folders like `.vscode/`. Do not commit large model caches (e.g., Hugging Face) to the repository.

## Pre-commit (Optional)

To enforce formatting, lint, type, and dependency sync locally:

```bash
pip install pre-commit
pre-commit install
```

Hooks run automatically on commit. Run against all files manually:

```bash
pre-commit run --all-files
```


### Manual Installation

If you prefer to install dependencies manually (ensure these match `pyproject.toml`):

<!-- BEGIN MANUAL_INSTALL -->
```bash
pip install "huggingface-hub>=0.23.0" "mlx>=0.29.1" "mlx-vlm>=0.0.9" "Pillow>=10.3.0" "tabulate>=0.9.0" "tzlocal>=5.0"
```
<!-- END MANUAL_INSTALL -->

## Requirements

- **Python**: 3.13+ (3.13 is the tested baseline)
- **Operating System**: macOS with Apple Silicon (MLX is Apple‑Silicon specific)

## Dependencies

Why so slim? The runtime dependency set is intentionally minimized to only the
packages directly imported by `check_models.py`. Everything else that might
inference helpers, PyTorch stack) lives in optional extras. Benefits:

- Faster cold installs / CI setup
- Smaller transitive surface → fewer unexpected resolver conflicts
- Clearer signal when a new import is introduced (you must add it to
  `[project.dependencies]` or tests + sync tooling will fail)
If you add a new top‑level import in `check_models.py`, promote its package
from an optional group (or add it fresh) into the runtime `dependencies` array
and re-run the sync helper.

Runtime (installed automatically via `pip install -e .` when executed inside `src/`, or via `make install` from repo root):

| Purpose | Package | Minimum |
|---------|---------|---------|
| Core tensor/runtime | `mlx` | `>=0.29.1` |
| Vision‑language utilities | `mlx-vlm` | `>=0.0.9` |
| Image processing & loading | `Pillow` | `>=10.3.0` |
| Model cache / discovery | `huggingface-hub` | `>=0.23.0` |
| Reporting / tables | `tabulate` | `>=0.9.0` |
| Local timezone conversion | `tzlocal` | `>=5.0` |

Optional (enable additional features):

| Feature | Package | Source | Install Command |
|---------|---------|--------|-----------------|
| Extended system metrics (RAM/CPU) | `psutil` | `extras` | `pip install -e ".[extras]"` or `make install` |
| Fast tokenizer backends | `tokenizers` | `extras` | `pip install -e ".[extras]"` or `make install` |
| Tensor operations (for some models) | `einops` | `extras` | `pip install -e ".[extras]"` or `make install` |
| Number-to-words conversion (for some models) | `num2words` | `extras` | `pip install -e ".[extras]"` or `make install` |
| Language model utilities | `mlx-lm` | `extras` | `pip install -e ".[extras]"` or `make install` |
| Transformer model support | `transformers` | `extras` | `pip install -e ".[extras]"` or `make install` |
| PyTorch stack (needed for some models) | `torch`, `torchvision`, `torchaudio` | `torch` | `pip install -e ".[torch]"` or `make install-torch` |

**Note**: Some models (e.g., Phi-3-vision, certain Florence2 variants) require PyTorch. If you encounter import errors for `torch`, `torchvision`, or `torchaudio`, install with:

```bash
# From root directory:
make install-torch

# Or from src/ directory:
pip install -e ".[torch]"

# Install everything (extras + torch + dev):
make install-all  # from root
pip install -e ".[extras,torch,dev]"  # from src/
```

Development / QA:

| Purpose | Package |
|---------|---------|
| Linting & formatting checks | `ruff` |
| Static type checking | `mypy` |
| Testing | `pytest`, `pytest-cov` |

### Minimal Install (runtime only)

<!-- BEGIN MINIMAL_INSTALL -->
```bash
pip install "huggingface-hub>=0.23.0" "mlx>=0.29.1" "mlx-vlm>=0.0.9" "Pillow>=10.3.0" "tabulate>=0.9.0" "tzlocal>=5.0"
```
<!-- END MINIMAL_INSTALL -->

### With Optional Extras

The `extras` group in `pyproject.toml` pulls in `psutil`, `tokenizers`, `mlx-lm`, and `transformers`:

```bash
pip install -e ".[extras]"  # adds psutil, tokenizers
```

To include the optional PyTorch stack when needed (macOS wheels include MPS acceleration):

```bash
pip install -e ".[torch]"
```

### Full Development Environment

```bash
pip install -e ".[dev,extras]"  # dev tools + optional metrics/tokenizers
```

Notes:

- `psutil` is optional (installed with `extras`); if absent the extended Apple Silicon hardware section omits RAM/cores.
- `extras` group bundles: psutil, tokenizers, mlx-lm, transformers. Install only if you need extended metrics or LM/transformer features.
- Keep `transformers` updated if using it: `pip install -U transformers`.
- `system_profiler` is a macOS built-in (no install needed) used for GPU name / core info.
- Torch is supported and can be installed when you need it for specific models; the script does not block Torch.
- The `tools/update.sh` helper supports environment flags: `INSTALL_TORCH=1` for PyTorch, `MLX_METAL_JIT=ON` for smaller binaries with runtime compilation (default: `OFF` for pre-built kernels), `CLEAN_BUILD=1` to clean build artifacts first.
- Installing `sentence-transformers` isn't necessary for this tool and may pull heavy backends into import paths; a heads‑up is logged if detected.
- Long embedded CSS / HTML lines are intentional (readability > artificial wrapping).
- Dependency versions in this README are automatically kept in sync with `pyproject.toml`; update the TOML first and reflect changes here.
- To clean build artifacts: `make clean` (project), `make clean-mlx` (local MLX repos), or `bash tools/clean_builds.sh`.

## Python API

The package exports a clean public API for programmatic use:

### Quality Analysis

```python
from check_models import analyze_generation_text, GenerationQualityAnalysis

# Analyze generated text for quality issues
text = "Model output goes here..."
analysis = analyze_generation_text(text, generated_tokens=50)

# Access analysis results
if analysis.is_repetitive:
    print(f"Repetitive token: {analysis.repeated_token}")
if analysis.hallucination_issues:
    print(f"Hallucinations detected: {analysis.hallucination_issues}")
if analysis.is_verbose:
    print("Output is excessively verbose")
if analysis.formatting_issues:
    print(f"Formatting problems: {analysis.formatting_issues}")
if analysis.has_excessive_bullets:
    print(f"Too many bullets: {analysis.bullet_count}")
```

### Core Functions

```python
from check_models import (
    process_image_with_model,    # Process single image with a model
    generate_markdown_report,     # Create Markdown report
    generate_html_report,         # Create HTML report
    get_system_info,              # Get system information dict
    format_field_value,           # Format metric values consistently
    format_overall_runtime,       # Format duration strings
)
```

See module docstrings and `__all__` exports for complete API reference.

## Usage

### Quick Start

```bash
# Run across all cached models with a custom prompt
python check_models.py -f ~/Pictures/Processed -p "What is the main object in this image?"

# Explicit model list (skips cache discovery)
python check_models.py -f ~/Pictures/Processed -m mlx-community/nanoLLaVA mlx-community/llava-1.5-7b-hf

# Exclude specific models from the automatic cache scan
python check_models.py -f ~/Pictures/Processed -e mlx-community/problematic-model other/model

# Combine explicit list with exclusions
python check_models.py -f ~/Pictures/Processed -m model1 model2 model3 -e model2

# Verbose (debug) mode for detailed logs
python check_models.py -f ~/Pictures/Processed -v
```

### Advanced Example

```bash
python check_models.py \
  --folder ~/Pictures/TestImages \
  --exclude "microsoft/Phi-3-vision-128k-instruct" \
  --prompt "Provide a detailed caption for this image" \
  --max-tokens 200 \
  --temperature 0.1 \
  --timeout 600 \
  --output-html ~/reports/vlm_benchmark.html \
  --output-markdown ~/reports/vlm_benchmark.md \
  --verbose
```

### Advanced Parameters Example

Control sampling behavior and memory optimization:

```bash
# Nucleus sampling with repetition penalty
python check_models.py \
  --folder ~/Pictures \
  --top-p 0.9 \
  --repetition-penalty 1.2 \
  --repetition-context-size 50

# Memory optimization for large models
python check_models.py \
  --folder ~/Pictures \
  --lazy-load \
  --max-kv-size 4096 \
  --kv-bits 4

# Combine sampling and memory optimization
python check_models.py \
  --folder ~/Pictures \
  --models mlx-community/Qwen2-VL-7B-Instruct \
  --top-p 0.95 \
  --repetition-penalty 1.1 \
  --lazy-load \
  --kv-bits 8 \
  --max-kv-size 8192
```

#### Sampling Parameters

- **`--top-p`**: Nucleus sampling controls output diversity. Lower values (e.g., 0.9) produce more focused text; 1.0 disables it.
- **`--repetition-penalty`**: Values > 1.0 discourage the model from repeating phrases. Useful for VLMs that tend to loop.
- **`--repetition-context-size`**: How many recent tokens to consider when applying repetition penalty (default: 20).

#### Memory Optimization

- **`--lazy-load`**: Loads model weights on-demand rather than all at once. Reduces peak memory usage, recommended for models larger than half your RAM.
- **`--max-kv-size`**: Limits the KV cache size to prevent memory overflow on long sequences.
- **`--kv-bits`**: Quantizes the KV cache to 4 or 8 bits, significantly reducing memory with minimal quality loss.
- **`--kv-group-size`**: Group size for KV quantization (advanced tuning).
- **`--quantized-kv-start`**: Token position to start quantization (advanced tuning).

**Tip**: For 16GB systems running large models, try `--lazy-load --kv-bits 4 --max-kv-size 4096`.

## Command Line Reference

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-f`, `--folder` | Path | `~/Pictures/Processed` | Folder of images to process (non‑recursive). |
| `--output-html` | Path | `output/results.html` | HTML report output file. |
| `--output-markdown` | Path | `results.md` | Markdown report output file. |
| `-m`, `--models` | list[str] | (none) | Explicit model IDs/paths; disables cache discovery. |
| `-e`, `--exclude` | list[str] | (none) | Models to exclude (applies to cache scan or explicit list). |
| `--trust-remote-code` | flag | `True` | Allow custom code from Hub models (SECURITY RISK). |
| `-p`, `--prompt` | str | (auto) | Custom prompt; if omitted a metadata‑aware prompt may be used. |
| `-x`, `--max-tokens` | int | 500 | Max new tokens to generate. |
| `-t`, `--temperature` | float | 0.1 | Sampling temperature. |
| `--top-p` | float | 1.0 | Nucleus sampling parameter (0.0-1.0); lower = more focused. |
| `--repetition-penalty` | float | (none) | Penalize repeated tokens (>1.0 discourages repetition). |
| `--repetition-context-size` | int | 20 | Context window size for repetition penalty. |
| `--lazy-load` | flag | `False` | Use lazy loading (loads weights on-demand, reduces memory). |
| `--max-kv-size` | int | (none) | Maximum KV cache size (limits memory for long sequences). |
| `--kv-bits` | int | (none) | Quantize KV cache to N bits (4 or 8); saves memory. |
| `--kv-group-size` | int | 64 | Quantization group size for KV cache. |
| `--quantized-kv-start` | int | 0 | Start position for KV cache quantization. |
| `--timeout` | float | 300 | Operation timeout (seconds) for model execution. |
| `-v`, `--verbose` | flag | `False` | Enable verbose + debug logging. |
| `--no-color` | flag | `False` | Disable ANSI colors in the CLI output. |
| `--force-color` | flag | `False` | Force-enable ANSI colors even if stderr is not a TTY. |
| `--width` | int | (auto) | Force a fixed output width (columns) for separators and wrapping. |

### Selection Logic

1. No selection flags: run all cached VLMs.
2. `--models` only: run exactly that list.
3. `--exclude` only: run cached minus excluded.
4. `--models` + `--exclude`: intersect explicit list then subtract exclusions.

The script warns about exclusions that don't match any candidate model.

## Output Formats

### CLI Output

Real-time colorized output showing:

- Model processing progress with success/failure indicators
- Performance metrics (tokens/second, memory usage, timing)
- Generated text preview
- Error diagnostics for failed models
- Final performance summary table

Color conventions:

- Identifiers (file/folder paths and model names) are shown in magenta for quick scanning.
- Failures are shown in red; the compact CLI table also highlights failed model names in red.

Width and color controls:

- Colors are enabled by default on TTYs. You can override with flags or environment variables:
  - Disable colors: `--no-color` or set `NO_COLOR=1`
  - Force-enable colors (even when not a TTY): `--force-color` or set `FORCE_COLOR=1`
- Output width is auto-detected and clamped for readability. You can force a specific width:
  - Use `--width 100` to render at 100 columns
  - Or set `MLX_VLM_WIDTH=100`
  These affect separator lengths and line wrapping for previews and summaries.

### HTML Report

Professional report featuring:

- Executive summary with test parameters
- Interactive performance table with sortable columns
- Model outputs and diagnostics
- System information and library versions
- Failed rows are highlighted in red for quick identification
- Responsive design for mobile viewing

### Markdown Report

GitHub-compatible format with:

- Performance metrics in table format
- Model outputs
- System and library version information
- Easy integration into documentation

## Additional Examples

```bash
# Test all available models
python check_models.py

# Test only fast models, exclude slow ones
python check_models.py --exclude "meta-llama/Llama-3.2-90B-Vision-Instruct"

# Test specific models for comparison
python check_models.py --models \
  "microsoft/Phi-3-vision-128k-instruct" \
  "Qwen/Qwen2-VL-7B-Instruct" \
  "BAAI/Bunny-v1_1-Llama-3-8B-V"

# Test a curated list but exclude one problematic model
python check_models.py \
  --models "model1" "model2" "model3" "model4" \
  --exclude "model3"
```

## Metrics Tracked

The script tracks and reports:

- **Token Metrics**: Prompt tokens, generation tokens, total processing
- **Speed Metrics**: Tokens per second for prompt processing and generation
- **Memory Usage**: Peak memory consumption during processing
- **Timing**: Total processing time per model
- **Success Rate**: Model success/failure statistics
- **Error Analysis**: Detailed error reporting and diagnostics

## Troubleshooting

### Common Issues

**No models found**: Ensure MLX-compatible VLMs are downloaded to your Hugging Face cache

```bash
# Download a model explicitly
huggingface-cli download microsoft/Phi-3-vision-128k-instruct
```

**Import errors**: Verify MLX installation on Apple Silicon Mac

```bash
pip install --upgrade mlx mlx-vlm
```

**Timeout errors**: Increase timeout for large models

```bash
python check_models.py --timeout 600  # 10 minutes
```

**Memory errors**: Test models individually or exclude large models

```bash
python check_models.py --exclude "meta-llama/Llama-3.2-90B-Vision-Instruct"
```

**Script crashes with mutex error**: If you see `libc++abi: terminating due to uncaught exception of type std::__1::system_error: mutex lock failed: Invalid argument`, TensorFlow is installed and conflicting with MLX.

The script automatically prevents TensorFlow from loading by setting `TRANSFORMERS_NO_TF=1` (unless you override with `MLX_VLM_ALLOW_TF=1`), so if TensorFlow gets imported anyway:

Option 1 - Keep TensorFlow but ensure it's blocked (safest, script does this automatically):

```bash
export TRANSFORMERS_NO_TF=1
python check_models.py  # Run normally - the script sets this env var automatically
```

Option 2 - Uninstall TensorFlow completely (recommended for MLX-only environments):

```bash
pip uninstall -y tensorflow tensorboard keras absl-py astunparse flatbuffers gast google_pasta grpcio h5py libclang ml_dtypes opt_einsum termcolor wrapt tensorboard-data-server
```

**Note**: Most MLX VLMs don't need TensorFlow. If a model actually requires it, you can allow TensorFlow with `MLX_VLM_ALLOW_TF=1`, but this may cause mutex crashes on Apple Silicon.

### Debug Mode

Use `--verbose` for detailed diagnostics:

```bash
python check_models.py --verbose
```

This provides:

- Detailed model loading information
- EXIF metadata extraction details
- Performance metric breakdowns
- Error stack traces
- Library version information

### Framework Detection and Automatic Blocking

The script **automatically** prevents TensorFlow, JAX, and Flax from loading to avoid conflicts with MLX on Apple Silicon:

- On startup, sets `TRANSFORMERS_NO_TF=1`, `TRANSFORMERS_NO_FLAX=1`, `TRANSFORMERS_NO_JAX=1` (unless you override with `MLX_VLM_ALLOW_TF=1`)
- PyTorch is allowed by default (some models require it, e.g., Phi-3-vision)
- Logs a warning if TensorFlow is detected but successfully blocked
- Also logs if `sentence-transformers` is present

**⚠️ About TensorFlow on Apple Silicon:**

If TensorFlow is installed, the script's automatic blocking (`TRANSFORMERS_NO_TF=1`) should prevent it from loading, avoiding mutex crashes. However, if you encounter the error `libc++abi: terminating due to uncaught exception of type std::__1::system_error: mutex lock failed: Invalid argument`:

1. **First try**: Verify the environment variable is set (the script does this automatically):

   ```bash
   export TRANSFORMERS_NO_TF=1
   python check_models.py
   ```

2. **If that fails**: Uninstall TensorFlow completely (recommended for MLX-only environments):

   ```bash
   pip uninstall -y tensorflow tensorboard keras
   ```

3. **If a model needs TensorFlow**: Set `MLX_VLM_ALLOW_TF=1` to allow it, but be aware this may cause crashes on Apple Silicon

**Why this matters**: TensorFlow's Abseil mutex implementation conflicts with MLX on macOS/ARM, causing crashes. Most MLX VLMs don't need TensorFlow.

## Notes

- **Platform**: Requires macOS with Apple Silicon for MLX support
- **Colors**: Uses ANSI color codes for CLI output (may not display correctly in all terminals)
- **Timeout**: Unix-only functionality (not available on Windows)
- **Security**: The `--trust-remote-code` flag allows arbitrary code execution from models
- **Performance**: First run may be slower due to model compilation and caching

## Project Structure

```text
mlx-vlm-check/
├── src/
│   ├── check_models.py      # Main script
│   ├── pyproject.toml       # Project configuration and dependencies  
│   ├── tools/               # Helper scripts
│   └── tests/               # Test suite
├── output/
│   ├── results.html        # Generated HTML report (default location)
│   └── results.md          # Generated Markdown report (default location)
├── docs/                   # Documentation
├── typings/                # Generated type stubs (git-ignored)
└── Makefile               # Root orchestration
```

**Output behavior**: By default, reports are written to `output/` (git-ignored). Override with `--output-html` / `--output-markdown`.

## Contributing

**For detailed contribution guidelines, coding standards, and project conventions, see:**

- [docs/CONTRIBUTING.md](../docs/CONTRIBUTING.md) - Setup, workflow, and PR process
- [docs/IMPLEMENTATION_GUIDE.md](../docs/IMPLEMENTATION_GUIDE.md) - Coding standards and architecture

### Developer Workflow (Makefile)

A `Makefile` at the repository root streamlines common tasks. It **auto-detects your active Python environment** and works with:

- **Virtual environments** (`venv`, `virtualenv`, `uv`, `poetry`, `pipenv`, etc.) - uses the active environment directly
- **Conda environments** - adapts based on active vs target environment:
  - **If target env is active**: runs commands directly
  - **If different conda env is active**: uses `conda run -n <target-env>`
  - **If no env is active**: uses `conda run -n <target-env>` (or system Python if conda unavailable)

The default conda target is `mlx-vlm`, but you can override it with `CONDA_ENV=your-env-name` for any make target.

**Recommendation**: Activate your environment first (e.g., `source .venv/bin/activate` or `conda activate mlx-vlm`) for best performance.

Key targets:

| Target | Purpose | Notes |
|--------|---------|-------|
| `make help` | Show all targets | Displays active vs target env. |
| `make install-dev` | Editable install with dev extras | Equivalent to changing into `src/` then `pip install -e .[dev]`. |
| `make install-markdownlint` | Install markdownlint-cli2 | Requires Node.js/npm. Optional for markdown linting. |
| `make install` | Runtime‑only editable install | No dev/test tooling. |
| `make bootstrap-dev` | Full dev setup | Installs Python deps + markdownlint + git hooks. |
| `make format` | Run `ruff format` | Applies canonical formatting. |
| `make lint` | Run `ruff check` (no fixes) | Fails on style violations. |
| `make lint-fix` | Run `ruff check --fix` | Auto‑fixes where safe. |
| `make typecheck` | Run `mypy` | Uses `src/pyproject.toml` config. |
| `make test` | Run pytest suite | Uses settings in `pyproject.toml`. |
| `make test-cov` | Pytest with coverage | Generates terminal + XML report. |
| `make quality` | Invoke integrated quality script | Wraps format + lint + mypy + markdownlint. |
| `make quality-strict` | Quality script (require tools, no stubs) | Adds `--require --no-stubs`. |
| `make run ARGS="..."` | Run the CLI script | Pass CLI args via `ARGS`. |
| `make smoke` | Fast help invocation | Sanity check only. |
| `make check` | format + lint + typecheck + test | Quick pre‑commit aggregate. |
| `make validate-env` | Check environment setup | Validates Python packages installed. |
| `make clean` | Remove caches / pyc | Safe cleanup. |

Examples:

```bash
# Install with dev dependencies (ruff, mypy, pytest, etc.)
make install-dev

# Format, then lint and auto-fix issues
make format
make lint-fix

# Type check and run tests
make typecheck
make test

# All core checks (use before committing)
make check

# Run CLI with arguments (quote ARGS value if it has spaces)
make run ARGS="--verbose --detailed-metrics --image sample.jpg --models microsoft/Florence-2-large"
```

Override variables on the fly:

```bash
make test CONDA_ENV=custom-mlx-env
make run CONDA_ENV=mlx-vlm ARGS="--verbose"
```

If you prefer manual commands, the traditional workflow still works:

1. `pip install -e ".[dev]"`
2. `ruff format check_models.py tests`
3. `ruff check --fix check_models.py tests`
4. `mypy --config-file pyproject.toml check_models.py`
5. `pytest -q`

#### Markdown Linting (Optional)

The project uses `markdownlint-cli2` to ensure consistent markdown formatting. This is **optional** but recommended for contributors editing documentation:

**Installation:**

```bash
# If you have Node.js/npm installed:
make install-markdownlint

# Or install manually:
npm install

# Or rely on npx (downloads on-demand):
# No installation needed - quality checks will use npx automatically
```

**Usage:**

- `make quality` - Automatically runs markdownlint if available (or via npx)
- Quality checks gracefully skip markdown linting if neither npm nor npx is available

**Requirements:**

- Node.js/npm (optional) - Install via `brew install node` on macOS or download from [nodejs.org](https://nodejs.org/)
- If npm is unavailable, the quality script will attempt to use `npx` as a fallback
- If neither is available, markdown linting is skipped with a warning

### Contribution Guidelines

- Keep patches focused; separate mechanical formatting changes from functional changes.
- Run `make check` (or at minimum `make test` and `make typecheck`) before opening a PR.
- Add or update tests when changing output formatting or public CLI flags.
- Prefer small helper functions over adding more branching to large blocks in `check_models.py`.
- Document new flags or output changes in this README (search for an existing section to extend rather than creating duplicates).
- For full conventions (naming, imports, dependency policy, quality gates), see `IMPLEMENTATION_GUIDE.md` in the repository root.

## Important Notes

- Timeout functionality requires UNIX (not available on Windows).
- For best results, ensure all dependencies are installed and models are downloaded/cached.

## License

MIT License - see LICENSE file for details.

## Quality checks and formatting

A small helper script runs formatting and static checks for this project.

- Location: `src/tools/check_quality.py`
- Defaults:
  - Targets only `check_models.py` unless paths are provided
  - Runs `ruff format` by default (skip with `--no-format`)
  - Runs `ruff check --fix` by default (skip fixing with `--no-fix`)
  - Runs `mypy` for type checking

Examples:

```bash
# Default: format + fixable lint + mypy on check_models.py
python tools/check_quality.py

# Skip auto-fix
python tools/check_quality.py --no-fix

# Skip formatting
python tools/check_quality.py --no-format

# Check multiple paths (from src/ directory)
python tools/check_quality.py . tools
```

Requirements: `ruff` and `mypy` (install with `pip install -e ".[dev]"`).
