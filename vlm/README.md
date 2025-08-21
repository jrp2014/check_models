# MLX Vision Language Model Checker (`check_models.py`)

`check_models.py` is a focused benchmarking and inspection tool for MLX-compatible Vision Language Models (VLMs) on Apple Silicon. It loads one or more local / cached models, optionally derives a prompt from image metadata (EXIF + GPS), runs generation, and reports performance (tokens, speed, timings, memory) plus outputs in colorized CLI, HTML, and Markdown formats.

## Capabilities

* Auto‑discovers locally cached MLX VLMs (Hugging Face cache) or runs an explicit list.
* Captures structured performance: generation time, model load time, total time, token counts, tokens/sec, peak memory.
* Extracts EXIF + GPS metadata (robust to partial corruption) for context.
* Provides compact console table + per‑model SUMMARY lines (machine parsable: `SUMMARY key=value ...`).
* Generates standalone HTML and GitHub‑friendly Markdown reports.
* Gracefully handles timeouts, load errors, and partial failures.

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

## Installation and Environment Setup

### Using `pyproject.toml` (recommended)

This project includes a `pyproject.toml` file that defines all dependencies and can be used with modern Python package managers.

#### With uv (fastest)

```bash
# Install uv if you haven't already
pip install uv

# Create environment and install dependencies
uv venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate     # On Windows
uv pip install -e .

# For development with additional tools
uv pip install -e ".[dev]"
```

#### With pip

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate     # On Windows

# Install the package
pip install -e .

# For development
pip install -e ".[dev]"
```

#### With conda

```bash
# Create conda environment
conda create -n mlx-vlm python=3.12
conda activate mlx-vlm

# Install dependencies
pip install -e .
```

### Manual Installation

If you prefer to install dependencies manually (ensure these match `pyproject.toml`):

```bash
pip install "mlx>=0.14.0" "mlx-vlm>=0.0.9" "Pillow>=10.0.0" "huggingface-hub>=0.23.0" "tabulate>=0.9.0" "tzlocal>=5.0"
```

## Requirements

* **Python**: 3.12+ (3.12 is the tested baseline)
* **Operating System**: macOS with Apple Silicon (MLX is Apple‑Silicon specific)
* **Models**: MLX-compatible Vision Language Models (fetched via Hugging Face)

## Dependencies

Runtime (installed automatically via `pip install -e .`):

| Purpose | Package | Minimum |
|---------|---------|---------|
| Core tensor/runtime | `mlx` | `>=0.14.0` |
| Vision‑language utilities | `mlx-vlm` | `>=0.0.9` |
| Image EXIF & loading | `Pillow` | `>=10.0.0` |
| Model cache / discovery | `huggingface-hub` | `>=0.23.0` |
| Tabular console & report formatting | `tabulate` | `>=0.9.0` |
| Local timezone conversion | `tzlocal` | `>=5.0` |

Optional (enable additional features if present):

| Feature | Package | Notes |
|---------|---------|-------|
| Additional model families / loaders | `transformers` | Via `extras`; use up-to-date (>=4.41.0) for latest multimodal fixes |
| Alternative MLX language utilities | `mlx-lm` | Via `extras` group |
| Fast tokenizer backends | `tokenizers` | Installed automatically with `transformers` (no manual install) |
| Extended system metrics (RAM/CPU) | `psutil` | Included in `extras`; optional for hardware block |

Development / QA:

| Purpose | Package |
|---------|---------|
| Linting & formatting checks | `ruff` |
| Static type checking | `mypy` |
| Testing | `pytest`, `pytest-cov` |

### Minimal Install (runtime only)

```bash
pip install "mlx>=0.14.0" "mlx-vlm>=0.0.9" "Pillow>=10.0.0" "huggingface-hub>=0.23.0" "tabulate>=0.9.0" "tzlocal>=5.0"
```

### With Optional Extras

The `extras` group in `pyproject.toml` pulls in `transformers`, `mlx-lm`, and `psutil`:

```bash
pip install -e ".[extras]"
```

### Full Development Environment

```bash
pip install -e ".[dev,extras]"
```

Notes:
 
* `psutil` is optional (installed with `extras`); if absent the extended Apple Silicon hardware section omits RAM/cores.
* `tokenizers` is a transitive dependency of `transformers`; you don't need to list or install it separately.
* `transformers` moves quickly for multimodal / vision improvements. Keeping it updated (within the declared range `>=4.41.0,<5`) is recommended:
  * Upgrade: `pip install -U transformers`
  * If a 5.x release appears, test locally before relaxing the `<5` upper bound.
  * Newer releases often fix chat template, processor, and safety issues relevant to VLMs.
* `system_profiler` is a macOS built-in (no install needed) used for GPU name / core info.
* Torch is supported and can be installed when you need it for specific models; the script does not block Torch.
* Installing `sentence-transformers` isn’t necessary for this tool and may pull heavy backends into import paths; a heads‑up is logged if detected.
* Long embedded CSS / HTML lines are intentional (readability > artificial wrapping).
* Dependency versions in this README are automatically kept in sync with `pyproject.toml`; update the TOML first and reflect changes here.

## Usage

### Quick Start

```bash
# Run across all cached models with a custom prompt
python check_models.py -p "What is the main object in this image?"

# Explicit model list (skips cache discovery)
python check_models.py -m mlx-community/nanoLLaVA mlx-community/llava-1.5-7b-hf

# Exclude specific models from the automatic cache scan
python check_models.py -e mlx-community/problematic-model other/model

# Combine explicit list with exclusions
python check_models.py -m model1 model2 model3 -e model2

# Verbose (debug) mode for detailed logs
python check_models.py -v
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

## Command Line Reference

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-f`, `--folder` | Path | `~/Pictures/Processed` | Folder of images to process (non‑recursive). |
| `--output-html` | Path | `results.html` | HTML report output file. |
| `--output-markdown` | Path | `results.md` | Markdown report output file. |
| `-m`, `--models` | list[str] | (none) | Explicit model IDs/paths; disables cache discovery. |
| `-e`, `--exclude` | list[str] | (none) | Models to exclude (applies to cache scan or explicit list). |
| `--trust-remote-code` | flag | `True` | Allow custom code from Hub models (SECURITY RISK). |
| `-p`, `--prompt` | str | (auto) | Custom prompt; if omitted a metadata‑aware prompt may be used. |
| `-x`, `--max-tokens` | int | 500 | Max new tokens to generate. |
| `-t`, `--temperature` | float | 0.1 | Sampling temperature. |
| `--timeout` | float | 300 | Operation timeout (seconds) for model execution. |
| `-v`, `--verbose` | flag | `False` | Enable verbose + debug logging. |

### Selection Logic

1. No selection flags: run all cached VLMs.
2. `--models` only: run exactly that list.
3. `--exclude` only: run cached minus excluded.
4. `--models` + `--exclude`: intersect explicit list then subtract exclusions.

The script warns about exclusions that don't match any candidate model.

## Output Formats

### CLI Output

Real-time colorized output showing:

* Model processing progress with success/failure indicators
* Performance metrics (tokens/second, memory usage, timing)
* Generated text preview
* Error diagnostics for failed models
* Final performance summary table

Color conventions:

* Identifiers (file/folder paths and model names) are shown in magenta for quick scanning.
* Failures are shown in red; the compact CLI table also highlights failed model names in red.

### HTML Report

Professional report featuring:

* Executive summary with test parameters
* Interactive performance table with sortable columns
* Model outputs and diagnostics
* System information and library versions
* Failed rows are highlighted in red for quick identification
* Responsive design for mobile viewing

### Markdown Report

GitHub-compatible format with:

* Performance metrics in table format
* Model outputs
* System and library version information
* Easy integration into documentation

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

* **Token Metrics**: Prompt tokens, generation tokens, total processing
* **Speed Metrics**: Tokens per second for prompt processing and generation
* **Memory Usage**: Peak memory consumption during processing
* **Timing**: Total processing time per model
* **Success Rate**: Model success/failure statistics
* **Error Analysis**: Detailed error reporting and diagnostics

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

### Debug Mode

Use `--verbose` for detailed diagnostics:

```bash
python check_models.py --verbose
```

This provides:

* Detailed model loading information
* EXIF metadata extraction details
* Performance metric breakdowns
* Error stack traces
* Library version information

### TensorFlow-related hangs on macOS/Apple Silicon

If the process appears to stall during imports with TensorFlow/Abseil mutex lines (e.g., from a `sample` stack trace), Transformers may be auto-importing TensorFlow even though MLX doesn’t need it. You can prevent this in two ways:

1. Set environment flags (recommended):

```bash
export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_NO_FLAX=1
export TRANSFORMERS_NO_JAX=1
```

1. Uninstall TensorFlow from the environment to avoid accidental imports (recommended for MLX-only envs):

```bash
pip uninstall -y tensorflow tensorflow-macos tensorflow-metal
```

Script behavior:

* On startup, the script sets `TRANSFORMERS_NO_TF=1`, `TRANSFORMERS_NO_FLAX=1`, `TRANSFORMERS_NO_JAX=1` unless you explicitly opt in with `MLX_VLM_ALLOW_TF=1`.
* Torch is allowed by default (some models require it).
* A startup log warns if TensorFlow is installed but disabled; a heads‑up is also logged if `sentence-transformers` is present.
* Warning: Installing TensorFlow on macOS/Apple Silicon can trigger hangs during import (Abseil mutex). Prefer not installing it in MLX‑only environments.

## Notes

* **Platform**: Requires macOS with Apple Silicon for MLX support
* **Colors**: Uses ANSI color codes for CLI output (may not display correctly in all terminals)
* **Timeout**: Unix-only functionality (not available on Windows)
* **Security**: The `--trust-remote-code` flag allows arbitrary code execution from models
* **Performance**: First run may be slower due to model compilation and caching

## Project Structure

```text
mlx-vlm-check/
├── check_models.py      # Main script
├── pyproject.toml       # Project configuration and dependencies  
├── README.md           # This file
└── results.html        # Generated HTML report (after running)
└── results.md          # Generated Markdown report (after running)
```

## Contributing

1. Install development dependencies: `pip install -e ".[dev]"`
2. Run linting: `ruff check .`
3. Run type checking: `mypy check_models.py`
4. Test your changes thoroughly

## Important Notes

* Timeout functionality requires UNIX (not available on Windows).
* For best results, ensure all dependencies are installed and models are downloaded/cached.

## License

MIT License - see LICENSE file for details.
