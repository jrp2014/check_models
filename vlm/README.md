# MLX Vision Language Model Checker (`check_models.py`)

`check_models.py` is a focused benchmarking and inspection tool for MLX-compatible Vision Language Models (VLMs) on Apple Silicon. It loads one or more local / cached models, optionally derives a prompt from image metadata (EXIF + GPS), runs generation, and reports performance (tokens, speed, timings, memory) plus outputs in colorized CLI, HTML, and Markdown formats.

## Who is this for?

- Users (including MLX/MLX‑VLM/MLX‑LM developers) who want to run models on their own images and quickly see outputs and metrics: see the TL;DR below.
- Contributors who want to work on `check_models.py` itself and keep quality gates green: see the Contributors section near the end.

## TL;DR for Users

Defaults assume you have Hugging Face models in your local cache. The tool will discover cached VLMs automatically (unless you pass `--models`), carefully read EXIF metadata (including GPS/time when present) to enrich prompts, and write reports to `results.html` and `results.md` in the current directory.

Quickest start on Apple Silicon (Python 3.13):

```bash
cd vlm
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Optional extras for system metrics & tokenizers
pip install -e ".[extras]"

# Run against a folder of images (change the path to your images)
python check_models.py -f ~/Pictures/Processed -p "Describe this image."

# Try specific models
python check_models.py -f ~/Pictures/Processed -m mlx-community/nanoLLaVA mlx-community/llava-1.5-7b-hf

# Exclude a model from auto-discovered cache
python check_models.py -f ~/Pictures/Processed -e "microsoft/Phi-3-vision-128k-instruct"
```

Prefer Make?

```bash
make -C vlm bootstrap-dev
make check_models ARGS="-f ~/Pictures/Processed -p 'Describe this image.'"
```

Key defaults and parameters:

- Models: discovered from Hugging Face cache. Use `--models` for explicit IDs, `--exclude` to filter.
- Images: `-f/--folder` points to your images; default is `~/Pictures/Processed`.
- Folder behavior: when you pass a folder, the script automatically selects the most recently modified image file in that folder (hidden files are ignored).
- Reports: `results.html` and `results.md` are created in the current directory; override via `--output-html` and `--output-markdown`.
- Prompting: If `--prompt` isn’t provided, the tool can compose a metadata‑aware prompt from EXIF data when available (camera, time, GPS).
- Runtime: `--timeout 300`, `--max-tokens 500`, `--temperature 0.1` by default.
- Security: `--trust-remote-code=True` by default for Hub models; only use with trusted sources.

## Capabilities

- Auto‑discovers locally cached MLX VLMs (Hugging Face cache) or runs an explicit list.
- Captures structured performance: generation time, model load time, total time, token counts, tokens/sec, peak memory (GB).
- Extracts EXIF + GPS metadata (robust to partial corruption) for context.
- Provides compact console table + per‑model SUMMARY lines (machine parsable: `SUMMARY key=value ...`).
- Generates standalone HTML and GitHub‑friendly Markdown reports.
- Gracefully handles timeouts, load errors, and partial failures.

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

#### With uv (fastest)

```bash
# Install uv if you haven't already
pip install uv

# Create environment and install dependencies
uv venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate     # On Windows
uv pip install -e .  # run inside the vlm/ directory (root-level Makefile is a shim)

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
pip install -e .  # run inside vlm/

# For development
pip install -e ".[dev]"
```

#### With conda

```bash
# Create conda environment
conda create -n mlx-vlm python=3.12
conda activate mlx-vlm

# Install dependencies
pip install -e .  # run inside vlm/
```

#### Automated Setup (Recommended)

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

- **Python**: 3.12+ (3.12 is the tested baseline)
- **Operating System**: macOS with Apple Silicon (MLX is Apple‑Silicon specific)

## Dependencies

Why so slim? The runtime dependency set is intentionally minimized to only the
packages directly imported by `vlm/check_models.py`. Everything else that might
inference helpers, PyTorch stack) lives in optional extras. Benefits:

- Faster cold installs / CI setup
- Smaller transitive surface → fewer unexpected resolver conflicts
- Clearer signal when a new import is introduced (you must add it to
  `[project.dependencies]` or tests + sync tooling will fail)
If you add a new top‑level import in `check_models.py`, promote its package
from an optional group (or add it fresh) into the runtime `dependencies` array
and re-run the sync helper.

Runtime (installed automatically via `pip install -e .` when executed inside `vlm/`, or via `make install` from repo root):

| Purpose | Package | Minimum |
|---------|---------|---------|
| Core tensor/runtime | `mlx` | `>=0.29.1` |
| Vision‑language utilities | `mlx-vlm` | `>=0.0.9` |
| Image processing & loading | `Pillow` | `>=10.3.0` |
| Model cache / discovery | `huggingface-hub` | `>=0.23.0` |
| Reporting / tables | `tabulate` | `>=0.9.0` |
| Local timezone conversion | `tzlocal` | `>=5.0` |

Optional (enable additional features):

| Feature | Package | Source |
|---------|---------|--------|
| Extended system metrics (RAM/CPU) | `psutil` | `extras` |
| Fast tokenizer backends | `tokenizers` | `extras` |
| Language model utilities | `mlx-lm` | `extras` |
| Transformer model support | `transformers` | `extras` |
| PyTorch stack | `torch`, `torchvision`, `torchaudio` | `torch` extra |

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
- The `tools/update.sh` helper supports adding Torch via an environment flag: `INSTALL_TORCH=1 ./tools/update.sh`.
- Installing `sentence-transformers` isn’t necessary for this tool and may pull heavy backends into import paths; a heads‑up is logged if detected.
- Long embedded CSS / HTML lines are intentional (readability > artificial wrapping).
- Dependency versions in this README are automatically kept in sync with `pyproject.toml`; update the TOML first and reflect changes here.

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

- On startup, the script sets `TRANSFORMERS_NO_TF=1`, `TRANSFORMERS_NO_FLAX=1`, `TRANSFORMERS_NO_JAX=1` unless you explicitly opt in with `MLX_VLM_ALLOW_TF=1`.
- Torch is allowed by default (some models require it).
- A startup log warns if TensorFlow is installed but disabled; a heads‑up is also logged if `sentence-transformers` is present.
- Warning: Installing TensorFlow on macOS/Apple Silicon can trigger hangs during import (Abseil mutex). Prefer not installing it in MLX‑only environments.

## Notes

- **Platform**: Requires macOS with Apple Silicon for MLX support
- **Colors**: Uses ANSI color codes for CLI output (may not display correctly in all terminals)
- **Timeout**: Unix-only functionality (not available on Windows)
- **Security**: The `--trust-remote-code` flag allows arbitrary code execution from models
- **Performance**: First run may be slower due to model compilation and caching

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

### Developer Workflow (Makefile)

A `Makefile` at the repository root streamlines common tasks. It auto‑detects whether the target conda environment (`mlx-vlm` by default) is already active; if not, it transparently prefixes commands with `conda run -n mlx-vlm` so you don't have to manually activate it.

Key targets:

| Target | Purpose | Notes |
|--------|---------|-------|
| `make help` | Show all targets | Displays active vs target env. |
| `make install-dev` | Editable install with dev extras | Equivalent to changing into `vlm/` then `pip install -e .[dev]`. |
| `make install` | Runtime‑only editable install | No dev/test tooling. |
| `make format` | Run `ruff format` | Applies canonical formatting. |
| `make lint` | Run `ruff check` (no fixes) | Fails on style violations. |
| `make lint-fix` | Run `ruff check --fix` | Auto‑fixes where safe. |
| `make typecheck` | Run `mypy` | Uses `vlm/pyproject.toml` config. |
| `make test` | Run pytest suite | Uses settings in `pyproject.toml`. |
| `make test-cov` | Pytest with coverage | Generates terminal + XML report. |
| `make quality` | Invoke integrated quality script | Wraps format + lint + mypy. |
| `make quality-strict` | Quality script (require tools, no stubs) | Adds `--require --no-stubs`. |
| `make run ARGS="..."` | Run the CLI script | Pass CLI args via `ARGS`. |
| `make smoke` | Fast help invocation | Sanity check only. |
| `make check` | format + lint + typecheck + test | Quick pre‑commit aggregate. |
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
2. `ruff format vlm/check_models.py vlm/tests`
3. `ruff check --fix vlm/check_models.py vlm/tests`
4. `mypy --config-file vlm/pyproject.toml vlm/check_models.py`
5. `pytest -q`

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

- Location: `vlm/tools/check_quality.py`
- Defaults:
  - Targets only `vlm/check_models.py` unless paths are provided
  - Runs `ruff format` by default (skip with `--no-format`)
  - Runs `ruff check --fix` by default (skip fixing with `--no-fix`)
  - Runs `mypy` for type checking

Examples:

```bash
# Default: format + fixable lint + mypy on check_models.py
python vlm/tools/check_quality.py

# Skip auto-fix
python vlm/tools/check_quality.py --no-fix

# Skip formatting
python vlm/tools/check_quality.py --no-format

# Check multiple paths
python vlm/tools/check_quality.py vlm tools
```

Requirements: `ruff` and `mypy` (install with `pip install -e ".[dev]"`).
