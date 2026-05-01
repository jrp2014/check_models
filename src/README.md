# MLX Vision Language Model Checker (`check_models.py`)

`check_models.py` is a comprehensive benchmarking and inspection tool designed for MLX-compatible Vision Language Models (VLMs) on Apple Silicon. It streamlines the process of validating model performance, quality, and resource usage across your local model collection.

> [!NOTE]
> This tool runs MLX-format Vision-Language Models hosted on the [Hugging Face Hub](https://huggingface.co). By default, it discovers and runs all models found in your local Hugging Face Hub cache, making it effortless to benchmark your entire library.

## Who is this for?

- **Users & Researchers**: Quickly benchmark models on your own images, compare performance (TPS, memory), and verify output quality without writing code.
- **Developers**: Validate model conversions, debug quantization issues, and ensure regression testing for MLX/MLX-VLM improvements.

## Quick Start

Get up and running immediately with your cached models.

### 1. Installation

The fastest way to start is using the automated setup script (requires Conda):

```bash
# Sets up a 'mlx-vlm' environment with Python 3.13 and all dependencies
bash tools/setup_conda_env.sh
conda activate mlx-vlm
```

*(See [Installation Details](#installation-and-environment-setup) for manual pip/uv methods)*

### 2. Run Your First Check

By default, the tool scans your Hugging Face cache for compatible models and runs them against the most recently modified image in `~/Pictures/Processed`.

```bash
# Run all cached models against the most recent image in your folder
python -m check_models --folder ~/Pictures/Processed --prompt "Describe this image."

# Run against a specific image file
python -m check_models --image ~/Pictures/Processed/sample.jpg
```

### 3. Common Commands

```bash
# Test specific models (downloads them if needed)
python -m check_models --models mlx-community/nanoLLaVA mlx-community/llava-1.5-7b-hf

# Exclude a problematic model from the batch
python -m check_models --exclude "microsoft/Phi-3-vision-128k-instruct"

# Run with detailed debug logging
python -m check_models --verbose

# Dry run: validate setup and show what would run without invoking models
python -m check_models --dry-run
```

**Python Version**: 3.13+ is recommended and tested.

## Capabilities

- **Model Discovery**: Auto-discovers locally cached MLX VLMs from Hugging Face cache or processes explicit model list with `--models`
- **Selection Control**: Use `--exclude` to filter models from cache scan or explicit list
- **Folder Mode**: Automatically selects most recently modified image from specified folder
- **Metadata Extraction**: Multi-source metadata: EXIF + GPS + IPTC keywords/caption + XMP (dc:subject, dc:title) + Windows XP keywords, with fail-soft strategy for partially corrupt data
- **Smart Prompting**: Generates structured cataloguing prompts (Title/Description/Keywords) that verify metadata against clearly visible image content, avoid speculation, and compact long metadata fields/keyword lists to keep prompt size manageable; `--prompt` overrides
- **Performance Metrics**:
  - Timing: generation_time, model_load_time, total_time
  - Detailed verbose timing: input validation, prompt prep, cleanup, first-token latency, stop reason (when available)
  - Tokens: total, prompt, generated with tokens/sec
  - Memory: peak, active delta, cached delta (GB)
- **Structured Logging**: Formatter-driven styling with LogStyles for consistent CLI output
- **Multiple Output Formats**:
  - **CLI**: Colorized with compact or detailed metrics modes
  - **HTML**: Standalone report with inline CSS, failed row highlighting
  - **Markdown**: GitHub-compatible summary plus standalone gallery Markdown artifact
  - **TSV/JSONL**: Machine-readable exports for downstream analysis
- **Error Handling**: Per-model isolation with detailed diagnostics; graceful timeout/failure handling
- **Machine Parsable**: SUMMARY lines with `key=value` format for automation
- **Visual Hierarchy**: Emoji prefixes, tree-structured metrics, wrapped text output

## Feature Highlights

| Area | Notes |
| ---- | ----- |
| Model discovery | Scans Hugging Face cache; explicit `--models` overrides. |
| Selection control | `--exclude` works with cache scan or explicit list. |
| Prompting | `--prompt` overrides; otherwise structured cataloguing prompt with IPTC/XMP keyword seeding. |
| Performance | generation_time, model_load_time, total_time, token counts, TPS, peak memory. |
| Reporting | CLI (color), HTML (standalone), Markdown (GitHub). |
| Robustness | Per‑model isolation; failures logged; SUMMARY lines for automation. |
| Timeout | Signal‑based (UNIX) manager; configurable per run. |
| Output preview | Non‑verbose mode still shows wrapped generated text (80 cols). |
| Metrics modes | Compact (default) or expanded with `--detailed-metrics`; the flag is ignored unless `--verbose` is also set, and detailed mode includes extra phase timings when available. |

## Installation and Environment Setup

### Automated Setup (Recommended)

For the easiest setup, use the provided shell script that automates the entire conda environment creation:

```bash
# Create environment with default name 'mlx-vlm'
bash tools/setup_conda_env.sh

# Activate
conda activate mlx-vlm
```

The script handles Python 3.13 setup, dependencies, and optional PyTorch support.

For clean machines and normal project use, this conda workflow is the supported path.

### Manual Installation

If you prefer to create the environment manually, use conda and run the package
install commands from the `src/` directory.

<details>
<summary>Click to view manual conda setup</summary>

```bash
cd src
conda create -n mlx-vlm python=3.13
conda activate mlx-vlm
pip install -e .
```

</details>

### Optional Dependencies

Unless noted otherwise, the `pip install -e ...` commands below assume you are
running them from `src/`. From the repository root, prefer `make install`,
`make dev`, or `make -C src install-torch`.

Install the core runtime or add optional model coverage as needed:

```bash
# Runtime only (includes mlx, mlx-lm, and mlx-vlm)
pip install -e .

# Add optional extras and torch-backed loaders for broader model coverage
pip install -e ".[extras,torch]"

# Install only the PyTorch subset if you do not need the other extras
pip install -e ".[torch]"

# Install everything for development
pip install -e ".[dev,extras,torch]"
```

## Usage Guide

### Basic Execution

The tool is flexible: it can scan your cache, run specific models, or process single images.

```bash
# 1. Run all cached models against a folder
python -m check_models --folder ~/Pictures/Processed

# 2. Run a specific model against a single image
python -m check_models --image test.jpg --models mlx-community/nanoLLaVA

# 3. Run with a custom prompt
python -m check_models --image test.jpg --prompt "Detailed caption."
```

### Advanced Examples

```bash
# Run across all cached models with a custom prompt
python -m check_models -f ~/Pictures/Processed -p "What is the main object in this image?"

# Explicit model list (skips cache discovery)
python -m check_models -f ~/Pictures/Processed -m mlx-community/nanoLLaVA mlx-community/llava-1.5-7b-hf

# Repeated -m/--models flags also accumulate model IDs
python -m check_models -f ~/Pictures/Processed \
  -m mlx-community/nanoLLaVA \
  -m mlx-community/llava-1.5-7b-hf mlx-community/Qwen2-VL-2B-Instruct

# Exclude specific models from the automatic cache scan
python -m check_models -f ~/Pictures/Processed -e mlx-community/problematic-model other/model

# Repeated -e/--exclude flags accumulate exclusions
python -m check_models -f ~/Pictures/Processed \
  -e Qwen/Qwen3-VL-2B-Instruct \
  -e mlx-community/Qwen3-VL-2B-Thinking-bf16

# Repeated --eos-tokens flags accumulate stop tokens
python -m check_models --image test.jpg --models mlx-community/nanoLLaVA \
  --eos-tokens '</think>' \
  --eos-tokens '\n' '<END>'

# Combine explicit list with exclusions
python -m check_models -f ~/Pictures/Processed -m model1 model2 model3 -e model2

# Verbose (debug) mode for detailed logs
python -m check_models -f ~/Pictures/Processed -v
```

<details>
<summary>Click for more complex examples</summary>

```bash
# Full benchmark run with HTML/Markdown reports
python -m check_models \
  --folder ~/Pictures/TestImages \
  --exclude "microsoft/Phi-3-vision-128k-instruct" \
  --prompt "Provide a detailed caption for this image" \
  --max-tokens 200 \
  --temperature 0.1 \
  --timeout 600 \
  --output-html ~/reports/vlm_benchmark.html \
  --output-markdown ~/reports/vlm_benchmark.md \
  --verbose

# Memory optimization for large models (4-bit KV cache)
python -m check_models \
  --folder ~/Pictures \
  --lazy-load \
  --max-kv-size 4096 \
  --kv-bits 4

# Sampling control (Nucleus sampling + Repetition penalty)
python -m check_models \
  --folder ~/Pictures \
  --top-p 0.9 \
  --repetition-penalty 1.2 \
  --repetition-context-size 50
```

</details>

### Understanding the Output

The tool generates multiple report formats in `output/` by default:

- **CLI**: Real-time colorized progress and metrics.
- **HTML** (`reports/results.html`): Interactive table with sortable columns and failed row highlighting.
- **Markdown** (`reports/results.md`): GitHub-compatible summary for documentation, with links to the canonical log and review artifacts.
- **Gallery Markdown** (`reports/model_gallery.md`): GitHub-compatible review artifact with image metadata, the full prompt, and one full-output section per model.
- **Review Markdown** (`reports/review.md`): Short automated digest grouped by likely owner and user-facing utility bucket.
- **TSV/JSONL** (`reports/results.tsv`, `results.jsonl`): Machine-readable formats for analysis.
- **Diagnostics** (`reports/diagnostics.md`): Failure-focused and compatibility-focused issue report (generated when failures, harness issues, or preflight compatibility warnings are present).
- **Log** (`check_models.log`): Canonical comprehensive run artifact, including the full per-model review block and full output/captured failure output.
- **History** (`results.history.jsonl`): Append-only run history for regressions/recoveries.
- **Issue templates** (`issues/`): Ready-to-file GitHub issue markdown for clustered crashes and harness problems (generated when failures are present).
- **Repro bundles** (`repro_bundles/`): JSON reproduction bundles per failed model, containing error details, CLI args, and environment for reproducibility.

The main Markdown report stays brief and points readers to `check_models.log`, `review.md`, and `model_gallery.md` for the full automated review and output evidence.

### Metrics Explained

- **TPS (Tokens/Sec)**: Speed of generation. Higher is better.
- **Peak Memory**: Maximum RAM used. Critical for hardware sizing.
- **Load Time**: Time to load weights into memory.
- **Tokens**: Breakdown of Prompt (input) vs Generated (output) counts.


> [!TIP]
> **Memory Units**: All memory metrics are normalized to GB (decimal).
> **Token Counts**: `tokens(total/prompt/gen)` shows the full breakdown.
> [!IMPORTANT]
> **Image Resolution vs Vision Encoder Input**: VLMs **never see your full-resolution image**. Every model downsamples the input to fit its vision encoder's fixed size before processing. A 50 MP image (8627×5760) becomes a ~0.2 MP thumbnail at 448×448 — a 250× reduction. This means fine details (text, small objects, texture) are lost before the model even starts.
> Typical input resolutions by model family:

| Resolution | Models |
| --- | --- |
| 224×224 | nanoLLaVA |
| 384–448 | PaliGemma2-448, Phi-3.5, FastVLM, LFM2 |
| 560–768 | SmolVLM, Idefics3, Molmo |
| 896–1344 | PaliGemma2-896, Qwen2-VL, Qwen3-VL (dynamic tiling) |

> The Qwen VL family uses **dynamic resolution** with tile-based encoding, processing the image in multiple patches at higher fidelity — which is why their prompt token counts are much larger (~16,000 vs ~500 for PaliGemma2). If a model reports "image too small to see," it is being honest about what it actually received.


### Configuration & Parameters

#### Controlling Repetitive Output

While MLX-VLM doesn't explicitly document these in all examples, the underlying `generate()` function (inherited from MLX-LM) supports **repetition penalty** parameters that can significantly reduce or eliminate repetitive text generation. These parameters work by penalizing tokens that have already appeared in recent context:

- `--repetition-penalty <float>`: Penalty factor for repeating tokens (must be ≥ 1.0). Higher values more strongly discourage repetition. Common range: 1.0-1.2. Default: `None` (no penalty).
- `--repetition-context-size <int>`: Number of recent tokens to check for repetition. Smaller values (10-20) only penalize immediate loops; larger values (50-100) prevent long-range repetition. Default: 20.

**Example usage**:

```bash
# Moderate penalty to reduce repetition
python -m check_models --image photo.jpg --repetition-penalty 1.1

# Strong penalty with larger context window
python -m check_models --image photo.jpg --repetition-penalty 1.15 --repetition-context-size 50
```

**How it works** (from MLX source): During generation, the model maintains a sliding window of the last N tokens (`repetition_context_size`). For each new token prediction, logits of tokens that appear in this window are divided by the `repetition_penalty` factor, making them less likely to be selected. This mechanism operates at the token level during sampling, before temperature is applied.

**Trade-offs**:

- ✅ **Benefit**: Dramatically reduces repetitive loops, hallucinated lists, and redundant output
- ⚠️ **Risk**: Overly aggressive penalties (>1.2) may harm output quality, forcing the model to use awkward synonyms or break natural repetition (e.g., proper nouns, technical terms)
- 💡 **Tip**: Start with 1.05-1.1 and increase gradually while monitoring quality flags in the output

The `check_models` tool's quality analysis detects repetition post-generation and flags it in reports. Using these parameters proactively can prevent repetitive output before it occurs, saving generation time and improving results.

#### KV Cache Quantization (Memory Optimization)

Vision-language models maintain a **key-value (KV) cache** during text generation to avoid recomputing attention for previous tokens. For long sequences or large models, this cache can consume significant memory. MLX-VLM supports KV cache quantization to reduce memory usage with minimal impact on output quality.

### Parameters

- `--max-kv-size <int>`: Maximum number of tokens to store in KV cache. Limits memory for very long sequences. Default: `None` (unlimited).
- `--kv-bits <number>`: Quantize KV cache instead of using full precision (typically 16-bit). Uniform quantization supports `2`, `3`, `4`, `5`, `6`, or `8`; fractional values such as `3.5` use upstream TurboQuant automatically. Default: `None` (no quantization).
- `--kv-quant-scheme <uniform|turboquant>`: Select the upstream KV quantization backend. Default: `uniform`.
- `--kv-group-size <int>`: Group size for quantization (larger = more compression, less accuracy). Default: `64`.
- `--quantized-kv-start <int>`: Token position to start quantization. Use `0` to quantize from the beginning, or a larger value to keep early tokens (e.g., system prompts) at full precision. Default: `0`.

### How It Works

From the MLX and MLX-VLM sources: uniform KV quantization uses MLX affine bit widths (`2`, `3`, `4`, `5`, `6`, or `8`) with `kv_group_size` token groups, while TurboQuant supports integer and `.5` bit widths and automatically handles fractional values such as `3.5` as lower-bit keys plus higher-bit values. Both reduce memory while maintaining most of the model's generation quality.

### Example Usage

```bash
# Moderate 8-bit quantization for 2× memory savings
python -m check_models --image photo.jpg --kv-bits 8

# Match upstream TurboQuant KV-cache handling
python -m check_models --image photo.jpg --kv-bits 3.5 --kv-quant-scheme turboquant

# Aggressive 4-bit quantization with larger groups (4× compression)
python -m check_models --image photo.jpg --kv-bits 4 --kv-group-size 128

# Quantize only after first 512 tokens (preserve system prompt precision)
python -m check_models --image photo.jpg --kv-bits 8 --quantized-kv-start 512

# Cap cache size for extremely long outputs
python -m check_models --image photo.jpg --max-kv-size 4096 --kv-bits 8
```

### When to Use

**Use KV quantization if:**

- ✅ Testing large models (>10B parameters) on limited RAM
- ✅ Generating long sequences (>1000 tokens)
- ✅ Running multiple models in parallel
- ✅ Encountering OOM (out of memory) errors

**Skip quantization if:**

- ❌ Models are small (<7B parameters)
- ❌ Sequences are short (<500 tokens)
- ❌ You need maximum quality for critical tasks

### Trade-offs

- **4-bit**: 75% memory reduction, slight quality degradation (noticeable in complex reasoning)
- **8-bit**: 50% memory reduction, minimal quality impact (recommended starting point)
- **Group size**: Larger groups save more memory but reduce precision; 64-128 is optimal for most cases

#### Temperature and Sampling

- `--temperature <float>`: Controls randomness in generation. `0.0` = deterministic (argmax), `1.0` = high diversity, `>1.0` = more random. Default: `0.0`.
- `--top-p <float>`: Nucleus sampling threshold. Only considers tokens whose cumulative probability is ≤ `top_p`. Range: `0.0-1.0`. Default: `1.0` (disabled).

These control the sampling strategy during generation. Higher temperature increases variety but can produce less coherent outputs. Top-p sampling (nucleus sampling) focuses on the most probable tokens.

**Example**:

```bash
# Deterministic output (default)
python -m check_models --image photo.jpg --temperature 0.0

# Balanced creativity
python -m check_models --image photo.jpg --temperature 0.7 --top-p 0.9

# Maximum diversity (risky)
python -m check_models --image photo.jpg --temperature 1.5 --top-p 0.95
```

#### Generation Control

- `--max-tokens <int>`: Maximum number of tokens to generate. Prevents runaway generation. Default: `500`.
- `--timeout <float>`: Timeout in seconds for each model's generation. Useful for identifying slow/hanging models. Default: `300.0` (5 minutes).

**Example**:

```bash
# Short captions only
python -m check_models --image photo.jpg --max-tokens 100

# Strict timeout for batch testing
python -m check_models --folder ~/images --timeout 120
```

#### Trust Remote Code

- `--trust-remote-code` (default): Allow execution of custom modeling code from Hugging Face repos. **Security risk** - only use with trusted models.
- `--no-trust-remote-code`: Disable custom code execution for maximum security.

Some models require custom Python code for their architecture. This flag enables loading that code.

**Examples**:

```bash
# Enable for models like Qwen or custom architectures (default behavior)
python -m check_models --image photo.jpg --models mlx-community/Qwen2-VL-7B-Instruct --trust-remote-code

# Disable for security (may cause some models to fail)
python -m check_models --image photo.jpg --no-trust-remote-code
```

> [!WARNING]
> **Security Risk**: `--trust-remote-code` (the default) executes arbitrary Python from the model repo. Use `--no-trust-remote-code` when running untrusted models.

#### Model Version & Adapter

- `--revision <str>`: Pin the model to a specific branch, tag, or commit SHA from the Hugging Face repo. Useful for reproducing results or avoiding regressions when a model updates. Default: `None` (latest revision on main/default branch).
- `--adapter-path <str>`: Path to a LoRA adapter directory to apply on top of the base model. Passed through to `mlx_vlm.utils.load(adapter_path=...)`. Default: `None` (no adapter).

**Examples**:

```bash
# Pin to a specific commit for reproducibility
python -m check_models --image photo.jpg --models mlx-community/Qwen2-VL-7B-Instruct \
  --revision abc1234

# Apply a LoRA fine-tune
python -m check_models --image photo.jpg --models mlx-community/nanoLLaVA \
  --adapter-path ~/adapters/my-lora

# Combine: specific revision + LoRA adapter
python -m check_models --image photo.jpg --models mlx-community/nanoLLaVA \
  --revision v1.0 --adapter-path ~/adapters/my-lora
```

### Environment Variables

Several behaviors can be customized via environment variables (useful for CI/automation):

| Variable | Purpose | Default | Example |
| -------- | ------- | ------- | ------- |
| `MLX_VLM_WIDTH` | Force CLI output width (columns) | Auto-detect terminal | `MLX_VLM_WIDTH=120` |
| `NO_COLOR` | Disable ANSI colors in output | Not set (colors enabled) | `NO_COLOR=1` |
| `FORCE_COLOR` | Force ANSI colors even in non-TTY | Not set | `FORCE_COLOR=1` |
| `TOKENIZERS_PARALLELISM` | Disable tokenizer parallelism warnings | `false` | `TOKENIZERS_PARALLELISM=true` |
| `MLX_METAL_JIT` | Optional `tools/update.sh` override (`MLX_METAL_JIT`) | Unset (uses MLX default `OFF`, pre-built kernels) | `MLX_METAL_JIT=ON` for runtime JIT |

**Examples**:

```bash
# Force wider output for CI logs
MLX_VLM_WIDTH=120 python -m check_models --folder ~/Pictures

# Disable colors for log file capture
NO_COLOR=1 python -m check_models > output.log 2>&1
```

## Git Hygiene and Caches

This repo excludes ephemeral caches and local environments via `.gitignore`. Common exclusions include `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`, `.venv/`, and editor folders like `.vscode/`. Do not commit large model caches (e.g., Hugging Face) to the repository.

## Pre-commit (Optional)

Recommended workflow:

```bash
cd src
python -m tools.install_precommit_hook
```

This installs the repo's custom git hooks directly and is the path used by
`tools/setup_conda_env.sh` when you opt into development dependencies.

Alternative workflow:

- `pre-commit` framework:

```bash
pip install pre-commit
pre-commit install
```

  This installs both commit-stage and pre-push hooks from the checked-in
  `.pre-commit-config.yaml`. The commit hook runs staged-file hygiene only; the
  push hook runs fast static checks plus the non-slow/non-e2e pytest subset.
  If you switch between workflows, rerun your preferred installer because both
  write to `.git/hooks/`.

Run the push-stage gate manually with:

```bash
pre-commit run --hook-stage pre-push --all-files
```

The commit-stage hook is intentionally staged-file based; run it by making a
normal commit, or call `bash src/tools/run_commit_hygiene.sh` directly after
staging files.


### Manual Installation

If you prefer to install dependencies manually (ensure these match `pyproject.toml`):

<!-- MANUAL_INSTALL_START -->
```bash
pip install "defusedxml>=0.7.1" "huggingface-hub[torch,typing]>=1.10.1" "mlx>=0.31.2" "mlx-lm>=0.31.3" "mlx-vlm>=0.4.4" "packaging>=26.0" "Pillow[xmp]>=10.3.0" "PyYAML>=6.0" "tabulate>=0.9.0" "transformers>=5.5.3" "wcwidth>=0.2.13"
```
<!-- MANUAL_INSTALL_END -->

## Requirements

- **Python**: 3.13+ (3.13 is the tested baseline)
- **Operating System**: macOS with Apple Silicon (MLX is Apple‑Silicon specific)



### Advanced Configuration

The tool uses a YAML configuration file to define thresholds for quality checks (hallucination, repetition, verbosity).

- **Default Config**: The tool ships with a bundled default `quality_config.yaml`. In this source tree, the canonical copy lives at `src/check_models_data/quality_config.yaml`.
- **Custom Config**: You can provide your own config file via `--quality-config path/to/config.yaml`.

**Key Configurable Areas:**

- **Repetition**: Thresholds for token and phrase repetition.
- **Hallucination**: Keywords and patterns that suggest hallucinated content (e.g., "based on the chart" when no chart exists).
- **Verbosity**: Limits on output length and meta-commentary patterns.
- **Formatting**: Rules for markdown headers, bullet points, and table structures.
- **Prompt Compaction**: Limits for metadata hints injected into the default prompt (`prompt_title_max_chars`, `prompt_description_max_chars`, `prompt_keyword_max_items`, `prompt_keyword_item_max_chars`).

See `src/check_models_data/quality_config.yaml` for the full schema and default values.

### Development Tools

The `src/tools/` directory contains scripts useful for development and verification:

- **Smoke Testing**: For quick verification, you can use the standard `mlx-vlm` CLI:

  ```bash
  python -m mlx_vlm.generate --model mlx-community/nanoLLaVA --image test.jpg
  ```

  Or refer to the official [test_smoke.py](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/tests/test_smoke.py) script.

- **E2E Smoke Tests**: The test suite includes end-to-end tests that run actual model inference:

  ```bash
  # Run E2E tests (requires cached model: qnguyen3/nanoLLaVA)
  pytest tests/test_e2e_smoke.py -v

  # Skip slow tests for quick iteration
  pytest -m "not slow"

  # Run all tests including E2E
  pytest tests/ -v
  ```

  > [!NOTE]
  > E2E tests require `qnguyen3/nanoLLaVA` to be cached. Run a quick inference first to download it:
  > `python -m check_models --models qnguyen3/nanoLLaVA --max-tokens 10`

- **`validate_env.py`**: Checks your environment for required dependencies and configuration.

  ```bash
  python -m tools.validate_env
  ```

- **`run_quality_checks.sh`**: Unified quality gate used by local dev and CI.

  ```bash
  # Run from repo root (recommended)
  make quality

  # Or run the script directly
  bash src/tools/run_quality_checks.sh
  ```

- **`run_ty_check.sh`**: Dedicated Ty entrypoint with explicit interpreter
  resolution for this repo.

  ```bash
  # Run from repo root (recommended)
  make ty

  # Or run the script directly
  bash src/tools/run_ty_check.sh
  ```

  This wrapper is the supported way to run Ty locally. It resolves the
  expected `mlx-vlm` conda interpreter and prints the target env, active env,
  resolved Python path, and resolved Ty binary before checking. Avoid relying
  on raw `ty check ...` environment auto-detection for this repo.

## Appendix: Dependencies

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
| ------- | ------- | ------- |
| Core tensor/runtime | `mlx` | `>=0.31.1` |
| Vision‑language utilities | `mlx-vlm` | `>=0.4.4` |
| Transformer compatibility surface | `transformers` | `>=5.5.3` |
| Image processing & loading | `Pillow[xmp]` | `>=10.3.0` |
| Safe XMP/XML parsing | `defusedxml` | `>=0.7.1` |
| Model cache / discovery | `huggingface-hub` | `>=1.10.1` |
| PEP 440 version parsing | `packaging` | `>=26.0` |
| Reporting / tables | `tabulate` | `>=0.9.0` |
| Configuration loading | `PyYAML` | `>=6.0` |
| Language model utilities | `mlx-lm` | `>=0.31.3` |

Optional (enable additional features):

| Feature | Package | Source | Install Command |
| ------- | ------- | ------ | --------------- |
| Extended system metrics (RAM/CPU) | `psutil` | `extras` | `pip install -e "src/[extras]"` |
| Fast tokenizer backends | `tokenizers` | `extras` | `pip install -e "src/[extras]"` |
| Tensor operations (for some models) | `einops` | `extras` | `pip install -e "src/[extras]"` |
| Number-to-words conversion (for some models) | `num2words` | `extras` | `pip install -e "src/[extras]"` |
| Transformer model support | `transformers` | core runtime | Installed by `make install` / `pip install -e src/` |
| PyTorch stack (needed for some models) | `torch`, `torchvision`, `torchaudio` | `torch` | `pip install -e "src/[torch]"` or `make -C src install-torch` |
| Vision backbones for FastVLM-style models | `timm` | `torch` | `pip install -e "src/[torch]"` or `make -C src install-torch` |

**Note**: Some models (e.g., Phi-3-vision, certain Florence2 variants) require PyTorch. If you encounter import errors for `torch`, `torchvision`, or `torchaudio`, install with:

```bash
# From root directory:
pip install -e "src/[torch]"
make -C src install-torch

# Or from src/ directory:
pip install -e ".[torch]"

# Install everything (extras + torch + dev):
make dev
make -C src install-all
pip install -e ".[extras,torch,dev]"  # from src/
```

Development / QA:

| Purpose | Package |
| ------- | ------- |
| Linting & formatting checks | `ruff` |
| Static type checking | `mypy` |
| Testing | `pytest`, `pytest-cov` |

### Minimal Install (runtime only)

<!-- MINIMAL_INSTALL_START -->
```bash
pip install "defusedxml>=0.7.1" "huggingface-hub[torch,typing]>=1.10.1" "mlx>=0.31.2" "mlx-lm>=0.31.3" "mlx-vlm>=0.4.4" "packaging>=26.0" "Pillow[xmp]>=10.3.0" "PyYAML>=6.0" "tabulate>=0.9.0" "transformers>=5.5.3" "wcwidth>=0.2.13"
```
<!-- MINIMAL_INSTALL_END -->

### With Optional Extras

The `extras` group in `pyproject.toml` pulls in `psutil`, `tokenizers`, `einops`, `num2words`, and `sentencepiece`:

```bash
pip install -e ".[extras,torch]"  # recommended for the widest optional feature/model coverage
```

If you only need the lighter optional extras without PyTorch, you can still install them separately:

```bash
pip install -e ".[extras]"
```

To include only the optional PyTorch stack when needed (macOS wheels include MPS acceleration):

```bash
pip install -e ".[torch]"
```

That `torch` extra also installs `timm`, which is required by FastVLM remote-code model loaders.

### Full Development Environment

```bash
pip install -e ".[dev,extras,torch]"  # dev tools + optional model/runtime deps
```


<!-- markdownlint-disable MD028 -->

> [!NOTE]
> `psutil` is optional (installed with `extras`); if absent the extended Apple Silicon hardware section omits RAM/cores.

> [!NOTE]
> `mlx-lm` is part of the core runtime dependency set. The `extras` group adds psutil, tokenizers, einops, num2words, and sentencepiece. For the widest model coverage, pair extras with `.[torch]` or install `.[extras,torch]` directly.

> [!NOTE]
> Project policy requires `transformers>=5.5.3` and validates the live
> `mlx_vlm` runtime contract during preflight so upstream API drift is surfaced
> before generation starts.

> [!NOTE]
> `system_profiler` is a macOS built-in (no install needed) used for GPU name / core info.

> [!NOTE]
> Torch is supported and can be installed when you need it for specific models; the script does not block Torch.

> [!NOTE]
> The `tools/update.sh` helper supports environment flags: `SKIP_TORCH=1` to skip PyTorch installation (torch is included by default), `MLX_METAL_JIT=ON` for smaller binaries with runtime compilation (default: `OFF` for pre-built kernels), `CLEAN_BUILD=1` to clean build artifacts first.

> [!NOTE]
> Installing `sentence-transformers` isn't necessary for this tool and may pull heavy backends into import paths; `check_models` ignores it in the normal execution path.

> [!NOTE]
> Long embedded CSS / HTML lines are intentional (readability > artificial wrapping).

> [!NOTE]
> To update dependency versions in this README:
>
> 1. Edit versions only in `pyproject.toml` (authoritative source).
> 2. Run the sync helper: `python -m tools.update_readme_deps` to regenerate the blocks between:
>    - `<!-- MANUAL_INSTALL_START -->` / `<!-- MANUAL_INSTALL_END -->`
>    - `<!-- MINIMAL_INSTALL_START -->` / `<!-- MINIMAL_INSTALL_END -->`
> 3. Commit both changed files together.

> [!NOTE]
> To clean build artifacts: `make clean` (project), `make clean-mlx` (local MLX repos), or `bash tools/clean_builds.sh`.

<!-- markdownlint-enable MD028 -->

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

# Check for harness/integration issues (mlx-vlm bugs vs model quality)
if analysis.has_harness_issue:
    print(f"⚠️ HARNESS ISSUE: {analysis.harness_issue_type}")
    print(f"   Details: {analysis.harness_issue_details}")
    # These indicate mlx-vlm integration bugs, not model quality problems
```

#### Harness Issue Detection

The quality analysis distinguishes between **model quality issues** (repetition, hallucinations, verbosity) and **harness/integration issues** (bugs in how mlx-vlm loads or runs the model). Harness issues are prefixed with `⚠️HARNESS:` in reports.

**Detected harness issues include:**

| Issue Type | Symptom | Likely Cause |
| ---------- | ------- | ------------ |
| `token_encoding` | BPE artifacts like `Ġ` or `Ċ` in output | Tokenizer decode bug |
| `special_token_leak` | `<\|end\|>`, `<\|endoftext\|>` visible | Stop token handling |
| `minimal_output` | Zero tokens or filler-only response | Model loading issue |
| `training_data_leak` | `# INSTRUCTION`, `### Response:` mid-output | Prompt template mismatch |

**Configuration**: Harness detection thresholds are configurable in the bundled `quality_config.yaml` (`src/check_models_data/quality_config.yaml` in this repo):

```yaml
# Harness/integration issue detection thresholds
min_bpe_artifact_count: 5       # Min BPE artifacts to flag encoding issue
min_tokens_for_substantial: 10  # Tokens below this are suspicious
min_words_for_filler_response: 15  # Words below this in filler response
long_prompt_tokens_threshold: 3000   # Prompt size where context-related failures become likely
severe_prompt_tokens_threshold: 12000  # Extreme prompt size risk threshold

# Default prompt compaction thresholds
prompt_title_max_chars: 120
prompt_description_max_chars: 420
prompt_keyword_max_items: 20
prompt_keyword_item_max_chars: 36
```

**When you see harness issues**: These typically indicate upstream bugs in mlx-vlm or model-specific integration problems. Consider:

1. Reporting the issue to [mlx-vlm](https://github.com/Blaizzy/mlx-vlm/issues)
2. Checking if a newer model version exists
3. Trying different prompt templates

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


## Command Line Reference

| Flag | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `-f`, `--folder` | Path | omitted | Folder to scan (non-recursive); the most recently modified image in that folder is used. If both `--folder` and `--image` are omitted, the most recently modified image in `~/Pictures/Processed` is used. |
| `-i`, `--image` | Path | omitted | Path to a specific image file to process directly. Requires a value when provided. |
| `--output-html` | Path | `output/reports/results.html` | HTML report output filename. |
| `--output-markdown` | Path | `output/reports/results.md` | Markdown report output filename. |
| `--output-gallery-markdown` | Path | `output/reports/model_gallery.md` | Standalone Markdown gallery artifact for qualitative output review. |
| `--output-review` | Path | `output/reports/review.md` | Markdown review digest grouped by owner and user bucket. |
| `--output-tsv` | Path | `output/reports/results.tsv` | TSV (tab-separated values) report output filename. |
| `--output-jsonl` | Path | `output/results.jsonl` | JSONL report output filename. |
| `--output-log` | Path | `output/check_models.log` | Command line output log filename. |
| `--output-env` | Path | `output/environment.log` | Environment log filename (pip freeze, conda list). |
| `--output-diagnostics` | Path | `output/reports/diagnostics.md` | Diagnostics report filename (generated on failures/harness issues). |
| `-m`, `--models` | list[str] | (none) | Explicit model IDs/paths; disables cache discovery. May be repeated; model lists accumulate across occurrences. |
| `-e`, `--exclude` | list[str] | (none) | Models to exclude (applies to cache scan or explicit list). May be repeated; exclusions accumulate across occurrences. |
| `--trust-remote-code` / `--no-trust-remote-code` | flag | `True` | Allow/disallow custom code from Hub models. Use `--no-trust-remote-code` for security. |
| `--revision` | str | (none) | Model revision (branch, tag, or commit) for reproducible runs. |
| `--adapter-path` | str | (none) | Path to LoRA adapter weights to apply on top of the base model. |
| `-p`, `--prompt` | str | omitted | Custom prompt text. Requires a value when provided; if omitted, a non-speculative metadata-verification prompt is generated automatically. |
| `--resize-shape` | int(s) | (none) | Resize image input before processor handling. Provide 1 integer for square resize or 2 for height width after one flag occurrence. |
| `--eos-tokens` | list[str] | (none) | Additional EOS tokens to stop on. Supports escaped values like `\n`. May be repeated; token lists accumulate across occurrences. |
| `--skip-special-tokens` | flag | `False` | Skip tokenizer special tokens in the detokenized output. |
| `--processor-kwargs` | JSON | (none) | Extra processor kwargs as a JSON object. Example: `'{"cropping": false, "max_patches": 3}'`. |
| `--enable-thinking` | flag | `False` | Enable thinking mode in the upstream chat template and generation flow. |
| `--thinking-budget` | int | (none) | Maximum number of thinking tokens before forcing the end token. |
| `--thinking-start-token` | str | (none) | Token marking the start of a thinking block, such as `<think>`. |
| `--thinking-end-token` | str | `</think>` | Token marking the end of a thinking block when thinking mode is enabled. |
| `-d`, `--detailed-metrics` | flag | `False` | Show expanded multi-line metrics block, including phase timings and stop reason when available; ignored unless `--verbose` is also set. |
| `--eval-mode` | str | `stress` | Evaluation lane: `stress` (default) = full cataloguing prompt, 500 token cap; `triage` = short prompt, 200 token cap, pass/fail only; `quality` = cataloguing prompt with generous 1000 token cap. |
| `-x`, `--max-tokens` | int | 500 | Max new tokens to generate. |
| `-t`, `--temperature` | float | 0.0 | Sampling temperature. |
| `--top-p` | float | 1.0 | Nucleus sampling parameter (0.0-1.0); lower = more focused. |
| `--min-p` | float | 0.0 | Minimum-probability sampling floor (0.0-1.0). 0.0 disables min-p filtering. |
| `--top-k` | int | 0 | Top-k sampling limit. 0 disables top-k filtering. |
| `-r`, `--repetition-penalty` | float | (none) | Penalize repeated tokens (>1.0 discourages repetition). |
| `--repetition-context-size` | int | 20 | Context window size for repetition penalty. |
| `-L`, `--lazy-load` | flag | `False` | Use lazy loading (loads weights on-demand, reduces memory). |
| `--max-kv-size` | int | (none) | Maximum KV cache size (limits memory for long sequences). |
| `-b`, `--kv-bits` | number | (none) | Quantize KV cache to N bits; uniform supports `2`, `3`, `4`, `5`, `6`, or `8`, and fractional values use TurboQuant. |
| `--kv-quant-scheme` | str | `uniform` | KV cache quantization backend: `uniform` or `turboquant`. |
| `-g`, `--kv-group-size` | int | 64 | Quantization group size for KV cache. |
| `--quantized-kv-start` | int | 0 | Start position for KV cache quantization. |
| `--prefill-step-size` | int | (none) | Step size for prompt prefill (`None` uses model default). |
| `-T`, `--timeout` | float | 300 | Operation timeout (seconds) for model execution. |
| `-v`, `--verbose` | flag | `False` | Enable verbose + debug logging. |
| `--no-color` | flag | `False` | Disable ANSI colors in the CLI output. |
| `--force-color` | flag | `False` | Force-enable ANSI colors even if stderr is not a TTY. |
| `--width` | int | (auto) | Force a fixed output width (columns) for separators and wrapping. |
| `-c`, `--quality-config` | Path | (none) | Path to custom quality configuration YAML file. |
| `--context-marker` | str | `Context:` | Marker used to identify context section in prompt. |
| `--prune-repro-days` | int | 90 | Delete repro bundles older than N days. Set 0 to disable pruning. |
| `--rerun-triage` | flag | `False` | Rerun triage-worthy models with a simple prompt for secondary evidence. First-pass results are never overwritten. |
| `-n`, `--dry-run` | flag | `False` | Validate arguments and show what would run without invoking models. |

### Selection Logic

Image selection logic:

1. If neither `--folder` nor `--image` is specified, the script uses the most recently modified image in the default folder (`~/Pictures/Processed`) and logs a diagnostic message.
2. If `--image` is provided, that image is processed directly.
3. If `--folder` is provided, the most recently modified image in the folder is used.

Model selection logic:

1. No model selection flags: run all cached VLMs.
2. `--models` only: run exactly that list.
3. `--exclude` only: run cached minus excluded.
4. `--models` + `--exclude`: intersect explicit list then subtract exclusions.

The script warns about exclusions that don't match any candidate model.

List-valued CLI flag semantics:

1. `--models`, `--exclude`, and `--eos-tokens` may be repeated; repeated occurrences accumulate values in order.
2. `--resize-shape` remains a single structured value; provide it after one flag occurrence.








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

Report featuring:

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

The main report also points to the standalone `model_gallery.md` artifact when it is
generated, so reviewers can switch to the dedicated model-by-model output view.

### Gallery Markdown Report

GitHub-compatible qualitative review artifact with:

- Populated image metadata fields when present (title, description, keywords, date, time, GPS)
- The full prompt in a fenced `text` block
- One easy-to-scan section per model with full generated output
- Existing success/failure gallery formatting reused from the main report path

### TSV Report

Tab-separated values for programmatic analysis (spreadsheets, `awk`, pandas, etc.):

- **Metadata comment**: The first line is a `# generated_at: <ISO timestamp>` comment
  indicating when the report was produced. Parsers should skip lines starting with `#`.
- **Header row**: Column names matching the CLI summary table.
- **Error diagnostics**: Two additional columns, `error_type` and `error_package`,
  are populated for failed models to support automated triage (e.g. filtering by
  `TimeoutError` or `mlx` package failures). These columns are empty for successful runs.

### JSONL Report

Line-delimited JSON for streaming ingestion:

- **Metadata header**: The first record (line 1) contains shared metadata
  (prompt, system info, timestamp) — JSONL v1.1 format.
- **Per-model records**: One JSON object per model with all metrics and error details.


### Diagnostics Report

A comprehensive Markdown report focused on upstream debugging and issue reporting:

- **Generation**: Created automatically when failures, harness issues (e.g., garbled output), or preflight compatibility warnings are detected.
- **Failures Clustered**: Groups similar errors together to identify systemic issues.
- **Reproducibility**: Includes explicit commands to reproduce specific failures.
- **Environment**: Captures full package versions and system specs.
- **Ready-to-File**: Formatted to be copy-pasted directly into GitHub issues.

### Preflight Compatibility Warnings

Some runs emit preflight compatibility warnings before inference starts. These warnings are informational by default.

- **What they mean**: `check_models` detected an upstream package or API-compatibility pattern that may matter for this environment or version combination.
- **What you should do**: keep running if outputs look healthy; investigate when the same run also shows API mismatches, startup hangs, or backend/runtime crashes.
- **What you should not do**: do not treat the warning alone as a failed benchmark.
- **When filing issues**: include the warning text and reported library versions so upstream maintainers can match it to the correct compatibility window.




## Metrics Tracked

The script tracks and reports:

- **Token Metrics**: Prompt tokens, generation tokens, total processing
- **Speed Metrics**: Tokens per second for prompt processing and generation
- **Memory Usage**: Peak memory consumption during processing
- **Timing**: Total processing time per model
- **Success Rate**: Model success/failure statistics
- **Error Analysis**: Detailed error reporting and diagnostics

## Troubleshooting

### Diagnostics

If neither `--folder` nor `--image` is specified, the script logs a diagnostic message indicating that it is using the most recently modified image in the default folder. This makes the omission behavior explicit without implying that `--folder` or `--image` can be passed without values.

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
python -m check_models --timeout 600  # 10 minutes
```

**Memory errors**: Test models individually or exclude large models

```bash
python -m check_models --exclude "meta-llama/Llama-3.2-90B-Vision-Instruct"
```

**Script crashes with mutex error**: If you see `libc++abi: terminating due to uncaught exception of type std::__1::system_error: mutex lock failed: Invalid argument`, TensorFlow is installed and conflicting with MLX.

Uninstall TensorFlow completely in MLX-only environments:

```bash
pip uninstall -y tensorflow tensorboard keras absl-py astunparse flatbuffers gast google_pasta grpcio h5py libclang ml_dtypes opt_einsum termcolor wrapt tensorboard-data-server
```

### Debug Mode

Use `--verbose` for detailed diagnostics:

```bash
python -m check_models --verbose
```

This provides:

- Detailed model loading information
- EXIF metadata extraction details
- Performance metric breakdowns
- Error stack traces
- Library version information

### Framework Detection Notes

- PyTorch is allowed by default; some models require it.
- `sentence-transformers` is not part of the normal `check_models` path.
- TensorFlow is not managed by `check_models`; if it is installed and you hit
  Apple Silicon mutex crashes, remove TensorFlow from the environment.

**Why this matters**: TensorFlow's Abseil mutex implementation can conflict with
MLX on macOS/ARM, causing crashes. Most MLX VLM workflows do not need
TensorFlow.

## Notes

- **Platform**: Requires macOS with Apple Silicon for MLX support
- **Colors**: Uses ANSI color codes for CLI output (may not display correctly in all terminals)
- **Timeout**: Unix-only functionality (not available on Windows)
- **Security**: The `--trust-remote-code` flag allows arbitrary code execution from models
- **Performance**: First run may be slower due to model compilation and caching

## Project Structure

```text
check_models/
├── src/
│   ├── check_models.py      # Main script
│   ├── Makefile             # Package-local automation targets
│   ├── pyproject.toml       # Project configuration and dependencies
│   ├── tools/               # Helper scripts
│   ├── tests/               # PyTest test suite
│   └── output/              # Generated outputs (git-ignored)
│       ├── reports/
│       │   ├── results.html
│       │   ├── results.md
│       │   ├── model_gallery.md
│       │   ├── review.md
│       │   ├── results.tsv
│       │   └── diagnostics.md
│       ├── issues/           # Generated GitHub issue templates
│       ├── repro_bundles/    # JSON reproduction bundles
│       ├── results.jsonl
│       ├── results.history.jsonl
│       ├── check_models.log
│       └── environment.log
├── docs/                    # Documentation
├── typings/                 # Generated type stubs (git-ignored)
└── Makefile                 # Root orchestration
```

**Output behaviour**: By default, outputs are written to `src/output/` (git-ignored).
Override with `--output-html`, `--output-markdown`, `--output-gallery-markdown`,
`--output-review`, `--output-tsv`, `--output-jsonl`, `--output-log`, `--output-env`,
and `--output-diagnostics`. Human-readable reports (HTML, Markdown, TSV, diagnostics)
default to `output/reports/`; machine-readable files (JSONL, history) and logs remain
in `output/`.

## Contributing

**For detailed contribution guidelines, coding standards, and project conventions, see:**

- [docs/CONTRIBUTING.md](../docs/CONTRIBUTING.md) - Setup, workflow, and PR process
- [docs/IMPLEMENTATION_GUIDE.md](../docs/IMPLEMENTATION_GUIDE.md) - Coding standards and architecture

### Developer Workflow (Makefile)

Use the root `Makefile` from the repository root and activate the conda env first:

```bash
conda activate mlx-vlm
```

Key commands:

- `make install` — install runtime package (`pip install -e src/`)
- `make dev` — install dev setup (`pip install -e "src/[dev,extras,torch]"`)
- `make test` — run pytest (`pytest src/tests/ -v`)
- `make vulture` — run the configured dead-code scan for `src/check_models.py` and `src/tools/`
- `make quality` — full gate (ruff format+lint, mypy, ty, pyrefly, vulture, pytest, shellcheck, markdownlint)
- `make ci` — strict CI-style pipeline
- `make deps-sync` — sync dependency blocks in docs from `pyproject.toml`
- `python -m tools.update_readme_deps --check` — verify dependency blocks are already synced (no writes)
- `make stubs` — regenerate local stubs in `typings/` (`mlx_lm`, `mlx_vlm`,
  `transformers`, `tokenizers`)

In VS Code, run the checked-in `Make: vulture` task to map Vulture findings
into `warning` entries in the Problems panel. The repo does not attach the
matcher to the broader `Make: quality` task, because mixed tool output can
create noisy or stale Problems entries. The repo does not auto-run Vulture on
save; rerun it after larger refactors or deletions.

For package-local targets (for example `install-dev`, `bootstrap-dev`, `lint-fix`), run:

```bash
make -C src help
```

### Git Hooks and Pre-Commit

Recommended workflow:

- Custom git hooks shipped with this repo:

  ```bash
  cd src
  python -m tools.install_precommit_hook
  ```

Alternative workflow:

- `pre-commit` framework:

  ```bash
  pre-commit install
  pre-commit run --hook-stage pre-push --all-files
  ```

Both workflows call the same shared scripts:

- commit stage: staged-file hygiene only
- push stage: fast static checks, Vulture, plus `pytest -m "not slow and not e2e"`

The push-stage gate also validates the checked-in GitHub workflow YAML and
keeps the CI/static tooling path aligned with the checked-in scripts.

### Markdown Linting (Optional)

`make quality` runs markdownlint via local install, global install, or `npx` fallback.
If you want a local install:

```bash
cd src
npm install
```

### Contribution Guidelines

- Keep patches focused; separate mechanical formatting changes from functional changes.
- Run `make quality` before opening a PR (or at minimum `make test` and `make typecheck`).
- Add or update tests when changing output formatting or public CLI flags.
- Prefer small helper functions over adding more branching to large blocks in `check_models.py`.
- Document new flags or output changes in this README (search for an existing section to extend rather than creating duplicates).
- For full conventions (naming, imports, dependency policy, quality gates), see `IMPLEMENTATION_GUIDE.md` in `docs/`.

## Important Notes

- Timeout functionality requires UNIX (not available on Windows).
- For best results, ensure all dependencies are installed and models are downloaded/cached.

## License

MIT License: See the [LICENSE](../LICENSE) file for details.
