# MLX Vision Language Model Checker

A comprehensive tool for testing and benchmarking Vision Language Models (VLMs) using Apple's MLX framework. This script processes images with multiple VLMs, extracts metadata, generates captions, and provides detailed performance analysis with professional reports.

## What This Script Does

The MLX VLM Checker is designed to:

- **Test Vision Language Models**: Automatically discover and test all locally cached VLMs or run specific models
- **Performance Benchmarking**: Measure and report token generation speeds, memory usage, and processing times
- **Image Analysis**: Extract EXIF metadata, GPS information, and generate contextual prompts
- **Professional Reporting**: Create detailed HTML and Markdown reports with performance metrics and model outputs
- **Flexible Model Selection**: Include specific models, exclude problematic ones, or test all available models
- **Error Handling**: Gracefully handle model failures, timeouts, and provide diagnostic information

## Features

- **Comprehensive Model Testing**: Process images with one or more MLX Vision Language Models
- **Smart Model Discovery**: Automatically scans local Hugging Face cache for available VLMs
- **Flexible Model Selection**:
  - Run all cached models
  - Specify explicit model lists
  - Exclude specific models from testing
  - Filter explicit model lists with exclusions
- **Rich Metadata Extraction**: Extracts EXIF data, GPS coordinates, and image information
- **Intelligent Prompting**: Generates context-aware prompts based on image metadata
- **Performance Metrics**: Tracks tokens/second, memory usage, processing time, and error rates
- **Multiple Output Formats**:
  - Colorized CLI output with real-time progress
  - Professional HTML reports with interactive tables
  - GitHub-compatible Markdown reports
- **Robust Error Handling**: Timeout management, graceful failure handling, and diagnostic output
- **Verbose Debugging**: Detailed logging and performance analysis in verbose mode

## Installation and Environment Setup

### Using pyproject.toml (Recommended)

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

If you prefer to install dependencies manually:

```bash
pip install mlx>=0.10.0 mlx-vlm>=0.0.9 Pillow>=10.0.0 huggingface-hub>=0.20.0 tabulate>=0.9.0 tzlocal>=5.0
```

## Requirements

- **Python**: 3.12+ (3.9+ may work but 3.12+ recommended)
- **Operating System**: macOS with Apple Silicon (required for MLX)
- **Dependencies**: Automatically handled by pyproject.toml
- **Models**: MLX-compatible Vision Language Models (downloaded automatically from Hugging Face)

## Usage

### Basic Usage

```bash
Test all models with custom prompt:

```bash
python check_models.py -p "What is the main object in this image?"
```

Test specific models only:

```bash
python check_models.py -m mlx-community/nanoLLaVA mlx-community/llava-1.5-7b-hf
```

Exclude problematic models:

```bash
python check_models.py -e mlx-community/problematic-model
```

Test specific models but exclude some:

```bash
python check_models.py -m model1 model2 model3 -e model2
# Only tests model1 and model3
```

### Verbose Output

Enable detailed logging for debugging:

- Detailed model loading information
- Token-by-token generation details
- Memory usage tracking
- Detailed error stack traces

```bash
python check_models.py -v
```

### Advanced Model Selection

```bash
# Test explicit models but exclude one from the list
python check_models.py --models model1 model2 model3 --exclude model2

# Exclude multiple models from cache scan
python check_models.py --exclude model1 model2 --verbose

# Use custom output locations
python check_models.py --output-html ~/reports/results.html --output-markdown ~/reports/results.md
```

### Complete Example

```bash
python check_models.py \
  --folder ~/Pictures/TestImages \
  --exclude "microsoft/Phi-3-vision-128k-instruct" \
  --prompt "Provide a detailed caption for this image" \
  --max-tokens 200 \
  --temperature 0.1 \
  --timeout 600 \
  --output-html ~/reports/vlm_benchmark.html \
  --verbose
```

## Command Line Arguments

### Core Options

- `-f, --folder`: Image folder to scan (default: `~/Pictures/Processed`)
- `-m, --models`: Specify explicit models by ID/path (overrides cache scan)
- `-e, --exclude`: Exclude models from processing (works with both explicit models and cache scan)
- `-p, --prompt`: Custom prompt for the models (auto-generated from metadata if not provided)

### Model Selection Logic

The script supports four model selection scenarios:

1. **No options**: Tests all locally cached VLMs
2. **--models only**: Tests only the specified models
3. **--exclude only**: Tests all cached models except excluded ones
4. **--models + --exclude**: Tests specified models minus excluded ones

**Warning System**: The script warns when excluded models aren't in the available set, helping you understand which exclusions are effective.

### Performance Options

- `-x, --max-tokens`: Maximum tokens to generate (default: 500)
- `-t, --temperature`: Sampling temperature 0.0-1.0 (default: 0.1)
- `--timeout`: Timeout in seconds for model operations (default: 300)

### Output Options

- `--output-html`: HTML report filename (default: `results.html`)
- `--output-markdown`: Markdown report filename (default: `results.md`)
- `-v, --verbose`: Enable detailed logging and debug output

### Security Options

- `--trust-remote-code`: Allow custom code from Hub models ⚠️ **SECURITY RISK**

## Output Formats

### CLI Output

Real-time colorized output showing:

- Model processing progress with success/failure indicators
- Performance metrics (tokens/second, memory usage, timing)
- Generated text preview
- Error diagnostics for failed models
- Final performance summary table

### HTML Report (`results.html`)

Professional report featuring:

- Executive summary with test parameters
- Interactive performance table with sortable columns
- Model outputs and diagnostics
- System information and library versions
- Responsive design for mobile viewing

### Markdown Report (`results.md`)

GitHub-compatible format with:

- Performance metrics in table format
- Model outputs
- System and library version information
- Easy integration into documentation

## Model Selection Examples

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

## Performance Metrics

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

1. Install development dependencies: `pip install -e ".[dev]"`
2. Run linting: `ruff check .`
3. Run type checking: `mypy check_models.py`
4. Test your changes thoroughly

## Important Notes

- Timeout functionality requires UNIX (not available on Windows).
- For best results, ensure all dependencies are installed and models are downloaded/cached.

## License

MIT License - see LICENSE file for details.
