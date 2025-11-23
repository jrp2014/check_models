# MLX VLM Check — Quick Overview

Lightweight CLI to run and benchmark MLX-compatible Vision-Language Models (VLMs) on Apple Silicon. Produces human- and machine-readable reports (HTML, Markdown, TSV, JSONL) and captures performance metrics (tokens/sec, memory, timings).

Quick start (recommended):


> [!NOTE]
> This tool runs MLX-format Vision-Language Models hosted on the [Hugging Face Hub](https://huggingface.co). By default it will run all the models found in your local Hugging Face Hub model cache (use `--models` to specify explicit model IDs).

## Why use this tool?

While `mlx-vlm` provides excellent raw generation capabilities, `check_models.py` adds a layer of **batch processing, standardized benchmarking, and reporting** essential for model evaluation:

- **Batch Processing**: Run multiple models against an image in one go.
- **Standardized Metrics**: Automatically capture tokens/sec, memory usage, and load times in a consistent format.
- **Rich Reporting**: Generate HTML, Markdown, and JSONL reports for easy sharing and analysis.
- **Robustness**: Handles model failures gracefully (timeouts, errors) without crashing the entire batch.
- **Metadata Awareness**: Uses EXIF and GPS data from images to generate context-aware prompts automatically.

## Ecosystem

This tool uses the broader MLX ecosystem on Apple Silicon:

- **[MLX](https://github.com/ml-explore/mlx)**: The array framework for machine learning on Apple Silicon.
- **[MLX VLM](https://github.com/ml-explore/mlx-vlm)**: The underlying library for running Vision-Language Models.
- **[Hugging Face Hub](https://huggingface.co)**: The source for models (look for `mlx-community` or models with `mlx` tags).  Image-Text-to-Text (I2T2) models are the category most likely to be supported.

## Quick Start

```bash
# Install runtime dependencies
make install

# Run all models against a folder (auto-selects most recent image) using the default built in prompt
python -m check_models --folder ~/Pictures/Processed

# Run them on a single image
python -m check_models --image /path/to/photo.jpg
```

## Documentation

- **[User Guide & CLI Reference](src/README.md)**: Full parameter reference, advanced usage, and troubleshooting.
- **[Contributor Guide](docs/CONTRIBUTING.md)**: Setup, workflow, and quality standards.

## Common Make Commands

```bash
make install   # install runtime dependencies
make dev       # install dev deps and set up hooks
make test      # run test suite
make quality   # run formatting, linting, and type checks
```

> [!TIP]
> **Platform**: macOS with Apple Silicon is required.
> **Python**: 3.13+ is recommended and tested.

License: See the [LICENSE](LICENSE) file.
