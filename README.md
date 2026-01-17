# MLX VLM Check

Lightweight CLI to run and benchmark MLX-compatible Vision-Language Models (VLMs) on Apple Silicon. Produces HTML/Markdown/TSV/JSONL reports and captures performance metrics (tokens/sec, memory, timings).

> [!NOTE]
> This tool runs MLX-format Vision-Language Models hosted on the [Hugging Face Hub](https://huggingface.co). By default it runs all models found in your local HF cache (use `--models` to specify explicit model IDs).

## Quick Start (fast path)

```bash
# Install runtime dependencies
make install

# Run all models against a folder (auto-selects most recent image) using the default built in prompt
python -m check_models --folder ~/Pictures/Processed

# Run them on a single image
python -m check_models --image /path/to/photo.jpg
```

## Why use it (short)

- Batch run multiple models against an image.
- Standardized metrics + rich reports for easy comparison.
- Robust error handling and metadata-aware prompts.

## Documentation (full details)

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

## Ecosystem (quick links)

- **[MLX](https://github.com/ml-explore/mlx)**: Array framework for Apple Silicon.
- **[MLX VLM](https://github.com/ml-explore/mlx-vlm)**: Underlying VLM runtime.
- **[Hugging Face Hub](https://huggingface.co)**: Model source (look for `mlx-community` or `mlx` tags).

License: See the [LICENSE](LICENSE) file.
