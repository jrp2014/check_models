# MLX VLM Check

Lightweight CLI to run and benchmark MLX-compatible Vision-Language Models (VLMs) on Apple Silicon. Produces HTML/Markdown/gallery Markdown/TSV/JSONL reports and captures performance metrics (tokens/sec, memory, timings).

> [!NOTE]
> This tool runs MLX-format Vision-Language Models hosted on the [Hugging Face Hub](https://huggingface.co). By default it runs cached models that pass the `mlx-vlm` server-supported cache filter; use `--models` to specify explicit model IDs.

## Quick Start (fast path)

```bash
# Create the recommended conda environment and install runtime dependencies
bash src/tools/setup_conda_env.sh
conda activate mlx-vlm
make install

# Run all models against a folder (auto-selects most recent image) using the default built in prompt
python -m check_models --folder ~/Pictures/Processed

# Run them on a single image
python -m check_models --image /path/to/photo.jpg
```

## First successful run (example)

```bash
python -m check_models --image ~/Pictures/sample.jpg
```

Expected outputs (default location: src/output/):

- results.html
- results.md
- model_gallery.md
- results.tsv
- results.jsonl
- results.history.jsonl
- diagnostics.md (only when failures, harness issues, text-sanity issues, or
  preflight warnings are detected)
- check_models.log
- environment.log

## Why use it (short)

- Batch run multiple models against an image.
- Standardized metrics + rich reports for easy comparison and qualitative review.
- Robust error handling and metadata-aware prompts.
- Explicit `triage`, metadata-blind, and metadata-assisted evaluation lanes,
  with lane-isolated history and capability comparisons.

## Documentation (full details)

- **[User Guide & CLI Reference](src/README.md)**: Full parameter reference, advanced usage, and troubleshooting.
- **[Contributor Guide](docs/CONTRIBUTING.md)**: Setup, workflow, and quality standards.

## Common Make Commands

```bash
make install   # install runtime dependencies
make dev       # install dev dependencies (dev + extras + torch)
make test      # run pytest only
make quality   # run full gate (ruff + typing + vulture + Skylos quality/audit + pytest + shellcheck + markdownlint)
make skylos-danger      # advisory Skylos workflow/security scan
make skylos-danger-llm  # same advisory scan with LLM-oriented output
make skylos-verify      # narrow Skylos file/range verifier (pass ARGS='--file ... --range ...')
```

`make skylos-danger` remains advisory for now, but the repo-root `--danger`
scan is currently clean, so it is a credible candidate for promotion into the
blocking gate later.

> [!TIP]
> **Platform**: macOS with Apple Silicon is required.
> **Python**: 3.13+ is recommended and tested.

## Ecosystem (quick links)

- **[MLX](https://github.com/ml-explore/mlx)**: Array framework for Apple Silicon.
- **[MLX VLM](https://github.com/Blaizzy/mlx-vlm)**: Underlying VLM runtime.
- **[Hugging Face Hub](https://huggingface.co)**: Model source (look for `mlx-community` or `mlx` tags).

License: See the [LICENSE](LICENSE) file.
