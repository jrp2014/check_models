# MLX VLM Check — Quick Overview

Lightweight CLI to run and benchmark MLX-compatible Vision-Language Models (VLMs) on Apple Silicon. Produces human- and machine-readable reports (HTML, Markdown, TSV, JSONL) and captures performance metrics (tokens/sec, memory, timings).

Quick start (recommended):

Note: This tool runs MLX-format Vision-Language Models hosted on the [Hugging Face Hub](https://huggingface.co). By default it will run all the models found in your local Hugging Face Hub model cache (use `--models` to specify explicit model IDs).

```bash
# Install runtime dependencies
make install

# Run all models against a folder (auto-selects most recent image) using the default built in prompt
python -m check_models --folder ~/Pictures/Processed

# Run them on a single image
python -m check_models --image /path/to/photo.jpg
```

Where to find detailed docs

- Canonical CLI reference and developer guide: [src/README.md](src/README.md) (full parameter reference, examples, troubleshooting).
- Developer and contribution guidelines: [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md).

Common make commands:

```bash
make install   # install runtime dependencies
make dev       # install dev deps and set up hooks
make test      # run test suite
make quality   # run formatting, linting, and type checks
```

Notes

- Default Python: 3.13+ (project is tested on 3.13).
- Platform: macOS with Apple Silicon (recommended).
- Output files are written to `output/` by default and are git-ignored.

For complete CLI usage, examples, and advanced configuration, see [src/README.md](src/README.md).

License: See the [LICENSE](LICENSE) file.
