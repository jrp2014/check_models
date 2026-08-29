# MLX VLM Check

Lightweight CLI to run and benchmark MLX-compatible Vision-Language Models (VLMs) on Apple Silicon. Produces HTML, Markdown, and JSONL reports plus maintainer-ready diagnostics, and captures performance metrics (tokens/sec, memory, timings).

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

Expected outputs (default location: `src/output/`):

Tracked in git (browsable on GitHub):

- `index.md` — run dashboard (counts, usability, top observations) + links
- `reports/results.html` — self-contained interactive report
- `reports/model_gallery.md` — complete per-model evidence gallery, with an
  Output-at-a-Glance table of every model's actual output
- `reports/diagnostics.md` — maintainer triage tables and per-model evidence
- `issues/run_summary.md` — paste-ready whole-run issue summary (when findings exist)
- `issues/issue_*.md` — per-crash issue drafts (hard actionable crashes only)
- `results.jsonl` — the sole schema-3 machine contract: a run-level metadata
  header (provenance, counts, artifact manifest, baseline comparison) followed
  by per-model records including full rendered prompts and captured upstream
  console output
- `check_models.log` — full run log (timeline, tee'd model console output)
- `environment.log` — complete dependency and toolchain inventory

Local only (gitignored, regenerated each run):

- `results.history.jsonl` — append-only run history

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
make install       # install runtime dependencies
make dev           # install dev dependencies (dev + extras + torch)
make update        # full update: conda/brew, local MLX builds, stubs, smoke (tools/update.sh)
make update-quick  # quick in-env refresh (pip upgrade + editable reinstall)
make test          # run pytest only
make quality       # run full gate (ruff + typing + vulture + Skylos quality/audit/danger + pytest + shellcheck + markdownlint)
make skylos-danger      # advisory Skylos workflow/security scan (diff-aware on PRs)
make skylos-danger-llm  # same advisory scan with LLM-oriented output
make skylos-verify      # narrow Skylos file/range verifier (pass ARGS='--file ... --range ...')
```

The Skylos `--danger` scan is part of the blocking `make quality` gate (full
mode); `make skylos-danger` runs the same scan in advisory, diff-aware form
for triage.

> [!TIP]
> **Platform**: macOS with Apple Silicon is required.
> **Python**: 3.13+ is recommended and tested.

## Ecosystem (quick links)

- **[MLX](https://github.com/ml-explore/mlx)**: Array framework for Apple Silicon.
- **[MLX VLM](https://github.com/Blaizzy/mlx-vlm)**: Underlying VLM runtime.
- **[Hugging Face Hub](https://huggingface.co)**: Model source (look for `mlx-community` or `mlx` tags).

License: See the [LICENSE](LICENSE) file.
