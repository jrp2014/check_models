
# MLX VLM Check Scripts

This repository hosts a small, focused project centered on `vlm/check_models.py`, a script for running MLX Vision-Language Models (VLMs) on images, collecting performance metrics, and generating HTML/Markdown reports.

If you just want to run the script on your images, see the “TL;DR for Users” section in `vlm/README.md`. If you want to contribute to the script or tooling, see the Contributors section in the same file.

## Layout

- `vlm/` — Python package with the main script, tests, and tools
  - `check_models.py` — the primary CLI and implementation
  - `tools/` — helper scripts (quality, stub generation, dependency sync)
  - `tests/` — unit tests for formatting and reporting
  - `pyproject.toml` — project metadata and tooling config for this package
- `typings/` — generated `.pyi` stubs for third‑party packages (not committed)
- `Makefile` — root shim forwarding to `vlm/Makefile`

## Quickstart

- Install with dev tools and extras:

```bash
make -C vlm bootstrap-dev
```

- Run quality checks (ruff + mypy):

```bash
make quality
```

- Run tests:

```bash
make test
```

- Run the VLM checker (pass flags via ARGS):

```bash
make check_models ARGS="--model mlx-community/Florence-2-large --image /path/to/image.jpg --max-tokens 256"
```

Notes:

- If you pass a directory path to `--image`, the script automatically selects the most recent image file in that folder (by modification time).

- Generate local type stubs (to improve mypy signal):

```bash
make -C vlm stubs
```

## Typings

Local type stubs live under `typings/` and are generated via:

```bash
python -m vlm.tools.generate_stubs mlx_vlm tokenizers
```

The directory is ignored by git. If stubs get stale or noisy:

```bash
make -C vlm stubs-clear && make -C vlm stubs
```

## Conventions

See `STYLE_GUIDE.md` for coding conventions, including dependency sync, pyproject shape, and quality gates. The root `pyproject.toml` is the source of truth for dependency lists; the README blocks are kept in sync by the `vlm/tools/update_readme_deps.py` script.

## Contributing

Please read `STYLE_GUIDE.md` before making changes. It covers formatting, typing, dependency synchronization, and the automated quality checks. Use `make quality` locally to verify your changes.
