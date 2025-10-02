
# MLX VLM Check

A benchmarking and testing tool for MLX Vision-Language Models (VLMs). Run VLMs on images, collect performance metrics, and generate detailed HTML/Markdown reports.

**Target audience**: Users who want to test and benchmark VLM models on macOS/Apple Silicon.

**Related documents**:

- [vlm/README.md](vlm/README.md) - Detailed usage, CLI options, and examples
- [CONTRIBUTING.md](CONTRIBUTING.md) - How to contribute to this project
- [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) - Technical standards for developers

## What This Does

`check_models.py` runs MLX-based Vision-Language Models on images and generates detailed performance reports including:

- **Model outputs**: Generated text from VLM inference
- **Performance metrics**: Generation time, tokens per second, memory usage
- **Image metadata**: EXIF data, GPS coordinates (when available)
- **System information**: Device specs, library versions
- **Multiple output formats**: HTML (styled, viewable in browser) and Markdown (for GitHub)

**Note on folder input**: When you pass a folder path, the tool automatically selects the most recently modified, non-hidden image file in that folder. Override this by passing a direct file path (see `vlm/README.md` for CLI options).LM Check Scripts

This repository hosts a small, focused project centered on `vlm/check_models.py`, a script for running MLX Vision-Language Models (VLMs) on images, collecting performance metrics, and generating HTML/Markdown reports.

If you just want to run the script on your images, see the “TL;DR for Users” section in `vlm/README.md`. If you want to contribute to the script or tooling, see the Contributors section in the same file.

Note on folder input: when you pass a folder path, the tool will automatically select the most recently modified, non-hidden image file in that folder. Override this by passing a direct file path (CLI in `vlm/README.md`).

## Repository Structure

```
.
├── vlm/                        # Main Python package
│   ├── check_models.py        # Primary CLI and implementation
│   ├── tools/                 # Helper scripts (quality checks, stubs, etc.)
│   ├── tests/                 # Unit tests
│   ├── pyproject.toml         # Package metadata and tooling config
│   └── README.md              # Detailed usage and CLI documentation
├── typings/                   # Generated type stubs (git-ignored)
├── Makefile                   # Convenience shim (forwards to vlm/Makefile)
├── CONTRIBUTING.md            # How to contribute (setup, workflow, PRs)
├── IMPLEMENTATION_GUIDE.md    # Technical standards and coding conventions
└── README.md                  # This file (project overview)
```

## Quick Start

### For Users (Running the Tool)

See [vlm/README.md](vlm/README.md) for:

- Installation instructions
- CLI options and examples
- Model selection and usage
- Output format details

**TL;DR**:

```bash
# Install
pip install -e vlm/

# Run on an image
python -m vlm.check_models --model mlx-community/Florence-2-large --image /path/to/image.jpg

# Generate HTML report
python -m vlm.check_models --model mlx-community/Florence-2-large --image /path/to/image.jpg --html results.html
```

### For Contributors (Development)

See [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Development environment setup
- How to make changes
- Quality checks and testing
- Pull request process

**TL;DR**:

```bash
# Setup dev environment
make -C vlm bootstrap-dev

# Run quality checks
make quality

# Run tests
make test
```

## Available Commands

```bash
# Development
make -C vlm bootstrap-dev     # Setup dev environment with all dependencies
make -C vlm stubs             # Generate type stubs for better type checking

# Quality & Testing
make quality                  # Run ruff format + lint + mypy
make test                     # Run all tests
make -C vlm ci                # Full CI pipeline (quality + tests)

# Dependencies
make -C vlm check-outdated    # Check for outdated packages
make -C vlm upgrade-deps      # Upgrade all dependencies
make -C vlm audit             # Security vulnerability scan

# Running the Tool
make check_models ARGS="..."  # Run check_models.py with custom arguments
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

## Documentation

- **[README.md](README.md)** (this file) - Project overview and quick start
- **[vlm/README.md](vlm/README.md)** - Detailed usage, CLI options, and examples
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - How to contribute (setup, workflow, PRs)
- **[IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)** - Technical standards and coding conventions

The `vlm/pyproject.toml` is the source of truth for dependency lists; the README blocks are kept in sync by the `tools/update_readme_deps.py` script.
