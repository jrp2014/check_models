# MLX VLM Check

Run and benchmark Vision-Language Models on Apple Silicon with MLX. Get performance metrics and detailed reports in seconds.

## Quick Start

**Requirements**: macOS with Apple Silicon (M1/M2/M3/M4), Python 3.13+

```bash
# 1. Install
make install

# 2. Run on an image
python -m check_models --model mlx-community/Florence-2-large --image /path/to/image.jpg

# 3. Get HTML report (specify output path)
python -m check_models --model mlx-community/Florence-2-large --image /path/to/image.jpg --html output/results.html
```

That's it! See [detailed usage](#detailed-usage) below for more options.

## What You Get

- ✅ **Instant results**: Run any MLX-compatible VLM on your images
- 📊 **Performance metrics**: Tokens/sec, memory usage, generation time
- 📝 **Multiple formats**: Beautiful HTML reports + GitHub-compatible Markdown
- 🖼️ **Smart image handling**: Folders automatically use the latest image
- 📍 **Metadata extraction**: EXIF data, GPS coordinates when available

## Detailed Usage

### Common Commands

```bash
# Run on a specific image
python -m check_models --image /path/to/photo.jpg

# Run on a folder (uses most recent image)
python -m check_models --folder ~/Pictures

# Specify a model
python -m check_models --model mlx-community/Qwen2-VL-2B-Instruct-4bit --image photo.jpg

# Generate HTML report (specify output path)
python -m check_models --image photo.jpg --html output/results.html

# Use a custom prompt
python -m check_models --image photo.jpg --prompt "Describe this in detail"
```

**💡 Tip**: Pass a folder path with `--folder` to automatically select the **most recently modified image** in that directory (hidden files are ignored).

### Using Make Commands

For convenience, many operations have `make` targets:

```bash
make install      # Install the package
make test         # Run tests
make quality      # Run code quality checks
make clean        # Remove generated files
```

See `make help` for all available commands.

## Quality Configuration

You can customize the quality detection rules (thresholds and patterns) by providing a YAML configuration file:

```bash
python -m check_models --quality-config my_quality_config.yaml --folder path/to/images --models mlx-community/Llama-3.2-11B-Vision-Instruct-4bit
```

A default configuration is provided in `src/quality_config.yaml`. You can copy this file and modify it to:

- Adjust thresholds for repetition, verbosity, etc.
- Add new hallucination patterns or refusal phrases.
- Change the list of "filler words" for generic output detection.

## For Contributors

Want to contribute? Great! See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for the complete setup and workflow guide, [IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md) for coding standards, and [STYLE_GUIDE.md](docs/STYLE_GUIDE.md) for conventions and best practices.

### Quick Setup

```bash
# Clone and setup
git clone https://github.com/jrp2014/check_models.git
cd check_models
make dev          # Install dev dependencies + setup hooks

# Run quality checks
make quality

# Run tests
make test
```

**Note**: Output files are written to `output/` by default (git-ignored). See [src/README.md](src/README.md) for detailed CLI options.

See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for the complete development guide.

## Development

### For Contributors

See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for:

- Development environment setup
- How to make changes
- Quality checks and testing
- Pull request process

**Quick development setup**:

```bash
# Setup dev environment with all dependencies
make dev

# Run quality checks (ruff + mypy)
make quality

# Run tests
make test

# Format code
make format
```

### Type Stubs

Local type stubs live under `typings/` and are generated automatically by running:

```bash
make typings
```

The directory is ignored by git. If stubs get stale, just rerun `make typings`.

### Dependencies

The `src/pyproject.toml` is the source of truth for dependency lists. To check for outdated packages:

```bash
make check-outdated
```

## Advanced Topics

### All Available Make Commands

Run `make help` to see all targets. Key commands:

- `make install` — Install the package
- `make dev` — Setup development environment
- `make update` — Update conda environment and dependencies
- `make upgrade-deps` — Upgrade all dependencies to latest versions
- `make test` — Run tests
- `make quality` — Run linting and type checks
- `make check` — Run full quality pipeline (format, lint, typecheck, test)
- `make clean` — Remove generated files

### Dependency Management

Dependencies are defined in `src/pyproject.toml` as the single source of truth.

```bash
make deps-sync    # Sync README dependency blocks with pyproject.toml
```

To add a new dependency, edit `src/pyproject.toml` in the appropriate section:

- Runtime: `[project.dependencies]`
- Development: `[project.optional-dependencies.dev]`
- Optional extras: `[project.optional-dependencies.extras]`

### Checking for Outdated Packages

```bash
python -m tools.check_outdated
```

## Repository Structure

```text
.
├── src/                       # Main Python package
│   ├── check_models.py        # Primary CLI and implementation
│   ├── tools/                 # Helper scripts (quality checks, stubs, etc.)
│   ├── tests/                 # Unit tests
│   └── pyproject.toml         # Package metadata and dependencies
├── docs/                      # Documentation
│   ├── CONTRIBUTING.md        # How to contribute
│   ├── IMPLEMENTATION_GUIDE.md    # Technical standards
│   └── notes/                 # Design notes and reviews
├── Makefile                   # Development commands
└── README.md                  # This file
```

## Common Issues

### TensorFlow Conflicts

If you encounter import errors related to TensorFlow (e.g., `"mlx_vlm: Failed to import"`), you can safely remove it:

```bash
pip uninstall -y tensorflow
```

**Why this works**: MLX-VLM doesn't require TensorFlow. It may be pulled in as an indirect dependency by some packages (like `transformers`), but it's not needed for MLX workflows. TensorFlow 2.20 has known issues on Python 3.13.

For more troubleshooting help, see the [detailed troubleshooting guide](src/README.md#troubleshooting) in `src/README.md`.

## More Documentation

- **[src/README.md](src/README.md)** - Complete CLI reference and usage examples
- **[docs/CONTRIBUTING.md](docs/CONTRIBUTING.md)** - Development workflow and guidelines
- **[docs/IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md)** - Technical standards and conventions
- **[docs/notes/](docs/notes/)** - Design decisions and technical notes

## Requirements

- **Python**: 3.13+
- **Platform**: macOS with Apple Silicon (M1/M2/M3/M4)
- **Framework**: MLX (Apple's machine learning framework)

## License

See [LICENSE](LICENSE) file for details.
