# MLX VLM Check

Run and benchmark Vision-Language Models on Apple Silicon with MLX. Get performance metrics and detailed reports in seconds.

## Quick Start

**Requirements**: macOS with Apple Silicon (M1/M2/M3/M4), Python 3.13+

```bash
# 1. Install
make install

# 2. Run on an image
python -m check_models --model mlx-community/Florence-2-large --image /path/to/image.jpg

# 3. Get HTML report
python -m check_models --model mlx-community/Florence-2-large --image /path/to/image.jpg --html
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

# Generate HTML report
python -m check_models --image photo.jpg --html

# Use a custom prompt
python -m check_models --image photo.jpg --prompt "Describe this in detail"
```

**💡 Tip**: Pass a folder path to automatically select the most recently modified image (hidden files are ignored).

### Using Make Commands

For convenience, many operations have `make` targets:

```bash
make install      # Install the package
make test         # Run tests
make quality      # Run code quality checks
make clean        # Remove generated files
```

See `make help` for all available commands.

## For Developers

### Quick Setup

```bash
# Clone and setup
git clone https://github.com/jrp2014/scripts.git
cd scripts
make dev          # Install dev dependencies + setup hooks

# Run quality checks
make quality

# Run tests
make test
```

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

Local type stubs live under `typings/` and are generated via:

```bash
python -m tools.generate_stubs mlx_vlm tokenizers
```

The directory is ignored by git. If stubs get stale:

```bash
make clean && python -m tools.generate_stubs mlx_vlm tokenizers
```

### Dependencies

The `src/pyproject.toml` is the source of truth for dependency lists. To check for outdated packages:

```bash
python -m tools.check_outdated
```

## Advanced Topics

### All Available Make Commands

Run `make help` to see all targets. Key commands:

| Command | Description |
|:--------|:------------|
| `make install` | Install the package |
| `make dev` | Setup development environment |
| `make update` | Update conda environment and dependencies |
| `make upgrade-deps` | Upgrade all dependencies to latest versions |
| `make test` | Run tests |
| `make quality` | Run linting and type checks |
| `make check` | Run full quality pipeline (format, lint, typecheck, test) |
| `make clean` | Remove generated files |

### Dependency Management

```bash
make upgrade-deps # Upgrade all dependencies to latest compatible versions
make lock-deps    # Generate requirements.txt from requirements.in
make sync-deps    # Sync installed packages with requirements.txt
```

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
