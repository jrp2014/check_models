# MLX VLM Check

A benchmarking and testing tool for MLX Vision-Language Models (VLMs) on Apple Silicon. Run VLMs on images, collect performance metrics, and generate detailed HTML/Markdown reports.

**🎯 Target Audience**: Users who want to test and benchmark Vision-Language Models on macOS/Apple Silicon using MLX.

## Quick Start

```bash
# 1. Install the package
make install

# 2. Run on an image
make check_models ARGS="--model mlx-community/Florence-2-large --image /path/to/image.jpg"

# Or use the CLI directly
python -m check_models --model mlx-community/Florence-2-large --image /path/to/image.jpg

# 3. Generate HTML report
python -m check_models --model mlx-community/Florence-2-large --image /path/to/image.jpg --html
```text
**📝 Note**: When you pass a folder path, the tool automatically selects the most recently modified, non-hidden image file in that folder. Pass a direct file path to override this behavior.

## What This Does

`check_models.py` runs MLX-based Vision-Language Models on images and generates detailed performance reports including:

- **Model outputs**: Generated text from VLM inference
- **Performance metrics**: Generation time, tokens per second, memory usage
- **Image metadata**: EXIF data, GPS coordinates (when available)
- **System information**: Device specs, library versions
- **Multiple output formats**: HTML (styled) and Markdown (GitHub-compatible)

Reports are saved to the `output/` directory by default.

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
│   ├── PYTHON_313_MIGRATION.md   # Python 3.13 migration notes
│   └── notes/                 # Design notes and reviews
├── output/                    # Generated reports (git-ignored)
├── typings/                   # Type stubs (git-ignored)
├── Makefile                   # Development commands
└── README.md                  # This file
```text
## Available Commands

Run `make` or `make help` to see all available commands:

| Command | Description |
|:--------|:------------|
| `make install` | Install the package |
| `make run` | Show usage help |
| `make demo` | Run example (if you have images) |
| `make clean` | Remove generated files |
| `make dev` | Setup development environment |
| `make update` | Update conda environment and project dependencies |
| `make upgrade-deps` | Upgrade all dependencies to latest compatible versions |
| `make lock-deps` | Generate requirements.txt from requirements.in |
| `make sync-deps` | Sync installed packages with requirements.txt |
| `make test` | Run tests |
| `make check` | Run core quality pipeline (format, lint, typecheck, test) |
| `make quality` | Run linting and type checks |
| `make ci` | Run full CI pipeline (strict mode) |
| `make format` | Format code with ruff |
| `make lint` | Lint code with ruff |
| `make typecheck` | Run mypy type checking |

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
```bash
### Dependencies

The `src/pyproject.toml` is the source of truth for dependency lists. To check for outdated packages:

```bash
python -m tools.check_outdated
```text
## Documentation

- **[src/README.md](src/README.md)**- Detailed usage, CLI options, and examples
-**[docs/CONTRIBUTING.md](docs/CONTRIBUTING.md)**- How to contribute to this project
-**[docs/IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md)**- Technical standards for developers
-**[docs/PYTHON_313_MIGRATION.md](docs/PYTHON_313_MIGRATION.md)**- Python 3.13 migration notes
-**[docs/notes/](docs/notes/)**- Design notes and formatting reviews

## Requirements

-**Python**: 3.13+
- **Platform**: macOS with Apple Silicon (M1/M2/M3)
- **Framework**: MLX (Apple's machine learning framework)

## License

See [LICENSE](LICENSE) file for details.
