# Project Restructure Plan - MLX VLM Check

## Current Issues

1. **Duplicate/confusing file locations**:

   - `pyproject.toml` at root AND in vlm/ (vlm/ is the real one)
   - `pytest.ini` at root AND in vlm/
   - `results.html/md` generated in both root AND vlm/
   - Two READMEs with overlapping content

2. **Unclear entry point**:

   - Not obvious that `vlm/` is the main package
   - Root Makefile is just a proxy
   - Need to know to go to `vlm/` subdirectory

3. **Scattered documentation**:

   - Main README at root
   - Detailed README in vlm/
   - CONTRIBUTING.md at root
   - IMPLEMENTATION_GUIDE.md at root

## Proposed Structure

```
mlx-vlm-check/
├── README.md                          # 🎯 MAIN ENTRY POINT - Quick start
├── Makefile                           # Simple top-level targets
├── .gitignore
├── .editorconfig
├── .pre-commit-config.yaml
│
├── docs/                              # 📚 All documentation
│   ├── CONTRIBUTING.md
│   ├── IMPLEMENTATION_GUIDE.md
│   ├── PYTHON_313_MIGRATION.md
│   └── notes/
│       └── OUTPUT_FORMATTING_REVIEW.md
│
├── src/                               # 🐍 Python package (renamed from vlm/)
│   ├── check_models.py               # Main script
│   ├── __init__.py
│   ├── pyproject.toml                # Package config
│   ├── requirements.txt              # Locked dependencies
│   ├── requirements-dev.txt
│   ├── tests/                        # Unit tests
│   └── tools/                        # Maintenance scripts
│
├── typings/                           # Type stubs (git-ignored)
├── .github/                           # CI/CD workflows
└── output/                            # 📊 Generated reports go here
    ├── .gitignore                    # Ignore *.html, *.md
    └── README.md                     # Explains this is for outputs
```

## Proposed Changes

### 1. Simplify Root README.md

Make it the single source of truth with:

- What this project does (1-2 paragraphs)
- Quick start (copy-paste to get running in 30 seconds)
- Common commands (table format)
- Link to docs/ for details

### 2. Consolidate Documentation

Move all docs to `docs/` folder:

```bash
mkdir docs
mv CONTRIBUTING.md docs/
mv IMPLEMENTATION_GUIDE.md docs/
mv PYTHON_313_MIGRATION.md docs/
mv vlm/notes/OUTPUT_FORMATTING_REVIEW.md docs/notes/
```

### 3. Rename vlm/ to src/

More standard Python package structure:

```bash
mv vlm src
```

Update:

- Makefile paths
- GitHub workflows
- Import paths in tests
- pyproject.toml package name

### 4. Create output/ Directory

Stop polluting root/vlm with generated files:

```bash
mkdir -p output
echo "# Generated Reports\n\nThis directory contains generated HTML and Markdown reports.\n" > output/README.md
echo "*.html\n*.md\n!README.md" > output/.gitignore
```

Update check_models.py default output paths to `output/results.{html,md}`

### 5. Remove Duplicates

```bash
# Remove root duplicates (vlm/ versions are canonical)
rm pyproject.toml pytest.ini results.html results.md
```

### 6. Simplify Root Makefile

```makefile
.DEFAULT_GOAL := help

help:
	@echo "MLX VLM Check - Quick Commands"
	@echo ""
	@echo "Getting Started:"
	@echo "  make install          Install the package"
	@echo "  make run              Run on default image"
	@echo ""
	@echo "Development:"
	@echo "  make dev              Setup dev environment"
	@echo "  make test             Run tests"
	@echo "  make quality          Run quality checks"
	@echo ""
	@echo "See README.md for detailed usage"

install:
	pip install -e src/

dev:
	pip install -e "src/[dev,extras]"

run:
	python -m src.check_models --help

test:
	pytest src/tests/

quality:
	ruff format src/
	ruff check src/
	mypy src/

clean:
	rm -rf output/*.html output/*.md
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
```

### 7. Update Root README.md

```markdown
# MLX VLM Check

Benchmark and test Vision-Language Models on Apple Silicon using MLX.

## Quick Start

```bash

# 1. Install

pip install -e .

# 2. Run on an image

python -m check_models --model mlx-community/Florence-2-large --image ~/Pictures/test.jpg

# 3. View results

open output/results.html

```

## What This Does

- Runs MLX Vision-Language Models on images
- Collects performance metrics (speed, memory, tokens)
- Generates HTML and Markdown reports
- Extracts image metadata (EXIF, GPS)

## Common Commands

| Command | Description |
|---------|-------------|
| `make install` | Install the package |
| `make run` | Show usage help |
| `make dev` | Setup development environment |
| `make test` | Run tests |
| `make quality` | Run linting and type checks |

## Documentation

- [CONTRIBUTING.md](docs/CONTRIBUTING.md) - Development setup and workflow
- [IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md) - Coding standards
- [CLI Reference](docs/CLI_REFERENCE.md) - All command-line options

## Requirements

- macOS with Apple Silicon
- Python 3.13+
- MLX framework

## License

[Add license]
```

## Benefits

1. **Clear entry point**: README.md is the first thing users see
2. **Standard structure**: `src/` is a common Python pattern
3. **Clean root**: No generated files or test artifacts
4. **Organized docs**: Everything in `docs/`
5. **Simple commands**: `make install`, `make run`, `make dev`
6. **No duplication**: Single source of truth for each file

## Migration Steps

1. Create new directories
2. Move files
3. Update references
4. Test everything still works
5. Update CI/CD
6. Commit changes

## Breaking Changes

- Import paths change from `vlm.check_models` to `src.check_models`
  (or rename package to `mlx_vlm_check`)
- Default output location changes to `output/` directory
- Documentation URLs change

## Alternative: Minimal Cleanup (Less Disruptive)

If full restructure is too much, just do:

1. Remove duplicate files at root
2. Move docs to `docs/`
3. Create `output/` for generated files
4. Improve root README with quick start
5. Keep `vlm/` name (don't rename to `src/`)

This gives 80% of benefits with 20% of the work.
