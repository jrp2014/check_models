# Documentation Alignment - October 2025

## Summary

Comprehensive alignment of all documentation (READMEs, CONTRIBUTING.md, IMPLEMENTATION_GUIDE.md) with the October 2025 project restructuring and recent refactorings.

## Changes Made

### 1. Path References Updated

**Old references to `vlm/`** → **Updated to `src/`**

- All documentation now correctly references `src/` as the main package directory
- Root `Makefile` commands no longer use `make -C vlm` syntax
- Examples updated to use `make <target>` from repository root

**Files updated:**

- `src/README.md`
- `docs/CONTRIBUTING.md`
- `docs/IMPLEMENTATION_GUIDE.md`

### 2. Output Location Clarified

**Default output location** now clearly documented as `output/` directory:

- `output/results.html` (default HTML report)
- `output/results.md` (default Markdown report)
- Both files are git-ignored
- Override behavior documented with `--output-html` / `--output-markdown` flags

**Files updated:**

- `src/README.md` - Updated quickstart, CLI reference, project structure
- `README.md` - Clarified output location in contributor section
- `output/README.md` - Enhanced with customization examples

### 3. Folder Behavior Documented

**Image selection from folders** now consistently mentioned:

- When `--folder` is passed, the script automatically selects the **most recently modified image**
- Hidden files are ignored
- Behavior documented in quickstart and CLI sections

**Files updated:**

- `src/README.md` - Added explicit folder behavior in quickstart
- `README.md` - Enhanced tip with bold emphasis on "most recently modified"

### 4. Make Command Standardization

All `make -C vlm` and `make -C src` commands updated to root `Makefile` equivalents:

| Old Command | New Command |
|-------------|-------------|
| `make -C vlm quality` | `make quality` |
| `make -C vlm bootstrap-dev` | `make bootstrap-dev` |
| `make -C vlm sync-deps` | `make sync-deps` |
| `make -C vlm check-outdated` | `make check-outdated` |
| `make -C vlm audit` | `make audit` |
| `make -C vlm stubs` | `make stubs` |
| `make -C vlm ci` | `make ci` |

**Files updated:**

- `docs/CONTRIBUTING.md` - Updated all make examples
- `docs/IMPLEMENTATION_GUIDE.md` - Updated dependency management, troubleshooting, and workflow sections

### 5. Contributor References Enhanced

Added clear pointers to style guides and coding standards:

- Root `README.md` now has "For Contributors" section linking to:
  - `docs/CONTRIBUTING.md` - Setup and workflow
  - `docs/IMPLEMENTATION_GUIDE.md` - Coding standards
- `src/README.md` contributor section enhanced with both document references
- Consistent messaging about where to find detailed information

### 6. Typings Regeneration

Clarified stub generation commands:

```bash
# From repo root
make stubs

# Or from src/ directory  
python -m tools.generate_stubs mlx_vlm tokenizers
```

Both approaches documented in IMPLEMENTATION_GUIDE.md

### 7. Project Structure Documentation

Updated project structure diagrams to show:

```text
mlx-vlm-check/
├── src/                    # Main package (was vlm/)
│   ├── check_models.py
│   ├── pyproject.toml
│   ├── tools/
│   └── tests/
├── output/                 # Generated reports (git-ignored)
│   ├── results.html
│   └── results.md
├── docs/                   # All documentation
├── typings/                # Generated stubs (git-ignored)
└── Makefile               # Root orchestration
```

## Verification

All changes verified with:

```bash
make quality  # ✅ Passed: ruff format, ruff check, mypy, markdownlint
```

## Impact

### For Users

- Clearer understanding of where output files are created
- Consistent folder behavior documentation
- Simplified command examples (no more `make -C` confusion)

### For Contributors

- Direct links to contribution guidelines and coding standards
- Consistent make command syntax across all docs
- Clear typings regeneration workflow

### For Maintainers

- Single source of truth for project structure
- Consistent terminology (src/ not vlm/)
- All documentation aligned with October 2025 restructuring

## Related Documents

- [RESTRUCTURE_COMPLETED.md](RESTRUCTURE_COMPLETED.md) - Original restructuring notes
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Contributor workflow guide
- [IMPLEMENTATION_GUIDE.md](../IMPLEMENTATION_GUIDE.md) - Technical conventions

## Date

2025-10-17
