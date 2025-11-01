# Archived Quality Tooling

These files were part of the original complex quality checking infrastructure that was simplified on 2025-11-01.

## Archived Files

- `check_quality.py` (444 lines) - Complex Python orchestrator for quality checks
- `generate_stubs.py` (261 lines) - Generated type stubs for mlx_vlm package
- `patch_mlx_stubs.py` - Patched generated stubs to fix mypy issues
- `check_dependency_sync.py` (70 lines) - Verified pyproject.toml dependencies stayed in sync

## Why Archived

These files created significant maintenance overhead:

1. **Stub generation was brittle** - Broke on upstream changes, required constant patching
2. **Over-engineered orchestrator** - 444 lines when a 20-line bash script suffices
3. **Unnecessary complexity** - 85% of quality infrastructure code removed with no loss of quality

## Replacement

See `check_quality_simple.sh` - a simple 20-line bash script that does the same job:

```bash
ruff format --check .
ruff check .
mypy check_models.py
pytest -v
```

## If You Need These

These files are kept for reference but are no longer maintained. The simplified approach:

- Uses `ignore_missing_imports = true` instead of stub generation
- Uses simple bash script instead of complex Python orchestrator
- Maintains same quality standards with 85% less infrastructure code

See `docs/SIMPLIFICATION_2025_11.md` for full details.
