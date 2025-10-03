# Git Hooks Fixed for Restructure

## Issue

After restructuring the project (vlm/ → src/), the git hooks were still referencing the old `vlm/` paths, causing push failures.

## Hooks Fixed

### 1. `.git/hooks/pre-push`

**Before:**
```bash
if make -C vlm quality; then
    echo "✓ Pre-push quality checks passed"
    exit 0
else
    echo "❌ Quality checks failed. Fix issues before pushing."
    echo "💡 Run 'make -C vlm quality' to see details"
```

**After:**
```bash
if make quality; then
    echo "✓ Pre-push quality checks passed"
    exit 0
else
    echo "❌ Quality checks failed. Fix issues before pushing."
    echo "💡 Run 'make quality' to see details"
```

**Changes:**
- Changed `make -C vlm quality` to `make quality` (uses root Makefile)
- Updated help message to show correct command

### 2. `.git/hooks/pre-commit`

**Before:**
```bash
# 1) Sync README dependency blocks when pyproject changes
if git diff --cached --name-only | grep -q '^vlm/pyproject.toml$'; then
  echo '[pre-commit] Syncing README dependency blocks'
  (cd vlm && python tools/update_readme_deps.py) || exit 1
  git add vlm/README.md
fi

# 2) Ensure local type stubs for mypy (mlx_vlm, tokenizers)
if [ ! -f typings/mlx_vlm/__init__.pyi ] || [ ! -f typings/tokenizers/__init__.pyi ]; then
  echo '[pre-commit] Generating local type stubs (mlx_vlm, tokenizers)'
  python -m vlm.tools.generate_stubs mlx_vlm tokenizers || exit 1
  git add typings
fi
```

**After:**
```bash
# 1) Sync README dependency blocks when pyproject changes
if git diff --cached --name-only | grep -q '^src/pyproject.toml$'; then
  echo '[pre-commit] Syncing README dependency blocks'
  (cd src && python tools/update_readme_deps.py) || exit 1
  git add src/README.md
fi

# 2) Ensure local type stubs for mypy (mlx_vlm, tokenizers)
if [ ! -f typings/mlx_vlm/__init__.pyi ] || [ ! -f typings/tokenizers/__init__.pyi ]; then
  echo '[pre-commit] Generating local type stubs (mlx_vlm, tokenizers)'
  python -m tools.generate_stubs mlx_vlm tokenizers || exit 1
  git add typings
fi
```

**Changes:**
- Changed `vlm/pyproject.toml` to `src/pyproject.toml`
- Changed `(cd vlm && ...)` to `(cd src && ...)`
- Changed `git add vlm/README.md` to `git add src/README.md`
- Changed `python -m vlm.tools.generate_stubs` to `python -m tools.generate_stubs`

### 3. `src/tools/check_quality.py`

**Fixed references:**
- `DEFAULT_PATHS`: Changed from `["vlm/check_models.py"]` to `["check_models.py"]`
- Import path: Changed from `"vlm.tools.generate_stubs"` to `"tools.generate_stubs"`
- Config file: Changed from `repo_root / "vlm/pyproject.toml"` to `repo_root / "src/pyproject.toml"`
- Path resolution: Added `src_dir = repo_root / "src"` to resolve paths correctly

**Key changes:**
```python
# Before
DEFAULT_PATHS: Final[list[str]] = ["vlm/check_models.py"]
mod = importlib.import_module("vlm.tools.generate_stubs")
str(repo_root / "vlm/pyproject.toml")

# After
DEFAULT_PATHS: Final[list[str]] = ["check_models.py"]
mod = importlib.import_module("tools.generate_stubs")
str(repo_root / "src/pyproject.toml")
src_dir = repo_root / "src"
paths = [str((src_dir / p).resolve()) for p in args.paths]
```

## Testing

All hooks tested and working:

### Pre-push Hook
```bash
$ bash .git/hooks/pre-push
[pre-push] Running quality checks before push...
python tools/check_quality.py
[quality] ruff format ...
1 file left unchanged
[quality] ruff check ...
All checks passed!
[quality] mypy type check ...
Success: no issues found in 1 source file
✓ Pre-push quality checks passed
```

### Pre-commit Hook
```bash
$ git commit -m "fix: update check_quality.py paths from vlm/ to src/"
[main 92cac3d] fix: update check_quality.py paths from vlm/ to src/
 1 file changed, 13 insertions(+), 10 deletions(-)
```
(No errors = hook passed)

### Git Push
```bash
$ git push origin main
[pre-push] Running quality checks before push...
✓ Pre-push quality checks passed
Enumerating objects: 71, done.
...
To https://github.com/jrp2014/scripts.git
   0fa2a6e..92cac3d  main -> main
```

## Summary

✅ **All git hooks updated and working**
✅ **Quality checks passing** (ruff format, ruff check, mypy)
✅ **Git push successful** with all hooks running correctly
✅ **Pre-commit hook ready** for future commits

The restructure is now fully integrated with the git workflow!
