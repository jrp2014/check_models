"""Install Git hooks (pre-commit and pre-push) for this repo.

Pre-commit hook automates two maintenance steps:

1) Sync README dependency blocks when `src/pyproject.toml` changes
   - Runs: `cd src && python tools/update_readme_deps.py`
   - Adds changes to `src/README.md` back to the commit

2) Ensure local type stubs exist for third-party packages used by mypy
   - If `typings/mlx_vlm/__init__.pyi` or `typings/tokenizers/__init__.pyi` is missing,
     it runs: `python -m tools.generate_stubs mlx_vlm tokenizers`
   - Adds `typings/` changes back to the commit

Pre-push hook runs quality checks:
   - Runs: `make -C src quality`
   - Prevents pushing if quality checks fail

Usage:
  python -m tools.install_precommit_hook

Re-running this script overwrites existing hooks (after backing them up as `.bak`).
"""

from __future__ import annotations

import logging
import stat
from pathlib import Path

logger = logging.getLogger("precommit")

PRECOMMIT_HOOK_CONTENT = """#!/usr/bin/env bash
set -euo pipefail

# Run from repo root
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

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

exit 0
"""

PREPUSH_HOOK_CONTENT = """#!/usr/bin/env bash
# Pre-push hook: Run quality checks before pushing
set -euo pipefail

echo "[pre-push] Running quality checks before push..."

# Run from repo root
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

# Run full quality check
if make -C vlm quality; then
    echo "✓ Pre-push quality checks passed"
    exit 0
else
    echo ""
    echo "❌ Quality checks failed. Fix issues before pushing."
    echo "💡 Run 'make -C vlm quality' to see details"
    echo ""
    echo "To skip this check (not recommended), use:"
    echo "  git push --no-verify"
    exit 1
fi
"""


def _install_hook(hooks_dir: Path, hook_name: str, content: str) -> None:
    """Install a single git hook."""
    hook_path = hooks_dir / hook_name
    backup_path = hooks_dir / f"{hook_name}.bak"

    if hook_path.exists():
        # Backup existing hook
        hook_path.replace(backup_path)
        logger.info("[hooks] Existing %s backed up to %s", hook_name, backup_path)

    hook_path.write_text(content)
    # Make executable
    mode = hook_path.stat().st_mode
    hook_path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    logger.info("[hooks] Installed %s hook at %s", hook_name, hook_path)


def main() -> int:
    """Install or replace git hooks (pre-commit and pre-push).

    - Backs up existing hooks to .bak files
    - Writes new hooks and marks them executable
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    repo_root = Path(__file__).resolve().parents[2]
    git_dir = repo_root / ".git"
    hooks_dir = git_dir / "hooks"

    if not git_dir.exists():
        logger.error("[hooks] .git directory not found. Run this inside a git repository.")
        return 1

    hooks_dir.mkdir(parents=True, exist_ok=True)

    # Install pre-commit hook
    _install_hook(hooks_dir, "pre-commit", PRECOMMIT_HOOK_CONTENT)

    # Install pre-push hook
    _install_hook(hooks_dir, "pre-push", PREPUSH_HOOK_CONTENT)

    logger.info("[hooks] ✓ All hooks installed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
