"""Install a Git pre-commit hook for this repo.

The hook automates two maintenance steps:

1) Sync README dependency blocks when `pyproject.toml` changes
  - Runs: `python -m vlm.tools.update_readme_deps`
  - Adds changes to `README.md` back to the commit

2) Ensure local type stubs exist for third-party packages used by mypy
   - If `typings/mlx_vlm/__init__.pyi` or `typings/tokenizers/__init__.pyi` is missing,
     it runs: `python -m vlm.tools.generate_stubs mlx_vlm tokenizers`
   - Adds `typings/` changes back to the commit

Usage:
  python -m vlm.tools.install_precommit_hook

Re-running this script overwrites the existing `.git/hooks/pre-commit` (after backing it up
as `.git/hooks/pre-commit.bak`).
"""

from __future__ import annotations

import logging
import stat
from pathlib import Path

logger = logging.getLogger("precommit")

HOOK_CONTENT = """#!/usr/bin/env bash
set -euo pipefail

# Run from repo root
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

# 1) Sync README dependency blocks when pyproject changes
if git diff --cached --name-only | grep -q '^pyproject.toml$'; then
  echo '[pre-commit] Syncing README dependency blocks'
  python -m vlm.tools.update_readme_deps || exit 1
  git add vlm/README.md
fi

# 2) Ensure local type stubs for mypy (mlx_vlm, tokenizers)
if [ ! -f typings/mlx_vlm/__init__.pyi ] || [ ! -f typings/tokenizers/__init__.pyi ]; then
  echo '[pre-commit] Generating local type stubs (mlx_vlm, tokenizers)'
  python -m vlm.tools.generate_stubs mlx_vlm tokenizers || exit 1
  git add typings
fi

exit 0
"""


def main() -> int:
    """Install or replace the pre-commit hook under .git/hooks/pre-commit.

    - Backs up an existing hook to .git/hooks/pre-commit.bak
    - Writes the new hook and marks it executable
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    repo_root = Path(__file__).resolve().parents[2]
    git_dir = repo_root / ".git"
    hooks_dir = git_dir / "hooks"
    hook_path = hooks_dir / "pre-commit"
    backup_path = hooks_dir / "pre-commit.bak"

    if not git_dir.exists():
        logger.error("[hooks] .git directory not found. Run this inside a git repository.")
        return 1

    hooks_dir.mkdir(parents=True, exist_ok=True)

    if hook_path.exists():
        # Backup existing hook
        hook_path.replace(backup_path)
        logger.info("[hooks] Existing pre-commit backed up to %s", backup_path)

    hook_path.write_text(HOOK_CONTENT)
    # Make executable
    mode = hook_path.stat().st_mode
    hook_path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    logger.info("[hooks] Installed pre-commit hook at %s", hook_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
