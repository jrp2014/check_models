"""Install Git hooks (pre-commit and pre-push) for this repo.

Pre-commit hook runs the shared staged-file hygiene script:
   - validates staged YAML/TOML/shebang files
   - formats/lints staged Python files
   - fixes staged Markdown files
   - syncs README dependency blocks when `src/pyproject.toml` changes

Pre-push hook runs the shared fast static gate:
   - repo-wide format/lint/type checks
   - README dependency sync check
   - non-slow/non-e2e pytest subset

Usage:
  python -m tools.install_precommit_hook

Re-running this script overwrites existing hooks (after backing them up as `.bak`).
"""

from __future__ import annotations

import logging
import stat
from pathlib import Path

logger = logging.getLogger("precommit")

PRECOMMIT_HOOK_CONTENT = r"""#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"
exec bash src/tools/run_commit_hygiene.sh
"""

PREPUSH_HOOK_CONTENT = r"""#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"
exec bash src/tools/check_quality_simple.sh
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
