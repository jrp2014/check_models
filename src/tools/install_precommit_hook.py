"""Install Git hooks (pre-commit and pre-push) for this repo.

Pre-commit hook automates:
   - Auto-formats Python files with ruff before committing
   - Auto-fixes Markdown files with markdownlint (via npx) before committing
   - Syncs README dependency blocks when `src/pyproject.toml` changes
   - Runs: `cd src && python tools/update_readme_deps.py`
   - Adds changes back to the commit

Pre-push hook runs quality checks:
   - Runs simplified quality checks (format check, lint, tests)
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

PRECOMMIT_HOOK_CONTENT = r"""#!/usr/bin/env bash
set -euo pipefail

# Run from repo root
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

# Activate conda environment if conda is available
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate mlx-vlm 2>/dev/null || true
fi

# Auto-format Python files with ruff before committing
PYTHON_FILES=()
while IFS= read -r -d '' file; do
    PYTHON_FILES+=("$file")
done < <(git diff --cached --name-only --diff-filter=ACM -z -- '*.py')
if [ "${#PYTHON_FILES[@]}" -gt 0 ]; then
    echo '[pre-commit] Auto-formatting Python files with ruff...'
    python -m ruff format "${PYTHON_FILES[@]}" || exit 1
    # Re-stage the formatted files
    git add "${PYTHON_FILES[@]}"
fi

# Auto-fix Markdown files with markdownlint before committing
# Exclude src/output/ directory (generated reports can have long lines)
MD_FILES=()
while IFS= read -r -d '' file; do
    case "$file" in
        src/output/*) continue ;;
    esac
    MD_FILES+=("$file")
done < <(git diff --cached --name-only --diff-filter=ACM -z -- '*.md')
if [ "${#MD_FILES[@]}" -gt 0 ]; then
    echo '[pre-commit] Auto-fixing Markdown files with markdownlint...'
    if command -v npx &> /dev/null; then
        npx --yes markdownlint-cli2 --fix "${MD_FILES[@]}" ||
            echo "⚠️  Some markdown issues could not be auto-fixed"
        # Re-stage the fixed files
        git add "${MD_FILES[@]}"
    else
        echo "⚠️  npx not found - skipping markdown auto-fix (install Node.js for markdown linting)"
    fi
fi

# Sync README dependency blocks when pyproject changes
if git diff --cached --name-only | grep -q '^src/pyproject.toml$'; then
  echo '[pre-commit] Syncing README dependency blocks'
  (cd src && python tools/update_readme_deps.py) || exit 1
  git add src/README.md
fi

exit 0
"""

PREPUSH_HOOK_CONTENT = r"""#!/usr/bin/env bash
# Pre-push hook: Run quality checks before pushing
set -euo pipefail

echo "[pre-push] Running quality checks before push..."

# Run from repo root
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

# Activate conda environment if conda is available
if command -v conda &> /dev/null; then
    # Initialize conda for the current shell
    eval "$(conda shell.bash hook)"
    # Activate the mlx-vlm environment
    conda activate mlx-vlm 2>/dev/null || true
fi

# Run simplified quality checks
if bash src/tools/check_quality_simple.sh; then
    echo "✓ Pre-push quality checks passed"
    exit 0
else
    echo ""
    echo "❌ Quality checks failed. Fix issues before pushing."
    echo "💡 Run 'bash src/tools/check_quality_simple.sh' to see details"
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
