#!/usr/bin/env bash
# Staged-file hygiene checks for commit hooks.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common_quality.sh"

cd "$(quality_repo_root)"
quality_setup_python

echo "=== Commit Hygiene ==="

staged_files=()
python_files=()
markdown_files=()
needs_readme_sync=0

while IFS= read -r -d '' file; do
    staged_files+=("$file")

    case "$file" in
        src/pyproject.toml)
            needs_readme_sync=1
            ;;
    esac

    case "$file" in
        *.py|*.pyi)
            python_files+=("$file")
            ;;
        *.md)
            case "$file" in
                src/output/*|src/node_modules/*|*/node_modules/*)
                    ;;
                *)
                    markdown_files+=("$file")
                    ;;
            esac
            ;;
    esac
done < <(git diff --cached --name-only --diff-filter=ACMR -z)

if [ "${#staged_files[@]}" -eq 0 ]; then
    echo "✓ No staged files to check"
    exit 0
fi

if [ "$needs_readme_sync" -eq 1 ]; then
    readme_already_tracked=0
    echo "[commit] Syncing README dependency blocks"
    (
        cd src
        "$QUALITY_PYTHON" -m tools.update_readme_deps
    )
    git add src/README.md
    for markdown_file in "${markdown_files[@]}"; do
        if [ "$markdown_file" = "src/README.md" ]; then
            readme_already_tracked=1
            break
        fi
    done
    if [ "$readme_already_tracked" -eq 0 ]; then
        markdown_files+=("src/README.md")
    fi
fi

echo "[commit] Checking staged YAML/TOML/shebang files"
"$QUALITY_PYTHON" - "${staged_files[@]}" <<'PY'
from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError as err:
    print("❌ PyYAML is required for commit hygiene checks.")
    print("   Activate the repo conda env first: conda activate mlx-vlm")
    raise SystemExit(1) from err

errors: list[str] = []

for raw_path in sys.argv[1:]:
    path = Path(raw_path)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as err:
        errors.append(f"{path}: could not read file ({err})")
        continue

    if path.suffix in {".yaml", ".yml"}:
        try:
            list(yaml.safe_load_all(text))
        except yaml.YAMLError as err:
            errors.append(f"{path}: invalid YAML ({err})")
    elif path.suffix == ".toml":
        try:
            tomllib.loads(text)
        except tomllib.TOMLDecodeError as err:
            errors.append(f"{path}: invalid TOML ({err})")

    if text.startswith("#!") and not os.access(path, os.X_OK):
        errors.append(f"{path}: has a shebang but is not executable")

if errors:
    print("❌ Commit hygiene checks failed:")
    for error in errors:
        print(f"   - {error}")
    raise SystemExit(1)
PY

if [ "${#python_files[@]}" -gt 0 ]; then
    echo "[commit] Formatting staged Python files"
    "$QUALITY_PYTHON" -m ruff format "${python_files[@]}"
    echo "[commit] Linting staged Python files"
    "$QUALITY_PYTHON" -m ruff check "${python_files[@]}"
    git add "${python_files[@]}"
fi

if [ "${#markdown_files[@]}" -gt 0 ]; then
    echo "[commit] Fixing staged Markdown files"
    quality_run_markdownlint \
        --config "$(quality_repo_root)/.markdownlint.jsonc" \
        --fix \
        "${markdown_files[@]}"
    git add "${markdown_files[@]}"
fi

echo "✓ Commit hygiene checks passed"
