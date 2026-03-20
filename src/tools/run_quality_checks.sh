#!/usr/bin/env bash
# Deterministic full static quality checker for local runs and CI.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common_quality.sh"

cd "$(quality_src_root)"
quality_setup_python

quality_require_python_tool ty "Install dev dependencies with: pip install -e .[dev]"
quality_require_python_tool pyrefly "Install dev dependencies with: pip install -e .[dev]"
quality_require_command shellcheck "Install with: brew install shellcheck"

echo "=== Workflow YAML Validation ==="
quality_validate_yaml_files \
    "$(quality_repo_root)/.github/workflows/quality.yml" \
    "$(quality_repo_root)/.github/workflows/dependency-sync.yml" \
    "$(quality_repo_root)/.pre-commit-config.yaml"

echo "=== Dependency Sync (Check) ==="
"$QUALITY_PYTHON" -m tools.update_readme_deps --check

echo "=== Type Stub Preflight ==="

mkdir -p ../typings

"$QUALITY_PYTHON" - <<'PY'
from __future__ import annotations

from pathlib import Path

stub_root = Path("../typings")
expected_packages = ("mlx_lm", "mlx_vlm", "transformers", "tokenizers")
missing: list[str] = []
invalid: list[tuple[str, str, int, str]] = []

for package in expected_packages:
    package_root = stub_root / package.replace(".", "/")
    if not package_root.exists():
        missing.append(package)
        continue

    for stub_path in package_root.rglob("*.pyi"):
        try:
            compile(stub_path.read_text(encoding="utf-8"), str(stub_path), "exec")
        except SyntaxError as err:
            invalid.append(
                (
                    package,
                    str(stub_path.relative_to(stub_root)),
                    int(getattr(err, "lineno", 0) or 0),
                    str(getattr(err, "msg", "invalid syntax")),
                ),
            )
            break
        except OSError:
            invalid.append((package, str(stub_path.relative_to(stub_root)), 0, "read error"))
            break

if missing:
    print(
        "⚠️  Stub coverage warning: missing package stubs for "
        + ", ".join(sorted(missing)),
    )
if invalid:
    for package, relpath, line_no, message in invalid:
        suffix = f":{line_no}" if line_no else ""
        print(
            "⚠️  Stub syntax warning "
            f"({package}): {relpath}{suffix} ({message})",
        )

if not missing and not invalid:
    print("✓ Stub preflight: expected package stubs available and parseable")
PY

echo "=== Ruff Format ==="
"$QUALITY_PYTHON" -m ruff format --check .

echo "=== Ruff Lint ==="
"$QUALITY_PYTHON" -m ruff check .

echo "=== MyPy Type Check ==="
"$QUALITY_PYTHON" -m mypy check_models.py

echo "=== Suppression Audit ==="
"$QUALITY_PYTHON" -m tools.check_suppressions

echo "=== Ty Type Check ==="
quality_run_python_tool ty check check_models.py

echo "=== Pyrefly Type Check ==="
quality_run_python_tool pyrefly check check_models.py

echo "=== Pytest ==="
"$QUALITY_PYTHON" -m pytest -v

echo "=== ShellCheck ==="
shell_scripts=()
while IFS= read -r -d '' script_path; do
    shell_scripts+=("$script_path")
done < <(find tools -name "*.sh" -type f -print0)

if [ "${#shell_scripts[@]}" -gt 0 ]; then
    shellcheck -x "${shell_scripts[@]}"
fi

# Markdown linting runs from the repo root
cd "$(quality_repo_root)"

echo "=== Markdown Lint ==="
quality_run_markdownlint \
    --config .markdownlint.jsonc \
    "**/*.md" \
    "!src/node_modules/**" \
    "!**/node_modules/**"

echo ""
echo "✅ All quality checks passed!"
