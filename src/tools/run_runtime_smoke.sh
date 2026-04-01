#!/usr/bin/env bash
# Lightweight runtime/Metal smoke probe for CI and manual validation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common_quality.sh"

cd "$(quality_src_root)"
quality_setup_python

echo "=== Runtime Dependency Smoke ==="
"$QUALITY_PYTHON" - <<'PY'
from __future__ import annotations

import importlib.metadata
import tomllib
from pathlib import Path

pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
declared_runtime = {
    dep.split("[", 1)[0].split("<", 1)[0].split(">", 1)[0].split("=", 1)[0].strip()
    for dep in pyproject.get("project", {}).get("dependencies", [])
}
required = tuple(package for package in ("mlx", "mlx-vlm") if package in declared_runtime)

missing_packages: list[str] = []
versions: dict[str, str] = {}
for package in required:
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        missing_packages.append(package)

if missing_packages:
    print(
        "❌ Required runtime dependencies unavailable: "
        + ", ".join(sorted(missing_packages))
        + ". Install/repair these packages before running runtime smoke.",
    )
    raise SystemExit(1)

import check_models

missing_runtime = {
    name: message
    for name, message in check_models.MISSING_DEPENDENCIES.items()
    if name in required
}
if missing_runtime:
    print("❌ Required runtime dependencies unavailable:")
    for name in sorted(missing_runtime):
        print(f"   [{name}] {missing_runtime[name]}")
    raise SystemExit(1)

print(
    "✓ Runtime dependencies ready: "
    + ", ".join(f"{name}=={versions[name]}" for name in required),
)
PY
