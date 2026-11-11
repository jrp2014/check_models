"""Runtime dependency smoke probe for CI and manual validation.

Run by ``tools/run_runtime_smoke.sh``. Passes when the MLX packages that
``pyproject.toml`` declares as runtime dependencies are installed and
``check_models`` loads them without recording any as missing.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import sys
import tomllib
from pathlib import Path
from typing import Final

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from check_models_data.safe_io import read_text_no_follow

SRC_ROOT: Final = Path(__file__).resolve().parent.parent
SMOKE_PACKAGES: Final = ("mlx", "mlx-vlm")


def required_runtime_packages(pyproject_text: str) -> tuple[str, ...]:
    """Return the smoke packages that the project declares as runtime dependencies."""
    pyproject = tomllib.loads(pyproject_text)
    declared = {
        dep.split("[", 1)[0].split("<", 1)[0].split(">", 1)[0].split("=", 1)[0].strip()
        for dep in pyproject.get("project", {}).get("dependencies", [])
    }
    return tuple(package for package in SMOKE_PACKAGES if package in declared)


def main() -> int:
    """Report the runtime dependencies; 1 when one is not installed or not importable."""
    required = required_runtime_packages(read_text_no_follow(SRC_ROOT / "pyproject.toml"))

    versions: dict[str, str] = {}
    not_installed: list[str] = []
    for package in required:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            not_installed.append(package)
    if not_installed:
        print(
            "❌ Required runtime dependencies unavailable: "
            + ", ".join(sorted(not_installed))
            + ". Install/repair these packages before running runtime smoke.",
        )
        return 1

    # Loaded as a run loads it: each dependency in a child process first, so
    # that one which crashes while loading is named here instead of ending this.
    check_models = importlib.import_module("check_models")
    check_models.load_mlx_runtime(probe_first=True)
    not_importable = {
        name: message
        for name, message in check_models.MISSING_DEPENDENCIES.items()
        if name in required
    }
    if not_importable:
        print("❌ Required runtime dependencies unavailable:")
        for name in sorted(not_importable):
            print(f"   [{name}] {not_importable[name]}")
        return 1

    print(
        "✓ Runtime dependencies ready: "
        + ", ".join(f"{name}=={versions[name]}" for name in required),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
