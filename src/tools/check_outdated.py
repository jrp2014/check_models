"""Check for outdated dependencies."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Final

OUTDATED_TIMEOUT_SECONDS: Final[int] = int(os.environ.get("CHECK_OUTDATED_TIMEOUT", "20"))
NETWORK_ERROR_HINTS: Final[tuple[str, ...]] = (
    "timed out",
    "temporary failure",
    "temporary failure in name resolution",
    "name or service not known",
    "connection",
    "network is unreachable",
    "no route to host",
    "tls",
    "ssl",
    "certificate",
    "proxyerror",
)


def _canonical_name(name: str) -> str:
    """Normalize package names for stable comparisons."""
    return re.sub(r"[-_.]+", "-", name).lower().strip()


def _parse_requirement_name(requirement: str) -> str | None:
    """Parse package name from a dependency requirement string."""
    base = requirement.split(";", 1)[0].strip()
    if not base:
        return None
    if " @ " in base:
        return _canonical_name(base.split(" @ ", 1)[0].strip())

    match = re.match(r"^([A-Za-z0-9_.-]+)", base)
    if not match:
        return None
    return _canonical_name(match.group(1))


def _load_managed_packages(pyproject_path: Path) -> set[str]:
    """Load packages declared in pyproject dependencies and optional-dependencies."""
    if not pyproject_path.exists():
        return set()

    try:
        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return set()

    project = data.get("project", {})
    managed: set[str] = set()

    for dep in project.get("dependencies", []):
        name = _parse_requirement_name(str(dep))
        if name:
            managed.add(name)

    for deps in project.get("optional-dependencies", {}).values():
        for dep in deps:
            name = _parse_requirement_name(str(dep))
            if name:
                managed.add(name)

    return managed


def _print_outdated_rows(rows: list[dict[str, str]]) -> None:
    """Print compact rows of outdated package versions."""
    if not rows:
        return

    width = max(len(str(row.get("name", ""))) for row in rows)
    for row in sorted(rows, key=lambda item: str(item.get("name", "")).lower()):
        name = str(row.get("name", ""))
        current = str(row.get("version", "?"))
        latest = str(row.get("latest_version", "?"))
        print(f"  - {name:<{width}}  {current} -> {latest}")


def _looks_like_network_error(message: str) -> bool:
    """Return whether pip stderr/stdout text appears to be network-related."""
    lower = message.lower()
    return any(hint in lower for hint in NETWORK_ERROR_HINTS)


def main() -> int:
    """Check for outdated packages using pip list --outdated."""
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    managed = _load_managed_packages(pyproject_path)

    env = dict(os.environ)
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "list",
                "--outdated",
                "--format=json",
                "--disable-pip-version-check",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=OUTDATED_TIMEOUT_SECONDS,
            env=env,
        )
    except subprocess.TimeoutExpired:
        print(
            "⚠️  Timed out while querying outdated packages "
            f"(>{OUTDATED_TIMEOUT_SECONDS}s). Skipping.",
        )
        return 0

    combined_output = "\n".join(
        part.strip() for part in (result.stdout, result.stderr) if part and part.strip()
    )
    if result.returncode != 0:
        if combined_output and _looks_like_network_error(combined_output):
            print("⚠️  Could not query package index (network error). Skipping outdated check.")
            return 0
        print(f"⚠️  Error checking outdated packages:\n{combined_output}", file=sys.stderr)
        return 1

    try:
        outdated: list[dict[str, str]] = json.loads(result.stdout or "[]")
    except json.JSONDecodeError as e:
        print(f"⚠️  Failed to parse pip outdated output: {e}", file=sys.stderr)
        return 1

    if not outdated:
        print("✓ All packages up to date")
        return 0

    managed_rows: list[dict[str, str]] = []
    unmanaged_rows: list[dict[str, str]] = []
    for row in outdated:
        name = _canonical_name(str(row.get("name", "")))
        if name and name in managed:
            managed_rows.append(row)
        else:
            unmanaged_rows.append(row)

    print(f"⚠️  Outdated packages found ({len(outdated)} total):")
    if managed_rows:
        print(f"\nManaged by pyproject.toml ({len(managed_rows)}):")
        _print_outdated_rows(managed_rows)
    if unmanaged_rows:
        print(f"\nNot declared in pyproject.toml ({len(unmanaged_rows)}):")
        _print_outdated_rows(unmanaged_rows)

    print(
        "\n💡 Possible causes: dependency constraints, manual installs, or pinned local requirements."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
