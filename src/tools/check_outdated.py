#!/usr/bin/env python3
"""Check for outdated dependencies."""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    """Check for outdated packages using pip list --outdated."""
    result = subprocess.run(  # noqa: S603 - sys.executable is trusted
        [sys.executable, "-m", "pip", "list", "--outdated"],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        print(f"⚠️  Error checking outdated packages: {result.stderr}", file=sys.stderr)
        return 1

    output = result.stdout.strip()
    if output and not output.startswith("Package"):
        # Header line present but no packages listed
        print("✓ All packages up to date")
        return 0

    if "Package" in output:
        # Has header and packages
        lines = output.split("\n")
        if len(lines) > 2:  # Header + separator + at least one package
            print("⚠️  Outdated packages found:\n")
            print(output)
            print("\n💡 Run 'make upgrade-deps' to upgrade all dependencies")
            return 1

    print("✓ All packages up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
