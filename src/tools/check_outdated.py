"""Check for outdated dependencies."""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    """Check for outdated packages using pip list --outdated."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "list", "--outdated"],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        print(f"⚠️  Error checking outdated packages: {result.stderr}", file=sys.stderr)
        return 1

    output = result.stdout.strip()

    # If there are outdated packages, pip output will contain a table
    # checking for at least 3 lines: Header, Separator, and at least one Package
    min_lines_with_packages = 3
    if (
        output
        and output.startswith("Package")
        and len(output.splitlines()) >= min_lines_with_packages
    ):
        print("⚠️  Outdated packages found (likely held back by constraints or unmanaged):\n")
        print(output)
        print("\n💡 These packages were not updated. This usually means they are:")
        print("   1. Constrained by other dependencies (e.g. huggingface_hub needing fsspec<2026)")
        print("   2. Not managed by pyproject.toml (installed manually)")
        print("   3. Pinned in requirements.txt (if using local MLX builds)")
        return 1

    print("✓ All packages up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
