"""Patch known issues in MLX-generated type stubs.

MLX's auto-generated stubs sometimes contain syntax errors that prevent mypy
from running. This script applies targeted fixes to the installed MLX stubs.

Usage:
    python -m tools.patch_mlx_stubs

Common issues fixed:
- Parameters without defaults following parameters with defaults
"""

from __future__ import annotations

import logging
import re
import site
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def find_mlx_stubs() -> Path | None:
    """Locate the installed MLX core stubs directory."""
    for site_dir in site.getsitepackages():
        stub_file = Path(site_dir) / "mlx" / "core" / "__init__.pyi"
        if stub_file.exists():
            return stub_file
    return None


def patch_dequantize_signature(stub_file: Path) -> bool:
    """Fix parameter ordering in dequantize function signature.

    Issue: dtype parameter without default follows parameters with defaults.
    Fix: Add = None default to dtype parameter.
    """
    text = stub_file.read_text(encoding="utf-8")
    original = text

    # Pattern: dtype: Optional[Dtype], (without = None)
    # Should be: dtype: Optional[Dtype] = None,
    pattern = re.compile(
        r"(def dequantize\([^)]+dtype:\s*Optional\[Dtype\])" r"(\s*,\s*\*)",
    )
    replacement = r"\1 = None\2"

    text = pattern.sub(replacement, text)

    if text != original:
        stub_file.write_text(text, encoding="utf-8")
        logger.info("✓ Patched dequantize signature in %s", stub_file)
        return True
    return False


def main() -> int:
    """Apply all known patches to MLX stubs."""
    stub_file = find_mlx_stubs()
    if stub_file is None:
        logger.warning("MLX stubs not found in site-packages")
        return 1

    logger.info("Found MLX stubs at: %s", stub_file)

    patched = patch_dequantize_signature(stub_file)

    if patched:
        logger.info("✓ MLX stubs patched successfully")
        return 0

    logger.info("No patches needed (already applied or not applicable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
