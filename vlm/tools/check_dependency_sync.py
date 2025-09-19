"""Quick check that README dependency blocks match pyproject runtime deps.

Usage:
    python -m vlm.tools.check_dependency_sync

Exit codes:
    0: OK
    1: Drift detected
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from .update_readme_deps import MARKERS, build_install_command, parse_pyproject

logger = logging.getLogger(__name__)


def extract_block(readme: str, marker_key: str) -> str:
    """Return the raw command text inside a fenced block for a marker.

    Raises RuntimeError if the marker pair cannot be found.
    """
    start, end = MARKERS[marker_key]
    pattern = re.compile(rf"{re.escape(start)}\n```bash\n(.*?)\n```\n{re.escape(end)}", re.DOTALL)
    match = pattern.search(readme)
    if not match:
        msg = f"Could not find marker {marker_key}"
        raise RuntimeError(msg)
    return match.group(1).strip()


def main() -> int:
    """Validate README dependency blocks match pyproject runtime deps.

    Returns 0 when blocks are in sync, 1 otherwise.
    """
    repo_root = Path(__file__).resolve().parents[1]
    pyproject = repo_root / "pyproject.toml"
    readme = repo_root / "README.md"
    deps = parse_pyproject(pyproject)
    # deps already a mapping name -> spec
    expected_tail = build_install_command(deps).split(" ", 1)[1].strip()
    readme_text = readme.read_text(encoding="utf-8")
    mismatches: list[str] = []
    for key in MARKERS:
        block = extract_block(readme_text, key)
        body = block[len("pip install ") :].strip() if block.startswith("pip install ") else block
        if body != expected_tail:
            mismatches.append(key)
    if mismatches:
        logger.error("Dependency drift detected in blocks: %s", ", ".join(mismatches))
        return 1
    logger.info("Dependency blocks in README are up to date.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
