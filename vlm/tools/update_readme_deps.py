"""Sync dependency versions from pyproject.toml into README install snippets.

Usage:
    python tools/update_readme_deps.py

The script:
  * Parses [project.dependencies] and selected optional groups (extras) in pyproject.toml
  * Builds two blocks:
      - MANUAL_INSTALL: full runtime deps (core runtime section only)
      - MINIMAL_INSTALL: identical to runtime (can be adapted later if policy changes)
  * Replaces content between the marker pairs in README.md:
        <!-- BEGIN MANUAL_INSTALL --> ... <!-- END MANUAL_INSTALL -->
        <!-- BEGIN MINIMAL_INSTALL --> ... <!-- END MINIMAL_INSTALL -->

Design notes:
  * Keeps ordering stable: core deps sorted by name for determinism.
  * Quotes each spec to avoid shell globbing issues.
  * Leaves optional extras out (user can pip install -e ".[extras]").

Exit codes:
  0 success, 1 failure (with message to stderr)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_pyproject(path: Path) -> dict[str, str]:
    """Return mapping of dependency name to spec from [project.dependencies]."""
    content = path.read_text(encoding="utf-8")
    pattern = r"^\[project\.dependencies\](.*?)(^\[|\Z)"
    match = re.search(pattern, content, flags=re.MULTILINE | re.DOTALL)
    if not match:
        msg = "Could not locate [project.dependencies] section"
        raise RuntimeError(msg)
    block = match.group(1)
    deps: dict[str, str] = {}
    for raw in block.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name_part, spec_part = line.split("=", 1)
        name = name_part.strip()
        spec = spec_part.strip().strip('"')
        if "#" in spec:
            spec = spec.split("#", 1)[0].strip()
        deps[name] = spec
    return deps


def build_install_command(deps: dict[str, str]) -> str:
    """Compose a pip install command from dependencies."""
    parts: list[str] = []
    for name, spec in sorted(deps.items(), key=lambda kv: kv[0].lower()):
        parts.append(f'"{name}{spec}"')
    return "pip install " + " ".join(parts)


MARKERS = {
    "MANUAL_INSTALL": ("<!-- BEGIN MANUAL_INSTALL -->", "<!-- END MANUAL_INSTALL -->"),
    "MINIMAL_INSTALL": ("<!-- BEGIN MINIMAL_INSTALL -->", "<!-- END MINIMAL_INSTALL -->"),
}


def replace_between_markers(text: str, marker_key: str, replacement_block: str) -> str:
    """Replace fenced code block content between marker pair."""
    start, end = MARKERS[marker_key]
    pattern = re.compile(rf"({re.escape(start)})(.*?)(" + re.escape(end) + ")", re.DOTALL)

    def _repl(match: re.Match[str]) -> str:
        return f"{match.group(1)}\n```bash\n{replacement_block}\n```\n{match.group(3)}"

    new_text, count = pattern.subn(_repl, text, count=1)
    if count == 0:
        msg = f"Marker pair {marker_key} not found in README"
        raise RuntimeError(msg)
    return new_text


def main() -> int:
    """Run the sync and return exit code."""
    repo_root = Path(__file__).resolve().parents[1]
    pyproject_path = repo_root / "pyproject.toml"
    readme_path = repo_root / "README.md"
    try:
        deps = parse_pyproject(pyproject_path)
        runtime_deps = {k: v for k, v in deps.items() if k not in set()}
        install_cmd = build_install_command(runtime_deps)
        snippet = install_cmd[len("pip install ") :]
        readme_text = readme_path.read_text(encoding="utf-8")
        for key in MARKERS:
            readme_text = replace_between_markers(readme_text, key, snippet)
        readme_path.write_text(readme_text, encoding="utf-8")
    except Exception:
        logger.exception("update_readme_deps failed")
        return 1
    logger.info("README dependency blocks updated")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
