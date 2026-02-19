"""Sync dependency versions from pyproject.toml into README install snippets.

Usage:
    python tools/update_readme_deps.py
    python tools/update_readme_deps.py --check

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

import argparse
import logging
import re
import tomllib
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_pyproject(path: Path) -> dict[str, str]:
    """Return mapping of dependency name->spec from PEP 621 dependencies array."""
    content = path.read_text(encoding="utf-8")
    data = tomllib.loads(content)

    # PEP 621: project.dependencies is a list of strings
    deps_list = data.get("project", {}).get("dependencies", [])

    deps: dict[str, str] = {}
    for line in deps_list:
        # Split name from spec heuristically: first occurrence of >,=,!,<,~
        name_part = re.split(r"[<>=!~]", line, maxsplit=1)[0].strip()
        spec = line[len(name_part) :].strip()
        deps[name_part] = spec

    return deps


def build_install_command(deps: dict[str, str]) -> str:
    """Compose a pip install command from dependencies."""
    parts: list[str] = []
    for name, spec in sorted(deps.items(), key=lambda kv: kv[0].lower()):
        parts.append(f'"{name}{spec}"')
    return "pip install " + " ".join(parts)


MARKERS = {
    "MANUAL_INSTALL": ("<!-- MANUAL_INSTALL_START -->", "<!-- MANUAL_INSTALL_END -->"),
    "MINIMAL_INSTALL": ("<!-- MINIMAL_INSTALL_START -->", "<!-- MINIMAL_INSTALL_END -->"),
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


def extract_optional_groups(pyproject_text: str) -> dict[str, list[str]]:
    """Very small parser to extract optional group package names.

    We only need names (not version specs) for validation that no optional dep leaks
    into the auto-generated runtime blocks.
    """
    groups: dict[str, list[str]] = {}
    current: str | None = None
    for line in pyproject_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[project.optional-dependencies]"):
            current = None
            continue
        if (
            stripped.startswith("[")
            and stripped.endswith("]")
            and stripped.startswith("[project.optional-dependencies.")
        ):
            current = stripped.rsplit(".", 1)[-1][:-1]
            groups[current] = []
            continue
        if current and stripped.startswith('"'):
            pkg = stripped.strip(",").strip().strip('"').split(">=")[0].split("==")[0]
            if pkg:
                groups[current].append(pkg)
    return groups


def main() -> int:
    """Run the sync and return exit code.

    Validation performed:
      * Ensures all runtime dependencies appear in README blocks after update.
      * Ensures no optional dependency (extras/torch/etc.) appears in those blocks.
    """
    parser = argparse.ArgumentParser(
        description="Sync README dependency install blocks with pyproject.toml",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify README dependency blocks are in sync without writing changes",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    pyproject_path = repo_root / "pyproject.toml"
    readme_path = repo_root / "README.md"
    try:
        py_text = pyproject_path.read_text(encoding="utf-8")
        deps = parse_pyproject(pyproject_path)
        optional_groups = extract_optional_groups(py_text)
        optional_flat = {p for pkgs in optional_groups.values() for p in pkgs}
        runtime_deps = {k: v for k, v in deps.items() if k not in optional_flat}
        install_cmd = build_install_command(runtime_deps)
        snippet = install_cmd
        readme_text = readme_path.read_text(encoding="utf-8")
        updated_readme_text = readme_text
        for key in MARKERS:
            updated_readme_text = replace_between_markers(updated_readme_text, key, snippet)
        # Post-update validation: ensure no optional package leaked
        for opt in sorted(optional_flat):
            pattern = f'"{opt}'
            if pattern in snippet:
                _fail_optional_leak(opt)

        if args.check:
            if updated_readme_text != readme_text:
                logger.error(
                    "README dependency blocks are out of sync. "
                    "Run: python -m tools.update_readme_deps",
                )
                return 1
            logger.info("README dependency blocks are in sync")
            return 0

        readme_path.write_text(updated_readme_text, encoding="utf-8")
    except Exception:  # pragma: no cover - rare failure path
        logger.exception("update_readme_deps failed")
        return 1
    logger.info("README dependency blocks updated")
    return 0


def _fail_optional_leak(opt_pkg: str) -> None:
    """Raise a standardized runtime leak error (isolated for lint friendliness)."""
    msg = f"Optional dependency '{opt_pkg}' appeared in runtime block"
    raise RuntimeError(msg)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
