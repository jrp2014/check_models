"""Test that runtime dependency block in README matches pyproject runtime deps.

This enforces local parity in addition to CI. The test focuses only on runtime deps
(not optional extras groups) and uses the same parsing heuristics as the sync script.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

_TEST_FILE = Path(__file__).resolve()
# tests/ parent, then package root (vlm)
PKG_ROOT = _TEST_FILE.parents[1]
REPO_ROOT = PKG_ROOT.parent


def _first_existing(paths: list[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    msg = f"None of the candidate paths exist: {paths}"
    raise FileNotFoundError(msg)


PYPROJECT = _first_existing(
    [PKG_ROOT / "pyproject.toml", REPO_ROOT / "pyproject.toml"],
)  # prefer in-package
README = _first_existing(
    [PKG_ROOT / "README.md", REPO_ROOT / "README.md"],
)  # prefer in-package

MANUAL_MARKERS = ("<!-- BEGIN MANUAL_INSTALL -->", "<!-- END MANUAL_INSTALL -->")


def _parse_runtime_deps(text: str) -> dict[str, str]:
    data = tomllib.loads(text)
    # project.dependencies is a list of strings
    deps_list = data.get("project", {}).get("dependencies", [])

    deps: dict[str, str] = {}
    for line in deps_list:
        name = re.split(r"[<>=!~]", line, maxsplit=1)[0].strip()
        spec = line[len(name) :].strip()
        deps[name] = spec
    return deps


def _extract_manual_block(readme: str) -> str:
    start, end = MANUAL_MARKERS
    pattern = re.compile(rf"{re.escape(start)}(.*?){re.escape(end)}", re.DOTALL)
    m = pattern.search(readme)
    if not m:
        msg = "Manual install markers not found in README.md"
        raise RuntimeError(msg)
    return m.group(1)


def test_readme_runtime_block_matches_pyproject() -> None:
    py_text = PYPROJECT.read_text(encoding="utf-8")
    rd_text = README.read_text(encoding="utf-8")
    runtime_deps = _parse_runtime_deps(py_text)

    manual_block = _extract_manual_block(rd_text)
    # Pull quoted specs from pip install line
    quoted = re.findall(r'"([^"]+)"', manual_block)
    if not quoted:
        msg = "No quoted packages found in manual install block"
        raise RuntimeError(msg)

    seen: dict[str, str] = {}
    for q in quoted:
        name = re.split(r"[<>=!~]", q, maxsplit=1)[0]
        spec = q[len(name) :]
        seen[name] = spec

    # All runtime deps must exist
    missing = [k for k in runtime_deps if k not in seen]
    if missing:
        msg = f"Runtime deps missing from README: {missing}"
        raise RuntimeError(msg)

    # No extras should leak (heuristic: check optional groups defined later if needed)
    forbidden = {
        "psutil",
        "tokenizers",
        "mlx-lm",
        "transformers",
        "torch",
        "torchvision",
        "torchaudio",
    }
    leaked = sorted(forbidden & set(seen))
    if leaked:
        msg = f"Optional deps leaked into runtime block: {leaked}"
        raise RuntimeError(msg)
