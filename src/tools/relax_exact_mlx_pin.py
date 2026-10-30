"""Relax one installed distribution's exact pin on another package.

mlx-audio releases (0.6.1-0.6.3, and main as of 2026-10-28) pin ``mlx==0.32.4``
exactly, which can never resolve against a pinned local mlx source build
(PEP 440: ``0.32.5.devN+rev`` does not match ``==0.32.4``), so installing
mlx-vlm >= 0.7.11 (which requires ``mlx-audio>=0.6.2``) is ResolutionImpossible
whenever update.sh holds mlx at the local build. Until upstream relaxes the
pin, update.sh installs the required mlx-audio with ``--no-deps`` and calls
this tool to rewrite the exact pin in the *installed* metadata to the
equivalent floor (``mlx>=0.32.4``), so every later resolver pass (the editable
mlx-vlm install, the eager project reinstall, ``pip check``) sees a
satisfiable requirement. update.sh also holds mlx-audio at the relaxed
version for the rest of the run, so an eager upgrade cannot re-resolve the
PyPI metadata that still carries the pin.

Scope: only a bare ``Requires-Dist: <package>==X`` line (no extras, no
environment marker, a single ``==`` clause, no wildcard — ``==X.*`` has no
valid ``>=`` form) is rewritten, and the replacement is parsed as a
requirement before it is written. The rewrite goes through a sibling temporary
file and an atomic rename, so a failure leaves the original METADATA intact.
Exit 0 when the pin was relaxed, was
already relaxed, or the distribution no longer declares the dependency;
exit 1 when the distribution is not installed or its metadata cannot be
rewritten safely.
"""

from __future__ import annotations

import argparse
import re
import sys
import sysconfig
from pathlib import Path
from typing import TYPE_CHECKING, Final

from packaging.requirements import InvalidRequirement, Requirement

try:
    from tools.safe_io import read_text_no_follow, replace_text_no_follow
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools.safe_io import read_text_no_follow, replace_text_no_follow

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

DIST_INFO_SUFFIX: Final = ".dist-info"
NORMALIZE_NAME_RE: Final = re.compile(r"[-_.]+")
REQUIRES_DIST_PREFIX: Final = "Requires-Dist:"
# No "*": a wildcard pin (==X.*) has no valid ">=" form, so it is never
# relaxed; update.sh's assessment reports it as unsupported instead.
EXACT_VERSION_RE: Final = re.compile(r"^==\s*([A-Za-z0-9!+.]+)$")


def _normalize_distribution_name(name: str) -> str:
    return NORMALIZE_NAME_RE.sub("-", name).lower()


def _distribution_name(metadata_dir: Path) -> str:
    stem = metadata_dir.name.removesuffix(DIST_INFO_SUFFIX)
    name, separator, _version = stem.rpartition("-")
    return _normalize_distribution_name(name) if separator else ""


def _active_site_dirs() -> tuple[Path, ...]:
    paths = {
        Path(value)
        for key in ("purelib", "platlib")
        if (value := sysconfig.get_path(key)) is not None
    }
    return tuple(sorted(paths))


def _installed_metadata_files(
    distribution: str,
    site_dirs: Collection[Path],
) -> list[Path]:
    normalized = _normalize_distribution_name(distribution)
    found: list[Path] = []
    for site_dir in sorted(set(site_dirs)):
        if not site_dir.is_dir():
            continue
        for metadata_dir in sorted(site_dir.glob(f"*{DIST_INFO_SUFFIX}")):
            if _distribution_name(metadata_dir) != normalized:
                continue
            metadata_file = metadata_dir / "METADATA"
            if metadata_file.is_file():
                found.append(metadata_file)
    return found


def _bare_exact_pin_version(line: str, package: str) -> str | None:
    """Return X when ``line`` is a bare ``Requires-Dist: <package>==X``."""
    if not line.startswith(REQUIRES_DIST_PREFIX):
        return None
    try:
        requirement = Requirement(line.removeprefix(REQUIRES_DIST_PREFIX).strip())
    except InvalidRequirement:
        return None
    # Extras or an environment marker mean the pin is conditional; a multi-
    # clause specifier means it is not the plain exact pin this tool targets.
    if (
        _normalize_distribution_name(requirement.name) != _normalize_distribution_name(package)
        or requirement.extras
        or requirement.marker is not None
    ):
        return None
    match = EXACT_VERSION_RE.match(str(requirement.specifier))
    # Never write a replacement that is not itself a valid requirement.
    try:
        if match is not None:
            Requirement(f"{package}>={match.group(1)}")
    except InvalidRequirement:
        return None
    return match.group(1) if match else None


def relax_exact_pin(
    distribution: str,
    package: str,
    *,
    site_dirs: Collection[Path] | None = None,
) -> list[str]:
    """Rewrite ``package==X`` to ``package>=X`` in ``distribution``'s metadata.

    Returns one human-readable line per metadata file describing what was done.
    Raises ``RuntimeError`` when the distribution's metadata cannot be found.
    """
    roots = tuple(site_dirs) if site_dirs is not None else _active_site_dirs()
    metadata_files = _installed_metadata_files(distribution, roots)
    if not metadata_files:
        msg = f"no installed metadata found for distribution {distribution!r}"
        raise RuntimeError(msg)

    outcomes: list[str] = []
    for metadata_file in metadata_files:
        text = read_text_no_follow(metadata_file)
        lines = text.splitlines(keepends=True)
        relaxed: list[str] = []
        for index, line in enumerate(lines):
            pinned_version = _bare_exact_pin_version(line.rstrip("\r\n"), package)
            if pinned_version is None:
                continue
            ending = line[len(line.rstrip("\r\n")) :]
            lines[index] = f"{REQUIRES_DIST_PREFIX} {package}>={pinned_version}{ending}"
            relaxed.append(pinned_version)
        if relaxed:
            replace_text_no_follow(metadata_file, "".join(lines))
            versions = ", ".join(relaxed)
            outcomes.append(
                f"relaxed {package}=={versions} to {package}>={versions} in {metadata_file}"
            )
        else:
            outcomes.append(f"no bare exact {package} pin to relax in {metadata_file}")
    return outcomes


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("distribution", help="installed distribution whose metadata to rewrite")
    parser.add_argument("package", help="package whose bare exact pin becomes a >= floor")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Relax the requested pin and report what changed."""
    args = _parse_args(argv)
    try:
        outcomes = relax_exact_pin(args.distribution, args.package)
    except (OSError, RuntimeError) as exc:
        print(f"[update.sh] ERROR: unable to relax the {args.package} pin: {exc}", file=sys.stderr)
        return 1
    for outcome in outcomes:
        print(f"[update.sh] {outcome}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
