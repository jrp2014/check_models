"""Quarantine malformed pip metadata before packaging-tool upgrades."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import sysconfig
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

DIST_INFO_SUFFIX: Final = ".dist-info"
NORMALIZE_NAME_RE: Final = re.compile(r"[-_.]+")
QUARANTINE_PREFIX: Final = "check_models-pip-metadata-"


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


def quarantine_broken_metadata(
    target_names: Collection[str],
    *,
    site_dirs: Collection[Path] | None = None,
    quarantine_parent: Path | None = None,
) -> list[tuple[Path, Path]]:
    """Move malformed target metadata and return source/destination pairs."""
    normalized_targets = {_normalize_distribution_name(name) for name in target_names}
    roots = tuple(sorted(set(site_dirs if site_dirs is not None else _active_site_dirs())))
    broken: list[Path] = []

    for site_dir in roots:
        if not site_dir.is_dir():
            continue
        for metadata_dir in sorted(site_dir.glob(f"*{DIST_INFO_SUFFIX}")):
            if _distribution_name(metadata_dir) not in normalized_targets:
                continue
            if metadata_dir.is_symlink():
                msg = f"refusing symlinked metadata directory: {metadata_dir}"
                raise RuntimeError(msg)
            if not metadata_dir.is_dir():
                msg = f"metadata path is not a directory: {metadata_dir}"
                raise RuntimeError(msg)
            if not (metadata_dir / "METADATA").is_file() or not (metadata_dir / "RECORD").is_file():
                broken.append(metadata_dir)

    if not broken:
        return []

    quarantine_root = Path(
        tempfile.mkdtemp(
            prefix=QUARANTINE_PREFIX,
            dir=str(quarantine_parent) if quarantine_parent is not None else None,
        ),
    )
    moved: list[tuple[Path, Path]] = []
    for index, source in enumerate(broken, start=1):
        destination = quarantine_root / f"{index:02d}-{source.name}"
        shutil.move(source, destination)
        moved.append((source, destination))
    return moved


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target_names", nargs="+")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Quarantine malformed metadata requested on the command line."""
    args = _parse_args(argv)
    try:
        moved = quarantine_broken_metadata(args.target_names)
    except (OSError, RuntimeError) as exc:
        print(
            f"[update.sh] ERROR: unable to quarantine malformed pip metadata: {exc}",
            file=sys.stderr,
        )
        return 1

    if moved:
        print("[update.sh] Quarantined malformed pip metadata:")
        for source, destination in moved:
            print(f"   - {source} -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
