"""Syntax checks on repository files, for the quality gate and the commit hook.

Run by path, from any directory::

    python tools/check_file_syntax.py [--shebang] FILE...

* ``.yaml`` / ``.yml``: every document in the file must parse.
* ``.toml``: the file must parse.
* with ``--shebang``: a file that starts with ``#!`` must be executable.

Other files are only opened when ``--shebang`` asks, and then only their first
two bytes are read. Prints one line per problem and exits 1 if there is any.
"""

from __future__ import annotations

import argparse
import os
import sys
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Final

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from check_models_data.safe_io import open_regular_file_no_follow, read_text_no_follow

if TYPE_CHECKING:
    from collections.abc import Sequence

YAML_SUFFIXES: Final = frozenset({".yaml", ".yml"})
TOML_SUFFIX: Final = ".toml"
SHEBANG: Final = b"#!"


def _yaml_error(text: str) -> str | None:
    import yaml  # noqa: PLC0415 - only YAML files need PyYAML

    try:
        list(yaml.safe_load_all(text))
    except yaml.YAMLError as err:
        return f"invalid YAML ({err})"
    return None


def _toml_error(text: str) -> str | None:
    try:
        tomllib.loads(text)
    except tomllib.TOMLDecodeError as err:
        return f"invalid TOML ({err})"
    return None


def _starts_with_shebang(path: Path) -> bool:
    fd = open_regular_file_no_follow(path, os.O_RDONLY)
    try:
        return os.read(fd, len(SHEBANG)) == SHEBANG
    finally:
        os.close(fd)


def file_problems(path: Path, *, check_shebang: bool) -> list[str]:
    """Return what is wrong with one file, as messages without the file name."""
    # The caller's path may run through a symlinked directory (a checkout
    # under one, say); the file itself is still never followed.
    target = path.parent.resolve() / path.name
    problems: list[str] = []
    try:
        suffix = path.suffix
        if suffix in YAML_SUFFIXES or suffix == TOML_SUFFIX:
            text = read_text_no_follow(target)
            problem = _yaml_error(text) if suffix in YAML_SUFFIXES else _toml_error(text)
            if problem is not None:
                problems.append(problem)
        if check_shebang and _starts_with_shebang(target) and not os.access(path, os.X_OK):
            problems.append("has a shebang but is not executable")
    except UnicodeDecodeError as err:
        problems.append(f"is not UTF-8 text ({err})")
    except OSError as err:
        problems.append(f"could not read file ({err})")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    """Check each named file; 0 when none has a problem."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shebang", action="store_true", help="also require executable shebangs")
    parser.add_argument("files", nargs="*", type=Path, metavar="FILE")
    args = parser.parse_args(argv)

    try:
        errors = [
            f"{path}: {problem}"
            for path in args.files
            for problem in file_problems(path, check_shebang=args.shebang)
        ]
    except ModuleNotFoundError as err:
        if err.name != "yaml":
            raise
        print("❌ PyYAML is required to check YAML files.")
        print("   Activate the repo conda env first: conda activate mlx-vlm")
        return 1
    if errors:
        print("❌ File syntax checks failed:")
        for error in errors:
            print(f"   - {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
