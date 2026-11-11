"""Python steps of ``tools/update.sh``, kept in a file so they are linted, typed and tested.

``update.sh`` runs this file by path, ``python "$SCRIPT_DIR/update_helpers.py"
<command> ...``, with whichever interpreter it is updating, from any working
directory and before the project is reinstalled. It therefore imports only the
standard library at module level and never touches ``sys.path``; a command that
needs an installed package imports it when it runs. Commands that take a
document read it from standard input, so the shell decides which file that is.

Commands and their exit status:

* ``conda-plan-packages`` — stdin: ``conda ... --dry-run --json`` output.
  Prints every package name the plan links or unlinks, one per line. 1 when
  the plan is unreadable or conda reported an error.
* ``conda-change-ownership NAME...`` — stdin: ``conda list --json`` output.
  Prints ``<name> <owner>`` per name: ``conda``, ``pip``, ``absent`` or
  ``ambiguous``.
* ``clean-pip-backups`` — removes pip's leftover ``~name`` entries from the
  interpreter's site-packages and lists what it removed.
* ``mlx-backend-smoke MODEL PROMPT MAX_TOKENS EXPECTED`` — runs one
  ``mlx_vlm generate``; nonzero when it fails or its output lacks EXPECTED.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

PIP_BACKUP_PREFIX: Final = "~"
BACKEND_SMOKE_TIMEOUT_ENV: Final = "MLX_LOCAL_BUILD_SMOKE_TIMEOUT"
BACKEND_SMOKE_DEFAULT_TIMEOUT_SECONDS: Final = 240


def _stdin_text() -> str:
    """Return standard input decoded as UTF-8, whatever the locale says."""
    return sys.stdin.buffer.read().decode("utf-8")


def conda_plan_package_names(plan_text: str) -> list[str] | None:
    """Return every package a conda dry-run plan links or unlinks, sorted.

    Updates, new installs and removals alike, so no part of the transaction
    goes unchecked. ``None`` when the plan is unreadable or reports an error.
    """
    try:
        # Progress records, when conda emits any, come first, NUL-separated.
        plan = json.loads([chunk for chunk in plan_text.split("\0") if chunk.strip()][-1])
    except (ValueError, IndexError):
        return None
    if not isinstance(plan, dict) or not plan.get("success") or "error" in plan:
        return None
    actions = plan.get("actions", [])
    names: set[str] = set()
    for prefix_actions in actions if isinstance(actions, list) else [actions]:
        if not isinstance(prefix_actions, dict):
            return None
        for key in ("LINK", "UNLINK"):
            for record in prefix_actions.get(key, []):
                if not isinstance(record, dict) or not record.get("name"):
                    return None
                names.add(str(record["name"]))
    return sorted(names)


def _normalized_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def conda_change_owners(records_text: str, changes: Sequence[str]) -> list[tuple[str, str]]:
    """Say which manager owns each package conda plans to change.

    Ownership comes from conda's own package records: one from the pypi channel
    (or conda's ``<develop>`` label for an editable install) is a distribution
    pip installed, any other is conda's. A name with both kinds of record is
    ``ambiguous``, as is every name when the records are unreadable or empty.
    A name with no record is not installed: ``absent``.
    """
    try:
        records = json.loads(records_text)
    except ValueError:
        records = None
    readable = isinstance(records, list) and bool(records)
    owners: dict[str, set[str]] = {}
    for record in records if readable else []:
        if not isinstance(record, dict) or not record.get("name"):
            continue
        by_pip = record.get("platform") == "pypi" or record.get("channel") in ("pypi", "<develop>")
        owner = "pip" if by_pip else "conda"
        owners.setdefault(_normalized_package_name(str(record["name"])), set()).add(owner)

    result: list[tuple[str, str]] = []
    for change in changes:
        found = owners.get(_normalized_package_name(change), set())
        if not readable or len(found) > 1:
            result.append((change, "ambiguous"))
        else:
            result.append((change, next(iter(found)) if found else "absent"))
    return result


def remove_stale_pip_backups(site_dirs: Iterable[Path]) -> tuple[list[Path], list[Path]]:
    """Remove pip's ``~name`` leftovers from each site directory.

    An interrupted uninstall leaves them behind, and every later pip run then
    warns about an invalid distribution. A symlink is never removed. Returns
    the removed paths and the skipped symlinks.
    """
    removed: list[Path] = []
    skipped: list[Path] = []
    for site_dir in site_dirs:
        if not site_dir.is_dir():
            continue
        for path in sorted(site_dir.iterdir()):
            if not path.name.startswith(PIP_BACKUP_PREFIX):
                continue
            if path.is_symlink():
                skipped.append(path)
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed.append(path)
    return removed, skipped


def _active_site_dirs() -> list[Path]:
    return sorted(
        {
            Path(value)
            for key in ("purelib", "platlib")
            if (value := sysconfig.get_path(key)) is not None
        }
    )


def backend_smoke(
    model: str,
    prompt: str,
    max_tokens: str,
    expected: str,
    *,
    timeout_seconds: int,
) -> int:
    """Run one ``mlx_vlm generate`` and return 0 only when it printed ``expected``."""
    result = subprocess.run(  # noqa: S603 - this interpreter runs mlx_vlm with update.sh's arguments
        [
            sys.executable,
            "-m",
            "mlx_vlm",
            "generate",
            "--model",
            model,
            "--prompt",
            prompt,
            "--max-tokens",
            max_tokens,
        ],
        capture_output=True,
        check=False,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
    )
    output = f"{result.stdout}\n{result.stderr}"
    print(output)
    if result.returncode != 0:
        return result.returncode
    if expected and expected not in output:
        print(
            "[update.sh] ERROR: local MLX runtime smoke did not produce the "
            "expected deterministic output.",
            file=sys.stderr,
        )
        print(f"[update.sh] Expected substring: {expected!r}", file=sys.stderr)
        print(
            "[update.sh] This usually points at a broken local MLX backend artifact "
            "such as mlx.metallib.",
            file=sys.stderr,
        )
        return 1
    return 0


def _conda_plan_packages_command(_args: Sequence[str]) -> int:
    try:
        names = conda_plan_package_names(_stdin_text())
    except ValueError:  # not UTF-8
        return 1
    if names is None:
        return 1
    print("\n".join(names), end="\n" if names else "")
    return 0


def _conda_change_ownership_command(args: Sequence[str]) -> int:
    try:
        records_text = _stdin_text()
    except ValueError:  # not UTF-8: unreadable records make every name ambiguous
        records_text = ""
    for name, owner in conda_change_owners(records_text, args):
        print(name, owner)
    return 0


def _clean_pip_backups_command(_args: Sequence[str]) -> int:
    removed, skipped = remove_stale_pip_backups(_active_site_dirs())
    for path in skipped:
        print(f"[update.sh] Skipping stale pip backup symlink: {path}", file=sys.stderr)
    if removed:
        print("[update.sh] Removed stale pip invalid-distribution backup(s):")
        for path in removed:
            print(f"   - {path}")
    return 0


def _mlx_backend_smoke_command(args: Sequence[str]) -> int:
    model, prompt, max_tokens, expected = args
    timeout_seconds = int(
        os.environ.get(BACKEND_SMOKE_TIMEOUT_ENV, BACKEND_SMOKE_DEFAULT_TIMEOUT_SECONDS)
    )
    return backend_smoke(model, prompt, max_tokens, expected, timeout_seconds=timeout_seconds)


COMMANDS: Final[dict[str, Callable[[Sequence[str]], int]]] = {
    "conda-plan-packages": _conda_plan_packages_command,
    "conda-change-ownership": _conda_change_ownership_command,
    "clean-pip-backups": _clean_pip_backups_command,
    "mlx-backend-smoke": _mlx_backend_smoke_command,
}


def main(argv: Sequence[str]) -> int:
    """Run one command; 2 for an unknown command name."""
    if not argv or argv[0] not in COMMANDS:
        print(f"usage: update_helpers.py {{{'|'.join(COMMANDS)}}} [ARG...]", file=sys.stderr)
        return 2
    return COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
