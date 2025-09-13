#!/usr/bin/env python3
"""Quality checks runner for this repository.

Runs code formatting and static analysis tools with sensible defaults:

- ruff format (default; use --no-format to skip)
- ruff check (auto-fix enabled by default; use --no-fix to disable)
- mypy type checking

You can target specific paths; by default it runs only on `vlm/check_models.py`.

Exit status is non-zero if any executed check fails. Missing tools are
reported as warnings and skipped (use --require to fail if missing).
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Final

DEFAULT_PATHS: Final[list[str]] = ["vlm/check_models.py"]
ALLOWED_TOOLS: Final[set[str]] = {"ruff", "mypy"}
logger = logging.getLogger("quality")


def _run(cmd: list[str], *, cwd: Path | None = None) -> int:
    """Run a command and return its exit code, streaming output live.

    Safety: Only allows known tools (ruff, mypy) to be executed.
    """
    if not cmd:
        msg = "Empty command"
        raise ValueError(msg)
    tool = Path(cmd[0]).name
    if tool not in ALLOWED_TOOLS:
        msg = f"Disallowed tool: {tool}"
        raise ValueError(msg)
    # Use check=False and aggregate exit codes ourselves
    proc = subprocess.run(  # noqa: S603 - validated tool, no shell, fixed argv
        cmd,
        cwd=str(cwd) if cwd else None,
        check=False,
    )
    return int(proc.returncode or 0)


def _have(tool: str) -> bool:
    """Return True if the given tool is available on PATH."""
    return shutil.which(tool) is not None


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments for the quality checker."""
    parser = argparse.ArgumentParser(description="Run ruff and mypy quality checks.")
    parser.add_argument(
        "paths",
        nargs="*",
        default=DEFAULT_PATHS,
    help="Paths to check (default: vlm/check_models.py)",
    )
    parser.add_argument(
        "--no-format",
        action="store_true",
        help="Do not run 'ruff format' (formatting runs by default).",
    )
    parser.add_argument(
        "--no-fix",
        action="store_true",
        help="Do not run 'ruff check --fix' (auto-fix is enabled by default).",
    )
    parser.add_argument(
        "--require",
        action="store_true",
        help="Fail if a required tool is missing (otherwise tools are skipped).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0912 - cohesive CLI flow
    """Run selected quality checks and return combined exit status."""
    # Basic logging setup
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv or sys.argv[1:])
    repo_root = Path(__file__).resolve().parents[2]  # .../scripts
    paths = [str((repo_root / p).resolve()) for p in args.paths]

    overall_rc = 0

    # Formatter step (default on; disable with --no-format)
    if not args.no_format:
        if _have("ruff"):
            logger.info("[quality] ruff format ...")
            rc = _run(["ruff", "format", *paths], cwd=repo_root)
            overall_rc = overall_rc or rc
        else:
            msg = "[quality] ruff not found; skipping format"
            logger.warning(msg)
            if args.require:
                overall_rc = 1

    # Ruff check
    if _have("ruff"):
        logger.info("[quality] ruff check ...")
        cmd = ["ruff", "check", *paths]
        if not args.no_fix:
            cmd.append("--fix")
        rc = _run(cmd, cwd=repo_root)
        overall_rc = overall_rc or rc
    else:
        msg = "[quality] ruff not found; skipping lint"
        logger.warning(msg)
        if args.require:
            overall_rc = 1

    # mypy
    if _have("mypy"):
        logger.info("[quality] mypy type check ...")
        rc = _run(["mypy", *paths], cwd=repo_root)
        overall_rc = overall_rc or rc
    else:
        msg = "[quality] mypy not found; skipping type check"
        logger.warning(msg)
        if args.require:
            overall_rc = 1

    if overall_rc == 0:
        logger.info("[quality] All selected checks passed")
    else:
        logger.error("[quality] Checks failed")
    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
