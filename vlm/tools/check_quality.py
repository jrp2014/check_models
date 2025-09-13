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

# Optional: local stub generation utility
try:
    from vlm.tools.generate_stubs import run_stubgen as _run_stubgen  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001 - best-effort import; we fallback gracefully
    _run_stubgen = None  # type: ignore[assignment]

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
    parser.add_argument(
        "--no-stubs",
        action="store_true",
        help="Do not attempt to generate local type stubs before mypy.",
    )
    parser.add_argument(
        "--refresh-stubs",
        action="store_true",
        help="Force re-generate local stubs (clears and rebuilds).",
    )
    return parser.parse_args(argv)


def _ensure_stubs(repo_root: Path, *, refresh: bool, require: bool) -> int:
    """Ensure local type stubs exist for third-party packages.

    - If missing, attempt to generate using our local stub tool.
    - Returns 0 on success or if generation is skipped, non-zero on failure when
      generation was attempted and required.
    """
    typings = repo_root / "typings"
    need_stubs = refresh or not (
        (typings / "mlx_vlm" / "__init__.pyi").exists()
        and (typings / "tokenizers" / "__init__.pyi").exists()
    )
    if not need_stubs:
        return 0
    if _run_stubgen is None:
        # Tool not importable; skip unless required
        if require:
            logger.error("[quality] Stub generator not available; cannot generate stubs")
            return 1
        logger.warning("[quality] Stub generator not available; skipping stub generation")
        return 0
    logger.info("[quality] Generating local type stubs (mlx_vlm, tokenizers) ...")
    rc = int(_run_stubgen(["mlx_vlm", "tokenizers"]))
    if rc != 0 and require:
        logger.error("[quality] Stub generation failed (exit %s)", rc)
        return rc
    return 0


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0912 - cohesive CLI flow
    """Run selected quality checks and return combined exit status."""
    # Basic logging setup
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv or sys.argv[1:])
    repo_root = Path(__file__).resolve().parents[2]  # .../scripts
    paths = [str((repo_root / p).resolve()) for p in args.paths]

    overall_rc = 0

    # Optionally (re)generate stubs prior to mypy
    if not args.no_stubs:
        rc = _ensure_stubs(repo_root, refresh=args.refresh_stubs, require=args.require)
        overall_rc = overall_rc or rc

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
        # Explicitly point to the project config to honor mypy_path=typings
        rc = _run(
            [
                "mypy",
                "--config-file",
                str(repo_root / "vlm/pyproject.toml"),
                *paths,
            ],
            cwd=repo_root,
        )
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
