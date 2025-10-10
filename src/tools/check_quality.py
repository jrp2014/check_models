#!/usr/bin/env python3
"""Quality checks runner for this repository.

Runs code formatting and static analysis tools with sensible defaults:

- ruff format (default; use --no-format to skip)
- ruff check (auto-fix enabled by default; use --no-fix to disable)
- mypy type checking

You can target specific paths; by default it runs only on `check_models.py`.

Exit status is non-zero if any executed check fails. Missing tools are
reported as warnings and skipped (use --require to fail if missing).
"""

from __future__ import annotations

import argparse
import importlib
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Final

DEFAULT_PATHS: Final[list[str]] = ["check_models.py"]
ALLOWED_TOOLS: Final[set[str]] = {
    "ruff",
    "mypy",
    "stubgen",
    "markdownlint-cli2",
    "npx",
    "markdownlint",
}
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
    proc = subprocess.run(
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
        help="Paths to check (default: check_models.py)",
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
        "--no-md",
        action="store_true",
        help="Do not run markdownlint-cli2 on Markdown files (runs by default if available).",
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
    parser.add_argument(
        "--check-stubs",
        action="store_true",
        help=(
            "Also run mypy against the generated stubs (typings/mlx_vlm). "
            "Useful to audit stub quality; off by default to avoid noise."
        ),
    )
    parser.add_argument(
        "--stubs-path",
        default="typings/mlx_vlm",
        help="Path to the stubs package to type-check when using --check-stubs.",
    )
    return parser.parse_args(argv)


def _ensure_stubs(repo_root: Path, *, refresh: bool, require: bool) -> int:
    """Ensure local type stubs exist for third-party packages.

    - If missing, attempt to generate using our local stub tool.
    - Returns 0 on success or if generation is skipped, non-zero on failure when
      generation was attempted and required.
    """
    typings = repo_root / "typings"
    # Only require mlx_vlm stubs; tokenizers stubs are optional and often noisy.
    need_stubs = refresh or not (typings / "mlx_vlm" / "__init__.pyi").exists()
    if not need_stubs:
        return 0
    # Ensure src is on sys.path so 'tools' is importable when run as a script
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    # Import locally to avoid global import-sort issues and keep this optional.
    mod: Any | None = None
    try:
        mod = importlib.import_module("tools.generate_stubs")
        _run_stubgen = getattr(mod, "run_stubgen", None)
    except ImportError:
        # Best-effort import; fall back to invoking stubgen via CLI
        _run_stubgen = None

    if _run_stubgen is None:
        # Fallback: try calling 'stubgen' directly for mlx_vlm
        logger.warning("[quality] Stub generator not importable; trying 'stubgen' CLI ...")
        try:
            rc = _run(["stubgen", "-p", "mlx_vlm", "-o", str(typings)], cwd=repo_root)
        except (FileNotFoundError, OSError, subprocess.SubprocessError, ValueError):
            if require:
                logger.exception(
                    "[quality] Stub generator not available; cannot generate stubs",
                )
                return 1
            logger.warning("[quality] Stub generation skipped (no stubgen)")
            return 0
        else:
            return int(rc)
    if refresh:
        # Best-effort clear of existing per-package stub dirs
        target = typings / "mlx_vlm"
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
    logger.info("[quality] Generating local type stubs (mlx_vlm) ...")
    rc = int(_run_stubgen(["mlx_vlm"]))
    if rc != 0 and require:
        logger.error("[quality] Stub generation failed (exit %s)", rc)
        return rc
    # Apply post-processing patch to fix Optional annotations, if available
    try:
        _patch = getattr(mod, "_patch_mlx_vlm_stubs", None) if mod is not None else None
        if _patch is not None:
            _patch(typings)
    except (OSError, ValueError, TypeError, AttributeError, RuntimeError):
        # Non-fatal: continue; mypy may still pass if stubs happen to be fine
        logger.debug("[quality] Stub patch step skipped due to exception", exc_info=True)
    return 0


def _run_ruff(
    repo_root: Path,
    *,
    paths: list[str],
    do_format: bool,
    do_fix: bool,
    require: bool,
) -> int:
    overall = 0
    if do_format:
        if _have("ruff"):
            logger.info("[quality] ruff format ...")
            overall = overall or _run(["ruff", "format", *paths], cwd=repo_root)
        else:
            logger.warning("[quality] ruff not found; skipping format")
            if require:
                overall = 1
    if _have("ruff"):
        logger.info("[quality] ruff check ...")
        cmd = ["ruff", "check", *paths]
        if do_fix:
            cmd.append("--fix")
        overall = overall or _run(cmd, cwd=repo_root)
    else:
        logger.warning("[quality] ruff not found; skipping lint")
        if require:
            overall = 1
    return overall


def _run_mypy(
    repo_root: Path,
    *,
    paths: list[str],
    require: bool,
    check_stubs: bool,
    stubs_path: str,
) -> int:
    overall = 0
    if _have("mypy"):
        logger.info("[quality] mypy type check ...")
        overall = overall or _run(
            [
                "mypy",
                "--config-file",
                str(repo_root / "src/pyproject.toml"),
                "--exclude",
                r"typings/tokenizers/.*",
                *paths,
            ],
            cwd=repo_root,
        )
        if check_stubs:
            logger.info("[quality] mypy type check (stubs: %s) ...", stubs_path)
            overall = overall or _run(
                [
                    "mypy",
                    "--config-file",
                    str(repo_root / "src/pyproject.toml"),
                    str(repo_root / stubs_path),
                ],
                cwd=repo_root,
            )
    else:
        logger.warning("[quality] mypy not found; skipping type check")
        if require:
            overall = 1
    return overall


def _run_markdownlint(repo_root: Path, *, require: bool) -> int:
    """Run markdownlint on all Markdown files if available.

    Tries markdownlint-cli2 directly, then via npx. Returns exit code.
    """
    md_patterns = ["**/*.md"]
    if _have("markdownlint-cli2"):
        logger.info("[quality] markdownlint-cli2 ...")
        try:
            return _run(["markdownlint-cli2", *md_patterns], cwd=repo_root)
        except Exception:
            logger.exception("[quality] markdownlint-cli2 execution failed")
            return 1
    if _have("npx"):
        logger.info("[quality] npx markdownlint-cli2 ...")
        try:
            return _run(["npx", "--yes", "markdownlint-cli2", *md_patterns], cwd=repo_root)
        except Exception:
            logger.exception("[quality] npx markdownlint-cli2 execution failed")
            return 1
    logger.warning("[quality] markdownlint not found (install markdownlint-cli2 or npx)")
    return 1 if require else 0


def main(argv: list[str] | None = None) -> int:
    """Run selected quality checks and return combined exit status."""
    # Basic logging setup
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv or sys.argv[1:])
    repo_root = Path(__file__).resolve().parents[2]  # .../scripts
    src_dir = repo_root / "src"
    # Resolve paths relative to src/ directory
    paths = [str((src_dir / p).resolve()) for p in args.paths]

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

    # markdownlint (via markdownlint-cli2)
    if not args.no_md:
        rc_md = _run_markdownlint(repo_root, require=args.require)
        overall_rc = overall_rc or rc_md

    # mypy
    if _have("mypy"):
        logger.info("[quality] mypy type check ...")
        # Explicitly point to the project config to honor mypy_path=typings
        rc = _run(
            [
                "mypy",
                "--config-file",
                str(repo_root / "src/pyproject.toml"),
                "--exclude",
                r"typings/tokenizers/.*",
                *paths,
            ],
            cwd=repo_root,
        )
        overall_rc = overall_rc or rc
        # Optionally, also check the stubs package itself
        if args.check_stubs:
            logger.info("[quality] mypy type check (stubs: %s) ...", args.stubs_path)
            rc2 = _run(
                [
                    "mypy",
                    "--config-file",
                    str(repo_root / "src/pyproject.toml"),
                    str(repo_root / args.stubs_path),
                ],
                cwd=repo_root,
            )
            overall_rc = overall_rc or rc2
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
