"""Generate local .pyi stubs for packages lacking type information.

This writes stubs into the repository-local ``typings/`` directory and relies on
``mypy_path = ["typings"]`` in ``pyproject.toml`` to make mypy discover them.
Safe to run repeatedly; existing stubs for the target packages are overwritten.

Usage examples
--------------
    python -m tools.generate_stubs mlx_vlm
    python -m tools.generate_stubs --clear && \
        python -m tools.generate_stubs mlx_vlm

Notes
-----
- Requires mypy to be installed (provides ``stubgen``).
- Generated stubs are shallow by default; refine by hand for critical APIs.

"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # TC003: typing-only import
    from collections.abc import Iterable


def _patch_mlx_vlm_stubs(typings_dir: Path) -> None:
    """Patch known Optional signatures in generated mlx_vlm stubs.

    stubgen often emits parameters typed as non-Optional while using ``= None``
    defaults, which violates mypy's no_implicit_optional. Here we apply minimal,
    targeted edits to generated stubs so that they type-check cleanly:

    - models/base.pyi:
        BaseImageProcessor.__init__(..., crop_size: dict[str, int] | None = None, ...)
    - utils.pyi:
        * load_processor(..., eos_token_ids: int | list[int] | None = None, ...)
        * StoppingCriteria.add_eos_token_ids(self, new_eos_token_ids: int | list[int] | None = None)
        * StoppingCriteria.reset(self, eos_token_ids: list[int] | None = None)
    - convert.pyi: convert(..., upload_repo: str | None = None, ...)
    - generate.pyi:
        stream_generate/generate(
            ..., image: str | list[str] | None = None,
            audio: str | list[str] | None = None, ...
        )

    This function is idempotent and safe to run repeatedly.
    """

    def _patch_file(path: Path, patches: list[tuple[re.Pattern[str], str]]) -> None:
        if not path.exists():
            return
        text = path.read_text(encoding="utf-8")
        original = text
        for pattern, repl in patches:
            text = pattern.sub(repl, text)
        if text != original:
            path.write_text(text, encoding="utf-8")
            logger.info("[stubs] Patched %s", path.relative_to(typings_dir))

    mlx_root = typings_dir / "mlx_vlm"
    if not mlx_root.exists():
        return

    # models/base.pyi
    _patch_file(
        mlx_root / "models/base.pyi",
        patches=[
            # crop_size: dict[str, int] = None -> dict[str, int] | None = None
            (
                re.compile(r"(crop_size:\s*dict\[str,\s*int\])\s*=\s*None"),
                r"\1 | None = None",
            ),
        ],
    )

    # utils.pyi
    _patch_file(
        mlx_root / "utils.pyi",
        patches=[
            # load_processor(..., eos_token_ids=None, ...)
            #   -> eos_token_ids: int | list[int] | None = None
            (
                re.compile(
                    r"(def\s+load_processor\([^\)]*?eos_token_ids)\s*=\s*None",
                    re.DOTALL,
                ),
                r"\1: int | list[int] | None = None",
            ),
            # add_eos_token_ids(self, new_eos_token_ids: int | list[int] = None) -> include None
            (
                re.compile(
                    r"(add_eos_token_ids\(self,\s*new_eos_token_ids:\s*int\s*\|\s*list\[int\])\s*=\s*None",
                ),
                r"\1 | None = None",
            ),
            # reset(self, eos_token_ids: list[int] = None) -> list[int] | None
            (
                re.compile(r"(reset\(self,\s*eos_token_ids:\s*list\[int\])\s*=\s*None"),
                r"\1 | None = None",
            ),
        ],
    )

    # convert.pyi
    _patch_file(
        mlx_root / "convert.pyi",
        patches=[
            # upload_repo: str = None -> str | None = None
            (
                re.compile(r"(upload_repo:\s*str)\s*=\s*None"),
                r"\1 | None = None",
            ),
        ],
    )

    # generate.pyi
    _patch_file(
        mlx_root / "generate.pyi",
        patches=[
            # image/audio: str | list[str] = None -> include None
            (
                re.compile(r"(image:\s*str\s*\|\s*list\[str\])\s*=\s*None"),
                r"\1 | None = None",
            ),
            (
                re.compile(r"(audio:\s*str\s*\|\s*list\[str\])\s*=\s*None"),
                r"\1 | None = None",
            ),
        ],
    )


logger = logging.getLogger("generate_stubs")

REPO_ROOT = Path(__file__).resolve().parents[2]
TYPINGS_DIR = REPO_ROOT / "typings"

DEFAULT_PACKAGES = ["mlx_vlm"]
_PKG_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _validate_packages(packages: Iterable[str]) -> list[str]:
    """Return a validated list of package names (alnum/._- only).

    Raises ValueError if any name contains unexpected characters.
    """
    result: list[str] = []
    for name in packages:
        if not _PKG_RE.fullmatch(name):
            msg = f"Invalid package name for stubgen: {name!r}"
            raise ValueError(msg)
        result.append(name)
    return result


def run_stubgen(packages: Iterable[str]) -> int:
    """Invoke stubgen for the provided packages and return its exit code."""
    TYPINGS_DIR.mkdir(parents=True, exist_ok=True)
    pkg_list = _validate_packages(packages)

    # Find the stubgen executable - prefer the one in the same environment as Python
    stubgen_path = shutil.which("stubgen")
    if not stubgen_path:
        logger.error("[stubs] 'stubgen' command not found. Install mypy: pip install mypy")
        return 1

    # Verify it's mypy's stubgen by checking if it accepts -p flag
    # (mlx-vlm might have its own stubgen that uses --model)
    try:
        test_result = subprocess.run(
            [stubgen_path, "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        if "-p" not in test_result.stdout and "-p" not in test_result.stderr:
            # This might be the wrong stubgen, try to find mypy's version
            python_bin_dir = Path(sys.executable).parent
            alt_stubgen = python_bin_dir / "stubgen"
            if alt_stubgen.exists():
                stubgen_path = str(alt_stubgen)
                logger.info("[stubs] Using stubgen from Python environment: %s", stubgen_path)
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning("[stubs] Could not verify stubgen: %s", e)

    args = [stubgen_path]
    for pkg in pkg_list:
        args.extend(["-p", pkg])
    args.extend(["-o", str(TYPINGS_DIR)])
    try:
        # Suppress stderr to avoid confusing error messages from wrong stubgen commands
        # that might be in PATH before mypy's stubgen
        completed = subprocess.run(args, check=False, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        logger.exception(
            "[stubs] 'stubgen' (from mypy) not found. Install mypy: pip install mypy",
        )
        return 1
    else:
        return int(completed.returncode or 0)


def main() -> int:
    """CLI entry point: parse args, optionally clear, and generate stubs."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Generate local .pyi stubs for packages")
    parser.add_argument(
        "packages",
        nargs="*",
        default=DEFAULT_PACKAGES,
        help="Package names to generate stubs for (default: %(default)s)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Remove the entire typings/ directory first",
    )
    ns = parser.parse_args()

    if ns.clear and TYPINGS_DIR.exists():
        shutil.rmtree(TYPINGS_DIR)
        logger.info("[stubs] Cleared %s", TYPINGS_DIR)

    logger.info("[stubs] Generating stubs into: %s", TYPINGS_DIR)
    logger.info("[stubs] Packages: %s", ", ".join(ns.packages))
    code = run_stubgen(ns.packages)
    # Verify expected outputs exist when return code was 0
    if code == 0:
        missing: list[str] = []
        for pkg in ns.packages:
            pkg_dir = TYPINGS_DIR / pkg.replace(".", "/")
            if not pkg_dir.exists():
                missing.append(pkg)
        if missing:
            logger.warning(
                "[stubs] Some packages did not produce stubs (possibly not importable): %s",
                ", ".join(missing),
            )
            logger.warning(
                "[stubs] This is expected on non-macOS platforms (MLX requires Apple Silicon)",
            )
            # Don't fail - mypy has ignore_missing_imports=true and will work without stubs
            return 0
        # Apply post-processing patches for mlx_vlm stubs
        if any(pkg.split(".")[0] == "mlx_vlm" for pkg in ns.packages):
            _patch_mlx_vlm_stubs(TYPINGS_DIR)
        logger.info("[stubs] Done")
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
