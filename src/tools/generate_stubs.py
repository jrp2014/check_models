"""Generate local .pyi stubs for packages lacking type information.

This writes stubs into the repository-local ``typings/`` directory and relies on
``mypy_path = ["typings"]`` in ``pyproject.toml`` to make mypy discover them.
Safe to run repeatedly; existing stubs for the target packages are overwritten.

Usage examples
--------------
    python -m tools.generate_stubs
    python -m tools.generate_stubs --clear && \
        python -m tools.generate_stubs mlx_lm mlx_vlm transformers tokenizers

Notes:
-----
- Requires mypy to be installed (provides ``stubgen``).
- Generated stubs are shallow by default; refine by hand for critical APIs.

"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
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


def _patch_transformers_stubs(typings_dir: Path) -> None:
    """Patch known invalid placeholder tokens emitted in transformers stubs."""

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

    root = typings_dir / "transformers"
    if not root.exists():
        return

    # stubgen can emit "<ERROR>.join(...)" in dataclass field metadata for some
    # datasets modules. Replace with a stable placeholder join string.
    join_placeholder_fix = (
        re.compile(r"<ERROR>\.join\("),
        "', '.join(",
    )
    _patch_file(root / "data/datasets/glue.pyi", [join_placeholder_fix])
    _patch_file(root / "data/datasets/squad.pyi", [join_placeholder_fix])


def _validate_stub_syntax(typings_dir: Path, package_roots: Iterable[str]) -> None:
    """Warn and remove syntactically-invalid generated stubs.

    Invalid .pyi files can cause mypy to fail hard even when typing is optional.
    We remove only invalid files and keep the rest of the package stubs.
    """
    for package in package_roots:
        root = typings_dir / package.replace(".", "/")
        if not root.exists():
            continue

        invalid_files: list[tuple[Path, int, str]] = []
        for pyi_path in root.rglob("*.pyi"):
            try:
                compile(pyi_path.read_text(encoding="utf-8"), str(pyi_path), "exec")
            except SyntaxError as err:
                invalid_files.append(
                    (
                        pyi_path,
                        int(getattr(err, "lineno", 0) or 0),
                        str(getattr(err, "msg", "invalid syntax")),
                    ),
                )
            except OSError as err:
                logger.warning("[stubs] Could not read %s: %s", pyi_path, err)

        if not invalid_files:
            continue

        logger.warning(
            "[stubs] %d invalid stub file(s) detected under %s; removing broken files",
            len(invalid_files),
            root,
        )
        for pyi_path, line_no, message in invalid_files:
            logger.warning(
                "[stubs] Invalid syntax in %s:%d (%s)",
                pyi_path.relative_to(typings_dir),
                line_no,
                message,
            )
            try:
                pyi_path.unlink()
            except OSError as err:
                logger.warning("[stubs] Failed to remove %s: %s", pyi_path, err)


logger = logging.getLogger("generate_stubs")

REPO_ROOT = Path(__file__).resolve().parents[2]
TYPINGS_DIR = REPO_ROOT / "typings"
STUB_MANIFEST = ".stub_manifest.json"

DEFAULT_PACKAGES = ["mlx_lm", "mlx_vlm", "transformers", "tokenizers"]
PACKAGE_DISTRIBUTIONS = {
    "mlx_lm": "mlx-lm",
    "mlx_vlm": "mlx-vlm",
}
_PKG_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_TRANSFORMERS_STUBGEN_NOISE_TOKENS = (
    "Something went wrong trying to find the model name in the path:",
    "Config not found for model. You can manually add it to"
    " HARDCODED_CONFIG_FOR_MODELS in utils/auto_docstring.py",
    "No checkpoint found for ",
    " but not documented. Make sure to add it to the docstring of the function in ",
)


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


def _stub_manifest_path(typings_dir: Path) -> Path:
    return typings_dir / STUB_MANIFEST


def _python_version() -> str:
    return ".".join(str(part) for part in sys.version_info[:3])


def _distribution_name_for_package(package: str) -> str:
    top_level = package.split(".", maxsplit=1)[0]
    return PACKAGE_DISTRIBUTIONS.get(top_level, top_level.replace("_", "-"))


def _installed_distribution_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _current_stub_manifest_entry(package: str) -> dict[str, str] | None:
    distribution = _distribution_name_for_package(package)
    version = _installed_distribution_version(distribution)
    if version is None:
        return None
    return {"distribution": distribution, "version": version}


def _stub_target_exists(typings_dir: Path, package: str) -> bool:
    package_path = typings_dir / package.replace(".", "/")
    return package_path.exists() or package_path.with_suffix(".pyi").exists()


def _read_stub_manifest(typings_dir: Path) -> dict[str, object] | None:
    manifest_path = _stub_manifest_path(typings_dir)
    if not manifest_path.exists():
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        logger.warning("[stubs] Ignoring unreadable manifest %s: %s", manifest_path, err)
        return None

    if not isinstance(manifest, dict):
        logger.warning(
            "[stubs] Ignoring invalid manifest %s: expected a JSON object",
            manifest_path,
        )
        return None
    return manifest


def get_stub_refresh_reason(packages: Iterable[str], typings_dir: Path = TYPINGS_DIR) -> str | None:
    """Return the reason stubs should be regenerated, or None when they are fresh."""
    pkg_list = _validate_packages(packages)
    reason: str | None = None

    if not typings_dir.exists():
        reason = "the typings/ directory is missing"
    else:
        manifest = _read_stub_manifest(typings_dir)
        if manifest is None:
            reason = "the stub manifest is missing or unreadable"
        else:
            manifest_packages = manifest.get("packages")
            if not isinstance(manifest_packages, dict):
                reason = "the stub manifest is missing package metadata"
            else:
                current_python_version = _python_version()
                if manifest.get("python_version") != current_python_version:
                    reason = f"the Python version changed to {current_python_version}"
                else:
                    current_stubgen_version = _installed_distribution_version("mypy")
                    if manifest.get("stubgen_version") != current_stubgen_version:
                        reason = "the mypy/stubgen version changed"
                    else:
                        for package in pkg_list:
                            if not _stub_target_exists(typings_dir, package):
                                reason = f"{package} stubs are missing"
                                break

                            current_entry = _current_stub_manifest_entry(package)
                            if current_entry is None:
                                distribution = _distribution_name_for_package(package)
                                reason = f"{distribution} is not installed"
                                break

                            recorded_entry = manifest_packages.get(package)
                            if recorded_entry != current_entry:
                                reason = f"{package} version metadata changed"
                                break

    return reason


def _write_stub_manifest(packages: Iterable[str], typings_dir: Path = TYPINGS_DIR) -> None:
    manifest = _read_stub_manifest(typings_dir) or {}
    existing_packages = manifest.get("packages")
    updated_packages: dict[str, dict[str, str]] = (
        dict(existing_packages) if isinstance(existing_packages, dict) else {}
    )

    updated_manifest = {
        "packages": updated_packages,
        "python_version": _python_version(),
        "stubgen_version": _installed_distribution_version("mypy"),
    }

    for package in _validate_packages(packages):
        entry = _current_stub_manifest_entry(package)
        if entry is None:
            updated_packages.pop(package, None)
            continue
        updated_packages[package] = entry

    manifest_path = _stub_manifest_path(typings_dir)
    try:
        manifest_path.write_text(
            json.dumps(updated_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as err:
        logger.warning("[stubs] Failed to write manifest %s: %s", manifest_path, err)
    else:
        logger.info("[stubs] Updated %s", manifest_path.relative_to(typings_dir.parent))


def _split_stubgen_output(stdout_text: str, stderr_text: str) -> list[str]:
    """Split stubgen output into normalized, non-empty logical lines."""
    combined = "\n".join(part for part in (stdout_text, stderr_text) if part)
    if not combined:
        return []

    # Upstream can concatenate multiple [ERROR] entries without newlines.
    normalized = re.sub(r"(?<!\n)(\[ERROR\])", r"\n\1", combined)
    normalized = re.sub(r"(?<!\n)(Processed \d+ modules)", r"\n\1", normalized)
    return [line.strip() for line in normalized.splitlines() if line.strip()]


def _is_transformers_stubgen_noise(line: str) -> bool:
    """Return True for known non-actionable transformers stubgen noise lines."""
    return line.startswith("[ERROR]") and any(
        token in line for token in _TRANSFORMERS_STUBGEN_NOISE_TOKENS
    )


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
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        logger.exception(
            "[stubs] 'stubgen' (from mypy) not found. Install mypy: pip install mypy",
        )
        return 1
    else:
        output_lines = _split_stubgen_output(completed.stdout, completed.stderr)
        return_code = int(completed.returncode or 0)
        transformers_requested = any(pkg.split(".")[0] == "transformers" for pkg in pkg_list)

        if return_code == 0 and transformers_requested:
            suppressed_count = 0
            for line in output_lines:
                if _is_transformers_stubgen_noise(line):
                    suppressed_count += 1
                    continue
                logger.info("[stubgen] %s", line)
            if suppressed_count:
                logger.info(
                    "[stubs] Suppressed %d non-actionable transformers stubgen message(s)",
                    suppressed_count,
                )
        else:
            log_fn = logger.error if return_code else logger.info
            for line in output_lines:
                log_fn("[stubgen] %s", line)

        return return_code


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
    parser.add_argument(
        "--skip-if-fresh",
        action="store_true",
        help="Skip regeneration when existing stubs match installed package versions",
    )
    ns = parser.parse_args()
    packages = _validate_packages(ns.packages)

    if ns.clear and TYPINGS_DIR.exists():
        shutil.rmtree(TYPINGS_DIR)
        logger.info("[stubs] Cleared %s", TYPINGS_DIR)

    if ns.skip_if_fresh:
        refresh_reason = get_stub_refresh_reason(packages)
        if refresh_reason is None:
            logger.info("[stubs] Existing stubs are fresh; skipping regeneration")
            return 0
        logger.info("[stubs] Regenerating stubs because %s", refresh_reason)

    logger.info("[stubs] Generating stubs into: %s", TYPINGS_DIR)
    logger.info("[stubs] Packages: %s", ", ".join(packages))
    code = run_stubgen(packages)
    # Verify expected outputs exist when return code was 0
    if code == 0:
        missing: list[str] = []
        for pkg in packages:
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
        if any(pkg.split(".")[0] == "mlx_vlm" for pkg in packages):
            _patch_mlx_vlm_stubs(TYPINGS_DIR)
        if any(pkg.split(".")[0] == "transformers" for pkg in packages):
            _patch_transformers_stubs(TYPINGS_DIR)

        _validate_stub_syntax(TYPINGS_DIR, packages)
        _write_stub_manifest(packages)

        # Count and report generated stub files
        stub_count = sum(1 for _ in TYPINGS_DIR.rglob("*.pyi"))
        logger.info("[stubs] Generated %d stub files (.pyi)", stub_count)
        logger.info("[stubs] Done")
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
