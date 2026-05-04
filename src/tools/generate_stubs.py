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
        * stream_generate/generate(..., image: str | list[str] | None = None, ...)
        * stream_generate(... ) -> Generator[GenerationResult, None, None]
        * batch_generate(..., images/audios/prompts: ... | None = None, ...)

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
            # Import ProcessorMixin so generate()/stream_generate() can model
            # the multimodal processors returned by mlx_vlm.utils.load().
            (
                re.compile(
                    r"(from transformers import PreTrainedTokenizer as PreTrainedTokenizer\n)(?!from transformers\.processing_utils import ProcessorMixin as ProcessorMixin\n)",
                ),
                r"\1from transformers.processing_utils import ProcessorMixin as ProcessorMixin\n",
            ),
            # processor: PreTrainedTokenizer -> ProcessorMixin | PreTrainedTokenizer
            (
                re.compile(r"(processor:\s*)PreTrainedTokenizer\b"),
                r"\1ProcessorMixin | PreTrainedTokenizer",
            ),
            # Expand generate(...) to the explicit kwargs check_models forwards.
            (
                re.compile(
                    r"def generate\([^\n]*?verbose: bool = False, \*\*kwargs\) -> GenerationResult: \.\.\.",
                ),
                (
                    "def generate("
                    "model: nn.Module, "
                    "processor: ProcessorMixin | PreTrainedTokenizer, "
                    "prompt: str, "
                    "image: str | list[str] | None = None, "
                    "audio: str | list[str] | None = None, "
                    "verbose: bool = False, "
                    "*, "
                    "max_tokens: int = ..., "
                    "temperature: float = ..., "
                    "repetition_penalty: float | None = None, "
                    "repetition_context_size: int | None = ..., "
                    "top_p: float = ..., "
                    "min_p: float = ..., "
                    "top_k: int = ..., "
                    "max_kv_size: int | None = None, "
                    "kv_bits: int | None = None, "
                    "kv_group_size: int = ..., "
                    "quantized_kv_start: int = ..., "
                    "prefill_step_size: int | None = ..., "
                    "resize_shape: tuple[int, int] | None = None, "
                    "eos_tokens: list[str] | None = None, "
                    "skip_special_tokens: bool = False, "
                    "enable_thinking: bool = False, "
                    "thinking_budget: int | None = None, "
                    "thinking_end_token: str = ..., "
                    "thinking_start_token: str | None = None, "
                    "**kwargs"
                    ") -> GenerationResult: ..."
                ),
            ),
            # image/audio: str | list[str] = None -> include None
            (
                re.compile(r"(image:\s*str\s*\|\s*list\[str\])\s*=\s*None"),
                r"\1 | None = None",
            ),
            (
                re.compile(r"(audio:\s*str\s*\|\s*list\[str\])\s*=\s*None"),
                r"\1 | None = None",
            ),
            # stream_generate(...) -> str | Generator[str, None, None]
            #   -> Generator[GenerationResult, None, None]
            (
                re.compile(
                    r"(def\s+stream_generate\([^\)]*\)\s*->\s*)str\s*\|\s*Generator\[str,\s*None,\s*None\]",
                ),
                r"\1Generator[GenerationResult, None, None]",
            ),
            # batch_generate(..., images: str | list[str] = None, ...)
            (
                re.compile(r"(images:\s*str\s*\|\s*list\[str\])\s*=\s*None"),
                r"\1 | None = None",
            ),
            (
                re.compile(r"(audios:\s*str\s*\|\s*list\[str\])\s*=\s*None"),
                r"\1 | None = None",
            ),
            (
                re.compile(r"(prompts:\s*list\[str\])\s*=\s*None"),
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
    _patch_file(
        root / "processing_utils.pyi",
        [
            (
                re.compile(
                    r"(    audio_ids: Incomplete\n)(?!    tokenizer: PreTrainedTokenizerBase \| None\n)",
                ),
                (
                    r"\1"
                    "    tokenizer: PreTrainedTokenizerBase | None\n"
                    "    image_processor: Any | None\n"
                ),
            ),
        ],
    )


def _find_invalid_stub_files(
    typings_dir: Path,
    package_roots: Iterable[str],
) -> list[tuple[str, Path, int, str]]:
    """Return invalid stub files without mutating the typings tree."""
    invalid_files: list[tuple[str, Path, int, str]] = []
    for package in package_roots:
        root = typings_dir / package.replace(".", "/")
        if not root.exists():
            continue

        for pyi_path in root.rglob("*.pyi"):
            try:
                compile(pyi_path.read_text(encoding="utf-8"), str(pyi_path), "exec")
            except SyntaxError as err:
                invalid_files.append(
                    (
                        package,
                        pyi_path,
                        int(getattr(err, "lineno", 0) or 0),
                        str(getattr(err, "msg", "invalid syntax")),
                    ),
                )
            except OSError as err:
                logger.warning("[stubs] Could not read %s: %s", pyi_path, err)
    return invalid_files


def _validate_stub_syntax(typings_dir: Path, package_roots: Iterable[str]) -> None:
    """Warn and remove syntactically-invalid generated stubs.

    Invalid .pyi files can cause mypy to fail hard even when typing is optional.
    We remove only invalid files and keep the rest of the package stubs.
    """
    invalid_by_package: dict[str, list[tuple[Path, int, str]]] = {}
    for package, pyi_path, line_no, message in _find_invalid_stub_files(typings_dir, package_roots):
        invalid_by_package.setdefault(package, []).append((pyi_path, line_no, message))

    for package, invalid_files in invalid_by_package.items():
        root = typings_dir / package.replace(".", "/")
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
STUB_TOOL_VERSION = "6"

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
_VERSION_METADATA_CHANGED_SUFFIX = "version metadata changed"


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
                if manifest.get("tool_version") != STUB_TOOL_VERSION:
                    reason = "the local stub patcher version changed"
                elif manifest.get("python_version") != current_python_version:
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


def _is_version_metadata_only_refresh_reason(reason: str | None) -> bool:
    """Return whether the refresh reason is limited to package version metadata drift."""
    return isinstance(reason, str) and reason.endswith(_VERSION_METADATA_CHANGED_SUFFIX)


def _read_stub_file(path: Path, typings_dir: Path) -> tuple[str | None, list[str]]:
    """Read a stub file and return its contents plus any integrity issue."""
    if not path.exists():
        return None, [f"required stub file is missing: {path.relative_to(typings_dir)}"]
    try:
        return path.read_text(encoding="utf-8"), []
    except OSError as err:
        return None, [f"could not read {path.relative_to(typings_dir)}: {err}"]


def _verify_transformers_stub_contracts(typings_dir: Path) -> list[str]:
    """Verify that patched transformers stubs expose runtime ProcessorMixin attrs."""
    processing_utils_path = typings_dir / "transformers" / "processing_utils.pyi"
    text, issues = _read_stub_file(processing_utils_path, typings_dir)
    if text is None:
        return issues

    missing_tokens = [
        token
        for token in (
            "tokenizer: PreTrainedTokenizerBase | None",
            "image_processor: Any | None",
        )
        if token not in text
    ]
    if not missing_tokens:
        return []

    return [
        "transformers processing_utils stub is missing patched ProcessorMixin runtime attributes: "
        + ", ".join(missing_tokens)
    ]


def _verify_mlx_vlm_stub_contracts(typings_dir: Path) -> list[str]:
    """Verify that patched mlx_vlm stubs expose the generate contract we rely on."""
    generate_path = typings_dir / "mlx_vlm" / "generate.pyi"
    text, issues = _read_stub_file(generate_path, typings_dir)
    if text is None:
        return issues

    missing_tokens = [
        token
        for token in (
            "from transformers.processing_utils import ProcessorMixin as ProcessorMixin",
            "processor: ProcessorMixin | PreTrainedTokenizer",
            "temperature: float = ...",
            "thinking_end_token: str = ...",
        )
        if token not in text
    ]
    if not missing_tokens:
        return []

    return [
        "mlx_vlm generate stub is missing patched runtime-contract markers: "
        + ", ".join(missing_tokens)
    ]


def get_stub_integrity_issues(
    packages: Iterable[str],
    typings_dir: Path = TYPINGS_DIR,
    *,
    include_manifest: bool = True,
) -> list[str]:
    """Return deterministically-checkable issues for generated local stubs."""
    pkg_list = _validate_packages(packages)
    issues: list[str] = []

    if include_manifest:
        refresh_reason = get_stub_refresh_reason(pkg_list, typings_dir)
        if refresh_reason is not None:
            issues.append(f"stub manifest is stale: {refresh_reason}")

    for package, pyi_path, line_no, message in _find_invalid_stub_files(typings_dir, pkg_list):
        rel_path = pyi_path.relative_to(typings_dir)
        location = f"{rel_path}:{line_no}" if line_no else str(rel_path)
        issues.append(f"{package} stub syntax error in {location} ({message})")

    top_level_packages = {package.split(".", maxsplit=1)[0] for package in pkg_list}
    if "transformers" in top_level_packages:
        issues.extend(_verify_transformers_stub_contracts(typings_dir))
    if "mlx_vlm" in top_level_packages:
        issues.extend(_verify_mlx_vlm_stub_contracts(typings_dir))

    return issues


def refresh_stub_manifest_from_existing_stubs(
    packages: Iterable[str],
    typings_dir: Path = TYPINGS_DIR,
) -> bool:
    """Refresh manifest metadata when existing stubs still satisfy integrity checks.

    This is intentionally narrow: it repairs only package-version metadata drift
    after a best-effort regeneration attempt failed, and only when the checked-in
    stubs already pass syntax and contract validation.
    """
    pkg_list = _validate_packages(packages)
    refresh_reason = get_stub_refresh_reason(pkg_list, typings_dir)
    if not _is_version_metadata_only_refresh_reason(refresh_reason):
        return False

    structural_issues = get_stub_integrity_issues(
        pkg_list,
        typings_dir,
        include_manifest=False,
    )
    if structural_issues:
        return False

    _write_stub_manifest(pkg_list, typings_dir)
    if get_stub_refresh_reason(pkg_list, typings_dir) is not None:
        return False

    logger.info("[stubs] Refreshed stub manifest from existing verified stubs")
    return True


def _write_stub_manifest(packages: Iterable[str], typings_dir: Path = TYPINGS_DIR) -> None:
    manifest = _read_stub_manifest(typings_dir) or {}
    existing_packages = manifest.get("packages")
    updated_packages: dict[str, dict[str, str]] = (
        dict(existing_packages) if isinstance(existing_packages, dict) else {}
    )

    updated_manifest = {
        "packages": updated_packages,
        "tool_version": STUB_TOOL_VERSION,
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


def _default_stubgen_command() -> list[str]:
    """Resolve the preferred stubgen command for the active Python environment."""
    stubgen_path = shutil.which("stubgen")
    if stubgen_path:
        return [stubgen_path]

    alt_stubgen = Path(sys.executable).parent / "stubgen"
    if alt_stubgen.exists():
        logger.info("[stubs] Using stubgen from Python environment: %s", alt_stubgen)
        return [str(alt_stubgen)]

    logger.info("[stubs] Falling back to python -m mypy.stubgen")
    return [sys.executable, "-m", "mypy.stubgen"]


def _ensure_mypy_stubgen_command(stubgen_command: list[str]) -> list[str]:
    """Normalize the chosen stubgen command to mypy's implementation when possible."""
    try:
        test_result = subprocess.run(
            [*stubgen_command, "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as err:
        logger.warning("[stubs] Could not verify stubgen: %s", err)
        return stubgen_command

    if "-p" in test_result.stdout or "-p" in test_result.stderr:
        return stubgen_command

    alt_stubgen = Path(sys.executable).parent / "stubgen"
    if alt_stubgen.exists():
        logger.info("[stubs] Using stubgen from Python environment: %s", alt_stubgen)
        return [str(alt_stubgen)]
    return [sys.executable, "-m", "mypy.stubgen"]


def _log_stubgen_output(
    *,
    output_lines: list[str],
    return_code: int,
    transformers_requested: bool,
) -> None:
    """Log stubgen output with transformers-specific noise suppression."""
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
        return

    log_fn = logger.error if return_code else logger.info
    for line in output_lines:
        log_fn("[stubgen] %s", line)


def run_stubgen(packages: Iterable[str]) -> int:
    """Invoke stubgen for the provided packages and return its exit code."""
    TYPINGS_DIR.mkdir(parents=True, exist_ok=True)
    pkg_list = _validate_packages(packages)

    stubgen_command = _ensure_mypy_stubgen_command(_default_stubgen_command())

    args = [*stubgen_command]
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

    output_lines = _split_stubgen_output(completed.stdout, completed.stderr)
    return_code = completed.returncode or 0
    transformers_requested = any(pkg.split(".")[0] == "transformers" for pkg in pkg_list)
    _log_stubgen_output(
        output_lines=output_lines,
        return_code=return_code,
        transformers_requested=transformers_requested,
    )
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
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate stub freshness and required post-patch contracts without regenerating",
    )
    parser.add_argument(
        "--refresh-manifest-on-check",
        action="store_true",
        help=(
            "When --check sees only package version metadata drift, refresh the manifest "
            "from existing verified stubs instead of failing"
        ),
    )
    ns = parser.parse_args()
    packages = _validate_packages(ns.packages)

    if ns.check:
        if ns.refresh_manifest_on_check:
            refresh_stub_manifest_from_existing_stubs(packages)
        integrity_issues = get_stub_integrity_issues(packages)
        if integrity_issues:
            for issue in integrity_issues:
                logger.error("[stubs] %s", issue)
            return 1
        logger.info("[stubs] Stub integrity check passed")
        return 0

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
        integrity_issues = get_stub_integrity_issues(packages)
        if integrity_issues:
            for issue in integrity_issues:
                logger.error("[stubs] %s", issue)
            return 1

        # Count and report generated stub files
        stub_count = sum(1 for _ in TYPINGS_DIR.rglob("*.pyi"))
        logger.info("[stubs] Generated %d stub files (.pyi)", stub_count)
        logger.info("[stubs] Done")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
