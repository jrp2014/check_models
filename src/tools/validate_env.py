"""Validate the development environment is properly configured.

This script checks:
- Python version (>= 3.13)
- Conda environment (if CONDA_ENV is set)
- Required packages are installed with correct versions
- Tools (ruff, mypy, pytest) are available
- Git hooks are installed

Usage:
    python -m tools.validate_env
    python -m tools.validate_env --expected-conda-env my-env
    python -m tools.validate_env --fix  # Auto-fix issues
"""

from __future__ import annotations

import argparse
import importlib.metadata
import logging
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Final

from check_models_data.dependency_policy import (
    VALIDATE_ENV_CORE_FALLBACK_SPECS,
    VALIDATE_ENV_EXTRAS_FALLBACK_SPECS,
)

logger = logging.getLogger("validate-env")

try:
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet
    from packaging.version import InvalidVersion, Version
except ImportError:  # pragma: no cover - optional for bootstrap scenarios
    Requirement = None
    SpecifierSet = None
    InvalidVersion = ValueError
    Version = None

REQUIRED_PYTHON_VERSION: Final[tuple[int, int]] = (3, 13)
DEFAULT_CONDA_ENV: Final[str] = "mlx-vlm"
EXPECTED_CONDA_ENV_ENVVAR: Final[str] = "CHECK_MODELS_EXPECTED_CONDA_ENV"


class ValidationError(Exception):
    """Raised when environment validation fails."""


def _parse_dependency_spec_fallback(requirement: str) -> tuple[str, str]:
    """Best-effort parser for PEP 508-ish requirement strings."""
    base = requirement.split(";", 1)[0].strip()
    if not base:
        return "", ""

    if " @ " in base:
        return base.split(" @ ", 1)[0].strip(), ""

    match = re.match(r"^([A-Za-z0-9_.-]+)(?:\[[^\]]+\])?\s*(.*)$", base)
    if not match:
        return base, ""

    name = match.group(1).strip()
    spec = match.group(2).strip()
    if spec.startswith(("==", ">=", "<=", "!=", "~=", ">", "<")):
        return name, spec
    return name, ""


def _parse_dependency_spec(requirement: str) -> tuple[str, str]:
    """Parse dependency requirement into normalized package name + version spec."""
    if Requirement is None:
        return _parse_dependency_spec_fallback(requirement)

    try:
        parsed = Requirement(requirement)
    except Exception:
        return _parse_dependency_spec_fallback(requirement)

    return parsed.name, str(parsed.specifier)


def load_pyproject_deps() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Load dependencies from pyproject.toml."""
    candidate_paths = (
        Path(__file__).resolve().parents[1] / "pyproject.toml",
        Path(__file__).resolve().parents[2] / "pyproject.toml",
        Path("pyproject.toml").resolve(),
    )
    pyproject_path = next((path for path in candidate_paths if path.exists()), None)

    if pyproject_path is None:
        logger.warning(
            "Could not find pyproject.toml in expected locations: %s",
            ", ".join(str(path) for path in candidate_paths),
        )
        return {}, {}, {}

    try:
        with pyproject_path.open("rb") as f:
            data = tomllib.load(f)

    except Exception as e:
        logger.warning("Failed to parse pyproject.toml: %s", e)
        return {}, {}, {}

    project = data.get("project", {})
    dependencies = project.get("dependencies", [])
    optional_dependencies = project.get("optional-dependencies", {})

    core_deps: dict[str, str] = {}
    for dep in dependencies:
        name, spec = _parse_dependency_spec(dep)
        if name:
            core_deps[name] = spec

    dev_deps: dict[str, str] = {}
    extras_deps: dict[str, str] = {}

    for group, deps in optional_dependencies.items():
        target_dict = dev_deps if group == "dev" else extras_deps
        for dep in deps:
            name, spec = _parse_dependency_spec(dep)
            if name:
                target_dict[name] = spec

    return core_deps, extras_deps, dev_deps


# Load dependencies dynamically
CORE_PACKAGES, EXTRAS_PACKAGES, DEV_TOOLS = load_pyproject_deps()

# Fallback if parsing failed
if not CORE_PACKAGES:
    CORE_PACKAGES = dict(VALIDATE_ENV_CORE_FALLBACK_SPECS)

if not EXTRAS_PACKAGES:
    EXTRAS_PACKAGES = dict(VALIDATE_ENV_EXTRAS_FALLBACK_SPECS)

if not DEV_TOOLS:
    DEV_TOOLS = {
        "ruff": ">=0.1.0",
        "mypy": ">=1.8.0",
        "pytest": ">=8.0.0",
        "ty": "",
    }


def check_python_version() -> None:
    """Verify Python version meets minimum requirements."""
    current = sys.version_info[:2]
    if current < REQUIRED_PYTHON_VERSION:
        msg = (
            f"Python {REQUIRED_PYTHON_VERSION[0]}.{REQUIRED_PYTHON_VERSION[1]}+ "
            f"required, but found {current[0]}.{current[1]}"
        )
        raise ValidationError(msg)
    logger.info("✓ Python %d.%d.%d", *sys.version_info[:3])


def _resolve_expected_conda_env(cli_expected_env: str | None = None) -> str | None:
    """Resolve optional strict conda-env expectation from CLI or environment."""
    if cli_expected_env:
        normalized_cli = cli_expected_env.strip()
        if normalized_cli:
            return normalized_cli

    env_expected = os.environ.get(EXPECTED_CONDA_ENV_ENVVAR, "").strip()
    if env_expected:
        return env_expected

    return None


def check_conda_env_with_expected(expected_env: str | None) -> None:
    """Verify we're in an active conda environment, with optional strict name matching."""
    if not shutil.which("conda"):
        logger.info("⊘ Conda not found (skipping env check)")
        return

    active_env = os.environ.get("CONDA_DEFAULT_ENV")
    if not active_env:
        logger.warning("⚠ Conda is installed but no environment is active")
        logger.warning("  Activate with: conda activate %s", expected_env or DEFAULT_CONDA_ENV)
        return

    if expected_env and active_env != expected_env:
        logger.warning("⚠ Active conda env '%s' != expected '%s'", active_env, expected_env)
        logger.warning("  Switch with: conda activate %s", expected_env)
        return

    logger.info("✓ Conda environment '%s' active", active_env)


def check_conda_env() -> None:
    """Verify we're in an active conda environment, using optional configured strict matching."""
    check_conda_env_with_expected(_resolve_expected_conda_env())


def check_package(name: str, version_spec: str) -> bool:
    """Check if a package is installed and meets version requirements."""
    try:
        installed_version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        logger.warning("✗ %s %s (NOT INSTALLED)", name, version_spec)
        return False
    else:
        if not _version_matches_specifier(
            package_name=name,
            installed_version=installed_version,
            version_spec=version_spec,
        ):
            logger.warning(
                "✗ %s %s (installed: %s, DOES NOT SATISFY SPEC)",
                name,
                version_spec,
                installed_version,
            )
            return False

        logger.info("✓ %s %s (installed: %s)", name, version_spec, installed_version)
        return True


def _version_matches_specifier(
    *,
    package_name: str,
    installed_version: str,
    version_spec: str,
) -> bool:
    """Return whether installed version satisfies specifier string."""
    if not version_spec:
        return True

    if SpecifierSet is None or Version is None:
        logger.warning(
            "⚠ Skipping version constraint check for %s (%s): packaging not installed",
            package_name,
            version_spec,
        )
        return True

    try:
        specifier = SpecifierSet(version_spec)
    except Exception:
        logger.warning(
            "⚠ Skipping malformed version constraint check for %s: %s",
            package_name,
            version_spec,
        )
        return True

    try:
        parsed_version = Version(installed_version)
    except InvalidVersion:
        logger.warning(
            "⚠ Skipping version constraint check for %s: invalid installed version %s",
            package_name,
            installed_version,
        )
        return True

    return parsed_version in specifier


def check_packages(packages: dict[str, str]) -> bool:
    """Check all packages in the given dict."""
    all_ok = True
    for name, version_spec in packages.items():
        if not check_package(name, version_spec):
            all_ok = False
    return all_ok


def check_pip_consistency() -> bool:
    """Run pip's dependency consistency check with pragmatic warning handling."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as e:
        logger.warning("⚠ Could not run pip consistency check: %s", e)
        return True

    if result.returncode == 0:
        logger.info("✓ pip dependency consistency check passed")
        return True

    combined = "\n".join(
        part.strip() for part in (result.stdout, result.stderr) if part and part.strip()
    )
    lines = [
        line
        for line in combined.splitlines()
        if line.strip() and not line.strip().startswith("WARNING:")
    ]
    if not lines:
        logger.warning("✗ pip check failed without diagnostic output")
        return False

    torch_platform_warning = all(
        "torch" in line.lower() and "not supported on this platform" in line.lower()
        for line in lines
    )
    if torch_platform_warning:
        logger.warning("⚠ pip check reported Torch platform metadata warning:")
        for line in lines:
            logger.warning("  - %s", line)
        logger.warning("  Continuing; this warning can be non-fatal on Apple Silicon wheels.")
        return True

    logger.warning("✗ pip check reported dependency issues:")
    for line in lines:
        logger.warning("  - %s", line)
    return False


def check_git_hooks() -> bool:
    """Check if commit/push git hooks are installed."""
    repo_root = Path(__file__).resolve().parents[2]
    hooks_dir = repo_root / ".git" / "hooks"
    required_hooks = ("pre-commit", "pre-push")
    missing_hooks = [name for name in required_hooks if not (hooks_dir / name).exists()]

    if not missing_hooks:
        logger.info("✓ Git commit/push hooks installed")
        return True

    logger.warning("⚠ Git hook(s) missing: %s", ", ".join(missing_hooks))
    logger.warning("  Install with either:")
    logger.warning("    python -m tools.install_precommit_hook")
    logger.warning("    pre-commit install")
    return False


def check_precommit_framework() -> bool:
    """Check if pre-commit framework is available."""
    precommit_path = shutil.which("pre-commit")
    if not precommit_path:
        logger.warning("⚠ pre-commit framework not installed (optional)")
        logger.warning("  Install with: pip install pre-commit")
        logger.warning("  Then run: pre-commit install")
        return False

    # Check if hooks are installed
    result = subprocess.run(
        [precommit_path, "run", "--all-files", "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0 or "would be modified" in result.stdout:
        logger.info("✓ pre-commit framework configured")
        return True

    logger.warning("⚠ pre-commit framework hooks not installed")
    logger.warning("  Run: pre-commit install")
    return False


def fix_issues() -> None:
    """Attempt to auto-fix common issues."""
    logger.info("Attempting to auto-fix issues...")

    # Install missing packages
    logger.info("Installing missing packages...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", ".[dev]"],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    # Install git hooks
    logger.info("Installing git pre-commit hook...")
    subprocess.run(
        [sys.executable, "-m", "tools.install_precommit_hook"],
        check=True,
    )

    # Install pre-commit framework if available
    precommit_path = shutil.which("pre-commit")
    if precommit_path:
        logger.info("Installing pre-commit framework hooks...")
        subprocess.run([precommit_path, "install"], check=True)


def main() -> int:
    """Run environment validation."""
    parser = argparse.ArgumentParser(description="Validate development environment")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to auto-fix issues (install packages, hooks)",
    )
    parser.add_argument(
        "--expected-conda-env",
        type=str,
        default=None,
        help=(
            "Require a specific active conda env name. If omitted, any active conda env "
            "is accepted; you can also set CHECK_MODELS_EXPECTED_CONDA_ENV."
        ),
    )
    args = parser.parse_args()
    expected_conda_env = _resolve_expected_conda_env(args.expected_conda_env)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    logger.info("=== Environment Validation ===\n")

    issues = []

    try:
        # Python version
        logger.info("Checking Python version...")
        check_python_version()

        # Conda environment
        logger.info("\nChecking conda environment...")
        check_conda_env_with_expected(expected_conda_env)

        # Core packages
        logger.info("\nChecking core packages...")
        if not check_packages(CORE_PACKAGES):
            issues.append("Core packages missing or outdated")

        # Optional extras (informational only - don't fail if missing)
        logger.info("\nChecking optional extras packages...")
        check_packages(EXTRAS_PACKAGES)

        # Dev tools
        logger.info("\nChecking development tools...")
        if not check_packages(DEV_TOOLS):
            issues.append("Development tools missing or outdated")

        # pip dependency consistency (warnings only for known torch platform metadata issue)
        logger.info("\nChecking pip dependency consistency...")
        if not check_pip_consistency():
            issues.append("pip dependency consistency check failed")

        # Git hooks
        logger.info("\nChecking git hooks...")
        if not check_git_hooks():
            issues.append("Git commit/push hooks not installed")

        # Pre-commit framework
        logger.info("\nChecking pre-commit framework...")
        check_precommit_framework()

    except ValidationError as e:
        logger.warning("\n✗ VALIDATION FAILED: %s", e)
        return 1

    if issues:
        logger.warning("\n⚠ Issues found:")
        for issue in issues:
            logger.warning("  - %s", issue)

        if args.fix:
            logger.info("\n🔧 Attempting to fix issues...")
            try:
                fix_issues()
            except (subprocess.CalledProcessError, OSError, RuntimeError):
                logger.exception("\n✗ Fix failed")
                return 1
            else:
                logger.info("\n✓ Fixes applied. Re-run validation to verify.")
                return 0

        logger.info("\n💡 Run with --fix to attempt automatic fixes")
        return 1

    logger.info("\n✓ All checks passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
