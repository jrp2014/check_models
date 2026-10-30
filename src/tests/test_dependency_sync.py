"""Test that runtime dependency block in README matches pyproject runtime deps.

This enforces local parity in addition to CI. The test focuses only on runtime deps
(not optional extras groups) and parses dependencies with packaging's PEP 508 parser.
"""

from __future__ import annotations

import ast
import datetime as dt
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tomllib
import typing
import zipfile
from pathlib import Path
from textwrap import dedent

import huggingface_hub
import huggingface_hub.utils
import pytest
import yaml
from packaging.requirements import Requirement
from packaging.version import Version

import check_models
from check_models_data import dependency_policy
from tools import (
    check_dependency_advisories,
    check_suppressions,
    filter_danger_report,
    hub_precheck,
    quarantine_broken_pip_metadata,
    relax_exact_mlx_pin,
    safe_io,
    update_readme_deps,
    validate_env,
)

if typing.TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from urllib.request import Request

_TEST_FILE = Path(__file__).resolve()
# tests/ parent, then package root (vlm)
PKG_ROOT = _TEST_FILE.parents[1]
REPO_ROOT = PKG_ROOT.parent


def _write_pip_dist_info(
    site_dir: Path,
    distribution: str,
    version: str,
    *,
    complete: bool,
) -> Path:
    metadata_dir = site_dir / f"{distribution}-{version}.dist-info"
    metadata_dir.mkdir()
    safe_io.write_text_no_follow(metadata_dir / "INSTALLER", "pip\n")
    if complete:
        safe_io.write_text_no_follow(
            metadata_dir / "METADATA",
            f"Name: {distribution}\nVersion: {version}\n",
        )
        safe_io.write_text_no_follow(metadata_dir / "RECORD", "")
    return metadata_dir


def _first_existing(paths: list[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    msg = f"None of the candidate paths exist: {paths}"
    raise FileNotFoundError(msg)


PYPROJECT = _first_existing(
    [PKG_ROOT / "pyproject.toml", REPO_ROOT / "pyproject.toml"],
)  # prefer in-package
README = _first_existing(
    [PKG_ROOT / "README.md", REPO_ROOT / "README.md"],
)  # prefer in-package
PACKAGED_QUALITY_CONFIG = PKG_ROOT / "check_models_data" / "quality_config.yaml"
LEGACY_ROOT_QUALITY_CONFIG = PKG_ROOT / "quality_config.yaml"
ROOT_SKYLOS_CONFIG = REPO_ROOT / ".skylos" / "config.yaml"
COPILOT_INSTRUCTIONS = REPO_ROOT / ".github" / "copilot-instructions.md"
AGENT_SKILLS_DIR = REPO_ROOT / ".agents" / "skills"
# Upstream skill names a local skill may cite as its source (skills/skills/ in
# Blaizzy/mlx-vlm); a cited name outside this set is a typo or a removal.
UPSTREAM_MLX_VLM_SKILLS = frozenset(
    {
        "add-new-model",
        "benchmarking",
        "cli-inference",
        "contributing",
        "convert-quantize",
        "hf-cache-models",
        "reproducible-github-issues",
        "server-inference",
    }
)
# transformers keeps its skills under .ai/skills/ (linked into .agents and .claude).
UPSTREAM_TRANSFORMERS_SKILLS = frozenset({"add-or-fix-type-checking"})
# Upstream targets the guidance cites by name only (transformers' `make claude`).
UPSTREAM_MAKE_TARGET_MENTIONS = frozenset({"claude"})
SKYLOS_DANGER_ADVISORY_SCRIPT = PKG_ROOT / "tools" / "run_skylos_danger_advisory.sh"
SKYLOS_VERIFY_SCRIPT = PKG_ROOT / "tools" / "run_skylos_verify.sh"

SKYLOS_ADVISORY_QUALITY_IGNORES = {
    "SKY-C401",
    "SKY-L004",
    "SKY-L017",
    "SKY-L026",
    "SKY-L028",
    "SKY-L029",
    "SKY-P403",
    "SKY-Q306",
    "SKY-Q501",
    "SKY-Q502",
    "SKY-Q701",
    "SKY-Q702",
    "SKY-Q802",
    "SKY-Q803",
    "SKY-R104",
}
SKYLOS_MONOLITH_QUALITY_LIMITS = {
    "complexity": 24,
    "nesting": 6,
    "max_lines": 450,
    "duplicate_strings": 40,
}

MANUAL_MARKERS = ("<!-- MANUAL_INSTALL_START -->", "<!-- MANUAL_INSTALL_END -->")

IMPORT_NAME_BY_REQUIREMENT = {
    "huggingface-hub": "huggingface_hub",
    "mlx-vlm": "mlx_vlm",
    "pillow": "PIL",
    "pyyaml": "yaml",
}


def _extract_manual_block(readme: str) -> str:
    start, end = MANUAL_MARKERS
    pattern = re.compile(rf"{re.escape(start)}(.*?){re.escape(end)}", re.DOTALL)
    m = pattern.search(readme)
    if not m:
        msg = "Manual install markers not found in README.md"
        raise RuntimeError(msg)
    return m.group(1)


def _make_targets(makefile: Path) -> set[str]:
    """Return the explicit rule names a Makefile defines."""
    text = safe_io.read_text_no_follow(makefile)
    return set(re.findall(r"^([\w-]+):", text, re.MULTILINE))


def _documented_shell_snippets(text: str) -> list[str]:
    """Return inline code spans and fenced code lines from Markdown text."""
    fences = re.findall(r"```[a-z]*\n(.*?)```", text, re.DOTALL)
    without_fences = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    inline = re.findall(r"`([^`\n]+)`", without_fences)
    return [*inline, *(line.strip() for fence in fences for line in fence.splitlines())]


def test_safe_io_read_text_no_follow_rejects_symlinked_file(tmp_path: Path) -> None:
    """Maintenance-tool text reads should not follow attacker-swapped symlinks."""
    target_path = tmp_path / "target.txt"
    target_path.write_text("safe text", encoding="utf-8")
    symlink_path = tmp_path / "link.txt"
    symlink_path.symlink_to(target_path)

    with pytest.raises(OSError, match="Refusing to follow symlink"):
        safe_io.read_text_no_follow(symlink_path)


def test_safe_io_read_text_no_follow_enforces_byte_cap(tmp_path: Path) -> None:
    """Maintenance-tool text reads should reject unexpectedly large files."""
    text_path = tmp_path / "large.txt"
    text_path.write_text("abcdef", encoding="utf-8")

    with pytest.raises(OSError, match=r"exceeds 3 bytes"):
        safe_io.read_text_no_follow(text_path, max_bytes=3)


def test_readme_runtime_block_matches_pyproject() -> None:
    """Ensure the runtime dependencies in README match pyproject.toml."""
    rd_text = README.read_text(encoding="utf-8")
    runtime_deps = update_readme_deps.parse_pyproject(PYPROJECT)

    manual_block = _extract_manual_block(rd_text)
    # Pull quoted specs from pip install line
    quoted = re.findall(r'"([^"]+)"', manual_block)
    if not quoted:
        msg = "No quoted packages found in manual install block"
        raise RuntimeError(msg)

    seen: dict[str, str] = {}
    for q in quoted:
        key, spec = update_readme_deps._parse_requirement(q)
        seen[key] = spec

    # All runtime deps must exist
    missing = [k for k in runtime_deps if k not in seen]
    if missing:
        msg = f"Runtime deps missing from README: {missing}"
        raise RuntimeError(msg)

    # No extras should leak (heuristic: check optional groups defined later if needed)
    forbidden = {
        "psutil",
        "tokenizers",
        "torch",
        "torchvision",
        "torchaudio",
    }
    leaked = sorted(forbidden & set(seen))
    if leaked:
        msg = f"Optional deps leaked into runtime block: {leaked}"
        raise RuntimeError(msg)


def test_update_readme_deps_fallback_parser_without_packaging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dependency sync should run before project dependencies are installed."""
    monkeypatch.setattr(update_readme_deps, "_Requirement", None)

    assert update_readme_deps._parse_requirement("Pillow[xmp]>=10.3.0") == (
        "Pillow[xmp]",
        ">=10.3.0",
    )
    assert update_readme_deps._parse_requirement(
        "huggingface-hub[typing, torch]>=1.10.1",
    ) == ("huggingface-hub[torch,typing]", ">=1.10.1")

    groups = update_readme_deps.extract_optional_groups(
        """
        [project]
        dependencies = []

        [project.optional-dependencies]
        extras = ["tokenizers>=0.15.0", "Pillow[xmp]>=10.3.0"]
        """,
    )

    assert groups == {"extras": ["tokenizers", "Pillow"]}


def test_dependency_policy_module_tracks_pyproject_stack_floors() -> None:
    """Shared dependency policy should stay aligned with declared packaging metadata."""
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    runtime_deps = pyproject["project"]["dependencies"]
    extras_deps = pyproject["project"]["optional-dependencies"]["extras"]

    assert f"mlx>={dependency_policy.PROJECT_RUNTIME_STACK_MINIMUMS['mlx']}" in runtime_deps
    assert f"mlx-vlm>={dependency_policy.PROJECT_RUNTIME_STACK_MINIMUMS['mlx-vlm']}" in runtime_deps
    assert f"transformers{dependency_policy.PROJECT_TRANSFORMERS_VERSION_SPEC}" in runtime_deps
    assert (
        f"huggingface-hub[typing]>={dependency_policy.PROJECT_RUNTIME_STACK_MINIMUMS['huggingface-hub']}"
        in runtime_deps
    )
    # mlx-lm is neither imported nor required by mlx-vlm any more: nowhere.
    assert not any(dep.startswith("mlx-lm") for dep in runtime_deps + extras_deps)


def _normalized_requirement_names(requirements: Iterable[str]) -> set[str]:
    return {
        re.split(r"[\[<>=!~; ]", requirement, maxsplit=1)[0].strip().lower().replace("_", "-")
        for requirement in requirements
    }


def test_validate_env_fallback_matches_declared_runtime_dependencies() -> None:
    """The no-pyproject fallback must check exactly the declared hard runtime set.

    It exists for the case where pyproject.toml cannot be loaded, so a stale
    literal there would flag removed packages as missing while overlooking
    genuinely missing ones (as happened when wcwidth was retired).
    """
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    declared = _normalized_requirement_names(pyproject["project"]["dependencies"])
    fallback = _normalized_requirement_names(dependency_policy.VALIDATE_ENV_CORE_FALLBACK_SPECS)
    assert fallback == declared


def test_dependency_policy_tracks_current_upstream_transformers_floor() -> None:
    """The project floor tracks the released mlx-vlm stack (0.6.16, first py.typed)."""
    assert dependency_policy.PROJECT_RUNTIME_STACK_MINIMUMS["transformers"] == "5.14.0"
    assert dependency_policy.UPSTREAM_MLX_VLM_MINIMUMS["transformers"] == "5.14.0"


def test_dependency_policy_does_not_cap_transformers() -> None:
    """Transformers should retain its floor without an upper version bound."""
    assert dependency_policy.PROJECT_TRANSFORMERS_VERSION_SPEC == ">=5.14.0"


def test_pillow_floor_uses_security_fixed_release() -> None:
    """Pillow's declared floor should exclude vulnerabilities fixed in 12.3.0."""
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))

    assert dependency_policy.PROJECT_PILLOW_MINIMUM_VERSION == "12.3.0"
    assert (
        f"Pillow[xmp]>={dependency_policy.PROJECT_PILLOW_MINIMUM_VERSION}"
        in pyproject["project"]["dependencies"]
    )
    assert (
        dependency_policy.VALIDATE_ENV_CORE_FALLBACK_SPECS["Pillow"]
        == f">={dependency_policy.PROJECT_PILLOW_MINIMUM_VERSION}"
    )


def test_pyrefly_quality_check_passes_explicit_targets() -> None:
    """Pyrefly must run in single-file mode, not project discovery.

    Explicit targets enumerated via ``git ls-files`` bypass Pyrefly's
    filesystem discovery entirely, so the gate's file set cannot be eaten by
    parent-repo ignore files or future discovery heuristics. This is the
    second layer of the worktree defense; the generated config additionally
    neutralizes the known discovery pitfalls (see the companion generated-
    config test).
    """
    common_quality = (PKG_ROOT / "tools" / "common_quality.sh").read_text(encoding="utf-8")

    assert "quality_pyrefly_default_targets" in common_quality
    assert (
        'git -C "$(quality_src_root)" ls-files --cached --others --exclude-standard'
        in common_quality
    )
    assert '"${targets[@]}" 2>&1 | tee "$output_path"' in common_quality


def test_mypy_follows_imports_for_their_types() -> None:
    """Imported packages must keep their types; "skip" silently turned them into Any.

    Only the editable-checkout packages may be missing, and no override may
    reinstate skipping or hide errors in the files mypy is asked to check.
    """
    mypy_config = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["tool"]["mypy"]

    assert mypy_config["follow_imports"] == "silent"
    assert mypy_config["ignore_missing_imports"] is False
    for override in mypy_config.get("overrides", []):
        assert override.get("follow_imports", "silent") == "silent"
        assert "ignore_errors" not in override
        if override.get("ignore_missing_imports"):
            assert set(override["module"]) <= {"mlx", "mlx.*", "mlx_vlm", "mlx_vlm.*"}


def test_root_makefile_exposes_documented_maintenance_targets() -> None:
    """Contributor docs should be able to use maintenance targets from repo root."""
    root_makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")

    for target in ("check-outdated", "audit", "clean-mlx", "clean-mlx-dry-run"):
        assert f".PHONY: {target}" in root_makefile
        assert f"{target}:" in root_makefile


def test_dependency_docs_do_not_reference_removed_lockfile_workflows() -> None:
    """Live docs should not point contributors at removed requirements/lock workflows."""
    live_docs = {
        "implementation": (REPO_ROOT / "docs" / "IMPLEMENTATION_GUIDE.md").read_text(
            encoding="utf-8"
        ),
        "contributing": (REPO_ROOT / "docs" / "CONTRIBUTING.md").read_text(encoding="utf-8"),
        "src_makefile": (PKG_ROOT / "Makefile").read_text(encoding="utf-8"),
    }
    removed_phrases = (
        "make sync-deps",
        "make upgrade-deps",
        "src/requirements.txt",
        "requirements-dev.txt",
        "make -C vlm",
    )

    for doc_name, text in live_docs.items():
        for phrase in removed_phrases:
            assert phrase not in text, (doc_name, phrase)


def test_package_skylos_scan_excludes_generated_artifacts() -> None:
    """Package-local Skylos config should scan maintained source, not generated outputs."""
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    skylos_config = pyproject["tool"]["skylos"]
    skylos_gate = skylos_config["gate"]

    assert "addopts" not in skylos_config

    excludes = set(skylos_config["exclude"])
    assert {
        "output",
        "package-lock.json",
        "node_modules",
        "build",
        "dist",
        "*.egg-info",
        "check_models.suppression-audit*.py",
    } <= excludes
    assert set(skylos_config["ignore"]) >= SKYLOS_ADVISORY_QUALITY_IGNORES
    for key, value in SKYLOS_MONOLITH_QUALITY_LIMITS.items():
        assert skylos_config[key] == value
    assert skylos_gate == {
        "fail_on_critical": True,
        "max_critical": 0,
        "max_high": 0,
        "max_security": 0,
        "max_quality": 0,
        "strict": False,
    }


def test_root_skylos_config_mirrors_package_quality_policy() -> None:
    """Repo-root scans should use the same advisory quality calibration as package scans."""
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    package_config = pyproject["tool"]["skylos"]
    root_config = yaml.safe_load(ROOT_SKYLOS_CONFIG.read_text(encoding="utf-8"))

    assert isinstance(root_config, dict)
    assert {
        "output",
        "src/output",
        "package-lock.json",
        "node_modules",
        "build",
        "dist",
        "*.egg-info",
        "src/check_models.suppression-audit*.py",
        # Third-party checkouts and agent worktrees must never gate this repo.
        ".worktrees",
        ".claude",
    } <= set(root_config["exclude"])
    assert set(root_config["ignore"]) == set(package_config["ignore"])
    for key in SKYLOS_MONOLITH_QUALITY_LIMITS:
        assert root_config[key] == package_config[key]
    assert root_config["gate"] == package_config["gate"]


def test_ruff_uses_current_floor_and_all_stable_rules() -> None:
    """Every stable rule shipped by the supported Ruff release should be selected."""
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    ruff_config = pyproject["tool"]["ruff"]
    dev_deps = pyproject["project"]["optional-dependencies"]["dev"]

    assert "ruff>=0.16.0" in dev_deps
    assert ruff_config["required-version"] == ">=0.16.0"
    assert ruff_config["lint"]["select"] == ["ALL"]

    setup_script = (PKG_ROOT / "tools" / "setup_conda_env.sh").read_text(encoding="utf-8")
    assert '"ruff>=0.16.0"' in setup_script


def test_conda_setup_verifier_imports_declared_non_dev_dependencies() -> None:
    """The fresh setup smoke check should only import packages it just installed."""
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    setup_script = (PKG_ROOT / "tools" / "setup_conda_env.sh").read_text(encoding="utf-8")
    verifier_match = re.search(r'python -c "\n(.*?)\n"', setup_script, re.DOTALL)
    assert verifier_match is not None

    declared_requirements = [
        *pyproject["project"]["dependencies"],
        *pyproject["project"]["optional-dependencies"]["extras"],
        *pyproject["project"]["optional-dependencies"]["torch"],
    ]
    declared_imports = {
        IMPORT_NAME_BY_REQUIREMENT.get(requirement.name.lower(), requirement.name.replace("-", "_"))
        for requirement_text in declared_requirements
        for requirement in [Requirement(requirement_text)]
    }
    imported_modules = {
        alias.name.split(".", maxsplit=1)[0]
        for node in ast.walk(ast.parse(verifier_match.group(1)))
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert imported_modules <= declared_imports


def test_conda_setup_verifies_mlx_backend_pair() -> None:
    """Fresh setup should fail fast on mismatched or missing MLX Metal artifacts."""
    setup_script = (PKG_ROOT / "tools" / "setup_conda_env.sh").read_text(encoding="utf-8")

    assert "metadata.version('mlx-metal')" in setup_script
    assert "mlx/mlx-metal version mismatch" in setup_script
    assert "mlx.metallib" in setup_script
    assert "MLX Metal library missing" in setup_script
    assert "MLX editable install detected" in setup_script


def test_conda_setup_uses_current_huggingface_cli_installation() -> None:
    """Avoid installing removed huggingface-hub extras on any bootstrap path."""
    setup_script = (PKG_ROOT / "tools" / "setup_conda_env.sh").read_text(encoding="utf-8")

    assert '"huggingface_hub[cli]"' not in setup_script
    assert "command -v hf" in setup_script

    makefile = (PKG_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "huggingface_hub[cli]" not in makefile


@pytest.mark.subprocess
def test_common_quality_finds_conda_executable_without_sourcing_conda_sh(tmp_path: Path) -> None:
    """Quality helpers should resolve conda directly instead of sourcing conda.sh."""
    fake_conda = tmp_path / "miniconda3" / "bin" / "conda"
    fake_conda.parent.mkdir(parents=True)
    fake_conda.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_conda.chmod(0o755)

    output_path = tmp_path / "conda-bin.txt"
    run_script = tmp_path / "check_common_quality_conda_bin.sh"
    run_script.write_text(
        dedent(
            f"""\
            #!/usr/bin/env bash
            set -euo pipefail
            export HOME="{tmp_path}"
            export PATH=/usr/bin:/bin
            source "{PKG_ROOT / "tools" / "common_quality.sh"}"
            quality_find_conda_bin > "{output_path}"
            """
        ),
        encoding="utf-8",
    )
    run_script.chmod(0o755)

    result = subprocess.run(  # noqa: S603 - fixed /bin/bash runs a test-created script
        ["/bin/bash", str(run_script)],
        capture_output=True,
        text=True,
        check=False,
        cwd=PKG_ROOT,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert output_path.read_text(encoding="utf-8").strip() == str(fake_conda)


@pytest.mark.subprocess
def test_common_quality_rejects_local_python_fallback_without_override(tmp_path: Path) -> None:
    """Local quality runs should fail instead of silently using PATH python."""
    run_script = tmp_path / "reject_python_fallback.sh"
    run_script.write_text(
        dedent(
            f"""\
            #!/usr/bin/env bash
            set -euo pipefail
            unset CONDA_PREFIX
            unset CONDA_DEFAULT_ENV
            unset CI
            export HOME="{tmp_path}"
            export PATH=/usr/bin:/bin
            source "{PKG_ROOT / "tools" / "common_quality.sh"}"
            if quality_setup_python > "{tmp_path / "stdout.txt"}" 2> "{tmp_path / "stderr.txt"}"; then
                exit 44
            fi
            grep -q "Unable to resolve required conda environment" "{tmp_path / "stderr.txt"}"
            """
        ),
        encoding="utf-8",
    )
    run_script.chmod(0o755)

    result = subprocess.run(  # noqa: S603 - fixed /bin/bash runs a test-created script
        ["/bin/bash", str(run_script)],
        capture_output=True,
        text=True,
        check=False,
        cwd=PKG_ROOT,
    )

    assert result.returncode == 0, result.stderr or result.stdout


@pytest.mark.subprocess
def test_common_quality_python_tools_do_not_fall_back_to_path_by_default(tmp_path: Path) -> None:
    """Python tools should resolve from the chosen interpreter's bin directory."""
    env_bin = tmp_path / "env" / "bin"
    path_bin = tmp_path / "path-bin"
    env_bin.mkdir(parents=True)
    path_bin.mkdir()
    fake_python = env_bin / "python"
    fake_python.symlink_to(Path(sys.executable).resolve())
    fake_path_tool = path_bin / "ty"
    fake_path_tool.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_path_tool.chmod(0o755)

    run_script = tmp_path / "reject_path_tool_fallback.sh"
    run_script.write_text(
        dedent(
            f"""\
            #!/usr/bin/env bash
            set -euo pipefail
            cd "{PKG_ROOT}"
            export PATH="{path_bin}:/usr/bin:/bin"
            source tools/common_quality.sh
            QUALITY_PYTHON="{fake_python}"
            QUALITY_PYTHON_SOURCE="conda-env:mlx-vlm"
            export QUALITY_PYTHON QUALITY_PYTHON_SOURCE
            if quality_find_python_tool ty > "{tmp_path / "tool.txt"}"; then
                exit 44
            fi
            """
        ),
        encoding="utf-8",
    )
    run_script.chmod(0o755)

    result = subprocess.run(  # noqa: S603 - fixed /bin/bash runs a test-created script
        ["/bin/bash", str(run_script)],
        capture_output=True,
        text=True,
        check=False,
        cwd=PKG_ROOT,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_conda_setup_uses_conda_executable_without_sourcing_conda_sh() -> None:
    """Setup should use conda directly and avoid dynamic conda.sh sourcing."""
    setup_script = (PKG_ROOT / "tools" / "setup_conda_env.sh").read_text(encoding="utf-8")

    assert "source_conda_sh" not in setup_script
    assert "conda info --base" not in setup_script
    assert 'conda activate "$ENV_NAME"' not in setup_script
    assert "conda_cmd()" in setup_script
    assert "activate_environment_path()" in setup_script


def test_clean_builds_uses_static_help_and_guarded_direct_child_removal() -> None:
    """Cleanup tooling should avoid self-read help and unguarded direct-child rm calls."""
    clean_script = (PKG_ROOT / "tools" / "clean_builds.sh").read_text(encoding="utf-8")

    assert 'head -n 17 "$0"' not in clean_script
    assert "show_help()" in clean_script
    assert "remove_direct_child_dir()" in clean_script
    assert 'rm -rf "${dir:?}/$pattern"' not in clean_script
    assert "sudo rm -rf" not in clean_script


def test_packaged_quality_config_is_the_only_default_source() -> None:
    """The packaged config should be the sole checked-in default copy."""
    assert PACKAGED_QUALITY_CONFIG.exists()
    assert not LEGACY_ROOT_QUALITY_CONFIG.exists()


@pytest.mark.subprocess
def test_built_wheel_includes_packaged_quality_config(tmp_path: Path) -> None:
    """Built wheels should ship the packaged default quality config."""
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    # Build from a copy: setuptools writes build/ and *.egg-info next to the
    # sources. Those are not throwaway caches: a stale egg-info changes what
    # importlib.metadata reports for the package, and build/ holds a second
    # copy of the sources that the static checkers would scan.
    source_copy = tmp_path / "src"
    shutil.copytree(
        PKG_ROOT,
        source_copy,
        ignore=shutil.ignore_patterns(
            ".*",
            "__pycache__",
            "build",
            "dist",
            "*.egg-info",
            "node_modules",
            "output",
            "tests",
        ),
    )

    # Standard PEP 517 build: pip provisions the declared [build-system]
    # requirements in an isolated environment, so this passes or fails on the
    # checked-in packaging metadata alone, never on whatever happens to be
    # installed around it (a transitive setuptools once masked exactly that).
    result = subprocess.run(  # noqa: S603 - fixed interpreter builds the checked-in package
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "--wheel-dir",
            str(dist_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=source_copy,
    )

    assert result.returncode == 0, result.stderr or result.stdout

    wheel_paths = sorted(dist_dir.glob("check_models-*.whl"))
    assert len(wheel_paths) == 1

    with zipfile.ZipFile(wheel_paths[0]) as archive:
        assert "check_models_data/quality_config.yaml" in archive.namelist()


def test_markdownlint_cli2_is_repo_local_uncapped_and_updateable() -> None:
    """Keep markdownlint-cli2 aligned between npm metadata and update tooling.

    Policy: the spec is a caret range (never an exact pin), and update.sh moves
    to the latest release by default. The lockfile is deliberately untracked,
    so it is only cross-checked when a local `npm install` has produced one.
    """
    package_json = json.loads((PKG_ROOT / "package.json").read_text(encoding="utf-8"))

    markdownlint_spec = package_json["devDependencies"]["markdownlint-cli2"]
    assert markdownlint_spec.startswith("^"), "markdownlint-cli2 must stay uncapped (caret range)"
    # No repo-scoped npm scripts: the canonical markdownlint invocation lives in
    # tools/run_quality_checks.sh; a second spelling here would drift.
    assert "scripts" not in package_json

    lock_path = PKG_ROOT / "package-lock.json"
    if lock_path.exists():
        package_lock = json.loads(lock_path.read_text(encoding="utf-8"))
        assert (
            markdownlint_spec
            == package_lock["packages"][""]["devDependencies"]["markdownlint-cli2"]
        )
        locked_version = package_lock["packages"]["node_modules/markdownlint-cli2"]["version"]
        assert locked_version, "lockfile must resolve markdownlint-cli2"
        # Any security override for a transitive package must be what the lockfile
        # resolved (an override that no longer applies is stale and should go).
        for name, pinned in (package_json.get("overrides") or {}).items():
            assert package_lock["packages"][f"node_modules/{name}"]["version"] == pinned, name

    update_script = (PKG_ROOT / "tools" / "update.sh").read_text(encoding="utf-8")
    assert 'npm install --ignore-scripts --no-audit --no-fund --prefix "$PROJECT_ROOT"' in (
        update_script
    )
    assert (
        "npm install --ignore-scripts --no-audit --no-fund "
        '--prefix "$PROJECT_ROOT" --save-dev markdownlint-cli2@latest'
    ) in update_script
    assert "--save-exact" not in update_script
    # Latest is the default; the lockfile install is only the offline opt-out.
    assert '"${UPDATE_NODE_TOOLING:-1}" == "1"' in update_script
    assert update_script.index("markdownlint-cli2@latest") < update_script.index(
        "package-lock.json..."
    )


class _NpmOverrideRecord(typing.NamedTuple):
    """Why one npm override exists, and the upstream condition that retires it."""

    parent: str  # the package whose requirement necessitated the override
    parent_requirement: str  # that requirement, exactly as the lockfile records it
    fixed_version: str  # the first release fixing the advisory
    advisory: str


# Every entry in package.json's "overrides" is a temporary displacement of a
# requirement its parent package cannot yet resolve safely, so each must be
# recorded here with the exact parent requirement that necessitated it. The
# test below retires them: it fails when that requirement changes (drop the
# override if upstream now resolves a fixed release, otherwise re-review),
# when an override appears without a record, or when a record outlives its
# override.
_TRACKED_NPM_OVERRIDES: dict[str, _NpmOverrideRecord] = {
    "katex": _NpmOverrideRecord(
        parent="micromark-extension-math",
        parent_requirement="^0.16.0",
        fixed_version="0.18.2",
        advisory="GHSA-238p-pmpm-9mq7",
    ),
    "re2js": _NpmOverrideRecord(
        parent="markdownlint",
        parent_requirement="2.8.6",
        fixed_version="2.8.7",
        advisory="GHSA-gg6h-894w-4m9c",
    ),
}


def test_every_npm_override_is_tracked_and_retired_when_unneeded() -> None:
    """Each override names the upstream pin that necessitated it, and dies with it."""
    package = json.loads(safe_io.read_text_no_follow(PKG_ROOT / "package.json"))
    overrides = package.get("overrides") or {}
    assert set(overrides) == set(_TRACKED_NPM_OVERRIDES), (
        "every npm override needs a retirement record in _TRACKED_NPM_OVERRIDES, "
        "and every record a live override"
    )
    lock_path = PKG_ROOT / "package-lock.json"
    if not lock_path.exists():
        pytest.skip("npm install is needed to inspect resolved development dependencies")
    packages = json.loads(safe_io.read_text_no_follow(lock_path))["packages"]
    for name, record in _TRACKED_NPM_OVERRIDES.items():
        resolved = packages.get(f"node_modules/{name}")
        if resolved is not None:
            # The advisory must stay fixed whatever the override says.
            assert Version(resolved["version"]) >= Version(record.fixed_version), record.advisory
        upstream = packages.get(f"node_modules/{record.parent}", {}).get("dependencies", {})
        # Deliberately scoped to the exact upstream requirement that
        # necessitated the workaround, not a second implementation of npm's
        # range semantics.
        assert upstream.get(name) == record.parent_requirement, (
            f"{record.parent}'s {name} requirement changed: remove the temporary "
            f"{name} override if upstream now resolves a release fixing "
            f"{record.advisory}; otherwise re-review it"
        )


def test_generated_markdown_lint_guards_use_named_rule_sets() -> None:
    """Report-specific markdownlint guards should stay centralized."""
    source = (PKG_ROOT / "check_models.py").read_text(encoding="utf-8")

    # Behavioural checks on the live constants; exact source spelling is not
    # asserted so harmless refactors (annotation style, quoting) stay legal.
    assert check_models.MARKDOWNLINT_GALLERY_SUMMARY_RULES == "MD034 MD037 MD049"
    assert check_models.MARKDOWNLINT_TABLE_PIPE_RULES == "MD060"
    assert "MARKDOWNLINT_MAIN_TABLE_RULES" not in source
    assert "MARKDOWNLINT_DETAILS_RULES" not in source
    assert "<!-- markdownlint-disable MD033 MD034 MD037 MD049 -->" not in source
    assert "<!-- markdownlint-enable MD033 MD034 MD037 MD049 -->" not in source
    assert "<!-- markdownlint-disable MD034 -->" not in source
    assert "<!-- markdownlint-enable MD034 -->" not in source


def test_documented_make_commands_name_existing_targets() -> None:
    """Every `make` command in contributor and agent guidance must be a real target.

    Plain ``make X`` may be run from the repo root or ``src/``; ``make -C src X``
    must exist in ``src/Makefile``.
    """
    root_targets = _make_targets(REPO_ROOT / "Makefile")
    src_targets = _make_targets(PKG_ROOT / "Makefile")
    docs = [
        COPILOT_INSTRUCTIONS,
        REPO_ROOT / "AGENTS.md",
        REPO_ROOT / "README.md",
        README,
        REPO_ROOT / "docs" / "CONTRIBUTING.md",
        REPO_ROOT / "docs" / "IMPLEMENTATION_GUIDE.md",
        *sorted((REPO_ROOT / ".agents" / "workflows").glob("*.md")),
        *sorted(AGENT_SKILLS_DIR.glob("*/SKILL.md")),
    ]
    commands = [
        (doc.relative_to(REPO_ROOT).as_posix(), match.group(1), match.group(2))
        for doc in docs
        for snippet in _documented_shell_snippets(safe_io.read_text_no_follow(doc))
        for match in re.finditer(r"^make( -C src)? ([a-z][\w-]*)", snippet, re.MULTILINE)
    ]
    assert commands
    missing = [
        (doc, target)
        for doc, in_src, target in commands
        if target not in UPSTREAM_MAKE_TARGET_MENTIONS
        and target not in (src_targets if in_src else root_targets | src_targets)
    ]
    assert missing == []


def test_output_scratch_paths_are_gitignored() -> None:
    """`test_*` scratch output and the append-only history stay untracked."""
    gitignore_lines = {
        line.strip() for line in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    }

    assert {
        "src/output/test_*",
        "src/output/reports/test_*",
        "src/output/issues/test_*",
        "src/output/results.history.jsonl",
    }.issubset(gitignore_lines)


def test_readme_documents_only_the_simplified_report_contract() -> None:
    """Public output documentation should expose only retained report artifacts."""
    readme = README.read_text(encoding="utf-8")

    for artifact in (
        "reports/diagnostics.md",
        "reports/model_gallery.md",
        "reports/results.html",
        "results.jsonl",
        "index.md",
        "check_models.log",
        "environment.log",
        "results.history.jsonl",
    ):
        assert f"`{artifact}`" in readme

    for retired in (
        "results.md",
        "results.tsv",
        "review.md",
        "model_selection.md",
        "model_capabilities.md",
        "model_capabilities.json",
        "repro_bundles",
        "--output-markdown",
        "--output-review",
        "--output-model-selection",
        "--output-model-capabilities",
        "--output-model-capabilities-json",
        "--output-tsv",
    ):
        assert retired not in readme

    # Each documented status field lists every value the code can emit.
    for alias in (
        check_models.ExecutionStatus,
        check_models.ModelUsability,
        check_models.MaintainerStatus,
    ):
        for value in typing.get_args(alias.__value__):
            assert f"`{value}`" in readme, (alias.__name__, value)


def test_production_assessment_policy_has_no_fixture_specific_exceptions() -> None:
    """Policy feeders must not embed benchmark model, image, or location literals."""
    source = (PKG_ROOT / "check_models.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    string_literals = {
        node.value.casefold()
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert not any(re.fullmatch(r"org/[a-z0-9_-]+", value) for value in string_literals)
    assert string_literals.isdisjoint(
        {
            "granite",
            "pink couch",
            "tabby",
            "brick storefront",
            "harbor sunset",
            "remote controls",
            "deben estuary",
            "woodbridge",
            "welwyn garden city",
        }
    )

    config = yaml.safe_load(PACKAGED_QUALITY_CONFIG.read_text(encoding="utf-8"))
    assert set(config) == {"thresholds"}
    assert "min_keywords_for_duplication_check" not in config["thresholds"]


def test_python_sources_write_variation_selectors_as_escapes() -> None:
    r"""Raw U+FE0F must appear only via the \ufe0f escape in Python sources.

    CodeQL's tsg-python extractor mis-slices emoji variation selectors while
    evaluating string literals in any file containing PEP 695 syntax; when the
    same literal carries a %-format directive the extractor's error reporter
    crashes and the whole file is silently dropped from security analysis.
    Writing the selector as an escape keeps the rendered glyph identical while
    the source stays ASCII at that position.
    """
    # node_modules: third-party npm packages (katex) ship their own Python
    # sources, which are not this repository's to police — and iCloud can
    # leave evicted conflict copies there that time out on read.
    excluded_parts = {".worktrees", "build", "output", "check_models.egg-info", "node_modules"}
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in sorted(REPO_ROOT.glob("src/**/*.py"))
        if not excluded_parts.intersection(path.parts)
        and "\ufe0f" in path.read_text(encoding="utf-8")
    ]
    assert offenders == [], offenders


def test_quality_script_runs_skylos_quality_gate() -> None:
    """Local quality checks should include the calibrated Skylos quality gate."""
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    dev_deps = pyproject["project"]["optional-dependencies"]["dev"]
    quality_script = (PKG_ROOT / "tools" / "run_quality_checks.sh").read_text(encoding="utf-8")
    setup_script = (PKG_ROOT / "tools" / "setup_conda_env.sh").read_text(encoding="utf-8")

    skylos_specs = [dep for dep in dev_deps if dep.startswith("skylos")]
    assert len(skylos_specs) == 1
    # The setup script must install the same spec pyproject declares,
    # whatever that spec is (deliberately not pinned).
    assert f'"{skylos_specs[0]}"' in setup_script
    assert (
        'quality_require_python_tool skylos "Install dev dependencies with: pip install -e .[dev]"'
        in quality_script
    )
    assert 'echo "=== Skylos Quality Gate ==="' in quality_script
    # The ungated `-a` audit pass is gone: it rescanned the same tree for a
    # grade card; its one extra check (ai-defects) rides on the gated pass.
    assert 'echo "=== Skylos Audit Gate ==="' not in quality_script
    assert "SKYLOS_JOBS" not in quality_script
    # Every Skylos call goes through the wrapper, which raises the 4.36+ grep
    # verification cap (default 30 s) so the slower CI runner does not turn a
    # clean tree into SKY-ANALYSIS-INCOMPLETE (exit 2).
    common_quality = (PKG_ROOT / "tools" / "common_quality.sh").read_text(encoding="utf-8")
    assert 'SKYLOS_GREP_BUDGET="${SKYLOS_GREP_BUDGET:-300}"' in common_quality
    # The danger scan is deliberately blocking in full mode, and only via the
    # wrapper script (never a bare `skylos --danger` that would bypass the
    # worktree post-filter and non-interactive guards).
    assert 'bash "$SCRIPT_DIR/run_skylos_danger_advisory.sh" --full --gate' in quality_script
    # Fast mode keeps an advisory (non-blocking) danger scan since the ungated
    # `-a` audit pass, which used to provide it, is gone.
    assert 'bash "$SCRIPT_DIR/run_skylos_danger_advisory.sh" --full\n' in quality_script
    assert "skylos . --danger" not in quality_script
    # One markdownlint step (with the worktree exclusion) serves both modes:
    # one definition plus one call in fast mode and one in full mode.
    assert quality_script.count('"!**/.worktrees/**"') == 1
    assert quality_script.count("run_markdownlint_step") == 3
    # Skylos runs before pytest, never concurrently: its grep verification
    # aborts when files appear or vanish under it.
    assert quality_script.index('echo "=== Skylos Quality Gate ==="') < quality_script.index(
        'echo "=== Pytest ==="'
    )
    assert re.search(
        r"TERM=dumb NO_COLOR=1 CLICOLOR=0 FORCE_COLOR=0 PY_COLORS=0\s+\\?\s*"
        r"quality_run_skylos \. --quality --secrets --ai-defects --gate --no-upload "
        r"--format concise",
        quality_script,
    )
    # Dependency advisories are gated by their own step (Skylos's ignore list
    # cannot accept one advisory), which must run before pytest too.
    assert re.search(
        r"quality_run_skylos \. --sca --json -o \"\$sca_report_path\" --no-upload", quality_script
    )
    assert '"$QUALITY_PYTHON" -m tools.check_dependency_advisories "$sca_report_path"' in (
        quality_script
    )
    assert quality_script.index('echo "=== Skylos Dependency Scan ==="') < quality_script.index(
        'echo "=== Pytest ==="'
    )
    assert not re.search(r"quality_run_skylos \. -a\b", quality_script)


def test_skylos_danger_advisory_script_is_separate_and_agent_friendly() -> None:
    """Advisory Skylos danger scans should stay separate from the blocking quality gate."""
    script = SKYLOS_DANGER_ADVISORY_SCRIPT.read_text(encoding="utf-8")

    assert "--danger --json" in script
    assert "skylos cicd annotate" in script
    assert "--severity medium" in script
    assert "cicd gate" in script
    assert "--advisory" in script
    assert "--llm" in script
    assert ".skylos/skylos-danger-advisory.llm.txt" in script


@pytest.mark.subprocess
def test_skylos_danger_scan_excludes_third_party_worktrees(tmp_path: Path) -> None:
    """The real wrapper must pass its worktree exclusion to the scanner process."""
    repo_root = tmp_path / "repo"
    tools_dir = repo_root / "src" / "tools"
    tools_dir.mkdir(parents=True)
    script = tools_dir / SKYLOS_DANGER_ADVISORY_SCRIPT.name
    script.write_bytes(SKYLOS_DANGER_ADVISORY_SCRIPT.read_bytes())
    (tools_dir / "__init__.py").write_text("", encoding="utf-8")
    for helper_name in ("filter_danger_report.py", "safe_io.py"):
        helper = PKG_ROOT / "tools" / helper_name
        (tools_dir / helper_name).write_bytes(helper.read_bytes())
    call_log = tmp_path / "skylos-calls.log"
    (tools_dir / "common_quality.sh").write_text(
        dedent(
            """\
            quality_repo_root() { printf '%s\\n' "$TEST_REPO_ROOT"; }
            quality_setup_python() {
                QUALITY_PYTHON="$TEST_PYTHON"
                export QUALITY_PYTHON
            }
            quality_require_python_tool() { return 0; }
            quality_run_python_tool() {
                printf '%s\\n' "$*" >> "$TEST_CALL_LOG"
                shift
                while [ "$#" -gt 0 ]; do
                    if [ "$1" = "-o" ]; then
                        printf '{"danger":[]}\\n' > "$2"
                        break
                    fi
                    shift
                done
            }
            quality_run_skylos() { quality_run_python_tool skylos "$@"; }
            """
        ),
        encoding="utf-8",
    )

    result = subprocess.run(  # noqa: S603 - fixed /bin/bash runs a test-copied script
        ["/bin/bash", str(script), "--full", "--gate"],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
        env={
            **os.environ,
            "TEST_REPO_ROOT": str(repo_root),
            "TEST_PYTHON": sys.executable,
            "TEST_CALL_LOG": str(call_log),
        },
    )

    assert result.returncode == 0, result.stderr or result.stdout
    scan_call = next(
        line for line in call_log.read_text().splitlines() if "--danger --json" in line
    )
    assert "--exclude .worktrees" in scan_call


def test_skylos_verify_script_wraps_repo_context_verifier() -> None:
    """The Skylos verify helper should keep agent checks repo-scoped and deterministic."""
    script = SKYLOS_VERIFY_SCRIPT.read_text(encoding="utf-8")

    assert "Usage: bash tools/run_skylos_verify.sh" in script
    assert 'cd "$(quality_repo_root)"' in script
    assert "quality_require_python_tool skylos" in script
    assert 'quality_run_skylos verify . --project-context "$@"' in script


def test_defusedxml_probe_is_a_real_import_without_suppression() -> None:
    """Defusedxml is bound as a nullable module: scanner-visible, no lint suppression.

    A bare ``find_spec`` probe hid the dependency from static dependency
    scanners (SKY-U005 false positive), while a plain guarded import needed an
    F401 suppression. Binding the module (the psutil pattern) satisfies both.
    """
    check_models_source = (PKG_ROOT / "check_models.py").read_text(encoding="utf-8")
    defusedxml_probe = check_models_source[
        check_models_source.index("defusedxml is required") : check_models_source.index(
            "try:\n    import numpy as np",
        )
    ]

    assert "import defusedxml.ElementTree as _DefusedElementTree" in defusedxml_probe
    assert "defusedxml_etree = _DefusedElementTree" in defusedxml_probe
    assert 'find_spec("defusedxml.ElementTree")' not in defusedxml_probe
    assert "F401" not in defusedxml_probe
    assert "noqa" not in defusedxml_probe
    assert check_models._defusedxml_available is (check_models.defusedxml_etree is not None)


@pytest.mark.subprocess
def test_pyrefly_quality_gate_fails_on_warnings(tmp_path: Path) -> None:
    """The quality helper should treat Pyrefly warnings as gate failures."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()

    fake_python = fake_bin / "python"
    fake_python.symlink_to(Path(sys.executable).resolve())

    fake_pyrefly = fake_bin / "pyrefly"
    fake_pyrefly.write_text(
        dedent(
            """\
            #!/usr/bin/env bash
            echo ' INFO Checking project configured at `fake`'
            echo ' WARN synthetic warning [test-warning]'
            echo ' INFO 0 errors'
            exit 0
            """
        ),
        encoding="utf-8",
    )
    fake_pyrefly.chmod(0o755)

    output_log = tmp_path / "pyrefly.log"
    run_script = tmp_path / "run_pyrefly_gate.sh"
    run_script.write_text(
        dedent(
            f"""\
            #!/usr/bin/env bash
            set -euo pipefail
            cd \"{PKG_ROOT}\"
            source tools/common_quality.sh
            QUALITY_PYTHON=\"{fake_python}\"
            QUALITY_PYTHON_SOURCE=\"synthetic-test\"
            if quality_run_pyrefly_check > \"{output_log}\" 2>&1; then
                pyrefly_gate_status=0
            else
                pyrefly_gate_status=$?
            fi
            cat \"{output_log}\"
            test \"$pyrefly_gate_status\" -eq 1
            """
        ),
        encoding="utf-8",
    )
    run_script.chmod(0o755)
    bash_path = Path("/bin/bash")

    result = subprocess.run(  # noqa: S603 - fixed /bin/bash runs a test-created script
        [str(bash_path), str(run_script)],
        capture_output=True,
        text=True,
        check=False,
        cwd=PKG_ROOT,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "WARN synthetic warning [test-warning]" in result.stdout
    assert "Pyrefly emitted warnings; treat warnings as quality failures." in result.stdout


@pytest.mark.subprocess
def test_pyrefly_generated_config_neutralizes_parent_repo_ignore_files(
    tmp_path: Path,
) -> None:
    """The generated Pyrefly config must not inherit parent-repo ignore files.

    Linked worktrees (for example .claude/worktrees/*) sit under a hidden
    directory that the parent repo's .git/info/exclude ignores, so project
    discovery finds zero files unless the generated config disables ignore-file
    collection and Pyrefly's hidden-directory exclude heuristic, restoring the
    dropped default excludes explicitly.
    """
    config_path = tmp_path / "pyrefly-generated.toml"
    run_script = tmp_path / "write_pyrefly_config.sh"
    run_script.write_text(
        dedent(
            f"""\
            #!/usr/bin/env bash
            set -euo pipefail
            cd \"{PKG_ROOT}\"
            source tools/common_quality.sh
            QUALITY_PYTHON=\"{sys.executable}\"
            quality_write_pyrefly_config \"{config_path}\" \"{sys.executable}\"
            """
        ),
        encoding="utf-8",
    )
    run_script.chmod(0o755)

    result = subprocess.run(  # noqa: S603 - fixed /bin/bash runs a test-created script
        ["/bin/bash", str(run_script)],
        capture_output=True,
        text=True,
        check=False,
        cwd=PKG_ROOT,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    generated = tomllib.loads(config_path.read_text(encoding="utf-8"))

    assert generated["use-ignore-files"] is False
    assert generated["disable-project-excludes-heuristics"] is True

    # The generated config is a throwaway that lives in $TMPDIR, so every
    # pattern and path is anchored to the package root rather than resolved
    # relative to the config file.
    project_excludes = generated["project-excludes"]
    for restored_default in ("**/node_modules/", "**/__pycache__/", "**/.*/**"):
        assert f"{PKG_ROOT.resolve()}/{restored_default}" in project_excludes, (
            f"disabled exclude heuristics must be restored explicitly: {restored_default}"
        )
    assert generated["project-includes"] == [f"{PKG_ROOT.resolve()}/**/*.py"]
    assert generated["search-path"][0] == str(PKG_ROOT.resolve())
    for entry in generated["search-path"]:
        assert Path(entry).is_absolute(), entry


def _update_script_function(name: str) -> str:
    """Extract one top-level shell function body from update.sh."""
    update_script = (PKG_ROOT / "tools" / "update.sh").read_text(encoding="utf-8")
    match = re.search(
        rf"^{re.escape(name)}\(\) \{{\n.*?^\}}\n", update_script, re.DOTALL | re.MULTILINE
    )
    assert match is not None, name
    return match.group(0)


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        pytest.param("0 1 0 1 1", "skip", id="unchanged-clean-verified"),
        pytest.param("0 1 1 1 1", "rebuild", id="unchanged-dirty"),
        pytest.param("0 0 0 1 1", "rebuild", id="changed-head"),
        pytest.param("0 1 0 0 1", "rebuild", id="wrong-editable-origin"),
        pytest.param("1 1 0 1 1", "rebuild", id="force-reinstall"),
        pytest.param("0 1 0 1 0", "rebuild", id="metal-compiler-changed"),
    ],
)
def test_update_script_rebuild_decision(inputs: str, expected: str) -> None:
    """A rebuild is skipped only for an unchanged, clean, verified checkout.

    ``inputs`` is force, unchanged, dirty, verified, toolchain. The metallib is
    also a product of the Metal compiler, so an Xcode upgrade
    behind an unchanged HEAD must rebuild as well.

    A dirty checkout can leave modified C++, Metal, or packaging inputs
    behind an unchanged HEAD, so the compiled extension in use would predate
    the source the provenance claims; it must always rebuild.
    """
    function = _update_script_function("mlx_repo_rebuild_decision")
    result = subprocess.run(  # noqa: S603 - fixed /bin/bash evaluates an extracted repo function
        [
            "/bin/bash",
            "-c",
            f"{function}\nmlx_repo_rebuild_decision {inputs}",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == expected


_HEAD_255328713 = "2553287133a" + "0" * 29


@pytest.mark.parametrize(
    ("runtime", "metadata", "expected_status", "reason"),
    [
        ("0.32.4.dev20261002+255328713", "0.32.4.dev20261002+255328713", 0, ""),
        (
            "0.32.3.dev20260927+02ce1fb6a",
            "0.32.3.dev20260927+02ce1fb6a",
            1,
            "compiled mlx.core reports 0.32.3.dev20260927+02ce1fb6a",
        ),
        # A stale extension that still imports beside current metadata.
        (
            "0.32.3.dev20260927+02ce1fb6a",
            "0.32.4.dev20261002+255328713",
            1,
            "compiled mlx.core reports 0.32.3.dev20260927+02ce1fb6a",
        ),
        (
            "0.32.4.dev20261002+255328713",
            "0.32.3.dev20260927+02ce1fb6a",
            1,
            "mlx package metadata reports 0.32.3.dev20260927+02ce1fb6a",
        ),
        ("0.32.4", "0.32.4.dev20261002+255328713", 1, "compiled mlx.core reports 0.32.4,"),
        (None, "0.32.4.dev20261002+255328713", 1, "mlx.core does not import"),
    ],
)
def test_update_script_skips_a_rebuild_only_for_the_checkouts_own_build(
    *, runtime: str | None, metadata: str, expected_status: int, reason: str
) -> None:
    """Both the loaded binary and pip's record must be builds of checkout HEAD.

    Regressions: a lost mlx.core behind an unchanged pull was "verified" and
    skipped; and pip metadata alone passed a stale extension that still imports.
    """
    function = _update_script_function("installed_build_matches_checkout")
    python_stub = f"echo '{runtime}'" if runtime is not None else "return 1"
    stubs = (
        f"python() {{ {python_stub}; }}\n"
        f"get_installed_distribution_version() {{ echo '{metadata}'; }}\n"
        f"git() {{ echo '{_HEAD_255328713}'; }}\n"
    )
    result = subprocess.run(  # noqa: S603 - fixed /bin/bash evaluates an extracted repo function
        ["/bin/bash", "-c", f"{stubs}{function}\ninstalled_build_matches_checkout mlx /repo"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == expected_status
    assert reason in result.stdout
    other_package = subprocess.run(  # noqa: S603 - as above
        ["/bin/bash", "-c", f"{function}\ninstalled_build_matches_checkout other-pkg /repo"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert other_package.returncode == 0


@pytest.mark.parametrize(
    ("declared", "installed", "expected_status", "reason"),
    [
        pytest.param("0.7.11", "0.7.11", 0, "", id="consistent"),
        # A failed editable reinstall left the previous dist-info behind while
        # the .pth already imported the pulled checkout (2026-10-28); the next
        # run's pull was a no-op on a clean checkout, so without this check
        # the skew never healed.
        pytest.param(
            "0.7.11",
            "0.7.10",
            1,
            "mlx-vlm package metadata reports 0.7.10, not the 0.7.11 the checkout declares",
            id="stale-dist-info",
        ),
        pytest.param(
            None,
            "0.7.10",
            1,
            "not the <unknown> the checkout declares",
            id="undeclared-version",
        ),
    ],
)
def test_update_script_rebuild_check_compares_mlx_vlm_declared_version(
    tmp_path: Path, declared: str | None, installed: str, expected_status: int, reason: str
) -> None:
    """mlx-vlm's pip-recorded version must match the checkout's declared one."""
    repo = tmp_path / "mlx-vlm"
    (repo / "mlx_vlm").mkdir(parents=True)
    if declared is not None:
        safe_io.write_text_no_follow(
            repo / "mlx_vlm" / "version.py", f'__version__ = "{declared}"\n'
        )
    function = _update_script_function("installed_build_matches_checkout")
    stubs = f"get_installed_distribution_version() {{ echo '{installed}'; }}\n"
    result = subprocess.run(  # noqa: S603 - fixed /bin/bash evaluates an extracted repo function
        [
            "/bin/bash",
            "-c",
            f'{stubs}{function}\ninstalled_build_matches_checkout mlx-vlm "$1"',
            "_",
            str(repo),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == expected_status
    assert reason in result.stdout


_LOCAL_DEV_MLX = "0.32.5.dev20261028+7d2d94a55"


def _pin_assessment_report(tmp_path: Path, requires_dist: list[str]) -> Path:
    report = tmp_path / "report.json"
    safe_io.write_text_no_follow(
        report,
        json.dumps(
            {
                "install": [
                    {"metadata": {"version": "0.6.3", "requires_dist": requires_dist}},
                ],
            },
        ),
    )
    return report


@pytest.mark.parametrize(
    ("requires_dist", "local_mlx", "expected"),
    [
        pytest.param([], _LOCAL_DEV_MLX, "0.6.3\tnormal\t", id="no-mlx-requirement"),
        # Conditional pins (extras or markers) are not the unconditional
        # resolution blocker the reconcile targets.
        pytest.param(
            ['mlx==0.32.4; extra == "cuda"'], _LOCAL_DEV_MLX, "0.6.3\tnormal\t", id="marker-pin"
        ),
        pytest.param(
            ["mlx>=0.32.4"], _LOCAL_DEV_MLX, "0.6.3\tsatisfied\t>=0.32.4", id="floor-satisfied"
        ),
        # PEP 440: ==0.32.4 accepts 0.32.4+local; a shell string comparison
        # called this a conflict.
        pytest.param(
            ["mlx==0.32.4"], "0.32.4+local", "0.6.3\tsatisfied\t==0.32.4", id="pin-local-label"
        ),
        pytest.param(
            ["mlx==0.32.*"], _LOCAL_DEV_MLX, "0.6.3\tsatisfied\t==0.32.*", id="wildcard-satisfied"
        ),
        # The 2026-10-28 failure: every mlx-audio >=0.6.2 pins mlx==0.32.4,
        # which a local 0.32.5 dev build can never satisfy.
        pytest.param(
            ["mlx==0.32.4", "numpy>=1.26"], _LOCAL_DEV_MLX, "0.6.3\trelax\t0.32.4", id="relaxable"
        ),
        # ==0.33.* has no valid >= form; relaxing it would write invalid metadata.
        pytest.param(
            ["mlx==0.33.*"], _LOCAL_DEV_MLX, "0.6.3\tunsupported\t==0.33.*", id="wildcard-ahead"
        ),
        # A >=X floor the local build cannot meet must not be written either:
        # the resolution would still be impossible, with metadata already mutated.
        pytest.param(
            ["mlx==0.33.0"], _LOCAL_DEV_MLX, "0.6.3\tunsupported\t==0.33.0", id="pin-ahead"
        ),
    ],
)
def test_update_script_mlx_audio_pin_assessment(
    tmp_path: Path, requires_dist: list[str], local_mlx: str, expected: str
) -> None:
    """Requirement interpretation uses packaging semantics in one place."""
    report = _pin_assessment_report(tmp_path, requires_dist)
    function = _update_script_function("mlx_audio_pin_assessment")
    stubs = f'python() {{ {shlex.quote(sys.executable)} "$@"; }}\n'
    result = subprocess.run(  # noqa: S603 - fixed /bin/bash evaluates an extracted repo function
        [
            "/bin/bash",
            "-c",
            f'{stubs}{function}\nmlx_audio_pin_assessment "$1" "$2"',
            "_",
            str(report),
            local_mlx,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.rstrip("\n") == expected


def _run_reconcile_harness(
    tmp_path: Path,
    requires_dist: list[str],
    *,
    install_status: int = 0,
    relax_status: int = 0,
) -> tuple[int, str, str]:
    """Run the full reconcile function under set -e inside the caller's `if !`.

    The enclosing conditional is the regression surface: it makes set -e
    inert inside the function, so only explicit status checks can stop a
    failed install, metadata rewrite or hold from being recorded as success.
    Returns (exit status, stdout, final constraint-file content).
    """
    repo = tmp_path / "mlx-vlm"
    repo.mkdir()
    safe_io.write_text_no_follow(repo / "requirements.txt", "mlx>=0.32.4\nmlx-audio>=0.6.2\n")
    report = _pin_assessment_report(tmp_path, requires_dist)
    constraint = tmp_path / "pin.txt"
    safe_io.write_text_no_follow(constraint, f"mlx=={_LOCAL_DEV_MLX}\n")
    functions = _update_script_function("mlx_audio_pin_assessment") + _update_script_function(
        "reconcile_mlx_audio_with_local_mlx"
    )
    script = f"""
set -euo pipefail
pip() {{
    local previous="" arg
    if [[ " $* " == *" --report "* ]]; then
        for arg in "$@"; do
            if [[ "$previous" == "--report" ]]; then cp {shlex.quote(str(report))} "$arg"; fi
            previous="$arg"
        done
        return 0
    fi
    if [[ " $* " == *" --no-deps "* ]]; then return {install_status}; fi
    return 0
}}
python() {{
    if [[ "$1" == *relax_exact_mlx_pin.py ]]; then return {relax_status}; fi
    {shlex.quote(sys.executable)} "$@"
}}
get_installed_distribution_version() {{
    case "$1" in
        mlx) echo {shlex.quote(_LOCAL_DEV_MLX)} ;;
        mlx-audio) echo "0.5.9" ;;
    esac
}}
SCRIPT_DIR=/stub-tools
COMPLETED_STEPS=()
LOCAL_MLX_CONSTRAINT_FILE={shlex.quote(str(constraint))}
{functions}
if ! reconcile_mlx_audio_with_local_mlx {shlex.quote(str(repo))}; then
    echo STOPPED
    exit 3
fi
"""
    result = subprocess.run(  # noqa: S603 - fixed /bin/bash evaluates extracted repo functions
        ["/bin/bash", "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, result.stdout, safe_io.read_text_no_follow(constraint)


def test_update_script_reconcile_succeeds_and_holds_only_after_relaxing(tmp_path: Path) -> None:
    """The happy path relaxes the pin, then holds the version for the run."""
    status, stdout, constraint = _run_reconcile_harness(tmp_path, ["mlx==0.32.4"])
    assert status == 0
    assert "STOPPED" not in stdout
    assert "Holding mlx-audio at 0.6.3" in stdout
    assert constraint.endswith("mlx-audio==0.6.3\n")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        pytest.param({"install_status": 1}, "Could not install mlx-audio==0.6.3", id="install"),
        pytest.param({"relax_status": 1}, "Could not relax the mlx pin", id="metadata-rewrite"),
    ],
)
def test_update_script_reconcile_failures_stop_the_caller(
    tmp_path: Path, kwargs: dict[str, int], message: str
) -> None:
    """A failed install or metadata rewrite stops the run and holds nothing.

    Regression: invoked inside `if !`, the function once returned 0 past
    these failures, recorded the hold, and the run reported success.
    """
    status, stdout, constraint = _run_reconcile_harness(tmp_path, ["mlx==0.32.4"], **kwargs)
    assert status == 3
    assert "STOPPED" in stdout
    assert message in stdout
    assert "Holding" not in stdout
    assert "mlx-audio" not in constraint


def test_update_script_reconcile_refuses_an_unrelaxable_requirement(tmp_path: Path) -> None:
    """A pin whose >= floor the local build cannot meet stops before any change."""
    status, stdout, constraint = _run_reconcile_harness(tmp_path, ["mlx==0.33.0"])
    assert status == 3
    assert "STOPPED" in stdout
    assert "cannot safely relax" in stdout
    assert "mlx-audio" not in constraint


def _write_metadata_dist_info(site_dir: Path, name: str, version: str, metadata: str) -> Path:
    metadata_dir = site_dir / f"{name}-{version}.dist-info"
    metadata_dir.mkdir(parents=True)
    metadata_file = metadata_dir / "METADATA"
    safe_io.write_text_no_follow(metadata_file, metadata)
    return metadata_file


def test_relax_exact_mlx_pin_rewrites_only_the_bare_exact_pin(tmp_path: Path) -> None:
    """``mlx==X`` becomes ``mlx>=X``; conditional and multi-clause lines stay."""
    metadata_file = _write_metadata_dist_info(
        tmp_path,
        "mlx_audio",
        "0.6.3",
        "Metadata-Version: 2.4\n"
        "Name: mlx-audio\n"
        "Version: 0.6.3\n"
        "Requires-Dist: mlx==0.32.4\n"
        "Requires-Dist: numpy>=1.26.4\n"
        'Requires-Dist: mlx-lm==0.32.2; extra == "parity"\n',
    )
    outcomes = relax_exact_mlx_pin.relax_exact_pin("mlx-audio", "mlx", site_dirs=[tmp_path])
    assert outcomes == [f"relaxed mlx==0.32.4 to mlx>=0.32.4 in {metadata_file}"]
    rewritten = safe_io.read_text_no_follow(metadata_file)
    assert "Requires-Dist: mlx>=0.32.4\n" in rewritten
    assert "mlx==0.32.4" not in rewritten
    # The conditional mlx-lm pin and the unrelated floor are untouched.
    assert 'Requires-Dist: mlx-lm==0.32.2; extra == "parity"\n' in rewritten
    assert "Requires-Dist: numpy>=1.26.4\n" in rewritten
    # Idempotent: a second run finds nothing left to relax and still succeeds.
    assert relax_exact_mlx_pin.relax_exact_pin("mlx-audio", "mlx", site_dirs=[tmp_path]) == [
        f"no bare exact mlx pin to relax in {metadata_file}"
    ]


def test_relax_exact_mlx_pin_leaves_wildcard_pins_alone(tmp_path: Path) -> None:
    """``mlx==0.32.*`` has no valid ``>=`` form and must never be rewritten."""
    metadata_file = _write_metadata_dist_info(
        tmp_path,
        "mlx_audio",
        "0.6.3",
        "Metadata-Version: 2.4\nName: mlx-audio\nVersion: 0.6.3\nRequires-Dist: mlx==0.32.*\n",
    )
    before = safe_io.read_text_no_follow(metadata_file)
    outcomes = relax_exact_mlx_pin.relax_exact_pin("mlx-audio", "mlx", site_dirs=[tmp_path])
    assert outcomes == [f"no bare exact mlx pin to relax in {metadata_file}"]
    assert safe_io.read_text_no_follow(metadata_file) == before


def test_relax_exact_mlx_pin_failure_keeps_the_original_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A write that fails part-way leaves the installed METADATA byte-for-byte intact.

    Regression: the rewrite once truncated METADATA in place before writing,
    so a failure after opening it left an empty file in the environment.
    """
    metadata_file = _write_metadata_dist_info(
        tmp_path,
        "mlx_audio",
        "0.6.3",
        "Metadata-Version: 2.4\nName: mlx-audio\nVersion: 0.6.3\nRequires-Dist: mlx==0.32.4\n",
    )
    metadata_file.chmod(0o644)
    before = metadata_file.read_bytes()

    def fail_after_writing(_fd: int) -> None:
        raise OSError(28, "No space left on device")

    with monkeypatch.context() as patch:
        patch.setattr(safe_io.os, "fsync", fail_after_writing)
        with pytest.raises(OSError, match="No space left"):
            relax_exact_mlx_pin.relax_exact_pin("mlx-audio", "mlx", site_dirs=[tmp_path])
    assert metadata_file.read_bytes() == before
    assert sorted(path.name for path in metadata_file.parent.iterdir()) == ["METADATA"]

    # Once the failure clears, the rewrite lands and keeps the file's permissions.
    relax_exact_mlx_pin.relax_exact_pin("mlx-audio", "mlx", site_dirs=[tmp_path])
    assert "Requires-Dist: mlx>=0.32.4\n" in safe_io.read_text_no_follow(metadata_file)
    assert metadata_file.stat().st_mode & 0o777 == 0o644


def test_relax_exact_mlx_pin_requires_installed_metadata(tmp_path: Path) -> None:
    """A missing distribution is an error, not a silent no-op."""
    with pytest.raises(RuntimeError, match="no installed metadata"):
        relax_exact_mlx_pin.relax_exact_pin("mlx-audio", "mlx", site_dirs=[tmp_path])


def _conflict_probe(function_name: str, repo: Path) -> list[str]:
    """Run one extracted update.sh detector against a repo; return the paths it reports."""
    function = _update_script_function(function_name)
    result = subprocess.run(  # noqa: S603 - fixed /bin/bash evaluates an extracted repo function
        ["/bin/bash", "-c", f'set -euo pipefail\n{function}\n{function_name} "$1"', "_", str(repo)],
        capture_output=True,
        text=True,
        check=True,  # nothing found must not trip errexit/pipefail
    )
    return [line.removeprefix(f"{repo}/") for line in result.stdout.splitlines()]


def test_update_script_flags_conflict_copies_only_with_evidence(tmp_path: Path) -> None:
    """A numbered name alone is legitimate; one beside its original is only suspected."""
    repo = tmp_path / "repo"
    subprocess.run(  # noqa: S603 - fixed git command on a test-created directory
        ["git", "init", "-q", str(repo)],  # noqa: S607 - git from PATH, as update.sh uses it
        check=True,
    )
    for name in ("ops.py", "ops 2.py", "experiment 2.py", "notes 3"):
        safe_io.write_text_no_follow(repo / name, "x\n")
    assert _conflict_probe("find_suspected_conflict_copies", repo) == ["ops 2.py"]
    assert _conflict_probe("find_malformed_git_refs", repo) == []

    refs = repo / ".git" / "refs" / "heads"
    refs.mkdir(parents=True, exist_ok=True)
    safe_io.write_text_no_follow(refs / "main", "a" * 40 + "\n")
    safe_io.write_text_no_follow(refs / "main 2", "b" * 40 + "\n")
    assert _conflict_probe("find_malformed_git_refs", repo) == [".git/refs/heads/main 2"]


def test_update_script_leaves_setuptools_to_pip_above_mlxs_build_floor() -> None:
    """No upper cap: builds are isolated, and the old <82 cap only made the env look stale."""
    update_script = (PKG_ROOT / "tools" / "update.sh").read_text(encoding="utf-8")
    assert "<82" not in update_script
    assert "--no-build-isolation" not in update_script


_PROBE_FAKE_MLX_VERSION = "0.32.4.dev20261002+abc1234"


def _run_probe_python_next(
    tmp_path: Path,
    *,
    fail_regex: str = "",
    **probe_env: str,
) -> tuple[subprocess.CompletedProcess[str], list[str], Path]:
    """Run probe_python_next.sh against fake conda/python; return result, call log, mlx repo.

    The fake interpreter logs each call as ``PY[<cwd>] <args>`` (plus the
    contents of any ``--constraint`` file) and fails when its arguments match
    ``fail_regex``. The mlx checkout is a real one-commit git repo, so the
    script's clone step runs for real.
    """
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    call_log = tmp_path / "calls.log"
    fake_py = fake_bin / "python3.15"
    safe_io.write_text_no_follow(
        fake_py,
        dedent(
            f"""\
            #!/bin/bash
            printf 'PY[%s] %s\\n' "$PWD" "$*" >> "{call_log}"
            prev=""
            for arg in "$@"; do
                if [[ "$prev" == "--constraint" ]]; then
                    printf 'CONSTRAINT %s\\n' "$(cat "$arg")" >> "{call_log}"
                fi
                prev="$arg"
            done
            if [[ "${{1:-}}" == "-" ]]; then cat > /dev/null; fi
            if [[ -n "$FAKE_FAIL" ]] && printf '%s' "$*" | grep -qE -- "$FAKE_FAIL"; then
                exit 1
            fi
            case "${{1:-}}:${{2:-}}" in
                -V:*) echo "Python 3.15.0" ;;
                -c:*tomllib*) exec "{sys.executable}" "$@" ;;
                -c:*importlib.metadata*) echo "{_PROBE_FAKE_MLX_VERSION}" ;;
            esac
            """,
        ),
        mode=0o755,
    )
    safe_io.write_text_no_follow(
        fake_bin / "conda",
        dedent(
            f"""\
            #!/bin/bash
            case "$1" in
                env) printf 'base  /opt/conda\\nmlx-vlm-315  /opt/conda/envs/mlx-vlm-315\\n' ;;
                run) if [[ "$*" == *sys.executable* ]]; then echo "{fake_py}"; else echo "Python 3.14.9"; fi ;;
                *) echo "unexpected conda $*" >&2; exit 2 ;;
            esac
            """,
        ),
        mode=0o755,
    )

    mlx_repo = tmp_path / "mlx"
    git = ["git", "-C", str(mlx_repo), "-c", "user.name=probe", "-c", "user.email=probe@test"]
    subprocess.run(  # noqa: S603 - fixed git command on a test-created directory
        ["git", "init", "-q", str(mlx_repo)],  # noqa: S607 - git from PATH, as the probe uses it
        check=True,
    )
    safe_io.write_text_no_follow(mlx_repo / "setup.py", "# placeholder\n")
    subprocess.run([*git, "add", "setup.py"], check=True)  # noqa: S603 - fixed git command on a test repo
    subprocess.run([*git, "commit", "-q", "-m", "init"], check=True)  # noqa: S603 - fixed git command on a test repo

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    base_env = {
        k: v for k, v in os.environ.items() if not k.startswith(("PROBE_", "CONDA_DEFAULT"))
    }
    result = subprocess.run(  # noqa: S603 - fixed /bin/bash runs the shipped probe with fakes
        ["/bin/bash", str(PKG_ROOT / "tools" / "probe_python_next.sh")],
        capture_output=True,
        text=True,
        check=False,
        env={
            **base_env,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "TMPDIR": str(scratch),
            "PROBE_PYTHON": "3.15",
            "PROBE_MLX_REPO": str(mlx_repo),
            "FAKE_FAIL": fail_regex,
            **probe_env,
        },
    )
    assert not list(scratch.iterdir()), "probe left its temp dir behind"
    calls = safe_io.read_text_no_follow(call_log).splitlines() if call_log.exists() else []
    return result, calls, mlx_repo


def _probe_call_index(calls: list[str], needle: str) -> int:
    return next(i for i, line in enumerate(calls) if needle in line)


def test_probe_python_next_builds_mlx_in_a_clone_before_the_pypi_install(tmp_path: Path) -> None:
    """With PROBE_SOURCE_BUILD=1 a missing mlx wheel must not hide the source-build result.

    The build runs first, in a throwaway clone (an in-place build would
    overwrite the dylib and metallib the working env loads), and the project
    install is constrained to that build.
    """
    result, calls, mlx_repo = _run_probe_python_next(tmp_path, PROBE_SOURCE_BUILD="1")
    assert result.returncode == 0, result.stdout + result.stderr

    build = _probe_call_index(calls, "-m pip install -q .")
    project = _probe_call_index(calls, "[extras]")
    assert build < project
    assert calls[build].startswith(f"PY[{tmp_path / 'scratch'}/")
    assert "--no-build-isolation" not in calls[build]
    assert "--constraint" in calls[project]
    assert calls[project + 1] == f"CONSTRAINT mlx=={_PROBE_FAKE_MLX_VERSION}"
    assert "Core: viable, including the mlx source build" in result.stdout

    status = subprocess.run(  # noqa: S603 - fixed git command on a test repo
        ["git", "-C", str(mlx_repo), "status", "--porcelain", "--ignored"],  # noqa: S607 - git from PATH
        capture_output=True,
        text=True,
        check=True,
    )
    assert status.stdout == ""


@pytest.mark.parametrize(
    ("env", "fail_regex", "outcome"),
    [
        pytest.param(
            {"PROBE_SOURCE_BUILD": "1"},
            r"install -q \.$",
            (
                1,
                ["mlx source build FAILED", "Core: not viable yet"],
                "[extras]",
            ),
            id="failed-build-skips-pypi-install",
        ),
        pytest.param(
            {},
            r"\[extras\]",
            (
                1,
                ["PROBE_SOURCE_BUILD=1 builds it from the local checkout", "Core: not viable yet"],
                "-m pytest",
            ),
            id="missing-wheel-points-at-source-build",
        ),
        pytest.param(
            {"PROBE_TORCH": "1"},
            "torchvision>=",
            (
                0,
                [
                    "torch extra unavailable: about half the roster would fail",
                    "Core: viable from PyPI",
                ],
                "",
            ),
            id="torch-failure-stays-outside-core-verdict",
        ),
        pytest.param(
            {"PROBE_TORCH": "1"},
            r"\[extras\]",
            (
                1,
                ["Core: not viable yet", "Torch extra: available"],
                "",
            ),
            id="torch-checked-even-when-core-fails",
        ),
    ],
)
def test_probe_python_next_verdicts(
    tmp_path: Path,
    env: dict[str, str],
    fail_regex: str,
    outcome: tuple[int, list[str], str],
) -> None:
    """Each check's failure is reported, gates only the checks that need it, and torch never sets the exit."""
    returncode, expected, absent = outcome
    result, calls, _ = _run_probe_python_next(tmp_path, fail_regex=fail_regex, **env)
    assert result.returncode == returncode, result.stdout + result.stderr
    for text in expected:
        assert text in result.stdout
    if absent:
        assert not any(absent in line for line in calls)


def test_probe_python_next_torch_check_installs_the_pyproject_extra(tmp_path: Path) -> None:
    """The torch check installs exactly the torch extra's requirements, without the project."""
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    torch_extra = pyproject["project"]["optional-dependencies"]["torch"]

    result, calls, _ = _run_probe_python_next(tmp_path, PROBE_TORCH="1")
    assert result.returncode == 0, result.stdout + result.stderr
    torch_install = calls[_probe_call_index(calls, "torchvision>=")]
    assert torch_install.endswith("-m pip install -q " + " ".join(torch_extra))
    assert "Torch extra: available" in result.stdout
    assert "--index-url" not in "\n".join(calls)


def test_probe_python_next_torch_check_falls_back_to_the_pytorch_index(tmp_path: Path) -> None:
    """Torch wheels on PyTorch's own index only (as for 3.15) are found and named as such.

    Only the torch packages come from that index, with --no-deps because
    --index-url also governs dependency resolution; the rest of the extra and
    the torch packages' own dependencies then resolve from PyPI.
    """
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    torch_extra = pyproject["project"]["optional-dependencies"]["torch"]
    torch_family = [
        req for req in torch_extra if re.match(r"(torch|torchvision|torchaudio)\b(?![-_.])", req)
    ]
    rest = [req for req in torch_extra if req not in torch_family]
    assert torch_family
    assert rest

    # Fail only the first, all-PyPI install of the whole extra.
    pypi_only = " ".join(torch_extra).replace(".", r"\.") + "$"
    result, calls, _ = _run_probe_python_next(tmp_path, fail_regex=pypi_only, PROBE_TORCH="1")
    assert result.returncode == 0, result.stdout + result.stderr

    index_install = _probe_call_index(calls, "--index-url")
    assert calls[index_install].endswith(
        "-m pip install -q --no-deps --index-url https://download.pytorch.org/whl/cpu "
        + " ".join(torch_family)
    )
    assert calls[index_install + 1].endswith("-m pip install -q " + " ".join(rest + torch_family))
    assert "--index-url" not in calls[index_install + 1]
    assert "Torch extra: available from https://download.pytorch.org/whl/cpu only" in result.stdout
    assert "Core: viable from PyPI" in result.stdout


def test_probe_python_next_rejects_a_non_https_torch_index(tmp_path: Path) -> None:
    """PROBE_TORCH_INDEX reaches pip; anything but a plain https URL stops the probe first."""
    result, calls, _ = _run_probe_python_next(
        tmp_path, PROBE_TORCH="1", PROBE_TORCH_INDEX="http://example.test/simple"
    )
    assert result.returncode == 1
    assert "PROBE_TORCH_INDEX must be a plain https:// URL" in result.stderr
    assert calls == []


def test_update_script_wires_dirty_state_into_the_rebuild_decision() -> None:
    """Stage 2 records git status --porcelain; Stage 3 consults it before skipping."""
    update_script = (PKG_ROOT / "tools" / "update.sh").read_text(encoding="utf-8")

    assert 'if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then' in update_script
    assert "REPO_DIRTY[idx]=1" in update_script
    decision_call = (
        'mlx_repo_rebuild_decision "${FORCE_REINSTALL:-0}" "${REPO_UNCHANGED[idx]}" '
        '"${REPO_DIRTY[idx]}" "$editable_verified" "$toolchain_matches"'
    )
    # The stamp lives under .git so it never dirties the checkout it describes,
    # and it is written only after a successful mlx install.
    assert '"$1/.git/check_models_build_toolchain"' in update_script
    # A changed compiler only warns unless the rebuild is asked for: upstream
    # main may not compile under a new Xcode, and a failed build would swap
    # the working local build for a PyPI wheel.
    assert 'if [[ "${MLX_REBUILD_ON_TOOLCHAIN_CHANGE:-0}" == "1" ]]; then' in update_script
    assert "MLX_REBUILD_ON_TOOLCHAIN_CHANGE" in (PKG_ROOT / "README.md").read_text(encoding="utf-8")
    assert update_script.index("installed successfully") < update_script.index(
        'metal_compiler_version > "$(mlx_build_toolchain_stamp'
    )
    assert decision_call in update_script
    assert update_script.index("REPO_DIRTY[idx]=1") < update_script.index(decision_call)


def test_quality_script_drops_hook_exported_git_dir_before_entering_src() -> None:
    """Git exports GIT_DIR into hooks; left set, git treats src/ as the work tree.

    Every ``git diff`` a tool ran from ``src/`` then reported the whole
    repository as deleted (a Skylos SKY-L021 storm from linked worktrees).
    """
    quality_script = (PKG_ROOT / "tools" / "run_quality_checks.sh").read_text(encoding="utf-8")

    assert "\nunset GIT_DIR\n" in quality_script
    assert quality_script.index("unset GIT_DIR") < quality_script.index('cd "$(quality_src_root)"')


def test_update_script_uses_upstream_mlx_editable_dev_install() -> None:
    """Local MLX builds should follow upstream's editable dev install guidance."""
    update_script = (PKG_ROOT / "tools" / "update.sh").read_text(encoding="utf-8")
    assert 'INSTALL_CMD=(pip_install_verbose -e ".[dev]")' in update_script
    assert "macOS SDK" in update_script
    assert "Apple Clang" in update_script
    assert "Native arm64 shell detected" in update_script
    assert "MLX_LOCAL_BUILD_SMOKE" in update_script
    assert "backup_run_history" in update_script
    assert "HISTORY_BACKUP_DIR" in update_script
    assert "HISTORY_BACKUP_DIR" in (PKG_ROOT / "README.md").read_text(encoding="utf-8")
    assert "mlx.metallib" in update_script
    assert "MLX runtime backend provenance" in update_script


def test_update_script_import_repair_hint_uses_distribution_names() -> None:
    """Postflight repair output should use pip package names, not import module names."""
    update_script = (PKG_ROOT / "tools" / "update.sh").read_text(encoding="utf-8")

    assert "IMPORT_TO_PIP_PACKAGE" in update_script
    assert "printf '%s\\n' \"mlx-vlm\"" in update_script
    assert "REPAIR_PKGS" in update_script
    assert 'REPAIR_PKGS+=("$(IMPORT_TO_PIP_PACKAGE "$pkg")")' in update_script
    assert "Fix with: pip install ${REPAIR_PKGS[*]}" in update_script
    assert "Fix with: pip install ${MISSING_PKGS[*]}" not in update_script


def test_update_script_updates_system_packages_and_node_tooling_by_default() -> None:
    """The updater refreshes system packages and moves markdownlint-cli2 to latest by default."""
    update_script = (PKG_ROOT / "tools" / "update.sh").read_text(encoding="utf-8")
    readme = (PKG_ROOT / "README.md").read_text(encoding="utf-8")

    assert "UPDATE_SYSTEM_PACKAGES" in update_script
    assert "UPDATE_NODE_TOOLING" in update_script
    assert 'if [[ "${UPDATE_SYSTEM_PACKAGES:-1}" == "1" ]]; then' in update_script
    assert 'if [[ "${UPDATE_NODE_TOOLING:-1}" == "1" ]]; then' in update_script
    assert "Skipping conda base/environment package updates (UPDATE_SYSTEM_PACKAGES=0)" in (
        update_script
    )
    assert "Skipping Homebrew update/upgrade (UPDATE_SYSTEM_PACKAGES=0)" in update_script
    assert "Installing repo-local markdownlint tooling from package-lock.json" in update_script
    # Both switches are documented in the README environment table.
    assert "`UPDATE_SYSTEM_PACKAGES`" in readme
    assert "`UPDATE_NODE_TOOLING`" in readme

    package_latest = "markdownlint-cli2@latest"
    assert package_latest in update_script
    # Latest comes first (the default branch); the lockfile install is the opt-out.
    assert update_script.index(package_latest) < update_script.index(
        "Installing repo-local markdownlint tooling from package-lock.json"
    )


def test_update_script_cleans_stale_pip_invalid_distribution_backups() -> None:
    """update.sh should clean pip '~package' backups that cause invalid-distribution warnings."""
    update_script = (PKG_ROOT / "tools" / "update.sh").read_text(encoding="utf-8")

    assert "cleanup_pip_invalid_distribution_backups" in update_script
    assert 'if not name.startswith("~"):' in update_script
    assert "path.is_symlink()" in update_script
    assert "Removed stale pip invalid-distribution backup" in update_script
    assert "CLEAN_PIP_INVALID_DISTS=0" in update_script


def test_update_script_quarantines_broken_packaging_metadata_before_upgrade() -> None:
    """The packaging-tool upgrade should preflight only its own metadata."""
    update_script = (PKG_ROOT / "tools" / "update.sh").read_text(encoding="utf-8")
    helper_name = "quarantine_broken_pip_metadata.py"
    install_command = 'pip_install_tool pip wheel "setuptools>=80" build pyrefly'

    assert helper_name in update_script
    helper_position = update_script.index(helper_name)
    install_position = update_script.index(install_command)
    preflight = update_script[helper_position:install_position]
    assert "pip wheel setuptools build pyrefly" in " ".join(preflight.split())
    assert helper_position < install_position


def test_quarantine_broken_pip_metadata_moves_only_malformed_targets(
    tmp_path: Path,
) -> None:
    """A stale wheel metadata husk should not mask its healthy replacement."""
    quarantine = quarantine_broken_pip_metadata.quarantine_broken_metadata
    site_dir = tmp_path / "site-packages"
    site_dir.mkdir()
    quarantine_parent = tmp_path / "quarantine"
    quarantine_parent.mkdir()
    stale_wheel = _write_pip_dist_info(site_dir, "wheel", "0.47.0", complete=False)
    healthy_wheel = _write_pip_dist_info(site_dir, "wheel", "0.48.0", complete=True)
    unrelated = _write_pip_dist_info(site_dir, "example", "1.0.0", complete=False)

    moved = quarantine(
        ["wheel"],
        site_dirs=[site_dir],
        quarantine_parent=quarantine_parent,
    )

    assert len(moved) == 1
    source, destination = moved[0]
    assert source == stale_wheel
    assert not stale_wheel.exists()
    assert healthy_wheel.is_dir()
    assert unrelated.is_dir()
    assert destination.is_dir()
    assert (destination / "INSTALLER").read_text(encoding="utf-8") == "pip\n"
    assert destination.is_relative_to(quarantine_parent)


def test_quarantine_broken_pip_metadata_is_noop_for_healthy_target(
    tmp_path: Path,
) -> None:
    """Healthy metadata should not create an empty quarantine directory."""
    quarantine = quarantine_broken_pip_metadata.quarantine_broken_metadata
    site_dir = tmp_path / "site-packages"
    site_dir.mkdir()
    quarantine_parent = tmp_path / "quarantine"
    quarantine_parent.mkdir()
    healthy_wheel = _write_pip_dist_info(site_dir, "wheel", "0.48.0", complete=True)

    moved = quarantine(
        ["wheel"],
        site_dirs=[site_dir],
        quarantine_parent=quarantine_parent,
    )

    assert moved == []
    assert healthy_wheel.is_dir()
    assert list(quarantine_parent.iterdir()) == []


def test_quarantine_broken_pip_metadata_refuses_symlink(tmp_path: Path) -> None:
    """Metadata quarantine must not follow a matching symlink."""
    quarantine = quarantine_broken_pip_metadata.quarantine_broken_metadata
    site_dir = tmp_path / "site-packages"
    site_dir.mkdir()
    target = tmp_path / "outside"
    target.mkdir()
    metadata_link = site_dir / "wheel-0.47.0.dist-info"
    metadata_link.symlink_to(target, target_is_directory=True)

    with pytest.raises(RuntimeError, match="refusing symlinked metadata directory"):
        quarantine(["wheel"], site_dirs=[site_dir], quarantine_parent=tmp_path)

    assert metadata_link.is_symlink()
    assert target.is_dir()


def test_update_script_reconciles_project_after_mlx_dependency_churn() -> None:
    """update.sh should let pyproject.toml reconcile deps after MLX updates."""
    update_script = (PKG_ROOT / "tools" / "update.sh").read_text(encoding="utf-8")
    policy_source = (PKG_ROOT / "check_models_data" / "dependency_policy.py").read_text(
        encoding="utf-8",
    )
    main_flow = update_script[update_script.index("# Clean build artifacts if requested") :]

    assert "UPDATE_PIP_CONSTRAINT" not in update_script
    assert "UPDATE_PIP_CONSTRAINT_SPECS" not in policy_source
    # One shared invocation injects the private local-mlx pin (empty-safe on
    # bash 3.2); both public wrappers delegate to it.
    assert (
        'pip install "${args[@]}" ${LOCAL_MLX_PIN_ARGS[@]+"${LOCAL_MLX_PIN_ARGS[@]}"}'
        in update_script
    )
    assert update_script.count("run_eager_pip_install") >= 3  # def + two wrappers
    # Detection is a separate, non-mutating gate so set -e reaches the updater.
    assert "if local_mlx_repos_present; then" in update_script
    assert 'pip_install_tool "setuptools>=80"' in update_script

    local_update_pos = main_flow.index("update_local_mlx_repos")
    pypi_update_pos = main_flow.index("update_mlx_from_pypi_or_preserve_local")
    assert "pip_install mlx mlx-metal mlx-vlm" in _update_script_function(
        "update_mlx_from_pypi_or_preserve_local"
    )
    reconcile_pos = main_flow.index("reconcile_project_environment_from_pyproject")
    local_smoke_pos = main_flow.index("run_local_mlx_backend_smoke")
    critical_check_pos = main_flow.index("[update.sh] Verifying critical packages")

    assert local_update_pos < reconcile_pos
    assert pypi_update_pos < reconcile_pos
    assert reconcile_pos < local_smoke_pos
    assert reconcile_pos < critical_check_pos
    assert "python -m pip check" in update_script
    assert "python -m tools.validate_env" in update_script


def test_update_script_defers_macos_deployment_target_to_upstream_mlx() -> None:
    """Local mlx builds should let upstream MLX choose the deployment target."""
    update_script = (PKG_ROOT / "tools" / "update.sh").read_text(encoding="utf-8")

    mlx_build_start = update_script.index("# MLX build controls are passed to pip via CMAKE_ARGS.")
    mlx_build_end = update_script.index(
        "[[ ${REPO_SKIP[idx]} -eq 1 ]] && continue",
        mlx_build_start,
    )
    mlx_build = update_script[mlx_build_start:mlx_build_end]

    assert "MACOSX_DEPLOYMENT_TARGET" not in mlx_build
    assert "MLX_METAL_JIT" in mlx_build
    assert 'INSTALL_CMD=(pip_install_verbose -e ".[dev]")' in mlx_build


def test_quality_ci_defers_macos_deployment_target_to_upstream_mlx() -> None:
    """MacOS CI should let upstream MLX choose the deployment target."""
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "quality.yml").read_text(encoding="utf-8")
    )

    for job_name in ("static-quality", "runtime-smoke"):
        steps = workflow["jobs"][job_name]["steps"]
        step_names = {step.get("name") for step in steps}
        install_command = next(
            step["run"] for step in steps if step.get("name") == "Install dependencies"
        )

        assert "Target host macOS for native builds" not in step_names
        assert "MACOSX_DEPLOYMENT_TARGET" not in install_command


def test_ci_jobs_running_the_skylos_gate_check_out_full_history() -> None:
    """Skylos's PR gate resolves origin/main, which a shallow checkout lacks.

    Dependabot PRs #31 and #32 failed Static Quality with "Could not resolve
    PR policy base: origin/main" while skylos-advisory (full history) passed.
    """
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "quality.yml").read_text(encoding="utf-8")
    )
    for job_name in ("skylos-advisory", "static-quality"):
        checkout = next(
            step
            for step in workflow["jobs"][job_name]["steps"]
            if str(step.get("uses", "")).startswith("actions/checkout@")
        )
        assert checkout["with"]["fetch-depth"] == 0, job_name


def test_mlx_runtime_imports_avoid_static_native_module_resolution() -> None:
    """Static checkers should not need importable stubs for the native MLX module."""
    source_path = PKG_ROOT / "check_models.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    direct_mlx_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Import) and any(alias.name == "mlx.core" for alias in node.names)
    ]
    dynamic_mlx_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "mlx.core"
        and (
            (isinstance(node.func, ast.Name) and node.func.id == "import_module")
            or (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "importlib"
                and node.func.attr == "import_module"
            )
        )
    ]

    assert not direct_mlx_imports, source_path
    assert dynamic_mlx_imports, source_path


def test_workflows_pin_actions_and_keep_skylos_danger_advisory_nonblocking() -> None:
    """Workflow security hardening and advisory Skylos danger wiring should stay in place."""
    action_ref_pattern = re.compile(r"^[^@]+@[0-9a-f]{40}$")

    # Glob so a new workflow file cannot silently escape these requirements.
    workflow_paths = sorted(
        path
        for pattern in ("*.yml", "*.yaml")
        for path in (REPO_ROOT / ".github" / "workflows").glob(pattern)
    )
    assert workflow_paths, "no workflow files found"
    for workflow_path in workflow_paths:
        workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))

        assert workflow["permissions"] == {}
        for job_name, job in workflow["jobs"].items():
            assert job["permissions"] == {"contents": "read"}, job_name
            for step in job.get("steps", []):
                if "uses" in step:
                    assert action_ref_pattern.match(step["uses"]), (
                        workflow_path.name,
                        step["uses"],
                    )
                if step.get("name") == "Checkout":
                    assert step["with"]["persist-credentials"] is False
                if step.get("uses", "").startswith("actions/upload-artifact@"):
                    assert step["with"]["if-no-files-found"] == "error"

    quality_workflow = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "quality.yml").read_text(encoding="utf-8")
    )
    advisory_job = quality_workflow["jobs"]["skylos-advisory"]
    advisory_step_names = [step.get("name") for step in advisory_job["steps"]]

    assert advisory_job["runs-on"] == "ubuntu-latest"
    assert "Install Skylos" in advisory_step_names
    assert "Run Skylos advisory danger scan" in advisory_step_names

    static_quality_install = next(
        step["run"]
        for step in quality_workflow["jobs"]["static-quality"]["steps"]
        if step.get("name") == "Install dependencies"
    )
    assert "npm install --ignore-scripts --no-audit --no-fund --prefix src" in (
        static_quality_install
    )


def test_should_audit_path_excludes_generated_paths(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    included = repo_root / "src" / "module.py"
    excluded_build = repo_root / "src" / "build" / "lib" / "check_models.py"
    excluded_conda = repo_root / ".conda" / "lib" / "python3.13" / "site.py"
    excluded_worktree = repo_root / ".worktrees" / "upstream" / "module.py"
    excluded_output = repo_root / "src" / "output" / "results.md"
    included.parent.mkdir(parents=True)
    excluded_build.parent.mkdir(parents=True)
    excluded_conda.parent.mkdir(parents=True)
    excluded_worktree.parent.mkdir(parents=True)
    excluded_output.parent.mkdir(parents=True)
    included.write_text("print('ok')\n", encoding="utf-8")
    excluded_build.write_text("value = 1  # noqa: F401\n", encoding="utf-8")
    excluded_conda.write_text("value = 1  # noqa: F401\n", encoding="utf-8")
    excluded_worktree.write_text("value = 1  # noqa: F401\n", encoding="utf-8")
    excluded_output.write_text("<!-- markdownlint-disable MD028 -->\n", encoding="utf-8")

    assert check_suppressions.should_audit_path(included, repo_root) is True
    assert check_suppressions.should_audit_path(excluded_build, repo_root) is False
    assert check_suppressions.should_audit_path(excluded_conda, repo_root) is False
    assert check_suppressions.should_audit_path(excluded_worktree, repo_root) is False
    assert check_suppressions.should_audit_path(excluded_output, repo_root) is False


def test_find_suppressions_detects_specific_and_bare_directives(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.py"
    file_path.write_text(
        "value = 1  # noqa: F841\n"
        "other = 2  # noqa\n"
        "typed = value  # type: ignore[attr-defined]\n"
        "fallback = value  # type: ignore\n",
        encoding="utf-8",
    )

    findings = check_suppressions.find_suppressions(file_path)

    assert [finding.kind for finding in findings] == [
        "noqa",
        "bare-noqa",
        "type-ignore",
        "bare-type-ignore",
    ]
    assert findings[0].codes == ("F841",)
    assert findings[2].codes == ("attr-defined",)


def test_find_suppressions_ignores_suppression_text_inside_python_strings(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.py"
    file_path.write_text(
        'message = "Bare # noqa is not allowed"\nother = "# type: ignore[attr-defined]"\n',
        encoding="utf-8",
    )

    assert check_suppressions.find_suppressions(file_path) == []


def test_check_if_needed_fails_bare_suppressions(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    src_root = repo_root / "src"
    src_root.mkdir(parents=True)
    file_path = src_root / "sample.py"
    file_path.write_text("value = 1  # noqa\n", encoding="utf-8")
    finding = check_suppressions.SuppressionFinding(
        file_path=file_path,
        line_num=1,
        kind="bare-noqa",
        codes=(),
        line_text="value = 1  # noqa",
    )

    needed, reason = check_suppressions.check_if_needed(
        finding,
        repo_root=repo_root,
        src_root=src_root,
    )

    assert needed is False
    assert "not allowed" in reason


@pytest.mark.parametrize(
    ("checker_output", "expected_needed", "expected_reason"),
    [
        (
            "PLR0912 Too many branches",
            False,
            "No violation found for: PLR0915",
        ),
        (
            "PLR0912 Too many branches\nPLR0915 Too many statements",
            True,
            "Suppression needed: PLR0912, PLR0915 violations found",
        ),
    ],
)
def test_check_if_needed_requires_evidence_for_every_suppressed_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checker_output: str,
    expected_needed: bool,
    expected_reason: str,
) -> None:
    """Every code on a combined suppression must have checker evidence."""
    repo_root = tmp_path / "repo"
    src_root = repo_root / "src"
    src_root.mkdir(parents=True)
    file_path = src_root / "sample.py"
    file_path.write_text(
        "def sample():  # noqa: PLR0912, PLR0915 - fixture rationale\n    pass\n",
        encoding="utf-8",
    )
    finding = check_suppressions.SuppressionFinding(
        file_path=file_path,
        line_num=1,
        kind="noqa",
        codes=("PLR0912", "PLR0915"),
        line_text="def sample():  # noqa: PLR0912, PLR0915 - fixture rationale",
    )
    result = subprocess.CompletedProcess(
        args=["ruff"],
        returncode=1,
        stdout=checker_output,
        stderr="",
    )

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return result

    monkeypatch.setattr(check_suppressions, "_run_for_finding", fake_run)

    needed, reason = check_suppressions.check_if_needed(
        finding,
        repo_root=repo_root,
        src_root=src_root,
    )

    assert needed is expected_needed
    assert reason == expected_reason


def test_run_for_finding_uses_active_python_for_ruff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    src_root = repo_root / "src"
    src_root.mkdir(parents=True)
    file_path = src_root / "sample.py"
    file_path.write_text("unused = 1  # noqa: F841\n", encoding="utf-8")
    finding = check_suppressions.SuppressionFinding(
        file_path=file_path,
        line_num=1,
        kind="noqa",
        codes=("F841",),
        line_text="unused = 1  # noqa: F841",
    )
    recorded_args: list[str] = []

    def fake_run(
        args: list[str],
        *,
        capture_output: bool,
        text: bool,
        encoding: str,
        check: bool,
        cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        assert capture_output is True
        assert text is True
        assert encoding == "utf-8"
        assert check is False
        assert cwd == src_root
        recorded_args.extend(args)
        return subprocess.CompletedProcess(args=args, returncode=1, stdout="F841", stderr="")

    monkeypatch.setattr(check_suppressions.subprocess, "run", fake_run)

    result = check_suppressions._run_for_finding(
        finding,
        repo_root=repo_root,
        src_root=src_root,
    )

    assert result is not None
    assert recorded_args[:4] == [sys.executable, "-m", "ruff", "check"]


def test_find_suppressions_detects_ty_and_skylos_directives(tmp_path: Path) -> None:
    """Both ty and Skylos ignores are findings, even beside a noqa on the same line."""
    file_path = tmp_path / "sample.py"
    file_path.write_text(
        "def f(a, b):  # noqa: PLR0913 - shared why  # skylos: ignore[SKY-C303]\n"
        "    return a  # ty: ignore[invalid-return-type]  # stub is wrong\n"
        "x = 1  # skylos: ignore\n"
        "y = 2  # ty: ignore\n",
        encoding="utf-8",
    )

    findings = check_suppressions.find_suppressions(file_path)

    assert [(finding.line_num, finding.kind, finding.codes) for finding in findings] == [
        (1, "noqa", ("PLR0913",)),
        (1, "skylos-ignore", ("SKY-C303",)),
        (2, "ty-ignore", ("invalid-return-type",)),
        (3, "bare-skylos-ignore", ()),
        (4, "bare-ty-ignore", ()),
    ]
    # A Skylos ignore beside a justified noqa shares that justification; the
    # ty ignore carries its own.
    assert check_suppressions._suppression_rationale(findings[1]) == "shared why"
    assert check_suppressions._suppression_rationale(findings[2]) == "stub is wrong"
    for bare in findings[3:]:
        needed, reason = check_suppressions.check_if_needed(
            bare, repo_root=tmp_path, src_root=tmp_path
        )
        assert needed is False
        assert "not allowed" in reason


def test_skylos_audit_keeps_only_ignores_that_suppressed_a_finding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One scan, limited to the named rules; an ignore Skylos did not use is stale."""
    file_path = tmp_path / "sample.py"
    file_path.write_text(
        "fd = os.open(path)  # skylos: ignore[SKY-D215] canonicalized above\n"
        "chunk = os.read(fd, 10)  # skylos: ignore[SKY-P401] bounded read\n",
        encoding="utf-8",
    )
    findings = check_suppressions.find_suppressions(file_path)
    recorded: dict[str, object] = {}

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        recorded["args"] = args
        recorded["cwd"] = kwargs["cwd"]
        recorded["report"] = report = Path(args[args.index("-o") + 1])
        safe_io.write_text_no_follow(
            report,
            json.dumps(
                {
                    "suppressed": [
                        {"rule_id": "SKY-D215", "file": str(file_path), "line": 1},
                        {"file": str(file_path), "line": 2},
                    ]
                }
            ),
        )
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(check_suppressions, "_python_tool", lambda name: f"/bin/{name}")
    monkeypatch.setattr(check_suppressions.subprocess, "run", fake_run)

    outputs = check_suppressions._skylos_audit_outputs(findings, src_root=tmp_path)

    args = recorded["args"]
    assert isinstance(args, list)
    assert args[args.index("--select") + 1] == "SKY-D215,SKY-P401"
    assert args[args.index("--format") + 1] == "json-ci"
    assert recorded["cwd"] == tmp_path
    report_path = recorded["report"]
    assert isinstance(report_path, Path)
    assert not report_path.exists()
    verdicts = [
        check_suppressions.check_if_needed(
            finding,
            repo_root=tmp_path,
            src_root=tmp_path,
            audit_output=check_suppressions._batched_output_for(finding, {}, outputs),
        )
        for finding in findings
    ]
    assert verdicts == [
        (True, "Suppression needed: SKY-D215 violations found"),
        (False, "No violation found for: SKY-P401"),
    ]


def test_skylos_and_ty_audits_skip_when_the_tool_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_path = tmp_path / "sample.py"
    file_path.write_text(
        "x = f()  # skylos: ignore[SKY-D215] trusted literal\n"
        "y = g()  # ty: ignore[call-non-callable]  # stub is wrong\n",
        encoding="utf-8",
    )
    findings = check_suppressions.find_suppressions(file_path)
    monkeypatch.setattr(check_suppressions, "_python_tool", lambda _name: None)

    skylos_outputs = check_suppressions._skylos_audit_outputs(findings, src_root=tmp_path)

    assert skylos_outputs is None
    verdicts = [
        check_suppressions.check_if_needed(
            finding,
            repo_root=tmp_path,
            src_root=tmp_path,
            audit_output=check_suppressions._batched_output_for(finding, {}, skylos_outputs),
        )
        for finding in findings
    ]
    assert verdicts == [
        (True, "Skipped skylos audit because skylos is unavailable"),
        (True, "Skipped ty audit because ty is unavailable"),
    ]


def test_run_for_finding_reruns_ty_without_only_the_ty_ignore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ty variant keeps the line's type: ignore, so mypy's suppression is not counted."""
    file_path = tmp_path / "sample.py"
    line = "    def encoding(self) -> str:  # type: ignore[override]  # ty: ignore[bad]  # why"
    file_path.write_text(f"class A:\n{line}\n        return ''\n", encoding="utf-8")
    finding = check_suppressions.find_suppressions(file_path)[-1]
    assert finding.kind == "ty-ignore"
    seen: dict[str, object] = {}

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen["args"] = args
        seen["cwd"] = kwargs["cwd"]
        seen["line"] = safe_io.read_text_no_follow(Path(args[-1])).splitlines()[1]
        return subprocess.CompletedProcess(
            args=args, returncode=1, stdout="x.py:2:9: error[bad] message", stderr=""
        )

    monkeypatch.setattr(check_suppressions, "_python_tool", lambda name: f"/bin/{name}")
    monkeypatch.setattr(check_suppressions.subprocess, "run", fake_run)

    needed, reason = check_suppressions.check_if_needed(
        finding, repo_root=tmp_path, src_root=tmp_path
    )

    assert (needed, reason) == (True, "Suppression needed: bad violations found")
    args = seen["args"]
    assert isinstance(args, list)
    assert args[:4] == ["/bin/ty", "check", "--python", sys.executable]
    assert seen["cwd"] == tmp_path
    assert seen["line"] == "    def encoding(self) -> str:  # type: ignore[override]  # why"


def test_claude_instructions_are_a_link_to_agents_instructions() -> None:
    """One instruction file: CLAUDE.md is a symlink to AGENTS.md, never a mirrored copy."""
    claude = REPO_ROOT / "CLAUDE.md"
    assert claude.is_symlink(), "CLAUDE.md must be a symlink to AGENTS.md"
    assert claude.resolve() == (REPO_ROOT / "AGENTS.md").resolve()


def test_precommit_framework_config_installs_both_hooks() -> None:
    """The checked-in pre-commit config is the single hook mechanism."""
    config = yaml.safe_load((REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"))

    assert tuple(config["default_install_hook_types"]) == validate_env.REQUIRED_HOOK_TYPES
    entries = [hook["entry"] for repo in config["repos"] for hook in repo["hooks"]]
    for entry in entries:
        script = entry.removeprefix("bash ").split()[0]
        assert (REPO_ROOT / script).is_file(), entry
    assert any("run_commit_hygiene.sh" in entry for entry in entries)
    assert any("check_quality_simple.sh" in entry for entry in entries)


def test_python_floor_is_single_sourced() -> None:
    """Every encoding of the minimum Python version must agree with pyproject."""
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    requires_python = pyproject["project"]["requires-python"]
    assert requires_python.startswith(">=")
    floor = requires_python.removeprefix(">=").strip()

    assert pyproject["tool"]["mypy"]["python_version"] == floor
    assert pyproject["tool"]["pyright"]["pythonVersion"] == floor
    assert pyproject["tool"]["pyrefly"]["python-version"] == floor

    assert ".".join(str(part) for part in validate_env.REQUIRED_PYTHON_VERSION) == floor

    # Fresh envs are created on the working version, which may sit above the
    # floor; CI must test it (checked against the workflow versions below).
    setup_script = (PKG_ROOT / "tools" / "setup_conda_env.sh").read_text(encoding="utf-8")
    env_match = re.search(r'^ENV_PYTHON_VERSION="(3\.\d{1,2})"$', setup_script, re.MULTILINE)
    assert env_match is not None
    env_version = env_match.group(1)

    # CI runs the floor plus any newer candidate (a staged Python move is
    # rehearsed in CI before the working env follows). Every literal version
    # in the workflows must be >= the floor, and the floor itself must still be
    # exercised somewhere so the supported minimum stays tested.
    floor_parts = tuple(int(part) for part in floor.split("."))
    ci_versions: set[str] = set()
    for workflow_path in sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml")):
        workflow_text = workflow_path.read_text(encoding="utf-8")
        for match in re.finditer(r'"(3\.\d{1,2})"', workflow_text):
            ci_versions.add(match.group(1))
    assert env_version in ci_versions, f"CI does not test the fresh-env Python {env_version}"
    assert floor in ci_versions, f"CI no longer tests the Python floor {floor}: {ci_versions}"
    for version in ci_versions:
        assert tuple(int(part) for part in version.split(".")) >= floor_parts, version


def test_ci_next_python_lane_tracks_the_probe_and_skips_only_its_own_install_failure() -> None:
    """The static-quality lane for the next Python mirrors probe_python_next.sh.

    Its version is the probe's default and newer than the fresh-env Python,
    and its checks run only once the install step reports the stack
    available (which install failures may skip is tested separately).
    """
    probe_script = (PKG_ROOT / "tools" / "probe_python_next.sh").read_text(encoding="utf-8")
    probe_match = re.search(
        r'^PROBE_PYTHON="\$\{PROBE_PYTHON:-(3\.\d{1,2})\}"$', probe_script, re.MULTILINE
    )
    assert probe_match is not None
    setup_script = (PKG_ROOT / "tools" / "setup_conda_env.sh").read_text(encoding="utf-8")
    env_match = re.search(r'^ENV_PYTHON_VERSION="(3\.\d{1,2})"$', setup_script, re.MULTILINE)
    assert env_match is not None

    workflow = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "quality.yml").read_text(encoding="utf-8")
    )
    job = workflow["jobs"]["static-quality"]
    next_lanes = [
        entry["python-version"]
        for entry in job["strategy"]["matrix"].get("include", [])
        if entry.get("next-python") is True
    ]
    assert next_lanes == [probe_match.group(1)]
    assert tuple(map(int, next_lanes[0].split("."))) > tuple(
        map(int, env_match.group(1).split("."))
    )
    assert next_lanes[0] not in job["strategy"]["matrix"]["python-version"]

    steps = {step.get("name"): step for step in job["steps"]}
    setup = next(
        step for step in job["steps"] if step.get("uses", "").startswith("actions/setup-python@")
    )
    assert setup["with"]["allow-prereleases"] == "${{ matrix.next-python == true }}"
    install = steps["Install dependencies"]
    assert install["id"] == "install"
    assert install["env"]["NEXT_PYTHON"] == "${{ matrix.next-python == true }}"
    assert steps["Run static quality checks"]["if"] == "steps.install.outputs.available == 'true'"


_PIP_NO_MLX_DISTRIBUTION = (
    "ERROR: Could not find a version that satisfies the requirement mlx>=0.32.3 "
    "(from check-models) (from versions: none)\n"
    "ERROR: No matching distribution found for mlx>=0.32.3\n"
)
# Pip ends with the same "No matching distribution found" when it could not
# read the index at all; neither of these says the wheel is missing.
_PIP_CERTIFICATE_FAILURE = (
    "WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) "
    "after connection broken by 'SSLError(SSLCertVerificationError(1, '[SSL: "
    "CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer "
    "certificate (_ssl.c:1010)'))': /simple/mlx/\n"
    "Could not fetch URL https://pypi.org/simple/mlx/: There was a problem confirming the "
    "ssl certificate: HTTPSConnectionPool(host='pypi.org', port=443): Max retries exceeded "
    "with url: /simple/mlx/ - skipping\n" + _PIP_NO_MLX_DISTRIBUTION
)
_PIP_CONNECTION_FAILURE = (
    "WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) "
    "after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection."
    "HTTPSConnection object at 0x104>: Failed to establish a new connection: [Errno 8] "
    "nodename nor servname provided, or not known')': /simple/mlx/\n" + _PIP_NO_MLX_DISTRIBUTION
)
_PIP_BUILD_FAILURE = (
    "  error: subprocess-exited-with-error\n"
    "ERROR: Failed to build installable wheels for some pyproject.toml based projects "
    "(sentencepiece)\n"
)


def _run_ci_install_step(
    tmp_path: Path, next_python: str, pip_exit: int, pip_output: str, hold_until: str = ""
) -> tuple[subprocess.CompletedProcess[str], str, str]:
    """Run the workflow's real install script, as Actions does (bash -e), against fake tools.

    Returns the completed process, what it wrote to GITHUB_OUTPUT and the fake-tool call log.
    """
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "quality.yml").read_text(encoding="utf-8")
    )
    install = next(
        step
        for step in workflow["jobs"]["static-quality"]["steps"]
        if step.get("name") == "Install dependencies"
    )
    script = tmp_path / "install.sh"
    safe_io.write_text_no_follow(script, install["run"])

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "calls.log"
    fakes = {
        "python": 'if [[ "$1" == "-V" ]]; then echo "Python 3.15.0"; fi\n',
        "pip": (
            f'if [[ "$*" == *"src/.[dev]"* ]]; then printf "%s" "$FAKE_PIP_OUTPUT"; '
            f"exit {pip_exit}; fi\n"
        ),
        "npm": "",
    }
    for name, body in fakes.items():
        safe_io.write_text_no_follow(
            fake_bin / name,
            f'#!/bin/bash\necho "{name} $*" >> "{calls}"\n{body}',
            mode=0o755,
        )
    github_output = tmp_path / "github_output"
    safe_io.write_text_no_follow(github_output, "")

    result = subprocess.run(  # noqa: S603 - fixed /bin/bash runs the workflow step with fakes
        ["/bin/bash", "-e", str(script)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        cwd=tmp_path,
        env={
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "NEXT_PYTHON": next_python,
            "NEXT_PYTHON_HOLD_UNTIL": hold_until,
            "GITHUB_OUTPUT": str(github_output),
            "RUNNER_TEMP": str(tmp_path),
            "FAKE_PIP_OUTPUT": pip_output,
        },
    )
    return (
        result,
        safe_io.read_text_no_follow(github_output),
        safe_io.read_text_no_follow(calls),
    )


@pytest.mark.parametrize(
    ("next_python", "pip_exit", "pip_output", "expected"),
    [
        pytest.param(
            "true", 1, _PIP_NO_MLX_DISTRIBUTION, (0, "false"), id="next-missing-dist-skips"
        ),
        pytest.param("true", 1, _PIP_BUILD_FAILURE, (1, None), id="next-build-failure-fails"),
        pytest.param(
            "true", 1, _PIP_CERTIFICATE_FAILURE, (1, None), id="next-certificate-failure-fails"
        ),
        pytest.param(
            "true", 1, _PIP_CONNECTION_FAILURE, (1, None), id="next-connection-failure-fails"
        ),
        pytest.param("false", 1, _PIP_NO_MLX_DISTRIBUTION, (1, None), id="other-lane-never-skips"),
        pytest.param(
            "true", 0, "Successfully installed\n", (0, "true"), id="next-installs-and-runs"
        ),
        pytest.param(
            "false", 0, "Successfully installed\n", (0, "true"), id="other-installs-and-runs"
        ),
    ],
)
def test_ci_install_step_skips_only_a_missing_distribution_on_the_next_python_lane(
    tmp_path: Path,
    next_python: str,
    pip_exit: int,
    pip_output: str,
    expected: tuple[int, str | None],
) -> None:
    """Only a missing distribution on the next-Python lane may skip the checks.

    Not when pip could not read the index: a build, resolver, packaging or
    index failure fails the lane, as on every other.
    """
    returncode, available = expected
    result, outputs, calls = _run_ci_install_step(tmp_path, next_python, pip_exit, pip_output)
    assert result.returncode == returncode, result.stdout + result.stderr
    npm_ran = "npm install" in calls
    if available is None:
        assert outputs == ""
        assert not npm_ran
    else:
        assert outputs == f"available={available}\n"
        assert npm_ran is (available == "true")
    if available == "false":
        assert "No distribution for Python 3.15.0 yet: mlx>=0.32.3." in result.stdout


@pytest.mark.parametrize(
    ("next_python", "hold_until", "held"),
    [
        pytest.param("true", "9999-12-31", True, id="next-lane-held"),
        pytest.param("true", "2000-01-01", False, id="next-lane-hold-over"),
        pytest.param("false", "9999-12-31", False, id="other-lane-never-held"),
    ],
)
def test_ci_install_step_holds_the_next_python_lane_until_its_date(
    tmp_path: Path, next_python: str, hold_until: str, *, held: bool
) -> None:
    """Before NEXT_PYTHON_HOLD_UNTIL the next-Python lane skips without installing.

    From that date it installs and runs again with no edit; other lanes
    ignore the hold.
    """
    result, outputs, calls = _run_ci_install_step(
        tmp_path, next_python, 0, "Successfully installed\n", hold_until
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert outputs == f"available={'false' if held else 'true'}\n"
    assert ("pip install" in calls) is not held
    if held:
        assert "checks are held until 9999-12-31." in result.stdout


def test_ci_next_python_hold_is_a_date() -> None:
    """The workflow's hold value is empty or an ISO date, so the string comparison holds."""
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "quality.yml").read_text(encoding="utf-8")
    )
    install = next(
        step
        for step in workflow["jobs"]["static-quality"]["steps"]
        if step.get("name") == "Install dependencies"
    )
    hold = install["env"]["NEXT_PYTHON_HOLD_UNTIL"]
    assert hold == "" or dt.date.fromisoformat(hold).isoformat() == hold


_SUBPROCESS_TEXT_CALLS = frozenset({"run", "check_output", "Popen", "call", "check_call"})
# Receivers whose ``.open`` is not a text-file open (no encoding parameter).
_NON_TEXT_OPEN_RECEIVERS = frozenset({"Image", "os", "webbrowser"})


def _opens_a_text_file_without_encoding(call: ast.Call) -> bool:
    """Whether an ``open``/``io.open``/``Path.open`` call opens a text file without an encoding."""
    func = call.func
    receiver = (
        func.value.id
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
        else None
    )
    if receiver in _NON_TEXT_OPEN_RECEIVERS:
        return False
    # Builtin/io open(file, mode, buffering, encoding); Path.open(mode, buffering, encoding).
    mode_index = 1 if isinstance(func, ast.Name) or receiver == "io" else 0
    if len(call.args) > mode_index + 2:
        return False  # encoding passed positionally
    mode = next(
        (keyword.value for keyword in call.keywords if keyword.arg == "mode"),
        call.args[mode_index] if len(call.args) > mode_index else None,
    )
    is_binary = isinstance(mode, ast.Constant) and isinstance(mode.value, str) and "b" in mode.value
    return not is_binary


def _locale_dependent_text_io(call: ast.Call) -> str | None:
    """Describe a text-I/O call that leaves its encoding to the locale, else None.

    Python 3.15 (PEP 686) decodes such calls as UTF-8 whatever the locale, so
    naming the encoding keeps 3.13, 3.14 and 3.15 byte-for-byte alike.
    """
    func = call.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
    keywords = {keyword.arg for keyword in call.keywords}
    if "encoding" in keywords or None in keywords:  # explicit, or **kwargs we cannot see
        return None
    if name in _SUBPROCESS_TEXT_CALLS and keywords & {"text", "universal_newlines"}:
        description = f"subprocess.{name}(text=True)"
    elif name == "open" and _opens_a_text_file_without_encoding(call):
        description = "open() in text mode"
    elif (name == "read_text" and not call.args) or (name == "write_text" and len(call.args) < 2):
        description = f"{name}()"
    else:
        description = None
    return description


def test_shipped_text_io_names_its_encoding() -> None:
    """Shipped code must not depend on the locale's encoding (Python 3.15, PEP 686).

    3.15 makes UTF-8 the default for text I/O without an ``encoding``; on 3.13
    and 3.14 the locale decides. Ruff's PLW1514 only covers files and is a
    preview rule (preview is off here), and nothing lints ``subprocess(...,
    text=True)``. Tests are exempt: they read back what they wrote in one
    process, so a locale cannot make them disagree with themselves.
    """
    sources = [PKG_ROOT / "check_models.py", *sorted((PKG_ROOT / "tools").glob("*.py"))]
    offenders = [
        f"{path.relative_to(PKG_ROOT)}:{node.lineno}: {description}"
        for path in sources
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call)
        and (description := _locale_dependent_text_io(node)) is not None
    ]
    assert not offenders, "pass encoding= explicitly:\n" + "\n".join(offenders)


def test_locale_dependent_text_io_detector_recognises_each_form() -> None:
    """The detector behind the shipped-code check must flag and clear the forms it claims to."""

    def describe(source: str) -> str | None:
        call = ast.parse(source, mode="eval").body
        assert isinstance(call, ast.Call)
        return _locale_dependent_text_io(call)

    flagged = [
        "subprocess.run(cmd, capture_output=True, text=True)",
        "subprocess.check_output(cmd, universal_newlines=True)",
        "open(path)",
        "open(path, 'w')",
        "path.open()",
        "path.open(mode)",
        "path.read_text()",
        "path.write_text(data)",
    ]
    cleared = [
        "subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')",
        "subprocess.run(cmd, capture_output=True)",
        "subprocess.run(cmd, **kwargs)",
        "open(path, 'rb')",
        "open(path, 'r', -1, 'utf-8')",
        "path.open('rb')",
        "Image.open(path)",
        "os.open(path, flags)",
        "path.read_text(encoding='utf-8')",
        "path.read_text('utf-8')",
        "path.write_text(data, 'utf-8')",
    ]
    assert [source for source in flagged if describe(source) is None] == []
    assert [source for source in cleared if describe(source) is not None] == []


def test_update_smoke_defaults_are_documented() -> None:
    """The smoke model and expected output are documented once, in the README table."""
    update_script = (PKG_ROOT / "tools" / "update.sh").read_text(encoding="utf-8")
    model_match = re.search(r"MLX_LOCAL_BUILD_SMOKE_MODEL:-([^}]+)\}", update_script)
    expected_match = re.search(r"MLX_LOCAL_BUILD_SMOKE_EXPECTED:-([^}]+)\}", update_script)
    assert model_match is not None
    assert expected_match is not None
    model = model_match.group(1)
    expected = expected_match.group(1)

    package_readme = (PKG_ROOT / "README.md").read_text(encoding="utf-8")
    assert model in package_readme, "smoke model not documented in src/README.md"
    assert expected in package_readme, "smoke expected output not documented in src/README.md"


def test_section_banners_match_copilot_instructions() -> None:
    """The monolith's SECTION landmarks and the documented map must stay identical."""
    source = (PKG_ROOT / "check_models.py").read_text(encoding="utf-8")
    banners = re.findall(r"^# SECTION: (.+)$", source, re.MULTILINE)
    # Only the §3 table rows define the map; prose mentions elsewhere are free.
    documented = re.findall(
        r"^\|.*?`SECTION: ([^`]+)`",
        COPILOT_INSTRUCTIONS.read_text(encoding="utf-8"),
        re.MULTILINE,
    )

    assert banners, "no SECTION banners found in check_models.py"
    assert banners == documented, (
        "SECTION banners and copilot-instructions §3 have drifted (order matters)"
    )


def test_no_ci_only_mlx_core_stub_generation() -> None:
    """mlx.core stubs come from the mlx distribution itself; nothing must regenerate them.

    Both the PyPI wheel and an editable build ship ``mlx/core/*.pyi`` generated at
    build time by mlx's own ``nanobind_add_stub``. A recursive
    ``python -m nanobind.stubgen -m mlx.core -r`` run emits submodule stubs without
    imports (hundreds of mypy name-defined errors), so the retired CI step and the
    nanobind dev dependency must not come back.
    """
    quality_workflow = (REPO_ROOT / ".github" / "workflows" / "quality.yml").read_text(
        encoding="utf-8"
    )
    pyproject_text = (REPO_ROOT / "src" / "pyproject.toml").read_text(encoding="utf-8")

    for label, text in (
        ("quality.yml", quality_workflow),
        ("pyproject.toml", pyproject_text),
    ):
        assert "nanobind" not in text, f"{label} reintroduces nanobind stub generation"


def test_make_format_covers_every_gate_formatted_directory() -> None:
    """`make format` must reach every directory the full gate's `ruff format .` reaches."""
    makefile = (PKG_ROOT / "Makefile").read_text(encoding="utf-8")
    fmt_match = re.search(r"^FMT_PATHS := (.+)$", makefile, re.MULTILINE)
    assert fmt_match is not None
    fmt_paths = set(fmt_match.group(1).split())

    expected_dirs = {
        path.name
        for path in PKG_ROOT.iterdir()
        if path.is_dir()
        and not path.name.startswith(".")
        and path.name not in {"node_modules", "output", "build", "dist", "check_models.egg-info"}
        and any(path.glob("*.py"))
    }
    assert expected_dirs | {"check_models.py"} <= fmt_paths | expected_dirs & fmt_paths | {
        "check_models.py"
    }
    assert expected_dirs <= fmt_paths, (
        f"make format misses {expected_dirs - fmt_paths}; the gate's `ruff format .` "
        "would still reformat them"
    )


def test_quality_script_globs_workflow_yaml_files() -> None:
    """Workflow YAML validation must enumerate by glob, not by hardcoded list."""
    quality_script = (PKG_ROOT / "tools" / "run_quality_checks.sh").read_text(encoding="utf-8")

    assert 'find "$(quality_repo_root)/.github/workflows"' in quality_script
    assert ".github/workflows/quality.yml" not in quality_script


def test_danger_report_filter_drops_only_worktree_findings(tmp_path: Path) -> None:
    """The Skylos danger post-filter must drop .worktrees findings and keep the rest."""
    report: dict[str, object] = {
        "danger": [
            {"file": "/repo/.worktrees/mlx-vlm-x/.github/workflows/tests.yml", "rule": "D292"},
            {"file": "/repo/.claude/worktrees/agent-x/src/check_models.py", "rule": "D215"},
            {"file": "/repo/src/tools/update.sh", "rule": "D301"},
            "not-a-dict-entry",
        ],
        "grade": {"overall": {"score": 54}},
    }
    dropped = filter_danger_report.drop_worktree_findings(report)

    assert dropped == 2
    assert report["danger"] == [
        {"file": "/repo/src/tools/update.sh", "rule": "D301"},
        "not-a-dict-entry",
    ]

    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"danger": []}), encoding="utf-8")
    assert filter_danger_report.main(["prog", str(report_path)]) == 0


_BRACES = check_dependency_advisories.AcceptedAdvisory(
    rule_id="SKY-SCA-GHSA-test",
    ecosystem="npm",
    package="braces",
    version="3.0.3",
    manifest="package-lock.json",
    reason="reviewed: no fixed release",
)


def _sca_coverage(status: object = "complete", **overrides: object) -> dict[str, object]:
    """A finished Skylos sca_coverage block (shape of skylos 4.43 reports)."""
    return {
        "status": status,
        "complete": True,
        "query": {"status": "complete", "complete": True},
        "dependency_count": 87,
        "parse_error_count": 0,
        "unresolved_lockfile_dependency_count": 0,
        "unresolved_dependency_count": 0,
        **overrides,
    }


def _sca_finding(
    rule_id: str = "SKY-SCA-GHSA-test",
    *,
    package: str = "braces",
    version: str = "3.0.3",
    file: str = "package-lock.json",
    fixed_versions: list[str] | None = None,
    also_in: tuple[str, ...] = (),
) -> dict[str, object]:
    """One Skylos SCA finding; ``also_in`` adds merged occurrences in other lockfiles."""
    occurrences = [{"file": manifest, "line": 117} for manifest in (file, *also_in)]
    return {
        "rule_id": rule_id,
        "file": file,
        "line": 117,
        "message": f"{package}@{version}: advisory",
        "metadata": {
            "ecosystem": "npm",
            "package_name": package,
            "package_version": version,
            "fixed_version": None,
            "fixed_versions": fixed_versions or [],
            "dependency_occurrences": occurrences,
        },
    }


def _dependency_scan_report(coverage: object, *findings: dict[str, object]) -> dict[str, object]:
    return {
        "analysis_summary": {"sca_coverage": coverage},
        "dependency_vulnerabilities": list(findings),
    }


def _evaluate_scan(tmp_path: Path, report: object) -> tuple[bool, list[str]]:
    result: tuple[bool, list[str]] = check_dependency_advisories.evaluate(
        report, (_BRACES,), scan_root=tmp_path
    )
    return result


def test_dependency_advisory_gate_accepts_only_the_reviewed_occurrence(tmp_path: Path) -> None:
    """An exception covers its advisory on the reviewed package, version and manifest only."""
    passed, lines = _evaluate_scan(
        tmp_path, _dependency_scan_report(_sca_coverage(), _sca_finding())
    )
    assert passed
    assert any("Accepted SKY-SCA-GHSA-test" in line for line in lines)
    # Skylos may report the manifest as an absolute path under the scanned directory.
    absolute = _sca_finding(file=str(tmp_path / "package-lock.json"))
    assert _evaluate_scan(tmp_path, _dependency_scan_report(_sca_coverage(), absolute))[0]

    for other in (
        _sca_finding("SKY-SCA-GHSA-other"),
        _sca_finding(version="3.0.4"),
        _sca_finding(package="micromatch"),
        _sca_finding(file="tools/vendor/package-lock.json"),
        # Skylos merges the same package version from several lockfiles into one
        # finding: every merged occurrence must be the reviewed manifest.
        _sca_finding(also_in=("service/package-lock.json",)),
    ):
        passed, lines = _evaluate_scan(tmp_path, _dependency_scan_report(_sca_coverage(), other))
        assert not passed, other
        assert not any("Accepted" in line for line in lines), other
        assert any(line.startswith("❌ ") for line in lines)

    # A listed advisory that is no longer reported passes, with a prompt to drop it.
    passed, lines = _evaluate_scan(tmp_path, _dependency_scan_report(_sca_coverage()))
    assert passed
    assert any("no longer reported" in line for line in lines)


def test_dependency_advisory_gate_fails_an_exception_once_a_fix_ships(tmp_path: Path) -> None:
    """A fixed release turns the accepted advisory back into a blocking one."""
    report = _dependency_scan_report(_sca_coverage(), _sca_finding(fixed_versions=["3.0.4"]))
    passed, lines = _evaluate_scan(tmp_path, report)
    assert not passed
    assert any("fixed release now exists (3.0.4)" in line for line in lines)


def test_dependency_advisory_gate_reads_the_report_file(tmp_path: Path) -> None:
    report_path = tmp_path / "skylos-dependency-scan.json"
    report_path.write_text(
        json.dumps(_dependency_scan_report(_sca_coverage(), _sca_finding("SKY-SCA-GHSA-new"))),
        encoding="utf-8",
    )
    assert check_dependency_advisories.main(["prog", str(report_path)]) == 1
    assert check_dependency_advisories.main(["prog", str(tmp_path / "missing.json")]) == 1


@pytest.mark.parametrize(
    "coverage",
    [
        pytest.param(_sca_coverage("incomplete", complete=False), id="incomplete"),
        pytest.param(_sca_coverage("unavailable", complete=False), id="unavailable"),
        pytest.param(_sca_coverage("unknown", complete=False), id="unknown"),
        # Skylos's real status for a tree it found nothing to scan in.
        pytest.param(
            _sca_coverage("no_supported_manifests", complete=False, dependency_count=0),
            id="no-supported-manifests",
        ),
        pytest.param(_sca_coverage("some_future_status"), id="unrecognised-status"),
        pytest.param(_sca_coverage(None), id="missing-status"),
        pytest.param(_sca_coverage(complete=False), id="complete-flag-false"),
        pytest.param(_sca_coverage(query={"complete": False}), id="query-incomplete"),
        pytest.param(_sca_coverage(dependency_count=0), id="nothing-checked"),
        pytest.param(_sca_coverage(parse_error_count=1), id="parse-error"),
        pytest.param(
            _sca_coverage(
                "complete_with_unresolved_versions", unresolved_lockfile_dependency_count=2
            ),
            id="unresolved-lockfile",
        ),
        pytest.param(None, id="no-coverage-block"),
    ],
)
def test_dependency_advisory_gate_passes_only_a_finished_scan(
    tmp_path: Path, coverage: object
) -> None:
    """Only a recognised successful status passes; anything else fails, even with no findings."""
    passed, lines = _evaluate_scan(tmp_path, _dependency_scan_report(coverage))

    assert not passed
    assert "did not finish cleanly" in lines[0]


def test_dependency_advisory_gate_accepts_unresolved_pyproject_ranges(tmp_path: Path) -> None:
    """Unpinned pyproject ranges are accepted by policy and named in the output."""
    coverage = _sca_coverage("complete_with_unresolved_versions", unresolved_dependency_count=12)
    passed, lines = _evaluate_scan(tmp_path, _dependency_scan_report(coverage, _sca_finding()))

    assert passed
    assert any("12 declared version ranges" in line for line in lines)


def test_artifact_schema_version_constants_match_typed_dict_literals() -> None:
    """Writers/readers share version constants that must match the TypedDict Literals."""
    jsonl_literal = typing.get_args(
        typing.get_type_hints(check_models.JsonlMetadataRecord)["format_version"]
    )
    history_literal = typing.get_args(
        typing.get_type_hints(check_models.HistoryRunRecord)["format_version"]
    )

    assert jsonl_literal == (check_models.JSONL_FORMAT_VERSION,)
    assert history_literal == (check_models.HISTORY_FORMAT_VERSION,)


def test_check_if_needed_rejects_suppressions_without_justification(tmp_path: Path) -> None:
    """A suppression must say why it is safe, not only what it silences."""
    repo_root = tmp_path / "repo"
    src_root = repo_root / "src"
    src_root.mkdir(parents=True)
    file_path = src_root / "sample.py"
    file_path.write_text("unused = 1  # noqa: F841\n", encoding="utf-8")
    finding = check_suppressions.SuppressionFinding(
        file_path=file_path,
        line_num=1,
        kind="noqa",
        codes=("F841",),
        line_text="unused = 1  # noqa: F841",
    )

    needed, reason = check_suppressions.check_if_needed(
        finding,
        repo_root=repo_root,
        src_root=src_root,
    )

    assert needed is False
    assert "no justification" in reason
    # The rationale extractor recognises both comment styles.
    justified_noqa = check_suppressions.SuppressionFinding(
        file_path=file_path,
        line_num=1,
        kind="noqa",
        codes=("F841",),
        line_text="unused = 1  # noqa: F841 - fixture keeps the name visible",
    )
    justified_ignore = check_suppressions.SuppressionFinding(
        file_path=file_path,
        line_num=1,
        kind="type-ignore",
        codes=("assignment",),
        line_text="x = y  # type: ignore[assignment]  # narrowing is deliberate",
    )
    assert check_suppressions._suppression_rationale(justified_noqa)
    assert check_suppressions._suppression_rationale(justified_ignore)


def _run_update_pip_wrapper_harness(driver: str, tmp_path: Path) -> str:
    """Run update.sh's pin/wrapper helpers against a fake pip; return its arg log.

    The helpers are extracted verbatim from the script (the contiguous block
    from the pin state through ``pip_install_verbose``) so the test exercises
    the shipped code, not a copy.
    """
    script = (PKG_ROOT / "tools" / "update.sh").read_text(encoding="utf-8")
    start = script.index('LOCAL_MLX_CONSTRAINT_FILE=""')
    end = script.index("# Helper function: pip install for build/infrastructure tools only.")
    helpers = script[start:end]

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    pip_log = tmp_path / "pip-args.log"
    fake_pip = fake_bin / "pip"
    safe_io.write_text_no_follow(
        fake_pip,
        "#!/bin/bash\n"
        f'echo "ARGS: $*" >> "{pip_log}"\n'
        f'echo "CALLER_PIP_CONSTRAINT: ${{PIP_CONSTRAINT:-unset}}" >> "{pip_log}"\n',
        mode=0o755,
    )

    completed = subprocess.run(  # noqa: S603 - test harness, fixed bash, temp files
        ["/bin/bash", "-c", helpers + "\n" + driver],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"},
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    log_text = safe_io.read_text_no_follow(pip_log) if pip_log.exists() else ""
    return log_text + completed.stdout


def test_update_pip_wrappers_normal_and_forced_without_pin(tmp_path: Path) -> None:
    """Without a pin, wrappers pass eager upgrades and honour FORCE_REINSTALL."""
    log = _run_update_pip_wrapper_harness("pip_install somepkg", tmp_path)
    assert "--constraint" not in log
    assert "--force-reinstall" not in log

    log = _run_update_pip_wrapper_harness("FORCE_REINSTALL=1\npip_install somepkg", tmp_path)
    assert "--force-reinstall" in log
    assert "--constraint" not in log


def test_update_pip_wrappers_pin_constrains_and_suppresses_force(tmp_path: Path) -> None:
    """The mlx pin adds a private --constraint and outranks FORCE_REINSTALL.

    Regression: FORCE_REINSTALL plus the pin previously produced
    ResolutionImpossible (the exact local dev version cannot come from PyPI);
    local-source preservation must win, with the suppression logged.
    """
    driver = """
get_installed_distribution_version() { echo "0.32.2.dev20260825+abc123"; }
pin_local_mlx_build
pip_install somepkg
FORCE_REINSTALL=1
pip_install otherpkg
"""
    log = _run_update_pip_wrapper_harness(driver, tmp_path)
    assert "--constraint" in log
    assert "--force-reinstall" not in log
    assert "FORCE_REINSTALL suppressed" in log
    assert "Pinned local mlx 0.32.2.dev20260825+abc123" in log


def test_update_pin_failure_is_fatal_when_version_unreadable(tmp_path: Path) -> None:
    """An unpinnable local build must stop the run, not continue unprotected."""
    driver = """
get_installed_distribution_version() { echo ""; }
if pin_local_mlx_build; then
    echo "PIN_RC: 0"
else
    echo "PIN_RC: 1"
fi
"""
    log = _run_update_pip_wrapper_harness(driver, tmp_path)
    assert "PIN_RC: 1" in log
    assert "refusing to continue" in log


def test_update_pip_wrappers_preserve_caller_constraint_and_clean_up(tmp_path: Path) -> None:
    """Caller PIP_CONSTRAINT passes through untouched; the pin file is removed on exit."""
    caller_file = tmp_path / "caller-constraints.txt"
    caller_file.write_text("requests==2.0.0\n", encoding="utf-8")
    driver = f"""
export PIP_CONSTRAINT="{caller_file}"
get_installed_distribution_version() {{ echo "0.32.2.dev20260825+abc123"; }}
pin_local_mlx_build
pip_install somepkg
echo "PINFILE: $LOCAL_MLX_CONSTRAINT_FILE"
"""
    log = _run_update_pip_wrapper_harness(driver, tmp_path)
    assert f"CALLER_PIP_CONSTRAINT: {caller_file}" in log
    pin_file = next(
        line.split("PINFILE: ", 1)[1].strip()
        for line in log.splitlines()
        if line.startswith("PINFILE: ")
    )
    assert pin_file
    assert not Path(pin_file).exists(), "EXIT trap must remove the pin file"
    # And the script must never export global pip state for the pin.
    script = (PKG_ROOT / "tools" / "update.sh").read_text(encoding="utf-8")
    assert "export PIP_CONSTRAINT" not in script


def test_pip_show_helpers_survive_set_e_pipefail_with_chatty_pip(tmp_path: Path) -> None:
    """The helpers must not die under `set -euo pipefail` however chatty pip is.

    Regression: `pip show | awk '{...; exit}'` raced SIGPIPE — awk quit at the
    matched line, pip took EPIPE, and once the updater ran under live `set -e`
    the failing command substitution in `pin_local_mlx_build` killed the whole
    script with no output, right after "mlx installed successfully".
    """
    script = (PKG_ROOT / "tools" / "update.sh").read_text(encoding="utf-8")
    helper_names = (
        "pip_show_field",
        "get_editable_project_location",
        "get_installed_distribution_version",
    )
    helper_parts: list[str] = []
    for name in helper_names:
        match = re.search(rf"^{name}\(\) \{{\n.*?^\}}\n", script, re.MULTILINE | re.DOTALL)
        assert match is not None, f"helper {name} missing from update.sh"
        helper_parts.append(match.group(0))
    helper_block = "".join(helper_parts)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    lines = "\n".join(f"Filler-{i}: value" for i in range(4000))
    safe_io.write_text_no_follow(
        fake_python,
        "#!/bin/bash\n"
        'echo "Name: mlx"\n'
        'echo "Version: 0.32.2.dev1+abc"\n'
        f'cat <<"BULK"\n{lines}\nBULK\n'
        'echo "Editable project location: /tmp/mlx"\n',
        mode=0o755,
    )
    driver = (
        "set -euo pipefail\n"
        + helper_block
        + '\nversion="$(get_installed_distribution_version mlx)"\n'
        + 'location="$(get_editable_project_location mlx)"\n'
        + 'echo "VERSION: $version"\n'
        + 'echo "LOCATION: $location"\n'
    )
    completed = subprocess.run(  # noqa: S603 - test harness, fixed bash, temp files
        ["/bin/bash", "-c", driver],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"},
    )
    assert completed.returncode == 0, completed.stderr
    assert "VERSION: 0.32.2.dev1+abc" in completed.stdout
    assert "LOCATION: /tmp/mlx" in completed.stdout


def _skill_frontmatter(text: str) -> dict[str, object]:
    """Return a SKILL.md's YAML frontmatter as a mapping."""
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert match is not None, "SKILL.md must start with a frontmatter block"
    frontmatter = yaml.safe_load(match.group(1))
    assert isinstance(frontmatter, dict), "SKILL.md frontmatter must be a mapping"
    return frontmatter


def test_agent_skills_are_well_formed_and_listed() -> None:
    """Adapted from upstream mlx-vlm's ``skills/scripts/validate_skills.py``.

    Every skill: frontmatter ``name`` equal to its directory, a real trigger
    sentence as ``description``, a row in the Copilot skills table, and no
    ``uv`` commands in its code fences (this repo is conda + pip). Upstream
    skills cited as sources must exist upstream.
    """
    skill_dirs = sorted(path for path in AGENT_SKILLS_DIR.iterdir() if path.is_dir())
    assert skill_dirs, "no skills found"
    copilot_text = COPILOT_INSTRUCTIONS.read_text(encoding="utf-8")
    for skill_dir in skill_dirs:
        text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        frontmatter = _skill_frontmatter(text)
        assert frontmatter.get("name") == skill_dir.name, skill_dir.name
        assert len(str(frontmatter.get("description", ""))) >= 40, skill_dir.name
        assert f"| `{skill_dir.name}` |" in copilot_text, f"{skill_dir.name} missing from table"
        for fence in re.findall(r"```[a-z]*\n(.*?)```", text, re.DOTALL):
            for line in fence.splitlines():
                assert not line.strip().startswith("uv "), f"{skill_dir.name}: uv command"
        for cited in re.findall(r"skills/skills/([a-z0-9-]+)", text):
            assert cited in UPSTREAM_MLX_VLM_SKILLS, f"{skill_dir.name} cites unknown {cited}"
        for cited in re.findall(r"\.ai/skills/([a-z0-9-]+)", text):
            assert cited in UPSTREAM_TRANSFORMERS_SKILLS, f"{skill_dir.name} cites unknown {cited}"
    # Claude Code discovers project skills under .claude/skills; that path is a
    # committed link to the single source of truth, never a second copy.
    claude_skills = REPO_ROOT / ".claude" / "skills"
    assert claude_skills.is_symlink(), ".claude/skills must be a symlink, not a copy"
    assert claude_skills.resolve() == AGENT_SKILLS_DIR.resolve()


def test_typing_skill_does_not_cite_removed_stub_tooling() -> None:
    """Do not direct typing repairs into removed stub-generation infrastructure."""
    text = safe_io.read_text_no_follow(AGENT_SKILLS_DIR / "add-or-fix-type-checking/SKILL.md")
    assert "../typings" not in text
    assert "tools.generate_stubs" not in text


def test_benchmark_skill_isolates_retained_outputs() -> None:
    """Both sides retain separate artifacts, and B explicitly compares with A."""
    text = safe_io.read_text_no_follow(AGENT_SKILLS_DIR / "benchmarking-mlx-vlm/SKILL.md")
    commands = []
    for fence in re.findall(r"```bash\n(.*?)```", text, re.DOTALL):
        for line in fence.replace("\\\n", " ").splitlines():
            if "python -m check_models " in line:
                lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
                lexer.whitespace_split = True
                commands.append(list(lexer))
    assert len(commands) == 2
    outputs = [command[command.index("--output-dir") + 1] for command in commands]
    assert outputs[0] != outputs[1]
    assert all(output.startswith("$benchmark_dir/") for output in outputs)
    assert commands[0][commands[0].index("--compare-with") + 1] == "none"
    comparison = commands[1][commands[1].index("--compare-with") + 1]
    assert comparison == f"{outputs[0]}/results.jsonl"


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("cleanup", "answer\n12.0 0.5 stop\n"),
        ("missing", "<raw>answer</raw>\n12.0 0.5 stop\n"),
        ("nonstring", "<raw>answer</raw>\n12.0 0.5 stop\n"),
        ("empty", "\nNo chunks returned; no final-chunk metrics.\n"),
    ],
)
def test_native_repro_skill_stream_example(tmp_path: Path, case: str, expected: str) -> None:
    """Execute the documented probe with fake upstream modules, never MLX or weights."""
    text = safe_io.read_text_no_follow(AGENT_SKILLS_DIR / "native-mlx-vlm-repro/SKILL.md")
    example = next(
        fence
        for fence in re.findall(r"```python\n(.*?)```", text, re.DOTALL)
        if "from mlx_vlm.generate import stream_generate" in fence
    )
    setup = dedent(
        """\
        import sys
        from types import ModuleType, SimpleNamespace

        case = sys.argv[1]
        processor = SimpleNamespace()
        if case == "cleanup":
            processor.clean_output = lambda text: text.replace("<raw>", "").replace("</raw>", "")
        elif case == "nonstring":
            processor.clean_output = lambda text: None
        elif case == "empty":
            def unexpected_cleanup(text):
                raise AssertionError("An empty stream must not invoke output cleanup")
            processor.clean_output = unexpected_cleanup
        chunks = [] if case == "empty" else [
            SimpleNamespace(text="ignored draft", is_draft=True),
            SimpleNamespace(text="<raw>answer", is_draft=False),
            SimpleNamespace(text="</raw>", generation_tps=12.0, peak_memory=0.5, finish_reason="stop"),
        ]
        for name in ("mlx_vlm", "mlx_vlm.generate", "mlx_vlm.prompt_utils", "mlx_vlm.utils"):
            sys.modules[name] = ModuleType(name)
        sys.modules["mlx_vlm.generate"].stream_generate = lambda **kwargs: iter(chunks)
        sys.modules["mlx_vlm.prompt_utils"].apply_chat_template = lambda *a, **kw: "formatted"
        sys.modules["mlx_vlm.utils"].load = lambda *a, **kw: (SimpleNamespace(config={}), processor)
        MODEL, PROMPT, IMAGE = "fake-model", "prompt", "fake-image"
        LOAD_KWARGS, TEMPLATE_KWARGS, GENERATE_KWARGS, EOS_TOKENS = {}, {}, {}, []
        """
    )
    completed = subprocess.run(  # noqa: S603 - repo-owned example, fake upstream, no model loading
        [sys.executable, "-P", "-c", setup + "\n" + example, case],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == expected


def test_batched_noqa_audit_slices_ruff_output_per_finding(tmp_path: Path) -> None:
    """One ruff run over every noqa variant returns each finding only its own lines."""
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    safe_io.write_text_no_follow(first, "import os  # noqa: F401 - kept for the audit\n")
    safe_io.write_text_no_follow(
        second, "value = 1\nother = missing_name  # noqa: F821 - kept for the audit\n"
    )
    findings = [
        *check_suppressions.find_suppressions(first),
        *check_suppressions.find_suppressions(second),
    ]
    assert [finding.kind for finding in findings] == ["noqa", "noqa"]

    outputs = check_suppressions._batch_noqa_audit_outputs(findings, src_root=tmp_path)

    assert set(outputs) == {(first, 1), (second, 2)}
    assert "F401" in outputs[first, 1]
    assert "F821" in outputs[second, 2]
    assert "F401" not in outputs[second, 2]
    assert "second" not in outputs[first, 1]
    assert "first" not in outputs[second, 2]
    assert list(tmp_path.glob("*.suppression-audit.*")) == []
    for finding in findings:
        needed, reason = check_suppressions.check_if_needed(
            finding,
            repo_root=tmp_path,
            src_root=tmp_path,
            audit_output=outputs[finding.file_path, finding.line_num],
        )
        assert needed is True, reason


def test_batched_output_is_only_used_for_noqa_findings(tmp_path: Path) -> None:
    """A line carrying both a noqa and a type: ignore must not feed mypy's finding ruff output."""
    file_path = tmp_path / "mixed.py"
    safe_io.write_text_no_follow(
        file_path,
        "import os  # noqa: F401 - kept  # type: ignore[attr-defined]  # kept\n",
    )
    findings = check_suppressions.find_suppressions(file_path)
    kinds = {finding.kind for finding in findings}
    assert kinds >= {"noqa", "type-ignore"}
    outputs = {(file_path, 1): "mixed.py:1:8: F401 `os` imported but unused"}
    for finding in findings:
        chosen = check_suppressions._batched_output_for(finding, outputs)
        if finding.kind == "noqa":
            assert chosen is not None
        else:
            assert chosen is None


_TEXT_ONLY_MISTRAL_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}"
    "{{ '[INST]' + message['content'] + '[/INST]' }}{% elif message['role'] == 'system' %}"
    "{{ '[SYSTEM_PROMPT]' + message['content'] + '[/SYSTEM_PROMPT]' }}{% endif %}{% endfor %}"
)
_MULTIMODAL_MISTRAL_TEMPLATE = (
    "{%- for message in messages %}{%- if message['content'] is string %}{{ message['content'] }}"
    "{%- else %}{%- for block in message['content'] %}{%- if block['type'] == 'text' %}"
    "{{ block['text'] }}{%- endif %}{%- endfor %}{%- endif %}{%- endfor %}"
)


@pytest.mark.parametrize(
    ("url", "allowed"),
    [
        ("https://example.test/config.json", True),
        ("http://localhost/config.json", True),
        ("HTTPS://example.test/config.json", True),
        ("file:///tmp/config.json", False),
        ("data:text/plain,config", False),
        ("ftp://example.test/config.json", False),
    ],
)
def test_hub_precheck_fetch_allows_only_http_schemes(
    monkeypatch: pytest.MonkeyPatch, url: str, *, allowed: bool
) -> None:
    """A configured Hub endpoint must not turn metadata fetching into a local read."""
    calls: list[tuple[str, int, str | None]] = []

    def fetch(request: Request, *, timeout: int) -> io.BytesIO:
        calls.append((request.full_url, timeout, request.get_header("Authorization")))
        return io.BytesIO(b'{"model_type":"example"}')

    monkeypatch.setattr(hub_precheck, "urlopen", fetch)
    result = hub_precheck._fetch_text(url, {"Authorization": "Bearer fixture"})
    assert result == ('{"model_type":"example"}' if allowed else None)
    assert calls == ([(url, 30, "Bearer fixture")] if allowed else [])


def test_hub_precheck_template_shape_separates_text_only_from_multimodal_templates() -> None:
    """The Mistral-Small-3.2 conversion's template is the text-only shape; Ministral's is not."""
    shape = hub_precheck.template_shape
    assert shape(_TEXT_ONLY_MISTRAL_TEMPLATE) == "string-only"
    assert shape(_MULTIMODAL_MISTRAL_TEMPLATE) == "iterates-content"
    assert shape("{% if message.content is string %}{{ message.content }}{% endif %}") == (
        "iterates-content"
    )
    assert shape("{% for item in message.content %}{{ item.text }}{% endfor %}") == (
        "iterates-content"
    )
    assert shape("") == "absent"
    assert shape(None) == "absent"
    assert shape("{{ messages | tojson }}") == "unknown"


def test_hub_precheck_reads_named_templates_from_tokenizer_config() -> None:
    read = hub_precheck._template_from_tokenizer_config
    assert read(json.dumps({"chat_template": _TEXT_ONLY_MISTRAL_TEMPLATE})) == (
        _TEXT_ONLY_MISTRAL_TEMPLATE
    )
    named = json.dumps(
        {"chat_template": [{"name": "default", "template": "{{ x }}"}, {"name": "tool_use"}]}
    )
    assert read(named) == "{{ x }}"
    assert read(json.dumps({"model_max_length": 1})) is None
    assert read("not json") is None
    assert read(None) is None


def test_hub_precheck_layout_rule_matches_the_server_style_filter() -> None:
    assert (
        hub_precheck.layout_missing(["config.json", "tokenizer_config.json", "model.safetensors"])
        == []
    )
    assert (
        hub_precheck.layout_missing(
            ["config.json", "tokenizer_config.json", "model.safetensors.index.json"]
        )
        == []
    )
    assert hub_precheck.layout_missing(["config.json", "README.md"]) == [
        "tokenizer_config.json",
        "*.safetensors",
    ]


def _candidate(**overrides: object) -> hub_precheck.HubCandidate:
    base: dict[str, object] = {
        "repo_id": "org/model",
        "files": ("config.json", "tokenizer_config.json", "model.safetensors"),
        "size_gb": 4.2,
        "model_type": "mistral3",
        "resolved_model_type": "mistral3",
        "arch_supported": True,
        "template": "iterates-content",
    }
    base.update(overrides)
    return hub_precheck.HubCandidate(**typing.cast("dict[str, typing.Any]", base))


def test_hub_precheck_memory_verdict_uses_weights_against_memory_facts() -> None:
    """Weights larger than unified memory block; larger than the working set warn."""
    status, reasons = _candidate(weights_gb=173.8, memory_gb=137.4, working_set_gb=103.1).verdict()
    assert status == "BLOCKED"
    assert any("exceed this Mac's 137.4 GB of unified memory" in reason for reason in reasons)
    status, reasons = _candidate(weights_gb=110.0, memory_gb=137.4, working_set_gb=103.1).verdict()
    assert status == "WARN"
    assert any("recommended working set (103.1 GB)" in reason for reason in reasons)
    assert _candidate(weights_gb=6.0, memory_gb=137.4, working_set_gb=103.1).verdict() == (
        "OK",
        [],
    )
    # Unknown memory facts never produce a memory verdict.
    assert _candidate(weights_gb=500.0).verdict() == ("OK", [])
    assert hub_precheck._gigabytes(137_438_953_472) == 137.4
    assert hub_precheck._gigabytes(None) is None


def _fetch_with_siblings(
    monkeypatch: pytest.MonkeyPatch,
    siblings: Mapping[str, int | None],
    fetched: dict[str, str] | None = None,
    pipeline_tag: str | None = None,
    card_data: dict[str, object] | None = None,
    bases_of: Mapping[str, tuple[str, ...]] | None = None,
    cached: Mapping[str, tuple[str, ...]] | None = None,
) -> hub_precheck.HubCandidate:
    """Run fetch_candidate against a mocked hub listing (16 GB memory, 12 GB working set).

    Lineage lookups are injected too, so no test reads the hub or the real cache.
    """

    class _Sibling:
        def __init__(self, name: str, size: int | None) -> None:
            self.rfilename = name
            self.size = size

    class _Api:
        def model_info(self, _repo_id: str, **_kwargs: object) -> object:
            listing = [_Sibling(n, s) for n, s in siblings.items()]
            return type(
                "Info",
                (),
                {"siblings": listing, "pipeline_tag": pipeline_tag, "card_data": card_data},
            )()

    texts = {"config.json": '{"model_type": "qwen3_vl"}', **(fetched or {})}
    siblings = {**dict.fromkeys(texts, 100), **siblings}
    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_url", lambda _repo, name: name)
    monkeypatch.setattr(huggingface_hub.utils, "build_hf_headers", dict)
    monkeypatch.setattr(hub_precheck, "_fetch_text", lambda url, _headers: texts.get(url))
    return hub_precheck.fetch_candidate(
        "org/repo",
        arch_check=lambda raw: (str(raw), str(raw), True),
        memory_check=lambda: (16_000_000_000, 12_000_000_000),
        bases_of=lambda repo: (bases_of or {}).get(repo, ()),
        cached_bases=lambda: cached or {},
    )


def test_hub_precheck_memory_counts_only_the_weights_the_loader_selects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: a nested alternate checkpoint was added in, turning a 6 GB model BLOCKED."""
    base = {"config.json": 100, "tokenizer_config.json": 100}
    nested = _fetch_with_siblings(
        monkeypatch,
        {**base, "model.safetensors": 6_000_000_000, "original/model.safetensors": 20_000_000_000},
    )
    assert nested.weights_gb == 6.0
    assert nested.size_gb == 26.0  # the whole repository stays a separate figure
    assert "unified memory" not in " ".join(nested.verdict()[1])

    index = json.dumps({"weight_map": {"a": "model-1.safetensors", "b": "model-2.safetensors"}})
    indexed = _fetch_with_siblings(
        monkeypatch,
        {
            **base,
            "model.safetensors.index.json": 100,
            "model-1.safetensors": 4_000_000_000,
            "model-2.safetensors": 4_000_000_000,
            "extra.safetensors": 30_000_000_000,
        },
        {"model.safetensors.index.json": index},
    )
    assert indexed.weights_gb == 8.0

    unsized = _fetch_with_siblings(
        monkeypatch, {**base, "model-1.safetensors": 4_000_000_000, "model-2.safetensors": None}
    )
    assert unsized.weights_gb is None
    status, reasons = unsized.verdict()
    assert status == "WARN"
    assert any("memory fit not assessed" in reason for reason in reasons)


def test_hub_precheck_reads_the_template_in_the_processors_file_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: chat_template.json was never read (Idefics3 looked text-only)."""
    base = {"tokenizer_config.json": 100, "model.safetensors": 1_000}
    multimodal = '{"chat_template": "{% for part in message[\'content\'] %}{% endfor %}"}'
    text_only = '{"chat_template": "{{ bos }}{{ message[\'content\'] + eos }}"}'
    candidate = _fetch_with_siblings(
        monkeypatch,
        base,
        {"chat_template.json": multimodal, "tokenizer_config.json": text_only},
        pipeline_tag="image-text-to-text",
    )
    assert (candidate.template, candidate.template_source) == (
        "iterates-content",
        "chat_template.json",
    )
    assert candidate.pipeline_tag == "image-text-to-text"
    rendered = hub_precheck.render([candidate])
    assert "template from: chat_template.json; pipeline_tag: image-text-to-text" in rendered
    only_tokenizer = _fetch_with_siblings(monkeypatch, base, {"tokenizer_config.json": text_only})
    status, reasons = only_tokenizer.verdict()
    assert status == "BLOCKED"
    assert any("text-only chat template in tokenizer_config.json" in reason for reason in reasons)


def test_hub_precheck_records_checkpoint_model_code_instead_of_blocking_the_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """config.json model_file makes mlx-vlm import checkpoint code, so no package is needed."""
    config = '{"model_type": "brand_new", "model_file": "modeling.py"}'
    template = '{"chat_template": "{% for part in message[\'content\'] %}{% endfor %}"}'
    candidate = _fetch_with_siblings(
        monkeypatch,
        {"tokenizer_config.json": 100, "model.safetensors": 1_000},
        {"config.json": config, "tokenizer_config.json": template},
    )
    assert candidate.model_file == "modeling.py"
    assert candidate.arch_supported is None
    status, reasons = candidate.verdict()
    assert status == "WARN"
    assert any(
        "model_file='modeling.py'" in reason and "runs that file" in reason for reason in reasons
    )


def test_hub_precheck_lineage_helpers_follow_declared_bases() -> None:
    """Card bases normalise, chains stop at the hop limit and cycles, relatives match ids."""
    assert hub_precheck.declared_bases({"base_model": "org/base"}) == ("org/base",)
    assert hub_precheck.declared_bases({"base_model": ["a/x", " a/x ", "", 3, "b/y"]}) == (
        "a/x",
        "b/y",
    )
    assert hub_precheck.declared_bases({"license": "mit"}) == ()
    assert hub_precheck.declared_bases(None) == ()

    parents = {"q/quant": ("f/tune",), "f/tune": ("g/base",), "g/base": ("g/pre",)}
    looked_up: list[str] = []

    def bases_of(repo: str) -> tuple[str, ...]:
        looked_up.append(repo)
        return parents.get(repo, ())

    # Three hops are reported; the third is never looked up.
    assert hub_precheck.lineage(("f/tune",), bases_of) == (("f/tune", "g/base", "g/pre"), ())
    assert looked_up == ["f/tune", "g/base"]

    # A base whose card cannot be read stays in the chain and is reported as unread.
    def unreadable_base(repo: str) -> tuple[str, ...]:
        if repo == "g/base":
            raise hub_precheck.BaseCardUnreadableError(repo)
        return parents.get(repo, ())

    assert hub_precheck.lineage(("f/tune",), unreadable_base) == (
        ("f/tune", "g/base"),
        ("g/base",),
    )
    # A cycle, in any letter case, ends the chain instead of repeating it.
    cyclic = {"a/one": ("A/Two",), "a/two": ("a/ONE",)}
    assert hub_precheck.lineage(("a/one",), lambda repo: cyclic.get(repo.lower(), ())) == (
        ("a/one", "A/Two"),
        (),
    )

    cached = {
        "mlx/base-4bit": ("G/Base",),  # declares the shared base (case differs)
        "g/base": (),  # is the base itself
        "mlx/finetune-of-candidate": ("org/candidate",),  # declares the candidate
        "org/candidate": (),  # the candidate itself is never its own relative
        "mlx/unrelated": ("x/other",),
    }
    assert hub_precheck.cached_relatives("org/candidate", ("f/tune", "g/base"), cached) == (
        ("g/base", "g/base"),
        ("mlx/base-4bit", "G/Base"),
        ("mlx/finetune-of-candidate", "org/candidate"),
    )


def test_hub_precheck_reports_lineage_as_facts_not_verdicts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Regression: a Gemma-4-E4B fine-tune was downloaded before its card showed the base."""
    template = '{"chat_template": "{% for part in message[\'content\'] %}{% endfor %}"}'
    candidate = _fetch_with_siblings(
        monkeypatch,
        {"tokenizer_config.json": 100, "model.safetensors": 1_000},
        {"tokenizer_config.json": template},
        card_data={"base_model": "nvidia/Bolt-E4B-BF16"},
        bases_of={"nvidia/Bolt-E4B-BF16": ("google/gemma-4-E4B-it",)},
        cached={"mlx-community/gemma-4-e4b-it-4bit": ("google/gemma-4-E4B-it",)},
    )
    assert candidate.base_models == ("nvidia/Bolt-E4B-BF16", "google/gemma-4-E4B-it")
    assert candidate.cached_relatives == (
        ("mlx-community/gemma-4-e4b-it-4bit", "google/gemma-4-E4B-it"),
    )
    assert candidate.already_cached is False
    # Lineage prompts a look at the card; it never changes the verdict.
    assert candidate.verdict() == ("OK", [])
    rendered = hub_precheck.render([candidate])
    assert "base: nvidia/Bolt-E4B-BF16 → google/gemma-4-E4B-it" in rendered
    assert (
        "cached models with a shared base: mlx-community/gemma-4-e4b-it-4bit "
        "(via google/gemma-4-E4B-it)"
    ) in rendered

    assert hub_precheck.main(["org/repo", "--json"], fetch=lambda _repo: candidate) == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["base_models"] == ["nvidia/Bolt-E4B-BF16", "google/gemma-4-E4B-it"]
    assert payload["cached_relatives"] == [
        ["mlx-community/gemma-4-e4b-it-4bit", "google/gemma-4-E4B-it"]
    ]

    # Without a declared base there is no chain, and a cached copy is named as such.
    plain = _fetch_with_siblings(
        monkeypatch,
        {"tokenizer_config.json": 100, "model.safetensors": 1_000},
        {"tokenizer_config.json": template},
        cached={"org/repo": ()},
    )
    assert (plain.base_models, plain.cached_relatives, plain.already_cached) == ((), (), True)
    assert "already cached: yes" in hub_precheck.render([plain])
    assert "base:" not in hub_precheck.render([plain])


def test_hub_precheck_weight_selection_mirrors_the_loader() -> None:
    """Index shards that exist win; else root-level files, never consolidated or nested ones."""
    select = hub_precheck.selected_weight_files
    files = ["a.safetensors", "consolidated.safetensors", "sub/b.safetensors", "s1.safetensors"]
    assert select(files, None) == ("a.safetensors", "s1.safetensors")
    assert select(files, json.dumps({"weight_map": {"x": "s1.safetensors"}})) == ("s1.safetensors",)
    # An index none of whose shards exist, or a broken one, falls back like the loader.
    assert select(files, json.dumps({"weight_map": {"x": "gone.safetensors"}})) == (
        "a.safetensors",
        "s1.safetensors",
    )
    assert select(files, "{not json") == ("a.safetensors", "s1.safetensors")


def test_hub_precheck_verdicts_block_layout_architecture_and_text_only_templates() -> None:
    assert _candidate().verdict() == ("OK", [])
    status, reasons = _candidate(template="string-only").verdict()
    assert status == "BLOCKED"
    assert any("text-only chat template" in reason for reason in reasons)
    status, reasons = _candidate(
        model_type="internvl", resolved_model_type="internvl", arch_supported=False
    ).verdict()
    assert status == "BLOCKED"
    assert "no mlx-vlm package for model_type 'internvl'" in reasons[0]
    status, reasons = _candidate(files=("config.json",)).verdict()
    assert status == "BLOCKED"
    assert "missing tokenizer_config.json, *.safetensors" in reasons[0]
    status, reasons = _candidate(template="unknown", arch_supported=None).verdict()
    assert status == "WARN"
    assert len(reasons) == 2
    status, reasons = _candidate(error="404 Client Error").verdict()
    assert status == "BLOCKED"
    assert reasons == ["hub read failed: 404 Client Error"]


def test_hub_precheck_cli_renders_verdicts_and_exits_nonzero_when_blocked(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI takes an injected fetcher so no network is touched."""
    candidates = {
        "org/good": _candidate(repo_id="org/good"),
        "org/text-only": _candidate(repo_id="org/text-only", template="string-only"),
    }
    assert hub_precheck.main(["org/good"], fetch=candidates.__getitem__) == 0
    out = capsys.readouterr().out
    assert out.startswith("OK ")
    assert "org/good" in out
    assert hub_precheck.main(["org/good", "org/text-only"], fetch=candidates.__getitem__) == 1
    out = capsys.readouterr().out
    assert "BLOCKED" in out
    assert "text-only chat template" in out
    assert hub_precheck.main(["org/text-only", "--json"], fetch=candidates.__getitem__) == 1
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["status"] == "BLOCKED"
    assert payload["repo_id"] == "org/text-only"


def test_arch_precheck_for_model_type_resolves_aliases_and_rejects_empty() -> None:
    """The hub pre-check and the cached-repo precheck share one resolution."""
    assert check_models.arch_precheck_for_model_type(None) == (None, None, None)
    assert check_models.arch_precheck_for_model_type("") == (None, None, None)
    model_type, resolved, supported = check_models.arch_precheck_for_model_type("MISTRAL3")
    assert (model_type, resolved) == ("mistral3", "mistral3")
    assert supported in (True, None)


def _git(cwd: Path, *args: str) -> str:
    """Run git with a fixed identity; return stdout."""
    return subprocess.run(  # noqa: S603 - fixed git argv built by the test
        [  # noqa: S607 - git from PATH in a test
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.invalid",
            "-c",
            "init.defaultBranch=main",
            *args,
        ],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def test_update_pull_failure_stops_with_diagnostics(tmp_path: Path) -> None:
    """A diverged checkout stops the run; the report names both sides of the divergence.

    Continuing past a failed pull left the repo unbuilt and, for mlx, never
    pinned, so later eager upgrades could replace the local build.
    """
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q")
    safe_io.write_text_no_follow(origin / "kernel.h", "v1\n")
    _git(origin, "add", "kernel.h")
    _git(origin, "commit", "-q", "-m", "initial")
    _git(tmp_path, "clone", "-q", str(origin), "checkout")
    checkout = tmp_path / "checkout"
    safe_io.write_text_no_follow(origin / "kernel.h", "v2 upstream\n")
    _git(origin, "commit", "-q", "-am", "upstream fix")
    safe_io.write_text_no_follow(checkout / "local.txt", "local\n")
    _git(checkout, "add", "local.txt")
    _git(checkout, "commit", "-q", "-m", "local change")

    function = _update_script_function("report_git_pull_failure")
    result = subprocess.run(  # noqa: S603 - fixed /bin/bash evaluates an extracted repo function
        [
            "/bin/bash",
            "-c",
            (
                f"{function}\ngit pull -q --ff-only 2>/dev/null && exit 9\n"
                f'report_git_pull_failure mlx "{checkout}"'
            ),
        ],
        cwd=checkout,
        capture_output=True,
        text=True,
        check=True,
    )
    report = result.stdout
    assert "git pull --ff-only failed for mlx" in report
    assert "upstream fix" in report
    assert "local change" in report
    assert "Nothing was built or reinstalled" in report

    script = (PKG_ROOT / "tools" / "update.sh").read_text(encoding="utf-8")
    stage1 = script[script.index("# Stage 1: Sync") : script.index("# Stage 2: Verify")]
    assert "if git pull --ff-only; then" in stage1
    assert "return 1" in stage1[stage1.index("report_git_pull_failure") :]
    # mlx-vlm's requirements.txt is its dependency list; a separate
    # requirements pass ran before the mlx pin and could replace the local build.
    assert "-r requirements.txt" not in script


def test_update_reinstall_that_replaces_a_verified_local_build_stops(tmp_path: Path) -> None:
    """After the eager project reinstall, every install Stage 3b verified must still verify."""
    function = _update_script_function("verify_local_installs_survived_reconcile")
    driver = f"""
{function}
get_installed_distribution_version() {{ echo "0.7.4"; }}
pip_show_field() {{ echo "/site-packages"; }}
VERIFIED_LOCAL_INSTALLS=("mlx=/repo/mlx" "mlx-vlm=/repo/mlx-vlm")
verify_expected_editable_install() {{ [[ "$1" == "mlx" ]]; }}
verify_local_installs_survived_reconcile && exit 9
echo "status=$?"
VERIFIED_LOCAL_INSTALLS=()
verify_local_installs_survived_reconcile && echo "empty-ok"
"""
    result = subprocess.run(  # noqa: S603 - fixed /bin/bash evaluates an extracted repo function
        ["/bin/bash", "-c", driver],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "replaced the local mlx-vlm build from /repo/mlx-vlm" in result.stdout
    assert "Installed now: 0.7.4" in result.stdout
    assert "status=1" in result.stdout
    assert "empty-ok" in result.stdout

    script = (PKG_ROOT / "tools" / "update.sh").read_text(encoding="utf-8")
    assert (
        "reconcile_project_environment_from_pyproject\nverify_local_installs_survived_reconcile\n"
        in script
    )
    assert 'VERIFIED_LOCAL_INSTALLS+=("${REPO_NAMES[idx]}=${REPO_PATHS[idx]}")' in script


def test_update_failed_mlx_build_pins_the_surviving_local_install(tmp_path: Path) -> None:
    """A compile failure leaves the old editable build installed; later installs stay pinned.

    Regression: the failure branch set REPO_SKIP without pinning, so the final
    eager project reinstall ran unconstrained and would have replaced the
    working local build with the PyPI wheel once a release overtook the local
    dev version.
    """
    survivor = """
get_installed_distribution_version() { echo "0.32.3.dev20260912+229f5b430"; }
verify_expected_editable_install() { return 0; }
preserve_local_mlx_after_failed_build /repo/mlx
pip_install somepkg
"""
    log = _run_update_pip_wrapper_harness(survivor, tmp_path)
    assert "previous local build is still installed" in log
    assert "Pinned local mlx 0.32.3.dev20260912+229f5b430" in log
    assert "--constraint" in log

    replaced = """
get_installed_distribution_version() { echo "0.32.2"; }
verify_expected_editable_install() { return 1; }
preserve_local_mlx_after_failed_build /repo/mlx
pip_install somepkg
"""
    # A separate directory: the harness appends to one pip log per directory.
    replaced_dir = tmp_path / "replaced"
    replaced_dir.mkdir()
    log = _run_update_pip_wrapper_harness(replaced, replaced_dir)
    assert "nothing to pin" in log
    assert "--constraint" not in log

    script = (PKG_ROOT / "tools" / "update.sh").read_text(encoding="utf-8")
    failure_branch = script[script.index('echo "\u274c Failed to install ${REPO_NAMES[idx]}"') :]
    assert 'preserve_local_mlx_after_failed_build "${REPO_PATHS[idx]}"' in failure_branch
    # A failed Stage 3 install stops the run after the restore attempts: a run
    # that continued (until 2026-10-28) exited 0 on a stale editable install.
    stop_block = failure_branch[: failure_branch.index("Stage 3b")]
    assert "failed to build/install from" in stop_block
    assert "return 1" in stop_block


# ---------------------------------------------------------------------------
# Updater environment targeting, conda ownership, preflight ordering
# ---------------------------------------------------------------------------


def _fake_tool_bin(tmp_path: Path, log: Path, names: Iterable[str]) -> Path:
    """Fake executables that only record their invocation in ``log``."""
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir(exist_ok=True)
    for name in names:
        safe_io.write_text_no_follow(
            fake_bin / name, f'#!/bin/bash\necho "{name} $*" >> "{log}"\n', mode=0o755
        )
    return fake_bin


def _fake_python_prefix(tmp_path: Path, name: str) -> Path:
    """An env prefix whose bin/python is an executable stub."""
    prefix = tmp_path / "envs" / name
    (prefix / "bin").mkdir(parents=True)
    safe_io.write_text_no_follow(prefix / "bin" / "python", "#!/bin/bash\n", mode=0o755)
    return prefix


def _resolve_update_target(env: Mapping[str, str], path: str) -> tuple[int, dict[str, str], str]:
    """Run update.sh's target resolution with exactly ``env``; return status, target, output."""
    function = _update_script_function("resolve_update_target")
    driver = (
        f"{function}\n"
        "status=0; resolve_update_target || status=$?\n"
        'echo "STATUS=$status"; echo "TYPE=${UPDATE_ENV_TYPE:-}"\n'
        'echo "NAME=${UPDATE_ENV_NAME:-}"; echo "PYTHON=${UPDATE_PYTHON:-}"\n'
    )
    result = subprocess.run(  # noqa: S603 - fixed /bin/bash evaluates an extracted repo function
        ["/bin/bash", "-c", driver],
        capture_output=True,
        text=True,
        check=True,
        env={"PATH": path, "CONDA_ENV": "mlx-vlm", **env},
    )
    fields = dict(
        line.split("=", 1) for line in result.stdout.splitlines() if line.split("=", 1)[0].isupper()
    )
    return int(fields.pop("STATUS")), fields, result.stdout


def test_update_target_is_the_active_conda_env_when_it_matches(tmp_path: Path) -> None:
    """The correct env active: its own interpreter is the target, whatever PATH holds."""
    prefix = _fake_python_prefix(tmp_path, "mlx-vlm")
    status, target, _ = _resolve_update_target(
        {"CONDA_DEFAULT_ENV": "mlx-vlm", "CONDA_PREFIX": str(prefix)}, "/usr/bin:/bin"
    )
    assert status == 0
    assert target == {"TYPE": "conda", "NAME": "mlx-vlm", "PYTHON": f"{prefix}/bin/python"}


def test_update_target_refuses_a_different_active_conda_env(tmp_path: Path) -> None:
    """Regression: CONDA_ENV=review-target with mlx-vlm active updated mlx-vlm."""
    prefix = _fake_python_prefix(tmp_path, "mlx-vlm")
    status, target, output = _resolve_update_target(
        {
            "CONDA_DEFAULT_ENV": "mlx-vlm",
            "CONDA_PREFIX": str(prefix),
            "CONDA_ENV": "review-target",
        },
        "/usr/bin:/bin",
    )
    assert status == 1
    assert target["PYTHON"] == ""
    assert "'mlx-vlm' is active, but the update target is 'review-target'" in output
    assert "Nothing was changed" in output


def test_update_target_with_no_env_active(tmp_path: Path) -> None:
    """No env active: refuse when conda could provide the target; else ask for system Python."""
    log = tmp_path / "calls.log"
    with_conda = _fake_tool_bin(tmp_path, log, ["conda"])
    status, target, output = _resolve_update_target({}, f"{with_conda}:/usr/bin:/bin")
    assert status == 1
    assert target["PYTHON"] == ""
    assert "conda activate mlx-vlm" in output

    system_bin = tmp_path / "system-bin"
    system_bin.mkdir()
    safe_io.write_text_no_follow(system_bin / "python3", "#!/bin/bash\n", mode=0o755)
    status, target, _ = _resolve_update_target({}, str(system_bin))
    assert status == 2, "a system Python is only used after the caller confirms"
    assert target == {"TYPE": "system", "NAME": "system", "PYTHON": f"{system_bin}/python3"}
    assert not log.exists(), "resolution must not run conda"


def test_update_target_is_an_active_venv(tmp_path: Path) -> None:
    """An active venv is the target, as in the Makefiles."""
    prefix = _fake_python_prefix(tmp_path, "dev-venv")
    status, target, _ = _resolve_update_target({"VIRTUAL_ENV": str(prefix)}, "/usr/bin:/bin")
    assert status == 0
    assert target == {"TYPE": "venv", "NAME": "dev-venv", "PYTHON": f"{prefix}/bin/python"}


def _copy_update_script_layout(tmp_path: Path) -> Path:
    """update.sh in a check_models/src/tools layout whose parent holds no MLX checkouts."""
    tools_dir = tmp_path / "work" / "check_models" / "src" / "tools"
    tools_dir.mkdir(parents=True)
    shutil.copy2(PKG_ROOT / "tools" / "update.sh", tools_dir / "update.sh")
    return tools_dir / "update.sh"


def test_update_script_mismatched_env_stops_before_any_change(tmp_path: Path) -> None:
    """The whole script, not just the resolver: no tool runs when the target is wrong."""
    log = tmp_path / "calls.log"
    fake_bin = _fake_tool_bin(tmp_path, log, ["conda", "brew", "npm", "pip", "python", "python3"])
    script = _copy_update_script_layout(tmp_path)
    result = subprocess.run(  # noqa: S603 - fixed bash runs a copied repo script with fake tools
        ["/bin/bash", str(script)],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        env={
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "HOME": str(tmp_path),
            "CONDA_DEFAULT_ENV": "mlx-vlm",
            "CONDA_PREFIX": str(_fake_python_prefix(tmp_path, "mlx-vlm")),
            "CONDA_ENV": "review-target",
        },
    )
    assert result.returncode == 1
    assert "update target is 'review-target'" in result.stdout
    assert not log.exists(), safe_io.read_text_no_follow(log)


def test_update_script_broken_checkout_stops_in_preflight(tmp_path: Path) -> None:
    """Repository problems that need no pull stop the run before conda, brew, npm or pip."""
    log = tmp_path / "calls.log"
    fake_bin = _fake_tool_bin(tmp_path, log, ["conda", "brew", "npm", "pip"])
    script = _copy_update_script_layout(tmp_path)
    mlx_vlm = tmp_path / "work" / "mlx-vlm"
    subprocess.run(  # noqa: S603 - fixed git command on a test-created directory
        ["git", "init", "-q", str(mlx_vlm)],  # noqa: S607 - git from PATH, as update.sh uses it
        check=True,
    )
    refs = mlx_vlm / ".git" / "refs" / "heads"
    refs.mkdir(parents=True, exist_ok=True)
    safe_io.write_text_no_follow(refs / "main 2", "b" * 40 + "\n")
    venv = _fake_python_prefix(tmp_path, "venv")
    result = subprocess.run(  # noqa: S603 - fixed bash runs a copied repo script with fake tools
        ["/bin/bash", str(script)],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        env={"PATH": f"{fake_bin}:/usr/bin:/bin", "HOME": str(tmp_path), "VIRTUAL_ENV": str(venv)},
    )
    assert result.returncode == 1
    assert "mlx-vlm has malformed git refs" in result.stdout
    assert "Nothing was changed." in result.stdout
    assert "Planned changes" not in result.stdout
    assert not log.exists(), safe_io.read_text_no_follow(log)


def test_update_script_failure_reports_the_steps_already_completed() -> None:
    """A failing step names itself and every step that finished before it."""
    functions = "".join(
        _update_script_function(name) for name in ("run_step", "report_update_outcome")
    )
    driver = (
        "set -euo pipefail\n"
        "cleanup_local_mlx_constraint() { :; }\n"
        'COMPLETED_STEPS=()\nCURRENT_STEP=""\n'
        f"{functions}"
        "trap report_update_outcome EXIT\n"
        'run_step "conda base and environment updates" true\n'
        'run_step "Homebrew update and upgrade" false\n'
        'run_step "never reached" true\n'
    )
    result = subprocess.run(  # noqa: S603 - fixed /bin/bash evaluates extracted repo functions
        ["/bin/bash", "-c", driver], capture_output=True, text=True, check=False
    )
    assert result.returncode == 1
    assert (
        "Completed before the stop:\n     - conda base and environment updates\n" in result.stdout
    )
    assert "Failed during (may be partly applied): Homebrew update and upgrade" in result.stdout
    assert "never reached" not in result.stdout


def test_update_script_main_flow_runs_preflight_before_the_first_change() -> None:
    """Target resolution, repo checks and the plan precede every changing step."""
    update_script = (PKG_ROOT / "tools" / "update.sh").read_text(encoding="utf-8")
    main_flow = update_script[update_script.index("# ── Preflight") :]
    first_change = main_flow.index("run_step ")
    for preflight in (
        "resolve_update_target ||",
        "preflight_local_mlx_repos",
        "report_update_plan",
    ):
        assert main_flow.index(preflight) < first_change, preflight
    assert main_flow.index("trap report_update_outcome EXIT") < first_change
    assert 'pip() { "$UPDATE_PYTHON" -m pip "$@"; }' in main_flow
    assert main_flow.index("snapshot_update_state >") < first_change


def _git_commit(repo: Path, message: str) -> str:
    """Make an empty commit in ``repo``; return its full revision."""
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run(  # noqa: S603 - fixed git command on a test-created repository
        [*git, "commit", "-q", "--allow-empty", "-m", message], check=True
    )
    return subprocess.run(  # noqa: S603 - fixed git command on a test-created repository
        ["git", "-C", str(repo), "rev-parse", "HEAD"],  # noqa: S607 - git from PATH, as update.sh
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _run_update_change_report(tmp_path: Path, before: str, after: str) -> str:
    """Run update.sh's change report on two snapshot texts in a check_models layout."""
    script = _copy_update_script_layout(tmp_path)
    safe_io.write_text_no_follow(tmp_path / "before.tsv", before)
    safe_io.write_text_no_follow(tmp_path / "after.tsv", after)
    functions = "".join(
        _update_script_function(name) for name in ("diff_update_state", "report_update_changes")
    )
    driver = (
        f"set -euo pipefail\nSCRIPT_DIR={script.parent}\n{functions}"
        f"report_update_changes {tmp_path / 'before.tsv'} {tmp_path / 'after.tsv'}\n"
    )
    result = subprocess.run(  # noqa: S603 - fixed /bin/bash evaluates extracted repo functions
        ["/bin/bash", "-c", driver], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr + result.stdout
    return result.stdout


@pytest.mark.subprocess
def test_update_change_report_tabulates_versions_and_checkout_commits(tmp_path: Path) -> None:
    """The end-of-run table lists changed, new and removed entries and pulled commits."""
    mlx = tmp_path / "work" / "mlx"
    subprocess.run(  # noqa: S603 - fixed git command on a test-created directory
        ["git", "init", "-q", str(mlx)],  # noqa: S607 - git from PATH, as update.sh uses it
        check=True,
    )
    old = _git_commit(mlx, "base")
    for message in ("one", "two", "three"):
        new = _git_commit(mlx, message)
    before = (
        f"git\t\t\ngit\tmlx\t{old}\npip\tnumpy\t2.3.0\npip\tgone\t1.0\npip\t\t\n"
        "conda\tpython\t3.14.7\nconda\t\t\nconda base\tconda\t26.1.0\nconda base\t\t\n"
        "brew\tcmake\t4.1.0\nbrew\t\t\nnpm\tmarkdownlint-cli2\t0.18.0\nnpm\t\t\n"
    )
    # cask was unreadable before the run, npm after it: neither is compared.
    after = (
        f"git\t\t\ngit\tmlx\t{new}\npip\tnumpy\t2.3.1\npip\tadded\t5.2.0\npip\t\t\n"
        "conda\tpython\t3.14.8\nconda\t\t\nconda base\tconda\t26.1.0\nconda base\t\t\n"
        "brew\tcmake\t4.1.0\nbrew\t\t\ncask\tghostty\t1.2.0\ncask\t\t\n"
        "npm\tmarkdownlint-cli2\t0.18.0\n"
    )
    output = _run_update_change_report(tmp_path, before, after)
    rows = [line.split() for line in output.splitlines()[2:]]
    assert output.splitlines()[1] == "[update.sh] What changed in this run:"
    assert rows == [
        ["Source", "Package", "Before", "After"],
        ["git", "mlx", old[:9], new[:9], "(+3", "commits)"],
        ["pip", "added", "-", "5.2.0", "(new)"],
        ["pip", "gone", "1.0", "removed"],
        ["pip", "numpy", "2.3.0", "2.3.1"],
        ["conda", "python", "3.14.7", "3.14.8"],
        ["Not", "compared", "(could", "not", "be", "read", "before", "the", "run):", "cask"],
        ["Not", "compared", "(could", "not", "be", "read", "after", "the", "run):", "npm"],
    ]


@pytest.mark.subprocess
def test_update_change_report_lists_a_first_install_in_an_empty_source(tmp_path: Path) -> None:
    """A source read but empty before the run reports what the run installed into it.

    Regression: without read markers an empty source looked unreadable, so a
    first markdownlint-cli2 install was reported as "Nothing changed".
    """
    before = "pip\tnumpy\t2.3.0\npip\t\t\nnpm\t\t\n"
    after = "pip\tnumpy\t2.3.0\npip\t\t\nnpm\tmarkdownlint-cli2\t0.18.1\nnpm\t\t\n"
    output = _run_update_change_report(tmp_path, before, after)
    rows = [line.split() for line in output.splitlines()[2:]]
    assert rows == [
        ["Source", "Package", "Before", "After"],
        ["npm", "markdownlint-cli2", "-", "0.18.1", "(new)"],
    ]


@pytest.mark.subprocess
def test_update_change_report_says_when_nothing_changed(tmp_path: Path) -> None:
    """An unchanged environment gets one line, not an empty table."""
    snapshot = "pip\tnumpy\t2.3.0\npip\t\t\nbrew\tcmake\t4.1.0\nbrew\t\t\n"
    output = _run_update_change_report(tmp_path, snapshot, snapshot)
    assert output.strip() == (
        "[update.sh] Nothing changed: every package version and checkout revision"
        " is as before the run."
    )


@pytest.mark.subprocess
def test_update_state_snapshot_reads_each_source(tmp_path: Path) -> None:
    """Checkouts, Python distributions, conda, Homebrew and markdownlint all land in it."""
    script = _copy_update_script_layout(tmp_path)
    mlx_vlm = tmp_path / "work" / "mlx-vlm"
    subprocess.run(  # noqa: S603 - fixed git command on a test-created directory
        ["git", "init", "-q", str(mlx_vlm)],  # noqa: S607 - git from PATH, as update.sh uses it
        check=True,
    )
    head = _git_commit(mlx_vlm, "base")
    project_root = script.parent.parent
    package_json = project_root / "node_modules" / "markdownlint-cli2" / "package.json"
    package_json.parent.mkdir(parents=True)
    safe_io.write_text_no_follow(package_json, '{"version": "0.18.1"}\n')
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    safe_io.write_text_no_follow(
        fake_bin / "conda",
        "#!/bin/bash\n"
        'echo "# packages in environment"\n'
        'echo "python 3.14.7 h0 conda-forge"\n'
        'echo "pydantic 2.14.1 pypi_0 pypi"\n'
        'echo "check-models 0.17.46 dev_0 <develop>"\n',
        mode=0o755,
    )
    safe_io.write_text_no_follow(
        fake_bin / "brew",
        '#!/bin/bash\n[[ "$*" == *--cask* ]] && echo "ghostty 1.2.0" || echo "cmake 4.0.9 4.1.0"\n',
        mode=0o755,
    )
    driver = (
        f"set -euo pipefail\nSCRIPT_DIR={script.parent}\nPROJECT_ROOT={project_root}\n"
        f"UPDATE_PYTHON={sys.executable}\nUPDATE_ENV_TYPE=conda\nCONDA_ENV=mlx-vlm\n"
        f"{_update_script_function('snapshot_update_state')}snapshot_update_state\n"
    )
    result = subprocess.run(  # noqa: S603 - fixed /bin/bash evaluates an extracted repo function
        ["/bin/bash", "-c", driver],
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "CONDA_DEFAULT_ENV": "mlx-vlm",
        },
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert f"git\tmlx-vlm\t{head}" in lines
    assert "pip\tpytest\t" + pytest.__version__ in lines
    assert "npm\tmarkdownlint-cli2\t0.18.1" in lines
    assert "conda\tpython\t3.14.7" in lines
    assert "conda base\tpython\t3.14.7" in lines
    # conda list shows pip installs despite --no-pip; only the pip source keeps them.
    assert not [line for line in lines if line.startswith("conda") and "pydantic" in line]
    assert not [line for line in lines if line.startswith("conda") and "check-models" in line]
    assert "brew\tcmake\t4.1.0" in lines
    assert "cask\tghostty\t1.2.0" in lines
    markers = {line.split("\t", 1)[0] for line in lines if line.endswith("\t\t")}
    assert markers == {"git", "pip", "npm", "conda", "conda base", "brew", "cask"}


@pytest.mark.subprocess
def test_update_state_snapshot_leaves_an_unreadable_checkout_unmarked(tmp_path: Path) -> None:
    """A checkout whose HEAD cannot be read is not compared, so it never looks removed."""
    script = _copy_update_script_layout(tmp_path)
    (tmp_path / "work" / "mlx" / ".git").mkdir(parents=True)
    driver = (
        f"set -euo pipefail\nSCRIPT_DIR={script.parent}\nPROJECT_ROOT={script.parent.parent}\n"
        f"UPDATE_PYTHON={sys.executable}\nUPDATE_SYSTEM_PACKAGES=0\n"
        f"{_update_script_function('snapshot_update_state')}snapshot_update_state\n"
    )
    result = subprocess.run(  # noqa: S603 - fixed /bin/bash evaluates an extracted repo function
        ["/bin/bash", "-c", driver],
        capture_output=True,
        text=True,
        check=False,
        env={key: value for key, value in os.environ.items() if key != "CONDA_DEFAULT_ENV"},
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert not [line for line in lines if line.startswith("git\t")]
    assert "pip\t\t" in lines


@pytest.mark.subprocess
def test_update_outcome_reports_changes_on_success_and_after_a_stop(tmp_path: Path) -> None:
    """The change table follows a clean run and a failed one; the before file is removed."""
    functions = "".join(
        _update_script_function(name)
        for name in (
            "run_step",
            "report_update_outcome",
            "diff_update_state",
            "report_update_changes",
        )
    )
    before = tmp_path / "before.tsv"
    for final, expected_status in (("true", 0), ("false", 1)):
        safe_io.write_text_no_follow(before, "pip\tnumpy\t2.3.0\npip\t\t\n")
        driver = (
            "set -euo pipefail\n"
            f"SCRIPT_DIR={tmp_path}\n"
            "cleanup_local_mlx_constraint() { :; }\n"
            "snapshot_update_state() { printf 'pip\\tnumpy\\t2.3.1\\npip\\t\\t\\n'; }\n"
            f'COMPLETED_STEPS=()\nCURRENT_STEP=""\nUPDATE_STATE_BEFORE={before}\n'
            f"{functions}"
            "trap report_update_outcome EXIT\n"
            'run_step "packaging tools" true\n'
            f'run_step "project reinstall" {final}\n'
        )
        result = subprocess.run(  # noqa: S603 - fixed /bin/bash evaluates extracted repo functions
            ["/bin/bash", "-c", driver], capture_output=True, text=True, check=False
        )
        assert result.returncode == expected_status, result.stderr
        assert "[update.sh] What changed in this run:" in result.stdout
        assert re.search(r"pip\s+numpy\s+2\.3\.0\s+2\.3\.1", result.stdout)
        assert ("Failed during" in result.stdout) == (expected_status != 0)
        assert not before.exists()


def test_make_update_passes_the_resolved_target_to_the_updater() -> None:
    """``make update`` hands update.sh the same CONDA_ENV the other targets announce."""
    result = subprocess.run(  # noqa: S603 - fixed make dry run of a repo target
        ["make", "-n", "-C", str(PKG_ROOT), "update", "CONDA_ENV=review-target"],  # noqa: S607 - make from PATH
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "CONDA_DEFAULT_ENV": "mlx-vlm"},
    )
    assert "CONDA_ENV='review-target' bash tools/update.sh" in result.stdout


_CONDA_RECORDS = [
    {"name": "openssl", "channel": "conda-forge", "platform": "osx-arm64"},
    {"name": "mlx", "channel": "pypi", "platform": "pypi", "build_string": "pypi_0"},
    {"name": "numpy", "channel": "conda-forge", "platform": "osx-arm64"},
    {"name": "numpy", "channel": "pypi", "platform": "pypi", "build_string": "pypi_0"},
    {"name": "typing_extensions", "channel": "conda-forge", "platform": "noarch"},
]


def _conda_update_decision(
    tmp_path: Path, changes: list[str], records: object, *, force: str = "0"
) -> tuple[dict[str, str], str]:
    """Run the extracted ownership lookup and decision; return owners and the decision."""
    records_file = tmp_path / "records.json"
    safe_io.write_text_no_follow(
        records_file, records if isinstance(records, str) else json.dumps(records)
    )
    functions = "".join(
        _update_script_function(name)
        for name in ("conda_change_ownership", "conda_update_all_decision")
    )
    driver = (
        f'python() {{ "{sys.executable}" "$@"; }}\n'
        f"{functions}"
        'ownership="$(conda_change_ownership "$@")"\n'
        'echo "$ownership"\n'
        f'echo "DECISION $(conda_update_all_decision {force} "$ownership")"\n'
    )
    result = subprocess.run(  # noqa: S603 - fixed /bin/bash evaluates extracted repo functions
        ["/bin/bash", "-c", driver, "_", str(records_file), *changes],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = dict(line.split(" ", 1) for line in result.stdout.splitlines())
    return lines, lines.pop("DECISION")


def test_conda_update_proceeds_when_every_change_is_conda_owned(tmp_path: Path) -> None:
    """Regression: pip lists conda's own distributions, so every update looked like a conflict."""
    owners, decision = _conda_update_decision(
        tmp_path, ["openssl", "typing-extensions"], _CONDA_RECORDS
    )
    assert owners == {"openssl": "conda", "typing-extensions": "conda"}
    assert decision == "update"


def test_conda_update_skips_a_pip_owned_change(tmp_path: Path) -> None:
    """A distribution only pip installed is protected; CONDA_UPDATE_ALL=1 overrides."""
    owners, decision = _conda_update_decision(tmp_path, ["openssl", "mlx"], _CONDA_RECORDS)
    assert owners == {"openssl": "conda", "mlx": "pip"}
    assert decision == "skip"
    _, forced = _conda_update_decision(tmp_path, ["mlx"], _CONDA_RECORDS, force="1")
    assert forced == "update"


@pytest.mark.parametrize(
    ("changes", "records", "owner"),
    [
        pytest.param(["numpy"], _CONDA_RECORDS, "ambiguous", id="pip-over-conda"),
        pytest.param(["openssl"], "conda list failed\n", "ambiguous", id="unreadable-records"),
        pytest.param(["openssl"], [], "ambiguous", id="empty-records"),
    ],
)
def test_conda_update_skips_when_ownership_is_ambiguous(
    tmp_path: Path, changes: list[str], records: object, owner: str
) -> None:
    """Uncertain ownership keeps the conservative outcome."""
    owners, decision = _conda_update_decision(tmp_path, changes, records)
    assert owners == dict.fromkeys(changes, owner)
    assert decision == "skip"


def test_conda_update_treats_an_uninstalled_package_as_a_new_install(tmp_path: Path) -> None:
    """A package with no record replaces nothing, so it does not block the update."""
    owners, decision = _conda_update_decision(tmp_path, ["libnew", "openssl"], _CONDA_RECORDS)
    assert owners == {"libnew": "absent", "openssl": "conda"}
    assert decision == "update"


def _conda_plan_record(name: str, version: str) -> dict[str, str]:
    return {"name": name, "version": version, "channel": "conda-forge", "platform": "osx-arm64"}


# A real `conda update --all --dry-run --json` shape: an ordinary update
# (openssl, unlinked and relinked) plus a new install of numpy, which conda's
# text listing shows without an "old --> new" arrow.
_CONDA_PLAN_WITH_PIP_COLLISION = {
    "success": True,
    "dry_run": True,
    "prefix": "/envs/review-target",
    "actions": {
        "PREFIX": "/envs/review-target",
        "FETCH": [],
        "UNLINK": [_conda_plan_record("openssl", "3.5.5")],
        "LINK": [_conda_plan_record("openssl", "3.5.6"), _conda_plan_record("numpy", "2.3.4")],
    },
}


def _run_update_conda_packages(
    tmp_path: Path, plan: object, records: object, *, force: str = "0"
) -> tuple[str, str]:
    """Run update.sh's real conda step against a fake conda; return output and conda calls."""
    plan_file = tmp_path / "plan.json"
    records_file = tmp_path / "env-records.json"
    log = tmp_path / "conda-calls.log"
    for path, payload in ((plan_file, plan), (records_file, records)):
        safe_io.write_text_no_follow(
            path, payload if isinstance(payload, str) else json.dumps(payload)
        )
    functions = "".join(
        _update_script_function(name)
        for name in (
            "conda_plan_packages",
            "conda_change_ownership",
            "conda_update_all_decision",
            "update_conda_packages",
        )
    )
    driver = (
        f'python() {{ "{sys.executable}" "$@"; }}\n'
        "conda() {\n"
        f'    echo "conda $*" >> "{log}"\n'
        '    case "$*" in\n'
        f'        *--dry-run*) cat "{plan_file}" ;;\n'
        f'        "list -n review-target --json") cat "{records_file}" ;;\n'
        "    esac\n"
        "}\n"
        f"{functions}"
        "UPDATE_ENV_TYPE=conda UPDATE_ENV_NAME=review-target update_conda_packages\n"
    )
    result = subprocess.run(  # noqa: S603 - fixed /bin/bash evaluates extracted repo functions
        ["/bin/bash", "-c", driver],
        capture_output=True,
        text=True,
        check=True,
        env={
            "PATH": "/usr/bin:/bin",
            "TMPDIR": str(tmp_path),
            "CONDA_DEFAULT_ENV": "review-target",
            "CONDA_ENV": "review-target",
            "CONDA_UPDATE_ALL": force,
        },
    )
    return result.stdout, safe_io.read_text_no_follow(log)


def test_conda_update_checks_new_installs_in_the_plan_not_just_updates(tmp_path: Path) -> None:
    """Regression: an ordinary update plus a conda install over pip's numpy was approved.

    The text parser saw only "old --> new" rows (openssl), so the plan read as
    all conda-owned and `conda update --all -y` replaced pip's numpy too.
    """
    records = [
        {"name": "openssl", "channel": "conda-forge", "platform": "osx-arm64"},
        {"name": "numpy", "channel": "pypi", "platform": "pypi", "build_string": "pypi_0"},
    ]
    output, calls = _run_update_conda_packages(tmp_path, _CONDA_PLAN_WITH_PIP_COLLISION, records)
    assert "numpy (installed by pip)" in output
    assert "Skipping conda update --all" in output
    assert "conda update -n review-target --all -y" not in calls

    _, forced_calls = _run_update_conda_packages(
        tmp_path, _CONDA_PLAN_WITH_PIP_COLLISION, records, force="1"
    )
    assert "conda update -n review-target --all -y" in forced_calls

    # The same plan with numpy not installed at all is a plain new install.
    output, calls = _run_update_conda_packages(
        tmp_path, _CONDA_PLAN_WITH_PIP_COLLISION, records[:1]
    )
    assert "conda-owned or newly installed" in output
    assert "conda update -n review-target --all -y" in calls


@pytest.mark.parametrize(
    ("plan", "expected"),
    [
        pytest.param(
            {"success": True, "message": "All requested packages already installed."},
            "already up to date",
            id="up-to-date",
        ),
        pytest.param(
            {"exception_name": "PackagesNotFoundError", "error": "...", "message": "..."},
            "plan could not be read",
            id="conda-error",
        ),
        pytest.param("not json\n", "plan could not be read", id="unreadable"),
        pytest.param(
            {"success": True, "actions": {"LINK": [{"version": "1"}]}},
            "plan could not be read",
            id="nameless-record",
        ),
    ],
)
def test_conda_update_never_applies_an_unusable_or_empty_plan(
    tmp_path: Path, plan: object, expected: str
) -> None:
    """Only a readable plan with changes reaches the ownership check, let alone the update."""
    output, calls = _run_update_conda_packages(tmp_path, plan, _CONDA_RECORDS)
    assert expected in output
    assert "--all -y" not in calls


def test_conda_update_skips_without_an_ownership_answer() -> None:
    """No ownership lines (a failed lookup) is not evidence that conda owns anything."""
    function = _update_script_function("conda_update_all_decision")
    result = subprocess.run(  # noqa: S603 - fixed /bin/bash evaluates an extracted repo function
        ["/bin/bash", "-c", f'{function}\nconda_update_all_decision 0 ""'],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "skip"


# ---------------------------------------------------------------------------
# Commit hygiene: direct runs must not widen what the user staged
# ---------------------------------------------------------------------------


def _hygiene_git(repo: Path, *args: str) -> str:
    return subprocess.run(  # noqa: S603 - fixed git command on a test-created repo
        ["git", "-C", str(repo), *args],  # noqa: S607 - git from PATH
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _hygiene_repo(tmp_path: Path) -> Path:
    """A disposable repo carrying the hygiene script, with one committed baseline."""
    repo = tmp_path / "repo"
    tools_dir = repo / "src" / "tools"
    tools_dir.mkdir(parents=True)
    for name in ("run_commit_hygiene.sh", "common_quality.sh"):
        shutil.copy2(PKG_ROOT / "tools" / name, tools_dir / name)
    _hygiene_git(tmp_path, "init", "-q", str(repo))
    _hygiene_git(repo, "config", "user.email", "test@example.com")
    _hygiene_git(repo, "config", "user.name", "Test")
    safe_io.write_text_no_follow(repo / "src" / "app.py", "value = 0\n")
    safe_io.write_text_no_follow(repo / "src" / "notes.md", "# Notes\n\nOld.\n")
    safe_io.write_text_no_follow(repo / "src" / "README.md", "# Readme\n")
    safe_io.write_text_no_follow(repo / "src" / "pyproject.toml", '[project]\nname = "x"\n')
    _hygiene_git(repo, "add", "-A")
    _hygiene_git(repo, "commit", "-q", "-m", "baseline")
    return repo


def _run_hygiene(repo: Path) -> subprocess.CompletedProcess[str]:
    # The fallback resolves $CONDA_PREFIX/bin/python, i.e. this test's interpreter.
    return subprocess.run(  # noqa: S603 - fixed bash runs the copied repo script
        ["/bin/bash", str(repo / "src" / "tools" / "run_commit_hygiene.sh")],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo,
        env={
            **os.environ,
            "CONDA_ENV": "no-such-env",
            "CONDA_PREFIX": str(Path(sys.executable).parent.parent),
            "QUALITY_ALLOW_PYTHON_FALLBACK": "1",
        },
    )


def test_direct_hygiene_run_still_fixes_fully_staged_files(tmp_path: Path) -> None:
    """With nothing unstaged, the fixer's change is staged as before."""
    repo = _hygiene_repo(tmp_path)
    safe_io.write_text_no_follow(repo / "src" / "app.py", "value=1\n")
    _hygiene_git(repo, "add", "src/app.py")

    result = _run_hygiene(repo)

    assert result.returncode == 0, result.stdout + result.stderr
    assert _hygiene_git(repo, "show", ":src/app.py") == "value = 1\n"
    assert _hygiene_git(repo, "diff", "--name-only") == ""


@pytest.mark.parametrize(
    ("path", "staged", "unstaged"),
    [
        pytest.param("src/app.py", "value = 1\n", "value = 2\n", id="python"),
        pytest.param(
            "src/notes.md", "# Notes\n\nStaged.\n", "# Notes\n\nUnstaged.\n", id="markdown"
        ),
    ],
)
def test_direct_hygiene_run_preserves_partial_staging(
    tmp_path: Path, path: str, staged: str, unstaged: str
) -> None:
    """Regression: a staged ``value = 1`` became ``value = 2`` from an unstaged edit."""
    repo = _hygiene_repo(tmp_path)
    safe_io.write_text_no_follow(repo / path, staged)
    _hygiene_git(repo, "add", path)
    safe_io.write_text_no_follow(repo / path, unstaged)

    result = _run_hygiene(repo)

    assert result.returncode == 1
    assert f"   - {path}" in result.stdout
    assert _hygiene_git(repo, "show", f":{path}") == staged
    assert safe_io.read_text_no_follow(repo / path) == unstaged


def test_direct_hygiene_run_does_not_stage_unstaged_readme_edits(tmp_path: Path) -> None:
    """README sync regenerates and stages src/README.md, so its own edits must be staged first."""
    repo = _hygiene_repo(tmp_path)
    safe_io.write_text_no_follow(
        repo / "src" / "pyproject.toml", '[project]\nname = "x"\nversion = "1"\n'
    )
    _hygiene_git(repo, "add", "src/pyproject.toml")
    safe_io.write_text_no_follow(repo / "src" / "README.md", "# Readme\n\nUnstaged draft.\n")

    result = _run_hygiene(repo)

    assert result.returncode == 1
    assert "src/README.md (regenerated from the staged src/pyproject.toml)" in result.stdout
    assert _hygiene_git(repo, "diff", "--cached", "--name-only") == "src/pyproject.toml\n"
    assert (
        safe_io.read_text_no_follow(repo / "src" / "README.md") == "# Readme\n\nUnstaged draft.\n"
    )


# ---------------------------------------------------------------------------
# bootstrap-dev: one hook install, and success means the environment validated
# ---------------------------------------------------------------------------


def _run_bootstrap(
    tmp_path: Path, *, validation_status: int
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    log = tmp_path / "python-calls.log"
    fake_python = tmp_path / "fake-python"
    safe_io.write_text_no_follow(
        fake_python,
        "#!/bin/bash\n"
        f'echo "$*" >> "{log}"\n'
        f'if [ "$*" = "-m tools.validate_env" ]; then exit {validation_status}; fi\n',
        mode=0o755,
    )
    result = subprocess.run(  # noqa: S603 - fixed make target with a fake interpreter
        ["make", "-s", "-C", str(PKG_ROOT), "bootstrap-dev", f"PYTHON={fake_python}"],  # noqa: S607 - make from PATH
        capture_output=True,
        text=True,
        check=False,
        # A venv selects RUN_PY=$(PYTHON); no npm on PATH makes markdownlint the optional omission.
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "VIRTUAL_ENV": str(tmp_path / "venv")},
    )
    return result, safe_io.read_text_no_follow(log).splitlines()


def test_bootstrap_fails_when_environment_validation_fails(tmp_path: Path) -> None:
    """Required validation failing must fail the target, never print "Bootstrap complete"."""
    result, calls = _run_bootstrap(tmp_path, validation_status=1)
    assert result.returncode != 0
    assert "Bootstrap failed" in result.stdout
    assert "Bootstrap complete" not in result.stdout
    assert calls.count("-m pre_commit install") == 1


def test_bootstrap_reports_optional_omissions_on_success(tmp_path: Path) -> None:
    """Optional steps that were skipped are summarised after a successful bootstrap."""
    result, calls = _run_bootstrap(tmp_path, validation_status=0)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Bootstrap complete, with an optional step skipped:" in result.stdout
    assert "npm not found" in result.stdout
    assert calls.count("-m pre_commit install") == 1
