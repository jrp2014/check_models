"""Test that runtime dependency block in README matches pyproject runtime deps.

This enforces local parity in addition to CI. The test focuses only on runtime deps
(not optional extras groups) and parses dependencies with packaging's PEP 508 parser.
"""

from __future__ import annotations

import ast
import json
import logging
import re
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path
from textwrap import dedent

import pytest
import yaml
from packaging.requirements import Requirement

from check_models_data import dependency_policy
from tools import (
    bugtest,
    check_suppressions,
    generate_stubs,
    install_precommit_hook,
    safe_io,
    update_readme_deps,
)

_TEST_FILE = Path(__file__).resolve()
# tests/ parent, then package root (vlm)
PKG_ROOT = _TEST_FILE.parents[1]
REPO_ROOT = PKG_ROOT.parent


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
AGENT_QUALITY_WORKFLOW = REPO_ROOT / ".agents" / "workflows" / "quality.md"
SKYLOS_DANGER_ADVISORY_SCRIPT = PKG_ROOT / "tools" / "run_skylos_danger_advisory.sh"
SKYLOS_VERIFY_SCRIPT = PKG_ROOT / "tools" / "run_skylos_verify.sh"

SKYLOS_ADVISORY_QUALITY_IGNORES = {
    "SKY-C303",
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
    "SKY-U005",
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
    "mlx-lm": "mlx_lm",
    "mlx-vlm": "mlx_vlm",
    "pillow": "PIL",
    "pyyaml": "yaml",
}


def _dependency_key(requirement: Requirement) -> str:
    extras = f"[{','.join(sorted(requirement.extras))}]" if requirement.extras else ""
    return f"{requirement.name}{extras}"


def _dependency_spec(requirement: Requirement) -> str:
    marker = f"; {requirement.marker}" if requirement.marker else ""
    return f"{requirement.specifier}{marker}"


def _parse_runtime_deps(text: str) -> dict[str, str]:
    data = tomllib.loads(text)
    # project.dependencies is a list of strings
    deps_list = data.get("project", {}).get("dependencies", [])

    deps: dict[str, str] = {}
    for line in deps_list:
        requirement = Requirement(line)
        deps[_dependency_key(requirement)] = _dependency_spec(requirement)
    return deps


def _extract_manual_block(readme: str) -> str:
    start, end = MANUAL_MARKERS
    pattern = re.compile(rf"{re.escape(start)}(.*?){re.escape(end)}", re.DOTALL)
    m = pattern.search(readme)
    if not m:
        msg = "Manual install markers not found in README.md"
        raise RuntimeError(msg)
    return m.group(1)


def _write_stub_manifest(
    typings_dir: Path,
    *,
    mlx_vlm_version: str,
    stubgen_version: str,
) -> None:
    manifest_path = typings_dir / generate_stubs.STUB_MANIFEST
    safe_io.write_text_no_follow(
        manifest_path,
        json.dumps(
            {
                "packages": {
                    "mlx_vlm": {
                        "distribution": "mlx-vlm",
                        "version": mlx_vlm_version,
                    },
                },
                "tool_version": generate_stubs.STUB_TOOL_VERSION,
                "python_version": generate_stubs._python_version(),
                "stubgen_version": stubgen_version,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


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
    py_text = PYPROJECT.read_text(encoding="utf-8")
    rd_text = README.read_text(encoding="utf-8")
    runtime_deps = _parse_runtime_deps(py_text)

    manual_block = _extract_manual_block(rd_text)
    # Pull quoted specs from pip install line
    quoted = re.findall(r'"([^"]+)"', manual_block)
    if not quoted:
        msg = "No quoted packages found in manual install block"
        raise RuntimeError(msg)

    seen: dict[str, str] = {}
    for q in quoted:
        requirement = Requirement(q)
        seen[_dependency_key(requirement)] = _dependency_spec(requirement)

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
    assert f"mlx-lm>={dependency_policy.PROJECT_RUNTIME_STACK_MINIMUMS['mlx-lm']}" in runtime_deps
    assert f"mlx-vlm>={dependency_policy.PROJECT_RUNTIME_STACK_MINIMUMS['mlx-vlm']}" in runtime_deps
    assert f"transformers{dependency_policy.PROJECT_TRANSFORMERS_VERSION_SPEC}" in runtime_deps
    assert (
        f"huggingface-hub[torch,typing]>={dependency_policy.PROJECT_RUNTIME_STACK_MINIMUMS['huggingface-hub']}"
        in runtime_deps
    )
    assert not any(dep.startswith("mlx-lm>=") for dep in extras_deps)


def test_dependency_policy_tracks_current_upstream_transformers_floor() -> None:
    """Project and upstream mlx-lm policy should share the current Transformers floor."""
    assert dependency_policy.PROJECT_RUNTIME_STACK_MINIMUMS["transformers"] == "5.7.0"
    assert dependency_policy.UPSTREAM_MLX_LM_MINIMUMS["transformers"] == "5.7.0"


def test_ty_uses_generated_typings_search_path() -> None:
    """Ensure ty resolves repo-local generated stubs like mlx_vlm.*."""
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    ty_env = pyproject["tool"]["ty"]["environment"]
    assert ty_env["extra-paths"] == ["../typings"]


def test_type_checkers_use_repo_root_generated_typings() -> None:
    """All type checkers should resolve stubs generated at repo-root typings/."""
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))

    assert pyproject["tool"]["mypy"]["mypy_path"] == "../typings"
    assert pyproject["tool"]["ty"]["environment"]["extra-paths"] == ["../typings"]
    assert pyproject["tool"]["pyrefly"]["search-path"] == ["../typings"]

    generate_stubs_source = (PKG_ROOT / "tools" / "generate_stubs.py").read_text(encoding="utf-8")
    assert 'mypy_path = ["../typings"]' in generate_stubs_source


def test_mypy_uses_generated_typings_without_gating_stub_internals() -> None:
    """Generated third-party stubs should inform call sites, not fail strict checks."""
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    overrides = pyproject["tool"]["mypy"]["overrides"]

    generated_overrides = [
        override
        for override in overrides
        if set(override.get("module", []))
        & {
            "mlx_lm",
            "mlx_lm.*",
            "mlx_vlm",
            "mlx_vlm.*",
            "tokenizers",
            "tokenizers.*",
            "transformers",
            "transformers.*",
        }
    ]

    assert generated_overrides
    for override in generated_overrides:
        assert override["ignore_errors"] is True


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
        "CI uses lock files",
        "compatible with lock files",
        "make sync-deps",
        "make upgrade-deps",
        "src/requirements.txt",
        "requirements-dev.txt",
        "make -C vlm",
    )

    for doc_name, text in live_docs.items():
        for phrase in removed_phrases:
            assert phrase not in text, (doc_name, phrase)


def test_root_readme_describes_supported_cache_filter() -> None:
    """The quick README should match the detailed supported-cache discovery docs."""
    root_readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "server-supported cache filter" in root_readme
    assert "all models found in your local HF cache" not in root_readme


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
        "max_quality": 10000,
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
    } <= set(root_config["exclude"])
    assert set(root_config["ignore"]) == set(package_config["ignore"])
    for key in SKYLOS_MONOLITH_QUALITY_LIMITS:
        assert root_config[key] == package_config[key]
    assert root_config["gate"] == package_config["gate"]


def test_pydantic_is_managed_as_a_dev_dependency() -> None:
    """Keep pydantic in the managed dev dependency set and setup fallback."""
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    dev_deps = pyproject["project"]["optional-dependencies"]["dev"]

    assert "pydantic>=2.0.0" in dev_deps

    setup_script = (PKG_ROOT / "tools" / "setup_conda_env.sh").read_text(encoding="utf-8")
    assert '"pydantic>=2.0.0"' in setup_script


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
    """Avoid installing removed huggingface-hub extras during fresh setup."""
    setup_script = (PKG_ROOT / "tools" / "setup_conda_env.sh").read_text(encoding="utf-8")

    assert '"huggingface_hub[cli]"' not in setup_script
    assert "command -v hf" in setup_script


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

    result = subprocess.run(  # noqa: S603
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

    result = subprocess.run(  # noqa: S603
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

    result = subprocess.run(  # noqa: S603
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

    # Fixed local command against the checked-in repo; no user input reaches subprocess.
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(dist_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=PKG_ROOT,
    )

    assert result.returncode == 0, result.stderr or result.stdout

    wheel_paths = sorted(dist_dir.glob("check_models-*.whl"))
    assert len(wheel_paths) == 1

    with zipfile.ZipFile(wheel_paths[0]) as archive:
        assert "check_models_data/quality_config.yaml" in archive.namelist()


def test_markdownlint_cli2_is_pinned_repo_local_and_updateable() -> None:
    """Keep markdownlint-cli2 aligned between npm metadata and update tooling."""
    package_json = json.loads((PKG_ROOT / "package.json").read_text(encoding="utf-8"))
    package_lock = json.loads((PKG_ROOT / "package-lock.json").read_text(encoding="utf-8"))

    markdownlint_spec = package_json["devDependencies"]["markdownlint-cli2"]
    assert markdownlint_spec == "^0.23.0"
    assert markdownlint_spec == package_lock["packages"][""]["devDependencies"]["markdownlint-cli2"]
    assert package_lock["packages"]["node_modules/markdownlint-cli2"]["version"] == "0.23.0"
    assert package_json["overrides"]["smol-toml"] == "1.6.1"
    assert package_lock["packages"]["node_modules/smol-toml"]["version"] == "1.6.1"

    update_script = (PKG_ROOT / "tools" / "update.sh").read_text(encoding="utf-8")
    assert 'npm install --ignore-scripts --prefix "$PROJECT_ROOT"' in update_script
    assert (
        'npm install --prefix "$PROJECT_ROOT" --save-dev markdownlint-cli2@latest' in update_script
    )
    assert update_script.index("markdownlint-cli2@latest") > update_script.index(
        "UPDATE_NODE_TOOLING"
    )


def test_generated_markdown_lint_guards_use_named_rule_sets() -> None:
    """Report-specific markdownlint guards should stay centralized."""
    source = (PKG_ROOT / "check_models.py").read_text(encoding="utf-8")

    assert 'MARKDOWNLINT_MAIN_TABLE_RULES: Final[str] = "MD033 MD034 MD037 MD049"' in source
    assert 'MARKDOWNLINT_GALLERY_SUMMARY_RULES: Final[str] = "MD034"' in source
    assert 'MARKDOWNLINT_TABLE_PIPE_RULES: Final[str] = "MD060"' in source
    assert "<!-- markdownlint-disable MD033 MD034 MD037 MD049 -->" not in source
    assert "<!-- markdownlint-enable MD033 MD034 MD037 MD049 -->" not in source
    assert "<!-- markdownlint-disable MD034 -->" not in source
    assert "<!-- markdownlint-enable MD034 -->" not in source


def test_agent_quality_guidance_avoids_redundant_pytest_after_quality() -> None:
    """Agent-facing workflow docs should treat make quality as the full pytest gate."""
    copilot_text = COPILOT_INSTRUCTIONS.read_text(encoding="utf-8")
    quality_workflow = AGENT_QUALITY_WORKFLOW.read_text(encoding="utf-8")

    assert "full pytest" in copilot_text
    assert "Do not run it again after a successful `make quality`" in copilot_text
    assert "make skylos-danger" in copilot_text
    assert "make skylos-danger-llm" in copilot_text
    assert "make skylos-verify" in copilot_text
    assert "could be promoted later" in copilot_text
    assert "`make quality` already runs the full pytest suite" in quality_workflow
    assert "make skylos-danger-llm" in quality_workflow
    assert "make skylos-verify" in quality_workflow
    assert "`make test` — execute unit tests" not in copilot_text


def test_output_artifact_policy_is_documented_and_gitignored() -> None:
    """Generated output docs should match the repo ignore policy."""
    output_readme = (PKG_ROOT / "output" / "README.md").read_text(encoding="utf-8")
    gitignore_lines = {
        line.strip() for line in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    }

    for phrase in (
        "production Markdown reports are linted by the quality gate",
        "use the `test_` prefix",
        "do not commit ad-hoc debug output",
    ):
        assert phrase in output_readme

    assert {
        "src/output/test_*",
        "src/output/reports/test_*",
        "src/output/issues/test_*",
        "src/output/repro_bundles/test_*",
    }.issubset(gitignore_lines)


def test_validation_artifact_hygiene_policy_is_documented() -> None:
    """Validation guidance should forbid dirtying tracked benchmark assets."""
    required_phrase = "Validation tests must not rewrite tracked `src/output/` assets"

    docs = {
        "copilot instructions": COPILOT_INSTRUCTIONS.read_text(encoding="utf-8"),
        "agent instructions": (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8"),
        "claude instructions": (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8"),
        "quality workflow": (REPO_ROOT / ".github" / "workflows" / "quality.yml").read_text(
            encoding="utf-8"
        ),
        "contributor guide": (REPO_ROOT / "docs" / "CONTRIBUTING.md").read_text(encoding="utf-8"),
        "cli readme": (PKG_ROOT / "README.md").read_text(encoding="utf-8"),
    }

    for label, text in docs.items():
        assert required_phrase in text, label

    for label, text in docs.items():
        assert "temp directory" in text or "`test_*`" in text, label


def test_stub_refresh_reason_is_none_for_fresh_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    versions = {"mlx-vlm": "0.31.0", "mypy": "1.18.0"}

    def _installed_version(distribution: str) -> str | None:
        return versions.get(distribution)

    typings_dir = tmp_path / "typings"
    (typings_dir / "mlx_vlm").mkdir(parents=True)
    _write_stub_manifest(typings_dir, mlx_vlm_version="0.31.0", stubgen_version="1.18.0")

    monkeypatch.setattr(
        generate_stubs,
        "_installed_distribution_version",
        _installed_version,
    )

    assert generate_stubs.get_stub_refresh_reason(["mlx_vlm"], typings_dir) is None


def test_stub_refresh_reason_detects_version_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    versions = {"mlx-vlm": "0.31.0", "mypy": "1.18.0"}

    def _installed_version(distribution: str) -> str | None:
        return versions.get(distribution)

    typings_dir = tmp_path / "typings"
    (typings_dir / "mlx_vlm").mkdir(parents=True)
    _write_stub_manifest(typings_dir, mlx_vlm_version="0.30.0", stubgen_version="1.18.0")

    monkeypatch.setattr(
        generate_stubs,
        "_installed_distribution_version",
        _installed_version,
    )

    assert (
        generate_stubs.get_stub_refresh_reason(["mlx_vlm"], typings_dir)
        == "mlx_vlm version metadata changed"
    )


def test_stub_integrity_issues_detect_missing_mlx_vlm_contract_markers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stub integrity checks should fail loudly when required mlx_vlm patches are absent."""
    versions = {"mlx-vlm": "0.31.0", "mypy": "1.18.0"}

    def _installed_version(distribution: str) -> str | None:
        return versions.get(distribution)

    typings_dir = tmp_path / "typings"
    mlx_vlm_dir = typings_dir / "mlx_vlm"
    mlx_vlm_dir.mkdir(parents=True)
    (mlx_vlm_dir / "generate.pyi").write_text(
        "from transformers import PreTrainedTokenizer as PreTrainedTokenizer\n"
        "def generate(model: object, processor: PreTrainedTokenizer, prompt: str) -> object: ...\n",
        encoding="utf-8",
    )
    _write_stub_manifest(typings_dir, mlx_vlm_version="0.31.0", stubgen_version="1.18.0")

    monkeypatch.setattr(generate_stubs, "_installed_distribution_version", _installed_version)

    issues = generate_stubs.get_stub_integrity_issues(["mlx_vlm"], typings_dir)
    assert any(
        "mlx_vlm generate stub is missing patched runtime-contract markers" in issue
        for issue in issues
    )


def test_stub_integrity_issues_accept_package_layout_mlx_vlm_generate_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """mlx_vlm integrity checks should accept package-layout generate dispatch stubs."""
    versions = {"mlx-vlm": "0.31.0", "mypy": "1.18.0"}

    def _installed_version(distribution: str) -> str | None:
        return versions.get(distribution)

    typings_dir = tmp_path / "typings"
    dispatch_dir = typings_dir / "mlx_vlm" / "generate"
    dispatch_dir.mkdir(parents=True)
    (dispatch_dir / "dispatch.pyi").write_text(
        "import mlx.nn as nn\n"
        "from transformers import PreTrainedTokenizer as PreTrainedTokenizer\n"
        "from transformers.processing_utils import ProcessorMixin as ProcessorMixin\n"
        "from typing import Generator\n"
        "\n"
        "def stream_generate(model: nn.Module, processor: ProcessorMixin | PreTrainedTokenizer, "
        "prompt: str, image: str | list[str] | None = None, "
        "audio: str | list[str] | None = None, video: str | list[str] | None = None, "
        "**kwargs) -> Generator[GenerationResult, None, None]: ...\n"
        "def generate(model: nn.Module, processor: ProcessorMixin | PreTrainedTokenizer, "
        "prompt: str, image: str | list[str] | None = None, "
        "audio: str | list[str] | None = None, video: str | list[str] | None = None, "
        "verbose: bool = False, *, max_tokens: int = ..., temperature: float = ..., "
        "thinking_end_token: str = ..., thinking_start_token: str | None = None, "
        "**kwargs) -> GenerationResult: ...\n",
        encoding="utf-8",
    )
    _write_stub_manifest(typings_dir, mlx_vlm_version="0.31.0", stubgen_version="1.18.0")

    monkeypatch.setattr(generate_stubs, "_installed_distribution_version", _installed_version)

    issues = generate_stubs.get_stub_integrity_issues(["mlx_vlm"], typings_dir)
    assert issues == []


def test_refresh_stub_manifest_from_existing_stubs_repairs_version_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Version-only manifest drift should be repairable from verified local stubs."""
    versions = {"tokenizers": "0.22.2", "mypy": "1.18.0"}

    def _installed_version(distribution: str) -> str | None:
        return versions.get(distribution)

    typings_dir = tmp_path / "typings"
    tokenizers_dir = typings_dir / "tokenizers"
    tokenizers_dir.mkdir(parents=True)
    (tokenizers_dir / "__init__.pyi").write_text("class Encoding: ...\n", encoding="utf-8")
    manifest_path = typings_dir / generate_stubs.STUB_MANIFEST
    manifest_path.write_text(
        json.dumps(
            {
                "packages": {
                    "tokenizers": {
                        "distribution": "tokenizers",
                        "version": "0.22.1",
                    },
                },
                "tool_version": generate_stubs.STUB_TOOL_VERSION,
                "python_version": generate_stubs._python_version(),
                "stubgen_version": "1.18.0",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(generate_stubs, "_installed_distribution_version", _installed_version)

    assert generate_stubs.refresh_stub_manifest_from_existing_stubs(["tokenizers"], typings_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["packages"]["tokenizers"]["version"] == "0.22.2"


def test_patch_transformers_stubs_adds_processor_runtime_attrs(tmp_path: Path) -> None:
    """Patched ProcessorMixin stubs should expose tokenizer/image processor attrs."""
    typings_dir = tmp_path / "typings"
    transformers_dir = typings_dir / "transformers"
    transformers_dir.mkdir(parents=True)
    processing_utils_path = transformers_dir / "processing_utils.pyi"
    processing_utils_path.write_text(
        "from .tokenization_utils_base import PreTrainedTokenizerBase as PreTrainedTokenizerBase\n"
        "from _typeshed import Incomplete\n"
        "from typing import Any\n"
        "\n"
        "class PushToHubMixin: ...\n"
        "\n"
        "class ProcessorMixin(PushToHubMixin):\n"
        "    valid_processor_kwargs = object\n"
        "    image_ids: Incomplete\n"
        "    video_ids: Incomplete\n"
        "    audio_ids: Incomplete\n",
        encoding="utf-8",
    )

    generate_stubs._patch_transformers_stubs(typings_dir)

    patched = processing_utils_path.read_text(encoding="utf-8")
    assert "tokenizer: PreTrainedTokenizerBase | None" in patched
    assert "image_processor: Any | None" in patched


def test_patch_transformers_stubs_widens_existing_processor_runtime_attrs(
    tmp_path: Path,
) -> None:
    """Current upstream ProcessorMixin attrs should be widened without duplication."""
    typings_dir = tmp_path / "typings"
    transformers_dir = typings_dir / "transformers"
    transformers_dir.mkdir(parents=True)
    processing_utils_path = transformers_dir / "processing_utils.pyi"
    processing_utils_path.write_text(
        "from .tokenization_utils_base import PreTrainedTokenizerBase as PreTrainedTokenizerBase\n"
        "from typing import Any\n"
        "\n"
        "class PushToHubMixin: ...\n"
        "\n"
        "class ProcessorMixin(PushToHubMixin):\n"
        "    tokenizer: Any\n"
        "    feature_extractor: Any\n"
        "    image_processor: Any\n"
        "    video_processor: Any\n",
        encoding="utf-8",
    )

    generate_stubs._patch_transformers_stubs(typings_dir)
    generate_stubs._patch_transformers_stubs(typings_dir)

    patched = processing_utils_path.read_text(encoding="utf-8")
    assert patched.count("    tokenizer: PreTrainedTokenizerBase | None\n") == 1
    assert patched.count("    image_processor: Any | None\n") == 1
    assert "    tokenizer: Any\n" not in patched
    assert "    image_processor: Any\n" not in patched


def test_patch_mlx_vlm_stubs_widens_generate_processor_type(tmp_path: Path) -> None:
    """Patched mlx_vlm stubs should accept ProcessorMixin processors."""
    typings_dir = tmp_path / "typings"
    mlx_vlm_dir = typings_dir / "mlx_vlm"
    mlx_vlm_dir.mkdir(parents=True)
    generate_path = mlx_vlm_dir / "generate.pyi"
    generate_path.write_text(
        "import mlx.nn as nn\n"
        "from transformers import PreTrainedTokenizer as PreTrainedTokenizer\n"
        "from transformers.processing_utils import ProcessorMixin as ProcessorMixin\n"
        "\n"
        "def generate(model: nn.Module, processor: ProcessorMixin | PreTrainedTokenizer, "
        "prompt: str, image: str | list[str] | None = None, "
        "audio: str | list[str] | None = None, verbose: bool = False, **kwargs) "
        "-> GenerationResult: ...\n",
        encoding="utf-8",
    )

    generate_stubs._patch_mlx_vlm_stubs(typings_dir)

    patched = generate_path.read_text(encoding="utf-8")
    assert "from transformers.processing_utils import ProcessorMixin as ProcessorMixin\n" in patched
    assert "processor: ProcessorMixin | PreTrainedTokenizer" in patched
    assert "video: str | list[str] | None = None" in patched
    assert "temperature: float = ..." in patched
    assert "kv_bits: float | None = None" in patched
    assert "kv_quant_scheme: str = ..." in patched
    assert "thinking_end_token: str = ..." in patched


def test_patch_mlx_vlm_stubs_widens_package_layout_generate_types(tmp_path: Path) -> None:
    """Package-layout mlx_vlm generate stubs should receive the same contract patches."""
    typings_dir = tmp_path / "typings"
    generate_dir = typings_dir / "mlx_vlm" / "generate"
    generate_dir.mkdir(parents=True)
    legacy_generate_path = typings_dir / "mlx_vlm" / "generate.pyi"
    legacy_generate_path.write_text(
        "def generate(*args, **kwargs) -> object: ...\n", encoding="utf-8"
    )

    dispatch_path = generate_dir / "dispatch.pyi"
    dispatch_path.write_text(
        "import mlx.nn as nn\n"
        "from transformers import PreTrainedTokenizer as PreTrainedTokenizer\n"
        "from typing import Generator\n"
        "\n"
        "def stream_generate(model: nn.Module, processor: PreTrainedTokenizer, "
        "prompt: str, image: str | list[str] = None, audio: str | list[str] = None, "
        "video: str | list[str] = None, **kwargs) -> str | Generator[str, None, None]: ...\n"
        "def generate(model: nn.Module, processor: PreTrainedTokenizer, prompt: str, "
        "image: str | list[str] = None, audio: str | list[str] = None, "
        "video: str | list[str] = None, verbose: bool = False, **kwargs) "
        "-> GenerationResult: ...\n",
        encoding="utf-8",
    )
    ar_path = generate_dir / "ar.pyi"
    ar_path.write_text(
        "def batch_generate(model, processor, images: str | list[str] = None, "
        "audios: str | list[str] = None, prompts: list[str] = None, **kwargs): ...\n",
        encoding="utf-8",
    )

    generate_stubs._patch_mlx_vlm_stubs(typings_dir)

    patched_dispatch = dispatch_path.read_text(encoding="utf-8")
    assert (
        "from transformers.processing_utils import ProcessorMixin as ProcessorMixin\n"
        in patched_dispatch
    )
    assert "processor: ProcessorMixin | PreTrainedTokenizer" in patched_dispatch
    assert "video: str | list[str] | None = None" in patched_dispatch
    assert "temperature: float = ..." in patched_dispatch
    assert "thinking_end_token: str = ..." in patched_dispatch

    patched_ar = ar_path.read_text(encoding="utf-8")
    assert "images: str | list[str] | None = None" in patched_ar
    assert "audios: str | list[str] | None = None" in patched_ar
    assert "prompts: list[str] | None = None" in patched_ar
    assert not legacy_generate_path.exists()


def test_patch_stub_file_applies_replacements_and_reports_change(tmp_path: Path) -> None:
    """Shared stub patch helper should report whether a file changed."""
    stub_path = tmp_path / "sample.pyi"
    stub_path.write_text("def f(value: str = None) -> None: ...\n", encoding="utf-8")

    changed = generate_stubs._patch_stub_file(
        stub_path,
        [(re.compile(r"(value:\s*str)\s*=\s*None"), r"\1 | None = None")],
    )

    assert changed is True
    assert "value: str | None = None" in stub_path.read_text(encoding="utf-8")
    assert generate_stubs._patch_stub_file(stub_path, []) is False


def test_stub_patch_audit_reports_named_patch_that_no_longer_changes_raw_stub(
    tmp_path: Path,
) -> None:
    """Patch audits should flag stale shims when raw stubs already have the fix."""
    typings_dir = tmp_path / "typings"
    package_dir = typings_dir / "pkg"
    package_dir.mkdir(parents=True)
    stub_path = package_dir / "sample.pyi"
    stub_path.write_text("def f(value: str | None = None) -> None: ...\n", encoding="utf-8")

    issues = generate_stubs._patch_stub_file_in_typings(
        typings_dir,
        stub_path,
        [
            generate_stubs._stub_patch(
                "optional value default",
                re.compile(r"(value:\s*str)\s*=\s*None"),
                r"\1 | None = None",
            ),
        ],
        audit=True,
    )

    assert issues == ["pkg/sample.pyi: patch 'optional value default' did not change the raw stub"]


def test_stub_patch_audit_accepts_group_when_one_alternative_changes(
    tmp_path: Path,
) -> None:
    """Alternative patch patterns should audit as one semantic shim."""
    typings_dir = tmp_path / "typings"
    package_dir = typings_dir / "pkg"
    package_dir.mkdir(parents=True)
    stub_path = package_dir / "sample.pyi"
    stub_path.write_text("class Processor:\n    tokenizer: Any\n", encoding="utf-8")

    issues = generate_stubs._patch_stub_file_in_typings(
        typings_dir,
        stub_path,
        [
            generate_stubs._stub_patch_group(
                "processor runtime attrs",
                [
                    (re.compile(r"(    tokenizer:) Any$"), r"\1 object | None"),
                    (re.compile(r"(    image_processor:) Any$"), r"\1 object | None"),
                ],
            ),
        ],
        audit=True,
    )

    assert issues == []
    assert "tokenizer: object | None" in stub_path.read_text(encoding="utf-8")


def test_patch_stub_file_rejects_symlink_target(tmp_path: Path) -> None:
    """Stub patching should not follow a symlink target."""
    target_path = tmp_path / "target.pyi"
    target_path.write_text("def f(value: str = None) -> None: ...\n", encoding="utf-8")
    stub_path = tmp_path / "sample.pyi"
    stub_path.symlink_to(target_path)

    with pytest.raises(OSError, match="symlink"):
        generate_stubs._patch_stub_file(
            stub_path,
            [(re.compile(r"(value:\s*str)\s*=\s*None"), r"\1 | None = None")],
        )

    assert target_path.read_text(encoding="utf-8") == "def f(value: str = None) -> None: ...\n"


def test_write_stub_manifest_rejects_symlink_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Stub manifest writes should not follow a symlink target."""
    versions = {"tokenizers": "0.22.2", "mypy": "1.18.0"}

    def _installed_version(distribution: str) -> str | None:
        return versions.get(distribution)

    typings_dir = tmp_path / "typings"
    typings_dir.mkdir()
    target_path = tmp_path / "target-manifest.json"
    manifest_path = typings_dir / generate_stubs.STUB_MANIFEST
    manifest_path.symlink_to(target_path)
    monkeypatch.setattr(generate_stubs, "_installed_distribution_version", _installed_version)
    caplog.set_level(logging.WARNING, logger="generate_stubs")

    generate_stubs._write_stub_manifest(["tokenizers"], typings_dir)

    assert not target_path.exists()
    assert "Refusing to follow symlink" in caplog.text


def test_install_hook_rejects_symlink_target(tmp_path: Path) -> None:
    """Hook installation should not write through a symlink target."""
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    target_path = tmp_path / "target-hook"
    hook_path = hooks_dir / "pre-commit"
    hook_path.symlink_to(target_path)

    with pytest.raises(OSError, match="symlink"):
        install_precommit_hook._install_hook(hooks_dir, "pre-commit", "#!/usr/bin/env bash\n")

    assert not target_path.exists()


def test_update_script_verifies_stub_integrity_and_logs_local_provenance() -> None:
    """Local update tooling should verify stub contracts and log editable provenance."""
    update_script = (PKG_ROOT / "tools" / "update.sh").read_text(encoding="utf-8")
    quality_script = (PKG_ROOT / "tools" / "run_quality_checks.sh").read_text(encoding="utf-8")

    assert (
        'run_generate_stubs_command "$SCRIPT_DIR" --check --refresh-manifest-on-check mlx_lm mlx_vlm transformers tokenizers'
        in update_script
    )
    assert (
        '"$QUALITY_PYTHON" -m tools.generate_stubs --check --refresh-manifest-on-check'
        in quality_script
    )
    assert "mlx_lm mlx_vlm transformers tokenizers" in quality_script
    assert "Local package provenance:" in update_script


def test_bugtest_formats_metal_regression_warning() -> None:
    """The Metal backend probe should provide a pasteable maintenance warning."""
    assert bugtest.classify_relative_error(0.001) == "ok"
    assert bugtest.classify_relative_error(1.11) == "buggy"

    message = bugtest.format_probe_message(
        status="buggy",
        relative_error=1.11,
        metal_version="Apple metal version 32023.883 (metalfe-32023.883)",
    )

    assert "still appears broken" in message
    assert "relative error 1.110000" in message
    assert "metalfe-32023.883" in message
    assert "MLX_METAL_GPU_ARCH=applegpu_g16s" in message


def test_update_script_runs_nonblocking_metal_bug_reminder() -> None:
    """update.sh should surface the Metal regression probe without blocking updates."""
    update_script = (PKG_ROOT / "tools" / "update.sh").read_text(encoding="utf-8")

    assert "run_metal_bug_reminder" in update_script
    assert 'python "$SCRIPT_DIR/bugtest.py" --warn-only' in update_script
    assert "MLX_METAL_BUG_REMINDER" in update_script


def test_quality_script_runs_skylos_quality_gate() -> None:
    """Local quality checks should include the calibrated Skylos quality gate."""
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    dev_deps = pyproject["project"]["optional-dependencies"]["dev"]
    quality_script = (PKG_ROOT / "tools" / "run_quality_checks.sh").read_text(encoding="utf-8")
    setup_script = (PKG_ROOT / "tools" / "setup_conda_env.sh").read_text(encoding="utf-8")

    assert "skylos>=4.27.0" in dev_deps
    assert '"skylos>=4.27.0"' in setup_script
    assert (
        'quality_require_python_tool skylos "Install dev dependencies with: pip install -e .[dev]"'
        in quality_script
    )
    assert 'echo "=== Skylos Quality Gate ==="' in quality_script
    assert "SKYLOS_JOBS" not in quality_script
    assert "--danger" not in quality_script
    assert re.search(
        r"TERM=dumb NO_COLOR=1 CLICOLOR=0 FORCE_COLOR=0 PY_COLORS=0\s+\\?\s*"
        r"quality_run_python_tool skylos \. --quality --secrets --sca --gate --no-upload "
        r"--format concise",
        quality_script,
    )


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


def test_skylos_verify_script_wraps_repo_context_verifier() -> None:
    """The Skylos verify helper should keep agent checks repo-scoped and deterministic."""
    script = SKYLOS_VERIFY_SCRIPT.read_text(encoding="utf-8")

    assert "Usage: bash tools/run_skylos_verify.sh" in script
    assert 'cd "$(quality_repo_root)"' in script
    assert "quality_require_python_tool skylos" in script
    assert 'quality_run_python_tool skylos verify . --project-context "$@"' in script


def test_tsv_output_tests_use_safe_text_reads_for_skylos_advisory_scan() -> None:
    """TSV fixtures should use bounded no-follow reads instead of suppressing Skylos."""
    test_source = (PKG_ROOT / "tests" / "test_tsv_output.py").read_text(encoding="utf-8")

    assert "from tools import safe_io" in test_source
    assert "safe_io.read_text_no_follow(path)" in test_source
    assert "path.read_text" not in test_source


def test_defusedxml_probe_avoids_unused_import_suppression() -> None:
    """The defusedxml availability probe should not require a dead import suppression."""
    check_models_source = (PKG_ROOT / "check_models.py").read_text(encoding="utf-8")
    defusedxml_probe = check_models_source[
        check_models_source.index("defusedxml is required") : check_models_source.index(
            "try:\n    import numpy as np",
        )
    ]

    assert 'find_spec("defusedxml.ElementTree")' in defusedxml_probe
    assert "import defusedxml.ElementTree" not in defusedxml_probe
    assert "F401" not in defusedxml_probe


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

    result = subprocess.run(  # noqa: S603
        [str(bash_path), str(run_script)],
        capture_output=True,
        text=True,
        check=False,
        cwd=PKG_ROOT,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "WARN synthetic warning [test-warning]" in result.stdout
    assert "Pyrefly emitted warnings; treat warnings as quality failures." in result.stdout


def test_update_script_uses_upstream_mlx_editable_dev_install() -> None:
    """Local MLX builds should follow upstream's editable dev install guidance."""
    update_script = (PKG_ROOT / "tools" / "update.sh").read_text(encoding="utf-8")
    contributing = (REPO_ROOT / "docs" / "CONTRIBUTING.md").read_text(encoding="utf-8")

    assert 'INSTALL_CMD=(pip_install_verbose -e ".[dev]")' in update_script
    assert "macOS SDK" in update_script
    assert "Apple Clang" in update_script
    assert "Native arm64 shell detected" in update_script
    assert "MLX_LOCAL_BUILD_SMOKE" in update_script
    assert "mlx.metallib" in update_script
    assert "MLX runtime backend provenance" in update_script
    assert "SKIP_TORCH=1 bash tools/update.sh" in contributing
    assert "# Skip PyTorch support" in contributing
    assert "MLX_LOCAL_BUILD_SMOKE=0" in contributing


def test_update_script_import_repair_hint_uses_distribution_names() -> None:
    """Postflight repair output should use pip package names, not import module names."""
    update_script = (PKG_ROOT / "tools" / "update.sh").read_text(encoding="utf-8")

    assert "IMPORT_TO_PIP_PACKAGE" in update_script
    assert "printf '%s\\n' \"mlx-lm\"" in update_script
    assert "printf '%s\\n' \"mlx-vlm\"" in update_script
    assert "REPAIR_PKGS" in update_script
    assert 'REPAIR_PKGS+=("$(IMPORT_TO_PIP_PACKAGE "$pkg")")' in update_script
    assert "Fix with: pip install ${REPAIR_PKGS[*]}" in update_script
    assert "Fix with: pip install ${MISSING_PKGS[*]}" not in update_script


def test_update_script_keeps_system_and_node_latest_updates_opt_in() -> None:
    """The unified updater should not upgrade system tooling unless requested."""
    update_script = (PKG_ROOT / "tools" / "update.sh").read_text(encoding="utf-8")

    assert "UPDATE_SYSTEM_PACKAGES" in update_script
    assert "UPDATE_NODE_TOOLING" in update_script
    assert 'if [[ "${UPDATE_SYSTEM_PACKAGES:-0}" == "1" ]]; then' in update_script
    assert 'if [[ "${UPDATE_NODE_TOOLING:-0}" == "1" ]]; then' in update_script
    assert "Skipping conda base/environment package updates" in update_script
    assert "Skipping Homebrew update/upgrade" in update_script
    assert "Installing repo-local markdownlint tooling from package-lock.json" in update_script

    package_latest = "markdownlint-cli2@latest"
    assert package_latest in update_script
    assert update_script.index(package_latest) > update_script.index("UPDATE_NODE_TOOLING")


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


def test_workflows_pin_actions_and_keep_skylos_danger_advisory_nonblocking() -> None:
    """Workflow security hardening and advisory Skylos danger wiring should stay in place."""
    action_ref_pattern = re.compile(r"^[^@]+@[0-9a-f]{40}$")

    for workflow_path in (
        REPO_ROOT / ".github" / "workflows" / "dependency-sync.yml",
        REPO_ROOT / ".github" / "workflows" / "quality.yml",
    ):
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
    assert "npm install --ignore-scripts --prefix src" in static_quality_install


def test_should_audit_path_excludes_generated_and_archived_paths(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    included = repo_root / "src" / "module.py"
    excluded_build = repo_root / "src" / "build" / "lib" / "check_models.py"
    excluded_conda = repo_root / ".conda" / "lib" / "python3.13" / "site.py"
    excluded_output = repo_root / "src" / "output" / "results.md"
    excluded_archived = repo_root / "src" / "tools" / ".archived" / "old.py"
    included.parent.mkdir(parents=True)
    excluded_build.parent.mkdir(parents=True)
    excluded_conda.parent.mkdir(parents=True)
    excluded_output.parent.mkdir(parents=True)
    excluded_archived.parent.mkdir(parents=True)
    included.write_text("print('ok')\n", encoding="utf-8")
    excluded_build.write_text("value = 1  # noqa: F401\n", encoding="utf-8")
    excluded_conda.write_text("value = 1  # noqa: F401\n", encoding="utf-8")
    excluded_output.write_text("<!-- markdownlint-disable MD028 -->\n", encoding="utf-8")
    excluded_archived.write_text("x = 1  # noqa: F841\n", encoding="utf-8")

    assert check_suppressions.should_audit_path(included, repo_root) is True
    assert check_suppressions.should_audit_path(excluded_build, repo_root) is False
    assert check_suppressions.should_audit_path(excluded_conda, repo_root) is False
    assert check_suppressions.should_audit_path(excluded_output, repo_root) is False
    assert check_suppressions.should_audit_path(excluded_archived, repo_root) is False


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
        check: bool,
        cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        assert capture_output is True
        assert text is True
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
