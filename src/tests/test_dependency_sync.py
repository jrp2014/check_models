"""Test that runtime dependency block in README matches pyproject runtime deps.

This enforces local parity in addition to CI. The test focuses only on runtime deps
(not optional extras groups) and uses the same parsing heuristics as the sync script.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

from tools import check_suppressions, generate_stubs

if TYPE_CHECKING:
    import pytest

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

MANUAL_MARKERS = ("<!-- MANUAL_INSTALL_START -->", "<!-- MANUAL_INSTALL_END -->")


def _parse_runtime_deps(text: str) -> dict[str, str]:
    data = tomllib.loads(text)
    # project.dependencies is a list of strings
    deps_list = data.get("project", {}).get("dependencies", [])

    deps: dict[str, str] = {}
    for line in deps_list:
        name = re.split(r"[<>=!~]", line, maxsplit=1)[0].strip()
        spec = line[len(name) :].strip()
        deps[name] = spec
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
    manifest_path.write_text(
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
        encoding="utf-8",
    )


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
        name = re.split(r"[<>=!~]", q, maxsplit=1)[0]
        spec = q[len(name) :]
        seen[name] = spec

    # All runtime deps must exist
    missing = [k for k in runtime_deps if k not in seen]
    if missing:
        msg = f"Runtime deps missing from README: {missing}"
        raise RuntimeError(msg)

    # No extras should leak (heuristic: check optional groups defined later if needed)
    forbidden = {
        "psutil",
        "tokenizers",
        "mlx-lm",
        "torch",
        "torchvision",
        "torchaudio",
    }
    leaked = sorted(forbidden & set(seen))
    if leaked:
        msg = f"Optional deps leaked into runtime block: {leaked}"
        raise RuntimeError(msg)


def test_ty_uses_generated_typings_search_path() -> None:
    """Ensure ty resolves repo-local generated stubs like mlx_vlm.*."""
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    ty_env = pyproject["tool"]["ty"]["environment"]
    assert ty_env["extra-paths"] == ["../typings"]


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
    assert "temperature: float = ..." in patched
    assert "thinking_end_token: str = ..." in patched


def test_should_audit_path_excludes_generated_and_archived_paths(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    included = repo_root / "src" / "module.py"
    excluded_conda = repo_root / ".conda" / "lib" / "python3.13" / "site.py"
    excluded_output = repo_root / "src" / "output" / "results.md"
    excluded_archived = repo_root / "src" / "tools" / ".archived" / "old.py"
    included.parent.mkdir(parents=True)
    excluded_conda.parent.mkdir(parents=True)
    excluded_output.parent.mkdir(parents=True)
    excluded_archived.parent.mkdir(parents=True)
    included.write_text("print('ok')\n", encoding="utf-8")
    excluded_conda.write_text("value = 1  # noqa: F401\n", encoding="utf-8")
    excluded_output.write_text("<!-- markdownlint-disable MD028 -->\n", encoding="utf-8")
    excluded_archived.write_text("x = 1  # noqa: F841\n", encoding="utf-8")

    assert check_suppressions.should_audit_path(included, repo_root) is True
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
