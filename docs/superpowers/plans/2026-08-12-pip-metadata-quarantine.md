# Pip Metadata Quarantine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `src/tools/update.sh` automatically and recoverably quarantine malformed pip metadata for the five core packaging tools before upgrading them.

**Architecture:** Add one typed Python helper under `src/tools/` to isolate and test the filesystem operation. The shell updater invokes that helper immediately before its existing packaging-tool install; malformed target `.dist-info` directories move to a temporary quarantine, while healthy and unrelated metadata remain untouched.

**Tech Stack:** Python 3.13 (`pathlib`, `sysconfig`, `tempfile`, `shutil`, `argparse`), Bash, pytest, Ruff, mypy, ty, pyrefly, shellcheck.

## Global Constraints

- Activate the `mlx-vlm` conda environment before every Python, pytest, or make command.
- Limit automatic repair to `pip`, `wheel`, `setuptools`, `build`, and `pyrefly`.
- Treat a target `.dist-info` directory as malformed when `METADATA` or `RECORD` is missing.
- Move malformed metadata to a temporary quarantine; never delete it.
- Refuse matching symlinks and stop the updater if quarantine fails.
- Leave healthy target metadata and every unrelated distribution untouched.
- Add tests only to the existing `src/tests/test_dependency_sync.py` file.
- Update `CHANGELOG.md` under `[Unreleased]`.
- Run formatting and lint repair before the full quality gate.

---

### Task 1: Implement and test metadata quarantine

**Files:**

- Create: `src/tools/quarantine_broken_pip_metadata.py`
- Modify: `src/tests/test_dependency_sync.py`

**Interfaces:**

- Consumes: target distribution names supplied by `update.sh`; active `purelib` and `platlib` paths from `sysconfig` when tests do not inject paths.
- Produces: `quarantine_broken_metadata(target_names: Collection[str], *, site_dirs: Collection[Path] | None = None, quarantine_parent: Path | None = None) -> list[tuple[Path, Path]]` and a CLI accepting one or more target distribution names.

- [ ] **Step 1: Write the failing filesystem regression test**

Add `import runpy` to `src/tests/test_dependency_sync.py`, then add a temporary dynamic loader and the exact observed stale/new metadata case. The leading file assertion ensures the test fails as an assertion, rather than a collection error, before the helper exists.

```python
PIP_METADATA_HELPER = PKG_ROOT / "tools" / "quarantine_broken_pip_metadata.py"


def _load_pip_metadata_helper() -> dict[str, object]:
    assert PIP_METADATA_HELPER.is_file(), "pip metadata quarantine helper is missing"
    return runpy.run_path(str(PIP_METADATA_HELPER), run_name="pip_metadata_helper_test")


def _write_dist_info(
    site_dir: Path,
    distribution: str,
    version: str,
    *,
    complete: bool,
) -> Path:
    metadata_dir = site_dir / f"{distribution}-{version}.dist-info"
    metadata_dir.mkdir()
    (metadata_dir / "INSTALLER").write_text("pip\n", encoding="utf-8")
    if complete:
        (metadata_dir / "METADATA").write_text(
            f"Name: {distribution}\nVersion: {version}\n",
            encoding="utf-8",
        )
        (metadata_dir / "RECORD").write_text("", encoding="utf-8")
    return metadata_dir


def test_quarantine_broken_pip_metadata_moves_only_malformed_targets(
    tmp_path: Path,
) -> None:
    """A stale wheel metadata husk should not mask its healthy replacement."""
    helper = _load_pip_metadata_helper()
    quarantine = typing.cast(typing.Callable[..., list[tuple[Path, Path]]], helper["quarantine_broken_metadata"])
    site_dir = tmp_path / "site-packages"
    site_dir.mkdir()
    quarantine_parent = tmp_path / "quarantine"
    quarantine_parent.mkdir()
    stale_wheel = _write_dist_info(site_dir, "wheel", "0.47.0", complete=False)
    healthy_wheel = _write_dist_info(site_dir, "wheel", "0.48.0", complete=True)
    unrelated = _write_dist_info(site_dir, "example", "1.0.0", complete=False)

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
```

- [ ] **Step 2: Run the regression test and verify RED**

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && pytest src/tests/test_dependency_sync.py::test_quarantine_broken_pip_metadata_moves_only_malformed_targets -q'
```

Expected: FAIL with `pip metadata quarantine helper is missing`.

- [ ] **Step 3: Implement the minimal typed helper**

Create `src/tools/quarantine_broken_pip_metadata.py` with the following behavior. Keep name normalization and active-site discovery private; keep the quarantine function public for direct tests.

```python
"""Quarantine malformed pip metadata before packaging-tool upgrades."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import sysconfig
import tempfile
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Final

DIST_INFO_SUFFIX: Final = ".dist-info"
NORMALIZE_NAME_RE: Final = re.compile(r"[-_.]+")
QUARANTINE_PREFIX: Final = "check_models-pip-metadata-"


def _normalize_distribution_name(name: str) -> str:
    return NORMALIZE_NAME_RE.sub("-", name).lower()


def _distribution_name(metadata_dir: Path) -> str:
    stem = metadata_dir.name.removesuffix(DIST_INFO_SUFFIX)
    name, separator, _version = stem.rpartition("-")
    return _normalize_distribution_name(name) if separator else ""


def _active_site_dirs() -> tuple[Path, ...]:
    paths = {
        Path(value)
        for key in ("purelib", "platlib")
        if (value := sysconfig.get_path(key)) is not None
    }
    return tuple(sorted(paths))


def quarantine_broken_metadata(
    target_names: Collection[str],
    *,
    site_dirs: Collection[Path] | None = None,
    quarantine_parent: Path | None = None,
) -> list[tuple[Path, Path]]:
    """Move malformed target metadata and return source/destination pairs."""
    normalized_targets = {_normalize_distribution_name(name) for name in target_names}
    roots = tuple(sorted(set(site_dirs if site_dirs is not None else _active_site_dirs())))
    broken: list[Path] = []

    for site_dir in roots:
        if not site_dir.is_dir():
            continue
        for metadata_dir in sorted(site_dir.glob(f"*{DIST_INFO_SUFFIX}")):
            if _distribution_name(metadata_dir) not in normalized_targets:
                continue
            if metadata_dir.is_symlink():
                msg = f"refusing symlinked metadata directory: {metadata_dir}"
                raise RuntimeError(msg)
            if not metadata_dir.is_dir():
                msg = f"metadata path is not a directory: {metadata_dir}"
                raise RuntimeError(msg)
            if not (metadata_dir / "METADATA").is_file() or not (
                metadata_dir / "RECORD"
            ).is_file():
                broken.append(metadata_dir)

    if not broken:
        return []

    quarantine_root = Path(
        tempfile.mkdtemp(
            prefix=QUARANTINE_PREFIX,
            dir=str(quarantine_parent) if quarantine_parent is not None else None,
        ),
    )
    moved: list[tuple[Path, Path]] = []
    for index, source in enumerate(broken, start=1):
        destination = quarantine_root / f"{index:02d}-{source.name}"
        shutil.move(source, destination)
        moved.append((source, destination))
    return moved


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target_names", nargs="+")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        moved = quarantine_broken_metadata(args.target_names)
    except (OSError, RuntimeError) as exc:
        print(
            f"[update.sh] ERROR: unable to quarantine malformed pip metadata: {exc}",
            file=sys.stderr,
        )
        return 1

    if moved:
        print("[update.sh] Quarantined malformed pip metadata:")
        for source, destination in moved:
            print(f"   - {source} -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the regression test and verify GREEN**

Run the Step 2 command again.

Expected: PASS.

- [ ] **Step 5: Add healthy-metadata and symlink tests**

Add these tests beside the regression test:

```python
def test_quarantine_broken_pip_metadata_is_noop_for_healthy_target(
    tmp_path: Path,
) -> None:
    """Healthy metadata should not create an empty quarantine directory."""
    helper = _load_pip_metadata_helper()
    quarantine = typing.cast(typing.Callable[..., list[tuple[Path, Path]]], helper["quarantine_broken_metadata"])
    site_dir = tmp_path / "site-packages"
    site_dir.mkdir()
    quarantine_parent = tmp_path / "quarantine"
    quarantine_parent.mkdir()
    healthy_wheel = _write_dist_info(site_dir, "wheel", "0.48.0", complete=True)

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
    helper = _load_pip_metadata_helper()
    quarantine = typing.cast(typing.Callable[..., list[tuple[Path, Path]]], helper["quarantine_broken_metadata"])
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
```

- [ ] **Step 6: Run all three helper tests**

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && pytest src/tests/test_dependency_sync.py -k "quarantine_broken_pip_metadata" -q'
```

Expected: 3 passed.

- [ ] **Step 7: Refactor the tests to use the normal explicit module import**

Remove `import runpy` and `_load_pip_metadata_helper`. Add
`quarantine_broken_pip_metadata` to the existing `from tools import (...)`
block, and replace each dynamic lookup with:

```python
quarantine = quarantine_broken_pip_metadata.quarantine_broken_metadata
```

Run the Step 6 command again and expect 3 passed.

- [ ] **Step 8: Commit the tested helper**

Run:

```bash
git add src/tools/quarantine_broken_pip_metadata.py src/tests/test_dependency_sync.py
git commit -m "fix: quarantine malformed pip metadata"
```

Expected: the commit contains the typed helper and its three filesystem tests.

### Task 2: Integrate recovery into the updater

**Files:**

- Modify: `src/tools/update.sh`
- Modify: `src/tests/test_dependency_sync.py`
- Modify: `CHANGELOG.md`

**Interfaces:**

- Consumes: `src/tools/quarantine_broken_pip_metadata.py` CLI with positional target names.
- Produces: a preflight call before `pip_install_tool pip wheel "setuptools>=80,<82" build pyrefly`; a changelog record of the repair policy.

- [ ] **Step 1: Write the failing integration-order test**

Add this test near the existing updater cleanup test:

```python
def test_update_script_quarantines_broken_packaging_metadata_before_upgrade() -> None:
    """The packaging-tool upgrade should preflight only its own metadata."""
    update_script = (PKG_ROOT / "tools" / "update.sh").read_text(encoding="utf-8")
    helper_name = "quarantine_broken_pip_metadata.py"
    install_command = 'pip_install_tool pip wheel "setuptools>=80,<82" build pyrefly'

    assert helper_name in update_script
    helper_position = update_script.index(helper_name)
    install_position = update_script.index(install_command)
    preflight = update_script[helper_position:install_position]
    assert "pip wheel setuptools build pyrefly" in " ".join(preflight.split())
    assert helper_position < install_position
```

- [ ] **Step 2: Run the integration-order test and verify RED**

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && pytest src/tests/test_dependency_sync.py::test_update_script_quarantines_broken_packaging_metadata_before_upgrade -q'
```

Expected: FAIL because `update.sh` does not yet contain the helper name.

- [ ] **Step 3: Add the preflight call immediately before the existing install**

Change the core packaging-tool block in `src/tools/update.sh` to:

```bash
# Ensure global Python packaging tools are current. Quarantine malformed
# metadata for only these requested tools before pip attempts an uninstall.
python "$SCRIPT_DIR/quarantine_broken_pip_metadata.py" \
	pip wheel setuptools build pyrefly

# Use pip_install_tool (non-eager) to avoid cascading upgrades of shared deps
echo "[update.sh] Updating core Python packaging tools (pip, wheel, setuptools, build, pyrefly)..."
pip_install_tool pip wheel "setuptools>=80,<82" build pyrefly
```

- [ ] **Step 4: Run the helper and integration tests and verify GREEN**

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && pytest src/tests/test_dependency_sync.py -k "quarantine_broken_pip_metadata or quarantines_broken_packaging_metadata" -q'
```

Expected: 4 passed.

- [ ] **Step 5: Update the changelog**

Add under `CHANGELOG.md` → `[Unreleased]` → `### Fixed`:

```markdown
- Harden `src/tools/update.sh` against mixed conda/pip metadata damage: before
  upgrading its five core packaging tools, it now moves only malformed matching
  `.dist-info` directories to a reported temporary quarantine. This recovers
  from pip's `uninstall-no-record-file` failure without deleting metadata or
  touching unrelated distributions.
```

- [ ] **Step 6: Run focused tests and shell syntax validation**

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && pytest src/tests/test_dependency_sync.py -k "quarantine_broken_pip_metadata or quarantines_broken_packaging_metadata" -q && bash -n src/tools/update.sh'
```

Expected: 4 passed and `bash -n` exits successfully.

- [ ] **Step 7: Run the prescribed formatting and lint sequence**

Run in order:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && make format'
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && make -C src lint-fix'
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && make lint'
```

Expected: all commands exit successfully. Inspect the diff after formatting and retain only changes belonging to this task.

- [ ] **Step 8: Run commit hygiene and the full quality gate**

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && bash src/tools/run_commit_hygiene.sh'
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && make quality'
```

Expected: both commands exit successfully, including the full pytest, shellcheck, and Markdown lint suites.

- [ ] **Step 9: Inspect and commit the implementation**

Run:

```bash
git -c core.fsmonitor=false diff --check
git -c core.fsmonitor=false status --short
git diff -- src/tools/quarantine_broken_pip_metadata.py src/tools/update.sh src/tests/test_dependency_sync.py CHANGELOG.md
git add src/tools/quarantine_broken_pip_metadata.py src/tools/update.sh src/tests/test_dependency_sync.py CHANGELOG.md
git commit -m "fix: quarantine broken pip metadata during updates"
```

Expected: the commit contains only the helper, updater integration, regression tests, and changelog entry.
