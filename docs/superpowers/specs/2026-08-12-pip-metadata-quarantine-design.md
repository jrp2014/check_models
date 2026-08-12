# Pip Metadata Quarantine Design

## Context

`src/tools/update.sh` upgrades the core packaging tools with pip even when the
active environment was created by conda. A conda-owned distribution can lack
pip's `RECORD` file, and an interrupted or cross-manager upgrade can leave a
stale `.dist-info` directory without `METADATA`. Pip may select that stale
entry ahead of a valid newer entry, report the installed version as `None`, and
abort with `uninstall-no-record-file`.

The observed environment contained a valid `wheel-0.48.0.dist-info` beside a
stale `wheel-0.47.0.dist-info` containing neither `METADATA` nor `RECORD`.
Because `update.sh` currently invokes pip directly, it cannot recover from this
state.

## Goals

- Let the updater recover automatically from malformed pip metadata for the
  five packaging tools it is about to install: `pip`, `wheel`, `setuptools`,
  `build`, and `pyrefly`.
- Make recovery narrow, visible, and reversible.
- Exercise the recovery behavior with filesystem-based tests rather than only
  asserting shell-script source text.
- Preserve updater behavior for healthy metadata and unrelated distributions.

## Non-goals

- Repair arbitrary packages or general conda environment inconsistency.
- Repair missing package payload files or native libraries.
- Delete quarantined metadata.
- Change which packaging-tool versions `update.sh` requests.
- Transfer all packaging-tool ownership to conda.

## Chosen Design

Add `src/tools/quarantine_broken_pip_metadata.py`, a small typed helper invoked
by `update.sh` immediately before its core packaging-tool upgrade. Keeping the
filesystem operation in a Python module makes the behavior directly testable
while leaving the shell script responsible only for orchestration.

The helper will expose a function with this contract:

```python
def quarantine_broken_metadata(
    target_names: Collection[str],
    *,
    site_dirs: Collection[Path] | None = None,
    quarantine_parent: Path | None = None,
) -> list[tuple[Path, Path]]:
    """Move malformed target metadata and return source/destination pairs."""
```

When `site_dirs` is omitted, the helper obtains the active interpreter's
`purelib` and `platlib` paths from `sysconfig`, removes duplicate paths, and
examines direct `*.dist-info` children only. Tests provide temporary site
directories explicitly.

Distribution names are derived from the normalized `.dist-info` directory
name and compared with normalized target names. A matching directory is
malformed when either `METADATA` or `RECORD` is absent. This covers both the
observed stale entry and conda metadata that pip cannot uninstall, without
touching well-formed entries.

If malformed entries exist, the helper creates one temporary quarantine
directory with a `check_models-pip-metadata-` prefix. Each moved directory gets
a unique destination name, and the helper prints the original and quarantine
paths. It returns the same mappings for tests and callers. When no malformed
entry exists, it creates no quarantine directory and produces no recovery
warning.

The helper never follows or moves a symlink presented as a matching
`.dist-info` directory. It raises an error instead. Filesystem failures also
raise an error. The CLI converts these failures to a concise stderr message and
a nonzero exit, so `set -e` stops `update.sh` before pip runs. Quarantine uses a
move, not deletion, so the original metadata remains recoverable until the
operating system cleans its temporary directory.

`update.sh` will call the helper with exactly:

```text
pip wheel setuptools build pyrefly
```

The call will appear immediately before the existing
`pip_install_tool pip wheel "setuptools>=80,<82" build pyrefly` command.

## Alternatives Considered

### Inline Python in `update.sh`

This would avoid a new module but would make realistic filesystem behavior
awkward to test. The existing inline stale-backup cleanup is consequently
covered only by shell-source assertions. The new recovery path is riskier and
benefits from direct tests.

### Retry pip after parsing its error

This waits until pip has failed and couples recovery to pip's current error
wording. It also requires capturing and replaying the install command. A
preflight is simpler and prevents the known failure deterministically.

### Use conda for all packaging tools

This does not cover venv or uv environments, and conda channels may lag or lack
the requested tool versions. It also does not repair metadata already left by
cross-manager updates.

## Testing

Add tests to `src/tests/test_dependency_sync.py` that use temporary directories
to verify:

1. A malformed stale `wheel` entry is moved while a valid newer `wheel` entry
   and an unrelated malformed distribution remain untouched.
2. Healthy target metadata causes no move and no quarantine directory.
3. A matching symlink is rejected without moving its target.
4. `update.sh` invokes the helper for exactly the five approved tools before
   running the packaging-tool pip install.

The focused tests will run first. Project formatting, Ruff lint, commit hygiene,
and the full quality gate will then validate the integrated change in the
prescribed order.

## Documentation

Add an entry under `CHANGELOG.md` `[Unreleased]` describing automatic,
recoverable quarantine of malformed packaging-tool metadata. No user-facing
option or dependency-policy documentation changes are required.
