# Suppression Audit Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the suppression audit prove that every listed checker code is necessary and remove one broad-exception suppression that has no fallible operation to protect.

**Architecture:** Keep the existing temporary-file checker dispatch intact and tighten only `check_if_needed()`'s interpretation of checker output. Preserve runtime-probe fault isolation around third-party MLX calls, while making the deterministic mlx-vlm availability record direct.

**Tech Stack:** Python 3.13, pytest, Ruff, mypy, ty, pyrefly, Bash quality tooling.

## Global Constraints

- Always activate the `mlx-vlm` Conda environment before Python or Make commands.
- Keep `src/check_models.py` as an intentional single-file monolith.
- Add tests only to existing `src/tests/test_*.py` files.
- Do not add or broaden lint suppressions.
- Do not rewrite tracked `src/output/` assets during validation.
- Update `CHANGELOG.md` under `[Unreleased]`.

---

### Task 1: Require Evidence for Every Suppressed Code

**Files:**

- Modify: `src/tools/check_suppressions.py:277-304`
- Test: `src/tests/test_dependency_sync.py:1650-1725`

**Interfaces:**

- Consumes: `check_if_needed(finding: SuppressionFinding, *, repo_root: Path, src_root: Path) -> tuple[bool, str]`
- Produces: the same public return type, with failure reasons naming every stale code.

- [ ] **Step 1: Write the failing partial-evidence regression test**

Add a test that replaces `_run_for_finding()` with a deterministic completed
process reporting only `PLR0912` for a finding that names both `PLR0912` and
`PLR0915`:

```python
def test_check_if_needed_rejects_partially_stale_multi_code_suppression(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    src_root = repo_root / "src"
    src_root.mkdir(parents=True)
    file_path = src_root / "sample.py"
    file_path.write_text("def sample():  # noqa: PLR0912, PLR0915\n    pass\n")
    finding = check_suppressions.SuppressionFinding(
        file_path=file_path,
        line_num=1,
        kind="noqa",
        codes=("PLR0912", "PLR0915"),
        line_text="def sample():  # noqa: PLR0912, PLR0915",
    )
    result = subprocess.CompletedProcess(
        args=["ruff"],
        returncode=1,
        stdout="PLR0912 Too many branches",
        stderr="",
    )
    monkeypatch.setattr(check_suppressions, "_run_for_finding", lambda *args, **kwargs: result)

    needed, reason = check_suppressions.check_if_needed(
        finding,
        repo_root=repo_root,
        src_root=src_root,
    )

    assert needed is False
    assert reason == "No violation found for: PLR0915"
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
PYTHONPATH=src pytest src/tests/test_dependency_sync.py::test_check_if_needed_rejects_partially_stale_multi_code_suppression -q
```

Expected: FAIL because current `check_if_needed()` returns success as soon as it
finds `PLR0912`.

- [ ] **Step 3: Implement complete marker checking**

Replace the early-return loop with a complete missing-code calculation:

```python
missing_codes = tuple(
    code
    for code in finding.codes
    if (f"[{code}]" if finding.kind == "type-ignore" else code) not in output
)
if missing_codes:
    return False, f"No violation found for: {', '.join(missing_codes)}"
return True, f"Suppression needed: {', '.join(finding.codes)} violations found"
```

- [ ] **Step 4: Add the all-codes control test**

Use the same two-code finding with checker output containing both markers and
assert:

```python
assert needed is True
assert reason == "Suppression needed: PLR0912, PLR0915 violations found"
```

- [ ] **Step 5: Run focused and surrounding tests and verify GREEN**

Run:

```bash
PYTHONPATH=src pytest src/tests/test_dependency_sync.py -k suppression -q
```

Expected: all suppression-audit tests pass.

- [ ] **Step 6: Commit Task 1**

```bash
git add src/tools/check_suppressions.py src/tests/test_dependency_sync.py
git commit -m "fix: require evidence for every suppressed code"
```

### Task 2: Remove the Redundant Runtime-Probe Suppression

**Files:**

- Modify: `src/check_models.py:19687-19698`
- Test: `src/tests/test_jsonl_output.py:1340-1380`
- Modify: `CHANGELOG.md:7`

**Interfaces:**

- Consumes: `MISSING_DEPENDENCIES: dict[str, str]`
- Produces: unchanged `collect_runtime_fingerprint() -> dict[str, RuntimeProbeResult]` output with `mlx_vlm` status `ok` or `unavailable`.

- [ ] **Step 1: Add availability-state regression coverage**

Add two tests using `unittest.mock.patch.dict`:

```python
def test_collect_runtime_fingerprint_reports_mlx_vlm_available(self) -> None:
    with patch.dict(check_models.MISSING_DEPENDENCIES, {}, clear=True):
        fingerprint = check_models.collect_runtime_fingerprint()
    assert fingerprint["mlx_vlm"] == {"status": "ok"}

def test_collect_runtime_fingerprint_reports_mlx_vlm_unavailable(self) -> None:
    with patch.dict(
        check_models.MISSING_DEPENDENCIES,
        {"mlx-vlm": "not imported"},
        clear=True,
    ):
        fingerprint = check_models.collect_runtime_fingerprint()
    assert fingerprint["mlx_vlm"] == {
        "status": "unavailable",
        "detail": "not imported",
    }
```

- [ ] **Step 2: Run the availability tests as a behavior baseline**

Run:

```bash
PYTHONPATH=src pytest src/tests/test_jsonl_output.py -k 'runtime_fingerprint and mlx_vlm' -q
```

Expected: PASS before refactoring, establishing preserved behavior.

- [ ] **Step 3: Remove the deterministic broad catch**

Replace the `mlx-vlm` `try`/`except Exception` block with direct status
construction:

```python
if "mlx-vlm" not in MISSING_DEPENDENCIES:
    probes["mlx_vlm"] = RuntimeProbeResult(status="ok")
else:
    probes["mlx_vlm"] = RuntimeProbeResult(
        status="unavailable",
        detail=MISSING_DEPENDENCIES.get("mlx-vlm", "not imported"),
    )
```

- [ ] **Step 4: Document the hardening**

Add this `[Unreleased]` bullet under `Changed`:

```markdown
- Make suppression auditing require evidence for every listed checker code and
  remove a redundant broad-exception suppression from deterministic runtime
  fingerprinting.
```

- [ ] **Step 5: Verify runtime behavior and the live audit**

Run:

```bash
PYTHONPATH=src pytest src/tests/test_jsonl_output.py -k runtime_fingerprint -q
python -m tools.check_suppressions
```

Expected: runtime fingerprint tests pass; the live audit reports 29 necessary
suppressions and zero failures.

- [ ] **Step 6: Commit Task 2**

```bash
git add CHANGELOG.md src/check_models.py src/tests/test_jsonl_output.py
git commit -m "refactor: remove redundant runtime probe suppression"
```

### Task 3: Repository Verification

**Files:**

- Verify only; do not modify tracked output assets.

**Interfaces:**

- Consumes: completed Tasks 1 and 2.
- Produces: a clean branch with all quality gates passing.

- [ ] **Step 1: Apply repository formatting**

Run:

```bash
make format
make -C src lint-fix
```

Expected: formatting completes without introducing suppressions.

- [ ] **Step 2: Run the early lint gate**

Run:

```bash
make lint
```

Expected: Ruff reports no errors.

- [ ] **Step 3: Run the full quality gate**

Run:

```bash
make quality
```

Expected: all static checks, suppression audit, tests, ShellCheck, and
Markdownlint pass.

- [ ] **Step 4: Verify branch hygiene**

Run:

```bash
git status --short --branch
git diff --check main...HEAD
```

Expected: no uncommitted tracked files and no whitespace errors.
