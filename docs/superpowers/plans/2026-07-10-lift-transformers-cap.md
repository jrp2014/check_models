# Lift the Transformers Version Cap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the temporary Transformers upper bound now that the supported
editable `mlx-lm` GitHub `main` checkout contains the registration fix.

**Architecture:** Keep the shared dependency policy as the source for the
Transformers minimum, and make packaging, runtime validation, generated install
snippets, and prose documentation consume that uncapped policy. Delete the
cap-only runtime branches instead of retaining inactive compatibility code.

**Tech Stack:** Python 3.13, `packaging`, pytest, TOML project metadata,
Markdown documentation, Make, and the `mlx-vlm` conda environment.

## Global Constraints

- The Transformers requirement is exactly `transformers>=5.7.0`.
- The MLX language-model requirement remains `mlx-lm>=0.31.3`.
- Compatibility is based on editable `mlx-lm` GitHub `main` containing fix
  commit `ab1806e8d66f74728072e96d55acb6c451e92e88` or later.
- Do not add a direct Git dependency or change `src/tools/update.sh`.
- Missing Transformers and versions below 5.7.0 remain runtime errors.
- Do not split `src/check_models.py`.
- Run every Python, pytest, and Make command after activating conda environment
  `mlx-vlm`.
- Tests and validation must not rewrite tracked `src/output/` assets.
- Update `CHANGELOG.md` under `[Unreleased]` with the final net dependency
  policy.

---

### Task 1: Remove the Cap from Policy and Runtime Validation

**Files:**

- Modify: `src/tests/test_dependency_sync.py:261`
- Modify: `src/tests/test_pure_logic_functions.py:1062`
- Modify: `src/check_models_data/dependency_policy.py:15`
- Modify: `src/check_models.py:83,97,7841,7911,21881`
- Modify: `src/pyproject.toml:37`
- Test: `src/tests/test_dependency_sync.py`
- Test: `src/tests/test_pure_logic_functions.py`
- Test: `src/tests/test_validate_env.py`

**Interfaces:**

- Consumes: `PROJECT_RUNTIME_STACK_MINIMUMS["transformers"] == "5.7.0"` and
  `_detect_upstream_version_issues(versions: LibraryVersionDict) -> list[str]`.
- Produces: `PROJECT_TRANSFORMERS_VERSION_SPEC == ">=5.7.0"`; packaging and
  runtime validation that reject only missing or below-floor Transformers
  versions.

- [ ] **Step 1: Add the failing shared-policy regression**

Add this test immediately after
`test_dependency_policy_tracks_current_upstream_transformers_floor` in
`src/tests/test_dependency_sync.py`:

```python
def test_dependency_policy_does_not_cap_transformers() -> None:
    """Transformers should retain its floor without an upper version bound."""
    assert dependency_policy.PROJECT_TRANSFORMERS_VERSION_SPEC == ">=5.7.0"
```

- [ ] **Step 2: Replace the old cap diagnostic test with the acceptance regression**

Replace
`test_detect_upstream_version_issues_reports_transformers_above_supported_cap`
in `src/tests/test_pure_logic_functions.py` with:

```python
def test_detect_upstream_version_issues_accepts_transformers_without_upper_cap(
    self,
    mod: types.ModuleType,
) -> None:
    """Transformers releases above 5.12 should be accepted after the MLX fix."""
    issues = mod._detect_upstream_version_issues(
        {
            "mlx-vlm": "0.6.3",
            "mlx-lm": "0.31.3",
            "mlx": "0.31.2",
            "mlx-audio": "0.4.3",
            "transformers": "5.13.0",
            "huggingface-hub": "1.10.1",
        },
    )

    assert issues == []
```

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
PYTHONPATH=src pytest \
  src/tests/test_dependency_sync.py::test_dependency_policy_does_not_cap_transformers \
  src/tests/test_pure_logic_functions.py::TestPreflightDependencyDiagnostics::test_detect_upstream_version_issues_accepts_transformers_without_upper_cap \
  -q
```

Expected: two assertion failures. The policy value still includes `<5.13.0`,
and the preflight result still contains the cap-specific compatibility issue.

- [ ] **Step 4: Simplify the shared Transformers spec**

Replace the cap constants in
`src/check_models_data/dependency_policy.py` with:

```python
PROJECT_MIN_TRANSFORMERS_VERSION: Final[str] = PROJECT_RUNTIME_STACK_MINIMUMS["transformers"]
PROJECT_TRANSFORMERS_VERSION_SPEC: Final[str] = f">={PROJECT_MIN_TRANSFORMERS_VERSION}"
```

Delete `PROJECT_MAX_TRANSFORMERS_VERSION_EXCLUSIVE` entirely. Keep
`PROJECT_RUNTIME_STACK_SPECS["transformers"]` pointed at
`PROJECT_TRANSFORMERS_VERSION_SPEC`.

- [ ] **Step 5: Remove cap-only runtime code**

In `src/check_models.py`:

1. Replace `from packaging.specifiers import InvalidSpecifier, SpecifierSet`
   by deleting the import because neither symbol remains used.
2. Remove `PROJECT_MAX_TRANSFORMERS_VERSION_EXCLUSIVE` from the dependency
   policy import list.
3. Delete `_is_version_in_specifier()` in full.
4. Delete the `transformers_version` block at the end of
   `_detect_upstream_version_issues()` that reports an outside-supported-range
   issue.
5. Delete the final `elif not _is_version_in_specifier(...)` branch in
   `_raise_for_missing_runtime_dependencies()`.

Retain the existing missing-package and below-minimum branches:

```python
if transformers_version is None:
    MISSING_DEPENDENCIES["transformers"] = (
        "Core dependency missing: transformers. "
        f"Please install transformers>={PROJECT_MIN_TRANSFORMERS_VERSION}."
    )
elif not _is_version_at_least(transformers_version, PROJECT_MIN_TRANSFORMERS_VERSION):
    MISSING_DEPENDENCIES["transformers"] = (
        f"Core dependency too old: transformers=={transformers_version}. "
        f"Need transformers>={PROJECT_MIN_TRANSFORMERS_VERSION}."
    )
```

- [ ] **Step 6: Remove the packaging upper bound**

Replace the Transformers dependency and its comment in `src/pyproject.toml`
with:

```toml
    # Project-level compatibility floor for the MLX ecosystem; current
    # mlx-lm GitHub main supports Transformers 5.13.0 and later.
    "transformers>=5.7.0",
```

- [ ] **Step 7: Run the focused tests and verify GREEN**

Run:

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
PYTHONPATH=src pytest \
  src/tests/test_dependency_sync.py::test_dependency_policy_does_not_cap_transformers \
  src/tests/test_pure_logic_functions.py::TestPreflightDependencyDiagnostics::test_detect_upstream_version_issues_accepts_transformers_without_upper_cap \
  src/tests/test_validate_env.py::test_load_pyproject_deps_tracks_shared_runtime_policy \
  -q
```

Expected: `3 passed` with no warnings or errors.

- [ ] **Step 8: Run the complete policy and pure-logic regression files**

Run:

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
PYTHONPATH=src pytest \
  src/tests/test_dependency_sync.py \
  src/tests/test_pure_logic_functions.py \
  src/tests/test_validate_env.py \
  -q
```

Expected: all tests pass and tracked `src/output/` assets remain unchanged.

- [ ] **Step 9: Commit the policy and runtime change**

```bash
git add \
  src/check_models.py \
  src/check_models_data/dependency_policy.py \
  src/pyproject.toml \
  src/tests/test_dependency_sync.py \
  src/tests/test_pure_logic_functions.py
git commit -m "fix: lift transformers version cap"
```

### Task 2: Synchronize Documentation and Run the Full Quality Gate

**Files:**

- Modify: `src/README.md:562,719,770,776,809,812`
- Modify: `docs/IMPLEMENTATION_GUIDE.md:983,1138,1152`
- Modify: `CHANGELOG.md:35,177`
- Test: `src/tests/test_dependency_sync.py`
- Test: `src/tests/test_validate_env.py`

**Interfaces:**

- Consumes: `src/pyproject.toml` dependency `transformers>=5.7.0` and
  `PROJECT_TRANSFORMERS_VERSION_SPEC == ">=5.7.0"` from Task 1.
- Produces: synchronized generated install commands, active documentation with
  no obsolete cap wording, and a clean full repository quality gate.

- [ ] **Step 1: Regenerate README dependency blocks**

Run:

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
make deps-sync
```

Expected: the two marked install blocks in `src/README.md` change from
`"transformers<5.13.0,>=5.7.0"` to `"transformers>=5.7.0"`.

- [ ] **Step 2: Update handwritten README compatibility wording**

Make these exact policy edits outside generated blocks in `src/README.md`:

```markdown
| Transformer compatibility surface | `transformers` | `>=5.7.0` |
```

Replace each phrase
`` `transformers>=5.7.0,<5.13.0` compatibility window `` with
`` `transformers>=5.7.0` compatibility floor ``. Change the project-policy
note to:

```markdown
> Project policy requires `transformers>=5.7.0` and validates the live
> `mlx_vlm` runtime contract during preflight so upstream API drift is surfaced
> before generation starts.
```

- [ ] **Step 3: Update implementation guidance**

In `docs/IMPLEMENTATION_GUIDE.md`:

1. Replace the optional-group wording with
   `` `tokenizers` / `sentencepiece` specs track the `transformers>=5.7.0`
   compatibility floor ``.
2. Change the dependency example to `"transformers>=5.7.0",`.
3. Replace the cap rationale bullet with:

```markdown
- Lower bounds allow automatic security and compatibility updates; temporary
  upper bounds are removed once the supported upstream stack resolves the
  incompatibility.
```

- [ ] **Step 4: Record the final unreleased policy**

In `CHANGELOG.md`, replace the existing two Changed bullets about the bounded
Transformers compatibility window with:

```markdown
- Align the project Transformers floor with `mlx-lm>=0.31.3`
  (`transformers>=5.7.0`) and keep the shared dependency policy, install
  snippets, environment validation, and optional model-support floors in sync.
- Lift the temporary Transformers 5.13 upper bound after `mlx-lm` GitHub main
  replaced its string tokenizer registration with the tokenizer class, allowing
  the editable upstream stack to consume current Transformers releases.
```

Delete the obsolete Fixed bullet that says `update.sh` prevents selection of
`transformers==5.13.0`.

- [ ] **Step 5: Verify no active cap wording or symbols remain**

Run:

```bash
rg -n \
  'PROJECT_MAX_TRANSFORMERS_VERSION_EXCLUSIVE|transformers>=5\.7\.0,<5\.13\.0|transformers<5\.13\.0|above_supported_cap|outside supported range' \
  src/check_models.py \
  src/check_models_data \
  src/pyproject.toml \
  src/tests \
  src/README.md \
  docs/IMPLEMENTATION_GUIDE.md \
  CHANGELOG.md
```

Expected: no output. Historical released changelog or archived notes are not in
this active-file scan and do not need rewriting.

- [ ] **Step 6: Verify dependency synchronization**

Run:

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
cd src
python -m tools.update_readme_deps --check
```

Expected: the README dependency blocks are already synchronized and the command
exits successfully.

- [ ] **Step 7: Apply formatting and clear Ruff issues**

Run in order from the repository root:

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
make format
make -C src lint-fix
make lint
```

Expected: formatting completes, safe lint fixes make no unintended changes, and
Ruff format/lint checks pass.

- [ ] **Step 8: Run commit hygiene**

Run:

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
bash src/tools/run_commit_hygiene.sh
```

Expected: shell, Markdown, generated dependency, and repository hygiene checks
pass.

- [ ] **Step 9: Run the full quality gate**

Run:

```bash
source /Users/jrp/miniconda3/etc/profile.d/conda.sh
conda activate mlx-vlm
make quality
```

Expected: Ruff format/lint, mypy, ty, pyrefly, Vulture, Skylos, full pytest,
ShellCheck, and markdownlint all pass.

- [ ] **Step 10: Confirm only intentional files changed**

Run:

```bash
git status --short
git diff --check
git diff --stat 964f97c1
```

Expected: no tracked `src/output/` changes, no whitespace errors, and only the
policy, tests, packaging, generated README, implementation guide, changelog, and
this plan differ from the design-only baseline.

- [ ] **Step 11: Commit documentation and synchronized metadata**

```bash
git add \
  CHANGELOG.md \
  docs/IMPLEMENTATION_GUIDE.md \
  docs/superpowers/plans/2026-07-10-lift-transformers-cap.md \
  src/README.md
git commit -m "docs: document uncapped transformers support"
```
