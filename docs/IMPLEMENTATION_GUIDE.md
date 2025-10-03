# Implementation Guide

This guide defines technical conventions and implementation details for developers working on this codebase. It covers coding standards, architecture decisions, and detailed technical policies.

**Target audience**: Developers actively modifying code, automated agents making changes, code reviewers.

> **Note**: This project was restructured in October 2025 to follow Python best practices. The old `vlm/` directory is now `src/`. See [notes/RESTRUCTURE_COMPLETED.md](notes/RESTRUCTURE_COMPLETED.md) for details.

**Related documents**:

- [CONTRIBUTING.md](CONTRIBUTING.md) - How to contribute (setup, workflow, PR process)
- [README.md](../README.md) - Project overview and quick start
- [notes/](notes/) - Design notes and project evolution
- [DEPENDENCY_STRATEGY.md](DEPENDENCY_STRATEGY.md) - Dependency management workflows (embedded below)

## Table of Contents

- [Core Priorities](#core-priorities)
- [Philosophy](#philosophy)
- [Repository Structure](#repository-structure)
- [Code Standards](#code-standards)
  - [Type Annotations](#type-annotations)
  - [Paths & Files](#paths--files)
  - [Missing Values and Sentinels](#missing-values-and-sentinels)
  - [Error Handling](#error-handling)
  - [Constants & Naming](#constants--naming)
- [Quality & Linting](#quality--linting)
- [Function Design](#function-design)
- [Comments & Documentation](#comments--documentation)
- [Output & Formatting](#output--formatting)
- [External Processes](#external-processes)
- [Dependencies](#dependencies)
- [Testing](#testing)
- [Quick Checklist](#quick-checklist)
- [Dependency Management Strategy](#dependency-management-strategy)

---

## Core Priorities

**In order of importance**:

1. **Correctness & determinism**
2. **Full type annotations** (no untyped defs in new/edited code unless impossible due to dynamic library APIs)
3. **Ruff cleanliness** (avoid new violations; existing explicit ignores are intentional)
4. **Readability & comprehensibility** over mechanical micro-refactors
5. **Strong, meaningful comments** instead of over-factoring into tiny single-use helpers

## Philosophy

### Core Principle: Readable First

A single medium-sized, well-commented function is often clearer than a web of one-line helpers.

**Only extract a helper when it**:

- Is reused (≥2 call sites), OR
- Encapsulates a distinct conceptual step that benefits from a name, OR
- Simplifies testing or isolates side effects

**Do NOT** introduce a new function solely to silence a complexity/length warning if it harms cohesion. Prefer an inline code comment or a local inner helper (with a brief comment) instead of a file-level public function.

## Repository Structure

### Directory Layout

- **`src/`** - Main Python package containing all code and its `pyproject.toml`
- **`docs/`** - All documentation (CONTRIBUTING.md, IMPLEMENTATION_GUIDE.md, etc.)
- **`docs/notes/`** - Design notes, reviews, and project evolution documentation
- **`output/`** - Generated reports (HTML/Markdown, git-ignored)
- **`typings/`** - Generated `.pyi` stubs (not committed, regenerate via stubs tools)
- **Root `Makefile`** - User-friendly commands that orchestrate `src/` operations
- **`README.md`** - Project overview and quick start guide (repo root)
- **`src/README.md`** - Detailed package usage and CLI documentation

### Generated Artifacts

- Generated artifacts (e.g., `typings/`) are **not committed**
- Regenerate via `make -C vlm stubs`
- See `.gitignore` for complete list of excluded files

### Typings Policy

- Third-party stubs are generated locally into `typings/` using `src/tools/generate_stubs.py`
- `typings/` is git-ignored; do not commit generated stubs
- `mypy_path = ["typings"]` is configured in `src/pyproject.toml` so mypy picks them up

**Generating stubs**:

```bash
python -m tools.generate_stubs mlx_vlm tokenizers
```

Or use the quality check with stub generation:

```bash
make quality  # Automatically generates stubs if missing
```

## Code Standards

### Type Annotations

**Requirements**:

- All new functions must be fully typed (parameters + return type)
- Use `| None` over `Optional[T]` (Python 3.13 style)
- Prefer concrete container types (e.g., `list[str]`, `dict[str, Any]`)
- Use `Protocol` or `TypedDict` only when it materially clarifies usage or makes the code more robust

**Type narrowing**:

- Runtime casts: use `typing.cast` sparingly; prefer narrowing via `if` guards
- Replace blanket `# type: ignore` with specific codes (e.g., `# type: ignore[attr-defined]`)

### Paths & Files

**Standards**:

- Prefer `pathlib.Path` for internal path handling and operations (`open`, `resolve`, `suffix`, joins)
- Public functions that accept a path-like input should use a `PathLike` alias (`str | Path`)
- Convert to `Path` at the top of the function for uniform handling
- Only convert to `str` at library boundaries that explicitly require string paths (e.g., third-party APIs or JSON serialization)
- Keep this conversion local to the call site

**Do not accept raw `str` for paths unless necessary for interop**; use `Path`/`PathLike` and document the expectation in docstrings.

### Missing Values and Sentinels

**Policy**:

- Return `None` for missing/unavailable values instead of sentinel strings like "N/A" or "Unknown"
- Functions that may not be able to compute a value should reflect this in the return type with `| None`
- Callers must handle `None` explicitly (e.g., via branching or a user-facing fallback)
- Logging/printing: prefer empty string or a localized message at the presentation layer

**Rationale**: `None` preserves semantic meaning for "not present" and avoids accidental comparisons against magic strings. It also simplifies downstream formatting logic and enables proper type checking.

**Library Versions**:

- Library version values are `str | None` (optional)
- When a version is unknown or a library isn't installed, return `None`
- Let presentation layers render an empty value
- CLI/Markdown/HTML should not insert placeholders like "N/A" or "-"

### Centralized Display Normalization

**Use `format_field_value(field_name, value)` for all metric formatting**

Responsibilities:

- Converting `None` → `""` for display
- Normalizing memory values to GB with adaptive precision
- Formatting time (seconds) and TPS consistently
- Applying numeric formatting consistently

**Do not duplicate formatting logic**. If a new metric is introduced, extend the central formatter instead of branching inline.

**Examples**:

```python
from vlm.check_models import format_field_value

# Memory (mlx bytes -> decimal GB normalization)
format_field_value("peak_memory", 8_589_934_592)  # "8.6"
format_field_value("peak_memory", 0.25)          # "0.25"

# Time (seconds)
format_field_value("generation_time", 1.2345)    # "1.23s"

# TPS (adaptive precision)
format_field_value("generation_tps", 1234.56)    # "1,235"

# Missing values → empty string
format_field_value("peak_memory", None)           # ""
```

### Error Handling

**General Policy**:

- Fail fast on programmer errors (raise) vs. silent pass
- Wrap external calls (filesystem, subprocess, model load) with concise try/except that annotates context
- Preserve original exception context with `raise ... from e` when enriching messages

**Exception Handling Policy**:

- **Catch specific exceptions** rather than broad `Exception` or bare `except`
- Use the most specific exception type that correctly handles the error condition:
  - File operations: `OSError`, `FileNotFoundError`, `PermissionError`
  - Type errors: `TypeError`, `AttributeError`
  - Value errors: `ValueError`, `KeyError`
  - External library failures: Library-specific exceptions (e.g., `HFValidationError`, `UnidentifiedImageError`)
- Only catch `Exception` when you truly need to handle any error (e.g., at a top-level error boundary)
- When catching multiple related exceptions, use tuple syntax: `except (TypeError, ValueError, AttributeError):`
- Document unexpected exceptions in comments if the cause isn't obvious from context
- Avoid silent failures: Always log errors or re-raise with context
- Use `logger.exception()` to include traceback automatically

### Constants & Naming

**Constants**:

- Use `UPPER_SNAKE_CASE` for constants near top of file or in a dedicated constants section
- Extract magic numbers into named `Final` constants at module or block level when:
  - The value is used in multiple places (DRY principle)
  - The semantic meaning isn't immediately obvious from context (e.g., `5.0` → `IMAGE_OPEN_TIMEOUT`)
  - The value represents a domain concept (e.g., `80` → `GENERATION_WRAP_WIDTH`)
- Group related constants together with brief comments explaining their purpose
- Use type annotations with `Final` to prevent reassignment: `TIMEOUT: Final[float] = 5.0`

**Don't extract constants for**:

- Single-use values where the semantic meaning is clear from immediate context
- Values that are part of an algorithm's definition
- Trivial offsets like `0` or `1` when used in obvious contexts (e.g., `list[0]`)

**Naming conventions**:

- Constants: `UPPER_SNAKE_CASE`
- Private helpers: prefix with `_` if not part of the public surface
- Avoid adding a constant for a value used only once unless it improves semantics

## Quality & Linting

### Ruff Configuration

Current config (see `pyproject.toml`):

- `select = ["ALL"]` + targeted `ignore` list
- Long line rule `E501` ignored intentionally: long literals (HTML/CSS, structured log templates) are acceptable if breaking harms readability
- Docstring rules (D100-D107) ignored: we require docstrings only for non-obvious or externally consumed functions
- Formatter conflicts (`COM812`, `ISC001`) ignored: ruff format handles these
- Complexity warnings (e.g., `C901`, `PLR0913`, `PLR0915`, `PLR0912`) may appear

**Handling complexity warnings**:

- Do not refactor purely to silence them if it reduces clarity
- Instead:
  - Add a top-of-function comment summarizing the flow; OR
  - If truly egregious and conceptually separable, refactor
  - Suppression (`# noqa: C901`) is allowed with an explanatory comment above it

**Usage**:

- Use `ruff format` for layout formatting
- Use `ruff check --fix` to apply automated fixes for style violations

### Markdown Linting

Markdown consistency is enforced (optionally) via `markdownlint-cli2` using the configuration in `.markdownlint.jsonc`:

- Long lines (MD013) are disabled to allow readable HTML/CSS blocks and wide tables
- Inline HTML is allowed (MD033) because the codebase already sanitizes/escapes disallowed tags
- Duplicate headings (MD024) are permitted; some conceptual repeats are intentional
- Follow existing heading spacing (blank line before/after) and prefer asterisk `*` for unordered lists
- Run locally with: `npx markdownlint-cli2 "**/*.md"` (or rely on the pre-commit hook if installed)

### Suppressions

When suppressing a rule:

```python
# Reason: Brief justification (why alternative is worse here)
# noqa: C901
```

Avoid drive-by suppressions without explanation.

## Function Design

### Size & Complexity

**Acceptable to exceed default complexity/statement counts when**:

- The function represents a linear, cohesive pipeline (e.g., formatting a report, orchestrating a model run)
- Splitting would scatter state or require threading many parameters

**Before refactoring for size**:

1. Would extracting reduce repeated logic? If yes, extract.
2. Would naming the extracted concept make the caller *clearer*? If yes, extract.
3. Otherwise, keep inline and reinforce with structured comments.

### Example: Acceptable Large Function Structure

```python
def process_image_pipeline(...) -> Result:  # noqa: C901 (Reason: cohesive multi-step pipeline)
    """High-level orchestration of the end-to-end image processing workflow.
    Steps: load → exif parse → model infer → format results.
    """
    # --- Load & validate input ---
    ...
    # --- Extract metadata ---
    ...
    # --- Run model inference ---
    ...
    # --- Aggregate & return ---
    return result
```

## Comments & Documentation

### Docstrings

- High-level docstring for modules and complex public functions: what + key decisions
- Omit obvious restatements of code
- Keep docstrings focused on intent and non-obvious decisions

### Inline Comments

- Use to delineate logical sections: `# --- Section: parsing EXIF metadata ---`
- Keep comments <100 chars when feasible
- Allow overflow for aligned tables/HTML/CSS
- Avoid commenting trivial transformations (e.g., `i += 1  # increment i`)

### Imports

- Standard library, then third-party, then local
- Keep grouped; one blank line between groups
- Use explicit imports over `from x import *`
- Avoid aliasing unless the alias is a widely recognized shorthand (e.g., `import numpy as np`)

### Optional Imports

- For optional deps (e.g., `psutil`), import in a `try/except ImportError` and fall back to `None`
- If needed for typing, use a specific ignore like `# type: ignore[assignment]` (avoid bare `# type: ignore`)

## Output & Formatting

### Logging Policy

- Use the central logger
- Avoid stray `print` except for deliberate user-facing terminal blocks (currently generation output preview)
- SUMMARY log lines must remain machine-parsable (`SUMMARY key=value ...`)
- Generated model output (non-verbose mode) respects 80-column wrapping while preserving logical newlines

### Color Conventions

**CLI and reports**:

- Identifiers (file/folder paths and model names): magenta in CLI for quick visual scanning
- Failures: red in CLI; in HTML tables failed rows are styled with a red background/text using the `failed-row` class
- Markdown: no colors (GitHub Markdown doesn't support ANSI); consider adding a textual marker like "FAILED" when needed

### Markdown Tables and Alignment

- Markdown doesn't support vertical alignment controls
- Most renderers (e.g., GitHub) top-align cells by default
- Enforce vertical top alignment only in the HTML report via inline CSS
- Keep Markdown simple for portability

### HTML Output Security

**Always escape user-controlled content before inserting into HTML**:

- Model names, file paths, generated text, error messages
- Any data that originates from external sources (user input, file metadata, model outputs)
- Use `html.escape(text, quote=True)` for all user-controlled strings before HTML insertion
- Apply escaping at the point of HTML generation, not earlier (to preserve original data for other uses)

**Allowlist approach**:

- Preserve a small allowlist of inline HTML tags that GitHub renders safely: `br`, `b`, `strong`, `i`, `em`, `code`
- Escape all other HTML-like tags by replacing `<` and `>` with `&lt;` and `&gt;`

**For Markdown output specifically**:

- Escape only the pipe character `|` (critical for table layout)
- Do not escape backslashes to avoid double-escaping user content
- Convert newlines to `<br>` to maintain row structure in pipe tables
- Keep other characters as-is to minimize visual noise

**Example**:

```text
Input:  a|b\\c <unk>\nNext line
Output: a\|b\\c &lt;unk&gt;<br>Next line

- Only the pipe `|` is escaped (to preserve table columns)
- Backslashes are preserved (no extra escaping added)
- Unknown HTML-like tags (e.g., <unk>) are neutralized
- Newlines become <br> to keep single-row structure
```

### Timezones

- Prefer `datetime.UTC` for aware datetimes
- Localize with `tzlocal.get_localzone()` when formatting local times

### Backend Import Guard Policy

- Default: block heavy backends that are not needed for MLX workflows to avoid macOS/Apple Silicon hangs:
  - Set `TRANSFORMERS_NO_TF=1`, `TRANSFORMERS_NO_FLAX=1`, `TRANSFORMERS_NO_JAX=1` at process start
  - Torch is allowed by default (some models require it)
- Opt-in to TensorFlow/Flax/JAX by setting `MLX_VLM_ALLOW_TF=1` before running
- Warning: Installing TensorFlow on macOS/ARM may trigger an Abseil mutex stall during import
- Note: Installing `sentence-transformers` isn't required here and may import heavy backends

## External Processes

### Subprocesses & Paths

- Prefer explicit executable paths if security warnings arise (e.g., `/usr/sbin/system_profiler`)
- Timeouts for subprocesses interacting with the system (already present in device info retrieval)

### Memory / Performance Reporting

- Keep formatting logic centralized (e.g., `format_field_value`)
- If a new metric aligns with existing heuristics, extend the existing function rather than branching inline
- Do not perform extra per-cell formatting in table builders

**Memory values**:

- Mixed sources require heuristic unit detection
- MLX returns bytes; mlx-vlm returns decimal GB (bytes/1e9)
- The central formatter handles this automatically via threshold detection

**Memory display**:

- Always show GB (decimal GB)
- Use adaptive precision: integers for large values, one decimal for mid-range, two decimals for fractions

### HTML / CSS Blocks

- Long embedded style strings may exceed line length
- Accept and keep them visually structured
- Do not introduce awkward concatenations solely to appease line length

## Dependencies

### Adding Dependencies

- Justify each new dependency in a brief comment or commit message
- Prefer standard library or existing dependencies first

### Dependency Version Synchronization

**Runtime dependency versions MUST stay consistent** between `pyproject.toml` and the install snippets in `src/README.md`.

**Current slim runtime set** (authoritative in `src/pyproject.toml`):
`mlx`, `mlx-vlm`, `Pillow`, `huggingface-hub`, `tabulate`, `tzlocal`

**If you add a new import in `src/check_models.py`, you MUST also**:

1. Add the dependency to `[project].dependencies` in `src/pyproject.toml` (not just an optional group)
2. Run `python -m tools.update_readme_deps` from the src directory
3. Commit both `src/pyproject.toml` and the updated `src/README.md`

The `test_dependency_sync` test and CI will fail otherwise.

**Optional groups**:

- `extras`: `psutil`, `tokenizers`, `mlx-lm`, `transformers`
- `torch`: `torch`, `torchvision`, `torchaudio`

**Mechanism**:

1. Edit versions only in `pyproject.toml` (authoritative source)
2. Run the sync helper: `python -m vlm.tools.update_readme_deps` to regenerate the blocks between:
   - `<!-- BEGIN MANUAL_INSTALL -->` / `<!-- END MANUAL_INSTALL -->`
   - `<!-- BEGIN MINIMAL_INSTALL -->` / `<!-- END MINIMAL_INSTALL -->`
3. Commit both changed files together

**Guidelines**:

- Do NOT hand-edit inside the marked blocks; the script will overwrite them
- If adding a new runtime dependency, ensure it is placed in `[project.dependencies]`
- When removing a dependency, delete it from `pyproject.toml`, run the sync script, and verify it disappears
- Optional/extras groups are intentionally excluded from automatic blocks; document their usage separately

**Rationale**: Single source of truth avoids configuration drift and stale README instructions.

### pyproject.toml Conventions (PEP 621 Compliance)

The `pyproject.toml` MUST conform to the official packaging guide: <https://packaging.python.org/en/latest/guides/writing-pyproject-toml>

**Required conventions**:

1. Use PEP 621 metadata under the single `[project]` table
2. Runtime dependencies belong in the `dependencies = [ ... ]` array
3. Optional/development groups live under `[project.optional-dependencies]`
4. The CLI entry point MUST use a fully-qualified module path in `[project.scripts]`
5. Build backend stays explicit in `[build-system]`
6. Tool configs (`[tool.ruff]`, `[tool.mypy]`, etc.) must NOT shadow standard PEP 621 fields
7. Keep ordering logically grouped: `[project]` → `[project.urls]` → `[project.scripts]` → dependencies → extras → build-system → tool configs
8. Do NOT introduce dynamic version computation; version is a literal string
9. When adding a new tool section, prefer concise comments referencing its upstream documentation
10. Any change to runtime dependencies MUST be followed by running the README sync script

**Validation checklist before committing pyproject changes**:

- [ ] PEP 621 core fields present (name/version/description/authors/readme/license/classifiers)
- [ ] `dependencies` array present within `[project]` section
- [ ] No stale legacy `[project.dependencies]` table remains
- [ ] Extras defined only if referenced in docs/README
- [ ] CLI script path correct & importable
- [ ] Author information updated (no placeholder)
- [ ] Appropriate classifiers added
- [ ] No duplicate dependency sections
- [ ] Tool configs valid
- [ ] README dependency blocks regenerated
- [ ] Tool sections still parse

**Automation**:

- A GitHub Actions workflow (`.github/workflows/dependency-sync.yml`) enforces that README dependency blocks match `pyproject.toml`
- If it fails, run: `cd src && python tools/update_readme_deps.py`

**Rationale**: Following the packaging guide ensures forward compatibility with modern build backends, simplifies automated parsing, and avoids ambiguous duplication of dependency sources.

## Testing

### Test Philosophy

- Minimal, focused tests for parsing/formatting helpers
- Avoid over-mocking
- Integration smoke tests acceptable for model pipeline (skipped if environment unavailable)

### Running Tests

```bash
make test           # Run all tests
pytest src/tests/   # Direct pytest invocation
```

### Git Hygiene and Caches

Do not commit ephemeral caches or local environment files. This repository includes a root `.gitignore` and `src/.gitignore` that exclude common caches and artifacts:

- Python: `__pycache__/`, `*.py[cod]`
- Tools: `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`, `.hypothesis/`, `.tox/`, `.nox/`
- Editors/IDE: `.vscode/`, `.idea/`
- Packaging: `build/`, `dist/`, `*.egg-info/`, `*.whl`
- Environments: `.venv/`, `venv/`, `env/`, `.env*`, `*.env`
- macOS: `.DS_Store`

If new tooling introduces a cache directory, add it to the appropriate `.gitignore`.

Generated artifacts like `results.html` and `results.md` are acceptable to commit for sharing, but large binary caches from Hugging Face or MLX should remain local.

## Quick Checklist

Before committing changes:

- [ ] Types added/maintained
- [ ] No new Ruff violations (run `ruff check --select ALL`)
- [ ] Complexity refactors only when they materially improve clarity
- [ ] Single-use helpers avoided unless naming clarifies intent
- [ ] Comments explain non-obvious logic and boundary decisions
- [ ] Logging consistent; SUMMARY line preserved if relevant
- [ ] No unscoped broad `except:` blocks
- [ ] No bare `# type: ignore` (use specific code)

## Migration Path (Future Tightening)

Potential future enhancements (not required now):

- Gradually re-enable selected docstring rules for public API boundaries
- Introduce targeted complexity caps per module
- Add mypy stricter flags (e.g., `disallow_untyped_defs`) after legacy code trimmed

---

## Dependency Management Strategy

### Philosophy: Easy Updates + Reproducible Builds

This project uses **pip-tools** to balance two goals:

1. **Easy updates**: Source files use `>=` constraints (not pinned versions)
2. **Reproducible builds**: Lock files pin exact versions for consistency

## How It Works

### Source Files (Flexible)

**`requirements.in`** and **`requirements-dev.in`** specify minimum versions with `>=`:

```ini
# ✅ Good: Allows updates
mlx>=0.29.1
ruff>=0.1.0

# ❌ Avoid: Freezes versions
mlx==0.29.1
ruff==0.1.0
```

**Why `>=`?**

- Allows automatic security patches
- Compatible with Dependabot updates
- Easy manual upgrades: just run `make upgrade-deps`

### Lock Files (Reproducible)

**`requirements.txt`** and **`requirements-dev.txt`** pin exact versions:

```txt
# Generated by pip-compile
mlx==0.29.1
  # via -r requirements.in
ruff==0.1.0
  # via -r requirements-dev.in
```

**Why lock files?**

- Reproducible builds across machines
- Consistent CI/CD environments
- Rollback safety if updates break things

## Update Workflows

### Routine Updates (Weekly/Monthly)

Update all dependencies to latest compatible versions:

```bash
# 1. Upgrade lock files
make -C src upgrade-deps

# 2. Sync your environment
make -C src sync-deps

# 3. Test thoroughly
make quality
make test

# 4. Commit if tests pass
git add src/requirements*.txt
git commit -m "chore(deps): upgrade dependencies"
```

### Adding New Dependencies

```bash
# 1. Edit requirements.in or requirements-dev.in
echo "new-package>=1.0.0" >> src/requirements.in

# 2. Regenerate lock files
make -C src lock-deps

# 3. Sync and test
make -C src sync-deps
make test

# 4. Commit both .in and .txt files
git add src/requirements*.in src/requirements*.txt
git commit -m "feat(deps): add new-package for X functionality"
```

### Checking for Outdated Packages

```bash
# Quick check
make -C vlm check-outdated

# Detailed info
pip list --outdated
```

### Security Audits

```bash
# Check for known vulnerabilities
make -C vlm audit
```

## Automated Updates

### Dependabot (Enabled)

**Configuration**: `.github/dependabot.yml`

- Runs weekly on Mondays
- Creates PRs for outdated packages
- Auto-labels with `dependencies` tag
- Limited to 5 open PRs max

**When Dependabot PR arrives**:

1. Review changes in PR
2. CI automatically tests
3. Merge if green
4. Pull and sync: `git pull && make -C vlm sync-deps`

## Version Constraints Guide

### Recommended Patterns

```ini
# ✅ Minimum version (allows updates)
package>=1.2.3

# ✅ Compatible release (allows patches & minor)
package~=1.2.3    # Equivalent to >=1.2.3,<1.3.0

# ⚠️ Use sparingly: exclude known broken version
package>=1.2.3,!=1.2.5

# ❌ Avoid: completely pinned
package==1.2.3
```

### When to Pin Versions

**Only pin in `.in` files if**:

1. **Breaking changes**: Package has history of breaking semver
2. **Critical stability**: Core dependency must be tested explicitly
3. **Compatibility**: Known incompatibility with newer versions

**Example** (hypothetical):

```ini
# Pinned due to breaking change in 0.30.0
mlx>=0.29.1,<0.30.0
```

## Makefile Targets Reference

| Command | Purpose | When to Use |
|---------|---------|-------------|
| `make lock-deps` | Generate `.txt` from `.in` | After editing `.in` files |
| `make sync-deps` | Sync env with `.txt` | After pulling new `.txt` files |
| `make upgrade-deps` | Upgrade all to latest | Weekly/monthly updates |
| `make check-outdated` | List outdated packages | Before deciding to upgrade |
| `make audit` | Security vulnerability scan | Before releases |

## CI/CD Integration

### Lock File Enforcement

CI uses lock files for reproducibility:

```yaml
# .github/workflows/quality.yml
- name: Install dependencies
  run: pip install -e src/.[dev]
```

### Dependency Sync Check

Pre-commit hook ensures `pyproject.toml` and `requirements.in` stay synchronized for README documentation purposes (via `tools/update_readme_deps.py`).

## Best Practices

### ✅ Do

- Use `>=` in `.in` files for flexibility
- Commit both `.in` and `.txt` files together
- Run `make check-outdated` regularly
- Test thoroughly after `upgrade-deps`
- Review Dependabot PRs within a week

### ❌ Don't

- Pin versions in `.in` files without good reason
- Commit `.txt` files without regenerating from `.in`
- Skip testing after dependency updates
- Ignore security audit warnings
- Let Dependabot PRs pile up

## Troubleshooting

### "Dependency conflict during upgrade"

```bash
# Try with backtracking resolver (default)
make -C src upgrade-deps

# If still fails, check which package is causing conflict
pip install pip-tools
pip-compile src/requirements.in --output-file=- 2>&1 | grep -i conflict

# Temporarily pin the problematic package in .in file
echo "problematic-package<2.0.0" >> src/requirements.in
make -C src lock-deps
```

### "Environment out of sync"

```bash
# Force sync to lock file state
make -C vlm sync-deps

# Or nuclear option: recreate environment
conda deactivate
conda remove -n mlx-vlm --all
conda create -n mlx-vlm python=3.13
conda activate mlx-vlm
make -C vlm bootstrap-dev
```

### "Dependabot PR causes test failures"

```bash
# Locally reproduce the issue
git checkout -b test-dependabot-update
git pull origin <dependabot-branch>
make -C vlm sync-deps
make -C vlm ci

# If tests fail, investigate and either:
# a) Fix code to work with new version
# b) Add version constraint to .in file to exclude problematic version
```

## Summary

✅ **Current State**: All dependencies use `>=` for easy updates  
✅ **Lock Files**: Ensure reproducibility via pip-compile  
✅ **Automation**: Dependabot handles routine updates  
✅ **Flexibility**: Can upgrade anytime with `make upgrade-deps`  
✅ **Safety**: Lock files + CI prevent surprises

**You have the best of both worlds!** 🎉

---

## When in Doubt

1. Optimize for the next reader (could be an automated agent)
2. Prefer clarity over cleverness
3. Leave the codebase slightly better than you found it

---

*Feel free to extend this guide as practices evolve.*
