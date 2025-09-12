# Project Style Guide

This guide defines coding conventions for this repository so automated agents (and humans) can make consistent, high‑quality changes.

## Core Priorities (In Order)

1. Correctness & determinism.
2. Full type annotations (no untyped defs in new/edited code unless impossible due to dynamic library APIs).
3. Ruff cleanliness (avoid new violations; existing explicit ignores are intentional).
4. Readability & comprehensibility over mechanical micro‑refactors.
5. Maintain strong, meaningful comments instead of over‑factoring into tiny single‑use helpers.

## Philosophy

"Readable first." A single medium‑sized, well‑commented function is often clearer than a web of one‑line helpers. Only extract a helper when it:

- Is reused (≥2 call sites), or
- Encapsulates a distinct conceptual step that benefits from a name, or
- Simplifies testing or isolates side effects.

Do NOT introduce a new function solely to silence a complexity / length warning if it harms cohesion. Prefer an inline code comment or a local inner helper (with a brief comment) instead of a file‑level public function.

## Type Annotations

- All new functions must be fully typed (parameters + return type).
- Use `| None` over `Optional[T]` (Python 3.12 style).
- Prefer concrete container types (e.g., `list[str]`, `dict[str, Any]`).
- Use `Protocol` or `TypedDict` only when it materially clarifies usage or makes the code more robust.
- Runtime casts: use `typing.cast` sparingly; prefer narrowing via `if` guards.
- Replace blanket `# type: ignore` with specific codes (e.g., `# type: ignore[attr-defined]`).

## Ruff / Linting

Current config (see `pyproject.toml`):

- `select = ["ALL"]` + targeted `ignore` list.
- Long line rule `E501` ignored intentionally: long literals (HTML/CSS, structured log templates) are acceptable if breaking harms readability.
- Docstring rules (D100–D107) ignored: we require docstrings only for non‑obvious or externally consumed functions.
- Formatter conflicts (`COM812`, `ISC001`) ignored: ruff format handles these.
- Complexity warnings (e.g., `C901`, `PLR0913`, `PLR0915`, `PLR0912`) may appear; do not refactor purely to silence them if it reduces clarity. Instead:
  - Add a top‑of‑function comment summarizing the flow; OR
  - If truly egregious and conceptually separable, refactor.
  - Suppression (`# noqa: C901`) is allowed with an explanatory comment above it.
- Use `ruff format` for layout formatting.
- Use `ruff check --fix` to apply automated fixes for style violations.

## Function Size & Complexity

Acceptable to exceed default complexity / statement counts when:

- The function represents a linear, cohesive pipeline (e.g., formatting a report, orchestrating a model run).
- Splitting would scatter state or require threading many parameters.

Before refactoring for size:

1. Would extracting reduce repeated logic? If yes, extract.
2. Would naming the extracted concept make the caller *clearer*? If yes, extract.
3. Otherwise, keep inline and reinforce with structured comments.

## Comments & Docstrings

- High‑level docstring for modules and complex public functions: what + key decisions (omit obvious restatements of code).
- Inline comments: use to delineate logical sections (`# --- Section: parsing EXIF metadata ---`).
- Keep comments <100 chars when feasible; allow overflow for aligned tables / HTML / CSS.
- Avoid commenting trivial transformations (e.g., `i += 1  # increment i`).

## Imports

- Standard library, then third‑party, then local. Keep grouped; one blank line between groups.
- Use explicit imports over `from x import *`.
- Avoid aliasing unless the alias is a widely recognized shorthand (e.g., `import numpy as np`).

## Logging & Output

- Use the central logger; avoid stray `print` except for deliberate user‑facing terminal blocks (currently generation output preview).
- SUMMARY log lines must remain machine‑parsable (`SUMMARY key=value ...`).
- Generated model output (non‑verbose mode) respects 80‑column wrapping while preserving logical newlines.

### Color conventions (CLI and reports)

- Identifiers (file/folder paths and model names): magenta in CLI for quick visual scanning.
- Failures: red in CLI; in HTML tables failed rows are styled with a red background/text using the `failed-row` class.
- Markdown: no colors (GitHub Markdown doesn’t support ANSI); consider adding a textual marker like "FAILED" when needed.

### Markdown tables and alignment

- Markdown doesn’t support vertical alignment controls; most renderers (e.g., GitHub) top-align cells by default.
- Enforce vertical top alignment only in the HTML report via inline CSS; keep Markdown simple for portability.

### Timezones

- Prefer `datetime.UTC` for aware datetimes; localize with `tzlocal.get_localzone()` when formatting local times.

### Optional imports

- For optional deps (e.g., `psutil`), import in a `try/except ImportError` and fall back to `None`.
- If needed for typing, use a specific ignore like `# type: ignore[assignment]` (avoid bare `# type: ignore`).

### Backend import guard policy

- Default: block heavy backends that are not needed for MLX workflows to avoid macOS/Apple Silicon hangs:
  - Set `TRANSFORMERS_NO_TF=1`, `TRANSFORMERS_NO_FLAX=1`, `TRANSFORMERS_NO_JAX=1` at process start.
  - Torch is allowed by default (some models require it).
- Opt‑in to TensorFlow/Flax/JAX by setting `MLX_VLM_ALLOW_TF=1` before running.
- Warning: Installing TensorFlow on macOS/ARM may trigger an Abseil mutex stall during import in unrelated code paths. Prefer not installing it in MLX‑only environments or keep it disabled via the guard.
- Note: Installing `sentence-transformers` isn’t required here and may import heavy backends; the script logs a heads‑up if it’s detected.

## Constants & Naming

- Constants: UPPER_SNAKE_CASE near top of file or in a dedicated constants section when shared.
- Private helpers: prefix with `_` if not part of the public surface.
- Avoid adding a constant for a value used only once unless it improves semantics (e.g., `GENERATION_WRAP_WIDTH`).

## Error Handling

- Fail fast on programmer errors (raise) vs. silent pass.
- Wrap external calls (filesystem, subprocess, model load) with concise try/except that annotates context.
- Preserve original exception context with `raise ... from e` when enriching messages.

## External Processes & Paths

- Prefer explicit executable paths if security warnings arise (e.g., `/usr/sbin/system_profiler`).
- Timeouts for subprocesses interacting with the system (already present in device info retrieval).

## Memory / Performance Reporting

- Keep formatting logic centralized (e.g., `format_field_value`). If a new metric aligns with existing heuristics, extend the existing function rather than branching inline.
- Memory values: Mixed sources require heuristic unit detection. MLX returns bytes; mlx-vlm returns decimal GB (bytes/1e9). The `format_field_value` function handles this automatically via threshold detection.
- Memory display: All memory values formatted as GB with adaptive decimal places (0, 1, or 2 decimals based on magnitude).

## HTML / CSS Blocks

- Long embedded style strings may exceed line length. Accept and keep them visually structured; do not introduce awkward concatenations solely to appease line length.

## Adding Dependencies

- Justify each new dependency in a brief comment or commit message.
- Prefer standard library or existing dependencies first.

### Dependency Version Synchronization

Runtime dependency versions MUST stay consistent between `pyproject.toml` and the install snippets in `vlm/README.md`.

Mechanism:

1. Edit versions only in `pyproject.toml` (authoritative source).
2. Run the sync helper: `python -m vlm.tools.update_readme_deps` (or via a future pre-commit/CI hook) to regenerate the blocks between:

- `<!-- BEGIN MANUAL_INSTALL -->` / `<!-- END MANUAL_INSTALL -->`
- `<!-- BEGIN MINIMAL_INSTALL -->` / `<!-- END MINIMAL_INSTALL -->`

1. Commit both changed files together.

Guidelines:

- Do NOT hand-edit inside the marked blocks; the script will overwrite them.
- If adding a new runtime dependency, ensure it is placed in `[project.dependencies]` (not only an optional group) or it will not appear in the synced snippets.
- When removing a dependency, delete it from `pyproject.toml`, run the sync script, and verify it disappears from both blocks.
- Optional / extras groups are intentionally excluded from automatic blocks; document their usage separately.

Rationale: Single source of truth avoids configuration drift and stale README instructions.

### pyproject.toml Conventions (PEP 621 Compliance)

The `pyproject.toml` MUST conform to the official packaging guide: <https://packaging.python.org/en/latest/guides/writing-pyproject-toml>

Required conventions for this project:

1. Use PEP 621 metadata under the single `[project]` table (name, version, description, authors, readme, license, keywords, classifiers, urls).
2. Runtime dependencies belong in the `dependencies = [ ... ]` array (NOT a legacy `[project.dependencies]` table). Each entry is a single quoted spec, optionally with an inline comment.
3. Optional/development groups live under `[project.optional-dependencies]` using standard PEP 621 extras (`dev`, `extras`, etc.).
4. The CLI entry point MUST use a fully-qualified module path in `[project.scripts]` (e.g., `vlm.check_models:main_cli`).
5. Build backend stays explicit in `[build-system]` (setuptools minimal requirement pinned to a modern version).
6. Tool configs (`[tool.ruff]`, `[tool.mypy]`, `[tool.pylance]`, etc.) must NOT shadow standard PEP 621 fields (avoid redefining metadata there).
7. Keep ordering logically grouped: `[project]` → `[project.urls]` → `[project.scripts]` → dependencies array → extras → build-system → tool configs.
8. Do NOT introduce dynamic version computation; version is a literal string (`version = "x.y.z"`).
9. When adding a new tool section, prefer concise comments referencing its upstream documentation.
10. Any change to runtime dependencies MUST be followed by running the README sync script (`cd vlm && python tools/update_readme_deps.py`).

Validation checklist before committing pyproject changes:

- [ ] PEP 621 core fields present (name/version/description/authors/readme/license/classifiers)
- [ ] `dependencies` array present within `[project]` section and sorted conceptually (grouped by comment blocks)
- [ ] No stale legacy `[project.dependencies]` table remains
- [ ] Extras defined only if referenced in docs / README
- [ ] CLI script path correct & importable
- [ ] Author information updated (no placeholder "Your Name")
- [ ] Appropriate classifiers added (including "Python :: 3 :: Only", OS-specific, etc.)
- [ ] No duplicate dependency sections (e.g., redundant `[tool.uv]` when `[project.optional-dependencies]` exists)
- [ ] Tool configs valid (mypy settings don't include formatter-specific fields)
- [ ] README dependency blocks regenerated
- [ ] Tool sections still parse (run a local `pip install -e .` or `uv pip install -e .`)

Automation:

- A GitHub Actions workflow (`.github/workflows/dependency-sync.yml`) enforces that README dependency blocks match `pyproject.toml`. If the workflow fails, run:

```bash
cd vlm && python tools/update_readme_deps.py
git add vlm/README.md
git commit -m "Sync README dependency blocks"
```

- (Optional) Add a local pre-commit hook to auto-run the sync when `pyproject.toml` changes:

```bash
cat > .git/hooks/pre-commit <<'HOOK'
#!/usr/bin/env bash
if git diff --cached --name-only | grep -q '^vlm/pyproject.toml$'; then
  echo '[pre-commit] Syncing README dependency blocks'
  cd vlm && python tools/update_readme_deps.py || exit 1
  git add vlm/README.md
fi
HOOK
chmod +x .git/hooks/pre-commit
```

Rationale: Following the packaging guide ensures forward compatibility with modern build backends, simplifies automated parsing (agents & scripts), and avoids ambiguous duplication of dependency sources.

## Git Hygiene and Caches

- Do not commit ephemeral caches or local environment files. This repository includes a root `.gitignore` and `vlm/.gitignore` that exclude common caches and artifacts:
  - Python: `__pycache__/`, `*.py[cod]`
  - Tools: `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`, `.hypothesis/`, `.tox/`, `.nox/`
  - Editors/IDE: `.vscode/`, `.idea/`
  - Packaging: `build/`, `dist/`, `*.egg-info/`, `*.whl`
  - Environments: `.venv/`, `venv/`, `env/`, `.env*`, `*.env`
  - macOS: `.DS_Store`
- If new tooling introduces a cache directory, add it to the appropriate `.gitignore`.
- Generated artifacts like `results.html` and `results.md` are acceptable to commit for sharing, but large binary caches from Hugging Face or MLX should remain local.

## Tests (If/When Added)

- Minimal, focused tests for parsing/formatting helpers.
- Avoid over‑mocking; integration smoke tests acceptable for model pipeline (skipped if environment unavailable).

## Suppressions

When suppressing a rule:

```python
# Reason: Brief justification (why alternative is worse here)
# noqa: C901
```

Avoid drive‑by suppressions without explanation.

## Migration Path (Future Tightening)

Potential future enhancements (not required now):

- Gradually re‑enable selected docstring rules for public API boundaries.
- Introduce targeted complexity caps per module.
- Add mypy stricter flags (e.g., `disallow_untyped_defs`) after legacy code trimmed.

## Quick Checklist for Contributors / Agents

- [ ] Types added/maintained.
- [ ] No new Ruff violations (run `ruff check --select ALL`).
- [ ] Complexity refactors only when they materially improve clarity.
- [ ] Single‑use helpers avoided unless naming clarifies intent.
- [ ] Comments explain non‑obvious logic and boundary decisions.
- [ ] Logging consistent; SUMMARY line preserved if relevant.
- [ ] No unscoped broad `except:` blocks.
- [ ] No bare `# type: ignore` (use specific code).

## Example: Acceptable Large Function Structure

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

## When in Doubt

1. Optimize for the next reader (could be an automated agent).
2. Prefer clarity over cleverness.
3. Leave the codebase slightly better than you found it.

---
Feel free to extend this guide as practices evolve.
