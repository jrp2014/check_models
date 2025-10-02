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

## Repository structure conventions

- Single focused package `vlm/` contains all Python code and its `pyproject.toml`.
- Root `Makefile` is a shim that forwards targets to `vlm/Makefile`.
- Generated artifacts (e.g., `typings/`) are not committed — regenerate via `make -C vlm stubs`.
- README at repo root documents high‑level layout and quickstart; package‑specific README lives in `vlm/`.

## Typings policy

- Third‑party stubs are generated locally into `typings/` using `vlm/tools/generate_stubs.py`.
- `typings/` is git‑ignored; do not commit generated stubs.
- `mypy_path = ["typings"]` is configured in `vlm/pyproject.toml` so mypy picks them up.
- Make targets:
  - `make -C vlm stubs` — generate/update stubs
  - `make -C vlm stubs-clear` — clear generated stubs

## Dependency source of truth

- `vlm/pyproject.toml` is authoritative for dependencies.
- Use `tools/update_readme_deps.py` to sync README install blocks.
- Keep runtime set slim — only libraries imported by `check_models.py` — and move optional tooling to extras.

## Quality gates

- Run `make quality` locally (ruff format+lint, mypy on `check_models.py`).
- CI should verify formatting, dependency sync, and run tests.

## Type Annotations

- All new functions must be fully typed (parameters + return type).
- Use `| None` over `Optional[T]` (Python 3.12 style).
- Prefer concrete container types (e.g., `list[str]`, `dict[str, Any]`).
- Use `Protocol` or `TypedDict` only when it materially clarifies usage or makes the code more robust.
- Runtime casts: use `typing.cast` sparingly; prefer narrowing via `if` guards.
- Replace blanket `# type: ignore` with specific codes (e.g., `# type: ignore[attr-defined]`).

### Paths & Files

- Prefer `pathlib.Path` for internal path handling and operations (`open`, `resolve`, `suffix`, joins).
- Public functions that accept a path-like input should use a `PathLike` alias (`str | Path`). Convert to `Path` at the top of the function for uniform handling.
- Only convert to `str` at library boundaries that explicitly require string paths (e.g., third-party APIs or JSON serialization of paths). Keep this conversion local to the call site.
- Do not accept raw `str` for paths unless necessary for interop; use `Path`/`PathLike` and document the expectation in docstrings.

## Missing Values and Sentinels

- Return `None` for missing/unavailable values instead of sentinel strings like "N/A" or "Unknown".
- Functions that may not be able to compute a value should reflect this in the return type with `| None`.
- Callers must handle `None` explicitly (e.g., via branching or a user-facing fallback), rather than baking sentinels into the data layer.
- Logging/printing: prefer empty string or a localized message at the presentation layer, e.g.,
  - `logger.info("Date: %s", date or "")` if the field is optional.
  - For summaries intended for machines, omit missing key/value pairs entirely rather than outputting placeholder text.

Rationale: `None` preserves semantic meaning for "not present" and avoids accidental comparisons against magic strings. It also simplifies downstream formatting logic and enables proper type checking.

### Library Versions (Optional values)

- Library version values are `str | None` (optional). When a version is unknown or a library isn't installed, return `None` and let presentation layers render an empty value.
- CLI/Markdown/HTML should not insert placeholders like "N/A" or "-" for versions. Prefer blank values for a clean, non‑misleading display.

### Centralized Display Normalization

- Use `format_field_value(field_name, value)` for all metric formatting. It is responsible for:
  - Converting `None` → `""` for display.
  - Normalizing memory values to GB with adaptive precision.
  - Formatting time (seconds) and TPS consistently.
  - Applying numeric formatting consistently.
- Callers should avoid duplicating formatting logic (e.g., do not re‑format numeric strings or add ad‑hoc fallbacks). If a new metric is introduced, extend the central formatter instead of branching inline.

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

### Markdown Linting

Markdown consistency is enforced (optionally) via `markdownlint-cli2` using the configuration in `.markdownlint.jsonc`:

- Long lines (MD013) are disabled to allow readable HTML/CSS blocks and wide tables.
- Inline HTML is allowed (MD033) because the codebase already sanitizes/escapes disallowed tags during report generation.
- Duplicate headings (MD024) are permitted; some conceptual repeats are intentional (e.g., Notes vs Important Notes).
- If you introduce new documentation, follow existing heading spacing (blank line before/after) and prefer asterisk `*` for unordered lists (already standard in `README.md`).
- Run locally with: `npx markdownlint-cli2 "**/*.md"` (or rely on the pre-commit hook if installed).

Rationale: Lightweight rules catch accidental spacing/list inconsistencies without blocking intentional formatting choices used for clarity.

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

### Markdown/HTML escaping policy

- Preserve a small allowlist of inline HTML tags that GitHub renders safely: `br`, `b`, `strong`, `i`, `em`, `code`.
- Escape all other HTML‑like tags by replacing `<` and `>` with `&lt;` and `&gt;` (prevents accidental interpretation, e.g., `<s>`).
- For Markdown output specifically:
  - Escape only the pipe character `|` (critical for table layout). Do not escape backslashes to avoid double‑escaping user content.
  - Convert newlines to `<br>` to maintain row structure in pipe tables.
  - Keep other characters as‑is to minimize visual noise.

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

### Exception Handling Policy

- **Catch specific exceptions** rather than broad `Exception` or bare `except`. Use the most specific exception type that correctly handles the error condition:
  - File operations: `OSError`, `FileNotFoundError`, `PermissionError`
  - Type errors: `TypeError`, `AttributeError`
  - Value errors: `ValueError`, `KeyError`
  - External library failures: Library-specific exceptions (e.g., `HFValidationError`, `UnidentifiedImageError`)
- Only catch `Exception` when you truly need to handle any error (e.g., at a top-level error boundary for user-facing error messages).
- When catching multiple related exceptions, use tuple syntax: `except (TypeError, ValueError, AttributeError):`
- Document unexpected exceptions in comments if the cause isn't obvious from context.
- Avoid silent failures: Always log errors or re-raise with context. Use `logger.exception()` to include traceback automatically.

### Constants & Magic Numbers

- **Extract magic numbers** into named `Final` constants at the module or block level when:
  - The value is used in multiple places (DRY principle)
  - The semantic meaning isn't immediately obvious from context (e.g., `5.0` → `IMAGE_OPEN_TIMEOUT`)
  - The value represents a domain concept (e.g., `80` → `GENERATION_WRAP_WIDTH`)
- Group related constants together with brief comments explaining their purpose.
- Use type annotations with `Final` to prevent reassignment: `TIMEOUT: Final[float] = 5.0`
- Constants should be UPPER_SNAKE_CASE and placed near the top of the file after imports.
- Don't extract constants for:
  - Single-use values where the semantic meaning is clear from immediate context
  - Values that are part of an algorithm's definition (e.g., mathematical constants like `0.5` in a midpoint calculation)
  - Trivial offsets like `0` or `1` when used in obvious contexts (e.g., `list[0]`)

### HTML Output Security

- **Always escape user-controlled content** before inserting into HTML:
  - Model names, file paths, generated text, error messages
  - Any data that originates from external sources (user input, file metadata, model outputs)
- Use `html.escape(text, quote=True)` for all user-controlled strings before HTML insertion.
- Apply escaping at the point of HTML generation, not earlier (to preserve original data for other uses).
- Allowlist approach: Only specific, known-safe HTML tags should be permitted in Markdown rendering (e.g., `<br>`, `<code>`).
- For user-generated content displayed in HTML tables, escape both:
  - Row headers (model identifiers, paths)
  - Cell contents (output text, error messages)
- Rationale: Prevents HTML injection and ensures generated reports render correctly even if model outputs contain HTML-like syntax.

## External Processes & Paths

- Prefer explicit executable paths if security warnings arise (e.g., `/usr/sbin/system_profiler`).
- Timeouts for subprocesses interacting with the system (already present in device info retrieval).

## Memory / Performance Reporting

- Keep formatting logic centralized (e.g., `format_field_value`). If a new metric aligns with existing heuristics, extend the existing function rather than branching inline. Do not perform extra per‑cell formatting in table builders.
- Memory values: Mixed sources require heuristic unit detection. MLX returns bytes; mlx-vlm returns decimal GB (bytes/1e9). The central formatter handles this automatically via threshold detection.
- Memory display: Always show GB (decimal GB). Use adaptive precision: integers for large values, one decimal for mid‑range, two decimals for fractions.

## HTML / CSS Blocks

- Long embedded style strings may exceed line length. Accept and keep them visually structured; do not introduce awkward concatenations solely to appease line length.

## Adding Dependencies

- Justify each new dependency in a brief comment or commit message.
- Prefer standard library or existing dependencies first.

### Dependency Version Synchronization

Runtime dependency versions MUST stay consistent between `pyproject.toml` and the install snippets in `vlm/README.md`.

Current slim runtime set (authoritative in `pyproject.toml`):
`mlx`, `mlx-vlm`, `Pillow`, `huggingface-hub`, `tabulate`, `tzlocal`.

If you add a new import in `vlm/check_models.py`, you MUST also:

1. Add the dependency to `[project].dependencies` (not just an optional group).
2. Run `python -m vlm.tools.update_readme_deps`.
3. Commit both `pyproject.toml` and the updated `README.md`.

The `test_dependency_sync` test and CI will fail otherwise.

Optional groups:

- `extras`: `psutil`, `tokenizers`, `mlx-lm`, `transformers`
- `torch`: `torch`, `torchvision`, `torchaudio`

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

- A GitHub Actions workflow (`.github/workflows/dependency-sync.yml`) enforces that README dependency blocks match `pyproject.toml`. If it fails, run:

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

## Conventions in Practice

A few short examples to make common conventions concrete:

### Metric formatting via the central formatter

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

### Rendering library versions (optional values)

```python
versions = {"mlx": "0.19.0", "mlx-vlm": None, "Pillow": "10.4.0"}

# CLI (example):
# mlx:       0.19.0
# mlx-vlm:
# Pillow:    10.4.0

# Markdown:
# - `mlx`: `0.19.0`
# - `mlx-vlm`: ``
# - `Pillow`: `10.4.0`
```

### Markdown escaping policy (tables)

```text
Input:  a|b\\c <unk>\nNext line
Output: a\|b\\c &lt;unk&gt;<br>Next line

- Only the pipe `|` is escaped (to preserve table columns).
- Backslashes are preserved (no extra escaping added).
- Unknown HTML-like tags (e.g., <unk>) are neutralized.
- Newlines become <br> to keep single-row structure.
```

## When in Doubt

1. Optimize for the next reader (could be an automated agent).
2. Prefer clarity over cleverness.
3. Leave the codebase slightly better than you found it.

---
Feel free to extend this guide as practices evolve.
