---
name: add-or-fix-type-checking
description: >
  Fix broken typing checks detected by mypy, ty, pyrefly, or `make quality`.
  Use when typing errors appear in local runs, CI, or PR logs.
---

# Add or Fix Type Checking

## Input

- `<target>`: file or directory to type-check (defaults to `src/check_models.py`).
- Optional `make quality` or CI output showing typing failures.

## Environment

```bash
conda activate mlx-vlm                          # REQUIRED before any command
cd src && python -m tools.validate_env && cd ..  # quick sanity check
```

## Workflow

### 1. Identify scope from the failing run

- If you already have `make quality` or CI output, extract the failing
  file/module paths.
- If not, run:

  ```bash
  make quality
  ```

- Choose the narrowest target that covers the failures.

### 2. Run the individual checkers for a focused baseline

This repo uses **three type checkers** in order of priority:

```bash
# Primary — mypy (from src/)
cd src && mypy --config-file pyproject.toml check_models.py tests

# Secondary — ty (from repo root; resolves conda interpreter)
make ty

# Tertiary — pyrefly (from repo root)
make quality   # pyrefly runs as part of the pipeline
```

You can also run mypy on a specific file:

```bash
cd src && mypy --config-file pyproject.toml check_models.py
```

### 3. Triage errors by category before fixing anything

- Wrong/missing type annotations on signatures
- Attribute access on union types (e.g. `X | None`)
- Functions returning broad unions (e.g. `str | list | dict`)
- Protocol/TypedDict self-type issues
- Dynamic attributes on objects or modules
- Third-party stub gaps (missing kwargs, missing attributes)
  - Stubs live in `typings/`; regenerate with `make stubs`

### 4. Apply fixes using this priority order (simplest first)

#### a. Narrow unions with `isinstance()` / `if x is None` / `hasattr()`

This is the primary tool for resolving union-type errors. All three checkers
narrow through these patterns, including the negative forms:

```python
# Narrow X | None — use `if ...: raise`, never `assert`
if x is None:
    raise ValueError("x must not be None")
x.method()  # checkers know x is X here

# Narrow str | UploadFile
if isinstance(field, str):
    raise TypeError("Expected file upload, got string")
await field.read()  # checkers know field is UploadFile here
```

#### b. Use local variables to help checkers track narrowing across closures

When `self.x` is `X | None` and you need to pass it to nested functions or
closures, checkers cannot track that `self.x` stays non-None. Copy to a
local variable and narrow the local:

```python
manager = self.batching_manager
if manager is None:
    raise RuntimeError("Manager not initialized")
# Use `manager` (not `self.batching_manager`) in nested functions
```

#### c. Split chained calls when the intermediate type is a broad union

If `func().method()` fails because `func()` returns a union, split it:

```python
# BAD: checkers can't narrow through chained calls
result = func(return_dict=True).to(device)["input_ids"]

# GOOD: split, narrow, then chain
result = func(return_dict=True)
if not hasattr(result, "to"):
    raise TypeError("Expected dict-like result")
inputs = result.to(device)["input_ids"]
```

#### d. Fix incorrect type hints at the source

If a parameter is typed `X | None` but can never be `None` when actually
called, remove `None` from the hint.

#### e. Annotate untyped attributes

Add type annotations to instance variables set in `__init__` or elsewhere
(e.g. `self.foo: list[int] = []`). Declare class-level attributes that are
set dynamically later (e.g. `_cache: dict[str, Any]`).

#### f. Use `@overload` for methods with input-dependent return types

When a method returns different types based on the input type (e.g.
`__getitem__` with str vs int keys), use `@overload` to declare each
signature separately:

```python
from typing import overload

@overload
def __getitem__(self, item: str) -> ValueType: ...
@overload
def __getitem__(self, item: int) -> EncodingType: ...

def __getitem__(self, item: int | str) -> ValueType | EncodingType:
    ...  # actual implementation
```

This eliminates `cast()` calls at usage sites by giving the checker
precise return types for each call pattern.

#### g. Make container classes generic to propagate value types

When a class holds values whose type changes after transformation, make the
class generic so methods can return narrowed types:

```python
from typing import Generic
from typing_extensions import TypeVar

_V = TypeVar("_V", default=Any)

class MyDict(UserDict, Generic[_V]):
    @overload
    def __getitem__(self, item: str) -> _V: ...
    # ...
```

#### h. Use `cast()` as a last resort before `# type: ignore`

Use when you've structurally validated the type but the checker can't see it:
pattern-matched AST nodes, known-typed dict values, or validated API
responses.

```python
# After structural validation confirms the type:
stmt = cast(cst.Assign, node.body[0])
```

Do not use `cast()` when `@overload` or generics can solve it at the source.

#### i. Use `# type: ignore` only for third-party stub defects

This means cases where the third-party package's type stubs are wrong or
incomplete and there is no way to narrow or cast around it.

Always add the specific error code: `# type: ignore[call-arg]`, not bare
`# type: ignore`.

### 5. Things to NEVER do

- **Never use `assert` for type narrowing.** Asserts are stripped by
  `python -O` and must not be relied on for correctness. Use `if ...: raise`
  instead.
- **Never use `# type: ignore` as a first resort.** Exhaust all approaches
  above first.
- **Never add bare `# type: ignore`** without a specific error code.
- Do not use `cast()` when `@overload` or generics can eliminate it at the
  source.
- Do not add helper methods or abstractions just to satisfy the type checker
  (especially for only 1–2 occurrences).
- Do not add `if x is not None` guards for values guaranteed non-None by the
  call chain; fix the annotation instead.

### 6. Project-specific conventions

These rules are **mandatory** for this codebase (see `docs/IMPLEMENTATION_GUIDE.md`):

- `from __future__ import annotations` at top of every file.
- Full type annotations: all parameters + return types. Use `| None`, `list[str]`
  (not `Optional`, `List`).
- `Final` for constants: `TIMEOUT: Final[float] = 5.0`.
- `pathlib.Path` for all paths; convert to `str` only at library call boundaries.
- `raise SystemExit(code)` instead of `sys.exit()` (better for type narrowing).
- `Protocol` over ABCs for optional-dependency typing.
- Return `None` for missing values (not sentinel strings like `"N/A"`).
- Import type-only symbols under `if TYPE_CHECKING:` to avoid circular deps
  and runtime overhead.
- Replace blanket `# type: ignore` with specific codes
  (e.g., `# type: ignore[attr-defined]`).
- No suppressions (`# noqa`, `# type: ignore`) without a documented reason.
- Optional dependencies use `try/except ImportError` → `None` fallback pattern.

### 7. Stubs management

Third-party stubs live in `typings/` at the repo root (git-ignored). All
three checkers reference them:

- mypy: `mypy_path = "typings"` in `[tool.mypy]`
- ty: `extra-paths = ["../typings"]` in `[tool.ty.environment]`
- pyrefly: `search-path = ["../typings"]` in `[tool.pyrefly]`

If stub gaps cause errors:

```bash
make stubs          # Regenerate all stubs
make clean-stubs    # Remove stubs, then regenerate
```

### 8. Verify and close the loop

- Re-run the individual checker that failed:

  ```bash
  cd src && mypy --config-file pyproject.toml check_models.py
  make ty
  ```

- Re-run the full pipeline to confirm no regressions:

  ```bash
  make quality
  ```

- Ensure runtime behavior did not change and run relevant tests:

  ```bash
  make test
  ```
