# Plan: Harden MLX Integration

The audit points to four concrete weaknesses: runtime API drift is only partially detected, `SupportsGenerationResult` in [src/check_models.py](src/check_models.py) assumes looser `None` semantics than the audited upstream `mlx-vlm` defaults, version comparison is too numeric-only for dev/rc/editable builds, and the local-clone/stub workflow in [src/tools/update.sh](src/tools/update.sh) and [src/tools/generate_stubs.py](src/tools/generate_stubs.py) does not fail loudly enough when upstream changes. The recommended direction is a latest-first hardening pass, not pinning. Take the opportunity to shorten or simplify the code, if the opportunity arises naturally during the hardening.

1. Baseline dependency policy.
   - Reconcile the version floors in [src/pyproject.toml](src/pyproject.toml), [src/check_models.py](src/check_models.py), and [src/tools/validate_env.py](src/tools/validate_env.py).
   - Raise minimums where needed, but keep floor-only ranges.

2. Always-on API drift preflight. Depends on 1.
   - Extend [src/check_models.py](src/check_models.py) so `_collect_preflight_package_issues()` validates the exact upstream contract this repo uses: `load`, `generate`, `apply_chat_template`, `load_image`, and the `GenerationResult` fields read via `SupportsGenerationResult`.
   - Improve `_ensure_generation_runtime_symbols()`, `_classify_error()`, and `_attribute_error_to_package()` so missing symbol, signature drift, and result-shape drift are diagnosed separately.

3. Typing and normalization hardening. Parallel with 4 after 1.
   - Align `SupportsGenerationResult`, `StrictGenerateCallable`, and related helpers in [src/check_models.py](src/check_models.py) with audited upstream defaults, so empty-string and zero defaults are handled intentionally rather than as accidental `None` substitutes.

4. Tooling and provenance hardening. Parallel with 3 after 1.
   - Improve [src/tools/update.sh](src/tools/update.sh) to log repo root, branch, short SHA, and installed distribution version for local editable `mlx`, `mlx-lm`, and `mlx-vlm`.
   - Make [src/tools/generate_stubs.py](src/tools/generate_stubs.py) fail loudly when upstream stub shapes drift, and add a check-only or freshness path that can be reused by [src/tools/run_quality_checks.sh](src/tools/run_quality_checks.sh) if it stays deterministic.

5. Version parsing and diagnostics quality. Depends on 2 and 4.
   - Replace the current numeric-only version comparison in [src/check_models.py](src/check_models.py) with PEP 440 aware parsing so dev, rc, and local editable versions are classified correctly.
   - Surface compatibility issues through the existing preflight/reporting path instead of adding separate ad hoc logging.

6. Tests, docs, and maintainer workflow. Depends on 2-5.
   - Extend [src/tests/test_error_classification.py](src/tests/test_error_classification.py), [src/tests/test_validate_env.py](src/tests/test_validate_env.py), and either [src/tests/test_pure_logic_functions.py](src/tests/test_pure_logic_functions.py) or [src/tests/test_process_image_mock.py](src/tests/test_process_image_mock.py).
   - Update [src/README.md](src/README.md) if behavior is user-facing and record the hardening in [CHANGELOG.md](CHANGELOG.md).

## Verification

1. `conda activate mlx-vlm` then `cd src && python -m tools.validate_env && cd ..`
2. `pytest src/tests/test_error_classification.py src/tests/test_validate_env.py -q` plus the touched preflight/type-normalization tests
3. `make quality`
4. Manual smoke check of a light CLI path plus a mocked or preflight-only generation path
5. If [src/tools/update.sh](src/tools/update.sh) changes, validate both PyPI-style and local-sibling-clone flows

## Decisions

- Latest-first with raised minimum floors only
- API drift checks run during normal preflight
- Reuse the existing diagnostics hooks in [src/check_models.py](src/check_models.py), not a separate subsystem

If this scope matches what you want, approve the plan and the next pass can execute it in phases.
