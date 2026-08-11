# Nanobind 2.14 Development-Tool Update

## Goal

Update `check_models` from nanobind 2.13.0 to 2.14.0 without changing the
nanobind version used to compile MLX itself.

## Scope

- Change the exact development dependency in `src/pyproject.toml` to
  `nanobind==2.14.0`.
- Refresh repository-maintained dependency documentation or lock artefacts with
  the existing dependency-sync tooling.
- Regenerate or validate the MLX stubs with the existing project tooling.
- Record the update under `CHANGELOG.md` `[Unreleased]`.
- Do not modify the adjacent MLX checkout or its CMake nanobind 2.13.0 pin.

## Compatibility Basis

Nanobind is used here as a development-time `mlx.core` stub generator, not as an
MLX runtime dependency. An isolated probe ran nanobind 2.14.0 against the current
MLX extension compiled with nanobind 2.13.0. Recursive generation completed and
all eight emitted stub files were byte-identical to nanobind 2.13.0 output.

## Verification

Run dependency sync, stub generation/validation, formatting and lint, followed
by the complete `make quality` gate. Any generated tracked stubs must be reviewed
for semantic changes; tracked run outputs under `src/output/` must remain
untouched.
