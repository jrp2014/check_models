# Lift the Transformers Version Cap Design

## Context

`check_models` currently requires `transformers>=5.7.0,<5.13.0`. The upper
bound was added because `mlx-lm` registered `NewlineTokenizer` by string name,
which Transformers 5.13 rejects. Upstream `mlx-lm` commit
`ab1806e8d66f74728072e96d55acb6c451e92e88` replaced the string with the
tokenizer class on 2026-07-08. The current GitHub `main` revision checked for
this change is `a790972f0f844d81067ed45c28b524220a10c019`.

The repository updater already pulls sibling `mlx-lm` GitHub checkouts,
installs them editably, verifies their install origin, and preserves those
editable installs during project dependency reconciliation. GitHub `main`
still reports package version `0.31.3`, so version metadata alone cannot
distinguish the fixed checkout from the April PyPI release.

## Decision

Remove the Transformers upper bound and retain the existing minimums:

- `transformers>=5.7.0`
- `mlx-lm>=0.31.3`

The supported development baseline is the latest editable `mlx-lm` GitHub
`main` checkout managed by `src/tools/update.sh`. The project will not add a
direct Git dependency, pin an upstream commit in packaging metadata, or change
the updater. This deliberately keeps normal dependency declarations and the
existing sibling-repository workflow intact.

## Code and Policy Changes

`src/check_models_data/dependency_policy.py` will keep
`PROJECT_TRANSFORMERS_VERSION_SPEC`, but it will contain only the minimum
specifier. The cap-specific maximum constant will be removed.

`src/check_models.py` will continue rejecting missing or below-minimum
Transformers versions. It will stop reporting or rejecting versions at or above
5.13.0. The special full-specifier compatibility helper and its cap-only imports
will be removed because normal minimum-version checks cover the remaining
runtime policy.

`src/pyproject.toml` will declare `transformers>=5.7.0`. The dependency sync
tool will regenerate the marked install snippets in `src/README.md`. Handwritten
README and implementation-guide wording will describe a minimum floor instead
of a bounded compatibility window. The unreleased changelog will describe the
net uncapped policy and upstream fix rather than retaining obsolete cap notes.

## Error Handling

Missing Transformers and versions below 5.7.0 remain hard dependency errors.
Preflight diagnostics continue reporting below-floor versions using the shared
dependency policy. There will be no special error path for newer Transformers
versions.

## Testing and Validation

The change will follow a red-green sequence:

1. Change the dependency-policy regression to require exactly
   `transformers>=5.7.0`.
2. Replace the above-cap diagnostic test with one proving Transformers 5.13.0
   produces no dependency issue when all other versions meet their floors.
3. Run those tests against the current implementation and confirm they fail for
   the existing cap.
4. Remove the cap-specific production policy and runtime branches, then rerun
   the tests and confirm they pass.
5. Run dependency synchronization and its check mode.
6. Run the prescribed formatting, Ruff fix/lint, commit-hygiene, and full
   `make quality` gates in the `mlx-vlm` conda environment.

## Success Criteria

- Packaging, shared policy, environment validation, README install snippets,
  and developer documentation all express `transformers>=5.7.0` with no upper
  bound.
- Transformers 5.13.0 and later are not rejected solely for their version.
- Missing or pre-5.7.0 Transformers versions retain their existing failures.
- No cap-specific constant, diagnostic, or obsolete compatibility wording
  remains in active project files.
- The full repository quality gate passes without rewriting tracked benchmark
  output artifacts.
