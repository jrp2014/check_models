# Suppression Audit Hardening Design

## Context

`src/tools/run_quality_checks.sh` runs `tools.check_suppressions` after mypy.
The audit covers targeted Ruff `noqa` directives, mypy `type: ignore`
directives, and ShellCheck suppressions. The current repository has 30 targeted
Ruff suppressions and no mypy or ShellCheck suppressions.

The audit proves that removing a directive makes its checker emit at least one
listed code. For a directive containing multiple codes, however, it returns
success as soon as the first code is found. A partially stale directive can
therefore pass. Manual review also found one broad-exception suppression around
the `mlx-vlm` availability fingerprint even though that block performs no
fallible third-party call.

## Goals

- Require every code named by a suppression directive to be independently
  demonstrated by the corresponding checker.
- Remove the unnecessary broad-exception block and its `BLE001` suppression.
- Preserve targeted suppressions whose underlying operations genuinely require
  them, including third-party runtime probes, fixed subprocess invocations,
  delayed imports, validated URL access, and the dual-format renderer.
- Keep the audit output actionable when only some codes are stale.

## Non-goals

- Refactor the intentional `check_models.py` monolith.
- Split the dual HTML/Markdown cataloging renderer merely to satisfy complexity
  thresholds.
- Replace targeted security suppressions with broader project-level ignores.
- Add type annotations solely for appearance; no `type: ignore` directives are
  currently present.

## Design

`check_if_needed()` will collect the expected output marker for every code and
compare the complete set against the checker output. It will pass only when all
markers are present. When one or more are absent, it will fail with a message
that names the stale codes so a maintainer can narrow the directive directly.

The `mlx-vlm` availability entry in `collect_runtime_fingerprint()` will be
constructed without a `try`/`except Exception` wrapper. The surrounding MLX
runtime probes will retain broad catches because third-party runtime and Metal
calls may raise implementation-specific exceptions and the fingerprint contract
requires each probe to record an error rather than abort the report.

## Testing

Add regression coverage to the existing dependency-sync test module:

- a multi-code suppression passes when every marker is reported;
- a multi-code suppression fails and identifies the missing marker when only a
  subset is reported.

Update the runtime-fingerprint coverage to confirm `mlx-vlm` availability and
unavailability remain represented after removing the redundant exception block.
Run the focused tests first, then formatting, Ruff lint, and the full quality
gate.

## Documentation

Record the stricter audit and removal of the stale suppression under
`CHANGELOG.md`'s `[Unreleased]` section.
