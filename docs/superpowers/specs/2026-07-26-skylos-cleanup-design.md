# Skylos Finding Cleanup Design

## Objective

Clear the ten findings reported by `skylos . -a --llm` and the separate
documentation-only `--danger` false positive without weakening production types,
raising thresholds, or adding suppressions. Keep Skylos advisory rather than
making it a blocking gate until the tool matures further.

## Scope

The change covers:

- six `SKY-L012` reports involving valid PEP 695 aliases or Pillow's imported
  `Image` module;
- four `SKY-Q301` complexity findings in current report/log/JSONL helpers;
- the `SKY-D260` prose match in the archived subtractive-reporting design;
- focused regression coverage, the existing full quality gate, and explicit
  post-change Skylos audit and danger scans.

It does not change report semantics, generated artifacts, model classifications,
quality thresholds, ignored Skylos rules, or gate/advisory policy.

## Design

### Scanner-compatible test typing

Retain the production PEP 695 aliases. Tests will stop asking Skylos to resolve
those aliases through the executable `check_models` module:

- use equally narrow test-local `Literal` aliases where literal-domain checking
  matters;
- use the exact structural mapping type for library-version fixtures;
- patch `PIL.Image.open` through the test's direct Pillow import, which is the same
  module object used by `check_models`.

This removes scanner ambiguity without broadening the public API or duplicating
production runtime behavior.

### Complexity reduction

Reduce each flagged function below the configured complexity limit by moving
cohesive construction or collection work into narrowly typed helpers and reusing
existing formatting and metric utilities:

- prompt fact formatting;
- comparison-row/ranking data collection;
- performance-highlight metric collection;
- JSONL metrics and failure record construction.

Helpers must represent meaningful data boundaries, not one-line wrappers. Output
ordering, strings, metrics, failure evidence, and JSON schema remain unchanged.

### Documentation wording

Rephrase the archived phrase that matches Skylos's prompt-injection detector while
preserving its technical meaning. No detector suppression or documentation
exclusion will be added.

## Verification

1. Capture the current ten-finding audit and one-finding danger scan as the red
   baseline.
2. Run the existing focused report, metrics, EXIF, quality-analysis, and JSONL
   tests after each relevant refactor.
3. Run formatting, Ruff autofix/lint, and the full `make quality` gate.
4. Require `skylos . -a --llm` to report zero findings.
5. Require `make skylos-danger-llm` to report zero findings.
6. Confirm no lint suppression, Skylos ignore, threshold, or tracked output change
   was introduced.

## Follow-up archival

After the Skylos cleanup is verified, reduce active maintenance surface without
discarding useful history:

- retire `src/tools/qwen3_vl_sequential_repro.py`, the only manual one-off
  reproducer not used by Make, CI, hooks, setup, updates, or maintained analysis;
- replace its active documentation and dedicated test references with one concise
  historical note under `docs/notes/archive/`;
- move completed Superpowers plans and specifications dated before 26 July 2026
  into `docs/notes/archive/superpowers/`, while retaining the current Skylos plan
  and design in their active directories;
- delete the two byte-for-byte duplicate suppression-audit files whose names end
  in `2.md` instead of retaining duplicate archive copies;
- move the stale May 2026 Skylos quality backlog into `docs/notes/archive/` and
  update the notes index;
- retain the still-valid GPS/EXIF reference and every operational tool with an
  active automation, setup, update, or maintained-analysis role.

Archival must not change runtime behavior, generated output artifacts, or quality
policy. Git history remains the authoritative source for the retired executable.
