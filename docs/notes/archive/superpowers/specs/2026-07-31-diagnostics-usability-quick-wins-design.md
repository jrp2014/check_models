# Diagnostics and Gallery Quick Wins Design

**Date:** 2026-07-31

**Status:** Approved scope awaiting written-spec review

## Goal

Implement four presentation and navigation improvements identified in
`docs/notes/DIAGNOSTICS_USEFULNESS_RECOMMENDATIONS.md` without changing runtime
inference, cached assessment semantics, machine-readable schemas, or the retained
artifact set.

## Chosen approach

Make narrow changes in the existing report renderers and documentation. Reuse
`ResultAssessment`, existing gallery rows, existing report blocks, and generated
issue paths. Do not introduce new assessment axes, ranking scores, report files,
or command-line flags.

Two broader alternatives were rejected for this change:

- A new slim aggregate diagnostics artifact would address more of the maintainer
  recommendations, but expands the artifact contract and requires new output-path
  plumbing.
- A new recommendation or scoring layer could drive selector ordering, but would
  conflict with the facts-only, no-winner contract and duplicate cached assessment
  logic.

## Behaviour

### 1. Usable-first gallery presentation

The Markdown chooser, HTML chooser, and Markdown complete-evidence section will
use one shared ordering policy:

1. `usable`
2. `usable_with_caveats`
3. `unusable`
4. `not_evaluated`

Models with the same usability remain ordered by model identifier. Existing
lowest-memory, fastest-generation, avoid, filtering, and sortable-table behaviour
is unchanged. The ordering remains a presentation choice based solely on the
cached assessment.

### 2. Crash facts above optional traceback detail

Crash evidence will lead with the existing root exception chain, followed by the
existing execution/provenance facts. The complete traceback remains unmodified and
available, but direct issue drafts render it in a collapsed details block so phase,
package, root error, and resolved revision stay above the fold. The full aggregate
diagnostics workbook continues to render actionable crash tracebacks expanded.

This requires a narrow presentation option on the shared crash-evidence builder;
it must not alter captured traceback data or machine artifacts.

### 3. Issue-draft navigation from the output index

`generate_output_index_report` will accept the issue-report mapping produced during
diagnostics generation. When crash drafts exist, the index adds an `Issue drafts`
section containing one model-labelled link per generated draft, ordered by model
identifier. When no drafts exist, the index remains the existing seven-link file
with no empty issue section.

The report orchestration will pass the already-generated mapping into the index
job. It will not rediscover files from disk, so stale or unrelated files cannot
enter the current-run index.

### 4. Advertise the triage caption lane

The README evaluation-lane guidance will explicitly identify `triage` as the lane
for comparing plain caption output and will add a runnable `--eval-mode triage`
example alongside the existing blind and assisted examples. No prompt, token cap,
or CLI behaviour changes.

## Error handling and compatibility

- No new filesystem discovery or failure mode is introduced.
- Issue links are conditional on successfully generated issue reports.
- Existing relative and GitHub Markdown link styles apply to issue-draft links.
- Existing exact evidence, home-path sanitization, assessment fields, JSONL schema
  `2.0`, and `run.json` artifact contract remain unchanged.
- The intentional `src/check_models.py` monolith remains intact.

## Testing

Tests will be added or updated in existing files before production changes:

- `src/tests/test_report_generation.py` will verify usable-first ordering in
  Markdown and HTML, complete-evidence ordering, crash-fact placement, collapsed
  issue tracebacks with exact evidence retained, and conditional issue links in
  both relative and GitHub link styles.
- `src/tests/test_metrics_modes.py` exact index expectations will continue to prove
  that runs without issue drafts retain the seven-link index.
- `src/tests/test_dependency_sync.py` will verify the README triage-caption wording
  and example if the existing documentation-policy tests provide the appropriate
  home for that contract.

After focused red/green cycles, validation will follow repository policy: activate
the `mlx-vlm` Conda environment, run formatting, Ruff lint-fix/lint, commit hygiene,
and the full `make quality` gate. Generated validation artifacts will be written to
temporary paths rather than tracked `src/output/` snapshots.

## Documentation and release notes

Update `src/README.md` for the triage example and chooser behaviour. Record the
maintainer-visible report improvements under `[Unreleased]` in `CHANGELOG.md`.
