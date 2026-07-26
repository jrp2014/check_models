# Report Evidence Refinement Design

## Purpose

Make the generated reports more useful as evidence without duplicating or
silently shortening model output:

- consolidate the two skim-first tables in `model_gallery.md`;
- preserve complete generated text wherever the consolidated gallery summary
  displays a successful model response; and
- add complete, expandable generated-output evidence to each applicable issue
  section in `diagnostics.md`.

The change must reuse the existing report primitives and generation-result
helpers, keep `src/check_models.py` as a single-file implementation, avoid lint
or type-checker suppressions, and keep validation output out of tracked
`src/output/` paths.

## Current Behaviour

`model_gallery.md` contains both a **Model Quality Summary** and an **All Model
Output and Cost Summary**. Each table independently calls a preview helper that
shortens successful output to a configured character limit. The later
per-model section contains the complete response, so long outputs are shown
three times conceptually but only one copy is complete.

`diagnostics.md` renders classifications, affected models, reproduction
commands, acceptance signals, and prompt-burden evidence. Generated issue
drafts already use shared evidence helpers to show model output, but the main
diagnostics issue sections do not retain the affected successful model's
actual response.

## Gallery Design

Replace the two existing summary sections with one **Model Output and Cost
Summary**. It will retain the useful columns from the cost table:

- model link;
- result/review status;
- complete output or concise failure diagnostic;
- generated-token count;
- total time;
- generation throughput;
- peak memory; and
- quality/diagnostic signal.

For successful results, the table cell must contain the complete generated
text. The renderer may flatten whitespace and escape Markdown so the response
remains a valid pipe-table cell, but it must not apply a character limit,
head/tail selection, ellipsis, or any other content-removing transformation.
The detailed per-model gallery remains unchanged and therefore preserves the
original line structure in its complete-output block.

For failed results, the table may continue to show a bounded diagnostic rather
than an unbounded traceback. Failure diagnostics are harness evidence, not
model-produced text, and full crash evidence remains available in maintainer
artifacts.

The obsolete quality-summary renderer and successful-output preview helpers
that become single-use or unused should be removed. Existing status and quality
signal helpers should be retained and reused by the consolidated table.

## Diagnostics Design

Each scoped issue section in `diagnostics.md` will include a **Generated Output
Evidence** subsection after **Expected and Actual Behaviour** and before the
native reproduction material.

For every affected result with non-empty generated text, render one closed
`<details>` block. Its summary identifies the model and states that the block
contains complete generated output. Expanding the block reveals a fenced
`text` code block containing the response exactly as captured, apart from the
existing safe Markdown/code-block handling.

Use the existing `ReportDetails`, `ReportCodeBlock`, issue-cluster result
selection, generation-text extraction, and Markdown rendering primitives. Do
not introduce a separate diagnostics-only HTML formatter. Multiple affected
models receive separate expandable blocks so evidence remains attributable.

If an issue cluster contains no captured generated text, omit this subsection.
Crash evidence continues to be handled by the crash/failure matrix; this
change does not require a model response as additional proof of a hard crash.

## Data Flow

The change introduces no new persisted fields or schema versions:

1. `PerformanceResult.generation.text` remains the canonical captured output.
2. The gallery consolidated summary reads that text directly for successful
   results, flattens only whitespace, escapes it, and places it in the table.
3. The diagnostics issue renderer iterates the existing cluster results,
   selects non-empty captured text, and passes each complete response through
   `ReportDetails` and `ReportCodeBlock`.
4. Machine-readable JSONL, TSV, history, repro bundles, HTML, and the detailed
   gallery output remain unchanged.

## Testing Strategy

Add tests to the existing `src/tests/test_report_generation.py` and, only if
needed for the shared Markdown primitive, `src/tests/test_markdown_formatting.py`.
Tests must write reports under `tmp_path`.

The tests will prove that:

- the gallery contains one summary heading rather than the two former summary
  headings;
- a response longer than the former preview limit appears through its unique
  final sentinel text in the consolidated table;
- gallery output contains no preview-only head/tail marker for that response;
- runtime, memory, status, and quality-signal columns remain present;
- diagnostics include a generated-output evidence subsection for successful
  issue results;
- each affected model has a closed expandable details block;
- the expanded text code block contains both unique beginning and ending
  sentinels from the complete response; and
- issue sections with no captured model output do not emit an empty evidence
  subsection.

Run the focused report-generation and Markdown-formatting tests first, then
format, lint, and run the full project quality gate in the documented order.

## Documentation and Generated Artifacts

Update the report descriptions in `src/README.md` and add an `[Unreleased]`
entry to `CHANGELOG.md`. The documentation must describe one consolidated
gallery summary with complete successful output and expandable complete output
evidence in diagnostics.

Tracked production reports may be regenerated intentionally through the normal
report-generation/finalization path after implementation. Automated tests must
not rewrite them. If replaying the current run is not supported without model
execution, leave tracked reports for the next normal benchmark rather than
manually fabricating generated artifacts.

## Further Utility Recommendations

The following findings are recommendations only and are outside this
implementation:

1. Record the effective thinking/EOS protocol before treating thinking
   delimiters as confirmed mlx-vlm faults.
2. Move the full reliability-gated model table below the quick chooser in
   `model_selection.md` so the user-facing answer appears first.
3. Add image count, run count, and variance/confidence to recommendations so a
   single-image run is not mistaken for broad model-quality evidence.
4. Reduce the per-model duplication in `review.md`, keeping it focused on
   shortlists, watchlists, and maintainer escalations.
5. Link diagnostics issue-matrix rows directly to their detailed issue
   sections.
6. Record source revision and install provenance consistently for editable
   mlx-vlm and mlx-lm installations as well as MLX.
7. For multi-image evaluations, aggregate failure signatures and identify the
   images that trigger each signature.

## Acceptance Criteria

- `model_gallery.md` has one skim-first output/cost summary.
- Every successful response shown in that summary is complete and has no
  character-based shortening.
- The detailed per-model gallery remains complete and structurally unchanged.
- Applicable `diagnostics.md` issue sections contain complete generated output
  in per-model expandable text-code blocks.
- Hard-crash reporting remains conclusive without requiring generated output.
- No machine-readable schema changes are introduced.
- Focused tests and the full quality gate pass without adding lint or typing
  suppressions.
