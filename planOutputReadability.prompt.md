## Plan: Improve Output Readability

Improve human readability across the final human-facing output files by standardizing a small set of reusable output stanzas, replacing dense prose with scan-friendly bullets/tables where appropriate, and keeping one shared content model with thin Markdown/HTML renderers. Focus on file outputs first: `results.md`, `model_gallery.md`, `review.md`, `diagnostics.md`, and generated issue markdown; only touch console/log formatting when a shared helper change directly supports those file outputs. Use the opportunity to rationalize the code that builds the shared content payloads and refresh test coverage for the new output structure, but avoid a full templating system or major schema overhaul in favour of lightweight builders and renderers that slot into the existing report generation flow. A more consistent presentation of key diagnostics should make it easier for maintainers to quickly understand the state of each model and identify next steps without digging through dense paragraphs or hunting for key signals in inconsistent formats.  Any significant reduction in the size of generating code should also make it easier to maintain and evolve the reports over time.

## Steps

1. Phase 1 — Define the stanza taxonomy and scope. Inventory the repeated human-facing content that appears across reports and classify it into a short list of reusable stanza types.
   Depends on: none.
   Stanza candidates to formalize:
   - review verdict / recommendation block
   - maintainer triage block
   - failure detail block
   - token/timing summary block
   - priority/action-summary block
   - full output / traceback details block
2. In the same phase, set presentation rules for scanability so each stanza has a consistent shape.
   Rules to encode:
   - key diagnostics should appear as bullets, short labeled rows, or tables rather than wrapped prose paragraphs
   - section order should lead with “at a glance” content, then evidence, then verbose details
   - tables should be reserved for multi-model comparison; per-model diagnostics should use short labeled lists
   - verbose material should move into details blocks or later sections
3. Phase 2 — Build shared stanza payload helpers in `/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py` using lightweight ordered row/section builders, not a new report framework.
   Depends on: 1-2.
   Recommended shared payload builders:
   - unify `_build_review_block_rows()` and `_build_markdown_review_block_rows()` behind one canonical review payload with format-specific label/text transforms
   - keep `_build_maintainer_triage_rows()` as the canonical maintainer payload and reuse it more broadly where maintainer diagnostics appear
   - extract a reusable failure-detail payload from `_build_gallery_error_block_lines()`, diagnostics failure clusters, and issue-template generation
   - extract reusable token/timing summary payloads from gallery success blocks and any diagnostics/review sections that surface the same data
4. Add thin format renderers for those payloads instead of duplicating wording inline.
   Depends on: 3.
   Renderer approach:
   - Markdown renderer should emit bullets or short labeled rows via existing helpers such as `_append_markdown_labeled_value()` and `_append_markdown_details_block()`
   - HTML renderer should reuse the same payloads but choose list/table wrappers appropriate for the page
   - preserve existing central numeric formatting through `format_field_value()`
5. Phase 3 — Migrate the most cluttered report sections first, in priority order.
   Depends on: 3-4.
   Priority slice 1:
   - `/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py` gallery per-model review/error blocks from `_build_gallery_success_block_lines()` and `_build_gallery_error_block_lines()`
   - goal: concise “Recommendation / Key signals / Maintainer routing / Token summary / Suggested next step” top block, followed by output or traceback
6. Priority slice 2:
   Depends on: 5.
   - diagnostics failure clusters and harness/preflight sections in `generate_diagnostics_report()` and related helpers
   - goal: standard reusable “Observed / Likely owner / Why it matters / Suggested next step / Evidence” stanza instead of dense prose or repeated ad hoc label blocks
7. Priority slice 3:
   Depends on: 5.
   - review digest sections in `generate_review_report()` and shared priority/action-summary helpers such as `_format_review_priorities_parts()`, `_format_action_snapshot_parts()`, and `_format_failures_by_package_parts()`
   - goal: consistent columns/labels across user buckets, maintainer queues, and package/action summaries
8. Priority slice 4:
   Depends on: 5-7.
   - issue markdown generation via `_build_issue_triage_section()` and nearby failure-report builders
   - goal: reuse the same maintainer/failure stanzas already used in diagnostics, so issue templates read like a focused subset of diagnostics rather than a separately worded report
9. Phase 4 — Tighten the section hierarchy and spacing across the final output files after stanza reuse is in place.
   Depends on: 5-8.
   Specific cleanup pass:
   - normalize heading depth and “at a glance” ordering across `results.md`, `model_gallery.md`, `review.md`, and `diagnostics.md`
   - ensure each report opens with a compact summary block before dense evidence
   - convert any remaining paragraph-style diagnostic summaries into bullets or compact rows where they communicate discrete facts
10. Phase 5 — Refresh tracked artifacts and test coverage for the new wording/structure.
   Depends on: 5-9.
   Update focused assertions in existing tests instead of adding new test files.
   Target files:

- `/Users/jrp/Documents/AI/mlx/check_models/src/tests/test_report_generation.py`
- `/Users/jrp/Documents/AI/mlx/check_models/src/tests/test_markdown_formatting.py`
- `/Users/jrp/Documents/AI/mlx/check_models/src/tests/test_html_formatting.py`
- any diagnostics-specific or gallery-specific assertions that pin exact labels/headings

1. Regenerate tracked markdown artifacts after the formatter changes land.
   Depends on: 10.
   Likely artifacts:

- `/Users/jrp/Documents/AI/mlx/check_models/src/output/reports/model_gallery.md`
- `/Users/jrp/Documents/AI/mlx/check_models/src/output/reports/diagnostics.md`
- `/Users/jrp/Documents/AI/mlx/check_models/src/output/reports/review.md`
- `/Users/jrp/Documents/AI/mlx/check_models/src/output/reports/results.md` if shared stanzas affect top summaries there

1. Phase 6 — Run focused validation, then broaden only if needed.
   Depends on: 10-11.
   Recommended order:

- focused pytest slices for report generation and markdown formatting
- Ruff and mypy on touched files
- repo wrappers for ty and pyrefly on `/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py`
- markdownlint on changed markdown artifacts
- optional full `make quality` after the focused checks are green

## Relevant Files

- `/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py` — all human-facing report generators and the shared formatting helpers live here; primary functions include `generate_markdown_report()`, `generate_markdown_gallery_report()`, `generate_review_report()`, `generate_diagnostics_report()`, `_build_review_block_rows()`, `_build_markdown_review_block_rows()`, `_build_maintainer_triage_rows()`, `_append_markdown_labeled_value()`, `_append_markdown_details_block()`, `_format_action_snapshot_parts()`, `_format_review_priorities_parts()`, and `_format_failures_by_package_parts()`.
- `/Users/jrp/Documents/AI/mlx/check_models/src/tests/test_report_generation.py` — broad coverage for report content, headings, links, and triage blocks.
- `/Users/jrp/Documents/AI/mlx/check_models/src/tests/test_markdown_formatting.py` — markdown-specific wrapping, emphasis, blockquote, and gallery formatting checks.
- `/Users/jrp/Documents/AI/mlx/check_models/src/tests/test_html_formatting.py` — HTML structure and presentation assertions for shared report content.
- `/Users/jrp/Documents/AI/mlx/check_models/src/output/reports/model_gallery.md` — tracked artifact useful for before/after readability review.
- `/Users/jrp/Documents/AI/mlx/check_models/src/output/reports/diagnostics.md` — tracked artifact with dense diagnostic content and existing maintainer triage sections.
- `/Users/jrp/Documents/AI/mlx/check_models/src/output/reports/review.md` — tracked digest artifact that should align with the same reusable stanza language.
- `/Users/jrp/Documents/AI/mlx/check_models/src/output/reports/results.md` — summary artifact whose top-level action/priority blocks should stay aligned with the same phrasing.
- `/Users/jrp/Documents/AI/mlx/check_models/CHANGELOG.md` — update under `[Unreleased]` for maintainer-visible output/readability changes.

## Verification

1. Run focused report-generation tests for gallery, diagnostics, review, and shared action-summary sections in `/Users/jrp/Documents/AI/mlx/check_models/src/tests/test_report_generation.py`.
2. Run markdown-formatting coverage in `/Users/jrp/Documents/AI/mlx/check_models/src/tests/test_markdown_formatting.py` to catch wrapping/list/details regressions.
3. Run HTML-formatting coverage in `/Users/jrp/Documents/AI/mlx/check_models/src/tests/test_html_formatting.py` if any shared stanza is rendered in HTML.
4. Run Ruff and mypy on the touched files, then the repo wrappers for ty and pyrefly on `/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py`.
5. Run markdownlint on each regenerated markdown artifact.
6. Manually review the tracked `model_gallery.md`, `diagnostics.md`, and `review.md` outputs for scanability: key diagnostics should be discoverable in the first screenful of each section without reading a dense paragraph.

## Decisions

- Included scope: human-facing final output files and generated issue markdown.
- Excluded scope: TSV/JSONL schema changes and console/file-log presentation, unless a shared helper change is required to support clearer file outputs.
- Recommended reuse strategy: small shared stanza payload builders plus thin renderers, not a new templating subsystem.
- Keep `format_field_value()` as the single numeric/text metric normalizer; do not duplicate formatting logic while improving stanza structure.
- Preserve existing markdownlint/test-sensitive behaviors around blockquotes, details blocks, and heading names unless there is a clear readability gain and the test/fixture updates are planned alongside it.

## Further Considerations

1. Recommendation: treat HTML as a first-class consumer of the same stanza payloads, but allow HTML to remain denser than Markdown when tables are the clearer representation.
2. Recommendation: keep the gallery’s more human-readable wording layer separate from canonical log wording, but source both from the same underlying payload fields so content stays consistent even when phrasing differs.
3. Recommendation: prioritize the top-of-section “at a glance” blocks first; that will deliver the largest readability gain before deeper section-by-section cleanup.
