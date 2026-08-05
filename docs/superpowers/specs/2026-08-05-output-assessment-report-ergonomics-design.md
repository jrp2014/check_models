# Output Assessment and Report Ergonomics Design

## Goal

Make retained benchmark outputs classify incomplete responses accurately, treat
properly closed thinking sections as neutral evidence, and present compact,
self-contained maintainer reports without inventing irreproducible commands.

## Assessment semantics

Assessment remains mechanical, image-independent, and model-name-independent.
Thinking delimiters are evidence, not an error by themselves:

- a properly closed thinking block followed by substantive final text is fully
  usable; its markers remain only in machine evidence;
- an opening delimiter seeded by the rendered prompt and a closing delimiter in
  generated text form one complete thinking block;
- an unclosed thinking block remains an incomplete-output observation;
- a closed thinking block without substantive final text is an unusable missing
  final answer;
- credible token-cap evidence already recorded by quality analysis, including an
  abrupt textual tail or dangling Markdown/list syntax, produces a truncation
  observation;
- truncation that leaves the requested response incomplete is unusable.

The canonical assessment is fixed once. JSONL, run JSON, diagnostics, gallery,
HTML, issue summary, and terminal output continue to consume that assessment
rather than applying presentation-specific overrides.

## Reproduction contract

Shared diagnostics and crash reports use the existing publication-safe
reproduction builder.

- When a public source URL and matching SHA-256 are retained, the report provides
  a download, digest verification, and one-process native mlx-vlm command pinned
  to the resolved model revision.
- For a local-only image, the report embeds the exact prompt and records image
  format, dimensions, byte size, and SHA-256. It explicitly withholds a complete
  command instead of referring to an unavailable basename such as `cats.jpg`.
- The gallery image remains a sanitised visual preview. It is not represented as
  byte-identical reproduction input.
- Private absolute paths never enter publication-safe artifacts.

This follows the project-native mlx-vlm reproduction workflow: conda plus pip,
one model per process, native `python -m mlx_vlm.generate`, and no automatic
upstream filing without a confirmed native reproduction.

## Report and terminal consolidation

- Replace the fixed-width comparison strings with the existing Rich table helper
  so terminal width accounts for the log prefix.
- Use one actionability sort key for diagnostics, the run issue summary, and
  completed-model terminal groups.
- Rename `Actionable Failures` to `Crashes requiring action`.
- Remove the redundant `Maintainer` column from completed observed-model tables.
- Rename gallery ranking headings so they state that usable-with-caveats results
  are included.
- Include `issues/run_summary.md` in the run JSON artifact manifest only when the
  canonical assessment says the run has surfaced results.

Implementation should delete or reuse parallel formatting and sorting logic where
possible. No new report-specific classifier or table framework is introduced.

## Validation

Focused tests are written and observed failing before production changes. They
cover:

- closed thinking plus a final answer is usable with neutral marker evidence;
- prompt-seeded thinking opens are completed by a generated close;
- thinking-only, abrupt-tail, and dangling-Markdown token-cap output is unusable;
- local-image diagnostics contain exact facts and no fake runnable command;
- public-image diagnostics remain digest-verified and runnable;
- diagnostics and issue summaries use the same actionability order;
- Rich comparison output, compact completed tables, renamed headings, and the
  conditional artifact manifest.

Generated reports in tests go to temporary paths. The fresh tracked
`src/output/` run is not regenerated or edited. After focused tests, run the
repository-prescribed format, safe lint-fix/lint, and full quality gates.

## Follow-up skills inventory

After the implementation, capture a separate project note that identifies the
existing mlx-vlm skills worth retaining and the highest-value missing recurring
workflows. This is an inventory and prioritised follow-up, not an expansion into
multiple new skill implementations in this change.

## Documentation hygiene

After implementation and verification, inventory the active implementation
specifications and notes. Move clearly completed or superseded material into the
existing `docs/notes/archive/` hierarchy with Git-aware moves so history remains
recoverable. Keep current operational guidance, the skills inventory, and this
change's live specification and plan outside the archive until the work itself is
complete.
