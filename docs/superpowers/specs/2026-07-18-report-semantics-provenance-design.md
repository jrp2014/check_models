# Report Semantics, Presentation, and Provenance Design

## Purpose

Make the current-run artifacts internally consistent and more defensible for
both audiences of `check_models.py`:

- upstream maintainers need evidence that distinguishes a confirmed library
  fault from an observation that still needs a controlled reproduction; and
- model users need one recommendation vocabulary that means the same thing in
  the gallery, review, selection, HTML, TSV, JSONL, and history artifacts.

The implementation will correct two classifications exposed by the 18 July
run, repair concrete HTML and TSV defects, and record enough local provenance
to reproduce a result without making network access part of report generation.
It will reuse the existing review, triage, recommendation, table, and snapshot
helpers in the intentional `src/check_models.py` monolith. Multi-image analysis
is explicitly outside scope.

## Semantic Model

Execution outcome and recommendation status answer different questions and
must remain separate.

The execution outcome is one of:

- `completed`: the model returned a generation;
- `failed`: the model was evaluated and the attempt failed; or
- `indeterminate`: an external dependency could not be contacted, so the model
  was not evaluated conclusively.

The canonical user recommendation is one of:

- `recommended`: completed, compatible, and presentation-ready for the tested
  lane, with no material output warning;
- `caveat`: completed and potentially useful, but affected by a token cap,
  visible thinking/control syntax, incomplete requested structure, uncertain
  configuration/context behaviour, or another material warning;
- `avoid`: a conclusive failure or severely degraded output makes the model a
  poor choice for the tested lane; or
- `not-evaluated`: the attempt was indeterminate.

`success=True` therefore means only that generation completed. It must not
select a green gallery icon, make a result recommendation-eligible, or imply a
clean triage pass by itself.

The existing `JsonlReviewRecord` remains the canonical policy payload. Its
verdict, user bucket, and evidence will feed `ModelRecommendationView` and all
human and machine renderers. No second, competing disposition classifier will
be added. `clean-triage-pass` remains presentation wording for an ungrounded
clean result, not an independent stored status.

## Maintainer Confidence and Issue Readiness

Maintainer diagnostics need a second distinction: whether evidence is ready to
file as an upstream fault.

- A **confirmed issue** has direct failure evidence or a controlled comparison
  that isolates the suspected component. It may enter the issue matrix and
  generate an issue-ready draft.
- An **observation requiring reproduction** has useful evidence but does not
  isolate a faulty component. It remains visible in diagnostics, including the
  complete model output and a concrete next experiment, but does not become an
  issue draft or a confirmed owner count.
- An **indeterminate attempt** did not evaluate the model and is reported only
  as a retry requirement.

This classification will be derived from the existing review, quality, prompt,
and runtime evidence when the shared diagnostics snapshot is built. Report
renderers will consume the cached result rather than repeat owner heuristics.

### Thinking/template observation

For the current Qwen thinking result, the decisive facts are that thinking was
requested off, the rendered assistant prefix nevertheless opened a thinking
block, the model emitted a closed thinking trace and a usable final answer, and
raw special tokens were retained for diagnosis. This is evidence of a
thinking-mode/template/configuration interaction, not evidence that mlx-vlm
failed to stop at an EOS token.

The general classification rule is:

- a complete thinking trace is never a stop-token fault merely because its
  delimiter remains visible;
- when requested thinking state and the rendered prompt/template disagree,
  classify the result as a configuration/template observation;
- route the first reproduction to model configuration and mlx-vlm template
  integration, with medium confidence and no issue-ready draft; and
- retain requested thinking state, rendered prompt evidence, recognised start
  and end markers, EOS tokens, skip-special-tokens state, and complete output.

An unclosed trace or a run that never reaches a final answer may still receive
the existing degraded/cutoff recommendation, but blame is not inferred from the
delimiter alone.

### Context-boundary observation

For the current PaliGemma result, the prompt contains 4,103 tokens, about 4,097
of them are visual/non-text tokens, and the observed prefill boundary is 4,096.
The single-word output is poor evidence of usability but does not isolate an
MLX long-context failure. The checkpoint is also a pretrained (`-pt-`) variant,
and no reduced-image or alternate-boundary control was run.

The general classification rule is:

- a weak successful generation under a large visual-token burden remains a
  `context_budget` user caveat;
- a single unpaired result is an observation requiring reproduction rather
  than a confirmed `long_context` library issue;
- diagnostics must display total, text, and non-text prompt tokens, processed
  image dimensions and patch count when available, the relevant runtime
  boundary when known, checkpoint/configuration clues, and complete output;
  and
- the next action is a controlled reduced-image or lower-visual-token rerun
  before assigning the problem to mlx, mlx-vlm, or the model.

This rule is evidence-based and must not special-case the PaliGemma model name.

## Canonical Recommendation Use

The existing review payload will be completed before report context creation.
`ModelRecommendationView.eligible` will be derived from the canonical user
bucket plus compatibility and lane-quality facts. Only `recommended` rows are
eligible for best-model, Pareto, memory-budget, and top-candidate lists.

All report surfaces will use the same status:

- the model gallery heading icon and recommendation line;
- review bucket summaries and per-model verdicts;
- model-selection chooser and shortlist eligibility;
- HTML row metadata, filters, and recommendation summary;
- TSV and JSONL recommendation fields; and
- history/capability current-status derivation.

Visible control tags, a complete visible thinking trace, or a token-cap signal
are material presentation warnings. They cannot be labelled `recommended` or
`clean-triage-pass`, although useful output may remain `caveat`. Severe loops,
unusable degeneration, conclusive runtime failures, and semantic mismatches
remain `avoid`. External-connectivity failures remain `not-evaluated` and do
not reduce the number of successful models by pretending to be model failures.

The gallery continues to preserve the complete generated output. This change
alters labels and eligibility, not evidence retention or token-based model
generation limits.

## HTML Repairs

HTML tables will keep display strings separate from sort keys.

- Numeric sort values will be generated by a dedicated, anchored numeric parser
  or from typed metric values. Unit letters such as the `e` characters in
  `GB`/`Memory` must never enter the key.
- `error_package` will be removed from the numeric field set. It is text in all
  renderers.
- Every table will have a concise `<caption>`.
- Column headers will use `scope="col"` and expose `aria-sort`; the sort script
  will update `aria-sort` on the active header.
- Filter status text will use an `aria-live` status region.
- Row metadata will expose execution outcome and canonical recommendation as
  separate attributes. The UI will filter by the named recommendation statuses
  rather than the internal eligibility boolean.

The report remains dependency-free and self-contained.

## TSV Repairs

TSV will be written with Python's delimiter-aware CSV writer using a tab
delimiter and newline terminator. `tabulate(tablefmt="tsv")` will no longer be
used because its alignment padding makes headers and cells unsuitable as a
machine artifact.

Header labels will be HTML-free and stripped. Embedded tabs and newlines in
model text will continue to be normalised so every model occupies one physical
line, and complete generated text will remain exempt from the ordinary compact
cell limit.

The TSV schema will contain a stable compact core:

- model and execution outcome;
- canonical recommendation and compatibility status;
- principal token, throughput, timing, and peak-memory metrics;
- prompt-burden classification;
- complete generated text; and
- concise error fields when failures are present.

Optional score, image-processing, owner, and failure columns will be included
only when at least one row has a value. Columns that are empty for the entire
run will not produce wide blank spreadsheet regions. JSONL remains the stable,
exhaustive machine schema. The TSV metadata comment will identify its format
and direct consumers needing a fixed exhaustive schema to JSONL.

## Provenance

Run-level component provenance will extend the existing version snapshot for
the packages that materially affect generation and image handling, including
mlx, mlx-vlm, mlx-lm, transformers, tokenizers, Pillow, and check_models.

For each available component record:

- installed distribution version;
- installation kind (`editable`, `wheel`, `source-tree`, or `unknown`);
- a publication-safe source location, with the home directory normalised to
  `~`;
- source revision when a local Git checkout can be resolved safely; and
- direct-URL/VCS revision metadata when provided by the installed distribution.

Local discovery must be read-only and must not contact package indexes or
GitHub. Missing provenance is represented explicitly rather than guessed.

Each model result will also record the resolved local Hugging Face snapshot
revision when it can be derived from the selected snapshot path. Requested and
resolved revisions remain distinct. Model provenance will include the model
identifier, requested revision when one was supplied, resolved snapshot
revision, and a home-relative cache/snapshot location when useful. A missing
local snapshot revision remains `unknown`; report generation must not download
or resolve it over the network.

The provenance payload will be reused in JSONL metadata, `run.json`, diagnostic
environment sections, and reproduction bundles. Human reports will use a
compact component table and link to machine artifacts for exhaustive detail.
Public report fields will normalise home-directory prefixes consistently.

## Data Flow and Code Size

The implementation will favour extending existing paths over adding parallel
rendering machinery:

1. Quality analysis produces review evidence once.
2. Report-context construction caches canonical reviews, maintainer triage,
   issue readiness, recommendations, and machine facts.
3. Diagnostics partitions confirmed issues from reproduction observations.
4. Gallery, review, selection, HTML, TSV, JSONL, and history render cached
   status facts.
5. A shared provenance collector extends the existing distribution metadata
   and MLX source-provenance helpers.

Obsolete renderer-specific status derivations, duplicated eligibility checks,
and the aligned TSV writer will be removed where the shared implementation
makes them unnecessary. New lint or type-checker suppressions are not an
acceptable substitute for precise types.

## Testing Strategy

Tests will be added only to existing files and will write to `tmp_path` or
gitignored test-output paths.

`src/tests/test_quality_analysis.py` and
`src/tests/test_report_generation.py` will prove that:

- requested thinking off plus a template-opened, closed trace becomes a
  configuration/template observation rather than a stop-token issue;
- a complete visible thinking trace is a user caveat and is not shortlist
  eligible;
- a single weak result near a visual prefill/context boundary remains a
  context-budget observation and produces no confirmed upstream issue draft;
- diagnostics retain decisive prompt/configuration evidence, complete output,
  and the controlled-rerun action for both observations;
- loops, conclusive failures, and indeterminate connectivity retain their
  distinct canonical meanings; and
- all user-facing report builders consume the same recommendation status.

`src/tests/test_html_formatting.py` and the existing HTML report tests will
prove that:

- memory sort values are valid finite numbers;
- error-package cells are textual;
- captions, scoped headers, `aria-sort`, and the live filter count exist; and
- execution and recommendation filters use separate canonical attributes.

`src/tests/test_tsv_output.py` will prove that:

- output is literal unpadded TSV with stripped headers;
- the compact core has canonical status fields;
- all-empty optional columns are omitted and populated optional columns remain;
- tabs/newlines cannot break row structure; and
- complete generated output is retained once.

`src/tests/test_jsonl_output.py` and existing reproduction/report tests will
prove that:

- component install kind, normalised source, and local revision are captured
  when discoverable;
- unavailable fields are explicit and require no network;
- model requested and resolved revisions remain distinct; and
- the same provenance payload reaches JSONL metadata, `run.json`, diagnostics,
  and reproduction bundles.

Focused tests will run first. Final verification will follow the documented
order: formatting, Ruff lint repair/check, then the complete quality gate in
the `mlx-vlm` conda environment. Validation will not rewrite tracked
`src/output/` assets.

## Documentation and Release Notes

Update `src/README.md` to define execution outcome, recommendation status,
confirmed issue versus reproduction observation, TSV's compact adaptive
columns, and the provenance fields. Add an `[Unreleased]` entry to
`CHANGELOG.md` covering maintainer classification, report consistency, HTML/TSV
repairs, and provenance.

Checked-in reports will not be manually rewritten by tests. If a safe replay of
the current captured run can regenerate them without model execution, it may be
used deliberately; otherwise the improved artifacts will appear on the next
normal benchmark run.

## Out of Scope

- multi-image aggregation, variance, and image-specific failure clustering;
- network resolution of model or package revisions;
- removing or shortening raw generated output;
- changing model generation token limits; and
- splitting `src/check_models.py` into modules.

## Acceptance Criteria

- The two current maintainer signals are observations requiring controlled
  reproduction, not confirmed mlx-vlm or mlx issue drafts.
- Execution and recommendation statuses are distinct and consistent across all
  report surfaces.
- Only presentation-ready outputs are `recommended` and shortlist eligible.
- HTML numeric sorting works for memory and other units, error package is text,
  and the table has the specified accessibility semantics.
- TSV has no alignment padding or wholly empty optional columns and retains
  complete generated evidence.
- Component and model provenance is locally discoverable, publication-safe,batton
  and shared across machine and maintainer artifacts.
- No multi-image behaviour changes.
- `CHANGELOG.md` and `src/README.md` document the contract.
- Focused tests and the complete quality gate pass without new lint or typing
  suppressions and without rewriting tracked output during validation.
