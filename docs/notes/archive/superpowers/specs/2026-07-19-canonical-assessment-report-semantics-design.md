# Canonical Assessment and Report Semantics Design

Date: 2026-07-19
Status: Approved design

## Purpose

Make the two purpose-specific classifications, their evidence, and confidence
language consistent across every relevant generated artifact while reducing the
size and complexity of `src/check_models.py`.

The two classifications reflect the outputs' two main audiences:

- a model-user assessment answers whether the model is useful for local image
  captioning and keywording in the current run;
- a maintainer assessment answers whether the observation is actionable for
  mlx-vlm, mlx, transformers, a model configuration, or another owning layer.

They share canonical observed evidence but remain independent decisions. A
model can be unsuitable for users without presenting an issue-ready upstream
defect, and an issue-ready integration failure can say nothing about the
model's caption quality.

The design treats issue publication as a strict boundary: only confirmed
upstream findings may generate issue drafts. Uncertain observations must remain
visible and actionable in diagnostics without being presented as upstream
defects.

## Goals

- Compute each model's observed evidence and canonical decisions once.
- Use those decisions unchanged in Markdown, HTML, TSV, JSONL, logs, issue
  drafts, and historical capability reports.
- Generate issue drafts only for confirmed upstream findings.
- Separate current-run recommendation, historical reliability, compatibility,
  and maintainer readiness.
- Treat reference-keyword overlap as a weak sanity signal, not a recall target.
- State clearly that automated scores are one-image proxy measurements rather
  than human visual ground truth.
- Preserve complete raw output and traceback evidence.
- Make `src/check_models.py` shorter by deleting duplicated classification and
  presentation paths.
- Preserve the intentional single-file architecture.

## Non-goals

- Adding an embedding model, LLM judge, or external semantic evaluator.
- Building image- or model-specific classification rules.
- Requiring models to reproduce every reference keyword.
- Claiming general model quality from one image or one run.
- Splitting `src/check_models.py` into new production modules.
- Adding multi-image analysis.

## Terminology

### Current recommendation

The model-user conclusion for this run:

- `recommended`
- `caveat`
- `avoid`
- `not_evaluated`

### Historical reliability

A separate conclusion derived from lane-compatible history:

- `stable`
- `variable`
- `insufficient_evidence`
- `consistently_unsuitable`

Historical reliability must never overwrite or masquerade as the current
recommendation.

### Maintainer readiness

The publication state of a diagnostic observation:

- `issue_ready`
- `needs_reproduction`
- `harness_observation`
- `not_applicable`

Only `issue_ready` findings may create entries under `src/output/issues/`.

### Compatibility status

Execution and integration status, independent of model usefulness:

- `clean`
- `integration_warning`
- `crashed`
- `indeterminate`

## Architecture

### Observed evidence

Introduce a fully typed immutable internal record containing facts only. It
should include, where available:

- whether upstream mlx-vlm load or generation was invoked;
- execution phase and exception origin;
- exception chain and traceback;
- complete raw generated output;
- stop reason and requested/generated token counts;
- raw image dimensions, megapixels, size, and hash;
- processed image dimensions and patch count;
- text and non-text prompt-token estimates;
- parsed output sections;
- exact structural anomalies;
- controlled-reproduction status and configuration delta;
- reference-keyword overlap state;
- mechanical formatting and repetition observations.

Evidence collection must not assign an owner, recommendation, readiness state,
or issue title.

### Canonical assessments

Introduce one fully typed immutable assessment bundle per result containing two
narrow, purpose-specific assessments.

The model-user assessment should contain:

- execution outcome;
- compatibility status;
- output-anomaly tuple;
- current recommendation;
- concise user-facing evidence codes;
- raw proxy measurements and their declared scope.

The maintainer assessment should contain:

- failure origin;
- maintainer readiness;
- suspected owner and confidence;
- controlled-reproduction status;
- concise maintainer-facing evidence codes;
- next action.

Both assessments are computed from the same immutable evidence and stored by
model in `ReportRenderContext`. Renderers and machine serializers must consume
the relevant cached assessment rather than invoke classification functions
independently. Shared evidence collection should be consolidated; distinct
audience policy must not be collapsed merely to reduce code.

### Data flow

1. Execute the model and retain raw runtime data.
2. Collect observed evidence.
3. Classify canonical model-user and maintainer assessments from the evidence
   and their separate policies.
4. Build audience-specific presentation records from those assessments.
5. Render every artifact from the result, assessment, and presentation record.
6. Filter `issue_ready` assessments before issue clustering and issue-draft
   generation.
7. Aggregate historical reliability separately from current assessment.

## Classification Policy

### Failure origin

Classify origin from the execution boundary and exception chain:

- `harness_preflight`: a local validation or harness assertion failed before
  upstream mlx-vlm load/generation was called;
- `upstream_load`: the invoked upstream load path failed;
- `upstream_generation`: the invoked upstream generation path failed;
- `external_service`: network or remote-service access prevented evaluation;
- `unknown`: available evidence cannot identify the boundary.

Do not infer origin from the model name or a message substring when traceback
origin and invocation state are available.

### Confirmed-only issue gate

The maintainer-readiness decision is a small pure function:

| Evidence | Readiness |
| --- | --- |
| Harness/preflight failure | `harness_observation` |
| External connectivity or service failure | `not_applicable` |
| Unknown failure origin | `needs_reproduction` |
| Heuristic output anomaly without controlled reproduction | `needs_reproduction` |
| Controlled native rerun reproduces the anomaly | `issue_ready` |
| Upstream-origin load/generation exception | `issue_ready` |
| Controlled rerun does not reproduce | `not_applicable` |
| Controlled rerun is itself indeterminate | `needs_reproduction` |

An upstream-origin crash needs no additional reproduction. A harness-origin
failure does not become an upstream issue merely because it prevented the
requested image-description task.

### Output anomalies

Use narrow, non-overlapping anomaly types, including:

- special-token wrapper or leakage;
- thinking trace;
- reasoning budget exhausted before final answer;
- contiguous text repetition;
- contiguous numeric repetition;
- mixed-script corruption;
- encoding corruption;
- missing required sections;
- token-cap truncation;
- irrelevant-output smell.

Coherent reasoning that reaches the output cap is not numeric repetition,
mixed-script corruption, or token soup.

Special-token wrappers must remain in raw evidence. A normalised scoring copy
may remove recognised wrappers so that otherwise valid sections are parsed
correctly. Recognition should use tokenizer/special-token diagnostics and
generic delimiter structure, not model-specific exceptions.

Numeric-loop classification requires genuinely repetitive contiguous numeric
output. Repeated factual numbers distributed through coherent prose do not
qualify.

## Keyword Policy

Reference keywords are indicators only. Define three overlap states:

- `not_assessable`: either side lacks a usable keyword list;
- `no_overlap`: no meaningful normalised keyword or token overlaps;
- `some_overlap`: at least one meaningful overlap exists.

Normalisation remains elementary: case folding, punctuation and whitespace
normalisation, and existing generic singular/plural handling. Do not add
embeddings, semantic categories, subject dictionaries, or fixture-specific
aliases.

Policy requirements:

- Do not enumerate missing reference keywords in primary reports.
- Do not calculate or reward keyword recall percentage.
- Do not penalise omission of individual keywords.
- Do not require exact copying of reference phrases.
- `no_overlap` is a smell, not proof of failure.
- `no_overlap` alone cannot produce `avoid`.
- A stronger conclusion requires independent evidence such as unrelated
  descriptive text or a broken output contract.
- Continue mechanical checks for a missing keyword section, malformed
  separators, excessive count, exact/near duplicates, and repetitive output.

## Recommendation and Ranking Policy

Canonical current recommendation should depend primarily on:

1. successful evaluation;
2. presence of required output sections;
3. absence of severe corruption, repetition, or cutoff;
4. absence of multiple independent irrelevance signals;
5. practical speed and memory only when ordering otherwise usable models.

Do not present a composite score as objective visual quality. Existing
experimental 0–100 measures may remain as secondary diagnostics if they are
clearly named, scoped, and excluded from the canonical recommendation.

Primary chooser tables should use simple, named ordering policies such as:

- lowest memory among recommended models;
- fastest generation among recommended models;
- contract-clean recommended models ordered by a declared proxy measure;
- best caveated fallback when no recommended model meets a resource tier.

Every table must display or state its sort policy.

## Confidence Language

Replace unqualified `grounded` or `visual quality` claims with explicit scope:

> Automated metadata-assisted proxy; one image; no human visual ground truth.

Keep separate measurements for:

- mechanical contract compliance;
- reference-keyword overlap smell;
- basic description/detail proxy;
- draft-change proxy;
- efficiency;
- current recommendation;
- historical reliability.

No report may imply that metadata agreement proves visual accuracy.

## Report Responsibilities

### `diagnostics.md`

The maintainer-first evidence report, ordered as:

1. confirmed issue-ready findings;
2. observations requiring controlled reproduction;
3. harness observations;
4. indeterminate attempts;
5. model-quality observations;
6. environment and provenance.

Every observation must include complete expandable generated output or complete
relevant traceback evidence. Raw image dimensions, megapixels, and text/non-text
prompt burden must appear in run conditions. Unavailable processed dimensions
or patch counts must be labelled unavailable rather than omitted.

### `issues/`

Contains only issue-ready findings. A run with serious but unconfirmed
observations may legitimately produce an empty current issue queue.

Issue drafts contain the complete relevant evidence, native reproduction,
environment, provenance, and an explicit fix signal. Issue clustering occurs
only after readiness filtering.

### `model_selection.md`

Contains current-run model-user recommendations only:

- put Quick Chooser before the full current-run matrix;
- provide a caveated fallback when a memory tier has no recommended model;
- separate not-evaluated, avoid, and caveated fallback sections;
- show the actual sort key in structured-metadata tables;
- move the complete model matrix to an expandable appendix.

### `model_capabilities.md`

Contains historical reliability and must show current recommendation in a
separate column. Replace the ambiguous `Use` values with explicit historical
status and, where useful, a historically strongest task signal. Include run
count, recommended rate, current status, and a concise reason for variability.

### `results.md`

Provides a compact compatibility and performance summary. Winner labels must
state their population, for example `fastest recommended`, `lowest-memory
recommended`, or `fastest completed`.

All counts and winner selections must derive from the canonical assessments.

### `review.md`

Becomes a canonical-decision audit rather than an independently ranked report.
It may show evidence and decisions but must not calculate alternative
recommendations.

### HTML, TSV, JSONL, and `run.json`

These are projections of the same assessment records. Preserve stable public
fields where practical, but add explicit current recommendation, historical
status, maintainer readiness, failure origin, confidence scope, and keyword
overlap state as versioned schema fields.

HTML retains filters, sorting, embedded image, and complete expandable output.
TSV must retain a header-first, rectangular structure.

The human run contract should expose `trust_remote_code`, source revisions,
prompt hash, image identity, and common generation settings.

## Code Simplification

Implementation is deletion-oriented:

1. Compute and cache the two canonical, purpose-specific assessments once per
   result from one evidence record.
2. Retain JSON/TSV record types as serialised views, not competing internal
   models.
3. Use narrow model-user and maintainer presentation records; share factual
   formatting helpers without merging the two audience decisions.
4. Delete duplicate review and maintainer-triage builders after migration.
5. Delete report-specific recommendation and readiness decisions.
6. Filter issue-ready results before clustering, removing repeated readiness
   checks throughout issue rendering.
7. Make historical aggregation consume current assessment rather than
   reconstruct current recommendation.
8. Replace static label, icon, owner, and next-action branches with narrow
   typed lookup tables where clearer.
9. Consolidate repeated table-row construction around canonical presentation
   data.
10. Do not retain compatibility wrappers for private helpers solely to reduce
    patch size.

The final `src/check_models.py` must have fewer lines than it did before this
work. If it grows, revisit the representation and delete duplicate paths before
considering the implementation complete.

## Error Handling

- Preserve raw exception chains and origin paths.
- Record whether the upstream boundary was entered.
- Never convert external connectivity into a package failure.
- Never convert a harness observation into an upstream issue without a
  controlled native reproduction.
- Preserve controlled-rerun configuration and outcome, including an
  indeterminate rerun.
- Report unavailable evidence explicitly.
- Keep raw output immutable even when a normalised scoring copy is used.

## Testing Strategy

Add table-driven tests to existing test modules for:

- harness preflight failure;
- upstream load and generation crashes;
- external connectivity interruption;
- unknown failure origin;
- coherent reasoning reaching the token cap;
- genuine contiguous numeric repetition;
- genuine mixed-script or encoding corruption;
- special-token-wrapped valid output;
- missing required sections;
- zero, partial, and unavailable keyword overlap;
- confirmed, disproved, and indeterminate controlled reproductions.

Add property-style invariants:

- only `issue_ready` assessments create issue drafts;
- raw output is unchanged;
- wrapper normalisation does not hide the wrapper from evidence;
- missing individual reference keywords never causes demotion;
- zero keyword overlap alone never causes `avoid`;
- indeterminate attempts do not count as evaluated failures;
- one current recommendation is reproduced across all artifacts;
- historical reliability cannot overwrite current recommendation;
- production classifiers contain no fixture-, image-, or model-specific terms.

Use unrelated synthetic fixtures covering several subject types. Tests must
write reports only to temporary or gitignored paths and must not rewrite
tracked `src/output/` artifacts.

Run formatting, Ruff lint fixing, Ruff lint, the affected tests, and the full
quality gate in the documented conda environment. Add no lint suppression
unless an existing documented policy proves it necessary.

## Suggested Implementation Sequence

1. Add the evidence and two canonical-assessment types and table-driven
   decision tests.
2. Compute assessments in the shared report context while preserving existing
   serialized fields.
3. Migrate diagnostics and confirmed-only issue generation.
4. Migrate model selection, review, results, HTML, TSV, and JSONL.
5. Separate historical reliability in capability aggregation.
6. Simplify keyword overlap and confidence language.
7. Delete superseded builders, decisions, and presentation branches.
8. Update documentation and `[Unreleased]` changelog entries.
9. Verify generated artifacts in temporary output and run the full quality
   gate.

## Acceptance Criteria

- Harness-origin preflight failures do not generate upstream issue drafts.
- Upstream-origin crashes remain issue-ready without an additional rerun.
- Coherent capped reasoning is not classified as token soup or numeric looping.
- Valid wrapped output retains its sections while wrapper leakage remains
  visible.
- Issue drafts contain only confirmed findings and complete evidence.
- Current recommendations agree across all human and machine artifacts.
- Historical reliability is visibly separate from current recommendation.
- Missing individual reference keywords do not affect recommendation.
- Zero keyword overlap is reported only as a smell and cannot alone cause
  `avoid`.
- Reports describe automated proxy scope and do not claim human-grounded visual
  quality.
- Diagnostics distinguishes raw image burden from unavailable processed-image
  evidence.
- TSV, HTML, Markdown, JSONL, capability JSON, and run manifests validate.
- No production rule contains subject-, image-, or model-specific exceptions.
- No new lint suppressions are introduced.
- The full quality gate passes.
- `src/check_models.py` is shorter than at the start of implementation.
