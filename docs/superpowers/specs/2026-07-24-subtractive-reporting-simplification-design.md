# Subtractive Reporting Simplification Design

Date: 2026-07-24
Status: Approved design
Supersedes: `2026-07-19-canonical-assessment-report-semantics-design.md`

## Purpose

Make `check_models.py` more robust by reducing its reporting and assessment
responsibilities to the two purposes that matter:

1. retain actionable evidence for maintainers of mlx-vlm, mlx-lm, mlx,
   Transformers, model repositories, and adjacent components;
2. present complete model output and practical resource measurements to people
   choosing a local model for image description and keywording.

The current implementation computes too many overlapping scores,
classifications, rankings, and report-specific projections. Those layers have
created contradictory statuses and false precision. This design replaces them
with one small, facts-first assessment and two skim-first human reports.

The implementation is a subtractive refactor. Proven model execution,
provenance, logging, and raw-result capture remain intact. Obsolete code and
artifacts are deleted instead of hidden behind compatibility paths.

## Design Principles

- Capture facts generously and infer causes sparingly.
- Preserve complete output as evidence.
- Prefer a small number of conservative observations to semantic scoring.
- Compute current-run semantics once and render them unchanged everywhere.
- Keep maintainer actionability separate from model-user usability.
- Treat successful but suspicious output as an observation, not an upstream
  failure.
- Keep the primary reports limited to the current run.
- Make unavailable information explicit rather than estimating it.
- Remove code, fields, reports, aliases, and lint suppressions that no longer
  serve the reduced contract.
- Preserve the intentional single-file architecture of `src/check_models.py`.

## Goals

- Keep model discovery, loading, generation, isolation, timing, memory,
  exception capture, prompt diagnostics, and component provenance reliable.
- Replace competing assessment paths with one fully typed immutable current-run
  assessment per model.
- Retain only mechanical or directly observable output observations.
- Generate one maintainer report and one model-user report, plus an HTML copy.
- Preserve the complete generated output in every relevant artifact.
- Restrict automatic issue drafts to hard failures and directly evidenced
  protocol violations.
- Remove A-F grades, 0-100 semantic scores, ownership-confidence scores, and
  historical influence on current recommendations.
- Make `src/check_models.py` materially shorter through net deletion.
- Reduce inline lint suppressions and retain only narrow, documented exceptions
  that cannot be eliminated structurally.

## Non-goals

- Judging general model quality from one image.
- Adding an LLM judge, embeddings, taxonomies, or image-specific rules.
- Diagnosing the owning library from poor generated text alone.
- Automatically filing upstream issues for heuristic output anomalies.
- Maintaining retired report schemas or compatibility aliases.
- Splitting `check_models.py` into production modules.
- Building a new historical benchmarking system.
- Adding multi-image analysis.

## Preserved Execution Boundary

The refactor must not redesign the successful inference path. It preserves:

- cached-model discovery and selection;
- mlx-vlm model and processor loading;
- chat-template and generation-argument forwarding;
- timeout and per-model exception isolation;
- complete generated-text capture;
- prompt, tokenizer, EOS, thinking-token, and special-token diagnostics;
- timing, token, memory, and stop-reason capture;
- model and component revision provenance;
- connectivity failures as indeterminate attempts;
- maximalist runtime logging where it provides raw evidence.

Changes to these areas are permitted only when required to make retained facts
accurate, such as recording configured EOS or thinking markers. They must not
be expanded into new scoring or classification features.

## Canonical Current-run Assessment

Each `PerformanceResult` is converted once into one immutable assessment.
Renderers and serializers consume that assessment without reclassification.

### Execution Status

The execution status is one of:

- `completed`: generation returned without an exception;
- `crashed`: local or upstream execution raised an exception;
- `indeterminate`: an external service or connectivity failure prevented a
  meaningful evaluation.

A successful generation is never assigned a failure origin. For a crash, the
assessment retains factual phase, module, package, exception-chain, and
traceback information without converting those facts into a confidence score.

### Model-user Usability

The current-run usability status is one of:

- `usable`: completed output with the requested structure and no severe
  mechanical problem;
- `usable_with_caveats`: completed, inspectable output with a non-fatal
  mechanical observation;
- `unusable`: completed output that is empty, structurally unusable, severely
  repetitive, or clearly cut off before a usable answer;
- `not_evaluated`: crashed or indeterminate.

This is a current-run structural assessment, not a semantic quality grade.
Users choose among structurally usable outputs by inspecting the complete
responses and considering valid speed and memory measurements.

### Maintainer Status

The maintainer status is one of:

- `actionable_failure`: a hard crash or directly evidenced protocol violation;
- `observation_needs_reproduction`: a successful run produced suspicious or
  unusable output whose cause is not proven;
- `none`: this run contains no maintainer-facing problem.

Only `actionable_failure` can create an issue draft. A poor or bizarre
successful response remains visible with complete evidence but is not assigned
to mlx-vlm, mlx, Transformers, or a model repository until a controlled
reproduction establishes that boundary.

A directly evidenced protocol violation means that recorded runtime facts show
an explicit upstream API contract was breached. Failure to follow the catalogue
prompt, poor prose, missing keywords, or an unexpected-looking response is not
by itself a protocol violation.

### Conservative Observations

The assessment may contain a small fixed tuple of directly observable labels:

- `empty_output`;
- `minimal_output`;
- `repeated_output`;
- `missing_requested_sections`;
- `token_cap_truncation`;
- `prompt_instruction_echo`;
- `unexpected_special_token`;
- `thinking_trace_present`;
- `thinking_trace_incomplete`;
- `no_keyword_overlap`.

Observations must remain narrow:

- truncation requires a recorded stop reason or token-cap evidence;
- repetition requires a repeated contiguous output pattern;
- missing sections apply only when the active prompt requested those sections;
- thinking wrappers consistent with configured model behaviour are
  informational;
- an empty thinking wrapper is not a fault;
- special-token assessment uses tokenizer and configured generation evidence
  where available;
- no keyword overlap is only a smell and cannot by itself make output unusable;
- partial keyword overlap is not scored and omission of individual reference
  keywords is not penalised.

Unknown or ambiguous cases produce no observation. The complete output remains
available for human interpretation.

## Removed Assessment Machinery

The implementation removes machinery that is outside the reduced contract,
including:

- A-F grades and composite catalogue-utility scores;
- 0-100 description, keyword, metadata-agreement, context-integration,
  draft-improvement, visual-description, and enrichment rankings when their
  only purpose is report ranking;
- score deltas against existing metadata;
- semantic winner and loser labels;
- ownership-confidence scores and heuristic suspected-owner promotion;
- successful-run failure origins;
- duplicate review, compatibility, recommendation, readiness, and verdict
  projections;
- legacy status aliases such as report-specific triage buckets;
- historical capability aggregation used to influence current output;
- code and configuration that exist only for retired reports or scores.

Mechanical parsing needed to recognise requested sections, exact prompt echo,
or repetition remains. Deletion must follow callers and tests so that dead
compatibility layers are not left behind.

## Evidence Preservation

Complete output is a primary requirement:

- `results.jsonl` stores the complete generated string as captured;
- every model in `model_gallery.md` contains the complete output in an
  expandable code block;
- every model in HTML contains the complete output in an expandable block;
- every model mentioned in `diagnostics.md` contains its complete output;
- table previews are navigation aids and never substitute for evidence;
- escaping for Markdown, HTML, or JSON must not alter the
  underlying captured text;
- no report-specific shortening or summarisation is applied;
- the configured generation token limit is the only intentional model-output
  limit.

For a crash, evidence priority is different:

1. root exception and traceback;
2. execution phase and component provenance;
3. any partial generated output;
4. captured upstream stdout and stderr.

An empty output is rendered explicitly as empty rather than omitted.

## Human-facing Artifacts

### Maintainer Diagnostics

`src/output/reports/diagnostics.md` is the only maintainer report. Its
skim-first order is:

1. run outcome and affected-model counts;
2. actionable failures;
3. successful observations requiring reproduction;
4. indeterminate attempts;
5. generation settings, image/prompt burden, environment, and provenance.

Each entry includes complete output or complete relevant traceback evidence,
plus a reproduction command. Direct factual fields such as exception module,
processor class, tokenizer class, model revision, stop reason, prompt tokens,
and configured EOS/thinking tokens are preferred to inferred ownership prose.

Issue drafts are conditional supporting artifacts. They are generated only for
`actionable_failure`, contain the same evidence as diagnostics, and do not
introduce another classification. Routine issue queues and reproduction-bundle
indexes are removed when they duplicate `diagnostics.md`, `results.jsonl`, and
`run.json`.

### Model Gallery and Chooser

`src/output/reports/model_gallery.md` is the only Markdown report for model
users. Its order is:

1. a compact current-run chooser;
2. unusable and not-evaluated results that users should avoid;
3. usable models grouped by memory or speed needs;
4. a complete per-model gallery.

The chooser shows:

- current-run usability;
- valid generation speed;
- peak memory;
- output token count;
- concise mechanical observations;
- a short preview linked to the complete output.

It does not claim semantic winners. Models with valid usable output may be
ordered by an explicit resource policy such as lowest memory or fastest valid
generation. A generation too short to measure meaningfully displays
`insufficient sample`; it is excluded from fastest and average throughput
statistics.

### HTML

`src/output/reports/results.html` is a standalone rendering of the same current
chooser, complete model gallery, and maintainer diagnostics. It consumes the
same assessment records and may add navigation, filtering, and expandable
presentation, but it cannot add statuses, scores, or classifications.

### Navigation

A minimal `src/output/index.md` may remain solely to link the two reports, HTML,
and machine artifacts. It contains no independent analysis.

## Machine Artifacts

`src/output/results.jsonl` remains the canonical per-model stream. Each result
contains:

- execution facts;
- complete output or failure evidence;
- timing and resource facts;
- prompt and token diagnostics;
- model provenance;
- the one current-run assessment.

`src/output/run.json` remains the canonical run-level record and artifact
manifest. It contains run configuration, image identity, prompt identity,
component provenance, counts, and paths to retained artifacts.

The schema receives a breaking version increment because obsolete fields and
artifacts are deliberately removed. No compatibility aliases are added.

The append-only raw history may remain as secondary data, but current reports
must not read it or derive capability rankings from it. A future explicit
history tool or appendix is outside this refactor.

The maximalist log may remain as raw operational evidence. It must not emit a
preliminary block labelled canonical if final assessment occurs later.

## Retired Default Artifacts

The following overlapping outputs are removed from the default artifact set:

- `reports/results.md`;
- `reports/review.md`;
- `reports/model_selection.md`;
- `reports/model_capabilities.md`;
- `model_capabilities.json`;
- `reports/results.tsv`;
- routine multi-issue queue summaries;
- reproduction-bundle indexes that duplicate canonical machine evidence.

Documentation, artifact manifests, cleanup logic, tests, and checked-in sample
outputs must be updated consistently. Retired artifacts must not continue to be
generated as silent compatibility copies.

## Data Flow

The retained data flow is deliberately short:

1. execute one model and capture `PerformanceResult` facts;
2. derive one immutable current-run assessment;
3. append the complete per-model JSONL record;
4. after the run, render diagnostics, gallery, and HTML from the same results
   and assessments;
5. write `run.json` with counts and the retained artifact manifest;
6. generate an issue draft only for an actionable failure.

Report renderers do not call detector or classification functions. Machine
serializers do not reconstruct decisions. Historical data is not part of this
flow.

## Robustness Requirements

- Report generation performs no network access.
- Missing optional evidence is rendered as `unavailable`.
- Counts use execution status and cannot be altered by report filtering.
- Connectivity failures reduce evaluated counts and remain indeterminate.
- Successful output anomalies do not become failures or issue-ready defects.
- Throughput aggregation excludes insufficient samples.
- Every retained artifact obtains statuses from the same assessment object.
- Renderer failure for one optional presentation must not corrupt canonical
  JSONL evidence.
- Generated reports use deterministic ordering and escaping.
- Tests and validation write only under temporary or gitignored paths.
- User-generated `src/output/` files are not rewritten merely to run tests.

## Lint and Type Cleanup

The implementation audits existing suppressions in `check_models.py` and
affected tests:

- remove suppressions whose code or renderer is deleted;
- replace broad exception suppressions with narrower exception handling where
  upstream APIs make that reliable;
- replace complexity suppressions by deleting or simplifying the affected
  rendering path;
- use complete, narrow annotations for the retained assessment and renderers;
- reuse upstream `mlx_vlm.generate.types.GenerateKwargs` where it safely
  replaces a duplicated local contract, without making runtime availability
  brittle;
- introduce no bare `noqa`, blanket `type: ignore`, or project-wide exclusion;
- retain a targeted security suppression only when the command or URL is
  structurally constrained and the rule cannot be satisfied another way;
- keep every retained suppression specifically justified and covered by the
  existing suppression audit.

Generated Markdown should use short lists and narrow tables to avoid repeated
inline markdownlint disables. If expandable `<details>` requires an HTML
allowance, prefer one documented configuration allowance for `details` and
`summary` over repeated generated suppression comments.

## Testing Strategy

Tests should be contract-focused and substantially simpler than the retired
report test surface.

### Assessment Tests

Cover:

- completed usable output;
- completed output with a caveat;
- empty and minimal output;
- contiguous repetition;
- required-section absence;
- recorded token-cap truncation;
- exact prompt-instruction echo;
- expected thinking wrappers;
- incomplete thinking trace;
- unexpected special-token output;
- no keyword overlap as a non-decisive smell;
- hard crash;
- indeterminate connectivity failure.

### Cross-artifact Tests

Verify:

- JSONL, diagnostics, gallery, and HTML use identical statuses;
- complete output survives JSON, Markdown, and HTML rendering;
- crash traceback precedes captured secondary output;
- status totals agree with attempted, evaluated, successful, failed, and
  indeterminate counts;
- insufficient throughput samples are excluded from rankings and aggregates;
- no retired artifact appears in `run.json` or the navigation index;
- issue drafts exist only for actionable failures;
- report generation does not read history or access the network.

### Quality Gates

Before completion:

1. run format and safe lint fixes;
2. clear Ruff lint errors;
3. run the suppression audit and remove avoidable findings;
4. run focused assessment and renderer tests;
5. run the full `make quality` gate in the `mlx-vlm` Conda environment;
6. confirm validation did not modify user-generated outputs.

## Migration and Documentation

- Update `CHANGELOG.md` under `[Unreleased]`.
- Update the CLI/output documentation to name only retained artifacts and
  statuses.
- Remove documentation for semantic grades and retired reports.
- Increment the machine-output schema for the breaking simplification.
- Refresh checked-in example outputs only as an intentional final step.
- Preserve unrelated user changes and generated run artifacts throughout the
  implementation.

The 2026-07-19 canonical-assessment design remains useful historical context,
but this design supersedes its multi-report, score-preserving, historical
capability, and confidence machinery.

## Acceptance Criteria

- A run still attempts the same selected models and captures the same raw
  runtime evidence.
- Every successful model's complete output is available in JSONL, gallery, and
  HTML.
- Every diagnostic entry contains complete output; crashes prioritise complete
  traceback evidence.
- Maintainer and model-user statuses are narrow, current-run only, and
  consistent across artifacts.
- No A-F or 0-100 semantic score appears in a primary report.
- Successful suspicious output cannot generate an issue draft without a
  directly evidenced protocol failure.
- Historical data cannot change current-run usability.
- Retired reports and their dedicated code paths are removed.
- `check_models.py` has a substantial net line reduction with no hidden legacy
  compatibility pipeline.
- Avoidable lint suppressions are removed and no unexplained suppression is
  added.
- The full quality gate passes without modifying user-generated outputs.
