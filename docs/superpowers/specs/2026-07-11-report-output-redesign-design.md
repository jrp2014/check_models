# Report Output and Assisted Evaluation Redesign

**Status:** Proposed implementation specification  
**Date:** 2026-07-11  
**Scope:** `src/check_models.py`, existing tests, report documentation, and tracked
benchmark artifacts

## Summary

`check_models` will continue to exercise mlx-vlm across locally available VLMs,
but its generated output will be reorganized around two clearly separated
audiences:

1. mlx-vlm and related stack maintainers who need a self-contained, skimmable,
   actionable issue report; and
2. local model users who need defensible quality, compatibility, latency, and
   memory comparisons.

The redesign keeps the current filenames and the HTML report. It changes how
facts are classified, scored, and presented. In particular, assisted evaluation
will distinguish authoritative structured metadata from fallible LLM-written
descriptive metadata, and prompt burden reporting will distinguish visual input
from textual instructions.

Implementation must reuse the existing report context, diagnostics snapshot,
issue clustering, repro-command, table-rendering, history, and artifact-writing
infrastructure. It must not introduce a second reporting pipeline. Consolidation
should remove more production code than it adds where practical; any net growth
in `src/check_models.py` requires an explicit justification in the implementation
handoff.

## Goals

- Make `diagnostics.md` suitable for direct use as an mlx-vlm GitHub issue.
- Treat any crash during the requested image task as a definitive compatibility
  failure, independently of confidence in component ownership.
- Preserve the exception chain and communicate owner attribution separately.
- Distinguish visual-token/image burden from textual prompt burden.
- Make assisted evaluation measure useful enrichment of trusted context and
  improvement of fallible draft metadata.
- Make model rankings explainable, role-specific, and consistent across reports.
- Keep `results.html` as a primary human-readable artifact.
- Preserve stable paths and machine-readable backward compatibility.
- Reduce duplicated report shaping and rendering code.

## Non-goals

- Splitting the intentional `src/check_models.py` monolith.
- Adding an external or hosted LLM judge.
- Automatically opening upstream issues.
- Claiming general model capability from a single image or a single run.
- Requiring differential reproductions before reporting that a model crashed.
- Removing current report files or machine-readable fields in this release.
- Redesigning model discovery, model loading, or the mlx-vlm generation call.

## Accepted Product Decisions

### Crash semantics

A model that raises while asked to describe an image has failed the task. The
human-facing task outcome is therefore `crashed`, regardless of whether the
exception ultimately belongs to mlx-vlm, MLX, model configuration, memory
pressure, Transformers, or another layer.

Root-cause confidence is a separate dimension. Differential runs may improve
owner attribution, but their absence must not weaken or hide the compatibility
failure.

### Metadata provenance

Extracted metadata has two provenance classes:

1. **Authoritative structured context** — GPS/resolved location, capture date and
   time, and equivalent structured fields that the harness treats as factual.
2. **Draft descriptive metadata** — title, description, and keywords previously
   produced by an LLM. These are suggestions to verify and improve, not ground
   truth.

An ideal assisted model combines authoritative location context with accurate
visual observations, corrects weak draft descriptions, and adds useful visible
detail. It is not penalized merely for using authoritative nonvisual context.

### Primary output structure

The primary artifacts are:

- `index.md` — concise audience router and current-run contract;
- `reports/diagnostics.md` — self-contained mlx-vlm issue report;
- `reports/results.html` — retained full HTML report;
- `reports/model_selection.md` — practical user shortlist;
- `reports/model_gallery.md` — complete per-model output evidence; and
- `results.jsonl` — complete machine-readable run evidence.

The following remain supported but secondary: `results.md`, `review.md`,
`model_capabilities.md`, `model_capabilities.json`, `results.tsv`, `run.json`,
issue drafts, history, logs, and repro bundles.

No new HTML artifact is required. `results.html` is the retained HTML copy and
must expose the same canonical run, diagnostic, and selection facts as the
Markdown entry points without duplicating their data-shaping logic.

## Architecture and Reuse Strategy

### Extend canonical contexts instead of adding parallel pipelines

The implementation will extend existing data flow:

```text
PerformanceResult + MetadataDict + PromptDiagnostics
                    |
                    v
          ReportRenderContext
          DiagnosticsSnapshot
          IssueCluster
                    |
          canonical view builders
                    |
      Markdown / HTML / JSONL renderers
```

The following existing internals are the required integration points:

- `_build_report_render_context()` and `ReportRenderContext` for shared
  run/model analysis;
- `_build_diagnostics_snapshot()`, `DiagnosticsSnapshot`, `DiagnosticsContext`,
  and `IssueCluster` for diagnostic grouping;
- `_build_issue_clusters()` and the existing issue queue helpers for owner and
  cluster derivation;
- `build_native_mlx_vlm_repro_command_spec()` for inline upstream-native repros;
- `build_check_models_repro_command_spec()` and repro bundles for supplemental
  harness evidence;
- `_extract_trusted_hint_bundle()` and the current metadata agreement pipeline
  for prompt/reference parsing;
- `PromptDiagnostics` and `ImageInputProfile` for burden accounting;
- the existing report block/table/escaping infrastructure for Markdown and
  HTML formatting;
- `ReportGenerationInputs`, `_build_report_artifacts()`, and the safe artifact
  writers for output orchestration; and
- current JSONL/history serializers for optional provenance fields.

New dataclasses are allowed only when they replace several loosely coupled
tuples/dictionaries or remove repeated derivation. One-use wrapper types and
renderer-specific copies of the same facts are prohibited.

### Code-size budget

The implementation plan must identify deletions before additions. Expected
consolidations include:

- one canonical issue-queue row builder used by diagnostics, issue index, review,
  and HTML;
- one reproduction summary assembled from existing command specs;
- one run-contract block used by Markdown and HTML;
- one model recommendation view used by selection, capability, and HTML output;
- removal of duplicate issue tables and repeated environment/prompt prose; and
- removal or folding of helpers made redundant by provenance-aware scoring.

The preferred outcome is a net decrease in production Python lines. At minimum,
the affected report/diagnostic/scoring functions must not grow overall without a
documented reason tied to new upstream data that cannot replace existing code.
Tests and documentation are excluded from this size budget.

## Functional Requirements

### 1. Provenance-aware assisted prompt

`_build_cataloguing_prompt()` will render separate context blocks.

```text
Authoritative context:
- Location: Deben Estuary, Woodbridge, England
- Capture date/time: ...

Use this factual context where it improves the catalogue record. Do not claim
that contextual facts are visually observable.

Draft descriptive metadata:
- Existing title: ...
- Existing description: ...
- Existing keywords: ...

Treat this draft as fallible. Retain supported details, correct errors, and add
important visible information.
```

Requirements:

- Blind mode continues to expose no metadata of either provenance class.
- Assisted mode may incorporate authoritative context without requiring visual
  confirmation.
- Draft descriptive claims still require image support or authoritative context.
- Location hierarchy may be useful as keywords without requiring every hierarchy
  element in title or prose.
- Capture date/time should be available but not mandatory in prose when it adds
  no cataloguing value.
- User-provided prompts remain unchanged.

Metadata classification should reuse `MetadataDict` and existing extraction
fields. Add a single shared provenance view only if it reduces repeated field
selection in prompt construction, scoring, and reports.

### 2. Provenance-aware scoring and labels

Assisted evaluation will report four separate signals:

1. **Visual-description quality** — structure, specificity, useful visible
   detail, and current visual/output heuristics. This remains explicitly
   heuristic unless human references are supplied.
2. **Authoritative-context integration** — useful incorporation of trusted
   location and other structured facts.
3. **Draft improvement** — novelty, correction, and useful expansion relative
   to the prior LLM draft.
4. **Output quality** — requested sections, concision, readability, repetition,
   keyword count/diversity, and clean termination.

Requirements:

- Blanket `metadata_borrowing` must not be a negative assisted-mode label for
  correct location/context use.
- Human-facing labels should use `unverified-context-copy` only when output
  reproduces context in a misleading way, such as presenting GPS or dates as
  visible image content.
- Exact or near-exact copying of draft descriptive metadata may be labelled
  `low-draft-improvement`; it must not be called hallucination without visual
  evidence.
- Missing authoritative primary location is a context-integration omission, not
  a visual-semantic failure.
- Missing a draft keyword is not automatically an error.
- Existing machine fields should remain readable. If `metadata_borrowing` is
  retained for schema compatibility, document it as deprecated and derive new
  presentation labels from the provenance-aware signals.
- Current utility/capability composites may consume the new sub-scores, but all
  reports must show the ranking policy and disqualifying gates.

### 3. Image burden versus textual prompt burden

Prompt diagnostics must distinguish:

- original image dimensions and megapixels;
- processor-visible/resized dimensions when available;
- tile/patch count when available;
- measured or estimated visual tokens;
- measured or estimated textual instruction tokens;
- template/system tokens when separable;
- total prompt tokens; and
- confidence/source for each estimate.

Requirements:

- Never label `total - estimated text` as exact visual tokens when template or
  special tokens may be included.
- Use `visual/input burden`, `text burden`, or `mixed burden` in human reports.
- Reserve `long text context` for genuinely large textual inputs.
- Thresholds belong in `quality_config.yaml`/`QualityThresholds`, not inline
  renderer code.
- Existing `PromptDiagnostics` and `ImageInputProfile` should be extended rather
  than replaced.
- Missing upstream measurements must render as unavailable, not as zero.
- The report should show whether a large original image was resized before
  processing; original megapixels alone must not imply a large model input.

### 4. Crash and exception presentation

Every unsuccessful generation/load result must appear as a crash or explicit
failure in diagnostics and model-user reports.

The canonical failure presentation derives from existing `PerformanceResult`
fields and contains:

- task outcome;
- failure phase and stage;
- primary/root exception;
- secondary exception raised during cleanup/synchronization, if present;
- suspected owner and owner confidence;
- affected model/revision;
- current-run reproduction count; and
- short impact statement.

Requirements:

- Do not require controls or repeated runs before reporting the crash.
- Do not collapse a multi-exception chain into a contradictory single cause.
- When an `IndexError` is followed by Metal OOM during synchronization, show the
  chronology and mark ownership unresolved unless current evidence proves it.
- “Crashed” is never softened to a model-quality caveat.
- Repro history is supplemental confidence, not a prerequisite.

Prefer a shared failure narrative builder over adding duplicate stored fields.
Add an optional exception-chain JSON field only if the existing root/error/
traceback fields cannot represent chronology without reparsing prose.

### 5. `diagnostics.md` as an mlx-vlm issue report

The report must be understandable without opening another generated document.
Links remain useful for full evidence but cannot contain the only reproduction
instructions.

Required section order:

1. **Title and impact summary** — tested models, crashes, confirmed/probable
   integration findings, mlx-vlm version/revision, hardware.
2. **Run conditions** — evaluation lane, image/input profile, generation
   settings, authoritative metadata exposure.
3. **Crash/failure matrix** — all task crashes, regardless of owner confidence.
4. **mlx-vlm issue matrix** — confidence, symptom, models, reproduction status,
   suspected layer, and evidence link.
5. **Per-cluster evidence** — expected result, actual result, minimal upstream
   traceback/output, one representative native mlx-vlm command, and acceptance
   signal.
6. **Input-burden analysis** — visual versus textual burden, only where relevant
   to a finding.
7. **Probable/untriaged observations** — clearly separated from confirmed
   defects.
8. **Compact environment appendix** — key versions/revisions and link to full
   environment data.

Requirements:

- Inline a representative native command using
  `build_native_mlx_vlm_repro_command_spec()`.
- State expected and actual behavior for each issue cluster.
- Include image filename, dimensions, SHA-256, and availability instructions.
- Avoid private absolute paths in copy/paste commands.
- Include git SHA/branch/dirty state for editable mlx-vlm and MLX installs when
  available.
- Treat model/config-only quality observations as an appendix or omit them from
  the mlx-vlm issue body.
- Remove the complete list of unflagged models and general benchmark rankings.
- Move exhaustive environment paths/hashes and full prompts to linked supporting
  artifacts unless directly necessary for reproduction.
- Replace the current ultra-wide issue table with a compact matrix or short
  evidence cards that render well on GitHub.
- Coordinate-language output must not be described as token soup merely because
  a numeric-loop detector fired.

### 6. Text-sanity evidence gate

An automatically generated issue claim must be supported by its displayed
excerpt.

Requirements:

- Numeric-loop detection must exclude ordinary coordinates, timestamps,
  dimensions, exposure fractions, and other structured numeric metadata.
- A `gibberish`, `mixed-script`, or `token-soup` label requires a sample that
  visibly contains the asserted defect.
- Detector classification and rendered explanation must come from the same
  canonical signal object.
- Unsupported detector findings may remain machine-readable warnings but cannot
  produce an upstream issue cluster.
- Reuse the existing regex/detector aggregation pipeline; do not add a separate
  issue-only detector pass.

### 7. Model-selection output

`model_selection.md` must explicitly state the evidence scope: lane, image/run
count, history count, and whether the result is a single-image shortlist or a
multi-image capability assessment.

Compatibility, quality, and efficiency remain independent dimensions:

- **Compatibility:** crashed, integration warning, completed cleanly.
- **Quality:** visual-description heuristic, context integration, draft
  improvement, structure/output quality.
- **Efficiency:** first-token latency, total task latency, generation throughput,
  peak memory, and post-load peak delta.

Requirements:

- A crashed model cannot appear in a usable shortlist.
- A harness/integration warning must be visible even when quality is high.
- “Fastest usable” requires explicit compatibility and minimum-quality gates.
- Every winner states its policy, such as raw quality, reliability-gated quality,
  best under a memory budget, or best quality/latency balance.
- Use Pareto-style choices rather than relying only on one composite score.
- Group related quantizations/model variants where repository naming permits and
  show quality, memory, and latency deltas.
- TPS remains available but cannot be the sole latency measure across different
  tokenizers/output lengths.
- Single-image results must be labelled as shortlisting evidence, not general
  model capability.

### 8. HTML report retention

`results.html` remains a primary artifact and must retain complete model output
access.

Requirements:

- Show the same canonical run contract, crash facts, burden terminology,
  provenance-aware scores, and ranking policies as Markdown reports.
- Continue to include sortable/scannable model results and links to complete
  evidence.
- Reuse shared view rows/blocks; do not independently recompute diagnostic or
  selection facts in the HTML renderer.
- Preserve safe escaping and current standalone-file behavior.

### 9. Other artifact roles

- `index.md` routes users and maintainers to the primary artifacts and includes
  only the current-run contract and top-level counts.
- `model_gallery.md` remains the complete evidence record, with per-model output,
  task outcome, score components, input burden, time, and memory.
- `results.jsonl` carries optional provenance, burden, failure chronology, and
  score-component fields.
- `results.md`, `review.md`, and capability artifacts may retain detailed views
  but must not contradict primary recommendations. They should consume the same
  canonical recommendation and issue views.
- TSV remains spreadsheet-oriented and adds new scalar fields without embedded
  prose where possible.

## Ranking and Evidence Policy

Reports must not present different “best model” answers without naming why they
differ. The canonical recommendation view will expose:

- eligibility gate result;
- ranking policy identifier and human label;
- component scores used;
- evidence scope (`images`, `runs`, `lane`, `history_runs`);
- caveats; and
- the underlying model result link.

Suggested policies are:

- highest raw assisted-enrichment score;
- highest reliability-gated assisted-enrichment score;
- best under 4/8/16/32 GB;
- fastest eligible by total task time;
- lowest memory above a quality threshold; and
- best quality/latency Pareto candidate.

Thresholds and weights remain centralized in quality configuration. Reports
must not introduce renderer-specific scoring.

## Data and Schema Compatibility

- Keep JSONL format version `2.0` when adding optional fields.
- Keep history format version `1.0` when adding optional fields.
- Preserve current filenames and CLI arguments.
- Preserve `stress` and `quality` as deprecated selection aliases, never stored
  as separate lanes.
- Keep existing JSON keys readable. New provenance-aware fields are additive.
- If a field’s semantics change, add a replacement field and deprecate the old
  interpretation in documentation rather than silently changing consumers.
- Old history rows without new provenance/burden fields remain loadable and are
  excluded only where existing lane-isolation policy already requires it.

Candidate optional machine fields include:

```text
metadata_provenance.authoritative_context
metadata_provenance.draft_descriptive_fields
scores.context_integration
scores.draft_improvement
scores.visual_description_heuristic
prompt_diagnostics.text_tokens
prompt_diagnostics.visual_tokens
prompt_diagnostics.template_tokens
prompt_diagnostics.token_count_source
image_profile.processed_dimensions
image_profile.patch_count
failure.exception_chain
failure.owner_confidence
```

Exact nesting should follow the existing JSONL record conventions and avoid
duplicating values already present elsewhere in the record.

## Error Handling and Degraded Data

- With missing descriptive metadata, automatic lane selection continues to use
  blind mode; explicitly requested assisted mode continues to fail validation.
- Missing authoritative location leaves context-integration unscored rather
  than scored as zero.
- Missing draft description leaves draft-improvement unscored.
- Missing processed image/token measurements renders `unavailable` in human
  output and `null`/omitted in machine output.
- Report generation must continue if optional burden or provenance derivation
  fails; log one actionable warning and retain core results.
- Failure reporting must preserve the original traceback and must not replace an
  upstream exception with a local wrapper message.

## Testing Requirements

Use existing test modules; do not create standalone test scripts.

### Prompt and provenance tests

Add cases to existing pure-logic/parameter tests for:

- assisted prompts separating authoritative context from draft descriptions;
- blind prompts exposing neither class;
- correct location being available for enrichment without being described as
  visually observed;
- user-provided prompt preservation; and
- missing authoritative/draft fields.

### Scoring tests

Add cases to existing quality/report tests for:

- correct location use not producing `metadata_borrowing`;
- verbatim draft copying producing low improvement but not hallucination;
- missing draft keywords not automatically becoming semantic failure;
- missing trusted primary location affecting only context integration;
- coordinates not triggering issue-grade numeric-loop/token-soup findings; and
- separate score components driving one canonical recommendation view.

### Failure and diagnostics tests

Add cases to `test_report_generation.py` and existing JSONL tests for:

- every crash appearing in the failure matrix;
- primary and secondary exceptions rendered in chronological order;
- unresolved owner confidence without weakening the crash outcome;
- an inline native mlx-vlm repro command;
- expected/actual behavior and image hash in the issue report;
- model-only observations kept out of the main mlx-vlm issue matrix; and
- unsupported detector excerpts blocked from issue clustering.

### Input-burden tests

Add cases to existing processing/report tests for:

- large image plus short text classified as visual burden;
- small image plus long text classified as text burden;
- mixed burden;
- unavailable processor token details;
- original versus processed dimensions; and
- estimates clearly labelled as estimates.

### Report consistency tests

Add cross-artifact assertions that Markdown, HTML, JSONL, and TSV consume the
same canonical task outcome, recommendation policy, and score components.
Confirm that `results.html` is still generated and contains the retained primary
views.

Generated validation artifacts must go to `tmp_path` or gitignored `test_*`
locations. Tracked `src/output/` snapshots are updated only as an intentional
final fixture refresh.

## Documentation Requirements

Update:

- `src/README.md` for assisted metadata provenance, report roles, score meaning,
  and prompt/input burden terminology;
- `docs/IMPLEMENTATION_GUIDE.md` for canonical report-view reuse and code-size
  expectations;
- `docs/CONTRIBUTING.md` for report consistency and fixture testing; and
- `CHANGELOG.md` under `[Unreleased]`.

The output README/index must identify `diagnostics.md` as issue-ready for
mlx-vlm and `results.html` as the retained full HTML report.

## Acceptance Criteria

The implementation is complete when:

1. A model crash is unambiguously reported as a failed task even when owner
   confidence is unresolved.
2. Multi-exception failures show causal chronology without contradictory
   “root-cause” headlines.
3. `diagnostics.md` can be pasted into an mlx-vlm issue without requiring another
   generated Markdown file for the minimal repro.
4. Model/config-only findings are not presented as confirmed mlx-vlm defects.
5. Coordinate/date metadata cannot create unsupported token-soup issue claims.
6. Assisted prompts and scores distinguish trusted structured context from
   fallible LLM-written descriptive metadata.
7. Correct location enrichment is rewarded; merely copying a weak draft is not.
8. Reports distinguish visual input burden from textual prompt burden and label
   estimates accurately.
9. `model_selection.md` explains eligibility and ranking policy for every winner.
10. `results.html` remains generated and agrees with canonical Markdown/JSONL
    facts.
11. Existing filenames, CLI compatibility, JSONL/history readers, and lane
    isolation remain intact.
12. The implementation reuses existing contexts, issue clusters, repro specs,
    and renderers rather than adding a parallel pipeline.
13. Production Python line growth is avoided where consolidation can replace
    duplication; any unavoidable net increase is documented with added/deleted
    counts and rationale.
14. Focused tests and the full `make quality` gate pass.

## Implementation Boundaries for the Follow-on Plan

The implementation plan should be organized so that each step deletes or folds
old logic as soon as its canonical replacement exists. A suitable dependency
order is:

1. provenance and burden data contracts;
2. prompt/scoring behavior;
3. crash narrative and issue evidence views;
4. diagnostics report consolidation;
5. model recommendation consolidation and HTML reuse;
6. secondary artifact alignment;
7. documentation, tracked snapshot refresh, code-size audit, and full quality
   verification.

This ordering is descriptive, not an implementation plan. The follow-on plan
must name exact functions/tests and include review checkpoints for output
correctness and production-code reduction.
