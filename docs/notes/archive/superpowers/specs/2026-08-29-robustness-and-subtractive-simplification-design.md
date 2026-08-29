# Robustness and Subtractive Simplification Design

Date: 2026-08-29
Status: Implementation plan requested

## Purpose

Reduce `src/check_models.py` by removing duplicated classifications, machine
schemas, artifact descriptions, CLI compatibility paths, and archaeological
tests. The work must improve failure containment and report accuracy while
preserving the intentional single-file production architecture and the useful
diagnostic evidence collected for model users and mlx-vlm maintainers.

## Priorities

1. A completed model sweep must survive any ordinary report or comparison
   exception and retain every artifact that can still be generated.
2. Every surface that classifies generated text must consume the same
   observation and usability rules.
3. Human navigation must expose only artifacts successfully produced by the
   current run, never stale files left by an earlier run.
4. Machine-readable run facts must have one canonical retained representation.
5. Compatibility code and tests must justify their continuing cost.
6. Net deletion is preferred, but never at the expense of exact evidence,
   isolation, malformed-input handling, security, or reproducibility.

## Global Constraints

- Keep `src/check_models.py` as the intentional production monolith.
- Use Python 3.13 or newer in the `mlx-vlm` conda environment.
- Preserve the three resolved evaluation lanes: `triage`, `blind`, and
  `assisted`; `auto` remains a selection mode, not a persisted lane.
- Preserve complete generated output, failure traceback, captured upstream
  output, model/component provenance, prompt diagnostics, timing, memory,
  telemetry, and reproduction evidence.
- Keep `results.history.jsonl` local-only and on its existing schema unless a
  task explicitly requires otherwise.
- Do not split production logic into new modules.
- Add or modify tests only in existing `src/tests/test_*.py` files.
- Validation must write to `tmp_path` or another untracked path, never to
  tracked `src/output/` artifacts.
- Do not add Pydantic or another schema dependency. Consolidating the schemas
  removes the principal reason to add one.
- Update `CHANGELOG.md` under `[Unreleased]`, the detailed CLI documentation,
  and architecture documentation for every public contract change.

## Chosen Design

### 1. One canonical observation projection

Extract the pure text-to-observation portion of `_assessment_observations` so
both completed `PerformanceResult` assessment and
`tools.analyze_output_quality` consume the same ordered `ObservationCode`
tuple. A second helper will derive the completed-result usability and
maintainer status from that tuple.

Remove the cached `PerformanceResult.quality_issues` string,
`_build_quality_issues_string`, and the tuple-returning `_analyze_text_quality`
wrapper. Human log labels will be rendered from canonical observation codes and
the observation registry. Failure stdout may retain a structured
`quality_analysis` as neutral evidence, but it will not carry a second verdict
string.

Remove `GenerationQualityAnalysis.requested_max_tokens`: the requested limit is
an analysis input and remains on `PerformanceResult`, but the copied analysis
field has no reader.

The standalone analyser keeps its human and JSON modes. JSON output gains the
canonical observation tuple, usability, and maintainer status. Exit status is
nonzero exactly when canonical usability is `unusable`. Empty output, minimal
output, incomplete thinking, thinking-only output, prompt echo, catalogue
preamble, missing requested sections, repetition, and evidenced truncation can
therefore no longer disagree with the main reports.

### 2. Genuine report and comparison isolation

Report generation is an explicit fault-containment boundary. Catch
`Exception`, not a hand-maintained subset, around:

- each ordinary artifact job;
- diagnostics and issue-draft generation;
- paste-ready run-summary generation; and
- comparison computation and display.

`KeyboardInterrupt`, `SystemExit`, and other `BaseException` subclasses still
propagate. Every caught exception is logged with traceback and recorded as an
unsuccessful `ReportArtifactOutcome`. This narrow broad catch receives a
documented Ruff suppression because catching ordinary implementation defects
is the purpose of the boundary.

### 3. Current-run artifact truth

Merge `ReportArtifactSpec` into `ReportArtifact`. One ordered artifact plan will
hold the key, public key, path, log label, dashboard label, dashboard purpose,
and optional generation job.

The successful `ReportArtifactOutcome` keys from the current run will control:

- links written to `index.md`;
- rows shown in the terminal artifact dashboard;
- links included in the run issue summary; and
- whether an HTML report may be opened while `--open-report` still exists.

An old file that still exists after a failed renderer is not a current-run
artifact and must not be linked or opened. The file may remain on disk as
recoverable prior evidence; the current-run surfaces simply omit it and log the
failed outcome.

### 4. Constraint aggregation by declared bounds

Aggregate title and keyword counts by their own `(minimum, maximum)` bounds.
The common one-range case retains its compact existing sentence. A malformed or
mixed retained source produces one factual bullet per range rather than
applying the final model's range to every observation.

### 5. One schema-3 machine artifact

`results.jsonl` becomes the sole current-run machine contract at schema version
`3.0`. Its metadata row absorbs the non-duplicated run-level fields currently
held by `run.json`:

- prompt digest;
- total runtime and outcome counts;
- artifact manifest;
- check_models producer identity;
- publication-safe image identity;
- shared generation settings;
- `trust_remote_code`;
- comparison result; and
- cache-discovery facts.

Library versions, component provenance, evaluation mode, prompt,
metadata-exposure state, execution mode, system facts, and runtime fingerprint
already belong in the metadata row. Model provenance and prompt-burden facts
remain on each result row and are not repeated in metadata maps.

Introduce one immutable in-memory retained-run value containing a validated
metadata record and ordered result records. Serialization, run-summary loading,
report-only regeneration, and comparison loading consume that value. Remove
`RunJsonReportRecord`, its subordinate aggregation-only types, the run JSON
serializer and loader, the `run_json` output path, and every cross-file join.

Schema 2 baselines are not kept behind a permanent compatibility adapter. The
first run after the schema-3 release reports that the retained baseline is not
comparable; the next run compares normally. Existing tracked schema-2 outputs
are replaced by the next real sweep, not rewritten by validation tests.

### 6. One output root

Replace the seven individual output-path options with:

```text
--output-dir PATH
```

`ReportOutputPaths.from_root(PATH)` derives the canonical layout:

```text
PATH/index.md
PATH/results.jsonl
PATH/check_models.log
PATH/environment.log
PATH/reports/results.html
PATH/reports/model_gallery.md
PATH/reports/diagnostics.md
PATH/issues/
```

The default remains `src/output`. This removes the surprising coupling in
which the index and issue directory were inferred from the `run.json` path and
substantially shortens every CLI/e2e test invocation.

### 7. Remove low-value CLI compatibility

The schema-3 minor release also removes:

- deprecated `stress` and `quality` evaluation aliases;
- `--open-report` and the `webbrowser` dependency; and
- `--detailed-metrics`.

Verbose mode always renders the existing detailed metrics block. Non-verbose
mode remains compact. `auto`, `triage`, `blind`, and `assisted` are the only
accepted evaluation inputs. No hidden aliases or ignored flags remain.

### 8. Subtractive test policy

Keep tests that protect current behaviour, exact evidence, escaping, security,
safe paths, malformed retained input, model discovery, native isolation,
thinking delimiters, generation-API parity, comparison integrity, and
cross-artifact consistency.

Delete tests whose only purpose is to blacklist identifiers, phrases, or APIs
removed by the earlier semantic-scoring refactor. Consolidate overlapping HTML
tests that independently assert the absence of retired grades, winners, owner
confidence, or semantic rankings into one canonical current-vocabulary test.
Delete tests for removed schemas, flags, and `quality_issues` rather than
preserving compatibility scaffolding to satisfy them.

Retain the nanobind/stub-generation policy test: it protects a comparatively
recent, costly CI failure mode and is not merely semantic-report archaeology.

## Release and Sequencing

The work has two independently shippable phases:

1. **Robustness phase:** canonical observation projection, broad artifact
   isolation, outcome-driven navigation, constraint aggregation, and associated
   local deletion. This phase does not change the artifact schema or CLI.
2. **Subtractive contract phase:** schema 3, `--output-dir`, compatibility-flag
   removal, test pruning, and documentation. This phase releases as version
   `0.16.0`.

Each phase must pass focused tests and the full quality gate before the next
phase begins. The schema phase must additionally render representative reports
from fixtures into a temporary directory and verify all Markdown links and
machine records without invoking a real model.

## Non-goals

- Splitting `check_models.py` into packages or production modules.
- Replacing the typed report-block AST or Rich terminal rendering.
- Removing HTML, diagnostics, the Markdown gallery, run summary, issue drafts,
  operational logs, or history.
- Reducing model/runtime telemetry, prompt diagnostics, burden facts, cache
  discovery, or upstream reproduction evidence.
- Adding semantic scoring or model-specific assessment exceptions.
- Re-running the real model matrix merely to validate code formatting or unit
  behaviour.
