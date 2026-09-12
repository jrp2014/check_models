# Implementation Guide

A practical reference for changing `check_models`: where to work, which
contracts to preserve, and how to verify a change. The project serves upstream
maintainers diagnosing integration failures and people choosing models for
accurate image metadata. Mechanical checks support those decisions; they do
not establish factual accuracy.

Read the [canonical project instructions](../.github/copilot-instructions.md)
first. This guide explains implementation boundaries, not a second set of
setup instructions or CLI defaults.

## Find what you need

| Task | Start here |
| --- | --- |
| Set up the environment, contribute or release | [Contributor guide](CONTRIBUTING.md) |
| Understand flags, defaults and upstream coverage | [CLI reference](../src/README.md) |
| Find the right function or test | [Architecture and navigation](#architecture-and-navigation) |
| Change generation or an assessment | [Generation and assessment](#generation-and-assessment) |
| Change a report or reproduction | [Reports and evidence](#reports-and-evidence) |
| Refactor or add a helper | [Philosophy](#philosophy) and [Code standards](#code-standards) |
| Change dependencies | [Dependency management strategy](#dependency-management-strategy) |
| Validate and hand off a change | [Validation and handoff](#validation-and-handoff) |

## Philosophy

Prefer correctness, diagnostic evidence and clarity over performance or the
smallest possible line count. `src/check_models.py` is intentionally one file;
do not split it into modules.

For simplification, work in this order:

1. Remove dead code, checking references in both production code and tests.
2. Consolidate implementations that share the same rules and invariants.
3. Extract a helper only when it removes duplication, names a distinct step,
   or isolates side effects for testing.

A cohesive, well-commented function is often clearer than many tiny helpers.
Do not move code solely to satisfy a statement-count limit, but do not suppress
a valid warning merely because refactoring is inconvenient. Explain boundaries
and non-obvious decisions; omit comments that just narrate the next line.

Use the standard library and existing dependencies before writing another
parser, formatter or validation framework. Keep human display strings separate
from raw machine facts. Measure optimisations without changing what a metric
or observation means.

## Architecture and navigation

Use the `SECTION:` banners and function names rather than copied line numbers.
The complete section map lives in the canonical instructions.

```bash
rg -n 'SECTION:' src/check_models.py
rg -n 'def process_image_with_model|def _run_model_generation|def _generate_with_repetition_guard' src/check_models.py
```

| Area | Main entry points | Existing tests to extend |
| --- | --- | --- |
| CLI and prompt policy | `_build_cli_parser`, `_apply_eval_mode_defaults`, `prepare_prompt` | `test_parameter_validation.py`, `test_cli_help_output.py` |
| Cache discovery and eligibility | `get_cached_model_ids`, `_classify_image_capability` | `test_model_discovery.py` |
| Per-model execution | `_run_one_model`, `process_image_with_model`, `_run_model_generation` | `test_process_image_mock.py` |
| Streaming integration | `_generate_with_repetition_guard` | `test_process_image_mock.py` |
| Output analysis and assessment | `analyze_generation_text`, `_populate_result_quality_analysis`, `_assess_result` | `test_quality_analysis.py` |
| Reports and finalisation | `ReportRenderContext`, `_generate_reports_and_log_outputs`, `finalize_execution` | `test_report_generation.py`, `test_jsonl_output.py` |
| Display and escaping | `format_field_value`, shared report renderers | `test_html_formatting.py`, `test_markdown_formatting.py` |
| Cross-run comparison | `compare_run_results`, `_history_tps_bands`, `_comparison_view` | `test_report_generation.py`, `test_jsonl_output.py` |
| Packaging and tooling | `src/pyproject.toml`, `src/tools/` | `test_dependency_sync.py`, `test_validate_env.py` |

Test filenames above are relative to `src/tests/`. Shared image fixtures live
in `conftest.py`; runtime thresholds live in
[`quality_config.yaml`](../src/check_models_data/quality_config.yaml), with typed
defaults in `QualityThresholds` and `FormattingThresholds`.

### Generation and assessment

Preserve these boundaries when threading a flag or refactoring a pipeline:

- **Dispatch:** every attempt, including differential triage, goes through
  `_run_one_model`. `--isolate` adds a child interpreter, not a separate
  assessment policy. Keep `_isolated_worker_spec` and `_isolated_params_from_spec`
  matched and test their round trip; mocking the subprocess alone does not
  exercise the child parser.
- **Generation:** `_generate_with_repetition_guard` wraps upstream
  `stream_generate`. Preserve EOS registration/reset, draft-chunk exclusion,
  final-chunk metrics, verbose echo, `text_already_printed`, and the optional
  processor `clean_output` hook. Test changes at this seam. A repetition abort
  is not an ordinary token-cap stop and must not enter comparable-throughput
  samples.
- **Failures:** retain the exception chain, deepest evidenced phase, model
  revision and available partial output. Before the first token is not
  necessarily prefill: input preparation and other upstream work can fail
  there. An OOM alone demonstrates a capacity failure, not a library defect.
- **Lane:** resolve `auto` to `triage`, `blind` or `assisted` before constructing
  the prompt. Record the resolved lane and whether metadata hints were exposed;
  a custom prompt does not automatically receive those hints.
- **Assessment profile:** select `general` or `metadata` by prompt origin or an
  explicit flag, never by interpreting prompt prose. General checks are
  task-independent; metadata adds labelled Title/Description/Keywords fields
  and duplicate keywords. Custom prompts and differential triage default to
  general. Preserve the profile in worker payloads and retained results.
- **Evidence:** a short answer, copied hint or length count is not automatically
  a fault. Accept conventional Markdown around field labels. Keep old
  observation codes readable without restoring retired detectors.
- **Thinking:** a closed thinking block followed by a substantive answer is
  neutral evidence. Recognise configured delimiters and prompt-seeded openers;
  distinguish incomplete or thinking-only output from a usable final answer.
  Preserve raw markers, and keep unexpected channel/control-token leakage
  distinct from legitimate thinking wrappers.

### Timing and comparison

Synchronise MLX evaluation before ending measured work. Preserve each metric's
source and unit: allocator calls return bytes, while upstream generation
metrics commonly report decimal GB. Record post-cleanup active/cache memory on
failures as well as successes; one non-zero sample does not prove a leak.

The reported prefill/first-token estimate is derived from prompt tokens and
prompt throughput; it does not include all input preparation. Do not relabel it
as measured end-to-end time to first token.

Keep comparison withholding rules central. Different images, prompts, settings
or lanes must not become apparent model regressions; absent facts are unknown,
not evidence of equivalence. Throughput also depends on execution mode and
hardware. History bands must exclude the judged run and use matching workload,
model-revision and effective-setting evidence. `_comparison_view` supplies
shared human-facing cells; JSON retains structured numeric facts.

For native reproductions, cache questions, performance evidence or upstream
fixes, use the relevant adapted [project skill](../.agents/skills/). Consult the
CLI reference's upstream coverage matrix before expanding scope; server cache
reuse and other unexercised workflows are not evidence about this direct runner.

## Reports and evidence

The default root is `src/output/`; `--output-dir` relocates the whole layout.
Keep the artifacts complementary rather than making each repeat everything.

| Artifact, relative to the output root | Purpose |
| --- | --- |
| `issues/run_summary.md` | Primary skim surface: scope, comparison and model outcomes |
| `reports/model_gallery.md` | Model selection, readable answers and exact complete output |
| `reports/diagnostics.md` | Highlighted maintainer evidence with shared reproduction context |
| `reports/results.html` | Self-contained gallery and diagnostics for local viewing |
| `issues/issue_*.md` | Individual actionable crash drafts |
| `results.jsonl` | Sole current-run machine contract: metadata header and per-model records |
| `index.md` | Navigation to artifacts successfully generated this run |
| `check_models.log`, `environment.log` | Operational timeline and environment evidence |
| `results.history.jsonl` | Append-only local cross-run history |

All current-run artifacts, including bounded preview assets, are retained in
Git. **History is the exception:** it is local-only and gitignored. Tests must
not rewrite any of these retained outputs.

### Assembly and presentation

- Compute assessments and aggregates once in the shared context. Use typed
  report blocks to render the same diagnostic hierarchy to Markdown and HTML;
  do not recreate classification rules inside renderers.
- Emit run-wide prompt, environment and reproduction context once per aggregate
  report. Show exact completed output once in diagnostics; the gallery owns
  the readable/raw pair. Omit optional empty fact rows.
- Use `format_field_value(field_name, value)` for metric formatting and existing
  Rich/report helpers for layout. Remove redundant columns before adding width
  calculations. Keep units and precision consistent across surfaces.
- Preserve raw values and output separately from escaped display copies. Use
  the shared escapers at the rendering boundary, including table pipes and
  newlines. Model-authored HTML must remain inert; do not expand the markup
  allowlist or insert raw model text into HTML.
- Preserve exact evidence, including tabs and trailing spaces. Generate
  Markdown in the style enforced by [`.markdownlint.jsonc`](../.markdownlint.jsonc),
  rather than repairing reports after generation. Any unavoidable evidence
  exception must be narrowly scoped.
- Use the central logger and `LogStyles` helpers. Keep `SUMMARY` and `REPRO`
  records parseable and warnings attributable to a model. Captured file-log
  output must not cause a second terminal echo.
- Publish only successful `ReportArtifactOutcome` entries in navigation and
  manifests, never stale files discovered on disk. Preserve atomic final
  reconciliation of `results.jsonl`; there is no separate `run.json`.

### Reproduction inputs

An issue reader needs the exact prompt and obtainable inputs, not a path that
exists only on the author's machine. For a public original, retain its URL and
SHA-256 and produce a download/verify/native-run command. For a local-only
original, report format, dimensions, byte size and digest.

A gallery preview is a bounded re-encoding, **not the original inference
input**. Any command using it must explicitly call it a stand-in and verify its
own digest. Do not copy full-resolution photographs into the report tree or
claim that reproducing on a preview establishes the same full-resolution bug.
Model-load failures can use any image because they precede image decoding.

## Code standards

### Types, missing values and paths

- Use `from __future__ import annotations`, fully typed parameters and return
  values, modern `list[str]` / `T | None` syntax, and `Final` constants.
- Prefer narrowing with runtime guards over casts. Use `Protocol` for external
  interfaces and `TypedDict` for record shapes when they clarify the boundary.
  Put type-only imports under `TYPE_CHECKING` where runtime inspection permits.
- Use `None` for unavailable facts, not zero or strings such as `"N/A"`.
  Presentation helpers decide how absence appears. Preserve that distinction
  in JSON and comparisons.
- Use `Path` internally; accept the existing `PathLike` alias where needed and
  convert to strings at library boundaries. Use the existing safe-I/O helpers
  for writes with path/symlink safety requirements.
- Keep adjustable thresholds in the YAML/typed configuration, fixed domain
  values in named constants, and obvious local offsets inline.
- Use upstream-shipped types rather than generating local stubs. Third-party
  implementation exclusions must not weaken checking of this project's call
  sites. Run `make ty` for an explicitly resolved project interpreter.

### Errors and external processes

Catch specific exceptions at the boundary that can handle them. Preserve
context with `raise ... from error`; use `logger.exception` when recording a
traceback. Broad catches belong only at deliberate containment boundaries,
with an explanation. Do not silently turn programmer errors into missing data.

Optional dependency imports may fail softly and populate `MISSING_DEPENDENCIES`;
required runtime dependencies must fail clearly before inference. Do not alter
Transformers backend-selection environment variables to hide import problems.

Use `exit_with_cli_error` for normal CLI errors and `raise SystemExit(code)` for
termination. Keep subprocess commands as argument lists, use bounded timeouts,
and capture enough exit/signal/stderr evidence to diagnose failures.

### Image metadata

Preserve independent EXIF passes: baseline IFD0 tags, Exif SubIFD and GPS IFD.
Corrupt optional metadata should not discard valid fields from another pass.
Reuse `_convert_gps_coordinate` for full DMS, degrees/minutes or degrees-only
coordinates; decode byte-valued cardinal references defensively. Display
unsigned magnitudes with N/S/E/W, avoiding double application of a sign.
Use timezone-aware datetimes and retain capture context separately from model
instructions and visible-image claims.

## Dependency management strategy

Use **conda + pip**, with the `mlx-vlm` environment active. Setup, updater
options and recovery procedures belong in the contributor guide and CLI
reference, not in another install recipe here.

- [`src/pyproject.toml`](../src/pyproject.toml) owns package metadata, dependency
  ranges, extras, build configuration and checker settings. Keep the version
  literal and dependencies in the `[project]` array, not a separate table.
- [`dependency_policy.py`](../src/check_models_data/dependency_policy.py) holds
  runtime compatibility rules and upstream floors used by preflight. Update
  affected policy and packaging declarations together.
- Declare direct third-party dependencies explicitly in the appropriate
  runtime or optional group; do not rely on an incidental transitive install.
  Standard-library imports do not require packaging changes.
- Prefer lower bounds; document specific evidence for a cap, exclusion or pin.
  Do not duplicate current version numbers or package lists in this guide.
- Run `make deps-sync` after changing dependencies. Commit the generated
  `src/README.md` blocks with the declarations; never hand-edit those blocks.
- Local editable upstream checkouts and CI's released packages can differ.
  Tests inspecting upstream interfaces must support both environments and
  report their provenance. Use the existing updater to preserve local builds;
  do not add a parallel dependency or lockfile workflow.

## Validation and handoff

Run commands from the repository root unless stated otherwise, after:

```bash
conda activate mlx-vlm
```

For a code change, start with tests for the affected boundary, then follow the
canonical gate order:

```bash
pytest src/tests/test_quality_analysis.py -q  # choose the relevant existing file
make format
make -C src lint-fix                        # when safe Ruff fixes are available
make lint
bash src/tools/run_commit_hygiene.sh
make quality
```

`make quality` includes the full pytest suite and required Markdown linting;
do not run `make test` again just to repeat it. Static checks, including Skylos,
run before pytest. Use `make help` for the current target list and the
contributor guide for tooling installation.

### Tests and artifact hygiene

- Add regression cases to existing `src/tests/test_*.py` files and reuse
  `conftest.py` fixtures. Test behaviour, failure paths and serialization
  boundaries, not just the presence of an implementation string.
- Render representative reports into `tmp_path` and check escaping, links,
  assessment consistency and Markdown lint before a costly real-model run.
  Tests must not rewrite tracked `src/output/` assets or leave generated files
  under `src/`; ordinary gitignored tool caches may stay there.
- Keep wheel builds and temporary checker configs outside the source tree.
  Packaging output is not a harmless cache: it can change import metadata or
  give static checkers duplicate sources to scan.
- Never use a real-model sweep as a substitute for deterministic tests. Fix
  and validate a bad baseline before comparing another run against it.

### Lints and review

Fix code before adding suppressions. An unavoidable upstream/tool false
positive needs the exact rule and a nearby explanation. Do not introduce bare
`noqa`, bare `type: ignore`, or blanket exclusions to pass project-owned code.
Function size alone is not a false positive.

Ruff unsafe fixes are optional, only when faster than correcting the code
directly. Apply safe fixes first; preview unsafe changes, critically inspect
every semantic change, and run targeted tests. They are not a routine extra
step or permission to accept a transformation without review.

Before handing off:

- Check the diff for unintended semantic changes, duplicate helper families
  and lost evidence. For report redesigns, inspect production line growth and
  explain genuinely new capability in the changelog.
- Update only the documentation affected: CLI behaviour in `src/README.md`,
  setup/workflow in `CONTRIBUTING.md`, conventions here. Record
  maintainer-relevant changes under `[Unreleased]` in `CHANGELOG.md`.
- State what was validated and any environment limitations. Do not claim the
  full gate passed when only focused checks ran.
- Record the tested commit. A clean fast-forward to that exact commit need not
  repeat the gate; changed code, a merge commit or conflict resolution does.
