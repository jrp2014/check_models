# Changelog

Notable changes to this project will be documented in this file.

## [Unreleased]

### Fixed

- Fix generated GitHub issue and report markdown to pass markdownlint (MD022,
  MD031, MD032, MD036): add blank lines around headings, fenced code blocks,
  and lists; replace bold emphasis with proper `###` subheadings in action
  snapshot.
- Add markdownlint to `--fast` quality checks so the pre-push hook catches
  markdown lint errors before they reach CI.
- Fix harness issue title generator to flatten multiline model output and strip
  trailing punctuation, preventing MD022/MD026 in generated issue templates.
- Escape `*` in repeated-token review text to prevent MD037 (spaces inside
  emphasis markers) when model output contains `*/` sequences.
- Pre-escape HTML tags in quality warning issue text to prevent MD045 (images
  without alt text) when model output contains `<img>` tags.
- Add MD045 to blockquote markdownlint-disable comments since model output can
  contain HTML img tags split across wrapped lines.

## [0.4.0] - 2026-04-19

### Added

- Added `unknown_runtime_anomaly` verdict for near-empty outputs that pass all
  quality checks but score grade F, signalling unexplained failures that need
  manual triage.
- Added `needs_triage` user bucket for classifier uncertainty, mapped from
  `unknown_runtime_anomaly` verdict. Review reports now render four buckets:
  recommended, caveat, needs_triage, avoid.
- Added `anomaly_min_output_chars` threshold to `quality_config.yaml` (default
  20) controlling the near-empty output threshold for anomaly promotion.
- Added `unknown_runtime_anomaly` to `QUALITY_BREAKING_LABELS` so anomalous
  outputs are excluded from recommendation candidate rows.
- Added property-style invariant tests (`TestClassificationInvariants`) covering
  hint relationship, truncation severity, bucket monotonicity, clean evidence,
  unknown anomaly triage, and history forward-compat invariants (93 new test
  cases).
- Added `TestIssueDirectoryInvariants` verifying issue directory reflects exactly
  the current run output.
- Added `--eval-mode` CLI flag with three evaluation lanes: `stress` (default,
  current behavior), `triage` (short prompt, 200 token cap, pass/fail only),
  and `quality` (cataloguing prompt with generous 1000 token cap).
- Added `--prune-repro-days N` CLI flag to automatically clean up repro bundles
  older than N days (default: 90). Set to 0 to disable.
- Added `Critical` severity tier in diagnostics for error clusters affecting
  ≥5 models, above the existing `High` threshold.
- Added `nonvisual_location_terms` config list in `quality_config.yaml` for
  maintainer-tunable nonvisual metadata terms.
- Added `penalized_echo_ratio` metric to `compute_information_gain()` that
  excludes visual hint term reuse from echo penalties.
- Added `add-or-fix-type-checking` agentic skill
  (`.agents/skills/add-or-fix-type-checking/SKILL.md`) providing a structured
  workflow for diagnosing and fixing typing errors from mypy, ty, and pyrefly.
  Adapted from the Hugging Face transformers skill to this project's three-checker
  pipeline, stub management, and coding conventions.
- Added automatic generation of standalone GitHub issue reports for clustered
  runtime failures and harness issues, placing ready-to-file markdown documents
  in `output/issues/`.
- Added `JsonlSignatureComponents` TypedDict exposing structured error signature
  fields (`error_code`, `normalized_message`, `traceback_signature`) in JSONL
  result records for failed models, enabling precise cross-run clustering without
  re-parsing raw messages.
- Bumped JSONL `format_version` from `"1.4"` to `"2.0"` to reflect the addition
  of `runtime_fingerprint` (metadata) and `signature_components` (results).
- Added round-trip schema tests (`TestSchemaVersioning`) verifying JSONL metadata
  and result records survive JSON serialization with correct format versions.
- Added `collect_runtime_fingerprint()` emitting per-probe `RuntimeProbeResult`
  records (metal_gpu, mlx_framework, mlx_vlm, gpu_memory) into JSONL metadata
  and history records for runtime environment attestation.
- Added `TestRuntimeFingerprint` canary tests (6 cases) verifying fingerprint
  probes, JSONL metadata integration, and history record integration.
- Added `--rerun-triage` CLI flag for automatic differential reruns of
  triage-worthy models (runtime failures and unknown anomalies) with a simple
  prompt. First-pass results are never overwritten (G3). Rerun evidence is
  emitted as `rerun_summary` in JSONL result records.
- Added `RerunEvidence` dataclass and `_select_rerun_candidates()` selection
  logic for identifying models that merit a secondary evidence pass.

### Changed

- Fixed `_classify_hint_relationship` regression: metadata-only trusted hints
  (e.g. GPS, timestamps) no longer mis-trigger `ignores_trusted_hints` when
  there are no visual terms to match against.
- Fixed `_classify_review_verdict` omitting `abrupt_tail` from degradation
  reasons, which caused clearly cut-off outputs to be classified as benign
  `token_cap` instead of `cutoff_degraded`.
- Fixed `_generate_github_issue_reports` not cleaning stale `issue_*.md` files
  before writing new ones, leaving outdated reports from larger previous runs.
- Changed mypy `follow_imports` for mlx_lm/mlx_vlm from `"silent"` to `"normal"`
  so mypy actively type-checks call sites against auto-generated stub signatures.
  transformers/tokenizers remain on `"silent"` to avoid noise from generated stubs.
- `_prune_repro_bundles` now handles `.json` bundle files (not just directories)
  and supports run-count retention via `max_runs` parameter. Empty directories
  are cleaned up automatically.
- Harness issue titles in generated GitHub issue reports now sanitize BPE byte
  artifacts (Ġ, Ċ) into readable text via `_sanitize_bpe_display`.
- TSV report cells are hard-capped at `MAX_TSV_CELL_CHARS` (200) to prevent
  oversized rows from CJK or verbose model output.
- Empty "recommended" user bucket in review.md now includes an explanation
  rather than bare "None.".
- Action Snapshot in results.md restructured from flat bullet list into three
  sub-groups: Failures & Triage, Quality & Metadata, Runtime.
- `finalize_execution` now calls `_clean_stale_toplevel_reports` to remove
  old top-level report files superseded by `output/reports/` copies.
- Fixed issue template writer to emit real newlines instead of literal `\n`
  strings, resolving MD047 lint failures on generated issue files.
- Fixed emphasis style in gallery cross-reference line (asterisks → underscores)
  to match markdownlint MD049 config.
- Updated `src/README.md` CLI reference: added 12 missing flags (`--resize-shape`,
  `--eos-tokens`, `--skip-special-tokens`, `--processor-kwargs`, `--enable-thinking`,
  `--thinking-budget`, `--thinking-start-token`, `--thinking-end-token`,
  `--eval-mode`, `--min-p`, `--top-k`, `--prune-repro-days`), fixed output paths
  from flat `output/` to `output/reports/`, documented `issues/` and
  `repro_bundles/` subdirectories, updated Project Structure tree.
- Updated `src/output/README.md` with directory structure diagram, `reports/`
  hierarchy, and conditional `issues/` and `repro_bundles/` subdirectories.
- Updated `.github/copilot-instructions.md`: refreshed section map line numbers
  (~20,100 lines), key files table sizes, output path descriptions, and all
  common edit recipe line references.

- Inlined `_process_ifd0`, `_coerce_exif_tag_id`, and `_process_gps_ifd` into
  `get_exif_data` — three-pass Pillow EXIF extraction is now direct code in the
  single function, removing ~40 lines of indirection.
- Replaced `_append_markdown_table` and custom Markdown table formatting with
  `tabulate(tablefmt="github")` across all diagnostics and review sections.
- Simplified `_sha256_file` to use `hashlib.file_digest()` (Python 3.11+),
  removing the custom chunked-read loop.
- Inlined `_escape_markdown_in_text`, `_escape_markdown_diagnostics`, and
  `_escape_markdown_gallery_warning` — thin escape-delegate wrappers replaced
  by direct `MARKDOWN_ESCAPER.escape()` / `DIAGNOSTICS_ESCAPER.escape()` calls
  at each call site.
- Split `cutoff` verdict into `token_cap` (benign cap hit, structurally sound
  output) and `cutoff_degraded` (cap hit with quality degradation). Token-capped
  A/B-grade models now reach `recommended` or `caveat` instead of `avoid`.
- Added `runtime_failure` verdict for models that crash (exception, OOM, timeout),
  distinct from `harness` verdict which is reserved for successful runs with
  template/encoding leaks.
- Improved failure clustering with message-only merge pass that collapses
  variants differing only in stack traces (e.g. broadcast_shapes dimensions).
- Reconciled utility grade with user bucket: A/B-grade clean or token-capped
  models can now reach `recommended`; C-or-below capped models go to `caveat`.
- Excluded nonvisual metadata terms from hint overlap ratio denominator so
  models are not penalized for omitting non-verifiable location metadata.
- Removed Action Snapshot duplication from review.md (now cross-references
  results.md).
- Expanded `NONVISUAL_CONTEXT_TERMS` with generic location descriptors (town,
  village, parish, county, borough, district, municipality, province, region).
- Hardened Markdown gallery blockquote rendering so label-only model-output
  lines such as `Description:` and stray lone `-` markers are neutralized
  before markdownlint can misread them as headings.
- Hardened Markdown artifact normalization so trailing BOM and zero-width
  format characters are stripped before write-out, preventing hidden
  `MD009/no-trailing-spaces` failures in generated gallery output.
- Updated workspace file associations so generated `PKG-INFO` package-metadata
  files open as properties instead of Markdown, avoiding editor-only
  markdownlint noise on `.egg-info` metadata.


## [0.3.3] - 2026-04-12

### Changed

- Reduced avoidable MLX teardown overhead after successful model runs by
  keeping the post-decode synchronize needed for accurate timing and memory
  sampling, but no longer forcing a second parameter-evaluation step before
  cleanup and skipping the cleanup-side synchronize when the run has already
  crossed the success-path barrier.
- Updated the Pyrefly quality gate so `make quality` now runs in project mode,
  prints warning-level diagnostics during the check, ships
  `types-defusedxml` in the dev toolchain, and declares `defusedxml` as an
  explicit runtime dependency instead of relying on Pillow's transitive graph.
- Aligned the fast pre-push quality script with the full quality gate so its
  Pyrefly step now also runs in project mode instead of only checking
  `check_models.py`.
- Consolidated the duplicated fast and full quality-script logic so
  `src/tools/check_quality_simple.sh` now delegates to a parameterized
  `src/tools/run_quality_checks.sh --fast` path, reducing drift between the
  pre-push hook and the full local/CI quality gate.
- Made image-metadata evaluation more decision-ready by adding separate
  description and keyword quality scorecards, surfacing those scores in the
  cataloging logs and recommendation summaries, and highlighting the best
  models for end-to-end cataloging, description quality, and keywording
  instead of relying on a single blended utility rank alone.
- Replaced the last direct `requests` and `tzlocal` runtime usage with stdlib
  `urllib` and local-time conversion via `datetime.astimezone()`, trimming the
  runtime dependency surface while preserving URL EXIF extraction and localized
  timestamp behavior.
- Hardened the `compute_task_compliance` metrics to strictly enforce the exact
  prompt constraints. Instead of awarding points merely for implicit or fuzzy section presence,
  the tool now parses output into explicit sections and awards passing compliance scores only
  when the model's word counts, sentence lengths, and keyword counts meet the respective
  bounds defined in `quality_config.yaml`.
- Hardened the MLX stack compatibility policy and diagnostics by centralizing
  shared version floors, raising the validated `mlx-vlm`, `mlx-lm`,
  `transformers`, and `huggingface-hub` minimums, switching runtime version
  comparisons to PEP 440 semantics, and adding always-on preflight checks for
  the exact `mlx_vlm` call surfaces and `GenerationResult` fields used by
  `src/check_models.py`.
- Tightened the local maintenance workflow so `src/tools/generate_stubs.py`
  now exposes a strict `--check` mode, fails loudly when required
  `mlx_vlm`/`transformers` stub patches drift, `src/tools/run_quality_checks.sh`
  treats stub-integrity failures as blocking, and `src/tools/update.sh` logs
  local editable MLX repo provenance before verifying checked-in stubs.
- Expanded the repo-level recommended VS Code extensions in
  `.vscode/extensions.json` to include YAML, GitHub Actions, and ShellCheck
  support, matching the checked-in workflow YAML, shell quality gates, and
  pre-push tooling used by this project.
- Removed the Jupyter VS Code extension from the repo-level recommended
  extension list in `.vscode/extensions.json`, keeping the workspace defaults
  focused on the Python CLI and quality-tooling workflow used in this project.
- Added an explicit `Generated Text:` label to the non-verbose per-model preview
  path in `src/check_models.py`, so emitted model output is clearly identified
  even when the run stays in compact summary mode.
- Replaced the hardcoded Pyrefly `conda-environment` setting with shared
  quality-script injection of `python-interpreter-path`, so local checks still
  target the resolved repo Python while GitHub Actions no longer fails the
  Pyrefly step on runners without `conda`.
- Clarified CLI help and README wording for `--folder`, `--image`, and `--prompt`
  so omitted flags describe their fallback behavior without implying the flags
  themselves accept empty values.
- Removed stale static-analysis downgrades in `src/pyproject.toml` by dropping
  unused Ruff tool-file `D`/`ANN` ignores, the unused Ty
  `unresolved-import=warn` downgrade, and a Pyrefly `ignore-missing-imports`
  list; Pyrefly now resolves imports against the `mlx-vlm` conda environment
  directly instead of masking system-interpreter drift.
- Made the packaged `quality_config.yaml` the single canonical default source,
  taught `load_quality_config()` to read the bundled resource by default, and
  added wheel-content plus anti-duplication regression tests so future
  packaging changes cannot silently drop or reintroduce split runtime-default
  copies.
- Reworked the automated review payload and `src/output/review.md` generation to
  carry concrete prompt/output evidence, show `review.md` as a user-first digest
  with review priorities and compact bucket tables, and replace the most
  repetitive owner-level next-action prose with evidence-aware guidance.
- Tightened several noisy quality heuristics in
  `src/check_models_data/quality_config.yaml` by
  requiring stronger evidence for repetition, context ignorance, context echo,
  genericity, and verbosity before flagging outputs, while keeping harness and
  contract failures visible in maintainer-facing diagnostics.
- Added Vulture to the managed `dev` dependency set, wired it into the full
  and fast quality scripts plus `make quality` / `make vulture`, and added a
  checked-in VS Code task/problem matcher so dead-code findings can surface as
  Problems warnings before commit-time checks. Expanded the checked-in Vulture
  scope to cover the Python utilities under `src/tools/` as well, while still
  excluding the archived tool stash. Tightened the task matcher to only catch
  real Vulture diagnostics with confidence suffixes, avoiding bogus Problems
  entries from unrelated `file:line:` output emitted by other quality steps,
  and limited Problems publishing to the dedicated `Make: vulture` task so the
  broader mixed-output quality task no longer leaves stale warnings behind.
  Aligned the `make vulture` and quality-script invocations with the checked-in
  `[tool.vulture]` config as well, so the broadened `src/tools/` scan actually
  runs instead of being overridden by an old explicit `check_models.py` path.
- Bumped the repo-local `markdownlint-cli2` npm tooling in `src/package.json`
  and `src/package-lock.json` to `0.22.0`, and changed `src/tools/update.sh`
  to refresh that tool from npm's `latest` tag on each update run so Markdown
  lint tooling stays current automatically. Added an npm override for
  `smol-toml@1.6.1` as well, so the current latest `markdownlint-cli2`
  dependency tree is patched against the published moderate-severity TOML
  parser DoS advisory without downgrading the linter.
- Made protocol method bodies in `src/check_models.py` explicit stub
  implementations (`...`) instead of docstring-only placeholders so Pylance and
  other control-flow checkers no longer misread those type-only call surfaces
  as returning `None`.
- Added `pydantic` to the managed `dev` dependency group in
  `src/pyproject.toml` and aligned the `src/tools/setup_conda_env.sh`
  fallback installer so repo update/bootstrap flows keep it current through the
  existing dependency-management path.
- Tightened external-library typing in `src/check_models.py` by giving the
  imported `mlx_vlm` callables explicit local protocol signatures, narrowing
  loaded processor/config annotations toward `transformers` types, extending
  generation-result protocols with upstream throughput fields, and replacing
  broad kwargs bags with typed helper shapes while keeping missing third-party
  stubs as soft warnings via best-effort local stub refreshes in
  `src/tools/run_quality_checks.sh`.
- Tightened several remaining metadata and EXIF typing surfaces in
  `src/check_models.py` by replacing broad `dict[str, Any]` helper returns
  with explicit IPTC/XMP typed shapes, narrowing Pillow tag lookups, and
  adding small protocol-based annotations around EXIF and GPS helper usage.
- Tightened additional low-risk helper typing in `src/check_models.py` by
  validating quality-config sections as string-keyed mappings and normalizing
  macOS `system_profiler` JSON into typed helper shapes before GPU/tooling
  extraction.
- Removed redundant config and hardware-probe logic in `src/check_models.py`
  by centralizing string-keyed mapping validation and reusing a cached
  `get_device_info()` path for both generic GPU info and Apple Silicon details.
- Removed a few dead or single-use wrapper helpers in `src/check_models.py`
  by deleting unused display helpers and inlining thin markdown/probe wrappers
  that only obscured one call site each.
- Further shortened `src/check_models.py` by inlining single-use prompt,
  runtime-formatting, chat-kwargs, and traceback-cleanup helpers in place
  where the surrounding call site was clearer than the extra wrapper.
- Fixed Ty invocation in `src/tools/run_quality_checks.sh` and
  `src/tools/check_quality_simple.sh` to pass the resolved repo Python
  explicitly, eliminating false `unresolved-import` warnings when Ty fell back
  to the wrong Conda site-packages instead of the `mlx-vlm` environment.
- Added a dedicated `make ty` / `src/tools/run_ty_check.sh` path plus
  explicit Ty environment diagnostics so local runs report the target conda
  env, active env, resolved Python, resolved Ty binary, and any fallback use
  instead of silently relying on ambiguous environment auto-detection.
- Removed several dead or redundant helper layers in `src/check_models.py`,
  including unused diagnostics/runtime helpers, now-unused protocol
  definitions, a duplicate quality-analysis accessor, and a few thin
  report/context wrappers, to shrink the monolith without changing report
  behavior.
- Compacted harness/review ownership routing in `src/check_models.py` by
  centralizing repeated owner maps and composite next-action rules, removing
  unused review-owner inputs, and collapsing one verbose quality-signal
  summary ladder into table-driven logic without loosening diagnostics or type
  checking.
- Redirected the E2E smoke test helper in `src/tests/test_e2e_smoke.py` to
  send standalone gallery and review artifacts to the test temp output
  directory as well, so `make quality` no longer rewrites tracked
  `src/output/model_gallery.md` and `src/output/review.md` during pytest runs.
- Hardened `src/tools/update.sh` so local MLX refreshes now fail fast when the
  active macOS toolchain does not expose `metal` and `metallib` via `xcrun`,
  and so editable-install origin checks cover `mlx` in addition to `mlx-lm`
  and `mlx-vlm`.
- Switched the local `mlx` editable install path in `src/tools/update.sh` to
  invoke `pip install -v -e .` so MLX builds emit full pip build logs during
  updates.
- Improved canonical review and diagnostics wording for `huggingface-hub`
  model-load failures so transient Hub disconnects are attributed to cache /
  network / Hub availability issues instead of falling through to a generic
  `model` owner diagnosis.
- Raised the declared `huggingface-hub` runtime floor to `>=1.8.0` in the
  project metadata, environment validator, and synced install docs so the repo
  matches the current active MLX environment and no longer advertises the old
  pre-1.0 Hub client baseline.

## [0.3.2] - 2026-03-27

### Changed

- Escaped inline emphasis markers in model-gallery quality warning bullets so
  arbitrary model output like `*/` sequences no longer trips markdownlint
  `MD049` in `src/output/model_gallery.md` after report generation.
- Stripped trailing non-breaking spaces from wrapped Markdown blockquotes and
  broadened trailing-whitespace normalization so generated gallery output no
  longer trips markdownlint `MD009` on model lines that end with `U+00A0`.
- Escaped square-bracket syntax in wrapped Markdown blockquote output so raw
  model text such as Python indexing expressions no longer trips markdownlint
  `MD052` in generated review artifacts under `src/output/`.
- Narrowed Markdown generator suppressions by wrapping more formatter-owned
  prose to the configured width, switching generator-emitted labels from `**`
  to repo-preferred underscore emphasis, and updating the manual failures table
  to spaced pipe separators so generated reports rely less on `MD013`, `MD049`,
  and `MD060` disables.
- Refreshed the MLX stack compatibility policy to track current upstream stable
  releases more closely: `mlx>=0.31.1`, `mlx-vlm>=0.4.1`, `mlx-lm>=0.31.1`,
  and `transformers>=5.4.0`, and aligned preflight diagnostics and fallback
  environment validation tables with current upstream minimum requirements.
- Dropped TensorFlow from packaged optional dependencies, since `check_models`
  does not import it directly, and added `timm` to the `torch` extra so
  FastVLM-style remote-code models install their required vision backbone.
- Updated `src/tools/update.sh` to keep Markdown lint tooling repo-local via
  `npm install --prefix src` instead of globally mutating npm packages, while
  preserving the default MLX build path with `MLX_METAL_JIT` unset/off unless
  a user explicitly opts in.
- Tightened the staged Markdown pre-commit hook so committed report artifacts
  under `src/output/` are no longer excluded from markdownlint fixing, which
  lets commit-time hygiene catch lint regressions in generated Markdown such as
  `review.md` before CI does.
- Made `check_models.log` the canonical full-fidelity run artifact by adding a
  fixed per-model review block with verdict, trusted-hint handling, contract
  and utility summaries, ownership hints, token accounting, next actions, and
  full generated or captured failure output even when console logging stays
  concise.
- Reworked trusted-hint quality analysis so only prompt title/description/
  keyword hints are treated as reusable draft content, while capture metadata,
  GPS, timestamps, source labels, and explicit location labels are stripped out
  of hint-usage scoring and can instead be flagged separately as nonvisual
  metadata borrowing.
- Added ordered automated review verdicts (`clean`, `harness`, `cutoff`,
  `context_budget`, `model_shortcoming`), surfaced them in `results.jsonl`
  review payloads and the new `output/review.md` digest, and wired the same
  review data into gallery/report output so users and maintainers see one
  consistent judgment across artifacts.
- Tightened MLX runtime metric capture so compact logging now reads the
  stored `cache_memory` field consistently, and generation results backfill
  peak memory from `mx.get_peak_memory()` when upstream results omit it.
- Added a narrow one-shot retry for known upstream `mlx-vlm` BPE streaming
  detokenizer UTF-8 decode failures, temporarily switching the retry attempt
  to lossy byte flushing so intermittent `UnicodeDecodeError` crashes can be
  worked around locally while the upstream bug is fixed.
- Added first-token latency capture to runtime diagnostics by deriving it from
  upstream `mlx_vlm.generate` prompt-throughput metrics, and preserved that
  signal through final result finalization so reports and compact logs can show
  real latency data instead of an always-empty field.
- Exposed upstream `mlx_vlm.generate` sampling controls for `min_p` and `top_k`
  through CLI validation, repro-command generation, and runtime parameter
  forwarding, and corrected the local `mlx_vlm` stub so `stream_generate()` is
  typed as yielding `GenerationResult` objects instead of plain strings.
- Hardened quality-analysis propagation in `src/check_models.py` so successful
  runs cache structured quality results at result-construction time, report
  generation backfills missing quality analysis for synthetic success rows, and
  failed runs can retain quality flags from captured stdout instead of dropping
  those signals on exception paths.
- Tightened `quality_config.yaml` loading so invalid threshold bounds or
  malformed detector regex patterns fail closed with a warning instead of
  silently weakening output-quality checks.
- Tightened diagnostics snapshot generation so Markdown diagnostics backfill
  missing success-side quality analysis before harness/stack classification,
  preventing clean or unflagged buckets from hiding reportable issues when the
  report is built from minimally populated `PerformanceResult` rows.
- Marked prompt-less cached quality analysis as incomplete and refreshed it
  when the original prompt becomes available again, so report rendering and
  diagnostics do not label those runs as fully clean while context-sensitive
  checks remain unavailable.
- Expanded long-context stack-signal heuristics so successful runs with extreme
  prompt lengths can also be surfaced for context-echo and degeneration
  symptoms, instead of only empty-output and low-ratio cases.
- Tightened diagnostics owner attribution so stack-signal anomalies can reuse
  matching preflight compatibility hints and point at narrower likely owners
  such as `transformers / mlx-vlm` instead of always collapsing into the broad
  `mlx-vlm / mlx` bucket.
- Tightened failed-run package attribution so chained tracebacks now prefer the
  deeper upstream owner, avoiding `mlx-vlm` wrapper attribution when the root
  failure actually comes from packages such as `transformers`.
- Tightened harness diagnostics owner attribution so prompt-template successes
  no longer collapse into generic runtime ownership, and maintainer summaries
  now distinguish model-config style harness issues from long-context runtime
  anomalies.
- Tightened harness maintainer triage so mixed harness-owner runs now render as
  separate action-summary and priority-table rows instead of collapsing config
  and runtime signals into one generic maintainer bucket.
- Tightened diagnostics maintainer triage so mixed stack-signal and preflight
  owner classes also render as separate action-summary and priority-table rows,
  instead of collapsing upstream compatibility and runtime issues into one
  merged owner bucket.
- Tightened the CLI maintainer summary so mixed harness, stack-signal, and
  preflight owner classes are logged as separate owner-specific lines with next
  actions, instead of collapsing them into one broad summary sentence.
- Tightened the detailed preflight diagnostics section so mixed owner classes
  render as separate owner buckets with owner-specific trackers and next
  actions, instead of one flat mixed warning list.
- Added an env-aware `make analyze-quality` workflow plus a standalone
  `tools.analyze_output_quality` CLI so quality and harness heuristics can be
  exercised directly against arbitrary text or saved outputs without running a
  model.
- Tightened `tools.analyze_output_quality` argument validation so conflicting
  `--prompt` and `--prompt-file` sources now fail fast instead of silently
  allowing ambiguous prompt context.
- Added a machine-readable `--json` mode to `tools.analyze_output_quality` so
  quality-analysis results can be consumed directly by scripts and triage
  tooling without scraping the human report output.
- Tightened clean-machine environment setup so the bootstrap script now prints
  repo-root-safe usage commands, probes common Miniforge/Mambaforge conda
  locations, and `tools.validate_env` no longer falsely rejects custom active
  conda env names unless strict matching is explicitly requested.
- Tightened report-generation typing in `src/check_models.py` by replacing broad
  `getattr`/cast access in gallery helpers with narrower protocol-based checks
  for throughput and cached quality-analysis fields.
- Optimized model generation prefill time by changing the default
  `--prefill-step-size` from `None` (which deferred to mlx-lm's conservative 512
  token default) to `4096`, significantly speeding up context evaluation on Mac
  GPUs for long prompts.
- Realigned contributor and user-facing documentation so the root README,
  `src/README.md`, and workflow docs consistently describe the standalone
  `model_gallery.md` artifact, the recommended conda bootstrap path, and the
  current split quality/runtime CI behavior.
- Made the repo-root `Makefile` delegate to the env-aware `src/Makefile` so
  common install/test/format/typecheck commands respect the intended Python
  environment instead of invoking bare `python`/`pip`/tool binaries.
- Expanded markdown quality coverage to include committed Markdown under
  `src/output/`, and updated bot guidance files so `MetricValue`, report
  artifact lists, and `make quality` expectations match the current codebase.
- Hardened local git-hook environment bootstrap so pre-commit and pre-push can
  locate and activate the intended conda env even when git launches hooks
  without `conda` on `PATH`, and YAML validation failures now print an explicit
  PyYAML/environment setup hint instead of a raw traceback.
- Made `tools/setup_conda_env.sh` install the repo's custom git hooks whenever
  development dependencies are selected, and updated contributor docs to treat
  that env-aware hook installer as the default local workflow.
- Extended `tools/setup_conda_env.sh` so dev setup also installs the repo-local
  `markdownlint-cli2` dependency when `npm` is available, and now warns
  explicitly when Node.js tooling is still missing for staged Markdown hooks.
- Hardened pre-push and full quality scripts so env-installed tools such as
  `ty` and `pyrefly` are resolved from the selected Python environment instead
  of relying on shell `PATH`, avoiding false "command not found" failures when
  git launches hooks from a reduced environment.
- Tightened clean-machine bootstrap so `tools/setup_conda_env.sh` now installs
  `cmake` from `conda-forge` explicitly and falls back to `pip` if needed,
  instead of assuming the user's existing conda channel configuration exposes a
  usable `cmake` package for the target platform.
- Narrowed the preflight `transformers` backend-guard warning so it now refers
  specifically to the TF/FLAX/JAX guard env vars used by `check_models`, rather
  than implying that all upstream `USE_*` backend toggles disappeared.
- Documented preflight compatibility warnings in `src/README.md` and tightened
  the shared HTML/Markdown action snapshot plus diagnostics report wording so
  those warnings are clearly framed as informational by default, with explicit
  guidance on when users should and should not escalate them.

## [0.3.1] - 2026-03-14

### Added

- Added a standalone GitHub-compatible Markdown gallery artifact for model-output
  review, with populated image metadata, the full prompt, and one easy-to-scan
  section per model output.

### Changed

- Made the human-facing report artifacts more actionable and consistent by
  reusing shared triage/rendering helpers for HTML, Markdown, and the standalone
  gallery, adding maintainer-oriented failure ownership summaries plus reviewer
  shortlists and per-model assessment cues.
- Reduced internal redundancy in `src/check_models.py` by merging HTML and Markdown
  cataloging summary rendering paths into a single `_format_cataloging_summary` function.
- Added lazily-evaluated `.successful` and `.failed` cached properties to `ResultSet`
  to optimize repeat list comprehensions during report generation.
- Cleaned up the quality-analysis detector subsystem in `src/check_models.py`
  by reusing the shared configured-pattern lookup path across more detectors,
  tightening non-trivial local annotations, and centralizing repeated
  context-stopword filtering used by prompt-context analysis.
- Expanded `src/tools/check_suppressions.py` into a stricter repo-wide
  suppression audit and wired it into the fast/full quality scripts so stale
  `noqa`, `type: ignore[...]`, and `shellcheck disable=` comments are flagged
  automatically.

- Narrowed legacy Transformers backend guard exports so `check_models` now sets
  only `TRANSFORMERS_NO_*` / `USE_*` variables still referenced by the
  installed `transformers` version, instead of exporting the full legacy set
  unconditionally.
- Tightened the default cataloguing prompt again to separate section labels from
  instruction text, reducing the chance that weaker models copy prompt
  instructions into the generated `Title:` field.
- Made captured timing data easier to see in default outputs by adding compact
  per-model runtime hints to `check_models.log` and a short aggregate timing
  snapshot to the Markdown/HTML/diagnostics report summaries.
- Clarified `--detailed-metrics` behavior in CLI/docs and now warn when it is
  provided without `--verbose`, since that combination otherwise falls back to
  compact output.

## [0.3.0] - 2026-03-07

- Hardened local hook and CI quality workflows:
  - split shared gates into commit hygiene, pre-push fast checks, full static
    quality, and separate runtime smoke probing;
  - removed tool-install side effects from quality enforcement scripts and made
    the full quality gate non-mutating;
  - aligned the checked-in `pre-commit` config, custom git-hook installer,
    workflow docs, and environment validation around the same commit/push
    behavior;
  - simplified ancillary Node tooling around `npm install`, moved dependency-sync
    CI to Ubuntu with path filters, and split GitHub Actions into
    `static-quality` and `runtime-smoke` jobs;
  - ensured the static-quality path creates a repo-root `typings/` directory so
    `ty` can consume the configured extra search path on fresh CI checkouts;
  - aligned active contributor/implementation documentation with the split
    commit/push hooks, static-vs-runtime CI separation, stub preflight, and
    current markdownlint enforcement behavior.
- Expanded verbose CLI detailed metrics output to include additional runtime
  phase timings (`input_validation`, `prompt_prep`, `cleanup`,
  `first_token_latency`) plus `stop_reason` when that metadata is available.

- Reduced redundant internal formatting and diagnostics code in `src/check_models.py`:
  - inlined a few single-use helpers in import-probe, diagnostics, and failure-capture paths;
  - centralized numeric-string coercion used by field formatting and numeric stats aggregation;
  - deduplicated the shared EXIF datetime parsing path while preserving existing helper behavior.

- Began expanding `src/check_models.py` generation parity with upstream `mlx_vlm.generate`:
  - added CLI/runtime support for `resize_shape`, `eos_tokens`, `skip_special_tokens`, and JSON `processor_kwargs` passthrough;
  - added opt-in thinking-mode support via `enable_thinking`, `thinking_budget`, `thinking_start_token`, and `thinking_end_token`;
  - normalizes these values centrally during CLI validation so model runs receive consistent typed kwargs.

- Refactored `diagnostics.md` output structure for improved issue triage utility:
  - merged 'Potential Stack Issues' into a sub-section of 'Harness/Integration
    Issues' to reduce redundancy;
  - replaced the generic attachment guidance for JSON repro bundles with explicit
    direct GitHub repository links to the `output/repro_bundles/` directory.
- Tightened default cataloging prompt generation in `src/check_models.py`:
  - now requires strict three-section output with explicit anti-CoT and
    anti-verbatim-copy rules;
  - keeps `Context:` metadata hints concise and tagged as high-confidence
    hints rather than instruction text.
- Revised the default cataloging prompt toward non-speculative, visibility-only
  output:
  - metadata hints are now framed as a draft record to verify against the
    image, not a source to elaborate from;
  - prompt wording now explicitly prefers omission over guessing and forbids
    inferred identity/location/event/brand/species/time-period details unless
    visually obvious;
  - title/keyword quality thresholds now align with the stricter prompt
    contract (`Title: 5-10 words`, `Keywords: 10-18 terms`).
- Strengthened output-quality heuristics for prompt-contract enforcement:
  - added explicit flags for missing sections, title length, description
    sentence count, keyword count, and keyword-duplication violations;
  - added reasoning/prompt-echo leakage detection and context-regurgitation
    (`context-echo`) detection;
  - improved context-ignorance matching with alias support (for example
    `UK` vs `United Kingdom`) and filtering of non-semantic prompt-label terms.
- Improved quality-issue serialization robustness: JSONL list conversion now
  preserves single issue items that contain commas inside parenthesized detail
  payloads (for example repetitive phrase previews).
-- Polished generated report outputs for faster maintainer triage without
  changing core runtime behavior:
  - added a compact `Action Summary` near the top of `diagnostics.md` with
    explicit owner labels and next actions;
  - expanded diagnostics reproducibility guidance to include both exact rerun
    commands and portable dependency/import probes that do not require local
    image assets (now centralized in a single portable triage section instead
    of repeated per failure cluster);
  - added a concise `Action Snapshot` near the top of `results.md` to separate
    framework/runtime failures from low-utility model watchlist signals.
- Updated local MLX build integration in `src/tools/update.sh` to support
  explicit `MLX_METAL_JIT` pass-through via `-DMLX_METAL_JIT=<ON|OFF>` when
  requested, while leaving MLX's default behavior untouched when unset; also
  refreshed docs that previously referenced the older
  `MLX_BUILD_METAL_KERNELS` mapping.
- Hardened local-build detection in `src/tools/update.sh` for `mlx-lm` and
  `mlx-vlm`:
  - verifies editable install origin paths against local repo paths after local
    rebuilds (instead of relying on version strings);
  - preserves editable installs when deciding whether to skip PyPI MLX
    ecosystem upgrades, even when package versions look like release versions.
- Expanded local stub generation defaults to include `transformers` alongside
  `mlx_lm`, `mlx_vlm`, and `tokenizers` (`tools/generate_stubs.py`,
  `tools/update.sh`, and README command docs).
- Tightened type-checker stub integration:
  - mypy now prefers following generated stubs for `mlx_lm`, `mlx_vlm`,
    `transformers`, and `tokenizers` (`follow_imports = "silent"` override);
  - quality checks now emit explicit warnings when expected stub packages are
    missing or contain invalid syntax.
- Reduced upstream stubgen noise during `transformers` stub generation:
  known non-actionable `auto_docstring` diagnostics are now suppressed and
  replaced with a concise suppression count, while actionable/non-zero-exit
  stubgen output is still surfaced.
- Improved failed-model summary lines in `check_models.log` to include decoded
  maintainer hints (`owner≈... | component=... | likely=...`) plus a compact
  normalized symptom excerpt, making canonical error codes actionable without
  consulting internal token mappings.
- Tightened local type-check tooling integration:
  - `ty` now searches repo-local generated stubs via
    `tool.ty.environment.extra-paths`, which resolves `mlx_vlm.*`
    submodules during `ty check`;
  - `src/tools/run_quality_checks.sh` now calls stub generation in a
    skip-if-fresh mode backed by a `typings/.stub_manifest.json` cache, so
    stubs are regenerated only when installed package versions change or the
    cache is missing.
- Compressed `src/check_models.py` with low-risk internal cleanup:
  - removed stale private dead code and one unused lint-suppression path;
  - merged duplicated HTML/Markdown issue-summary rendering around shared
    summary collectors and aggregate-stat rows;
  - standardized diagnostics section assembly around the existing Markdown
    section helpers to reduce repeated divider/heading boilerplate.
- Raised the project Transformers floor to `>=5.2.0` and aligned packaging,
  runtime checks, and docs/tests to that policy.
- Aligned preflight package-floor diagnostics in `src/check_models.py` with
  current upstream dependency declarations from `mlx-vlm` and `mlx-lm`
  repositories (instead of stricter ad-hoc floors), reducing false-positive
  compatibility warnings.
- Updated backend-guard behavior for Transformers integration: when
  `MLX_VLM_ALLOW_TF` is not set, `check_models.py` now applies both legacy
  `TRANSFORMERS_NO_*` and compatibility `USE_*` env guards, and diagnostics now
  explicitly report when newer Transformers versions ignore both families.
- Updated `src/tools/update.sh` so local MLX builds apply `MLX_METAL_JIT`
  through MLX's current CMake build flag
  (`CMAKE_ARGS=-DMLX_METAL_JIT=<ON|OFF>`), ensuring the selected
  kernel mode is honored during `pip install -e .`.
- Audited inline comments in `src/check_models.py` and removed stale
  refactor-history notes that no longer describe current behavior, while
  keeping explanatory comments for runtime/error-handling decisions.
- Updated maintainer map in `.github/copilot-instructions.md` to reflect the
  current monolith/test sizes and refreshed function line anchors.
- Normalized `src/README.md` command examples to use
  `python -m check_models`, matching the documented package entrypoint.
- Further compressed `src/check_models.py` (about 200 lines) by deduplicating
  EXIF date/time extraction paths, centralizing special-token leak pattern
  tables, and trimming verbose internal docstrings while preserving behavior.
- Diagnostics report generation now emits clearer issue-facing sections:
  `To reproduce` uses a single repro bullet, model output is shown inline when
  available, and technical traceback/captured logs are grouped in one
  collapsible `Detailed trace logs` section.
- Diagnostics report layout now surfaces `Priority Summary` near the top for
  faster triage, while moving the full `Environment` table near the bottom
  (just before reproducibility details).
- Terminal alignment now manages Unicode display width via `wcwidth` with
  safe fallback behavior, improving centered headers and metric-label padding
  when wide glyphs/emoji appear in output.
- Tightened model-generation typing in `src/check_models.py` by reducing
  ambiguous `Any` usage in `_load_model` / `_run_model_generation` where
  upstream function signatures allow safe narrowing.
- Terminal `Model Comparison (current run)` table now explicitly right-aligns
  numeric columns (TPS/timing/memory) and left-aligns text columns for easier
  visual scanning.
- Improved dependency-management tooling:
  - `src/tools/check_outdated.py` now uses JSON output with timeout/network-aware
    handling, and groups results into pyproject-managed vs unmanaged packages.
  - `src/tools/validate_env.py` now parses dependency specs robustly (including
    extras syntax) and validates installed versions against declared constraints.
- Stub generation now targets a broader default set (`mlx-lm`, `mlx-vlm`,
  `tokenizers`) and the quality gate runs a stub preflight before type checks.
- CI MLX core stubs are now written to repo-local `typings/` instead of
  mutating `site-packages`, improving reproducibility across runs.
- Refactored diagnostics/reporting internals for readability and lower
  duplication without intended behavioral changes:
  - removed single-use diagnostics wrappers and unused model cleanup state;
  - deduplicated diagnostics list + traceback normalization paths;
  - centralized diagnostics prose mappings;
  - simplified repro command assembly.
- Simplified diagnostics failure-cluster filing guidance so it only includes
  the repro command bullet; full traceback/captured-output diagnostics remain
  available in the existing collapsible sections.
- Aligned runtime-dependency preflight behavior between CLI and diagnostics:
  when core runtime packages are unavailable, the CLI now logs a structured
  environment-failure message and writes a minimal `diagnostics.md` focused on
  the missing dependencies and environment fingerprint instead of per-model
  failures.
- Centralized diagnostics configuration (history depth, snippet lengths,
  traceback tail lines, and cluster thresholds) into a single
  `DiagnosticsConfig` struct near the diagnostics helpers to make future tuning
  easier.
- Added concise one-line micro-summaries at the top of the Harness/Integration
  Issues and Long-Context/Stack Issues sections in `diagnostics.md` to improve
  scanability in large reports.
- Updated the `Models Not Flagged` diagnostics subsection for successful models
  with non-fatal quality issues:
  - renamed the warnings bucket to `Ran, but with quality warnings`;
  - added a per-model one-line warning summary derived from already captured
    quality-analysis signals.
- Added diagnostics completeness/runtime verification near the report footer:
  - explicit `Coverage & Runtime Metrics` section validates that each model run
    appears exactly once across detailed vs summary diagnostics buckets;
  - added aggregate runtime metrics (total model-runtime sum and average per
    model), with a clear fallback note when per-model timing fields are
    missing.
- Optimized hot paths in `src/check_models.py`:
  - Hugging Face cache scans are now reused via a per-run cache helper.
  - Quality analysis is reused when `PerformanceResult.quality_analysis`
    already exists, avoiding repeated `analyze_generation_text()` calls.
  - Diagnostics generation now computes failure clusters once and reuses them
    across sections.

### Fixed

- Fixed Markdown report generation so `results.md` avoids markdownlint
  violations from prompt/error rendering:
  - prompt output now uses a plain fenced code block (not blockquote-wrapped)
    to prevent `MD028/no-blanks-blockquote` when prompt text contains blank
    lines;
  - gallery error prose now escapes emphasis markers in non-URL segments so
    identifiers like `LanguageModel.__call__()` do not trigger
    `MD050/strong-style`.

- Replaced a brittle type-only dependency on internal Transformers module
  `transformers.tokenization_python.PythonBackend` with a stable `Any` cast at
  the mlx-vlm call boundary, avoiding reliance on non-public import paths.
- Fixed `src/Makefile` `ci` target to use available commands (ruff + mypy +
  dependency sync check) instead of referencing removed tooling.
- Added `--check` mode to `src/tools/update_readme_deps.py` so CI/developers can
  verify README dependency block sync without rewriting files.
- Fixed noisy accidental pasted output in `src/tools/update.sh` banner section.
- `tools.validate_env` now treats the known `pip check` Torch
  "not supported on this platform" message as a warning (non-fatal), while still
  failing on real dependency inconsistencies.
- Fixed `_load_model` return typing to align with runtime processor type,
  eliminating `ty` `invalid-return-type` failures in CI/local quality checks.
- Improved diagnostics issue-report readability/safety by using clearer prose and
  escaping token-leak snippets for Markdown/HTML-safe rendering.
- Stabilized terminal summary-table alignment by sanitizing non-ASCII note
  glyphs (for example warning emoji) in the model comparison table output.

## [0.2.0] - 2026-02-15

### Added

- New preflight package-risk diagnostics in `check_models.py` to surface common
  upstream compatibility issues early (MLX/MLX-VLM/MLX-LM/Transformers version
  mismatches and known problematic package states).
- End-of-run model comparison summary in `check_models.log` now includes a
  compact per-model table plus ASCII charts (TPS, overall efficiency, failure
  stage frequency) to make same-run model triage faster.
- History comparison output now includes run-over-run tabular deltas and
  transition-oriented ASCII charts to highlight regressions/recoveries/new or
  missing models across successive runs.
- Additional diagnostics stack-signal and long-context-breakdown analysis
  sections to better triage quality and failure patterns.
- Phase-aware failure attribution in model execution (`import`, `model_load`,
  `tokenizer_load`, `processor_load`, `model_preflight`, `prefill`, `decode`)
  so reports can identify the first failing runtime stage.
- Canonical failure metadata for each failed model:
  machine-readable `error_code` plus stable `error_signature` for clustering
  related failures across models/runs.
- Per-failure reproducibility bundle export (`output/repro_bundles/*.json`)
  with args, environment fingerprint, prompt/image hashes, traceback, captured
  output, and exact rerun command.
- Diagnostics report now emits copy/paste-ready issue templates per failure
  cluster, including canonical signature, minimal repro command, environment
  fingerprint, and likely upstream issue tracker.
- Model preflight validators for tokenizer/processor/config/snapshot layout to
  detect packaging and compatibility defects before generation begins.

### Changed

- Hardened runtime dependency handling for core MLX stack:
  `mlx`, `mlx-vlm`, and `mlx-lm` are now treated as required for model execution.
- Added explicit early package/runtime preflight in
  `src/tools/run_quality_checks.sh` to fail fast when required runtime deps are
  missing or broken.
- Updated CLI execution flow so argument/path validation and `--dry-run`
  behavior are evaluated before runtime dependency hard-fail, while still
  enforcing a hard stop before any model inference starts.
- Improved type-checking signal quality without globally hiding import/stub
  issues:
  - `pyright` missing import/stub diagnostics now emit warnings.
  - `mypy` uses targeted third-party overrides instead of global
    `ignore_missing_imports`.
  - `pyrefly` no longer uses wildcard missing-import ignore.
  - `ty` unresolved-import is now warning-level (visible, non-blocking).
- Strengthened E2E runtime gating to require usable `mlx` + `mlx-vlm` +
  `mlx-lm` for inference smoke tests.
- Improved CI MLX stub-generation step robustness to tolerate missing/generated
  stub-path variance as a non-fatal warning (typing-accuracy degradation only).
- Improved prompt-context compaction and keyword-hint summarization for long
  metadata inputs to reduce prompt bloat while preserving useful grounding.
- Improved diagnostics report reproducibility command construction and captured
  output sanitization for cleaner issue filing.
- Diagnostics reports now include preflight compatibility warnings in
  `diagnostics.md` (with likely package ownership and suggested issue trackers),
  and surface those warnings in the priority summary.
- Refactored portions of `check_models.py` for readability/maintainability
  without intended behavioral change.
- Tightened report/summary typing with explicit TypedDict/type-alias structures
  for model issue summaries and aggregate performance statistics.
- Failure clustering in diagnostics now keys off canonical signatures instead of
  raw message text heuristics, improving cross-model bucketing stability.
- JSONL result rows now include `failure_phase`, `error_code`, and
  `error_signature` fields (metadata format version bumped to `1.2`).
- Reduced `check_models.py` redundancy in model-issue summarization by removing
  dead `context_ignored` summary output paths and consolidating quality/delta
  bucketing flow while preserving report/log semantics.
- Added maintainer-oriented monolith guidance to
  `docs/IMPLEMENTATION_GUIDE.md` describing refactor order, correctness-vs-
  performance boundaries, and practical navigation/checklist steps for
  `check_models.py`.
- Further reduced duplicated summary-rendering logic by introducing shared
  top-performer/resource metric collectors reused by both HTML and Markdown
  issue summaries, keeping output semantics unchanged while trimming repeated
  key checks/formatting branches.
- Reused a shared quality-issue section collector for HTML and Markdown
  summaries, reducing duplicate failed/repetitive/hallucination/formatting
  extraction logic while preserving section content and emphasis.

### Fixed

- Fixed CI regression where runtime dependency hard-fail masked CLI argument and
  folder validation tests by firing too early in startup.
- Fixed diagnostics markdown formatting/lint edge cases in generated reports.
- Fixed `reportTypedDictNotRequiredAccess` lint failures by replacing direct
  optional-key TypedDict indexing with safe `.get(..., default)` patterns in
  summary formatters and JSONL tests.
- Fixed remaining `ModelIssueSummary` optional-key TypedDict access warnings in
  analysis helpers by replacing `setdefault(...)` read-path usage with explicit
  `get(...)` plus guarded initialization, preserving runtime behavior while
  satisfying strict Pylance/Pyright checks.
- Removed weak/avoidable lint suppressions by:
  - replacing shell word-splitting patterns with array-safe handling in quality
    and hook scripts;
  - replacing unnecessary test `type: ignore` usage with explicit type asserts.

## [0.1.1] - 2026-02-15

### Changed

- Compressed `src/check_models.py` logic (~45 lines removed) by deduplicating EXIF time extraction, quality-issue formatting, and token-leakage pattern data.
- Refactored `src/tools/check_outdated.py` to remove redundant string parsing logic and improve table detection.
- Improved test robustness by replacing flaky `time.sleep()` calls with deterministic `os.utime()` timestamp updates in `conftest.py` and image workflow tests.
- Reverted CI test skips for missing dependencies; the suite now enforces a hard failure if `mlx-vlm` is missing, ensuring environment integrity.
- Final run summary now reports configured `--output-log` and `--output-env` paths
  instead of always showing default log file locations.
- Synced documentation with current CLI behavior and outputs, including:
  `--output-diagnostics`, `--revision`, `--adapter-path`, and
  `--prefill-step-size`, plus `results.history.jsonl` / `diagnostics.md`.
- Updated quality and bot guidance docs (`AGENTS.md`, Copilot instructions,
  contributing docs, and pre-commit hook naming) to match the current CI gate.
- Improved `src/tools/run_quality_checks.sh` conda initialization by resolving
  the base path via `conda info --base` before fallback probe paths.

### Fixed

- Fixed `check_outdated.py` logic to correctly identify outdated package tables.

## [0.1.0] - 2026-02-08

### Added

- `--revision` CLI flag for pinning model versions (branch, tag, or commit hash)
- `--adapter-path` CLI flag for applying LoRA adapter weights
- `--prefill-step-size` CLI flag wired through to `generate()`
- TSV metadata comment line (`# generated_at: <timestamp>`) at top of output
- `error_type` and `error_package` columns in TSV reports for programmatic triage
- Warning when the default image folder (`~/Pictures/Processed`) does not exist
- JSONL v1.1 format with shared metadata header (prompt, system info, timestamp)
- 37 unit tests for pure-logic functions (`test_pure_logic_functions.py`)
- 11 report-generation edge-case tests (`test_report_generation.py`)
- 4 mock-based `process_image_with_model` tests (`test_process_image_mock.py`)
- Append-only history JSONL (`results.history.jsonl`) with per-run regression/recovery
  comparison summary (always prints "Regressions" and "Recoveries" sections)
- IPTC/IIM metadata extraction (keywords, caption) via Pillow `IptcImagePlugin`
- XMP metadata extraction (dc:subject, dc:title, dc:description) via `Image.getxmp()`
- Windows EXIF XPKeywords extraction (UTF-16LE semicolon-delimited)
- Keyword merging across IPTC, XMP, and XP sources (deduplicated, order-preserved)
- Structured stock-photo cataloguing prompt with Title/Description/Keywords sections
  and keyword taxonomy guidance (subjects, concepts, mood, style, colors, use-case)
- Existing metadata seeded into prompt (description, title, keywords, GPS, date)
- `Pillow[xmp]` extra for XMP metadata support (pulls in `defusedxml` transitively)
- Diagnostics report (`output/diagnostics.md`) auto-generated when failures or
  harness issues are detected — structured for filing upstream GitHub issues
  against mlx-vlm / mlx / transformers with error-pattern clustering, full
  error messages, traceback excerpts, environment table, and priority summary
- `--output-diagnostics` CLI flag for specifying diagnostics report path
- YAML config schema validation — warns on unknown threshold keys
- `CHANGELOG.md` (this file)
- `quality-strict` and `install-markdownlint` targets in root Makefile
- local npm-based markdownlint-cli2 tooling for CI/dev setup
...existing code...

### Changed

- `DEFAULT_TEMPERATURE` set to `0.0` (greedy/deterministic, matching mlx-vlm upstream)
- `--folder` and `--image` are now a mutually exclusive group in argparse
- Renamed `_apply_exclusions` → `apply_exclusions` (public API for testability)
- Bare URLs in generated Markdown reports are now auto-wrapped as `<URL>`
- Fixed `KeyboardInterrupt` / `SystemExit` / fatal exception handlers
- Unified CI action versions (`checkout@v4`, `setup-python@v5`, `setup-node@v4`)
  with concurrency groups and artifact upload on failure
- Added npm caching to CI quality workflow
- Fixed `setup_conda_env.sh` path references in documentation
  (was `./setup_conda_env.sh`, now `bash src/tools/setup_conda_env.sh`)
- Cleaned up VS Code settings (removed deprecated Python linting settings,
  aligned `typeCheckingMode`, fixed `launch.json` and `tasks.json`)
- Removed duplicated HF cache setup from `test_model_discovery.py` (now in `conftest.py`)
- Documented TSV and JSONL output format details in `src/README.md`
- Updated `QUALITY_IMPROVEMENT_PLAN_2026_02.md` — all items resolved

### Removed

- Duplicated sections in `IMPLEMENTATION_GUIDE.md` (Error Handling, Markdown Linting)
- 30 stale files archived from `docs/notes/` to `docs/notes/archive/`


The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
