# Changelog

Notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- CLI (0.17.107): `--reasoning-effort` now reaches only chat templates
  that read `reasoning_effort` or `reasoning_strength`, under the names each
  reads; a template reading neither gets nothing, with an INFO line per
  model, and keeps the automatic thinking budget (0.17.106 sent both names to
  every model and turned the budget off everywhere). Where a template
  enforces an allowlist without the value, `--reasoning-effort-fallback`
  (default `higher`; or `lower`, `skip`) picks the nearest declared level,
  falling back to the other direction when none lies that way. Each model's
  declared efforts are recorded for every run (`declared_reasoning_efforts`
  in prompt diagnostics and a `diagnostics.md` row), read with the rules of
  mlx-vlm's server model metadata reimplemented in the harness rather than
  imported from server internals. With the option, each record carries the
  outcome, the run summary counts outcomes, the run header records the
  request (a one-sided request is a settings change in `--compare-with`),
  and each native repro command carries the value that model's template
  received rather than the requested one.
- CLI (0.17.106): `--reasoning-effort LEVEL` passes a reasoning effort to
  chat templates, matching the flag mlx-vlm's generate CLI gained in 0.7.12
  (6dc1562). The value is lowercased and sent as both `reasoning_effort` and
  `reasoning_strength` (the name Muse Glimmer's template reads); omitted, each
  template keeps its own default; `none` or `off` is passed on as a value, and
  an empty value is refused. Like `--thinking-mode`, it turns off the
  automatic thinking budget and appears in native repro commands. Prompt
  diagnostics treat the two names as one setting, so a template that reads
  either is not reported as ignoring it.
- Run summary (0.17.96): the *Native repro* column distinguishes
  *confirmed*, *not reproduced* and *not yet checked* (replacing
  "candidate" and "crash: triage"), per observation mlx-vlm can produce.
  A native rerun is recorded with `python -m tools.add_native_repro_note
  MODEL OBSERVATION confirmed|not_reproduced --checker NAME --evidence
  TEXT`, which appends to `native_repro_notes.yaml` in the output directory
  keyed to the model revision and the mlx, mlx-vlm and transformers builds.
  A note applies only while all of those match; otherwise the cell reads
  "not yet checked (earlier check out of date)". A *Native reproduction
  checks* section lists the applicable checks with their evidence, and an
  unreadable file is reported with every observation left unchecked.
  Recurrence in the harness never counts. The accuracy-notes and
  native-notes files share one strict parser.
- Reports (0.17.95): optional human accuracy notes. A person who has checked
  an output against the image records it with `python -m
  tools.add_accuracy_note MODEL --reviewer NAME --note TEXT`, which appends
  to `accuracy_notes.yaml` in the output directory, keyed to the image
  digest, resolved model revision and output digest from the retained
  `results.jsonl`. The gallery and HTML report show a note above that
  model's evidence only while all three still match, mark it out of date
  otherwise, and say in the input summary how many outputs were reviewed. A
  file that cannot be read is reported and no output is shown as reviewed.
  Notes stay apart from the mechanical checks and are not written to
  `results.jsonl`. `src/output/accuracy_notes.yaml` starts with john's
  Molmo2-8B note from the 2026-10-25 review (tabby cats called calico); the
  retained reports show it from the next sweep.
- Docs: a fixed five-image set in `src/tools/image_set.tsv` (people,
  clutter, scene text, document text, infrared night lighting), each a
  public URL pinned to a repository commit with its SHA-256, and a README
  loop that runs each into its own `--output-dir` with
  `--image-source-url`. `cats.jpg` stays the integration regression image.
- Tooling: `tools.hub_precheck` reports each candidate's lineage as facts:
  the `base_model` chain its model cards declare (up to three hops, read
  from the hub) and the cached models that are, or declare, one of those
  bases (read from the local cards), plus whether the candidate is already
  cached. Nemotron-3.5-Bolt-E4B was downloaded before anyone saw that it is
  a post-trained Gemma-4-E4B-it with Gemma's vision tower; the check now
  prints "cached models with a shared base: mlx-community/gemma-4-e4b-it-4bit
  (via google/gemma-4-E4B-it)". Lineage never changes the OK/WARN/BLOCKED
  verdict, and it depends on publishers declaring their bases.
- Tooling: `src/tools/retire_old_models.sh` removes the retirement
  candidates from mlx-vlm#2546 that have a newer version of the same model
  line: 6 whose successor is in the roster (43.2 GB) by default, and 6 more
  whose successor exists only upstream (179.9 GB) with `--upstream`. It is a
  dry run unless given `--apply`, deletes exactly what the dry run lists
  through `hf cache rm` (which frees the shared blobs only those repos
  used), and refuses to run during a sweep. It does not run
  `hf cache prune`, which would also remove detached revisions anywhere in
  the cache, such as snapshots kept for reproduction.
- Docs: the README explains how to cap or raise image resolution without a
  dedicated flag, by passing `min_pixels`/`max_pixels` through
  `--processor-kwargs`. It records what was measured: Qwen2-VL-2B's prompt
  fell from 16,577 to 1,629 tokens and its peak memory from 9.35 to 5.58 GB
  on a 58 MP photo, while SmolVLM-256M, whose processor ignores the keys,
  was unchanged. It points to `--resize-shape` as the uniform alternative.
- Comparison: a run that used `--processor-kwargs` or `--resize-shape`, set
  on one side only, is now incomparable with the other, with the reason
  named ("generation settings differ: processor_kwargs unset → …"). It used
  to be rated "unknown", because a key recorded on one side was treated as a
  roster artefact. That rule still holds for every other setting.
- Run summary: a summary rendered by different check_models code than the
  run says so in a *Provenance* row, such as "run captured by check_models
  0.17.46 @ 3d1b38a4f; report rendered by 0.17.50 @ ac8822bc0; original
  assessments retained". Regeneration keeps each model's recorded assessment,
  so a newer report can still show an observation since retired (the
  2026-10-11 19:53 sweep's "Dusk" place name) or wording since corrected; the
  row explains why. It is absent when the version and revision match and
  neither tree had uncommitted changes. A dirty tree's revision does not
  identify its code, so a dirty capture or render always shows the row,
  marking that side "(uncommitted changes)", including a summary written
  during a run from a dirty tree.
- Streaming boundary: each generation records how its chunks released text:
  retained chunks, those that released none, the longest consecutive stretch
  of them, and the characters in the last chunk. A BPE detokenizer holds text
  back until a token starting with a space arrives, so Qwen2-VL's 265 repeats
  of `<|im_start|>assistant` on 2026-10-11 came as about 970 empty chunks and
  one large final flush. Results carry the counts as `metrics.text_delivery`
  in `results.jsonl`; diagnostics show them only after a stretch of 32 or
  more empty chunks, as counts with no quality judgment. Generation and its
  text are unchanged.
- E2E smoke: `test_full_inference_with_tiny_local_checkpoint` writes a
  random-weight Qwen2-VL checkpoint of about 150 KB (tokenizer, chat template,
  image-processor config, safetensors) to a temp directory and runs it
  through the full CLI, so CI exercises the real mlx-vlm `load` →
  `apply_chat_template` → `stream_generate` path without a download. Before,
  every CI test of that path was mocked and the cached-model e2e tests
  skipped. The language head is zeroed, so greedy output is a deterministic
  "a a a a". The technique comes from upstream mlx-vlm's
  `test_extraction_models`; the rest of upstream's suite tests mlx-vlm
  internals and was not adopted.
- System telemetry: each snapshot check (before load and after cleanup) also
  reads the GPU's "Device Utilization %" from `ioreg` (IOAccelerator, no root).
  Both checks sit outside check_models' own GPU work, and the reading drops
  to 0 within about 50 ms of that work ending, so a non-zero value is reported
  as "GPU activity detected outside harness inference". Each check is one
  instant, so zero readings do not show an uncontended run: activity between
  checks is never observed, and the diagnostics line says so. Records gain
  `gpu_samples`, `gpu_utilization_max_pct` and `gpu_busy_samples`;
  diagnostics state it per model, and the comparison's run-environment note
  counts the models affected. It is a recorded fact and does not exclude
  timings. The 2026-10-11 00:22 sweep ran while DxO PhotoLab
  exported photos and decoded at about half speed, with every other recorded
  reading unchanged. Continuous sampling, which overlaps inference, does not
  take the reading.
- Python 3.15 readiness checks, with no change to the CLI's behaviour:
  - CI's static-quality matrix gains a 3.15 lane (`next-python`, pre-releases
    allowed). mlx has no cp315 wheel or sdist yet, so `pip install -e
    "src/.[dev]"` fails there. On that lane only, and only when pip reports
    "No matching distribution found", the install step warns (naming the
    missing requirements) and skips the checks; a build, resolver or
    packaging failure still fails it, and so does a connection, certificate,
    proxy or HTTP error reading the index, after which pip prints the same
    message. Once the stack installs, the lane runs the full gate like the
    others. A test runs the step's real script against a fake pip for each
    case, and another keeps the lane's version equal to
    `probe_python_next.sh`'s default and above the fresh-env Python.
  - `make probe-python-next` with `PROBE_TORCH=1` retries torch,
    torchvision and torchaudio from PyTorch's own index (`PROBE_TORCH_INDEX`,
    default `https://download.pytorch.org/whl/cpu`) when PyPI has no wheel.
    They install from it with `--no-deps`, because `--index-url` also serves
    dependencies; their dependencies and the rest of the extra then resolve
    from PyPI. PyTorch's 3.15 macOS wheels are not on PyPI. The verdict names
    the source that worked and prints MPS availability.
  - `test_shipped_text_io_names_its_encoding` fails on text I/O in
    `check_models.py` or `tools/*.py` that leaves its encoding to the
    locale: `subprocess` calls with `text=True`, text-mode `open`,
    `read_text` and `write_text`. Python 3.15 (PEP 686) decodes these as
    UTF-8 whatever the locale. Ruff's PLW1514 covers only files and is a
    preview rule. The ten `subprocess` calls it found (the import probe,
    `system_profiler`, macOS toolchain and git probes, and three tools) now
    pass `encoding="utf-8"`. That is what a UTF-8 macOS locale already gave,
    so they decode the same on 3.13, 3.14 and 3.15.

- `tools/update.sh` ends every run, including one stopped by a failed step,
  with a table of what changed: each Python package (pip-installed), conda
  env, conda base, Homebrew and markdownlint-cli2 version that moved, was
  added or was removed, and each local `mlx`/`mlx-vlm` checkout's old and new
  revision with the number of commits pulled. It compares snapshots taken
  before the first change and at exit; an unchanged run says so in one line.
  Each snapshot marks the sources it read, so a first install into a source
  that was empty (markdownlint-cli2 before any `npm install`) is listed as
  new, and a source unreadable before or after the run is named instead of
  being listed as all added or all removed.

### Changed

- Development tooling: a committed `src/.npmrc` sets `ignore-scripts=true`
  and `min-release-age=7` for the markdownlint-cli2 install, so every
  `npm install` in `src/` (the Makefile, `setup_conda_env.sh` and plain
  `npm install --prefix src` included, not just CI and `update.sh`) skips
  dependency lifecycle scripts and versions under a week old, matching
  Dependabot's cooldown. npm 11.10+ honours the age setting (a
  `markdownlint-cli2@latest` install resolves the newest release older than
  a week); CI's npm 10 ignores it. Clears Skylos 5.6.0's
  `install_scripts_disabled` posture check and the npm half of
  `release_age_cooldown`.
- Docs: `src/README.md` (Requirements) now states when the next Python
  counts as ready. Every compiled dependency must publish a wheel for it on
  Apple Silicon; a pure-Python package without one (multiprocess, docopt) is
  not a blocker, and dev-only tools (Skylos and its tree-sitter bindings)
  are tracked apart from the runtime. The CI lane's `--only-binary :all:`
  and the probe's source builds are both cruder than that rule, so their
  verdicts need reading package by package; `copilot-instructions.md`
  points there. The stale "mlx has none yet" note is gone. Status for 3.15
  on 2026-11-12: PyYAML is the only compiled runtime dependency without a
  3.15 macOS wheel (miniaudio 1.73 and sentencepiece 0.2.3 now have them).
  The CI lane, which installs `.[dev]`, stops at PyYAML. After that it
  would stop at multiprocess (pure Python, not a blocker by this rule) and
  tree-sitter (dev only). tree-sitter's master builds 3.15 wheels for its
  next release (likely 0.27), but Skylos 5.6.0 still requires
  `tree-sitter<0.27`.
- Quality gate (0.17.123): the Skylos dependency scan targets
  `pyproject.toml` instead of `.`. Skylos finds manifests by walking the
  project root whatever the target, so the report is the same; a `.` target
  also repeated the source and dead-code analysis the quality gate had just
  run. *Behaviour preserved:* `tools.check_dependency_advisories` reads the
  same report (its coverage block and dependency findings) and decides the
  same way; the step's output and exit status do not change. *Benefit:* the
  step takes about 1 s instead of 35 s (sandbox, 4 cores, Skylos 5.6.0), in
  fast and full mode, and its report is 16 KB instead of 18 MB. *Evidence:*
  both commands run the same `scan_dependencies(project_root)` call; on
  this tree both reports have identical coverage, supply-chain summary and
  findings, and the checker's verdict is the same; a manifest added in a
  subdirectory was found by both. A new test runs real Skylos on a
  temporary project with a second manifest outside the target and compares
  the two scans. *Not claimed:* advisory findings were not compared, since
  the sandbox cannot reach OSV (both runs end "no dependencies were
  checked" here). The other Skylos runs stay separate: merging the
  dependency scan into the quality gate scan also costs only ~2 s, but
  Skylos writes one output format per run, so the gate would lose its
  concise listing; the danger gate scans from the repository root (for
  `.github` workflows), where a quality/secrets scan stops on `CLAUDE.md`;
  the suppression audit covers danger rules and runs in the background.

- Dependencies (0.17.121): `check_models.py` no longer imports numpy. Its
  only use was a fallback for numpy's version when the installed metadata
  could not be read, behind a guard for numpy being absent; the script
  calls nothing in numpy, and mlx-vlm, which does, cannot be imported
  without it. The version table now reads numpy's installed metadata, as it
  does for transformers and tokenizers. *Behaviour preserved:* the numpy
  version in every report, for any install that has metadata (pip, conda);
  a numpy importable from a path with no metadata now shows as unavailable.
  *Benefit:* one dependency guard and its module-level name are gone, and
  help and the tools that only read retained output no longer load numpy.
  *Evidence:* those five entry points run with a numpy package that records
  its import (no import, where there was one); the version read without an
  import equals the installed one; the Skylos quality gate still passes, so
  the declared dependency is not reported as unused. *Not claimed:* a
  start-up saving worth having (about 20 ms of 230 ms here). numpy stays a
  declared dependency; whether the project still needs to declare it was
  not examined. The Pillow and defusedxml guards stay: one gives a clear
  message, the other a working degraded run.

- Value checks (0.17.118, step 6 of john's 2026-11-11 plan): the rule "an
  integer, or a number, that is not a boolean" was written out at 30 places
  in four spellings (`isinstance(x, int) and not isinstance(x, bool)`, its
  two negations, and the `int | float` forms). It is now two predicates,
  `_is_int` and `_is_number`, and each of the 30 places calls one. They are
  `TypeGuard`s, not `TypeIs`: `bool` is an `int` to a type checker, so a
  `TypeIs[int]` that answers False for `True` would narrow the other branch
  wrongly. `_is_detail_count`, which was the same rule under a third name,
  is gone. Nothing else in the three JSON mechanisms is merged: the
  comparison readers raise, the image and metric readers return None or a
  default, and the generic dataclass decoder does not validate primitives,
  so their rules differ and they stay separate.

  Behaviour preserved: every one of the 30 places accepts and rejects
  exactly what it did; the rewrite was done by a script over the syntax
  tree and each replacement is the same boolean expression.

  Benefit: a reader sees what is being asked instead of decoding one of
  four spellings, and a new check cannot forget the boolean half.

  Evidence: before the change, each of the 30 checks was weakened in turn
  (made to accept booleans) and the suite run: 3 were noticed, 27 were not.
  Tests were then written at the level of each check's caller (a retained
  header with `true` for the duration or a count is refused; a `true` token
  count, image width, quantisation width, memory figure, token id or
  logit bias is not read as 1; a float is not read as an integer where an
  integer is required), all passing on the unchanged code, and the
  weakening repeated: all 30 are now noticed. The same tests pass after
  the rewrite, with mypy, ty and pyrefly.

  Observed, not changed: about 25 other places test `isinstance(value,
  int)` or `int | float` with no boolean exclusion, several of them on
  retained records or `config.json` values (`_prefill_seconds`,
  `_record_on_battery`, `_result_generation_tps` and
  `_result_peak_memory_gb` among them); a `true` there is read as 1. They
  are a different rule today, so they were not folded in: tightening them
  changes what damaged data does, which is a decision and not a
  simplification. The loader already refuses `true` in the header fields
  it validates.

- Start-up (0.17.117, step 5 of john's 2026-11-11 plan, second part):
  importing `check_models` no longer imports `mlx` or `mlx-vlm` and no
  longer spawns the import probes. `load_mlx_runtime(probe_first=...)` does
  what the module body did, once per process, and is called by `main()`
  (before the run clock starts, and before environment set-up, as before),
  by the isolated worker as its first step, by `tests/conftest.py` once per
  test worker, and by the two tools that need the runtime
  (`runtime_smoke.py`, probed; `hub_precheck.py`'s memory check, unprobed
  as before). Whether to probe is now the caller's statement instead of a
  guess from `sys.argv` at import time: `_import_probe_unneeded` and its
  flag list are gone, and `main()` passes `not args.dry_run`. The Pillow,
  numpy and defusedxml guards are untouched; they are a separate decision.

  Behaviour preserved: a run probes each dependency in a child process and
  then imports it, in the same order and with the same messages and
  `MISSING_DEPENDENCIES` entries; a dry run, an isolated worker and
  `CHECK_MODELS_SKIP_IMPORT_PROBE=1` import unprobed; a dependency that
  kills the probe is still reported as missing, with the minimal
  diagnostics report, instead of ending the run. Deliberately changed:
  `--help`, an argument error caught by the parser, and every program that
  only imports the script (`add_accuracy_note`, `add_native_repro_note`,
  `analyze_output_quality`, `hub_precheck`'s architecture check) no longer
  load MLX at all; a dry run given through an abbreviation (`--dry`) or a
  differently named launcher is no longer probed, where the `sys.argv`
  guess missed it; a probe-timeout warning now appears after argument
  parsing rather than before it; Pillow and numpy are imported before
  `mlx.core`, where they used to come between it and `mlx-vlm`.

  Benefit: a tool that reads retained output works where MLX cannot
  initialise (a headless or virtualised session, a broken install), and
  starts in about a tenth of the time (below). The crash protection has one
  named owner with a docstring, and the conditions for skipping it are
  arguments at three call sites. The dependence of the test suite on a
  working runtime is one explicit line in `conftest.py`.

  Evidence: 14 tests run the real entry points in new interpreters against
  fake `mlx` and `mlx_vlm` packages that record each import or kill the
  process on import. Nine were written first and pass on the unchanged
  code (probe then import for a run; no probe for `--dry-run`, `-n`, the
  worker and the opt-out; a killed probe becomes the missing-dependency
  exit and report; an unprobed dry run dies with the dependency;
  `hub_precheck` reads the working set from MLX; `runtime_smoke` names a
  dependency that kills its probe). Five fail on the unchanged code and
  pass now: importing the script, `--help` and the three tools load
  nothing, with a recording runtime and with one that kills on import. They
  replace the test of the removed argv-guessing function. Real plain,
  `--isolate` and `--dry-run` runs of a tiny local checkpoint in new
  processes, and an `add_accuracy_note` run on the retained output, give
  the same files before and after once clock times, measured durations,
  rates and digests are masked (9, 9, 1 and 12 files), and the JSONL
  records are structurally identical apart from clock times and measured
  floats. `--help` output is byte-identical. Measured in the Linux sandbox
  (best of three, CPU-only MLX): `import check_models` 3.42 s to 0.36 s,
  `--help` 2.07 s to 0.63 s, `add_accuracy_note --help` 3.40 s to 0.34 s,
  `hub_precheck`'s architecture check 1.92 s to 0.50 s; `runtime_smoke`
  (3.6 s) and `hub_precheck`'s memory check (1.7 s) are unchanged, as
  intended.

  Cost, and what is not claimed: the six deferred import statements and
  four `global` statements each carry a lint suppression (121 to 131 in
  all: ten added in the script and one in the new test helper, one
  removed with the argv guess).
  The test suite still loads MLX in every worker: with loading
  left out, 3 tests of the whole suite fail (the tiny-checkpoint end-to-end
  run, the weight measurement and the upstream drift check), but loading
  inside a test would rebind names another test has patched, so making
  the pure tests independent of the runtime is a separate change. Not
  checked: a run on macOS with Metal, and the times there.

- Reports (0.17.116, step 5 of john's 2026-11-11 plan, first part): report
  and reproduction code no longer receives the parsed-arguments namespace.
  `ProcessImageParams` is split in two without repeating a field:
  `ModelRunSettings` holds the load and generation settings that are the
  same for every model (each field is the command-line value of its name),
  and `ProcessImageParams` extends it with what belongs to one execution
  (model, image, prompt, timeout, verbosity, lazy loading, explicit-sampling
  provenance and the derived options). Reports receive `ModelRunSettings`;
  the three other values they read from the command line are typed fields of
  `ReportGenerationInputs` (`image_source_url`, `compare_with`,
  `isolated`). The per-model evidence chain, which carried the whole
  namespace through five functions to read one attribute, takes
  `requested_revision`. The per-model overlay is `dataclasses.replace`
  rather than a copied namespace with attributes set on it, and the
  reasoning effort a model's template received is an explicit argument of
  the command builder, so the requested effort cannot be printed by
  mistake. `_PARAMS_NOT_COPIED_FROM_ARGS` is gone: the class boundary now
  says which fields are copied from the arguments by name.

  Behaviour preserved: requested values, the explicit-sampling set and each
  model's effective values keep their meaning, and every report file is
  unchanged. A retained record still yields a command from the settings
  every model shared plus that model's own; the values a record does not
  hold are now one named record, `_UNRECORDED_RUN_SETTINGS`, in place of
  defaults repeated in `getattr` calls. Retained values are restated
  without checking their types, as before.

  Benefit: a report function's signature says what it reads, the type
  checkers see the settings, and a setting missing from the arguments fails
  at the conversion instead of silently taking a default in a report.

  Evidence: two tests written first, and passing before the change, run the
  real finalisation from a parsed command line (every load and generation
  option in one; isolation, no remote code, an adapter and sampling in the
  other) and pin the per-model, shared and retained-record commands and the
  recorded execution mode, trust setting and image source. A scratch
  harness wrote every output of three such runs (index, issue drafts, run
  summary, diagnostics, gallery, HTML, JSONL, history) before and after:
  identical once timestamps and durations are masked. Existing tests keep
  their expected commands; only how they build their input changed (37
  tests), and those that called finalisation with a one-attribute namespace
  now pass parsed arguments. Not changed: start-up, validation and the
  model loop still read the namespace.

- Shell tools (0.17.115, step 4 of john's 2026-11-11 plan, second part):
  seven Python heredocs moved out of the shell scripts into three checked
  files under `src/tools/`, each run by path so it works from any directory
  and, for the updater, before the project is reinstalled.
  `update_helpers.py` (standard library only, never touches `sys.path`)
  holds four updater steps: the conda plan's package names, who owns each
  package conda would change, removal of pip's stale `~name` backups, and
  the local-build smoke generation. `check_file_syntax.py` replaces the
  gate's YAML heredoc and the commit hook's YAML/TOML/shebang heredoc.
  `runtime_smoke.py` is the body of `run_runtime_smoke.sh`.

  Behaviour preserved: each step prints the same lines and returns the same
  status to its caller; the updater's conda guard still treats an
  unreadable plan or package list as "do not proceed"; a symlinked `~name`
  entry is still skipped, not removed. Deliberate differences: the two conda
  shell functions read their document from standard input, and the caller
  that made the temporary file redirects it (with the redirect inside the
  functions, Skylos reported SKY-D215 on the function parameter); PyYAML is needed only when a YAML file is checked; a YAML or
  TOML file that is not UTF-8 is now an error in the commit hook (it was
  decoded with replacement characters); the hook reads two bytes of other
  staged files for the shebang check rather than the whole file; both
  callers print one heading, "File syntax checks failed"; and the smoke
  generation decodes mlx-vlm's output as UTF-8 whatever the locale.

  Benefit: this Python is now covered by Ruff, the three type checkers and
  Skylos, and can be tested without a shell harness. It is not a line
  saving: the four scripts lose 264 lines (`update.sh` 2,000 to 1,857,
  `common_quality.sh` 429 to 401, `run_commit_hygiene.sh` 170 to 124,
  `run_runtime_smoke.sh` 61 to 14) and the three files add 439.

  Left inline on purpose: three short updater heredocs. The package
  snapshot (25 lines) must run with an untouched `sys.path`, because
  `src/check_models.egg-info` would otherwise be listed as an installed
  distribution; the MLX provenance report (31) and the model-cache probe
  (13) are small and only report. The 14- and 15-line heredocs in
  `probe_python_next.sh` and `run_skylos_danger_advisory.sh` also stay.

  Evidence: tests run the real callers, not only the functions. The
  updater's backup-cleanup step is run from another directory, with and
  without its opt-out, and the removal itself against a scratch
  site-packages; the smoke step runs against a fake `mlx_vlm` for pass,
  wrong output and failure; the existing conda-guard tests now go through
  the file; the gate's
  syntax step is run from another directory; the commit hook is run in a
  scratch repository with a broken TOML file, a broken YAML file and a
  non-executable script; `run_runtime_smoke.sh` printed the same line
  before and after, also from `/`. One source-text test (that `update.sh`
  contained the cleanup code) is replaced by those. Not checked: a full
  `update.sh` run on macOS, and the runtime smoke job on a Mac runner.

- Tools (0.17.115): `check_outdated.py`, `check_suppressions.py` and
  `update_readme_deps.py` put their own checkout first on `sys.path` before
  importing `check_models_data.safe_io`, as `add_accuracy_note.py` already
  did. Since 0.17.113 they tried the import first and added the checkout
  only on failure, which cannot recover when an older, non-editable
  `check_models` without `safe_io` is installed: the stale package is
  already loaded. A test runs each path-run tool with such a package ahead
  on the path; it failed for those three before the change. An editable
  install, which is what the setup scripts create, was not affected.

- Quality gate (0.17.114, step 4 of john's 2026-11-11 plan, first part):
  the Pyrefly step reads `[tool.pyrefly]` from `pyproject.toml` directly and
  adds only `--python-interpreter-path`. The gate used to generate a
  temporary config on every run (a 95-line Python heredoc) that switched off
  ignore files and the hidden-directory heuristic and restored the default
  excludes by hand, so that a worktree under `.claude/worktrees/` was not
  skipped. With the explicit file list the gate already passes, Pyrefly
  1.4.1 does not need that.

  Behaviour preserved: the same files are checked (`git ls-files`, tracked
  and untracked), with the same interpreter, version and search path, and
  warnings still fail the gate. Benefit: one configuration for editors and
  the gate, and `common_quality.sh` is 429 lines, from 524. Evidence: a new
  test builds a real repository whose `.git/info/exclude` ignores
  `.claude/`, adds a linked worktree under it, and runs the real gate
  function with the real Pyrefly: a clean worktree passes, and a type error
  in a tracked file and another in an untracked file both fail it. The test
  passed on the old generated config first. On a hidden worktree of this
  repository (53 files, two planted errors, one import of a worktree-only
  name), the old and new invocations printed identical diagnostics; on the
  checkout both report 0 diagnostics, 4 suppressed. The generated-config
  test and a source-text check of the target list are replaced by that test.

- File I/O (0.17.113, step 3 of john's 2026-11-11 plan): one implementation
  of the symlink-safe file helpers, in the packaged
  `check_models_data/safe_io.py`, whose docstring is the contract.
  `check_models.py` had its own copy and the tools had `tools/safe_io.py`
  (deleted); the script cannot import `tools`, which is not in the wheel. The
  script binds the shared functions under its existing private names, and
  tools and tests import them from the package.

  Behaviour preserved: no symlinked parent directory (bar the macOS `/tmp`
  and `/var` aliases, for the script), no symlink or non-regular file in the
  final position, parents created on write, a new file gets its mode less
  the umask, the read byte cap, and all-or-nothing replacement with no
  staging file left behind.

  Behaviour changed, where the two copies disagreed:
  - Replacement keeps an existing file's permission bits (the script's copy
    reset them to the default) and flushes the staged copy to disk before
    the rename (the script's copy did not).
  - Replacement refuses a symlink or other non-regular file at the target.
    The script's copy renamed over the link; every other write already
    refused one.
  - Replacement creates a missing file (the tools' copy required one), and
    the tools accept the macOS `/tmp` and `/var` aliases as the script does.
  - `read_text_no_follow` lost its unused `encoding` parameter; writes
    gained `append` and `mode` everywhere.

  Benefit: one place defines what "safe" means for every artifact, note and
  report the project writes, and a fix there reaches the script and the
  tools alike. Evidence: 15 contract tests in `TestFileHelperContract`
  (creation, replacement, failure cleanup, symlinks, permissions, reading,
  and a copy of the declared wheel contents exercised by an interpreter
  with no site-packages, outside the checkout). Run against the two old
  copies, 6 of the 13 behavioural rows failed on each: the differences
  above, the parameters each lacked, and one differently worded refusal. A wheel
  built from these sources held exactly `check_models.py` and the four
  `check_models_data` files; installed into an empty directory, it passed
  the same tests from outside the checkout, with `check_models` and the
  helpers both loading from the installed copy. The three caller tests that
  injected a failure by patching the script's writer now fail the disk
  flush instead, so they still prove the callers keep the previous file.

- Section landmarks (0.17.111, step 1 of john's 2026-11-11 plan): the
  `# SECTION:` map of `check_models.py` now describes what is under each
  heading. Twenty sections are added and six renamed (21 to 41), so the
  3,200 lines once under an EXIF heading and the 4,400-line "MODEL
  PROCESSING" stretch are named for their contents (report context, model
  snapshot facts, gallery chooser; argument validation, prompt diagnostics,
  error classification, model loading, generation, telemetry). The
  logging classes that sat under "FORMATTING" are back under the logging
  heading, and "RETAINED RUN VALIDATION" starts at its first validator.
  The agent instructions' section table lists all 41 in order. Behaviour
  preserved: no code moved or changed; the parsed program (AST) before and
  after the landmark edits is identical. Benefit: the file's own table of
  contents can be trusted for navigation. Evidence: the AST comparison,
  the existing test that keeps banners and table in step, and a check that
  every name the table quotes is defined in the section it is listed under.
- Timeout help (0.17.111): `--timeout` help, the README and the
  `TimeoutManager` docstring now say what each limit can do. In-process the
  limit is a `SIGALRM` timer covering load, prompt preparation and
  generation; it cannot interrupt a call stuck in native code until that
  call returns (a 3.9 s native call under a 1 s limit raised after 4.1 s).
  With `--isolate` the parent ends the child 120 s after the limit
  regardless. Behaviour preserved: wording only. A test ties the help's
  grace period to `_ISOLATION_TIMEOUT_GRACE_SECONDS`.
- Code quality review (0.17.109): `_classify_error` reads its fourteen
  status rules from one table (`_ERROR_STATUS_RULES`) instead of a chain of
  `if` statements, with the same statuses in the same order. `main()` tells
  a missing runtime dependency from other fatal errors by a
  `MissingRuntimeDependenciesError` type, not by the start of the message
  text. `PhaseTimer.duration` is a plain lookup, and unowned errors return
  the shared `_UNKNOWN_OWNER` constant.
- Quality gate (0.17.109): Skylos runs with `SKYLOS_NO_CLIPBOARD=1`, the
  opt-out it has had since 5.0.0, in place of a stub `pyperclip` module on
  `PYTHONPATH`. The `SKYLOS_GREP_BUDGET` comment gives Skylos's real default
  (120 s). `common_quality.sh` loses a stray level of indentation left by a
  removed wrapper (whitespace only), and `run_quality_checks.sh` prints one
  heading for `ruff format --check` in both modes and says why it clears
  its arguments before calling pyrefly with `"$@"` (ShellCheck SC2119).
- `update.sh` (0.17.109): the two remaining `pip show | grep | awk` version
  lookups use `get_installed_distribution_version`, the helper written
  because that pipeline once ended the script under `pipefail`.

- Docs: releases 0.1.0 to 0.15.0 moved from `CHANGELOG.md` to
  `docs/notes/archive/CHANGELOG_0.1.0-0.15.0.md` unchanged, halving the live
  changelog (332 KB to 163 KB); a closing "Older releases" section links
  it. `src/.gitignore` drops entries the root `.gitignore` already covers.
- CI: the Python 3.15 lane installs wheels only (`--only-binary :all:`),
  so it no longer compiles libraries from source (the 6 November run
  built miniaudio, PyYAML and tree-sitter). Until every requirement has a
  3.15 wheel it warns "No wheel for Python 3.15 yet" and skips its checks,
  then gates again with no edit. `NEXT_PYTHON_HOLD_UNTIL` is cleared: the
  2026-11-01 hold has ended and the wheel check now decides when the lane
  runs.
- CI: the macOS jobs stay on `macos-15` and add a `macos-latest` lane
  (GitHub's current default macOS image): static quality runs Python 3.14
  on both, 3.13 on macos-15 and the 3.15 lane on macos-latest, and the
  runtime smoke runs on both. Job and failure-artifact
  names now carry the runner.
- Image metadata and quality gate (0.17.105): the URL cap is stated
  directly as `MAX_IMAGE_URL_BYTES = 315_000_000` (sized for an uncompressed
  8-bit RGB 105 Mpx image; it is a download cap, not a pixel limit). Tests
  now fail if the download read is not bounded to cap + 1 and check that a
  response of exactly the cap is accepted. The Skylos quality-gate step
  moved into `quality_run_skylos_quality_gate` and
  `quality_print_skylos_advice` in `common_quality.sh`; advice lines whose
  path contains spaces are now repeated too, and an executable test covers
  a clean scan, advice, a path with a space and a failing scan (exit status
  kept, no reminder).
- Image metadata and quality gate (0.17.104): the `--image` URL cap is now
  `MAX_IMAGE_URL_PIXELS * 3` bytes (105 Mpx, ~315 MB), enough for an
  uncompressed 8-bit RGB image from a medium-format camera. `make quality`
  now repeats any Skylos finding that passes the gate (advice-only rules
  such as SKY-P401) under a "Skylos advice" heading just before the final
  "passed" line, without blocking. `--all-advice` showed no further hidden
  advice on this tree and is not used, because it would count advice
  against `max_quality = 0`.
- Image metadata (0.17.103): an `--image` URL download for EXIF reads at
  most `MAX_IMAGE_URL_BYTES` (64 MiB) and rejects a larger response before
  Pillow sees it; the EXIF step then logs the failure and continues without
  metadata. Removed the unused `FIELD_ABBREVIATIONS` table. The Skylos gate
  comments now say that optional advice rules such as SKY-P401 are reported
  but not counted by `max_quality = 0`.
- Tooling: rechecked every inline suppression against Skylos 5.3.0 and
  ty 0.0.95; all 133 are still needed, so the Skylos floor stays 5.0.0.
  5.3.0's Namespace-write fix covers only literal `setattr` names on a fresh
  `argparse.Namespace()`, so the SKY-D228 ignore on `_html_model_link`
  stays and its comment now names the shapes that still trigger it.
- Suppression audit: the long silent pause after "Found N inline
  suppression(s)" was the Skylos scan, which is CPU-bound (no network) and
  dominated by `check_models.py`. Skylos now scans only the files that carry
  `skylos: ignore` comments instead of all of `src/`, runs in the background
  while Ruff, mypy, ty and ShellCheck audit the rest, and the audit prints
  when it starts and when it is waiting on it. Verdicts are unchanged; on
  4 cores the audit went from about 116 s to 80 s.

- Report publication (0.17.94): the run summary is rendered once, from the
  in-memory run that the final `results.jsonl` header records, instead of
  being rendered from the file, the header rewritten, and the file read and
  rendered again. Loading and validating `results.jsonl` is now reserved for
  regeneration (`regenerate_run_issue_summary`), which still reproduces the
  shipped summary byte for byte. The pre-report checkpoint write and the
  final atomic header rewrite are unchanged. The run's end time is taken
  just before the documents that state it (the summary and the index), so
  every other renderer counts toward the duration and the summary's own
  render (about 0.03 s for the 43-model sweep) falls outside it, like the
  index's. If the final rewrite fails, the summary is rendered again from
  the checkpoint run so it matches the header on disk.
- Run comparison (0.17.94): the nested records retained under their own
  field names (prefill and memory changes, crash continuity, text
  divergence, architecture commits, observation history) are written and
  read by one helper driven by each record's field types and defaults,
  instead of naming every field three times. Each field is still
  type-checked on read, and a missing field takes a default only where the
  record declares one. Model changes, throughput flags and component
  changes keep explicit adapters for their renamed or paired wire keys.
  Retained comparisons serialize byte for byte as before.

- Reports (0.17.92): the Markdown and HTML choosers now read one shared
  column description (header, display text, sort value and cell kind) and one
  set of resource-highlight sentences. Links, escaping, the preview's
  reasoning disclosure and sorting stay in each renderer. HTML no longer keeps
  parallel header, cell, sort-value and raw-column-index lists, and chooser
  rows carry their execution and maintainer status instead of looking them up
  by model name. The standalone HTML report now builds the same report
  context as a full run, with no run mode recorded, in place of a second,
  overlapping context class. It still never measures the machine, and its
  lane still reads "unknown". Rendered reports are byte-for-byte unchanged.

- Tests: dropped the tests that froze documentation prose rather than
  behaviour. Gone: exact agent-guidance sentences (quality order, matrix
  preflight, Markdown style), the README evidence-boundary sentences, the
  output-hygiene sentence required in six documents, the README cache-filter
  wording, exact README table rows and CONTRIBUTING comments, and the
  approximate `check_models.py` line count (the count is also gone from the
  agent file map). In their place: every `make` command quoted in the
  agent and contributor docs and skills must be a real Makefile target; the
  README must list every value of the `execution`, `usability` and
  `maintainer_status` types the code emits; the pre-commit hook types are
  compared with `validate_env.REQUIRED_HOOK_TYPES` itself; and the test
  session now fails if any test adds, removes or rewrites a file under
  `src/output/`, which enforces the rule the removed sentence check only
  required the docs to state. Script-content checks for the updater, setup,
  packaging and quality tooling are unchanged.

- Docs: clarify the project's single-image MLX VLM comparison and diagnostics
  focus in both README openings, link to sample reports from the root README,
  and align package descriptions, keywords, and documentation links.

- Console (0.17.90): each model's answer streams to the terminal as it is
  generated, with or without `--verbose`. Without it, answers used to appear
  only at the end of the sweep. "Per-model results" is now the same in both
  modes: a summary line per model, any warnings, a note when a clean answer
  used the whole token budget, and full details for failures. It no longer
  prints each answer a second time. `--verbose` still adds DEBUG logging and
  mlx-vlm's own progress output. A non-verbose file log now holds each
  answer in the model's captured-output block, as a verbose one already did;
  `results.jsonl` is unchanged, since it drops captured stdout that matches
  the retained answer.
- Tooling (0.17.89): the Skylos floor is 5.0.0. Its danger gate found two
  new things. Report links now render as plain text unless their target has
  no scheme or uses `https` (SKY-D228); every target check_models builds is a
  `#anchor`, a relative artifact path or a GitHub URL, so reports are
  unchanged, but a `javascript:` target or a colon in an odd filename can no
  longer become a link. Dependabot waits seven days before proposing a new
  release in every ecosystem (SKY-D361). 5.0 no longer applies SKY-C303 to
  tests, so that suppression in `test_analyze_output_quality.py` went. It
  also found `PERFORMANCE_TIMING_FIELDS` unused; the constant is removed.
  ruff 0.17.2 and ty 0.0.91 need no changes; ty still needs the two
  `TextIOBase` property ignores. markdownlint-cli2 moves to `^0.24.0`, the
  range `update.sh` now writes. It pins smol-toml 1.9.1 itself, so the
  `smol-toml` 1.9.0 override, which had started forcing a downgrade, is gone;
  the `katex` override stays.

- Suppression audit (0.17.88): the Skylos recheck asks for `--format json-ci`
  instead of `--json`. Plain JSON carries Skylos's dead-code definitions and
  evidence, which made the report 16.9 MB in CI, over the 16 MiB safe-read
  cap, so the 0.17.87 gate failed on both Python lanes. The `json-ci` report
  holds the same `suppressed` list in about 1.9 MB.
- Suppression audit (0.17.87): `tools/check_suppressions.py` now rechecks
  `# ty: ignore[...]` and `# skylos: ignore[...]` comments as well. ty reruns
  on a copy without the ty ignore (keeping any `type: ignore` on the line), as
  mypy and Ruff already did for theirs. Skylos runs once, limited to the rules
  the comments name (about 45 s), and a comment fails when Skylos's own list
  of suppressed findings does not include it. Bare ty or Skylos ignores fail,
  and a ty or Skylos ignore with no reason of its own may share the reason
  given by a `noqa` on the same line. A missing ty or Skylos skips its checks,
  as a missing ShellCheck already does. The first run found two stale
  `SKY-P401` ignores on bounded `os.read` calls in `check_models.py` and
  `tools/safe_io.py`. Skylos 4.50.1 treats a read with a size as bounded, so
  both are removed.
- CI (0.17.86): the Python 3.15 lane of the static-quality job is on hold
  until 2026-11-01, at john's request. Before that date (UTC) its install
  step prints a "Next Python lane on hold" warning and skips the checks,
  as it already does while a requirement has no 3.15 distribution; from
  2026-11-01 it installs and gates again with no edit. The date is
  `NEXT_PYTHON_HOLD_UNTIL` on that step; clearing it lifts the hold early.
  The lane runs alongside 3.13 and 3.14, so the hold frees a macOS runner
  rather than shortening the workflow.
- Tooling (0.17.83): the last traces of locally generated `typings/` stubs,
  retired earlier, are gone. ty runs without `--no-respect-ignore-files`
  (an explicit target inside a gitignored worktree is still checked; tested
  with a planted error under `.claude/worktrees/`). The generated Pyrefly
  config no longer falls back to the primary checkout for search-path
  entries missing from a worktree. `typings/` is no longer gitignored, set
  as the VS Code and Cursor stub path, excluded from search and file
  watching, or skipped by the suppression auditor and two tests. A
  `typings/` folder left in an existing checkout now shows as untracked and
  can be deleted.
- Typing and lint suppressions (0.17.82), from john's review of 0.17.79,
  with no output change. mypy now uses `follow_imports = "silent"` instead of
  `"skip"`, which had turned every name imported from a Python package into
  `Any`: `jinja2.Environment` and `huggingface_hub.scan_cache_dir` now carry
  their real signatures, and the monolith still passes. The per-package
  overrides that skipped or ignored defusedxml, huggingface_hub, numpy, PIL,
  psutil, PyYAML, tokenizers and transformers are gone, since each ships
  inline types or a `types-*` dev stub; only mlx and mlx-vlm, often editable
  checkouts, may still be unresolved. The comments about locally generated
  stubs, retired earlier, are removed. Thirteen inline suppressions go with
  less code: `_add_output_path_arguments` takes the parser and creates its
  own "Output Reports" group, so the `_ArgumentAdder` protocol and its
  `ANN401` are gone (help and parsing are byte-identical); `test_e2e_smoke.py`
  drops its copy of the Hugging Face cache setup that `conftest.py` already
  performs before collection, and with it six `E402`s; `os` in
  `tools/hub_precheck.py`, `hub_precheck` in `test_quality_analysis.py` and
  `huggingface_hub` in `test_dependency_sync.py` are top-level imports, and
  `conftest.py` reads `check_models.logger` directly.
- Cleanup (0.17.80), from john's review of 0.17.77, with no output change:
  `DiagnosticsArtifacts.run_issue_summary` (set, never read) and
  `FormattingThresholds.min_separator_chars` (unused) are gone. `ReportLink`
  is folded into `ReportTargetLink`, whose target now carries the leading `#`
  for an in-page anchor; Markdown and HTML escaping are unchanged.
  `_RESERVED_PROCESSOR_KWARG_KEYS` is built from `_SENT_GENERATE_KEYWORDS`
  plus `audio`, `image`, `processor` and `prompt` instead of repeating the
  31 keywords. `_dedupe_preserve_order` is `list(dict.fromkeys(items))`, and
  the comparison record's `text_divergence` entries use `_asdict()`, whose
  keys and values match the previous hand-written mapping.
- Generation setup (0.17.79): one `_build_generate_kwargs(params)` builds
  every keyword sent to mlx-vlm, always-sent settings first and optional ones
  after; `_build_generate_extra_kwargs` is gone. The prompt preparation reads
  `generation_config.json` once and passes it to the sampling defaults, the
  unapplied-settings record and the EOS-by-file record, which each read it
  before. `_DECLARED_SAMPLING_KEYS` and `_DECODING_KEYS` are built from
  `_CHECKPOINT_SAMPLING_DESTS` instead of repeating it. Behaviour is
  unchanged apart from the key order of a result's recorded `generate_kwargs`.
- Generation timing (0.17.75): the single-use `PerfCounterTimer` class is
  gone; `_execute_prepared_generation` takes two `time.perf_counter()`
  readings at the same points, so the generation duration still starts
  before the decode phase and stops after `mx.synchronize()`. `PhaseTimer`
  is unchanged.
- Report escaping (0.17.74): the stateless `HTMLSelectiveEscaper`,
  `MarkdownPipeEscaper` and `DiagnosticsEscaper` classes and their global
  instances are now the functions `escape_html_selectively`,
  `escape_markdown_cell` and `escape_diagnostics_cell`. Escaping is
  unchanged, including the two whitespace policies; a differential check of
  90,000 random inputs against the old classes gave identical output.
- Quality analysis (0.17.73): `GenerationQualityAnalysis.word_count` is gone.
  It was computed on every analysis and never read; the standalone
  `tools/analyze_output_quality.py` counts its own displayed words. Worker
  payloads that still carry the key load unchanged.
- Prompt burden (0.17.72): the intermediate `PromptBurden` dataclass is gone.
  `_prompt_burden_for_result()` builds the retained `prompt_burden` record
  directly, with the same six keys and values; the never-read fields
  (`visual_tokens_est`, `template_tokens_est`, `original_width`,
  `original_height`, `source`, `reason`), the image-profile argument, and
  the classifier's unused reason return went with it.
- Report publication: roll back the 0.17.65/0.17.67 optimizations and their
  draft-summary timing machinery. Summaries once again read the canonical
  JSONL and are refreshed after final timing is recorded. Keep atomic writes,
  failure recovery, and tests for timing consistency and exact regeneration.
- Prompt diagnostics: remove the unused image patch count field and display
  row; no runtime path populated it. Older retained records remain readable.
- Diagnostics (0.17.71): the "Processed image" row and the
  `processed_image_width` / `processed_image_height` fields in
  `prompt_diagnostics` and `prompt_burden` are gone. They only ever held the
  `--resize-shape` bounds, in swapped order, shown as measured dimensions even
  for models that ignored the option. The setting stays recorded in
  `generation_settings`; retained runs that carry the old fields still load.
- Docs (0.17.70): `--resize-shape` is no longer described as a uniform,
  fixed-size resize. mlx-vlm ignores it, with a warning, for models that use
  its own image processor (Phi-3-Vision, Molmo, Gemma 4 and others), and
  otherwise scales the image to fit the bounds while keeping its aspect
  ratio. The README and `--help` now say so, including that mlx-vlm reads
  the first of two values as the width bound.
- Run duration (0.17.69): the end-to-end duration again includes building
  the run summary. 0.17.67 took the finish time before the summary ran, so
  a summary that took two minutes went uncounted (600 s recorded against
  720 s elapsed). The summary's findings, tables and comparison are now
  built before the finish time; only its timing rows, artifact links,
  Markdown rendering and file write come after it.
- Refactor (0.17.67): the run issue summary is rendered once per run
  instead of twice. It now runs with the output index, after the other
  renderers, from the written run carrying the end-to-end duration and
  finish time the final header records. One helper computes that timing for
  both, so the summary, the index and the header still agree, and
  regenerating the summary from `results.jsonl` reproduces it byte for byte.
  If the final header rewrite fails, the summary is rendered again from the
  pre-report header that stays on disk, so it still matches the file.
- `--models` (0.17.66) now rejects a model ID listed more than once,
  before any model runs. Every report keys a run's attempts by model ID, so
  a repeated ID used to run inference twice and then fail publication's
  identity check; repeated attempts are not a supported feature. A
  regression test now covers the gallery evidence pairing from 3cf3bf5:
  each result keeps the row built from it, including when a directly built
  context holds two attempts of one model.
- Refactor (0.17.65): publication no longer reads `results.jsonl` back.
  Writing the retained run returns the text it wrote, and validating that
  text gives exactly what reloading the file gave. Both run-summary renders
  (before and after the header gains its end-to-end timing) and the closing
  "Overall runtime" line now use it, which removes three reloads per run.
  Finalisation returns the rewritten record instead of a boolean, and when
  the rewrite fails the closing line still falls back to the pre-report
  header. The summary renderer takes an already-validated run, and
  regeneration still loads and validates the file. Atomic writes, the
  manifest and the second render that keeps the summary's timing in step
  with the header are unchanged; dropping that render is left for a
  separate change.
- Refactor (0.17.64): the run issue summary's four no-summary branches
  (JSONL failed, not needed, diagnostics failed, rendering raised) share one
  helper that removes an earlier run's summary and records why none was
  written. Not needed is still no outcome unless the stale summary could not
  be removed. The two prerequisite messages lose their trailing full stop,
  and every cleanup failure is appended as "; stale summary cleanup failed:
  ...".
- HF skills: replace the broad generated CLI entrypoint with maintained,
  command-specific guidance using the existing conda CLI and explicit mutation
  boundaries. Refresh the on-demand command catalogue from Hub 2.2.0, omit
  generic installer advice and the incorrect output-format prose, and lint the
  maintained entrypoint. Remove the empty generated-skill manifest. Cache
  diagnosis no longer treats a working alternative checkpoint as proof of blame.
- Refactor (0.17.63): a run comparison reads and decodes the retained
  history once. The throughput bands and the earlier-runs-on-this-mlx-version
  count share that read but keep their own selection rules: the version count
  still reads every record, whatever the workload, and still reports
  unreadable lines; the bands still need a matching fingerprint, revision and
  settings. The bands now take the newest ten matching runs, newest first,
  and stop there instead of keeping every matching record and slicing. The
  mlx-vlm architecture-commit list also reads the `mlx_vlm/models/` listing
  (`git ls-tree`) once per revision instead of once per affected
  architecture. Results are unchanged.
- Refactor (0.17.62): the gallery rows and the per-model assessment,
  provenance and result lookups are derived once per report context and
  shared by the HTML report and the Markdown gallery. Before, each model's
  row (final-answer extraction, field parsing, preview) was rebuilt four
  times when both formats were written, once for each chooser and once for
  each complete-evidence section, and the lookup dictionaries were rebuilt by
  each renderer. The values are computed on first use and held by the context
  itself, not a module-level cache, so a context built with `replace()`
  derives its own. Escaping, Markdown formatting and each format's layout are
  unchanged, and each evidence entry keeps the row built from its own result.
- Refactor (0.17.61): each answer's labelled fields are parsed once per
  analysis. The keyword-repetition check, the Title/Description/Keywords
  checks, the keyword counts and the hint-keyword count read one parse
  instead of extracting the sections up to three times and splitting the
  keywords again for each. Each `<Field> hint:` line is likewise compared
  with the answer once, so the description-hint percentage and the echo flag
  share one measurement. The percentage still counts the first description
  hint, the echo still checks every hint line, and repetition still reads a
  keyword list under the general profile. Results are unchanged.
- Correct four repository skill workflows: use upstream-shipped types instead
  of removed stub-generation commands; isolate benchmark artifacts and require
  exact-build restoration; handle native stream cleanup and empty results;
  preserve original reproduction settings and avoid attributing a runtime
  failure solely from an alternative checkpoint's success. Add focused skill
  contract and fake-stream example checks, without model inference.
- The blind lane now gives the model the camera's capture date/time and
  GPS, stated as authoritative facts it must not claim are visible, just as
  the assisted lane does. The camera records them; only the descriptive
  title, description and keywords are hints, and the blind lane still
  withholds those. Without camera data the blind prompt is unchanged. Reports
  label the blind lane's prompt as having "no description or keyword hints"
  instead of "the prompt carried nothing about the image", and blind runs
  before and after this change are not comparable because the prompt differs.
- Tooling: the Skylos floor is now 4.50.1, which no longer reports a prose
  value under a key containing "token" as a high-entropy secret (SKY-S101;
  4.47.2 still does). The `skylos: ignore[SKY-S101]` on the
  `upstream_prefill_first_token` label is gone. ty 0.0.89 and mypy still
  need the `TextIOBase` property-override ignores.
- Tooling: cleared the seven warnings Pyrefly 1.4.0 adds (it fails the gate
  on warnings): exception tuples no longer list a subclass of a type they
  already catch (`URLError` and `FileNotFoundError` under `OSError`,
  `HFValidationError` under `ValueError`), a redundant `bool()` is gone, and
  an unreachable `raise SystemExit` after a `NoReturn` call is removed.
  Behaviour is unchanged.
- CI: the Static Quality job checks out full history (`fetch-depth: 0`), as
  skylos-advisory already did. On pull requests, Skylos 4.50's `cicd gate`
  resolves `origin/main` as its policy base, which a shallow checkout lacks:
  Dependabot PRs #31 and #32 failed with "Could not resolve PR policy base:
  origin/main". Local runs, on Skylos 4.47.2, never saw it. A test keeps both
  Skylos-gated jobs on full history.
- Diagnostics partitioning is one pass: each result goes into the first named
  bucket its cached assessment selects (actionable, observations,
  indeterminate, compliance, clean), and each bucket keeps the actionability
  order. It replaces five filter-and-sort passes. The diagnostics Markdown
  and the HTML report each partition once and pass the result to both the
  evidence blocks and the provenance section, which had partitioned again.
  Output is unchanged; a test checks the buckets on a mixed run.
- The streaming repetition guard keeps its 600-character tail as chunks
  arrive instead of joining the whole accumulated output every 25 chunks and
  discarding all but the end. The full pieces are still kept for the answer
  and detection is unchanged; tests pin the abort to the first check past the
  200-token floor and show a unit whose four repeats overflow the tail never
  trips it. The saving is small, not seconds off a sweep.
- Docs: the implementation guide records the deferred plan to store
  run-level comparison notes as stable codes with factual fields at the next
  retained-schema change, replacing regeneration's match on the GPU note's
  English prefix while keeping that reader for older runs.
- Removed the `unverified_place_name` observation (added in 0.17.29). It was
  a heuristic: any capitalised word after "in", "at", "near", "outside" or
  "around", or beside a feature word such as "River", that the prompt lacked.
  The 2026-10-11 19:53 sweep flagged "Dusk", "Sunset" and "UK" (the hints
  said "United Kingdom"), and telling places from other words would need an
  ever-growing exception list or a gazetteer. Retained records with the code
  still render, names included. `prompt_hint_echoed`, a counted overlap with
  the prompt's own hint, stays.
- CLI declarations: the KV quantization schemes are one `KvQuantScheme`
  alias, and the `--kv-*-scheme` and `--assessment-profile` choices are read
  from their aliases instead of repeating the values. Options no longer
  restate argparse's own defaults (`default=None`, `default=False` on
  `store_true`, `type=str`), `--repetition-context-size` uses
  `DEFAULT_PENALTY_CONTEXT_SIZE`, and the presence/frequency context-size
  checks share one loop. Every option parses as before and keeps its flags,
  default, choices and help (compared before and after, including `--help`).
- Comparison serialization: `_run_comparison_to_json` builds its lists and
  mappings with checked `list[JsonLike](...)`/`dict[str, JsonLike](...)`
  constructors instead of casts, and `_run_comparison_from_json` passes the
  baseline identity fields straight to `RunComparison` instead of through an
  untyped dict, so the type checkers now verify those names and types. The
  retained JSON is unchanged.
- Report construction: the HTML and Markdown galleries take their per-model
  evidence entries, in usable-first order, from one
  `_gallery_evidence_entries` helper; the Markdown gallery's run-input lines
  use the same `_run_input_rows_for_context` as HTML; and
  `DiagnosticsPartitions.highlighted` names the maintainer lane (actionable,
  observations, indeterminate) that three call sites assembled by hand. HTML,
  gallery and diagnostics output are byte-identical on a fixed fixture.
- Run summary: "Earlier retained runs on this mlx version" is a fact about
  the run, so it now shows (and is logged) even when a changed image or
  prompt withholds the comparison; the 2026-10-10 sweep on a new photo had
  recorded the count but hidden it.
- Run summary: the baseline comparison adds "Earlier retained runs on this
  mlx version", counted from `results.history.jsonl` across every retained
  run ("none recorded on ..." for a fresh build, never "first run": history
  records that cannot be read or name no mlx version are counted and named as
  not counted, since the history may be incomplete). Four sweeps
  (2026-09-26, 10-02, 10-08, 10-09) were each the first on a new mlx build and
  timed short-prompt prefill (and diffusiongemma's decode) far slower; an
  immediate rerun was back to normal each time.
- Run summary: the prefill-change caption says "not demonstrated speed
  changes" (it covers slowdowns too); the evidence-links caveat moves to the
  Full artifacts section and "What this run measures" below the run facts, so
  the maintainer verdict is followed directly by the numbers.
- Run summary, from john's review of the warm-rerun report:
  - A "Largest prefill changes" table lists up to five models whose prefill
    tok/s left the fixed 0.85-1.15x band, as baseline and current prefill
    seconds (prompt tokens divided by prefill tok/s) with the ratio, retained
    as `prefill_changes` in the comparison block and logged. A prefill change
    alone counts as a change, so the log no longer follows such a table with
    "No changes beyond noise".
  - The opening now says how the completions split (no observations,
    prompt-compliance only, native-reproduction candidates) and carries the
    evidence-link caveat, which moved up from Run context.
  - "Model quality at a glance" is now "Mechanical checks at a glance", with a
    "Native repro" column; the gallery's "Avoid for This Run" is now "Major
    Mechanical Concerns in This Run".
  - The baseline caveat no longer weighs single-model against broad changes:
    a difference shows that outputs or timings differ, not why.
  - Review rows show a compact history (the longest-running observation and
    how many others, with the newest start); every observation's dates are in
    a collapsed "History by observation" table below.
  - Observation clusters list only signatures shared by two or more models,
    with each model linked to its diagnostics.
- `index.md` names the run's start time and the check_models version and
  revision that produced it.
- Baseline comparison: the generation and prefill tok/s ratio rows name the
  model at each end of the range ("lowest 0.12 (model), highest 8.73
  (model)"), kept as `min_model`/`max_model` in the retained comparison block;
  older records still show the bare range. The caveat under the comparison
  says each sweep makes one generation per model, with no warm-up or repeats.
- "Major concerns" now says which kind: "generation" when generation itself
  failed (no text, repeated text, no final answer, incomplete reasoning) and
  "answer format" when an answer was generated but missed the requested form
  (labels, extra text, echoed instructions, token limit). Shown in the
  run-summary tables, both gallery choosers and `index.md`'s counts; within
  major concerns the at-a-glance table lists answer-format rows first.
- The run summary's History column puts each observation on its own line,
  with its first date after a comma rather than a second semicolon.
- Report wording now states what was measured, not why: the prompt-hint
  observation reads "Output repeats the prompt's own hint text" (it no longer
  adds "instead of describing the image", which an accurate hint does not
  show), and the baseline caveat drops "usually run-to-run variance".
  `index.md`'s top
  observations line is now "Observations, most important first" and says how
  many kinds it left out, so an unlisted observation is not read as absent.
  The changed-text table shows "not recorded" for missing quantization and
  prompt token counts, matching its decoding column, instead of "-".
- `tools/update.sh`: a local `mlx`/`mlx-vlm` checkout whose HEAD cannot be
  read leaves the checkouts unmarked in that snapshot, so it is named as not
  compared instead of listed as removed. An all-unreadable change list now
  says "Nothing changed in the sources read both before and after the run".
- Tooling: `[tool.pyrefly]` now sets `search-path = ["."]`, so editor
  Pyrefly (1.3.2) finds `check_models` and `check_models_data` instead of
  inferring an import root of `src/src`; the gate's generated config already
  led with that directory.
- Baseline comparison: the changed-text models moved from one long summary
  line into a collapsed table (decoding, checkpoint quantization, shared
  prefix, length, prompt token count); the summary now states only the count.
  A new "Greedy text changed, by checkpoint quantization" row counts greedy
  pairs by this run's `config.json` quantization (quantized, not quantized,
  unknown). The 2026-10-08 sweep showed why: 10 of 22 quantized greedy models
  changed text and 0 of 7 unquantized ones did, while mlx changed its
  quantized matmul accumulation.
- Baseline comparison: the per-model mlx-vlm commit list now also covers the
  sibling model packages and modules each architecture imports directly (one
  level, read at the current revision) and names them. Before, `qwen3_5_moe`
  models showed "no commits touched" although mlx-vlm #2442 changed the
  `qwen3_5` code they import.
- Tooling: the Skylos floor is now 4.47.1, which also reads module lists
  from platform wheels, so an editable mlx build is verified and the two
  `skylos: ignore[SKY-D223]` comments on the first `mlx` imports are gone.
  Checked with an editable-style mlx (loader-only `RECORD`, dev version):
  4.44.0 still flags both imports, 4.47.1 flags neither. The SKY-S101
  workaround still reproduces on 4.47.1 and stays.
- Tooling: `src/package.json` overrides two transitive markdownlint-cli2
  packages that Skylos's dependency scan flags: katex 0.19.0
  (GHSA-238p-pmpm-9mq7, fixed in 0.18.2; micromark-extension-math still asks
  for `^0.16.0`) and smol-toml 1.9.0 (GHSA-r4xh-jqrq-34v2; markdownlint-cli2
  0.23.3 pins 1.8.0 exactly). Drop each once markdownlint-cli2 resolves a
  fixed version itself.
- Docs: reworded the run-summary line in `src/README.md` that Skylos
  4.47.1 flagged as a prompt-injection pattern (SKY-D260); the meaning is
  unchanged.
- Typing: dropped a `cast` in `_auto_thinking_budget_kwargs` that ty 0.0.85
  reports as redundant (`redundant-cast`).
- Run summary: the quality table reports counted facts that were hidden.
  "Prompt / output tok" puts the generated tokens beside the prompt tokens,
  and the intro names the run's `max_tokens`, so an answer that only just
  fitted (995 of 1,000) is visible. A "Hint text" column gives the percent
  of the description lying in four-word runs copied from the prompt's
  description hint, so copying below the 80% echo threshold shows too. It
  is rounded down, so a shown 80% always means the echo flag fired. It
  is retained as `description_hint_percent` and `description_hint_scope`
  observation details (metadata profile only).
- Assessment: an answer with no labelled field at all is now compared whole
  with the description hint, so a prompt-only model that hands the hint back
  as plain prose (moondream2 did) is reported as a repeated prompt hint. Its
  share is labelled "of answer".
- Run summary: each model requiring review gets a History column (a fact
  row for a crash) that dates every one of its current observations
  separately, so a new problem is never folded into an older one: when it
  was first observed, and how many of the model's consecutive runs, ending
  with this one, have shown it. A run that did not attempt the model is
  skipped, and one that showed it without the problem breaks the run of
  runs. The runs come from the git history of the tracked `results.jsonl`
  at `HEAD`, independent of `--compare-with`, deduplicated by run timestamp
  so a report-only correction counts once. At most 80 versions are read,
  stopping at the first schema-2 version. While older runs stay unread, a
  streak that no attempt read has broken is shown as "N+ runs" since a date
  "or earlier", and an observation no run read showed is "not in earlier
  runs read" rather than "first seen this run". The facts are kept in a new
  `observation_history` metadata block, so regeneration needs no git.
  `results.history.jsonl` is unchanged.
- Tooling: the Skylos floor is now 4.44.0, which checks an editable
  install's modules against the published wheel instead of trusting its
  loader-only `RECORD`. That clears SKY-D223 for an editable mlx-vlm, so its
  suppression is gone. It does not clear it for an editable mlx: Skylos only
  reads module lists from portable (`py3-none-any`) wheels, and mlx publishes
  platform wheels alone. The two first `mlx` imports keep
  `skylos: ignore[SKY-D223]` until Skylos accepts those. The SKY-S101
  workaround is still needed on 4.44.0.
- Agent skills: added `triaging-skylos-findings`, adapted from the `skylos`
  and `skylos-security` skills in Skylos's own repository. It maps where
  Skylos gates this repo, how to reproduce one finding with `--select` and
  `--diff`, the order to try fix, inline suppression and project-wide ignore,
  re-testing version-pinned workarounds after an upgrade, and drafting a
  Skylos bug report. The parts about developing Skylos itself (its rule
  catalogue, cloud policy, LLM evidence filters, benchmarks) were left out.
- Tooling: the Skylos quality gate no longer fails locally on "Unverified
  import" (SKY-D223) for `mlx` and `mlx_vlm`. Both are declared dependencies,
  but Skylos 4.43.2 reads an editable install's `RECORD` (the loader files)
  as an empty module inventory and prefers it over PyPI's. The three first
  imports carry a `skylos: ignore[SKY-D223]`, to be dropped once Skylos fixes
  that precedence.
- Tooling: `update.sh` checks conda's whole `update --all` plan before
  applying it. It reads the `--dry-run --json` transaction and checks every
  package the plan links or unlinks, so a new install or removal that would
  replace a pip-owned package (shown without an `old --> new` arrow in conda's
  text listing) now skips the update, as a pip-owned upgrade already did. A
  package that is not installed at all counts as a plain new install, and an
  editable install (`<develop>`) counts as pip's.
- Tooling: `make quality` runs the Skylos dependency scan as its own step,
  gated by `tools/check_dependency_advisories.py`. Skylos's ignore list does
  not reach dependency advisories, so this is the only way to accept one.
  - The scan passes only with a recognised successful status (`complete`, or
    `complete_with_unresolved_versions`, accepted for pyproject's unpinned
    ranges) after a completed advisory query, with at least one dependency
    checked and no lockfile parse or resolution gaps. Any other status,
    including `no_supported_manifests` and unknown ones, fails.
  - An exception matches only its reviewed advisory, ecosystem, package,
    version and manifest. The one exception is GHSA-vfj7-8cjw-p6xm for braces
    3.0.3 in `src/package-lock.json` (via markdownlint-cli2), which has no
    fixed release. It fails once a fixed release exists, and the step says
    when it stops being reported, so it can be dropped.
- `check_models.log` and the terminal are quieter. Under `--verbose` a
  completed model's entry in "Per-model results" is its summary line plus any
  warnings. The token, timing and memory tree has been removed (the
  comparison table and `results.jsonl` carry those figures), along with the
  metrics legend that described it and the rule between consecutive
  completed models. Each model's captured block in the file log no longer
  repeats the harness's own console lines, which the log already holds with
  their own timestamps. It also drops progress bars that reached 100%,
  keeping a bar's last state only when it stopped short or moved bytes.
  `results.jsonl` and the failure capture are unchanged.
- Tooling: fresh envs from `tools/setup_conda_env.sh` (and the manual conda
  recipe) are created on Python 3.14, the version the working `mlx-vlm` env
  runs; the supported floor (`requires-python`, type-checker targets) stays
  3.13 and CI still tests both. `make probe-python-next` now defaults to
  3.15. A test requires CI to cover the fresh-env version.
- Tooling: `tools/update.sh` updates only the environment it names. The
  target is the active venv, or the conda env in `CONDA_ENV` (which
  `make update` now passes through), and a different or missing active env
  stops the run before any change; Python and pip calls go to that env's
  interpreter. It now runs every check that needs no pull or build (target
  env, malformed refs, missing tracked files, MLX build requirements) and
  prints its planned changes before the first change, and a failure lists the
  steps that had completed and the one that failed.
- Tooling: `conda update --all` in `update.sh` decides ownership from
  `conda list --json` records, not `pip list` (which also lists
  conda-installed distributions, so ordinary conda updates were skipped as
  pip conflicts). Pip-owned or ambiguously owned packages still skip it unless
  `CONDA_UPDATE_ALL=1`.
- Tooling: running `run_commit_hygiene.sh` directly refuses, changing nothing,
  when a file it would check or re-stage (including `src/README.md` for the
  dependency sync) also has unstaged edits; it previously staged those edits.
  Commits through pre-commit are unaffected.
- Tooling: `make bootstrap-dev` installs the hooks once, fails when
  environment validation fails, and reports a skipped optional step
  (markdownlint-cli2 via npm) in its final line instead of always printing
  "Bootstrap complete".
- Tooling: the local-mlx pin file uses an explicit `mktemp` template, so the
  updater's pin helpers also work with GNU `mktemp`.
- Report publication reads results and system facts from the render context
  alone. `ReportGenerationInputs` keeps only publication settings and derives
  `results` and `system_info` from the context, so the publication step no
  longer reconciles two result collections. No report output changes.
- Harness measurements stay on harness-owned objects. The upstream
  `GenerationResult` is no longer given `time`, `active_memory`,
  `cache_memory`, `model_load_active_memory`, `model_weight_buffer`, stream
  observations or prompt diagnostics; a successful run returns them beside
  the upstream result, and they land on `PerformanceResult`,
  `RuntimeDiagnostics` and `StreamObservations`. Isolation serialises the
  upstream result's declared fields only, dropping the `__dict__` and `dir()`
  fallbacks, and reports no longer fall back to memory attributes on the
  generation object.
- Run-start facts are captured once in `RunStartFacts` (library versions,
  system facts, preflight issues, whether `environment.log` was written),
  returned by `setup_environment` and passed to finalisation, instead of being
  stored as private attributes on the CLI namespace. Reporting never probes
  the system: a render context, standalone HTML context or version printout
  built without captured system facts shows them as unavailable instead of
  measuring them at report time.
- The baseline comparison records, for each changed text, the shared prefix
  in characters, both lengths, and whether the prompt token count changed
  (e.g. "shared prefix 216 characters; length 404 → 422; same prompt
  token count"; a count, not a claim the token ids matched); one answer
  extending the other reads as the shorter length shared. One calculation decides which texts changed and describes them.
  Kept in `results.jsonl` as `comparison.text_divergence`; counted in
  characters, since the record has no per-token text.
- Checkpoint sampling follows mlx-vlm's own `load()`: only the
  `generation_config.json` keys in `mlx_vlm.utils.GENERATION_CONFIG_DEFAULT_KEYS`
  (temperature, top_p, top_k, with do_sample) are applied. A declared
  `min_p` or `repetition_penalty` is no longer applied (7 of 51 cached
  models declared a repetition penalty), so those models' text can change
  once against older baselines. A missing key list is reported as API drift.
- New recorded facts (no verdicts):
  - per model: `generation_config.json` settings nothing applied (e.g. a
    declared `max_new_tokens` of 16384), EOS ids by declaring file, which
    files carry a chat template, the variables the template reads (via
    `jinja2.meta`) and any passed chat-template option it never reads, and
    the loaded weights' own size (`mx.get_array_buffer_size`) beside
    post-load active memory;
  - per run: MLX_/MTL_ environment variables (a difference withholds the
    throughput comparison and splits history bands), the Metal max buffer
    size and buffer-count limit, the MLX memory limit and the GPU wired
    limit.
- `tools.hub_precheck` reads the chat template in AutoProcessor's file
  order (`.jinja`, `chat_template.json`, `processor_config.json`,
  `tokenizer_config.json`) and names the file; it had never read
  `chat_template.json`, so Idefics3 looked text-only and pixtral
  template-less. It also records the hub `pipeline_tag`, and a config
  `model_file` (checkpoint-shipped code mlx-vlm imports instead of its own
  packages) is reported instead of failing the architecture check, here and
  in the cached-repo check.
- The native repro command passes `--verbose`, so upstream prints token
  counts, tok/s and peak memory to compare with the harness's.
- Dependencies: `mlx>=0.32.3` (for `mx.get_array_buffer_size`) and a
  declared `Jinja2>=3.1.0`.

- Console log simplifications. Each per-model result opens with one readable
  line (`[n/N] <model>: completed, unusable output; observations: …; next
  step: …`) instead of wrapped `key=value` pairs; the DEBUG `REPRO` line
  stays the grep-able record. The verbose "Output Observations" repeat of
  the ⚠️ warnings is gone, with the helpers that built it. Metric trees pad
  labels to their own longest label, so values line up, and the first-token
  rows read "First token (upstream)" and "First token (measured)". A verbose
  run's closing summary points to the version and system tables at the top
  of the log instead of printing them again. The EXIF table keeps empty
  tags and shows undecoded binary values (PrintImageMatching, maker notes)
  in hex in verbose mode (first 256 bytes, with the total size) and as
  `[binary data, N bytes]` otherwise.
- The console comparison table's two-letter `E/U` column is replaced by a
  one-word `Result` (usable, caveats, unusable, crashed, no verdict): the
  old codes reused C and X with different meanings in each position, so
  "completed, unusable output" (`C/X`) read as a crash. The recommended
  working set is shown in decimal GB there and in the system facts, like
  peak memory (it had read 108 GB against a 115 GB figure elsewhere).
- The `--system-telemetry` help now lists everything it records (thermal
  state, memory pressure, CPU speed limit, power source and energy mode) and
  says when the default checks are taken (before load and after cleanup).
  The per-model telemetry note says which readings a count comes from (both
  checks, N of M checks, or readings every 2 s during the run) and what a
  thermal level means for that model's timings; a warm but unthrottled
  "fair" state is logged as information rather than a ⚠️ warning. The note
  (`System state for <model>`) is now logged in the model's closing lines,
  after its output and observations, instead of just before them, where it
  read as a preamble to the next model; it states that a throttled model's
  output is still recorded and assessed, and only its speed is left out.
- System telemetry records macOS's thermal state (`NSProcessInfo.thermalState`,
  read through the Objective-C runtime with `ctypes`: no dependency, no
  subprocess, no privileges) with every probe: `thermal_samples`,
  `thermal_state_max`, `thermal_elevated_samples`, plus `thermal_state_max`
  per model in the history. The existing `pmset -g therm` CPU speed limit
  reads unthrottled on Apple Silicon even under GPU heat. From "serious" up,
  where macOS reduces performance, a model's timings are left out of noise
  bands and throughput ratios and named in the comparison, like a sleep gap;
  "fair" is recorded and noted but still compared.
- Crash issue drafts and the run summary's crash sections open with "Before
  filing: rule out the checkpoint", listing only the checks the run's own
  evidence calls for: no loader for the model type (a support request, not a
  defect), processor files missing from the snapshot, a family README to
  read for a corrected checkpoint, and a load-phase failure to retry with
  another conversion. The `upstream-mlx-vlm-issues` skill routes checkpoint
  faults to the model's Hugging Face repo rather than mlx-vlm, so
  maintainers are not sent checkpoint misconfigurations as bugs.
- The `hf-cache-mlx-vlm-models` skill records what the sweeps have shown
  about checkpoints: the signals of a misconfigured checkpoint and the check
  that catches each (missing processor config, flattened weight keys,
  text-only templates, unloadable config, no loader package, weights beyond
  unified memory, incomplete shards), that a family README can name a
  corrected repo, that runnable-but-unusable output is a finding to report
  rather than a reason to prune, and which timing effects are environmental.
- The coding conventions in `.github/copilot-instructions.md` now require
  the bounded, non-link-following I/O helpers for every file read and write
  (Skylos SKY-D324/SKY-D325 recur otherwise), including installed package
  sources.
- The architecture pre-check counts an installed mlx-vlm model package only
  when its `__init__.py` binds `Model` and its config class (`ModelConfig`
  or `ModelArgs`), read statically without importing, after Nativ's
  capability-manifest generator; a same-named folder is no longer enough.
  Image-generation, detection and shared-submodule packages drop out. No
  cached model changes verdict.
- `tools.hub_precheck` compares a candidate's safetensors weights with this
  Mac's unified memory (blocked when larger) and Metal's recommended working
  set (a warning when larger), using the harness's own memory facts.
- Issue drafts and `diagnostics.md` state whether the installed mlx-vlm
  ships a README for the model's family ("Family README in installed
  mlx-vlm"), read without importing mlx-vlm; the README text is not parsed.
  The `upstream-mlx-vlm-issues` skill now reads that README before filing,
  tries any checkpoint it recommends, and records it in the issue's Model
  section.
- The `native-mlx-vlm-repro` skill reads the model family's upstream README
  first (it may name a corrected checkpoint to try before filing), states
  that the native CLI ignores `generation_config.json` so repro commands must
  carry sampling explicitly, and registers `--eos-tokens` in its Python
  `stream_generate` example as `generate()` and the harness do.
- Agent instructions now tell Claude Code to invoke the repository skills
  with the Skill tool rather than read their `SKILL.md` files, so their use
  is visible to `/skill-doctor`; other agents still read the files.
- `tools/update.sh` stops with diagnostics instead of working around
  trouble in the local MLX checkouts. `git pull` is now `--ff-only`, and a
  pull that cannot complete prints the branch state and both sides of any
  divergence, then exits; it used to skip that repo and continue, leaving a
  local mlx unpinned. The separate `requirements.txt` pass is gone: mlx has
  none, mlx-vlm's is its own dependency list, and the pass ran before the
  local mlx pin, so a newer PyPI mlx could replace the local build and force
  a rebuild. After the final project reinstall, every local install verified
  earlier is re-checked, and the run stops if one was replaced.
- The baseline comparison records facts about the runs even when their
  outputs cannot be compared (a new image or prompt): which models were added
  or are absent, and whether an absent model is still cached but not selected
  or no longer in the cache, plus model revisions and upstream commit ranges.
  Only output, quality and performance comparisons are withheld; the special
  incomparable branches in the serialiser, loader and renderers are gone.
- Crash continuity per model from failure signatures, which are now kept in
  the retained failure record (`failure.signature`): same or different
  signature observed, newly observed failure, completed this run after
  failing in the baseline, or insufficient evidence. A note flags signatures
  from different harness versions, since a renamed phase changes a signature
  without changing the fault. Nothing is labelled "resolved".
- Text changes are split by the effective decoding of both runs: greedy,
  sampled with the same settings and seed, mode or settings changed, or not
  recorded.
- For each model whose outcome, text or failure moved, the comparison lists
  the upstream mlx-vlm commits between the runs that touched its resolved
  architecture package (local path-limited `git log`), distinguishing "no
  commits touched it" from "history unavailable". Labelled as context, not
  attribution.
- Repository-scoped Codex configuration disables the overlapping Superpowers
  workflow. Agent guidance selects the repository's HF CLI reference over
  duplicate personal/plugin copies, preserving specialised skills and personal
  defaults; the current Codex per-skill project-scope limitation is documented.
- Agent guidance explicitly authorizes the safe local edit–test–fix loop for
  implementation requests while retaining authorization boundaries around
  downloads, dependency changes, model/data caches, expensive inference,
  commits, and publishing.
- Each model's run logs one short line as each slow stretch before the first
  token begins ("Loading model weights and processor...", "Rendering the
  prompt...", "Preparing image inputs and prefilling..."), so a long pause
  reads as work rather than a hang. The checkpoint-defaults line now reads
  "Token decoding settings for ..." instead of "Sampling for ...".
- Skills follow Anthropic's current authoring guidance: each description says
  what the skill does and then when to use it, in the third person, and no
  longer carries rules (the conda-not-uv and don't-file-unasked rules already
  stand in the skill bodies, and descriptions load into every session). The
  cache skill's description now says what it does, not only when. Shouted
  rules ("Things to NEVER do", bold "Never" and "Do not") are restated
  plainly with their reasons, which the latest models follow without
  over-applying.
- Refreshed repository-local agent guidance for Codex/GPT-6 Astra: clearer
  outcomes, task scope, evidence, and validation reporting with less duplicated
  procedure. The cache skill now points to maintained rules instead of copying
  feature internals; the typing skill retains its Transformers-derived workflow
  and examples with small repository-specific clarifications.
- `--help` and `--dry-run` skip the dependency import probe, which costs a
  second full `mlx_vlm` import: `--help` drops from 5.7 s to 2.2 s. The probe
  shields a sweep in progress from a hard-crashing import; neither command
  invokes a model. Real runs are unchanged.
- Dry-run output says things once: the truncated `Final prompt` preview is
  omitted (the whole prompt follows), the image block is gone (the run header
  already names the image and its dimensions, and small files read "0.00 MB"),
  and the model list marks unknown image capability without repeating the
  evidence printed in the warning above it.
- `.github/copilot-instructions.md` line counts refreshed.
- Hugging Face CLI references use `hf` (0.17.36): the "no models found" error
  and the README troubleshooting example named the removed `huggingface-cli`
  (and a non-MLX model). The `hf-cache-mlx-vlm-models` skill now says how to
  remove a cached model under huggingface_hub 1.32's shared blob store
  (`hf cache rm` then `hf cache prune`, never `rm -rf models--…`).
- Isolated workers receive the parameters the parent resolved, not the CLI
  namespace: the spec is `ProcessImageParams` plus the quality-config path,
  and the child no longer re-interprets CLI state. The JSON round trip now
  restores `frozenset`, integer-keyed dicts (`logit_bias`), PEP 695 aliases
  and `Path` versus URL image inputs, and the params builder converts
  argparse's `nargs` lists to the tuples the dataclass declares
  (`resize_shape`, `eos_tokens`), which the round trip exposed.
- The run issue summary is a planned `ReportArtifact` like every other
  output, still scheduled explicitly. Its separate manifest, log, dashboard
  and finalization handling is gone; all four now follow its outcome. The
  artifact plan is no longer rebuilt after the comparison is computed, since
  no job reads it.
- Less repeated data plumbing in the monolith, no behaviour change intended:
  one generation call instead of two branches and a passthrough adapter
  (one deliberate behaviour change: `--processor-kwargs` keys named
  `on_first_token` or `observations` used to replace the harness's values
  silently, because the old path bound them with `functools.partial`; every
  key naming one of the guarded call's own arguments is now rejected at
  argument validation, with the names taken from the call's signature. A key
  such as `model` already failed, as a duplicate-keyword `TypeError` at
  generation time; it now fails earlier and more clearly); finalization
  updates the
  existing runtime diagnostics with cleanup facts via `dataclasses.replace`
  instead of copying five fields by hand, so later fields cannot be dropped;
  cache discovery selects the `main` revision once; and the current run's
  comparison inputs come from the same retained-run projection as the
  baseline's rather than from a second reading of live inputs.
- The 4,838-line "RUN COMPARISON" section is split by accurate banners into
  run comparison, retained-run validation, run issue summary, output index,
  native reproduction commands and issue drafts, report publication, and
  triage reruns/finalization; the Copilot section map lists all seven.
- `tools/update.sh` only warns when the Metal compiler differs from the one
  recorded for the local mlx build; `MLX_REBUILD_ON_TOOLCHAIN_CHANGE=1` opts
  into the rebuild. Upstream `main` may not compile under a brand-new Xcode,
  and a failed build would replace the working local build with a PyPI wheel.
- `tools/update.sh` rebuilds the local mlx checkout when the Metal compiler
  differs from the one that built the current install, even if upstream is
  unchanged and the checkout is clean: the metallib is a product of the
  toolchain too, so an Xcode upgrade previously left old kernels in use. The
  compiler version is recorded under the checkout's `.git/` after each
  successful build (invisible to `git status`); a build with no record
  rebuilds once to create it.
- More recorded facts for evaluating a sweep, none of them judgments: the
  comparison names the models whose text changed, gives the prefill tok/s
  ratio beside the decode one, and lists the upstream commits between the
  two runs for each editable checkout (`git log A..B`, local only); the
  header carries `phase_totals_s`; and provenance records the revision the
  binary was *built* from (`built_revision`), labelling runs with it and
  saying so when an editable checkout is ahead of its compiled build.
- Two new mechanical observations, both caveats rather than failures.
  `prompt_hint_echoed`: the Description reproduces the prompt's own
  "Description hint" (share of its words inside four-word runs shared with
  the hint at or above `hint_echo_min_coverage`, 0.8). On the retained swan
  sweep six models handed the hint back and passed every structural check.
  `unverified_place_name`: the prose names a place none of whose words the
  prompt supplied (a capitalised name beside a natural or landmark feature
  word, or after a locative preposition; no gazetteer). It found "River
  Cam" and "River Thames, Lambeth" for a photograph taken at Ely, and
  nothing on the Bath or cats sweeps.
- Less hand-copying between representations of the same facts:
  `_process_image_params_from_args` copies the 35 same-named CLI values by
  dataclass field enumeration (only the 12 derived or overridable fields are
  written out); the JSONL metrics and the history row share one
  `GenerationFactsRecord` declaration and one `_generation_facts_record`
  projection, so history rows now also carry `prompt_tps`; the observation
  detail labels are a module constant pinned against the TypedDict; and the
  isolated-worker round-trip test compares every parameter field with
  non-default flags instead of seven.
- Removed two leftovers: the telemetry record builder's legacy
  `(cpu, pressure)` pair normaliser, which only tests exercised (they build
  `_TelemetryProbe` now), and the output index's provisional job, which the
  orchestrator always replaced before anything ran it.
- The run comparison notes a macOS version change between runs. The first
  sweep after the 26 to 27 upgrade read prefill throughput 2-5x lower with
  identical outputs and returned to baseline on the next run: Metal
  pipeline caches were cold, not mlx-vlm slower.
- New `tools/hub_precheck.py` judges hub checkpoints before download from
  three small hub reads: server-style layout, architecture through the
  harness's own `arch_precheck_for_model_type` (factored out of the
  cached-repo precheck), and chat-template shape (`string-only` templates
  fail at prefill for vision families, as the Mistral-Small-3.2 conversion
  did). Documented in the `hf-cache-mlx-vlm-models` skill.
- The generated `hf-cli` skill (`hf skills add`) is checked in under
  `.agents/skills/hf-cli/` and listed in the Copilot skills table; the gate's
  markdownlint skips that generated file (regenerate with `hf skills add
  --force`, never hand-edit).
- System telemetry records the power source and `pmset` power mode per probe
  (`power_samples`, `on_battery_samples`, `power_mode_max`) and the run
  header records both at run start. The run comparison reports them as
  "Run environment" notes without withholding throughput: on Apple silicon
  the battery effect is unproven and confounded with cold weight caches.
- A model whose wall clock outran the process clock by more than 5 s (the
  Mac slept or was suspended mid-model; `perf_counter` stops during sleep)
  gets `wall_clock_gap_s` in its telemetry, a warning, a diagnostics note,
  and is excluded from throughput comparison. Detected even with telemetry
  off. Found when a sweep on battery slept for 11 minutes after one model
  and the run total no longer matched the per-phase timers.
- The quality gate runs one Skylos scan instead of two: the ungated `-a`
  audit pass, which re-analysed the same tree for a grade card, is gone and
  its one extra check (`--ai-defects`) rides on the gated pass. Saves about
  8 s of the 46 s gate; the danger advisory wrapper is unchanged.
- The suppression audit lints every noqa variant in one `ruff check` run and
  slices the concise output per finding, instead of one interpreter start per
  suppression (56 today).
- The pytest worker cap is 10 (`-n auto --maxprocesses=10`) in the quality
  gate and the `Makefile` test targets: measured fastest on the 6P+12E-core
  laptop (18.6 s versus 20.3 s at 8), with the curve flat because each worker
  pays about 1.8 s to import the monolith before its first test.
- `README.md` explains greedy decoding versus sampling next to the temperature
  flags: what temperature 0 means, when `--seed` matters, and why neither mode
  is simply better.
- `--seed` defaults to 0 instead of none: with sampling settings now taken from
  each checkpoint, a fixed seed keeps sampled outputs reproducible from run to
  run on a given MLX version. Native repro commands carry `--seed 0`.
- Sampling settings left unset on the command line (`--temperature`,
  `--top-p`, `--top-k`, `--min-p`, `--repetition-penalty`) now take the
  values each checkpoint declares in its `generation_config.json`; a given
  flag wins for every model, a declared `do_sample: false` keeps greedy
  decoding, and out-of-range declared values are ignored. Triage reruns keep
  pinning temperature 0. Every result records where each effective setting
  came from (`sampling_sources`: cli / generation_config / default), the
  console says which values a model took from its checkpoint, the
  diagnostics carry a "Sampling settings source" row, and the run summary
  counts models that sampled with checkpoint settings. Per-model settings
  already feed history matching, so models whose settings changed fall back
  to the default noise band until new history accumulates.
- Four measurements replace estimates, all read from the stream the harness
  already consumes: the time to first token is now measured wall clock from
  the generate call to the first chunk (`time_to_first_token_s`; the chooser's
  "Prefill/first" column prefers it and falls back to upstream's model-loop
  proxy); peak memory is recorded on the first chunk
  (`first_token_peak_memory_gb`) so the prefill peak is separable from the
  decode peak; special tokens the stream emits are detected by token id
  (EOS and configured stops excluded) and folded into the existing
  control-token observation; and the prompt's image-token count is exact,
  from a second upstream `prepare_inputs` pass (about half a second on a
  50 MP image, inside the prompt-prep phase) that is trusted only when it
  yields exactly the prompt total generation reports.
- Each result records the checkpoint's own sampling defaults from
  `generation_config.json` (`declared_sampling`: do_sample, temperature,
  top_p, top_k, min_p, repetition_penalty) and the diagnostics say when the
  run's temperature differed. Facts only; the harness keeps its fixed
  settings.
- Reorganised the implementation guide into a task-oriented reference with a
  code/test navigation map and concise generation, assessment and evidence
  contracts. Removed duplicated setup/dependency recipes, stale examples and
  contradictory artifact, linting and suppression advice.
- Compression: the four usability statuses have one canonical order and one
  gloss map in the monolith (five parallel tables, sort keys and label copies
  are gone); `CLAUDE.md` is a symlink to `AGENTS.md` instead of a hand-mirrored
  copy; the dependency-sync test uses `update_readme_deps.parse_pyproject`
  instead of its own copy of the tool's parser; and `update.sh` variables are
  documented once, in the README table, with the contributor guide pointing at
  it rather than repeating rows.
- `.claude/skills` is a committed symlink to `.agents/skills` (transformers'
  `make claude` pattern) so Claude Code discovers the project skills
  natively from the one source; the skill-validation test checks the link,
  and the type-checking skill now names its transformers origin.
- `make clean` no longer deletes the untracked `output/results.history.jsonl`;
  the append-only cross-run history is the one output that must survive a
  clean, and it now has its own backup path in `tools/update.sh`.
- Cruft: the superseded cats.jpg preview asset (`source-image-c4aeb8d1…jpg`,
  referenced by no current report) is removed; the root `clean` target no
  longer names the long-gone `run.json`; two MLX-LM mentions left in the
  contributor guide and CLI README are gone.
- `tools/update.sh` starts by copying a dated snapshot of the untracked
  `output/results.history.jsonl` to `HISTORY_BACKUP_DIR` (default
  `../../backups` beside the local MLX repos; `SKIP_HISTORY_BACKUP=1` skips),
  skipping the copy when the newest snapshot is byte-identical, and pruning
  older snapshots that are byte-prefixes of the new one (the history is
  append-only, so they carry nothing it lacks; a snapshot that diverged is
  kept). Every copy and every deletion is printed. The history is what makes
  like-for-like comparisons across weeks possible and iCloud sync is not a
  backup.
- `results.jsonl` rows retain less that is already retained: for a completed
  model whose captured STDOUT is the verbose passthrough of `generated_text`,
  only the STDERR half of the capture is kept (transcripts that
  `clean_output` later reduced, truncated captures and failures keep the
  whole capture), and the bounded `rendered_prompt_preview` is written only
  when the full rendered prompt is absent. The file log keeps the complete
  capture regardless.
- Every summary surface (index, run summary, gallery, diagnostics, HTML)
  states beside the evaluation lane whether the prompt exposed the image's
  own description and keyword hints, since a clean catalogue answer proves
  less when its fields may be copied from the prompt.
- The run summary's quality table explains that a major-concerns row whose
  observations are format failures only is a chooser verdict and is not
  repeated under attempts requiring review, which lists only results whose
  observations may point at mlx-vlm.
- Gallery and HTML evidence blocks drop padding: image-side processor facts
  appear only when captured, the always-negligible input-validation time is
  gone, memory rows carry their unit, and the diagnostics clean-completion
  row no longer appends post-cleanup residues to the peak.
- The output index labels `results.html` as a self-contained page to
  download, because GitHub renders a linked HTML blob as source.
- The local-MLX runtime smoke in `tools/update.sh` defaults to
  `mlx-community/nanoLLaVA-1.5-4bit` (0.6 GB, 4-bit, so the quantised Metal
  kernels are exercised too) instead of the no-longer-cached
  `MiniCPM-V-4.6-8bit`; measured twice, it answers the smoke prompt with the
  same expected substring, so `MLX_LOCAL_BUILD_SMOKE_EXPECTED` is unchanged.
- The file log is now a self-contained timeline. The `System: … GPU: …`
  line is logged once per run instead of once per model; the per-model
  word-count line is replaced by one grep-able `REPRO model=<id>
  revision=<sha> execution=<state> phase=<phase> stop=<reason>
  observations=<codes> kwargs=<effective generate kwargs>` line at DEBUG,
  emitted for completed and failed attempts alike; and every continuation
  line of a captured mlx-vlm console block is prefixed with `[<model id>]`,
  so `grep '<model id>' check_models.log` returns that model's whole block.
- New `upstream-mlx-vlm-contributing` skill, adapted from upstream mlx-vlm's
  `contributing` skill for conda + pip and for this project's route (a
  confirmed harness finding becomes a small upstream fix from the editable
  checkout): where processor/model/test code goes, focused upstream tests,
  reproducing upstream's black/isort/autoflake hooks with isolated ruff
  rather than installing them, PR expectations. The upstream-only
  `add-new-model`, `convert-quantize` and `server-inference` skills and the
  `mlx-vlm-skills` plugin are referenced from it rather than replicated. The
  four existing adaptations now name their upstream source files (added by
  #1747, not #1343), the cache-layout skill follows upstream #2076 (the filter
  is the server's opt-in `--model-discovery hf-cache` mode), and a test
  adapted from upstream's `validate_skills.py` checks every skill's
  frontmatter, table listing, cited sources and absence of `uv` commands.
- The gallery's "Output at a Glance" preview is field-aware when the
  catalogue fields were detected: the title, the start of the description,
  and the first keywords with their count, within the same 280-character
  budget, so the keywords (usually the weakest field) are no longer hidden
  behind a long description. Undetected fields say so; outputs without
  labelled fields keep the head preview.
- Mechanical-check wording is consistent across surfaces: the output index,
  the diagnostics counts and tables, and the HTML filter now say "no
  concerns detected / concerns detected / major concerns / not assessed"
  instead of exposing the machine usability codes.
- A final answer emitted twice, verbatim after whitespace collapsing with
  at most a short separator between the copies (Step's `</think>`), is a
  new `final_answer_duplicated` observation ("Final answer emitted twice,
  around </think>") with the separator retained as evidence; strict parsing
  still reads the first copy.
- Generation failures are no longer all labelled "decode". A phase already
  tagged deeper in the call wins; otherwise the streaming boundary decides
  between `generation_before_first_token` ("generation, before first token")
  and `generation_after_first_token`. Before-first-token is reported as
  exactly that, never assumed to be prefill: input preparation, cache set-up
  and the first decode step all happen there. Output finalisation (echo,
  `clean_output`, result assembly) is covered by the same boundary, so a
  failure there keeps the boundary it reached. The `--isolate` child reports
  the same boundary for native crashes, and the run summary table names the
  phase with its human label ("crashed during generation, before first
  token") rather than the machine identifier.
- The run summary separates *Reached token limit* (stop reason
  `max_tokens`, neutral on its own) from *Incomplete output at token limit*
  (the `token_cap_truncation` observation with degradation evidence), so a
  model that supplies a usable answer exactly at the cap keeps its neutral
  classification.
- Missing-field wording is now "Required labelled fields not detected:
  keywords" (console: "Labelled fields not detected"). The parsing stays
  strict for automated consumers; the label no longer implies the content
  itself is absent, which the visible answer lets a human judge.
- Out-of-memory crashes get a "Memory capacity context (informational)"
  block beside the exception in diagnostics and issue drafts: checkpoint
  weights on disk and quantization, machine RAM and recommended working set,
  input image dimensions, the available prompt-token evidence, and the
  upstream memory estimate when mlx-vlm printed one, followed by a note that
  an OOM is a resource-capacity outcome rather than proof of a library
  defect. Checkpoint size is not used to skip models. Metal command-buffer
  wording ("Insufficient Memory", `kIOGPUCommandBufferCallbackErrorOutOfMemory`)
  now classifies as the OOM stage instead of a generic model error.
- Producer provenance inside a checkout (editable install or bare source
  tree) reports the version `src/pyproject.toml` declares alongside the Git
  revision; installed-distribution metadata, which only records the last
  `pip install`, is the fallback outside a checkout. The component provenance
  table draws on the same helper.
- The quality gate runs sequentially again: the background Skylos and
  pytest lanes, the quiet-tree guard (`tools/quiet_tree.py`), its
  iCloud-sync detection and their tests are gone. Overlapping the two long
  stages required the whole source tree to stay provably quiet, and the
  machinery that demanded grew past what a ~15 s saving justified. Kept:
  `tmp_path`-only test outputs, the temporary wheel-build copy, the
  eight-worker cap, and the per-process application caches.
- Pytest's bytecode and result caches live in the tree again: the conftest
  `sys.pycache_prefix` override and the gate's `PYTHONPYCACHEPREFIX` /
  `cache_dir` relocation existed only to keep the tree quiet under a
  concurrent Skylos scan, and splitting the result cache meant a shell
  `pytest --lf` after a gate failure did not see the gate's last-failed set.
  The Pyrefly wrapper keeps writing its throwaway config under `$TMPDIR`, and
  the wheel test keeps building from a copy, because those reasons survive
  (an aborted run must leave nothing behind; a stale `*.egg-info` changes
  package metadata). The project instructions now distinguish generated
  output, which never lands under `src/`, from ordinary gitignored tool
  caches, which may.
- Last quiet-tree leftovers: the gate test no longer pins the absence of the
  removed lane and cache-relocation strings, an unused output-directory
  constant left the version/env test, and the docs stop offering a
  gitignored `src/output/test_*` path as a place for *pytest* output (it
  remains the convention for manual validation runs, and `clean_builds.sh`
  now removes those `test_*` directories as well as files). Four stale
  `src/output/test_*` directories written by the last pre-`tmp_path` gate
  were deleted from the working tree.
- The pre-durability `reports/assets/source-image.jpg` is left frozen rather
  than deleted: retained diagnostics and pasted issues may still reference
  it, and it is simply never overwritten again.
- The retained run summary was regenerated with the fixed generator so its
  opening paragraph no longer claims range checks.
- Review follow-ups since 0.17.0: the run summary's opening paragraph no
  longer claims the output was checked "within the ranges the prompt
  states" (0.17 stopped enforcing them) and defers to the selected
  assessment profile's stated scope; the gallery preview asset is now named
  by the digest of its bytes (`source-image-<sha256 prefix>.jpg`) and
  retained across sweeps, so a reproduction command pasted into an issue
  keeps verifying after the input photograph changes, and the report calls
  it a "retained preview" that resolves once the run's artifacts are
  committed; the incomplete-cache warning for explicit `--models` states
  what the layout check actually knows (the cached *main* revision is
  absent or incomplete), says the run *may* need to download for the
  revision it will load, and its pre-fetch advice honours `--revision`; the
  obsolete `cleanup_test_outputs` fixture that deleted `test_*` files under
  `src/output/` after every test is gone; and the quiet-tree guard moved to
  `tools/quiet_tree.py`, now checks the package root itself so a file
  created and deleted directly under `src/` is caught, and is unit-tested.
  One caveat found while proving it: iCloud Drive's Desktop & Documents sync
  rewrites directory modification times a moment after any change, so on a
  tree under `~/Documents` a directory-only finding (no surviving file
  change) is reported as a warning naming the sync rather than failing the
  run; file-level findings always fail. Every pytest run (not only the gate)
  now redirects bytecode out of the tree, which removes the last in-tree
  write that triggered those touches.
  A directory bump explained solely by an ignored tool cache appearing for
  the first time (`.skylos` on a fresh CI checkout, created by the
  concurrent Skylos lane) is not a finding; a bump with no surviving new
  entry — a create-then-delete transient — still is.
- Markdown key/value rows that hold a bare URL (the new published-preview
  row, and the public-source row) now render as autolinks: the escaper used
  to wrap the URL in angle brackets and then HTML-escape them, which left a
  bare URL for markdownlint (MD034) and could mangle underscores in the
  path. Regenerated artifacts are lint-clean again.
- An explicitly requested `--models` entry whose cached snapshot is
  incomplete (a config or index present, weights still downloading) is now
  announced up front, in the run and in `--dry-run`, with the layout reasons
  and a note that the missing files will be fetched from the Hub during the
  run, bounded by `--timeout`. Default discovery already skipped such repos;
  explicit selection bypasses that check and previously said nothing.
- Cache discovery now flags, rather than runs and crashes, two kinds of
  cached repo that cannot serve the image-description task: speculative
  draft models (recognised from the installed mlx-vlm's
  `DRAFTER_KIND_BY_MODEL_TYPE` table and drafter packages — 19 types today —
  a `dflash_config.projector_type` of `dspark`, or a DSpark/DFlash/EAGLE-3
  architecture name) and image-producing families whose upstream loader
  declares `is_image_generation_model` or `is_image_edit_model` (bonsai,
  ernie_image, flux2, ideogram4, mage_flow, z_image today). Both lists are
  parsed from mlx-vlm's source without importing it, like the existing
  `MODEL_REMAPPING` alias table, so they track upstream automatically; the
  skip reason names the evidence. Adapted from Nativ's draft-model discovery
  and capability manifest.
- The test suite no longer writes anywhere under `src/` while the gate's
  Skylos lane scans it, closing the last race between the two lanes: the
  four CLI tests that wrote to gitignored `src/output/test_*` directories
  now use `tmp_path`, the wheel-build test builds from a copy of the package
  so setuptools's `build/` and `*.egg-info` never land in the tree, and the
  Pyrefly wrapper writes its generated config and output under `$TMPDIR`
  (with project includes, excludes and search path anchored to the package
  root). `conftest.py` now fails the session if any path under the package
  changed during the run, so the guarantee is enforced rather than assumed.
- The quality gate runs its two long poles — the three Skylos scans and the
  pytest suite — as background lanes while the quick static checks stream in
  the foreground, then prints each lane whole in a fixed order, so the log
  reads as before and the wall time drops from about 44 s to roughly the
  longest lane. A failing foreground step kills the lanes; a failing lane is
  reported after both have printed. Fast (push-hook) mode uses the same
  lanes with its smaller test set. The lanes fork only after the
  tree-writing static checks finish, and the pytest lane keeps its bytecode
  and result cache under `$TMPDIR`, because Skylos's dead-code grep
  verification aborts with `SKY-ANALYSIS-INCOMPLETE` when files appear or
  vanish under it (seen on a 3-core CI runner, never locally).
- The test suite runs in about half the wall time (roughly 35 s to 17 s on
  an 18-core M5 Max) with no test removed: `CHECK_MODELS_SKIP_IMPORT_PROBE=1`
  (new, documented) lets the suite skip the subprocess import probes that
  cost every xdist worker ~2 s of start-up, component provenance is memoised
  per process instead of re-running `git rev-parse` and metadata lookups for
  every report surface (a conftest fixture clears it between tests), the
  upstream CLI-parity test reads mlx-vlm's parser in-process instead of
  spawning a second interpreter, and pytest is capped at eight workers
  (`-n auto --maxprocesses=8`) because per-worker start-up made 18 workers
  slower than 6.
- A dependency import probe that merely times out is now inconclusive — a
  warning is logged and the in-process import proceeds — instead of marking
  mlx-vlm unavailable. Under load (the parallel test suite, a busy machine)
  the probe expired while the import would have succeeded, which skipped the
  four end-to-end tests on every gate run and could mark a real sweep's
  dependency missing.
- Reproduction inputs now name the committed gallery preview as a shareable,
  digest-verifiable stand-in when the original photograph is unpublished: its
  raw GitHub URL, dimensions, size and SHA-256 plus a download-verify-run
  command, explicitly labelled as reproducing on the preview rather than on
  the exact inference input. Diagnostics, the HTML report and crash drafts
  all carry it.
- File digests and the report image preview are cached per (path, size,
  mtime) for the run, so the input photograph is hashed and re-encoded once
  rather than once per report surface.
- The resolved prompt configuration is logged as one line — lane, prompt
  source (built-in with or without hints, brief triage caption, or custom),
  assessment profile and effective max tokens — so a run's effective settings
  are visible without inferring them from lane defaults. The `--prompt`,
  `--assessment-profile` and `--rerun-triage` help texts now state exactly
  what each option overrides and what it leaves in place.
- Cruft pass: archived the thinking-budget evidence note (upstream #1819 closed
  as completed), removed a stale 98 MB session worktree and empty `.cursor`,
  `.worktrees` and `docs/superpowers` directories, trimmed `src/.gitignore` to
  the entries the root ignore file does not cover, and refreshed the README
  analysis sample, "clean completion" wording and test-suite size figures.
  Everything shipped since 0.15.0 (0.16.0–0.17.0) is now recorded under the
  `[0.17.0]` heading below.

### Removed

- `update.sh` local-MLX stage (0.17.112, step 2 of john's 2026-11-11 plan):
  three workarounds are gone; `update.sh` is 2,000 lines, from 2,201.
  - The mlx-audio pin relaxer (`mlx_audio_pin_assessment`,
    `reconcile_mlx_audio_with_local_mlx` and `tools/relax_exact_mlx_pin.py`).
    It installed mlx-audio with `--no-deps` and rewrote its installed
    metadata, because 0.6.2-0.6.4 pinned `mlx==0.32.4`, which a local
    `0.32.5.devN` build cannot satisfy. mlx-audio 0.6.5 declares
    `mlx>=0.32.4`, so pip resolves mlx-vlm's requirements under the local
    mlx pin unaided. Should a later mlx-audio pin mlx exactly again, the
    mlx-vlm install fails with pip's `ResolutionImpossible` and the run
    stops; nothing rewrites installed metadata any more.
  - The PyPI fallback after a failed build. It ran only when `pip show`
    failed after the failed install, asked PyPI for the local build's dev
    version (which PyPI never has), then installed the newest release, and
    the run stopped anyway. Its premise, that pip uninstalls before
    building, is wrong: pip builds the editable wheel first, so a failed
    build leaves the previous install in place (checked with a package whose
    build raises: exit 1, the old version still installed).
  - `REPO_SKIP`. It was set only on the failure that already stops the run,
    so every loop that consulted it saw zeros; a local `install_failed` flag
    carries the failure past the `CMAKE_ARGS` restore to the stop.

  Behaviour preserved: mlx builds before mlx-vlm; mlx-vlm installs under the
  local mlx pin; an unchanged, clean, verified checkout skips its rebuild and
  mlx is still pinned; a failed build or install stops the run with a
  nonzero status after the caller's `CMAKE_ARGS` is put back, pinning a
  surviving local mlx; the editable-origin and build-match checks are
  untouched. Benefit: the stage no longer edits another package's installed
  metadata or installs a PyPI wheel where a local build was asked for, and
  its failure path is one flag. Evidence: six tests run the shipped
  `update_local_mlx_repos` against fake `git` and `pip`, checking exit
  status, pip arguments, `CMAKE_ARGS` and the pin file; run against the old
  script they show the fallback's `pip install mlx==<dev version>` and the
  relaxer's dry-run probe. Resolution was checked in three fresh
  environments holding an editable stand-in `mlx 0.32.5.dev20261110+abc123456`
  under the same pin (Linux, Python 3.13): empty, mlx-audio 0.6.4 as
  published (`mlx==0.32.4`), and mlx-audio 0.6.4 with its metadata relaxed by
  the removed tool (`mlx>=0.32.4`, the state of john's environment). In each,
  the stage's own install command on mlx-vlm main (0.8.0) exited 0, moved
  mlx-audio to 0.6.5 with its published metadata, left the stand-in mlx
  editable at the pinned version, and `pip check` was clean; a dry run of the
  eager project reinstall under the pin then planned no mlx, mlx-vlm or
  mlx-audio install. Not checked here: macOS markers and a real compiled
  mlx, which a full `update.sh` run on the Mac covers.
  The relaxer's tests and the source-text checks on the failure branch are
  replaced by those behavioural tests.

- Dead code (0.17.109): `format_field_label` and `log_generated_text` had no
  callers in the script, the tools or the tests and are removed, along with
  their `__all__` entries (which had hidden them from vulture).

- Cruft: `src/tools/retire_old_models.sh` is removed (john's 2026-10-25
  review). Its dated retirement list has been applied: none of its twelve
  targets is in the 2026-10-25 sweep's cache. Git history keeps it.
- `ResultSet` (0.17.81) is no longer part of `check_models`'s public names.
  Production read only its sorted `.results`; `.successful`, `.failed` and
  the unused `_fields` slot served tests alone. `ReportRenderContext` and
  `CanonicalHtmlReportContext` now hold the time-sorted results directly as
  a `results` tuple, so `context.result_set.results` becomes
  `context.results`. A script that imports `ResultSet` from `check_models`
  must now sort its own results. Report order is unchanged.

### Fixed

- Value checks (0.17.122): 23 more readers of JSON and runtime values
  tested `isinstance(value, int)` or `int | float` without excluding
  booleans, so a `true` was read as 1: one parameter in a `config.json`,
  1-bit quantization, a 1 s first-token time, 1 tok/s in a comparison or a
  history band, a 1 GB peak, a run "on battery". They now call `_is_int` /
  `_is_number`, the predicates from 0.17.118. The readers: model burden
  from `config.json` (4), runtime durations, the declared-sampling fact (2),
  the repetition guard's token count, history-band samples (2), decoding
  settings, prefill seconds and ratio (3), thermal and battery telemetry
  (2), comparison throughput and peak memory (2), and the run summary's
  quality cells (5). *Behaviour changed:* only for a boolean where a
  number belongs, which is now treated as not recorded. *Evidence:* 19 new
  cases, one per check, each failing before and passing after; with the
  change in place each check was weakened back in turn and 19 of 23 were
  noticed. The other four compare the value with a threshold above 1
  (thermal state 2, 200 tokens, 16 tokens), so a boolean never passed them
  and the change there is spelling only. *Left as they are:* EXIF tag ids
  and GPS values read through Pillow (TIFF has no boolean type), and
  `_generation_int_metric`, whose input is already checked.

- System facts (0.17.121): a run without psutil completed with its CPU core
  counts, available memory and swap use missing and nothing saying why.
  psutil is an optional extra, absent from CI and from a plain install. The
  run now states it once ("psutil could not be imported (optional: pip
  install psutil); CPU core counts, available memory and swap use were not
  read and are missing from this run's system facts") among the start-up
  warnings, so it is on the console, in the log, in the `results.jsonl`
  header and in `run_summary.md`. *Unchanged:* the run still completes and
  total RAM is still read from `sysctl` or MLX. *Evidence:* the real command
  line run to completion against a fake MLX runtime, once with a fake psutil
  (the four facts recorded, no note) and once with psutil unimportable (the
  four absent, the note in all three places).

- Run summary (0.17.120): the native command printed for a failed model
  that recorded no settings of its own (a load crash, say) stated
  `--temperature 0.0` whenever the run's other models had not all used one
  temperature, and `--max-tokens 500` when no model recorded any, whatever
  the run had asked for. The retained record holds only the settings every
  model shared plus each model's own, so those values were stand-ins. The
  command now leaves out any setting the record holds for other models but
  not for this one, and a line under it names them ("The run did not record
  these settings for this model, so the command leaves them out: ..."). A
  sampled command likewise no longer states an unrecorded `--top-p` or
  `--top-k` at its default. *Unchanged:* commands for models that recorded
  their settings; the load-only command (`--prompt x --max-tokens 8
  --temperature 0.0`), which claims nothing about the run; and the
  commands in `diagnostics.md` and the issue drafts, which are written from
  the run's own settings. *Evidence:* the real finalisation run from a
  command line asking for temperature 0.25, with one model that sampled at
  its checkpoint's 1.0 and one load crash: the summary gave the load crash
  `--temperature 0.0` before and no temperature now; a second run where no
  model recorded settings; three tests, all failing on the previous code.

- Start-up (0.17.119): an mlx or mlx-vlm import that failed in this process
  was always reported as "not found, please install it", whatever went
  wrong. A module mlx-vlm needs being absent, or mlx-vlm having moved a name
  this script imports, read the same as an uninstalled package, beside an
  environment table listing the package as installed. The message now says
  "not found" only when Python reports the package itself absent; otherwise
  it says the package was found but could not be imported and gives the
  error. The import probe passes when mlx-vlm has moved a name or one of its
  submodules fails, so an ordinary run reached the wrong message too, as
  did a dry run with any import failure. *Evidence:* five runs of the real
  command line against fake `mlx` and `mlx_vlm` packages (a moved name, an
  absent module, a library that cannot be loaded, and each package absent);
  three fail on the previous code, the two "absent" ones pass on both.
  Found by running the tool with numpy made unimportable: a dry run said
  mlx-vlm was not installed. *Unchanged:* a probed failure keeps its
  message, and the placeholder functions still raise the "not found" text.

- Quality gate (0.17.114): `quality_run_pyrefly_check` took its exit status
  from a `pyrefly | tee` pipeline, which is `tee`'s status unless the caller
  set `pipefail`. Every current caller does, so no run was affected, but the
  hidden-worktree test, calling it without, saw Pyrefly print two errors
  and the function return 0. The output now goes to a file and is printed
  afterwards, so the status is Pyrefly's own.

- `run_skylos_danger_advisory.sh` (0.17.109): the "no findings" branch ran
  `raise SystemExit("0")`, which prints `0` to stderr and exits 1, so a
  report with no danger list would have stopped the script under `set -e`
  with an empty count. It now prints `0` and exits 0. `$QUALITY_PYTHON` is
  quoted there.
- Quality gate (0.17.110): Skylos 5.5.0 (10 November) made the danger gate
  report 21 findings in tests that 5.3.0 and 5.4.0 pass, all false
  positives. SKY-D213 now flags a literal `trust_remote_code=True` on any
  callee, here `argparse.Namespace` and the project's own settings objects
  in eight fixtures that load nothing; each carries a reasoned ignore.
  SKY-D212 now also checks the script handed to `["/bin/bash", "-c", ...]`
  and counts a function's parameters as untrusted, which caught thirteen
  tests that run `update.sh` functions with pytest's own values. Those twenty call sites now share one helper,
  `_run_bash_script`, with one reasoned ignore, which also removes twenty
  repeated `noqa: S603` comments (inline suppressions fall from 134 to
  124). The floor is `skylos>=5.5.0`, in `pyproject.toml` and
  `setup_conda_env.sh`, because older releases would report the new
  ignores as unused. Gate thresholds are unchanged.
- Quality gate (0.17.109): ty 0.0.100 renamed `invalid-property-type-override`
  to `incompatible-property-override`, so the two stream-property ignores in
  `check_models.py` stopped matching and the suppression audit failed on a
  fresh install. The ignores use the new name and the floor is `ty>=0.0.100`
  (0.0.99 and older do not know it).
- Agent instructions: the section table named a `DiagnosticsConfig` class
  that does not exist; it now names `ResultAssessment`.

- Reasoning effort (0.17.108, john's 2026-11-07 review): the chat template
  is now selected and parsed once, and its variables, declared efforts and
  enforcement come from that one template (the processor's, else the
  tokenizer's; from a named set, `default`). Previously the variables were
  the union of every named template, so an effort read only by `tool_use`
  counted as applied and dropped the automatic thinking budget. An enforced
  `["none"]` allowlist is now told apart from a bare `== "none"` branch
  (both read as `none` upstream): only the enforced one restricts the
  request, so `high` maps to `none` (or is skipped) instead of reaching a
  template that raises. Diagnostics record `declared_reasoning_efforts_enforced`.
- Lint: `_human_observation_labels` combines the catalog-constraint
  `elif` and its nested `if` into one condition (Ruff SIM102).
- Run summary (0.17.101, review of 8e17405): native-reproduction matching
  now includes each model's own inputs. The experiment label adds the
  model's rendered-prompt digest and a digest of its effective generation
  arguments (`prompt_diagnostics.generate_kwargs`) to the run's image,
  prompt and settings digests, so a changed chat-template render (Muse's
  template embeds the date) or effective temperature no longer reads as
  "confirmed", and an old "not reproduced" cannot dismiss an input-dependent
  problem. When a model's inputs are not recorded, no match is claimed: the
  cell reads "not yet checked (earlier check not comparable: inputs not
  recorded)" and `add_native_repro_note` refuses to record. A crash before
  any input was prepared (import, model, processor or tokenizer load,
  preflight) keys on "model inputs not reached". The retained llm-jp note
  gains the digests from the harness record; its evidence says so and why
  (the native flags are the recorded effective arguments, the template has no
  date or time variable, and the output matched byte-for-byte).
- Output data (0.17.100, review of 8e17405): the retained llm-jp
  `native_repro_notes.yaml` note, recorded on a branch before 0.17.99 made
  `experiment` a required key, was merged without it, so the reader rejected
  the whole file and the published native confirmation would have read "not
  yet checked" on the next summary render (and `add_native_repro_note`
  refused to append). The note gains the retained run's experiment label,
  which its recorded evidence matches (same image digest, recorded prompt,
  `--max-tokens 1000 --seed 0 --prefill-step-size 2048`). A new test parses
  every committed notes file with the current strict reader, so a schema
  change must migrate retained notes in the same commit.
- update.sh tooling (0.17.100, review of 8e17405): `relax_exact_mlx_pin.py`
  rewrote the installed `METADATA` by truncating it in place, so a failure
  part-way through left an empty file. It now writes a sibling temporary
  file (`safe_io.replace_text_no_follow`: flushed, original permissions
  kept) and renames it over the original only after the write succeeds; a
  failure-path test checks the original bytes survive and no temporary file
  is left.
- update.sh (0.17.99, review of 0.17.98): the mlx-audio reconciliation runs
  inside the caller's `if !`, where `set -e` is inert, so a failed `--no-deps`
  install or metadata rewrite could be recorded as success with the hold
  still written; every step now checks its own status, and the hold is
  written only after the rewrite succeeds. Requirement interpretation moved
  into one packaging-backed assessment (PEP 440): `mlx==0.32.4` is now
  recognised as satisfied by a local `0.32.4+local` build, a wildcard pin
  (`==X.*`, which has no valid `>=` form) or a pin whose `>=` floor the local
  build cannot meet is reported as unsupported and stops the run instead of
  being rewritten, and `relax_exact_mlx_pin.py` validates the replacement
  requirement before writing it.
- Run summary (0.17.99, review of 0.17.96): a native-reproduction check now
  applies only to the experiment it reran. `native_repro_notes.yaml` entries
  gain a required `experiment` field (image, prompt and generation-settings
  digests, recorded by `tools.add_native_repro_note` from the retained run;
  no earlier file exists to migrate), and a note from a different image,
  prompt or settings shows as "not yet checked (earlier check out of date)"
  instead of confirming — or dismissing — an observation it never reran.
- Development tooling: temporarily override markdownlint's exact `re2js` 2.8.6
  dependency to address GHSA-gg6h-894w-4m9c — first with 2.8.7, then raised to
  2.9.1, the latest release (john's call, 2026-10-30). A dependency test rejects
  vulnerable resolutions and flags the override for removal when markdownlint
  changes that requirement. Retired 2026-11-12: markdownlint 0.42.1 (via
  markdownlint-cli2 0.24.1) itself pins `re2js` 2.9.1, past the 2.8.7 fix, so
  the override and its retirement record are gone; katex is the only override
  left. No advisory suppression was ever added.
- Development tooling (0.17.102): the override-retirement guard now covers
  every npm override, not just re2js. Each `package.json` override must have
  a record naming the parent requirement that necessitated it and the
  advisory's first fixed release; the test fails when that requirement
  changes (drop or re-review the override), when an override appears without
  a record, or when a record outlives its override. This closes the gap
  where the katex 0.19.0 override (GHSA-238p-pmpm-9mq7, necessitated by
  micromark-extension-math's `^0.16.0`) had no retirement watch and could
  have outlived its reason silently. The variation-selector source scan now
  excludes `node_modules` (third-party katex ships Python sources, and an
  iCloud-evicted conflict copy there timed out the read).
- Remove the obsolete braces advisory exception, which the dependency scan no
  longer reports; retain the warning for any future stale exceptions.
- Work around Skylos 5.2.0 false positives with two audited, line-local ignores:
  escaped fragment links and image URLs already scheme-checked by their caller.
  Remove each ignore when upstream recognises the proof; the suppression audit
  rejects obsolete ignores. Reword an innocuous changelog phrase misclassified
  as prompt injection. Hub prechecks now explicitly reject non-HTTP(S) URLs,
  rather than suppressing that separate URL-scheme finding.
- update.sh (0.17.98): a failed Stage 3 build/install of a local MLX repo now
  stops the run (nonzero exit, with the stop report and the end-of-run change
  table) after the existing restore attempts, instead of warning and
  continuing — this also changes the failed-*mlx*-build path, which used to
  preserve-and-continue to exit 0. On 2026-10-28 a failed editable mlx-vlm
  reinstall (ResolutionImpossible, below) was only a warning: Stage 3b never
  verified mlx-vlm, "critical packages verified" passed on the stale install,
  and the run exited 0 while the pulled checkout's `.pth` imported 0.7.11
  with 0.7.10 dist-info in site-packages (the harness's "mlx-vlm is
  installed at 2 versions" warning). The rebuild eligibility check and Stage 3b now
  also compare mlx-vlm's pip-recorded version with the checkout's declared
  `mlx_vlm/version.py`, so a leftover skew forces a rebuild on the next run
  instead of passing on the editable path alone.
- update.sh (0.17.98): mlx-vlm >= 0.7.11 requires `mlx-audio>=0.6.2`, and
  every release in that range (0.6.2, 0.6.3 — and upstream main as of
  2026-10-28) pins `mlx==0.32.4` exactly, which a pinned local mlx source
  build can never satisfy, so the editable mlx-vlm install (and the later
  eager project reinstall) ended in ResolutionImpossible. When the local mlx
  pin is active, update.sh now resolves the mlx-audio candidate pip would
  pick and, if its bare exact mlx pin conflicts, installs it with
  `--no-deps`, relaxes the pin in the installed metadata to the `>=` floor
  (new `tools/relax_exact_mlx_pin.py`), and holds mlx-audio at that version
  for the rest of the run, so `pip check` stays green and eager passes reuse
  the relaxed install. The step runs even when the mlx-vlm rebuild is
  skipped, and retires itself once upstream ships a release without the
  exact pin.
- Notes (0.17.97, john's 2026-10-27 review): the note tools parse the
  existing notes file, append to its `notes` list and write the whole
  document, keeping its leading comments, so an empty file, `notes: []` or
  an indented list no longer breaks a save. The file is replaced through a
  staged sibling inside the error handling, so a failed write leaves earlier
  notes intact. A stale note no longer says "This output has not been
  reviewed" when another note applies to the current output. `notes:` set
  to anything other than a list or null (`false`, `0`, `{}`) is rejected
  rather than read as empty.
- Notes are published without a sweep: both note tools re-render
  `issues/run_summary.md` from the retained run, which gains a *Human
  accuracy notes* section listing the notes that apply. The gallery and HTML
  report still show them from the next run, since they render from live
  results.
- Tests: the mocked `_run_model_generation` tests no longer import the real
  `mlx_vlm.utils` (and with it native MLX) through the prompt-token
  diagnostics, which aborted the suite where Metal is unavailable.
- Run comparison: the field list of each plain comparison record is
  reflected once per class rather than once per record.
- Timing (0.17.93): a model's total time ends after cleanup, the one end
  boundary for every timed phase, so load, preparation, generation and
  cleanup never add up to more than the total. It used to stop before
  cleanup: LFM2.5's retained record showed 0.809 s against phases summing to
  about 0.865 s. The post-cleanup probes (allocator sampling, telemetry,
  model burden) stay outside it. The run's phase totals are recomputed from
  the final end-to-end duration, so `per_model_total` plus
  `outside_model_loop` equals `total_runtime_seconds`; the residual used to
  come from the pre-report figure (248.422 s against 250.244 s in the
  2026-10-25 sweep).
- Run comparison (0.17.93): a model list longer than eight collapses to a
  bare count, "11 models". It used to add "(targeted run against a
  full-sweep baseline)", an explanation inferred from the list's length
  alone; the 2026-10-25 summary attached it to models no longer cached, and
  it would equally have described nine newly added models. Its commit is
  titled 0.17.92, a number the chooser refactor had already taken.
- Tests (0.17.85): `test_shared_flag_defaults_match_mlx_vlm_generate` passes
  against mlx-vlm 73d366e and later. Upstream now defaults `--temperature`,
  `--top-p` and `--top-k` to None, meaning the checkpoint's value or else a
  library fallback, which is the rule the harness applies itself. The test
  compares our defaults with the fallback upstream's
  `resolve_sampling_defaults` gives a checkpoint that declares nothing, so a
  real change in upstream's fallback still fails it. This replaces 0.17.84,
  which listed the three flags as deliberate divergences. CI missed the
  drift because it installs the PyPI mlx-vlm release (0.7.8), which
  predates 73d366e.
- Run comparison (0.17.78): a sampled record without `top_k` makes the pair
  "settings not recorded" rather than "same settings". Before 0.17.76 the
  harness left out a `top_k` of 0, and from mlx-vlm 73d366e an omitted one is
  filled from the checkpoint, so such a record does not show what ran. The
  exception is a record whose `declared_sampling` has no `top_k`: nothing
  could fill it, so it ran with 0. A setting recorded on both sides that
  differs still marks the pair "changed", and greedy pairs are unaffected.
  This replaces 0.17.76's reading of a missing `top_k` as 0.
- Native repro commands (0.17.77): a sampled run's command now states
  `--top-p` and `--top-k` even at their defaults (1.0 and 0). From mlx-vlm
  73d366e the CLI fills omitted ones from `generation_config.json`, so the
  command could sample differently from the run. Greedy commands are
  unchanged. The harness still applies checkpoint sampling itself rather
  than leaving it to mlx-vlm, since it records each setting's source,
  supports mlx-vlm releases before 73d366e, and ignores out-of-range values.
  The `native-mlx-vlm-repro` skill now describes both CLI behaviours.
- Sampling (0.17.76): `top_k` is now always passed to mlx-vlm, including 0
  (disabled). mlx-vlm 73d366e ("Honor checkpoint sampling defaults") fills an
  omitted `top_k` from the checkpoint's `generation_config.json`, so the
  harness's old habit of leaving out a 0 let a sampled run use the
  checkpoint's `top_k` while the record said 0: for example `--top-k 0` given
  on the command line, or `--temperature 0.7` on a `do_sample: false`
  checkpoint. Greedy runs were unaffected. Pillow 13.0.0 needed no change: none of its
  removed APIs are used and the test suite passes on it.
- Run summary: a failed second render no longer empties the summary. The
  summary is rendered again once the run's final timing is known, and a
  failure there is meant to keep the first render, but the Markdown writer
  truncated the file before writing, so a disk-full error left it empty while
  the error was only logged. Markdown artifacts now write to a staged sibling
  and rename it into place, as `results.jsonl` already did.
- Run summary comparison: a model whose usability moved only because a
  retired check disappeared is no longer listed as changed. With
  `unverified_place_name` gone, an unchanged answer whose baseline was
  "concerns detected" for that check alone moved to "no concerns detected"
  and still read as a model change. Such a model is now left out of the
  per-model changes, and the retired-check note counts it and its usability
  move as an assessment change, not a model change.
- Assessment: an answer that hands back the prompt's own instructions is
  reported again, as `prompt_instruction_echo`. On the 2026-10-16 blind sweep
  nanoLLaVA answered "Title: A Concrete 5-10-word Title" with the
  description requirement as its Description, and was rated usable: the
  observation code survived 0.17.0, but its detector had been removed with
  the old heuristics. It is now counted like the repeated prompt hint: a
  title or description (or an unlabelled answer) with 80% or more of its
  words in four-word runs copied from the prompt minus its camera-fact and
  hint lines. Replayed over both 2026-10-16 sweeps, it flags only nanoLLaVA
  and SmolVLM-256M, both blind. Results record the fields as
  `instruction_echo_fields`.
- Diagnostics: the prompt composition no longer reports a false exact zero.
  The 2026-10-16 sweep said Phi-3.5-vision's 1,151-token prompt held "0 image
  tokens (exact)": Phi-3-vision marks image positions with negative ids, and
  the count matched only the configured image id. Negative ids now count as
  image positions (a vocabulary has none), and a count that finds no image
  positions, now or in an older retained record, is treated as unknown, so
  the split falls back to the labelled estimate.
- Run summary comparison: an observation check that check_models has
  since removed no longer shows up as a per-model change. The 2026-10-16
  sweep listed "-unsupplied place name" for gemma-3n and MolmoPoint, whose
  text was identical, because the baseline predated the check's removal in
  0.17.48. Readers' views now leave retired checks out of the delta, drop
  models with nothing else moved, and name each retired check once with its
  removal release and how many models carried it. `results.jsonl` keeps the
  recorded observation sets, so retained comparisons read the same way.
- Run summary comparison: the changed-text table gains a column of
  generated token counts, baseline then now. Muse-Glimmer's text went from 4,677 to 66
  characters, which read like a scaffolding fix; both runs used all 1,000
  tokens on reasoning, and only this one reached its answer channel. New
  comparisons record the counts; older ones say "not recorded".
- One run duration everywhere. The run summary was rendered before the
  `results.jsonl` header gained its end-to-end duration, so the 2026-10-16
  sweep's summary said 10m 41s while the index and log said 10m 44s, and its
  finish time disagreed with `results.jsonl`. The index and the final header
  now share one end time, the summary is rendered again from the final
  header (also matching what a regeneration produces), and the closing log
  line reads the recorded duration back.
- The repetition guard's token-id check (0.17.49) is reverted. It stopped a
  loop that the BPE detokenizer was holding back, as with Qwen2-VL's 265
  repeats of `<|im_start|>assistant` on 2026-10-11, but the early stop
  discarded the held-back text, which was the evidence of the loop.
  Reconstructing that text would mean decoding outside mlx-vlm. Such a loop
  now runs to the token cap again and keeps mlx-vlm's own finalised text; a
  comment at the guard records the limit, and a test streams the case and
  checks all 265 repeats survive.
- Regenerating a run summary recomputes the current run's GPU note from the
  records instead of replaying the stored run-time string. Regenerating the
  2026-10-11 19:53 sweep had restored its stale "detected GPU activity ...
  36 of 47 models" claim; it now gives the unreliable-reading note. A
  regression test regenerates a retained run carrying the stale note.
- GPU readings from before the 0.17.47 stale-read fix are marked unreliable
  wherever they are shown, with the raw numbers kept. New telemetry records
  carry `gpu_sample_method: "reread"`, and readers treat records without it
  (0.17.44-0.17.46) as unreliable: never "GPU activity detected outside
  harness inference". The retained 2026-10-11 19:53 summary, diagnostics and
  HTML report now say its non-zero readings are unreliable and not evidence
  of other GPU work; `results.jsonl` is unchanged.
- The quality table's intro now says "tokens in count the image's tokens
  too ...; tokens out are those generated". Skylos 4.47.2 flagged the
  previous wording of that sentence as a prompt-injection pattern (SKY-D260)
  in the 2026-10-11 19:53 run summary, although earlier summaries with the
  identical sentence passed, so the detector weighs surrounding text. The
  meaning is unchanged; the retained summary was regenerated with the new
  wording.
- The snapshot GPU reading no longer reports check_models' own work as
  outside activity. The first `ioreg` read after a long gap returns a stale
  figure: in the 2026-10-11 19:53 sweep the after-cleanup check showed this
  process's finished inference at 47-99% for 36 of 47 models, while a read a
  moment later gave 0. Each sample now discards one read, waits 50 ms and
  keeps the next, and the after-cleanup check first synchronizes MLX. A trace
  of six models then read 0 at all twelve checks, while a separate
  GPU-loading process still showed 24-26%.
- `update.sh`'s end-of-run change table listed every pip-installed package
  twice, once as `pip` and again as `conda`: `conda list` ignores `--no-pip`
  and always includes site-packages installs (channel `pypi`, or `<develop>`
  for an editable install). The conda and conda-base rows now drop those
  records, so each package appears under the manager that installed it.

- A machine with no Hugging Face cache directory crashed on any explicit
  `--models` run, local paths included: huggingface_hub's `CacheNotFound` is
  a bare `Exception`, so it escaped the cache-scan handlers. The scan now
  reports it as `FileNotFoundError`, and a failed scan warns once rather
  than at every caller. The tests had hidden this by always creating a cache
  directory.
- ty 0.0.86 added `invalid-property-type-override`, which ignores mypy's
  `# type: ignore[override]`, so CI failed on the `_TeeCaptureStream`
  `encoding` and `errors` properties. Each now also carries a
  `# ty: ignore[invalid-property-type-override]`: typeshed declares both as
  writable attributes, but CPython's are read-only descriptors, so the
  property is correct. The dev extra now requires `ty>=0.0.86`, because older
  ty warns that the rule is unknown.
- Crash attribution no longer reads the interpreter's environment path as
  a package: with a conda environment named `mlx-vlm`, every frame under
  `envs/mlx-vlm/…/site-packages/` matched mlx-vlm, so a crash raised in
  huggingface_hub or transformers code (Llama-4-Scout's config validation)
  was labelled an mlx-vlm fault. Crash drafts keep three things apart: the
  observed raise site (the innermost library frames), the likely package
  (the heuristic label, shown as "Likely package" and marked unconfirmed
  when the innermost frame is mlx-vlm's own), and who fixes it, which stays
  open because a library can raise on a bad call. The maintainer verdict
  still counts every crash as mlx-vlm's to look at, and names the ones
  raised in another library's code.
- The dependency gate's braces exception now has to match every lockfile
  occurrence Skylos merges into one finding (`metadata.dependency_occurrences`),
  not just the finding's first file, so the same braces 3.0.3 in another
  lockfile fails.
- Image capability is also read one sub-config down: composite checkpoints
  keep their vision settings in a `*_config` section (Qwen3-Omni has
  `thinker_config.vision_config` and an empty top-level `vision_config`),
  so `mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit` was skipped as
  "text-only generation". Nested evidence is named with its parent.
- The Skylos dependency scan removes its report once checked (and the
  quality step clears any leftover first), instead of leaving the ~15 MB
  file in `src/.skylos/`, where the next run's quality step stopped on it (`SKY-ANALYSIS-INCOMPLETE`:
  file exceeds the secret scan's 8 MB limit), failing every gate and
  pre-push check after the first.
- The per-model "Mechanical observations" log line names each finding in
  words, using the report labels (e.g. "Output repeats the prompt's own hint
  text instead of describing the image") instead of internal codes such as
  `prompt_hint_echoed`. The codes stay in `results.jsonl` and in the
  file log's `REPRO` line.
- The loaded-weight size (`model_weight_buffer_gb`) is measured over the
  model's parameters, the arrays upstream load evaluates. Measuring the whole
  module tree also reached private derived arrays (rotary caches) that are
  still unevaluated after load, so `mx.get_array_buffer_size` refused and
  every model logged a traceback with no size recorded. With `--lazy` the
  weights are unevaluated and the size is left unrecorded, without forcing
  an evaluation.
- Chat-template facts follow transformers' real behaviour: the source
  order is the processor loader's precedence (a `processor_config.json` key,
  then the legacy `chat_template.json`, then `chat_template.jinja`, then the
  tokenizer config), checked in a test against
  `ProcessorMixin.get_processor_dict`, in both the harness and
  `tools.hub_precheck`; templates parse with transformers' extensions
  (`{% generation %}` blocks, loop controls); and a template that could not
  be analysed is recorded as unavailable rather than as reading nothing.
- The allocation-capacity section also renders for the Metal buffer-count
  limit, which is classified as an error rather than out-of-memory.
- The decoding-label lookup in the baseline comparison is annotated
  `dict[str, str]`; Pylance inferred the constant's literal keys and
  rejected a plain `str` lookup.
- Allocation failures name the Metal limit mlx reported: one buffer above
  the maximum buffer size, the buffer-count limit (a number of buffers, not
  bytes; no longer classed as out-of-memory), or exhausted memory
  (`[malloc] Unable to allocate`, which the old `metal::malloc` match
  missed).
- System facts labelled "(run start)" (available memory, swap, power source
  and mode) are taken once at the start of the run and reused by the reports
  and the closing summary; both had re-read them at the end, so an unplugged
  start read Battery at the top of the log and AC under the same label at
  the bottom, and the retained results recorded end-of-run values.
- The console log's per-model results open with a "Per-model results
  (failures first, then fastest to slowest)" heading, so the last model's
  closing notes no longer read as the first result's, and entries are
  numbered `[n/N]` rather than `[RUN n/N]`, which implied run order.
- The per-model timing tree no longer gives cleanup a share of the total:
  cleanup runs after the total is taken, so the shares summed past 100%. It
  is shown as "Cleanup (after total)".
- Cache discovery skips a repo whose `config.json` is not a readable JSON
  object or has no `model_type` (cache-layout reason), as mlx-vlm's server
  listing does; upstream `load()` dispatches on `model_type`, so such a repo
  could only produce a load crash misreported as a model failure. Only a
  file actually on disk is judged. No currently cached model is affected.
- Streaming diagnostics separate first stream activity from the first
  generated token. A speculative or diffusion draft, or upstream's zero-token
  terminal result, no longer sets the time to first token, the first-token
  peak memory or the before/after-first-token failure boundary; only a
  non-draft chunk carrying a token does. The stream is closed on every exit,
  so an early abort runs upstream's cleanup at once; a cleanup error never
  replaces an exception already in flight (a Ctrl-C cancellation stays a
  cancellation, a model error stays that error, with the cleanup error as a
  note). Time to first token can
  read slightly later than before for draft-first models.
- The API-drift check no longer treats `**kwargs` acceptance as proof that
  upstream consumes a generation setting: every keyword the harness sends
  must also be declared in `mlx_vlm.generate.types.GenerateKwargs`, when the
  installed mlx-vlm has it. All currently sent keywords are declared. Like
  the other upstream probes, it never imports mlx-vlm when the import
  safety probe has marked that import unsafe.
- Image validation closes the image it loads as a probe.
- New tests: native repro commands parse with mlx-vlm's own
  `parse_arguments()` (after upstream test_cli.py's `_args()`), with every
  value round-tripping; upstream `generate()` and the harness accumulator
  agree on the same synthetic stream; a pinned revision reaches `load()`; the report
  preview corrects EXIF orientation and keeps PNG transparency; malformed
  images are rejected.
- Follow-ups to the code-review fixes:
  - A cached repository that lacks a requested revision now triggers the
    one cache rescan, so a revision downloaded mid-run resolves (and
    burden facts, sampling and provenance see it). An explicit revision
    still missing after the rescan returns nothing, never another snapshot.
  - A model absent from this run is called "no longer in the cache" only
    when its baseline row shows it was loaded from the Hugging Face cache.
    Local directories, in any path form, and rows without provenance are
    "not run this time".
  - Parameter-size tokens must match whole, so `E12B` or `E4.5B` no longer
    yield a total from their tail.
- Code-review fixes in `check_models.py`:
  - Report escaping. A tag whose quoted attribute contains `<` is now
    escaped, and `&quot;` is no longer double-escaped in tables. A bare URL
    followed by `]` or `>` is wrapped whole and never takes in a `<`.
    Blockquotes neutralise `~~~` fences and one- or two-character setext
    underlines. Image metadata (IPTC/XMP) in the gallery is escaped.
  - Comparison. Settings recorded in only one run (a targeted rerun against
    a full sweep) are listed as unverified instead of making the runs
    incomparable. Baseline revisions must be commit ids before they reach
    git, and a `--compare-with` ref starting with `-` is refused. The
    decoding split counts the same completed pairs as the text count. Changed
    text and changed failure signatures count as changes, and the log no
    longer says "no changes" for incomparable runs.
  - Regeneration. The summary rejects a retained comparison its renderers
    cannot format, and links the gallery only when the manifest lists it.
  - Reproduction. Per-model repro commands carry the model's effective
    checkpoint sampling (native mlx-vlm does not read
    `generation_config.json`).
  - Runs and analysis:
    - A cache miss rescans once, so a model downloaded mid-run gets its
      checkpoint sampling.
    - Any existing directory passed to `--models` is treated as a local path.
    - Isolated-worker failures record their own traceback instead of
      `NoneType: None`.
    - Every generation exception keeps its prompt diagnostics.
    - Reasoning markers are located in the original text, since `casefold()`
      can lengthen it.
    - Captured output from a generation-phase failure no longer marks a
      crash as a connectivity problem.
    - A single run-on keyword is truncated in previews.
    - `E4B`-style effective sizes are not recorded as total parameter
      counts.
    - The working set is shown in the same decimal GB as peak memory.
    - Non-ASCII `--eos-tokens` are no longer mangled.
    - A NaN temperature and infinite KV bits are rejected.
- Progress bars printed during a run (mlx-vlm's prefill bar under
  `--verbose`, huggingface_hub's fetch and Xet bars) render as they would
  without the harness: full terminal width in Unicode blocks, so their right
  edges line up. The per-model output capture stood in for stdout/stderr but
  reported no encoding and no file descriptor, so tqdm fell back to ragged
  ten-character ASCII `#` bars; it now forwards both from the real stream.
- ty 0.0.84 stopped inferring `dict(mapping)` from the expected
  `dict[str, JsonLike]` type and expanded the recursive alias instead,
  failing four call sites; they now spell the type out
  (`dict[str, JsonLike](...)`), which every checker reads the same way.
- Dropped the stale `smol-toml` 1.6.1 npm override (added in March for an
  earlier advisory) and moved to markdownlint-cli2 0.23.3, which brings
  `smol-toml` 1.8.0; Skylos's dependency scan flagged 1.6.1 for a newly
  published denial-of-service advisory (GHSA-7w5x-hrqm-74c2, fixed in 1.7.1).
- `tools/update.sh` now always moves markdownlint-cli2 to the latest npm
  release (`UPDATE_NODE_TOOLING` defaults to `1`; `0` installs from the local
  lockfile when offline), with `--ignore-scripts` on that path too. The
  package stays a caret range with no override.
- Console rules and banners no longer wrap (0.17.37). They were drawn at the
  full console width although every line carries the handler's 11-column
  `[HH:MM:SS]` prefix (20 with `--verbose`), so each rule left an orphan
  fragment on the next line on any terminal under about 131 columns.
  `get_terminal_width()` now returns the width left for content;
  `_console_total_width()` sizes the console itself. The never-assigned
  `WIDTH_OVERRIDE` global is gone (`--width` has always travelled through
  `MLX_VLM_WIDTH`).
- Default discovery no longer skips models downloaded by huggingface_hub 1.32+
  as "weight shards missing" (0.17.35). That release keeps weights in a
  hub-wide, content-addressed `blobs/<xx>/<hash>` directory and makes each
  repo's blob a link into it; the snapshot-file containment rule only trusted
  the per-repo store, so complete, readable models were rejected. In the
  standard cache layout it now also accepts that one sibling directory, and
  nothing else.
- The retained-row validator checks `emitted_special_tokens` and
  `duplicated_answer_separator`. Its four field groups were hand-kept lists
  that lagged `JsonlObservationDetailsRecord`; they are now derived from the
  TypedDict's declared types, and a test pins that every declared field is
  in exactly one group.
- A token cap that lands mid keyword list is reported as truncation even
  when it falls after a word rather than a comma. Muse Glimmer spent its
  1000 tokens on a hidden reasoning channel, its visible answer stopped at
  "Keywords: mute swan, river", and with all three labels present it was
  rated usable with no observations.
- `tools/update.sh` pins the surviving local mlx build when a rebuild fails.
  The failure branch used to skip the pin, so the final eager project
  reinstall ran unconstrained and would have replaced a working local build
  with the PyPI wheel once a release overtook the local dev version. Found
  when Xcode 27's Metal Performance Primitives header rejected mlx's
  gated-delta NAX kernel and the build failed with the old build intact.
- A model the machine slept through (`wall_clock_gap_s`) keeps its peak-memory
  comparison and its history rows are skipped when throughput noise bands
  are built; only its rate is excluded.
- Telemetry reports low power mode as a per-probe count
  (`low_power_samples`) instead of a max over pmset power modes, which had
  labelled high-power mode as low power.
- The batched suppression audit hands ruff output only to noqa findings; a
  line carrying both a noqa and a type: ignore audits the latter with mypy.
- Fast quality checks run the Skylos danger advisory (non-blocking), which
  the removed `-a` audit pass used to provide in that mode.
- Token-id special-token evidence now survives analysis. The id detector
  selects from the tokenizer's `all_special_ids`, but the analysis filtered
  its output against the tokenizer's whole special-token vocabulary (the
  strip list for the analysis copy), so every id-detected token was
  discarded whenever prompt diagnostics were populated and no run ever
  reported `emitted_special_tokens`. Only the generation contract's own
  wrappers neutralise id evidence now: the ids upstream's stopping criteria
  stop on (tokenizer EOS plus generation_config stops such as `<turn|>` or
  `<end_of_utterance>`), configured stops and thinking start/end; a
  recognised thinking trace still neutralises its own markers. Id evidence
  is reported as "Special tokens emitted (by token id)" and counts as
  leakage only when the token also reached the returned text, so markers a
  processor consumed (GLM's box markers under `clean_output`) are visible
  facts without a usability penalty.
- Checkpoint-declared sampling values that JSON parses to `inf` (`1e400`)
  or to an integer too wide for a float are rejected as invalid instead of
  being forwarded to generation or raising `OverflowError` during
  preparation; the harness default and a `default` source are kept.
- The `torch` extra is back in `tools/update.sh`'s default install set (the
  short-lived `INSTALL_TORCH=1` opt-in is reverted). mlx-vlm computes nothing
  with torch, but transformers 5 constructs torch/torchvision-backed video
  and image processors inside `AutoProcessor` for about half the roster, so
  without the extra 16 of 33 cached models fail before any image is seen.
- `tools/update.sh` parses conda's dry-run change lines again: conda prints
  `old --> new` with two dashes and the parser expected one, so it matched
  nothing and always reported the conda environment as already up to date.
  Python patch releases (the env pins `python 3.14.*`) were therefore never
  adopted by the script. A test runs the real grep pattern against captured
  conda output.
- The diagnostics triage table's "Mechanical checks" column now shows the
  human gloss (no concerns detected / concerns detected / major concerns)
  like every other surface; it still printed the machine usability code.
- The Skylos wrapper raises Skylos 4.36's grep-verification cap
  (`SKYLOS_GREP_BUDGET`, default 30 s) to 300 s: the three-core CI runner
  exceeded the default on this tree and Skylos exited 2 as "analysis
  incomplete" although the scan was clean. A cap, not a duration, so local
  runs are unaffected.
- The crash section of the run summary no longer guards a `model_provenance`
  the JSONL validator already requires; ty 0.0.80 flags that guard as a
  redundant condition and exits non-zero, which turned the gate red.
- The text-divergence shared prefix is counted with a plain character
  comparison instead of `os.path.commonprefix`, which Ruff's new RUF071 flags.
  Ruff's suggested `commonpath` would compare path components, not text, so it
  was not applied; the counts are unchanged.

## [0.17.0] - 2026-09-05

### Assessment profiles (0.17.0)

- Clarified lane, custom-prompt, assessment-profile, and token-budget precedence
  with an interaction table and explicit triage-rerun overrides in CLI help and
  the README. Startup now logs the resolved lane, prompt source/hint exposure,
  assessment profile, and token budget together; execution behaviour is unchanged.
- Added `--assessment-profile general|metadata`, independent of prompt wording
  and the evaluation lane. Built-in metadata prompts select metadata checks;
  custom prompts and triage select general checks. Explicit selection overrides
  the default; differential triage reruns remain general. The profile is retained
  in the JSONL header and each result assessment.
- Reports state the assessment scope and use "no concerns detected" rather than
  implying that an arbitrary task or factual-accuracy check passed. Metadata
  checks retain required fields, duplicate keywords, and neutral title/keyword
  counts. Duplicate keywords have their own `duplicate_keywords` observation.
- Removed prompt-contract recognition, prose range parsing, sentence counting,
  hint-overlap and instruction-echo heuristics, their obsolete tests/thresholds,
  and `--context-marker`. Short answers, copied hints and prefaces no longer
  cause automatic downgrades. Cap hits without stronger evidence remain neutral;
  absence of final punctuation is not proof of truncation.
- General checks retain crashes, empty output, repetition, control-token evidence,
  incomplete thinking and runtime facts. Complete answers and raw output remain
  available for human review. Older retained assessments remain readable and
  are labelled as having no recorded profile; report regeneration does not
  silently reclassify them. Comparisons with different assessment profiles are
  withheld rather than reported as model regressions.

### Added

- Tests for eight live functions no test had exercised (`_model_burden_rows`,
  `_diagnostics_environment_section`, `_write_environment_failure_diagnostics`,
  `filter_and_format_tags`, `pretty_print_exif`, `_decode_iptc_keywords`,
  `_append_markdown_section`, `_parse_processor_kwargs_arg`).
- Streaming repetition guard: generation now runs through a thin accumulator
  over upstream `stream_generate` (reproducing `generate()` semantics — joined
  chunk text, final-chunk metrics) that stops decoding once the output tail is
  four exact repeats of a substantial unit past a 200-token floor. An abort
  sets `finish_reason="repetition_abort"`, which flows into the runtime stop
  reason and a new `repetition_abort` observation ("stopped early:
  repeating"); aborted generations are excluded from cross-run throughput
  comparisons and history bands (see Fixed below). The token-cap observation
  no longer conflates degenerate loops with genuinely long answers.
- Failures inside a repository's `trust_remote_code` modules now attribute to
  a dedicated `model-repo-code` owner instead of the library whose message
  they resemble: a `transformers_modules` frame (the dynamic-module cache) is
  checked before the message-first attribution flow, so e.g. a repo's fast
  image processor importing a symbol transformers removed routes the issue to
  the model repository rather than to transformers or mlx-vlm.
- The run issue summary counts "Hit the token cap" and "Stopped early for
  repetition" in its header, and renders a "Constraint-failure breakdown"
  section aggregating fleet-wide catalogue-constraint failures (title-length,
  keyword-count with medians, duplicate keywords) — separating "the prompt is
  hard for everyone" from "individual models are sloppy" at a glance.

### Fixed

- The description sentence counter no longer raises on a terminator with
  nothing before it (`Description: ... The mill spans a river.` crashed
  inside successful-result construction) and no longer splits after an
  abbreviation followed by a number (`Built approx. 1750, ...` counted as
  three sentences): a boundary now needs a capital letter after the
  whitespace, more abbreviations are known, and the docstring states the
  residual over-count risk instead of claiming none. Both cases are
  regression-tested through the assessment path.
- History noise bands additionally match each model's effective generation
  settings (`generation_settings` is now recorded per model in
  `results.history.jsonl`), so a thinking-budget change that the run-level
  fingerprint cannot see — it keeps only settings common to the whole sweep
  — no longer blends two workloads into one band.
- The reasoning disclosure and omitted-character count now come from the
  spans the delimiter processing actually removed, not from re-finding the
  answer's first characters in the raw output; a model that drafts its final
  answer inside the thinking block previously had most of its trace
  mis-attributed to the answer.
- Baseline comparisons no longer treat runs on different hardware as
  like-for-like for performance: the chip (`system["GPU/Chip"]`) is now a
  comparison fact, so a differing chip withholds the tok/s ratio, throughput
  flags and peak-memory moves (quality transitions still show), a missing chip
  on either side is reported as an unverified fact, the baseline's hardware
  appears in the component rows, and the retained `comparison` block carries
  a `hardware` pair alongside `execution_mode`.
- The retained `results.jsonl` is now rewritten atomically (staged next to
  the target and renamed into place), so a failure during the final manifest
  reconciliation leaves the pre-report file intact instead of truncated, as
  the failure message already promised.
- `--isolate` works again: the parent wrote the child spec with keys such
  as `params.model_identifier` (a mechanical rename had rewritten the JSON
  key strings too) while the child still read `model_identifier`, so every
  isolated model failed with `KeyError` before inference. The spec writer
  and the child parser are now adjacent functions with a round-trip test
  that runs both for real; the existing isolation tests mock the subprocess
  and could not see the drift.
- The pre-push quality gate no longer misfires from a linked git worktree.
  Git exports `GIT_DIR` into hook processes; with `GIT_WORK_TREE` unset,
  git then treats the current directory as the work tree, so every
  `git diff` run from `src/` reported the whole repository as deleted and
  Skylos `SKY-L021` flagged every validation call as "removed" (it also
  scanned the excluded `src/output/` log). `run_quality_checks.sh` now
  unsets `GIT_DIR` so git rediscovers the repository normally; the
  short-lived `SKY-L021` ignore is gone and the upstream bug is reported.

- `update.sh` no longer skips rebuilding a local MLX checkout that has
  uncommitted or untracked changes: an unchanged HEAD with modified C++,
  Metal, or packaging inputs left an older compiled extension, metallib, or
  dylib in use under a provenance claiming the current source. The
  decision is a pure shell function with a behavioural test over the
  unchanged-clean, dirty, changed-HEAD, wrong-editable-origin, and
  FORCE_REINSTALL cases.
- The schema-3 `results.jsonl` header is rewritten last, after every
  report outcome is known: its `artifacts` manifest lists only artifacts
  this run produced (matching `index.md`, never a failed renderer's stale
  file), and its `total_runtime_seconds` / `timestamp` are end-to-end.
  The HTML runtime row, rendered before reports finish, is labelled
  "Model sweep runtime"; the console line is computed at exit.
- `validate_env`'s no-pyproject fallback tracks the declared runtime set
  again (it still required the retired `wcwidth` and omitted `numpy` and
  `rich`); a test now holds the two name sets equal. The implementation
  guide's runtime list matches `pyproject.toml`.
- `validate_env --fix` installs the pre-commit hooks once, interpreter-
  qualified, instead of a second time via whatever `pre-commit` is on PATH.
- `update.sh` no longer advertises or detects `uv`; `common_quality.sh`
  drops a no-op `quality_activate_conda` shim.
- The packaged-wheel test builds through standard PEP 517 isolation from
  the declared `[build-system]` instead of `--no-build-isolation`: it had
  only ever passed on CI because `huggingface-hub[torch]` dragged `torch`
  and, transitively, `setuptools` into the environment, and it broke the
  moment that unrelated extra was removed.
- Skylos no longer overwrites the user's clipboard from the quality gate:
  every Skylos invocation goes through `quality_run_skylos`, which puts a
  throwaway `pyperclip` stub (raising `PyperclipException`) first on
  `PYTHONPATH` so the grade renderer's unconditional badge copy takes its
  quiet no-clipboard branch. No Skylos pin or dependency change.
- Run timing is one wall-clock concept a skimmer can read: the overall
  runtime is now measured on the wall clock (it previously used a perf
  counter that stops during system sleep, so a sweep that slept mid-run
  reported 15 minutes for a 20-minute run), the schema-3 header retains
  the run's `started_at` (optional, so existing files load unchanged),
  `run_summary.md` opens with *Run started / Run finished / Run duration*,
  `index.md`'s dashboard leads with the duration, and long durations render
  as `15m 26s` / `1h 05m 12s` instead of `925.55s`. The retained-run window
  prefers `started_at`; older headers bound the start by the earliest
  retained result timestamp, which fixes the summary omitting the run's own
  `check_models.log` and `environment.log` as "stale" while `index.md`
  still linked them.
- Every scripted `npm install` (`update.sh`, `setup_conda_env.sh`, the
  Makefile's lazy tooling installs, and the CI static-quality job) passes
  `--no-audit --no-fund`: nothing consumes the audit or funding output, and
  a degraded npm advisory endpoint (registry ping 0.3 s, bulk-advisories
  POST timing out with zero bytes) left every `npm install` hanging in its
  final audit phase on multiple consecutive runs — on CI that would hold a
  runner until the job timeout.
- The repetition guard mirrors two upstream `generate()` behaviours added
  for diffusion_gemma (mlx-vlm #2101): the joined text passes through the
  processor's optional `clean_output` hook (stripping leaked
  `<|channel>` scaffolding that plain mlx-vlm users no longer see), and
  the verbose echo skips chunks upstream marks `text_already_printed`.
  Without the hook, stream-based results would report artifacts as
  maintainer observations that upstream's own path already removes.
- Every report surface now opens with one canonical scope statement
  (`_run_objective_statement`): the run probes exactly one narrow task —
  catalogue metadata for a single photograph (in the assisted lane, aided
  by camera capture context and draft hints from a more capable model) —
  and says nothing about fitness for other uses. Shared verbatim by
  run_summary.md, the model gallery, diagnostics.md, results.html, and
  index.md so the framing cannot drift between surfaces.
- `run_summary.md` now explains itself to a first-time reader: an opening
  paragraph states the narrow objective under test (catalogue metadata for
  one photograph, in the assisted lane aided by camera capture context and
  draft hints from a more capable model — not general model quality), the
  exact prompt is embedded collapsed at run level, and the at-a-glance
  section defines the usability vocabulary and the Total / Gen tok/s /
  Peak GB columns.
- Verbose sweeps stream generated text live again: the repetition guard now
  echoes each retained (non-draft) chunk to stdout as it arrives, restoring
  the ergonomics lost when guarded generation replaced upstream
  `generate(verbose=True)` (which echoed) with `stream_generate` (which does
  not). The echo runs inside the tee capture, so the live console and the
  retained upstream-output capture both keep the text; draft chunks are
  neither echoed nor retained, and non-verbose runs are unchanged.
- Observation `details` contents are validated at the loader against their
  declared shapes (string lists of strings, non-bool integer counts,
  exactly-two-int ranges, plain-string fragments; unknown keys stay
  permitted for forward compatibility), so a mistyped retained value like
  `"title_word_count": "five"` is rejected before a renderer compares it.
- The schema-3 loader also validates the nested structures reports index
  into (`assessment.details`, `prompt_diagnostics.generate_kwargs` must be
  mappings when present), and comparison rehydration is strictly typed: a
  JSON string where a boolean/enum/timestamp belongs raises into the
  degrade-to-no-baseline path instead of being truthiness-coerced (e.g.
  `"false"` no longer reads as `True` for `throughput_comparable`).
- Remaining `run.json` descriptions in the root README, feature lists, the
  monolith's architecture header, and internal docstrings now describe the
  schema-3 `results.jsonl` contract.
- Report, diagnostics, summary, and comparison rendering are contained at
  their orchestration boundaries: an unexpected exception in one renderer
  degrades that artifact (and logs it) instead of terminating finalisation,
  and a comparison render crash degrades the comparison to "none" rather
  than losing the remaining artifacts. `KeyboardInterrupt`/`SystemExit`
  still propagate.
- Navigation surfaces (`index.md`, the terminal artifact log) list only
  artifacts whose generation outcome succeeded this run; a stale file left
  on disk by an earlier sweep is never presented as current, and
  `environment.log` is reported from an explicit success signal instead of
  `Path.exists()`.
- The whole-run constraint-failure aggregate groups observations by their
  actual `(min, max)` bounds instead of assuming one fleet-wide range, so
  mixed-prompt retained artifacts aggregate correctly; medians stay
  fractional and below-range is separated from above-range.

- The upstream-parity tests no longer import `mlx_vlm` in the pytest
  interpreter: importing any submodule executes the package root, which
  initializes `mlx.core`'s native Metal backend and can abort the whole
  process (not raise) when Metal is unavailable — `pytest.importorskip`
  cannot catch a native abort. The GenerateKwargs parity test now reads the
  installed `types.py` as source (AST), and the CLI-defaults parity test
  captures upstream argparse defaults in a subprocess probe that skips
  cleanly when the child dies. Retiring the in-process capture also removes
  the suite's last `type: ignore` monkeypatch suppression.
- Review fixes on the repetition guard: the streaming wrapper now performs
  upstream `generate()`'s stopping-criteria setup (custom `--eos-tokens`
  register on the tokenizer, and reset to the model default otherwise, so a
  prior model's registration cannot leak) and skips speculative/diffusion
  draft chunks, whose text upstream excludes from the final answer.
  Repetition-aborted generations are excluded from cross-run throughput
  comparisons and history noise bands (a rate over a few hundred tokens is
  not comparable with a full-length run). The runtime API drift check now
  validates `stream_generate` — the call the harness actually makes — and
  the unused `generate` module symbol is removed. The constraint-failure
  aggregate validates retained `details` ranges before indexing (stale or
  malformed artifacts degrade instead of crashing summary regeneration),
  keeps fractional medians, and splits below-range from above-range counts.
- The e2e smoke tests' fixture model is now `LiquidAI/LFM2.5-VL-450M-MLX-bf16`
  (the smallest usable model in the standing suite); they previously targeted
  the retired `nanoLLaVA-1.5-4bit` and silently skipped their two
  real-inference cases once that cache entry was removed.
- Name-based parameter estimates use `Decimal` arithmetic:
  `int(float("4.1") * 1e9)` truncates to 4,099,999,999 because 4.1 is
  inexact in binary — the same defect mlx-lm fixed in its `_parse_size`
  (ml-explore/mlx-lm#1726).

- CodeQL can analyze the monolith again: GitHub's Python extractor
  (tsg-python) mis-slices the U+FE0F emoji variation selector while
  evaluating string literals in any file containing PEP 695 syntax, and when
  the same literal carries a %-format directive its error reporter crashes
  ("not enough arguments for format string"), silently dropping the whole
  file from security analysis — `src/check_models.py` had been unanalyzable
  since at least 2026-03 (when the first `type` alias landed). All `⚠️`
  glyphs in Python sources are now written with the selector escaped
  (`⚠\ufe0f`), which renders identically and keeps the source ASCII at
  that position; a source-hygiene test guards the invariant, and the root
  cause was bisected to a three-line reproducer with a local CodeQL CLI.

### Removed

- `_parameter_count_from_name` compatibility wrapper; callers and tests use
  `_parameter_counts_from_name` (total, active) directly.
- `mlx-lm` is gone from the project entirely: nothing imports it and
  mlx-vlm has not depended on it since 0.6.14, so the `extras` entry, the
  legacy mlx-vlm < 0.6.14 floor logic and its policy constants, the report
  version rows, the error-attribution and traceback markers, the mypy
  override, and the local-build/verify/clean stages in `update.sh`,
  `setup_conda_env.sh`, `probe_python_next.sh` and `clean_builds.sh` no
  longer mention it. A sibling `mlx-lm` checkout is simply ignored.
- Tooling cruft found in the maintainability review: `bugtest.py` and
  `update.sh`'s Metal-regression reminder (the M5 NAX matmul regression it
  probed for is long fixed, and the runtime smoke already catches a broken
  backend), the custom `install_precommit_hook.py` (the checked-in
  `.pre-commit-config.yaml` installs the same two hooks via
  `pre-commit install`, which `make dev`, `setup_conda_env.sh`, and
  `validate_env`'s auto-fix now run), the `update-env` / `update-full` /
  `check_models-demo` alias targets, and the never-filed workflow-hardening
  PR draft (archived). `make probe-python-next` is now listed with the
  other maintenance targets. All three type checkers (mypy, ty, pyrefly)
  stay in the gate.
- Dead code and rolled-our-own utilities found in a maintainability
  review: `ResultSet.get_fields` / `_get_available_fields` (never used by
  the harness), the `_version_components` fallback (PEP 440 parsing via
  `packaging` already sorts dev, rc, and local versions; an unparseable
  installed version now simply fails the floor check), and the optional
  `wcwidth` dependency — terminal display width now comes from
  `rich.cells.cell_len`, since `rich` is already a hard dependency.
- Unused dependency declarations: `pydantic` (no consumer anywhere in the
  tree), `types-tqdm` (no `tqdm`), and the `torch` extra on
  `huggingface-hub`, which pulled `torch` into the base install and
  silently defeated the opt-in `torch` group.
- The `run.json` artifact and its writer (`save_run_json_report`,
  `RunJsonReportRecord`, `--output-run-json`): the schema-3 `results.jsonl`
  metadata header now carries the complete run-level contract. A schema-2
  retained baseline is rejected by the single loader (no adapter), so the
  first `--compare-with` against an old sweep is skipped with a logged
  reason and the next schema-3 run establishes the new baseline.
- The seven per-artifact output flags (`--output-html`,
  `--output-gallery-markdown`, `--output-jsonl`, `--output-run-json`,
  `--output-log`, `--output-env`, `--output-diagnostics`), replaced by the
  single `--output-dir` root; retired flags are rejected with an actionable
  error.
- The retired `stress`/`quality` evaluation-mode aliases (now rejected at
  parsing), `--detailed-metrics`, and `--open-report`; verbose mode always
  renders the full detailed metrics tree, and the unreachable compact
  verbose renderer and legend branch are deleted.
- Dead definitions with no reference sites (`FLOAT_ZERO_EPSILON`,
  `ERROR_MESSAGE_TRUNCATE_LEN`, `BPE_BYTE_ARTIFACTS`, `_html_code_block`,
  and the test-only `_public_output_artifact_map`), plus a duplicate
  `--output-dir` parsing test already covered by the layout-derivation test.
- Reporting-archaeology tests asserting the absence of long-retired
  surfaces, replaced by one canonical retired-terms guard across every
  rendered artifact (test count 903 → 896 before the new schema-3 loader
  regressions).

- The mlx-vlm stub-generation subsystem is fully retired: mlx-vlm 0.6.16+
  ships a PEP 561 `py.typed` marker on PyPI (from our Blaizzy/mlx-vlm#1985),
  so every dependency in the typed surface now provides its own types.
  Deleted `src/tools/generate_stubs.py` with its patches/contracts/manifest
  handling, the `stubs`/`stubs-clear`/`clean-stubs` Make targets, the
  update.sh and quality-gate stub steps, the `../typings` search paths in
  mypy/ty/pyrefly/vulture configuration, and the stub contract tests. The
  mlx-vlm floor rises to 0.6.16 (first release shipping `py.typed`) so the
  guarantee holds by construction.

### Changed

- Format compliance is no longer confusable with accuracy on the influential
  surfaces: chooser, output-at-a-glance and run-summary tables head their
  axis "Format/structural usability", resource highlights and the
  run-summary/diagnostics sections say "passing mechanical checks" instead of
  "clean", and the chooser explanation states outright that a model can pass
  every check while copying hint keywords or misidentifying the subject.
  Machine codes (`usable`, `usable_with_caveats`, observation codes) are
  unchanged.
- The description part of the catalogue contract is now assessed, conservatively:
  a description with more sentences than the prompt's requested range is a
  `catalog_constraint_violation` with `description_sentence_count` and
  `description_sentence_range` evidence, surfaced in the per-model label,
  the constraint-failure breakdown and the details table. The sentence
  counter only splits on a terminator followed by whitespace and a capital
  or digit, ignores abbreviations, initials, dotted acronyms and decimals,
  and checks the upper bound only, so it can under-count but never invent a
  violation. The observation label reads "Title, description or keywords do
  not meet requested constraints".
- Performance comparability uses a fuller identity. Hardware identity is
  now chip plus GPU core count plus RAM, so the same chip with fewer cores
  or less memory withholds throughput. History noise bands are grouped by a
  comparison fingerprint (prompt, image digest, generation settings, lane,
  execution mode, hardware) that `results.history.jsonl` now records as
  `comparison_fingerprint`, and a model's samples must come from the
  revision under test (`resolved_revision` is now stored per model);
  older history rows without those facts fall back to the fixed ±15% band.
- Parameter estimates from model names distinguish active from total: an
  `A3B`-style token is an active-parameter designation, so
  `Kimi-VL-A3B` reports "Active parameter count: 3.00B (name-estimate; total
  not stated in the name)" instead of a 3B checkpoint, and `30B-A3B` names
  report "30.00B total, 3.00B active".
- Chooser and output-at-a-glance previews now show each model's final
  answer: a closed thinking trace (emitted or prompt-seeded, the same rule
  the assessment uses) is left out of the preview, reported as an
  omitted-character count in Markdown, and opened under a `<details>`
  disclosure in the HTML chooser. Thinking models such as ERNIE and
  Qwen3-VL-Thinking no longer spend their preview on scratch work.
- Every summary surface — run summary, output index, gallery, HTML report,
  and diagnostics — now leads with the evaluation lane and the input image
  (format, dimensions with megapixels, size), so a 66-megapixel input is
  visible before any per-model prefill timing. The run summary's
  "Evaluation mode" row became "Evaluation lane" within that block.
- Resource highlights rank by time to complete the task end-to-end instead
  of decode tok/s: the gallery and HTML "Fastest clean completion" and
  "Average clean-completion throughput" lines became "Quickest clean
  completion (end-to-end, including model load)" plus an explicit caveat
  that tok/s is not averaged across models; the terminal summary's
  "Fastest (tps)", "Average TPS", and "Memory efficiency: tokens/GB" lines
  are replaced by "Quickest completion". Per-model tok/s stays in the
  chooser.
- Project instructions and adapted skills no longer describe removed
  machinery: `run.json` references now point at the `results.jsonl`
  metadata header (and `issues/run_summary.md` as the skim surface), the
  native-repro Python shape loops over `stream_generate` the way the harness
  does instead of calling upstream `generate()`, `mlx-lm` is stated as not a
  dependency, and the generation seam, `--isolate` spec pairing, and atomic
  final rewrite are documented so future changes start from the real code.
- Suppression review: configuration-wide suppressions that suppressed
  nothing or hid real findings are gone. Ruff `S311` (no `random` use) and
  the `tests.*` mypy override are removed (markdownlint `MD041` stays off,
  now with a rationale naming the two deliberately H1-less files); the
  argument ceiling is enforced again (`PLR0913` and Skylos `SKY-C303` are
  no longer ignored globally — the four builders that exceed it carry
  individually justified suppressions); `ANN401` applies everywhere except
  argparse's keyword protocol, and `_open_image_for_exif` is typed as
  returning `Image.Image`; the blanket test `ARG001`/`PLR0915` exclusions
  are replaced by removing five dead parameters (including an unused
  `harness_type` on a test helper) and three targeted long-test
  suppressions; the `tools/**` `S603`/`S607`/`BLE001` exclusions are
  replaced by narrowed exception types in `validate_env`, a resolved
  shellcheck path, per-call rationales, and removal of a duplicate
  `pre-commit install` subprocess. `tools.check_suppressions` now states
  that it audits inline suppressions only.
- The test suite runs in parallel (`pytest -n auto`) from `make test`,
  `make test-cov`, and both gate invocations — pytest-xdist was already a
  dev dependency that nothing used; wall clock drops from ~61 s to ~34 s
  locally. A suite-wide `MLX_VLM_WIDTH` pin in `conftest.py` makes console
  rendering deterministic: under xdist a worker's stdout is a pipe, the
  width fallback was narrower, and two working-set assertions failed on
  truncated tree rows.
- `_run_model_isolated` takes the already-built `ProcessImageParams` from
  `_run_one_model` instead of re-threading the same eight keyword
  arguments through a third signature.
- Output-quality observation is a single canonical projection
  (`_quality_observations` + `_completed_assessment`) shared by the main
  harness and `tools/analyze_output_quality.py`, which now emits the same
  `assessment` block as the reports and exits non-zero only for unusable
  output. The legacy per-result quality-issue strings are deleted.
- `results.jsonl` is schema `3.0` and the sole current-run machine
  contract: the metadata header adds the prompt SHA-256, total runtime,
  outcome counts, artifact paths, producer identity, source-image facts,
  common generation settings, remote-code policy, and the embedded baseline
  comparison; per-model rows add their prompt-token burden. The single
  loader validates every required header field (including counts
  consistency against the loaded rows) and every consumed row field, and
  summary regeneration rehydrates the retained comparison.
- `--output-dir` is the only output-location control; the canonical layout
  (`index.md`, `results.jsonl`, logs at the root; HTML/gallery/diagnostics
  under `reports/`; conditional `issues/`) is derived from that root.

- Skylos quality findings now gate: `max_quality` drops from 10000 to 0 in
  both the package and root configurations, so "quality passed" means no
  quality warnings, not merely none blocking. The per-model comparison's
  throughput/memory accumulation moved into `_compare_model_performance`,
  bringing `compare_run_results` back under the configured complexity
  ceiling; the repository scan is clean at the zero budget. Version-drift
  noise remains absorbed by the per-rule ignore list, not a numeric
  allowance.
- `update.sh` skips a local MLX repo's rebuild when `git pull` brought no
  new commits **and** the installed package verifies as the editable from
  that checkout — the verification is the dependency-change guard, so a
  PyPI release that clobbered the editable, a rebuilt environment, or a
  missing install still forces the full rebuild, `FORCE_REINSTALL=1`
  still rebuilds unconditionally, and a skipped mlx build still applies
  its pin against PyPI releases.
- The `tokenizers` ceiling is widened to `<0.24.0` (from `<0.23.0`) in both
  `src/pyproject.toml` and the dependency policy: transformers 5.16.0
  requires `tokenizers>=0.23.1,<0.24.0`, so the old ceiling would have made
  transformers unresolvable past 5.15.1. This lifts the long-standing hold
  on the Dependabot tokenizers bump, which proposed exactly this range.
- The stale-index glob-fallback rescue is aligned with Nativ's merged fix for
  the same snapshot class (Blaizzy/nativ#370): the rescuing weight set must
  stand on its own — exactly one loose full-checkpoint file, or exactly the
  complete 1..N of one `model`-stem shard series. Mixed series, an adapter,
  or a stray loose file beside the series would be merged into the blind
  glob load and no longer rescue. One deliberate divergence remains: a
  malformed or empty index beside a self-standing weight set still rescues
  here, because mlx-vlm's Python loader swallows index errors and globs,
  while Nativ's pre-flight hides that case.

## Older releases

Releases 0.1.0 to 0.15.0 are in
[docs/notes/archive/CHANGELOG_0.1.0-0.15.0.md](docs/notes/archive/CHANGELOG_0.1.0-0.15.0.md).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
