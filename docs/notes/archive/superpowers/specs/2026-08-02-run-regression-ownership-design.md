# Run Regression Ownership and Upstream Fixes Design

## Goal

Turn the 2 August benchmark evidence into narrowly owned fixes: improve
`check_models` where its assessment or execution policy is wrong, and submit an
upstream `mlx-vlm` fix only where the same defect reproduces through native
`mlx-vlm`.

The retained `src/output/` snapshot remains unchanged. New reports will acquire
the improved behaviour when the maintainer reruns the matrix.

## Evidence and ownership

The current and recent retained runs use the same Transformers 5.14.1,
tokenizers 0.22.2, model revisions, and materially identical image-generation
code. The `mlx-vlm` source changes between the immediately preceding and current
runs affect video fallback only. The new image and assisted metadata prompt are
therefore the important changed variables, not a Transformers upgrade.

Native isolation establishes these ownership decisions:

- `mlx-community/Step-3.7-Flash-oQ2e` generates a correct caption through native
  `mlx-vlm`. Its reported processor crash is a `check_models` false positive:
  `Step3VLProcessor` is callable and supports images without exposing an
  `image_processor` attribute.
- `mlx-community/Idefics3-8B-Llama3-bf16` emits a literal
  `<end_of_utterance>` through native `mlx-vlm`. Configuring that declared token
  as EOS produces the same answer without the delimiter. This belongs upstream.
- Both Molmo variants return an immediate empty answer for the long catalogue
  prompt even after the image is resized, while a short one-sentence prompt on
  the same resized image succeeds. This is prompt-following capacity, not an
  image-size or runtime defect.
- Repetition, missing fields, long thinking traces, and raw GLM/DiffusionGemma
  protocol markers do not become upstream defects merely because they are poor
  catalogue answers. They require a smaller native runtime failure before an
  upstream change is justified.

## `check_models` changes

### Runtime correctness

Processor preflight will require only the interfaces the harness actually uses:
a callable processor and a resolvable generation tokenizer. It will not require
the optional `image_processor` attribute. Native `mlx-vlm.generate()` remains
the authority on whether a processor can consume the supplied image.

When loading a Hub identifier fails solely because of connectivity and a
resolved local snapshot is available, the harness will retry once with that
snapshot path. A requested immutable revision must match the snapshot before it
is used. Non-connectivity failures will not be retried, and `--force-download`
will preserve its online semantics.

### Catalogue contract observations

For an output with all three catalogue sections, mechanical validation will
record:

- title word counts outside the requested 5–10 range;
- keyword counts outside the requested 10–18 range; and
- duplicate keywords after case-folding and whitespace normalisation.

These are one stable `catalog_constraint_violation` observation with structured
details. They make a completion `usable_with_caveats`, not unusable, because the
answer remains straightforward to repair.

Configured special tokens that denote a conversation, message, turn, or
utterance boundary and remain visible in generated text will be reported as
`role_boundary_token_present`. Tokenizer-declared wrappers will continue to be
removed only from the semantic-analysis copy, never from retained exact output.

### Issue-summary ergonomics and provenance

Rows within each execution-status table will be sorted by the most severe human
observation and then model name. A non-completed result with no observations
will show a concise failure description, such as `Network connection reset
during model loading`, rather than `none`.

Run context will include `trust_remote_code` and check_models version, revision,
and dirty state when available. Producer dirtiness will ignore tracked generated
files under `src/output/`, so writing a run does not mark its own source tree
dirty; other tracked changes still do.

Repository links remain canonical GitHub URLs. The summary will make their
mutable `main`-branch lifecycle explicit and include the producer revision so a
reader can identify the code used for the run. Creating immutable links to
freshly generated, not-yet-committed output is intentionally not attempted.

The detailed crash issue will show the dependency subset relevant to a model
load/generation failure and link to the complete environment artifact instead of
repeating unrelated SDK/tool fingerprints.

## `mlx-vlm` pull request

The upstream change will teach the Idefics3 processor/loading path to merge the
processor's `<end_of_utterance>` token ID into its stopping criteria. It must
preserve every existing EOS token, deduplicate IDs, and stop before detokenizing
the delimiter. The test will use a small fake tokenizer/processor boundary and
will not load model weights.

The PR body will include the native reproduction, exact model revision,
environment versions, expected and actual output, and the confirmed
`--eos-tokens '<end_of_utterance>'` control result. No issue or PR will be opened
for observations that remain attributable to model capability.

## Tests and verification

Every production change starts with a focused failing test. `check_models`
tests use existing test files and temporary output paths. The retained output
snapshot is not regenerated.

After focused tests, run `make format`, safe Ruff fixes if useful, `make lint`,
commit hygiene, and `make quality` in the `mlx-vlm` Conda environment. The
upstream worktree will run its focused processor tests and formatter/pre-commit
checks before a branch is pushed and a PR is created.
