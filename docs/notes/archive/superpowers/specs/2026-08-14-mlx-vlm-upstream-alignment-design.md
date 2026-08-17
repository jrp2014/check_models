# MLX-VLM Upstream Alignment Design

## Context

`check_models.py` is a focused single-image benchmark and diagnostic harness for
MLX vision-language models. It deliberately calls the direct `mlx_vlm` load,
chat-template, image-loading, and generation APIs rather than exercising the
server.

The current `mlx-vlm` main branch has broadened beyond that original surface. It
now includes native text-only model implementations, embedding models,
generative and sequence-classifier rerankers, image and video generation,
image editing, audio services, speculative decoding, continuous batching,
automatic prefix caching (APC), OpenAI and Anthropic protocol routes, and
realtime APIs. Its `/models` endpoint still uses a deliberately broad cache
layout test: a model repo with a cached `main` revision, config and tokenizer
metadata, and safetensors weights. `check_models` currently mirrors that test
and describes every match as an mlx-vlm-compatible VLM.

That equivalence no longer holds. A cached text-only, embedding, reranking, or
generation-pipeline model can pass the server-style layout filter even though
feeding it an image-description prompt is meaningless. Treating the resulting
output or failure as a broken VLM would mislead both model choosers and
mlx-vlm maintainers.

There is also routine dependency and documentation drift. Current mlx-vlm
release metadata requires newer MLX and Transformers floors than this project
declares, several documents still mention the removed `tabulate` dependency,
and the documented upstream surface is incomplete. mlx-vlm PR #1713 is one
specific example: it fixes APC reuse for growing prepared prompts, but ordinary
`check_models` runs neither construct an APC manager nor reuse server prompt
caches, so the change does not affect them.

Nativ is an actively developed consumer of the same mlx-vlm server. Its useful
idioms for this project are evidence-backed model capability classification and
concise environment/context facts. Its live UI monitoring, private sensor
integration, and model-fit estimates do not belong in this batch harness.

## Goals

- Keep the default cached-model selection meaningful for a single-image
  description benchmark as mlx-vlm expands into other model kinds.
- Make every cached-repo exclusion visible and specific; no model should simply
  disappear from the run list.
- Retain the exclusion reason and its supporting evidence in machine-readable
  artifacts as well as console and dry-run output.
- Preserve forward compatibility by distinguishing confidently unsupported
  model kinds from unknown or newly introduced model layouts.
- Align declared runtime floors with the latest released mlx-vlm stack while
  continuing to support development against mlx-vlm main.
- Document exactly which mlx-vlm surfaces this project exercises and which it
  deliberately leaves to native mlx-vlm tools.
- Reuse existing dependencies and compact report machinery rather than adding a
  new diagnostics framework.

## Non-goals

- Turn `check_models` into an integration suite for every mlx-vlm endpoint.
- Benchmark text-only generation, embeddings, reranking, audio/video,
  image/video generation, image editing, fine-tuning, or distributed inference.
- Exercise continuous batching, APC, server prompt/vision caches, streaming
  protocols, structured output, tool calling, or realtime routes.
- Add speculative decoding to baseline comparisons. A drafter is a second model
  and changes timing, memory, provenance, and failure attribution.
- Reproduce Nativ's live telemetry UI, private sensor access, or predictive
  memory-fit heuristics.
- Reject an unfamiliar future model solely because its capability cannot be
  inferred.

## Chosen Design

### 1. Separate cache-layout eligibility from image-benchmark capability

Keep the existing server-style cache-layout test because it remains useful and
tracks mlx-vlm's local model discovery. Add a second, explicit classification
layer for the benchmark's actual requirement: whether the cached repo appears
to be a generative model that consumes image input.

The classifier will read only bounded metadata already present in the cached
snapshot, primarily `config.json` and, when present, `model_index.json`. It will
derive a small capability record from explicit configuration keys, model type,
architectures, task stamps, and strongly indicative descriptors. Relevant
classes include:

- image-consuming text generation;
- text-only generation;
- embedding;
- reranking;
- speculative drafter;
- image or video generation/editing;
- audio-only or other non-image generation;
- unknown.

The design adapts Nativ's convention of recording capabilities with evidence,
but narrows it to the decision this project needs. It will not copy Nativ's
complete evolving capability catalogue or large UI-oriented resolver.

The result is tri-state for this benchmark:

- `yes`: positive evidence that the model consumes image input and produces
  text;
- `no`: positive evidence that it is a different model kind or a text-only
  generator;
- `unknown`: insufficient or contradictory evidence.

Default discovery will run `yes` and `unknown` candidates. It will skip only
`no` candidates. This deliberately favours trying a new VLM over silently
excluding it.

Explicit `--models` selection remains an override. An explicitly requested
model runs even when cached metadata classifies it as non-image-capable, but the
classification and warning remain visible so the result is interpreted
correctly.

### 2. Make exclusions explicit everywhere they matter

Every cached repo omitted from a default run will have one or more concrete
reasons, for example:

- `cache layout: missing tokenizer_config.json`;
- `model purpose: text-only generation (model_type=afm7)`;
- `model purpose: sequence-classifier reranker`;
- `model purpose: embedding model (mlx_embeddings.kind=embedding)`;
- `model purpose: speculative drafter (speculators_model_type=...)`;
- `model purpose: image-generation pipeline, not image-to-text generation`.

Console selection output and `--dry-run` will list the skipped repo and the
human-readable reason. The same structured classification, evidence, and
decision will be retained in the run-level machine artifacts so downstream
tools can distinguish an intentional non-test from a crash or unavailable
model. Existing architecture support remains a separate fact: “installed
mlx-vlm has a package for this model type” is not proof that the model consumes
images.

Unknown classifications will also be visible, but as a warning attached to a
model that remains selected rather than as an exclusion.

### 3. Keep quality and maintainer assessments scoped to valid attempts

A confident non-image classification is a selection decision, not an unusable
VLM verdict. Skipped non-image models will not enter completion-quality tables,
regression comparisons, or mlx-vlm failure counts. If explicitly selected, the
capability evidence will accompany the result so maintainers can see that the
model was run outside the default benchmark scope.

No semantic image-quality classifier is added. Existing output observations
remain mechanical and operate only after generation.

### 4. Align the supported runtime stack

Raise the project and validation-policy floors to the latest released mlx-vlm
stack used by the direct API surface:

- `mlx>=0.32.0`;
- `mlx-vlm>=0.6.13`;
- `transformers>=5.14.0`.

Keep `mlx-lm>=0.31.3` as a direct project dependency because this project uses
it even though mlx-vlm main has removed its own mlx-lm dependency. Keep the
existing Hugging Face Hub and other project floors unless a used API requires a
change; version alignment means compatible justified minima, not mechanically
copying every latest installed version.

Remove `tabulate` from the environment-validation fallback and all current
dependency documentation. Report tables continue to use the project's compact
shared Markdown renderer and Rich for terminal output.

The existing release-versus-main API drift checks remain scoped to parameters
that `check_models` actually sends. Server-only and optional upstream arguments
must not create false drift failures.

### 5. Enrich environment evidence without a new dependency

Use the existing optional `psutil` integration to retain concise available
memory and swap-use facts when available. These are environmental context, not
proof of an MLX or model defect. Missing psutil continues to degrade gracefully.

This complements measured MLX peak/active/cache memory. It does not replace
those measurements with Nativ's predictive fit heuristic and does not add
private macOS sensor code.

### 6. Document upstream coverage and direction of travel

Add one authoritative coverage matrix to the detailed README and reference it
from the implementation guide. It will distinguish:

- exercised direct surfaces: model loading, adapters/revisions, image loading,
  chat templates, processor passthrough, still-image generation, sampling and
  penalty controls, thinking controls, KV-cache controls, timing, and allocator
  evidence;
- deliberately unexercised direct surfaces: multi-image, audio/video inputs,
  speculative decoding, prompt/vision cache reuse, image/video generation and
  image editing;
- server-only surfaces: OpenAI/Anthropic/realtime protocols, continuous
  batching, APC, queues, cache management, embeddings, reranking, audio routes,
  tools/structured outputs, health, and metrics;
- separate workflows: conversion, fine-tuning, evaluation suites, and
  distributed execution.

The matrix will explicitly record why mlx-vlm PR #1713 has no effect on normal
`check_models` runs: its APC prefix-reuse path requires cache reuse that this
direct, isolated, one-model-at-a-time harness does not enable.

The docs will also:

- stop describing the broad server cache layout as proof of VLM suitability;
- include the current thinking-mode and split KV-cache controls;
- remove stale `uv`, `tabulate`, and old dependency-floor references;
- explain why speculative decoding belongs in a future explicit benchmark lane
  rather than the baseline;
- update the cache-discovery agent skill and project instructions to use the
  layout-plus-capability terminology.

## Alternatives Considered

### Documentation-only alignment

This is the smallest patch, but it leaves default discovery able to run a
text-only model or reranker against an image and report the result as VLM
breakage. The upstream expansion has made that a correctness problem rather
than merely a documentation omission.

### Strict positive allowlist of known VLM architectures

This would keep non-image models out, but every new mlx-vlm architecture would
be skipped until this project was updated. That conflicts with the project's
role as an early compatibility and regression probe. The tri-state design runs
unknowns and excludes only confident negatives.

### Full mlx-vlm feature parity

Adding server, audio/video, embedding, reranking, image generation, continuous
batching, APC, and speculative-decoding tests would dilute the single-image
benchmark and multiply runtime, configuration, and reporting semantics. Native
mlx-vlm tests and focused repro commands are the appropriate tools for those
surfaces.

## Testing

Add focused tests to existing test modules for:

1. A normal VLM config with vision evidence is selected.
2. A known text-only generative config is skipped with an explicit reason.
3. Embedding, sequence-classifier reranker, drafter, and image-generation
   configurations are skipped with distinct explicit reasons.
4. An unfamiliar or incomplete config is classified unknown and still selected
   with a warning.
5. Explicit `--models` overrides the default capability exclusion while
   retaining the classification warning/evidence.
6. Dry-run and console skip messages name every excluded cached repo and reason.
7. Machine artifacts retain the structured classification and evidence without
   counting skipped non-image models as failed VLM attempts.
8. System memory/swap facts are captured when psutil exposes them and omitted
   cleanly when it does not.
9. Dependency-policy tests match `pyproject.toml`, mlx-vlm 0.6.13's used API
   contract, and the generated README dependency blocks.
10. Documentation checks reject stale `tabulate`, `uv`, old floor, and
    server-filter-as-VLM-proof wording.

After focused tests, run dependency synchronization, formatting, Ruff safe
fixes/lint, and the full quality gate in the prescribed conda environment.
Generated-report fixtures must write only to temporary or ignored output paths.

## Change Management

Record the user-visible discovery semantics, diagnostic evidence, dependency
floors, and documentation corrections under `CHANGELOG.md` `[Unreleased]`.
Do not regenerate or modify retained real-run `src/output/` artifacts; the next
real run will naturally publish classifications produced by the new code.
