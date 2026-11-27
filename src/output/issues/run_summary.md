# mlx-vlm compatibility findings across 42 cached vision-language models

**For mlx-vlm maintainers:** no crashes need action; 4 other results need
reproducing with mlx-vlm alone before anything is reported (2 of 4 unchanged
since the baseline).

**Completed models:** 34 with no mechanical observations, 4 with answer-format
or hint-reuse observations only (labels or form, or text reused from the
prompt's hints, which is allowed), and 4 with observations mlx-vlm can produce
(such as repetition, missing final answers or visible control tokens); only
those 4 are candidates for native reproduction.

## Run summary

- *Run started:* 2026-11-27 13:11:59 GMT
- *Run finished:* 2026-11-27 13:18:03 GMT
- *Run duration:* 6m 04s
- *Time by phase:* generation 258s, model load 81s, prompt prep 1s, cleanup
  5s, outside the model loop 18s
- *Evaluation lane:* blind
- *Prompt hints:* no description or keyword hints
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Input image:* JPEG, 640 x 480 pixels (0.3 MP), 0.2 MB
- *Models attempted:* 42
- *Sampling settings:* checkpoint generation_config.json values for 18 of 42
  completed models where the command line left them unset; harness defaults
  elsewhere
- *Completed:* 42
- *Crashed:* 0
- *Indeterminate:* 0
- *Crashes requiring action:* 0
- *Other results requiring review:* 4
- *Reached token limit:* 3 (3 with incomplete output)
- *Stopped early for repetition:* 0

**What this run measures.** This run records model responses to one shared
image and prompt (evaluation lane: blind). Mechanical checks are not
factual-accuracy judgments; inspect the image, prompt and final answers before
choosing a model. Results do not establish fitness for other tasks. Every
locally cached MLX vision-language model got the same image and prompt
(reproduced below) through mlx-vlm's generation pipeline.

<details>
<summary>Exact prompt sent to every model</summary>

```text
Create British-English catalogue metadata using only clearly visible facts. Omit uncertain details and unsupported identity, location, event, brand, species, period, or intent.

Write:
- a concrete 5-10-word title;
- a 1-2-sentence factual description of the main subject, setting, action, lighting, and distinctive details;
- 10-18 unique, comma-separated keywords.

Return exactly these three sections and nothing else:
Title:
Description:
Keywords:
```

</details>

## Since the baseline sweep

- *Baseline:* ec94ae2b:src/output/results.jsonl
- *Baseline run timestamp:* 2026-11-21 23:48:03 GMT
- *Baseline check_models:* 0.17.146 @ 6f3a5278e
- *Baseline mlx:* 0.32.5.dev20261121+8803ac3ca @ 8803ac3ca
- *Baseline mlx-vlm:* 0.8.0 @ f1618d6d5
- *Baseline transformers:* 5.24.0
- *Baseline python:* 3.14.7
- *Baseline hardware:* Apple M5 Max, 40 GPU cores, 128.0 GB RAM
- *Earlier retained runs on this mlx version:* 1 on
  0.33.0.dev20261127+266a9bd01
- *Models compared:* 41
- *Identical generated text:* 17 of 41 completed in both
- *Generated text changed:* 24 models (listed below)
- *Text changed, by decoding:* greedy 10 of 24; sampled, same settings and
  seed 14 of 17
- *Greedy text changed, by checkpoint quantization:* quantized 10 of 19; not
  quantized 0 of 5
- *Generation tok/s ratio (now/baseline):* 1.018 over 41 models; lowest 0.72
  (mlx-community/diffusiongemma-26B-A4B-it-mxfp8), highest 1.16
  (mlx-community/llm-jp-4-vl-9b-mlx-4bit)
- *Prefill tok/s ratio (now/baseline):* 0.798 over 41 models; lowest 0.13
  (mlx-community/MiniCPM-V-4.7-1B-4bit), highest 12.87
  (LiquidAI/LFM2.5-VL-450M-MLX-bf16)
- *Throughput noise band:* history (last 6 same-prompt runs, Tukey fence, at
  least ±10% of the median)

Run environment: current run detected GPU activity outside harness inference
at the checks around 3 of 42 models (up to 10%); baseline run detected GPU
activity outside harness inference at the checks around 7 of 41 models (up to
55%).

- New this run (no baseline): `thinkingmachines/Inkling-2-Mini`

| Model | Execution | Usability | Observation delta |
| --- | --- | --- | --- |
| mlx-community/gemma-4-26b-a4b-it-4bit | completed | no concerns detected → concerns detected | +duplicate keywords |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | completed | no concerns detected → concerns detected | +duplicate keywords |
| mlx-community/MiniCPM-o-4_5-4bit | completed | concerns detected → no concerns detected | -duplicate keywords |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | completed | concerns detected → no concerns detected | -duplicate keywords |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | completed | no concerns detected → major concerns | +duplicate keywords; +labelled fields not detected; +repeated text; +role tokens visible; +cut off at token limit; +control tokens visible |

<details>
<summary>Generated text changed for 24 models</summary>

| Model | Decoding | Weights | Shared prefix (chars) | Length (chars) | Output tokens | Prompt token count |
| --- | --- | --- | --- | --- | --- | --- |
| mlx-community/North-Micro-Vision-Instruct-4bit | sampled | 4-bit affine | 13 | 344 → 257 | 78 → 57 | same |
| mlx-community/granite-4.0-3b-vision-4bit | greedy | 4-bit affine | 29 | 228 → 280 | 57 → 68 | same |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | greedy | 4-bit affine | 11 | 435 → 422 | 102 → 103 | same |
| mlx-community/gemma-4-26b-a4b-it-4bit | sampled | 4-bit affine | 82 | 356 → 330 | 75 → 77 | same |
| mlx-community/gemma-4-e4b-it-4bit | sampled | 4-bit affine | 26 | 310 → 316 | 67 → 72 | same |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit | greedy | 4-bit affine | 95 | 342 → 315 | 73 → 77 | same |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | sampled | 4-bit affine | 134 | 386 → 453 | 83 → 97 | same |
| mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit | greedy | 4-bit affine | 245 | 360 → 397 | 81 → 87 | same |
| mlx-community/gemma-4-12B-it-4bit | sampled | 4-bit affine | 55 | 397 → 353 | 88 → 84 | same |
| mlx-community/MiniCPM-V-4.7-1B-4bit | greedy | 4-bit affine | 6 | 998 → 1,408 | 268 → 368 | same |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | greedy | 8-bit mxfp8 | 66 | 279 → 285 | 66 → 70 | same |
| nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit | sampled | 4-bit affine | 141 | 358 → 347 | 87 → 87 | same |
| mlx-community/Molmo2-8B-4bit | greedy | 4-bit affine | 78 | 431 → 569 | 95 → 124 | same |
| mlx-community/MiniCPM-o-4_5-4bit | sampled | 4-bit affine | 300 | 1,576 → 1,657 | 367 → 368 | same |
| mlx-community/InternVL3-14B-4bit | greedy | 4-bit | 325 | 409 → 395 | 90 → 87 | same |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | sampled | 4-bit affine | 79 | 1,959 → 1,867 | 540 → 495 | same |
| mlx-community/Mistral-Small-4-119B-2603-4bit | greedy | 4-bit affine | 174 | 378 → 374 | 82 → 82 | same |
| mlx-community/MiniCPM-V-4.7-35B-A3B-4bit | sampled | 4-bit affine | 218 | 1,858 → 2,043 | 431 → 477 | same |
| nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | sampled | 4-bit affine | 436 | 2,704 → 3,395 | 706 → 879 | same |
| mlx-community/Qwen3.5-35B-A3B-4bit | sampled | 4-bit affine | 580 | 3,469 → 2,858 | 898 → 768 | same |
| mlx-community/GLM-4.6V-Flash-4bit | sampled | 4-bit affine | 360 | 3,332 → 2,069 | 819 → 499 | same |
| mlx-community/Qwen3.6-35B-A3B-4bit | sampled | 4-bit affine | 131 | 2,382 → 3,709 | 653 → 909 | same |
| mlx-community/AREX-2-4bit | greedy | 4-bit affine | 702 | 3,301 → 2,291 | 865 → 551 | same |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | sampled | 4-bit affine | 0 | 400 → 4,343 | 884 → 1,000 | same |

</details>

Generation tok/s outside the expected band for 1 model (mlx changed since the
baseline: 8803ac3ca..266a9bd01):

| Model | Baseline tok/s | Now tok/s | Ratio | Expected band |
| --- | --- | --- | --- | --- |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | 144.9 | 116.4 | 0.80 | 122.1-149.3 (history, n=5) |

Largest prefill changes outside 0.85-1.15x (prefill seconds are prompt tokens
divided by prefill tok/s; one timed generation per sweep, so observations, not
demonstrated speed changes):

| Model | Baseline prefill s | Now prefill s | Prefill tok/s ratio |
| --- | --- | --- | --- |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | 0.80 | 0.06 | 12.87 |
| mlx-community/MiniCPM-V-4.7-1B-4bit | 0.11 | 0.83 | 0.13 |
| mlx-community/Molmo2-8B-4bit | 0.45 | 1.53 | 0.29 |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit | 0.23 | 0.63 | 0.37 |
| mlx-community/nanoLLaVA-1.5-4bit | 0.09 | 0.21 | 0.41 |

<details>
<summary>mlx: 48 upstream commit(s) since the baseline (8803ac3ca..266a9bd01)</summary>

- 266a9bd01 python: Return empty tuple when no arg in `broadcast_shapes`
  (#4971)
- c91080830 [Metal] Add int8 NAX path for affine quantized_matmul (#4802)
- 6ad98a1a5 docs: Fix missing scalar types in ops signature (#4978)
- 241f12bdc [Metal] Use the depthwise conv1d kernel for padded and flipped
  convs (#4962)
- 7cd4c4715 python: Add scalar support for grad (#4952)
- 4839b654a [Metal] Reduce SDPA vector kernel tile size for head dim &gt; 256
  (#4941)
- 77d58f725 Fix exclusive scan in place (#4966)
- b6be86cce Compiler fusion (#4953)
- 23276ecbe [CUDA] Grouped mm  (#4390)
- 8feb9fee6 [Metal] Fix FFT many axes (#4968)
- b19f29a31 Always clip gradients in fp32 (#4972)
- 5996d291e Minor bump to 0.33.0 (#4969)
- cfc675e9e Fix macOS 14 build (#4960)
- d20ac3bbc [CUDA] Synchronize LayerNorm reduction scratch reuse (#4866)
- 5598dfa0a Fix NaN propagation in CPU cummax and cummin for float16 and
  bfloat16 (#4912)
- ... and 33 more (see results.jsonl)

</details>

<details>
<summary>mlx-vlm: 60 upstream commit(s) since the baseline (f1618d6d5..e8035ae3b)</summary>

- e8035ae3 fix(cluster): reuse verified weights across metadata changes
  (#2833)
- 71f584bb Merge pull request #2830 from
  nassarofficial/feat/granite-docling-mtp
- c9853d08 Add Granite for Docling MTP drafter for speculative decoding
- 36494b42 Merge pull request #2819 from lucasnewman/version-081
- db0dbc76 Merge pull request #2806 from
  betizzel/fix/prompt-admission-releases-lazy-output
- c94e3fbe Merge pull request #2810 from
  betizzel/pr/inkling-tool-call-streaming
- c9ec8a6a Merge main and reconcile Inkling deltas with tool parse errors
- 948d2d79 Validate Inkling tool deltas and preserve token-limit completion
- 541d1a4d Make the APC logits lifetime test independent of allocator timing
- 906044bf Merge pull request #2814 from m4zar/qwen3-vl-tied-lm-head
- 8f35c6d6 Merge pull request #2820 from lucasnewman/image-gen-threading-fixes
- 766e21a0 Cover right-padded prefill in the APC logits lifetime test
- d7216529 Fix threading issues in various image gen models.
- d089582b Bump version to 0.8.1
- 466069f5 Merge pull request #2817 from
  Neywa-Labs/pc/report-malformed-tool-calls
- ... and 45 more (see results.jsonl)

</details>

<details>
<summary>mlx-vlm commits touching the affected models' architectures and their imports</summary>

- `mlx-community/North-Micro-Vision-Instruct-4bit` (`cohere_compass`; imports
  `base.py`, `cache.py`, `mlp.py`, `rope_utils.py`): f2956e68 Add a new cache
  type that preserves an attention sink after the newest key.
- `mlx-community/granite-4.0-3b-vision-4bit` (`granite4_vision`; imports
  `base.py`, `rope_utils.py`): no commits touched
  `mlx_vlm/models/granite4_vision/` or the entries it imports
- `mlx-community/Ministral-3-3B-Instruct-2512-4bit` (`mistral3`; imports
  `base.py`, `cache.py`, `mistral4`, `mistral_large4`, `pixtral`,
  `rope_utils.py`, `switch_layers.py`): f2956e68 Add a new cache type that
  preserves an attention sink after the newest key.
- `mlx-community/gemma-4-26b-a4b-it-4bit` (`gemma4`; imports `base.py`,
  `cache.py`, `qwen3_5`, `rope_utils.py`, `switch_layers.py`): f2956e68 Add a
  new cache type that preserves an attention sink after the newest key.
- `mlx-community/gemma-4-e4b-it-4bit` (`gemma4`; imports `base.py`,
  `cache.py`, `qwen3_5`, `rope_utils.py`, `switch_layers.py`): f2956e68 Add a
  new cache type that preserves an attention sink after the newest key.
- `mlx-community/LFM2.5-VL-3B-OptiQ-4bit` (`lfm2_vl`; imports `attention.py`,
  `base.py`, `interpolate.py`, `lfm2`, `mlp.py`): no commits touched
  `mlx_vlm/models/lfm2_vl/` or the entries it imports
- `mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit` (`qwen3_vl_moe`; imports
  `base.py`, `cache.py`, `mlp.py`, `qwen3_vl`, `rope_utils.py`,
  `switch_layers.py`): 925f4e58 Drop lm_head of tied-embedding checkpoints in
  qwen3_vl sanitize; f2956e68 Add a new cache type that preserves an attention
  sink after the newest key.
- `mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit` (`qwen3_omni_moe`; imports
  `base.py`, `cache.py`, `mlp.py`, `qwen3_vl`, `rope_utils.py`,
  `switch_layers.py`): 925f4e58 Drop lm_head of tied-embedding checkpoints in
  qwen3_vl sanitize; f2956e68 Add a new cache type that preserves an attention
  sink after the newest key.
- `mlx-community/gemma-4-12B-it-4bit` (`gemma4_unified`; imports `base.py`,
  `gemma4`): no commits touched `mlx_vlm/models/gemma4_unified/` or the
  entries it imports
- `mlx-community/MiniCPM-V-4.7-1B-4bit` (`minicpmv4_7`; imports `base.py`,
  `minicpmv4_6`, `qwen3_5_moe`): no commits touched
  `mlx_vlm/models/minicpmv4_7/` or the entries it imports
- `mlx-community/diffusiongemma-26B-A4B-it-mxfp8` (`diffusion_gemma`; imports
  `base.py`, `cache.py`, `diffusion_visualizer.py`, `gemma4`, `qwen3_vl`,
  `rope_utils.py`, `switch_layers.py`): 925f4e58 Drop lm_head of
  tied-embedding checkpoints in qwen3_vl sanitize; f2956e68 Add a new cache
  type that preserves an attention sink after the newest key.
- `nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit` (`mistral3`;
  imports `base.py`, `cache.py`, `mistral4`, `mistral_large4`, `pixtral`,
  `rope_utils.py`, `switch_layers.py`): f2956e68 Add a new cache type that
  preserves an attention sink after the newest key.
- `mlx-community/Molmo2-8B-4bit` (`molmo2`; imports `base.py`, `cache.py`,
  `rope_utils.py`): f2956e68 Add a new cache type that preserves an attention
  sink after the newest key.
- `mlx-community/MiniCPM-o-4_5-4bit` (`minicpmo`; imports `base.py`,
  `cache.py`, `llama`, `qwen3_vl`): 925f4e58 Drop lm_head of tied-embedding
  checkpoints in qwen3_vl sanitize; f2956e68 Add a new cache type that
  preserves an attention sink after the newest key.
- `mlx-community/InternVL3-14B-4bit` (`internvl_chat`; imports `base.py`,
  `cache.py`, `mlp.py`): f2956e68 Add a new cache type that preserves an
  attention sink after the newest key.
- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit` (`ernie4_5_moe_vl`;
  imports `base.py`, `cache.py`, `rope_utils.py`, `switch_layers.py`):
  f2956e68 Add a new cache type that preserves an attention sink after the
  newest key.
- `mlx-community/Mistral-Small-4-119B-2603-4bit` (`mistral3`; imports
  `base.py`, `cache.py`, `mistral4`, `mistral_large4`, `pixtral`,
  `rope_utils.py`, `switch_layers.py`): f2956e68 Add a new cache type that
  preserves an attention sink after the newest key.
- `mlx-community/MiniCPM-V-4.7-35B-A3B-4bit` (`minicpmv4_7`; imports
  `base.py`, `minicpmv4_6`, `qwen3_5_moe`): no commits touched
  `mlx_vlm/models/minicpmv4_7/` or the entries it imports
- `nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit`
  (`nemotron_h_nano_omni`; imports `base.py`, `interpolate.py`, `nemotron_h`):
  no commits touched `mlx_vlm/models/nemotron_h_nano_omni/` or the entries it
  imports
- `mlx-community/Qwen3.5-35B-A3B-4bit` (`qwen3_5_moe`; imports `base.py`,
  `qwen3_5`, `qwen3_vl`, `switch_layers.py`): 925f4e58 Drop lm_head of
  tied-embedding checkpoints in qwen3_vl sanitize
- `mlx-community/GLM-4.6V-Flash-4bit` (`glm4v`; imports `base.py`,
  `kernels.py`, `mlp.py`, `rope_utils.py`): no commits touched
  `mlx_vlm/models/glm4v/` or the entries it imports
- `mlx-community/Qwen3.6-35B-A3B-4bit` (`qwen3_5_moe`; imports `base.py`,
  `qwen3_5`, `qwen3_vl`, `switch_layers.py`): 925f4e58 Drop lm_head of
  tied-embedding checkpoints in qwen3_vl sanitize
- `mlx-community/AREX-2-4bit` (`qwen3_5`; imports `activations.py`, `base.py`,
  `cache.py`, `quantized_verifier.py`, `qwen3_vl`, `rope_utils.py`): 925f4e58
  Drop lm_head of tied-embedding checkpoints in qwen3_vl sanitize; f2956e68
  Add a new cache type that preserves an attention sink after the newest key.
- `mlx-community/Muse-Glimmer-30B-OptiQ-4bit` (`muse_glimmer`; imports
  `activations.py`, `base.py`, `cache.py`, `rope_utils.py`): f2956e68 Add a
  new cache type that preserves an attention sink after the newest key.

Context, not attribution: shared generation, sampling and processor code can
change a model without touching its own package.

</details>

Mechanical diff only: one image and one generation per model in each sweep,
with no warm-up or repeats. A difference shows that outputs or timings differ,
not why, whether it affects one model or many; attributing a cause needs
matched repeats or a native reproduction.

## Mechanical checks at a glance

Every attempted model, ordered by its mechanical checks, with counted facts.
"No concerns detected" is not an accuracy verdict: read the final answers in
the gallery. "Major concerns: generation" means generation itself failed (no
text, repeated text, or no final answer); "major concerns: answer format"
means an answer was generated but missed the requested form. Tokens in count
the image's tokens too, which drive prefill time; tokens out are those
generated (limit 1,000). *Native repro* says whether a person reran an
observation outside this harness on the same model revision and builds:
confirmed, not reproduced, or not yet checked. Recurring here alone is not
native confirmation. Keywords are counted from the answer's Keywords field.

| Model | Mechanical checks | Total | Gen tok/s | Peak GB | Prompt / output tok | Keywords | Observed | Native repro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ibm-granite/granite-vision-4.1-4b | no concerns detected | 3.15s | 63.7 tok/s | 9.7 | 734 / 56 | 10 | none | - |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | no concerns detected | 0.64s | 569 tok/s | 1.3 | 350 / 47 | 6 | none | - |
| mlx-community/AREX-2-4bit | no concerns detected | 21.25s | 32.4 tok/s | 17 | 461 / 551 | 15 | none | - |
| mlx-community/aya-vision-8b-4bit | no concerns detected | 4.01s | 102 tok/s | 6.6 | 2,909 / 82 | 10 | none | - |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | no concerns detected | 6.96s | 116 tok/s | 19 | 545 / 495 | 12 | none | - |
| mlx-community/gemma-4-12B-it-4bit | no concerns detected | 3.77s | 63.3 tok/s | 7.5 | 390 / 84 | 17 | none | - |
| mlx-community/gemma-4-e4b-it-4bit | no concerns detected | 2.74s | 124 tok/s | 6.0 | 386 / 72 | 14 | none | - |
| mlx-community/GLM-4.6V-Flash-4bit | no concerns detected | 8.12s | 82.6 tok/s | 8.0 | 495 / 499 | 11 | none | - |
| mlx-community/granite-4.0-3b-vision-4bit | no concerns detected | 1.48s | 186 tok/s | 4.7 | 734 / 68 | 10 | none | - |
| mlx-community/InternVL3-14B-4bit | no concerns detected | 5.38s | 55.4 tok/s | 10 | 3,435 / 87 | 17 | none | - |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit | no concerns detected | 1.89s | 220 tok/s | 3.6 | 344 / 77 | 10 | none | - |
| mlx-community/MiniCPM-o-4_5-4bit | no concerns detected | 5.09s | 105 tok/s | 7.0 | 173 / 368 | 16 | none | - |
| mlx-community/MiniCPM-V-4.7-1B-4bit | no concerns detected | 3.08s | 325 tok/s | 3.5 | 330 / 368 | 16 | none | - |
| mlx-community/MiniCPM-V-4.7-35B-A3B-4bit | no concerns detected | 8.77s | 87.7 tok/s | 24 | 330 / 477 | 16 | none | - |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | no concerns detected | 3.30s | 69.9 tok/s | 8.9 | 1,070 / 81 | 10 | none | - |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | no concerns detected | 1.61s | 212 tok/s | 3.7 | 1,069 / 103 | 13 | none | - |
| mlx-community/Mistral-Small-4-119B-2603-4bit | no concerns detected | 13.96s | 93.1 tok/s | 69 | 549 / 82 | 17 | none | - |
| mlx-community/Molmo2-8B-4bit | no concerns detected | 4.39s | 73.6 tok/s | 7.9 | 873 / 124 | 11 | none | - |
| mlx-community/MolmoPoint-8B-4bit | no concerns detected | 5.19s | 31.4 tok/s | 12 | 860 / 70 | 10 | none | - |
| mlx-community/North-Micro-Vision-Instruct-4bit | no concerns detected | 1.59s | 266 tok/s | 3.3 | 418 / 57 | 10 | none | - |
| mlx-community/Phi-3.5-vision-instruct-bf16 | no concerns detected | 3.13s | 59.0 tok/s | 9.1 | 883 / 94 | 19 | none | - |
| mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit | no concerns detected | 3.87s | 95.3 tok/s | 23 | 409 / 87 | 18 | none | - |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | no concerns detected | 3.12s | 135 tok/s | 19 | 409 / 97 | 18 | none | - |
| mlx-community/Qwen3-VL-8B-Instruct-4bit | no concerns detected | 1.81s | 106 tok/s | 6.8 | 409 / 60 | 10 | none | - |
| mlx-community/Qwen3.5-35B-A3B-4bit | no concerns detected | 8.90s | 129 tok/s | 21 | 419 / 768 | 18 | none | - |
| mlx-community/Qwen3.6-35B-A3B-4bit | no concerns detected | 9.97s | 130 tok/s | 21 | 419 / 909 | 17 | none | - |
| mlx-community/Qwen3.8-27B-nvfp4 | no concerns detected | 22.51s | 32.6 tok/s | 17 | 461 / 640 | 14 | none | - |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | no concerns detected | 1.76s | 125 tok/s | 5.6 | 1,204 / 31 | 4 | none | - |
| nativ-community/Mage-VL-OptiQ-4bit | no concerns detected | 1.66s | 153 tok/s | 5.3 | 420 / 88 | 10 | none | - |
| nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit | no concerns detected | 5.00s | 36.5 tok/s | 16 | 538 / 87 | 19 | none | - |
| nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | no concerns detected | 8.19s | 166 tok/s | 21 | 421 / 879 | 18 | none | - |
| nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 | no concerns detected | 57.73s | 31.0 tok/s | 81 | 420 / 797 | 15 | none | - |
| sahilchachra/LensVLM-9B-MXFP4 | no concerns detected | 3.59s | 103 tok/s | 6.7 | 419 / 198 | 11 | none | - |
| TechnoBaptist/Ternary-Bonsai-2-27B-mlx-2bit | no concerns detected | 15.97s | 42.7 tok/s | 11 | 461 / 517 | 14 | none | - |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | concerns detected | 6.20s | 47.9 tok/s | 28 | 386 / 70 | 15 | duplicate keywords | - |
| mlx-community/gemma-4-26b-a4b-it-4bit | concerns detected | 3.50s | 126 tok/s | 16 | 390 / 77 | 17 | duplicate keywords | - |
| mlx-community/FastVLM-0.5B-bf16 | major concerns: answer format | 1.45s | 398 tok/s | 1.8 | 120 / 32 | - | labelled fields not detected | - |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns: answer format | 2.41s | insufficient sample | 7.0 | 3,512 / 14 | - | control tokens visible; labelled fields not detected | not yet checked (earlier check out of date) |
| mlx-community/nanoLLaVA-1.5-4bit | major concerns: answer format | 1.09s | 400 tok/s | 1.3 | 116 / 97 | - | labelled fields not detected | - |
| thinkingmachines/Inkling-2-Mini | major concerns: answer format | 29.74s | 37.0 tok/s | 22 | 318 / 1,000 | - | control tokens visible; labelled fields not detected; cut off at token limit; role tokens visible | not yet checked |
| mlx-community/InternVL3_5-1B-4bit | major concerns: generation | 3.82s | 380 tok/s | 2.3 | 3,438 / 1,000 | 144 | repeated text; cut off at token limit; duplicate keywords | not yet checked |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | major concerns: generation | 44.14s | 25.0 tok/s | 25 | 545 / 1,000 | 42 | repeated text; control tokens visible; labelled fields not detected; cut off at token limit; role tokens visible; duplicate keywords | not yet checked |

## Native reproduction checks

Recorded in native_repro_notes.yaml for this run's model revisions, builds
(mlx 0.33.0.dev20261127+266a9bd01 @ 266a9bd01; mlx-vlm 0.8.1 @ e8035ae3b;
transformers 5.24.1) and experiment inputs (image, prompt, generation
settings, and each model's rendered prompt and effective generation
arguments). Earlier checks shown separately do not establish this result.

No recorded check applies to this run.

<details>
<summary>Earlier native checks (out of date)</summary>

| Model                                 | Observation            | Prior status | Prior stack                                                                                   | Checked            | Evidence                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
|---------------------------------------|------------------------|--------------|-----------------------------------------------------------------------------------------------|--------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | control tokens visible | confirmed    | mlx 0.32.5.dev20261030+1ba7f969a @ 1ba7f969a; mlx-vlm 0.7.11 @ 918763c68; transformers 5.22.1 | Claude, 2026-10-30 | python -m mlx_vlm.generate --model mlx-community/llm-jp-4-vl-9b-mlx-4bit --revision 9c056d48b --image &lt;original 63MP JPEG, sha256 ecdd79a7...&gt; --prompt &lt;recorded prompt&gt; --max-tokens 1000 --temperature 0.0 --top-p 1.0 --top-k 0 --seed 0 --prefill-step-size 2048 --trust-remote-code reproduced the harness output byte-for-byte: '<\|channel\|> analysis<\|message\|> ...' visible in decoded text, 29 tokens; <\|channel\|>(9)/<\|message\|>(12) are special:true in tokenizer.json but are not stripped, and the processor has no clean_output/Harmony channel parser. Rendered-prompt and effective-argument digests added 2026-10-30 from the harness record, not measured natively: the native flags above are the recorded effective arguments, the llm-jp template has no date or time variable, and the output matched byte-for-byte. |

</details>

## Human accuracy notes

0 of 42 outputs reviewed by a person for this exact image, model revision and
output (accuracy_notes.yaml); the others are unreviewed; 1 earlier note out of
date. Each note is a person's judgement against the image, not one of the
mechanical checks.

## Completed attempts requiring review

*History* dates each observation (a crash counts as one) over this model's
retained runs: 50 earlier retained runs from git
`HEAD:src/output/results.jsonl`, back to 2026-08-30; older runs were not read.
A run that did not attempt the model is skipped, and a report-only correction
of a run counts once. "Last N runs" and "N runs since" count consecutive runs
ending with this one; "first" is the earliest run read that showed it. The
table shows each model's longest-running observation; every observation's
dates are under *History by observation*.

| Model | Mechanical checks | Since baseline | History | Observed result | Evidence |
| --- | --- | --- | --- | --- | --- |
| mlx-community/InternVL3_5-1B-4bit | major concerns: generation | unchanged | duplicate keywords: 4 runs since 2026-11-16; +2 more, newest 2 runs since 2026-11-21 | Response repeats the same text; Response appears cut off at the token limit; Duplicate keywords: british english, british english examples of metadata, british english examples of details, british english examples of context, british english examples of information, british english examples of data, british english examples of content, british english examples of examples | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-internvl35-1b-4bit) |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | major concerns: generation | changed | duplicate keywords: back this run; +5 more, newest back this run | Response repeats the same text; Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description; Response appears cut off at the token limit; Conversation-role control tokens remain visible; Duplicate keywords: location, event, brand, species, period, or intent | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-muse-glimmer-30b-optiq-4bit) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns: answer format | unchanged | all 2: 41+ runs since 2026-09-12 or earlier | Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description, keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit) |
| thinkingmachines/Inkling-2-Mini | major concerns: answer format | new | all 4: not in earlier runs read | Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description, keywords; Response appears cut off at the token limit; Conversation-role control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-thinkingmachines-inkling-2-mini) |

<details>
<summary>History by observation</summary>

| Model | Observation | Persistence |
| --- | --- | --- |
| mlx-community/InternVL3_5-1B-4bit | duplicate keywords | last 4 runs (since 2026-11-16), first 2026-10-04 |
| mlx-community/InternVL3_5-1B-4bit | repeated text | last 2 runs (since 2026-11-21), first 2026-10-04 |
| mlx-community/InternVL3_5-1B-4bit | cut off at token limit | last 2 runs (since 2026-11-21), first 2026-10-16 |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | duplicate keywords | back this run, first 2026-09-06 |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | labelled fields not detected | back this run, first 2026-09-06 |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | repeated text | not in earlier runs read |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | role tokens visible | back this run, first 2026-09-06 |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | cut off at token limit | back this run, first 2026-09-06 |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | control tokens visible | back this run, first 2026-09-13 |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | labelled fields not detected | last 41+ runs (since 2026-09-12 or earlier) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | control tokens visible | last 41+ runs (since 2026-09-12 or earlier) |
| thinkingmachines/Inkling-2-Mini | labelled fields not detected | not in earlier runs read |
| thinkingmachines/Inkling-2-Mini | role tokens visible | not in earlier runs read |
| thinkingmachines/Inkling-2-Mini | cut off at token limit | not in earlier runs read |
| thinkingmachines/Inkling-2-Mini | control tokens visible | not in earlier runs read |

</details>

## Completions without detected concerns

34 completions without detected concerns; 4 more completed with answer-format or hint-reuse observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 640 x 480 pixels, 173,131 bytes
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: seed:* 0
- *Trust remote code:* true
- *check_models version:* 0.17.147
- *check_models revision:* ec94ae2b6badcfcbcd2759b02ae4ba7659f6e6fb
- *check_models source dirty:* false
- *mlx-vlm:* 0.8.1
- *mlx-vlm source revision:* e8035ae3b9740d5e3fc6cb0de4894132d60c8682
- *mlx:* 0.33.0.dev20261127+266a9bd01
- *mlx source revision:* 266a9bd01
- *transformers:* 5.24.1
- *macOS Version:* 27.2.1
- *GPU/Chip:* Apple M5 Max
- *Python Version:* 3.14.7

## Full artifacts

**Evidence links** target the repository's mutable main branch: they show this
run only once these artifacts are committed, and a later run's commit replaces
them. Before sharing upstream, pin them to the commit that published these
artifacts.

| Artifact | Link |
| --- | --- |
| Diagnostics | [diagnostics.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md) |
| Model gallery | [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md) |
| Results JSONL | [results.jsonl](https://github.com/jrp2014/check_models/blob/main/src/output/results.jsonl) |
| Environment | [environment.log](https://github.com/jrp2014/check_models/blob/main/src/output/environment.log) |
| Log | [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log) |
