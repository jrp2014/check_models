# mlx-vlm compatibility findings across 50 cached vision-language models

**For mlx-vlm maintainers:** no crashes need action; 7 other results need
reproducing with mlx-vlm alone before anything is reported (6 of 7 unchanged
since the baseline).

**What this run measures.** This run records model responses to one shared
image and prompt (evaluation lane: assisted). Mechanical checks are not
factual-accuracy judgments; inspect the image, prompt and final answers before
choosing a model. Results do not establish fitness for other tasks. Every
locally cached MLX vision-language model got the same image and prompt
(reproduced below) through mlx-vlm's generation pipeline.

## Run summary

- *Run started:* 2026-10-08 21:48:10 BST
- *Run finished:* 2026-10-08 21:59:07 BST
- *Run duration:* 10m 57s
- *Time by phase:* generation 513s, model load 90s, prompt prep 37s, cleanup
  6s, outside the model loop 17s
- *Evaluation lane:* assisted
- *Prompt hints:* the image's description and keyword hints were included in
  the prompt, so field content may be copied from them rather than seen
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Input image:* JPEG, 9,641 x 6,427 pixels (62.0 MP), 58.1 MB
- *Models attempted:* 50
- *Sampling settings:* checkpoint generation_config.json values for 23 of 50
  completed models where the command line left them unset; harness defaults
  elsewhere
- *Completed:* 50
- *Crashed:* 0
- *Indeterminate:* 0
- *Crashes requiring action:* 0
- *Other results requiring review:* 7
- *Reached token limit:* 2 (2 with incomplete output)
- *Stopped early for repetition:* 3

<details>
<summary>Exact prompt sent to every model</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-10-03 18:18:19 UTC+01:00

Descriptive hints:
- Description hint: UK Border Security Command patrol vessels, including the BSC Defender and BSC Volunteer, are moored side-by-side in Ramsgate Harbour, Kent, against a dramatic sunset and the town's cliffside skyline.
- Keyword hints: Border security vessels, Buildings, Coast, Dover, Dusk, England, Fleet, Harbor, Horizon, Kent, Lifebuoy, Marina, Maritime, Mooring, Patrol boat, Patrol boats, Pier, Port, Ramsgate, Reflection

Write:
- a concrete 5-10-word title;
- a 1-2-sentence factual description combining relevant context with the main visible subject, setting, action, lighting, and distinctive details;
- 10-18 unique, comma-separated keywords covering relevant context and visible details.

Return exactly these three sections and nothing else:
Title:
Description:
Keywords:
```

</details>

## Since the baseline sweep

- *Baseline:* ea5af6e8:src/output/results.jsonl
- *Baseline run timestamp:* 2026-10-04 21:56:04 BST
- *Baseline check_models:* 0.17.39 @ 728b90004
- *Baseline mlx:* 0.32.4.dev20261004+65be04707 @ 65be04707
- *Baseline mlx-vlm:* 0.7.4 @ 6ecadd767
- *Baseline transformers:* 5.18.0
- *Baseline python:* 3.14.7
- *Baseline hardware:* Apple M5 Max, 40 GPU cores, 128.0 GB RAM
- *Models compared:* 50
- *Identical generated text:* 27 of 50 completed in both
- *Generated text changed:* mlx-community/InternVL3_5-1B-4bit (shared prefix
  489 characters; length 765 → 768; same prompt token count),
  mlx-community/gemma-4-e4b-it-4bit (shared prefix 49 characters; length 371 →
  357; same prompt token count), mlx-community/gemma-4-26b-a4b-it-4bit (shared
  prefix 63 characters; length 504 → 502; same prompt token count),
  mlx-community/LFM2.5-VL-3B-OptiQ-4bit (shared prefix 114 characters; length
  374 → 366; same prompt token count),
  mlx-community/Ministral-3-3B-Instruct-2512-4bit (shared prefix 318
  characters; length 606 → 580; same prompt token count),
  mlx-community/granite-4.0-3b-vision-4bit (shared prefix 73 characters;
  length 457 → 411; same prompt token count),
  mlx-community/gemma-3n-E4B-it-4bit (shared prefix 307 characters; length 554
  → 571; same prompt token count), mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit
  (shared prefix 6 characters; length 577 → 562; same prompt token count),
  mlx-community/diffusiongemma-26B-A4B-it-mxfp8 (shared prefix 26 characters;
  length 413 → 411; same prompt token count),
  mlx-community/North-Micro-Vision-Instruct-4bit (shared prefix 82 characters;
  length 746 → 709; same prompt token count), mlx-community/MiniCPM-V-4.6-4bit
  (shared prefix 8 characters; length 473 → 2,535; same prompt token count),
  mlx-community/gemma-4-31b-it-4bit (shared prefix 197 characters; length 465
  → 488; same prompt token count), mlx-community/gemma-3-27b-it-qat-4bit
  (shared prefix 113 characters; length 609 → 592; same prompt token count),
  mlx-community/GLM-4.6V-Flash-4bit (shared prefix 355 characters; length 400
  → 429; same prompt token count),
  mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit (shared prefix 20
  characters; length 2,322 → 2,727; same prompt token count),
  mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit (shared prefix 26
  characters; length 548 → 523; same prompt token count),
  mlx-community/X-Reasoner-7B-8bit (shared prefix 532 characters; length 3,877
  → 848; same prompt token count),
  mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit (shared prefix 46 characters;
  length 609 → 606; same prompt token count),
  mlx-community/Step-3.7-Flash-oQ3e (shared prefix 26 characters; length 487 →
  495; same prompt token count), mlx-community/Qwen3.5-35B-A3B-4bit (shared
  prefix 7 characters; length 642 → 632; same prompt token count),
  mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit (shared prefix 112 characters;
  length 589 → 554; same prompt token count),
  mlx-community/Qwen3-VL-8B-Instruct-4bit (shared prefix 480 characters;
  length 518 → 513; same prompt token count),
  mlx-community/Muse-Glimmer-30B-OptiQ-4bit (shared prefix 2,086 characters;
  length 4,481 → 4,644; same prompt token count)
- *Text changed, by decoding:* greedy 10 of 29; sampled, same settings and
  seed 13 of 21
- *Generation tok/s ratio (now/baseline):* 1.009 (range 0.83-1.06, 47 models)
- *Prefill tok/s ratio (now/baseline):* 0.883 (range 0.12-1.16, 50 models)
- *Throughput noise band:* fixed ±15% fallback (insufficient history)

Run environment: current run on battery power for 50 of 50 models; baseline
run on battery power for 50 of 50 models.

| Model | Execution | Usability | Observation delta |
| --- | --- | --- | --- |
| vikhyatk/moondream2 | completed | major concerns | +prompt hint repeated |
| mlx-community/gemma-4-e4b-it-4bit | completed | concerns detected → no concerns detected | -prompt hint repeated |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit | completed | no concerns detected → concerns detected | +prompt hint repeated |
| mlx-community/granite-4.0-3b-vision-4bit | completed | no concerns detected → concerns detected | +prompt hint repeated |
| mlx-community/MiniCPM-V-4.6-4bit | completed | major concerns → no concerns detected | -incomplete thinking block |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | completed | no concerns detected → concerns detected | +duplicate keywords |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | completed | no concerns detected → concerns detected | +prompt hint repeated |
| mlx-community/X-Reasoner-7B-8bit | completed | major concerns | +stopped early: repeating; -cut off at token limit |

Generation tok/s outside the expected band for 1 model (mlx changed since the
baseline: 65be04707..3c40e8f92):

| Model | Baseline tok/s | Now tok/s | Ratio | Expected band |
| --- | --- | --- | --- | --- |
| mlx-community/nanoLLaVA-1.5-4bit | 340.8 | 283.2 | 0.83 | 289.7-392.0 (fallback) |

<details>
<summary>mlx: 22 upstream commit(s) since the baseline (65be04707..3c40e8f92)</summary>

- 3c40e8f92 Thin N GEMM kernels (#4640)
- 494c8a6c9 Keep split-K quantized matmul partials in float32 (#4641)
- 970295978 Add MLX_CONV_WINOGRAD switch to disable winograd for conv2d
  (#4639)
- 5dbe9991e Fix foreign buffer ownership (#4634)
- 6553c47ba Update acknowledgments for past pinv contribution (#4645)
- 53fbe5667 Fix cross-process race in CPU compile cache (#4638)
- 9fcc72bb0 Fix vmap of an inverse real FFT to an odd length (#4637)
- 34d14f9a2 Fix gpu boolean masking in 0 dim array (#4635)
- 9cb9c06fb Fix 32 bit dispatch grid overflow in Metal reductions (#4437)
- 0b37b0c71 Collapse the non sorted axes to pick the contiguous sort kernel
  (#4366)
- f14e6d2d6 Fix float16 overflow in clip_grad_norm (#4626)
- 017e1c3b9 Fix slicing's non contiguous start (#4599)
- 2b2bb4979 Promote integer inputs in logcumsumexp (#4625)
- a59cc2319 [metal] Add env var based residency refresh interval (#4633)
- ea7e39730 Tune M1 quantized matmul dispatch for six-row large shapes (#4629)
- ... and 7 more (see results.jsonl)

</details>

<details>
<summary>mlx-vlm: 31 upstream commit(s) since the baseline (6ecadd767..1cc602543)</summary>

- 1cc60254 Add Xing4.0 model (#2465)
- fdd94f39 Add support for Clef decision model (#2459)
- 8f8efa57 Merge pull request #2435 from
  atesahmet0/fix/sam-tracker-rope-reference
- 09b57dd7 Merge pull request #2450 from ishaanzee/rfdetr-fix-feature-indexes
- 7d0de225 Merge pull request #2452 from
  ishaanzee/rfdetr-fix-sanitize-idempotent
- d4eb8285 Merge pull request #2453 from ishaanzee/rfdetr-fix-small
- e2f08ab2 Merge pull request #2454 from ishaanzee/rfdetr-fix-bf16
- 4f4634bb Use fused SDPA in the RF-DETR backbone attention (#2443)
- d13bd3cc qwen3_5: use the packed gated delta kernel for multi-token calls
  (#2442)
- fec3f503 Add EmbeddingGemma 2 multimodal embedding support (#2446)
- 09bfebf0 Fix tracker RoPE axis order to match Meta checkpoints
- a2c3d64c Fix off-by-one RF-DETR backbone feature layers
- 99b01556 Compute grid_sample in float32 for half-precision inputs
- 816d4712 Fix RF-DETR small variant config
- bb85ca0c Make RF-DETR sanitize idempotent
- ... and 16 more (see results.jsonl)

</details>

<details>
<summary>mlx-vlm commits touching the affected models' architectures</summary>

- `vikhyatk/moondream2` (`moondream2`): no commits touched
  `mlx_vlm/models/moondream2/`
- `mlx-community/InternVL3_5-1B-4bit` (`internvl`): no commits touched
  `mlx_vlm/models/internvl/`
- `mlx-community/gemma-4-e4b-it-4bit` (`gemma4`): fec3f503 Add EmbeddingGemma
  2 multimodal embedding support (#2446)
- `mlx-community/gemma-4-26b-a4b-it-4bit` (`gemma4`): fec3f503 Add
  EmbeddingGemma 2 multimodal embedding support (#2446)
- `mlx-community/LFM2.5-VL-3B-OptiQ-4bit` (`lfm2_vl`): no commits touched
  `mlx_vlm/models/lfm2_vl/`
- `mlx-community/Ministral-3-3B-Instruct-2512-4bit` (`mistral3`): no commits
  touched `mlx_vlm/models/mistral3/`
- `mlx-community/granite-4.0-3b-vision-4bit` (`granite4_vision`): no commits
  touched `mlx_vlm/models/granite4_vision/`
- `mlx-community/gemma-3n-E4B-it-4bit` (`gemma3n`): no commits touched
  `mlx_vlm/models/gemma3n/`
- `mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit` (`qwen3_5_moe`): no commits
  touched `mlx_vlm/models/qwen3_5_moe/`
- `mlx-community/diffusiongemma-26B-A4B-it-mxfp8` (`diffusion_gemma`): no
  commits touched `mlx_vlm/models/diffusion_gemma/`
- `mlx-community/North-Micro-Vision-Instruct-4bit` (`cohere_compass`): no
  commits touched `mlx_vlm/models/cohere_compass/`
- `mlx-community/MiniCPM-V-4.6-4bit` (`minicpmv4_6`): no commits touched
  `mlx_vlm/models/minicpmv4_6/`
- `mlx-community/gemma-4-31b-it-4bit` (`gemma4`): fec3f503 Add EmbeddingGemma
  2 multimodal embedding support (#2446)
- `mlx-community/gemma-3-27b-it-qat-4bit` (`gemma3`): no commits touched
  `mlx_vlm/models/gemma3/`
- `mlx-community/GLM-4.6V-Flash-4bit` (`glm4v`): no commits touched
  `mlx_vlm/models/glm4v/`
- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit` (`ernie4_5_moe_vl`): no
  commits touched `mlx_vlm/models/ernie4_5_moe_vl/`
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` (`mistral3`): no
  commits touched `mlx_vlm/models/mistral3/`
- `mlx-community/X-Reasoner-7B-8bit` (`qwen2_5_vl`): no commits touched
  `mlx_vlm/models/qwen2_5_vl/`
- `mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit` (`qwen3_omni_moe`): no
  commits touched `mlx_vlm/models/qwen3_omni_moe/`
- `mlx-community/Step-3.7-Flash-oQ3e` (`step3p7`): no commits touched
  `mlx_vlm/models/step3p7/`
- `mlx-community/Qwen3.5-35B-A3B-4bit` (`qwen3_5_moe`): no commits touched
  `mlx_vlm/models/qwen3_5_moe/`
- `mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit` (`qwen3_vl_moe`): no commits
  touched `mlx_vlm/models/qwen3_vl_moe/`
- `mlx-community/Qwen3-VL-8B-Instruct-4bit` (`qwen3_vl`): bb5d9e75 Merge pull
  request #2413 from Marian2110/fix/qwen3-vl-video-timestamps; 4fcbdf32 Align
  timestamps with video metadata.; 7c25bbb3 Use the sampled frames for
  Qwen3-VL video timestamps
- `mlx-community/Muse-Glimmer-30B-OptiQ-4bit` (`muse_glimmer`): no commits
  touched `mlx_vlm/models/muse_glimmer/`

Context, not attribution: shared generation, sampling and processor code can
change a model without touching its own package.

</details>

Mechanical diff only: one image, temperature as configured; single-observation
flips on one model are usually run-to-run variance, broad shifts are not.

## Model quality at a glance

Every attempted model, ordered by its mechanical checks, with counted facts.
"No concerns detected" is not an accuracy verdict: read the final answers in
the gallery. Prompt tokens include the image tokens, which drive prefill time;
output tokens are the tokens generated (limit 1,000). Keywords are counted
from the answer's Keywords field, with how many appear verbatim in the
prompt's keyword hints. Hint text is the percent of the description's words
lying in four-word runs copied from the prompt's description hint (of the
whole answer when no field is labelled), rounded down; 80% or more is reported
as a repeated prompt hint.

| Model | Mechanical checks | Total | Gen tok/s | Peak GB | Prompt / output tok | Keywords | Hint text | Observed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | no concerns detected | 2.66s | 491 tok/s | 1.9 | 2,103 / 58 | 10 (6 from hints) | 0% | none |
| mlx-community/aya-vision-8b-4bit | no concerns detected | 5.01s | 103 tok/s | 6.5 | 2,070 / 132 | 20 (20 from hints) | 56% | none |
| mlx-community/gemma-3-27b-it-qat-4bit | no concerns detected | 8.89s | 31.1 tok/s | 17 | 572 / 138 | 18 (13 from hints) | 33% | none |
| mlx-community/gemma-4-12B-it-4bit | no concerns detected | 5.27s | 61.6 tok/s | 7.6 | 577 / 108 | 18 (12 from hints) | 31% | none |
| mlx-community/gemma-4-26b-a4b-it-4bit | no concerns detected | 4.92s | 124 tok/s | 16 | 577 / 106 | 15 (11 from hints) | 68% | none |
| mlx-community/gemma-4-31b-it-4bit | no concerns detected | 8.91s | 26.6 tok/s | 20 | 577 / 107 | 17 (15 from hints) | 62% | none |
| mlx-community/gemma-4-e4b-it-4bit | no concerns detected | 3.96s | 121 tok/s | 6.0 | 573 / 79 | 15 (8 from hints) | 60% | none |
| mlx-community/GLM-4.6V-Flash-4bit | no concerns detected | 9.37s | 79.1 tok/s | 8.7 | 6,339 / 98 | 12 (7 from hints) | 66% | none |
| mlx-community/Idefics3-8B-Llama3-bf16 | no concerns detected | 8.95s | 34.0 tok/s | 18 | 2,601 / 130 | 11 (1 from hints) | 60% | none |
| mlx-community/InternVL3-14B-4bit | no concerns detected | 6.25s | 57.8 tok/s | 10 | 2,091 / 113 | 19 (19 from hints) | 56% | none |
| mlx-community/InternVL3-8B-bf16 | no concerns detected | 5.97s | 37.7 tok/s | 17 | 2,091 / 80 | 15 (15 from hints) | 38% | none |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit | no concerns detected | 20.95s | 65.2 tok/s | 20 | 1,312 / 995 | 12 (10 from hints) | 14% | none |
| mlx-community/MiniCPM-o-4_5-4bit | no concerns detected | 3.42s | 105 tok/s | 7.0 | 369 / 88 | 18 (18 from hints) | 19% | none |
| mlx-community/MiniCPM-V-4.6-4bit | no concerns detected | 6.33s | 286 tok/s | 3.2 | 910 / 662 | 18 (8 from hints) | 16% | none |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | no concerns detected | 6.67s | 66.4 tok/s | 13 | 2,905 / 140 | 16 (3 from hints) | 33% | none |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | no concerns detected | 3.79s | 188 tok/s | 7.8 | 2,904 / 132 | 14 (1 from hints) | 26% | none |
| mlx-community/Molmo2-8B-4bit | no concerns detected | 6.86s | 73.4 tok/s | 8.1 | 1,502 / 148 | 24 (20 from hints) | 61% | none |
| mlx-community/North-Micro-Vision-Instruct-4bit | no concerns detected | 5.32s | 206 tok/s | 3.9 | 4,065 / 148 | 20 (20 from hints) | 46% | none |
| mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit | no concerns detected | 6.24s | 104 tok/s | 24 | 1,267 / 129 | 18 (17 from hints) | 51% | none |
| mlx-community/Phi-3.5-vision-instruct-bf16 | no concerns detected | 4.95s | 57.6 tok/s | 9.3 | 1,115 / 144 | 19 (14 from hints) | 44% | none |
| mlx-community/pixtral-12b-8bit | no concerns detected | 7.78s | 40.1 tok/s | 16 | 3,095 / 119 | 23 (20 from hints) | 67% | none |
| mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit | no concerns detected | 24.14s | 72.4 tok/s | 26 | 12,768 / 136 | 20 (20 from hints) | 48% | none |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | no concerns detected | 38.32s | 87.8 tok/s | 23 | 16,525 / 133 | 19 (16 from hints) | 31% | none |
| mlx-community/Qwen3-VL-8B-Instruct-4bit | no concerns detected | 38.79s | 71.0 tok/s | 11 | 16,525 / 112 | 16 (13 from hints) | 51% | none |
| mlx-community/Qwen3.5-35B-A3B-4bit | no concerns detected | 35.26s | 112 tok/s | 25 | 16,541 / 152 | 18 (11 from hints) | 22% | none |
| mlx-community/Qwen3.8-27B-nvfp4 | no concerns detected | 53.68s | 30.1 tok/s | 21 | 16,541 / 121 | 16 (10 from hints) | 30% | none |
| nativ-community/Mage-VL-OptiQ-4bit | no concerns detected | 4.62s | 129 tok/s | 5.4 | 4,188 / 118 | 20 (20 from hints) | 72% | none |
| nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit | no concerns detected | 7.73s | 36.9 tok/s | 18 | 1,251 / 114 | 18 (10 from hints) | 61% | none |
| nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | no concerns detected | 9.14s | 159 tok/s | 23 | 3,606 / 132 | 17 (13 from hints) | 75% | none |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | concerns detected | 10.36s | 30.6 tok/s | 23 | 2,372 / 117 | 19 (9 from hints) | 100% | prompt hint repeated |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | concerns detected | 6.96s | 69.0 tok/s | 28 | 573 / 90 | 15 (10 from hints) | 92% | prompt hint repeated |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | concerns detected | 9.89s | 127 tok/s | 19 | 1,617 / 758 | 43 (13 from hints) | 0% | duplicate keywords |
| mlx-community/GLM-4.6V-nvfp4 | concerns detected | 23.79s | 42.8 tok/s | 78 | 6,339 / 106 | 19 (19 from hints) | 100% | prompt hint repeated |
| mlx-community/granite-4.0-3b-vision-4bit | concerns detected | 3.83s | 180 tok/s | 4.8 | 1,366 / 104 | 15 (9 from hints) | 88% | prompt hint repeated |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit | concerns detected | 3.57s | 206 tok/s | 4.0 | 2,094 / 92 | 17 (16 from hints) | 89% | prompt hint repeated |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | concerns detected | 3.66s | 128 tok/s | 5.6 | 1,407 / 114 | 20 (20 from hints) | 100% | prompt hint repeated |
| mlx-community/Step-3.7-Flash-oQ3e | concerns detected | 37.61s | 53.3 tok/s | 92 | 3,468 / 112 | 20 (20 from hints) | 100% | prompt hint repeated |
| mlx-community/FastVLM-0.5B-bf16 | major concerns | 3.63s | 368 tok/s | 1.8 | 312 / 44 | - | 29% of answer | labelled fields not detected |
| mlx-community/gemma-3n-E4B-it-4bit | major concerns | 4.85s | 96.1 tok/s | 7.1 | 571 / 132 | 20 (7 from hints) | 11% | labelled fields not detected |
| mlx-community/granite-vision-3.2-2b-nvfp4 | major concerns | 5.18s | 145 tok/s | 4.3 | 5,615 / 86 | 18 (15 from hints) | - | labelled fields not detected; duplicate keywords |
| mlx-community/InternVL3_5-1B-4bit | major concerns | 2.62s | 394 tok/s | 2.1 | 2,094 / 200 | 59 (16 from hints) | 25% | repeated text; stopped early: repeating; duplicate keywords |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | major concerns | 37.87s | 20.2 tok/s | 15 | 290 / 667 | 45 (6 from hints) | 26% | repeated text; duplicate keywords |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns | 3.53s | 104 tok/s | 6.7 | 2,174 / 18 | - | 0% of answer | control tokens visible; labelled fields not detected |
| mlx-community/MolmoPoint-8B-4bit | major concerns | 9.49s | 31.8 tok/s | 13 | 3,104 / 113 | - | 32% of answer | labelled fields not detected |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | major concerns | 51.80s | 24.8 tok/s | 25 | 4,390 / 1,000 | 2 (0 from hints) | - | control tokens visible; labelled fields not detected; cut off at token limit; role tokens visible |
| mlx-community/nanoLLaVA-1.5-4bit | major concerns | 5.56s | 283 tok/s | 1.5 | 308 / 1,000 | - | - | repeated text; labelled fields not detected; cut off at token limit |
| mlx-community/Qwen2-VL-2B-mlx | major concerns | 39.16s | 125 tok/s | 9.4 | 16,536 / 225 | 22 (6 from hints) | 5% | stopped early: repeating; duplicate keywords |
| mlx-community/SmolVLM-256M-Instruct-4bit | major concerns | 2.40s | 542 tok/s | 1.1 | 1,186 / 22 | - | 0% of answer | labelled fields not detected |
| mlx-community/X-Reasoner-7B-8bit | major concerns | 18.02s | 59.7 tok/s | 14 | 16,536 / 200 | 53 (12 from hints) | 67% | repeated text; stopped early: repeating; duplicate keywords |
| vikhyatk/moondream2 | major concerns | 2.95s | 167 tok/s | 4.8 | 1,011 / 48 | - | 91% of answer | labelled fields not detected; prompt hint repeated |

## Observation clusters

Repeated mechanical observation signatures among results requiring review.

| Observed result | Models |
| --- | --- |
| Response repeats the same text; Generation was stopped early after sustained repeated output; Repeated keyword entries | 2 |
| Response repeats the same text; Required labelled fields not detected; Response appears cut off at the token limit | 1 |
| Response repeats the same text; Repeated keyword entries | 1 |
| Generation was stopped early after sustained repeated output; Repeated keyword entries | 1 |
| Unrecognised model control tokens remain visible; Required labelled fields not detected | 1 |
| Unrecognised model control tokens remain visible; Required labelled fields not detected; Response appears cut off at the token limit; Conversation-role control tokens remain visible | 1 |

## Completed attempts requiring review

*History* dates each observation (a crash counts as one) over this model's
retained runs: 30 earlier retained runs from git
`HEAD:src/output/results.jsonl`, back to 2026-08-30; older runs were not read.
A run that did not attempt the model is skipped, and a report-only correction
of a run counts once. "Last N runs" counts consecutive runs ending with this
one; "first" is the earliest run read that showed it.

| Model | Mechanical checks | Since baseline | History | Observed result | Evidence |
| --- | --- | --- | --- | --- | --- |
| mlx-community/InternVL3_5-1B-4bit | major concerns | unchanged | duplicate keywords: last 2 runs (since 2026-10-04); repeated text: last 2 runs (since 2026-10-04); stopped early: repeating: last 2 runs (since 2026-10-04) | Response repeats the same text; Generation was stopped early after sustained repeated output; Duplicate keywords: dover | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-internvl35-1b-4bit) |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | major concerns | unchanged | duplicate keywords: last 2 runs (since 2026-10-04); first 2026-09-25; repeated text: last 2 runs (since 2026-10-04); first 2026-09-18 | Response repeats the same text; Duplicate keywords: sea | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llama-32-11b-vision-instruct-8bit) |
| mlx-community/nanoLLaVA-1.5-4bit | major concerns | unchanged | labelled fields not detected: last 25+ runs (since 2026-09-06 or earlier); repeated text: last 2 runs (since 2026-10-04); cut off at token limit: last 2 runs (since 2026-10-04) | Response repeats the same text; Required labelled fields not detected: description, keywords; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-nanollava-15-4bit) |
| mlx-community/X-Reasoner-7B-8bit | major concerns | changed | duplicate keywords: last 15 runs (since 2026-09-25); first 2026-09-06; repeated text: last 2 runs (since 2026-10-04); first 2026-08-30; stopped early: repeating: back this run; first 2026-09-06 | Response repeats the same text; Generation was stopped early after sustained repeated output; Duplicate keywords: horizon | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-x-reasoner-7b-8bit) |
| mlx-community/Qwen2-VL-2B-mlx | major concerns | unchanged | duplicate keywords: last 2+ runs (since 2026-10-04 or earlier); stopped early: repeating: last 2+ runs (since 2026-10-04 or earlier) | Generation was stopped early after sustained repeated output; Duplicate keywords: lifeboat station | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-qwen2-vl-2b-mlx) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns | unchanged | labelled fields not detected: last 21+ runs (since 2026-09-12 or earlier); control tokens visible: last 21+ runs (since 2026-09-12 or earlier) | Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description, keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit) |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | major concerns | unchanged | labelled fields not detected: last 15 runs (since 2026-09-25); first 2026-09-06; role tokens visible: last 2 runs (since 2026-10-04); first 2026-09-06; cut off at token limit: last 16 runs (since 2026-09-18); first 2026-09-06; control tokens visible: last 2 runs (since 2026-10-04); first 2026-09-13 | Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description; Response appears cut off at the token limit; Conversation-role control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-muse-glimmer-30b-optiq-4bit) |

## Completions without detected concerns

29 completions without detected concerns; 14 more completed with prompt-compliance observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 9,641 x 6,427 pixels, 58,125,687 bytes
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: seed:* 0
- *Trust remote code:* true
- *check_models version:* 0.17.40
- *check_models revision:* ea5af6e8aaf310143be43466aa1d87d74bb42727
- *check_models source dirty:* false
- *mlx-vlm:* 0.7.7
- *mlx-vlm source revision:* 1cc6025432f61b146cb72c7516209f7259fd45d0
- *mlx:* 0.32.4.dev20261008+3c40e8f92
- *mlx source revision:* 3c40e8f92
- *transformers:* 5.19.0
- *macOS Version:* 27.0.1
- *GPU/Chip:* Apple M5 Max
- *Python Version:* 3.14.7

GitHub links target the repository's mutable main branch; they resolve to this
run's evidence only once these artifacts are committed, and a later run's
commit supersedes them. Pin links to that artifact commit when durable issue
evidence is required.

## Full artifacts

| Artifact | Link |
| --- | --- |
| Diagnostics | [diagnostics.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md) |
| Model gallery | [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md) |
| Results JSONL | [results.jsonl](https://github.com/jrp2014/check_models/blob/main/src/output/results.jsonl) |
| Environment | [environment.log](https://github.com/jrp2014/check_models/blob/main/src/output/environment.log) |
| Log | [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log) |
