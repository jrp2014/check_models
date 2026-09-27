# mlx-vlm compatibility findings across 50 cached vision-language models

**For mlx-vlm maintainers:** no crashes need action; 3 other results need
reproducing with mlx-vlm alone before anything is reported (all 3 unchanged
since the baseline).

**What this run measures.** This run records model responses to one shared
image and prompt (evaluation lane: assisted). Mechanical checks are not
factual-accuracy judgments; inspect the image, prompt and final answers before
choosing a model. Results do not establish fitness for other tasks. Every
locally cached MLX vision-language model got the same image and prompt
(reproduced below) through mlx-vlm's generation pipeline.

## Run summary

- *Run started:* 2026-09-27 22:31:57 BST
- *Run finished:* 2026-09-27 22:58:21 BST
- *Run duration:* 26m 23s
- *Time by phase:* generation 639s, model load 397s, prompt prep 48s, cleanup
  7s, outside the model loop 498s
- *Evaluation lane:* assisted
- *Prompt hints:* the image's description and keyword hints were included in
  the prompt, so field content may be copied from them rather than seen
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Input image:* JPEG, 9,805 x 6,538 pixels (64.1 MP), 54.2 MB
- *Models attempted:* 50
- *Sampling settings:* checkpoint generation_config.json values for 25 of 50
  completed models where the command line left them unset; harness defaults
  elsewhere
- *Completed:* 50
- *Crashed:* 0
- *Indeterminate:* 0
- *Crashes requiring action:* 0
- *Other results requiring review:* 3
- *Reached token limit:* 2 (2 with incomplete output)
- *Stopped early for repetition:* 0

<details>
<summary>Exact prompt sent to every model</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-09-26 17:36:29 UTC+01:00
- GPS: 50.682100°N, 3.466600°W

Descriptive hints:
- Description hint: A blue and light blue motor cabin cruiser with a tan canopy moored at a marina dock alongside other sailboats and leisure boats on calm water, casting clear reflections in the early evening light.
- Keyword hints: Boat, Boat canopy, Boat fender, Boating, Cabin cruiser, Calm Water, Dock, Harbor, Marina, Mast, Mooring, Motorboat, Nautical, Reflection, Rope, Sailboat, Sailing, Water reflection, Watercraft, Yacht

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

- *Baseline:* f4f9f6c0:src/output/results.jsonl
- *Baseline run timestamp:* 2026-09-27 21:23:48 BST
- *Baseline check_models:* 0.17.38 @ de73403c1
- *Baseline mlx:* 0.32.3.dev20260927+09e67c686 @ 09e67c686
- *Baseline mlx-vlm:* 0.7.3 @ 967bf90b8
- *Baseline transformers:* 5.17.0
- *Baseline python:* 3.14.7
- *Baseline hardware:* Apple M5 Max, 40 GPU cores, 128.0 GB RAM
- *Models compared:* 50
- *Identical generated text:* 50 of 50 completed in both
- *Text changed, by decoding:* greedy 0 of 27; sampled, same settings and seed
  0 of 23
- *Generation tok/s ratio (now/baseline):* 0.974 (range 0.61-1.19, 49 models)
- *Prefill tok/s ratio (now/baseline):* 1.005 (range 0.89-1.53, 49 models)
- *Throughput noise band:* fixed ±15% fallback (insufficient history)

Run environment: current run on battery power for 1 of 50 models; current run
slept or was suspended during 1 model(s) (LiquidAI/LFM2.5-VL-450M-MLX-bf16);
those are excluded from throughput comparison; baseline run on battery power
for 6 of 50 models.

No execution, usability, or observation-set changes against the baseline.

mlx changed since the baseline (09e67c686..02ce1fb6a). A new mlx build
compiles each Metal shader the first time it is used, so the first run after a
rebuild reads slow, most of all for small, fast models; rerun before reading
these as regressions. Generation tok/s outside the expected band for 7 models:

| Model | Baseline tok/s | Now tok/s | Ratio | Expected band |
| --- | --- | --- | --- | --- |
| mlx-community/nanoLLaVA-1.5-4bit | 334.5 | 221.5 | 0.66 | 284.3-384.7 (fallback) |
| mlx-community/SmolVLM-256M-Instruct-4bit | 523.8 | 321.4 | 0.61 | 445.2-602.3 (fallback) |
| mlx-community/gemma-4-e4b-it-4bit | 103.0 | 122.9 | 1.19 | 87.5-118.4 (fallback) |
| mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit | 99.0 | 79.8 | 0.81 | 84.1-113.8 (fallback) |
| mlx-community/MiniCPM-V-4.6-4bit | 294.7 | 237.1 | 0.80 | 250.5-338.9 (fallback) |
| mlx-community/North-Micro-Vision-Instruct-4bit | 210.4 | 151.0 | 0.72 | 178.8-241.9 (fallback) |
| mlx-community/Qwen3.5-35B-A3B-4bit | 105.0 | 73.7 | 0.70 | 89.3-120.8 (fallback) |

<details>
<summary>mlx: 1 upstream commit(s) since the baseline (09e67c686..02ce1fb6a)</summary>

- 02ce1fb6a Add fused Metal kernels for fast.cross_entropy (#4520)

</details>

Mechanical diff only: one image, temperature as configured; single-observation
flips on one model are usually run-to-run variance, broad shifts are not.

## Model quality at a glance

Every attempted model, ordered by its mechanical checks, with counted facts.
"No concerns detected" is not an accuracy verdict: read the final answers in
the gallery. Prompt tokens include the image tokens, which drive prefill time.
Keywords are counted from the answer's Keywords field, with how many appear
verbatim in the prompt's keyword hints.

| Model | Mechanical checks | Total | Gen tok/s | Peak GB | Prompt tok | Keywords | Observed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mlx-community/aya-vision-8b-4bit | no concerns detected | 5.51s | 102 tok/s | 6.5 | 2,098 | 17 (10 from hints) | none |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | no concerns detected | 7.15s | 57.2 tok/s | 28 | 597 | 14 (12 from hints) | none |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | no concerns detected | 8.87s | 108 tok/s | 19 | 1,646 | 15 (12 from hints) | none |
| mlx-community/gemma-4-12B-it-4bit | no concerns detected | 5.82s | 61.2 tok/s | 7.6 | 601 | 20 (20 from hints) | none |
| mlx-community/gemma-4-26b-a4b-it-4bit | no concerns detected | 5.64s | 110 tok/s | 16 | 601 | 17 (15 from hints) | none |
| mlx-community/gemma-4-31b-it-4bit | no concerns detected | 9.72s | 26.9 tok/s | 20 | 601 | 17 (17 from hints) | none |
| mlx-community/granite-4.0-3b-vision-4bit | no concerns detected | 3.73s | 174 tok/s | 4.6 | 1,384 | 10 (3 from hints) | none |
| mlx-community/InternVL3-14B-4bit | no concerns detected | 6.26s | 52.1 tok/s | 10 | 2,120 | 20 (20 from hints) | none |
| mlx-community/InternVL3-8B-bf16 | no concerns detected | 6.28s | 35.2 tok/s | 17 | 2,120 | 13 (13 from hints) | none |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | no concerns detected | 10.48s | 20.9 tok/s | 15 | 309 | 10 (2 from hints) | none |
| mlx-community/MiniCPM-o-4_5-4bit | no concerns detected | 3.54s | 102 tok/s | 7.0 | 398 | 16 (11 from hints) | none |
| mlx-community/MiniCPM-V-4.6-4bit | no concerns detected | 5.33s | 237 tok/s | 3.3 | 938 | 17 (12 from hints) | none |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | no concerns detected | 8.18s | 63.8 tok/s | 13 | 2,935 | 20 (7 from hints) | none |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | no concerns detected | 4.36s | 175 tok/s | 7.8 | 2,934 | 13 (2 from hints) | none |
| mlx-community/Molmo2-8B-4bit | no concerns detected | 6.38s | 70.1 tok/s | 8.6 | 1,531 | 36 (19 from hints) | none |
| mlx-community/North-Micro-Vision-Instruct-4bit | no concerns detected | 5.94s | 151 tok/s | 3.9 | 4,091 | 19 (19 from hints) | none |
| mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit | no concerns detected | 6.66s | 79.8 tok/s | 24 | 1,295 | 18 (18 from hints) | none |
| mlx-community/Qwen3-VL-2B-Thinking-bf16 | no concerns detected | 28.60s | 84.5 tok/s | 8.4 | 16,556 | 16 (14 from hints) | none |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | no concerns detected | 42.99s | 75.0 tok/s | 23 | 16,554 | 20 (20 from hints) | none |
| mlx-community/Qwen3-VL-32B-Instruct-4bit | no concerns detected | 76.85s | 20.5 tok/s | 26 | 16,554 | 18 (17 from hints) | none |
| mlx-community/Qwen3-VL-8B-Instruct-4bit | no concerns detected | 43.29s | 69.1 tok/s | 11 | 16,554 | 18 (15 from hints) | none |
| mlx-community/Qwen3.5-35B-A3B-4bit | no concerns detected | 44.16s | 73.7 tok/s | 25 | 16,569 | 19 (19 from hints) | none |
| mlx-community/Qwen3.5-9B-MLX-4bit | no concerns detected | 37.66s | 89.3 tok/s | 11 | 16,569 | 15 (7 from hints) | none |
| mlx-community/Qwen3.8-27B-nvfp4 | no concerns detected | 56.30s | 29.4 tok/s | 21 | 16,569 | 16 (5 from hints) | none |
| nativ-community/MiMo-V2.6-Distill-Qwen-9B-MLX-4bit | no concerns detected | 34.47s | 87.4 tok/s | 11 | 16,566 | 15 (1 from hints) | none |
| nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit | no concerns detected | 8.84s | 35.9 tok/s | 18 | 1,281 | 18 (11 from hints) | none |
| nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | no concerns detected | 9.20s | 135 tok/s | 23 | 3,636 | 16 (16 from hints) | none |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | concerns detected | 302.10s | 462 tok/s | 1.9 | 2,123 | 14 (8 from hints) | duplicate keywords; prompt hint repeated |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | concerns detected | 10.78s | 29.0 tok/s | 23 | 2,402 | 20 (20 from hints) | prompt hint repeated |
| mlx-community/gemma-3-27b-it-qat-4bit | concerns detected | 9.22s | 30.8 tok/s | 17 | 596 | 18 (14 from hints) | unsupplied place name |
| mlx-community/gemma-4-e4b-it-4bit | concerns detected | 4.09s | 123 tok/s | 5.9 | 597 | 15 (14 from hints) | duplicate keywords |
| mlx-community/GLM-4.6V-Flash-4bit | concerns detected | 9.27s | 70.4 tok/s | 8.7 | 6,393 | 9 (7 from hints) | prompt hint repeated |
| mlx-community/GLM-4.6V-nvfp4 | concerns detected | 23.58s | 39.0 tok/s | 78 | 6,393 | 20 (20 from hints) | prompt hint repeated |
| mlx-community/Idefics3-8B-Llama3-bf16 | concerns detected | 7.89s | 32.7 tok/s | 18 | 2,620 | 10 (1 from hints) | prompt hint repeated |
| mlx-community/Phi-3.5-vision-instruct-bf16 | concerns detected | 4.24s | 55.6 tok/s | 9.3 | 1,144 | 10 (8 from hints) | prompt hint repeated |
| mlx-community/pixtral-12b-8bit | concerns detected | 7.91s | 39.5 tok/s | 16 | 3,125 | 20 (20 from hints) | prompt hint repeated |
| mlx-community/Qwen2-VL-7B-Instruct-4bit | concerns detected | 43.34s | 87.1 tok/s | 9.3 | 16,565 | 20 (20 from hints) | prompt hint repeated |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | concerns detected | 3.96s | 128 tok/s | 5.6 | 1,438 | 19 (19 from hints) | prompt hint repeated |
| mlx-community/Step-3.7-Flash-oQ3e | concerns detected | 39.70s | 49.5 tok/s | 92 | 3,498 | 20 (20 from hints) | prompt hint repeated |
| mlx-community/X-Reasoner-7B-8bit | concerns detected | 20.19s | 57.7 tok/s | 14 | 16,565 | 33 (18 from hints) | duplicate keywords |
| nativ-community/Mage-VL-OptiQ-4bit | concerns detected | 5.21s | 129 tok/s | 5.4 | 4,217 | 20 (20 from hints) | prompt hint repeated |
| mlx-community/FastVLM-0.5B-bf16 | major concerns | 3.36s | 320 tok/s | 2.2 | 341 | - | labelled fields not detected |
| mlx-community/gemma-3n-E4B-it-4bit | major concerns | 5.20s | 74.5 tok/s | 7.2 | 595 | - | labelled fields not detected |
| mlx-community/granite-vision-3.2-2b-nvfp4 | major concerns | 5.56s | 133 tok/s | 4.4 | 5,666 | 20 (20 from hints) | labelled fields not detected |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit | major concerns | 21.43s | 59.4 tok/s | 20 | 1,331 | 100 (5 from hints) | repeated text; labelled fields not detected; cut off at token limit; incomplete thinking block; duplicate keywords |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit | major concerns | 3.91s | 203 tok/s | 4.0 | 2,115 | - | labelled fields not detected |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns | 3.83s | 107 tok/s | 6.7 | 2,201 | - | control tokens visible; labelled fields not detected |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | major concerns | 52.96s | 24.5 tok/s | 25 | 4,410 | 20 (0 from hints) | control tokens visible; labelled fields not detected; cut off at token limit; role tokens visible |
| mlx-community/nanoLLaVA-1.5-4bit | major concerns | 2.37s | 222 tok/s | 1.8 | 337 | - | labelled fields not detected |
| mlx-community/SmolVLM-256M-Instruct-4bit | major concerns | 2.60s | 321 tok/s | 1.1 | 1,217 | - | labelled fields not detected |

## Completed attempts requiring review

| Model | Mechanical checks | Since baseline | Observed result | Evidence |
| --- | --- | --- | --- | --- |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit | major concerns | unchanged | Response repeats the same text; Required labelled fields not detected: title; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Duplicate keywords: dock, marina, calm water, reflections, early evening, blue hull, tan canopy, blue and light blue, marina dock, leisure boats, early evening light, blue and light blue hull, other sailboats, correct conflicts, moored | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-8bit) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns | unchanged | Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description, keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit) |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | major concerns | unchanged | Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description; Response appears cut off at the token limit; Conversation-role control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-muse-glimmer-30b-optiq-4bit) |

## Completions without detected concerns

27 completions without detected concerns; 20 more completed with prompt-compliance observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 9,805 x 6,538 pixels, 54,173,041 bytes
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: seed:* 0
- *Trust remote code:* true
- *check_models version:* 0.17.38
- *check_models revision:* f4f9f6c0e7754fc8f4481b1c2bda07ba34909dab
- *check_models source dirty:* false
- *mlx-vlm:* 0.7.3
- *mlx-vlm source revision:* 967bf90b8e7ae6e5110b6da87134f30d94591d2e
- *mlx:* 0.32.3.dev20260927+02ce1fb6a
- *mlx source revision:* 02ce1fb6a
- *transformers:* 5.17.0
- *macOS Version:* 27.0
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
