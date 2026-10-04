# mlx-vlm compatibility findings across 52 cached vision-language models

**For mlx-vlm maintainers:** no crashes need action; 2 other results need
reproducing with mlx-vlm alone before anything is reported (all 2 unchanged
since the baseline).

**What this run measures.** This run records model responses to one shared
image and prompt (evaluation lane: assisted). Mechanical checks are not
factual-accuracy judgments; inspect the image, prompt and final answers before
choosing a model. Results do not establish fitness for other tasks. Every
locally cached MLX vision-language model got the same image and prompt
(reproduced below) through mlx-vlm's generation pipeline.

## Run summary

- *Run started:* 2026-10-04 00:57:49 BST
- *Run finished:* 2026-10-04 01:10:54 BST
- *Run duration:* 13m 04s
- *Time by phase:* generation 633s, model load 93s, prompt prep 40s, cleanup
  6s, outside the model loop 17s
- *Evaluation lane:* assisted
- *Prompt hints:* the image's description and keyword hints were included in
  the prompt, so field content may be copied from them rather than seen
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Input image:* JPEG, 9,805 x 6,538 pixels (64.1 MP), 54.2 MB
- *Models attempted:* 52
- *Sampling settings:* checkpoint generation_config.json values for 25 of 52
  completed models where the command line left them unset; harness defaults
  elsewhere
- *Completed:* 52
- *Crashed:* 0
- *Indeterminate:* 0
- *Crashes requiring action:* 0
- *Other results requiring review:* 2
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

- *Baseline:* 894cc3a8:src/output/results.jsonl
- *Baseline run timestamp:* 2026-10-04 00:38:52 BST
- *Baseline check_models:* 0.17.39 @ 7859973d5
- *Baseline mlx:* 0.32.4.dev20261003+0e3ff3643 @ 0e3ff3643
- *Baseline mlx-vlm:* 0.7.4 @ 6ecadd767
- *Baseline transformers:* 5.18.0
- *Baseline python:* 3.14.7
- *Baseline hardware:* Apple M5 Max, 40 GPU cores, 128.0 GB RAM
- *Models compared:* 52
- *Identical generated text:* 52 of 52 completed in both
- *Text changed, by decoding:* greedy 0 of 29; sampled, same settings and seed
  0 of 23
- *Generation tok/s ratio (now/baseline):* 1.014 (range 0.95-1.78, 51 models)
- *Prefill tok/s ratio (now/baseline):* 1.047 (range 0.95-1.59, 51 models)
- *Throughput noise band:* history (last 8 same-prompt runs, Tukey fence, at
  least ±10% of the median)

Run environment: baseline run slept or was suspended during 1 model(s)
(mlx-community/Qwen3.8-27B-nvfp4); those are excluded from throughput
comparison.

- In baseline, no longer in the cache:
  `mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit`

No execution, usability, or observation-set changes against the baseline.

Generation tok/s outside the expected band for 1 model:

| Model | Baseline tok/s | Now tok/s | Ratio | Expected band |
| --- | --- | --- | --- | --- |
| mlx-community/granite-vision-3.2-2b-nvfp4 | 141.7 | 144.3 | 1.02 | 117.5-143.6 (history, n=8) |

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
| mlx-community/aya-vision-8b-4bit | no concerns detected | 5.15s | 102 tok/s | 6.5 | 2,098 | 17 (10 from hints) | none |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | no concerns detected | 6.35s | 64.4 tok/s | 28 | 597 | 14 (12 from hints) | none |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | no concerns detected | 8.51s | 109 tok/s | 19 | 1,646 | 15 (12 from hints) | none |
| mlx-community/gemma-4-12B-it-4bit | no concerns detected | 5.26s | 62.0 tok/s | 7.6 | 601 | 20 (20 from hints) | none |
| mlx-community/gemma-4-26b-a4b-it-4bit | no concerns detected | 5.13s | 110 tok/s | 16 | 601 | 17 (15 from hints) | none |
| mlx-community/gemma-4-31b-it-4bit | no concerns detected | 9.20s | 26.7 tok/s | 20 | 601 | 17 (17 from hints) | none |
| mlx-community/granite-4.0-3b-vision-4bit | no concerns detected | 3.27s | 178 tok/s | 4.6 | 1,384 | 14 (8 from hints) | none |
| mlx-community/InternVL3-14B-4bit | no concerns detected | 5.71s | 57.7 tok/s | 10 | 2,120 | 20 (20 from hints) | none |
| mlx-community/InternVL3-8B-bf16 | no concerns detected | 5.90s | 37.6 tok/s | 17 | 2,120 | 13 (13 from hints) | none |
| mlx-community/InternVL3_5-1B-4bit | no concerns detected | 2.59s | 344 tok/s | 2.1 | 2,123 | 23 (19 from hints) | none |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | no concerns detected | 9.93s | 21.2 tok/s | 15 | 309 | 10 (2 from hints) | none |
| mlx-community/MiniCPM-o-4_5-4bit | no concerns detected | 3.39s | 104 tok/s | 7.0 | 398 | 16 (11 from hints) | none |
| mlx-community/MiniCPM-V-4.6-4bit | no concerns detected | 5.09s | 242 tok/s | 3.2 | 938 | 17 (12 from hints) | none |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | no concerns detected | 8.10s | 64.3 tok/s | 13 | 2,935 | 20 (7 from hints) | none |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | no concerns detected | 4.24s | 181 tok/s | 7.8 | 2,934 | 13 (2 from hints) | none |
| mlx-community/Molmo2-8B-4bit | no concerns detected | 5.99s | 70.6 tok/s | 8.5 | 1,531 | 36 (19 from hints) | none |
| mlx-community/North-Micro-Vision-Instruct-4bit | no concerns detected | 5.51s | 156 tok/s | 3.9 | 4,091 | 19 (19 from hints) | none |
| mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit | no concerns detected | 6.38s | 73.7 tok/s | 24 | 1,295 | 18 (18 from hints) | none |
| mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit | no concerns detected | 26.81s | 62.7 tok/s | 26 | 12,797 | 20 (20 from hints) | none |
| mlx-community/Qwen3-VL-2B-Thinking-bf16 | no concerns detected | 28.64s | 85.5 tok/s | 8.4 | 16,556 | 16 (14 from hints) | none |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | no concerns detected | 39.46s | 76.7 tok/s | 23 | 16,554 | 20 (20 from hints) | none |
| mlx-community/Qwen3-VL-32B-Instruct-4bit | no concerns detected | 72.38s | 20.3 tok/s | 26 | 16,554 | 18 (17 from hints) | none |
| mlx-community/Qwen3-VL-8B-Instruct-4bit | no concerns detected | 38.57s | 68.4 tok/s | 11 | 16,554 | 18 (15 from hints) | none |
| mlx-community/Qwen3.5-35B-A3B-4bit | no concerns detected | 36.36s | 72.8 tok/s | 25 | 16,569 | 19 (19 from hints) | none |
| mlx-community/Qwen3.5-9B-MLX-4bit | no concerns detected | 36.43s | 91.2 tok/s | 11 | 16,569 | 15 (7 from hints) | none |
| mlx-community/Qwen3.8-27B-nvfp4 | no concerns detected | 55.92s | 29.6 tok/s | 21 | 16,569 | 16 (5 from hints) | none |
| nativ-community/MiMo-V2.6-Distill-Qwen-9B-MLX-4bit | no concerns detected | 33.02s | 88.4 tok/s | 11 | 16,566 | 15 (1 from hints) | none |
| nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit | no concerns detected | 8.40s | 36.9 tok/s | 18 | 1,281 | 18 (11 from hints) | none |
| nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | no concerns detected | 8.81s | 136 tok/s | 23 | 3,636 | 17 (16 from hints) | none |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | concerns detected | 2.21s | 470 tok/s | 1.9 | 2,123 | 14 (8 from hints) | duplicate keywords; prompt hint repeated |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | concerns detected | 10.29s | 30.4 tok/s | 23 | 2,402 | 20 (20 from hints) | prompt hint repeated |
| mlx-community/gemma-3-27b-it-qat-4bit | concerns detected | 8.69s | 30.8 tok/s | 17 | 596 | 18 (14 from hints) | unsupplied place name |
| mlx-community/gemma-4-e4b-it-4bit | concerns detected | 3.95s | 123 tok/s | 5.9 | 597 | 15 (14 from hints) | duplicate keywords |
| mlx-community/GLM-4.6V-Flash-4bit | concerns detected | 8.63s | 78.6 tok/s | 8.7 | 6,393 | 9 (7 from hints) | prompt hint repeated |
| mlx-community/GLM-4.6V-nvfp4 | concerns detected | 20.39s | 43.2 tok/s | 78 | 6,393 | 20 (20 from hints) | prompt hint repeated |
| mlx-community/Idefics3-8B-Llama3-bf16 | concerns detected | 7.25s | 35.3 tok/s | 18 | 2,620 | 10 (1 from hints) | prompt hint repeated |
| mlx-community/Phi-3.5-vision-instruct-bf16 | concerns detected | 4.14s | 55.6 tok/s | 9.3 | 1,144 | 10 (8 from hints) | prompt hint repeated |
| mlx-community/pixtral-12b-8bit | concerns detected | 7.63s | 40.3 tok/s | 16 | 3,125 | 20 (20 from hints) | prompt hint repeated |
| mlx-community/Qwen2-VL-7B-Instruct-4bit | concerns detected | 44.45s | 93.0 tok/s | 9.3 | 16,565 | 20 (20 from hints) | prompt hint repeated |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | concerns detected | 3.53s | 128 tok/s | 5.6 | 1,438 | 19 (19 from hints) | prompt hint repeated |
| mlx-community/Step-3.7-Flash-oQ3e | concerns detected | 34.30s | 50.1 tok/s | 92 | 3,498 | 20 (20 from hints) | prompt hint repeated |
| mlx-community/X-Reasoner-7B-8bit | concerns detected | 18.52s | 59.7 tok/s | 14 | 16,565 | 33 (18 from hints) | duplicate keywords |
| nativ-community/Mage-VL-OptiQ-4bit | concerns detected | 4.64s | 128 tok/s | 5.4 | 4,217 | 20 (20 from hints) | prompt hint repeated |
| mlx-community/FastVLM-0.5B-bf16 | major concerns | 3.04s | 310 tok/s | 2.2 | 341 | - | labelled fields not detected |
| mlx-community/gemma-3n-E4B-it-4bit | major concerns | 5.12s | 71.8 tok/s | 7.2 | 595 | - | labelled fields not detected |
| mlx-community/granite-vision-3.2-2b-nvfp4 | major concerns | 5.07s | 144 tok/s | 4.4 | 5,666 | 20 (20 from hints) | labelled fields not detected |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit | major concerns | 20.41s | 62.2 tok/s | 20 | 1,331 | 126 (19 from hints) | repeated text; labelled fields not detected; cut off at token limit; incomplete thinking block; duplicate keywords |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit | major concerns | 3.74s | 208 tok/s | 4.0 | 2,115 | - | labelled fields not detected |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns | 3.56s | 108 tok/s | 6.7 | 2,201 | - | control tokens visible; labelled fields not detected |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | major concerns | 53.32s | 24.3 tok/s | 25 | 4,410 | - | labelled fields not detected; cut off at token limit |
| mlx-community/nanoLLaVA-1.5-4bit | major concerns | 1.99s | 198 tok/s | 1.8 | 337 | - | labelled fields not detected |
| mlx-community/SmolVLM-256M-Instruct-4bit | major concerns | 2.33s | 317 tok/s | 1.1 | 1,217 | - | labelled fields not detected |

## Completed attempts requiring review

| Model | Mechanical checks | Since baseline | Observed result | Evidence |
| --- | --- | --- | --- | --- |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit | major concerns | unchanged | Response repeats the same text; Required labelled fields not detected: title, description; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Duplicate keywords: boat, boat canopy, boat fender, boating, cabin cruiser, calm water, dock, harbor, marina, mast, nautical, reflection, rope, sailboat, sailing, water reflection, watercraft, mooring, motorboat, yacht wait | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-8bit) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns | unchanged | Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description, keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit) |

## Completions without detected concerns

29 completions without detected concerns; 21 more completed with prompt-compliance observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 9,805 x 6,538 pixels, 54,173,041 bytes
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: seed:* 0
- *Trust remote code:* true
- *check_models version:* 0.17.39
- *check_models revision:* 894cc3a84fba334445b8ac780a139ee51496d08c
- *check_models source dirty:* false
- *mlx-vlm:* 0.7.4
- *mlx-vlm source revision:* 6ecadd767ca1c7c54763289c750dd180d7945037
- *mlx:* 0.32.4.dev20261003+0e3ff3643
- *mlx source revision:* 0e3ff3643
- *transformers:* 5.18.0
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
