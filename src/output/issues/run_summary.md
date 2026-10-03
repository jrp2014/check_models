# mlx-vlm compatibility findings across 51 cached vision-language models

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

- *Run started:* 2026-10-02 23:56:43 BST
- *Run finished:* 2026-10-03 00:10:17 BST
- *Run duration:* 13m 33s
- *Time by phase:* generation 661s, model load 97s, prompt prep 38s, cleanup
  6s, outside the model loop 16s
- *Evaluation lane:* assisted
- *Prompt hints:* the image's description and keyword hints were included in
  the prompt, so field content may be copied from them rather than seen
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Input image:* JPEG, 9,805 x 6,538 pixels (64.1 MP), 54.2 MB
- *Models attempted:* 51
- *Sampling settings:* checkpoint generation_config.json values for 25 of 51
  completed models where the command line left them unset; harness defaults
  elsewhere
- *Completed:* 51
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

- *Baseline:* 058f75b1:src/output/results.jsonl
- *Baseline run timestamp:* 2026-10-02 14:15:17 BST
- *Baseline check_models:* 0.17.38 @ 2a6bcd0b2
- *Baseline mlx:* 0.32.4.dev20261002+255328713 @ 255328713
- *Baseline mlx-vlm:* 0.7.4 @ 66e68ce36
- *Baseline transformers:* 5.18.0
- *Baseline python:* 3.14.7
- *Baseline hardware:* Apple M5 Max, 40 GPU cores, 128.0 GB RAM
- *Models compared:* 51
- *Identical generated text:* 51 of 51 completed in both
- *Text changed, by decoding:* greedy 0 of 28; sampled, same settings and seed
  0 of 16; decoding mode or settings changed 0 of 7
- *Generation tok/s ratio (now/baseline):* 1.024 (range 0.95-1.81, 51 models)
- *Prefill tok/s ratio (now/baseline):* 1.024 (range 0.84-1.65, 51 models)
- *Throughput noise band:* history (last 6 same-prompt runs, Tukey fence, at
  least ±10% of the median)

Run environment: current run on battery power for 51 of 51 models.

No execution, usability, or observation-set changes against the baseline.

Generation tok/s outside the expected band for 12 models:

| Model | Baseline tok/s | Now tok/s | Ratio | Expected band |
| --- | --- | --- | --- | --- |
| mlx-community/nanoLLaVA-1.5-4bit | 207.0 | 375.5 | 1.81 | 183.4-249.0 (history, n=6) |
| mlx-community/SmolVLM-256M-Instruct-4bit | 325.0 | 505.4 | 1.56 | 290.4-356.5 (history, n=6) |
| mlx-community/InternVL3_5-1B-4bit | 335.3 | 422.5 | 1.26 | 285.0-385.6 (fallback) |
| mlx-community/FastVLM-0.5B-bf16 | 335.0 | 370.3 | 1.11 | 301.1-368.0 (history, n=6) |
| mlx-community/gemma-4-26b-a4b-it-4bit | 106.3 | 125.1 | 1.18 | 98.4-120.3 (history, n=6) |
| mlx-community/gemma-3n-E4B-it-4bit | 70.8 | 93.0 | 1.31 | 65.0-79.4 (history, n=6) |
| mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit | 74.4 | 105.5 | 1.42 | 67.8-85.1 (history, n=6) |
| mlx-community/MiniCPM-V-4.6-4bit | 238.9 | 301.1 | 1.26 | 203.1-274.7 (fallback) |
| mlx-community/North-Micro-Vision-Instruct-4bit | 155.4 | 201.2 | 1.29 | 138.6-179.7 (history, n=6) |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | 107.8 | 124.1 | 1.15 | 91.6-123.9 (fallback) |
| nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | 126.8 | 156.6 | 1.23 | 107.8-145.9 (fallback) |
| mlx-community/Qwen3.5-35B-A3B-4bit | 75.2 | 105.8 | 1.41 | 66.8-81.6 (history, n=6) |

<details>
<summary>mlx-vlm: 3 upstream commit(s) since the baseline (66e68ce36..6ecadd767)</summary>

- 6ecadd76 Add conversation compaction to the Responses API (#2408)
- f409f7fe Feat/decision cli (#2402)
- 31215370 Deduplicate rotate_half and check_array_shape into shared modules
  (#2414)

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
| mlx-community/aya-vision-8b-4bit | no concerns detected | 5.69s | 104 tok/s | 6.5 | 2,098 | 17 (10 from hints) | none |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | no concerns detected | 6.31s | 62.7 tok/s | 28 | 597 | 14 (12 from hints) | none |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | no concerns detected | 8.40s | 124 tok/s | 19 | 1,646 | 15 (12 from hints) | none |
| mlx-community/gemma-4-12B-it-4bit | no concerns detected | 5.26s | 61.0 tok/s | 7.6 | 601 | 20 (20 from hints) | none |
| mlx-community/gemma-4-26b-a4b-it-4bit | no concerns detected | 5.25s | 125 tok/s | 16 | 601 | 17 (15 from hints) | none |
| mlx-community/gemma-4-31b-it-4bit | no concerns detected | 9.18s | 26.3 tok/s | 20 | 601 | 17 (17 from hints) | none |
| mlx-community/granite-4.0-3b-vision-4bit | no concerns detected | 3.27s | 175 tok/s | 4.6 | 1,384 | 14 (8 from hints) | none |
| mlx-community/InternVL3-14B-4bit | no concerns detected | 5.69s | 56.9 tok/s | 10 | 2,120 | 20 (20 from hints) | none |
| mlx-community/InternVL3-8B-bf16 | no concerns detected | 5.90s | 37.4 tok/s | 17 | 2,120 | 13 (13 from hints) | none |
| mlx-community/InternVL3_5-1B-4bit | no concerns detected | 2.75s | 423 tok/s | 2.1 | 2,123 | 23 (19 from hints) | none |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | no concerns detected | 9.64s | 21.8 tok/s | 15 | 309 | 10 (2 from hints) | none |
| mlx-community/MiniCPM-o-4_5-4bit | no concerns detected | 3.32s | 107 tok/s | 7.0 | 398 | 16 (11 from hints) | none |
| mlx-community/MiniCPM-V-4.6-4bit | no concerns detected | 4.47s | 301 tok/s | 3.2 | 938 | 17 (12 from hints) | none |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | no concerns detected | 7.71s | 67.1 tok/s | 13 | 2,935 | 20 (7 from hints) | none |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | no concerns detected | 4.34s | 190 tok/s | 7.8 | 2,934 | 13 (2 from hints) | none |
| mlx-community/Molmo2-8B-4bit | no concerns detected | 5.82s | 73.4 tok/s | 8.5 | 1,531 | 36 (19 from hints) | none |
| mlx-community/North-Micro-Vision-Instruct-4bit | no concerns detected | 5.08s | 201 tok/s | 3.9 | 4,091 | 19 (19 from hints) | none |
| mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit | no concerns detected | 5.71s | 105 tok/s | 24 | 1,295 | 18 (18 from hints) | none |
| mlx-community/Qwen3-VL-2B-Thinking-bf16 | no concerns detected | 31.13s | 86.1 tok/s | 8.4 | 16,556 | 16 (14 from hints) | none |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | no concerns detected | 49.88s | 84.2 tok/s | 23 | 16,554 | 20 (20 from hints) | none |
| mlx-community/Qwen3-VL-32B-Instruct-4bit | no concerns detected | 79.47s | 18.4 tok/s | 26 | 16,554 | 18 (17 from hints) | none |
| mlx-community/Qwen3-VL-8B-Instruct-4bit | no concerns detected | 42.11s | 68.1 tok/s | 11 | 16,554 | 18 (15 from hints) | none |
| mlx-community/Qwen3.5-35B-A3B-4bit | no concerns detected | 38.48s | 106 tok/s | 25 | 16,569 | 19 (19 from hints) | none |
| mlx-community/Qwen3.5-9B-MLX-4bit | no concerns detected | 39.36s | 91.2 tok/s | 11 | 16,569 | 15 (7 from hints) | none |
| mlx-community/Qwen3.8-27B-nvfp4 | no concerns detected | 62.43s | 27.6 tok/s | 21 | 16,569 | 16 (5 from hints) | none |
| nativ-community/MiMo-V2.6-Distill-Qwen-9B-MLX-4bit | no concerns detected | 31.77s | 87.8 tok/s | 11 | 16,566 | 15 (1 from hints) | none |
| nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit | no concerns detected | 9.02s | 36.6 tok/s | 18 | 1,281 | 18 (11 from hints) | none |
| nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | no concerns detected | 9.19s | 157 tok/s | 23 | 3,636 | 17 (16 from hints) | none |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | concerns detected | 2.18s | 483 tok/s | 1.9 | 2,123 | 14 (8 from hints) | duplicate keywords; prompt hint repeated |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | concerns detected | 9.96s | 30.6 tok/s | 23 | 2,402 | 20 (20 from hints) | prompt hint repeated |
| mlx-community/gemma-3-27b-it-qat-4bit | concerns detected | 8.44s | 31.5 tok/s | 17 | 596 | 18 (14 from hints) | unsupplied place name |
| mlx-community/gemma-4-e4b-it-4bit | concerns detected | 3.84s | 122 tok/s | 5.9 | 597 | 15 (14 from hints) | duplicate keywords |
| mlx-community/GLM-4.6V-Flash-4bit | concerns detected | 8.70s | 76.3 tok/s | 8.7 | 6,393 | 9 (7 from hints) | prompt hint repeated |
| mlx-community/GLM-4.6V-nvfp4 | concerns detected | 20.65s | 43.0 tok/s | 78 | 6,393 | 20 (20 from hints) | prompt hint repeated |
| mlx-community/Idefics3-8B-Llama3-bf16 | concerns detected | 7.67s | 35.5 tok/s | 18 | 2,620 | 10 (1 from hints) | prompt hint repeated |
| mlx-community/Phi-3.5-vision-instruct-bf16 | concerns detected | 3.92s | 59.6 tok/s | 9.3 | 1,144 | 10 (8 from hints) | prompt hint repeated |
| mlx-community/pixtral-12b-8bit | concerns detected | 7.60s | 40.0 tok/s | 16 | 3,125 | 20 (20 from hints) | prompt hint repeated |
| mlx-community/Qwen2-VL-7B-Instruct-4bit | concerns detected | 56.64s | 87.8 tok/s | 9.3 | 16,565 | 20 (20 from hints) | prompt hint repeated |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | concerns detected | 3.56s | 122 tok/s | 5.6 | 1,438 | 19 (19 from hints) | prompt hint repeated |
| mlx-community/Step-3.7-Flash-oQ3e | concerns detected | 36.90s | 51.6 tok/s | 92 | 3,498 | 20 (20 from hints) | prompt hint repeated |
| mlx-community/X-Reasoner-7B-8bit | concerns detected | 20.96s | 58.5 tok/s | 14 | 16,565 | 33 (18 from hints) | duplicate keywords |
| nativ-community/Mage-VL-OptiQ-4bit | concerns detected | 4.92s | 129 tok/s | 5.4 | 4,217 | 20 (20 from hints) | prompt hint repeated |
| mlx-community/FastVLM-0.5B-bf16 | major concerns | 3.49s | 370 tok/s | 2.2 | 341 | - | labelled fields not detected |
| mlx-community/gemma-3n-E4B-it-4bit | major concerns | 4.82s | 93.0 tok/s | 7.2 | 595 | - | labelled fields not detected |
| mlx-community/granite-vision-3.2-2b-nvfp4 | major concerns | 5.43s | 131 tok/s | 4.4 | 5,666 | 20 (20 from hints) | labelled fields not detected |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit | major concerns | 19.35s | 66.0 tok/s | 20 | 1,331 | 126 (19 from hints) | repeated text; labelled fields not detected; cut off at token limit; incomplete thinking block; duplicate keywords |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit | major concerns | 4.04s | 212 tok/s | 4.0 | 2,115 | - | labelled fields not detected |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns | 3.97s | 107 tok/s | 6.7 | 2,201 | - | control tokens visible; labelled fields not detected |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | major concerns | 59.45s | 21.6 tok/s | 25 | 4,410 | 13 (0 from hints) | control tokens visible; labelled fields not detected; cut off at token limit; role tokens visible; duplicate keywords |
| mlx-community/nanoLLaVA-1.5-4bit | major concerns | 2.01s | 376 tok/s | 1.8 | 337 | - | labelled fields not detected |
| mlx-community/SmolVLM-256M-Instruct-4bit | major concerns | 2.37s | 505 tok/s | 1.1 | 1,217 | - | labelled fields not detected |

## Completed attempts requiring review

| Model | Mechanical checks | Since baseline | Observed result | Evidence |
| --- | --- | --- | --- | --- |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit | major concerns | unchanged | Response repeats the same text; Required labelled fields not detected: title, description; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Duplicate keywords: boat, boat canopy, boat fender, boating, cabin cruiser, calm water, dock, harbor, marina, mast, nautical, reflection, rope, sailboat, sailing, water reflection, watercraft, mooring, motorboat, yacht wait | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-8bit) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns | unchanged | Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description, keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit) |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | major concerns | unchanged | Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description; Response appears cut off at the token limit; Conversation-role control tokens remain visible; Duplicate keywords: setting, action, lighting | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-muse-glimmer-30b-optiq-4bit) |

## Completions without detected concerns

28 completions without detected concerns; 20 more completed with prompt-compliance observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 9,805 x 6,538 pixels, 54,173,041 bytes
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: seed:* 0
- *Trust remote code:* true
- *check_models version:* 0.17.39
- *check_models revision:* 058f75b1504337b679c12f063cdec00e7e6cfc57
- *check_models source dirty:* false
- *mlx-vlm:* 0.7.4
- *mlx-vlm source revision:* 6ecadd767ca1c7c54763289c750dd180d7945037
- *mlx:* 0.32.4.dev20261002+255328713
- *mlx source revision:* 255328713
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
