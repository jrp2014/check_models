# mlx-vlm compatibility findings across 42 cached vision-language models

## Run summary

- *Run timestamp:* 2026-08-23 23:10:25 BST
- *Evaluation mode:* assisted
- *Models attempted:* 42
- *Completed:* 42
- *Crashed:* 0
- *Indeterminate:* 0
- *Crashes requiring action:* 0
- *Other results requiring review:* 10

Observations are mechanical facts from one image, not general model-quality
judgements.

## Since the baseline sweep

- *Baseline:* 6f54a4fe:src/output/results.jsonl
- *Baseline run timestamp:* 2026-08-23 12:44:16 BST
- *Baseline check_models:* 0.14.1 @ d717f91bf
- *Baseline mlx:* 0.32.2.dev20260823+d9077d831 @ d9077d831
- *Baseline mlx-vlm:* 0.6.15 @ 332873ff2
- *Baseline transformers:* 5.15.1
- *Baseline python:* 3.14.7
- *Models compared:* 42
- *Identical generated text:* 40 of 42 completed in both
- *Generation tok/s ratio (now/baseline):* 1.023 (range 0.85-1.71, 42 models)
- *Throughput noise band:* history (last 8 same-prompt runs, Tukey fence, at
  least ±10% of the median)

| Model | Execution | Usability | Observation delta |
| --- | --- | --- | --- |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | completed | usable_with_caveats | -catalog_constraint_violation |

| Model | Baseline tok/s | Now tok/s | Ratio | Expected band |
| --- | --- | --- | --- | --- |
| mlx-community/gemma-3-27b-it-qat-4bit | 30.0 | 26.4 | 0.88 | 27.9-34.1 (history, n=8) |
| mlx-community/GLM-4.6V-Flash-mxfp4 | 72.6 | 65.2 | 0.90 | 67.4-82.4 (history, n=6) |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | 19.4 | 16.6 | 0.85 | 16.8-21.8 (history, n=8) |

Mechanical diff only: one image, temperature as configured; single-observation
flips on one model are usually run-to-run variance, broad shifts are not.

## Model quality at a glance

Every attempted model ranked by current-run usability, with captured resource
facts. Usability reflects this single image and prompt only; the model gallery
holds full outputs and the diagnostics report holds maintainer evidence.

| Model | Usability | Total | Gen tok/s | Peak GB | Observed |
| --- | --- | --- | --- | --- | --- |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | usable | 1.63s | 478 tok/s | 1.9 | none |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | usable | 11.03s | 29.8 tok/s | 24 | none |
| mlx-community/gemma-4-26b-a4b-it-4bit | usable | 5.25s | 120 tok/s | 16 | none |
| mlx-community/gemma-4-31b-it-4bit | usable | 8.51s | 26.6 tok/s | 20 | none |
| mlx-community/Idefics3-8B-Llama3-bf16 | usable | 9.99s | 33.3 tok/s | 18 | none |
| mlx-community/LFM2.5-VL-1.6B-bf16 | usable | 2.29s | 194 tok/s | 4.0 | none |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | usable | 6.90s | 65.6 tok/s | 14 | none |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | usable | 3.64s | 181 tok/s | 9.0 | none |
| mlx-community/Ornith-1.0-35B-bf16 | usable | 81.83s | 63.4 tok/s | 74 | none |
| mlx-community/pixtral-12b-8bit | usable | 7.19s | 39.6 tok/s | 16 | none |
| mlx-community/Qwen3.5-35B-A3B-4bit | usable | 72.84s | 100 tok/s | 24 | none |
| mlx-community/Qwen3.5-9B-MLX-4bit | usable | 74.79s | 90.3 tok/s | 10.0 | none |
| mlx-community/Qwen3.8-27B-4bit | usable | 95.76s | 27.3 tok/s | 21 | none |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | usable | 2.65s | 125 tok/s | 5.5 | none |
| mlx-community/Step-3.7-Flash-oQ2e | usable | 25.91s | 42.5 tok/s | 70 | none |
| mlx-community/diffusiongemma-26B-A4B-it-8bit | usable with caveats | 6.28s | 48.7 tok/s | 29 | control tokens visible; title/keyword constraints failed |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | usable with caveats | 5.51s | 75.4 tok/s | 28 | control tokens visible |
| mlx-community/gemma-3-27b-it-qat-4bit | usable with caveats | 9.32s | 26.4 tok/s | 17 | title/keyword constraints failed |
| mlx-community/GLM-4.6V-nvfp4 | usable with caveats | 30.42s | 38.9 tok/s | 78 | control tokens visible; title/keyword constraints failed |
| mlx-community/InternVL3-8B-bf16 | usable with caveats | 6.18s | 36.1 tok/s | 17 | title/keyword constraints failed |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 | usable with caveats | 130.90s | 4.86 tok/s | 40 | role tokens visible |
| mlx-community/MiniCPM-V-4.6-8bit | usable with caveats | 2.25s | 272 tok/s | 3.8 | title/keyword constraints failed |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4 | usable with caveats | 7.94s | 60.1 tok/s | 15 | title/keyword constraints failed |
| mlx-community/Molmo-7B-D-0924-8bit | usable with caveats | 5.13s | 53.1 tok/s | 11 | title/keyword constraints failed |
| mlx-community/Phi-3.5-vision-instruct-bf16 | usable with caveats | 3.22s | 57.3 tok/s | 9.6 | title/keyword constraints failed |
| mlx-community/Qwen3-VL-2B-Thinking-bf16 | usable with caveats | 37.57s | 84.5 tok/s | 8.4 | title/keyword constraints failed |
| mlx-community/Qwen3.6-27B-mxfp8 | usable with caveats | 100.44s | 15.8 tok/s | 33 | title/keyword constraints failed |
| mlx-community/X-Reasoner-7B-8bit | usable with caveats | 29.07s | 50.4 tok/s | 13 | title/keyword constraints failed |
| mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX | unusable | 29.01s | 40.9 tok/s | 15 | echoes instructions; extra text before Title; cut off at token limit; title/keyword constraints failed |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 | unusable | 28.78s | 55.3 tok/s | 60 | cut off at token limit; title/keyword constraints failed |
| mlx-community/FastVLM-0.5B-bf16 | unusable | 2.46s | 350 tok/s | 2.1 | missing required fields; echoes instructions; extra text before Title |
| mlx-community/gemma-3n-E4B-it-bf16 | unusable | 6.93s | 47.3 tok/s | 17 | missing required fields |
| mlx-community/GLM-4.1V-9B-Thinking-8bit | unusable | 32.82s | 42.0 tok/s | 13 | missing required fields; extra text before Title |
| mlx-community/GLM-4.6V-Flash-mxfp4 | unusable | 25.53s | 65.2 tok/s | 8.4 | repeated text; cut off at token limit; title/keyword constraints failed |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | unusable | 64.41s | 16.6 tok/s | 15 | repeated text; cut off at token limit; title/keyword constraints failed |
| mlx-community/llava-v1.6-mistral-7b-8bit | unusable | 5.08s | 63.6 tok/s | 9.7 | missing required fields |
| mlx-community/MolmoPoint-8B-fp16 | unusable | 27.78s | 6.42 tok/s | 24 | missing required fields |
| mlx-community/nanoLLaVA-1.5-4bit | unusable | 1.69s | 306 tok/s | 2.4 | missing required fields; echoes instructions |
| mlx-community/paligemma2-3b-pt-896-4bit | unusable | 26.11s | 43.7 tok/s | 4.4 | repeated text; missing required fields; echoes instructions; cut off at token limit |
| mlx-community/Qwen2-VL-2B-Instruct-4bit | unusable | 89.27s | 207 tok/s | 5.1 | repeated text; cut off at token limit; title/keyword constraints failed |
| mlx-community/Qwen3-VL-2B-Instruct-bf16 | unusable | 39.08s | 82.8 tok/s | 8.4 | repeated text; cut off at token limit; title/keyword constraints failed |
| Qwen/Qwen3-VL-2B-Instruct | unusable | 27.02s | 92.5 tok/s | 8.4 | repeated text; cut off at token limit; title/keyword constraints failed |

## Observation clusters

Repeated mechanical observation signatures among results requiring review.

| Observed result | Models |
| --- | --- |
| Response repeats the same text; Response appears cut off at the token limit; Title or keywords do not meet requested constraints | 5 |
| Response repeats the same text; Required fields are missing or empty; Response repeats the task instructions instead of only returning the requested fields; Response appears cut off at the token limit | 1 |
| Unrecognised model control tokens remain visible; Title or keywords do not meet requested constraints | 2 |
| Unrecognised model control tokens remain visible | 1 |
| Conversation-role control tokens remain visible | 1 |

## Completed attempts requiring review

| Model | Usability | Observed result | Evidence |
| --- | --- | --- | --- |
| mlx-community/GLM-4.6V-Flash-mxfp4 | unusable | Response repeats the same text; Response appears cut off at the token limit; Title has 3 words (requested 5-10); Keyword list has 382 terms (requested 10-18); Duplicate keywords: seafront, uk, gbr, europe, war memorial, stone column, eagle statue, sea, coastal, memorial, remembrance, war, suffolk county | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-46v-flash-mxfp4) |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | unusable | Response repeats the same text; Response appears cut off at the token limit; Title has 4 words (requested 5-10); Keyword list has 306 terms (requested 10-18); Duplicate keywords: historical landmark, historical significance, cultural icon, historical icon | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llama-32-11b-vision-instruct-8bit) |
| mlx-community/paligemma2-3b-pt-896-4bit | unusable | Response repeats the same text; Missing or empty fields: Title, Description, Keywords; Response repeats the task instructions instead of only returning the requested fields; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-paligemma2-3b-pt-896-4bit) |
| mlx-community/Qwen2-VL-2B-Instruct-4bit | unusable | Response repeats the same text; Response appears cut off at the token limit; Keyword list has 259 terms (requested 10-18); Duplicate keywords: stone column, bird statue, people walking, clear sky, calm sea, stone pathway, landmark, scenic beauty, seaside town | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-qwen2-vl-2b-instruct-4bit) |
| mlx-community/Qwen3-VL-2B-Instruct-bf16 | unusable | Response repeats the same text; Response appears cut off at the token limit; Title has 3 words (requested 5-10); Keyword list has 330 terms (requested 10-18); Duplicate keywords: seafront, memorial, sea, england, uk, europe, 1939 1945, war, commemoration, plaques, lamppost, blue, sky, stone, column, bronze, eagle | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-qwen3-vl-2b-instruct-bf16) |
| Qwen/Qwen3-VL-2B-Instruct | unusable | Response repeats the same text; Response appears cut off at the token limit; Title has 3 words (requested 5-10); Keyword list has 330 terms (requested 10-18); Duplicate keywords: seafront, memorial, sea, england, uk, europe, 1939 1945, war, commemoration, plaques, lamppost, blue, sky, stone, column, bronze, eagle | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-qwen-qwen3-vl-2b-instruct) |
| mlx-community/diffusiongemma-26B-A4B-it-8bit | usable with caveats | Unrecognised model control tokens remain visible; Duplicate keywords: memorial | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-8bit) |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | usable with caveats | Unrecognised model control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-mxfp8) |
| mlx-community/GLM-4.6V-nvfp4 | usable with caveats | Unrecognised model control tokens remain visible; Title has 4 words (requested 5-10) | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-46v-nvfp4) |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 | usable with caveats | Conversation-role control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16) |

## Clean completions

15 clean completions (`LiquidAI/LFM2.5-VL-450M-MLX-bf16`, `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`, `mlx-community/Idefics3-8B-Llama3-bf16`, `mlx-community/LFM2.5-VL-1.6B-bf16`, `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`, `mlx-community/Ministral-3-3B-Instruct-2512-4bit`, `mlx-community/Ornith-1.0-35B-bf16`, `mlx-community/Qwen3.5-35B-A3B-4bit`, `mlx-community/Qwen3.5-9B-MLX-4bit`, `mlx-community/Qwen3.8-27B-4bit`, `mlx-community/SmolVLM2-2.2B-Instruct-mlx`, `mlx-community/Step-3.7-Flash-oQ2e`, `mlx-community/gemma-4-26b-a4b-it-4bit`, `mlx-community/gemma-4-31b-it-4bit`, `mlx-community/pixtral-12b-8bit`); 17 more completed with prompt-compliance observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 6,656 x 8,880 pixels, 31,372,387 bytes
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: temperature:* 0.0
- *Generation: top_p:* 1.0
- *Trust remote code:* true
- *check_models version:* 0.14.1
- *check_models revision:* 6f54a4fe2c64bf189ce319a8f8aa8ade9e97dafe
- *check_models source dirty:* false
- *mlx-vlm:* 0.6.16
- *mlx-vlm source revision:* 5fa03cfcc670163754afc2c419d15f2ebf6a5abc
- *mlx:* 0.32.2.dev20260823+451dc8759
- *mlx source revision:* 451dc8759703b8e3f3cde34251292edaff63a50f
- *transformers:* 5.15.1
- *macOS Version:* 26.6.2
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
| Run JSON | [run.json](https://github.com/jrp2014/check_models/blob/main/src/output/run.json) |
| Environment | [environment.log](https://github.com/jrp2014/check_models/blob/main/src/output/environment.log) |
| Log | [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log) |
