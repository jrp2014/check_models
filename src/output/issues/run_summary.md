# mlx-vlm compatibility findings across 42 cached vision-language models

## Run summary

- *Run timestamp:* 2026-08-22 22:51:17 BST
- *Evaluation mode:* assisted
- *Models attempted:* 42
- *Completed:* 42
- *Crashed:* 0
- *Indeterminate:* 0
- *Crashes requiring action:* 0
- *Other results requiring review:* 10

Observations are mechanical facts from one image, not general model-quality
judgements.

## Model quality at a glance

Every attempted model ranked by current-run usability, with captured resource
facts. Usability reflects this single image and prompt only; the model gallery
holds full outputs and the diagnostics report holds maintainer evidence.

| Model | Usability | Total | Gen tok/s | Peak GB | Observed |
| --- | --- | --- | --- | --- | --- |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | usable | 2.09s | 483 tok/s | 1.9 | none |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | usable | 11.57s | 30.3 tok/s | 24 | none |
| mlx-community/gemma-4-26b-a4b-it-4bit | usable | 4.28s | 129 tok/s | 16 | none |
| mlx-community/gemma-4-31b-it-4bit | usable | 8.67s | 25.8 tok/s | 20 | none |
| mlx-community/Idefics3-8B-Llama3-bf16 | usable | 10.51s | 32.3 tok/s | 18 | none |
| mlx-community/LFM2.5-VL-1.6B-bf16 | usable | 2.51s | 182 tok/s | 4.0 | none |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | usable | 6.68s | 65.5 tok/s | 14 | none |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | usable | 3.57s | 178 tok/s | 9.0 | none |
| mlx-community/Ornith-1.0-35B-bf16 | usable | 76.09s | 62.0 tok/s | 74 | none |
| mlx-community/pixtral-12b-8bit | usable | 7.05s | 39.5 tok/s | 16 | none |
| mlx-community/Qwen3.5-35B-A3B-4bit | usable | 62.05s | 110 tok/s | 24 | none |
| mlx-community/Qwen3.5-9B-MLX-4bit | usable | 62.75s | 93.1 tok/s | 10.0 | none |
| mlx-community/Qwen3.8-27B-4bit | usable | 82.98s | 30.0 tok/s | 21 | none |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | usable | 2.82s | 125 tok/s | 5.5 | none |
| mlx-community/Step-3.7-Flash-oQ2e | usable | 26.49s | 45.2 tok/s | 70 | none |
| mlx-community/diffusiongemma-26B-A4B-it-8bit | usable with caveats | 6.50s | 63.0 tok/s | 29 | control tokens visible |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | usable with caveats | 5.54s | 69.8 tok/s | 28 | control tokens visible |
| mlx-community/gemma-3-27b-it-qat-4bit | usable with caveats | 8.54s | 31.2 tok/s | 17 | title/keyword constraints failed |
| mlx-community/GLM-4.6V-nvfp4 | usable with caveats | 27.82s | 40.1 tok/s | 78 | control tokens visible; title/keyword constraints failed |
| mlx-community/InternVL3-8B-bf16 | usable with caveats | 6.16s | 33.9 tok/s | 17 | title/keyword constraints failed |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 | usable with caveats | 138.72s | 4.63 tok/s | 40 | role tokens visible |
| mlx-community/MiniCPM-V-4.6-8bit | usable with caveats | 2.91s | 264 tok/s | 3.8 | title/keyword constraints failed |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4 | usable with caveats | 7.16s | 61.8 tok/s | 15 | title/keyword constraints failed |
| mlx-community/Molmo-7B-D-0924-8bit | usable with caveats | 5.39s | 51.3 tok/s | 11 | title/keyword constraints failed |
| mlx-community/Phi-3.5-vision-instruct-bf16 | usable with caveats | 3.26s | 55.3 tok/s | 9.6 | title/keyword constraints failed |
| mlx-community/Qwen3-VL-2B-Thinking-bf16 | usable with caveats | 30.00s | 89.5 tok/s | 8.4 | title/keyword constraints failed |
| mlx-community/Qwen3.6-27B-mxfp8 | usable with caveats | 87.68s | 17.7 tok/s | 33 | title/keyword constraints failed |
| mlx-community/X-Reasoner-7B-8bit | usable with caveats | 24.20s | 56.4 tok/s | 13 | title/keyword constraints failed |
| mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX | unusable | 28.52s | 41.8 tok/s | 15 | echoes instructions; extra text before Title; cut off at token limit; title/keyword constraints failed |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 | unusable | 34.46s | 58.7 tok/s | 60 | cut off at token limit; title/keyword constraints failed |
| mlx-community/FastVLM-0.5B-bf16 | unusable | 3.19s | 352 tok/s | 1.9 | missing required fields; echoes instructions; extra text before Title |
| mlx-community/gemma-3n-E4B-it-bf16 | unusable | 6.91s | 48.6 tok/s | 17 | missing required fields |
| mlx-community/GLM-4.1V-9B-Thinking-8bit | unusable | 29.69s | 46.3 tok/s | 13 | missing required fields; extra text before Title |
| mlx-community/GLM-4.6V-Flash-mxfp4 | unusable | 21.06s | 75.6 tok/s | 8.4 | repeated text; cut off at token limit; title/keyword constraints failed |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | unusable | 57.60s | 18.6 tok/s | 15 | repeated text; cut off at token limit; title/keyword constraints failed |
| mlx-community/llava-v1.6-mistral-7b-8bit | unusable | 4.25s | 63.9 tok/s | 9.7 | missing required fields |
| mlx-community/MolmoPoint-8B-fp16 | unusable | 31.60s | 5.96 tok/s | 24 | missing required fields |
| mlx-community/nanoLLaVA-1.5-4bit | unusable | 1.61s | 370 tok/s | 2.4 | missing required fields; echoes instructions |
| mlx-community/paligemma2-3b-pt-896-4bit | unusable | 24.91s | 45.9 tok/s | 4.4 | repeated text; missing required fields; echoes instructions; cut off at token limit |
| mlx-community/Qwen2-VL-2B-Instruct-4bit | unusable | 74.01s | 210 tok/s | 5.1 | repeated text; cut off at token limit; title/keyword constraints failed |
| mlx-community/Qwen3-VL-2B-Instruct-bf16 | unusable | 30.84s | 87.2 tok/s | 8.4 | repeated text; cut off at token limit; title/keyword constraints failed |
| Qwen/Qwen3-VL-2B-Instruct | unusable | 26.07s | 92.4 tok/s | 8.4 | repeated text; cut off at token limit; title/keyword constraints failed |

## Observation clusters

Repeated mechanical observation signatures among results requiring review.

| Observed result | Models |
| --- | --- |
| Response repeats the same text; Response appears cut off at the token limit; Title or keywords do not meet requested constraints | 5 |
| Response repeats the same text; Required fields are missing or empty; Response repeats the task instructions instead of only returning the requested fields; Response appears cut off at the token limit | 1 |
| Unrecognised model control tokens remain visible | 2 |
| Unrecognised model control tokens remain visible; Title or keywords do not meet requested constraints | 1 |
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
| mlx-community/diffusiongemma-26B-A4B-it-8bit | usable with caveats | Unrecognised model control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-8bit) |
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
- *check_models version:* 0.14.0
- *check_models revision:* 5a78e984657f492733f34cbe3d7887058ef60060
- *check_models source dirty:* false
- *mlx-vlm:* 0.6.15
- *mlx-vlm source revision:* 18e9b979f9e6b9edfad20832698db1171320247c
- *mlx:* 0.32.2.dev20260822+846d17622
- *mlx source revision:* 846d176227a0ac13d2667e58d2bb68b322109ab0
- *transformers:* 5.15.1
- *macOS Version:* 26.6.2
- *GPU/Chip:* Apple M5 Max
- *Python Version:* 3.13.14

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
