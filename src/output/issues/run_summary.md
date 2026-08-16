# mlx-vlm compatibility findings across 42 cached vision-language models

## Run summary

- *Run timestamp:* 2026-08-16 22:30:27 BST
- *Evaluation mode:* assisted
- *Models attempted:* 42
- *Completed:* 42
- *Crashed:* 0
- *Indeterminate:* 0
- *Crashes requiring action:* 0
- *Other results requiring review:* 11

Observations are mechanical facts from one image, not general model-quality
judgements.

## Model quality at a glance

Every attempted model ranked by current-run usability, with captured resource
facts. Usability reflects this single image and prompt only; the model gallery
holds full outputs and the diagnostics report holds maintainer evidence.

| Model | Usability | Total | Gen tok/s | Peak GB | Observed |
| --- | --- | --- | --- | --- | --- |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | usable | 1.51s | 479 tok/s | 1.9 | none |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | usable | 10.88s | 29.9 tok/s | 24 | none |
| mlx-community/gemma-4-26b-a4b-it-4bit | usable | 4.11s | 130 tok/s | 16 | none |
| mlx-community/gemma-4-31b-it-4bit | usable | 8.21s | 27.7 tok/s | 20 | none |
| mlx-community/LFM2.5-VL-1.6B-bf16 | usable | 2.27s | 186 tok/s | 4.2 | none |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | usable | 6.74s | 66.3 tok/s | 14 | none |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | usable | 3.24s | 188 tok/s | 9.0 | none |
| mlx-community/pixtral-12b-8bit | usable | 6.91s | 39.7 tok/s | 16 | none |
| mlx-community/Qwen3.5-35B-A3B-4bit | usable | 68.72s | 109 tok/s | 24 | none |
| mlx-community/Qwen3.5-9B-MLX-4bit | usable | 76.50s | 91.2 tok/s | 10.0 | none |
| mlx-community/Qwen3.8-27B-4bit | usable | 101.65s | 25.9 tok/s | 22 | none |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | usable | 2.42s | 126 tok/s | 5.5 | none |
| mlx-community/Step-3.7-Flash-oQ2e | usable | 24.87s | 44.0 tok/s | 70 | none |
| mlx-community/diffusiongemma-26B-A4B-it-8bit | usable with caveats | 5.74s | 55.5 tok/s | 29 | control tokens visible; title/keyword constraints failed |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | usable with caveats | 5.56s | 58.1 tok/s | 28 | control tokens visible |
| mlx-community/gemma-3-27b-it-qat-4bit | usable with caveats | 8.04s | 31.0 tok/s | 18 | title/keyword constraints failed |
| mlx-community/GLM-4.6V-nvfp4 | usable with caveats | 23.32s | 40.9 tok/s | 78 | control tokens visible; title/keyword constraints failed |
| mlx-community/Idefics3-8B-Llama3-bf16 | usable with caveats | 8.49s | 32.6 tok/s | 18 | role tokens visible |
| mlx-community/InternVL3-8B-bf16 | usable with caveats | 5.94s | 34.1 tok/s | 17 | title/keyword constraints failed |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 | usable with caveats | 180.49s | 4.66 tok/s | 40 | role tokens visible; title/keyword constraints failed |
| mlx-community/MiniCPM-V-4.6-8bit | usable with caveats | 2.01s | 271 tok/s | 3.8 | title/keyword constraints failed |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4 | usable with caveats | 6.85s | 64.0 tok/s | 15 | title/keyword constraints failed |
| mlx-community/Molmo-7B-D-0924-8bit | usable with caveats | 5.06s | 52.7 tok/s | 11 | title/keyword constraints failed |
| mlx-community/Ornith-1.0-35B-bf16 | usable with caveats | 64.94s | 63.9 tok/s | 74 | title/keyword constraints failed |
| mlx-community/Phi-3.5-vision-instruct-bf16 | usable with caveats | 3.12s | 56.8 tok/s | 9.6 | title/keyword constraints failed |
| mlx-community/Qwen3-VL-2B-Thinking-bf16 | usable with caveats | 29.54s | 90.6 tok/s | 8.4 | title/keyword constraints failed |
| mlx-community/Qwen3.6-27B-mxfp8 | usable with caveats | 103.41s | 15.4 tok/s | 35 | title/keyword constraints failed |
| mlx-community/X-Reasoner-7B-8bit | usable with caveats | 26.34s | 51.8 tok/s | 13 | title/keyword constraints failed |
| mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX | unusable | 28.36s | 41.4 tok/s | 15 | echoes instructions; extra text before Title; cut off at token limit; title/keyword constraints failed |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 | unusable | 24.49s | 59.2 tok/s | 60 | cut off at token limit; title/keyword constraints failed |
| mlx-community/FastVLM-0.5B-bf16 | unusable | 2.03s | 353 tok/s | 2.2 | missing required fields; echoes instructions; extra text before Title |
| mlx-community/gemma-3n-E4B-it-bf16 | unusable | 6.45s | 48.8 tok/s | 17 | missing required fields |
| mlx-community/GLM-4.1V-9B-Thinking-8bit | unusable | 31.53s | 44.6 tok/s | 13 | missing required fields; extra text before Title |
| mlx-community/GLM-4.6V-Flash-mxfp4 | unusable | 22.35s | 74.2 tok/s | 8.4 | repeated text; cut off at token limit; title/keyword constraints failed |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | unusable | 51.57s | 20.8 tok/s | 15 | repeated text; cut off at token limit; title/keyword constraints failed |
| mlx-community/llava-v1.6-mistral-7b-8bit | unusable | 4.54s | 64.3 tok/s | 9.7 | missing required fields |
| mlx-community/MolmoPoint-8B-fp16 | unusable | 29.22s | 6.04 tok/s | 24 | missing required fields |
| mlx-community/nanoLLaVA-1.5-4bit | unusable | 1.46s | 374 tok/s | 2.4 | missing required fields; echoes instructions |
| mlx-community/paligemma2-3b-pt-896-4bit | unusable | 23.81s | 48.0 tok/s | 4.4 | repeated text; missing required fields; echoes instructions; cut off at token limit |
| mlx-community/Qwen2-VL-2B-Instruct-4bit | unusable | 70.47s | 223 tok/s | 5.1 | repeated text; cut off at token limit; title/keyword constraints failed |
| mlx-community/Qwen3-VL-2B-Instruct-bf16 | unusable | 28.98s | 91.2 tok/s | 8.4 | repeated text; cut off at token limit; title/keyword constraints failed |
| Qwen/Qwen3-VL-2B-Instruct | unusable | 25.95s | 93.4 tok/s | 8.4 | repeated text; cut off at token limit; title/keyword constraints failed |

## Observation clusters

Repeated mechanical observation signatures among results requiring review.

| Observed result | Models |
| --- | --- |
| Response repeats the same text; Response appears cut off at the token limit; Title or keywords do not meet requested constraints | 5 |
| Response repeats the same text; Required fields are missing or empty; Response repeats the task instructions instead of only returning the requested fields; Response appears cut off at the token limit | 1 |
| Unrecognised model control tokens remain visible; Title or keywords do not meet requested constraints | 2 |
| Unrecognised model control tokens remain visible | 1 |
| Conversation-role control tokens remain visible | 1 |
| Conversation-role control tokens remain visible; Title or keywords do not meet requested constraints | 1 |

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
| mlx-community/Idefics3-8B-Llama3-bf16 | usable with caveats | Conversation-role control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-idefics3-8b-llama3-bf16) |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 | usable with caveats | Conversation-role control tokens remain visible; Keyword list has 21 terms (requested 10-18) | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16) |

## Clean completions

13 clean completions (`LiquidAI/LFM2.5-VL-450M-MLX-bf16`, `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`, `mlx-community/LFM2.5-VL-1.6B-bf16`, `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`, `mlx-community/Ministral-3-3B-Instruct-2512-4bit`, `mlx-community/Qwen3.5-35B-A3B-4bit`, `mlx-community/Qwen3.5-9B-MLX-4bit`, `mlx-community/Qwen3.8-27B-4bit`, `mlx-community/SmolVLM2-2.2B-Instruct-mlx`, `mlx-community/Step-3.7-Flash-oQ2e`, `mlx-community/gemma-4-26b-a4b-it-4bit`, `mlx-community/gemma-4-31b-it-4bit`, `mlx-community/pixtral-12b-8bit`); 18 more completed with prompt-compliance observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 6,656 x 8,880 pixels, 31,372,387 bytes
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: temperature:* 0.0
- *Generation: top_p:* 1.0
- *Trust remote code:* true
- *check_models version:* 0.11.0
- *check_models revision:* e4e0cef29909bb7dab471999fc7238567cfada10
- *check_models source dirty:* false
- *mlx-vlm:* 0.6.14
- *mlx:* 0.32.1.dev20260816+c2bcf47ee
- *transformers:* 5.15.0
- *macOS Version:* 26.6.1
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
