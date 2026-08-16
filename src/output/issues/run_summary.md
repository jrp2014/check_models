# mlx-vlm compatibility findings across 41 cached vision-language models

## Run summary

- *Run timestamp:* 2026-08-16 18:47:12 BST
- *Evaluation mode:* blind
- *Models attempted:* 41
- *Completed:* 41
- *Crashed:* 0
- *Indeterminate:* 0
- *Crashes requiring action:* 0
- *Other results requiring review:* 9

Observations are mechanical facts from one image, not general model-quality
judgements.

## Model quality at a glance

Every attempted model ranked by current-run usability, with captured resource
facts. Usability reflects this single image and prompt only; the model gallery
holds full outputs and the diagnostics report holds maintainer evidence.

| Model | Usability | Total | Gen tok/s | Peak GB | Observed |
| --- | --- | --- | --- | --- | --- |
| mlx-community/gemma-3-27b-it-qat-4bit | usable | 6.04s | 30.3 tok/s | 18 | none |
| mlx-community/gemma-4-26b-a4b-it-4bit | usable | 3.41s | 125 tok/s | 16 | none |
| mlx-community/gemma-4-31b-it-4bit | usable | 6.49s | 27.0 tok/s | 19 | none |
| mlx-community/InternVL3-8B-bf16 | usable | 5.08s | 35.8 tok/s | 17 | none |
| mlx-community/LFM2.5-VL-1.6B-bf16 | usable | 1.25s | 184 tok/s | 4.1 | none |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | usable | 3.20s | 69.7 tok/s | 9.8 | none |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4 | usable | 3.75s | 64.9 tok/s | 10 | none |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | usable | 1.73s | 194 tok/s | 4.5 | none |
| mlx-community/Molmo-7B-D-0924-8bit | usable | 3.29s | 53.6 tok/s | 11 | none |
| mlx-community/MolmoPoint-8B-fp16 | usable | 16.24s | 6.04 tok/s | 23 | none |
| mlx-community/Phi-3.5-vision-instruct-bf16 | usable | 2.39s | 57.7 tok/s | 9.3 | none |
| mlx-community/pixtral-12b-8bit | usable | 4.90s | 40.4 tok/s | 15 | none |
| mlx-community/Qwen3-VL-2B-Instruct-bf16 | usable | 1.62s | 135 tok/s | 5.3 | none |
| mlx-community/Qwen3-VL-2B-Thinking-bf16 | usable | 7.54s | 131 tok/s | 5.3 | none |
| mlx-community/Qwen3.5-35B-A3B-4bit | usable | 3.55s | 124 tok/s | 21 | none |
| mlx-community/Qwen3.5-9B-MLX-4bit | usable | 2.63s | 101 tok/s | 7.1 | none |
| mlx-community/Qwen3.6-27B-mxfp8 | usable | 9.90s | 18.4 tok/s | 30 | none |
| mlx-community/Step-3.7-Flash-oQ2e | usable | 15.65s | 43.4 tok/s | 65 | none |
| mlx-community/X-Reasoner-7B-8bit | usable | 3.09s | 65.9 tok/s | 10 | none |
| Qwen/Qwen3-VL-2B-Instruct | usable | 1.36s | 138 tok/s | 5.2 | none |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | usable with caveats | 0.62s | 516 tok/s | 1.4 | title/keyword constraints failed |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | usable with caveats | 6.12s | 30.9 tok/s | 20 | title/keyword constraints failed |
| mlx-community/diffusiongemma-26B-A4B-it-8bit | usable with caveats | 5.25s | 49.0 tok/s | 29 | control tokens visible |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | usable with caveats | 5.47s | 41.7 tok/s | 28 | control tokens visible |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 | usable with caveats | 23.49s | 55.6 tok/s | 60 | title/keyword constraints failed |
| mlx-community/gemma-3n-E4B-it-bf16 | usable with caveats | 4.11s | 45.8 tok/s | 17 | title/keyword constraints failed |
| mlx-community/GLM-4.6V-Flash-mxfp4 | usable with caveats | 2.44s | 89.8 tok/s | 7.8 | title/keyword constraints failed |
| mlx-community/GLM-4.6V-nvfp4 | usable with caveats | 14.17s | 49.9 tok/s | 63 | control tokens visible |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 | usable with caveats | 121.53s | 4.8 tok/s | 40 | role tokens visible |
| mlx-community/Ornith-1.0-35B-bf16 | usable with caveats | 18.87s | 64.8 tok/s | 71 | title/keyword constraints failed |
| mlx-community/Qwen2-VL-2B-Instruct-4bit | usable with caveats | 0.90s | 311 tok/s | 2.5 | title/keyword constraints failed |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | usable with caveats | 1.36s | 124 tok/s | 5.5 | title/keyword constraints failed |
| mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX | unusable | 25.39s | 43.7 tok/s | 14 | missing required fields; echoes instructions; extra text before Title; cut off at token limit |
| mlx-community/FastVLM-0.5B-bf16 | unusable | 0.81s | 341 tok/s | 2.1 | missing required fields |
| mlx-community/GLM-4.1V-9B-Thinking-8bit | unusable | 23.02s | 47.7 tok/s | 13 | extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed |
| mlx-community/Idefics3-8B-Llama3-bf16 | unusable | 3.37s | insufficient sample | 18 | missing required fields; role tokens visible |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | unusable | 66.18s | 15.9 tok/s | 15 | repeated text; extra text before Title; cut off at token limit; title/keyword constraints failed |
| mlx-community/llava-v1.6-mistral-7b-8bit | unusable | 23.37s | 51.0 tok/s | 9.7 | repeated text; missing required fields; cut off at token limit |
| mlx-community/MiniCPM-V-4.6-8bit | unusable | 1.23s | 278 tok/s | 3.0 | missing required fields; extra text before Title |
| mlx-community/nanoLLaVA-1.5-4bit | unusable | 0.88s | 349 tok/s | 2.0 | missing required fields |
| mlx-community/paligemma2-3b-pt-896-4bit | unusable | 24.91s | 45.1 tok/s | 4.2 | repeated text; missing required fields; echoes instructions; extra text before Title; cut off at token limit |

## Observation clusters

Repeated mechanical observation signatures among results requiring review.

| Observed result | Models |
| --- | --- |
| Response repeats the same text; Required fields are missing or empty; Response appears cut off at the token limit | 1 |
| Response repeats the same text; Required fields are missing or empty; Response repeats the task instructions instead of only returning the requested fields; Extra text appears before the Title field; Response appears cut off at the token limit | 1 |
| Response repeats the same text; Extra text appears before the Title field; Response appears cut off at the token limit; Title or keywords do not meet requested constraints | 1 |
| Unrecognised model control tokens remain visible | 3 |
| Required fields are missing or empty; Conversation-role control tokens remain visible | 1 |
| Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Title or keywords do not meet requested constraints | 1 |
| Conversation-role control tokens remain visible | 1 |

## Completed attempts requiring review

| Model | Usability | Observed result | Evidence |
| --- | --- | --- | --- |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | unusable | Response repeats the same text; Extra text appears before the Title field; Response appears cut off at the token limit; Keyword list has 313 terms (requested 10-18); Duplicate keywords: household frames | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llama-32-11b-vision-instruct-8bit) |
| mlx-community/llava-v1.6-mistral-7b-8bit | unusable | Response repeats the same text; Missing or empty fields: Title, Description, Keywords; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llava-v16-mistral-7b-8bit) |
| mlx-community/paligemma2-3b-pt-896-4bit | unusable | Response repeats the same text; Missing or empty fields: Title; Response repeats the task instructions instead of only returning the requested fields; Extra text appears before the Title field; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-paligemma2-3b-pt-896-4bit) |
| mlx-community/diffusiongemma-26B-A4B-it-8bit | usable with caveats | Unrecognised model control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-8bit) |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | usable with caveats | Unrecognised model control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-mxfp8) |
| mlx-community/GLM-4.6V-nvfp4 | usable with caveats | Unrecognised model control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-46v-nvfp4) |
| mlx-community/Idefics3-8B-Llama3-bf16 | unusable | Missing or empty fields: Title, Description, Keywords; Conversation-role control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-idefics3-8b-llama3-bf16) |
| mlx-community/GLM-4.1V-9B-Thinking-8bit | unusable | Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Title has 21 words (requested 5-10) | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-41v-9b-thinking-8bit) |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 | usable with caveats | Conversation-role control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16) |

## Clean completions

20 clean completions (`Qwen/Qwen3-VL-2B-Instruct`, `mlx-community/InternVL3-8B-bf16`, `mlx-community/LFM2.5-VL-1.6B-bf16`, `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`, `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`, `mlx-community/Ministral-3-3B-Instruct-2512-4bit`, `mlx-community/Molmo-7B-D-0924-8bit`, `mlx-community/MolmoPoint-8B-fp16`, `mlx-community/Phi-3.5-vision-instruct-bf16`, `mlx-community/Qwen3-VL-2B-Instruct-bf16`, `mlx-community/Qwen3-VL-2B-Thinking-bf16`, `mlx-community/Qwen3.5-35B-A3B-4bit`, `mlx-community/Qwen3.5-9B-MLX-4bit`, `mlx-community/Qwen3.6-27B-mxfp8`, `mlx-community/Step-3.7-Flash-oQ2e`, `mlx-community/X-Reasoner-7B-8bit`, `mlx-community/gemma-3-27b-it-qat-4bit`, `mlx-community/gemma-4-26b-a4b-it-4bit`, `mlx-community/gemma-4-31b-it-4bit`, `mlx-community/pixtral-12b-8bit`); 12 more completed with prompt-compliance observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 640 x 480 pixels, 173,131 bytes
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: temperature:* 0.0
- *Generation: top_p:* 1.0
- *Trust remote code:* true
- *check_models version:* 0.10.0
- *check_models revision:* 519ba509eb8dbc03125dc283e5936c4ca584b953
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
