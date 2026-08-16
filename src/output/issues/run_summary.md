# mlx-vlm compatibility findings across 41 cached vision-language models

## Run summary

- *Run timestamp:* 2026-08-16 17:50:01 BST
- *Evaluation mode:* blind
- *Models attempted:* 41
- *Completed:* 41
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
| mlx-community/gemma-3-27b-it-qat-4bit | usable | 5.91s | 31.6 tok/s | 18 | none |
| mlx-community/gemma-4-26b-a4b-it-4bit | usable | 3.28s | 130 tok/s | 16 | none |
| mlx-community/gemma-4-31b-it-4bit | usable | 6.28s | 28.1 tok/s | 19 | none |
| mlx-community/InternVL3-8B-bf16 | usable | 4.60s | 34.4 tok/s | 17 | none |
| mlx-community/LFM2.5-VL-1.6B-bf16 | usable | 1.23s | 185 tok/s | 4.1 | none |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | usable | 3.21s | 69.5 tok/s | 9.8 | none |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4 | usable | 3.76s | 64.6 tok/s | 10 | none |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | usable | 1.84s | 193 tok/s | 4.5 | none |
| mlx-community/Molmo-7B-D-0924-8bit | usable | 3.32s | 53.7 tok/s | 11 | none |
| mlx-community/MolmoPoint-8B-fp16 | usable | 15.80s | 6.21 tok/s | 23 | none |
| mlx-community/Phi-3.5-vision-instruct-bf16 | usable | 2.36s | 59.6 tok/s | 9.3 | none |
| mlx-community/pixtral-12b-8bit | usable | 4.89s | 40.5 tok/s | 15 | none |
| mlx-community/Qwen3-VL-2B-Instruct-bf16 | usable | 1.48s | 129 tok/s | 5.3 | none |
| mlx-community/Qwen3.5-35B-A3B-4bit | usable | 3.60s | 121 tok/s | 21 | none |
| mlx-community/Qwen3.5-9B-MLX-4bit | usable | 2.53s | 101 tok/s | 7.1 | none |
| mlx-community/Qwen3.6-27B-mxfp8 | usable | 9.52s | 19.2 tok/s | 30 | none |
| mlx-community/Step-3.7-Flash-oQ2e | usable | 13.23s | 44.2 tok/s | 65 | none |
| mlx-community/X-Reasoner-7B-8bit | usable | 3.11s | 66.1 tok/s | 10 | none |
| Qwen/Qwen3-VL-2B-Instruct | usable | 1.36s | 138 tok/s | 5.2 | none |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | usable with caveats | 0.64s | 516 tok/s | 1.3 | title/keyword constraints failed |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | usable with caveats | 6.33s | 31.2 tok/s | 20 | title/keyword constraints failed |
| mlx-community/diffusiongemma-26B-A4B-it-8bit | usable with caveats | 4.92s | 60.8 tok/s | 29 | control tokens visible; title/keyword constraints failed |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | usable with caveats | 4.90s | 59.3 tok/s | 28 | control tokens visible |
| mlx-community/gemma-3n-E4B-it-bf16 | usable with caveats | 4.08s | 47.8 tok/s | 17 | title/keyword constraints failed |
| mlx-community/GLM-4.6V-Flash-mxfp4 | usable with caveats | 2.33s | 92.0 tok/s | 7.8 | title/keyword constraints failed |
| mlx-community/GLM-4.6V-nvfp4 | usable with caveats | 13.99s | 52.6 tok/s | 63 | control tokens visible |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 | usable with caveats | 121.10s | 4.83 tok/s | 40 | role tokens visible |
| mlx-community/Ornith-1.0-35B-bf16 | usable with caveats | 18.97s | 66.2 tok/s | 71 | title/keyword constraints failed |
| mlx-community/Qwen2-VL-2B-Instruct-4bit | usable with caveats | 0.83s | 307 tok/s | 2.5 | title/keyword constraints failed |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | usable with caveats | 1.34s | 125 tok/s | 5.5 | title/keyword constraints failed |
| mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX | unusable | 25.25s | 44.0 tok/s | 14 | missing required fields; echoes instructions; extra text before Title; cut off at token limit |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 | unusable | 27.62s | 57.9 tok/s | 60 | repeated text; extra text before Title; title/keyword constraints failed |
| mlx-community/FastVLM-0.5B-bf16 | unusable | 1.03s | 343 tok/s | 2.2 | missing required fields |
| mlx-community/GLM-4.1V-9B-Thinking-8bit | unusable | 21.89s | 50.4 tok/s | 13 | extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed |
| mlx-community/Idefics3-8B-Llama3-bf16 | unusable | 3.37s | insufficient sample | 18 | missing required fields; role tokens visible |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | unusable | 65.99s | 15.9 tok/s | 15 | repeated text; extra text before Title; cut off at token limit; title/keyword constraints failed |
| mlx-community/llava-v1.6-mistral-7b-8bit | unusable | 21.33s | 54.7 tok/s | 9.7 | repeated text; missing required fields; cut off at token limit |
| mlx-community/MiniCPM-V-4.6-8bit | unusable | 1.22s | 278 tok/s | 3.0 | missing required fields; extra text before Title |
| mlx-community/nanoLLaVA-1.5-4bit | unusable | 0.83s | 381 tok/s | 2.1 | missing required fields |
| mlx-community/paligemma2-3b-pt-896-4bit | unusable | 23.79s | 47.0 tok/s | 4.2 | repeated text; missing required fields; echoes instructions; extra text before Title; cut off at token limit |
| mlx-community/Qwen3-VL-2B-Thinking-bf16 | unusable | 7.62s | 130 tok/s | 5.3 | echoes instructions; extra text before Title |

## Observation clusters

Repeated mechanical observation signatures among results requiring review.

| Observed result | Models |
| --- | --- |
| Response repeats the same text; Required fields are missing or empty; Response appears cut off at the token limit | 1 |
| Response repeats the same text; Extra text appears before the Title field; Title or keywords do not meet requested constraints | 1 |
| Response repeats the same text; Required fields are missing or empty; Response repeats the task instructions instead of only returning the requested fields; Extra text appears before the Title field; Response appears cut off at the token limit | 1 |
| Response repeats the same text; Extra text appears before the Title field; Response appears cut off at the token limit; Title or keywords do not meet requested constraints | 1 |
| Unrecognised model control tokens remain visible | 2 |
| Unrecognised model control tokens remain visible; Title or keywords do not meet requested constraints | 1 |
| Required fields are missing or empty; Conversation-role control tokens remain visible | 1 |
| Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Title or keywords do not meet requested constraints | 1 |
| Conversation-role control tokens remain visible | 1 |

## Completed attempts requiring review

| Model | Usability | Observed result | Evidence |
| --- | --- | --- | --- |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 | unusable | Response repeats the same text; Extra text appears before the Title field; Keyword list has 33 terms (requested 10-18); Duplicate keywords: couch, pet, sofa, remote, television, television remote, blurry, indoor, home, furniture, domestic cat, animal sleeping | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16) |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | unusable | Response repeats the same text; Extra text appears before the Title field; Response appears cut off at the token limit; Keyword list has 313 terms (requested 10-18); Duplicate keywords: household frames | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llama-32-11b-vision-instruct-8bit) |
| mlx-community/llava-v1.6-mistral-7b-8bit | unusable | Response repeats the same text; Missing or empty fields: Title, Description, Keywords; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llava-v16-mistral-7b-8bit) |
| mlx-community/paligemma2-3b-pt-896-4bit | unusable | Response repeats the same text; Missing or empty fields: Title; Response repeats the task instructions instead of only returning the requested fields; Extra text appears before the Title field; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-paligemma2-3b-pt-896-4bit) |
| mlx-community/diffusiongemma-26B-A4B-it-8bit | usable with caveats | Unrecognised model control tokens remain visible; Duplicate keywords: domestic | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-8bit) |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | usable with caveats | Unrecognised model control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-mxfp8) |
| mlx-community/GLM-4.6V-nvfp4 | usable with caveats | Unrecognised model control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-46v-nvfp4) |
| mlx-community/Idefics3-8B-Llama3-bf16 | unusable | Missing or empty fields: Title, Description, Keywords; Conversation-role control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-idefics3-8b-llama3-bf16) |
| mlx-community/GLM-4.1V-9B-Thinking-8bit | unusable | Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Title has 21 words (requested 5-10) | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-41v-9b-thinking-8bit) |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 | usable with caveats | Conversation-role control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16) |

## Clean completions

19 clean completions (`Qwen/Qwen3-VL-2B-Instruct`, `mlx-community/InternVL3-8B-bf16`, `mlx-community/LFM2.5-VL-1.6B-bf16`, `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`, `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`, `mlx-community/Ministral-3-3B-Instruct-2512-4bit`, `mlx-community/Molmo-7B-D-0924-8bit`, `mlx-community/MolmoPoint-8B-fp16`, `mlx-community/Phi-3.5-vision-instruct-bf16`, `mlx-community/Qwen3-VL-2B-Instruct-bf16`, `mlx-community/Qwen3.5-35B-A3B-4bit`, `mlx-community/Qwen3.5-9B-MLX-4bit`, `mlx-community/Qwen3.6-27B-mxfp8`, `mlx-community/Step-3.7-Flash-oQ2e`, `mlx-community/X-Reasoner-7B-8bit`, `mlx-community/gemma-3-27b-it-qat-4bit`, `mlx-community/gemma-4-26b-a4b-it-4bit`, `mlx-community/gemma-4-31b-it-4bit`, `mlx-community/pixtral-12b-8bit`); 12 more completed with prompt-compliance observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 640 x 480 pixels, 173,131 bytes
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: temperature:* 0.0
- *Generation: top_p:* 1.0
- *Trust remote code:* true
- *check_models version:* 0.10.0
- *check_models revision:* 157c9b18be54da6d0bb09fc96208e9fb46078f91
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
