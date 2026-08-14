# mlx-vlm compatibility findings across 41 cached vision-language models

## Run summary

- *Run timestamp:* 2026-08-14 18:45:04 BST
- *Evaluation mode:* assisted
- *Models attempted:* 41
- *Completed:* 40
- *Crashed:* 1
- *Indeterminate:* 0
- *Crashes requiring action:* 1
- *Other results requiring review:* 10

Observations are mechanical facts from one image, not general model-quality
judgements.

## Model quality at a glance

Every attempted model ranked by current-run usability, with captured resource
facts. Usability reflects this single image and prompt only; the model gallery
holds full outputs and the diagnostics report holds maintainer evidence.

| Model | Usability | Total | Gen tok/s | Peak GB | Observed |
| --- | --- | --- | --- | --- | --- |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | usable | 1.47s | 515 tok/s | 1.2 | none |
| mlx-community/gemma-4-26b-a4b-it-4bit | usable | 4.14s | 129 tok/s | 16 | none |
| mlx-community/gemma-4-31b-it-4bit | usable | 7.48s | 27.2 tok/s | 20 | none |
| mlx-community/LFM2.5-VL-1.6B-bf16 | usable | 2.87s | 156 tok/s | 4.1 | none |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | usable | 13.46s | 11.7 tok/s | 15 | none |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | usable | 8.19s | 45.6 tok/s | 12 | none |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | usable | 4.29s | 144 tok/s | 7.1 | none |
| mlx-community/Ornith-1.0-35B-bf16 | usable | 93.40s | 60.1 tok/s | 74 | none |
| mlx-community/Qwen3.5-35B-A3B-4bit | usable | 87.25s | 108 tok/s | 24 | none |
| mlx-community/Qwen3.5-9B-MLX-4bit | usable | 75.86s | 87.7 tok/s | 10.0 | none |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | usable with caveats | 9.83s | 29.6 tok/s | 22 | title/keyword constraints failed |
| mlx-community/diffusiongemma-26B-A4B-it-8bit | usable with caveats | 5.96s | 55.3 tok/s | 29 | control tokens visible |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | usable with caveats | 5.91s | 53.0 tok/s | 28 | control tokens visible; title/keyword constraints failed |
| mlx-community/gemma-3-27b-it-qat-4bit | usable with caveats | 8.98s | 26.6 tok/s | 18 | title/keyword constraints failed |
| mlx-community/GLM-4.6V-Flash-mxfp4 | usable with caveats | 10.58s | 74.5 tok/s | 8.4 | title/keyword constraints failed |
| mlx-community/GLM-4.6V-nvfp4 | usable with caveats | 26.92s | 37.5 tok/s | 77 | control tokens visible; title/keyword constraints failed |
| mlx-community/Idefics3-8B-Llama3-bf16 | usable with caveats | 8.22s | 30.3 tok/s | 18 | role tokens visible |
| mlx-community/InternVL3-8B-bf16 | usable with caveats | 6.06s | 34.1 tok/s | 17 | title/keyword constraints failed |
| mlx-community/MiniCPM-V-4.6-8bit | usable with caveats | 2.61s | 238 tok/s | 3.8 | title/keyword constraints failed |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4 | usable with caveats | 10.20s | 41.2 tok/s | 13 | title/keyword constraints failed |
| mlx-community/Molmo-7B-D-0924-8bit | usable with caveats | 6.24s | 38.8 tok/s | 11 | title/keyword constraints failed |
| mlx-community/Phi-3.5-vision-instruct-bf16 | usable with caveats | 3.31s | 56.3 tok/s | 9.6 | title/keyword constraints failed |
| mlx-community/pixtral-12b-8bit | usable with caveats | 6.59s | 39.6 tok/s | 15 | title/keyword constraints failed |
| mlx-community/Qwen3-VL-2B-Instruct-bf16 | usable with caveats | 32.29s | 83.5 tok/s | 8.4 | title/keyword constraints failed |
| mlx-community/Qwen3.6-27B-mxfp8 | usable with caveats | 114.12s | 14.4 tok/s | 35 | title/keyword constraints failed |
| mlx-community/Step-3.7-Flash-oQ2e | usable with caveats | 39.57s | 39.9 tok/s | 70 | title/keyword constraints failed |
| Qwen/Qwen3-VL-2B-Instruct | usable with caveats | 17.08s | 94.2 tok/s | 8.4 | title/keyword constraints failed |
| mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX | unusable | 28.00s | 41.9 tok/s | 14 | missing required fields; echoes instructions; cut off at token limit |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 | unusable | 32.00s | 43.7 tok/s | 60 | repeated text; missing required fields; cut off at token limit; incomplete thinking block |
| mlx-community/FastVLM-0.5B-bf16 | unusable | 2.45s | 338 tok/s | 2.1 | missing required fields; echoes instructions |
| mlx-community/gemma-3n-E4B-it-bf16 | unusable | 7.46s | 47.4 tok/s | 17 | missing required fields |
| mlx-community/GLM-4.1V-9B-Thinking-8bit | unusable | 34.74s | 40.6 tok/s | 13 | echoes instructions; extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 | unusable | 220.02s | 4.66 tok/s | 40 | repeated text; echoes instructions; extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed |
| mlx-community/llava-v1.6-mistral-7b-8bit | unusable | 5.48s | 60.6 tok/s | 9.7 | missing required fields |
| mlx-community/MolmoPoint-8B-fp16 | unusable | 40.08s | 5.64 tok/s | 24 | missing required fields |
| mlx-community/nanoLLaVA-1.5-4bit | unusable | 1.56s | 351 tok/s | 2.4 | missing required fields; echoes instructions |
| mlx-community/paligemma2-3b-pt-896-4bit | unusable | 24.82s | 46.0 tok/s | 4.4 | repeated text; missing required fields; cut off at token limit |
| mlx-community/Qwen2-VL-2B-Instruct-4bit | unusable | 100.13s | 144 tok/s | 5.1 | missing required fields |
| mlx-community/Qwen3-VL-2B-Thinking-bf16 | unusable | 44.58s | 68.1 tok/s | 8.4 | missing required fields; cut off at token limit; incomplete thinking block |
| mlx-community/X-Reasoner-7B-8bit | unusable | 44.52s | 46.8 tok/s | 13 | repeated text; cut off at token limit; title/keyword constraints failed |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | not evaluated | 2.03s | - | - | crashed during decode |

## Crashes requiring action

### mlx-community/SmolVLM2-2.2B-Instruct-mlx

- *Execution / usability:* crashed / not evaluated
- *Phase:* decode
- *Stage:* Model Error
- *Resolved revision:* 844516024a1c4400d34489b89ee067d794e432ed

Root exception chain

```text
ValueError: not enough values to unpack (expected 3, got 2)
caused by: ValueError: Model generation failed for mlx-community/SmolVLM2-2.2B-Instruct-mlx: not enough values to unpack (expected 3, got 2)
```

#### Reproduction inputs

- *Image format:* JPEG
- *Image dimensions:* 9,836 x 5,952 pixels
- *Image size:* 60,138,414 bytes
- *Image SHA-256:* bc3fa055e1e116232f77aa68c1d8d22130a1f596762a98cb8cd691667bbdcab2

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-08-13 17:23:18 UTC+01:00
- GPS: 51.957967°N, 1.346900°E

Descriptive hints:
- Title hint: Seafront, Felixstowe, England, UK, GBR, Europe
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Keyword hints: Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, gbr, seafront

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

The original local input is not published, so this report does not claim a
complete reproduction command. Use a shareable equivalent image or add the
original image before filing.

| Evidence | Link |
| --- | --- |
| Full diagnostics | [model evidence](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-smolvlm2-22b-instruct-mlx) |
| Detailed issue draft | [crash draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_mlx-community_SmolVLM2-2.2B-Instruct-mlx.md) |

## Observation clusters

Repeated mechanical observation signatures among results requiring review.

| Observed result | Models |
| --- | --- |
| Response repeats the same text; Required fields are missing or empty; Response appears cut off at the token limit | 1 |
| Response repeats the same text; Required fields are missing or empty; Response appears cut off at the token limit; Internal reasoning block appears incomplete | 1 |
| Response repeats the same text; Response appears cut off at the token limit; Title or keywords do not meet requested constraints | 1 |
| Response repeats the same text; Response repeats the task instructions instead of only returning the requested fields; Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Title or keywords do not meet requested constraints | 1 |
| Unrecognised model control tokens remain visible; Title or keywords do not meet requested constraints | 2 |
| Unrecognised model control tokens remain visible | 1 |
| Required fields are missing or empty; Response appears cut off at the token limit; Internal reasoning block appears incomplete | 1 |
| Response repeats the task instructions instead of only returning the requested fields; Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Title or keywords do not meet requested constraints | 1 |
| Conversation-role control tokens remain visible | 1 |

## Completed attempts requiring review

| Model | Usability | Observed result | Evidence |
| --- | --- | --- | --- |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 | unusable | Response repeats the same text; Missing or empty fields: Title, Description, Keywords; Response appears cut off at the token limit; Internal reasoning block appears incomplete | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16) |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 | unusable | Response repeats the same text; Response repeats the task instructions instead of only returning the requested fields; Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Title has 11 words (requested 5-10); Keyword list has 85 terms (requested 10-18); Duplicate keywords: seafront, england, uk, gbr, europe, rocky shoreline, people, coastal buildings, clear sky, calm sea, birds, beachgoers, utc 01 00, any vision, 2026 08 13, adobe stock | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16) |
| mlx-community/paligemma2-3b-pt-896-4bit | unusable | Response repeats the same text; Missing or empty fields: Title, Description, Keywords; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-paligemma2-3b-pt-896-4bit) |
| mlx-community/X-Reasoner-7B-8bit | unusable | Response repeats the same text; Response appears cut off at the token limit; Keyword list has 179 terms (requested 10-18); Duplicate keywords: promenade, east suffolk coastline, east suffolk beach, felixstowe historic buildings, felixstowe historic architecture, felixstowe greenery, felixstowe historic structures, felixstowe historic town, east suffolk beachgoers, east suffolk beach buildings, uk seafront greenery, uk beach greenery | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-x-reasoner-7b-8bit) |
| mlx-community/diffusiongemma-26B-A4B-it-8bit | usable with caveats | Unrecognised model control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-8bit) |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | usable with caveats | Unrecognised model control tokens remain visible; Duplicate keywords: coastal | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-mxfp8) |
| mlx-community/GLM-4.6V-nvfp4 | usable with caveats | Unrecognised model control tokens remain visible; Title has 3 words (requested 5-10) | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-46v-nvfp4) |
| mlx-community/Qwen3-VL-2B-Thinking-bf16 | unusable | Missing or empty fields: Title, Description, Keywords; Response appears cut off at the token limit; Internal reasoning block appears incomplete | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-qwen3-vl-2b-thinking-bf16) |
| mlx-community/GLM-4.1V-9B-Thinking-8bit | unusable | Response repeats the task instructions instead of only returning the requested fields; Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Title has 4 words (requested 5-10); Keyword list has 69 terms (requested 10-18); Duplicate keywords: felixstowe, suffolk, uk, seafront, coastal steps, beach, people, swimming, historic buildings, daylight, europe, east suffolk, england, gbr, seagulls, seaside, wait, with people swimming and walking near distinctive coastal steps, under clear daylight with historic buildings in the background | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-41v-9b-thinking-8bit) |
| mlx-community/Idefics3-8B-Llama3-bf16 | usable with caveats | Conversation-role control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-idefics3-8b-llama3-bf16) |

## Clean completions

10 clean completions (`LiquidAI/LFM2.5-VL-450M-MLX-bf16`, `mlx-community/LFM2.5-VL-1.6B-bf16`, `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`, `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`, `mlx-community/Ministral-3-3B-Instruct-2512-4bit`, `mlx-community/Ornith-1.0-35B-bf16`, `mlx-community/Qwen3.5-35B-A3B-4bit`, `mlx-community/Qwen3.5-9B-MLX-4bit`, `mlx-community/gemma-4-26b-a4b-it-4bit`, `mlx-community/gemma-4-31b-it-4bit`); 20 more completed with prompt-compliance observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 9,836 x 5,952 pixels, 60,138,414 bytes
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: temperature:* 0.0
- *Generation: top_p:* 1.0
- *Trust remote code:* true
- *check_models version:* 0.9.1
- *check_models revision:* 31afd35eed71ff34a8310726bf24b8e2719b2403
- *check_models source dirty:* false
- *mlx-vlm:* 0.6.14
- *mlx:* 0.32.1.dev20260814+3d23f7d87
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
