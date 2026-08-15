# mlx-vlm compatibility findings across 41 cached vision-language models

## Run summary

- *Run timestamp:* 2026-08-16 00:48:24 BST
- *Evaluation mode:* assisted
- *Models attempted:* 41
- *Completed:* 40
- *Crashed:* 1
- *Indeterminate:* 0
- *Crashes requiring action:* 1
- *Other results requiring review:* 11

Observations are mechanical facts from one image, not general model-quality
judgements.

## Model quality at a glance

Every attempted model ranked by current-run usability, with captured resource
facts. Usability reflects this single image and prompt only; the model gallery
holds full outputs and the diagnostics report holds maintainer evidence.

| Model | Usability | Total | Gen tok/s | Peak GB | Observed |
| --- | --- | --- | --- | --- | --- |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | usable | 2.21s | 503 tok/s | 1.1 | none |
| mlx-community/gemma-4-26b-a4b-it-4bit | usable | 5.15s | 91.3 tok/s | 16 | none |
| mlx-community/gemma-4-31b-it-4bit | usable | 10.04s | 16.3 tok/s | 20 | none |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | usable | 11.16s | 15.0 tok/s | 15 | none |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | usable | 7.70s | 58.3 tok/s | 14 | none |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | usable | 3.60s | 172 tok/s | 9.0 | none |
| mlx-community/Ornith-1.0-35B-bf16 | usable | 113.30s | 48.9 tok/s | 74 | none |
| mlx-community/Phi-3.5-vision-instruct-bf16 | usable | 4.44s | 50.9 tok/s | 9.4 | none |
| mlx-community/Qwen3.5-35B-A3B-4bit | usable | 59.31s | 90.4 tok/s | 24 | none |
| mlx-community/Qwen3.6-27B-mxfp8 | usable | 125.25s | 13.0 tok/s | 35 | none |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | usable with caveats | 14.63s | 22.2 tok/s | 24 | title/keyword constraints failed |
| mlx-community/diffusiongemma-26B-A4B-it-8bit | usable with caveats | 6.05s | 62.0 tok/s | 29 | control tokens visible |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | usable with caveats | 5.61s | 66.2 tok/s | 28 | control tokens visible |
| mlx-community/gemma-3-27b-it-qat-4bit | usable with caveats | 9.05s | 25.2 tok/s | 18 | title/keyword constraints failed |
| mlx-community/GLM-4.6V-Flash-mxfp4 | usable with caveats | 14.03s | 67.3 tok/s | 8.4 | title/keyword constraints failed |
| mlx-community/GLM-4.6V-nvfp4 | usable with caveats | 50.48s | 26.7 tok/s | 78 | control tokens visible; title/keyword constraints failed |
| mlx-community/Idefics3-8B-Llama3-bf16 | usable with caveats | 10.10s | 27.6 tok/s | 18 | role tokens visible; title/keyword constraints failed |
| mlx-community/InternVL3-8B-bf16 | usable with caveats | 7.97s | 25.9 tok/s | 17 | title/keyword constraints failed |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4 | usable with caveats | 7.81s | 60.6 tok/s | 15 | title/keyword constraints failed |
| mlx-community/Molmo-7B-D-0924-8bit | usable with caveats | 5.17s | 48.3 tok/s | 11 | title/keyword constraints failed |
| mlx-community/pixtral-12b-8bit | usable with caveats | 9.44s | 34.1 tok/s | 16 | title/keyword constraints failed |
| mlx-community/Qwen2-VL-2B-Instruct-4bit | usable with caveats | 107.81s | 186 tok/s | 5.1 | title/keyword constraints failed; draft hints copied unchanged |
| mlx-community/Qwen3.5-9B-MLX-4bit | usable with caveats | 67.65s | 83.8 tok/s | 10.0 | title/keyword constraints failed |
| mlx-community/Step-3.7-Flash-oQ2e | usable with caveats | 39.54s | 36.1 tok/s | 70 | title/keyword constraints failed |
| mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX | unusable | 36.13s | 35.2 tok/s | 15 | missing required fields; echoes instructions; cut off at token limit |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 | unusable | 57.57s | 29.1 tok/s | 60 | extra text before Title; title/keyword constraints failed |
| mlx-community/FastVLM-0.5B-bf16 | unusable | 3.61s | 290 tok/s | 2.1 | missing required fields |
| mlx-community/gemma-3n-E4B-it-bf16 | unusable | 9.93s | 40.2 tok/s | 17 | missing required fields |
| mlx-community/GLM-4.1V-9B-Thinking-8bit | unusable | 42.63s | 34.0 tok/s | 13 | repeated text; missing required fields; extra text before Title; cut off at token limit; incomplete thinking block |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 | unusable | 189.23s | 4.56 tok/s | 40 | repeated text; missing required fields; extra text before Title; role tokens visible |
| mlx-community/LFM2.5-VL-1.6B-bf16 | unusable | 7.89s | 164 tok/s | 4.1 | repeated text; cut off at token limit; title/keyword constraints failed |
| mlx-community/llava-v1.6-mistral-7b-8bit | unusable | 6.14s | 57.3 tok/s | 9.7 | missing required fields |
| mlx-community/MiniCPM-V-4.6-8bit | unusable | 2.21s | 247 tok/s | 3.7 | missing required fields; extra text before Title |
| mlx-community/MolmoPoint-8B-fp16 | unusable | 24.23s | 6.27 tok/s | 24 | missing required fields |
| mlx-community/nanoLLaVA-1.5-4bit | unusable | 2.54s | 186 tok/s | 2.1 | missing required fields |
| mlx-community/paligemma2-3b-pt-896-4bit | unusable | 26.82s | 32.1 tok/s | 4.3 | repeated text; missing required fields; echoes instructions |
| mlx-community/Qwen3-VL-2B-Instruct-bf16 | unusable | 32.66s | 92.1 tok/s | 8.4 | repeated text; title/keyword constraints failed |
| mlx-community/Qwen3-VL-2B-Thinking-bf16 | unusable | 38.17s | 80.9 tok/s | 8.4 | extra text before Title; title/keyword constraints failed |
| mlx-community/X-Reasoner-7B-8bit | unusable | 49.21s | 45.8 tok/s | 13 | repeated text; cut off at token limit; title/keyword constraints failed |
| Qwen/Qwen3-VL-2B-Instruct | unusable | 20.17s | 86.8 tok/s | 8.4 | repeated text; title/keyword constraints failed |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | not evaluated | 2.94s | - | - | crashed during decode |

## Crashes requiring action

### mlx-community/SmolVLM2-2.2B-Instruct-mlx

- *Execution / usability:* crashed / not evaluated
- *Phase:* decode
- *Stage:* Model Error
- *Resolved revision:* 844516024a1c4400d34489b89ee067d794e432ed

Root exception chain

```text
ValueError: Image features and image tokens do not match: tokens: 81, features 1053
caused by: ValueError: Model generation failed for mlx-community/SmolVLM2-2.2B-Instruct-mlx: Image features and image tokens do not match: tokens: 81, features 1053
```

#### Reproduction inputs

- *Image format:* JPEG
- *Image dimensions:* 6,656 x 8,880 pixels
- *Image size:* 79,069,278 bytes
- *Image SHA-256:* 771ab1bcadbb99020fb1a6270d6f36e8dd613cc3132c390bed714290bda2dd05

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-08-15 15:59:46 UTC+01:00
- GPS: 51.128800°N, 1.319100°E

Descriptive hints:
- Title hint: Dover Castle, Dover, England, UK, GBR, Europe
- Description hint: An exterior view of a historic medieval stone castle, featuring round towers, an arched entranceway, and a small bridge, built on a steep grassy hill under a partly cloudy sky.
- Keyword hints: Adobe Stock, Any Vision, Arch, Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent, Sky, Stone, Tower, UK, United Kingdom, Wall, ancient, architecture, blue

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
| Response repeats the same text; Response appears cut off at the token limit; Title or keywords do not meet requested constraints | 2 |
| Response repeats the same text; Title or keywords do not meet requested constraints | 2 |
| Response repeats the same text; Required fields are missing or empty; Response repeats the task instructions instead of only returning the requested fields | 1 |
| Response repeats the same text; Required fields are missing or empty; Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete | 1 |
| Response repeats the same text; Required fields are missing or empty; Extra text appears before the Title field; Conversation-role control tokens remain visible | 1 |
| Unrecognised model control tokens remain visible | 2 |
| Unrecognised model control tokens remain visible; Title or keywords do not meet requested constraints | 1 |
| Conversation-role control tokens remain visible; Title or keywords do not meet requested constraints | 1 |

## Completed attempts requiring review

| Model | Usability | Observed result | Evidence |
| --- | --- | --- | --- |
| mlx-community/GLM-4.1V-9B-Thinking-8bit | unusable | Response repeats the same text; Missing or empty fields: Title; Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-41v-9b-thinking-8bit) |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 | unusable | Response repeats the same text; Missing or empty fields: Title; Extra text appears before the Title field; Conversation-role control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16) |
| mlx-community/LFM2.5-VL-1.6B-bf16 | unusable | Response repeats the same text; Response appears cut off at the token limit; Title has 13 words (requested 5-10); Keyword list has 454 terms (requested 10-18); Duplicate keywords: uk, architecture, blue, castle | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-lfm25-vl-16b-bf16) |
| mlx-community/paligemma2-3b-pt-896-4bit | unusable | Response repeats the same text; Missing or empty fields: Title, Description, Keywords; Response repeats the task instructions instead of only returning the requested fields | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-paligemma2-3b-pt-896-4bit) |
| mlx-community/Qwen3-VL-2B-Instruct-bf16 | unusable | Response repeats the same text; Title has 3 words (requested 5-10); Keyword list has 33 terms (requested 10-18); Duplicate keywords: england, uk, europe, fortress, castle, stone, tower, wall, hill, sky, ancient, architecture, united kingdom | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-qwen3-vl-2b-instruct-bf16) |
| mlx-community/X-Reasoner-7B-8bit | unusable | Response repeats the same text; Response appears cut off at the token limit; Title has 4 words (requested 5-10); Keyword list has 357 terms (requested 10-18); Duplicate keywords: kent, england, uk, europe, medieval, stone, round towers, small bridge, grassy hill, partly cloudy sky, architecture, ancient, arched entranceway, historic, fortress | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-x-reasoner-7b-8bit) |
| Qwen/Qwen3-VL-2B-Instruct | unusable | Response repeats the same text; Title has 3 words (requested 5-10); Keyword list has 33 terms (requested 10-18); Duplicate keywords: england, uk, europe, fortress, castle, stone, tower, wall, hill, sky, ancient, architecture, united kingdom | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-qwen-qwen3-vl-2b-instruct) |
| mlx-community/diffusiongemma-26B-A4B-it-8bit | usable with caveats | Unrecognised model control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-8bit) |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | usable with caveats | Unrecognised model control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-mxfp8) |
| mlx-community/GLM-4.6V-nvfp4 | usable with caveats | Unrecognised model control tokens remain visible; Title has 4 words (requested 5-10) | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-46v-nvfp4) |
| mlx-community/Idefics3-8B-Llama3-bf16 | usable with caveats | Conversation-role control tokens remain visible; Keyword list has 24 terms (requested 10-18); Duplicate keywords: europe, sky | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-idefics3-8b-llama3-bf16) |

## Clean completions

10 clean completions (`LiquidAI/LFM2.5-VL-450M-MLX-bf16`, `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`, `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`, `mlx-community/Ministral-3-3B-Instruct-2512-4bit`, `mlx-community/Ornith-1.0-35B-bf16`, `mlx-community/Phi-3.5-vision-instruct-bf16`, `mlx-community/Qwen3.5-35B-A3B-4bit`, `mlx-community/Qwen3.6-27B-mxfp8`, `mlx-community/gemma-4-26b-a4b-it-4bit`, `mlx-community/gemma-4-31b-it-4bit`); 19 more completed with prompt-compliance observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 6,656 x 8,880 pixels, 79,069,278 bytes
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: temperature:* 0.0
- *Generation: top_p:* 1.0
- *Trust remote code:* true
- *check_models version:* 0.10.0
- *check_models revision:* 3dd40931dfdffb61563b75d4782104e7bd2c2f6a
- *check_models source dirty:* false
- *mlx-vlm:* 0.6.14
- *mlx:* 0.32.1.dev20260815+9ab977b56
- *transformers:* 5.15.0
- *macOS Version:* 26.6.1
- *GPU/Chip:* Apple M5 Max
- *Python Version:* 3.13.14

GitHub links target the repository's mutable main branch; they resolve to this
run's evidence only once these artifacts are committed, and a later run's
commit supersedes them. Pin links to that artifact commit when durable issue
evidence is required.

## Full artifacts

Stale retained artifacts omitted because their timestamps fall outside this
run: `check_models.log`, `environment.log`.

| Artifact | Link |
| --- | --- |
| Diagnostics | [diagnostics.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md) |
| Model gallery | [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md) |
| Results JSONL | [results.jsonl](https://github.com/jrp2014/check_models/blob/main/src/output/results.jsonl) |
| Run JSON | [run.json](https://github.com/jrp2014/check_models/blob/main/src/output/run.json) |
