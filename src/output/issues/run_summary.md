# mlx-vlm compatibility findings across 41 cached vision-language models

## Run summary

- *Run timestamp:* 2026-08-29 02:11:28 BST
- *Evaluation mode:* assisted
- *Models attempted:* 41
- *Completed:* 39
- *Crashed:* 2
- *Indeterminate:* 0
- *Crashes requiring action:* 2
- *Other results requiring review:* 9
- *Hit the token cap:* 2
- *Stopped early for repetition:* 4

Observations are mechanical facts from one image, not general model-quality
judgements.

## Since the baseline sweep

- *Baseline:* 701362c5:src/output/results.jsonl
- *Baseline run timestamp:* 2026-08-28 20:07:01 BST
- *Baseline check_models:* 0.15.0 @ 16c1d8663
- *Baseline mlx:* 0.32.3.dev20260828+99e45f71d @ 99e45f71d
- *Baseline mlx-vlm:* 0.7.0rc0 @ 24b244ee2
- *Baseline transformers:* 5.16.1
- *Baseline python:* 3.14.7
- *Models compared:* 41
- *Identical generated text:* 33 of 39 completed in both
- *Generation tok/s ratio (now/baseline):* 0.956 (range 0.42-2.16, 35 models)
- *Throughput noise band:* fixed ±15% fallback (insufficient history)

| Model | Execution | Usability | Observation delta |
| --- | --- | --- | --- |
| mlx-community/LFM2.5-VL-1.6B-bf16 | completed | unusable | +stopped early: repeating; -cut off at token limit |
| jinaai/jina-vlm-mlx | completed | unusable | +stopped early: repeating; -cut off at token limit |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 | completed | unusable | +stopped early: repeating; -cut off at token limit |
| mlx-community/X-Reasoner-7B-8bit | completed | unusable | +stopped early: repeating; -repeated text; -cut off at token limit |

| Model | Baseline tok/s | Now tok/s | Ratio | Expected band |
| --- | --- | --- | --- | --- |
| mlx-community/diffusiongemma-26B-A4B-it-8bit | 53.4 | 70.9 | 1.33 | 45.4-61.4 (fallback) |
| mlx-community/North-Micro-Vision-Instruct-4bit | 230.4 | 166.1 | 0.72 | 195.9-265.0 (fallback) |
| mlx-community/gemma-3-27b-it-qat-4bit | 31.3 | 21.7 | 0.69 | 26.6-36.0 (fallback) |
| mlx-community/Step-3.7-Flash-oQ2e | 18.8 | 40.5 | 2.16 | 16.0-21.6 (fallback) |
| mlx-community/Qwen3-VL-2B-Thinking-bf16 | 90.3 | 46.7 | 0.52 | 76.8-103.9 (fallback) |
| mlx-community/Qwen3.5-35B-A3B-4bit | 110.5 | 65.7 | 0.60 | 93.9-127.0 (fallback) |
| mlx-community/Qwen3.5-9B-MLX-4bit | 89.0 | 37.5 | 0.42 | 75.6-102.3 (fallback) |
| mlx-community/Qwen3.8-27B-4bit | 29.5 | 17.6 | 0.60 | 25.1-33.9 (fallback) |

Mechanical diff only: one image, temperature as configured; single-observation
flips on one model are usually run-to-run variance, broad shifts are not.

## Model quality at a glance

Every attempted model ranked by current-run usability, with captured resource
facts. Usability reflects this single image and prompt only; the model gallery
holds full outputs and the diagnostics report holds maintainer evidence.

| Model | Usability | Total | Gen tok/s | Peak GB | Observed |
| --- | --- | --- | --- | --- | --- |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | usable | 1.74s | 484 tok/s | 1.9 | none |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | usable | 10.55s | 29.0 tok/s | 23 | none |
| mlx-community/gemma-3-27b-it-qat-4bit | usable | 15.17s | 21.7 tok/s | 17 | none |
| mlx-community/gemma-4-26b-a4b-it-4bit | usable | 4.45s | 116 tok/s | 16 | none |
| mlx-community/gemma-4-31b-it-4bit | usable | 7.77s | 27.6 tok/s | 20 | none |
| mlx-community/granite-4.0-3b-vision-4bit | usable | 2.48s | 169 tok/s | 4.6 | none |
| mlx-community/Idefics3-8B-Llama3-bf16 | usable | 9.13s | 32.5 tok/s | 18 | none |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | usable | 6.99s | 65.6 tok/s | 13 | none |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | usable | 3.61s | 189 tok/s | 7.8 | none |
| mlx-community/Molmo2-8B-4bit | usable | 4.47s | 72.0 tok/s | 8.8 | none |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | usable | 63.05s | 79.0 tok/s | 23 | none |
| mlx-community/Qwen3.5-35B-A3B-4bit | usable | 68.39s | 65.7 tok/s | 24 | none |
| mlx-community/diffusiongemma-26B-A4B-it-8bit | usable with caveats | 5.75s | 70.9 tok/s | 29 | control tokens visible |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | usable with caveats | 6.02s | 57.8 tok/s | 28 | control tokens visible |
| mlx-community/GLM-4.6V-Flash-mxfp4 | usable with caveats | 10.28s | 76.1 tok/s | 8.4 | title/keyword constraints failed |
| mlx-community/GLM-4.6V-nvfp4 | usable with caveats | 28.19s | 38.8 tok/s | 78 | control tokens visible |
| mlx-community/InternVL3-8B-bf16 | usable with caveats | 6.08s | 34.6 tok/s | 17 | title/keyword constraints failed |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 | usable with caveats | 158.48s | 4.46 tok/s | 40 | role tokens visible |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4 | usable with caveats | 7.18s | 61.1 tok/s | 13 | title/keyword constraints failed |
| mlx-community/North-Micro-Vision-Instruct-4bit | usable with caveats | 6.01s | 166 tok/s | 3.9 | title/keyword constraints failed |
| mlx-community/Ornith-1.0-35B-bf16 | usable with caveats | 75.73s | 61.2 tok/s | 74 | title/keyword constraints failed; draft hints copied unchanged |
| mlx-community/Phi-3.5-vision-instruct-bf16 | usable with caveats | 4.30s | 54.8 tok/s | 9.4 | title/keyword constraints failed |
| mlx-community/pixtral-12b-8bit | usable with caveats | 7.01s | 39.3 tok/s | 16 | title/keyword constraints failed |
| mlx-community/Qwen3-VL-2B-Thinking-bf16 | usable with caveats | 43.07s | 46.7 tok/s | 8.4 | title/keyword constraints failed |
| mlx-community/Qwen3.5-9B-MLX-4bit | usable with caveats | 85.51s | 37.5 tok/s | 10.0 | title/keyword constraints failed |
| mlx-community/Qwen3.6-27B-mxfp8 | usable with caveats | 111.70s | 15.8 tok/s | 33 | title/keyword constraints failed |
| mlx-community/Qwen3.8-27B-4bit | usable with caveats | 96.92s | 17.6 tok/s | 21 | title/keyword constraints failed |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | usable with caveats | 2.75s | 127 tok/s | 5.5 | title/keyword constraints failed; draft hints copied unchanged |
| mlx-community/Step-3.7-Flash-oQ2e | usable with caveats | 40.16s | 40.5 tok/s | 70 | title/keyword constraints failed; draft hints copied unchanged |
| Qwen/Qwen3-VL-2B-Instruct | usable with caveats | 16.00s | 90.1 tok/s | 8.4 | title/keyword constraints failed |
| jinaai/jina-vlm-mlx | unusable | 5.22s | 85.1 tok/s | 3.7 | repeated text; stopped early: repeating; title/keyword constraints failed |
| mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX | unusable | 28.95s | 40.9 tok/s | 15 | missing required fields; echoes instructions; cut off at token limit |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 | unusable | 19.02s | 60.9 tok/s | 60 | repeated text; stopped early: repeating; title/keyword constraints failed |
| mlx-community/FastVLM-0.5B-bf16 | unusable | 2.46s | 300 tok/s | 2.1 | missing required fields |
| mlx-community/gemma-3n-E4B-it-bf16 | unusable | 6.79s | 47.7 tok/s | 17 | missing required fields |
| mlx-community/GLM-4.1V-9B-Thinking-8bit | unusable | 29.68s | 46.0 tok/s | 13 | repeated text; extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed |
| mlx-community/LFM2.5-VL-1.6B-bf16 | unusable | 5.47s | 192 tok/s | 4.0 | repeated text; stopped early: repeating; title/keyword constraints failed |
| mlx-community/MiniCPM-V-4.6-8bit | unusable | 2.21s | 254 tok/s | 3.8 | missing required fields; extra text before Title |
| mlx-community/X-Reasoner-7B-8bit | unusable | 28.15s | 55.3 tok/s | 13 | stopped early: repeating; title/keyword constraints failed |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit | not evaluated | 0.27s | - | - | crashed during model_load |
| tencent/Youtu-VL-4B-Instruct | not evaluated | 1.57s | - | - | crashed during model_load |

## Constraint-failure breakdown

How the fleet failed the catalogue constraints — a skew toward one constraint
suggests prompt difficulty rather than individual model faults.

- Title length: 4 model(s) outside 5-10 words (4 below, 0 above; median
  observed 4)
- Keyword count: 17 model(s) outside 10-18 (0 below, 17 above; median observed
  20)
- Duplicate keywords: 7 model(s)

## Crashes requiring action

### mlx-community/LFM2.5-VL-3B-OptiQ-4bit

- *Execution / usability:* crashed / not evaluated
- *Phase:* model_load
- *Stage:* Model Error
- *Resolved revision:* 12c5ae49304158b0a133fcea9ba4486a6d6c8cad

Root exception chain

```text
ValueError: Received 600 parameters not in model; families: model; representative parameters: model.embed_tokens.biases, model.embed_tokens.scales, model.embed_tokens.weight.
caused by: ValueError: Model loading failed: Received 600 parameters not in model; families: model; representative parameters: model.embed_tokens.biases, model.embed_tokens.scales, model.embed_tokens.weight.
```

#### Reproduction inputs

- *Image format:* JPEG
- *Image dimensions:* 9,409 x 6,273 pixels
- *Image size:* 51,431,731 bytes
- *Image SHA-256:* dadec238f988c92cd592f7ba686543f85856f67b00665ba8d8d2830881d211b5

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-08-21 14:34:53 UTC+01:00
- GPS: 51.441113°N, 0.565406°W

Descriptive hints:
- Description hint: A weathered wooden boathouse built on stilts stands over the edge of a serene pond, framed by lush overhanging tree branches, wooden decking, and surrounding wetland foliage.
- Keyword hints: Cloudy Sky, Foliage, Forest, Grass, Lake, Landscape, Leaves, Marshland, Moss, Outdoors, Pond, Reeds, Trees, Water reflection, Wetland, Wooden shed, architecture, bird hide, birdwatching, boardwalk

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

The crash occurred during model load, before image decoding, so the exact
input image is not required: substitute any local image for the placeholder
path and run one native mlx-vlm process.

```bash
python -m mlx_vlm.generate --model mlx-community/LFM2.5-VL-3B-OptiQ-4bit --image any-local-image.jpg --prompt x --max-tokens 8 --temperature 0.0 --revision 12c5ae49304158b0a133fcea9ba4486a6d6c8cad --trust-remote-code
```

| Evidence | Link |
| --- | --- |
| Full diagnostics | [model evidence](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-lfm25-vl-3b-optiq-4bit) |
| Detailed issue draft | [crash draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_mlx-community_LFM2.5-VL-3B-OptiQ-4bit.md) |

### tencent/Youtu-VL-4B-Instruct

- *Execution / usability:* crashed / not evaluated
- *Phase:* model_load
- *Stage:* Lib Version
- *Resolved revision:* 8d30a0e49662a1d628a472b12df264dbcd768753

Root exception chain

```text
ImportError: cannot import name 'DefaultFastImageProcessorKwargs' from 'transformers.image_processing_utils_fast' (unknown location)
caused by: ValueError: Model loading failed: cannot import name 'DefaultFastImageProcessorKwargs' from 'transformers.image_processing_utils_fast' (unknown location)
```

#### Reproduction inputs

- *Image format:* JPEG
- *Image dimensions:* 9,409 x 6,273 pixels
- *Image size:* 51,431,731 bytes
- *Image SHA-256:* dadec238f988c92cd592f7ba686543f85856f67b00665ba8d8d2830881d211b5

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-08-21 14:34:53 UTC+01:00
- GPS: 51.441113°N, 0.565406°W

Descriptive hints:
- Description hint: A weathered wooden boathouse built on stilts stands over the edge of a serene pond, framed by lush overhanging tree branches, wooden decking, and surrounding wetland foliage.
- Keyword hints: Cloudy Sky, Foliage, Forest, Grass, Lake, Landscape, Leaves, Marshland, Moss, Outdoors, Pond, Reeds, Trees, Water reflection, Wetland, Wooden shed, architecture, bird hide, birdwatching, boardwalk

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

The crash occurred during model load, before image decoding, so the exact
input image is not required: substitute any local image for the placeholder
path and run one native mlx-vlm process.

```bash
python -m mlx_vlm.generate --model tencent/Youtu-VL-4B-Instruct --image any-local-image.jpg --prompt x --max-tokens 8 --temperature 0.0 --revision 8d30a0e49662a1d628a472b12df264dbcd768753 --trust-remote-code
```

| Evidence | Link |
| --- | --- |
| Full diagnostics | [model evidence](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-tencent-youtu-vl-4b-instruct) |
| Detailed issue draft | [crash draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_tencent_Youtu-VL-4B-Instruct.md) |

## Observation clusters

Repeated mechanical observation signatures among results requiring review.

| Observed result | Models |
| --- | --- |
| Response repeats the same text; Generation was stopped early after sustained repeated output; Title or keywords do not meet requested constraints | 3 |
| Response repeats the same text; Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Title or keywords do not meet requested constraints | 1 |
| Generation was stopped early after sustained repeated output; Title or keywords do not meet requested constraints | 1 |
| Unrecognised model control tokens remain visible | 3 |
| Conversation-role control tokens remain visible | 1 |

## Completed attempts requiring review

| Model | Usability | Observed result | Evidence |
| --- | --- | --- | --- |
| jinaai/jina-vlm-mlx | unusable | Response repeats the same text; Generation was stopped early after sustained repeated output; Title has 4 words (requested 5-10); Keyword list has 61 terms (requested 10-18); Duplicate keywords: boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-jinaai-jina-vlm-mlx) |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 | unusable | Response repeats the same text; Generation was stopped early after sustained repeated output; Duplicate keywords: boardwalk | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16) |
| mlx-community/GLM-4.1V-9B-Thinking-8bit | unusable | Response repeats the same text; Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Keyword list has 80 terms (requested 10-18); Duplicate keywords: foliage, pond, wetland, wooden shed, trees, water reflection, marshland, reeds, leaves, outdoors, landscape, moss, boardwalk, architecture, grass, bird hide, birdwatching | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-41v-9b-thinking-8bit) |
| mlx-community/LFM2.5-VL-1.6B-bf16 | unusable | Response repeats the same text; Generation was stopped early after sustained repeated output; Keyword list has 76 terms (requested 10-18); Duplicate keywords: elevated, wetland, pond, outdoor, railings, reflection, natural, environment, architecture, birdwatching, peaceful, serene | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-lfm25-vl-16b-bf16) |
| mlx-community/X-Reasoner-7B-8bit | unusable | Generation was stopped early after sustained repeated output; Keyword list has 135 terms (requested 10-18); Duplicate keywords: pond, wetland, grass, reeds, trees, architecture, moss, weathered, landscape, outdoor, nature, structure, environment, ecosystem, habitat, accurate, precise, detailed, comprehensive, thorough | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-x-reasoner-7b-8bit) |
| mlx-community/diffusiongemma-26B-A4B-it-8bit | usable with caveats | Unrecognised model control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-8bit) |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | usable with caveats | Unrecognised model control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-mxfp8) |
| mlx-community/GLM-4.6V-nvfp4 | usable with caveats | Unrecognised model control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-46v-nvfp4) |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 | usable with caveats | Conversation-role control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16) |

## Clean completions

12 clean completions (`LiquidAI/LFM2.5-VL-450M-MLX-bf16`, `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`, `mlx-community/Idefics3-8B-Llama3-bf16`, `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`, `mlx-community/Ministral-3-3B-Instruct-2512-4bit`, `mlx-community/Molmo2-8B-4bit`, `mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit`, `mlx-community/Qwen3.5-35B-A3B-4bit`, `mlx-community/gemma-3-27b-it-qat-4bit`, `mlx-community/gemma-4-26b-a4b-it-4bit`, `mlx-community/gemma-4-31b-it-4bit`, `mlx-community/granite-4.0-3b-vision-4bit`); 18 more completed with prompt-compliance observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 9,409 x 6,273 pixels, 51,431,731 bytes
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: temperature:* 0.0
- *Generation: top_p:* 1.0
- *Trust remote code:* true
- *check_models version:* 0.15.0
- *check_models revision:* 701362c5e26c6db3340353fad597d04039fb1d52
- *check_models source dirty:* false
- *mlx-vlm:* 0.7.0rc0
- *mlx-vlm source revision:* 24b244ee29f1646e14d1ba935ba2c6bafd3f78f6
- *mlx:* 0.32.3.dev20260828+99e45f71d
- *mlx source revision:* 99e45f71dcb4318e2c2530e66038045795883ad2
- *transformers:* 5.16.1
- *macOS Version:* 26.6.2
- *GPU/Chip:* Apple M5 Max
- *Python Version:* 3.14.7

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
