# mlx-vlm compatibility findings across 42 cached vision-language models

**What this run measures.** These models serve many purposes; this run probes
exactly one narrow task: producing catalogue metadata for a single photograph
from the assisted-lane prompt and whatever context it supplies — here,
camera-recorded capture context plus draft descriptive hints previously
produced by a more capable model. Results say nothing about a model's fitness
for other uses. check_models gave every locally cached MLX vision-language
model the same image and the same prompt (reproduced below), through mlx-vlm's
generation pipeline, and recorded mechanical facts about each attempt: whether
it ran, whether the output supplied the requested Title/Description/Keywords
structure within the ranges the prompt states, and its speed and memory. There
is no semantic quality scoring; every observation is a reproducible mechanical
fact from this one image and prompt.

## Run summary

- *Run started:* 2026-09-04 14:14:55 BST
- *Run finished:* 2026-09-04 15:03:14 BST
- *Run duration:* 48m 18s
- *Evaluation mode:* assisted
- *Models attempted:* 42
- *Completed:* 41
- *Crashed:* 1
- *Indeterminate:* 0
- *Crashes requiring action:* 1
- *Other results requiring review:* 5
- *Hit the token cap:* 4
- *Stopped early for repetition:* 0

Observations are mechanical facts from one image, not general model-quality
judgements.

<details>
<summary>Exact prompt sent to every model</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-08-31 15:02:20 UTC+01:00

Descriptive hints:
- Description hint: Visitors walk along the pathway and relax in front of the historic Winchester City Mill, an ancient watermill situated over the River Itchen in Winchester, Hampshire, England.
- Keyword hints: Adobe Stock, Any Vision, Arch, Blue sky, Chimney, Clay tiles, Elderly woman, England, Girls, Hampshire, Mill, National Trust, Pedestrians, People, Rapids, River Itchen, Riverbank, Scenery, Sitting, Stone wall

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

- *Baseline:* 37840f07:src/output/results.jsonl
- *Baseline run timestamp:* 2026-09-04 13:16:45 BST
- *Baseline check_models:* 0.16.6 @ 8b5e2e71c
- *Baseline mlx:* 0.32.3.dev20260904+b6368984b @ b6368984b
- *Baseline mlx-vlm:* 0.7.0rc0 @ 5c9b5f52a
- *Baseline transformers:* 5.16.1
- *Baseline python:* 3.14.7
- *Models compared:* 42
- *Identical generated text:* 39 of 41 completed in both
- *Generation tok/s ratio (now/baseline):* 1.000 (range 0.96-1.42, 41 models)
- *Throughput noise band:* fixed ±15% fallback (insufficient history)

No execution, usability, or observation-set changes against the baseline.

| Model | Baseline tok/s | Now tok/s | Ratio | Expected band |
| --- | --- | --- | --- | --- |
| mlx-community/diffusiongemma-26B-A4B-it-8bit | 57.7 | 81.9 | 1.42 | 49.1-66.4 (fallback) |

Mechanical diff only: one image, temperature as configured; single-observation
flips on one model are usually run-to-run variance, broad shifts are not.

## Model quality at a glance

Every attempted model ranked by current-run usability, with captured resource
facts. Usability reflects this single image and prompt only: *usable* means
the output followed the prompt's requested structure; *usable with caveats*
means repairable deviations (constraint misses, visible control tokens);
*unusable* means mechanically broken output (repetition, missing sections,
truncation); *not evaluated* means the attempt crashed. Total is end-to-end
wall time including model load, Gen tok/s is decode-only throughput, and Peak
GB is peak MLX memory. The model gallery holds full outputs and the
diagnostics report holds maintainer evidence.

| Model | Usability | Total | Gen tok/s | Peak GB | Observed |
| --- | --- | --- | --- | --- | --- |
| mlx-community/diffusiongemma-26B-A4B-it-8bit | usable | 5.60s | 81.9 tok/s | 29 | none |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | usable | 5.95s | 56.2 tok/s | 28 | none |
| mlx-community/gemma-4-26b-a4b-it-4bit | usable | 4.34s | 129 tok/s | 16 | none |
| mlx-community/gemma-4-31b-it-4bit | usable | 7.98s | 25.1 tok/s | 20 | none |
| mlx-community/granite-4.0-3b-vision-4bit | usable | 2.47s | 176 tok/s | 4.7 | none |
| mlx-community/Idefics3-8B-Llama3-bf16 | usable | 8.87s | 32.3 tok/s | 18 | none |
| mlx-community/InternVL3-8B-bf16 | usable | 5.74s | 34.3 tok/s | 17 | none |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit | usable | 2.39s | 210 tok/s | 4.0 | none |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | usable | 6.99s | 67.1 tok/s | 13 | none |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | usable | 3.33s | 190 tok/s | 7.8 | none |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | usable | 50.63s | 25.4 tok/s | 25 | none |
| mlx-community/Qwen3-VL-2B-Thinking-bf16 | usable | 28.07s | 91.5 tok/s | 8.4 | none |
| mlx-community/Qwen3.5-35B-A3B-4bit | usable | 57.81s | 110 tok/s | 24 | none |
| mlx-community/Qwen3.6-27B-mxfp8 | usable | 86.51s | 17.8 tok/s | 33 | none |
| mlx-community/Qwen3.8-27B-4bit | usable | 74.62s | 30.5 tok/s | 21 | none |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | usable with caveats | 1.62s | 483 tok/s | 1.9 | title/keyword constraints failed |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | usable with caveats | 9.91s | 30.2 tok/s | 23 | title/keyword constraints failed |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 | usable with caveats | 19.05s | 63.1 tok/s | 60 | title/keyword constraints failed |
| mlx-community/gemma-3-27b-it-qat-4bit | usable with caveats | 8.05s | 31.2 tok/s | 17 | title/keyword constraints failed |
| mlx-community/GLM-4.6V-nvfp4 | usable with caveats | 25.67s | 39.9 tok/s | 78 | control tokens visible |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 | usable with caveats | 157.63s | 4.67 tok/s | 40 | role tokens visible |
| mlx-community/LFM2.5-VL-1.6B-bf16 | usable with caveats | 2.23s | 183 tok/s | 4.0 | title/keyword constraints failed |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4 | usable with caveats | 6.86s | 64.7 tok/s | 13 | title/keyword constraints failed |
| mlx-community/Molmo2-8B-4bit | usable with caveats | 4.40s | 72.7 tok/s | 8.1 | title/keyword constraints failed |
| mlx-community/North-Micro-Vision-Instruct-4bit | usable with caveats | 5.62s | 229 tok/s | 3.9 | title/keyword constraints failed |
| mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit | usable with caveats | 5.41s | 107 tok/s | 24 | title/keyword constraints failed |
| mlx-community/Phi-3.5-vision-instruct-bf16 | usable with caveats | 4.08s | 56.8 tok/s | 9.5 | title/keyword constraints failed |
| mlx-community/pixtral-12b-8bit | usable with caveats | 6.80s | 38.9 tok/s | 16 | title/keyword constraints failed |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | usable with caveats | 58.72s | 87.1 tok/s | 23 | title/keyword constraints failed |
| mlx-community/Qwen3.5-9B-MLX-4bit | usable with caveats | 56.12s | 92.1 tok/s | 10.0 | title/keyword constraints failed |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | usable with caveats | 2.73s | 125 tok/s | 5.4 | title/keyword constraints failed |
| mlx-community/Step-3.7-Flash-oQ2e | usable with caveats | 25.61s | 46.6 tok/s | 70 | title/keyword constraints failed |
| mlx-community/X-Reasoner-7B-8bit | usable with caveats | 20.47s | 57.8 tok/s | 13 | title/keyword constraints failed |
| jinaai/jina-vlm-mlx | unusable | 4.61s | 139 tok/s | 3.7 | missing required fields |
| mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX | unusable | 27.99s | 42.0 tok/s | 15 | missing required fields; echoes instructions; extra text before Title; cut off at token limit |
| mlx-community/FastVLM-0.5B-bf16 | unusable | 2.43s | 359 tok/s | 2.2 | missing required fields |
| mlx-community/gemma-3n-E4B-it-bf16 | unusable | 6.81s | 48.8 tok/s | 17 | missing required fields; extra text before Title |
| mlx-community/GLM-4.1V-9B-Thinking-8bit | unusable | 29.38s | 46.4 tok/s | 13 | echoes instructions; extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed |
| mlx-community/GLM-4.6V-Flash-mxfp4 | unusable | 20.79s | 76.4 tok/s | 8.4 | repeated text; cut off at token limit; title/keyword constraints failed |
| mlx-community/MiniCPM-V-4.6-8bit | unusable | 2.21s | 288 tok/s | 3.3 | missing required fields; extra text before Title |
| Qwen/Qwen3-VL-2B-Instruct | unusable | 25.88s | 92.2 tok/s | 8.4 | repeated text; cut off at token limit; title/keyword constraints failed |
| tencent/Youtu-VL-4B-Instruct | not evaluated | 1.06s | - | - | crashed during model_load |

## Constraint-failure breakdown

How the fleet failed the catalogue constraints — a skew toward one constraint
suggests prompt difficulty rather than individual model faults.

- Title length: 7 model(s) outside 5-10 words (6 below, 1 above; median
  observed 4)
- Keyword count: 16 model(s) outside 10-18 (0 below, 16 above; median observed
  20)
- Duplicate keywords: 5 model(s)

## Crashes requiring action

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
- *Image dimensions:* 6,656 x 9,984 pixels
- *Image size:* 66,295,254 bytes
- *Image SHA-256:* 168b4850b1427394bbe84a99ffd05533ffaf7e995e4213d9ce2d36c959e70c7b

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-08-31 15:02:20 UTC+01:00

Descriptive hints:
- Description hint: Visitors walk along the pathway and relax in front of the historic Winchester City Mill, an ancient watermill situated over the River Itchen in Winchester, Hampshire, England.
- Keyword hints: Adobe Stock, Any Vision, Arch, Blue sky, Chimney, Clay tiles, Elderly woman, England, Girls, Hampshire, Mill, National Trust, Pedestrians, People, Rapids, River Itchen, Riverbank, Scenery, Sitting, Stone wall

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
| Response repeats the same text; Response appears cut off at the token limit; Title or keywords do not meet requested constraints | 2 |
| Unrecognised model control tokens remain visible | 1 |
| Response repeats the task instructions instead of only returning the requested fields; Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Title or keywords do not meet requested constraints | 1 |
| Conversation-role control tokens remain visible | 1 |

## Completed attempts requiring review

| Model | Usability | Observed result | Evidence |
| --- | --- | --- | --- |
| mlx-community/GLM-4.6V-Flash-mxfp4 | unusable | Response repeats the same text; Response appears cut off at the token limit; Title has 3 words (requested 5-10); Keyword list has 230 terms (requested 10-18); Duplicate keywords: hampshire, england, rapids, arched doorway, tourist attraction, historic watermill, diamond paned windows, weathered roof tiles, red brick chimney, stone archway over river, mossy stone wall, green ivy, traditional english mill architecture, historic watermill on river itchen, winchester, national trust property, scenic riverfront, historic mill with water flow, traditional english architecture, stone and brick building | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-46v-flash-mxfp4) |
| Qwen/Qwen3-VL-2B-Instruct | unusable | Response repeats the same text; Response appears cut off at the token limit; Title has 4 words (requested 5-10); Keyword list has 327 terms (requested 10-18); Duplicate keywords: national trust, pedestrians, riverbank, stone wall, brick building, blue sky, chimney, clay tiles, elderly woman, girls, rapids, scenery, sitting, people, river | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-qwen-qwen3-vl-2b-instruct) |
| mlx-community/GLM-4.6V-nvfp4 | usable with caveats | Unrecognised model control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-46v-nvfp4) |
| mlx-community/GLM-4.1V-9B-Thinking-8bit | unusable | Response repeats the task instructions instead of only returning the requested fields; Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Keyword list has 42 terms (requested 10-18); Duplicate keywords: blue sky, chimney, clay tiles, elderly woman, girls, hampshire, mill, national trust, pedestrians, people, river itchen, riverbank, scenery, sitting | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-41v-9b-thinking-8bit) |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 | usable with caveats | Conversation-role control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16) |

## Clean completions

15 clean completions (`mlx-community/Idefics3-8B-Llama3-bf16`, `mlx-community/InternVL3-8B-bf16`, `mlx-community/LFM2.5-VL-3B-OptiQ-4bit`, `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`, `mlx-community/Ministral-3-3B-Instruct-2512-4bit`, `mlx-community/Muse-Glimmer-30B-OptiQ-4bit`, `mlx-community/Qwen3-VL-2B-Thinking-bf16`, `mlx-community/Qwen3.5-35B-A3B-4bit`, `mlx-community/Qwen3.6-27B-mxfp8`, `mlx-community/Qwen3.8-27B-4bit`, `mlx-community/diffusiongemma-26B-A4B-it-8bit`, `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`, `mlx-community/gemma-4-26b-a4b-it-4bit`, `mlx-community/gemma-4-31b-it-4bit`, `mlx-community/granite-4.0-3b-vision-4bit`); 21 more completed with prompt-compliance observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 6,656 x 9,984 pixels, 66,295,254 bytes
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: temperature:* 0.0
- *Generation: top_p:* 1.0
- *Trust remote code:* true
- *check_models version:* 0.16.8
- *check_models revision:* 37840f076c0c3e515bb01d1a0d9097ae73f6fb59
- *check_models source dirty:* false
- *mlx-vlm:* 0.7.0rc0
- *mlx-vlm source revision:* 5c9b5f52adfeab35b5ece0bb2d6e4d44541d9e32
- *mlx:* 0.32.3.dev20260904+b6368984b
- *mlx source revision:* b6368984b8e02a3fb3ee7986846c0fb85e1fccf7
- *transformers:* 5.16.1
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
| Environment | [environment.log](https://github.com/jrp2014/check_models/blob/main/src/output/environment.log) |
| Log | [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log) |
