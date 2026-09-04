# mlx-vlm compatibility findings across 32 cached vision-language models

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

- *Run started:* 2026-09-04 19:22:31 BST
- *Run finished:* 2026-09-04 19:33:10 BST
- *Run duration:* 10m 36s
- *Evaluation mode:* assisted
- *Models attempted:* 32
- *Completed:* 32
- *Crashed:* 0
- *Indeterminate:* 0
- *Crashes requiring action:* 0
- *Other results requiring review:* 3
- *Hit the token cap:* 0
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

- *Baseline:* cf87381f:src/output/results.jsonl
- *Baseline run timestamp:* 2026-09-04 15:03:14 BST
- *Baseline check_models:* 0.16.8 @ 37840f076
- *Baseline mlx:* 0.32.3.dev20260904+b6368984b @ b6368984b
- *Baseline mlx-vlm:* 0.7.0rc0 @ 5c9b5f52a
- *Baseline transformers:* 5.16.1
- *Baseline python:* 3.14.7
- *Models compared:* 29
- *Identical generated text:* 28 of 29 completed in both
- *Generation tok/s ratio (now/baseline):* withheld (execution mode or inputs
  not established as like-for-like)
- *Throughput noise band:* fixed ±15% fallback (insufficient history)
- *Execution mode:* isolated now vs in_process in the baseline — per-process
  start-up differs, so throughput is not directly comparable

- New this run (no baseline):
  `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit`,
  `mlx-community/GLM-4.6V-Flash-4bit`,
  `mlx-community/Kimi-VL-A3B-Thinking-2506-8bit`
- In baseline, not run this time: 13 models (targeted run against a full-sweep
  baseline)

| Model | Execution | Usability | Observation delta |
| --- | --- | --- | --- |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | completed | usable → usable with caveats | +title/keyword constraints failed |

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
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | usable | 6.66s | 134 tok/s | 19 | none |
| mlx-community/gemma-4-26b-a4b-it-4bit | usable | 4.67s | 128 tok/s | 16 | none |
| mlx-community/gemma-4-31b-it-4bit | usable | 8.05s | 26.5 tok/s | 19 | none |
| mlx-community/granite-4.0-3b-vision-4bit | usable | 2.81s | 178 tok/s | 4.7 | none |
| mlx-community/Idefics3-8B-Llama3-bf16 | usable | 9.40s | 32.3 tok/s | 18 | none |
| mlx-community/InternVL3-8B-bf16 | usable | 6.17s | 34.4 tok/s | 17 | none |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit | usable | 2.91s | 213 tok/s | 4.0 | none |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | usable | 7.13s | 67.4 tok/s | 13 | none |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | usable | 3.79s | 190 tok/s | 7.8 | none |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | usable | 50.46s | 25.2 tok/s | 25 | none |
| mlx-community/Qwen3-VL-2B-Thinking-bf16 | usable | 29.95s | 89.8 tok/s | 8.4 | none |
| mlx-community/Qwen3.5-35B-A3B-4bit | usable | 59.95s | 116 tok/s | 24 | none |
| mlx-community/Qwen3.8-27B-4bit | usable | 79.21s | 30.1 tok/s | 21 | none |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | usable with caveats | 1.69s | 484 tok/s | 1.9 | title/keyword constraints failed |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | usable with caveats | 9.90s | 30.8 tok/s | 23 | title/keyword constraints failed |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | usable with caveats | 6.24s | 66.4 tok/s | 28 | title/keyword constraints failed |
| mlx-community/gemma-3-27b-it-qat-4bit | usable with caveats | 8.55s | 29.9 tok/s | 17 | title/keyword constraints failed |
| mlx-community/GLM-4.6V-Flash-4bit | usable with caveats | 8.57s | 78.6 tok/s | 8.7 | control tokens visible |
| mlx-community/GLM-4.6V-nvfp4 | usable with caveats | 20.53s | 44.2 tok/s | 78 | control tokens visible |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit | usable with caveats | 18.49s | 67.2 tok/s | 20 | role tokens visible |
| mlx-community/LFM2.5-VL-1.6B-bf16 | usable with caveats | 2.72s | 189 tok/s | 4.0 | title/keyword constraints failed |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4 | usable with caveats | 7.30s | 64.8 tok/s | 13 | title/keyword constraints failed |
| mlx-community/Molmo2-8B-4bit | usable with caveats | 4.99s | 73.6 tok/s | 8.5 | title/keyword constraints failed |
| mlx-community/North-Micro-Vision-Instruct-4bit | usable with caveats | 6.31s | 229 tok/s | 3.9 | title/keyword constraints failed |
| mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit | usable with caveats | 5.61s | 112 tok/s | 24 | title/keyword constraints failed |
| mlx-community/Phi-3.5-vision-instruct-bf16 | usable with caveats | 4.45s | 58.3 tok/s | 9.5 | title/keyword constraints failed |
| mlx-community/pixtral-12b-8bit | usable with caveats | 7.34s | 39.4 tok/s | 16 | title/keyword constraints failed |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | usable with caveats | 58.82s | 88.4 tok/s | 22 | title/keyword constraints failed |
| mlx-community/Qwen3.5-9B-MLX-4bit | usable with caveats | 60.70s | 91.5 tok/s | 10.0 | title/keyword constraints failed |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | usable with caveats | 3.00s | 127 tok/s | 5.4 | title/keyword constraints failed |
| mlx-community/Step-3.7-Flash-oQ2e | usable with caveats | 22.92s | 46.7 tok/s | 70 | title/keyword constraints failed |
| mlx-community/X-Reasoner-7B-8bit | usable with caveats | 22.04s | 59.1 tok/s | 13 | title/keyword constraints failed |

## Constraint-failure breakdown

How the fleet failed the catalogue constraints — a skew toward one constraint
suggests prompt difficulty rather than individual model faults.

- Title length: 5 model(s) outside 5-10 words (4 below, 1 above; median
  observed 4)
- Keyword count: 12 model(s) outside 10-18 (0 below, 12 above; median observed
  19.5)
- Duplicate keywords: 2 model(s)

## Observation clusters

Repeated mechanical observation signatures among results requiring review.

| Observed result | Models |
| --- | --- |
| Unrecognised model control tokens remain visible | 2 |
| Conversation-role control tokens remain visible | 1 |

## Completed attempts requiring review

| Model | Usability | Observed result | Evidence |
| --- | --- | --- | --- |
| mlx-community/GLM-4.6V-Flash-4bit | usable with caveats | Unrecognised model control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-46v-flash-4bit) |
| mlx-community/GLM-4.6V-nvfp4 | usable with caveats | Unrecognised model control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-46v-nvfp4) |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit | usable with caveats | Conversation-role control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-8bit) |

## Clean completions

13 clean completions (`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit`, `mlx-community/Idefics3-8B-Llama3-bf16`, `mlx-community/InternVL3-8B-bf16`, `mlx-community/LFM2.5-VL-3B-OptiQ-4bit`, `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`, `mlx-community/Ministral-3-3B-Instruct-2512-4bit`, `mlx-community/Muse-Glimmer-30B-OptiQ-4bit`, `mlx-community/Qwen3-VL-2B-Thinking-bf16`, `mlx-community/Qwen3.5-35B-A3B-4bit`, `mlx-community/Qwen3.8-27B-4bit`, `mlx-community/gemma-4-26b-a4b-it-4bit`, `mlx-community/gemma-4-31b-it-4bit`, `mlx-community/granite-4.0-3b-vision-4bit`); 16 more completed with prompt-compliance observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 6,656 x 9,984 pixels, 66,295,254 bytes
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: temperature:* 0.0
- *Generation: top_p:* 1.0
- *Trust remote code:* true
- *check_models version:* 0.16.8
- *check_models revision:* cf87381f7a261c63e072c0b432bbc6000268bf3e
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
