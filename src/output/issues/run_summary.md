# mlx-vlm compatibility findings across 33 cached vision-language models

**What this run measures.** This run records model responses to one shared
image and prompt (evaluation lane: assisted). Mechanical checks are not
factual-accuracy judgments; inspect the image, prompt and final answers before
choosing a model. Results do not establish fitness for other tasks.
check_models gave every locally cached MLX vision-language model the same
image and the same prompt (reproduced below), through mlx-vlm's generation
pipeline, and recorded mechanical facts about each attempt: whether it ran,
what the selected assessment profile checked (stated under *Assessment*
below), and its speed and memory. There is no semantic quality scoring; every
observation is a reproducible mechanical fact from this one image and prompt.

## Run summary

- *Run started:* 2026-09-11 21:48:11 BST
- *Run finished:* 2026-09-11 21:56:42 BST
- *Run duration:* 8m 30s
- *Evaluation lane:* assisted
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Input image:* JPEG, 8,693 x 5,796 pixels (50.4 MP), 43.9 MB
- *Models attempted:* 33
- *Completed:* 33
- *Crashed:* 0
- *Indeterminate:* 0
- *Crashes requiring action:* 0
- *Other results requiring review:* 2
- *Reached token limit:* 2
- *Incomplete output at token limit:* 1
- *Stopped early for repetition:* 0

Observations are mechanical facts from one image, not general model-quality
judgements.

<details>
<summary>Exact prompt sent to every model</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-09-01 16:08:37 UTC+01:00
- GPS: 51.385512°N, 2.363704°W

Descriptive hints:
- Description hint: A street-level view looking downhill along Gay Street from The Circus, showcasing the classic Georgian architecture of Bath stone terraced townhouses against a dramatic cloudy sky in Bath, Somerset, England.
- Keyword hints: Architecture, Bath, Bath England, Bath Stone, Cars, Chimneys, Cityscape, Cloudy Sky, England, Gay Street, Georgian architecture, Hills, Lamp post, Parked Cars, Railings, Sash Windows, Somerset, Street, Street Scene, Street signs

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

**Not directly comparable** — the per-model diff is withheld because the runs
differ in: prompt differs; evaluation lane differs (blind → assisted); image
differs (sha256 dea9e7ef9738… → 398a0b2c7ac9…). Treat any difference against
this baseline as a change of inputs, not a change of model or runtime
behaviour.

- *Baseline:* 0932adcc:src/output/results.jsonl
- *Baseline run timestamp:* 2026-09-06 22:21:23 BST
- *Baseline check_models:* 0.17.15 @ 1ceac9f36
- *Baseline mlx:* 0.32.3.dev20260906+ce916dbbc @ ce916dbbc
- *Baseline mlx-vlm:* 0.7.0rc0 @ d5064772d
- *Baseline transformers:* 5.16.1
- *Baseline python:* 3.14.7
- *Baseline hardware:* Apple M5 Max, 40 GPU cores, 128.0 GB RAM

## Model quality at a glance

Every attempted model ranked by mechanical observations, with captured
resource facts. No concerns detected is not a task-compliance or accuracy
verdict. Consult the assessment scope above and inspect the final answers.
Crashes and integration signals have expanded maintainer evidence.

| Model | Mechanical checks | Total | Gen tok/s | Peak GB | Observed |
| --- | --- | --- | --- | --- | --- |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | no concerns detected | 9.51s | 29.8 tok/s | 23 | none |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | no concerns detected | 6.89s | 50.7 tok/s | 28 | none |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | no concerns detected | 12.11s | 108 tok/s | 19 | none |
| mlx-community/gemma-3-27b-it-qat-4bit | no concerns detected | 9.11s | 29.3 tok/s | 17 | none |
| mlx-community/gemma-4-26b-a4b-it-4bit | no concerns detected | 4.26s | 130 tok/s | 16 | none |
| mlx-community/gemma-4-31b-it-4bit | no concerns detected | 8.42s | 26.1 tok/s | 20 | none |
| mlx-community/GLM-4.6V-Flash-4bit | no concerns detected | 9.30s | 77.2 tok/s | 8.7 | none |
| mlx-community/granite-4.0-3b-vision-4bit | no concerns detected | 2.77s | 178 tok/s | 4.6 | none |
| mlx-community/Idefics3-8B-Llama3-bf16 | no concerns detected | 10.38s | 31.7 tok/s | 18 | none |
| mlx-community/InternVL3-8B-bf16 | no concerns detected | 6.40s | 33.9 tok/s | 17 | none |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit | no concerns detected | 21.71s | 62.4 tok/s | 20 | none |
| mlx-community/LFM2.5-VL-1.6B-bf16 | no concerns detected | 2.45s | 188 tok/s | 4.1 | none |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit | no concerns detected | 2.36s | 212 tok/s | 4.0 | none |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | no concerns detected | 6.35s | 67.0 tok/s | 13 | none |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4 | no concerns detected | 7.20s | 59.8 tok/s | 13 | none |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | no concerns detected | 3.21s | 180 tok/s | 7.8 | none |
| mlx-community/North-Micro-Vision-Instruct-4bit | no concerns detected | 4.36s | 224 tok/s | 3.9 | none |
| mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit | no concerns detected | 5.59s | 104 tok/s | 24 | none |
| mlx-community/pixtral-12b-8bit | no concerns detected | 7.58s | 39.3 tok/s | 16 | none |
| mlx-community/Qwen3-VL-2B-Thinking-bf16 | no concerns detected | 27.36s | 88.1 tok/s | 8.4 | none |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | no concerns detected | 39.85s | 84.4 tok/s | 23 | none |
| mlx-community/Qwen3.5-35B-A3B-4bit | no concerns detected | 37.02s | 109 tok/s | 25 | none |
| mlx-community/Qwen3.5-9B-MLX-4bit | no concerns detected | 40.85s | 91.2 tok/s | 11 | none |
| mlx-community/Qwen3.8-27B-4bit | no concerns detected | 59.40s | 29.7 tok/s | 21 | none |
| mlx-community/Step-3.7-Flash-oQ3e | no concerns detected | 37.78s | 50.2 tok/s | 92 | none |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | concerns detected | 1.62s | 484 tok/s | 1.9 | duplicate keywords |
| mlx-community/GLM-4.6V-nvfp4 | concerns detected | 26.44s | 42.7 tok/s | 78 | control tokens visible |
| mlx-community/Phi-3.5-vision-instruct-bf16 | concerns detected | 4.48s | 55.9 tok/s | 9.3 | duplicate keywords |
| mlx-community/X-Reasoner-7B-8bit | concerns detected | 20.10s | 58.1 tok/s | 14 | duplicate keywords |
| mlx-community/Molmo2-8B-4bit | major concerns | 6.49s | 71.0 tok/s | 8.1 | labelled fields not detected |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | major concerns | 58.28s | 22.3 tok/s | 25 | labelled fields not detected; cut off at token limit; role tokens visible |
| mlx-community/nanoLLaVA-1.5-4bit | major concerns | 1.45s | 338 tok/s | 1.8 | labelled fields not detected |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | major concerns | 2.32s | 127 tok/s | 5.6 | labelled fields not detected |

## Constraint-failure breakdown

How the fleet failed the catalogue constraints — a skew toward one constraint
suggests prompt difficulty rather than individual model faults.

- Duplicate keywords: 3 model(s)

## Observation clusters

Repeated mechanical observation signatures among results requiring review.

| Observed result | Models |
| --- | --- |
| Unrecognised model control tokens remain visible | 1 |
| Required labelled fields not detected; Response appears cut off at the token limit; Conversation-role control tokens remain visible | 1 |

## Completed attempts requiring review

| Model | Mechanical checks | Observed result | Evidence |
| --- | --- | --- | --- |
| mlx-community/GLM-4.6V-nvfp4 | concerns detected | Unrecognised model control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-46v-nvfp4) |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | major concerns | Required labelled fields not detected: title, description; Response appears cut off at the token limit; Conversation-role control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-muse-glimmer-30b-optiq-4bit) |

## Completions without detected concerns

25 completions without detected concerns (`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`, `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit`, `mlx-community/GLM-4.6V-Flash-4bit`, `mlx-community/Idefics3-8B-Llama3-bf16`, `mlx-community/InternVL3-8B-bf16`, `mlx-community/Kimi-VL-A3B-Thinking-2506-8bit`, `mlx-community/LFM2.5-VL-1.6B-bf16`, `mlx-community/LFM2.5-VL-3B-OptiQ-4bit`, `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`, `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`, `mlx-community/Ministral-3-3B-Instruct-2512-4bit`, `mlx-community/North-Micro-Vision-Instruct-4bit`, `mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit`, `mlx-community/Qwen3-VL-2B-Thinking-bf16`, `mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit`, `mlx-community/Qwen3.5-35B-A3B-4bit`, `mlx-community/Qwen3.5-9B-MLX-4bit`, `mlx-community/Qwen3.8-27B-4bit`, `mlx-community/Step-3.7-Flash-oQ3e`, `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`, `mlx-community/gemma-3-27b-it-qat-4bit`, `mlx-community/gemma-4-26b-a4b-it-4bit`, `mlx-community/gemma-4-31b-it-4bit`, `mlx-community/granite-4.0-3b-vision-4bit`, `mlx-community/pixtral-12b-8bit`); 6 more completed with prompt-compliance observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 8,693 x 5,796 pixels, 43,870,091 bytes
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: temperature:* 0.0
- *Generation: top_p:* 1.0
- *Trust remote code:* true
- *check_models version:* 0.17.16
- *check_models revision:* 0932adcce170ad2ac2cd42b49f1878c028810b08
- *check_models source dirty:* true
- *mlx-vlm:* 0.7.0
- *mlx-vlm source revision:* d2a1434a03e4c9975b0d505e7178e0cfc4082a83
- *mlx:* 0.32.3.dev20260911+dfe17bafb
- *mlx source revision:* dfe17bafb23e66fe56596df532a497ab3611d0e5
- *transformers:* 5.17.0
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
