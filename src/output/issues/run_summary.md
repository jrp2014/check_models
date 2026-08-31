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

- *Run timestamp:* 2026-08-30 22:59:15 BST
- *Evaluation mode:* assisted
- *Models attempted:* 42
- *Completed:* 40
- *Crashed:* 2
- *Indeterminate:* 0
- *Crashes requiring action:* 2
- *Other results requiring review:* 7
- *Hit the token cap:* 3
- *Stopped early for repetition:* 2

Observations are mechanical facts from one image, not general model-quality
judgements.

<details>
<summary>Exact prompt sent to every model</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-08-18 17:46:05 UTC+01:00
- GPS: 55.951722°N, 3.201417°W

Descriptive hints:
- Title hint: City Centre, Edinburgh, Scotland, UK, GBR, Europe
- Description hint: Extensive scaffolding covers the facade of a building undergoing major renovation and redevelopment along Princes Street in Edinburgh, Scotland, while pedestrians walk past temporary construction fences and a Boots pharmacy beneath an overcast sky.
- Keyword hints: 10 Best (structured), Adobe Stock, Any Vision, City Centre, Civil engineering, Construction fence, Construction site, Crane, Edinburgh, Europe, Fence, Modern Architecture, Objects, Overcast, Overcast Sky, Pedestrians, Princes Street, Roadworks, Scaffolding, Scotland

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
differ in: prompt differs; image differs (sha256 b31874639694… →
a843ca79cc4b…). Treat any difference against this baseline as a change of
inputs, not a change of model or runtime behaviour.

- *Baseline:* 082cb805:src/output/results.jsonl
- *Baseline run timestamp:* 2026-08-30 01:47:23 BST
- *Baseline check_models:* 0.16.5 @ 7299db1db
- *Baseline mlx:* 0.32.3.dev20260829+052e77db9 @ 052e77db9
- *Baseline mlx-vlm:* 0.7.0rc0 @ f5d9533a9
- *Baseline transformers:* 5.16.1
- *Baseline python:* 3.14.7

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
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | usable | 6.16s | 46.6 tok/s | 28 | none |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 | usable | 18.61s | 62.8 tok/s | 60 | none |
| mlx-community/gemma-3-27b-it-qat-4bit | usable | 7.89s | 31.5 tok/s | 17 | none |
| mlx-community/gemma-4-26b-a4b-it-4bit | usable | 4.04s | 128 tok/s | 16 | none |
| mlx-community/gemma-4-31b-it-4bit | usable | 7.65s | 26.3 tok/s | 20 | none |
| mlx-community/InternVL3-8B-bf16 | usable | 5.94s | 34.3 tok/s | 17 | none |
| mlx-community/LFM2.5-VL-1.6B-bf16 | usable | 2.31s | 184 tok/s | 4.0 | none |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit | usable | 2.39s | 211 tok/s | 4.0 | none |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | usable | 3.19s | 189 tok/s | 7.8 | none |
| mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit | usable | 5.27s | 106 tok/s | 25 | none |
| mlx-community/Phi-3.5-vision-instruct-bf16 | usable | 4.43s | 56.5 tok/s | 9.4 | none |
| mlx-community/pixtral-12b-8bit | usable | 6.16s | 39.6 tok/s | 16 | none |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | usable | 51.82s | 86.5 tok/s | 23 | none |
| mlx-community/Qwen3.5-35B-A3B-4bit | usable | 54.11s | 111 tok/s | 24 | none |
| mlx-community/Qwen3.5-9B-MLX-4bit | usable | 54.88s | 91.7 tok/s | 10.0 | none |
| mlx-community/Qwen3.6-27B-mxfp8 | usable | 79.30s | 17.6 tok/s | 33 | none |
| mlx-community/Qwen3.8-27B-4bit | usable | 75.26s | 30.7 tok/s | 21 | none |
| mlx-community/Step-3.7-Flash-oQ2e | usable | 22.50s | 46.5 tok/s | 70 | none |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | usable with caveats | 1.73s | 480 tok/s | 1.9 | title/keyword constraints failed |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | usable with caveats | 9.29s | 30.4 tok/s | 23 | title/keyword constraints failed |
| mlx-community/diffusiongemma-26B-A4B-it-8bit | usable with caveats | 5.78s | 61.3 tok/s | 29 | title/keyword constraints failed |
| mlx-community/GLM-4.6V-Flash-mxfp4 | usable with caveats | 8.97s | 80.1 tok/s | 8.4 | title/keyword constraints failed |
| mlx-community/GLM-4.6V-nvfp4 | usable with caveats | 20.04s | 43.8 tok/s | 78 | control tokens visible; title/keyword constraints failed |
| mlx-community/granite-4.0-3b-vision-4bit | usable with caveats | 2.21s | 171 tok/s | 4.7 | title/keyword constraints failed |
| mlx-community/Idefics3-8B-Llama3-bf16 | usable with caveats | 8.59s | 32.2 tok/s | 18 | title/keyword constraints failed |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 | usable with caveats | 166.01s | 4.67 tok/s | 40 | role tokens visible |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | usable with caveats | 6.22s | 67.2 tok/s | 13 | title/keyword constraints failed |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4 | usable with caveats | 6.82s | 64.7 tok/s | 13 | title/keyword constraints failed |
| mlx-community/Molmo2-8B-4bit | usable with caveats | 4.39s | 72.7 tok/s | 9.1 | title/keyword constraints failed |
| mlx-community/North-Micro-Vision-Instruct-4bit | usable with caveats | 5.20s | 233 tok/s | 3.9 | title/keyword constraints failed |
| mlx-community/Qwen3-VL-2B-Thinking-bf16 | usable with caveats | 24.81s | 92.6 tok/s | 8.4 | title/keyword constraints failed |
| Qwen/Qwen3-VL-2B-Instruct | usable with caveats | 16.08s | 92.2 tok/s | 8.4 | title/keyword constraints failed |
| jinaai/jina-vlm-mlx | unusable | 4.64s | 138 tok/s | 3.7 | repeated text; stopped early: repeating; title/keyword constraints failed |
| mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX | unusable | 28.08s | 42.1 tok/s | 15 | missing required fields; echoes instructions; cut off at token limit |
| mlx-community/FastVLM-0.5B-bf16 | unusable | 2.42s | 352 tok/s | 2.2 | missing required fields; extra text before Title |
| mlx-community/gemma-3n-E4B-it-bf16 | unusable | 3.26s | insufficient sample | 17 | empty response; missing required fields |
| mlx-community/GLM-4.1V-9B-Thinking-8bit | unusable | 29.30s | 46.6 tok/s | 13 | repeated text; extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed |
| mlx-community/MiniCPM-V-4.6-8bit | unusable | 1.98s | 273 tok/s | 3.8 | missing required fields; extra text before Title |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | unusable | 3.44s | 124 tok/s | 5.5 | repeated text; stopped early: repeating; missing required fields |
| mlx-community/X-Reasoner-7B-8bit | unusable | 36.59s | 55.6 tok/s | 13 | repeated text; cut off at token limit; title/keyword constraints failed |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | not evaluated | 0.50s | - | - | crashed during model_load |
| tencent/Youtu-VL-4B-Instruct | not evaluated | 1.17s | - | - | crashed during model_load |

## Constraint-failure breakdown

How the fleet failed the catalogue constraints — a skew toward one constraint
suggests prompt difficulty rather than individual model faults.

- Title length: 7 model(s) outside 5-10 words (7 below, 0 above; median
  observed 4)
- Keyword count: 8 model(s) outside 10-18 (0 below, 8 above; median observed
  20)
- Duplicate keywords: 8 model(s)

## Crashes requiring action

### mlx-community/Muse-Glimmer-30B-OptiQ-4bit

- *Execution / usability:* crashed / not evaluated
- *Phase:* model_load
- *Stage:* Model Error
- *Resolved revision:* b4a74fa6001f1eca3b23eeeb702ffad2773a218f

Root exception chain

```text
ValueError: Received 1460 parameters not in model; families: embed_tokens, layers, norm; representative parameters: embed_tokens.biases, embed_tokens.scales, embed_tokens.weight.
caused by: ValueError: Model loading failed: Received 1460 parameters not in model; families: embed_tokens, layers, norm; representative parameters: embed_tokens.biases, embed_tokens.scales, embed_tokens.weight.
```

#### Reproduction inputs

- *Image format:* JPEG
- *Image dimensions:* 9,984 x 6,656 pixels
- *Image size:* 43,299,212 bytes
- *Image SHA-256:* a843ca79cc4b147bd543f362fdd35173cc6793bdf5c739fe4ec9a2a95de92d76

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-08-18 17:46:05 UTC+01:00
- GPS: 55.951722°N, 3.201417°W

Descriptive hints:
- Title hint: City Centre, Edinburgh, Scotland, UK, GBR, Europe
- Description hint: Extensive scaffolding covers the facade of a building undergoing major renovation and redevelopment along Princes Street in Edinburgh, Scotland, while pedestrians walk past temporary construction fences and a Boots pharmacy beneath an overcast sky.
- Keyword hints: 10 Best (structured), Adobe Stock, Any Vision, City Centre, Civil engineering, Construction fence, Construction site, Crane, Edinburgh, Europe, Fence, Modern Architecture, Objects, Overcast, Overcast Sky, Pedestrians, Princes Street, Roadworks, Scaffolding, Scotland

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
python -m mlx_vlm.generate --model mlx-community/Muse-Glimmer-30B-OptiQ-4bit --image any-local-image.jpg --prompt x --max-tokens 8 --temperature 0.0 --revision b4a74fa6001f1eca3b23eeeb702ffad2773a218f --trust-remote-code
```

| Evidence | Link |
| --- | --- |
| Full diagnostics | [model evidence](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-muse-glimmer-30b-optiq-4bit) |
| Detailed issue draft | [crash draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_mlx-community_Muse-Glimmer-30B-OptiQ-4bit.md) |

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
- *Image dimensions:* 9,984 x 6,656 pixels
- *Image size:* 43,299,212 bytes
- *Image SHA-256:* a843ca79cc4b147bd543f362fdd35173cc6793bdf5c739fe4ec9a2a95de92d76

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-08-18 17:46:05 UTC+01:00
- GPS: 55.951722°N, 3.201417°W

Descriptive hints:
- Title hint: City Centre, Edinburgh, Scotland, UK, GBR, Europe
- Description hint: Extensive scaffolding covers the facade of a building undergoing major renovation and redevelopment along Princes Street in Edinburgh, Scotland, while pedestrians walk past temporary construction fences and a Boots pharmacy beneath an overcast sky.
- Keyword hints: 10 Best (structured), Adobe Stock, Any Vision, City Centre, Civil engineering, Construction fence, Construction site, Crane, Edinburgh, Europe, Fence, Modern Architecture, Objects, Overcast, Overcast Sky, Pedestrians, Princes Street, Roadworks, Scaffolding, Scotland

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
| No response text was returned; Required fields are missing or empty | 1 |
| Response repeats the same text; Generation was stopped early after sustained repeated output; Required fields are missing or empty | 1 |
| Response repeats the same text; Generation was stopped early after sustained repeated output; Title or keywords do not meet requested constraints | 1 |
| Response repeats the same text; Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Title or keywords do not meet requested constraints | 1 |
| Response repeats the same text; Response appears cut off at the token limit; Title or keywords do not meet requested constraints | 1 |
| Unrecognised model control tokens remain visible; Title or keywords do not meet requested constraints | 1 |
| Conversation-role control tokens remain visible | 1 |

## Completed attempts requiring review

| Model | Usability | Observed result | Evidence |
| --- | --- | --- | --- |
| mlx-community/gemma-3n-E4B-it-bf16 | unusable | No response text was returned; Missing or empty fields: Title, Description, Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-gemma-3n-e4b-it-bf16) |
| jinaai/jina-vlm-mlx | unusable | Response repeats the same text; Generation was stopped early after sustained repeated output; Keyword list has 49 terms (requested 10-18); Duplicate keywords: european city centre renewal project, european city centre redevelopment project, european city centre transformation project | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-jinaai-jina-vlm-mlx) |
| mlx-community/GLM-4.1V-9B-Thinking-8bit | unusable | Response repeats the same text; Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Keyword list has 53 terms (requested 10-18); Duplicate keywords: princes street, construction site, scaffolding, pedestrians, overcast, boots pharmacy, construction fence, crane, city centre, scotland, europe, roadworks, modern architecture, fence, civil engineering | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-41v-9b-thinking-8bit) |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | unusable | Response repeats the same text; Generation was stopped early after sustained repeated output; Missing or empty fields: Title, Description, Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-smolvlm2-22b-instruct-mlx) |
| mlx-community/X-Reasoner-7B-8bit | unusable | Response repeats the same text; Response appears cut off at the token limit; Title has 4 words (requested 5-10); Keyword list has 308 terms (requested 10-18); Duplicate keywords: scaffolding, construction site, construction equipment, construction progress, building maintenance, construction safety, building renovation, building restoration, commercial building, construction activity, pedestrian walkway, construction site signage, building exterior, construction materials, building facade | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-x-reasoner-7b-8bit) |
| mlx-community/GLM-4.6V-nvfp4 | usable with caveats | Unrecognised model control tokens remain visible; Title has 4 words (requested 5-10) | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-46v-nvfp4) |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 | usable with caveats | Conversation-role control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16) |

## Clean completions

18 clean completions (`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`, `mlx-community/InternVL3-8B-bf16`, `mlx-community/LFM2.5-VL-1.6B-bf16`, `mlx-community/LFM2.5-VL-3B-OptiQ-4bit`, `mlx-community/Ministral-3-3B-Instruct-2512-4bit`, `mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit`, `mlx-community/Phi-3.5-vision-instruct-bf16`, `mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit`, `mlx-community/Qwen3.5-35B-A3B-4bit`, `mlx-community/Qwen3.5-9B-MLX-4bit`, `mlx-community/Qwen3.6-27B-mxfp8`, `mlx-community/Qwen3.8-27B-4bit`, `mlx-community/Step-3.7-Flash-oQ2e`, `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`, `mlx-community/gemma-3-27b-it-qat-4bit`, `mlx-community/gemma-4-26b-a4b-it-4bit`, `mlx-community/gemma-4-31b-it-4bit`, `mlx-community/pixtral-12b-8bit`); 15 more completed with prompt-compliance observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 9,984 x 6,656 pixels, 43,299,212 bytes
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: temperature:* 0.0
- *Generation: top_p:* 1.0
- *Trust remote code:* true
- *check_models version:* 0.16.6
- *check_models revision:* 082cb805666ca30ed48f5b7c35252fe92f945ef1
- *check_models source dirty:* false
- *mlx-vlm:* 0.7.0rc0
- *mlx-vlm source revision:* b0991483509f50058b2db773b672177763a79c4e
- *mlx:* 0.32.3.dev20260830+37c26e575
- *mlx source revision:* 37c26e5755da637255d57ea34b4879196a485301
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
