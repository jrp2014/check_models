# mlx-vlm compatibility findings across 49 cached vision-language models

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

- *Run started:* 2026-09-18 23:30:14 BST
- *Run finished:* 2026-09-18 23:43:34 BST
- *Run duration:* 13m 19s
- *Time by phase:* generation 660s, model load 92s, prompt prep 33s, cleanup
  6s, outside the model loop 14s
- *Evaluation lane:* assisted
- *Prompt hints:* the image's description and keyword hints were included in
  the prompt, so field content may be copied from them rather than seen
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Input image:* JPEG, 9,984 x 6,656 pixels (66.5 MP), 44.7 MB
- *Models attempted:* 49
- *Sampling settings:* checkpoint generation_config.json values for 21 of 46
  completed models where the command line left them unset; harness defaults
  elsewhere
- *Completed:* 46
- *Crashed:* 3
- *Indeterminate:* 0
- *Crashes requiring action:* 3
- *Other results requiring review:* 5
- *Reached token limit:* 3
- *Incomplete output at token limit:* 3
- *Stopped early for repetition:* 4

Observations are mechanical facts from one image, not general model-quality
judgements.

<details>
<summary>Exact prompt sent to every model</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-09-12 17:41:02 UTC+01:00
- GPS: 52.393850°N, 0.270830°E

Descriptive hints:
- Description hint: A solitary white swan glides gracefully across calm river waters framed by lush foliage, with leisure boats and cruisers moored alongside riverside residential buildings in the background.
- Keyword hints: Adobe Stock, Any Vision, Bird, Canal, Greenery, Marina, Mooring, Motorboat, Pier, Riverbank, Swimming, Trees, Vegetation, Water reflection, Waterfowl, Waterfront, Waterway, aquatic bird, architecture, boat

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

- *Baseline:* 44d15af1:src/output/results.jsonl
- *Baseline run timestamp:* 2026-09-18 21:42:05 BST
- *Baseline check_models:* 0.17.26 @ e5d6baca2
- *Baseline mlx:* 0.32.3.dev20260912+229f5b430 @ 59d600b5e
- *Baseline mlx-vlm:* 0.7.1 @ 4774f3d7c
- *Baseline transformers:* 5.17.0
- *Baseline python:* 3.14.7
- *Baseline hardware:* Apple M5 Max, 40 GPU cores, 128.0 GB RAM
- *Models compared:* 49
- *Identical generated text:* 46 of 46 completed in both
- *Generation tok/s ratio (now/baseline):* 1.020 (range 0.95-1.23, 42 models)
- *Prefill tok/s ratio (now/baseline):* 1.316 (range 0.84-4.92, 46 models)
- *Throughput noise band:* history (last 5 same-prompt runs, Tukey fence, at
  least ±10% of the median)

Model revisions changed since the baseline (their rows below reflect a
different snapshot, not only a different run):
`mlx-community/Mage-VL-OptiQ-4bit` bde6c9c71 → c98dad5f9,
`mlx-community/LFM2.5-VL-3B-OptiQ-4bit` 12c5ae493 → 7886c0b4a,
`mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit` 5f31fcd08 → 4620fdbbd,
`mlx-community/Muse-Glimmer-30B-OptiQ-4bit` b4a74fa60 → 98377360c,
`mlx-community/Qwen3.8-27B-4bit` 3e6447f08 → 10c35caaf

| Model | Execution | Usability | Observation delta |
| --- | --- | --- | --- |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | completed | major concerns | +prompt hint repeated |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | completed | no concerns detected → concerns detected | +prompt hint repeated |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | completed | concerns detected | +prompt hint repeated |
| mlx-community/pixtral-12b-8bit | completed | no concerns detected → concerns detected | +prompt hint repeated |
| mlx-community/gemma-3-27b-it-qat-4bit | completed | no concerns detected → concerns detected | +unsupplied place name |
| mlx-community/GLM-4.6V-nvfp4 | completed | no concerns detected → concerns detected | +prompt hint repeated |
| mlx-community/Qwen2-VL-7B-Instruct-4bit | completed | no concerns detected → concerns detected | +prompt hint repeated |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | completed | no concerns detected → major concerns | +cut off at token limit |
| mlx-community/Qwen3.8-27B-4bit | completed | no concerns detected → concerns detected | +unsupplied place name |

| Model | Baseline tok/s | Now tok/s | Ratio | Expected band |
| --- | --- | --- | --- | --- |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit | 182.1 | 213.9 | 1.17 | 154.8-209.4 (fallback) |

<details>
<summary>mlx-vlm: 5 upstream commit(s) since the baseline (4774f3d7c..e79b0e041)</summary>

- e79b0e04 Merge pull request #2306 from lucasnewman/llada-image-support
- 1d0d16ec Remove unrelated change.
- 1b4f992b Merge remote-tracking branch 'origin/main' into llada-image-support
- b3a7e346 Add support for LLaDa-Image model family.
- 48e3521b Automatically handle repo variants of supported image models.

</details>

Mechanical diff only: one image, temperature as configured; single-observation
flips on one model are usually run-to-run variance, broad shifts are not.

## Model quality at a glance

Every attempted model ranked by mechanical observations, with captured
resource facts. No concerns detected is not a task-compliance or accuracy
verdict. Consult the assessment scope above and inspect the final answers.
Crashes and integration signals have expanded maintainer evidence. A
major-concerns row whose observations are format failures only (for example
labelled fields not detected) is a chooser verdict, not a maintainer signal,
so it is not repeated under attempts requiring review; that list holds results
whose observations may point at mlx-vlm rather than at the model.

| Model | Mechanical checks | Total | Gen tok/s | Peak GB | Observed |
| --- | --- | --- | --- | --- | --- |
| mlx-community/aya-vision-8b-4bit | no concerns detected | 4.62s | 100 tok/s | 6.5 | none |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | no concerns detected | 11.48s | 29.6 tok/s | 23 | none |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | no concerns detected | 16.34s | 77.8 tok/s | 19 | none |
| mlx-community/gemma-4-12B-it-4bit | no concerns detected | 5.10s | 60.4 tok/s | 7.6 | none |
| mlx-community/gemma-4-26b-a4b-it-4bit | no concerns detected | 5.04s | 108 tok/s | 16 | none |
| mlx-community/gemma-4-31b-it-4bit | no concerns detected | 8.45s | 26.3 tok/s | 20 | none |
| mlx-community/gemma-4-e4b-it-4bit | no concerns detected | 3.74s | 125 tok/s | 6.0 | none |
| mlx-community/GLM-4.6V-Flash-4bit | no concerns detected | 9.76s | 74.1 tok/s | 8.8 | none |
| mlx-community/granite-4.0-3b-vision-4bit | no concerns detected | 2.91s | 176 tok/s | 4.7 | none |
| mlx-community/Idefics3-8B-Llama3-bf16 | no concerns detected | 8.50s | 35.5 tok/s | 18 | none |
| mlx-community/InternVL3-14B-4bit | no concerns detected | 6.02s | 57.4 tok/s | 10 | none |
| mlx-community/InternVL3-8B-bf16 | no concerns detected | 5.78s | 37.7 tok/s | 17 | none |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit | no concerns detected | 3.36s | 214 tok/s | 4.0 | none |
| mlx-community/MiniCPM-o-4_5-4bit | no concerns detected | 3.23s | 105 tok/s | 7.0 | none |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | no concerns detected | 6.77s | 67.2 tok/s | 13 | none |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4 | no concerns detected | 7.77s | 62.3 tok/s | 13 | none |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | no concerns detected | 3.75s | 191 tok/s | 7.8 | none |
| mlx-community/North-Micro-Vision-Instruct-4bit | no concerns detected | 4.87s | 154 tok/s | 3.9 | none |
| mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit | no concerns detected | 6.73s | 71.8 tok/s | 24 | none |
| mlx-community/Phi-3.5-vision-instruct-bf16 | no concerns detected | 4.21s | 54.9 tok/s | 9.3 | none |
| mlx-community/Qwen3-VL-2B-Thinking-bf16 | no concerns detected | 27.70s | 86.3 tok/s | 8.4 | none |
| mlx-community/Qwen3-VL-32B-Instruct-4bit | no concerns detected | 77.47s | 20.4 tok/s | 26 | none |
| mlx-community/Qwen3-VL-8B-Instruct-4bit | no concerns detected | 39.47s | 69.7 tok/s | 11 | none |
| mlx-community/Qwen3.5-35B-A3B-4bit | no concerns detected | 39.73s | 72.4 tok/s | 25 | none |
| mlx-community/Qwen3.5-9B-MLX-4bit | no concerns detected | 44.16s | 91.4 tok/s | 11 | none |
| mlx-community/Step-3.7-Flash-oQ3e | no concerns detected | 36.64s | 47.4 tok/s | 92 | none |
| mlx-community/X-Reasoner-7B-8bit | no concerns detected | 18.44s | 57.6 tok/s | 14 | none |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | concerns detected | 6.81s | 46.1 tok/s | 28 | duplicate keywords; prompt hint repeated |
| mlx-community/gemma-3-27b-it-qat-4bit | concerns detected | 9.47s | 30.3 tok/s | 17 | unsupplied place name |
| mlx-community/GLM-4.6V-nvfp4 | concerns detected | 23.56s | 41.6 tok/s | 78 | prompt hint repeated |
| mlx-community/Molmo2-8B-4bit | concerns detected | 6.16s | 72.5 tok/s | 8.6 | duplicate keywords |
| mlx-community/pixtral-12b-8bit | concerns detected | 7.12s | 40.2 tok/s | 16 | prompt hint repeated |
| mlx-community/Qwen2-VL-7B-Instruct-4bit | concerns detected | 41.22s | 91.3 tok/s | 9.3 | prompt hint repeated |
| mlx-community/Qwen3.8-27B-4bit | concerns detected | 65.24s | 29.8 tok/s | 21 | unsupplied place name |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | concerns detected | 3.33s | 131 tok/s | 5.6 | prompt hint repeated |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | major concerns | 2.26s | 476 tok/s | 1.9 | repeated text; stopped early: repeating; duplicate keywords; prompt hint repeated |
| mlx-community/gemma-3n-E4B-it-4bit | major concerns | 4.43s | 71.0 tok/s | 7.2 | labelled fields not detected |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit | major concerns | 20.21s | 63.0 tok/s | 20 | cut off at token limit |
| mlx-community/Llama-3.2-11B-Vision-Instruct-4bit | major concerns | 14.66s | 25.7 tok/s | 9.8 | repeated text; stopped early: repeating; duplicate keywords |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | major concerns | 57.79s | 18.8 tok/s | 15 | repeated text; cut off at token limit |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns | 5.26s | 102 tok/s | 6.7 | repeated text; stopped early: repeating; control tokens visible; labelled fields not detected |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | major concerns | 53.04s | 24.3 tok/s | 25 | cut off at token limit |
| mlx-community/nanoLLaVA-1.5-4bit | major concerns | 2.34s | 193 tok/s | 1.8 | labelled fields not detected |
| mlx-community/paligemma2-10b-mix-448-4bit | major concerns | 4.02s | insufficient sample | 9.7 | labelled fields not detected |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | major concerns | 41.31s | 74.7 tok/s | 23 | stopped early: repeating; duplicate keywords |
| mlx-community/SmolVLM-256M-Instruct-4bit | major concerns | 2.21s | 320 tok/s | 1.1 | labelled fields not detected |
| mlx-community/InternVL3_5-1B-4bit | not assessed | 0.16s | - | - | crashed during model loading |
| mlx-community/Mage-VL-OptiQ-4bit | not assessed | 2.13s | - | - | crashed during generation, before first token |
| mlx-community/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-4bit | not assessed | 0.25s | - | - | crashed during model loading |

## Constraint-failure breakdown

How the fleet failed the catalogue constraints — a skew toward one constraint
suggests prompt difficulty rather than individual model faults.

- Duplicate keywords: 5 model(s)

## Crashes requiring action

### mlx-community/InternVL3_5-1B-4bit

- *Execution / usability:* crashed / not assessed
- *Phase:* model_load
- *Stage:* Unsupported Arch
- *Resolved revision:* f9d179a8be8ac53e96c6ee5cce8493856d4b8f09

Root exception chain

```text
ValueError: Model type internvl not supported. Error: No module named 'mlx_vlm.speculative.drafters.internvl'
caused by: ValueError: Model loading failed: Model type internvl not supported. Error: No module named 'mlx_vlm.speculative.drafters.internvl'
```

#### Reproduction inputs

- *Image format:* JPEG
- *Image dimensions:* 9,984 x 6,656 pixels
- *Image size:* 44,654,464 bytes
- *Image SHA-256:* c109700d51a838d36e8d56fc769534be3598a8af6310eed6d0afa19ecbe7dccf

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-09-12 17:41:02 UTC+01:00
- GPS: 52.393850°N, 0.270830°E

Descriptive hints:
- Description hint: A solitary white swan glides gracefully across calm river waters framed by lush foliage, with leisure boats and cruisers moored alongside riverside residential buildings in the background.
- Keyword hints: Adobe Stock, Any Vision, Bird, Canal, Greenery, Marina, Mooring, Motorboat, Pier, Riverbank, Swimming, Trees, Vegetation, Water reflection, Waterfowl, Waterfront, Waterway, aquatic bird, architecture, boat

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
python -m mlx_vlm.generate --model mlx-community/InternVL3_5-1B-4bit --image any-local-image.jpg --prompt x --max-tokens 8 --temperature 0.0 --revision f9d179a8be8ac53e96c6ee5cce8493856d4b8f09 --trust-remote-code
```

| Evidence | Link |
| --- | --- |
| Full diagnostics | [model evidence](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-internvl35-1b-4bit) |
| Detailed issue draft | [crash draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_mlx-community_InternVL3_5-1B-4bit.md) |

### mlx-community/Mage-VL-OptiQ-4bit

- *Execution / usability:* crashed / not assessed
- *Phase:* generation_before_first_token
- *Stage:* Model Error
- *Resolved revision:* c98dad5f92f13334cc679c93fb9f185ab49ed626

Root exception chain

```text
ValueError: cu_seqlens mismatch: total_patches=7600 calculated=3800 grid=[(1, 50, 76)]
caused by: ValueError: Model generation failed for mlx-community/Mage-VL-OptiQ-4bit: cu_seqlens mismatch: total_patches=7600 calculated=3800 grid=[(1, 50, 76)]
```

#### Reproduction inputs

- *Image format:* JPEG
- *Image dimensions:* 9,984 x 6,656 pixels
- *Image size:* 44,654,464 bytes
- *Image SHA-256:* c109700d51a838d36e8d56fc769534be3598a8af6310eed6d0afa19ecbe7dccf

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-09-12 17:41:02 UTC+01:00
- GPS: 52.393850°N, 0.270830°E

Descriptive hints:
- Description hint: A solitary white swan glides gracefully across calm river waters framed by lush foliage, with leisure boats and cruisers moored alongside riverside residential buildings in the background.
- Keyword hints: Adobe Stock, Any Vision, Bird, Canal, Greenery, Marina, Mooring, Motorboat, Pier, Riverbank, Swimming, Trees, Vegetation, Water reflection, Waterfowl, Waterfront, Waterway, aquatic bird, architecture, boat

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
| Full diagnostics | [model evidence](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-mage-vl-optiq-4bit) |
| Detailed issue draft | [crash draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_mlx-community_Mage-VL-OptiQ-4bit.md) |

### mlx-community/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-4bit

- *Execution / usability:* crashed / not assessed
- *Phase:* model_load
- *Stage:* Model Error
- *Resolved revision:* 8f86ad8f279ce1ec3b8b65970f32da1e89ab7a45

Root exception chain

```text
ValueError: Received 729 parameters not in model; families: backbone, lm_head; representative parameters: backbone.embeddings.biases, backbone.embeddings.scales, backbone.embeddings.weight.
caused by: ValueError: Model loading failed: Received 729 parameters not in model; families: backbone, lm_head; representative parameters: backbone.embeddings.biases, backbone.embeddings.scales, backbone.embeddings.weight.
```

#### Reproduction inputs

- *Image format:* JPEG
- *Image dimensions:* 9,984 x 6,656 pixels
- *Image size:* 44,654,464 bytes
- *Image SHA-256:* c109700d51a838d36e8d56fc769534be3598a8af6310eed6d0afa19ecbe7dccf

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-09-12 17:41:02 UTC+01:00
- GPS: 52.393850°N, 0.270830°E

Descriptive hints:
- Description hint: A solitary white swan glides gracefully across calm river waters framed by lush foliage, with leisure boats and cruisers moored alongside riverside residential buildings in the background.
- Keyword hints: Adobe Stock, Any Vision, Bird, Canal, Greenery, Marina, Mooring, Motorboat, Pier, Riverbank, Swimming, Trees, Vegetation, Water reflection, Waterfowl, Waterfront, Waterway, aquatic bird, architecture, boat

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
python -m mlx_vlm.generate --model mlx-community/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-4bit --image any-local-image.jpg --prompt x --max-tokens 8 --temperature 0.0 --revision 8f86ad8f279ce1ec3b8b65970f32da1e89ab7a45 --trust-remote-code
```

| Evidence | Link |
| --- | --- |
| Full diagnostics | [model evidence](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-nvidia-nemotron-3-nano-omni-30b-a3b-4bit) |
| Detailed issue draft | [crash draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_mlx-community_NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-4bit.md) |

## Observation clusters

Repeated mechanical observation signatures among results requiring review.

| Observed result | Models |
| --- | --- |
| Response repeats the same text; Generation was stopped early after sustained repeated output; Repeated keyword entries; Output repeats the prompt's own hint text instead of describing the image | 1 |
| Response repeats the same text; Generation was stopped early after sustained repeated output; Unrecognised model control tokens remain visible; Required labelled fields not detected | 1 |
| Response repeats the same text; Generation was stopped early after sustained repeated output; Repeated keyword entries | 1 |
| Response repeats the same text; Response appears cut off at the token limit | 1 |
| Generation was stopped early after sustained repeated output; Repeated keyword entries | 1 |

## Completed attempts requiring review

| Model | Mechanical checks | Observed result | Evidence |
| --- | --- | --- | --- |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | major concerns | Response repeats the same text; Generation was stopped early after sustained repeated output; Duplicate keywords: canal, water, architecture, waterway, aquatic bird, waterfront; Repeats the prompt's hint instead of describing the image: description | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-liquidai-lfm25-vl-450m-mlx-bf16) |
| mlx-community/Llama-3.2-11B-Vision-Instruct-4bit | major concerns | Response repeats the same text; Generation was stopped early after sustained repeated output; Duplicate keywords: riverside, riverside scenery, riverside beauty, riverside charm, riverside tranquility, riverside serenity, riverside calmness, riverside peacefulness | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llama-32-11b-vision-instruct-4bit) |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | major concerns | Response repeats the same text; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llama-32-11b-vision-instruct-8bit) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns | Response repeats the same text; Generation was stopped early after sustained repeated output; Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description, keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit) |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | major concerns | Generation was stopped early after sustained repeated output; Duplicate keywords: wildlife, animal, bird | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-qwen3-vl-30b-a3b-instruct-4bit) |

## Completions without detected concerns

27 completions without detected concerns (`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`, `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit`, `mlx-community/GLM-4.6V-Flash-4bit`, `mlx-community/Idefics3-8B-Llama3-bf16`, `mlx-community/InternVL3-14B-4bit`, `mlx-community/InternVL3-8B-bf16`, `mlx-community/LFM2.5-VL-3B-OptiQ-4bit`, `mlx-community/MiniCPM-o-4_5-4bit`, `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`, `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`, `mlx-community/Ministral-3-3B-Instruct-2512-4bit`, `mlx-community/North-Micro-Vision-Instruct-4bit`, `mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit`, `mlx-community/Phi-3.5-vision-instruct-bf16`, `mlx-community/Qwen3-VL-2B-Thinking-bf16`, `mlx-community/Qwen3-VL-32B-Instruct-4bit`, `mlx-community/Qwen3-VL-8B-Instruct-4bit`, `mlx-community/Qwen3.5-35B-A3B-4bit`, `mlx-community/Qwen3.5-9B-MLX-4bit`, `mlx-community/Step-3.7-Flash-oQ3e`, `mlx-community/X-Reasoner-7B-8bit`, `mlx-community/aya-vision-8b-4bit`, `mlx-community/gemma-4-12B-it-4bit`, `mlx-community/gemma-4-26b-a4b-it-4bit`, `mlx-community/gemma-4-31b-it-4bit`, `mlx-community/gemma-4-e4b-it-4bit`, `mlx-community/granite-4.0-3b-vision-4bit`); 14 more completed with prompt-compliance observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 9,984 x 6,656 pixels, 44,654,464 bytes
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: seed:* 0
- *Trust remote code:* true
- *check_models version:* 0.17.31
- *check_models revision:* 44d15af188596e99645f997ef7551bdd95ce39b1
- *check_models source dirty:* true
- *mlx-vlm:* 0.7.1
- *mlx-vlm source revision:* e79b0e041677ec4ca5333ba750376bb4e8c434cb
- *mlx:* 0.32.3.dev20260912+229f5b430
- *mlx source revision:* 229f5b430
- *transformers:* 5.17.0
- *macOS Version:* 27.0
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
