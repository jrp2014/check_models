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

- *Run started:* 2026-09-18 21:27:50 BST
- *Run finished:* 2026-09-18 21:42:03 BST
- *Run duration:* 14m 12s
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
- *Incomplete output at token limit:* 2
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

- *Baseline:* e5d6baca:src/output/results.jsonl
- *Baseline run timestamp:* 2026-09-13 21:59:50 BST
- *Baseline check_models:* 0.17.25 @ 11e6c82dc
- *Baseline mlx:* 0.32.3.dev20260912+229f5b430 @ 229f5b430
- *Baseline mlx-vlm:* 0.7.0 @ 45d6e125a
- *Baseline transformers:* 5.17.0
- *Baseline python:* 3.14.7
- *Baseline hardware:* Apple M5 Max, 40 GPU cores, 128.0 GB RAM
- *Models compared:* 49
- *Identical generated text:* 40 of 44 completed in both
- *Generation tok/s ratio (now/baseline):* 0.996 (range 0.86-1.16, 41 models)
- *Throughput noise band:* history (last 4 same-prompt runs, Tukey fence, at
  least ±10% of the median)

- In baseline, not run this time: `apple/FastVLM-7B-int4`,
  `mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit`

| Model | Execution | Usability | Observation delta |
| --- | --- | --- | --- |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | completed | concerns detected → major concerns | +repeated text; +stopped early: repeating |
| mlx-community/aya-vision-8b-4bit | completed | concerns detected → no concerns detected | -control tokens visible |
| mlx-community/Llama-3.2-11B-Vision-Instruct-4bit | crashed → completed | not assessed → major concerns | +duplicate keywords; +repeated text; +stopped early: repeating |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | completed | major concerns → no concerns detected | -labelled fields not detected; -role tokens visible; -cut off at token limit; -control tokens visible |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | crashed → completed | not assessed → major concerns | +repeated text; +cut off at token limit |

| Model | Baseline tok/s | Now tok/s | Ratio | Expected band |
| --- | --- | --- | --- | --- |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit | 206.3 | 182.1 | 0.88 | 185.3-226.5 (history, n=4) |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | 43.6 | 37.5 | 0.86 | 43.7-57.1 (history, n=4) |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | 21.4 | 23.6 | 1.10 | 19.2-23.5 (history, n=4) |

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
| mlx-community/aya-vision-8b-4bit | no concerns detected | 4.94s | 101 tok/s | 6.5 | none |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | no concerns detected | 12.43s | 30.4 tok/s | 23 | none |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | no concerns detected | 17.04s | 82.2 tok/s | 19 | none |
| mlx-community/gemma-3-27b-it-qat-4bit | no concerns detected | 10.01s | 30.2 tok/s | 17 | none |
| mlx-community/gemma-4-12B-it-4bit | no concerns detected | 5.67s | 59.5 tok/s | 7.6 | none |
| mlx-community/gemma-4-26b-a4b-it-4bit | no concerns detected | 5.49s | 106 tok/s | 16 | none |
| mlx-community/gemma-4-31b-it-4bit | no concerns detected | 8.88s | 25.8 tok/s | 20 | none |
| mlx-community/gemma-4-e4b-it-4bit | no concerns detected | 4.37s | 122 tok/s | 5.9 | none |
| mlx-community/GLM-4.6V-Flash-4bit | no concerns detected | 10.48s | 75.8 tok/s | 8.7 | none |
| mlx-community/GLM-4.6V-nvfp4 | no concerns detected | 33.23s | 41.2 tok/s | 78 | none |
| mlx-community/granite-4.0-3b-vision-4bit | no concerns detected | 4.10s | 168 tok/s | 4.7 | none |
| mlx-community/Idefics3-8B-Llama3-bf16 | no concerns detected | 9.45s | 35.2 tok/s | 18 | none |
| mlx-community/InternVL3-14B-4bit | no concerns detected | 8.03s | 54.1 tok/s | 10 | none |
| mlx-community/InternVL3-8B-bf16 | no concerns detected | 6.17s | 36.4 tok/s | 17 | none |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit | no concerns detected | 5.24s | 182 tok/s | 4.0 | none |
| mlx-community/MiniCPM-o-4_5-4bit | no concerns detected | 3.87s | 102 tok/s | 7.0 | none |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | no concerns detected | 6.52s | 64.3 tok/s | 13 | none |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4 | no concerns detected | 8.08s | 60.2 tok/s | 13 | none |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | no concerns detected | 3.91s | 179 tok/s | 7.8 | none |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | no concerns detected | 58.62s | 23.6 tok/s | 25 | none |
| mlx-community/North-Micro-Vision-Instruct-4bit | no concerns detected | 5.13s | 159 tok/s | 3.9 | none |
| mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit | no concerns detected | 8.16s | 74.4 tok/s | 24 | none |
| mlx-community/Phi-3.5-vision-instruct-bf16 | no concerns detected | 4.50s | 55.9 tok/s | 9.3 | none |
| mlx-community/pixtral-12b-8bit | no concerns detected | 7.59s | 38.7 tok/s | 16 | none |
| mlx-community/Qwen2-VL-7B-Instruct-4bit | no concerns detected | 43.44s | 89.1 tok/s | 9.3 | none |
| mlx-community/Qwen3-VL-2B-Thinking-bf16 | no concerns detected | 29.50s | 86.1 tok/s | 8.4 | none |
| mlx-community/Qwen3-VL-32B-Instruct-4bit | no concerns detected | 74.99s | 19.5 tok/s | 26 | none |
| mlx-community/Qwen3-VL-8B-Instruct-4bit | no concerns detected | 41.03s | 69.7 tok/s | 11 | none |
| mlx-community/Qwen3.5-35B-A3B-4bit | no concerns detected | 38.20s | 73.6 tok/s | 25 | none |
| mlx-community/Qwen3.5-9B-MLX-4bit | no concerns detected | 38.00s | 89.6 tok/s | 11 | none |
| mlx-community/Qwen3.8-27B-4bit | no concerns detected | 65.42s | 26.8 tok/s | 21 | none |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | no concerns detected | 3.55s | 129 tok/s | 5.6 | none |
| mlx-community/Step-3.7-Flash-oQ3e | no concerns detected | 43.93s | 47.8 tok/s | 92 | none |
| mlx-community/X-Reasoner-7B-8bit | no concerns detected | 19.44s | 56.8 tok/s | 14 | none |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | concerns detected | 8.87s | 37.5 tok/s | 28 | duplicate keywords |
| mlx-community/Molmo2-8B-4bit | concerns detected | 8.49s | 71.5 tok/s | 8.1 | duplicate keywords |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | major concerns | 2.41s | 470 tok/s | 1.9 | repeated text; stopped early: repeating; duplicate keywords |
| mlx-community/gemma-3n-E4B-it-4bit | major concerns | 5.98s | 69.6 tok/s | 6.9 | labelled fields not detected |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit | major concerns | 23.62s | 60.0 tok/s | 20 | cut off at token limit |
| mlx-community/Llama-3.2-11B-Vision-Instruct-4bit | major concerns | 15.93s | 26.1 tok/s | 9.2 | repeated text; stopped early: repeating; duplicate keywords |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | major concerns | 63.38s | 17.0 tok/s | 15 | repeated text; cut off at token limit |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns | 5.51s | 98.7 tok/s | 6.7 | repeated text; stopped early: repeating; control tokens visible; labelled fields not detected |
| mlx-community/nanoLLaVA-1.5-4bit | major concerns | 2.73s | 198 tok/s | 1.5 | labelled fields not detected |
| mlx-community/paligemma2-10b-mix-448-4bit | major concerns | 5.05s | insufficient sample | 9.7 | labelled fields not detected |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | major concerns | 40.50s | 74.8 tok/s | 23 | stopped early: repeating; duplicate keywords |
| mlx-community/SmolVLM-256M-Instruct-4bit | major concerns | 2.46s | 314 tok/s | 1.1 | labelled fields not detected |
| mlx-community/InternVL3_5-1B-4bit | not assessed | 0.19s | - | - | crashed during model loading |
| mlx-community/Mage-VL-OptiQ-4bit | not assessed | 2.79s | - | - | crashed during generation, before first token |
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
- *Resolved revision:* bde6c9c7146acff6af09e203245014f19306c5c5

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
| Response repeats the same text; Generation was stopped early after sustained repeated output; Repeated keyword entries | 2 |
| Response repeats the same text; Generation was stopped early after sustained repeated output; Unrecognised model control tokens remain visible; Required labelled fields not detected | 1 |
| Response repeats the same text; Response appears cut off at the token limit | 1 |
| Generation was stopped early after sustained repeated output; Repeated keyword entries | 1 |

## Completed attempts requiring review

| Model | Mechanical checks | Observed result | Evidence |
| --- | --- | --- | --- |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | major concerns | Response repeats the same text; Generation was stopped early after sustained repeated output; Duplicate keywords: canal, water, architecture, waterway, aquatic bird, waterfront | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-liquidai-lfm25-vl-450m-mlx-bf16) |
| mlx-community/Llama-3.2-11B-Vision-Instruct-4bit | major concerns | Response repeats the same text; Generation was stopped early after sustained repeated output; Duplicate keywords: riverside, riverside scenery, riverside beauty, riverside charm, riverside tranquility, riverside serenity, riverside calmness, riverside peacefulness | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llama-32-11b-vision-instruct-4bit) |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | major concerns | Response repeats the same text; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llama-32-11b-vision-instruct-8bit) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns | Response repeats the same text; Generation was stopped early after sustained repeated output; Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description, keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit) |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | major concerns | Generation was stopped early after sustained repeated output; Duplicate keywords: wildlife, animal, bird | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-qwen3-vl-30b-a3b-instruct-4bit) |

## Completions without detected concerns

34 completions without detected concerns (`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`, `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit`, `mlx-community/GLM-4.6V-Flash-4bit`, `mlx-community/GLM-4.6V-nvfp4`, `mlx-community/Idefics3-8B-Llama3-bf16`, `mlx-community/InternVL3-14B-4bit`, `mlx-community/InternVL3-8B-bf16`, `mlx-community/LFM2.5-VL-3B-OptiQ-4bit`, `mlx-community/MiniCPM-o-4_5-4bit`, `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`, `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`, `mlx-community/Ministral-3-3B-Instruct-2512-4bit`, `mlx-community/Muse-Glimmer-30B-OptiQ-4bit`, `mlx-community/North-Micro-Vision-Instruct-4bit`, `mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit`, `mlx-community/Phi-3.5-vision-instruct-bf16`, `mlx-community/Qwen2-VL-7B-Instruct-4bit`, `mlx-community/Qwen3-VL-2B-Thinking-bf16`, `mlx-community/Qwen3-VL-32B-Instruct-4bit`, `mlx-community/Qwen3-VL-8B-Instruct-4bit`, `mlx-community/Qwen3.5-35B-A3B-4bit`, `mlx-community/Qwen3.5-9B-MLX-4bit`, `mlx-community/Qwen3.8-27B-4bit`, `mlx-community/SmolVLM2-2.2B-Instruct-mlx`, `mlx-community/Step-3.7-Flash-oQ3e`, `mlx-community/X-Reasoner-7B-8bit`, `mlx-community/aya-vision-8b-4bit`, `mlx-community/gemma-3-27b-it-qat-4bit`, `mlx-community/gemma-4-12B-it-4bit`, `mlx-community/gemma-4-26b-a4b-it-4bit`, `mlx-community/gemma-4-31b-it-4bit`, `mlx-community/gemma-4-e4b-it-4bit`, `mlx-community/granite-4.0-3b-vision-4bit`, `mlx-community/pixtral-12b-8bit`); 7 more completed with prompt-compliance observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 9,984 x 6,656 pixels, 44,654,464 bytes
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: seed:* 0
- *Trust remote code:* true
- *check_models version:* 0.17.26
- *check_models revision:* e5d6baca2ad0c1dd4365a0738b6bbd9c1321ea00
- *check_models source dirty:* false
- *mlx-vlm:* 0.7.1
- *mlx-vlm source revision:* 4774f3d7cc1ab762f665aeb1f7ec775e4f97d01a
- *mlx:* 0.32.3.dev20260912+229f5b430
- *mlx source revision:* 59d600b5e64c238427d0f8d897ab7c682ef4d3d2
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
