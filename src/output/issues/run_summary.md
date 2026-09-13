# mlx-vlm compatibility findings across 51 cached vision-language models

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

- *Run started:* 2026-09-13 21:47:09 BST
- *Run finished:* 2026-09-13 21:59:48 BST
- *Run duration:* 12m 37s
- *Evaluation lane:* assisted
- *Prompt hints:* the image's description and keyword hints were included in
  the prompt, so field content may be copied from them rather than seen
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Input image:* JPEG, 9,984 x 6,656 pixels (66.5 MP), 44.7 MB
- *Models attempted:* 51
- *Sampling settings:* checkpoint generation_config.json values for 21 of 44
  completed models where the command line left them unset; harness defaults
  elsewhere
- *Completed:* 44
- *Crashed:* 7
- *Indeterminate:* 0
- *Crashes requiring action:* 7
- *Other results requiring review:* 4
- *Reached token limit:* 2
- *Incomplete output at token limit:* 2
- *Stopped early for repetition:* 2

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

- *Baseline:* 11e6c82d:src/output/results.jsonl
- *Baseline run timestamp:* 2026-09-13 02:35:21 BST
- *Baseline check_models:* 0.17.24 @ 09b5430fd
- *Baseline mlx:* 0.32.3.dev20260912+229f5b430 @ 229f5b430
- *Baseline mlx-vlm:* 0.7.0 @ 45d6e125a
- *Baseline transformers:* 5.17.0
- *Baseline python:* 3.14.7
- *Baseline hardware:* Apple M5 Max, 40 GPU cores, 128.0 GB RAM
- *Models compared:* 40
- *Identical generated text:* 38 of 38 completed in both
- *Generation tok/s ratio (now/baseline):* 1.002 (range 0.82-1.92, 36 models)
- *Throughput noise band:* fixed ±15% fallback (insufficient history)

- New this run (no baseline): 11 models (targeted run against a full-sweep
  baseline)
- In baseline, not run this time: `mlx-community/InternVL3_5-30B-A3B-4bit`

No execution, usability, or observation-set changes against the baseline.

| Model | Baseline tok/s | Now tok/s | Ratio | Expected band |
| --- | --- | --- | --- | --- |
| mlx-community/gemma-3n-E4B-it-4bit | 60.3 | 70.5 | 1.17 | 51.3-69.4 (fallback) |
| mlx-community/Phi-3.5-vision-instruct-bf16 | 36.6 | 55.0 | 1.50 | 31.1-42.1 (fallback) |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | 53.0 | 43.6 | 0.82 | 45.0-60.9 (fallback) |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4 | 64.1 | 53.2 | 0.83 | 54.5-73.7 (fallback) |
| mlx-community/Qwen3-VL-2B-Thinking-bf16 | 69.8 | 85.9 | 1.23 | 59.4-80.3 (fallback) |
| mlx-community/Qwen3-VL-8B-Instruct-4bit | 59.4 | 70.5 | 1.19 | 50.5-68.4 (fallback) |
| mlx-community/Qwen3.8-27B-4bit | 14.4 | 27.5 | 1.92 | 12.2-16.5 (fallback) |

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
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | no concerns detected | 11.48s | 29.5 tok/s | 23 | none |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | no concerns detected | 16.26s | 78.8 tok/s | 19 | none |
| mlx-community/gemma-3-27b-it-qat-4bit | no concerns detected | 9.38s | 30.7 tok/s | 17 | none |
| mlx-community/gemma-4-12B-it-4bit | no concerns detected | 5.33s | 60.7 tok/s | 7.6 | none |
| mlx-community/gemma-4-26b-a4b-it-4bit | no concerns detected | 5.00s | 104 tok/s | 16 | none |
| mlx-community/gemma-4-31b-it-4bit | no concerns detected | 8.54s | 25.8 tok/s | 20 | none |
| mlx-community/gemma-4-e4b-it-4bit | no concerns detected | 3.67s | 123 tok/s | 5.9 | none |
| mlx-community/GLM-4.6V-Flash-4bit | no concerns detected | 9.62s | 74.9 tok/s | 8.7 | none |
| mlx-community/GLM-4.6V-nvfp4 | no concerns detected | 21.87s | 41.1 tok/s | 78 | none |
| mlx-community/granite-4.0-3b-vision-4bit | no concerns detected | 3.00s | 172 tok/s | 4.7 | none |
| mlx-community/Idefics3-8B-Llama3-bf16 | no concerns detected | 9.01s | 32.3 tok/s | 18 | none |
| mlx-community/InternVL3-14B-4bit | no concerns detected | 6.41s | 55.4 tok/s | 10 | none |
| mlx-community/InternVL3-8B-bf16 | no concerns detected | 6.39s | 31.4 tok/s | 17 | none |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit | no concerns detected | 3.15s | 206 tok/s | 4.0 | none |
| mlx-community/MiniCPM-o-4_5-4bit | no concerns detected | 3.40s | 104 tok/s | 7.0 | none |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | no concerns detected | 6.58s | 64.7 tok/s | 13 | none |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4 | no concerns detected | 8.50s | 53.2 tok/s | 13 | none |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | no concerns detected | 3.90s | 181 tok/s | 7.8 | none |
| mlx-community/North-Micro-Vision-Instruct-4bit | no concerns detected | 5.00s | 166 tok/s | 3.9 | none |
| mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit | no concerns detected | 6.58s | 75.7 tok/s | 24 | none |
| mlx-community/Phi-3.5-vision-instruct-bf16 | no concerns detected | 4.20s | 55.0 tok/s | 9.3 | none |
| mlx-community/pixtral-12b-8bit | no concerns detected | 7.61s | 38.4 tok/s | 16 | none |
| mlx-community/Qwen2-VL-7B-Instruct-4bit | no concerns detected | 45.84s | 90.6 tok/s | 9.3 | none |
| mlx-community/Qwen3-VL-2B-Thinking-bf16 | no concerns detected | 29.64s | 85.9 tok/s | 8.4 | none |
| mlx-community/Qwen3-VL-32B-Instruct-4bit | no concerns detected | 76.75s | 18.6 tok/s | 26 | none |
| mlx-community/Qwen3-VL-8B-Instruct-4bit | no concerns detected | 42.99s | 70.5 tok/s | 11 | none |
| mlx-community/Qwen3.5-35B-A3B-4bit | no concerns detected | 38.97s | 72.5 tok/s | 25 | none |
| mlx-community/Qwen3.5-9B-MLX-4bit | no concerns detected | 40.15s | 89.0 tok/s | 11 | none |
| mlx-community/Qwen3.8-27B-4bit | no concerns detected | 62.11s | 27.5 tok/s | 21 | none |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | no concerns detected | 3.32s | 125 tok/s | 5.6 | none |
| mlx-community/Step-3.7-Flash-oQ3e | no concerns detected | 40.68s | 47.3 tok/s | 92 | none |
| mlx-community/X-Reasoner-7B-8bit | no concerns detected | 18.87s | 58.1 tok/s | 14 | none |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | concerns detected | 2.08s | 475 tok/s | 1.9 | duplicate keywords |
| mlx-community/aya-vision-8b-4bit | concerns detected | 4.58s | 102 tok/s | 6.5 | control tokens visible |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | concerns detected | 7.22s | 43.6 tok/s | 28 | duplicate keywords |
| mlx-community/Molmo2-8B-4bit | concerns detected | 6.44s | 68.9 tok/s | 8.1 | duplicate keywords |
| mlx-community/gemma-3n-E4B-it-4bit | major concerns | 4.47s | 70.5 tok/s | 7.2 | labelled fields not detected |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit | major concerns | 22.86s | 54.5 tok/s | 20 | cut off at token limit |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns | 5.63s | 99.5 tok/s | 6.7 | repeated text; stopped early: repeating; control tokens visible; labelled fields not detected |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | major concerns | 61.04s | 21.4 tok/s | 25 | control tokens visible; labelled fields not detected; cut off at token limit; role tokens visible |
| mlx-community/nanoLLaVA-1.5-4bit | major concerns | 2.27s | 202 tok/s | 1.8 | labelled fields not detected |
| mlx-community/paligemma2-10b-mix-448-4bit | major concerns | 4.73s | insufficient sample | 9.7 | labelled fields not detected |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | major concerns | 44.51s | 74.4 tok/s | 23 | stopped early: repeating; duplicate keywords |
| mlx-community/SmolVLM-256M-Instruct-4bit | major concerns | 2.35s | 315 tok/s | 1.1 | labelled fields not detected |
| apple/FastVLM-7B-int4 | not assessed | 0.14s | - | - | crashed during model loading |
| mlx-community/InternVL3_5-1B-4bit | not assessed | 0.14s | - | - | crashed during model loading |
| mlx-community/Llama-3.2-11B-Vision-Instruct-4bit | not assessed | 2.49s | - | - | crashed during generation, before first token |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | not assessed | 2.88s | - | - | crashed during generation, before first token |
| mlx-community/Mage-VL-OptiQ-4bit | not assessed | 0.25s | - | - | crashed during model loading |
| mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit | not assessed | 1.85s | - | - | crashed during prompt preparation |
| mlx-community/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-4bit | not assessed | 0.29s | - | - | crashed during model loading |

## Constraint-failure breakdown

How the fleet failed the catalogue constraints — a skew toward one constraint
suggests prompt difficulty rather than individual model faults.

- Duplicate keywords: 4 model(s)

## Crashes requiring action

### apple/FastVLM-7B-int4

- *Execution / usability:* crashed / not assessed
- *Phase:* model_load
- *Stage:* Model Error
- *Resolved revision:* 1aeadbaaba011276f3dcda9582e5e64e2a90873a

Root exception chain

```text
KeyError: 'vision_tower.vision_model.patch_embed.blocks.1.reparam_conv.weight'
caused by: ValueError: Model loading failed: 'vision_tower.vision_model.patch_embed.blocks.1.reparam_conv.weight'
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
python -m mlx_vlm.generate --model apple/FastVLM-7B-int4 --image any-local-image.jpg --prompt x --max-tokens 8 --temperature 0.0 --revision 1aeadbaaba011276f3dcda9582e5e64e2a90873a --trust-remote-code
```

| Evidence | Link |
| --- | --- |
| Full diagnostics | [model evidence](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-apple-fastvlm-7b-int4) |
| Detailed issue draft | [crash draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_apple_FastVLM-7B-int4.md) |

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

### mlx-community/Llama-3.2-11B-Vision-Instruct-4bit

- *Execution / usability:* crashed / not assessed
- *Phase:* generation_before_first_token
- *Stage:* Model Error
- *Resolved revision:* 82f31be9840fa0d4c7e99257fe2e28b59a46df97

Root exception chain

```text
ValueError: [broadcast_shapes] Shapes (1,1,301,6404) and (1,32,300,6404) cannot be broadcast.
caused by: ValueError: Model generation failed for mlx-community/Llama-3.2-11B-Vision-Instruct-4bit: [broadcast_shapes] Shapes (1,1,301,6404) and (1,32,300,6404) cannot be broadcast.
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
| Full diagnostics | [model evidence](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llama-32-11b-vision-instruct-4bit) |
| Detailed issue draft | [crash draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_mlx-community_Llama-3.2-11B-Vision-Instruct-4bit.md) |

### mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

- *Execution / usability:* crashed / not assessed
- *Phase:* generation_before_first_token
- *Stage:* Model Error
- *Resolved revision:* 8451adc50203b50b8f4199e75e753fb9c06e2af6

Root exception chain

```text
ValueError: [broadcast_shapes] Shapes (1,1,301,6404) and (1,32,300,6404) cannot be broadcast.
caused by: ValueError: Model generation failed for mlx-community/Llama-3.2-11B-Vision-Instruct-8bit: [broadcast_shapes] Shapes (1,1,301,6404) and (1,32,300,6404) cannot be broadcast.
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
| Full diagnostics | [model evidence](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llama-32-11b-vision-instruct-8bit) |
| Detailed issue draft | [crash draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_mlx-community_Llama-3.2-11B-Vision-Instruct-8bit.md) |

### mlx-community/Mage-VL-OptiQ-4bit

- *Execution / usability:* crashed / not assessed
- *Phase:* model_load
- *Stage:* Model Error
- *Resolved revision:* bde6c9c7146acff6af09e203245014f19306c5c5

Root exception chain

```text
ValueError: Received 904 parameters not in model; families: model; representative parameters: model.embed_tokens.biases, model.embed_tokens.scales, model.embed_tokens.weight.
caused by: ValueError: Model loading failed: Received 904 parameters not in model; families: model; representative parameters: model.embed_tokens.biases, model.embed_tokens.scales, model.embed_tokens.weight.
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
python -m mlx_vlm.generate --model mlx-community/Mage-VL-OptiQ-4bit --image any-local-image.jpg --prompt x --max-tokens 8 --temperature 0.0 --revision bde6c9c7146acff6af09e203245014f19306c5c5 --trust-remote-code
```

| Evidence | Link |
| --- | --- |
| Full diagnostics | [model evidence](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-mage-vl-optiq-4bit) |
| Detailed issue draft | [crash draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_mlx-community_Mage-VL-OptiQ-4bit.md) |

### mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit

- *Execution / usability:* crashed / not assessed
- *Phase:* prefill
- *Stage:* Error
- *Resolved revision:* 2a1d5eabfc504747bdc24178394821a1efc0edde

Root exception chain

```text
TypeError: can only concatenate str (not "list") to str
caused by: ValueError: Prompt prefill failed for mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit: can only concatenate str (not "list") to str
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
| Full diagnostics | [model evidence](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-mistral-small-32-24b-instruct-2506-4bit) |
| Detailed issue draft | [crash draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_mlx-community_Mistral-Small-3.2-24B-Instruct-2506-4bit.md) |

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
| Response repeats the same text; Generation was stopped early after sustained repeated output; Unrecognised model control tokens remain visible; Required labelled fields not detected | 1 |
| Generation was stopped early after sustained repeated output; Repeated keyword entries | 1 |
| Unrecognised model control tokens remain visible | 1 |
| Unrecognised model control tokens remain visible; Required labelled fields not detected; Response appears cut off at the token limit; Conversation-role control tokens remain visible | 1 |

## Completed attempts requiring review

| Model | Mechanical checks | Observed result | Evidence |
| --- | --- | --- | --- |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns | Response repeats the same text; Generation was stopped early after sustained repeated output; Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description, keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit) |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | major concerns | Generation was stopped early after sustained repeated output; Duplicate keywords: wildlife, animal, bird | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-qwen3-vl-30b-a3b-instruct-4bit) |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | major concerns | Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description; Response appears cut off at the token limit; Conversation-role control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-muse-glimmer-30b-optiq-4bit) |
| mlx-community/aya-vision-8b-4bit | concerns detected | Unrecognised model control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-aya-vision-8b-4bit) |

## Completions without detected concerns

32 completions without detected concerns (`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`, `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit`, `mlx-community/GLM-4.6V-Flash-4bit`, `mlx-community/GLM-4.6V-nvfp4`, `mlx-community/Idefics3-8B-Llama3-bf16`, `mlx-community/InternVL3-14B-4bit`, `mlx-community/InternVL3-8B-bf16`, `mlx-community/LFM2.5-VL-3B-OptiQ-4bit`, `mlx-community/MiniCPM-o-4_5-4bit`, `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`, `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`, `mlx-community/Ministral-3-3B-Instruct-2512-4bit`, `mlx-community/North-Micro-Vision-Instruct-4bit`, `mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit`, `mlx-community/Phi-3.5-vision-instruct-bf16`, `mlx-community/Qwen2-VL-7B-Instruct-4bit`, `mlx-community/Qwen3-VL-2B-Thinking-bf16`, `mlx-community/Qwen3-VL-32B-Instruct-4bit`, `mlx-community/Qwen3-VL-8B-Instruct-4bit`, `mlx-community/Qwen3.5-35B-A3B-4bit`, `mlx-community/Qwen3.5-9B-MLX-4bit`, `mlx-community/Qwen3.8-27B-4bit`, `mlx-community/SmolVLM2-2.2B-Instruct-mlx`, `mlx-community/Step-3.7-Flash-oQ3e`, `mlx-community/X-Reasoner-7B-8bit`, `mlx-community/gemma-3-27b-it-qat-4bit`, `mlx-community/gemma-4-12B-it-4bit`, `mlx-community/gemma-4-26b-a4b-it-4bit`, `mlx-community/gemma-4-31b-it-4bit`, `mlx-community/gemma-4-e4b-it-4bit`, `mlx-community/granite-4.0-3b-vision-4bit`, `mlx-community/pixtral-12b-8bit`); 8 more completed with prompt-compliance observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 9,984 x 6,656 pixels, 44,654,464 bytes
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: seed:* 0
- *Trust remote code:* true
- *check_models version:* 0.17.25
- *check_models revision:* 11e6c82dc61f08eb3dc36a3188a789da1295093b
- *check_models source dirty:* false
- *mlx-vlm:* 0.7.0
- *mlx-vlm source revision:* 45d6e125ab174cc279edea417f6be734870ff161
- *mlx:* 0.32.3.dev20260912+229f5b430
- *mlx source revision:* 229f5b430df7926743c5b6ac62068cae2ebc8978
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
