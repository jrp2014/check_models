# mlx-vlm compatibility findings across 41 cached vision-language models

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

- *Run started:* 2026-09-13 02:12:19 BST
- *Run finished:* 2026-09-13 02:35:20 BST
- *Run duration:* 22m 59s
- *Evaluation lane:* assisted
- *Prompt hints:* the image's description and keyword hints were included in
  the prompt, so field content may be copied from them rather than seen
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Input image:* JPEG, 9,984 x 6,656 pixels (66.5 MP), 44.7 MB
- *Models attempted:* 41
- *Sampling settings:* checkpoint generation_config.json values for 19 of 38
  completed models where the command line left them unset; harness defaults
  elsewhere
- *Completed:* 38
- *Crashed:* 3
- *Indeterminate:* 0
- *Crashes requiring action:* 3
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

- *Baseline:* 09b5430f:src/output/results.jsonl
- *Baseline run timestamp:* 2026-09-13 00:31:29 BST
- *Baseline check_models:* 0.17.24 @ ad84f2cf2
- *Baseline mlx:* 0.32.3.dev20260912+229f5b430 @ 229f5b430
- *Baseline mlx-vlm:* 0.7.0 @ b5379f697
- *Baseline transformers:* 5.17.0
- *Baseline python:* 3.14.7
- *Baseline hardware:* Apple M5 Max, 40 GPU cores, 128.0 GB RAM
- *Models compared:* 36
- *Identical generated text:* 35 of 35 completed in both
- *Generation tok/s ratio (now/baseline):* 1.012 (range 0.51-1.19, 33 models)
- *Throughput noise band:* fixed ±15% fallback (insufficient history)

- New this run (no baseline): `mlx-community/InternVL3_5-30B-A3B-4bit`,
  `mlx-community/Llama-3.2-11B-Vision-Instruct-4bit`,
  `mlx-community/Qwen3-VL-8B-Instruct-4bit`,
  `mlx-community/aya-vision-8b-4bit`, `mlx-community/gemma-3n-E4B-it-4bit`
- In baseline, not run this time: `mlx-community/LFM2.5-VL-1.6B-bf16`,
  `mlx-community/MiniCPM-V-4.6-4bit`

No execution, usability, or observation-set changes against the baseline.

| Model | Baseline tok/s | Now tok/s | Ratio | Expected band |
| --- | --- | --- | --- | --- |
| mlx-community/North-Micro-Vision-Instruct-4bit | 137.0 | 159.2 | 1.16 | 116.4-157.5 (fallback) |
| mlx-community/Phi-3.5-vision-instruct-bf16 | 55.1 | 36.6 | 0.66 | 46.8-63.3 (fallback) |
| mlx-community/gemma-4-31b-it-4bit | 21.8 | 26.0 | 1.19 | 18.5-25.0 (fallback) |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4 | 55.6 | 64.1 | 1.15 | 47.2-63.9 (fallback) |
| mlx-community/gemma-3-27b-it-qat-4bit | 25.8 | 30.2 | 1.17 | 21.9-29.7 (fallback) |
| mlx-community/Qwen3-VL-2B-Thinking-bf16 | 84.8 | 69.8 | 0.82 | 72.1-97.5 (fallback) |
| mlx-community/Qwen3.8-27B-4bit | 28.1 | 14.4 | 0.51 | 23.9-32.4 (fallback) |

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
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | no concerns detected | 11.48s | 29.7 tok/s | 23 | none |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | no concerns detected | 17.04s | 74.0 tok/s | 19 | none |
| mlx-community/gemma-3-27b-it-qat-4bit | no concerns detected | 9.80s | 30.2 tok/s | 17 | none |
| mlx-community/gemma-4-26b-a4b-it-4bit | no concerns detected | 5.32s | 105 tok/s | 16 | none |
| mlx-community/gemma-4-31b-it-4bit | no concerns detected | 8.48s | 26.0 tok/s | 20 | none |
| mlx-community/gemma-4-e4b-it-4bit | no concerns detected | 3.73s | 122 tok/s | 5.9 | none |
| mlx-community/GLM-4.6V-Flash-4bit | no concerns detected | 9.73s | 75.2 tok/s | 8.7 | none |
| mlx-community/GLM-4.6V-nvfp4 | no concerns detected | 28.26s | 40.9 tok/s | 78 | none |
| mlx-community/granite-4.0-3b-vision-4bit | no concerns detected | 3.01s | 172 tok/s | 4.7 | none |
| mlx-community/Idefics3-8B-Llama3-bf16 | no concerns detected | 8.96s | 32.6 tok/s | 18 | none |
| mlx-community/InternVL3-8B-bf16 | no concerns detected | 6.00s | 34.9 tok/s | 17 | none |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit | no concerns detected | 3.11s | 205 tok/s | 4.0 | none |
| mlx-community/MiniCPM-o-4_5-4bit | no concerns detected | 3.27s | 103 tok/s | 7.0 | none |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | no concerns detected | 6.31s | 67.1 tok/s | 13 | none |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4 | no concerns detected | 7.69s | 64.1 tok/s | 13 | none |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | no concerns detected | 3.78s | 190 tok/s | 7.8 | none |
| mlx-community/North-Micro-Vision-Instruct-4bit | no concerns detected | 6.02s | 159 tok/s | 3.9 | none |
| mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit | no concerns detected | 6.69s | 73.0 tok/s | 24 | none |
| mlx-community/Phi-3.5-vision-instruct-bf16 | no concerns detected | 5.24s | 36.6 tok/s | 9.3 | none |
| mlx-community/pixtral-12b-8bit | no concerns detected | 7.31s | 37.7 tok/s | 16 | none |
| mlx-community/Qwen3-VL-2B-Thinking-bf16 | no concerns detected | 35.94s | 69.8 tok/s | 8.4 | none |
| mlx-community/Qwen3-VL-8B-Instruct-4bit | no concerns detected | 48.18s | 59.4 tok/s | 11 | none |
| mlx-community/Qwen3.5-35B-A3B-4bit | no concerns detected | 46.02s | 74.4 tok/s | 25 | none |
| mlx-community/Qwen3.5-9B-MLX-4bit | no concerns detected | 46.37s | 88.2 tok/s | 11 | none |
| mlx-community/Qwen3.8-27B-4bit | no concerns detected | 76.80s | 14.4 tok/s | 21 | none |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | no concerns detected | 3.74s | 123 tok/s | 5.6 | none |
| mlx-community/Step-3.7-Flash-oQ3e | no concerns detected | 100.61s | 48.6 tok/s | 92 | none |
| mlx-community/X-Reasoner-7B-8bit | no concerns detected | 17.88s | 57.2 tok/s | 14 | none |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | concerns detected | 2.17s | 475 tok/s | 1.9 | duplicate keywords |
| mlx-community/aya-vision-8b-4bit | concerns detected | 5.14s | 102 tok/s | 6.5 | control tokens visible |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | concerns detected | 6.71s | 53.0 tok/s | 28 | duplicate keywords |
| mlx-community/Molmo2-8B-4bit | concerns detected | 6.24s | 72.1 tok/s | 8.1 | duplicate keywords |
| mlx-community/gemma-3n-E4B-it-4bit | major concerns | 5.22s | 60.3 tok/s | 7.1 | labelled fields not detected |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit | major concerns | 20.55s | 61.5 tok/s | 20 | cut off at token limit |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns | 5.24s | 100 tok/s | 6.7 | repeated text; stopped early: repeating; control tokens visible; labelled fields not detected |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | major concerns | 59.35s | 21.6 tok/s | 25 | control tokens visible; labelled fields not detected; cut off at token limit; role tokens visible |
| mlx-community/nanoLLaVA-1.5-4bit | major concerns | 2.29s | 204 tok/s | 1.8 | labelled fields not detected |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | major concerns | 46.20s | 71.2 tok/s | 23 | stopped early: repeating; duplicate keywords |
| mlx-community/InternVL3_5-30B-A3B-4bit | not assessed | 0.16s | - | - | crashed during model loading |
| mlx-community/Llama-3.2-11B-Vision-Instruct-4bit | not assessed | 2.54s | - | - | crashed during generation, before first token |
| mlx-community/Mage-VL-OptiQ-4bit | not assessed | 0.19s | - | - | crashed during model loading |

## Constraint-failure breakdown

How the fleet failed the catalogue constraints — a skew toward one constraint
suggests prompt difficulty rather than individual model faults.

- Duplicate keywords: 4 model(s)

## Crashes requiring action

### mlx-community/InternVL3_5-30B-A3B-4bit

- *Execution / usability:* crashed / not assessed
- *Phase:* model_load
- *Stage:* Unsupported Arch
- *Resolved revision:* ed2ce3381528db1c5b70a2aad78a6390997e9250

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
python -m mlx_vlm.generate --model mlx-community/InternVL3_5-30B-A3B-4bit --image any-local-image.jpg --prompt x --max-tokens 8 --temperature 0.0 --revision ed2ce3381528db1c5b70a2aad78a6390997e9250 --trust-remote-code
```

| Evidence | Link |
| --- | --- |
| Full diagnostics | [model evidence](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-internvl35-30b-a3b-4bit) |
| Detailed issue draft | [crash draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_mlx-community_InternVL3_5-30B-A3B-4bit.md) |

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

28 completions without detected concerns (`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`, `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit`, `mlx-community/GLM-4.6V-Flash-4bit`, `mlx-community/GLM-4.6V-nvfp4`, `mlx-community/Idefics3-8B-Llama3-bf16`, `mlx-community/InternVL3-8B-bf16`, `mlx-community/LFM2.5-VL-3B-OptiQ-4bit`, `mlx-community/MiniCPM-o-4_5-4bit`, `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`, `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`, `mlx-community/Ministral-3-3B-Instruct-2512-4bit`, `mlx-community/North-Micro-Vision-Instruct-4bit`, `mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit`, `mlx-community/Phi-3.5-vision-instruct-bf16`, `mlx-community/Qwen3-VL-2B-Thinking-bf16`, `mlx-community/Qwen3-VL-8B-Instruct-4bit`, `mlx-community/Qwen3.5-35B-A3B-4bit`, `mlx-community/Qwen3.5-9B-MLX-4bit`, `mlx-community/Qwen3.8-27B-4bit`, `mlx-community/SmolVLM2-2.2B-Instruct-mlx`, `mlx-community/Step-3.7-Flash-oQ3e`, `mlx-community/X-Reasoner-7B-8bit`, `mlx-community/gemma-3-27b-it-qat-4bit`, `mlx-community/gemma-4-26b-a4b-it-4bit`, `mlx-community/gemma-4-31b-it-4bit`, `mlx-community/gemma-4-e4b-it-4bit`, `mlx-community/granite-4.0-3b-vision-4bit`, `mlx-community/pixtral-12b-8bit`); 6 more completed with prompt-compliance observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 9,984 x 6,656 pixels, 44,654,464 bytes
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: seed:* 0
- *Trust remote code:* true
- *check_models version:* 0.17.24
- *check_models revision:* 09b5430fd4adb1ca5f371bf11544425753636488
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
