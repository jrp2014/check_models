# mlx-vlm compatibility findings across 48 cached vision-language models

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

- *Run started:* 2026-09-25 23:52:46 BST
- *Run finished:* 2026-09-26 00:05:48 BST
- *Run duration:* 13m 00s
- *Time by phase:* generation 640s, model load 88s, prompt prep 36s, cleanup
  6s, outside the model loop 16s
- *Evaluation lane:* assisted
- *Prompt hints:* the image's description and keyword hints were included in
  the prompt, so field content may be copied from them rather than seen
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Input image:* JPEG, 9,984 x 6,656 pixels (66.5 MP), 49.4 MB
- *Models attempted:* 48
- *Sampling settings:* checkpoint generation_config.json values for 22 of 46
  completed models where the command line left them unset; harness defaults
  elsewhere
- *Completed:* 46
- *Crashed:* 2
- *Indeterminate:* 0
- *Crashes requiring action:* 2
- *Other results requiring review:* 8
- *Reached token limit:* 2
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
- Capture date/time: 2026-09-19 17:12:46 UTC+01:00

Descriptive hints:
- Description hint: Two sailors steer small dinghies—a Vortex catamaran (sail number 1067) on the left and a Laser dinghy (sail number GBR 188572) on the right—across calm coastal or river waters against a backdrop of dense green woodland.
- Keyword hints: Boat, Boating, Catamaran, Clouds, Dinghy, Estuary, Forest, Laser dinghy, Life jacket, Man, Mast, Outdoor recreation, River, Sailboat, Sailing, Sailor, Shoreline, Sky, Trees, Water

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

- *Baseline:* 29bae6c8:src/output/results.jsonl
- *Baseline run timestamp:* 2026-09-25 23:40:47 BST
- *Baseline check_models:* 0.17.38 @ 691db9dc9
- *Baseline mlx:* 0.32.3.dev20260925+073d2252c @ 073d2252c
- *Baseline mlx-vlm:* 0.7.3 @ 990a02876
- *Baseline transformers:* 5.17.0
- *Baseline python:* 3.14.7
- *Baseline hardware:* Apple M5 Max, 40 GPU cores, 128.0 GB RAM
- *Models compared:* 48
- *Identical generated text:* 46 of 46 completed in both
- *Text changed, by decoding:* greedy 0 of 26; sampled, same settings and seed
  0 of 20
- *Generation tok/s ratio (now/baseline):* 0.979 (range 0.89-1.08, 42 models)
- *Prefill tok/s ratio (now/baseline):* 0.992 (range 0.82-1.17, 46 models)
- *Throughput noise band:* fixed ±15% fallback (insufficient history)

No execution, usability, or observation-set changes against the baseline.

| Model | Failure continuity |
| --- | --- |
| mlx-community/InternVL3_5-1B-4bit | same failure signature observed |
| mlx-community/Mage-VL-OptiQ-4bit | same failure signature observed |

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
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | no concerns detected | 2.24s | 480 tok/s | 1.9 | none |
| mlx-community/aya-vision-8b-4bit | no concerns detected | 4.93s | 98.5 tok/s | 6.5 | none |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | no concerns detected | 10.46s | 30.0 tok/s | 23 | none |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | no concerns detected | 7.13s | 46.5 tok/s | 28 | none |
| mlx-community/gemma-3-27b-it-qat-4bit | no concerns detected | 10.33s | 30.4 tok/s | 17 | none |
| mlx-community/gemma-4-26b-a4b-it-4bit | no concerns detected | 5.44s | 104 tok/s | 16 | none |
| mlx-community/gemma-4-31b-it-4bit | no concerns detected | 8.45s | 26.3 tok/s | 20 | none |
| mlx-community/gemma-4-e4b-it-4bit | no concerns detected | 4.07s | 121 tok/s | 5.9 | none |
| mlx-community/GLM-4.6V-Flash-4bit | no concerns detected | 9.69s | 73.1 tok/s | 8.7 | none |
| mlx-community/GLM-4.6V-nvfp4 | no concerns detected | 23.42s | 39.5 tok/s | 78 | none |
| mlx-community/granite-4.0-3b-vision-4bit | no concerns detected | 3.27s | 166 tok/s | 4.7 | none |
| mlx-community/InternVL3-14B-4bit | no concerns detected | 6.17s | 55.4 tok/s | 10 | none |
| mlx-community/InternVL3-8B-bf16 | no concerns detected | 6.77s | 36.8 tok/s | 17 | none |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit | no concerns detected | 16.34s | 61.5 tok/s | 20 | none |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit | no concerns detected | 3.58s | 209 tok/s | 4.0 | none |
| mlx-community/MiniCPM-o-4_5-4bit | no concerns detected | 3.45s | 104 tok/s | 7.0 | none |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | no concerns detected | 7.37s | 66.3 tok/s | 13 | none |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | no concerns detected | 4.14s | 186 tok/s | 7.8 | none |
| mlx-community/North-Micro-Vision-Instruct-4bit | no concerns detected | 5.40s | 147 tok/s | 3.9 | none |
| mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit | no concerns detected | 6.70s | 73.1 tok/s | 24 | none |
| mlx-community/Phi-3.5-vision-instruct-bf16 | no concerns detected | 4.76s | 55.7 tok/s | 9.3 | none |
| mlx-community/pixtral-12b-8bit | no concerns detected | 7.59s | 39.5 tok/s | 16 | none |
| mlx-community/Qwen3-VL-2B-Thinking-bf16 | no concerns detected | 30.87s | 78.2 tok/s | 8.4 | none |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | no concerns detected | 41.84s | 74.8 tok/s | 23 | none |
| mlx-community/Qwen3-VL-32B-Instruct-4bit | no concerns detected | 75.46s | 19.8 tok/s | 26 | none |
| mlx-community/Qwen3-VL-8B-Instruct-4bit | no concerns detected | 39.81s | 67.4 tok/s | 11 | none |
| mlx-community/Qwen3.5-35B-A3B-4bit | no concerns detected | 39.29s | 68.6 tok/s | 25 | none |
| mlx-community/Qwen3.5-9B-MLX-4bit | no concerns detected | 40.84s | 88.1 tok/s | 11 | none |
| mlx-community/Qwen3.8-27B-nvfp4 | no concerns detected | 60.79s | 28.0 tok/s | 21 | none |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | no concerns detected | 3.22s | 124 tok/s | 5.6 | none |
| mlx-community/Step-3.7-Flash-oQ3e | no concerns detected | 35.27s | 46.4 tok/s | 92 | none |
| mlx-community/gemma-4-12B-it-4bit | concerns detected | 5.27s | 60.8 tok/s | 7.6 | duplicate keywords |
| mlx-community/Idefics3-8B-Llama3-bf16 | concerns detected | 9.75s | 33.4 tok/s | 18 | duplicate keywords |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | major concerns | 7.88s | 99.0 tok/s | 19 | stopped early: repeating; labelled fields not detected; incomplete thinking block |
| mlx-community/FastVLM-0.5B-bf16 | major concerns | 3.13s | 308 tok/s | 2.2 | labelled fields not detected |
| mlx-community/gemma-3n-E4B-it-4bit | major concerns | 6.01s | 69.8 tok/s | 7.2 | labelled fields not detected |
| mlx-community/granite-vision-3.2-2b-nvfp4 | major concerns | 4.57s | 146 tok/s | 4.2 | labelled fields not detected |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | major concerns | 57.91s | 18.7 tok/s | 15 | repeated text; cut off at token limit; duplicate keywords |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns | 3.52s | 107 tok/s | 6.7 | control tokens visible; labelled fields not detected |
| mlx-community/MiniCPM-V-4.6-4bit | major concerns | 2.94s | 222 tok/s | 3.3 | incomplete thinking block |
| mlx-community/Molmo2-8B-4bit | major concerns | 6.42s | 72.2 tok/s | 8.2 | stopped early: repeating; duplicate keywords |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | major concerns | 55.42s | 23.9 tok/s | 25 | control tokens visible; labelled fields not detected; cut off at token limit; role tokens visible |
| mlx-community/nanoLLaVA-1.5-4bit | major concerns | 1.99s | 210 tok/s | 1.8 | labelled fields not detected |
| mlx-community/Qwen2-VL-7B-Instruct-4bit | major concerns | 43.97s | 88.4 tok/s | 9.3 | repeated text; stopped early: repeating; duplicate keywords |
| mlx-community/SmolVLM-256M-Instruct-4bit | major concerns | 2.35s | 303 tok/s | 1.1 | labelled fields not detected |
| mlx-community/X-Reasoner-7B-8bit | major concerns | 20.97s | 56.2 tok/s | 14 | stopped early: repeating; duplicate keywords |
| mlx-community/InternVL3_5-1B-4bit | not assessed | 0.17s | - | - | crashed during model loading |
| mlx-community/Mage-VL-OptiQ-4bit | not assessed | 3.28s | - | - | crashed during generation, before first token |

## Constraint-failure breakdown

How the fleet failed the catalogue constraints — a skew toward one constraint
suggests prompt difficulty rather than individual model faults.

- Duplicate keywords: 6 model(s)

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
- *Image size:* 49,407,373 bytes
- *Image SHA-256:* 97f53d5eeb6a63e6e685321bc95e66807a87db2e01f63f48a48a99f9342c7d12

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-09-19 17:12:46 UTC+01:00

Descriptive hints:
- Description hint: Two sailors steer small dinghies—a Vortex catamaran (sail number 1067) on the left and a Laser dinghy (sail number GBR 188572) on the right—across calm coastal or river waters against a backdrop of dense green woodland.
- Keyword hints: Boat, Boating, Catamaran, Clouds, Dinghy, Estuary, Forest, Laser dinghy, Life jacket, Man, Mast, Outdoor recreation, River, Sailboat, Sailing, Sailor, Shoreline, Sky, Trees, Water

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
- *Image size:* 49,407,373 bytes
- *Image SHA-256:* 97f53d5eeb6a63e6e685321bc95e66807a87db2e01f63f48a48a99f9342c7d12

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-09-19 17:12:46 UTC+01:00

Descriptive hints:
- Description hint: Two sailors steer small dinghies—a Vortex catamaran (sail number 1067) on the left and a Laser dinghy (sail number GBR 188572) on the right—across calm coastal or river waters against a backdrop of dense green woodland.
- Keyword hints: Boat, Boating, Catamaran, Clouds, Dinghy, Estuary, Forest, Laser dinghy, Life jacket, Man, Mast, Outdoor recreation, River, Sailboat, Sailing, Sailor, Shoreline, Sky, Trees, Water

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

## Observation clusters

Repeated mechanical observation signatures among results requiring review.

| Observed result | Models |
| --- | --- |
| Response repeats the same text; Generation was stopped early after sustained repeated output; Repeated keyword entries | 1 |
| Response repeats the same text; Response appears cut off at the token limit; Repeated keyword entries | 1 |
| Generation was stopped early after sustained repeated output; Repeated keyword entries | 2 |
| Generation was stopped early after sustained repeated output; Required labelled fields not detected; Internal reasoning block appears incomplete | 1 |
| Unrecognised model control tokens remain visible; Required labelled fields not detected | 1 |
| Unrecognised model control tokens remain visible; Required labelled fields not detected; Response appears cut off at the token limit; Conversation-role control tokens remain visible | 1 |
| Internal reasoning block appears incomplete | 1 |

## Completed attempts requiring review

| Model | Mechanical checks | Observed result | Evidence |
| --- | --- | --- | --- |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | major concerns | Response repeats the same text; Response appears cut off at the token limit; Duplicate keywords: adventure, learning, education, training, practice, improvement, progress, success, achievement, accomplishment, pride, satisfaction, happiness, joy, laughter, smiles, gratitude, appreciation, wonder, awe, amazement, enthusiasm, excitement, exploration, discovery | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llama-32-11b-vision-instruct-8bit) |
| mlx-community/Qwen2-VL-7B-Instruct-4bit | major concerns | Response repeats the same text; Generation was stopped early after sustained repeated output; Duplicate keywords: trees, forest, mast, life jacket, river, estuary, sky, shoreline, sail, boat, boating | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-qwen2-vl-7b-instruct-4bit) |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | major concerns | Generation was stopped early after sustained repeated output; Required labelled fields not detected: title, description, keywords; Internal reasoning block appears incomplete | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-ernie-45-vl-28b-a3b-thinking-4bit) |
| mlx-community/Molmo2-8B-4bit | major concerns | Generation was stopped early after sustained repeated output; Duplicate keywords: blue stripes, white boat, blue canopy, white hull | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-molmo2-8b-4bit) |
| mlx-community/X-Reasoner-7B-8bit | major concerns | Generation was stopped early after sustained repeated output; Duplicate keywords: blue and white forest, blue and white sky, blue and white water, blue and white sail | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-x-reasoner-7b-8bit) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns | Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description, keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit) |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | major concerns | Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description; Response appears cut off at the token limit; Conversation-role control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-muse-glimmer-30b-optiq-4bit) |
| mlx-community/MiniCPM-V-4.6-4bit | major concerns | Internal reasoning block appears incomplete | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-minicpm-v-46-4bit) |

## Completions without detected concerns

31 completions without detected concerns (`LiquidAI/LFM2.5-VL-450M-MLX-bf16`, `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`, `mlx-community/GLM-4.6V-Flash-4bit`, `mlx-community/GLM-4.6V-nvfp4`, `mlx-community/InternVL3-14B-4bit`, `mlx-community/InternVL3-8B-bf16`, `mlx-community/Kimi-VL-A3B-Thinking-2506-8bit`, `mlx-community/LFM2.5-VL-3B-OptiQ-4bit`, `mlx-community/MiniCPM-o-4_5-4bit`, `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`, `mlx-community/Ministral-3-3B-Instruct-2512-4bit`, `mlx-community/North-Micro-Vision-Instruct-4bit`, `mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit`, `mlx-community/Phi-3.5-vision-instruct-bf16`, `mlx-community/Qwen3-VL-2B-Thinking-bf16`, `mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit`, `mlx-community/Qwen3-VL-32B-Instruct-4bit`, `mlx-community/Qwen3-VL-8B-Instruct-4bit`, `mlx-community/Qwen3.5-35B-A3B-4bit`, `mlx-community/Qwen3.5-9B-MLX-4bit`, `mlx-community/Qwen3.8-27B-nvfp4`, `mlx-community/SmolVLM2-2.2B-Instruct-mlx`, `mlx-community/Step-3.7-Flash-oQ3e`, `mlx-community/aya-vision-8b-4bit`, `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`, `mlx-community/gemma-3-27b-it-qat-4bit`, `mlx-community/gemma-4-26b-a4b-it-4bit`, `mlx-community/gemma-4-31b-it-4bit`, `mlx-community/gemma-4-e4b-it-4bit`, `mlx-community/granite-4.0-3b-vision-4bit`, `mlx-community/pixtral-12b-8bit`); 7 more completed with prompt-compliance observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 9,984 x 6,656 pixels, 49,407,373 bytes
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: seed:* 0
- *Trust remote code:* true
- *check_models version:* 0.17.38
- *check_models revision:* 29bae6c868c9e1e58221eb34da8336fa58581ed1
- *check_models source dirty:* false
- *mlx-vlm:* 0.7.3
- *mlx-vlm source revision:* 990a028763b2f8c605329f29ae4e747f9285902d
- *mlx:* 0.32.3.dev20260925+073d2252c
- *mlx source revision:* 073d2252c
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
