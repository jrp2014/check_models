# mlx-vlm compatibility findings across 50 cached vision-language models

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

- *Run started:* 2026-09-26 23:17:17 BST
- *Run finished:* 2026-09-26 23:32:45 BST
- *Run duration:* 15m 26s
- *Time by phase:* generation 757s, model load 110s, prompt prep 41s, cleanup
  7s, outside the model loop 18s
- *Evaluation lane:* assisted
- *Prompt hints:* the image's description and keyword hints were included in
  the prompt, so field content may be copied from them rather than seen
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Input image:* JPEG, 9,984 x 6,656 pixels (66.5 MP), 49.4 MB
- *Models attempted:* 50
- *Sampling settings:* checkpoint generation_config.json values for 24 of 49
  completed models where the command line left them unset; harness defaults
  elsewhere
- *Completed:* 49
- *Crashed:* 1
- *Indeterminate:* 0
- *Crashes requiring action:* 1
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

- *Baseline:* 6cd0a83e:src/output/results.jsonl
- *Baseline run timestamp:* 2026-09-26 00:05:50 BST
- *Baseline check_models:* 0.17.38 @ 29bae6c86
- *Baseline mlx:* 0.32.3.dev20260925+073d2252c @ 073d2252c
- *Baseline mlx-vlm:* 0.7.3 @ 990a02876
- *Baseline transformers:* 5.17.0
- *Baseline python:* 3.14.7
- *Baseline hardware:* Apple M5 Max, 40 GPU cores, 128.0 GB RAM
- *Models compared:* 47
- *Identical generated text:* 45 of 46 completed in both
- *Generated text changed:* mlx-community/Muse-Glimmer-30B-OptiQ-4bit
- *Text changed, by decoding:* greedy 0 of 26; sampled, same settings and seed
  1 of 20
- *Generation tok/s ratio (now/baseline):* 0.989 (range 0.67-1.12, 42 models)
- *Prefill tok/s ratio (now/baseline):* 0.715 (range 0.11-1.09, 46 models)
- *Throughput noise band:* fixed ±15% fallback (insufficient history)

- New this run (no baseline): `nativ-community/Mage-VL-OptiQ-4bit`,
  `nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit`,
  `nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit`
- In baseline, no longer in the cache: `mlx-community/Mage-VL-OptiQ-4bit`

No execution, usability, or observation-set changes against the baseline.

| Model | Baseline tok/s | Now tok/s | Ratio | Expected band |
| --- | --- | --- | --- | --- |
| mlx-community/nanoLLaVA-1.5-4bit | 210.2 | 163.5 | 0.78 | 178.7-241.8 (fallback) |
| mlx-community/gemma-4-e4b-it-4bit | 120.8 | 96.4 | 0.80 | 102.7-139.0 (fallback) |
| mlx-community/granite-4.0-3b-vision-4bit | 165.9 | 129.2 | 0.78 | 141.0-190.8 (fallback) |
| mlx-community/gemma-4-26b-a4b-it-4bit | 103.8 | 76.1 | 0.73 | 88.2-119.4 (fallback) |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | 46.5 | 39.2 | 0.84 | 39.6-53.5 (fallback) |
| mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit | 73.1 | 60.2 | 0.82 | 62.1-84.0 (fallback) |
| mlx-community/Phi-3.5-vision-instruct-bf16 | 55.7 | 37.1 | 0.67 | 47.4-64.1 (fallback) |
| mlx-community/Qwen3-VL-32B-Instruct-4bit | 19.8 | 16.7 | 0.84 | 16.9-22.8 (fallback) |

| Model | Failure continuity |
| --- | --- |
| mlx-community/InternVL3_5-1B-4bit | same failure signature observed |

<details>
<summary>mlx: 3 upstream commit(s) since the baseline (073d2252c..a2a09fd56)</summary>

- a2a09fd56 [Metal] Gather mm improvement (#4567)
- 465b7a5de Support shapeless compilation of scan operations (#4510)
- c0418f6e2 Improve metal memory usage for SDPA D512 (#4487)

</details>

<details>
<summary>mlx-vlm: 6 upstream commit(s) since the baseline (990a02876..0e0a9d8c8)</summary>

- 0e0a9d8c docs: link corrected multimodal checkpoints (#2372)
- 0824cf45 tests: remove restored Mistral test class (#2371)
- f083061d Cleanup/moondream qwen test ownership (#2366)
- 95b01cca Refactor/gliner model scope (#2363)
- ca6ace59 Cleanup/model test ownership (#2364)
- 250afd92 Feat/lfm2 colbert (#2360)

</details>

<details>
<summary>mlx-vlm commits touching the affected models' architectures</summary>

- `mlx-community/InternVL3_5-1B-4bit` (`internvl`): no commits touched
  `mlx_vlm/models/internvl/`
- `mlx-community/Muse-Glimmer-30B-OptiQ-4bit` (`muse_glimmer`): no commits
  touched `mlx_vlm/models/muse_glimmer/`

Context, not attribution: shared generation, sampling and processor code can
change a model without touching its own package.

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
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | no concerns detected | 2.39s | 475 tok/s | 1.9 | none |
| mlx-community/aya-vision-8b-4bit | no concerns detected | 5.87s | 91.7 tok/s | 6.5 | none |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | no concerns detected | 15.65s | 29.3 tok/s | 23 | none |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | no concerns detected | 8.55s | 39.2 tok/s | 28 | none |
| mlx-community/gemma-3-27b-it-qat-4bit | no concerns detected | 11.33s | 29.4 tok/s | 17 | none |
| mlx-community/gemma-4-26b-a4b-it-4bit | no concerns detected | 6.53s | 76.1 tok/s | 16 | none |
| mlx-community/gemma-4-31b-it-4bit | no concerns detected | 8.31s | 26.8 tok/s | 20 | none |
| mlx-community/gemma-4-e4b-it-4bit | no concerns detected | 4.82s | 96.4 tok/s | 5.9 | none |
| mlx-community/GLM-4.6V-Flash-4bit | no concerns detected | 10.58s | 73.6 tok/s | 8.7 | none |
| mlx-community/GLM-4.6V-nvfp4 | no concerns detected | 35.98s | 40.2 tok/s | 78 | none |
| mlx-community/granite-4.0-3b-vision-4bit | no concerns detected | 4.57s | 129 tok/s | 4.7 | none |
| mlx-community/InternVL3-14B-4bit | no concerns detected | 7.52s | 56.3 tok/s | 10 | none |
| mlx-community/InternVL3-8B-bf16 | no concerns detected | 6.81s | 36.6 tok/s | 17 | none |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit | no concerns detected | 18.59s | 60.7 tok/s | 20 | none |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit | no concerns detected | 3.83s | 206 tok/s | 4.0 | none |
| mlx-community/MiniCPM-o-4_5-4bit | no concerns detected | 3.62s | 104 tok/s | 7.0 | none |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | no concerns detected | 7.30s | 65.5 tok/s | 13 | none |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | no concerns detected | 4.09s | 188 tok/s | 7.8 | none |
| mlx-community/North-Micro-Vision-Instruct-4bit | no concerns detected | 5.30s | 165 tok/s | 3.9 | none |
| mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit | no concerns detected | 7.59s | 60.2 tok/s | 24 | none |
| mlx-community/Phi-3.5-vision-instruct-bf16 | no concerns detected | 6.37s | 37.1 tok/s | 9.3 | none |
| mlx-community/pixtral-12b-8bit | no concerns detected | 7.85s | 37.2 tok/s | 16 | none |
| mlx-community/Qwen3-VL-2B-Thinking-bf16 | no concerns detected | 28.55s | 87.3 tok/s | 8.4 | none |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | no concerns detected | 38.61s | 75.2 tok/s | 23 | none |
| mlx-community/Qwen3-VL-32B-Instruct-4bit | no concerns detected | 73.38s | 16.7 tok/s | 26 | none |
| mlx-community/Qwen3-VL-8B-Instruct-4bit | no concerns detected | 43.38s | 67.6 tok/s | 11 | none |
| mlx-community/Qwen3.5-35B-A3B-4bit | no concerns detected | 40.68s | 73.5 tok/s | 25 | none |
| mlx-community/Qwen3.5-9B-MLX-4bit | no concerns detected | 39.83s | 84.5 tok/s | 11 | none |
| mlx-community/Qwen3.8-27B-nvfp4 | no concerns detected | 66.31s | 28.2 tok/s | 21 | none |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | no concerns detected | 3.41s | 125 tok/s | 5.6 | none |
| mlx-community/Step-3.7-Flash-oQ3e | no concerns detected | 106.26s | 44.0 tok/s | 92 | none |
| nativ-community/Mage-VL-OptiQ-4bit | no concerns detected | 4.88s | 124 tok/s | 5.4 | none |
| nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit | no concerns detected | 8.66s | 35.0 tok/s | 18 | none |
| nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | no concerns detected | 10.71s | 91.5 tok/s | 23 | none |
| mlx-community/gemma-4-12B-it-4bit | concerns detected | 6.20s | 58.1 tok/s | 7.6 | duplicate keywords |
| mlx-community/Idefics3-8B-Llama3-bf16 | concerns detected | 10.30s | 34.4 tok/s | 18 | duplicate keywords |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | major concerns | 9.03s | 95.2 tok/s | 19 | stopped early: repeating; labelled fields not detected; incomplete thinking block |
| mlx-community/FastVLM-0.5B-bf16 | major concerns | 4.13s | 311 tok/s | 1.8 | labelled fields not detected |
| mlx-community/gemma-3n-E4B-it-4bit | major concerns | 8.06s | 60.6 tok/s | 6.9 | labelled fields not detected |
| mlx-community/granite-vision-3.2-2b-nvfp4 | major concerns | 4.78s | 141 tok/s | 4.2 | labelled fields not detected |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | major concerns | 58.73s | 18.7 tok/s | 15 | repeated text; cut off at token limit; duplicate keywords |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns | 3.45s | 111 tok/s | 6.7 | control tokens visible; labelled fields not detected |
| mlx-community/MiniCPM-V-4.6-4bit | major concerns | 4.73s | 247 tok/s | 3.3 | incomplete thinking block |
| mlx-community/Molmo2-8B-4bit | major concerns | 8.66s | 70.3 tok/s | 8.1 | stopped early: repeating; duplicate keywords |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | major concerns | 63.41s | 20.7 tok/s | 25 | control tokens visible; labelled fields not detected; cut off at token limit; role tokens visible |
| mlx-community/nanoLLaVA-1.5-4bit | major concerns | 2.41s | 164 tok/s | 1.4 | labelled fields not detected |
| mlx-community/Qwen2-VL-7B-Instruct-4bit | major concerns | 43.45s | 89.5 tok/s | 9.3 | repeated text; stopped early: repeating; duplicate keywords |
| mlx-community/SmolVLM-256M-Instruct-4bit | major concerns | 2.63s | 322 tok/s | 1.1 | labelled fields not detected |
| mlx-community/X-Reasoner-7B-8bit | major concerns | 20.22s | 56.2 tok/s | 14 | stopped early: repeating; duplicate keywords |
| mlx-community/InternVL3_5-1B-4bit | not assessed | 0.13s | - | - | crashed during model loading |

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

34 completions without detected concerns (`LiquidAI/LFM2.5-VL-450M-MLX-bf16`, `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`, `mlx-community/GLM-4.6V-Flash-4bit`, `mlx-community/GLM-4.6V-nvfp4`, `mlx-community/InternVL3-14B-4bit`, `mlx-community/InternVL3-8B-bf16`, `mlx-community/Kimi-VL-A3B-Thinking-2506-8bit`, `mlx-community/LFM2.5-VL-3B-OptiQ-4bit`, `mlx-community/MiniCPM-o-4_5-4bit`, `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`, `mlx-community/Ministral-3-3B-Instruct-2512-4bit`, `mlx-community/North-Micro-Vision-Instruct-4bit`, `mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit`, `mlx-community/Phi-3.5-vision-instruct-bf16`, `mlx-community/Qwen3-VL-2B-Thinking-bf16`, `mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit`, `mlx-community/Qwen3-VL-32B-Instruct-4bit`, `mlx-community/Qwen3-VL-8B-Instruct-4bit`, `mlx-community/Qwen3.5-35B-A3B-4bit`, `mlx-community/Qwen3.5-9B-MLX-4bit`, `mlx-community/Qwen3.8-27B-nvfp4`, `mlx-community/SmolVLM2-2.2B-Instruct-mlx`, `mlx-community/Step-3.7-Flash-oQ3e`, `mlx-community/aya-vision-8b-4bit`, `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`, `mlx-community/gemma-3-27b-it-qat-4bit`, `mlx-community/gemma-4-26b-a4b-it-4bit`, `mlx-community/gemma-4-31b-it-4bit`, `mlx-community/gemma-4-e4b-it-4bit`, `mlx-community/granite-4.0-3b-vision-4bit`, `mlx-community/pixtral-12b-8bit`, `nativ-community/Mage-VL-OptiQ-4bit`, `nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit`, `nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit`); 7 more completed with prompt-compliance observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 9,984 x 6,656 pixels, 49,407,373 bytes
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: seed:* 0
- *Trust remote code:* true
- *check_models version:* 0.17.38
- *check_models revision:* 6cd0a83e7afcc0892806ff31d87d38973d8a0a8c
- *check_models source dirty:* false
- *mlx-vlm:* 0.7.3
- *mlx-vlm source revision:* 0e0a9d8c81282a100fec028474a7a688551f5e78
- *mlx:* 0.32.3.dev20260926+a2a09fd56
- *mlx source revision:* a2a09fd56
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
