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

- *Run started:* 2026-09-27 00:47:07 BST
- *Run finished:* 2026-09-27 01:01:53 BST
- *Run duration:* 14m 44s
- *Time by phase:* generation 726s, model load 103s, prompt prep 39s, cleanup
  7s, outside the model loop 17s
- *Evaluation lane:* assisted
- *Prompt hints:* the image's description and keyword hints were included in
  the prompt, so field content may be copied from them rather than seen
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Input image:* JPEG, 9,984 x 6,656 pixels (66.5 MP), 49.4 MB
- *Models attempted:* 50
- *Sampling settings:* checkpoint generation_config.json values for 25 of 50
  completed models where the command line left them unset; harness defaults
  elsewhere
- *Completed:* 50
- *Crashed:* 0
- *Indeterminate:* 0
- *Crashes requiring action:* 0
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

- *Baseline:* e7629fd5:src/output/results.jsonl
- *Baseline run timestamp:* 2026-09-26 23:57:05 BST
- *Baseline check_models:* 0.17.38 @ 8f3ccf2be
- *Baseline mlx:* 0.32.3.dev20260926+a2a09fd56 @ a2a09fd56
- *Baseline mlx-vlm:* 0.7.3 @ 3a653e7c6
- *Baseline transformers:* 5.17.0
- *Baseline python:* 3.14.7
- *Baseline hardware:* Apple M5 Max, 40 GPU cores, 128.0 GB RAM
- *Models compared:* 50
- *Identical generated text:* 49 of 50 completed in both
- *Generated text changed:* mlx-community/Muse-Glimmer-30B-OptiQ-4bit
- *Text changed, by decoding:* greedy 0 of 27; sampled, same settings and seed
  1 of 23
- *Generation tok/s ratio (now/baseline):* 0.998 (range 0.89-1.40, 46 models)
- *Prefill tok/s ratio (now/baseline):* 0.986 (range 0.78-1.39, 50 models)
- *Throughput noise band:* history (last 5 same-prompt runs, Tukey fence, at
  least ±10% of the median)

- In baseline, no longer in the cache: `mlx-community/InternVL3_5-1B-4bit`

| Model | Execution | Usability | Observation delta |
| --- | --- | --- | --- |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | completed | major concerns | +unsupplied place name |

| Model | Baseline tok/s | Now tok/s | Ratio | Expected band |
| --- | --- | --- | --- | --- |
| mlx-community/gemma-4-31b-it-4bit | 24.9 | 22.2 | 0.89 | 23.7-29.0 (history, n=5) |

<details>
<summary>mlx-vlm: 13 upstream commit(s) since the baseline (3a653e7c6..e6bf06faa)</summary>

- e6bf06fa Feat/lfm2 encoder (#2359)
- ebc14f13 Merge pull request #2365 from pierre427/fix/tool-stream-parity
- 70f316e9 Add unfinished call cleanup for /v1/messages.
- 82bdbc5e Merge pull request #2373 from lucasnewman/image-sampling-defaults
- 0dbc3176 Preserve literal tool markup in Anthropic streams without tools
- d4ff8bdd Merge branch 'main' into fix/tool-stream-parity
- bb65a91b Merge branch 'main' into image-sampling-defaults
- 188d5fbd Use model-based sampling defaults for image generation.
- 18479f05 Remove the unused re import
- 3c001d01 Strip protocol markers, not tags; stream the extractor's call
  separator
- 8e0b53e2 Keep &lt;...&gt; text in content next to a tool call
- 8ff71517 Finish tool-call streams: fallback path, finish-less end, exact
  strip
- 9a2e3271 Match streamed tool-call content to the non-streamed content

</details>

<details>
<summary>mlx-vlm commits touching the affected models' architectures</summary>

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
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | no concerns detected | 2.17s | 478 tok/s | 1.9 | none |
| mlx-community/aya-vision-8b-4bit | no concerns detected | 4.86s | 98.7 tok/s | 6.5 | none |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | no concerns detected | 10.72s | 29.3 tok/s | 23 | none |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | no concerns detected | 7.15s | 44.0 tok/s | 28 | none |
| mlx-community/gemma-3-27b-it-qat-4bit | no concerns detected | 10.39s | 30.0 tok/s | 17 | none |
| mlx-community/gemma-4-26b-a4b-it-4bit | no concerns detected | 5.49s | 101 tok/s | 16 | none |
| mlx-community/gemma-4-31b-it-4bit | no concerns detected | 9.58s | 22.2 tok/s | 20 | none |
| mlx-community/gemma-4-e4b-it-4bit | no concerns detected | 4.17s | 111 tok/s | 5.9 | none |
| mlx-community/GLM-4.6V-Flash-4bit | no concerns detected | 9.63s | 73.9 tok/s | 8.7 | none |
| mlx-community/GLM-4.6V-nvfp4 | no concerns detected | 36.86s | 39.9 tok/s | 78 | none |
| mlx-community/granite-4.0-3b-vision-4bit | no concerns detected | 3.27s | 169 tok/s | 4.7 | none |
| mlx-community/InternVL3-14B-4bit | no concerns detected | 6.24s | 54.1 tok/s | 10 | none |
| mlx-community/InternVL3-8B-bf16 | no concerns detected | 7.03s | 33.9 tok/s | 17 | none |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit | no concerns detected | 16.43s | 60.9 tok/s | 20 | none |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit | no concerns detected | 3.59s | 205 tok/s | 4.0 | none |
| mlx-community/MiniCPM-o-4_5-4bit | no concerns detected | 3.49s | 104 tok/s | 7.0 | none |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | no concerns detected | 7.31s | 65.3 tok/s | 13 | none |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | no concerns detected | 4.10s | 187 tok/s | 7.8 | none |
| mlx-community/North-Micro-Vision-Instruct-4bit | no concerns detected | 5.61s | 156 tok/s | 3.9 | none |
| mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit | no concerns detected | 6.89s | 72.8 tok/s | 24 | none |
| mlx-community/Phi-3.5-vision-instruct-bf16 | no concerns detected | 5.02s | 52.1 tok/s | 9.3 | none |
| mlx-community/pixtral-12b-8bit | no concerns detected | 8.06s | 38.4 tok/s | 16 | none |
| mlx-community/Qwen3-VL-2B-Thinking-bf16 | no concerns detected | 30.62s | 82.8 tok/s | 8.4 | none |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | no concerns detected | 42.88s | 73.3 tok/s | 23 | none |
| mlx-community/Qwen3-VL-32B-Instruct-4bit | no concerns detected | 75.17s | 19.6 tok/s | 26 | none |
| mlx-community/Qwen3-VL-8B-Instruct-4bit | no concerns detected | 40.87s | 68.7 tok/s | 11 | none |
| mlx-community/Qwen3.5-35B-A3B-4bit | no concerns detected | 39.80s | 70.8 tok/s | 25 | none |
| mlx-community/Qwen3.5-9B-MLX-4bit | no concerns detected | 41.11s | 85.3 tok/s | 11 | none |
| mlx-community/Qwen3.8-27B-nvfp4 | no concerns detected | 68.66s | 26.9 tok/s | 21 | none |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | no concerns detected | 3.19s | 126 tok/s | 5.6 | none |
| mlx-community/Step-3.7-Flash-oQ3e | no concerns detected | 45.94s | 47.7 tok/s | 92 | none |
| nativ-community/Mage-VL-OptiQ-4bit | no concerns detected | 4.87s | 126 tok/s | 5.4 | none |
| nativ-community/MiMo-V2.6-Distill-Qwen-9B-MLX-4bit | no concerns detected | 39.08s | 84.4 tok/s | 11 | none |
| nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit | no concerns detected | 8.26s | 36.1 tok/s | 18 | none |
| nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | no concerns detected | 9.99s | 131 tok/s | 23 | none |
| mlx-community/gemma-4-12B-it-4bit | concerns detected | 5.25s | 60.0 tok/s | 7.6 | duplicate keywords |
| mlx-community/Idefics3-8B-Llama3-bf16 | concerns detected | 9.68s | 33.8 tok/s | 18 | duplicate keywords |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | major concerns | 8.05s | 97.4 tok/s | 19 | stopped early: repeating; labelled fields not detected; incomplete thinking block |
| mlx-community/FastVLM-0.5B-bf16 | major concerns | 3.15s | 297 tok/s | 2.2 | labelled fields not detected |
| mlx-community/gemma-3n-E4B-it-4bit | major concerns | 5.99s | 70.0 tok/s | 7.2 | labelled fields not detected |
| mlx-community/granite-vision-3.2-2b-nvfp4 | major concerns | 4.98s | 145 tok/s | 4.2 | labelled fields not detected |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | major concerns | 58.17s | 18.6 tok/s | 15 | repeated text; cut off at token limit; duplicate keywords |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns | 3.57s | 107 tok/s | 6.7 | control tokens visible; labelled fields not detected |
| mlx-community/MiniCPM-V-4.6-4bit | major concerns | 2.98s | 221 tok/s | 3.3 | incomplete thinking block |
| mlx-community/Molmo2-8B-4bit | major concerns | 6.47s | 71.0 tok/s | 8.2 | stopped early: repeating; duplicate keywords |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | major concerns | 58.65s | 22.5 tok/s | 25 | control tokens visible; labelled fields not detected; cut off at token limit; role tokens visible; unsupplied place name |
| mlx-community/nanoLLaVA-1.5-4bit | major concerns | 1.94s | 196 tok/s | 1.6 | labelled fields not detected |
| mlx-community/Qwen2-VL-7B-Instruct-4bit | major concerns | 46.05s | 84.0 tok/s | 9.3 | repeated text; stopped early: repeating; duplicate keywords |
| mlx-community/SmolVLM-256M-Instruct-4bit | major concerns | 2.31s | 308 tok/s | 1.1 | labelled fields not detected |
| mlx-community/X-Reasoner-7B-8bit | major concerns | 22.07s | 55.4 tok/s | 14 | stopped early: repeating; duplicate keywords |

## Constraint-failure breakdown

How the fleet failed the catalogue constraints — a skew toward one constraint
suggests prompt difficulty rather than individual model faults.

- Duplicate keywords: 6 model(s)

## Observation clusters

Repeated mechanical observation signatures among results requiring review.

| Observed result | Models |
| --- | --- |
| Response repeats the same text; Generation was stopped early after sustained repeated output; Repeated keyword entries | 1 |
| Response repeats the same text; Response appears cut off at the token limit; Repeated keyword entries | 1 |
| Generation was stopped early after sustained repeated output; Repeated keyword entries | 2 |
| Generation was stopped early after sustained repeated output; Required labelled fields not detected; Internal reasoning block appears incomplete | 1 |
| Unrecognised model control tokens remain visible; Required labelled fields not detected | 1 |
| Unrecognised model control tokens remain visible; Required labelled fields not detected; Response appears cut off at the token limit; Conversation-role control tokens remain visible; Names a place the prompt did not supply | 1 |
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
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | major concerns | Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description; Response appears cut off at the token limit; Conversation-role control tokens remain visible; Names a place the prompt did not supply: River That | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-muse-glimmer-30b-optiq-4bit) |
| mlx-community/MiniCPM-V-4.6-4bit | major concerns | Internal reasoning block appears incomplete | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-minicpm-v-46-4bit) |

## Completions without detected concerns

35 completions without detected concerns (`LiquidAI/LFM2.5-VL-450M-MLX-bf16`, `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`, `mlx-community/GLM-4.6V-Flash-4bit`, `mlx-community/GLM-4.6V-nvfp4`, `mlx-community/InternVL3-14B-4bit`, `mlx-community/InternVL3-8B-bf16`, `mlx-community/Kimi-VL-A3B-Thinking-2506-8bit`, `mlx-community/LFM2.5-VL-3B-OptiQ-4bit`, `mlx-community/MiniCPM-o-4_5-4bit`, `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`, `mlx-community/Ministral-3-3B-Instruct-2512-4bit`, `mlx-community/North-Micro-Vision-Instruct-4bit`, `mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit`, `mlx-community/Phi-3.5-vision-instruct-bf16`, `mlx-community/Qwen3-VL-2B-Thinking-bf16`, `mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit`, `mlx-community/Qwen3-VL-32B-Instruct-4bit`, `mlx-community/Qwen3-VL-8B-Instruct-4bit`, `mlx-community/Qwen3.5-35B-A3B-4bit`, `mlx-community/Qwen3.5-9B-MLX-4bit`, `mlx-community/Qwen3.8-27B-nvfp4`, `mlx-community/SmolVLM2-2.2B-Instruct-mlx`, `mlx-community/Step-3.7-Flash-oQ3e`, `mlx-community/aya-vision-8b-4bit`, `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`, `mlx-community/gemma-3-27b-it-qat-4bit`, `mlx-community/gemma-4-26b-a4b-it-4bit`, `mlx-community/gemma-4-31b-it-4bit`, `mlx-community/gemma-4-e4b-it-4bit`, `mlx-community/granite-4.0-3b-vision-4bit`, `mlx-community/pixtral-12b-8bit`, `nativ-community/Mage-VL-OptiQ-4bit`, `nativ-community/MiMo-V2.6-Distill-Qwen-9B-MLX-4bit`, `nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit`, `nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit`); 7 more completed with prompt-compliance observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 9,984 x 6,656 pixels, 49,407,373 bytes
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: seed:* 0
- *Trust remote code:* true
- *check_models version:* 0.17.38
- *check_models revision:* e7629fd568aa6da8222cc31f1e70b21ed3ec96f8
- *check_models source dirty:* false
- *mlx-vlm:* 0.7.3
- *mlx-vlm source revision:* e6bf06faa0dda2598df9cd5fb6f2f204c4415b4f
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
