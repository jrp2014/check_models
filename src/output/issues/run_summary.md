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

- *Run started:* 2026-09-27 21:10:26 BST
- *Run finished:* 2026-09-27 21:23:47 BST
- *Run duration:* 13m 19s
- *Time by phase:* generation 646s, model load 96s, prompt prep 40s, cleanup
  6s, outside the model loop 17s
- *Evaluation lane:* assisted
- *Prompt hints:* the image's description and keyword hints were included in
  the prompt, so field content may be copied from them rather than seen
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Input image:* JPEG, 9,805 x 6,538 pixels (64.1 MP), 54.2 MB
- *Models attempted:* 50
- *Sampling settings:* checkpoint generation_config.json values for 25 of 50
  completed models where the command line left them unset; harness defaults
  elsewhere
- *Completed:* 50
- *Crashed:* 0
- *Indeterminate:* 0
- *Crashes requiring action:* 0
- *Other results requiring review:* 3
- *Reached token limit:* 2
- *Incomplete output at token limit:* 2
- *Stopped early for repetition:* 0

Observations are mechanical facts from one image, not general model-quality
judgements.

<details>
<summary>Exact prompt sent to every model</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-09-26 17:36:29 UTC+01:00
- GPS: 50.682100°N, 3.466600°W

Descriptive hints:
- Description hint: A blue and light blue motor cabin cruiser with a tan canopy moored at a marina dock alongside other sailboats and leisure boats on calm water, casting clear reflections in the early evening light.
- Keyword hints: Boat, Boat canopy, Boat fender, Boating, Cabin cruiser, Calm Water, Dock, Harbor, Marina, Mast, Mooring, Motorboat, Nautical, Reflection, Rope, Sailboat, Sailing, Water reflection, Watercraft, Yacht

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

- *Baseline:* de73403c:src/output/results.jsonl
- *Baseline run timestamp:* 2026-09-27 01:01:54 BST
- *Baseline check_models:* 0.17.38 @ e7629fd56
- *Baseline mlx:* 0.32.3.dev20260926+a2a09fd56 @ a2a09fd56
- *Baseline mlx-vlm:* 0.7.3 @ e6bf06faa
- *Baseline transformers:* 5.17.0
- *Baseline python:* 3.14.7
- *Baseline hardware:* Apple M5 Max, 40 GPU cores, 128.0 GB RAM

**Not directly comparable** — output, quality and performance comparisons are
withheld because the runs differ in: prompt differs; image differs (sha256
97f53d5eeb6a… → bd1c0e60ad08…). The roster, revisions and upstream changes
below are facts about the runs and are still shown.

Run environment: current run on battery power for 6 of 50 models.

<details>
<summary>mlx: 1 upstream commit(s) since the baseline (a2a09fd56..09e67c686)</summary>

- 09e67c686 Fix perf regression for upsample `linear` mode align_corners=False
  (#4500)

</details>

<details>
<summary>mlx-vlm: 1 upstream commit(s) since the baseline (e6bf06faa..967bf90b8)</summary>

- 967bf90b Return CacheList from Florence-2 and Nemotron-Parse make_cache
  (#2381)

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
| mlx-community/aya-vision-8b-4bit | no concerns detected | 5.07s | 101 tok/s | 6.5 | none |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | no concerns detected | 6.65s | 55.0 tok/s | 28 | none |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | no concerns detected | 8.19s | 118 tok/s | 19 | none |
| mlx-community/gemma-4-12B-it-4bit | no concerns detected | 5.25s | 61.2 tok/s | 7.6 | none |
| mlx-community/gemma-4-26b-a4b-it-4bit | no concerns detected | 5.12s | 112 tok/s | 16 | none |
| mlx-community/gemma-4-31b-it-4bit | no concerns detected | 10.59s | 23.8 tok/s | 20 | none |
| mlx-community/granite-4.0-3b-vision-4bit | no concerns detected | 3.76s | 169 tok/s | 4.6 | none |
| mlx-community/InternVL3-14B-4bit | no concerns detected | 5.70s | 55.7 tok/s | 10 | none |
| mlx-community/InternVL3-8B-bf16 | no concerns detected | 6.08s | 36.1 tok/s | 17 | none |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | no concerns detected | 9.62s | 21.5 tok/s | 15 | none |
| mlx-community/MiniCPM-o-4_5-4bit | no concerns detected | 3.24s | 106 tok/s | 7.0 | none |
| mlx-community/MiniCPM-V-4.6-4bit | no concerns detected | 4.52s | 295 tok/s | 3.3 | none |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | no concerns detected | 7.73s | 64.9 tok/s | 13 | none |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | no concerns detected | 3.93s | 187 tok/s | 7.8 | none |
| mlx-community/Molmo2-8B-4bit | no concerns detected | 5.78s | 72.2 tok/s | 8.5 | none |
| mlx-community/North-Micro-Vision-Instruct-4bit | no concerns detected | 5.07s | 210 tok/s | 3.9 | none |
| mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit | no concerns detected | 5.85s | 99.0 tok/s | 24 | none |
| mlx-community/Qwen3-VL-2B-Thinking-bf16 | no concerns detected | 27.77s | 89.2 tok/s | 8.4 | none |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | no concerns detected | 42.73s | 83.7 tok/s | 23 | none |
| mlx-community/Qwen3-VL-32B-Instruct-4bit | no concerns detected | 74.51s | 21.0 tok/s | 26 | none |
| mlx-community/Qwen3-VL-8B-Instruct-4bit | no concerns detected | 42.92s | 70.7 tok/s | 11 | none |
| mlx-community/Qwen3.5-35B-A3B-4bit | no concerns detected | 41.27s | 105 tok/s | 25 | none |
| mlx-community/Qwen3.5-9B-MLX-4bit | no concerns detected | 42.30s | 90.6 tok/s | 11 | none |
| mlx-community/Qwen3.8-27B-nvfp4 | no concerns detected | 55.64s | 29.4 tok/s | 21 | none |
| nativ-community/MiMo-V2.6-Distill-Qwen-9B-MLX-4bit | no concerns detected | 36.32s | 84.0 tok/s | 11 | none |
| nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit | no concerns detected | 8.51s | 36.4 tok/s | 18 | none |
| nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | no concerns detected | 9.54s | 144 tok/s | 23 | none |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | concerns detected | 2.14s | 486 tok/s | 1.9 | duplicate keywords; prompt hint repeated |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | concerns detected | 10.05s | 30.5 tok/s | 23 | prompt hint repeated |
| mlx-community/gemma-3-27b-it-qat-4bit | concerns detected | 8.57s | 30.8 tok/s | 17 | unsupplied place name |
| mlx-community/gemma-4-e4b-it-4bit | concerns detected | 5.07s | 103 tok/s | 5.9 | duplicate keywords |
| mlx-community/GLM-4.6V-Flash-4bit | concerns detected | 8.62s | 75.9 tok/s | 8.7 | prompt hint repeated |
| mlx-community/GLM-4.6V-nvfp4 | concerns detected | 32.92s | 41.8 tok/s | 78 | prompt hint repeated |
| mlx-community/Idefics3-8B-Llama3-bf16 | concerns detected | 7.12s | 34.3 tok/s | 18 | prompt hint repeated |
| mlx-community/Phi-3.5-vision-instruct-bf16 | concerns detected | 3.98s | 58.6 tok/s | 9.3 | prompt hint repeated |
| mlx-community/pixtral-12b-8bit | concerns detected | 7.91s | 39.5 tok/s | 16 | prompt hint repeated |
| mlx-community/Qwen2-VL-7B-Instruct-4bit | concerns detected | 47.93s | 91.8 tok/s | 9.3 | prompt hint repeated |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | concerns detected | 3.47s | 129 tok/s | 5.6 | prompt hint repeated |
| mlx-community/Step-3.7-Flash-oQ3e | concerns detected | 39.24s | 51.3 tok/s | 92 | prompt hint repeated |
| mlx-community/X-Reasoner-7B-8bit | concerns detected | 19.11s | 57.4 tok/s | 14 | duplicate keywords |
| nativ-community/Mage-VL-OptiQ-4bit | concerns detected | 4.84s | 125 tok/s | 5.4 | prompt hint repeated |
| mlx-community/FastVLM-0.5B-bf16 | major concerns | 2.95s | 359 tok/s | 2.2 | labelled fields not detected |
| mlx-community/gemma-3n-E4B-it-4bit | major concerns | 4.46s | 85.0 tok/s | 7.2 | labelled fields not detected |
| mlx-community/granite-vision-3.2-2b-nvfp4 | major concerns | 5.33s | 129 tok/s | 4.4 | labelled fields not detected |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit | major concerns | 20.29s | 63.2 tok/s | 20 | repeated text; labelled fields not detected; cut off at token limit; incomplete thinking block; duplicate keywords |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit | major concerns | 3.56s | 203 tok/s | 4.0 | labelled fields not detected |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns | 3.56s | 107 tok/s | 6.7 | control tokens visible; labelled fields not detected |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | major concerns | 53.48s | 24.1 tok/s | 25 | control tokens visible; labelled fields not detected; cut off at token limit; role tokens visible |
| mlx-community/nanoLLaVA-1.5-4bit | major concerns | 1.94s | 335 tok/s | 1.8 | labelled fields not detected |
| mlx-community/SmolVLM-256M-Instruct-4bit | major concerns | 2.31s | 524 tok/s | 1.1 | labelled fields not detected |

## Constraint-failure breakdown

How the fleet failed the catalogue constraints — a skew toward one constraint
suggests prompt difficulty rather than individual model faults.

- Duplicate keywords: 4 model(s)

## Observation clusters

Repeated mechanical observation signatures among results requiring review.

| Observed result | Models |
| --- | --- |
| Response repeats the same text; Required labelled fields not detected; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Repeated keyword entries | 1 |
| Unrecognised model control tokens remain visible; Required labelled fields not detected | 1 |
| Unrecognised model control tokens remain visible; Required labelled fields not detected; Response appears cut off at the token limit; Conversation-role control tokens remain visible | 1 |

## Completed attempts requiring review

| Model | Mechanical checks | Observed result | Evidence |
| --- | --- | --- | --- |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit | major concerns | Response repeats the same text; Required labelled fields not detected: title; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Duplicate keywords: dock, marina, calm water, reflections, early evening, blue hull, tan canopy, blue and light blue, marina dock, leisure boats, early evening light, blue and light blue hull, other sailboats, correct conflicts, moored | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-8bit) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns | Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description, keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit) |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | major concerns | Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description; Response appears cut off at the token limit; Conversation-role control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-muse-glimmer-30b-optiq-4bit) |

## Completions without detected concerns

27 completions without detected concerns (`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit`, `mlx-community/InternVL3-14B-4bit`, `mlx-community/InternVL3-8B-bf16`, `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`, `mlx-community/MiniCPM-V-4.6-4bit`, `mlx-community/MiniCPM-o-4_5-4bit`, `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`, `mlx-community/Ministral-3-3B-Instruct-2512-4bit`, `mlx-community/Molmo2-8B-4bit`, `mlx-community/North-Micro-Vision-Instruct-4bit`, `mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit`, `mlx-community/Qwen3-VL-2B-Thinking-bf16`, `mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit`, `mlx-community/Qwen3-VL-32B-Instruct-4bit`, `mlx-community/Qwen3-VL-8B-Instruct-4bit`, `mlx-community/Qwen3.5-35B-A3B-4bit`, `mlx-community/Qwen3.5-9B-MLX-4bit`, `mlx-community/Qwen3.8-27B-nvfp4`, `mlx-community/aya-vision-8b-4bit`, `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`, `mlx-community/gemma-4-12B-it-4bit`, `mlx-community/gemma-4-26b-a4b-it-4bit`, `mlx-community/gemma-4-31b-it-4bit`, `mlx-community/granite-4.0-3b-vision-4bit`, `nativ-community/MiMo-V2.6-Distill-Qwen-9B-MLX-4bit`, `nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit`, `nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit`); 20 more completed with prompt-compliance observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 9,805 x 6,538 pixels, 54,173,041 bytes
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: seed:* 0
- *Trust remote code:* true
- *check_models version:* 0.17.38
- *check_models revision:* de73403c150baab6edb790ca0453c113782d44c4
- *check_models source dirty:* true
- *mlx-vlm:* 0.7.3
- *mlx-vlm source revision:* 967bf90b8e7ae6e5110b6da87134f30d94591d2e
- *mlx:* 0.32.3.dev20260927+09e67c686
- *mlx source revision:* 09e67c686
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
