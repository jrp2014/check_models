# mlx-vlm compatibility findings across 51 cached vision-language models

**For mlx-vlm maintainers:** no crashes need action; 6 other results need
reproducing with mlx-vlm alone before anything is reported.

**Completed models:** 41 with no mechanical observations, 4 with
prompt-compliance observations only (labels, form or copied hint text), and 6
with observations mlx-vlm can produce (such as repetition, missing final
answers or visible control tokens); only those 6 are candidates for native
reproduction.

## Run summary

- *Run started:* 2026-10-16 23:15:50 BST
- *Run finished:* 2026-10-16 23:21:25 BST
- *Run duration:* 5m 34s
- *Time by phase:* generation 211s, model load 96s, prompt prep 2s, cleanup
  6s, outside the model loop 24s
- *Evaluation lane:* blind
- *Prompt hints:* no description or keyword hints
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Input image:* JPEG, 640 x 480 pixels (0.3 MP), 0.2 MB
- *Models attempted:* 51
- *Sampling settings:* checkpoint generation_config.json values for 22 of 51
  completed models where the command line left them unset; harness defaults
  elsewhere
- *Completed:* 51
- *Crashed:* 0
- *Indeterminate:* 0
- *Crashes requiring action:* 0
- *Other results requiring review:* 6
- *Reached token limit:* 3 (3 with incomplete output)
- *Stopped early for repetition:* 1

**What this run measures.** This run records model responses to one shared
image and prompt (evaluation lane: blind). Mechanical checks are not
factual-accuracy judgments; inspect the image, prompt and final answers before
choosing a model. Results do not establish fitness for other tasks. Every
locally cached MLX vision-language model got the same image and prompt
(reproduced below) through mlx-vlm's generation pipeline.

<details>
<summary>Exact prompt sent to every model</summary>

```text
Create British-English catalogue metadata using only clearly visible facts. Omit uncertain details and unsupported identity, location, event, brand, species, period, or intent.

Write:
- a concrete 5-10-word title;
- a 1-2-sentence factual description of the main subject, setting, action, lighting, and distinctive details;
- 10-18 unique, comma-separated keywords.

Return exactly these three sections and nothing else:
Title:
Description:
Keywords:
```

</details>

## Since the baseline sweep

- *Baseline:* 5bfc8f11:src/output/results.jsonl
- *Baseline run timestamp:* 2026-10-16 17:06:36 BST
- *Baseline check_models:* 0.17.57 @ 1436f7ced
- *Baseline mlx:* 0.32.5.dev20261016+b6642e604 @ b6642e604
- *Baseline mlx-vlm:* 0.7.8 @ 259802ec0
- *Baseline transformers:* 5.20.1
- *Baseline python:* 3.14.7
- *Baseline hardware:* Apple M5 Max, 40 GPU cores, 128.0 GB RAM
- *Earlier retained runs on this mlx version:* 2 on
  0.32.5.dev20261016+b6642e604

**Not directly comparable** — output, quality and performance comparisons are
withheld because the runs differ in: prompt differs; evaluation lane differs
(assisted → blind); image differs (sha256 5526381f0507… → dea9e7ef9738…). The
roster, revisions and upstream changes below are facts about the runs and are
still shown.

Run environment: current run on battery power for 51 of 51 models; current run
detected GPU activity outside harness inference at the checks around 4 of 51
models (up to 39%); baseline run detected GPU activity outside harness
inference at the checks around 8 of 51 models (up to 6%).

<details>
<summary>mlx-vlm: 41 upstream commit(s) since the baseline (259802ec0..b38cfa7b8)</summary>

- b38cfa7b Merge pull request #2537 from lucasnewman/flux-3-image
- 816defaf Add support for Flux-3 image generation and editing.
- 1a4ddb68 Fix remaining affine QMV precision mismatches on MLX 0.32.4 (#2527)
- 9a0a0aa3 Merge pull request #2504 from Goekdeniz-Guelmez/add-grpo-for-vlms
- e13f9865 Use cached frozen vision features in GRPO parity test
- e12a7fbd Merge branch 'main' into add-grpo-for-vlms
- 088003ca Fix GRPO SDPA gradient parity test fixture
- b5223199 Enhance GRPO training notebook with W&amp;B integration and update
  installation instructions
- 82acaafe Merge branch 'main' into add-grpo-for-vlms
- 8068d053 Merge branch 'main' into add-grpo-for-vlms
- d4e4ddaf Merge branch 'main' into add-grpo-for-vlms
- f5949361 Merge branch 'main' into add-grpo-for-vlms
- 3707ce0f Merge branch 'main' into add-grpo-for-vlms
- 1d48b34d Merge upstream main and preserve GRPO training changes
- aaddce01 Refactor test patches to use import_module for generate_step in
  GRPO tests
- ... and 26 more (see results.jsonl)

</details>

Mechanical diff only: one image and one generation per model in each sweep,
with no warm-up or repeats. A difference shows that outputs or timings differ,
not why, whether it affects one model or many; attributing a cause needs
matched repeats or a native reproduction.

## Mechanical checks at a glance

Every attempted model, ordered by its mechanical checks, with counted facts.
"No concerns detected" is not an accuracy verdict: read the final answers in
the gallery. "Major concerns: generation" means generation itself failed (no
text, repeated text, or no final answer); "major concerns: answer format"
means an answer was generated but missed the requested form. Tokens in count
the image's tokens too, which drive prefill time; tokens out are those
generated (limit 1,000). Keywords are counted from the answer's Keywords
field.

| Model | Mechanical checks | Total | Gen tok/s | Peak GB | Prompt / output tok | Keywords | Observed | Native repro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | no concerns detected | 0.58s | 521 tok/s | 1.2 | 350 / 47 | 6 | none | - |
| mlx-community/AREX-2-4bit | no concerns detected | 5.16s | 33.3 tok/s | 17 | 421 / 80 | 8 | none | - |
| mlx-community/aya-vision-8b-4bit | no concerns detected | 4.09s | 100 tok/s | 6.5 | 2,909 / 82 | 10 | none | - |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | no concerns detected | 5.08s | 54.0 tok/s | 28 | 386 / 67 | 15 | none | - |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | no concerns detected | 6.38s | 109 tok/s | 18 | 545 / 478 | 17 | none | - |
| mlx-community/gemma-3-27b-it-qat-4bit | no concerns detected | 5.94s | 31.5 tok/s | 17 | 379 / 89 | 18 | none | - |
| mlx-community/gemma-3n-E4B-it-4bit | no concerns detected | 2.89s | 72.3 tok/s | 7.2 | 378 / 63 | 15 | none | - |
| mlx-community/gemma-4-12B-it-4bit | no concerns detected | 3.33s | 62.6 tok/s | 7.5 | 390 / 77 | 15 | none | - |
| mlx-community/gemma-4-26b-a4b-it-4bit | no concerns detected | 3.64s | 115 tok/s | 16 | 390 / 81 | 12 | none | - |
| mlx-community/gemma-4-e4b-it-4bit | no concerns detected | 2.42s | 125 tok/s | 5.9 | 386 / 62 | 12 | none | - |
| mlx-community/GLM-4.6V-Flash-4bit | no concerns detected | 2.54s | 86.5 tok/s | 8.0 | 499 / 49 | 5 | none | - |
| mlx-community/GLM-4.6V-nvfp4 | no concerns detected | 12.08s | 49.9 tok/s | 63 | 499 / 85 | 17 | none | - |
| mlx-community/granite-4.0-3b-vision-4bit | no concerns detected | 1.33s | 179 tok/s | 4.7 | 734 / 66 | 10 | none | - |
| mlx-community/granite-vision-3.2-2b-nvfp4 | no concerns detected | 2.22s | 174 tok/s | 3.9 | 3,095 / 119 | 7 | none | - |
| mlx-community/InternVL3-14B-4bit | no concerns detected | 4.96s | 56.1 tok/s | 10 | 3,435 / 90 | 18 | none | - |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit | no concerns detected | 9.61s | 67.8 tok/s | 20 | 489 / 435 | 11 | none | - |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit | no concerns detected | 1.69s | 214 tok/s | 3.6 | 344 / 74 | 10 | none | - |
| mlx-community/MiniCPM-o-4_5-4bit | no concerns detected | 2.27s | 103 tok/s | 7.0 | 177 / 103 | 11 | none | - |
| mlx-community/MiniCPM-V-4.6-4bit | no concerns detected | 3.01s | 255 tok/s | 3.1 | 330 / 478 | 9 | none | - |
| mlx-community/MiniCPM-V-4.7-1B-4bit | no concerns detected | 1.35s | 261 tok/s | 3.6 | 332 / 64 | 12 | none | - |
| mlx-community/MiniCPM-V-4.7-35B-A3B-4bit | no concerns detected | 4.15s | 78.9 tok/s | 24 | 332 / 86 | 15 | none | - |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | no concerns detected | 3.31s | 68.7 tok/s | 9.8 | 1,070 / 72 | 10 | none | - |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | no concerns detected | 1.92s | 198 tok/s | 4.5 | 1,069 / 100 | 10 | none | - |
| mlx-community/Molmo2-8B-4bit | no concerns detected | 3.05s | 71.3 tok/s | 8.5 | 873 / 95 | 10 | none | - |
| mlx-community/MolmoPoint-8B-4bit | no concerns detected | 4.50s | 32.0 tok/s | 12 | 860 / 70 | 10 | none | - |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | no concerns detected | 44.80s | 23.1 tok/s | 25 | 545 / 937 | 15 | none | - |
| mlx-community/North-Micro-Vision-Instruct-4bit | no concerns detected | 1.70s | 164 tok/s | 3.5 | 418 / 78 | 12 | none | - |
| mlx-community/Phi-3.5-vision-instruct-bf16 | no concerns detected | 2.78s | 59.4 tok/s | 9.1 | 883 / 94 | 19 | none | - |
| mlx-community/pixtral-12b-8bit | no concerns detected | 4.80s | 41.1 tok/s | 15 | 1,335 / 89 | 17 | none | - |
| mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit | no concerns detected | 3.80s | 84.5 tok/s | 23 | 409 / 87 | 18 | none | - |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | no concerns detected | 3.28s | 88.4 tok/s | 19 | 409 / 83 | 19 | none | - |
| mlx-community/Qwen3-VL-8B-Instruct-4bit | no concerns detected | 1.80s | 102 tok/s | 6.8 | 409 / 60 | 10 | none | - |
| mlx-community/Qwen3.5-35B-A3B-4bit | no concerns detected | 4.11s | 79.3 tok/s | 21 | 421 / 102 | 19 | none | - |
| mlx-community/Qwen3.8-27B-nvfp4 | no concerns detected | 6.49s | 30.5 tok/s | 17 | 421 / 113 | 14 | none | - |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | no concerns detected | 1.37s | 131 tok/s | 5.6 | 1,204 / 31 | 4 | none | - |
| nativ-community/Mage-VL-OptiQ-4bit | no concerns detected | 1.60s | 149 tok/s | 4.9 | 420 / 89 | 10 | none | - |
| nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit | no concerns detected | 5.03s | 37.3 tok/s | 17 | 538 / 87 | 18 | none | - |
| nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | no concerns detected | 3.25s | 141 tok/s | 21 | 421 / 65 | 10 | none | - |
| nativ-community/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-4bit | no concerns detected | 6.89s | 61.4 tok/s | 46 | 420 / 105 | 16 | none | - |
| sahilchachra/LensVLM-9B-MXFP4 | no concerns detected | 2.49s | 105 tok/s | 6.8 | 421 / 111 | 10 | none | - |
| TechnoBaptist/Ternary-Bonsai-2-27B-mlx-2bit | no concerns detected | 4.76s | 40.9 tok/s | 11 | 421 / 94 | 14 | none | - |
| mlx-community/FastVLM-0.5B-bf16 | major concerns: answer format | 0.88s | 356 tok/s | 2.2 | 120 / 32 | - | labelled fields not detected | - |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns: answer format | 2.66s | insufficient sample | 7.0 | 3,512 / 14 | - | control tokens visible; labelled fields not detected | candidate |
| mlx-community/nanoLLaVA-1.5-4bit | major concerns: answer format | 1.05s | 224 tok/s | 1.6 | 116 / 97 | - | labelled fields not detected | - |
| mlx-community/SmolVLM-256M-Instruct-4bit | major concerns: answer format | 0.64s | insufficient sample | 1.1 | 983 / 8 | - | labelled fields not detected | - |
| mlx-community/Step-3.7-Flash-oQ3e | major concerns: answer format | 32.15s | 50.5 tok/s | 87 | 286 / 236 | 19 | answer emitted twice; control tokens visible | candidate |
| vikhyatk/moondream2 | major concerns: answer format | 1.41s | 177 tok/s | 4.5 | 833 / 34 | - | labelled fields not detected | - |
| mlx-community/InternVL3_5-1B-4bit | major concerns: generation | 4.09s | 319 tok/s | 2.3 | 3,438 / 1,000 | 144 | repeated text; cut off at token limit; duplicate keywords | candidate |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | major concerns: generation | 56.01s | 19.1 tok/s | 15 | 107 / 1,000 | 141 | repeated text; cut off at token limit; duplicate keywords | candidate |
| mlx-community/Qwen2-VL-2B-mlx | major concerns: generation | 8.11s | 141 tok/s | 5.4 | 511 / 1,000 | - | repeated text; labelled fields not detected; cut off at token limit | candidate |
| mlx-community/X-Reasoner-7B-8bit | major concerns: generation | 5.79s | 64.9 tok/s | 10 | 511 / 275 | 54 | repeated text; stopped early: repeating; duplicate keywords | candidate |

## Observation clusters

Observation signatures shared by two or more results requiring review.

| Observed result | Models |
| --- | --- |
| Response repeats the same text; Response appears cut off at the token limit; Repeated keyword entries | [mlx-community/InternVL3_5-1B-4bit](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-internvl35-1b-4bit), [mlx-community/Llama-3.2-11B-Vision-Instruct-8bit](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llama-32-11b-vision-instruct-8bit) |

## Completed attempts requiring review

*History* dates each observation (a crash counts as one) over this model's
retained runs: 38 earlier retained runs from git
`HEAD:src/output/results.jsonl`, back to 2026-08-30; older runs were not read.
A run that did not attempt the model is skipped, and a report-only correction
of a run counts once. "Last N runs" and "N runs since" count consecutive runs
ending with this one; "first" is the earliest run read that showed it. The
table shows each model's longest-running observation; every observation's
dates are under *History by observation*.

| Model | Mechanical checks | History | Observed result | Evidence |
| --- | --- | --- | --- | --- |
| mlx-community/InternVL3_5-1B-4bit | major concerns: generation | duplicate keywords: back this run; +2 more, newest back this run | Response repeats the same text; Response appears cut off at the token limit; Duplicate keywords: british english, british english examples of metadata, british english examples of details, british english examples of context, british english examples of information, british english examples of data, british english examples of content, british english examples of examples | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-internvl35-1b-4bit) |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | major concerns: generation | all 3: back this run | Response repeats the same text; Response appears cut off at the token limit; Duplicate keywords: feline household domesticated animals, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort, feline household domesticated relaxation, feline household domesticated comfort and relaxation | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llama-32-11b-vision-instruct-8bit) |
| mlx-community/Qwen2-VL-2B-mlx | major concerns: generation | repeated text: 6 runs since 2026-10-10; +2 more, newest 4 runs since 2026-10-11 | Response repeats the same text; Required labelled fields not detected: keywords; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-qwen2-vl-2b-mlx) |
| mlx-community/X-Reasoner-7B-8bit | major concerns: generation | all 3: back this run | Response repeats the same text; Generation was stopped early after sustained repeated output; Duplicate keywords: feline behavior, feline companions, feline comfort, feline companionship, feline lifestyle, feline rest, feline home | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-x-reasoner-7b-8bit) |
| mlx-community/Step-3.7-Flash-oQ3e | major concerns: answer format | all 2: back this run | Final answer emitted twice, around &lt;/think&gt;; Unrecognised model control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-step-37-flash-oq3e) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns: answer format | all 2: 29+ runs since 2026-09-12 or earlier | Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description, keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit) |

<details>
<summary>History by observation</summary>

| Model | Observation | Persistence |
| --- | --- | --- |
| mlx-community/InternVL3_5-1B-4bit | duplicate keywords | back this run, first 2026-10-04 |
| mlx-community/InternVL3_5-1B-4bit | repeated text | back this run, first 2026-10-04 |
| mlx-community/InternVL3_5-1B-4bit | cut off at token limit | not in earlier runs read |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | duplicate keywords | back this run, first 2026-09-25 |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | repeated text | back this run, first 2026-09-18 |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | cut off at token limit | back this run, first 2026-09-18 |
| mlx-community/Qwen2-VL-2B-mlx | labelled fields not detected | last 4 runs (since 2026-10-11) |
| mlx-community/Qwen2-VL-2B-mlx | repeated text | last 6 runs (since 2026-10-10) |
| mlx-community/Qwen2-VL-2B-mlx | cut off at token limit | last 4 runs (since 2026-10-11), first 2026-10-10 |
| mlx-community/X-Reasoner-7B-8bit | duplicate keywords | back this run, first 2026-09-06 |
| mlx-community/X-Reasoner-7B-8bit | repeated text | back this run, first 2026-08-30 |
| mlx-community/X-Reasoner-7B-8bit | stopped early: repeating | back this run, first 2026-09-06 |
| mlx-community/Step-3.7-Flash-oQ3e | answer emitted twice | back this run, first 2026-09-06 |
| mlx-community/Step-3.7-Flash-oQ3e | control tokens visible | back this run, first 2026-09-06 |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | labelled fields not detected | last 29+ runs (since 2026-09-12 or earlier) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | control tokens visible | last 29+ runs (since 2026-09-12 or earlier) |

</details>

## Completions without detected concerns

41 completions without detected concerns; 4 more completed with prompt-compliance observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 640 x 480 pixels, 173,131 bytes
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: seed:* 0
- *Trust remote code:* true
- *check_models version:* 0.17.71
- *check_models revision:* 5bfc8f11d92aaa023134754062ea98d36dbd5a86
- *check_models source dirty:* false
- *mlx-vlm:* 0.7.8
- *mlx-vlm source revision:* b38cfa7b84b7824400290e6a2bb7d9fd3fe3690f
- *mlx:* 0.32.5.dev20261016+b6642e604
- *mlx source revision:* b6642e604
- *transformers:* 5.20.1
- *macOS Version:* 27.0.1
- *GPU/Chip:* Apple M5 Max
- *Python Version:* 3.14.7

## Full artifacts

**Evidence links** target the repository's mutable main branch: they show this
run only once these artifacts are committed, and a later run's commit replaces
them. Before sharing upstream, pin them to the commit that published these
artifacts.

| Artifact | Link |
| --- | --- |
| Diagnostics | [diagnostics.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md) |
| Model gallery | [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md) |
| Results JSONL | [results.jsonl](https://github.com/jrp2014/check_models/blob/main/src/output/results.jsonl) |
| Environment | [environment.log](https://github.com/jrp2014/check_models/blob/main/src/output/environment.log) |
| Log | [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log) |
