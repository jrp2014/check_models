# mlx-vlm compatibility findings across 41 cached vision-language models

**For mlx-vlm maintainers:** no crashes need action; 3 other results need
reproducing with mlx-vlm alone before anything is reported.

**Completed models:** 34 with no mechanical observations, 4 with
prompt-compliance observations only (labels, form or copied hint text), and 3
with observations mlx-vlm can produce (such as repetition, missing final
answers or visible control tokens); only those 3 are candidates for native
reproduction.

## Run summary

- *Run started:* 2026-11-11 18:43:43 GMT
- *Run finished:* 2026-11-11 18:47:27 GMT
- *Run duration:* 3m 44s
- *Time by phase:* generation 123s, model load 78s, prompt prep 1s, cleanup
  5s, outside the model loop 16s
- *Evaluation lane:* blind
- *Prompt hints:* no description or keyword hints
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Input image:* JPEG, 640 x 480 pixels (0.3 MP), 0.2 MB
- *Models attempted:* 41
- *Sampling settings:* checkpoint generation_config.json values for 17 of 41
  completed models where the command line left them unset; harness defaults
  elsewhere
- *Completed:* 41
- *Crashed:* 0
- *Indeterminate:* 0
- *Crashes requiring action:* 0
- *Other results requiring review:* 3
- *Reached token limit:* 2 (2 with incomplete output)
- *Stopped early for repetition:* 0

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

- *Baseline:* 95efedab:src/output/results.jsonl
- *Baseline run timestamp:* 2026-11-06 14:43:35 GMT
- *Baseline check_models:* 0.17.105 @ b21c07f22
- *Baseline mlx:* 0.32.5.dev20261106+9f806b360 @ 9f806b360
- *Baseline mlx-vlm:* 0.7.12 @ dcea78779
- *Baseline transformers:* 5.22.1
- *Baseline python:* 3.14.7
- *Baseline hardware:* Apple M5 Max, 40 GPU cores, 128.0 GB RAM
- *Earlier retained runs on this mlx version:* none recorded on
  0.32.5.dev20261107+fc344c986

**Not directly comparable** — output, quality and performance comparisons are
withheld because the runs differ in: prompt differs; evaluation lane differs
(assisted → blind); image differs (sha256 ecdd79a7ae79… → dea9e7ef9738…). The
roster, revisions and upstream changes below are facts about the runs and are
still shown.

Run environment: macOS changed since the baseline: 27.2 → 27.2.1; current run
detected GPU activity outside harness inference at the checks around 8 of 41
models (up to 8%); baseline run on battery power for 41 of 41 models; baseline
run detected GPU activity outside harness inference at the checks around 19 of
41 models (up to 37%).

<details>
<summary>mlx: 1 upstream commit(s) since the baseline (9f806b360..fc344c986)</summary>

- fc344c986 [CUDA] Pass MLX_CUDA_ARCH_LIST as a single quoted macro (#4839)

</details>

<details>
<summary>mlx-vlm: 50 upstream commit(s) since the baseline (dcea78779..2bb129235)</summary>

- 2bb12923 Use eps=1e-6 in the RF-DETR projector LayerNorms (#2709)
- efe2624f Merge pull request #2693 from e4c5/contributing-sign-commits
- 0097e8eb Merge pull request #2703 from lucasnewman/minimax-h3-image-quants
- 7fbe49ce Bump version to 0.8.0 (#2707)
- c85cef39 Avoid shared-GPU reference stalls in distributed tests (#2706)
- ba88debe Merge pull request #2705 from
  ahuzmeza/qwen3-vl-lora-deepstack-eval-1110
- c0980db2 Add native Ripple-65B-Preview and DFlash2 support (#2697)
- 4feb2d9e Add Ling 3.1 Flash support (#2696)
- e56ba1d1 Skip the deepstack eval in qwen3_vl and qwen3_vl_moe while training
- 9be4b98f Add quantized model support for MiniMax-H3-Image.
- bdc6bd73 Add MiniMax-H3-Image model (#2702)
- 8f2338a2 Merge pull request #2590 from
  Goekdeniz-Guelmez/(feat)-add-qat-training-too
- f1697c06 Start learning-rate decay at the warmup boundary
- 1764baff Restore saved QAT settings before applying defaults
- 5100480d Export attached MTP heads as separate drafters
- ... and 35 more (see results.jsonl)

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
generated (limit 1,000). *Native repro* says whether a person reran an
observation outside this harness on the same model revision and builds:
confirmed, not reproduced, or not yet checked. Recurring here alone is not
native confirmation. Keywords are counted from the answer's Keywords field.

| Model | Mechanical checks | Total | Gen tok/s | Peak GB | Prompt / output tok | Keywords | Observed | Native repro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ibm-granite/granite-vision-4.1-4b | no concerns detected | 2.44s | 65.4 tok/s | 9.7 | 721 / 55 | 9 | none | - |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | no concerns detected | 0.56s | 516 tok/s | 1.3 | 350 / 47 | 6 | none | - |
| mlx-community/AREX-2-4bit | no concerns detected | 5.17s | 33.0 tok/s | 17 | 421 / 80 | 8 | none | - |
| mlx-community/aya-vision-8b-4bit | no concerns detected | 3.98s | 100 tok/s | 6.5 | 2,909 / 82 | 10 | none | - |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | no concerns detected | 4.89s | 66.9 tok/s | 28 | 386 / 66 | 14 | none | - |
| mlx-community/gemma-4-12B-it-4bit | no concerns detected | 3.55s | 62.8 tok/s | 7.5 | 390 / 88 | 18 | none | - |
| mlx-community/gemma-4-26b-a4b-it-4bit | no concerns detected | 3.42s | 126 tok/s | 16 | 390 / 75 | 15 | none | - |
| mlx-community/gemma-4-e4b-it-4bit | no concerns detected | 2.39s | 125 tok/s | 5.9 | 386 / 67 | 14 | none | - |
| mlx-community/GLM-4.6V-Flash-4bit | no concerns detected | 2.15s | 85.7 tok/s | 8.0 | 499 / 49 | 5 | none | - |
| mlx-community/granite-4.0-3b-vision-4bit | no concerns detected | 1.35s | 181 tok/s | 4.7 | 734 / 57 | 10 | none | - |
| mlx-community/InternVL3-14B-4bit | no concerns detected | 4.85s | 55.4 tok/s | 10 | 3,435 / 90 | 18 | none | - |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit | no concerns detected | 1.46s | 214 tok/s | 3.7 | 344 / 73 | 10 | none | - |
| mlx-community/MiniCPM-o-4_5-4bit | no concerns detected | 2.33s | 106 tok/s | 7.0 | 177 / 99 | 11 | none | - |
| mlx-community/MiniCPM-V-4.7-1B-4bit | no concerns detected | 1.42s | 314 tok/s | 3.6 | 332 / 64 | 12 | none | - |
| mlx-community/MiniCPM-V-4.7-35B-A3B-4bit | no concerns detected | 4.19s | 86.8 tok/s | 24 | 332 / 93 | 18 | none | - |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | no concerns detected | 3.35s | 68.7 tok/s | 8.9 | 1,070 / 81 | 10 | none | - |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | no concerns detected | 1.63s | 201 tok/s | 3.7 | 1,069 / 102 | 12 | none | - |
| mlx-community/Mistral-Small-4-119B-2603-4bit | no concerns detected | 12.56s | 90.2 tok/s | 69 | 549 / 82 | 18 | none | - |
| mlx-community/Molmo2-8B-4bit | no concerns detected | 2.98s | 73.2 tok/s | 8.2 | 873 / 95 | 10 | none | - |
| mlx-community/MolmoPoint-8B-4bit | no concerns detected | 4.34s | 32.6 tok/s | 12 | 860 / 70 | 10 | none | - |
| mlx-community/North-Micro-Vision-Instruct-4bit | no concerns detected | 1.65s | 246 tok/s | 3.5 | 418 / 78 | 12 | none | - |
| mlx-community/Phi-3.5-vision-instruct-bf16 | no concerns detected | 2.92s | 59.8 tok/s | 9.1 | 883 / 94 | 19 | none | - |
| mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit | no concerns detected | 3.76s | 94.6 tok/s | 23 | 409 / 81 | 18 | none | - |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | no concerns detected | 3.11s | 129 tok/s | 19 | 409 / 83 | 10 | none | - |
| mlx-community/Qwen3-VL-8B-Instruct-4bit | no concerns detected | 1.83s | 103 tok/s | 6.8 | 409 / 60 | 10 | none | - |
| mlx-community/Qwen3.5-35B-A3B-4bit | no concerns detected | 4.02s | 113 tok/s | 21 | 421 / 88 | 10 | none | - |
| mlx-community/Qwen3.8-27B-nvfp4 | no concerns detected | 7.10s | 33.0 tok/s | 17 | 421 / 109 | 13 | none | - |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | no concerns detected | 1.44s | 125 tok/s | 5.6 | 1,204 / 31 | 4 | none | - |
| nativ-community/Mage-VL-OptiQ-4bit | no concerns detected | 1.70s | 150 tok/s | 5.3 | 420 / 88 | 10 | none | - |
| nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit | no concerns detected | 4.86s | 36.3 tok/s | 16 | 538 / 87 | 18 | none | - |
| nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | no concerns detected | 3.05s | 159 tok/s | 21 | 421 / 65 | 10 | none | - |
| nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 | no concerns detected | 34.31s | 30.9 tok/s | 81 | 420 / 89 | 13 | none | - |
| sahilchachra/LensVLM-9B-MXFP4 | no concerns detected | 2.72s | 101 tok/s | 6.8 | 421 / 111 | 10 | none | - |
| TechnoBaptist/Ternary-Bonsai-2-27B-mlx-2bit | no concerns detected | 4.73s | 43.4 tok/s | 11 | 421 / 94 | 14 | none | - |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | concerns detected | 6.16s | 136 tok/s | 19 | 545 / 540 | 16 | duplicate keywords | - |
| mlx-community/FastVLM-0.5B-bf16 | major concerns: answer format | 1.22s | 371 tok/s | 2.2 | 120 / 32 | - | labelled fields not detected | - |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns: answer format | 2.58s | insufficient sample | 7.0 | 3,512 / 14 | - | control tokens visible; labelled fields not detected | not yet checked (earlier check out of date) |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | major concerns: answer format | 43.35s | 25.2 tok/s | 25 | 545 / 1,000 | 9 | control tokens visible; labelled fields not detected; cut off at token limit; role tokens visible | not yet checked |
| mlx-community/nanoLLaVA-1.5-4bit | major concerns: answer format | 0.96s | 373 tok/s | 1.5 | 116 / 97 | - | labelled fields not detected | - |
| mlx-community/Qwen3.6-35B-A3B-4bit | major concerns: answer format | 3.84s | 116 tok/s | 21 | 421 / 96 | 20 | labelled fields not detected | - |
| mlx-community/InternVL3_5-1B-4bit | major concerns: generation | 3.90s | 349 tok/s | 2.3 | 3,438 / 1,000 | 144 | repeated text; cut off at token limit; duplicate keywords | not yet checked |

## Native reproduction checks

Recorded in native_repro_notes.yaml for this run's model revisions, builds
(mlx 0.32.5.dev20261107+fc344c986 @ fc344c986; mlx-vlm 0.8.0 @ 2bb129235;
transformers 5.22.2) and experiment inputs (image, prompt, generation
settings, and each model's rendered prompt and effective generation
arguments). A check made on another revision, build or experiment, or for a
model whose inputs were not recorded, does not apply and is not listed.

No recorded check applies to this run.

## Human accuracy notes

1 of 41 outputs reviewed by a person for this exact image, model revision and
output (accuracy_notes.yaml); the others are unreviewed. Each note is a
person's judgement against the image, not one of the mechanical checks.

| Model                        | Reviewed         | Note                                                                                     |
|------------------------------|------------------|------------------------------------------------------------------------------------------|
| mlx-community/Molmo2-8B-4bit | john, 2026-10-25 | Describes the pictured tabby cats as calico cats and gives incorrect positional details. |

## Completed attempts requiring review

*History* dates each observation (a crash counts as one) over this model's
retained runs: 45 earlier retained runs from git
`HEAD:src/output/results.jsonl`, back to 2026-08-30; older runs were not read.
A run that did not attempt the model is skipped, and a report-only correction
of a run counts once. "Last N runs" and "N runs since" count consecutive runs
ending with this one; "first" is the earliest run read that showed it. The
table shows each model's longest-running observation; every observation's
dates are under *History by observation*.

| Model | Mechanical checks | History | Observed result | Evidence |
| --- | --- | --- | --- | --- |
| mlx-community/InternVL3_5-1B-4bit | major concerns: generation | duplicate keywords: 8 runs since 2026-10-16; +2 more, newest back this run | Response repeats the same text; Response appears cut off at the token limit; Duplicate keywords: british english, british english examples of metadata, british english examples of details, british english examples of context, british english examples of information, british english examples of data, british english examples of content, british english examples of examples | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-internvl35-1b-4bit) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns: answer format | all 2: 36+ runs since 2026-09-12 or earlier | Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description, keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit) |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | major concerns: answer format | labelled fields not detected: 4 runs since 2026-10-30; +3 more, newest 2 runs since 2026-11-06 | Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description; Response appears cut off at the token limit; Conversation-role control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-muse-glimmer-30b-optiq-4bit) |

<details>
<summary>History by observation</summary>

| Model | Observation | Persistence |
| --- | --- | --- |
| mlx-community/InternVL3_5-1B-4bit | duplicate keywords | last 8 runs (since 2026-10-16), first 2026-10-04 |
| mlx-community/InternVL3_5-1B-4bit | repeated text | last 8 runs (since 2026-10-16), first 2026-10-04 |
| mlx-community/InternVL3_5-1B-4bit | cut off at token limit | back this run, first 2026-10-16 |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | labelled fields not detected | last 36+ runs (since 2026-09-12 or earlier) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | control tokens visible | last 36+ runs (since 2026-09-12 or earlier) |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | labelled fields not detected | last 4 runs (since 2026-10-30), first 2026-09-06 |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | role tokens visible | last 2 runs (since 2026-11-06), first 2026-09-06 |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | cut off at token limit | last 4 runs (since 2026-10-30), first 2026-09-06 |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | control tokens visible | last 2 runs (since 2026-11-06), first 2026-09-13 |

</details>

## Completions without detected concerns

34 completions without detected concerns; 4 more completed with prompt-compliance observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 640 x 480 pixels, 173,131 bytes
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: seed:* 0
- *Trust remote code:* true
- *check_models version:* 0.17.119
- *check_models revision:* 95efedab3f06f516b6822bbbe6087920dd9a59b6
- *check_models source dirty:* false
- *mlx-vlm:* 0.8.0
- *mlx-vlm source revision:* 2bb1292351fa2e93231d2bca1b0963e80e117ee5
- *mlx:* 0.32.5.dev20261107+fc344c986
- *mlx source revision:* fc344c986
- *transformers:* 5.22.2
- *macOS Version:* 27.2.1
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
