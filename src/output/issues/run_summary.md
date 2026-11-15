# mlx-vlm compatibility findings across 41 cached vision-language models

**For mlx-vlm maintainers:** no crashes need action; 1 other result needs
reproducing with mlx-vlm alone before anything is reported.

**Completed models:** 33 with no mechanical observations, 7 with
prompt-compliance observations only (labels, form or copied hint text), and 1
with observations mlx-vlm can produce (such as repetition, missing final
answers or visible control tokens); only those 1 are candidates for native
reproduction.

## Run summary

- *Run started:* 2026-11-15 00:17:44 GMT
- *Run finished:* 2026-11-15 00:28:19 GMT
- *Run duration:* 10m 35s
- *Time by phase:* generation 495s, model load 90s, prompt prep 25s, cleanup
  5s, outside the model loop 19s
- *Evaluation lane:* assisted
- *Prompt hints:* the image's description and keyword hints were included in
  the prompt, so field content may be copied from them rather than seen
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Input image:* JPEG, 9,984 x 5,616 pixels (56.1 MP), 28.7 MB
- *Models attempted:* 41
- *Sampling settings:* checkpoint generation_config.json values for 17 of 41
  completed models where the command line left them unset; harness defaults
  elsewhere
- *Completed:* 41
- *Crashed:* 0
- *Indeterminate:* 0
- *Crashes requiring action:* 0
- *Other results requiring review:* 1
- *Reached token limit:* 0
- *Stopped early for repetition:* 0

**What this run measures.** This run records model responses to one shared
image and prompt (evaluation lane: assisted). Mechanical checks are not
factual-accuracy judgments; inspect the image, prompt and final answers before
choosing a model. Results do not establish fitness for other tasks. Every
locally cached MLX vision-language model got the same image and prompt
(reproduced below) through mlx-vlm's generation pipeline.

<details>
<summary>Exact prompt sent to every model</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-11-07 16:58:09 UTC+01:00

Descriptive hints:
- Description hint: A small flock of sheep gathers and grazes along a grassy path in a rural meadow on a quiet, foggy morning. A small flock of sheep grazes and stands along a grassy path in an open meadow surrounded by misty autumn trees and rural landscape.
- Keyword hints: Agriculture, British countryside, Bushes, Countryside, English landscape, Farmland, Field, Grass, Grazing, Landscape, Livestock, Meadow, Mist, Overcast Sky, Pasture, Rural, Rural scenery, Sheep, Shrubs, Standing

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

- *Baseline:* 0aaa2c8e:src/output/results.jsonl
- *Baseline run timestamp:* 2026-11-11 18:47:27 GMT
- *Baseline check_models:* 0.17.119 @ 95efedab3
- *Baseline mlx:* 0.32.5.dev20261107+fc344c986 @ fc344c986
- *Baseline mlx-vlm:* 0.8.0 @ 2bb129235
- *Baseline transformers:* 5.22.2
- *Baseline python:* 3.14.7
- *Baseline hardware:* Apple M5 Max, 40 GPU cores, 128.0 GB RAM
- *Earlier retained runs on this mlx version:* 1 on
  0.32.5.dev20261107+fc344c986

**Not directly comparable** — output, quality and performance comparisons are
withheld because the runs differ in: prompt differs; evaluation lane differs
(blind → assisted); image differs (sha256 dea9e7ef9738… → cd7ad46ae878…). The
roster, revisions and upstream changes below are facts about the runs and are
still shown.

Model revisions changed since the baseline (their rows below reflect a
different snapshot, not only a different run):
`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` da5b2650a → 3acb09951,
`mlx-community/Qwen3.8-27B-nvfp4` 5ff8ef173 → f9cb64950

Run environment: current run on battery power for 41 of 41 models; current run
detected GPU activity outside harness inference at the checks around 2 of 41
models (up to 9%); baseline run detected GPU activity outside harness
inference at the checks around 8 of 41 models (up to 8%).

<details>
<summary>mlx-vlm: 12 upstream commit(s) since the baseline (2bb129235..e267ccf74)</summary>

- e267ccf7 fix(server): preserve RDMA links when importing cluster workers
  (#2730)
- 3d58bac9 [Server] Add distributed streaming, resumable serving, and cluster
  management (#2712)
- 10199b59 Merge pull request #2678 from Neywa-Labs/pc/deepseek-v41-dspark
- 8c5641c3 Merge pull request #2677 from Neywa-Labs/pc/native-mtp
- 6291b6ca Merge pull request #2676 from Neywa-Labs/pc/official-dspark
- dee11be3 Add native DeepSeek V4.1 DSpark and transactional cache support
- f417d6f5 Add native Mellum MTP and Qwen text-only MTP aliases
- 9891c730 Support official Red Hat and OpenBMB DSpark checkpoints
- 24ec162a Merge pull request #2719 from jjbuck/gemma4-per-layer-config
- 626e3e7d Fix sparse Gemma 4 per-layer attention shapes
- 9543e2f2 Gemma 4: read per_layer_config for the full-attention shapes
- 633757c8 Expose DeepSeek sharding and fix Mistral processor loading (#2717)

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
native confirmation. Keywords are counted from the answer's Keywords field,
with how many appear verbatim in the prompt's keyword hints. Hint text is the
percent of the description's words lying in four-word runs copied from the
prompt's description hint (of the whole answer when no field is labelled),
rounded down; 80% or more is reported as a repeated prompt hint.

| Model | Mechanical checks | Total | Gen tok/s | Peak GB | Prompt / output tok | Keywords | Hint text | Observed | Native repro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ibm-granite/granite-vision-4.1-4b | no concerns detected | 4.23s | 65.2 tok/s | 9.8 | 1,502 / 79 | 11 (8 from hints) | 38% | none | - |
| mlx-community/AREX-2-4bit | no concerns detected | 57.54s | 30.5 tok/s | 21 | 16,653 / 119 | 20 (20 from hints) | 55% | none | - |
| mlx-community/aya-vision-8b-4bit | no concerns detected | 4.98s | 97.7 tok/s | 6.6 | 2,429 / 129 | 19 (19 from hints) | 42% | none | - |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | no concerns detected | 9.57s | 130 tok/s | 19 | 1,606 / 598 | 17 (14 from hints) | 58% | none | - |
| mlx-community/gemma-4-12B-it-4bit | no concerns detected | 5.30s | 62.1 tok/s | 7.6 | 592 / 111 | 20 (20 from hints) | 37% | none | - |
| mlx-community/gemma-4-26b-a4b-it-4bit | no concerns detected | 4.64s | 126 tok/s | 16 | 592 / 90 | 16 (14 from hints) | 44% | none | - |
| mlx-community/gemma-4-e4b-it-4bit | no concerns detected | 3.54s | 124 tok/s | 6.0 | 588 / 92 | 16 (11 from hints) | 27% | none | - |
| mlx-community/InternVL3-14B-4bit | no concerns detected | 5.73s | 56.2 tok/s | 10 | 2,619 / 102 | 19 (19 from hints) | 44% | none | - |
| mlx-community/InternVL3_5-1B-4bit | no concerns detected | 2.34s | 401 tok/s | 2.2 | 2,622 / 107 | 20 (12 from hints) | 50% | none | - |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit | no concerns detected | 3.10s | 207 tok/s | 4.0 | 2,626 / 108 | 18 (18 from hints) | 70% | none | - |
| mlx-community/MiniCPM-o-4_5-4bit | no concerns detected | 3.31s | 104 tok/s | 7.1 | 385 / 105 | 16 (12 from hints) | 40% | none | - |
| mlx-community/MiniCPM-V-4.7-35B-A3B-4bit | no concerns detected | 5.88s | 87.0 tok/s | 25 | 932 / 123 | 18 (14 from hints) | 61% | none | - |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | no concerns detected | 5.68s | 66.3 tok/s | 12 | 2,583 / 120 | 15 (5 from hints) | 45% | none | - |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | no concerns detected | 3.22s | 188 tok/s | 6.4 | 2,582 / 107 | 13 (1 from hints) | 37% | none | - |
| mlx-community/Mistral-Small-4-119B-2603-4bit | no concerns detected | 21.87s | 78.1 tok/s | 71 | 2,062 / 93 | 17 (16 from hints) | 61% | none | - |
| mlx-community/Molmo2-8B-4bit | no concerns detected | 4.67s | 71.0 tok/s | 8.1 | 1,518 / 117 | 20 (20 from hints) | 40% | none | - |
| mlx-community/MolmoPoint-8B-4bit | no concerns detected | 8.55s | 30.5 tok/s | 13 | 3,120 / 109 | 20 (20 from hints) | 18% | none | - |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | no concerns detected | 50.96s | 24.6 tok/s | 25 | 4,432 / 990 | 15 (11 from hints) | 25% | none | - |
| mlx-community/North-Micro-Vision-Instruct-4bit | no concerns detected | 5.15s | 206 tok/s | 3.7 | 4,050 / 121 | 19 (19 from hints) | 40% | none | - |
| mlx-community/Phi-3.5-vision-instruct-bf16 | no concerns detected | 3.68s | 57.7 tok/s | 9.3 | 1,131 / 87 | 10 (8 from hints) | 53% | none | - |
| mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit | no concerns detected | 24.17s | 73.2 tok/s | 26 | 12,833 / 120 | 18 (18 from hints) | 56% | none | - |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | no concerns detected | 38.90s | 84.7 tok/s | 23 | 16,637 / 111 | 18 (17 from hints) | 43% | none | - |
| mlx-community/Qwen3-VL-8B-Instruct-4bit | no concerns detected | 40.83s | 68.1 tok/s | 11 | 16,637 / 95 | 18 (14 from hints) | 41% | none | - |
| mlx-community/Qwen3.5-35B-A3B-4bit | no concerns detected | 33.85s | 111 tok/s | 25 | 16,653 / 114 | 18 (18 from hints) | 32% | none | - |
| mlx-community/Qwen3.6-35B-A3B-4bit | no concerns detected | 37.17s | 112 tok/s | 25 | 16,653 / 128 | 24 (18 from hints) | 35% | none | - |
| mlx-community/Qwen3.8-27B-nvfp4 | no concerns detected | 62.42s | 30.9 tok/s | 21 | 16,653 / 129 | 20 (20 from hints) | 45% | none | - |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | no concerns detected | 3.35s | 124 tok/s | 5.6 | 1,415 / 84 | 20 (20 from hints) | 0% | none | - |
| nativ-community/Mage-VL-OptiQ-4bit | no concerns detected | 4.43s | 124 tok/s | 5.4 | 4,146 / 102 | 20 (20 from hints) | 47% | none | - |
| nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit | no concerns detected | 7.09s | 35.8 tok/s | 18 | 1,113 / 111 | 17 (12 from hints) | 50% | none | - |
| nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | no concerns detected | 6.34s | 161 tok/s | 23 | 3,641 / 108 | 18 (18 from hints) | 65% | none | - |
| nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 | no concerns detected | 41.70s | 30.4 tok/s | 87 | 3,640 / 105 | 17 (10 from hints) | 55% | none | - |
| sahilchachra/LensVLM-9B-MXFP4 | no concerns detected | 5.39s | 99.7 tok/s | 7.4 | 1,841 / 157 | 20 (20 from hints) | 49% | none | - |
| TechnoBaptist/Ternary-Bonsai-2-27B-mlx-2bit | no concerns detected | 66.06s | 35.9 tok/s | 17 | 16,653 / 110 | 14 (13 from hints) | 44% | none | - |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | concerns detected | 1.79s | 476 tok/s | 1.7 | 2,634 / 65 | 12 (9 from hints) | 57% | duplicate keywords | - |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | concerns detected | 6.16s | 66.2 tok/s | 28 | 588 / 77 | 13 (11 from hints) | 90% | prompt hint repeated | - |
| mlx-community/GLM-4.6V-Flash-4bit | concerns detected | 8.14s | 78.4 tok/s | 8.7 | 6,339 / 94 | 20 (20 from hints) | 100% | prompt hint repeated | - |
| mlx-community/granite-4.0-3b-vision-4bit | concerns detected | 2.84s | 179 tok/s | 4.8 | 1,524 / 70 | 13 (7 from hints) | 88% | prompt hint repeated | - |
| mlx-community/MiniCPM-V-4.7-1B-4bit | concerns detected | 3.09s | 302 tok/s | 3.7 | 932 / 104 | 20 (20 from hints) | 86% | prompt hint repeated | - |
| mlx-community/FastVLM-0.5B-bf16 | major concerns: answer format | 3.14s | 359 tok/s | 2.2 | 328 / 40 | - | 25% of answer | labelled fields not detected | - |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns: answer format | 3.57s | 101 tok/s | 6.7 | 2,698 / 30 | - | 72% of answer | control tokens visible; labelled fields not detected | not yet checked (earlier check out of date) |
| mlx-community/nanoLLaVA-1.5-4bit | major concerns: answer format | 2.07s | 265 tok/s | 1.6 | 324 / 84 | - | 25% | labelled fields not detected | - |

## Native reproduction checks

Recorded in native_repro_notes.yaml for this run's model revisions, builds
(mlx 0.32.5.dev20261107+fc344c986 @ fc344c986; mlx-vlm 0.8.0 @ e267ccf74;
transformers 5.23.0) and experiment inputs (image, prompt, generation
settings, and each model's rendered prompt and effective generation
arguments). A check made on another revision, build or experiment, or for a
model whose inputs were not recorded, does not apply and is not listed.

No recorded check applies to this run.

## Human accuracy notes

0 of 41 outputs reviewed by a person for this exact image, model revision and
output (accuracy_notes.yaml); the others are unreviewed; 1 note for other
images or models. Each note is a person's judgement against the image, not one
of the mechanical checks.

## Completed attempts requiring review

*History* dates each observation (a crash counts as one) over this model's
retained runs: 46 earlier retained runs from git
`HEAD:src/output/results.jsonl`, back to 2026-08-30; older runs were not read.
A run that did not attempt the model is skipped, and a report-only correction
of a run counts once. "Last N runs" and "N runs since" count consecutive runs
ending with this one; "first" is the earliest run read that showed it. The
table shows each model's longest-running observation; every observation's
dates are under *History by observation*.

| Model | Mechanical checks | History | Observed result | Evidence |
| --- | --- | --- | --- | --- |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns: answer format | all 2: 37+ runs since 2026-09-12 or earlier | Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description, keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit) |

<details>
<summary>History by observation</summary>

| Model | Observation | Persistence |
| --- | --- | --- |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | labelled fields not detected | last 37+ runs (since 2026-09-12 or earlier) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | control tokens visible | last 37+ runs (since 2026-09-12 or earlier) |

</details>

## Completions without detected concerns

33 completions without detected concerns; 7 more completed with prompt-compliance observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 9,984 x 5,616 pixels, 28,722,313 bytes
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: seed:* 0
- *Trust remote code:* true
- *check_models version:* 0.17.124
- *check_models revision:* 0aaa2c8e9923fb55e5c2f12816585a939f2d8bed
- *check_models source dirty:* false
- *mlx-vlm:* 0.8.0
- *mlx-vlm source revision:* e267ccf740b8b9dfc4529665b5efd8b9e1eff187
- *mlx:* 0.32.5.dev20261107+fc344c986
- *mlx source revision:* fc344c986
- *transformers:* 5.23.0
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
