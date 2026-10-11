# mlx-vlm compatibility findings across 47 cached vision-language models

**For mlx-vlm maintainers:** no crashes need action; 5 other results need
reproducing with mlx-vlm alone before anything is reported.

**Completed models:** 26 with no mechanical observations, 16 with
prompt-compliance observations only (labels, form or copied hint text), and 5
with observations mlx-vlm can produce (such as repetition, missing final
answers or visible control tokens); only those 5 are candidates for native
reproduction.

## Run summary

- *Run started:* 2026-10-11 19:53:58 BST
- *Run finished:* 2026-10-11 20:05:10 BST
- *Run duration:* 11m 11s
- *Time by phase:* generation 541s, model load 78s, prompt prep 33s, cleanup
  5s, outside the model loop 18s
- *Evaluation lane:* assisted
- *Prompt hints:* the image's description and keyword hints were included in
  the prompt, so field content may be copied from them rather than seen
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Input image:* JPEG, 6,213 x 9,320 pixels (57.9 MP), 52.1 MB
- *Models attempted:* 47
- *Sampling settings:* checkpoint generation_config.json values for 20 of 47
  completed models where the command line left them unset; harness defaults
  elsewhere
- *Completed:* 47
- *Crashed:* 0
- *Indeterminate:* 0
- *Crashes requiring action:* 0
- *Other results requiring review:* 5
- *Reached token limit:* 2 (2 with incomplete output)
- *Stopped early for repetition:* 2

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
- Capture date/time: 2026-10-10 18:09:17 UTC+01:00
- GPS: 52.628000°N, 1.293600°E

Descriptive hints:
- Description hint: A low-angle view of the medieval tower and Victorian lead flèche of the Church of St Peter Mancroft illuminated in evening light, with pedestrians walking along the street below in Norwich, Norfolk, England.
- Keyword hints: Archway, Bell Tower, Church Tower, Clear Sky, England, Gothic Architecture, Historic Landmark, Historic architecture, Landmark, Norwich, Pedestrians, Perpendicular Gothic, Spire, St Peter Mancroft, Stone Masonry, Stone facade, Trees, Twilight, United Kingdom, Walking

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

- *Baseline:* 3d1b38a4:src/output/results.jsonl
- *Baseline run timestamp:* 2026-10-11 00:35:32 BST
- *Baseline check_models:* 0.17.43 @ 84f7be8c9
- *Baseline mlx:* 0.32.4.dev20261010+06eb7483f @ 06eb7483f
- *Baseline mlx-vlm:* 0.7.7 @ 952d4f6bc
- *Baseline transformers:* 5.19.0
- *Baseline python:* 3.14.7
- *Baseline hardware:* Apple M5 Max, 40 GPU cores, 128.0 GB RAM
- *Earlier retained runs on this mlx version:* 2 on
  0.32.4.dev20261010+06eb7483f

**Not directly comparable** — output, quality and performance comparisons are
withheld because the runs differ in: prompt differs; image differs (sha256
712a5faa6eab… → 5526381f0507…). The roster, revisions and upstream changes
below are facts about the runs and are still shown.

Run environment: current run on battery power for 47 of 47 models; current run
detected GPU activity outside harness inference at the checks around 36 of 47
models (up to 99%).

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
field, with how many appear verbatim in the prompt's keyword hints. Hint text
is the percent of the description's words lying in four-word runs copied from
the prompt's description hint (of the whole answer when no field is labelled),
rounded down; 80% or more is reported as a repeated prompt hint.

| Model | Mechanical checks | Total | Gen tok/s | Peak GB | Prompt / output tok | Keywords | Hint text | Observed | Native repro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | no concerns detected | 2.07s | 481 tok/s | 1.9 | 2,131 / 83 | 15 (9 from hints) | 19% | none | - |
| mlx-community/aya-vision-8b-4bit | no concerns detected | 4.54s | 103 tok/s | 6.5 | 2,108 / 102 | 18 (4 from hints) | 23% | none | - |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | no concerns detected | 6.08s | 74.4 tok/s | 28 | 611 / 92 | 15 (12 from hints) | 0% | none | - |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | no concerns detected | 8.48s | 126 tok/s | 19 | 1,656 / 617 | 35 (13 from hints) | 22% | none | - |
| mlx-community/gemma-3-27b-it-qat-4bit | no concerns detected | 8.91s | 31.4 tok/s | 17 | 610 / 145 | 18 (15 from hints) | 22% | none | - |
| mlx-community/gemma-4-12B-it-4bit | no concerns detected | 5.12s | 62.7 tok/s | 7.6 | 615 / 118 | 17 (14 from hints) | 57% | none | - |
| mlx-community/gemma-4-26b-a4b-it-4bit | no concerns detected | 4.74s | 124 tok/s | 16 | 615 / 105 | 16 (15 from hints) | 42% | none | - |
| mlx-community/gemma-4-e4b-it-4bit | no concerns detected | 3.79s | 125 tok/s | 6.0 | 611 / 108 | 15 (13 from hints) | 20% | none | - |
| mlx-community/GLM-4.6V-nvfp4 | no concerns detected | 20.78s | 45.0 tok/s | 78 | 6,372 / 112 | 19 (18 from hints) | 44% | none | - |
| mlx-community/InternVL3-14B-4bit | no concerns detected | 5.44s | 57.4 tok/s | 10 | 2,132 / 98 | 18 (18 from hints) | 47% | none | - |
| mlx-community/InternVL3_5-1B-4bit | no concerns detected | 2.26s | 421 tok/s | 2.1 | 2,135 / 126 | 23 (16 from hints) | 26% | none | - |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit | no concerns detected | 15.34s | 66.6 tok/s | 20 | 1,342 / 750 | 19 (19 from hints) | 75% | none | - |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | no concerns detected | 8.74s | 22.0 tok/s | 15 | 320 / 101 | 11 (9 from hints) | 11% | none | - |
| mlx-community/MiniCPM-o-4_5-4bit | no concerns detected | 3.21s | 106 tok/s | 7.1 | 410 / 104 | 15 (12 from hints) | 13% | none | - |
| mlx-community/MiniCPM-V-4.6-4bit | no concerns detected | 3.49s | 304 tok/s | 3.2 | 950 / 333 | 18 (10 from hints) | 45% | none | - |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | no concerns detected | 6.31s | 66.3 tok/s | 13 | 2,959 / 122 | 16 (11 from hints) | 35% | none | - |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | no concerns detected | 3.62s | 188 tok/s | 7.8 | 2,958 / 113 | 17 (10 from hints) | 14% | none | - |
| mlx-community/North-Micro-Vision-Instruct-4bit | no concerns detected | 5.30s | 210 tok/s | 3.9 | 4,101 / 158 | 20 (20 from hints) | 46% | none | - |
| mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit | no concerns detected | 26.70s | 73.4 tok/s | 26 | 12,809 / 120 | 18 (18 from hints) | 20% | none | - |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | no concerns detected | 43.17s | 86.9 tok/s | 23 | 16,566 / 144 | 21 (18 from hints) | 31% | none | - |
| mlx-community/Qwen3-VL-8B-Instruct-4bit | no concerns detected | 42.92s | 69.5 tok/s | 11 | 16,566 / 91 | 10 (9 from hints) | 13% | none | - |
| mlx-community/Qwen3.5-35B-A3B-4bit | no concerns detected | 41.44s | 110 tok/s | 25 | 16,579 / 130 | 18 (14 from hints) | 11% | none | - |
| mlx-community/Qwen3.8-27B-nvfp4 | no concerns detected | 60.95s | 30.2 tok/s | 21 | 16,579 / 133 | 15 (14 from hints) | 34% | none | - |
| mlx-community/X-Reasoner-7B-8bit | no concerns detected | 18.03s | 60.0 tok/s | 14 | 16,577 / 127 | 22 (17 from hints) | 57% | none | - |
| nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | no concerns detected | 8.19s | 159 tok/s | 23 | 3,642 / 101 | 12 (11 from hints) | 61% | none | - |
| TechnoBaptist/Ternary-Bonsai-2-27B-mlx-2bit | no concerns detected | 69.83s | 36.3 tok/s | 17 | 16,579 / 123 | 16 (14 from hints) | 50% | none | - |
| mlx-community/GLM-4.6V-Flash-4bit | concerns detected | 8.98s | 78.8 tok/s | 8.7 | 6,372 / 121 | 20 (20 from hints) | 100% | prompt hint repeated | - |
| mlx-community/granite-4.0-3b-vision-4bit | concerns detected | 3.27s | 181 tok/s | 4.8 | 1,410 / 105 | 10 (9 from hints) | 85% | prompt hint repeated | - |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit | concerns detected | 3.09s | 213 tok/s | 4.0 | 2,122 / 84 | 13 (12 from hints) | 100% | prompt hint repeated | - |
| mlx-community/Molmo2-8B-4bit | concerns detected | 5.14s | 73.0 tok/s | 8.2 | 1,562 / 141 | 19 (15 from hints) | 43% | duplicate keywords | - |
| mlx-community/pixtral-12b-8bit | concerns detected | 7.38s | 40.1 tok/s | 16 | 3,152 / 115 | 20 (20 from hints) | 86% | prompt hint repeated | - |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | concerns detected | 3.50s | 128 tok/s | 5.6 | 1,446 / 128 | 21 (20 from hints) | 100% | duplicate keywords; prompt hint repeated | - |
| mlx-community/Step-3.7-Flash-oQ3e | concerns detected | 35.43s | 52.9 tok/s | 92 | 3,506 / 118 | 20 (20 from hints) | 100% | prompt hint repeated | - |
| nativ-community/Mage-VL-OptiQ-4bit | concerns detected | 4.57s | 128 tok/s | 5.4 | 4,229 / 125 | 20 (20 from hints) | 96% | prompt hint repeated | - |
| nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit | concerns detected | 7.23s | 37.0 tok/s | 18 | 1,299 / 110 | 18 (16 from hints) | 84% | prompt hint repeated | - |
| sahilchachra/LensVLM-9B-MXFP4 | concerns detected | 4.17s | 105 tok/s | 7.4 | 1,843 / 93 | 10 (10 from hints) | 100% | prompt hint repeated | - |
| mlx-community/FastVLM-0.5B-bf16 | major concerns: answer format | 2.87s | 367 tok/s | 2.2 | 353 / 44 | - | 100% of answer | labelled fields not detected; prompt hint repeated | - |
| mlx-community/gemma-3n-E4B-it-4bit | major concerns: answer format | 5.63s | 98.1 tok/s | 7.2 | 609 / 261 | - | 54% of answer | labelled fields not detected; unsupplied place name | - |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns: answer format | 3.41s | 106 tok/s | 6.7 | 2,211 / 19 | - | 0% of answer | control tokens visible; labelled fields not detected | candidate |
| mlx-community/MolmoPoint-8B-4bit | major concerns: answer format | 9.63s | 31.8 tok/s | 13 | 3,164 / 144 | - | 26% of answer | labelled fields not detected; unsupplied place name | - |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | major concerns: answer format | 51.01s | 25.0 tok/s | 25 | 4,418 / 1,000 | 1 (0 from hints) | - | control tokens visible; labelled fields not detected; cut off at token limit; role tokens visible; unsupplied place name | candidate |
| mlx-community/nanoLLaVA-1.5-4bit | major concerns: answer format | 2.08s | 372 tok/s | 1.8 | 349 / 131 | - | 25% | labelled fields not detected | - |
| mlx-community/SmolVLM-256M-Instruct-4bit | major concerns: answer format | 2.04s | 541 tok/s | 1.1 | 1,225 / 36 | - | 100% of answer | labelled fields not detected; prompt hint repeated | - |
| vikhyatk/moondream2 | major concerns: answer format | 3.02s | 171 tok/s | 4.8 | 1,038 / 37 | - | 100% of answer | labelled fields not detected; prompt hint repeated | - |
| mlx-community/granite-vision-3.2-2b-nvfp4 | major concerns: generation | 5.58s | 150 tok/s | 4.3 | 5,688 / 200 | 48 (14 from hints) | - | repeated text; stopped early: repeating; labelled fields not detected; duplicate keywords | candidate |
| mlx-community/Phi-3.5-vision-instruct-bf16 | major concerns: generation | 5.59s | 59.0 tok/s | 9.3 | 1,151 / 200 | 37 (16 from hints) | 50% | repeated text; stopped early: repeating; duplicate keywords | candidate |
| mlx-community/Qwen2-VL-2B-mlx | major concerns: generation | 50.35s | 124 tok/s | 9.4 | 16,577 / 1,000 | - | 9% | repeated text; control tokens visible; labelled fields not detected; cut off at token limit | candidate |

## Completed attempts requiring review

*History* dates each observation (a crash counts as one) over this model's
retained runs: 35 earlier retained runs from git
`HEAD:src/output/results.jsonl`, back to 2026-08-30; older runs were not read.
A run that did not attempt the model is skipped, and a report-only correction
of a run counts once. "Last N runs" and "N runs since" count consecutive runs
ending with this one; "first" is the earliest run read that showed it. The
table shows each model's longest-running observation; every observation's
dates are under *History by observation*.

| Model | Mechanical checks | History | Observed result | Evidence |
| --- | --- | --- | --- | --- |
| mlx-community/granite-vision-3.2-2b-nvfp4 | major concerns: generation | duplicate keywords: 7 runs since 2026-10-04; +3 more, newest back this run | Response repeats the same text; Generation was stopped early after sustained repeated output; Required labelled fields not detected: description; Duplicate keywords: norwich, trees | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-granite-vision-32-2b-nvfp4) |
| mlx-community/Phi-3.5-vision-instruct-bf16 | major concerns: generation | duplicate keywords: back this run; +2 more, newest not in earlier runs read | Response repeats the same text; Generation was stopped early after sustained repeated output; Duplicate keywords: historic architecture, stone architecture | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-phi-35-vision-instruct-bf16) |
| mlx-community/Qwen2-VL-2B-mlx | major concerns: generation | repeated text: 3 runs since 2026-10-10; +3 more, newest not in earlier runs read | Response repeats the same text; Unrecognised model control tokens remain visible; Required labelled fields not detected: keywords; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-qwen2-vl-2b-mlx) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns: answer format | all 2: 26+ runs since 2026-09-12 or earlier | Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description, keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit) |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | major concerns: answer format | cut off at token limit: 21 runs since 2026-09-18; +4 more, newest back this run | Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description; Response appears cut off at the token limit; Conversation-role control tokens remain visible; Names a place the prompt did not supply: Dusk | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-muse-glimmer-30b-optiq-4bit) |

<details>
<summary>History by observation</summary>

| Model | Observation | Persistence |
| --- | --- | --- |
| mlx-community/granite-vision-3.2-2b-nvfp4 | duplicate keywords | last 7 runs (since 2026-10-04) |
| mlx-community/granite-vision-3.2-2b-nvfp4 | labelled fields not detected | back this run, first 2026-09-25 or earlier |
| mlx-community/granite-vision-3.2-2b-nvfp4 | repeated text | not in earlier runs read |
| mlx-community/granite-vision-3.2-2b-nvfp4 | stopped early: repeating | not in earlier runs read |
| mlx-community/Phi-3.5-vision-instruct-bf16 | duplicate keywords | back this run, first 2026-09-11 |
| mlx-community/Phi-3.5-vision-instruct-bf16 | repeated text | not in earlier runs read |
| mlx-community/Phi-3.5-vision-instruct-bf16 | stopped early: repeating | not in earlier runs read |
| mlx-community/Qwen2-VL-2B-mlx | labelled fields not detected | not in earlier runs read |
| mlx-community/Qwen2-VL-2B-mlx | repeated text | last 3 runs (since 2026-10-10) |
| mlx-community/Qwen2-VL-2B-mlx | cut off at token limit | back this run, first 2026-10-10 |
| mlx-community/Qwen2-VL-2B-mlx | control tokens visible | not in earlier runs read |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | labelled fields not detected | last 26+ runs (since 2026-09-12 or earlier) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | control tokens visible | last 26+ runs (since 2026-09-12 or earlier) |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | labelled fields not detected | last 20 runs (since 2026-09-25), first 2026-09-06 |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | role tokens visible | last 7 runs (since 2026-10-04), first 2026-09-06 |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | cut off at token limit | last 21 runs (since 2026-09-18), first 2026-09-06 |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | control tokens visible | last 7 runs (since 2026-10-04), first 2026-09-13 |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | unsupplied place name | back this run, first 2026-09-27 |

</details>

## Completions without detected concerns

26 completions without detected concerns; 16 more completed with prompt-compliance observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 6,213 x 9,320 pixels, 52,096,760 bytes
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: seed:* 0
- *Trust remote code:* true
- *check_models version:* 0.17.46
- *check_models revision:* 3d1b38a4f8bdf2152010e5d5ec4e6e8cd0ba7782
- *check_models source dirty:* false
- *mlx-vlm:* 0.7.7
- *mlx-vlm source revision:* 952d4f6bc65bd5095e74abe71c78038764ee08aa
- *mlx:* 0.32.4.dev20261010+06eb7483f
- *mlx source revision:* 06eb7483f
- *transformers:* 5.19.0
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
