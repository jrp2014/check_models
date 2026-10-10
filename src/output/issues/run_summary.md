# mlx-vlm compatibility findings across 53 cached vision-language models

**For mlx-vlm maintainers:** no crashes need action; 8 other results need
reproducing with mlx-vlm alone before anything is reported.

**Completed models:** 30 with no mechanical observations, 15 with
prompt-compliance observations only (labels, form or copied hint text), and 8
with observations mlx-vlm can produce (such as repetition, missing final
answers or visible control tokens); only those 8 are candidates for native
reproduction.

## Run summary

- *Run started:* 2026-10-10 23:35:53 BST
- *Run finished:* 2026-10-10 23:49:28 BST
- *Run duration:* 13m 34s
- *Time by phase:* generation 663s, model load 100s, prompt prep 34s, cleanup
  7s, outside the model loop 18s
- *Evaluation lane:* assisted
- *Prompt hints:* the image's description and keyword hints were included in
  the prompt, so field content may be copied from them rather than seen
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Input image:* JPEG, 5,800 x 8,389 pixels (48.7 MP), 40.8 MB
- *Models attempted:* 53
- *Sampling settings:* checkpoint generation_config.json values for 23 of 53
  completed models where the command line left them unset; harness defaults
  elsewhere
- *Completed:* 53
- *Crashed:* 0
- *Indeterminate:* 0
- *Crashes requiring action:* 0
- *Other results requiring review:* 8
- *Reached token limit:* 5 (5 with incomplete output)
- *Stopped early for repetition:* 3

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
- Capture date/time: 2026-10-10 17:21:00 UTC+01:00
- GPS: 52.629112°N, 1.288265°E

Descriptive hints:
- Description hint: A street-level architectural view of the exterior of The Shopkeeper Store, located at No. 76, featuring a traditional black-painted storefront adorned with gold detailing on the ground floor, a red brick middle story with three sash windows, and twin slate-grey gabled dormers on the upper level.
- Keyword hints: Adobe Stock, Any Vision, Chimney, Entrance, Europe, Gable, Objects, Red brick, Roof, Sash Window, Shopfront, Signage, United Kingdom, architectural detail, architecture, boutique, brick building, brick wall, british, building exterior

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

- *Baseline:* 505bca0c:src/output/results.jsonl
- *Baseline run timestamp:* 2026-10-09 23:32:24 BST
- *Baseline check_models:* 0.17.41 @ d04168248
- *Baseline mlx:* 0.32.4.dev20261009+99f109b56 @ 99f109b56
- *Baseline mlx-vlm:* 0.7.7 @ 952d4f6bc
- *Baseline transformers:* 5.19.0
- *Baseline python:* 3.14.7
- *Baseline hardware:* Apple M5 Max, 40 GPU cores, 128.0 GB RAM

**Not directly comparable** — output, quality and performance comparisons are
withheld because the runs differ in: prompt differs; image differs (sha256
9f0e8d795514… → 95f6022daf7b…). The roster, revisions and upstream changes
below are facts about the runs and are still shown.

Run environment: baseline run on battery power for 50 of 50 models.

- New this run (no baseline): `TechnoBaptist/Ternary-Bonsai-2-27B-mlx-2bit`,
  `mlx-community/AREX-2-4bit`, `sahilchachra/LensVLM-9B-MXFP4`

<details>
<summary>mlx: 7 upstream commit(s) since the baseline (99f109b56..06eb7483f)</summary>

- 06eb7483f Fix lost rank output in the distributed launcher (#4615)
- 19ffe23bc Add stft and istft to mlx.core.fft (#3639)
- af9a59240 Fix direct CPU conv writing past output (#4668)
- 0fdcdc0e2 Fix missing stream in layer_norm and sdpa fallbacks (#4663)
- 0ab821b1f Fix conv input gradient shape with uneven or large padding (#4662)
- 9ab799919 [CUDA] Only count consecutive misses in the LRUCache thrashing
  check (#4661)
- c06ce82f1 Replace "pull_request_target" event with "pull_request" in
  update_bypass_list workflow (#4667)

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
means an answer was generated but missed the requested form. Prompt tokens
include the image tokens, which drive prefill time; output tokens are the
tokens generated (limit 1,000). Keywords are counted from the answer's
Keywords field, with how many appear verbatim in the prompt's keyword hints.
Hint text is the percent of the description's words lying in four-word runs
copied from the prompt's description hint (of the whole answer when no field
is labelled), rounded down; 80% or more is reported as a repeated prompt hint.

| Model | Mechanical checks | Total | Gen tok/s | Peak GB | Prompt / output tok | Keywords | Hint text | Observed | Native repro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | no concerns detected | 1.77s | 484 tok/s | 1.9 | 2,144 / 83 | 14 (10 from hints) | 19% | none | - |
| mlx-community/AREX-2-4bit | no concerns detected | 56.81s | 30.7 tok/s | 21 | 16,586 / 137 | 20 (20 from hints) | 62% | none | - |
| mlx-community/aya-vision-8b-4bit | no concerns detected | 4.76s | 101 tok/s | 6.5 | 2,123 / 125 | 13 (13 from hints) | 45% | none | - |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | no concerns detected | 10.69s | 30.1 tok/s | 23 | 2,498 / 134 | 20 (10 from hints) | 28% | none | - |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | no concerns detected | 6.79s | 49.6 tok/s | 28 | 626 / 95 | 14 (12 from hints) | 44% | none | - |
| mlx-community/gemma-3-27b-it-qat-4bit | no concerns detected | 9.64s | 29.9 tok/s | 17 | 625 / 161 | 20 (12 from hints) | 31% | none | - |
| mlx-community/gemma-4-12B-it-4bit | no concerns detected | 5.13s | 60.0 tok/s | 7.7 | 630 / 106 | 16 (10 from hints) | 0% | none | - |
| mlx-community/gemma-4-26b-a4b-it-4bit | no concerns detected | 4.93s | 110 tok/s | 16 | 630 / 104 | 16 (13 from hints) | 21% | none | - |
| mlx-community/gemma-4-31b-it-4bit | no concerns detected | 8.92s | 26.3 tok/s | 20 | 630 / 109 | 16 (15 from hints) | 25% | none | - |
| mlx-community/gemma-4-e4b-it-4bit | no concerns detected | 3.58s | 125 tok/s | 6.0 | 626 / 104 | 15 (6 from hints) | 0% | none | - |
| mlx-community/GLM-4.6V-nvfp4 | no concerns detected | 34.07s | 44.1 tok/s | 78 | 6,445 / 145 | 18 (9 from hints) | 64% | none | - |
| mlx-community/granite-4.0-3b-vision-4bit | no concerns detected | 3.33s | 174 tok/s | 4.6 | 1,420 / 160 | 26 (11 from hints) | 66% | none | - |
| mlx-community/Idefics3-8B-Llama3-bf16 | no concerns detected | 9.93s | 34.4 tok/s | 18 | 2,641 / 200 | 20 (20 from hints) | 41% | none | - |
| mlx-community/InternVL3-14B-4bit | no concerns detected | 6.11s | 56.3 tok/s | 10 | 2,142 / 134 | 19 (19 from hints) | 50% | none | - |
| mlx-community/InternVL3-8B-bf16 | no concerns detected | 6.33s | 36.8 tok/s | 17 | 2,142 / 104 | 15 (8 from hints) | 25% | none | - |
| mlx-community/MiniCPM-o-4_5-4bit | no concerns detected | 2.98s | 104 tok/s | 7.0 | 420 / 94 | 16 (12 from hints) | 16% | none | - |
| mlx-community/MiniCPM-V-4.6-4bit | no concerns detected | 4.91s | 300 tok/s | 3.2 | 963 / 826 | 19 (9 from hints) | 18% | none | - |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | no concerns detected | 8.28s | 65.7 tok/s | 13 | 3,031 / 244 | 20 (1 from hints) | 17% | none | - |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | no concerns detected | 3.87s | 184 tok/s | 8.1 | 3,030 / 166 | 14 (1 from hints) | 0% | none | - |
| mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit | no concerns detected | 5.45s | 101 tok/s | 25 | 1,330 / 127 | 19 (16 from hints) | 69% | none | - |
| mlx-community/Phi-3.5-vision-instruct-bf16 | no concerns detected | 4.71s | 56.0 tok/s | 9.3 | 1,169 / 147 | 22 (7 from hints) | 0% | none | - |
| mlx-community/pixtral-12b-8bit | no concerns detected | 8.70s | 34.7 tok/s | 16 | 3,297 / 133 | 24 (20 from hints) | 45% | none | - |
| mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit | no concerns detected | 24.63s | 71.1 tok/s | 26 | 12,814 / 141 | 18 (16 from hints) | 0% | none | - |
| mlx-community/Qwen3-VL-8B-Instruct-4bit | no concerns detected | 37.25s | 68.9 tok/s | 11 | 16,570 / 87 | 10 (8 from hints) | 13% | none | - |
| mlx-community/Qwen3.5-35B-A3B-4bit | no concerns detected | 38.66s | 108 tok/s | 25 | 16,586 / 140 | 18 (11 from hints) | 18% | none | - |
| mlx-community/Qwen3.8-27B-nvfp4 | no concerns detected | 59.46s | 29.0 tok/s | 21 | 16,586 / 128 | 16 (11 from hints) | 10% | none | - |
| mlx-community/Step-3.7-Flash-oQ3e | no concerns detected | 48.27s | 47.4 tok/s | 92 | 3,522 / 117 | 18 (18 from hints) | 41% | none | - |
| nativ-community/Mage-VL-OptiQ-4bit | no concerns detected | 5.34s | 119 tok/s | 5.4 | 4,188 / 198 | 19 (11 from hints) | 26% | none | - |
| nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit | no concerns detected | 8.22s | 35.1 tok/s | 18 | 1,353 / 136 | 18 (8 from hints) | 31% | none | - |
| TechnoBaptist/Ternary-Bonsai-2-27B-mlx-2bit | no concerns detected | 67.45s | 36.2 tok/s | 17 | 16,586 / 182 | 18 (14 from hints) | 29% | none | - |
| mlx-community/GLM-4.6V-Flash-4bit | concerns detected | 11.08s | 77.7 tok/s | 8.7 | 6,445 / 181 | 23 (15 from hints) | 55% | duplicate keywords | - |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit | concerns detected | 2.99s | 211 tok/s | 4.0 | 2,136 / 112 | 19 (19 from hints) | 82% | prompt hint repeated; unsupplied place name | - |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | concerns detected | 13.42s | 20.6 tok/s | 15 | 329 / 194 | 30 (1 from hints) | 0% | duplicate keywords | - |
| mlx-community/Molmo2-8B-4bit | concerns detected | 5.38s | 72.5 tok/s | 8.3 | 1,356 / 187 | 25 (14 from hints) | 33% | duplicate keywords | - |
| mlx-community/North-Micro-Vision-Instruct-4bit | concerns detected | 4.93s | 209 tok/s | 3.9 | 4,093 / 126 | 19 (19 from hints) | 96% | prompt hint repeated | - |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | concerns detected | 36.59s | 84.9 tok/s | 23 | 16,570 / 133 | 19 (9 from hints) | 33% | unsupplied place name | - |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | concerns detected | 3.07s | 121 tok/s | 5.6 | 1,463 / 102 | 20 (20 from hints) | 86% | prompt hint repeated | - |
| nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | concerns detected | 8.47s | 152 tok/s | 23 | 3,681 / 117 | 17 (9 from hints) | 25% | duplicate keywords | - |
| sahilchachra/LensVLM-9B-MXFP4 | concerns detected | 4.44s | 104 tok/s | 7.5 | 1,872 / 136 | 20 (20 from hints) | 100% | prompt hint repeated | - |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | major concerns: answer format | 15.65s | 81.5 tok/s | 19 | 1,669 / 1,000 | 36 (13 from hints) | 10% | cut off at token limit; duplicate keywords | - |
| mlx-community/FastVLM-0.5B-bf16 | major concerns: answer format | 2.50s | 367 tok/s | 2.1 | 363 / 30 | - | 0% of answer | labelled fields not detected | - |
| mlx-community/gemma-3n-E4B-it-4bit | major concerns: answer format | 7.27s | 83.5 tok/s | 7.2 | 624 / 352 | - | 55% of answer | labelled fields not detected | - |
| mlx-community/granite-vision-3.2-2b-nvfp4 | major concerns: answer format | 5.70s | 134 tok/s | 4.4 | 5,877 / 85 | 11 (5 from hints) | - | labelled fields not detected; duplicate keywords | - |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns: answer format | 3.30s | 102 tok/s | 6.7 | 2,234 / 26 | - | 0% of answer | control tokens visible; labelled fields not detected | candidate |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | major concerns: answer format | 51.69s | 25.0 tok/s | 25 | 4,458 / 1,000 | 11 (0 from hints) | - | control tokens visible; labelled fields not detected; cut off at token limit; role tokens visible | candidate |
| mlx-community/SmolVLM-256M-Instruct-4bit | major concerns: answer format | 1.93s | 506 tok/s | 1.1 | 1,242 / 80 | - | 11% of answer | labelled fields not detected | - |
| vikhyatk/moondream2 | major concerns: answer format | 2.64s | 160 tok/s | 4.8 | 1,041 / 57 | - | - | labelled fields not detected | - |
| mlx-community/InternVL3_5-1B-4bit | major concerns: generation | 2.23s | 410 tok/s | 2.1 | 2,145 / 200 | 49 (9 from hints) | 33% | repeated text; stopped early: repeating; duplicate keywords | candidate |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit | major concerns: generation | 19.06s | 65.6 tok/s | 20 | 1,326 / 1,000 | - | 22% | labelled fields not detected; cut off at token limit; incomplete thinking block | candidate |
| mlx-community/MolmoPoint-8B-4bit | major concerns: generation | 11.28s | 31.5 tok/s | 13 | 3,174 / 200 | 37 (1 from hints) | 40% | repeated text; stopped early: repeating; duplicate keywords | candidate |
| mlx-community/nanoLLaVA-1.5-4bit | major concerns: generation | 2.20s | 328 tok/s | 1.8 | 359 / 200 | - | 44% | repeated text; stopped early: repeating; labelled fields not detected | candidate |
| mlx-community/Qwen2-VL-2B-mlx | major concerns: generation | 42.65s | 123 tok/s | 9.4 | 16,581 / 1,000 | 12 (6 from hints) | 0% | repeated text; cut off at token limit | candidate |
| mlx-community/X-Reasoner-7B-8bit | major concerns: generation | 37.90s | 52.8 tok/s | 14 | 16,581 / 1,000 | 223 (14 from hints) | 28% | repeated text; cut off at token limit; duplicate keywords | candidate |

## Observation clusters

Observation signatures shared by two or more results requiring review.

| Observed result | Models |
| --- | --- |
| Response repeats the same text; Generation was stopped early after sustained repeated output; Repeated keyword entries | [mlx-community/InternVL3_5-1B-4bit](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-internvl35-1b-4bit), [mlx-community/MolmoPoint-8B-4bit](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-molmopoint-8b-4bit) |

## Completed attempts requiring review

*History* dates each observation (a crash counts as one) over this model's
retained runs: 33 earlier retained runs from git
`HEAD:src/output/results.jsonl`, back to 2026-08-30; older runs were not read.
A run that did not attempt the model is skipped, and a report-only correction
of a run counts once. "Last N runs" and "N runs since" count consecutive runs
ending with this one; "first" is the earliest run read that showed it. The
table shows each model's longest-running observation; every observation's
dates are under *History by observation*.

| Model | Mechanical checks | History | Observed result | Evidence |
| --- | --- | --- | --- | --- |
| mlx-community/InternVL3_5-1B-4bit | major concerns: generation | all 3: 5 runs since 2026-10-04 | Response repeats the same text; Generation was stopped early after sustained repeated output; Duplicate keywords: signage, photo | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-internvl35-1b-4bit) |
| mlx-community/MolmoPoint-8B-4bit | major concerns: generation | all 3: not in earlier runs read | Response repeats the same text; Generation was stopped early after sustained repeated output; Duplicate keywords: united kingdom | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-molmopoint-8b-4bit) |
| mlx-community/nanoLLaVA-1.5-4bit | major concerns: generation | labelled fields not detected: 28+ runs since 2026-09-06 or earlier; +2 more, newest not in earlier runs read | Response repeats the same text; Generation was stopped early after sustained repeated output; Required labelled fields not detected: keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-nanollava-15-4bit) |
| mlx-community/Qwen2-VL-2B-mlx | major concerns: generation | all 2: not in earlier runs read | Response repeats the same text; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-qwen2-vl-2b-mlx) |
| mlx-community/X-Reasoner-7B-8bit | major concerns: generation | duplicate keywords: 18 runs since 2026-09-25; +2 more, newest back this run | Response repeats the same text; Response appears cut off at the token limit; Duplicate keywords: closed sign, boutique, traditional, entrance, building, closed, closed boutique, closed building front door, closed shop front door, closed store front door, closed boutique front door, closed building entrance door, closed shop entrance door, closed store entrance door, closed boutique entrance door, closed building exterior door, closed shop exterior door, closed store exterior door, closed boutique exterior door | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-x-reasoner-7b-8bit) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns: answer format | all 2: 24+ runs since 2026-09-12 or earlier | Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description, keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit) |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | major concerns: answer format | cut off at token limit: 19 runs since 2026-09-18; +3 more, newest 5 runs since 2026-10-04 | Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description; Response appears cut off at the token limit; Conversation-role control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-muse-glimmer-30b-optiq-4bit) |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit | major concerns: generation | all 3: back this run | Required labelled fields not detected: keywords; Response appears cut off at the token limit; Internal reasoning block appears incomplete | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-8bit) |

<details>
<summary>History by observation</summary>

| Model | Observation | Persistence |
| --- | --- | --- |
| mlx-community/InternVL3_5-1B-4bit | duplicate keywords | last 5 runs (since 2026-10-04) |
| mlx-community/InternVL3_5-1B-4bit | repeated text | last 5 runs (since 2026-10-04) |
| mlx-community/InternVL3_5-1B-4bit | stopped early: repeating | last 5 runs (since 2026-10-04) |
| mlx-community/MolmoPoint-8B-4bit | duplicate keywords | not in earlier runs read |
| mlx-community/MolmoPoint-8B-4bit | repeated text | not in earlier runs read |
| mlx-community/MolmoPoint-8B-4bit | stopped early: repeating | not in earlier runs read |
| mlx-community/nanoLLaVA-1.5-4bit | labelled fields not detected | last 28+ runs (since 2026-09-06 or earlier) |
| mlx-community/nanoLLaVA-1.5-4bit | repeated text | last 5 runs (since 2026-10-04) |
| mlx-community/nanoLLaVA-1.5-4bit | stopped early: repeating | not in earlier runs read |
| mlx-community/Qwen2-VL-2B-mlx | repeated text | not in earlier runs read |
| mlx-community/Qwen2-VL-2B-mlx | cut off at token limit | not in earlier runs read |
| mlx-community/X-Reasoner-7B-8bit | duplicate keywords | last 18 runs (since 2026-09-25), first 2026-09-06 |
| mlx-community/X-Reasoner-7B-8bit | repeated text | last 5 runs (since 2026-10-04), first 2026-08-30 |
| mlx-community/X-Reasoner-7B-8bit | cut off at token limit | back this run, first 2026-08-30 |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | labelled fields not detected | last 24+ runs (since 2026-09-12 or earlier) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | control tokens visible | last 24+ runs (since 2026-09-12 or earlier) |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | labelled fields not detected | last 18 runs (since 2026-09-25), first 2026-09-06 |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | role tokens visible | last 5 runs (since 2026-10-04), first 2026-09-06 |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | cut off at token limit | last 19 runs (since 2026-09-18), first 2026-09-06 |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | control tokens visible | last 5 runs (since 2026-10-04), first 2026-09-13 |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit | labelled fields not detected | back this run, first 2026-09-27 |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit | incomplete thinking block | back this run, first 2026-09-27 |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit | cut off at token limit | back this run, first 2026-09-13 |

</details>

## Completions without detected concerns

30 completions without detected concerns; 15 more completed with prompt-compliance observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 5,800 x 8,389 pixels, 40,750,483 bytes
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: seed:* 0
- *Trust remote code:* true
- *check_models version:* 0.17.42
- *check_models revision:* 505bca0c63fb89de384a1d1421e16f0de00d3e06
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
