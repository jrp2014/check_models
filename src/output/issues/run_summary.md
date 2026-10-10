# mlx-vlm compatibility findings across 47 cached vision-language models

**For mlx-vlm maintainers:** no crashes need action; 8 other results need
reproducing with mlx-vlm alone before anything is reported.

**Completed models:** 29 with no mechanical observations, 10 with
prompt-compliance observations only (labels, form or copied hint text), and 8
with observations mlx-vlm can produce (such as repetition, missing final
answers or visible control tokens); only those 8 are candidates for native
reproduction.

## Run summary

- *Run started:* 2026-10-11 00:22:15 BST
- *Run finished:* 2026-10-11 00:35:30 BST
- *Run duration:* 13m 15s
- *Time by phase:* generation 627s, model load 111s, prompt prep 37s, cleanup
  7s, outside the model loop 19s
- *Evaluation lane:* assisted
- *Prompt hints:* the image's description and keyword hints were included in
  the prompt, so field content may be copied from them rather than seen
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Input image:* JPEG, 9,984 x 6,656 pixels (66.5 MP), 33.5 MB
- *Models attempted:* 47
- *Sampling settings:* checkpoint generation_config.json values for 20 of 47
  completed models where the command line left them unset; harness defaults
  elsewhere
- *Completed:* 47
- *Crashed:* 0
- *Indeterminate:* 0
- *Crashes requiring action:* 0
- *Other results requiring review:* 8
- *Reached token limit:* 2 (2 with incomplete output)
- *Stopped early for repetition:* 4

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
- Capture date/time: 2026-10-10 16:41:54 UTC+01:00
- GPS: 52.628900°N, 1.292500°E

Descriptive hints:
- Description hint: A bronze lion sculpture by Alfred Hardiman stands outside City Hall overlooking the Market Place, with the historic 15th-century flint Norwich Guildhall visible in the background in Norwich, Norfolk, England.
- Keyword hints: Adobe Stock, Any Vision, Blue sky, British heritage, Car, East Anglia, England, Europe, Gothic Architecture, Guildhall, Historic Landmark, Lion, Norfolk, Norwich, Norwich Guildhall, Pedestrian, Sculpture, Sightseeing, Statue, Street Scene

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

- *Baseline:* 84f7be8c:src/output/results.jsonl
- *Baseline run timestamp:* 2026-10-10 23:49:31 BST
- *Baseline check_models:* 0.17.42 @ 505bca0c6
- *Baseline mlx:* 0.32.4.dev20261010+06eb7483f @ 06eb7483f
- *Baseline mlx-vlm:* 0.7.7 @ 952d4f6bc
- *Baseline transformers:* 5.19.0
- *Baseline python:* 3.14.7
- *Baseline hardware:* Apple M5 Max, 40 GPU cores, 128.0 GB RAM
- *Earlier retained runs on this mlx version:* 1 on
  0.32.4.dev20261010+06eb7483f

**Not directly comparable** — output, quality and performance comparisons are
withheld because the runs differ in: prompt differs; image differs (sha256
95f6022daf7b… → 712a5faa6eab…). The roster, revisions and upstream changes
below are facts about the runs and are still shown.

- In baseline, no longer in the cache: `mlx-community/AREX-2-4bit`,
  `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`,
  `mlx-community/Idefics3-8B-Llama3-bf16`, `mlx-community/InternVL3-8B-bf16`,
  `mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit`,
  `mlx-community/gemma-4-31b-it-4bit`

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
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | no concerns detected | 2.12s | 409 tok/s | 1.9 | 2,128 / 90 | 10 (7 from hints) | 18% | none | - |
| mlx-community/aya-vision-8b-4bit | no concerns detected | 5.15s | 96.1 tok/s | 6.5 | 2,100 / 119 | 17 (11 from hints) | 17% | none | - |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | no concerns detected | 6.55s | 77.5 tok/s | 28 | 605 / 87 | 15 (14 from hints) | 79% | none | - |
| mlx-community/gemma-3-27b-it-qat-4bit | no concerns detected | 12.50s | 19.6 tok/s | 17 | 604 / 140 | 19 (12 from hints) | 10% | none | - |
| mlx-community/gemma-4-26b-a4b-it-4bit | no concerns detected | 5.52s | 99.4 tok/s | 16 | 609 / 106 | 16 (14 from hints) | 33% | none | - |
| mlx-community/gemma-4-e4b-it-4bit | no concerns detected | 3.83s | 121 tok/s | 5.9 | 605 / 87 | 15 (7 from hints) | 16% | none | - |
| mlx-community/GLM-4.6V-Flash-4bit | no concerns detected | 12.12s | 76.4 tok/s | 8.7 | 6,460 / 99 | 9 (8 from hints) | 52% | none | - |
| mlx-community/GLM-4.6V-nvfp4 | no concerns detected | 38.60s | 37.0 tok/s | 78 | 6,460 / 108 | 17 (13 from hints) | 48% | none | - |
| mlx-community/InternVL3-14B-4bit | no concerns detected | 6.77s | 53.5 tok/s | 10 | 2,124 / 123 | 21 (19 from hints) | 0% | none | - |
| mlx-community/MiniCPM-o-4_5-4bit | no concerns detected | 3.25s | 105 tok/s | 7.0 | 402 / 91 | 12 (10 from hints) | 26% | none | - |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | no concerns detected | 7.10s | 64.4 tok/s | 13 | 2,935 / 130 | 16 (8 from hints) | 25% | none | - |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | no concerns detected | 5.20s | 169 tok/s | 7.8 | 2,934 / 128 | 14 (4 from hints) | 16% | none | - |
| mlx-community/Molmo2-8B-4bit | no concerns detected | 6.38s | 65.9 tok/s | 8.1 | 1,535 / 181 | 19 (14 from hints) | 19% | none | - |
| mlx-community/MolmoPoint-8B-4bit | no concerns detected | 10.83s | 29.8 tok/s | 13 | 3,137 / 129 | 17 (12 from hints) | 0% | none | - |
| mlx-community/North-Micro-Vision-Instruct-4bit | no concerns detected | 6.22s | 143 tok/s | 3.9 | 4,095 / 171 | 20 (20 from hints) | 24% | none | - |
| mlx-community/Phi-3.5-vision-instruct-bf16 | no concerns detected | 5.98s | 50.4 tok/s | 9.3 | 1,149 / 128 | 17 (12 from hints) | 59% | none | - |
| mlx-community/pixtral-12b-8bit | no concerns detected | 7.44s | 40.2 tok/s | 16 | 3,125 / 110 | 20 (20 from hints) | 53% | none | - |
| mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit | no concerns detected | 30.73s | 50.7 tok/s | 26 | 12,801 / 109 | 18 (14 from hints) | 45% | none | - |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | no concerns detected | 45.59s | 75.5 tok/s | 23 | 16,558 / 161 | 20 (14 from hints) | 18% | none | - |
| mlx-community/Qwen3-VL-8B-Instruct-4bit | no concerns detected | 45.80s | 62.3 tok/s | 11 | 16,558 / 125 | 18 (15 from hints) | 7% | none | - |
| mlx-community/Qwen3.5-35B-A3B-4bit | no concerns detected | 47.62s | 49.1 tok/s | 25 | 16,574 / 148 | 18 (9 from hints) | 35% | none | - |
| mlx-community/Qwen3.8-27B-nvfp4 | no concerns detected | 68.19s | 27.6 tok/s | 21 | 16,574 / 124 | 17 (11 from hints) | 12% | none | - |
| mlx-community/Step-3.7-Flash-oQ3e | no concerns detected | 52.59s | 48.7 tok/s | 92 | 3,502 / 122 | 20 (20 from hints) | 69% | none | - |
| mlx-community/X-Reasoner-7B-8bit | no concerns detected | 19.84s | 52.8 tok/s | 14 | 16,569 / 116 | 21 (13 from hints) | 37% | none | - |
| nativ-community/Mage-VL-OptiQ-4bit | no concerns detected | 5.39s | 126 tok/s | 5.4 | 4,221 / 157 | 17 (15 from hints) | 37% | none | - |
| nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit | no concerns detected | 15.04s | 13.4 tok/s | 18 | 1,281 / 133 | 18 (10 from hints) | 44% | none | - |
| nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | no concerns detected | 10.26s | 122 tok/s | 23 | 3,636 / 113 | 19 (13 from hints) | 64% | none | - |
| sahilchachra/LensVLM-9B-MXFP4 | no concerns detected | 5.77s | 101 tok/s | 7.5 | 1,886 / 223 | 20 (20 from hints) | 77% | none | - |
| TechnoBaptist/Ternary-Bonsai-2-27B-mlx-2bit | no concerns detected | 68.57s | 32.5 tok/s | 17 | 16,574 / 125 | 17 (16 from hints) | 0% | none | - |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | concerns detected | 11.43s | 93.4 tok/s | 19 | 1,648 / 640 | 52 (16 from hints) | 40% | duplicate keywords | - |
| mlx-community/gemma-4-12B-it-4bit | concerns detected | 5.20s | 58.6 tok/s | 7.6 | 609 / 108 | 17 (9 from hints) | 41% | duplicate keywords | - |
| mlx-community/granite-4.0-3b-vision-4bit | concerns detected | 3.11s | 170 tok/s | 4.7 | 1,390 / 91 | 17 (13 from hints) | 100% | prompt hint repeated | - |
| mlx-community/granite-vision-3.2-2b-nvfp4 | concerns detected | 5.61s | 146 tok/s | 4.2 | 5,595 / 230 | 19 (10 from hints) | 38% | duplicate keywords | - |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit | concerns detected | 4.23s | 201 tok/s | 4.0 | 2,120 / 93 | 15 (10 from hints) | 87% | prompt hint repeated | - |
| mlx-community/FastVLM-0.5B-bf16 | major concerns: answer format | 5.00s | 305 tok/s | 2.1 | 345 / 42 | - | 100% of answer | labelled fields not detected; prompt hint repeated | - |
| mlx-community/gemma-3n-E4B-it-4bit | major concerns: answer format | 6.25s | 65.4 tok/s | 7.0 | 603 / 186 | - | 23% of answer | labelled fields not detected | - |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns: answer format | 3.44s | insufficient sample | 6.7 | 2,205 / 15 | - | 0% of answer | control tokens visible; labelled fields not detected | candidate |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | major concerns: answer format | 57.24s | 22.7 tok/s | 25 | 4,413 / 1,000 | 32 (3 from hints) | - | control tokens visible; labelled fields not detected; cut off at token limit; role tokens visible; duplicate keywords | candidate |
| mlx-community/nanoLLaVA-1.5-4bit | major concerns: answer format | 2.51s | 200 tok/s | 1.8 | 341 / 162 | - | 4% | labelled fields not detected | - |
| mlx-community/SmolVLM-256M-Instruct-4bit | major concerns: answer format | 2.92s | 315 tok/s | 1.1 | 1,218 / 45 | - | 87% of answer | labelled fields not detected; prompt hint repeated | - |
| vikhyatk/moondream2 | major concerns: answer format | 7.03s | 165 tok/s | 4.8 | 1,034 / 36 | - | 100% of answer | labelled fields not detected; prompt hint repeated | - |
| mlx-community/InternVL3_5-1B-4bit | major concerns: generation | 3.67s | 225 tok/s | 2.1 | 2,127 / 200 | 47 (12 from hints) | 36% | repeated text; stopped early: repeating; duplicate keywords | candidate |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit | major concerns: generation | 22.18s | 55.7 tok/s | 20 | 1,334 / 1,000 | - | 4% of answer | labelled fields not detected; cut off at token limit; incomplete thinking block | candidate |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | major concerns: generation | 17.72s | 15.6 tok/s | 15 | 311 / 200 | 44 (8 from hints) | 0% | repeated text; stopped early: repeating; duplicate keywords | candidate |
| mlx-community/MiniCPM-V-4.6-4bit | major concerns: generation | 3.64s | 208 tok/s | 3.1 | 943 / 124 | 21 (14 from hints) | 0% | incomplete thinking block | candidate |
| mlx-community/Qwen2-VL-2B-mlx | major concerns: generation | 50.47s | 113 tok/s | 9.4 | 16,569 / 200 | 39 (9 from hints) | 30% | repeated text; stopped early: repeating; duplicate keywords | candidate |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | major concerns: generation | 4.00s | 125 tok/s | 5.6 | 1,439 / 200 | - | 0% of answer | repeated text; stopped early: repeating; labelled fields not detected | candidate |

## Observation clusters

Observation signatures shared by two or more results requiring review.

| Observed result | Models |
| --- | --- |
| Response repeats the same text; Generation was stopped early after sustained repeated output; Repeated keyword entries | [mlx-community/InternVL3_5-1B-4bit](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-internvl35-1b-4bit), [mlx-community/Llama-3.2-11B-Vision-Instruct-8bit](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llama-32-11b-vision-instruct-8bit), [mlx-community/Qwen2-VL-2B-mlx](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-qwen2-vl-2b-mlx) |

## Completed attempts requiring review

*History* dates each observation (a crash counts as one) over this model's
retained runs: 34 earlier retained runs from git
`HEAD:src/output/results.jsonl`, back to 2026-08-30; older runs were not read.
A run that did not attempt the model is skipped, and a report-only correction
of a run counts once. "Last N runs" and "N runs since" count consecutive runs
ending with this one; "first" is the earliest run read that showed it. The
table shows each model's longest-running observation; every observation's
dates are under *History by observation*.

| Model | Mechanical checks | History | Observed result | Evidence |
| --- | --- | --- | --- | --- |
| mlx-community/InternVL3_5-1B-4bit | major concerns: generation | all 3: 6 runs since 2026-10-04 | Response repeats the same text; Generation was stopped early after sustained repeated output; Duplicate keywords: norwich guildhall, lion, norwich, southeast europe, southeast | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-internvl35-1b-4bit) |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | major concerns: generation | duplicate keywords: 6 runs since 2026-10-04; +2 more, newest back this run | Response repeats the same text; Generation was stopped early after sustained repeated output; Duplicate keywords: statue of a lion, lion statue | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llama-32-11b-vision-instruct-8bit) |
| mlx-community/Qwen2-VL-2B-mlx | major concerns: generation | repeated text: 2 runs since 2026-10-10; +2 more, newest back this run | Response repeats the same text; Generation was stopped early after sustained repeated output; Duplicate keywords: medieval town, medieval town square | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-qwen2-vl-2b-mlx) |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | major concerns: generation | all 3: back this run | Response repeats the same text; Generation was stopped early after sustained repeated output; Required labelled fields not detected: title, description, keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-smolvlm2-22b-instruct-mlx) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns: answer format | all 2: 25+ runs since 2026-09-12 or earlier | Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description, keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit) |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | major concerns: answer format | cut off at token limit: 20 runs since 2026-09-18; +4 more, newest back this run | Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description; Response appears cut off at the token limit; Conversation-role control tokens remain visible; Duplicate keywords: correct conflicts, and add important visible details prefer image evidence when a hint conflicts, and omit uncertain details | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-muse-glimmer-30b-optiq-4bit) |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit | major concerns: generation | all 3: 2 runs since 2026-10-10 | Required labelled fields not detected: title, description, keywords; Response appears cut off at the token limit; Internal reasoning block appears incomplete | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-8bit) |
| mlx-community/MiniCPM-V-4.6-4bit | major concerns: generation | incomplete thinking block: back this run | Internal reasoning block appears incomplete | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-minicpm-v-46-4bit) |

<details>
<summary>History by observation</summary>

| Model | Observation | Persistence |
| --- | --- | --- |
| mlx-community/InternVL3_5-1B-4bit | duplicate keywords | last 6 runs (since 2026-10-04) |
| mlx-community/InternVL3_5-1B-4bit | repeated text | last 6 runs (since 2026-10-04) |
| mlx-community/InternVL3_5-1B-4bit | stopped early: repeating | last 6 runs (since 2026-10-04) |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | duplicate keywords | last 6 runs (since 2026-10-04), first 2026-09-25 |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | repeated text | back this run, first 2026-09-18 |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | stopped early: repeating | not in earlier runs read |
| mlx-community/Qwen2-VL-2B-mlx | duplicate keywords | back this run, first 2026-10-04 or earlier |
| mlx-community/Qwen2-VL-2B-mlx | repeated text | last 2 runs (since 2026-10-10) |
| mlx-community/Qwen2-VL-2B-mlx | stopped early: repeating | back this run, first 2026-10-04 or earlier |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | labelled fields not detected | back this run, first 2026-08-30 |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | repeated text | back this run, first 2026-08-30 |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | stopped early: repeating | back this run, first 2026-08-30 |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | labelled fields not detected | last 25+ runs (since 2026-09-12 or earlier) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | control tokens visible | last 25+ runs (since 2026-09-12 or earlier) |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | duplicate keywords | back this run, first 2026-09-06 |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | labelled fields not detected | last 19 runs (since 2026-09-25), first 2026-09-06 |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | role tokens visible | last 6 runs (since 2026-10-04), first 2026-09-06 |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | cut off at token limit | last 20 runs (since 2026-09-18), first 2026-09-06 |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | control tokens visible | last 6 runs (since 2026-10-04), first 2026-09-13 |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit | labelled fields not detected | last 2 runs (since 2026-10-10), first 2026-09-27 |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit | incomplete thinking block | last 2 runs (since 2026-10-10), first 2026-09-27 |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit | cut off at token limit | last 2 runs (since 2026-10-10), first 2026-09-13 |
| mlx-community/MiniCPM-V-4.6-4bit | incomplete thinking block | back this run, first 2026-09-12 or earlier |

</details>

## Completions without detected concerns

29 completions without detected concerns; 10 more completed with prompt-compliance observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 9,984 x 6,656 pixels, 33,496,486 bytes
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: seed:* 0
- *Trust remote code:* true
- *check_models version:* 0.17.43
- *check_models revision:* 84f7be8c977eb9c1905d83ac199e3eb5e257e8df
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
