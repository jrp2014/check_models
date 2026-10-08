# mlx-vlm compatibility findings across 50 cached vision-language models

**For mlx-vlm maintainers:** no crashes need action; 7 other results need
reproducing with mlx-vlm alone before anything is reported (all 7 unchanged
since the baseline).

**What this run measures.** This run records model responses to one shared
image and prompt (evaluation lane: assisted). Mechanical checks are not
factual-accuracy judgments; inspect the image, prompt and final answers before
choosing a model. Results do not establish fitness for other tasks. Every
locally cached MLX vision-language model got the same image and prompt
(reproduced below) through mlx-vlm's generation pipeline.

## Run summary

- *Run started:* 2026-10-08 22:34:43 BST
- *Run finished:* 2026-10-08 22:45:51 BST
- *Run duration:* 11m 07s
- *Time by phase:* generation 524s, model load 90s, prompt prep 37s, cleanup
  6s, outside the model loop 15s
- *Evaluation lane:* assisted
- *Prompt hints:* the image's description and keyword hints were included in
  the prompt, so field content may be copied from them rather than seen
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Input image:* JPEG, 9,641 x 6,427 pixels (62.0 MP), 58.1 MB
- *Models attempted:* 50
- *Sampling settings:* checkpoint generation_config.json values for 23 of 50
  completed models where the command line left them unset; harness defaults
  elsewhere
- *Completed:* 50
- *Crashed:* 0
- *Indeterminate:* 0
- *Crashes requiring action:* 0
- *Other results requiring review:* 7
- *Reached token limit:* 2 (2 with incomplete output)
- *Stopped early for repetition:* 3

<details>
<summary>Exact prompt sent to every model</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-10-03 18:18:19 UTC+01:00

Descriptive hints:
- Description hint: UK Border Security Command patrol vessels, including the BSC Defender and BSC Volunteer, are moored side-by-side in Ramsgate Harbour, Kent, against a dramatic sunset and the town's cliffside skyline.
- Keyword hints: Border security vessels, Buildings, Coast, Dover, Dusk, England, Fleet, Harbor, Horizon, Kent, Lifebuoy, Marina, Maritime, Mooring, Patrol boat, Patrol boats, Pier, Port, Ramsgate, Reflection

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

- *Baseline:* 68e62635:src/output/results.jsonl
- *Baseline run timestamp:* 2026-10-08 21:59:11 BST
- *Baseline check_models:* 0.17.40 @ ea5af6e8a
- *Baseline mlx:* 0.32.4.dev20261008+3c40e8f92 @ 3c40e8f92
- *Baseline mlx-vlm:* 0.7.7 @ 1cc602543
- *Baseline transformers:* 5.19.0
- *Baseline python:* 3.14.7
- *Baseline hardware:* Apple M5 Max, 40 GPU cores, 128.0 GB RAM
- *Models compared:* 50
- *Identical generated text:* 50 of 50 completed in both
- *Text changed, by decoding:* greedy 0 of 29; sampled, same settings and seed
  0 of 21
- *Greedy text changed, by checkpoint quantization:* quantized 0 of 22; not
  quantized 0 of 7
- *Generation tok/s ratio (now/baseline):* 0.997 (range 0.86-1.14, 47 models)
- *Prefill tok/s ratio (now/baseline):* 1.162 (range 0.86-8.73, 50 models)
- *Throughput noise band:* fixed ±15% fallback (insufficient history)

Run environment: current run on battery power for 50 of 50 models; baseline
run on battery power for 50 of 50 models.

No execution, usability, or observation-set changes against the baseline.

Mechanical diff only: one image, temperature as configured; single-observation
flips on one model are usually run-to-run variance, broad shifts are not.

## Model quality at a glance

Every attempted model, ordered by its mechanical checks, with counted facts.
"No concerns detected" is not an accuracy verdict: read the final answers in
the gallery. Prompt tokens include the image tokens, which drive prefill time;
output tokens are the tokens generated (limit 1,000). Keywords are counted
from the answer's Keywords field, with how many appear verbatim in the
prompt's keyword hints. Hint text is the percent of the description's words
lying in four-word runs copied from the prompt's description hint (of the
whole answer when no field is labelled), rounded down; 80% or more is reported
as a repeated prompt hint.

| Model | Mechanical checks | Total | Gen tok/s | Peak GB | Prompt / output tok | Keywords | Hint text | Observed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | no concerns detected | 2.03s | 493 tok/s | 1.9 | 2,103 / 58 | 10 (6 from hints) | 0% | none |
| mlx-community/aya-vision-8b-4bit | no concerns detected | 4.95s | 102 tok/s | 6.5 | 2,070 / 132 | 20 (20 from hints) | 56% | none |
| mlx-community/gemma-3-27b-it-qat-4bit | no concerns detected | 8.74s | 31.5 tok/s | 17 | 572 / 138 | 18 (13 from hints) | 33% | none |
| mlx-community/gemma-4-12B-it-4bit | no concerns detected | 5.08s | 61.3 tok/s | 7.6 | 577 / 108 | 18 (12 from hints) | 31% | none |
| mlx-community/gemma-4-26b-a4b-it-4bit | no concerns detected | 4.84s | 125 tok/s | 16 | 577 / 106 | 15 (11 from hints) | 68% | none |
| mlx-community/gemma-4-31b-it-4bit | no concerns detected | 8.95s | 26.3 tok/s | 20 | 577 / 107 | 17 (15 from hints) | 62% | none |
| mlx-community/gemma-4-e4b-it-4bit | no concerns detected | 3.66s | 126 tok/s | 5.9 | 573 / 79 | 15 (8 from hints) | 60% | none |
| mlx-community/GLM-4.6V-Flash-4bit | no concerns detected | 8.62s | 78.7 tok/s | 8.7 | 6,339 / 98 | 12 (7 from hints) | 66% | none |
| mlx-community/Idefics3-8B-Llama3-bf16 | no concerns detected | 8.11s | 35.4 tok/s | 18 | 2,601 / 130 | 11 (1 from hints) | 60% | none |
| mlx-community/InternVL3-14B-4bit | no concerns detected | 5.81s | 56.9 tok/s | 10 | 2,091 / 113 | 19 (19 from hints) | 56% | none |
| mlx-community/InternVL3-8B-bf16 | no concerns detected | 5.89s | 37.5 tok/s | 17 | 2,091 / 80 | 15 (15 from hints) | 38% | none |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit | no concerns detected | 19.27s | 66.1 tok/s | 20 | 1,312 / 995 | 12 (10 from hints) | 14% | none |
| mlx-community/MiniCPM-o-4_5-4bit | no concerns detected | 3.18s | 106 tok/s | 7.0 | 369 / 88 | 18 (18 from hints) | 19% | none |
| mlx-community/MiniCPM-V-4.6-4bit | no concerns detected | 4.85s | 302 tok/s | 3.2 | 910 / 662 | 18 (8 from hints) | 16% | none |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | no concerns detected | 6.89s | 66.2 tok/s | 13 | 2,905 / 140 | 16 (3 from hints) | 33% | none |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | no concerns detected | 3.83s | 188 tok/s | 7.8 | 2,904 / 132 | 14 (1 from hints) | 26% | none |
| mlx-community/Molmo2-8B-4bit | no concerns detected | 5.24s | 71.7 tok/s | 8.5 | 1,502 / 148 | 24 (20 from hints) | 61% | none |
| mlx-community/North-Micro-Vision-Instruct-4bit | no concerns detected | 5.24s | 209 tok/s | 3.9 | 4,065 / 148 | 20 (20 from hints) | 46% | none |
| mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit | no concerns detected | 5.69s | 106 tok/s | 24 | 1,267 / 129 | 18 (17 from hints) | 51% | none |
| mlx-community/Phi-3.5-vision-instruct-bf16 | no concerns detected | 4.72s | 59.1 tok/s | 9.3 | 1,115 / 144 | 19 (14 from hints) | 44% | none |
| mlx-community/pixtral-12b-8bit | no concerns detected | 7.68s | 40.0 tok/s | 16 | 3,095 / 119 | 23 (20 from hints) | 67% | none |
| mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit | no concerns detected | 27.35s | 65.4 tok/s | 26 | 12,768 / 136 | 20 (20 from hints) | 48% | none |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | no concerns detected | 40.99s | 83.0 tok/s | 23 | 16,525 / 133 | 19 (16 from hints) | 31% | none |
| mlx-community/Qwen3-VL-8B-Instruct-4bit | no concerns detected | 41.11s | 67.9 tok/s | 11 | 16,525 / 112 | 16 (13 from hints) | 51% | none |
| mlx-community/Qwen3.5-35B-A3B-4bit | no concerns detected | 38.57s | 102 tok/s | 25 | 16,541 / 152 | 18 (11 from hints) | 22% | none |
| mlx-community/Qwen3.8-27B-nvfp4 | no concerns detected | 59.97s | 27.4 tok/s | 21 | 16,541 / 121 | 16 (10 from hints) | 30% | none |
| nativ-community/Mage-VL-OptiQ-4bit | no concerns detected | 4.61s | 129 tok/s | 5.4 | 4,188 / 118 | 20 (20 from hints) | 72% | none |
| nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit | no concerns detected | 7.37s | 36.8 tok/s | 18 | 1,251 / 114 | 18 (10 from hints) | 61% | none |
| nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | no concerns detected | 8.44s | 156 tok/s | 23 | 3,606 / 132 | 17 (13 from hints) | 75% | none |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | concerns detected | 10.08s | 30.5 tok/s | 23 | 2,372 / 117 | 19 (9 from hints) | 100% | prompt hint repeated |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | concerns detected | 6.21s | 78.5 tok/s | 28 | 573 / 90 | 15 (10 from hints) | 92% | prompt hint repeated |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | concerns detected | 9.53s | 126 tok/s | 19 | 1,617 / 758 | 43 (13 from hints) | 0% | duplicate keywords |
| mlx-community/GLM-4.6V-nvfp4 | concerns detected | 21.98s | 45.3 tok/s | 78 | 6,339 / 106 | 19 (19 from hints) | 100% | prompt hint repeated |
| mlx-community/granite-4.0-3b-vision-4bit | concerns detected | 3.22s | 182 tok/s | 4.8 | 1,366 / 104 | 15 (9 from hints) | 88% | prompt hint repeated |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit | concerns detected | 3.35s | 213 tok/s | 4.0 | 2,094 / 92 | 17 (16 from hints) | 89% | prompt hint repeated |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | concerns detected | 3.51s | 118 tok/s | 5.6 | 1,407 / 114 | 20 (20 from hints) | 100% | prompt hint repeated |
| mlx-community/Step-3.7-Flash-oQ3e | concerns detected | 39.52s | 49.0 tok/s | 92 | 3,468 / 112 | 20 (20 from hints) | 100% | prompt hint repeated |
| mlx-community/FastVLM-0.5B-bf16 | major concerns | 2.85s | 369 tok/s | 2.2 | 312 / 44 | - | 29% of answer | labelled fields not detected |
| mlx-community/gemma-3n-E4B-it-4bit | major concerns | 4.42s | 98.9 tok/s | 7.2 | 571 / 132 | 20 (7 from hints) | 11% | labelled fields not detected |
| mlx-community/granite-vision-3.2-2b-nvfp4 | major concerns | 5.14s | 145 tok/s | 4.3 | 5,615 / 86 | 18 (15 from hints) | - | labelled fields not detected; duplicate keywords |
| mlx-community/InternVL3_5-1B-4bit | major concerns | 2.66s | 365 tok/s | 2.1 | 2,094 / 200 | 59 (16 from hints) | 25% | repeated text; stopped early: repeating; duplicate keywords |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | major concerns | 38.49s | 19.5 tok/s | 15 | 290 / 667 | 45 (6 from hints) | 26% | repeated text; duplicate keywords |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns | 3.46s | 108 tok/s | 6.7 | 2,174 / 18 | - | 0% of answer | control tokens visible; labelled fields not detected |
| mlx-community/MolmoPoint-8B-4bit | major concerns | 8.78s | 32.2 tok/s | 13 | 3,104 / 113 | - | 32% of answer | labelled fields not detected |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | major concerns | 55.17s | 23.4 tok/s | 25 | 4,390 / 1,000 | 2 (0 from hints) | - | control tokens visible; labelled fields not detected; cut off at token limit; role tokens visible |
| mlx-community/nanoLLaVA-1.5-4bit | major concerns | 5.02s | 311 tok/s | 1.8 | 308 / 1,000 | - | - | repeated text; labelled fields not detected; cut off at token limit |
| mlx-community/Qwen2-VL-2B-mlx | major concerns | 40.66s | 122 tok/s | 9.4 | 16,536 / 225 | 22 (6 from hints) | 5% | stopped early: repeating; duplicate keywords |
| mlx-community/SmolVLM-256M-Instruct-4bit | major concerns | 2.29s | 468 tok/s | 1.1 | 1,186 / 22 | - | 0% of answer | labelled fields not detected |
| mlx-community/X-Reasoner-7B-8bit | major concerns | 19.70s | 58.9 tok/s | 14 | 16,536 / 200 | 53 (12 from hints) | 67% | repeated text; stopped early: repeating; duplicate keywords |
| vikhyatk/moondream2 | major concerns | 4.57s | 166 tok/s | 4.8 | 1,011 / 48 | - | 91% of answer | labelled fields not detected; prompt hint repeated |

## Observation clusters

Repeated mechanical observation signatures among results requiring review.

| Observed result | Models |
| --- | --- |
| Response repeats the same text; Generation was stopped early after sustained repeated output; Repeated keyword entries | 2 |
| Response repeats the same text; Required labelled fields not detected; Response appears cut off at the token limit | 1 |
| Response repeats the same text; Repeated keyword entries | 1 |
| Generation was stopped early after sustained repeated output; Repeated keyword entries | 1 |
| Unrecognised model control tokens remain visible; Required labelled fields not detected | 1 |
| Unrecognised model control tokens remain visible; Required labelled fields not detected; Response appears cut off at the token limit; Conversation-role control tokens remain visible | 1 |

## Completed attempts requiring review

*History* dates each observation (a crash counts as one) over this model's
retained runs: 31 earlier retained runs from git
`HEAD:src/output/results.jsonl`, back to 2026-08-30; older runs were not read.
A run that did not attempt the model is skipped, and a report-only correction
of a run counts once. "Last N runs" counts consecutive runs ending with this
one; "first" is the earliest run read that showed it.

| Model | Mechanical checks | Since baseline | History | Observed result | Evidence |
| --- | --- | --- | --- | --- | --- |
| mlx-community/InternVL3_5-1B-4bit | major concerns | unchanged | duplicate keywords: last 3 runs (since 2026-10-04); repeated text: last 3 runs (since 2026-10-04); stopped early: repeating: last 3 runs (since 2026-10-04) | Response repeats the same text; Generation was stopped early after sustained repeated output; Duplicate keywords: dover | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-internvl35-1b-4bit) |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | major concerns | unchanged | duplicate keywords: last 3 runs (since 2026-10-04); first 2026-09-25; repeated text: last 3 runs (since 2026-10-04); first 2026-09-18 | Response repeats the same text; Duplicate keywords: sea | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llama-32-11b-vision-instruct-8bit) |
| mlx-community/nanoLLaVA-1.5-4bit | major concerns | unchanged | labelled fields not detected: last 26+ runs (since 2026-09-06 or earlier); repeated text: last 3 runs (since 2026-10-04); cut off at token limit: last 3 runs (since 2026-10-04) | Response repeats the same text; Required labelled fields not detected: description, keywords; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-nanollava-15-4bit) |
| mlx-community/X-Reasoner-7B-8bit | major concerns | unchanged | duplicate keywords: last 16 runs (since 2026-09-25); first 2026-09-06; repeated text: last 3 runs (since 2026-10-04); first 2026-08-30; stopped early: repeating: last 2 runs (since 2026-10-08); first 2026-09-06 | Response repeats the same text; Generation was stopped early after sustained repeated output; Duplicate keywords: horizon | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-x-reasoner-7b-8bit) |
| mlx-community/Qwen2-VL-2B-mlx | major concerns | unchanged | duplicate keywords: last 3+ runs (since 2026-10-04 or earlier); stopped early: repeating: last 3+ runs (since 2026-10-04 or earlier) | Generation was stopped early after sustained repeated output; Duplicate keywords: lifeboat station | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-qwen2-vl-2b-mlx) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns | unchanged | labelled fields not detected: last 22+ runs (since 2026-09-12 or earlier); control tokens visible: last 22+ runs (since 2026-09-12 or earlier) | Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description, keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit) |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | major concerns | unchanged | labelled fields not detected: last 16 runs (since 2026-09-25); first 2026-09-06; role tokens visible: last 3 runs (since 2026-10-04); first 2026-09-06; cut off at token limit: last 17 runs (since 2026-09-18); first 2026-09-06; control tokens visible: last 3 runs (since 2026-10-04); first 2026-09-13 | Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description; Response appears cut off at the token limit; Conversation-role control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-muse-glimmer-30b-optiq-4bit) |

## Completions without detected concerns

29 completions without detected concerns; 14 more completed with prompt-compliance observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 9,641 x 6,427 pixels, 58,125,687 bytes
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: seed:* 0
- *Trust remote code:* true
- *check_models version:* 0.17.41
- *check_models revision:* 68e62635cda8314d7d5737321676ff234920c4f3
- *check_models source dirty:* false
- *mlx-vlm:* 0.7.7
- *mlx-vlm source revision:* 1cc6025432f61b146cb72c7516209f7259fd45d0
- *mlx:* 0.32.4.dev20261008+3c40e8f92
- *mlx source revision:* 3c40e8f92
- *transformers:* 5.19.0
- *macOS Version:* 27.0.1
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
