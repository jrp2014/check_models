# mlx-vlm compatibility findings across 50 cached vision-language models

**For mlx-vlm maintainers:** no crashes need action; 8 other results need
reproducing with mlx-vlm alone before anything is reported.

**What this run measures.** This run records model responses to one shared
image and prompt (evaluation lane: assisted). Mechanical checks are not
factual-accuracy judgments; inspect the image, prompt and final answers before
choosing a model. Results do not establish fitness for other tasks. Every
locally cached MLX vision-language model got the same image and prompt
(reproduced below) through mlx-vlm's generation pipeline.

## Run summary

- *Run started:* 2026-10-04 21:44:59 BST
- *Run finished:* 2026-10-04 21:56:03 BST
- *Run duration:* 11m 03s
- *Time by phase:* generation 520s, model load 88s, prompt prep 37s, cleanup
  5s, outside the model loop 17s
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
- *Other results requiring review:* 8
- *Reached token limit:* 3 (3 with incomplete output)
- *Stopped early for repetition:* 2

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

- *Baseline:* 728b9000:src/output/results.jsonl
- *Baseline run timestamp:* 2026-10-04 01:10:55 BST
- *Baseline check_models:* 0.17.39 @ 894cc3a84
- *Baseline mlx:* 0.32.4.dev20261003+0e3ff3643 @ 0e3ff3643
- *Baseline mlx-vlm:* 0.7.4 @ 6ecadd767
- *Baseline transformers:* 5.18.0
- *Baseline python:* 3.14.7
- *Baseline hardware:* Apple M5 Max, 40 GPU cores, 128.0 GB RAM

**Not directly comparable** — output, quality and performance comparisons are
withheld because the runs differ in: prompt differs; image differs (sha256
bd1c0e60ad08… → 9f0e8d795514…). The roster, revisions and upstream changes
below are facts about the runs and are still shown.

Run environment: current run on battery power for 50 of 50 models.

- New this run (no baseline): `mlx-community/MolmoPoint-8B-4bit`,
  `mlx-community/Qwen2-VL-2B-mlx`, `vikhyatk/moondream2`
- In baseline, no longer in the cache:
  `mlx-community/Qwen2-VL-7B-Instruct-4bit`,
  `mlx-community/Qwen3-VL-2B-Thinking-bf16`,
  `mlx-community/Qwen3-VL-32B-Instruct-4bit`,
  `mlx-community/Qwen3.5-9B-MLX-4bit`,
  `nativ-community/MiMo-V2.6-Distill-Qwen-9B-MLX-4bit`

<details>
<summary>mlx: 3 upstream commit(s) since the baseline (0e3ff3643..65be04707)</summary>

- 65be04707 [CUDA] Export CommandEncoder and Device (#4619)
- d956ba2e4 Add jvp support for logcumsumexp (#4611)
- 8c56dd880 Fix CPU float16 SIMD comparison masks (#4602)

</details>

Mechanical diff only: one image, temperature as configured; single-observation
flips on one model are usually run-to-run variance, broad shifts are not.

## Model quality at a glance

Every attempted model, ordered by its mechanical checks, with counted facts.
"No concerns detected" is not an accuracy verdict: read the final answers in
the gallery. Prompt tokens include the image tokens, which drive prefill time.
Keywords are counted from the answer's Keywords field, with how many appear
verbatim in the prompt's keyword hints.

| Model | Mechanical checks | Total | Gen tok/s | Peak GB | Prompt tok | Keywords | Observed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | no concerns detected | 2.11s | 481 tok/s | 1.9 | 2,103 | 10 (6 from hints) | none |
| mlx-community/aya-vision-8b-4bit | no concerns detected | 5.00s | 102 tok/s | 6.5 | 2,070 | 20 (20 from hints) | none |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | no concerns detected | 10.27s | 30.0 tok/s | 23 | 2,372 | 19 (9 from hints) | none |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | no concerns detected | 8.50s | 125 tok/s | 19 | 1,617 | 16 (11 from hints) | none |
| mlx-community/gemma-3-27b-it-qat-4bit | no concerns detected | 8.67s | 31.2 tok/s | 17 | 572 | 19 (12 from hints) | none |
| mlx-community/gemma-4-12B-it-4bit | no concerns detected | 5.10s | 61.1 tok/s | 7.6 | 577 | 18 (12 from hints) | none |
| mlx-community/gemma-4-26b-a4b-it-4bit | no concerns detected | 5.01s | 122 tok/s | 16 | 577 | 15 (11 from hints) | none |
| mlx-community/gemma-4-31b-it-4bit | no concerns detected | 9.01s | 26.2 tok/s | 20 | 577 | 18 (16 from hints) | none |
| mlx-community/GLM-4.6V-Flash-4bit | no concerns detected | 8.58s | 78.7 tok/s | 8.7 | 6,339 | 9 (5 from hints) | none |
| mlx-community/granite-4.0-3b-vision-4bit | no concerns detected | 3.27s | 178 tok/s | 4.8 | 1,366 | 16 (7 from hints) | none |
| mlx-community/Idefics3-8B-Llama3-bf16 | no concerns detected | 8.30s | 34.3 tok/s | 18 | 2,601 | 11 (1 from hints) | none |
| mlx-community/InternVL3-14B-4bit | no concerns detected | 6.01s | 57.2 tok/s | 10 | 2,091 | 19 (19 from hints) | none |
| mlx-community/InternVL3-8B-bf16 | no concerns detected | 5.95s | 37.3 tok/s | 17 | 2,091 | 15 (15 from hints) | none |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit | no concerns detected | 19.45s | 65.0 tok/s | 20 | 1,312 | 12 (10 from hints) | none |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit | no concerns detected | 3.34s | 210 tok/s | 4.0 | 2,094 | 18 (11 from hints) | none |
| mlx-community/MiniCPM-o-4_5-4bit | no concerns detected | 3.17s | 106 tok/s | 7.0 | 369 | 18 (18 from hints) | none |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | no concerns detected | 6.97s | 66.0 tok/s | 13 | 2,905 | 16 (3 from hints) | none |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | no concerns detected | 3.87s | 186 tok/s | 7.8 | 2,904 | 14 (0 from hints) | none |
| mlx-community/Molmo2-8B-4bit | no concerns detected | 5.41s | 71.4 tok/s | 8.1 | 1,502 | 24 (20 from hints) | none |
| mlx-community/North-Micro-Vision-Instruct-4bit | no concerns detected | 5.34s | 208 tok/s | 3.9 | 4,065 | 20 (20 from hints) | none |
| mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit | no concerns detected | 5.79s | 103 tok/s | 24 | 1,267 | 18 (17 from hints) | none |
| mlx-community/Phi-3.5-vision-instruct-bf16 | no concerns detected | 4.76s | 58.4 tok/s | 9.3 | 1,115 | 19 (14 from hints) | none |
| mlx-community/pixtral-12b-8bit | no concerns detected | 7.74s | 39.8 tok/s | 16 | 3,095 | 23 (20 from hints) | none |
| mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit | no concerns detected | 26.29s | 72.8 tok/s | 26 | 12,768 | 18 (17 from hints) | none |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | no concerns detected | 36.09s | 85.3 tok/s | 23 | 16,525 | 21 (16 from hints) | none |
| mlx-community/Qwen3-VL-8B-Instruct-4bit | no concerns detected | 37.84s | 69.6 tok/s | 11 | 16,525 | 16 (13 from hints) | none |
| mlx-community/Qwen3.5-35B-A3B-4bit | no concerns detected | 35.62s | 106 tok/s | 25 | 16,541 | 16 (8 from hints) | none |
| mlx-community/Qwen3.8-27B-nvfp4 | no concerns detected | 61.13s | 29.3 tok/s | 21 | 16,541 | 16 (10 from hints) | none |
| nativ-community/Mage-VL-OptiQ-4bit | no concerns detected | 4.62s | 127 tok/s | 5.4 | 4,188 | 20 (20 from hints) | none |
| nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit | no concerns detected | 7.59s | 36.6 tok/s | 18 | 1,251 | 18 (10 from hints) | none |
| nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | no concerns detected | 8.62s | 156 tok/s | 23 | 3,606 | 17 (13 from hints) | none |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | concerns detected | 6.12s | 80.2 tok/s | 28 | 573 | 16 (11 from hints) | prompt hint repeated |
| mlx-community/gemma-4-e4b-it-4bit | concerns detected | 3.70s | 124 tok/s | 6.0 | 573 | 15 (10 from hints) | prompt hint repeated |
| mlx-community/GLM-4.6V-nvfp4 | concerns detected | 21.05s | 44.7 tok/s | 78 | 6,339 | 19 (19 from hints) | prompt hint repeated |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | concerns detected | 3.44s | 127 tok/s | 5.6 | 1,407 | 20 (20 from hints) | prompt hint repeated |
| mlx-community/Step-3.7-Flash-oQ3e | concerns detected | 36.69s | 53.1 tok/s | 92 | 3,468 | 20 (20 from hints) | prompt hint repeated |
| mlx-community/FastVLM-0.5B-bf16 | major concerns | 2.94s | 371 tok/s | 2.2 | 312 | - | labelled fields not detected |
| mlx-community/gemma-3n-E4B-it-4bit | major concerns | 4.45s | 92.6 tok/s | 7.2 | 571 | 19 (7 from hints) | labelled fields not detected |
| mlx-community/granite-vision-3.2-2b-nvfp4 | major concerns | 5.21s | 144 tok/s | 4.3 | 5,615 | 18 (15 from hints) | labelled fields not detected; duplicate keywords |
| mlx-community/InternVL3_5-1B-4bit | major concerns | 2.57s | 422 tok/s | 2.1 | 2,094 | 55 (16 from hints) | repeated text; stopped early: repeating; duplicate keywords |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | major concerns | 38.95s | 19.3 tok/s | 15 | 290 | 45 (6 from hints) | repeated text; duplicate keywords |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns | 4.13s | 107 tok/s | 6.7 | 2,174 | - | control tokens visible; labelled fields not detected |
| mlx-community/MiniCPM-V-4.6-4bit | major concerns | 2.96s | 304 tok/s | 3.2 | 910 | 15 (8 from hints) | incomplete thinking block |
| mlx-community/MolmoPoint-8B-4bit | major concerns | 10.01s | 31.4 tok/s | 13 | 3,104 | - | labelled fields not detected |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | major concerns | 54.80s | 23.6 tok/s | 25 | 4,390 | 2 (0 from hints) | control tokens visible; labelled fields not detected; cut off at token limit; role tokens visible |
| mlx-community/nanoLLaVA-1.5-4bit | major concerns | 5.04s | 341 tok/s | 1.8 | 308 | - | repeated text; labelled fields not detected; cut off at token limit |
| mlx-community/Qwen2-VL-2B-mlx | major concerns | 37.17s | 125 tok/s | 9.4 | 16,536 | 22 (6 from hints) | stopped early: repeating; duplicate keywords |
| mlx-community/SmolVLM-256M-Instruct-4bit | major concerns | 2.23s | 553 tok/s | 1.1 | 1,186 | - | labelled fields not detected |
| mlx-community/X-Reasoner-7B-8bit | major concerns | 32.49s | 57.8 tok/s | 14 | 16,536 | 390 (19 from hints) | repeated text; cut off at token limit; duplicate keywords |
| vikhyatk/moondream2 | major concerns | 2.99s | 162 tok/s | 4.8 | 1,011 | - | labelled fields not detected |

## Completed attempts requiring review

| Model | Mechanical checks | Observed result | Evidence |
| --- | --- | --- | --- |
| mlx-community/InternVL3_5-1B-4bit | major concerns | Response repeats the same text; Generation was stopped early after sustained repeated output; Duplicate keywords: dover, horizon, time, date, timezone | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-internvl35-1b-4bit) |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | major concerns | Response repeats the same text; Duplicate keywords: sea | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llama-32-11b-vision-instruct-8bit) |
| mlx-community/nanoLLaVA-1.5-4bit | major concerns | Response repeats the same text; Required labelled fields not detected: description, keywords; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-nanollava-15-4bit) |
| mlx-community/X-Reasoner-7B-8bit | major concerns | Response repeats the same text; Response appears cut off at the token limit; Duplicate keywords: sunset, buildings, marina, mooring, patrol boats, water, horizon, england, maritime, port, pier, coast, dover, fleet, harbor, lifebuoy, patrol boat, ramsgate, reflection | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-x-reasoner-7b-8bit) |
| mlx-community/Qwen2-VL-2B-mlx | major concerns | Generation was stopped early after sustained repeated output; Duplicate keywords: lifeboat station | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-qwen2-vl-2b-mlx) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns | Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description, keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit) |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | major concerns | Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description; Response appears cut off at the token limit; Conversation-role control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-muse-glimmer-30b-optiq-4bit) |
| mlx-community/MiniCPM-V-4.6-4bit | major concerns | Internal reasoning block appears incomplete | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-minicpm-v-46-4bit) |

## Completions without detected concerns

31 completions without detected concerns; 11 more completed with prompt-compliance observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 9,641 x 6,427 pixels, 58,125,687 bytes
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: seed:* 0
- *Trust remote code:* true
- *check_models version:* 0.17.39
- *check_models revision:* 728b90004483ca88aab05cbc80b600ccddb76acc
- *check_models source dirty:* false
- *mlx-vlm:* 0.7.4
- *mlx-vlm source revision:* 6ecadd767ca1c7c54763289c750dd180d7945037
- *mlx:* 0.32.4.dev20261004+65be04707
- *mlx source revision:* 65be04707
- *transformers:* 5.18.0
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
