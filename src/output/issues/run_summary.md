# mlx-vlm compatibility findings across 47 cached vision-language models

**For mlx-vlm maintainers:** no crashes need action; 4 other results need
reproducing with mlx-vlm alone before anything is reported (all 4 unchanged
since the baseline).

**Completed models:** 26 with no mechanical observations, 17 with
prompt-compliance observations only (labels, form or copied hint text), and 4
with observations mlx-vlm can produce (such as repetition, missing final
answers or visible control tokens); only those 4 are candidates for native
reproduction.

## Run summary

- *Run started:* 2026-10-16 13:52:16 BST
- *Run finished:* 2026-10-16 14:03:00 BST
- *Run duration:* 10m 44s
- *Time by phase:* generation 498s, model load 83s, prompt prep 33s, cleanup
  5s, outside the model loop 26s
- *Evaluation lane:* assisted
- *Prompt hints:* the image's description and keyword hints were included in
  the prompt, so field content may be copied from them rather than seen
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Input image:* JPEG, 6,213 x 9,320 pixels (57.9 MP), 52.1 MB
- *Provenance:* run captured by check_models 0.17.55 @ c960bfb1e; report
  rendered by 0.17.56 @ 570bc55ce; original assessments retained
- *Models attempted:* 47
- *Sampling settings:* checkpoint generation_config.json values for 20 of 47
  completed models where the command line left them unset; harness defaults
  elsewhere
- *Completed:* 47
- *Crashed:* 0
- *Indeterminate:* 0
- *Crashes requiring action:* 0
- *Other results requiring review:* 4
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

- *Baseline:* c960bfb1:src/output/results.jsonl
- *Baseline run timestamp:* 2026-10-11 20:05:12 BST
- *Baseline check_models:* 0.17.46 @ 3d1b38a4f
- *Baseline mlx:* 0.32.4.dev20261010+06eb7483f @ 06eb7483f
- *Baseline mlx-vlm:* 0.7.7 @ 952d4f6bc
- *Baseline transformers:* 5.19.0
- *Baseline python:* 3.14.7
- *Baseline hardware:* Apple M5 Max, 40 GPU cores, 128.0 GB RAM
- *Earlier retained runs on this mlx version:* none recorded on
  0.32.5.dev20261016+b6642e604
- *Models compared:* 47
- *Identical generated text:* 45 of 47 completed in both
- *Generated text changed:* 2 models (listed below)
- *Text changed, by decoding:* greedy 0 of 29; sampled, same settings and seed
  2 of 18
- *Greedy text changed, by checkpoint quantization:* quantized 0 of 24; not
  quantized 0 of 5
- *Generation tok/s ratio (now/baseline):* 0.984 over 45 models; lowest 0.86
  (mlx-community/nanoLLaVA-1.5-4bit), highest 1.01
  (LiquidAI/LFM2.5-VL-450M-MLX-bf16)
- *Prefill tok/s ratio (now/baseline):* 0.995 over 47 models; lowest 0.65
  (mlx-community/FastVLM-0.5B-bf16), highest 2.00
  (nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit)
- *Throughput noise band:* fixed ±15% fallback (insufficient history)

Run environment: current run detected GPU activity outside harness inference
at the checks around 4 of 47 models (up to 14%); baseline run on battery power
for 47 of 47 models; baseline run's GPU readings were non-zero around 36 of 47
models (up to 99%), but were taken before the 0.17.47 stale-read fix, so
unreliable: such readings can count this run's own inference and are not
evidence of other GPU work.

| Model | Execution | Usability | Observation delta |
| --- | --- | --- | --- |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | completed | major concerns | -role tokens visible; -control tokens visible |

Left out of the observation delta: unsupplied place name, a check removed in
check_models 0.17.48, recorded for 3 models in only one of the two runs.

<details>
<summary>Generated text changed for 2 models</summary>

| Model | Decoding | Weights | Shared prefix (chars) | Length (chars) | Output tokens | Prompt token count |
| --- | --- | --- | --- | --- | --- | --- |
| nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | sampled | 4-bit affine | 209 | 465 → 553 | 101 → 121 | same |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | sampled | 4-bit affine | 0 | 4,677 → 66 | 1,000 → 1,000 | same |

</details>

Largest prefill changes outside 0.85-1.15x (prefill seconds are prompt tokens
divided by prefill tok/s; one timed generation per sweep, so observations, not
demonstrated speed changes):

| Model | Baseline prefill s | Now prefill s | Prefill tok/s ratio |
| --- | --- | --- | --- |
| nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | 4.13 | 2.06 | 2.00 |
| mlx-community/FastVLM-0.5B-bf16 | 0.10 | 0.16 | 0.65 |
| mlx-community/Qwen2-VL-2B-mlx | 39.75 | 31.55 | 1.26 |
| mlx-community/GLM-4.6V-nvfp4 | 10.51 | 13.05 | 0.81 |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | 37.79 | 30.68 | 1.23 |

<details>
<summary>mlx: 18 upstream commit(s) since the baseline (06eb7483f..b6642e604)</summary>

- b6642e604 docs: MetalLibrary in custom extension guide (#4701)
- 5c582abc5 Fix conv2d OOB reads for partial output-channel tiles (#4653)
- e4eaae2b2 [CUDA] Fix grouped gemm with large data (#4696)
- a82b8fb57 [CUDA] Run the vector SDPA kernels at head_dim 256 (#4698)
- 10e0f83bc ci: Run update_bypass_list on push instead of pull_request (#4671)
- 7c8a41163 docs: Publishing a New Release (#4691)
- 9cb7f7773 Patch bump to 0.32.5 (#4707)
- 36969c094 [BUG] Metal fence deadlock (#4689)
- bebd6422f Fix missing stream in gather matmul indices cast (#4687)
- 70cbe18a0 Adding minmax method to MLX to follow Numpy/Jax API (#4674)
- bb3e2d376 Add Metal extension build support (#4004)
- 588553ca9 Enable non-transposed NAX gather_qmm_rhs (#4684)
- 243bdf09c Fix missing add_temporary in pad_gpu/fill_gpu (#4660)
- 388f227f4 python: Make arctan2/radians/degrees support scalar inputs (#4679)
- 8295d5e54 Fix metal conv bounds checks and integer overflows (#4672)
- ... and 3 more (see results.jsonl)

</details>

<details>
<summary>mlx-vlm: 17 upstream commit(s) since the baseline (952d4f6bc..6d2f226dd)</summary>

- 6d2f226d Add live image resolution controls to the server (#2529)
- 63fc9795 Add Nemotron 3.5 Super VL with MTP and Bolt support (#2528)
- 4623cf9d Fix Mistral Large 3 image feature merging (#2516)
- 1ffeb07a qwen3_5: read strided K/V in ragged decode attention instead of
  copying (#2510)
- 5229dbdd Simplify MiniCPM-V 4.7 vision cache handling (#2514)
- bd6163b0 Fix MiniCPM vision dtype with quantized embeddings (#2511)
- 97902318 Allow compaction between completed tool groups (#2501)
- 757fb94b Stream tool arguments through OpenAI-compatible endpoints (#2502)
- 2b93e778 Fix compaction when max output fills the context window (#2458)
- 0602e658 Fix cancellation during multi-row prefill (#2273)
- fdb26251 Fix Muse Glimmer ignoring explicit RoPE scaling settings (#2494)
- e182243b Reformat trainer for all modalities (#1939)
- e928101a Add MiniCPM-V 4.7 1B and 35B-A3B with a custom processor (#2499)
- fb4b1e2f Stabilize tool schema ordering before prompt rendering (#2498)
- c32e07b8 Load bundled drafters directly in CLI and server (#2457)
- ... and 2 more (see results.jsonl)

</details>

<details>
<summary>mlx-vlm commits touching the affected models' architectures and their imports</summary>

- `nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit`
  (`nemotron_h_nano_omni`; imports `base.py`, `interpolate.py`, `nemotron_h`):
  63fc9795 Add Nemotron 3.5 Super VL with MTP and Bolt support (#2528);
  035d6abb Preserve NVFP4 global scales (#2437)
- `mlx-community/Muse-Glimmer-30B-OptiQ-4bit` (`muse_glimmer`; imports
  `activations.py`, `base.py`, `cache.py`, `rope_utils.py`): 0602e658 Fix
  cancellation during multi-row prefill (#2273); fdb26251 Fix Muse Glimmer
  ignoring explicit RoPE scaling settings (#2494)

Context, not attribution: shared generation, sampling and processor code can
change a model without touching its own package.

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
field, with how many appear verbatim in the prompt's keyword hints. Hint text
is the percent of the description's words lying in four-word runs copied from
the prompt's description hint (of the whole answer when no field is labelled),
rounded down; 80% or more is reported as a repeated prompt hint.

| Model | Mechanical checks | Total | Gen tok/s | Peak GB | Prompt / output tok | Keywords | Hint text | Observed | Native repro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | no concerns detected | 2.00s | 484 tok/s | 1.9 | 2,131 / 83 | 15 (9 from hints) | 19% | none | - |
| mlx-community/aya-vision-8b-4bit | no concerns detected | 4.63s | 101 tok/s | 6.5 | 2,108 / 102 | 18 (4 from hints) | 23% | none | - |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | no concerns detected | 6.30s | 69.9 tok/s | 28 | 611 / 92 | 15 (12 from hints) | 0% | none | - |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | no concerns detected | 8.80s | 121 tok/s | 19 | 1,656 / 617 | 35 (13 from hints) | 22% | none | - |
| mlx-community/gemma-3-27b-it-qat-4bit | no concerns detected | 9.07s | 31.0 tok/s | 17 | 610 / 145 | 18 (15 from hints) | 22% | none | - |
| mlx-community/gemma-4-12B-it-4bit | no concerns detected | 5.23s | 61.6 tok/s | 7.6 | 615 / 118 | 17 (14 from hints) | 57% | none | - |
| mlx-community/gemma-4-26b-a4b-it-4bit | no concerns detected | 4.84s | 122 tok/s | 16 | 615 / 105 | 16 (15 from hints) | 42% | none | - |
| mlx-community/gemma-4-e4b-it-4bit | no concerns detected | 3.88s | 123 tok/s | 6.0 | 611 / 108 | 15 (13 from hints) | 20% | none | - |
| mlx-community/GLM-4.6V-nvfp4 | no concerns detected | 25.69s | 44.2 tok/s | 78 | 6,372 / 112 | 19 (18 from hints) | 44% | none | - |
| mlx-community/InternVL3-14B-4bit | no concerns detected | 5.63s | 56.5 tok/s | 10 | 2,132 / 98 | 18 (18 from hints) | 47% | none | - |
| mlx-community/InternVL3_5-1B-4bit | no concerns detected | 2.33s | 419 tok/s | 2.1 | 2,135 / 126 | 23 (16 from hints) | 26% | none | - |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit | no concerns detected | 15.60s | 65.3 tok/s | 20 | 1,342 / 750 | 19 (19 from hints) | 75% | none | - |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | no concerns detected | 8.81s | 21.8 tok/s | 15 | 320 / 101 | 11 (9 from hints) | 11% | none | - |
| mlx-community/MiniCPM-o-4_5-4bit | no concerns detected | 3.27s | 103 tok/s | 7.1 | 410 / 104 | 15 (12 from hints) | 13% | none | - |
| mlx-community/MiniCPM-V-4.6-4bit | no concerns detected | 3.56s | 301 tok/s | 3.2 | 950 / 333 | 18 (10 from hints) | 45% | none | - |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | no concerns detected | 6.39s | 64.9 tok/s | 13 | 2,959 / 122 | 16 (11 from hints) | 35% | none | - |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | no concerns detected | 3.68s | 185 tok/s | 7.8 | 2,958 / 113 | 17 (10 from hints) | 14% | none | - |
| mlx-community/North-Micro-Vision-Instruct-4bit | no concerns detected | 5.20s | 207 tok/s | 3.9 | 4,101 / 158 | 20 (20 from hints) | 46% | none | - |
| mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit | no concerns detected | 24.93s | 71.7 tok/s | 26 | 12,809 / 120 | 18 (18 from hints) | 20% | none | - |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | no concerns detected | 36.14s | 85.7 tok/s | 23 | 16,566 / 144 | 21 (18 from hints) | 31% | none | - |
| mlx-community/Qwen3-VL-8B-Instruct-4bit | no concerns detected | 35.68s | 68.4 tok/s | 11 | 16,566 / 91 | 10 (9 from hints) | 13% | none | - |
| mlx-community/Qwen3.5-35B-A3B-4bit | no concerns detected | 35.02s | 106 tok/s | 25 | 16,579 / 130 | 18 (14 from hints) | 11% | none | - |
| mlx-community/Qwen3.8-27B-nvfp4 | no concerns detected | 54.76s | 29.4 tok/s | 21 | 16,579 / 133 | 15 (14 from hints) | 34% | none | - |
| mlx-community/X-Reasoner-7B-8bit | no concerns detected | 17.10s | 59.3 tok/s | 14 | 16,577 / 127 | 22 (17 from hints) | 57% | none | - |
| nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | no concerns detected | 6.51s | 149 tok/s | 23 | 3,642 / 121 | 17 (13 from hints) | 58% | none | - |
| TechnoBaptist/Ternary-Bonsai-2-27B-mlx-2bit | no concerns detected | 58.41s | 35.8 tok/s | 17 | 16,579 / 123 | 16 (14 from hints) | 50% | none | - |
| mlx-community/GLM-4.6V-Flash-4bit | concerns detected | 9.08s | 77.9 tok/s | 8.7 | 6,372 / 121 | 20 (20 from hints) | 100% | prompt hint repeated | - |
| mlx-community/granite-4.0-3b-vision-4bit | concerns detected | 3.19s | 178 tok/s | 4.8 | 1,410 / 105 | 10 (9 from hints) | 85% | prompt hint repeated | - |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit | concerns detected | 3.14s | 209 tok/s | 4.0 | 2,122 / 84 | 13 (12 from hints) | 100% | prompt hint repeated | - |
| mlx-community/Molmo2-8B-4bit | concerns detected | 5.08s | 72.0 tok/s | 8.2 | 1,562 / 141 | 19 (15 from hints) | 43% | duplicate keywords | - |
| mlx-community/pixtral-12b-8bit | concerns detected | 7.55s | 39.6 tok/s | 16 | 3,152 / 115 | 20 (20 from hints) | 86% | prompt hint repeated | - |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | concerns detected | 3.49s | 125 tok/s | 5.6 | 1,446 / 128 | 21 (20 from hints) | 100% | duplicate keywords; prompt hint repeated | - |
| mlx-community/Step-3.7-Flash-oQ3e | concerns detected | 38.89s | 52.2 tok/s | 92 | 3,506 / 118 | 20 (20 from hints) | 100% | prompt hint repeated | - |
| nativ-community/Mage-VL-OptiQ-4bit | concerns detected | 4.63s | 126 tok/s | 5.4 | 4,229 / 125 | 20 (20 from hints) | 96% | prompt hint repeated | - |
| nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit | concerns detected | 7.34s | 36.5 tok/s | 18 | 1,299 / 110 | 18 (16 from hints) | 84% | prompt hint repeated | - |
| sahilchachra/LensVLM-9B-MXFP4 | concerns detected | 4.24s | 102 tok/s | 7.4 | 1,843 / 93 | 10 (10 from hints) | 100% | prompt hint repeated | - |
| mlx-community/FastVLM-0.5B-bf16 | major concerns: answer format | 3.00s | 361 tok/s | 2.0 | 353 / 44 | - | 100% of answer | labelled fields not detected; prompt hint repeated | - |
| mlx-community/gemma-3n-E4B-it-4bit | major concerns: answer format | 5.82s | 94.3 tok/s | 7.2 | 609 / 261 | - | 54% of answer | labelled fields not detected | - |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns: answer format | 3.46s | 106 tok/s | 6.7 | 2,211 / 19 | - | 0% of answer | control tokens visible; labelled fields not detected | candidate |
| mlx-community/MolmoPoint-8B-4bit | major concerns: answer format | 9.85s | 31.0 tok/s | 13 | 3,164 / 144 | - | 26% of answer | labelled fields not detected | - |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | major concerns: answer format | 52.01s | 24.9 tok/s | 25 | 4,418 / 1,000 | - | - | labelled fields not detected; cut off at token limit | - |
| mlx-community/nanoLLaVA-1.5-4bit | major concerns: answer format | 2.14s | 319 tok/s | 1.8 | 349 / 131 | - | 25% | labelled fields not detected | - |
| mlx-community/SmolVLM-256M-Instruct-4bit | major concerns: answer format | 2.12s | 512 tok/s | 1.1 | 1,225 / 36 | - | 100% of answer | labelled fields not detected; prompt hint repeated | - |
| vikhyatk/moondream2 | major concerns: answer format | 3.01s | 165 tok/s | 4.8 | 1,038 / 37 | - | 100% of answer | labelled fields not detected; prompt hint repeated | - |
| mlx-community/granite-vision-3.2-2b-nvfp4 | major concerns: generation | 5.75s | 148 tok/s | 4.3 | 5,688 / 200 | 48 (14 from hints) | - | repeated text; stopped early: repeating; labelled fields not detected; duplicate keywords | candidate |
| mlx-community/Phi-3.5-vision-instruct-bf16 | major concerns: generation | 5.75s | 57.2 tok/s | 9.3 | 1,151 / 200 | 37 (16 from hints) | 50% | repeated text; stopped early: repeating; duplicate keywords | candidate |
| mlx-community/Qwen2-VL-2B-mlx | major concerns: generation | 42.25s | 124 tok/s | 9.4 | 16,577 / 1,000 | - | 9% | repeated text; control tokens visible; labelled fields not detected; cut off at token limit | candidate |

## Completed attempts requiring review

*History* dates each observation (a crash counts as one) over this model's
retained runs: 36 earlier retained runs from git
`HEAD:src/output/results.jsonl`, back to 2026-08-30; older runs were not read.
A run that did not attempt the model is skipped, and a report-only correction
of a run counts once. "Last N runs" and "N runs since" count consecutive runs
ending with this one; "first" is the earliest run read that showed it. The
table shows each model's longest-running observation; every observation's
dates are under *History by observation*.

| Model | Mechanical checks | Since baseline | History | Observed result | Evidence |
| --- | --- | --- | --- | --- | --- |
| mlx-community/granite-vision-3.2-2b-nvfp4 | major concerns: generation | unchanged | duplicate keywords: 8 runs since 2026-10-04; +3 more, newest 2 runs since 2026-10-11 | Response repeats the same text; Generation was stopped early after sustained repeated output; Required labelled fields not detected: description; Duplicate keywords: norwich, trees | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-granite-vision-32-2b-nvfp4) |
| mlx-community/Phi-3.5-vision-instruct-bf16 | major concerns: generation | unchanged | all 3: 2 runs since 2026-10-11 | Response repeats the same text; Generation was stopped early after sustained repeated output; Duplicate keywords: historic architecture, stone architecture | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-phi-35-vision-instruct-bf16) |
| mlx-community/Qwen2-VL-2B-mlx | major concerns: generation | unchanged | repeated text: 4 runs since 2026-10-10; +3 more, newest 2 runs since 2026-10-11 | Response repeats the same text; Unrecognised model control tokens remain visible; Required labelled fields not detected: keywords; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-qwen2-vl-2b-mlx) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns: answer format | unchanged | all 2: 27+ runs since 2026-09-12 or earlier | Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description, keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit) |

<details>
<summary>History by observation</summary>

| Model | Observation | Persistence |
| --- | --- | --- |
| mlx-community/granite-vision-3.2-2b-nvfp4 | duplicate keywords | last 8 runs (since 2026-10-04) |
| mlx-community/granite-vision-3.2-2b-nvfp4 | labelled fields not detected | last 2 runs (since 2026-10-11), first 2026-09-25 or earlier |
| mlx-community/granite-vision-3.2-2b-nvfp4 | repeated text | last 2 runs (since 2026-10-11) |
| mlx-community/granite-vision-3.2-2b-nvfp4 | stopped early: repeating | last 2 runs (since 2026-10-11) |
| mlx-community/Phi-3.5-vision-instruct-bf16 | duplicate keywords | last 2 runs (since 2026-10-11), first 2026-09-11 |
| mlx-community/Phi-3.5-vision-instruct-bf16 | repeated text | last 2 runs (since 2026-10-11) |
| mlx-community/Phi-3.5-vision-instruct-bf16 | stopped early: repeating | last 2 runs (since 2026-10-11) |
| mlx-community/Qwen2-VL-2B-mlx | labelled fields not detected | last 2 runs (since 2026-10-11) |
| mlx-community/Qwen2-VL-2B-mlx | repeated text | last 4 runs (since 2026-10-10) |
| mlx-community/Qwen2-VL-2B-mlx | cut off at token limit | last 2 runs (since 2026-10-11), first 2026-10-10 |
| mlx-community/Qwen2-VL-2B-mlx | control tokens visible | last 2 runs (since 2026-10-11) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | labelled fields not detected | last 27+ runs (since 2026-09-12 or earlier) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | control tokens visible | last 27+ runs (since 2026-09-12 or earlier) |

</details>

## Completions without detected concerns

26 completions without detected concerns; 17 more completed with prompt-compliance observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 6,213 x 9,320 pixels, 52,096,760 bytes
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: seed:* 0
- *Trust remote code:* true
- *check_models version:* 0.17.55
- *check_models revision:* c960bfb1e22bb7fe5ab3aa9767f0092f16ac047e
- *check_models source dirty:* false
- *mlx-vlm:* 0.7.8
- *mlx-vlm source revision:* 6d2f226dde69a32f897ce4722682fb220b261ec6
- *mlx:* 0.32.5.dev20261016+b6642e604
- *mlx source revision:* b6642e604
- *transformers:* 5.20.0
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
