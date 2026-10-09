# mlx-vlm compatibility findings across 50 cached vision-language models

**For mlx-vlm maintainers:** no crashes need action; 7 other results need
reproducing with mlx-vlm alone before anything is reported (all 7 unchanged
since the baseline).

**Completed models:** 28 with no mechanical observations, 15 with
prompt-compliance observations only (labels, form or copied hint text), and 7
with observations mlx-vlm can produce (such as repetition, missing final
answers or visible control tokens); only those 7 are candidates for native
reproduction.

**Evidence links** target the repository's mutable main branch: they show this
run only once these artifacts are committed, and a later run's commit replaces
them. Before sharing upstream, pin them to the commit that published these
artifacts.

**What this run measures.** This run records model responses to one shared
image and prompt (evaluation lane: assisted). Mechanical checks are not
factual-accuracy judgments; inspect the image, prompt and final answers before
choosing a model. Results do not establish fitness for other tasks. Every
locally cached MLX vision-language model got the same image and prompt
(reproduced below) through mlx-vlm's generation pipeline.

## Run summary

- *Run started:* 2026-10-09 23:21:34 BST
- *Run finished:* 2026-10-09 23:32:21 BST
- *Run duration:* 10m 46s
- *Time by phase:* generation 496s, model load 96s, prompt prep 38s, cleanup
  6s, outside the model loop 16s
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

- *Baseline:* d0416824:src/output/results.jsonl
- *Baseline run timestamp:* 2026-10-08 22:45:53 BST
- *Baseline check_models:* 0.17.41 @ 68e62635c
- *Baseline mlx:* 0.32.4.dev20261008+3c40e8f92 @ 3c40e8f92
- *Baseline mlx-vlm:* 0.7.7 @ 1cc602543
- *Baseline transformers:* 5.19.0
- *Baseline python:* 3.14.7
- *Baseline hardware:* Apple M5 Max, 40 GPU cores, 128.0 GB RAM
- *Models compared:* 50
- *Identical generated text:* 47 of 50 completed in both
- *Generated text changed:* 3 models (listed below)
- *Text changed, by decoding:* greedy 1 of 29; sampled, same settings and seed
  2 of 21
- *Greedy text changed, by checkpoint quantization:* quantized 0 of 22; not
  quantized 1 of 7
- *Generation tok/s ratio (now/baseline):* 0.987 over 47 models; lowest 0.77
  (mlx-community/diffusiongemma-26B-A4B-it-mxfp8), highest 1.13
  (mlx-community/nanoLLaVA-1.5-4bit)
- *Prefill tok/s ratio (now/baseline):* 0.849 over 50 models; lowest 0.10
  (mlx-community/FastVLM-0.5B-bf16), highest 1.46
  (mlx-community/Qwen2-VL-2B-mlx)
- *Throughput noise band:* fixed ±15% fallback (insufficient history)

Run environment: current run on battery power for 50 of 50 models; baseline
run on battery power for 50 of 50 models.

| Model | Execution | Usability | Observation delta |
| --- | --- | --- | --- |
| mlx-community/MiniCPM-V-4.6-4bit | completed | no concerns detected → concerns detected | +duplicate keywords |

<details>
<summary>Generated text changed for 3 models</summary>

| Model | Decoding | Weights | Shared prefix (chars) | Length (chars) | Prompt token count |
| --- | --- | --- | --- | --- | --- |
| mlx-community/InternVL3-8B-bf16 | greedy | not quantized | 334 | 350 → 349 | same |
| mlx-community/MiniCPM-V-4.6-4bit | sampled | 4-bit affine | 499 | 2,535 → 2,193 | same |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | sampled | 4-bit affine | 1,404 | 4,644 → 4,523 | same |

</details>

Generation tok/s outside the expected band for 1 model (mlx changed since the
baseline: 3c40e8f92..99f109b56):

| Model | Baseline tok/s | Now tok/s | Ratio | Expected band |
| --- | --- | --- | --- | --- |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | 78.5 | 60.7 | 0.77 | 66.7-90.2 (fallback) |

Largest prefill changes outside 0.85-1.15x (prefill seconds are prompt tokens
divided by prefill tok/s; one timed generation per sweep, so observations, not
demonstrated speedups):

| Model | Baseline prefill s | Now prefill s | Prefill tok/s ratio |
| --- | --- | --- | --- |
| mlx-community/FastVLM-0.5B-bf16 | 0.10 | 1.00 | 0.10 |
| mlx-community/MiniCPM-V-4.6-4bit | 0.24 | 1.90 | 0.13 |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit | 0.66 | 2.97 | 0.22 |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | 0.31 | 1.25 | 0.25 |
| mlx-community/Molmo2-8B-4bit | 0.74 | 2.89 | 0.26 |

<details>
<summary>mlx: 13 upstream commit(s) since the baseline (3c40e8f92..99f109b56)</summary>

- 99f109b56 Validate .npy shape in mlx/io/load.cpp (#4657)
- 84a72e34e Fix input/output scalar merging in compile (#4658)
- 43f5eaae9 Fix Python dtpye -&gt; mlx conversion narrowing (#4656)
- a87691d2d Add thin M GEMM dispatch (#4654)
- 77bf1fa01 Fix incorrect output mask computation in `mx.block_masked_mm`
  (#4652)
- 789be76e2 [Metal] Fix SDPA max threadgroup for M1/M2 generation (#4643)
- 2654664a3 Add FloorDivide primitive for integer floor division (#4642)
- 8d5bdbc91 Fix cpu binary_op_dispatch_dims on large data (#4648)
- 0d8236fa8 Fix missing break binary bool (#4651)
- 95f8533a7 Fix metal negative stride index in masked_scatter/scatter/SDPA
  mask/segmented_gemm (#4650)
- cd30780dd Fix pickle bfloat16 strides (#4649)
- 4089fd56f Update Metal Complex division in numpy style to avoid overflow
  condition (#4644)
- e0408d473 Add 1-bit affine quantization support (Metal) (#3161)

</details>

<details>
<summary>mlx-vlm: 12 upstream commit(s) since the baseline (1cc602543..952d4f6bc)</summary>

- 952d4f6b Merge pull request #2487 from lucasnewman/migrate-to-httpx2
- 886a6fe2 Merge pull request #2449 from
  Lazarus-931/fix/2426-expert-activation
- 448b3f9f Migrate to httpx2, fix CI.
- c2445476 Merge pull request #2468 from ishaanzee/rfdetr-seg-fast-upsample
- 67b71093 Merge pull request #2462 from joshuaswarren/inkling-metal-gate
- a72efccd inkling: use the Metal kernels only when Metal is available
- 370f4ccf Merge pull request #2467 from
  ykhrustalev/ykhrustalev/liquidai-d1-models-port
- be0dbae9 fix: let decision models declare the media they read
- 005bcd6d feat: add LiquidAI d1-3B and d1-omni-600M decision models
- 933558a6 Use the separable interpolation kernel for the RF-DETR segmentation
  upsample
- 6e6f363d Require a switch layer to declare its activation for expert offload
- 207587b0 Write a resident weight index when repacking for expert offload

</details>

<details>
<summary>mlx-vlm commits touching the affected models' architectures and their imports</summary>

- `mlx-community/InternVL3-8B-bf16` (`internvl_chat`; imports `base.py`,
  `cache.py`, `mlp.py`): no commits touched `mlx_vlm/models/internvl_chat/` or
  the entries it imports
- `mlx-community/MiniCPM-V-4.6-4bit` (`minicpmv4_6`; imports `base.py`,
  `qwen3_5`): no commits touched `mlx_vlm/models/minicpmv4_6/` or the entries
  it imports
- `mlx-community/Muse-Glimmer-30B-OptiQ-4bit` (`muse_glimmer`; imports
  `activations.py`, `base.py`, `cache.py`, `rope_utils.py`): no commits
  touched `mlx_vlm/models/muse_glimmer/` or the entries it imports

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
means an answer was generated but missed the requested form. Prompt tokens
include the image tokens, which drive prefill time; output tokens are the
tokens generated (limit 1,000). Keywords are counted from the answer's
Keywords field, with how many appear verbatim in the prompt's keyword hints.
Hint text is the percent of the description's words lying in four-word runs
copied from the prompt's description hint (of the whole answer when no field
is labelled), rounded down; 80% or more is reported as a repeated prompt hint.

| Model | Mechanical checks | Total | Gen tok/s | Peak GB | Prompt / output tok | Keywords | Hint text | Observed | Native repro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | no concerns detected | 2.36s | 482 tok/s | 1.9 | 2,103 / 58 | 10 (6 from hints) | 0% | none | - |
| mlx-community/aya-vision-8b-4bit | no concerns detected | 5.15s | 97.9 tok/s | 6.5 | 2,070 / 132 | 20 (20 from hints) | 56% | none | - |
| mlx-community/gemma-3-27b-it-qat-4bit | no concerns detected | 9.16s | 30.7 tok/s | 17 | 572 / 138 | 18 (13 from hints) | 33% | none | - |
| mlx-community/gemma-4-12B-it-4bit | no concerns detected | 5.39s | 61.2 tok/s | 7.6 | 577 / 108 | 18 (12 from hints) | 31% | none | - |
| mlx-community/gemma-4-26b-a4b-it-4bit | no concerns detected | 5.03s | 120 tok/s | 16 | 577 / 106 | 15 (11 from hints) | 68% | none | - |
| mlx-community/gemma-4-31b-it-4bit | no concerns detected | 9.04s | 26.3 tok/s | 20 | 577 / 107 | 17 (15 from hints) | 62% | none | - |
| mlx-community/gemma-4-e4b-it-4bit | no concerns detected | 4.02s | 122 tok/s | 6.0 | 573 / 79 | 15 (8 from hints) | 60% | none | - |
| mlx-community/GLM-4.6V-Flash-4bit | no concerns detected | 9.31s | 79.5 tok/s | 8.7 | 6,339 / 98 | 12 (7 from hints) | 66% | none | - |
| mlx-community/Idefics3-8B-Llama3-bf16 | no concerns detected | 8.94s | 35.3 tok/s | 18 | 2,601 / 130 | 11 (1 from hints) | 60% | none | - |
| mlx-community/InternVL3-14B-4bit | no concerns detected | 7.07s | 57.4 tok/s | 10 | 2,091 / 113 | 19 (19 from hints) | 56% | none | - |
| mlx-community/InternVL3-8B-bf16 | no concerns detected | 6.26s | 36.4 tok/s | 17 | 2,091 / 80 | 15 (15 from hints) | 38% | none | - |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit | no concerns detected | 23.02s | 60.2 tok/s | 20 | 1,312 / 995 | 12 (10 from hints) | 14% | none | - |
| mlx-community/MiniCPM-o-4_5-4bit | no concerns detected | 3.37s | 102 tok/s | 7.0 | 369 / 88 | 18 (18 from hints) | 19% | none | - |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | no concerns detected | 6.91s | 64.5 tok/s | 13 | 2,905 / 140 | 16 (3 from hints) | 33% | none | - |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | no concerns detected | 5.20s | 181 tok/s | 7.8 | 2,904 / 132 | 14 (1 from hints) | 26% | none | - |
| mlx-community/Molmo2-8B-4bit | no concerns detected | 7.57s | 68.9 tok/s | 8.1 | 1,502 / 148 | 24 (20 from hints) | 61% | none | - |
| mlx-community/North-Micro-Vision-Instruct-4bit | no concerns detected | 5.27s | 207 tok/s | 3.9 | 4,065 / 148 | 20 (20 from hints) | 46% | none | - |
| mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit | no concerns detected | 6.08s | 104 tok/s | 24 | 1,267 / 129 | 18 (17 from hints) | 51% | none | - |
| mlx-community/Phi-3.5-vision-instruct-bf16 | no concerns detected | 4.94s | 58.6 tok/s | 9.3 | 1,115 / 144 | 19 (14 from hints) | 44% | none | - |
| mlx-community/pixtral-12b-8bit | no concerns detected | 7.44s | 39.8 tok/s | 16 | 3,095 / 119 | 23 (20 from hints) | 67% | none | - |
| mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit | no concerns detected | 22.56s | 72.1 tok/s | 26 | 12,768 / 136 | 20 (20 from hints) | 48% | none | - |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | no concerns detected | 31.53s | 87.0 tok/s | 23 | 16,525 / 133 | 19 (16 from hints) | 31% | none | - |
| mlx-community/Qwen3-VL-8B-Instruct-4bit | no concerns detected | 32.36s | 70.1 tok/s | 11 | 16,525 / 112 | 16 (13 from hints) | 51% | none | - |
| mlx-community/Qwen3.5-35B-A3B-4bit | no concerns detected | 32.01s | 104 tok/s | 25 | 16,541 / 152 | 18 (11 from hints) | 22% | none | - |
| mlx-community/Qwen3.8-27B-nvfp4 | no concerns detected | 51.54s | 29.4 tok/s | 21 | 16,541 / 121 | 16 (10 from hints) | 30% | none | - |
| nativ-community/Mage-VL-OptiQ-4bit | no concerns detected | 4.66s | 127 tok/s | 5.4 | 4,188 / 118 | 20 (20 from hints) | 72% | none | - |
| nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit | no concerns detected | 8.21s | 36.2 tok/s | 18 | 1,251 / 114 | 18 (10 from hints) | 61% | none | - |
| nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | no concerns detected | 9.55s | 155 tok/s | 23 | 3,606 / 132 | 17 (13 from hints) | 75% | none | - |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | concerns detected | 10.47s | 30.5 tok/s | 23 | 2,372 / 117 | 19 (9 from hints) | 100% | prompt hint repeated | - |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | concerns detected | 8.15s | 60.7 tok/s | 28 | 573 / 90 | 15 (10 from hints) | 92% | prompt hint repeated | - |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | concerns detected | 10.20s | 125 tok/s | 19 | 1,617 / 758 | 43 (13 from hints) | 0% | duplicate keywords | - |
| mlx-community/GLM-4.6V-nvfp4 | concerns detected | 24.15s | 45.3 tok/s | 78 | 6,339 / 106 | 19 (19 from hints) | 100% | prompt hint repeated | - |
| mlx-community/granite-4.0-3b-vision-4bit | concerns detected | 3.72s | 177 tok/s | 4.8 | 1,366 / 104 | 15 (9 from hints) | 88% | prompt hint repeated | - |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit | concerns detected | 3.81s | 188 tok/s | 4.0 | 2,094 / 92 | 17 (16 from hints) | 89% | prompt hint repeated | - |
| mlx-community/MiniCPM-V-4.6-4bit | concerns detected | 6.19s | 288 tok/s | 3.2 | 910 / 527 | 36 (9 from hints) | 26% | duplicate keywords | - |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | concerns detected | 3.85s | 122 tok/s | 5.6 | 1,407 / 114 | 20 (20 from hints) | 100% | prompt hint repeated | - |
| mlx-community/Step-3.7-Flash-oQ3e | concerns detected | 39.75s | 51.1 tok/s | 92 | 3,468 / 112 | 20 (20 from hints) | 100% | prompt hint repeated | - |
| mlx-community/FastVLM-0.5B-bf16 | major concerns: answer format | 3.82s | 367 tok/s | 1.8 | 312 / 44 | - | 29% of answer | labelled fields not detected | - |
| mlx-community/gemma-3n-E4B-it-4bit | major concerns: answer format | 5.42s | 94.6 tok/s | 7.1 | 571 / 132 | 20 (7 from hints) | 11% | labelled fields not detected | - |
| mlx-community/granite-vision-3.2-2b-nvfp4 | major concerns: answer format | 4.99s | 143 tok/s | 4.3 | 5,615 / 86 | 18 (15 from hints) | - | labelled fields not detected; duplicate keywords | - |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns: answer format | 3.55s | 103 tok/s | 6.7 | 2,174 / 18 | - | 0% of answer | control tokens visible; labelled fields not detected | candidate |
| mlx-community/MolmoPoint-8B-4bit | major concerns: answer format | 10.12s | 30.4 tok/s | 12 | 3,104 / 113 | - | 32% of answer | labelled fields not detected | - |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | major concerns: answer format | 54.07s | 24.3 tok/s | 25 | 4,390 / 1,000 | 2 (0 from hints) | - | control tokens visible; labelled fields not detected; cut off at token limit; role tokens visible | candidate |
| mlx-community/SmolVLM-256M-Instruct-4bit | major concerns: answer format | 2.58s | 527 tok/s | 1.1 | 1,186 / 22 | - | 0% of answer | labelled fields not detected | - |
| vikhyatk/moondream2 | major concerns: answer format | 4.24s | 162 tok/s | 4.8 | 1,011 / 48 | - | 91% of answer | labelled fields not detected; prompt hint repeated | - |
| mlx-community/InternVL3_5-1B-4bit | major concerns: generation | 2.78s | 388 tok/s | 2.1 | 2,094 / 200 | 59 (16 from hints) | 25% | repeated text; stopped early: repeating; duplicate keywords | candidate |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | major concerns: generation | 41.12s | 18.4 tok/s | 15 | 290 / 667 | 45 (6 from hints) | 26% | repeated text; duplicate keywords | candidate |
| mlx-community/nanoLLaVA-1.5-4bit | major concerns: generation | 4.76s | 352 tok/s | 1.8 | 308 / 1,000 | - | - | repeated text; labelled fields not detected; cut off at token limit | candidate |
| mlx-community/Qwen2-VL-2B-mlx | major concerns: generation | 29.21s | 125 tok/s | 9.4 | 16,536 / 225 | 22 (6 from hints) | 5% | stopped early: repeating; duplicate keywords | candidate |
| mlx-community/X-Reasoner-7B-8bit | major concerns: generation | 20.06s | 57.3 tok/s | 14 | 16,536 / 200 | 53 (12 from hints) | 67% | repeated text; stopped early: repeating; duplicate keywords | candidate |

## Observation clusters

Observation signatures shared by two or more results requiring review.

| Observed result | Models |
| --- | --- |
| Response repeats the same text; Generation was stopped early after sustained repeated output; Repeated keyword entries | [mlx-community/InternVL3_5-1B-4bit](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-internvl35-1b-4bit), [mlx-community/X-Reasoner-7B-8bit](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-x-reasoner-7b-8bit) |

## Completed attempts requiring review

*History* dates each observation (a crash counts as one) over this model's
retained runs: 32 earlier retained runs from git
`HEAD:src/output/results.jsonl`, back to 2026-08-30; older runs were not read.
A run that did not attempt the model is skipped, and a report-only correction
of a run counts once. "Last N runs" and "N runs since" count consecutive runs
ending with this one; "first" is the earliest run read that showed it. The
table shows each model's longest-running observation; every observation's
dates are under *History by observation*.

| Model | Mechanical checks | Since baseline | History | Observed result | Evidence |
| --- | --- | --- | --- | --- | --- |
| mlx-community/InternVL3_5-1B-4bit | major concerns: generation | unchanged | all 3: 4 runs since 2026-10-04 | Response repeats the same text; Generation was stopped early after sustained repeated output; Duplicate keywords: dover | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-internvl35-1b-4bit) |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | major concerns: generation | unchanged | all 2: 4 runs since 2026-10-04 | Response repeats the same text; Duplicate keywords: sea | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llama-32-11b-vision-instruct-8bit) |
| mlx-community/nanoLLaVA-1.5-4bit | major concerns: generation | unchanged | labelled fields not detected: 27+ runs since 2026-09-06 or earlier; +2 more, newest 4 runs since 2026-10-04 | Response repeats the same text; Required labelled fields not detected: description, keywords; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-nanollava-15-4bit) |
| mlx-community/X-Reasoner-7B-8bit | major concerns: generation | unchanged | duplicate keywords: 17 runs since 2026-09-25; +2 more, newest 3 runs since 2026-10-08 | Response repeats the same text; Generation was stopped early after sustained repeated output; Duplicate keywords: horizon | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-x-reasoner-7b-8bit) |
| mlx-community/Qwen2-VL-2B-mlx | major concerns: generation | unchanged | all 2: 4+ runs since 2026-10-04 or earlier | Generation was stopped early after sustained repeated output; Duplicate keywords: lifeboat station | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-qwen2-vl-2b-mlx) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns: answer format | unchanged | all 2: 23+ runs since 2026-09-12 or earlier | Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description, keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit) |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | major concerns: answer format | unchanged | cut off at token limit: 18 runs since 2026-09-18; +3 more, newest 4 runs since 2026-10-04 | Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description; Response appears cut off at the token limit; Conversation-role control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-muse-glimmer-30b-optiq-4bit) |

<details>
<summary>History by observation</summary>

| Model | Observation | Persistence |
| --- | --- | --- |
| mlx-community/InternVL3_5-1B-4bit | duplicate keywords | last 4 runs (since 2026-10-04) |
| mlx-community/InternVL3_5-1B-4bit | repeated text | last 4 runs (since 2026-10-04) |
| mlx-community/InternVL3_5-1B-4bit | stopped early: repeating | last 4 runs (since 2026-10-04) |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | duplicate keywords | last 4 runs (since 2026-10-04), first 2026-09-25 |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | repeated text | last 4 runs (since 2026-10-04), first 2026-09-18 |
| mlx-community/nanoLLaVA-1.5-4bit | labelled fields not detected | last 27+ runs (since 2026-09-06 or earlier) |
| mlx-community/nanoLLaVA-1.5-4bit | repeated text | last 4 runs (since 2026-10-04) |
| mlx-community/nanoLLaVA-1.5-4bit | cut off at token limit | last 4 runs (since 2026-10-04) |
| mlx-community/X-Reasoner-7B-8bit | duplicate keywords | last 17 runs (since 2026-09-25), first 2026-09-06 |
| mlx-community/X-Reasoner-7B-8bit | repeated text | last 4 runs (since 2026-10-04), first 2026-08-30 |
| mlx-community/X-Reasoner-7B-8bit | stopped early: repeating | last 3 runs (since 2026-10-08), first 2026-09-06 |
| mlx-community/Qwen2-VL-2B-mlx | duplicate keywords | last 4+ runs (since 2026-10-04 or earlier) |
| mlx-community/Qwen2-VL-2B-mlx | stopped early: repeating | last 4+ runs (since 2026-10-04 or earlier) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | labelled fields not detected | last 23+ runs (since 2026-09-12 or earlier) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | control tokens visible | last 23+ runs (since 2026-09-12 or earlier) |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | labelled fields not detected | last 17 runs (since 2026-09-25), first 2026-09-06 |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | role tokens visible | last 4 runs (since 2026-10-04), first 2026-09-06 |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | cut off at token limit | last 18 runs (since 2026-09-18), first 2026-09-06 |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | control tokens visible | last 4 runs (since 2026-10-04), first 2026-09-13 |

</details>

## Completions without detected concerns

28 completions without detected concerns; 15 more completed with prompt-compliance observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 9,641 x 6,427 pixels, 58,125,687 bytes
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: seed:* 0
- *Trust remote code:* true
- *check_models version:* 0.17.41
- *check_models revision:* d041682481c0aba8ef8984eb6d2b1d485321f4be
- *check_models source dirty:* false
- *mlx-vlm:* 0.7.7
- *mlx-vlm source revision:* 952d4f6bc65bd5095e74abe71c78038764ee08aa
- *mlx:* 0.32.4.dev20261009+99f109b56
- *mlx source revision:* 99f109b56
- *transformers:* 5.19.0
- *macOS Version:* 27.0.1
- *GPU/Chip:* Apple M5 Max
- *Python Version:* 3.14.7

## Full artifacts

| Artifact | Link |
| --- | --- |
| Diagnostics | [diagnostics.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md) |
| Model gallery | [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md) |
| Results JSONL | [results.jsonl](https://github.com/jrp2014/check_models/blob/main/src/output/results.jsonl) |
| Environment | [environment.log](https://github.com/jrp2014/check_models/blob/main/src/output/environment.log) |
| Log | [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log) |
