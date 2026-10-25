# mlx-vlm compatibility findings across 43 cached vision-language models

**For mlx-vlm maintainers:** no crashes need action; 3 other results need
reproducing with mlx-vlm alone before anything is reported (all 3 unchanged
since the baseline).

**Completed models:** 37 with no mechanical observations, 3 with
prompt-compliance observations only (labels, form or copied hint text), and 3
with observations mlx-vlm can produce (such as repetition, missing final
answers or visible control tokens); only those 3 are candidates for native
reproduction.

## Run summary

- *Run started:* 2026-10-25 01:53:46 BST
- *Run finished:* 2026-10-25 01:57:57 BST
- *Run duration:* 4m 10s
- *Time by phase:* generation 142s, model load 84s, prompt prep 1s, cleanup
  5s, outside the model loop 20s
- *Evaluation lane:* blind
- *Prompt hints:* no description or keyword hints
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Input image:* JPEG, 640 x 480 pixels (0.3 MP), 0.2 MB
- *Models attempted:* 43
- *Sampling settings:* checkpoint generation_config.json values for 18 of 43
  completed models where the command line left them unset; harness defaults
  elsewhere
- *Completed:* 43
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

- *Baseline:* baa72d7e:src/output/results.jsonl
- *Baseline run timestamp:* 2026-10-23 15:53:06 BST
- *Baseline check_models:* 0.17.89 @ 361071fde
- *Baseline mlx:* 0.32.5.dev20261023+afc03c511 @ afc03c511
- *Baseline mlx-vlm:* 0.7.10 @ 4319ef2c8
- *Baseline transformers:* 5.20.2
- *Baseline python:* 3.14.7
- *Baseline hardware:* Apple M5 Max, 40 GPU cores, 128.0 GB RAM
- *Earlier retained runs on this mlx version:* none recorded on
  0.32.5.dev20261024+56c2bf923
- *Models compared:* 40
- *Identical generated text:* 39 of 40 completed in both
- *Generated text changed:* 1 models (listed below)
- *Text changed, by decoding:* greedy 0 of 23; sampled, same settings and seed
  1 of 17
- *Greedy text changed, by checkpoint quantization:* quantized 0 of 19; not
  quantized 0 of 4
- *Generation tok/s ratio (now/baseline):* 1.004 over 40 models; lowest 0.63
  (mlx-community/Qwen3.5-35B-A3B-4bit), highest 1.13
  (mlx-community/Muse-Glimmer-30B-OptiQ-4bit)
- *Prefill tok/s ratio (now/baseline):* 0.731 over 40 models; lowest 0.19
  (mlx-community/FastVLM-0.5B-bf16), highest 1.10
  (mlx-community/Ministral-3-14B-Instruct-2512-mxfp4)
- *Throughput noise band:* fixed ±15% fallback (insufficient history)

Run environment: current run on battery power for 43 of 43 models; current run
detected GPU activity outside harness inference at the checks around 3 of 43
models (up to 27%); baseline run detected GPU activity outside harness
inference at the checks around 34 of 51 models (up to 38%).

- New this run (no baseline): `ibm-granite/granite-vision-4.1-4b`,
  `mlx-community/Mistral-Small-4-119B-2603-4bit`,
  `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`
- In baseline, no longer in the cache: 11 models (targeted run against a
  full-sweep baseline)

| Model | Execution | Usability | Observation delta |
| --- | --- | --- | --- |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | completed | major concerns → no concerns detected | -labelled fields not detected; -cut off at token limit |

<details>
<summary>Generated text changed for 1 models</summary>

| Model | Decoding | Weights | Shared prefix (chars) | Length (chars) | Output tokens | Prompt token count |
| --- | --- | --- | --- | --- | --- | --- |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | sampled | 4-bit affine | 30 | 163 → 428 | 1,000 → 786 | same |

</details>

Generation tok/s outside the expected band for 8 models (mlx changed since the
baseline: afc03c511..56c2bf923):

| Model | Baseline tok/s | Now tok/s | Ratio | Expected band |
| --- | --- | --- | --- | --- |
| mlx-community/MiniCPM-V-4.7-1B-4bit | 308.9 | 251.0 | 0.81 | 262.6-355.3 (fallback) |
| mlx-community/North-Micro-Vision-Instruct-4bit | 223.3 | 159.6 | 0.71 | 189.8-256.8 (fallback) |
| mlx-community/nanoLLaVA-1.5-4bit | 308.4 | 209.0 | 0.68 | 262.1-354.6 (fallback) |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | 114.1 | 85.0 | 0.74 | 97.0-131.2 (fallback) |
| mlx-community/Qwen3.5-35B-A3B-4bit | 113.0 | 71.5 | 0.63 | 96.1-130.0 (fallback) |
| mlx-community/MiniCPM-V-4.6-4bit | 298.8 | 245.8 | 0.82 | 254.0-343.6 (fallback) |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | 51.6 | 37.4 | 0.72 | 43.9-59.4 (fallback) |
| mlx-community/InternVL3_5-1B-4bit | 355.4 | 297.0 | 0.84 | 302.1-408.7 (fallback) |

Largest prefill changes outside 0.85-1.15x (prefill seconds are prompt tokens
divided by prefill tok/s; one timed generation per sweep, so observations, not
demonstrated speed changes):

| Model | Baseline prefill s | Now prefill s | Prefill tok/s ratio |
| --- | --- | --- | --- |
| mlx-community/FastVLM-0.5B-bf16 | 0.10 | 0.54 | 0.19 |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | 0.06 | 0.27 | 0.21 |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit | 0.12 | 0.47 | 0.25 |
| TechnoBaptist/Ternary-Bonsai-2-27B-mlx-2bit | 0.72 | 2.54 | 0.28 |
| mlx-community/MiniCPM-V-4.6-4bit | 0.24 | 0.79 | 0.30 |

<details>
<summary>mlx: 2 upstream commit(s) since the baseline (afc03c511..56c2bf923)</summary>

- 56c2bf923 Fix FFT read/write past buffer end in partial threadgroups (#4740)
- 360078a8d Adding sliding window attention (#4734)

</details>

<details>
<summary>mlx-vlm: 4 upstream commit(s) since the baseline (4319ef2c8..e9af37662)</summary>

- e9af3766 Add MiniMax M3.1 Flash support  (#2580)
- 3fc4e5a9 perf: make d1 and d1-omni faster and use less memory (#2500)
- a4b58693 Expose generation timings in Responses API results (#2583)
- c184d277 fix: support ModelOpt NVFP4 AWQ checkpoints (#2582)

</details>

<details>
<summary>mlx-vlm commits touching the affected models' architectures and their imports</summary>

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
means an answer was generated but missed the requested form. Tokens in count
the image's tokens too, which drive prefill time; tokens out are those
generated (limit 1,000). Keywords are counted from the answer's Keywords
field.

| Model | Mechanical checks | Total | Gen tok/s | Peak GB | Prompt / output tok | Keywords | Observed | Native repro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ibm-granite/granite-vision-4.1-4b | no concerns detected | 2.30s | 68.0 tok/s | 9.7 | 721 / 55 | 9 | none | - |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | no concerns detected | 0.81s | 532 tok/s | 1.3 | 350 / 47 | 6 | none | - |
| mlx-community/AREX-2-4bit | no concerns detected | 5.94s | 33.3 tok/s | 17 | 421 / 80 | 8 | none | - |
| mlx-community/aya-vision-8b-4bit | no concerns detected | 4.12s | 98.9 tok/s | 6.5 | 2,909 / 82 | 10 | none | - |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | no concerns detected | 6.25s | 37.4 tok/s | 28 | 386 / 72 | 15 | none | - |
| mlx-community/gemma-4-12B-it-4bit | no concerns detected | 3.94s | 62.0 tok/s | 7.5 | 390 / 88 | 18 | none | - |
| mlx-community/gemma-4-26b-a4b-it-4bit | no concerns detected | 3.71s | 110 tok/s | 16 | 390 / 88 | 15 | none | - |
| mlx-community/gemma-4-e4b-it-4bit | no concerns detected | 2.53s | 122 tok/s | 5.9 | 386 / 67 | 14 | none | - |
| mlx-community/GLM-4.6V-Flash-4bit | no concerns detected | 2.94s | 86.2 tok/s | 8.0 | 499 / 49 | 5 | none | - |
| mlx-community/granite-4.0-3b-vision-4bit | no concerns detected | 1.30s | 171 tok/s | 4.7 | 734 / 58 | 10 | none | - |
| mlx-community/InternVL3-14B-4bit | no concerns detected | 5.10s | 56.1 tok/s | 10 | 3,435 / 90 | 18 | none | - |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit | no concerns detected | 2.01s | 217 tok/s | 3.6 | 344 / 79 | 10 | none | - |
| mlx-community/MiniCPM-o-4_5-4bit | no concerns detected | 2.37s | 106 tok/s | 7.0 | 177 / 99 | 11 | none | - |
| mlx-community/MiniCPM-V-4.6-4bit | no concerns detected | 3.16s | 246 tok/s | 3.1 | 330 / 331 | 10 | none | - |
| mlx-community/MiniCPM-V-4.7-1B-4bit | no concerns detected | 1.43s | 251 tok/s | 3.6 | 332 / 66 | 13 | none | - |
| mlx-community/MiniCPM-V-4.7-35B-A3B-4bit | no concerns detected | 4.28s | 75.6 tok/s | 24 | 332 / 92 | 18 | none | - |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | no concerns detected | 3.32s | 69.6 tok/s | 9.8 | 1,070 / 72 | 10 | none | - |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | no concerns detected | 1.93s | 201 tok/s | 4.5 | 1,069 / 102 | 12 | none | - |
| mlx-community/Mistral-Small-4-119B-2603-4bit | no concerns detected | 9.77s | 81.0 tok/s | 70 | 549 / 82 | 18 | none | - |
| mlx-community/Molmo2-8B-4bit | no concerns detected | 4.14s | 73.4 tok/s | 7.9 | 873 / 95 | 10 | none | - |
| mlx-community/MolmoPoint-8B-4bit | no concerns detected | 5.29s | 33.0 tok/s | 12 | 860 / 70 | 10 | none | - |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | no concerns detected | 35.14s | 25.5 tok/s | 25 | 545 / 786 | 13 | none | - |
| mlx-community/North-Micro-Vision-Instruct-4bit | no concerns detected | 1.79s | 160 tok/s | 3.5 | 418 / 78 | 12 | none | - |
| mlx-community/Phi-3.5-vision-instruct-bf16 | no concerns detected | 2.91s | 58.8 tok/s | 9.1 | 883 / 94 | 19 | none | - |
| mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit | no concerns detected | 3.74s | 79.6 tok/s | 23 | 409 / 81 | 18 | none | - |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | no concerns detected | 3.24s | 85.0 tok/s | 19 | 409 / 83 | 10 | none | - |
| mlx-community/Qwen3-VL-8B-Instruct-4bit | no concerns detected | 1.75s | 106 tok/s | 6.8 | 409 / 60 | 10 | none | - |
| mlx-community/Qwen3.5-35B-A3B-4bit | no concerns detected | 4.11s | 71.5 tok/s | 21 | 421 / 88 | 10 | none | - |
| mlx-community/Qwen3.8-27B-nvfp4 | no concerns detected | 6.24s | 31.9 tok/s | 17 | 421 / 109 | 13 | none | - |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | no concerns detected | 1.65s | 131 tok/s | 5.6 | 1,204 / 31 | 4 | none | - |
| nativ-community/Mage-VL-OptiQ-4bit | no concerns detected | 1.63s | 147 tok/s | 4.9 | 420 / 89 | 10 | none | - |
| nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit | no concerns detected | 5.36s | 36.4 tok/s | 17 | 538 / 87 | 18 | none | - |
| nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | no concerns detected | 3.41s | 138 tok/s | 21 | 421 / 65 | 10 | none | - |
| nativ-community/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-4bit | no concerns detected | 7.17s | 61.1 tok/s | 46 | 420 / 101 | 15 | none | - |
| nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 | no concerns detected | 30.06s | 29.7 tok/s | 81 | 420 / 89 | 13 | none | - |
| sahilchachra/LensVLM-9B-MXFP4 | no concerns detected | 2.67s | 103 tok/s | 6.8 | 421 / 111 | 10 | none | - |
| TechnoBaptist/Ternary-Bonsai-2-27B-mlx-2bit | no concerns detected | 6.26s | 39.9 tok/s | 11 | 421 / 94 | 14 | none | - |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | concerns detected | 6.78s | 119 tok/s | 18 | 545 / 540 | 16 | duplicate keywords | - |
| mlx-community/FastVLM-0.5B-bf16 | major concerns: answer format | 1.34s | 328 tok/s | 2.2 | 120 / 32 | - | labelled fields not detected | - |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns: answer format | 3.26s | insufficient sample | 7.0 | 3,512 / 14 | - | control tokens visible; labelled fields not detected | candidate |
| mlx-community/nanoLLaVA-1.5-4bit | major concerns: answer format | 1.27s | 209 tok/s | 1.2 | 116 / 97 | - | labelled fields not detected | - |
| mlx-community/InternVL3_5-1B-4bit | major concerns: generation | 4.54s | 297 tok/s | 2.3 | 3,438 / 1,000 | 144 | repeated text; cut off at token limit; duplicate keywords | candidate |
| mlx-community/X-Reasoner-7B-8bit | major concerns: generation | 17.15s | 64.5 tok/s | 10 | 511 / 1,000 | 189 | repeated text; cut off at token limit; duplicate keywords | candidate |

## Observation clusters

Observation signatures shared by two or more results requiring review.

| Observed result | Models |
| --- | --- |
| Response repeats the same text; Response appears cut off at the token limit; Repeated keyword entries | [mlx-community/InternVL3_5-1B-4bit](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-internvl35-1b-4bit), [mlx-community/X-Reasoner-7B-8bit](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-x-reasoner-7b-8bit) |

## Completed attempts requiring review

*History* dates each observation (a crash counts as one) over this model's
retained runs: 40 earlier retained runs from git
`HEAD:src/output/results.jsonl`, back to 2026-08-30; older runs were not read.
A run that did not attempt the model is skipped, and a report-only correction
of a run counts once. "Last N runs" and "N runs since" count consecutive runs
ending with this one; "first" is the earliest run read that showed it. The
table shows each model's longest-running observation; every observation's
dates are under *History by observation*.

| Model | Mechanical checks | Since baseline | History | Observed result | Evidence |
| --- | --- | --- | --- | --- | --- |
| mlx-community/InternVL3_5-1B-4bit | major concerns: generation | unchanged | all 3: 3 runs since 2026-10-16 | Response repeats the same text; Response appears cut off at the token limit; Duplicate keywords: british english, british english examples of metadata, british english examples of details, british english examples of context, british english examples of information, british english examples of data, british english examples of content, british english examples of examples | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-internvl35-1b-4bit) |
| mlx-community/X-Reasoner-7B-8bit | major concerns: generation | unchanged | duplicate keywords: 3 runs since 2026-10-16; +2 more, newest 2 runs since 2026-10-23 | Response repeats the same text; Response appears cut off at the token limit; Duplicate keywords: feline behavior study, feline comfort study, feline rest study, feline lifestyle study, feline domestic life study, feline behavior analysis, feline comfort analysis, feline rest analysis, feline lifestyle analysis, feline domestic life analysis, feline behavior observation, feline comfort observation, feline rest observation, feline lifestyle observation, feline domestic life observation, feline behavior research, feline comfort research, feline rest research, feline lifestyle research, feline domestic life research | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-x-reasoner-7b-8bit) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns: answer format | unchanged | all 2: 31+ runs since 2026-09-12 or earlier | Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description, keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit) |

<details>
<summary>History by observation</summary>

| Model | Observation | Persistence |
| --- | --- | --- |
| mlx-community/InternVL3_5-1B-4bit | duplicate keywords | last 3 runs (since 2026-10-16), first 2026-10-04 |
| mlx-community/InternVL3_5-1B-4bit | repeated text | last 3 runs (since 2026-10-16), first 2026-10-04 |
| mlx-community/InternVL3_5-1B-4bit | cut off at token limit | last 3 runs (since 2026-10-16) |
| mlx-community/X-Reasoner-7B-8bit | duplicate keywords | last 3 runs (since 2026-10-16), first 2026-09-06 |
| mlx-community/X-Reasoner-7B-8bit | repeated text | last 3 runs (since 2026-10-16), first 2026-08-30 |
| mlx-community/X-Reasoner-7B-8bit | cut off at token limit | last 2 runs (since 2026-10-23), first 2026-08-30 |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | labelled fields not detected | last 31+ runs (since 2026-09-12 or earlier) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | control tokens visible | last 31+ runs (since 2026-09-12 or earlier) |

</details>

## Completions without detected concerns

37 completions without detected concerns; 3 more completed with prompt-compliance observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 640 x 480 pixels, 173,131 bytes
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: seed:* 0
- *Trust remote code:* true
- *check_models version:* 0.17.91
- *check_models revision:* baa72d7e8e0334cc740fd4c23949ea8f757230f4
- *check_models source dirty:* false
- *mlx-vlm:* 0.7.10
- *mlx-vlm source revision:* e9af376626b04ba0cdad23b1476fd7a4a583d721
- *mlx:* 0.32.5.dev20261024+56c2bf923
- *mlx source revision:* 56c2bf923
- *transformers:* 5.20.2
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
