# mlx-vlm compatibility findings across 45 cached vision-language models

**For mlx-vlm maintainers:** no crashes need action; 2 other results need
reproducing with mlx-vlm alone before anything is reported (all 2 unchanged
since the baseline).

**Completed models:** 30 with no mechanical observations, 13 with
prompt-compliance observations only (labels, form or copied hint text), and 2
with observations mlx-vlm can produce (such as repetition, missing final
answers or visible control tokens); only those 2 are candidates for native
reproduction.

## Run summary

- *Run started:* 2026-10-31 23:36:41 GMT
- *Run finished:* 2026-10-31 23:48:40 GMT
- *Run duration:* 11m 58s
- *Time by phase:* generation 557s, model load 93s, prompt prep 38s, cleanup
  6s, outside the model loop 24s
- *Evaluation lane:* assisted
- *Prompt hints:* the image's description and keyword hints were included in
  the prompt, so field content may be copied from them rather than seen
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Input image:* JPEG, 6,656 x 9,483 pixels (63.1 MP), 60.5 MB
- *Models attempted:* 45
- *Sampling settings:* checkpoint generation_config.json values for 20 of 45
  completed models where the command line left them unset; harness defaults
  elsewhere
- *Completed:* 45
- *Crashed:* 0
- *Indeterminate:* 0
- *Crashes requiring action:* 0
- *Other results requiring review:* 2
- *Reached token limit:* 1 (1 with incomplete output)
- *Stopped early for repetition:* 1

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
- Capture date/time: 2026-10-17 15:44:34 UTC+01:00
- GPS: 51.751667°N, 1.252222°W

Descriptive hints:
- Description hint: A view of a historic stone gothic tower and rooftops partially framed by dense green trees under an overcast sky.
- Keyword hints: Architecture, Branch, Building, Chimney, Cloud, Cloudy sky, England, English countryside, Foliage, Historic Building, Leaves, Outdoors, Overcast, Oxford, Oxfordshire, Pinnacle, Roof, Rooftops, Scenic, Sky

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

- *Baseline:* 995582a3:src/output/results.jsonl
- *Baseline run timestamp:* 2026-10-30 14:22:29 GMT
- *Baseline check_models:* 0.17.98 @ b93edf990
- *Baseline mlx:* 0.32.5.dev20261030+1ba7f969a @ 1ba7f969a
- *Baseline mlx-vlm:* 0.7.11 @ 918763c68
- *Baseline transformers:* 5.22.1
- *Baseline python:* 3.14.7
- *Baseline hardware:* Apple M5 Max, 40 GPU cores, 128.0 GB RAM
- *Earlier retained runs on this mlx version:* none recorded on
  0.32.5.dev20261031+339279525
- *Models compared:* 43
- *Identical generated text:* 40 of 43 completed in both
- *Generated text changed:* 3 models (listed below)
- *Text changed, by decoding:* greedy 2 of 25; sampled, same settings and seed
  1 of 18
- *Greedy text changed, by checkpoint quantization:* quantized 1 of 20; not
  quantized 1 of 5
- *Generation tok/s ratio (now/baseline):* 0.976 over 42 models; lowest 0.90
  (mlx-community/MolmoPoint-8B-4bit), highest 1.18
  (mlx-community/diffusiongemma-26B-A4B-it-mxfp8)
- *Prefill tok/s ratio (now/baseline):* 1.130 over 43 models; lowest 0.58
  (mlx-community/Mistral-Small-4-119B-2603-4bit), highest 6.90
  (LiquidAI/LFM2.5-VL-450M-MLX-bf16)
- *Throughput noise band:* fixed ±15% fallback (insufficient history)

Run environment: current run detected GPU activity outside harness inference
at the checks around 39 of 45 models (up to 32%); baseline run detected GPU
activity outside harness inference at the checks around 19 of 43 models (up to
46%).

- New this run (no baseline): `LiquidAI/LFM3-VL-2B-MLX-8bit`,
  `mlx-community/Qwen3.6-35B-A3B-4bit`

No execution, usability, or observation-set changes against the baseline.

<details>
<summary>Generated text changed for 3 models</summary>

| Model | Decoding | Weights | Shared prefix (chars) | Length (chars) | Output tokens | Prompt token count |
| --- | --- | --- | --- | --- | --- | --- |
| mlx-community/granite-4.0-3b-vision-4bit | greedy | 4-bit affine | 274 | 312 → 304 | 68 → 67 | same |
| ibm-granite/granite-vision-4.1-4b | greedy | not quantized | 183 | 461 → 447 | 109 → 106 | same |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | sampled | 4-bit affine | 6 | 342 → 330 | 1,000 → 1,000 | same |

</details>

Generation tok/s outside the expected band for 1 model (mlx changed since the
baseline: 1ba7f969a..339279525):

| Model | Baseline tok/s | Now tok/s | Ratio | Expected band |
| --- | --- | --- | --- | --- |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | 46.8 | 55.4 | 1.18 | 39.7-53.8 (fallback) |

Largest prefill changes outside 0.85-1.15x (prefill seconds are prompt tokens
divided by prefill tok/s; one timed generation per sweep, so observations, not
demonstrated speed changes):

| Model | Baseline prefill s | Now prefill s | Prefill tok/s ratio |
| --- | --- | --- | --- |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | 0.78 | 0.11 | 6.90 |
| mlx-community/FastVLM-0.5B-bf16 | 1.00 | 0.22 | 4.58 |
| mlx-community/MiniCPM-V-4.6-4bit | 1.25 | 0.37 | 3.34 |
| mlx-community/nanoLLaVA-1.5-4bit | 0.28 | 0.09 | 2.97 |
| ibm-granite/granite-vision-4.1-4b | 1.73 | 0.74 | 2.32 |

<details>
<summary>mlx: 6 upstream commit(s) since the baseline (1ba7f969a..339279525)</summary>

- 339279525 Fix sharded-to-all bias and integer split positions on quantized
  layers (#4715)
- 700520ae7 Fix Metal expm1 argument reduction accuracy (#4780)
- 4ca8d5d5a Cache generated source code and hash in custom kernels (#4568)
- f7f86bd71 Launch fast.cross_entropy in slices past the Metal grid limit
  (#4776)
- 93e65c0b1 [Metal] Fix fill loop in `layer_norm` (#4772)
- 6a304d14f Fix negative padding with input_dilation &gt; 1 in conv_general
  (#4688)

</details>

<details>
<summary>mlx-vlm: 10 upstream commit(s) since the baseline (918763c68..4693cd0ea)</summary>

- 4693cd0e Merge pull request #2636 from Blaizzy/pc/mage-vl-cowork
- 3072d7f0 Restore shared model test files to PR base
- 52a1764f Revert "fix: match Mage text RoPE and SiLU rounding"
- 8fb85118 fix: match Mage text RoPE and SiLU rounding
- ae763da4 fix: match PyTorch BF16 arithmetic in Mage-VL vision
- 34bfba5d Merge pull request #2633 from Blaizzy/pc/openai-stream-keepalive
- 63fc4125 ci: move tests to supported macOS 15 runners
- 99bee88d Add native Mage-VL-Cowork support
- 4eb43997 Limit keepalives to the OpenAI streaming layer
- 1f9d2fe8 Keep OpenAI streams alive during long prefills

</details>

<details>
<summary>mlx-vlm commits touching the affected models' architectures and their imports</summary>

- `mlx-community/granite-4.0-3b-vision-4bit` (`granite4_vision`; imports
  `base.py`, `rope_utils.py`): no commits touched
  `mlx_vlm/models/granite4_vision/` or the entries it imports
- `ibm-granite/granite-vision-4.1-4b` (`granite4_vision`; imports `base.py`,
  `rope_utils.py`): no commits touched `mlx_vlm/models/granite4_vision/` or
  the entries it imports
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
| ibm-granite/granite-vision-4.1-4b | no concerns detected | 5.39s | 63.7 tok/s | 9.6 | 1,430 / 106 | 20 (20 from hints) | 54% | none | - |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | no concerns detected | 2.49s | 486 tok/s | 1.9 | 2,104 / 69 | 8 (3 from hints) | 35% | none | - |
| mlx-community/AREX-2-4bit | no concerns detected | 58.81s | 28.8 tok/s | 21 | 16,594 / 114 | 20 (20 from hints) | 44% | none | - |
| mlx-community/aya-vision-8b-4bit | no concerns detected | 5.15s | 97.1 tok/s | 6.5 | 2,081 / 117 | 14 (8 from hints) | 36% | none | - |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | no concerns detected | 7.20s | 55.4 tok/s | 28 | 586 / 76 | 14 (14 from hints) | 55% | none | - |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | no concerns detected | 11.74s | 111 tok/s | 19 | 1,596 / 855 | 18 (15 from hints) | 19% | none | - |
| mlx-community/gemma-4-12B-it-4bit | no concerns detected | 5.54s | 60.3 tok/s | 7.6 | 590 / 104 | 20 (20 from hints) | 30% | none | - |
| mlx-community/gemma-4-26b-a4b-it-4bit | no concerns detected | 5.08s | 108 tok/s | 16 | 590 / 75 | 16 (13 from hints) | 57% | none | - |
| mlx-community/gemma-4-e4b-it-4bit | no concerns detected | 4.13s | 119 tok/s | 6.0 | 586 / 83 | 14 (6 from hints) | 18% | none | - |
| mlx-community/GLM-4.6V-Flash-4bit | no concerns detected | 9.89s | 75.4 tok/s | 8.7 | 6,343 / 104 | 9 (6 from hints) | 29% | none | - |
| mlx-community/InternVL3-14B-4bit | no concerns detected | 6.48s | 54.7 tok/s | 10 | 2,104 / 108 | 18 (17 from hints) | 38% | none | - |
| mlx-community/MiniCPM-o-4_5-4bit | no concerns detected | 3.63s | 100 tok/s | 7.1 | 382 / 94 | 19 (19 from hints) | 20% | none | - |
| mlx-community/MiniCPM-V-4.7-35B-A3B-4bit | no concerns detected | 6.35s | 78.1 tok/s | 25 | 927 / 108 | 18 (13 from hints) | 33% | none | - |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | no concerns detected | 7.04s | 63.3 tok/s | 13 | 3,046 / 119 | 16 (4 from hints) | 46% | none | - |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | no concerns detected | 4.24s | 179 tok/s | 8.2 | 3,045 / 153 | 14 (2 from hints) | 22% | none | - |
| mlx-community/Mistral-Small-4-119B-2603-4bit | no concerns detected | 23.29s | 62.8 tok/s | 73 | 2,525 / 120 | 16 (8 from hints) | 13% | none | - |
| mlx-community/Molmo2-8B-4bit | no concerns detected | 5.69s | 67.4 tok/s | 8.0 | 1,318 / 133 | 20 (20 from hints) | 36% | none | - |
| mlx-community/MolmoPoint-8B-4bit | no concerns detected | 9.62s | 27.1 tok/s | 13 | 3,136 / 112 | 16 (5 from hints) | 13% | none | - |
| mlx-community/North-Micro-Vision-Instruct-4bit | no concerns detected | 5.70s | 160 tok/s | 3.7 | 4,044 / 110 | 20 (20 from hints) | 44% | none | - |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | no concerns detected | 35.33s | 75.4 tok/s | 23 | 16,578 / 155 | 19 (19 from hints) | 27% | none | - |
| mlx-community/Qwen3-VL-8B-Instruct-4bit | no concerns detected | 37.09s | 66.3 tok/s | 11 | 16,578 / 87 | 10 (3 from hints) | 40% | none | - |
| mlx-community/Qwen3.5-35B-A3B-4bit | no concerns detected | 37.18s | 74.4 tok/s | 25 | 16,594 / 112 | 18 (16 from hints) | 0% | none | - |
| mlx-community/Qwen3.6-35B-A3B-4bit | no concerns detected | 37.51s | 73.8 tok/s | 25 | 16,594 / 127 | 24 (14 from hints) | 10% | none | - |
| mlx-community/Qwen3.8-27B-nvfp4 | no concerns detected | 57.19s | 29.0 tok/s | 21 | 16,594 / 121 | 17 (12 from hints) | 0% | none | - |
| nativ-community/Mage-VL-OptiQ-4bit | no concerns detected | 4.98s | 122 tok/s | 5.4 | 4,173 / 99 | 20 (20 from hints) | 25% | none | - |
| nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit | no concerns detected | 7.81s | 33.6 tok/s | 18 | 1,313 / 111 | 19 (10 from hints) | 50% | none | - |
| nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | no concerns detected | 7.53s | 127 tok/s | 23 | 3,641 / 135 | 20 (20 from hints) | 52% | none | - |
| nativ-community/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-4bit | no concerns detected | 13.16s | 60.3 tok/s | 52 | 3,640 / 123 | 20 (20 from hints) | 60% | none | - |
| nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 | no concerns detected | 50.20s | 30.1 tok/s | 87 | 3,640 / 133 | 20 (20 from hints) | 44% | none | - |
| TechnoBaptist/Ternary-Bonsai-2-27B-mlx-2bit | no concerns detected | 70.09s | 33.5 tok/s | 17 | 16,594 / 118 | 18 (18 from hints) | 20% | none | - |
| LiquidAI/LFM3-VL-2B-MLX-8bit | concerns detected | 6.74s | 191 tok/s | 5.0 | 16,572 / 91 | 20 (20 from hints) | 100% | prompt hint repeated | - |
| mlx-community/granite-4.0-3b-vision-4bit | concerns detected | 3.39s | 170 tok/s | 4.6 | 1,454 / 67 | 10 (5 from hints) | 100% | prompt hint repeated | - |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit | concerns detected | 3.38s | 202 tok/s | 4.0 | 2,096 / 83 | 15 (15 from hints) | 90% | prompt hint repeated | - |
| mlx-community/MiniCPM-V-4.6-4bit | concerns detected | 6.15s | 240 tok/s | 3.1 | 925 / 752 | 21 (10 from hints) | 42% | duplicate keywords | - |
| mlx-community/MiniCPM-V-4.7-1B-4bit | concerns detected | 3.34s | 252 tok/s | 3.7 | 927 / 114 | 25 (18 from hints) | 34% | duplicate keywords | - |
| mlx-community/Phi-3.5-vision-instruct-bf16 | concerns detected | 5.27s | 55.8 tok/s | 9.3 | 1,120 / 149 | 20 (15 from hints) | 27% | duplicate keywords | - |
| mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit | concerns detected | 23.98s | 66.2 tok/s | 26 | 12,683 / 90 | 18 (16 from hints) | 86% | prompt hint repeated | - |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | concerns detected | 3.37s | 121 tok/s | 5.6 | 1,419 / 74 | 18 (18 from hints) | 100% | prompt hint repeated | - |
| mlx-community/X-Reasoner-7B-8bit | concerns detected | 20.56s | 56.6 tok/s | 14 | 16,589 / 229 | 51 (20 from hints) | 53% | duplicate keywords | - |
| sahilchachra/LensVLM-9B-MXFP4 | concerns detected | 4.90s | 98.4 tok/s | 7.3 | 1,802 / 83 | 15 (15 from hints) | 100% | prompt hint repeated | - |
| mlx-community/FastVLM-0.5B-bf16 | major concerns: answer format | 3.45s | 342 tok/s | 2.1 | 325 / 22 | - | 100% of answer | labelled fields not detected; prompt hint repeated | - |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns: answer format | 3.72s | 98.7 tok/s | 6.7 | 2,186 / 29 | - | 83% of answer | control tokens visible; labelled fields not detected; prompt hint repeated | not yet checked (earlier check out of date) |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | major concerns: answer format | 54.83s | 23.6 tok/s | 25 | 4,367 / 1,000 | - | 21% | labelled fields not detected; cut off at token limit | - |
| mlx-community/nanoLLaVA-1.5-4bit | major concerns: answer format | 2.89s | 199 tok/s | 1.8 | 321 / 148 | - | 19% | labelled fields not detected | - |
| mlx-community/InternVL3_5-1B-4bit | major concerns: generation | 3.21s | 303 tok/s | 2.1 | 2,107 / 200 | 57 (10 from hints) | 29% | repeated text; stopped early: repeating; duplicate keywords | not yet checked |

## Native reproduction checks

Recorded in native_repro_notes.yaml for this run's model revisions, builds
(mlx 0.32.5.dev20261031+339279525 @ 339279525; mlx-vlm 0.7.11 @ 4693cd0ea;
transformers 5.22.1) and experiment inputs (image, prompt, generation
settings, and each model's rendered prompt and effective generation
arguments). A check made on another revision, build or experiment, or for a
model whose inputs were not recorded, does not apply and is not listed.

No recorded check applies to this run.

## Human accuracy notes

0 of 45 outputs reviewed by a person for this exact image, model revision and
output (accuracy_notes.yaml); the others are unreviewed; 1 note for other
images or models. Each note is a person's judgement against the image, not one
of the mechanical checks.

## Completed attempts requiring review

*History* dates each observation (a crash counts as one) over this model's
retained runs: 43 earlier retained runs from git
`HEAD:src/output/results.jsonl`, back to 2026-08-30; older runs were not read.
A run that did not attempt the model is skipped, and a report-only correction
of a run counts once. "Last N runs" and "N runs since" count consecutive runs
ending with this one; "first" is the earliest run read that showed it. The
table shows each model's longest-running observation; every observation's
dates are under *History by observation*.

| Model | Mechanical checks | Since baseline | History | Observed result | Evidence |
| --- | --- | --- | --- | --- | --- |
| mlx-community/InternVL3_5-1B-4bit | major concerns: generation | unchanged | duplicate keywords: 6 runs since 2026-10-16; +2 more, newest 3 runs since 2026-10-28 | Response repeats the same text; Generation was stopped early after sustained repeated output; Duplicate keywords: sky, sky view, skyline | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-internvl35-1b-4bit) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns: answer format | unchanged | labelled fields not detected: 34+ runs since 2026-09-12 or earlier; +2 more, newest 3 runs since 2026-10-28 | Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description, keywords; Repeats the prompt's hint text: description | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit) |

<details>
<summary>History by observation</summary>

| Model | Observation | Persistence |
| --- | --- | --- |
| mlx-community/InternVL3_5-1B-4bit | duplicate keywords | last 6 runs (since 2026-10-16), first 2026-10-04 |
| mlx-community/InternVL3_5-1B-4bit | repeated text | last 6 runs (since 2026-10-16), first 2026-10-04 |
| mlx-community/InternVL3_5-1B-4bit | stopped early: repeating | last 3 runs (since 2026-10-28), first 2026-10-04 |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | labelled fields not detected | last 34+ runs (since 2026-09-12 or earlier) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | prompt hint repeated | last 3 runs (since 2026-10-28) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | control tokens visible | last 34+ runs (since 2026-09-12 or earlier) |

</details>

## Completions without detected concerns

30 completions without detected concerns; 13 more completed with prompt-compliance observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 6,656 x 9,483 pixels, 60,507,861 bytes
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: seed:* 0
- *Trust remote code:* true
- *check_models version:* 0.17.105
- *check_models revision:* 995582a343f75f934f0c4ee7858dd07f892c7e6e
- *check_models source dirty:* false
- *mlx-vlm:* 0.7.11
- *mlx-vlm source revision:* 4693cd0eaf022ae7ef61c6fdb55e17e6399879a7
- *mlx:* 0.32.5.dev20261031+339279525
- *mlx source revision:* 339279525
- *transformers:* 5.22.1
- *macOS Version:* 27.2
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
