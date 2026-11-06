# mlx-vlm compatibility findings across 41 cached vision-language models

**For mlx-vlm maintainers:** no crashes need action; 3 other results need
reproducing with mlx-vlm alone before anything is reported (2 of 3 unchanged
since the baseline).

**Completed models:** 28 with no mechanical observations, 10 with
prompt-compliance observations only (labels, form or copied hint text), and 3
with observations mlx-vlm can produce (such as repetition, missing final
answers or visible control tokens); only those 3 are candidates for native
reproduction.

## Run summary

- *Run started:* 2026-11-06 14:32:53 GMT
- *Run finished:* 2026-11-06 14:43:35 GMT
- *Run duration:* 10m 42s
- *Time by phase:* generation 489s, model load 88s, prompt prep 35s, cleanup
  5s, outside the model loop 24s
- *Evaluation lane:* assisted
- *Prompt hints:* the image's description and keyword hints were included in
  the prompt, so field content may be copied from them rather than seen
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Input image:* JPEG, 6,656 x 9,483 pixels (63.1 MP), 60.5 MB
- *Models attempted:* 41
- *Sampling settings:* checkpoint generation_config.json values for 17 of 41
  completed models where the command line left them unset; harness defaults
  elsewhere
- *Completed:* 41
- *Crashed:* 0
- *Indeterminate:* 0
- *Crashes requiring action:* 0
- *Other results requiring review:* 3
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

- *Baseline:* b21c07f2:src/output/results.jsonl
- *Baseline run timestamp:* 2026-10-31 23:48:40 GMT
- *Baseline check_models:* 0.17.105 @ 995582a34
- *Baseline mlx:* 0.32.5.dev20261031+339279525 @ 339279525
- *Baseline mlx-vlm:* 0.7.11 @ 4693cd0ea
- *Baseline transformers:* 5.22.1
- *Baseline python:* 3.14.7
- *Baseline hardware:* Apple M5 Max, 40 GPU cores, 128.0 GB RAM
- *Earlier retained runs on this mlx version:* none recorded on
  0.32.5.dev20261106+9f806b360
- *Models compared:* 41
- *Identical generated text:* 35 of 41 completed in both
- *Generated text changed:* 6 models (listed below)
- *Text changed, by decoding:* greedy 2 of 24; sampled, same settings and seed
  4 of 17
- *Greedy text changed, by checkpoint quantization:* quantized 2 of 19; not
  quantized 0 of 5
- *Generation tok/s ratio (now/baseline):* 1.014 over 40 models; lowest 0.75
  (mlx-community/FastVLM-0.5B-bf16), highest 1.12
  (mlx-community/Mistral-Small-4-119B-2603-4bit)
- *Prefill tok/s ratio (now/baseline):* 0.997 over 41 models; lowest 0.24
  (mlx-community/MiniCPM-V-4.7-1B-4bit), highest 2.17
  (mlx-community/diffusiongemma-26B-A4B-it-mxfp8)
- *Throughput noise band:* fixed ±15% fallback (insufficient history)

Run environment: current run on battery power for 41 of 41 models; current run
detected GPU activity outside harness inference at the checks around 19 of 41
models (up to 37%); baseline run detected GPU activity outside harness
inference at the checks around 39 of 45 models (up to 32%).

- In baseline, no longer in the cache: `LiquidAI/LFM3-VL-2B-MLX-8bit`,
  `mlx-community/MiniCPM-V-4.6-4bit`, `mlx-community/X-Reasoner-7B-8bit`,
  `nativ-community/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-4bit`

| Model | Execution | Usability | Observation delta |
| --- | --- | --- | --- |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | completed | no concerns detected → concerns detected | +duplicate keywords |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | completed | major concerns | +role tokens visible; +control tokens visible |

<details>
<summary>Generated text changed for 6 models</summary>

| Model | Decoding | Weights | Shared prefix (chars) | Length (chars) | Output tokens | Prompt token count |
| --- | --- | --- | --- | --- | --- | --- |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | greedy | 8-bit mxfp8 | 128 | 345 → 317 | 76 → 70 | same |
| mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit | greedy | 4-bit affine | 35 | 376 → 362 | 90 → 85 | same |
| mlx-community/Qwen3.6-35B-A3B-4bit | sampled | 4-bit affine | 72 | 600 → 408 | 127 → 104 | same |
| mlx-community/Qwen3.5-35B-A3B-4bit | sampled | 4-bit affine | 88 | 509 → 431 | 112 → 125 | same |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | sampled | 4-bit affine | 0 | 330 → 4,213 | 1,000 → 1,000 | same |
| mlx-community/Qwen3.8-27B-nvfp4 | sampled | 4-bit nvfp4 | 105 | 505 → 526 | 121 → 121 | same |

</details>

Generation tok/s outside the expected band for 1 model (mlx changed since the
baseline: 339279525..9f806b360):

| Model | Baseline tok/s | Now tok/s | Ratio | Expected band |
| --- | --- | --- | --- | --- |
| mlx-community/FastVLM-0.5B-bf16 | 342.4 | 257.9 | 0.75 | 291.0-393.7 (fallback) |

Largest prefill changes outside 0.85-1.15x (prefill seconds are prompt tokens
divided by prefill tok/s; one timed generation per sweep, so observations, not
demonstrated speed changes):

| Model | Baseline prefill s | Now prefill s | Prefill tok/s ratio |
| --- | --- | --- | --- |
| mlx-community/MiniCPM-V-4.7-1B-4bit | 0.24 | 1.03 | 0.24 |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | 0.11 | 0.43 | 0.26 |
| mlx-community/FastVLM-0.5B-bf16 | 0.22 | 0.58 | 0.38 |
| mlx-community/nanoLLaVA-1.5-4bit | 0.09 | 0.23 | 0.41 |
| mlx-community/gemma-4-e4b-it-4bit | 0.19 | 0.41 | 0.45 |

<details>
<summary>mlx: 17 upstream commit(s) since the baseline (339279525..9f806b360)</summary>

- 9f806b360 [CUDA] Change gradient layout if second argument is transposed
  (#4822)
- a159fb8f6 [CPU] Fix accumulation precision in affine quantized matmul
  (#4803)
- 3a601e0ba docs: Fix keyword-only parameters in op signatures (#4816)
- f06dd4dcb Fix row offsets and partition stride in GPU sort (#4807)
- c69303b00 docs: random.normal is not bit-exact across GPU architectures
  (#4810)
- 48143570b [Metal] Route qqmm without global scales like quantized_matmul
  (#4796)
- c823b0125 Respect MLX_ENABLE_TF32 in gated_delta_update (#4792)
- b7779c158 [CUDA] Sequential kernel for gated_delta_update (#4783)
- ba8c005c1 [Metal] Add affine qmv_mma for 3-16 rows (#4787)
- 55626cb48 Validate trace input ids match trace inputs in import_function
  (#4790)
- e80ca687e [CUDA] Fix `gather_mm` with sorted_indices=True (#4771)
- d3aa895f1 [CUDA] top left mask for sdpa backward (#4820)
- a8f66b590 Keep double precision in CPU exp, sin, cos, erf (#4789)
- 9d5b21d31 Check output size for 64-bit indexing in GPU SliceUpdate (#4794)
- 3d2fbcdf1 Add freqs and partial_rotary_factor support to RoPE (#4732)
- ... and 2 more (see results.jsonl)

</details>

<details>
<summary>mlx-vlm: 138 upstream commit(s) since the baseline (4693cd0ea..dcea78779)</summary>

- dcea7877 Merge pull request #2535 from
  nicoloangileri/fix/muse-glimmer-processor-save
- 5cdc659b Merge pull request #2665 from pujitha24/auto/issue-2660
- 72309403 Bump version to 0.7.12 (#2663)
- 329084ae feat: serve d1-2B and d1-omni-2B on /v1/decisions (#2657)
- 4a991d7f fix: update project references after Neywa Labs transfer (#2667)
- 003311ec Merge pull request #2473 from AkshayShah03/fix-eagle3-bind-no-embed
- 12c4d7d5 Merge pull request #2486 from
  AkshayShah03/fix-anthropic-stream-stop-sequences
- 2c1eabf0 Merge branch 'main' into fix-eagle3-bind-no-embed
- dbf9473a Merge main into fix-anthropic-stream-stop-sequences
- a1ccb1af Fix streamed stop boundaries for tools and overlapping markers
- 472b48fb Merge pull request #2476 from AkshayShah03/fix-paddleocr-lora
- a477f23c Merge pull request #2492 from
  AkshayShah03/fix-phi3v-multiturn-image-numbering
- f0c0d2a4 Merge main into fix-phi3v-multiturn-image-numbering
- 0ed40c2b Merge branch 'main' into fix-paddleocr-lora
- 022713ae Merge pull request #2484 from
  AkshayShah03/fix-max-completion-tokens
- ... and 123 more (see results.jsonl)

</details>

<details>
<summary>mlx-vlm commits touching the affected models' architectures and their imports</summary>

- `mlx-community/diffusiongemma-26B-A4B-it-mxfp8` (`diffusion_gemma`; imports
  `base.py`, `cache.py`, `diffusion_visualizer.py`, `gemma4`, `qwen3_vl`,
  `rope_utils.py`, `switch_layers.py`): 08b7fd87 Generalize trailing-logit
  projection and greedy selection (#2664); 0a69de95 Support resumable
  diffusion generation at block boundaries (#2662); 3eab0cd1 fix: scale the
  spatial dimensions when interpolating position embeddings; e5297aa0 Merge
  upstream main into fix/processor-rescale-domain; 62632826 fix: preserve the
  input value domain through video frame resizing; b953dcf8 Merge pull request
  #2658 from ahuzmeza/mlxvlm-rope-text-only-siblings; d5f19134 fix: preserve
  one-token MRoPE positions with padding; d3a16c34 Keep resumed DFlash and MTP
  requests in shared speculative batches (#2648); 85a44641 Merge pull request
  #2587 from pujitha24/auto/issue-2517; fb3b4623 Reduce APC merge memory and
  preserve custom checkpoints (#2646); 50113346 Make the LFM3-VL processor
  follow the transformers processor contract; b167017e fix(qwen3_vl): compute
  text-only rope deltas from unpadded length
- `mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit` (`qwen3_omni_moe`; imports
  `base.py`, `cache.py`, `mlp.py`, `qwen3_vl`, `rope_utils.py`,
  `switch_layers.py`): 08b7fd87 Generalize trailing-logit projection and
  greedy selection (#2664); 3eab0cd1 fix: scale the spatial dimensions when
  interpolating position embeddings; e5297aa0 Merge upstream main into
  fix/processor-rescale-domain; 62632826 fix: preserve the input value domain
  through video frame resizing; b953dcf8 Merge pull request #2658 from
  ahuzmeza/mlxvlm-rope-text-only-siblings; d5f19134 fix: preserve one-token
  MRoPE positions with padding; d3a16c34 Keep resumed DFlash and MTP requests
  in shared speculative batches (#2648); 85a44641 Merge pull request #2587
  from pujitha24/auto/issue-2517; fb3b4623 Reduce APC merge memory and
  preserve custom checkpoints (#2646); 50113346 Make the LFM3-VL processor
  follow the transformers processor contract; b167017e fix(qwen3_vl): compute
  text-only rope deltas from unpadded length
- `mlx-community/Qwen3.6-35B-A3B-4bit` (`qwen3_5_moe`; imports `base.py`,
  `qwen3_5`, `qwen3_vl`, `switch_layers.py`): 08b7fd87 Generalize
  trailing-logit projection and greedy selection (#2664); 3eab0cd1 fix: scale
  the spatial dimensions when interpolating position embeddings; e5297aa0
  Merge upstream main into fix/processor-rescale-domain; 62632826 fix:
  preserve the input value domain through video frame resizing; 98bcc2fd Merge
  pull request #2649 from xocialize/qwen3_5-vlm-yarn; 27139d32 fix(qwen3_5):
  default YaRN original context length; d5f19134 fix: preserve one-token MRoPE
  positions with padding; a78e7ef6 qwen3_5: apply YaRN from rope_parameters on
  the VLM rotary; 3bfc0aa5 fix: text-only MRoPE deltas ignore left padding in
  sibling models; 85a44641 Merge pull request #2587 from
  pujitha24/auto/issue-2517; 50113346 Make the LFM3-VL processor follow the
  transformers processor contract; b167017e fix(qwen3_vl): compute text-only
  rope deltas from unpadded length
- `mlx-community/Qwen3.5-35B-A3B-4bit` (`qwen3_5_moe`; imports `base.py`,
  `qwen3_5`, `qwen3_vl`, `switch_layers.py`): 08b7fd87 Generalize
  trailing-logit projection and greedy selection (#2664); 3eab0cd1 fix: scale
  the spatial dimensions when interpolating position embeddings; e5297aa0
  Merge upstream main into fix/processor-rescale-domain; 62632826 fix:
  preserve the input value domain through video frame resizing; 98bcc2fd Merge
  pull request #2649 from xocialize/qwen3_5-vlm-yarn; 27139d32 fix(qwen3_5):
  default YaRN original context length; d5f19134 fix: preserve one-token MRoPE
  positions with padding; a78e7ef6 qwen3_5: apply YaRN from rope_parameters on
  the VLM rotary; 3bfc0aa5 fix: text-only MRoPE deltas ignore left padding in
  sibling models; 85a44641 Merge pull request #2587 from
  pujitha24/auto/issue-2517; 50113346 Make the LFM3-VL processor follow the
  transformers processor contract; b167017e fix(qwen3_vl): compute text-only
  rope deltas from unpadded length
- `mlx-community/Muse-Glimmer-30B-OptiQ-4bit` (`muse_glimmer`; imports
  `activations.py`, `base.py`, `cache.py`, `rope_utils.py`): dcea7877 Merge
  pull request #2535 from nicoloangileri/fix/muse-glimmer-processor-save;
  08b7fd87 Generalize trailing-logit projection and greedy selection (#2664);
  3eab0cd1 fix: scale the spatial dimensions when interpolating position
  embeddings; d3a16c34 Keep resumed DFlash and MTP requests in shared
  speculative batches (#2648); fb3b4623 Reduce APC merge memory and preserve
  custom checkpoints (#2646); 4bb49d4e Keep the model config out of the Muse
  Glimmer processor config
- `mlx-community/Qwen3.8-27B-nvfp4` (`qwen3_5`; imports `activations.py`,
  `base.py`, `cache.py`, `quantized_verifier.py`, `qwen3_vl`,
  `rope_utils.py`): 08b7fd87 Generalize trailing-logit projection and greedy
  selection (#2664); 3eab0cd1 fix: scale the spatial dimensions when
  interpolating position embeddings; e5297aa0 Merge upstream main into
  fix/processor-rescale-domain; 62632826 fix: preserve the input value domain
  through video frame resizing; 98bcc2fd Merge pull request #2649 from
  xocialize/qwen3_5-vlm-yarn; 27139d32 fix(qwen3_5): default YaRN original
  context length; b953dcf8 Merge pull request #2658 from
  ahuzmeza/mlxvlm-rope-text-only-siblings; d5f19134 fix: preserve one-token
  MRoPE positions with padding; d3a16c34 Keep resumed DFlash and MTP requests
  in shared speculative batches (#2648); a78e7ef6 qwen3_5: apply YaRN from
  rope_parameters on the VLM rotary; 3bfc0aa5 fix: text-only MRoPE deltas
  ignore left padding in sibling models; 85a44641 Merge pull request #2587
  from pujitha24/auto/issue-2517; fb3b4623 Reduce APC merge memory and
  preserve custom checkpoints (#2646); 50113346 Make the LFM3-VL processor
  follow the transformers processor contract; b167017e fix(qwen3_vl): compute
  text-only rope deltas from unpadded length

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
| ibm-granite/granite-vision-4.1-4b | no concerns detected | 6.03s | 65.6 tok/s | 9.6 | 1,430 / 106 | 20 (20 from hints) | 54% | none | - |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | no concerns detected | 2.57s | 487 tok/s | 1.9 | 2,104 / 69 | 8 (3 from hints) | 35% | none | - |
| mlx-community/AREX-2-4bit | no concerns detected | 48.99s | 30.3 tok/s | 21 | 16,594 / 114 | 20 (20 from hints) | 44% | none | - |
| mlx-community/aya-vision-8b-4bit | no concerns detected | 5.30s | 102 tok/s | 6.5 | 2,081 / 117 | 14 (8 from hints) | 36% | none | - |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | no concerns detected | 11.58s | 116 tok/s | 19 | 1,596 / 855 | 18 (15 from hints) | 19% | none | - |
| mlx-community/gemma-4-12B-it-4bit | no concerns detected | 5.67s | 62.7 tok/s | 7.6 | 590 / 104 | 20 (20 from hints) | 30% | none | - |
| mlx-community/gemma-4-26b-a4b-it-4bit | no concerns detected | 5.04s | 111 tok/s | 16 | 590 / 75 | 16 (13 from hints) | 57% | none | - |
| mlx-community/gemma-4-e4b-it-4bit | no concerns detected | 4.28s | 124 tok/s | 6.0 | 586 / 83 | 14 (6 from hints) | 18% | none | - |
| mlx-community/GLM-4.6V-Flash-4bit | no concerns detected | 9.48s | 78.5 tok/s | 8.7 | 6,343 / 104 | 9 (6 from hints) | 29% | none | - |
| mlx-community/InternVL3-14B-4bit | no concerns detected | 6.30s | 55.2 tok/s | 10 | 2,104 / 108 | 18 (17 from hints) | 38% | none | - |
| mlx-community/MiniCPM-o-4_5-4bit | no concerns detected | 3.72s | 100 tok/s | 7.1 | 382 / 94 | 19 (19 from hints) | 20% | none | - |
| mlx-community/MiniCPM-V-4.7-35B-A3B-4bit | no concerns detected | 6.53s | 76.0 tok/s | 25 | 927 / 108 | 18 (13 from hints) | 33% | none | - |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | no concerns detected | 6.81s | 62.9 tok/s | 13 | 3,046 / 119 | 16 (4 from hints) | 46% | none | - |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | no concerns detected | 4.28s | 177 tok/s | 8.2 | 3,045 / 153 | 14 (2 from hints) | 22% | none | - |
| mlx-community/Mistral-Small-4-119B-2603-4bit | no concerns detected | 22.59s | 70.2 tok/s | 73 | 2,525 / 120 | 16 (8 from hints) | 13% | none | - |
| mlx-community/Molmo2-8B-4bit | no concerns detected | 6.49s | 67.9 tok/s | 8.0 | 1,318 / 133 | 20 (20 from hints) | 36% | none | - |
| mlx-community/MolmoPoint-8B-4bit | no concerns detected | 10.67s | 26.8 tok/s | 13 | 3,136 / 112 | 16 (5 from hints) | 13% | none | - |
| mlx-community/North-Micro-Vision-Instruct-4bit | no concerns detected | 5.60s | 159 tok/s | 3.7 | 4,044 / 110 | 20 (20 from hints) | 44% | none | - |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | no concerns detected | 34.96s | 77.1 tok/s | 23 | 16,578 / 155 | 19 (19 from hints) | 27% | none | - |
| mlx-community/Qwen3-VL-8B-Instruct-4bit | no concerns detected | 37.26s | 66.5 tok/s | 11 | 16,578 / 87 | 10 (3 from hints) | 40% | none | - |
| mlx-community/Qwen3.5-35B-A3B-4bit | no concerns detected | 34.86s | 73.5 tok/s | 25 | 16,594 / 125 | 18 (8 from hints) | 0% | none | - |
| mlx-community/Qwen3.6-35B-A3B-4bit | no concerns detected | 34.23s | 72.9 tok/s | 25 | 16,594 / 104 | 19 (16 from hints) | 20% | none | - |
| mlx-community/Qwen3.8-27B-nvfp4 | no concerns detected | 53.71s | 29.8 tok/s | 21 | 16,594 / 121 | 16 (12 from hints) | 0% | none | - |
| nativ-community/Mage-VL-OptiQ-4bit | no concerns detected | 4.94s | 128 tok/s | 5.4 | 4,173 / 99 | 20 (20 from hints) | 25% | none | - |
| nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit | no concerns detected | 7.93s | 35.8 tok/s | 18 | 1,313 / 111 | 19 (10 from hints) | 50% | none | - |
| nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | no concerns detected | 7.00s | 141 tok/s | 23 | 3,641 / 135 | 20 (20 from hints) | 52% | none | - |
| nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 | no concerns detected | 51.05s | 29.0 tok/s | 87 | 3,640 / 133 | 20 (20 from hints) | 44% | none | - |
| TechnoBaptist/Ternary-Bonsai-2-27B-mlx-2bit | no concerns detected | 58.33s | 34.8 tok/s | 17 | 16,594 / 118 | 18 (18 from hints) | 20% | none | - |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | concerns detected | 6.71s | 51.8 tok/s | 28 | 586 / 70 | 12 (8 from hints) | 55% | duplicate keywords | - |
| mlx-community/granite-4.0-3b-vision-4bit | concerns detected | 3.33s | 179 tok/s | 4.6 | 1,454 / 67 | 10 (5 from hints) | 100% | prompt hint repeated | - |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit | concerns detected | 3.49s | 196 tok/s | 4.0 | 2,096 / 83 | 15 (15 from hints) | 90% | prompt hint repeated | - |
| mlx-community/MiniCPM-V-4.7-1B-4bit | concerns detected | 4.14s | 253 tok/s | 3.6 | 927 / 114 | 25 (18 from hints) | 34% | duplicate keywords | - |
| mlx-community/Phi-3.5-vision-instruct-bf16 | concerns detected | 5.39s | 54.3 tok/s | 9.3 | 1,120 / 149 | 20 (15 from hints) | 27% | duplicate keywords | - |
| mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit | concerns detected | 22.62s | 67.8 tok/s | 26 | 12,683 / 85 | 16 (16 from hints) | 86% | prompt hint repeated | - |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | concerns detected | 3.59s | 124 tok/s | 5.6 | 1,419 / 74 | 18 (18 from hints) | 100% | prompt hint repeated | - |
| sahilchachra/LensVLM-9B-MXFP4 | concerns detected | 5.23s | 95.9 tok/s | 7.3 | 1,802 / 83 | 15 (15 from hints) | 100% | prompt hint repeated | - |
| mlx-community/FastVLM-0.5B-bf16 | major concerns: answer format | 4.04s | 258 tok/s | 1.8 | 325 / 22 | - | 100% of answer | labelled fields not detected; prompt hint repeated | - |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns: answer format | 3.67s | 105 tok/s | 6.7 | 2,186 / 29 | - | 83% of answer | control tokens visible; labelled fields not detected; prompt hint repeated | not yet checked (earlier check out of date) |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | major concerns: answer format | 54.01s | 23.6 tok/s | 25 | 4,367 / 1,000 | 15 (2 from hints) | - | control tokens visible; labelled fields not detected; cut off at token limit; role tokens visible | not yet checked |
| mlx-community/nanoLLaVA-1.5-4bit | major concerns: answer format | 2.93s | 203 tok/s | 1.5 | 321 / 148 | - | 19% | labelled fields not detected | - |
| mlx-community/InternVL3_5-1B-4bit | major concerns: generation | 3.10s | 329 tok/s | 2.1 | 2,107 / 200 | 57 (10 from hints) | 29% | repeated text; stopped early: repeating; duplicate keywords | not yet checked |

## Native reproduction checks

Recorded in native_repro_notes.yaml for this run's model revisions, builds
(mlx 0.32.5.dev20261106+9f806b360 @ 9f806b360; mlx-vlm 0.7.12 @ dcea78779;
transformers 5.22.1) and experiment inputs (image, prompt, generation
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
retained runs: 44 earlier retained runs from git
`HEAD:src/output/results.jsonl`, back to 2026-08-30; older runs were not read.
A run that did not attempt the model is skipped, and a report-only correction
of a run counts once. "Last N runs" and "N runs since" count consecutive runs
ending with this one; "first" is the earliest run read that showed it. The
table shows each model's longest-running observation; every observation's
dates are under *History by observation*.

| Model | Mechanical checks | Since baseline | History | Observed result | Evidence |
| --- | --- | --- | --- | --- | --- |
| mlx-community/InternVL3_5-1B-4bit | major concerns: generation | unchanged | duplicate keywords: 7 runs since 2026-10-16; +2 more, newest 4 runs since 2026-10-28 | Response repeats the same text; Generation was stopped early after sustained repeated output; Duplicate keywords: sky, sky view, skyline | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-internvl35-1b-4bit) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns: answer format | unchanged | labelled fields not detected: 35+ runs since 2026-09-12 or earlier; +2 more, newest 4 runs since 2026-10-28 | Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description, keywords; Repeats the prompt's hint text: description | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit) |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | major concerns: answer format | changed | labelled fields not detected: 3 runs since 2026-10-30; +3 more, newest back this run | Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description; Response appears cut off at the token limit; Conversation-role control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-muse-glimmer-30b-optiq-4bit) |

<details>
<summary>History by observation</summary>

| Model | Observation | Persistence |
| --- | --- | --- |
| mlx-community/InternVL3_5-1B-4bit | duplicate keywords | last 7 runs (since 2026-10-16), first 2026-10-04 |
| mlx-community/InternVL3_5-1B-4bit | repeated text | last 7 runs (since 2026-10-16), first 2026-10-04 |
| mlx-community/InternVL3_5-1B-4bit | stopped early: repeating | last 4 runs (since 2026-10-28), first 2026-10-04 |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | labelled fields not detected | last 35+ runs (since 2026-09-12 or earlier) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | prompt hint repeated | last 4 runs (since 2026-10-28) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | control tokens visible | last 35+ runs (since 2026-09-12 or earlier) |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | labelled fields not detected | last 3 runs (since 2026-10-30), first 2026-09-06 |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | role tokens visible | back this run, first 2026-09-06 |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | cut off at token limit | last 3 runs (since 2026-10-30), first 2026-09-06 |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | control tokens visible | back this run, first 2026-09-13 |

</details>

## Completions without detected concerns

28 completions without detected concerns; 10 more completed with prompt-compliance observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 6,656 x 9,483 pixels, 60,507,861 bytes
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: seed:* 0
- *Trust remote code:* true
- *check_models version:* 0.17.105
- *check_models revision:* b21c07f22d7b94d05eccfc5c09d17d830cf8f5ab
- *check_models source dirty:* false
- *mlx-vlm:* 0.7.12
- *mlx-vlm source revision:* dcea7877909c4cbf970457130a5345693f23669b
- *mlx:* 0.32.5.dev20261106+9f806b360
- *mlx source revision:* 9f806b360
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
