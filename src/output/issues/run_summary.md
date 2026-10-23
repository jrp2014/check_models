# mlx-vlm compatibility findings across 51 cached vision-language models

**For mlx-vlm maintainers:** no crashes need action; 5 other results need
reproducing with mlx-vlm alone before anything is reported (4 of 5 unchanged
since the baseline).

**Completed models:** 40 with no mechanical observations, 6 with
prompt-compliance observations only (labels, form or copied hint text), and 5
with observations mlx-vlm can produce (such as repetition, missing final
answers or visible control tokens); only those 5 are candidates for native
reproduction.

## Run summary

- *Run started:* 2026-10-23 15:47:27 BST
- *Run finished:* 2026-10-23 15:53:06 BST
- *Run duration:* 5m 38s
- *Time by phase:* generation 216s, model load 89s, prompt prep 2s, cleanup
  6s, outside the model loop 28s
- *Evaluation lane:* blind
- *Prompt hints:* no description or keyword hints
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Input image:* JPEG, 640 x 480 pixels (0.3 MP), 0.2 MB
- *Models attempted:* 51
- *Sampling settings:* checkpoint generation_config.json values for 22 of 51
  completed models where the command line left them unset; harness defaults
  elsewhere
- *Completed:* 51
- *Crashed:* 0
- *Indeterminate:* 0
- *Crashes requiring action:* 0
- *Other results requiring review:* 5
- *Reached token limit:* 5 (5 with incomplete output)
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

- *Baseline:* 361071fd:src/output/results.jsonl
- *Baseline run timestamp:* 2026-10-16 23:21:25 BST
- *Baseline check_models:* 0.17.71 @ 5bfc8f11d
- *Baseline mlx:* 0.32.5.dev20261016+b6642e604 @ b6642e604
- *Baseline mlx-vlm:* 0.7.8 @ b38cfa7b8
- *Baseline transformers:* 5.20.1
- *Baseline python:* 3.14.7
- *Baseline hardware:* Apple M5 Max, 40 GPU cores, 128.0 GB RAM
- *Earlier retained runs on this mlx version:* none recorded on
  0.32.5.dev20261023+afc03c511
- *Models compared:* 51
- *Identical generated text:* 27 of 51 completed in both
- *Generated text changed:* 24 models (listed below)
- *Text changed, by decoding:* greedy 10 of 31; sampled, same settings and
  seed 14 of 20
- *Greedy text changed, by checkpoint quantization:* quantized 10 of 26; not
  quantized 0 of 5
- *Generation tok/s ratio (now/baseline):* 0.997 over 50 models; lowest 0.92
  (vikhyatk/moondream2), highest 1.49
  (mlx-community/SmolVLM-256M-Instruct-4bit)
- *Prefill tok/s ratio (now/baseline):* 1.042 over 51 models; lowest 0.47
  (mlx-community/MiniCPM-V-4.6-4bit), highest 2.28
  (mlx-community/Kimi-VL-A3B-Thinking-2506-8bit)
- *Throughput noise band:* fixed ±15% fallback (insufficient history)

Model revisions changed since the baseline (their rows below reflect a
different snapshot, not only a different run):
`mlx-community/LFM2.5-VL-3B-OptiQ-4bit` 7886c0b4a → b6311011d

Run environment: current run detected GPU activity outside harness inference
at the checks around 34 of 51 models (up to 38%); baseline run on battery
power for 51 of 51 models; baseline run detected GPU activity outside harness
inference at the checks around 4 of 51 models (up to 39%).

| Model | Execution | Usability | Observation delta |
| --- | --- | --- | --- |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | completed | no concerns detected → concerns detected | +duplicate keywords |
| mlx-community/Step-3.7-Flash-oQ3e | completed | major concerns → no concerns detected | -answer emitted twice; -control tokens visible |
| mlx-community/X-Reasoner-7B-8bit | completed | major concerns | +cut off at token limit; -stopped early: repeating |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | completed | no concerns detected → major concerns | +labelled fields not detected; +cut off at token limit |

<details>
<summary>Generated text changed for 24 models</summary>

| Model | Decoding | Weights | Shared prefix (chars) | Length (chars) | Output tokens | Prompt token count |
| --- | --- | --- | --- | --- | --- | --- |
| mlx-community/MiniCPM-V-4.7-1B-4bit | greedy | 4-bit affine | 250 | 257 → 264 | 64 → 66 | same |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit | greedy | 4-bit affine | 95 | 340 → 322 | 74 → 79 | same |
| mlx-community/granite-4.0-3b-vision-4bit | greedy | 4-bit affine | 8 | 281 → 235 | 66 → 58 | same |
| mlx-community/gemma-4-e4b-it-4bit | sampled | 4-bit affine | 26 | 261 → 310 | 62 → 67 | same |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | greedy | 4-bit affine | 11 | 407 → 435 | 100 → 102 | same |
| nativ-community/Mage-VL-OptiQ-4bit | greedy | 4-bit affine | 396 | 427 → 430 | 89 → 89 | same |
| mlx-community/gemma-3n-E4B-it-4bit | greedy | 4-bit | 109 | 263 → 261 | 63 → 63 | same |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | sampled | 4-bit affine | 108 | 387 → 386 | 83 → 83 | same |
| mlx-community/Qwen3.5-35B-A3B-4bit | sampled | 4-bit affine | 22 | 465 → 400 | 102 → 88 | same |
| mlx-community/gemma-4-26b-a4b-it-4bit | sampled | 4-bit affine | 133 | 391 → 395 | 81 → 88 | same |
| mlx-community/MiniCPM-o-4_5-4bit | sampled | 4-bit affine | 269 | 470 → 444 | 103 → 99 | same |
| mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit | greedy | 4-bit affine | 245 | 394 → 360 | 87 → 81 | same |
| mlx-community/MiniCPM-V-4.6-4bit | sampled | 4-bit affine | 39 | 1,889 → 1,309 | 478 → 331 | same |
| mlx-community/MiniCPM-V-4.7-35B-A3B-4bit | sampled | 4-bit affine | 336 | 385 → 406 | 86 → 92 | same |
| mlx-community/gemma-4-12B-it-4bit | sampled | 4-bit affine | 80 | 323 → 397 | 77 → 88 | same |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | greedy | 8-bit mxfp8 | 66 | 276 → 301 | 67 → 72 | same |
| nativ-community/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-4bit | sampled | 4-bit affine | 360 | 455 → 460 | 105 → 101 | same |
| mlx-community/Qwen3.8-27B-nvfp4 | sampled | 4-bit nvfp4 | 255 | 580 → 531 | 113 → 109 | same |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | sampled | 4-bit affine | 79 | 1,744 → 1,959 | 478 → 540 | same |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit | sampled | 8-bit | 217 | 1,852 → 1,457 | 435 → 319 | same |
| mlx-community/Qwen2-VL-2B-mlx | sampled | not quantized | 345 | 4,778 → 4,777 | 1,000 → 1,000 | same |
| mlx-community/Step-3.7-Flash-oQ3e | greedy | 3-bit affine | 11 | 1,036 → 460 | 236 → 101 | same |
| mlx-community/X-Reasoner-7B-8bit | greedy | 8-bit | 77 | 1,166 → 4,953 | 275 → 1,000 | same |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | sampled | 4-bit affine | 6 | 410 → 163 | 937 → 1,000 | same |

</details>

Generation tok/s outside the expected band for 9 models (mlx changed since the
baseline: b6642e604..afc03c511):

| Model | Baseline tok/s | Now tok/s | Ratio | Expected band |
| --- | --- | --- | --- | --- |
| mlx-community/SmolVLM-256M-Instruct-4bit | 392.9 | 585.6 | 1.49 | 333.9-451.8 (fallback) |
| mlx-community/MiniCPM-V-4.7-1B-4bit | 261.1 | 308.9 | 1.18 | 221.9-300.3 (fallback) |
| mlx-community/nanoLLaVA-1.5-4bit | 224.1 | 308.4 | 1.38 | 190.5-257.7 (fallback) |
| mlx-community/North-Micro-Vision-Instruct-4bit | 164.0 | 223.3 | 1.36 | 139.4-188.6 (fallback) |
| mlx-community/gemma-3n-E4B-it-4bit | 72.3 | 91.2 | 1.26 | 61.5-83.2 (fallback) |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | 88.4 | 114.1 | 1.29 | 75.1-101.7 (fallback) |
| mlx-community/Qwen3.5-35B-A3B-4bit | 79.3 | 113.0 | 1.43 | 67.4-91.2 (fallback) |
| mlx-community/MiniCPM-V-4.6-4bit | 255.3 | 298.8 | 1.17 | 217.0-293.6 (fallback) |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | 109.2 | 138.2 | 1.26 | 92.8-125.6 (fallback) |

Largest prefill changes outside 0.85-1.15x (prefill seconds are prompt tokens
divided by prefill tok/s; one timed generation per sweep, so observations, not
demonstrated speed changes):

| Model | Baseline prefill s | Now prefill s | Prefill tok/s ratio |
| --- | --- | --- | --- |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit | 0.59 | 0.26 | 2.28 |
| mlx-community/MiniCPM-V-4.6-4bit | 0.11 | 0.24 | 0.47 |
| TechnoBaptist/Ternary-Bonsai-2-27B-mlx-2bit | 1.15 | 0.72 | 1.60 |
| mlx-community/GLM-4.6V-nvfp4 | 2.18 | 1.44 | 1.51 |
| mlx-community/X-Reasoner-7B-8bit | 0.34 | 0.23 | 1.43 |

<details>
<summary>mlx: 17 upstream commit(s) since the baseline (b6642e604..afc03c511)</summary>

- afc03c511 Fix GPU pad for outputs larger than 2^31 elements (#4744)
- 53c187057 [CUDA] Fix gather_qmm for experts past 2^32 weight elements
  (#4738)
- 06122f8bc Use data_size for gather/scatter large-offset checks (#4724)
- c9c8ed559 Fix data race in distributed ring on pending send/recv lists
  (#4737)
- 8e88e5056 Fix/padding jvp vjp (#4736)
- 007478f1a Widen affine qmv_wide row tiles to 8 (#4728)
- 59cf1adcd Fix zero-copy numpy arrays dtype mismatch (#4731)
- 81770e8d8 Fix normal loc scale shape (#4730)
- 7cfbc6a3a Fix `mx.take`'s negative indices (#4729)
- c6abf0dbb [Metal] Add qmv_mma/qmv_nax kernels for 6-16 rows with large N/K
  (#4712)
- 52a6c234a [Metal] Add fast route for dequantizing fp4 to half in fp_qmm_t
  (#4709)
- bff10220c [Metal] Add staged affine_qmm_t_nax kernels with split-k (#4682)
- f41bec734 Fix unaligned scalar reads and oversized contiguous buffer views
  (#4721)
- f93902c42 Make `broadcast_arrays` return a tuple (#4713)
- 9cb928211 Allow configuring the CPU JIT cache directory (#4714)
- ... and 2 more (see results.jsonl)

</details>

<details>
<summary>mlx-vlm: 29 upstream commit(s) since the baseline (b38cfa7b8..4319ef2c8)</summary>

- 4319ef2c feat: native Nemotron 3.5 Parse MLX port (#2562)
- 18f1528c Fix text-only input truncation (#2552)
- 03580bf1 fix: record converted dtype in exported configs (#2577)
- c3e709fc fix: keep converted metadata writable (#2298)
- abd3b2d7 Reduce CI turnaround with sharding and cheaper parity tests (#2567)
- 8826b112 Merge pull request #2560 from bxespinozar/add-oxr
- 7a7fe08f Merge pull request #2167 from
  TheAngryPit/codex/realtime-stt-audio-queue
- a0b9352e fix: preserve OXR image processor settings
- ce1e3343 Add SoMark OXR-1.0 document parsing model
- 82d720c4 Fix audio cancellation errors and realtime transcription timeouts
- 8c061ef5 Fix fused NVFP4 rowwise scales during expert offload repack (#2566)
- 5a7fec15 Preserve NVFP4 scales and batch offloaded MoE execution (#2538)
- a549bc77 Add native Flint BF16 support and bump to 0.7.10 (#2565)
- f1a9b7ba Merge pull request #2542 from lucasnewman/muse-spark-open
- 8db4e488 Merge branch 'main' into muse-spark-open
- ... and 14 more (see results.jsonl)

</details>

<details>
<summary>mlx-vlm commits touching the affected models' architectures and their imports</summary>

- `mlx-community/MiniCPM-V-4.7-1B-4bit` (`minicpmv4_7`; imports `base.py`,
  `minicpmv4_6`, `qwen3_5_moe`): b5672947 Fix rotating cache sinks and
  absolute position handling (#2548)
- `mlx-community/LFM2.5-VL-3B-OptiQ-4bit` (`lfm2_vl`; imports `attention.py`,
  `base.py`, `interpolate.py`, `lfm2`, `mlp.py`): b5672947 Fix rotating cache
  sinks and absolute position handling (#2548)
- `mlx-community/granite-4.0-3b-vision-4bit` (`granite4_vision`; imports
  `base.py`, `rope_utils.py`): b5672947 Fix rotating cache sinks and absolute
  position handling (#2548)
- `mlx-community/gemma-4-e4b-it-4bit` (`gemma4`; imports `base.py`,
  `cache.py`, `qwen3_5`, `rope_utils.py`, `switch_layers.py`): 5a7fec15
  Preserve NVFP4 scales and batch offloaded MoE execution (#2538); b5672947
  Fix rotating cache sinks and absolute position handling (#2548); b8bf04c6
  Add MOSS-Image-SFT image generation model (#2541)
- `mlx-community/Ministral-3-3B-Instruct-2512-4bit` (`mistral3`; imports
  `base.py`, `cache.py`, `mistral4`, `pixtral`, `rope_utils.py`,
  `switch_layers.py`): 5a7fec15 Preserve NVFP4 scales and batch offloaded MoE
  execution (#2538); b5672947 Fix rotating cache sinks and absolute position
  handling (#2548)
- `nativ-community/Mage-VL-OptiQ-4bit` (`mage_vl`; imports `base.py`, `qwen3`,
  `qwen3_vl`, `rope_utils.py`): b5672947 Fix rotating cache sinks and absolute
  position handling (#2548)
- `mlx-community/gemma-3n-E4B-it-4bit` (`gemma3n`; imports `base.py`,
  `cache.py`, `kernels.py`, `rope_utils.py`): b5672947 Fix rotating cache
  sinks and absolute position handling (#2548)
- `mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit` (`qwen3_vl_moe`; imports
  `base.py`, `cache.py`, `mlp.py`, `qwen3_vl`, `rope_utils.py`,
  `switch_layers.py`): 5a7fec15 Preserve NVFP4 scales and batch offloaded MoE
  execution (#2538); b5672947 Fix rotating cache sinks and absolute position
  handling (#2548)
- `mlx-community/Qwen3.5-35B-A3B-4bit` (`qwen3_5_moe`; imports `base.py`,
  `qwen3_5`, `qwen3_vl`, `switch_layers.py`): 5a7fec15 Preserve NVFP4 scales
  and batch offloaded MoE execution (#2538); b5672947 Fix rotating cache sinks
  and absolute position handling (#2548); b8bf04c6 Add MOSS-Image-SFT image
  generation model (#2541)
- `mlx-community/gemma-4-26b-a4b-it-4bit` (`gemma4`; imports `base.py`,
  `cache.py`, `qwen3_5`, `rope_utils.py`, `switch_layers.py`): 5a7fec15
  Preserve NVFP4 scales and batch offloaded MoE execution (#2538); b5672947
  Fix rotating cache sinks and absolute position handling (#2548); b8bf04c6
  Add MOSS-Image-SFT image generation model (#2541)
- `mlx-community/MiniCPM-o-4_5-4bit` (`minicpmo`; imports `base.py`,
  `cache.py`, `llama`, `qwen3_vl`): b5672947 Fix rotating cache sinks and
  absolute position handling (#2548)
- `mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit` (`qwen3_omni_moe`; imports
  `base.py`, `cache.py`, `mlp.py`, `qwen3_vl`, `rope_utils.py`,
  `switch_layers.py`): 5a7fec15 Preserve NVFP4 scales and batch offloaded MoE
  execution (#2538); b5672947 Fix rotating cache sinks and absolute position
  handling (#2548)
- `mlx-community/MiniCPM-V-4.6-4bit` (`minicpmv4_6`; imports `base.py`,
  `qwen3_5`): b5672947 Fix rotating cache sinks and absolute position handling
  (#2548); b8bf04c6 Add MOSS-Image-SFT image generation model (#2541)
- `mlx-community/MiniCPM-V-4.7-35B-A3B-4bit` (`minicpmv4_7`; imports
  `base.py`, `minicpmv4_6`, `qwen3_5_moe`): b5672947 Fix rotating cache sinks
  and absolute position handling (#2548)
- `mlx-community/gemma-4-12B-it-4bit` (`gemma4_unified`; imports `base.py`,
  `gemma4`): b5672947 Fix rotating cache sinks and absolute position handling
  (#2548)
- `mlx-community/diffusiongemma-26B-A4B-it-mxfp8` (`diffusion_gemma`; imports
  `base.py`, `cache.py`, `diffusion_visualizer.py`, `gemma4`, `qwen3_vl`,
  `rope_utils.py`, `switch_layers.py`): 5a7fec15 Preserve NVFP4 scales and
  batch offloaded MoE execution (#2538); b5672947 Fix rotating cache sinks and
  absolute position handling (#2548)
- `nativ-community/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-4bit`
  (`nemotron_h_nano_omni`; imports `base.py`, `interpolate.py`, `nemotron_h`):
  b5672947 Fix rotating cache sinks and absolute position handling (#2548)
- `mlx-community/Qwen3.8-27B-nvfp4` (`qwen3_5`; imports `activations.py`,
  `base.py`, `cache.py`, `quantized_verifier.py`, `qwen3_vl`,
  `rope_utils.py`): b5672947 Fix rotating cache sinks and absolute position
  handling (#2548); b8bf04c6 Add MOSS-Image-SFT image generation model (#2541)
- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit` (`ernie4_5_moe_vl`;
  imports `base.py`, `cache.py`, `rope_utils.py`, `switch_layers.py`):
  5a7fec15 Preserve NVFP4 scales and batch offloaded MoE execution (#2538);
  b5672947 Fix rotating cache sinks and absolute position handling (#2548)
- `mlx-community/Kimi-VL-A3B-Thinking-2506-8bit` (`kimi_vl`; imports
  `base.py`, `kernels.py`, `mlp.py`, `rope_utils.py`, `switch_layers.py`):
  5a7fec15 Preserve NVFP4 scales and batch offloaded MoE execution (#2538);
  b5672947 Fix rotating cache sinks and absolute position handling (#2548)
- `mlx-community/Qwen2-VL-2B-mlx` (`qwen2_vl`; imports `base.py`, `cache.py`,
  `mlp.py`, `qwen3_vl`, `rope_utils.py`): b5672947 Fix rotating cache sinks
  and absolute position handling (#2548)
- `mlx-community/Step-3.7-Flash-oQ3e` (`step3p7`; imports `base.py`,
  `cache.py`, `qwen3_vl`, `rope_utils.py`, `switch_layers.py`): 5a7fec15
  Preserve NVFP4 scales and batch offloaded MoE execution (#2538); b5672947
  Fix rotating cache sinks and absolute position handling (#2548)
- `mlx-community/X-Reasoner-7B-8bit` (`qwen2_5_vl`; imports `base.py`,
  `cache.py`, `mlp.py`, `qwen3_vl`, `rope_utils.py`): b5672947 Fix rotating
  cache sinks and absolute position handling (#2548)
- `mlx-community/Muse-Glimmer-30B-OptiQ-4bit` (`muse_glimmer`; imports
  `activations.py`, `base.py`, `cache.py`, `rope_utils.py`): b5672947 Fix
  rotating cache sinks and absolute position handling (#2548)

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
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | no concerns detected | 0.57s | 523 tok/s | 1.2 | 350 / 47 | 6 | none | - |
| mlx-community/AREX-2-4bit | no concerns detected | 4.49s | 31.0 tok/s | 17 | 421 / 80 | 8 | none | - |
| mlx-community/aya-vision-8b-4bit | no concerns detected | 4.15s | 94.8 tok/s | 6.5 | 2,909 / 82 | 10 | none | - |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | no concerns detected | 5.23s | 51.6 tok/s | 28 | 386 / 72 | 15 | none | - |
| mlx-community/gemma-3-27b-it-qat-4bit | no concerns detected | 6.18s | 29.0 tok/s | 17 | 379 / 89 | 18 | none | - |
| mlx-community/gemma-3n-E4B-it-4bit | no concerns detected | 2.73s | 91.2 tok/s | 7.2 | 378 / 63 | 15 | none | - |
| mlx-community/gemma-4-12B-it-4bit | no concerns detected | 3.60s | 60.1 tok/s | 7.5 | 390 / 88 | 18 | none | - |
| mlx-community/gemma-4-26b-a4b-it-4bit | no concerns detected | 3.61s | 117 tok/s | 16 | 390 / 88 | 15 | none | - |
| mlx-community/gemma-4-e4b-it-4bit | no concerns detected | 2.44s | 120 tok/s | 5.9 | 386 / 67 | 14 | none | - |
| mlx-community/GLM-4.6V-Flash-4bit | no concerns detected | 2.16s | 86.4 tok/s | 8.0 | 499 / 49 | 5 | none | - |
| mlx-community/GLM-4.6V-nvfp4 | no concerns detected | 9.67s | 51.8 tok/s | 63 | 499 / 85 | 17 | none | - |
| mlx-community/granite-4.0-3b-vision-4bit | no concerns detected | 1.33s | 171 tok/s | 4.7 | 734 / 58 | 10 | none | - |
| mlx-community/granite-vision-3.2-2b-nvfp4 | no concerns detected | 2.62s | 165 tok/s | 3.9 | 3,095 / 119 | 7 | none | - |
| mlx-community/InternVL3-14B-4bit | no concerns detected | 4.79s | 56.2 tok/s | 10 | 3,435 / 90 | 18 | none | - |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit | no concerns detected | 6.98s | 73.0 tok/s | 20 | 489 / 319 | 12 | none | - |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit | no concerns detected | 1.53s | 217 tok/s | 3.6 | 344 / 79 | 10 | none | - |
| mlx-community/MiniCPM-o-4_5-4bit | no concerns detected | 2.26s | 102 tok/s | 7.0 | 177 / 99 | 11 | none | - |
| mlx-community/MiniCPM-V-4.6-4bit | no concerns detected | 2.47s | 299 tok/s | 3.1 | 330 / 331 | 10 | none | - |
| mlx-community/MiniCPM-V-4.7-1B-4bit | no concerns detected | 1.36s | 309 tok/s | 3.6 | 332 / 66 | 13 | none | - |
| mlx-community/MiniCPM-V-4.7-35B-A3B-4bit | no concerns detected | 4.13s | 87.3 tok/s | 24 | 332 / 92 | 18 | none | - |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | no concerns detected | 3.28s | 67.2 tok/s | 9.8 | 1,070 / 72 | 10 | none | - |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | no concerns detected | 1.83s | 195 tok/s | 4.5 | 1,069 / 102 | 12 | none | - |
| mlx-community/Molmo2-8B-4bit | no concerns detected | 2.97s | 71.4 tok/s | 8.3 | 873 / 95 | 10 | none | - |
| mlx-community/MolmoPoint-8B-4bit | no concerns detected | 4.61s | 31.0 tok/s | 12 | 860 / 70 | 10 | none | - |
| mlx-community/North-Micro-Vision-Instruct-4bit | no concerns detected | 1.72s | 223 tok/s | 3.5 | 418 / 78 | 12 | none | - |
| mlx-community/Phi-3.5-vision-instruct-bf16 | no concerns detected | 2.87s | 57.1 tok/s | 9.1 | 883 / 94 | 19 | none | - |
| mlx-community/pixtral-12b-8bit | no concerns detected | 4.85s | 38.9 tok/s | 15 | 1,335 / 89 | 17 | none | - |
| mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit | no concerns detected | 3.70s | 90.4 tok/s | 23 | 409 / 81 | 18 | none | - |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | no concerns detected | 3.11s | 114 tok/s | 19 | 409 / 83 | 10 | none | - |
| mlx-community/Qwen3-VL-8B-Instruct-4bit | no concerns detected | 1.90s | 100.0 tok/s | 6.8 | 409 / 60 | 10 | none | - |
| mlx-community/Qwen3.5-35B-A3B-4bit | no concerns detected | 3.66s | 113 tok/s | 21 | 421 / 88 | 10 | none | - |
| mlx-community/Qwen3.8-27B-nvfp4 | no concerns detected | 6.25s | 31.4 tok/s | 17 | 421 / 109 | 13 | none | - |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | no concerns detected | 1.41s | 122 tok/s | 5.6 | 1,204 / 31 | 4 | none | - |
| mlx-community/Step-3.7-Flash-oQ3e | no concerns detected | 23.45s | 48.6 tok/s | 87 | 286 / 101 | 18 | none | - |
| nativ-community/Mage-VL-OptiQ-4bit | no concerns detected | 1.73s | 143 tok/s | 4.9 | 420 / 89 | 10 | none | - |
| nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit | no concerns detected | 5.08s | 34.8 tok/s | 17 | 538 / 87 | 18 | none | - |
| nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | no concerns detected | 3.18s | 146 tok/s | 21 | 421 / 65 | 10 | none | - |
| nativ-community/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-4bit | no concerns detected | 6.45s | 65.7 tok/s | 46 | 420 / 101 | 15 | none | - |
| sahilchachra/LensVLM-9B-MXFP4 | no concerns detected | 2.79s | 98.4 tok/s | 6.8 | 421 / 111 | 10 | none | - |
| TechnoBaptist/Ternary-Bonsai-2-27B-mlx-2bit | no concerns detected | 3.80s | 43.5 tok/s | 11 | 421 / 94 | 14 | none | - |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | concerns detected | 4.89s | 138 tok/s | 18 | 545 / 540 | 16 | duplicate keywords | - |
| mlx-community/FastVLM-0.5B-bf16 | major concerns: answer format | 0.91s | 378 tok/s | 2.2 | 120 / 32 | - | labelled fields not detected | - |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns: answer format | 2.75s | insufficient sample | 7.0 | 3,512 / 14 | - | control tokens visible; labelled fields not detected | candidate |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | major concerns: answer format | 48.34s | 22.6 tok/s | 25 | 545 / 1,000 | - | labelled fields not detected; cut off at token limit | - |
| mlx-community/nanoLLaVA-1.5-4bit | major concerns: answer format | 1.14s | 308 tok/s | 1.6 | 116 / 97 | - | labelled fields not detected | - |
| mlx-community/SmolVLM-256M-Instruct-4bit | major concerns: answer format | 0.70s | insufficient sample | 1.1 | 983 / 8 | - | labelled fields not detected | - |
| vikhyatk/moondream2 | major concerns: answer format | 1.35s | 162 tok/s | 4.5 | 833 / 34 | - | labelled fields not detected | - |
| mlx-community/InternVL3_5-1B-4bit | major concerns: generation | 3.80s | 355 tok/s | 2.3 | 3,438 / 1,000 | 144 | repeated text; cut off at token limit; duplicate keywords | candidate |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | major concerns: generation | 56.65s | 18.7 tok/s | 15 | 107 / 1,000 | 141 | repeated text; cut off at token limit; duplicate keywords | candidate |
| mlx-community/Qwen2-VL-2B-mlx | major concerns: generation | 8.56s | 134 tok/s | 5.4 | 511 / 1,000 | - | repeated text; labelled fields not detected; cut off at token limit | candidate |
| mlx-community/X-Reasoner-7B-8bit | major concerns: generation | 17.65s | 62.1 tok/s | 10 | 511 / 1,000 | 189 | repeated text; cut off at token limit; duplicate keywords | candidate |

## Observation clusters

Observation signatures shared by two or more results requiring review.

| Observed result | Models |
| --- | --- |
| Response repeats the same text; Response appears cut off at the token limit; Repeated keyword entries | [mlx-community/InternVL3_5-1B-4bit](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-internvl35-1b-4bit), [mlx-community/X-Reasoner-7B-8bit](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-x-reasoner-7b-8bit), [mlx-community/Llama-3.2-11B-Vision-Instruct-8bit](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llama-32-11b-vision-instruct-8bit) |

## Completed attempts requiring review

*History* dates each observation (a crash counts as one) over this model's
retained runs: 39 earlier retained runs from git
`HEAD:src/output/results.jsonl`, back to 2026-08-30; older runs were not read.
A run that did not attempt the model is skipped, and a report-only correction
of a run counts once. "Last N runs" and "N runs since" count consecutive runs
ending with this one; "first" is the earliest run read that showed it. The
table shows each model's longest-running observation; every observation's
dates are under *History by observation*.

| Model | Mechanical checks | Since baseline | History | Observed result | Evidence |
| --- | --- | --- | --- | --- | --- |
| mlx-community/InternVL3_5-1B-4bit | major concerns: generation | unchanged | all 3: 2 runs since 2026-10-16 | Response repeats the same text; Response appears cut off at the token limit; Duplicate keywords: british english, british english examples of metadata, british english examples of details, british english examples of context, british english examples of information, british english examples of data, british english examples of content, british english examples of examples | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-internvl35-1b-4bit) |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | major concerns: generation | unchanged | all 3: 2 runs since 2026-10-16 | Response repeats the same text; Response appears cut off at the token limit; Duplicate keywords: feline household domesticated animals, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort, feline household domesticated relaxation, feline household domesticated comfort and relaxation | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llama-32-11b-vision-instruct-8bit) |
| mlx-community/Qwen2-VL-2B-mlx | major concerns: generation | unchanged | repeated text: 7 runs since 2026-10-10; +2 more, newest 5 runs since 2026-10-11 | Response repeats the same text; Required labelled fields not detected: keywords; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-qwen2-vl-2b-mlx) |
| mlx-community/X-Reasoner-7B-8bit | major concerns: generation | changed | duplicate keywords: 2 runs since 2026-10-16; +2 more, newest back this run | Response repeats the same text; Response appears cut off at the token limit; Duplicate keywords: feline behavior study, feline comfort study, feline rest study, feline lifestyle study, feline domestic life study, feline behavior analysis, feline comfort analysis, feline rest analysis, feline lifestyle analysis, feline domestic life analysis, feline behavior observation, feline comfort observation, feline rest observation, feline lifestyle observation, feline domestic life observation, feline behavior research, feline comfort research, feline rest research, feline lifestyle research, feline domestic life research | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-x-reasoner-7b-8bit) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns: answer format | unchanged | all 2: 30+ runs since 2026-09-12 or earlier | Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description, keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit) |

<details>
<summary>History by observation</summary>

| Model | Observation | Persistence |
| --- | --- | --- |
| mlx-community/InternVL3_5-1B-4bit | duplicate keywords | last 2 runs (since 2026-10-16), first 2026-10-04 |
| mlx-community/InternVL3_5-1B-4bit | repeated text | last 2 runs (since 2026-10-16), first 2026-10-04 |
| mlx-community/InternVL3_5-1B-4bit | cut off at token limit | last 2 runs (since 2026-10-16) |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | duplicate keywords | last 2 runs (since 2026-10-16), first 2026-09-25 |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | repeated text | last 2 runs (since 2026-10-16), first 2026-09-18 |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | cut off at token limit | last 2 runs (since 2026-10-16), first 2026-09-18 |
| mlx-community/Qwen2-VL-2B-mlx | labelled fields not detected | last 5 runs (since 2026-10-11) |
| mlx-community/Qwen2-VL-2B-mlx | repeated text | last 7 runs (since 2026-10-10) |
| mlx-community/Qwen2-VL-2B-mlx | cut off at token limit | last 5 runs (since 2026-10-11), first 2026-10-10 |
| mlx-community/X-Reasoner-7B-8bit | duplicate keywords | last 2 runs (since 2026-10-16), first 2026-09-06 |
| mlx-community/X-Reasoner-7B-8bit | repeated text | last 2 runs (since 2026-10-16), first 2026-08-30 |
| mlx-community/X-Reasoner-7B-8bit | cut off at token limit | back this run, first 2026-08-30 |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | labelled fields not detected | last 30+ runs (since 2026-09-12 or earlier) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | control tokens visible | last 30+ runs (since 2026-09-12 or earlier) |

</details>

## Completions without detected concerns

40 completions without detected concerns; 6 more completed with prompt-compliance observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 640 x 480 pixels, 173,131 bytes
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: seed:* 0
- *Trust remote code:* true
- *check_models version:* 0.17.89
- *check_models revision:* 361071fde312cce00393d7fe264fa707026032f5
- *check_models source dirty:* false
- *mlx-vlm:* 0.7.10
- *mlx-vlm source revision:* 4319ef2c837b7226b758c3dd69edda51b7079785
- *mlx:* 0.32.5.dev20261023+afc03c511
- *mlx source revision:* afc03c511
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
