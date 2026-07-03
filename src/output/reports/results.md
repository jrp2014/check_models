# Model Performance Results

Generated on: 2026-07-04 00:01:55 BST

## Run Contract

- Mode: triage
- Semantic rankings: ungrounded (ungrounded)
- Primary selection tasks: brief captions; structured title/description/keywords when descriptive image metadata is available

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ 4 (top owners: model-config=2, mlx=1,
  mlx-vlm=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=5, mechanically clean
  outputs=44/57.
- _Useful now:_ 30 model(s) shortlisted for caption review.
- _Review watchlist:_ 21 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Quality signal frequency:_ cutoff=6, harness=5, reasoning_leak=4,
  text_sanity=3, generation_loop=3, lang_mixing=2.

### Runtime

- _Runtime pattern:_ post-prefill decode dominates measured phase time (49%;
  19/61 measured model(s)).
- _Phase totals:_ model load=126.55s, local prompt prep=0.20s, upstream
  prefill / first-token=54.71s, post-prefill decode=180.18s, cleanup=6.92s.
- _Generation total:_ 234.90s across 57 model(s); upstream prefill /
  first-token split available for 57/57 model(s).
- _What this likely means:_ Most measured runtime is spent generating after
  the first token is available.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=49, exception=4, max_tokens=8.
- _Validation overhead:_ 0.10s total (avg 0.00s across 61 model(s)).
- _Upstream prefill / first-token latency:_ Avg 0.96s | Min 0.02s | Max 4.75s
  across 57 model(s).

## 🏆 Performance Highlights

- **Fastest:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (562.4 tps)
- **💾 Most efficient:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (1.0 GB)
- **⚡ Fastest load:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (0.34s)
- **📊 Average TPS:** 94.3 across 57 models

## 📈 Resource Usage

- **Input image size:** 0.31 MP
- **Average peak delta from post-load:** 3.13 GB
- **Peak memory delta / MP:** 10420 MB/MP
- **Average peak memory:** 18.4 GB
- **Memory efficiency:** 44 tokens/GB

## ⚠️ Quality Issues

- **❌ Failed Models (4):**
  - `mlx-community/LFM2.5-VL-1.6B-bf16` (`Weight Mismatch`)
  - `mlx-community/MolmoPoint-8B-fp16` (`Model Error`)
  - `mlx-community/diffusiongemma-26B-A4B-it-8bit` (`Processor Error`)
  - `mlx-community/diffusiongemma-26B-A4B-it-mxfp8` (`Processor Error`)
- **🔄 Repetitive Output (1):**
  - `mlx-community/gemma-3n-E2B-4bit` (token: `phrase: "have this image. have..."`)
- **📝 Formatting Issues (1):**
  - `mlx-community/Idefics3-8B-Llama3-bf16`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 94.3 | Min: 4.7 | Max: 562
- **Peak Memory**: Avg: 18 | Min: 1.0 | Max: 71
- **Total Time**: Avg: 6.20s | Min: 0.41s | Max: 46.88s
- **Generation Time**: Avg: 4.12s | Min: 0.06s | Max: 43.63s
- **Model Load Time**: Avg: 2.07s | Min: 0.34s | Max: 9.56s

## Caption Selection

Triage mode suppresses cataloging and keyword scores. Brief-caption
recommendations are ungrounded unless descriptive image metadata is present;
use `model_selection.md` for the caption shortlist and `model_gallery.md` for
full output evidence.

## 🚨 Failures by Package (Actionable)

| Package        |   Failures | Error Types     | Affected Models                                                                                 |
|----------------|------------|-----------------|-------------------------------------------------------------------------------------------------|
| `model-config` |          2 | Processor Error | `mlx-community/diffusiongemma-26B-A4B-it-8bit`, `mlx-community/diffusiongemma-26B-A4B-it-mxfp8` |
| `mlx`          |          1 | Weight Mismatch | `mlx-community/LFM2.5-VL-1.6B-bf16`                                                             |
| `mlx-vlm`      |          1 | Model Error     | `mlx-community/MolmoPoint-8B-fp16`                                                              |

### Actionable Items by Package

#### model-config

- mlx-community/diffusiongemma-26B-A4B-it-8bit (Processor Error)
  - Error: `Model preflight failed for mlx-community/diffusiongemma-26B-A4B-it-8bit: Loaded processor has no image_processor; exp...`
  - Type: `ValueError`
- mlx-community/diffusiongemma-26B-A4B-it-mxfp8 (Processor Error)
  - Error: `Model preflight failed for mlx-community/diffusiongemma-26B-A4B-it-mxfp8: Loaded processor has no image_processor; ex...`
  - Type: `ValueError`

#### mlx

- mlx-community/LFM2.5-VL-1.6B-bf16 (Weight Mismatch)
  - Error: `Model loading failed: Missing 2 parameters: <br>multi_modal_projector.layer_norm.bias,<br>multi_modal_projector.layer_norm....`
  - Type: `ValueError`

#### mlx-vlm

- mlx-community/MolmoPoint-8B-fp16 (Model Error)
  - Error: `Model loading failed: property 'eos_token_id' of 'ModelConfig' object has no setter`
  - Type: `ValueError`

_Prompt used:_

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Describe this image briefly.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 372.31s

_Detailed machine-readable metrics remain in `results.tsv` and
`results.jsonl`; this Markdown table keeps the high-signal columns for human
review._

<!-- markdownlint-disable MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Gen TPS |   Peak (GB) | Finish Reason   |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                     |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|----------:|------------:|:----------------|-----------------:|-----------:|------------:|:-----------------------------------|----------------:|
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |                   |                       |                |           |             |                 |                  |      0.22s |       0.22s |                                    |             mlx |
| `mlx-community/MolmoPoint-8B-fp16`                      |                   |                       |                |           |             |                 |                  |      0.23s |       0.24s |                                    |         mlx-vlm |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit`          |                   |                       |                |           |             |                 |                  |      3.99s |       5.42s |                                    |    model-config |
| `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`         |                   |                       |                |           |             |                 |                  |      4.02s |       5.32s |                                    |    model-config |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      |                80 |                    10 |             90 |       562 |           1 | stop            |            0.06s |      0.34s |       0.41s |                                    |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               269 |                    10 |            279 |       327 |           3 | stop            |            0.12s |      0.54s |       0.66s |                                    |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |                26 |                    15 |             41 |       330 |         2.2 | stop            |            0.16s |      0.62s |       0.78s |                                    |                 |
| `mlx-community/MiniCPM-V-4.6-8bit`                      |               228 |                    22 |            250 |       278 |           3 | stop            |            0.21s |      0.98s |       1.21s |                                    |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |               417 |                    52 |            469 |       326 |         2.5 | stop            |            0.33s |      0.49s |       0.83s |                                    |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |                22 |                    81 |            103 |       296 |         1.8 | stop            |            0.36s |      0.47s |       0.83s |                                    |                 |
| `qnguyen3/nanoLLaVA`                                    |                22 |                    35 |             57 |       112 |           4 | stop            |            0.44s |      0.57s |       1.02s |                                    |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,196 |                    13 |          1,209 |       131 |         5.5 | stop            |            0.52s |      0.73s |       1.26s |                                    |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             1,196 |                    13 |          1,209 |       129 |         5.5 | stop            |            0.55s |      0.61s |       1.17s |                                    |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |               770 |                    19 |            789 |      62.2 |         9.2 | stop            |            0.58s |      0.95s |       1.53s |                                    |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               286 |                    24 |            310 |       110 |          17 | stop            |            0.63s |      2.40s |       3.06s |                                    |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |               949 |                    99 |          1,048 |       204 |         4.4 | stop            |            0.75s |      0.94s |       1.70s |                                    |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |               408 |                    42 |            450 |      93.8 |         7.7 | stop            |            0.76s |      1.32s |       2.09s |                                    |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |               315 |                    96 |            411 |       137 |         5.2 | stop            |            0.81s |      0.67s |       1.50s |                                    |                 |
| `mlx-community/Qwen3-VL-2B-Instruct-bf16`               |               315 |                    96 |            411 |       134 |         5.3 | stop            |            0.83s |      0.78s |       1.63s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |               319 |                    63 |            382 |       119 |          21 | stop            |            0.91s |      2.47s |       3.39s |                                    |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |               319 |                    71 |            390 |      98.6 |           7 | stop            |            1.00s |      1.40s |       2.42s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |               319 |                    58 |            377 |       101 |          30 | stop            |            1.01s |      3.07s |       4.09s |                                    |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |               417 |                    65 |            482 |      65.1 |          10 | stop            |            1.30s |      1.12s |       2.42s |                                    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |               950 |                    52 |          1,002 |      67.5 |          10 | stop            |            1.40s |      1.38s |       2.79s |                                    |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,103 |                     3 |          4,106 |      73.3 |         4.6 | stop            |            1.43s |      1.20s |       2.64s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |               408 |                    69 |            477 |      64.2 |          10 | stop            |            1.44s |      1.39s |       2.84s |                                    |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |               317 |                   196 |            513 |       134 |         5.3 | stop            |            1.58s |      0.73s |       2.33s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |               950 |                    68 |          1,018 |      70.8 |         9.7 | stop            |            1.59s |      1.31s |       2.90s |                                    |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               286 |                    25 |            311 |      27.7 |          19 | stop            |            1.69s |      2.67s |       4.37s |                                    |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,327 |                    23 |          2,350 |      33.7 |          19 | stop            |            1.76s |      1.91s |       3.68s | formatting                         |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |                97 |                   187 |            284 |       130 |         5.5 | stop            |            1.81s |      0.57s |       2.38s |                                    |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |                16 |                    25 |             41 |      22.8 |          15 | stop            |            2.46s |      1.49s |       3.96s |                                    |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               266 |                   200 |            466 |      91.9 |           6 | length          |            2.46s |      1.67s |       4.15s | repetitive(phrase: "have th...     |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             3,341 |                    48 |          3,389 |      34.7 |          18 | stop            |            2.70s |      1.58s |       4.28s |                                    |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |               319 |                    71 |            390 |      33.6 |          19 | stop            |            2.73s |      2.19s |       4.94s |                                    |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |               319 |                    40 |            359 |      19.5 |          30 | stop            |            2.76s |      3.23s |       6.02s |                                    |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             3,341 |                    15 |          3,356 |      33.1 |          19 | stop            |            2.89s |      1.60s |       4.50s |                                    |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |               399 |                   200 |            599 |      79.3 |          16 | length          |            2.90s |      1.92s |       4.83s | reasoning-leak, token-cap          |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               274 |                   124 |            398 |      47.9 |          17 | stop            |            2.93s |      2.38s |       5.32s |                                    |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |               399 |                   200 |            599 |      74.5 |          20 | length          |            3.10s |      2.18s |       5.28s | reasoning-leak, cutoff             |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |               319 |                    49 |            368 |        19 |          30 | stop            |            3.29s |      3.02s |       6.34s |                                    |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |               417 |                    90 |            507 |      33.1 |          19 | stop            |            3.38s |      2.09s |       5.48s | ⚠️harness(encoding)                |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |               770 |                   200 |            970 |      57.9 |         9.2 | length          |            3.71s |      0.93s |       4.65s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             2,356 |                    55 |          2,411 |      61.5 |         9.7 | stop            |            3.76s |      0.89s |       4.65s |                                    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,031 |                   101 |          1,132 |      32.5 |          11 | stop            |            4.05s |      1.59s |       5.65s |                                    |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               275 |                    91 |            366 |      26.5 |          19 | stop            |            4.23s |      2.33s |       6.58s |                                    |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |               408 |                    71 |            479 |      51.8 |          63 | stop            |            4.35s |      6.18s |      10.55s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |               319 |                    55 |            374 |      66.9 |          71 | stop            |            4.53s |      8.26s |      12.80s |                                    |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             1,330 |                   174 |          1,504 |      44.5 |          14 | stop            |            4.84s |      1.62s |       6.47s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |               439 |                   200 |            639 |      65.4 |          60 | length          |            5.08s |      5.37s |      10.46s | generation_loop(degeneration), ... |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |               745 |                    55 |            800 |      52.2 |          20 | stop            |            5.36s |      1.30s |       6.66s |                                    |                 |
| `mlx-community/Ornith-1.0-35B-bf16`                     |               319 |                    41 |            360 |        63 |          71 | stop            |            5.42s |      9.56s |      15.01s |                                    |                 |
| `mlx-community/pixtral-12b-8bit`                        |             1,239 |                   200 |          1,439 |        40 |          15 | length          |            5.85s |      1.72s |       7.57s | cutoff                             |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |               745 |                    55 |            800 |      30.4 |          27 | stop            |            6.50s |      1.76s |       8.28s |                                    |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,031 |                   119 |          1,150 |      19.5 |          10 | stop            |            6.51s |      1.46s |       7.97s |                                    |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               275 |                    70 |            345 |      12.2 |          33 | stop            |            6.65s |      3.51s |      10.18s |                                    |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               274 |                    36 |            310 |      7.65 |          64 | stop            |            7.17s |      6.11s |      13.29s |                                    |                 |
| `mlx-community/pixtral-12b-bf16`                        |             1,239 |                   200 |          1,439 |      20.2 |          27 | length          |           10.78s |      2.56s |      13.34s | cutoff                             |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |                17 |                    77 |             94 |      5.17 |          25 | stop            |           16.60s |      2.21s |      18.82s |                                    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,031 |                   159 |          1,190 |      5.47 |          26 | stop            |           30.25s |      2.51s |      32.76s |                                    |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |               399 |                   200 |            599 |       4.7 |          39 | length          |           43.63s |      3.24s |      46.88s | reasoning-leak, cutoff             |                 |

<!-- markdownlint-enable MD033 MD034 MD037 MD049 -->

_Companion artifacts:_

- _Model-selection shortlist:_ [model_selection.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_selection.md)
- _Standalone output gallery:_ [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md)
- _Automated review digest:_ [review.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/review.md)
- _Canonical run log:_ [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log)

---

## System/Hardware Information

- _OS:_ Darwin 25.5.0
- _macOS Version:_ 26.5.2
- _SDK Version:_ 26.5
- _Active Developer Directory:_ /Applications/Xcode.app/Contents/Developer
- _Xcode Version:_ 26.6
- _Xcode Build:_ 17F113
- _SDK Path:_ /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk
- _Metal SDK:_ MacOSX26.5.sdk
- _Metal Compiler Version:_ Apple metal version 32023.883 (metalfe-32023.883)
- _Metallib Linker Version:_ AIR-LLD 32023.883 (metalfe-32023.883) (compatible
  with legacy metallib linker)
- _Apple Clang Version:_ Apple clang version 21.0.0 (clang-2100.1.1.101)
- _Python Version:_ 3.13.13
- _Architecture:_ arm64
- _GPU/Chip:_ Apple M5 Max
- _GPU Cores:_ 40
- _Metal Support:_ Metal 4
- _RAM:_ 128.0 GB
- _CPU Cores (Physical):_ 18
- _CPU Cores (Logical):_ 18
- _MLX Install Type:_ editable local source
- _MLX Distribution Root:_ /Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages
- _mlx-metal Distribution:_ not installed; local editable mlx supplies backend
- _MLX Core Extension:_ /Users/jrp/Documents/AI/mlx/mlx/python/mlx/core.cpython-313-darwin.so
- _MLX Metallib:_ /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib
  (162,451,352 bytes,
  sha256=7e5c9a3a3225bf3b04a5fe67c50602975d3698a45e2113433465848af47fd70c)
- _MLX libmlx.dylib:_ /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib
  (21,747,136 bytes,
  sha256=6ae55e0952bcc7c19fd80f6f62dd1d0158448e0a58ce92384445c8f333f352ee)

## Library Versions

- `numpy`: `2.5.0`
- `mlx`: `0.32.0.dev20260703+de7b4ed9`
- `mlx-vlm`: `0.6.3`
- `mlx-lm`: `0.31.3`
- `mlx-audio`: `0.4.4`
- `huggingface-hub`: `1.22.0`
- `transformers`: `5.12.1`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.3.0`

Report generated on: 2026-07-04 00:01:55 BST
