# Model Performance Results

_Generated on 2026-06-27 23:58:03 BST_

## Run Contract

- Mode: triage
- Semantic rankings: ungrounded (ungrounded)
- Primary selection tasks: brief captions; structured title/description/keywords when descriptive image metadata is available

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ 2 (top owners: mlx=1, mlx-vlm=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=5, mechanically clean
  outputs=45/58.
- _Useful now:_ 29 model(s) shortlisted for caption review.
- _Review watchlist:_ 22 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Quality signal frequency:_ cutoff=6, harness=5, reasoning_leak=4,
  text_sanity=3, generation_loop=3, lang_mixing=2.

### Runtime

- _Runtime pattern:_ post-prefill decode dominates measured phase time (50%;
  22/60 measured model(s)).
- _Phase totals:_ model load=134.08s, local prompt prep=0.20s, upstream
  prefill / first-token=80.23s, post-prefill decode=221.12s, cleanup=8.74s.
- _Generation total:_ 301.35s across 58 model(s); upstream prefill /
  first-token split available for 58/58 model(s).
- _What this likely means:_ Most measured runtime is spent generating after
  the first token is available.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=50, exception=2, max_tokens=8.
- _Validation overhead:_ 0.20s total (avg 0.00s across 60 model(s)).
- _Upstream prefill / first-token latency:_ Avg 1.38s | Min 0.02s | Max 7.89s
  across 58 model(s).

## 🏆 Performance Highlights

- **Fastest:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (556.0 tps)
- **💾 Most efficient:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (1.0 GB)
- **⚡ Fastest load:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (0.32s)
- **📊 Average TPS:** 84.6 across 58 models

## 📈 Resource Usage

- **Input image size:** 0.31 MP
- **Average peak delta from post-load:** 3.08 GB
- **Peak memory delta / MP:** 10256 MB/MP
- **Average peak memory:** 17.9 GB
- **Memory efficiency:** 44 tokens/GB

## ⚠️ Quality Issues

- **❌ Failed Models (2):**
  - `mlx-community/LFM2.5-VL-1.6B-bf16` (`Weight Mismatch`)
  - `mlx-community/MolmoPoint-8B-fp16` (`Model Error`)
- **🔄 Repetitive Output (1):**
  - `mlx-community/gemma-3n-E2B-4bit` (token: `phrase: "have this image. have..."`)
- **📝 Formatting Issues (1):**
  - `mlx-community/Idefics3-8B-Llama3-bf16`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 84.6 | Min: 3.37 | Max: 556
- **Peak Memory**: Avg: 18 | Min: 1.0 | Max: 71
- **Total Time**: Avg: 7.51s | Min: 0.39s | Max: 65.22s
- **Generation Time**: Avg: 5.20s | Min: 0.06s | Max: 61.29s
- **Model Load Time**: Avg: 2.30s | Min: 0.32s | Max: 11.29s

## Caption Selection

Triage mode suppresses cataloging and keyword scores. Brief-caption
recommendations are ungrounded unless descriptive image metadata is present;
use `model_selection.md` for the caption shortlist and `model_gallery.md` for
full output evidence.

## 🚨 Failures by Package (Actionable)

| Package   |   Failures | Error Types     | Affected Models                     |
|-----------|------------|-----------------|-------------------------------------|
| `mlx`     |          1 | Weight Mismatch | `mlx-community/LFM2.5-VL-1.6B-bf16` |
| `mlx-vlm` |          1 | Model Error     | `mlx-community/MolmoPoint-8B-fp16`  |

### Actionable Items by Package

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

_Overall runtime:_ 445.91s

_Detailed machine-readable metrics remain in `results.tsv` and
`results.jsonl`; this Markdown table keeps the high-signal columns for human
review._

<!-- markdownlint-disable MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Gen TPS |   Peak (GB) | Finish Reason   |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                     |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|----------:|------------:|:----------------|-----------------:|-----------:|------------:|:-----------------------------------|----------------:|
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |                   |                       |                |           |             |                 |                  |      0.13s |       0.14s |                                    |             mlx |
| `mlx-community/MolmoPoint-8B-fp16`                      |                   |                       |                |           |             |                 |                  |      0.46s |       0.47s |                                    |         mlx-vlm |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      |                80 |                    10 |             90 |       556 |           1 | stop            |            0.06s |      0.32s |       0.39s |                                    |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |                26 |                    15 |             41 |       325 |         2.1 | stop            |            0.16s |      0.59s |       0.76s |                                    |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               269 |                    10 |            279 |       230 |           3 | stop            |            0.20s |      0.57s |       0.78s |                                    |                 |
| `mlx-community/MiniCPM-V-4.6-8bit`                      |               228 |                    22 |            250 |       180 |           3 | stop            |            0.26s |      0.92s |       1.20s |                                    |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |                22 |                    81 |            103 |       383 |         1.9 | stop            |            0.30s |      0.44s |       0.74s |                                    |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |               417 |                    52 |            469 |       284 |         2.5 | stop            |            0.37s |      0.71s |       1.10s |                                    |                 |
| `qnguyen3/nanoLLaVA`                                    |                22 |                    35 |             57 |       108 |         3.8 | stop            |            0.51s |      0.70s |       1.22s |                                    |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,196 |                    13 |          1,209 |       127 |         5.5 | stop            |            0.57s |      0.69s |       1.27s |                                    |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             1,196 |                    13 |          1,209 |       120 |         5.5 | stop            |            0.63s |      0.58s |       1.22s |                                    |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               286 |                    24 |            310 |       115 |          17 | stop            |            0.64s |      3.02s |       3.66s |                                    |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |               770 |                    19 |            789 |      51.7 |         9.2 | stop            |            0.71s |      0.95s |       1.67s |                                    |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |               408 |                    42 |            450 |      93.2 |         7.7 | stop            |            0.77s |      1.57s |       2.35s |                                    |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |               315 |                    96 |            411 |       133 |         5.2 | stop            |            0.85s |      0.66s |       1.53s |                                    |                 |
| `mlx-community/Qwen3-VL-2B-Instruct-bf16`               |               315 |                    96 |            411 |       129 |         5.2 | stop            |            0.87s |      0.79s |       1.68s |                                    |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |               949 |                    99 |          1,048 |       179 |         4.4 | stop            |            0.88s |      0.88s |       1.77s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |               319 |                    63 |            382 |       120 |          21 | stop            |            0.97s |      2.41s |       3.39s |                                    |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |               319 |                    71 |            390 |      98.5 |           7 | stop            |            1.01s |      1.41s |       2.42s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |               319 |                    58 |            377 |       102 |          30 | stop            |            1.32s |      3.17s |       4.50s |                                    |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,103 |                     3 |          4,106 |      71.9 |         4.6 | stop            |            1.38s |      1.49s |       2.88s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |               408 |                    69 |            477 |      63.2 |          10 | stop            |            1.47s |      1.39s |       2.87s |                                    |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               286 |                    25 |            311 |      27.1 |          19 | stop            |            1.76s |      2.69s |       4.47s |                                    |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,327 |                    23 |          2,350 |      34.5 |          19 | stop            |            1.79s |      2.00s |       3.81s | formatting                         |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |               950 |                    52 |          1,002 |      61.3 |          10 | stop            |            1.89s |      1.35s |       3.25s |                                    |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |                97 |                   187 |            284 |       133 |         5.5 | stop            |            2.02s |      0.57s |       2.59s |                                    |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |               417 |                    65 |            482 |        38 |          10 | stop            |            2.07s |      1.11s |       3.18s |                                    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |               950 |                    68 |          1,018 |      49.2 |         9.7 | stop            |            2.36s |      1.27s |       3.65s |                                    |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |               317 |                   196 |            513 |       102 |         5.3 | stop            |            2.48s |      0.75s |       3.25s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               266 |                   200 |            466 |      72.9 |         5.8 | length          |            3.04s |      1.51s |       4.56s | repetitive(phrase: "have th...     |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |               399 |                   200 |            599 |      77.1 |          16 | length          |            3.07s |      2.14s |       5.23s | reasoning-leak, token-cap          |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |               319 |                    40 |            359 |        19 |          30 | stop            |            3.08s |      3.09s |       6.20s |                                    |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |               319 |                    71 |            390 |      30.3 |          19 | stop            |            3.13s |      2.10s |       5.25s |                                    |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |               319 |                    49 |            368 |      19.1 |          30 | stop            |            3.30s |      3.00s |       6.33s |                                    |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |               399 |                   200 |            599 |      66.6 |          20 | length          |            3.43s |      2.16s |       5.61s | reasoning-leak, cutoff             |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |               417 |                    90 |            507 |      32.8 |          19 | stop            |            3.46s |      2.05s |       5.52s | ⚠️harness(encoding)                |                 |
| `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`         |               286 |                    16 |            302 |       7.3 |          28 | stop            |            3.61s |      3.99s |       7.62s |                                    |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             3,341 |                    48 |          3,389 |      35.5 |          18 | stop            |            3.64s |      1.56s |       5.21s |                                    |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               274 |                   124 |            398 |      45.5 |          18 | stop            |            3.76s |      2.92s |       6.70s |                                    |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |               770 |                   200 |            970 |      56.3 |         9.2 | length          |            3.85s |      1.08s |       4.95s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |                16 |                    25 |             41 |      13.3 |          15 | stop            |            3.87s |      1.50s |       5.38s |                                    |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             2,356 |                    55 |          2,411 |      59.4 |         9.7 | stop            |            3.96s |      0.88s |       4.85s |                                    |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             3,341 |                    15 |          3,356 |      23.7 |          19 | stop            |            4.47s |      1.76s |       6.24s |                                    |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             1,330 |                   174 |          1,504 |      44.1 |          14 | stop            |            4.93s |      1.57s |       6.51s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit`          |               286 |                    26 |            312 |      6.94 |          29 | stop            |            5.10s |      3.47s |       8.61s |                                    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,031 |                   101 |          1,132 |      21.8 |          11 | stop            |            5.99s |      1.87s |       7.88s |                                    |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |               745 |                    55 |            800 |        51 |          20 | stop            |            6.03s |      1.17s |       7.20s |                                    |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               275 |                    70 |            345 |      12.5 |          33 | stop            |            6.58s |      3.43s |      10.02s |                                    |                 |
| `mlx-community/pixtral-12b-8bit`                        |             1,239 |                   200 |          1,439 |      33.4 |          15 | length          |            6.88s |      1.78s |       8.67s | cutoff                             |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |               745 |                    55 |            800 |      23.5 |          27 | stop            |            7.01s |      1.68s |       8.71s |                                    |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               275 |                    91 |            366 |      17.1 |          19 | stop            |            7.30s |      2.42s |       9.73s |                                    |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |               408 |                    71 |            479 |      51.5 |          63 | stop            |            8.60s |      8.96s |      17.57s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |               319 |                    55 |            374 |      58.5 |          71 | stop            |            8.84s |     11.29s |      20.15s |                                    |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,031 |                   119 |          1,150 |      13.2 |          10 | stop            |            9.53s |      1.49s |      11.04s |                                    |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |               439 |                   200 |            639 |      62.8 |          60 | length          |            9.54s |      9.12s |      18.67s | generation_loop(degeneration), ... |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               274 |                    36 |            310 |      7.61 |          64 | stop            |           11.03s |     10.76s |      21.81s |                                    |                 |
| `mlx-community/pixtral-12b-bf16`                        |             1,239 |                   200 |          1,439 |      18.7 |          27 | length          |           11.59s |      2.52s |      14.12s | cutoff                             |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |                17 |                    77 |             94 |      5.03 |          25 | stop            |           18.71s |      2.72s |      21.46s |                                    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,031 |                   159 |          1,190 |      4.81 |          26 | stop            |           34.44s |      2.57s |      37.04s |                                    |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |               399 |                   200 |            599 |      3.37 |          39 | length          |           61.29s |      3.91s |      65.22s | reasoning-leak, cutoff             |                 |

<!-- markdownlint-enable MD033 MD034 MD037 MD049 -->

_Companion artifacts:_

- _Model-selection shortlist:_ [model_selection.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_selection.md)
- _Standalone output gallery:_ [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md)
- _Automated review digest:_ [review.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/review.md)
- _Canonical run log:_ [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log)

---

## System/Hardware Information

- _OS:_ Darwin 25.5.0
- _macOS Version:_ 26.5.1
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
  (162,369,176 bytes,
  sha256=28369eb4da5ecc579c68497402a2b76dd8fa31df58c6ae2b42e7150dc3b90445)
- _MLX libmlx.dylib:_ /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib
  (21,710,112 bytes,
  sha256=44c96c82bbc3808ee6aea73ee73bd83f2a13c00fbd99eca6828ce5359001c319)

## Library Versions

- `numpy`: `2.5.0`
- `mlx`: `0.32.0.dev20260627+548dd80e`
- `mlx-vlm`: `0.6.3`
- `mlx-lm`: `0.31.3`
- `mlx-audio`: `0.4.4`
- `huggingface-hub`: `1.21.0`
- `transformers`: `5.12.1`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-06-27 23:58:03 BST_
