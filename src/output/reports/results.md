# Model Performance Results

Generated on: 2026-07-04 19:55:05 BST

## Run Contract

- Mode: triage
- Semantic rankings: ungrounded (ungrounded)
- Primary selection tasks: brief captions; structured title/description/keywords when descriptive image metadata is available

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ 3 (top owners: model-config=2, mlx=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=9, mechanically clean
  outputs=41/58.
- _Useful now:_ 19 model(s) shortlisted for caption review.
- _Review watchlist:_ 32 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Quality signal frequency:_ harness=9, cutoff=6, reasoning_leak=4,
  text_sanity=2, generation_loop=2, context_budget=1.

### Runtime

- _Runtime pattern:_ post-prefill decode dominates measured phase time (52%;
  17/61 measured model(s)).
- _Phase totals:_ model load=135.36s, local prompt prep=0.20s, upstream
  prefill / first-token=55.58s, post-prefill decode=213.37s, cleanup=7.04s.
- _Generation total:_ 268.95s across 58 model(s); upstream prefill /
  first-token split available for 58/58 model(s).
- _What this likely means:_ Most measured runtime is spent generating after
  the first token is available.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=51, exception=3, max_tokens=7.
- _Validation overhead:_ 0.10s total (avg 0.00s across 61 model(s)).
- _Upstream prefill / first-token latency:_ Avg 0.96s | Min 0.02s | Max 6.37s
  across 58 model(s).

## 🏆 Performance Highlights

- **Fastest:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (549.6 tps)
- **💾 Most efficient:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (1.0 GB)
- **⚡ Fastest load:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (0.37s)
- **📊 Average TPS:** 88.9 across 58 models

## 📈 Resource Usage

- **Input image size:** 0.31 MP
- **Average peak delta from post-load:** 3.23 GB
- **Peak memory delta / MP:** 10764 MB/MP
- **Average peak memory:** 18.5 GB
- **Memory efficiency:** 43 tokens/GB

## ⚠️ Quality Issues

- **❌ Failed Models (3):**
  - `mlx-community/MiniCPM-V-4.6-8bit` (`Model Error`)
  - `mlx-community/diffusiongemma-26B-A4B-it-8bit` (`Processor Error`)
  - `mlx-community/diffusiongemma-26B-A4B-it-mxfp8` (`Processor Error`)
- **🔄 Repetitive Output (1):**
  - `mlx-community/gemma-3n-E2B-4bit` (token: `phrase: "have this image. have..."`)
- **📝 Formatting Issues (1):**
  - `mlx-community/Idefics3-8B-Llama3-bf16`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 88.9 | Min: 4.81 | Max: 550
- **Peak Memory**: Avg: 19 | Min: 1.0 | Max: 71
- **Total Time**: Avg: 6.85s | Min: 0.45s | Max: 45.95s
- **Generation Time**: Avg: 4.64s | Min: 0.07s | Max: 42.60s
- **Model Load Time**: Avg: 2.20s | Min: 0.37s | Max: 8.68s

## Caption Selection

Triage mode suppresses cataloging and keyword scores. Brief-caption
recommendations are ungrounded unless descriptive image metadata is present;
use `model_selection.md` for the caption shortlist and `model_gallery.md` for
full output evidence.

## 🚨 Failures by Package (Actionable)

| Package        |   Failures | Error Types     | Affected Models                                                                                 |
|----------------|------------|-----------------|-------------------------------------------------------------------------------------------------|
| `model-config` |          2 | Processor Error | `mlx-community/diffusiongemma-26B-A4B-it-8bit`, `mlx-community/diffusiongemma-26B-A4B-it-mxfp8` |
| `mlx`          |          1 | Model Error     | `mlx-community/MiniCPM-V-4.6-8bit`                                                              |

### Actionable Items by Package

#### model-config

- mlx-community/diffusiongemma-26B-A4B-it-8bit (Processor Error)
  - Error: `Model preflight failed for mlx-community/diffusiongemma-26B-A4B-it-8bit: Loaded processor has no image_processor; exp...`
  - Type: `ValueError`
- mlx-community/diffusiongemma-26B-A4B-it-mxfp8 (Processor Error)
  - Error: `Model preflight failed for mlx-community/diffusiongemma-26B-A4B-it-mxfp8: Loaded processor has no image_processor; ex...`
  - Type: `ValueError`

#### mlx

- mlx-community/MiniCPM-V-4.6-8bit (Model Error)
  - Error: `Model loading failed: Received 512 parameters not in model: <br>language_model.model.model.embed_tokens.weight,<br>language...`
  - Type: `ValueError`

_Prompt used:_

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Describe this image briefly.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 415.23s

_Detailed machine-readable metrics remain in `results.tsv` and
`results.jsonl`; this Markdown table keeps the high-signal columns for human
review._

<!-- markdownlint-disable MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Gen TPS |   Peak (GB) | Finish Reason   |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                     |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|----------:|------------:|:----------------|-----------------:|-----------:|------------:|:-----------------------------------|----------------:|
| `mlx-community/MiniCPM-V-4.6-8bit`                      |                   |                       |                |           |             |                 |                  |      0.16s |       0.16s |                                    |             mlx |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit`          |                   |                       |                |           |             |                 |                  |      3.84s |       5.20s |                                    |    model-config |
| `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`         |                   |                       |                |           |             |                 |                  |      3.80s |       5.08s |                                    |    model-config |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      |                80 |                    10 |             90 |       550 |           1 | stop            |            0.07s |      0.37s |       0.45s |                                    |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               269 |                    10 |            279 |       297 |           3 | stop            |            0.14s |      0.54s |       0.68s |                                    |                 |
| `mlx-community/Qwen3-VL-2B-Instruct-bf16`               |               315 |                     3 |            318 |       160 |         5.3 | stop            |            0.14s |      0.78s |       0.93s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |                26 |                    15 |             41 |       323 |         2.1 | stop            |            0.16s |      0.60s |       0.77s |                                    |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |               269 |                    13 |            282 |       198 |         4.1 | stop            |            0.17s |      0.56s |       0.73s |                                    |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |               417 |                    25 |            442 |       243 |         2.5 | stop            |            0.27s |      0.54s |       0.81s |                                    |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |               417 |                     3 |            420 |      89.9 |          10 | stop            |            0.33s |      1.14s |       1.48s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |                22 |                    81 |            103 |       331 |         1.8 | stop            |            0.34s |      0.52s |       0.87s |                                    |                 |
| `qnguyen3/nanoLLaVA`                                    |                22 |                    35 |             57 |       115 |           4 | stop            |            0.43s |      0.55s |       0.99s |                                    |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             1,196 |                    13 |          1,209 |       129 |         5.5 | stop            |            0.53s |      0.61s |       1.15s |                                    |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |               770 |                    19 |            789 |        63 |         9.2 | stop            |            0.59s |      0.91s |       1.50s |                                    |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |               770 |                    19 |            789 |        58 |         9.2 | stop            |            0.62s |      0.93s |       1.55s |                                    |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               286 |                    24 |            310 |       114 |          17 | stop            |            0.63s |      2.37s |       3.03s |                                    |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,196 |                    13 |          1,209 |       120 |         5.5 | stop            |            0.69s |      0.83s |       1.53s |                                    |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |               317 |                    76 |            393 |       133 |         5.3 | stop            |            0.70s |      0.74s |       1.45s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |               408 |                    42 |            450 |      93.1 |         7.7 | stop            |            0.76s |      1.29s |       2.06s |                                    |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |               973 |                    98 |          1,071 |       188 |         4.5 | stop            |            0.81s |      0.93s |       1.75s |                                    |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |               315 |                     3 |            318 |        25 |         4.6 | stop            |            0.91s |      1.29s |       2.22s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |               319 |                    30 |            349 |      99.7 |          31 | stop            |            1.12s |      3.22s |       4.35s |                                    |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |               319 |                    20 |            339 |      34.1 |          19 | stop            |            1.20s |      2.17s |       3.39s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |               319 |                    50 |            369 |       118 |          22 | stop            |            1.21s |      2.58s |       3.81s |                                    |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,103 |                     3 |          4,106 |      77.3 |         4.6 | stop            |            1.33s |      1.16s |       2.50s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |               408 |                    69 |            477 |      64.2 |          10 | stop            |            1.45s |      1.41s |       2.87s |                                    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |               974 |                    50 |          1,024 |      62.8 |          10 | stop            |            1.62s |      1.39s |       3.02s |                                    |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               286 |                    25 |            311 |      28.7 |          19 | stop            |            1.63s |      2.95s |       4.59s |                                    |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |               319 |                     2 |            321 |      36.1 |          30 | stop            |            1.65s |      3.14s |       4.81s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |               974 |                    60 |          1,034 |        67 |         9.8 | stop            |            1.74s |      1.30s |       3.05s |                                    |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,327 |                    23 |          2,350 |      33.6 |          19 | stop            |            1.77s |      1.92s |       3.70s | formatting                         |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |                97 |                   187 |            284 |       132 |         5.5 | stop            |            1.78s |      0.66s |       2.44s |                                    |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               266 |                   200 |            466 |       120 |           6 | length          |            1.94s |      1.45s |       3.39s | repetitive(phrase: "have th...     |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |               319 |                   166 |            485 |        91 |         7.8 | stop            |            2.48s |      1.34s |       3.83s |                                    |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             3,341 |                    48 |          3,389 |      34.5 |          18 | stop            |            2.67s |      1.57s |       4.25s |                                    |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |               441 |                    65 |            506 |      33.2 |          19 | stop            |            2.68s |      2.13s |       4.82s | ⚠️harness(encoding)                |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             3,341 |                    15 |          3,356 |      33.4 |          19 | stop            |            2.89s |      1.69s |       4.58s |                                    |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               274 |                   124 |            398 |      48.5 |          17 | stop            |            2.90s |      2.24s |       5.15s |                                    |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |                16 |                    25 |             41 |      17.2 |          15 | stop            |            3.05s |      1.50s |       4.55s |                                    |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             2,356 |                    55 |          2,411 |      62.1 |         9.7 | stop            |            3.05s |      0.97s |       4.03s |                                    |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |               399 |                   200 |            599 |      72.7 |          20 | length          |            3.19s |      2.15s |       5.35s | reasoning-leak, cutoff             |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |               399 |                   200 |            599 |      79.8 |          16 | length          |            3.36s |      5.85s |       9.21s | reasoning-leak, token-cap          |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,031 |                   101 |          1,132 |      34.7 |          11 | stop            |            3.79s |      1.61s |       5.41s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |               319 |                     3 |            322 |      77.7 |          71 | stop            |            3.80s |      8.30s |      12.10s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |               408 |                    71 |            479 |      51.8 |          63 | stop            |            3.91s |      6.17s |      10.10s |                                    |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               275 |                    91 |            366 |      26.3 |          18 | stop            |            4.26s |      2.25s |       6.52s |                                    |                 |
| `mlx-community/Ornith-1.0-35B-bf16`                     |               319 |                    44 |            363 |      61.7 |          71 | stop            |            4.81s |      8.53s |      13.36s |                                    |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               275 |                    70 |            345 |      17.5 |          32 | stop            |            4.88s |      3.29s |       8.18s |                                    |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             1,330 |                   174 |          1,504 |      44.3 |          14 | stop            |            4.89s |      1.68s |       6.58s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/pixtral-12b-8bit`                        |             1,239 |                   200 |          1,439 |      40.2 |          15 | length          |            5.87s |      1.64s |       7.52s | cutoff                             |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |               745 |                    55 |            800 |      51.1 |          20 | stop            |            6.00s |      1.19s |       7.20s |                                    |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,031 |                   119 |          1,150 |      19.6 |          10 | stop            |            6.48s |      1.46s |       7.94s |                                    |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |               745 |                    54 |            799 |      30.4 |          27 | stop            |            6.84s |      1.74s |       8.59s |                                    |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               274 |                    36 |            310 |      7.55 |          64 | stop            |            7.49s |      6.17s |      13.67s |                                    |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |               439 |                   200 |            639 |        64 |          60 | length          |            9.52s |      8.68s |      18.21s | generation_loop(degeneration), ... |                 |
| `mlx-community/pixtral-12b-bf16`                        |             1,239 |                   200 |          1,439 |      19.9 |          27 | length          |           10.89s |      2.54s |      13.44s | cutoff                             |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |               319 |                    85 |            404 |      14.2 |          30 | stop            |           12.26s |      3.29s |      15.58s |                                    |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |                17 |                    77 |             94 |      5.06 |          25 | stop            |           17.32s |      2.78s |      20.11s |                                    |                 |
| `mlx-community/MolmoPoint-8B-fp16`                      |               766 |                   163 |            929 |      6.01 |          23 | stop            |           29.60s |      2.19s |      31.81s |                                    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,031 |                   159 |          1,190 |      5.39 |          26 | stop            |           30.71s |      2.45s |      33.17s |                                    |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |               399 |                   200 |            599 |      4.81 |          39 | length          |           42.60s |      3.34s |      45.95s | reasoning-leak, cutoff             |                 |

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
  sha256=53b0e529da8969b02cd891b10e5c7b24413dc65c0ccc092343d438e39e13a7d0)

## Library Versions

- `numpy`: `2.5.1`
- `mlx`: `0.32.0.dev20260704+de7b4ed9`
- `mlx-vlm`: `0.6.4`
- `mlx-lm`: `0.31.3`
- `mlx-audio`: `0.4.4`
- `huggingface-hub`: `1.22.0`
- `transformers`: `5.12.1`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.3.0`

Report generated on: 2026-07-04 19:55:05 BST
