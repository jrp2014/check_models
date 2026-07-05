# Model Performance Results

Generated on: 2026-07-05 01:34:36 BST

## Run Contract

- Mode: triage
- Semantic rankings: ungrounded (ungrounded)
- Primary selection tasks: brief captions; structured title/description/keywords when descriptive image metadata is available

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ 1 (top owners: huggingface-hub=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=8, mechanically clean
  outputs=43/60.
- _Useful now:_ 19 model(s) shortlisted for caption review.
- _Review watchlist:_ 33 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Quality signal frequency:_ harness=8, cutoff=6, reasoning_leak=4,
  text_sanity=3, generation_loop=2, long_context=1.

### Runtime

- _Runtime pattern:_ model load dominates measured phase time (67%; 38/61
  measured model(s)).
- _Phase totals:_ model load=597.46s, local prompt prep=0.23s, upstream
  prefill / first-token=63.86s, post-prefill decode=224.04s, cleanup=7.31s.
- _Generation total:_ 287.90s across 60 model(s); upstream prefill /
  first-token split available for 60/60 model(s).
- _What this likely means:_ Cold model load time is a major share of runtime
  for this cohort.
- _Suggested next action:_ Consider staged runs, model reuse, or narrowing the
  model set before reruns.
- _Termination reasons:_ completed=53, exception=1, max_tokens=7.
- _Validation overhead:_ 0.11s total (avg 0.00s across 61 model(s)).
- _Upstream prefill / first-token latency:_ Avg 1.06s | Min 0.02s | Max 6.11s
  across 60 model(s).

## 🏆 Performance Highlights

- **Fastest:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (521.2 tps)
- **💾 Most efficient:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (1.0 GB)
- **⚡ Fastest load:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (0.34s)
- **📊 Average TPS:** 91.8 across 60 models

## 📈 Resource Usage

- **Input image size:** 0.31 MP
- **Average peak delta from post-load:** 3.19 GB
- **Peak memory delta / MP:** 10624 MB/MP
- **Average peak memory:** 18.6 GB
- **Memory efficiency:** 42 tokens/GB

## ⚠️ Quality Issues

- **❌ Failed Models (1):**
  - `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` (`Model Error`)
- **🔄 Repetitive Output (1):**
  - `mlx-community/gemma-3n-E2B-4bit` (token: `phrase: "have this image. have..."`)
- **📝 Formatting Issues (1):**
  - `mlx-community/Idefics3-8B-Llama3-bf16`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 91.8 | Min: 4.34 | Max: 521
- **Peak Memory**: Avg: 19 | Min: 1.0 | Max: 71
- **Total Time**: Avg: 7.14s | Min: 0.41s | Max: 49.12s
- **Generation Time**: Avg: 4.80s | Min: 0.06s | Max: 45.75s
- **Model Load Time**: Avg: 2.33s | Min: 0.34s | Max: 11.37s

## Caption Selection

Triage mode suppresses cataloging and keyword scores. Brief-caption
recommendations are ungrounded unless descriptive image metadata is present;
use `model_selection.md` for the caption shortlist and `model_gallery.md` for
full output evidence.

## 🚨 Failures by Package (Actionable)

| Package           |   Failures | Error Types   | Affected Models                                         |
|-------------------|------------|---------------|---------------------------------------------------------|
| `huggingface-hub` |          1 | Model Error   | `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |

### Actionable Items by Package

#### huggingface-hub

- mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit (Model Error)
  - Error: `Model loading failed: Operation timed out after 300.0 seconds`
  - Type: `ValueError`

_Prompt used:_

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Describe this image briefly.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 894.00s

_Detailed machine-readable metrics remain in `results.tsv` and
`results.jsonl`; this Markdown table keeps the high-signal columns for human
review._

<!-- markdownlint-disable MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Gen TPS |   Peak (GB) | Finish Reason   |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                     |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|----------:|------------:|:----------------|-----------------:|-----------:|------------:|:-----------------------------------|----------------:|
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |                   |                       |                |           |             |                 |                  |    457.68s |     457.69s |                                    | huggingface-hub |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      |                80 |                    10 |             90 |       521 |           1 | stop            |            0.06s |      0.34s |       0.41s |                                    |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               269 |                    10 |            279 |       310 |           3 | stop            |            0.13s |      0.48s |       0.61s |                                    |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |               315 |                     3 |            318 |       152 |         5.2 | stop            |            0.15s |      0.66s |       0.82s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/Qwen3-VL-2B-Instruct-bf16`               |               315 |                     3 |            318 |       143 |         5.3 | stop            |            0.16s |      0.72s |       0.88s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |                26 |                    15 |             41 |       320 |         2.1 | stop            |            0.17s |      0.62s |       0.80s |                                    |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |               269 |                    13 |            282 |       187 |         4.1 | stop            |            0.17s |      0.54s |       0.72s |                                    |                 |
| `mlx-community/MiniCPM-V-4.6-8bit`                      |               228 |                    22 |            250 |       263 |           3 | stop            |            0.23s |      0.83s |       1.08s |                                    |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |               417 |                    25 |            442 |       291 |         2.5 | stop            |            0.26s |      0.57s |       0.84s |                                    |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |                22 |                    81 |            103 |       357 |         1.8 | stop            |            0.32s |      0.45s |       0.78s |                                    |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |               417 |                     3 |            420 |      90.7 |          10 | stop            |            0.34s |      1.12s |       1.47s | ⚠️harness(prompt_template)         |                 |
| `qnguyen3/nanoLLaVA`                                    |                22 |                    35 |             57 |       113 |           4 | stop            |            0.44s |      0.53s |       0.97s |                                    |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,196 |                    13 |          1,209 |       126 |         5.5 | stop            |            0.56s |      0.65s |       1.22s |                                    |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             1,196 |                    13 |          1,209 |       126 |         5.5 | stop            |            0.59s |      0.55s |       1.14s |                                    |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               286 |                    24 |            310 |       118 |          17 | stop            |            0.62s |      2.32s |       2.96s |                                    |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |               770 |                    19 |            789 |      54.7 |         9.2 | stop            |            0.65s |      0.89s |       1.55s |                                    |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |               770 |                    19 |            789 |      52.6 |         9.2 | stop            |            0.70s |      0.91s |       1.62s |                                    |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |               317 |                    76 |            393 |       130 |         5.2 | stop            |            0.71s |      0.71s |       1.43s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |               408 |                    42 |            450 |      90.5 |         7.7 | stop            |            0.79s |      1.32s |       2.12s |                                    |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |               973 |                    98 |          1,071 |       195 |         4.5 | stop            |            0.79s |      0.89s |       1.69s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |               319 |                    30 |            349 |      96.2 |          31 | stop            |            1.13s |      3.08s |       4.22s |                                    |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |               319 |                    20 |            339 |      33.5 |          19 | stop            |            1.24s |      2.15s |       3.41s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |               319 |                    50 |            369 |       111 |          22 | stop            |            1.26s |      2.50s |       3.77s |                                    |                 |
| `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`         |               286 |                    15 |            301 |      20.3 |          28 | stop            |            1.34s |      3.13s |       4.48s |                                    |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,103 |                     3 |          4,106 |        70 |         4.6 | stop            |            1.41s |      1.12s |       2.54s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |               408 |                    69 |            477 |        63 |          10 | stop            |            1.50s |      1.39s |       2.90s |                                    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |               974 |                    50 |          1,024 |      64.7 |          10 | stop            |            1.58s |      1.36s |       2.94s |                                    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |               974 |                    60 |          1,034 |      67.6 |         9.8 | stop            |            1.64s |      1.31s |       2.97s |                                    |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               286 |                    25 |            311 |      27.4 |          19 | stop            |            1.71s |      2.60s |       4.32s |                                    |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |               319 |                     2 |            321 |      34.9 |          30 | stop            |            1.71s |      3.03s |       4.76s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit`          |               286 |                    26 |            312 |        25 |          29 | stop            |            1.74s |      3.24s |       5.01s |                                    |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |                97 |                   187 |            284 |       131 |         5.5 | stop            |            1.80s |      0.56s |       2.36s |                                    |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,327 |                    23 |          2,350 |      33.3 |          19 | stop            |            1.80s |      1.89s |       3.70s | formatting                         |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               266 |                   200 |            466 |       124 |           6 | length          |            1.88s |      1.41s |       3.29s | repetitive(phrase: "have th...     |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             3,341 |                    48 |          3,389 |      34.5 |          18 | stop            |            2.69s |      1.57s |       4.27s |                                    |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |               319 |                   166 |            485 |      83.9 |         7.8 | stop            |            2.77s |      1.37s |       4.16s | gibberish(mixed_script_noise)      |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             3,341 |                    15 |          3,356 |      33.6 |          19 | stop            |            2.89s |      1.64s |       4.54s |                                    |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               274 |                   124 |            398 |      48.5 |          17 | stop            |            2.91s |      2.21s |       5.13s |                                    |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |                16 |                    25 |             41 |      19.4 |          15 | stop            |            2.93s |      1.50s |       4.45s |                                    |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |               399 |                   200 |            599 |      70.2 |          20 | length          |            3.31s |      2.24s |       5.56s | reasoning-leak, cutoff             |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               275 |                    91 |            366 |      30.9 |          18 | stop            |            3.68s |      2.19s |       5.89s |                                    |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             2,356 |                    55 |          2,411 |      60.1 |         9.7 | stop            |            3.92s |      0.87s |       4.80s |                                    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,031 |                   101 |          1,132 |      32.1 |          11 | stop            |            4.09s |      1.66s |       5.76s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |               319 |                     3 |            322 |      70.4 |          71 | stop            |            4.65s |      9.18s |      13.84s | ⚠️harness(prompt_template)         |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |               399 |                   200 |            599 |      64.1 |          16 | length          |            4.94s |      5.85s |      10.80s | reasoning-leak, token-cap          |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |               745 |                    55 |            800 |      50.5 |          20 | stop            |            5.61s |      1.18s |       6.80s |                                    |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               275 |                    70 |            345 |      14.6 |          32 | stop            |            5.74s |      3.27s |       9.02s |                                    |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             1,330 |                   174 |          1,504 |      35.9 |          14 | stop            |            6.01s |      1.77s |       7.79s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/pixtral-12b-8bit`                        |             1,239 |                   200 |          1,439 |      38.1 |          15 | length          |            6.10s |      1.64s |       7.75s | cutoff                             |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,031 |                   119 |          1,150 |      19.3 |          10 | stop            |            6.58s |      1.44s |       8.03s |                                    |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |               745 |                    55 |            800 |        30 |          27 | stop            |            6.68s |      1.77s |       8.46s |                                    |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |               408 |                    71 |            479 |      51.1 |          63 | stop            |            7.51s |      8.38s |      15.91s |                                    |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               274 |                    36 |            310 |      8.11 |          63 | stop            |            8.24s |      6.78s |      15.04s |                                    |                 |
| `mlx-community/Ornith-1.0-35B-bf16`                     |               319 |                    44 |            363 |      54.6 |          71 | stop            |            8.36s |     11.37s |      19.75s |                                    |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |               439 |                   200 |            639 |      64.3 |          60 | length          |            9.18s |     10.66s |      19.90s | generation_loop(degeneration), ... |                 |
| `mlx-community/pixtral-12b-bf16`                        |             1,239 |                   200 |          1,439 |        20 |          27 | length          |           10.90s |      2.55s |      13.46s | cutoff                             |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |               319 |                    85 |            404 |      13.7 |          30 | stop            |           11.99s |      3.14s |      15.15s |                                    |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |                17 |                    77 |             94 |      4.34 |          25 | stop            |           21.24s |      3.72s |      24.98s |                                    |                 |
| `mlx-community/MolmoPoint-8B-fp16`                      |               766 |                   163 |            929 |      5.83 |          23 | stop            |           28.92s |      2.17s |      31.10s |                                    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,031 |                   159 |          1,190 |       5.4 |          26 | stop            |           30.70s |      2.45s |      33.16s |                                    |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |               399 |                   200 |            599 |      4.48 |          39 | length          |           45.75s |      3.36s |      49.12s | reasoning-leak, cutoff             |                 |

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
  sha256=9d942d98a9a9f3e42b3f22c6606bc1ee621d28a9fb512d0cdba6edbb9ef79df8)

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

Report generated on: 2026-07-05 01:34:36 BST
