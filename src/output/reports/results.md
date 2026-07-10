# Model Performance Results

Generated on: 2026-07-10 21:31:49 BST

## Run Contract

- Mode: triage
- Semantic rankings: ungrounded (ungrounded)
- Primary selection tasks: brief captions; structured title/description/keywords when descriptive image metadata is available

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ 1 (top owners: mlx-vlm=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=2, mechanically clean
  outputs=48/60.
- _Useful now:_ 29 model(s) shortlisted for caption review.
- _Review watchlist:_ 21 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Quality signal frequency:_ cutoff=7, reasoning_leak=4, harness=2,
  generation_loop=2, long_context=1, context_budget=1.

### Runtime

- _Runtime pattern:_ post-prefill decode dominates measured phase time (38%;
  19/61 measured model(s)).
- _Phase totals:_ model load=132.35s, local prompt prep=0.21s, upstream
  prefill / first-token=47.82s, post-prefill decode=206.16s, generation total
  (unsplit)=148.26s, cleanup=7.69s.
- _Generation total:_ 402.25s across 61 model(s); upstream prefill /
  first-token split available for 60/61 model(s).
- _What this likely means:_ Most measured runtime is spent generating after
  the first token is available.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=52, exception=1, max_tokens=8.
- _Validation overhead:_ 0.11s total (avg 0.00s across 61 model(s)).
- _Upstream prefill / first-token latency:_ Avg 0.80s | Min 0.02s | Max 5.02s
  across 60 model(s).

## 🏆 Performance Highlights

- **Fastest:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (531.9 tps)
- **💾 Most efficient:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (1.0 GB)
- **⚡ Fastest load:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (0.35s)
- **📊 Average TPS:** 92.7 across 60 models

## 📈 Resource Usage

- **Input image size:** 0.31 MP
- **Average peak delta from post-load:** 3.14 GB
- **Peak memory delta / MP:** 10452 MB/MP
- **Average peak memory:** 17.8 GB
- **Memory efficiency:** 44 tokens/GB

## ⚠️ Quality Issues

- **❌ Failed Models (1):**
  - `mlx-community/gemma-4-31b-bf16` (`Model Error`)
- **🔄 Repetitive Output (1):**
  - `mlx-community/gemma-3n-E2B-4bit` (token: `phrase: "have this image. have..."`)
- **📝 Formatting Issues (1):**
  - `mlx-community/Idefics3-8B-Llama3-bf16`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 92.7 | Min: 4.14 | Max: 532
- **Peak Memory**: Avg: 18 | Min: 1.0 | Max: 71
- **Total Time**: Avg: 6.17s | Min: 0.42s | Max: 47.46s
- **Generation Time**: Avg: 4.23s | Min: 0.06s | Max: 44.13s
- **Model Load Time**: Avg: 1.92s | Min: 0.35s | Max: 6.70s

## Caption Selection

Triage mode suppresses cataloging and keyword scores. Brief-caption
recommendations are ungrounded unless descriptive image metadata is present;
use `model_selection.md` for the caption shortlist and `model_gallery.md` for
full output evidence.

## 🚨 Failures by Package (Actionable)

| Package   |   Failures | Error Types   | Affected Models                  |
|-----------|------------|---------------|----------------------------------|
| `mlx-vlm` |          1 | Model Error   | `mlx-community/gemma-4-31b-bf16` |

### Actionable Items by Package

#### mlx-vlm

- mlx-community/gemma-4-31b-bf16 (Model Error)
  - Error: `Model runtime error during generation for mlx-community/gemma-4-31b-bf16: [METAL] Command buffer execution failed: In...`
  - Type: `ValueError`

_Prompt used:_

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Describe this image briefly.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 544.01s

_Detailed machine-readable metrics remain in `results.tsv` and
`results.jsonl`; this Markdown table keeps the high-signal columns for human
review._

<!-- markdownlint-disable MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Gen TPS |   Peak (GB) | Finish Reason   |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                     |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|----------:|------------:|:----------------|-----------------:|-----------:|------------:|:-----------------------------------|----------------:|
| `mlx-community/gemma-4-31b-bf16`                        |                   |                       |                |           |             |                 |          148.26s |     17.01s |     165.28s | gibberish(char_noise)              |         mlx-vlm |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      |                80 |                    10 |             90 |       532 |           1 | stop            |            0.06s |      0.35s |       0.42s |                                    |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               269 |                    10 |            279 |       320 |           3 | stop            |            0.12s |      0.52s |       0.64s |                                    |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |                26 |                    15 |             41 |       304 |         2.1 | stop            |            0.16s |      0.62s |       0.79s |                                    |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |               269 |                    13 |            282 |       190 |         4.1 | stop            |            0.16s |      0.64s |       0.81s |                                    |                 |
| `mlx-community/MiniCPM-V-4.6-8bit`                      |               228 |                    17 |            245 |       267 |           3 | stop            |            0.20s |      0.91s |       1.13s |                                    |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |                22 |                    81 |            103 |       374 |         1.9 | stop            |            0.30s |      0.59s |       0.89s |                                    |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |               417 |                    52 |            469 |       329 |         2.5 | stop            |            0.32s |      0.50s |       0.83s |                                    |                 |
| `qnguyen3/nanoLLaVA`                                    |                22 |                    35 |             57 |       116 |         4.2 | stop            |            0.40s |      0.60s |       1.00s |                                    |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             1,196 |                    13 |          1,209 |       129 |         5.5 | stop            |            0.52s |      0.69s |       1.22s |                                    |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,196 |                    13 |          1,209 |       124 |         5.5 | stop            |            0.53s |      0.72s |       1.26s |                                    |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |               770 |                    19 |            789 |      58.9 |         9.2 | stop            |            0.59s |      0.90s |       1.49s |                                    |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |               770 |                    19 |            789 |      57.5 |         9.2 | stop            |            0.61s |      0.96s |       1.57s |                                    |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |               973 |                    97 |          1,070 |       203 |         4.5 | stop            |            0.75s |      0.97s |       1.74s |                                    |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               286 |                    21 |            307 |      62.6 |          17 | stop            |            0.78s |      2.68s |       3.47s |                                    |                 |
| `mlx-community/Qwen3-VL-2B-Instruct-bf16`               |               315 |                    89 |            404 |       133 |         5.3 | stop            |            0.79s |      0.80s |       1.59s |                                    |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |               315 |                    89 |            404 |       133 |         5.2 | stop            |            0.79s |      0.68s |       1.48s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |               319 |                    51 |            370 |       118 |          21 | stop            |            0.80s |      2.44s |       3.26s |                                    |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |               408 |                    42 |            450 |      83.9 |         7.7 | stop            |            0.82s |      1.29s |       2.12s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |               319 |                    61 |            380 |      99.5 |          30 | stop            |            1.07s |      3.13s |       4.20s |                                    |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |               319 |                    75 |            394 |      95.4 |           7 | stop            |            1.07s |      1.43s |       2.51s |                                    |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,103 |                     3 |          4,106 |      84.5 |         4.6 | stop            |            1.12s |      1.17s |       2.30s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`         |               286 |                    19 |            305 |      24.4 |          28 | stop            |            1.23s |      3.14s |       4.38s |                                    |                 |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit`          |               286 |                    21 |            307 |      27.5 |          29 | stop            |            1.25s |      3.33s |       4.61s |                                    |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |               417 |                    65 |            482 |      66.6 |          10 | stop            |            1.27s |      1.12s |       2.40s |                                    |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |               408 |                    58 |            466 |        59 |          10 | stop            |            1.36s |      1.42s |       2.79s |                                    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |               974 |                    54 |          1,028 |      66.8 |          10 | stop            |            1.47s |      1.50s |       2.98s |                                    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |               974 |                    60 |          1,034 |      70.2 |         9.8 | stop            |            1.51s |      1.47s |       2.99s |                                    |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |               317 |                   188 |            505 |       134 |         5.3 | stop            |            1.52s |      0.69s |       2.22s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,327 |                    23 |          2,350 |      33.7 |          19 | stop            |            1.82s |      1.95s |       3.78s | formatting                         |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |                97 |                   200 |            297 |       132 |         5.5 | length          |            1.86s |      0.61s |       2.48s | token-cap                          |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               266 |                   200 |            466 |       123 |           6 | length          |            1.87s |      1.42s |       3.30s | repetitive(phrase: "have th...     |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               286 |                    25 |            311 |        21 |          20 | stop            |            1.96s |      2.61s |       4.59s |                                    |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |                16 |                    25 |             41 |      22.7 |          15 | stop            |            2.46s |      1.50s |       3.98s |                                    |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |               319 |                    64 |            383 |      33.6 |          19 | stop            |            2.52s |      2.08s |       4.62s |                                    |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |               399 |                   167 |            566 |      77.9 |          16 | stop            |            2.53s |      1.95s |       4.49s | reasoning-leak                     |                 |
| `mlx-community/Ornith-1.0-35B-bf16`                     |               319 |                    41 |            360 |      61.8 |          71 | stop            |            2.66s |      6.70s |       9.39s |                                    |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             3,341 |                    48 |          3,389 |      33.9 |          18 | stop            |            2.75s |      1.61s |       4.38s |                                    |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             2,356 |                    55 |          2,411 |      63.8 |         9.7 | stop            |            2.75s |      0.90s |       3.66s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |               319 |                    35 |            354 |      70.5 |          71 | stop            |            2.76s |      6.39s |       9.16s |                                    |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |               319 |                    41 |            360 |      19.5 |          30 | stop            |            2.80s |      3.02s |       5.84s |                                    |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               274 |                   124 |            398 |      48.4 |          17 | stop            |            2.90s |      2.29s |       5.21s |                                    |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |               399 |                   200 |            599 |      73.9 |          20 | length          |            3.12s |      2.18s |       5.31s | reasoning-leak, cutoff             |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |               319 |                    49 |            368 |      19.4 |          30 | stop            |            3.21s |      3.09s |       6.32s |                                    |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             3,341 |                    20 |          3,361 |      32.7 |          19 | stop            |            3.32s |      1.75s |       5.07s |                                    |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               275 |                    91 |            366 |      31.8 |          18 | stop            |            3.58s |      2.28s |       5.87s |                                    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,031 |                   101 |          1,132 |      35.9 |          11 | stop            |            3.67s |      1.56s |       5.25s |                                    |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |               441 |                    92 |            533 |      27.1 |          20 | stop            |            4.16s |      2.14s |       6.31s |                                    |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |               408 |                    91 |            499 |      42.9 |          63 | stop            |            4.18s |      5.57s |       9.76s |                                    |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |               439 |                   200 |            639 |      59.2 |          60 | length          |            4.41s |      4.76s |       9.18s | generation_loop(degeneration), ... |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               275 |                    70 |            345 |      17.8 |          32 | stop            |            4.79s |      3.31s |       8.11s |                                    |                 |
| `mlx-community/pixtral-12b-8bit`                        |             1,239 |                   200 |          1,439 |      39.5 |          15 | length          |            5.79s |      1.66s |       7.46s | cutoff                             |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |               745 |                    55 |            800 |      52.1 |          20 | stop            |            5.89s |      1.18s |       7.08s |                                    |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,031 |                   119 |          1,150 |      19.8 |          10 | stop            |            6.39s |      1.48s |       7.89s |                                    |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |               745 |                    55 |            800 |      30.4 |          27 | stop            |            6.85s |      1.71s |       8.57s |                                    |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             1,330 |                   200 |          1,530 |      34.4 |          14 | length          |            6.98s |      1.64s |       8.63s | reasoning-leak, cutoff             |                 |
| `mlx-community/pixtral-12b-bf16`                        |             1,239 |                   200 |          1,439 |        20 |          27 | length          |           10.80s |      2.57s |      13.37s | cutoff                             |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |                17 |                    77 |             94 |      4.14 |          25 | stop            |           20.58s |      2.17s |      22.76s |                                    |                 |
| `mlx-community/MolmoPoint-8B-fp16`                      |               766 |                   163 |            929 |      5.93 |          23 | stop            |           28.36s |      2.20s |      30.57s |                                    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,031 |                   159 |          1,190 |      5.41 |          26 | stop            |           30.52s |      2.48s |      33.01s |                                    |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |               399 |                   200 |            599 |      4.64 |          39 | length          |           44.13s |      3.31s |      47.46s | reasoning-leak, cutoff             |                 |

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
  (162,449,848 bytes,
  sha256=1078bd042297dbbf704a414617a7988c55b0001ea69d7cb478bcafa2fdfdeecb)
- _MLX libmlx.dylib:_ /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib
  (21,697,568 bytes,
  sha256=e61c827cd79f978aa5eacc136f65d6dea065005787f3a1457dc9d4512d6ee9cf)

## Library Versions

- `numpy`: `2.5.1`
- `mlx`: `0.32.1.dev20260710+4367c73b`
- `mlx-vlm`: `0.6.5`
- `mlx-lm`: `0.31.3`
- `mlx-audio`: `0.4.5`
- `huggingface-hub`: `1.23.0`
- `transformers`: `5.13.0`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.3.0`

Report generated on: 2026-07-10 21:31:49 BST
