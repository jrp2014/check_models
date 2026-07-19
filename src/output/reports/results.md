# Model Performance Results

Generated on: 2026-07-19 01:16:18 BST

## Run Contract

- Evaluation lane: triage
- Metadata exposed to prompt: no
- Semantic rankings: ungrounded (caption hygiene only)
- Primary selection tasks: brief captions; structured title/description/keywords when descriptive image metadata is available

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ none.
- _Maintainer signals:_ harness-risk successes=2, mechanically clean
  outputs=46/60.
- _Useful now:_ 31 model(s) shortlisted for caption review.
- _Review watchlist:_ 19 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Quality signal frequency:_ cutoff=7, formatting=3, harness=2,
  generation_loop=2, thinking_trace=2, long_context=1.

### Runtime

- _Runtime pattern:_ model load dominates measured phase time (43%; 41/60
  measured model(s)).
- _Phase totals:_ model load=208.75s, local prompt prep=0.22s, upstream model
  prefill / first-token=61.49s, input preparation + decode=208.31s,
  cleanup=6.60s.
- _Generation total:_ 269.80s across 60 model(s); upstream model prefill /
  first-token split available for 60/60 model(s).
- _What this likely means:_ Cold model load time is a major share of runtime
  for this cohort.
- _Suggested next action:_ Consider staged runs, model reuse, or narrowing the
  model set before reruns.
- _Termination reasons:_ completed=52, max_tokens=8.
- _Validation overhead:_ 0.11s total (avg 0.00s across 60 model(s)).
- _Upstream model prefill / first-token time:_ Avg 1.02s | Min 0.02s | Max
  6.04s across 60 model(s).

## 🏆 Performance Highlights

- **Fastest:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (537.4 tps)
- **💾 Most efficient:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (1.0 GB)
- **⚡ Fastest load:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (0.38s)
- **📊 Average TPS:** 92.3 across 60 models

## 📈 Resource Usage

- **Input image size:** 0.31 MP
- **Average peak delta from post-load:** 2.03 GB
- **Peak memory delta / MP:** 6754 MB/MP
- **Average peak memory:** 17.7 GB
- **Memory efficiency:** 44 tokens/GB

## ⚠️ Quality Issues

- **🔄 Repetitive Output (1):**
  - `mlx-community/gemma-3n-E2B-4bit` (token: `phrase: "have this image. have..."`)
- **📝 Formatting Issues (3):**
  - `mlx-community/Idefics3-8B-Llama3-bf16`
  - `mlx-community/diffusiongemma-26B-A4B-it-8bit`
  - `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 92.3 | Min: 4.66 | Max: 537
- **Peak Memory**: Avg: 18 | Min: 1.0 | Max: 71
- **Total Time**: Avg: 7.99s | Min: 0.45s | Max: 107.68s
- **Generation Time**: Avg: 4.50s | Min: 0.06s | Max: 43.98s
- **Model Load Time**: Avg: 3.48s | Min: 0.38s | Max: 63.69s

## Caption Selection

Triage mode suppresses cataloging and keyword scores. Brief-caption
recommendations are ungrounded unless descriptive image metadata is present;
use `model_selection.md` for the caption shortlist and `model_gallery.md` for
full output evidence.

_Prompt used:_

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Describe this image briefly.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 486.64s

_Detailed machine-readable metrics remain in `results.tsv` and
`results.jsonl`; this Markdown table keeps the high-signal columns for human
review._

<!-- markdownlint-disable MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Gen TPS |                                         Peak (GB) | Finish Reason   |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                     | Error Package   |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|----------:|--------------------------------------------------:|:----------------|-----------------:|-----------:|------------:|:-----------------------------------|:----------------|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      |                80 |                    10 |             90 |       537 | 1.0 GB (0.883% of 108 GB recommended working set) | stop            |            0.06s |      0.38s |       0.45s |                                    |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               269 |                    10 |            279 |       298 |  3.0 GB (2.58% of 108 GB recommended working set) | stop            |            0.12s |      0.78s |       0.90s |                                    |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |                26 |                    15 |             41 |       329 |  2.1 GB (1.82% of 108 GB recommended working set) | stop            |            0.15s |      0.94s |       1.09s |                                    |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |               269 |                    13 |            282 |       180 |  4.1 GB (3.56% of 108 GB recommended working set) | stop            |            0.17s |      0.65s |       0.82s |                                    |                 |
| `mlx-community/MiniCPM-V-4.6-8bit`                      |               228 |                    17 |            245 |       238 |  3.0 GB (2.59% of 108 GB recommended working set) | stop            |            0.21s |      1.09s |       1.31s |                                    |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |                22 |                    81 |            103 |       354 |  1.8 GB (1.58% of 108 GB recommended working set) | stop            |            0.31s |      0.53s |       0.85s |                                    |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |               417 |                    52 |            469 |       321 |  2.5 GB (2.17% of 108 GB recommended working set) | stop            |            0.33s |      0.65s |       0.99s |                                    |                 |
| `qnguyen3/nanoLLaVA`                                    |                22 |                    35 |             57 |       115 |  4.0 GB (3.47% of 108 GB recommended working set) | stop            |            0.43s |      1.07s |       1.50s |                                    |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             1,196 |                    13 |          1,209 |       124 |  5.5 GB (4.75% of 108 GB recommended working set) | stop            |            0.54s |      0.83s |       1.37s |                                    |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               286 |                    21 |            307 |       126 |   16 GB (14.1% of 108 GB recommended working set) | stop            |            0.57s |      2.74s |       3.34s |                                    |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |               770 |                    19 |            789 |      60.4 |     9.2 GB (8% of 108 GB recommended working set) | stop            |            0.59s |      1.35s |       1.94s |                                    |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |               770 |                    19 |            789 |      58.5 |  9.2 GB (7.99% of 108 GB recommended working set) | stop            |            0.61s |      2.08s |       2.69s |                                    |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,196 |                    13 |          1,209 |       127 |  5.5 GB (4.74% of 108 GB recommended working set) | stop            |            0.63s |      1.30s |       1.93s |                                    |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |               973 |                    97 |          1,070 |       203 |  4.5 GB (3.91% of 108 GB recommended working set) | stop            |            0.76s |      2.28s |       3.05s |                                    |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |               408 |                    42 |            450 |      91.7 |  7.7 GB (6.65% of 108 GB recommended working set) | stop            |            0.77s |      1.38s |       2.16s |                                    |                 |
| `mlx-community/Qwen3-VL-2B-Instruct-bf16`               |               315 |                    89 |            404 |       134 |  5.3 GB (4.56% of 108 GB recommended working set) | stop            |            0.78s |      0.83s |       1.62s |                                    |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |               315 |                    89 |            404 |       136 |  4.8 GB (4.17% of 108 GB recommended working set) | stop            |            0.79s |      0.71s |       1.51s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |               319 |                    51 |            370 |       112 |   21 GB (18.6% of 108 GB recommended working set) | stop            |            0.84s |      2.87s |       3.73s |                                    |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |               319 |                    75 |            394 |       101 |  7.0 GB (6.08% of 108 GB recommended working set) | stop            |            1.02s |      1.59s |       2.63s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |               319 |                    61 |            380 |      94.2 |   30 GB (26.1% of 108 GB recommended working set) | stop            |            1.13s |      3.20s |       4.34s |                                    |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,103 |                     3 |          4,106 |        75 |  4.6 GB (3.96% of 108 GB recommended working set) | stop            |            1.18s |      1.87s |       3.05s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |               408 |                    58 |            466 |      63.8 |   10 GB (8.94% of 108 GB recommended working set) | stop            |            1.27s |      1.47s |       2.75s |                                    |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |               417 |                    65 |            482 |      65.7 |   10 GB (8.85% of 108 GB recommended working set) | stop            |            1.29s |      1.21s |       2.50s |                                    |                 |
| `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`         |               282 |                    23 |            305 |      24.5 |   28 GB (24.4% of 108 GB recommended working set) | stop            |            1.41s |      3.30s |       4.71s | formatting                         |                 |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit`          |               282 |                    30 |            312 |      32.1 |     29 GB (25% of 108 GB recommended working set) | stop            |            1.44s |      3.40s |       4.86s | formatting                         |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |               974 |                    54 |          1,028 |      66.5 |   10 GB (8.83% of 108 GB recommended working set) | stop            |            1.48s |      1.59s |       3.08s |                                    |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |               317 |                   188 |            505 |       136 |  5.3 GB (4.56% of 108 GB recommended working set) | stop            |            1.50s |      0.89s |       2.40s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |               974 |                    60 |          1,034 |      69.9 |  9.8 GB (8.46% of 108 GB recommended working set) | stop            |            1.53s |      1.52s |       3.06s |                                    |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               286 |                    25 |            311 |      28.6 |   19 GB (16.8% of 108 GB recommended working set) | stop            |            1.62s |      2.78s |       4.41s |                                    |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,327 |                    23 |          2,350 |      33.4 |   19 GB (16.2% of 108 GB recommended working set) | stop            |            1.77s |      2.19s |       3.97s | formatting                         |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |                97 |                   200 |            297 |       131 |  5.5 GB (4.75% of 108 GB recommended working set) | length          |            1.89s |      0.69s |       2.58s | token-cap                          |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               266 |                   200 |            466 |       119 |  6.0 GB (5.17% of 108 GB recommended working set) | length          |            1.93s |      1.62s |       3.56s | repetitive(phrase: "have th...     |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |               319 |                    64 |            383 |      33.6 |   19 GB (16.5% of 108 GB recommended working set) | stop            |            2.55s |      2.41s |       4.99s |                                    |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |                16 |                    25 |             41 |      21.1 |     15 GB (13% of 108 GB recommended working set) | stop            |            2.64s |      1.66s |       4.31s |                                    |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             3,341 |                    48 |          3,389 |      34.5 |     18 GB (16% of 108 GB recommended working set) | stop            |            2.68s |      1.76s |       4.45s |                                    |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             2,356 |                    55 |          2,411 |      62.7 |  9.7 GB (8.41% of 108 GB recommended working set) | stop            |            2.81s |      0.97s |       3.78s |                                    |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |               319 |                    41 |            360 |      19.6 |   30 GB (25.7% of 108 GB recommended working set) | stop            |            2.82s |      3.25s |       6.10s |                                    |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |               399 |                   167 |            566 |      79.9 |   16 GB (13.5% of 108 GB recommended working set) | stop            |            2.88s |      2.72s |       5.61s | thinking-trace                     |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               274 |                   124 |            398 |      48.5 |   17 GB (14.9% of 108 GB recommended working set) | stop            |            2.90s |      2.34s |       5.26s |                                    |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             3,341 |                    20 |          3,361 |      32.7 |   19 GB (16.3% of 108 GB recommended working set) | stop            |            3.04s |      1.67s |       4.71s |                                    |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |               319 |                    49 |            368 |      19.3 |   30 GB (25.8% of 108 GB recommended working set) | stop            |            3.25s |      3.15s |       6.42s |                                    |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |               399 |                   200 |            599 |      68.2 |   20 GB (17.2% of 108 GB recommended working set) | length          |            3.37s |      2.44s |       5.82s | thinking-incomplete, cutoff        |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               275 |                    91 |            366 |      31.5 |   18 GB (15.7% of 108 GB recommended working set) | stop            |            3.61s |      2.33s |       5.95s |                                    |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |               441 |                    92 |            533 |      30.7 |   20 GB (17.1% of 108 GB recommended working set) | stop            |            3.70s |      2.59s |       6.30s |                                    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,031 |                   101 |          1,132 |        34 |    11 GB (9.6% of 108 GB recommended working set) | stop            |            3.84s |      1.74s |       5.59s |                                    |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               275 |                    70 |            345 |      17.8 |   32 GB (27.4% of 108 GB recommended working set) | stop            |            4.81s |      3.53s |       8.35s |                                    |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |               745 |                    55 |            800 |      50.9 |   20 GB (17.2% of 108 GB recommended working set) | stop            |            5.18s |      1.46s |       6.65s |                                    |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             1,330 |                   200 |          1,530 |      44.3 |   14 GB (11.9% of 108 GB recommended working set) | length          |            5.45s |      7.34s |      12.80s | reasoning-leak, cutoff             |                 |
| `mlx-community/pixtral-12b-8bit`                        |             1,239 |                   200 |          1,439 |      39.8 |   15 GB (12.6% of 108 GB recommended working set) | length          |            5.80s |      1.79s |       7.60s | cutoff                             |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,031 |                   119 |          1,150 |      19.6 |   10 GB (9.04% of 108 GB recommended working set) | stop            |            6.50s |      1.76s |       8.26s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |               319 |                    35 |            354 |      69.3 |   71 GB (61.7% of 108 GB recommended working set) | stop            |            6.56s |     10.86s |      17.44s |                                    |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |               745 |                    55 |            800 |      30.6 |   27 GB (23.4% of 108 GB recommended working set) | stop            |            6.78s |      2.43s |       9.22s |                                    |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |               408 |                    91 |            499 |      51.6 |   63 GB (54.6% of 108 GB recommended working set) | stop            |            6.96s |      7.85s |      14.83s |                                    |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |               439 |                   200 |            639 |      63.4 |   60 GB (51.8% of 108 GB recommended working set) | length          |            8.74s |      9.92s |      18.67s | generation_loop(degeneration), ... |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               274 |                    36 |            310 |      7.93 |     63 GB (55% of 108 GB recommended working set) | stop            |            8.99s |      7.73s |      16.73s |                                    |                 |
| `mlx-community/pixtral-12b-bf16`                        |             1,239 |                   200 |          1,439 |      17.8 |     27 GB (23% of 108 GB recommended working set) | length          |           12.02s |      3.28s |      15.32s | cutoff                             |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |                17 |                    77 |             94 |      5.06 |   25 GB (21.6% of 108 GB recommended working set) | stop            |           17.70s |      2.99s |      20.70s |                                    |                 |
| `mlx-community/MolmoPoint-8B-fp16`                      |               766 |                   163 |            929 |      5.97 |   23 GB (20.1% of 108 GB recommended working set) | stop            |           28.24s |      2.63s |      30.89s |                                    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,031 |                   159 |          1,190 |      5.02 |   26 GB (22.4% of 108 GB recommended working set) | stop            |           32.87s |      2.72s |      35.60s |                                    |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |               399 |                   200 |            599 |      4.66 |     39 GB (34% of 108 GB recommended working set) | length          |           43.98s |     63.69s |     107.68s | thinking-trace, cutoff             |                 |

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
- _MLX Device:_ Apple M5 Max
- _GPU Architecture:_ applegpu_g17s
- _GPU Cores:_ 40
- _Metal Support:_ Metal 4
- _RAM:_ 128.0 GB
- _Recommended Working Set:_ 108 GB
- _Fused Attention:_ Available
- _CPU Cores (Physical):_ 18
- _CPU Cores (Logical):_ 18
- _MLX Install Type:_ editable local source
- _MLX Distribution Root:_ ~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages
- _mlx-metal Distribution:_ not installed; local editable mlx supplies backend
- _MLX Core Extension:_ ~/Documents/AI/mlx/mlx/python/mlx/core.cpython-313-darwin.so
- _MLX Metallib:_ ~/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib
  (162,449,848 bytes,
  sha256=1078bd042297dbbf704a414617a7988c55b0001ea69d7cb478bcafa2fdfdeecb)
- _MLX libmlx.dylib:_ ~/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib
  (21,697,568 bytes,
  sha256=107989d6cb7f822e366d22e8259ea8eb2e68c15271ab6dc981da277ecf282cb0)

## Library Versions

- `numpy`: `2.5.1`
- `mlx`: `0.32.1.dev20260719+b7c3dd6d`
- `mlx-vlm`: `0.6.5`
- `mlx-lm`: `0.31.3`
- `mlx-audio`: `0.4.4`
- `huggingface-hub`: `1.24.0`
- `transformers`: `5.14.1`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.3.0`

Report generated on: 2026-07-19 01:16:18 BST
