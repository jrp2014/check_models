# Model Performance Results

Generated on: 2026-07-18 22:56:47 BST

## Run Contract

- Evaluation lane: triage
- Metadata exposed to prompt: no
- Semantic rankings: ungrounded (caption hygiene only)
- Primary selection tasks: brief captions; structured title/description/keywords when descriptive image metadata is available

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ none.
- _Maintainer signals:_ harness-risk successes=2, mechanically clean
  outputs=47/61.
- _Useful now:_ 31 model(s) shortlisted for caption review.
- _Review watchlist:_ 19 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Quality signal frequency:_ cutoff=7, formatting=3, harness=2,
  generation_loop=2, thinking_trace=2, long_context=1.

### Runtime

- _Runtime pattern:_ input preparation + decode dominates measured phase time
  (45%; 19/61 measured model(s)).
- _Phase totals:_ model load=155.27s, local prompt prep=0.25s, upstream model
  prefill / first-token=99.85s, input preparation + decode=218.29s,
  cleanup=8.23s.
- _Generation total:_ 318.13s across 61 model(s); upstream model prefill /
  first-token split available for 61/61 model(s).
- _What this likely means:_ Most residual generation-call time falls outside
  the upstream model-loop first-token window, combining input preparation with
  token decoding.
- _Suggested next action:_ Profile prepare_inputs() and image preprocessing,
  then use generation TPS and token counts to assess the decode contribution.
- _Termination reasons:_ completed=53, max_tokens=8.
- _Validation overhead:_ 0.13s total (avg 0.00s across 61 model(s)).
- _Upstream model prefill / first-token time:_ Avg 1.64s | Min 0.02s | Max
  20.41s across 61 model(s).

## 🏆 Performance Highlights

- **Fastest:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (530.9 tps)
- **💾 Most efficient:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (1.0 GB)
- **⚡ Fastest load:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (0.34s)
- **📊 Average TPS:** 88.3 across 61 models

## 📈 Resource Usage

- **Input image size:** 0.31 MP
- **Average peak delta from post-load:** 2.02 GB
- **Peak memory delta / MP:** 6723 MB/MP
- **Average peak memory:** 18.6 GB
- **Memory efficiency:** 42 tokens/GB

## ⚠️ Quality Issues

- **🔄 Repetitive Output (1):**
  - `mlx-community/gemma-3n-E2B-4bit` (token: `phrase: "have this image. have..."`)
- **📝 Formatting Issues (3):**
  - `mlx-community/Idefics3-8B-Llama3-bf16`
  - `mlx-community/diffusiongemma-26B-A4B-it-8bit`
  - `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 88.3 | Min: 4.34 | Max: 531
- **Peak Memory**: Avg: 19 | Min: 1.0 | Max: 71
- **Total Time**: Avg: 7.77s | Min: 0.40s | Max: 51.89s
- **Generation Time**: Avg: 5.22s | Min: 0.06s | Max: 47.71s
- **Model Load Time**: Avg: 2.55s | Min: 0.34s | Max: 13.16s

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

_Overall runtime:_ 483.60s

_Detailed machine-readable metrics remain in `results.tsv` and
`results.jsonl`; this Markdown table keeps the high-signal columns for human
review._

<!-- markdownlint-disable MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Gen TPS |                                         Peak (GB) | Finish Reason   |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                     |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|----------:|--------------------------------------------------:|:----------------|-----------------:|-----------:|------------:|:-----------------------------------|----------------:|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      |                80 |                    10 |             90 |       531 | 1.0 GB (0.883% of 108 GB recommended working set) | stop            |            0.06s |      0.34s |       0.40s |                                    |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               269 |                    10 |            279 |       282 |  3.0 GB (2.58% of 108 GB recommended working set) | stop            |            0.14s |      0.53s |       0.67s |                                    |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |                26 |                    15 |             41 |       314 |  2.1 GB (1.84% of 108 GB recommended working set) | stop            |            0.17s |      0.65s |       0.83s |                                    |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |               269 |                    13 |            282 |       195 |  4.1 GB (3.56% of 108 GB recommended working set) | stop            |            0.17s |      0.53s |       0.71s |                                    |                 |
| `mlx-community/MiniCPM-V-4.6-8bit`                      |               228 |                    17 |            245 |       250 |  3.0 GB (2.59% of 108 GB recommended working set) | stop            |            0.21s |      0.93s |       1.16s |                                    |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |                22 |                    81 |            103 |       344 |  1.9 GB (1.64% of 108 GB recommended working set) | stop            |            0.32s |      0.47s |       0.79s |                                    |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |               417 |                    52 |            469 |       322 |  2.5 GB (2.17% of 108 GB recommended working set) | stop            |            0.33s |      0.49s |       0.82s |                                    |                 |
| `qnguyen3/nanoLLaVA`                                    |                22 |                    35 |             57 |       114 |  4.0 GB (3.46% of 108 GB recommended working set) | stop            |            0.43s |      0.53s |       0.97s |                                    |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             1,196 |                    13 |          1,209 |       118 |  5.5 GB (4.75% of 108 GB recommended working set) | stop            |            0.55s |      0.60s |       1.15s |                                    |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,196 |                    13 |          1,209 |       126 |  5.5 GB (4.74% of 108 GB recommended working set) | stop            |            0.56s |      0.70s |       1.27s |                                    |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               286 |                    21 |            307 |       126 |   16 GB (14.1% of 108 GB recommended working set) | stop            |            0.57s |      2.35s |       2.95s |                                    |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |               770 |                    19 |            789 |      59.7 |     9.2 GB (8% of 108 GB recommended working set) | stop            |            0.61s |      0.89s |       1.51s |                                    |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |               770 |                    19 |            789 |      57.3 |  9.2 GB (7.99% of 108 GB recommended working set) | stop            |            0.62s |      0.97s |       1.60s |                                    |                 |
| `mlx-community/Qwen3-VL-2B-Instruct-bf16`               |               315 |                    89 |            404 |       131 |  5.2 GB (4.55% of 108 GB recommended working set) | stop            |            0.80s |      0.72s |       1.52s |                                    |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |               973 |                    97 |          1,070 |       183 |  4.5 GB (3.91% of 108 GB recommended working set) | stop            |            0.80s |      0.90s |       1.71s |                                    |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |               315 |                    89 |            404 |       127 |  5.1 GB (4.42% of 108 GB recommended working set) | stop            |            0.83s |      0.76s |       1.61s |                                    |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |               408 |                    42 |            450 |      80.8 |  7.7 GB (6.65% of 108 GB recommended working set) | stop            |            0.87s |      1.64s |       2.52s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |               319 |                    51 |            370 |       106 |   21 GB (18.6% of 108 GB recommended working set) | stop            |            0.88s |      2.65s |       3.53s |                                    |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |               319 |                    75 |            394 |      95.2 |  7.0 GB (6.08% of 108 GB recommended working set) | stop            |            1.08s |      1.41s |       2.50s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |               319 |                    61 |            380 |      98.7 |   30 GB (26.1% of 108 GB recommended working set) | stop            |            1.11s |      3.25s |       4.37s |                                    |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,103 |                     3 |          4,106 |      76.3 |  4.6 GB (3.96% of 108 GB recommended working set) | stop            |            1.20s |      1.19s |       2.39s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`         |               282 |                    23 |            305 |      35.1 |   28 GB (24.4% of 108 GB recommended working set) | stop            |            1.21s |      3.46s |       4.68s | formatting                         |                 |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit`          |               282 |                    25 |            307 |      32.2 |     29 GB (25% of 108 GB recommended working set) | stop            |            1.27s |      3.36s |       4.67s | formatting                         |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |               417 |                    65 |            482 |      59.3 |   10 GB (8.85% of 108 GB recommended working set) | stop            |            1.42s |      1.15s |       2.58s |                                    |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |               317 |                   188 |            505 |       130 |  5.3 GB (4.56% of 108 GB recommended working set) | stop            |            1.57s |      0.69s |       2.27s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               286 |                    25 |            311 |      28.7 |   19 GB (16.8% of 108 GB recommended working set) | stop            |            1.62s |      2.58s |       4.21s |                                    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |               974 |                    54 |          1,028 |        63 |   10 GB (8.83% of 108 GB recommended working set) | stop            |            1.66s |      1.47s |       3.13s |                                    |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |               408 |                    58 |            466 |      43.3 |   10 GB (8.94% of 108 GB recommended working set) | stop            |            1.73s |      1.73s |       3.48s |                                    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |               974 |                    60 |          1,034 |        58 |  9.8 GB (8.46% of 108 GB recommended working set) | stop            |            1.86s |      1.33s |       3.20s |                                    |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |                97 |                   200 |            297 |       122 |  5.5 GB (4.75% of 108 GB recommended working set) | length          |            2.00s |      0.62s |       2.62s | token-cap                          |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,327 |                    23 |          2,350 |      32.5 |   19 GB (16.2% of 108 GB recommended working set) | stop            |            2.07s |      1.94s |       4.02s | formatting                         |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               266 |                   200 |            466 |       110 |  6.0 GB (5.17% of 108 GB recommended working set) | length          |            2.09s |      1.49s |       3.59s | repetitive(phrase: "have th...     |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |               319 |                    64 |            383 |      33.4 |   19 GB (16.5% of 108 GB recommended working set) | stop            |            2.55s |      2.12s |       4.69s |                                    |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |                16 |                    25 |             41 |      19.8 |     15 GB (13% of 108 GB recommended working set) | stop            |            2.83s |      1.49s |       4.32s |                                    |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |               319 |                    41 |            360 |      19.4 |   30 GB (25.7% of 108 GB recommended working set) | stop            |            2.87s |      3.25s |       6.15s |                                    |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             2,356 |                    55 |          2,411 |      61.1 |  9.7 GB (8.41% of 108 GB recommended working set) | stop            |            2.91s |      0.88s |       3.80s |                                    |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               274 |                   124 |            398 |      47.6 |   17 GB (14.9% of 108 GB recommended working set) | stop            |            2.96s |      2.26s |       5.24s |                                    |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |               319 |                    49 |            368 |        18 |   30 GB (25.8% of 108 GB recommended working set) | stop            |            3.44s |      3.38s |       6.84s |                                    |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |               399 |                   200 |            599 |      65.3 |   20 GB (17.2% of 108 GB recommended working set) | length          |            3.51s |      2.30s |       5.83s | thinking-incomplete, cutoff        |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               275 |                    91 |            366 |      31.2 |   18 GB (15.7% of 108 GB recommended working set) | stop            |            3.68s |      2.31s |       6.00s |                                    |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |               399 |                   167 |            566 |        69 |   16 GB (13.8% of 108 GB recommended working set) | stop            |            3.80s |      5.00s |       8.81s | thinking-trace                     |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |               441 |                    92 |            533 |      29.7 |   20 GB (17.1% of 108 GB recommended working set) | stop            |            3.86s |      2.23s |       6.11s |                                    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,031 |                   101 |          1,132 |      33.5 |    11 GB (9.6% of 108 GB recommended working set) | stop            |            3.88s |      1.72s |       5.61s |                                    |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             3,341 |                    48 |          3,389 |      32.7 |     18 GB (16% of 108 GB recommended working set) | stop            |            4.14s |      1.96s |       6.12s |                                    |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               275 |                    70 |            345 |      16.3 |   32 GB (27.4% of 108 GB recommended working set) | stop            |            5.22s |      3.46s |       8.69s |                                    |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |               745 |                    55 |            800 |      51.4 |   20 GB (17.2% of 108 GB recommended working set) | stop            |            5.67s |      1.27s |       6.94s |                                    |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             3,341 |                    20 |          3,361 |      31.3 |   19 GB (16.3% of 108 GB recommended working set) | stop            |            5.81s |      1.96s |       7.78s |                                    |                 |
| `mlx-community/pixtral-12b-8bit`                        |             1,239 |                   200 |          1,439 |      39.9 |   15 GB (12.6% of 108 GB recommended working set) | length          |            5.85s |      1.65s |       7.51s | cutoff                             |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             1,330 |                   200 |          1,530 |      37.8 |   14 GB (11.9% of 108 GB recommended working set) | length          |            6.29s |      1.71s |       8.02s | reasoning-leak, cutoff             |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,031 |                   119 |          1,150 |      19.4 |   10 GB (9.04% of 108 GB recommended working set) | stop            |            6.61s |      1.50s |       8.11s |                                    |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |               745 |                    53 |            798 |      30.2 |   27 GB (23.4% of 108 GB recommended working set) | stop            |            6.90s |      1.72s |       8.63s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |               319 |                    35 |            354 |      64.4 |   71 GB (61.7% of 108 GB recommended working set) | stop            |            8.50s |     11.94s |      20.44s |                                    |                 |
| `mlx-community/pixtral-12b-bf16`                        |             1,239 |                   200 |          1,439 |      20.7 |     27 GB (23% of 108 GB recommended working set) | length          |           10.48s |      2.55s |      13.04s | cutoff                             |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               274 |                    36 |            310 |      7.84 |     63 GB (55% of 108 GB recommended working set) | stop            |           11.07s |     10.31s |      21.40s |                                    |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |               439 |                   200 |            639 |      58.9 |   59 GB (51.5% of 108 GB recommended working set) | length          |           12.37s |      9.75s |      22.13s | generation_loop(degeneration), ... |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |               408 |                    91 |            499 |      28.2 |   63 GB (54.6% of 108 GB recommended working set) | stop            |           12.79s |     10.49s |      23.30s |                                    |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |                17 |                    77 |             94 |      5.05 |   25 GB (21.6% of 108 GB recommended working set) | stop            |           17.99s |      3.70s |      21.69s |                                    |                 |
| `mlx-community/Ornith-1.0-35B-bf16`                     |               319 |                    41 |            360 |      42.9 |   71 GB (61.7% of 108 GB recommended working set) | stop            |           21.38s |     13.16s |      34.56s |                                    |                 |
| `mlx-community/MolmoPoint-8B-fp16`                      |               766 |                   163 |            929 |      5.89 |   23 GB (20.1% of 108 GB recommended working set) | stop            |           28.64s |      2.24s |      30.90s |                                    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,031 |                   159 |          1,190 |      4.91 |   26 GB (22.4% of 108 GB recommended working set) | stop            |           33.62s |      2.50s |      36.13s |                                    |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |               399 |                   200 |            599 |      4.34 |     39 GB (34% of 108 GB recommended working set) | length          |           47.71s |      4.16s |      51.89s | thinking-trace, cutoff             |                 |

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
  sha256=bef52521877d51c72e90bb0a5dc2823d77e222799855fc5f7bbb21ac8bcd129f)

## Library Versions

- `numpy`: `2.5.1`
- `mlx`: `0.32.1.dev20260718+b7c3dd6d`
- `mlx-vlm`: `0.6.5`
- `mlx-lm`: `0.31.3`
- `mlx-audio`: `0.4.4`
- `huggingface-hub`: `1.24.0`
- `transformers`: `5.14.1`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.3.0`

Report generated on: 2026-07-18 22:56:47 BST
