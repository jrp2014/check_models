# Model Performance Results

Generated on: 2026-07-10 15:07:22 BST

## Run Contract

- Mode: triage
- Semantic rankings: ungrounded (ungrounded)
- Primary selection tasks: brief captions; structured title/description/keywords when descriptive image metadata is available

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ 2 (top owners: mlx-vlm=1, huggingface-hub=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=2, mechanically clean
  outputs=47/59.
- _Useful now:_ 28 model(s) shortlisted for caption review.
- _Review watchlist:_ 21 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Quality signal frequency:_ cutoff=7, reasoning_leak=4, harness=2,
  generation_loop=2, long_context=1, context_budget=1.

### Runtime

- _Runtime pattern:_ model load dominates measured phase time (68%; 35/61
  measured model(s)).
- _Phase totals:_ model load=885.63s, local prompt prep=0.21s, upstream
  prefill / first-token=51.80s, post-prefill decode=198.92s, generation total
  (unsplit)=159.27s, cleanup=7.71s.
- _Generation total:_ 409.99s across 60 model(s); upstream prefill /
  first-token split available for 59/60 model(s).
- _What this likely means:_ Cold model load time is a major share of runtime
  for this cohort.
- _Suggested next action:_ Consider staged runs, model reuse, or narrowing the
  model set before reruns.
- _Termination reasons:_ completed=51, exception=2, max_tokens=8.
- _Validation overhead:_ 0.10s total (avg 0.00s across 61 model(s)).
- _Upstream prefill / first-token latency:_ Avg 0.88s | Min 0.02s | Max 5.05s
  across 59 model(s).

## 🏆 Performance Highlights

- **Fastest:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (544.0 tps)
- **💾 Most efficient:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (1.0 GB)
- **⚡ Fastest load:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (0.34s)
- **📊 Average TPS:** 95.4 across 59 models

## 📈 Resource Usage

- **Input image size:** 0.31 MP
- **Average peak delta from post-load:** 3.13 GB
- **Peak memory delta / MP:** 10445 MB/MP
- **Average peak memory:** 17.8 GB
- **Memory efficiency:** 45 tokens/GB

## ⚠️ Quality Issues

- **❌ Failed Models (2):**
  - `mlx-community/gemma-4-31b-bf16` (`Model Error`)
  - `mlx-community/gemma-4-31b-it-4bit` (`Model Error`)
- **🔄 Repetitive Output (1):**
  - `mlx-community/gemma-3n-E2B-4bit` (token: `phrase: "have this image. have..."`)
- **📝 Formatting Issues (1):**
  - `mlx-community/Idefics3-8B-Llama3-bf16`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 95.4 | Min: 4.67 | Max: 544
- **Peak Memory**: Avg: 18 | Min: 1.0 | Max: 71
- **Total Time**: Avg: 6.21s | Min: 0.41s | Max: 47.09s
- **Generation Time**: Avg: 4.25s | Min: 0.06s | Max: 43.88s
- **Model Load Time**: Avg: 1.95s | Min: 0.34s | Max: 8.41s

## Caption Selection

Triage mode suppresses cataloging and keyword scores. Brief-caption
recommendations are ungrounded unless descriptive image metadata is present;
use `model_selection.md` for the caption shortlist and `model_gallery.md` for
full output evidence.

## 🚨 Failures by Package (Actionable)

| Package           |   Failures | Error Types   | Affected Models                     |
|-------------------|------------|---------------|-------------------------------------|
| `mlx-vlm`         |          1 | Model Error   | `mlx-community/gemma-4-31b-bf16`    |
| `huggingface-hub` |          1 | Model Error   | `mlx-community/gemma-4-31b-it-4bit` |

### Actionable Items by Package

#### mlx-vlm

- mlx-community/gemma-4-31b-bf16 (Model Error)
  - Error: `Model runtime error during generation for mlx-community/gemma-4-31b-bf16: [METAL] Command buffer execution failed: In...`
  - Type: `ValueError`

#### huggingface-hub

- mlx-community/gemma-4-31b-it-4bit (Model Error)
  - Error: `Model loading failed: Operation timed out after 300.0 seconds`
  - Type: `ValueError`

_Prompt used:_

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Describe this image briefly.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 1305.18s

_Detailed machine-readable metrics remain in `results.tsv` and
`results.jsonl`; this Markdown table keeps the high-signal columns for human
review._

<!-- markdownlint-disable MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Gen TPS |   Peak (GB) | Finish Reason   |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                     |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|----------:|------------:|:----------------|-----------------:|-----------:|------------:|:-----------------------------------|----------------:|
| `mlx-community/gemma-4-31b-bf16`                        |                   |                       |                |           |             |                 |          159.27s |     17.62s |     176.90s | gibberish(char_noise)              |         mlx-vlm |
| `mlx-community/gemma-4-31b-it-4bit`                     |                   |                       |                |           |             |                 |                  |    752.72s |     752.73s |                                    | huggingface-hub |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      |                80 |                    10 |             90 |       544 |           1 | stop            |            0.06s |      0.34s |       0.41s |                                    |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               269 |                    10 |            279 |       314 |           3 | stop            |            0.12s |      0.52s |       0.64s |                                    |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |                26 |                    15 |             41 |       321 |         2.2 | stop            |            0.16s |      0.62s |       0.78s |                                    |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |               269 |                    13 |            282 |       194 |         4.1 | stop            |            0.16s |      0.54s |       0.71s |                                    |                 |
| `mlx-community/MiniCPM-V-4.6-8bit`                      |               228 |                    17 |            245 |       265 |           3 | stop            |            0.20s |      0.94s |       1.15s |                                    |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |                22 |                    81 |            103 |       384 |         1.9 | stop            |            0.29s |      0.48s |       0.77s |                                    |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |               417 |                    52 |            469 |       329 |         2.5 | stop            |            0.33s |      0.51s |       0.84s |                                    |                 |
| `qnguyen3/nanoLLaVA`                                    |                22 |                    35 |             57 |       116 |         4.1 | stop            |            0.41s |      0.60s |       1.01s |                                    |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,196 |                    13 |          1,209 |       126 |         5.5 | stop            |            0.53s |      0.70s |       1.23s |                                    |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |               770 |                    19 |            789 |      60.7 |         9.2 | stop            |            0.58s |      0.95s |       1.53s |                                    |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |               770 |                    19 |            789 |      60.8 |         9.2 | stop            |            0.59s |      0.91s |       1.50s |                                    |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             1,196 |                    13 |          1,209 |       124 |         5.5 | stop            |            0.64s |      0.67s |       1.31s |                                    |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |               973 |                    97 |          1,070 |       203 |         4.5 | stop            |            0.75s |      1.05s |       1.82s |                                    |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |               315 |                    89 |            404 |       138 |         5.2 | stop            |            0.77s |      0.67s |       1.45s |                                    |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |               408 |                    42 |            450 |      93.4 |         7.7 | stop            |            0.77s |      1.33s |       2.10s |                                    |                 |
| `mlx-community/Qwen3-VL-2B-Instruct-bf16`               |               315 |                    89 |            404 |       133 |         5.3 | stop            |            0.79s |      0.70s |       1.49s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |               319 |                    51 |            370 |       117 |          21 | stop            |            0.82s |      2.46s |       3.28s |                                    |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |               319 |                    75 |            394 |       101 |           7 | stop            |            1.02s |      1.37s |       2.40s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |               319 |                    61 |            380 |      98.1 |          30 | stop            |            1.08s |      3.09s |       4.18s |                                    |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,103 |                     3 |          4,106 |      85.8 |         4.6 | stop            |            1.11s |      1.18s |       2.30s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`         |               286 |                    16 |            302 |      21.1 |          28 | stop            |            1.22s |      3.14s |       4.37s |                                    |                 |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit`          |               286 |                    25 |            311 |        33 |          29 | stop            |            1.25s |      3.32s |       4.60s |                                    |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |               408 |                    58 |            466 |      64.2 |          10 | stop            |            1.27s |      1.43s |       2.71s |                                    |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |               417 |                    65 |            482 |      66.5 |          10 | stop            |            1.27s |      1.17s |       2.45s |                                    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |               974 |                    54 |          1,028 |        67 |          10 | stop            |            1.47s |      1.37s |       2.85s |                                    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |               974 |                    60 |          1,034 |      70.4 |         9.8 | stop            |            1.51s |      1.31s |       2.83s |                                    |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |               317 |                   188 |            505 |       134 |         5.3 | stop            |            1.52s |      0.73s |       2.26s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               286 |                    21 |            307 |      61.9 |          17 | stop            |            1.54s |      2.37s |       3.92s |                                    |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,327 |                    23 |          2,350 |      33.1 |          19 | stop            |            1.78s |      1.91s |       3.69s | formatting                         |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |                97 |                   200 |            297 |       130 |         5.5 | length          |            1.90s |      0.75s |       2.65s | token-cap                          |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               266 |                   200 |            466 |       122 |           6 | length          |            1.93s |      1.46s |       3.40s | repetitive(phrase: "have th...     |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |                16 |                    25 |             41 |      22.7 |          15 | stop            |            2.47s |      1.48s |       3.96s |                                    |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |               399 |                   167 |            566 |      80.1 |          16 | stop            |            2.49s |      1.93s |       4.42s | reasoning-leak                     |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |               319 |                    64 |            383 |      33.6 |          19 | stop            |            2.53s |      2.15s |       4.70s |                                    |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             3,341 |                    48 |          3,389 |      34.4 |          18 | stop            |            2.66s |      1.73s |       4.40s |                                    |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             2,356 |                    55 |          2,411 |      64.3 |         9.7 | stop            |            2.78s |      0.96s |       3.75s |                                    |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |               319 |                    41 |            360 |      19.5 |          30 | stop            |            2.80s |      3.07s |       5.90s |                                    |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               274 |                   124 |            398 |      47.8 |          17 | stop            |            2.94s |      2.26s |       5.21s |                                    |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             3,341 |                    20 |          3,361 |        33 |          19 | stop            |            3.01s |      1.64s |       4.66s |                                    |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |               399 |                   200 |            599 |      73.6 |          20 | length          |            3.15s |      2.17s |       5.33s | reasoning-leak, cutoff             |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               275 |                    91 |            366 |      31.8 |          18 | stop            |            3.59s |      2.22s |       5.82s |                                    |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |               441 |                    92 |            533 |      31.2 |          20 | stop            |            3.64s |      2.11s |       5.76s |                                    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,031 |                   101 |          1,132 |      36.1 |          11 | stop            |            3.77s |      1.59s |       5.38s |                                    |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |               319 |                    49 |            368 |      15.8 |          30 | stop            |            3.87s |      3.01s |       6.90s |                                    |                 |
| `mlx-community/Ornith-1.0-35B-bf16`                     |               319 |                    41 |            360 |      61.7 |          71 | stop            |            4.10s |      8.41s |      12.54s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |               319 |                    35 |            354 |        70 |          71 | stop            |            4.39s |      7.69s |      12.08s |                                    |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |               408 |                    91 |            499 |      51.6 |          63 | stop            |            4.41s |      5.74s |      10.17s |                                    |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |               439 |                   200 |            639 |      63.9 |          60 | length          |            4.99s |      4.90s |       9.90s | generation_loop(degeneration), ... |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               275 |                    70 |            345 |        17 |          32 | stop            |            5.01s |      3.34s |       8.36s |                                    |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             1,330 |                   200 |          1,530 |      44.2 |          14 | length          |            5.46s |      1.56s |       7.03s | reasoning-leak, cutoff             |                 |
| `mlx-community/pixtral-12b-8bit`                        |             1,239 |                   200 |          1,439 |      40.2 |          15 | length          |            5.70s |      1.69s |       7.40s | cutoff                             |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |               745 |                    55 |            800 |      52.2 |          20 | stop            |            5.95s |      1.29s |       7.25s |                                    |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,031 |                   119 |          1,150 |      19.8 |          10 | stop            |            6.39s |      1.47s |       7.87s |                                    |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |               745 |                    53 |            798 |      30.5 |          27 | stop            |            6.81s |      1.75s |       8.57s |                                    |                 |
| `mlx-community/pixtral-12b-bf16`                        |             1,239 |                   200 |          1,439 |        20 |          27 | length          |           10.80s |      2.57s |      13.37s | cutoff                             |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |                17 |                    77 |             94 |      5.08 |          25 | stop            |           16.90s |      2.15s |      19.06s |                                    |                 |
| `mlx-community/MolmoPoint-8B-fp16`                      |               766 |                   163 |            929 |      6.03 |          23 | stop            |           27.93s |      2.22s |      30.16s |                                    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,031 |                   159 |          1,190 |      5.42 |          26 | stop            |           30.46s |      2.46s |      32.93s |                                    |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |               399 |                   200 |            599 |      4.67 |          39 | length          |           43.88s |      3.20s |      47.09s | reasoning-leak, cutoff             |                 |

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
  sha256=a96d8dc798ee2a07ecd2a03f916dd5568d35fbd50cf296876dd893230fdf2391)

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

Report generated on: 2026-07-10 15:07:22 BST
