# Model Performance Results

Generated on: 2026-07-18 19:23:21 BST

## Run Contract

- Evaluation lane: triage
- Metadata exposed to prompt: no
- Semantic rankings: ungrounded (caption hygiene only)
- Primary selection tasks: brief captions; structured title/description/keywords when descriptive image metadata is available

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ 6 (top owners: mlx-vlm=6).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=2, mechanically clean
  outputs=42/55.
- _Useful now:_ 28 model(s) shortlisted for caption review.
- _Review watchlist:_ 17 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Quality signal frequency:_ cutoff=7, formatting=3, harness=2,
  generation_loop=2, thinking_trace=2, context_budget=1.

### Runtime

- _Runtime pattern:_ model load dominates measured phase time (74%; 51/61
  measured model(s)).
- _Phase totals:_ model load=767.61s, local prompt prep=0.22s, upstream model
  prefill / first-token=63.63s, input preparation + decode=202.17s,
  cleanup=6.58s.
- _Generation total:_ 265.80s across 55 model(s); upstream model prefill /
  first-token split available for 55/55 model(s).
- _What this likely means:_ Cold model load time is a major share of runtime
  for this cohort.
- _Suggested next action:_ Consider staged runs, model reuse, or narrowing the
  model set before reruns.
- _Termination reasons:_ completed=48, exception=6, max_tokens=7.
- _Validation overhead:_ 0.11s total (avg 0.00s across 61 model(s)).
- _Upstream model prefill / first-token time:_ Avg 1.16s | Min 0.03s | Max
  6.32s across 55 model(s).

## 🏆 Performance Highlights

- **Fastest:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (504.3 tps)
- **💾 Most efficient:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (1.0 GB)
- **⚡ Fastest load:** `qnguyen3/nanoLLaVA` (0.71s)
- **📊 Average TPS:** 88.4 across 55 models

## 📈 Resource Usage

- **Input image size:** 0.31 MP
- **Average peak delta from post-load:** 2.12 GB
- **Peak memory delta / MP:** 7057 MB/MP
- **Average peak memory:** 19.5 GB
- **Memory efficiency:** 42 tokens/GB

## ⚠️ Quality Issues

- **❌ Failed Models (6):**
  - `mlx-community/FastVLM-0.5B-bf16` (`Model Error`)
  - `mlx-community/GLM-4.6V-Flash-6bit` (`Model Error`)
  - `mlx-community/Qwen3.5-9B-MLX-4bit` (`Model Error`)
  - `mlx-community/Qwen3.6-27B-mxfp8` (`Model Error`)
  - `mlx-community/SmolVLM-Instruct-bf16` (`Model Error`)
  - `mlx-community/SmolVLM2-2.2B-Instruct-mlx` (`Model Error`)
- **🔄 Repetitive Output (1):**
  - `mlx-community/gemma-3n-E2B-4bit` (token: `phrase: "have this image. have..."`)
- **📝 Formatting Issues (3):**
  - `mlx-community/Idefics3-8B-Llama3-bf16`
  - `mlx-community/diffusiongemma-26B-A4B-it-8bit`
  - `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 88.4 | Min: 4.53 | Max: 504
- **Peak Memory**: Avg: 19 | Min: 1.0 | Max: 71
- **Total Time**: Avg: 14.78s | Min: 1.14s | Max: 69.24s
- **Generation Time**: Avg: 4.83s | Min: 0.07s | Max: 45.27s
- **Model Load Time**: Avg: 9.93s | Min: 0.71s | Max: 53.49s

## Caption Selection

Triage mode suppresses cataloging and keyword scores. Brief-caption
recommendations are ungrounded unless descriptive image metadata is present;
use `model_selection.md` for the caption shortlist and `model_gallery.md` for
full output evidence.

## 🚨 Failures by Package (Actionable)

| Package   |   Failures | Error Types   | Affected Models                                                                                                                                                                                                                   |
|-----------|------------|---------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-vlm` |          6 | Model Error   | `mlx-community/FastVLM-0.5B-bf16`, `mlx-community/GLM-4.6V-Flash-6bit`, `mlx-community/Qwen3.5-9B-MLX-4bit`, `mlx-community/Qwen3.6-27B-mxfp8`, `mlx-community/SmolVLM-Instruct-bf16`, `mlx-community/SmolVLM2-2.2B-Instruct-mlx` |

### Actionable Items by Package

#### mlx-vlm

- mlx-community/FastVLM-0.5B-bf16 (Model Error)
  - Error: `RemoteProtocolError: Server disconnected without sending a response.`
  - Suspected owner: `mlx-vlm` (medium)
- mlx-community/GLM-4.6V-Flash-6bit (Model Error)
  - Error: `RemoteProtocolError: Server disconnected without sending a response.`
  - Suspected owner: `mlx-vlm` (medium)
- mlx-community/Qwen3.5-9B-MLX-4bit (Model Error)
  - Error: `RemoteProtocolError: Server disconnected without sending a response.`
  - Suspected owner: `mlx-vlm` (medium)
- mlx-community/Qwen3.6-27B-mxfp8 (Model Error)
  - Error: `RemoteProtocolError: Server disconnected without sending a response.`
  - Suspected owner: `mlx-vlm` (medium)
- mlx-community/SmolVLM-Instruct-bf16 (Model Error)
  - Error: `RemoteProtocolError: Server disconnected without sending a response.`
  - Suspected owner: `mlx-vlm` (medium)
- mlx-community/SmolVLM2-2.2B-Instruct-mlx (Model Error)
  - Error: `RemoteProtocolError: Server disconnected without sending a response.`
  - Suspected owner: `mlx-vlm` (medium)

_Prompt used:_

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Describe this image briefly.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 1041.43s

_Detailed machine-readable metrics remain in `results.tsv` and
`results.jsonl`; this Markdown table keeps the high-signal columns for human
review._

<!-- markdownlint-disable MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Gen TPS |   Peak (GB) | Finish Reason   |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                     |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|----------:|------------:|:----------------|-----------------:|-----------:|------------:|:-----------------------------------|----------------:|
| `mlx-community/FastVLM-0.5B-bf16`                       |                   |                       |                |           |             |                 |                  |     33.13s |      33.13s |                                    |         mlx-vlm |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |                   |                       |                |           |             |                 |                  |     30.54s |      30.54s |                                    |         mlx-vlm |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |                   |                       |                |           |             |                 |                  |     45.00s |      45.00s |                                    |         mlx-vlm |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |                   |                       |                |           |             |                 |                  |     30.45s |      30.46s |                                    |         mlx-vlm |
| `mlx-community/SmolVLM-Instruct-bf16`                   |                   |                       |                |           |             |                 |                  |     51.47s |      51.47s |                                    |         mlx-vlm |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |                   |                       |                |           |             |                 |                  |     30.70s |      30.70s |                                    |         mlx-vlm |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      |                80 |                    10 |             90 |       504 |           1 | stop            |            0.07s |     31.06s |      31.14s |                                    |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               269 |                    10 |            279 |       313 |           3 | stop            |            0.13s |      4.67s |       4.80s |                                    |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |               269 |                    13 |            282 |       188 |         4.1 | stop            |            0.16s |      2.36s |       2.53s |                                    |                 |
| `mlx-community/MiniCPM-V-4.6-8bit`                      |               228 |                    17 |            245 |       259 |           3 | stop            |            0.21s |      2.62s |       2.84s |                                    |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |                22 |                    81 |            103 |       369 |         1.8 | stop            |            0.31s |      0.93s |       1.24s |                                    |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |               417 |                    52 |            469 |       325 |         2.5 | stop            |            0.33s |      1.68s |       2.02s |                                    |                 |
| `qnguyen3/nanoLLaVA`                                    |                22 |                    35 |             57 |       116 |           4 | stop            |            0.42s |      0.71s |       1.14s |                                    |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               286 |                    21 |            307 |       128 |          16 | stop            |            0.56s |      5.39s |       5.98s |                                    |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |               770 |                    19 |            789 |      60.4 |         9.2 | stop            |            0.58s |      5.27s |       5.86s |                                    |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |               770 |                    19 |            789 |      60.7 |         9.2 | stop            |            0.60s |     14.44s |      15.04s |                                    |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,196 |                    13 |          1,209 |       121 |         5.5 | stop            |            0.62s |     53.49s |      54.11s |                                    |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |               973 |                    97 |          1,070 |       196 |         4.5 | stop            |            0.77s |      1.82s |       2.60s |                                    |                 |
| `mlx-community/Qwen3-VL-2B-Instruct-bf16`               |               315 |                    89 |            404 |       135 |         5.3 | stop            |            0.78s |      3.32s |       4.10s |                                    |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |               315 |                    89 |            404 |       134 |         5.1 | stop            |            0.79s |     46.87s |      47.69s |                                    |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |               408 |                    42 |            450 |      89.3 |         7.7 | stop            |            0.81s |      4.65s |       5.47s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |               319 |                    51 |            370 |       117 |          21 | stop            |            0.82s |      7.96s |       8.79s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |               319 |                    61 |            380 |      98.5 |          30 | stop            |            1.07s |      4.87s |       5.95s |                                    |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,103 |                     3 |          4,106 |      82.3 |         4.6 | stop            |            1.15s |      1.44s |       2.60s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |               417 |                    65 |            482 |      65.7 |          10 | stop            |            1.28s |     25.52s |      26.81s |                                    |                 |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit`          |               282 |                    25 |            307 |      27.9 |          29 | stop            |            1.35s |      5.10s |       6.48s | formatting                         |                 |
| `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`         |               282 |                    23 |            305 |      24.9 |          28 | stop            |            1.38s |      5.06s |       6.45s | formatting                         |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |               317 |                   188 |            505 |       135 |         5.3 | stop            |            1.51s |      2.98s |       4.51s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |               974 |                    54 |          1,028 |      64.1 |          10 | stop            |            1.53s |      1.61s |       3.15s |                                    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |               974 |                    60 |          1,034 |      67.5 |         9.8 | stop            |            1.57s |      1.86s |       3.44s |                                    |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               286 |                    25 |            311 |      28.7 |          19 | stop            |            1.61s |      3.69s |       5.31s |                                    |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,327 |                    23 |          2,350 |      34.2 |          19 | stop            |            1.73s |     31.33s |      33.07s | formatting                         |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               266 |                   200 |            466 |       124 |           6 | length          |            1.86s |     12.59s |      14.46s | repetitive(phrase: "have th...     |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |               319 |                    64 |            383 |      33.7 |          19 | stop            |            2.52s |      3.98s |       6.52s |                                    |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |                16 |                    25 |             41 |      22.1 |          15 | stop            |            2.55s |      2.92s |       5.48s |                                    |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             3,341 |                    48 |          3,389 |      34.5 |          18 | stop            |            2.66s |     19.66s |      22.33s |                                    |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |               399 |                   167 |            566 |        79 |          15 | stop            |            2.76s |     21.81s |      24.57s | thinking-trace                     |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             2,356 |                    55 |          2,411 |      63.3 |         9.7 | stop            |            2.79s |      1.39s |       4.19s |                                    |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |               319 |                    41 |            360 |      19.6 |          30 | stop            |            2.82s |      5.19s |       8.03s |                                    |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               274 |                   124 |            398 |      48.6 |          17 | stop            |            2.89s |      3.19s |       6.09s |                                    |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             3,341 |                    20 |          3,361 |      33.1 |          19 | stop            |            2.97s |     11.77s |      14.75s |                                    |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |               399 |                   200 |            599 |      69.1 |          20 | length          |            3.33s |     22.32s |      25.66s | thinking-incomplete, cutoff        |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               275 |                    91 |            366 |      31.8 |          18 | stop            |            3.57s |      3.90s |       7.48s |                                    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,031 |                   101 |          1,132 |      34.9 |          11 | stop            |            3.76s |      2.24s |       6.00s |                                    |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |               441 |                    92 |            533 |      29.9 |          20 | stop            |            3.81s |     16.13s |      19.95s |                                    |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |               745 |                    55 |            800 |      51.5 |          20 | stop            |            4.59s |      1.45s |       6.06s |                                    |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               275 |                    70 |            345 |      17.9 |          32 | stop            |            4.77s |      7.43s |      12.21s |                                    |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             1,330 |                   200 |          1,530 |      44.3 |          14 | length          |            5.45s |      5.58s |      11.05s | reasoning-leak, cutoff             |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |               319 |                    35 |            354 |        70 |          71 | stop            |            5.46s |     10.47s |      15.94s |                                    |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |               745 |                    53 |            798 |      30.1 |          27 | stop            |            5.64s |      8.10s |      13.75s |                                    |                 |
| `mlx-community/pixtral-12b-8bit`                        |             1,239 |                   200 |          1,439 |      40.1 |          15 | length          |            5.78s |      1.72s |       7.50s | cutoff                             |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,031 |                   119 |          1,150 |      19.6 |          10 | stop            |            6.47s |      2.46s |       8.94s |                                    |                 |
| `mlx-community/Ornith-1.0-35B-bf16`                     |               319 |                    41 |            360 |      60.6 |          71 | stop            |            7.01s |     13.61s |      20.64s |                                    |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |               408 |                    91 |            499 |      49.9 |          63 | stop            |            7.61s |     10.41s |      18.03s |                                    |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |               439 |                   200 |            639 |      61.6 |          60 | length          |            8.13s |     24.23s |      32.37s | generation_loop(degeneration), ... |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               274 |                    36 |            310 |      7.58 |          63 | stop            |            9.85s |      9.18s |      19.04s |                                    |                 |
| `mlx-community/pixtral-12b-bf16`                        |             1,239 |                   200 |          1,439 |      19.7 |          27 | length          |           10.97s |      3.03s |      14.01s | cutoff                             |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |                17 |                    77 |             94 |      5.08 |          25 | stop            |           17.73s |      6.26s |      24.00s |                                    |                 |
| `mlx-community/MolmoPoint-8B-fp16`                      |               766 |                   163 |            929 |      5.65 |          23 | stop            |           29.88s |     17.96s |      47.86s |                                    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,031 |                   159 |          1,190 |      5.37 |          26 | stop            |           30.79s |      2.69s |      33.49s |                                    |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |               399 |                   200 |            599 |      4.53 |          39 | length          |           45.27s |     23.96s |      69.24s | thinking-trace, cutoff             |                 |

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
- _MLX Distribution Root:_ ~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages
- _mlx-metal Distribution:_ not installed; local editable mlx supplies backend
- _MLX Core Extension:_ ~/Documents/AI/mlx/mlx/python/mlx/core.cpython-313-darwin.so
- _MLX Metallib:_ ~/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib
  (162,449,848 bytes,
  sha256=1078bd042297dbbf704a414617a7988c55b0001ea69d7cb478bcafa2fdfdeecb)
- _MLX libmlx.dylib:_ ~/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib
  (21,697,568 bytes,
  sha256=c03f8b1be065c164ed6c9380cc87b205f1c11c0e5b0cda028984159c57c79a3b)

## Library Versions

- `numpy`: `2.5.1`
- `mlx`: `0.32.1.dev20260718+b7c3dd6d`
- `mlx-vlm`: `0.6.5`
- `mlx-lm`: `0.31.3`
- `mlx-audio`: `0.4.5`
- `huggingface-hub`: `1.24.0`
- `transformers`: `5.14.1`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.3.0`

Report generated on: 2026-07-18 19:23:21 BST
