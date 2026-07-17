# Model Performance Results

Generated on: 2026-07-17 22:28:52 BST

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
- _Useful now:_ 32 model(s) shortlisted for caption review.
- _Review watchlist:_ 19 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Quality signal frequency:_ cutoff=7, formatting=3, harness=2,
  generation_loop=2, thinking_trace=2, context_budget=1.

### Runtime

- _Runtime pattern:_ input preparation + decode dominates measured phase time
  (52%; 19/61 measured model(s)).
- _Phase totals:_ model load=125.69s, local prompt prep=0.23s, upstream model
  prefill / first-token=53.88s, input preparation + decode=205.73s,
  cleanup=6.59s.
- _Generation total:_ 259.61s across 61 model(s); upstream model prefill /
  first-token split available for 61/61 model(s).
- _What this likely means:_ Most residual generation-call time falls outside
  the upstream model-loop first-token window, combining input preparation with
  token decoding.
- _Suggested next action:_ Profile prepare_inputs() and image preprocessing,
  then use generation TPS and token counts to assess the decode contribution.
- _Termination reasons:_ completed=53, max_tokens=8.
- _Validation overhead:_ 0.10s total (avg 0.00s across 61 model(s)).
- _Upstream model prefill / first-token time:_ Avg 0.88s | Min 0.02s | Max
  5.05s across 61 model(s).

## 🏆 Performance Highlights

- **Fastest:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (530.2 tps)
- **💾 Most efficient:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (1.0 GB)
- **⚡ Fastest load:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (0.31s)
- **📊 Average TPS:** 93.6 across 61 models

## 📈 Resource Usage

- **Input image size:** 0.31 MP
- **Average peak delta from post-load:** 2.02 GB
- **Peak memory delta / MP:** 6748 MB/MP
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

- **Generation Tps**: Avg: 93.6 | Min: 4.67 | Max: 530
- **Peak Memory**: Avg: 19 | Min: 1.0 | Max: 71
- **Total Time**: Avg: 6.33s | Min: 0.38s | Max: 47.14s
- **Generation Time**: Avg: 4.26s | Min: 0.06s | Max: 43.84s
- **Model Load Time**: Avg: 2.06s | Min: 0.31s | Max: 8.69s

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

_Overall runtime:_ 393.25s

_Detailed machine-readable metrics remain in `results.tsv` and
`results.jsonl`; this Markdown table keeps the high-signal columns for human
review._

<!-- markdownlint-disable MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Gen TPS |   Peak (GB) | Finish Reason   |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                     |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|----------:|------------:|:----------------|-----------------:|-----------:|------------:|:-----------------------------------|----------------:|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      |                80 |                    10 |             90 |       530 |           1 | stop            |            0.06s |      0.31s |       0.38s |                                    |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               269 |                    10 |            279 |       316 |           3 | stop            |            0.12s |      0.55s |       0.67s |                                    |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |               269 |                    13 |            282 |       193 |         4.1 | stop            |            0.16s |      0.56s |       0.73s |                                    |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |                26 |                    15 |             41 |       306 |         2.2 | stop            |            0.17s |      0.65s |       0.83s |                                    |                 |
| `mlx-community/MiniCPM-V-4.6-8bit`                      |               228 |                    17 |            245 |       268 |           3 | stop            |            0.20s |      0.85s |       1.06s |                                    |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |                22 |                    81 |            103 |       366 |         1.9 | stop            |            0.31s |      0.48s |       0.79s |                                    |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |               417 |                    52 |            469 |       320 |         2.5 | stop            |            0.34s |      0.54s |       0.88s |                                    |                 |
| `qnguyen3/nanoLLaVA`                                    |                22 |                    35 |             57 |       115 |           4 | stop            |            0.43s |      0.57s |       1.00s |                                    |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             1,196 |                    13 |          1,209 |       128 |         5.5 | stop            |            0.52s |      0.65s |       1.17s |                                    |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,196 |                    13 |          1,209 |       126 |         5.5 | stop            |            0.55s |      0.51s |       1.06s |                                    |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |               770 |                    19 |            789 |      61.6 |         9.2 | stop            |            0.58s |      0.96s |       1.55s |                                    |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |               770 |                    19 |            789 |      61.1 |         9.2 | stop            |            0.58s |      0.96s |       1.54s |                                    |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               286 |                    21 |            307 |       128 |          16 | stop            |            0.63s |      2.36s |       3.02s |                                    |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |               973 |                    97 |          1,070 |       203 |         4.5 | stop            |            0.75s |      0.90s |       1.66s |                                    |                 |
| `mlx-community/Qwen3-VL-2B-Instruct-bf16`               |               315 |                    89 |            404 |       135 |         5.3 | stop            |            0.78s |      0.71s |       1.49s |                                    |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |               315 |                    89 |            404 |       133 |         5.2 | stop            |            0.79s |      0.52s |       1.33s |                                    |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |               408 |                    42 |            450 |        88 |         7.7 | stop            |            0.80s |      1.36s |       2.16s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |               319 |                    51 |            370 |       117 |          21 | stop            |            0.82s |      2.49s |       3.31s |                                    |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |               319 |                    75 |            394 |       101 |           7 | stop            |            1.02s |      1.38s |       2.41s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |               319 |                    61 |            380 |      99.7 |          30 | stop            |            1.06s |      3.14s |       4.21s |                                    |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,103 |                     3 |          4,106 |      81.8 |         4.6 | stop            |            1.14s |      1.22s |       2.37s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`         |               282 |                    21 |            303 |      26.9 |          28 | stop            |            1.23s |      5.02s |       6.26s | formatting                         |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |               417 |                    65 |            482 |      66.5 |          10 | stop            |            1.27s |      1.13s |       2.41s |                                    |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |               408 |                    58 |            466 |        63 |          10 | stop            |            1.29s |      1.41s |       2.71s |                                    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |               974 |                    54 |          1,028 |      67.1 |          10 | stop            |            1.47s |      1.34s |       2.81s |                                    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |               974 |                    60 |          1,034 |      70.4 |         9.8 | stop            |            1.51s |      1.31s |       2.83s |                                    |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |               317 |                   188 |            505 |       135 |         5.3 | stop            |            1.51s |      0.71s |       2.23s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit`          |               282 |                    75 |            357 |      68.7 |          29 | stop            |            1.57s |      4.29s |       5.89s | formatting                         |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               286 |                    25 |            311 |      28.9 |          19 | stop            |            1.60s |      2.60s |       4.21s |                                    |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,327 |                    23 |          2,350 |      33.6 |          19 | stop            |            1.78s |      1.95s |       3.74s | formatting                         |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               266 |                   200 |            466 |       126 |           6 | length          |            1.83s |      1.55s |       3.39s | repetitive(phrase: "have th...     |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |                97 |                   200 |            297 |       129 |         5.5 | length          |            1.90s |      0.66s |       2.56s | token-cap                          |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |               399 |                   167 |            566 |        81 |          16 | stop            |            2.44s |      1.21s |       3.66s | thinking-trace                     |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |                16 |                    25 |             41 |      22.7 |          15 | stop            |            2.46s |      1.51s |       3.97s |                                    |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |               319 |                    64 |            383 |      33.2 |          19 | stop            |            2.54s |      2.12s |       4.68s |                                    |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             3,341 |                    48 |          3,389 |      34.2 |          18 | stop            |            2.69s |      1.60s |       4.30s |                                    |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             2,356 |                    55 |          2,411 |      63.8 |         9.7 | stop            |            2.77s |      0.91s |       3.69s |                                    |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |               319 |                    41 |            360 |      19.5 |          30 | stop            |            2.78s |      3.20s |       6.01s |                                    |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               274 |                   124 |            398 |      48.3 |          17 | stop            |            2.90s |      2.24s |       5.16s |                                    |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             3,341 |                    20 |          3,361 |      32.4 |          19 | stop            |            3.09s |      1.65s |       4.75s |                                    |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |               399 |                   200 |            599 |        74 |          20 | length          |            3.11s |      2.22s |       5.34s | thinking-incomplete, cutoff        |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |               319 |                    49 |            368 |      19.3 |          30 | stop            |            3.23s |      3.07s |       6.31s |                                    |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               275 |                    91 |            366 |      31.7 |          18 | stop            |            3.59s |      2.23s |       5.83s |                                    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,031 |                   101 |          1,132 |      34.5 |          11 | stop            |            3.79s |      1.63s |       5.44s |                                    |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |               441 |                    92 |            533 |      29.9 |          20 | stop            |            3.80s |      2.16s |       5.97s |                                    |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |               408 |                    91 |            499 |      49.6 |          63 | stop            |            4.26s |      5.81s |      10.08s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |               319 |                    35 |            354 |      69.9 |          71 | stop            |            4.43s |      7.84s |      12.28s |                                    |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               275 |                    70 |            345 |      17.9 |          32 | stop            |            4.77s |      3.35s |       8.13s |                                    |                 |
| `mlx-community/Ornith-1.0-35B-bf16`                     |               319 |                    41 |            360 |      60.7 |          71 | stop            |            4.79s |      8.69s |      13.50s |                                    |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |               439 |                   200 |            639 |      60.3 |          60 | length          |            4.88s |      5.17s |      10.06s | generation_loop(degeneration), ... |                 |
| `mlx-community/pixtral-12b-8bit`                        |             1,239 |                   200 |          1,439 |      40.2 |          15 | length          |            5.74s |      1.65s |       7.40s | cutoff                             |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |               745 |                    53 |            798 |      52.7 |          20 | stop            |            5.87s |      1.19s |       7.07s |                                    |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             1,330 |                   200 |          1,530 |      41.1 |          14 | length          |            5.87s |      1.61s |       7.50s | reasoning-leak, cutoff             |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,031 |                   119 |          1,150 |      19.7 |          10 | stop            |            6.43s |      1.46s |       7.90s |                                    |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |               745 |                    55 |            800 |      30.4 |          27 | stop            |            6.89s |      1.75s |       8.64s |                                    |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               274 |                    36 |            310 |      7.47 |          63 | stop            |            7.22s |      6.19s |      13.43s |                                    |                 |
| `mlx-community/pixtral-12b-bf16`                        |             1,239 |                   200 |          1,439 |      19.9 |          27 | length          |           10.88s |      2.56s |      13.44s | cutoff                             |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |                17 |                    77 |             94 |       5.1 |          25 | stop            |           16.77s |      1.02s |      17.80s |                                    |                 |
| `mlx-community/MolmoPoint-8B-fp16`                      |               766 |                   163 |            929 |      5.91 |          23 | stop            |           28.49s |      2.25s |      30.75s |                                    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,031 |                   159 |          1,190 |      5.41 |          26 | stop            |           30.58s |      2.49s |      33.08s |                                    |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |               399 |                   200 |            599 |      4.67 |          39 | length          |           43.84s |      3.30s |      47.14s | thinking-trace, cutoff             |                 |

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
  sha256=98e311dc5a6588305bef55d6f231605e3591120df70183b6cec2cf2d424d8362)

## Library Versions

- `numpy`: `2.5.1`
- `mlx`: `0.32.1.dev20260717+b7c3dd6d`
- `mlx-vlm`: `0.6.5`
- `mlx-lm`: `0.31.3`
- `mlx-audio`: `0.4.4`
- `huggingface-hub`: `1.24.0`
- `transformers`: `5.14.1`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.3.0`

Report generated on: 2026-07-17 22:28:52 BST
