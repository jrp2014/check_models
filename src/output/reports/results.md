# Model Performance Results

_Generated on 2026-05-03 00:34:40 BST_

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ 2 (top owners: mlx=1, model-config=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=9, clean outputs=30/51.
- _Useful now:_ 30 clean A/B model(s) worth first review.
- _Review watchlist:_ 19 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Quality signal frequency:_ harness=9, cutoff=8, reasoning_leak=6,
  formatting=4, long_context=2, verbose=2.

### Runtime

- _Runtime pattern:_ decode dominates measured phase time (90%; 49/53 measured
  model(s)).
- _Phase totals:_ model load=118.52s, prompt prep=0.16s, decode=1185.07s,
  cleanup=5.93s.
- _What this likely means:_ Most measured runtime is spent inside generation
  rather than load or prompt setup.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=51, exception=2.
- _Validation overhead:_ 18.10s total (avg 0.34s across 53 model(s)).
- _First-token latency:_ Avg 13.51s | Min 0.06s | Max 87.00s across 51
  model(s).

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/nanoLLaVA-1.5-4bit` (356.6 tps)
- **💾 Most efficient:** `mlx-community/nanoLLaVA-1.5-4bit` (1.8 GB)
- **⚡ Fastest load:** `mlx-community/nanoLLaVA-1.5-4bit` (0.45s)
- **📊 Average TPS:** 75.7 across 51 models

## 📈 Resource Usage

- **Total peak memory:** 1067.7 GB
- **Average peak memory:** 20.9 GB
- **Memory efficiency:** 212 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 1 | ✅ B: 42 | 🟡 C: 3 | 🟠 D: 1 | ❌ F: 4

**Average Utility Score:** 69/100

- **Best for cataloging:** `mlx-community/gemma-3-27b-it-qat-4bit` (🏆 A, 85/100)
- **Best descriptions:** `qnguyen3/nanoLLaVA` (90/100)
- **Best keywording:** `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` (96/100)
- **Worst for cataloging:** `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` (❌ F, 0/100)

### ⚠️ 5 Models with Low Utility (D/F)

- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/gemma-3n-E2B-4bit`: ❌ F (5/100) - Output too short to be useful
- `mlx-community/gemma-4-31b-bf16`: ❌ F (5/100) - Output too short to be useful
- `mlx-community/paligemma2-3b-pt-896-4bit`: ❌ F (5/100) - Output too short to be useful

## ⚠️ Quality Issues

- **❌ Failed Models (2):**
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit` (`Model Error`)
  - `mlx-community/MolmoPoint-8B-fp16` (`Processor Error`)
- **🔄 Repetitive Output (1):**
  - `mlx-community/Qwen2-VL-2B-Instruct-4bit` (token: `phrase: "' chinese: ' chinese:..."`)
- **📝 Formatting Issues (4):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-Flash-mxfp4`
  - `mlx-community/GLM-4.6V-nvfp4`
  - `mlx-community/Idefics3-8B-Llama3-bf16`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 75.7 | Min: 4.71 | Max: 357
- **Peak Memory**: Avg: 21 | Min: 1.8 | Max: 78
- **Total Time**: Avg: 25.87s | Min: 1.73s | Max: 122.33s
- **Generation Time**: Avg: 23.24s | Min: 0.75s | Max: 118.57s
- **Model Load Time**: Avg: 2.28s | Min: 0.45s | Max: 13.09s

## ✅ Recommended Models

Quick picks based on end-to-end utility plus description and keyword strength.

- _Best end-to-end cataloging:_ [`mlx-community/gemma-3-27b-it-qat-4bit`](model_gallery.md#model-mlx-community-gemma-3-27b-it-qat-4bit)
  (Utility A 85/100 | Description 82 | Keywords 92 | Speed 29.5 TPS | Memory
  19 | Caveat nontext prompt burden=99%)
- _Best descriptions:_ [`qnguyen3/nanoLLaVA`](model_gallery.md#model-qnguyen3-nanollava)
  (Utility B 74/100 | Description 90 | Keywords 0 | Speed 115 TPS | Memory 3.9
  | Caveat nontext prompt burden=80%)
- _Best keywording:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
  (Utility B 75/100 | Description 82 | Keywords 96 | Speed 64.9 TPS | Memory
  13 | Caveat nontext prompt burden=100%)
- _Fastest generation:_ [`mlx-community/nanoLLaVA-1.5-4bit`](model_gallery.md#model-mlx-community-nanollava-15-4bit)
  (Utility B 75/100 | Description 86 | Keywords 0 | Speed 357 TPS | Memory 1.8
  | Caveat nontext prompt burden=80%)
- _Lowest memory footprint:_ [`mlx-community/nanoLLaVA-1.5-4bit`](model_gallery.md#model-mlx-community-nanollava-15-4bit)
  (Utility B 75/100 | Description 86 | Keywords 0 | Speed 357 TPS | Memory 1.8
  | Caveat nontext prompt burden=80%)
- _Best balance:_ [`mlx-community/gemma-3-27b-it-qat-4bit`](model_gallery.md#model-mlx-community-gemma-3-27b-it-qat-4bit)
  (Utility A 85/100 | Description 82 | Keywords 92 | Speed 29.5 TPS | Memory
  19 | Caveat nontext prompt burden=99%)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (2):_ [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](model_gallery.md#model-mlx-community-kimi-vl-a3b-thinking-8bit),
  [`mlx-community/MolmoPoint-8B-fp16`](model_gallery.md#model-mlx-community-molmopoint-8b-fp16).
  Example: `Model Error`.
- _🔄 Repetitive Output (1):_ [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](model_gallery.md#model-mlx-community-qwen2-vl-2b-instruct-4bit).
  Example: token: `phrase: "' chinese: ' chinese:..."`.
- _📝 Formatting Issues (4):_ [`mlx-community/GLM-4.6V-Flash-6bit`](model_gallery.md#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/GLM-4.6V-Flash-mxfp4`](model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4),
  [`mlx-community/GLM-4.6V-nvfp4`](model_gallery.md#model-mlx-community-glm-46v-nvfp4),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16).
- _Low-utility outputs (5):_ [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](model_gallery.md#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit),
  [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](model_gallery.md#model-mlx-community-qwen2-vl-2b-instruct-4bit),
  [`mlx-community/gemma-3n-E2B-4bit`](model_gallery.md#model-mlx-community-gemma-3n-e2b-4bit),
  [`mlx-community/gemma-4-31b-bf16`](model_gallery.md#model-mlx-community-gemma-4-31b-bf16),
  +1 more. Common weakness: Output too short to be useful.

## 🚨 Failures by Package (Actionable)

| Package | Failures | Error Types | Affected Models |
| --- | --- | --- | --- |
| `mlx` | 1 | Model Error | `mlx-community/Kimi-VL-A3B-Thinking-8bit` |
| `model-config` | 1 | Processor Error | `mlx-community/MolmoPoint-8B-fp16` |

### Actionable Items by Package

#### mlx

- mlx-community/Kimi-VL-A3B-Thinking-8bit (Model Error)
  - Error: `Model loading failed: Received 4 parameters not in model: <br>multi_modal_projector.linear_1.biases,<br>multi_modal_project...`
  - Type: `ValueError`

#### model-config

- mlx-community/MolmoPoint-8B-fp16 (Processor Error)
  - Error: `Model preflight failed for mlx-community/MolmoPoint-8B-fp16: Loaded processor has no image_processor; expected multim...`
  - Type: `ValueError`

_Prompt used:_

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Describe this picture
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 1330.96s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                    |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:----------------------------------|----------------:|
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |                   |                       |                |              |           |             |                  |      0.17s |       1.28s |                                   |             mlx |
| `mlx-community/MolmoPoint-8B-fp16`                      |                   |                       |                |              |           |             |                  |      2.16s |       3.53s |                                   |    model-config |
| `mlx-community/gemma-3n-E2B-4bit`                       |               264 |                     4 |            268 |         1135 |      92.2 |           6 |            0.75s |      1.47s |       2.57s | ⚠️harness(prompt_template)        |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |                20 |                   124 |            144 |          255 |       357 |         1.8 |            0.93s |      0.45s |       1.73s |                                   |                 |
| `qnguyen3/nanoLLaVA`                                    |                20 |                    51 |             71 |          174 |       115 |         3.9 |            1.06s |      0.52s |       1.91s |                                   |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |                85 |                   117 |            202 |         1445 |       188 |         3.5 |            1.08s |      0.52s |       1.93s |                                   |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               275 |                   292 |            567 |         3232 |       331 |           3 |            1.38s |      0.46s |       2.16s |                                   |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |               768 |                    53 |            821 |         3318 |      59.4 |         9.4 |            1.57s |      0.87s |       2.76s |                                   |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,101 |                     3 |          4,104 |         3659 |      81.6 |         4.6 |            1.72s |      1.27s |       3.35s | ⚠️harness(long_context), ...      |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |                24 |                   445 |            469 |          269 |       356 |         2.1 |            2.22s |      0.56s |       3.12s |                                   |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |                95 |                   266 |            361 |          291 |       129 |         5.5 |            3.05s |      0.58s |       3.97s |                                   |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,326 |                    48 |          2,374 |         2153 |      32.6 |          19 |            3.22s |      2.08s |       5.81s | formatting                        |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               278 |                   267 |            545 |          698 |       111 |          17 |            3.36s |      2.68s |       6.40s |                                   |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |             2,629 |                   355 |          2,984 |         2642 |       182 |         7.9 |            3.49s |      0.99s |       4.86s |                                   |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             2,174 |                    74 |          2,248 |          918 |      61.5 |         9.7 |            4.26s |      0.89s |       5.53s |                                   |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,195 |                   420 |          1,615 |         3069 |       122 |         5.5 |            4.41s |      0.60s |       5.32s |                                   |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             1,195 |                   420 |          1,615 |         3136 |       123 |         5.5 |            4.44s |      0.59s |       5.48s |                                   |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,029 |                   135 |          1,164 |         1315 |      32.8 |          11 |            5.54s |      1.64s |       7.53s |                                   |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               272 |                   237 |            509 |          842 |      48.1 |          17 |            5.73s |      2.19s |       8.26s |                                   |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             1,804 |                   125 |          1,929 |         1304 |      32.1 |          18 |            5.95s |      1.75s |       8.06s |                                   |                 |
| `mlx-community/pixtral-12b-8bit`                        |             2,887 |                   167 |          3,054 |         1730 |      39.2 |          16 |            6.46s |      2.28s |       9.09s |                                   |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |             1,011 |                   357 |          1,368 |         1241 |      69.9 |          18 |            6.50s |      2.17s |       9.02s | reasoning-leak                    |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               266 |                     5 |            271 |         46.9 |      9.51 |          64 |            6.78s |      8.82s |      15.94s | ⚠️harness(prompt_template)        |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,029 |                   118 |          1,147 |         2571 |      18.8 |          10 |            7.26s |      1.91s |       9.54s |                                   |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               278 |                   158 |            436 |          361 |      25.9 |          19 |            7.46s |      2.62s |      10.43s |                                   |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |             2,630 |                   311 |          2,941 |         1252 |      64.9 |          13 |            7.51s |      1.32s |       9.17s |                                   |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |             2,630 |                   344 |          2,974 |         1158 |      60.6 |          14 |            8.50s |      1.37s |      10.21s |                                   |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |             2,097 |                   172 |          2,269 |          696 |        31 |          22 |            9.12s |      2.30s |      11.77s | ⚠️harness(encoding), ...          |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |               768 |                   500 |          1,268 |         3305 |      57.1 |         9.3 |            9.45s |      1.12s |      10.90s | ⚠️harness(stop_token), ...        |                 |
| `mlx-community/pixtral-12b-bf16`                        |             2,887 |                   166 |          3,053 |         1540 |      20.1 |          28 |           10.66s |      2.51s |      13.50s |                                   |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             1,804 |                   344 |          2,148 |         2438 |      33.9 |          17 |           11.44s |      1.70s |      13.47s |                                   |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |             1,293 |                   500 |          1,793 |          454 |      59.6 |          60 |           11.99s |      6.00s |      18.49s | cutoff                            |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               273 |                   320 |            593 |          382 |      29.5 |          19 |           12.05s |      2.32s |      14.73s |                                   |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |             6,091 |                   500 |          6,591 |         1149 |      71.3 |         8.4 |           12.81s |      1.29s |      14.47s | ⚠️harness(stop_token), ...        |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             2,979 |                   500 |          3,479 |         1481 |      42.3 |          15 |           14.35s |      1.57s |      16.25s | degeneration, reasoning-leak, ... |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |             6,091 |                   500 |          6,591 |         1137 |      54.2 |          11 |           15.03s |      1.49s |      16.87s | ⚠️harness(stop_token), ...        |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |             1,207 |                   205 |          1,412 |         86.7 |        51 |          41 |           18.77s |      1.47s |      20.58s |                                   |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,029 |                   100 |          1,129 |          905 |      5.32 |          26 |           20.55s |      2.62s |      23.53s |                                   |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |                14 |                   421 |            435 |         10.2 |      20.1 |          15 |           22.80s |      1.51s |      24.64s |                                   |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |             1,207 |                   219 |          1,426 |           81 |      29.8 |          48 |           23.10s |      1.78s |      25.22s |                                   |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               273 |                   381 |            654 |          321 |      17.2 |          33 |           23.53s |      3.36s |      27.23s |                                   |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |            16,299 |                   219 |         16,518 |          863 |      55.7 |          14 |           23.60s |      1.12s |      25.06s |                                   |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |             6,091 |                   500 |          6,591 |          325 |      29.2 |          78 |           36.42s |      8.32s |      45.10s | ⚠️harness(stop_token), ...        |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |                15 |                   276 |            291 |         8.51 |      4.89 |          25 |           58.63s |      2.15s |      61.09s |                                   |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |             1,011 |                   277 |          1,288 |          760 |      4.71 |          39 |           60.89s |      3.23s |      64.45s | reasoning-leak                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |            16,290 |                   500 |         16,790 |          263 |      86.9 |          35 |           68.56s |      3.19s |      72.10s | cutoff                            |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |            16,290 |                   500 |         16,790 |          257 |       101 |          26 |           69.14s |      2.92s |      72.40s | cutoff                            |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |            16,290 |                   500 |         16,790 |          257 |      86.4 |          12 |           70.12s |      1.48s |      71.96s | degeneration, token-cap           |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |            16,299 |                   500 |         16,799 |          242 |       108 |         5.1 |           72.80s |      0.47s |      73.60s | ⚠️harness(long_context), ...      |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |            16,290 |                   500 |         16,790 |          233 |      59.3 |          76 |           79.53s |     13.09s |      92.96s | token-cap                         |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |            16,290 |                   500 |         16,790 |          211 |      27.7 |          26 |           96.07s |      2.17s |      98.63s | cutoff                            |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |            16,290 |                   500 |         16,790 |          206 |      16.4 |          38 |          110.45s |      3.03s |     113.84s | cutoff                            |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |            16,290 |                   500 |         16,790 |          187 |      16.3 |          38 |          118.57s |      3.40s |     122.33s | cutoff                            |                 |

<!-- markdownlint-enable MD013 MD033 MD034 MD037 MD049 -->

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](model_gallery.md)
- _Automated review digest:_ [review.md](review.md)
- _Canonical run log:_ [../check_models.log](../check_models.log)

---

## System/Hardware Information

- _OS:_ Darwin 25.4.0
- _macOS Version:_ 26.4.1
- _SDK Version:_ 26.4
- _Xcode Version:_ 26.4.1
- _Xcode Build:_ 17E202
- _Metal SDK:_ MacOSX.sdk
- _Python Version:_ 3.13.12
- _Architecture:_ arm64
- _GPU/Chip:_ Apple M5 Max
- _GPU Cores:_ 40
- _Metal Support:_ Metal 4
- _RAM:_ 128.0 GB
- _CPU Cores (Physical):_ 18
- _CPU Cores (Logical):_ 18

## Library Versions

- `numpy`: `2.4.4`
- `mlx`: `0.32.0.dev20260502+e8ebdebe`
- `mlx-vlm`: `0.4.5`
- `mlx-lm`: `0.31.3`
- `huggingface-hub`: `1.13.0`
- `transformers`: `5.7.0`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-05-03 00:34:40 BST_
