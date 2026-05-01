# Model Performance Results

_Generated on 2026-05-01 22:38:46 BST_

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ 2 (top owners: mlx=1, model-config=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=9, clean outputs=30/51.
- _Useful now:_ 30 clean A/B model(s) worth first review.
- _Review watchlist:_ 19 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Quality signal frequency:_ harness=9, cutoff=7, reasoning_leak=6,
  formatting=4, context_budget=2, long_context=2.

### Runtime

- _Runtime pattern:_ decode dominates measured phase time (89%; 49/53 measured
  model(s)).
- _Phase totals:_ model load=131.87s, prompt prep=0.16s, decode=1149.28s,
  cleanup=5.57s.
- _What this likely means:_ Most measured runtime is spent inside generation
  rather than load or prompt setup.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=51, exception=2.
- _Validation overhead:_ 17.40s total (avg 0.33s across 53 model(s)).
- _First-token latency:_ Avg 13.09s | Min 0.06s | Max 83.41s across 51
  model(s).

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/FastVLM-0.5B-bf16` (359.9 tps)
- **💾 Most efficient:** `mlx-community/nanoLLaVA-1.5-4bit` (1.8 GB)
- **⚡ Fastest load:** `mlx-community/SmolVLM-Instruct-bf16` (0.63s)
- **📊 Average TPS:** 78.1 across 51 models

## 📈 Resource Usage

- **Total peak memory:** 1067.6 GB
- **Average peak memory:** 20.9 GB
- **Memory efficiency:** 211 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 1 | ✅ B: 42 | 🟡 C: 3 | ❌ F: 5

**Average Utility Score:** 69/100

- **Best for cataloging:** `mlx-community/gemma-3-27b-it-qat-4bit` (🏆 A, 85/100)
- **Best descriptions:** `qnguyen3/nanoLLaVA` (90/100)
- **Best keywording:** `mlx-community/GLM-4.6V-Flash-mxfp4` (100/100)
- **Worst for cataloging:** `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` (❌ F, 0/100)

### ⚠️ 5 Models with Low Utility (D/F)

- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (16/100) - Output lacks detail
- `mlx-community/gemma-3n-E2B-4bit`: ❌ F (5/100) - Output too short to be useful
- `mlx-community/gemma-4-31b-bf16`: ❌ F (5/100) - Output too short to be useful
- `mlx-community/paligemma2-3b-pt-896-4bit`: ❌ F (5/100) - Output too short to be useful

## ⚠️ Quality Issues

- **❌ Failed Models (2):**
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit` (`Model Error`)
  - `mlx-community/MolmoPoint-8B-fp16` (`Processor Error`)
- **📝 Formatting Issues (4):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-Flash-mxfp4`
  - `mlx-community/GLM-4.6V-nvfp4`
  - `mlx-community/Idefics3-8B-Llama3-bf16`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 78.1 | Min: 4.72 | Max: 360
- **Peak Memory**: Avg: 21 | Min: 1.8 | Max: 78
- **Total Time**: Avg: 25.36s | Min: 2.02s | Max: 117.59s
- **Generation Time**: Avg: 22.53s | Min: 0.73s | Max: 113.66s
- **Model Load Time**: Avg: 2.49s | Min: 0.63s | Max: 13.03s

## ✅ Recommended Models

Quick picks based on end-to-end utility plus description and keyword strength.

- _Best end-to-end cataloging:_ [`mlx-community/gemma-3-27b-it-qat-4bit`](model_gallery.md#model-mlx-community-gemma-3-27b-it-qat-4bit)
  (Utility A 85/100 | Description 82 | Keywords 92 | Speed 26.0 TPS | Memory
  19 | Caveat nontext prompt burden=99%)
- _Best descriptions:_ [`qnguyen3/nanoLLaVA`](model_gallery.md#model-qnguyen3-nanollava)
  (Utility B 74/100 | Description 90 | Keywords 0 | Speed 114 TPS | Memory 3.8
  | Caveat nontext prompt burden=80%)
- _Best keywording:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
  (Utility B 75/100 | Description 82 | Keywords 96 | Speed 67.1 TPS | Memory
  13 | Caveat nontext prompt burden=100%)
- _Fastest generation:_ [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16)
  (Utility B 74/100 | Description 82 | Keywords 0 | Speed 360 TPS | Memory 2.1
  | Caveat nontext prompt burden=83%)
- _Lowest memory footprint:_ [`mlx-community/nanoLLaVA-1.5-4bit`](model_gallery.md#model-mlx-community-nanollava-15-4bit)
  (Utility B 75/100 | Description 86 | Keywords 0 | Speed 335 TPS | Memory 1.8
  | Caveat nontext prompt burden=80%)
- _Best balance:_ [`mlx-community/gemma-3-27b-it-qat-4bit`](model_gallery.md#model-mlx-community-gemma-3-27b-it-qat-4bit)
  (Utility A 85/100 | Description 82 | Keywords 92 | Speed 26.0 TPS | Memory
  19 | Caveat nontext prompt burden=99%)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (2):_ [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](model_gallery.md#model-mlx-community-kimi-vl-a3b-thinking-8bit),
  [`mlx-community/MolmoPoint-8B-fp16`](model_gallery.md#model-mlx-community-molmopoint-8b-fp16).
  Example: `Model Error`.
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

_Overall runtime:_ 1305.08s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                    |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:----------------------------------|----------------:|
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |                   |                       |                |              |           |             |                  |      2.56s |       2.88s |                                   |             mlx |
| `mlx-community/MolmoPoint-8B-fp16`                      |                   |                       |                |              |           |             |                  |      2.34s |       2.67s |                                   |    model-config |
| `mlx-community/gemma-3n-E2B-4bit`                       |               264 |                     4 |            268 |         1158 |      95.3 |           6 |            0.73s |      2.28s |       3.34s | ⚠️harness(prompt_template)        |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |                20 |                   124 |            144 |          253 |       335 |         1.8 |            0.97s |      0.73s |       2.02s |                                   |                 |
| `qnguyen3/nanoLLaVA`                                    |                20 |                    51 |             71 |          175 |       114 |         3.8 |            1.06s |      0.95s |       2.34s |                                   |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |                85 |                   117 |            202 |         1447 |       188 |         3.6 |            1.08s |      0.78s |       2.18s |                                   |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               275 |                   292 |            567 |         3363 |       331 |           3 |            1.37s |      0.71s |       2.40s |                                   |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |               768 |                    53 |            821 |         3477 |      59.2 |         9.4 |            1.57s |      1.02s |       2.91s |                                   |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,101 |                     3 |          4,104 |         3657 |      80.3 |         4.6 |            1.70s |      1.15s |       3.18s | ⚠️harness(long_context), ...      |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |                24 |                   445 |            469 |          273 |       360 |         2.1 |            2.27s |      1.06s |       3.68s |                                   |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |                95 |                   266 |            361 |          308 |       129 |         5.5 |            3.00s |      0.68s |       4.01s |                                   |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,326 |                    48 |          2,374 |         2268 |      33.2 |          19 |            3.08s |      2.00s |       5.42s | formatting                        |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               278 |                   267 |            545 |          708 |       111 |          17 |            3.29s |      2.68s |       6.32s |                                   |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |             2,629 |                   355 |          2,984 |         2682 |       187 |         7.9 |            3.42s |      1.55s |       5.32s |                                   |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             2,174 |                    74 |          2,248 |          948 |      61.9 |         9.7 |            4.15s |      1.16s |       5.65s |                                   |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,195 |                   420 |          1,615 |         3079 |       128 |         5.5 |            4.25s |      0.71s |       5.27s |                                   |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               266 |                     5 |            271 |         81.4 |      8.99 |          64 |            4.39s |      7.23s |      11.96s | ⚠️harness(prompt_template)        |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             1,195 |                   420 |          1,615 |         3177 |       123 |         5.5 |            4.41s |      0.63s |       5.38s |                                   |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,029 |                   135 |          1,164 |         1321 |      34.6 |          11 |            5.31s |      1.80s |       7.44s |                                   |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               272 |                   237 |            509 |          876 |      48.8 |          17 |            5.63s |      2.35s |       8.31s |                                   |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             1,804 |                   125 |          1,929 |         1343 |      32.1 |          18 |            5.80s |      2.27s |       8.40s |                                   |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |             1,011 |                   357 |          1,368 |         1275 |      74.3 |          18 |            6.13s |      2.56s |       9.00s | reasoning-leak                    |                 |
| `mlx-community/pixtral-12b-8bit`                        |             2,887 |                   167 |          3,054 |         1774 |      39.3 |          16 |            6.40s |      1.86s |       8.59s |                                   |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,029 |                   118 |          1,147 |         2784 |      19.7 |          10 |            6.86s |      2.11s |       9.30s |                                   |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               278 |                   158 |            436 |          381 |      26.9 |          19 |            7.11s |      2.70s |      10.16s |                                   |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |             2,630 |                   311 |          2,941 |         1347 |      67.1 |          13 |            7.12s |      1.76s |       9.22s |                                   |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |             2,630 |                   344 |          2,974 |         1142 |        63 |          14 |            8.33s |      1.90s |      10.56s |                                   |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |             2,097 |                   172 |          2,269 |          662 |      31.3 |          22 |            9.22s |      2.52s |      12.11s | ⚠️harness(encoding), ...          |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |               768 |                   500 |          1,268 |         3303 |        57 |         9.3 |            9.46s |      1.13s |      10.92s | ⚠️harness(stop_token), ...        |                 |
| `mlx-community/pixtral-12b-bf16`                        |             2,887 |                   166 |          3,053 |         1792 |      19.9 |          28 |           10.47s |      2.59s |      13.38s |                                   |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             1,804 |                   344 |          2,148 |         2475 |      34.8 |          17 |           11.17s |      2.31s |      13.81s |                                   |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |             1,293 |                   500 |          1,793 |          531 |      59.4 |          60 |           11.60s |      6.22s |      18.18s | cutoff                            |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |             6,091 |                   500 |          6,591 |         1140 |      72.4 |         8.4 |           12.72s |      1.56s |      14.63s | ⚠️harness(stop_token), ...        |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               273 |                   320 |            593 |          385 |        26 |          19 |           13.52s |      2.72s |      16.60s |                                   |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             2,979 |                   500 |          3,479 |         1448 |      41.9 |          15 |           14.52s |      1.82s |      16.68s | degeneration, reasoning-leak, ... |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |             6,091 |                   500 |          6,591 |         1145 |      54.9 |          11 |           14.89s |      1.53s |      16.77s | ⚠️harness(stop_token), ...        |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |             1,207 |                   205 |          1,412 |         86.8 |      52.1 |          41 |           18.68s |      1.25s |      20.27s |                                   |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,029 |                   100 |          1,129 |          926 |      5.44 |          26 |           20.05s |      2.84s |      23.23s |                                   |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |                14 |                   421 |            435 |         10.4 |      21.1 |          15 |           21.78s |      1.64s |      23.74s |                                   |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |             1,207 |                   205 |          1,412 |         81.7 |      30.1 |          48 |           22.41s |      1.95s |      24.70s |                                   |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |            16,299 |                   219 |         16,518 |          848 |      54.8 |          14 |           23.97s |      1.15s |      25.46s |                                   |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               273 |                   381 |            654 |          325 |      16.6 |          33 |           24.25s |      3.53s |      28.13s |                                   |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |             6,091 |                   500 |          6,591 |          426 |      36.1 |          78 |           28.63s |      6.94s |      35.91s | ⚠️harness(stop_token), ...        |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |                15 |                   276 |            291 |         8.95 |      4.98 |          25 |           57.54s |      2.52s |      60.39s |                                   |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |             1,011 |                   277 |          1,288 |          760 |      4.72 |          39 |           60.68s |      3.83s |      64.84s | reasoning-leak                    |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |            16,299 |                    13 |         16,312 |          251 |       207 |         5.1 |           65.68s |      0.69s |      66.69s | ⚠️harness(long_context), ...      |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |            16,290 |                   500 |         16,790 |          271 |       104 |          26 |           65.68s |      3.34s |      69.37s | cutoff                            |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |            16,290 |                   500 |         16,790 |          274 |        88 |          35 |           65.84s |      3.14s |      69.32s | cutoff                            |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |            16,290 |                   500 |         16,790 |          250 |      84.5 |          12 |           71.90s |      2.20s |      74.44s | degeneration, token-cap           |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |            16,290 |                   500 |         16,790 |          233 |      61.2 |          76 |           79.26s |     13.03s |      92.62s | token-cap                         |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |            16,290 |                   500 |         16,790 |          227 |      29.5 |          26 |           89.45s |      4.50s |      94.31s | cutoff                            |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |            16,290 |                   500 |         16,790 |          199 |      16.5 |          38 |          112.81s |      3.81s |     116.97s | cutoff                            |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |            16,290 |                   500 |         16,790 |          195 |        17 |          38 |          113.66s |      3.57s |     117.59s | cutoff                            |                 |

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
- `mlx`: `0.32.0.dev20260501+e8ebdebe`
- `mlx-vlm`: `0.4.5`
- `mlx-lm`: `0.31.3`
- `huggingface-hub`: `1.13.0`
- `transformers`: `5.7.0`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-05-01 22:38:46 BST_
