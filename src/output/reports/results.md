# Model Performance Results

_Generated on 2026-05-02 11:57:26 BST_

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
  formatting=4, long_context=2, context_budget=2.

### Runtime

- _Runtime pattern:_ decode dominates measured phase time (92%; 49/53 measured
  model(s)).
- _Phase totals:_ model load=107.57s, prompt prep=0.16s, decode=1256.36s,
  cleanup=5.78s.
- _What this likely means:_ Most measured runtime is spent inside generation
  rather than load or prompt setup.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=51, exception=2.
- _Validation overhead:_ 18.60s total (avg 0.35s across 53 model(s)).
- _First-token latency:_ Avg 14.86s | Min 0.06s | Max 103.41s across 51
  model(s).

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/FastVLM-0.5B-bf16` (358.3 tps)
- **💾 Most efficient:** `mlx-community/nanoLLaVA-1.5-4bit` (1.9 GB)
- **⚡ Fastest load:** `mlx-community/LFM2-VL-1.6B-8bit` (0.46s)
- **📊 Average TPS:** 76.9 across 51 models

## 📈 Resource Usage

- **Total peak memory:** 1067.7 GB
- **Average peak memory:** 20.9 GB
- **Memory efficiency:** 211 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 1 | ✅ B: 42 | 🟡 C: 3 | ❌ F: 5

**Average Utility Score:** 69/100

- **Best for cataloging:** `mlx-community/gemma-3-27b-it-qat-4bit` (🏆 A, 85/100)
- **Best descriptions:** `qnguyen3/nanoLLaVA` (90/100)
- **Best keywording:** `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` (96/100)
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

- **Generation Tps**: Avg: 76.9 | Min: 4.74 | Max: 358
- **Peak Memory**: Avg: 21 | Min: 1.9 | Max: 78
- **Total Time**: Avg: 27.06s | Min: 1.93s | Max: 151.38s
- **Generation Time**: Avg: 24.63s | Min: 0.74s | Max: 147.92s
- **Model Load Time**: Avg: 2.06s | Min: 0.46s | Max: 9.25s

## ✅ Recommended Models

Quick picks based on end-to-end utility plus description and keyword strength.

- _Best end-to-end cataloging:_ [`mlx-community/gemma-3-27b-it-qat-4bit`](model_gallery.md#model-mlx-community-gemma-3-27b-it-qat-4bit)
  (Utility A 85/100 | Description 82 | Keywords 92 | Speed 31.6 TPS | Memory
  19 | Caveat nontext prompt burden=99%)
- _Best descriptions:_ [`qnguyen3/nanoLLaVA`](model_gallery.md#model-qnguyen3-nanollava)
  (Utility B 74/100 | Description 90 | Keywords 0 | Speed 116 TPS | Memory 3.9
  | Caveat nontext prompt burden=80%)
- _Best keywording:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
  (Utility B 75/100 | Description 82 | Keywords 96 | Speed 66.4 TPS | Memory
  13 | Caveat nontext prompt burden=100%)
- _Fastest generation:_ [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16)
  (Utility B 74/100 | Description 82 | Keywords 0 | Speed 358 TPS | Memory 2.1
  | Caveat nontext prompt burden=83%)
- _Lowest memory footprint:_ [`mlx-community/nanoLLaVA-1.5-4bit`](model_gallery.md#model-mlx-community-nanollava-15-4bit)
  (Utility B 75/100 | Description 86 | Keywords 0 | Speed 352 TPS | Memory 1.9
  | Caveat nontext prompt burden=80%)
- _Best balance:_ [`mlx-community/gemma-3-27b-it-qat-4bit`](model_gallery.md#model-mlx-community-gemma-3-27b-it-qat-4bit)
  (Utility A 85/100 | Description 82 | Keywords 92 | Speed 31.6 TPS | Memory
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

_Overall runtime:_ 1389.28s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                    |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:----------------------------------|----------------:|
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |                   |                       |                |              |           |             |                  |      0.19s |       0.52s |                                   |             mlx |
| `mlx-community/MolmoPoint-8B-fp16`                      |                   |                       |                |              |           |             |                  |      2.20s |       2.53s |                                   |    model-config |
| `mlx-community/gemma-3n-E2B-4bit`                       |               264 |                     4 |            268 |         1150 |      92.6 |           6 |            0.74s |      1.45s |       2.53s | ⚠️harness(prompt_template)        |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |                20 |                   124 |            144 |          243 |       352 |         1.9 |            0.97s |      0.58s |       2.91s |                                   |                 |
| `qnguyen3/nanoLLaVA`                                    |                20 |                    51 |             71 |          173 |       116 |         3.9 |            1.06s |      0.55s |       1.93s |                                   |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |                85 |                   117 |            202 |         1456 |       188 |         3.6 |            1.09s |      0.54s |       1.96s |                                   |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               275 |                   292 |            567 |         3347 |       332 |           3 |            1.37s |      0.46s |       2.15s |                                   |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |               768 |                    53 |            821 |         3408 |      59.2 |         9.4 |            1.57s |      0.85s |       2.74s |                                   |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,101 |                     3 |          4,104 |         3842 |      86.7 |         4.6 |            1.65s |      1.11s |       3.09s | ⚠️harness(long_context), ...      |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |                24 |                   445 |            469 |          258 |       358 |         2.1 |            2.21s |      0.71s |       3.25s |                                   |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |                95 |                   266 |            361 |          313 |       131 |         5.5 |            2.95s |      0.59s |       3.86s |                                   |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,326 |                    48 |          2,374 |         2266 |      32.7 |          19 |            3.12s |      2.10s |       5.58s | formatting                        |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               278 |                   267 |            545 |          703 |       118 |          17 |            3.17s |      2.34s |       5.86s |                                   |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |             2,629 |                   355 |          2,984 |         2663 |       188 |         7.9 |            3.41s |      0.93s |       4.68s |                                   |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               266 |                     5 |            271 |          115 |      8.99 |          64 |            3.42s |      6.62s |      10.37s | ⚠️harness(prompt_template)        |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,195 |                   420 |          1,615 |         3126 |       128 |         5.5 |            4.23s |      0.65s |       5.19s |                                   |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             1,195 |                   420 |          1,615 |         3179 |       126 |         5.5 |            4.30s |      0.68s |       5.30s |                                   |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,029 |                   135 |          1,164 |         1316 |      33.9 |          11 |            5.39s |      1.64s |       7.35s |                                   |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               272 |                   237 |            509 |          873 |      48.8 |          17 |            5.63s |      2.20s |       8.17s |                                   |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             1,804 |                   125 |          1,929 |         1348 |      32.3 |          18 |            5.77s |      1.74s |       7.85s |                                   |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |             1,011 |                   357 |          1,368 |         1276 |      74.4 |          18 |            6.13s |      2.03s |       8.46s | reasoning-leak                    |                 |
| `mlx-community/pixtral-12b-8bit`                        |             2,887 |                   167 |          3,054 |         1779 |      39.4 |          16 |            6.39s |      1.75s |       8.48s |                                   |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,029 |                   118 |          1,147 |         2756 |      19.9 |          10 |            6.82s |      1.43s |       8.59s |                                   |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |             2,630 |                   311 |          2,941 |         1294 |      66.4 |          13 |            7.26s |      1.34s |       8.94s |                                   |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |             2,630 |                   344 |          2,974 |         1293 |      63.3 |          14 |            8.01s |      1.42s |       9.76s |                                   |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |             2,097 |                   172 |          2,269 |          714 |      32.1 |          22 |            8.83s |      2.11s |      11.38s | ⚠️harness(encoding), ...          |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |               768 |                   500 |          1,268 |         3240 |      56.9 |         9.3 |            9.46s |      1.00s |      10.78s | ⚠️harness(stop_token), ...        |                 |
| `mlx-community/pixtral-12b-bf16`                        |             2,887 |                   166 |          3,053 |         1804 |      19.9 |          28 |           10.48s |      2.56s |      13.37s |                                   |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |             1,293 |                   500 |          1,793 |          766 |      60.6 |          60 |           10.68s |      4.89s |      15.91s | cutoff                            |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             1,804 |                   344 |          2,148 |         2465 |      34.3 |          17 |           11.31s |      1.82s |      13.46s |                                   |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               273 |                   320 |            593 |          389 |      31.6 |          19 |           11.33s |      2.43s |      14.09s |                                   |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               278 |                   158 |            436 |          385 |      15.4 |          19 |           11.52s |      2.57s |      14.43s |                                   |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             2,174 |                    74 |          2,248 |          265 |      22.6 |         9.7 |           12.21s |      0.95s |      13.52s |                                   |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |             6,091 |                   500 |          6,591 |         1047 |      71.6 |         8.4 |           13.30s |      1.37s |      15.03s | ⚠️harness(stop_token), ...        |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             2,979 |                   500 |          3,479 |         1509 |      41.5 |          15 |           14.56s |      1.68s |      16.58s | degeneration, reasoning-leak, ... |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |             6,091 |                   500 |          6,591 |         1111 |      54.5 |          11 |           15.12s |      1.36s |      16.82s | ⚠️harness(stop_token), ...        |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,029 |                   100 |          1,129 |          927 |      5.47 |          26 |           19.93s |      2.54s |      22.81s |                                   |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |                14 |                   421 |            435 |         10.5 |      21.1 |          15 |           21.72s |      1.50s |      23.54s |                                   |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |            16,299 |                   219 |         16,518 |          918 |      57.9 |          14 |           22.25s |      1.16s |      23.74s |                                   |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               273 |                   381 |            654 |          325 |      17.7 |          33 |           22.83s |      3.70s |      26.86s |                                   |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |             1,207 |                   205 |          1,412 |         63.8 |      52.4 |          41 |           23.66s |      1.17s |      25.16s |                                   |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |             6,091 |                   500 |          6,591 |          518 |      37.9 |          78 |           25.44s |      6.20s |      31.99s | ⚠️harness(stop_token), ...        |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |             1,207 |                   221 |          1,428 |         61.7 |      30.3 |          48 |           27.67s |      1.72s |      29.73s |                                   |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |                15 |                   276 |            291 |         8.78 |         5 |          25 |           57.36s |      2.17s |      59.85s |                                   |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |             1,011 |                   277 |          1,288 |          770 |      4.74 |          39 |           60.37s |      3.27s |      63.97s | reasoning-leak                    |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |            16,299 |                    13 |         16,312 |          273 |       210 |         5.1 |           60.58s |      0.49s |      61.39s | ⚠️harness(long_context), ...      |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |            16,290 |                   500 |         16,790 |          242 |      30.3 |          26 |           84.53s |      2.15s |      87.04s | cutoff                            |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |            16,290 |                   500 |         16,790 |          208 |      65.1 |          76 |           87.08s |      9.25s |      96.67s | token-cap                         |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |            16,290 |                   500 |         16,790 |          205 |      72.9 |          12 |           87.09s |      1.37s |      88.80s | degeneration, token-cap           |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |            16,290 |                   500 |         16,790 |          202 |      53.7 |          26 |           90.56s |      2.60s |      93.50s | cutoff                            |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |            16,290 |                   500 |         16,790 |          231 |      18.2 |          38 |           98.94s |      3.16s |     102.45s | cutoff                            |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |            16,290 |                   500 |         16,790 |          168 |      90.4 |          35 |          102.99s |      3.14s |     106.68s | cutoff                            |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |            16,290 |                   500 |         16,790 |          158 |      11.4 |          38 |          147.92s |      3.11s |     151.38s | cutoff                            |                 |

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

_Report generated on: 2026-05-02 11:57:26 BST_
