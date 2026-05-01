# Model Performance Results

_Generated on 2026-05-01 17:56:05 BST_

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
- _Termination reasons:_ completed=51, exception=2.

### Runtime

- _Runtime pattern:_ decode dominates measured phase time (91%; 49/53 measured
  model(s)).
- _Phase totals:_ model load=104.66s, prompt prep=0.16s, decode=1112.64s,
  cleanup=5.35s.
- _What this likely means:_ Most measured runtime is spent inside generation
  rather than load or prompt setup.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=51, exception=2.

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/nanoLLaVA-1.5-4bit` (370.3 tps)
- **💾 Most efficient:** `mlx-community/nanoLLaVA-1.5-4bit` (1.9 GB)
- **⚡ Fastest load:** `HuggingFaceTB/SmolVLM-Instruct` (0.39s)
- **📊 Average TPS:** 78.1 across 51 models

## 📈 Resource Usage

- **Total peak memory:** 1067.8 GB
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

- **Generation Tps**: Avg: 78.1 | Min: 4.67 | Max: 370
- **Peak Memory**: Avg: 21 | Min: 1.9 | Max: 78
- **Total Time**: Avg: 24.15s | Min: 1.74s | Max: 110.10s
- **Generation Time**: Avg: 21.82s | Min: 0.74s | Max: 106.66s
- **Model Load Time**: Avg: 2.00s | Min: 0.39s | Max: 8.69s

### ⏱ Runtime Interpretation

- **Runtime pattern:** decode dominates measured phase time (91%; 49/53 measured model(s)).
- **Phase totals:** model load=104.66s, prompt prep=0.16s, decode=1112.64s, cleanup=5.35s.
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.
- **Termination reasons:** completed=51, exception=2.

### ⏱ Timing Snapshot

- **Validation overhead:** 17.39s total (avg 0.33s across 53 model(s)).
- **First-token latency:** Avg 12.38s | Min 0.06s | Max 77.20s across 51 model(s).

## ✅ Recommended Models

Quick picks based on end-to-end utility plus description and keyword strength.

- _Best end-to-end cataloging:_ [`mlx-community/gemma-3-27b-it-qat-4bit`](model_gallery.md#model-mlx-community-gemma-3-27b-it-qat-4bit)
  (A 85/100 | Desc 82 | Keywords 92 | Gen 28.6 TPS | Peak 19 | A 85/100 |
  nontext prompt burden=99%)
- _Best descriptions:_ [`qnguyen3/nanoLLaVA`](model_gallery.md#model-qnguyen3-nanollava)
  (B 74/100 | Desc 90 | Keywords 0 | Gen 115 TPS | Peak 4.0 | B 74/100 |
  nontext prompt burden=80%)
- _Best keywording:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
  (B 75/100 | Desc 82 | Keywords 96 | Gen 66.6 TPS | Peak 13 | B 75/100 |
  nontext prompt burden=100%)
- _Fastest generation:_ [`mlx-community/nanoLLaVA-1.5-4bit`](model_gallery.md#model-mlx-community-nanollava-15-4bit)
  (B 75/100 | Desc 86 | Keywords 0 | Gen 370 TPS | Peak 1.9 | B 75/100 |
  nontext prompt burden=80%)
- _Lowest memory footprint:_ [`mlx-community/nanoLLaVA-1.5-4bit`](model_gallery.md#model-mlx-community-nanollava-15-4bit)
  (B 75/100 | Desc 86 | Keywords 0 | Gen 370 TPS | Peak 1.9 | B 75/100 |
  nontext prompt burden=80%)
- _Best balance:_ [`mlx-community/gemma-3-27b-it-qat-4bit`](model_gallery.md#model-mlx-community-gemma-3-27b-it-qat-4bit)
  (A 85/100 | Desc 82 | Keywords 92 | Gen 28.6 TPS | Peak 19 | A 85/100 |
  nontext prompt burden=99%)

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

_Overall runtime:_ 1240.84s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Active Mem (GB) |   Cache Mem (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                    |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|------------------:|-----------------:|-----------------:|-----------:|------------:|:----------------------------------|----------------:|
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |                   |                       |                |              |           |             |                   |                  |                  |      0.34s |       0.67s |                                   |             mlx |
| `mlx-community/MolmoPoint-8B-fp16`                      |                   |                       |                |              |           |             |                   |                  |                  |      2.22s |       2.55s |                                   |    model-config |
| `mlx-community/gemma-3n-E2B-4bit`                       |               264 |                     4 |            268 |         1168 |      92.9 |           6 |               4.2 |             0.03 |            0.74s |      1.44s |       2.52s | ⚠️harness(prompt_template)        |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |                20 |                   124 |            144 |          255 |       370 |         1.9 |              0.57 |             0.23 |            0.92s |      0.49s |       1.74s |                                   |                 |
| `qnguyen3/nanoLLaVA`                                    |                20 |                    51 |             71 |          180 |       115 |           4 |                 2 |              1.4 |            1.06s |      0.57s |       1.95s |                                   |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |                85 |                   117 |            202 |         1399 |       185 |         3.6 |                 3 |                0 |            1.10s |      0.58s |       2.01s |                                   |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               275 |                   292 |            567 |         3301 |       328 |           3 |               1.9 |             0.02 |            1.38s |      0.48s |       2.18s |                                   |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |               768 |                    53 |            821 |         3389 |      59.3 |         9.4 |               7.7 |             0.39 |            1.57s |      0.86s |       2.76s |                                   |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,101 |                     3 |          4,104 |         3705 |      81.1 |         4.6 |               1.6 |              1.6 |            1.68s |      1.18s |       3.19s | ⚠️harness(long_context), ...      |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |                24 |                   445 |            469 |          265 |       358 |         2.1 |               1.2 |             0.03 |            2.21s |      0.59s |       3.14s |                                   |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |                95 |                   266 |            361 |          310 |       131 |         5.5 |               4.2 |             0.13 |            2.96s |      0.61s |       3.90s |                                   |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,326 |                    48 |          2,374 |         2283 |        33 |          19 |                16 |             0.36 |            3.08s |      1.93s |       5.34s | formatting                        |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               278 |                   267 |            545 |          704 |       115 |          17 |                15 |             0.18 |            3.21s |      2.35s |       5.92s |                                   |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               266 |                     5 |            271 |          119 |      9.07 |          64 |                58 |             0.45 |            3.32s |      6.07s |       9.72s | ⚠️harness(prompt_template)        |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |             2,629 |                   355 |          2,984 |         2521 |       187 |         7.9 |               2.6 |             0.35 |            3.51s |      0.99s |       4.85s |                                   |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             2,174 |                    74 |          2,248 |          989 |      62.5 |         9.7 |               7.5 |             0.77 |            4.03s |      0.93s |       5.30s |                                   |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,195 |                   420 |          1,615 |         3190 |       125 |         5.5 |               4.2 |             0.37 |            4.32s |      0.39s |       5.05s |                                   |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             1,195 |                   420 |          1,615 |         3154 |       121 |         5.5 |               4.2 |             0.37 |            4.49s |      0.64s |       5.46s |                                   |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,029 |                   135 |          1,164 |         1326 |      34.6 |          11 |               7.3 |              1.4 |            5.29s |      1.59s |       7.22s |                                   |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               272 |                   237 |            509 |          873 |      48.8 |          17 |                15 |             0.03 |            5.63s |      2.19s |       8.16s |                                   |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             1,804 |                   125 |          1,929 |         1348 |      32.4 |          18 |                15 |             0.42 |            5.76s |      1.77s |       7.86s |                                   |                 |
| `mlx-community/pixtral-12b-8bit`                        |             2,887 |                   167 |          3,054 |         1772 |      39.2 |          16 |                13 |             0.51 |            6.42s |      1.63s |       8.39s |                                   |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |             1,011 |                   357 |          1,368 |          933 |      72.6 |          18 |                13 |              1.1 |            6.56s |      1.94s |       8.84s | reasoning-leak                    |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,029 |                   118 |          1,147 |         2781 |      19.7 |          10 |               5.7 |              3.3 |            6.88s |      1.46s |       8.67s |                                   |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               278 |                   158 |            436 |          385 |      27.4 |          19 |                17 |             0.46 |            6.99s |      2.57s |       9.90s |                                   |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |             2,630 |                   311 |          2,941 |         1246 |      66.6 |          13 |               7.5 |             0.52 |            7.32s |      1.36s |       9.04s |                                   |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |             2,630 |                   344 |          2,974 |         1210 |      63.2 |          14 |               7.9 |             0.52 |            8.17s |      1.38s |       9.89s |                                   |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |             2,097 |                   172 |          2,269 |          662 |      31.6 |          22 |                16 |             0.39 |            9.17s |      2.10s |      11.64s | ⚠️harness(encoding), ...          |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |               768 |                   500 |          1,268 |         3407 |      58.7 |         9.3 |               7.7 |             0.48 |            9.19s |      0.93s |      10.45s | ⚠️harness(stop_token), ...        |                 |
| `mlx-community/pixtral-12b-bf16`                        |             2,887 |                   166 |          3,053 |         1798 |      20.3 |          28 |                24 |             0.51 |           10.29s |      2.56s |      13.17s |                                   |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |             1,293 |                   500 |          1,793 |          859 |      60.2 |          60 |                55 |             0.12 |           10.56s |      4.96s |      15.87s | cutoff                            |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             1,804 |                   344 |          2,148 |         2467 |      34.5 |          17 |                15 |             0.15 |           11.25s |      1.76s |      13.34s |                                   |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               273 |                   320 |            593 |          395 |      28.6 |          19 |                16 |             0.39 |           12.38s |      2.33s |      15.05s |                                   |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |             6,091 |                   500 |          6,591 |         1136 |        72 |         8.4 |               6.3 |             0.37 |           12.78s |      1.31s |      14.43s | ⚠️harness(stop_token), ...        |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             2,979 |                   500 |          3,479 |         1460 |      42.1 |          15 |                12 |             0.67 |           14.45s |      1.60s |      16.39s | degeneration, reasoning-leak, ... |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |             6,091 |                   500 |          6,591 |         1126 |        55 |          11 |               8.8 |             0.37 |           14.96s |      1.40s |      16.69s | ⚠️harness(stop_token), ...        |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,029 |                   100 |          1,129 |          936 |      5.43 |          26 |                18 |              5.7 |           20.06s |      2.46s |      22.85s |                                   |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |                14 |                   421 |            435 |         10.4 |      21.2 |          15 |                11 |             0.54 |           21.70s |      1.63s |      23.66s |                                   |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |             1,207 |                   205 |          1,412 |           65 |      52.4 |          41 |               8.4 |             0.12 |           23.34s |      1.21s |      24.90s |                                   |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |            16,299 |                   219 |         16,518 |          875 |      54.6 |          14 |               8.8 |              1.3 |           23.37s |      1.09s |      24.80s |                                   |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               273 |                   381 |            654 |          326 |      16.8 |          33 |                29 |             0.39 |           23.95s |      3.40s |      27.69s |                                   |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |             6,091 |                   500 |          6,591 |          498 |      36.4 |          78 |                58 |              1.3 |           26.41s |      5.51s |      32.25s | ⚠️harness(stop_token), ...        |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |             1,207 |                   215 |          1,422 |         62.5 |      30.7 |          48 |                15 |             0.11 |           27.15s |      1.84s |      29.32s |                                   |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |                15 |                   276 |            291 |         9.04 |      5.07 |          25 |                20 |              3.2 |           56.54s |      2.19s |      59.06s |                                   |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |            16,299 |                    13 |         16,312 |          270 |       209 |         5.1 |               1.2 |             0.78 |           61.16s |      0.49s |      61.98s | ⚠️harness(long_context), ...      |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |             1,011 |                   277 |          1,288 |          704 |      4.67 |          39 |                31 |              4.3 |           61.46s |      3.27s |      65.06s | reasoning-leak                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |            16,290 |                   500 |         16,790 |          288 |       102 |          26 |                19 |             0.77 |           62.24s |      2.50s |      65.08s | cutoff                            |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |            16,290 |                   500 |         16,790 |          289 |      87.3 |          35 |                27 |             0.76 |           62.83s |      3.17s |      66.34s | cutoff                            |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |            16,290 |                   500 |         16,790 |          271 |      64.1 |          76 |                65 |             0.76 |           68.90s |      8.69s |      77.93s | token-cap                         |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |            16,290 |                   500 |         16,790 |          269 |      47.2 |          12 |               5.6 |             0.95 |           72.01s |      1.38s |      73.74s | degeneration, token-cap           |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |            16,290 |                   500 |         16,790 |          226 |      28.5 |          26 |                15 |              1.6 |           90.50s |      2.16s |      93.02s | cutoff                            |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |            16,290 |                   500 |         16,790 |          228 |      17.6 |          38 |                27 |              1.6 |          100.73s |      3.04s |     104.11s | cutoff                            |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |            16,290 |                   500 |         16,790 |          211 |      17.4 |          38 |                27 |              1.6 |          106.66s |      3.09s |     110.10s | cutoff                            |                 |

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

_Report generated on: 2026-05-01 17:56:05 BST_
