# Model Performance Results

_Generated on 2026-04-26 22:21:24 BST_

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ 1 (top owners: model-config=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=9, clean outputs=30/52.
- _Useful now:_ 30 clean A/B model(s) worth first review.
- _Review watchlist:_ 20 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Quality signal frequency:_ harness=9, cutoff=8, reasoning_leak=7,
  formatting=4, token_cap=3, verbose=3.
- _Termination reasons:_ completed=52, exception=1.

### Runtime

- _Runtime pattern:_ decode dominates measured phase time (92%; 50/53 measured
  model(s)).
- _Phase totals:_ model load=106.96s, prompt prep=0.16s, decode=1237.59s,
  cleanup=6.51s.
- _What this likely means:_ Most measured runtime is spent inside generation
  rather than load or prompt setup.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=52, exception=1.

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/FastVLM-0.5B-bf16` (357.4 tps)
- **💾 Most efficient:** `mlx-community/nanoLLaVA-1.5-4bit` (1.8 GB)
- **⚡ Fastest load:** `mlx-community/LFM2-VL-1.6B-8bit` (0.50s)
- **📊 Average TPS:** 73.8 across 52 models

## 📈 Resource Usage

- **Total peak memory:** 1087.1 GB
- **Average peak memory:** 20.9 GB
- **Memory efficiency:** 209 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 1 | ✅ B: 43 | 🟡 C: 3 | 🟠 D: 1 | ❌ F: 4

**Average Utility Score:** 69/100

- **Best for cataloging:** `mlx-community/gemma-3-27b-it-qat-4bit` (🏆 A, 85/100)
- **Best descriptions:** `qnguyen3/nanoLLaVA` (90/100)
- **Best keywording:** `mlx-community/GLM-4.6V-Flash-mxfp4` (100/100)
- **Worst for cataloging:** `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` (❌ F, 0/100)

### ⚠️ 5 Models with Low Utility (D/F)

- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/gemma-3n-E2B-4bit`: ❌ F (5/100) - Output too short to be useful
- `mlx-community/gemma-4-31b-bf16`: ❌ F (5/100) - Output too short to be useful
- `mlx-community/paligemma2-3b-pt-896-4bit`: ❌ F (5/100) - Output too short to be useful

## ⚠️ Quality Issues

- **❌ Failed Models (1):**
  - `mlx-community/MolmoPoint-8B-fp16` (`Processor Error`)
- **🔄 Repetitive Output (1):**
  - `mlx-community/Qwen2-VL-2B-Instruct-4bit` (token: `phrase: "' chinese: ' chinese:..."`)
- **📝 Formatting Issues (4):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-Flash-mxfp4`
  - `mlx-community/GLM-4.6V-nvfp4`
  - `mlx-community/Idefics3-8B-Llama3-bf16`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 73.8 | Min: 3.28 | Max: 357
- **Peak Memory**: Avg: 21 | Min: 1.8 | Max: 78
- **Total Time**: Avg: 26.19s | Min: 1.97s | Max: 172.27s
- **Generation Time**: Avg: 23.80s | Min: 0.74s | Max: 168.71s
- **Model Load Time**: Avg: 2.02s | Min: 0.50s | Max: 10.29s

### ⏱ Runtime Interpretation

- **Runtime pattern:** decode dominates measured phase time (92%; 50/53 measured model(s)).
- **Phase totals:** model load=106.96s, prompt prep=0.16s, decode=1237.59s, cleanup=6.51s.
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.
- **Termination reasons:** completed=52, exception=1.

### ⏱ Timing Snapshot

- **Validation overhead:** 19.52s total (avg 0.37s across 53 model(s)).
- **First-token latency:** Avg 13.63s | Min 0.06s | Max 113.56s across 52 model(s).

## ✅ Recommended Models

Quick picks based on end-to-end utility plus description and keyword strength.

- _Best end-to-end cataloging:_ [`mlx-community/gemma-3-27b-it-qat-4bit`](model_gallery.md#model-mlx-community-gemma-3-27b-it-qat-4bit)
  (A 85/100 | Desc 82 | Keywords 92 | Gen 14.9 TPS | Peak 19 | A 85/100 |
  nontext prompt burden=99%)
- _Best descriptions:_ [`qnguyen3/nanoLLaVA`](model_gallery.md#model-qnguyen3-nanollava)
  (B 74/100 | Desc 90 | Keywords 0 | Gen 113 TPS | Peak 3.8 | B 74/100 |
  nontext prompt burden=80%)
- _Best keywording:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
  (B 75/100 | Desc 82 | Keywords 96 | Gen 66.0 TPS | Peak 13 | B 75/100 |
  nontext prompt burden=100%)
- _Fastest generation:_ [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16)
  (B 74/100 | Desc 82 | Keywords 0 | Gen 357 TPS | Peak 2.2 | B 74/100 |
  nontext prompt burden=83%)
- _Lowest memory footprint:_ [`mlx-community/nanoLLaVA-1.5-4bit`](model_gallery.md#model-mlx-community-nanollava-15-4bit)
  (B 75/100 | Desc 86 | Keywords 0 | Gen 232 TPS | Peak 1.8 | B 75/100 |
  nontext prompt burden=80%)
- _Best balance:_ [`mlx-community/gemma-3-27b-it-qat-4bit`](model_gallery.md#model-mlx-community-gemma-3-27b-it-qat-4bit)
  (A 85/100 | Desc 82 | Keywords 92 | Gen 14.9 TPS | Peak 19 | A 85/100 |
  nontext prompt burden=99%)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (1):_ [`mlx-community/MolmoPoint-8B-fp16`](model_gallery.md#model-mlx-community-molmopoint-8b-fp16).
  Example: `Processor Error`.
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
| `model-config` | 1 | Processor Error | `mlx-community/MolmoPoint-8B-fp16` |

### Actionable Items by Package

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

_Overall runtime:_ 1371.49s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                    |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:----------------------------------|----------------:|
| `mlx-community/MolmoPoint-8B-fp16`                      |                   |                       |                |              |           |             |                  |      2.16s |       2.49s |                                   |    model-config |
| `mlx-community/gemma-3n-E2B-4bit`                       |               264 |                     4 |            268 |         1146 |      92.8 |           6 |            0.74s |      1.38s |       2.53s | ⚠️harness(prompt_template)        |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |                85 |                   117 |            202 |         1478 |       187 |         3.6 |            1.10s |      0.54s |       1.97s |                                   |                 |
| `qnguyen3/nanoLLaVA`                                    |                20 |                    51 |             71 |          153 |       113 |         3.8 |            1.35s |      0.68s |       2.60s |                                   |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |                20 |                   124 |            144 |          223 |       232 |         1.8 |            1.37s |      0.56s |       2.36s |                                   |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |               766 |                    41 |            807 |         3392 |      58.5 |         9.1 |            1.37s |      0.82s |       2.52s |                                   |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               275 |                   292 |            567 |         3348 |       322 |           3 |            1.40s |      0.50s |       2.23s |                                   |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |                24 |                   445 |            469 |          277 |       357 |         2.2 |            2.21s |      0.56s |       3.10s |                                   |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,101 |                     3 |          4,104 |         2407 |      64.2 |         4.6 |            2.29s |      1.33s |       4.19s | ⚠️harness(long_context), ...      |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               266 |                     5 |            271 |          152 |      9.37 |          64 |            2.84s |      6.13s |       9.45s | ⚠️harness(prompt_template)        |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |                95 |                   266 |            361 |          249 |       132 |         5.5 |            3.02s |      0.61s |       4.15s |                                   |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,326 |                    48 |          2,374 |         2234 |      32.7 |          19 |            3.13s |      1.89s |       5.36s | formatting                        |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |             2,629 |                   355 |          2,984 |         2637 |       187 |         7.9 |            3.44s |      0.92s |       4.70s |                                   |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |             1,011 |                   345 |          1,356 |         1353 |       132 |          18 |            3.89s |      1.92s |       6.13s | reasoning-leak                    |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,195 |                   420 |          1,615 |         3184 |       126 |         5.5 |            4.29s |      0.63s |       5.23s |                                   |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               278 |                   267 |            545 |          705 |      71.2 |          17 |            4.66s |      2.37s |       7.62s |                                   |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |             1,011 |                   296 |          1,307 |         1283 |      79.9 |          37 |            5.17s |      3.30s |       8.81s | reasoning-leak                    |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |             1,011 |                   500 |          1,511 |         1362 |       117 |          22 |            5.60s |      2.14s |       8.08s | reasoning-leak, token-cap, ...    |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             1,195 |                   420 |          1,615 |         2338 |        89 |         5.5 |            5.85s |      0.59s |       6.82s |                                   |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             1,804 |                   125 |          1,929 |         1330 |      31.8 |          18 |            5.85s |      1.64s |       7.81s |                                   |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             2,174 |                    74 |          2,248 |          516 |      55.4 |         9.7 |            6.21s |      0.86s |       7.50s |                                   |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |             2,630 |                   311 |          2,941 |         1268 |        66 |          13 |            7.32s |      1.31s |       8.96s |                                   |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |             2,098 |                   135 |          2,233 |          707 |      31.6 |          22 |            7.79s |      1.97s |      10.10s | ⚠️harness(encoding), ...          |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               272 |                   237 |            509 |          869 |      33.2 |          17 |            7.91s |      2.20s |      10.45s |                                   |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |             2,630 |                   344 |          2,974 |         1236 |      62.8 |          14 |            8.14s |      1.33s |       9.80s |                                   |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,029 |                   135 |          1,164 |         1048 |      20.2 |          11 |            8.35s |      1.56s |      10.40s |                                   |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,029 |                   118 |          1,147 |         2702 |      13.6 |          10 |            9.56s |      1.43s |      11.53s |                                   |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |               766 |                   500 |          1,266 |         3298 |      55.2 |         9.1 |            9.74s |      0.91s |      10.97s | ⚠️harness(stop_token), ...        |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |             1,293 |                   500 |          1,793 |          846 |      59.8 |          60 |           10.63s |      4.75s |      15.71s | cutoff                            |                 |
| `mlx-community/pixtral-12b-8bit`                        |             2,887 |                   167 |          3,054 |          864 |      21.9 |          16 |           11.50s |      1.63s |      13.46s |                                   |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             1,804 |                   344 |          2,148 |         2328 |      33.9 |          17 |           11.55s |      1.67s |      13.56s |                                   |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |             6,091 |                   500 |          6,591 |         1054 |      72.4 |         8.4 |           13.16s |      1.31s |      14.81s | ⚠️harness(stop_token), ...        |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               278 |                   158 |            436 |          315 |      12.8 |          19 |           13.73s |      2.55s |      16.63s |                                   |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             2,979 |                   500 |          3,479 |         1525 |      42.1 |          15 |           14.36s |      1.65s |      16.35s | degeneration, reasoning-leak, ... |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |             6,091 |                   500 |          6,591 |         1109 |        54 |          11 |           15.21s |      1.37s |      16.91s | ⚠️harness(stop_token), ...        |                 |
| `mlx-community/pixtral-12b-bf16`                        |             2,887 |                   166 |          3,053 |          995 |        13 |          28 |           16.19s |      2.50s |      19.10s |                                   |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |             1,207 |                   205 |          1,412 |         73.2 |      50.3 |          41 |           21.41s |      1.06s |      22.80s |                                   |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               273 |                   320 |            593 |          308 |      14.9 |          19 |           22.91s |      2.27s |      25.58s |                                   |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |                14 |                   421 |            435 |          9.9 |      19.4 |          15 |           23.57s |      1.47s |      25.37s |                                   |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |             6,091 |                   500 |          6,591 |          567 |      37.9 |          78 |           24.41s |      5.51s |      30.26s | ⚠️harness(stop_token), ...        |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |             1,207 |                   205 |          1,412 |         68.1 |        30 |          48 |           25.37s |      1.58s |      27.28s |                                   |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,029 |                   100 |          1,129 |          883 |      3.28 |          26 |           32.17s |      2.41s |      35.07s |                                   |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |            16,299 |                   219 |         16,518 |          524 |      30.7 |          14 |           39.15s |      1.20s |      40.84s |                                   |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               273 |                   381 |            654 |          270 |      9.07 |          33 |           43.49s |      3.37s |      47.22s |                                   |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |                15 |                   276 |            291 |          8.9 |      4.95 |          25 |           57.94s |      2.25s |      60.51s |                                   |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |            16,290 |                   500 |         16,790 |          288 |       101 |          26 |           62.26s |      2.46s |      65.05s | cutoff                            |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |            16,290 |                   500 |         16,790 |          286 |      87.3 |          35 |           63.39s |      3.14s |      66.87s | cutoff                            |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |            16,299 |                   500 |         16,799 |          251 |       198 |         5.1 |           68.09s |      0.50s |      68.92s | ⚠️harness(long_context), ...      |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |            16,290 |                   500 |         16,790 |          264 |      63.4 |          76 |           70.46s |     10.29s |      81.09s | token-cap                         |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |            16,290 |                   500 |         16,790 |          223 |      49.3 |          12 |           84.05s |      1.35s |      85.74s | degeneration, token-cap           |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |            16,290 |                   500 |         16,790 |          219 |      29.1 |          26 |           92.34s |      2.17s |      94.85s | cutoff                            |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |            16,290 |                   500 |         16,790 |          209 |      17.8 |          38 |          106.94s |      3.12s |     110.41s | cutoff                            |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |            16,290 |                   500 |         16,790 |          143 |      9.19 |          38 |          168.71s |      3.15s |     172.27s | cutoff                            |                 |

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
- `mlx`: `0.32.0.dev20260426+211e57be`
- `mlx-vlm`: `0.4.5`
- `mlx-lm`: `0.31.3`
- `huggingface-hub`: `1.12.0`
- `transformers`: `5.7.0.dev0`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-04-26 22:21:24 BST_
