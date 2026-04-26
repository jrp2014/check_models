# Model Performance Results

_Generated on 2026-04-26 20:52:02 BST_

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ 1 (top owners: model-config=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=9, clean outputs=30/52.
- _Useful now:_ 30 clean A/B model(s) worth first review.
- _Review watchlist:_ 20 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Quality signal frequency:_ harness=9, reasoning_leak=7, cutoff=7,
  formatting=4, token_cap=3, verbose=3.
- _Termination reasons:_ completed=52, exception=1.

### Runtime

- _Runtime pattern:_ decode dominates measured phase time (90%; 50/53 measured
  model(s)).
- _Phase totals:_ model load=114.98s, prompt prep=0.15s, decode=1036.22s,
  cleanup=5.15s.
- _What this likely means:_ Most measured runtime is spent inside generation
  rather than load or prompt setup.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=52, exception=1.

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/nanoLLaVA-1.5-4bit` (394.0 tps)
- **💾 Most efficient:** `mlx-community/nanoLLaVA-1.5-4bit` (1.8 GB)
- **⚡ Fastest load:** `mlx-community/nanoLLaVA-1.5-4bit` (0.47s)
- **📊 Average TPS:** 83.4 across 52 models

## 📈 Resource Usage

- **Total peak memory:** 1087.2 GB
- **Average peak memory:** 20.9 GB
- **Memory efficiency:** 209 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 1 | ✅ B: 43 | 🟡 C: 3 | ❌ F: 5

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

- **❌ Failed Models (1):**
  - `mlx-community/MolmoPoint-8B-fp16` (`Processor Error`)
- **📝 Formatting Issues (4):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-Flash-mxfp4`
  - `mlx-community/GLM-4.6V-nvfp4`
  - `mlx-community/Idefics3-8B-Llama3-bf16`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 83.4 | Min: 4.98 | Max: 394
- **Peak Memory**: Avg: 21 | Min: 1.8 | Max: 78
- **Total Time**: Avg: 22.43s | Min: 1.69s | Max: 105.91s
- **Generation Time**: Avg: 19.93s | Min: 0.74s | Max: 102.50s
- **Model Load Time**: Avg: 2.17s | Min: 0.47s | Max: 10.63s

### ⏱ Runtime Interpretation

- **Runtime pattern:** decode dominates measured phase time (90%; 50/53 measured model(s)).
- **Phase totals:** model load=114.98s, prompt prep=0.15s, decode=1036.22s, cleanup=5.15s.
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.
- **Termination reasons:** completed=52, exception=1.

### ⏱ Timing Snapshot

- **Validation overhead:** 17.28s total (avg 0.33s across 53 model(s)).
- **First-token latency:** Avg 11.89s | Min 0.05s | Max 73.92s across 52 model(s).

## ✅ Recommended Models

Quick picks based on end-to-end utility plus description and keyword strength.

- _Best end-to-end cataloging:_ [`mlx-community/gemma-3-27b-it-qat-4bit`](model_gallery.md#model-mlx-community-gemma-3-27b-it-qat-4bit)
  (A 85/100 | Desc 82 | Keywords 92 | Gen 30.8 TPS | Peak 19 | A 85/100 |
  nontext prompt burden=99%)
- _Best descriptions:_ [`qnguyen3/nanoLLaVA`](model_gallery.md#model-qnguyen3-nanollava)
  (B 74/100 | Desc 90 | Keywords 0 | Gen 115 TPS | Peak 4.1 | B 74/100 |
  nontext prompt burden=80%)
- _Best keywording:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
  (B 75/100 | Desc 82 | Keywords 96 | Gen 66.2 TPS | Peak 13 | B 75/100 |
  nontext prompt burden=100%)
- _Fastest generation:_ [`mlx-community/nanoLLaVA-1.5-4bit`](model_gallery.md#model-mlx-community-nanollava-15-4bit)
  (B 75/100 | Desc 86 | Keywords 0 | Gen 394 TPS | Peak 1.8 | B 75/100 |
  nontext prompt burden=80%)
- _Lowest memory footprint:_ [`mlx-community/nanoLLaVA-1.5-4bit`](model_gallery.md#model-mlx-community-nanollava-15-4bit)
  (B 75/100 | Desc 86 | Keywords 0 | Gen 394 TPS | Peak 1.8 | B 75/100 |
  nontext prompt burden=80%)
- _Best balance:_ [`mlx-community/gemma-3-27b-it-qat-4bit`](model_gallery.md#model-mlx-community-gemma-3-27b-it-qat-4bit)
  (A 85/100 | Desc 82 | Keywords 92 | Gen 30.8 TPS | Peak 19 | A 85/100 |
  nontext prompt burden=99%)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (1):_ [`mlx-community/MolmoPoint-8B-fp16`](model_gallery.md#model-mlx-community-molmopoint-8b-fp16).
  Example: `Processor Error`.
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

_Overall runtime:_ 1174.46s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                    |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:----------------------------------|----------------:|
| `mlx-community/MolmoPoint-8B-fp16`                      |                   |                       |                |              |           |             |                  |      2.15s |       2.47s |                                   |    model-config |
| `mlx-community/gemma-3n-E2B-4bit`                       |               264 |                     4 |            268 |         1159 |      93.8 |           6 |            0.74s |      1.38s |       2.46s | ⚠️harness(prompt_template)        |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |                20 |                   124 |            144 |          255 |       394 |         1.8 |            0.89s |      0.47s |       1.69s |                                   |                 |
| `qnguyen3/nanoLLaVA`                                    |                20 |                    51 |             71 |          191 |       115 |         4.1 |            1.05s |      0.51s |       1.88s |                                   |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |                85 |                   117 |            202 |         1556 |       191 |         3.6 |            1.08s |      0.53s |       1.94s |                                   |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |               766 |                    41 |            807 |         3372 |      60.2 |         9.1 |            1.36s |      0.82s |       2.49s |                                   |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               275 |                   292 |            567 |         3394 |       333 |           3 |            1.38s |      0.49s |       2.19s |                                   |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,101 |                     3 |          4,104 |         3836 |      83.4 |         4.6 |            1.64s |      1.13s |       3.11s | ⚠️harness(long_context), ...      |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |                24 |                   445 |            469 |         80.1 |       358 |         2.2 |            2.49s |      0.58s |       3.41s |                                   |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               266 |                     5 |            271 |          146 |      9.28 |          64 |            2.91s |      5.98s |       9.22s | ⚠️harness(prompt_template)        |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |                95 |                   266 |            361 |          314 |       131 |         5.5 |            2.95s |      0.60s |       3.88s |                                   |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,326 |                    48 |          2,374 |         2198 |      32.7 |          19 |            3.15s |      1.85s |       5.35s | formatting                        |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               278 |                   267 |            545 |          717 |       117 |          17 |            3.17s |      2.33s |       5.86s |                                   |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |             2,629 |                   355 |          2,984 |         2650 |       187 |         7.9 |            3.44s |      0.92s |       4.69s |                                   |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             2,174 |                    74 |          2,248 |         1111 |        63 |         9.7 |            3.78s |      0.88s |       4.99s |                                   |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |             1,011 |                   345 |          1,356 |         1352 |       132 |          18 |            3.91s |      2.06s |       6.28s | reasoning-leak                    |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,195 |                   420 |          1,615 |         3178 |       127 |         5.5 |            4.26s |      0.60s |       5.18s |                                   |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             1,195 |                   420 |          1,615 |         3187 |       125 |         5.5 |            4.37s |      0.59s |       5.29s |                                   |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |             1,011 |                   296 |          1,307 |         1306 |      80.8 |          37 |            5.11s |      3.23s |       8.67s | reasoning-leak                    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,029 |                   135 |          1,164 |         1329 |      35.1 |          11 |            5.22s |      1.58s |       7.14s |                                   |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               272 |                   237 |            509 |          871 |      48.5 |          17 |            5.68s |      2.22s |       8.23s |                                   |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |             1,011 |                   500 |          1,511 |         1066 |       120 |          22 |            5.71s |      2.10s |       8.14s | reasoning-leak, token-cap, ...    |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             1,804 |                   125 |          1,929 |         1316 |      32.3 |          18 |            5.82s |      1.62s |       7.77s |                                   |                 |
| `mlx-community/pixtral-12b-8bit`                        |             2,887 |                   167 |          3,054 |         1653 |      39.1 |          16 |            6.53s |      1.79s |       8.65s |                                   |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,029 |                   118 |          1,147 |         2810 |      19.7 |          10 |            6.87s |      1.43s |       8.63s |                                   |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               278 |                   158 |            436 |          386 |      26.4 |          19 |            7.21s |      2.49s |      10.03s |                                   |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |             2,630 |                   311 |          2,941 |         1323 |      66.2 |          13 |            7.22s |      1.30s |       8.85s |                                   |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |             2,098 |                   135 |          2,233 |          704 |      31.9 |          22 |            7.76s |      1.96s |      10.06s | ⚠️harness(encoding), ...          |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |             2,630 |                   344 |          2,974 |         1319 |      63.1 |          14 |            7.99s |      1.33s |       9.67s |                                   |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |               766 |                   500 |          1,266 |         2771 |      57.1 |         9.1 |            9.48s |      0.86s |      10.67s | ⚠️harness(stop_token), ...        |                 |
| `mlx-community/pixtral-12b-bf16`                        |             2,887 |                   166 |          3,053 |         1810 |      19.8 |          28 |           10.48s |      2.51s |      13.31s |                                   |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             1,804 |                   344 |          2,148 |         2462 |      33.9 |          17 |           11.44s |      1.58s |      13.35s |                                   |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               273 |                   320 |            593 |          401 |      30.8 |          19 |           11.54s |      2.30s |      14.18s |                                   |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |             6,091 |                   500 |          6,591 |         1075 |      72.9 |         8.4 |           12.99s |      1.28s |      14.60s | ⚠️harness(stop_token), ...        |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             2,979 |                   500 |          3,479 |         1520 |      42.6 |          15 |           14.21s |      1.58s |      16.12s | degeneration, reasoning-leak, ... |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |             1,293 |                   500 |          1,793 |          239 |      60.5 |          60 |           14.46s |     10.21s |      25.02s | cutoff                            |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |             6,091 |                   500 |          6,591 |         1129 |      54.8 |          11 |           14.99s |      1.37s |      16.69s | ⚠️harness(stop_token), ...        |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,029 |                   100 |          1,129 |          938 |      5.44 |          26 |           20.02s |      2.42s |      22.77s |                                   |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |             1,207 |                   205 |          1,412 |         71.8 |      52.1 |          41 |           21.60s |      1.07s |      23.01s |                                   |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |                14 |                   421 |            435 |         10.4 |      21.2 |          15 |           21.65s |      1.58s |      23.57s |                                   |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |            16,299 |                   219 |         16,518 |          933 |      57.1 |          14 |           22.03s |      1.09s |      23.45s |                                   |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               273 |                   381 |            654 |          327 |      17.7 |          33 |           22.90s |      3.34s |      26.58s |                                   |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |             1,207 |                   205 |          1,412 |         61.7 |      30.4 |          48 |           27.12s |      1.59s |      29.04s |                                   |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |             6,091 |                   500 |          6,591 |          398 |      38.1 |          78 |           29.00s |      8.80s |      38.14s | ⚠️harness(stop_token), ...        |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |                15 |                   276 |            291 |         8.68 |      4.98 |          25 |           57.55s |      2.15s |      60.03s |                                   |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |            16,290 |                   500 |         16,790 |          299 |       105 |          26 |           60.04s |      2.45s |      62.83s | cutoff                            |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |            16,290 |                   500 |         16,790 |          303 |      90.6 |          12 |           60.10s |      1.39s |      61.83s | degeneration, token-cap           |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |            16,290 |                   500 |         16,790 |          299 |      88.4 |          35 |           60.87s |      3.11s |      64.32s | cutoff                            |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |            16,299 |                    13 |         16,312 |          267 |       207 |         5.1 |           61.72s |      0.55s |      62.59s | ⚠️harness(long_context), ...      |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |            16,290 |                   500 |         16,790 |          276 |        64 |          76 |           67.54s |     10.63s |      78.50s | token-cap                         |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |            16,290 |                   500 |         16,790 |          232 |      29.7 |          26 |           87.89s |      2.12s |      90.36s | cutoff                            |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |            16,290 |                   500 |         16,790 |          227 |      17.9 |          38 |          100.39s |      3.03s |     103.77s | cutoff                            |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |            16,290 |                   500 |         16,790 |          220 |        18 |          38 |          102.50s |      3.05s |     105.91s | cutoff                            |                 |

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
- `transformers`: `5.6.2`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-04-26 20:52:02 BST_
