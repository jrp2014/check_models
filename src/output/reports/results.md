# Model Performance Results

_Generated on 2026-05-02 00:10:46 BST_

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ 3 (top owners: mlx=1, model-config=1,
  huggingface-hub=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=9, clean outputs=29/50.
- _Useful now:_ 29 clean A/B model(s) worth first review.
- _Review watchlist:_ 19 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Quality signal frequency:_ harness=9, cutoff=7, reasoning_leak=6,
  formatting=4, long_context=2, context_budget=2.

### Runtime

- _Runtime pattern:_ decode dominates measured phase time (80%; 48/53 measured
  model(s)).
- _Phase totals:_ model load=276.81s, prompt prep=0.15s, decode=1143.26s,
  cleanup=5.56s.
- _What this likely means:_ Most measured runtime is spent inside generation
  rather than load or prompt setup.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=50, exception=3.
- _Validation overhead:_ 17.40s total (avg 0.33s across 53 model(s)).
- _First-token latency:_ Avg 13.28s | Min 0.06s | Max 78.49s across 50
  model(s).

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/FastVLM-0.5B-bf16` (355.0 tps)
- **💾 Most efficient:** `mlx-community/nanoLLaVA-1.5-4bit` (1.8 GB)
- **⚡ Fastest load:** `mlx-community/nanoLLaVA-1.5-4bit` (0.45s)
- **📊 Average TPS:** 76.2 across 50 models

## 📈 Resource Usage

- **Total peak memory:** 1062.1 GB
- **Average peak memory:** 21.2 GB
- **Memory efficiency:** 212 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 1 | ✅ B: 41 | 🟡 C: 3 | ❌ F: 5

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

- **❌ Failed Models (3):**
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit` (`Model Error`)
  - `mlx-community/MolmoPoint-8B-fp16` (`Processor Error`)
  - `mlx-community/SmolVLM2-2.2B-Instruct-mlx` (`Model Error`)
- **📝 Formatting Issues (4):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-Flash-mxfp4`
  - `mlx-community/GLM-4.6V-nvfp4`
  - `mlx-community/Idefics3-8B-Llama3-bf16`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 76.2 | Min: 4.71 | Max: 355
- **Peak Memory**: Avg: 21 | Min: 1.8 | Max: 78
- **Total Time**: Avg: 27.60s | Min: 1.73s | Max: 112.97s
- **Generation Time**: Avg: 22.87s | Min: 0.73s | Max: 108.03s
- **Model Load Time**: Avg: 4.40s | Min: 0.45s | Max: 76.40s

## ✅ Recommended Models

Quick picks based on end-to-end utility plus description and keyword strength.

- _Best end-to-end cataloging:_ [`mlx-community/gemma-3-27b-it-qat-4bit`](model_gallery.md#model-mlx-community-gemma-3-27b-it-qat-4bit)
  (Utility A 85/100 | Description 82 | Keywords 92 | Speed 27.1 TPS | Memory
  19 | Caveat nontext prompt burden=99%)
- _Best descriptions:_ [`qnguyen3/nanoLLaVA`](model_gallery.md#model-qnguyen3-nanollava)
  (Utility B 74/100 | Description 90 | Keywords 0 | Speed 114 TPS | Memory 3.8
  | Caveat nontext prompt burden=80%)
- _Best keywording:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
  (Utility B 75/100 | Description 82 | Keywords 96 | Speed 65.9 TPS | Memory
  13 | Caveat nontext prompt burden=100%)
- _Fastest generation:_ [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16)
  (Utility B 74/100 | Description 82 | Keywords 0 | Speed 355 TPS | Memory 2.1
  | Caveat nontext prompt burden=83%)
- _Lowest memory footprint:_ [`mlx-community/nanoLLaVA-1.5-4bit`](model_gallery.md#model-mlx-community-nanollava-15-4bit)
  (Utility B 75/100 | Description 86 | Keywords 0 | Speed 342 TPS | Memory 1.8
  | Caveat nontext prompt burden=80%)
- _Best balance:_ [`mlx-community/gemma-3-27b-it-qat-4bit`](model_gallery.md#model-mlx-community-gemma-3-27b-it-qat-4bit)
  (Utility A 85/100 | Description 82 | Keywords 92 | Speed 27.1 TPS | Memory
  19 | Caveat nontext prompt burden=99%)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (3):_ [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](model_gallery.md#model-mlx-community-kimi-vl-a3b-thinking-8bit),
  [`mlx-community/MolmoPoint-8B-fp16`](model_gallery.md#model-mlx-community-molmopoint-8b-fp16),
  [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](model_gallery.md#model-mlx-community-smolvlm2-22b-instruct-mlx).
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
| `huggingface-hub` | 1 | Model Error | `mlx-community/SmolVLM2-2.2B-Instruct-mlx` |

### Actionable Items by Package

#### mlx

- mlx-community/Kimi-VL-A3B-Thinking-8bit (Model Error)
  - Error: `Model loading failed: Received 4 parameters not in model: <br>multi_modal_projector.linear_1.biases,<br>multi_modal_project...`
  - Type: `ValueError`

#### model-config

- mlx-community/MolmoPoint-8B-fp16 (Processor Error)
  - Error: `Model preflight failed for mlx-community/MolmoPoint-8B-fp16: Loaded processor has no image_processor; expected multim...`
  - Type: `ValueError`

#### huggingface-hub

- mlx-community/SmolVLM2-2.2B-Instruct-mlx (Model Error)
  - Error: `Model loading failed: Server disconnected without sending a response.`
  - Type: `ValueError`

_Prompt used:_

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Describe this picture
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 1443.87s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                    |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:----------------------------------|----------------:|
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |                   |                       |                |              |           |             |                  |      0.24s |       0.57s |                                   |             mlx |
| `mlx-community/MolmoPoint-8B-fp16`                      |                   |                       |                |              |           |             |                  |      2.23s |       2.56s |                                   |    model-config |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |                   |                       |                |              |           |             |                  |     54.25s |      54.58s |                                   | huggingface-hub |
| `mlx-community/gemma-3n-E2B-4bit`                       |               264 |                     4 |            268 |         1142 |      92.7 |           6 |            0.73s |     76.40s |      77.47s | ⚠️harness(prompt_template)        |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |                20 |                   124 |            144 |          252 |       342 |         1.8 |            0.95s |      0.45s |       1.73s |                                   |                 |
| `qnguyen3/nanoLLaVA`                                    |                20 |                    51 |             71 |          176 |       114 |         3.8 |            1.07s |      0.58s |       1.98s |                                   |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |                85 |                   117 |            202 |         1343 |       182 |         3.5 |            1.12s |      0.62s |       2.08s |                                   |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               275 |                   292 |            567 |         3299 |       324 |           3 |            1.42s |      0.46s |       2.22s |                                   |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |               768 |                    53 |            821 |         3355 |      60.2 |         9.4 |            1.56s |      0.85s |       2.72s |                                   |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,101 |                     3 |          4,104 |         3813 |      83.9 |         4.6 |            1.65s |      1.16s |       3.15s | ⚠️harness(long_context), ...      |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |                24 |                   445 |            469 |          259 |       355 |         2.1 |            2.32s |      0.72s |       3.40s |                                   |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,326 |                    48 |          2,374 |         2274 |      32.8 |          19 |            3.10s |      1.90s |       5.34s | formatting                        |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               278 |                   267 |            545 |          713 |       113 |          17 |            3.24s |      2.33s |       5.90s |                                   |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               266 |                     5 |            271 |          115 |      9.16 |          64 |            3.40s |      6.35s |      10.07s | ⚠️harness(prompt_template)        |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |             2,629 |                   355 |          2,984 |         2697 |       187 |         7.9 |            3.41s |      0.94s |       4.69s |                                   |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             2,174 |                    74 |          2,248 |         1136 |      62.7 |         9.7 |            3.75s |      0.96s |       5.03s |                                   |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,195 |                   420 |          1,615 |         3106 |       128 |         5.5 |            4.26s |      0.75s |       5.32s |                                   |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             1,195 |                   420 |          1,615 |         3167 |       126 |         5.5 |            4.34s |      4.13s |       8.79s |                                   |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,029 |                   135 |          1,164 |         1300 |      33.8 |          11 |            5.41s |      1.55s |       7.31s |                                   |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               272 |                   237 |            509 |          875 |      48.3 |          17 |            5.66s |      2.17s |       8.15s |                                   |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             1,804 |                   125 |          1,929 |         1347 |      32.2 |          18 |            5.78s |      1.78s |       7.89s |                                   |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |             1,011 |                   357 |          1,368 |         1222 |      71.4 |          18 |            6.38s |      2.15s |       8.87s | reasoning-leak                    |                 |
| `mlx-community/pixtral-12b-8bit`                        |             2,887 |                   167 |          3,054 |         1760 |        39 |          16 |            6.45s |      1.80s |       8.58s |                                   |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,029 |                   118 |          1,147 |         2707 |      19.7 |          10 |            6.91s |      1.55s |       8.80s |                                   |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               278 |                   158 |            436 |          366 |      27.2 |          19 |            7.07s |      2.62s |      10.03s |                                   |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |             2,630 |                   311 |          2,941 |         1337 |      65.9 |          13 |            7.23s |      1.41s |       8.98s |                                   |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |             2,630 |                   344 |          2,974 |         1324 |      63.1 |          14 |            7.99s |      1.61s |       9.93s |                                   |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |             2,097 |                   172 |          2,269 |          724 |      31.4 |          22 |            8.93s |      2.09s |      11.35s | ⚠️harness(encoding), ...          |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |               768 |                   500 |          1,268 |         3129 |      56.8 |         9.3 |            9.51s |      1.03s |      10.87s | ⚠️harness(stop_token), ...        |                 |
| `mlx-community/pixtral-12b-bf16`                        |             2,887 |                   166 |          3,053 |         1775 |      20.4 |          28 |           10.28s |      2.64s |      13.25s |                                   |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             1,804 |                   344 |          2,148 |         2471 |      34.5 |          17 |           11.27s |      2.07s |      13.67s |                                   |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |             1,293 |                   500 |          1,793 |          480 |      59.6 |          60 |           11.83s |      6.25s |      18.42s | cutoff                            |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               273 |                   320 |            593 |          358 |      27.1 |          19 |           13.05s |      6.34s |      19.74s |                                   |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             2,979 |                   500 |          3,479 |         1492 |      42.5 |          15 |           14.27s |      1.77s |      16.38s | degeneration, reasoning-leak, ... |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |             6,091 |                   500 |          6,591 |          888 |      67.6 |         8.4 |           14.73s |      1.36s |      16.43s | ⚠️harness(stop_token), ...        |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |             6,091 |                   500 |          6,591 |          941 |      48.4 |          11 |           17.26s |      1.50s |      19.10s | ⚠️harness(stop_token), ...        |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,029 |                   100 |          1,129 |          930 |      5.42 |          26 |           20.11s |      2.51s |      22.96s |                                   |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |            16,299 |                   219 |         16,518 |         1033 |        53 |          14 |           20.66s |     10.13s |      31.10s |                                   |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |                14 |                   421 |            435 |         10.4 |      21.3 |          15 |           21.60s |      1.52s |      23.45s |                                   |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |             1,207 |                   205 |          1,412 |         67.2 |      51.8 |          41 |           22.74s |      1.26s |      24.33s |                                   |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               273 |                   381 |            654 |          325 |      17.6 |          33 |           22.96s |     22.54s |      45.87s |                                   |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |             1,207 |                   205 |          1,412 |         62.1 |      30.5 |          48 |           26.98s |      1.75s |      29.06s |                                   |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |             6,091 |                   500 |          6,591 |          381 |        35 |          78 |           30.81s |      7.20s |      38.34s | ⚠️harness(stop_token), ...        |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |                15 |                   276 |            291 |         8.81 |         5 |          25 |           57.41s |      2.27s |      60.01s |                                   |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |             1,011 |                   277 |          1,288 |          767 |      4.71 |          39 |           60.79s |      3.32s |      64.44s | reasoning-leak                    |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |            16,299 |                    13 |         16,312 |          261 |       209 |         5.1 |           63.23s |      0.49s |      64.05s | ⚠️harness(long_context), ...      |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |            16,290 |                   500 |         16,790 |          266 |      87.2 |          12 |           67.74s |      1.49s |      69.58s | degeneration, token-cap           |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |            16,290 |                   500 |         16,790 |          264 |      91.8 |          26 |           68.01s |      2.60s |      70.94s | cutoff                            |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |            16,290 |                   500 |         16,790 |          253 |      77.6 |          35 |           71.78s |      3.51s |      75.64s | cutoff                            |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |            16,290 |                   500 |         16,790 |          239 |      55.4 |          76 |           78.24s |      9.19s |      87.78s | token-cap                         |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |            16,290 |                   500 |         16,790 |          230 |      28.5 |          26 |           89.16s |      2.31s |      91.82s | cutoff                            |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |            16,290 |                   500 |         16,790 |          211 |      17.4 |          38 |          106.72s |      3.12s |     110.19s | cutoff                            |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |            16,290 |                   500 |         16,790 |          208 |      17.4 |          38 |          108.03s |      4.58s |     112.97s | cutoff                            |                 |

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

_Report generated on: 2026-05-02 00:10:46 BST_
