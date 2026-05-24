# Model Performance Results

_Generated on 2026-05-24 21:46:43 BST_

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ 3 (top owners: mlx=2, mlx-lm=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=9, mechanically clean
  outputs=30/53.
- _Useful now:_ 33 mechanically clean A/B model(s) with useful signals.
- _Review watchlist:_ 18 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Quality signal frequency:_ harness=9, cutoff=9, reasoning_leak=6,
  generation_loop=5, token_cap=5, formatting=4.

### Runtime

- _Runtime pattern:_ upstream prefill / first-token dominates measured phase
  time (60%; 17/56 measured model(s)).
- _Phase totals:_ model load=127.69s, local prompt prep=0.17s, upstream
  prefill / first-token=668.00s, post-prefill decode=298.02s, generation total
  (unsplit)=10.08s, cleanup=6.69s.
- _Generation total:_ 976.10s across 53 model(s); upstream prefill /
  first-token split available for 51/53 model(s).
- _What this likely means:_ Most measured runtime is spent inside upstream
  generation before the first token is available.
- _Suggested next action:_ Inspect prompt/image token accounting,
  dynamic-resolution image burden, prefill step sizing, and cache/prefill
  behavior.
- _Termination reasons:_ completed=37, exception=3, max_tokens=16.
- _Validation overhead:_ 16.64s total (avg 0.30s across 56 model(s)).
- _Upstream prefill / first-token latency:_ Avg 13.10s | Min 0.11s | Max
  83.25s across 51 model(s).

## 🏆 Performance Highlights

- **Fastest:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (491.7 tps)
- **💾 Most efficient:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (1.3 GB)
- **⚡ Fastest load:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (0.35s)
- **📊 Average TPS:** 79.6 across 53 models

## 📈 Resource Usage

- **Total peak memory:** 1086.6 GB
- **Average peak memory:** 20.5 GB
- **Memory efficiency:** 202 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 1 | ✅ B: 45 | 🟡 C: 2 | 🟠 D: 1 | ❌ F: 4

**Average Utility Score:** 70/100

- **Best for cataloging:** `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` (🏆 A, 85/100)
- **Best descriptions:** `mlx-community/gemma-4-31b-it-4bit` (100/100)
- **Best keywording:** `mlx-community/Qwen3.5-9B-MLX-4bit` (93/100)
- **Worst for cataloging:** `mlx-community/gemma-3n-E2B-4bit` (❌ F, 0/100)

### ⚠️ 5 Models with Low Utility (D/F)

- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: 🟠 D (50/100) - Keywords are not specific or diverse enough
- `mlx-community/gemma-3n-E2B-4bit`: ❌ F (0/100) - Empty or minimal output
- `mlx-community/gemma-4-31b-bf16`: ❌ F (0/100) - Empty or minimal output
- `mlx-community/paligemma2-3b-pt-896-4bit`: ❌ F (34/100) - Output lacks detail

## ⚠️ Quality Issues

- **❌ Failed Models (3):**
  - `facebook/pe-av-large` (`Model Error`)
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit` (`Model Error`)
  - `mlx-community/LFM2.5-VL-1.6B-bf16` (`Weight Mismatch`)
- **🔄 Repetitive Output (1):**
  - `mlx-community/Qwen2-VL-2B-Instruct-4bit` (token: `phrase: "answered by answered by..."`)
- **👻 Hallucinations (1):**
  - `microsoft/Phi-3.5-vision-instruct`
- **📝 Formatting Issues (4):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-Flash-mxfp4`
  - `mlx-community/GLM-4.6V-nvfp4`
  - `mlx-community/Idefics3-8B-Llama3-bf16`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 79.6 | Min: 0 | Max: 492
- **Peak Memory**: Avg: 21 | Min: 1.3 | Max: 78
- **Total Time**: Avg: 21.13s | Min: 1.45s | Max: 98.83s
- **Generation Time**: Avg: 18.42s | Min: 0.83s | Max: 95.43s
- **Model Load Time**: Avg: 2.40s | Min: 0.35s | Max: 11.94s

## ✅ Usable Diagnostic Candidates

Models that passed mechanical diagnostics and retained useful cataloguing or
metadata-alignment signals.

- _Best end-to-end cataloging:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
  (Utility A 85/100 | Description 78 | Keywords 87 | Speed 66.2 TPS | Memory
  12 | Caveat nontext prompt burden=100%)
- _Best descriptions:_ [`mlx-community/gemma-4-31b-it-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-gemma-4-31b-it-4bit)
  (Utility B 71/100 | Description 100 | Keywords 0 | Speed 26.9 TPS | Memory
  19 | Caveat nontext prompt burden=98%)
- _Best keywording:_ [`mlx-community/Qwen3.5-9B-MLX-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen35-9b-mlx-4bit)
  (Utility B 80/100 | Description 83 | Keywords 93 | Speed 88.1 TPS | Memory
  12 | Caveat hit token cap (200); nontext prompt burden=100%)
- _Fastest generation:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16)
  (Utility B 75/100 | Description 87 | Keywords 0 | Speed 492 TPS | Memory 1.3
  | Caveat nontext prompt burden=98%)
- _Lowest memory footprint:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16)
  (Utility B 75/100 | Description 87 | Keywords 0 | Speed 492 TPS | Memory 1.3
  | Caveat nontext prompt burden=98%)
- _Best balance:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
  (Utility A 85/100 | Description 78 | Keywords 87 | Speed 66.2 TPS | Memory
  12 | Caveat nontext prompt burden=100%)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (3):_ [`facebook/pe-av-large`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-facebook-pe-av-large),
  [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-kimi-vl-a3b-thinking-8bit),
  [`mlx-community/LFM2.5-VL-1.6B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-lfm25-vl-16b-bf16).
  Example: `Model Error`.
- _🔄 Repetitive Output (1):_ [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen2-vl-2b-instruct-4bit).
  Example: token: `phrase: "answered by answered by..."`.
- _👻 Hallucinations (1):_ [`microsoft/Phi-3.5-vision-instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-microsoft-phi-35-vision-instruct).
- _📝 Formatting Issues (4):_ [`mlx-community/GLM-4.6V-Flash-6bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/GLM-4.6V-Flash-mxfp4`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4),
  [`mlx-community/GLM-4.6V-nvfp4`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-nvfp4),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16).
- _Low-utility outputs (5):_ [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit),
  [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen2-vl-2b-instruct-4bit),
  [`mlx-community/gemma-3n-E2B-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-gemma-3n-e2b-4bit),
  [`mlx-community/gemma-4-31b-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-gemma-4-31b-bf16),
  +1 more. Common weakness: Output too short to be useful.

## 🚨 Failures by Package (Actionable)

| Package   |   Failures | Error Types                  | Affected Models                                                                |
|-----------|------------|------------------------------|--------------------------------------------------------------------------------|
| `mlx`     |          2 | Model Error, Weight Mismatch | `mlx-community/Kimi-VL-A3B-Thinking-8bit`, `mlx-community/LFM2.5-VL-1.6B-bf16` |
| `mlx-lm`  |          1 | Model Error                  | `facebook/pe-av-large`                                                         |

### Actionable Items by Package

#### mlx

- mlx-community/Kimi-VL-A3B-Thinking-8bit (Model Error)
  - Error: `Model loading failed: Received 4 parameters not in model: <br>multi_modal_projector.linear_1.biases,<br>multi_modal_project...`
  - Type: `ValueError`
- mlx-community/LFM2.5-VL-1.6B-bf16 (Weight Mismatch)
  - Error: `Model loading failed: Missing 2 parameters: <br>multi_modal_projector.layer_norm.bias,<br>multi_modal_projector.layer_norm....`
  - Type: `ValueError`

#### mlx-lm

- facebook/pe-av-large (Model Error)
  - Error: `Model loading failed: Model type pe_audio_video not supported.`
  - Type: `ValueError`

_Prompt used:_

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Describe this image briefly.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 1131.01s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Cached Tokens |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                     |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|----------------:|-----------------:|-----------:|------------:|:-----------------------------------|----------------:|
| `facebook/pe-av-large`                                  |                   |                       |                |              |           |             |                 |                  |      0.11s |       1.22s |                                    |          mlx-lm |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |                   |                       |                |              |           |             |                 |                  |      0.17s |       1.30s |                                    |             mlx |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |                   |                       |                |              |           |             |                 |                  |      0.21s |       1.41s |                                    |             mlx |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      |               269 |                    62 |            331 |          767 |       492 |         1.3 |               0 |            0.83s |      0.35s |       1.45s |                                    |                 |
| `qnguyen3/nanoLLaVA`                                    |                22 |                    52 |             74 |          187 |       114 |           4 |               0 |            1.00s |      0.54s |       1.82s |                                    |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |                22 |                    93 |            115 |          202 |       252 |         1.8 |               0 |            1.01s |      0.63s |       2.05s |                                    |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               269 |                   106 |            375 |          707 |       333 |           3 |               0 |            1.04s |      0.46s |       1.77s |                                    |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |                 0 |                     0 |              0 |            0 |         0 |         5.8 |               0 |            1.38s |      1.68s |       3.57s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/MiniCPM-V-4.6-8bit`                      |               616 |                    40 |            656 |          918 |       215 |         3.3 |               0 |            1.39s |      0.99s |       2.68s |                                    |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |              4103 |                     9 |           4112 |         3595 |      63.5 |         4.6 |               0 |            1.73s |      1.13s |       3.15s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               284 |                    64 |            348 |          379 |       112 |          17 |               0 |            1.75s |      2.46s |       4.52s |                                    |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |                26 |                    21 |             47 |           30 |       348 |         1.8 |               0 |            1.82s |      0.59s |       2.70s |                                    |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |               770 |                    92 |            862 |         3098 |      58.6 |         9.2 |               0 |            2.19s |      0.90s |       3.36s |                                    |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |              1196 |                   123 |           1319 |         2951 |       113 |         5.5 |               0 |            2.30s |      1.69s |       4.38s |                                    |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |              2277 |                   189 |           2466 |         2655 |       182 |         6.4 |               0 |            2.35s |      0.91s |       3.56s |                                    |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |              2327 |                    31 |           2358 |         2179 |      33.2 |          19 |               0 |            2.54s |      1.90s |       4.74s | formatting                         |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |                97 |                   192 |            289 |          291 |       121 |         5.5 |               0 |            2.58s |      0.58s |       3.48s |                                    |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               284 |                    43 |            327 |          365 |      26.9 |          19 |               0 |            2.86s |      2.77s |       5.94s |                                    |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |              1196 |                   123 |           1319 |          606 |       124 |         5.5 |               0 |            3.52s |      0.67s |       4.45s |                                    |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |              1964 |                    70 |           2034 |          789 |      58.5 |         9.7 |               0 |            4.31s |      1.37s |       5.96s |                                    |                 |
| `mlx-community/InternVL3-14B-8bit`                      |              2317 |                    58 |           2375 |         1120 |      32.4 |          18 |               0 |            4.35s |      1.73s |       6.36s |                                    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |              2278 |                   145 |           2423 |         1211 |      66.2 |          12 |               0 |            4.52s |      1.28s |       6.11s |                                    |                 |
| `mlx-community/InternVL3-8B-bf16`                       |              2317 |                   110 |           2427 |         2562 |      34.4 |          18 |               0 |            4.58s |      1.69s |       6.54s |                                    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |              2278 |                   151 |           2429 |         1101 |      63.5 |          12 |               0 |            4.90s |      1.36s |       6.54s |                                    |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |               770 |                   200 |            970 |          696 |      57.1 |         9.2 |               0 |            4.99s |      0.93s |       6.19s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               274 |                   200 |            474 |          654 |        44 |          17 |               0 |            5.41s |      2.59s |       8.32s | cutoff                             |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               275 |                   106 |            381 |          303 |      25.9 |          19 |               0 |            5.47s |      2.73s |       8.51s |                                    |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |              1033 |                   200 |           1233 |          449 |      73.6 |          18 |               0 |            5.65s |      2.67s |       8.59s | reasoning-leak, token-cap          |                 |
| `mlx-community/pixtral-12b-8bit`                        |              2349 |                   152 |           2501 |         1663 |      36.9 |          15 |               0 |            5.99s |      1.75s |       8.02s |                                    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |              1031 |                   137 |           1168 |          956 |      29.1 |          11 |               0 |            6.42s |      1.82s |       8.57s |                                    |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |              2440 |                   200 |           2640 |         1287 |      42.9 |          14 |               0 |            7.01s |      1.59s |       8.90s | generation_loop(degeneration), ... |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |              1745 |                   162 |           1907 |          677 |        32 |          21 |               0 |            8.11s |      2.05s |      10.44s | ⚠️harness(encoding), ...           |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |                16 |                   127 |            143 |         10.1 |        20 |          15 |               0 |            8.34s |      1.60s |      10.23s |                                    |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               275 |                   109 |            384 |          277 |      15.4 |          33 |               0 |            8.53s |      3.96s |      12.96s |                                    |                 |
| `mlx-community/gemma-4-31b-bf16`                        |                 0 |                     0 |              0 |            0 |         0 |          64 |               0 |            8.70s |     11.94s |      20.93s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |              6045 |                   200 |           6245 |         1033 |      72.5 |         8.4 |               0 |            9.05s |      1.35s |      10.70s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |              6045 |                   174 |           6219 |         1024 |      50.7 |          11 |               0 |            9.73s |      1.40s |      11.41s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |              1031 |                   189 |           1220 |         2588 |      19.4 |          10 |               0 |           10.58s |      1.49s |      12.36s |                                    |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |              1270 |                   200 |           1470 |          192 |      59.1 |          60 |               0 |           10.68s |     10.17s |      21.29s | generation_loop(degeneration), ... |                 |
| `mlx-community/pixtral-12b-bf16`                        |              2349 |                   200 |           2549 |         1750 |      20.4 |          27 |               0 |           11.58s |      2.52s |      14.38s | token-cap                          |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |              1201 |                    94 |           1295 |         66.5 |      50.2 |          41 |               0 |           20.76s |      1.34s |      22.37s |                                    |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |              6045 |                   189 |           6234 |          393 |        38 |          78 |               0 |           20.84s |      8.32s |      29.45s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |              1201 |                    84 |           1285 |         72.1 |      17.1 |          48 |               0 |           22.30s |      1.71s |      24.29s |                                    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |              1031 |                   121 |           1152 |          842 |      5.24 |          26 |               0 |           24.85s |      2.70s |      27.99s |                                    |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |             16346 |                    71 |          16417 |          719 |      52.6 |          14 |               0 |           25.09s |      1.31s |      26.78s |                                    |                 |
| `mlx-community/MolmoPoint-8B-fp16`                      |              2818 |                   131 |           2949 |          718 |      5.98 |          26 |               0 |           26.51s |      2.21s |      29.10s |                                    |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |                17 |                   146 |            163 |          5.6 |      4.98 |          25 |               0 |           32.85s |      2.80s |      35.93s |                                    |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |              1033 |                   200 |           1233 |          702 |      4.55 |          39 |               0 |           46.03s |      3.27s |      49.58s | reasoning-leak, cutoff             |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |             16337 |                   200 |          16537 |          292 |      97.4 |          26 |               0 |           58.67s |      2.47s |      61.42s | cutoff                             |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |             16337 |                   200 |          16537 |          294 |      85.1 |          35 |               0 |           58.69s |      3.12s |      62.09s | token-cap                          |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |             16337 |                   200 |          16537 |          289 |      88.1 |          12 |               0 |           59.46s |      1.32s |      61.07s | token-cap                          |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |             16337 |                   200 |          16537 |          269 |      63.7 |          76 |               0 |           64.60s |     11.63s |      76.52s | cutoff                             |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |             16346 |                   200 |          16546 |          243 |       196 |         5.1 |               0 |           68.88s |      0.48s |      69.64s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |             16337 |                   200 |          16537 |          215 |      29.1 |          26 |               0 |           83.59s |      2.12s |      86.01s | cutoff                             |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |             16337 |                   200 |          16537 |          214 |      17.8 |          39 |               0 |           88.37s |      3.09s |      91.75s | cutoff                             |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |             16337 |                   200 |          16537 |          196 |      17.4 |          39 |               0 |           95.43s |      3.09s |      98.83s | token-cap                          |                 |

<!-- markdownlint-enable MD013 MD033 MD034 MD037 MD049 -->

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md)
- _Automated review digest:_ [review.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/review.md)
- _Canonical run log:_ [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log)

---

## System/Hardware Information

- _OS:_ Darwin 25.5.0
- _macOS Version:_ 26.5
- _SDK Version:_ 26.5
- _Xcode Version:_ 26.5
- _Xcode Build:_ 17F42
- _Metal SDK:_ MacOSX.sdk
- _Python Version:_ 3.13.13
- _Architecture:_ arm64
- _GPU/Chip:_ Apple M5 Max
- _GPU Cores:_ 40
- _Metal Support:_ Metal 4
- _RAM:_ 128.0 GB
- _CPU Cores (Physical):_ 18
- _CPU Cores (Logical):_ 18

## Library Versions

- `numpy`: `2.4.6`
- `mlx`: `0.31.2`
- `mlx-metal`: `0.31.2`
- `mlx-vlm`: `0.5.0`
- `mlx-lm`: `0.31.3`
- `mlx-audio`: `0.4.3`
- `huggingface-hub`: `1.16.1`
- `transformers`: `5.9.0`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-05-24 21:46:43 BST_
