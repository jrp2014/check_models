# Model Performance Results

_Generated on 2026-06-12 13:18:08 BST_

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ 3 (top owners: mlx=1, mlx-vlm=1,
  huggingface-hub=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=6, mechanically clean
  outputs=40/56.
- _Useful now:_ 28 mechanically clean A/B model(s) with useful signals.
- _Review watchlist:_ 22 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Quality signal frequency:_ harness=6, cutoff=6, generation_loop=5,
  reasoning_leak=4, degeneration=3, token_cap=3.

### Runtime

- _Runtime pattern:_ model load dominates measured phase time (86%; 35/59
  measured model(s)).
- _Phase totals:_ model load=1551.66s, local prompt prep=0.19s, upstream
  prefill / first-token=61.22s, post-prefill decode=178.49s, cleanup=6.25s.
- _Generation total:_ 239.71s across 56 model(s); upstream prefill /
  first-token split available for 56/56 model(s).
- _What this likely means:_ Cold model load time is a major share of runtime
  for this cohort.
- _Suggested next action:_ Consider staged runs, model reuse, or narrowing the
  model set before reruns.
- _Termination reasons:_ completed=46, exception=3, max_tokens=10.
- _Validation overhead:_ 0.11s total (avg 0.00s across 59 model(s)).
- _Upstream prefill / first-token latency:_ Avg 1.09s | Min 0.06s | Max 6.56s
  across 56 model(s).

## 🏆 Performance Highlights

- **Fastest:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (551.6 tps)
- **💾 Most efficient:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (1.0 GB)
- **⚡ Fastest load:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (0.36s)
- **📊 Average TPS:** 95.8 across 56 models

## 📈 Resource Usage

- **Input image size:** 0.31 MP
- **Average peak delta from post-load:** 3.06 GB
- **Peak memory delta / MP:** 10196 MB/MP
- **Average peak memory:** 17.5 GB
- **Memory efficiency:** 46 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** ✅ B: 32 | 🟡 C: 8 | 🟠 D: 5 | ❌ F: 11

**Average Utility Score:** 58/100

- **Best for cataloging:** `Qwen/Qwen3-VL-2B-Instruct` (✅ B, 75/100)
- **Best descriptions:** `mlx-community/gemma-4-31b-bf16` (93/100)
- **Best keywording:** `mlx-community/Ministral-3-3B-Instruct-2512-4bit` (73/100)
- **Worst for cataloging:** `mlx-community/paligemma2-3b-pt-896-4bit` (❌ F, 0/100)

### ⚠️ 16 Models with Low Utility (D/F)

- `HuggingFaceTB/SmolVLM-Instruct`: ❌ F (33/100) - Output lacks detail
- `LiquidAI/LFM2.5-VL-450M-MLX-bf16`: ❌ F (34/100) - Output lacks detail
- `microsoft/Phi-3.5-vision-instruct`: 🟠 D (35/100) - Output lacks detail
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`: 🟠 D (47/100) - Keywords are not specific or diverse enough
- `mlx-community/FastVLM-0.5B-bf16`: ❌ F (31/100) - Output lacks detail
- `mlx-community/Idefics3-8B-Llama3-bf16`: ❌ F (30/100) - Keywords are not specific or diverse enough
- `mlx-community/InternVL3-14B-8bit`: ❌ F (31/100) - Output lacks detail
- `mlx-community/LFM2-VL-1.6B-8bit`: ❌ F (34/100) - Output lacks detail
- `mlx-community/MiniCPM-V-4.6-8bit`: 🟠 D (47/100) - Keywords are not specific or diverse enough
- `mlx-community/Phi-3.5-vision-instruct-bf16`: 🟠 D (35/100) - Output lacks detail
- `mlx-community/Qwen3-VL-2B-Instruct-bf16`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Qwen3-VL-2B-Thinking-bf16`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/SmolVLM-Instruct-bf16`: ❌ F (33/100) - Output lacks detail
- `mlx-community/gemma-3n-E2B-4bit`: 🟠 D (50/100) - Keywords are not specific or diverse enough
- `mlx-community/paligemma2-3b-pt-896-4bit`: ❌ F (0/100) - Empty or minimal output

## ⚠️ Quality Issues

- **❌ Failed Models (3):**
  - `mlx-community/LFM2.5-VL-1.6B-bf16` (`Weight Mismatch`)
  - `mlx-community/MolmoPoint-8B-fp16` (`Model Error`)
  - `mlx-community/diffusiongemma-26B-A4B-it-8bit` (`Model Error`)
- **🔄 Repetitive Output (1):**
  - `mlx-community/gemma-3n-E2B-4bit` (token: `phrase: "have this image. have..."`)
- **📝 Formatting Issues (1):**
  - `mlx-community/Idefics3-8B-Llama3-bf16`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 95.8 | Min: 4.7 | Max: 552
- **Peak Memory**: Avg: 17 | Min: 0.97 | Max: 71
- **Total Time**: Avg: 6.43s | Min: 0.51s | Max: 47.06s
- **Generation Time**: Avg: 4.28s | Min: 0.11s | Max: 43.74s
- **Model Load Time**: Avg: 2.14s | Min: 0.36s | Max: 10.17s

## ✅ Usable Diagnostic Candidates

Models that passed mechanical diagnostics and retained useful cataloguing or
metadata-alignment signals.

- _Best end-to-end cataloging:_ [`mlx-community/Qwen3.5-35B-A3B-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen35-35b-a3b-4bit)
  (Utility B 75/100 | Description 87 | Keywords 0 | Speed 114 TPS | Memory 21
  | Caveat nontext prompt burden=98%)
- _Best descriptions:_ [`mlx-community/Qwen3.5-35B-A3B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen35-35b-a3b-bf16)
  (Utility B 74/100 | Description 93 | Keywords 0 | Speed 67.6 TPS | Memory 71
  | Caveat nontext prompt burden=98%)
- _Best keywording:_ [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-ministral-3-3b-instruct-2512-4bit)
  (Utility B 70/100 | Description 74 | Keywords 73 | Speed 194 TPS | Memory
  4.4 | Caveat nontext prompt burden=99%)
- _Fastest generation:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16)
  (Utility F 34/100 | Description 75 | Keywords 0 | Speed 552 TPS | Memory
  0.97 | Caveat nontext prompt burden=92%)
- _Lowest memory footprint:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16)
  (Utility F 34/100 | Description 75 | Keywords 0 | Speed 552 TPS | Memory
  0.97 | Caveat nontext prompt burden=92%)
- _Best balance:_ [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-smolvlm2-22b-instruct-mlx)
  (Utility B 75/100 | Description 85 | Keywords 0 | Speed 132 TPS | Memory 5.5
  | Caveat nontext prompt burden=94%)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (3):_ [`mlx-community/LFM2.5-VL-1.6B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-lfm25-vl-16b-bf16),
  [`mlx-community/MolmoPoint-8B-fp16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-molmopoint-8b-fp16),
  [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-diffusiongemma-26b-a4b-it-8bit).
  Example: `Weight Mismatch`.
- _🔄 Repetitive Output (1):_ [`mlx-community/gemma-3n-E2B-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-gemma-3n-e2b-4bit).
  Example: token: `phrase: "have this image. have..."`.
- _📝 Formatting Issues (1):_ [`mlx-community/Idefics3-8B-Llama3-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16).
- _Low-utility outputs (16):_ [`HuggingFaceTB/SmolVLM-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16),
  [`microsoft/Phi-3.5-vision-instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-microsoft-phi-35-vision-instruct),
  [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit),
  +12 more. Common weakness: Output lacks detail.

## 🚨 Failures by Package (Actionable)

| Package           |   Failures | Error Types     | Affected Models                                |
|-------------------|------------|-----------------|------------------------------------------------|
| `mlx`             |          1 | Weight Mismatch | `mlx-community/LFM2.5-VL-1.6B-bf16`            |
| `mlx-vlm`         |          1 | Model Error     | `mlx-community/MolmoPoint-8B-fp16`             |
| `huggingface-hub` |          1 | Model Error     | `mlx-community/diffusiongemma-26B-A4B-it-8bit` |

### Actionable Items by Package

#### mlx

- mlx-community/LFM2.5-VL-1.6B-bf16 (Weight Mismatch)
  - Error: `Model loading failed: Missing 2 parameters: <br>multi_modal_projector.layer_norm.bias,<br>multi_modal_projector.layer_norm....`
  - Type: `ValueError`

#### mlx-vlm

- mlx-community/MolmoPoint-8B-fp16 (Model Error)
  - Error: `Model loading failed: property 'eos_token_id' of 'ModelConfig' object has no setter`
  - Type: `ValueError`

#### huggingface-hub

- mlx-community/diffusiongemma-26B-A4B-it-8bit (Model Error)
  - Error: `Model loading failed: Operation timed out after 300.0 seconds`
  - Type: `ValueError`

_Prompt used:_

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Describe this image briefly.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 1801.57s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Cached Tokens | Finish Reason   |   Diffusion Canvas Tokens | Diffusion Denoising Steps   |   Diffusion Work Tokens |   Diffusion Canvas Tps |   Diffusion Work Tps | Is Draft   | Draft Text   | Text Already Printed   | Diffusion Step   | Diffusion Total Steps   | Diffusion Canvas Index   | Diffusion Block Complete   |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                     |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|----------------:|:----------------|--------------------------:|:----------------------------|------------------------:|-----------------------:|---------------------:|:-----------|:-------------|:-----------------------|:-----------------|:------------------------|:-------------------------|:---------------------------|-----------------:|-----------:|------------:|:-----------------------------------|----------------:|
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |                   |                       |                |              |           |             |                 |                 |                           |                             |                         |                        |                      |            |              |                        |                  |                         |                          |                            |                  |      0.17s |       1.06s |                                    |             mlx |
| `mlx-community/MolmoPoint-8B-fp16`                      |                   |                       |                |              |           |             |                 |                 |                           |                             |                         |                        |                      |            |              |                        |                  |                         |                          |                            |                  |      0.25s |       1.11s |                                    |         mlx-vlm |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit`          |                   |                       |                |              |           |             |                 |                 |                           |                             |                         |                        |                      |            |              |                        |                  |                         |                          |                            |                  |   1431.29s |    1432.22s |                                    | huggingface-hub |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               269 |                    10 |            279 |         4659 |       322 |           3 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.11s |      0.57s |       0.69s |                                    |                 |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      |                80 |                    10 |             90 |          730 |       552 |        0.97 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.14s |      0.36s |       0.51s |                                    |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |                26 |                    15 |             41 |          366 |       332 |         2.1 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.15s |      0.66s |       0.82s |                                    |                 |
| `mlx-community/MiniCPM-V-4.6-8bit`                      |               228 |                    22 |            250 |         2287 |       273 |           3 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.22s |      0.96s |       1.19s |                                    |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |                22 |                    81 |            103 |          321 |       369 |         1.8 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.30s |      0.47s |       0.78s |                                    |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |               417 |                    52 |            469 |         2859 |       319 |         2.5 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.33s |      0.57s |       0.91s |                                    |                 |
| `qnguyen3/nanoLLaVA`                                    |                22 |                    35 |             57 |          220 |       114 |         3.9 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.43s |      0.61s |       1.05s |                                    |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             1,196 |                    13 |          1,209 |         3388 |       127 |         5.5 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.53s |      0.94s |       1.48s | ⚠️harness(prompt_template)         |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,196 |                    13 |          1,209 |         3142 |       129 |         5.5 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.53s |      0.78s |       1.32s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |               770 |                    19 |            789 |         2994 |      60.4 |         9.2 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.59s |      0.98s |       1.57s |                                    |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               286 |                    24 |            310 |          689 |       116 |          17 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.63s |      2.35s |       3.01s |                                    |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |               408 |                    42 |            450 |         1353 |      92.9 |         7.7 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.77s |      1.26s |       2.03s |                                    |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |               949 |                    99 |          1,048 |         4185 |       194 |         4.4 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.77s |      0.92s |       1.70s |                                    |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |               315 |                    95 |            410 |         3388 |       129 |         4.7 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.86s |      0.83s |       1.71s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |               319 |                    63 |            382 |          846 |       114 |          21 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.94s |      2.64s |       3.59s |                                    |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |               319 |                    71 |            390 |         1206 |      99.7 |           7 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.99s |      1.67s |       2.68s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |               319 |                    58 |            377 |          691 |      94.5 |          30 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.09s |      3.12s |       4.22s |                                    |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,103 |                     3 |          4,106 |         3672 |        76 |         4.6 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.18s |      1.16s |       2.35s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |               417 |                    65 |            482 |         1783 |      66.4 |          10 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.28s |      1.19s |       2.47s |                                    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |               950 |                    52 |          1,002 |         1519 |      66.3 |          10 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.42s |      1.35s |       2.78s |                                    |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |               408 |                    69 |            477 |         1102 |      63.4 |          10 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.47s |      1.41s |       2.90s |                                    |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |               317 |                   200 |            517 |         3900 |       133 |           5 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.62s |      0.73s |       2.36s | generation_loop(degeneration), ... |                 |
| `mlx-community/Qwen3-VL-2B-Instruct-bf16`               |               315 |                   200 |            515 |         4080 |       133 |           5 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.62s |      0.76s |       2.39s | generation_loop(degeneration), ... |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |               950 |                    68 |          1,018 |         1503 |      69.8 |         9.7 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.62s |      1.32s |       2.96s |                                    |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               286 |                    25 |            311 |          383 |      28.6 |          19 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.64s |      2.54s |       4.18s |                                    |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |                97 |                   187 |            284 |          342 |       132 |         5.5 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.77s |      0.65s |       2.42s |                                    |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,327 |                    23 |          2,350 |         2197 |      33.9 |          19 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.79s |      1.96s |       3.76s | formatting                         |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               266 |                   200 |            466 |         1128 |       122 |           6 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.89s |      1.40s |       3.30s | repetitive(phrase: "have th...     |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |                16 |                    25 |             41 |         11.6 |      22.4 |          15 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            2.51s |      1.57s |       4.10s |                                    |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             3,341 |                    48 |          3,389 |         2529 |      34.3 |          18 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            2.75s |      1.59s |       4.35s |                                    |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |               319 |                    71 |            390 |          504 |      33.4 |          19 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            2.78s |      2.14s |       4.94s |                                    |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |               319 |                    40 |            359 |          441 |      19.4 |          30 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            2.80s |      3.21s |       6.03s |                                    |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             2,356 |                    55 |          2,411 |         1253 |      62.7 |         9.7 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            2.83s |      0.88s |       3.71s |                                    |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               274 |                   124 |            398 |          790 |      48.4 |          17 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            2.92s |      2.18s |       5.11s |                                    |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             3,341 |                    15 |          3,356 |         1334 |        33 |          19 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            2.99s |      1.65s |       4.65s |                                    |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |               399 |                   200 |            599 |         1355 |      76.8 |          16 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            3.03s |      2.12s |       5.16s | reasoning-leak, token-cap          |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |               319 |                    49 |            368 |          462 |      19.4 |          30 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            3.24s |      3.18s |       6.44s |                                    |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |               399 |                   200 |            599 |         1378 |      70.4 |          20 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            3.27s |      2.32s |       5.59s | reasoning-leak, cutoff             |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |               417 |                    90 |            507 |          605 |      32.7 |          19 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            3.45s |      2.21s |       5.68s | ⚠️harness(encoding)                |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               275 |                    91 |            366 |          375 |      31.7 |          19 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            3.63s |      2.32s |       5.96s |                                    |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |               770 |                   200 |            970 |         3791 |      57.9 |         9.2 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            3.73s |      1.04s |       4.78s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,031 |                   101 |          1,132 |         1315 |        34 |          11 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            3.84s |      1.59s |       5.44s |                                    |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               275 |                    70 |            345 |          297 |      17.7 |          33 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            4.90s |      3.34s |       8.25s |                                    |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             1,330 |                   174 |          1,504 |         1435 |      43.1 |          14 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            4.98s |      1.58s |       6.57s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |               745 |                    55 |            800 |          185 |      50.9 |          20 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            5.13s |      1.24s |       6.38s |                                    |                 |
| `mlx-community/pixtral-12b-8bit`                        |             1,239 |                   200 |          1,439 |         1579 |      39.9 |          15 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            5.81s |      1.66s |       7.49s | cutoff                             |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |               745 |                    55 |            800 |          167 |      30.8 |          27 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            6.26s |      1.84s |       8.11s |                                    |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,031 |                   119 |          1,150 |         2672 |      19.6 |          11 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            6.49s |      1.46s |       7.96s |                                    |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |               408 |                    71 |            479 |         73.1 |      49.7 |          63 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            7.02s |      8.18s |      15.22s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |               319 |                    55 |            374 |         49.9 |      67.6 |          71 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            7.21s |     10.17s |      17.39s |                                    |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               274 |                    36 |            310 |         58.9 |      7.62 |          64 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            9.40s |      7.90s |      17.30s |                                    |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |               439 |                   200 |            639 |         66.9 |        64 |          60 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            9.70s |      9.61s |      19.32s | generation_loop(degeneration), ... |                 |
| `mlx-community/pixtral-12b-bf16`                        |             1,239 |                   200 |          1,439 |         1521 |      20.4 |          27 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |           10.62s |      2.53s |      13.16s | cutoff                             |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |                17 |                    77 |             94 |         8.76 |      5.09 |          25 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |           17.08s |      2.47s |      19.57s |                                    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,031 |                   159 |          1,190 |          901 |      5.35 |          26 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |           30.90s |      2.50s |      33.41s |                                    |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |               399 |                   200 |            599 |          431 |       4.7 |          39 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |           43.74s |      3.32s |      47.06s | reasoning-leak, cutoff             |                 |

<!-- markdownlint-enable MD013 MD033 MD034 MD037 MD049 -->

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md)
- _Automated review digest:_ [review.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/review.md)
- _Canonical run log:_ [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log)

---

## System/Hardware Information

- _OS:_ Darwin 25.5.0
- _macOS Version:_ 26.5.1
- _SDK Version:_ 26.5
- _Active Developer Directory:_ /Applications/Xcode.app/Contents/Developer
- _Xcode Version:_ 26.5
- _Xcode Build:_ 17F42
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
- _MLX Distribution Root:_ /Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages
- _mlx-metal Distribution:_ not installed; local editable mlx supplies backend
- _MLX Core Extension:_ /Users/jrp/Documents/AI/mlx/mlx/python/mlx/core.cpython-313-darwin.so
- _MLX Metallib:_ /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib
  (157,751,704 bytes,
  sha256=ba9913d81d92bbbde42bbc6dda27e80ecb31db6031fa073e6c8aeb0666d47c33)
- _MLX libmlx.dylib:_ /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib
  (21,676,160 bytes,
  sha256=811f9557132c55cc5b95a4dcbdb6e3757ed88cfa5bcfabf8b3561959383335a5)

## Library Versions

- `numpy`: `2.4.6`
- `mlx`: `0.32.0.dev20260612+337f736a`
- `mlx-vlm`: `0.6.3`
- `mlx-lm`: `0.31.3`
- `mlx-audio`: `0.4.4`
- `huggingface-hub`: `1.19.0`
- `transformers`: `5.11.0`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-06-12 13:18:08 BST_
