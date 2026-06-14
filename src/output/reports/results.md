# Model Performance Results

_Generated on 2026-06-14 00:54:53 BST_

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ 2 (top owners: mlx=1, mlx-vlm=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=7, mechanically clean
  outputs=41/57.
- _Useful now:_ 28 mechanically clean A/B model(s) with useful signals.
- _Review watchlist:_ 23 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Quality signal frequency:_ harness=7, cutoff=6, generation_loop=4,
  reasoning_leak=4, text_sanity=3, degeneration=2.

### Runtime

- _Runtime pattern:_ post-prefill decode dominates measured phase time (46%;
  21/59 measured model(s)).
- _Phase totals:_ model load=139.91s, local prompt prep=0.21s, upstream
  prefill / first-token=75.94s, post-prefill decode=188.14s, cleanup=8.75s.
- _Generation total:_ 264.08s across 57 model(s); upstream prefill /
  first-token split available for 57/57 model(s).
- _What this likely means:_ Most measured runtime is spent generating after
  the first token is available.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=48, exception=2, max_tokens=9.
- _Validation overhead:_ 0.14s total (avg 0.00s across 59 model(s)).
- _Upstream prefill / first-token latency:_ Avg 1.33s | Min 0.02s | Max 8.92s
  across 57 model(s).

## 🏆 Performance Highlights

- **Fastest:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (553.3 tps)
- **💾 Most efficient:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (1.0 GB)
- **⚡ Fastest load:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (0.32s)
- **📊 Average TPS:** 85.5 across 57 models

## 📈 Resource Usage

- **Input image size:** 0.31 MP
- **Average peak delta from post-load:** 3.08 GB
- **Peak memory delta / MP:** 10277 MB/MP
- **Average peak memory:** 17.7 GB
- **Memory efficiency:** 45 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** ✅ B: 33 | 🟡 C: 8 | 🟠 D: 6 | ❌ F: 10

**Average Utility Score:** 59/100

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
- `mlx-community/SmolVLM-Instruct-bf16`: ❌ F (33/100) - Output lacks detail
- `mlx-community/diffusiongemma-26B-A4B-it-8bit`: 🟠 D (48/100) - Keywords are not specific or diverse enough
- `mlx-community/gemma-3n-E2B-4bit`: 🟠 D (50/100) - Keywords are not specific or diverse enough
- `mlx-community/paligemma2-3b-pt-896-4bit`: ❌ F (0/100) - Empty or minimal output

## ⚠️ Quality Issues

- **❌ Failed Models (2):**
  - `mlx-community/LFM2.5-VL-1.6B-bf16` (`Weight Mismatch`)
  - `mlx-community/MolmoPoint-8B-fp16` (`Model Error`)
- **🔄 Repetitive Output (1):**
  - `mlx-community/gemma-3n-E2B-4bit` (token: `phrase: "have this image. have..."`)
- **📝 Formatting Issues (1):**
  - `mlx-community/Idefics3-8B-Llama3-bf16`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 85.5 | Min: 4.75 | Max: 553
- **Peak Memory**: Avg: 18 | Min: 1.0 | Max: 71
- **Total Time**: Avg: 7.10s | Min: 0.39s | Max: 46.88s
- **Generation Time**: Avg: 4.63s | Min: 0.07s | Max: 43.27s
- **Model Load Time**: Avg: 2.45s | Min: 0.32s | Max: 13.19s

## ✅ Usable Diagnostic Candidates

Models that passed mechanical diagnostics and retained useful cataloguing or
metadata-alignment signals.

- _Best end-to-end cataloging:_ [`mlx-community/Qwen3.5-35B-A3B-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen35-35b-a3b-4bit)
  (Utility B 75/100 | Description 87 | Keywords 0 | Speed 108 TPS | Memory 21
  | Caveat nontext prompt burden=98%)
- _Best descriptions:_ [`mlx-community/Qwen3.5-35B-A3B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen35-35b-a3b-bf16)
  (Utility B 74/100 | Description 93 | Keywords 0 | Speed 57.4 TPS | Memory 71
  | Caveat nontext prompt burden=98%)
- _Best keywording:_ [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-ministral-3-3b-instruct-2512-4bit)
  (Utility B 70/100 | Description 74 | Keywords 73 | Speed 188 TPS | Memory
  4.4 | Caveat nontext prompt burden=99%)
- _Fastest generation:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16)
  (Utility F 34/100 | Description 75 | Keywords 0 | Speed 553 TPS | Memory 1.0
  | Caveat nontext prompt burden=92%)
- _Lowest memory footprint:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16)
  (Utility F 34/100 | Description 75 | Keywords 0 | Speed 553 TPS | Memory 1.0
  | Caveat nontext prompt burden=92%)
- _Best balance:_ [`Qwen/Qwen3-VL-2B-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-qwen-qwen3-vl-2b-instruct)
  (Utility B 75/100 | Description 81 | Keywords 0 | Speed 134 TPS | Memory 5.2
  | Caveat nontext prompt burden=98%)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (2):_ [`mlx-community/LFM2.5-VL-1.6B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-lfm25-vl-16b-bf16),
  [`mlx-community/MolmoPoint-8B-fp16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-molmopoint-8b-fp16).
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

| Package   |   Failures | Error Types     | Affected Models                     |
|-----------|------------|-----------------|-------------------------------------|
| `mlx`     |          1 | Weight Mismatch | `mlx-community/LFM2.5-VL-1.6B-bf16` |
| `mlx-vlm` |          1 | Model Error     | `mlx-community/MolmoPoint-8B-fp16`  |

### Actionable Items by Package

#### mlx

- mlx-community/LFM2.5-VL-1.6B-bf16 (Weight Mismatch)
  - Error: `Model loading failed: Missing 2 parameters: <br>multi_modal_projector.layer_norm.bias,<br>multi_modal_projector.layer_norm....`
  - Type: `ValueError`

#### mlx-vlm

- mlx-community/MolmoPoint-8B-fp16 (Model Error)
  - Error: `Model loading failed: property 'eos_token_id' of 'ModelConfig' object has no setter`
  - Type: `ValueError`

_Prompt used:_

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Describe this image briefly.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 416.13s

<!-- markdownlint-disable MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Cached Tokens | Finish Reason   |   Diffusion Canvas Tokens | Diffusion Denoising Steps   |   Diffusion Work Tokens |   Diffusion Canvas Tps |   Diffusion Work Tps | Is Draft   | Draft Text   | Text Already Printed   | Diffusion Step   | Diffusion Total Steps   | Diffusion Canvas Index   | Diffusion Block Complete   |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                     |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|----------------:|:----------------|--------------------------:|:----------------------------|------------------------:|-----------------------:|---------------------:|:-----------|:-------------|:-----------------------|:-----------------|:------------------------|:-------------------------|:---------------------------|-----------------:|-----------:|------------:|:-----------------------------------|----------------:|
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |                   |                       |                |              |           |             |                 |                 |                           |                             |                         |                        |                      |            |              |                        |                  |                         |                          |                            |                  |      0.13s |       1.14s |                                    |             mlx |
| `mlx-community/MolmoPoint-8B-fp16`                      |                   |                       |                |              |           |             |                 |                 |                           |                             |                         |                        |                      |            |              |                        |                  |                         |                          |                            |                  |      0.17s |       1.13s |                                    |         mlx-vlm |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      |                80 |                    10 |             90 |         3448 |       553 |           1 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.07s |      0.32s |       0.39s |                                    |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               269 |                    10 |            279 |         3588 |       295 |           3 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.15s |      0.66s |       0.81s |                                    |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |                26 |                    15 |             41 |          302 |       271 |         2.1 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.19s |      0.74s |       0.95s |                                    |                 |
| `mlx-community/MiniCPM-V-4.6-8bit`                      |               228 |                    22 |            250 |         2121 |       251 |           3 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.25s |      1.05s |       1.31s |                                    |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |                22 |                    81 |            103 |          317 |       323 |         1.8 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.33s |      0.48s |       0.82s |                                    |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |               417 |                    52 |            469 |         2989 |       226 |         2.5 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.40s |      0.60s |       1.01s |                                    |                 |
| `qnguyen3/nanoLLaVA`                                    |                22 |                    35 |             57 |          197 |       108 |         3.8 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.47s |      0.63s |       1.10s |                                    |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,196 |                    13 |          1,209 |         3060 |       125 |         5.5 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.55s |      0.64s |       1.20s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |               770 |                    19 |            789 |         2678 |      57.9 |         9.2 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.64s |      0.90s |       1.54s |                                    |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               286 |                    24 |            310 |          657 |       103 |          17 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.68s |      2.63s |       3.33s |                                    |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |               949 |                    99 |          1,048 |         3713 |       188 |         4.4 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.82s |      0.95s |       1.79s |                                    |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |               408 |                    42 |            450 |         1223 |        87 |         7.7 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.83s |      1.60s |       2.44s |                                    |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |               315 |                    96 |            411 |         3479 |       134 |         5.2 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.84s |      0.65s |       1.50s |                                    |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             1,196 |                    13 |          1,209 |         2103 |       104 |         5.5 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.86s |      0.69s |       1.57s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |               319 |                    63 |            382 |          704 |       108 |          21 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.04s |      2.59s |       3.64s |                                    |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |               319 |                    71 |            390 |         1111 |      95.5 |           7 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.05s |      1.67s |       2.73s |                                    |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |               417 |                    65 |            482 |         1387 |      63.9 |          10 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.34s |      1.19s |       2.54s |                                    |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,103 |                     3 |          4,106 |         3057 |      69.2 |         4.6 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.41s |      1.44s |       2.87s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |               408 |                    69 |            477 |         1080 |        60 |          10 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.55s |      1.52s |       3.08s |                                    |                 |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit`          |               286 |                    18 |            304 |          901 |      17.7 |          30 |               0 | stop            |                       200 | 6                           |                    1200 |                    197 |                1,183 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.56s |      3.55s |       5.14s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |               319 |                    58 |            377 |          320 |      88.4 |          30 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.66s |      3.44s |       5.11s |                                    |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               286 |                    25 |            311 |          344 |        28 |          19 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.74s |      2.99s |       4.75s |                                    |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,327 |                    23 |          2,350 |         2201 |      33.6 |          19 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.79s |      1.89s |       3.69s | formatting                         |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |               950 |                    52 |          1,002 |          999 |      58.6 |          10 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.85s |      1.43s |       3.29s |                                    |                 |
| `mlx-community/Qwen3-VL-2B-Instruct-bf16`               |               315 |                   200 |            515 |         3648 |       109 |         5.2 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.97s |      0.77s |       2.75s | generation_loop(degeneration), ... |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |                97 |                   187 |            284 |          292 |       119 |         5.5 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.99s |      0.69s |       2.69s |                                    |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |               317 |                   196 |            513 |         3032 |       102 |         4.8 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            2.07s |      0.87s |       2.97s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               266 |                   200 |            466 |         1046 |       109 |         5.8 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            2.09s |      1.48s |       3.58s | repetitive(phrase: "have th...     |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |               950 |                    68 |          1,018 |         1014 |      58.5 |         9.7 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            2.12s |      1.87s |       4.01s |                                    |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             3,341 |                    15 |          3,356 |         1383 |      32.3 |          19 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            2.91s |      1.60s |       4.52s |                                    |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             3,341 |                    48 |          3,389 |         2215 |      33.9 |          18 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            2.96s |      1.64s |       4.61s |                                    |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |               399 |                   200 |            599 |         1123 |      78.7 |          16 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            3.08s |      2.36s |       5.45s | reasoning-leak, token-cap          |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               274 |                   124 |            398 |          769 |      43.9 |          17 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            3.20s |      2.19s |       5.40s |                                    |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |               319 |                    71 |            390 |          405 |      28.9 |          19 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            3.27s |      2.45s |       5.75s |                                    |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |               417 |                    90 |            507 |          594 |      32.6 |          19 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            3.47s |      2.02s |       5.51s | ⚠️harness(encoding)                |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |                16 |                    25 |             41 |         8.15 |      16.1 |          15 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            3.54s |      1.58s |       5.13s |                                    |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |               319 |                    40 |            359 |          226 |      18.4 |          30 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            3.60s |      3.73s |       7.35s |                                    |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |               319 |                    49 |            368 |          424 |        17 |          30 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            3.66s |      3.12s |       6.82s |                                    |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |               399 |                   200 |            599 |         1340 |        61 |          20 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            3.75s |      2.28s |       6.05s | reasoning-leak, cutoff             |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |               770 |                   200 |            970 |         3752 |        57 |         9.2 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            3.80s |      0.91s |       4.71s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             2,356 |                    55 |          2,411 |          814 |      58.6 |         9.7 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            3.91s |      0.87s |       4.79s |                                    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,031 |                   101 |          1,132 |         1186 |      31.6 |          11 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            4.17s |      1.63s |       5.82s |                                    |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               275 |                    91 |            366 |          323 |      24.7 |          19 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            4.56s |      2.54s |       7.11s |                                    |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |               745 |                    55 |            800 |          202 |      46.8 |          20 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            4.89s |      1.35s |       6.24s |                                    |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               275 |                    70 |            345 |          280 |      17.5 |          33 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            5.00s |      3.74s |       8.75s |                                    |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             1,330 |                   174 |          1,504 |         1345 |      42.9 |          14 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            5.06s |      1.73s |       6.80s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |               745 |                    55 |            800 |          196 |      29.5 |          27 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            5.70s |      1.76s |       7.46s |                                    |                 |
| `mlx-community/pixtral-12b-8bit`                        |             1,239 |                   200 |          1,439 |         1308 |      39.5 |          15 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            6.02s |      1.66s |       7.69s | cutoff                             |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,031 |                   119 |          1,150 |         2341 |      17.7 |          10 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            7.20s |      1.91s |       9.13s |                                    |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |               408 |                    71 |            479 |         58.7 |      41.8 |          63 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            8.66s |     11.40s |      20.08s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |               319 |                    55 |            374 |         36.6 |      57.4 |          71 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            9.68s |     13.19s |      22.89s |                                    |                 |
| `mlx-community/pixtral-12b-bf16`                        |             1,239 |                   200 |          1,439 |         1418 |        20 |          27 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |           10.87s |      2.51s |      13.39s | cutoff                             |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |               439 |                   200 |            639 |         49.2 |      56.2 |          60 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |           12.49s |     12.18s |      24.68s | generation_loop(degeneration), ... |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               274 |                    36 |            310 |         33.2 |      6.83 |          64 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |           13.56s |     10.56s |      24.14s |                                    |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |                17 |                    77 |             94 |         7.17 |      5.05 |          25 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |           17.64s |      2.69s |      20.34s |                                    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,031 |                   159 |          1,190 |          877 |      5.21 |          26 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |           31.84s |      2.50s |      34.37s |                                    |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |               399 |                   200 |            599 |          461 |      4.75 |          39 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |           43.27s |      3.59s |      46.88s | reasoning-leak, cutoff             |                 |

<!-- markdownlint-enable MD033 MD034 MD037 MD049 -->

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
  (21,671,712 bytes,
  sha256=3a48ca2ae7659130de59374b0c50f6ba11b1dfecb8f6a2549f1b72fb41ac921c)

## Library Versions

- `numpy`: `2.4.6`
- `mlx`: `0.32.0.dev20260614+89064477`
- `mlx-vlm`: `0.6.3`
- `mlx-lm`: `0.31.3`
- `mlx-audio`: `0.4.4`
- `huggingface-hub`: `1.19.0`
- `transformers`: `5.12.0`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-06-14 00:54:53 BST_
