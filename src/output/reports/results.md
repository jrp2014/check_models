# Model Performance Results

_Generated on 2026-06-18 23:07:28 BST_

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ 2 (top owners: mlx=1, mlx-vlm=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=7, mechanically clean
  outputs=43/58.
- _Useful now:_ 29 mechanically clean A/B model(s) with useful signals.
- _Review watchlist:_ 22 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Quality signal frequency:_ harness=7, cutoff=6, reasoning_leak=4,
  text_sanity=3, generation_loop=3, lang_mixing=2.

### Runtime

- _Runtime pattern:_ post-prefill decode dominates measured phase time (51%;
  19/60 measured model(s)).
- _Phase totals:_ model load=112.11s, local prompt prep=0.19s, upstream
  prefill / first-token=49.94s, post-prefill decode=178.39s, cleanup=6.24s.
- _Generation total:_ 228.33s across 58 model(s); upstream prefill /
  first-token split available for 58/58 model(s).
- _What this likely means:_ Most measured runtime is spent generating after
  the first token is available.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=50, exception=2, max_tokens=8.
- _Validation overhead:_ 0.11s total (avg 0.00s across 60 model(s)).
- _Upstream prefill / first-token latency:_ Avg 0.86s | Min 0.06s | Max 4.94s
  across 58 model(s).

## 🏆 Performance Highlights

- **Fastest:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (562.7 tps)
- **💾 Most efficient:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (1.0 GB)
- **⚡ Fastest load:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (0.32s)
- **📊 Average TPS:** 94.6 across 58 models

## 📈 Resource Usage

- **Input image size:** 0.31 MP
- **Average peak delta from post-load:** 3.12 GB
- **Peak memory delta / MP:** 10410 MB/MP
- **Average peak memory:** 17.9 GB
- **Memory efficiency:** 44 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** ✅ B: 34 | 🟡 C: 9 | 🟠 D: 5 | ❌ F: 10

**Average Utility Score:** 60/100

- **Best for cataloging:** `Qwen/Qwen3-VL-2B-Instruct` (✅ B, 75/100)
- **Best descriptions:** `mlx-community/gemma-4-31b-bf16` (93/100)
- **Best keywording:** `mlx-community/Ministral-3-3B-Instruct-2512-4bit` (73/100)
- **Worst for cataloging:** `mlx-community/paligemma2-3b-pt-896-4bit` (❌ F, 0/100)

### ⚠️ 15 Models with Low Utility (D/F)

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
- `mlx-community/SmolVLM-Instruct-bf16`: ❌ F (33/100) - Output lacks detail
- `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`: ❌ F (30/100) - Output lacks detail
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

- **Generation Tps**: Avg: 94.6 | Min: 4.71 | Max: 563
- **Peak Memory**: Avg: 18 | Min: 1.0 | Max: 71
- **Total Time**: Avg: 5.87s | Min: 0.44s | Max: 46.83s
- **Generation Time**: Avg: 3.94s | Min: 0.12s | Max: 43.56s
- **Model Load Time**: Avg: 1.93s | Min: 0.32s | Max: 8.21s

## ✅ Usable Diagnostic Candidates

Models that passed mechanical diagnostics and retained useful cataloguing or
metadata-alignment signals.

- _Best end-to-end cataloging:_ [`mlx-community/Qwen3.5-35B-A3B-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen35-35b-a3b-4bit)
  (Utility B 75/100 | Description 87 | Keywords 0 | Speed 117 TPS | Memory 21
  | Caveat nontext prompt burden=98%)
- _Best descriptions:_ [`mlx-community/Qwen3.5-35B-A3B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen35-35b-a3b-bf16)
  (Utility B 74/100 | Description 93 | Keywords 0 | Speed 69.5 TPS | Memory 71
  | Caveat nontext prompt burden=98%)
- _Best keywording:_ [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-ministral-3-3b-instruct-2512-4bit)
  (Utility B 70/100 | Description 74 | Keywords 73 | Speed 204 TPS | Memory
  4.4 | Caveat nontext prompt burden=99%)
- _Fastest generation:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16)
  (Utility F 34/100 | Description 75 | Keywords 0 | Speed 563 TPS | Memory 1.0
  | Caveat nontext prompt burden=92%)
- _Lowest memory footprint:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16)
  (Utility F 34/100 | Description 75 | Keywords 0 | Speed 563 TPS | Memory 1.0
  | Caveat nontext prompt burden=92%)
- _Best balance:_ [`Qwen/Qwen3-VL-2B-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-qwen-qwen3-vl-2b-instruct)
  (Utility B 75/100 | Description 81 | Keywords 0 | Speed 135 TPS | Memory 5.2
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
- _Low-utility outputs (15):_ [`HuggingFaceTB/SmolVLM-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16),
  [`microsoft/Phi-3.5-vision-instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-microsoft-phi-35-vision-instruct),
  [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit),
  +11 more. Common weakness: Output lacks detail.

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

_Overall runtime:_ 349.57s

<!-- markdownlint-disable MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Cached Tokens | Finish Reason   |   Diffusion Canvas Tokens | Diffusion Denoising Steps   |   Diffusion Work Tokens |   Diffusion Canvas Tps |   Diffusion Work Tps | Is Draft   | Draft Text   | Text Already Printed   | Diffusion Step   | Diffusion Total Steps   | Diffusion Canvas Index   | Diffusion Block Complete   |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                     |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|----------------:|:----------------|--------------------------:|:----------------------------|------------------------:|-----------------------:|---------------------:|:-----------|:-------------|:-----------------------|:-----------------|:------------------------|:-------------------------|:---------------------------|-----------------:|-----------:|------------:|:-----------------------------------|----------------:|
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |                   |                       |                |              |           |             |                 |                 |                           |                             |                         |                        |                      |            |              |                        |                  |                         |                          |                            |                  |      0.13s |       1.00s |                                    |             mlx |
| `mlx-community/MolmoPoint-8B-fp16`                      |                   |                       |                |              |           |             |                 |                 |                           |                             |                         |                        |                      |            |              |                        |                  |                         |                          |                            |                  |      0.15s |       1.00s |                                    |         mlx-vlm |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      |                80 |                    10 |             90 |          960 |       563 |           1 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.12s |      0.32s |       0.44s |                                    |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               269 |                    10 |            279 |         4167 |       323 |           3 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.12s |      0.47s |       0.59s |                                    |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |                26 |                    15 |             41 |          328 |       331 |         2.1 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.16s |      0.57s |       0.74s |                                    |                 |
| `mlx-community/MiniCPM-V-4.6-8bit`                      |               228 |                    22 |            250 |         2202 |       278 |           3 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.21s |      0.89s |       1.12s |                                    |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |                22 |                    81 |            103 |          301 |       377 |         1.9 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.30s |      0.48s |       0.79s |                                    |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |               417 |                    52 |            469 |         2510 |       323 |         2.5 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.35s |      0.55s |       0.91s |                                    |                 |
| `qnguyen3/nanoLLaVA`                                    |                22 |                    35 |             57 |          215 |       108 |           4 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.45s |      0.52s |       0.97s |                                    |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,196 |                    13 |          1,209 |         3183 |       132 |         5.5 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.52s |      0.70s |       1.24s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             1,196 |                    13 |          1,209 |         3405 |       128 |         5.5 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.53s |      0.56s |       1.09s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |               770 |                    19 |            789 |         3066 |      60.4 |         9.2 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.58s |      0.84s |       1.43s |                                    |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               286 |                    24 |            310 |          719 |       117 |          17 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.62s |      2.31s |       2.93s |                                    |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |               949 |                    99 |          1,048 |         3781 |       204 |         4.4 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.75s |      0.92s |       1.69s |                                    |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |               408 |                    42 |            450 |         1346 |      93.5 |         7.7 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.77s |      1.27s |       2.04s |                                    |                 |
| `mlx-community/Qwen3-VL-2B-Instruct-bf16`               |               315 |                    96 |            411 |         3786 |       134 |         5.3 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.84s |      0.76s |       1.60s |                                    |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |               315 |                    96 |            411 |         2538 |       135 |         5.2 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.86s |      0.67s |       1.54s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |               319 |                    63 |            382 |          804 |       117 |          21 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.95s |      2.43s |       3.39s |                                    |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |               319 |                    71 |            390 |         1449 |       101 |           7 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.98s |      1.36s |       2.34s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |               319 |                    58 |            377 |          712 |      99.2 |          30 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.04s |      3.05s |       4.10s |                                    |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,103 |                     3 |          4,106 |         3786 |      82.5 |         4.6 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.14s |      1.10s |       2.25s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`         |               286 |                    19 |            305 |          941 |      24.9 |          30 |               0 | stop            |                       200 | 5                           |                    1000 |                    262 |                1,309 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.23s |      3.23s |       4.47s |                                    |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |               417 |                    65 |            482 |         1791 |      66.9 |          10 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.27s |      1.17s |       2.44s |                                    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |               950 |                    52 |          1,002 |         1538 |      67.5 |          10 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.40s |      1.40s |       2.81s |                                    |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |               408 |                    69 |            477 |         1027 |      64.1 |          10 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.49s |      1.38s |       2.88s |                                    |                 |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit`          |               286 |                    24 |            310 |          926 |      23.5 |          30 |               0 | stop            |                       200 | 7                           |                    1400 |                    196 |                1,370 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.53s |      3.31s |       4.86s |                                    |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |               317 |                   196 |            513 |         3774 |       134 |         5.3 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.58s |      0.71s |       2.31s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |               950 |                    68 |          1,018 |         1516 |      70.2 |         9.7 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.61s |      1.37s |       2.99s |                                    |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               286 |                    25 |            311 |          390 |      28.9 |          19 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.61s |      2.51s |       4.13s |                                    |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,327 |                    23 |          2,350 |         2246 |      33.5 |          19 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.77s |      1.84s |       3.61s | formatting                         |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |                97 |                   187 |            284 |          345 |       127 |         5.5 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.82s |      0.56s |       2.39s |                                    |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               266 |                   200 |            466 |         1065 |       123 |         5.9 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.89s |      1.38s |       3.28s | repetitive(phrase: "have th...     |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |                16 |                    25 |             41 |         11.9 |      22.8 |          15 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            2.46s |      1.51s |       3.98s |                                    |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             3,341 |                    48 |          3,389 |         2748 |      34.4 |          18 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            2.72s |      1.53s |       4.26s |                                    |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |               319 |                    40 |            359 |          461 |      19.7 |          30 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            2.74s |      3.12s |       5.89s |                                    |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             2,356 |                    55 |          2,411 |         1236 |      63.8 |         9.7 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            2.79s |      0.87s |       3.67s |                                    |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |               319 |                    71 |            390 |          492 |      33.3 |          19 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            2.79s |      2.17s |       4.98s |                                    |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               274 |                   124 |            398 |          864 |      48.9 |          17 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            2.87s |      2.22s |       5.10s |                                    |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             3,341 |                    15 |          3,356 |         1351 |      33.4 |          19 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            2.95s |      1.73s |       4.69s |                                    |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |               399 |                   200 |            599 |         1422 |        77 |          16 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            2.97s |      1.90s |       4.88s | reasoning-leak, token-cap          |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |               399 |                   200 |            599 |         1385 |      72.8 |          20 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            3.17s |      2.09s |       5.27s | reasoning-leak, cutoff             |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |               319 |                    49 |            368 |          470 |      19.5 |          30 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            3.20s |      2.98s |       6.21s |                                    |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |               417 |                    90 |            507 |          637 |      33.2 |          19 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            3.38s |      2.02s |       5.41s | ⚠️harness(encoding)                |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               275 |                    91 |            366 |          392 |      31.9 |          19 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            3.58s |      2.25s |       5.83s |                                    |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |               770 |                   200 |            970 |         3642 |      58.2 |         9.2 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            3.72s |      0.87s |       4.59s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |               408 |                    71 |            479 |          174 |        52 |          63 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            3.72s |      5.60s |       9.33s |                                    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,031 |                   101 |          1,132 |         1328 |      35.2 |          11 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            3.73s |      1.63s |       5.37s |                                    |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |               439 |                   200 |            639 |          266 |      65.9 |          60 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            4.70s |      5.04s |       9.75s | generation_loop(degeneration), ... |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               275 |                    70 |            345 |          321 |        18 |          33 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            4.77s |      3.34s |       8.12s |                                    |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             1,330 |                   174 |          1,504 |         1447 |      44.4 |          14 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            4.85s |      1.53s |       6.40s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |               319 |                    55 |            374 |           75 |      69.5 |          71 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            5.06s |      8.21s |      13.28s |                                    |                 |
| `mlx-community/pixtral-12b-8bit`                        |             1,239 |                   200 |          1,439 |         1633 |      40.1 |          15 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            5.76s |      1.74s |       7.51s | cutoff                             |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |               745 |                    55 |            800 |          152 |      51.6 |          20 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            6.00s |      1.19s |       7.20s |                                    |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,031 |                   119 |          1,150 |         2741 |      19.7 |          10 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            6.43s |      1.47s |       7.91s |                                    |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |               745 |                    55 |            800 |          151 |      30.5 |          27 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            6.77s |      1.70s |       8.47s |                                    |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               274 |                    36 |            310 |          110 |      7.55 |          64 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            7.28s |      6.16s |      13.46s |                                    |                 |
| `mlx-community/pixtral-12b-bf16`                        |             1,239 |                   200 |          1,439 |         1551 |      20.4 |          27 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |           10.62s |      2.48s |      13.12s | cutoff                             |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |                17 |                    77 |             94 |         10.1 |      5.16 |          25 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |           16.64s |      2.15s |      18.80s |                                    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,031 |                   159 |          1,190 |          919 |       5.4 |          26 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |           30.61s |      2.41s |      33.03s |                                    |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |               399 |                   200 |            599 |          492 |      4.71 |          39 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |           43.56s |      3.26s |      46.83s | reasoning-leak, cutoff             |                 |

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
  (21,690,224 bytes,
  sha256=041e6c9d17ac3ea47f7c55ee230306c2a70ad4b2879ce2e5f48ac6a4d351a31e)

## Library Versions

- `numpy`: `2.4.6`
- `mlx`: `0.32.0.dev20260618+350f2cf8`
- `mlx-vlm`: `0.6.3`
- `mlx-lm`: `0.31.3`
- `mlx-audio`: `0.4.4`
- `huggingface-hub`: `1.20.0`
- `transformers`: `5.12.1`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-06-18 23:07:28 BST_
