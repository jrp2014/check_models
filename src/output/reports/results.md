# Model Performance Results

_Generated on 2026-06-05 13:31:25 BST_

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ 3 (top owners: mlx=2, mlx-lm=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=10, mechanically clean
  outputs=30/53.
- _Useful now:_ 25 mechanically clean A/B model(s) with useful signals.
- _Review watchlist:_ 20 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Quality signal frequency:_ harness=10, cutoff=7, reasoning_leak=6,
  formatting=4, text_sanity=4, token_cap=4.

### Runtime

- _Runtime pattern:_ upstream prefill / first-token dominates measured phase
  time (65%; 20/56 measured model(s)).
- _Phase totals:_ model load=109.95s, local prompt prep=0.18s, upstream
  prefill / first-token=699.06s, post-prefill decode=262.01s, cleanup=6.07s.
- _Generation total:_ 961.08s across 53 model(s); upstream prefill /
  first-token split available for 53/53 model(s).
- _What this likely means:_ Most measured runtime is spent inside upstream
  generation before the first token is available.
- _Suggested next action:_ Inspect prompt/image token accounting,
  dynamic-resolution image burden, prefill step sizing, and cache/prefill
  behavior.
- _Termination reasons:_ completed=39, exception=3, max_tokens=14.
- _Validation overhead:_ 7.04s total (avg 0.13s across 56 model(s)).
- _Upstream prefill / first-token latency:_ Avg 13.19s | Min 0.12s | Max
  85.90s across 53 model(s).

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/gemma-4-31b-bf16` (52980.1 tps)
- **💾 Most efficient:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (1.3 GB)
- **⚡ Fastest load:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (0.35s)
- **📊 Average TPS:** 1988.2 across 53 models

## 📈 Resource Usage

- **Input image size:** 66.45 MP
- **Average peak delta from post-load:** 5.61 GB
- **Peak memory delta / MP:** 86 MB/MP
- **Average peak memory:** 20.6 GB
- **Memory efficiency:** 202 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** ✅ B: 37 | 🟡 C: 8 | ❌ F: 8

**Average Utility Score:** 63/100

- **Best for cataloging:** `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX` (✅ B, 80/100)
- **Best descriptions:** `mlx-community/InternVL3-8B-bf16` (93/100)
- **Best keywording:** `mlx-community/Qwen3.5-9B-MLX-4bit` (93/100)
- **Worst for cataloging:** `mlx-community/Qwen2-VL-2B-Instruct-4bit` (❌ F, 0/100)

### ⚠️ 8 Models with Low Utility (D/F)

- `LiquidAI/LFM2.5-VL-450M-MLX-bf16`: ❌ F (17/100) - Output lacks detail
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/FastVLM-0.5B-bf16`: ❌ F (31/100) - Output lacks detail
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (0/100) - Empty or minimal output
- `mlx-community/gemma-3n-E2B-4bit`: ❌ F (0/100) - Empty or minimal output
- `mlx-community/gemma-4-31b-bf16`: ❌ F (0/100) - Empty or minimal output
- `mlx-community/llava-v1.6-mistral-7b-8bit`: ❌ F (35/100) - Output lacks detail
- `mlx-community/paligemma2-3b-pt-896-4bit`: ❌ F (0/100) - Empty or minimal output

## ⚠️ Quality Issues

- **❌ Failed Models (3):**
  - `facebook/pe-av-large` (`Model Error`)
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit` (`Model Error`)
  - `mlx-community/LFM2.5-VL-1.6B-bf16` (`Weight Mismatch`)
- **📝 Formatting Issues (4):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-Flash-mxfp4`
  - `mlx-community/GLM-4.6V-nvfp4`
  - `mlx-community/Idefics3-8B-Llama3-bf16`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 1,988 | Min: 4.54 | Max: 52,980
- **Peak Memory**: Avg: 21 | Min: 1.3 | Max: 78
- **Total Time**: Avg: 20.34s | Min: 0.91s | Max: 101.94s
- **Generation Time**: Avg: 18.13s | Min: 0.44s | Max: 98.54s
- **Model Load Time**: Avg: 2.07s | Min: 0.35s | Max: 8.70s

## ✅ Usable Diagnostic Candidates

Models that passed mechanical diagnostics and retained useful cataloguing or
metadata-alignment signals.

- _Best end-to-end cataloging:_ [`mlx-community/Qwen3.5-9B-MLX-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen35-9b-mlx-4bit)
  (Utility B 80/100 | Description 82 | Keywords 93 | Speed 90.5 TPS | Memory
  11 | Caveat hit token cap (200); nontext prompt burden=100%)
- _Best descriptions:_ [`mlx-community/InternVL3-8B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-internvl3-8b-bf16)
  (Utility B 69/100 | Description 93 | Keywords 0 | Speed 34.3 TPS | Memory 17
  | Caveat nontext prompt burden=100%)
- _Best keywording:_ [`mlx-community/Qwen3.5-9B-MLX-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen35-9b-mlx-4bit)
  (Utility B 80/100 | Description 82 | Keywords 93 | Speed 90.5 TPS | Memory
  11 | Caveat hit token cap (200); nontext prompt burden=100%)
- _Fastest generation:_ [`mlx-community/gemma-4-31b-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-gemma-4-31b-bf16)
  (Utility F 0/100 | Description 0 | Keywords 0 | Speed 52,980 TPS | Memory 64
  | Caveat Output appears truncated to about 1 tokens.; nontext prompt
  burden=98%)
- _Lowest memory footprint:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16)
  (Utility F 17/100 | Description 50 | Keywords 0 | Speed 442 TPS | Memory 1.3
  | Caveat nontext prompt burden=98%)
- _Best balance:_ [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-llama-32-11b-vision-instruct-8bit)
  (Utility B 80/100 | Description 88 | Keywords 0 | Speed 18.1 TPS | Memory 15
  | Caveat nontext prompt burden=62%)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (3):_ [`facebook/pe-av-large`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-facebook-pe-av-large),
  [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-kimi-vl-a3b-thinking-8bit),
  [`mlx-community/LFM2.5-VL-1.6B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-lfm25-vl-16b-bf16).
  Example: `Model Error`.
- _📝 Formatting Issues (4):_ [`mlx-community/GLM-4.6V-Flash-6bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/GLM-4.6V-Flash-mxfp4`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4),
  [`mlx-community/GLM-4.6V-nvfp4`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-nvfp4),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16).
- _Low-utility outputs (8):_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16),
  [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit),
  [`mlx-community/FastVLM-0.5B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-fastvlm-05b-bf16),
  [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen2-vl-2b-instruct-4bit),
  +4 more. Common weakness: Output lacks detail.

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

_Overall runtime:_ 1088.25s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Cached Tokens | Finish Reason   |   Diffusion Canvas Tokens | Diffusion Denoising Steps   |   Diffusion Work Tokens |   Diffusion Canvas Tps |   Diffusion Work Tps | Is Draft   | Draft Text   | Text Already Printed   | Diffusion Step   | Diffusion Total Steps   | Diffusion Canvas Index   |   Generation (s) |   Load (s) |   Total (s) | Quality Issues               |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|----------------:|:----------------|--------------------------:|:----------------------------|------------------------:|-----------------------:|---------------------:|:-----------|:-------------|:-----------------------|:-----------------|:------------------------|:-------------------------|-----------------:|-----------:|------------:|:-----------------------------|----------------:|
| `facebook/pe-av-large`                                  |                   |                       |                |              |           |             |                 |                 |                           |                             |                         |                        |                      |            |              |                        |                  |                         |                          |                  |      0.12s |       1.14s |                              |          mlx-lm |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |                   |                       |                |              |           |             |                 |                 |                           |                             |                         |                        |                      |            |              |                        |                  |                         |                          |                  |      0.22s |       1.23s |                              |             mlx |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |                   |                       |                |              |           |             |                 |                 |                           |                             |                         |                        |                      |            |              |                        |                  |                         |                          |                  |      0.13s |       1.16s |                              |             mlx |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      |               277 |                    12 |            289 |         1344 |       442 |         1.3 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            0.44s |      0.35s |       0.91s |                              |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               277 |                    58 |            335 |          709 |       329 |           3 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            0.80s |      0.47s |       1.41s |                              |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |                22 |                   124 |            146 |          111 |       377 |         1.7 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            0.89s |      0.50s |       1.53s |                              |                 |
| `qnguyen3/nanoLLaVA`                                    |                22 |                    58 |             80 |          189 |       114 |           4 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            0.98s |      0.55s |       1.65s |                              |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               280 |                    30 |            310 |          437 |       116 |          17 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            1.26s |      2.41s |       3.81s |                              |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             1,196 |                    69 |          1,265 |         3089 |       124 |         5.5 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            1.44s |      0.65s |       2.23s |                              |                 |
| `mlx-community/MiniCPM-V-4.6-8bit`                      |               612 |                    24 |            636 |          704 |       248 |         3.3 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            1.46s |      0.89s |       2.49s |                              |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               266 |                     1 |            267 |          229 |     47904 |         5.9 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            1.47s |      1.51s |       3.11s | ⚠️harness(prompt_template)   |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,103 |                     3 |          4,106 |         3497 |      80.8 |         4.6 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            1.58s |      1.17s |       2.88s | ⚠️harness(long_context), ... |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |                97 |                   138 |            235 |          315 |       129 |         5.5 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            1.86s |      0.73s |       2.73s |                              |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |             2,613 |                    84 |          2,697 |         2479 |       186 |         7.8 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            1.90s |      0.98s |       3.01s |                              |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |                26 |                    19 |             45 |           29 |       338 |         1.8 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            1.94s |      0.66s |       2.74s |                              |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               280 |                    27 |            307 |          376 |      28.1 |          19 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            2.07s |      2.56s |       4.76s |                              |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,327 |                    19 |          2,346 |         2140 |      32.9 |          19 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            2.18s |      2.00s |       4.32s | formatting                   |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             1,805 |                    37 |          1,842 |         2330 |      34.3 |          17 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            2.29s |      1.71s |       4.13s |                              |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |               770 |                   103 |            873 |         3146 |      56.2 |         9.2 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            2.36s |      0.93s |       3.41s |                              |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             1,805 |                    25 |          1,830 |         1160 |      32.3 |          18 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            2.77s |      1.79s |       4.69s |                              |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               268 |                     1 |            269 |          115 |     52980 |          64 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            2.79s |      6.16s |       9.07s | ⚠️harness(prompt_template)   |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             2,160 |                    12 |          2,172 |          971 |      65.3 |         9.7 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            2.95s |      0.91s |       3.99s | ⚠️harness(prompt_template)   |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,196 |                    69 |          1,265 |          512 |       121 |         5.5 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            3.40s |      0.72s |       4.24s |                              |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               275 |                    77 |            352 |          344 |      30.9 |          19 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            3.59s |      2.35s |       6.08s |                              |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |             2,614 |                    84 |          2,698 |         1315 |        64 |          13 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            3.72s |      1.36s |       5.22s |                              |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |             2,614 |                    83 |          2,697 |         1254 |      62.7 |          13 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            3.80s |      1.38s |       5.31s |                              |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,031 |                    85 |          1,116 |         1050 |      34.8 |          11 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            3.83s |      1.58s |       5.54s |                              |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |               770 |                   200 |            970 |          664 |      57.4 |         9.2 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            4.91s |      0.93s |       5.96s | ⚠️harness(stop_token), ...   |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               274 |                   200 |            474 |          338 |      47.3 |          17 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            5.35s |      2.28s |       7.76s | cutoff                       |                 |
| `mlx-community/pixtral-12b-8bit`                        |             2,804 |                   129 |          2,933 |         1615 |      38.2 |          16 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            5.49s |      1.68s |       7.29s |                              |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               275 |                    78 |            353 |          327 |      17.4 |          33 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            5.65s |      3.49s |       9.26s |                              |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |             1,039 |                   194 |          1,233 |          359 |      73.1 |          18 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            5.96s |      2.16s |       8.25s | reasoning-leak               |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,031 |                   102 |          1,133 |         2743 |      19.2 |          10 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            6.07s |      1.46s |       7.66s |                              |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |             1,295 |                   200 |          1,495 |          466 |      60.8 |          60 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            6.68s |      5.60s |      12.57s | token-cap                    |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |             2,081 |                   116 |          2,197 |          675 |        31 |          22 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            7.23s |      2.13s |       9.49s | ⚠️harness(encoding)          |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             2,895 |                   200 |          3,095 |         1362 |        41 |          15 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            7.37s |      1.63s |       9.13s | reasoning-leak, cutoff       |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |                16 |                   109 |            125 |         8.94 |      18.1 |          15 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            8.14s |      1.51s |       9.78s |                              |                 |
| `mlx-community/pixtral-12b-bf16`                        |             2,804 |                   127 |          2,931 |         1801 |      19.4 |          28 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            8.48s |      2.55s |      11.16s |                              |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |             6,157 |                   200 |          6,357 |         1089 |      77.7 |         8.4 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            8.53s |      1.37s |      10.05s | ⚠️harness(stop_token), ...   |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |             6,157 |                   182 |          6,339 |          997 |      54.5 |          11 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            9.79s |      1.51s |      11.44s | ⚠️harness(stop_token), ...   |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |             1,201 |                    72 |          1,273 |         86.6 |      49.3 |          41 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           16.14s |      1.24s |      17.51s |                              |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |             1,201 |                    79 |          1,280 |         82.3 |      30.2 |          48 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           17.99s |      2.04s |      20.15s |                              |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |             6,157 |                   200 |          6,357 |          398 |        38 |          78 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           21.05s |      6.96s |      28.16s | ⚠️harness(stop_token), ...   |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |            16,250 |                    75 |         16,325 |          843 |      58.3 |          13 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           21.16s |      1.19s |      22.48s |                              |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |                17 |                   105 |            122 |          5.8 |      4.88 |          25 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           24.74s |      2.22s |      27.09s |                              |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,031 |                   131 |          1,162 |          845 |      5.23 |          26 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           26.64s |      2.49s |      29.26s |                              |                 |
| `mlx-community/MolmoPoint-8B-fp16`                      |             2,818 |                   146 |          2,964 |          676 |      5.94 |          26 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           29.40s |      2.25s |      31.80s |                              |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |             1,039 |                   200 |          1,239 |          681 |      4.54 |          39 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           46.06s |      3.38s |      49.58s | reasoning-leak, cutoff       |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |            16,241 |                   200 |         16,441 |          264 |       103 |          26 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           64.04s |      2.56s |      66.81s | token-cap                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |            16,241 |                   200 |         16,441 |          262 |      89.2 |          35 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           64.76s |      3.22s |      68.13s | cutoff                       |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |            16,241 |                   200 |         16,441 |          256 |      90.5 |          11 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           66.40s |      1.42s |      67.96s | token-cap                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |            16,241 |                   200 |         16,441 |          249 |      64.5 |          76 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           69.08s |      8.70s |      77.91s | cutoff                       |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |            16,250 |                     4 |         16,254 |          235 |       284 |         5.1 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           69.79s |      0.54s |      70.47s | ⚠️harness(long_context), ... |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |            16,241 |                   200 |         16,441 |          204 |      27.2 |          26 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           87.73s |      2.33s |      90.20s | cutoff                       |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |            16,241 |                   200 |         16,441 |          194 |      17.1 |          38 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           96.12s |      3.13s |      99.39s | cutoff                       |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |            16,241 |                   200 |         16,441 |          189 |      16.7 |          38 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           98.54s |      3.18s |     101.94s | token-cap                    |                 |

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
  sha256=e50cd0c2d2ae16781f644476459cbc2ca23b0d428a897140740f86678b5e2bf5)
- _MLX libmlx.dylib:_ /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib
  (21,675,136 bytes,
  sha256=7c19dc6a8bf6b56db0155bed7a1d2b0a54182b42eba2fcf0f42b97d3d4c57eff)

## Library Versions

- `numpy`: `2.4.6`
- `mlx`: `0.32.0.dev20260605+6ea7a00d`
- `mlx-vlm`: `0.6.1`
- `mlx-lm`: `0.31.3`
- `mlx-audio`: `0.4.3`
- `huggingface-hub`: `1.18.0`
- `transformers`: `5.10.2`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-06-05 13:31:25 BST_
