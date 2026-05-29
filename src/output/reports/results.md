# Model Performance Results

_Generated on 2026-05-29 21:59:41 BST_

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ 3 (top owners: mlx=2, mlx-lm=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=10, mechanically clean
  outputs=30/53.
- _Useful now:_ 23 mechanically clean A/B model(s) with useful signals.
- _Review watchlist:_ 22 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Quality signal frequency:_ harness=10, cutoff=8, reasoning_leak=6,
  generation_loop=5, formatting=4, text_sanity=4.

### Runtime

- _Runtime pattern:_ upstream prefill / first-token dominates measured phase
  time (65%; 19/56 measured model(s)).
- _Phase totals:_ model load=109.53s, local prompt prep=0.18s, upstream
  prefill / first-token=725.11s, post-prefill decode=279.41s, cleanup=6.32s.
- _Generation total:_ 1004.52s across 53 model(s); upstream prefill /
  first-token split available for 53/53 model(s).
- _What this likely means:_ Most measured runtime is spent inside upstream
  generation before the first token is available.
- _Suggested next action:_ Inspect prompt/image token accounting,
  dynamic-resolution image burden, prefill step sizing, and cache/prefill
  behavior.
- _Termination reasons:_ completed=38, exception=3, max_tokens=15.
- _Validation overhead:_ 6.95s total (avg 0.12s across 56 model(s)).
- _Upstream prefill / first-token latency:_ Avg 13.68s | Min 0.03s | Max
  100.02s across 53 model(s).

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/gemma-3n-E2B-4bit` (84502.3 tps)
- **💾 Most efficient:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (1.4 GB)
- **⚡ Fastest load:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (0.44s)
- **📊 Average TPS:** 2873.7 across 53 models

## 📈 Resource Usage

- **Input image size:** 66.45 MP
- **Average peak delta from post-load:** 5.62 GB
- **Peak memory delta / MP:** 87 MB/MP
- **Average peak memory:** 20.6 GB
- **Memory efficiency:** 201 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** ✅ B: 37 | 🟡 C: 8 | ❌ F: 8

**Average Utility Score:** 63/100

- **Best for cataloging:** `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX` (✅ B, 80/100)
- **Best descriptions:** `mlx-community/gemma-4-31b-it-4bit` (93/100)
- **Best keywording:** `mlx-community/Qwen3.5-9B-MLX-4bit` (93/100)
- **Worst for cataloging:** `mlx-community/gemma-3n-E2B-4bit` (❌ F, 0/100)

### ⚠️ 8 Models with Low Utility (D/F)

- `LiquidAI/LFM2.5-VL-450M-MLX-bf16`: ❌ F (17/100) - Output lacks detail
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/FastVLM-0.5B-bf16`: ❌ F (31/100) - Output lacks detail
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (5/100) - Output too short to be useful
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

- **Generation Tps**: Avg: 2,874 | Min: 4.23 | Max: 84,502
- **Peak Memory**: Avg: 21 | Min: 1.4 | Max: 78
- **Total Time**: Avg: 21.15s | Min: 0.81s | Max: 121.77s
- **Generation Time**: Avg: 18.95s | Min: 0.25s | Max: 118.56s
- **Model Load Time**: Avg: 2.06s | Min: 0.44s | Max: 8.26s

## ✅ Usable Diagnostic Candidates

Models that passed mechanical diagnostics and retained useful cataloguing or
metadata-alignment signals.

- _Best end-to-end cataloging:_ [`mlx-community/Qwen3.5-27B-mxfp8`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen35-27b-mxfp8)
  (Utility B 80/100 | Description 82 | Keywords 86 | Speed 11.2 TPS | Memory
  38 | Caveat hit token cap (200); nontext prompt burden=100%)
- _Best descriptions:_ [`mlx-community/Molmo-7B-D-0924-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-molmo-7b-d-0924-bf16)
  (Utility B 73/100 | Description 93 | Keywords 0 | Speed 29.5 TPS | Memory 48
  | Caveat nontext prompt burden=100%)
- _Best keywording:_ [`mlx-community/Qwen3.5-27B-mxfp8`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen35-27b-mxfp8)
  (Utility B 80/100 | Description 82 | Keywords 86 | Speed 11.2 TPS | Memory
  38 | Caveat hit token cap (200); nontext prompt burden=100%)
- _Fastest generation:_ [`mlx-community/gemma-3n-E2B-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-gemma-3n-e2b-4bit)
  (Utility F 0/100 | Description 0 | Keywords 0 | Speed 84,502 TPS | Memory
  6.0 | Caveat Output appears truncated to about 1 tokens.; nontext prompt
  burden=98%)
- _Lowest memory footprint:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16)
  (Utility F 17/100 | Description 50 | Keywords 0 | Speed 502 TPS | Memory 1.4
  | Caveat nontext prompt burden=98%)
- _Best balance:_ [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-llama-32-11b-vision-instruct-8bit)
  (Utility B 80/100 | Description 88 | Keywords 0 | Speed 14.5 TPS | Memory 15
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

_Overall runtime:_ 1131.75s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Cached Tokens | Finish Reason   |   Diffusion Canvas Tokens | Diffusion Denoising Steps   |   Diffusion Work Tokens |   Diffusion Canvas Tps |   Diffusion Work Tps | Is Draft   | Draft Text   | Text Already Printed   | Diffusion Step   | Diffusion Total Steps   | Diffusion Canvas Index   |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                     |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|----------------:|:----------------|--------------------------:|:----------------------------|------------------------:|-----------------------:|---------------------:|:-----------|:-------------|:-----------------------|:-----------------|:------------------------|:-------------------------|-----------------:|-----------:|------------:|:-----------------------------------|----------------:|
| `facebook/pe-av-large`                                  |                   |                       |                |              |           |             |                 |                 |                           |                             |                         |                        |                      |            |              |                        |                  |                         |                          |                  |      0.11s |       1.20s |                                    |          mlx-lm |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |                   |                       |                |              |           |             |                 |                 |                           |                             |                         |                        |                      |            |              |                        |                  |                         |                          |                  |      0.18s |       1.74s |                                    |             mlx |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |                   |                       |                |              |           |             |                 |                 |                           |                             |                         |                        |                      |            |              |                        |                  |                         |                          |                  |      0.13s |       1.10s |                                    |             mlx |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      |               277 |                    12 |            289 |         9417 |       502 |         1.4 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            0.25s |      0.44s |       0.81s |                                    |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               277 |                    54 |            331 |         3200 |       319 |           3 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            0.47s |      0.45s |       1.05s |                                    |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               266 |                     1 |            267 |         1174 |     84502 |           6 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            0.53s |      1.42s |       2.08s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |                22 |                   124 |            146 |          317 |       379 |         1.9 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            0.74s |      0.46s |       1.32s |                                    |                 |
| `mlx-community/MiniCPM-V-4.6-8bit`                      |               612 |                    24 |            636 |         2555 |       237 |         3.3 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            0.84s |      0.93s |       1.92s |                                    |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               280 |                    21 |            301 |          703 |       114 |          17 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            0.93s |      2.33s |       3.40s |                                    |                 |
| `qnguyen3/nanoLLaVA`                                    |                22 |                    58 |             80 |          242 |       115 |           4 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            0.96s |      0.51s |       1.59s |                                    |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |                26 |                    19 |             45 |          262 |       303 |         2.1 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            1.13s |      0.65s |       1.93s |                                    |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,196 |                    69 |          1,265 |         3143 |       126 |         5.5 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            1.40s |      0.65s |       2.16s |                                    |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             1,196 |                    69 |          1,265 |         2957 |       127 |         5.5 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            1.43s |      0.59s |       2.13s |                                    |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,103 |                     3 |          4,106 |         3792 |      82.5 |         4.6 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            1.48s |      1.13s |       2.74s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |                97 |                   138 |            235 |          318 |       130 |         5.5 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            1.85s |      0.70s |       2.67s |                                    |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               280 |                    30 |            310 |          389 |      28.5 |          19 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            2.12s |      2.50s |       4.74s |                                    |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |               770 |                   103 |            873 |         3122 |      59.1 |         9.2 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            2.25s |      0.89s |       3.26s |                                    |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |             2,613 |                    84 |          2,697 |         2065 |       144 |         7.8 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            2.25s |      0.98s |       3.36s |                                    |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             1,805 |                    37 |          1,842 |         1879 |      31.8 |          17 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            2.55s |      1.71s |       4.39s |                                    |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,327 |                    19 |          2,346 |         1513 |      30.3 |          19 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            2.65s |      1.93s |       4.72s | formatting                         |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               268 |                     1 |            269 |          116 |     63492 |          64 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            2.70s |      6.02s |       8.84s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             2,160 |                    12 |          2,172 |         1024 |      65.6 |         9.7 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            2.81s |      0.88s |       3.80s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             1,805 |                    25 |          1,830 |          866 |      27.9 |          18 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            3.42s |      1.77s |       5.31s |                                    |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               275 |                    77 |            352 |          398 |      31.4 |          19 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            3.45s |      2.31s |       5.89s |                                    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,031 |                    85 |          1,116 |         1332 |      35.3 |          11 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            3.60s |      1.60s |       5.32s |                                    |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |               770 |                   200 |            970 |         2594 |        51 |         9.2 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            4.52s |      0.90s |       5.56s | ⚠️harness(stop_token), ...         |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |             1,039 |                   200 |          1,239 |         1067 |      62.1 |          18 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            4.59s |      1.98s |       6.69s | reasoning-leak, cutoff             |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |             2,614 |                    83 |          2,697 |          903 |      59.3 |          13 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            4.70s |      1.42s |       6.27s |                                    |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               274 |                   200 |            474 |          866 |      48.6 |          17 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            4.72s |      2.18s |       7.03s | cutoff                             |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |             2,614 |                    84 |          2,698 |          829 |      58.5 |          13 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            5.03s |      1.41s |       6.58s |                                    |                 |
| `mlx-community/pixtral-12b-8bit`                        |             2,804 |                   129 |          2,933 |         1767 |      39.1 |          16 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            5.26s |      1.63s |       7.01s |                                    |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               275 |                    78 |            353 |          326 |      17.8 |          33 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            5.52s |      3.43s |       9.07s |                                    |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,031 |                   102 |          1,133 |         2743 |      19.7 |          10 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            5.92s |      1.41s |       7.45s |                                    |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |             1,295 |                   200 |          1,495 |          330 |      56.2 |          60 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            8.10s |      7.20s |      15.61s | token-cap                          |                 |
| `mlx-community/pixtral-12b-bf16`                        |             2,804 |                   127 |          2,931 |         1835 |      19.9 |          28 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            8.26s |      2.52s |      10.90s |                                    |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |             2,081 |                   116 |          2,197 |          592 |      25.8 |          22 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            8.46s |      2.12s |      10.72s | ⚠️harness(encoding)                |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             2,895 |                   200 |          3,095 |         1167 |      35.5 |          15 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            8.51s |      1.63s |      10.28s | reasoning-leak, cutoff             |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |                16 |                   109 |            125 |         10.8 |      14.5 |          15 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            9.29s |      1.52s |      10.93s |                                    |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |             6,157 |                   182 |          6,339 |          739 |        42 |          11 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           12.95s |      1.44s |      14.52s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |             6,157 |                   200 |          6,357 |          650 |      61.9 |         8.4 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           13.01s |      1.37s |      14.52s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |             1,201 |                    72 |          1,273 |         79.8 |      48.9 |          41 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           17.33s |      1.22s |      18.67s |                                    |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |             1,201 |                    57 |          1,258 |         75.1 |      29.5 |          48 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           18.68s |      1.69s |      20.49s |                                    |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |            16,250 |                    75 |         16,325 |          869 |      58.2 |          13 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           20.56s |      1.12s |      21.80s |                                    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,031 |                   131 |          1,162 |          924 |      5.38 |          26 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           25.85s |      2.43s |      28.41s |                                    |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |                17 |                   105 |            122 |         8.19 |      4.46 |          25 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           25.92s |      2.33s |      28.38s |                                    |                 |
| `mlx-community/MolmoPoint-8B-fp16`                      |             2,818 |                   146 |          2,964 |         1154 |      5.94 |          26 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           27.68s |      2.20s |      30.01s |                                    |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |             6,157 |                   200 |          6,357 |          279 |      27.3 |          78 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           29.98s |      7.71s |      37.82s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |             1,039 |                   200 |          1,239 |          632 |      4.23 |          39 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           49.48s |      3.37s |      52.99s | reasoning-leak, cutoff             |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |            16,241 |                   200 |         16,441 |          270 |      89.6 |          35 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           63.02s |      3.11s |      66.25s | cutoff                             |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |            16,241 |                   200 |         16,441 |          257 |      88.7 |          11 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           66.14s |      1.37s |      67.63s | generation_loop(degeneration), ... |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |            16,241 |                   200 |         16,441 |          251 |       109 |          26 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           67.13s |      2.58s |      69.91s | generation_loop(degeneration), ... |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |            16,241 |                   200 |         16,441 |          249 |      66.4 |          76 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           69.06s |      8.26s |      77.45s | cutoff                             |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |            16,250 |                     9 |         16,259 |          227 |       218 |         5.1 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           72.11s |      0.48s |      72.71s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |            16,241 |                   200 |         16,441 |          199 |      27.9 |          26 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           89.30s |      2.12s |      91.57s | cutoff                             |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |            16,241 |                   200 |         16,441 |          197 |      16.6 |          38 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           95.13s |      3.06s |      98.32s | cutoff                             |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |            16,241 |                   200 |         16,441 |          162 |      11.2 |          38 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |          118.56s |      3.07s |     121.77s | token-cap                          |                 |

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
- _MLX Install Type:_ wheel/site-packages
- _MLX Distribution Root:_ /Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages
- _mlx-metal Distribution Root:_ /Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages
- _MLX Core Extension:_ /Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/mlx/core.cpython-313-darwin.so
- _MLX Metallib:_ /Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/mlx/lib/mlx.metallib
  (157,748,008 bytes,
  sha256=8c8bfcece8c0610745b68879771e5aa1b92b29fa5e17172e5508e4f5153d8d15)
- _MLX libmlx.dylib:_ /Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/mlx/lib/libmlx.dylib
  (21,653,808 bytes,
  sha256=2ee6fbd32ff22e22e1301ebe3c3bece95584104ff9cbc900513d41a095211bbd)

## Library Versions

- `numpy`: `2.4.6`
- `mlx`: `0.31.2`
- `mlx-metal`: `0.31.2`
- `mlx-vlm`: `0.5.0`
- `mlx-lm`: `0.31.3`
- `mlx-audio`: `0.4.3`
- `huggingface-hub`: `1.17.0`
- `transformers`: `5.9.0`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-05-29 21:59:41 BST_
