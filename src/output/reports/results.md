# Model Performance Results

_Generated on 2026-06-07 21:35:51 BST_

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ 2 (top owners: mlx=1, huggingface-hub=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=6, mechanically clean
  outputs=41/57.
- _Useful now:_ 29 mechanically clean A/B model(s) with useful signals.
- _Review watchlist:_ 22 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Quality signal frequency:_ harness=6, cutoff=6, generation_loop=5,
  reasoning_leak=4, degeneration=3, token_cap=3.

### Runtime

- _Runtime pattern:_ model load dominates measured phase time (52%; 32/59
  measured model(s)).
- _Phase totals:_ model load=296.69s, local prompt prep=0.19s, upstream
  prefill / first-token=61.51s, post-prefill decode=205.44s, cleanup=6.13s.
- _Generation total:_ 266.96s across 57 model(s); upstream prefill /
  first-token split available for 57/57 model(s).
- _What this likely means:_ Cold model load time is a major share of runtime
  for this cohort.
- _Suggested next action:_ Consider staged runs, model reuse, or narrowing the
  model set before reruns.
- _Termination reasons:_ completed=47, exception=2, max_tokens=10.
- _Validation overhead:_ 0.10s total (avg 0.00s across 59 model(s)).
- _Upstream prefill / first-token latency:_ Avg 1.08s | Min 0.07s | Max 4.85s
  across 57 model(s).

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/nanoLLaVA-1.5-4bit` (375.4 tps)
- **💾 Most efficient:** `mlx-community/FastVLM-0.5B-bf16` (1.8 GB)
- **⚡ Fastest load:** `mlx-community/LFM2-VL-1.6B-8bit` (0.48s)
- **📊 Average TPS:** 88.5 across 57 models

## 📈 Resource Usage

- **Input image size:** 0.31 MP
- **Average peak delta from post-load:** 3.16 GB
- **Peak memory delta / MP:** 10532 MB/MP
- **Average peak memory:** 17.6 GB
- **Memory efficiency:** 46 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** ✅ B: 33 | 🟡 C: 8 | 🟠 D: 5 | ❌ F: 11

**Average Utility Score:** 58/100

- **Best for cataloging:** `Qwen/Qwen3-VL-2B-Instruct` (✅ B, 75/100)
- **Best descriptions:** `mlx-community/gemma-4-31b-bf16` (93/100)
- **Best keywording:** `mlx-community/Ministral-3-3B-Instruct-2512-4bit` (73/100)
- **Worst for cataloging:** `mlx-community/paligemma2-3b-pt-896-4bit` (❌ F, 0/100)

### ⚠️ 16 Models with Low Utility (D/F)

- `HuggingFaceTB/SmolVLM-Instruct`: ❌ F (33/100) - Output lacks detail
- `microsoft/Phi-3.5-vision-instruct`: 🟠 D (35/100) - Output lacks detail
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`: 🟠 D (47/100) - Keywords are not specific or diverse enough
- `mlx-community/FastVLM-0.5B-bf16`: ❌ F (31/100) - Output lacks detail
- `mlx-community/Idefics3-8B-Llama3-bf16`: ❌ F (30/100) - Keywords are not specific or diverse enough
- `mlx-community/InternVL3-14B-8bit`: ❌ F (31/100) - Output lacks detail
- `mlx-community/LFM2-VL-1.6B-8bit`: ❌ F (34/100) - Output lacks detail
- `mlx-community/LFM2.5-VL-1.6B-bf16`: ❌ F (33/100) - Output lacks detail
- `mlx-community/MiniCPM-V-4.6-8bit`: 🟠 D (47/100) - Keywords are not specific or diverse enough
- `mlx-community/Phi-3.5-vision-instruct-bf16`: 🟠 D (35/100) - Output lacks detail
- `mlx-community/Qwen3-VL-2B-Instruct-bf16`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Qwen3-VL-2B-Thinking-bf16`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/SmolVLM-Instruct-bf16`: ❌ F (33/100) - Output lacks detail
- `mlx-community/gemma-3n-E2B-4bit`: 🟠 D (50/100) - Keywords are not specific or diverse enough
- `mlx-community/paligemma2-3b-pt-896-4bit`: ❌ F (0/100) - Empty or minimal output

## ⚠️ Quality Issues

- **❌ Failed Models (2):**
  - `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (`Model Error`)
  - `facebook/pe-av-large` (`Model Error`)
- **🔄 Repetitive Output (1):**
  - `mlx-community/gemma-3n-E2B-4bit` (token: `phrase: "have this image. have..."`)
- **📝 Formatting Issues (1):**
  - `mlx-community/Idefics3-8B-Llama3-bf16`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 88.5 | Min: 4.72 | Max: 375
- **Peak Memory**: Avg: 18 | Min: 1.8 | Max: 71
- **Total Time**: Avg: 6.60s | Min: 0.75s | Max: 47.17s
- **Generation Time**: Avg: 4.68s | Min: 0.17s | Max: 43.54s
- **Model Load Time**: Avg: 1.91s | Min: 0.48s | Max: 7.63s

## ✅ Usable Diagnostic Candidates

Models that passed mechanical diagnostics and retained useful cataloguing or
metadata-alignment signals.

- _Best end-to-end cataloging:_ [`mlx-community/Qwen3.5-35B-A3B-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen35-35b-a3b-4bit)
  (Utility B 75/100 | Description 87 | Keywords 0 | Speed 117 TPS | Memory 21
  | Caveat nontext prompt burden=98%)
- _Best descriptions:_ [`mlx-community/Qwen3.5-35B-A3B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen35-35b-a3b-bf16)
  (Utility B 74/100 | Description 93 | Keywords 0 | Speed 69.9 TPS | Memory 71
  | Caveat nontext prompt burden=98%)
- _Best keywording:_ [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-ministral-3-3b-instruct-2512-4bit)
  (Utility B 70/100 | Description 74 | Keywords 73 | Speed 199 TPS | Memory
  4.4 | Caveat nontext prompt burden=99%)
- _Fastest generation:_ [`mlx-community/nanoLLaVA-1.5-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-nanollava-15-4bit)
  (Utility C 59/100 | Description 70 | Keywords 0 | Speed 375 TPS | Memory 1.9
  | Caveat nontext prompt burden=73%)
- _Lowest memory footprint:_ [`mlx-community/FastVLM-0.5B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-fastvlm-05b-bf16)
  (Utility F 31/100 | Description 75 | Keywords 0 | Speed 314 TPS | Memory 1.8
  | Caveat nontext prompt burden=77%)
- _Best balance:_ [`Qwen/Qwen3-VL-2B-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-qwen-qwen3-vl-2b-instruct)
  (Utility B 75/100 | Description 81 | Keywords 0 | Speed 134 TPS | Memory 5.2
  | Caveat nontext prompt burden=98%)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (2):_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16),
  [`facebook/pe-av-large`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-facebook-pe-av-large).
  Example: `Model Error`.
- _🔄 Repetitive Output (1):_ [`mlx-community/gemma-3n-E2B-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-gemma-3n-e2b-4bit).
  Example: token: `phrase: "have this image. have..."`.
- _📝 Formatting Issues (1):_ [`mlx-community/Idefics3-8B-Llama3-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16).
- _Low-utility outputs (16):_ [`HuggingFaceTB/SmolVLM-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`microsoft/Phi-3.5-vision-instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-microsoft-phi-35-vision-instruct),
  [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit),
  [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16),
  +12 more. Common weakness: Output lacks detail.

## 🚨 Failures by Package (Actionable)

| Package           |   Failures | Error Types   | Affected Models                    |
|-------------------|------------|---------------|------------------------------------|
| `mlx`             |          1 | Model Error   | `LiquidAI/LFM2.5-VL-450M-MLX-bf16` |
| `huggingface-hub` |          1 | Model Error   | `facebook/pe-av-large`             |

### Actionable Items by Package

#### mlx

- LiquidAI/LFM2.5-VL-450M-MLX-bf16 (Model Error)
  - Error: `Model loading failed: Received 2 parameters not in model: <br>multi_modal_projector.layer_norm.bias,<br>multi_modal_project...`
  - Type: `ValueError`

#### huggingface-hub

- facebook/pe-av-large (Model Error)
  - Error: `Model loading failed: [Errno 2] No such file or directory: '/Users/jrp/.cache/huggingface/hub/models--facebook--pe-av...`
  - Type: `ValueError`

_Prompt used:_

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Describe this image briefly.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 573.14s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Cached Tokens | Finish Reason   |   Diffusion Canvas Tokens | Diffusion Denoising Steps   |   Diffusion Work Tokens |   Diffusion Canvas Tps |   Diffusion Work Tps | Is Draft   | Draft Text   | Text Already Printed   | Diffusion Step   | Diffusion Total Steps   | Diffusion Canvas Index   |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                     |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|----------------:|:----------------|--------------------------:|:----------------------------|------------------------:|-----------------------:|---------------------:|:-----------|:-------------|:-----------------------|:-----------------|:------------------------|:-------------------------|-----------------:|-----------:|------------:|:-----------------------------------|----------------:|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      |                   |                       |                |              |           |             |                 |                 |                           |                             |                         |                        |                      |            |              |                        |                  |                         |                          |                  |      0.14s |       0.96s |                                    |             mlx |
| `facebook/pe-av-large`                                  |                   |                       |                |              |           |             |                 |                 |                           |                             |                         |                        |                      |            |              |                        |                  |                         |                          |                  |    187.91s |     188.99s |                                    | huggingface-hub |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |               269 |                    13 |            282 |         3865 |       193 |         4.1 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            0.17s |      0.58s |       0.75s |                                    |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |                22 |                    81 |            103 |          219 |       375 |         1.9 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            0.33s |      0.50s |       0.84s |                                    |                 |
| `qnguyen3/nanoLLaVA`                                    |                22 |                    35 |             57 |          240 |       115 |           4 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            0.42s |      0.65s |       1.08s |                                    |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               269 |                    10 |            279 |          717 |       326 |           3 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            0.43s |      0.48s |       0.92s |                                    |                 |
| `mlx-community/MiniCPM-V-4.6-8bit`                      |               228 |                    22 |            250 |          699 |       276 |           3 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            0.44s |      0.93s |       1.38s |                                    |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |               417 |                    52 |            469 |         1407 |       324 |         2.5 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            0.48s |      0.49s |       0.98s |                                    |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             1,196 |                    13 |          1,209 |         3394 |       129 |         5.5 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            0.53s |      0.58s |       1.11s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |               770 |                    19 |            789 |         3841 |      60.1 |         9.2 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            0.58s |      0.87s |       1.45s |                                    |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |               949 |                    99 |          1,048 |         3034 |       199 |         4.4 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            0.84s |      1.02s |       1.87s |                                    |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |               408 |                    42 |            450 |         1052 |      93.2 |         7.7 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            0.85s |      1.31s |       2.17s |                                    |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |               319 |                    71 |            390 |         1227 |       101 |           7 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            0.98s |      1.35s |       2.33s |                                    |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               286 |                    24 |            310 |          349 |       115 |          17 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            1.04s |      2.34s |       3.41s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |               319 |                    63 |            382 |          569 |       117 |          21 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            1.11s |      2.46s |       3.58s |                                    |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |                26 |                    15 |             41 |         24.4 |       314 |         1.8 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            1.15s |      0.59s |       1.74s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |               319 |                    58 |            377 |          534 |        99 |          30 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            1.19s |      3.06s |       4.27s |                                    |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,103 |                     3 |          4,106 |         3569 |      81.8 |         4.6 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            1.21s |      1.20s |       2.42s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |               417 |                    65 |            482 |         1471 |      66.8 |          10 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            1.32s |      1.15s |       2.48s |                                    |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |               315 |                    89 |            404 |          457 |       134 |         5.2 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            1.39s |      0.66s |       2.06s |                                    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |               950 |                    52 |          1,002 |         1502 |      65.8 |          10 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            1.44s |      1.37s |       2.82s |                                    |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |               317 |                   200 |            517 |         3725 |       135 |         5.3 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            1.60s |      0.76s |       2.38s | generation_loop(degeneration), ... |                 |
| `mlx-community/Qwen3-VL-2B-Instruct-bf16`               |               315 |                   200 |            515 |         3689 |       134 |         5.3 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            1.61s |      0.74s |       2.35s | generation_loop(degeneration), ... |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               286 |                    25 |            311 |          392 |      28.7 |          19 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            1.62s |      2.51s |       4.14s |                                    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |               950 |                    68 |          1,018 |         1517 |        69 |         9.7 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            1.63s |      1.45s |       3.09s |                                    |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,327 |                    23 |          2,350 |         2265 |      33.4 |          19 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            1.76s |      1.87s |       3.64s | formatting                         |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |                97 |                   187 |            284 |          337 |       131 |         5.5 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            1.79s |      0.56s |       2.36s |                                    |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |               408 |                    69 |            477 |          488 |        64 |          10 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            1.94s |      1.49s |       3.45s |                                    |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,196 |                    13 |          1,209 |          628 |       126 |         5.5 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            2.06s |      0.69s |       2.76s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             3,341 |                    48 |          3,389 |         2682 |      34.8 |          18 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            2.65s |      1.77s |       4.43s |                                    |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |                16 |                    25 |             41 |         9.92 |      22.2 |          15 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            2.76s |      1.65s |       4.42s |                                    |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |               319 |                    71 |            390 |          497 |      33.5 |          19 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            2.77s |      2.17s |       4.96s |                                    |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               266 |                   200 |            466 |          223 |       120 |           6 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            2.87s |      1.39s |       4.27s | repetitive(phrase: "have th...     |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |               319 |                    40 |            359 |          389 |      19.6 |          30 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            2.88s |      3.15s |       6.05s |                                    |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               274 |                   124 |            398 |          749 |      48.7 |          17 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            2.93s |      2.20s |       5.14s |                                    |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             3,341 |                    15 |          3,356 |         1289 |      33.4 |          19 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            3.07s |      1.72s |       4.80s |                                    |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             2,356 |                    55 |          2,411 |         1093 |      63.6 |         9.7 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            3.08s |      0.88s |       3.97s |                                    |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |               319 |                    49 |            368 |          471 |      19.5 |          30 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            3.21s |      3.00s |       6.23s |                                    |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |               399 |                   200 |            599 |          485 |      79.5 |          16 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            3.45s |      2.00s |       5.46s | reasoning-leak, token-cap          |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |               417 |                    90 |            507 |          558 |      33.1 |          19 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            3.48s |      2.01s |       5.51s | ⚠️harness(encoding)                |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |               399 |                   200 |            599 |          668 |      68.8 |          20 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            3.64s |      2.18s |       5.82s | reasoning-leak, cutoff             |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               275 |                    91 |            366 |          340 |      31.8 |          19 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            3.69s |      2.31s |       6.02s |                                    |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |               408 |                    71 |            479 |          161 |      51.4 |          63 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            3.93s |      5.48s |       9.43s |                                    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,031 |                   101 |          1,132 |          906 |      34.8 |          11 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            4.13s |      1.65s |       5.79s |                                    |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |               770 |                   200 |            970 |          814 |      59.9 |         9.2 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            4.34s |      0.88s |       5.23s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |               319 |                    55 |            374 |         88.7 |      69.9 |          71 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            4.40s |      7.63s |      12.04s |                                    |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               275 |                    70 |            345 |          322 |      17.9 |          33 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            4.78s |      3.36s |       8.15s |                                    |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             1,330 |                   174 |          1,504 |         1226 |      44.1 |          14 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            5.04s |      1.56s |       6.61s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |               439 |                   200 |            639 |          215 |        65 |          60 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            5.20s |      5.05s |      10.40s | generation_loop(degeneration), ... |                 |
| `mlx-community/pixtral-12b-8bit`                        |             1,239 |                   200 |          1,439 |         1442 |        40 |          15 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            5.87s |      1.64s |       7.52s | cutoff                             |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |               745 |                    55 |            800 |          154 |      50.8 |          20 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            5.96s |      1.30s |       7.26s |                                    |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |               745 |                    55 |            800 |          154 |        30 |          27 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            6.69s |      1.69s |       8.39s |                                    |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,031 |                   119 |          1,150 |         2753 |      18.7 |          10 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            6.76s |      1.40s |       8.17s |                                    |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               274 |                    36 |            310 |          110 |      7.52 |          64 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |            7.30s |      6.02s |      13.34s |                                    |                 |
| `mlx-community/pixtral-12b-bf16`                        |             1,239 |                   200 |          1,439 |         1547 |      20.1 |          27 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           10.79s |      2.48s |      13.28s | cutoff                             |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |                17 |                    77 |             94 |         7.78 |      5.11 |          25 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           17.27s |      2.14s |      19.42s |                                    |                 |
| `mlx-community/MolmoPoint-8B-fp16`                      |               766 |                   163 |            929 |          328 |      5.95 |          23 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           29.84s |      2.17s |      32.03s |                                    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,031 |                   159 |          1,190 |          847 |       5.4 |          26 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           30.74s |      2.43s |      33.18s |                                    |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |               399 |                   200 |            599 |          428 |      4.72 |          39 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           43.54s |      3.61s |      47.17s | reasoning-leak, cutoff             |                 |

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
  (21,675,136 bytes,
  sha256=6255fc531acc826e8625f261237f6bb6c75490177d3b769ab70c1ff9f71b6d7f)

## Library Versions

- `numpy`: `2.4.6`
- `mlx`: `0.32.0.dev20260607+8f0e8b14`
- `mlx-vlm`: `0.6.2`
- `mlx-lm`: `0.31.3`
- `mlx-audio`: `0.4.4`
- `huggingface-hub`: `1.18.0`
- `transformers`: `5.10.2`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-06-07 21:35:51 BST_
