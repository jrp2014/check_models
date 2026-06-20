# Model Performance Results

_Generated on 2026-06-20 00:48:07 BST_

## Run Contract

- Mode: triage
- Semantic rankings: ungrounded (ungrounded)
- Primary selection tasks: brief captions; structured title/description/keywords when descriptive image metadata is available

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ 2 (top owners: mlx=1, mlx-vlm=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=7, mechanically clean
  outputs=43/58.
- _Useful now:_ 30 model(s) shortlisted for caption review.
- _Review watchlist:_ 22 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Quality signal frequency:_ harness=7, cutoff=6, reasoning_leak=4,
  text_sanity=3, generation_loop=3, lang_mixing=2.

### Runtime

- _Runtime pattern:_ post-prefill decode dominates measured phase time (55%;
  23/60 measured model(s)).
- _Phase totals:_ model load=123.65s, local prompt prep=0.20s, upstream
  prefill / first-token=54.09s, post-prefill decode=228.36s, cleanup=7.99s.
- _Generation total:_ 282.44s across 58 model(s); upstream prefill /
  first-token split available for 58/58 model(s).
- _What this likely means:_ Most measured runtime is spent generating after
  the first token is available.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=50, exception=2, max_tokens=8.
- _Validation overhead:_ 0.11s total (avg 0.00s across 60 model(s)).
- _Upstream prefill / first-token latency:_ Avg 0.93s | Min 0.02s | Max 4.23s
  across 58 model(s).

## 🏆 Performance Highlights

- **Fastest:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (445.9 tps)
- **💾 Most efficient:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (1.0 GB)
- **⚡ Fastest load:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (0.39s)
- **📊 Average TPS:** 71.3 across 58 models

## 📈 Resource Usage

- **Input image size:** 0.31 MP
- **Average peak delta from post-load:** 3.14 GB
- **Peak memory delta / MP:** 10460 MB/MP
- **Average peak memory:** 17.9 GB
- **Memory efficiency:** 44 tokens/GB

## ⚠️ Quality Issues

- **❌ Failed Models (2):**
  - `mlx-community/LFM2.5-VL-1.6B-bf16` (`Weight Mismatch`)
  - `mlx-community/MolmoPoint-8B-fp16` (`Model Error`)
- **🔄 Repetitive Output (1):**
  - `mlx-community/gemma-3n-E2B-4bit` (token: `phrase: "have this image. have..."`)
- **📝 Formatting Issues (1):**
  - `mlx-community/Idefics3-8B-Llama3-bf16`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 71.3 | Min: 3.49 | Max: 446
- **Peak Memory**: Avg: 18 | Min: 0.97 | Max: 71
- **Total Time**: Avg: 7.01s | Min: 0.45s | Max: 62.20s
- **Generation Time**: Avg: 4.87s | Min: 0.05s | Max: 58.60s
- **Model Load Time**: Avg: 2.13s | Min: 0.39s | Max: 6.79s

## Caption Selection

Triage mode suppresses cataloging and keyword scores. Brief-caption
recommendations are ungrounded unless descriptive image metadata is present;
use `model_selection.md` for the caption shortlist and `model_gallery.md` for
full output evidence.

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

_Overall runtime:_ 417.31s

<!-- markdownlint-disable MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Cached Tokens | Finish Reason   |   Diffusion Canvas Tokens | Diffusion Denoising Steps   |   Diffusion Work Tokens |   Diffusion Canvas Tps |   Diffusion Work Tps | Is Draft   | Draft Text   | Text Already Printed   | Diffusion Step   | Diffusion Total Steps   | Diffusion Canvas Index   | Diffusion Block Complete   |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                     |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|----------------:|:----------------|--------------------------:|:----------------------------|------------------------:|-----------------------:|---------------------:|:-----------|:-------------|:-----------------------|:-----------------|:------------------------|:-------------------------|:---------------------------|-----------------:|-----------:|------------:|:-----------------------------------|----------------:|
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |                   |                       |                |              |           |             |                 |                 |                           |                             |                         |                        |                      |            |              |                        |                  |                         |                          |                            |                  |      0.12s |       1.03s |                                    |             mlx |
| `mlx-community/MolmoPoint-8B-fp16`                      |                   |                       |                |              |           |             |                 |                 |                           |                             |                         |                        |                      |            |              |                        |                  |                         |                          |                            |                  |      0.20s |       1.09s |                                    |         mlx-vlm |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      |                80 |                    10 |             90 |         5175 |       446 |        0.97 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.05s |      0.39s |       0.45s |                                    |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               269 |                    10 |            279 |         3936 |       262 |           3 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.13s |      0.57s |       0.70s |                                    |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |                26 |                    15 |             41 |          256 |       251 |         2.2 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.19s |      0.71s |       0.91s |                                    |                 |
| `mlx-community/MiniCPM-V-4.6-8bit`                      |               228 |                    22 |            250 |         1848 |       204 |           3 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.26s |      0.98s |       1.27s |                                    |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |               417 |                    52 |            469 |         2651 |       240 |         2.5 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.40s |      0.56s |       0.97s |                                    |                 |
| `qnguyen3/nanoLLaVA`                                    |                22 |                    35 |             57 |          212 |        93 |         4.2 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.51s |      0.58s |       1.09s |                                    |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |                22 |                    81 |            103 |          252 |       176 |         1.9 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.56s |      0.70s |       1.27s |                                    |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,196 |                    13 |          1,209 |         2506 |       109 |         5.5 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.66s |      0.71s |       1.37s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |               770 |                    19 |            789 |         2630 |      52.2 |         9.2 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.68s |      0.93s |       1.62s |                                    |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             1,196 |                    13 |          1,209 |         2314 |      96.9 |         5.5 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.73s |      0.65s |       1.39s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               286 |                    24 |            310 |          617 |      90.3 |          17 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.74s |      2.68s |       3.43s |                                    |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |               949 |                    99 |          1,048 |         2908 |       156 |         4.4 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.98s |      1.16s |       2.16s |                                    |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |               408 |                    42 |            450 |         1093 |      71.1 |         7.7 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            0.98s |      1.95s |       2.93s |                                    |                 |
| `mlx-community/Qwen3-VL-2B-Instruct-bf16`               |               315 |                    96 |            411 |         3212 |       110 |         5.3 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.01s |      0.89s |       1.90s |                                    |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |               315 |                    96 |            411 |         2858 |       111 |         5.2 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.02s |      0.73s |       1.76s |                                    |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |               319 |                    71 |            390 |         1062 |      78.2 |           7 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.22s |      1.92s |       3.16s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |               319 |                    63 |            382 |          754 |      70.8 |          21 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.33s |      2.89s |       4.23s |                                    |                 |
| `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`         |               286 |                    21 |            307 |          777 |      23.1 |          30 |               0 | stop            |                       200 | 6                           |                    1200 |                    220 |                1,318 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.46s |      3.45s |       4.92s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |               319 |                    58 |            377 |          624 |      60.9 |          30 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.48s |      3.48s |       4.97s |                                    |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |               417 |                    65 |            482 |         1439 |      54.9 |          10 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.54s |      1.17s |       2.72s |                                    |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,103 |                     3 |          4,106 |         2566 |      68.3 |         4.6 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.67s |      1.38s |       3.06s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |               408 |                    69 |            477 |         1049 |      52.9 |          10 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.71s |      1.75s |       3.48s |                                    |                 |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit`          |               286 |                    28 |            314 |          798 |      20.3 |          31 |               0 | stop            |                       200 | 7                           |                    1400 |                    145 |                1,017 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.93s |      3.47s |       5.42s |                                    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |               950 |                    52 |          1,002 |         1024 |      50.4 |          10 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            1.98s |      1.59s |       3.58s |                                    |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               286 |                    25 |            311 |          309 |      22.8 |          19 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            2.04s |      2.81s |       4.86s |                                    |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |               317 |                   196 |            513 |         3093 |       103 |         5.3 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            2.04s |      0.78s |       2.84s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |                97 |                   187 |            284 |          269 |       109 |         5.5 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            2.15s |      0.57s |       2.73s |                                    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |               950 |                    68 |          1,018 |         1073 |      54.2 |         9.7 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            2.16s |      1.56s |       3.73s |                                    |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,327 |                    23 |          2,350 |         1565 |      27.8 |          19 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            2.36s |      2.27s |       4.65s | formatting                         |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               266 |                   200 |            466 |         1028 |        74 |           6 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            2.97s |      1.68s |       4.67s | repetitive(phrase: "have th...     |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |               319 |                    55 |            374 |          168 |      50.2 |          71 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            3.01s |      6.79s |       9.81s |                                    |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |               319 |                    71 |            390 |          432 |      26.9 |          19 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            3.39s |      2.56s |       5.98s |                                    |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |               319 |                    40 |            359 |          383 |      15.4 |          30 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            3.44s |      3.45s |       6.92s |                                    |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               274 |                   124 |            398 |          737 |      39.2 |          17 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            3.55s |      2.44s |       6.01s |                                    |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             3,341 |                    48 |          3,389 |         1667 |      29.6 |          18 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            3.66s |      1.63s |       5.30s |                                    |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |               408 |                    71 |            479 |          216 |      39.3 |          63 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            3.71s |      5.93s |       9.65s |                                    |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |                16 |                    25 |             41 |          8.5 |        13 |          15 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            3.83s |      1.75s |       5.59s |                                    |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |               399 |                   200 |            599 |         1200 |      58.1 |          16 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            3.89s |      2.37s |       6.27s | reasoning-leak, token-cap          |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |               319 |                    49 |            368 |          391 |      15.7 |          30 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            3.95s |      3.48s |       7.46s |                                    |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |               417 |                    90 |            507 |          502 |      25.9 |          19 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            4.32s |      2.39s |       6.72s | ⚠️harness(encoding)                |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |               770 |                   200 |            970 |         2547 |      49.3 |         9.2 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            4.38s |      0.94s |       5.32s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             2,356 |                    55 |          2,411 |          712 |      51.8 |         9.7 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            4.44s |      0.97s |       5.42s |                                    |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |               399 |                   200 |            599 |         1170 |      50.3 |          20 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            4.45s |      2.45s |       6.91s | reasoning-leak, cutoff             |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               275 |                    91 |            366 |          325 |      24.4 |          19 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            4.61s |      2.54s |       7.16s |                                    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,031 |                   101 |          1,132 |         1069 |      28.2 |          11 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            4.63s |      1.83s |       6.47s |                                    |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |               439 |                   200 |            639 |          433 |      53.1 |          60 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            4.79s |      4.86s |       9.67s | generation_loop(degeneration), ... |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             3,341 |                    15 |          3,356 |          789 |      27.6 |          19 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            4.81s |      1.67s |       6.49s |                                    |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |               745 |                    55 |            800 |          204 |      43.4 |          20 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            4.95s |      1.36s |       6.32s |                                    |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |               745 |                    55 |            800 |          191 |      26.7 |          27 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            5.98s |      1.97s |       7.96s |                                    |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             1,330 |                   174 |          1,504 |         1085 |        36 |          14 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            6.08s |      1.86s |       7.95s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               275 |                    70 |            345 |          275 |      13.7 |          33 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            6.13s |      3.66s |       9.80s |                                    |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               274 |                    36 |            310 |          112 |      6.76 |          64 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            7.81s |      6.30s |      14.13s |                                    |                 |
| `mlx-community/pixtral-12b-8bit`                        |             1,239 |                   200 |          1,439 |         1233 |      29.4 |          15 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            7.83s |      1.90s |       9.74s | cutoff                             |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,031 |                   119 |          1,150 |         2401 |      15.4 |          11 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |            8.18s |      1.71s |       9.89s |                                    |                 |
| `mlx-community/pixtral-12b-bf16`                        |             1,239 |                   200 |          1,439 |         1276 |      16.5 |          27 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |           13.09s |      2.78s |      15.88s | cutoff                             |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |                17 |                    77 |             94 |         8.27 |      4.26 |          25 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |           20.17s |      2.51s |      22.69s |                                    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,031 |                   159 |          1,190 |          784 |      4.36 |          26 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |           37.81s |      2.67s |      40.49s |                                    |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |               399 |                   200 |            599 |          417 |      3.49 |          39 |               0 | length          |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        | 0                          |           58.60s |      3.59s |      62.20s | reasoning-leak, cutoff             |                 |

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
  (157,739,544 bytes,
  sha256=79cffa58fb03a5b79bd95708adfc3f36d96a5930cbd6bb011d33be8240d12089)
- _MLX libmlx.dylib:_ /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib
  (21,691,776 bytes,
  sha256=efe539c4f4b73598aba1dc2f04da6bfdeed9bb2170db2da6af7757f8083c0fb3)

## Library Versions

- `numpy`: `2.4.6`
- `mlx`: `0.32.0.dev20260620+c9e8ee6b`
- `mlx-vlm`: `0.6.3`
- `mlx-lm`: `0.31.3`
- `mlx-audio`: `0.4.4`
- `huggingface-hub`: `1.20.1`
- `transformers`: `5.12.1`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-06-20 00:48:07 BST_
