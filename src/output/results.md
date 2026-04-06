# Model Performance Results

_Generated on 2026-04-06 23:09:06 BST_

## 🎯 Action Snapshot

- _Framework/runtime failures:_ 1 (top owners: model-config=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=12, clean outputs=19/50.
- _Useful now:_ 15 clean A/B model(s) worth first review.
- _Review watchlist:_ 32 model(s) with breaking or lower-value output.
- _Preflight compatibility:_ 1 informational warning(s); do not treat these
  alone as run failures.
- _Escalate only if:_ they line up with unexpected TF/Flax/JAX imports,
  startup hangs, or backend/runtime crashes.
- _Quality signal frequency:_ cutoff=16, harness=12, repetitive=10,
  degeneration=7, formatting=6, reasoning_leak=4.
- _Runtime pattern:_ decode dominates measured phase time (90%; 46/51 measured
  model(s)).
- _Phase totals:_ model load=107.60s, prompt prep=0.15s, decode=1000.34s,
  cleanup=4.90s.
- _What this likely means:_ Most measured runtime is spent inside generation
  rather than load or prompt setup.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=50, exception=1.

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/nanoLLaVA-1.5-4bit` (380.0 tps)
- **💾 Most efficient:** `mlx-community/nanoLLaVA-1.5-4bit` (1.8 GB)
- **⚡ Fastest load:** `mlx-community/nanoLLaVA-1.5-4bit` (0.45s)
- **📊 Average TPS:** 85.7 across 50 models

## 📈 Resource Usage

- **Total peak memory:** 994.5 GB
- **Average peak memory:** 19.9 GB
- **Memory efficiency:** 256 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 13 | ✅ B: 14 | 🟡 C: 12 | ❌ F: 11

**Average Utility Score:** 58/100

- **Best for cataloging:** `mlx-community/Qwen3.5-27B-4bit` (🏆 A, 90/100)
- **Worst for cataloging:** `mlx-community/gemma-3n-E2B-4bit` (❌ F, 0/100)

### ⚠️ 11 Models with Low Utility (D/F)

- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: ❌ F (25/100) - Lacks visual description of image
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Idefics3-8B-Llama3-bf16`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: ❌ F (25/100) - Lacks visual description of image
- `mlx-community/LFM2-VL-1.6B-8bit`: ❌ F (16/100) - Output lacks detail
- `mlx-community/Phi-3.5-vision-instruct-bf16`: ❌ F (17/100) - Output lacks detail
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (5/100) - Output too short to be useful
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/gemma-3n-E2B-4bit`: ❌ F (0/100) - Empty or minimal output
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (0/100) - Empty or minimal output
- `mlx-community/paligemma2-3b-pt-896-4bit`: ❌ F (17/100) - Output lacks detail

## ⚠️ Quality Issues

- **❌ Failed Models (1):**
  - `mlx-community/MolmoPoint-8B-fp16` (`Processor Error`)
- **🔄 Repetitive Output (10):**
  - `HuggingFaceTB/SmolVLM-Instruct` (token: `unt`)
  - `Qwen/Qwen3-VL-2B-Instruct` (token: `phrase: "and consistent, and consistent..."`)
  - `jqlive/Kimi-VL-A3B-Thinking-2506-6bit` (token: `1.`)
  - `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX` (token: `phrase: "we can also mention..."`)
  - `mlx-community/InternVL3-8B-bf16` (token: `phrase: "rencontre rencontre rencontre ..."`)
  - `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16` (token: `0.`)
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit` (token: `phrase: "1. these information is..."`)
  - `mlx-community/SmolVLM-Instruct-bf16` (token: `unt`)
  - `mlx-community/paligemma2-3b-ft-docci-448-bf16` (token: `phrase: "the light fixtures are..."`)
  - `qnguyen3/nanoLLaVA` (token: `Baz`)
- **📝 Formatting Issues (6):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-Flash-mxfp4`
  - `mlx-community/GLM-4.6V-nvfp4`
  - `mlx-community/Idefics3-8B-Llama3-bf16`
  - `mlx-community/Molmo-7B-D-0924-bf16`
  - `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 85.7 | Min: 5.03 | Max: 380
- **Peak Memory**: Avg: 20 | Min: 1.8 | Max: 78
- **Total Time**: Avg: 22.29s | Min: 1.22s | Max: 109.61s
- **Generation Time**: Avg: 20.01s | Min: 0.34s | Max: 106.27s
- **Model Load Time**: Avg: 2.09s | Min: 0.45s | Max: 10.39s

### ⏱ Runtime Interpretation

- **Runtime pattern:** decode dominates measured phase time (90%; 46/51 measured model(s)).
- **Phase totals:** model load=107.60s, prompt prep=0.15s, decode=1000.34s, cleanup=4.90s.
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.
- **Termination reasons:** completed=50, exception=1.

### ⏱ Timing Snapshot

- **Validation overhead:** 9.09s total (avg 0.18s across 51 model(s)).
- **First-token latency:** Avg 12.07s | Min 0.05s | Max 77.55s across 50 model(s).

## ✅ Recommended Models

Quick picks based on cached quality, speed, and memory signals from this run.

- _Best cataloging quality:_ [`mlx-community/Qwen3.5-27B-4bit`](model_gallery.md#model-mlx-community-qwen35-27b-4bit)
  (A 90/100 | Gen 24.0 TPS | Peak 26 | A 90/100 | Special control token
  &lt;/think&gt; appeared in generated text. | hit token cap (500) | nontext
  prompt burden=100%)
- _Fastest generation:_ [`mlx-community/nanoLLaVA-1.5-4bit`](model_gallery.md#model-mlx-community-nanollava-15-4bit)
  (A 80/100 | Gen 380 TPS | Peak 1.8 | A 80/100 | nontext prompt burden=80%)
- _Lowest memory footprint:_ [`mlx-community/nanoLLaVA-1.5-4bit`](model_gallery.md#model-mlx-community-nanollava-15-4bit)
  (A 80/100 | Gen 380 TPS | Peak 1.8 | A 80/100 | nontext prompt burden=80%)
- _Best balance:_ [`mlx-community/nanoLLaVA-1.5-4bit`](model_gallery.md#model-mlx-community-nanollava-15-4bit)
  (A 80/100 | Gen 380 TPS | Peak 1.8 | A 80/100 | nontext prompt burden=80%)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (1):_ [`mlx-community/MolmoPoint-8B-fp16`](model_gallery.md#model-mlx-community-molmopoint-8b-fp16).
  Example: `Processor Error`.
- _🔄 Repetitive Output (10):_ [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`Qwen/Qwen3-VL-2B-Instruct`](model_gallery.md#model-qwen-qwen3-vl-2b-instruct),
  [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](model_gallery.md#model-jqlive-kimi-vl-a3b-thinking-2506-6bit),
  [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](model_gallery.md#model-mlx-community-apriel-15-15b-thinker-6bit-mlx),
  +6 more. Example: token: `unt`.
- _📝 Formatting Issues (6):_ [`mlx-community/GLM-4.6V-Flash-6bit`](model_gallery.md#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/GLM-4.6V-Flash-mxfp4`](model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4),
  [`mlx-community/GLM-4.6V-nvfp4`](model_gallery.md#model-mlx-community-glm-46v-nvfp4),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  +2 more.
- _Low-utility outputs (11):_ [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](model_gallery.md#model-jqlive-kimi-vl-a3b-thinking-2506-6bit),
  [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](model_gallery.md#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](model_gallery.md#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16),
  +7 more. Common weakness: Lacks visual description of image.

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

<!-- markdownlint-disable MD028 MD037 -->
>
> Describe this picture
<!-- markdownlint-enable MD028 MD037 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 1122.73s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Token |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                    |   Error Package |
|:--------------------------------------------------------|--------:|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:----------------------------------|----------------:|
| `mlx-community/MolmoPoint-8B-fp16`                      |         |                   |                       |                |              |           |             |                  |      2.88s |       3.05s |                                   |    model-config |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |       7 |               271 |                     7 |            278 |         3373 |       329 |         2.9 |            0.34s |      0.70s |       1.22s |                                   |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |       1 |               264 |                     2 |            266 |         1128 |        76 |           6 |            0.53s |      1.45s |       2.17s | ⚠️harness(prompt_template)        |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |  32,007 |               768 |                    16 |            784 |         3438 |      62.8 |         9.4 |            0.73s |      0.95s |       1.86s | refusal(explicit_refusal)         |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | 151,645 |                20 |                   168 |            188 |          302 |       380 |         1.8 |            0.81s |      0.45s |       1.44s |                                   |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |       7 |                79 |                   137 |            216 |         1723 |       190 |         3.7 |            1.02s |      0.62s |       1.82s |                                   |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |       1 |             4,101 |                    12 |          4,113 |         3804 |      62.1 |         4.6 |            1.60s |      1.15s |       2.93s | ⚠️harness(long_context), ...      |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |       1 |             1,029 |                     2 |          1,031 |          916 |      10.8 |          26 |            1.63s |      2.47s |       4.29s | ⚠️harness(prompt_template)        |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       | 151,645 |                24 |                   420 |            444 |          251 |       362 |         2.1 |            1.73s |      0.57s |       2.48s |                                   |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |       2 |             3,619 |                   202 |          3,821 |         1947 |       177 |          13 |            3.35s |      1.21s |       4.74s |                                   |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |       2 |             2,942 |                    98 |          3,040 |         1554 |      63.5 |         9.7 |            3.83s |      0.98s |       4.99s |                                   |                 |
| `mlx-community/InternVL3-14B-8bit`                      | 151,645 |             2,572 |                    50 |          2,622 |         1242 |      32.3 |          18 |            3.96s |      1.89s |       6.03s |                                   |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |  49,153 |                95 |                   500 |            595 |          241 |       130 |         5.8 |            4.66s |      0.77s |       5.61s | formatting                        |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |     106 |               272 |                   204 |            476 |          802 |      49.1 |          17 |            4.78s |      2.23s |       7.20s |                                   |                 |
| `qnguyen3/nanoLLaVA`                                    |  91,711 |                20 |                   500 |            520 |          173 |       115 |         3.9 |            4.79s |      0.59s |       5.56s | repetitive(Baz), cutoff           |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |   1,597 |             1,559 |                   500 |          2,059 |         3162 |       125 |         5.8 |            4.91s |      0.62s |       5.70s | repetitive(unt), cutoff           |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |   1,597 |             1,559 |                   500 |          2,059 |         2424 |       120 |         5.8 |            5.22s |      0.64s |       6.04s | repetitive(unt), cutoff           |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |     220 |             1,047 |                   500 |          1,547 |          925 |       132 |          18 |            5.33s |      2.01s |       7.52s | repetitive(1.), degeneration, ... |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |       1 |             1,029 |                   142 |          1,171 |         1316 |      34.2 |          11 |            5.36s |      1.56s |       7.10s |                                   |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |   2,294 |             1,047 |                   500 |          1,547 |         1361 |       113 |          22 |            5.60s |      3.32s |       9.10s | repetitive(phrase: "1. thes...    |                 |
| `mlx-community/pixtral-12b-8bit`                        |       2 |             4,167 |                   136 |          4,303 |         1838 |      39.2 |          18 |            6.05s |      1.67s |       7.90s |                                   |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |       2 |             3,620 |                   193 |          3,813 |         1005 |      64.4 |          18 |            6.94s |      1.54s |       8.68s |                                   |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |      15 |             1,047 |                   500 |          1,547 |         1330 |      80.6 |          37 |            7.50s |      3.45s |      11.14s | repetitive(0.), degeneration, ... |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |     106 |               273 |                   119 |            392 |          320 |      17.8 |          33 |            7.83s |      3.38s |      11.40s |                                   |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |     106 |               273 |                   212 |            485 |          384 |        31 |          19 |            7.84s |      2.35s |      10.39s |                                   |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |       2 |             3,620 |                   279 |          3,899 |          973 |      59.2 |          19 |            8.79s |      1.48s |      10.45s |                                   |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |     525 |               768 |                   500 |          1,268 |         3025 |      56.6 |         9.3 |            9.35s |      0.88s |      10.43s | ⚠️harness(stop_token), ...        |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |       2 |             3,087 |                   160 |          3,247 |          595 |      31.1 |          27 |           10.70s |      2.06s |      12.98s | ⚠️harness(encoding), ...          |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |     476 |             6,095 |                   500 |          6,595 |         1118 |      73.3 |         8.4 |           12.54s |      1.26s |      13.98s | ⚠️harness(stop_token), ...        |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      | 128,009 |                14 |                   213 |            227 |         10.2 |      19.5 |          15 |           12.58s |      1.53s |      14.30s |                                   |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |     274 |             1,271 |                   500 |          1,771 |          216 |      59.5 |          60 |           14.82s |      9.36s |      24.37s | ⚠️harness(stop_token), ...        |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |  53,763 |             6,095 |                   500 |          6,595 |         1073 |      55.2 |          11 |           15.03s |      1.38s |      16.60s | ⚠️harness(stop_token), ...        |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |   7,693 |             4,259 |                   500 |          4,759 |         1476 |      41.1 |          15 |           15.35s |      1.58s |      17.12s | repetitive(phrase: "we can...     |                 |
| `mlx-community/InternVL3-8B-bf16`                       | 151,645 |             2,572 |                   434 |          3,006 |         2708 |      34.8 |          18 |           15.79s |      1.89s |      17.86s | repetitive(phrase: "rencont...    |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | 128,256 |             3,038 |                   500 |          3,538 |         2277 |      32.5 |          19 |           17.10s |      1.88s |      19.16s | formatting                        |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    | 151,643 |             1,199 |                    90 |          1,289 |         78.7 |        50 |          41 |           17.55s |      1.29s |      19.02s |                                   |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      | 151,645 |            16,280 |                   218 |         16,498 |         1057 |      57.4 |          13 |           19.82s |      1.15s |      21.15s |                                   |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |   7,481 |            16,271 |                   500 |         16,771 |         1153 |      87.2 |         8.6 |           20.59s |      0.95s |      21.74s | cutoff                            |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |     323 |            16,269 |                   500 |         16,769 |         1157 |        88 |         8.6 |           20.62s |      0.76s |      21.59s | ⚠️harness(long_context), ...      |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | 128,009 |                15 |                   112 |            127 |         8.48 |      5.03 |          25 |           24.32s |      2.21s |      26.71s |                                   |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |     714 |             1,029 |                   500 |          1,529 |         2728 |      19.5 |          10 |           26.35s |      1.39s |      27.94s | repetitive(phrase: "the lig...    |                 |
| `mlx-community/pixtral-12b-bf16`                        |   6,780 |             4,167 |                   500 |          4,667 |         1997 |        20 |          29 |           31.08s |      2.56s |      33.82s |                                   |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |     387 |             6,095 |                   500 |          6,595 |          287 |      34.4 |          78 |           36.09s |      9.10s |      45.38s | ⚠️harness(stop_token), ...        |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    | 109,966 |             1,199 |                   500 |          1,699 |         72.9 |      30.8 |          48 |           49.86s |      1.87s |      51.91s | cutoff, formatting                |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |   1,637 |            16,271 |                   500 |         16,771 |          309 |      89.1 |          35 |           59.04s |      3.15s |      62.37s | cutoff                            |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | 151,645 |            16,280 |                     4 |         16,284 |          272 |       240 |         5.1 |           60.41s |      0.60s |      61.20s | ⚠️harness(long_context), ...      |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |     579 |            16,271 |                   500 |         16,771 |          293 |      64.9 |          76 |           63.96s |     10.39s |      74.52s | degeneration, cutoff              |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |   1,301 |            16,271 |                   500 |         16,771 |          272 |       106 |          26 |           65.23s |      2.52s |      67.93s | cutoff                            |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |     314 |            16,271 |                   500 |         16,771 |          230 |      91.7 |          12 |           76.91s |      1.30s |      78.39s | degeneration, cutoff              |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |   3,221 |            16,271 |                   500 |         16,771 |          225 |        24 |          26 |           93.85s |      2.48s |      96.54s | ⚠️harness(stop_token)             |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |   4,022 |            16,271 |                   500 |         16,771 |          210 |      17.8 |          38 |          106.27s |      3.14s |     109.61s | cutoff                            |                 |

<!-- markdownlint-enable MD013 MD033 MD034 MD037 MD049 -->

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](model_gallery.md)
- _Automated review digest:_ [review.md](review.md)
- _Canonical run log:_ [check_models.log](check_models.log)

---

## System/Hardware Information

- _OS:_ Darwin 25.4.0
- _macOS Version:_ 26.4
- _SDK Version:_ 26.4
- _Xcode Version:_ 26.4
- _Xcode Build:_ 17E192
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
- `mlx`: `0.31.2.dev20260406+b98831ad`
- `mlx-vlm`: `0.4.4`
- `mlx-lm`: `0.31.2`
- `huggingface-hub`: `1.9.0`
- `transformers`: `5.5.0`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-04-06 23:09:06 BST_
