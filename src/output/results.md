# Model Performance Results

_Generated on 2026-04-06 17:15:49 BST_

## 🎯 Action Snapshot

- _Framework/runtime failures:_ 3 (top owners: huggingface-hub=2,
  model-config=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=11, clean outputs=19/48.
- _Useful now:_ 15 clean A/B model(s) worth first review.
- _Review watchlist:_ 30 model(s) with breaking or lower-value output.
- _Preflight compatibility:_ 1 informational warning(s); do not treat these
  alone as run failures.
- _Escalate only if:_ they line up with unexpected TF/Flax/JAX imports,
  startup hangs, or backend/runtime crashes.
- _Quality signal frequency:_ cutoff=15, harness=11, repetitive=9,
  formatting=5, reasoning_leak=4, degeneration=3.
- _Runtime pattern:_ decode dominates measured phase time (63%; 39/51 measured
  model(s)).
- _Phase totals:_ model load=511.16s, prompt prep=0.15s, decode=889.02s,
  cleanup=4.81s.
- _What this likely means:_ Most measured runtime is spent inside generation
  rather than load or prompt setup.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=48, exception=3.

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/FastVLM-0.5B-bf16` (356.7 tps)
- **💾 Most efficient:** `mlx-community/nanoLLaVA-1.5-4bit` (1.9 GB)
- **⚡ Fastest load:** `qnguyen3/nanoLLaVA` (0.72s)
- **📊 Average TPS:** 84.3 across 48 models

## 📈 Resource Usage

- **Total peak memory:** 906.6 GB
- **Average peak memory:** 18.9 GB
- **Memory efficiency:** 232 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 21 | ✅ B: 6 | 🟡 C: 9 | 🟠 D: 1 | ❌ F: 11

**Average Utility Score:** 58/100

- **Best for cataloging:** `mlx-community/Qwen3.5-27B-4bit` (🏆 A, 100/100)
- **Worst for cataloging:** `mlx-community/LFM2-VL-1.6B-8bit` (❌ F, 0/100)

### ⚠️ 12 Models with Low Utility (D/F)

- `Qwen/Qwen3-VL-2B-Instruct`: ❌ F (16/100) - Output lacks detail
- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`: ❌ F (25/100) - Lacks visual description of image
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Idefics3-8B-Llama3-bf16`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: ❌ F (25/100) - Lacks visual description of image
- `mlx-community/LFM2-VL-1.6B-8bit`: ❌ F (0/100) - Empty or minimal output
- `mlx-community/Phi-3.5-vision-instruct-bf16`: ❌ F (17/100) - Output lacks detail
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (5/100) - Output too short to be useful
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/gemma-3n-E2B-4bit`: ❌ F (0/100) - Empty or minimal output
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (0/100) - Empty or minimal output
- `mlx-community/pixtral-12b-bf16`: 🟠 D (40/100) - Lacks visual description of image

## ⚠️ Quality Issues

- **❌ Failed Models (3):**
  - `mlx-community/MolmoPoint-8B-fp16` (`Processor Error`)
  - `mlx-community/Qwen3.5-35B-A3B-4bit` (`Model Error`)
  - `mlx-community/Qwen3.5-35B-A3B-6bit` (`Model Error`)
- **🔄 Repetitive Output (9):**
  - `HuggingFaceTB/SmolVLM-Instruct` (token: `phrase: "treasured treasured treasured ..."`)
  - `jqlive/Kimi-VL-A3B-Thinking-2506-6bit` (token: `答案内容1.`)
  - `mlx-community/InternVL3-8B-bf16` (token: `phrase: "strugg strugg strugg strugg..."`)
  - `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16` (token: `0.`)
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit` (token: `phrase: "will be used to..."`)
  - `mlx-community/Qwen3-VL-2B-Thinking-bf16` (token: `phrase: "well-ventilated, and stylishly..."`)
  - `mlx-community/SmolVLM-Instruct-bf16` (token: `phrase: "treasured treasured treasured ..."`)
  - `mlx-community/paligemma2-3b-ft-docci-448-bf16` (token: `phrase: "black and white and..."`)
  - `qnguyen3/nanoLLaVA` (token: `Baz`)
- **📝 Formatting Issues (5):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-Flash-mxfp4`
  - `mlx-community/GLM-4.6V-nvfp4`
  - `mlx-community/Idefics3-8B-Llama3-bf16`
  - `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 84.3 | Min: 4.91 | Max: 357
- **Peak Memory**: Avg: 19 | Min: 1.9 | Max: 78
- **Total Time**: Avg: 26.44s | Min: 1.88s | Max: 174.95s
- **Generation Time**: Avg: 18.52s | Min: 0.35s | Max: 109.21s
- **Model Load Time**: Avg: 7.74s | Min: 0.72s | Max: 111.70s

### ⏱ Runtime Interpretation

- **Runtime pattern:** decode dominates measured phase time (63%; 39/51 measured model(s)).
- **Phase totals:** model load=511.16s, prompt prep=0.15s, decode=889.02s, cleanup=4.81s.
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.
- **Termination reasons:** completed=48, exception=3.

### ⏱ Timing Snapshot

- **Validation overhead:** 8.69s total (avg 0.17s across 51 model(s)).
- **First-token latency:** Avg 9.74s | Min 0.06s | Max 79.44s across 48 model(s).

## ✅ Recommended Models

Quick picks based on cached quality, speed, and memory signals from this run.

- _Best cataloging quality:_ [`mlx-community/Qwen3.5-27B-4bit`](model_gallery.md#model-mlx-community-qwen35-27b-4bit)
  (A 100/100 | Gen 28.5 TPS | Peak 26 | A 100/100 | Special control token
  &lt;/think&gt; appeared in generated text. | hit token cap (500) | nontext
  prompt burden=100%)
- _Fastest generation:_ [`mlx-community/FastVLM-0.5B-bf16`](model_gallery.md#model-mlx-community-fastvlm-05b-bf16)
  (A 80/100 | Gen 357 TPS | Peak 2.2 | A 80/100 | hit token cap (500) |
  output/prompt=2083.33% | nontext prompt burden=83%)
- _Lowest memory footprint:_ [`mlx-community/nanoLLaVA-1.5-4bit`](model_gallery.md#model-mlx-community-nanollava-15-4bit)
  (A 80/100 | Gen 345 TPS | Peak 1.9 | A 80/100 | nontext prompt burden=80%)
- _Best balance:_ [`mlx-community/Molmo-7B-D-0924-8bit`](model_gallery.md#model-mlx-community-molmo-7b-d-0924-8bit)
  (A 90/100 | Gen 52.4 TPS | Peak 41 | A 90/100 | nontext prompt burden=100%)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (3):_ [`mlx-community/MolmoPoint-8B-fp16`](model_gallery.md#model-mlx-community-molmopoint-8b-fp16),
  [`mlx-community/Qwen3.5-35B-A3B-4bit`](model_gallery.md#model-mlx-community-qwen35-35b-a3b-4bit),
  [`mlx-community/Qwen3.5-35B-A3B-6bit`](model_gallery.md#model-mlx-community-qwen35-35b-a3b-6bit).
  Example: `Processor Error`.
- _🔄 Repetitive Output (9):_ [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](model_gallery.md#model-jqlive-kimi-vl-a3b-thinking-2506-6bit),
  [`mlx-community/InternVL3-8B-bf16`](model_gallery.md#model-mlx-community-internvl3-8b-bf16),
  [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](model_gallery.md#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16),
  +5 more. Example: token: `phrase: "treasured treasured treasured ..."`.
- _📝 Formatting Issues (5):_ [`mlx-community/GLM-4.6V-Flash-6bit`](model_gallery.md#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/GLM-4.6V-Flash-mxfp4`](model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4),
  [`mlx-community/GLM-4.6V-nvfp4`](model_gallery.md#model-mlx-community-glm-46v-nvfp4),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  +1 more.
- _Low-utility outputs (12):_ [`Qwen/Qwen3-VL-2B-Instruct`](model_gallery.md#model-qwen-qwen3-vl-2b-instruct),
  [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](model_gallery.md#model-jqlive-kimi-vl-a3b-thinking-2506-6bit),
  [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](model_gallery.md#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  +8 more. Common weakness: Output lacks detail.

## 🚨 Failures by Package (Actionable)

| Package | Failures | Error Types | Affected Models |
| --- | --- | --- | --- |
| `huggingface-hub` | 2 | Model Error | `mlx-community/Qwen3.5-35B-A3B-4bit`, `mlx-community/Qwen3.5-35B-A3B-6bit` |
| `model-config` | 1 | Processor Error | `mlx-community/MolmoPoint-8B-fp16` |

### Actionable Items by Package

#### huggingface-hub

- mlx-community/Qwen3.5-35B-A3B-4bit (Model Error)
  - Error: `Model loading failed: Server disconnected without sending a response.`
  - Type: `ValueError`
- mlx-community/Qwen3.5-35B-A3B-6bit (Model Error)
  - Error: `Model loading failed: Server disconnected without sending a response.`
  - Type: `ValueError`

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

_Overall runtime:_ 1414.42s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Token |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                    |   Error Package |
|:--------------------------------------------------------|--------:|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:----------------------------------|----------------:|
| `mlx-community/MolmoPoint-8B-fp16`                      |         |                   |                       |                |              |           |             |                  |      5.86s |       6.03s |                                   |    model-config |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |         |                   |                       |                |              |           |             |                  |     48.57s |      48.77s |                                   | huggingface-hub |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |         |                   |                       |                |              |           |             |                  |     85.23s |      85.39s |                                   | huggingface-hub |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |       7 |               275 |                     4 |            279 |         3481 |       333 |         2.9 |            0.35s |      2.49s |       3.01s | ⚠️harness(prompt_template)        |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |       1 |               264 |                     2 |            266 |         1152 |      76.1 |           6 |            0.57s |      1.66s |       2.42s | ⚠️harness(prompt_template)        |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |  32,007 |               768 |                    16 |            784 |         3400 |      60.4 |         9.4 |            0.78s |      1.24s |       2.19s | refusal(explicit_refusal)         |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | 151,645 |                20 |                   147 |            167 |          272 |       345 |         1.9 |            0.87s |      0.81s |       1.88s |                                   |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |       1 |             1,029 |                     2 |          1,031 |          920 |      10.6 |          26 |            1.69s |      3.53s |       5.39s | ⚠️harness(prompt_template)        |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |       7 |                85 |                   287 |            372 |         1345 |       188 |         3.8 |            1.84s |      1.13s |       3.15s |                                   |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |       1 |             4,101 |                    22 |          4,123 |         3793 |      59.6 |         4.6 |            1.86s |      1.31s |       3.35s |                                   |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |     576 |                24 |                   500 |            524 |          254 |       357 |         2.2 |            2.23s |     21.50s |      23.92s | cutoff, verbose                   |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |       2 |             2,629 |                   342 |          2,971 |         2755 |       186 |         7.8 |            3.19s |      2.69s |       6.06s |                                   |                 |
| `mlx-community/InternVL3-14B-8bit`                      | 151,645 |             1,804 |                    69 |          1,873 |         1332 |      32.3 |          18 |            3.91s |      1.91s |       5.99s |                                   |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |       2 |             2,174 |                    54 |          2,228 |          782 |        60 |         9.7 |            4.20s |      1.09s |       5.49s |                                   |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |       2 |             2,630 |                   133 |          2,763 |         1349 |      63.4 |          13 |            4.43s |      4.11s |       8.71s |                                   |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |  49,153 |                95 |                   500 |            595 |          301 |       129 |         5.5 |            4.68s |      2.63s |       7.48s | formatting                        |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |       2 |             2,630 |                   166 |          2,796 |         1356 |      66.5 |          13 |            4.83s |      7.15s |      12.16s |                                   |                 |
| `mlx-community/pixtral-12b-8bit`                        |       2 |             2,823 |                   109 |          2,932 |         1693 |      38.3 |          16 |            4.89s |      1.75s |       6.82s |                                   |                 |
| `qnguyen3/nanoLLaVA`                                    |  91,711 |                20 |                   500 |            520 |          172 |       113 |         3.9 |            4.95s |      0.72s |       5.83s | repetitive(Baz), cutoff           |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |  10,726 |             1,037 |                   500 |          1,537 |         1328 |       133 |          18 |            4.95s |      2.41s |       7.52s | repetitive(答案内容1.), cutoff    |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |  43,227 |             1,195 |                   500 |          1,695 |         3086 |       123 |         5.5 |            4.96s |      2.57s |       7.69s | repetitive(phrase: "treasur...    |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |  43,227 |             1,195 |                   500 |          1,695 |         3158 |       123 |         5.5 |            4.96s |      0.88s |       6.01s | repetitive(phrase: "treasur...    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |       1 |             1,029 |                   127 |          1,156 |         1315 |        34 |          11 |            4.98s |      2.46s |       7.62s |                                   |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |   2,040 |             1,037 |                   500 |          1,537 |         1357 |       120 |          22 |            5.37s |      4.32s |       9.86s | repetitive(phrase: "will be...    |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |     106 |               272 |                   240 |            512 |          855 |      47.8 |          17 |            5.66s |      2.44s |       8.28s |                                   |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |     106 |               273 |                   174 |            447 |          396 |      30.6 |          19 |            6.71s |      2.61s |       9.52s |                                   |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |       2 |             2,097 |                   119 |          2,216 |          724 |      31.5 |          22 |            7.07s |      3.79s |      11.03s | ⚠️harness(encoding)               |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      | 128,009 |                14 |                   121 |            135 |         10.4 |      22.1 |          15 |            7.15s |      2.35s |       9.67s |                                   |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |      15 |             1,037 |                   500 |          1,537 |         1346 |      80.7 |          37 |            7.51s |      5.40s |      13.08s | repetitive(0.), degeneration, ... |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |     525 |               768 |                   500 |          1,268 |         3183 |      55.8 |         9.3 |            9.50s |     20.52s |      30.19s | ⚠️harness(stop_token), ...        |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    | 151,336 |             6,155 |                   319 |          6,474 |         1185 |      70.1 |         8.4 |           10.04s |      7.84s |      18.04s | ⚠️harness(stop_token), ...        |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     | 151,336 |             6,155 |                   303 |          6,458 |         1157 |      54.9 |          11 |           11.15s |     13.85s |      25.18s | ⚠️harness(stop_token), ...        |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |     326 |             1,293 |                   500 |          1,793 |          504 |      58.9 |          60 |           11.64s |      9.33s |      21.15s | cutoff                            |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |   6,906 |             2,915 |                   500 |          3,415 |         1498 |      42.2 |          15 |           14.17s |     11.32s |      25.66s | reasoning-leak, cutoff            |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | 128,256 |             2,326 |                   500 |          2,826 |         2276 |      32.2 |          19 |           17.04s |      9.40s |      26.62s | formatting                        |                 |
| `mlx-community/InternVL3-8B-bf16`                       | 151,645 |             1,804 |                   329 |          2,133 |         2765 |      33.6 |          18 |           18.68s |      4.37s |      23.22s | repetitive(phrase: "strugg...     |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |      15 |            16,237 |                   500 |         16,737 |         1238 |      89.7 |         8.6 |           19.41s |      0.97s |      20.55s | degeneration, cutoff              |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |   6,188 |            16,239 |                   500 |         16,739 |          932 |      88.5 |         8.6 |           23.84s |     12.09s |      36.12s | ⚠️harness(long_context), ...      |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    | 151,643 |             1,207 |                   290 |          1,497 |         64.2 |      52.4 |          41 |           25.00s |      1.68s |      26.85s |                                   |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |     578 |             1,029 |                   500 |          1,529 |         2673 |      19.3 |          10 |           26.74s |      1.59s |      28.51s | repetitive(phrase: "black a...    |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      | 151,645 |            16,248 |                   476 |         16,724 |          903 |      54.2 |          13 |           27.46s |      3.66s |      31.30s |                                   |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |     279 |             6,155 |                   500 |          6,655 |          419 |      37.4 |          78 |           28.35s |     27.64s |      56.18s | ⚠️harness(stop_token), ...        |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 | 236,771 |               273 |                   500 |            773 |          315 |      17.1 |          33 |           30.39s |      5.30s |      35.87s | degeneration, cutoff              |                 |
| `mlx-community/pixtral-12b-bf16`                        |   6,780 |             2,823 |                   500 |          3,323 |         1656 |      20.3 |          28 |           30.61s |      2.84s |      33.64s |                                   |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | 128,009 |                15 |                   193 |            208 |         8.74 |      4.91 |          25 |           41.36s |      2.73s |      44.25s |                                   |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    | 151,643 |             1,207 |                   358 |          1,565 |           62 |        30 |          48 |           53.50s |     21.78s |      75.45s |                                   |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | 151,645 |            16,248 |                     6 |         16,254 |          276 |       224 |         5.1 |           59.63s |      1.68s |      61.48s | ⚠️harness(long_context), ...      |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |     318 |            16,239 |                   500 |         16,739 |          303 |      89.6 |          11 |           59.95s |      3.42s |      63.56s |                                   |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |  26,025 |            16,239 |                   500 |         16,739 |          300 |      61.8 |          76 |           63.07s |    111.70s |     174.95s | cutoff                            |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |     513 |            16,239 |                   500 |         16,739 |          215 |      28.5 |          26 |           93.70s |      2.68s |      96.58s | ⚠️harness(stop_token)             |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |   1,216 |            16,239 |                   500 |         16,739 |          204 |      17.2 |          38 |          109.21s |     10.17s |     119.56s | cutoff                            |                 |

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
- `mlx`: `0.31.2.dev20260406+6a9a121d`
- `mlx-vlm`: `0.4.4`
- `mlx-lm`: `0.31.2`
- `huggingface-hub`: `1.9.0`
- `transformers`: `5.5.0`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-04-06 17:15:49 BST_
