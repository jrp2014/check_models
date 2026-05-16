# Model Performance Results

_Generated on 2026-05-17 00:16:46 BST_

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ 3 (top owners: mlx=2, mlx-lm=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=9, clean outputs=5/52.
- _Useful now:_ 3 clean A/B model(s) worth first review.
- _Review watchlist:_ 48 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Quality signal frequency:_ cutoff=31, degeneration=16, repetitive=13,
  harness=9, token_cap=7, reasoning_leak=2.

### Runtime

- _Runtime pattern:_ post-prefill decode dominates measured phase time (62%;
  38/55 measured model(s)).
- _Phase totals:_ model load=104.65s, local prompt prep=0.16s, upstream
  prefill / first-token=85.59s, post-prefill decode=323.98s, cleanup=5.61s.
- _Generation total:_ 409.57s across 52 model(s); upstream prefill /
  first-token split available for 52/52 model(s).
- _What this likely means:_ Most measured runtime is spent generating after
  the first token is available.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=13, exception=3, max_tokens=39.
- _Validation overhead:_ 9.54s total (avg 0.17s across 55 model(s)).
- _Upstream prefill / first-token latency:_ Avg 1.65s | Min 0.04s | Max 18.47s
  across 52 model(s).

## 🏆 Performance Highlights

- **Fastest:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (551.0 tps)
- **💾 Most efficient:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (1.4 GB)
- **⚡ Fastest load:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (0.32s)
- **📊 Average TPS:** 86.9 across 52 models

## 📈 Resource Usage

- **Total peak memory:** 1010.7 GB
- **Average peak memory:** 19.4 GB
- **Memory efficiency:** 53 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** ✅ B: 7 | 🟡 C: 3 | 🟠 D: 20 | ❌ F: 22

**Average Utility Score:** 35/100

- **Best for cataloging:** `jqlive/Kimi-VL-A3B-Thinking-2506-6bit` (✅ B, 75/100)
- **Best descriptions:** `qnguyen3/nanoLLaVA` (90/100)
- **Best keywording:** `mlx-community/gemma-4-31b-it-4bit` (44/100)
- **Worst for cataloging:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (❌ F, 0/100)

### ⚠️ 42 Models with Low Utility (D/F)

- `HuggingFaceTB/SmolVLM-Instruct`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `LiquidAI/LFM2.5-VL-450M-MLX-bf16`: ❌ F (0/100) - Empty or minimal output
- `meta-llama/Llama-3.2-11B-Vision-Instruct`: ❌ F (5/100) - Output too short to be useful
- `microsoft/Phi-3.5-vision-instruct`: ❌ F (30/100) - Keywords are not specific or diverse enough
- `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (0/100) - Empty or minimal output
- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/GLM-4.6V-Flash-6bit`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/GLM-4.6V-Flash-mxfp4`: ❌ F (5/100) - Output too short to be useful
- `mlx-community/GLM-4.6V-nvfp4`: ❌ F (5/100) - Output too short to be useful
- `mlx-community/Idefics3-8B-Llama3-bf16`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/InternVL3-14B-8bit`: ❌ F (10/100) - Keywords are not specific or diverse enough
- `mlx-community/InternVL3-8B-bf16`: ❌ F (19/100) - Output lacks detail
- `mlx-community/LFM2-VL-1.6B-8bit`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`: 🟠 D (50/100) - Keywords are not specific or diverse enough
- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`: 🟠 D (50/100) - Keywords are not specific or diverse enough
- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`: 🟠 D (42/100) - Keywords are not specific or diverse enough
- `mlx-community/Molmo-7B-D-0924-8bit`: ❌ F (31/100) - Keywords are not specific or diverse enough
- `mlx-community/Phi-3.5-vision-instruct-bf16`: ❌ F (30/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: 🟠 D (50/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen3.5-27B-4bit`: ❌ F (6/100) - Output too short to be useful
- `mlx-community/Qwen3.5-27B-mxfp8`: 🟠 D (50/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen3.5-35B-A3B-4bit`: ❌ F (5/100) - Output too short to be useful
- `mlx-community/Qwen3.5-35B-A3B-6bit`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen3.5-35B-A3B-bf16`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Qwen3.5-9B-MLX-4bit`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/SmolVLM-Instruct-bf16`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/gemma-3-27b-it-qat-4bit`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/gemma-3-27b-it-qat-8bit`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/gemma-3n-E2B-4bit`: ❌ F (5/100) - Output too short to be useful
- `mlx-community/gemma-3n-E4B-it-bf16`: ❌ F (25/100) - Keywords are not specific or diverse enough
- `mlx-community/gemma-4-26b-a4b-it-4bit`: ❌ F (1/100) - Output too short to be useful
- `mlx-community/gemma-4-31b-bf16`: 🟠 D (44/100) - Keywords are not specific or diverse enough
- `mlx-community/gemma-4-31b-it-4bit`: 🟠 D (43/100) - Lacks visual description of image
- `mlx-community/llava-v1.6-mistral-7b-8bit`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (5/100) - Output too short to be useful
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (5/100) - Output too short to be useful
- `mlx-community/paligemma2-3b-pt-896-4bit`: ❌ F (16/100) - Output lacks detail
- `mlx-community/pixtral-12b-8bit`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/pixtral-12b-bf16`: 🟠 D (45/100) - Keywords are not specific or diverse enough

## ⚠️ Quality Issues

- **❌ Failed Models (3):**
  - `facebook/pe-av-large` (`Model Error`)
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit` (`Model Error`)
  - `mlx-community/LFM2.5-VL-1.6B-bf16` (`Weight Mismatch`)
- **🔄 Repetitive Output (13):**
  - `mlx-community/FastVLM-0.5B-bf16` (token: `phrase: "2000, 2000, 2000, 2000,..."`)
  - `mlx-community/GLM-4.6V-Flash-6bit` (token: `1.`)
  - `mlx-community/Idefics3-8B-Llama3-bf16` (token: `comm,`)
  - `mlx-community/InternVL3-14B-8bit` (token: `phrase: "2018年，所以 2018年，所以 2018年，所以 201..."`)
  - `mlx-community/LFM2-VL-1.6B-8bit` (token: `and`)
  - `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` (token: `phrase: "更 falls 更 falls..."`)
  - `mlx-community/MolmoPoint-8B-fp16` (token: `phrase: "there is a black..."`)
  - `mlx-community/Qwen2-VL-2B-Instruct-4bit` (token: `phrase: "a. a. a. a...."`)
  - `mlx-community/Qwen3.5-27B-mxfp8` (token: `===`)
  - `mlx-community/Qwen3.5-35B-A3B-6bit` (token: `7`)
  - `mlx-community/SmolVLM2-2.2B-Instruct-mlx` (token: `phrase: "saff saff saff saff..."`)
  - `mlx-community/gemma-3n-E4B-it-bf16` (token: `a__`)
  - `mlx-community/llava-v1.6-mistral-7b-8bit` (token: `phrase: "as- and lastly as..."`)
- **📝 Formatting Issues (1):**
  - `mlx-community/Idefics3-8B-Llama3-bf16`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 86.9 | Min: 4.75 | Max: 551
- **Peak Memory**: Avg: 19 | Min: 1.4 | Max: 71
- **Total Time**: Avg: 10.06s | Min: 0.80s | Max: 47.05s
- **Generation Time**: Avg: 7.88s | Min: 0.30s | Max: 43.65s
- **Model Load Time**: Avg: 2.00s | Min: 0.32s | Max: 8.57s

## ✅ Recommended Models

Quick picks based on end-to-end utility plus description and keyword strength.

- _Best end-to-end cataloging:_ [`qnguyen3/nanoLLaVA`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-qnguyen3-nanollava)
  (Utility B 75/100 | Description 90 | Keywords 0 | Speed 115 TPS | Memory 4.0
  | Caveat nontext prompt burden=73%)
- _Best descriptions:_ [`qnguyen3/nanoLLaVA`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-qnguyen3-nanollava)
  (Utility B 75/100 | Description 90 | Keywords 0 | Speed 115 TPS | Memory 4.0
  | Caveat nontext prompt burden=73%)
- _Best keywording:_ [`mlx-community/nanoLLaVA-1.5-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-nanollava-15-4bit)
  (Utility B 75/100 | Description 87 | Keywords 0 | Speed 324 TPS | Memory 1.8
  | Caveat nontext prompt burden=73%)
- _Fastest generation:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16)
  (Utility F 0/100 | Description 0 | Keywords 0 | Speed 551 TPS | Memory 1.4 |
  Caveat Output appears truncated to about 4 tokens.; nontext prompt
  burden=98%)
- _Lowest memory footprint:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16)
  (Utility F 0/100 | Description 0 | Keywords 0 | Speed 551 TPS | Memory 1.4 |
  Caveat Output appears truncated to about 4 tokens.; nontext prompt
  burden=98%)
- _Best balance:_ [`mlx-community/nanoLLaVA-1.5-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-nanollava-15-4bit)
  (Utility B 75/100 | Description 87 | Keywords 0 | Speed 324 TPS | Memory 1.8
  | Caveat nontext prompt burden=73%)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (3):_ [`facebook/pe-av-large`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-facebook-pe-av-large),
  [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-kimi-vl-a3b-thinking-8bit),
  [`mlx-community/LFM2.5-VL-1.6B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-lfm25-vl-16b-bf16).
  Example: `Model Error`.
- _🔄 Repetitive Output (13):_ [`mlx-community/FastVLM-0.5B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-fastvlm-05b-bf16),
  [`mlx-community/GLM-4.6V-Flash-6bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  [`mlx-community/InternVL3-14B-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-internvl3-14b-8bit),
  +9 more. Example: token: `phrase: "2000, 2000, 2000, 2000,..."`.
- _📝 Formatting Issues (1):_ [`mlx-community/Idefics3-8B-Llama3-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16).
- _Low-utility outputs (42):_ [`HuggingFaceTB/SmolVLM-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16),
  [`meta-llama/Llama-3.2-11B-Vision-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-meta-llama-llama-32-11b-vision-instruct),
  [`microsoft/Phi-3.5-vision-instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-microsoft-phi-35-vision-instruct),
  +38 more. Common weakness: Keywords are not specific or diverse enough.

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

_Overall runtime:_ 532.99s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                     |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:-----------------------------------|----------------:|
| `facebook/pe-av-large`                                  |                   |                       |                |              |           |             |                  |      0.12s |       1.06s |                                    |          mlx-lm |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |                   |                       |                |              |           |             |                  |      0.19s |       1.15s |                                    |             mlx |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |                   |                       |                |              |           |             |                  |      0.22s |       1.18s |                                    |             mlx |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      |               269 |                     4 |            273 |         7376 |       551 |         1.4 |            0.30s |      0.32s |       0.80s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |                22 |                   111 |            133 |          280 |       324 |         1.8 |            0.76s |      0.49s |       1.45s |                                    |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               269 |                   200 |            469 |         2976 |       334 |           3 |            0.96s |      0.55s |       1.68s | repetitive(and), cutoff            |                 |
| `qnguyen3/nanoLLaVA`                                    |                22 |                    73 |             95 |          224 |       115 |           4 |            1.08s |      0.51s |       1.78s |                                    |                 |
| `mlx-community/InternVL3-8B-bf16`                       |               781 |                    12 |            793 |         1607 |      37.4 |          17 |            1.09s |      1.68s |       2.95s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |                26 |                   200 |            226 |         89.6 |       360 |         2.2 |            1.16s |      0.59s |       1.93s | repetitive(phrase: "2000, 2...     |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,031 |                     6 |          1,037 |         1311 |      39.6 |          11 |            1.22s |      1.58s |       2.99s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |               593 |                     3 |            596 |          514 |      42.4 |          18 |            1.51s |      2.25s |       3.96s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |               803 |                   200 |          1,003 |         1986 |       241 |         2.5 |            1.52s |      0.49s |       2.19s | repetitive(phrase: "a. a. a...     |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,103 |                     9 |          4,112 |         3747 |      63.7 |         4.6 |            1.53s |      1.11s |       2.83s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |               807 |                     3 |            810 |          602 |      43.6 |          20 |            1.70s |      2.02s |       3.90s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |             1,339 |                   200 |          1,539 |         2898 |       200 |         5.4 |            1.74s |      0.92s |       2.84s | token-cap                          |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               266 |                   200 |            466 |         1091 |       123 |           6 |            2.14s |      1.39s |       3.71s | cutoff                             |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |                97 |                   200 |            297 |          280 |       131 |         5.5 |            2.24s |      0.63s |       3.04s | repetitive(phrase: "saff sa...     |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,196 |                   200 |          1,396 |         2727 |       130 |         5.5 |            2.28s |      0.71s |       3.16s | cutoff                             |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             1,196 |                   200 |          1,396 |         3000 |       123 |         5.5 |            2.36s |      0.56s |       3.10s | cutoff                             |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,031 |                     6 |          1,037 |          913 |      6.36 |          26 |            2.38s |      2.44s |       5.01s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               284 |                   200 |            484 |          620 |       118 |          17 |            2.42s |      2.32s |       4.94s | token-cap                          |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |               593 |                   200 |            793 |         1237 |       112 |          22 |            2.54s |      2.44s |       5.16s | degeneration, token-cap            |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |               593 |                   200 |            793 |         1265 |        99 |         7.2 |            2.77s |      1.32s |       4.27s | degeneration, cutoff               |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |               790 |                     8 |            798 |          279 |      47.5 |          63 |            3.28s |      5.88s |       9.35s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |               790 |                   200 |            990 |         1180 |      80.4 |         7.8 |            3.44s |      1.26s |       4.87s | degeneration, token-cap            |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |               745 |                   200 |            945 |         1394 |      73.6 |          16 |            3.63s |      2.11s |       5.91s | reasoning-leak, cutoff             |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |               593 |                   200 |            793 |         1507 |      93.1 |          30 |            3.71s |      3.08s |       6.97s | repetitive(7), degeneration, ...   |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |               770 |                   200 |            970 |         2684 |      58.5 |         9.2 |            3.99s |      0.87s |       5.03s | cutoff                             |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |               770 |                   200 |            970 |         2269 |      58.4 |         9.2 |            4.04s |      0.88s |       5.10s | cutoff                             |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |               803 |                   200 |          1,003 |         1384 |      62.8 |          10 |            4.05s |      1.11s |       5.34s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |             1,340 |                   200 |          1,540 |         1175 |      68.6 |          11 |            4.34s |      1.31s |       5.84s | degeneration, cutoff               |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |               790 |                   200 |            990 |         1107 |      59.8 |          10 |            4.35s |      1.45s |       5.98s | repetitive(1.), degeneration, ...  |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,031 |                    76 |          1,107 |         2678 |      19.6 |          10 |            4.55s |      1.44s |       6.18s |                                    |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               274 |                   200 |            474 |          821 |      48.5 |          17 |            4.73s |      2.21s |       7.13s | repetitive(a__), cutoff            |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |               825 |                   200 |          1,025 |          475 |      53.2 |          60 |            5.91s |      5.02s |      11.27s | cutoff                             |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             1,964 |                   200 |          2,164 |          771 |      61.3 |         9.7 |            6.09s |      0.90s |       7.17s | repetitive(phrase: "as- and...     |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |             1,340 |                   200 |          1,540 |         1249 |      65.7 |          11 |            6.38s |      1.33s |       7.88s | repetitive(phrase: "更 falls...    |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             2,440 |                   200 |          2,640 |         1163 |      42.7 |          14 |            7.06s |      1.57s |       8.81s | token-cap                          |                 |
| `mlx-community/InternVL3-14B-8bit`                      |               781 |                   200 |            981 |          902 |      33.2 |          17 |            7.18s |      1.73s |       9.10s | repetitive(phrase: "2018年，所...  |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |               593 |                   200 |            793 |          145 |        69 |          71 |            7.30s |      8.57s |      16.05s | degeneration, cutoff               |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               275 |                   200 |            475 |          330 |      31.1 |          19 |            7.55s |      2.30s |      10.03s | cutoff                             |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,327 |                   200 |          2,527 |         1882 |      32.5 |          19 |            7.71s |      1.85s |       9.74s | repetitive(comm,), cutoff, ...     |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               284 |                   200 |            484 |          328 |      27.6 |          19 |            8.38s |      2.51s |      11.06s | degeneration, token-cap            |                 |
| `mlx-community/pixtral-12b-8bit`                        |             2,349 |                   200 |          2,549 |         1423 |      39.1 |          16 |           10.59s |      1.69s |      12.46s | cutoff                             |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |                16 |                   200 |            216 |         11.9 |      21.8 |          15 |           10.79s |      1.52s |      12.49s | degeneration, token-cap            |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |               593 |                   200 |            793 |          452 |      19.2 |          30 |           12.04s |      3.05s |      15.29s | repetitive(===), degeneration, ... |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |               593 |                   200 |            793 |          455 |      19.1 |          30 |           12.04s |      3.07s |      15.30s | degeneration, cutoff               |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               275 |                   200 |            475 |          282 |      17.5 |          33 |           12.69s |      3.40s |      16.27s | cutoff                             |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |             1,201 |                    30 |          1,231 |         65.9 |      47.1 |          41 |           19.15s |      1.23s |      20.56s |                                    |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |             1,201 |                    48 |          1,249 |           65 |      29.6 |          48 |           20.38s |      1.68s |      22.24s |                                    |                 |
| `mlx-community/pixtral-12b-bf16`                        |             2,349 |                   200 |          2,549 |         1667 |        20 |          27 |           20.74s |      2.54s |      23.47s | cutoff                             |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               272 |                   200 |            472 |         82.4 |      7.66 |          64 |           29.71s |      6.69s |      36.58s | cutoff                             |                 |
| `mlx-community/MolmoPoint-8B-fp16`                      |             1,203 |                   200 |          1,403 |         1150 |      5.98 |          24 |           34.89s |      2.15s |      37.22s | repetitive(phrase: "there i...     |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |                17 |                   200 |            217 |         10.2 |      5.05 |          25 |           41.52s |      2.17s |      43.87s | degeneration, cutoff               |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |               745 |                   200 |            945 |          703 |      4.75 |          39 |           43.65s |      3.22s |      47.05s | reasoning-leak, cutoff             |                 |

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
- _Python Version:_ 3.13.12
- _Architecture:_ arm64
- _GPU/Chip:_ Apple M5 Max
- _GPU Cores:_ 40
- _Metal Support:_ Metal 4
- _RAM:_ 128.0 GB
- _CPU Cores (Physical):_ 18
- _CPU Cores (Logical):_ 18

## Library Versions

- `numpy`: `2.4.5`
- `mlx`: `0.32.0.dev20260516+7b7c1240`
- `mlx-vlm`: `0.5.0`
- `mlx-lm`: `0.31.3`
- `mlx-audio`: `0.4.3`
- `huggingface-hub`: `1.15.0`
- `transformers`: `5.8.1`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-05-17 00:16:46 BST_
