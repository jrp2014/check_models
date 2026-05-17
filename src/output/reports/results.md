# Model Performance Results

_Generated on 2026-05-17 21:19:22 BST_

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ 3 (top owners: mlx=2, mlx-lm=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=8, clean outputs=5/52.
- _Useful now:_ 3 clean A/B model(s) worth first review.
- _Review watchlist:_ 49 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Quality signal frequency:_ cutoff=31, repetitive=17, degeneration=10,
  harness=8, token_cap=8, reasoning_leak=3.

### Runtime

- _Runtime pattern:_ post-prefill decode dominates measured phase time (62%;
  39/55 measured model(s)).
- _Phase totals:_ model load=122.79s, local prompt prep=0.17s, upstream
  prefill / first-token=93.70s, post-prefill decode=362.49s, cleanup=6.47s.
- _Generation total:_ 456.19s across 52 model(s); upstream prefill /
  first-token split available for 52/52 model(s).
- _What this likely means:_ Most measured runtime is spent generating after
  the first token is available.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=11, exception=3, max_tokens=41.
- _Validation overhead:_ 11.45s total (avg 0.21s across 55 model(s)).
- _Upstream prefill / first-token latency:_ Avg 1.80s | Min 0.03s | Max 14.49s
  across 52 model(s).

## 🏆 Performance Highlights

- **Fastest:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (426.4 tps)
- **💾 Most efficient:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (1.3 GB)
- **⚡ Fastest load:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (0.35s)
- **📊 Average TPS:** 75.3 across 52 models

## 📈 Resource Usage

- **Total peak memory:** 1010.2 GB
- **Average peak memory:** 19.4 GB
- **Memory efficiency:** 54 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** ✅ B: 7 | 🟡 C: 3 | 🟠 D: 23 | ❌ F: 19

**Average Utility Score:** 37/100

- **Best for cataloging:** `jqlive/Kimi-VL-A3B-Thinking-2506-6bit` (✅ B, 75/100)
- **Best descriptions:** `qnguyen3/nanoLLaVA` (90/100)
- **Best keywording:** `mlx-community/Qwen3.5-27B-4bit` (65/100)
- **Worst for cataloging:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (❌ F, 0/100)

### ⚠️ 42 Models with Low Utility (D/F)

- `HuggingFaceTB/SmolVLM-Instruct`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `LiquidAI/LFM2.5-VL-450M-MLX-bf16`: ❌ F (0/100) - Empty or minimal output
- `meta-llama/Llama-3.2-11B-Vision-Instruct`: ❌ F (0/100) - Output too short to be useful
- `microsoft/Phi-3.5-vision-instruct`: 🟠 D (36/100) - Keywords are not specific or diverse enough
- `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`: ❌ F (16/100) - Output lacks detail
- `mlx-community/FastVLM-0.5B-bf16`: ❌ F (19/100) - Output lacks detail
- `mlx-community/GLM-4.6V-Flash-6bit`: ❌ F (20/100) - Keywords are not specific or diverse enough
- `mlx-community/GLM-4.6V-Flash-mxfp4`: 🟠 D (50/100) - Lacks visual description of image
- `mlx-community/GLM-4.6V-nvfp4`: ❌ F (5/100) - Output too short to be useful
- `mlx-community/Idefics3-8B-Llama3-bf16`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/InternVL3-14B-8bit`: ❌ F (30/100) - Keywords are not specific or diverse enough
- `mlx-community/InternVL3-8B-bf16`: 🟠 D (50/100) - Keywords are not specific or diverse enough
- `mlx-community/LFM2-VL-1.6B-8bit`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`: 🟠 D (50/100) - Keywords are not specific or diverse enough
- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`: 🟠 D (50/100) - Keywords are not specific or diverse enough
- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`: 🟠 D (42/100) - Keywords are not specific or diverse enough
- `mlx-community/Molmo-7B-D-0924-bf16`: ❌ F (0/100) - Empty or minimal output
- `mlx-community/Phi-3.5-vision-instruct-bf16`: 🟠 D (36/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: 🟠 D (50/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen3.5-27B-mxfp8`: ❌ F (1/100) - Output too short to be useful
- `mlx-community/Qwen3.5-35B-A3B-4bit`: ❌ F (12/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen3.5-35B-A3B-6bit`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen3.5-35B-A3B-bf16`: ❌ F (20/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen3.5-9B-MLX-4bit`: ❌ F (5/100) - Output too short to be useful
- `mlx-community/Qwen3.6-27B-mxfp8`: 🟠 D (50/100) - Keywords are not specific or diverse enough
- `mlx-community/SmolVLM-Instruct-bf16`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/gemma-3-27b-it-qat-4bit`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/gemma-3-27b-it-qat-8bit`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/gemma-3n-E2B-4bit`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/gemma-3n-E4B-it-bf16`: 🟠 D (50/100) - Keywords are not specific or diverse enough
- `mlx-community/gemma-4-26b-a4b-it-4bit`: ❌ F (19/100) - Output lacks detail
- `mlx-community/gemma-4-31b-bf16`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/gemma-4-31b-it-4bit`: 🟠 D (44/100) - Keywords are not specific or diverse enough
- `mlx-community/llava-v1.6-mistral-7b-8bit`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (5/100) - Output too short to be useful
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (5/100) - Output too short to be useful
- `mlx-community/paligemma2-3b-pt-896-4bit`: ❌ F (16/100) - Output lacks detail
- `mlx-community/pixtral-12b-8bit`: 🟠 D (45/100) - Keywords are not specific or diverse enough

## ⚠️ Quality Issues

- **❌ Failed Models (3):**
  - `facebook/pe-av-large` (`Model Error`)
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit` (`Model Error`)
  - `mlx-community/LFM2.5-VL-1.6B-bf16` (`Weight Mismatch`)
- **🔄 Repetitive Output (17):**
  - `mlx-community/GLM-4.6V-Flash-6bit` (token: `.`)
  - `mlx-community/GLM-4.6V-Flash-mxfp4` (token: `phrase: "in order to get..."`)
  - `mlx-community/Idefics3-8B-Llama3-bf16` (token: `comm,`)
  - `mlx-community/InternVL3-8B-bf16` (token: `phrase: "theorem: theorem: theorem: the..."`)
  - `mlx-community/LFM2-VL-1.6B-8bit` (token: `and`)
  - `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` (token: `phrase: "2u 2u 2u 2u..."`)
  - `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` (token: `phrase: "更 更 更 更..."`)
  - `mlx-community/Molmo-7B-D-0924-8bit` (token: `phrase: "the image shows a..."`)
  - `mlx-community/MolmoPoint-8B-fp16` (token: `phrase: "there is a black..."`)
  - `mlx-community/Qwen2-VL-2B-Instruct-4bit` (token: `phrase: "a. a. a. a...."`)
  - `mlx-community/Qwen3.5-35B-A3B-4bit` (token: `phrase: "14:30:00 14:30:00 14:30:00 14:..."`)
  - `mlx-community/Qwen3.5-35B-A3B-6bit` (token: `only`)
  - `mlx-community/Qwen3.6-27B-mxfp8` (token: `phrase: "```python >>> print("hello, wo..."`)
  - `mlx-community/SmolVLM2-2.2B-Instruct-mlx` (token: `phrase: "saff saff saff saff..."`)
  - `mlx-community/gemma-3n-E4B-it-bf16` (token: `sup`)
  - `mlx-community/gemma-4-31b-it-4bit` (token: `phrase: "[cnai- [cnai- [cnai- [cnai-..."`)
  - `mlx-community/llava-v1.6-mistral-7b-8bit` (token: `phrase: "out of course. out..."`)
- **👻 Hallucinations (1):**
  - `mlx-community/Qwen3.5-27B-4bit`
- **📝 Formatting Issues (2):**
  - `mlx-community/Idefics3-8B-Llama3-bf16`
  - `mlx-community/Qwen3.5-27B-4bit`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 75.3 | Min: 4.33 | Max: 426
- **Peak Memory**: Avg: 19 | Min: 1.3 | Max: 71
- **Total Time**: Avg: 11.34s | Min: 0.87s | Max: 51.64s
- **Generation Time**: Avg: 8.77s | Min: 0.33s | Max: 47.98s
- **Model Load Time**: Avg: 2.35s | Min: 0.35s | Max: 11.32s

## ✅ Recommended Models

Quick picks based on end-to-end utility plus description and keyword strength.

- _Best end-to-end cataloging:_ [`qnguyen3/nanoLLaVA`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-qnguyen3-nanollava)
  (Utility B 75/100 | Description 90 | Keywords 0 | Speed 115 TPS | Memory 4.0
  | Caveat nontext prompt burden=73%)
- _Best descriptions:_ [`qnguyen3/nanoLLaVA`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-qnguyen3-nanollava)
  (Utility B 75/100 | Description 90 | Keywords 0 | Speed 115 TPS | Memory 4.0
  | Caveat nontext prompt burden=73%)
- _Best keywording:_ [`mlx-community/nanoLLaVA-1.5-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-nanollava-15-4bit)
  (Utility B 75/100 | Description 87 | Keywords 0 | Speed 307 TPS | Memory 1.7
  | Caveat nontext prompt burden=73%)
- _Fastest generation:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16)
  (Utility F 0/100 | Description 0 | Keywords 0 | Speed 426 TPS | Memory 1.3 |
  Caveat Output appears truncated to about 3 tokens.; nontext prompt
  burden=98%)
- _Lowest memory footprint:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16)
  (Utility F 0/100 | Description 0 | Keywords 0 | Speed 426 TPS | Memory 1.3 |
  Caveat Output appears truncated to about 3 tokens.; nontext prompt
  burden=98%)
- _Best balance:_ [`mlx-community/nanoLLaVA-1.5-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-nanollava-15-4bit)
  (Utility B 75/100 | Description 87 | Keywords 0 | Speed 307 TPS | Memory 1.7
  | Caveat nontext prompt burden=73%)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (3):_ [`facebook/pe-av-large`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-facebook-pe-av-large),
  [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-kimi-vl-a3b-thinking-8bit),
  [`mlx-community/LFM2.5-VL-1.6B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-lfm25-vl-16b-bf16).
  Example: `Model Error`.
- _🔄 Repetitive Output (17):_ [`mlx-community/GLM-4.6V-Flash-6bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/GLM-4.6V-Flash-mxfp4`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  [`mlx-community/InternVL3-8B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-internvl3-8b-bf16),
  +13 more. Example: token: `.`.
- _👻 Hallucinations (1):_ [`mlx-community/Qwen3.5-27B-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen35-27b-4bit).
- _📝 Formatting Issues (2):_ [`mlx-community/Idefics3-8B-Llama3-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  [`mlx-community/Qwen3.5-27B-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-qwen35-27b-4bit).
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

_Overall runtime:_ 600.62s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                     |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:-----------------------------------|----------------:|
| `facebook/pe-av-large`                                  |                   |                       |                |              |           |             |                  |      0.14s |       1.15s |                                    |          mlx-lm |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |                   |                       |                |              |           |             |                  |      0.26s |       1.27s |                                    |             mlx |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |                   |                       |                |              |           |             |                  |      0.27s |       1.29s |                                    |             mlx |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      |               269 |                     3 |            272 |         8460 |       426 |         1.3 |            0.33s |      0.35s |       0.87s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |                26 |                     9 |             35 |          292 |       286 |         2.1 |            0.45s |      0.66s |       1.32s |                                    |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |                22 |                    99 |            121 |          253 |       307 |         1.7 |            0.81s |      0.54s |       1.58s |                                    |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               269 |                   200 |            469 |         2781 |       314 |           3 |            1.04s |      0.52s |       1.76s | repetitive(and), cutoff            |                 |
| `qnguyen3/nanoLLaVA`                                    |                22 |                    71 |             93 |          226 |       115 |           4 |            1.09s |      0.55s |       1.84s |                                    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,031 |                     6 |          1,037 |         1294 |      39.8 |          11 |            1.26s |      1.66s |       3.15s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,103 |                     9 |          4,112 |         3670 |      62.2 |         4.6 |            1.59s |      1.25s |       3.06s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |               803 |                   200 |          1,003 |         2059 |       187 |         2.5 |            1.78s |      0.59s |       2.57s | repetitive(phrase: "a. a. a...     |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |             1,339 |                   200 |          1,539 |         2536 |       178 |         5.4 |            2.02s |      1.13s |       3.41s | token-cap                          |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |                97 |                   200 |            297 |          281 |       126 |         5.5 |            2.31s |      0.57s |       3.09s | repetitive(phrase: "saff sa...     |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,196 |                   200 |          1,396 |         2703 |       124 |         5.5 |            2.38s |      0.50s |       3.07s | cutoff                             |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,031 |                     6 |          1,037 |          897 |      6.42 |          26 |            2.41s |      2.51s |       5.14s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             1,196 |                   200 |          1,396 |         2662 |       118 |         5.5 |            2.50s |      0.60s |       3.31s | cutoff                             |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               284 |                   200 |            484 |          531 |       100 |          17 |            2.85s |      2.81s |       5.92s | degeneration, token-cap            |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |               593 |                   200 |            793 |         1075 |      96.9 |          22 |            2.94s |      2.66s |       5.82s | repetitive(phrase: "14:30:0...     |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |               593 |                   200 |            793 |         1155 |        91 |         7.2 |            3.08s |      1.64s |       4.95s | degeneration, token-cap            |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |               593 |                   200 |            793 |          936 |      76.5 |          30 |            3.60s |      3.35s |       7.16s | repetitive(only), cutoff           |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |               745 |                   200 |            945 |         1380 |      70.1 |          16 |            3.81s |      2.05s |       6.06s | reasoning-leak, cutoff             |                 |
| `mlx-community/InternVL3-14B-8bit`                      |               781 |                    74 |            855 |          774 |      28.9 |          17 |            3.89s |      1.86s |       5.98s |                                    |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               266 |                   200 |            466 |          666 |        61 |           6 |            4.04s |      1.75s |       6.04s | degeneration, cutoff               |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |               770 |                   200 |            970 |         2584 |      55.9 |         9.2 |            4.19s |      0.96s |       5.36s | cutoff                             |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |               803 |                   200 |          1,003 |         1352 |      60.9 |          10 |            4.20s |      1.21s |       5.61s | ⚠️harness(stop_token), ...         |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |               770 |                   200 |            970 |         2446 |      55.2 |         9.2 |            4.25s |      0.91s |       5.37s | cutoff                             |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,031 |                    76 |          1,107 |         2645 |      19.3 |          10 |            4.64s |      1.48s |       6.33s |                                    |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |               790 |                   200 |            990 |          990 |      53.3 |          10 |            4.86s |      1.60s |       6.67s | repetitive(.), degeneration, ...   |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |             1,340 |                   200 |          1,540 |         1044 |      61.5 |          11 |            4.88s |      1.41s |       6.51s | repetitive(phrase: "2u 2u 2...     |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |               790 |                   200 |            990 |         1141 |      70.9 |         7.9 |            4.95s |      1.48s |       6.64s | repetitive(phrase: "in orde...     |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |             1,340 |                   200 |          1,540 |          938 |        56 |          11 |            5.35s |      1.50s |       7.07s | repetitive(phrase: "更 更 更 更... |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               274 |                   200 |            474 |          687 |      39.5 |          17 |            5.83s |      2.74s |       8.81s | repetitive(sup), cutoff            |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             1,866 |                   200 |          2,066 |          732 |      59.2 |         9.7 |            6.24s |      0.96s |       7.41s | repetitive(phrase: "out of...      |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |               790 |                     8 |            798 |          125 |      42.5 |          63 |            6.82s |      8.82s |      15.86s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/InternVL3-8B-bf16`                       |               781 |                   200 |            981 |         1412 |      32.5 |          17 |            7.05s |      1.79s |       9.09s | repetitive(phrase: "theorem...     |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             2,440 |                   200 |          2,640 |         1011 |      39.2 |          14 |            7.83s |      1.64s |       9.68s | token-cap                          |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,327 |                   200 |          2,527 |         1687 |      30.8 |          19 |            8.23s |      1.94s |      10.38s | repetitive(comm,), cutoff, ...     |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               275 |                   200 |            475 |          296 |      27.7 |          19 |            8.48s |      2.55s |      11.25s | cutoff                             |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |               807 |                   200 |          1,007 |          543 |        29 |          20 |            8.70s |      2.11s |      11.03s | ⚠️harness(encoding), ...           |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |               593 |                   200 |            793 |          466 |        28 |          18 |            8.74s |      2.25s |      11.21s | hallucination, reasoning-leak, ... |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               284 |                   200 |            484 |          287 |      24.1 |          19 |            9.61s |      2.85s |      12.66s | repetitive(phrase: "[cnai-...      |                 |
| `mlx-community/pixtral-12b-8bit`                        |             2,349 |                   200 |          2,549 |         1424 |      38.9 |          16 |            9.96s |      1.75s |      11.92s | token-cap                          |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |               825 |                   200 |          1,025 |          144 |      48.5 |          60 |           10.29s |      9.29s |      19.96s | cutoff                             |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |                16 |                   200 |            216 |         10.7 |        20 |          15 |           11.83s |      1.62s |      13.65s | degeneration, token-cap            |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |               593 |                   200 |            793 |         70.8 |      55.6 |          71 |           12.36s |     11.32s |      23.90s | cutoff                             |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |               593 |                   200 |            793 |          421 |      17.4 |          30 |           13.22s |      3.27s |      16.73s | repetitive(phrase: "```pyth...     |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |               593 |                   200 |            793 |          394 |      16.7 |          30 |           13.81s |      3.29s |      17.35s | degeneration, cutoff               |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               275 |                   200 |            475 |          245 |      14.9 |          33 |           14.96s |      3.77s |      18.95s | degeneration, cutoff               |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |             1,201 |                     3 |          1,204 |         82.9 |        19 |          48 |           15.05s |      1.87s |      17.15s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/pixtral-12b-bf16`                        |             2,349 |                   200 |          2,549 |         1665 |      20.2 |          27 |           18.59s |      2.55s |      21.34s | cutoff                             |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |             1,201 |                   200 |          1,401 |         87.7 |      44.4 |          41 |           18.59s |      1.34s |      20.17s | repetitive(phrase: "the ima...     |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               272 |                   200 |            472 |         55.9 |      7.52 |          64 |           31.82s |      8.40s |      40.43s | cutoff                             |                 |
| `mlx-community/MolmoPoint-8B-fp16`                      |             1,203 |                   200 |          1,403 |         1007 |      5.76 |          24 |           36.34s |      2.30s |      38.85s | repetitive(phrase: "there i...     |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |                17 |                   200 |            217 |         10.1 |      4.96 |          25 |           42.35s |      2.17s |      44.73s | degeneration, token-cap            |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |               745 |                   200 |            945 |          621 |      4.33 |          39 |           47.98s |      3.45s |      51.64s | reasoning-leak, cutoff             |                 |

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
- _Python Version:_ 3.13.13
- _Architecture:_ arm64
- _GPU/Chip:_ Apple M5 Max
- _GPU Cores:_ 40
- _Metal Support:_ Metal 4
- _RAM:_ 128.0 GB
- _CPU Cores (Physical):_ 18
- _CPU Cores (Logical):_ 18

## Library Versions

- `numpy`: `2.4.5`
- `mlx`: `0.32.0.dev20260517+7b7c1240`
- `mlx-vlm`: `0.5.0`
- `mlx-lm`: `0.31.3`
- `mlx-audio`: `0.4.3`
- `huggingface-hub`: `1.15.0`
- `transformers`: `5.8.1`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-05-17 21:19:22 BST_
