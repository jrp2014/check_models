# Model Performance Results

_Generated on 2026-05-17 22:38:17 BST_

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ 3 (top owners: mlx=2, mlx-lm=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=13, mechanically clean
  outputs=5/52.
- _Useful now:_ 5 mechanically clean A/B model(s) with useful signals.
- _Review watchlist:_ 47 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Quality signal frequency:_ cutoff=30, generation_loop=28, repetitive=17,
  harness=13, degeneration=11, token_cap=6.

### Runtime

- _Runtime pattern:_ upstream prefill / first-token dominates measured phase
  time (53%; 12/55 measured model(s)).
- _Phase totals:_ model load=116.21s, local prompt prep=0.17s, upstream
  prefill / first-token=636.14s, post-prefill decode=438.28s, cleanup=6.01s.
- _Generation total:_ 1074.41s across 52 model(s); upstream prefill /
  first-token split available for 52/52 model(s).
- _What this likely means:_ Most measured runtime is spent inside upstream
  generation before the first token is available.
- _Suggested next action:_ Inspect prompt/image token accounting,
  dynamic-resolution image burden, prefill step sizing, and cache/prefill
  behavior.
- _Termination reasons:_ completed=13, exception=3, max_tokens=39.
- _Validation overhead:_ 11.45s total (avg 0.21s across 55 model(s)).
- _Upstream prefill / first-token latency:_ Avg 12.23s | Min 0.03s | Max
  76.87s across 52 model(s).

## 🏆 Performance Highlights

- **Fastest:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (573.6 tps)
- **💾 Most efficient:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (1.4 GB)
- **⚡ Fastest load:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (0.36s)
- **📊 Average TPS:** 85.0 across 52 models

## 📈 Resource Usage

- **Total peak memory:** 1087.1 GB
- **Average peak memory:** 20.9 GB
- **Memory efficiency:** 201 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** ✅ B: 8 | 🟡 C: 1 | 🟠 D: 22 | ❌ F: 21

**Average Utility Score:** 36/100

- **Best for cataloging:** `jqlive/Kimi-VL-A3B-Thinking-2506-6bit` (✅ B, 75/100)
- **Best descriptions:** `qnguyen3/nanoLLaVA` (90/100)
- **Best keywording:** `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` (31/100)
- **Worst for cataloging:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (❌ F, 0/100)

### ⚠️ 43 Models with Low Utility (D/F)

- `HuggingFaceTB/SmolVLM-Instruct`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `LiquidAI/LFM2.5-VL-450M-MLX-bf16`: ❌ F (0/100) - Empty or minimal output
- `meta-llama/Llama-3.2-11B-Vision-Instruct`: ❌ F (0/100) - Output too short to be useful
- `microsoft/Phi-3.5-vision-instruct`: ❌ F (30/100) - Keywords are not specific or diverse enough
- `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`: ❌ F (16/100) - Output lacks detail
- `mlx-community/GLM-4.6V-Flash-6bit`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/GLM-4.6V-Flash-mxfp4`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/GLM-4.6V-nvfp4`: ❌ F (35/100) - Keywords are not specific or diverse enough
- `mlx-community/Idefics3-8B-Llama3-bf16`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/InternVL3-14B-8bit`: ❌ F (19/100) - Keywords are not specific or diverse enough
- `mlx-community/InternVL3-8B-bf16`: ❌ F (16/100) - Output lacks detail
- `mlx-community/LFM2-VL-1.6B-8bit`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`: 🟠 D (50/100) - Keywords are not specific or diverse enough
- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`: 🟠 D (44/100) - Keywords are not specific or diverse enough
- `mlx-community/Molmo-7B-D-0924-8bit`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/Molmo-7B-D-0924-bf16`: ❌ F (0/100) - Empty or minimal output
- `mlx-community/Phi-3.5-vision-instruct-bf16`: ❌ F (30/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen3.5-27B-4bit`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/Qwen3.5-27B-mxfp8`: ❌ F (17/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen3.5-35B-A3B-4bit`: ❌ F (25/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen3.5-35B-A3B-6bit`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen3.5-35B-A3B-bf16`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/Qwen3.5-9B-MLX-4bit`: ❌ F (6/100) - Output too short to be useful
- `mlx-community/Qwen3.6-27B-mxfp8`: ❌ F (0/100) - Output too short to be useful
- `mlx-community/SmolVLM-Instruct-bf16`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/X-Reasoner-7B-8bit`: 🟠 D (42/100) - Keywords are not specific or diverse enough
- `mlx-community/gemma-3-27b-it-qat-4bit`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/gemma-3-27b-it-qat-8bit`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/gemma-3n-E2B-4bit`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/gemma-3n-E4B-it-bf16`: ❌ F (6/100) - Output too short to be useful
- `mlx-community/gemma-4-26b-a4b-it-4bit`: ❌ F (17/100) - Output lacks detail
- `mlx-community/gemma-4-31b-bf16`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/gemma-4-31b-it-4bit`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/llava-v1.6-mistral-7b-8bit`: 🟠 D (45/100) - Keywords are not specific or diverse enough
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (5/100) - Output too short to be useful
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (5/100) - Output too short to be useful
- `mlx-community/paligemma2-3b-pt-896-4bit`: ❌ F (16/100) - Output lacks detail
- `mlx-community/pixtral-12b-8bit`: 🟠 D (49/100) - Keywords are not specific or diverse enough
- `mlx-community/pixtral-12b-bf16`: 🟠 D (45/100) - Keywords are not specific or diverse enough

## ⚠️ Quality Issues

- **❌ Failed Models (3):**
  - `facebook/pe-av-large` (`Model Error`)
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit` (`Model Error`)
  - `mlx-community/LFM2.5-VL-1.6B-bf16` (`Weight Mismatch`)
- **🔄 Repetitive Output (17):**
  - `HuggingFaceTB/SmolVLM-Instruct` (token: `phrase: "ter budget ter budget..."`)
  - `mlx-community/GLM-4.6V-Flash-6bit` (token: `phrase: "1.0, 1.0, 1.0, 1.0,..."`)
  - `mlx-community/GLM-4.6V-Flash-mxfp4` (token: `phrase: "and the idea is..."`)
  - `mlx-community/Idefics3-8B-Llama3-bf16` (token: `comm,`)
  - `mlx-community/InternVL3-14B-8bit` (token: `phrase: "2018 2018 2018 2018..."`)
  - `mlx-community/LFM2-VL-1.6B-8bit` (token: `and`)
  - `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` (token: `phrase: "1st 1st 1st 1st..."`)
  - `mlx-community/Ministral-3-3B-Instruct-2512-4bit` (token: `phrase: ". . . ...."`)
  - `mlx-community/Molmo-7B-D-0924-8bit` (token: `phrase: "the image shows a..."`)
  - `mlx-community/Qwen3.5-27B-mxfp8` (token: `2v,`)
  - `mlx-community/Qwen3.5-35B-A3B-6bit` (token: `,`)
  - `mlx-community/Qwen3.5-35B-A3B-bf16` (token: `all,`)
  - `mlx-community/SmolVLM-Instruct-bf16` (token: `phrase: "ter budget ter budget..."`)
  - `mlx-community/SmolVLM2-2.2B-Instruct-mlx` (token: `phrase: "saff saff saff saff..."`)
  - `mlx-community/gemma-3n-E2B-4bit` (token: `phrase: "rew rew rew rew..."`)
  - `mlx-community/llava-v1.6-mistral-7b-8bit` (token: `phrase: "out of course. out..."`)
  - `mlx-community/pixtral-12b-8bit` (token: `phrase: "the other the other..."`)
- **📝 Formatting Issues (2):**
  - `mlx-community/Idefics3-8B-Llama3-bf16`
  - `mlx-community/gemma-3-27b-it-qat-4bit`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 85.0 | Min: 4.69 | Max: 574
- **Peak Memory**: Avg: 21 | Min: 1.4 | Max: 79
- **Total Time**: Avg: 23.10s | Min: 0.85s | Max: 171.65s
- **Generation Time**: Avg: 20.66s | Min: 0.29s | Max: 168.29s
- **Model Load Time**: Avg: 2.22s | Min: 0.36s | Max: 10.32s

## ✅ Usable Diagnostic Candidates

Models that passed mechanical diagnostics and retained useful cataloguing or
metadata-alignment signals.

- _Best end-to-end cataloging:_ [`qnguyen3/nanoLLaVA`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-qnguyen3-nanollava)
  (Utility B 75/100 | Description 90 | Keywords 0 | Speed 112 TPS | Memory 4.0
  | Caveat nontext prompt burden=73%)
- _Best descriptions:_ [`qnguyen3/nanoLLaVA`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-qnguyen3-nanollava)
  (Utility B 75/100 | Description 90 | Keywords 0 | Speed 112 TPS | Memory 4.0
  | Caveat nontext prompt burden=73%)
- _Best keywording:_ [`mlx-community/nanoLLaVA-1.5-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-nanollava-15-4bit)
  (Utility B 75/100 | Description 87 | Keywords 0 | Speed 364 TPS | Memory 1.9
  | Caveat nontext prompt burden=73%)
- _Fastest generation:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16)
  (Utility F 0/100 | Description 0 | Keywords 0 | Speed 574 TPS | Memory 1.4 |
  Caveat Output appears truncated to about 3 tokens.; nontext prompt
  burden=98%)
- _Lowest memory footprint:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16)
  (Utility F 0/100 | Description 0 | Keywords 0 | Speed 574 TPS | Memory 1.4 |
  Caveat Output appears truncated to about 3 tokens.; nontext prompt
  burden=98%)
- _Best balance:_ [`mlx-community/nanoLLaVA-1.5-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-nanollava-15-4bit)
  (Utility B 75/100 | Description 87 | Keywords 0 | Speed 364 TPS | Memory 1.9
  | Caveat nontext prompt burden=73%)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (3):_ [`facebook/pe-av-large`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-facebook-pe-av-large),
  [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-kimi-vl-a3b-thinking-8bit),
  [`mlx-community/LFM2.5-VL-1.6B-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-lfm25-vl-16b-bf16).
  Example: `Model Error`.
- _🔄 Repetitive Output (17):_ [`HuggingFaceTB/SmolVLM-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`mlx-community/GLM-4.6V-Flash-6bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/GLM-4.6V-Flash-mxfp4`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  +13 more. Example: token: `phrase: "ter budget ter budget..."`.
- _📝 Formatting Issues (2):_ [`mlx-community/Idefics3-8B-Llama3-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  [`mlx-community/gemma-3-27b-it-qat-4bit`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-mlx-community-gemma-3-27b-it-qat-4bit).
- _Low-utility outputs (43):_ [`HuggingFaceTB/SmolVLM-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16),
  [`meta-llama/Llama-3.2-11B-Vision-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-meta-llama-llama-32-11b-vision-instruct),
  [`microsoft/Phi-3.5-vision-instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-microsoft-phi-35-vision-instruct),
  +39 more. Common weakness: Keywords are not specific or diverse enough.

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

_Overall runtime:_ 1211.76s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                     |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:-----------------------------------|----------------:|
| `facebook/pe-av-large`                                  |                   |                       |                |              |           |             |                  |      0.26s |       1.28s |                                    |          mlx-lm |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |                   |                       |                |              |           |             |                  |      0.23s |       1.22s |                                    |             mlx |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |                   |                       |                |              |           |             |                  |      0.26s |       1.26s |                                    |             mlx |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      |               269 |                     3 |            272 |         8674 |       574 |         1.4 |            0.29s |      0.36s |       0.85s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |                22 |                    99 |            121 |          270 |       364 |         1.9 |            0.71s |      0.53s |       1.45s |                                    |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               269 |                   200 |            469 |         2838 |       333 |           3 |            0.97s |      0.62s |       1.79s | repetitive(and), ...               |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |                26 |                    35 |             61 |          276 |       353 |         2.1 |            1.00s |      0.61s |       1.82s |                                    |                 |
| `qnguyen3/nanoLLaVA`                                    |                22 |                    71 |             93 |          185 |       112 |           4 |            1.12s |      0.56s |       1.89s |                                    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,031 |                     6 |          1,037 |         1314 |      37.6 |          11 |            1.30s |      1.68s |       3.19s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,103 |                     9 |          4,112 |         3729 |      64.5 |         4.6 |            1.62s |      1.15s |       2.99s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             2,317 |                    11 |          2,328 |         2096 |      36.9 |          18 |            1.81s |      1.74s |       3.76s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               266 |                   200 |            466 |         1046 |       119 |           6 |            2.27s |      1.56s |       4.05s | repetitive(phrase: "rew rew...     |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |                97 |                   200 |            297 |          288 |       133 |         5.5 |            2.33s |      0.59s |       3.15s | repetitive(phrase: "saff sa...     |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |             2,277 |                   200 |          2,477 |         2391 |       192 |         6.4 |            2.39s |      0.93s |       3.54s | repetitive(phrase: ". . . ....     |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,196 |                   200 |          1,396 |         2821 |       128 |         5.5 |            2.43s |      0.46s |       3.09s | repetitive(phrase: "ter bud...     |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,031 |                     6 |          1,037 |          905 |      6.44 |          26 |            2.45s |      2.44s |       5.13s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               284 |                   200 |            484 |          612 |       117 |          17 |            2.53s |      2.34s |       5.10s | generation_loop(numeric_loop), ... |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             1,196 |                   200 |          1,396 |         2781 |       122 |         5.5 |            2.58s |      0.65s |       3.44s | repetitive(phrase: "ter bud...     |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |               770 |                   200 |            970 |         2594 |      57.5 |         9.2 |            4.08s |      1.02s |       5.30s | token-cap                          |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |             1,033 |                   200 |          1,233 |         1206 |      72.3 |          18 |            4.09s |      2.01s |       6.31s | reasoning-leak, cutoff             |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |               770 |                   200 |            970 |         2673 |      57.3 |         9.2 |            4.09s |      0.90s |       5.21s | token-cap                          |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,031 |                    76 |          1,107 |         2697 |      19.7 |          10 |            4.62s |      1.45s |       6.28s |                                    |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               274 |                   200 |            474 |          826 |      48.7 |          17 |            4.76s |      2.26s |       7.24s | cutoff                             |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |             2,278 |                   200 |          2,478 |         1090 |      67.1 |          12 |            5.46s |      1.34s |       7.02s | repetitive(phrase: "1st 1st...     |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             1,866 |                   200 |          2,066 |          736 |      62.2 |         9.7 |            6.22s |      0.93s |       7.37s | repetitive(phrase: "out of...      |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               275 |                   200 |            475 |          332 |      31.1 |          19 |            7.59s |      2.33s |      10.13s | cutoff, formatting                 |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,327 |                   200 |          2,527 |         1829 |      31.4 |          19 |            8.19s |      2.05s |      10.49s | repetitive(comm,), ...             |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               284 |                   200 |            484 |          325 |      27.2 |          19 |            8.61s |      2.61s |      11.43s | generation_loop(degeneration), ... |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             2,317 |                   200 |          2,517 |         1064 |      31.9 |          18 |            8.88s |      1.79s |      10.88s | repetitive(phrase: "2018 20...     |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |             2,278 |                   200 |          2,478 |         1120 |      64.2 |          13 |            9.03s |      1.37s |      10.61s | generation_loop(numeric_loop), ... |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |             1,270 |                   200 |          1,470 |          239 |      54.2 |          60 |            9.57s |      8.78s |      18.71s | cutoff                             |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |             1,745 |                   200 |          1,945 |          586 |      31.5 |          21 |            9.72s |      2.11s |      12.06s | ⚠️harness(encoding), ...           |                 |
| `mlx-community/pixtral-12b-8bit`                        |             2,349 |                   200 |          2,549 |         1447 |      39.2 |          16 |           10.01s |      1.66s |      11.88s | repetitive(phrase: "the oth...     |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |                16 |                   200 |            216 |         11.8 |      21.8 |          15 |           10.87s |      1.61s |      12.69s | generation_loop(token_noise), ...  |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |             6,045 |                   200 |          6,245 |          801 |      64.3 |         8.4 |           11.01s |      1.35s |      12.58s | repetitive(phrase: "and the...     |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |             6,045 |                   200 |          6,245 |          826 |      54.4 |          11 |           11.32s |      1.41s |      12.95s | repetitive(phrase: "1.0, 1....     |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             2,440 |                   200 |          2,640 |         1170 |        43 |          15 |           12.40s |      1.67s |      14.28s | cutoff                             |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               275 |                   200 |            475 |          283 |      17.4 |          33 |           12.80s |      3.46s |      16.47s | cutoff                             |                 |
| `mlx-community/pixtral-12b-bf16`                        |             2,349 |                   200 |          2,549 |         1663 |      21.1 |          27 |           18.06s |      2.61s |      20.89s | cutoff                             |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |             1,201 |                     3 |          1,204 |         62.8 |      18.5 |          48 |           19.92s |      1.75s |      21.88s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/MolmoPoint-8B-fp16`                      |             2,818 |                   107 |          2,925 |         1208 |      5.97 |          26 |           20.86s |      2.25s |      23.32s |                                    |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |             1,201 |                   200 |          1,401 |         69.6 |      51.5 |          41 |           21.84s |      1.22s |      23.26s | repetitive(phrase: "the ima...     |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |            16,176 |                   200 |         16,376 |          877 |      57.3 |          13 |           22.58s |      1.19s |      23.98s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               272 |                   200 |            472 |         74.7 |      7.35 |          64 |           31.28s |      6.91s |      38.40s | cutoff                             |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |             6,045 |                    58 |          6,103 |          416 |      37.3 |          79 |           36.65s |      9.22s |      46.15s | degeneration                       |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |                17 |                   200 |            217 |         9.77 |      5.02 |          25 |           41.93s |      2.26s |      44.41s | generation_loop(token_noise), ...  |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |             1,033 |                   200 |          1,233 |          685 |      4.69 |          39 |           44.72s |      3.28s |      48.20s | reasoning-leak, token-cap          |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |            16,167 |                    12 |         16,179 |          301 |      96.2 |          11 |           54.47s |      1.45s |      56.13s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |            16,167 |                   200 |         16,367 |          311 |      88.1 |          35 |           54.83s |      3.15s |      58.19s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |            16,167 |                   200 |         16,367 |          294 |       106 |          26 |           57.57s |      2.54s |      60.33s | generation_loop(token_noise), ...  |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |            16,167 |                   200 |         16,367 |          283 |      64.7 |          76 |           60.86s |     10.32s |      71.39s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |            16,176 |                   200 |         16,376 |          263 |       198 |           5 |           63.18s |      0.58s |      63.98s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |            16,167 |                   200 |         16,367 |          214 |      30.3 |          26 |           82.93s |      2.18s |      85.33s | generation_loop(degeneration), ... |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |            16,167 |                   200 |         16,367 |          219 |      18.3 |          38 |           85.36s |      3.11s |      88.70s | generation_loop(degeneration), ... |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |            16,167 |                   200 |         16,367 |          210 |      18.2 |          40 |          168.29s |      3.14s |     171.65s | ⚠️harness(long_context), ...       |                 |

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

_Report generated on: 2026-05-17 22:38:17 BST_
