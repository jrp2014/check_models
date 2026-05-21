# Model Performance Results

_Generated on 2026-05-21 23:12:48 BST_

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
  time (54%; 12/55 measured model(s)).
- _Phase totals:_ model load=106.81s, local prompt prep=0.17s, upstream
  prefill / first-token=650.69s, post-prefill decode=431.64s, cleanup=5.57s.
- _Generation total:_ 1082.34s across 52 model(s); upstream prefill /
  first-token split available for 52/52 model(s).
- _What this likely means:_ Most measured runtime is spent inside upstream
  generation before the first token is available.
- _Suggested next action:_ Inspect prompt/image token accounting,
  dynamic-resolution image burden, prefill step sizing, and cache/prefill
  behavior.
- _Termination reasons:_ completed=13, exception=3, max_tokens=39.
- _Validation overhead:_ 11.23s total (avg 0.20s across 55 model(s)).
- _Upstream prefill / first-token latency:_ Avg 12.51s | Min 0.03s | Max
  81.21s across 52 model(s).

## 🏆 Performance Highlights

- **Fastest:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (494.7 tps)
- **💾 Most efficient:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (1.4 GB)
- **⚡ Fastest load:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (0.35s)
- **📊 Average TPS:** 83.9 across 52 models

## 📈 Resource Usage

- **Total peak memory:** 1087.2 GB
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

- **Generation Tps**: Avg: 83.9 | Min: 4.74 | Max: 495
- **Peak Memory**: Avg: 21 | Min: 1.4 | Max: 79
- **Total Time**: Avg: 23.07s | Min: 0.84s | Max: 175.14s
- **Generation Time**: Avg: 20.81s | Min: 0.30s | Max: 171.79s
- **Model Load Time**: Avg: 2.04s | Min: 0.35s | Max: 9.42s

## ✅ Usable Diagnostic Candidates

Models that passed mechanical diagnostics and retained useful cataloguing or
metadata-alignment signals.

- _Best end-to-end cataloging:_ [`qnguyen3/nanoLLaVA`](model_gallery.md#model-qnguyen3-nanollava)
  (Utility B 75/100 | Description 90 | Keywords 0 | Speed 115 TPS | Memory 4.0
  | Caveat nontext prompt burden=73%)
- _Best descriptions:_ [`qnguyen3/nanoLLaVA`](model_gallery.md#model-qnguyen3-nanollava)
  (Utility B 75/100 | Description 90 | Keywords 0 | Speed 115 TPS | Memory 4.0
  | Caveat nontext prompt burden=73%)
- _Best keywording:_ [`mlx-community/nanoLLaVA-1.5-4bit`](model_gallery.md#model-mlx-community-nanollava-15-4bit)
  (Utility B 75/100 | Description 87 | Keywords 0 | Speed 374 TPS | Memory 1.9
  | Caveat nontext prompt burden=73%)
- _Fastest generation:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16)
  (Utility F 0/100 | Description 0 | Keywords 0 | Speed 495 TPS | Memory 1.4 |
  Caveat Output appears truncated to about 3 tokens.; nontext prompt
  burden=98%)
- _Lowest memory footprint:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16)
  (Utility F 0/100 | Description 0 | Keywords 0 | Speed 495 TPS | Memory 1.4 |
  Caveat Output appears truncated to about 3 tokens.; nontext prompt
  burden=98%)
- _Best balance:_ [`mlx-community/nanoLLaVA-1.5-4bit`](model_gallery.md#model-mlx-community-nanollava-15-4bit)
  (Utility B 75/100 | Description 87 | Keywords 0 | Speed 374 TPS | Memory 1.9
  | Caveat nontext prompt burden=73%)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _❌ Failed Models (3):_ [`facebook/pe-av-large`](model_gallery.md#model-facebook-pe-av-large),
  [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](model_gallery.md#model-mlx-community-kimi-vl-a3b-thinking-8bit),
  [`mlx-community/LFM2.5-VL-1.6B-bf16`](model_gallery.md#model-mlx-community-lfm25-vl-16b-bf16).
  Example: `Model Error`.
- _🔄 Repetitive Output (17):_ [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`mlx-community/GLM-4.6V-Flash-6bit`](model_gallery.md#model-mlx-community-glm-46v-flash-6bit),
  [`mlx-community/GLM-4.6V-Flash-mxfp4`](model_gallery.md#model-mlx-community-glm-46v-flash-mxfp4),
  [`mlx-community/Idefics3-8B-Llama3-bf16`](model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  +13 more. Example: token: `phrase: "ter budget ter budget..."`.
- _📝 Formatting Issues (2):_ [`mlx-community/Idefics3-8B-Llama3-bf16`](model_gallery.md#model-mlx-community-idefics3-8b-llama3-bf16),
  [`mlx-community/gemma-3-27b-it-qat-4bit`](model_gallery.md#model-mlx-community-gemma-3-27b-it-qat-4bit).
- _Low-utility outputs (43):_ [`HuggingFaceTB/SmolVLM-Instruct`](model_gallery.md#model-huggingfacetb-smolvlm-instruct),
  [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](model_gallery.md#model-liquidai-lfm25-vl-450m-mlx-bf16),
  [`meta-llama/Llama-3.2-11B-Vision-Instruct`](model_gallery.md#model-meta-llama-llama-32-11b-vision-instruct),
  [`microsoft/Phi-3.5-vision-instruct`](model_gallery.md#model-microsoft-phi-35-vision-instruct),
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

_Overall runtime:_ 1209.68s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Cached Tokens |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                     |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|----------------:|-----------------:|-----------:|------------:|:-----------------------------------|----------------:|
| `facebook/pe-av-large`                                  |                   |                       |                |              |           |             |                 |                  |      0.19s |       1.20s |                                    |          mlx-lm |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |                   |                       |                |              |           |             |                 |                  |      0.16s |       1.20s |                                    |             mlx |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |                   |                       |                |              |           |             |                 |                  |      0.31s |       1.35s |                                    |             mlx |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      |               269 |                     3 |            272 |         7937 |       495 |         1.4 |               0 |            0.30s |      0.35s |       0.84s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |                22 |                    99 |            121 |          323 |       374 |         1.9 |               0 |            0.70s |      0.47s |       1.38s |                                    |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               269 |                   200 |            469 |         2842 |       333 |           3 |               0 |            0.96s |      0.47s |       1.63s | repetitive(and), ...               |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |                26 |                    35 |             61 |          168 |       352 |         2.2 |               0 |            1.00s |      0.59s |       1.80s |                                    |                 |
| `qnguyen3/nanoLLaVA`                                    |                22 |                    71 |             93 |          206 |       115 |           4 |               0 |            1.10s |      0.53s |       1.82s |                                    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,031 |                     6 |          1,037 |         1311 |      40.1 |          11 |               0 |            1.29s |      1.57s |       3.07s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,103 |                     9 |          4,112 |         3708 |        62 |         4.6 |               0 |            1.63s |      1.13s |       2.98s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             2,317 |                    11 |          2,328 |         2085 |      37.3 |          18 |               0 |            1.81s |      1.69s |       3.70s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               266 |                   200 |            466 |          957 |       121 |           6 |               0 |            2.25s |      1.40s |       3.86s | repetitive(phrase: "rew rew...     |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |                97 |                   200 |            297 |          291 |       130 |         5.5 |               0 |            2.33s |      0.64s |       3.18s | repetitive(phrase: "saff sa...     |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |             2,277 |                   200 |          2,477 |         2391 |       192 |         6.4 |               0 |            2.38s |      0.91s |       3.51s | repetitive(phrase: ". . . ....     |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             1,196 |                   200 |          1,396 |         2798 |       127 |         5.5 |               0 |            2.45s |      0.59s |       3.25s | repetitive(phrase: "ter bud...     |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,031 |                     6 |          1,037 |          908 |      6.43 |          26 |               0 |            2.46s |      2.46s |       5.14s | ⚠️harness(prompt_template)         |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,196 |                   200 |          1,396 |         2685 |       124 |         5.5 |               0 |            2.49s |      0.66s |       3.35s | repetitive(phrase: "ter bud...     |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               284 |                   200 |            484 |          619 |       118 |          17 |               0 |            2.50s |      2.72s |       5.45s | generation_loop(numeric_loop), ... |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |               770 |                   200 |            970 |         2618 |      59.3 |         9.2 |               0 |            3.97s |      0.89s |       5.07s | token-cap                          |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |             1,033 |                   200 |          1,233 |         1266 |      72.9 |          18 |               0 |            3.97s |      1.97s |       6.16s | reasoning-leak, cutoff             |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |               770 |                   200 |            970 |         2685 |      57.9 |         9.2 |               0 |            4.04s |      0.90s |       5.15s | token-cap                          |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,031 |                    76 |          1,107 |         2690 |      19.6 |          10 |               0 |            4.63s |      1.40s |       6.23s |                                    |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               274 |                   200 |            474 |          829 |      48.7 |          17 |               0 |            4.76s |      2.22s |       7.20s | cutoff                             |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |             2,278 |                   200 |          2,478 |         1061 |      67.7 |          12 |               0 |            5.49s |      1.33s |       7.04s | repetitive(phrase: "1st 1st...     |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |             1,270 |                   200 |          1,470 |          653 |      53.8 |          60 |               0 |            6.24s |      4.98s |      11.60s | cutoff                             |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             1,866 |                   200 |          2,066 |          683 |      62.1 |         9.7 |               0 |            6.42s |      0.92s |       7.55s | repetitive(phrase: "out of...      |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               275 |                   200 |            475 |          330 |      30.8 |          19 |               0 |            7.66s |      2.29s |      10.17s | cutoff, formatting                 |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,327 |                   200 |          2,527 |         1840 |      32.4 |          19 |               0 |            7.89s |      1.94s |      10.06s | repetitive(comm,), ...             |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               284 |                   200 |            484 |          331 |      26.5 |          19 |               0 |            8.76s |      2.91s |      11.88s | generation_loop(degeneration), ... |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             2,317 |                   200 |          2,517 |         1059 |      32.1 |          18 |               0 |            8.82s |      1.77s |      10.80s | repetitive(phrase: "2018 20...     |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |             2,278 |                   200 |          2,478 |         1122 |      63.6 |          13 |               0 |            9.08s |      1.35s |      10.63s | generation_loop(numeric_loop), ... |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |             1,745 |                   200 |          1,945 |          612 |        32 |          21 |               0 |            9.50s |      2.05s |      11.76s | ⚠️harness(encoding), ...           |                 |
| `mlx-community/pixtral-12b-8bit`                        |             2,349 |                   200 |          2,549 |         1446 |      39.1 |          16 |               0 |           10.04s |      1.64s |      11.88s | repetitive(phrase: "the oth...     |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |             6,045 |                   200 |          6,245 |          795 |      73.4 |         8.4 |               0 |           10.69s |      1.30s |      12.22s | repetitive(phrase: "and the...     |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |                16 |                   200 |            216 |         11.5 |      21.4 |          15 |               0 |           11.04s |      1.45s |      12.70s | generation_loop(token_noise), ...  |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |             6,045 |                   200 |          6,245 |          809 |      55.5 |          11 |               0 |           11.39s |      1.43s |      13.04s | repetitive(phrase: "1.0, 1....     |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             2,440 |                   200 |          2,640 |         1207 |      43.7 |          15 |               0 |           12.25s |      1.59s |      14.06s | cutoff                             |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               275 |                   200 |            475 |          284 |      17.7 |          33 |               0 |           12.63s |      3.37s |      16.21s | cutoff                             |                 |
| `mlx-community/pixtral-12b-bf16`                        |             2,349 |                   200 |          2,549 |         1667 |      19.9 |          27 |               0 |           18.91s |      2.54s |      21.65s | cutoff                             |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |             1,201 |                     3 |          1,204 |         65.9 |      20.4 |          48 |               0 |           18.99s |      1.81s |      21.01s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/MolmoPoint-8B-fp16`                      |             2,818 |                   107 |          2,925 |         1210 |      5.99 |          26 |               0 |           20.77s |      2.20s |      23.19s |                                    |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |             1,201 |                   200 |          1,401 |         70.6 |      52.6 |          41 |               0 |           21.50s |      1.28s |      22.99s | repetitive(phrase: "the ima...     |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |            16,176 |                   200 |         16,376 |          861 |      57.6 |          13 |               0 |           22.83s |      1.20s |      24.24s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |             6,045 |                    58 |          6,103 |          493 |      38.9 |          79 |               0 |           29.37s |      6.32s |      35.92s | degeneration                       |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               272 |                   200 |            472 |          110 |      7.24 |          64 |               0 |           30.50s |      6.24s |      36.95s | cutoff                             |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |                17 |                   200 |            217 |           10 |      5.05 |          25 |               0 |           41.62s |      2.15s |      43.98s | generation_loop(token_noise), ...  |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |             1,033 |                   200 |          1,233 |          780 |      4.74 |          39 |               0 |           44.06s |      3.25s |      47.53s | reasoning-leak, token-cap          |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |            16,167 |                    12 |         16,179 |          290 |      93.2 |          11 |               0 |           56.59s |      1.33s |      58.13s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |            16,167 |                   200 |         16,367 |          274 |       104 |          26 |               0 |           61.67s |      2.50s |      64.38s | generation_loop(token_noise), ...  |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |            16,167 |                   200 |         16,367 |          279 |      64.5 |          76 |               0 |           61.82s |      9.42s |      71.45s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |            16,176 |                   200 |         16,376 |          268 |       200 |           5 |               0 |           61.99s |      0.52s |      62.72s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |            16,167 |                   200 |         16,367 |          265 |      84.7 |          35 |               0 |           64.09s |      3.13s |      67.43s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |            16,167 |                   200 |         16,367 |          219 |      30.6 |          26 |               0 |           81.04s |      2.21s |      83.48s | generation_loop(degeneration), ... |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |            16,167 |                   200 |         16,367 |          206 |      18.3 |          38 |               0 |           89.88s |      3.06s |      93.16s | generation_loop(degeneration), ... |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |            16,167 |                   200 |         16,367 |          199 |      18.2 |          40 |               0 |          171.79s |      3.13s |     175.14s | ⚠️harness(long_context), ...       |                 |

<!-- markdownlint-enable MD013 MD033 MD034 MD037 MD049 -->

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](model_gallery.md)
- _Automated review digest:_ [review.md](review.md)
- _Canonical run log:_ [check_models.log](../check_models.log)

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

- `numpy`: `2.4.6`
- `mlx`: `0.32.0.dev20260521+5d1c0e4c`
- `mlx-vlm`: `0.5.0`
- `mlx-lm`: `0.31.3`
- `mlx-audio`: `0.4.3`
- `huggingface-hub`: `1.16.1`
- `transformers`: `5.9.0`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-05-21 23:12:48 BST_
