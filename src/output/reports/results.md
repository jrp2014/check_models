# Model Performance Results

_Generated on 2026-06-28 22:11:08 BST_

## Run Contract

- Mode: triage
- Semantic rankings: ungrounded (ungrounded)
- Primary selection tasks: brief captions; structured title/description/keywords when descriptive image metadata is available

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ 2 (top owners: mlx=1, mlx-vlm=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=5, mechanically clean
  outputs=45/58.
- _Useful now:_ 29 model(s) shortlisted for caption review.
- _Review watchlist:_ 23 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Quality signal frequency:_ cutoff=6, harness=5, reasoning_leak=4,
  text_sanity=3, generation_loop=3, lang_mixing=2.

### Runtime

- _Runtime pattern:_ model load dominates measured phase time (47%; 41/60
  measured model(s)).
- _Phase totals:_ model load=225.60s, local prompt prep=0.19s, upstream
  prefill / first-token=62.74s, post-prefill decode=181.15s, cleanup=6.57s.
- _Generation total:_ 243.89s across 58 model(s); upstream prefill /
  first-token split available for 58/58 model(s).
- _What this likely means:_ Cold model load time is a major share of runtime
  for this cohort.
- _Suggested next action:_ Consider staged runs, model reuse, or narrowing the
  model set before reruns.
- _Termination reasons:_ completed=50, exception=2, max_tokens=8.
- _Validation overhead:_ 0.18s total (avg 0.00s across 60 model(s)).
- _Upstream prefill / first-token latency:_ Avg 1.08s | Min 0.02s | Max 7.26s
  across 58 model(s).

## 🏆 Performance Highlights

- **Fastest:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (547.4 tps)
- **💾 Most efficient:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (1.0 GB)
- **⚡ Fastest load:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (0.41s)
- **📊 Average TPS:** 92.4 across 58 models

## 📈 Resource Usage

- **Input image size:** 0.31 MP
- **Average peak delta from post-load:** 3.07 GB
- **Peak memory delta / MP:** 10239 MB/MP
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

- **Generation Tps**: Avg: 92.4 | Min: 4.64 | Max: 547
- **Peak Memory**: Avg: 18 | Min: 1.0 | Max: 71
- **Total Time**: Avg: 8.09s | Min: 0.49s | Max: 49.48s
- **Generation Time**: Avg: 4.20s | Min: 0.05s | Max: 44.37s
- **Model Load Time**: Avg: 3.87s | Min: 0.41s | Max: 46.81s

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

_Overall runtime:_ 477.90s

_Detailed machine-readable metrics remain in `results.tsv` and
`results.jsonl`; this Markdown table keeps the high-signal columns for human
review._

<!-- markdownlint-disable MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Gen TPS |   Peak (GB) | Finish Reason   |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                     |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|----------:|------------:|:----------------|-----------------:|-----------:|------------:|:-----------------------------------|----------------:|
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |                   |                       |                |           |             |                 |                  |      0.41s |       0.41s |                                    |             mlx |
| `mlx-community/MolmoPoint-8B-fp16`                      |                   |                       |                |           |             |                 |                  |      0.55s |       0.56s |                                    |         mlx-vlm |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      |                80 |                    10 |             90 |       547 |           1 | stop            |            0.05s |      0.41s |       0.49s |                                    |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               269 |                    10 |            279 |       322 |           3 | stop            |            0.12s |      2.83s |       2.96s |                                    |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |                26 |                    15 |             41 |       329 |         2.2 | stop            |            0.16s |      0.71s |       0.87s |                                    |                 |
| `mlx-community/MiniCPM-V-4.6-8bit`                      |               228 |                    22 |            250 |       277 |           3 | stop            |            0.22s |      0.88s |       1.11s |                                    |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |                22 |                    81 |            103 |       332 |         1.8 | stop            |            0.33s |      0.52s |       0.85s |                                    |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |               417 |                    52 |            469 |       313 |         2.5 | stop            |            0.34s |      0.79s |       1.13s |                                    |                 |
| `qnguyen3/nanoLLaVA`                                    |                22 |                    35 |             57 |       114 |           4 | stop            |            0.44s |      0.78s |       1.23s |                                    |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             1,196 |                    13 |          1,209 |       128 |         5.5 | stop            |            0.53s |      0.76s |       1.30s |                                    |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,196 |                    13 |          1,209 |       131 |         5.5 | stop            |            0.53s |      2.20s |       2.74s |                                    |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |               770 |                    19 |            789 |      60.1 |         9.2 | stop            |            0.59s |      1.20s |       1.79s |                                    |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               286 |                    24 |            310 |       115 |          17 | stop            |            0.62s |      3.18s |       3.82s |                                    |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |               408 |                    42 |            450 |      93.1 |         7.7 | stop            |            0.76s |      1.29s |       2.06s |                                    |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |               949 |                    99 |          1,048 |       198 |         4.4 | stop            |            0.77s |      2.54s |       3.33s |                                    |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |               315 |                    96 |            411 |       131 |         5.2 | stop            |            0.85s |      0.73s |       1.59s |                                    |                 |
| `mlx-community/Qwen3-VL-2B-Instruct-bf16`               |               315 |                    96 |            411 |       129 |         5.3 | stop            |            0.87s |      0.74s |       1.62s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |               319 |                    63 |            382 |       110 |          21 | stop            |            0.98s |      3.65s |       4.65s |                                    |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |               319 |                    71 |            390 |      99.4 |           7 | stop            |            1.01s |     15.94s |      16.98s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |               319 |                    58 |            377 |      93.7 |          30 | stop            |            1.13s |      3.97s |       5.10s |                                    |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,103 |                     3 |          4,106 |        77 |         4.6 | stop            |            1.16s |      2.30s |       3.47s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`         |               286 |                    22 |            308 |      28.5 |          28 | stop            |            1.25s |      5.02s |       6.28s |                                    |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |               417 |                    65 |            482 |      65.7 |          10 | stop            |            1.29s |      1.22s |       2.51s |                                    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |               950 |                    52 |          1,002 |      65.6 |          10 | stop            |            1.44s |      1.51s |       2.96s |                                    |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |               408 |                    69 |            477 |      63.7 |          10 | stop            |            1.45s |      1.49s |       2.96s |                                    |                 |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit`          |               286 |                    19 |            305 |      17.9 |          29 | stop            |            1.57s |      3.34s |       4.94s |                                    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |               950 |                    68 |          1,018 |      69.4 |         9.7 | stop            |            1.64s |      1.53s |       3.18s |                                    |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               286 |                    25 |            311 |      28.5 |          19 | stop            |            1.64s |      3.56s |       5.21s |                                    |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |               317 |                   196 |            513 |       128 |         5.3 | stop            |            1.65s |      0.97s |       2.63s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,327 |                    23 |          2,350 |      34.1 |          19 | stop            |            1.76s |     13.73s |      15.51s | formatting                         |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |                97 |                   187 |            284 |       129 |         5.5 | stop            |            1.81s |      0.58s |       2.40s |                                    |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               266 |                   200 |            466 |       121 |           6 | length          |            1.91s |      2.25s |       4.17s | repetitive(phrase: "have th...     |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |                16 |                    25 |             41 |      22.4 |          15 | stop            |            2.52s |      1.55s |       4.07s |                                    |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             3,341 |                    48 |          3,389 |        35 |          18 | stop            |            2.66s |     46.81s |      49.48s |                                    |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             3,341 |                    15 |          3,356 |      33.2 |          19 | stop            |            2.85s |      1.77s |       4.63s |                                    |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |               319 |                    71 |            390 |      32.2 |          19 | stop            |            2.88s |      3.05s |       5.96s |                                    |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |               319 |                    40 |            359 |      18.7 |          30 | stop            |            2.90s |      3.33s |       6.25s |                                    |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             2,356 |                    55 |          2,411 |      60.7 |         9.7 | stop            |            2.91s |      0.89s |       3.81s |                                    |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               274 |                   124 |            398 |      47.8 |          17 | stop            |            2.94s |      3.32s |       6.27s |                                    |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |               399 |                   200 |            599 |      79.1 |          16 | length          |            2.97s |      2.69s |       5.67s | reasoning-leak, token-cap          |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |               399 |                   200 |            599 |      72.3 |          20 | length          |            3.19s |      2.43s |       5.63s | reasoning-leak, cutoff             |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |               319 |                    49 |            368 |      19.2 |          30 | stop            |            3.25s |      3.13s |       6.40s |                                    |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |               417 |                    90 |            507 |        33 |          19 | stop            |            3.40s |      3.20s |       6.61s | ⚠️harness(encoding)                |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               275 |                    91 |            366 |      31.5 |          19 | stop            |            3.62s |      2.93s |       6.56s |                                    |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |               770 |                   200 |            970 |      58.2 |         9.2 | length          |            3.71s |      1.36s |       5.09s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,031 |                   101 |          1,132 |      34.3 |          11 | stop            |            3.81s |      1.62s |       5.45s |                                    |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             1,330 |                   174 |          1,504 |      44.2 |          14 | stop            |            4.87s |      1.88s |       6.76s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               275 |                    70 |            345 |      17.5 |          33 | stop            |            4.90s |      3.92s |       8.82s |                                    |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |               745 |                    55 |            800 |      51.4 |          20 | stop            |            5.31s |      1.32s |       6.64s |                                    |                 |
| `mlx-community/pixtral-12b-8bit`                        |             1,239 |                   200 |          1,439 |      39.9 |          15 | length          |            5.79s |      1.70s |       7.50s | cutoff                             |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,031 |                   119 |          1,150 |      19.4 |          10 | stop            |            6.53s |      4.14s |      10.68s |                                    |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |               745 |                    55 |            800 |        30 |          27 | stop            |            6.77s |      3.02s |       9.79s |                                    |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |               408 |                    71 |            479 |      51.2 |          63 | stop            |            7.21s |      8.74s |      15.97s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |               319 |                    55 |            374 |      67.8 |          71 | stop            |            8.10s |     11.70s |      19.82s |                                    |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |               439 |                   200 |            639 |        65 |          60 | length          |            9.07s |      9.43s |      18.51s | generation_loop(degeneration), ... |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               274 |                    36 |            310 |      7.53 |          64 | stop            |            9.57s |      8.41s |      17.99s |                                    |                 |
| `mlx-community/pixtral-12b-bf16`                        |             1,239 |                   200 |          1,439 |      20.4 |          27 | length          |           10.63s |      3.08s |      13.72s | cutoff                             |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |                17 |                    77 |             94 |      5.07 |          25 | stop            |           16.95s |      2.53s |      19.51s |                                    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,031 |                   159 |          1,190 |      5.28 |          26 | stop            |           31.33s |      2.68s |      34.03s |                                    |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |               399 |                   200 |            599 |      4.64 |          39 | length          |           44.37s |      3.38s |      47.76s | reasoning-leak, cutoff             |                 |

<!-- markdownlint-enable MD033 MD034 MD037 MD049 -->

_Companion artifacts:_

- _Model-selection shortlist:_ [model_selection.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_selection.md)
- _Standalone output gallery:_ [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md)
- _Automated review digest:_ [review.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/review.md)
- _Canonical run log:_ [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log)

---

## System/Hardware Information

- _OS:_ Darwin 25.5.0
- _macOS Version:_ 26.5.1
- _SDK Version:_ 26.5
- _Active Developer Directory:_ /Applications/Xcode.app/Contents/Developer
- _Xcode Version:_ 26.6
- _Xcode Build:_ 17F113
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
  (162,451,352 bytes,
  sha256=c2b26a5583dd2776ccab3b125611a6924bdc21c4f0880c222e3e8975933f962e)
- _MLX libmlx.dylib:_ /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib
  (21,746,656 bytes,
  sha256=e58e30c3625106bcb0e150263126a0de69e71cc5ae21c066901112cadac758d7)

## Library Versions

- `numpy`: `2.5.0`
- `mlx`: `0.32.0.dev20260628+e94b4150`
- `mlx-vlm`: `0.6.3`
- `mlx-lm`: `0.31.3`
- `mlx-audio`: `0.4.4`
- `huggingface-hub`: `1.21.0`
- `transformers`: `5.12.1`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-06-28 22:11:08 BST_
