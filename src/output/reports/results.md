# Model Performance Results

Generated on: 2026-07-05 22:43:04 BST

## Run Contract

- Mode: triage
- Semantic rankings: ungrounded (ungrounded)
- Primary selection tasks: brief captions; structured title/description/keywords when descriptive image metadata is available

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ none.
- _Maintainer signals:_ harness-risk successes=8, mechanically clean
  outputs=44/61.
- _Useful now:_ 20 model(s) shortlisted for caption review.
- _Review watchlist:_ 34 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Quality signal frequency:_ harness=8, cutoff=6, reasoning_leak=4,
  text_sanity=3, generation_loop=2, long_context=1.

### Runtime

- _Runtime pattern:_ post-prefill decode dominates measured phase time (52%;
  19/61 measured model(s)).
- _Phase totals:_ model load=137.23s, local prompt prep=0.21s, upstream
  prefill / first-token=57.44s, post-prefill decode=218.07s, cleanup=6.81s.
- _Generation total:_ 275.50s across 61 model(s); upstream prefill /
  first-token split available for 61/61 model(s).
- _What this likely means:_ Most measured runtime is spent generating after
  the first token is available.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=54, max_tokens=7.
- _Validation overhead:_ 0.11s total (avg 0.00s across 61 model(s)).
- _Upstream prefill / first-token latency:_ Avg 0.94s | Min 0.02s | Max 6.38s
  across 61 model(s).

## 🏆 Performance Highlights

- **Fastest:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (532.9 tps)
- **💾 Most efficient:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (1.0 GB)
- **⚡ Fastest load:** `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (0.32s)
- **📊 Average TPS:** 93.7 across 61 models

## 📈 Resource Usage

- **Input image size:** 0.31 MP
- **Average peak delta from post-load:** 3.18 GB
- **Peak memory delta / MP:** 10606 MB/MP
- **Average peak memory:** 18.6 GB
- **Memory efficiency:** 41 tokens/GB

## ⚠️ Quality Issues

- **🔄 Repetitive Output (1):**
  - `mlx-community/gemma-3n-E2B-4bit` (token: `phrase: "have this image. have..."`)
- **📝 Formatting Issues (1):**
  - `mlx-community/Idefics3-8B-Llama3-bf16`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 93.7 | Min: 4.72 | Max: 533
- **Peak Memory**: Avg: 19 | Min: 1.0 | Max: 71
- **Total Time**: Avg: 6.78s | Min: 0.38s | Max: 46.98s
- **Generation Time**: Avg: 4.52s | Min: 0.06s | Max: 43.53s
- **Model Load Time**: Avg: 2.25s | Min: 0.32s | Max: 11.22s

## Caption Selection

Triage mode suppresses cataloging and keyword scores. Brief-caption
recommendations are ungrounded unless descriptive image metadata is present;
use `model_selection.md` for the caption shortlist and `model_gallery.md` for
full output evidence.

_Prompt used:_

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Describe this image briefly.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 420.84s

_Detailed machine-readable metrics remain in `results.tsv` and
`results.jsonl`; this Markdown table keeps the high-signal columns for human
review._

<!-- markdownlint-disable MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Gen TPS |   Peak (GB) | Finish Reason   |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                     |   Error Package |
|:--------------------------------------------------------|------------------:|----------------------:|---------------:|----------:|------------:|:----------------|-----------------:|-----------:|------------:|:-----------------------------------|----------------:|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      |                80 |                    10 |             90 |       533 |           1 | stop            |            0.06s |      0.32s |       0.38s |                                    |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |               269 |                    10 |            279 |       310 |           3 | stop            |            0.12s |      0.48s |       0.61s |                                    |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |               315 |                     3 |            318 |       149 |         5.2 | stop            |            0.15s |      0.65s |       0.81s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/Qwen3-VL-2B-Instruct-bf16`               |               315 |                     3 |            318 |       155 |         5.3 | stop            |            0.15s |      0.77s |       0.92s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |                26 |                    15 |             41 |       316 |         2.1 | stop            |            0.16s |      0.58s |       0.76s |                                    |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |               269 |                    13 |            282 |       194 |         4.1 | stop            |            0.17s |      0.54s |       0.71s |                                    |                 |
| `mlx-community/MiniCPM-V-4.6-8bit`                      |               228 |                    22 |            250 |       269 |           3 | stop            |            0.22s |      0.84s |       1.08s |                                    |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |               417 |                    25 |            442 |       309 |         2.5 | stop            |            0.25s |      0.49s |       0.76s |                                    |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |                22 |                    81 |            103 |       368 |         1.8 | stop            |            0.31s |      0.57s |       0.89s |                                    |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |               417 |                     3 |            420 |      88.4 |          10 | stop            |            0.34s |      1.22s |       1.57s | ⚠️harness(prompt_template)         |                 |
| `qnguyen3/nanoLLaVA`                                    |                22 |                    35 |             57 |       114 |         3.8 | stop            |            0.44s |      0.55s |       0.99s |                                    |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |             1,196 |                    13 |          1,209 |       124 |         5.5 | stop            |            0.54s |      0.58s |       1.12s |                                    |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |             1,196 |                    13 |          1,209 |       126 |         5.5 | stop            |            0.55s |      0.66s |       1.22s |                                    |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |               770 |                    19 |            789 |      60.1 |         9.2 | stop            |            0.60s |      0.88s |       1.48s |                                    |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |               770 |                    19 |            789 |      58.8 |         9.2 | stop            |            0.60s |      0.90s |       1.50s |                                    |                 |
| `mlx-community/gemma-4-26b-a4b-it-4bit`                 |               286 |                    24 |            310 |       116 |          17 | stop            |            0.63s |      2.35s |       2.99s |                                    |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |               317 |                    76 |            393 |       133 |         5.3 | stop            |            0.70s |      0.79s |       1.51s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |               973 |                    98 |          1,071 |       203 |         4.5 | stop            |            0.77s |      1.02s |       1.80s |                                    |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |               408 |                    42 |            450 |        93 |         7.7 | stop            |            0.77s |      1.33s |       2.11s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |               319 |                    30 |            349 |      97.8 |          31 | stop            |            1.13s |      3.07s |       4.20s |                                    |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |             4,103 |                     3 |          4,106 |      78.7 |         4.6 | stop            |            1.17s |      1.43s |       2.61s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Qwen3.5-27B-4bit`                        |               319 |                    20 |            339 |      34.5 |          19 | stop            |            1.21s |      2.13s |       3.36s |                                    |                 |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    |               319 |                    50 |            369 |       117 |          22 | stop            |            1.21s |      2.48s |       3.71s |                                    |                 |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit`          |               286 |                    21 |            307 |      27.1 |          29 | stop            |            1.28s |      3.33s |       4.64s |                                    |                 |
| `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`         |               286 |                    16 |            302 |      16.9 |          28 | stop            |            1.42s |      3.16s |       4.59s |                                    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |               974 |                    50 |          1,024 |      66.8 |          10 | stop            |            1.44s |      1.38s |       2.83s |                                    |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |               408 |                    69 |            477 |      63.9 |          10 | stop            |            1.46s |      1.40s |       2.88s |                                    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |               974 |                    60 |          1,034 |      70.3 |         9.8 | stop            |            1.52s |      1.36s |       2.89s |                                    |                 |
| `mlx-community/Qwen3.6-27B-mxfp8`                       |               319 |                     2 |            321 |      36.1 |          30 | stop            |            1.62s |      3.04s |       4.68s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/gemma-4-31b-it-4bit`                     |               286 |                    25 |            311 |      28.8 |          19 | stop            |            1.63s |      2.56s |       4.19s |                                    |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |             2,327 |                    23 |          2,350 |      33.6 |          19 | stop            |            1.78s |      1.88s |       3.68s | formatting                         |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |                97 |                   187 |            284 |       129 |         5.5 | stop            |            1.81s |      0.62s |       2.43s |                                    |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |               266 |                   200 |            466 |       121 |           6 | length          |            1.92s |      1.43s |       3.36s | repetitive(phrase: "have th...     |                 |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     |               319 |                   166 |            485 |       100 |         7.8 | stop            |            2.32s |      1.34s |       3.67s | gibberish(mixed_script_noise)      |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |                16 |                    25 |             41 |      22.6 |          15 | stop            |            2.49s |      1.48s |       3.97s |                                    |                 |
| `mlx-community/InternVL3-8B-bf16`                       |             3,341 |                    48 |          3,389 |      33.9 |          18 | stop            |            2.70s |      1.59s |       4.30s |                                    |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |             2,356 |                    55 |          2,411 |      63.6 |         9.7 | stop            |            2.81s |      0.88s |       3.70s |                                    |                 |
| `mlx-community/InternVL3-14B-8bit`                      |             3,341 |                    15 |          3,356 |      33.5 |          19 | stop            |            2.86s |      1.63s |       4.49s |                                    |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |               274 |                   124 |            398 |      48.5 |          17 | stop            |            2.92s |      2.21s |       5.15s |                                    |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |               399 |                   200 |            599 |      77.3 |          16 | length          |            3.03s |      1.92s |       4.96s | reasoning-leak, token-cap          |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |               399 |                   200 |            599 |      73.2 |          20 | length          |            3.17s |      2.11s |       5.29s | reasoning-leak, cutoff             |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |               441 |                    84 |            525 |      31.2 |          20 | stop            |            3.66s |      2.11s |       5.77s |                                    |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |               275 |                    91 |            366 |      31.1 |          18 | stop            |            3.67s |      2.23s |       5.92s |                                    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |             1,031 |                   101 |          1,132 |      34.7 |          11 | stop            |            3.79s |      1.63s |       5.44s |                                    |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |               275 |                    70 |            345 |      17.9 |          32 | stop            |            4.81s |      3.29s |       8.12s |                                    |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |             1,330 |                   174 |          1,504 |      44.3 |          14 | stop            |            4.88s |      1.57s |       6.47s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |               319 |                     3 |            322 |      77.1 |          71 | stop            |            5.32s |      9.76s |      15.09s | ⚠️harness(prompt_template)         |                 |
| `mlx-community/pixtral-12b-8bit`                        |             1,239 |                   200 |          1,439 |      40.1 |          15 | length          |            5.77s |      1.67s |       7.45s | cutoff                             |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |               745 |                    55 |            800 |        51 |          20 | stop            |            6.03s |      1.24s |       7.28s |                                    |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |             1,031 |                   119 |          1,150 |      19.6 |          10 | stop            |            6.48s |      1.43s |       7.92s |                                    |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |               745 |                    53 |            798 |      30.4 |          27 | stop            |            6.84s |      1.71s |       8.56s |                                    |                 |
| `mlx-community/Ornith-1.0-35B-bf16`                     |               319 |                    44 |            363 |      58.7 |          71 | stop            |            7.34s |     11.22s |      18.59s |                                    |                 |
| `mlx-community/GLM-4.6V-nvfp4`                          |               408 |                    71 |            479 |      51.2 |          63 | stop            |            7.78s |      8.33s |      16.13s |                                    |                 |
| `mlx-community/gemma-4-31b-bf16`                        |               274 |                    36 |            310 |      7.61 |          63 | stop            |            8.98s |      7.59s |      16.58s |                                    |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |               439 |                   200 |            639 |      64.5 |          60 | length          |            9.30s |      9.53s |      18.84s | generation_loop(degeneration), ... |                 |
| `mlx-community/Qwen3.5-27B-mxfp8`                       |               319 |                    85 |            404 |      19.5 |          30 | stop            |            9.92s |      3.09s |      13.03s |                                    |                 |
| `mlx-community/pixtral-12b-bf16`                        |             1,239 |                   200 |          1,439 |      19.7 |          27 | length          |           10.99s |      2.50s |      13.50s | cutoff                             |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |                17 |                    77 |             94 |      5.17 |          25 | stop            |           16.85s |      2.33s |      19.19s |                                    |                 |
| `mlx-community/MolmoPoint-8B-fp16`                      |               766 |                   163 |            929 |      5.93 |          23 | stop            |           28.43s |      2.17s |      30.62s |                                    |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |             1,031 |                   159 |          1,190 |       5.4 |          26 | stop            |           30.67s |      2.43s |      33.11s |                                    |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |               399 |                   200 |            599 |      4.72 |          39 | length          |           43.53s |      3.44s |      46.98s | reasoning-leak, cutoff             |                 |

<!-- markdownlint-enable MD033 MD034 MD037 MD049 -->

_Companion artifacts:_

- _Model-selection shortlist:_ [model_selection.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_selection.md)
- _Standalone output gallery:_ [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md)
- _Automated review digest:_ [review.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/review.md)
- _Canonical run log:_ [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log)

---

## System/Hardware Information

- _OS:_ Darwin 25.5.0
- _macOS Version:_ 26.5.2
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
  sha256=7e5c9a3a3225bf3b04a5fe67c50602975d3698a45e2113433465848af47fd70c)
- _MLX libmlx.dylib:_ /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib
  (21,747,136 bytes,
  sha256=7f99f8ba35fabb3f26e8a3016e5bd6410c975b68147be239054e5fa74232541c)

## Library Versions

- `numpy`: `2.5.1`
- `mlx`: `0.32.0.dev20260705+de7b4ed9`
- `mlx-vlm`: `0.6.4`
- `mlx-lm`: `0.31.3`
- `mlx-audio`: `0.4.4`
- `huggingface-hub`: `1.22.0`
- `transformers`: `5.12.1`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.3.0`

Report generated on: 2026-07-05 22:43:04 BST
