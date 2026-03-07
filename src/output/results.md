# Model Performance Results

_Generated on 2026-03-07 18:31:40 GMT_

## 🎯 Action Snapshot

- **Framework/runtime failures:** 2 (top owners: mlx-vlm=1, model-config=1).
- **Next action:** review `Failures by Package (Actionable)` and `diagnostics.md`.
- **Model output watchlist:** 19 model(s) graded D/F.
- **Next action:** prioritize A/B models and de-prioritize D/F outputs.
- **Vs existing metadata:** better=6, neutral=1, worse=39 (baseline A 82/100).

## 🏆 Performance Highlights

- **Fastest:** `prince-canuma/Florence-2-large-ft` (314.5 tps)
- **💾 Most efficient:** `mlx-community/FastVLM-0.5B-bf16` (2.2 GB)
- **⚡ Fastest load:** `mlx-community/LFM2-VL-1.6B-8bit` (1.15s)
- **📊 Average TPS:** 80.8 across 46 models

## 📈 Resource Usage

- **Total peak memory:** 790.5 GB
- **Average peak memory:** 17.2 GB
- **Memory efficiency:** 236 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 7 | ✅ B: 8 | 🟡 C: 12 | 🟠 D: 6 | ❌ F: 13

**Average Utility Score:** 51/100

**Existing Metadata Baseline:** 🏆 A (82/100)
**Vs Existing Metadata:** Avg Δ -32 | Better: 6, Neutral: 1, Worse: 39

- **Best for cataloging:** `mlx-community/Qwen3.5-35B-A3B-bf16` (🏆 A, 92/100)
- **Worst for cataloging:** `prince-canuma/Florence-2-large-ft` (❌ F, 0/100)

### ⚠️ 19 Models with Low Utility (D/F)

- `HuggingFaceTB/SmolVLM-Instruct`: ❌ F (1/100) - Output lacks detail
- `microsoft/Phi-3.5-vision-instruct`: 🟠 D (46/100) - Mostly echoes context without adding value
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (7/100) - Output too short to be useful
- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`: 🟠 D (41/100) - Lacks visual description of image
- `mlx-community/FastVLM-0.5B-bf16`: ❌ F (27/100) - Mostly echoes context without adding value
- `mlx-community/LFM2-VL-1.6B-8bit`: ❌ F (28/100) - Mostly echoes context without adding value
- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`: 🟠 D (48/100) - Mostly echoes context without adding value
- `mlx-community/Molmo-7B-D-0924-8bit`: ❌ F (14/100) - Output lacks detail
- `mlx-community/Molmo-7B-D-0924-bf16`: ❌ F (14/100) - Output lacks detail
- `mlx-community/Phi-3.5-vision-instruct-bf16`: 🟠 D (46/100) - Mostly echoes context without adding value
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (28/100) - Mostly echoes context without adding value
- `mlx-community/Qwen3-VL-2B-Thinking-bf16`: 🟠 D (50/100) - Lacks visual description of image
- `mlx-community/SmolVLM-Instruct-bf16`: ❌ F (1/100) - Output lacks detail
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: ❌ F (4/100) - Output lacks detail
- `mlx-community/gemma-3-27b-it-qat-8bit`: 🟠 D (45/100) - Mostly echoes context without adding value
- `mlx-community/gemma-3n-E2B-4bit`: ❌ F (25/100) - Lacks visual description of image
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (10/100) - Output lacks detail
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: ❌ F (7/100) - Output too short to be useful
- `prince-canuma/Florence-2-large-ft`: ❌ F (0/100) - Output too short to be useful

## ⚠️ Quality Issues

- **❌ Failed Models (2):**
  - `microsoft/Florence-2-large-ft` (`Model Error`)
  - `mlx-community/deepseek-vl2-8bit` (`Processor Error`)
- **🔄 Repetitive Output (10):**
  - `Qwen/Qwen3-VL-2B-Instruct` (token: `phrase: "tudor, tudor, tudor, tudor,..."`)
  - `meta-llama/Llama-3.2-11B-Vision-Instruct` (token: `phrase: "tradition, culture, community,..."`)
  - `microsoft/Phi-3.5-vision-instruct` (token: `phrase: "united kingdom, united kingdom..."`)
  - `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16` (token: `phrase: ""springtime greenery," "spring..."`)
  - `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit` (token: `phrase: "cemetery, graveyard, old, trad..."`)
  - `mlx-community/Phi-3.5-vision-instruct-bf16` (token: `phrase: "united kingdom, united kingdom..."`)
  - `mlx-community/Qwen3-VL-2B-Thinking-bf16` (token: `phrase: "mary's churchyard lenham engla..."`)
  - `mlx-community/X-Reasoner-7B-8bit` (token: `phrase: "ancient, ancient gravestones, ..."`)
  - `mlx-community/gemma-3n-E2B-4bit` (token: `16:23:54:`)
  - `mlx-community/paligemma2-3b-pt-896-4bit` (token: `phrase: "15-30 unique comma-separated t..."`)
- **📝 Formatting Issues (4):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-Flash-mxfp4`
  - `mlx-community/Idefics3-8B-Llama3-bf16`
  - `prince-canuma/Florence-2-large-ft`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 80.8 | Min: 3.41 | Max: 315
- **Peak Memory**: Avg: 17 | Min: 2.2 | Max: 74
- **Total Time**: Avg: 26.86s | Min: 2.31s | Max: 158.07s
- **Generation Time**: Avg: 23.23s | Min: 1.16s | Max: 150.10s
- **Model Load Time**: Avg: 3.63s | Min: 1.15s | Max: 14.80s

## 🚨 Failures by Package (Actionable)

<!-- markdownlint-disable MD060 -->

| Package | Failures | Error Types | Affected Models |
|---------|----------|-------------|-----------------|
| `mlx-vlm` | 1 | Model Error | `microsoft/Florence-2-large-ft` |
| `model-config` | 1 | Processor Error | `mlx-community/deepseek-vl2-8bit` |

<!-- markdownlint-enable MD060 -->

### Actionable Items by Package

#### mlx-vlm

- **microsoft/Florence-2-large-ft** (Model Error)
  - Error: `Model generation failed for microsoft/Florence-2-large-ft: Failed to process inputs with error: can only concatenate ...`
  - Type: `ValueError`

#### model-config

- **mlx-community/deepseek-vl2-8bit** (Processor Error)
  - Error: `Model preflight failed for mlx-community/deepseek-vl2-8bit: Loaded processor has no image_processor; expected multimo...`
  - Type: `ValueError`

**Prompt used:**

```text
Analyze this image for cataloguing metadata.

Return exactly these three sections, and nothing else:

Title: 6-12 words, descriptive and concrete.

Description: 1-2 factual sentences covering key subjects, setting, and action.

Keywords: 15-30 unique comma-separated terms, ordered most specific to most general.

Rules:
- Use only visually supported facts.
- If hints conflict with the image, trust the image.
- Do not output reasoning, notes, or extra sections.
- Do not copy context hints verbatim.

Context: Existing metadata hints (high confidence; use only if visually consistent):
- Description hint: , St Mary's Church, Lenham, England, United Kingdom, UK Here is a caption for the image, written in a neutral but descriptive style suitable for a periodical, blog, or magazine. **Caption:** A late winter afternoon in the historic village of Lenham, Kent, England. This view from the churchyard of St Mary's Church looks out over the village square. In the foreground, ancient, moss-covered headstones dot the landsca...
- Keyword hints: Christianity, Cypress Tree, England, Europe, Grass, Graveyard, Half-timbered House, Kent, Lenham, Parish church, Quaint, Spring, St Mary's Church, Tudor-style Houses, UK, United Kingdom, ancient, ancient gravestones, bare trees, blue
- Capture metadata: Taken on 2026-02-28 16:22:54 GMT (at 16:22:54 local time). GPS: 51.237300°N, 0.718917°E.
```

**Note:** Results sorted: errors first, then by generation time (fastest to slowest).

**Overall runtime:** 1247.82s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Token |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                     |   Error Package |
|:--------------------------------------------------------|--------:|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:-----------------------------------|----------------:|
| `microsoft/Florence-2-large-ft`                         |         |                   |                       |                |              |           |             |                  |            |             |                                    |         mlx-vlm |
| `mlx-community/deepseek-vl2-8bit`                       |         |                   |                       |                |              |           |             |                  |            |             |                                    |    model-config |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |       7 |               623 |                   130 |            753 |         2493 |       295 |         2.8 |            1.16s |      1.15s |       2.31s | description-sentences(3), ...      |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |       7 |               623 |                   155 |            778 |         2585 |       181 |           4 |            1.51s |      1.21s |       2.72s | description-sentences(3), ...      |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |       1 |             1,393 |                     5 |          1,398 |         1466 |      20.3 |          11 |            1.58s |      4.11s |       5.69s | ⚠️harness(prompt_template), ...    |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       |  151645 |               377 |                   190 |            567 |         2264 |       307 |         2.2 |            1.64s |      1.31s |       2.95s | fabrication, title-length(4), ...  |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |   49279 |             1,470 |                     9 |          1,479 |         1240 |       118 |         5.5 |            1.67s |      1.37s |       3.04s | ⚠️harness(prompt_template), ...    |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |   49154 |             1,570 |                    19 |          1,589 |         1278 |       113 |         5.5 |            1.80s |      1.60s |       3.40s | missing-sections(title+descrip...  |                 |
| `qnguyen3/nanoLLaVA`                                    |  151645 |               373 |                   110 |            483 |         1720 |      95.1 |         4.6 |            1.86s |      1.16s |       3.02s | missing-sections(keywords), ...    |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |   49154 |             1,570 |                    19 |          1,589 |         1139 |       115 |         5.5 |            1.94s |      1.73s |       3.67s | missing-sections(title+descrip...  |                 |
| `prince-canuma/Florence-2-large-ft`                     |       0 |               933 |                   500 |          1,433 |         3879 |       315 |         5.1 |            2.21s |      1.67s |       3.88s | ⚠️harness(stop_token), ...         |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |   11297 |               373 |                   500 |            873 |         1793 |       286 |         2.5 |            2.45s |      1.20s |       3.65s | keyword-count(126), ...            |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |       1 |             1,393 |                    14 |          1,407 |          479 |      28.3 |          12 |            3.79s |      4.01s |       7.80s | ⚠️harness(prompt_template), ...    |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |       2 |             2,964 |                   176 |          3,140 |         1016 |       160 |         7.8 |            4.38s |      1.82s |       6.19s |                                    |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       |  236787 |               629 |                   500 |          1,129 |          884 |       100 |         5.9 |            6.11s |      3.73s |       9.84s | repetitive(16:23:54:), ...         |                 |
| `mlx-community/InternVL3-8B-bf16`                       |  151645 |             2,156 |                   112 |          2,268 |          815 |      31.6 |          17 |            6.72s |      3.32s |      10.04s | title-length(5)                    |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |  159131 |             1,360 |                   500 |          1,860 |          744 |       113 |          18 |            6.82s |      4.36s |      11.18s | missing-sections(title+descrip...  |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |     276 |             1,360 |                   500 |          1,860 |          832 |       102 |          22 |            7.11s |      4.13s |      11.24s | title-length(70), ...              |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |     106 |               637 |                   273 |            910 |          809 |        39 |          17 |            8.18s |      5.37s |      13.56s | missing-sections(title+descrip...  |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |      11 |             1,360 |                   500 |          1,860 |          832 |      68.5 |          37 |            9.63s |      6.15s |      15.78s | missing-sections(title+descrip...  |                 |
| `mlx-community/InternVL3-14B-8bit`                      |  151645 |             2,156 |                   112 |          2,268 |          361 |        29 |          17 |           10.37s |      3.56s |      13.93s | title-length(5)                    |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |       2 |             2,965 |                   162 |          3,127 |          408 |        58 |          13 |           10.42s |      2.79s |      13.21s |                                    |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |     106 |               638 |                   139 |            777 |          142 |      23.9 |          19 |           10.72s |      5.42s |      16.15s |                                    |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |    3303 |             1,172 |                   500 |          1,672 |         1229 |      52.1 |         9.4 |           10.95s |      2.05s |      13.00s | repetitive(phrase: "united kin...  |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |       2 |             2,965 |                   187 |          3,152 |          403 |      54.7 |          13 |           11.14s |      2.72s |      13.86s |                                    |                 |
| `mlx-community/pixtral-12b-8bit`                        |       2 |             3,155 |                   130 |          3,285 |          453 |      33.4 |          16 |           11.24s |      3.45s |      14.69s | title-length(5), ...               |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |  128001 |             2,654 |                   228 |          2,882 |          743 |      29.4 |          18 |           11.73s |      3.78s |      15.50s | description-sentences(6), ...      |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |    3303 |             1,172 |                   500 |          1,672 |         1072 |      46.8 |         9.4 |           12.20s |      1.83s |      14.03s | repetitive(phrase: "united kin...  |                 |
| `mlx-community/pixtral-12b-bf16`                        |       2 |             3,155 |                   130 |          3,285 |          546 |      18.7 |          28 |           13.11s |      5.11s |      18.23s | title-length(5), ...               |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |     106 |               638 |                   132 |            770 |          170 |      14.5 |          34 |           13.24s |      7.59s |      20.83s |                                    |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |       2 |             2,566 |                    99 |          2,665 |          225 |      41.2 |          11 |           14.21s |      1.98s |      16.19s | missing-sections(title+descrip...  |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |       1 |             1,393 |                    49 |          1,442 |          375 |      4.54 |          26 |           14.90s |      5.76s |      20.67s | missing-sections(title+descrip...  |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |     376 |             1,676 |                   500 |          2,176 |          522 |      45.6 |          60 |           14.92s |      9.56s |      24.48s | repetitive(phrase: ""springtim...  |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |     108 |             4,465 |                   500 |          4,965 |         1618 |      41.5 |         4.3 |           15.34s |      3.28s |      18.62s | repetitive(phrase: "15-30 uniq...  |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |       2 |             2,455 |                   106 |          2,561 |          198 |      27.4 |          22 |           16.70s |      4.86s |      21.56s | ⚠️harness(encoding), ...           |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |    1058 |             3,246 |                   500 |          3,746 |          389 |      29.5 |          15 |           25.65s |      3.35s |      29.00s | missing-sections(title+descrip...  |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |     198 |             6,489 |                   500 |          6,989 |          313 |      62.2 |         8.4 |           29.14s |      2.57s |      31.71s | missing-sections(title+descrip...  |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |    9254 |             6,489 |                   500 |          6,989 |          291 |      46.6 |          11 |           33.44s |      2.71s |      36.15s | missing-sections(title+descrip...  |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |  151643 |             1,552 |                    15 |          1,567 |         46.2 |      37.3 |          41 |           34.81s |      2.33s |      37.14s | missing-sections(title+descrip...  |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |  151643 |             1,552 |                    15 |          1,567 |         45.4 |      25.6 |          48 |           35.55s |      3.34s |      38.89s | missing-sections(title+descrip...  |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |   90151 |            16,590 |                   500 |         17,090 |          372 |      74.7 |         8.3 |           51.78s |      1.68s |      53.45s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |     594 |            16,592 |                   500 |         17,092 |          333 |      73.7 |         8.3 |           57.15s |      1.93s |      59.08s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      |   27959 |            16,601 |                   500 |         17,101 |          364 |      42.6 |          13 |           57.83s |      2.59s |      60.42s | ⚠️harness(long_context), ...       |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |      11 |               343 |                   500 |            843 |         96.7 |      8.47 |          15 |           62.97s |      2.92s |      65.89s | repetitive(phrase: "cemetery, ...  |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |  151645 |            16,601 |                   126 |         16,727 |          201 |       178 |         5.1 |           83.87s |      1.48s |      85.36s | description-sentences(3), ...      |                 |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    |   15879 |            16,604 |                   500 |         17,104 |          209 |      54.4 |          74 |           89.20s |     14.80s |     104.00s | missing-sections(title+descrip...  |                 |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    |     303 |            16,604 |                   500 |         17,104 |          189 |      71.2 |          33 |           95.40s |      6.06s |     101.46s | degeneration, fabrication          |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |   34243 |               344 |                   500 |            844 |          106 |      3.41 |          25 |          150.10s |      7.97s |     158.07s | repetitive(phrase: "tradition, ... |                 |

<!-- markdownlint-enable MD013 MD033 MD034 MD037 MD049 -->

## Model Gallery

Full output from each model:


<!-- markdownlint-disable MD013 MD033 -->

### ❌ microsoft/Florence-2-large-ft

**Status:** Failed (Model Error)
**Error:**

> Model generation failed for microsoft/Florence-2-large-ft: Failed to process
> inputs with error: can only concatenate str (not "NoneType") to str
**Type:** `ValueError`
**Phase:** `decode`
**Code:** `MLX_VLM_DECODE_MODEL`
**Package:** `mlx-vlm`

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 873, in process_inputs_with_fallback
    return process_inputs(
        processor,
    ...<5 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 859, in process_inputs
    return process_method(**args)
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/models/florence2/processing_florence2.py", line 163, in __call__
    image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/image_processing_utils.py", line 50, in __call__
    return self.preprocess(images, *args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/image_processing_utils_fast.py", line 860, in preprocess
    self._validate_preprocess_kwargs(**kwargs)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/image_processing_utils_fast.py", line 823, in _validate_preprocess_kwargs
    validate_fast_preprocess_arguments(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        do_rescale=do_rescale,
        ^^^^^^^^^^^^^^^^^^^^^^
    ...<10 lines>...
        data_format=data_format,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/image_processing_utils_fast.py", line 106, in validate_fast_preprocess_arguments
    raise ValueError("Only returning PyTorch tensors is currently supported.")
ValueError: Only returning PyTorch tensors is currently supported.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 886, in process_inputs_with_fallback
    return process_inputs(
        processor,
    ...<5 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 859, in process_inputs
    return process_method(**args)
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/models/florence2/processing_florence2.py", line 185, in __call__
    self.image_token * self.num_image_tokens
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + self.tokenizer.bos_token
    ^~~~~~~~~~~~~~~~~~~~~~~~~~
TypeError: can only concatenate str (not "NoneType") to str

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 10842, in _run_model_generation
    output: GenerationResult | SupportsGenerationResult = generate(
                                                          ~~~~~~~~^
        model=model,
        ^^^^^^^^^^^^
    ...<13 lines>...
        **extra_kwargs,
        ^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 663, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 506, in stream_generate
    inputs = prepare_inputs(
        processor,
    ...<6 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1110, in prepare_inputs
    inputs = process_inputs_with_fallback(
        processor,
    ...<4 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 896, in process_inputs_with_fallback
    raise ValueError(
        f"Failed to process inputs with error: {fallback_error}"
    ) from fallback_error
ValueError: Failed to process inputs with error: can only concatenate str (not "NoneType") to str

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 10972, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
        phase_callback=_update_phase,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 10867, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(msg), "decode") from gen_known_err
ValueError: Model generation failed for microsoft/Florence-2-large-ft: Failed to process inputs with error: can only concatenate str (not "NoneType") to str
```

</details>

---

### ❌ mlx-community/deepseek-vl2-8bit

**Status:** Failed (Processor Error)
**Error:**

> Model preflight failed for mlx-community/deepseek-vl2-8bit: Loaded processor
> has no image\_processor; expected multimodal processor.
**Type:** `ValueError`
**Phase:** `processor_load`
**Code:** `MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR`
**Package:** `model-config`

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 10798, in _run_model_generation
    _run_model_preflight_validators(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        model_identifier=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
        phase_callback=phase_callback,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 10723, in _run_model_preflight_validators
    _raise_preflight_error(
    ~~~~~~~~~~~~~~~~~~~~~~^
        "Loaded processor has no image_processor; expected multimodal processor.",
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        phase="processor_load",
        ^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 10637, in _raise_preflight_error
    raise _tag_exception_failure_phase(ValueError(message), phase)
ValueError: Loaded processor has no image_processor; expected multimodal processor.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 10972, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
        phase_callback=_update_phase,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 10810, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(message), phase) from preflight_err
ValueError: Model preflight failed for mlx-community/deepseek-vl2-8bit: Loaded processor has no image_processor; expected multimodal processor.
```

</details>

---

### ✅ mlx-community/LFM2-VL-1.6B-8bit

**Metrics:** 295 TPS | 130 tokens

```text
Title: 6-12 words, descriptive and concrete.

Description: A late winter afternoon in the historic village of Lenham, Kent, England. This view from the churchyard of St Mary's Church looks out over the village square. In the foreground, ancient, moss-covered headstones dot the landsca...
Keywords: Christianity, Cypress Tree, England, Europe, Grass, Graveyard, Half-timbered House, Kent, Lenham, Parish church, Quaint, Spring, St Mary's Church, Tudor-style Houses, UK, United Kingdom, ancient, ancient gravestones, bare trees, blue
```

---

### ✅ mlx-community/LFM2.5-VL-1.6B-bf16

**Metrics:** 181 TPS | 155 tokens

```text
Title: St Mary's Church, Lenham, England, United Kingdom, UK

Description: The image depicts the historic churchyard of St Mary's Church in Lenham, Kent, England. The foreground is filled with ancient, moss-covered headstones, while the background features Tudor-style half-timbered houses and a cypress tree. The scene is set on a late winter afternoon, with a clear blue sky and scattered clouds.

Keywords: Christianity, Cypress Tree, England, Europe, Grass, Graveyard, Half-timbered House, Kent, Lenham, Parish church, Quaint, Spring, St Mary's Church, Tudor-style Houses, UK, United Kingdom, ancient, ancient gravestones, bare trees, blue
```

---

### ✅ mlx-community/paligemma2-3b-ft-docci-448-bf16

**Metrics:** 20.3 TPS | 5 tokens

```text
- Daytime.
```

---

### ✅ mlx-community/FastVLM-0.5B-bf16

**Metrics:** 307 TPS | 190 tokens

```text
Title: Medieval Village in England
Description: A late winter afternoon in the historic village of Lenham, Kent, England. This view from the churchyard of St Mary's Church looks out over the village square. In the foreground, ancient, moss-covered headstones dot the landsca...
- Keywords: Christianity, Cypress Tree, England, Europe, Grass, Graveyard, Half-timbered House, Kent, Lenham, Parish church, Quaint, Spring, St Mary's Church, Tudor-style Houses, UK, United Kingdom, ancient, ancient gravestones, bare trees, blue
- Metadata: Taken on 2026-02-28 16:22:54 GMT (at 16:22:54 local time). GPS: 51.237300°N, 0.718917°E.
```

---

### ✅ mlx-community/SmolVLM2-2.2B-Instruct-mlx

**Metrics:** 118 TPS | 9 tokens

```text
 Cemetery, England, United Kingdom, UK
```

---

### ✅ mlx-community/SmolVLM-Instruct-bf16

**Metrics:** 113 TPS | 19 tokens

```text
 A late winter afternoon in the historic village of lenham, kent, england.
```

---

### ✅ qnguyen3/nanoLLaVA

**Metrics:** 95.1 TPS | 110 tokens

```text
Title: 6-12 words, descriptive and concrete.
Description: A picturesque scene of a village in England, with a churchyard that is covered in moss. The church is old and has a stone structure, and there are several old, moss-covered headstones scattered around. The village square is full of life, with people going about their day. The sky above is a beautiful blue with white clouds, and the clouds are fluffy and light. The image captures the essence of a quiet, serene village at the end of the year.
```

---

### ✅ HuggingFaceTB/SmolVLM-Instruct

**Metrics:** 115 TPS | 19 tokens

```text
 A late winter afternoon in the historic village of lenham, kent, england.
```

---

### ✅ prince-canuma/Florence-2-large-ft

**Metrics:** 315 TPS | 500 tokens

```text
<s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s>
```

---

### ✅ mlx-community/nanoLLaVA-1.5-4bit

**Metrics:** 286 TPS | 500 tokens

```text
Title: 6-12 words, descriptive and concrete.

Description: 1-2 factual sentences covering key subjects, setting, and action.

Keywords: 15-30 unique comma-separated terms, ordered most specific to most general.

Rules:
- Use only visually supported facts.
- If hints conflict with the image, trust the image.
- Do not copy context hints verbatim.

Context: Existing metadata hints (high confidence; use only if visually consistent):
- Description hint: A late winter afternoon in the historic village of Lenham, Kent, England. This view from the churchyard of St Mary's Church looks out over the village square. In the foreground, ancient, moss-covered headstones dot the landsca...
- Keyword hints: Christianity, Cypress Tree, England, Europe, Grass, Graveyard, Half-timbered House, Kent, Lenham, Parish church, Quaint, Spring, St Mary's Church, Tudor-style Houses, UK, United Kingdom, England, Europe, grass, graveyards, half-timbered houses, Kent, Lenham, Parish church, Tudor-style Houses, UK, United Kingdom, England, Europe, grass, graveyards, half-timbered houses, Kent, Lenham, Parish church, Tudor-style Houses, UK, United Kingdom, England, Europe, grass, graveyards, half-timbered houses, Kent, Lenham, Parish church, Tudor-style Houses, UK, United Kingdom, England, Europe, grass, graveyards, half-timbered houses, Kent, Lenham, Parish church, Tudor-style Houses, UK, United Kingdom, England, Europe, grass, graveyards, half-timbered houses, Kent, Lenham, Parish church, Tudor-style Houses, UK, United Kingdom, England, Europe, grass, graveyards, half-timbered houses, Kent, Lenham, Parish church, Tudor-style Houses, UK, United Kingdom, England, Europe, grass, graveyards, half-timbered houses, Kent, Lenham, Parish church, Tudor-style Houses, UK, United Kingdom, England, Europe, grass, graveyards, half-timbered houses, Kent, Lenham, Parish church, Tudor-style Houses, UK, United Kingdom, England, Europe, grass, graveyards, half-timbered houses, Kent, Lenham, Parish church, Tudor-style
```

---

### ✅ mlx-community/paligemma2-10b-ft-docci-448-6bit

**Metrics:** 28.3 TPS | 14 tokens

```text
- Use only the hints that are visually supported by the image.
```

---

### ✅ mlx-community/Ministral-3-3B-Instruct-2512-4bit

**Metrics:** 160 TPS | 176 tokens

```text
Title: Historic Churchyard in Lenham, Kent – Mossy Graveyard with Tudor Villages

Description: A serene graveyard in Lenham, Kent, England, featuring ancient, weathered gravestones with patches of grass and wildflowers. The backdrop includes traditional Tudor-style brick houses with exposed timber framing, set against a partly cloudy sky.

Keywords: ancient gravestones, churchyard, Lenham, Kent, UK, St Mary’s Church, Tudor architecture, brick houses, half-timbered buildings, moss-covered stones, historic village, graveyard setting, England, European heritage, spring wildflowers, rural England, medieval cemetery, stone monuments, village square, late winter/early spring, traditional English village, parish church graveyard, blue skies, bare deciduous trees, quaint countryside, heritage preservation.
```

---

### ✅ mlx-community/gemma-3n-E2B-4bit

**Metrics:** 100 TPS | 500 tokens

```text
 12:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54: 16:23:54:
```

---

### ✅ mlx-community/InternVL3-8B-bf16

**Metrics:** 31.6 TPS | 112 tokens

```text
Title: Historic graveyard in Lenham, Kent

Description: The image shows an old graveyard with weathered tombstones, set against a backdrop of traditional Tudor-style houses. The scene is framed by a tall cypress tree and a partly cloudy sky.

Keywords: graveyard, tombstones, Lenham, Kent, England, Tudor-style, houses, traditional, cypress tree, weathered, moss, churchyard, village, historic, ancient, gravestones, grass, partly cloudy, sky, bare trees, Christianity, Europe, United Kingdom
```

---

### ✅ jqlive/Kimi-VL-A3B-Thinking-2506-6bit

**Metrics:** 113 TPS | 500 tokens

```text
◁think▷Got it, let's analyze the image for cataloguing metadata. First, the title should be descriptive and concrete. Looking at the image: it's a graveyard with ancient headstones, a tall cypress tree, half-timbered houses in the background, in Lenham, Kent, UK. So the title could be "Ancient graveyard with moss-covered headstones and half-timbered houses in Lenham, Kent, UK".

For the description: It's a graveyard scene in Lenham, Kent, UK, with ancient headstones, a cypress tree, and half-timbered houses in the background under a partly cloudy sky.

Keywords: Graveyard, ancient headstones, moss-covered, cypress tree, half-timbered houses, Lenham, Kent, UK, England, Europe, grass, graveyard, Christianity, UK, United Kingdom, ancient, spring (since daffodils might indicate spring), Tudor-style architecture (half-timbered houses are Tudor-style), bare trees (some trees without leaves), blue sky.

Wait, need to check the context hints. The context mentioned "Spring" and "bare trees" which matches the daffodils and some trees without leaves. Also, "Tudor-style Houses" is in the keywords. So include those. Let's list 15-30 unique terms, ordered from most specific to most general. Let's check the context's keywords: Christianity, Cypress Tree, England, Europe, Grass, Graveyard, Half-timbered House, Kent, Lenham, Parish church, Quaint, Spring, St Mary's Church, Tudor-style Houses, UK, United Kingdom, ancient, ancient gravestones, bare trees, blue. But the image might not have St Mary's Church visible? Wait, the context caption mentions St Mary's Church, but the image might not show it. Wait, the image shows a graveyard, so maybe "Parish church" is implied. Let's focus on what's visible. The image has a graveyard (so Graveyard), ancient headstones, cypress tree, half-timbered houses, Lenham, Kent, UK, grass, daffodils (spring), bare trees, blue sky, Christianity (since it's a graveyard), Tudor-style houses, ancient. Let's list them in order of specificity: Graveyard, ancient headstones, moss-covered (but maybe not in context, but the image has moss), cypress tree, half-timber
```

---

### ✅ mlx-community/Kimi-VL-A3B-Thinking-8bit

**Metrics:** 102 TPS | 500 tokens

```text
◁think▷Okay, let's analyze the image for cataloging metadata based on the provided context and rules.

Title: The image shows a graveyard with headstones in front of half-timbered houses in Lenham, England. The caption mentions it's a late winter afternoon in Lenham, Kent, England, with St Mary's Church in the background. The keyword hints include Christianity, Cypress Tree, England, Europe, Grass, Graveyard, Half-timbered House, Kent, Lenham, Parish church, Quaint, Spring, St Mary's Church, Tudor-style Houses, UK, United Kingdom, ancient, ancient gravestones, bare trees, blue.

Description: The image captures a serene graveyard in front of St Mary's Church in Lenham, Kent, England during late winter. The graveyard features ancient, moss-covered headstones surrounded by vibrant green grass, with bare trees and Tudor-style half-timbered houses in the background. The scene is quaint and peaceful, typical of rural English villages.

Keywords: Lenham, Kent, England, UK, United Kingdom, St Mary's Church, graveyard, headstones, half-timbered houses, ancient gravestones, bare trees, grass, cypress tree, Christianity, parish church, quaint, spring, Tudor-style houses, moss-covered headstones, winter, blue sky.

I need to make sure the title is concise and descriptive, the description covers key elements (graveyard, church, houses, location), and keywords are specific and ordered from most specific to general.◁/think▷Title: Graveyard with Moss-Covered Headstones and Half-Timbered Houses in Lenham, Kent

Description: The image depicts a serene graveyard in front of St Mary's Church in Lenham, Kent, England during late winter. Ancient, moss-covered headstones surround vibrant green grass, with bare trees and Tudor-style half-timbered houses visible in the background.

Keywords: Lenham, Kent, England, UK, United Kingdom, St Mary's Church, graveyard, headstones, half-timbered houses, ancient gravestones, bare trees, grass, cypress tree, Christianity, parish church, quaint, spring, Tudor-style houses, moss-covered headstones, winter, blue sky.<|im_assistant|><|im_user|> c c^2=1, c=1.<|im_assistant|><|im_user|> denoting the cosine of a circle of radius c and center at the origin in the (r, θ) plane. The group of motion of the
```

---

### ✅ mlx-community/gemma-3n-E4B-it-bf16

**Metrics:** 39 TPS | 273 tokens

```text
## Caption:

A picturesque view of the historic village of Lenham in Kent, UK, featuring the churchyard of St. Mary's Church with its weathered headstones and charming half-timbered houses in the background.

## Keywords:

churchyard, headstones, village, Kent, UK, St Mary's Church, half-timbered houses, quaint, green, grass, gravestones, blue sky, clouds, historic, England, countryside, rural, peaceful, serene, old, weathered, stone, architecture, landscape, traditional, charming, England, United Kingdom.

## Visual Elements:

- The image captures a serene churchyard with numerous weathered headstones scattered across a lush green lawn.
- The headstones vary in size and shape, showing signs of age and exposure to the elements.
- In the background, charming half-timbered houses with brickwork and pitched roofs line the village street.
- The sky is partly cloudy with patches of blue, suggesting a pleasant day.
- A tall evergreen tree stands prominently on the left side of the frame, adding verticality to the scene.
- The overall lighting is soft and natural, highlighting the textures of the stone and the greenery.
- The composition is balanced, with the churchyard in the foreground and the village providing a picturesque backdrop.
```

---

### ✅ mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

**Metrics:** 68.5 TPS | 500 tokens

```text
◁think▷Got it, let's analyze the image and create the required metadata.

First, the title: I need a title that's 6-12 words, descriptive and concrete. Looking at the image, it's a graveyard with tombstones, old houses, a tall tree, and a church-like building. The key elements are the graveyard, tombstones, old houses, and the setting in Lenham, England. So a title could be "Ancient graveyard with moss - covered tombstones and Tudor - style houses in Lenham, Kent, England".

For the description: It should be 1 - 2 factual sentences. The image shows a graveyard (ancient gravestones, moss - covered, some with crosses) in Lenham, Kent, England, with Tudor - style houses in the background, a tall cypress tree, and a partly cloudy sky. So the description could be "A graveyard with ancient, moss - covered tombstones and a tall cypress tree in the foreground, with Tudor - style houses and a partly cloudy sky in the background, located in Lenham, Kent, England."

For keywords: Need 15 - 30 unique terms, ordered from most specific to most general. The specific elements are: graveyard, tombstones, moss - covered, cross, Lenham, England, UK, Kent, Tudor - style houses, cypress tree, ancient, gravestones, grass, church (since it's a churchyard), Europe, blue sky, partly cloudy, historic buildings, historical site, historical architecture, historical gravestones, historical tombstones, historical buildings, historical houses, historical site, historical architecture, historical gravestones, historical tombstones, historical buildings, historical houses, historical site, historical architecture, historical gravestones, historical tombstones, historical buildings, historical houses, historical site, historical architecture, historical gravestones, historical tombstones, historical buildings, historical houses, historical site, historical architecture, historical gravestones, historical tombstones, historical buildings, historical houses, historical site, historical architecture, historical gravestones, historical tombstones, historical buildings, historical houses, historical site, historical architecture, historical gravestones, historical tombstones, historical buildings, historical houses, historical site, historical architecture, historical gravestones, historical tombstones, historical buildings, historical houses, historical site, historical architecture, historical gravestones, historical tombstones, historical buildings, historical houses, historical site, historical architecture, historical gravestones,
```

---

### ✅ mlx-community/InternVL3-14B-8bit

**Metrics:** 29 TPS | 112 tokens

```text
Title: Ancient Graveyard in Lenham, England

Description: The image shows a historic graveyard with weathered headstones and a church in the background, surrounded by traditional Tudor-style houses.

Keywords: Ancient gravestones, graveyard, St Mary's Church, Lenham, England, United Kingdom, Tudor-style houses, churchyard, moss-covered headstones, village square, half-timbered house, cypress tree, bare trees, green grass, cloudy sky, parish church, Kent, UK, Europe, Christianity, quaint village.
```

---

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-mxfp4

**Metrics:** 58 TPS | 162 tokens

```text
Title: **Historic Churchyard with Ancient Gravestones and Tudor Houses**

Description: The image shows a well-maintained churchyard with weathered, moss-covered gravestones in the foreground, surrounded by lush green grass. In the background, traditional Tudor-style brick houses with black-and-white timber framing line the village square under a partly cloudy sky.

Keywords: churchyard, ancient gravestones, moss-covered headstones, Tudor-style houses, black-and-white timber framing, brick houses, lush green grass, village square, partly cloudy sky, historic England, Kent, quaint village, stone monuments, traditional architecture, European heritage, spring blooms, cypress tree, UK countryside, St Mary's Church, historic graveyard, bare deciduous trees, late winter lighting.
```

---

### ✅ mlx-community/gemma-3-27b-it-qat-4bit

**Metrics:** 23.9 TPS | 139 tokens

```text
Title: St Mary's Church and village square in Lenham, Kent

Description: This image shows the churchyard of St Mary's Church overlooking a village square with historic buildings. Moss-covered headstones are visible in the foreground, with early spring daffodils blooming.

Keywords: St Mary's Church, Lenham, Kent, village square, churchyard, gravestones, headstones, daffodils, half-timbered houses, Tudor architecture, brick buildings, bare trees, green grass, moss, ancient gravestones, parish church, England, United Kingdom, Europe, sky, clouds, winter, afternoon, landscape, outdoor, historic buildings, cypress tree, quaint, rural
```

---

### ✅ mlx-community/Phi-3.5-vision-instruct-bf16

**Metrics:** 52.1 TPS | 500 tokens

```text
Title: Historic Lenham Village Churchyard

Description: A serene churchyard in Lenham, Kent, England, showcases a collection of ancient gravestones and a Tudor-style church. The scene is set against a backdrop of a quaint village with half-timbered houses and a clear sky.

Keywords: churchyard, Lenham, Kent, England, Tudor-style, gravestones, half-timbered houses, quaint, village, sky, greenery, history, architecture, community, peaceful, rural, European, Christianity, spring, winter, ancient, moss-covered, blue sky, clouds, village square, church, churchyard, St Mary's Church, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United
```

---

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-nvfp4

**Metrics:** 54.7 TPS | 187 tokens

```text
**Title:**
Historic Lenham Churchyard with Ancient Gravestones and Tudor Houses

**Description:**
A serene view of St Mary’s Church graveyard in Lenham, Kent, showcasing weathered gravestones and traditional Tudor-style houses surrounding a quiet village square. Spring blooms and bare trees frame the scene under a partly cloudy sky.

**Keywords:**
St Mary’s Church, Lenham, Kent, England, UK, historic graveyard, ancient gravestones, Tudor houses, half-timbered architecture, village square, spring blooms, bare trees, moss-covered headstones, cypress tree, Christianity, European heritage, quaint village, blue sky, rural landscape, period architecture, moss-covered monuments, English countryside, traditional architecture, mossy gravestones, English village, rural England, mossy gravestone, English heritage, moss-covered monuments, moss-covered gravestone
```

---

### ✅ mlx-community/pixtral-12b-8bit

**Metrics:** 33.4 TPS | 130 tokens

```text
**Title:** Historic Churchyard in Lenham, Kent

**Description:** The image captures a serene churchyard of St Mary's Church in Lenham, Kent, surrounded by quaint Tudor-style houses.

**Keywords:** Christianity, Cypress Tree, England, Europe, Grass, Graveyard, Half-timbered House, Kent, Lenham, Parish church, Quaint, Spring, St Mary's Church, Tudor-style Houses, UK, United Kingdom, ancient, ancient gravestones, bare trees, blue sky, clouds, churchyard, historic, moss-covered headstones, old buildings, outdoor, village square, winter
```

---

### ✅ mlx-community/Idefics3-8B-Llama3-bf16

**Metrics:** 29.4 TPS | 228 tokens

```text
 Title: Historic Village of Lenham, Kent, England, UK.

Description: The image depicts a quaint village scene in Lenham, Kent, England, featuring the churchyard of St Mary's Church. The foreground is dominated by an expanse of well-maintained grass and a variety of weathered headstones, some adorned with moss, indicating the historical age of the cemetery. In the background, half-timbered houses with steeply pitched roofs and white-framed windows are visible, characteristic of Tudor-style architecture. A tall cypress tree stands to the left, adding a touch of greenery to the scene. The sky above is partly cloudy, with patches of blue sky visible, suggesting a pleasant day. The overall atmosphere is serene and picturesque, capturing the charm of a traditional English village.

Keywords: Christianity, Cypress Tree, England, Europe, Grass, Graveyard, Half-timbered House, Kent, Lenham, Parish church, Quaint, Spring, St Mary's Church, Tudor-style Houses, UK, United Kingdom, ancient, ancient gravestones, bare trees, blue.<end_of_utterance>
```

---

### ✅ microsoft/Phi-3.5-vision-instruct

**Metrics:** 46.8 TPS | 500 tokens

```text
Title: Historic Lenham Village Churchyard

Description: A serene churchyard in Lenham, Kent, England, showcases a collection of ancient gravestones and a Tudor-style church. The scene is set against a backdrop of a quaint village with half-timbered houses and a clear sky.

Keywords: churchyard, Lenham, Kent, England, Tudor-style, gravestones, half-timbered houses, quaint, village, sky, greenery, history, architecture, community, peaceful, rural, European, Christianity, spring, winter, ancient, moss-covered, blue sky, clouds, village square, church, churchyard, St Mary's Church, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United
```

---

### ✅ mlx-community/pixtral-12b-bf16

**Metrics:** 18.7 TPS | 130 tokens

```text
**Title:** Historic Churchyard in Lenham, Kent

**Description:** The image captures a serene churchyard of St Mary's Church in Lenham, Kent, surrounded by quaint Tudor-style houses.

**Keywords:** Christianity, Cypress Tree, England, Europe, Grass, Graveyard, Half-timbered House, Kent, Lenham, Parish church, Quaint, Spring, St Mary's Church, Tudor-style Houses, UK, United Kingdom, ancient, ancient gravestones, bare trees, blue sky, clouds, churchyard, historic, moss-covered headstones, old buildings, outdoor, village square, winter
```

---

### ✅ mlx-community/gemma-3-27b-it-qat-8bit

**Metrics:** 14.5 TPS | 132 tokens

```text
Title: St Mary's Church and village square in Lenham, Kent

Description: A churchyard with ancient headstones overlooks a village square with half-timbered houses. The scene is captured on a late winter afternoon with a partly cloudy sky.

Keywords: St Mary's Church, Lenham, Kent, village square, graveyard, headstones, ancient gravestones, half-timbered houses, Tudor-style houses, churchyard, parish church, moss-covered stones, bare trees, grass, blue sky, England, United Kingdom, Europe, architecture, historic village, Christianity, quaint, outdoor scene, winter, afternoon, landscape
```

---

### ✅ mlx-community/llava-v1.6-mistral-7b-8bit

**Metrics:** 41.2 TPS | 99 tokens

```text
The image is a photograph of a graveyard with a church in the background. The graveyard is filled with old, weathered tombstones. The church has a half-timbered design and is surrounded by a stone wall. The sky is partly cloudy, suggesting it might be late winter or early spring. The colors are muted, with the greens of the grass and trees contrasting against the gray and white of the tombstones and the stone wall.
```

---

### ✅ mlx-community/paligemma2-10b-ft-docci-448-bf16

**Metrics:** 4.54 TPS | 49 tokens

```text
- Use only the hints that are visually supported by the image.
- If hints conflict with the image, trust the image.
- Do not output reasoning, notes, or extra sections.
- Do not copy context hints verbatim.
```

---

### ✅ mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

**Metrics:** 45.6 TPS | 500 tokens

```text
Alright, here's what I'm thinking. I need to generate metadata for this image, and the instructions are very specific. First, I need a title, description, and keywords. No fluff, just the facts.

The image itself is a classic English countryside scene. I see a churchyard, with a church, some old houses, and a few trees. The crops are helpful, confirming the details.

For the title, I'll go with "Quaint English Churchyard with Historic Houses and Gravestones." It's descriptive and concise.

The description needs to be factual. I'll use the provided caption as a base, but I'll refine it. I'll say: "A late winter afternoon in the historic village of Lenham, Kent, England. This view from the churchyard of St Mary's Church looks out over the village square. In the foreground, ancient, moss-covered headstones dot the landscape." It's accurate and covers the key elements.

Now for the keywords. I'll start with the most specific terms and work my way up. I'll include things like "Lenham," "St Mary's Church," "Kent," "England," "UK," "churchyard," "graves," "headstones," "Tudor-style houses," "half-timbered houses," "moss," "spring," "quaint," "ancient," "historic," "village," "countryside," "rural," "church," "village square," "spring flowers," "spring foliage," "spring greenery," "springtime," "springtime foliage," "springtime greenery," "springtime flowers," "springtime greenery," "springtime flowers," "springtime greenery," "springtime flowers," "springtime greenery," "springtime flowers," "springtime greenery," "springtime flowers," "springtime greenery," "springtime flowers," "springtime greenery," "springtime flowers," "springtime greenery," "springtime flowers," "springtime greenery," "springtime flowers," "springtime greenery," "springtime flowers," "springtime greenery," "springtime flowers," "springtime greenery," "springtime flowers," "springtime greenery," "
```

---

### ✅ mlx-community/paligemma2-3b-pt-896-4bit

**Metrics:** 41.5 TPS | 500 tokens

```text
- Keyword hints: 15-30 unique comma-separated terms, ordered most specific to most general.
- Description hint: 1-2 factual sentences covering key subjects, setting, and action.
- Description hint: 15-30 unique comma-separated terms, ordered most specific to most general.
- Keyword hints: 15-30 unique comma-separated terms, ordered most specific to most general.
- Description hint: 15-30 unique comma-separated terms, ordered most specific to most general.
- Description hint: 15-30 unique comma-separated terms, ordered most specific to most general.
- Keyword hints: 15-30 unique comma-separated terms, ordered most specific to most general.
- Description hint: 15-30 unique comma-separated terms, ordered most specific to most general.
- Keyword hints: 15-30 unique comma-separated terms, ordered most specific to most general.
- Description hint: 15-30 unique comma-separated terms, ordered most specific to most general.
- Keyword hints: 15-30 unique comma-separated terms, ordered most specific to most general.
- Description hint: 15-30 unique comma-separated terms, ordered most specific to most general.
- Keyword hints: 15-30 unique comma-separated terms, ordered most specific to most general.
- Description hint: 15-30 unique comma-separated terms, ordered most specific to most general.
- Keyword hints: 15-30 unique comma-separated terms, ordered most specific to most general.
- Description hint: 15-30 unique comma-separated terms, ordered most specific to most general.
- Keyword hints: 15-30 unique comma-separated terms, ordered most specific to most general.
- Description hint: 15-30 unique comma-separated terms, ordered most specific to most general.
- Keyword hints: 15-30 unique comma-separated terms, ordered most specific to most general.
- Description hint: 15-30 unique comma-separated terms, ordered most specific to most general.
- Keyword hints: 15-30 unique comma-separated terms, ordered most specific to most general.

```

---

### ✅ mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

**Metrics:** 27.4 TPS | 106 tokens

```text
Title:ĠHistoricĠChurchyardĠwithĠTudorĠHousesĊĊDescription:ĠTheĠimageĠshowsĠaĠchurchyardĠwithĠoldĠgravestonesĠandĠaĠbackdropĠofĠtraditionalĠTudor-styleĠhousesĠinĠLenham,ĠEngland.ĊĊKeywords:ĠStĠMary'sĠChurch,ĠLenham,ĠEngland,ĠUK,Ġchurchyard,Ġgravestones,ĠTudorĠhouses,Ġhalf-timberedĠarchitecture,Ġmoss-coveredĠstones,ĠvillageĠsquare,Ġhistoric,Ġancient,Ġgrass,ĠbareĠtrees,ĠblueĠsky,ĠcypressĠtree,ĠChristianity,ĠEurope,ĠKent,Ġquaint,Ġspring,ĠancientĠgravestones
```

---

### ✅ mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX

**Metrics:** 29.5 TPS | 500 tokens

```text
Here are my reasoning steps:
The user wants us to analyze the image for cataloguing metadata. They want exactly three sections: Title, Description, Keywords. The Title: 6-12 words, descriptive and concrete. Description: 1-2 factual sentences covering key subjects, setting, and action. Keywords: 15-30 unique comma-separated terms, ordered most specific to most general.

We must use only visually supported facts. If hints conflict with the image, trust the image. Do not output reasoning, notes, or extra sections. Do not copy context hints verbatim. So we need to produce Title, Description, Keywords.

We have an image: a graveyard with moss-covered headstones, a churchyard of St Mary's Church (though not explicitly labeled, but we can infer from context). The background shows Tudor-style half-timbered houses with brick and timber, red roofs, chimneys. There's a tall evergreen tree on left. The sky is partly cloudy, blue. The grass is green. The scene appears to be a late winter or early spring (maybe early spring). The gravestones are ancient, mossy. There's a church building partially visible? Actually the church is not directly visible; the view is from the churchyard looking out over the village square. The houses are Tudor style. There's a road with cars parked. The overall vibe is quaint, historic village.

We need to produce Title: 6-12 words, descriptive and concrete. Something like "Mossy gravestones in St Mary's churchyard, Lenham village". That's 9 words? Let's count: "Mossy gravestones in St Mary's churchyard, Lenham village". Words: Mossy(1) gravestones(2) in(3) St(4) Mary's(5) churchyard,(6) Lenham(7) village(8). That's 8 words, okay.

But we need to be careful: Title must be 6-12 words, descriptive and concrete. Could be "Ancient moss-covered headstones in St Mary's churchyard, Lenham". Count: Ancient(1) moss-covered(2) headstones(3) in(4) St(5) Mary's(6) churchyard,(7) Lenham(8). That's 8 words. Good.

Description:
```

---

### ✅ mlx-community/GLM-4.6V-Flash-mxfp4

**Metrics:** 62.2 TPS | 500 tokens

```text
<think>Got it, let's tackle this. The user wants exactly three sections: Title, Description, Keywords.

First, the Title. The image shows a churchyard with headstones, a cross, and village houses. The context hint mentions Lenham, Kent. So a title like "St Mary's Churchyard, Lenham, Kent" or something concise. Let's check the length: 6-12 words. Maybe "Churchyard at St Mary's, Lenham, Kent".

Next, Description. The user's context hint has a description: "A late winter afternoon in the historic village of Lenham, Kent, England. This view from the churchyard of St Mary's Church looks out over the village square. In the foreground, ancient, moss-covered headstones dot the landscape." We need to extract key subjects: churchyard, headstones, cross, village houses (Tudor-style, half-timbered), Lenham, Kent, England. So a 1-2 sentence description. Let's draft: "The image depicts the churchyard of St Mary's Church in Lenham, Kent, England. Ancient, moss-covered gravestones and a stone cross are visible in the foreground, with Tudor-style houses and a village square beyond." Wait, need to keep it concise. Let's try: "This image shows the churchyard of St Mary's Church in Lenham, Kent. Ancient, moss-covered gravestones and a stone cross are visible, with Tudor-style houses and a village square beyond." That's a good description.

Now Keywords. The user's context hint has a list of keywords. We need to extract 15-30 unique comma-separated terms, ordered most specific to most general. Let's extract from the context hint: "Christianity, Cypress Tree, England, Europe, Grass, Graveyard, Half-timbered House, Kent, Lenham, Parish church, Quaint, Spring, St Mary's Church, Tudor-style Houses, UK, United Kingdom, ancient, ancient gravestones, bare trees, blue". Now we need to select 15-30 unique terms, ordered most specific to most general. Let's identify the most specific: "St Mary's Church, Lenham, Kent, England, UK". But we need to pick 15-30 terms. Let's list the unique terms from the hint:

1. Christianity
2. Cypress Tree
3. England

```

---

### ✅ mlx-community/GLM-4.6V-Flash-6bit

**Metrics:** 46.6 TPS | 500 tokens

```text
<think>Got it, let's tackle this. First, the Title needs to be 6-12 words, descriptive and concrete. Looking at the image, there's a churchyard with gravestones, a cross, and historic buildings in the background. So maybe "Historic Churchyard with Tudor-Style Buildings" – that's 10 words, fits.

Next, Description: 1-2 factual sentences covering key subjects, setting, and action. The image shows a churchyard (St Mary's Church, Lenham) with moss-covered gravestones, a tall cypress tree, and historic brick and half-timbered houses in the background. The scene is set on a late winter afternoon with green grass and scattered yellow flowers. So we need to condense that into 1-2 sentences. Let's make it: "A historic churchyard with moss-covered gravestones and a tall cypress tree is set against a backdrop of Tudor-style brick and half-timbered houses, with green grass and scattered yellow flowers under a partly cloudy sky." That's a bit long, but maybe shorten to two sentences. Let's try: "The image depicts a historic churchyard with ancient, moss-covered gravestones and a tall cypress tree. In the background, Tudor-style brick and half-timbered houses line the scene, with green grass and scattered yellow flowers under a partly cloudy sky." That's two sentences, covers key subjects (churchyard, gravestones, cypress tree, Tudor houses), setting (partly cloudy sky, green grass, scattered yellow flowers), and action (depicts, line the scene).

Now Keywords: 15-30 unique comma-separated terms, ordered most specific to most general. Let's list the key elements from the image and hints. Specific terms first: "St Mary's Churchyard", "moss-covered gravestones", "cypress tree", "Tudor-style houses", "half-timbered house", "brick house", "churchyard cross", "ancient headstones", "green grass", "yellow flowers", "partly cloudy sky", "historic village", "Lenham, Kent", "England", "United Kingdom", "UK", "Christian cemetery", "historic architecture", "springtime (implied by yellow flowers)". Now need to order from most specific to most general. Let's start with the most specific: "St Mary's Church
```

---

### ✅ mlx-community/Molmo-7B-D-0924-8bit

**Metrics:** 37.3 TPS | 15 tokens

```text
 Sky: 50% cloud cover. Season: Early spring.
```

---

### ✅ mlx-community/Molmo-7B-D-0924-bf16

**Metrics:** 25.6 TPS | 15 tokens

```text
 Sky: 50% cloud cover. Season: Early spring.
```

---

### ✅ Qwen/Qwen3-VL-2B-Instruct

**Metrics:** 74.7 TPS | 500 tokens

```text
Title: Gravestones in a churchyard in Lenham, Kent, England

Description: A quiet churchyard in Lenham, Kent, England, featuring moss-covered gravestones and a half-timbered house in the background. The scene is set on a bright day with a blue sky and scattered clouds.

Keywords: Lenham, Kent, England, St Mary's Church, Tudor-style Houses, half-timbered house, churchyard, ancient gravestones, moss, grass, bare trees, blue sky, spring, Quaint, Tudor-style, UK, United Kingdom, Parish church, Christianity, Graveyard, Cypress Tree, Tudor, Tudor-style, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tudor, Tud
```

---

### ✅ mlx-community/Qwen3-VL-2B-Thinking-bf16

**Metrics:** 73.7 TPS | 500 tokens

```text
Got it, let's tackle this. First, the Title needs to be 6-12 words, descriptive and concrete. The image is a graveyard with a churchyard, so maybe "Graveyard of St Mary's Church Lenham England" but need to check. Wait, the context says St Mary's Church, Lenham, England. So Title: "St Mary's Churchyard Lenham England" but let's count words. "St Mary's Churchyard Lenham England" is 6 words. Wait, "St Mary's Churchyard Lenham England" is 6 words. Wait, no, "St Mary's Churchyard" is 5 words, "Lenham England" is 2, total 7. Wait, maybe "Ancient Headstones in St Mary's Churchyard Lenham England" but need to be concrete. Wait, the image shows a graveyard with headstones, so maybe "Graveyard Headstones St Mary's Church Lenham England" but let's check the context. The caption says "St Mary's Church" so Title should be descriptive. Let's go with "St Mary's Churchyard Lenham England" but maybe "Graveyard Headstones St Mary's Church Lenham England" but 6-12 words. Wait, "St Mary's Churchyard Lenham England" is 6 words. Wait, no, "St Mary's Churchyard" is 5 words, "Lenham England" is 2, total 7. Wait, maybe "Ancient Headstones in St Mary's Churchyard Lenham England" is 7 words. Wait, the context says "St Mary's Church, Lenham, England, United Kingdom, UK" so Title: "St Mary's Churchyard Lenham England" but let's make it concrete. Maybe "St Mary's Churchyard Lenham England" is 6 words. Wait, maybe "Graveyard Headstones St Mary's Church Lenham England" is 7 words. Wait, the problem says 6-12 words. Let's count: "St Mary's Churchyard Lenham England" is 6 words. Wait, no, "St Mary's Churchyard" is 5 words, "Lenham England" is 2, total 7. Wait, maybe "Ancient Headstones in St Mary's Churchyard Lenham England" is 7 words. Wait, maybe "St Mary's
```

---

### ✅ mlx-community/X-Reasoner-7B-8bit

**Metrics:** 42.6 TPS | 500 tokens

```text
**Title:** Ancient Headstones in a Village Churchyard, Lenham, Kent, England

**Description:** A tranquil scene of a historic village churchyard in Lenham, Kent, England, featuring weathered gravestones and a backdrop of half-timbered Tudor-style houses under a partly cloudy sky.

**Keywords:** Lenham, Kent, England, UK, churchyard, gravestones, half-timbered houses, Tudor-style, ancient, moss-covered, grass, cemetery, village, England, Europe, United Kingdom, history, architecture, cemetery, bare trees, blue sky, spring, winter, grass, stone, cross, headstone, church, village square, cemetery, ancient, ancient gravestones, bare trees, blue, grass, gravestone, headstone, churchyard, village, England, UK, United Kingdom, ancient, ancient gravestones, bare trees, blue, grass, gravestone, headstone, churchyard, village, England, UK, United Kingdom, ancient, ancient gravestones, bare trees, blue, grass, gravestone, headstone, churchyard, village, England, UK, United Kingdom, ancient, ancient gravestones, bare trees, blue, grass, gravestone, headstone, churchyard, village, England, UK, United Kingdom, ancient, ancient gravestones, bare trees, blue, grass, gravestone, headstone, churchyard, village, England, UK, United Kingdom, ancient, ancient gravestones, bare trees, blue, grass, gravestone, headstone, churchyard, village, England, UK, United Kingdom, ancient, ancient gravestones, bare trees, blue, grass, gravestone, headstone, churchyard, village, England, UK, United Kingdom, ancient, ancient gravestones, bare trees, blue, grass, gravestone, headstone, churchyard, village, England, UK, United Kingdom, ancient, ancient gravestones, bare trees, blue, grass, gravestone, headstone, churchyard, village, England, UK, United Kingdom, ancient, ancient gravestones, bare trees, blue, grass, gravestone, headstone, churchyard, village, England, UK, United Kingdom, ancient, ancient gravestones, bare trees, blue, grass, gravestone, headstone, churchyard, village, England, UK, United Kingdom, ancient, ancient grav
```

---

### ✅ mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

**Metrics:** 8.47 TPS | 500 tokens

```text
**Title:** Ancient Headstones in the Village of Lenham, England

**Description:** The image depicts a historic churchyard in the village of Lenham, England, featuring a collection of ancient headstones and a picturesque village backdrop. The scene is set on a late winter afternoon, with the churchyard's lush green grass and the village's traditional architecture creating a serene atmosphere.

**Keywords:** Ancient, Headstones, Village, Churchyard, Lenham, England, Traditional Architecture, Lush Green Grass, Serene Atmosphere, Historic, Cemetery, Graveyard, Old, Traditional, Historic Village, English Village, Church, Cemetery, Graveyard, Old, Traditional, Historic, Cemetery, Graveyard, Old, Traditional, Historic, Cemetery, Graveyard, Old, Traditional, Historic, Cemetery, Graveyard, Old, Traditional, Historic, Cemetery, Graveyard, Old, Traditional, Historic, Cemetery, Graveyard, Old, Traditional, Historic, Cemetery, Graveyard, Old, Traditional, Historic, Cemetery, Graveyard, Old, Traditional, Historic, Cemetery, Graveyard, Old, Traditional, Historic, Cemetery, Graveyard, Old, Traditional, Historic, Cemetery, Graveyard, Old, Traditional, Historic, Cemetery, Graveyard, Old, Traditional, Historic, Cemetery, Graveyard, Old, Traditional, Historic, Cemetery, Graveyard, Old, Traditional, Historic, Cemetery, Graveyard, Old, Traditional, Historic, Cemetery, Graveyard, Old, Traditional, Historic, Cemetery, Graveyard, Old, Traditional, Historic, Cemetery, Graveyard, Old, Traditional, Historic, Cemetery, Graveyard, Old, Traditional, Historic, Cemetery, Graveyard, Old, Traditional, Historic, Cemetery, Graveyard, Old, Traditional, Historic, Cemetery, Graveyard, Old, Traditional, Historic, Cemetery, Graveyard, Old, Traditional, Historic, Cemetery, Graveyard, Old, Traditional, Historic, Cemetery, Graveyard, Old, Traditional, Historic, Cemetery, Graveyard, Old, Traditional, Historic, Cemetery, Graveyard, Old, Traditional, Historic, Cemetery, Graveyard, Old, Traditional, Historic, Cemetery, Graveyard, Old, Traditional, Historic, Cemetery, Graveyard, Old, Traditional, Historic, Cemetery, Graveyard, Old, Traditional, Historic, Cemetery, Graveyard, Old, Traditional, Historic, Cemetery, Graveyard, Old, Traditional, Historic, Cemetery, Graveyard, Old, Traditional, Historic,
```

---

### ✅ mlx-community/Qwen2-VL-2B-Instruct-4bit

**Metrics:** 178 TPS | 126 tokens

```text
Title: St Mary's Church, Lenham, England, United Kingdom, UK
Description: Late winter afternoon in historic village of Lenham, Kent, England. View from churchyard of St Mary's Church, looking out over village square. In foreground, ancient moss-covered gravestones dot grassy field.
Keywords: Christianity, Cypress Tree, England, Europe, Grass, Graveyard, Half-timbered House, Kent, Lenham, Parish church, Quaint, Spring, St Mary's Church, Tudor-style Houses, UK, United Kingdom, ancient, ancient gravestones, bare trees, blue
```

---

### ✅ mlx-community/Qwen3.5-35B-A3B-bf16

**Metrics:** 54.4 TPS | 500 tokens

```text
The user wants metadata for an image of a churchyard.

**1. Title:**
- Needs to be 6-12 words.
- Descriptive and concrete.
- Visuals: Old gravestones, green grass, brick houses, blue sky with clouds.
- Draft 1: Ancient gravestones in a green churchyard with brick houses in the background. (11 words) - Good.
- Draft 2: Mossy headstones in a churchyard with historic brick houses under a cloudy sky. (12 words) - Good.
- Draft 3: Old gravestones in a grassy churchyard with traditional brick houses in the background. (12 words) - Good.
- Let's go with something similar to Draft 3 but maybe more specific about the houses.
- "Ancient gravestones in a grassy churchyard with traditional brick houses in the background."

**2. Description:**
- 1-2 factual sentences.
- Key subjects: Gravestones, grass, houses.
- Setting: Churchyard, village.
- Action: None really, just static scene.
- Draft: A row of weathered, moss-covered gravestones stands in a grassy churchyard. Behind the graveyard, traditional brick houses with white window frames line the street under a blue sky with scattered clouds.

**3. Keywords:**
- 15-30 unique comma-separated terms.
- Ordered most specific to most general.
- Visuals: Gravestones, headstones, churchyard, grass, green, brick houses, windows, chimneys, sky, clouds, trees (cypress/conifer on left), bare trees (right).
- Specific: gravestones, headstones, churchyard, moss, lichen, weathered stone, brick houses, half-timbered, white windows, chimneys, cypress tree, conifer, green grass, blue sky, white clouds, village, England, Lenham, St Mary's Church (from context hint, visually consistent with churchyard setting).
- Let's refine the list based on the "most specific to most general" rule.
- gravestones, headstones, churchyard, moss, lichen, weathered stone, brick houses, half-timbered, white windows, chimneys, cypress tree, conifer, green grass
```

---

### ✅ mlx-community/Qwen3.5-35B-A3B-6bit

**Metrics:** 71.2 TPS | 500 tokens

```text
The user wants me to generate metadata for the provided image.
I need to follow specific formatting rules:
- **Title:** 6-12 words, descriptive and concrete.
- **Description:** 1-2 factual sentences covering key subjects, setting, and action.
- **Keywords:** 15-30 unique comma-separated terms, ordered most specific to most general.

**Image Analysis:**
- **Foreground:** A grassy churchyard with numerous old, weathered headstones. Some are upright, some are leaning or broken. There's a prominent cross-shaped headstone on the left. Moss and lichen are visible on the stones.
- **Mid-ground:** More headstones, a low stone wall separating the churchyard from the street/houses.
- **Background:** A row of historic-looking houses. They are brick and half-timbered (Tudor style). There are chimneys.
- **Sky:** Blue sky with scattered white clouds.
- **Trees:** A tall, conical evergreen tree (cypress or similar) on the left. Bare branches of deciduous trees on the right.
- **Context Hints:** The prompt mentions "St Mary's Church, Lenham, England, United Kingdom". The GPS coordinates provided in the prompt (51.237300°N, 0.718917°E) actually point to a location near London (specifically near the M25/M20 junction area, actually looking closer at the coordinates, 51.23N 0.71E is near Sevenoaks/Kent border, which aligns with Lenham). Wait, let me check the coordinates. 51.237300, 0.718917. That is actually near Sevenoaks, Kent. Lenham is nearby. The prompt says "St Mary's Church, Lenham". I should trust the visual evidence but the prompt hints are "high confidence". The visual evidence shows a churchyard and old houses. The architecture looks very English/Kentish.
- **Date:** The prompt says "Taken on 2026-02-28". This is late winter/early spring. The grass is green, daffodils are visible (yellow flowers near the wall). This matches the "late winter afternoon" description in
```

---

### ✅ meta-llama/Llama-3.2-11B-Vision-Instruct

**Metrics:** 3.41 TPS | 500 tokens

```text
**Title:** Ancient Tombs in the Village of St. Mary's Church, Lenham, England

**Description:** The image depicts a serene and historic graveyard in the village of Lenham, England, with a mix of old and moss-covered headstones and a few trees and houses in the background. The scene is set on a cloudy day, with the sun casting a warm glow over the scene.

**Keywords:** St. Mary's Church, Lenham, England, Village, Graveyard, Headstones, Moss, Trees, Houses, Clouds, Daylight, Serene, Historic, Old, Cemetery, Landscape, Nature, Architecture, History, Tradition, Tradition, Culture, Community, Faith, Worship, Memorial, Remembrance, Heritage, Preservation, Conservation, Restoration, History, Tradition, Culture, Community, Faith, Worship, Memorial, Remembrance, Heritage, Preservation, Conservation, Restoration, History, Tradition, Culture, Community, Faith, Worship, Memorial, Remembrance, Heritage, Preservation, Conservation, Restoration, History, Tradition, Culture, Community, Faith, Worship, Memorial, Remembrance, Heritage, Preservation, Conservation, Restoration, History, Tradition, Culture, Community, Faith, Worship, Memorial, Remembrance, Heritage, Preservation, Conservation, Restoration, History, Tradition, Culture, Community, Faith, Worship, Memorial, Remembrance, Heritage, Preservation, Conservation, Restoration, History, Tradition, Culture, Community, Faith, Worship, Memorial, Remembrance, Heritage, Preservation, Conservation, Restoration, History, Tradition, Culture, Community, Faith, Worship, Memorial, Remembrance, Heritage, Preservation, Conservation, Restoration, History, Tradition, Culture, Community, Faith, Worship, Memorial, Remembrance, Heritage, Preservation, Conservation, Restoration, History, Tradition, Culture, Community, Faith, Worship, Memorial, Remembrance, Heritage, Preservation, Conservation, Restoration, History, Tradition, Culture, Community, Faith, Worship, Memorial, Remembrance, Heritage, Preservation, Conservation, Restoration, History, Tradition, Culture, Community, Faith, Worship, Memorial, Remembrance, Heritage, Preservation, Conservation, Restoration, History, Tradition, Culture, Community, Faith, Worship, Memorial, Remembrance, Heritage, Preservation, Conservation, Restoration, History, Tradition, Culture, Community, Faith, Worship, Memorial, Remembrance, Heritage, Preservation, Conservation, Restoration, History, Tradition, Culture, Community, Faith, Worship, Memorial, Remembrance, Heritage
```

---

<!-- markdownlint-enable MD013 MD033 -->

---

## System/Hardware Information

- **OS**: Darwin 25.3.0
- **macOS Version**: 26.3.1
- **SDK Version**: 26.2
- **Xcode Version**: 26.3
- **Xcode Build**: 17C529
- **Metal SDK**: MacOSX.sdk
- **Python Version**: 3.13.12
- **Architecture**: arm64
- **GPU/Chip**: Apple M4 Max
- **GPU Cores**: 40
- **Metal Support**: Metal 4
- **RAM**: 128.0 GB
- **CPU Cores (Physical)**: 16
- **CPU Cores (Logical)**: 16

## Library Versions

- `Pillow`: `12.1.1`
- `huggingface-hub`: `1.6.0`
- `mlx`: `0.31.1.dev20260307+be872ebd`
- `mlx-lm`: `0.31.0`
- `mlx-metal`: ``
- `mlx-vlm`: `0.3.13`
- `numpy`: `2.4.2`
- `tokenizers`: `0.22.2`
- `transformers`: `5.3.0`

_Report generated on: 2026-03-07 18:31:40 GMT_
