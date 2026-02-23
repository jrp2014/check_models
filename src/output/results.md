# Model Performance Results

_Generated on 2026-02-23 12:54:48 GMT_

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/nanoLLaVA-1.5-4bit` (332.0 tps)
- **💾 Most efficient:** `mlx-community/FastVLM-0.5B-bf16` (2.2 GB)
- **⚡ Fastest load:** `mlx-community/LFM2-VL-1.6B-8bit` (0.96s)
- **📊 Average TPS:** 85.2 across 42 models

## 📈 Resource Usage

- **Total peak memory:** 649.5 GB
- **Average peak memory:** 15.5 GB
- **Memory efficiency:** 223 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 8 | ✅ B: 10 | 🟡 C: 7 | 🟠 D: 4 | ❌ F: 13

**Average Utility Score:** 51/100

**Existing Metadata Baseline:** 🏆 A (90/100)
**Vs Existing Metadata:** Avg Δ -39 | Better: 0, Neutral: 4, Worse: 38

- **Best for cataloging:** `mlx-community/Kimi-VL-A3B-Thinking-8bit` (🏆 A, 91/100)
- **Worst for cataloging:** `mlx-community/Molmo-7B-D-0924-8bit` (❌ F, 0/100)

### ⚠️ 17 Models with Low Utility (D/F)

- `HuggingFaceTB/SmolVLM-Instruct`: ❌ F (21/100) - Mostly echoes context without adding value
- `meta-llama/Llama-3.2-11B-Vision-Instruct`: 🟠 D (50/100) - Mostly echoes context without adding value
- `microsoft/Phi-3.5-vision-instruct`: 🟠 D (46/100) - Mostly echoes context without adding value
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (4/100) - Output too short to be useful
- `mlx-community/FastVLM-0.5B-bf16`: ❌ F (31/100) - Mostly echoes context without adding value
- `mlx-community/Idefics3-8B-Llama3-bf16`: ❌ F (34/100) - Mostly echoes context without adding value
- `mlx-community/Molmo-7B-D-0924-8bit`: ❌ F (0/100) - Empty or minimal output
- `mlx-community/Molmo-7B-D-0924-bf16`: ❌ F (0/100) - Empty or minimal output
- `mlx-community/Phi-3.5-vision-instruct-bf16`: 🟠 D (46/100) - Mostly echoes context without adding value
- `mlx-community/SmolVLM-Instruct-bf16`: ❌ F (21/100) - Mostly echoes context without adding value
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: ❌ F (28/100) - Mostly echoes context without adding value
- `mlx-community/gemma-3n-E2B-4bit`: ❌ F (4/100) - Output too short to be useful
- `mlx-community/llava-v1.6-mistral-7b-8bit`: 🟠 D (38/100) - Mostly echoes context without adding value
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (11/100) - Output lacks detail
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (10/100) - Output lacks detail
- `prince-canuma/Florence-2-large-ft`: ❌ F (0/100) - Output too short to be useful
- `qnguyen3/nanoLLaVA`: ❌ F (22/100) - Mostly echoes context without adding value

## ⚠️ Quality Issues

- **❌ Failed Models (4):**
  - `microsoft/Florence-2-large-ft` (`Model Error`)
  - `mlx-community/InternVL3-14B-8bit` (`API Mismatch`)
  - `mlx-community/InternVL3-8B-bf16` (`API Mismatch`)
  - `mlx-community/deepseek-vl2-8bit` (`Processor Error`)
- **🔄 Repetitive Output (8):**
  - `Qwen/Qwen3-VL-2B-Instruct` (token: `phrase: "car, car, car, car,..."`)
  - `meta-llama/Llama-3.2-11B-Vision-Instruct` (token: `phrase: "scenery, scenery, scenery, sce..."`)
  - `microsoft/Phi-3.5-vision-instruct` (token: `phrase: "united kingdom, united kingdom..."`)
  - `mlx-community/Idefics3-8B-Llama3-bf16` (token: `phrase: "lochside, trossachs, uk, unite..."`)
  - `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit` (token: `phrase: "revitalizing, revitalizing, re..."`)
  - `mlx-community/Phi-3.5-vision-instruct-bf16` (token: `phrase: "united kingdom, united kingdom..."`)
  - `mlx-community/SmolVLM2-2.2B-Instruct-mlx` (token: `phrase: "honda hr-v e:hev, light..."`)
  - `mlx-community/nanoLLaVA-1.5-4bit` (token: `phrase: "fog, fog, fog, fog,..."`)
- **👻 Hallucinations (2):**
  - `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`
  - `mlx-community/paligemma2-3b-pt-896-4bit`
- **📝 Formatting Issues (3):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-Flash-mxfp4`
  - `prince-canuma/Florence-2-large-ft`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 85.2 | Min: 0 | Max: 332
- **Peak Memory**: Avg: 15 | Min: 2.2 | Max: 60
- **Total Time**: Avg: 22.69s | Min: 2.23s | Max: 139.08s
- **Generation Time**: Avg: 19.72s | Min: 1.22s | Max: 134.45s
- **Model Load Time**: Avg: 2.97s | Min: 0.96s | Max: 13.27s

## 🚨 Failures by Package (Actionable)

<!-- markdownlint-disable MD060 -->

| Package | Failures | Error Types | Affected Models |
|---------|----------|-------------|-----------------|
| `mlx-vlm` | 3 | API Mismatch, Model Error | `microsoft/Florence-2-large-ft`, `mlx-community/InternVL3-14B-8bit`, `mlx-community/InternVL3-8B-bf16` |
| `model-config` | 1 | Processor Error | `mlx-community/deepseek-vl2-8bit` |

<!-- markdownlint-enable MD060 -->

### Actionable Items by Package

#### mlx-vlm

- **microsoft/Florence-2-large-ft** (Model Error)
  - Error: `Model generation failed for microsoft/Florence-2-large-ft: Failed to process inputs with error: can only concatenate ...`
  - Type: `ValueError`
- **mlx-community/InternVL3-14B-8bit** (API Mismatch)
  - Error: `Model runtime error during generation for mlx-community/InternVL3-14B-8bit: LanguageModel.__call__() got an unexpecte...`
  - Type: `ValueError`
- **mlx-community/InternVL3-8B-bf16** (API Mismatch)
  - Error: `Model runtime error during generation for mlx-community/InternVL3-8B-bf16: LanguageModel.__call__() got an unexpected...`
  - Type: `ValueError`

#### model-config

- **mlx-community/deepseek-vl2-8bit** (Processor Error)
  - Error: `Model preflight failed for mlx-community/deepseek-vl2-8bit: Loaded processor has no image_processor; expected multimo...`
  - Type: `ValueError`

> **Prompt used:**
>

> ```text
> Analyze this image for cataloguing metadata.
>
> Return exactly these three sections:
>
> Title: 6-12 words, descriptive and concrete.
>
> Description: 1-2 factual sentences covering key subjects, setting, and action.
>
> Keywords: 15-30 comma-separated terms, ordered most specific to most general.
> Use concise, image-grounded wording and avoid speculation.
>
> Context: Existing metadata hints (use only if visually consistent):
> - Description hint: , Loch Katrine, Stronachlachar, Scotland, United Kingdom, UK Based on the image and the provided context, here is a suitable caption: A vibrant blue electric vehicle provides a stark contrast to the muted tones of a rainy late-winter afternoon at Stronachlachar pier on Loch Katrine, Scotland. Rain falls on the calm waters of the loch and the surrounding bare-branched trees, with the distant hills of the Trossachs...
> - Keyword hints: Adobe Stock, Any Vision, Automobile, Automotive, Europe, Honda HR-V, Honda HR-V e:HEV, Light Blue SUV, Loch Katrine, Lochside, SUV, Scotland, Stronachlachar, Trossachs, UK, United Kingdom, Vehicles, Wet Road, asphalt, bare trees
> - Capture metadata: Taken on 2026-02-21 15:29:35 GMT (at 15:29:35 local time).
>
> Prioritize what is visibly present. If context conflicts with the image, trust the image.
> ```

**Note:** Results sorted: errors first, then by generation time (fastest to slowest).

**Overall runtime:** 971.42s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Token |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                    |   Error Package |
|:--------------------------------------------------------|--------:|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:----------------------------------|----------------:|
| `microsoft/Florence-2-large-ft`                         |         |                   |                       |                |              |           |             |                  |            |             |                                   |         mlx-vlm |
| `mlx-community/InternVL3-14B-8bit`                      |         |                   |                       |                |              |           |             |                  |            |             |                                   |         mlx-vlm |
| `mlx-community/InternVL3-8B-bf16`                       |         |                   |                       |                |              |           |             |                  |            |             |                                   |         mlx-vlm |
| `mlx-community/deepseek-vl2-8bit`                       |         |                   |                       |                |              |           |             |                  |            |             |                                   |    model-config |
| `qnguyen3/nanoLLaVA`                                    | 151,645 |               333 |                    62 |            395 |        1,606 |       102 |         4.7 |            1.22s |      1.01s |       2.23s |                                   |                 |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |       7 |               592 |                   250 |            842 |        2,471 |       292 |         2.8 |            1.50s |      0.96s |       2.45s |                                   |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |       7 |               592 |                   204 |            796 |        2,422 |       180 |         4.0 |            1.72s |      1.05s |       2.78s |                                   |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       | 151,645 |               337 |                   232 |            569 |        2,294 |       308 |         2.2 |            1.78s |      1.05s |       2.83s |                                   |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |  49,154 |             1,537 |                    37 |          1,574 |        1,278 |       115 |         5.5 |            1.85s |      1.22s |       3.07s | context-ignored                   |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |  49,154 |             1,537 |                    37 |          1,574 |        1,252 |       114 |         5.5 |            1.98s |      1.34s |       3.32s | context-ignored                   |                 |
| `prince-canuma/Florence-2-large-ft`                     |     0.0 |               905 |                   500 |          1,405 |        3,862 |       326 |         5.1 |            2.06s |      1.49s |       3.54s | ⚠️harness(stop_token), ...        |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |      11 |               333 |                   500 |            833 |        1,725 |       332 |         2.4 |            2.10s |      0.96s |       3.06s | repetitive(phrase: "fog, fog, ... |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |       1 |             1,346 |                    15 |          1,361 |          472 |      30.2 |          12 |            3.65s |      3.16s |       6.82s | context-ignored                   |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |       2 |             2,926 |                   185 |          3,111 |        1,019 |       160 |         7.8 |            4.32s |      1.64s |       5.96s |                                   |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |     106 |               593 |                   149 |            742 |          777 |      41.7 |          17 |            4.65s |      4.84s |       9.49s |                                   |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       | 236,772 |               585 |                   500 |          1,085 |        1,204 |       114 |         6.0 |            5.17s |      3.18s |       8.35s | context-ignored                   |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |      70 |             1,437 |                   500 |          1,937 |        1,223 |       113 |         5.5 |            5.93s |      1.18s |       7.11s | repetitive(phrase: "honda hr-v... |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |       1 |             1,346 |                    17 |          1,363 |          460 |      5.19 |          26 |            6.51s |      4.80s |      11.31s | context-ignored                   |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |   1,810 |             1,337 |                   500 |          1,837 |          783 |       114 |          18 |            6.63s |      3.49s |      10.11s |                                   |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |      61 |             1,337 |                   500 |          1,837 |          876 |       103 |          22 |            6.91s |      3.72s |      10.63s |                                   |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          | 163,585 |             1,337 |                   404 |          1,741 |          867 |      68.7 |          37 |            8.29s |      5.97s |      14.26s |                                   |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |     106 |               594 |                   137 |            731 |          159 |      28.0 |          19 |            8.95s |      4.92s |      13.88s |                                   |                 |
| `mlx-community/pixtral-12b-8bit`                        |       2 |             3,117 |                   118 |          3,235 |          553 |      35.3 |          16 |            9.31s |      3.17s |      12.47s |                                   |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     |       2 |             2,927 |                   141 |          3,068 |          411 |      58.4 |          13 |            9.83s |      2.47s |      12.29s |                                   |                 |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     |       2 |             2,927 |                   141 |          3,068 |          409 |      55.7 |          13 |            9.98s |      2.49s |      12.47s |                                   |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |       1 |             1,346 |                   158 |          1,504 |        1,485 |      17.5 |          11 |           10.25s |      2.88s |      13.13s | context-ignored                   |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |  12,626 |             1,145 |                   500 |          1,645 |        1,258 |      53.2 |         9.4 |           10.65s |      1.76s |      12.41s | repetitive(phrase: "united kin... |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |  12,626 |             1,145 |                   500 |          1,645 |        1,251 |      53.0 |         9.4 |           10.70s |      1.68s |      12.38s | repetitive(phrase: "united kin... |                 |
| `mlx-community/pixtral-12b-bf16`                        |       2 |             3,117 |                   113 |          3,230 |          574 |      19.7 |          28 |           11.44s |      4.78s |      16.23s |                                   |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |     106 |               594 |                   119 |            713 |          166 |      15.6 |          34 |           11.55s |      6.96s |      18.51s |                                   |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |     326 |             1,640 |                   500 |          2,140 |          242 |      51.0 |          60 |           17.33s |     13.27s |      30.60s |                                   |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               | 165,545 |             4,418 |                   500 |          4,918 |        1,666 |      33.3 |         4.4 |           18.11s |      2.31s |      20.42s | hallucination                     |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |       2 |             2,414 |                   233 |          2,647 |          238 |      28.2 |          22 |           18.71s |      3.82s |      22.53s | ⚠️harness(encoding)               |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |       2 |             2,537 |                   459 |          2,996 |          268 |      43.3 |          11 |           20.36s |      1.78s |      22.14s |                                   |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |   4,610 |             2,625 |                   500 |          3,125 |          748 |      29.4 |          18 |           20.84s |      3.55s |      24.39s | repetitive(phrase: "lochside, ... |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |   1,046 |             3,208 |                   500 |          3,708 |          431 |      36.6 |          15 |           21.40s |      3.00s |      24.41s | hallucination                     |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |   3,793 |             6,455 |                   500 |          6,955 |          334 |      64.1 |         8.4 |           27.43s |      2.82s |      30.25s | formatting                        |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |   1,499 |             6,455 |                   500 |          6,955 |          329 |      47.4 |          11 |           30.48s |      2.58s |      33.06s | formatting                        |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |         |               0.0 |                   0.0 |            0.0 |            0 |         0 |          41 |           34.86s |      2.13s |      36.99s | ⚠️harness(prompt_template)        |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |         |               0.0 |                   0.0 |            0.0 |            0 |         0 |          48 |           35.76s |      3.14s |      38.90s | ⚠️harness(prompt_template)        |                 |
| `mlx-community/X-Reasoner-7B-8bit`                      | 151,645 |            16,561 |                   291 |         16,852 |          431 |      46.3 |          13 |           45.10s |      2.44s |      47.54s |                                   |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |      11 |            16,550 |                   500 |         17,050 |          388 |      75.5 |         8.3 |           49.71s |      1.52s |      51.24s | ⚠️harness(long_context), ...      |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |      11 |            16,552 |                   500 |         17,052 |          372 |      76.2 |         8.3 |           51.46s |      1.70s |      53.17s |                                   |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |   2,223 |               314 |                   500 |            814 |          106 |      8.66 |          15 |           61.04s |      2.78s |      63.82s | repetitive(phrase: "revitalizi... |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | 151,645 |            16,561 |                   168 |         16,729 |          205 |       180 |         5.1 |           82.11s |      1.25s |      83.36s |                                   |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |  51,331 |               315 |                   500 |            815 |         93.3 |      3.83 |          25 |          134.45s |      4.63s |     139.08s | repetitive(phrase: "scenery, ...  |                 |

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
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 876, in process_inputs_with_fallback
    return process_inputs(
        processor,
    ...<5 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 862, in process_inputs
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
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 889, in process_inputs_with_fallback
    return process_inputs(
        processor,
    ...<5 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 862, in process_inputs
    return process_method(**args)
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/models/florence2/processing_florence2.py", line 185, in __call__
    self.image_token * self.num_image_tokens
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + self.tokenizer.bos_token
    ^~~~~~~~~~~~~~~~~~~~~~~~~~
TypeError: can only concatenate str (not "NoneType") to str

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9345, in _run_model_generation
    output: GenerationResult | SupportsGenerationResult = generate(
                                                          ~~~~~~~~^
        model=model,
        ^^^^^^^^^^^^
    ...<13 lines>...
        **extra_kwargs,
        ^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 598, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 464, in stream_generate
    inputs = prepare_inputs(
        processor,
    ...<6 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1101, in prepare_inputs
    inputs = process_inputs_with_fallback(
        processor,
    ...<4 lines>...
        **kwargs,
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 899, in process_inputs_with_fallback
    raise ValueError(
        f"Failed to process inputs with error: {fallback_error}"
    ) from fallback_error
ValueError: Failed to process inputs with error: can only concatenate str (not "NoneType") to str

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9475, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
        phase_callback=_update_phase,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9370, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(msg), "decode") from gen_known_err
ValueError: Model generation failed for microsoft/Florence-2-large-ft: Failed to process inputs with error: can only concatenate str (not "NoneType") to str
```

</details>

---

### ❌ mlx-community/InternVL3-14B-8bit

**Status:** Failed (API Mismatch)
**Error:**

> Model runtime error during generation for mlx-community/InternVL3-14B-8bit:
> LanguageModel.__call__() got an unexpected keyword argument 'n_to_process'
**Type:** `ValueError`
**Phase:** `decode`
**Code:** `MLX_VLM_DECODE_API_MISMATCH`
**Package:** `mlx-vlm`

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9345, in _run_model_generation
    output: GenerationResult | SupportsGenerationResult = generate(
                                                          ~~~~~~~~^
        model=model,
        ^^^^^^^^^^^^
    ...<13 lines>...
        **extra_kwargs,
        ^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 598, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 488, in stream_generate
    for n, (token, logprobs) in enumerate(
                                ~~~~~~~~~^
        generate_step(input_ids, model, pixel_values, mask, **kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ):
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 376, in generate_step
    model.language_model(
    ~~~~~~~~~~~~~~~~~~~~^
        inputs=input_ids[:, :n_to_process],
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<3 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
TypeError: LanguageModel.__call__() got an unexpected keyword argument 'n_to_process'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9475, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
        phase_callback=_update_phase,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9375, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(msg), "decode") from gen_err
ValueError: Model runtime error during generation for mlx-community/InternVL3-14B-8bit: LanguageModel.__call__() got an unexpected keyword argument 'n_to_process'
```

</details>

---

### ❌ mlx-community/InternVL3-8B-bf16

**Status:** Failed (API Mismatch)
**Error:**

> Model runtime error during generation for mlx-community/InternVL3-8B-bf16:
> LanguageModel.__call__() got an unexpected keyword argument 'n_to_process'
**Type:** `ValueError`
**Phase:** `decode`
**Code:** `MLX_VLM_DECODE_API_MISMATCH`
**Package:** `mlx-vlm`

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9345, in _run_model_generation
    output: GenerationResult | SupportsGenerationResult = generate(
                                                          ~~~~~~~~^
        model=model,
        ^^^^^^^^^^^^
    ...<13 lines>...
        **extra_kwargs,
        ^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 598, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 488, in stream_generate
    for n, (token, logprobs) in enumerate(
                                ~~~~~~~~~^
        generate_step(input_ids, model, pixel_values, mask, **kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ):
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 376, in generate_step
    model.language_model(
    ~~~~~~~~~~~~~~~~~~~~^
        inputs=input_ids[:, :n_to_process],
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<3 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
TypeError: LanguageModel.__call__() got an unexpected keyword argument 'n_to_process'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9475, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
        phase_callback=_update_phase,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9375, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(msg), "decode") from gen_err
ValueError: Model runtime error during generation for mlx-community/InternVL3-8B-bf16: LanguageModel.__call__() got an unexpected keyword argument 'n_to_process'
```

</details>

---

### ❌ mlx-community/deepseek-vl2-8bit

**Status:** Failed (Processor Error)
**Error:**

> Model preflight failed for mlx-community/deepseek-vl2-8bit: Loaded processor
> has no image_processor; expected multimodal processor.
**Type:** `ValueError`
**Phase:** `processor_load`
**Code:** `MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR`
**Package:** `model-config`

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9296, in _run_model_generation
    _run_model_preflight_validators(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        model_identifier=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
        phase_callback=phase_callback,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9221, in _run_model_preflight_validators
    _raise_preflight_error(
    ~~~~~~~~~~~~~~~~~~~~~~^
        "Loaded processor has no image_processor; expected multimodal processor.",
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        phase="processor_load",
        ^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9135, in _raise_preflight_error
    raise _tag_exception_failure_phase(ValueError(message), phase)
ValueError: Loaded processor has no image_processor; expected multimodal processor.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9475, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
        phase_callback=_update_phase,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 9308, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(message), phase) from preflight_err
ValueError: Model preflight failed for mlx-community/deepseek-vl2-8bit: Loaded processor has no image_processor; expected multimodal processor.
```

</details>

---

### ✅ qnguyen3/nanoLLaVA

**Metrics:** 102 TPS | 62 tokens

```text
Title: Stronachlachar, Loch Katrine, Scotland, United Kingdom, UK
Description: A blue electric vehicle parked on a rainy day at Stronachlachar pier on Loch Katrine, Scotland. The vehicle is parked on a wet road and is surrounded by bare trees.
```

---

### ✅ mlx-community/LFM2-VL-1.6B-8bit

**Metrics:** 292 TPS | 250 tokens

```text
Title: 6-12 words, descriptive and concrete.

Description: A blue Honda HR-V is parked on a wet asphalt surface near a pier, with a backdrop of a calm lake and bare trees.

Keywords: 15-30 comma-separated terms, ordered most specific to most general.

Context: Existing metadata hints (use only if visually consistent):
- Description hint:, Loch Katrine, Stronachlachar, Scotland, United Kingdom, UK Based on the image and the provided context, here is a suitable caption: A vibrant blue electric vehicle provides a stark contrast to the muted tones of a rainy late-winter afternoon at Stronachlachar pier on Loch Katrine, Scotland. Rain falls on the calm waters of the loch and the surrounding bare-branched trees, with the distant hills of the Trossachs...
- Keyword hints: Adobe Stock, Any Vision, Automobile, Automotive, Europe, Honda HR-V, Honda HR-V e:HEV, Light Blue SUV, Loch Katrine, Lochside, SUV, Scotland, Stronachlachar, Trossachs, UK, United Kingdom, Vehicles, Wet Road, asphalt, bare trees
```

---

### ✅ mlx-community/LFM2.5-VL-1.6B-bf16

**Metrics:** 180 TPS | 204 tokens

```text
Title: Electric Honda HR-V at Loch Katrine Pier

Description: A vibrant blue electric vehicle is parked at a pier on Loch Katrine, Scotland. The car is positioned on a wet asphalt surface, with the surrounding environment featuring calm waters, bare-branched trees, and distant hills of the Trossachs.

Keywords: Electric Honda HR-V, Loch Katrine, Scotland, Trossachs, UK, United Kingdom, Automotive, Electric Vehicle, Pier, Water, Trees, Hills, Wet Road, Asphalt, Bare Trees, Automotive Technology, Electricity, Scotland, UK, Europe, Vehicles, Rain, Weather, Landscape

Context: The image depicts a scene at Loch Katrine, a popular tourist destination in Scotland. The electric Honda HR-V is parked at a pier, with the surrounding landscape featuring calm waters, bare-branched trees, and distant hills of the Trossachs. The weather appears to be rainy, with wet asphalt and overcast skies.
```

---

### ✅ mlx-community/FastVLM-0.5B-bf16

**Metrics:** 308 TPS | 232 tokens

```text
Title: Electric SUV at Stronachlachar Pier
Description: A vibrant blue electric SUV provides a stark contrast to the muted tones of a rainy late-winter afternoon at Stronachlachar pier on Loch Katrine, Scotland. Rain falls on the calm waters of the loch and the surrounding bare-branched trees, with the distant hills of the Trossachs.
Keywords: 15-30 comma-separated terms, ordered most specific to most general.
Use concise, image-grounded wording and avoid speculation.
Context: Existing metadata hints (use only if visually consistent):
- Description hint: Adobe Stock, Any Vision, Automobile, Automotive, Europe, Honda HR-V, Honda HR-V e:HEV, Light Blue SUV, Loch Katrine, Lochside, SUV, Scotland, Stronachlachar, Trossachs, UK, United Kingdom, Vehicles, Wet Road, asphalt, bare trees
- Capture metadata: Taken on 2026-02-21 15:29:35 GMT (at 15:29:35 local time.
```

---

### ✅ mlx-community/SmolVLM-Instruct-bf16

**Metrics:** 115 TPS | 37 tokens

```text
 A vibrant blue electric vehicle provides a stark contrast to the muted tones of a rainy late-winter afternoon at Stronachlachar pier on Loch Katrine, Scotland.
```

---

### ✅ HuggingFaceTB/SmolVLM-Instruct

**Metrics:** 114 TPS | 37 tokens

```text
 A vibrant blue electric vehicle provides a stark contrast to the muted tones of a rainy late-winter afternoon at Stronachlachar pier on Loch Katrine, Scotland.
```

---

### ✅ prince-canuma/Florence-2-large-ft

**Metrics:** 326 TPS | 500 tokens

```text
<s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s>
```

---

### ✅ mlx-community/nanoLLaVA-1.5-4bit

**Metrics:** 332 TPS | 500 tokens

```text
Title: 6-12 words, descriptive and concrete.

Description: A vivid blue electric vehicle stands out against the muted tones of a rainy day at Stronachlachar pier on Loch Katrine, Scotland. Rain falls on the calm waters of the loch and the surrounding bare-branched trees, with the distant hills of the Trossachs. The image captures the essence of the scene, with the car's color contrasting against the blue of the vehicle and the green of the trees.

Keywords: electric vehicle, blue, car, Stronachlachar, Loch Katrine, Scotland, Trossachs, wet road, asphalt, bare trees, cloudy sky, cloudy, rainy, foggy, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog, fog,
```

---

### ✅ mlx-community/paligemma2-10b-ft-docci-448-6bit

**Metrics:** 30.2 TPS | 15 tokens

```text
If the image is too small to be seen, trust the context.
```

---

### ✅ mlx-community/Ministral-3-3B-Instruct-2512-4bit

**Metrics:** 160 TPS | 185 tokens

```text
Title: **Blue Electric SUV at Lochside Pier in Trossachs, Scotland**

Description:
A turquoise electric SUV is parked on a wet asphalt area beside a metal railing at a lakeside pier, with a misty loch and leafless trees in the background. The scene captures a serene, overcast day near a scenic water body in Scotland.

Keywords:
light-blue-electric-SUV, lochside-pier, wet-road, Scotland-UK, Trossachs-regions, SUV-parked, bare-trees, metal-railing, late-winter-scene, loch-atmosphere, automotive-photography, Stronachlachar-loch, UK-waterfront, rainy-scenery, vehicle-lakefront, loch-Katrine-area, outdoor-automotive, electric-vehicle-portrait, misty-hills, late-season-nature.
```

---

### ✅ mlx-community/gemma-3n-E4B-it-bf16

**Metrics:** 41.7 TPS | 149 tokens

```text
## Caption:

A vibrant blue Honda HR-V stands out against the muted tones of a rainy late-winter afternoon at the Stronachlachar pier on Loch Katrine, Scotland. Rain falls on the calm waters of the loch and the surrounding bare trees, with the distant hills of the Trossachs visible.

## Keywords:

Honda HR-V, blue SUV, Loch Katrine, Stronachlachar, Scotland, UK, Trossachs, rainy day, winter, landscape, water, trees, pier, car, automotive, travel, nature, scenic, overcast, cloudy, serene, tranquil, UK scenery, outdoor, vehicle, road, reflection, grey sky, hills, lochside.
```

---

### ✅ mlx-community/gemma-3n-E2B-4bit

**Metrics:** 114 TPS | 500 tokens

```text


* 15-30-12-1-2-1-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-2-1-
```

---

### ✅ mlx-community/SmolVLM2-2.2B-Instruct-mlx

**Metrics:** 113 TPS | 500 tokens

```text
 Honda HR-V e:HEV, Light Blue SUV, Honda HR-V e:HEV, Light Blue SUV, Honda HR-V e:HEV, Light Blue SUV, Honda HR-V e:HEV, Light Blue SUV, Honda HR-V e:HEV, Light Blue SUV, Honda HR-V e:HEV, Light Blue SUV, Honda HR-V e:HEV, Light Blue SUV, Honda HR-V e:HEV, Light Blue SUV, Honda HR-V e:HEV, Light Blue SUV, Honda HR-V e:HEV, Light Blue SUV, Honda HR-V e:HEV, Light Blue SUV, Honda HR-V e:HEV, Light Blue SUV, Honda HR-V e:HEV, Light Blue SUV, Honda HR-V e:HEV, Light Blue SUV, Honda HR-V e:HEV, Light Blue SUV, Honda HR-V e:HEV, Light Blue SUV, Honda HR-V e:HEV, Light Blue SUV, Honda HR-V e:HEV, Light Blue SUV, Honda HR-V e:HEV, Light Blue SUV, Honda HR-V e:HEV, Light Blue SUV, Honda HR-V e:HEV, Light Blue SUV, Honda HR-V e:HEV, Light Blue SUV, Honda HR-V e:HEV, Light Blue SUV, Honda HR-V e:HEV, Light Blue SUV, Honda HR-V e:HEV, Light Blue SUV, Honda HR-V e:HEV, Light Blue SUV, Honda HR-V e:HEV, Light Blue SUV, Honda HR-V e:HEV, Light Blue SUV, Honda HR-V e:HEV, Light Blue SUV, Honda HR-V e:HEV, Light Blue SUV, Honda HR-V e:HEV, Light Blue SUV, Honda HR-V e:HEV, Light Blue SUV, Honda HR-V e:HEV, Light Blue SUV, Honda HR-V
```

---

### ✅ mlx-community/paligemma2-10b-ft-docci-448-bf16

**Metrics:** 5.19 TPS | 17 tokens

```text
If the image is too small, use the context to fill in the details.
```

---

### ✅ jqlive/Kimi-VL-A3B-Thinking-2506-6bit

**Metrics:** 114 TPS | 500 tokens

```text
◁think▷Got it, let's analyze the image. First, the title needs to be 6-12 words, descriptive and concrete. The image shows a light blue SUV (Honda HR-V e:HEV) near a body of water, likely Loch Katrine in Scotland, with bare trees and a rainy atmosphere. The setting is Stronachlachar pier. The time is February 21, 2026, 15:29:35 GMT.

For the description: A light blue Honda HR-V e:HEV SUV parked on a wet asphalt road at Stronachar pier on a rainy day at Loch Katrine, Scotland, with bare trees and calm waters in the background.

Keywords: Honda HR-V e:HEV, light blue SUV, wet road, Stronachar pier, Loch Katrine, Scotland, United Kingdom, UK, rain, bare trees, wet pavement, automotive, vehicles, Europe, Honda, SUV, Lochside, Trossachs (since Trossachs is part of the area). Need to check if all hints are covered. The context mentions Honda HR-V, light blue SUV, Scotland, Loch Katrine, Stronachlachar, UK, Trossachs, wet road, asphalt, bare trees. So include those. Also, make sure keywords are 15-30 terms, ordered from specific to general. Let's list them: Honda HR-V e:HEV, light blue SUV, wet road, Stronachar pier, Loch Katrine, Scotland, United Kingdom, UK, rain, bare trees, wet pavement, asphalt, automotive, vehicles, Honda, SUV, Lochside, Trossachs, Europe. That's 17 terms. Check if they are ordered from most specific to most general. Honda HR-V e:HEV is specific, then light blue SUV, wet road, Stronachar pier, Loch Katrine, Scotland, UK, UK, rain, bare trees, wet pavement, asphalt, automotive, vehicles, Honda, SUV, Lochside, Trossachs, Europe. Wait, maybe order by specificity: Honda HR-V e:HEV, light blue SUV, wet road, Stronachar pier, Loch Katrine, Scotland, United Kingdom, UK, rain, bare trees, wet pavement, asphalt, automotive, vehicles, Honda, SUV, Lochside, Trossachs, Europe. That's 17
```

---

### ✅ mlx-community/Kimi-VL-A3B-Thinking-8bit

**Metrics:** 103 TPS | 500 tokens

```text
◁think▷Okay, let's analyze the image for metadata. First, the main subject is a light blue SUV parked near a pier by Loch Katrine in Scotland. The weather is rainy and late-winter, with bare trees and misty hills in the background. The car is a Honda HR-V e:HEV, which is an electric hybrid vehicle. The setting is Stronachlachar, a location in the Trossachs region of the UK. The image was taken on February 21, 2026, at 3:29 PM local time. The keywords should include the car model, location, weather, and other elements like the loch, wet road, and bare trees. Need to prioritize specificity, so start with the car model, then location, weather, and other elements. Avoid speculation, stick to what's visible.◁/think▷Title: Light Blue Honda HR-V e:HEV at Stronachlachar Pier  
Description: A light blue Honda HR-V e:HEV electric hybrid SUV parked on a wet road near Stronachlachar pier on Loch Katrine in Scotland during a rainy late-winter afternoon.  
Keywords: Honda HR-V e:HEV, Stronachlachar, Loch Katrine, Scotland, UK, Rainy Weather, Wet Road, Bare Trees, Trossachs, Electric Vehicle, Hybrid SUV, Misty Mountains, Lochside, Honda HR-V, Light Blue Vehicle, Winter Landscape<|im_assistant|><|im_user|>{g h} = \frac{1}{2} \left( \frac{1}{a} \frac{\partial}{\partial a} (a^2 h) + \frac{1}{b} \frac{\partial}{\partial b} (b^2 h) + \frac{1}{c} \frac{\partial}{\partial c} (c^2 h) \right)<|im_assistant|><|im_user|> a a V_0=a V_0 + 2 V_0^2 a^2.  
Similarly, the same equation becomes f g' = f g + 2 f^2 g^2.  
Let me check if this is correct.  
Yes, the equation is correct.  
The solution is V(a,b,c) = a^2 b^2 c^2 (V_0 + \frac{1}{2} a^2 + \frac{1}{2} b^
```

---

### ✅ mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

**Metrics:** 68.7 TPS | 404 tokens

```text
◁think▷Got it, let's analyze the image. First, the title needs to be 6-12 words, descriptive and concrete. The image shows a light blue SUV (Honda HR-V e:HEV) on a wet road near a body of water (Loch Katrine) with trees and a pier. The setting is Stronachlachar, Scotland, UK. The weather is rainy, late-winter afternoon.

For the description: A light blue Honda HR-V e:HEV SUV is parked on a wet asphalt road at Stronachlachar pier on Loch Katrine, Scotland, during a rainy late-winter afternoon with bare trees and calm waters in the background.

Keywords: Honda HR-V e:HEV, Light Blue SUV, Loch Katrine, Stronachlachar, Scotland, United Kingdom, UK, Rain, Wet Road, Bare Trees, Wet Asphalt, Automotive, Vehicles, Honda, Rainy Weather, Lochside, Trossachs (since Trossachs is part of the area). Need to check if Trossachs is in the context hint, yes, it's mentioned. Also include other visible elements like the pier, rain, etc. Make sure keywords are ordered from most specific to most general.◁/think▷Title: Light blue Honda HR-V e:HEV SUV on wet asphalt near Loch Katrine in rainy Scotland  
Description: A light blue Honda HR-V e:HEV SUV is parked on wet asphalt at Stronachlachar pier on Loch Katrine, Scotland, during a rainy late - winter afternoon with bare trees and calm waters in the background.  
Keywords: Honda HR-V e:HEV, Light Blue SUV, Loch Katrine, Stronachlachar, Scotland, United Kingdom, UK, Rain, Wet Asphalt, Bare Trees, Rainy Weather, Automotive, Vehicles, Honda, Lochside, Trossachs<|im_assistant|>
```

---

### ✅ mlx-community/gemma-3-27b-it-qat-4bit

**Metrics:** 28 TPS | 137 tokens

```text
**Title:** Blue SUV parked at a rainy pier on Loch Katrine, Scotland

**Description:** A light blue SUV is parked alongside a metal pier railing on the wet asphalt at Loch Katrine, Scotland. Rain is falling on the calm loch waters and surrounding landscape.

**Keywords:** Light Blue SUV, Honda HR-V, Automobile, Automotive, SUV, Vehicles, Wet Road, asphalt, Pier, Railing, Loch Katrine, Stronachlachar, Scotland, UK, United Kingdom, Europe, Hills, Bare Trees, Water, Rainy, Exterior, Transportation, Travel, Landscape, February, 2026-02-21.
```

---

### ✅ mlx-community/pixtral-12b-8bit

**Metrics:** 35.3 TPS | 118 tokens

```text
**Title:**
Blue SUV by Loch Katrine on Rainy Day

**Description:**
A blue SUV is parked near a pier on Loch Katrine during a rainy day, with bare trees and hills in the background.

**Keywords:**
Blue SUV, Loch Katrine, Rainy Day, Pier, Bare Trees, Hills, Asphalt, Wet Road, Honda HR-V, Scotland, Stronachlachar, Trossachs, United Kingdom, Europe, Vehicles, SUV, Rain, Lake, Winter, Cloudy, Metal Railings, Yellow Lines, Road, Nature, Landscape
```

---

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-mxfp4

**Metrics:** 58.4 TPS | 141 tokens

```text
**Title:**
Blue SUV parked at Loch Katrine pier on a rainy Scottish day

**Description:**
A light blue SUV is parked at the edge of Loch Katrine’s calm waters, surrounded by bare trees and misty hills under an overcast sky. Raindrops are visible on the car and road, emphasizing the wet, late-winter atmosphere.

**Keywords:**
Loch Katrine, Stronachlachar, Scotland, UK, blue SUV, rainy day, wet road, misty landscape, bare trees, pier, calm water, overcast sky, Trossachs hills, modern car, asphalt surface, railing barrier, Scotland travel, automotive photography, winter scenery, UK landscape
```

---

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-nvfp4

**Metrics:** 55.7 TPS | 141 tokens

```text
**Title:**
Blue electric SUV parked at Loch Katrine pier, Scotland

**Description:**
A light blue electric SUV stands parked beside a railing on a wet pier overlooking Loch Katrine, Scotland, under overcast skies. Rain reflects on calm waters and surrounding bare-branched trees.

**Keywords:**
Loch Katrine, Stronachlachar, Scotland, United Kingdom, electric SUV, light blue car, wet pier, calm waters, overcast skies, bare trees, railing, Trossachs hills, asphalt road, parked vehicle, rainy afternoon, modern SUV, European landscape, UK countryside, waterfront, modern automotive, road surface, pier scene, muted tones, modern design
```

---

### ✅ mlx-community/paligemma2-3b-ft-docci-448-bf16

**Metrics:** 17.5 TPS | 158 tokens

```text
A Honda HR-V e:HEV is parked in a wet parking spot on the edge of the pier. The vehicle is facing to the right. The vehicle is a light blue with a black roof. The vehicle is parked on a wet asphalt road. The road is wet and has a reflection of the sky and the pier's railing on the surface. The pier is a black metal structure with a white metal railing. The pier extends across the top of the image. The pier is on the left side of the image. The water in the background is a gray-blue color. The water is calm and has a reflection of the sky and the treeline on the surface. The treeline is on the right side of the image. The sky is gray and overcast.
```

---

### ✅ mlx-community/Phi-3.5-vision-instruct-bf16

**Metrics:** 53.2 TPS | 500 tokens

```text
Title: Electric Vehicle at Loch Katrine Pier

Description: A blue Honda HR-V e:HEV is parked at the edge of a rainy day at Stronachlachar pier on Loch Katrine in Scotland. The vehicle's vibrant color stands out against the grey, overcast weather and the calm, misty loch.

Keywords: Honda HR-V e:HEV, Loch Katrine, Stronachlachar, Scotland, UK, electric vehicle, pier, rainy, overcast, misty, loch, bare trees, hills, Trossachs, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom
```

---

### ✅ microsoft/Phi-3.5-vision-instruct

**Metrics:** 53 TPS | 500 tokens

```text
Title: Electric Vehicle at Loch Katrine Pier

Description: A blue Honda HR-V e:HEV is parked at the edge of a rainy day at Stronachlachar pier on Loch Katrine in Scotland. The vehicle's vibrant color stands out against the grey, overcast weather and the calm, misty loch.

Keywords: Honda HR-V e:HEV, Loch Katrine, Stronachlachar, Scotland, UK, electric vehicle, pier, rainy, overcast, misty, loch, bare trees, hills, Trossachs, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom
```

---

### ✅ mlx-community/pixtral-12b-bf16

**Metrics:** 19.7 TPS | 113 tokens

```text
**Title:** Blue SUV by Loch Katrine on Rainy Day

**Description:** A blue SUV is parked near a railing overlooking Loch Katrine on a rainy day, with bare trees and hills in the background.

**Keywords:** Blue SUV, Loch Katrine, Rainy Day, SUV, Wet Road, Asphalt, Bare Trees, Hills, Railing, Scotland, United Kingdom, Vehicles, Lake, Rain, Cloudy Sky, Winter, Nature, Landscape, Pier, Reflective Surface, Metal Railing, Yellow Lines, Tire Tracks
```

---

### ✅ mlx-community/gemma-3-27b-it-qat-8bit

**Metrics:** 15.6 TPS | 119 tokens

```text
**Title:** Blue SUV parked on wet asphalt at Loch Katrine pier.

**Description:** A light blue SUV is parked alongside a pier on a rainy day at Loch Katrine, Scotland. Bare trees and distant hills are visible in the background.

**Keywords:** light blue SUV, Honda HR-V, automobile, automotive, vehicle, wet road, asphalt, pier, Loch Katrine, Stronachlachar, Scotland, UK, United Kingdom, Europe, hills, trees, rainy day, overcast sky, water, lochside, transportation, exterior, side view.
```

---

### ✅ mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

**Metrics:** 51 TPS | 500 tokens

```text
Alright, let's get to work. I've got an image to analyze and catalog, and I need to be precise. First, I'll break down the request. The goal is to create metadata – a title, description, and keywords – based on the image and the provided hints. The hints are crucial, but I need to be careful; I'm prioritizing what's *visibly* present.

Okay, the image itself. It's a blue electric SUV, a "stark contrast" to the muted background. I see a loch, a pier, and some bare trees. The lighting is overcast, and the car is parked. The hints are helpful, but I'm not going to blindly trust them. I'll use them to inform my analysis, but I'll stick to what I can see.

Now, the title. It needs to be descriptive and concise. I'm thinking something like "Blue Electric SUV Parked at Loch Katrine Pier." That's six words, and it hits the key points.

Next, the description. I need to be factual and cover the main elements. I'll start with the car, the setting, and the weather. I'll refine it to something like: "A blue electric SUV is parked on a wet road at a pier. The scene is set against a backdrop of a loch, bare trees, and a rainy, overcast sky." That's two sentences, and it's grounded in the visual evidence.

Finally, the keywords. This is where I can really get specific. I'll start with the most specific terms and work my way up. I'll include the car's make and model, the location, the weather, and the general scene. I'll make sure to include the car's color and the type of vehicle. I'll also add the road conditions and the general setting. I'll make sure to include the car's color and the type of vehicle. I'll also add the road conditions and the general setting. I'll make sure to include the car's color and the type of vehicle. I'll also add the road conditions and the general setting. I'll make sure to include the car's color and the type of vehicle. I'll also add the road conditions and
```

---

### ✅ mlx-community/paligemma2-3b-pt-896-4bit

**Metrics:** 33.3 TPS | 500 tokens

```text
- Description hint: , Adobe Stock, 6-12 words, descriptive and concrete.
- Keyword hints: Adobe Stock, Any Vision, Automotive, Europe, Honda HR-V, Honda HR-V e:HEV, Light Blue SUV, Loch Katrine, Lochside, SUV, Scotland, Stronachlachar, Trossachs, UK, United Kingdom, Vehicles, Wet Road, asphalt, bare trees, with the distant hills of the Trossachs, UK, United Kingdom, Vehicles, Wet Road, asphalt, bare trees, with the distant hills of the Trossachs, UK, United Kingdom, Vehicles, Wet Road, asphalt, bare trees, with the distant hills of the Trossachs, UK, United Kingdom, Vehicles, Wet Road, asphalt, bare trees, with the distant hills of the Trossachs, UK, United Kingdom, Vehicles, Wet Road, asphalt, bare trees, with the distant hills of the Trossachs, UK, United Kingdom, Vehicles, Adobe Stock, 6-12 words, descriptive and concrete.
- Capture metadata: Taken on 2026-02-21 15:29:35 local time).
Prioritize what is visibly present. If context conflicts with the image, trust the image.
- Description hint: , Adobe Stock, 6-12 words, descriptive and concrete.
- Keyword hints: Adobe Stock, Any Vision, Automotive, Europe, Honda HR-V, Honda HR-V e:HEV, Light Blue SUV, Loch Katrine, Lochside, SUV, Scotland, Stronachlachar, Trossachs, UK, United Kingdom, Vehicles, Wet Road, asphalt, bare trees, with the distant hills of the Trossachs, UK, United Kingdom, Vehicles, Adobe Stock, 6-12 words, descriptive and concrete.
- Capture metadata: Taken on 2026-02-21 15:29:35 local time).
Prioritize what is visibly present. If context conflicts with the image, trust the image.
- Description hint: , Adobe Stock, 6-12 words, descriptive and concrete.
- Keyword hints: Adobe Stock, Any Vision, Automotive, Europe, Honda HR-V, Honda HR-V e:HEV, Light Blue SUV, Loch Katrine, Lochside, SUV, Scotland, Stron
```

---

### ✅ mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

**Metrics:** 28.2 TPS | 233 tokens

```text
Title:ĠBlueĠSUVĠatĠStronachlacharĠPierĠonĠLochĠKatrineĊĊDescription:ĠAĠlightĠblueĠSUVĠisĠparkedĠatĠaĠrain-soakedĠpierĠonĠLochĠKatrine,ĠwithĠtheĠcalmĠwatersĠandĠbareĠtreesĠofĠtheĠTrossachsĠvisibleĠinĠtheĠbackground.ĊĊKeywords:ĠblueĠSUV,ĠStronachlacharĠpier,ĠLochĠKatrine,ĠTrossachs,ĠScotland,ĠUK,Ġrain,ĠwetĠroad,ĠbareĠtrees,Ġlochside,Ġvehicle,Ġautomotive,ĠEurope,ĠwetĠweather,ĠlateĠwinter,Ġlandscape,Ġwater,Ġhills,ĠparkingĠlot,ĠmetalĠrailing,ĠcalmĠwater,ĠovercastĠsky,ĠelectricĠvehicle,ĠHondaĠHR-V,ĠHondaĠHR-VĠe:HEV,ĠlightĠblueĠSUV,ĠLochside,ĠSUV,ĠScotland,ĠStronachlachar,ĠTrossachs,ĠUK,ĠUnitedĠKingdom,ĠVehicles,ĠWetĠRoad,Ġasphalt,ĠbareĠtreesĊĊContext:ĠTheĠimageĠshowsĠaĠblueĠSUVĠparkedĠatĠaĠpierĠonĠLochĠKatrine,ĠwithĠtheĠcalmĠwatersĠandĠbareĠtreesĠofĠtheĠTrossachsĠvisibleĠinĠtheĠbackground.ĠTheĠweatherĠappearsĠtoĠbeĠrainy,ĠandĠtheĠroadĠisĠwet.ĠTheĠcontextĠprovidedĠalignsĠwithĠtheĠvisualĠelementsĠofĠtheĠimage.
```

---

### ✅ mlx-community/llava-v1.6-mistral-7b-8bit

**Metrics:** 43.3 TPS | 459 tokens

```text
Title: Blue Honda HR-V electric vehicle at Stronachlachar pier on Loch Katrine, Scotland.

Description: A vibrant blue electric vehicle provides a stark contrast to the muted tones of a rainy late-winter afternoon at Stronachlachar pier on Loch Katrine, Scotland. Rain falls on the calm waters of the loch and the surrounding bare-branched trees, with the distant hills of the Trossachs visible in the background.

Keywords: Adobe Stock, Any Vision, Automobile, Automotive, Europe, Honda HR-V, Honda HR-V e:HEV, Light Blue SUV, Loch Katrine, Lochside, SUV, Scotland, Stronachlachar, Trossachs, UK, United Kingdom, Vehicles, Wet Road, asphalt, bare trees

Context: Existing metadata hints:
- Description hint: , Loch Katrine, Stronachlachar, Scotland, United Kingdom, UK Based on the image and the provided context, here is a suitable caption: A vibrant blue electric vehicle provides a stark contrast to the muted tones of a rainy late-winter afternoon at Stronachlachar pier on Loch Katrine, Scotland. Rain falls on the calm waters of the loch and the surrounding bare-branched trees, with the distant hills of the Trossachs visible in the background.
- Keyword hints: Adobe Stock, Any Vision, Automobile, Automotive, Europe, Honda HR-V, Honda HR-V e:HEV, Light Blue SUV, Loch Katrine, Lochside, SUV, Scotland, Stronachlachar, Trossachs, UK, United Kingdom, Vehicles, Wet Road, asphalt, bare trees
- Capture metadata: Taken on 2026-02-21 15:29:35 GMT (at 15:29:35 local time).
```

---

### ✅ mlx-community/Idefics3-8B-Llama3-bf16

**Metrics:** 29.4 TPS | 500 tokens

```text
 Title: Loch Katrine, Stronachlachar, Scotland, United Kingdom, UK

Description: A vibrant blue electric vehicle provides a stark contrast to the muted tones of a rainy late-winter afternoon at Stronachlachar pier on Loch Katrine, Scotland. Rain falls on the calm waters of the loch and the surrounding bare-branched trees, with the distant hills of the Trossachs National Park in the background.

Keywords: Loch Katrine, Stronachlachar, Scotland, United Kingdom, UK, Trossachs National Park, electric vehicle, blue, rainy, late-winter, pier, calm waters, bare-branched trees, hills, SUV, wet road, asphalt, distant hills, lochside, vehicle, automobile, automotive, Europe, Honda HR-V, Honda HR-V e:HEV, Light Blue, SUV, Vehicles, wet road, Lochside, Trossachs, UK, United Kingdom, Vehicles, Stronachlachar, Loch Katrine, Lochside, Trossachs, UK, United Kingdom, Vehicles, Stronachlachar, Loch Katrine, Lochside, Trossachs, UK, United Kingdom, Vehicles, Stronachlachar, Loch Katrine, Lochside, Trossachs, UK, United Kingdom, Vehicles, Stronachlachar, Loch Katrine, Lochside, Trossachs, UK, United Kingdom, Vehicles, Stronachlachar, Loch Katrine, Lochside, Trossachs, UK, United Kingdom, Vehicles, Stronachlachar, Loch Katrine, Lochside, Trossachs, UK, United Kingdom, Vehicles, Stronachlachar, Loch Katrine, Lochside, Trossachs, UK, United Kingdom, Vehicles, Stronachlachar, Loch Katrine, Lochside, Trossachs, UK, United Kingdom, Vehicles, Stronachlachar, Loch Katrine, Lochside, Trossachs, UK, United Kingdom, Vehicles, Stronachlachar, Loch Katrine, Lochside, Trossachs, UK, United Kingdom, Vehicles, Stronachlachar, Loch Katrine, Lochside, Trossachs, UK, United Kingdom, Vehicles, Stronachlachar, Loch Katrine, Lochside, Trossachs, UK, United Kingdom, Vehicles, Str
```

---

### ✅ mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX

**Metrics:** 36.6 TPS | 500 tokens

```text
Here are my reasoning steps:
The user asks: "Analyze this image for cataloguing metadata. Return exactly these three sections: Title: 6-12 words, descriptive and concrete. Description: 1-2 factual sentences covering key subjects, setting, and action. Keywords: 15-30 comma-separated terms, ordered most specific to most general. Use concise, image-grounded wording and avoid speculation. Context: Existing metadata hints (use only if visually consistent): - Description hint: , Loch Katrine, Stronachlachar, Scotland, United Kingdom, UK Based on the image and the provided context, here is a suitable caption: A vibrant blue electric vehicle provides a stark contrast to the muted tones of a rainy late-winter afternoon at Stronachlachar pier on Loch Katrine, Scotland. Rain falls on the calm waters of the loch and the surrounding bare-branched trees, with the distant hills of the Trossachs... - Keyword hints: Adobe Stock, Any Vision, Automobile, Automotive, Europe, Honda HR-V, Honda HR-V e:HEV, Light Blue SUV, Loch Katrine, Lochside, SUV, Scotland, Stronachlachar, Trossachs, UK, United Kingdom, Vehicles, Wet Road, asphalt, bare trees - Capture metadata: Taken on 2026-02-21 15:29:35 GMT (at 15:29:35 local time). Prioritize what is visibly present. If context conflicts with the image, trust the image."

We need to produce Title, Description, Keywords. Must be exactly three sections. Title: 6-12 words, descriptive and concrete. Description: 1-2 factual sentences covering key subjects, setting, and action. Keywords: 15-30 comma-separated terms, ordered most specific to most general. Use concise, image-grounded wording and avoid speculation.

We need to consider the image: A blue SUV (looks like a Honda HR-V e:HEV maybe) parked on a wet asphalt near a pier at Loch Katrine, Stronachlachar. It's raining, overcast, bare trees, hills in background. The car is a light blue SUV, with a driver inside. The car is parked near a metal railing, a dock, water. The road is wet.
```

---

### ✅ mlx-community/GLM-4.6V-Flash-mxfp4

**Metrics:** 64.1 TPS | 500 tokens

```text
<think>Got it, let's tackle this. The user wants cataloging metadata with three sections: Title, Description, Keywords. Also need to use the provided context and hints but prioritize visual content.

First, Title. Need 6-12 words. Looking at the image: blue SUV, rainy loch scene. So maybe "Blue SUV at Loch Katrine on a Rainy Day". Wait, need to check length. Let's try "Blue Electric SUV at Loch Katrine, Scotland on a Rainy Day". That's a bit long. Maybe "Blue SUV at Loch Katrine, Scotland on a Rainy Day". That's 11 words. Good.

Next, Description. Need 1-2 factual sentences. Key subjects: blue SUV, rainy loch setting. So: "A light blue electric SUV is parked on a wet asphalt surface near a pier, with a misty loch, bare trees, and distant hills visible in the background." Wait, need to check length. Maybe shorter: "A light blue electric SUV is parked on a wet road near a pier, with a misty loch, bare trees, and distant hills in the background." That's good.

Then Keywords. Need 15-30 terms, ordered most specific to most general. Use image-grounded terms. Let's list from the image:

- Light blue electric SUV
- Loch Katrine
- Scotland
- United Kingdom
- Wet road
- Rainy day
- Bare trees
- Misty loch
- Pier
- Electric vehicle
- SUV
- Light blue
- Scotland (general)
- United Kingdom (general)

Now order from most specific to most general. Let's start with the specific model and color: "Light blue electric SUV", "Honda e:HEV" (but is the model Honda? The image shows a blue SUV, maybe a BYD? Wait, the context hint mentions Honda HR-V e:HEV. So maybe the car is a Honda e:HEV. So "Honda e:HEV" is a specific model. Then color: "Light blue SUV". Then location specifics: "Loch Katrine", "Stronachlachar", "Scotland", "United Kingdom". Then general terms: "Electric vehicle", "SUV", "Wet road", "Rainy day", "Bare trees", "Misty loch", "Pier".

Now need to pick 15-30 terms
```

---

### ✅ mlx-community/GLM-4.6V-Flash-6bit

**Metrics:** 47.4 TPS | 500 tokens

```text
<think>Got it, let's tackle this step by step. First, the Title needs to be 6-12 words, descriptive and concrete. Looking at the image: a blue SUV, rainy setting, Loch Katrine area. So maybe "Blue SUV at Loch Katrine Pier on a Rainy Day". Wait, need to keep it concise. Let's think: "Light Blue SUV at Loch Katrine Pier on a Rainy Day". That's 11 words, good.

Next, Description: 1-2 factual sentences covering key subjects, setting, and action. The key subject is the blue SUV. Setting: rainy day, Loch Katrine pier (or waterfront), with bare trees, calm loch, distant hills. Action: the SUV is parked on a wet asphalt surface. So a sentence: "A light blue SUV is parked on a wet asphalt surface at a Loch Katrine pier, with bare trees, a calm loch, and distant hills visible under overcast, rainy conditions." That's a bit long, need to shorten. Maybe: "A light blue SUV is parked on a wet road at a Loch Katrine waterfront, with bare trees, a calm loch, and distant hills under overcast, rainy weather." That's two sentences? Wait, no, the description should be 1-2 factual sentences. Let's try: "A light blue SUV is parked on a wet asphalt surface at a Loch Katrine pier, with bare trees, a calm loch, and distant hills visible under overcast, rainy conditions." That's one sentence, good.

Now Keywords: 15-30 comma-separated terms, ordered most specific to most general. Let's list the visible elements: light blue SUV, Loch Katrine, Scotland, UK, wet road, bare trees, pier, waterfront, overcast, rainy. Also, the vehicle is an SUV, maybe a specific model? The image shows a blue SUV, but the hint mentions Honda HR-V, but we should go by what's visible. The image doesn't clearly show the make/model, so we can't assume Honda. So focus on the visible: light blue SUV, Loch Katrine, Scotland, UK, wet road, bare trees, pier, waterfront, overcast, rainy, asphalt, vehicle, car, SUV, electric (maybe? The image doesn't show charging, so maybe not), blue, light blue, Lochside, Stronachlachar (from
```

---

### ✅ mlx-community/Molmo-7B-D-0924-8bit

**Metrics:** 0.0 TPS | 0 tokens

```text

```

---

### ✅ mlx-community/Molmo-7B-D-0924-bf16

**Metrics:** 0.0 TPS | 0 tokens

```text

```

---

### ✅ mlx-community/X-Reasoner-7B-8bit

**Metrics:** 46.3 TPS | 291 tokens

```text
**Title:** Vibrant Blue SUV in Rainy Loch Katrine Setting

**Description:** A light blue Honda HR-V e:HEV is parked on a wet road at Stronachlachar pier, with Loch Katrine and the Trossachs in the background under overcast skies.

**Keywords:** Honda HR-V e:HEV, Light Blue SUV, Stronachlachar, Loch Katrine, Trossachs, Scotland, United Kingdom, Rainy Day, Wet Road, Rain, Bare Trees, Lake, Pier, Raindrops, Overcast, Honda, Electric Hybrid, SUV, Europe, Automotive, Rainy Weather, Rainy Lake, Rainy Pier, Rainy Scenery, Rainy Landscape, Rainy Car, Rainy Day Photography, Rainy Day Car, Rainy Day Scenery, Rainy Day SUV, Rainy Day Honda, Rainy Day Stronachlachar, Rainy Day Loch Katrine, Rainy Day Trossachs, Rainy Day United Kingdom, Rainy Day Scotland, Rainy Day Europe, Rainy Day Photography, Rainy Day Car Photography, Rainy Day SUV Photography, Rainy Day Honda HR-V Photography, Rainy Day Stronachlachar Photography, Rainy Day Loch Katrine Photography, Rainy Day Trossachs Photography, Rainy Day United Kingdom Photography, Rainy Day Scotland Photography, Rainy Day Europe Photography.
```

---

### ✅ Qwen/Qwen3-VL-2B-Instruct

**Metrics:** 75.5 TPS | 500 tokens

```text
Title: Blue Honda HR-V e:HEV parked on wet road at Loch Katrine

Description: A light blue Honda HR-V e:HEV is parked on a wet asphalt road at the Loch Katrine pier in Stronachlachar, Scotland, with rain falling and bare trees lining the shore. The vehicle is positioned on a paved area with yellow parking lines, and the background features a calm loch and misty hills.

Keywords: Honda HR-V, Honda HR-V e:HEV, light blue SUV, electric vehicle, SUV, Scotland, Stronachlachar, Loch Katrine, Trossachs, UK, United Kingdom, wet road, asphalt, bare trees, rain, lochside, vehicle, parking, parking lines, car, car park, electric, car park, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car, car,
```

---

### ✅ mlx-community/Qwen3-VL-2B-Thinking-bf16

**Metrics:** 76.2 TPS | 500 tokens

```text
Got it, let's analyze the image step by step. First, the main subject is a light blue SUV parked near a waterfront. The setting is a rainy day, with wet roads and trees. The background has a loch (lake) and hills. The car is a Honda HR-V e:HEV, as per the context hints. Let's check the metadata.

Title: Needs to be 6-12 words. Descriptive and concrete. So "Light blue Honda HR-V e:HEV parked by Loch Katrine on rainy day" – that's 10 words. Wait, check if it's concrete. The car is light blue, Honda HR-V e:HEV, parked by the loch on a rainy day. Yes.

Description: 1-2 factual sentences. Key subjects: light blue SUV, setting: rainy day, location: near Loch Katrine, Scotland. Action: parked by the waterfront. So "A light blue Honda HR-V e:HEV is parked on a wet asphalt surface near Loch Katrine, Scotland, with rain falling on the surrounding trees and water."

Keywords: Need 15-30 comma-separated terms. Most specific to most general. Let's list: light blue SUV, Honda HR-V e:HEV, Loch Katrine, Scotland, UK, rainy day, wet road, asphalt, waterfront, trees, hills, loch, Scotland, United Kingdom, Vehicles, Honda, SUV, light blue, rain, wet, road, parking, lochside, Scotland, UK, United Kingdom, Vehicles, Honda, SUV, light blue, rain, wet, road, parking, lochside, Scotland, UK, United Kingdom, Vehicles, Honda, SUV, light blue, rain, wet, road, parking, lochside, Scotland, UK, United Kingdom, Vehicles, Honda, SUV, light blue, rain, wet, road, parking, lochside, Scotland, UK, United Kingdom, Vehicles, Honda, SUV, light blue, rain, wet, road, parking, lochside, Scotland, UK, United Kingdom, Vehicles, Honda, SUV, light blue, rain, wet, road, parking, lochside, Scotland, UK, United Kingdom, Vehicles, Honda, SUV, light blue, rain, wet, road, parking, lochside, Scotland, UK, United Kingdom, Vehicles, Honda, SUV, light blue, rain,
```

---

### ✅ mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

**Metrics:** 8.66 TPS | 500 tokens

```text
**Title:** A Blue SUV on a Rainy Day at Loch Katrine

**Description:** The image features a vibrant blue Honda HR-V e:HEV parked at the edge of a wet road near a pier on Loch Katrine, with the surrounding landscape shrouded in mist and rain. The scene is characterized by the calm waters of the loch and the bare-branched trees in the background.

**Keywords:** Honda HR-V e:HEV, Loch Katrine, Scotland, UK, Blue SUV, Rainy Day, Wet Road, Pier, Trees, Misty Landscape, Calm Waters, Vehicle, Car, Road, Landscape, Nature, Outdoor, Daytime, Cloudy, Overcast, Foggy, Misty, Atmospheric, Moody, Serene, Peaceful, Calm, Quiet, Still, Tranquil, Soothing, Comforting, Uplifting, Inspiring, Motivating, Energizing, Revitalizing, Refreshing, Rejuvenating, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revitalizing, Revital
```

---

### ✅ mlx-community/Qwen2-VL-2B-Instruct-4bit

**Metrics:** 180 TPS | 168 tokens

```text
Title: "Electric Vehicle at Loch Katrine, Scotland"
Description: "A vibrant blue electric vehicle provides a stark contrast to the muted tones of a rainy late-winter afternoon at Stronachlachar pier on Loch Katrine, Scotland."
Keywords: "Electric Vehicle," "Loch Katrine," "Scotland," "UK," "Honda HR-V e:HEV," "Lochside," "SUV," "Rain," "Asphalt," "Bare Trees," "Trossachs"
Context: The image shows a vibrant blue electric vehicle parked at a pier on Loch Katrine, Scotland, with a backdrop of bare-branched trees and distant hills. The description includes specific details about the vehicle, the location, and the weather conditions, while the keywords provide additional information about the vehicle and the setting.
```

---

### ✅ meta-llama/Llama-3.2-11B-Vision-Instruct

**Metrics:** 3.83 TPS | 500 tokens

```text
**Title:** "A Rainy Day at Stronachlachar Pier, Loch Katrine, Scotland"

**Description:** A vibrant blue Honda HR-V e:HEV is parked on a wet road near the pier at Stronachlachar, Loch Katrine, on a rainy late-winter afternoon. The surrounding landscape is characterized by bare trees and misty hills.

**Keywords:** Honda HR-V, Honda HR-V e:HEV, Light Blue SUV, Loch Katrine, Lochside, Stronachlachar, Trossachs, Scotland, UK, United Kingdom, Vehicles, Wet Road, asphalt, bare trees, rain, misty hills, pier, car, parking lot, road, landscape, nature, outdoor, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery, scenery
```

---

<!-- markdownlint-enable MD013 MD033 -->

---

## System/Hardware Information

- **OS**: Darwin 25.3.0
- **macOS Version**: 26.3
- **SDK Version**: 26.2
- **Xcode Version**: 26.2
- **Xcode Build**: 17C52
- **Metal SDK**: MacOSX.sdk
- **Python Version**: 3.13.11
- **Architecture**: arm64
- **GPU/Chip**: Apple M4 Max
- **GPU Cores**: 40
- **Metal Support**: Metal 4
- **RAM**: 128.0 GB
- **CPU Cores (Physical)**: 16
- **CPU Cores (Logical)**: 16

## Library Versions

- `Pillow`: `12.1.1`
- `huggingface-hub`: `1.4.1`
- `mlx`: `0.30.7.dev20260223+d4c81062`
- `mlx-lm`: `0.30.8`
- `mlx-metal`: ``
- `mlx-vlm`: `0.3.12`
- `numpy`: `2.4.2`
- `tokenizers`: `0.22.2`
- `transformers`: `5.2.0`

_Report generated on: 2026-02-23 12:54:48 GMT_
