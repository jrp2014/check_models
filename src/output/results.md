# Model Performance Results

_Generated on 2026-02-13 13:18:47 GMT_

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/nanoLLaVA-1.5-4bit` (315.7 tps)
- **💾 Most efficient:** `mlx-community/nanoLLaVA-1.5-4bit` (2.8 GB)
- **⚡ Fastest load:** `HuggingFaceTB/SmolVLM-Instruct` (0.00s)
- **📊 Average TPS:** 71.1 across 33 models

## 📈 Resource Usage

- **Total peak memory:** 659.7 GB
- **Average peak memory:** 20.0 GB
- **Memory efficiency:** 173 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 1 | ✅ B: 6 | 🟡 C: 5 | 🟠 D: 6 | ❌ F: 15

**Average Utility Score:** 39/100

- **Best for cataloging:** `mlx-community/gemma-3-27b-it-qat-4bit` (🏆 A, 88/100)
- **Worst for cataloging:** `mlx-community/Molmo-7B-D-0924-8bit` (❌ F, 0/100)

### ⚠️ 21 Models with Low Utility (D/F)

- `Qwen/Qwen3-VL-2B-Instruct`: ❌ F (27/100) - Mostly echoes context without adding value
- `microsoft/Phi-3.5-vision-instruct`: ❌ F (26/100) - Mostly echoes context without adding value
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (7/100) - Output too short to be useful
- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`: 🟠 D (43/100) - Mostly echoes context without adding value
- `mlx-community/InternVL3-14B-8bit`: ❌ F (31/100) - Mostly echoes context without adding value
- `mlx-community/LFM2-VL-1.6B-8bit`: ❌ F (26/100) - Mostly echoes context without adding value
- `mlx-community/LFM2.5-VL-1.6B-bf16`: ❌ F (24/100) - Mostly echoes context without adding value
- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`: 🟠 D (47/100) - Mostly echoes context without adding value
- `mlx-community/Molmo-7B-D-0924-8bit`: ❌ F (0/100) - Empty or minimal output
- `mlx-community/Molmo-7B-D-0924-bf16`: ❌ F (0/100) - Empty or minimal output
- `mlx-community/Phi-3.5-vision-instruct-bf16`: ❌ F (26/100) - Mostly echoes context without adding value
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: ❌ F (21/100) - Mostly echoes context without adding value
- `mlx-community/deepseek-vl2-8bit`: 🟠 D (45/100) - Mostly echoes context without adding value
- `mlx-community/gemma-3-27b-it-qat-8bit`: 🟠 D (47/100) - Mostly echoes context without adding value
- `mlx-community/llava-v1.6-mistral-7b-8bit`: ❌ F (25/100) - Mostly echoes context without adding value
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (5/100) - Output too short to be useful
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (23/100) - Lacks visual description of image
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: 🟠 D (48/100) - Lacks visual description of image
- `mlx-community/pixtral-12b-8bit`: ❌ F (24/100) - Mostly echoes context without adding value
- `mlx-community/pixtral-12b-bf16`: 🟠 D (39/100) - Mostly echoes context without adding value
- `qnguyen3/nanoLLaVA`: ❌ F (15/100) - Mostly echoes context without adding value

## ⚠️ Quality Issues

- **❌ Failed Models (9):**
  - `microsoft/Florence-2-large-ft` (`Weight Mismatch`)
  - `mlx-community/GLM-4.6V-Flash-6bit` (`Model Error`)
  - `mlx-community/GLM-4.6V-Flash-mxfp4` (`Model Error`)
  - `mlx-community/Idefics3-8B-Llama3-bf16` (`Model Error`)
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit` (`Model Error`)
  - `mlx-community/X-Reasoner-7B-8bit` (`OOM`)
  - `mlx-community/gemma-3n-E2B-4bit` (`No Chat Template`)
  - `mlx-community/paligemma2-3b-pt-896-4bit` (`Model Error`)
  - `prince-canuma/Florence-2-large-ft` (`Model Error`)
- **🔄 Repetitive Output (6):**
  - `Qwen/Qwen3-VL-2B-Instruct` (token: `phrase: "churchyard, churchyard, church..."`)
  - `microsoft/Phi-3.5-vision-instruct` (token: `phrase: "weather, weather, weather, wea..."`)
  - `mlx-community/Phi-3.5-vision-instruct-bf16` (token: `phrase: "weather, weather, weather, wea..."`)
  - `mlx-community/Qwen2-VL-2B-Instruct-4bit` (token: `phrase: "travel photography, outdoor ph..."`)
  - `mlx-community/deepseek-vl2-8bit` (token: `phrase: "english country church, englis..."`)
  - `mlx-community/paligemma2-3b-ft-docci-448-bf16` (token: `phrase: "the windows are set..."`)

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 71.1 | Min: 0 | Max: 316
- **Peak Memory**: Avg: 20 | Min: 2.8 | Max: 73
- **Total Time**: Avg: 27.08s | Min: 2.59s | Max: 117.80s
- **Generation Time**: Avg: 23.97s | Min: 1.67s | Max: 116.52s
- **Model Load Time**: Avg: 3.12s | Min: 0.90s | Max: 9.25s

## 🚨 Failures by Package (Actionable)

<!-- markdownlint-disable MD060 -->

| Package | Failures | Error Types | Affected Models |
|---------|----------|-------------|-----------------|
| `mlx-vlm` | 6 | Model Error, No Chat Template | `mlx-community/GLM-4.6V-Flash-6bit`, `mlx-community/GLM-4.6V-Flash-mxfp4`, `mlx-community/Kimi-VL-A3B-Thinking-8bit`, `mlx-community/gemma-3n-E2B-4bit`, `mlx-community/paligemma2-3b-pt-896-4bit`, `prince-canuma/Florence-2-large-ft` |
| `mlx` | 3 | Model Error, OOM, Weight Mismatch | `microsoft/Florence-2-large-ft`, `mlx-community/Idefics3-8B-Llama3-bf16`, `mlx-community/X-Reasoner-7B-8bit` |

<!-- markdownlint-enable MD060 -->

### Actionable Items by Package

#### mlx-vlm

- **mlx-community/GLM-4.6V-Flash-6bit** (Model Error)
  - Error: `Model generation failed for mlx-community/GLM-4.6V-Flash-6bit: [broadcast_shapes] Shapes (3,1,2048) and (3,1,6035) ca...`
  - Type: `ValueError`
- **mlx-community/GLM-4.6V-Flash-mxfp4** (Model Error)
  - Error: `Model generation failed for mlx-community/GLM-4.6V-Flash-mxfp4: [broadcast_shapes] Shapes (3,1,2048) and (3,1,6035) c...`
  - Type: `ValueError`
- **mlx-community/Kimi-VL-A3B-Thinking-8bit** (Model Error)
  - Error: `Model generation failed for mlx-community/Kimi-VL-A3B-Thinking-8bit: [broadcast_shapes] Shapes (999,2048) and (1,0,20...`
  - Type: `ValueError`
- **mlx-community/gemma-3n-E2B-4bit** (No Chat Template)
  - Error: `Cannot use chat template functions because tokenizer.chat_template is not set and no template argument was passed! Fo...`
  - Type: `ValueError`
- **mlx-community/paligemma2-3b-pt-896-4bit** (Model Error)
  - Error: `Model generation failed for mlx-community/paligemma2-3b-pt-896-4bit: [broadcast_shapes] Shapes (1,4,2,2048,4096) and ...`
  - Type: `ValueError`
- **prince-canuma/Florence-2-large-ft** (Model Error)
  - Error: `Model loading failed: RobertaTokenizer has no attribute additional_special_tokens`
  - Type: `ValueError`

#### mlx

- **microsoft/Florence-2-large-ft** (Weight Mismatch)
  - Error: `Model loading failed: Missing 1 parameters:
language_model.lm_head.weight.`
  - Type: `ValueError`
- **mlx-community/Idefics3-8B-Llama3-bf16** (Model Error)
  - Error: `Model loading failed: Expected shape (4096, 4096) but received shape (1024, 4096) for parameter language_model.layers...`
  - Type: `ValueError`
- **mlx-community/X-Reasoner-7B-8bit** (OOM)
  - Error: `Model runtime error during generation for mlx-community/X-Reasoner-7B-8bit: [metal::malloc] Attempting to allocate 13...`
  - Type: `ValueError`

> **Prompt used:**
>

> ```text
> Analyze this image and provide structured metadata for image cataloguing.
>
> Respond with exactly these three sections:
>
> Title: A concise, descriptive title (5–12 words).
>
> Description: A factual 1–3 sentence description of what the image shows, covering key subjects, setting, and action.
>
> Keywords: 25–50 comma-separated keywords ordered from most specific to most general, covering:
> - Specific subjects (people, objects, animals, landmarks)
> - Setting and location (urban, rural, indoor, outdoor)
> - Actions and activities
> - Concepts and themes (e.g. teamwork, solitude, celebration)
> - Mood and emotion (e.g. serene, dramatic, joyful)
> - Visual style (e.g. close-up, wide-angle, aerial, silhouette, bokeh)
> - Colors and lighting (e.g. golden hour, blue tones, high-key)
> - Seasonal and temporal context
> - Use-case relevance (e.g. business, travel, food, editorial)
>
> Context: The image is described as ', St Michael's Parish Church, Bishop's Stortford, England, United Kingdom, UK
St Michael's Parish Church in Bishop's Stortford, England, is pictured here on a late afternoon under a cool February sky. This Grade I listed building is a prominent example of Perpendicular Gothic architecture, with its current structure dating largely from the early 15th century. The photograph captures the intricate details of the church's traditional flint-knapped walls and stone-dressed windows, set against the backdrop of its historic churchyard. The soft, even light of the overcast day accentuates the textures and formidable presence of this centuries-old landmark.'.
> Existing keywords: Bishop's Stortford, British, Christian, Christianity, England, English Church, Europe, Faith, Flint Knapped Wall, Flint Wall, Gothic Window, Grass, Graveyard, Hertfordshire, Parish church, Perpendicular Gothic, St Michael's Church, St Michael's Parish Church, UK, United Kingdom, ancient, arched window, architecture, battlements, black, brown, building, car, cemetery, church, church spire, churchyard, clock tower, cloudy, cloudy sky, cross, exterior, gothic architecture, gravestone, grey, headstone, heritage, historic, historic building, landmark, medieval, old, overcast, overcast sky, place of worship, religion, religious building, spire, stone, tombstone, tourism, tower, travel, weathervane, worship
> Taken on 2026-02-07 16:11:23 GMT (at 16:11:23 local time).
>
> Be factual about visual content.  Include relevant conceptual and emotional keywords where clearly supported by the image.
> ```

**Note:** Results sorted: errors first, then by generation time (fastest to slowest).

**Overall runtime:** 953.90s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Token |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                    |   Error Package |
|:--------------------------------------------------------|--------:|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:----------------------------------|----------------:|
| `microsoft/Florence-2-large-ft`                         |         |                   |                       |                |              |           |             |                  |            |             |                                   |             mlx |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |         |                   |                       |                |              |           |             |                  |            |             |                                   |         mlx-vlm |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |         |                   |                       |                |              |           |             |                  |            |             |                                   |         mlx-vlm |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |         |                   |                       |                |              |           |             |                  |            |             |                                   |             mlx |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |         |                   |                       |                |              |           |             |                  |            |             |                                   |         mlx-vlm |
| `mlx-community/X-Reasoner-7B-8bit`                      |         |                   |                       |                |              |           |             |                  |            |             |                                   |             mlx |
| `mlx-community/gemma-3n-E2B-4bit`                       |         |                   |                       |                |              |           |             |                  |            |             |                                   |         mlx-vlm |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |         |                   |                       |                |              |           |             |                  |            |             |                                   |         mlx-vlm |
| `prince-canuma/Florence-2-large-ft`                     |         |                   |                       |                |              |           |             |                  |            |             |                                   |         mlx-vlm |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |       7 |               863 |                   281 |          1,144 |        3,001 |       288 |         2.9 |            1.67s |      0.92s |       2.59s |                                   |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |       7 |               863 |                   187 |          1,050 |        3,002 |       178 |         4.0 |            1.70s |      1.07s |       2.77s |                                   |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | 151,645 |               585 |                   440 |          1,025 |        2,527 |       316 |         2.8 |            2.01s |      0.90s |       2.91s | verbose                           |                 |
| `qnguyen3/nanoLLaVA`                                    | 151,645 |               585 |                   148 |            733 |        2,481 |      99.2 |         4.9 |            2.13s |      1.02s |       3.15s |                                   |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |  49,279 |             1,692 |                   136 |          1,828 |        1,215 |       114 |         5.5 |            2.97s |      1.42s |       4.39s |                                   |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |       1 |             1,609 |                     4 |          1,613 |          450 |      48.6 |          12 |            3.95s |      3.19s |       7.14s | ⚠️harness(prompt_template), ...   |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |  49,154 |             1,792 |                   364 |          2,156 |        1,341 |       111 |         5.5 |            4.96s |      1.30s |       6.26s | verbose                           |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |  49,154 |             1,792 |                   364 |          2,156 |        1,051 |       111 |         5.5 |            5.32s |      1.24s |       6.56s | verbose                           |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |       2 |             3,468 |                   307 |          3,775 |          992 |       157 |         9.0 |            5.76s |      1.69s |       7.45s |                                   |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |       1 |             1,609 |                    20 |          1,629 |          480 |      5.27 |          27 |            7.46s |      4.86s |      12.32s | context-ignored                   |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |     414 |             1,570 |                   500 |          2,070 |          666 |       106 |          18 |            7.60s |      3.39s |      10.99s |                                   |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |      11 |             1,570 |                   500 |          2,070 |          890 |      65.2 |          37 |           10.04s |      5.94s |      15.99s |                                   |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |     106 |               856 |                   351 |          1,207 |          490 |      41.6 |          17 |           10.49s |      4.85s |      15.34s | context-ignored                   |                 |
| `mlx-community/deepseek-vl2-8bit`                       |      14 |             2,576 |                   500 |          3,076 |          714 |      60.1 |          32 |           12.38s |      5.49s |      17.87s | repetitive(phrase: "english co... |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |  93,938 |             1,908 |                   500 |          2,408 |          571 |      48.2 |          60 |           14.43s |      9.25s |      23.68s |                                   |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |       2 |             3,022 |                   372 |          3,394 |          296 |      41.1 |          11 |           19.58s |      1.78s |      21.36s |                                   |                 |
| `mlx-community/pixtral-12b-8bit`                        |  14,543 |             3,704 |                   500 |          4,204 |          515 |      33.4 |          16 |           22.48s |      3.14s |      25.63s | verbose                           |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |   1,338 |             3,795 |                   500 |          4,295 |          420 |      36.0 |          15 |           23.23s |      3.08s |      26.31s |                                   |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |     106 |               856 |                   288 |          1,144 |          189 |      14.3 |          34 |           25.03s |      7.11s |      32.13s |                                   |                 |
| `mlx-community/InternVL3-14B-8bit`                      | 151,645 |             3,904 |                   393 |          4,297 |          368 |      27.6 |          18 |           25.32s |      3.20s |      28.51s | verbose                           |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 | 236,764 |               856 |                   500 |          1,356 |          190 |      23.9 |          19 |           25.70s |      5.04s |      30.74s | verbose                           |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      | 128,009 |               565 |                   196 |            761 |          148 |       8.6 |          15 |           26.97s |      2.95s |      29.92s |                                   |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         | 235,265 |             1,609 |                   500 |          2,109 |        1,529 |      17.6 |          11 |           29.76s |      2.92s |      32.68s | repetitive(phrase: "the window... |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |       2 |             2,968 |                   464 |          3,432 |          231 |      27.4 |          23 |           30.07s |      3.84s |      33.92s | ⚠️harness(encoding)               |                 |
| `mlx-community/pixtral-12b-bf16`                        |  38,188 |             3,704 |                   500 |          4,204 |          573 |      18.7 |          28 |           33.47s |      4.90s |      38.37s | verbose                           |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |         |               0.0 |                   0.0 |            0.0 |            0 |         0 |          41 |           35.01s |      2.22s |      37.23s |                                   |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |         |               0.0 |                   0.0 |            0.0 |            0 |         0 |          48 |           35.80s |      3.29s |      39.09s |                                   |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |  14,826 |             1,446 |                   500 |          1,946 |          921 |      12.2 |          12 |           42.93s |      1.71s |      44.64s | repetitive(phrase: "weather, ...  |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |  14,826 |             1,446 |                   500 |          1,946 |        1,020 |      12.0 |          12 |           43.60s |      2.26s |      45.86s | repetitive(phrase: "weather, ...  |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | 128,009 |               566 |                   176 |            742 |          152 |      3.82 |          25 |           50.07s |      4.34s |      54.41s |                                   |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |  17,428 |            16,858 |                   500 |         17,358 |          383 |      75.2 |         8.3 |           51.03s |      1.57s |      52.59s | repetitive(phrase: "churchyard... |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |      11 |            16,860 |                   500 |         17,360 |          312 |      72.0 |         8.3 |           61.46s |      1.73s |      63.20s |                                   |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |      11 |            16,869 |                   500 |         17,369 |          149 |       174 |          73 |          116.52s |      1.28s |     117.80s | repetitive(phrase: "travel pho... |                 |

<!-- markdownlint-enable MD013 MD033 MD034 MD037 MD049 -->

## Model Gallery

Full output from each model:


<!-- markdownlint-disable MD013 MD033 -->

### ❌ microsoft/Florence-2-large-ft

**Status:** Failed (Weight Mismatch)
**Error:** Model loading failed: Missing 1 parameters:
language_model.lm_head.weight.
**Type:** `ValueError`
**Package:** `mlx`

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6889, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6855, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<3 lines>...
        trust_remote_code=params.trust_remote_code,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 394, in load
    model = load_model(model_path, lazy, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 314, in load_model
    model.load_weights(list(weights.items()))
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx/python/mlx/nn/layers/base.py", line 191, in load_weights
    raise ValueError(f"Missing {num_missing} parameters: \n{missing}.")
ValueError: Missing 1 parameters:
language_model.lm_head.weight.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 7000, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6897, in _run_model_generation
    raise ValueError(error_details) from load_err
ValueError: Model loading failed: Missing 1 parameters:
language_model.lm_head.weight.
```

</details>

---

### ❌ mlx-community/GLM-4.6V-Flash-6bit

**Status:** Failed (Model Error)
**Error:**

> Model generation failed for mlx-community/GLM-4.6V-Flash-6bit:
> [broadcast_shapes] Shapes (3,1,2048) and (3,1,6035) cannot be broadcast.
**Type:** `ValueError`
**Package:** `mlx-vlm`

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6927, in _run_model_generation
    output: GenerationResult | SupportsGenerationResult = generate(
                                                          ~~~~~~~~^
        model=model,
        ^^^^^^^^^^^^
    ...<13 lines>...
        **extra_kwargs,
        ^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 599, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 489, in stream_generate
    for n, (token, logprobs) in enumerate(
                                ~~~~~~~~~^
        generate_step(input_ids, model, pixel_values, mask, **kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ):
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 378, in generate_step
    model.language_model(
    ~~~~~~~~~~~~~~~~~~~~^
        inputs=input_ids[:, :n_to_process],
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/glm4v/language.py", line 530, in __call__
    position_ids, rope_deltas = self.get_rope_index(
                                ~~~~~~~~~~~~~~~~~~~^
        inputs, image_grid_thw, video_grid_thw, mask
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/glm4v/language.py", line 456, in get_rope_index
    new_positions = mx.where(
        expanded_mask, expanded_positions, position_ids[:, i : i + 1, :]
    )
ValueError: [broadcast_shapes] Shapes (3,1,2048) and (3,1,6035) cannot be broadcast.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 7000, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6952, in _run_model_generation
    raise ValueError(msg) from gen_known_err
ValueError: Model generation failed for mlx-community/GLM-4.6V-Flash-6bit: [broadcast_shapes] Shapes (3,1,2048) and (3,1,6035) cannot be broadcast.
```

</details>

---

### ❌ mlx-community/GLM-4.6V-Flash-mxfp4

**Status:** Failed (Model Error)
**Error:**

> Model generation failed for mlx-community/GLM-4.6V-Flash-mxfp4:
> [broadcast_shapes] Shapes (3,1,2048) and (3,1,6035) cannot be broadcast.
**Type:** `ValueError`
**Package:** `mlx-vlm`

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6927, in _run_model_generation
    output: GenerationResult | SupportsGenerationResult = generate(
                                                          ~~~~~~~~^
        model=model,
        ^^^^^^^^^^^^
    ...<13 lines>...
        **extra_kwargs,
        ^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 599, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 489, in stream_generate
    for n, (token, logprobs) in enumerate(
                                ~~~~~~~~~^
        generate_step(input_ids, model, pixel_values, mask, **kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ):
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 378, in generate_step
    model.language_model(
    ~~~~~~~~~~~~~~~~~~~~^
        inputs=input_ids[:, :n_to_process],
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/glm4v/language.py", line 530, in __call__
    position_ids, rope_deltas = self.get_rope_index(
                                ~~~~~~~~~~~~~~~~~~~^
        inputs, image_grid_thw, video_grid_thw, mask
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/glm4v/language.py", line 456, in get_rope_index
    new_positions = mx.where(
        expanded_mask, expanded_positions, position_ids[:, i : i + 1, :]
    )
ValueError: [broadcast_shapes] Shapes (3,1,2048) and (3,1,6035) cannot be broadcast.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 7000, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6952, in _run_model_generation
    raise ValueError(msg) from gen_known_err
ValueError: Model generation failed for mlx-community/GLM-4.6V-Flash-mxfp4: [broadcast_shapes] Shapes (3,1,2048) and (3,1,6035) cannot be broadcast.
```

</details>

---

### ❌ mlx-community/Idefics3-8B-Llama3-bf16

**Status:** Failed (Model Error)
**Error:**

> Model loading failed: Expected shape (4096, 4096) but received shape (1024,
> 4096) for parameter language_model.layers.0.self_attn.k_proj.weight
**Type:** `ValueError`
**Package:** `mlx`

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6889, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6855, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<3 lines>...
        trust_remote_code=params.trust_remote_code,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 394, in load
    model = load_model(model_path, lazy, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 314, in load_model
    model.load_weights(list(weights.items()))
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx/python/mlx/nn/layers/base.py", line 200, in load_weights
    raise ValueError(
    ...<2 lines>...
    )
ValueError: Expected shape (4096, 4096) but received shape (1024, 4096) for parameter language_model.layers.0.self_attn.k_proj.weight

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 7000, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6897, in _run_model_generation
    raise ValueError(error_details) from load_err
ValueError: Model loading failed: Expected shape (4096, 4096) but received shape (1024, 4096) for parameter language_model.layers.0.self_attn.k_proj.weight
```

</details>

---

### ❌ mlx-community/Kimi-VL-A3B-Thinking-8bit

**Status:** Failed (Model Error)
**Error:**

> Model generation failed for mlx-community/Kimi-VL-A3B-Thinking-8bit:
> [broadcast_shapes] Shapes (999,2048) and (1,0,2048) cannot be broadcast.
**Type:** `ValueError`
**Package:** `mlx-vlm`

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6927, in _run_model_generation
    output: GenerationResult | SupportsGenerationResult = generate(
                                                          ~~~~~~~~^
        model=model,
        ^^^^^^^^^^^^
    ...<13 lines>...
        **extra_kwargs,
        ^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 599, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 489, in stream_generate
    for n, (token, logprobs) in enumerate(
                                ~~~~~~~~~^
        generate_step(input_ids, model, pixel_values, mask, **kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ):
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 359, in generate_step
    embedding_output = model.get_input_embeddings(
        input_ids, pixel_values, mask=mask, **kwargs
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/kimi_vl/kimi_vl.py", line 86, in get_input_embeddings
    final_inputs_embeds = self._prepare_inputs_for_multimodal(
        image_features, inputs_embeds, input_ids
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/kimi_vl/kimi_vl.py", line 97, in _prepare_inputs_for_multimodal
    inputs_embeds[:, image_positions, :] = image_features
    ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
ValueError: [broadcast_shapes] Shapes (999,2048) and (1,0,2048) cannot be broadcast.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 7000, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6952, in _run_model_generation
    raise ValueError(msg) from gen_known_err
ValueError: Model generation failed for mlx-community/Kimi-VL-A3B-Thinking-8bit: [broadcast_shapes] Shapes (999,2048) and (1,0,2048) cannot be broadcast.
```

</details>

---

### ❌ mlx-community/X-Reasoner-7B-8bit

**Status:** Failed (OOM)
**Error:**

> Model runtime error during generation for mlx-community/X-Reasoner-7B-8bit:
> [metal::malloc] Attempting to allocate 135699660800 bytes which is greater
> than the maximum allowed buffer size of 86586540032 bytes.
**Type:** `ValueError`
**Package:** `mlx`

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6927, in _run_model_generation
    output: GenerationResult | SupportsGenerationResult = generate(
                                                          ~~~~~~~~^
        model=model,
        ^^^^^^^^^^^^
    ...<13 lines>...
        **extra_kwargs,
        ^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 599, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 489, in stream_generate
    for n, (token, logprobs) in enumerate(
                                ~~~~~~~~~^
        generate_step(input_ids, model, pixel_values, mask, **kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ):
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 385, in generate_step
    mx.eval([c.state for c in prompt_cache])
    ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: [metal::malloc] Attempting to allocate 135699660800 bytes which is greater than the maximum allowed buffer size of 86586540032 bytes.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 7000, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6957, in _run_model_generation
    raise ValueError(msg) from gen_err
ValueError: Model runtime error during generation for mlx-community/X-Reasoner-7B-8bit: [metal::malloc] Attempting to allocate 135699660800 bytes which is greater than the maximum allowed buffer size of 86586540032 bytes.
```

</details>

---

### ❌ mlx-community/gemma-3n-E2B-4bit

**Status:** Failed (No Chat Template)
**Error:**

> Cannot use chat template functions because tokenizer.chat_template is not
> set and no template argument was passed! For information about writing
> templates and setting the tokenizer.chat_template attribute, please see the
> documentation at
> <https://huggingface.co/docs/transformers/main/en/chat_templating>
**Type:** `ValueError`
**Package:** `mlx-vlm`

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 7000, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6901, in _run_model_generation
    formatted_prompt: str | list[Any] = apply_chat_template(
                                        ~~~~~~~~~~~~~~~~~~~^
        processor=processor,
        ^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
        num_images=1,
        ^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/prompt_utils.py", line 567, in apply_chat_template
    return get_chat_template(processor, messages, add_generation_prompt)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/prompt_utils.py", line 448, in get_chat_template
    return processor.apply_chat_template(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        messages,
        ^^^^^^^^^
    ...<2 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/tokenization_utils_base.py", line 3009, in apply_chat_template
    chat_template = self.get_chat_template(chat_template, tools)
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/tokenization_utils_base.py", line 3191, in get_chat_template
    raise ValueError(
    ...<4 lines>...
    )
ValueError: Cannot use chat template functions because tokenizer.chat_template is not set and no template argument was passed! For information about writing templates and setting the tokenizer.chat_template attribute, please see the documentation at https://huggingface.co/docs/transformers/main/en/chat_templating
```

</details>

---

### ❌ mlx-community/paligemma2-3b-pt-896-4bit

**Status:** Failed (Model Error)
**Error:**

> Model generation failed for mlx-community/paligemma2-3b-pt-896-4bit:
> [broadcast_shapes] Shapes (1,4,2,2048,4096) and (2048,2048) cannot be
> broadcast.
**Type:** `ValueError`
**Package:** `mlx-vlm`

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6927, in _run_model_generation
    output: GenerationResult | SupportsGenerationResult = generate(
                                                          ~~~~~~~~^
        model=model,
        ^^^^^^^^^^^^
    ...<13 lines>...
        **extra_kwargs,
        ^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 599, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 489, in stream_generate
    for n, (token, logprobs) in enumerate(
                                ~~~~~~~~~^
        generate_step(input_ids, model, pixel_values, mask, **kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ):
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 378, in generate_step
    model.language_model(
    ~~~~~~~~~~~~~~~~~~~~^
        inputs=input_ids[:, :n_to_process],
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/paligemma/language.py", line 226, in __call__
    out = self.model(inputs, mask=mask, cache=cache, inputs_embeds=inputs_embeds)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/paligemma/language.py", line 200, in __call__
    h = layer(h, mask, c)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/paligemma/language.py", line 150, in __call__
    r = self.self_attn(self.input_layernorm(x), mask, cache)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/paligemma/language.py", line 99, in __call__
    scores = scores + mask
             ~~~~~~~^~~~~~
ValueError: [broadcast_shapes] Shapes (1,4,2,2048,4096) and (2048,2048) cannot be broadcast.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 7000, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6952, in _run_model_generation
    raise ValueError(msg) from gen_known_err
ValueError: Model generation failed for mlx-community/paligemma2-3b-pt-896-4bit: [broadcast_shapes] Shapes (1,4,2,2048,4096) and (2048,2048) cannot be broadcast.
```

</details>

---

### ❌ prince-canuma/Florence-2-large-ft

**Status:** Failed (Model Error)
**Error:**

> Model loading failed: RobertaTokenizer has no attribute
> additional_special_tokens
**Type:** `ValueError`
**Package:** `mlx-vlm`

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6889, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6855, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<3 lines>...
        trust_remote_code=params.trust_remote_code,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 404, in load
    processor = load_processor(model_path, True, eos_token_ids=eos_token_id, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 477, in load_processor
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/molmo/processing_molmo.py", line 758, in _patched_auto_processor_from_pretrained
    return _original_auto_processor_from_pretrained.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/phi3_v/processing_phi3_v.py", line 699, in _patched_auto_processor_from_pretrained
    return _original_auto_processor_from_pretrained.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/kimi_vl/processing_kimi_vl.py", line 555, in _patched_auto_processor_from_pretrained
    return _original_auto_processor_from_pretrained.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/models/auto/processing_auto.py", line 394, in from_pretrained
    return processor_class.from_pretrained(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/processing_utils.py", line 1403, in from_pretrained
    return cls.from_args_and_dict(args, processor_dict, **instantiation_kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/processing_utils.py", line 1170, in from_args_and_dict
    processor = cls(*args, **valid_kwargs)
  File "/Users/jrp/.cache/huggingface/modules/transformers_modules/microsoft/Florence_hyphen_2_hyphen_large_hyphen_ft/4a12a2b54b7016a48a22037fbd62da90cd566f2a/processing_florence2.py", line 87, in __init__
    tokenizer.additional_special_tokens + \
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/tokenization_utils_base.py", line 1291, in __getattr__
    raise AttributeError(f"{self.__class__.__name__} has no attribute {key}")
AttributeError: RobertaTokenizer has no attribute additional_special_tokens. Did you mean: 'add_special_tokens'?

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 7000, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6897, in _run_model_generation
    raise ValueError(error_details) from load_err
ValueError: Model loading failed: RobertaTokenizer has no attribute additional_special_tokens
```

</details>

---

### ✅ mlx-community/LFM2-VL-1.6B-8bit

**Metrics:** 288 TPS | 281 tokens

```text
Title: St Michael's Parish Church, Bishop's Stortford, England

Description: The image depicts St Michael's Parish Church, a Grade I listed building in Bishop's Stortford, England, United Kingdom. The church is a prominent example of Perpendicular Gothic architecture, with its current structure dating largely from the early 15th century. The photograph captures the intricate details of the church's traditional flint-knapped walls and stone-dressed windows, set against the backdrop of its historic churchyard. The soft, even light of the overcast day accentuates the textures and formidable presence of this centuries-old landmark.

Keywords: Bishop's Stortford, British, Christian, Christianity, England, English Church, Europe, Faith, Flint Knapped Wall, Flint Wall, Gothic Window, Grass, Graveyard, Hertfordshire, Parish church, Perpendicular Gothic, St Michael's Church, St Michael's Parish Church, UK, United Kingdom, ancient, arched window, architecture, battlements, black, brown, building, car, cemetery, church, church spire, stone, tombstone, heritage, historic, historic building, landmark, medieval, old, overcast, overcast sky, place of worship, religion, religious building, spire, stone, tombstone, tourism, tower, travel, weathervane, worship.
```

---

### ✅ mlx-community/LFM2.5-VL-1.6B-bf16

**Metrics:** 178 TPS | 187 tokens

```text
Title: St Michael's Parish Church, Bishop's Stortford, England

Description: The image depicts St Michael's Parish Church in Bishop's Stortford, England, a Grade I listed building showcasing Perpendicular Gothic architecture. The church's intricate details, such as the flint-knapped walls and stone-dressed windows, are highlighted against the backdrop of its historic churchyard. The overcast sky and soft lighting accentuate the textures and formidable presence of this centuries-old landmark.

Keywords: St Michael's Parish Church, Bishop's Stortford, England, Perpendicular Gothic, church, churchyard, church spire, stone, gravestone, heritage, historic building, landmark, medieval, old, overcast, overcast sky, place of worship, religion, religious building, spire, stone, tombstone, tourism, tower, travel, weathervane, worship
```

---

### ✅ mlx-community/nanoLLaVA-1.5-4bit

**Metrics:** 316 TPS | 440 tokens

```text
Title: A concise, descriptive title (5–12 words).

Description: A factual 1–3 sentence description of what the image shows, covering key subjects, setting, and action.

Keywords: 25–50 comma-separated keywords ordered from most specific to most general, covering:
- Specific subjects (people, objects, animals, landmarks)
- Setting and location (urban, rural, indoor, outdoor)
- Actions and activities
- Concepts and themes (e.g. teamwork, solitude, celebration)
- Mood and emotion (e.g. serene, dramatic, joyful)
- Visual style (e.g. close-up, wide-angle, aerial, silhouette, bokeh)
- Colors and lighting (e.g. golden hour, blue tones, high-key)
- Seasonal and temporal context
- Use-case relevance (e.g. business, travel, food, editorial)

Context: The image is described as 'St Michael's Parish Church, Bishop's Stortford, England, United Kingdom, UK St Michael's Parish Church in Bishop's Stortford, England is pictured here on a late afternoon under a cool February sky. This Grade I listed building is a prominent example of Perpendicular Gothic architecture, with its current structure dating largely from the early 15th century. The photograph captures the intricate details of the church's traditional flint-knapped walls and stone-dressed windows, set against the backdrop of its historic churchyard. The soft, even light of the overcast day accentuates the textures and formidable presence of this centuries-old landmark.'.
Existing keywords: Bishop's Stortford, British, Christianity, England, English Church, Europe, Faith, Flint Knapped Wall, Flint Wall, Gothic Window, Grass, Graveyard, Hertfordshire, Parish church, Perpendicular Gothic, St Michael's Church, St Michael's Parish Church, UK, United Kingdom, ancient, arched window, architecture, battlements, black, brown, building, car, cemetery, church, church spire, churchyard, clock tower, cloudy, cloudy sky, cross, tombstone, tourism, tower, travel, weathervane, worship.
```

---

### ✅ qnguyen3/nanoLLaVA

**Metrics:** 99.2 TPS | 148 tokens

```text
The image portrays a serene, late afternoon scene at St Michael's Parish Church in Bishop's Stortford, England. The church is a prominent example of Perpendicular Gothic architecture, with its current structure dating largely from the early 15th century. The image captures the intricate details of the church's traditional flint-knapped walls and stone-dressed windows, set against the backdrop of its historic churchyard. The soft, even light of the overcast day accentuates the textures and formidable presence of this centuries-old landmark. The image is taken on 2026-02-07 16:11:23 GMT (at 16:11:23 local time).
```

---

### ✅ mlx-community/SmolVLM2-2.2B-Instruct-mlx

**Metrics:** 114 TPS | 136 tokens

```text
 Title: St Michael's Parish Church, Bishop's Stortford, England, United Kingdom, UK
St Michael's Parish Church in Bishop's Stortford, England, is pictured here on a late afternoon under a cool February sky. This Grade I listed building is a prominent example of Perpendicular Gothic architecture, with its current structure dating largely from the early 15th century. The photograph captures the intricate details of the church's traditional flint-knapped walls and stone-dressed windows, set against the backdrop of its historic churchyard. The soft, even light of the overcast day accentuates the textures and formidable presence of this centuries-old landmark.
```

---

### ✅ mlx-community/paligemma2-10b-ft-docci-448-6bit

**Metrics:** 48.6 TPS | 4 tokens

```text
Be concise.
```

---

### ✅ mlx-community/SmolVLM-Instruct-bf16

**Metrics:** 111 TPS | 364 tokens

```text
 Image Description:

The image depicts St Michael's Parish Church, a prominent example of Perpendicular Gothic architecture, located in Bishop's Stortford, England. The church is a Grade I listed building, indicating its historical and architectural significance. The photograph captures the intricate details of the church's traditional flint-knapped walls and stone-dressed windows, set against the backdrop of its historic churchyard.

The church's exterior is characterized by its flint-knapped walls and stone-dressed windows, which are typical of the Perpendicular Gothic architectural style. The windows are tall and narrow, with pointed arches and stained glass, which adds to the church's Gothic aesthetic. The church's exterior is also adorned with battlements, which are characteristic of medieval defensive architecture.

The churchyard is expansive and well-maintained, with a variety of gravestones and headstones, indicating the church's long history and the presence of numerous burials. The churchyard is a serene and peaceful setting, with a few trees and shrubs adding to the natural beauty of the scene.

The church's spire, which is visible in the background, is a prominent feature of the church's architecture. The spire is tall and pointed, with a distinctive design that is characteristic of Gothic architecture. The spire is topped with a weathervane, which is used to indicate the wind direction.

The image is taken from a slightly elevated angle, providing a clear view of the church's exterior and the churchyard. The lighting is soft and even, which accentuates the textures and details of the church's architecture.

In summary, the image captures the historical and architectural significance of St Michael's Parish Church, showcasing its Gothic style and the serene and peaceful atmosphere of its churchyard.
```

---

### ✅ HuggingFaceTB/SmolVLM-Instruct

**Metrics:** 111 TPS | 364 tokens

```text
 Image Description:

The image depicts St Michael's Parish Church, a prominent example of Perpendicular Gothic architecture, located in Bishop's Stortford, England. The church is a Grade I listed building, indicating its historical and architectural significance. The photograph captures the intricate details of the church's traditional flint-knapped walls and stone-dressed windows, set against the backdrop of its historic churchyard.

The church's exterior is characterized by its flint-knapped walls and stone-dressed windows, which are typical of the Perpendicular Gothic architectural style. The windows are tall and narrow, with pointed arches and stained glass, which adds to the church's Gothic aesthetic. The church's exterior is also adorned with battlements, which are characteristic of medieval defensive architecture.

The churchyard is expansive and well-maintained, with a variety of gravestones and headstones, indicating the church's long history and the presence of numerous burials. The churchyard is a serene and peaceful setting, with a few trees and shrubs adding to the natural beauty of the scene.

The church's spire, which is visible in the background, is a prominent feature of the church's architecture. The spire is tall and pointed, with a distinctive design that is characteristic of Gothic architecture. The spire is topped with a weathervane, which is used to indicate the wind direction.

The image is taken from a slightly elevated angle, providing a clear view of the church's exterior and the churchyard. The lighting is soft and even, which accentuates the textures and details of the church's architecture.

In summary, the image captures the historical and architectural significance of St Michael's Parish Church, showcasing its Gothic style and the serene and peaceful atmosphere of its churchyard.
```

---

### ✅ mlx-community/Ministral-3-3B-Instruct-2512-4bit

**Metrics:** 157 TPS | 307 tokens

```text
Title: **Historic Perpendicular Gothic Church with Stone Battlements**

---
**Description:**
The image depicts an exterior view of a historic church with distinct Gothic architecture, featuring tall, pointed spires, flint-knapped walls, and elaborate arched stone windows. The setting is a quiet churchyard with gravestones, under a cloudy sky, capturing the timeless and solemn atmosphere of a heritage site.

---
**Keywords:**
ancient building, arched window, battlements, British heritage, church exterior, church spire, churchyard, cloudy sky, cold tones, decorative stonework, detailed architecture, England, European heritage, faith-based landmark, flint wall, Gothic architecture, gravestone, grey tones, heritage site, historic church, heritage stone, medieval stone, mossy stone, overcast lighting, place of worship, Perpendicular Gothic, religious architecture, serene atmosphere, spire, stone construction, stone church, stonework, tower, UK heritage, weathered stone, worship site, wide-angle shot

---
**Context:**
The image is described as **St Michael's Parish Church, Bishop's Stortford, England, United Kingdom**, captured under an overcast sky on a likely cool day in February. The photograph highlights the church’s Grade I listed status, Perpendicular Gothic design, and its early 15th-century construction, emphasizing its role as a significant heritage and religious site. The lighting and textures evoke a sense of historical reverence and quietude.
```

---

### ✅ mlx-community/paligemma2-10b-ft-docci-448-bf16

**Metrics:** 5.27 TPS | 20 tokens

```text
Be specific about visual content. Include relevant geographic and temporal keywords where clearly supported by the image.
```

---

### ✅ jqlive/Kimi-VL-A3B-Thinking-2506-6bit

**Metrics:** 106 TPS | 500 tokens

```text
◁think▷Got it, let's tackle this. First, I need to create a title, description, and keywords for the image of St Michael's Parish Church in Bishop's Stortford, England.

Starting with the title: it should be concise and descriptive. The image shows the church, so maybe "St Michael's Parish Church, Bishop's Stortford, England" but that's too long. Maybe "St Michael's Parish Church, Bishop's Stortford, England" is okay, but let's make it a bit more engaging. Wait, the existing keywords include "St Michael's Parish Church" and "St Michael's Church", so maybe "St Michael's Parish Church, Bishop's Stortford, England" is good.

Description: Need to cover key subjects (church, architecture, graveyard), setting (historic, outdoor, churchyard), action (no people, just the building), concepts (heritage, architecture), mood (serene, historic). So: "St Michael's Parish Church in Bishop's Stortford, England, is a historic Grade I listed Perpendicular Gothic church with a stone exterior, featuring arched windows and a tall spire, set against a cloudy sky in its historic churchyard."

Keywords: Need to list 25-50 keywords. Let's go through the existing ones and add more. Existing keywords include Bishop's Stortford, England, Christian, Christianity, England, English Church, Europe, Faith, Flint Knapped Wall, Flint Wall, Gothic Window, Grass, Graveyard, Hertfordshire, Parish church, Perpendicular Gothic, St Michael's Church, St Michael's Parish Church, UK, United Kingdom, ancient, arched window, architecture, battlements, black, brown, building, car, cemetery, church, church spire, churchyard, clock tower, cloudy, cloudy sky, cross, exterior, gothic architecture, gravestone, grey, headstone, heritage, historic, historic building, landmark, medieval, old, overcast, overcast sky, place of worship, religion, religious building, spire, stone, tombstone, tourism, tower, travel, weathervane, worship. Now add more: maybe "stone exterior", "tall spire", "architectural details", "churchyard", "Grade I listed", "medieval architecture", "religious landmark", "historic stone walls", "architectural features", "religious building", "
```

---

### ✅ mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

**Metrics:** 65.2 TPS | 500 tokens

```text
◁think▷Got it, let's tackle this. First, I need to create a title, description, and keywords for the image of St Michael's Parish Church in Bishop's Stortford, England.

Starting with the title: it should be concise and descriptive. The image shows the church with its architectural details, so maybe "St Michael's Parish Church, Bishop's Stortford, England: Perpendicular Gothic Architecture in a Historic Churchyard." That's 12 words, which fits the 5-12 word limit.

Next, the description: need to cover key subjects (the church, its architecture, the churchyard), setting (historic, outdoor, possibly rural or suburban), actions (the church is static, so maybe "capturing intricate details"), concepts/themes (heritage, history, architecture), mood (serene, historic), visual style (wide-angle, architectural photography), colors (neutral tones, stone, green grass), seasonal (February, overcast), use-case (travel, tourism, editorial). So the description would be: "St Michael's Parish Church in Bishop's Stortford, England, is depicted in this photograph showcasing its Perpendicular Gothic architecture. The image highlights the church's flint-knapped walls, arched windows, and historic churchyard, set against an overcast sky. The wide-angle view emphasizes the building's imposing structure and intricate stonework, capturing the serene atmosphere of a historic landmark."

Now keywords: need 25-50, ordered from specific to general. Let's list them:

Specific subjects: St Michael's Parish Church, Bishop's Stortford, England, England, UK, Hertfordshire, Christian, Christianity, Christian Church, Faith, Religion, Worship, Tourism, Travel, Historic Building, Heritage, Medieval, Gothic Architecture, Perpendicular Gothic, Flint Knapped Wall, Flint Wall, Stone Wall, Churchyard, Graveyard, Tombstone, Headstone, Grass, Grasses, Church Spire, Clock Tower, Weathervane, Cross, Architectural Details, Architectural Style, Architectural Photography, Overcast Sky, Cloudy Sky, Neutral Tones, Stone Tones, Brown Tones, Grey Tones, Historic Landmark, Religious Building, Place of Worship.

Wait, need to check the existing keywords provided. The existing ones include Bishop's Stortford, British, Christian, Christianity, England, English Church, Europe, Faith, Flint Knapped Wall, Flint Wall, Gothic Window, Grass,
```

---

### ✅ mlx-community/gemma-3n-E4B-it-bf16

**Metrics:** 41.6 TPS | 351 tokens

```text
## Image Description: St. Michael's Church, Billericay

**Description:** A low-angle, wide shot captures the exterior of St. Michael's Church in Billericay, England. The Grade I listed building showcases intricate stonework, pointed arch windows, and a prominent tower under an overcast sky.

**Keywords:**

**People:** None visible.
**Objects:** Church, tower, windows, spire, gravestones, roof, clock, weather vane, stone, grass.
**Setting & Location:** Exterior, England, United Kingdom, Billericay, churchyard, historic site.
**Actions & Activities:** Standing, architectural.
**Concepts & Themes:** History, architecture, religion, heritage, preservation, time.
**Mood & Emotion:** Serene, historic, solemn, grand, timeless.
**Visual Style:** Wide-angle, low angle, architectural photography, overcast lighting.
**Colors & Lighting:** Earthy tones (brown, grey, green), muted colors, soft light, overcast.
**Composition & Perspective:** Low angle emphasizing height, symmetrical elements, rule of thirds.
**Technical Details:** Daylight, natural light, sharp focus.

**Context:** The image depicts the impressive architecture of St. Michael's Church, a significant historical landmark in Billericay, UK. The Grade I listed building, dating back to the early 15th century, exhibits classic Perpendicular Gothic design with its detailed stonework, large windows, and imposing tower. The churchyard, with its weathered gravestones, adds to the sense of history and tranquility. The overcast sky provides a soft, diffused light that highlights the texture and grandeur of the building.
```

---

### ✅ mlx-community/deepseek-vl2-8bit

**Metrics:** 60.1 TPS | 500 tokens

```text
**Title:** A Stunning View of St. Michael's Church in Bishop's Stortford, UK

**Description:** The image shows the historic St. Michael's Parish Church in Bishop's Stortford, a prime example of English Perpendicular Gothic architecture. The building is constructed from a mix of flint and stone, with a tall, pointed spire and large, arched windows. The structure is set against a backdrop of a cloudy, overcast sky, with a well-manicured lawn in the foreground. The scene exudes a serene and historical atmosphere, with the building's gothic details standing out against the overcast sky.

**Keywords:** St. Michael's Church, Bishop's Stortford, UK, English Heritage, gothic architecture, English Parish Church, historic building, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church,
```

---

### ✅ mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

**Metrics:** 48.2 TPS | 500 tokens

```text
Let's tackle this. First, the title needs to be concise. The image is of St Michael's Parish Church in Bishop's Stortford, England. So maybe "St Michael's Parish Church, Bishop's Stortford, England" – that's 7 words, fits the range.

Next, the description. It's a Grade I listed building, Perpendicular Gothic, early 15th century. The photo is from late afternoon, overcast sky. Key elements: flint-knapped walls, stone-dressed windows, churchyard. So: "St Michael's Parish Church, a Grade I listed Perpendicular Gothic building from the early 15th century, is shown in Bishop's Stortford, England, with its flint-knapped walls and stone windows against a cool, overcast late afternoon sky." That covers the subject, setting, and action (photograph capturing details).

Keywords: Need to order from specific to general. Let's list the specific ones first: St Michael's Parish Church, Bishop's Stortford, England, UK, Grade I listed building, Perpendicular Gothic architecture, early 15th century, flint-knapped walls, stone-dressed windows, churchyard, late afternoon, overcast sky, flint wall, gothic window, church spire, clock tower, graveyard, headstone, tombstone, weathervane, cross, ancient, historic, heritage, place of worship, religious building, church, churchyard, stone, brick, overcast, cloudy, February (wait, the context says late February, but the date is 2026-02-07, so February). Wait, the existing keywords have some, but need to make sure they're ordered correctly. Let's check:

Most specific: St Michael's Parish Church, Bishop's Stortford, England, UK, Grade I listed building, Perpendicular Gothic architecture, early 15th century, flint-knapped walls, stone-dressed windows, churchyard, late afternoon, overcast sky, flint wall, gothic window, church spire, clock tower, graveyard, headstone, tombstone, weathervane, cross, ancient, historic, heritage, place of worship,
```

---

### ✅ mlx-community/llava-v1.6-mistral-7b-8bit

**Metrics:** 41.1 TPS | 372 tokens

```text
Title: St Michael's Parish Church, Bishop's Stortford, England, United Kingdom, UK

Description: A photograph captures the intricate details of St Michael's Parish Church, a Grade I listed building in Bishop's Stortford, England, UK. The image showcases the traditional flint-knapped walls and stone-dressed windows, set against the backdrop of the historic churchyard. The soft, even light of the overcast day accentuates the textures and formidable presence of this centuries-old landmark.

Keywords: St Michael's Parish Church, Bishop's Stortford, English Church, Europe, Faith, Christianity, England, Gothic architecture, Heritage, Historic, Landmark, Medieval, Old, Perpendicular Gothic, Religion, Religious building, St Michael's Church, St Michael's Parish Church, United Kingdom, UK, Architecture, Church, Churchyard, Clock tower, Overcast sky, Stone, Tombstone, Tourism, Worship

Context: The image is described as 'St Michael's Parish Church in Bishop's Stortford, England, UK, pictured here on a late afternoon under a cool February sky. This Grade I listed building is a prominent example of Perpendicular Gothic architecture, with its current structure dating largely from the early 15th century. The photograph captures the intricate details of the church's traditional flint-knapped walls and stone-dressed windows, set against the backdrop of its historic churchyard. The soft, even light of the overcast day accentuates the textures and formidable presence of this centuries-old landmark.'
```

---

### ✅ mlx-community/pixtral-12b-8bit

**Metrics:** 33.4 TPS | 500 tokens

```text
### Title
St Michael's Parish Church, Bishop's Stortford

### Description
St Michael's Parish Church in Bishop's Stortford, England, is pictured here on a late afternoon under a cool February sky. This Grade I listed building is a prominent example of Perpendicular Gothic architecture, with its current structure dating largely from the early 15th century. The photograph captures the intricate details of the church's traditional flint-knapped walls and stone-dressed windows, set against the backdrop of its historic churchyard.

### Keywords
St Michael's Parish Church, Bishop's Stortford, British, Christian, Christianity, England, English Church, Europe, Faith, Flint Knapped Wall, Flint Wall, Gothic Window, Grass, Graveyard, Hertfordshire, Parish church, Perpendicular Gothic, St Michael's Church, St Michael's Parish Church, UK, United Kingdom, ancient, arched window, architecture, battlements, black, brown, building, cemetery, church, church spire, churchyard, clock tower, cloudy, cloudy sky, cross, exterior, gothic architecture, gravestone, grey, headstone, heritage, historic, historic building, landmark, medieval, old, overcast, overcast sky, place of worship, religion, religious building, spire, stone, tombstone, tourism, tower, travel, weathervane, worship, February, late afternoon, historic churchyard, flint-knapped walls, stone-dressed windows, Perpendicular Gothic architecture, centuries-old landmark, soft light, textures, formidable presence, traditional, traditional architecture, historic site, religious heritage, serene, dramatic, grey tones, stone structure, religious symbolism, spiritual, architectural detail, historic preservation, cultural significance, architectural landmark, English heritage, English architecture, English churchyard, English cemetery, English gravestone, English headstone, English tombstone, English church, English church spire, English churchyard, English church architecture, English religious building, English religious heritage, English religious landmark, English religious site, English religious tourism, English religious travel, English religious worship, English religious faith, English religious symbolism, English religious architecture, English religious heritage, English religious tourism, English religious travel, English religious worship, English religious faith, English religious symbolism, English religious architecture, English religious heritage, English religious tourism, English religious travel, English religious worship, English religious
```

---

### ✅ mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX

**Metrics:** 36 TPS | 500 tokens

```text
Here are my reasoning steps:
The user wants us to "Analyze this image and provide structured metadata for image cataloguing. Respond with exactly these three sections: Title, Description, Keywords. Also include Context: The image is described as ... (the context is given). The user also gave existing keywords. The user wants us to be factual about visual content, include relevant conceptual and emotional keywords where clearly supported.

We need to produce Title (5-12 words), Description (1-3 sentences), Keywords (25-50 comma-separated keywords ordered from most specific to most general). The keywords should cover specific subjects, setting, actions, concepts, mood, visual style, colors, lighting, seasonal, temporal context, use-case relevance.

We need to incorporate the context: St Michael's Parish Church, Bishop's Stortford, England, UK, late afternoon, cool February sky, Grade I listed building, Perpendicular Gothic architecture, early 15th century, flint-knapped walls, stone-dressed windows, historic churchyard, soft even light of overcast day, textures, formidable presence.

We need to be factual about visual content: The image shows a large stone church with flint-knapped walls, tall windows, a spire, a clock tower, battlements, a graveyard with gravestones, a weathervane on the spire, overcast sky, etc.

We need to produce a Title: 5-12 words. Something like "St Michael's Parish Church, Bishop's Stortford, Gothic Exterior". That's 9 words. Or "St Michael's Parish Church, Bishop's Stortford, Gothic Exterior in February". That's 10 words. Or "St Michael's Parish Church, Bishop's Stortford, Perpendicular Gothic". That's 7 words. Title should be concise, descriptive, 5-12 words.

Description: 1-3 sentences. Something like: "The image shows St Michael's Parish Church in Bishop's Stortford, England, captured on a cool February afternoon under an overcast sky. The Grade I listed Perpendicular Gothic building features flint-knapped walls, tall stone windows, a soaring spire, and a historic churchyard with gravestones. Soft, even light highlights the texture of the stone and the church's formidable presence."

That's 3 sentences, factual.


```

---

### ✅ mlx-community/gemma-3-27b-it-qat-8bit

**Metrics:** 14.3 TPS | 288 tokens

```text
**Title:** St Michael's Parish Church, Bishop's Stortford, England – February Afternoon

**Description:** This photograph depicts St Michael's Parish Church in Bishop's Stortford, Hertfordshire, England, a Grade I listed building showcasing Perpendicular Gothic architecture. The church is framed against a cloudy February sky, with its flint-knapped walls and stone windows prominently visible. Gravestones dot the surrounding churchyard, adding to the historic atmosphere.

**Keywords:** St Michael's Parish Church, Bishop's Stortford, Hertfordshire, England, UK, United Kingdom, Perpendicular Gothic, flint knapped wall, stone window, arched window, church spire, churchyard, graveyard, cemetery, gravestone, headstone, tombstone, historic building, landmark, medieval architecture, place of worship, religious building, church, exterior, overcast sky, cloudy, grey tones, brown tones, stone, tower, clock tower, battlements, cross, weathervane, heritage, tourism, travel, architecture, ancient, historic, religious site, European architecture, British architecture, place of interest, overcast lighting, soft light, winter, February, afternoon, solemn, quiet, stillness, texture, detail, wide-angle, building exterior, cultural heritage, religious art, gothic window, stone dressing, traditional architecture, European landmark, place of remembrance, solemnity, cultural tourism, architectural photography.
```

---

### ✅ mlx-community/InternVL3-14B-8bit

**Metrics:** 27.6 TPS | 393 tokens

```text
Title: Gothic Church with Battlements and Graveyard

Description: The image shows St Michael's Parish Church in Bishop's Stortford, England, featuring a tall spire, arched windows, and a historic graveyard with gravestones in the foreground. The architecture is Perpendicular Gothic, with intricate stone and flint walls under an overcast sky.

Keywords: Bishop's Stortford, British, Christian, Christianity, England, English Church, Europe, Faith, Flint Knapped Wall, Flint Wall, Gothic Window, Grass, Graveyard, Hertfordshire, Parish church, Perpendicular Gothic, St Michael's Church, St Michael's Parish Church, UK, United Kingdom, ancient, arched window, architecture, battlements, black, brown, building, cloudy, cloudy sky, cross, exterior, gothic architecture, gravestone, grey, headstone, heritage, historic, historic building, landmark, medieval, old, overcast, overcast sky, place of worship, religion, religious building, spire, stone, tombstone, tourism, tower, travel, weathervane, worship, serene, dramatic, wide-angle, muted tones, historical, solemn, weathered stone, traditional, quiet, contemplative

Context: The image is described as 'St Michael's Parish Church, Bishop's Stortford, England, United Kingdom, UK. St Michael's Parish Church in Bishop's Stortford, England, is pictured here on a late afternoon under a cool February sky. This Grade I listed building is a prominent example of Perpendicular Gothic architecture, with its current structure dating largely from the early 15th century. The photograph captures the intricate details of the church's traditional flint-knapped walls and stone-dressed windows, set against the backdrop of its historic churchyard. The soft, even light of the overcast day accentuates the textures and formidable presence of this centuries-old landmark.'
```

---

### ✅ mlx-community/gemma-3-27b-it-qat-4bit

**Metrics:** 23.9 TPS | 500 tokens

```text
## Title: St Michael's Parish Church, Bishop's Stortford, England – February Afternoon

## Description:
This photograph depicts the exterior of St Michael's Parish Church in Bishop's Stortford, England, a Grade I listed building showcasing Perpendicular Gothic architecture. The church is framed by a historic churchyard with numerous gravestones, and is photographed under a cloudy, overcast sky. The image highlights the church’s flint-knapped walls, arched windows, and prominent spire.

## Keywords:
St Michael's Parish Church, Bishop's Stortford, Hertfordshire, England, UK, United Kingdom, Perpendicular Gothic, flint knapped wall, arched window, church spire, churchyard, gravestones, headstones, tombstone, medieval architecture, historic building, landmark, place of worship, religious building, church, exterior, overcast sky, cloudy sky, grey tones, stone wall, spire, clock tower, weathervane, cross, gothic window, church architecture, British architecture, European architecture, heritage, tourism, travel, photography, architectural detail, religious site, place of interest, cultural heritage, history, ancient, solemn, peaceful, quiet, overcast light, February, winter, afternoon, wide-angle, exterior view, stone masonry, building facade, cultural landmark, religious art, religious monument, textured wall, stone structure, landmark architecture, religious institution, historical site, cultural tourism, architectural photography, religious building exterior, stone detailing, gothic style, religious practice, cultural preservation, stone texture, architectural design, religious history, stone construction, cultural identity, religious symbol, stone craftsmanship, architectural heritage, religious tradition, stone architecture, cultural significance, religious community, stone building, cultural expression, religious belief, stone work, cultural value, religious experience, stone feature, cultural context, religious influence, stone element, cultural impact, religious legacy, stone material, cultural asset, religious impact, stone surface, cultural resource, religious influence, stone composition, cultural element, religious foundation, stone form, cultural artifact, religious structure, stone pattern, cultural object, religious practice, stone design, cultural symbol, religious institution, stone element, cultural expression, stone feature, cultural context, stone composition, stone pattern, stone texture, stone work, stone surface, stone material, stone form, stone detailing, stone construction, stone masonry, stone architecture, stone building, stone feature, stone element, stone composition, stone pattern,
```

---

### ✅ mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

**Metrics:** 8.6 TPS | 196 tokens

```text
**Title:** St Michael's Parish Church, Bishop's Stortford, England

**Description:** The image depicts St Michael's Parish Church in Bishop's Stortford, England, showcasing its historic architecture and surrounding environment. The church's stone structure and intricate details are prominently featured against a cloudy sky.

**Keywords:**

* Specific subjects: church, building, architecture, stone, window, clock tower, spire, cross, weathervane
* Setting and location: England, UK, Europe, outdoor, rural
* Actions and activities: none
* Concepts and themes: history, heritage, tradition, faith, religion
* Mood and emotion: serene, peaceful, contemplative
* Visual style: architectural, detailed, close-up, wide-angle
* Colors and lighting: stone, brown, grey, cloudy sky
* Seasonal and temporal context: winter, late afternoon
* Use-case relevance: cultural, historical, religious, tourism, travel
```

---

### ✅ mlx-community/paligemma2-3b-ft-docci-448-bf16

**Metrics:** 17.6 TPS | 500 tokens

```text
A low-angle, three-quarters view of the side of St Michael's Parish Church in Bishop's Stortford, England. The church is a light brown stone with a brick-layer's bond pattern. The church is made of light brown stone with a brick-layer's bond pattern. The church has a tall, pointed, gray metal steeple on the right side of the church. The steeple is the tallest structure in the image. The church has a row of five windows on the left side of the church. The windows are rectangular and have a black metal frame. The windows are set in a row of three. The windows are set in a row of three. The windows are set in a row of three. The windows are set in a row of three. The windows are set in a row of three. The windows are set in a row of three. The windows are set in a row of three. The windows are set in a row of three. The windows are set in a row of three. The windows are set in a row of three. The windows are set in a row of three. The windows are set in a row of three. The windows are set in a row of three. The windows are set in a row of three. The windows are set in a row of three. The windows are set in a row of three. The windows are set in a row of three. The windows are set in a row of three. The windows are set in a row of three. The windows are set in a row of three. The windows are set in a row of three. The windows are set in a row of three. The windows are set in a row of three. The windows are set in a row of three. The windows are set in a row of three. The windows are set in a row of three. The windows are set in a row of three. The windows are set in a row of three. The windows are set in a row of three. The windows are set in a row of three. The windows are set in a row of three. The windows are set in a row of three. The windows are set in a row of three. The windows are set in a row of three. The windows are set in a row of three. The windows are set in a row of three. The windows are set in a row of three. The windows are set in a row of three.
```

---

### ✅ mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

**Metrics:** 27.4 TPS | 464 tokens

```text
Title:ĠStĠMichael'sĠParishĠChurch,ĠBishop'sĠStortfordĊĊDescription:ĠTheĠimageĠshowsĠStĠMichael'sĠParishĠChurchĠinĠBishop'sĠStortford,ĠEngland,ĠaĠhistoricĠGradeĠIĠlistedĠbuildingĠfeaturingĠPerpendicularĠGothicĠarchitecture.ĠTheĠchurch'sĠflint-knappedĠwallsĠandĠstone-trimmedĠwindowsĠareĠprominentlyĠdisplayed,ĠsetĠagainstĠanĠovercastĠsky.ĊĊKeywords:ĠStĠMichael'sĠParishĠChurch,ĠBishop'sĠStortford,ĠPerpendicularĠGothicĠarchitecture,ĠGradeĠIĠlistedĠbuilding,Ġflint-knappedĠwalls,Ġstone-trimmedĠwindows,ĠchurchĠspire,Ġchurchyard,Ġgravestones,ĠovercastĠsky,ĠhistoricĠbuilding,ĠmedievalĠarchitecture,ĠplaceĠofĠworship,ĠreligiousĠbuilding,Ġtourism,Ġtravel,Ġheritage,ĠEngland,ĠUK,ĠEurope,ĠChristian,ĠChristianity,Ġfaith,Ġarchitecture,Ġlandmark,Ġold,Ġancient,Ġexterior,Ġcloudy,Ġgrey,Ġbrown,Ġblack,Ġgrass,Ġcemetery,Ġheadstone,Ġtombstone,Ġcross,Ġweathervane,ĠclockĠtower,Ġbattlements,ĠarchedĠwindow,ĠgothicĠwindow,Ġcar,Ġbuilding,Ġspire,Ġtower,Ġreligion,Ġworship,Ġhistoric,Ġheritage,Ġtourism,Ġtravel,Ġovercast,ĠcloudyĠsky,ĠplaceĠofĠworship,ĠreligiousĠbuilding,Ġtourism,Ġtravel,Ġheritage,ĠEngland,ĠUK,ĠEurope,ĠChristian,ĠChristianity,Ġfaith,Ġarchitecture,Ġlandmark,Ġold,Ġancient,Ġexterior,Ġcloudy,Ġgrey,Ġbrown,Ġblack,Ġgrass,Ġcemetery,Ġheadstone,Ġtombstone,Ġcross,Ġweathervane,ĠclockĠtower,Ġbattlements,ĠarchedĠwindow,ĠgothicĠwindow,Ġcar,Ġbuilding,Ġspire,Ġtower,Ġreligion,ĠworshipĊĊContext:ĠTheĠimageĠisĠdescribedĠasĠ',ĠStĠMichael'sĠParishĠChurch,ĠBishop'sĠStortford,ĠEngland,ĠUnitedĠKingdom,ĠUKStĠMichael'sĠParishĠChurchĠinĠBishop'sĠStortford,ĠEngland,ĠisĠpicturedĠhereĠonĠaĠlateĠafternoonĠunderĠaĠcoolĠFebruaryĠsky.ĠThisĠGradeĠIĠlistedĠbuildingĠisĠaĠprominentĠexampleĠofĠPerpendicularĠGothicĠarchitecture,ĠwithĠitsĠcurrentĠstructureĠdatingĠlargelyĠfromĠtheĠearlyĠ15thĠcentury.ĠTheĠphotographĠcapturesĠtheĠintricateĠdetailsĠofĠtheĠchurch'sĠtraditionalĠflint-knappedĠwallsĠandĠstone-trimmedĠwindows,ĠsetĠagainstĠtheĠbackdropĠofĠitsĠhistoricĠchurchyard.ĠTheĠsoft,ĠevenĠlightĠofĠtheĠovercastĠdayĠaccentuatesĠtheĠtexturesĠandĠformidableĠpresenceĠofĠthisĠcenturies-oldĠlandmark.'.
```

---

### ✅ mlx-community/pixtral-12b-bf16

**Metrics:** 18.7 TPS | 500 tokens

```text
### Title
St Michael's Parish Church, Bishop's Stortford

### Description
St Michael's Parish Church in Bishop's Stortford, England, is pictured here on a late afternoon under a cool February sky. This Grade I listed building is a prominent example of Perpendicular Gothic architecture, with its current structure dating largely from the early 15th century. The photograph captures the intricate details of the church's traditional flint-knapped walls and stone-dressed windows, set against the backdrop of its historic churchyard.

### Keywords
St Michael's Parish Church, Bishop's Stortford, British, Christian, Christianity, England, English Church, Europe, Faith, Flint Knapped Wall, Flint Wall, Gothic Window, Grass, Graveyard, Hertfordshire, Parish church, Perpendicular Gothic, St Michael's Church, St Michael's Parish Church, UK, United Kingdom, ancient, arched window, architecture, battlements, black, brown, building, cemetery, church, church spire, churchyard, clock tower, cloudy, cloudy sky, cross, exterior, gothic architecture, gravestone, grey, headstone, heritage, historic, historic building, landmark, medieval, old, overcast, overcast sky, place of worship, religion, religious building, spire, stone, tombstone, tourism, tower, travel, weathervane, worship, February, late afternoon, historic churchyard, stone-dressed windows, flint-knapped walls, Perpendicular Gothic architecture, overcast day, textures, formidable presence, centuries-old landmark, traditional, serene, historic site, religious heritage, architectural detail, stone structure, English heritage, English architecture, British heritage, British architecture, Gothic architecture, medieval architecture, historic architecture, religious tourism, heritage tourism, architectural landmark, architectural detail, stone masonry, church architecture, church exterior, churchyard grass, churchyard tombstones, churchyard gravestones, churchyard headstones, churchyard cross, churchyard clock tower, churchyard spire, churchyard weathervane, churchyard tourism, churchyard heritage, churchyard history, churchyard architecture, churchyard design, churchyard landscape, churchyard setting, churchyard environment, churchyard atmosphere, churchyard ambiance, churchyard mood, churchyard emotion, churchyard feeling, churchyard experience, churchyard visit, churchyard tour, churchyard exploration
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

### ✅ microsoft/Phi-3.5-vision-instruct

**Metrics:** 12.2 TPS | 500 tokens

```text
Title: St Michael's Parish Church in Bishop's Stortford

Description: The St Michael's Parish Church in Bishop's Stortford, England, stands as a Grade I listed building, showcasing Perpendicular Gothic architecture with its early 15th-century origins. The image captures the church's detailed flint-knapped walls and stone-dressed windows, set against a historic churchyard, under a cool February sky.

Keywords: St Michael's Parish Church, Bishop's Stortford, England, United Kingdom, Grade I listed, Perpendicular Gothic, early 15th century, flint-knapped walls, stone-dressed windows, churchyard, historic, architecture, overcast, cool, February, traditional, landmark, heritage, medieval, Gothic, Christianity, Christian, architecture, tourism, historic building, worship, stone, tombstone, cross, church spire, churchyard, cemetery, church, churchyard, architecture, overcast sky, weathervane, travel, UK, English Church, Europe, Faith, architecture, building, car, cemetery, church, church spire, churchyard, clock tower, battlements, black, brown, building, tourism, travel, weathervane, weather, weathervane, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather
```

---

### ✅ mlx-community/Phi-3.5-vision-instruct-bf16

**Metrics:** 12 TPS | 500 tokens

```text
Title: St Michael's Parish Church in Bishop's Stortford

Description: The St Michael's Parish Church in Bishop's Stortford, England, stands as a Grade I listed building, showcasing Perpendicular Gothic architecture with its early 15th-century origins. The image captures the church's detailed flint-knapped walls and stone-dressed windows, set against a historic churchyard, under a cool February sky.

Keywords: St Michael's Parish Church, Bishop's Stortford, England, United Kingdom, Grade I listed, Perpendicular Gothic, early 15th century, flint-knapped walls, stone-dressed windows, churchyard, historic, architecture, overcast, cool, February, traditional, landmark, heritage, medieval, Gothic, Christianity, Christian, architecture, tourism, historic building, worship, stone, tombstone, cross, church spire, churchyard, cemetery, church, churchyard, architecture, overcast sky, weathervane, travel, UK, English Church, Europe, Faith, architecture, building, car, cemetery, church, church spire, churchyard, clock tower, battlements, black, brown, building, tourism, travel, weathervane, weather, weathervane, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather
```

---

### ✅ meta-llama/Llama-3.2-11B-Vision-Instruct

**Metrics:** 3.82 TPS | 176 tokens

```text
**Title:** St. Michael's Parish Church, Bishop's Stortford, England

**Description:** The image depicts St. Michael's Parish Church in Bishop's Stortford, England, showcasing its historic architecture and surrounding cemetery. The church's stone walls and clock tower are prominent features of the scene.

**Keywords:**

* Specific subjects: Church, Cemetery, Graveyard, Tombstone, Headstone, Cross, Weathervane
* Setting and location: England, Bishop's Stortford, Outdoor
* Actions and activities: None
* Concepts and themes: History, Heritage, Tradition, Faith, Worship
* Visual style: Wide-angle, Aerial
* Colors and lighting: Grey, Brown, Black
* Seasonal and temporal context: Late afternoon, Overcast
* Use-case relevance: Tourism, Travel, Cultural, Historical, Spiritual
```

---

### ✅ Qwen/Qwen3-VL-2B-Instruct

**Metrics:** 75.2 TPS | 500 tokens

```text
Title: St Michael's Church, Bishop's Stortford, England

Description: A low-angle view of St Michael's Parish Church in Bishop's Stortford, England, showcasing its historic stone walls, arched Gothic windows, and prominent spire. The churchyard is visible with gravestones and a bench, under an overcast sky. The scene is quiet and solemn, with the soft light of a cloudy day highlighting the textures of the ancient building.

Keywords: Bishop's Stortford, British, Christian, Christianity, England, English Church, Europe, Faith, Flint Knapped Wall, Flint Wall, Gothic Window, Grass, Graveyard, Hertfordshire, Parish church, Perpendicular Gothic, St Michael's Church, St Michael's Parish Church, UK, United Kingdom, ancient, arched window, battlements, black, brown, building, car, cemetery, church, church spire, churchyard, clock tower, cloudy, cloudy sky, cross, exterior, gothic architecture, gravestone, grey, headstone, heritage, historic, historic building, landmark, medieval, old, overcast, overcast sky, place of worship, religion, religious building, spire, stone, tombstone, tourism, travel, weathervane, worship, churchyard, church, church spire, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard
```

---

### ✅ mlx-community/Qwen3-VL-2B-Thinking-bf16

**Metrics:** 72 TPS | 500 tokens

```text
Got it, let's tackle this image cataloguing task. First, I need to follow the structure: Title, Description, Keywords, Context. Wait, the user said "Context" is part of the response, but the instructions say "Respond with exactly these three sections: Title, Description, Keywords, Context". Wait, no, the user says "three sections" but the example shows Title, Description, Keywords, Context. Wait, let me check again.

Wait, the user says: "Respond with exactly these three sections: Title: A concise, descriptive title (5–12 words). Description: A factual 1–3 sentence description of what the image shows, covering key subjects, setting, and action. Keywords: 25–50 comma-separated keywords ordered from most specific to most general, covering: ... Context: The image is described as ', St Michael's Parish Church, Bishop's Stortford, England, United Kingdom, UK...'. Existing keywords: ... Taken on 2026-02-07 16:11:23 GMT (at 16:11:23 local time)."

Wait, no, the user's instruction says "Respond with exactly these three sections: Title, Description, Keywords, Context". Wait, no, the user says "three sections" but the example shows Title, Description, Keywords, Context. Wait, maybe a typo. Wait, the user says: "Respond with exactly these three sections: Title: ... Description: ... Keywords: ... Context: ...". So four sections? Wait, no, the user says "three sections" but the example has Title, Description, Keywords, Context. Wait, no, let's check the user's exact instruction:

"Respond with exactly these three sections:

Title: A concise, descriptive title (5–12 words).

Description: A factual 1–3 sentence description of what the image shows, covering key subjects, setting, and action.

Keywords: 25–50 comma-separated keywords ordered from most specific to most general, covering:
- Specific subjects (people, objects, animals, landmarks)
- Setting and location (urban, rural, indoor, outdoor)
- Actions and activities
- Concepts and themes (e.g. teamwork, solitude, celebration)
- Mood and emotion (e.g. serene, dramatic, joyful)
- Visual style (e.g. close-up, wide-angle, aerial,
```

---

### ✅ mlx-community/Qwen2-VL-2B-Instruct-4bit

**Metrics:** 174 TPS | 500 tokens

```text
Title: St Michael's Parish Church, Bishop's Stortford, England, United Kingdom, UK
Description: A Grade I listed building in Bishop's Stortford, England, featuring Perpendicular Gothic architecture and a historic churchyard.
Keywords: St Michael's Church, St Michael's Parish Church, Grade I listed building, Perpendicular Gothic architecture, historic churchyard, Bishop's Stortford, England, UK, United Kingdom, Gothic window, stone walls, church spire, cemetery, headstone, heritage site, religious building, travel destination, historic landmark, cloudy sky, overcast weather, green grass, stone structure, historical significance, outdoor setting, architectural detail, religious context, British culture, England landmarks, travel photography, outdoor photography, historical architecture, religious building, travel photography, outdoor photography, historical architecture, religious building, travel photography, outdoor photography, historical architecture, religious building, travel photography, outdoor photography, historical architecture, religious building, travel photography, outdoor photography, historical architecture, religious building, travel photography, outdoor photography, historical architecture, religious building, travel photography, outdoor photography, historical architecture, religious building, travel photography, outdoor photography, historical architecture, religious building, travel photography, outdoor photography, historical architecture, religious building, travel photography, outdoor photography, historical architecture, religious building, travel photography, outdoor photography, historical architecture, religious building, travel photography, outdoor photography, historical architecture, religious building, travel photography, outdoor photography, historical architecture, religious building, travel photography, outdoor photography, historical architecture, religious building, travel photography, outdoor photography, historical architecture, religious building, travel photography, outdoor photography, historical architecture, religious building, travel photography, outdoor photography, historical architecture, religious building, travel photography, outdoor photography, historical architecture, religious building, travel photography, outdoor photography, historical architecture, religious building, travel photography, outdoor photography, historical architecture, religious building, travel photography, outdoor photography, historical architecture, religious building, travel photography, outdoor photography, historical architecture, religious building, travel photography, outdoor photography, historical architecture, religious building, travel photography, outdoor photography, historical architecture, religious building, travel photography, outdoor photography, historical architecture, religious building, travel photography, outdoor photography, historical architecture, religious building, travel photography, outdoor photography, historical architecture, religious building, travel photography, outdoor photography, historical architecture, religious building, travel photography, outdoor photography, historical architecture, religious building, travel photography, outdoor photography,
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
- **Python Version**: 3.13.9
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
- `mlx`: `0.30.7.dev20260213+c184262d`
- `mlx-lm`: `0.30.7`
- `mlx-metal`: ``
- `mlx-vlm`: `0.3.11`
- `numpy`: `2.4.2`
- `tokenizers`: `0.22.2`
- `transformers`: `5.1.0`

_Report generated on: 2026-02-13 13:18:47 GMT_
