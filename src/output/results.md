# Model Performance Results

_Generated on 2026-02-08 22:32:28 GMT_

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/nanoLLaVA-1.5-4bit` (324.3 tps)
- **💾 Most efficient:** `mlx-community/nanoLLaVA-1.5-4bit` (2.8 GB)
- **⚡ Fastest load:** `HuggingFaceTB/SmolVLM-Instruct` (0.00s)
- **📊 Average TPS:** 58.8 across 31 models

## 📈 Resource Usage

- **Total peak memory:** 654.7 GB
- **Average peak memory:** 21.1 GB
- **Memory efficiency:** 170 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** ✅ B: 2 | 🟡 C: 10 | 🟠 D: 6 | ❌ F: 13

**Average Utility Score:** 39/100

- **Best for cataloging:** `mlx-community/paligemma2-3b-ft-docci-448-bf16` (✅ B, 72/100)
- **Worst for cataloging:** `mlx-community/Molmo-7B-D-0924-8bit` (❌ F, 0/100)

### ⚠️ 19 Models with Low Utility (D/F)

- `Qwen/Qwen3-VL-2B-Instruct`: 🟠 D (40/100) - Mostly echoes context without adding value
- `microsoft/Phi-3.5-vision-instruct`: ❌ F (26/100) - Mostly echoes context without adding value
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (7/100) - Output too short to be useful
- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`: 🟠 D (42/100) - Lacks visual description of image
- `mlx-community/InternVL3-14B-8bit`: ❌ F (29/100) - Mostly echoes context without adding value
- `mlx-community/Molmo-7B-D-0924-8bit`: ❌ F (0/100) - Empty or minimal output
- `mlx-community/Molmo-7B-D-0924-bf16`: ❌ F (0/100) - Empty or minimal output
- `mlx-community/Phi-3.5-vision-instruct-bf16`: ❌ F (26/100) - Mostly echoes context without adding value
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: 🟠 D (44/100) - Mostly echoes context without adding value
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: ❌ F (21/100) - Mostly echoes context without adding value
- `mlx-community/deepseek-vl2-8bit`: 🟠 D (45/100) - Mostly echoes context without adding value
- `mlx-community/gemma-3-27b-it-qat-8bit`: 🟠 D (47/100) - Mostly echoes context without adding value
- `mlx-community/gemma-3n-E4B-it-bf16`: 🟠 D (48/100) - Mostly echoes context without adding value
- `mlx-community/llava-v1.6-mistral-7b-8bit`: ❌ F (25/100) - Mostly echoes context without adding value
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (5/100) - Output too short to be useful
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (23/100) - Lacks visual description of image
- `mlx-community/pixtral-12b-8bit`: ❌ F (34/100) - Mostly echoes context without adding value
- `mlx-community/pixtral-12b-bf16`: ❌ F (31/100) - Mostly echoes context without adding value
- `qnguyen3/nanoLLaVA`: ❌ F (15/100) - Mostly echoes context without adding value

## ⚠️ Quality Issues

- **❌ Failed Models (11):**
  - `microsoft/Florence-2-large-ft` (`Weight Mismatch`)
  - `mlx-community/GLM-4.6V-Flash-6bit` (`Model Error`)
  - `mlx-community/GLM-4.6V-Flash-mxfp4` (`Model Error`)
  - `mlx-community/Idefics3-8B-Llama3-bf16` (`Model Error`)
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit` (`Model Error`)
  - `mlx-community/LFM2-VL-1.6B-8bit` (`Model Error`)
  - `mlx-community/LFM2.5-VL-1.6B-bf16` (`Model Error`)
  - `mlx-community/X-Reasoner-7B-8bit` (`OOM`)
  - `mlx-community/gemma-3n-E2B-4bit` (`No Chat Template`)
  - `mlx-community/paligemma2-3b-pt-896-4bit` (`Model Error`)
  - `prince-canuma/Florence-2-large-ft` (`Model Error`)
- **🔄 Repetitive Output (5):**
  - `microsoft/Phi-3.5-vision-instruct` (token: `phrase: "church exterior, church design..."`)
  - `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16` (token: `phrase: "churchyard, churchyard, church..."`)
  - `mlx-community/Phi-3.5-vision-instruct-bf16` (token: `phrase: "church exterior, church design..."`)
  - `mlx-community/Qwen2-VL-2B-Instruct-4bit` (token: `phrase: "autumnal scenery, autumnal lan..."`)
  - `mlx-community/deepseek-vl2-8bit` (token: `phrase: "english countryside, english c..."`)

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 58.8 | Min: 0 | Max: 324
- **Peak Memory**: Avg: 21 | Min: 2.8 | Max: 73
- **Total Time**: Avg: 28.99s | Min: 2.87s | Max: 115.83s
- **Generation Time**: Avg: 25.71s | Min: 1.98s | Max: 114.59s
- **Model Load Time**: Avg: 3.28s | Min: 0.89s | Max: 11.95s

## 🚨 Failures by Package (Actionable)

<!-- markdownlint-disable MD060 -->

| Package | Failures | Error Types | Affected Models |
|---------|----------|-------------|-----------------|
| `mlx-vlm` | 8 | Model Error, No Chat Template | `mlx-community/GLM-4.6V-Flash-6bit`, `mlx-community/GLM-4.6V-Flash-mxfp4`, `mlx-community/Kimi-VL-A3B-Thinking-8bit`, `mlx-community/LFM2-VL-1.6B-8bit`, `mlx-community/LFM2.5-VL-1.6B-bf16`, `mlx-community/gemma-3n-E2B-4bit`, `mlx-community/paligemma2-3b-pt-896-4bit`, `prince-canuma/Florence-2-large-ft` |
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
- **mlx-community/LFM2-VL-1.6B-8bit** (Model Error)
  - Error: `Model generation failed for mlx-community/LFM2-VL-1.6B-8bit: Image features and image tokens do not match: tokens: 24...`
  - Type: `ValueError`
- **mlx-community/LFM2.5-VL-1.6B-bf16** (Model Error)
  - Error: `Model generation failed for mlx-community/LFM2.5-VL-1.6B-bf16: Image features and image tokens do not match: tokens: ...`
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
> Taken around 2026-02-07 16:11:23 GMT at 16:11:23.
>
> Be factual about visual content.  Include relevant conceptual and emotional keywords where clearly supported by the image.
> ```

**Note:** Results sorted: errors first, then by generation time (fastest to slowest).

**Overall runtime:** 961.95s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Token |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                    |   Error Package |
|:--------------------------------------------------------|--------:|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:----------------------------------|----------------:|
| `microsoft/Florence-2-large-ft`                         |         |                   |                       |                |              |           |             |                  |            |             |                                   |             mlx |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |         |                   |                       |                |              |           |             |                  |            |             |                                   |         mlx-vlm |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |         |                   |                       |                |              |           |             |                  |            |             |                                   |         mlx-vlm |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 |         |                   |                       |                |              |           |             |                  |            |             |                                   |             mlx |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |         |                   |                       |                |              |           |             |                  |            |             |                                   |         mlx-vlm |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |         |                   |                       |                |              |           |             |                  |            |             |                                   |         mlx-vlm |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |         |                   |                       |                |              |           |             |                  |            |             |                                   |         mlx-vlm |
| `mlx-community/X-Reasoner-7B-8bit`                      |         |                   |                       |                |              |           |             |                  |            |             |                                   |             mlx |
| `mlx-community/gemma-3n-E2B-4bit`                       |         |                   |                       |                |              |           |             |                  |            |             |                                   |         mlx-vlm |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |         |                   |                       |                |              |           |             |                  |            |             |                                   |         mlx-vlm |
| `prince-canuma/Florence-2-large-ft`                     |         |                   |                       |                |              |           |             |                  |            |             |                                   |         mlx-vlm |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | 151,645 |               582 |                   440 |          1,022 |        2,537 |       324 |         2.8 |            1.98s |      0.89s |       2.87s | verbose                           |                 |
| `qnguyen3/nanoLLaVA`                                    | 151,645 |               582 |                   145 |            727 |        2,492 |      97.7 |         4.9 |            2.12s |      1.01s |       3.14s |                                   |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |  49,279 |             1,689 |                   136 |          1,825 |        1,126 |       108 |         5.5 |            3.09s |      1.27s |       4.36s |                                   |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |       1 |             1,606 |                     4 |          1,610 |          456 |      47.7 |          12 |            3.91s |      3.24s |       7.15s | ⚠️harness(prompt_template), ...   |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |  49,154 |             1,789 |                   388 |          2,177 |        1,402 |       110 |         5.5 |            5.14s |      1.24s |       6.38s | verbose                           |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |       2 |             3,465 |                   284 |          3,749 |        1,002 |       155 |         9.0 |            5.59s |      1.65s |       7.25s |                                   |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |  49,154 |             1,789 |                   388 |          2,177 |        1,227 |       102 |         5.5 |            5.62s |      1.24s |       6.86s | verbose                           |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |     106 |               853 |                   203 |          1,056 |          889 |      41.2 |          17 |            6.24s |      5.11s |      11.35s |                                   |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |   1,712 |             1,567 |                   500 |          2,067 |          857 |       102 |          18 |            7.21s |      3.27s |      10.48s |                                   |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |       1 |             1,606 |                    20 |          1,626 |          472 |      5.18 |          27 |            7.58s |      4.83s |      12.41s | context-ignored                   |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |   1,312 |             1,567 |                   500 |          2,067 |          894 |      64.3 |          37 |           10.20s |      5.92s |      16.12s |                                   |                 |
| `mlx-community/deepseek-vl2-8bit`                       |   3,947 |             2,573 |                   500 |          3,073 |          846 |      53.0 |          32 |           12.93s |      5.25s |      18.18s | repetitive(phrase: "english co... |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |  24,447 |             1,905 |                   500 |          2,405 |          290 |      45.0 |          60 |           18.42s |     11.95s |      30.37s | repetitive(phrase: "churchyard... |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |     106 |               853 |                   363 |          1,216 |          185 |      22.3 |          19 |           21.21s |      4.89s |      26.10s | verbose                           |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |       2 |             3,019 |                   373 |          3,392 |          261 |      35.0 |          11 |           22.52s |      1.77s |      24.29s |                                   |                 |
| `mlx-community/pixtral-12b-8bit`                        |   1,044 |             3,701 |                   500 |          4,201 |          459 |      31.0 |          16 |           24.51s |      3.12s |      27.63s | verbose                           |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |  68,398 |             3,792 |                   500 |          4,292 |          403 |      33.8 |          15 |           24.52s |      3.05s |      27.58s | verbose                           |                 |
| `mlx-community/InternVL3-14B-8bit`                      | 151,645 |             3,901 |                   369 |          4,270 |          346 |      26.7 |          18 |           25.54s |      3.22s |      28.76s | verbose                           |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      | 128,009 |               562 |                   193 |            755 |          141 |      8.41 |          15 |           27.28s |      2.79s |      30.07s |                                   |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |     106 |               853 |                   297 |          1,150 |          190 |      12.8 |          34 |           28.05s |      6.97s |      35.02s |                                   |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |     476 |             1,606 |                   500 |          2,106 |        1,480 |      16.9 |          12 |           30.97s |      2.90s |      33.86s |                                   |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |       2 |             2,965 |                   472 |          3,437 |          216 |      24.5 |          23 |           33.32s |      3.86s |      37.17s | ⚠️harness(encoding)               |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |         |               0.0 |                   0.0 |            0.0 |            0 |         0 |          48 |           33.91s |      3.15s |      37.06s |                                   |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |         |               0.0 |                   0.0 |            0.0 |            0 |         0 |          41 |           34.25s |      2.12s |      36.37s |                                   |                 |
| `mlx-community/pixtral-12b-bf16`                        |   1,044 |             3,701 |                   500 |          4,201 |          547 |      17.6 |          28 |           35.47s |      4.81s |      40.29s | verbose                           |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |  29,892 |             1,443 |                   500 |          1,943 |        1,018 |      12.1 |          12 |           43.06s |      1.73s |      44.79s | repetitive(phrase: "church ext... |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |  29,892 |             1,443 |                   500 |          1,943 |          995 |      12.0 |          12 |           43.43s |      1.70s |      45.13s | repetitive(phrase: "church ext... |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | 128,009 |               563 |                   174 |            737 |          156 |      3.74 |          25 |           50.47s |      4.19s |      54.66s |                                   |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |  27,848 |            16,855 |                   500 |         17,355 |          375 |      73.7 |         8.3 |           52.09s |      1.56s |      53.65s |                                   |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |  60,353 |            16,857 |                   500 |         17,357 |          312 |      67.1 |         8.3 |           61.80s |      1.68s |      63.49s |                                   |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |  41,674 |            16,866 |                   500 |         17,366 |          152 |       170 |          73 |          114.59s |      1.24s |     115.83s | repetitive(phrase: "autumnal s... |                 |

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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6887, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6853, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<3 lines>...
        trust_remote_code=params.trust_remote_code,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 316, in load
    model = load_model(model_path, lazy, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 245, in load_model
    model.load_weights(list(weights.items()))
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx/python/mlx/nn/layers/base.py", line 191, in load_weights
    raise ValueError(f"Missing {num_missing} parameters: \n{missing}.")
ValueError: Missing 1 parameters:
language_model.lm_head.weight.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6998, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6895, in _run_model_generation
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6925, in _run_model_generation
    output: GenerationResult | SupportsGenerationResult = generate(
                                                          ~~~~~~~~^
        model=model,
        ^^^^^^^^^^^^
    ...<13 lines>...
        **extra_kwargs,
        ^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 624, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 514, in stream_generate
    for n, (token, logprobs) in enumerate(
                                ~~~~~~~~~^
        generate_step(input_ids, model, pixel_values, mask, **kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ):
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 403, in generate_step
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6998, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6950, in _run_model_generation
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6925, in _run_model_generation
    output: GenerationResult | SupportsGenerationResult = generate(
                                                          ~~~~~~~~^
        model=model,
        ^^^^^^^^^^^^
    ...<13 lines>...
        **extra_kwargs,
        ^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 624, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 514, in stream_generate
    for n, (token, logprobs) in enumerate(
                                ~~~~~~~~~^
        generate_step(input_ids, model, pixel_values, mask, **kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ):
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 403, in generate_step
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6998, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6950, in _run_model_generation
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6887, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6853, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<3 lines>...
        trust_remote_code=params.trust_remote_code,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 316, in load
    model = load_model(model_path, lazy, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 245, in load_model
    model.load_weights(list(weights.items()))
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx/python/mlx/nn/layers/base.py", line 200, in load_weights
    raise ValueError(
    ...<2 lines>...
    )
ValueError: Expected shape (4096, 4096) but received shape (1024, 4096) for parameter language_model.layers.0.self_attn.k_proj.weight

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6998, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6895, in _run_model_generation
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6925, in _run_model_generation
    output: GenerationResult | SupportsGenerationResult = generate(
                                                          ~~~~~~~~^
        model=model,
        ^^^^^^^^^^^^
    ...<13 lines>...
        **extra_kwargs,
        ^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 624, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 514, in stream_generate
    for n, (token, logprobs) in enumerate(
                                ~~~~~~~~~^
        generate_step(input_ids, model, pixel_values, mask, **kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ):
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 384, in generate_step
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6998, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6950, in _run_model_generation
    raise ValueError(msg) from gen_known_err
ValueError: Model generation failed for mlx-community/Kimi-VL-A3B-Thinking-8bit: [broadcast_shapes] Shapes (999,2048) and (1,0,2048) cannot be broadcast.
```

</details>

---

### ❌ mlx-community/LFM2-VL-1.6B-8bit

**Status:** Failed (Model Error)
**Error:**

> Model generation failed for mlx-community/LFM2-VL-1.6B-8bit: Image features
> and image tokens do not match: tokens: 249, features 266
**Type:** `ValueError`
**Package:** `mlx-vlm`

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6925, in _run_model_generation
    output: GenerationResult | SupportsGenerationResult = generate(
                                                          ~~~~~~~~^
        model=model,
        ^^^^^^^^^^^^
    ...<13 lines>...
        **extra_kwargs,
        ^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 624, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 514, in stream_generate
    for n, (token, logprobs) in enumerate(
                                ~~~~~~~~~^
        generate_step(input_ids, model, pixel_values, mask, **kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ):
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 384, in generate_step
    embedding_output = model.get_input_embeddings(
        input_ids, pixel_values, mask=mask, **kwargs
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/lfm2_vl/lfm2_vl.py", line 153, in get_input_embeddings
    final_inputs_embeds = self.merge_input_ids_with_image_features(
        image_features, inputs_embeds, input_ids, self.config.image_token_index
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/lfm2_vl/lfm2_vl.py", line 170, in merge_input_ids_with_image_features
    raise ValueError(
        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
    )
ValueError: Image features and image tokens do not match: tokens: 249, features 266

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6998, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6950, in _run_model_generation
    raise ValueError(msg) from gen_known_err
ValueError: Model generation failed for mlx-community/LFM2-VL-1.6B-8bit: Image features and image tokens do not match: tokens: 249, features 266
```

</details>

---

### ❌ mlx-community/LFM2.5-VL-1.6B-bf16

**Status:** Failed (Model Error)
**Error:**

> Model generation failed for mlx-community/LFM2.5-VL-1.6B-bf16: Image
> features and image tokens do not match: tokens: 249, features 266
**Type:** `ValueError`
**Package:** `mlx-vlm`

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6925, in _run_model_generation
    output: GenerationResult | SupportsGenerationResult = generate(
                                                          ~~~~~~~~^
        model=model,
        ^^^^^^^^^^^^
    ...<13 lines>...
        **extra_kwargs,
        ^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 624, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 514, in stream_generate
    for n, (token, logprobs) in enumerate(
                                ~~~~~~~~~^
        generate_step(input_ids, model, pixel_values, mask, **kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ):
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 384, in generate_step
    embedding_output = model.get_input_embeddings(
        input_ids, pixel_values, mask=mask, **kwargs
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/lfm2_vl/lfm2_vl.py", line 153, in get_input_embeddings
    final_inputs_embeds = self.merge_input_ids_with_image_features(
        image_features, inputs_embeds, input_ids, self.config.image_token_index
    )
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/lfm2_vl/lfm2_vl.py", line 170, in merge_input_ids_with_image_features
    raise ValueError(
        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
    )
ValueError: Image features and image tokens do not match: tokens: 249, features 266

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6998, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6950, in _run_model_generation
    raise ValueError(msg) from gen_known_err
ValueError: Model generation failed for mlx-community/LFM2.5-VL-1.6B-bf16: Image features and image tokens do not match: tokens: 249, features 266
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6925, in _run_model_generation
    output: GenerationResult | SupportsGenerationResult = generate(
                                                          ~~~~~~~~^
        model=model,
        ^^^^^^^^^^^^
    ...<13 lines>...
        **extra_kwargs,
        ^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 624, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 514, in stream_generate
    for n, (token, logprobs) in enumerate(
                                ~~~~~~~~~^
        generate_step(input_ids, model, pixel_values, mask, **kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ):
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 410, in generate_step
    mx.eval([c.state for c in prompt_cache])
    ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: [metal::malloc] Attempting to allocate 135699660800 bytes which is greater than the maximum allowed buffer size of 86586540032 bytes.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6998, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6955, in _run_model_generation
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6998, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6899, in _run_model_generation
    formatted_prompt: str | list[Any] = apply_chat_template(
                                        ~~~~~~~~~~~~~~~~~~~^
        processor=processor,
        ^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
        num_images=1,
        ^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/prompt_utils.py", line 563, in apply_chat_template
    return get_chat_template(processor, messages, add_generation_prompt)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/prompt_utils.py", line 444, in get_chat_template
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6925, in _run_model_generation
    output: GenerationResult | SupportsGenerationResult = generate(
                                                          ~~~~~~~~^
        model=model,
        ^^^^^^^^^^^^
    ...<13 lines>...
        **extra_kwargs,
        ^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 624, in generate
    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 514, in stream_generate
    for n, (token, logprobs) in enumerate(
                                ~~~~~~~~~^
        generate_step(input_ids, model, pixel_values, mask, **kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ):
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate.py", line 403, in generate_step
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6998, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6950, in _run_model_generation
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6887, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6853, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<3 lines>...
        trust_remote_code=params.trust_remote_code,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 326, in load
    processor = load_processor(model_path, True, eos_token_ids=eos_token_id, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 399, in load_processor
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6998, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6895, in _run_model_generation
    raise ValueError(error_details) from load_err
ValueError: Model loading failed: RobertaTokenizer has no attribute additional_special_tokens
```

</details>

---

### ✅ mlx-community/nanoLLaVA-1.5-4bit

**Metrics:** 324 TPS | 440 tokens

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

**Metrics:** 97.7 TPS | 145 tokens

```text
The image portrays a serene, late afternoon scene at St Michael's Parish Church in Bishop's Stortford, England. The church is a prominent example of Perpendicular Gothic architecture, with its current structure dating largely from the early 15th century. The image captures the intricate details of the church's traditional flint-knapped walls and stone-dressed windows, set against the backdrop of its historic churchyard. The soft, even light of the overcast day accentuates the textures and formidable presence of this centuries-old landmark. The image is taken around 2026-02-07 16:11:23 GMT at 16:11:23.
```

---

### ✅ mlx-community/SmolVLM2-2.2B-Instruct-mlx

**Metrics:** 108 TPS | 136 tokens

```text
 Title: St Michael's Parish Church, Bishop's Stortford, England, United Kingdom, UK
St Michael's Parish Church in Bishop's Stortford, England, is pictured here on a late afternoon under a cool February sky. This Grade I listed building is a prominent example of Perpendicular Gothic architecture, with its current structure dating largely from the early 15th century. The photograph captures the intricate details of the church's traditional flint-knapped walls and stone-dressed windows, set against the backdrop of its historic churchyard. The soft, even light of the overcast day accentuates the textures and formidable presence of this centuries-old landmark.
```

---

### ✅ mlx-community/paligemma2-10b-ft-docci-448-6bit

**Metrics:** 47.7 TPS | 4 tokens

```text
Be concise.
```

---

### ✅ HuggingFaceTB/SmolVLM-Instruct

**Metrics:** 110 TPS | 388 tokens

```text
 Image Description:

The image depicts St Michael's Parish Church, a prominent example of Perpendicular Gothic architecture, located in Bishop's Stortford, England. The church is a Grade I listed building, indicating its historical and architectural significance. The photograph captures the intricate details of the church's traditional flint-knapped walls and stone-dressed windows, set against the backdrop of its historic churchyard.

The church's exterior is characterized by its flint-knapped walls and stone-dressed windows, which are typical of the Perpendicular Gothic architectural style. The windows are tall and narrow, with pointed arches and stained glass, which adds to the church's Gothic aesthetic. The church's exterior is also adorned with battlements, which are characteristic of medieval defensive architecture.

The churchyard is expansive and well-maintained, with a variety of gravestones and headstones, indicating the church's long history and the presence of numerous burials. The churchyard is a serene and peaceful setting, with a few trees and shrubs adding to the natural beauty of the scene.

The church's spire, which is visible in the background, is a prominent feature of the church's architecture. The spire is tall and pointed, with a distinctive design that is characteristic of Gothic architecture. The spire is topped with a weathervane, which is used to indicate the wind direction.

The image is taken from a slightly elevated angle, providing a clear view of the church's exterior and the churchyard. The lighting is soft and even, which accentuates the textures and details of the church's architecture.

In summary, the image captures the historical and architectural significance of St Michael's Parish Church, with its flint-knapped walls, stone-dressed windows, battlements, and expansive churchyard. The image provides a detailed and informative look at the church's exterior and its surroundings.
```

---

### ✅ mlx-community/Ministral-3-3B-Instruct-2512-4bit

**Metrics:** 155 TPS | 284 tokens

```text
**Title:**
St Michael’s Parish Church – Gothic Heritage in England

---

**Description:**
The image depicts an exterior view of a historic parish church with distinct Perpendicular Gothic architecture. The structure features flint-knapped walls, stone-dressed windows with intricate tracery, and a prominent clock tower topped with a spire. The churchyard includes weathered gravestones and a serene, grassy setting under an overcast sky.

---

**Keywords:**
ancient building, arched window, battlements, British heritage, church exterior, church spire, churchyard, cloudy sky, clock tower, cold tones, decorative stonework, detailed Gothic architecture, England, European heritage, faith-based landmark, flint wall, gravestone, grey lighting, historic church, heritage site, medieval architecture, mossy stone, overcast day, place of worship, Perpendicular Gothic, religious architecture, serene atmosphere, spire, stone construction, tower, UK, weathered tombstones, wind vane, worship site, wide-angle shot

---
**Context:**
*(Note: The provided context includes a fictional date for the photo, which is not verifiable. The actual date in the image metadata is fictionalized as 2026-02-07. The visual content aligns with the description of a historic English church, emphasizing architectural and cultural significance.)*
```

---

### ✅ mlx-community/SmolVLM-Instruct-bf16

**Metrics:** 102 TPS | 388 tokens

```text
 Image Description:

The image depicts St Michael's Parish Church, a prominent example of Perpendicular Gothic architecture, located in Bishop's Stortford, England. The church is a Grade I listed building, indicating its historical and architectural significance. The photograph captures the intricate details of the church's traditional flint-knapped walls and stone-dressed windows, set against the backdrop of its historic churchyard.

The church's exterior is characterized by its flint-knapped walls and stone-dressed windows, which are typical of the Perpendicular Gothic architectural style. The windows are tall and narrow, with pointed arches and stained glass, which adds to the church's Gothic aesthetic. The church's exterior is also adorned with battlements, which are characteristic of medieval defensive architecture.

The churchyard is expansive and well-maintained, with a variety of gravestones and headstones, indicating the church's long history and the presence of numerous burials. The churchyard is a serene and peaceful setting, with a few trees and shrubs adding to the natural beauty of the scene.

The church's spire, which is visible in the background, is a prominent feature of the church's architecture. The spire is tall and pointed, with a distinctive design that is characteristic of Gothic architecture. The spire is topped with a weathervane, which is used to indicate the wind direction.

The image is taken from a slightly elevated angle, providing a clear view of the church's exterior and the churchyard. The lighting is soft and even, which accentuates the textures and details of the church's architecture.

In summary, the image captures the historical and architectural significance of St Michael's Parish Church, with its flint-knapped walls, stone-dressed windows, battlements, and expansive churchyard. The image provides a detailed and informative look at the church's exterior and its surroundings.
```

---

### ✅ mlx-community/gemma-3n-E4B-it-bf16

**Metrics:** 41.2 TPS | 203 tokens

```text
## Image Caption:

**Title:** Ancient St. Michael's Church in Stortford, UK

**Description:** A detailed, low-angle view of the historic St. Michael's Church in Bishop's Stortford, England, showcasing its weathered stone architecture and prominent tower under an overcast sky.

**Keywords:**

Church, St. Michael's Church, Bishop's Stortford, England, UK, historic, architecture, stone, Gothic, tower, spire, weathered, building, England, United Kingdom, medieval, ancient, exterior, facade, windows, buttresses, graveyard, gravestones, overcast, sky, historical, religious, landmark, England, British, countryside, old, stone building, churchyard, steeple, brick, pointed arch, craftsmanship, heritage, preservation, England, rural, traditional, timeless, solid, imposing, detailed, textured, vertical, horizontal, low angle, cloudy, grey, serene, enduring, classic, picturesque, England.
```

---

### ✅ jqlive/Kimi-VL-A3B-Thinking-2506-6bit

**Metrics:** 102 TPS | 500 tokens

```text
◁think▷Got it, let's tackle this. First, I need to create a title, description, and keywords for the image of St Michael's Parish Church in Bishop's Stortford, England.

Starting with the title: it should be concise and descriptive. The image shows the church, so maybe "St Michael's Parish Church, Bishop's Stortford, England" but that's too long. Maybe "St Michael's Parish Church, Bishop's Stortford, England" is okay, but let's make it a bit more engaging. Wait, the existing keywords include "St Michael's Parish Church" and "St Michael's Church". Maybe "St Michael's Parish Church, Bishop's Stortford, England" is sufficient.

Description: Need to cover key subjects (church, architecture), setting (historic, churchyard, outdoor), action (no people, just the building), concepts (heritage, architecture, history), mood (serene, historic). So: "St Michael's Parish Church in Bishop's Stortford, England, is a historic Grade I listed building showcasing Perpendicular Gothic architecture. The image captures its stone walls, arched windows, and churchyard under an overcast sky, highlighting its medieval heritage and architectural details."

Keywords: Need to list 25-50 keywords. Let's go through the existing ones and add more. Existing keywords include Bishop's Stortford, England, Christian, Christianity, England, English Church, Europe, Faith, Flint Knapped Wall, Flint Wall, Gothic Window, Grass, Graveyard, Hertfordshire, Parish church, Perpendicular Gothic, St Michael's Church, St Michael's Parish Church, UK, United Kingdom, ancient, arched window, architecture, battlements, black, brown, building, car, cemetery, church, church spire, churchyard, clock tower, cloudy, cloudy sky, cross, exterior, gothic architecture, gravestone, grey, headstone, heritage, historic, historic building, landmark, medieval, old, overcast, overcast sky, place of worship, religion, religious building, spire, stone, tombstone, tourism, tower, travel, weathervane, worship. Let's add more specific ones like "stone walls", "arched windows", "clock tower", "churchyard", "medieval architecture", "Grade I listed", "Perpendicular Gothic", "overcast lighting", "historic landmark", "religious site", "arch
```

---

### ✅ mlx-community/paligemma2-10b-ft-docci-448-bf16

**Metrics:** 5.18 TPS | 20 tokens

```text
Be specific about visual content. Include relevant geographic and temporal keywords where clearly supported by the image.
```

---

### ✅ mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

**Metrics:** 64.3 TPS | 500 tokens

```text
◁think▷Got it, let's tackle this. First, I need to create a title, description, and keywords for the image of St Michael's Parish Church in Bishop's Stortford, England.

Starting with the title: it should be concise and descriptive. The image shows the church, so maybe "St Michael's Parish Church in Bishop's Stortford, England, on a Cloudy Late Afternoon". That captures the location, the building, and the time of day.

For the description: need to cover key subjects (the church, its architecture), setting (historic, churchyard, outdoor), action (no people, just the building), concepts (Perpendicular Gothic architecture, historic landmark), mood (serene, historic), visual style (wide-angle, architectural photography), colors (neutral tones, stone, green grass), seasonal (February, overcast), use-case (travel, tourism, heritage). So the description should mention the church's architecture, location, time of day, and key features like flint walls, Gothic windows, churchyard.

Keywords: need 25-50, ordered from specific to general. Let's list specific subjects: St Michael's Parish Church, Bishop's Stortford, England, UK, Christian, Christianity, Perpendicular Gothic architecture, Grade I listed building, flint-knapped walls, stone-dressed windows, churchyard, graveyard, tombstones, church spire, clock tower, weathervane. Then setting: outdoor, historic, rural (since it's in a churchyard, maybe rural setting). Actions: no people, so maybe not actions. Concepts: heritage, tourism, historic landmark, religious building. Mood: serene, solemn. Visual style: wide-angle, architectural photography. Colors: grey, brown, green (grass), overcast. Temporal: February, late afternoon. Use-case: travel, tourism, heritage site. Also include existing keywords from the context: Christian, England, Hertfordshire, Europe, Faith, etc. Need to order them from most specific to most general. Let's structure:

Specific subjects: St Michael's Parish Church, Bishop's Stortford, England, UK, Christian, Christianity, Hertfordshire, Europe, Faith, Perpendicular Gothic architecture, Grade I listed building, flint-knapped walls, stone-dressed windows, churchyard, graveyard, tombstones, church spire, clock tower, weathervane, weathervane, over
```

---

### ✅ mlx-community/deepseek-vl2-8bit

**Metrics:** 53 TPS | 500 tokens

```text
**Title:** A Stunning View of St. Michael's Church in Bishop's Stortford, UK

**Description:** The image shows the historic St. Michael's Parish Church in Bishop's Stortford, a prime example of English Perpendicular Gothic architecture. The building is constructed from a mix of flint and stone, with a tall, pointed spire and large, arched windows. The structure is set against a backdrop of a cloudy, overcast sky, with a well-manicured lawn in the foreground. The scene exudes a serene and historical atmosphere, with the building's intricate stonework and gothic details standing out against the overcast sky.

**Keywords:** St. Michael's Church, Bishop's Stortford, UK, English Heritage, gothic architecture, English Parish Church, historic building, English Heritage, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English Countryside, English
```

---

### ✅ mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

**Metrics:** 45 TPS | 500 tokens

```text
Alright, let's get to work. I've got an image of St Michael's Parish Church in Bishop's Stortford, and I need to generate some metadata for it. First things first, I need a title. It has to be concise, descriptive, and between five and twelve words. "St Michael's Parish Church in Bishop's Stortford, England" is a good start, but I can do better. "St Michael's Church, Bishop's Stortford, England" is even more concise. Perfect.

Next, the description. It needs to be factual, one to three sentences, covering the key elements. I'll start with the obvious: the church itself. Then, I'll add the setting and the mood. The image is a wide shot, so I'll mention that. Okay, here we go: "A wide-angle photograph of St Michael's Parish Church in Bishop's Stortford, England. The stone church, with its tall spire and Gothic architecture, stands against a backdrop of a cloudy sky and a grassy churchyard. The overcast light highlights the texture of the ancient stone walls." That's solid.

Now for the keywords. This is where I can really dig in. I'll start with the most specific and work my way up. I'll need to cover subjects, setting, actions, concepts, mood, visual style, colors, season, and use-case. Let's see... I'll start with the specific subjects: St Michael's Church, Bishop's Stortford, England, UK, flint-knapped walls, stone-dressed windows, church spire, churchyard, graveyard, gravestone, headstone, tombstone, cross, weathervane, clock tower, car, and the date. Then the setting: outdoor, rural, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard
```

---

### ✅ mlx-community/gemma-3-27b-it-qat-4bit

**Metrics:** 22.3 TPS | 363 tokens

```text
## Title: St Michael's Parish Church, Bishop's Stortford, England – Gothic Architecture

## Description:
This photograph depicts the exterior of St Michael's Parish Church in Bishop's Stortford, England, a Grade I listed building showcasing Perpendicular Gothic architecture. The church is framed by a historic churchyard with numerous gravestones, and is photographed under a cloudy, late afternoon sky. A clock tower and spire are prominent features of the building’s design.

## Keywords:
St Michael's Parish Church, Bishop's Stortford, Hertfordshire, England, UK, United Kingdom, Perpendicular Gothic, Gothic architecture, flint knapped wall, arched window, church spire, clock tower, churchyard, graveyard, gravestone, headstone, tombstone, historic building, landmark, medieval architecture, religious building, place of worship, church, exterior, stone, building, tower, spire, overcast sky, cloudy, grey tones, brown tones, black, winter, February, late afternoon, soft light, even lighting, texture, detail, heritage, tourism, travel, European architecture, British architecture, Christian architecture, Christianity, faith, religious site, ancient, old, historic, solemn, peaceful, quiet, stillness, cultural heritage, architectural detail, stone masonry, church clock, weathervane, battlements, wide-angle, exterior view, overcast day, cultural landmark, stone wall, religious monument, architectural photography, landscape, urban landscape, cultural tourism, historical site, European travel, UK travel, stone structure, religious history, architectural style, cultural icon, landmark architecture, exterior architecture, stone building, religious art, historical architecture, cultural destination, stone texture, architectural design, religious building exterior, stone facade, architectural heritage.
```

---

### ✅ mlx-community/llava-v1.6-mistral-7b-8bit

**Metrics:** 35 TPS | 373 tokens

```text
Title: St Michael's Parish Church, Bishop's Stortford, England, United Kingdom, UK

Description: A photograph captures the intricate details of St Michael's Parish Church, a Grade I listed building in Bishop's Stortford, England, UK. The image showcases the traditional flint-knapped walls and stone-dressed windows, set against the backdrop of the historic churchyard. The soft, even light of the overcast day accentuates the textures and formidable presence of this centuries-old landmark.

Keywords: St Michael's Parish Church, Bishop's Stortford, English Church, Europe, Faith, Christianity, England, Gothic architecture, Heritage, Historic, Landmark, Medieval, Old, Perpendicular Gothic, Religion, Religious building, St Michael's Church, St Michael's Parish Church, United Kingdom, UK, Architecture, Church, Churchyard, Clock tower, Overcast sky, Stone, Tombstone, Tourism, Worship.

Context: The image is described as 'St Michael's Parish Church in Bishop's Stortford, England, UK, pictured here on a late afternoon under a cool February sky. This Grade I listed building is a prominent example of Perpendicular Gothic architecture, with its current structure dating largely from the early 15th century. The photograph captures the intricate details of the church's traditional flint-knapped walls and stone-dressed windows, set against the backdrop of its historic churchyard. The soft, even light of the overcast day accentuates the textures and formidable presence of this centuries-old landmark.'
```

---

### ✅ mlx-community/pixtral-12b-8bit

**Metrics:** 31 TPS | 500 tokens

```text
### Title
St Michael's Parish Church, Bishop's Stortford

### Description
St Michael's Parish Church in Bishop's Stortford, England, is pictured here on a late afternoon under a cool February sky. This Grade I listed building is a prominent example of Perpendicular Gothic architecture, with its current structure dating largely from the early 15th century. The photograph captures the intricate details of the church's traditional flint-knapped walls and stone-dressed windows, set against the backdrop of its historic churchyard.

### Keywords
St Michael's Parish Church, Bishop's Stortford, British, Christian, Christianity, England, English Church, Europe, Faith, Flint Knapped Wall, Flint Wall, Gothic Window, Grass, Graveyard, Hertfordshire, Parish church, Perpendicular Gothic, St Michael's Church, St Michael's Parish Church, UK, United Kingdom, ancient, arched window, architecture, battlements, black, brown, building, cemetery, church, church spire, churchyard, clock tower, cloudy, cloudy sky, cross, exterior, gothic architecture, gravestone, grey, headstone, heritage, historic, historic building, landmark, medieval, old, overcast, overcast sky, place of worship, religion, religious building, spire, stone, tombstone, tourism, tower, travel, weathervane, worship, February, late afternoon, historic churchyard, flint-knapped walls, stone-dressed windows, Perpendicular Gothic architecture, centuries-old landmark, soft light, textures, formidable presence, traditional, traditional architecture, religious heritage, historic site, architectural detail, stone masonry, church architecture, English heritage, English history, English landmarks, English tourism, British heritage, British history, British landmarks, British tourism, religious tourism, historic tourism, architectural tourism, winter, winter landscape, winter scene, winter architecture, winter church, winter graveyard, winter travel, winter tourism, overcast sky, cloudy day, grey sky, grey tones, stone structure, stone building, stone church, stone architecture, stone masonry, stonework, flint stone, flint wall, flint architecture, flint building, flint church, flint structure, flint masonry, flint stonework, flint construction, flint design, flint texture, flint pattern, flint appearance, flint look,
```

---

### ✅ mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX

**Metrics:** 33.8 TPS | 500 tokens

```text
Here are my reasoning steps:
The user asks: "Analyze this image and provide structured metadata for image cataloguing. Respond with exactly these three sections: Title: A concise, descriptive title (5–12 words). Description: A factual 1–3 sentence description of what the image shows, covering key subjects, setting, and action. Keywords: 25–50 comma-separated keywords ordered from most specific to most general, covering: - Specific subjects (people, objects, animals, landmarks) - Setting and location (urban, rural, indoor, outdoor) - Actions and activities - Concepts and themes (e.g. teamwork, solitude, celebration) - Mood and emotion (e.g. serene, dramatic, joyful) - Visual style (e.g. close-up, wide-angle, aerial, silhouette, bokeh) - Colors and lighting (e.g. golden hour, blue tones, high-key) - Seasonal and temporal context - Use-case relevance (e.g. business, travel, food, editorial)."

Then they provide context: "The image is described as ', St Michael's Parish Church, Bishop's Stortford, England, United Kingdom, UK St Michael's Parish Church in Bishop's Stortford, England, is pictured here on a late afternoon under a cool February sky. This Grade I listed building is a prominent example of Perpendicular Gothic architecture, with its current structure dating largely from the early 15th century. The photograph captures the intricate details of the church's traditional flint-knapped walls and stone-dressed windows, set against the backdrop of its historic churchyard. The soft, even light of the overcast day accentuates the textures and formidable presence of this centuries-old landmark.'."

They also provide existing keywords: a long list.

We need to produce Title (5-12 words), Description (1-3 sentences), Keywords (25-50 keywords). Must be factual about visual content, include relevant conceptual and emotional keywords where clearly supported.

We need to incorporate the context and existing keywords but we can produce a new set of keywords. Must be 25-50 keywords, ordered from most specific to most general. So we need to list them in descending order of specificity: start with specific subjects like "St Michael's Parish Church", "Bishop's Stortford", "flint-knapped walls", "Perpendicular Gothic
```

---

### ✅ mlx-community/InternVL3-14B-8bit

**Metrics:** 26.7 TPS | 369 tokens

```text
Title: Gothic Church with Battlements and Graveyard

Description: The image shows St Michael's Parish Church in Bishop's Stortford, England, featuring a tall spire, arched windows, and a historic graveyard with gravestones in the foreground.

Keywords: Bishop's Stortford, British, Christian, Christianity, England, English Church, Europe, Faith, Flint Knapped Wall, Flint Wall, Gothic Window, Grass, Graveyard, Hertfordshire, Parish church, Perpendicular Gothic, St Michael's Church, St Michael's Parish Church, UK, United Kingdom, ancient, arched window, architecture, battlements, black, brown, building, cloudy, cloudy sky, cross, exterior, gothic architecture, gravestone, grey, headstone, heritage, historic, historic building, landmark, medieval, old, overcast, overcast sky, place of worship, religion, religious building, spire, stone, tombstone, tourism, tower, travel, weathervane, worship, serene, dramatic, wide-angle, muted tones, historical, solemn, architectural details, weathered stone

Context: The image is described as 'St Michael's Parish Church, Bishop's Stortford, England, United Kingdom, UK. St Michael's Parish Church in Bishop's Stortford, England, is pictured here on a late afternoon under a cool February sky. This Grade I listed building is a prominent example of Perpendicular Gothic architecture, with its current structure dating largely from the early 15th century. The photograph captures the intricate details of the church's traditional flint-knapped walls and stone-dressed windows, set against the backdrop of its historic churchyard. The soft, even light of the overcast day accentuates the textures and formidable presence of this centuries-old landmark.'
```

---

### ✅ mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

**Metrics:** 8.41 TPS | 193 tokens

```text
**Title:** St Michael's Parish Church, Bishop's Stortford, England

**Description:** The image depicts St Michael's Parish Church in Bishop's Stortford, England, showcasing its historic architecture and surrounding environment. The church's stone structure and intricate details are prominently featured against a cloudy sky.

**Keywords:**

* Specific subjects: church, building, architecture, stone, window, clock tower, spire, cross, graveyard, tombstone
* Setting and location: England, UK, Europe, outdoor, rural
* Actions and activities: none
* Concepts and themes: history, heritage, tradition, faith, religion
* Mood and emotion: serene, peaceful, contemplative
* Visual style: wide-angle, detailed, high-contrast
* Colors and lighting: muted, earthy tones, cloudy sky
* Seasonal and temporal context: winter, late afternoon
* Use-case relevance: cultural, historical, educational, spiritual
```

---

### ✅ mlx-community/gemma-3-27b-it-qat-8bit

**Metrics:** 12.8 TPS | 297 tokens

```text
## Title: St Michael's Parish Church, Bishop's Stortford, England – February View

## Description:
This photograph depicts St Michael's Parish Church in Bishop's Stortford, England, a Grade I listed building showcasing Perpendicular Gothic architecture. The church is framed against a cloudy February sky, with its flint-knapped walls and stone windows prominently visible. Gravestones dot the surrounding churchyard, adding to the historic atmosphere of the scene.

## Keywords:
St Michael's Parish Church, Bishop's Stortford, Perpendicular Gothic, flint knapped wall, Gothic window, church spire, churchyard, gravestone, headstone, tombstone, medieval architecture, historic building, landmark, place of worship, religious building, England, Hertfordshire, UK, United Kingdom, British, Christian, Christianity, church, exterior, arched window, battlements, clock tower, tower, spire, stone, grey, brown, overcast sky, cloudy, February, winter, historic, heritage, tourism, travel, religious site, European architecture, place of interest, building exterior, overcast, texture, detail, solemn, quiet, stillness, tradition, faith, worship, cultural heritage, architectural detail, wide-angle, daylight, soft light, even lighting, stone wall, church grounds, graveyard, ancient, old, cultural landmark, European travel, architectural photography, exterior view, religious architecture, stone masonry, cultural tourism.
```

---

### ✅ mlx-community/paligemma2-3b-ft-docci-448-bf16

**Metrics:** 16.9 TPS | 500 tokens

```text
A low-angle, three-quarters view of the side of St Michael's Parish Church in Bishop's Stortford, England. The church is a light brown stone with a brick and stone exterior. The exterior is made of light brown stone with a row of five windows with a pointed top. The windows are divided into three by a thin, dark brown wooden frame. The windows are a light brown with a dark brown tint. The windows are lined up in a row, with the left side having five windows, the middle having four, and the right side having three. The windows are all in the shape of a rectangle with a pointed top. The windows are lined up in a row, with the left side having five windows, the middle having four, and the right side having three. The windows are a light brown with a dark brown tint. The windows are in the shape of a rectangle with a pointed top. The windows are lined up in a row, with the left side having five windows, the middle having four, and the right side having three. The windows are a light brown with a dark brown tint. The windows are in the shape of a rectangle with a pointed top. The windows are lined up in a row, with the left side having five windows, the middle having four, and the right side having three. The windows are a light brown with a dark brown tint. The windows are in the shape of a rectangle with a pointed top. The windows are lined up in a row, with the left side having five windows, the middle having four, and the right side having three. The windows are a light brown with a dark brown tint. The windows are in the shape of a rectangle with a pointed top. The windows are lined up in a row, with the left side having five windows, the middle having four, and the right side having three. The windows are a light brown with a dark brown tint. The windows are in the shape of a rectangle with a pointed top. The windows are lined up in a row, with the left side having five windows, the middle having four, and the right side having three. The windows are a light brown with a dark brown tint. The windows are in the shape of a rectangle with a pointed top. The windows are lined up in a row, with the left side having five windows, the middle having four, and the right side having three. The windows are a light brown with a
```

---

### ✅ mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

**Metrics:** 24.5 TPS | 472 tokens

```text
Title:ĠStĠMichael'sĠParishĠChurch,ĠBishop'sĠStortfordĊĊDescription:ĠTheĠimageĠshowsĠStĠMichael'sĠParishĠChurchĠinĠBishop'sĠStortford,ĠEngland,ĠaĠhistoricĠGradeĠIĠlistedĠbuildingĠfeaturingĠPerpendicularĠGothicĠarchitecture.ĠTheĠchurch'sĠflint-knappedĠwallsĠandĠstone-trimmedĠwindowsĠareĠprominentlyĠdisplayed,ĠsetĠagainstĠaĠbackdropĠofĠitsĠhistoricĠchurchyard.ĊĊKeywords:ĠStĠMichael'sĠParishĠChurch,ĠBishop'sĠStortford,ĠPerpendicularĠGothicĠarchitecture,ĠGradeĠIĠlistedĠbuilding,Ġflint-knappedĠwalls,Ġstone-trimmedĠwindows,Ġchurchyard,ĠhistoricĠchurch,ĠmedievalĠarchitecture,ĠGothicĠwindow,ĠchurchĠspire,Ġbattlements,Ġgravestone,Ġheadstone,Ġtombstone,Ġcemetery,ĠplaceĠofĠworship,ĠreligiousĠbuilding,ĠChristian,ĠChristianity,Ġfaith,Ġheritage,Ġlandmark,Ġancient,Ġold,ĠovercastĠsky,Ġcloudy,Ġgrey,Ġbrown,Ġblack,Ġstone,Ġgrass,Ġexterior,Ġtourism,Ġtravel,Ġarchitecture,ĠhistoricĠbuilding,ĠEngland,ĠUK,ĠUnitedĠKingdom,ĠEurope,ĠBritish,ĠHertfordshire,Ġchurch,Ġspire,Ġweathervane,ĠclockĠtower,Ġtower,Ġcar,Ġcross,Ġreligion,Ġworship,Ġhistoric,Ġmedieval,Ġovercast,ĠcloudyĠsky,ĠplaceĠofĠworship,ĠreligiousĠbuilding,ĠChristian,ĠChristianity,Ġfaith,Ġheritage,Ġlandmark,Ġancient,Ġold,ĠovercastĠsky,Ġcloudy,Ġgrey,Ġbrown,Ġblack,Ġstone,Ġgrass,Ġexterior,Ġtourism,Ġtravel,Ġarchitecture,ĠhistoricĠbuilding,ĠEngland,ĠUK,ĠUnitedĠKingdom,ĠEurope,ĠBritish,ĠHertfordshire,Ġchurch,Ġspire,Ġweathervane,ĠclockĠtower,Ġtower,Ġcar,Ġcross,Ġreligion,ĠworshipĊĊContext:ĠTheĠimageĠisĠdescribedĠasĠStĠMichael'sĠParishĠChurch,ĠBishop'sĠStortford,ĠEngland,ĠUnitedĠKingdom,ĠUK.ĠStĠMichael'sĠParishĠChurchĠinĠBishop'sĠStortford,ĠEngland,ĠisĠpicturedĠhereĠonĠaĠlateĠafternoonĠunderĠaĠcoolĠFebruaryĠsky.ĠThisĠGradeĠIĠlistedĠbuildingĠisĠaĠprominentĠexampleĠofĠPerpendicularĠGothicĠarchitecture,ĠwithĠitsĠcurrentĠstructureĠdatingĠlargelyĠfromĠtheĠearlyĠ15thĠcentury.ĠTheĠphotographĠcapturesĠtheĠintricateĠdetailsĠofĠtheĠchurch'sĠtraditionalĠflint-knappedĠwallsĠandĠstone-trimmedĠwindows,ĠsetĠagainstĠtheĠbackdropĠofĠitsĠhistoricĠchurchyard.ĠTheĠsoft,ĠevenĠlightĠofĠtheĠovercastĠdayĠaccentuatesĠtheĠtexturesĠandĠformidableĠpresenceĠofĠthisĠcenturies-oldĠlandmark.
```

---

### ✅ mlx-community/Molmo-7B-D-0924-bf16

**Metrics:** 0.0 TPS | 0 tokens

```text

```

---

### ✅ mlx-community/Molmo-7B-D-0924-8bit

**Metrics:** 0.0 TPS | 0 tokens

```text

```

---

### ✅ mlx-community/pixtral-12b-bf16

**Metrics:** 17.6 TPS | 500 tokens

```text
### Title
St Michael's Parish Church, Bishop's Stortford

### Description
St Michael's Parish Church in Bishop's Stortford, England, is pictured here on a late afternoon under a cool February sky. This Grade I listed building is a prominent example of Perpendicular Gothic architecture, with its current structure dating largely from the early 15th century. The photograph captures the intricate details of the church's traditional flint-knapped walls and stone-dressed windows, set against the backdrop of its historic churchyard.

### Keywords
St Michael's Parish Church, Bishop's Stortford, British, Christian, Christianity, England, English Church, Europe, Faith, Flint Knapped Wall, Flint Wall, Gothic Window, Grass, Graveyard, Hertfordshire, Parish church, Perpendicular Gothic, St Michael's Church, St Michael's Parish Church, UK, United Kingdom, ancient, arched window, architecture, battlements, black, brown, building, cemetery, church, church spire, churchyard, clock tower, cloudy, cloudy sky, cross, exterior, gothic architecture, gravestone, grey, headstone, heritage, historic, historic building, landmark, medieval, old, overcast, overcast sky, place of worship, religion, religious building, spire, stone, tombstone, tourism, tower, travel, weathervane, worship, February, late afternoon, historic churchyard, flint-knapped walls, stone-dressed windows, Perpendicular Gothic architecture, Grade I listed building, centuries-old landmark, soft light, overcast day, textures, formidable presence, traditional, traditional architecture, religious heritage, English heritage, historic site, architectural detail, stone masonry, church architecture, church spire, church tower, church clock, church cross, church weathervane, church graveyard, church tombstone, church headstone, church gravestone, church grass, churchyard grass, churchyard tombstone, churchyard headstone, churchyard gravestone, churchyard grass, churchyard stone, churchyard cross, churchyard heritage, churchyard tourism, churchyard travel, churchyard worship, churchyard place of worship, churchyard religious building, churchyard historic building, churchyard historic site, churchyard landmark, churchyard medieval, churchyard ancient, churchyard overcast, churchyard cloudy, churchyard cloudy sky,
```

---

### ✅ mlx-community/Phi-3.5-vision-instruct-bf16

**Metrics:** 12.1 TPS | 500 tokens

```text
Title: St Michael's Parish Church in Bishop's Stortford

Description: The St Michael's Parish Church in Bishop's Stortford, England, stands as a Grade I listed building, showcasing Perpendicular Gothic architecture with its early 15th-century origins. The image captures the church's detailed flint-knapped walls and stone-dressed windows, set against a historic churchyard, under a cool February sky.

Keywords: St Michael's Parish Church, Bishop's Stortford, England, United Kingdom, Grade I listed, Perpendicular Gothic, early 15th century, flint-knapped walls, stone-dressed windows, churchyard, historic, architecture, overcast, cool, February, traditional, landmark, heritage, medieval, old, Christianity, Christian, architecture, tourism, historic building, Gothic, church, spire, cross, stone, tombstone, overcast sky, weathervane, worship, faith, faith-based, landmark, architecture, history, tourism, travel, UK, English, European, Christian, Christianity, architecture, overcast, weather, sky, church, churchyard, church spire, church building, church architecture, church design, church history, church exterior, church interior, church exterior, church design, church architecture, church exterior, church design, church architecture, church exterior, church design, church architecture, church exterior, church design, church architecture, church exterior, church design, church architecture, church exterior, church design, church architecture, church exterior, church design, church architecture, church exterior, church design, church architecture, church exterior, church design, church architecture, church exterior, church design, church architecture, church exterior, church design, church architecture, church exterior, church design, church architecture, church exterior, church design, church architecture, church exterior, church design, church architecture, church exterior, church design, church architecture, church exterior, church design, church architecture, church exterior, church design, church architecture, church exterior, church design, church architecture, church exterior, church design, church architecture, church exterior, church design, church architecture, church exterior, church design, church architecture, church exterior, church design, church architecture, church exterior, church design, church architecture, church exterior,
```

---

### ✅ microsoft/Phi-3.5-vision-instruct

**Metrics:** 12 TPS | 500 tokens

```text
Title: St Michael's Parish Church in Bishop's Stortford

Description: The St Michael's Parish Church in Bishop's Stortford, England, stands as a Grade I listed building, showcasing Perpendicular Gothic architecture with its early 15th-century origins. The image captures the church's detailed flint-knapped walls and stone-dressed windows, set against a historic churchyard, under a cool February sky.

Keywords: St Michael's Parish Church, Bishop's Stortford, England, United Kingdom, Grade I listed, Perpendicular Gothic, early 15th century, flint-knapped walls, stone-dressed windows, churchyard, historic, architecture, overcast, cool, February, traditional, landmark, heritage, medieval, old, Christianity, Christian, architecture, tourism, historic building, Gothic, church, spire, cross, stone, tombstone, overcast sky, weathervane, worship, faith, faith-based, landmark, architecture, history, tourism, travel, UK, English, European, Christian, Christianity, architecture, overcast, weather, sky, church, churchyard, church spire, church building, church architecture, church design, church history, church exterior, church interior, church exterior, church design, church architecture, church exterior, church design, church architecture, church exterior, church design, church architecture, church exterior, church design, church architecture, church exterior, church design, church architecture, church exterior, church design, church architecture, church exterior, church design, church architecture, church exterior, church design, church architecture, church exterior, church design, church architecture, church exterior, church design, church architecture, church exterior, church design, church architecture, church exterior, church design, church architecture, church exterior, church design, church architecture, church exterior, church design, church architecture, church exterior, church design, church architecture, church exterior, church design, church architecture, church exterior, church design, church architecture, church exterior, church design, church architecture, church exterior, church design, church architecture, church exterior, church design, church architecture, church exterior, church design, church architecture, church exterior, church design, church architecture, church exterior, church design, church architecture, church exterior,
```

---

### ✅ meta-llama/Llama-3.2-11B-Vision-Instruct

**Metrics:** 3.74 TPS | 174 tokens

```text
**Title:** St. Michael's Parish Church, Bishop's Stortford, England

**Description:** The image depicts St. Michael's Parish Church in Bishop's Stortford, England, showcasing its historic architecture and surrounding cemetery. The church's stone walls and clock tower are prominent features of the scene.

**Keywords:**

* Specific subjects: church, cemetery, clock tower, stone walls, graveyard
* Setting and location: England, Bishop's Stortford, outdoor
* Actions and activities: None visible
* Concepts and themes: History, heritage, tradition, reverence
* Emotional tone: Serene, contemplative
* Visual style: Wide-angle, natural lighting
* Colors: Earthy tones, stone, greenery
* Seasonal and temporal context: Daytime, overcast
* Use-case relevance: Cultural, historical, spiritual
```

---

### ✅ Qwen/Qwen3-VL-2B-Instruct

**Metrics:** 73.7 TPS | 500 tokens

```text
Title: St Michael's Church, Bishop's Stortford, England

Description: St Michael's Parish Church in Bishop's Stortford, England, is depicted from a low-angle perspective, showcasing its historic stone walls and arched windows. The churchyard is visible in the foreground with gravestones and a clock tower, under an overcast sky. The scene captures the solemn and timeless atmosphere of a historic religious building.

Keywords: Bishop's Stortford, British, Christian, Christianity, England, English Church, Europe, Faith, Flint Knapped Wall, Flint Wall, Gothic Window, Grass, Graveyard, Hertfordshire, Parish church, Perpendicular Gothic, St Michael's Church, St Michael's Parish Church, UK, United Kingdom, ancient, arched window, battlements, black, brown, building, car, cemetery, church, church spire, churchyard, clock tower, cloudy, cloudy sky, cross, exterior, gothic architecture, gravestone, grey, headstone, heritage, historic, historic building, landmark, medieval, old, overcast, overcast sky, place of worship, religion, religious building, spire, stone, tombstone, tourism, travel, weathervane, worship, arched window, battlements, black, brown, building, car, cemetery, church, church spire, churchyard, clock tower, cloudy, cloudy sky, cross, exterior, gothic architecture, gravestone, grey, headstone, heritage, historic, historic building, landmark, medieval, old, overcast, overcast sky, place of worship, religion, religious building, spire, stone, tombstone, tourism, travel, weathervane, worship, arched window, battlements, black, brown, building, car, cemetery, church, church spire, churchyard, clock tower, cloudy, cloudy sky, cross, exterior, gothic architecture, gravestone, grey, headstone, heritage, historic, historic building, landmark, medieval, old, overcast, overcast sky, place of worship, religion, religious building, spire, stone, tombstone, tourism, travel, weathervane, worship, arched window, battlements, black, brown, building, car, cemetery, church, church spire, churchyard, clock tower, cloudy, cloudy sky, cross, exterior, gothic architecture, gravestone, grey, headstone, heritage
```

---

### ✅ mlx-community/Qwen3-VL-2B-Thinking-bf16

**Metrics:** 67.1 TPS | 500 tokens

```text
Got it, let's tackle this image analysis. First, I need to create the three sections: Title, Description, Keywords, and Context. Wait, the user said "structured metadata for image cataloguing" with three sections: Title, Description, Keywords. Wait, no, the user says "Respond with exactly these three sections: Title: A concise, descriptive title (5–12 words). Description: A factual 1–3 sentence description... Keywords: 25–50 comma-separated keywords... Context: The image is described as..., existing keywords...". Wait, no, the user's instruction says "Analyze this image and provide structured metadata for image cataloguing. Respond with exactly these three sections: Title: ... Description: ... Keywords: ... Context: ...". Wait, no, the user's example shows the Context section is a description of the image, but the user says "Context: The image is described as ', St Michael's Parish Church, Bishop's Stortford, England, United Kingdom, UK...'" so maybe the Context section is a separate part. Wait, let's check the user's exact instruction:

"Respond with exactly these three sections:

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

Context: The image is described as ', St Michael's Parish Church, Bishop's Stortford, England, United Kingdom, UK...'. Existing keywords: ... Taken around 2026-02-07 16:11:23 GMT at 16:11:23."

Wait, the user's instruction says "Context: The image is described as ...,
```

---

### ✅ mlx-community/Qwen2-VL-2B-Instruct-4bit

**Metrics:** 170 TPS | 500 tokens

```text
Title: St Michael's Parish Church, Bishop's Stortford, England, United Kingdom, UK
Description: A late afternoon photograph of St Michael's Parish Church in Bishop's Stortford, England, featuring the intricate details of its traditional flint-knapped walls and stone-dressed windows, set against the backdrop of its historic churchyard.
Keywords: St Michael's Church, St Michael's Parish Church, Bishop's Stortford, England, UK, Grade I listed building, Perpendicular Gothic architecture, historic building, churchyard, cemetery, church spire, gothic architecture, medieval, old, overcast sky, cloudy, church spire, stone walls, stone windows, historic landmark, religious building, place of worship, travel, heritage, tourism, weather vane, weathervane, stone tower, green grass, gravestones, headstones, black and brown colors, grey sky, cloudy weather, autumn, autumnal colors, autumnal weather, autumnal atmosphere, autumnal mood, autumnal scenery, autumnal landscape, autumnal scenery, autumnal landscape, autumnal scenery, autumnal landscape, autumnal scenery, autumnal landscape, autumnal scenery, autumnal landscape, autumnal scenery, autumnal landscape, autumnal scenery, autumnal landscape, autumnal scenery, autumnal landscape, autumnal scenery, autumnal landscape, autumnal scenery, autumnal landscape, autumnal scenery, autumnal landscape, autumnal scenery, autumnal landscape, autumnal scenery, autumnal landscape, autumnal scenery, autumnal landscape, autumnal scenery, autumnal landscape, autumnal scenery, autumnal landscape, autumnal scenery, autumnal landscape, autumnal scenery, autumnal landscape, autumnal scenery, autumnal landscape, autumnal scenery, autumnal landscape, autumnal scenery, autumnal landscape, autumnal scenery, autumnal landscape, autumnal scenery, autumnal landscape, autumnal scenery, autumnal landscape, autumnal scenery, autumnal landscape, autumnal scenery, autumnal landscape, autumnal scenery, autumnal landscape, autumnal scenery, autumnal landscape, autumnal scenery, autumnal landscape, autumnal scenery, autumnal landscape, autumnal scenery, autumnal landscape, autumnal scenery, autumnal landscape, autumnal scenery, autumnal landscape, autumnal scenery, autumnal landscape, autumnal scenery, autumnal landscape, autumnal scenery, autumnal landscape, autumnal scenery, autumn
```

---

<!-- markdownlint-enable MD013 MD033 -->

---

## System/Hardware Information

- **OS**: Darwin 25.2.0
- **macOS Version**: 26.2
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

- `Pillow`: `12.1.0`
- `huggingface-hub`: `1.4.1`
- `mlx`: `0.30.7.dev20260208+9cd4b9be`
- `mlx-lm`: `0.30.6`
- `mlx-metal`: ``
- `mlx-vlm`: `0.3.11`
- `numpy`: `2.4.2`
- `tokenizers`: `0.22.2`
- `transformers`: `5.1.0`

_Report generated on: 2026-02-08 22:32:28 GMT_
