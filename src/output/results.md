# Model Performance Results

_Generated on 2026-02-14 18:19:37 GMT_

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/nanoLLaVA-1.5-4bit` (318.9 tps)
- **💾 Most efficient:** `mlx-community/FastVLM-0.5B-bf16` (2.1 GB)
- **⚡ Fastest load:** `mlx-community/nanoLLaVA-1.5-4bit` (1.00s)
- **📊 Average TPS:** 76.5 across 40 models

## 📈 Resource Usage

- **Total peak memory:** 732.4 GB
- **Average peak memory:** 18.3 GB
- **Memory efficiency:** 192 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🏆 A: 2 | ✅ B: 6 | 🟡 C: 9 | 🟠 D: 7 | ❌ F: 16

**Average Utility Score:** 42/100

- **Best for cataloging:** `mlx-community/paligemma2-3b-pt-896-4bit` (🏆 A, 89/100)
- **Worst for cataloging:** `mlx-community/Molmo-7B-D-0924-8bit` (❌ F, 0/100)

### ⚠️ 23 Models with Low Utility (D/F)

- `Qwen/Qwen3-VL-2B-Instruct`: ❌ F (27/100) - Mostly echoes context without adding value
- `microsoft/Phi-3.5-vision-instruct`: ❌ F (26/100) - Mostly echoes context without adding value
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (7/100) - Output too short to be useful
- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`: 🟠 D (43/100) - Mostly echoes context without adding value
- `mlx-community/Idefics3-8B-Llama3-bf16`: ❌ F (31/100) - Mostly echoes context without adding value
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
- `mlx-community/gemma-3n-E2B-4bit`: 🟠 D (45/100) - Lacks visual description of image
- `mlx-community/gemma-3n-E4B-it-bf16`: 🟠 D (47/100) - Mostly echoes context without adding value
- `mlx-community/llava-v1.6-mistral-7b-8bit`: ❌ F (25/100) - Mostly echoes context without adding value
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (5/100) - Output too short to be useful
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (10/100) - Output lacks detail
- `mlx-community/pixtral-12b-8bit`: ❌ F (24/100) - Mostly echoes context without adding value
- `mlx-community/pixtral-12b-bf16`: 🟠 D (39/100) - Mostly echoes context without adding value
- `qnguyen3/nanoLLaVA`: ❌ F (15/100) - Mostly echoes context without adding value

## ⚠️ Quality Issues

- **❌ Failed Models (3):**
  - `microsoft/Florence-2-large-ft` (`Weight Mismatch`)
  - `mlx-community/X-Reasoner-7B-8bit` (`OOM`)
  - `prince-canuma/Florence-2-large-ft` (`Model Error`)
- **🔄 Repetitive Output (8):**
  - `Qwen/Qwen3-VL-2B-Instruct` (token: `phrase: "churchyard, churchyard, church..."`)
  - `microsoft/Phi-3.5-vision-instruct` (token: `phrase: "weather, weather, weather, wea..."`)
  - `mlx-community/Phi-3.5-vision-instruct-bf16` (token: `phrase: "weather, weather, weather, wea..."`)
  - `mlx-community/Qwen2-VL-2B-Instruct-4bit` (token: `phrase: "travel photography, outdoor ph..."`)
  - `mlx-community/deepseek-vl2-8bit` (token: `phrase: "english country church, englis..."`)
  - `mlx-community/gemma-3n-E2B-4bit` (token: `phrase: "* 16:11:23 * 16:11:23..."`)
  - `mlx-community/gemma-3n-E4B-it-bf16` (token: `phrase: "churchyard setting, churchyard..."`)
  - `mlx-community/paligemma2-3b-pt-896-4bit` (token: `phrase: "historic house-to-house, histo..."`)
- **📝 Formatting Issues (3):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/GLM-4.6V-Flash-mxfp4`
  - `mlx-community/Idefics3-8B-Llama3-bf16`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 76.5 | Min: 0 | Max: 319
- **Peak Memory**: Avg: 18 | Min: 2.1 | Max: 73
- **Total Time**: Avg: 24.63s | Min: 2.78s | Max: 121.43s
- **Generation Time**: Avg: 21.54s | Min: 1.65s | Max: 120.14s
- **Model Load Time**: Avg: 3.09s | Min: 1.00s | Max: 9.18s

## 🚨 Failures by Package (Actionable)

<!-- markdownlint-disable MD060 -->

| Package | Failures | Error Types | Affected Models |
|---------|----------|-------------|-----------------|
| `mlx` | 2 | OOM, Weight Mismatch | `microsoft/Florence-2-large-ft`, `mlx-community/X-Reasoner-7B-8bit` |
| `mlx-vlm` | 1 | Model Error | `prince-canuma/Florence-2-large-ft` |

<!-- markdownlint-enable MD060 -->

### Actionable Items by Package

#### mlx

- **microsoft/Florence-2-large-ft** (Weight Mismatch)
  - Error: `Model loading failed: Missing 1 parameters:
language_model.lm_head.weight.`
  - Type: `ValueError`
- **mlx-community/X-Reasoner-7B-8bit** (OOM)
  - Error: `Model runtime error during generation for mlx-community/X-Reasoner-7B-8bit: [metal::malloc] Attempting to allocate 13...`
  - Type: `ValueError`

#### mlx-vlm

- **prince-canuma/Florence-2-large-ft** (Model Error)
  - Error: `Model loading failed: RobertaTokenizer has no attribute additional_special_tokens`
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

**Overall runtime:** 1024.63s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Token |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                    |   Error Package |
|:--------------------------------------------------------|--------:|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:----------------------------------|----------------:|
| `microsoft/Florence-2-large-ft`                         |         |                   |                       |                |              |           |             |                  |            |             |                                   |             mlx |
| `mlx-community/X-Reasoner-7B-8bit`                      |         |                   |                       |                |              |           |             |                  |            |             |                                   |             mlx |
| `prince-canuma/Florence-2-large-ft`                     |         |                   |                       |                |              |           |             |                  |            |             |                                   |         mlx-vlm |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |       7 |               863 |                   281 |          1,144 |        3,068 |       294 |         2.9 |            1.65s |      1.14s |       2.79s |                                   |                 |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |       7 |               863 |                   187 |          1,050 |        3,047 |       180 |         4.0 |            1.67s |      1.11s |       2.78s |                                   |                 |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | 151,645 |               585 |                   440 |          1,025 |        2,731 |       319 |         2.8 |            1.98s |      1.00s |       2.98s | verbose                           |                 |
| `mlx-community/FastVLM-0.5B-bf16`                       | 151,645 |               589 |                   319 |            908 |        1,591 |       301 |         2.1 |            2.11s |      1.10s |       3.21s | verbose                           |                 |
| `qnguyen3/nanoLLaVA`                                    | 151,645 |               585 |                   148 |            733 |        2,530 |      99.4 |         5.1 |            2.11s |      1.07s |       3.18s |                                   |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |  49,279 |             1,692 |                   136 |          1,828 |        1,359 |       112 |         5.5 |            2.79s |      1.41s |       4.20s |                                   |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |       1 |             1,609 |                     4 |          1,613 |          463 |      33.2 |          12 |            3.88s |      3.21s |       7.09s | ⚠️harness(prompt_template), ...   |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |  49,154 |             1,792 |                   364 |          2,156 |        1,403 |       111 |         5.5 |            4.88s |      1.35s |       6.24s | verbose                           |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |  49,154 |             1,792 |                   364 |          2,156 |        1,409 |       110 |         5.5 |            4.90s |      1.34s |       6.24s | verbose                           |                 |
| `mlx-community/gemma-3n-E2B-4bit`                       | 236,825 |               847 |                   500 |          1,347 |        1,354 |       107 |         6.1 |            5.59s |      3.71s |       9.30s | repetitive(phrase: "* 16:11:23... |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |       2 |             3,468 |                   307 |          3,775 |        1,008 |       159 |         9.0 |            5.67s |      1.74s |       7.41s |                                   |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |       1 |             1,609 |                    10 |          1,619 |          463 |       5.3 |          27 |            5.67s |      4.95s |      10.62s | context-ignored                   |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |      25 |             1,570 |                   500 |          2,070 |          930 |       113 |          18 |            6.62s |      3.39s |      10.01s |                                   |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |   1,048 |             1,570 |                   500 |          2,070 |          973 |       102 |          22 |            7.11s |      3.84s |      10.95s |                                   |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |   1,230 |             1,570 |                   500 |          2,070 |          969 |      68.4 |          37 |            9.58s |      5.91s |      15.49s |                                   |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |       1 |             1,609 |                   157 |          1,766 |        1,532 |      17.0 |          11 |           10.58s |      3.04s |      13.62s |                                   |                 |
| `mlx-community/deepseek-vl2-8bit`                       |      14 |             2,576 |                   500 |          3,076 |          868 |      59.6 |          32 |           11.80s |      5.52s |      17.32s | repetitive(phrase: "english co... |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    | 236,764 |               855 |                   500 |          1,355 |          892 |      41.4 |          17 |           13.35s |      5.08s |      18.43s | repetitive(phrase: "churchyard... |                 |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      |  93,938 |             1,908 |                   500 |          2,408 |          589 |      48.9 |          60 |           14.15s |      9.18s |      23.33s |                                   |                 |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |  19,409 |             4,681 |                   500 |          5,181 |        1,642 |      41.7 |         4.5 |           15.30s |      2.52s |      17.81s | repetitive(phrase: "historic h... |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |       2 |             3,022 |                   372 |          3,394 |          303 |      42.7 |          11 |           19.01s |      1.89s |      20.90s |                                   |                 |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | 128,001 |             2,876 |                   447 |          3,323 |          764 |      29.7 |          18 |           19.16s |      3.69s |      22.85s | formatting                        |                 |
| `mlx-community/pixtral-12b-8bit`                        |  14,543 |             3,704 |                   500 |          4,204 |          541 |      34.6 |          16 |           21.61s |      3.28s |      24.89s | verbose                           |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |   1,338 |             3,795 |                   500 |          4,295 |          428 |      35.9 |          15 |           23.09s |      3.08s |      26.17s |                                   |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 | 236,764 |               856 |                   500 |          1,356 |          182 |      27.2 |          19 |           23.44s |      5.02s |      28.46s | verbose                           |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |     106 |               856 |                   288 |          1,144 |          177 |      15.3 |          34 |           23.99s |      7.19s |      31.19s |                                   |                 |
| `mlx-community/InternVL3-14B-8bit`                      | 151,645 |             3,904 |                   393 |          4,297 |          407 |      28.0 |          18 |           24.10s |      3.33s |      27.43s | verbose                           |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      | 128,009 |               565 |                   196 |            761 |          167 |      9.14 |          15 |           25.16s |      2.85s |      28.00s |                                   |                 |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    |      11 |             6,592 |                   500 |          7,092 |          342 |      63.0 |         8.4 |           27.51s |      2.36s |      29.87s | formatting                        |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |       2 |             2,968 |                   464 |          3,432 |          231 |      27.5 |          23 |           30.01s |      3.85s |      33.86s | ⚠️harness(encoding)               |                 |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |   3,593 |             6,592 |                   500 |          7,092 |          342 |      47.5 |          11 |           30.15s |      2.60s |      32.75s | verbose, formatting               |                 |
| `mlx-community/pixtral-12b-bf16`                        |  38,188 |             3,704 |                   500 |          4,204 |          577 |      19.4 |          28 |           32.51s |      4.90s |      37.41s | verbose                           |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |         |               0.0 |                   0.0 |            0.0 |            0 |         0 |          41 |           34.29s |      2.18s |      36.47s |                                   |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |         |               0.0 |                   0.0 |            0.0 |            0 |         0 |          48 |           35.51s |      3.27s |      38.79s |                                   |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |  14,826 |             1,446 |                   500 |          1,946 |        1,038 |      12.3 |          12 |           42.30s |      1.81s |      44.11s | repetitive(phrase: "weather, ...  |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |  14,826 |             1,446 |                   500 |          1,946 |        1,027 |      12.2 |          12 |           42.69s |      1.72s |      44.41s | repetitive(phrase: "weather, ...  |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | 128,009 |               566 |                   176 |            742 |          161 |      3.89 |          25 |           49.09s |      4.24s |      53.33s |                                   |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |  17,428 |            16,858 |                   500 |         17,358 |          399 |      76.3 |         8.3 |           49.25s |      1.64s |      50.89s | repetitive(phrase: "churchyard... |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |      11 |            16,860 |                   500 |         17,360 |          337 |      75.7 |         8.3 |           57.08s |      1.83s |      58.91s |                                   |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |      11 |            16,869 |                   500 |         17,369 |          144 |       178 |          73 |          120.14s |      1.29s |     121.43s | repetitive(phrase: "travel pho... |                 |

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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 7773, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 7739, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<3 lines>...
        trust_remote_code=params.trust_remote_code,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 395, in load
    model = load_model(model_path, lazy, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 315, in load_model
    model.load_weights(list(weights.items()))
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx/python/mlx/nn/layers/base.py", line 191, in load_weights
    raise ValueError(f"Missing {num_missing} parameters: \n{missing}.")
ValueError: Missing 1 parameters:
language_model.lm_head.weight.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 7922, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 7781, in _run_model_generation
    raise ValueError(error_details) from load_err
ValueError: Model loading failed: Missing 1 parameters:
language_model.lm_head.weight.
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 7811, in _run_model_generation
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 7922, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 7841, in _run_model_generation
    raise ValueError(msg) from gen_err
ValueError: Model runtime error during generation for mlx-community/X-Reasoner-7B-8bit: [metal::malloc] Attempting to allocate 135699660800 bytes which is greater than the maximum allowed buffer size of 86586540032 bytes.
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 7773, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 7739, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<3 lines>...
        trust_remote_code=params.trust_remote_code,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 405, in load
    processor = load_processor(model_path, True, eos_token_ids=eos_token_id, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 478, in load_processor
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True, **kwargs)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/molmo/processing_molmo.py", line 758, in _patched_auto_processor_from_pretrained
    return _original_auto_processor_from_pretrained.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/base.py", line 355, in _patched_auto_processor_from_pretrained
    return previous_from_pretrained.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
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
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/kimi_vl/processing_kimi_vl.py", line 595, in _patched_auto_processor_from_pretrained
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 7922, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 7781, in _run_model_generation
    raise ValueError(error_details) from load_err
ValueError: Model loading failed: RobertaTokenizer has no attribute additional_special_tokens
```

</details>

---

### ✅ mlx-community/LFM2-VL-1.6B-8bit

**Metrics:** 294 TPS | 281 tokens

```text
Title: St Michael's Parish Church, Bishop's Stortford, England

Description: The image depicts St Michael's Parish Church, a Grade I listed building in Bishop's Stortford, England, United Kingdom. The church is a prominent example of Perpendicular Gothic architecture, with its current structure dating largely from the early 15th century. The photograph captures the intricate details of the church's traditional flint-knapped walls and stone-dressed windows, set against the backdrop of its historic churchyard. The soft, even light of the overcast day accentuates the textures and formidable presence of this centuries-old landmark.

Keywords: Bishop's Stortford, British, Christian, Christianity, England, English Church, Europe, Faith, Flint Knapped Wall, Flint Wall, Gothic Window, Grass, Graveyard, Hertfordshire, Parish church, Perpendicular Gothic, St Michael's Church, St Michael's Parish Church, UK, United Kingdom, ancient, arched window, architecture, battlements, black, brown, building, car, cemetery, church, church spire, stone, tombstone, heritage, historic, historic building, landmark, medieval, old, overcast, overcast sky, place of worship, religion, religious building, spire, stone, tombstone, tourism, tower, travel, weathervane, worship.
```

---

### ✅ mlx-community/LFM2.5-VL-1.6B-bf16

**Metrics:** 180 TPS | 187 tokens

```text
Title: St Michael's Parish Church, Bishop's Stortford, England

Description: The image depicts St Michael's Parish Church in Bishop's Stortford, England, a Grade I listed building showcasing Perpendicular Gothic architecture. The church's intricate details, such as the flint-knapped walls and stone-dressed windows, are highlighted against the backdrop of its historic churchyard. The overcast sky and soft lighting accentuate the textures and formidable presence of this centuries-old landmark.

Keywords: St Michael's Parish Church, Bishop's Stortford, England, Perpendicular Gothic, church, churchyard, church spire, stone, gravestone, heritage, historic building, landmark, medieval, old, overcast, overcast sky, place of worship, religion, religious building, spire, stone, tombstone, tourism, tower, travel, weathervane, worship
```

---

### ✅ mlx-community/nanoLLaVA-1.5-4bit

**Metrics:** 319 TPS | 440 tokens

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

### ✅ mlx-community/FastVLM-0.5B-bf16

**Metrics:** 301 TPS | 319 tokens

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

Context: The image is described as ', St Michael's Parish Church, Bishop's Stortford, England, United Kingdom, UK. St Michael's Parish Church in Bishop's Stortford, England, is pictured here on a late afternoon under a cool February sky. This Grade I listed building is a prominent example of Perpendicular Gothic architecture, with its current structure dating largely from the early 15th century. The photograph captures the intricate details of the church's traditional flint-knapped walls and stone-dressed windows, set against the backdrop of its historic churchyard. The soft, even light of the overcast day accentuates the textures and formidable presence of this centuries-old landmark.'
```

---

### ✅ qnguyen3/nanoLLaVA

**Metrics:** 99.4 TPS | 148 tokens

```text
The image portrays a serene, late afternoon scene at St Michael's Parish Church in Bishop's Stortford, England. The church is a prominent example of Perpendicular Gothic architecture, with its current structure dating largely from the early 15th century. The image captures the intricate details of the church's traditional flint-knapped walls and stone-dressed windows, set against the backdrop of its historic churchyard. The soft, even light of the overcast day accentuates the textures and formidable presence of this centuries-old landmark. The image is taken on 2026-02-07 16:11:23 GMT (at 16:11:23 local time).
```

---

### ✅ mlx-community/SmolVLM2-2.2B-Instruct-mlx

**Metrics:** 112 TPS | 136 tokens

```text
 Title: St Michael's Parish Church, Bishop's Stortford, England, United Kingdom, UK
St Michael's Parish Church in Bishop's Stortford, England, is pictured here on a late afternoon under a cool February sky. This Grade I listed building is a prominent example of Perpendicular Gothic architecture, with its current structure dating largely from the early 15th century. The photograph captures the intricate details of the church's traditional flint-knapped walls and stone-dressed windows, set against the backdrop of its historic churchyard. The soft, even light of the overcast day accentuates the textures and formidable presence of this centuries-old landmark.
```

---

### ✅ mlx-community/paligemma2-10b-ft-docci-448-6bit

**Metrics:** 33.2 TPS | 4 tokens

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

**Metrics:** 110 TPS | 364 tokens

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

### ✅ mlx-community/gemma-3n-E2B-4bit

**Metrics:** 107 TPS | 500 tokens

```text


* 16:11:23
* 16:11:23
* 16:11:23
* 16:11:23
* 16:11:23
* 16:11:23
* 16:11:23
* 16:11:23
* 16:11:23
* 16:11:23
* 16:11:23
* 16:11:23
* 16:11:23
* 16:11:23
* 16:11:23
* 16:11:23
* 16:11:23
* 16:11:23
* 16:11:23
* 16:11:23
* 16:11:23
* 16:11:23
* 16:11:23
* 16:11:23
* 16:11:23
* 16:11:23
* 16:11:23
* 16:11:23
* 16:11:23
* 16:11:23
* 16:11:23
* 16:11:23
* 16:11:23
* 16:11:23
* 16:11:23
* 16:11:23
* 16:11:23
* 16:11:23
* 16:11:23
* 16:11:23
* 16:11:23
* 16:11:23
* 16:11:23
* 16:11:23
* 16:11:23
* 16
```

---

### ✅ mlx-community/Ministral-3-3B-Instruct-2512-4bit

**Metrics:** 159 TPS | 307 tokens

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

**Metrics:** 5.3 TPS | 10 tokens

```text
Be specific about the subject of the image.
```

---

### ✅ jqlive/Kimi-VL-A3B-Thinking-2506-6bit

**Metrics:** 113 TPS | 500 tokens

```text
◁think▷Got it, let's tackle this. First, I need to create a title, description, and keywords for the image of St Michael's Parish Church in Bishop's Stortford, England.

Starting with the title: it should be concise and descriptive. The image shows the church, so maybe "St Michael's Parish Church, Bishop's Stortford, England" but that's too long. Maybe "St Michael's Parish Church, Bishop's Stortford, England" is okay, but let's make it a bit more engaging. Wait, the existing keywords include "St Michael's Parish Church" and "St Michael's Church" so maybe "St Michael's Parish Church, Bishop's Stortford, England" is good.

Description: Need to cover key subjects (church, architecture, graveyard), setting (historic, outdoor, churchyard), action (no specific action, so focus on what's shown: the church's architecture, gravestones, the stone walls, the spire). Also mention the architectural style (Perpendicular Gothic), materials (flint-knapped walls, stone-dressed windows), time of day (late afternoon, overcast, cool February sky), and the mood (serene, historic). So the description should be 1-3 sentences.

Keywords: Need 25-50 keywords. Let's list from the existing ones and add more. Existing keywords include Bishop's Stortford, St Michael's Parish Church, England, England, Christian, Christianity, England, English Church, Europe, Faith, Flint Knapped Wall, Flint Wall, Gothic Window, Grass, Graveyard, Hertfordshire, Parish church, Perpendicular Gothic, UK, United Kingdom, ancient, arched window, architecture, battlements, black, brown, building, car, cemetery, church, church spire, churchyard, clock tower, cloudy, cloudy sky, cross, exterior, gothic architecture, gravestone, grey, headstone, heritage, historic, historic building, landmark, medieval, old, overcast, overcast sky, place of worship, religion, religious building, spire, stone, tombstone, tourism, tower, travel, weathervane, worship. Also, add visual style (aerial? No, it's a ground view. Close-up? No, it's a wide shot. Wide-angle? Maybe. Colors: grey, brown, green (grass). Lighting: overcast, soft light. Seasonal:
```

---

### ✅ mlx-community/Kimi-VL-A3B-Thinking-8bit

**Metrics:** 102 TPS | 500 tokens

```text
◁think▷Okay, let's tackle this query. The user wants me to analyze an image and provide structured metadata. The image is of St Michael's Parish Church in Bishop's Stortford, England. The existing keywords and context are already provided, so I need to build on that.

First, the title should be concise and descriptive. Since it's a church, maybe something like "St Michael's Parish Church in Bishop's Stortford, England". That's straightforward and includes the key location.

For the description, I need to cover key subjects, setting, and action. The image shows the church's exterior, focusing on its architecture. The context mentions it's a Grade I listed building with Perpendicular Gothic style, built in the early 15th century. The photo was taken on a cloudy February afternoon. The walls are flint-knapped and stone-dressed, with arched windows and a churchyard. The overcast lighting highlights textures and the building's historic presence.

Keywords should start with specific subjects: St Michael's Parish Church, Bishop's Stortford, England, UK. Then setting: urban, outdoor, churchyard. Actions/activities: photography, tourism, worship. Concepts/themes: heritage, historic architecture, religious building. Mood: serene, solemn. Visual style: exterior, wide-angle, overcast lighting. Colors: grey, brown, stone, flint-knapped walls. Seasonal: winter, February. Use-case: tourism, travel, heritage site.

I need to make sure the keywords are ordered from most specific to general. Also, include any new concepts or emotions not in the existing list, like "serene" or "solemn". Check if "graveyard" is relevant, which it is. Maybe add "clock tower" and "weathervane" from the existing keywords. Also, "Perpendicular Gothic architecture" is specific. Avoid repeating existing keywords but ensure all relevant ones are included. Make sure the title is exactly 5-12 words. Let me structure all this into the three sections as specified.◁/think▷Title: St Michael's Parish Church in Bishop's Stortford, England  
Description: A photograph of St Michael's Parish Church in Bishop's Stortford, England, showcasing its Grade I listed Perpendicular Gothic architecture with flint-knapped walls, arched windows, and a historic churchyard under an overcast February sky.  
Keywords: St Michael's
```

---

### ✅ mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

**Metrics:** 68.4 TPS | 500 tokens

```text
◁think▷Got it, let's tackle this. First, I need to create a title, description, and keywords for the image of St Michael's Parish Church in Bishop's Stortford, England.

Starting with the title: it should be concise and descriptive. The image shows the church with its architectural details, so maybe "St Michael's Parish Church, Bishop's Stortford, England: Perpendicular Gothic Architecture in a Historic Churchyard." That's 12 words, which fits the 5-12 word limit.

Next, the description: need to cover key subjects (the church, its architecture, the churchyard), setting (historic, outdoor, possibly rural or urban? The description mentions Bishop's Stortford, which is a town, so maybe urban setting), action (the church is static, so no action, but the light and atmosphere). The description should be 1-3 sentences. Let's structure it: "St Michael's Parish Church in Bishop's Stortford, England, is depicted under an overcast sky, showcasing its Perpendicular Gothic architecture with flint-knapped walls and arched windows. The churchyard with gravestones and the church spire are visible, highlighting its historic and religious significance."

Keywords: Need 25-50 comma-separated keywords. Let's list from the context and image. Specific subjects: St Michael's Parish Church, Bishop's Stortford, England, UK, Christian, Christianity, England, English Church, Hertfordshire, Perpendicular Gothic, Flint Knapped Wall, Flint Wall, Gothic Window, Grass, Graveyard, Tombstone, Headstone, Churchyard, Stone, Stone Building, Historic Building, Heritage, Medieval, Architecture, Tourism, Travel, Religion, Religious Building, Place of Worship, Overcast, Cloudy Sky, Black, Brown, Grey, Clock Tower, Church Spire, Weathervane, Battlements, Car (maybe not, but the image might have a car? Wait, the context says "car" is in existing keywords but the image might not show a car. Wait, the image provided is a clear church, no car visible. So maybe exclude. Let's check the existing keywords: existing ones include Bishop's Stortford, British, Christian, Christianity, England, English Church, Europe, Faith, Flint Knapped Wall, Flint Wall, Gothic Window, Grass, Graveyard, Hertfordshire, Parish church, Perpendicular Gothic, St Michael's Church, St
```

---

### ✅ mlx-community/paligemma2-3b-ft-docci-448-bf16

**Metrics:** 17 TPS | 157 tokens

```text
A low-angle view of the side of St Michael's Parish Church in Bishop's Stortford, England. The church is a light brown stone with a tall, narrow tower on the right side of the image. The tower is made of red brick with a pointed top. The church has a row of windows on the left side of the church, with a row of windows on the right side of the church. The windows are divided into three sections by a thin metal frame. The windows are rectangular and have a pointed top. The windows are dark gray, with a light gray frame. The church is built on a hill, with a grassy area in front of the church. There are four gravestones in the grassy area. The sky is gray and gloomy.
```

---

### ✅ mlx-community/deepseek-vl2-8bit

**Metrics:** 59.6 TPS | 500 tokens

```text
**Title:** A Stunning View of St. Michael's Church in Bishop's Stortford, UK

**Description:** The image shows the historic St. Michael's Parish Church in Bishop's Stortford, a prime example of English Perpendicular Gothic architecture. The building is constructed from a mix of flint and stone, with a tall, pointed spire and large, arched windows. The structure is set against a backdrop of a cloudy, overcast sky, with a well-manicured lawn in the foreground. The scene exudes a serene and historical atmosphere, with the building's gothic details standing out against the overcast sky.

**Keywords:** St. Michael's Church, Bishop's Stortford, UK, English Heritage, gothic architecture, English Parish Church, historic building, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church, English country church,
```

---

### ✅ mlx-community/gemma-3n-E4B-it-bf16

**Metrics:** 41.4 TPS | 500 tokens

```text
**Title:** Ancient Stone Church in England

**Description:** A detailed, low-angle view captures the weathered stone facade of St. Michael's Church in Bishop's Stortford, England, showcasing its Perpendicular Gothic architecture and historic gravestones in the foreground.

**Keywords:** St Michael's Church, Bishop's Stortford, England, UK, church, Gothic architecture, Perpendicular Gothic, stone, weathered, facade, tower, spire, gravestones, headstones, grass, historic, ancient, building, religious, England, UK, architectural, exterior, cloudy sky, low angle, detailed, historic building, churchyard, stone walls, window, buttress, England landscape, British architecture, religious building, old, heritage, stone masonry, church tower, steeple, graveyard, overcast, historical landmark, architectural detail, stone texture, vertical, building exterior, church spire, stone construction, weathered stone, church windows, church roof, church details, church history, church grounds, church setting, church landscape, church view, church photography, church architecture, church exterior, church building, church grounds, churchyard view, churchyard setting, churchyard landscape, churchyard photography, churchyard details, churchyard history, churchyard setting, churchyard landscape, churchyard photography, churchyard details, churchyard history, churchyard setting, churchyard landscape, churchyard photography, churchyard details, churchyard history, churchyard setting, churchyard landscape, churchyard photography, churchyard details, churchyard history, churchyard setting, churchyard landscape, churchyard photography, churchyard details, churchyard history, churchyard setting, churchyard landscape, churchyard photography, churchyard details, churchyard history, churchyard setting, churchyard landscape, churchyard photography, churchyard details, churchyard history, churchyard setting, churchyard landscape, churchyard photography, churchyard details, churchyard history, churchyard setting, churchyard landscape, churchyard photography, churchyard details, churchyard history, churchyard setting, churchyard landscape, churchyard photography, churchyard details, churchyard history, churchyard setting, churchyard landscape, churchyard photography, churchyard details, churchyard history, churchyard setting, churchyard landscape, churchyard photography, churchyard details, churchyard history, churchyard setting, churchyard landscape, churchyard photography, churchyard details, churchyard history, churchyard setting, churchyard landscape,
```

---

### ✅ mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

**Metrics:** 48.9 TPS | 500 tokens

```text
Let's tackle this. First, the title needs to be concise. The image is of St Michael's Parish Church in Bishop's Stortford, England. So maybe "St Michael's Parish Church, Bishop's Stortford, England" – that's 7 words, fits the range.

Next, the description. It's a Grade I listed building, Perpendicular Gothic, early 15th century. The photo is from late afternoon, overcast sky. Key elements: flint-knapped walls, stone-dressed windows, churchyard. So: "St Michael's Parish Church, a Grade I listed Perpendicular Gothic building from the early 15th century, is shown in Bishop's Stortford, England, with its flint-knapped walls and stone windows against a cool, overcast late afternoon sky." That covers the subject, setting, and action (photograph capturing details).

Keywords: Need to order from specific to general. Let's list the specific ones first: St Michael's Parish Church, Bishop's Stortford, England, UK, Grade I listed building, Perpendicular Gothic architecture, early 15th century, flint-knapped walls, stone-dressed windows, churchyard, late afternoon, overcast sky, flint wall, gothic window, church spire, clock tower, graveyard, headstone, tombstone, weathervane, cross, ancient, historic, heritage, place of worship, religious building, church, churchyard, stone, brick, overcast, cloudy, February (wait, the context says late February, but the date is 2026-02-07, so February). Wait, the existing keywords have some, but need to make sure they're ordered correctly. Let's check:

Most specific: St Michael's Parish Church, Bishop's Stortford, England, UK, Grade I listed building, Perpendicular Gothic architecture, early 15th century, flint-knapped walls, stone-dressed windows, churchyard, late afternoon, overcast sky, flint wall, gothic window, church spire, clock tower, graveyard, headstone, tombstone, weathervane, cross, ancient, historic, heritage, place of worship,
```

---

### ✅ mlx-community/paligemma2-3b-pt-896-4bit

**Metrics:** 41.7 TPS | 500 tokens

```text
Keywords: 25–50 comma-separated keywords ordered from most specific to most general, covering:
- Building, Car, Church, Church spire, Cemetery, Church tower, Cloud, Historic building, Historic landmark, Historic monument, Historic site, Historic town, Historic village, Historic window, House, House, Housebuilding, House-to-house, Historic House, Historic Housebuilding, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House, Historic House-to-House
```

---

### ✅ mlx-community/llava-v1.6-mistral-7b-8bit

**Metrics:** 42.7 TPS | 372 tokens

```text
Title: St Michael's Parish Church, Bishop's Stortford, England, United Kingdom, UK

Description: A photograph captures the intricate details of St Michael's Parish Church, a Grade I listed building in Bishop's Stortford, England, UK. The image showcases the traditional flint-knapped walls and stone-dressed windows, set against the backdrop of the historic churchyard. The soft, even light of the overcast day accentuates the textures and formidable presence of this centuries-old landmark.

Keywords: St Michael's Parish Church, Bishop's Stortford, English Church, Europe, Faith, Christianity, England, Gothic architecture, Heritage, Historic, Landmark, Medieval, Old, Perpendicular Gothic, Religion, Religious building, St Michael's Church, St Michael's Parish Church, United Kingdom, UK, Architecture, Church, Churchyard, Clock tower, Overcast sky, Stone, Tombstone, Tourism, Worship

Context: The image is described as 'St Michael's Parish Church in Bishop's Stortford, England, UK, pictured here on a late afternoon under a cool February sky. This Grade I listed building is a prominent example of Perpendicular Gothic architecture, with its current structure dating largely from the early 15th century. The photograph captures the intricate details of the church's traditional flint-knapped walls and stone-dressed windows, set against the backdrop of its historic churchyard. The soft, even light of the overcast day accentuates the textures and formidable presence of this centuries-old landmark.'
```

---

### ✅ mlx-community/Idefics3-8B-Llama3-bf16

**Metrics:** 29.7 TPS | 447 tokens

```text
 Title: St Michael's Parish Church in Bishop's Stortford, England, United Kingdom, UK

Description: St Michael's Parish Church in Bishop's Stortford, England, is pictured here on a late afternoon under a cool February sky. This Grade I listed building is a prominent example of Perpendicular Gothic architecture, with its current structure dating largely from the early 15th century. The photograph captures the intricate details of the church's traditional flint-knapped walls and stone-dressed windows, set against the backdrop of its historic churchyard. The soft, even light of the overcast day accentuates the textures and formidable presence of this centuries-old landmark.

Keywords: Bishop's Stortford, British, Christian, Christianity, England, English Church, Europe, Faith, Flint Knapped Wall, Flint Wall, Gothic Window, Grass, Graveyard, Hertfordshire, Parish church, Perpendicular Gothic, St Michael's Church, St Michael's Parish Church, UK, United Kingdom, ancient, arched window, architecture, battlements, black, brown, building, car, cemetery, church, church spire, churchyard, clock tower, cloudy, cloudy sky, cross, exterior, gothic architecture, gravestone, grey, headstone, heritage, historic, historic building, landmark, medieval, old, overcast, overcast sky, place of worship, religion, religious building, spire, stone, tombstone, tourism, tower, travel, weathervane, worship

Context: The image is described as 'St Michael's Parish Church, Bishop's Stortford, England, United Kingdom, UK. St Michael's Parish Church in Bishop's Stortford, England, is pictured here on a late afternoon under a cool February sky. This Grade I listed building is a prominent example of Perpendicular Gothic architecture, with its current structure dating largely from the early 15th century. The photograph captures the intricate details of the church's traditional flint-knapped walls and stone-dressed windows, set against the backdrop of its historic churchyard. The soft, even light of the overcast day accentuates the textures and formidable presence of this centuries-old landmark.'<end_of_utterance>
```

---

### ✅ mlx-community/pixtral-12b-8bit

**Metrics:** 34.6 TPS | 500 tokens

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

**Metrics:** 35.9 TPS | 500 tokens

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

### ✅ mlx-community/gemma-3-27b-it-qat-4bit

**Metrics:** 27.2 TPS | 500 tokens

```text
## Title: St Michael's Parish Church, Bishop's Stortford, England – February Afternoon

## Description:
This photograph depicts the exterior of St Michael's Parish Church in Bishop's Stortford, England, a Grade I listed building showcasing Perpendicular Gothic architecture. The church is framed by a historic churchyard with numerous gravestones, and is photographed under a cloudy, overcast sky. The image highlights the church’s flint-knapped walls, arched windows, and prominent spire.

## Keywords:
St Michael's Parish Church, Bishop's Stortford, Hertfordshire, England, UK, United Kingdom, Perpendicular Gothic, flint knapped wall, arched window, church spire, churchyard, gravestones, headstones, tombstone, medieval architecture, historic building, landmark, place of worship, religious building, church, exterior, overcast sky, cloudy sky, grey tones, stone wall, spire, clock tower, weathervane, cross, gothic window, church architecture, British architecture, European architecture, heritage, tourism, travel, photography, architectural detail, religious site, place of interest, cultural heritage, history, ancient, solemn, peaceful, quiet, overcast light, February, winter, afternoon, wide-angle, exterior view, stone masonry, building facade, cultural landmark, religious art, religious monument, textured wall, stone structure, landmark architecture, religious institution, historical site, cultural tourism, architectural photography, religious building exterior, stone detailing, gothic style, religious practice, cultural preservation, stone texture, architectural design, religious history, stone construction, cultural identity, religious symbol, stone craftsmanship, architectural heritage, religious tradition, stone architecture, cultural significance, religious community, stone building, cultural expression, religious belief, stone work, cultural value, religious experience, stone feature, cultural context, religious influence, stone element, cultural impact, religious legacy, stone material, cultural asset, religious impact, stone surface, cultural resource, religious influence, stone composition, cultural element, religious foundation, stone form, cultural artifact, religious structure, stone pattern, cultural object, religious practice, stone design, cultural symbol, religious institution, stone element, cultural expression, stone feature, cultural context, stone composition, stone pattern, stone texture, stone work, stone surface, stone material, stone form, stone detailing, stone construction, stone masonry, stone architecture, stone building, stone feature, stone element, stone composition, stone pattern,
```

---

### ✅ mlx-community/gemma-3-27b-it-qat-8bit

**Metrics:** 15.3 TPS | 288 tokens

```text
**Title:** St Michael's Parish Church, Bishop's Stortford, England – February Afternoon

**Description:** This photograph depicts St Michael's Parish Church in Bishop's Stortford, Hertfordshire, England, a Grade I listed building showcasing Perpendicular Gothic architecture. The church is framed against a cloudy February sky, with its flint-knapped walls and stone windows prominently visible. Gravestones dot the surrounding churchyard, adding to the historic atmosphere.

**Keywords:** St Michael's Parish Church, Bishop's Stortford, Hertfordshire, England, UK, United Kingdom, Perpendicular Gothic, flint knapped wall, stone window, arched window, church spire, churchyard, graveyard, cemetery, gravestone, headstone, tombstone, historic building, landmark, medieval architecture, place of worship, religious building, church, exterior, overcast sky, cloudy, grey tones, brown tones, stone, tower, clock tower, battlements, cross, weathervane, heritage, tourism, travel, architecture, ancient, historic, religious site, European architecture, British architecture, place of interest, overcast lighting, soft light, winter, February, afternoon, solemn, quiet, stillness, texture, detail, wide-angle, building exterior, cultural heritage, religious art, gothic window, stone dressing, traditional architecture, European landmark, place of remembrance, solemnity, cultural tourism, architectural photography.
```

---

### ✅ mlx-community/InternVL3-14B-8bit

**Metrics:** 28 TPS | 393 tokens

```text
Title: Gothic Church with Battlements and Graveyard

Description: The image shows St Michael's Parish Church in Bishop's Stortford, England, featuring a tall spire, arched windows, and a historic graveyard with gravestones in the foreground. The architecture is Perpendicular Gothic, with intricate stone and flint walls under an overcast sky.

Keywords: Bishop's Stortford, British, Christian, Christianity, England, English Church, Europe, Faith, Flint Knapped Wall, Flint Wall, Gothic Window, Grass, Graveyard, Hertfordshire, Parish church, Perpendicular Gothic, St Michael's Church, St Michael's Parish Church, UK, United Kingdom, ancient, arched window, architecture, battlements, black, brown, building, cloudy, cloudy sky, cross, exterior, gothic architecture, gravestone, grey, headstone, heritage, historic, historic building, landmark, medieval, old, overcast, overcast sky, place of worship, religion, religious building, spire, stone, tombstone, tourism, tower, travel, weathervane, worship, serene, dramatic, wide-angle, muted tones, historical, solemn, weathered stone, traditional, quiet, contemplative

Context: The image is described as 'St Michael's Parish Church, Bishop's Stortford, England, United Kingdom, UK. St Michael's Parish Church in Bishop's Stortford, England, is pictured here on a late afternoon under a cool February sky. This Grade I listed building is a prominent example of Perpendicular Gothic architecture, with its current structure dating largely from the early 15th century. The photograph captures the intricate details of the church's traditional flint-knapped walls and stone-dressed windows, set against the backdrop of its historic churchyard. The soft, even light of the overcast day accentuates the textures and formidable presence of this centuries-old landmark.'
```

---

### ✅ mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

**Metrics:** 9.14 TPS | 196 tokens

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

### ✅ mlx-community/GLM-4.6V-Flash-mxfp4

**Metrics:** 63 TPS | 500 tokens

```text
<think>Got it, let's tackle this. The user wants structured metadata for image cataloging. First, I need to extract the Title, Description, and Keywords based on the provided context.

First, the Title. It needs to be concise (5-12 words). Looking at the context, the main subject is St Michael's Parish Church in Bishop's Stortford. So a good title would be "St Michael's Parish Church, Bishop's Stortford".

Next, the Description. It needs to be a 1-3 sentence factual description. The context mentions it's a Grade I listed building, Perpendicular Gothic architecture, flint-knapped walls, stone-dressed windows, historic churchyard with gravestones, overcast sky, late afternoon. So a concise description would be: "St Michael's Parish Church in Bishop's Stortford, England, is a Grade I listed Perpendicular Gothic building with flint-knapped walls, stone-dressed windows, and a historic churchyard with gravestones, captured on an overcast late afternoon."

Now, Keywords. The user wants 25-50 comma-separated keywords, ordered from most specific to most general. Let's extract relevant keywords from the context and image.

First, specific subjects: St Michael's Parish Church, Bishop's Stortford, England, UK, historic building, Grade I listed, Perpendicular Gothic architecture, flint-knapped wall, stone-dressed window, church spire, clock tower, gravestone, headstone, tombstone, churchyard, cemetery, place of worship, religious building, landmark, medieval architecture, old building, historic site, heritage building, stone building, brick building (but the main is flint), spire, weathervane, cross, clock face, gravestones, churchyard grass, benches (from image: "black benches" are visible), overcast sky, cloudy sky, late afternoon light, cool February sky (from context), historic churchyard, gravestones, church spire, clock tower, Perpendicular Gothic, flint-knapped wall, stone-dressed window, Bishop's Stortford, Hertfordshire, England, United Kingdom, UK, historic building, Grade I listed, Perpendicular Gothic architecture, flint-knapped wall, stone-dressed window, church spire, clock tower, gravestone, headstone, tombstone, churchyard, cemetery, place of worship,
```

---

### ✅ mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

**Metrics:** 27.5 TPS | 464 tokens

```text
Title:ĠStĠMichael'sĠParishĠChurch,ĠBishop'sĠStortfordĊĊDescription:ĠTheĠimageĠshowsĠStĠMichael'sĠParishĠChurchĠinĠBishop'sĠStortford,ĠEngland,ĠaĠhistoricĠGradeĠIĠlistedĠbuildingĠfeaturingĠPerpendicularĠGothicĠarchitecture.ĠTheĠchurch'sĠflint-knappedĠwallsĠandĠstone-trimmedĠwindowsĠareĠprominentlyĠdisplayed,ĠsetĠagainstĠanĠovercastĠsky.ĊĊKeywords:ĠStĠMichael'sĠParishĠChurch,ĠBishop'sĠStortford,ĠPerpendicularĠGothicĠarchitecture,ĠGradeĠIĠlistedĠbuilding,Ġflint-knappedĠwalls,Ġstone-trimmedĠwindows,ĠchurchĠspire,Ġchurchyard,Ġgravestones,ĠovercastĠsky,ĠhistoricĠbuilding,ĠmedievalĠarchitecture,ĠplaceĠofĠworship,ĠreligiousĠbuilding,Ġtourism,Ġtravel,Ġheritage,ĠEngland,ĠUK,ĠEurope,ĠChristian,ĠChristianity,Ġfaith,Ġarchitecture,Ġlandmark,Ġold,Ġancient,Ġexterior,Ġcloudy,Ġgrey,Ġbrown,Ġblack,Ġgrass,Ġcemetery,Ġheadstone,Ġtombstone,Ġcross,Ġweathervane,ĠclockĠtower,Ġbattlements,ĠarchedĠwindow,ĠgothicĠwindow,Ġcar,Ġbuilding,Ġspire,Ġtower,Ġreligion,Ġworship,Ġhistoric,Ġheritage,Ġtourism,Ġtravel,Ġovercast,ĠcloudyĠsky,ĠplaceĠofĠworship,ĠreligiousĠbuilding,Ġtourism,Ġtravel,Ġheritage,ĠEngland,ĠUK,ĠEurope,ĠChristian,ĠChristianity,Ġfaith,Ġarchitecture,Ġlandmark,Ġold,Ġancient,Ġexterior,Ġcloudy,Ġgrey,Ġbrown,Ġblack,Ġgrass,Ġcemetery,Ġheadstone,Ġtombstone,Ġcross,Ġweathervane,ĠclockĠtower,Ġbattlements,ĠarchedĠwindow,ĠgothicĠwindow,Ġcar,Ġbuilding,Ġspire,Ġtower,Ġreligion,ĠworshipĊĊContext:ĠTheĠimageĠisĠdescribedĠasĠ',ĠStĠMichael'sĠParishĠChurch,ĠBishop'sĠStortford,ĠEngland,ĠUnitedĠKingdom,ĠUKStĠMichael'sĠParishĠChurchĠinĠBishop'sĠStortford,ĠEngland,ĠisĠpicturedĠhereĠonĠaĠlateĠafternoonĠunderĠaĠcoolĠFebruaryĠsky.ĠThisĠGradeĠIĠlistedĠbuildingĠisĠaĠprominentĠexampleĠofĠPerpendicularĠGothicĠarchitecture,ĠwithĠitsĠcurrentĠstructureĠdatingĠlargelyĠfromĠtheĠearlyĠ15thĠcentury.ĠTheĠphotographĠcapturesĠtheĠintricateĠdetailsĠofĠtheĠchurch'sĠtraditionalĠflint-knappedĠwallsĠandĠstone-trimmedĠwindows,ĠsetĠagainstĠtheĠbackdropĠofĠitsĠhistoricĠchurchyard.ĠTheĠsoft,ĠevenĠlightĠofĠtheĠovercastĠdayĠaccentuatesĠtheĠtexturesĠandĠformidableĠpresenceĠofĠthisĠcenturies-oldĠlandmark.'.
```

---

### ✅ mlx-community/GLM-4.6V-Flash-6bit

**Metrics:** 47.5 TPS | 500 tokens

```text
<think>Got it, let's tackle this. First, the title needs to be concise, 5-12 words. Looking at the image, it's St Michael's Parish Church in Bishop's Stortford. So maybe "St Michael's Parish Church, Bishop's Stortford".

Then the description: factual, 1-3 sentences. Need to cover key subjects (church, spire, graveyard), setting (historic churchyard, overcast sky), action (static, no people, but the presence of gravestones suggests a quiet, solemn setting).

So description: "The image depicts St Michael's Parish Church in Bishop's Stortford, England, showcasing its Perpendicular Gothic architecture, flint-knapped walls, and a tall spire, set against an overcast sky with a historic churchyard featuring gravestones in the foreground."

Wait, but need to keep it 1-3 sentences. Maybe: "The image shows St Michael's Parish Church in Bishop's Stortford, England, with its Perpendicular Gothic architecture, flint walls, and a tall spire, set in a historic churchyard with gravestones under an overcast sky."

That's better, concise.

Now keywords: need 25-50, ordered from most specific to most general. Let's list specific subjects first: St Michael's Parish Church, Bishop's Stortford church, flint-knapped church, Perpendicular Gothic church, tall spire church, clock tower church, Gothic window church, arched window church, gravestone, tombstone, churchyard, cemetery, historic building, landmark, medieval architecture, old church, stone church, brick church (wait, the tower is brick? The main building is flint, tower is brick? So maybe include brick tower).

Setting: historic churchyard, rural (but Bishop's Stortford is a town, so maybe urban-rural mix), outdoor, England, Hertfordshire, UK.

Actions: static, no people, but the presence of gravestones suggests a solemn, quiet setting.

Concepts: heritage, historic landmark, religious building, place of worship, faith, Christianity, medieval heritage, architectural heritage.

Mood: serene, solemn, quiet, timeless, historic.

Visual style: wide-angle (since it's a large building, wide-angle to capture the whole structure), exterior shot, overcast lighting, natural light, muted colors (grey, brown, green).


```

---

### ✅ mlx-community/pixtral-12b-bf16

**Metrics:** 19.4 TPS | 500 tokens

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

### ✅ mlx-community/Phi-3.5-vision-instruct-bf16

**Metrics:** 12.3 TPS | 500 tokens

```text
Title: St Michael's Parish Church in Bishop's Stortford

Description: The St Michael's Parish Church in Bishop's Stortford, England, stands as a Grade I listed building, showcasing Perpendicular Gothic architecture with its early 15th-century origins. The image captures the church's detailed flint-knapped walls and stone-dressed windows, set against a historic churchyard, under a cool February sky.

Keywords: St Michael's Parish Church, Bishop's Stortford, England, United Kingdom, Grade I listed, Perpendicular Gothic, early 15th century, flint-knapped walls, stone-dressed windows, churchyard, historic, architecture, overcast, cool, February, traditional, landmark, heritage, medieval, Gothic, Christianity, Christian, architecture, tourism, historic building, worship, stone, tombstone, cross, church spire, churchyard, cemetery, church, churchyard, architecture, overcast sky, weathervane, travel, UK, English Church, Europe, Faith, architecture, building, car, cemetery, church, church spire, churchyard, clock tower, battlements, black, brown, building, tourism, travel, weathervane, weather, weathervane, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather, weather
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

### ✅ meta-llama/Llama-3.2-11B-Vision-Instruct

**Metrics:** 3.89 TPS | 176 tokens

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

**Metrics:** 76.3 TPS | 500 tokens

```text
Title: St Michael's Church, Bishop's Stortford, England

Description: A low-angle view of St Michael's Parish Church in Bishop's Stortford, England, showcasing its historic stone walls, arched Gothic windows, and prominent spire. The churchyard is visible with gravestones and a bench, under an overcast sky. The scene is quiet and solemn, with the soft light of a cloudy day highlighting the textures of the ancient building.

Keywords: Bishop's Stortford, British, Christian, Christianity, England, English Church, Europe, Faith, Flint Knapped Wall, Flint Wall, Gothic Window, Grass, Graveyard, Hertfordshire, Parish church, Perpendicular Gothic, St Michael's Church, St Michael's Parish Church, UK, United Kingdom, ancient, arched window, battlements, black, brown, building, car, cemetery, church, church spire, churchyard, clock tower, cloudy, cloudy sky, cross, exterior, gothic architecture, gravestone, grey, headstone, heritage, historic, historic building, landmark, medieval, old, overcast, overcast sky, place of worship, religion, religious building, spire, stone, tombstone, tourism, travel, weathervane, worship, churchyard, church, church spire, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard, churchyard
```

---

### ✅ mlx-community/Qwen3-VL-2B-Thinking-bf16

**Metrics:** 75.7 TPS | 500 tokens

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

**Metrics:** 178 TPS | 500 tokens

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
- `mlx`: `0.30.7.dev20260214+c184262d`
- `mlx-lm`: `0.30.7`
- `mlx-metal`: ``
- `mlx-vlm`: `0.3.12`
- `numpy`: `2.4.2`
- `tokenizers`: `0.22.2`
- `transformers`: `5.1.0`

_Report generated on: 2026-02-14 18:19:37 GMT_
