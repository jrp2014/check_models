# Model Performance Results

_Generated on 2026-02-08 00:01:56 GMT_

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/nanoLLaVA-1.5-4bit` (317.0 tps)
- **💾 Most efficient:** `mlx-community/nanoLLaVA-1.5-4bit` (3.0 GB)
- **⚡ Fastest load:** `HuggingFaceTB/SmolVLM-Instruct` (0.00s)
- **📊 Average TPS:** 56.5 across 30 models

## 📈 Resource Usage

- **Total peak memory:** 591.6 GB
- **Average peak memory:** 19.7 GB
- **Memory efficiency:** 184 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** ✅ B: 6 | 🟡 C: 6 | 🟠 D: 8 | ❌ F: 10

**Average Utility Score:** 43/100

- **Best for cataloging:** `mlx-community/gemma-3n-E4B-it-bf16` (✅ B, 73/100)
- **Worst for cataloging:** `mlx-community/Molmo-7B-D-0924-8bit` (❌ F, 0/100)

### ⚠️ 18 Models with Low Utility (D/F)

- `Qwen/Qwen3-VL-2B-Instruct`: 🟠 D (46/100) - Mostly echoes context without adding value
- `microsoft/Phi-3.5-vision-instruct`: ❌ F (28/100) - Mostly echoes context without adding value
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (7/100) - Output too short to be useful
- `mlx-community/InternVL3-14B-8bit`: 🟠 D (40/100) - Mostly echoes context without adding value
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`: ❌ F (35/100) - Mostly echoes context without adding value
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`: 🟠 D (47/100) - Mostly echoes context without adding value
- `mlx-community/Molmo-7B-D-0924-8bit`: ❌ F (0/100) - Empty or minimal output
- `mlx-community/Molmo-7B-D-0924-bf16`: ❌ F (0/100) - Empty or minimal output
- `mlx-community/Phi-3.5-vision-instruct-bf16`: ❌ F (28/100) - Mostly echoes context without adding value
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: 🟠 D (40/100) - Mostly echoes context without adding value
- `mlx-community/Qwen3-VL-2B-Thinking-bf16`: 🟠 D (44/100) - Lacks visual description of image
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`: ❌ F (26/100) - Mostly echoes context without adding value
- `mlx-community/llava-v1.6-mistral-7b-8bit`: 🟠 D (39/100) - Mostly echoes context without adding value
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: ❌ F (12/100) - Output lacks detail
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (22/100) - Lacks visual description of image
- `mlx-community/pixtral-12b-8bit`: 🟠 D (49/100) - Mostly echoes context without adding value
- `mlx-community/pixtral-12b-bf16`: 🟠 D (49/100) - Mostly echoes context without adding value
- `qnguyen3/nanoLLaVA`: ❌ F (30/100) - Mostly echoes context without adding value

## ⚠️ Quality Issues

- **❌ Failed Models (10):**
  - `microsoft/Florence-2-large-ft` (`Weight Mismatch`)
  - `mlx-community/GLM-4.6V-Flash-6bit` (`Model Error`)
  - `mlx-community/GLM-4.6V-Flash-mxfp4` (`Model Error`)
  - `mlx-community/Idefics3-8B-Llama3-bf16` (`Model Error`)
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit` (`Model Error`)
  - `mlx-community/LFM2-VL-1.6B-8bit` (`Model Error`)
  - `mlx-community/LFM2.5-VL-1.6B-bf16` (`Model Error`)
  - `mlx-community/gemma-3n-E2B-4bit` (`No Chat Template`)
  - `mlx-community/paligemma2-3b-pt-896-4bit` (`Model Error`)
  - `prince-canuma/Florence-2-large-ft` (`Model Error`)
- **🔄 Repetitive Output (7):**
  - `Qwen/Qwen3-VL-2B-Instruct` (token: `phrase: "architectural preservation, ar..."`)
  - `microsoft/Phi-3.5-vision-instruct` (token: `phrase: "traditional, modern, historic,..."`)
  - `mlx-community/InternVL3-14B-8bit` (token: `phrase: "skeleton in window display,..."`)
  - `mlx-community/Phi-3.5-vision-instruct-bf16` (token: `phrase: "traditional, modern, historic,..."`)
  - `mlx-community/Qwen2-VL-2B-Instruct-4bit` (token: `phrase: "urban environment, urban stree..."`)
  - `mlx-community/SmolVLM2-2.2B-Instruct-mlx` (token: `phrase: "- english market town..."`)
  - `mlx-community/paligemma2-3b-ft-docci-448-bf16` (token: `phrase: "the windows are mostly..."`)

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 56.5 | Min: 0 | Max: 317
- **Peak Memory**: Avg: 20 | Min: 3.0 | Max: 73
- **Total Time**: Avg: 34.59s | Min: 3.17s | Max: 172.82s
- **Generation Time**: Avg: 31.40s | Min: 2.24s | Max: 171.09s
- **Model Load Time**: Avg: 3.18s | Min: 0.93s | Max: 8.36s

## 🚨 Failures by Package (Actionable)

<!-- markdownlint-disable MD060 -->

| Package | Failures | Error Types | Affected Models |
|---------|----------|-------------|-----------------|
| `mlx-vlm` | 8 | Model Error, No Chat Template | `mlx-community/GLM-4.6V-Flash-6bit`, `mlx-community/GLM-4.6V-Flash-mxfp4`, `mlx-community/Kimi-VL-A3B-Thinking-8bit`, `mlx-community/LFM2-VL-1.6B-8bit`, `mlx-community/LFM2.5-VL-1.6B-bf16`, `mlx-community/gemma-3n-E2B-4bit`, `mlx-community/paligemma2-3b-pt-896-4bit`, `prince-canuma/Florence-2-large-ft` |
| `mlx` | 2 | Model Error, Weight Mismatch | `microsoft/Florence-2-large-ft`, `mlx-community/Idefics3-8B-Llama3-bf16` |

<!-- markdownlint-enable MD060 -->

### Actionable Items by Package

#### mlx-vlm

- **mlx-community/GLM-4.6V-Flash-6bit** (Model Error)
  - Error: `Model generation failed for mlx-community/GLM-4.6V-Flash-6bit: [broadcast_shapes] Shapes (3,1,2048) and (3,1,6116) ca...`
  - Type: `ValueError`
- **mlx-community/GLM-4.6V-Flash-mxfp4** (Model Error)
  - Error: `Model generation failed for mlx-community/GLM-4.6V-Flash-mxfp4: [broadcast_shapes] Shapes (3,1,2048) and (3,1,6116) c...`
  - Type: `ValueError`
- **mlx-community/Kimi-VL-A3B-Thinking-8bit** (Model Error)
  - Error: `Model generation failed for mlx-community/Kimi-VL-A3B-Thinking-8bit: [broadcast_shapes] Shapes (975,2048) and (1,0,20...`
  - Type: `ValueError`
- **mlx-community/LFM2-VL-1.6B-8bit** (Model Error)
  - Error: `Model generation failed for mlx-community/LFM2-VL-1.6B-8bit: Image features and image tokens do not match: tokens: 25...`
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

> **Prompt used:**
>
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
> Context: The image is described as ', Windhill, Bishop's Stortford, England, United Kingdom, UK
A row of historic timber-framed buildings lines the narrow street of Windhill in the market town of Bishop's Stortford, Hertfordshire, England. Photographed on a cool late afternoon in February, the scene is dominated by the distinctive black-and-white Tudor-style architecture, with its weathered, moss-covered tile roofs set against an overcast winter sky.

These Grade II listed structures, with parts dating back to the 17th century, stand as a testament to the town's long history. Today, they serve a mix of purposes, housing private residences alongside modern businesses. On the right, the warm lights of the "Sabrosa del Fuego" tapas restaurant create a welcoming contrast to the muted afternoon light, illustrating how contemporary life continues within these centuries-old walls. The image captures the enduring character of an English town, where historical architecture is an integral part of the everyday streetscape.'.
> Existing keywords: Adobe Stock, Any Vision, Bishop's Stortford, British, Chalkboard sign, England, English Street Scene, English pub, Europe, Hertfordshire, High Street, Historic pub exterior, Objects, Quaint town, Santa hat, Skeleton decoration, Skeleton in window, Tudor Architecture, Tudor Style, UK, United Kingdom, Windhill, Winter in England, architecture, bare tree, black, brick, brick chimney, brown, building, chimney, cloudy, european, evening, exterior, facade, half-timbered, half-timbered building, historic building, landmark, moss, no people, old building, overcast, overcast sky, pavement, pub, restaurant, road, road sign, skeleton, street lamp, street scene, tiled roof, tourism, town, townscape, traditional, traffic sign, travel, wheelie bin, white, window, winter
> Taken around 2026-02-07 16:04:43 GMT at 16:04:43.
>
> Be factual about visual content.  Include relevant conceptual and emotional keywords where clearly supported by the image.

**Note:** Results sorted: errors first, then by generation time (fastest to slowest).

**Overall runtime:** 1065.47s

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
| `mlx-community/gemma-3n-E2B-4bit`                       |         |                   |                       |                |              |           |             |                  |            |             |                                   |         mlx-vlm |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |         |                   |                       |                |              |           |             |                  |            |             |                                   |         mlx-vlm |
| `prince-canuma/Florence-2-large-ft`                     |         |                   |                       |                |              |           |             |                  |            |             |                                   |         mlx-vlm |
| `mlx-community/nanoLLaVA-1.5-4bit`                      |     198 |               669 |                   500 |          1,169 |        2,493 |       317 |         3.0 |            2.24s |      0.93s |       3.17s |                                   |                 |
| `qnguyen3/nanoLLaVA`                                    | 151,645 |               669 |                   184 |            853 |        2,756 |      94.8 |         5.0 |            2.58s |      1.05s |       3.63s |                                   |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |       1 |             1,687 |                    15 |          1,702 |          466 |      42.1 |          12 |            4.27s |      3.26s |       7.53s | context-ignored                   |                 |
| `HuggingFaceTB/SmolVLM-Instruct`                        |  49,154 |             1,884 |                   466 |          2,350 |        1,343 |       102 |         5.6 |            6.33s |      1.26s |       7.59s | context-ignored                   |                 |
| `mlx-community/SmolVLM-Instruct-bf16`                   |  49,154 |             1,884 |                   466 |          2,350 |        1,299 |      99.9 |         5.6 |            6.53s |      1.33s |       7.86s | context-ignored                   |                 |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |     198 |             1,784 |                   500 |          2,284 |        1,302 |      99.8 |         5.5 |            6.72s |      1.34s |       8.06s | repetitive(phrase: "- english...  |                 |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 |  96,584 |             1,622 |                   500 |          2,122 |          869 |       103 |          18 |            7.23s |      3.27s |      10.50s | verbose                           |                 |
| `mlx-community/deepseek-vl2-8bit`                       |       1 |             2,452 |                   240 |          2,692 |          857 |      59.6 |          32 |            7.33s |      5.33s |      12.66s | context-ignored                   |                 |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |       2 |             3,213 |                   465 |          3,678 |          891 |       128 |         7.5 |            7.57s |      2.48s |      10.05s |                                   |                 |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |       1 |             1,687 |                    20 |          1,707 |          479 |       5.1 |          28 |            7.75s |      4.85s |      12.60s | context-ignored                   |                 |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |     106 |               933 |                   366 |          1,299 |          903 |      40.9 |          17 |           10.30s |      4.98s |      15.28s |                                   |                 |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |  54,217 |             1,622 |                   500 |          2,122 |          864 |      58.2 |          37 |           13.74s |      8.36s |      22.10s | verbose                           |                 |
| `mlx-community/pixtral-12b-8bit`                        |   8,351 |             3,395 |                   500 |          3,895 |          483 |      29.9 |          16 |           24.07s |      3.09s |      27.16s | verbose                           |                 |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |     106 |               933 |                   409 |          1,342 |          172 |      21.7 |          19 |           24.60s |      5.10s |      29.70s | verbose                           |                 |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |   1,897 |             3,486 |                   500 |          3,986 |          386 |      31.3 |          15 |           25.33s |      3.13s |      28.45s |                                   |                 |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |  10,297 |             2,913 |                   500 |          3,413 |          243 |      37.3 |          11 |           25.75s |      1.78s |      27.52s |                                   |                 |
| `mlx-community/InternVL3-14B-8bit`                      |   3,241 |             2,452 |                   500 |          2,952 |          393 |      24.6 |          17 |           27.07s |      3.39s |      30.46s | repetitive(phrase: "skeleton i... |                 |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |       2 |             2,717 |                   339 |          3,056 |          198 |      21.4 |          22 |           29.89s |      3.89s |      33.78s | ⚠️harness(encoding)               |                 |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |     611 |             1,687 |                   500 |          2,187 |        1,501 |      17.0 |          12 |           30.83s |      3.00s |      33.82s | repetitive(phrase: "the window... |                 |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |     106 |               933 |                   387 |          1,320 |          186 |      13.3 |          34 |           34.50s |      7.07s |      41.57s | verbose                           |                 |
| `mlx-community/Molmo-7B-D-0924-8bit`                    |         |               0.0 |                   0.0 |            0.0 |            0 |         0 |          41 |           34.72s |      2.49s |      37.20s |                                   |                 |
| `mlx-community/Molmo-7B-D-0924-bf16`                    |         |               0.0 |                   0.0 |            0.0 |            0 |         0 |          48 |           35.41s |      3.17s |      38.59s |                                   |                 |
| `mlx-community/pixtral-12b-bf16`                        |   1,044 |             3,395 |                   500 |          3,895 |          518 |      17.3 |          27 |           35.72s |      4.91s |      40.63s | verbose                           |                 |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |  29,892 |             1,536 |                   500 |          2,036 |          988 |      11.4 |          12 |           45.74s |      1.75s |      47.49s | repetitive(phrase: "traditiona... |                 |
| `microsoft/Phi-3.5-vision-instruct`                     |  29,892 |             1,536 |                   500 |          2,036 |        1,028 |      11.0 |          12 |           47.36s |      1.88s |      49.24s | repetitive(phrase: "traditiona... |                 |
| `Qwen/Qwen3-VL-2B-Instruct`                             |  42,463 |            16,936 |                   500 |         17,436 |          361 |      71.7 |         8.3 |           54.35s |      1.77s |      56.12s | repetitive(phrase: "architectu... |                 |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | 128,009 |               650 |                   213 |            863 |          156 |      3.67 |          25 |           62.58s |      4.20s |      66.78s | context-ignored                   |                 |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |   3,897 |            16,938 |                   500 |         17,438 |          262 |      67.9 |         8.3 |           72.52s |      1.95s |      74.48s | context-ignored                   |                 |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      |     596 |               649 |                   500 |          1,149 |          140 |      6.84 |          15 |           78.02s |      2.77s |      80.79s | verbose                           |                 |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |  28,322 |            16,947 |                   500 |         17,447 |          101 |       156 |          73 |          171.09s |      1.73s |     172.82s | repetitive(phrase: "urban envi... |                 |

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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6855, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6821, in _load_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6966, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6863, in _run_model_generation
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
> [broadcast_shapes] Shapes (3,1,2048) and (3,1,6116) cannot be broadcast.
**Type:** `ValueError`
**Package:** `mlx-vlm`

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6893, in _run_model_generation
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
ValueError: [broadcast_shapes] Shapes (3,1,2048) and (3,1,6116) cannot be broadcast.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6966, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6918, in _run_model_generation
    raise ValueError(msg) from gen_known_err
ValueError: Model generation failed for mlx-community/GLM-4.6V-Flash-6bit: [broadcast_shapes] Shapes (3,1,2048) and (3,1,6116) cannot be broadcast.
```

</details>

---

### ❌ mlx-community/GLM-4.6V-Flash-mxfp4

**Status:** Failed (Model Error)
**Error:**

> Model generation failed for mlx-community/GLM-4.6V-Flash-mxfp4:
> [broadcast_shapes] Shapes (3,1,2048) and (3,1,6116) cannot be broadcast.
**Type:** `ValueError`
**Package:** `mlx-vlm`

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6893, in _run_model_generation
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
ValueError: [broadcast_shapes] Shapes (3,1,2048) and (3,1,6116) cannot be broadcast.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6966, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6918, in _run_model_generation
    raise ValueError(msg) from gen_known_err
ValueError: Model generation failed for mlx-community/GLM-4.6V-Flash-mxfp4: [broadcast_shapes] Shapes (3,1,2048) and (3,1,6116) cannot be broadcast.
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6855, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6821, in _load_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6966, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6863, in _run_model_generation
    raise ValueError(error_details) from load_err
ValueError: Model loading failed: Expected shape (4096, 4096) but received shape (1024, 4096) for parameter language_model.layers.0.self_attn.k_proj.weight
```

</details>

---

### ❌ mlx-community/Kimi-VL-A3B-Thinking-8bit

**Status:** Failed (Model Error)
**Error:**

> Model generation failed for mlx-community/Kimi-VL-A3B-Thinking-8bit:
> [broadcast_shapes] Shapes (975,2048) and (1,0,2048) cannot be broadcast.
**Type:** `ValueError`
**Package:** `mlx-vlm`

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6893, in _run_model_generation
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
ValueError: [broadcast_shapes] Shapes (975,2048) and (1,0,2048) cannot be broadcast.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6966, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6918, in _run_model_generation
    raise ValueError(msg) from gen_known_err
ValueError: Model generation failed for mlx-community/Kimi-VL-A3B-Thinking-8bit: [broadcast_shapes] Shapes (975,2048) and (1,0,2048) cannot be broadcast.
```

</details>

---

### ❌ mlx-community/LFM2-VL-1.6B-8bit

**Status:** Failed (Model Error)
**Error:**

> Model generation failed for mlx-community/LFM2-VL-1.6B-8bit: Image features
> and image tokens do not match: tokens: 253, features 260
**Type:** `ValueError`
**Package:** `mlx-vlm`

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6893, in _run_model_generation
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
ValueError: Image features and image tokens do not match: tokens: 253, features 260

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6966, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6918, in _run_model_generation
    raise ValueError(msg) from gen_known_err
ValueError: Model generation failed for mlx-community/LFM2-VL-1.6B-8bit: Image features and image tokens do not match: tokens: 253, features 260
```

</details>

---

### ❌ mlx-community/LFM2.5-VL-1.6B-bf16

**Status:** Failed (Model Error)
**Error:**

> Model generation failed for mlx-community/LFM2.5-VL-1.6B-bf16: Image
> features and image tokens do not match: tokens: 253, features 260
**Type:** `ValueError`
**Package:** `mlx-vlm`

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6893, in _run_model_generation
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
ValueError: Image features and image tokens do not match: tokens: 253, features 260

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6966, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6918, in _run_model_generation
    raise ValueError(msg) from gen_known_err
ValueError: Model generation failed for mlx-community/LFM2.5-VL-1.6B-bf16: Image features and image tokens do not match: tokens: 253, features 260
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6966, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6867, in _run_model_generation
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6893, in _run_model_generation
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6966, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6918, in _run_model_generation
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6855, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6821, in _load_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6966, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 6863, in _run_model_generation
    raise ValueError(error_details) from load_err
ValueError: Model loading failed: RobertaTokenizer has no attribute additional_special_tokens
```

</details>

---

### ✅ mlx-community/nanoLLaVA-1.5-4bit

**Metrics:** 317 TPS | 500 tokens

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

Context: The image is described as 'Windhill, Bishop's Stortford, England, United Kingdom, UK. A row of historic timber-framed buildings lines the narrow street of Windhill in the market town of Bishop's Stortford, Hertfordshire, England. Photographed on a cool late afternoon in February, the scene is dominated by the distinctive black-and-white Tudor-style architecture, with its weathered, moss-covered tile roofs set against an overcast winter sky. These Grade II listed structures, with parts dating back to the 17th century, stand as a testament to the town's long history. Today, they serve a mix of purposes, housing private residences alongside modern businesses. On the right, the warm lights of the 'Sabrosa del Fuego' tapas restaurant create a welcoming contrast to the muted afternoon light, illustrating how contemporary life continues within these centuries-old walls. The image captures the enduring character of an English town, where historical architecture is an integral part of the everyday streetscape.'.
Existing keywords: Adobe Stock, Any Vision, Bishop's Stortford, British, Chalkboard sign, England, English Street Scene, English pub, Europe, Hertfordshire, High Street, Historic pub exterior, Objects, Quaint town, Santa hat, Skeleton decoration, Skeleton in window, Tudor Architecture, Tudor Style, UK, United Kingdom, Windhill, Winter in England, architecture, bare tree, black, brick, brick chimney, brown, building, chimney, overcast, overcast sky, pavement, window, winter

```

---

### ✅ qnguyen3/nanoLLaVA

**Metrics:** 94.8 TPS | 184 tokens

```text
The image captures a serene, historical street scene in Bishop's Stortford, England. The town is characterized by its historic architecture, including a row of Grade II listed buildings. The street is lined with a variety of structures, including a black-and-white Tudor-style building, a black and white stone building, and a black and white brick building. The buildings are set against an overcast winter sky, and the street is lined with a variety of objects, including a black and white stone chimney, a black and white brick chimney, and a black and white stone building. The scene is further enhanced by the presence of a black and white stone building with a skeleton decoration, a black and white stone building with a white window, and a black and white stone building with a white chimney. The image captures the enduring character of an English town, where historical architecture is an integral part of the everyday streetscape.
```

---

### ✅ mlx-community/paligemma2-10b-ft-docci-448-6bit

**Metrics:** 42.1 TPS | 15 tokens

```text
Be specific about the image. Include as many relevant keywords as possible.
```

---

### ✅ HuggingFaceTB/SmolVLM-Instruct

**Metrics:** 102 TPS | 466 tokens

```text
 Image Description:

The image depicts a row of historic timber-framed buildings lining a narrow street in a quaint town. The buildings are primarily constructed with wooden beams and have a distinctive black-and-white color scheme, giving them a charming and rustic appearance. The buildings are of varying heights, with the tallest one being in the center of the image. The roofs of the buildings are covered in dark shingles, and there are several chimneys visible, indicating that they are traditional English homes.

The street in front of the buildings is narrow and paved with cobblestones, which adds to the quaint and charming atmosphere of the town. The street is lined with a few trees, which are bare, suggesting that it might be winter or early spring. The sky is overcast, with a mix of grey and light blue, which adds to the overall mood of the image.

On the right side of the image, there is a modern business establishment, the "Sabrosa del Fuego," which is a tapas restaurant. The restaurant has a warm and inviting ambiance, with warm lighting and a colorful sign hanging outside. The sign features a fiery orange flame, which is a common symbol for Spanish cuisine.

The buildings in the image are primarily made of wood and have a traditional, half-timbered style, which is a common feature in English architecture. The windows are small and have a traditional design, with black frames and white shutters. The buildings are well-maintained, with some moss growing on the walls, indicating that they have been well-cared for.

The image captures the essence of a small town in England, with its historic architecture and traditional charm. The presence of a modern business establishment, the "Sabrosa del Fuego," adds a touch of modernity to the scene, highlighting the town's blend of old and new.

In summary, the image showcases a row of historic timber-framed buildings in a quaint English town, with a modern business establishment on the right side. The buildings have a traditional, half-timbered style, and the street is lined with cobblestones. The image captures the enduring character of an English town, with its blend of historical architecture and modern businesses.
```

---

### ✅ mlx-community/SmolVLM-Instruct-bf16

**Metrics:** 99.9 TPS | 466 tokens

```text
 Image Description:

The image depicts a row of historic timber-framed buildings lining a narrow street in a quaint town. The buildings are primarily constructed with wooden beams and have a distinctive black-and-white color scheme, giving them a charming and rustic appearance. The buildings are of varying heights, with the tallest one being in the center of the image. The roofs of the buildings are covered in dark shingles, and there are several chimneys visible, indicating that they are traditional English homes.

The street in front of the buildings is narrow and paved with cobblestones, which adds to the quaint and charming atmosphere of the town. The street is lined with a few trees, which are bare, suggesting that it might be winter or early spring. The sky is overcast, with a mix of grey and light blue, which adds to the overall mood of the image.

On the right side of the image, there is a modern business establishment, the "Sabrosa del Fuego," which is a tapas restaurant. The restaurant has a warm and inviting ambiance, with warm lighting and a colorful sign hanging outside. The sign features a fiery orange flame, which is a common symbol for Spanish cuisine.

The buildings in the image are primarily made of wood and have a traditional, half-timbered style, which is a common feature in English architecture. The windows are small and have a traditional design, with black frames and white shutters. The buildings are well-maintained, with some moss growing on the walls, indicating that they have been well-cared for.

The image captures the essence of a small town in England, with its historic architecture and traditional charm. The presence of a modern business establishment, the "Sabrosa del Fuego," adds a touch of modernity to the scene, highlighting the town's blend of old and new.

In summary, the image showcases a row of historic timber-framed buildings in a quaint English town, with a modern business establishment on the right side. The buildings have a traditional, half-timbered style, and the street is lined with cobblestones. The image captures the enduring character of an English town, with its blend of historical architecture and modern businesses.
```

---

### ✅ mlx-community/SmolVLM2-2.2B-Instruct-mlx

**Metrics:** 99.8 TPS | 500 tokens

```text
 Title: Tudor-style building in Bishop's Stortford, Hertfordshire, England.

Description: A row of historic timber-framed buildings line the narrow street of Windhill in the market town of Bishop's Stortford, Hertfordshire, England. Photographed on a cool late afternoon in February, the scene is dominated by the distinctive black-and-white Tudor-style architecture, with its weathered, moss-covered tile roofs set against an overcast winter sky. These Grade II listed structures, with parts dating back to the 17th century, stand as a testament to the town's long history. Today, they serve a mix of purposes, housing private residences alongside modern businesses. On the right, the warm lights of the "Sabrosa del Fuego" tapas restaurant create a welcoming contrast to the muted afternoon light, illustrating how contemporary life continues within these centuries-old walls. The image captures the enduring character of an English town, where historical architecture is an integral part of the everyday streetscape.

Keywords:
- Tudor-style building
- Hertfordshire, England
- Bishop's Stortford, Hertfordshire, England
- Historic pub exterior
- English pub
- English town
- English street scene
- English townscape
- English market town
- English market town
- English market town
- English market town
- English market town
- English market town
- English market town
- English market town
- English market town
- English market town
- English market town
- English market town
- English market town
- English market town
- English market town
- English market town
- English market town
- English market town
- English market town
- English market town
- English market town
- English market town
- English market town
- English market town
- English market town
- English market town
- English market town
- English market town
- English market town
- English market town
- English market town
- English market town
- English market town
- English market town
- English market town
- English market town
- English market town
- English market town
- English market town
- English market town
- English market town
- English market town
- English market town
- English market town
- English market town
- English market town

```

---

### ✅ jqlive/Kimi-VL-A3B-Thinking-2506-6bit

**Metrics:** 103 TPS | 500 tokens

```text
◁think▷Got it, let's tackle this. First, I need to create a title, description, and keywords for the image. Let's start with the title. The image shows a row of historic timber-framed buildings in Windhill, Bishop's Stortford, England. The key elements are the Tudor-style architecture, the street scene, and the "Sabrosa del Fuego" restaurant. So a concise title could be "Historic Tudor-style buildings and a tapas restaurant in Windhill, England."

Next, the description. It needs to be 1-3 sentences. The image depicts a row of historic timber-framed buildings in Windhill, a market town in Hertfordshire, UK. The scene is set on a cool late afternoon in February, with Tudor architecture featuring weathered roofs and black-and-white timber framing. The "Sabrosa del Fuego" tapas restaurant is visible, with warm lights contrasting the overcast sky. These Grade II listed buildings, dating to the 17th century, blend historical charm with modern use as residences and businesses.

Now keywords. Let's list them from most specific to general. Specific subjects: timber-framed buildings, Tudor architecture, "Sabrosa del Fuego" restaurant, chimneys, street sign ("HIGH ST"), windows, doors, roof tiles. Setting: urban, outdoor, street scene. Actions/activities: none prominent, but the restaurant's lights are on, indicating it's open. Concepts/themes: historical preservation, blend of old and new, community. Mood/emotion: serene, quaint, cozy. Visual style: wide-angle, daytime, overcast. Colors: white, black, brown, muted tones. Seasonal: winter (bare tree, overcast sky, February). Use-case relevance: travel, tourism, editorial.

Wait, need to check the existing keywords provided. They have some, but I should include relevant ones. Let's compile:

Keywords: Tudor architecture, timber-framed buildings, "Sabrosa del Fuego" tapas restaurant, Windhill, England, Hertfordshire, UK, High Street, Grade II listed, 17th century, half-timbered, black-and-white timber framing, overcast winter, street scene, historic pub exterior, residential and commercial use, warm interior lighting, architectural heritage, European townscape, winter in England, bare tree, chimney, street sign, paved road, tourism, traditional architecture, half-timbered building, historic building, landmark, moss
```

---

### ✅ mlx-community/deepseek-vl2-8bit

**Metrics:** 59.6 TPS | 240 tokens

```text
- **Title:** "Historic Street Scene with Timeless Charm and Modern Vibe"

- **Description:** The image captures a charming street in a quaint English town, likely in the UK, with a focus on a row of traditional, half-timbered buildings. The scene is set on a cloudy day, with a mix of overcast and overcast weather. The street is lined with a mix of old and new elements, with a mix of modern and historical architecture. The atmosphere is enhanced by the presence of a pub, "Sabrosa del Fuego," which has a warm, inviting glow. The street is quiet, with no visible people, adding to the serene and peaceful ambiance. The overall scene exudes a sense of history and coziness, with the overcast sky adding a soft, diffused light that highlights the textures and colors of the buildings. The scene is a beautiful blend of the old and the new, with the presence of a modern sign for a tapas bar and a traditional pub, suggesting a vibrant local culture. The overall mood is one of peace and nostalgia, with the overcast sky adding a touch of melancholy and calm to the scene.
```

---

### ✅ mlx-community/Ministral-3-3B-Instruct-2512-4bit

**Metrics:** 128 TPS | 465 tokens

```text
Title: **Historic Tudor Row: Windhill’s Charming Market Town**

---
**Description:**
The image depicts a row of traditional Tudor-style buildings on a narrow street in Windhill, Bishop’s Stortford, England. The architecture features exposed timber framing, white plaster walls, and weathered moss-covered slate roofs. The scene is calm and slightly muted, with an overcast sky casting a cool tone. On the right, a modern business—"Sabrosa del Fuego" (a tapas/restaurant)—brightens the facade with warm lighting, blending historical charm with contemporary use.

---
**Keywords:**
Historic Tudor building, English heritage architecture, quaint market town, narrow street, half-timbered facade, white plaster walls, moss-covered roof, brick chimneys, overcast winter sky, dim lighting, street-side business, tapas restaurant, warm interior glow, vintage pub exterior, skeletal window decoration, black-and-white aesthetic, quaint English town, old-world charm, nostalgic streetscape, architectural preservation, late afternoon ambiance, moody lighting, traditional pub, heritage conservation, urban-rural blend, nostalgic atmosphere, architectural detail, street lamps, cobblestone-like pavement, small-town England, winter season, English countryside, timeless setting, nostalgic vibe, historic ambiance, vintage aesthetic, street signage, roadside establishment, architectural heritage, English pub culture, winter street scene, classic Tudor style, old-world elegance, atmospheric photography, street scene with lighting contrast, heritage-listed buildings, late winter day, cool tones, muted colors, overcast winter sky, street-level view, architectural landmark, English market town, historic facade, warm lighting contrast, skeleton-themed decor, quaint English pub exterior.

---
**Context:**
The image captures a serene, historically rich street in Windhill, Bishop’s Stortford, England, showcasing the enduring appeal of Tudor-style architecture. The juxtaposition of old-world charm with a modern tapas restaurant highlights the town’s blend of heritage and contemporary life. The overcast lighting and skeletal window decoration evoke a nostalgic, slightly melancholic mood, typical of late-winter English scenes. This setting is ideal for tourism, real estate, or historical documentation.
```

---

### ✅ mlx-community/paligemma2-10b-ft-docci-448-bf16

**Metrics:** 5.1 TPS | 20 tokens

```text
Be specific about visual content. Include relevant object and action keywords where clearly supported by the image.
```

---

### ✅ mlx-community/gemma-3n-E4B-it-bf16

**Metrics:** 40.9 TPS | 366 tokens

```text
## Image Description: Quaint Historic Street in Bishop's Stortford

**Title:** Historic Half-Timbered Buildings on a UK High Street

**Description:** A charming street scene in Bishop's Stortford, UK, showcases a row of well-preserved, historic half-timbered buildings under a cloudy sky. A modern tapas restaurant adds a touch of contemporary life to the traditional architecture.

**Keywords:**

* **People:** None visible
* **Objects:** Buildings, street sign, windows, doors, signboards, street lamps, trash bin, pavement, road, awning, menu board
* **Setting & Location:** Town, street, UK, England, Bishop's Stortford, Hertfordshire, high street, urban
* **Actions & Activities:** Standing, signage, business
* **Concepts & Themes:** History, tradition, architecture, local life, dining, quaintness, heritage
* **Mood & Emotion:** Peaceful, historic, charming, inviting, slightly overcast
* **Visual Style:** Wide-angle, eye-level, daytime, exterior
* **Colors & Lighting:** White, black, brown, red, blue, grey, natural light, overcast
* **Composition & Perspective:** Horizontal, centered, detailed
* **Technical Details:** Adobe Stock, high-resolution, photograph, outdoor

**Context:** The image captures a picturesque view of a street in Bishop's Stortford, England. The buildings are notable for their distinctive black-and-white Tudor-style architecture, dating back to the 17th century. The presence of a modern tapas restaurant, "Sabrosa del Fuego," creates a blend of old and new within the historic setting. The overall atmosphere is one of a traditional English town with a touch of contemporary vibrancy.
```

---

### ✅ mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

**Metrics:** 58.2 TPS | 500 tokens

```text
◁think▷Got it, let's tackle this. First, I need to create structured metadata for the image. The user wants three sections: Title, Description, Keywords, and Context. Let's start with the title. The image shows a row of historic timber-framed buildings in Windhill, England, with Tudor architecture, a street scene, and a pub. So the title should be concise and descriptive. Maybe "Historic Timber-Framed Buildings in Windhill, England: Tudor Architecture and Urban Streetscape".

Next, the description. It needs to be 1-3 sentences. The image depicts a row of historic timber-framed buildings in Windhill, Hertfordshire, UK, showcasing Tudor-style architecture with black-and-white timber framing, moss-covered roofs, and a mix of residential and commercial use. The scene is set on a cool late afternoon in February, with a pub ("Sabrosa del Fuego") visible, blending historical charm with modern urban life. That covers key elements: subjects (buildings, pub), setting (urban, England), action (streetscape, architectural details), and mood (enduring character, blend of old and new).

Now keywords. Need 25-50, ordered from specific to general. Let's list them: Tudor Architecture, Timber-Framed Buildings, Windhill, England, Hertfordshire, UK, High Street, Historic Pub Exterior, Black-and-White Timber Framing, Moss-Covered Roofs, Urban Streetscape, Late Afternoon Light, Winter in England, Chalkboard Sign, Skeleton Decorations, Half-Timbered Buildings, Historic Building, Tourism, Traditional Architecture, Overcast Sky, Street Lamp, Road Sign, Traffic Sign, Private Residences, Modern Businesses, European Town, Architectural Details, Historical Preservation, Chalkboard Sign, Skeleton in Window, Urban Life, Seasonal Context (Winter), Architectural Style, Building Facade, Townscape, Historical Significance, Architectural Details, Urban Environment, Traditional Architecture, Historical Architecture, Urban Setting, Historic Town, Timber Framing, Urban Street, Historic Pub, Street Scene, Townscape, Historical Building, Urban Life, Seasonal Context, Architectural Details, Building Facade, Townscape, Historical Preservation, Urban Environment, Traditional Architecture, Historical Architecture, Urban Setting, Historic Town, Timber Framing, Urban Street, Historic Pub, Street Scene, Townscape, Historical Building, Urban Life, Seasonal Context, Architectural Details, Building Facade, Townscape, Historical
```

---

### ✅ mlx-community/pixtral-12b-8bit

**Metrics:** 29.9 TPS | 500 tokens

```text
### Title
Historic Tudor-Style Buildings on Windhill Street, Bishop's Stortford

### Description
The image depicts a row of historic Tudor-style buildings on Windhill Street in Bishop's Stortford, Hertfordshire, England. The buildings feature distinctive black-and-white timber framing and moss-covered tile roofs, set against an overcast winter sky. One of the buildings houses the "Sabrosa del Fuego" tapas restaurant, with warm lights inviting passersby.

### Keywords
Tudor Architecture, Historic Building, Historic Pub Exterior, Half-Timbered Building, Half-Timbered, Historic, England, British, Europe, Hertfordshire, Bishop's Stortford, Windhill, UK, United Kingdom, English Street Scene, English Pub, Quaint Town, Townscape, Town, Tourism, Travel, Traditional, Historic Architecture, Moss, Tile Roof, Tiled Roof, Brick Chimney, Chimney, Brick, Black, Brown, White, Window, Windows, Facade, Exterior, Street Scene, Street Lamp, Road, Road Sign, Traffic Sign, High Street, Overcast, Overcast Sky, Cloudy, Clouds, Winter, Winter in England, Bare Tree, Tree, Skeleton, Skeleton Decoration, Skeleton in Window, Santa Hat, Chalkboard Sign, Restaurant, Pub, Objects, No People, No Person, Historic, Enduring Character, Enduring, Character, Tourism, Travel, Tourism Attraction, Historic Town, Historic District, Historic Area, Historic Site, Heritage, Heritage Building, Heritage Architecture, Heritage Preservation, Heritage Tourism, Heritage Site, Heritage Structure, Heritage Town, Heritage City, Heritage Village, Heritage District, Heritage Area, Heritage Landmark, Heritage Monument, Heritage Memorial, Heritage Memorial Site, Heritage Memorial Structure, Heritage Memorial Building, Heritage Memorial Monument, Heritage Memorial Landmark, Heritage Memorial Area, Heritage Memorial District, Heritage Memorial Site, Heritage Memorial Structure, Heritage Memorial Building, Heritage Memorial Monument, Heritage Memorial Landmark, Heritage Memorial Area, Heritage Memorial District, Heritage Memorial Site, Heritage Memorial Structure, Heritage Memorial Building, Heritage Memorial Monument, Heritage Memorial Landmark, Heritage Memorial Area, Heritage Memorial District, Heritage Memorial Site, Heritage Memorial Structure, Heritage Memorial Building, Heritage Memorial Monument, Heritage Memorial Landmark, Heritage Memorial Area, Heritage Memorial District, Heritage Memorial Site, Heritage Memorial Structure, Heritage Memorial Building, Heritage Memorial Monument, Heritage Memorial Landmark
```

---

### ✅ mlx-community/gemma-3-27b-it-qat-4bit

**Metrics:** 21.7 TPS | 409 tokens

```text
## Title: Historic Tudor Buildings Line Windhill Street, Bishop's Stortford

## Description:
This image showcases a row of well-preserved, black-and-white timber-framed buildings along Windhill in Bishop's Stortford, England. A tapas restaurant, "Sabrosa del Fuego," occupies one of the buildings, with signage indicating its offerings. The scene is captured on a cool, overcast afternoon in February, highlighting the architectural character of the town.

## Keywords:
Tudor architecture, half-timbered building, Windhill, Bishop's Stortford, Hertfordshire, England, UK, United Kingdom, historic building, English town, street scene, townscape, architecture, building exterior, facade, black and white architecture, timber framing, 17th century architecture, Grade II listed building, pub, restaurant, tapas restaurant, Sabrosa del Fuego, chalkboard sign, street lamp, road, pavement, brick chimney, tiled roof, moss, overcast sky, cloudy, winter, February, late afternoon, cool light, muted tones, European architecture, traditional architecture, quaint town, landmark, street photography, urban landscape, building detail, exterior view, storefront, signage, road sign, wheelie bin, bare tree, chimney, brick, white, black, brown, evening light, historic preservation, cultural heritage, tourism, travel, European travel, English countryside, quaint, charming, cozy, welcoming, street view, storefront, pedestrian street, building interiors visible, window display, skeleton decoration, Santa hat, High Street, Adobe Stock, Any Vision, 2026, GMT, 16:04:43, exterior, building, landmark, moss, no people, old building, overcast, overcast sky, pavement, pub, restaurant, road, road sign, skeleton, street lamp, street scene, tiled roof, tourism, town, townscape, traditional, traffic sign, travel, wheelie bin, white, window, winter
```

---

### ✅ mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX

**Metrics:** 31.3 TPS | 500 tokens

```text
Here are my reasoning steps:
The user wants a structured metadata for image cataloguing. They want exactly three sections: Title, Description, Keywords. They also gave a context description and existing keywords. They want us to be factual about visual content, include relevant conceptual and emotional keywords where clearly supported. They want 25–50 comma-separated keywords ordered from most specific to most general, covering categories: specific subjects, setting and location, actions and activities, concepts and themes, mood and emotion, visual style, colors and lighting, seasonal and temporal context, use-case relevance.

We need to produce Title (5–12 words), Description (1–3 sentences), Keywords (25–50). Must be exactly these three sections. No extra text. Must be factual. Use the context and image description. The image shows a row of historic timber-framed buildings on a narrow street in Windhill, Bishop's Stortford, England. The architecture is Tudor style, black and white half-timbered, moss-covered tile roofs, overcast winter sky. There's a "Sabrosa del Fuego" tapas restaurant with warm lights. No people visible. There's a chalkboard sign, a street lamp, a wheelie bin, etc. The scene is a quiet street, no people, overcast, winter. The building is Grade II listed, parts from 17th century. The image is taken in late February, cool, overcast. The visual style is wide-angle street view, capturing the entire facade. Colors: muted, whites, blacks, browns, moss green, overcast sky. Lighting: diffused, cool, overcast. Mood: serene, historic, quiet, nostalgic, calm.

We need to produce keywords from most specific to most general. Let's think of specific subjects: "Sabrosa del Fuego tapas restaurant", "half-timbered Tudor building", "Grade II listed building", "black-and-white timber framing", "moss-covered tiled roof", "chalkboard sign", "street lamp", "wheelie bin", "overcast sky", "narrow street", "Windhill street", "Bishop's Stortford", "Hertfordshire", "England", "United Kingdom", "winter", "February", "cool overcast afternoon", "historic architecture", "residential and commercial mix", "no people",
```

---

### ✅ mlx-community/llava-v1.6-mistral-7b-8bit

**Metrics:** 37.3 TPS | 500 tokens

```text
Title: Historic Pub in Windhill, Bishop's Stortford, England

Description: The image showcases a quaint town street scene in Windhill, Bishop's Stortford, England, with a focus on a historic pub. The architecture is characterized by Tudor-style timber-framed buildings, and the scene is set against an overcast winter sky. The image captures the enduring character of an English town, where historical architecture is an integral part of the everyday streetscape.

Keywords: Historic pub, Windhill, Bishop's Stortford, England, United Kingdom, UK, Tudor-style architecture, timber-framed buildings, historic buildings, landmark, overcast sky, winter, street lamp, street scene, pavement, road sign, travel, tourism, traditional, town, townscape, architecture, wheelie bin, no people, skeleton, window, evening, brick chimney, brick, brown, moss, half-timbered, half-timbered building, road, road sign, chalkboard sign, pub, restaurant, European, Hertfordshire, evening, winter, overcast, skeleton, window, wheelie bin, brick chimney, brick, brown, moss, half-timbered, half-timbered building, road, road sign, chalkboard sign, pub, restaurant, European, Hertfordshire, evening, winter, overcast, skeleton, window, wheelie bin, brick chimney, brick, brown, moss, half-timbered, half-timbered building, road, road sign, chalkboard sign, pub, restaurant, European, Hertfordshire, evening, winter, overcast, skeleton, window, wheelie bin, brick chimney, brick, brown, moss, half-timbered, half-timbered building, road, road sign, chalkboard sign, pub, restaurant, European, Hertfordshire, evening, winter, overcast, skeleton, window, wheelie bin, brick chimney, brick, brown, moss, half-timbered, half-timbered building, road, road sign, chalkboard sign, pub, restaurant, European, Hertfordshire, evening, winter, overcast, skeleton, window, wheel
```

---

### ✅ mlx-community/InternVL3-14B-8bit

**Metrics:** 24.6 TPS | 500 tokens

```text
Title: Historic Tudor Buildings on High Street

Description: A row of traditional Tudor-style buildings with timber framing and moss-covered roofs lines a street, featuring a restaurant with illuminated signage and a skeleton decoration in a window.

Keywords: Tudor Architecture, timber-framed building, historic building, High Street, Bishop's Stortford, England, UK, overcast sky, cloudy day, skeleton decoration, restaurant, Sabrosa del Fuego, tapas restaurant, black-and-white facade, moss-covered roof, street scene, no people, quaint town, European architecture, Grade II listed, winter, skeleton in window, brick chimney, traditional pub, travel, tourism, urban setting, old buildings, warm lights, signage, skeleton, festive decoration, skeleton with Santa hat, skeleton in window display, skeleton decoration, skeleton figure, skeleton in window, skeleton in display, skeleton in shop window, skeleton in building window, skeleton in window display, skeleton in window, skeleton in window display, skeleton in window, skeleton in window display, skeleton in window, skeleton in window display, skeleton in window, skeleton in window display, skeleton in window, skeleton in window display, skeleton in window, skeleton in window display, skeleton in window, skeleton in window display, skeleton in window, skeleton in window display, skeleton in window, skeleton in window display, skeleton in window, skeleton in window display, skeleton in window, skeleton in window display, skeleton in window, skeleton in window display, skeleton in window, skeleton in window display, skeleton in window, skeleton in window display, skeleton in window, skeleton in window display, skeleton in window, skeleton in window display, skeleton in window, skeleton in window display, skeleton in window, skeleton in window display, skeleton in window, skeleton in window display, skeleton in window, skeleton in window display, skeleton in window, skeleton in window display, skeleton in window, skeleton in window display, skeleton in window, skeleton in window display, skeleton in window, skeleton in window display, skeleton in window, skeleton in window display, skeleton in window, skeleton in window display, skeleton in window, skeleton in window display, skeleton in window, skeleton in window display, skeleton in window, skeleton in window display, skeleton in window, skeleton in window display, skeleton in window, skeleton in window display, skeleton in window, skeleton in window display, skeleton in window, skeleton in window display, skeleton in window, skeleton in window display, skeleton in window
```

---

### ✅ mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

**Metrics:** 21.4 TPS | 339 tokens

```text
Title:ĠHistoricĠTudorĠBuildingsĠonĠWindhill,ĠBishop'sĠStortfordĊĊDescription:ĠTheĠimageĠshowsĠaĠrowĠofĠhistoricĠtimber-framedĠbuildingsĠonĠWindhillĠinĠBishop'sĠStortford,ĠEngland.ĠTheĠbuildings,ĠwithĠtheirĠdistinctiveĠblack-and-whiteĠTudorĠarchitecture,ĠareĠsetĠagainstĠanĠovercastĠwinterĠsky,ĠandĠoneĠofĠthemĠhousesĠaĠtapasĠrestaurantĠwithĠwarmĠinteriorĠlighting.ĊĊKeywords:ĠTudorĠarchitecture,ĠhistoricĠbuildings,ĠWindhill,ĠBishop'sĠStortford,ĠEngland,ĠUK,Ġtimber-framed,Ġblack-and-white,ĠovercastĠsky,Ġwinter,Ġevening,ĠstreetĠscene,ĠnarrowĠstreet,Ġmoss-coveredĠroofs,ĠtileĠroofs,Ġchimneys,Ġrestaurant,Ġtapas,ĠwarmĠlighting,ĠstreetĠlamps,ĠtrafficĠsign,Ġpavement,Ġroad,ĠbareĠtrees,ĠGradeĠIIĠlisted,Ġ17thĠcentury,ĠmarketĠtown,ĠHertfordshire,ĠEuropean,Ġquaint,Ġtraditional,Ġtourism,Ġtravel,Ġexterior,Ġfacade,Ġwindows,Ġdoors,Ġsignage,Ġchalkboard,ĠskeletonĠdecoration,ĠSantaĠhat,ĠwheelieĠbin,Ġurban,Ġrural,Ġhistoric,Ġcultural,Ġheritage,Ġarchitecture,Ġdesign,ĠurbanĠlandscape,Ġtown,Ġstreet,Ġbuildings,Ġexterior,Ġwinter,Ġovercast,Ġcloudy,Ġevening,Ġlight,Ġshadow,Ġcontrast,Ġtexture,Ġpattern,Ġdetail,Ġhistoric,Ġcultural,Ġheritage,Ġarchitecture,Ġdesign,ĠurbanĠlandscape,Ġtown,Ġstreet,Ġbuildings,Ġexterior,Ġwinter,Ġovercast,Ġcloudy,Ġevening,Ġlight,Ġshadow,Ġcontrast,Ġtexture,Ġpattern,ĠdetailĊĊContext:ĠTheĠimageĠcapturesĠtheĠenduringĠcharacterĠofĠanĠEnglishĠtown,ĠwhereĠhistoricalĠarchitectureĠisĠanĠintegralĠpartĠofĠtheĠeverydayĠstreetscape.ĠTheĠsceneĠillustratesĠhowĠcontemporaryĠlifeĠcontinuesĠwithinĠcenturies-oldĠwalls,ĠblendingĠhistoryĠwithĠmodern-dayĠactivities.
```

---

### ✅ mlx-community/paligemma2-3b-ft-docci-448-bf16

**Metrics:** 17 TPS | 500 tokens

```text
A row of historic half-timbered buildings stand along the narrow street of Windhill in the market town of Bishop's Stortford, Hertfordshire, England. The buildings are a mix of private residences and modern businesses. The exterior of the buildings is white with black trim, and the interior of the buildings is a mix of white and black. The exterior of the buildings is a mix of white and black trim, and the interior of the buildings is a mix of white and black. The windows of the buildings are black, with a black frame. The windows are mostly rectangular, with a small square on the left side of the image. The windows are mostly white with black trim. The windows are mostly rectangular, with a small square on the left side of the image. The windows are mostly white with black trim. The windows are mostly rectangular, with a small square on the left side of the image. The windows are mostly white with black trim. The windows are mostly rectangular, with a small square on the left side of the image. The windows are mostly white with black trim. The windows are mostly rectangular, with a small square on the left side of the image. The windows are mostly white with black trim. The windows are mostly rectangular, with a small square on the left side of the image. The windows are mostly white with black trim. The windows are mostly rectangular, with a small square on the left side of the image. The windows are mostly white with black trim. The windows are mostly rectangular, with a small square on the left side of the image. The windows are mostly white with black trim. The windows are mostly rectangular, with a small square on the left side of the image. The windows are mostly white with black trim. The windows are mostly rectangular, with a small square on the left side of the image. The windows are mostly white with black trim. The windows are mostly rectangular, with a small square on the left side of the image. The windows are mostly white with black trim. The windows are mostly rectangular, with a small square on the left side of the image. The windows are mostly white with black trim. The windows are mostly rectangular, with a small square on the left side of the image. The windows are mostly white with black trim. The windows are mostly rectangular, with a small square on the left side of the image. The windows are mostly white with black trim. The windows are mostly rectangular, with a small square on
```

---

### ✅ mlx-community/gemma-3-27b-it-qat-8bit

**Metrics:** 13.3 TPS | 387 tokens

```text
## Title: Historic Tudor Buildings Line Windhill Street in Bishop's Stortford

## Description:
This image depicts a row of well-preserved, black-and-white timber-framed buildings on Windhill in Bishop's Stortford, England. A tapas restaurant, "Sabrosa del Fuego," occupies one of the buildings, with signage and outdoor menus visible. The scene is captured on a cloudy winter afternoon, showcasing the town's historic architecture integrated with modern commercial use.

## Keywords:
Tudor architecture, timber-framed building, half-timbered building, Windhill, Bishop's Stortford, Hertfordshire, England, UK, United Kingdom, historic building, old building, street scene, townscape, European architecture, 17th century architecture, black and white architecture, facade, exterior, building exterior, restaurant, tapas restaurant, Sabrosa del Fuego, signage, outdoor seating, street lamp, pavement, road, traffic sign, brick chimney, tiled roof, moss, overcast sky, cloudy, winter, February, late afternoon, muted light, traditional architecture, quaint town, historic preservation, commercial building, urban scene, street photography, architectural detail, European town, British town, tourism, travel, landmark, high street, brick, white, black, brown, evening light, weathered, Grade II listed, narrow street, chimney, window, road sign, chalkboard sign, skeleton decoration, Santa hat, bare tree, wheelie bin, no people, architectural photography, cultural heritage, historic district, quaint, charm, character, enduring, integrated, commercial use, street level, perspective, architectural style, European lifestyle, British culture, local business, town center, street view, building detail, mossy roof, traditional storefront, quaint architecture, European street, British street, winter afternoon, muted tones, overcast weather, street ambiance, local scene.
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

### ✅ mlx-community/pixtral-12b-bf16

**Metrics:** 17.3 TPS | 500 tokens

```text
### Title
Historic Tudor-Style Buildings on Windhill Street, Bishop's Stortford

### Description
The image depicts a row of historic Tudor-style buildings on Windhill Street in Bishop's Stortford, Hertfordshire, England. The buildings feature distinctive black-and-white timber framing and moss-covered tile roofs, set against an overcast winter sky. One of the buildings houses the "Sabrosa del Fuego" tapas restaurant, with warm lights inviting passersby.

### Keywords
Tudor Architecture, Historic Building, Historic Pub Exterior, Half-Timbered Building, Half-Timbered, Historic, Black-and-White, Moss, Moss-Covered Roof, Tile Roof, Weathered Roof, Overcast Sky, Winter, Winter in England, Cloudy, Cloudy Sky, Historic Architecture, Historic Town, English Street Scene, English Pub, British, Europe, Europe, Hertfordshire, Bishop's Stortford, Windhill, United Kingdom, UK, England, England, Street Scene, Street Lamp, Street, Road, Road Sign, Traffic Sign, High Street, Pavement, Facade, Exterior, Building, Chimney, Brick Chimney, Brick, Brown, Black, White, Window, Windows, Skeleton, Skeleton Decoration, Skeleton in Window, Santa Hat, Quaint Town, Tourism, Townscape, Town, Traditional, Enduring Character, English Town, Contemporary Life, Tapas Restaurant, Warm Lights, Welcoming, Overcast, Chalkboard Sign, Objects, No People, Wheelie Bin, Bare Tree, European, Evening, Travel, Tourism, Historic, Historical, Architectural, Heritage, Market Town, Residence, Business, Mixed Use, Grade II Listed, 17th Century, 17th Century Building, Historic Preservation, Historic District, Historic Streetscape, Historic Buildings, Historic Architecture, Historic Structures, Historic Towns, Historic Townscapes, Historic Urban Landscape, Historic Urban Areas, Historic Urbanism, Historic Urban Environment, Historic Urban Planning, Historic Urban Design, Historic Urban Development, Historic Urban Infrastructure, Historic Urban Spaces, Historic Urban Places, Historic Urban Areas, Historic Urban Districts, Historic Urban Centers, Historic Urban Corridors, Historic Urban Networks, Historic Urban Grids, Historic Urban Layouts, Historic Urban Patterns, Historic Urban Forms, Historic Urban Structures, Historic Urban Spaces, Historic Urban Places, Historic Urban Areas,
```

---

### ✅ mlx-community/Phi-3.5-vision-instruct-bf16

**Metrics:** 11.4 TPS | 500 tokens

```text
Title: Historic Timber-Framed Buildings in Bishop's Stortford

Description: The image showcases a row of historic timber-framed buildings in Windhill, Bishop's Stortford, England, captured on a cool late afternoon in February. The buildings, with their distinctive black-and-white Tudor-style architecture, feature weathered, moss-covered tile roofs and are set against an overcast winter sky. The scene includes the "Sabrosa del Fuego" tapas restaurant, which adds a modern touch to the traditional setting.

Keywords: Historic, timber-framed, Tudor-style, architecture, black-and-white, moss-covered, tile roofs, overcast, winter, England, Hertfordshire, UK, Grade II listed, private residences, modern businesses, "Sabrosa del Fuego", tapas restaurant, traditional, everyday streetscape, English town, enduring character, contemporary life, historic buildings, architecture, everyday life, English pub, street scene, traditional, modern, historic, architecture, English, overcast, winter, street, town, townscape, traditional, modern, historic, architecture, everyday, English, overcast, winter, street, town, townscape, traditional, modern, historic, architecture, everyday, English, overcast, winter, street, town, townscape, traditional, modern, historic, architecture, everyday, English, overcast, winter, street, town, townscape, traditional, modern, historic, architecture, everyday, English, overcast, winter, street, town, townscape, traditional, modern, historic, architecture, everyday, English, overcast, winter, street, town, townscape, traditional, modern, historic, architecture, everyday, English, overcast, winter, street, town, townscape, traditional, modern, historic, architecture, everyday, English, overcast, winter, street, town, townscape, traditional, modern, historic, architecture, everyday, English, overcast, winter, street, town, townscape, traditional, modern, historic, architecture, everyday, English, overcast, winter, street, town, townscape, traditional, modern, historic, architecture, everyday, English, overcast, winter,
```

---

### ✅ microsoft/Phi-3.5-vision-instruct

**Metrics:** 11 TPS | 500 tokens

```text
Title: Historic Timber-Framed Buildings in Bishop's Stortford

Description: The image showcases a row of historic timber-framed buildings in Windhill, Bishop's Stortford, England, captured on a cool late afternoon in February. The buildings, with their distinctive black-and-white Tudor-style architecture, feature weathered, moss-covered tile roofs and are set against an overcast winter sky. The scene includes the "Sabrosa del Fuego" tapas restaurant, which adds a modern touch to the traditional setting.

Keywords: Historic, timber-framed, Tudor-style, architecture, black-and-white, moss-covered, tile roofs, overcast, winter, England, Hertfordshire, UK, Grade II listed, private residences, modern businesses, "Sabrosa del Fuego", tapas restaurant, traditional, everyday streetscape, English town, enduring character, contemporary life, historic buildings, architecture, everyday life, English pub, street scene, traditional, modern, historic, architecture, English, overcast, winter, street, town, townscape, traditional, modern, historic, architecture, everyday, English, overcast, winter, street, town, townscape, traditional, modern, historic, architecture, everyday, English, overcast, winter, street, town, townscape, traditional, modern, historic, architecture, everyday, English, overcast, winter, street, town, townscape, traditional, modern, historic, architecture, everyday, English, overcast, winter, street, town, townscape, traditional, modern, historic, architecture, everyday, English, overcast, winter, street, town, townscape, traditional, modern, historic, architecture, everyday, English, overcast, winter, street, town, townscape, traditional, modern, historic, architecture, everyday, English, overcast, winter, street, town, townscape, traditional, modern, historic, architecture, everyday, English, overcast, winter, street, town, townscape, traditional, modern, historic, architecture, everyday, English, overcast, winter, street, town, townscape, traditional, modern, historic, architecture, everyday, English, overcast, winter,
```

---

### ✅ Qwen/Qwen3-VL-2B-Instruct

**Metrics:** 71.7 TPS | 500 tokens

```text
Title: Skeleton in Window, Tudor Pub

Description: A historic Tudor-style pub in Bishop's Stortford, England, features a spooky skeleton decoration in a window, with a chalkboard menu and a festive atmosphere. The building's black-and-white half-timbered facade and moss-covered roof are set against an overcast sky, capturing a quiet winter scene.

Keywords: skeleton, skeleton in window, skeleton decoration, half-timbered building, Tudor Architecture, Tudor Style, historic building, black, white, brick, brick chimney, moss, overcast sky, winter, street scene, pub, restaurant, chalkboard sign, road sign, street lamp, window, building, pavement, road, town, townscape, European, England, United Kingdom, Hertfordshire, Bishop's Stortford, Windhill, Christmas, Halloween, spooky, festive, holiday, holiday atmosphere, winter holiday, winter season, cold, dark, quiet, cozy, cozy atmosphere, traditional, old building, architectural detail, architectural style, architectural heritage, architectural beauty, architectural charm, architectural character, architectural design, architectural history, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural preservation, architectural
```

---

### ✅ meta-llama/Llama-3.2-11B-Vision-Instruct

**Metrics:** 3.67 TPS | 213 tokens

```text
**Title:** Historic Timber-Framed Buildings in Windhill, England

**Description:** The image depicts a row of historic timber-framed buildings in Windhill, England, showcasing their unique architecture and character. The buildings are situated on a narrow street, with a mix of old and new structures, including a modern restaurant with a sign that reads "Sabrosa del Fuego" and offers tapas and cocktails.

**Keywords:**

* Specific subjects: buildings, street, road, streetlights, signs, trees, grass
* Setting and location: urban, street, sidewalk, pavement, road, grassy area
* Actions and activities: none visible
* Concepts and themes: history, architecture, culture, community, tradition
* Mood and emotion: calm, serene, peaceful
* Visual style: wide-angle, daytime, natural light
* Colors and lighting: natural, earthy tones, warm lighting
* Seasonal and temporal context: winter, late afternoon
* Use-case relevance: cultural, historical, architectural, travel, tourism
```

---

### ✅ mlx-community/Qwen3-VL-2B-Thinking-bf16

**Metrics:** 67.9 TPS | 500 tokens

```text
Got it, let's tackle this image cataloguing task. First, I need to follow the structure: Title, Description, Keywords, Context. Wait, the user said "Context" is the existing description, but the problem says "Context: The image is described as ...". Wait, no, the user provided a context section that's a description, but the task says "Context: The image is described as ...". Wait, no, looking back: the user says "Context: The image is described as ', Windhill, Bishop's Stortford, England, United Kingdom, UK...'" but then the actual context is the paragraph they provided. Wait, no, the user's instruction says: "Context: The image is described as ', Windhill, Bishop's Stortford, England, United Kingdom, UK...'" but then the paragraph is the context. Wait, no, the user's context is the paragraph they gave, which is the description of the image. Wait, no, the user says: "Context: The image is described as ', Windhill, Bishop's Stortford, England, United Kingdom, UK...'" but then the paragraph is the context. Wait, no, the user provided a context section that's a description of the image. Wait, no, the user's instruction says: "Context: The image is described as ', Windhill, Bishop's Stortford, England, United Kingdom, UK...'" but then the paragraph is the context. Wait, no, the user's context is the paragraph they wrote, which is the description. Wait, no, the user says: "Context: The image is described as ', Windhill, Bishop's Stortford, England, United Kingdom, UK...'" but then the paragraph is the context. Wait, no, the user's context is the paragraph they provided, which is the description. Wait, no, the user's instruction says: "Context: The image is described as ', Windhill, Bishop's Stortford, England, United Kingdom, UK...'" but then the paragraph is the context. Wait, no, the user's context is the paragraph they wrote, which is the description. Wait, no, the user says: "Context: The image is described as ', Windhill, Bishop's Stortford, England, United Kingdom, UK...'" but then the paragraph is the context. Wait, no, the user's context is the paragraph they provided
```

---

### ✅ mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

**Metrics:** 6.84 TPS | 500 tokens

```text
### Title
"Timber-Framed Buildings in a Small English Town"

### Description
The image depicts a row of historic timber-framed buildings in a small English town, specifically in the market town of Bishop's Stortford, Hertfordshire. The buildings, with their distinctive black-and-white Tudor-style architecture, are situated along a narrow street, with the "Sabrosa del Fuego" tapas restaurant on the right. The scene is set against an overcast winter sky, with the buildings' weathered, moss-covered tile roofs standing out against the gloomy backdrop.

### Keywords
Timber-framed buildings, historic architecture, English town, market town, Hertfordshire, Bishop's Stortford, Sabrosa del Fuego, tapas restaurant, overcast sky, winter, weathered tile roofs, moss-covered, black-and-white, Tudor-style, narrow street, small town, historic buildings, English architecture, market town, Hertfordshire, Bishop's Stortford, Sabrosa del Fuego, tapas restaurant, overcast sky, winter, weathered tile roofs, moss-covered, black-and-white, Tudor-style, narrow street, small town, historic buildings, English architecture, market town, Hertfordshire, Bishop's Stortford, Sabrosa del Fuego, tapas restaurant, overcast sky, winter, weathered tile roofs, moss-covered, black-and-white, Tudor-style, narrow street, small town, historic buildings, English architecture, market town, Hertfordshire, Bishop's Stortford, Sabrosa del Fuego, tapas restaurant, overcast sky, winter, weathered tile roofs, moss-covered, black-and-white, Tudor-style, narrow street, small town, historic buildings, English architecture, market town, Hertfordshire, Bishop's Stortford, Sabrosa del Fuego, tapas restaurant, overcast sky, winter, weathered tile roofs, moss-covered, black-and-white, Tudor-style, narrow street, small town, historic buildings, English architecture, market town, Hertfordshire, Bishop's Stortford, Sabrosa del Fuego, tapas restaurant, overcast sky, winter, weathered tile roofs, moss-covered, black-and-white, Tudor-style, narrow street, small town, historic buildings, English architecture, market town, Hertfordshire, Bishop's
```

---

### ✅ mlx-community/Qwen2-VL-2B-Instruct-4bit

**Metrics:** 156 TPS | 500 tokens

```text
Title: Windhill, Bishop's Stortford, England, UK
Description: A row of historic timber-framed buildings line the narrow street of Windhill in the market town of Bishop's Stortford, Hertfordshire, England. Photographed on a cool late afternoon in February, the scene is dominated by the distinctive black-and-white Tudor-style architecture, with its weathered, moss-covered tile roofs set against an overcast winter sky.
Keywords: Windhill, Bishop's Stortford, England, UK, Hertfordshire, Tudor Architecture, Winter in England, Historic pub exterior, Quaint town, Architecture, Street Scene, Winter, Grade II listed, Overcast sky, Urban setting, Historical buildings, Pedestrian street, Urban landscape, Winter atmosphere, Urban environment, Urban street, Urban street scene, Urban architecture, Urban setting, Urban landscape, Urban environment, Urban street, Urban street scene, Urban architecture, Urban setting, Urban landscape, Urban environment, Urban street, Urban street scene, Urban architecture, Urban setting, Urban landscape, Urban environment, Urban street, Urban street scene, Urban architecture, Urban setting, Urban landscape, Urban environment, Urban street, Urban street scene, Urban architecture, Urban setting, Urban landscape, Urban environment, Urban street, Urban street scene, Urban architecture, Urban setting, Urban landscape, Urban environment, Urban street, Urban street scene, Urban architecture, Urban setting, Urban landscape, Urban environment, Urban street, Urban street scene, Urban architecture, Urban setting, Urban landscape, Urban environment, Urban street, Urban street scene, Urban architecture, Urban setting, Urban landscape, Urban environment, Urban street, Urban street scene, Urban architecture, Urban setting, Urban landscape, Urban environment, Urban street, Urban street scene, Urban architecture, Urban setting, Urban landscape, Urban environment, Urban street, Urban street scene, Urban architecture, Urban setting, Urban landscape, Urban environment, Urban street, Urban street scene, Urban architecture, Urban setting, Urban landscape, Urban environment, Urban street, Urban street scene, Urban architecture, Urban setting, Urban landscape, Urban environment, Urban street, Urban street scene, Urban architecture, Urban setting, Urban landscape, Urban environment, Urban street, Urban street scene, Urban architecture, Urban setting, Urban landscape, Urban environment, Urban street, Urban street scene, Urban architecture, Urban setting, Urban landscape, Urban environment, Urban street, Urban street scene, Urban architecture, Urban setting, Urban
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
- `mlx`: `0.30.7.dev20260207+8fe1d092`
- `mlx-lm`: `0.30.6`
- `mlx-metal`: ``
- `mlx-vlm`: `0.3.11`
- `numpy`: `2.4.2`
- `tokenizers`: `0.22.2`
- `transformers`: `5.1.0`

_Report generated on: 2026-02-08 00:01:56 GMT_
