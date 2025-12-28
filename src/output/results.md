# Model Performance Results

_Generated on 2025-12-28 20:50:37 GMT_

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/LFM2-VL-1.6B-8bit` (312.1 tps)
- **💾 Most efficient:** `qnguyen3/nanoLLaVA` (4.5 GB)
- **⚡ Fastest load:** `HuggingFaceTB/SmolVLM-Instruct` (0.00s)
- **📊 Average TPS:** 63.3 across 26 models

## 📈 Resource Usage

- **Total peak memory:** 485.0 GB
- **Average peak memory:** 18.7 GB
- **Memory efficiency:** 174 tokens/GB

## ⚠️ Quality Issues

- **❌ Failed Models (10):**
  - `microsoft/Florence-2-large-ft` (`Model Error`)
  - `microsoft/Phi-3.5-vision-instruct` (`Error`)
  - `mlx-community/InternVL3-14B-8bit` (`Error`)
  - `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16` (`Lib Version`)
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit` (`Lib Version`)
  - `mlx-community/Phi-3.5-vision-instruct-bf16` (`Error`)
  - `mlx-community/Qwen2-VL-2B-Instruct-4bit` (`OOM`)
  - `mlx-community/deepseek-vl2-8bit` (`Error`)
  - `mlx-community/gemma-3-12b-pt-8bit` (`Error`)
  - `prince-canuma/Florence-2-large-ft` (`Error`)
- **📝 Formatting Issues (2):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/Idefics3-8B-Llama3-bf16`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 63.3 | Min: 3.8 | Max: 312
- **Peak Memory**: Avg: 19 | Min: 4.5 | Max: 53
- **Total Time**: Avg: 23.59s | Min: 2.52s | Max: 60.82s
- **Generation Time**: Avg: 20.04s | Min: 1.41s | Max: 55.09s
- **Model Load Time**: Avg: 3.55s | Min: 1.11s | Max: 7.39s

> **Prompt used:**
>
> Provide a factual caption, description, and keywords suitable for cataloguing, or searching for, the image.
>
> Context: The image relates to ', Beach, Gullane, Scotland, United Kingdom, UK
> On a quiet December afternoon in Gullane, Scotland, clusters of bright orange sea buckthorn berries provide a vibrant splash of colour against the muted tones of the coastal landscape. From a grassy clifftop, the view extends over the sandy beach where two people are taking a stroll along the shoreline of the Firth of Forth.
>
> This area of East Lothian is renowned for its sweeping beaches and world-class golf courses. The coastline also holds historical significance, with many of its beaches, including Gullane, having been fortified with anti-tank defences during World War II, remnants of which can still be seen today.'
>
> The photo was taken around 2025-12-28 14:59:01 GMT from GPS 56.038550°N, 2.843717°W . Focus on visual content, drawing on any available contextual information for specificity. Do not speculate.

**Note:** Results sorted: errors first, then by generation time (fastest to slowest).

**Overall runtime:** 652.11s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Token |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues       |
|:--------------------------------------------------------|--------:|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:---------------------|
| `microsoft/Florence-2-large-ft`                         |         |                   |                       |                |              |           |             |                  |            |             |                      |
| `microsoft/Phi-3.5-vision-instruct`                     |         |                   |                       |                |              |           |             |                  |            |             |                      |
| `mlx-community/InternVL3-14B-8bit`                      |         |                   |                       |                |              |           |             |                  |            |             |                      |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |         |                   |                       |                |              |           |             |                  |            |             |                      |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |         |                   |                       |                |              |           |             |                  |            |             |                      |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |         |                   |                       |                |              |           |             |                  |            |             |                      |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |         |                   |                       |                |              |           |             |                  |            |             |                      |
| `mlx-community/deepseek-vl2-8bit`                       |         |                   |                       |                |              |           |             |                  |            |             |                      |
| `mlx-community/gemma-3-12b-pt-8bit`                     |         |                   |                       |                |              |           |             |                  |            |             |                      |
| `prince-canuma/Florence-2-large-ft`                     |         |                   |                       |                |              |           |             |                  |            |             |                      |
| `qnguyen3/nanoLLaVA`                                    | 151,645 |               260 |                    65 |            325 |        1,118 |       101 |         4.5 |            1.41s |      1.11s |       2.52s |                      |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |  49,279 |             1,337 |                    47 |          1,384 |        1,160 |       118 |         5.5 |            2.01s |      1.42s |       3.43s |                      |
| `HuggingFaceTB/SmolVLM-Instruct`                        |  49,154 |             1,437 |                    19 |          1,456 |          991 |       118 |         5.5 |            2.25s |      1.48s |       3.73s |                      |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |       1 |             4,335 |                    22 |          4,357 |        1,294 |       110 |         8.6 |            3.97s |      2.93s |       6.90s | context-ignored      |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |       1 |             1,263 |                    87 |          1,350 |          479 |      41.3 |          11 |            5.16s |      3.84s |       8.99s | context-ignored      |
| `mlx-community/SmolVLM-Instruct-bf16`                   |  49,154 |             1,437 |                   409 |          1,846 |        1,197 |       113 |         5.5 |            5.29s |      1.39s |       6.68s |                      |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |       7 |             2,036 |                    15 |          2,051 |          434 |       312 |          53 |            5.36s |      1.33s |       6.68s |                      |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |       2 |             2,910 |                   305 |          3,215 |          858 |       156 |         7.4 |            5.75s |      2.01s |       7.77s |                      |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |     106 |               508 |                   230 |            738 |          575 |      41.6 |          17 |            6.85s |      5.28s |      12.14s |                      |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |       1 |             1,263 |                   128 |          1,391 |        1,457 |      17.6 |          11 |            8.57s |      4.98s |      13.54s | context-ignored      |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |       2 |             2,447 |                     8 |          2,455 |          259 |      47.6 |          12 |           10.05s |      1.93s |      11.98s | context-ignored      |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |     106 |               508 |                   305 |            813 |          174 |      27.9 |          19 |           14.28s |      5.18s |      19.46s | bullets(21)          |
| `mlx-community/pixtral-12b-8bit`                        |       2 |             3,071 |                   281 |          3,352 |          495 |      34.3 |          15 |           14.80s |      3.48s |      18.28s | bullets(19)          |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | 128,001 |             2,549 |                   364 |          2,913 |          665 |      29.0 |          18 |           16.84s |      3.72s |      20.57s | formatting           |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |       2 |             2,345 |                   194 |          2,539 |          213 |      27.7 |          22 |           18.44s |      4.19s |      22.63s |                      |
| `mlx-community/pixtral-12b-bf16`                        |       2 |             3,071 |                   277 |          3,348 |          531 |      19.2 |          27 |           20.59s |      5.23s |      25.82s | bullets(20)          |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |   1,056 |             3,169 |                   500 |          3,669 |          397 |      35.3 |          15 |           22.56s |      3.31s |      25.87s |                      |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |     106 |               508 |                   360 |            868 |          166 |      15.4 |          34 |           26.88s |      7.39s |      34.26s | bullets(25)          |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      | 128,009 |               237 |                   222 |            459 |         74.9 |      8.61 |          15 |           29.38s |      2.95s |      32.33s | context-ignored      |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |       1 |             1,263 |                   128 |          1,391 |          448 |      4.84 |          27 |           29.70s |      7.27s |      36.97s | context-ignored      |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |  66,452 |             6,382 |                   500 |          6,882 |          304 |      47.3 |          13 |           32.03s |      2.76s |      34.79s | formatting           |
| `mlx-community/Molmo-7B-D-0924-8bit`                    | 151,643 |             1,452 |                   187 |          1,639 |         43.1 |      44.6 |          41 |           38.47s |      5.96s |      44.43s |                      |
| `mlx-community/Molmo-7B-D-0924-bf16`                    | 151,643 |             1,452 |                   289 |          1,741 |         43.5 |      27.9 |          47 |           44.25s |      3.71s |      47.97s |                      |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |      11 |            16,479 |                   500 |         16,979 |          386 |      71.8 |          12 |           50.23s |      1.85s |      52.08s |                      |
| `Qwen/Qwen3-VL-2B-Instruct`                             |  22,012 |            16,477 |                   500 |         16,977 |          382 |      70.3 |          12 |           50.92s |      1.81s |      52.73s | verbose, bullets(54) |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | 128,009 |               238 |                   189 |            427 |         50.7 |       3.8 |          25 |           55.09s |      5.73s |      60.82s |                      |

<!-- markdownlint-enable MD013 MD033 MD034 MD037 MD049 -->

## Model Gallery

Full output from each model:

<!-- markdownlint-disable MD013 -->

### ❌ microsoft/Florence-2-large-ft

**Status:** Failed (Model Error)
**Error:** Model loading failed: Missing 1 parameters:
language_model.lm_head.weight.
**Type:** `ValueError`

---

### ❌ microsoft/Phi-3.5-vision-instruct

**Status:** Failed (Error)
**Error:**

> Model generation failed for microsoft/Phi-3.5-vision-instruct: Failed to
> process inputs with error: Phi3VProcessor.__call__() got an unexpected
> keyword argument 'padding_side'
**Type:** `ValueError`

---

### ❌ mlx-community/InternVL3-14B-8bit

**Status:** Failed (Error)
**Error:**

> Model loading failed: Received a InternVLImageProcessor for argument
> image_processor, but a ImageProcessingMixin was expected.
**Type:** `ValueError`

---

### ❌ mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

**Status:** Failed (Lib Version)
**Error:**

> Model loading failed: cannot import name '_validate_images_text_input_order'
> from 'transformers.processing_utils'
> (/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/processing_utils.py)
**Type:** `ValueError`

---

### ❌ mlx-community/Kimi-VL-A3B-Thinking-8bit

**Status:** Failed (Lib Version)
**Error:**

> Model loading failed: cannot import name '_validate_images_text_input_order'
> from 'transformers.processing_utils'
> (/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/processing_utils.py)
**Type:** `ValueError`

---

### ❌ mlx-community/Phi-3.5-vision-instruct-bf16

**Status:** Failed (Error)
**Error:**

> Model loading failed:
> /Users/jrp/.cache/huggingface/hub/models--mlx-community--Phi-3.5-vision-instruct-bf16/snapshots/d8da684308c275a86659e2b36a9189b2f4aec8ea
> does not appear to have a file named image_processing_phi3_v.py. Checkout
> 'https://huggingface.co//Users/jrp/.cache/huggingface/hub/models--mlx-community--Phi-3.5-vision-instruct-bf16/snapshots/d8da684308c275a86659e2b36a9189b2f4aec8ea/tree/main'
> for available files.
**Type:** `ValueError`

---

### ❌ mlx-community/Qwen2-VL-2B-Instruct-4bit

**Status:** Failed (OOM)
**Error:**

> Model runtime error during generation for
> mlx-community/Qwen2-VL-2B-Instruct-4bit: [metal::malloc] Attempting to
> allocate 134767706112 bytes which is greater than the maximum allowed buffer
> size of 86586540032 bytes.
**Type:** `ValueError`

---

### ❌ mlx-community/deepseek-vl2-8bit

**Status:** Failed (Error)
**Error:**

> Model runtime error during generation for mlx-community/deepseek-vl2-8bit:
> std::bad_cast
**Type:** `ValueError`

---

### ❌ mlx-community/gemma-3-12b-pt-8bit

**Status:** Failed (Error)
**Error:**

> Cannot use chat template functions because tokenizer.chat_template is not
> set and no template argument was passed! For information about writing
> templates and setting the tokenizer.chat_template attribute, please see the
> documentation at
> https://huggingface.co/docs/transformers/main/en/chat_templating
**Type:** `ValueError`

---

### ❌ prince-canuma/Florence-2-large-ft

**Status:** Failed (Error)
**Error:**

> Model loading failed: RobertaTokenizer has no attribute
> additional_special_tokens
**Type:** `ValueError`

---

### ✅ qnguyen3/nanoLLaVA

**Metrics:** 101 TPS | 65 tokens

```text
A picturesque coastal scene captures the essence of Gullane, Scotland, with its sandy beach, rocky cliffs, and beachgoers enjoying the day. The image is a blend of natural beauty and human activity, with the bright orange sea buckthorn berries adding a splash of color against the muted tones of the landscape.
```

---

### ✅ mlx-community/SmolVLM2-2.2B-Instruct-mlx

**Metrics:** 118 TPS | 47 tokens

```text
 Beach, Gullane, Scotland, United Kingdom, UK, December, 2025, GPS 56.038550°N, 2.843717°W
```

---

### ✅ HuggingFaceTB/SmolVLM-Instruct

**Metrics:** 118 TPS | 19 tokens

```text
 Bright orange sea buckthorn berries on a coastal clifftop in Gullane, Scotland.
```

---

### ✅ mlx-community/paligemma2-3b-pt-896-4bit

**Metrics:** 110 TPS | 22 tokens

```text
Provide a factual caption, description, and keywords suitable for cataloguing, or searching for, the image.
```

---

### ✅ mlx-community/paligemma2-10b-ft-docci-448-6bit

**Metrics:** 41.3 TPS | 87 tokens

```text
A long-shot view of a beach with two people walking along the shore. The beach is a light brown sand with a dark brown rock formation in the top left corner. The water is a light blue with white foam at the shoreline. The sky is a light blue with no visible cloud coverage. In the foreground is a cluster of sea buckthorn bushes with orange berries. The bushes are growing on a hill with long grass.
```

---

### ✅ mlx-community/SmolVLM-Instruct-bf16

**Metrics:** 113 TPS | 409 tokens

```text
 The image depicts a coastal scene in Gullane, Scotland, captured on December 28, 2025. The focal point of the image is a cluster of bright orange sea buckthorn berries, which are a distinctive and vibrant addition to the otherwise muted coastal landscape. The berries are clustered together on the branches of a shrub or small tree, creating a striking contrast against the muted tones of the surrounding environment.

The sea buckthorn berries are a bright orange color, which stands out against the muted tones of the coastal landscape. The berries are small and round, and they are clustered together in groups. The clusters are dense and appear to be quite large, suggesting that the shrub or tree is quite mature. The berries are also quite small, which helps to create a sense of depth and texture in the image.

The background of the image is dominated by the beach and the Firth of Forth. The beach is sandy and appears to be quite wide, with a few people walking along it. The Firth of Forth is a large body of water that stretches out to the horizon, and it is visible in the distance behind the beach. The water is a light blue color, and it appears to be relatively calm.

The image is taken from a slightly elevated perspective, which gives the viewer a clear view of the beach and the Firth of Forth. The sky is clear, with a few clouds visible in the distance. The overall atmosphere of the image is peaceful and serene, with the bright orange berries providing a focal point that draws the viewer's attention to the scene.

The image is a good example of how natural elements can create a striking and visually appealing scene. The bright orange sea buckthorn berries are a unique and eye-catching feature, and they help to draw the viewer's attention to the natural beauty of the coastal landscape. The image is also a good example of how a single element can create a sense of depth and texture in a scene.
```

---

### ✅ mlx-community/LFM2-VL-1.6B-8bit

**Metrics:** 312 TPS | 15 tokens

```text
Gullane Beach, Scotland, December 28, 2025
```

---

### ✅ mlx-community/Ministral-3-3B-Instruct-2512-4bit

**Metrics:** 156 TPS | 305 tokens

```text
**Caption:**
A serene coastal scene at Gullane Beach, East Lothian, Scotland. Vibrant clusters of orange sea buckthorn berries contrast beautifully with the tranquil shoreline, framing a peaceful walk along the sandy beach by the Firth of Forth.

---

**Description:**
The image captures a coastal landscape in Gullane, Scotland, featuring a grassy clifftop with dense patches of bright orange sea buckthorn berries. The foreground is dominated by these vivid berries, adding a striking splash of color to the scene. In the background, a sandy beach stretches out toward the water, where two people are leisurely strolling along the shoreline. The coastline is calm, with gentle waves rolling onto the beach. The muted tones of the coastal environment are softened by the natural light, creating a peaceful and picturesque atmosphere.

---

**Keywords:**
- Gullane Beach, Scotland
- East Lothian coastline
- Sea buckthorn berries
- Sandy beach, Firth of Forth
- Coastal walk, peaceful scenery
- Natural landscape, vibrant flora
- World-class golf courses
- WWII anti-tank defences
- December afternoon, coastal view
- Shoreline, clifftop vegetation
- Historical coastal fortifications
- Gullane, United Kingdom, UK
- Coastal erosion, serene ambiance
- Wildlife-friendly coastal area

---
This cataloguing ensures the image can be accurately referenced for research, tourism, or documentation purposes.
```

---

### ✅ mlx-community/gemma-3n-E4B-it-bf16

**Metrics:** 41.6 TPS | 230 tokens

```text
## Caption:

A vibrant burst of orange sea buckthorn berries contrasts with the muted tones of the coastal landscape in Gullane, Scotland. Two figures stroll along the sandy beach of the Firth of Forth on a cool December day.

## Description:

This photograph captures a scenic view of Gullane beach in East Lothian, Scotland. The foreground is dominated by low-lying bushes laden with bright orange sea buckthorn berries, adding a striking splash of color to the scene. Beyond the berries, a wide expanse of sandy beach stretches towards the calm, grey-blue waters of the Firth of Forth. Two small figures can be seen walking along the shoreline, enjoying the coastal path. The sky is overcast, contributing to the cool, slightly melancholic atmosphere. The coastline features rocky outcrops and evidence of historical defenses from World War II, highlighting the area's rich history.

## Keywords:

Gullane, Scotland, UK, beach, sea buckthorn, berries, coastline, Firth of Forth, East Lothian, coastal walk, winter, December, landscape, nature, wildlife, historical, cliffs, overcast, tranquil, scenic.
```

---

### ✅ mlx-community/paligemma2-3b-ft-docci-448-bf16

**Metrics:** 17.6 TPS | 128 tokens

```text
A long-shot view of a beach with a grassy hill in the foreground. The foreground is filled with a mass of dried, dead grass with orange berries growing on the branches. The background is a view of the beach, with a small group of people walking along the shore in the distance. The beach is a sandy, tan color, with a small area of black rock visible on the left side of the image. The water is a light blue-green color, with a small wave breaking on the shore. The sky is a light blue color, with a slight view of the horizon in the top right corner of the image.
```

---

### ✅ mlx-community/llava-v1.6-mistral-7b-8bit

**Metrics:** 47.6 TPS | 8 tokens

```text
The image is a photograph.
```

---

### ✅ mlx-community/gemma-3-27b-it-qat-4bit

**Metrics:** 27.9 TPS | 305 tokens

```text
Here's a factual caption, description, and keyword list for the image, based on your provided context:

**Caption:** Sea buckthorn berries add colour to a December coastal scene at Gullane Beach, Scotland.

**Description:** A vertical photograph showing a view of Gullane Beach, East Lothian, Scotland, taken from a grassy clifftop. In the foreground, branches laden with bright orange sea buckthorn berries dominate the lower portion of the frame. Behind the berries, a sandy beach stretches along the shoreline. Two figures are visible walking along the beach. The Firth of Forth is visible in the background under a pale blue sky. Sparse vegetation and dry grasses are visible on the clifftop.

**Keywords:**

*   Gullane
*   Beach
*   Scotland
*   East Lothian
*   Firth of Forth
*   Sea Buckthorn
*   Berries
*   Coastal Landscape
*   December
*   Winter
*   UK
*   United Kingdom
*   Shoreline
*   People
*   Clifftop
*   Vegetation
*   56.038550°N, 2.843717°W
*   2025-12-28
*   Coastal Defence (potential, based on context)
*   World War II (potential, based on context)
*   Golf Courses (potential, based on context)




```

---

### ✅ mlx-community/pixtral-12b-8bit

**Metrics:** 34.3 TPS | 281 tokens

```text
### Caption
A serene December afternoon at Gullane Beach, Scotland, showcases vibrant orange sea buckthorn berries against the muted coastal landscape. Two individuals stroll along the sandy shore of the Firth of Forth, with remnants of World War II anti-tank defenses visible in the distance.

### Description
The image captures a tranquil scene at Gullane Beach in East Lothian, Scotland. From a grassy clifftop, the viewer's gaze is drawn to the sandy beach below, where two people are walking along the shoreline. The coastline is adorned with clusters of bright orange sea buckthorn berries, providing a striking contrast to the otherwise muted colors of the landscape. The calm waters of the Firth of Forth stretch out to the horizon, and remnants of anti-tank defenses from World War II can be seen along the beach, adding a historical dimension to the scene.

### Keywords
- Gullane Beach
- Scotland
- Firth of Forth
- Sea Buckthorn
- Coastal Landscape
- December
- Anti-tank Defenses
- East Lothian
- World War II
- Shoreline
- Sandy Beach
- Clifftop
- Historical Significance
- Tranquil Scene
- Muted Colors
- Orange Berries
- Walking
- Remnants
- Historical Dimension
```

---

### ✅ mlx-community/Idefics3-8B-Llama3-bf16

**Metrics:** 29 TPS | 364 tokens

```text
 The image depicts a serene coastal scene in Gullane, Scotland, captured on December 28, 2025, at 14:59:01 GMT. The photograph was taken from a grassy clifftop, offering a panoramic view of the sandy beach below. The focal point of the image is the vibrant orange sea buckthorn berries that are scattered across the landscape. These berries add a striking contrast to the otherwise muted tones of the coastal environment.

In the foreground, the grassy terrain is interspersed with clusters of these bright orange berries, creating a visually appealing mosaic. The berries are part of the sea buckthorn plant, known scientifically as Hippophae rhamnoides, which is native to the coastal regions of Europe and Asia. The plant is valued for its medicinal properties and is often used in traditional medicine.

The sandy beach below is bordered by the Firth of Forth, a large estuary that flows into the North Sea. The water appears calm, reflecting the clear sky above. The shoreline is dotted with a few people, who are likely enjoying the peaceful atmosphere of the beach. The presence of these individuals adds a sense of scale to the expansive landscape.

In the background, the coastline stretches out, with more sandy beaches and possibly some rocky formations. The historical significance of the area is evident, as remnants of anti-tank defences from World War II can still be seen along the coast. These defences were part of the extensive fortifications built to protect the coastline from potential invasions.

The image captures the essence of a tranquil coastal setting, with the vibrant orange sea buckthorn berries providing a focal point of colour against the natural beauty of the Scottish landscape. The photograph serves as a testament to the region's natural beauty and historical significance.<end_of_utterance>
```

---

### ✅ mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

**Metrics:** 27.7 TPS | 194 tokens

```text
Caption: "Vibrant Sea Buckthorn Berries on Gullane Beach, Scotland"

Description: The image captures a serene coastal scene in Gullane, Scotland, on a quiet December afternoon. In the foreground, clusters of bright orange sea buckthorn berries stand out against the muted tones of the grassy clifftop. The view extends over a sandy beach where two people are seen walking along the shoreline of the Firth of Forth. The calm sea and the expansive beach create a peaceful atmosphere, highlighting the natural beauty of this East Lothian coastline.

Keywords: Gullane, Scotland, beach, sea buckthorn berries, coastal landscape, Firth of Forth, East Lothian, sandy beach, shoreline, nature, winter, December, United Kingdom, UK, coastal walk, tranquil, scenic, coastal scenery, sea, grassy clifftop, vibrant colors, natural beauty.
```

---

### ✅ mlx-community/pixtral-12b-bf16

**Metrics:** 19.2 TPS | 277 tokens

```text
### Caption
A serene December afternoon in Gullane, Scotland, showcases vibrant orange sea buckthorn berries against the muted coastal landscape. From a grassy clifftop, the view extends over a sandy beach where two people stroll along the shoreline of the Firth of Forth.

### Description
The image captures a tranquil scene at a beach in Gullane, Scotland. In the foreground, clusters of bright orange sea buckthorn berries stand out vividly against the muted greens and browns of the coastal vegetation. The vantage point is from a grassy clifftop, overlooking a sandy beach where two individuals are seen walking along the shoreline. The expansive view extends to the horizon, where the calm waters of the Firth of Forth meet the sky. The scene is peaceful, highlighting the natural beauty of the area.

### Keywords
- Gullane, Scotland
- Beach
- Sea buckthorn berries
- Firth of Forth
- Coastal landscape
- December
- Clifftop
- Sandy beach
- Shoreline
- East Lothian
- Anti-tank defences
- World War II
- Golf courses
- Historical significance
- Tranquil scene
- Natural beauty
- Walking
- Horizon
- Calm waters
- Sky
```

---

### ✅ mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX

**Metrics:** 35.3 TPS | 500 tokens

```text
Here are my reasoning steps:
We need to produce a factual caption, description, and keywords suitable for cataloguing or searching the image. The user gave context: "Beach, Gullane, Scotland, United Kingdom, UK. On a quiet December afternoon in Gullane, Scotland, clusters of bright orange sea buckthorn berries provide a vibrant splash of colour against the muted tones of the coastal landscape. From a grassy clifftop, the view extends over the sandy beach where two people are taking a stroll along the shoreline of the Firth of Forth. This area of East Lothian is renowned for its sweeping beaches and world-class golf courses. The coastline also holds historical significance, with many of its beaches, including Gullane, having been fortified with anti-tank defences during World War II, remnants of which can still be seen today."

The photo taken around 2025-12-28 14:59:01 GMT from GPS 56.038550°N, 2.843717°W. Focus on visual content, drawing on any available contextual information for specificity. Do not speculate.

Thus we need to produce a factual caption, description, and keywords. The caption should be concise, describing the main visual elements: sea buckthorn berries, orange clusters, grassy clifftop, view over sandy beach, two people walking, Firth of Forth, overcast sky, muted tones, etc. The description should be more detailed, maybe a paragraph. Keywords should be relevant for cataloguing/search: "Gullane Beach", "East Lothian", "Scotland", "United Kingdom", "sea buckthorn", "orange berries", "coastal landscape", "sandy beach", "Firth of Forth", "clifftop", "two people walking", "December", "overcast sky", "anti-tank defences", "World War II remnants", "photography", "nature", "coastline", "grass", "shrubbery", "wildlife", "sea buckthorn (Hippophae rhamnoides)", "Golf courses", "tourist destination", "photographed 2025-12-28", "GPS 56.038550N 2.8
```

---

### ✅ mlx-community/gemma-3-27b-it-qat-8bit

**Metrics:** 15.4 TPS | 360 tokens

```text
Here's a factual caption, description, and keyword list for the image, based on the provided context and visual content:

**Caption:** Sea buckthorn berries add colour to the Gullane coastline, East Lothian, Scotland, December 2025.

**Description:** A vertical photograph taken on December 28, 2025, at 14:59:01 GMT (56.038550°N, 2.843717°W) shows a view of Gullane beach, Scotland. In the foreground, clusters of bright orange sea buckthorn berries are prominent on branches of shrubs growing on a grassy clifftop. The sandy beach extends into the distance, with two figures walking along the shoreline. The Firth of Forth is visible as the body of water, and the sky is a muted blue. Sparse vegetation and remnants of possible anti-tank defences are visible on the beach.

**Keywords:**

*   Gullane
*   Scotland
*   Beach
*   East Lothian
*   Firth of Forth
*   Sea Buckthorn
*   Berries
*   Coastline
*   Winter
*   December
*   UK
*   United Kingdom
*   Coastal Landscape
*   Clifftop
*   Shoreline
*   People
*   Anti-tank defences (possible)
*   Vegetation
*   2025
*   56.038550°N, 2.843717°W
*   Outdoor
*   Landscape Photography
*   Natural Environment
*   Grassy Bank
*   Coastal Defence (possible)




```

---

### ✅ mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

**Metrics:** 8.61 TPS | 222 tokens

```text
The image depicts a serene beach scene with a focus on the vibrant orange sea buckthorn bushes in the foreground. The image is set in a natural setting, with the beach and ocean visible in the background.

* The image features a beach with a calm and peaceful atmosphere.
	+ The beach is sandy and stretches along the coastline.
	+ The ocean is calm and blue, with gentle waves crashing against the shore.
* The foreground is dominated by the orange sea buckthorn bushes.
	+ The bushes are bright orange and have a distinctive shape.
	+ They are scattered throughout the foreground, adding a pop of color to the image.
* The background of the image is the ocean and the sky.
	+ The ocean is calm and peaceful, with a few waves visible.
	+ The sky is a light blue color, with a few clouds visible.

Overall, the image presents a peaceful and serene beach scene, with a focus on the natural beauty of the sea buckthorn bushes. The image does not show any significant changes or comparisons, but rather presents a static view of the scene.
```

---

### ✅ mlx-community/paligemma2-10b-ft-docci-448-bf16

**Metrics:** 4.84 TPS | 128 tokens

```text
A long-shot view of a beach with two people walking along the shore. The beach is a light brown sand with a dark brown rock formation on the left side. The water is a light blue with white foam on the shore. The sky is a light blue with no cloud coverage. In the foreground is a large bush with orange berries. The bush is casting a shadow over the beach. The bush is in the foreground, with the beach in the background. The beach is in the middle of the image. The water is in the top right corner of the image. The sky is in the top left corner of the image.
```

---

### ✅ mlx-community/GLM-4.6V-Flash-6bit

**Metrics:** 47.3 TPS | 500 tokens

```text
<think>Got it, let's tackle this. The user wants a factual caption, description, and keywords for cataloguing. First, I need to extract key elements from the context.

First, the location: Beach, Gullane, Scotland, United Kingdom. Then the time: December afternoon, 2025-12-28 14:59:01 GMT. The main visual elements: sea buckthorn berries (bright orange), sandy beach, two people strolling, Firth of Forth, coastal landscape with grassy clifftop, historical context about WWII anti-tank defences (though the image might not show them, but the context mentions they're present).

Now, structure the caption, description, and keywords.

Caption should be concise, include location, date/time, and key visual elements.

Description should elaborate on the scene, elements like sea buckthorn, beach, people, water body (Firth of Forth), coastal vegetation, and any relevant context (though keep factual, no speculation).

Keywords should be specific terms that would help in searching or cataloguing. Include location terms, subject matter (sea buckthorn, beach, coastal landscape), time (December, afternoon), geographical features (Firth of Forth, sandy beach, grassy clifftop), and any relevant historical or environmental terms.

Now, let's draft each part.

Caption: "Beach scene at Gullane, Scotland, on a December afternoon, featuring bright orange sea buckthorn berries in the foreground, a sandy beach with two people strolling, and the Firth of Forth in the distance."

Description: "The image captures a coastal landscape in Gullane, Scotland, during a quiet December afternoon. In the foreground, dense clusters of sea buckthorn (Hippophae rhamnoides) with vibrant orange berries stand out against the muted tones of the coastal vegetation. The middle ground reveals a sandy beach stretching toward the shoreline, where two small figures are visible strolling along the water’s edge. Beyond the beach, the calm waters of the Firth of Forth extend to the horizon under a pale blue sky. The scene is framed by a grassy clifftop in the foreground, with sparse, leafless shrubs interspersed with patches of green grass. The coastal landscape also includes rocky outcrops extending into the water near the beach, adding texture to the shoreline. The overall atmosphere is serene and tranquil
```

---

### ✅ mlx-community/Molmo-7B-D-0924-8bit

**Metrics:** 44.6 TPS | 187 tokens

```text
 'Beach, Gullane, Scotland, United Kingdom, UK
Sea buckthorn berries, December, Coastal landscape, Firth of Forth, Gullane beach, Anti-tank defences, World War II, East Lothian, Golf courses, Historical significance

On a quiet December afternoon in Gullane, Scotland, clusters of bright orange sea buckthorn berries provide a vibrant splash of colour against the muted tones of the coastal landscape. From a grassy clifftop, the view extends over the sandy beach where two people are taking a stroll along the shoreline of the Firth of Forth.

This area of East Lothian is renowned for its sweeping beaches and world-class golf courses. The coastline also holds historical significance, with many of its beaches, including Gullane, having been fortified with anti-tank defences during World War II, remnants of which can still be seen today.'
```

---

### ✅ mlx-community/Molmo-7B-D-0924-bf16

**Metrics:** 27.9 TPS | 289 tokens

```text
 Here's a factual caption, description, and keywords suitable for cataloguing or searching for the image:

Caption:
'A serene December scene at Gullane Beach, East Lothian, Scotland. Two figures walk along the shoreline, surrounded by the muted tones of the coastal landscape. Bright orange sea buckthorn berries provide a striking contrast against the grey sky and sandy beach. The image captures the essence of winter along the Firth of Forth, showcasing the rugged beauty of this part of the Scottish coastline.'

Description:
'This photograph, taken on December 28, 2025, at 14:59:01 GMT, from coordinates 56.038550°N, 2.843717°W, depicts a quiet afternoon at Gullane Beach. The scene features two people strolling along the shoreline, with the sea and sky dominating the upper portion of the image. The foreground is dotted with vibrant orange sea buckthorn berries, adding a pop of color to the otherwise muted winter landscape.'

Keywords:
Gullane Beach, Scotland, United Kingdom, UK, December, Sea Buckthorn, Coastal Landscape, Firth of Forth, East Lothian, World War II, Anti-Tank Defences, Beaches, Sandy Beach, Winter Scene, Serene Afternoon, Coastal Photography, Nature Photography, Scottish Coastline
```

---

### ✅ mlx-community/Qwen3-VL-2B-Thinking-bf16

**Metrics:** 71.8 TPS | 500 tokens

```text
Got it, let's break this down. First, the task is to provide a factual caption, description, and keywords for the image. The context given is about Gullane, Scotland, a quiet December afternoon, sea buckthorn berries, beach, Firth of Forth, etc.

First, the caption should be factual, based on the context. Let's check the context: "On a quiet December afternoon in Gullane, Scotland, clusters of bright orange sea buckthorn berries provide a vibrant splash of colour against the muted tones of the coastal landscape. From a grassy clifftop, the view extends over the sandy beach where two people are taking a stroll along the shoreline of the Firth of Forth."

So the caption needs to include key elements: location (Gullane, Scotland, UK), time (December afternoon), main subject (sea buckthorn berries, beach, two people walking), and the Firth of Forth.

Then the description: should be a detailed but factual account. Mention the setting: coastal landscape, sandy beach, grassy clifftop, sea buckthorn berries (bright orange), two people walking, Firth of Forth, December afternoon, muted tones of the landscape contrasting with the berries.

Keywords: need to be specific for cataloging/searching. Let's list them: Gullane, Scotland, UK, December, beach, sea buckthorn berries, Firth of Forth, coastal landscape, grassy clifftop, two people walking, sandy beach, winter, coastal defense remnants (but wait, the context says "remnants of which can still be seen today" but the main focus is the berries and the beach. Wait, the context mentions historical significance but the image is about the current scene. So maybe include "coastal defense remnants" as a keyword? Wait, the context says "the coastline also holds historical significance, with many of its beaches, including Gullane, having been fortified with anti-tank defences during World War II, remnants of which can still be seen today." But the image is the main focus, so maybe include that as a keyword? Wait, the task says "suitable for cataloguing, or searching for, the image." So keywords should be relevant to the image content, not just the context. Wait, the context is given as a description, so the keywords should be based on the image's content.

Wait,
```

---

### ✅ Qwen/Qwen3-VL-2B-Instruct

**Metrics:** 70.3 TPS | 500 tokens

```text
### Factual Caption
A vibrant cluster of orange sea buckthorn berries in the foreground of a coastal scene at Gullane, Scotland, on a quiet December afternoon. The view extends to a sandy beach where two people are strolling along the shoreline of the Firth of Forth, with the calm sea and distant cliffs under a pale blue sky.

### Description
The image captures a serene coastal landscape in Gullane, East Lothian, Scotland. In the foreground, dense clusters of bright orange sea buckthorn berries stand out against the muted tones of dry grass and bare branches, creating a striking contrast. The scene is viewed from a grassy clifftop, offering a panoramic view of the sandy beach below. Two people are seen walking along the shoreline, adding a sense of scale and human presence to the tranquil setting. The Firth of Forth stretches out to the horizon, with gentle waves lapping against the shore. The sky is overcast, casting a soft, diffused light over the entire scene, which enhances the natural colors and textures of the landscape. The coastline is marked by rocky outcrops and the remnants of historical anti-tank defenses, adding a layer of historical context to the picturesque view.

### Keywords
- Gullane
- Scotland
- East Lothian
- Firth of Forth
- Sea buckthorn berries
- Coastal landscape
- Sandy beach
- People walking
- Overcast sky
- Natural scenery
- Historical defenses
- December
- Coastal photography
- UK
- Sea
- Beach
- Orange berries
- Clifftop
- Nature
- Wildlife
- Landscape photography
- Coastal view
- Sea buckthorn
- Scotland, UK
- Gullane, Scotland
- Firth of Forth
- Orange berries
- Coastal scene
- People
- Sandy beach
- Sea buckthorn
- Natural scenery
- Overcast sky
- Coastal photography
- Sea buckthorn berries
- Gullane, Scotland
- East Lothian
- Firth of Forth
- Sea
- Beach
- People walking
- Natural scenery
- Historical defenses
- December
- Coastal view
- Sea buckthorn
- Scotland, UK
- Gullane, Scotland
- Firth of Forth
- Orange berries
- Coastal scene
- People
- Sandy beach
- Sea buck
```

---

### ✅ meta-llama/Llama-3.2-11B-Vision-Instruct

**Metrics:** 3.8 TPS | 189 tokens

```text
**Image Details**

* Location: Gullane, East Lothian, Scotland, UK
* Date: December
* Time: Afternoon
* Weather: Overcast
* Scene: Beach
* Features:
	+ Orange sea buckthorn bushes
	+ Sandy beach
	+ Rocky shoreline
	+ Two people walking on the beach
* Atmosphere: Peaceful
* Mood: Serene

**Image Description**

The image captures a serene and peaceful scene of a beach in Gullane, East Lothian, Scotland, UK. The overcast sky and calm sea create a soothing atmosphere, while the orange sea buckthorn bushes add a pop of color to the scene. Two people are seen walking on the beach, adding a sense of human presence to the otherwise peaceful scene. The image conveys a sense of tranquility and relaxation, inviting the viewer to step into the serene coastal environment.
```

---

<!-- markdownlint-enable MD013 -->

---

## System/Hardware Information

- **OS**: Darwin 25.2.0
- **macOS Version**: 26.2
- **SDK Version**: 26.2
- **Python Version**: 3.13.9
- **Architecture**: arm64
- **GPU/Chip**: Apple M4 Max
- **GPU Cores**: 40
- **Metal Support**: Metal 4
- **RAM**: 128.0 GB
- **CPU Cores (Physical)**: 16
- **CPU Cores (Logical)**: 16

## Library Versions

- `Pillow`: `12.0.0`
- `huggingface-hub`: `1.2.3`
- `mlx`: `0.30.3.dev20251228+d9b950eb`
- `mlx-lm`: `0.30.0`
- `mlx-vlm`: `0.3.10`
- `tokenizers`: `0.22.1`
- `transformers`: `5.0.0rc1`

_Report generated on: 2025-12-28 20:50:37 GMT_
