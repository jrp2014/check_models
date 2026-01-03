# Model Performance Results

_Generated on 2026-01-03 00:29:57 GMT_

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/LFM2-VL-1.6B-8bit` (291.0 tps)
- **💾 Most efficient:** `qnguyen3/nanoLLaVA` (4.5 GB)
- **⚡ Fastest load:** `HuggingFaceTB/SmolVLM-Instruct` (0.00s)
- **📊 Average TPS:** 61.8 across 26 models

## 📈 Resource Usage

- **Total peak memory:** 485.8 GB
- **Average peak memory:** 18.7 GB
- **Memory efficiency:** 179 tokens/GB

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

- **Generation Tps**: Avg: 61.8 | Min: 3.81 | Max: 291
- **Peak Memory**: Avg: 19 | Min: 4.5 | Max: 53
- **Total Time**: Avg: 24.70s | Min: 2.94s | Max: 83.12s
- **Generation Time**: Avg: 19.93s | Min: 1.92s | Max: 53.71s
- **Model Load Time**: Avg: 4.77s | Min: 1.02s | Max: 43.05s

> **Prompt used:**
>
> Provide a factual caption, description, and keywords suitable for cataloguing, or searching for, the image.
>
> Context: The image relates to 'Gingerbread House, Morrison's Haven, Prestongrange, Scotland, United Kingdom, UK
> Here is a caption for the image, written in a neutral and descriptive style suitable for a periodical, blog, or photography website.
>
> **Caption:**
>
> The iconic silhouette of Arthur's Seat is visible across the Firth of Forth during a late winter afternoon in January. This photograph was taken from the rocky shoreline of Morrison's Haven in Prestongrange, East Lothian, Scotland. As the sun dips towards the horizon, its golden light breaks through a band of heavy clouds, illuminating the sky above the distant Edinburgh skyline. In the foreground, the cool, dark stones of the coast contrast with the choppy, blue-grey water as small waves break on the shore.
>
> Historically, Morrison's Haven was a significant 16th-century harbour, crucial to the area's once-thriving coal and salt industries. Today, it serves as a quiet location offering expansive views of the capital city and the rugged beauty of the Scottish coastline.'
>
> The photo was taken around 2026-01-02 15:56:46 GMT from GPS 55.953633°N, 3.007600°W . Focus on visual content, drawing on any available contextual information for specificity. Do not speculate.

**Note:** Results sorted: errors first, then by generation time (fastest to slowest).

**Overall runtime:** 679.98s

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
| `qnguyen3/nanoLLaVA`                                    | 151,645 |               329 |                   125 |            454 |        1,365 |       101 |         4.5 |            1.92s |      1.02s |       2.94s | context-ignored      |
| `HuggingFaceTB/SmolVLM-Instruct`                        |  49,154 |             1,514 |                    93 |          1,607 |        1,072 |       114 |         5.5 |            2.63s |      1.37s |       4.00s | context-ignored      |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |  49,279 |             1,414 |                   167 |          1,581 |        1,202 |       115 |         5.5 |            2.98s |      1.31s |       4.29s |                      |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |       1 |             4,415 |                   121 |          4,536 |        1,258 |       111 |         8.8 |            4.91s |      2.83s |       7.74s |                      |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |       1 |             1,343 |                    90 |          1,433 |          477 |      41.2 |          12 |            5.31s |      3.83s |       9.14s | context-ignored      |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |       7 |             2,111 |                   171 |          2,282 |          462 |       291 |          53 |            5.62s |      1.15s |       6.76s |                      |
| `mlx-community/SmolVLM-Instruct-bf16`                   |  12,753 |             1,514 |                   500 |          2,014 |        1,236 |       113 |         5.5 |            6.03s |      1.34s |       7.37s | verbose              |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |       2 |             2,969 |                   373 |          3,342 |          863 |       156 |         7.4 |            6.14s |      1.91s |       8.05s | verbose              |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |       1 |             1,343 |                    99 |          1,442 |        1,483 |      17.7 |          11 |            6.83s |      3.51s |      10.34s | context-ignored      |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | 128,001 |             2,616 |                   143 |          2,759 |          672 |      29.2 |          19 |            9.17s |      3.58s |      12.75s | formatting           |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |     106 |               585 |                   474 |          1,059 |          607 |      41.6 |          17 |           12.70s |      5.42s |      18.12s | verbose              |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |       2 |             2,517 |                   118 |          2,635 |          239 |      41.0 |          12 |           13.77s |      1.86s |      15.63s |                      |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |     106 |               585 |                   373 |            958 |          177 |      27.9 |          19 |           17.01s |      5.10s |      22.11s | bullets(24)          |
| `mlx-community/pixtral-12b-8bit`                        |       2 |             3,127 |                   364 |          3,491 |          487 |      33.6 |          16 |           17.58s |      3.61s |      21.19s | verbose, bullets(24) |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      | 128,009 |               305 |                   124 |            429 |         94.7 |      8.78 |          15 |           17.70s |      2.86s |      20.56s | context-ignored      |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |       2 |             2,404 |                   255 |          2,659 |          212 |      27.4 |          22 |           20.95s |      4.13s |      25.08s |                      |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         | 115,430 |             3,225 |                   500 |          3,725 |          400 |      35.1 |          15 |           22.63s |      3.23s |      25.85s | verbose              |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |       1 |             1,343 |                   102 |          1,445 |          468 |      4.89 |          27 |           24.08s |      5.35s |      29.43s | context-ignored      |
| `mlx-community/pixtral-12b-bf16`                        |       2 |             3,127 |                   350 |          3,477 |          534 |      18.9 |          27 |           24.70s |      5.13s |      29.83s | verbose, bullets(24) |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |   3,884 |             6,385 |                   500 |          6,885 |          300 |      47.3 |          13 |           32.25s |      2.66s |      34.91s | formatting           |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |     106 |               585 |                   407 |            992 |          145 |      14.3 |          34 |           32.93s |      7.23s |      40.16s | bullets(27)          |
| `mlx-community/Molmo-7B-D-0924-8bit`                    | 151,643 |             1,513 |                   238 |          1,751 |         44.1 |      45.1 |          41 |           40.07s |     43.05s |      83.12s |                      |
| `mlx-community/Molmo-7B-D-0924-bf16`                    | 151,643 |             1,513 |                   245 |          1,758 |         43.7 |      28.1 |          48 |           43.79s |      3.64s |      47.43s |                      |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | 128,009 |               306 |                   151 |            457 |         65.4 |      3.81 |          25 |           44.78s |      5.51s |      50.29s |                      |
| `Qwen/Qwen3-VL-2B-Instruct`                             | 151,645 |            16,546 |                   217 |         16,763 |          374 |      68.8 |          12 |           48.04s |      1.69s |      49.73s | bullets(20)          |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |     279 |            16,548 |                   500 |         17,048 |          358 |      71.4 |          12 |           53.71s |      1.77s |      55.48s | verbose              |

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
> (/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages/transformers/processing_utils.py)
**Type:** `ValueError`

---

### ❌ mlx-community/Kimi-VL-A3B-Thinking-8bit

**Status:** Failed (Lib Version)
**Error:**

> Model loading failed: cannot import name '_validate_images_text_input_order'
> from 'transformers.processing_utils'
> (/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages/transformers/processing_utils.py)
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

**Metrics:** 101 TPS | 125 tokens

```text
The image captures a picturesque scene of the Firth of Forth during winter, with the iconic Arthur's Seat in the backdrop. The photograph was taken on January 2nd, 2026, at 15:56:46 GMT. The sky is filled with heavy clouds, and the water is choppy, with small waves breaking on the shore. The photograph was taken from a GPS 55.953633°N, 3.007600°W, and was taken at 15:56:46 GMT.
```

---

### ✅ HuggingFaceTB/SmolVLM-Instruct

**Metrics:** 114 TPS | 93 tokens

```text
 The image captures a serene winter scene in Scotland, showcasing the iconic Arthur's Seat mountain and the Firth of Forth. The sky is painted in hues of orange and yellow, with dark clouds overhead. The water in the foreground is a cool, dark blue, contrasting with the bright, golden light breaking through the clouds. The shoreline is rocky, with small waves breaking on the shore. The image is a beautiful and serene depiction of the Scottish landscape.
```

---

### ✅ mlx-community/SmolVLM2-2.2B-Instruct-mlx

**Metrics:** 115 TPS | 167 tokens

```text
 The image captures the stunning view of Arthur's Seat, a prominent mountain in the Firth of Forth, Scotland. The photograph was taken from Morrison's Haven in Prestongrange, East Lothian, during a late winter afternoon in January. The sky is painted with hues of orange and yellow, reflecting the setting sun. The mountain's silhouette stands prominently against the sky, while the water of the Firth of Forth churns with small waves, adding a dynamic element to the scene. The foreground is dominated by dark rocks, providing a stark contrast to the lighter water and sky. The image also provides historical context, noting that Morrison's Haven was a significant 16th-century harbour and today serves as a quiet location offering expansive views of the capital city and the rugged beauty of the Scottish coastline.
```

---

### ✅ mlx-community/paligemma2-3b-pt-896-4bit

**Metrics:** 111 TPS | 121 tokens

```text
**Caption:**Morrison's Haven is a quiet location, with a view of Edinburgh and the rugged beauty of the Scottish coastline. The iconic silhouette of Arthur's Seat is visible across the Firth of Forth during a late winter afternoon in January. This photograph was taken around 2026-01-02 15:56:46 GMT from GPS 55.953633°N, 3.007600°W . Focus on visual content, drawing on any available contextual information for specificity. Do not speculate.
```

---

### ✅ mlx-community/paligemma2-10b-ft-docci-448-6bit

**Metrics:** 41.2 TPS | 90 tokens

```text
A long-shot view of a body of water with a rocky shore in the foreground. The water is dark blue and has small waves coming towards the shore. The rocks on the shore are dark and wet, and there are some trees on the other side of the water in the distance. The sky is orange and blue, and there are dark clouds in the sky. There is a mountain in the distance on the other side of the water.
```

---

### ✅ mlx-community/LFM2-VL-1.6B-8bit

**Metrics:** 291 TPS | 171 tokens

```text
Caption:

The iconic silhouette of Arthur's Seat is visible across the Firth of Forth during a late winter afternoon in January. This photograph was taken from the rocky shoreline of Morrison's Haven in Prestongrange, East Lothian, Scotland. As the sun dips towards the horizon, its golden light breaks through a band of heavy clouds, illuminating the sky above the distant Edinburgh skyline. In the foreground, the cool, dark stones of the coast contrast with the choppy, blue-grey water as small waves break on the shore. Historically, Morrison's Haven was a significant 16th-century harbour, crucial to the area's once-thriving coal and salt industries. Today, it serves as a quiet location offering expansive views of the capital city and the rugged beauty of the Scottish coastline.
```

---

### ✅ mlx-community/SmolVLM-Instruct-bf16

**Metrics:** 113 TPS | 500 tokens

```text
 **Image Description:**

The image depicts a serene coastal scene, featuring a body of water, rocky shorelines, and distant mountainous terrain. The focal point is a large, dark mountain, which appears to be Arthur's Seat, a prominent landmark in Edinburgh, Scotland. The mountain is situated in the background, partially obscured by a layer of clouds that lend a dramatic and ethereal quality to the scene.

The foreground of the image shows a rocky shoreline with small waves breaking on the shore. The water is a deep blue-grey, reflecting the light of the setting sun. The shoreline is composed of a mix of large and small rocks, some of which are covered in moss or other vegetation.

In the background, the sky is a mix of dark and light colors, with the lightest areas being near the horizon and the darkest areas closer to the mountain. The clouds are dense and appear to be moving quickly, suggesting a windy or stormy day. The sky is lit up by the setting sun, casting a golden glow over the landscape.

The mountain is partially obscured by the clouds, but its outline is still visible. The mountain is relatively tall and wide, with a flat top that appears to be covered in vegetation. The mountain is a prominent feature in the landscape, and its silhouette is clearly defined against the sky.

The image captures a moment of tranquility and natural beauty, with the setting sun casting a warm glow on the scene. The water is calm, with small waves breaking on the shore, creating a peaceful and serene atmosphere.

**Contextual Information:**

Arthur's Seat is a significant landmark in Edinburgh, Scotland. It is a popular tourist destination due to its historical and cultural significance. The mountain has been a part of the city's landscape for centuries, and it played a crucial role in the city's salt and coal industries in the 16th century. Today, it is a popular hiking destination and offers stunning views of the city and the surrounding countryside.

The image was taken on January 2, 2026, at approximately 15:56 GMT. The coordinates provided are accurate to within a few meters, indicating the precision of the location. The image was taken from a GPS position, which is a common method for capturing precise geographic data.

**Image Analysis:**

The image is a beautiful and serene landscape photograph that captures the natural beauty of the Scottish
```

---

### ✅ mlx-community/Ministral-3-3B-Instruct-2512-4bit

**Metrics:** 156 TPS | 373 tokens

```text
Based on the provided context and the visual elements of the image, here is a factual caption and description tailored to the location you mentioned, while ensuring accuracy with available data:

**Caption:**
A serene coastal scene at Morrison’s Haven near Prestongrange, Scotland, captures the interplay of rugged shoreline and the tranquil Firth of Forth. The photograph captures the dramatic contrast between the rocky foreground and the expansive, slightly choppy waters of the estuary. The sky, partially obscured by clouds, is bathed in a soft, golden hue as the sun sets, casting a peaceful ambiance over the scene.

**Description:**
The image depicts a coastal landscape with a rocky shoreline in the foreground. The waves gently lap against the jagged stones, creating a natural breakwater. In the background, the calm waters of the Firth of Forth stretch out, leading to a distant view of hills or low mountains. The lighting suggests late afternoon or early evening, with a mix of cloud cover and a warm, golden glow from the setting sun.

**Keywords:**
- Coastal landscape, Scotland, Firth of Forth, rocky shoreline, natural breakwater, late afternoon, golden light, waves, estuary, Prestongrange, East Lothian, UK, serene seascape, coastal erosion, maritime scenery.

*Note:* Since the provided context mentions a fictional or hypothetical date (2026-01-02) and the image itself does not align with known locations of Morrison’s Haven, the caption and description are based on the visual elements and the given fictional location description. If this is not the actual image of Morrison’s Haven, the caption should be adjusted to reflect the actual location and date of the photograph. For precise identification, a direct reference to the actual image or verified location would be required.
```

---

### ✅ mlx-community/paligemma2-3b-ft-docci-448-bf16

**Metrics:** 17.7 TPS | 99 tokens

```text
A large, dark boulder sits on the left edge of the frame, with a small wave breaking on the shore. The water is a deep blue-green, with the sun shining on the left side of the water. The water ripples and creates a white foam on the surface. The horizon is a deep blue, with a band of bright orange and yellow from the setting sun. The distant skyline is a dark silhouette of a mountain range. The sky is filled with thick, dark clouds.
```

---

### ✅ mlx-community/Idefics3-8B-Llama3-bf16

**Metrics:** 29.2 TPS | 143 tokens

```text
 The image depicts a serene coastal scene at Morrison's Haven in Prestongrange, Scotland. The photograph captures the tranquil waters of the Firth of Forth, with the silhouette of Arthur's Seat, a prominent hill in Edinburgh, visible in the background. The sky is adorned with a mix of dark clouds and patches of golden sunlight, suggesting a late afternoon or early evening setting. The shoreline is characterized by dark, rugged stones, and the waves gently lapping against the shore create a soothing melody. The photograph was taken on January 2, 2026, at 15:56:46 GMT, with the GPS coordinates 55.953633°N, 3.007600°W.<end_of_utterance>
```

---

### ✅ mlx-community/gemma-3n-E4B-it-bf16

**Metrics:** 41.6 TPS | 474 tokens

```text
## Caption:

**Golden Hour at the Shores of Moray Firth**

A dramatic coastal scene unfolds at Moray's Haven in Prestongrange, East Lothian, captured during a late January afternoon. The iconic silhouette of Arthur's Seat rises in the distance across the Firth of Forth, framed by a dynamic sky of heavy clouds and streaks of warm, golden light.

The rugged shoreline, characterized by dark stones and the gentle rhythm of waves, contrasts beautifully with the cool, grey-blue waters of the Firth. This picturesque location, historically a vital hub for the region's fishing and shipbuilding industries, now offers stunning panoramic views of Edinburgh and the scenic Scottish coastline.

**Captured on January 2nd, 2026.**

**Coordinates:** 55.953633°N, 3.007600°W

## Description:

This photograph showcases the captivating beauty of the coastline at Moray's Haven in Prestongrange, East Lothian. The image captures a serene yet dramatic moment during a late January afternoon, with the sun casting a warm glow on the horizon.

The foreground features a rocky shoreline, with dark, weathered stones meeting the gentle waves of the Moray Firth. The water displays a range of cool tones, from deep blue to grey, reflecting the overcast sky.

In the midground, a long, low-lying landmass stretches across the Firth, providing a visual anchor and a sense of depth. Beyond this, the majestic outline of Arthur's Seat dominates the skyline, its familiar form softened by the atmospheric conditions.

The sky is a key element of the composition, with a mix of dark, heavy clouds and patches of warm, golden light breaking through. This interplay of light and shadow creates a visually compelling and evocative scene.

The overall mood of the photograph is one of tranquility and natural beauty, highlighting the rugged charm of the Scottish coastline.

## Keywords:

Moray's Haven, Prestongrange, East Lothian, Scotland, UK, Firth of Forth, Arthur's Seat, coastline, sea, waves, rocks, sky, clouds, golden hour, winter, landscape, scenic, coastal, nature, photography, January, 2026.
```

---

### ✅ mlx-community/llava-v1.6-mistral-7b-8bit

**Metrics:** 41 TPS | 118 tokens

```text
The image is a photograph of a coastal landscape during the late afternoon. The sky is a mix of blue, grey, and orange hues, with the sun low on the horizon. The silhouette of Arthur's Seat is visible in the distance, with the coastline and water in the foreground. The photograph captures the tranquil beauty of the Scottish coastline, with the city skyline faintly visible in the distance. The image is taken from the rocky shoreline of Morrison's Haven, East Lothian, Scotland.
```

---

### ✅ mlx-community/gemma-3-27b-it-qat-4bit

**Metrics:** 27.9 TPS | 373 tokens

```text
## Image Catalogue Information:

**Caption:**

A late winter view of Arthur's Seat across the Firth of Forth, photographed from the rocky shoreline of Morrison's Haven, Prestongrange, East Lothian, Scotland. The image captures a low sun breaking through cloud cover, illuminating the skyline of Edinburgh in the distance. The foreground features dark, textured rocks and choppy blue-grey water with small waves.

**Description:**

Wide-angle landscape photograph depicting a coastal scene at Morrison's Haven, Prestongrange, Scotland. The composition is divided into three distinct layers: the foreground of dark, angular rocks and breaking waves; the mid-ground of the Firth of Forth with a choppy sea surface; and the background featuring the distant silhouette of Arthur's Seat and the Edinburgh skyline. The sky is dominated by heavy, layered clouds with a band of golden light breaking through near the horizon. The overall tone is cool and atmospheric, with a contrast between the dark foreground and the illuminated background.

**Keywords:**

*   Morrison's Haven
*   Prestongrange
*   East Lothian
*   Scotland
*   Edinburgh
*   Arthur's Seat
*   Firth of Forth
*   Scottish Coastline
*   Coastal Landscape
*   Winter
*   January
*   Seascape
*   Rocks
*   Waves
*   Sky
*   Clouds
*   Golden Hour
*   Horizon
*   Landscape Photography
*   UK
*   United Kingdom
*   55.953633°N, 3.007600°W
*   2026-01-02 15:56:46 GMT
*   Historical Site (Coal/Salt Industry)




```

---

### ✅ mlx-community/pixtral-12b-8bit

**Metrics:** 33.6 TPS | 364 tokens

```text
### Description
This image captures the scenic view of Arthur's Seat, a prominent hill in Edinburgh, Scotland, as seen from the rocky shoreline of Morrison's Haven in Prestongrange, East Lothian. The photograph is taken during a late winter afternoon, with the sun casting a golden hue through the clouds, illuminating the sky and creating a dramatic backdrop. The foreground features the rugged, dark stones of the coast contrasting with the choppy, blue-grey waters of the Firth of Forth. Small waves gently break against the shore, adding a sense of movement to the tranquil scene.

### Keywords
- Arthur's Seat
- Morrison's Haven
- Prestongrange
- East Lothian
- Firth of Forth
- Edinburgh skyline
- Late winter afternoon
- Rocky shoreline
- Golden sunset
- Scottish coastline
- Historical harbour
- Coal and salt industries
- Tranquil scene
- Waves breaking on the shore

### Cataloguing Information
- **Location:** Morrison's Haven, Prestongrange, East Lothian, Scotland, United Kingdom
- **Date:** 2026-01-02
- **Time:** 15:56:46 GMT
- **GPS Coordinates:** 55.953633° N, 3.007600° W
- **Subject:** Landscape photography of Arthur's Seat and the Firth of Forth
- **Historical Context:** Former harbour significant for coal and salt industries
- **Season:** Late winter
- **Weather:** Cloudy with patches of sunlight
- **Mood:** Tranquil and serene
- **Photographer:** [Photographer's Name]
```

---

### ✅ mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

**Metrics:** 8.78 TPS | 124 tokens

```text
I'm happy to help you with your question. However, I must point out that the text you provided is not a question, but rather a passage of text that appears to be a poetic or descriptive piece about a place or a scene. It does not contain a specific question that I can answer.

If you could provide a specific question related to the text, I would be happy to try and help you. Alternatively, if you would like me to summarize the passage or provide some general information about the place or scene it describes, I can do that as well. Please let me know how I can assist you.
```

---

### ✅ mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

**Metrics:** 27.4 TPS | 255 tokens

```text
**Caption:**
A serene winter scene at Morrison's Haven, Prestongrange, Scotland, showcasing the rugged coastline and distant view of Arthur's Seat across the Firth of Forth. The late afternoon sun casts a golden glow through the clouds, highlighting the contrast between the dark rocks and the choppy waters.

**Description:**
The image captures a tranquil moment at Morrison's Haven, a historic harbour in Prestongrange, East Lothian. The foreground features a rocky shoreline with dark, jagged stones, where small waves break gently. Beyond the shore, the Firth of Forth stretches out, its waters a mix of blue and grey under the fading light. In the distance, the silhouette of Arthur's Seat, a prominent hill in Edinburgh, rises against a sky painted with hues of gold and soft clouds. The scene evokes a sense of quiet beauty and historical significance, reflecting the area's past as a key hub for coal and salt industries.

**Keywords:**
Morrison's Haven, Prestongrange, Scotland, UK, Arthur's Seat, Firth of Forth, winter, coastline, rocky shore, sunset, golden light, historical harbour, East Lothian, Edinburgh skyline, choppy water, natural landscape, photography.
```

---

### ✅ mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX

**Metrics:** 35.1 TPS | 500 tokens

```text
Here are my reasoning steps:
The user wants a factual caption, description, and keywords suitable for cataloguing or searching for the image. The context is "Gingerbread House, Morrison's Haven, Prestongrange, Scotland, United Kingdom, UK". The user gave a caption example that mentions Arthur's Seat, Firth of Forth, etc. However, the user wants a factual caption, description, and keywords. The user says "Provide a factual caption, description, and keywords suitable for cataloguing, or searching for, the image." The user also gave a context: "The image relates to 'Gingerbread House, Morrison's Haven, Prestongrange, Scotland, United Kingdom, UK". The user gave a caption example that mentions Arthur's Seat, etc. The user says "Focus on visual content, drawing on any available contextual information for specificity. Do not speculate."

Thus we need to produce a factual caption, description, and keywords. The image is a seascape at sunset (or sunrise?) with mountains in the distance, a rocky shoreline, waves, clouds, etc. The location is Morrison's Haven, Prestongrange, East Lothian, Scotland. The user gave a caption that mentions Arthur's Seat, but we need to verify if Arthur's Seat is visible. The image shows mountains in the distance across a body of water. The caption says "Arthur's Seat". However, we need to be careful: Arthur's Seat is in Edinburgh, which is about 30 km east of Prestongrange? Actually, Prestongrange is in East Lothian, near the Firth of Forth. Arthur's Seat is a hill in Holyrood Park in Edinburgh, which is across the Firth of Forth. So it's plausible that Arthur's Seat is visible across the water. However, the user says "Do not speculate." So we need to be careful about stating that Arthur's Seat is visible. If we are not certain, we should not claim that. The user gave a caption that includes that, but we need to produce a factual caption based on the image. The user says "Focus on visual content, drawing on any available contextual information for specificity. Do not speculate." So we need to describe what we see: a rocky shoreline, waves, a body of water (likely the Firth of Forth), a distant mountain range (maybe the Lammermuir Hills or the Ochil Hills?),
```

---

### ✅ mlx-community/paligemma2-10b-ft-docci-448-bf16

**Metrics:** 4.89 TPS | 102 tokens

```text
A long-shot view of a body of water with a rocky shore on the left side of the image. The water is dark blue, and there are small waves coming onto the shore. There are rocks on the shore in the foreground of the image. There is a hill in the distance on the right side of the image. There is a body of land in the distance on the left side of the image. The sky is orange and blue, and there are clouds covering the top of the image.
```

---

### ✅ mlx-community/pixtral-12b-bf16

**Metrics:** 18.9 TPS | 350 tokens

```text
### Description
This image captures a serene coastal scene at Morrison's Haven in Prestongrange, East Lothian, Scotland. The photograph is taken from the rocky shoreline, offering a view across the Firth of Forth towards the iconic silhouette of Arthur's Seat in Edinburgh. The late winter afternoon sky is filled with dramatic clouds, through which the sun casts a golden light, illuminating the horizon. The foreground features dark, rugged stones contrasting with the choppy, blue-grey waters of the Firth of Forth, where small waves gently break on the shore.

### Keywords
- Morrison's Haven
- Prestongrange
- Arthur's Seat
- Firth of Forth
- Edinburgh skyline
- Rocky shoreline
- Late winter afternoon
- Golden sunset
- Scottish coastline
- Historical harbour
- Coastal beauty
- East Lothian
- Scotland
- United Kingdom
- UK

### Cataloguing Information
- **Location:** Morrison's Haven, Prestongrange, East Lothian, Scotland, United Kingdom
- **Date:** 2026-01-02
- **Time:** 15:56:46 GMT
- **GPS Coordinates:** 55.953633° N, 3.007600° W
- **Subject:** Coastal landscape, historical harbour, Arthur's Seat silhouette, sunset
- **Season:** Late winter
- **Mood:** Serene, dramatic, natural beauty
- **Historical Context:** Formerly significant 16th-century harbour for coal and salt industries
- **Photographer:** [Photographer's Name]
```

---

### ✅ mlx-community/GLM-4.6V-Flash-6bit

**Metrics:** 47.3 TPS | 500 tokens

```text
<think>Got it, let's tackle this. The user wants a factual caption, description, and keywords for cataloguing/searching.

First, I need to extract key elements from the provided context. Let's list out the main components:

1. **Subject Matter**: Arthur's Seat silhouette, Firth of Forth, Morrison's Haven shoreline, Edinburgh skyline.
2. **Time & Season**: Late winter afternoon, January (specifically 2026-01-02 15:56:46 GMT, but maybe just the month/year for cataloguing).
3. **Location Details**: Morrison's Haven, Prestongrange, East Lothian, Scotland, United Kingdom. GPS coordinates provided.
4. **Visual Elements**: Rocky shoreline (dark stones), choppy blue-grey water, small waves breaking, golden light breaking through heavy clouds, distant Edinburgh skyline with a tower (maybe the Scott Monument? But the context doesn't specify, so keep it general).
5. **Historical Context**: 16th-century harbour, coal and salt industries, now quiet location with views.

Now, structure the caption, description, and keywords.

**Caption**: Should be concise but include key identifiers. Maybe: "Silhouette of Arthur's Seat across the Firth of Forth from Morrison's Haven, Prestongrange, Scotland, during a late winter afternoon in January."

**Description**: More detailed, covering visual elements, time, location, and historical context. Need to be factual, no speculation.

**Keywords**: Should be searchable terms. Include location names, landmarks, time of day, weather, water features, historical aspects.

Let me check if all key points are included:

- Arthur's Seat silhouette
- Firth of Forth
- Morrison's Haven shoreline
- Prestongrange, East Lothian, Scotland
- Late winter afternoon, January (specific date/time from GPS, but maybe just month/year for cataloguing)
- Rocky shoreline (dark stones), choppy blue-grey water, small waves breaking
- Golden light breaking through heavy clouds
- Distant Edinburgh skyline
- 16th-century harbour, coal and salt industries, now quiet location with views

Now, draft the caption, description, and keywords.

**Caption**: Silhouette of Arthur's Seat across the Firth of Forth from Morrison's Haven, Prestongrange, Scotland, during a late winter afternoon in January.

**Description**: This image captures the iconic silhouette of Arthur's Seat as seen
```

---

### ✅ mlx-community/gemma-3-27b-it-qat-8bit

**Metrics:** 14.3 TPS | 407 tokens

```text
## Image Metadata:

**Caption:**

The silhouette of Arthur's Seat is visible across the Firth of Forth during a late winter afternoon in January. This photograph was taken from the rocky shoreline of Morrison's Haven in Prestongrange, East Lothian, Scotland. As the sun dips towards the horizon, its golden light breaks through a band of heavy clouds, illuminating the sky above the distant Edinburgh skyline. In the foreground, the cool, dark stones of the coast contrast with the choppy, blue-grey water as small waves break on the shore. Historically, Morrison's Haven was a significant 16th-century harbour, crucial to the area's once-thriving coal and salt industries.

**Description:**

A landscape photograph capturing a coastal view at Morrison's Haven, Prestongrange, Scotland. The foreground features a rocky shoreline with waves breaking on the beach. The Firth of Forth extends to the horizon, with the Edinburgh skyline and Arthur's Seat visible in the distance, silhouetted against a dramatic sky of golden light breaking through heavy clouds. The water is choppy and blue-grey.

**Keywords:**

*   Morrison's Haven
*   Prestongrange
*   East Lothian
*   Scotland
*   Edinburgh
*   Arthur's Seat
*   Firth of Forth
*   Coastline
*   Coastal Landscape
*   Winter
*   January
*   Sunset
*   Sky
*   Clouds
*   Rocks
*   Waves
*   Harbour (historical)
*   Coal Industry (historical)
*   Salt Industry (historical)
*   UK
*   United Kingdom
*   Landscape Photography
*   2026-01-02
*   55.953633°N, 3.007600°W
*   GMT
*   Silhouetted skyline
*   Golden hour




```

---

### ✅ mlx-community/Molmo-7B-D-0924-8bit

**Metrics:** 45.1 TPS | 238 tokens

```text
 Here is a caption for the image, written in a neutral and descriptive style suitable for a periodical, blog, or photography website:

**Caption:**

The iconic silhouette of Arthur's Seat is visible across the Firth of Forth during a late winter afternoon in January. This photograph was taken from the rocky shoreline of Morrison's Haven in Prestongrange, East Lothian, Scotland. As the sun dips towards the horizon, its golden light breaks through a band of heavy clouds, illuminating the sky above the distant Edinburgh skyline. In the foreground, the cool, dark stones of the coast contrast with the choppy, blue-grey water as small waves break on the shore.

Historically, Morrison's Haven was a significant 16th-century harbour, crucial to the area's once-thriving coal and salt trade. Today, it serves as a quiet location offering expansive views of the capital city and the rugged beauty of the Scottish coastline.

Keywords: Arthur's Seat, Firth of Forth, Edinburgh, Scotland, East Lothian, January, sunset, clouds, horizon, rocky shore, waves, historical harbour, coal trade, salt trade, Morrison's Haven
```

---

### ✅ mlx-community/Molmo-7B-D-0924-bf16

**Metrics:** 28.1 TPS | 245 tokens

```text
 Here is a caption for the image, written in a neutral and descriptive style suitable for a periodical, blog, or photography website:

**Caption:**

The iconic silhouette of Arthur's Seat is visible across the Firth of Forth during a late winter afternoon in January. This photograph was taken from the rocky shoreline of Morrison's Haven in Prestongrange, East Lothian, Scotland. As the sun dips towards the horizon, its golden light breaks through a band of heavy clouds, illuminating the sky above the distant Edinburgh skyline. In the foreground, the cool, dark stones of the coast contrast with the choppy, blue-grey water as small waves break on the shore.

Historically, Morrison's Haven was a significant 16th-century harbour, crucial to the area's once-thriving coal and salt trade. Today, it serves as a quiet location offering expansive views of the capital city and the rugged beauty of the Scottish coastline.

The photo was taken around 2026-01-02 15:56:46 GMT from GPS 55.953633°N, 3.007600°W.
```

---

### ✅ meta-llama/Llama-3.2-11B-Vision-Instruct

**Metrics:** 3.81 TPS | 151 tokens

```text
I can provide a factual caption for the image, but I must correct the location information. The image appears to be of Arthur's Seat, a hill in Edinburgh, Scotland, not a location called "Gingerbread House" or "Morrison's Haven." Here is a revised caption:

**Caption:**

Arthur's Seat, a prominent hill in Edinburgh, Scotland, is seen at sunset. The image captures the hill's silhouette against the vibrant sky, with the city of Edinburgh visible in the distance. The photo was taken from the shore of the Firth of Forth, showcasing the natural beauty of the area. The image highlights the unique landscape of the area, with the hill's rugged shape and the surrounding water creating a picturesque scene.
```

---

### ✅ Qwen/Qwen3-VL-2B-Instruct

**Metrics:** 68.8 TPS | 217 tokens

```text
**Caption:**

A serene coastal scene at sunset, featuring the iconic Arthur's Seat mountain in the distance across the Firth of Forth. The photograph captures the rugged shoreline of Morrison's Haven in Prestongrange, East Lothian, Scotland, with the dark, textured rocks of the coast in the foreground. The sky is a dramatic canvas of deep blues and oranges, with clouds illuminated by the setting sun, while the distant Edinburgh skyline is visible on the horizon. The water is choppy, with small waves breaking on the shore.

**Keywords:**
- Arthur's Seat
- Morrison's Haven
- Prestongrange
- East Lothian
- Scotland
- Firth of Forth
- Coastal landscape
- Sunset
- Rocky shoreline
- Edinburgh skyline
- Arthur's Seat mountain
- Scottish coastline
- Winter sunset
- Photography
- Natural scenery
- Coastal view
- Historical harbour
- GPS coordinates
- 2026-01-02
- 15:56:46 GMT
```

---

### ✅ mlx-community/Qwen3-VL-2B-Thinking-bf16

**Metrics:** 71.4 TPS | 500 tokens

```text
Got it, let's tackle this. First, I need to create a factual caption, description, and keywords for the image. The context given is about Gingerbread House, Morrison's Haven, Prestongrange, Scotland, UK. Wait, but the image shows a coastal scene with Arthur's Seat, Firth of Forth, etc. Wait, the caption mentions Arthur's Seat, but the context says Gingerbread House? Wait, no, the user provided a caption that's supposed to be for the image, but the context says "The image relates to 'Gingerbread House, Morrison's Haven, Prestongrange, Scotland, United Kingdom, UK'". Wait, maybe there's a mix-up. Wait, the caption in the problem is the one to use, but the context is about Gingerbread House? Wait, no, the user says: "Context: The image relates to 'Gingerbread House, Morrison's Haven, Prestongrange, Scotland, United Kingdom, UK'." But the image in the problem is a coastal scene with Arthur's Seat. Wait, maybe the caption is supposed to be based on the image, but the context is a bit confusing. Wait, no, the user says: "Here is a caption for the image, written in a neutral and descriptive style suitable for a periodical, blog, or photography website." So the caption is the one provided, but we need to make sure it's factual. Wait, the problem says: "Provide a factual caption, description, and keywords suitable for cataloguing, or searching for, the image." So the caption should be factual, based on the image. Wait, the context says the image relates to Gingerbread House, but the image is of a coastal scene. Wait, maybe the Gingerbread House is a typo or misreference. Wait, no, the user provided a caption that's supposed to be for the image, but the context is about Gingerbread House. Wait, maybe the user made a mistake, but we need to follow the instructions.

Wait, the problem says: "Context: The image relates to 'Gingerbread House, Morrison's Haven, Prestongrange, Scotland, United Kingdom, UK'." But the image in the problem is a coastal scene with Arthur's Seat. Wait, maybe the Gingerbread House is a fictional or mislabeled thing, but the user provided a caption that's supposed to be for the image. Wait, no, the
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

- `Pillow`: `12.1.0`
- `huggingface-hub`: `1.2.3`
- `mlx`: `0.30.3.dev20260103+8de9ceb7`
- `mlx-lm`: `0.30.0`
- `mlx-vlm`: `0.3.10`
- `tokenizers`: `0.22.1`
- `transformers`: `5.0.0rc1`

_Report generated on: 2026-01-03 00:29:57 GMT_
