# Model Performance Results

_Generated on 2026-01-10 22:26:56 GMT_

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/LFM2-VL-1.6B-8bit` (288.6 tps)
- **💾 Most efficient:** `mlx-community/LFM2.5-VL-1.6B-bf16` (4.4 GB)
- **⚡ Fastest load:** `HuggingFaceTB/SmolVLM-Instruct` (0.00s)
- **📊 Average TPS:** 61.0 across 28 models

## 📈 Resource Usage

- **Total peak memory:** 502.1 GB
- **Average peak memory:** 17.9 GB
- **Memory efficiency:** 178 tokens/GB

## ⚠️ Quality Issues

- **❌ Failed Models (9):**
  - `microsoft/Florence-2-large-ft` (`Weight Mismatch`)
  - `mlx-community/InternVL3-14B-8bit` (`Processor Error`)
  - `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16` (`Lib Version`)
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit` (`Lib Version`)
  - `mlx-community/Phi-3.5-vision-instruct-bf16` (`Config Missing`)
  - `mlx-community/Qwen2-VL-2B-Instruct-4bit` (`OOM`)
  - `mlx-community/deepseek-vl2-8bit` (`Type Cast Error`)
  - `mlx-community/gemma-3-12b-pt-8bit` (`No Chat Template`)
  - `prince-canuma/Florence-2-large-ft` (`Model Error`)
- **🔄 Repetitive Output (3):**
  - `meta-llama/Llama-3.2-11B-Vision-Instruct` (token: `phrase: "seasons, seasons, seasons, sea..."`)
  - `microsoft/Phi-3.5-vision-instruct` (token: `phrase: "# instruction # solution..."`)
  - `mlx-community/paligemma2-10b-ft-docci-448-bf16` (token: `phrase: "the sun is hitting..."`)
- **👻 Hallucinations (1):**
  - `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`
- **📝 Formatting Issues (2):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/Idefics3-8B-Llama3-bf16`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 61.0 | Min: 3.84 | Max: 289
- **Peak Memory**: Avg: 18 | Min: 4.4 | Max: 53
- **Total Time**: Avg: 29.42s | Min: 2.56s | Max: 138.10s
- **Generation Time**: Avg: 26.30s | Min: 1.55s | Max: 133.89s
- **Model Load Time**: Avg: 3.12s | Min: 0.99s | Max: 7.14s

> **Prompt used:**
>
> Provide a factual caption, description, and keywords suitable for cataloguing, or searching for, the image.
>
> Context: The image relates to 'Trossachs, Stirling, Scotland, United Kingdom, UK
> Here is a caption for the image:
>
> The low sun of a January afternoon casts a warm, golden light across a mountain peak in the Trossachs, Stirling, Scotland. Below, the glen remains in cool shadow, where the white trunks of leafless birch trees stand out against the dark winter vegetation and lingering patches of snow. This striking contrast between light and shadow highlights the rugged beauty of the Scottish landscape in winter. The Trossachs region, often called 'The Highlands in Miniature,' has a rich history and was famously romanticized by the writings of Sir Walter Scott, which helped define the enduring image of the Highlands.'
>
> The photo was taken around 2026-01-03 14:57:54 GMT . Focus on visual content, drawing on any available contextual information for specificity. Do not speculate.

**Note:** Results sorted: errors first, then by generation time (fastest to slowest).

**Overall runtime:** 855.26s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Token |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                    |
|:--------------------------------------------------------|--------:|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:----------------------------------|
| `microsoft/Florence-2-large-ft`                         |         |                   |                       |                |              |           |             |                  |            |             |                                   |
| `mlx-community/InternVL3-14B-8bit`                      |         |                   |                       |                |              |           |             |                  |            |             |                                   |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |         |                   |                       |                |              |           |             |                  |            |             |                                   |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |         |                   |                       |                |              |           |             |                  |            |             |                                   |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |         |                   |                       |                |              |           |             |                  |            |             |                                   |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |         |                   |                       |                |              |           |             |                  |            |             |                                   |
| `mlx-community/deepseek-vl2-8bit`                       |         |                   |                       |                |              |           |             |                  |            |             |                                   |
| `mlx-community/gemma-3-12b-pt-8bit`                     |         |                   |                       |                |              |           |             |                  |            |             |                                   |
| `prince-canuma/Florence-2-large-ft`                     |         |                   |                       |                |              |           |             |                  |            |             |                                   |
| `qnguyen3/nanoLLaVA`                                    | 151,645 |               236 |                    91 |            327 |        1,025 |       103 |         4.5 |            1.55s |      1.01s |       2.56s |                                   |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |  49,279 |             1,314 |                   109 |          1,423 |        1,135 |       115 |         5.5 |            2.49s |      1.28s |       3.77s |                                   |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |       7 |             2,022 |                    14 |          2,036 |          793 |       289 |          53 |            3.09s |      0.99s |       4.07s |                                   |
| `HuggingFaceTB/SmolVLM-Instruct`                        |  49,154 |             1,414 |                   195 |          1,609 |        1,186 |       115 |         5.5 |            3.25s |      1.30s |       4.55s |                                   |
| `mlx-community/SmolVLM-Instruct-bf16`                   |  49,154 |             1,414 |                   222 |          1,636 |        1,161 |       113 |         5.5 |            3.55s |      1.31s |       4.86s |                                   |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     |       7 |             2,022 |                   241 |          2,263 |        1,744 |      90.4 |         4.4 |            4.23s |      1.17s |       5.40s |                                   |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |       1 |             4,315 |                    22 |          4,337 |        1,134 |       114 |         8.6 |            4.33s |      2.86s |       7.19s | context-ignored                   |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |       1 |             1,243 |                    94 |          1,337 |          487 |      41.8 |          11 |            5.13s |      3.84s |       8.98s |                                   |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |       2 |             2,994 |                   262 |          3,256 |          816 |       156 |         7.9 |            5.68s |      2.03s |       7.71s |                                   |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |       1 |             1,243 |                   113 |          1,356 |        1,449 |      18.0 |          11 |            7.44s |      3.59s |      11.02s | context-ignored                   |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |       2 |             2,516 |                     8 |          2,524 |          252 |      48.5 |          12 |           10.48s |      1.82s |      12.30s | context-ignored                   |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |     106 |               487 |                   460 |            947 |          707 |      41.0 |          17 |           12.24s |      4.96s |      17.20s | verbose                           |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | 128,001 |             2,535 |                   281 |          2,816 |          689 |      29.2 |          18 |           13.66s |      3.60s |      17.26s | formatting                        |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |     106 |               487 |                   300 |            787 |          166 |      27.4 |          19 |           14.25s |      4.97s |      19.22s | bullets(25)                       |
| `mlx-community/pixtral-12b-8bit`                        |       2 |             3,237 |                   350 |          3,587 |          508 |      34.0 |          16 |           16.99s |      3.30s |      20.29s | verbose, bullets(20)              |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |       2 |             2,429 |                   241 |          2,670 |          215 |      26.9 |          22 |           20.59s |      4.06s |      24.65s |                                   |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |     106 |               487 |                   282 |            769 |          159 |      15.2 |          34 |           21.97s |      7.14s |      29.11s | bullets(26)                       |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |   7,934 |             3,335 |                   500 |          3,835 |          396 |      34.1 |          15 |           23.44s |      3.26s |      26.70s | hallucination                     |
| `mlx-community/pixtral-12b-bf16`                        |       2 |             3,237 |                   351 |          3,588 |          529 |      19.1 |          27 |           24.79s |      5.13s |      29.91s | verbose, bullets(19)              |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |  19,050 |             6,265 |                   500 |          6,765 |          290 |      44.6 |          13 |           33.17s |      2.62s |      35.79s | verbose, formatting               |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      | 128,009 |               223 |                   243 |            466 |         73.7 |      7.37 |          15 |           36.36s |      3.03s |      39.39s | context-ignored                   |
| `mlx-community/Molmo-7B-D-0924-8bit`                    | 151,643 |             1,428 |                   126 |          1,554 |         42.3 |      43.5 |          41 |           37.16s |      5.42s |      42.58s |                                   |
| `mlx-community/Molmo-7B-D-0924-bf16`                    | 151,643 |             1,428 |                   172 |          1,600 |         44.4 |      22.8 |          47 |           40.15s |      3.65s |      43.81s |                                   |
| `microsoft/Phi-3.5-vision-instruct`                     |      13 |             1,019 |                   500 |          1,519 |          915 |      10.9 |          11 |           47.43s |      1.92s |      49.35s | repetitive(phrase: "# instruct... |
| `Qwen/Qwen3-VL-2B-Instruct`                             |  18,634 |            16,493 |                   500 |         16,993 |          376 |      69.6 |          12 |           51.43s |      1.58s |      53.01s | verbose, bullets(61)              |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |  14,190 |            16,495 |                   500 |         16,995 |          365 |      70.4 |          12 |           52.72s |      1.82s |      54.54s |                                   |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |     578 |             1,243 |                   500 |          1,743 |          438 |      4.91 |          27 |          104.99s |      5.48s |     110.47s | repetitive(phrase: "the sun is... |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              |      11 |               224 |                   500 |            724 |         66.6 |      3.84 |          25 |          133.89s |      4.21s |     138.10s | repetitive(phrase: "seasons, ...  |

<!-- markdownlint-enable MD013 MD033 MD034 MD037 MD049 -->

## Model Gallery

Full output from each model:

<!-- markdownlint-disable MD013 -->

### ❌ microsoft/Florence-2-large-ft

**Status:** Failed (Weight Mismatch)
**Error:** Model loading failed: Missing 1 parameters:
language_model.lm_head.weight.
**Type:** `ValueError`
**Package:** `mlx`

---

### ❌ mlx-community/InternVL3-14B-8bit

**Status:** Failed (Processor Error)
**Error:**

> Model loading failed: Received a InternVLImageProcessor for argument
> image_processor, but a ImageProcessingMixin was expected.
**Type:** `ValueError`
**Package:** `mlx-vlm`

---

### ❌ mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

**Status:** Failed (Lib Version)
**Error:**

> Model loading failed: cannot import name '_validate_images_text_input_order'
> from 'transformers.processing_utils'
> (/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/processing_utils.py)
**Type:** `ValueError`
**Package:** `mlx-vlm`

---

### ❌ mlx-community/Kimi-VL-A3B-Thinking-8bit

**Status:** Failed (Lib Version)
**Error:**

> Model loading failed: cannot import name '_validate_images_text_input_order'
> from 'transformers.processing_utils'
> (/opt/homebrew/Caskroom/miniconda/base/envs/mlx-vlm/lib/python3.13/site-packages/transformers/processing_utils.py)
**Type:** `ValueError`
**Package:** `mlx-vlm`

---

### ❌ mlx-community/Phi-3.5-vision-instruct-bf16

**Status:** Failed (Config Missing)
**Error:**

> Model loading failed:
> /Users/jrp/.cache/huggingface/hub/models--mlx-community--Phi-3.5-vision-instruct-bf16/snapshots/d8da684308c275a86659e2b36a9189b2f4aec8ea
> does not appear to have a file named image_processing_phi3_v.py. Checkout
> 'https://huggingface.co//Users/jrp/.cache/huggingface/hub/models--mlx-community--Phi-3.5-vision-instruct-bf16/snapshots/d8da684308c275a86659e2b36a9189b2f4aec8ea/tree/main'
> for available files.
**Type:** `ValueError`
**Package:** `mlx-vlm`

---

### ❌ mlx-community/Qwen2-VL-2B-Instruct-4bit

**Status:** Failed (OOM)
**Error:**

> Model runtime error during generation for
> mlx-community/Qwen2-VL-2B-Instruct-4bit: [metal::malloc] Attempting to
> allocate 135433060352 bytes which is greater than the maximum allowed buffer
> size of 86586540032 bytes.
**Type:** `ValueError`
**Package:** `mlx`

---

### ❌ mlx-community/deepseek-vl2-8bit

**Status:** Failed (Type Cast Error)
**Error:**

> Model runtime error during generation for mlx-community/deepseek-vl2-8bit:
> std::bad_cast
**Type:** `ValueError`
**Package:** `mlx`

---

### ❌ mlx-community/gemma-3-12b-pt-8bit

**Status:** Failed (No Chat Template)
**Error:**

> Cannot use chat template functions because tokenizer.chat_template is not
> set and no template argument was passed! For information about writing
> templates and setting the tokenizer.chat_template attribute, please see the
> documentation at
> https://huggingface.co/docs/transformers/main/en/chat_templating
**Type:** `ValueError`
**Package:** `mlx-vlm`

---

### ❌ prince-canuma/Florence-2-large-ft

**Status:** Failed (Model Error)
**Error:**

> Model loading failed: RobertaTokenizer has no attribute
> additional_special_tokens
**Type:** `ValueError`
**Package:** `mlx-vlm`

---

### ✅ qnguyen3/nanoLLaVA

**Metrics:** 103 TPS | 91 tokens

```text
The image captures a picturesque winter scene in the Trossachs, Stirling, Scotland. The mountain peak is blanketed in snow, with the sun casting a warm, golden light on the landscape. The trees, stripped of their leaves, stand tall against the winter sky, their branches reaching out into the sky. The image captures the contrast between the warm sun and the cool winter, with the snow-covered mountains and trees standing out against the sky.
```

---

### ✅ mlx-community/SmolVLM2-2.2B-Instruct-mlx

**Metrics:** 115 TPS | 109 tokens

```text
 The image is a landscape photograph of a mountain peak in the Trossachs region of Scotland. The peak is bathed in warm, golden light from the low sun, casting long shadows and highlighting the rugged beauty of the Scottish landscape. Below the peak, the glen remains in cool shadow, with the white trunks of leafless birch trees standing out against the dark winter vegetation and patches of snow. The image was taken on January 3, 2026, at 14:57:54 GMT.
```

---

### ✅ mlx-community/LFM2-VL-1.6B-8bit

**Metrics:** 289 TPS | 14 tokens

```text
Winter landscape of the Trossachs, Stirling, Scotland.
```

---

### ✅ HuggingFaceTB/SmolVLM-Instruct

**Metrics:** 115 TPS | 195 tokens

```text
 The image depicts a mountainous landscape in the Trossachs region of Scotland. The Trossachs are often referred to as 'The Highlands in Miniature,' and this description accurately captures the rugged beauty of the scene. The image shows a mountain peak bathed in warm sunlight, with the lower part of the mountain shrouded in shadow. The mountain is covered in a mix of brown and gray rocks, with patches of green vegetation peeking through. The foreground of the image features a dense forest of birch trees, which stand out against the dark backdrop of the mountain. The trees are bare, with only a few patches of snow still clinging to their branches. The ground is covered in a mix of brown and gray vegetation, with patches of snow still visible in some areas. The sky above is a mix of blue and gray, with a few scattered clouds. The overall effect is one of natural beauty and tranquility, with the sun casting a warm glow over the landscape.
```

---

### ✅ mlx-community/SmolVLM-Instruct-bf16

**Metrics:** 113 TPS | 222 tokens

```text
 The image depicts a mountainous landscape in the Trossachs region of Scotland. The Trossachs, also known as 'The Highlands in Miniature,' is a rugged and rugged area characterized by its high peaks, deep valleys, and dense forests. The image captures a scene of a mountain peak bathed in warm sunlight, with the surrounding landscape in shadow. The foreground features a dense forest of birch trees, which stand out against the dark underbrush and patches of snow. The mountain in the background is partially obscured by the trees, but its outline is still visible, showing a rugged and rocky terrain. The sky above is clear, with a few wisps of cloud, and the sun is casting a warm glow on the landscape.

The image provides a vivid portrayal of the Trossachs region's natural beauty, characterized by its rugged terrain and dense forests. The contrast between the bright sunlight and the dark shadows adds depth and texture to the scene, highlighting the region's natural beauty. The image captures the essence of the Scottish Highlands, which is often romanticized in literature and film.
```

---

### ✅ mlx-community/LFM2.5-VL-1.6B-bf16

**Metrics:** 90.4 TPS | 241 tokens

```text
Caption: "Golden Hour in the Trossachs: A Winter Landscape of Contrasts"

Description: This image captures the serene beauty of the Trossachs region in Stirling, Scotland, during the winter months. The sun, low in the sky, casts a warm, golden light across the rugged mountain peak, highlighting the textures and contours of the landscape. Below, the glen is cloaked in cool shadows, where the stark white trunks of leafless birch trees stand out against the dark winter vegetation and remnants of snow. The scene is a study in contrasts, with the bright sunlight illuminating the peaks and the darker, shadowed areas of the forest below. The Trossachs, often referred to as 'The Highlands in Miniature,' is a region steeped in history and romance, famously romanticized by Sir Walter Scott, which has contributed to its enduring image as a picturesque and tranquil Scottish landscape.

Keywords: Trossachs, Stirling, Scotland, winter landscape, mountain peak, birch trees, snow, golden hour, contrast, Scottish Highlands, Sir Walter Scott, natural beauty, tranquility, history, romance.
```

---

### ✅ mlx-community/paligemma2-3b-pt-896-4bit

**Metrics:** 114 TPS | 22 tokens

```text
Provide a factual caption, description, and keywords suitable for cataloguing, or searching for, the image.
```

---

### ✅ mlx-community/paligemma2-10b-ft-docci-448-6bit

**Metrics:** 41.8 TPS | 94 tokens

```text
A long-shot view of a mountain range in Scotland. The mountain is in the center of the frame, with the peak in the top right corner. The mountain is covered in trees and bushes that are brown and dead. The trees are in the foreground, and the mountain is in the background. The sun is hitting the top of the mountain, and the trees are in the shade. The sky is blue, and there are clouds in the top left corner.
```

---

### ✅ mlx-community/Ministral-3-3B-Instruct-2512-4bit

**Metrics:** 156 TPS | 262 tokens

```text
Based on the provided contextual information for specificity and the visual content of the image, here is a refined caption and description:

**Caption:**
Golden winter light bathes the majestic peaks of the Trossachs, Stirling, Scotland. The rugged slopes of the mountains contrast sharply with the leafless birch trees and scattered patches of snow, capturing the serene and raw beauty of a Scottish winter landscape.

**Description:**
The image depicts a winter scene in the Trossachs region, characterized by a prominent mountain peak bathed in warm golden light from a low sun. The foreground is dominated by bare birch trees, their white trunks stark against the dark, wintry underbrush. Snow lingers on the ground and lower branches, adding a touch of crispness to the scene. The overall atmosphere is tranquil and evocative of the rugged, natural beauty of Scotland’s Highlands.

**Keywords:**
- Scottish Highlands
- Trossachs
- Winter landscape
- Mountain peaks
- Birch trees
- Snow-covered
- Golden light
- Stirling, Scotland
- Natural beauty
- Winter scenery
- Minimalist landscape
- Snowy glen
- Low-angle sunlight
- Winter solitude

This caption and description are tailored to the visual elements and the specific context provided.
```

---

### ✅ mlx-community/paligemma2-3b-ft-docci-448-bf16

**Metrics:** 18 TPS | 113 tokens

```text
A vertical, long shot of a mountain range in the Scottish Highlands. The mountain range is in the center of the frame, with the lower half obscured by the foreground. The upper half of the frame is the sky, which is a light blue with a hint of white. The sky is mostly covered in clouds. The sun is setting on the mountain, creating a golden glow on the lower half of the mountain. The lower half of the frame is a field of leafless trees with a dusting of snow. The foreground is the lower half of the frame.
```

---

### ✅ mlx-community/llava-v1.6-mistral-7b-8bit

**Metrics:** 48.5 TPS | 8 tokens

```text
The image is a photograph.
```

---

### ✅ mlx-community/gemma-3n-E4B-it-bf16

**Metrics:** 41 TPS | 460 tokens

```text
## Caption:

A low sun casts a warm, golden light across a mountain peak in the Trossachs, Stirling, Scotland. Below, the glen remains in cool shadow, where the white trunks of leafless birch trees stand out against the dark winter vegetation and lingering patches of snow. This striking contrast between light and shadow highlights the rugged beauty of the Scottish landscape. The Trossachs region, often called 'The Highlands in Miniature,' has a rich history and was famously romanticized by the writings of Sir Walter Scott, who helped define the enduring image of the Highlands.

## Visual Content:

The image captures a winter scene in the Trossachs region of Scotland. The focal point is a prominent mountain peak bathed in the warm, golden light of a low sun, likely during sunrise or sunset. The mountain's slopes are rocky and textured, with patches of brown and hints of snow.

In the foreground and midground, a glen stretches out, largely in shadow. Numerous slender, white-barked birch trees stand starkly against the darker backdrop. Their branches are bare, indicative of winter. Patches of snow are visible on the ground, particularly in the shaded areas.

The overall color palette is cool, dominated by dark blues and browns in the shadows, contrasted by the warm golden hues on the illuminated mountain. The sky visible above the mountain is a pale blue with some wispy clouds. The composition emphasizes the vastness of the landscape and the dramatic interplay of light and shadow.

## Contextual Information:

The image depicts the Trossachs, a region in Stirling, Scotland, often referred to as "The Highlands in Miniature" due to its diverse landscapes. The presence of mountains, glens, and birch woodlands is characteristic of this area. The winter setting, with leafless trees and snow patches, is typical of the Scottish climate during this season.

The mention of Sir Walter Scott is significant as he played a crucial role in popularizing the romantic image of the Scottish Highlands through his novels, which often featured landscapes similar to the Trossachs. This historical context adds a layer of cultural significance to the visual representation of the region. The image effectively conveys the rugged beauty and atmospheric quality often associated with the Scottish Highlands.
```

---

### ✅ mlx-community/Idefics3-8B-Llama3-bf16

**Metrics:** 29.2 TPS | 281 tokens

```text
 The image depicts a serene winter landscape in the Trossachs, Stirling, Scotland. The scene is characterized by a low sun casting a warm, golden light across a mountain peak in the distance. The mountain is rugged and covered with a mix of brown and green vegetation, indicating a typical Scottish landscape. The lower part of the image shows a glen, which remains in cool shadow, contrasting with the light on the mountain. The glen is dotted with leafless birch trees, their white trunks standing out against the dark winter vegetation and patches of snow. The contrast between light and shadow highlights the rugged beauty of the Scottish landscape in winter.

The Trossachs region, often referred to as 'The Highlands in Miniature,' has a rich history and cultural significance. The area has been romanticized by the writings of Sir Walter Scott, which have contributed to the enduring image of the Highlands. The image captures the essence of this romanticized landscape, with its dramatic lighting and stark contrasts.

The photograph was taken on January 3, 2026, at 14:57:54 GMT. The time of day and the low angle of the sun suggest a late afternoon or early evening setting, which is typical for capturing such dramatic lighting effects. The image provides a snapshot of the beauty and tranquility of the Scottish winter landscape, highlighting the region's natural beauty and cultural heritage.<end_of_utterance>
```

---

### ✅ mlx-community/gemma-3-27b-it-qat-4bit

**Metrics:** 27.4 TPS | 300 tokens

```text
## Image Catalogue Information:

**Caption:** Mountain peak illuminated by low afternoon sunlight in the Trossachs, Stirling, Scotland, January 3rd, 2026. The foreground glen is in shadow, featuring birch trees and patches of snow.

**Description:** A landscape photograph depicting a mountain peak partially illuminated by warm, golden light. The lower portion of the image shows a glen in shadow, with the white bark of leafless birch trees prominently visible against darker winter vegetation. Patches of snow are present in the glen. The photograph was taken in the Trossachs region of Stirling, Scotland, on January 3rd, 2026, at approximately 14:57:54 GMT. The image showcases a strong contrast between light and shadow.

**Keywords:**

*   Trossachs
*   Stirling
*   Scotland
*   Scottish Highlands
*   Mountain
*   Landscape
*   Winter
*   January
*   Afternoon Light
*   Birch Trees
*   Snow
*   Glen
*   Light and Shadow
*   UK
*   United Kingdom
*   2026-01-03
*   14:57:54 GMT
*   Natural Landscape
*   Outdoor Photography
*   Trees
*   Vegetation
*   Hill
*   Peak
*   White bark trees
*   Contrast




```

---

### ✅ mlx-community/pixtral-12b-8bit

**Metrics:** 34 TPS | 350 tokens

```text
### Caption
The low sun of a January afternoon casts a warm, golden light across a mountain peak in the Trossachs, Stirling, Scotland. Below, the glen remains in cool shadow, where the white trunks of leafless birch trees stand out against the dark winter vegetation and lingering patches of snow. This striking contrast between light and shadow highlights the rugged beauty of the Scottish landscape in winter. The Trossachs region, often called 'The Highlands in Miniature,' has a rich history and was famously romanticized by the writings of Sir Walter Scott, which helped define the enduring image of the Highlands.

### Description
The image captures a serene winter scene in the Trossachs, Stirling, Scotland. The focal point is a mountain peak bathed in the warm, golden light of the late afternoon sun. The peak's rugged terrain is illuminated, creating a striking contrast with the cool shadows of the glen below. Leafless birch trees, their white trunks stark against the dark winter vegetation, dot the landscape. Patches of snow are scattered across the ground, adding to the wintry atmosphere. The Trossachs, known as 'The Highlands in Miniature,' is celebrated for its natural beauty and historical significance, having been immortalized in the writings of Sir Walter Scott.

### Keywords
- Trossachs
- Stirling
- Scotland
- United Kingdom
- UK
- Winter landscape
- Mountain peak
- Golden light
- Birch trees
- Snow
- Sir Walter Scott
- Highlands in Miniature
- Natural beauty
- Historical significance
- January
- Afternoon sun
- Rugged terrain
- Contrast
- Shadow
- Winter vegetation
```

---

### ✅ mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

**Metrics:** 26.9 TPS | 241 tokens

```text
Caption:
A winter scene in the Trossachs, Stirling, Scotland, showcasing a mountain peak bathed in the golden light of a low January sun, while the glen below remains in cool shadow with leafless birch trees and patches of snow.

Description:
The image captures a winter landscape in the Trossachs region of Stirling, Scotland. A prominent mountain peak is illuminated by the warm, golden light of the low afternoon sun, creating a striking contrast with the cool, shadowed glen below. The glen is filled with leafless birch trees, their white trunks standing out against the dark winter vegetation. Patches of snow are visible on the ground, adding to the wintry atmosphere. The scene highlights the rugged beauty of the Scottish Highlands in winter, with a mix of natural elements that emphasize the region's dramatic and serene landscape.

Keywords:
Trossachs, Stirling, Scotland, UK, winter, mountain, peak, glen, birch trees, snow, golden light, low sun, January, Scottish Highlands, landscape, nature, rugged beauty, leafless trees, shadow, light contrast, winter vegetation, natural scenery.
```

---

### ✅ mlx-community/gemma-3-27b-it-qat-8bit

**Metrics:** 15.2 TPS | 282 tokens

```text
Here's a factual caption, description, and keywords for cataloguing the image, based on your provided information and visual analysis:

**Caption:**

Mountain peak illuminated by low afternoon sunlight in the Trossachs, Stirling, Scotland, January 3, 2026.

**Description:**

A landscape photograph depicting a mountain peak bathed in golden light. The lower slopes and glen are in shadow, revealing a forest of predominantly leafless birch trees with white bark. Patches of snow are visible in the foreground and on the lower slopes. The time of day is late afternoon, indicated by the low angle of the sun. The scene showcases the rugged terrain and winter conditions of the Trossachs region in Scotland.

**Keywords:**

*   Trossachs
*   Stirling
*   Scotland
*   UK
*   United Kingdom
*   Mountain
*   Winter
*   Landscape
*   Birch Trees
*   Snow
*   Glen
*   Afternoon Light
*   January
*   2026
*   Golden Hour
*   Light and Shadow
*   Scottish Highlands
*   Natural Landscape
*   Forest
*   Leafless Trees
*   Outdoor Photography
*   Hill
*   Peak
*   Vegetation
*   White Bark
*   Rugged Terrain




```

---

### ✅ mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX

**Metrics:** 34.1 TPS | 500 tokens

```text
Here are my reasoning steps:
The user asks: "Provide a factual caption, description, and keywords suitable for cataloguing, or searching for, the image. Context: The image relates to 'Trossachs, Stirling, Scotland, United Kingdom, UK' Here is a caption for the image: ... The low sun of a January afternoon casts a warm, golden light across a mountain peak in the Trossachs, Stirling, Scotland. Below, the glen remains in cool shadow, where the white trunks of leafless birch trees stand out against the dark winter vegetation and lingering patches of snow. This striking contrast between light and shadow highlights the rugged beauty of the Scottish landscape in winter. The Trossachs region, often called 'The Highlands in Miniature,' has a rich history and was famously romanticized by the writings of Sir Walter Scott, which helped define the enduring image of the Highlands. The photo was taken around 2026-01-03 14:57:54 GMT . Focus on visual content, drawing on any available contextual information for specificity. Do not speculate."

Thus we need to produce a factual caption, description, and keywords suitable for cataloguing or searching for the image. The user gave a sample caption, but we need to produce a final answer: a factual caption, description, and keywords. The user says "Focus on visual content, drawing on any available contextual information for specificity. Do not speculate." So we need to avoid speculation beyond what is visible or given. The image shows a mountain peak in the Trossachs, low sun casting warm light, birch trees, winter, snow patches, etc. The location is Trossachs, Stirling, Scotland, UK. The date is around 2026-01-03 14:57:54 GMT. So we can incorporate that. The caption should be factual, not overly interpretive. The description should be detailed, describing the visual elements: mountain, lighting, trees, snow, sky, etc. Keywords: Trossachs, Stirling, Scotland, UK, winter, mountain, low sun, birch trees, snow, landscape, glen, contrast, light and shadow, January, 2026, etc.

We need to produce a caption, description, and keywords. The caption can be a short sentence or two summarizing the image. The description can be a longer
```

---

### ✅ mlx-community/pixtral-12b-bf16

**Metrics:** 19.1 TPS | 351 tokens

```text
### Caption
The low sun of a January afternoon casts a warm, golden light across a mountain peak in the Trossachs, Stirling, Scotland. Below, the glen remains in cool shadow, where the white trunks of leafless birch trees stand out against the dark winter vegetation and lingering patches of snow. This striking contrast between light and shadow highlights the rugged beauty of the Scottish landscape in winter. The Trossachs region, often called 'The Highlands in Miniature,' has a rich history and was famously romanticized by the writings of Sir Walter Scott, which helped define the enduring image of the Highlands.

### Description
The image captures a serene winter scene in the Trossachs, Stirling, Scotland. The focal point is a mountain peak bathed in the warm, golden light of the late afternoon sun. The peak's rugged terrain is illuminated, creating a striking contrast with the cool shadows that envelop the glen below. Leafless birch trees, their white trunks stark against the dark winter vegetation, dot the landscape. Patches of snow are scattered across the ground, adding to the wintry atmosphere. The Trossachs, known as 'The Highlands in Miniature,' is celebrated for its natural beauty and historical significance, having been romanticized by Sir Walter Scott's writings.

### Keywords
- Trossachs
- Stirling
- Scotland
- United Kingdom
- UK
- Mountain peak
- Winter landscape
- Birch trees
- Snow
- Golden light
- Sir Walter Scott
- Highlands in Miniature
- Scottish landscape
- January
- Winter beauty
- Natural scenery
- Historical significance
- Romanticized landscape
- Contrast of light and shadow
```

---

### ✅ mlx-community/GLM-4.6V-Flash-6bit

**Metrics:** 44.6 TPS | 500 tokens

```text
<think>Got it, let's tackle this. The user wants a factual caption, description, and keywords for cataloguing/searching.

First, start with the caption. It should be concise but include key elements: location (Trossachs, Stirling, Scotland), time (January afternoon, low sun), visual elements (golden light on mountain peak, cool shadow in glen, leafless birch trees with white trunks, winter vegetation, patches of snow).

Then the description: need to elaborate on the visual content, structure, and context. Mention the mountain peak (likely a specific mountain in Trossachs, but since the context says Trossachs, maybe not need to name the peak unless known, but the user didn't specify, so keep general). The contrast between light and shadow, the type of trees (birch, leafless), the winter landscape with snow patches.

Keywords: Need to include location terms (Trossachs, Stirling, Scotland, UK), time-related (January, afternoon, low sun), visual elements (mountain peak, golden light, cool shadow, birch trees, leafless, winter vegetation, snow patches, rugged landscape, contrast light shadow).

Now, structure the caption, description, and keywords clearly. Make sure all information is factual, based on the provided context and visual content.</think>
<|begin_of_box|>### Caption  
Low sun of a January afternoon casts warm golden light on a mountain peak in the Trossachs, Stirling, Scotland. Below, the glen remains in cool shadow, with leafless birch trees (white trunks) standing out against dark winter vegetation and lingering snow patches.  

### Description  
The image depicts a winter landscape in the Trossachs region of Stirling, Scotland. A prominent mountain peak dominates the upper portion of the frame, bathed in the warm, golden hue of low afternoon sunlight. This light creates a striking contrast with the lower glen, which remains in cool shadow.  

In the foreground and midground, leafless birch trees with distinctive white trunks are scattered across the landscape. The ground is covered with dark winter vegetation, interspersed with patches of lingering snow, particularly visible at the bottom of the frame.  

The overall composition emphasizes the rugged beauty of the Scottish landscape in winter, with the interplay of light and shadow, and the contrast between the sunlit mountain and the shadowed glen.  

### Keywords  
Trossachs; Stirling; Scotland
```

---

### ✅ mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

**Metrics:** 7.37 TPS | 243 tokens

```text
I'm happy to help you with your question. However, I want to clarify that I'm a large language model, I don't have the capability to access real-time information or browse the internet. I can only provide information based on my training data, which may not be up-to-date or accurate.

That being said, I can try to provide a general response to your question. It seems like you're asking for a factual caption, description, and keywords for an image of a mountainous landscape. If I had to provide a response, I would say that the image is a serene and peaceful representation of a mountainous landscape, with a focus on the natural beauty of the scene.

However, I want to emphasize that this is just a general response, and I may not have the most accurate or up-to-date information. If you're looking for a more specific or detailed response, I would recommend searching for more information on the internet or consulting with an expert in the field.

Additionally, I want to note that the image you provided is not visible in this text-based interface. If you could provide more context or information about the image, I may be able to provide a more specific and accurate response.
```

---

### ✅ mlx-community/Molmo-7B-D-0924-8bit

**Metrics:** 43.5 TPS | 126 tokens

```text
 Here is a caption for the image:

"The low sun of a January afternoon illuminates a mountain peak in the Trossachs, Stirling, Scotland. Below, the glen remains in cool shadow, where the white trunks of leafless birch trees stand out against the dark winter vegetation and lingering patches of snow. This striking contrast between light and shadow highlights the rugged beauty of the Scottish landscape in winter. The Trossachs region, often called 'The Highlands in Miniature,' has a rich history and was famously romanticized by the writings of Sir Walter Scott, which helped define the enduring image of the Highlands.'
```

---

### ✅ mlx-community/Molmo-7B-D-0924-bf16

**Metrics:** 22.8 TPS | 172 tokens

```text
 Here is a caption for the image:

"The low sun of a January afternoon illuminates a mountain peak in the Trossachs, Stirling, Scotland. Below, the glen remains in cool shadow, where the white trunks of leafless birch trees stand out against the dark winter vegetation and lingering patches of snow. This striking contrast between light and shadow highlights the rugged beauty of the Scottish landscape in winter. The Trossachs region, often called 'The Highlands in Miniature,' has a rich history and was famously romanticized by the writings of Sir Walter Scott, which helped define the enduring image of the Highlands.'

The photo was taken around 2026-01-03 14:57:54 GMT. Focus on visual content, drawing on any available contextual information for specificity. Do not speculate.'
```

---

### ✅ microsoft/Phi-3.5-vision-instruct

**Metrics:** 10.9 TPS | 500 tokens

```text
Caption: "Winter's Embrace in the Trossachs, Stirling, Scotland"

Description: This image captures the serene beauty of the Trossachs region in Stirling, Scotland, during the winter season. The low winter sun bathes a mountain peak in a warm, golden hue, creating a striking contrast with the cool, shadowed glen below. Leafless birch trees stand as silent sentinels in the foreground, their white trunks stark against the dark winter foliage and patches of snow. The scene is a testament to the rugged and enduring beauty of the Scottish Highlands, a landscape that has inspired countless works of literature and art.

Keywords: Trossachs, Stirling, Scotland, United Kingdom, UK, winter, mountain peak, birch trees, golden light, landscape, Highlands, Sir Walter Scott, literature, art, Scottish Highlands, natural beauty, contrast, serene, enduring.<|end|><|endoftext|>
# INSTRUCTION



# SOLUTION


# INSTRUCTION


# SOLUTION


# INSTRUCTION


# SOLUTION


# INSTRUCTION


# SOLUTION


# INSTRUCTION


# SOLUTION


# INSTRUCTION


# SOLUTION


# INSTRUCTION


# SOLUTION


# INSTRUCTION


# SOLUTION


# INSTRUCTION


# SOLUTION


# INSTRUCTION


# SOLUTION


# INSTRUCTION


# SOLUTION


# INSTRUCTION


# SOLUTION


# INSTRUCTION


# SOLUTION


# INSTRUCTION


# SOLUTION


# INSTRUCTION


# SOLUTION


# INSTRUCTION


# SOLUTION


# INSTRUCTION


# SOLUTION


# INSTRUCTION



```

---

### ✅ Qwen/Qwen3-VL-2B-Instruct

**Metrics:** 69.6 TPS | 500 tokens

```text
### Factual Caption
A serene winter scene in the Trossachs, Stirling, Scotland, captures the dramatic interplay of light and shadow. The low sun of a January afternoon bathes the rugged mountain peak in a warm, golden glow, contrasting sharply with the cool, shadowed valley below. The foreground is filled with the stark, white trunks of leafless birch trees, which stand out against the dark, dense winter vegetation and scattered patches of snow. This image beautifully illustrates the rugged and picturesque landscape of the Scottish Highlands, a region renowned for its natural beauty and historical significance.

### Description
The photograph depicts a mountainous landscape in the Trossachs, a region in Stirling, Scotland, during winter. The central focus is a prominent, rocky mountain peak that is illuminated by the low sun, casting a warm golden light across its slopes. The lower slopes and valleys are shrouded in cool, shadowed tones, creating a striking contrast. The foreground is dominated by the bare, white trunks of birch trees, which are silhouetted against the darker, more muted vegetation and snow. The overall scene is serene and atmospheric, capturing the quiet beauty of the Scottish Highlands in winter.

### Keywords
- Trossachs
- Stirling
- Scotland
- United Kingdom
- UK
- Mountain
- Winter
- Golden light
- Low sun
- Birches
- Snow
- Landscape
- Highlands
- Sir Walter Scott
- Natural beauty
- Serene
- Photographic
- Landscape photography
- Winter landscape
- Scottish Highlands
- Mountain peak
- Tree
- Shadow
- Light and shadow
- Nature
- Winter scene
- Glacial landscape
- Mountain view
- Snow-covered
- Bare trees
- Golden hour
- Landscape photography
- Nature photography
- Winter photography
- Scottish Highlands
- Mountain landscape
- Winter landscape photography
- Nature photography
- Winter scene photography
- Scottish Highlands photography
- Mountain photography
- Winter landscape photography
- Nature photography
- Winter scene photography
- Scottish Highlands photography
- Mountain photography
- Winter landscape photography
- Nature photography
- Winter scene photography
- Scottish Highlands photography
- Mountain photography
- Winter landscape photography
- Nature photography
- Winter scene photography
- Scottish Highlands photography
- Mountain photography
- Winter landscape photography
- Nature photography
- Winter scene photography
- Scottish Highlands photography
- Mountain
```

---

### ✅ mlx-community/Qwen3-VL-2B-Thinking-bf16

**Metrics:** 70.4 TPS | 500 tokens

```text
Got it, let's break this down. First, the caption given is a bit long, but we need to create a factual caption, description, and keywords. Let's start with the factual caption. The context says it's Trossachs, Stirling, Scotland, UK. The caption mentions January afternoon, low sun, golden light on a mountain peak, glen in cool shadow, leafless birch trees, snow patches. So we need to make sure it's factual, not speculative. The photo was taken around 2026-01-03, but since it's a description, we can note the date as part of the context.

First, factual caption: Should be concise, include location, time, elements. Let's check the original caption: "The low sun of a January afternoon casts a warm, golden light across a mountain peak in the Trossachs, Stirling, Scotland. Below, the glen remains in cool shadow, where the white trunks of leafless birch trees stand out against the dark winter vegetation and lingering patches of snow. This striking contrast between light and shadow highlights the rugged beauty of the Scottish landscape in winter. The Trossachs region, often called 'The Highlands in Miniature,' has a rich history and was famously romanticized by the writings of Sir Walter Scott, which helped define the enduring image of the Highlands."

Wait, the user says "Do not speculate. Focus on visual content, drawing on any available contextual information for specificity." So we need to stick to what's in the caption. But the caption is already provided, so maybe we need to refine it to be factual. Wait, the user says "Here is a caption for the image" but then says "Provide a factual caption, description, and keywords suitable for cataloguing, or searching for, the image." So the caption given is the one they provided, but we need to make sure it's factual. Wait, the context says the image is in Trossachs, Stirling, Scotland, UK. The caption mentions that. So the factual caption should be based on the context and the caption provided.

Wait, the user says "Context: The image relates to 'Trossachs, Stirling, Scotland, United Kingdom, UK'". So the location is fixed. The caption they provided is a description, but we need to create a factual caption (maybe a more precise one) and a description, and keywords.

Wait
```

---

### ✅ mlx-community/paligemma2-10b-ft-docci-448-bf16

**Metrics:** 4.91 TPS | 500 tokens

```text
A long-shot view of a mountain in Scotland. The mountain is in the middle, and it is lit up by the sun. The sun is hitting the top of the mountain, and the side of the mountain is covered in shadow. The sun is hitting the top of the mountain, and the side of the mountain is covered in shadow. The sun is hitting the top of the mountain, and the side of the mountain is covered in shadow. The sun is hitting the top of the mountain, and the side of the mountain is covered in shadow. The sun is hitting the top of the mountain, and the side of the mountain is covered in shadow. The sun is hitting the top of the mountain, and the side of the mountain is covered in shadow. The sun is hitting the top of the mountain, and the side of the mountain is covered in shadow. The sun is hitting the top of the mountain, and the side of the mountain is covered in shadow. The sun is hitting the top of the mountain, and the side of the mountain is covered in shadow. The sun is hitting the top of the mountain, and the side of the mountain is covered in shadow. The sun is hitting the top of the mountain, and the side of the mountain is covered in shadow. The sun is hitting the top of the mountain, and the side of the mountain is covered in shadow. The sun is hitting the top of the mountain, and the side of the mountain is covered in shadow. The sun is hitting the top of the mountain, and the side of the mountain is covered in shadow. The sun is hitting the top of the mountain, and the side of the mountain is covered in shadow. The sun is hitting the top of the mountain, and the side of the mountain is covered in shadow. The sun is hitting the top of the mountain, and the side of the mountain is covered in shadow. The sun is hitting the top of the mountain, and the side of the mountain is covered in shadow. The sun is hitting the top of the mountain, and the side of the mountain is covered in shadow. The sun is hitting the top of the mountain, and the side of the mountain is covered in shadow. The sun is hitting the top of the mountain, and the side of the mountain is covered in shadow. The sun is hitting the top of the mountain, and the side of the mountain is covered in shadow. The sun is hitting the top of the mountain, and
```

---

### ✅ meta-llama/Llama-3.2-11B-Vision-Instruct

**Metrics:** 3.84 TPS | 500 tokens

```text
Here is a factual caption, description, and keywords for the image:

**Image Title:** Winter Landscape in the Trossachs, Scotland

**Description:** A serene winter landscape in the Trossachs, Scotland, featuring a mountain peak in the background and a forest of bare trees in the foreground.

**Keywords:** Winter, Landscape, Trossachs, Scotland, Mountain, Trees, Nature, Scenic, Outdoor, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons, Seasons,
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
- `huggingface-hub`: `1.3.1`
- `mlx`: `0.30.3.dev20260110+15968392`
- `mlx-lm`: `0.30.2`
- `mlx-vlm`: `0.3.10`
- `numpy`: `2.4.1`
- `tokenizers`: `0.22.2`
- `transformers`: `5.0.0rc1`

_Report generated on: 2026-01-10 22:26:56 GMT_
