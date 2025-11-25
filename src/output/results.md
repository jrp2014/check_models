# Model Performance Results

_Generated on 2025-11-25 16:28:07 GMT_

## 🏆 Performance Highlights

- **Fastest:** `prince-canuma/Florence-2-large-ft` (328.3 tps)
- **💾 Most efficient:** `qnguyen3/nanoLLaVA` (4.5 GB)
- **⚡ Fastest load:** `HuggingFaceTB/SmolVLM-Instruct` (0.00s)
- **📊 Average TPS:** 66.9 across 26 models

## 📈 Resource Usage

- **Total peak memory:** 451.1 GB
- **Average peak memory:** 17.3 GB
- **Memory efficiency:** 163 tokens/GB

## ⚠️ Quality Issues

- **❌ Failed Models (7):**
  - `microsoft/Florence-2-large-ft` (`Model Error`)
  - `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16` (`Lib Version`)
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit` (`Lib Version`)
  - `mlx-community/Molmo-7B-D-0924-8bit` (`Missing Dep`)
  - `mlx-community/Molmo-7B-D-0924-bf16` (`Missing Dep`)
  - `mlx-community/Qwen2-VL-2B-Instruct-4bit` (`OOM`)
  - `mlx-community/gemma-3-12b-pt-8bit` (`Error`)
- **🔄 Repetitive Output (1):**
  - `mlx-community/paligemma2-10b-ft-docci-448-6bit` (token: `phrase: "the tiles are wet..."`)
- **📝 Formatting Issues (2):**
  - `mlx-community/Idefics3-8B-Llama3-bf16`
  - `prince-canuma/Florence-2-large-ft`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 66.9 | Min: 0 | Max: 328
- **Peak Memory**: Avg: 17 | Min: 4.5 | Max: 51
- **Total Time**: Avg: 21.81s | Min: 2.39s | Max: 73.27s
- **Generation Time**: Avg: 19.27s | Min: 1.38s | Max: 69.41s
- **Model Load Time**: Avg: 2.54s | Min: 0.79s | Max: 5.96s

> **Prompt used:**
>
> Provide a factual caption, description, and keywords suitable for cataloguing, or searching for, the image.
>
> Context: The image relates to ', White Rock, Hastings, England, United Kingdom, UK
> Here is a caption for the image, written for the specified audience and context:
>
> A damp, late autumn afternoon along the seafront at White Rock in Hastings, England. The low-angle perspective highlights the rain-slicked paving stones, which reflect the cool, overcast sky as dusk begins to fall. In the background, a row of hotels and apartments, typical of a British seaside town, lines the road. The motion blur of a passing car's taillights adds a streak of colour and a sense of movement to the otherwise quiet, atmospheric scene. This area was developed significantly during the Victorian era as Hastings grew into a popular coastal resort, and this image captures a modern, tranquil moment outside the peak tourist season.'
>
> The photo was taken around 2025-11-15 16:19:28 GMT . Focus on visual content, drawing on any available contextual information for specificity. Do not speculate.

**Note:** Results sorted: errors first, then by generation time (fastest to slowest).

**Overall runtime:** 653.24s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                         |   Token |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                   |
|:---------------------------------------------------|--------:|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:---------------------------------|
| `microsoft/Florence-2-large-ft`                    |         |                   |                       |                |              |           |             |                  |            |             |                                  |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`     |         |                   |                       |                |              |           |             |                  |            |             |                                  |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`          |         |                   |                       |                |              |           |             |                  |            |             |                                  |
| `mlx-community/Molmo-7B-D-0924-8bit`               |         |                   |                       |                |              |           |             |                  |            |             |                                  |
| `mlx-community/Molmo-7B-D-0924-bf16`               |         |                   |                       |                |              |           |             |                  |            |             |                                  |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`          |         |                   |                       |                |              |           |             |                  |            |             |                                  |
| `mlx-community/gemma-3-12b-pt-8bit`                |         |                   |                       |                |              |           |             |                  |            |             |                                  |
| `qnguyen3/nanoLLaVA`                               | 151,645 |               249 |                    84 |            333 |        1,195 |       102 |         4.5 |            1.38s |      1.01s |       2.39s |                                  |
| `HuggingFaceTB/SmolVLM-Instruct`                   |  49,154 |             1,430 |                     9 |          1,439 |        1,215 |       131 |         5.5 |            1.71s |      1.26s |       2.97s |                                  |
| `mlx-community/SmolVLM-Instruct-bf16`              |  49,154 |             1,430 |                     9 |          1,439 |        1,138 |       130 |         5.5 |            1.79s |      1.11s |       2.90s |                                  |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`         |  49,279 |             1,330 |                    26 |          1,356 |        1,155 |       122 |         5.6 |            1.86s |      1.12s |       2.98s | context-ignored                  |
| `prince-canuma/Florence-2-large-ft`                |     0.0 |               233 |                   500 |            733 |          983 |       328 |         5.1 |            2.12s |      1.13s |       3.25s | lang_mixing, formatting, ...     |
| `mlx-community/deepseek-vl2-8bit`                  |         |               0.0 |                   0.0 |            0.0 |            0 |         0 |          32 |            2.26s |      5.00s |       7.26s |                                  |
| `mlx-community/LFM2-VL-1.6B-8bit`                  |       7 |             2,029 |                   141 |          2,170 |        1,147 |       290 |          51 |            2.77s |      0.79s |       3.56s |                                  |
| `mlx-community/paligemma2-3b-pt-896-4bit`          |       1 |             4,334 |                    14 |          4,348 |        1,134 |       118 |         8.8 |            4.29s |      1.67s |       5.97s | context-ignored                  |
| `mlx-community/gemma-3n-E4B-it-bf16`               |     106 |               504 |                   168 |            672 |          707 |      42.0 |          18 |            4.98s |      3.96s |       8.94s |                                  |
| `mlx-community/llava-v1.6-mistral-7b-8bit`         |       2 |             2,321 |                     8 |          2,329 |          234 |      47.9 |          12 |           10.57s |      1.66s |      12.23s | context-ignored                  |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`    |       1 |             1,262 |                   177 |          1,439 |        1,484 |      17.9 |          11 |           11.09s |      2.29s |      13.37s | context-ignored                  |
| `mlx-community/Idefics3-8B-Llama3-bf16`            | 128,001 |             2,547 |                   227 |          2,774 |          678 |      28.9 |          19 |           12.09s |      3.23s |      15.32s | formatting                       |
| `mlx-community/pixtral-12b-8bit`                   |       2 |             2,842 |                   291 |          3,133 |          497 |      34.3 |          15 |           14.57s |      2.79s |      17.36s | bullets(17)                      |
| `mlx-community/gemma-3-27b-it-qat-4bit`            |     106 |               504 |                   308 |            812 |          155 |      26.2 |          18 |           15.30s |      3.95s |      19.25s | bullets(31)                      |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`   |     714 |             1,262 |                   500 |          1,762 |          505 |      39.7 |          11 |           15.42s |      2.61s |      18.03s | repetitive(phrase: "the tiles... |
| `mlx-community/InternVL3-14B-8bit`                 | 151,645 |             2,032 |                   283 |          2,315 |          372 |      28.5 |          18 |           15.79s |      3.00s |      18.79s | bullets(18)                      |
| `mlx-community/pixtral-12b-bf16`                   |       2 |             2,842 |                   296 |          3,138 |          538 |      19.2 |          27 |           21.04s |      4.47s |      25.50s | bullets(18)                      |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`    |   5,315 |             2,933 |                   500 |          3,433 |          408 |      35.6 |          15 |           21.59s |      2.69s |      24.28s |                                  |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit` | 128,009 |               236 |                   171 |            407 |         82.8 |      9.21 |          15 |           21.74s |      2.48s |      24.22s | context-ignored                  |
| `mlx-community/Phi-3.5-vision-instruct-bf16`       |  32,007 |             1,042 |                   262 |          1,304 |          799 |      10.8 |          33 |           25.70s |      1.77s |      27.47s |                                  |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`   |       1 |             1,262 |                   115 |          1,377 |          470 |      5.02 |          27 |           25.95s |      4.26s |      30.21s | context-ignored                  |
| `mlx-community/gemma-3-27b-it-qat-8bit`            |     139 |               504 |                   500 |          1,004 |          163 |      15.0 |          32 |           36.76s |      5.96s |      42.72s | bullets(60)                      |
| `microsoft/Phi-3.5-vision-instruct`                |  29,892 |             1,042 |                   500 |          1,542 |          916 |      11.0 |          12 |           46.77s |      1.79s |      48.56s | lang_mixing                      |
| `Qwen/Qwen3-VL-2B-Instruct`                        | 151,645 |            16,503 |                   433 |         16,936 |          365 |      71.7 |          13 |           51.62s |      0.81s |      52.43s | verbose, bullets(27)             |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`          |      11 |            16,505 |                   500 |         17,005 |          299 |      71.5 |          13 |           62.49s |      1.31s |      63.81s | verbose                          |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`         | 128,009 |               237 |                   254 |            491 |         76.7 |      3.85 |          25 |           69.41s |      3.86s |      73.27s |                                  |

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

### ❌ mlx-community/Molmo-7B-D-0924-8bit

**Status:** Failed (Missing Dep)
**Error:**

> Model loading failed: This modeling file requires the following packages
> that were not found in your environment: tensorflow. Run `pip install
> tensorflow`
**Type:** `ValueError`

---

### ❌ mlx-community/Molmo-7B-D-0924-bf16

**Status:** Failed (Missing Dep)
**Error:**

> Model loading failed: This modeling file requires the following packages
> that were not found in your environment: tensorflow. Run `pip install
> tensorflow`
**Type:** `ValueError`

---

### ❌ mlx-community/Qwen2-VL-2B-Instruct-4bit

**Status:** Failed (OOM)
**Error:**

> Model runtime error during generation for
> mlx-community/Qwen2-VL-2B-Instruct-4bit: [metal::malloc] Attempting to
> allocate 135383101952 bytes which is greater than the maximum allowed buffer
> size of 86586540032 bytes.
**Type:** `ValueError`

---

### ❌ mlx-community/gemma-3-12b-pt-8bit

**Status:** Failed (Error)
**Error:**

> Cannot use apply_chat_template because this processor does not have a chat
> template.
**Type:** `ValueError`

---

### ✅ qnguyen3/nanoLLaVA

**Metrics:** 102 TPS | 84 tokens

```text
A rainy day in White Rock, Hastings, England, United Kingdom, UK. The image captures a street scene with a low-angle perspective, showing the wet pavement and the overcast sky. The street is lined with hotels and apartments, and the road is wet from the rain. The image was taken on 2025-11-15 16:19:28 GMT.
```

---

### ✅ HuggingFaceTB/SmolVLM-Instruct

**Metrics:** 131 TPS | 9 tokens

```text
 A rainy day in Hastings, England.
```

---

### ✅ mlx-community/SmolVLM-Instruct-bf16

**Metrics:** 130 TPS | 9 tokens

```text
 A rainy day in Hastings, England.
```

---

### ✅ mlx-community/SmolVLM2-2.2B-Instruct-mlx

**Metrics:** 122 TPS | 26 tokens

```text
 A wet pavement with three large concrete blocks in the foreground. In the background, there are white buildings and cars passing by.
```

---

### ✅ prince-canuma/Florence-2-large-ft

**Metrics:** 328 TPS | 500 tokens

```text
<s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s>
```

---

### ✅ mlx-community/deepseek-vl2-8bit

**Metrics:** 0.0 TPS | 0 tokens

```text

```

---

### ✅ mlx-community/LFM2-VL-1.6B-8bit

**Metrics:** 290 TPS | 141 tokens

```text
A damp, late autumn afternoon along the seafront at White Rock in Hastings, England, United Kingdom, United Kingdom. The low-angle perspective highlights the rain-slicked paving stones, which reflect the cool, overcast sky as dusk begins to fall. In the background, a row of hotels and apartments, typical of a British seaside town, lines the road. The motion blur of a passing car's taillights adds a streak of colour and a sense of movement to the otherwise quiet, atmospheric scene. This area was developed significantly during the Victorian era as Hastings grew into a popular coastal resort, and this image captures a modern, tranquil moment outside the peak tourist season.
```

---

### ✅ mlx-community/paligemma2-3b-pt-896-4bit

**Metrics:** 118 TPS | 14 tokens

```text
Provide a factual description, or a keyword search for the image.
```

---

### ✅ mlx-community/gemma-3n-E4B-it-bf16

**Metrics:** 42 TPS | 168 tokens

```text
## Caption:

Rain-slicked pavement reflects the soft glow of streetlights and distant car headlights along the promenade at White Rock, Hastings, UK, as dusk settles over the British seaside town.

## Context:

This image captures a typical evening scene on White Rock, a prominent promenade in Hastings, England. The architecture of the buildings lining the street is characteristic of a traditional British seaside resort. The wet pavement and muted lighting evoke a sense of quiet atmosphere as the day transitions into night. The presence of parked cars and streetlights further reinforces the location as a public space within a town.

## Keywords:

White Rock, Hastings, England, UK, promenade, seaside, dusk, evening, rain, streetlights, buildings, British, town, urban, pavement, reflection, architecture, cloudy, overcast.
```

---

### ✅ mlx-community/llava-v1.6-mistral-7b-8bit

**Metrics:** 47.9 TPS | 8 tokens

```text
The image is a photograph.
```

---

### ✅ mlx-community/paligemma2-3b-ft-docci-448-bf16

**Metrics:** 17.9 TPS | 177 tokens

```text
A low-angle, close up view of a row of concrete planters with a square pattern. The planters are in the center of the frame, with the top of the row of planters in the background. The planters are wet from the rain, with a green growth on the top of the left and right planters. The right planter is the only one that is not wet. The left and center planters have a slight view of the top of a black car in the background. The car is facing to the right. The background is a row of white and gray apartment buildings. The buildings are in the center of the frame, with the right side of the frame showing the back of a black car. The left side of the frame shows a gray wooden fence with a row of palm plants in the center. The sky is a light blue, overcast sky with a hint of gray.
```

---

### ✅ mlx-community/Idefics3-8B-Llama3-bf16

**Metrics:** 28.9 TPS | 227 tokens

```text
 The image depicts a tranquil, slightly overcast late afternoon scene along the seafront at White Rock in Hastings, England. The perspective is low, emphasizing the wet, reflective quality of the paving stones, which are glistening from recent rainfall. The sky above is a muted grey, suggesting a typical British weather pattern, and the lighting indicates the onset of dusk.

In the background, a row of hotels and apartments, characteristic of a traditional British seaside town, lines the road. These buildings are predominantly white, with some darker accents, and their windows reflect the ambient light, adding to the serene atmosphere. The motion blur of a passing car's taillights introduces a dynamic element to the otherwise still scene, hinting at the presence of life and movement in the area.

The area was developed during the Victorian era as Hastings grew into a popular coastal resort, and this image captures a quiet, off-peak moment, devoid of the bustling activity that might be expected during peak tourist season. The overall mood of the image is calm and reflective, with the wet pavement and overcast sky contributing to a subdued yet picturesque setting.<end_of_utterance>
```

---

### ✅ mlx-community/pixtral-12b-8bit

**Metrics:** 34.3 TPS | 291 tokens

```text
### Caption
A damp, late autumn afternoon along the seafront at White Rock in Hastings, England. The low-angle perspective highlights the rain-slicked paving stones, which reflect the cool, overcast sky as dusk begins to fall. In the background, a row of hotels and apartments, typical of a British seaside town, lines the road. The motion blur of a passing car's taillights adds a streak of colour and a sense of movement to the otherwise quiet, atmospheric scene.

### Description
The image captures a tranquil scene at White Rock in Hastings, England, during the late autumn season. The ground is wet from recent rain, creating reflective surfaces on the paving stones. The perspective is from a low angle, emphasizing the wet ground and the overcast sky. In the background, a row of white buildings, likely hotels or apartments, lines the street. These buildings are characteristic of British seaside architecture. A passing car's taillights are captured in motion blur, adding a dynamic element to the otherwise serene setting.

### Keywords
- White Rock
- Hastings
- England
- United Kingdom
- Seafront
- Autumn
- Rain
- Reflective Paving Stones
- Overcast Sky
- Hotels
- Apartments
- British Seaside Architecture
- Taillights
- Motion Blur
- Tranquil Scene
- Victorian Era
- Coastal Resort
```

---

### ✅ mlx-community/gemma-3-27b-it-qat-4bit

**Metrics:** 26.2 TPS | 308 tokens

```text
## Image Catalogue Information:

**Caption:** Wet paving stones and concrete planters along the seafront promenade at White Rock, Hastings, England, with hotels and apartments visible in the background. The image was taken on a damp afternoon in November 2025.

**Description:** A low-angle, medium-shot photograph depicting the wet, tiled pavement and concrete planters along the seafront at White Rock, Hastings, England. The paving stones reflect the overcast sky. A row of predominantly white and light-coloured hotels and apartments forms the backdrop. Motion blur from passing vehicle taillights is visible on the road. The foreground features square concrete planters with visible moss growth. Streetlights are visible in the distance. The scene conveys a quiet, atmospheric mood.

**Keywords:**

*   White Rock
*   Hastings
*   England
*   UK
*   United Kingdom
*   Seaside
*   Seafront
*   Promenade
*   November
*   Autumn
*   Rain
*   Wet Pavement
*   Hotels
*   Apartments
*   Architecture
*   Street Furniture
*   Concrete Planters
*   Streetlights
*   Motion Blur
*   Overcast
*   Dusk
*   2025
*   GMT
*   Coastal Town
*   Urban Landscape
*   Exterior
*   Low Angle
*   Reflections
*   Moss
*   Paving Stones
*   British Seaside Resort




```

---

### ✅ mlx-community/paligemma2-10b-ft-docci-448-6bit

**Metrics:** 39.7 TPS | 500 tokens

```text
The image is in the daytime, with the sky visible in the top left corner. The sky is overcast, with a light blue hue. The scene is well lit, with the buildings in the background being the brightest. The foreground is a dark gray, wet, square-tiled surface. The tiles are wet from a recent rainfall. The tiles are a dark gray, with a slight sheen from the water. The tiles are wet from a recent rainfall. The tiles are a dark gray, with a slight sheen from the water. The tiles are wet from a recent rainfall. The tiles are a dark gray, with a slight sheen from the water. The tiles are wet from a recent rainfall. The tiles are a dark gray, with a slight sheen from the water. The tiles are wet from a recent rainfall. The tiles are a dark gray, with a slight sheen from the water. The tiles are wet from a recent rainfall. The tiles are a dark gray, with a slight sheen from the water. The tiles are wet from a recent rainfall. The tiles are a dark gray, with a slight sheen from the water. The tiles are wet from a recent rainfall. The tiles are a dark gray, with a slight sheen from the water. The tiles are wet from a recent rainfall. The tiles are a dark gray, with a slight sheen from the water. The tiles are wet from a recent rainfall. The tiles are a dark gray, with a slight sheen from the water. The tiles are wet from a recent rainfall. The tiles are a dark gray, with a slight sheen from the water. The tiles are wet from a recent rainfall. The tiles are a dark gray, with a slight sheen from the water. The tiles are wet from a recent rainfall. The tiles are a dark gray, with a slight sheen from the water. The tiles are wet from a recent rainfall. The tiles are a dark gray, with a slight sheen from the water. The tiles are wet from a recent rainfall. The tiles are a dark gray, with a slight sheen from the water. The tiles are wet from a recent rainfall. The tiles are a dark gray, with a slight sheen from the water. The tiles are wet from a recent rainfall. The tiles are a dark gray, with a slight sheen from the water. The tiles are wet from a recent rainfall. The tiles are a dark gray, with a slight sheen from the water. The tiles are wet from a recent rainfall. The
```

---

### ✅ mlx-community/InternVL3-14B-8bit

**Metrics:** 28.5 TPS | 283 tokens

```text
**Caption:**
A tranquil scene at White Rock, Hastings, England, captured on a damp, late afternoon. The wet paving stones reflect the overcast sky as dusk approaches. In the background, a row of hotels and apartments typical of a British seaside town lines the road, with the motion blur of passing cars adding a touch of movement. This area, developed during the Victorian era, exudes a quiet, atmospheric charm outside the peak tourist season.

**Description:**
The image depicts a serene seaside promenade at White Rock in Hastings, England. The foreground features large, weathered concrete planters on a wet, reflective tiled surface. The background showcases a row of multi-story hotels and apartments with balconies and lit windows, typical of a British coastal town. The sky is overcast, and the scene is illuminated by the soft, fading light of dusk. A few cars are visible, with their taillights creating streaks of light due to motion blur, adding a subtle sense of activity to the otherwise calm setting.

**Keywords:**
- White Rock
- Hastings
- England
- UK
- Seafront
- British seaside town
- Victorian era
- Hotels
- Apartments
- Damp pavement
- Overcast sky
- Dusk
- Concrete planters
- Motion blur
- Taillights
- Quiet atmosphere
- Coastal resort
- Autumn afternoon
```

---

### ✅ mlx-community/pixtral-12b-bf16

**Metrics:** 19.2 TPS | 296 tokens

```text
### Caption
A damp, late autumn afternoon along the seafront at White Rock in Hastings, England. The low-angle perspective highlights the rain-slicked paving stones, which reflect the cool, overcast sky as dusk begins to fall. In the background, a row of hotels and apartments, typical of a British seaside town, lines the road. The motion blur of a passing car's taillights adds a streak of colour and a sense of movement to the otherwise quiet, atmospheric scene.

### Description
The image captures a tranquil moment on a rainy autumn afternoon at White Rock in Hastings, England. The wet pavement glistens under the overcast sky, creating a reflective surface that enhances the serene ambiance. The scene is framed by a row of traditional British seaside buildings, including hotels and apartments, which line the road. The buildings are characterized by their classic architecture, featuring multiple stories and balconies. A passing car's taillights create a dynamic contrast with the otherwise still and calm environment, adding a touch of color and movement to the scene.

### Keywords
- White Rock
- Hastings
- England
- United Kingdom
- Seafront
- Autumn
- Rain
- Reflective pavement
- Overcast sky
- Hotels
- Apartments
- British seaside town
- Taillights
- Dusk
- Tranquil
- Atmospheric
- Victorian architecture
- Motion blur
```

---

### ✅ mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX

**Metrics:** 35.6 TPS | 500 tokens

```text
Here are my reasoning steps:
The user asks: "Provide a factual caption, description, and keywords suitable for cataloguing, or searching for, the image. Context: The image relates to ', White Rock, Hastings, England, United Kingdom, UK

Here is a caption for the image, written for the specified audience and context:

A damp, late autumn afternoon along the seafront at White Rock in Hastings, England. The low-angle perspective highlights the rain-slicked paving stones, which reflect the cool, overcast sky as dusk begins to fall. In the background, a row of hotels and apartments, typical of a British seaside town, lines the road. The motion blur of a passing car's taillights adds a streak of colour and a sense of movement to the otherwise quiet, atmospheric scene. This area was developed significantly during the Victorian era as Hastings grew into a popular coastal resort, and this image captures a modern, tranquil moment outside the peak tourist season.'

The photo was taken around 2025-11-15 16:19:28 GMT . Focus on visual content, drawing on any available contextual information for specificity. Do not speculate."

Thus the user wants a factual caption, description, and keywords suitable for cataloguing or searching for the image. They gave a sample caption but likely want a refined version. The user says "Here is a caption for the image, written for the specified audience and context:" and then they provide a caption. Then they say "The photo was taken around 2025-11-15 16:19:28 GMT . Focus on visual content, drawing on any available contextual information for specificity. Do not speculate."

Thus they want a factual caption, description, and keywords. They gave a sample caption but maybe they want a revised version? Or they want a caption, description, and keywords. The sample caption is already there, but they might want a more concise version? The instruction: "Provide a factual caption, description, and keywords suitable for cataloguing, or searching for, the image." So we need to produce a caption, a description, and a list of keywords. The context is White Rock, Hastings, England, UK. The image shows a rainy, overcast evening, with concrete planters, wet pavement, white buildings, cars, etc. The user also gave a sample caption that includes some historical context
```

---

### ✅ mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

**Metrics:** 9.21 TPS | 171 tokens

```text
I'm happy to help you with your question, but I have to say that I'm a bit confused by the content of your message. It seems like you're trying to provide a factual caption, description, and keywords for an image, but the image itself is not visible to me. Additionally, the text you provided appears to be a jumbled collection of words and phrases that don't seem to be related to each other.

If you could provide more context or clarify what you're trying to achieve with your message, I'd be happy to try and assist you. Alternatively, if you could provide a clear and concise description of the image you're trying to describe, I can try to help you come up with a fitting caption and keywords.

Please let me know how I can help, and I'll do my best to provide a helpful response.
```

---

### ✅ mlx-community/Phi-3.5-vision-instruct-bf16

**Metrics:** 10.8 TPS | 262 tokens

```text
Caption: White Rock Seafront in Hastings, England - Late Autumn Evening.

Description: The image depicts a serene evening scene at the White Rock seafront in Hastings, England. The perspective is from a low angle, emphasizing the reflective, rain-slicked paving stones that stretch out in the foreground. The sky is overcast, suggesting a cool, damp atmosphere typical of late autumn. In the distance, a row of modern hotels and apartments can be seen, indicative of the area's development and its role as a coastal resort. The presence of a car's motion-blurred taillights adds a dynamic element to the otherwise still and tranquil setting. The architecture and the wet pavement suggest a recent rainfall, and the overall mood is one of quiet and calm, with the potential for a bustling atmosphere during peak tourist seasons.

Keywords: White Rock, Hastings, England, seafront, late autumn, overcast sky, rain-slicked pavement, modern hotels, Victorian era development, coastal resort, tranquil, dusk, motion blur, tourist season.
```

---

### ✅ mlx-community/paligemma2-10b-ft-docci-448-bf16

**Metrics:** 5.02 TPS | 115 tokens

```text
The image is a medium-close-up view of three white concrete cubes that are wet and have green algae growing on them. The cubes are sitting on a black tile floor that is also wet. Behind the cubes, there is a black metal fence that has green plants growing on it. On the other side of the fence, there is a road that has cars parked on the right side of it. On the left side of the road, there is a building that is white and has balconies on it. The sky is covered in clouds, and it is gray.
```

---

### ✅ mlx-community/gemma-3-27b-it-qat-8bit

**Metrics:** 15 TPS | 500 tokens

```text
Here's a factual caption, description, and keywords for cataloguing the image, based on your provided context and the visual content:

**Caption:**

White Rock seafront, Hastings, England, on a wet afternoon in November 2025. Rain-slicked paving stones reflect the overcast sky and surrounding buildings.

**Description:**

A low-angle, medium shot of the seafront promenade at White Rock, Hastings, England. The foreground features wet, dark grey paving stones with concrete planters. Reflections of the sky and buildings are visible on the wet surface. A row of white hotels and apartments lines the background, typical of Victorian-era seaside architecture. Motion blur from a passing vehicle’s taillights is visible on the left side of the frame. The scene is lit by overcast daylight. The date and time of the photograph are November 15, 2025, at 16:19:28 GMT.

**Keywords:**

*   White Rock
*   Hastings
*   England
*   UK
*   United Kingdom
*   Seaside
*   Seafront
*   Promenade
*   Victorian Architecture
*   Hotels
*   Apartments
*   Rain
*   Wet Pavement
*   November
*   2025
*   Overcast
*   Urban Landscape
*   Coastal Town
*   Motion Blur
*   Concrete Planters
*   Reflections
*   GMT 16:19:28
*   East Sussex
*   British Seaside Resort
*   Coastal Architecture
*   Street Photography
*   Exterior
*   Daytime
*   Buildings
*   Paving Stones
*   Taillights
*   Vehicle
*   Urban Scene
*   Coastal
*   Exterior View
*   Architecture
*   Travel
*   Landscape
*   Townscape
*   Street
*   Concrete
*   Planters
*   Wet
*   Grey
*   Dark
*   Exterior
*   Exterior View
*   Exterior Scene
*   Exterior Shot
*   Exterior Photograph
*   Exterior Image
*   Exterior Picture
*   Exterior View
*   Exterior Scene
*   Exterior Shot
*   Exterior Photograph
*   Exterior Image
*   Exterior Picture
*   Exterior View
*   Exterior Scene
*
```

---

### ✅ microsoft/Phi-3.5-vision-instruct

**Metrics:** 11 TPS | 500 tokens

```text
Caption: White Rock Seafront in Hastings, England - Late Autumn Evening

Description: The image depicts a serene evening scene at the White Rock seafront in Hastings, England. The wet pavement reflects the overcast sky, indicating recent rain. The perspective is from ground level, looking up towards the street lined with Victorian-era hotels and apartments. The lighting suggests it is dusk, with the street lights beginning to illuminate the area. A car's taillights are captured in motion, adding a dynamic element to the otherwise still scene. The architecture and the wet pavement suggest a typical British seaside town atmosphere.

Keywords: White Rock, Hastings, England, Victorian architecture, seafront, late autumn, dusk, wet pavement, overcast sky, street lights, car motion blur.<|end|><|endoftext|>


Here is a chart that shows the number of books read by students in different grades. The chart is a bar graph with the number of books on the vertical axis and the grades on the horizontal axis. The bars are colored in blue and the grades are labeled as 6, 7, 8, and 9. The chart shows that the students in grade 6 read the most books, followed by grade 7, grade 8, and grade 9.

The chart is a bar graph with the number of books on the vertical axis and the grades on the horizontal axis. The bars are colored in blue and the grades are labeled as 6, 7, 8, and 9. The chart shows that the students in grade 6 read the most books, followed by grade 7, grade 8, and grade 9.

The chart shows that the students in grade 6 read the most books, followed by grade 7, grade 8, and grade 9.

The chart shows that the students in grade 6 read the most books, followed by grade 7, grade 8, and grade 9.

The chart shows that the students in grade 6 read the most books, followed by grade 7, grade 8, and grade 9.

The chart shows that the students in grade 6 read the most books, followed by grade 7, grade 8,
```

---

### ✅ Qwen/Qwen3-VL-2B-Instruct

**Metrics:** 71.7 TPS | 433 tokens

```text
### Factual Caption
A low-angle view of a wet, paved promenade in White Rock, Hastings, England, on a cool, overcast day. The foreground features weathered concrete planters and a reflective, damp surface. In the background, a row of white and cream-colored buildings, typical of a coastal town, lines the street. The scene is illuminated by the soft glow of streetlights and the warm lights from the buildings, with the motion blur of a passing car's taillights adding a sense of movement to the tranquil atmosphere.

### Description
The image captures a serene, late afternoon scene at the seafront of White Rock, Hastings, England. The foreground is dominated by a wet, dark gray paved area with a geometric pattern of square tiles, reflecting the overcast sky. Several concrete planters, some with visible green moss and water stains, are arranged in a staggered formation. The ground is damp, suggesting recent rain. In the background, a row of modern and traditional buildings lines the street, with white facades and some dark trim. The buildings have multiple windows, some with lit interiors, and a few have balconies. A tall streetlight stands prominently in the middle ground. The sky is gray and cloudy, indicating a cool, possibly autumnal day. A few palm trees and other greenery are visible near the buildings. The overall atmosphere is calm and quiet, with the motion blur of a passing car's taillights adding a subtle sense of movement.

### Keywords
- White Rock
- Hastings
- England
- United Kingdom
- UK
- Seafront
- Promenade
- Damp pavement
- Concrete planters
- Overcast sky
- Late afternoon
- Coastal town
- Victorian era
- Modern buildings
- Streetlight
- Motion blur
- Taillights
- Reflective surface
- Greenery
- Coastal resort
- Rain
- Dusk
- Low-angle perspective
- Urban landscape
- British seaside town
- 2025-11-15
- 16:19:28 GMT
```

---

### ✅ mlx-community/Qwen3-VL-2B-Thinking-bf16

**Metrics:** 71.5 TPS | 500 tokens

```text
Got it, let's break this down. First, the task is to provide a factual caption, description, and keywords for the image. The context is White Rock, Hastings, England, UK. The user provided a caption, but we need to make sure it's factual and specific, based on the image.

First, the caption. The user's example is a bit long, but we need to stick to the context. Let's check the image: it's a seafront, damp, late autumn (so maybe early fall), overcast sky, rain-slicked pavement, hotels/apartments in the background, motion blur from a car. The location is White Rock, Hastings, UK. The timestamp is 2025-11-15 16:19:28 GMT, but since it's a future date, maybe that's a typo? Wait, the context says "around 2025-11-15", but the image is a real photo, so maybe the caption should be based on the image, not the timestamp. Wait, the user says "Do not speculate. Focus on visual content, drawing on any available contextual information for specificity."

So the caption should be factual. Let's structure it:

Factual Caption: A rain-slicked seafront promenade in White Rock, Hastings, England, United Kingdom, captured during a late autumn afternoon with overcast skies. The low-angle perspective shows wet paving stones reflecting the cool light, while a row of white and cream-colored multi-story residential buildings (hotels and apartments) line the background. Motion blur from a passing car with red taillights adds subtle movement to the tranquil scene, which is set against a backdrop of a cloudy sky and distant coastal elements.

Description: The image depicts a coastal promenade in White Rock, Hastings, England, UK. The foreground features several rectangular concrete bollards with visible moss and weathering, situated on wet pavement that reflects the overcast sky. Behind the bollards, a row of mid-rise residential buildings (white with dark trim) lines the seafront, with some illuminated windows and a few plants. In the background, a street with blurred car lights (red taillights) suggests movement, while the sky is gray and cloudy, indicating late autumn. The scene is calm, with no people visible, emphasizing the quiet atmosphere of the area.

Keywords: White Rock, Hastings, England,
```

---

### ✅ meta-llama/Llama-3.2-11B-Vision-Instruct

**Metrics:** 3.85 TPS | 254 tokens

```text
Here is a rewritten caption that meets the requirements:

**Image Description:**

* **Location:** White Rock, Hastings, England, UK
* **Scene:** A rainy day scene on the seafront, featuring a row of buildings and a street with cars and a bench
* **Composition:**
	+ The image is taken from a low angle, looking up at the buildings
	+ The foreground features a row of concrete planters with moss and lichen growing on them
	+ The middle ground shows a street with cars and a bench, with a row of buildings in the background
	+ The background features a cloudy sky and a row of buildings with a mix of old and new architecture
* **Mood:** The image conveys a sense of calmness and serenity, with the sound of rain and the smell of wet pavement creating a peaceful atmosphere
* **Time of Day:** The image was taken in the late afternoon or early evening, with the sun setting behind the buildings

**Keywords:**

* Seafront
* Rainy day
* Buildings
* Street
* Bench
* Cars
* Cloudy sky
* Peaceful atmosphere
* Late afternoon/early evening
* White Rock, Hastings, England, UK
```

---

<!-- markdownlint-enable MD013 -->

---

## System/Hardware Information

- **OS**: Darwin 25.1.0
- **macOS Version**: 26.1
- **SDK Version**: 26.1
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
- `huggingface-hub`: `0.36.0`
- `mlx`: `0.30.1.dev20251125+c9f4dc85`
- `mlx-lm`: `0.28.4`
- `mlx-vlm`: `0.3.7`
- `tokenizers`: `0.22.1`
- `transformers`: `4.57.3`

_Report generated on: 2025-11-25 16:28:07 GMT_
