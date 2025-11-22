# Model Performance Results

_Generated on 2025-11-22 19:59:43 GMT_

## 🏆 Performance Highlights

- **Fastest:** `prince-canuma/Florence-2-large-ft` (326.5 tps)
- **💾 Most efficient:** `qnguyen3/nanoLLaVA` (4.5 GB)
- **⚡ Fastest load:** `HuggingFaceTB/SmolVLM-Instruct` (0.00s)
- **📊 Average TPS:** 68.7 across 25 models

## 📈 Resource Usage

- **Total peak memory:** 728.3 GB
- **Average peak memory:** 29.1 GB
- **Memory efficiency:** 81 tokens/GB

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
  - `microsoft/Phi-3.5-vision-instruct` (token: `phrase: "# instruction # solution..."`)
- **📝 Formatting Issues (3):**
  - `mlx-community/Idefics3-8B-Llama3-bf16`
  - `mlx-community/deepseek-vl2-8bit`
  - `prince-canuma/Florence-2-large-ft`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 68.7 | Min: 3.86 | Max: 327
- **Peak Memory**: Avg: 29 | Min: 4.5 | Max: 74
- **Total Time**: Avg: 20.69s | Min: 2.10s | Max: 68.01s
- **Generation Time**: Avg: 17.75s | Min: 1.19s | Max: 63.92s
- **Model Load Time**: Avg: 2.94s | Min: 0.91s | Max: 8.07s

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

**Overall runtime:** 539.35s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                         |   Token |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) |                    Quality Issues |
|:---------------------------------------------------|--------:|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|----------------------------------:|
| `microsoft/Florence-2-large-ft`                    |         |                   |                       |                |              |           |             |                  |            |             |                                   |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`     |         |                   |                       |                |              |           |             |                  |            |             |                                   |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`          |         |                   |                       |                |              |           |             |                  |            |             |                                   |
| `mlx-community/Molmo-7B-D-0924-8bit`               |         |                   |                       |                |              |           |             |                  |            |             |                                   |
| `mlx-community/Molmo-7B-D-0924-bf16`               |         |                   |                       |                |              |           |             |                  |            |             |                                   |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`          |         |                   |                       |                |              |           |             |                  |            |             |                                   |
| `mlx-community/gemma-3-12b-pt-8bit`                |         |                   |                       |                |              |           |             |                  |            |             |                                   |
| `qnguyen3/nanoLLaVA`                               | 151,645 |               249 |                    64 |            313 |        1,197 |       103 |         4.5 |            1.19s |      0.91s |       2.10s |                                   |
| `HuggingFaceTB/SmolVLM-Instruct`                   |  49,154 |             1,430 |                     9 |          1,439 |        1,221 |       130 |         5.5 |            1.68s |      1.33s |       3.02s |                                   |
| `prince-canuma/Florence-2-large-ft`                |     0.0 |               233 |                   500 |            733 |          953 |       327 |         5.1 |            2.14s |      1.31s |       3.45s |      lang_mixing, formatting, ... |
| `mlx-community/SmolVLM-Instruct-bf16`              |  49,154 |             1,430 |                     9 |          1,439 |          883 |       130 |          47 |            2.16s |      1.13s |       3.29s |                                   |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`         |  49,279 |             1,330 |                    23 |          1,353 |          858 |       122 |          47 |            2.21s |      1.17s |       3.38s |                   context-ignored |
| `mlx-community/LFM2-VL-1.6B-8bit`                  |       7 |             2,029 |                   247 |          2,276 |        1,026 |       292 |          53 |            3.28s |      0.92s |       4.20s |                                   |
| `mlx-community/paligemma2-3b-pt-896-4bit`          |       1 |             4,334 |                    22 |          4,356 |        1,206 |       103 |         8.8 |            4.18s |      1.79s |       5.97s |                   context-ignored |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`   |       1 |             1,262 |                   126 |          1,388 |          510 |      40.6 |          11 |            5.92s |      2.66s |       8.58s |                   context-ignored |
| `mlx-community/gemma-3n-E4B-it-bf16`               |     106 |               504 |                   214 |            718 |          720 |      42.2 |          18 |            6.05s |      4.02s |      10.07s |                                   |
| `mlx-community/deepseek-vl2-8bit`                  |       1 |             2,041 |                   180 |          2,221 |          292 |      63.8 |          74 |           10.25s |      8.07s |      18.32s |                        formatting |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`    |       1 |             1,262 |                   180 |          1,442 |        1,441 |      17.6 |          11 |           11.50s |      2.41s |      13.91s |                   context-ignored |
| `mlx-community/llava-v1.6-mistral-7b-8bit`         |       2 |             2,321 |                     8 |          2,329 |          213 |      47.1 |          12 |           11.59s |      1.71s |      13.30s |                   context-ignored |
| `mlx-community/Idefics3-8B-Llama3-bf16`            | 128,001 |             2,547 |                   278 |          2,825 |          706 |      29.3 |          19 |           13.55s |      3.30s |      16.84s |       formatting, context-ignored |
| `mlx-community/InternVL3-14B-8bit`                 | 151,645 |             2,032 |                   264 |          2,296 |          386 |      28.8 |          18 |           14.83s |      3.13s |      17.96s |                       bullets(17) |
| `mlx-community/pixtral-12b-8bit`                   |       2 |             2,842 |                   315 |          3,157 |          498 |      34.0 |          15 |           15.35s |      2.85s |      18.19s |              verbose, bullets(19) |
| `mlx-community/gemma-3-27b-it-qat-4bit`            |     106 |               504 |                   334 |            838 |          142 |      26.9 |          60 |           16.27s |      4.03s |      20.30s |                       bullets(31) |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`    |   1,261 |             2,933 |                   500 |          3,433 |          406 |      35.7 |          15 |           21.58s |      2.75s |      24.32s |                                   |
| `mlx-community/pixtral-12b-bf16`                   |       2 |             2,842 |                   313 |          3,155 |          535 |      19.2 |          27 |           21.98s |      4.52s |      26.50s |              verbose, bullets(16) |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit` | 128,009 |               236 |                   164 |            400 |         65.7 |      8.44 |          48 |           23.36s |      2.56s |      25.91s |                   context-ignored |
| `mlx-community/Phi-3.5-vision-instruct-bf16`       |  32,007 |             1,042 |                   205 |          1,247 |          199 |      10.9 |          69 |           24.23s |      5.15s |      29.37s |                                   |
| `mlx-community/gemma-3-27b-it-qat-8bit`            |     106 |               504 |                   331 |            835 |          145 |      15.4 |          42 |           25.34s |      6.11s |      31.45s |                       bullets(26) |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`   |       1 |             1,262 |                   162 |          1,424 |          463 |      4.99 |          27 |           35.58s |      4.36s |      39.94s |                   context-ignored |
| `microsoft/Phi-3.5-vision-instruct`                |   2,692 |             1,042 |                   500 |          1,542 |          907 |      10.8 |          12 |           47.56s |      1.88s |      49.44s | repetitive(phrase: "# instruct... |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`          |     916 |            16,505 |                   500 |         17,005 |          326 |      71.9 |          54 |           57.92s |      1.42s |      59.34s |                           verbose |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`         | 128,009 |               237 |                   234 |            471 |         76.9 |      3.86 |          25 |           63.92s |      4.09s |      68.01s |                       bullets(18) |

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

**Metrics:** 103 TPS | 64 tokens

```text
The image captures a rainy day in White Rock, Hastings, England, with a view of the seafront. The low-angle perspective highlights the wet pavement and the overcast sky. The image was taken around 2025-11-15 16:19:28 GMT.
```

---

### ✅ HuggingFaceTB/SmolVLM-Instruct

**Metrics:** 130 TPS | 9 tokens

```text
 A rainy day in Hastings, England.
```

---

### ✅ prince-canuma/Florence-2-large-ft

**Metrics:** 327 TPS | 500 tokens

```text
<s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s>
```

---

### ✅ mlx-community/SmolVLM-Instruct-bf16

**Metrics:** 130 TPS | 9 tokens

```text
 A rainy day in Hastings, England.
```

---

### ✅ mlx-community/SmolVLM2-2.2B-Instruct-mlx

**Metrics:** 122 TPS | 23 tokens

```text
 A wet pavement with three large concrete blocks in the foreground. In the background, there are buildings and cars.
```

---

### ✅ mlx-community/LFM2-VL-1.6B-8bit

**Metrics:** 292 TPS | 247 tokens

```text
Caption: "Damp, late autumn afternoon at White Rock, Hastings, England. Rain-slicked paving stones reflect the overcast sky as dusk falls. Modern, tranquil scene outside the peak tourist season."

Description: The image captures a damp, late autumn afternoon along the seafront at White Rock in Hastings, England. The low-angle perspective highlights the rain-slicked paving stones, which reflect the cool, overcast sky as dusk begins to fall. In the background, a row of hotels and apartments, typical of a British seaside town, lines the road. The motion blur of a passing car's taillights adds a streak of color and a sense of movement to the otherwise quiet, atmospheric scene. This area was developed significantly during the Victorian era as Hastings grew into a popular coastal resort, and this image captures a modern, tranquil moment outside the peak tourist season.

Keywords: White Rock, Hastings, England, United Kingdom, United Kingdom, UK, seaside town, Victorian era, Victorian development, modern, tranquil, autumn, damp, paving stones, overcast sky, dusk, motion blur, car taillights, British coastal resort.
```

---

### ✅ mlx-community/paligemma2-3b-pt-896-4bit

**Metrics:** 103 TPS | 22 tokens

```text
Provide a factual caption, description, and keywords suitable for cataloguing, or searching for, the image.
```

---

### ✅ mlx-community/paligemma2-10b-ft-docci-448-6bit

**Metrics:** 40.6 TPS | 126 tokens

```text
The image is of a group of four concrete planters on a dark gray square tile surface. The planters are white with dark green algae growth on the sides. The planters are in the middle of the image, with the one on the left side of the image in the foreground. The other three planters are behind the one in the foreground. The planters are casting shadows on the ground. There is a row of white buildings in the background. The buildings are all the same height and have balconies on the second and third floors. There are cars parked on the side of the road in the background. The sky is gray and overcast.
```

---

### ✅ mlx-community/gemma-3n-E4B-it-bf16

**Metrics:** 42.2 TPS | 214 tokens

```text
## Caption:

Rain-slicked pavement reflects the soft glow of streetlights and distant car headlights along the promenade at White Rock, Hastings, UK, as dusk settles over the British seaside town. The architecture lining the street evokes a classic seaside charm, a testament to Hastings' historical growth as a popular coastal resort.

## Context:

The image captures a typical evening scene on White Rock, a prominent promenade in Hastings, East Sussex, United Kingdom. The wet pavement indicates recent or ongoing rain, creating reflections of the surrounding lights and buildings. The buildings in the background are characteristic of Victorian and Edwardian architecture often found in British seaside towns, reflecting Hastings' history as a significant resort destination during the Victorian era. The presence of streetlights and distant traffic further emphasizes the urban yet relaxed atmosphere of the location.

## Keywords:

White Rock, Hastings, England, UK, promenade, seaside, dusk, rain, reflections, buildings, architecture, British, coastal, urban, evening, streetlights, cars, Victorian, Edwardian, resort, pavement.
```

---

### ✅ mlx-community/deepseek-vl2-8bit

**Metrics:** 63.8 TPS | 180 tokens

```text
<start of description>
The image captures a quiet, overcast day in White Rock, Hastings, UK. The foreground is dominated by a series of large, light-colored, square planters with a slightly weathered appearance, likely made of concrete or a similar material. The ground is a dark, wet tarmac, reflecting the overcast sky and suggesting recent rain. In the background, there are multi-story buildings with a mix of modern and older architectural styles. The overall color palette is muted, with the grays of the sky and the buildings' facades, accented by the green of some plants and the dark, glistening road. The atmosphere is overcast, and the overall mood is calm and somewhat melancholic, typical of a rainy day in a British town. There are no visible people in the scene, and the image exudes a serene, almost desolate atmosphere.
```

---

### ✅ mlx-community/paligemma2-3b-ft-docci-448-bf16

**Metrics:** 17.6 TPS | 180 tokens

```text
A low-angle, close up view of three concrete cubes in the center of the frame. The cubes are a light gray with a dark gray concrete base. The top of the cubes are rounded. The left cube has a small patch of green algae growing on the top of it. The right cube has a small patch of green algae growing on the top of it. The right cube is slightly lower than the left cube. The left and right cubes are separated by a small gap. The cubes are placed on a dark gray, square tiled sidewalk. The sidewalk has a wet, damp surface. The background is a row of white and gray apartment buildings. The buildings are in the center of the frame. The buildings are in the background, with the left side of the frame showing a row of cars. The cars are facing to the right. The sky is a light blue, overcast sky.
```

---

### ✅ mlx-community/llava-v1.6-mistral-7b-8bit

**Metrics:** 47.1 TPS | 8 tokens

```text
The image is a photograph.
```

---

### ✅ mlx-community/Idefics3-8B-Llama3-bf16

**Metrics:** 29.3 TPS | 278 tokens

```text
 In the foreground of the image, there are four rectangular concrete blocks placed on the wet pavement. The blocks are positioned in a line, with two blocks on the left and two blocks on the right. The pavement is made of dark grey tiles, and it is wet from the rain, reflecting the overcast sky above. The sky is a muted grey, with some patches of darker clouds, suggesting a cloudy day.

In the background, there is a row of buildings that appear to be hotels or apartments. These buildings are white with some darker accents, and they have multiple windows. The architecture is typical of a seaside town, with a mix of modern and possibly older structures. The buildings are lined up along a road, and there are some trees and shrubs in front of them.

To the left of the buildings, there is a street lamp, and to the right, there is a car with its taillights visible. The car is moving, as indicated by the motion blur of its lights. The overall atmosphere of the image is calm and quiet, with a sense of stillness despite the motion of the car.

The image captures a moment of tranquility in a seaside town, with the wet pavement and overcast sky contributing to a serene mood. The buildings and the car suggest a blend of modern and traditional elements, typical of a town that has evolved over time.<end_of_utterance>
```

---

### ✅ mlx-community/InternVL3-14B-8bit

**Metrics:** 28.8 TPS | 264 tokens

```text
**Caption:**
A tranquil scene at White Rock, Hastings, England, on a damp, late afternoon. The wet paving stones reflect the overcast sky as dusk approaches. In the background, a row of hotels and apartments lines the road, with the motion blur of passing cars adding a touch of movement. This area, developed during the Victorian era, captures a quiet moment outside the peak tourist season.

**Description:**
The image depicts a serene seaside promenade at White Rock, Hastings, England. The foreground features large, weathered concrete planters on a wet, tiled pavement, reflecting the subdued light of the overcast sky. In the background, a row of multi-story hotels and apartments, typical of a British seaside town, stretches along the road. The motion blur of passing cars' taillights adds a dynamic element to the otherwise calm scene. The atmosphere suggests a quiet, late autumn afternoon, with the area appearing tranquil and devoid of crowds.

**Keywords:**
- White Rock
- Hastings
- England
- UK
- Seafront
- Hotels
- Apartments
- Victorian era
- Seaside town
- Overcast sky
- Wet pavement
- Concrete planters
- Motion blur
- Dusk
- Autumn
- Tranquil scene
- British coastal resort
```

---

### ✅ mlx-community/pixtral-12b-8bit

**Metrics:** 34 TPS | 315 tokens

```text
### Caption
A damp, late autumn afternoon along the seafront at White Rock in Hastings, England. The low-angle perspective highlights the rain-slicked paving stones, which reflect the cool, overcast sky as dusk begins to fall. In the background, a row of hotels and apartments, typical of a British seaside town, lines the road. The motion blur of a passing car's taillights adds a streak of colour and a sense of movement to the otherwise quiet, atmospheric scene.

### Description
The image captures a serene and slightly melancholic scene at White Rock in Hastings, England. The wet pavement, glistening under the overcast sky, leads the viewer's eye towards a row of white buildings characteristic of British seaside architecture. The buildings, likely hotels or apartments, are adorned with balconies and windows, some of which are illuminated, suggesting the onset of evening. A few cars are parked along the street, and the blurred taillights of a passing vehicle add a dynamic element to the otherwise still scene. The overall mood is tranquil and reflective, typical of a quiet moment outside the peak tourist season.

### Keywords
- White Rock
- Hastings
- England
- United Kingdom
- Seafront
- Autumn
- Rain
- Dusk
- Hotels
- Apartments
- British Seaside
- Reflections
- Taillights
- Motion Blur
- Victorian Architecture
- Tranquil
- Atmospheric
- Overcast Sky
- Wet Pavement
```

---

### ✅ mlx-community/gemma-3-27b-it-qat-4bit

**Metrics:** 26.9 TPS | 334 tokens

```text
## Catalog Entry for Image:

**Caption:** Wet paving stones reflecting the overcast sky at White Rock, Hastings, England, with hotels and apartments in the background. Taken November 15, 2025, at approximately 16:19 GMT.

**Description:** A low-angle, medium-shot photograph depicting the seafront promenade at White Rock, Hastings, England. The foreground features wet, dark grey paving stones with visible reflections of the overcast sky. Four concrete block planters, showing signs of moss growth, are positioned in the lower centre of the frame. Behind the planters, a row of predominantly white and light-coloured hotels and apartment buildings lines the road. Motion blur indicates a vehicle with red taillights passing along the road. Streetlights are visible in the background. The overall atmosphere is quiet and damp.

**Keywords:**

*   White Rock
*   Hastings
*   England
*   United Kingdom
*   UK
*   Seaside
*   Seafront
*   Promenade
*   Hotels
*   Apartments
*   Victorian Architecture (implied)
*   Rain
*   Wet Pavement
*   Overcast Sky
*   Dusk
*   November
*   2025
*   GMT
*   Urban Landscape
*   Street Photography
*   Concrete Planters
*   Motion Blur
*   Coastal Town
*   Exterior
*   Buildings
*   Streetlights
*   Reflections
*   Paving Stones
*   Moss
*   Coastal Architecture
*   British Seaside Resort




```

---

### ✅ mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX

**Metrics:** 35.7 TPS | 500 tokens

```text
Here are my reasoning steps:
The user asks: "Provide a factual caption, description, and keywords suitable for cataloguing, or searching for, the image. Context: The image relates to ', White Rock, Hastings, England, United Kingdom, UK"

They also gave a sample caption that is quite detailed, but they ask to "Provide a factual caption, description, and keywords suitable for cataloguing, or searching for, the image."

We need to produce a factual caption, description, and keywords. The image is a photo of a damp, late autumn afternoon along the seafront at White Rock in Hastings, England. The low-angle perspective shows rain-slicked paving stones reflecting the overcast sky as dusk begins. In the background, a row of hotels and apartments typical of a British seaside town lines the road. Motion blur of a passing car's taillights adds a streak of colour and a sense of movement to the otherwise quiet, atmospheric scene. The area was developed significantly during the Victorian era as Hastings grew into a popular coastal resort, and this image captures a modern, tranquil moment outside the peak tourist season.

But we need to be factual, not speculative. So we should not mention Victorian era unless we are sure. The user says "Do not speculate." So we should stick to what we can see: the image shows concrete planters, wet pavement, white buildings with balconies, cars, streetlights, etc. The location is White Rock, Hastings, UK. The time appears to be dusk/evening, overcast, wet. The perspective is low-angle. The image includes concrete planters (maybe water features). The background includes multi-story white buildings with balconies, typical of seaside architecture. There are cars on the road, some with taillights blurred. There are streetlights. The sky is overcast. The pavement is wet, reflecting the sky and lights. There are some plants in the planters. The overall mood is quiet, atmospheric.

We need to produce a caption, a description, and keywords.

The caption should be concise but descriptive, suitable for cataloguing. The description can be more detailed. Keywords should be relevant for searching.

We need to ensure we do not add speculation about historical context unless we are sure. The user gave a sample caption that includes Victorian era, but they said "Do not speculate." So we should not include that. We can mention that the area is a
```

---

### ✅ mlx-community/pixtral-12b-bf16

**Metrics:** 19.2 TPS | 313 tokens

```text
### Caption
A damp, late autumn afternoon along the seafront at White Rock in Hastings, England. The low-angle perspective highlights the rain-slicked paving stones, which reflect the cool, overcast sky as dusk begins to fall. In the background, a row of hotels and apartments, typical of a British seaside town, lines the road. The motion blur of a passing car's taillights adds a streak of colour and a sense of movement to the otherwise quiet, atmospheric scene.

### Description
The image captures a tranquil moment on a late autumn afternoon at White Rock in Hastings, England. The wet pavement, glistening under the overcast sky, leads the viewer's eye towards a row of Victorian-style hotels and apartments. The buildings, painted in shades of white and grey, stand tall against the backdrop of a setting sun. A few cars are parked along the street, with one car's taillights creating a dynamic motion blur, adding a touch of color to the otherwise muted palette. The scene is peaceful, evoking a sense of calm and quietude typical of the off-peak tourist season.

### Keywords
- White Rock, Hastings
- England, United Kingdom
- Seafront
- Victorian architecture
- Rain-slicked pavement
- Overcast sky
- Dusk
- Hotels and apartments
- Taillights
- Motion blur
- Autumn
- Tranquil scene
- British seaside town
- Off-peak season
- Victorian era
- Coastal resort
```

---

### ✅ mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

**Metrics:** 8.44 TPS | 164 tokens

```text
I'm happy to help you with your question, but I have to say that I'm a bit confused by the content of your message. It seems like you're trying to provide a factual caption, description, and keywords for an image, but the image itself is not visible to me. Additionally, the text you provided appears to be a jumbled collection of words and phrases that don't seem to be related to each other.

If you could provide more context or clarify what you're trying to achieve with your message, I'd be happy to try and assist you. Alternatively, if you could provide a clear and concise description of the image you're trying to caption, I'd be happy to try and help you come up with a fitting caption and keywords.

Please let me know how I can assist you further.
```

---

### ✅ mlx-community/Phi-3.5-vision-instruct-bf16

**Metrics:** 10.9 TPS | 205 tokens

```text
Caption: 'White Rock Seafront in Hastings, England - A serene, rainy evening scene with reflective paving stones and a row of modern buildings. Date: 2025-11-15 16:19:28 GMT.'

Description: This image captures a tranquil moment at the White Rock seafront in Hastings, England. The wet paving stones reflect the overcast sky, and the low lighting suggests it's either early morning or late evening. The background features a row of contemporary buildings, indicative of the area's development during the Victorian era. The motion blur of a car's taillights adds a dynamic element to the otherwise still scene.

Keywords: White Rock, Hastings, England, seafront, Victorian development, rainy evening, paving stones, modern buildings, overcast sky, motion blur, car taillights.
```

---

### ✅ mlx-community/gemma-3-27b-it-qat-8bit

**Metrics:** 15.4 TPS | 331 tokens

```text
Here's a factual caption, description, and keywords for cataloguing the image, based on your provided context and the visual content:

**Caption:**

White Rock seafront, Hastings, England, on a wet afternoon in November 2025. Rain-slicked paving stones reflect the overcast sky and surrounding buildings.

**Description:**

A low-angle, medium shot of the seafront promenade at White Rock, Hastings, England. The foreground features wet, dark grey paving stones with concrete planters. Reflections of the sky and buildings are visible on the wet surface. A row of white and pastel-coloured hotels and apartments lines the background. Motion blur from a passing vehicle’s taillights is visible on the left side of the frame. The scene is lit by overcast daylight. The date and time of the photograph are November 15, 2025, at 16:19:28 GMT.

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
*   2025
*   Rain
*   Wet pavement
*   Overcast
*   Buildings
*   Hotels
*   Apartments
*   Urban landscape
*   Coastal resort
*   Motion blur
*   Street photography
*   Concrete planters
*   Reflections
*   GMT
*   Victorian era (contextual - for related searches)
*   Coastal architecture
*   British seaside town
```

---

### ✅ mlx-community/paligemma2-10b-ft-docci-448-bf16

**Metrics:** 4.99 TPS | 162 tokens

```text
The image is of a low-angle view of three white concrete planters with green algae growth on the sides of them. The planters are placed on a dark gray square tile surface. The tiles are wet from a recent rainfall. The tiles are reflecting the sky and the planters. The sky is covered in low, gray clouds. The top of a white building is in the background. The building has a black roof. The building has balconies with black railings. The building is on the right side of the image. A row of cars is parked on the road in front of the building. The cars are facing the right side of the image. The cars' brake lights are on. A streetlight is on the left side of the image. The streetlight is reflecting off of the water on the tiles.
```

---

### ✅ microsoft/Phi-3.5-vision-instruct

**Metrics:** 10.8 TPS | 500 tokens

```text
Caption: White Rock Seafront in Hastings, England - Late Autumn Evening.

Description: The image depicts a serene evening scene at the White Rock seafront in Hastings, England. The wet pavement reflects the dimming light of the setting sun, with the sky transitioning from day to dusk. The architecture of the area is characterized by a row of modern hotels and apartments, indicative of the town's development during the Victorian era. A car's motion blur is captured in the background, adding a dynamic element to the otherwise still landscape. The date and time stamp on the image suggests it was taken in late November 2025.

Keywords: White Rock, Hastings, England, Victorian architecture, seafront, evening, dusk, pavement, reflection, car motion blur, tranquil, coastal resort.<|end|><|endoftext|>
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



# SOLUT
```

---

### ✅ mlx-community/Qwen3-VL-2B-Thinking-bf16

**Metrics:** 71.9 TPS | 500 tokens

```text
Got it, let's break this down. First, the task is to provide a factual caption, description, and keywords for the image. The context is White Rock, Hastings, England, UK. The given caption is already there, but we need to make sure it's factual, based on the image, and includes the context.

First, the caption. The user provided a caption, but we need to check if it's accurate. The context says it's a damp, late autumn afternoon. The image shows rain-slicked pavement, overcast sky, so that matches. The row of hotels/apartments—White Rock is a seaside town, Hastings has those. The motion blur of a car with taillights—yes, in the background, there are cars, so that's correct. The Victorian era development—Hastings did grow as a resort, White Rock is part of that. The photo was taken around 2025-11-15, which is future date, but the context says "around 2025-11-15 16:19:28 GMT", so we can note that.

Now, the description: need to be factual, specific. Let's list elements: location (White Rock, Hastings, England, UK), time (late autumn afternoon, around 2025-11-15, 16:19 GMT), weather (damp, overcast, rain-slicked pavement), visual elements (paving stones, reflections, motion blur of car taillights, row of white buildings with balconies, street lamps, plants), context (Victorian-era development, seaside resort, peak tourist season vs. tranquil moment).

Keywords: need to be specific for cataloging/searching. So terms like White Rock, Hastings, England, UK, seaside town, Victorian era, coastal resort, rain-slicked pavement, overcast sky, motion blur, car taillights, residential buildings, hotels, apartments, late autumn, damp, tranquil atmosphere, street lamps, pavement tiles, reflections.

Wait, the user says "Do not speculate. Focus on visual content, drawing on any available contextual information for specificity." So we can't add things not in the image. Let's check the image: the buildings are white, multi-story, some with balconies, street lamps, cars with red taillights, wet pavement, plants, over
```

---

### ✅ meta-llama/Llama-3.2-11B-Vision-Instruct

**Metrics:** 3.86 TPS | 234 tokens

```text
Here is a rewritten caption that meets the requirements:

**Image Description:**

* **Location:** White Rock, Hastings, England, UK
* **Scene:** A rainy day scene on the seafront, with a row of buildings and cars in the background
* **Composition:**
	+ The image features a row of concrete planters with green moss growing on them
	+ The planters are arranged in a row, with a bench in front of them
	+ The background is out of focus, but shows a row of buildings and cars
* **Atmosphere:** The image has a moody and atmospheric feel, with the rain and overcast sky creating a sense of gloominess
* **Mood:** The overall mood of the image is calm and peaceful, with the sound of the rain and the sight of the empty planters creating a sense of serenity

**Keywords:**

* Seafront
* Rainy day
* Concrete planters
* Green moss
* Bench
* Buildings
* Cars
* Overcast sky
* Peaceful
* Calm
* Serene
* Atmospheric
* Moody
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
- `mlx`: `0.30.1.dev20251122+5b0f0472`
- `mlx-lm`: `0.28.4`
- `mlx-vlm`: `0.3.7`
- `tokenizers`: `0.22.1`
- `transformers`: `4.57.1`

_Report generated on: 2025-11-22 19:59:43 GMT_
