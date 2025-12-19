# Model Performance Results

_Generated on 2025-12-19 22:43:23 GMT_

## 🏆 Performance Highlights

- **Fastest:** `mlx-community/Ministral-3-3B-Instruct-2512-4bit` (155.2 tps)
- **💾 Most efficient:** `mlx-community/SmolVLM2-2.2B-Instruct-mlx` (5.5 GB)
- **⚡ Fastest load:** `HuggingFaceTB/SmolVLM-Instruct` (0.00s)
- **📊 Average TPS:** 55.4 across 26 models

## 📈 Resource Usage

- **Total peak memory:** 467.9 GB
- **Average peak memory:** 18.0 GB
- **Memory efficiency:** 186 tokens/GB

## ⚠️ Quality Issues

- **❌ Failed Models (10):**
  - `microsoft/Florence-2-large-ft` (`Model Error`)
  - `microsoft/Phi-3.5-vision-instruct` (`Error`)
  - `mlx-community/InternVL3-14B-8bit` (`Error`)
  - `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16` (`Lib Version`)
  - `mlx-community/Kimi-VL-A3B-Thinking-8bit` (`Lib Version`)
  - `mlx-community/LFM2-VL-1.6B-8bit` (`Error`)
  - `mlx-community/Phi-3.5-vision-instruct-bf16` (`Error`)
  - `mlx-community/Qwen2-VL-2B-Instruct-4bit` (`OOM`)
  - `mlx-community/gemma-3-12b-pt-8bit` (`Error`)
  - `prince-canuma/Florence-2-large-ft` (`Error`)
- **🔄 Repetitive Output (2):**
  - `mlx-community/paligemma2-10b-ft-docci-448-6bit` (token: `phrase: "there are gray stone..."`)
  - `mlx-community/paligemma2-3b-ft-docci-448-bf16` (token: `phrase: "the squares alternate between..."`)
- **👻 Hallucinations (1):**
  - `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`
- **📝 Formatting Issues (2):**
  - `mlx-community/GLM-4.6V-Flash-6bit`
  - `mlx-community/Idefics3-8B-Llama3-bf16`

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 55.4 | Min: 3.89 | Max: 155
- **Peak Memory**: Avg: 18 | Min: 5.5 | Max: 48
- **Total Time**: Avg: 26.66s | Min: 2.80s | Max: 99.26s
- **Generation Time**: Avg: 23.09s | Min: 1.62s | Max: 94.97s
- **Model Load Time**: Avg: 3.57s | Min: 1.07s | Max: 7.16s

> **Prompt used:**
>
> Provide a factual caption, description, and keywords suitable for cataloguing, or searching for, the image.
>
> Context: The image relates to ', British Library, London, England, United Kingdom, UK
> On a late winter afternoon in London, the grand entrance hall of the British Library provides a warm and studious atmosphere for its visitors. As the December day fades to dusk, the building's modernist interior is illuminated, highlighting the striking interplay of red brick, white balconies, and geometric floor patterns.
>
> Throughout the expansive, multi-level space, scholars, students, and members of the public are seen working on laptops, reading, or relaxing in quiet contemplation. The scene captures the library's dual purpose as both a vital hub for research and a welcoming public space for learning and reflection.
>
> Designed by architect Colin St John Wilson and opened to the public in 1997, the British Library's St Pancras building was the largest UK public building constructed in the 20th century and houses a vast collection of the world's knowledge.'
>
> The photo was taken around 2025-12-13 16:27:09 GMT from GPS 51.529533°N, 0.127433°W . Focus on visual content, drawing on any available contextual information for specificity. Do not speculate.

**Note:** Results sorted: errors first, then by generation time (fastest to slowest).

**Overall runtime:** 727.46s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                              |   Token |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues                    |
|:--------------------------------------------------------|--------:|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:----------------------------------|
| `microsoft/Florence-2-large-ft`                         |         |                   |                       |                |              |           |             |                  |            |             |                                   |
| `microsoft/Phi-3.5-vision-instruct`                     |         |                   |                       |                |              |           |             |                  |            |             |                                   |
| `mlx-community/InternVL3-14B-8bit`                      |         |                   |                       |                |              |           |             |                  |            |             |                                   |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          |         |                   |                       |                |              |           |             |                  |            |             |                                   |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |         |                   |                       |                |              |           |             |                  |            |             |                                   |
| `mlx-community/LFM2-VL-1.6B-8bit`                       |         |                   |                       |                |              |           |             |                  |            |             |                                   |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            |         |                   |                       |                |              |           |             |                  |            |             |                                   |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               |         |                   |                       |                |              |           |             |                  |            |             |                                   |
| `mlx-community/gemma-3-12b-pt-8bit`                     |         |                   |                       |                |              |           |             |                  |            |             |                                   |
| `prince-canuma/Florence-2-large-ft`                     |         |                   |                       |                |              |           |             |                  |            |             |                                   |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              |  49,279 |             1,381 |                    12 |          1,393 |        1,194 |       127 |         5.5 |            1.62s |      1.38s |       2.99s |                                   |
| `HuggingFaceTB/SmolVLM-Instruct`                        |  49,154 |             1,481 |                    13 |          1,494 |        1,230 |       126 |         5.5 |            1.67s |      2.33s |       4.00s |                                   |
| `mlx-community/SmolVLM-Instruct-bf16`                   |  49,154 |             1,481 |                    13 |          1,494 |        1,238 |       123 |         5.5 |            1.69s |      1.66s |       3.35s |                                   |
| `qnguyen3/nanoLLaVA`                                    | 151,645 |               301 |                   110 |            411 |        1,278 |       105 |         7.8 |            1.73s |      1.07s |       2.80s |                                   |
| `mlx-community/deepseek-vl2-8bit`                       |       1 |             2,314 |                    59 |          2,373 |        1,081 |      64.2 |          32 |            3.53s |      5.37s |       8.90s | context-ignored                   |
| `mlx-community/paligemma2-3b-pt-896-4bit`               |       1 |             4,387 |                     9 |          4,396 |        1,246 |       124 |         8.7 |            3.92s |      3.26s |       7.18s | context-ignored                   |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       |       2 |             2,994 |                   333 |          3,327 |          848 |       155 |         7.7 |            6.00s |      2.01s |       8.01s |                                   |
| `mlx-community/gemma-3n-E4B-it-bf16`                    |     106 |               559 |                   409 |            968 |          737 |      41.9 |          18 |           10.86s |      5.00s |      15.86s | verbose                           |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              |       2 |             2,571 |                    15 |          2,586 |          240 |      46.0 |          12 |           11.40s |      1.94s |      13.33s |                                   |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        |     578 |             1,315 |                   500 |          1,815 |          503 |      41.5 |          12 |           15.00s |      3.86s |      18.86s | repetitive(phrase: "there are...  |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 |     106 |               559 |                   338 |            897 |          177 |      28.2 |          19 |           15.52s |      4.95s |      20.47s | bullets(33)                       |
| `mlx-community/pixtral-12b-8bit`                        |       2 |             3,161 |                   319 |          3,480 |          518 |      34.2 |          16 |           15.76s |      3.75s |      19.52s | verbose, bullets(30)              |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | 128,001 |             2,586 |                   364 |          2,950 |          681 |      29.1 |          19 |           16.68s |      3.83s |      20.50s | formatting                        |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |       2 |             2,429 |                   252 |          2,681 |          221 |      26.9 |          22 |           20.70s |      4.65s |      25.34s |                                   |
| `mlx-community/pixtral-12b-bf16`                        |       2 |             3,161 |                   299 |          3,460 |          537 |      19.4 |          27 |           21.60s |      5.17s |      26.77s | bullets(24)                       |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         |   5,965 |             3,259 |                   500 |          3,759 |          382 |      34.7 |          15 |           23.25s |      3.39s |      26.63s | hallucination                     |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        |       1 |             1,315 |                   120 |          1,435 |          474 |      5.07 |          27 |           26.81s |      5.54s |      32.34s |                                   |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 |     106 |               559 |                   375 |            934 |          169 |      15.5 |          32 |           27.90s |      7.16s |      35.07s | bullets(27)                       |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         |   4,868 |             1,315 |                   500 |          1,815 |        1,491 |      18.0 |          11 |           28.94s |      3.50s |      32.44s | repetitive(phrase: "the square... |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      | 128,009 |               275 |                   219 |            494 |         80.8 |      8.46 |          15 |           29.63s |      3.00s |      32.63s | context-ignored                   |
| `mlx-community/GLM-4.6V-Flash-6bit`                     |   5,248 |             6,386 |                   500 |          6,886 |          279 |      44.0 |          13 |           34.59s |      2.68s |      37.27s | verbose, formatting               |
| `mlx-community/Molmo-7B-D-0924-8bit`                    | 151,643 |             1,485 |                   232 |          1,717 |         42.8 |      45.6 |          41 |           40.26s |      5.80s |      46.06s |                                   |
| `mlx-community/Molmo-7B-D-0924-bf16`                    | 151,643 |             1,485 |                   233 |          1,718 |         43.7 |      28.5 |          48 |           42.57s |      3.78s |      46.35s |                                   |
| `Qwen/Qwen3-VL-2B-Instruct`                             | 151,645 |            16,618 |                   350 |         16,968 |          360 |      72.3 |          12 |           51.46s |      1.72s |      53.18s | verbose, bullets(32)              |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |     850 |            16,620 |                   500 |         17,120 |          370 |      72.4 |          12 |           52.26s |      1.74s |      54.00s |                                   |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | 128,009 |               276 |                   356 |            632 |         88.3 |      3.89 |          25 |           94.97s |      4.29s |      99.26s |                                   |

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

### ❌ mlx-community/LFM2-VL-1.6B-8bit

**Status:** Failed (Error)
**Error:** Unsupported model: lfm2-vl
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
> allocate 136434163712 bytes which is greater than the maximum allowed buffer
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

### ❌ prince-canuma/Florence-2-large-ft

**Status:** Failed (Error)
**Error:**

> Model loading failed: RobertaTokenizer has no attribute
> additional_special_tokens
**Type:** `ValueError`

---

### ✅ mlx-community/SmolVLM2-2.2B-Instruct-mlx

**Metrics:** 127 TPS | 12 tokens

```text
 grand entrance hall of the british library, london
```

---

### ✅ HuggingFaceTB/SmolVLM-Instruct

**Metrics:** 126 TPS | 13 tokens

```text
 British Library, London, England, United Kingdom, UK.
```

---

### ✅ mlx-community/SmolVLM-Instruct-bf16

**Metrics:** 123 TPS | 13 tokens

```text
 British Library, London, England, United Kingdom, UK.
```

---

### ✅ qnguyen3/nanoLLaVA

**Metrics:** 105 TPS | 110 tokens

```text
A photograph captures the grand entrance of the British Library, London, England, United Kingdom, UK, during the late winter afternoon. The image showcases the interior of the building, with students, scholars, and visitors engaged in various activities. The photograph was taken on December 13, 2025-12-13 16:27:09 GMT, and was taken from GPS 51.529533°N, 0.127433°W.
```

---

### ✅ mlx-community/deepseek-vl2-8bit

**Metrics:** 64.2 TPS | 59 tokens

```text
on the context, and the answer is a lot of the answer, and the answer, and the answer, andI'm ready to answer the answer, and theon theon thetheanswer, and thethebrickstriple, and the answer, and thelime?
```

---

### ✅ mlx-community/paligemma2-3b-pt-896-4bit

**Metrics:** 124 TPS | 9 tokens

```text
Provide a factual description of the scene.
```

---

### ✅ mlx-community/Ministral-3-3B-Instruct-2512-4bit

**Metrics:** 155 TPS | 333 tokens

```text
**Caption:**
*British Library St Pancras – Interior Grandeur: A Modernist Sanctuary of Knowledge*
The expansive, multi-level entrance hall of the British Library’s iconic St Pancras building showcases its architectural brilliance. Illuminated by natural light filtering through the vaulted ceiling, the space blends modernist design elements—such as red brick walls, geometric floor patterns, and white balconies—with serene study nooks.

---

**Description:**
This image depicts the grand, multi-story entrance hall of the British Library’s St Pancras building. The hall features a striking architectural design with a combination of red brick and white stone accents. Prominent geometric patterns on the floor, warm lighting, and open balconies create an inviting atmosphere. Visitors are scattered throughout the space, engaging in quiet study, reading, or working on laptops, embodying the library’s dual role as a research hub and a public space for learning.

---

**Keywords:**
- British Library St Pancras
- Modernist architecture
- Multi-level entrance hall
- Red brick and white stone
- Geometric floor patterns
- Study nooks
- Public research space
- Colin St John Wilson’s design
- Academic atmosphere
- London, UK
- Historical library building
- Open-plan interior
- Scholars, students, and visitors

---
*Note: The provided context aligns with the architectural and functional details of the British Library’s St Pancras building, but the photo timestamp (2025-12-13) is hypothetical and not verifiable without real-time data. The visual content is consistent with known architectural features.*
```

---

### ✅ mlx-community/gemma-3n-E4B-it-bf16

**Metrics:** 41.9 TPS | 409 tokens

```text
## Visual Description

The image is a high-angle, wide shot of the grand entrance hall of the British Library in London. The hall is spacious and multi-level, characterized by a modern architectural design. The dominant colors are warm tones of red brick, beige stone, and white.

The central area features a patterned floor of alternating dark and light tiles arranged in a geometric design. Several people are scattered throughout the hall, engaged in various activities such as working on laptops, reading, or simply relaxing. Some individuals are seated on benches or at tables, while others are standing.

The architecture is notable for its soaring ceilings, supported by large, white pillars. The walls are a mix of red brick and smooth, light-colored surfaces. A prominent feature is a large, curved brick wall that rises several stories. Staircases with glass railings connect the different levels, and escalators provide additional access.

Natural light filters in from above, illuminating the space, while artificial lighting is also present. The overall atmosphere is one of quiet activity and scholarly pursuit.

## Contextual Description

The British Library, located in London, UK, is a world-renowned research library and a significant cultural institution. Its St Pancras building, designed by architect Sir Norman Foster and opened in 1997, is considered a landmark of 20th-century British architecture. The library houses a vast collection of books, manuscripts, and other materials, making it a vital resource for researchers and the public alike.

The entrance hall serves as a welcoming and functional space for visitors, providing areas for reading, studying, and accessing different parts of the library. The design of the hall reflects the library's role as both a repository of knowledge and a public gathering place.

## Keywords

British Library, London, UK, architecture, entrance hall, modern, library, research, public space, St Pancras, Sir Norman Foster, interior, people, reading, studying, knowledge, cultural institution.
```

---

### ✅ mlx-community/llava-v1.6-mistral-7b-8bit

**Metrics:** 46 TPS | 15 tokens

```text
The British Library in London, England, United Kingdom, UK.
```

---

### ✅ mlx-community/paligemma2-10b-ft-docci-448-6bit

**Metrics:** 41.5 TPS | 500 tokens

```text
A high-angle view of the interior of the British Library in London, England. The library is a large, modern building with a tall, sloped red brick wall in the center of the room. The wall is angled from the bottom left corner of the image to the top right corner of the image. There is a gray stone staircase on the right side of the wall that leads up to the second floor of the library. There are gray stone pillars on the left and right sides of the staircase. There are gray stone pillars on the left and right sides of the staircase. There are gray stone pillars on the left and right sides of the staircase. There are gray stone pillars on the left and right sides of the staircase. There are gray stone pillars on the left and right sides of the staircase. There are gray stone pillars on the left and right sides of the staircase. There are gray stone pillars on the left and right sides of the staircase. There are gray stone pillars on the left and right sides of the staircase. There are gray stone pillars on the left and right sides of the staircase. There are gray stone pillars on the left and right sides of the staircase. There are gray stone pillars on the left and right sides of the staircase. There are gray stone pillars on the left and right sides of the staircase. There are gray stone pillars on the left and right sides of the staircase. There are gray stone pillars on the left and right sides of the staircase. There are gray stone pillars on the left and right sides of the staircase. There are gray stone pillars on the left and right sides of the staircase. There are gray stone pillars on the left and right sides of the staircase. There are gray stone pillars on the left and right sides of the staircase. There are gray stone pillars on the left and right sides of the staircase. There are gray stone pillars on the left and right sides of the staircase. There are gray stone pillars on the left and right sides of the staircase. There are gray stone pillars on the left and right sides of the staircase. There are gray stone pillars on the left and right sides of the staircase. There are gray stone pillars on the left and right sides of the staircase. There are gray stone pillars on the left and right sides of the staircase. There are gray stone pillars on the left and right sides of the staircase. There are gray stone pillars on the left and right sides of the staircase. There are gray stone pillars on the left and
```

---

### ✅ mlx-community/gemma-3-27b-it-qat-4bit

**Metrics:** 28.2 TPS | 338 tokens

```text
## Image Catalogue Information:

**Caption:** Interior view of the British Library’s St Pancras Reading Rooms, London, showing visitors studying and reading in the expansive hall.

**Description:** A high-angle, wide-shot photograph of the main entrance hall of the British Library’s St Pancras building. The image captures the multi-level interior space illuminated in the late afternoon light. Red brick walls are prominent, contrasted with white balconies and a geometric tiled floor. People are visible working on laptops, reading, and relaxing at tables throughout the hall. A large digital display screen is visible in the background. The architectural design emphasizes open space and natural light. The photograph was taken on December 13, 2025.

**Keywords:**

*   British Library
*   St Pancras
*   London
*   England
*   United Kingdom
*   Library
*   Architecture
*   Interior
*   Reading Room
*   Public Space
*   Study Space
*   Colin St John Wilson
*   Modernist Architecture
*   Red Brick
*   Digital Display
*   December
*   2025
*   51.529533°N, 0.127433°W
*   Learning
*   Research
*   Hall
*   Open Plan
*   Geometric Design
*   Public Building
*   Knowledge
*   Books
*   Students
*   Scholars
*   Visitors
*   Late Afternoon
*   Illumination
*   Multi-level
*   Tiled Floor




```

---

### ✅ mlx-community/pixtral-12b-8bit

**Metrics:** 34.2 TPS | 319 tokens

```text
### Caption
Interior view of the British Library's grand entrance hall in London, showcasing its modernist architectural design and the studious atmosphere as visitors engage in various activities.

### Description
The image captures the expansive, multi-level entrance hall of the British Library in London, illuminated by soft lighting as the day transitions into evening. The hall features striking architectural elements, including red brick walls, white balconies, and geometric floor patterns. Visitors are seen working on laptops, reading, or relaxing, highlighting the library's role as both a research hub and a public space for learning and reflection. The design, by architect Colin St John Wilson, combines historical and modern elements, creating a warm and inviting atmosphere.

### Keywords
- British Library
- London
- Interior architecture
- Modernist design
- Reading room
- Public library
- Research hub
- Architectural design
- Colin St John Wilson
- St Pancras building
- Study space
- Learning environment
- Geometric patterns
- Red brick
- White balconies
- Evening lighting
- Scholars
- Students
- Public space
- Reflection
- Knowledge
- UK public building
- 20th-century architecture
- Winter afternoon
- December
- Dusk
- Illumination
- St Pancras
- GPS 51.529533° N, 0.127433° W
- 2025-12-13 16:27:09 GMT
```

---

### ✅ mlx-community/Idefics3-8B-Llama3-bf16

**Metrics:** 29.1 TPS | 364 tokens

```text
 The image depicts the interior of the British Library in London, England, showcasing its grand entrance hall during the late afternoon. The library's modernist architecture is characterized by a combination of red brick, white balconies, and geometric floor patterns. The space is filled with natural light streaming through the large windows, creating a warm and inviting atmosphere.

In the foreground, the floor is composed of large, dark tiles arranged in a grid pattern. The central area of the hall features a large, open space with a few people seated on benches, engaged in reading or working on laptops. The seating areas are interspersed with potted plants, adding a touch of greenery to the space.

The upper levels of the hall are connected by a series of white balconies, which provide a sense of depth and height to the space. The balconies are lined with people, some of whom are standing and others who are seated, all engaged in various activities such as reading or conversing.

The walls of the hall are adorned with large, rectangular windows that allow natural light to flood the space. The windows are framed by white frames, which contrast with the red brick walls. The ceiling is high and features a series of large, geometric patterns that add to the modernist aesthetic of the building.

In the background, there is a staircase leading to the upper levels of the library. The staircase is flanked by red brick walls and white railings, matching the overall color scheme of the building. The overall atmosphere of the image is one of calm and contemplation, with the library's vast collection of knowledge and resources available to its visitors.

The image captures the essence of the British Library as a place of learning, research, and intellectual curiosity, where people from all walks of life come to explore and discover new ideas.<end_of_utterance>
```

---

### ✅ mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

**Metrics:** 26.9 TPS | 252 tokens

```text
Factual Caption:
Interior view of the grand entrance hall of the British Library, London, England, showcasing its modernist architecture with red brick walls, white balconies, and geometric floor patterns. The multi-level space is filled with visitors engaged in reading, studying, and working on laptops, creating a studious and welcoming atmosphere.

Description:
The image captures the expansive, multi-level interior of the British Library's St Pancras building. The architecture features a striking combination of red brick walls, white balconies, and a geometric floor pattern. The space is well-lit, highlighting the modernist design elements. Visitors are seen throughout the hall, some seated at tables working on laptops, others reading or relaxing. The scene reflects the library's role as a hub for research and a public space for learning and reflection.

Keywords:
British Library, London, England, United Kingdom, UK, modernist architecture, red brick, white balconies, geometric floor, multi-level space, visitors, scholars, students, laptops, reading, studying, public space, learning, reflection, St Pancras building, Colin St John Wilson, 20th century architecture, library interior, study area, quiet contemplation, research hub.
```

---

### ✅ mlx-community/pixtral-12b-bf16

**Metrics:** 19.4 TPS | 299 tokens

```text
### Caption
Late winter afternoon at the British Library, London, England, UK. The grand entrance hall is bathed in warm light, showcasing its modernist architecture with red brick, white balconies, and geometric floor patterns. Visitors are seen working, reading, and relaxing in the expansive, multi-level space.

### Description
The image captures the interior of the British Library's St Pancras building in London, England. The grand entrance hall is illuminated, highlighting the striking architectural features such as the red brick walls, white balconies, and geometric floor patterns. The space is multi-leveled, with several floors visible, each adorned with study areas and seating arrangements. Scholars, students, and the public are seen engaged in various activities like working on laptops, reading, and contemplating. The atmosphere is studious and welcoming, reflecting the library's role as both a research hub and a public space for learning and reflection.

### Keywords
- British Library
- London
- England
- United Kingdom
- Interior architecture
- Modernist design
- Red brick
- White balconies
- Geometric floor patterns
- Study space
- Research hub
- Public library
- Scholars
- Students
- Reading
- Relaxation
- St Pancras building
- Colin St John Wilson
- 20th-century architecture
- Late winter afternoon
- Illumination
- Multi-level space
- Learning environment
- Reflection
```

---

### ✅ mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX

**Metrics:** 34.7 TPS | 500 tokens

```text
Here are my reasoning steps:
We need to produce a factual caption, description, and keywords suitable for cataloguing or searching for the image. The user gave a context: British Library, London, England, UK, late winter afternoon, interior grand entrance hall, modernist interior, interplay of red brick, white balconies, geometric floor patterns, people working, reading, relaxing, etc. The photo taken around 2025-12-13 16:27:09 GMT from GPS coordinates 51.529533° N, 0.127433° W. The instruction: "Provide a factual caption, description, and keywords suitable for cataloguing, or searching for, the image. Focus on visual content, drawing on any available contextual information for specificity. Do not speculate."

Thus we need to produce three parts: caption, description, keywords. The caption should be concise, summarizing the image. The description should be more detailed, describing visual elements, architecture, people, lighting, etc. Keywords should be a list of terms for cataloguing/search.

We must not add speculation beyond what is given. The context includes details: British Library, St Pancras building, designed by Colin St John Wilson, opened 1997, largest UK public building constructed in 20th century, houses vast collection. The interior features red brick, white balconies, geometric floor patterns, multi-level space, people working on laptops, reading, relaxing. The time: late winter afternoon, December day fading to dusk, interior illuminated. The photo taken around 2025-12-13 16:27:09 GMT from GPS coordinates 51.529533° N, 0.127433° W.

We need to incorporate these details in the caption and description. The caption should be something like: "Interior atrium of the British Library, St Pancras, London, showing multi-level reading space with red brick walls, white balconies, geometric floor pattern, and visitors using laptops and reading on a late winter afternoon." The description should be more thorough: mention the architecture, lighting, materials, layout, people, signage, etc. Keywords: British Library, St Pancras, London, interior atrium, reading room, red brick, white balconies, geometric floor pattern
```

---

### ✅ mlx-community/paligemma2-10b-ft-docci-448-bf16

**Metrics:** 5.07 TPS | 120 tokens

```text
A high-angle view of the interior of the British Library in London, England. The library is a large, open space with a tall ceiling and a large red brick wall in the center of the room. There is a staircase leading up to the second floor in the center of the room. There are people sitting on the ground and on the second floor of the library. There is a blue sign on the wall in the background of the red brick wall with white text that reads "GEOG / MAP". There is a red circle on the floor in the bottom right corner of the image.
```

---

### ✅ mlx-community/gemma-3-27b-it-qat-8bit

**Metrics:** 15.5 TPS | 375 tokens

```text
Here's a factual caption, description, and keywords for cataloguing the image, based on the provided information:

**Caption:**

The grand entrance hall of the British Library, St Pancras, London, illuminated in the late afternoon light, December 13, 2025.

**Description:**

A high-angle, wide-shot photograph of the interior of the British Library’s St Pancras building. The image depicts a multi-level, expansive hall with a modernist architectural style. Key features include red brick walls, white balconies, and a geometric tiled floor. Visitors are visible working on laptops, reading, and relaxing at tables throughout the space. A large digital display screen is visible in the center of the hall. The building is brightly lit, creating a warm and studious atmosphere. The photograph was taken around 16:27:09 GMT at GPS coordinates 51.529533°N, 0.127433°W.

**Keywords:**

*   British Library
*   St Pancras
*   London
*   England
*   United Kingdom
*   UK
*   Library
*   Architecture
*   Modernism
*   Interior
*   Hall
*   Reading Room
*   Study Space
*   Public Space
*   Colin St John Wilson
*   1997 (Building Opening Date)
*   December
*   Winter
*   Afternoon
*   2025
*   Digital Display
*   Red Brick
*   Geometric Patterns
*   Study
*   Research
*   Learning
*   51.529533°N, 0.127433°W (GPS Coordinates)




```

---

### ✅ mlx-community/paligemma2-3b-ft-docci-448-bf16

**Metrics:** 18 TPS | 500 tokens

```text
A large, square, tiled floor is visible in the center of the frame. The floor is divided into four squares by a thick white border. The squares alternate between a light gray and dark gray. The squares alternate between a light gray and dark gray. The squares alternate between a light gray and dark gray. The squares alternate between a light gray and dark gray. The squares alternate between a light gray and dark gray. The squares alternate between a light gray and dark gray. The squares alternate between a light gray and dark gray. The squares alternate between a light gray and dark gray. The squares alternate between a light gray and dark gray. The squares alternate between a light gray and dark gray. The squares alternate between a light gray and dark gray. The squares alternate between a light gray and dark gray. The squares alternate between a light gray and dark gray. The squares alternate between a light gray and dark gray. The squares alternate between a light gray and dark gray. The squares alternate between a light gray and dark gray. The squares alternate between a light gray and dark gray. The squares alternate between a light gray and dark gray. The squares alternate between a light gray and dark gray. The squares alternate between a light gray and dark gray. The squares alternate between a light gray and dark gray. The squares alternate between a light gray and dark gray. The squares alternate between a light gray and dark gray. The squares alternate between a light gray and dark gray. The squares alternate between a light gray and dark gray. The squares alternate between a light gray and dark gray. The squares alternate between a light gray and dark gray. The squares alternate between a light gray and dark gray. The squares alternate between a light gray and dark gray. The squares alternate between a light gray and dark gray. The squares alternate between a light gray and dark gray. The squares alternate between a light gray and dark gray. The squares alternate between a light gray and dark gray. The squares alternate between a light gray and dark gray. The squares alternate between a light gray and dark gray. The squares alternate between a light gray and dark gray. The squares alternate between a light gray and dark gray. The squares alternate between a light gray and dark gray. The squares alternate between a light gray and dark gray. The squares alternate between a light gray and dark gray. The squares alternate between a light gray and dark gray. The squares alternate between a light gray and dark gray. The squares alternate between a light gray and dark
```

---

### ✅ mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

**Metrics:** 8.46 TPS | 219 tokens

```text
I'm happy to help you with your question, but I want to clarify that I'm a large language model, I don't have the capability to access real-time information or browse the internet. I can only provide information based on my training data, which may not be up-to-date or accurate.

That being said, I can try to provide a general response to your question. The image you provided appears to be a photograph of a large, open space with a high ceiling and a large staircase. The space appears to be well-lit and has a modern, industrial feel to it.

If you're looking for more information about the location or the image, I'd be happy to try to help you brainstorm some questions or provide some general information about the image. However, I won't be able to provide a specific answer to your question about the location of the image.

If you have any more information about the image, such as the location or the context in which it was taken, I may be able to provide more specific information or help you find the information you're looking for.
```

---

### ✅ mlx-community/GLM-4.6V-Flash-6bit

**Metrics:** 44 TPS | 500 tokens

```text
<think>Got it, let's tackle this. The user wants a factual caption, description, and keywords for cataloguing.

First, the caption needs to be concise but include key info. The description should elaborate on the visual elements and context. Keywords should be specific and cover location, building, features, time, etc.

Let's start with the caption. It should mention the British Library, London, and the grand entrance hall. Maybe include the time context.

Caption: "Grand entrance hall of the British Library, London, England, United Kingdom, captured on a late winter afternoon (December) in 2025."

Wait, the context says the photo was taken around 2025-12-13. So the caption should reflect that.

Now the description. Need to cover the visual elements: multi-level space, red brick walls, white balconies, geometric floor patterns (checkered or grid), escalators, people working/studying/relaxing, modernist architecture, lighting (warm, dusk), context of British Library, designed by Colin St John Wilson, opened 1997, largest 20th-century UK public building.

So structure the description to flow from the overall space to specific features and human activity.

Keywords: Need to be specific. British Library, London, England, UK, grand entrance hall, multi-level interior, red brick wall, white balconies, geometric floor pattern, escalator, people working/studying, modernist architecture, warm lighting, dusk, December, 2025, Colin St John Wilson, St Pancras building, largest 20th-century UK public building, research hub, public space.

Now, compile the caption, description, and keywords. Make sure they are factual and based on the provided context and visual content.</think>
<|begin_of_box|>### Caption  
Grand entrance hall of the British Library, London, England, United Kingdom, captured on a late winter afternoon (December) in 2025.  


### Description  
The image depicts the expansive, multi-level grand entrance hall of the British Library in London. The space features a striking interplay of architectural elements: red brick walls (including a prominent curved brick feature), white balconies with light-colored railings, and a geometric floor pattern of alternating dark and light tiles.  

Modernist design elements are evident, such as the high, curved ceiling with recessed lighting and skylights, and structural columns with light-colored bases. An escalator and staircases connect the multiple
```

---

### ✅ mlx-community/Molmo-7B-D-0924-8bit

**Metrics:** 45.6 TPS | 232 tokens

```text
 Caption:
A late winter afternoon in London, the grand entrance hall of the British Library provides a warm and studious atmosphere for its visitors. As the December day fades to dusk, the building's modernist interior is illuminated, highlighting the striking interplay of red brick, white balconies, and geometric floor patterns.

Description:
The image showcases the expansive, multi-level entrance hall of the British Library in St Pancras, London. The space is characterized by its modernist architecture, featuring a high, arched ceiling with a skylight, red brick walls, and white balconies on the second and third floors. The floor is adorned with a distinctive geometric pattern of white and black tiles. The hall is bustling with activity, as visitors work on laptops, read, or relax in quiet contemplation.

Keywords:
British Library, London, England, United Kingdom, UK, St Pancras, Modernist architecture, Reading room, Research hub, Public space, Winter scene, December, Dusk, Natural lighting, Red brick, White balconies, Geometric floor pattern, Study area, Knowledge hub, Public library, London landmark, Architectural photography
```

---

### ✅ mlx-community/Molmo-7B-D-0924-bf16

**Metrics:** 28.5 TPS | 233 tokens

```text
 Caption:
A late winter afternoon in London, the grand entrance hall of the British Library provides a warm and studious atmosphere for its visitors. As the December day fades to dusk, the building's modernist interior is illuminated, highlighting the striking interplay of red brick, white balconies, and geometric floor patterns.

Description:
The image showcases the expansive, multi-level entrance hall of the British Library in St Pancras, London. The space is characterized by its modernist architecture, featuring a high, arched ceiling with a skylight, red brick walls, and white balconies on the second and third floors. The floor is adorned with a distinctive geometric pattern of white and black tiles. The hall is bustling with activity, as visitors work on laptops, read, or relax in quiet contemplation.

Keywords:
British Library, London, England, United Kingdom, UK, St Pancras, Modernist architecture, Reading room, Research hub, Public space, Winter, Dusk, Natural lighting, Red brick, White balconies, Geometric floor pattern, Study area, Knowledge hub, Public library, Colin St John Wilson, 1997
```

---

### ✅ Qwen/Qwen3-VL-2B-Instruct

**Metrics:** 72.3 TPS | 350 tokens

```text
### Factual Caption
A wide-angle view of the grand, multi-level atrium of the British Library in London, showcasing its modernist architecture and the bustling activity of visitors. The space features a striking red brick wall, white balconies, and a geometrically patterned floor, with people working on laptops and reading in the open space.

### Description
The image captures the expansive, multi-level atrium of the British Library in London, a modernist architectural marvel. The central feature is a large, sweeping staircase with a red brick wall and a series of circular plaques, leading to upper levels. The floor is paved with a geometric pattern of light and dark tiles, and the space is illuminated by a combination of natural light from skylights and artificial lighting. The upper levels are lined with white balconies, and the lower level is filled with people working on laptops, reading, or relaxing. The overall atmosphere is one of quiet study and intellectual engagement.

### Keywords
- British Library
- London
- United Kingdom
- Architecture
- Modernist
- Atrium
- St Pancras
- Library
- Research
- Public Space
- Geometric Pattern
- Red Brick Wall
- White Balconies
- Laptop
- Study
- Intellectual Engagement
- Multi-level
- Skylights
- Natural Light
- Artificial Lighting
- People
- Scholars
- Students
- Members of the Public
- Open Space
- Study Area
- Work
- Reading
- Reflection
- 1997
- Colin St John Wilson
- GPS 51.529533°N, 0.127433°W
```

---

### ✅ mlx-community/Qwen3-VL-2B-Thinking-bf16

**Metrics:** 72.4 TPS | 500 tokens

```text
Got it, let's tackle this. First, I need to create a factual caption, description, and keywords for the image. The context given is about the British Library in London, so I should start by recalling the key elements from the context.

First, the caption should be factual, so it needs to include the location, time, and main features. The context says it's a late winter afternoon, so the time is around December (since it's late winter), 2025-12-13. The location is the British Library, London, UK. The description should detail the interior: multi-level, modernist design, red brick, white balconies, geometric floor patterns, people working, reading, etc. Keywords need to be specific for cataloging, so things like British Library, London, interior hall, modernist architecture, red brick, escalator, multi-level, scholarly activity, etc.

Let me check the context again. The context mentions the St Pancras building, designed by Colin St John Wilson, opened in 1997. The photo was taken around 2025-12-13, so the date is in the future, but we can use that as a timestamp. The image shows the grand entrance hall, with people working on laptops, reading, relaxing. The architectural elements: white balconies, red brick wall, geometric floor patterns, escalator, multiple levels.

So for the caption: "Grand entrance hall of the British Library, London, UK, captured during late winter afternoon (December 13, 2025) showcasing modernist architecture with red brick, white balconies, and geometric floor patterns. Scholars and visitors engage in research, reading, and quiet contemplation amidst the spacious, multi-level interior."

Description: A panoramic view of the British Library's St Pancras building entrance hall, featuring a multi-level structure with white balconies, red brick walls, and geometric tile flooring. The space includes an escalator leading to upper levels, numerous tables and chairs where people work on laptops or read, and a large blue banner with "SECR MAP" visible on the wall. The interior is well-lit with a combination of natural and artificial lighting, highlighting the architectural details and the active scholarly environment.

Keywords: British Library, London, UK, St Pancras Building, Modernist Architecture, Red Brick, White Balconies, Geometric Floor Patterns, Escalator
```

---

### ✅ meta-llama/Llama-3.2-11B-Vision-Instruct

**Metrics:** 3.89 TPS | 356 tokens

```text
The image depicts the interior of the British Library, a prominent cultural institution in London, England. The photograph showcases the building's modernist architecture, featuring a spacious atrium with a high ceiling and a large, open area with a central staircase. The image highlights the building's design, which includes a large, open space with a high ceiling and a central staircase, as well as a variety of seating areas and reading spaces.

* A large, open area with a high ceiling and a central staircase:
	+ The space is well-lit, with natural light pouring in through the large windows.
	+ The staircase is a prominent feature, with a sleek and modern design.
	+ The area is bustling with people, with some sitting at tables or reading in the seating areas.
* A large, open space with a high ceiling and a central staircase:
	+ The space is filled with natural light, which creates a bright and welcoming atmosphere.
	+ The staircase is a focal point, with a modern and sleek design that adds to the overall aesthetic of the space.
	+ The area is filled with people, with some working on laptops or reading books.
* A variety of seating areas and reading spaces:
	+ The seating areas are comfortable and inviting, with a variety of chairs and tables.
	+ The reading spaces are well-lit and quiet, with plenty of books and materials available.
	+ The areas are popular with students and researchers, who come to the library to study and work.

Overall, the image provides a glimpse into the daily life of the British Library, showcasing its modernist architecture and its role as a hub for learning and research. The image highlights the importance of the library as a place for people to come together, learn, and explore.
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
- `mlx`: `0.30.3.dev20251219+d9f4d8d5`
- `mlx-lm`: `0.30.0`
- `mlx-vlm`: `0.3.10`
- `tokenizers`: `0.22.1`
- `transformers`: `5.0.0rc1`

_Report generated on: 2025-12-19 22:43:23 GMT_
