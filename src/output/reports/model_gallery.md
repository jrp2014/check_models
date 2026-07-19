# Model Output Gallery

Generated on: 2026-07-19 20:37:23 BST

Complete per-model evidence artifact with image metadata, the source prompt,
summary tables, diagnostics, and full generated output for every attempted
model.

_Action Snapshot: see [results.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/results.md) for the full summary._

## Run Stamps

- `mlx-vlm`: `0.6.5`
- `mlx`: `0.32.1.dev20260719+b7c3dd6d`
- `mlx-lm`: `0.31.3`
- `transformers`: `5.14.1`
- `tokenizers`: `0.22.2`
- `huggingface-hub`: `1.24.0`
- _Python Version:_ 3.13.13
- _OS:_ Darwin 25.5.0
- _macOS Version:_ 26.5.2
- _GPU/Chip:_ Apple M5 Max
- _MLX Device:_ Apple M5 Max
- _GPU Architecture:_ applegpu_g17s
- _RAM:_ 128.0 GB
- _Recommended Working Set:_ 108 GB
- _Fused Attention:_ Available

## Image Metadata

- _Title:_ , The Fenchurch Building (The Walkie-Talkie), London, England, UK,
  GBR, Europe
- _Description:_ Walkie Talkie building known formally as 20 Fenchurch Street.
- _Keywords:_ Architecture, Building, Buildings, Cars, City, Cityscape,
  Commuting, England, Europe, Fenchurch Street, Illuminated, London, Modern,
  Night, Nightscape, Skyscraper, Street, Street signs, The Fenchurch Building
  (The Walkie-Talkie), UK, Urban, Urban landscape, Walkie Talkie building,
  Windows, bicycles, blue hour, british, business, capital, commercial,
  contemporary, corporate, dark, design, district, downtown, english, evening,
  exterior, facade, fchurch street, financial, glass, great britain, high,
  landmark, lights, looking up, office, perspective, scenic, skyline,
  skyscrapers, streetlights, structure, tall, tower, twilight, twilight sky,
  united kingdom, view, walkie talkie
- _Date:_ 2026-07-18 22:55:39 BST
- _Time:_ 22:55:39
- _GPS:_ 51.511300°N, 0.083400°W

## Prompt

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Analyze this image for cataloguing metadata, using British English.
>
> Describe visible details faithfully. If a visual detail is uncertain,
> ambiguous, partially obscured, or too small to verify, leave it out rather
> than guessing.
>
> Use authoritative context as supplied fact, and treat the descriptive
> metadata as a draft catalog record. Retain draft details that are consistent
> with the image, correct contradictions, and add important visible details.
> Authoritative context may supply identity and location even when they are
> not visually readable.
>
> &#8203;Return exactly these three sections, and nothing else:
>
> &#8203;Title:
> &#45; 5-10 words, concrete and factual; authoritative context may supply
> identity and location.
> &#45; Output only the title text after the label.
> &#45; Do not repeat or paraphrase these instructions in the title.
>
> &#8203;Description:
> &#45; 1-2 factual sentences combining supplied authoritative context with the
> main visible subject, setting, lighting, action, and distinctive visible
> details.
> &#45; Output only the description text after the label.
>
> &#8203;Keywords:
> &#45; 10-18 unique comma-separated terms covering supplied authoritative context
> and clearly visible subjects, setting, colors, composition, and style.
> &#45; Output only the keyword list after the label.
>
> &#8203;Rules:
> &#45; Distinguish supplied authoritative facts from visible details; do not
> present contextual facts as though they were read from the image.
> &#45; Reuse draft metadata when it is consistent with the image; authoritative
> context does not require separate visual proof.
> &#45; If metadata and image disagree, follow the image.
> &#45; Prefer omission to speculation.
> &#45; Do not copy prompt instructions into the Title, Description, or Keywords
> fields.
> &#45; Do not infer identity, location, event, brand, species, time period, or
> intent unless supplied as authoritative context or visually obvious.
> &#45; Do not output reasoning, notes, hedging, or extra sections.
>
> Context: Authoritative context:
> &#45; Location terms: England, Europe, UK, district, united kingdom
> &#45; Capture date/time: 2026-07-18 22:55:39 BST 22:55:39
> &#45; GPS: 51.511300°N, 0.083400°W
> &#45; Use this factual context where it improves the catalogue record; do not
> claim that contextual facts are visually observable.
>
> &#8203;Draft descriptive metadata:
> &#45; Existing title: The Fenchurch Building (The Walkie-Talkie), London,
> England, UK, GBR, Europe
> &#45; Existing description: Walkie Talkie building known formally as 20
> Fenchurch Street.
> &#45; Existing keywords: Architecture, Building, Buildings, Cars, City,
> Cityscape, Commuting, Fenchurch Street, Illuminated, London, Modern, Night,
> Nightscape, Skyscraper, Street, Street signs, The Fenchurch Building (The
> Walkie-Talkie), Urban, Urban landscape, Walkie Talkie building
> &#45; Treat this draft as fallible. Retain supported details, correct errors,
> and add important visible information.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

## Model Output and Cost Summary

Every model in this run, with a concise output preview or failure diagnostic beside runtime, memory, and quality signals. Complete output appears once in the expandable per-model evidence below.

<!-- markdownlint-disable MD034 MD049 -->

| Model                                                                                                                   | Result                            | Output preview / diagnostic                                                                                                                                                                                                                                                                          | Gen tok   | Total   | Gen TPS   | Peak GB                                          | Quality signal                                                                                                               |
|-------------------------------------------------------------------------------------------------------------------------|-----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|---------|-----------|--------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| [`mlx-community/Molmo-7B-D-0924-8bit`](#model-mlx-community-molmo-7b-d-0924-8bit)                                       | `not evaluated` / `indeterminate` | [unknown; network-error] Error: Network Error - Model loading failed: [Errno 54] Connection reset by peer                                                                                                                                                                                            | -         | 0.29s   | -         |                                                  | unknown; network-error                                                                                                       |
| [`mlx-community/Step-3.7-Flash-oQ2e`](#model-mlx-community-step-37-flash-oq2e)                                          | `avoid` / `runtime failure`       | [model-config; processor-error] Error: Processor Error - Model preflight failed for mlx-community/Step-3.7-Flash-oQ2e: Loaded processor has no image_processor; expected multimodal processor.                                                                                                       | -         | 11.39s  | -         |                                                  | model-config; processor-error                                                                                                |
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `caveat` / `clean`                | Title: Fenchurch Building (The Walkie-Talkie), London, England, UK, GBR, Europe Description: The Fenchurch Building, formally known as 20 Fenchurch Street, is a prominent skyscraper in London, England, United Kingdom. It is a modern architectural structure that stands out in t...             | 188       | 1.47s   | 509       | 1.5 GB (1.31% of 108 GB recommended working set) | low-draft-improvement                                                                                                        |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                            | `caveat` / `clean`                | Title: The Fenchurch Building (The Walkie-Talkie), London, England, UK, GBR, Europe Description: Walkie Talkie building known formally as 20 Fenchurch Street. Keywords: Architecture, Building, Buildings, Cars, City, Cityscape, Commuting, Fenchurch Street, Illuminated, London,...              | 112       | 2.25s   | 346       | 2.8 GB (2.42% of 108 GB recommended working set) | context-echo; low-draft-improvement                                                                                          |
| [`mlx-community/LFM2-VL-1.6B-8bit`](#model-mlx-community-lfm2-vl-16b-8bit)                                              | `caveat` / `clean`                | Title: The Fenchurch Building (The Walkie-Talkie), London, England, UK, GBR, Europe Description: The Fenchurch Building, also known as 20 Fenchurch Street, is a prominent skyscraper located in the heart of London, England. This modern architectural marvel stands tall against t...             | 201       | 1.71s   | 331       | 3.0 GB (2.58% of 108 GB recommended working set) | low-draft-improvement                                                                                                        |
| [`HuggingFaceTB/SmolVLM-Instruct`](#model-huggingfacetb-smolvlm-instruct)                                               | `avoid` / `model shortcoming`     | The Fenchurch Street, London, England, UK, GBR, Europe.                                                                                                                                                                                                                                              | 18        | 2.15s   | 131       | 5.7 GB (4.96% of 108 GB recommended working set) | low-draft-improvement; missing-sections                                                                                      |
| [`mlx-community/SmolVLM-Instruct-bf16`](#model-mlx-community-smolvlm-instruct-bf16)                                     | `avoid` / `model shortcoming`     | The Fenchurch Street, London, England, UK, GBR, Europe.                                                                                                                                                                                                                                              | 18        | 2.95s   | 124       | 5.7 GB (4.97% of 108 GB recommended working set) | low-draft-improvement; missing-sections                                                                                      |
| [`qnguyen3/nanoLLaVA`](#model-qnguyen3-nanollava)                                                                       | `avoid` / `model shortcoming`     | Title: The Walkie-Talkie Building (The Walkie-Talkie), London, England, UK, GBR, Europe Description: A tall, tall building with a glass facade, the name "The Walkie-Talkie" written on its side. The building is located in London, England, UK, and is part of the Walkie-Talkie di...             | 91        | 2.51s   | 113       | 4.7 GB (4.08% of 108 GB recommended working set) | low-draft-improvement; missing-sections                                                                                      |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `avoid` / `model shortcoming`     | Title: The Fenchurch Building (The Walkie-Talkie), London, England, UK, GBR, Europe Description: The Fenchurch Building, also known as 20 Fenchurch Street, is a prominent example of modernist architecture in London, UK. The building is a 20-story skyscraper with a distinctive...              | 179       | 2.47s   | 343       | 2.2 GB (1.88% of 108 GB recommended working set) | low-draft-improvement; missing-sections                                                                                      |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `recommended` / `clean`           | Title: The Fenchurch Building in London at night Description: The curved glass facade of the Fenchurch Building towers over a dark urban street in London at night. Street lamps and building windows provide points of light against the deep blue night sky. Keywords: Architecture...             | 101       | 5.10s   | 123       | 17 GB (14.3% of 108 GB recommended working set)  | clean                                                                                                                        |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `caveat` / `clean`                | Title: The Fenchurch Building (The Walkie-Talkie), London, England, UK, GBR, Europe Description: The Walkie-Talkie building, formally known as 20 Fenchurch Street, is a modern skyscraper in the Fenchurch Street district of London, England, UK. It is illuminated at night and is...             | 158       | 3.26s   | 132       | 5.5 GB (4.75% of 108 GB recommended working set) | low-draft-improvement                                                                                                        |
| [`mlx-community/paligemma2-10b-ft-docci-448-6bit`](#model-mlx-community-paligemma2-10b-ft-docci-448-6bit)               | `avoid` / `model shortcoming`     | - Do not copy prompt instructions into the Title, Description, or Keywords fields.                                                                                                                                                                                                                   | 17        | 4.32s   | 31.0      | 12 GB (10.5% of 108 GB recommended working set)  | low-draft-improvement; context-ignored; missing-sections                                                                     |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `caveat` / `clean`                | Title: The Walkie-Talkie Building at Night, Fenchurch Street, London Description: A modern skyscraper with a distinctive, asymmetrical design and glass facade dominates the night scene, its lights casting reflections on the surrounding urban streets. The area around 20 Fenchur...             | 151       | 3.70s   | 185       | 7.8 GB (6.76% of 108 GB recommended working set) | missing terms: Commuting, Nightscape, Street signs, The Fenchurch Building (The Walkie-Talkie), Urban landscape; keywords=22 |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `caveat` / `model shortcoming`    | <\|channel>thought <channel\|>Title: - 20 Fenchurch Street building at night in London Description: - The glass facade of 20 Fenchurch Street towers over a narrow urban street at dusk. The scene features modern architecture illuminated by street lamps and interior office windows...           | 104       | 6.65s   | 66.5      | 29 GB (25.4% of 108 GB recommended working set)  | missing-sections; formatting                                                                                                 |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `caveat` / `model shortcoming`    | The Fenchurch Building (The Walkie-Talkie), London, England, UK, GBR, Europe.<end_of_utterance>                                                                                                                                                                                                      | 26        | 4.97s   | 33.4      | 19 GB (16.6% of 108 GB recommended working set)  | low-draft-improvement; missing-sections; formatting                                                                          |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `caveat` / `model shortcoming`    | <\|channel>thought <channel\|>Title: - 20 Fenchurch Street building at night Description: - The curved skyscraper of 20 Fenchurch Street, known as the Walkie-Talkie, rises over a dark London street illuminated by streetlights and modern glass buildings. Keywords: - Architecture,...           | 100       | 6.56s   | 62.7      | 28 GB (24.6% of 108 GB recommended working set)  | low-draft-improvement; missing-sections; formatting                                                                          |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `recommended` / `clean`           | Title: The Walkie-Talkie Building, London, England Description: Modern skyscraper in London at night, illuminated with reflective glass facade, streetlights casting a warm glow, and a few cars on the road. Keywords: Architecture, Building, Buildings, Cars, City, Cityscape, Ill...             | 90        | 5.99s   | 34.2      | 18 GB (15.3% of 108 GB recommended working set)  | low-draft-improvement                                                                                                        |
| [`mlx-community/gemma-3n-E2B-4bit`](#model-mlx-community-gemma-3n-e2b-4bit)                                             | `avoid` / `cutoff degraded`       | - Do not take it is a 2026-07-18 22:55:39 22:55:39 - Do not take it is a 22:55:39 22:55:39 - Do not take it is a 22:55:39 22:55:39 - Do not take it is a 22:55:39 22:55:39 - Do not take it is a 22:55:39 22:55:39 - Do not take it is a 22:55:39 22:55:39 - Do not take it is a 22:5...             | 500       | 7.22s   | 123       | 6.0 GB (5.21% of 108 GB recommended working set) | repetitive; unverified-context-copy; low-draft-improvement                                                                   |
| [`meta-llama/Llama-3.2-11B-Vision-Instruct`](#model-meta-llama-llama-32-11b-vision-instruct)                            | `avoid` / `model shortcoming`     | The Fenchurch Building (The Walkie-Talkie)                                                                                                                                                                                                                                                           | 14        | 7.52s   | 5.39      | 25 GB (21.9% of 108 GB recommended working set)  | low-draft-improvement; missing-sections                                                                                      |
| [`mlx-community/paligemma2-10b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-10b-ft-docci-448-bf16)               | `avoid` / `model shortcoming`     | - Do not copy prompt instructions into the Title, Description, or Keywords fields.                                                                                                                                                                                                                   | 17        | 8.13s   | 5.54      | 27 GB (23.3% of 108 GB recommended working set)  | low-draft-improvement; context-ignored; missing-sections                                                                     |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                            | `caveat` / `clean`                | Title: The Fenchurch Building (The Walkie-Talkie), London, England, UK, GBR, Europe Description: The Fenchurch Building (The Walkie-Talkie), a modern skyscraper in London, England, UK, is illuminated at night. The building is located on Fenchurch Street and is a prominent land...             | 161       | 7.10s   | 62.1      | 9.7 GB (8.41% of 108 GB recommended working set) | low-draft-improvement                                                                                                        |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `recommended` / `clean`           | **Title:** *20 Fenchurch Street at Night, London, England, UK* **Description:** A night-time view of 20 Fenchurch Street, also known as "Walkie-Talkie," a modern skyscraper in London’s financial district, illuminated against a dark blue sky. Visible details include adjacent bu...             | 158       | 7.10s   | 62.9      | 13 GB (11.7% of 108 GB recommended working set)  | missing terms: Cars, City, Commuting, Nightscape, Street signs                                                               |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `caveat` / `clean`                | Title: - 20 Fenchurch Street, London Description: - The illuminated skyscraper known as 20 Fenchurch Street, or the Walkie-Talkie building, stands prominently in a London street at night. Keywords: - Architecture, Building, Buildings, Cars, City, Cityscape, Fenchurch Street, I...             | 112       | 7.46s   | 39.2      | 16 GB (13.6% of 108 GB recommended working set)  | low-draft-improvement                                                                                                        |
| [`mlx-community/InternVL3-14B-8bit`](#model-mlx-community-internvl3-14b-8bit)                                           | `recommended` / `clean`           | Title: The Fenchurch Building, London Night Description: The Fenchurch Building, also known as The Walkie-Talkie, stands prominently in London's cityscape at night, illuminated by interior lights. The scene captures a quiet urban street with modern architecture, streetlights,...              | 118       | 8.02s   | 32.1      | 18 GB (15.7% of 108 GB recommended working set)  | low-draft-improvement                                                                                                        |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `caveat` / `clean`                | Title: The Fenchurch Building (The Walkie-Talkie), London Description: The Fenchurch Building, formally known as 20 Fenchurch Street, is illuminated at night, dominating a street scene in London, England; the building’s distinctive curved facade is prominently visible. Keyword...             | 142       | 9.54s   | 30.9      | 18 GB (15.7% of 108 GB recommended working set)  | low-draft-improvement                                                                                                        |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `recommended` / `clean`           | Title: The Fenchurch Building (The Walkie-Talkie), London, England, UK Description: A night view of the modern skyscraper known as 20 Fenchurch Street, situated along a city street in London, England. The scene features illuminated office windows, street lighting, and parked b...             | 128       | 10.38s  | 26.4      | 20 GB (17.7% of 108 GB recommended working set)  | low-draft-improvement                                                                                                        |
| [`mlx-community/pixtral-12b-bf16`](#model-mlx-community-pixtral-12b-bf16)                                               | `caveat` / `clean`                | Title: - 20 Fenchurch Street, London Description: - The illuminated 20 Fenchurch Street, also known as The Walkie-Talkie, stands prominently in a quiet London street at night. Keywords: - Architecture, Building, Buildings, City, Cityscape, Fenchurch Street, Illuminated, London...             | 99        | 11.01s  | 20.1      | 28 GB (23.9% of 108 GB recommended working set)  | low-draft-improvement                                                                                                        |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `avoid` / `model shortcoming`     | ## Image Description for Captioning **Title:** Night View of The Walkie-Talkie Building in London **Description:** A low-angle, wide shot captures the imposing silhouette of The Walkie-Talkie building at night in London. The skyscraper dominates the frame, its dark, reflective...             | 323       | 10.71s  | 48.2      | 17 GB (14.9% of 108 GB recommended working set)  | unverified-context-copy; metadata-borrowing                                                                                  |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `recommended` / `clean`           | Title: The Fenchurch Building (The Walkie-Talkie), London, England, UK Description: The image shows a night scene featuring the distinctive, angular glass facade of The Fenchurch Building (The Walkie-Talkie) in London. The street is relatively quiet with a few cars visible, st...             | 134       | 9.67s   | 80.0      | 8.6 GB (7.42% of 108 GB recommended working set) | clean                                                                                                                        |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `caveat` / `clean`                | &lt;think&gt; &lt;/think&gt; Title: Nighttime urban street with modern skyscraper Description: The image shows a modern urban street at night, featuring a prominent illuminated skyscraper under a dark sky, with surrounding buildings and street lighting. Visible details include parked bicy... | 94        | 9.75s   | 24.7      | 4.1 GB (3.53% of 108 GB recommended working set) | formatting                                                                                                                   |
| [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](#model-jqlive-kimi-vl-a3b-thinking-2506-6bit)                                 | `avoid` / `cutoff degraded`       | ◁think▷Okay, let's tackle this. First, I need to analyze the image and create a catalog record with the specified sections: Title, Description, and Keywords. The user wants it in British English, so I'll make sure to use British terms like "lorry" instead of "truck" if needed,...             | 500       | 10.86s  | 71.0      | 18 GB (15.6% of 108 GB recommended working set)  | unverified-context-copy; low-draft-improvement; metadata-borrowing                                                           |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `caveat` / `clean`                | Title: The Fenchurch Building (The Walkie-Talkie), London Description: The Fenchurch Building, also known as the Walkie-Talkie, stands illuminated at night in London, England. The distinctive curved glass facade reflects the surrounding city lights, with a quiet street below f...             | 136       | 11.06s  | 30.3      | 23 GB (19.9% of 108 GB recommended working set)  | low-draft-improvement                                                                                                        |
| [`mlx-community/GLM-4.6V-Flash-6bit`](#model-mlx-community-glm-46v-flash-6bit)                                          | `caveat` / `context budget`       | <\|begin_of_box\|>Title: The Walkie-Talkie Building at Night Description: A nighttime urban scene featuring the distinctive, angular glass facade of the Walkie-Talkie building (20 Fenchurch Street) in London. The street is dimly lit with streetlights, showing a few vehicles with...           | 152       | 10.46s  | 57.7      | 11 GB (9.58% of 108 GB recommended working set)  | low-draft-improvement; context-budget; missing-sections                                                                      |
| [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](#model-mlx-community-kimi-vl-a3b-thinking-8bit)                             | `avoid` / `cutoff degraded`       | ◁think▷Okay, let's tackle this metadata cataloging task. First, I need to analyze the image based on the given context and the draft metadata provided. The image shows a tall, modern skyscraper at night. The building has a distinctive curved glass facade with illuminated windo...             | 500       | 12.03s  | 62.6      | 22 GB (19% of 108 GB recommended working set)    | cutoff; thinking-trace; missing-sections                                                                                     |
| [`microsoft/Phi-3.5-vision-instruct`](#model-microsoft-phi-35-vision-instruct)                                          | `avoid` / `cutoff degraded`       | Title: The Fenchurch Building (The Walkie-Talkie), London, England, UK, GBR, Europe Description: The iconic Fenchurch Building, also known as The Walkie-Talkie, stands tall in the heart of London's financial district. Its distinctive curved facade is illuminated against the ni...             | 500       | 10.77s  | 57.2      | 9.6 GB (8.3% of 108 GB recommended working set)  | repetitive; low-draft-improvement; cutoff                                                                                    |
| [`mlx-community/gemma-3-27b-it-qat-8bit`](#model-mlx-community-gemma-3-27b-it-qat-8bit)                                 | `caveat` / `clean`                | Title: The Fenchurch Building (The Walkie-Talkie), London Description: The Fenchurch Building, also known as 20 Fenchurch Street, is illuminated at night, viewed from a street level perspective in London, with a dark blue sky overhead. Streetlights cast a warm glow on the pave...             | 140       | 14.22s  | 17.4      | 32 GB (27.4% of 108 GB recommended working set)  | low-draft-improvement                                                                                                        |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `avoid` / `cutoff degraded`       | Let's tackle this step by step. First, the title needs to be 5-10 words, concrete and factual. The authoritative context gives location as London, England, UK, and the building is The Fenchurch Building (The Walkie-Talkie). So the title should be something like "The Fenchurch...              | 500       | 15.85s  | 59.6      | 60 GB (52.1% of 108 GB recommended working set)  | low-draft-improvement; cutoff; missing-sections                                                                              |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                             | `avoid` / `cutoff degraded`       | - Do not output the description text after the label. - Do not output the title text after the label. - Do not output the description text after the label. - Do not output the title text after the label. - Do not output the description text after the label. - Do not output the...             | 500       | 15.79s  | 42.4      | 4.6 GB (3.96% of 108 GB recommended working set) | repetitive; low-draft-improvement; cutoff                                                                                    |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `caveat` / `context budget`       | 1000000                                                                                                                                                                                                                                                                                              | 9         | 15.34s  | 103       | 8.6 GB (7.44% of 108 GB recommended working set) | harness:long-context; low-draft-improvement; context-budget                                                                  |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `avoid` / `cutoff degraded`       | Here are my reasoning steps: We need to produce a catalog record with three sections: Title, Description, Keywords. The image is a nighttime cityscape of a tall skyscraper, likely the Walkie Talkie building (20 Fenchurch Street) in London. The authoritative context includes lo...             | 500       | 16.39s  | 42.7      | 15 GB (13% of 108 GB recommended working set)    | unverified-context-copy; metadata-borrowing; cutoff                                                                          |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `caveat` / `context budget`       | <\|begin_of_box\|>Title: The Fenchurch Building (The Walkie-Talkie), London, England, UK, GBR, Europe Description: Nighttime urban street scene featuring the Fenchurch Building (The Walkie-Talkie) skyscraper, with illuminated windows, modern architecture, and a quiet city street...           | 146       | 21.40s  | 43.0      | 78 GB (67.3% of 108 GB recommended working set)  | low-draft-improvement; context-budget; missing-sections                                                                      |
| [`mlx-community/gemma-4-31b-bf16`](#model-mlx-community-gemma-4-31b-bf16)                                               | `avoid` / `model shortcoming`     | Image: - The image shows a tall, modern skyscraper with a distinctive curved design, illuminated at night. The building is surrounded by other urban structures, including smaller buildings and streetlights. The scene is set in a city environment, with a street visible in the f...             | 82        | 24.16s  | 7.5       | 65 GB (56% of 108 GB recommended working set)    | low-draft-improvement; missing-sections                                                                                      |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `avoid` / `model shortcoming`     | Title: - London, England, United Kingdom, 20 Fenchurch Street, The Walkie-Talkie, 20 Fenchurch Street, London, England, United Kingdom, GBR, Europe Description: - The Fenchurch Building (The Walkie-Talkie) is a prominent skyscraper located at 20 Fenchurch Street in London, Eng...             | 294       | 20.72s  | 18.0      | 4.1 GB (3.56% of 108 GB recommended working set) | unverified-context-copy; low-draft-improvement; metadata-borrowing                                                           |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `avoid` / `cutoff degraded`       | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -...             | 500       | 20.26s  | 93.3      | 8.6 GB (7.43% of 108 GB recommended working set) | harness:long-context; repetitive; low-draft-improvement                                                                      |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `caveat` / `context budget`       | <\|im_start\|> addCriterion("image:caption")                                                                                                                                                                                                                                                         | 9         | 21.48s  | 64.1      | 13 GB (11.7% of 108 GB recommended working set)  | harness:long-context; low-draft-improvement; context-budget                                                                  |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `recommended` / `clean`           | **Title:** Modern Skyscraper at Dusk, London Urban Street **Description:** A nighttime urban street in London features a distinctive, slanted-glass skyscraper with illuminated interiors, flanked by older buildings. The scene includes empty roads, streetlights, and a few parked...             | 122       | 28.24s  | 58.0      | 13 GB (11.3% of 108 GB recommended working set)  | missing terms: Cars, City, Commuting, Fenchurch Street, Nightscape                                                           |
| [`mlx-community/paligemma2-3b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-3b-ft-docci-448-bf16)                 | `avoid` / `cutoff degraded`       | - Do not copy-prompt instructions into the Title, Description, or Keywords fields. - Do not infer identity, location, event, brand, species, time period, or intent. - Use this factual context where it improves the catalogue record; do not claim that contextual facts are visual...             | 500       | 29.66s  | 19.0      | 11 GB (9.71% of 108 GB recommended working set)  | low-draft-improvement; cutoff; context-ignored                                                                               |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)            | `caveat` / `clean`                | Title: The Fenchurch Building (The Walkie-Talkie), London, England, UK, GBR, Europe Description: The image depicts the Fenchurch Building, also known as the Walkie-Talkie, in London, England, UK, at night. The building is a prominent feature of the city's skyline, with its dis...             | 187       | 39.33s  | 11.5      | 15 GB (13% of 108 GB recommended working set)    | low-draft-improvement                                                                                                        |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `avoid` / `cutoff degraded`       | Title: The Fenchurch Building (The Walkie-Talkie), London, England, UK, GBR, Europe Description: The iconic Fenchurch Building, also known as The Walkie-Talkie, stands tall in the heart of London's financial district. Its distinctive curved facade is illuminated against the ni...             | 500       | 48.11s  | 10.8      | 9.6 GB (8.31% of 108 GB recommended working set) | repetitive; low-draft-improvement; cutoff                                                                                    |
| [`mlx-community/Molmo-7B-D-0924-bf16`](#model-mlx-community-molmo-7b-d-0924-bf16)                                       | `caveat` / `clean`                | Never output the prompt again. Title: The Fenchurch Building (The Walkie-Talkie), London, England, UK, GBR, Europe Description: The iconic Walkie Talkie building, formally known as 20 Fenchurch Street, stands illuminated against the night sky in London's financial district. It...             | 175       | 53.58s  | 14.2      | 48 GB (41.4% of 108 GB recommended working set)  | low-draft-improvement                                                                                                        |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `recommended` / `clean`           | Title: The Fenchurch Building, London, UK, Night Description: The Fenchurch Building, also known as 20 Fenchurch Street, is a modern skyscraper in London, England, visible at night with its distinctive curved glass façade and illuminated upper floors. The surrounding urban str...             | 137       | 58.55s  | 92.1      | 11 GB (9.94% of 108 GB recommended working set)  | clean                                                                                                                        |
| [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16)                             | `avoid` / `cutoff degraded`       | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -...             | 500       | 57.68s  | 92.3      | 8.6 GB (7.44% of 108 GB recommended working set) | harness:long-context; repetitive; low-draft-improvement                                                                      |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `recommended` / `clean`           | Title: The Fenchurch Building at night in London Description: The Fenchurch Building, also known as the Walkie-Talkie, dominates the evening skyline of London, England, with its distinctive curved glass facade reflecting the deep blue twilight. The street level is illuminated...              | 133       | 63.41s  | 107       | 26 GB (22.6% of 108 GB recommended working set)  | clean                                                                                                                        |
| [`mlx-community/Qwen3.5-35B-A3B-6bit`](#model-mlx-community-qwen35-35b-a3b-6bit)                                        | `recommended` / `clean`           | Title: The Fenchurch Building at night in London Description: The Fenchurch Building, also known as the Walkie-Talkie, stands prominently in the City of London against a deep blue evening sky. The street level scene features illuminated shop fronts, parked bicycles, and road s...             | 132       | 66.04s  | 91.2      | 35 GB (30.1% of 108 GB recommended working set)  | clean                                                                                                                        |
| [`mlx-community/Qwen3.5-35B-A3B-bf16`](#model-mlx-community-qwen35-35b-a3b-bf16)                                        | `recommended` / `clean`           | Title: The Fenchurch Building at night in London Description: The Fenchurch Building, located at 20 Fenchurch Street in London, England, is captured at night with its distinctive curved glass façade reflecting the deep blue sky. Street-level details include illuminated shop fr...             | 131       | 78.47s  | 65.8      | 76 GB (65.8% of 108 GB recommended working set)  | clean                                                                                                                        |
| [`mlx-community/Qwen3.5-27B-4bit`](#model-mlx-community-qwen35-27b-4bit)                                                | `recommended` / `clean`           | Title: 20 Fenchurch Street, London, England, UK, Europe Description: A low-angle night view of the 20 Fenchurch Street skyscraper, known as the Walkie-Talkie, dominating the frame against a deep blue sky. The street below is illuminated by streetlights and vehicle headlights,...              | 136       | 75.00s  | 30.7      | 26 GB (22.4% of 108 GB recommended working set)  | clean                                                                                                                        |
| [`mlx-community/Qwen3.5-27B-mxfp8`](#model-mlx-community-qwen35-27b-mxfp8)                                              | `recommended` / `clean`           | Title: The Walkie-Talkie Building, London, England, UK Description: A low-angle night view captures the distinctive curved glass facade of 20 Fenchurch Street, known as The Walkie-Talkie, towering over a quiet London street. The scene is illuminated by warm streetlights and sc...             | 141       | 80.20s  | 18.3      | 38 GB (33.3% of 108 GB recommended working set)  | clean                                                                                                                        |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `recommended` / `clean`           | Title: 20 Fenchurch Street, London, England, UK, Europe Description: A low-angle night view of 20 Fenchurch Street, known as the Walkie-Talkie, towering over a quiet urban street in London. The scene is illuminated by streetlights and the building's internal lights, with a row...             | 144       | 90.46s  | 17.7      | 38 GB (33.3% of 108 GB recommended working set)  | clean                                                                                                                        |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                           | `caveat` / `clean`                | Title: - 20 Fenchurch Street, London, England, UK, GBR, Europe Description: - 20 Fenchurch Street, also known as The Walkie-Talkie, is a distinctive skyscraper in London, England, UK, captured at night. The image shows the building's unique curved glass facade reflecting the d...             | 155       | 91.20s  | 1.83      | 27 GB (23.2% of 108 GB recommended working set)  | low-draft-improvement                                                                                                        |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `avoid` / `cutoff degraded`       | ◁think▷Okay, let's tackle this. First, I need to analyze the image and create a catalog record following the given instructions. The image shows a tall, modern skyscraper at night, which is the Walkie-Talkie building in London, England. The scene is urban, with streetlights, b...             | 500       | 112.35s | 4.69      | 40 GB (34.7% of 108 GB recommended working set)  | low-draft-improvement; cutoff; thinking-incomplete                                                                           |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `recommended` / `clean`           | Title: - The Fenchurch Street skyscraper at night Description: - The Fenchurch Street skyscraper in London, England, rises against a deep blue night sky, its curved glass facade reflecting the surrounding city lights. Street-level details include illuminated shop fronts, parke...             | 126       | 144.77s | 57.3      | 76 GB (65.8% of 108 GB recommended working set)  | missing terms: Building, Buildings, Commuting, Modern, Nightscape                                                            |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                             | `avoid` / `cutoff degraded`       | -2026-07-18 22:55:39 BST 51.511300°N 0.083400°W 51.511300°N 0.083400°W 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BS...             | 500       | 250.86s | 225       | 5.1 GB (4.39% of 108 GB recommended working set) | harness:long-context; repetitive; unverified-context-copy                                                                    |
<!-- markdownlint-enable MD034 MD049 -->

## Quick Navigation

- _Successful outputs:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16),
  [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit),
  [`mlx-community/LFM2-VL-1.6B-8bit`](#model-mlx-community-lfm2-vl-16b-8bit),
  [`HuggingFaceTB/SmolVLM-Instruct`](#model-huggingfacetb-smolvlm-instruct),
  [`mlx-community/SmolVLM-Instruct-bf16`](#model-mlx-community-smolvlm-instruct-bf16),
  [`qnguyen3/nanoLLaVA`](#model-qnguyen3-nanollava),
  [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16),
  [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit),
  +52 more
- _Flagged outputs:_ [`mlx-community/Molmo-7B-D-0924-8bit`](#model-mlx-community-molmo-7b-d-0924-8bit),
  [`mlx-community/Step-3.7-Flash-oQ2e`](#model-mlx-community-step-37-flash-oq2e),
  [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16),
  [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit),
  [`mlx-community/LFM2-VL-1.6B-8bit`](#model-mlx-community-lfm2-vl-16b-8bit),
  [`HuggingFaceTB/SmolVLM-Instruct`](#model-huggingfacetb-smolvlm-instruct),
  [`mlx-community/SmolVLM-Instruct-bf16`](#model-mlx-community-smolvlm-instruct-bf16),
  [`qnguyen3/nanoLLaVA`](#model-qnguyen3-nanollava), +45 more
- _Failed outputs:_ [`mlx-community/Molmo-7B-D-0924-8bit`](#model-mlx-community-molmo-7b-d-0924-8bit),
  [`mlx-community/Step-3.7-Flash-oQ2e`](#model-mlx-community-step-37-flash-oq2e)

## Model Gallery

Full generated output by model:

<!-- markdownlint-disable MD033 MD034 -->

<a id="model-mlx-community-molmo-7b-d-0924-8bit"></a>

### ❔ mlx-community/Molmo-7B-D-0924-8bit

- _Recommendation:_ not evaluated; review verdict: indeterminate
- _Status:_ Failed (Network Error)
- _Owner:_ likely owner `unknown`; reported package `unknown`; failure stage
  `Network Error`; diagnostic code `UNKNOWN_MODEL_LOAD_NETWORK_ERROR`
- _Next step:_ Retry when external connectivity is stable; the model was not
  evaluated and the disconnect does not identify a faulty package.
- _Error summary:_ Model loading failed: [Errno 54] Connection reset by peer
- _Key signals:_ network error; unknown model load network error; external
  connectivity
- _Failure context:_ type `ValueError`; phase `model_load`; code
  `UNKNOWN_MODEL_LOAD_NETWORK_ERROR`; package `unknown`
- _Tokens:_ prompt n/a; estimated text n/a; estimated non-text n/a; generated
  n/a; requested max 500 tok; stop reason exception


<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_transports/default.py", line 101, in map_httpcore_exceptions
    yield
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_transports/default.py", line 250, in handle_request
    resp = self._pool.handle_request(req)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/connection_pool.py", line 256, in handle_request
    raise exc from None
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/connection_pool.py", line 236, in handle_request
    response = connection.handle_request(
        pool_request.request
    )
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/connection.py", line 103, in handle_request
    return self._connection.handle_request(request)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 136, in handle_request
    raise exc
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 106, in handle_request
    ) = self._receive_response_headers(**kwargs)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 177, in _receive_response_headers
    event = self._receive_event(timeout=timeout)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_sync/http11.py", line 217, in _receive_event
    data = self._network_stream.read(
        self.READ_NUM_BYTES, timeout=timeout
    )
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_backends/sync.py", line 126, in read
    with map_exceptions(exc_map):
         ~~~~~~~~~~~~~~^^^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/contextlib.py", line 162, in __exit__
    self.gen.throw(value)
    ~~~~~~~~~~~~~~^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpcore/_exceptions.py", line 14, in map_exceptions
    raise to_exc(exc) from exc
httpcore.ReadError: [Errno 54] Connection reset by peer

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 793, in load
    model_path = get_model_path(
        path_or_hf_repo, force_download=force_download, revision=revision
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 444, in get_model_path
    snapshot_download(
    ~~~~~~~~~~~~~~~~~^
        repo_id=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    ...<11 lines>...
        force_download=force_download,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/utils/_validators.py", line 88, in _inner_fn
    return fn(*args, **kwargs)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/_snapshot_download.py", line 259, in snapshot_download
    repo_info = api.repo_info(repo_id=repo_id, repo_type=repo_type, revision=revision)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/utils/_validators.py", line 88, in _inner_fn
    return fn(*args, **kwargs)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/hf_api.py", line 3630, in repo_info
    return method(
        repo_id,
    ...<4 lines>...
        files_metadata=files_metadata,
    )
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/utils/_validators.py", line 88, in _inner_fn
    return fn(*args, **kwargs)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/huggingface_hub/hf_api.py", line 3309, in model_info
    r = get_session().get(path, headers=headers, timeout=timeout, params=params)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_client.py", line 1053, in get
    return self.request(
           ~~~~~~~~~~~~^
        "GET",
        ^^^^^^
    ...<7 lines>...
        extensions=extensions,
        ^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_client.py", line 825, in request
    return self.send(request, auth=auth, follow_redirects=follow_redirects)
           ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_client.py", line 914, in send
    response = self._send_handling_auth(
        request,
    ...<2 lines>...
        history=[],
    )
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_client.py", line 942, in _send_handling_auth
    response = self._send_handling_redirects(
        request,
        follow_redirects=follow_redirects,
        history=history,
    )
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_client.py", line 979, in _send_handling_redirects
    response = self._send_single_request(request)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_client.py", line 1014, in _send_single_request
    response = transport.handle_request(request)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_transports/default.py", line 249, in handle_request
    with map_httpcore_exceptions():
         ~~~~~~~~~~~~~~~~~~~~~~~^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/contextlib.py", line 162, in __exit__
    self.gen.throw(value)
    ~~~~~~~~~~~~~~^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages/httpx/_transports/default.py", line 118, in map_httpcore_exceptions
    raise mapped_exc(message) from exc
httpx.ReadError: [Errno 54] Connection reset by peer

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
ValueError: Model loading failed: [Errno 54] Connection reset by peer
```

</details>

---

<a id="model-mlx-community-step-37-flash-oq2e"></a>

### ❌ mlx-community/Step-3.7-Flash-oQ2e

- _Recommendation:_ avoid for now; review verdict: runtime failure
- _Status:_ Failed (Processor Error)
- _Owner:_ likely owner `model-config`; reported package `model-config`;
  failure stage `Processor Error`; diagnostic code
  `MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR`
- _Next step:_ Inspect the model repo processor/preprocessor config and
  AutoProcessor mapping; the multimodal processor is missing or not exposing
  the image processor expected by mlx-vlm.
- _Error summary:_ see error details below
- _Key signals:_ processor error; model config processor load processor
- _Failure context:_ type `ValueError`; phase `processor_load`; code
  `MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR`; package `model-config`
- _Tokens:_ prompt n/a; estimated text n/a; estimated non-text n/a; generated
  n/a; requested max 500 tok; stop reason exception

_Error details:_

> Model preflight failed for mlx-community/Step-3.7-Flash-oQ2e: Loaded
> processor has no image_processor; expected multimodal processor.

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
ValueError: Loaded processor has no image_processor; expected multimodal processor.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
ValueError: Model preflight failed for mlx-community/Step-3.7-Flash-oQ2e: Loaded processor has no image_processor; expected multimodal processor.
```

</details>

---

<a id="model-liquidai-lfm25-vl-450m-mlx-bf16"></a>

### ⚠️ LiquidAI/LFM2.5-VL-450M-MLX-bf16

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Commuting; keywords=23;
  low-draft-improvement
- _Timing:_ Load 0.44s; Gen 0.75s; Total 1.47s
- _Throughput:_ Prompt 23,382 TPS (700 tok); Gen 509 TPS (188 tok)
- _Tokens:_ prompt 700 tok; estimated text 517 tok; estimated non-text 183
  tok; generated 188 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: LiquidAI/LFM2.5-VL-450M-MLX-bf16</summary>

```text
Title: Fenchurch Building (The Walkie-Talkie), London, England, UK, GBR, Europe

Description: The Fenchurch Building, formally known as 20 Fenchurch Street, is a prominent skyscraper in London, England, United Kingdom. It is a modern architectural structure that stands out in the cityscape. The building is illuminated at night, highlighting its sleek design and reflective glass facade. The Fenchurch Building is part of the London skyline and is a notable example of contemporary urban architecture.

Keywords: Architecture, Building, Buildings, City, Cityscape, Skyscraper, Street, Street signs, The Fenchurch Building (The Walkie-Talkie), Urban, Urban landscape, Walkie Talkie building, London, Modern, Night, Nightscape, Skyscraper, Street, Street signs, London, UK, GBR, Europe
```

</details>

⚠️ _Quality Warnings:_

- Description sentence violation (4; expected 1-2)
- Keyword count violation (23; expected 10-18)
- Low draft improvement

---

<a id="model-mlx-community-nanollava-15-4bit"></a>

### ⚠️ mlx-community/nanoLLaVA-1.5-4bit

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ keywords=20; context echo=76%; low-draft-improvement
- _Timing:_ Load 1.04s; Gen 0.91s; Total 2.25s
- _Throughput:_ Prompt 7,117 TPS (625 tok); Gen 346 TPS (112 tok)
- _Tokens:_ prompt 625 tok; estimated text 517 tok; estimated non-text 108
  tok; generated 112 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: mlx-community/nanoLLaVA-1.5-4bit</summary>

```text
Title: The Fenchurch Building (The Walkie-Talkie), London, England, UK, GBR, Europe
Description: Walkie Talkie building known formally as 20 Fenchurch Street.
Keywords: Architecture, Building, Buildings, Cars, City, Cityscape, Commuting, Fenchurch Street, Illuminated, London, Modern, Night, Nightscape, Skyscraper, Street, Street signs, The Fenchurch Building (The Walkie-Talkie), Urban, Urban landscape, Walkie Talkie building
```

</details>

⚠️ _Quality Warnings:_

- Title length violation (11 words; expected 5-10)
- Keyword count violation (20; expected 10-18)
- Context echo (76% overlap)
- Low draft improvement

---

<a id="model-mlx-community-lfm2-vl-16b-8bit"></a>

### ⚠️ mlx-community/LFM2-VL-1.6B-8bit

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: formally; keywords=20; low-draft-improvement
- _Timing:_ Load 0.33s; Gen 1.07s; Total 1.71s
- _Throughput:_ Prompt 8,461 TPS (890 tok); Gen 331 TPS (201 tok)
- _Tokens:_ prompt 890 tok; estimated text 517 tok; estimated non-text 373
  tok; generated 201 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: mlx-community/LFM2-VL-1.6B-8bit</summary>

```text
Title: The Fenchurch Building (The Walkie-Talkie), London, England, UK, GBR, Europe

Description: The Fenchurch Building, also known as 20 Fenchurch Street, is a prominent skyscraper located in the heart of London, England. This modern architectural marvel stands tall against the night sky, illuminated by its distinctive glass facade. The building is a testament to contemporary design, featuring sleek lines and a unique curved shape that sets it apart from its surroundings. The Fenchurch Building is not only an iconic landmark but also a hub of commercial activity, housing numerous offices and businesses.

Keywords: Architecture, Building, Buildings, Cars, City, Cityscape, Commuting, Fenchurch Street, Illuminated, London, Modern, Night, Nightscape, Skyscraper, Street, Street signs, The Fenchurch Building (The Walkie-Talkie), Urban, Urban landscape, Walkie Talkie building
```

</details>

⚠️ _Quality Warnings:_

- Title length violation (11 words; expected 5-10)
- Description sentence violation (4; expected 1-2)
- Keyword count violation (20; expected 10-18)
- Low draft improvement

---

<a id="model-huggingfacetb-smolvlm-instruct"></a>

### ❌ HuggingFaceTB/SmolVLM-Instruct

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Architecture, Building, Buildings, Cars, City; low-draft-improvement
- _Timing:_ Load 0.74s; Gen 1.13s; Total 2.15s
- _Throughput:_ Prompt 4,471 TPS (1,843 tok); Gen 131 TPS (18 tok)
- _Tokens:_ prompt 1843 tok; estimated text 517 tok; estimated non-text 1326
  tok; generated 18 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: HuggingFaceTB/SmolVLM-Instruct</summary>

```text
 The Fenchurch Street, London, England, UK, GBR, Europe.
```

</details>

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)
- Low draft improvement
- Degrades trusted hints

---

<a id="model-mlx-community-smolvlm-instruct-bf16"></a>

### ❌ mlx-community/SmolVLM-Instruct-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Architecture, Building, Buildings, Cars, City; low-draft-improvement
- _Timing:_ Load 1.45s; Gen 1.20s; Total 2.95s
- _Throughput:_ Prompt 4,432 TPS (1,843 tok); Gen 124 TPS (18 tok)
- _Tokens:_ prompt 1843 tok; estimated text 517 tok; estimated non-text 1326
  tok; generated 18 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: mlx-community/SmolVLM-Instruct-bf16</summary>

```text
 The Fenchurch Street, London, England, UK, GBR, Europe.
```

</details>

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)
- Low draft improvement
- Degrades trusted hints

---

<a id="model-qnguyen3-nanollava"></a>

### ❌ qnguyen3/nanoLLaVA

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; missing terms: Architecture,
  Cars, City, Cityscape, Commuting; low-draft-improvement
- _Timing:_ Load 0.80s; Gen 1.41s; Total 2.51s
- _Throughput:_ Prompt 6,166 TPS (625 tok); Gen 113 TPS (91 tok)
- _Tokens:_ prompt 625 tok; estimated text 517 tok; estimated non-text 108
  tok; generated 91 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: qnguyen3/nanoLLaVA</summary>

```text
Title: The Walkie-Talkie Building (The Walkie-Talkie), London, England, UK, GBR, Europe
Description: A tall, tall building with a glass facade, the name "The Walkie-Talkie" written on its side. The building is located in London, England, UK, and is part of the Walkie-Talkie district. The building is made of glass and has a modern design.
```

</details>

⚠️ _Quality Warnings:_

- Metadata alignment issue (low_metadata_alignment)
- Missing sections (keywords)
- Title length violation (12 words; expected 5-10)
- Description sentence violation (3; expected 1-2)
- Low draft improvement
- Degrades trusted hints

---

<a id="model-mlx-community-fastvlm-05b-bf16"></a>

### ❌ mlx-community/FastVLM-0.5B-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; missing terms: Cars, Cityscape,
  Commuting, Modern, Nightscape; low-draft-improvement
- _Timing:_ Load 0.61s; Gen 1.55s; Total 2.47s
- _Throughput:_ Prompt 6,085 TPS (629 tok); Gen 343 TPS (179 tok)
- _Tokens:_ prompt 629 tok; estimated text 517 tok; estimated non-text 112
  tok; generated 179 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: mlx-community/FastVLM-0.5B-bf16</summary>

```text
Title: The Fenchurch Building (The Walkie-Talkie), London, England, UK, GBR, Europe
Description: The Fenchurch Building, also known as 20 Fenchurch Street, is a prominent example of modernist architecture in London, UK. The building is a 20-story skyscraper with a distinctive curved facade and a glass exterior. It was completed in 1964 and is a well-known landmark in the city. The building is located in the Fenchurch district of London and is surrounded by other modernist buildings. The Fenchurch Building is a popular tourist attraction and is often featured in films and television shows. The building is illuminated at night and is a popular spot for tourists to take photos. The Fenchurch Building is a symbol of modernist architecture and is a key feature of the city's skyline.
```

</details>

⚠️ _Quality Warnings:_

- Missing sections (keywords)
- Title length violation (11 words; expected 5-10)
- Description sentence violation (7; expected 1-2)
- Low draft improvement

---

<a id="model-mlx-community-gemma-4-26b-a4b-it-4bit"></a>

### ✅ mlx-community/gemma-4-26b-a4b-it-4bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, City, Commuting, Street signs, The
  Fenchurch Building (The Walkie-Talkie)
- _Timing:_ Load 2.95s; Gen 1.82s; Total 5.10s
- _Throughput:_ Prompt 1,784 TPS (905 tok); Gen 123 TPS (101 tok)
- _Tokens:_ prompt 905 tok; estimated text 517 tok; estimated non-text 388
  tok; generated 101 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: mlx-community/gemma-4-26b-a4b-it-4bit</summary>

```text
Title:
The Fenchurch Building in London at night

Description:
The curved glass facade of the Fenchurch Building towers over a dark urban street in London at night. Street lamps and building windows provide points of light against the deep blue night sky.

Keywords:
Architecture, Building, Cityscape, Fenchurch Street, Illuminated, London, Modern, Night, Nightscape, Skyscraper, Street, Urban, 20 Fenchurch Street, England, UK, Europe
```

</details>

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-smolvlm2-22b-instruct-mlx"></a>

### ⚠️ mlx-community/SmolVLM2-2.2B-Instruct-mlx

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ keywords=20; low-draft-improvement
- _Timing:_ Load 0.78s; Gen 2.18s; Total 3.26s
- _Throughput:_ Prompt 2,192 TPS (743 tok); Gen 132 TPS (158 tok)
- _Tokens:_ prompt 743 tok; estimated text 517 tok; estimated non-text 226
  tok; generated 158 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: mlx-community/SmolVLM2-2.2B-Instruct-mlx</summary>

```text
 Title: The Fenchurch Building (The Walkie-Talkie), London, England, UK, GBR, Europe

Description: The Walkie-Talkie building, formally known as 20 Fenchurch Street, is a modern skyscraper in the Fenchurch Street district of London, England, UK. It is illuminated at night and is part of the Walkie-Talkie building complex.

Keywords: Architecture, Building, Buildings, Cars, City, Cityscape, Commuting, Fenchurch Street, Illuminated, London, Modern, Night, Nightscape, Skyscraper, Street, Street signs, The Fenchurch Building (The Walkie-Talkie), Urban, Urban landscape, Walkie Talkie building
```

</details>

⚠️ _Quality Warnings:_

- Title length violation (11 words; expected 5-10)
- Keyword count violation (20; expected 10-18)
- Low draft improvement

---

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-6bit"></a>

### ❌ mlx-community/paligemma2-10b-ft-docci-448-6bit

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Architecture, Building, Buildings, Cars, City; low-draft-improvement
- _Timing:_ Load 1.72s; Gen 2.29s; Total 4.32s
- _Throughput:_ Prompt 1,343 TPS (1,652 tok); Gen 31.0 TPS (17 tok)
- _Tokens:_ prompt 1652 tok; estimated text 517 tok; estimated non-text 1135
  tok; generated 17 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: mlx-community/paligemma2-10b-ft-docci-448-6bit</summary>

```text
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
```

</details>

⚠️ _Quality Warnings:_

- Context ignored (missing: Architecture, Building, Buildings, Cars, City)
- Metadata alignment issue (low_metadata_alignment)
- Missing sections (title, description, keywords)
- Low draft improvement
- Ignores trusted hints

---

<a id="model-mlx-community-ministral-3-3b-instruct-2512-4bit"></a>

### ⚠️ mlx-community/Ministral-3-3B-Instruct-2512-4bit

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Commuting, Nightscape, Street signs, The
  Fenchurch Building (The Walkie-Talkie), Urban landscape; keywords=22
- _Timing:_ Load 0.92s; Gen 2.49s; Total 3.70s
- _Throughput:_ Prompt 2,617 TPS (3,239 tok); Gen 185 TPS (151 tok)
- _Tokens:_ prompt 3239 tok; estimated text 517 tok; estimated non-text 2722
  tok; generated 151 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: mlx-community/Ministral-3-3B-Instruct-2512-4bit</summary>

```text
Title:
The Walkie-Talkie Building at Night, Fenchurch Street, London

Description:
A modern skyscraper with a distinctive, asymmetrical design and glass facade dominates the night scene, its lights casting reflections on the surrounding urban streets. The area around 20 Fenchurch Street shows a mix of pedestrian activity, street lamps, and a partially obscured construction crane, with a quiet, late-night atmosphere.

Keywords:
architecture, asymmetrical, building, cars, city, cityscape, construction, crane, dark sky, Fenchurch Street, illuminated, London, modern, night, night lighting, skyscraper, street, street lamps, urban, urban setting, Walkie-Talkie, walkways
```

</details>

⚠️ _Quality Warnings:_

- Keyword count violation (22; expected 10-18)

---

<a id="model-mlx-community-diffusiongemma-26b-a4b-it-8bit"></a>

### ⚠️ mlx-community/diffusiongemma-26B-A4B-it-8bit

- _Recommendation:_ use with caveats; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title; missing terms: Cars, City,
  Commuting, Street signs, The Fenchurch Building (The Walkie-Talkie);
  formatting=Unknown tags: &lt;channel|&gt;
- _Timing:_ Load 3.70s; Gen 2.61s; Total 6.65s
- _Throughput:_ Prompt 2,247 TPS (901 tok); Gen 66.5 TPS (104 tok)
- _Tokens:_ prompt 901 tok; estimated text 517 tok; estimated non-text 384
  tok; generated 104 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: mlx-community/diffusiongemma-26B-A4B-it-8bit</summary>

```text
<|channel>thought
<channel|>Title:
- 20 Fenchurch Street building at night in London

Description:
- The glass facade of 20 Fenchurch Street towers over a narrow urban street at dusk. The scene features modern architecture illuminated by street lamps and interior office windows against a deep blue sky.

Keywords:
- Architecture, building, cityscape, Fenchurch Street, illuminated, London, modern, night, nightscape, skyscraper, street, urban, glass, England, United Kingdom
```

</details>

⚠️ _Quality Warnings:_

- Metadata alignment issue (low_metadata_alignment)
- Missing sections (title)
- Unknown tags: <channel\|>

---

<a id="model-mlx-community-idefics3-8b-llama3-bf16"></a>

### ⚠️ mlx-community/Idefics3-8B-Llama3-bf16

- _Recommendation:_ use with caveats; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Architecture, Cars, City, Cityscape, Commuting; formatting=Unknown
  tags: &lt;end_of_utterance&gt;; low-draft-improvement
- _Timing:_ Load 2.00s; Gen 2.63s; Total 4.97s
- _Throughput:_ Prompt 2,369 TPS (2,907 tok); Gen 33.4 TPS (26 tok)
- _Tokens:_ prompt 2907 tok; estimated text 517 tok; estimated non-text 2390
  tok; generated 26 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: mlx-community/Idefics3-8B-Llama3-bf16</summary>

```text
 The Fenchurch Building (The Walkie-Talkie), London, England, UK, GBR, Europe.<end_of_utterance>
```

</details>

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)
- Low draft improvement
- Degrades trusted hints
- Unknown tags: <end_of_utterance>

---

<a id="model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8"></a>

### ⚠️ mlx-community/diffusiongemma-26B-A4B-it-mxfp8

- _Recommendation:_ use with caveats; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title; missing terms: Cars, Cityscape,
  Commuting, Street signs, The Fenchurch Building (The Walkie-Talkie);
  formatting=Unknown tags: &lt;channel|&gt;; low-draft-improvement
- _Timing:_ Load 3.59s; Gen 2.65s; Total 6.56s
- _Throughput:_ Prompt 2,262 TPS (901 tok); Gen 62.7 TPS (100 tok)
- _Tokens:_ prompt 901 tok; estimated text 517 tok; estimated non-text 384
  tok; generated 100 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: mlx-community/diffusiongemma-26B-A4B-it-mxfp8</summary>

```text
<|channel>thought
<channel|>Title:
- 20 Fenchurch Street building at night

Description:
- The curved skyscraper of 20 Fenchurch Street, known as the Walkie-Talkie, rises over a dark London street illuminated by streetlights and modern glass buildings.

Keywords:
- Architecture, Building, City, London, Modern, Night, Nightscape, Skyscraper, Street, Fenchurch Street, Urban, Glass, Evening, England, United Kingdom
```

</details>

⚠️ _Quality Warnings:_

- Missing sections (title)
- Low draft improvement
- Unknown tags: <channel\|>

---

<a id="model-mlx-community-internvl3-8b-bf16"></a>

### ✅ mlx-community/InternVL3-8B-bf16

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Commuting, Fenchurch Street, Nightscape, The
  Fenchurch Building (The Walkie-Talkie), Fenchurch; low-draft-improvement
- _Timing:_ Load 1.64s; Gen 4.05s; Total 5.99s
- _Throughput:_ Prompt 2,789 TPS (2,408 tok); Gen 34.2 TPS (90 tok)
- _Tokens:_ prompt 2408 tok; estimated text 517 tok; estimated non-text 1891
  tok; generated 90 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: mlx-community/InternVL3-8B-bf16</summary>

```text
Title:
The Walkie-Talkie Building, London, England

Description:
Modern skyscraper in London at night, illuminated with reflective glass facade, streetlights casting a warm glow, and a few cars on the road.

Keywords:
Architecture, Building, Buildings, Cars, City, Cityscape, Illuminated, London, Modern, Night, Skyscraper, Street, Street signs, Urban, Urban landscape, Walkie-Talkie Building
```

</details>

⚠️ _Quality Warnings:_

- Low draft improvement

---

<a id="model-mlx-community-gemma-3n-e2b-4bit"></a>

### ❌ mlx-community/gemma-3n-E2B-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Architecture, Building, Buildings, Cars, City;
  repetitive token=phrase: "- do not take..."
- _Timing:_ Load 2.07s; Gen 4.84s; Total 7.22s
- _Throughput:_ Prompt 2,737 TPS (891 tok); Gen 123 TPS (500 tok)
- _Tokens:_ prompt 891 tok; estimated text 517 tok; estimated non-text 374
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<details>
<summary>Complete generated output: mlx-community/gemma-3n-E2B-4bit</summary>

```text

- Do not take it is a 2026-07-18 22:55:39 22:55:39
- Do not take it is a 22:55:39 22:55:39
- Do not take it is a 22:55:39 22:55:39
- Do not take it is a 22:55:39 22:55:39
- Do not take it is a 22:55:39 22:55:39
- Do not take it is a 22:55:39 22:55:39
- Do not take it is a 22:55:39 22:55:39
- Do not take it is a 22:55:39 22:55:39
- Do not take it is a 22:55:39 22:55:39
- Do not take it is a 22:55:39 22:55:39
- Do not take it is a 22:55:39 22:55:39
- Do not take it is a 22:55:39 22:55:39
- Do not take it is a 22:55:39 22:55:39
- Do not take it is a 22:55:39 22:55:39
- Do not take it is a 22:55:39 22:55:39
- Do not take it is a 22:55:39 22:55:39
- Do not take it is a 22:55:39 22:55:39
- Do not take it is a 22:55:39 22:55:39
- Do not take it is a 22:55:39 22:
```

</details>

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "- do not take...")
- Context ignored (missing: Architecture, Building, Buildings, Cars, City)
- Generation loop (repetitive_tail)
- Metadata alignment issue (low_metadata_alignment)
- Missing sections (title, description, keywords)
- Unverified context copy
- Low draft improvement
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-meta-llama-llama-32-11b-vision-instruct"></a>

### ❌ meta-llama/Llama-3.2-11B-Vision-Instruct

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Architecture, Cars, City, Cityscape, Commuting; low-draft-improvement
- _Timing:_ Load 2.22s; Gen 4.99s; Total 7.52s
- _Throughput:_ Prompt 312 TPS (596 tok); Gen 5.39 TPS (14 tok)
- _Tokens:_ prompt 596 tok; estimated text 517 tok; estimated non-text 79 tok;
  generated 14 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: meta-llama/Llama-3.2-11B-Vision-Instruct</summary>

```text
The Fenchurch Building (The Walkie-Talkie)
```

</details>

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)
- Low draft improvement
- Degrades trusted hints

---

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-bf16"></a>

### ❌ mlx-community/paligemma2-10b-ft-docci-448-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Architecture, Building, Buildings, Cars, City; low-draft-improvement
- _Timing:_ Load 2.72s; Gen 5.10s; Total 8.13s
- _Throughput:_ Prompt 1,092 TPS (1,652 tok); Gen 5.54 TPS (17 tok)
- _Tokens:_ prompt 1652 tok; estimated text 517 tok; estimated non-text 1135
  tok; generated 17 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: mlx-community/paligemma2-10b-ft-docci-448-bf16</summary>

```text
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
```

</details>

⚠️ _Quality Warnings:_

- Context ignored (missing: Architecture, Building, Buildings, Cars, City)
- Metadata alignment issue (low_metadata_alignment)
- Missing sections (title, description, keywords)
- Low draft improvement
- Ignores trusted hints

---

<a id="model-mlx-community-llava-v16-mistral-7b-8bit"></a>

### ⚠️ mlx-community/llava-v1.6-mistral-7b-8bit

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: known, formally, 20; keywords=20;
  low-draft-improvement
- _Timing:_ Load 1.42s; Gen 5.37s; Total 7.10s
- _Throughput:_ Prompt 1,338 TPS (2,867 tok); Gen 62.1 TPS (161 tok)
- _Tokens:_ prompt 2867 tok; estimated text 517 tok; estimated non-text 2350
  tok; generated 161 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: mlx-community/llava-v1.6-mistral-7b-8bit</summary>

```text
Title: The Fenchurch Building (The Walkie-Talkie), London, England, UK, GBR, Europe

Description: The Fenchurch Building (The Walkie-Talkie), a modern skyscraper in London, England, UK, is illuminated at night. The building is located on Fenchurch Street and is a prominent landmark in the city.

Keywords: Architecture, Building, Buildings, Cars, City, Cityscape, Commuting, Fenchurch Street, Illuminated, London, Modern, Night, Nightscape, Skyscraper, Street, Street signs, The Fenchurch Building (The Walkie-Talkie), Urban, Urban landscape, Walkie Talkie building
```

</details>

⚠️ _Quality Warnings:_

- Title length violation (11 words; expected 5-10)
- Keyword count violation (20; expected 10-18)
- Low draft improvement

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-nvfp4"></a>

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-nvfp4

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, City, Commuting, Nightscape, Street
  signs
- _Timing:_ Load 1.39s; Gen 5.42s; Total 7.10s
- _Throughput:_ Prompt 1,310 TPS (3,240 tok); Gen 62.9 TPS (158 tok)
- _Tokens:_ prompt 3240 tok; estimated text 517 tok; estimated non-text 2723
  tok; generated 158 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: mlx-community/Ministral-3-14B-Instruct-2512-nvfp4</summary>

```text
**Title:**
*20 Fenchurch Street at Night, London, England, UK*

**Description:**
A night-time view of 20 Fenchurch Street, also known as "Walkie-Talkie," a modern skyscraper in London’s financial district, illuminated against a dark blue sky. Visible details include adjacent buildings, street lighting, a few parked e-bikes, road signs, and a nearly empty street with minimal evening activity.

**Keywords:**
Modern skyscraper, 20 Fenchurch Street, Walkie-Talkie, London, England, night photography, urban architecture, illuminated building, financial district, street lighting, road signs, e-bikes, dark blue sky, minimal traffic, European cityscape, commercial architecture
```

</details>

⚠️ _Quality Warnings:_

- Metadata alignment issue (low_metadata_alignment)

---

<a id="model-mlx-community-pixtral-12b-8bit"></a>

### ⚠️ mlx-community/pixtral-12b-8bit

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Commuting, The Fenchurch Building (The
  Walkie-Talkie), GBR, formally; keywords=19; low-draft-improvement
- _Timing:_ Load 1.69s; Gen 5.45s; Total 7.46s
- _Throughput:_ Prompt 1,600 TPS (3,433 tok); Gen 39.2 TPS (112 tok)
- _Tokens:_ prompt 3433 tok; estimated text 517 tok; estimated non-text 2916
  tok; generated 112 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: mlx-community/pixtral-12b-8bit</summary>

```text
Title:
- 20 Fenchurch Street, London

Description:
- The illuminated skyscraper known as 20 Fenchurch Street, or the Walkie-Talkie building, stands prominently in a London street at night.

Keywords:
- Architecture, Building, Buildings, Cars, City, Cityscape, Fenchurch Street, Illuminated, London, Modern, Night, Nightscape, Skyscraper, Street, Street signs, The Fenchurch Building, Urban, Urban landscape, Walkie Talkie building
```

</details>

⚠️ _Quality Warnings:_

- Title length violation (4 words; expected 5-10)
- Keyword count violation (19; expected 10-18)
- Low draft improvement

---

<a id="model-mlx-community-internvl3-14b-8bit"></a>

### ✅ mlx-community/InternVL3-14B-8bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Commuting, The Fenchurch Building (The
  Walkie-Talkie), GBR, formally; low-draft-improvement
- _Timing:_ Load 1.72s; Gen 5.99s; Total 8.02s
- _Throughput:_ Prompt 1,372 TPS (2,408 tok); Gen 32.1 TPS (118 tok)
- _Tokens:_ prompt 2408 tok; estimated text 517 tok; estimated non-text 1891
  tok; generated 118 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: mlx-community/InternVL3-14B-8bit</summary>

```text
Title:
The Fenchurch Building, London Night

Description:
The Fenchurch Building, also known as The Walkie-Talkie, stands prominently in London's cityscape at night, illuminated by interior lights. The scene captures a quiet urban street with modern architecture, streetlights, and a few parked bicycles.

Keywords:
Architecture, Building, City, Cityscape, Fenchurch Street, Illuminated, London, Modern, Night, Nightscape, Skyscraper, Street, Street signs, The Fenchurch Building, Urban, Urban landscape, Walkie Talkie building
```

</details>

⚠️ _Quality Warnings:_

- Low draft improvement

---

<a id="model-mlx-community-gemma-3-27b-it-qat-4bit"></a>

### ⚠️ mlx-community/gemma-3-27b-it-qat-4bit

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Commuting, GBR; keywords=23;
  low-draft-improvement
- _Timing:_ Load 2.81s; Gen 6.42s; Total 9.54s
- _Throughput:_ Prompt 663 TPS (900 tok); Gen 30.9 TPS (142 tok)
- _Tokens:_ prompt 900 tok; estimated text 517 tok; estimated non-text 383
  tok; generated 142 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: mlx-community/gemma-3-27b-it-qat-4bit</summary>

```text
Title:
The Fenchurch Building (The Walkie-Talkie), London

Description:
The Fenchurch Building, formally known as 20 Fenchurch Street, is illuminated at night, dominating a street scene in London, England; the building’s distinctive curved facade is prominently visible.

Keywords:
Architecture, Building, Buildings, City, Cityscape, Fenchurch Street, Illuminated, London, Modern, Night, Nightscape, Skyscraper, Street, Street signs, The Fenchurch Building (The Walkie-Talkie), Urban, Urban landscape, Walkie Talkie building, Curved facade, Dark sky, Street lighting, Pedestrian crossing, Road surface
```

</details>

⚠️ _Quality Warnings:_

- Keyword count violation (23; expected 10-18)
- Low draft improvement

---

<a id="model-mlx-community-gemma-4-31b-it-4bit"></a>

### ✅ mlx-community/gemma-4-31b-it-4bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Commuting, Street signs, GBR, formally;
  low-draft-improvement
- _Timing:_ Load 3.02s; Gen 7.04s; Total 10.38s
- _Throughput:_ Prompt 534 TPS (905 tok); Gen 26.4 TPS (128 tok)
- _Tokens:_ prompt 905 tok; estimated text 517 tok; estimated non-text 388
  tok; generated 128 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: mlx-community/gemma-4-31b-it-4bit</summary>

```text
Title: The Fenchurch Building (The Walkie-Talkie), London, England, UK

Description: A night view of the modern skyscraper known as 20 Fenchurch Street, situated along a city street in London, England. The scene features illuminated office windows, street lighting, and parked bicycles on the pavement.

Keywords: Architecture, Building, City, Cityscape, England, Fenchurch Street, Illuminated, London, Modern, Night, Nightscape, Skyscraper, Street, The Fenchurch Building (The Walkie-Talkie), UK, Urban, Urban landscape, Walkie Talkie building
```

</details>

⚠️ _Quality Warnings:_

- Low draft improvement

---

<a id="model-mlx-community-pixtral-12b-bf16"></a>

### ⚠️ mlx-community/pixtral-12b-bf16

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Commuting, Modern, The Fenchurch
  Building (The Walkie-Talkie), Walkie Talkie building; low-draft-improvement
- _Timing:_ Load 3.34s; Gen 7.37s; Total 11.01s
- _Throughput:_ Prompt 1,703 TPS (3,433 tok); Gen 20.1 TPS (99 tok)
- _Tokens:_ prompt 3433 tok; estimated text 517 tok; estimated non-text 2916
  tok; generated 99 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: mlx-community/pixtral-12b-bf16</summary>

```text
Title:
- 20 Fenchurch Street, London

Description:
- The illuminated 20 Fenchurch Street, also known as The Walkie-Talkie, stands prominently in a quiet London street at night.

Keywords:
- Architecture, Building, Buildings, City, Cityscape, Fenchurch Street, Illuminated, London, Night, Nightscape, Skyscraper, Street, Street signs, The Walkie-Talkie, Urban, Urban landscape
```

</details>

⚠️ _Quality Warnings:_

- Title length violation (4 words; expected 5-10)
- Low draft improvement

---

<a id="model-mlx-community-gemma-3n-e4b-it-bf16"></a>

### ❌ mlx-community/gemma-3n-E4B-it-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Commuting, Fenchurch Street, Nightscape,
  Street signs, The Fenchurch Building (The Walkie-Talkie); keywords=25;
  unverified-context-copy
- _Timing:_ Load 2.78s; Gen 7.61s; Total 10.71s
- _Throughput:_ Prompt 1,964 TPS (899 tok); Gen 48.2 TPS (323 tok)
- _Tokens:_ prompt 899 tok; estimated text 517 tok; estimated non-text 382
  tok; generated 323 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: mlx-community/gemma-3n-E4B-it-bf16</summary>

```text
## Image Description for Captioning

**Title:** Night View of The Walkie-Talkie Building in London

**Description:** A low-angle, wide shot captures the imposing silhouette of The Walkie-Talkie building at night in London. The skyscraper dominates the frame, its dark, reflective glass facade illuminated by the city lights. The building's distinctive curved top is clearly visible against the deep blue of the night sky, with faint stars scattered above.

The street below is dimly lit by streetlights, with blurred headlights of passing cars creating streaks of light. The surrounding buildings are also visible, their windows glowing with warm interior lights, contrasting with the cool blue of the sky. A few parked bicycles are visible on the sidewalk in the foreground, adding a touch of everyday life to the urban scene.

The overall atmosphere is one of a bustling city at night, with the modern architecture of The Walkie-Talkie standing as a prominent landmark. The image conveys a sense of urban grandeur and the energy of a major city.

**Keywords:** London, The Walkie-Talkie, skyscraper, night, city, urban, architecture, building, street, lights, stars, modern, cityscape, evening, reflection, blue, dark, cars, bicycles.

**Location:** London, England, UK

**Date/Time:** 2026-07-18 22:55:39 BST

**GPS:** 51.511300°N, 0.083400°W
```

</details>

⚠️ _Quality Warnings:_

- Metadata alignment issue (low_metadata_alignment)
- Description sentence violation (8; expected 1-2)
- Keyword count violation (25; expected 10-18)
- Unverified context copy
- suspicious_precision: 2 overly precise numbers

---

<a id="model-mlx-community-glm-46v-flash-mxfp4"></a>

### ✅ mlx-community/GLM-4.6V-Flash-mxfp4

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Commuting, Fenchurch Street, Illuminated, GBR,
  known
- _Timing:_ Load 1.35s; Gen 8.01s; Total 9.67s
- _Throughput:_ Prompt 1,129 TPS (6,647 tok); Gen 80.0 TPS (134 tok)
- _Tokens:_ prompt 6647 tok; estimated text 517 tok; estimated non-text 6130
  tok; generated 134 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: mlx-community/GLM-4.6V-Flash-mxfp4</summary>

```text
Title:
The Fenchurch Building (The Walkie-Talkie), London, England, UK

Description:
The image shows a night scene featuring the distinctive, angular glass facade of The Fenchurch Building (The Walkie-Talkie) in London. The street is relatively quiet with a few cars visible, street lighting illuminating the area, and modern urban architecture surrounding the scene.

Keywords:
The Fenchurch Building (The Walkie-Talkie), Architecture, Building, Buildings, City, Cityscape, London, Modern, Night, Nightscape, Skyscraper, Street, Street signs, Urban, Urban landscape, Walkie Talkie building
```

</details>

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-minicpm-v-46-8bit"></a>

### ⚠️ mlx-community/MiniCPM-V-4.6-8bit

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Cityscape, Commuting, Fenchurch Street,
  Nightscape; formatting=Empty thinking wrapper present
- _Timing:_ Load 0.96s; Gen 8.34s; Total 9.75s
- _Throughput:_ Prompt 350 TPS (1,243 tok); Gen 24.7 TPS (94 tok)
- _Tokens:_ prompt 1243 tok; estimated text 517 tok; estimated non-text 726
  tok; generated 94 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: mlx-community/MiniCPM-V-4.6-8bit</summary>

```text
<think>

</think>

Title:
Nighttime urban street with modern skyscraper

Description: The image shows a modern urban street at night, featuring a prominent illuminated skyscraper under a dark sky, with surrounding buildings and street lighting. Visible details include parked bicycles and street signs, indicating a bustling city environment.

Keywords: Architecture, Buildings, Skyscraper, City, Street, Illuminated, Night, Modern, Urban, London, UK, Evening, Urban landscape
```

</details>

⚠️ _Quality Warnings:_

- Metadata alignment issue (low_metadata_alignment)
- Empty thinking wrapper present

---

<a id="model-jqlive-kimi-vl-a3b-thinking-2506-6bit"></a>

### ❌ jqlive/Kimi-VL-A3B-Thinking-2506-6bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: City, Commuting, Skyscraper, Urban landscape,
  known; thinking trace incomplete
- _Timing:_ Load 1.99s; Gen 8.55s; Total 10.86s
- _Throughput:_ Prompt 1,677 TPS (1,623 tok); Gen 71.0 TPS (500 tok)
- _Tokens:_ prompt 1623 tok; estimated text 517 tok; estimated non-text 1106
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<details>
<summary>Complete generated output: jqlive/Kimi-VL-A3B-Thinking-2506-6bit</summary>

```text
◁think▷Okay, let's tackle this. First, I need to analyze the image and create a catalog record with the specified sections: Title, Description, and Keywords. The user wants it in British English, so I'll make sure to use British terms like "lorry" instead of "truck" if needed, but here it's not necessary.

Looking at the image, the main subject is the Walkie-Talkie building (20 Fenchurch Street) in London, UK. The image is a night scene with the building illuminated, showing its distinctive curved shape. The location is a street in London, England, UK. The time is 2026-07-18 22:55:39 BST, so it's nighttime. The lighting is dark with artificial lights from the building and streetlights. There are cars and bicycles visible, and the street is wet, reflecting lights. The building's glass facade is prominent.

For the title, I need a 5-10 word title. The existing draft was "The Fenchurch Building (The Walkie-Talkie), London, England, UK, GBR, Europe". But maybe shorten it to "The Walkie-Talkie Building, London Nightscape" to be more concise.

Description should be 1-2 sentences. It should mention the building, location, time, lighting, and key elements. So something like "A nighttime view of the Walkie-Talkie building (20 Fenchurch Street) illuminated at night, showcasing its distinctive curved glass facade, with streetlights, bicycles, and cars visible in the foreground, located in London, England, UK."

Keywords need 10-18 terms. From the image: Walkie-Talkie building, London, England, UK, night, nightscape, illuminated, glass facade, streetlights, bicycles, cars, wet street, reflections, urban, modern architecture, cityscape, street signs, buildings, night lighting, dark sky, urban environment, architectural details.

Check if all keywords are visible or from context. "Walkie-Talkie building" is visible. "London" is given. "Night" and "nightscape" are from the image. "Illuminated" is visible. "Glass facade" is visible. "Streetlights" and "bicycles" are visible. "Cars" are visible. "Wet street" and "reflections" from the wet pavement. "Urban"
```

</details>

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)
- Thinking trace incomplete (expected model protocol; opened with ◁think▷)
- Unverified context copy
- Low draft improvement
- Likely capped by max token budget

---

<a id="model-mlx-community-devstral-small-2-24b-instruct-2512-5bit"></a>

### ⚠️ mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Commuting, GBR, formally, 20; keywords=19;
  low-draft-improvement
- _Timing:_ Load 2.17s; Gen 8.58s; Total 11.06s
- _Throughput:_ Prompt 749 TPS (2,707 tok); Gen 30.3 TPS (136 tok)
- _Tokens:_ prompt 2707 tok; estimated text 517 tok; estimated non-text 2190
  tok; generated 136 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit</summary>

```text
Title:
The Fenchurch Building (The Walkie-Talkie), London

Description:
The Fenchurch Building, also known as the Walkie-Talkie, stands illuminated at night in London, England. The distinctive curved glass facade reflects the surrounding city lights, with a quiet street below featuring bicycles and traffic signs.

Keywords:
Architecture, Building, Buildings, Cars, City, Cityscape, Fenchurch Street, Illuminated, London, Modern, Night, Nightscape, Skyscraper, Street, Street signs, The Fenchurch Building (The Walkie-Talkie), Urban, Urban landscape, Walkie Talkie building
```

</details>

⚠️ _Quality Warnings:_

- Keyword count violation (19; expected 10-18)
- Low draft improvement

---

<a id="model-mlx-community-glm-46v-flash-6bit"></a>

### ⚠️ mlx-community/GLM-4.6V-Flash-6bit

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Run a controlled reduced-image or lower-visual-token comparison
  before assigning the context-boundary behaviour to mlx, mlx-vlm, or the
  model.
- _Key signals:_ output/prompt=2.29%; visual input burden=92%; missing
  sections: title; missing terms: Commuting, The Fenchurch Building (The
  Walkie-Talkie), GBR, known, formally
- _Timing:_ Load 1.41s; Gen 8.74s; Total 10.46s
- _Throughput:_ Prompt 1,173 TPS (6,647 tok); Gen 57.7 TPS (152 tok)
- _Tokens:_ prompt 6647 tok; estimated text 517 tok; estimated non-text 6130
  tok; generated 152 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: mlx-community/GLM-4.6V-Flash-6bit</summary>

```text
<|begin_of_box|>Title:
The Walkie-Talkie Building at Night

Description:
A nighttime urban scene featuring the distinctive, angular glass facade of the Walkie-Talkie building (20 Fenchurch Street) in London. The street is dimly lit with streetlights, showing a few vehicles with headlights on, and some green rental bikes parked on the left sidewalk.

Keywords:
Architecture, Building, Buildings, Cars, City, Cityscape, Fenchurch Street, Illuminated, London, Modern, Night, Nightscape, Skyscraper, Street, Street signs, The Walkie-Talkie Building, Urban, Urban landscape, Walkie-Talkie building, Glass facade, Nighttime urban scene, Green rental bikes, Headlights on vehicles<|end_of_box|>
```

</details>

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context_budget
- Missing sections (title)
- Keyword count violation (23; expected 10-18)
- Low draft improvement
- Contains unexpected table

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-8bit"></a>

### ❌ mlx-community/Kimi-VL-A3B-Thinking-8bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Cars, City, Nightscape, Street signs, Urban;
  thinking trace present
- _Timing:_ Load 2.19s; Gen 9.53s; Total 12.03s
- _Throughput:_ Prompt 1,668 TPS (1,623 tok); Gen 62.6 TPS (500 tok)
- _Tokens:_ prompt 1623 tok; estimated text 517 tok; estimated non-text 1106
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<details>
<summary>Complete generated output: mlx-community/Kimi-VL-A3B-Thinking-8bit</summary>

```text
◁think▷Okay, let's tackle this metadata cataloging task. First, I need to analyze the image based on the given context and the draft metadata provided. The image shows a tall, modern skyscraper at night. The building has a distinctive curved glass facade with illuminated windows. The location is specified as England, UK, specifically in the district of London, near Fenchurch Street. The photo was taken on July 18, 2026, at 10:55 PM BST. The GPS coordinates are 51.5113°N, 0.0834°W.

The existing draft metadata mentions "The Fenchurch Building (The Walkie-Talkie)" and "20 Fenchurch Street." The building is known for its unique shape, which resembles a walkie-talkie, hence the nickname. The image shows the building at night with its glass panels lit up, which is a key visual detail. The surrounding area includes a street scene with some vehicles and streetlights, but the main subject is clearly the skyscraper.

For the title, I should keep "The Fenchurch Building (The Walkie-Talkie)" as it's the official name and nickname. The location is already provided in the context, so I can include that. The draft title is already accurate, so no changes needed there.

For the description, I need to combine the authoritative context with visible details. The building's curved glass facade and illuminated windows are prominent. The street scene at night with vehicles and streetlights adds context. The GPS coordinates and date/time are part of the authoritative context but shouldn't be included in the description since they're not visually observable. The existing description mentions "Walkie Talkie building" which is correct, so I can use that. The keywords should include the official name, location, and other visible elements like architecture, night, cityscape, etc. I should avoid adding speculative terms like "commuting" unless it's clearly visible, which it isn't. The existing keywords are a good start but can be refined to include "glass facade" and "streetlights" based on the image.

I need to ensure that I don't add any inferred details. The building's nickname is confirmed by the context, so that's safe. The image shows it's a skyscraper, so that's included. The keywords should be concise and accurate. Let me compile the final entries based on this analysis.◁/think▷Title: The Fenchurch
```

</details>

⚠️ _Quality Warnings:_

- Excessive verbosity
- Missing sections (title, description, keywords)
- Thinking trace present (expected model protocol; ◁think▷)
- Likely capped by max token budget
- suspicious_precision: 2 overly precise numbers

---

<a id="model-microsoft-phi-35-vision-instruct"></a>

### ❌ microsoft/Phi-3.5-vision-instruct

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ hit token cap (500); missing terms: Cars, Commuting,
  Fenchurch Street, Nightscape, Street; keyword duplication=83%; repetitive
  token=phrase: "modern, illuminated, architect..."
- _Timing:_ Load 0.92s; Gen 9.55s; Total 10.77s
- _Throughput:_ Prompt 3,958 TPS (1,468 tok); Gen 57.2 TPS (500 tok)
- _Tokens:_ prompt 1468 tok; estimated text 517 tok; estimated non-text 951
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<details>
<summary>Complete generated output: microsoft/Phi-3.5-vision-instruct</summary>

```text
Title: The Fenchurch Building (The Walkie-Talkie), London, England, UK, GBR, Europe

Description: The iconic Fenchurch Building, also known as The Walkie-Talkie, stands tall in the heart of London's financial district. Its distinctive curved facade is illuminated against the night sky, creating a striking contrast with the surrounding modern architecture. The building's reflective glass surface mirrors the lights of the city, adding to its imposing presence.

Keywords: Fenchurch Building, The Walkie-Talkie, London, England, UK, GBR, Europe, modern architecture, illuminated, cityscape, skyscraper, urban landscape, night, modern, illuminated, reflective glass, financial district, city lights, urban, architecture, buildings, illumination, nighttime, skyline, modern, urban, city, lighting, modern, illuminated, architecture, buildings, urban, cityscape, modern, illuminated, architecture, buildings, urban, city, lighting, modern, illuminated, architecture, buildings, urban, cityscape, modern, illuminated, architecture, buildings, urban, city, lighting, modern, illuminated, architecture, buildings, urban, cityscape, modern, illuminated, architecture, buildings, urban, city, lighting, modern, illuminated, architecture, buildings, urban, cityscape, modern, illuminated, architecture, buildings, urban, city, lighting, modern, illuminated, architecture, buildings, urban, cityscape, modern, illuminated, architecture, buildings, urban, city, lighting, modern, illuminated, architecture, buildings, urban, cityscape, modern, illuminated, architecture, buildings, urban, city, lighting, modern, illuminated, architecture, buildings, urban, cityscape, modern, illuminated, architecture, buildings, urban, city, lighting, modern, illuminated, architecture, buildings, urban, cityscape, modern, illuminated, architecture, buildings, urban, city, lighting, modern, illuminated, architecture, buildings, urban, cityscape, modern, illuminated, architecture, buildings, urban, city, lighting, modern,
```

</details>

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "modern, illuminated, architect...")
- Generation loop (repetitive_tail)
- Title length violation (11 words; expected 5-10)
- Description sentence violation (3; expected 1-2)
- Keyword count violation (146; expected 10-18)
- Keyword duplication (83% duplicated terms)
- Low draft improvement
- Likely capped by max token budget

---

<a id="model-mlx-community-gemma-3-27b-it-qat-8bit"></a>

### ⚠️ mlx-community/gemma-3-27b-it-qat-8bit

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Commuting, Walkie Talkie building, GBR,
  formally; keywords=21; low-draft-improvement
- _Timing:_ Load 3.84s; Gen 10.07s; Total 14.22s
- _Throughput:_ Prompt 583 TPS (900 tok); Gen 17.4 TPS (140 tok)
- _Tokens:_ prompt 900 tok; estimated text 517 tok; estimated non-text 383
  tok; generated 140 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: mlx-community/gemma-3-27b-it-qat-8bit</summary>

```text
Title:
The Fenchurch Building (The Walkie-Talkie), London

Description:
The Fenchurch Building, also known as 20 Fenchurch Street, is illuminated at night, viewed from a street level perspective in London, with a dark blue sky overhead. Streetlights cast a warm glow on the pavement and surrounding buildings.

Keywords:
Architecture, Building, Buildings, City, Cityscape, Fenchurch Street, Illuminated, London, Modern, Night, Nightscape, Skyscraper, Street, Street signs, The Fenchurch Building (The Walkie-Talkie), Urban, Urban landscape, Dark sky, Exterior, Pavement, Street lighting
```

</details>

⚠️ _Quality Warnings:_

- Keyword count violation (21; expected 10-18)
- Low draft improvement

---

<a id="model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16"></a>

### ❌ mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: City, Commuting, GBR, known, formally;
  text-sanity=numeric_loop
- _Timing:_ Load 4.82s; Gen 10.72s; Total 15.85s
- _Throughput:_ Prompt 1,092 TPS (1,956 tok); Gen 59.6 TPS (500 tok)
- _Tokens:_ prompt 1956 tok; estimated text 517 tok; estimated non-text 1439
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<details>
<summary>Complete generated output: mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16</summary>

```text
Let's tackle this step by step. First, the title needs to be 5-10 words, concrete and factual. The authoritative context gives location as London, England, UK, and the building is The Fenchurch Building (The Walkie-Talkie). So the title should be something like "The Fenchurch Building (The Walkie-Talkie), London, UK" – that's 7 words, fits the requirement.

Next, the description. The draft says it's the Walkie-Talkie building at 20 Fenchurch Street. The image is a nightscape of the building, so I need to combine the authoritative context (location, building name) with the visible details (night, illuminated, skyscraper, street elements like lights, cars, bikes). Wait, the draft description is short, but let's make it more detailed but still factual. The image has the building's distinctive curved glass facade, lit up at night, with streetlights, cars, bikes, and the surrounding urban area. So the description should mention the building's name, location, night setting, illuminated facade, and key visible elements like streetlights, vehicles, and urban environment.

Then keywords. The draft has some, but let's check if they're accurate. The building is a skyscraper, modern, nightscape, cityscape, London, UK, Europe, architecture, illuminated, street, cars, bikes, street signs, urban landscape. Wait, the authoritative context includes GPS, but keywords should be visible or authoritative. Let's make sure they're unique and cover the main points. The building's name is important, so "Fenchurch Building (The Walkie-Talkie)" should be there. Also, "20 Fenchurch Street" is the address, which is authoritative. Let's list them: Architecture, Building, Fenchurch Building (The Walkie-Talkie), 20 Fenchurch Street, London, UK, Europe, Night, Nightscape, Skyscraper, Modern, Illuminated, Cityscape, Urban, Street, Cars, Bikes, Street Signs, Urban Landscape. Wait, but keywords need to be 10-18. Let's count: Architecture, Building, Fenchurch Building (The Walkie-Talk
```

</details>

⚠️ _Quality Warnings:_

- Text sanity issue (numeric_loop)
- Generation loop (numeric_loop)
- Missing sections (title, description, keywords)
- Low draft improvement
- Likely capped by max token budget

---

<a id="model-mlx-community-paligemma2-3b-pt-896-4bit"></a>

### ❌ mlx-community/paligemma2-3b-pt-896-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Architecture, Building, Buildings, Cars, City;
  excessive bullets=42
- _Timing:_ Load 1.72s; Gen 13.75s; Total 15.79s
- _Throughput:_ Prompt 3,893 TPS (4,724 tok); Gen 42.4 TPS (500 tok)
- _Tokens:_ prompt 4724 tok; estimated text 517 tok; estimated non-text 4207
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<details>
<summary>Complete generated output: mlx-community/paligemma2-3b-pt-896-4bit</summary>

```text
- Do not output the description text after the label.
- Do not output the title text after the label.
- Do not output the description text after the label.
- Do not output the title text after the label.
- Do not output the description text after the label.
- Do not output the title text after the label.
- Do not output the description text after the label.
- Do not output the title text after the label.
- Do not output the description text after the label.
- Do not output the title text after the label.
- Do not output the description text after the label.
- Do not output the title text after the label.
- Do not output the description text after the label.
- Do not output the title text after the label.
- Do not output the description text after the label.
- Do not output the title text after the label.
- Do not output the description text after the label.
- Do not output the title text after the label.
- Do not output the description text after the label.
- Do not output the title text after the label.
- Do not output the description text after the label.
- Do not output the title text after the label.
- Do not output the description text after the label.
- Do not output the title text after the label.
- Do not output the description text after the label.
- Do not output the title text after the label.
- Do not output the description text after the label.
- Do not output the title text after the label.
- Do not output the description text after the label.
- Do not output the title text after the label.
- Do not output the description text after the label.
- Do not output the title text after the label.
- Do not output the description text after the label.
- Do not output the title text after the label.
- Do not output the description text after the label.
- Do not output the title text after the label.
- Do not output the description text after the label.
- Do not output the title text after the label.
- Do not output the description text after the label.
- Do not output the title text after the label.
- Do not output the description text after the label.
- Do not output the title text after
```

</details>

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "- do not output...")
- Excessive bullet points (42)
- Context ignored (missing: Architecture, Building, Buildings, Cars, City)
- Generation loop (repetitive_tail)
- Metadata alignment issue (low_metadata_alignment)
- Missing sections (title, description, keywords)
- Low draft improvement
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-qwen3-vl-2b-thinking-bf16"></a>

### ⚠️ mlx-community/Qwen3-VL-2B-Thinking-bf16

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Run a controlled reduced-image or lower-visual-token comparison
  before assigning the context-boundary behaviour to mlx, mlx-vlm, or the
  model.
- _Key signals:_ Output appears truncated to about 9 tokens.; At mixed burden
  (16844 tokens), output stayed unusually short (9 tokens; ratio 0.1%; weak
  text signal truncated).; output/prompt=0.05%; mixed burden=97%
- _Timing:_ Load 0.73s; Gen 14.30s; Total 15.34s
- _Throughput:_ Prompt 1,252 TPS (16,844 tok); Gen 103 TPS (9 tok)
- _Tokens:_ prompt 16844 tok; estimated text 517 tok; estimated non-text 16327
  tok; generated 9 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: mlx-community/Qwen3-VL-2B-Thinking-bf16</summary>

```text
 1000000
```

</details>

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context_budget
- ⚠️HARNESS:long_context
- Context ignored (missing: Architecture, Building, Buildings, Cars, City)
- Metadata alignment issue (low_metadata_alignment)
- Low draft improvement
- Ignores trusted hints
- output:truncated(9tok)
- long_context_low_ratio(0.1%;16844->9;truncated)

---

<a id="model-mlx-community-apriel-15-15b-thinker-6bit-mlx"></a>

### ❌ mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, keywords;
  missing terms: Architecture, Cars, City, Commuting, Modern; reasoning leak
- _Timing:_ Load 1.66s; Gen 14.42s; Total 16.39s
- _Throughput:_ Prompt 1,557 TPS (3,524 tok); Gen 42.7 TPS (500 tok)
- _Tokens:_ prompt 3524 tok; estimated text 517 tok; estimated non-text 3007
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<details>
<summary>Complete generated output: mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX</summary>

```text
Here are my reasoning steps:
We need to produce a catalog record with three sections: Title, Description, Keywords. The image is a nighttime cityscape of a tall skyscraper, likely the Walkie Talkie building (20 Fenchurch Street) in London. The authoritative context includes location terms: England, Europe, UK, district, United Kingdom. Capture date/time: 2026-07-18 22:55:39 BST. GPS: 51.511300°N, 0.083400°W. The draft metadata includes title: "The Fenchurch Building (The Walkie-Talkie), London, England, UK, GBR, Europe". Description: "Walkie Talkie building known formally as 20 Fenchurch Street." Keywords: many.

We need to produce a title of 5-10 words, concrete and factual; can use authoritative context. Title should be something like "20 Fenchurch Street (Walkie‑Talkie), London, England". That's 7 words? Let's count: "20" (1), "Fenchurch" (2), "Street" (3), "(Walkie‑Talkie)," (4), "London," (5), "England" (6). That's 6 words. That's within 5-10. Or "The Walkie‑Talkie, 20 Fenchurch Street, London". That's 8 words. Let's decide.

Description: 1-2 factual sentences combining supplied authoritative context with main visible subject, setting, lighting, action, distinctive visible details. So we need to mention that the image shows the Walkie Talkie building at night, illuminated, with surrounding streets, streetlights, etc. Use authoritative context: location is London, England, UK. Capture date/time: 2026-07-18 22:55:39 BST. But we should not claim that the date/time is visible; we can incorporate it as context. The description should be factual, not speculation. So something like: "The photograph captures 20 Fenchurch Street, the Walkie‑Talkie skyscraper in London, illuminated against a deep blue night sky, with surrounding streets and streetlights visible." That's one sentence. Could add second
```

</details>

⚠️ _Quality Warnings:_

- Excessive verbosity
- Text sanity issue (numeric_loop)
- Generation loop (numeric_loop)
- Metadata alignment issue (low_metadata_alignment)
- Missing sections (title, keywords)
- Description sentence violation (8; expected 1-2)
- Reasoning leak (here are my reasoning steps)
- Unverified context copy
- Likely capped by max token budget
- suspicious_precision: 2 overly precise numbers

---

<a id="model-mlx-community-glm-46v-nvfp4"></a>

### ⚠️ mlx-community/GLM-4.6V-nvfp4

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Run a controlled reduced-image or lower-visual-token comparison
  before assigning the context-boundary behaviour to mlx, mlx-vlm, or the
  model.
- _Key signals:_ output/prompt=2.20%; visual input burden=92%; missing
  sections: title; missing terms: Commuting, known, formally, 20
- _Timing:_ Load 5.50s; Gen 15.53s; Total 21.40s
- _Throughput:_ Prompt 569 TPS (6,647 tok); Gen 43.0 TPS (146 tok)
- _Tokens:_ prompt 6647 tok; estimated text 517 tok; estimated non-text 6130
  tok; generated 146 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: mlx-community/GLM-4.6V-nvfp4</summary>

```text
<|begin_of_box|>Title:
The Fenchurch Building (The Walkie-Talkie), London, England, UK, GBR, Europe

Description:
Nighttime urban street scene featuring the Fenchurch Building (The Walkie-Talkie) skyscraper, with illuminated windows, modern architecture, and a quiet city street with parked bicycles, street signs, and distant moving cars under a dark blue sky.

Keywords:
Architecture, Building, Buildings, Cars, City, Cityscape, Fenchurch Street, Illuminated, London, Modern, Night, Nightscape, Skyscraper, Street, Street signs, The Fenchurch Building (The Walkie-Talkie), Urban, Urban landscape, Walkie Talkie building<|end_of_box|>
```

</details>

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context_budget
- Missing sections (title)
- Keyword count violation (19; expected 10-18)
- Low draft improvement
- Contains unexpected table

---

<a id="model-mlx-community-gemma-4-31b-bf16"></a>

### ❌ mlx-community/gemma-4-31b-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Architecture, Cityscape, Commuting, Fenchurch Street, London;
  low-draft-improvement
- _Timing:_ Load 7.58s; Gen 16.27s; Total 24.16s
- _Throughput:_ Prompt 185 TPS (893 tok); Gen 7.5 TPS (82 tok)
- _Tokens:_ prompt 893 tok; estimated text 517 tok; estimated non-text 376
  tok; generated 82 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: mlx-community/gemma-4-31b-bf16</summary>

```text


Image:
- The image shows a tall, modern skyscraper with a distinctive curved design, illuminated at night. The building is surrounded by other urban structures, including smaller buildings and streetlights. The scene is set in a city environment, with a street visible in the foreground, featuring parked bicycles and a few cars. The sky is dark, indicating nighttime, and the overall atmosphere is urban and contemporary.
```

</details>

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)
- Low draft improvement
- Degrades trusted hints

---

<a id="model-mlx-community-lfm25-vl-16b-bf16"></a>

### ❌ mlx-community/LFM2.5-VL-1.6B-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: formally; keywords=30;
  unverified-context-copy; low-draft-improvement
- _Timing:_ Load 1.59s; Gen 18.23s; Total 20.72s
- _Throughput:_ Prompt 1,083 TPS (890 tok); Gen 18.0 TPS (294 tok)
- _Tokens:_ prompt 890 tok; estimated text 517 tok; estimated non-text 373
  tok; generated 294 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: mlx-community/LFM2.5-VL-1.6B-bf16</summary>

```text
Title:
- London, England, United Kingdom, 20 Fenchurch Street, The Walkie-Talkie, 20 Fenchurch Street, London, England, United Kingdom, GBR, Europe

Description:
- The Fenchurch Building (The Walkie-Talkie) is a prominent skyscraper located at 20 Fenchurch Street in London, England, United Kingdom, GBR, Europe. The building is known for its distinctive curved glass façade and is a notable example of modern architecture in the city. The image captures the building at night, with its illuminated windows and sleek design standing out against the dark sky. The surrounding area appears to be a busy urban street, with other buildings and street signs visible. The photo was taken on 2026-07-18 at 22:55:39 BST, providing a clear view of the building's impressive height and sleek design.

Keywords:
- London, England, United Kingdom, 20 Fenchurch Street, The Walkie-Talkie, London, England, United Kingdom, GBR, Europe, Architecture, Building, Buildings, Cars, City, Cityscape, Commuting, Fenchurch Street, Illuminated, London, Modern, Night, Nightscape, Skyscraper, Street, Street signs, The Fenchurch Building (The Walkie-Talkie), Urban, Urban landscape, Walkie Talkie building
```

</details>

⚠️ _Quality Warnings:_

- Title length violation (19 words; expected 5-10)
- Description sentence violation (5; expected 1-2)
- Keyword count violation (30; expected 10-18)
- Unverified context copy
- Low draft improvement

---

<a id="model-qwen-qwen3-vl-2b-instruct"></a>

### ❌ Qwen/Qwen3-VL-2B-Instruct

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect KV/cache behavior, memory pressure, and long-context
  execution.
- _Key signals:_ At mixed burden (16842 tokens), output became repetitive.;
  hit token cap (500); missing sections: title, description, keywords; missing
  terms: Architecture, Building, Buildings, Cars, City
- _Timing:_ Load 0.68s; Gen 19.30s; Total 20.26s
- _Throughput:_ Prompt 1,275 TPS (16,842 tok); Gen 93.3 TPS (500 tok)
- _Tokens:_ prompt 16842 tok; estimated text 517 tok; estimated non-text 16325
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<details>
<summary>Complete generated output: Qwen/Qwen3-VL-2B-Instruct</summary>

```text
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
```

</details>

⚠️ _Quality Warnings:_

- ⚠️HARNESS:long_context
- Repetitive output (-)
- Context ignored (missing: Architecture, Building, Buildings, Cars, City)
- Output degeneration (character_loop: ' -' repeated)
- Text sanity issue (gibberish(token_noise))
- Generation loop (repetitive_tail)
- Metadata alignment issue (low_metadata_alignment)
- Missing sections (title, description, keywords)
- Low draft improvement
- Likely capped by max token budget
- Ignores trusted hints
- long_context_repetition(16842tok)

---

<a id="model-mlx-community-x-reasoner-7b-8bit"></a>

### ⚠️ mlx-community/X-Reasoner-7B-8bit

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Run a controlled reduced-image or lower-visual-token comparison
  before assigning the context-boundary behaviour to mlx, mlx-vlm, or the
  model.
- _Key signals:_ Output appears truncated to about 9 tokens.; At mixed burden
  (16853 tokens), output stayed unusually short (9 tokens; ratio 0.1%; weak
  text signal truncated).; output/prompt=0.05%; mixed burden=97%
- _Timing:_ Load 1.79s; Gen 19.38s; Total 21.48s
- _Throughput:_ Prompt 911 TPS (16,853 tok); Gen 64.1 TPS (9 tok)
- _Tokens:_ prompt 16853 tok; estimated text 517 tok; estimated non-text 16336
  tok; generated 9 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: mlx-community/X-Reasoner-7B-8bit</summary>

```text
<|im_start|>
 addCriterion("image:caption")
```

</details>

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context_budget
- ⚠️HARNESS:long_context
- Context ignored (missing: Architecture, Building, Buildings, Cars, City)
- Metadata alignment issue (low_metadata_alignment)
- Low draft improvement
- Ignores trusted hints
- output:truncated(9tok)
- long_context_low_ratio(0.1%;16853->9;truncated)

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-mxfp4"></a>

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-mxfp4

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, City, Commuting, Fenchurch Street,
  Nightscape
- _Timing:_ Load 3.45s; Gen 23.44s; Total 28.24s
- _Throughput:_ Prompt 163 TPS (3,240 tok); Gen 58.0 TPS (122 tok)
- _Tokens:_ prompt 3240 tok; estimated text 517 tok; estimated non-text 2723
  tok; generated 122 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: mlx-community/Ministral-3-14B-Instruct-2512-mxfp4</summary>

```text
**Title:**
Modern Skyscraper at Dusk, London Urban Street

**Description:**
A nighttime urban street in London features a distinctive, slanted-glass skyscraper with illuminated interiors, flanked by older buildings. The scene includes empty roads, streetlights, and a few parked vehicles, with a pedestrian bridge connecting adjacent structures.

**Keywords:**
Skyscraper, night photography, urban street, London, modern architecture, glass facade, illuminated building, streetlights, empty road, pedestrian bridge, cityscape, European urban, dusk lighting, concrete structures, traffic signs
```

</details>

⚠️ _Quality Warnings:_

- Metadata alignment issue (low_metadata_alignment)

---

<a id="model-mlx-community-paligemma2-3b-ft-docci-448-bf16"></a>

### ❌ mlx-community/paligemma2-3b-ft-docci-448-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Architecture, Building, Buildings, Cars, City;
  low-draft-improvement
- _Timing:_ Load 2.01s; Gen 27.34s; Total 29.66s
- _Throughput:_ Prompt 3,399 TPS (1,652 tok); Gen 19.0 TPS (500 tok)
- _Tokens:_ prompt 1652 tok; estimated text 517 tok; estimated non-text 1135
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<details>
<summary>Complete generated output: mlx-community/paligemma2-3b-ft-docci-448-bf16</summary>

```text
- Do not copy-prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent.
- Use this factual context where it improves the catalogue record; do not claim that contextual facts are visually observable.
- Do not copy-prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent.
- Use this factual context where it improves the catalogue record; do not claim that contextual facts are visually observable.
- Do not copy-prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent.
- Use this factual context where it improves the catalogue record; do not claim that contextual facts are visually observable.
- Do not copy-prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent.
- Use this factual context where it improves the catalogue record; do not claim that contextual facts are visually observable.
- Do not copy-prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent.
- Use this factual context where it improves the catalogue record; do not claim that contextual facts are visually observable.
- Do not copy-prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent.
- Use this factual context where it improves the catalogue record; do not claim that contextual facts are visually observable.
- Do not copy-prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent.
- Use this factual context where it improves the catalogue record; do not claim that contextual facts are visually observable.
- Do not copy-prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent.
- Use this factual context where it improves the catalogue record; do not claim that contextual facts are visually observable.
- Do not copy
```

</details>

⚠️ _Quality Warnings:_

- Context ignored (missing: Architecture, Building, Buildings, Cars, City)
- Metadata alignment issue (low_metadata_alignment)
- Missing sections (title, description, keywords)
- Low draft improvement
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-llama-32-11b-vision-instruct-8bit"></a>

### ⚠️ mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: formally, 20; keywords=20;
  low-draft-improvement
- _Timing:_ Load 2.11s; Gen 36.65s; Total 39.33s
- _Throughput:_ Prompt 31.2 TPS (595 tok); Gen 11.5 TPS (187 tok)
- _Tokens:_ prompt 595 tok; estimated text 517 tok; estimated non-text 78 tok;
  generated 187 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: mlx-community/Llama-3.2-11B-Vision-Instruct-8bit</summary>

```text
Title:
The Fenchurch Building (The Walkie-Talkie), London, England, UK, GBR, Europe

Description:
The image depicts the Fenchurch Building, also known as the Walkie-Talkie, in London, England, UK, at night. The building is a prominent feature of the city's skyline, with its distinctive curved shape and modern architecture. The surrounding area is well-lit, with streetlights and building lights illuminating the scene. The image captures the urban landscape of London, with the Fenchurch Building standing out as a notable landmark.

Keywords:
Architecture, Building, Buildings, Cars, City, Cityscape, Commuting, Fenchurch Street, Illuminated, London, Modern, Night, Nightscape, Skyscraper, Street, Street signs, The Fenchurch Building (The Walkie-Talkie), Urban, Urban landscape, Walkie Talkie building
```

</details>

⚠️ _Quality Warnings:_

- Title length violation (11 words; expected 5-10)
- Description sentence violation (4; expected 1-2)
- Keyword count violation (20; expected 10-18)
- Low draft improvement

---

<a id="model-mlx-community-phi-35-vision-instruct-bf16"></a>

### ❌ mlx-community/Phi-3.5-vision-instruct-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ hit token cap (500); missing terms: Cars, Commuting,
  Fenchurch Street, Nightscape, Street; keyword duplication=83%; repetitive
  token=phrase: "modern, illuminated, architect..."
- _Timing:_ Load 0.73s; Gen 47.09s; Total 48.11s
- _Throughput:_ Prompt 3,538 TPS (1,468 tok); Gen 10.8 TPS (500 tok)
- _Tokens:_ prompt 1468 tok; estimated text 517 tok; estimated non-text 951
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<details>
<summary>Complete generated output: mlx-community/Phi-3.5-vision-instruct-bf16</summary>

```text
Title: The Fenchurch Building (The Walkie-Talkie), London, England, UK, GBR, Europe

Description: The iconic Fenchurch Building, also known as The Walkie-Talkie, stands tall in the heart of London's financial district. Its distinctive curved facade is illuminated against the night sky, creating a striking contrast with the surrounding modern architecture. The building's reflective glass surface mirrors the lights of the city, adding to its imposing presence.

Keywords: Fenchurch Building, The Walkie-Talkie, London, England, UK, GBR, Europe, modern architecture, illuminated, cityscape, skyscraper, urban landscape, night, modern, illuminated, reflective glass, financial district, city lights, urban, architecture, buildings, illumination, nighttime, skyline, modern, urban, city, lighting, modern, illuminated, architecture, buildings, urban, cityscape, modern, illuminated, architecture, buildings, urban, city, lighting, modern, illuminated, architecture, buildings, urban, cityscape, modern, illuminated, architecture, buildings, urban, city, lighting, modern, illuminated, architecture, buildings, urban, cityscape, modern, illuminated, architecture, buildings, urban, city, lighting, modern, illuminated, architecture, buildings, urban, cityscape, modern, illuminated, architecture, buildings, urban, city, lighting, modern, illuminated, architecture, buildings, urban, cityscape, modern, illuminated, architecture, buildings, urban, city, lighting, modern, illuminated, architecture, buildings, urban, cityscape, modern, illuminated, architecture, buildings, urban, city, lighting, modern, illuminated, architecture, buildings, urban, cityscape, modern, illuminated, architecture, buildings, urban, city, lighting, modern, illuminated, architecture, buildings, urban, cityscape, modern, illuminated, architecture, buildings, urban, city, lighting, modern, illuminated, architecture, buildings, urban, cityscape, modern, illuminated, architecture, buildings, urban, city, lighting, modern,
```

</details>

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "modern, illuminated, architect...")
- Generation loop (repetitive_tail)
- Title length violation (11 words; expected 5-10)
- Description sentence violation (3; expected 1-2)
- Keyword count violation (146; expected 10-18)
- Keyword duplication (83% duplicated terms)
- Low draft improvement
- Likely capped by max token budget

---

<a id="model-mlx-community-molmo-7b-d-0924-bf16"></a>

### ⚠️ mlx-community/Molmo-7B-D-0924-bf16

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ keywords=20; low-draft-improvement
- _Timing:_ Load 2.14s; Gen 51.15s; Total 53.58s
- _Throughput:_ Prompt 48.1 TPS (1,812 tok); Gen 14.2 TPS (175 tok)
- _Tokens:_ prompt 1812 tok; estimated text 517 tok; estimated non-text 1295
  tok; generated 175 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: mlx-community/Molmo-7B-D-0924-bf16</summary>

```text
 Never output the prompt again.

Title: The Fenchurch Building (The Walkie-Talkie), London, England, UK, GBR, Europe

Description: The iconic Walkie Talkie building, formally known as 20 Fenchurch Street, stands illuminated against the night sky in London's financial district. Its distinctive curved glass facade reflects the surrounding cityscape, while a few cars and pedestrians are visible on the nearby street. The building's unique architecture and prominent location make it a recognizable landmark of the London skyline.

Keywords: Architecture, Building, Buildings, Cars, City, Cityscape, Commuting, Fenchurch Street, Illuminated, London, Modern, Night, Nightscape, Skyscraper, Street, Street signs, The Fenchurch Building (The Walkie-Talkie), Urban, Urban landscape, Walkie Talkie building
```

</details>

⚠️ _Quality Warnings:_

- Title length violation (11 words; expected 5-10)
- Description sentence violation (3; expected 1-2)
- Keyword count violation (20; expected 10-18)
- Low draft improvement

---

<a id="model-mlx-community-qwen35-9b-mlx-4bit"></a>

### ✅ mlx-community/Qwen3.5-9B-MLX-4bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Commuting, The Fenchurch Building (The
  Walkie-Talkie), GBR, formally
- _Timing:_ Load 1.76s; Gen 56.48s; Total 58.55s
- _Throughput:_ Prompt 311 TPS (16,872 tok); Gen 92.1 TPS (137 tok)
- _Tokens:_ prompt 16872 tok; estimated text 517 tok; estimated non-text 16355
  tok; generated 137 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: mlx-community/Qwen3.5-9B-MLX-4bit</summary>

```text
Title:
The Fenchurch Building, London, UK, Night

Description:
The Fenchurch Building, also known as 20 Fenchurch Street, is a modern skyscraper in London, England, visible at night with its distinctive curved glass façade and illuminated upper floors. The surrounding urban streetscape includes streetlights, road signs, parked bicycles, and nearby commercial buildings under a deep blue twilight sky.

Keywords:
Architecture, Building, Buildings, City, Cityscape, Fenchurch Street, Illuminated, London, Modern, Night, Nightscape, Skyscraper, Street, Street signs, Urban, Urban landscape, Walkie Talkie building
```

</details>

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-qwen3-vl-2b-instruct-bf16"></a>

### ❌ mlx-community/Qwen3-VL-2B-Instruct-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect KV/cache behavior, memory pressure, and long-context
  execution.
- _Key signals:_ At mixed burden (16842 tokens), output became repetitive.;
  hit token cap (500); missing sections: title, description, keywords; missing
  terms: Architecture, Building, Buildings, Cars, City
- _Timing:_ Load 0.56s; Gen 56.84s; Total 57.68s
- _Throughput:_ Prompt 332 TPS (16,842 tok); Gen 92.3 TPS (500 tok)
- _Tokens:_ prompt 16842 tok; estimated text 517 tok; estimated non-text 16325
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<details>
<summary>Complete generated output: mlx-community/Qwen3-VL-2B-Instruct-bf16</summary>

```text
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
```

</details>

⚠️ _Quality Warnings:_

- ⚠️HARNESS:long_context
- Repetitive output (-)
- Context ignored (missing: Architecture, Building, Buildings, Cars, City)
- Output degeneration (character_loop: ' -' repeated)
- Text sanity issue (gibberish(token_noise))
- Generation loop (repetitive_tail)
- Metadata alignment issue (low_metadata_alignment)
- Missing sections (title, description, keywords)
- Low draft improvement
- Likely capped by max token budget
- Ignores trusted hints
- long_context_repetition(16842tok)

---

<a id="model-mlx-community-qwen35-35b-a3b-4bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-4bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, City, Commuting, Fenchurch Street,
  Nightscape
- _Timing:_ Load 2.85s; Gen 60.26s; Total 63.41s
- _Throughput:_ Prompt 290 TPS (16,872 tok); Gen 107 TPS (133 tok)
- _Tokens:_ prompt 16872 tok; estimated text 517 tok; estimated non-text 16355
  tok; generated 133 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: mlx-community/Qwen3.5-35B-A3B-4bit</summary>

```text
Title:
The Fenchurch Building at night in London

Description:
The Fenchurch Building, also known as the Walkie-Talkie, dominates the evening skyline of London, England, with its distinctive curved glass facade reflecting the deep blue twilight. The street level is illuminated by warm streetlights and interior office lights, highlighting the entrance and surrounding urban infrastructure.

Keywords:
The Fenchurch Building, Walkie-Talkie, London, England, UK, Europe, Night, Skyscraper, Glass facade, Street lights, Urban landscape, Architecture, Cityscape, Blue sky, Office lights, Street level, Modern design, Metropolitan
```

</details>

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-qwen35-35b-a3b-6bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-6bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Cityscape, Commuting, Modern, The
  Fenchurch Building (The Walkie-Talkie)
- _Timing:_ Load 3.62s; Gen 62.11s; Total 66.04s
- _Throughput:_ Prompt 282 TPS (16,872 tok); Gen 91.2 TPS (132 tok)
- _Tokens:_ prompt 16872 tok; estimated text 517 tok; estimated non-text 16355
  tok; generated 132 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: mlx-community/Qwen3.5-35B-A3B-6bit</summary>

```text
Title:
The Fenchurch Building at night in London

Description:
The Fenchurch Building, also known as the Walkie-Talkie, stands prominently in the City of London against a deep blue evening sky. The street level scene features illuminated shop fronts, parked bicycles, and road signage on a quiet urban thoroughfare.

Keywords:
The Fenchurch Building, Walkie-Talkie, 20 Fenchurch Street, London, City of London, England, UK, Europe, Night, Nightscape, Skyscraper, Architecture, Street, Street signs, Bicycles, Illuminated, Urban, Blue sky
```

</details>

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-qwen35-35b-a3b-bf16"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-bf16

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, City, Commuting, Modern, The Fenchurch
  Building (The Walkie-Talkie)
- _Timing:_ Load 10.71s; Gen 67.45s; Total 78.47s
- _Throughput:_ Prompt 261 TPS (16,872 tok); Gen 65.8 TPS (131 tok)
- _Tokens:_ prompt 16872 tok; estimated text 517 tok; estimated non-text 16355
  tok; generated 131 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: mlx-community/Qwen3.5-35B-A3B-bf16</summary>

```text
Title:
The Fenchurch Building at night in London

Description:
The Fenchurch Building, located at 20 Fenchurch Street in London, England, is captured at night with its distinctive curved glass façade reflecting the deep blue sky. Street-level details include illuminated shop fronts, parked bicycles, and traffic signage on the wet pavement.

Keywords:
The Fenchurch Building, 20 Fenchurch Street, London, England, UK, Europe, Night, Nightscape, Skyscraper, Architecture, Glass façade, Street, Urban, Cityscape, Street signs, Bicycles, Wet pavement, Blue sky
```

</details>

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-qwen35-27b-4bit"></a>

### ✅ mlx-community/Qwen3.5-27B-4bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Building, Buildings, Cars, City, Commuting
- _Timing:_ Load 2.33s; Gen 72.35s; Total 75.00s
- _Throughput:_ Prompt 251 TPS (16,872 tok); Gen 30.7 TPS (136 tok)
- _Tokens:_ prompt 16872 tok; estimated text 517 tok; estimated non-text 16355
  tok; generated 136 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: mlx-community/Qwen3.5-27B-4bit</summary>

```text
Title:
20 Fenchurch Street, London, England, UK, Europe

Description:
A low-angle night view of the 20 Fenchurch Street skyscraper, known as the Walkie-Talkie, dominating the frame against a deep blue sky. The street below is illuminated by streetlights and vehicle headlights, with green rental bicycles parked on the pavement to the left.

Keywords:
20 Fenchurch Street, Architecture, Blue hour, Cityscape, England, London, Modern, Night, Skyscraper, Street, Urban, United Kingdom, Walkie-Talkie, Windows, Glass, Facade, Nighttime, Europe
```

</details>

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-qwen35-27b-mxfp8"></a>

### ✅ mlx-community/Qwen3.5-27B-mxfp8

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, City, Commuting, Street signs, The
  Fenchurch Building (The Walkie-Talkie)
- _Timing:_ Load 3.52s; Gen 76.36s; Total 80.20s
- _Throughput:_ Prompt 249 TPS (16,872 tok); Gen 18.3 TPS (141 tok)
- _Tokens:_ prompt 16872 tok; estimated text 517 tok; estimated non-text 16355
  tok; generated 141 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: mlx-community/Qwen3.5-27B-mxfp8</summary>

```text
Title:
The Walkie-Talkie Building, London, England, UK

Description:
A low-angle night view captures the distinctive curved glass facade of 20 Fenchurch Street, known as The Walkie-Talkie, towering over a quiet London street. The scene is illuminated by warm streetlights and scattered office windows, with a few parked bicycles and a distant vehicle visible on the wet pavement below.

Keywords:
Architecture, Building, Cityscape, England, Europe, Fenchurch Street, Illuminated, London, Modern, Night, Nightscape, Skyscraper, Street, The Walkie-Talkie, Urban, United Kingdom, Walkie Talkie building
```

</details>

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-qwen36-27b-mxfp8"></a>

### ✅ mlx-community/Qwen3.6-27B-mxfp8

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, City, Commuting, Street signs, The
  Fenchurch Building (The Walkie-Talkie)
- _Timing:_ Load 3.53s; Gen 86.59s; Total 90.46s
- _Throughput:_ Prompt 217 TPS (16,872 tok); Gen 17.7 TPS (144 tok)
- _Tokens:_ prompt 16872 tok; estimated text 517 tok; estimated non-text 16355
  tok; generated 144 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: mlx-community/Qwen3.6-27B-mxfp8</summary>

```text
Title:
20 Fenchurch Street, London, England, UK, Europe

Description:
A low-angle night view of 20 Fenchurch Street, known as the Walkie-Talkie, towering over a quiet urban street in London. The scene is illuminated by streetlights and the building's internal lights, with a row of green rental bicycles parked on the sidewalk in the foreground.

Keywords:
20 Fenchurch Street, Architecture, Bicycle, Cityscape, England, Europe, Fenchurch Street, Illuminated, London, Modern, Night, Nightscape, Skyscraper, Street, The Fenchurch Building, Urban, United Kingdom, Walkie-Talkie
```

</details>

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-molmopoint-8b-fp16"></a>

### ⚠️ mlx-community/MolmoPoint-8B-fp16

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: formally; keywords=20; low-draft-improvement
- _Timing:_ Load 2.76s; Gen 88.14s; Total 91.20s
- _Throughput:_ Prompt 1,330 TPS (3,440 tok); Gen 1.83 TPS (155 tok)
- _Tokens:_ prompt 3440 tok; estimated text 517 tok; estimated non-text 2923
  tok; generated 155 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: mlx-community/MolmoPoint-8B-fp16</summary>

```text
Title:
- 20 Fenchurch Street, London, England, UK, GBR, Europe

Description:
- 20 Fenchurch Street, also known as The Walkie-Talkie, is a distinctive skyscraper in London, England, UK, captured at night. The image shows the building's unique curved glass facade reflecting the dark blue sky, with illuminated windows and streetlights visible in the foreground.

Keywords:
- Architecture, Building, Buildings, Cars, City, Cityscape, Commuting, Fenchurch Street, Illuminated, London, Modern, Night, Nightscape, Skyscraper, Street, Street signs, The Fenchurch Building (The Walkie-Talkie), Urban, Urban landscape, Walkie Talkie building
```

</details>

⚠️ _Quality Warnings:_

- Keyword count violation (20; expected 10-18)
- Low draft improvement

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-2506-bf16"></a>

### ❌ mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: known, formally, 20; thinking trace incomplete
- _Timing:_ Load 3.29s; Gen 108.75s; Total 112.35s
- _Throughput:_ Prompt 1,101 TPS (1,623 tok); Gen 4.69 TPS (500 tok)
- _Tokens:_ prompt 1623 tok; estimated text 517 tok; estimated non-text 1106
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<details>
<summary>Complete generated output: mlx-community/Kimi-VL-A3B-Thinking-2506-bf16</summary>

```text
◁think▷Okay, let's tackle this. First, I need to analyze the image and create a catalog record following the given instructions. The image shows a tall, modern skyscraper at night, which is the Walkie-Talkie building in London, England. The scene is urban, with streetlights, buildings, and a dark blue sky. The building has a distinctive curved top and illuminated windows.

For the title, I should keep it concise and factual. The existing draft title is "The Fenchurch Building (The Walkie-Talkie), London, England, UK, GBR, Europe". That seems accurate.

Next, the description needs to combine the authoritative context with visible details. The building is the main subject, it's nighttime, illuminated, with a dark sky, streetlights, and the surrounding urban environment. So something like "The illuminated Walkie-Talkie building at night, a modern skyscraper in an urban street with streetlights and adjacent buildings under a dark blue sky."

Keywords should include terms from the draft and visible elements. The draft has Architecture, Building, Buildings, Cars, City, Cityscape, Commuting, Fenchurch Street, Illuminated, London, Modern, Night, Nightscape, Skyscraper, Street, Street signs, The Fenchurch Building (The Walkie-Talkie), Urban, Urban landscape, Walkie Talkie building. I can add "Nighttime" and "Skyscraper" if needed, but the draft already has those. Maybe check if "Nightscape" is redundant with Night. Also, "Urban landscape" is covered by Urban. So the keywords would be: Architecture, Building, Buildings, Cars, City, Cityscape, Commuting, Fenchurch Street, Illuminated, London, Modern, Night, Nightscape, Skyscraper, Street, Street signs, The Fenchurch Building (The Walkie-Talkie), Urban, Walkie Talkie building.

Wait, the draft already includes "Walkie Talkie building" and "The Fenchurch Building (The Walkie-Talkie)", so maybe keep those. Also, check if "Night" and "Nightscape" are both needed. Since the image is nighttime, both are appropriate. So the keywords list should be as per the draft but maybe adjust for uniqueness. Let me list them again: Architecture, Building, Buildings, Cars, City, Cityscape, Comm
```

</details>

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)
- Thinking trace incomplete (expected model protocol; opened with ◁think▷)
- Low draft improvement
- Likely capped by max token budget

---

<a id="model-mlx-community-ornith-10-35b-bf16"></a>

### ✅ mlx-community/Ornith-1.0-35B-bf16

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Building, Buildings, Commuting, Modern,
  Nightscape
- _Timing:_ Load 14.81s; Gen 129.56s; Total 144.77s
- _Throughput:_ Prompt 134 TPS (16,872 tok); Gen 57.3 TPS (126 tok)
- _Tokens:_ prompt 16872 tok; estimated text 517 tok; estimated non-text 16355
  tok; generated 126 tok; requested max 500 tok; stop reason completed

<details>
<summary>Complete generated output: mlx-community/Ornith-1.0-35B-bf16</summary>

```text
Title:
- The Fenchurch Street skyscraper at night

Description:
- The Fenchurch Street skyscraper in London, England, rises against a deep blue night sky, its curved glass facade reflecting the surrounding city lights. Street-level details include illuminated shop fronts, parked bicycles, and road signs, with a few cars visible on the wet pavement.

Keywords:
- Fenchurch Street, London, England, UK, Europe, Skyscraper, Night, Cityscape, Urban, Architecture, Glass, Reflections, Street, Cars, Bicycles, Road signs, Illuminated, Blue sky
```

</details>

⚠️ _Quality Warnings:_

- Metadata alignment issue (low_metadata_alignment)

---

<a id="model-mlx-community-qwen2-vl-2b-instruct-4bit"></a>

### ❌ mlx-community/Qwen2-VL-2B-Instruct-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect KV/cache behavior, memory pressure, and long-context
  execution.
- _Key signals:_ At mixed burden (16853 tokens), output became repetitive.;
  hit token cap (500); missing sections: title, description, keywords; missing
  terms: Architecture, Building, Buildings, Cars, City
- _Timing:_ Load 0.55s; Gen 250.01s; Total 250.86s
- _Throughput:_ Prompt 68.2 TPS (16,853 tok); Gen 225 TPS (500 tok)
- _Tokens:_ prompt 16853 tok; estimated text 517 tok; estimated non-text 16336
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<details>
<summary>Complete generated output: mlx-community/Qwen2-VL-2B-Instruct-4bit</summary>

```text
-2026-07-18 22:55:39 BST 51.511300°N 0.083400°W 51.511300°N 0.083400°W 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22
```

</details>

⚠️ _Quality Warnings:_

- ⚠️HARNESS:long_context
- Repetitive output (phrase: "22:55:39 bst 22:55:39 bst...")
- Context ignored (missing: Architecture, Building, Buildings, Cars, City)
- Generation loop (repetitive_tail)
- Metadata alignment issue (low_metadata_alignment)
- Missing sections (title, description, keywords)
- Unverified context copy
- Low draft improvement
- Likely capped by max token budget
- Ignores trusted hints
- suspicious_precision: 4 overly precise numbers
- long_context_repetition(16853tok)

---

<!-- markdownlint-enable MD033 MD034 -->
