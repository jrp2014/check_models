# Model Output Gallery

Generated on: 2026-07-21 08:38:01 BST

Complete per-model evidence artifact with image metadata, the source prompt,
summary tables, diagnostics, and full generated output for every attempted
model.

_Action Snapshot: see [results.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/results.md) for the full summary._

## Run Stamps

- `mlx-vlm`: `0.6.6`
- `mlx`: `0.32.1.dev20260721+30a19f72`
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

- _Title:_ , Lifeboat Station, Southend-on-Sea, England, UK, GBR, Europe
- _Description:_ A lifeboats station with a ferris wheel in the background.
- _Keywords:_ Adobe Stock, Any Vision, Asphalt, British seaside, Cloudy,
  England, Essex, Europe, Fence, Flag, Hotel, Lifeboat Station, Modern
  Architecture, Objects, Overcast, Overcast Sky, Pier, RNLI, Sign,
  Southend-on-Sea, Thames Estuary, UK, Waterfront, amusement ride, antenna,
  architecture, building, city, closed doors, cloudy sky, coast guard station,
  coastal, coastal town, coastguard, curved roof, daytime, emergency,
  emergency building, emergency services, empty, exterior, facade, ferris
  wheel, garage doors, lifeboat, marine rescue, maritime, maritime rescue,
  outdoors, parking lot, rescue, rescue station, roller shutters, safety, sea
  rescue, seaside, seaside resort, shutters, street view, tourism, uk coast,
  urban, vacation destination, windows, yellow lines
- _Date:_ 2026-07-18 17:57:06 BST
- _Time:_ 17:57:06
- _GPS:_ 50.817441°N, 0.134547°W

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
> &#45; Location terms: Adobe Stock, Any Vision, England, Europe, UK
> &#45; Capture date/time: 2026-07-18 17:57:06 BST 17:57:06
> &#45; GPS: 50.817441°N, 0.134547°W
> &#45; Use this factual context where it improves the catalogue record; do not
> claim that contextual facts are visually observable.
>
> &#8203;Draft descriptive metadata:
> &#45; Existing title: Lifeboat Station, Southend-on-Sea, England, UK, GBR,
> Europe
> &#45; Existing description: A lifeboats station with a ferris wheel in the
> background.
> &#45; Existing keywords: Asphalt, British seaside, Cloudy, Essex, Fence, Flag,
> Hotel, Lifeboat Station, Modern Architecture, Objects, Overcast, Overcast
> Sky, Pier, RNLI, Sign, Southend-on-Sea, Thames Estuary, Waterfront,
> amusement ride, antenna
> &#45; Treat this draft as fallible. Retain supported details, correct errors,
> and add important visible information.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

## Model Output and Cost Summary

Every model in this run, with a concise output preview or failure diagnostic beside runtime, memory, and quality signals. Complete output appears once in the expandable per-model evidence below.

<!-- markdownlint-disable MD034 MD049 -->

| Model                                                                                                                   | Result                              | Output preview / diagnostic                                                                                                                                                                                                                                                                          | Gen tok   | Total   | Gen TPS   | Peak GB                                          | Quality signal                                              |
|-------------------------------------------------------------------------------------------------------------------------|-------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|---------|-----------|--------------------------------------------------|-------------------------------------------------------------|
| [`mlx-community/Step-3.7-Flash-oQ2e`](#model-mlx-community-step-37-flash-oq2e)                                          | `not evaluated` / `runtime failure` | [model-config; processor-error] Error: Processor Error - Model preflight failed for mlx-community/Step-3.7-Flash-oQ2e: Loaded processor has no image_processor; expected multimodal processor.                                                                                                       | -         | 7.19s   | -         |                                                  | model-config; processor-error                               |
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `recommended` / `clean`             | Title: Lifeboat Station, Southend-on-Sea, England, UK, GBR, Europe Description: A lifeboats station with a ferris wheel in the background. Keywords: Lifeboat Station, Southend-on-Sea, England, UK, GBR, Europe, Fence, Flag, Hotel, Lifeboat Station, Modern Architecture, Objects,...             | 112       | 1.18s   | 506       | 1.5 GB (1.3% of 108 GB recommended working set)  | context-echo; low-draft-improvement                         |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                            | `avoid` / `model shortcoming`       | Title: Lifeboat Station, Southend-on-Sea, England, UK, GBR, Europe Description: A lifeboats station with a ferris wheel in the background.                                                                                                                                                           | 37        | 1.50s   | 338       | 2.9 GB (2.5% of 108 GB recommended working set)  | low-draft-improvement; missing-sections                     |
| [`mlx-community/LFM2-VL-1.6B-8bit`](#model-mlx-community-lfm2-vl-16b-8bit)                                              | `recommended` / `clean`             | Title: Southend Lifeboat Station Description: The Southend Lifeboat Station is a modern structure with a ferris wheel visible in the background. The station is located in Southend-on-Sea, England, and is surrounded by various objects such as a fence, flag, and hotel. The sky i...             | 139       | 1.69s   | 328       | 3.0 GB (2.58% of 108 GB recommended working set) | low-draft-improvement                                       |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `recommended` / `clean`             | &lt;think&gt; &lt;/think&gt; Title: Lifeboat Station, Southend-on-Sea, England, UK Description: A lifeboat station with large garage doors and a ferris wheel in the background under an overcast sky. The setting includes urban and recreational elements. Keywords: asphalt, lifeboat station,... | 90        | 2.35s   | 275       | 4.1 GB (3.52% of 108 GB recommended working set) | formatting                                                  |
| [`HuggingFaceTB/SmolVLM-Instruct`](#model-huggingfacetb-smolvlm-instruct)                                               | `avoid` / `model shortcoming`       | Title: Lifeboat Station, Southend-on-Sea, England, UK, GBR, Europe Description: A lifeboats station with a ferris wheel in the background.                                                                                                                                                           | 40        | 2.23s   | 127       | 5.7 GB (4.96% of 108 GB recommended working set) | low-draft-improvement; missing-sections                     |
| [`mlx-community/SmolVLM-Instruct-bf16`](#model-mlx-community-smolvlm-instruct-bf16)                                     | `avoid` / `model shortcoming`       | Title: Lifeboat Station, Southend-on-Sea, England, UK, GBR, Europe Description: A lifeboats station with a ferris wheel in the background.                                                                                                                                                           | 40        | 2.20s   | 127       | 5.7 GB (4.97% of 108 GB recommended working set) | low-draft-improvement; missing-sections                     |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `recommended` / `clean`             | Title: Lifeboat Station, Southend-on-Sea, England, UK, GBR, Europe Description: A lifeboats station with a ferris wheel in the background. Keywords: Asphalt, British seaside, Cloudy, Essex, Fence, Flag, Hotel, Lifeboat Station, Modern Architecture, Objects, Overcast, Overcast...              | 106       | 2.59s   | 135       | 5.5 GB (4.75% of 108 GB recommended working set) | context-echo; low-draft-improvement                         |
| [`qnguyen3/nanoLLaVA`](#model-qnguyen3-nanollava)                                                                       | `avoid` / `model shortcoming`       | Title: Lifeboats Station, Southend-on-Sea, England, UK, GBR, Europe Description: A lifeboats station with a ferris wheel in the background. The station is located in Southend-on-Sea, England, UK, and is part of the GBR region in Europe. The station is equipped with modern faci...             | 111       | 2.82s   | 87.4      | 4.9 GB (4.21% of 108 GB recommended working set) | low-draft-improvement; missing-sections                     |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `caveat` / `model shortcoming`      | Title: - Lifeboat Station, Southend-on-Sea, England, UK, GBR, Europe Description: - A lifeboat station with a ferris wheel in the background, located in Southend-on-Sea, England, UK, GBR, Europe. - The station features a modern architectural design with a prominent lifeboat si...             | 270       | 2.73s   | 190       | 4.1 GB (3.56% of 108 GB recommended working set) | unverified-context-copy; metadata-borrowing                 |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `recommended` / `clean`             | Title: Lifeboat Station, Southend-on-Sea, England, UK, GBR, Europe Description: A lifeboats station with a ferris wheel in the background. Keywords: Asphalt, British seaside, Cloudy, Essex, Fence, Flag, Hotel, Lifeboat Station, Modern Architecture, Objects, Overcast, Overcast...              | 284       | 2.85s   | 346       | 2.2 GB (1.88% of 108 GB recommended working set) | low-draft-improvement                                       |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `recommended` / `clean`             | Title: RNLI lifeboat station in Southend-on-Sea, England Description: An RNLI lifeboat station with two large green roller doors stands under an overcast sky, with a large ferris wheel visible in the background. The building features a curved roofline and a white sign with the...             | 127       | 4.69s   | 124       | 17 GB (14.3% of 108 GB recommended working set)  | clean                                                       |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `recommended` / `clean`             | Title: Lifeboat Station at Southend-on-Sea Description: The image shows a functional lifeboat station in Southend-on-Sea, England, with a modern industrial design featuring closed green rolling doors. The station is situated on a waterfront, adjacent to a fenced area and a lar...             | 146       | 3.34s   | 187       | 6.4 GB (5.55% of 108 GB recommended working set) | no flagged signals                                          |
| [`mlx-community/paligemma2-10b-ft-docci-448-6bit`](#model-mlx-community-paligemma2-10b-ft-docci-448-6bit)               | `avoid` / `model shortcoming`       | - Do not copy prompt instructions into the Title, Description, or Keywords fields.                                                                                                                                                                                                                   | 17        | 4.14s   | 31.7      | 12 GB (10.5% of 108 GB recommended working set)  | low-draft-improvement; context-ignored; missing-sections    |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `avoid` / `model shortcoming`       | <\|channel>thought <channel\|>Title: - RNLI lifeboat station in Southend-on-Sea, England Description: A modern RNLI lifeboat station featuring two large green garage doors under a curved roof, set against a ferris wheel and a hotel building under an overcast sky. Keywords: Aspha...           | 103       | 5.90s   | 72.5      | 28 GB (24.6% of 108 GB recommended working set)  | missing-sections; formatting                                |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `avoid` / `model shortcoming`       | <\|channel>thought <channel\|>Title: - RNLI lifeboat station in Southend-on-Sea, Essex Description: An RNLI lifeboat station with two large green roller doors and a curved roof stands under an overcast sky. A large ferris wheel and a multi-story hotel are visible in the backgrou...           | 111       | 6.29s   | 66.0      | 29 GB (25.3% of 108 GB recommended working set)  | low-draft-improvement; missing-sections; formatting         |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `recommended` / `clean`             | Title: Lifeboat Station, Southend-on-Sea, England, UK Description: The Lifeboat Station in Southend-on-Sea, England, UK, is situated on a waterfront with a ferris wheel visible in the background. The station's entrance is marked by a sign and is adjacent to a building with a m...             | 161       | 4.84s   | 56.9      | 9.6 GB (8.31% of 108 GB recommended working set) | low-draft-improvement                                       |
| [`microsoft/Phi-3.5-vision-instruct`](#model-microsoft-phi-35-vision-instruct)                                          | `recommended` / `clean`             | Title: Lifeboat Station, Southend-on-Sea, England, UK Description: The Lifeboat Station in Southend-on-Sea, England, UK, is situated on a waterfront with a ferris wheel visible in the background. The station's entrance is marked by a sign and is adjacent to a building with a m...             | 161       | 4.83s   | 56.7      | 9.6 GB (8.3% of 108 GB recommended working set)  | low-draft-improvement                                       |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `recommended` / `clean`             | Title: Lifeboat Station, Southend-on-Sea, England Description: A lifeboat station with a Ferris wheel and a hotel in the background, under an overcast sky. Keywords: Asphalt, British seaside, Cloudy, Essex, Fence, Flag, Hotel, Lifeboat Station, Modern Architecture, Objects, Ov...             | 107       | 6.73s   | 33.9      | 18 GB (15.7% of 108 GB recommended working set)  | low-draft-improvement                                       |
| [`mlx-community/gemma-3n-E2B-4bit`](#model-mlx-community-gemma-3n-e2b-4bit)                                             | `avoid` / `cutoff degraded`         | - T-18-07-18 17:57:06 17:57:06 - 17:57:06 - 17:57:06 - 17:57:06 - 17:57:06 - 17:57:06 - 17:57:06 - 17:57:06 - 17:57:06 - 17:57:06 - 17:57:06 - 17:57:06 - 17:57:06 - 17:57:06 - 17:57:06 - 17:57:06 - 17:57:06 - 17:57:06 - 17:57:06 - 17:57:06 - 17:57:06 - 17:57:06 - 17:57:06 - 17...             | 500       | 6.49s   | 125       | 6.1 GB (5.26% of 108 GB recommended working set) | repetitive; unverified-context-copy; low-draft-improvement  |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `recommended` / `clean`             | **Title:** Lifeboat Station and Ferris Wheel, Southend-on-Sea, UK **Description:** A lifeboat station with a green garage door and curved roof, positioned near a pier, with a large amusement Ferris wheel and a multi-storey hotel visible in the background under an overcast sky....             | 148       | 6.50s   | 63.6      | 12 GB (10.5% of 108 GB recommended working set)  | clean                                                       |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `recommended` / `clean`             | **Title:** Lifeboat Station with Southend Pier Amusement Ride **Description:** The image shows the RNLI lifeboat station in Southend-on-Sea, England, featuring a modern single-storey building with a green garage door and a blue sign reading "Lifeboats." In the background, a la...             | 164       | 6.52s   | 66.9      | 12 GB (10.1% of 108 GB recommended working set)  | clean                                                       |
| [`mlx-community/paligemma2-10b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-10b-ft-docci-448-bf16)               | `avoid` / `model shortcoming`       | - Do not copy prompt instructions into the Title, Description, or Keywords fields.                                                                                                                                                                                                                   | 17        | 7.69s   | 5.56      | 27 GB (23.3% of 108 GB recommended working set)  | low-draft-improvement; context-ignored; missing-sections    |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `recommended` / `clean`             | Title: - Lifeboat Station, Southend-on-Sea Description: - The lifeboat station in Southend-on-Sea, England, features a prominent RNLI sign and is situated near a ferris wheel and a multi-story hotel. Keywords: - Lifeboat Station, RNLI, Southend-on-Sea, Ferris Wheel, Hotel, Clo...             | 118       | 6.98s   | 39.6      | 15 GB (13.4% of 108 GB recommended working set)  | low-draft-improvement                                       |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `recommended` / `clean`             | Title: Lifeboat Station in Southend-on-Sea, England, UK Description: A lifeboat station featuring two large green roller shutters and a curved roof under an overcast sky. A large white Ferris wheel and a multi-storey hotel are visible in the background. Keywords: Lifeboat Stat...             | 106       | 8.84s   | 27.3      | 20 GB (17.6% of 108 GB recommended working set)  | clean                                                       |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `caveat` / `clean`                  | Title: Lifeboat Station, Southend-on-Sea, England, UK, GBR, Europe. Description: A lifeboat station with a large green shuttered door and a sign reading "Lifeboats" above it, is situated in a coastal town with a ferris wheel visible in the background. The sky is overcast and t...             | 135       | 8.19s   | 31.7      | 19 GB (16.6% of 108 GB recommended working set)  | low-draft-improvement; formatting                           |
| [`mlx-community/InternVL3-14B-8bit`](#model-mlx-community-internvl3-14b-8bit)                                           | `recommended` / `clean`             | Title: Lifeboat Station, Southend-on-Sea, UK Description: The image shows a lifeboat station with a large green garage door, situated near a pier with a Ferris wheel in the background. The sky is overcast, and a hotel building is visible to the right. Keywords: Asphalt, Britis...             | 119       | 8.23s   | 32.1      | 19 GB (16% of 108 GB recommended working set)    | low-draft-improvement                                       |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                            | `caveat` / `model shortcoming`      | Title: Lifeboat Station, Southend-on-Sea, England, UK, GBR, Europe Description: A lifeboat station with a ferris wheel in the background. Keywords: Asphalt, British seaside, Cloudy, Essex, Fence, Flag, Hotel, Lifeboat Station, Modern Architecture, Objects, Overcast, Overcast S...             | 259       | 8.05s   | 62.2      | 9.7 GB (8.41% of 108 GB recommended working set) | instruction-echo; low-draft-improvement; reasoning-leak     |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `recommended` / `clean`             | Title: Lifeboat Station, Southend-on-Sea, England, UK Description: The Southend-on-Sea lifeboat station is a pale green building with a curved roof, situated on a paved waterfront with a large Ferris wheel visible in the background under a cloudy sky. A flag bearing the RNLI e...             | 159       | 9.47s   | 31.1      | 18 GB (15.7% of 108 GB recommended working set)  | no flagged signals                                          |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `recommended` / `clean`             | Title: Lifeboat Station, Southend-on-Sea, England Description: A RNLI lifeboat station with a large Ferris wheel visible in the background. The scene is set on a cloudy day at a British seaside location. Keywords: Lifeboat Station, RNLI, Southend-on-Sea, England, UK, Ferris wh...             | 109       | 8.77s   | 80.6      | 8.4 GB (7.29% of 108 GB recommended working set) | low-draft-improvement                                       |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `avoid` / `model shortcoming`       | ## Image Description: The image shows a lifeboat station in Southend-on-Sea, England, on an overcast day. The station is a low, white building with two large, closed green roller shutter doors. Above the doors, a sign reads "Lifeboats" in blue lettering on a white background,...              | 316       | 9.90s   | 48.8      | 17 GB (14.9% of 108 GB recommended working set)  | missing-sections                                            |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `recommended` / `clean`             | Title: Lifeboat Station, Southend-on-Sea Description: A lifeboat station with a green roller shutter and a sign reading "Lifeboats" in Southend-on-Sea, England, with a Ferris wheel and a multi-storey hotel visible in the background. Keywords: Asphalt, British seaside, Cloudy,...              | 125       | 10.27s  | 30.0      | 22 GB (18.7% of 108 GB recommended working set)  | context-echo; low-draft-improvement                         |
| [`mlx-community/GLM-4.6V-Flash-6bit`](#model-mlx-community-glm-46v-flash-6bit)                                          | `caveat` / `clean`                  | <\|begin_of_box\|>Title: Lifeboat Station, Southend-on-Sea, England, UK Description: A RNLI lifeboat station with a large Ferris wheel visible in the background, set against an overcast sky. The station features a curved roofline, large green roller doors, and a blue "Lifeboats"...           | 121       | 9.62s   | 57.9      | 11 GB (9.57% of 108 GB recommended working set)  | special token wrapper                                       |
| [`mlx-community/pixtral-12b-bf16`](#model-mlx-community-pixtral-12b-bf16)                                               | `recommended` / `clean`             | Title: - Lifeboat Station, Southend-on-Sea Description: - The lifeboat station in Southend-on-Sea, England, features a prominent RNLI sign and is situated near an amusement park with a Ferris wheel and a multi-story hotel. Keywords: - Lifeboat Station, RNLI, Southend-on-Sea, F...             | 123       | 10.86s  | 20.2      | 27 GB (23.7% of 108 GB recommended working set)  | low-draft-improvement                                       |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)            | `recommended` / `clean`             | **Title:** Lifeboat Station, Southend-on-Sea, England, UK, GBR, Europe **Description:** A modern lifeboat station with a large garage door and a sign reading "Lifeboats" is situated in a parking lot, with a Ferris wheel visible in the background. The building's architecture is...             | 139       | 10.29s  | 21.8      | 15 GB (13.1% of 108 GB recommended working set)  | no flagged signals                                          |
| [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](#model-jqlive-kimi-vl-a3b-thinking-2506-6bit)                                 | `avoid` / `cutoff degraded`         | ◁think▷Got it, let's analyze the image step by step. First, the title needs to be 5-10 words, concrete and factual. The authoritative context mentions Southend-on-Sea, England, UK, so that's a key location. The main subject is a lifeboat station with a ferris wheel in the back...             | 500       | 10.79s  | 71.3      | 18 GB (15.5% of 108 GB recommended working set)  | low-draft-improvement; cutoff; thinking-incomplete          |
| [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](#model-mlx-community-kimi-vl-a3b-thinking-8bit)                             | `avoid` / `cutoff degraded`         | ◁think▷Okay, let's tackle this metadata cataloguing task. First, I need to analyze the image and extract the metadata based on what's visible and the provided authoritative context. The user wants me to focus on visible details only, using the context where necessary but not i...             | 500       | 11.47s  | 66.1      | 22 GB (19% of 108 GB recommended working set)    | low-draft-improvement; cutoff; thinking-incomplete          |
| [`mlx-community/gemma-3-27b-it-qat-8bit`](#model-mlx-community-gemma-3-27b-it-qat-8bit)                                 | `recommended` / `clean`             | Title: Lifeboat Station, Southend-on-Sea, England Description: The Southend-on-Sea lifeboat station, a pale green building with a curved roof, stands on the waterfront with a large Ferris wheel visible in the background under an overcast sky. A flag flies above the station, an...             | 149       | 13.93s  | 17.7      | 32 GB (27.4% of 108 GB recommended working set)  | no flagged signals                                          |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `avoid` / `cutoff degraded`         | Let's tackle this step by step. First, the title needs to be 5-10 words, concrete and factual. The authoritative context has Location terms including Southend-on-Sea, England, UK, GBR, Europe. The existing title is "Lifeboat Station, Southend-on-Sea, England, UK, GBR, Europe"...              | 500       | 16.57s  | 54.4      | 60 GB (52.1% of 108 GB recommended working set)  | low-draft-improvement; cutoff; missing-sections             |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                             | `avoid` / `cutoff degraded`         | - Do not output the title text after the label. - Do not output the description text after the label. - Do not output the description text after the label. - Do not output the description text after the label. - Do not output the description text after the label. - Do not outp...             | 500       | 13.87s  | 47.5      | 4.6 GB (3.96% of 108 GB recommended working set) | repetitive; low-draft-improvement; cutoff                   |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `avoid` / `cutoff degraded`         | Here are my reasoning steps: We need to produce a catalog record with three sections: Title, Description, Keywords. The title: 5-10 words, concrete and factual; can use authoritative context to supply identity and location. The existing title is "Lifeboat Station, Southend-on-...             | 500       | 16.11s  | 42.4      | 15 GB (12.7% of 108 GB recommended working set)  | low-draft-improvement; cutoff; reasoning-leak               |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `caveat` / `clean`                  | <\|begin_of_box\|>Title: Lifeboat Station, Southend-on-Sea, England, UK Description: A lifeboat station building with a large green roller door, a blue "Lifeboats" sign, and RNLI insignia, set against a cloudy sky with a Ferris wheel and a multi-story hotel visible in the backgr...           | 126       | 20.16s  | 43.4      | 78 GB (67.2% of 108 GB recommended working set)  | low-draft-improvement                                       |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `caveat` / `context budget`         | <\|im_start\|> addCriterion("image:caption")                                                                                                                                                                                                                                                         | 9         | 19.67s  | 62.5      | 14 GB (11.7% of 108 GB recommended working set)  | harness:long-context; low-draft-improvement; context-budget |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `avoid` / `context budget`          | The image shows a building with a sign that reads "Lifeboats" and a Ferris wheel in the background. The building has a green door and a blue sign with the word "Lifeboats" on it. The Ferris wheel is located to the left of the building, and there is a building with a sign that...              | 90        | 19.44s  | 93.9      | 8.6 GB (7.47% of 108 GB recommended working set) | low-draft-improvement; context-budget; missing-sections     |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `avoid` / `cutoff degraded`         | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -...             | 500       | 20.48s  | 93.3      | 8.6 GB (7.46% of 108 GB recommended working set) | harness:long-context; repetitive; low-draft-improvement     |
| [`meta-llama/Llama-3.2-11B-Vision-Instruct`](#model-meta-llama-llama-32-11b-vision-instruct)                            | `avoid` / `model shortcoming`       | Lifeboat Station, Southend-on-Sea, England, UK, GBR, Europe A lifeboat station with a modern architectural design, featuring a large garage door and a sign that reads "Lifeboats" in white letters on a blue background, is situated in a parking lot with yellow lines and a brick...              | 90        | 22.51s  | 5.1       | 25 GB (21.9% of 108 GB recommended working set)  | low-draft-improvement; missing-sections                     |
| [`mlx-community/Molmo-7B-D-0924-bf16`](#model-mlx-community-molmo-7b-d-0924-bf16)                                       | `caveat` / `harness`                | No generated text captured.                                                                                                                                                                                                                                                                          | 1         | 22.44s  | 45,282    | 48 GB (41.4% of 108 GB recommended working set)  | harness:prompt-template                                     |
| [`mlx-community/Molmo-7B-D-0924-8bit`](#model-mlx-community-molmo-7b-d-0924-8bit)                                       | `recommended` / `clean`             | Do not remove existing terms unless they are clearly contradicted by the image. Title: Lifeboat Station, Southend-on-Sea, England, UK, GBR, Europe Description: A lifeboats station with a large ferris wheel in the background. The station features two green garage doors with "Li...             | 144       | 24.29s  | 51.8      | 41 GB (35.2% of 108 GB recommended working set)  | no flagged signals                                          |
| [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16)                             | `avoid` / `cutoff degraded`         | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -...             | 500       | 23.98s  | 91.4      | 8.6 GB (7.47% of 108 GB recommended working set) | harness:long-context; repetitive; low-draft-improvement     |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                           | `recommended` / `clean`             | Title: Lifeboat Station, Southend-on-Sea, England, UK Description: A lifeboat station with a large white building featuring a curved roof and green garage doors. A ferris wheel is visible in the background, and a flag with a red cross on a white background is flying from a fla...             | 138       | 29.02s  | 5.96      | 27 GB (23.1% of 108 GB recommended working set)  | low-draft-improvement                                       |
| [`mlx-community/paligemma2-3b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-3b-ft-docci-448-bf16)                 | `avoid` / `cutoff degraded`         | - Do not copy-prompt instructions into the title, description, or keywords fields. - Do not infer identity, location, event, brand, or style. - Do not infer the text of the image. - Do not use the text of the image to create a new image. - Do not use the text of the image to c...             | 500       | 29.04s  | 19.0      | 11 GB (9.7% of 108 GB recommended working set)   | repetitive; low-draft-improvement; cutoff                   |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `recommended` / `clean`             | Title: - RNLI Lifeboat Station with Ferris Wheel and Hotel Description: - A white RNLI lifeboat station with green roller doors stands in the foreground under an overcast sky, with a large white Ferris wheel visible to the left and a multi-storey hotel building to the right. K...             | 120       | 59.93s  | 63.0      | 76 GB (65.8% of 108 GB recommended working set)  | clean                                                       |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `recommended` / `clean`             | Title: RNLI Lifeboat Station Southend-on-Sea Description: A lifeboat station with closed green roller shutters and an RNLI sign stands in Southend-on-Sea, England, with a large Ferris wheel and hotel buildings visible in the background under an overcast sky. Keywords: RNLI, Li...             | 116       | 56.79s  | 108       | 26 GB (22.6% of 108 GB recommended working set)  | clean                                                       |
| [`mlx-community/Qwen3.5-35B-A3B-6bit`](#model-mlx-community-qwen35-35b-a3b-6bit)                                        | `recommended` / `clean`             | Title: RNLI Lifeboat Station Southend-on-Sea England Description: A white RNLI lifeboat station with large green roller doors stands in the foreground under an overcast sky, with a large Ferris wheel visible to the left and a multi-storey hotel building in the background. Keyw...             | 118       | 58.52s  | 92.5      | 35 GB (30.1% of 108 GB recommended working set)  | clean                                                       |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `recommended` / `context budget`    | Title: RNLI Lifeboat Station, Southend-on-Sea, UK Description: A modern RNLI lifeboat station with green roller shutter doors and a curved white roof, situated in a paved forecourt under an overcast sky, with a large Ferris wheel and a multi-storey hotel visible in the backgro...             | 176       | 57.98s  | 93.0      | 11 GB (9.94% of 108 GB recommended working set)  | unverified-context-copy; metadata-borrowing; context-budget |
| [`mlx-community/Qwen3.5-35B-A3B-bf16`](#model-mlx-community-qwen35-35b-a3b-bf16)                                        | `recommended` / `clean`             | Title: RNLI Lifeboat Station Southend-on-Sea England Description: A white RNLI lifeboat station with closed green roller shutters stands in the foreground under an overcast sky, with a large Ferris wheel and the Palace Hotel visible in the background. Keywords: RNLI, Lifeboats...             | 114       | 65.81s  | 66.5      | 76 GB (65.8% of 108 GB recommended working set)  | clean                                                       |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                             | `avoid` / `cutoff degraded`         | -001.jpg - 17:57:06 BST - 50.81744°N, 0.13454°W - 17:57:06 BST: The image is taken at 17:57:06 BST. The time is in the 17:57:06 format, which is the time in the BST (British Standard Time) zone. The time is in the 17:57:06 format, which is the time in the BST (British Standard...             | 500       | 68.01s  | 222       | 5.1 GB (4.41% of 108 GB recommended working set) | harness:long-context; repetitive; unverified-context-copy   |
| [`mlx-community/gemma-4-31b-bf16`](#model-mlx-community-gemma-4-31b-bf16)                                               | `avoid` / `cutoff degraded`         | Image: - A lifeboat station with a ferris wheel in the background. - The image shows a lifeboat station with a ferris wheel in the background. The lifeboat station is a modern building with a curved roof and large green doors. The ferris wheel is a large, colorful structure wi...             | 500       | 78.77s  | 7.22      | 65 GB (56% of 108 GB recommended working set)    | repetitive; low-draft-improvement; cutoff                   |
| [`mlx-community/Qwen3.5-27B-mxfp8`](#model-mlx-community-qwen35-27b-mxfp8)                                              | `recommended` / `clean`             | Title: RNLI Lifeboat Station, Southend-on-Sea, England Description: A modern white lifeboat station building with two large green roller doors stands under an overcast sky, featuring a blue "Lifeboats" sign with the RNLI logo. In the background, a large white Ferris wheel is v...             | 158       | 83.42s  | 17.5      | 38 GB (33.3% of 108 GB recommended working set)  | clean                                                       |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `recommended` / `clean`             | Title: RNLI Lifeboat Station, Southend-on-Sea, England Description: A white RNLI lifeboat station building with green roller doors stands in the foreground, featuring a blue sign and a flag on the roof. In the background, a large Ferris wheel and a multi-story hotel building a...             | 127       | 83.54s  | 18.1      | 38 GB (33.3% of 108 GB recommended working set)  | no flagged signals                                          |
| [`mlx-community/Qwen3.5-27B-4bit`](#model-mlx-community-qwen35-27b-4bit)                                                | `recommended` / `clean`             | Title: RNLI Lifeboat Station, Southend-on-Sea, Essex, England Description: A modern white RNLI lifeboat station with two large green roller shutter doors stands in the foreground, featuring a blue sign with the organisation's logo. In the background, a large white Ferris wheel...             | 139       | 82.88s  | 31.0      | 26 GB (22.4% of 108 GB recommended working set)  | clean                                                       |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `avoid` / `cutoff degraded`         | ◁think▷Okay, let's tackle this. First, I need to analyze the image and create a catalog record with the specified sections: Title, Description, and Keywords. The user wants it in British English, so I should use terms like "Lifeboats" instead of "Lifeboat" and "Pier" instead o...             | 500       | 112.66s | 4.67      | 40 GB (34.7% of 108 GB recommended working set)  | repetitive; low-draft-improvement; cutoff                   |
<!-- markdownlint-enable MD034 MD049 -->

## Quick Navigation

- _Successful outputs:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16),
  [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit),
  [`mlx-community/LFM2-VL-1.6B-8bit`](#model-mlx-community-lfm2-vl-16b-8bit),
  [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit),
  [`HuggingFaceTB/SmolVLM-Instruct`](#model-huggingfacetb-smolvlm-instruct),
  [`mlx-community/SmolVLM-Instruct-bf16`](#model-mlx-community-smolvlm-instruct-bf16),
  [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx),
  [`qnguyen3/nanoLLaVA`](#model-qnguyen3-nanollava), +53 more
- _Flagged outputs:_ [`mlx-community/Step-3.7-Flash-oQ2e`](#model-mlx-community-step-37-flash-oq2e),
  [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16),
  [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit),
  [`mlx-community/LFM2-VL-1.6B-8bit`](#model-mlx-community-lfm2-vl-16b-8bit),
  [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit),
  [`HuggingFaceTB/SmolVLM-Instruct`](#model-huggingfacetb-smolvlm-instruct),
  [`mlx-community/SmolVLM-Instruct-bf16`](#model-mlx-community-smolvlm-instruct-bf16),
  [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx),
  +44 more
- _Failed outputs:_ [`mlx-community/Step-3.7-Flash-oQ2e`](#model-mlx-community-step-37-flash-oq2e)

## Model Gallery

Full generated output by model:

<!-- markdownlint-disable MD033 MD034 -->

<a id="model-mlx-community-step-37-flash-oq2e"></a>

### ❔ mlx-community/Step-3.7-Flash-oQ2e

- _Verdict:_ runtime_failure | user=not_evaluated
- _Status:_ Failed (Processor Error)
- _Maintainer:_ issue_ready | owner=model-config | confidence=medium
- _Next action:_ File the retained reproduction evidence with the suspected
  owner.
- _Error summary:_ see error details below
- _Why:_ execution failure
- _Failure context:_ type `ValueError`; phase `processor_load`; code
  `MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR`; package `model-config`

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

### ✅ LiquidAI/LFM2.5-VL-450M-MLX-bf16

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: LiquidAI/LFM2.5-VL-450M-MLX-bf16</summary>

```text
Title: Lifeboat Station, Southend-on-Sea, England, UK, GBR, Europe

Description: A lifeboats station with a ferris wheel in the background.

Keywords: Lifeboat Station, Southend-on-Sea, England, UK, GBR, Europe, Fence, Flag, Hotel, Lifeboat Station, Modern Architecture, Objects, Overcast, Overcast Sky, Pier, RNLI, Sign, Southend-on-Sea, Thames Estuary, Waterfront, amusement ride, antenna
```

</details>

⚠️ _Quality Warnings:_

- Keyword count violation (22; expected 10-18)
- Context echo (47% overlap)
- Low draft improvement

---

<a id="model-mlx-community-nanollava-15-4bit"></a>

### ❌ mlx-community/nanoLLaVA-1.5-4bit

- _Verdict:_ model_shortcoming | user=avoid
- _Why:_ missing required sections | irrelevant output smell | missing
  sections: keywords
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/nanoLLaVA-1.5-4bit</summary>

```text
Title: Lifeboat Station, Southend-on-Sea, England, UK, GBR, Europe
Description: A lifeboats station with a ferris wheel in the background.
```

</details>

⚠️ _Quality Warnings:_

- Metadata alignment issue (low_metadata_alignment)
- Missing sections (keywords)
- Low draft improvement

---

<a id="model-mlx-community-lfm2-vl-16b-8bit"></a>

### ✅ mlx-community/LFM2-VL-1.6B-8bit

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/LFM2-VL-1.6B-8bit</summary>

```text
Title: Southend Lifeboat Station

Description: The Southend Lifeboat Station is a modern structure with a ferris wheel visible in the background. The station is located in Southend-on-Sea, England, and is surrounded by various objects such as a fence, flag, and hotel. The sky is overcast, and the image was captured on July 18, 2026, at 5:57 PM BST.

Keywords: Southend, lifeboat station, ferris wheel, England, UK, modern architecture, objects, overcast sky, Southend-on-Sea, Thames Estuary, waterfront, amusement ride, antenna
```

</details>

⚠️ _Quality Warnings:_

- Metadata alignment issue (low_metadata_alignment)
- Title length violation (3 words; expected 5-10)
- Description sentence violation (3; expected 1-2)
- Low draft improvement

---

<a id="model-mlx-community-minicpm-v-46-8bit"></a>

### ✅ mlx-community/MiniCPM-V-4.6-8bit

- _Verdict:_ clean | user=recommended
- _Why:_ Empty thinking wrapper present
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/MiniCPM-V-4.6-8bit</summary>

```text
<think>

</think>

Title:
Lifeboat Station, Southend-on-Sea, England, UK

Description: A lifeboat station with large garage doors and a ferris wheel in the background under an overcast sky. The setting includes urban and recreational elements.

Keywords: asphalt, lifeboat station, ferris wheel, overcast, UK, England, urban, recreational, gate, flag, modern architecture, parking, brick, antenna
```

</details>

⚠️ _Quality Warnings:_

- Empty thinking wrapper present

---

<a id="model-huggingfacetb-smolvlm-instruct"></a>

### ❌ HuggingFaceTB/SmolVLM-Instruct

- _Verdict:_ model_shortcoming | user=avoid
- _Why:_ missing required sections | irrelevant output smell | missing
  sections: keywords
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: HuggingFaceTB/SmolVLM-Instruct</summary>

```text
 Title: Lifeboat Station, Southend-on-Sea, England, UK, GBR, Europe
Description: A lifeboats station with a ferris wheel in the background.
```

</details>

⚠️ _Quality Warnings:_

- Metadata alignment issue (low_metadata_alignment)
- Missing sections (keywords)
- Low draft improvement

---

<a id="model-mlx-community-smolvlm-instruct-bf16"></a>

### ❌ mlx-community/SmolVLM-Instruct-bf16

- _Verdict:_ model_shortcoming | user=avoid
- _Why:_ missing required sections | irrelevant output smell | missing
  sections: keywords
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/SmolVLM-Instruct-bf16</summary>

```text
 Title: Lifeboat Station, Southend-on-Sea, England, UK, GBR, Europe
Description: A lifeboats station with a ferris wheel in the background.
```

</details>

⚠️ _Quality Warnings:_

- Metadata alignment issue (low_metadata_alignment)
- Missing sections (keywords)
- Low draft improvement

---

<a id="model-mlx-community-smolvlm2-22b-instruct-mlx"></a>

### ✅ mlx-community/SmolVLM2-2.2B-Instruct-mlx

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/SmolVLM2-2.2B-Instruct-mlx</summary>

```text
 Title: Lifeboat Station, Southend-on-Sea, England, UK, GBR, Europe

Description: A lifeboats station with a ferris wheel in the background.

Keywords: Asphalt, British seaside, Cloudy, Essex, Fence, Flag, Hotel, Lifeboat Station, Modern Architecture, Objects, Overcast, Overcast Sky, Pier, RNLI, Sign, Southend-on-Sea, Thames Estuary, Waterfront, amusement ride, antenna
```

</details>

⚠️ _Quality Warnings:_

- Keyword count violation (20; expected 10-18)
- Context echo (77% overlap)
- Low draft improvement

---

<a id="model-qnguyen3-nanollava"></a>

### ❌ qnguyen3/nanoLLaVA

- _Verdict:_ model_shortcoming | user=avoid
- _Why:_ missing required sections | irrelevant output smell | missing
  sections: keywords
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: qnguyen3/nanoLLaVA</summary>

```text
Title: Lifeboats Station, Southend-on-Sea, England, UK, GBR, Europe
Description: A lifeboats station with a ferris wheel in the background. The station is located in Southend-on-Sea, England, UK, and is part of the GBR region in Europe. The station is equipped with modern facilities and is part of a larger complex that includes a ferris wheel and a waterfront. The station is situated in a picturesque location with a backdrop of the Thames Estuary and the South Coast of England.
```

</details>

⚠️ _Quality Warnings:_

- Missing sections (keywords)
- Description sentence violation (4; expected 1-2)
- Low draft improvement

---

<a id="model-mlx-community-lfm25-vl-16b-bf16"></a>

### ⚠️ mlx-community/LFM2.5-VL-1.6B-bf16

- _Verdict:_ model_shortcoming | user=caveat
- _Why:_ irrelevant output smell
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/LFM2.5-VL-1.6B-bf16</summary>

```text
Title:
- Lifeboat Station, Southend-on-Sea, England, UK, GBR, Europe

Description:
- A lifeboat station with a ferris wheel in the background, located in Southend-on-Sea, England, UK, GBR, Europe.
- The station features a modern architectural design with a prominent lifeboat sign and a flag flying atop the building.
- The surrounding area includes a ferris wheel, suggesting the presence of an amusement park or fairground nearby.
- The sky is overcast, indicating cloudy weather conditions.
- The station is situated near a waterfront, with the Thames Estuary visible in the background.
- The asphalt ground and parking area are visible in the foreground, with yellow lines marking the parking spaces.
- The image was captured on 2026-07-18 at 17:57:06 BST.

Keywords:
- Lifeboat Station, Southend-on-Sea, England, UK, GBR, Europe, Modern Architecture, Ferris Wheel, Amusement Park, Overcast, Cloudy, Waterfront, Asphalt, British Seaside, Pier, Flag, Hotel, Sign, Thames Estuary, Parking, Overcast Sky, Station, RNLI, Objects, Southend-on-Sea, European City
```

</details>

⚠️ _Quality Warnings:_

- Description sentence violation (7; expected 1-2)
- Keyword count violation (26; expected 10-18)
- Unverified context copy

---

<a id="model-mlx-community-fastvlm-05b-bf16"></a>

### ✅ mlx-community/FastVLM-0.5B-bf16

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/FastVLM-0.5B-bf16</summary>

```text
Title: Lifeboat Station, Southend-on-Sea, England, UK, GBR, Europe
Description: A lifeboats station with a ferris wheel in the background.
Keywords: Asphalt, British seaside, Cloudy, Essex, Fence, Flag, Hotel, Lifeboat Station, Modern Architecture, Objects, Overcast, Overcast Sky, Pier, RNLI, Sign, Southend-on-Sea, Thames Estuary, Waterfront, amusement ride, antenna
Draft metadata:
- Existing title: Lifeboat Station, Southend-on-Sea, England, UK, GBR, Europe
- Existing description: A lifeboats station with a ferris wheel in the background.
- Existing keywords: Asphalt, British seaside, Cloudy, Essex, Fence, Flag, Hotel, Lifeboat Station, Modern Architecture, Objects, Overcast, Overcast Sky, Pier, RNLI, Sign, Southend-on-Sea, Thames Estuary, Waterfront, amusement ride, antenna
- The image shows a lifeboats station with a ferris wheel in the background, suggesting a coastal location. The station is part of the Southend-on-Sea, England, UK, GBR, Europe, and is associated with the RNLI (Royal National Lifeboat Institution). The weather is cloudy, and the image is taken during the day. The station appears to be a modern structure with a contemporary design.
```

</details>

⚠️ _Quality Warnings:_

- Keyword count violation (56; expected 10-18)
- Keyword duplication (43% duplicated terms)
- Low draft improvement

---

<a id="model-mlx-community-gemma-4-26b-a4b-it-4bit"></a>

### ✅ mlx-community/gemma-4-26b-a4b-it-4bit

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/gemma-4-26b-a4b-it-4bit</summary>

```text
Title:
RNLI lifeboat station in Southend-on-Sea, England

Description:
An RNLI lifeboat station with two large green roller doors stands under an overcast sky, with a large ferris wheel visible in the background. The building features a curved roofline and a white sign with the word 'Lifeboats' and the RNLI logo.

Keywords:
Asphalt, British seaside, cloudy, Essex, ferris wheel, flag, hotel, lifeboat station, modern architecture, overcast, pier, RNLI, sign, Southend-on-Sea, Thames Estuary, waterfront, amusement ride, antenna
```

</details>

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-ministral-3-3b-instruct-2512-4bit"></a>

### ✅ mlx-community/Ministral-3-3B-Instruct-2512-4bit

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/Ministral-3-3B-Instruct-2512-4bit</summary>

```text
Title:
Lifeboat Station at Southend-on-Sea

Description:
The image shows a functional lifeboat station in Southend-on-Sea, England, with a modern industrial design featuring closed green rolling doors. The station is situated on a waterfront, adjacent to a fenced area and a large amusement ride, including a Ferris wheel, under an overcast sky.

Keywords:
Lifeboat Station, Southend-on-Sea, Essex, RNLI, green doors, industrial architecture, fenced perimeter, Ferris wheel, amusement ride, antenna, overcast sky, waterfront, British seaside, signage, British flag, modern building, rolling doors, outdoor parking, cloudy conditions
```

</details>

⚠️ _Quality Warnings:_

- Keyword count violation (19; expected 10-18)

---

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-6bit"></a>

### ❌ mlx-community/paligemma2-10b-ft-docci-448-6bit

- _Verdict:_ model_shortcoming | user=avoid
- _Why:_ missing required sections | irrelevant output smell | missing
  sections: title, description, keywords
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/paligemma2-10b-ft-docci-448-6bit</summary>

```text
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
```

</details>

⚠️ _Quality Warnings:_

- No overlap with supplied context indicators
- Metadata alignment issue (low_metadata_alignment)
- Missing sections (title, description, keywords)
- Low draft improvement

---

<a id="model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8"></a>

### ❌ mlx-community/diffusiongemma-26B-A4B-it-mxfp8

- _Verdict:_ model_shortcoming | user=avoid
- _Why:_ missing required sections | irrelevant output smell | missing
  sections: title | Unknown tags: <channel|>
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/diffusiongemma-26B-A4B-it-mxfp8</summary>

```text
<|channel>thought
<channel|>Title:
- RNLI lifeboat station in Southend-on-Sea, England

Description:
A modern RNLI lifeboat station featuring two large green garage doors under a curved roof, set against a ferris wheel and a hotel building under an overcast sky.

Keywords:
Asphalt, British seaside, Cloudy, Essex, Fence, Flag, Hotel, Lifeboat Station, Modern Architecture, RNLI, Sign, Southend-on-Sea, Waterfront, amusement ride, antenna
```

</details>

⚠️ _Quality Warnings:_

- Metadata alignment issue (low_metadata_alignment)
- Missing sections (title)
- Unknown tags: <channel\|>

---

<a id="model-mlx-community-diffusiongemma-26b-a4b-it-8bit"></a>

### ❌ mlx-community/diffusiongemma-26B-A4B-it-8bit

- _Verdict:_ model_shortcoming | user=avoid
- _Why:_ missing required sections | irrelevant output smell | missing
  sections: title | Unknown tags: <channel|>
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/diffusiongemma-26B-A4B-it-8bit</summary>

```text
<|channel>thought
<channel|>Title:
- RNLI lifeboat station in Southend-on-Sea, Essex

Description:
An RNLI lifeboat station with two large green roller doors and a curved roof stands under an overcast sky. A large ferris wheel and a multi-story hotel are visible in the background.

Keywords:
RNLI, lifeboat station, Southend-on-Sea, Essex, England UK, ferris wheel, architecture, overcast, cloudy, asphalt, coastal, flag, hotel, antenna, waterfront, seaside, urban
```

</details>

⚠️ _Quality Warnings:_

- Metadata alignment issue (low_metadata_alignment)
- Missing sections (title)
- Low draft improvement
- Unknown tags: <channel\|>

---

<a id="model-mlx-community-phi-35-vision-instruct-bf16"></a>

### ✅ mlx-community/Phi-3.5-vision-instruct-bf16

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/Phi-3.5-vision-instruct-bf16</summary>

```text
Title: Lifeboat Station, Southend-on-Sea, England, UK

Description: The Lifeboat Station in Southend-on-Sea, England, UK, is situated on a waterfront with a ferris wheel visible in the background. The station's entrance is marked by a sign and is adjacent to a building with a modern architectural style.

Keywords: Lifeboat Station, Southend-on-Sea, England, UK, GBR, Europe, Waterfront, Modern Architecture, Ferris Wheel, British Seaside, Cloudy, Overcast, Thames Estuary, RNLI, Sign, Hotel, Antenna, Objects, Essex, Pier
```

</details>

⚠️ _Quality Warnings:_

- Keyword count violation (20; expected 10-18)
- Low draft improvement

---

<a id="model-microsoft-phi-35-vision-instruct"></a>

### ✅ microsoft/Phi-3.5-vision-instruct

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: microsoft/Phi-3.5-vision-instruct</summary>

```text
Title: Lifeboat Station, Southend-on-Sea, England, UK

Description: The Lifeboat Station in Southend-on-Sea, England, UK, is situated on a waterfront with a ferris wheel visible in the background. The station's entrance is marked by a sign and is adjacent to a building with a modern architectural style.

Keywords: Lifeboat Station, Southend-on-Sea, England, UK, GBR, Europe, Waterfront, Modern Architecture, Ferris Wheel, British Seaside, Cloudy, Overcast, Thames Estuary, RNLI, Sign, Hotel, Antenna, Objects, Essex, Pier
```

</details>

⚠️ _Quality Warnings:_

- Keyword count violation (20; expected 10-18)
- Low draft improvement

---

<a id="model-mlx-community-internvl3-8b-bf16"></a>

### ✅ mlx-community/InternVL3-8B-bf16

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/InternVL3-8B-bf16</summary>

```text
Title:
Lifeboat Station, Southend-on-Sea, England

Description:
A lifeboat station with a Ferris wheel and a hotel in the background, under an overcast sky.

Keywords:
Asphalt, British seaside, Cloudy, Essex, Fence, Flag, Hotel, Lifeboat Station, Modern Architecture, Objects, Overcast, Pier, RNLI, Sign, Southend-on-Sea, Waterfront, amusement ride, antenna, building, garage, gate, railing, satellite dish, structure, tower, window
```

</details>

⚠️ _Quality Warnings:_

- Keyword count violation (26; expected 10-18)
- Low draft improvement

---

<a id="model-mlx-community-gemma-3n-e2b-4bit"></a>

### ❌ mlx-community/gemma-3n-E2B-4bit

- _Verdict:_ cutoff_degraded | user=avoid
- _Why:_ text repetition | missing required sections | token cap truncation |
  hit token cap (500)
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/gemma-3n-E2B-4bit</summary>

```text

- T-18-07-18 17:57:06 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57
```

</details>

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "17:57:06 - 17:57:06 -...")
- Excessive bullet points (44)
- No overlap with supplied context indicators
- Text sanity issue (gibberish(token_noise))
- Generation loop (repetitive_tail)
- Metadata alignment issue (low_metadata_alignment)
- Missing sections (title, description, keywords)
- Unverified context copy
- Low draft improvement
- Likely capped by max token budget

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-nvfp4"></a>

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-nvfp4

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/Ministral-3-14B-Instruct-2512-nvfp4</summary>

```text
**Title:**
Lifeboat Station and Ferris Wheel, Southend-on-Sea, UK

**Description:**
A lifeboat station with a green garage door and curved roof, positioned near a pier, with a large amusement Ferris wheel and a multi-storey hotel visible in the background under an overcast sky. The station displays a Union Jack flag and an antenna on its roof.

**Keywords:**
Lifeboat Station, RNLI, Southend-on-Sea, Pier, Ferris Wheel, Pier Hotel, Overcast Sky, Green Garage Door, Union Jack Flag, Antenna, Modern Station, Pier Fencing, Thames Estuary, UK Seaside, Pier Amusement, Pierfront, Cloudy Weather
```

</details>

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-mxfp4"></a>

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-mxfp4

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/Ministral-3-14B-Instruct-2512-mxfp4</summary>

```text
**Title:**
Lifeboat Station with Southend Pier Amusement Ride

**Description:**
The image shows the RNLI lifeboat station in Southend-on-Sea, England, featuring a modern single-storey building with a green garage door and a blue sign reading "Lifeboats." In the background, a large Ferris wheel and a multi-storey hotel are visible under an overcast sky, with the setting suggesting a coastal amusement pier.

**Keywords:**
Lifeboat Station, RNLI, Southend-on-Sea, UK, coastal architecture, green garage door, blue signage, Ferris wheel, amusement pier, overcast sky, modern building, multi-storey hotel, asphalt pavement, flagpole, antenna, barbed wire fence, Thames Estuary, British seaside
```

</details>

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-bf16"></a>

### ❌ mlx-community/paligemma2-10b-ft-docci-448-bf16

- _Verdict:_ model_shortcoming | user=avoid
- _Why:_ missing required sections | irrelevant output smell | missing
  sections: title, description, keywords
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/paligemma2-10b-ft-docci-448-bf16</summary>

```text
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
```

</details>

⚠️ _Quality Warnings:_

- No overlap with supplied context indicators
- Metadata alignment issue (low_metadata_alignment)
- Missing sections (title, description, keywords)
- Low draft improvement

---

<a id="model-mlx-community-pixtral-12b-8bit"></a>

### ✅ mlx-community/pixtral-12b-8bit

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/pixtral-12b-8bit</summary>

```text
Title:
- Lifeboat Station, Southend-on-Sea

Description:
- The lifeboat station in Southend-on-Sea, England, features a prominent RNLI sign and is situated near a ferris wheel and a multi-story hotel.

Keywords:
- Lifeboat Station, RNLI, Southend-on-Sea, Ferris Wheel, Hotel, Cloudy Sky, Overcast, British Architecture, Signage, Flag, Asphalt, Fence, Modern Building, Waterfront, Pier, Amusement Ride, Antenna, Essex, Thames Estuary
```

</details>

⚠️ _Quality Warnings:_

- Keyword count violation (19; expected 10-18)
- Low draft improvement

---

<a id="model-mlx-community-gemma-4-31b-it-4bit"></a>

### ✅ mlx-community/gemma-4-31b-it-4bit

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/gemma-4-31b-it-4bit</summary>

```text
Title: Lifeboat Station in Southend-on-Sea, England, UK

Description: A lifeboat station featuring two large green roller shutters and a curved roof under an overcast sky. A large white Ferris wheel and a multi-storey hotel are visible in the background.

Keywords: Lifeboat Station, Southend-on-Sea, England, UK, Europe, Ferris wheel, amusement ride, green shutters, overcast sky, hotel, asphalt, antenna, flag, waterfront, British seaside, architecture, RNLI, sign
```

</details>

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-idefics3-8b-llama3-bf16"></a>

### ⚠️ mlx-community/Idefics3-8B-Llama3-bf16

- _Verdict:_ clean | user=caveat
- _Why:_ special token wrapper | Unknown tags: <end_of_utterance>
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/Idefics3-8B-Llama3-bf16</summary>

```text
 Title: Lifeboat Station, Southend-on-Sea, England, UK, GBR, Europe.

Description: A lifeboat station with a large green shuttered door and a sign reading "Lifeboats" above it, is situated in a coastal town with a ferris wheel visible in the background. The sky is overcast and the station is surrounded by asphalt.

Keywords: Asphalt, British seaside, Cloudy, Essex, Fence, Flag, Hotel, Lifeboat Station, Modern Architecture, Objects, Overcast, Overcast Sky, Pier, RNLI, Sign, Southend-on-Sea, Thames Estuary, Waterfront, amusement ride, antenna.<end_of_utterance>
```

</details>

⚠️ _Quality Warnings:_

- Keyword count violation (20; expected 10-18)
- Low draft improvement
- Unknown tags: <end_of_utterance>

---

<a id="model-mlx-community-internvl3-14b-8bit"></a>

### ✅ mlx-community/InternVL3-14B-8bit

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/InternVL3-14B-8bit</summary>

```text
Title:
Lifeboat Station, Southend-on-Sea, UK

Description:
The image shows a lifeboat station with a large green garage door, situated near a pier with a Ferris wheel in the background. The sky is overcast, and a hotel building is visible to the right.

Keywords:
Asphalt, British seaside, Cloudy, Essex, Fence, Flag, Hotel, Lifeboat Station, Modern Architecture, Objects, Overcast, Overcast Sky, Pier, RNLI, Sign, Southend-on-Sea, Thames Estuary, Waterfront, amusement ride, antenna
```

</details>

⚠️ _Quality Warnings:_

- Keyword count violation (20; expected 10-18)
- Low draft improvement

---

<a id="model-mlx-community-llava-v16-mistral-7b-8bit"></a>

### ⚠️ mlx-community/llava-v1.6-mistral-7b-8bit

- _Verdict:_ model_shortcoming | user=caveat
- _Why:_ irrelevant output smell
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/llava-v1.6-mistral-7b-8bit</summary>

```text
Title: Lifeboat Station, Southend-on-Sea, England, UK, GBR, Europe

Description: A lifeboat station with a ferris wheel in the background.

Keywords: Asphalt, British seaside, Cloudy, Essex, Fence, Flag, Hotel, Lifeboat Station, Modern Architecture, Objects, Overcast, Overcast Sky, Pier, RNLI, Sign, Southend-on-Sea, Thames Estuary, Waterfront, amusement ride, antenna

Rules:
- Distinguish supplied authoritative facts from visible details; do not present contextual facts as though they were read from the image.
- Reuse draft metadata when it is consistent with the image; authoritative context does not require separate visual proof.
- If metadata and image disagree, follow the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless supplied as authoritative context or visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.
```

</details>

⚠️ _Quality Warnings:_

- Keyword count violation (40; expected 10-18)
- Reasoning leak (do not output reasoning)
- Instruction echo
- Low draft improvement

---

<a id="model-mlx-community-gemma-3-27b-it-qat-4bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-4bit

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/gemma-3-27b-it-qat-4bit</summary>

```text
Title:
Lifeboat Station, Southend-on-Sea, England, UK

Description:
The Southend-on-Sea lifeboat station is a pale green building with a curved roof, situated on a paved waterfront with a large Ferris wheel visible in the background under a cloudy sky. A flag bearing the RNLI emblem flies above the station entrance, and a brick outbuilding is visible to the left.

Keywords:
Asphalt, British seaside, Cloudy, Essex, Fence, Flag, Lifeboat Station, Modern Architecture, Objects, Overcast, Overcast Sky, Pier, RNLI, Sign, Southend-on-Sea, Thames Estuary, Waterfront, amusement ride, antenna, brickwork, curved roof, grey sky, parking, pale green, paved area
```

</details>

⚠️ _Quality Warnings:_

- Keyword count violation (25; expected 10-18)

---

<a id="model-mlx-community-glm-46v-flash-mxfp4"></a>

### ✅ mlx-community/GLM-4.6V-Flash-mxfp4

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/GLM-4.6V-Flash-mxfp4</summary>

```text
Title:
Lifeboat Station, Southend-on-Sea, England

Description:
A RNLI lifeboat station with a large Ferris wheel visible in the background. The scene is set on a cloudy day at a British seaside location.

Keywords:
Lifeboat Station, RNLI, Southend-on-Sea, England, UK, Ferris wheel, amusement ride, cloudy weather, waterfront, modern architecture, green garage doors, British seaside, Essex, overcast sky, antenna, flag, sign, parking lot, fence, hotel, waterfront
```

</details>

⚠️ _Quality Warnings:_

- Keyword count violation (21; expected 10-18)
- Low draft improvement

---

<a id="model-mlx-community-gemma-3n-e4b-it-bf16"></a>

### ❌ mlx-community/gemma-3n-E4B-it-bf16

- _Verdict:_ model_shortcoming | user=avoid
- _Why:_ missing required sections | irrelevant output smell | missing
  sections: title, description, keywords
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/gemma-3n-E4B-it-bf16</summary>

```text
## Image Description:

The image shows a lifeboat station in Southend-on-Sea, England, on an overcast day. The station is a low, white building with two large, closed green roller shutter doors. Above the doors, a sign reads "Lifeboats" in blue lettering on a white background, flanked by the RNLI (Royal National Lifeboat Institution) logo. A small, arched window is situated above the sign.

To the left of the lifeboat station, a brick building with a metal gate is visible. In the background, a large Ferris wheel stands prominently, suggesting a seaside location. The sky is uniformly grey, indicating cloudy weather. A British flag is flying on a pole attached to the lifeboat station.

The foreground shows a paved area with yellow parking lines. The overall scene conveys a sense of a functional, yet somewhat weathered, maritime facility in a coastal town.

## Visual Details:

* **Main Subject:** A lifeboat station.
* **Setting:** A coastal area, likely Southend-on-Sea, England.
* **Weather:** Overcast, grey sky.
* **Architecture:** A low, white building with green roller shutter doors and an arched window.
* **Signage:** "Lifeboats" with the RNLI logo.
* **Background Elements:** A large Ferris wheel, a brick building, a British flag.
* **Foreground:** Paved area with yellow parking lines.
* **Overall Impression:** A functional maritime facility in a seaside town on a cloudy day.
```

</details>

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)

---

<a id="model-mlx-community-devstral-small-2-24b-instruct-2512-5bit"></a>

### ✅ mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit</summary>

```text
Title:
Lifeboat Station, Southend-on-Sea

Description:
A lifeboat station with a green roller shutter and a sign reading "Lifeboats" in Southend-on-Sea, England, with a Ferris wheel and a multi-storey hotel visible in the background.

Keywords:
Asphalt, British seaside, Cloudy, Essex, Fence, Flag, Hotel, Lifeboat Station, Modern Architecture, Objects, Overcast, Overcast Sky, Pier, RNLI, Sign, Southend-on-Sea, Thames Estuary, Waterfront, amusement ride, antenna
```

</details>

⚠️ _Quality Warnings:_

- Keyword count violation (20; expected 10-18)
- Context echo (42% overlap)
- Low draft improvement

---

<a id="model-mlx-community-glm-46v-flash-6bit"></a>

### ⚠️ mlx-community/GLM-4.6V-Flash-6bit

- _Verdict:_ clean | user=caveat
- _Why:_ special token wrapper
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/GLM-4.6V-Flash-6bit</summary>

```text
<|begin_of_box|>Title:
Lifeboat Station, Southend-on-Sea, England, UK

Description:
A RNLI lifeboat station with a large Ferris wheel visible in the background, set against an overcast sky. The station features a curved roofline, large green roller doors, and a blue "Lifeboats" sign with the RNLI emblem.

Keywords:
Asphalt, British seaside, Cloudy, Essex, Flag, Ferris wheel, Lifeboat Station, Modern Architecture, Overcast, Overcast Sky, RNLI, Sign, Southend-on-Sea, Waterfront, amusement ride<|end_of_box|>
```

</details>

⚠️ _Quality Warnings:_

- Metadata alignment issue (low_metadata_alignment)

---

<a id="model-mlx-community-pixtral-12b-bf16"></a>

### ✅ mlx-community/pixtral-12b-bf16

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/pixtral-12b-bf16</summary>

```text
Title:
- Lifeboat Station, Southend-on-Sea

Description:
- The lifeboat station in Southend-on-Sea, England, features a prominent RNLI sign and is situated near an amusement park with a Ferris wheel and a multi-story hotel.

Keywords:
- Lifeboat Station, RNLI, Southend-on-Sea, Ferris Wheel, Hotel, Cloudy Sky, Overcast, Flag, Sign, Modern Architecture, Asphalt, Pier, Fence, Amusement Ride, Antenna, British Seaside, Essex, Thames Estuary, Europe, UK
```

</details>

⚠️ _Quality Warnings:_

- Keyword count violation (20; expected 10-18)
- Low draft improvement

---

<a id="model-mlx-community-llama-32-11b-vision-instruct-8bit"></a>

### ✅ mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/Llama-3.2-11B-Vision-Instruct-8bit</summary>

```text
**Title:** Lifeboat Station, Southend-on-Sea, England, UK, GBR, Europe

**Description:** A modern lifeboat station with a large garage door and a sign reading "Lifeboats" is situated in a parking lot, with a Ferris wheel visible in the background. The building's architecture is characterized by its clean lines and minimal ornamentation, with a focus on functionality.

**Keywords:** Asphalt, British seaside, Cloudy, Fence, Flag, Hotel, Lifeboat Station, Modern Architecture, Objects, Overcast, Overcast Sky, Pier, RNLI, Sign, Southend-on-Sea, Thames Estuary, Waterfront, amusement ride, antenna
```

</details>

⚠️ _Quality Warnings:_

- Keyword count violation (19; expected 10-18)

---

<a id="model-jqlive-kimi-vl-a3b-thinking-2506-6bit"></a>

### ❌ jqlive/Kimi-VL-A3B-Thinking-2506-6bit

- _Verdict:_ cutoff_degraded | user=avoid
- _Why:_ thinking trace | reasoning budget exhausted | missing required
  sections | hit token cap (500)
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: jqlive/Kimi-VL-A3B-Thinking-2506-6bit</summary>

```text
◁think▷Got it, let's analyze the image step by step. First, the title needs to be 5-10 words, concrete and factual. The authoritative context mentions Southend-on-Sea, England, UK, so that's a key location. The main subject is a lifeboat station with a ferris wheel in the background. So the title could be "Lifeboat Station, Southend-on-Sea, England, UK" or similar.

For the description, we need 1-2 factual sentences combining context and visible details. The lifeboat station building, the ferris wheel (amusement ride), the cloudy sky, the parking lot, the buildings in the background. So something like "A lifeboat station building with green doors and a 'Lifeboats' sign, a Ferris wheel, and buildings under an overcast sky in Southend-on-Sea, England."

Keywords: Need 10-18 unique terms. From the image: Lifeboats, Southend-on-Sea, England, UK, Ferris wheel, cloudy, overcast, building, parking lot, flag, buildings, modern architecture, asphalt, fence, flagpole, satellite dish, antenna, Thames Estuary (from authoritative context). Let's list them: Lifeboats, Southend-on-Sea, England, UK, Ferris wheel, cloudy, overcast, building, parking lot, flag, buildings, modern architecture, asphalt, fence, flagpole, satellite dish, antenna, Thames Estuary, RNLI (from existing keywords but maybe not visible? Wait, the existing keywords had RNLI but maybe not visible. Wait, the image has "Lifeboats" sign, which is RNLI. So include that. Also, the flag is the English flag. So keywords: Lifeboats, Southend-on-Sea, England, UK, Ferris wheel, cloudy, overcast, building, parking lot, flag, buildings, modern architecture, asphalt, fence, flagpole, satellite dish, antenna, Thames Estuary, RNLI.

Wait, check the existing draft keywords: Asphalt, British seaside, Cloudy, Essex, Fence, Flag, Hotel, Lifeboat Station, Modern Architecture, Objects, Overcast, Overcast Sky, Pier, RNLI, Sign, Southend-on-Sea, Thames Estuary, Waterfront, amusement ride, antenna. But the user says to treat the draft as fallible
```

</details>

⚠️ _Quality Warnings:_

- Metadata alignment issue (low_metadata_alignment)
- Missing sections (title, description)
- Keyword count violation (79; expected 10-18)
- Keyword duplication (52% duplicated terms)
- Thinking trace incomplete (expected model protocol; opened with ◁think▷)
- Low draft improvement
- Likely capped by max token budget

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-8bit"></a>

### ❌ mlx-community/Kimi-VL-A3B-Thinking-8bit

- _Verdict:_ cutoff_degraded | user=avoid
- _Why:_ thinking trace | reasoning budget exhausted | missing required
  sections | hit token cap (500)
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/Kimi-VL-A3B-Thinking-8bit</summary>

```text
◁think▷Okay, let's tackle this metadata cataloguing task. First, I need to analyze the image and extract the metadata based on what's visible and the provided authoritative context. The user wants me to focus on visible details only, using the context where necessary but not inferring anything not clearly shown.

Starting with the title. The existing draft title is "Lifeboat Station, Southend-on-Sea, England, UK, GBR, Europe". The image shows a lifeboat station building with a Ferris wheel in the background. The location is Southend-on-Sea, which matches the draft. The Ferris wheel might be part of an amusement area, but the image doesn't clearly show if it's a Ferris wheel or another type of ride. Since the draft includes "amusement ride" in keywords, maybe it's safe to include that. The title should be concise, so "Lifeboat Station, Southend-on-Sea, England, UK" seems accurate. The GBR and Europe tags are part of the authoritative context, so include them as they are part of the context, not inferred.

Next, the description. The existing draft mentions a lifeboats station with a Ferris wheel in the background. The image confirms the lifeboat station, and the Ferris wheel is visible. The flag on the building is Union Jack, which is part of the authoritative context. The building has green garage doors and a sign. The sky is overcast, so "overcast" should be included. The Ferris wheel is a distinctive feature, so "Ferris wheel" is visible. The parking area with yellow lines is also visible. The description should combine these elements: "A lifeboat station with green garage doors and a Union Jack flag, situated in Southend-on-Sea, England, under an overcast sky, with a Ferris wheel visible in the background."

For keywords, the existing ones include Lifeboat Station, Southend-on-Sea, Thames Estuary, Waterfront, etc. The Ferris wheel is visible, so add "Ferris wheel". The flag is Union Jack. The building has green garage doors, so "green garage doors". The parking area with yellow lines is visible. The overcast sky is there. The existing keywords include "amusement ride", so maybe "amusement ride" is acceptable even if it's a Ferris wheel. Also, "modern architecture" from the draft. The building's
```

</details>

⚠️ _Quality Warnings:_

- Output degeneration (incomplete_sentence: ends with 's')
- Generation loop (degeneration)
- Missing sections (title, description, keywords)
- Thinking trace incomplete (expected model protocol; opened with ◁think▷)
- Low draft improvement
- Likely capped by max token budget

---

<a id="model-mlx-community-gemma-3-27b-it-qat-8bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-8bit

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/gemma-3-27b-it-qat-8bit</summary>

```text
Title:
Lifeboat Station, Southend-on-Sea, England

Description:
The Southend-on-Sea lifeboat station, a pale green building with a curved roof, stands on the waterfront with a large Ferris wheel visible in the background under an overcast sky. A flag flies above the station, and a brick building is to the left, with a multi-storey hotel visible to the right.

Keywords:
Asphalt, British seaside, Cloudy, Essex, Fence, Flag, Ferris wheel, Hotel, Lifeboat Station, Modern Architecture, Overcast, Overcast Sky, Pier, RNLI, Sign, Southend-on-Sea, Thames Estuary, Waterfront, brickwork, curved roof, pale green
```

</details>

⚠️ _Quality Warnings:_

- Keyword count violation (21; expected 10-18)

---

<a id="model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16"></a>

### ❌ mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

- _Verdict:_ cutoff_degraded | user=avoid
- _Why:_ missing required sections | token cap truncation | hit token cap
  (500) | missing sections: title, description, keywords
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16</summary>

```text
Let's tackle this step by step. First, the title needs to be 5-10 words, concrete and factual. The authoritative context has Location terms including Southend-on-Sea, England, UK, GBR, Europe. The existing title is "Lifeboat Station, Southend-on-Sea, England, UK, GBR, Europe" but that's a bit long. Wait, the rules say to use authoritative context where it improves the record. The main subject is the lifeboat station. Let's make it concise: "Lifeboat Station, Southend-on-Sea, England, UK" – that's 6 words, good.

Next, the description. The existing one is "A lifeboats station with a ferris wheel in the background." But wait, the authoritative context has RNLI (Royal National Lifeboat Institution), which is a key part of the lifeboat station. Also, the image has a ferris wheel (amusement ride), cloudy sky, overcast, the station's building (white, arched roof, green doors?), the RNLI sign? Wait, the draft keywords have RNLI. Let's combine: "A Royal National Lifeboat Institution (RNLI) lifeboat station with a ferris wheel in the background, set against a cloudy sky." Wait, but does the image have the RNLI sign? The existing keywords include RNLI, so maybe it's visible. Also, the setting is a seaside area (Southend-on-Sea is a seaside town). Let's make it more accurate: "A Royal National Lifeboat Institution (RNLI) lifeboat station with a ferris wheel in the background, located in a seaside area under a cloudy sky." Wait, but the rules say to use authoritative context where it improves the record. The authoritative context has RNLI, so including that is good. Also, the existing description mentions the ferris wheel, which is in the image. Let's check the existing description: "A lifeboats station with a ferris wheel in the background." But "lifeboats station" should be "lifeboat station" (singular, as per RNLI's name). So correct that. Also, add the RNLI since it's authoritative context. So description: "A Royal National Lifeboat Institution (RN
```

</details>

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)
- Low draft improvement
- Likely capped by max token budget

---

<a id="model-mlx-community-paligemma2-3b-pt-896-4bit"></a>

### ❌ mlx-community/paligemma2-3b-pt-896-4bit

- _Verdict:_ cutoff_degraded | user=avoid
- _Why:_ text repetition | missing required sections | token cap truncation |
  hit token cap (500)
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/paligemma2-3b-pt-896-4bit</summary>

```text
- Do not output the title text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after
```

</details>

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "- do not output...")
- Excessive bullet points (42)
- No overlap with supplied context indicators
- Generation loop (repetitive_tail)
- Metadata alignment issue (low_metadata_alignment)
- Missing sections (title, description, keywords)
- Low draft improvement
- Likely capped by max token budget

---

<a id="model-mlx-community-apriel-15-15b-thinker-6bit-mlx"></a>

### ❌ mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX

- _Verdict:_ cutoff_degraded | user=avoid
- _Why:_ missing required sections | token cap truncation | hit token cap
  (500) | missing sections: description, keywords
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX</summary>

```text
Here are my reasoning steps:
We need to produce a catalog record with three sections: Title, Description, Keywords. The title: 5-10 words, concrete and factual; can use authoritative context to supply identity and location. The existing title is "Lifeboat Station, Southend-on-Sea, England, UK, GBR, Europe". That's too many words (maybe 9 words? Let's count: Lifeboat (1) Station, (2) Southend-on-Sea, (3) England, (4) UK, (5) GBR, (6) Europe (7). That's 7 words, but includes many location terms. The instruction says 5-10 words, concrete and factual; we can keep "Lifeboat Station, Southend-on-Sea, England". That's 5 words? Let's count: Lifeboat (1) Station, (2) Southend-on-Sea, (3) England (4). Actually "Lifeboat Station, Southend-on-Sea, England" is 4 words? Let's count: "Lifeboat" (1), "Station," (2), "Southend-on-Sea," (3), "England" (4). That's 4 words, but we need 5-10 words. Could add "UK". So "Lifeboat Station, Southend-on-Sea, England, UK". That's 5 words? Count: Lifeboat (1), Station, (2), Southend-on-Sea, (3), England, (4), UK (5). Yes 5 words. That's fine.

But we need to ensure we don't repeat or paraphrase instructions. Title should be just the title text after the label. So we output:

Title:
Lifeboat Station, Southend-on-Sea, England, UK

Now description: 1-2 factual sentences combining supplied authoritative context with main visible subject, setting, lighting, action, distinctive visible details. Must not claim contextual facts as visually observable. So we can say: "The image shows the RNLI lifeboat station in Southend-on-Sea, England, with a large green garage door and a sign reading 'Lifeboats'. In the background, a Ferris wheel rises above a cloudy sky, and a Union Jack flag flies on the building." That
```

</details>

⚠️ _Quality Warnings:_

- Metadata alignment issue (low_metadata_alignment)
- Missing sections (description, keywords)
- Title length violation (83 words; expected 5-10)
- Reasoning leak (here are my reasoning steps)
- Low draft improvement
- Likely capped by max token budget

---

<a id="model-mlx-community-glm-46v-nvfp4"></a>

### ⚠️ mlx-community/GLM-4.6V-nvfp4

- _Verdict:_ clean | user=caveat
- _Why:_ special token wrapper
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/GLM-4.6V-nvfp4</summary>

```text
<|begin_of_box|>Title:
Lifeboat Station, Southend-on-Sea, England, UK

Description:
A lifeboat station building with a large green roller door, a blue "Lifeboats" sign, and RNLI insignia, set against a cloudy sky with a Ferris wheel and a multi-story hotel visible in the background.

Keywords:
Asphalt, British seaside, Cloudy, Essex, Fence, Flag, Hotel, Lifeboat Station, Modern Architecture, Objects, Overcast, Overcast Sky, Pier, RNLI, Sign, Southend-on-Sea, Thames Estuary, Waterfront, amusement ride, antenna<|end_of_box|>
```

</details>

⚠️ _Quality Warnings:_

- Keyword count violation (20; expected 10-18)
- Low draft improvement

---

<a id="model-mlx-community-x-reasoner-7b-8bit"></a>

### ⚠️ mlx-community/X-Reasoner-7B-8bit

- _Verdict:_ context_budget | user=caveat
- _Why:_ Output appears truncated to about 9 tokens. | At mixed burden (16933
  tokens), output stayed unusually short (9 tokens; ratio 0.1%; weak text
  signal truncated). | special token wrapper | harness contract
- _Maintainer:_ issue_ready | owner=mlx | confidence=high
- _Next action:_ File the retained reproduction evidence with the suspected
  owner.

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
- No overlap with supplied context indicators
- Metadata alignment issue (low_metadata_alignment)
- Low draft improvement
- output:truncated(9tok)
- long_context_low_ratio(0.1%;16933->9;truncated)

---

<a id="model-mlx-community-qwen3-vl-2b-thinking-bf16"></a>

### ❌ mlx-community/Qwen3-VL-2B-Thinking-bf16

- _Verdict:_ context_budget | user=avoid
- _Why:_ missing required sections | missing sections: title, description,
  keywords
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/Qwen3-VL-2B-Thinking-bf16</summary>

```text
The image shows a building with a sign that reads "Lifeboats" and a Ferris wheel in the background. The building has a green door and a blue sign with the word "Lifeboats" on it. The Ferris wheel is located to the left of the building, and there is a building with a sign that reads "Palace" in the background. The sky is overcast, and the ground is paved with yellow lines.
```

</details>

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context_budget
- Missing sections (title, description, keywords)
- Low draft improvement

---

<a id="model-qwen-qwen3-vl-2b-instruct"></a>

### ❌ Qwen/Qwen3-VL-2B-Instruct

- _Verdict:_ cutoff_degraded | user=avoid
- _Why:_ At mixed burden (16922 tokens), output became repetitive. | harness
  contract | text repetition | missing required sections
- _Maintainer:_ issue_ready | owner=mlx | confidence=high
- _Next action:_ File the retained reproduction evidence with the suspected
  owner.

<details>
<summary>Complete generated output: Qwen/Qwen3-VL-2B-Instruct</summary>

```text
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
```

</details>

⚠️ _Quality Warnings:_

- ⚠️HARNESS:long_context
- Repetitive output (-)
- Output degeneration (character_loop: ' -' repeated)
- Text sanity issue (gibberish(token_noise))
- Generation loop (repetitive_tail)
- Metadata alignment issue (low_metadata_alignment)
- Missing sections (title, description, keywords)
- Low draft improvement
- Likely capped by max token budget
- long_context_repetition(16922tok)

---

<a id="model-meta-llama-llama-32-11b-vision-instruct"></a>

### ❌ meta-llama/Llama-3.2-11B-Vision-Instruct

- _Verdict:_ model_shortcoming | user=avoid
- _Why:_ missing required sections | irrelevant output smell | missing
  sections: title, description, keywords
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: meta-llama/Llama-3.2-11B-Vision-Instruct</summary>

```text
Lifeboat Station, Southend-on-Sea, England, UK, GBR, Europe

A lifeboat station with a modern architectural design, featuring a large garage door and a sign that reads "Lifeboats" in white letters on a blue background, is situated in a parking lot with yellow lines and a brick wall to the left. In the background, a Ferris wheel and a hotel are visible, set against an overcast sky.
```

</details>

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)
- Low draft improvement

---

<a id="model-mlx-community-molmo-7b-d-0924-bf16"></a>

### ⚠️ mlx-community/Molmo-7B-D-0924-bf16

- _Verdict:_ harness | user=caveat
- _Why:_ harness contract
- _Maintainer:_ issue_ready | owner=model-config | confidence=high
- _Next action:_ File the retained reproduction evidence with the suspected
  owner.

<details>
<summary>Complete generated output: mlx-community/Molmo-7B-D-0924-bf16</summary>

```text

```

</details>

⚠️ _Quality Warnings:_

- ⚠️HARNESS:prompt_template
- output:empty(1tok)

---

<a id="model-mlx-community-molmo-7b-d-0924-8bit"></a>

### ✅ mlx-community/Molmo-7B-D-0924-8bit

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/Molmo-7B-D-0924-8bit</summary>

```text
 Do not remove existing terms unless they are clearly contradicted by the image.
Title: Lifeboat Station, Southend-on-Sea, England, UK, GBR, Europe

Description: A lifeboats station with a large ferris wheel in the background. The station features two green garage doors with "Lifeboats" written above them. A British flag is visible on the roof. The sky is overcast, and the scene includes a hotel and other buildings in the distance.

Keywords: Asphalt, British seaside, Cloudy, Essex, Fence, Flag, Hotel, Lifeboat Station, Modern Architecture, Objects, Overcast, Overcast Sky, Pier, Thames Estuary, Waterfront, amusement ride, antenna
```

</details>

⚠️ _Quality Warnings:_

- Description sentence violation (4; expected 1-2)

---

<a id="model-mlx-community-qwen3-vl-2b-instruct-bf16"></a>

### ❌ mlx-community/Qwen3-VL-2B-Instruct-bf16

- _Verdict:_ cutoff_degraded | user=avoid
- _Why:_ At mixed burden (16922 tokens), output became repetitive. | harness
  contract | text repetition | missing required sections
- _Maintainer:_ issue_ready | owner=mlx | confidence=high
- _Next action:_ File the retained reproduction evidence with the suspected
  owner.

<details>
<summary>Complete generated output: mlx-community/Qwen3-VL-2B-Instruct-bf16</summary>

```text
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
```

</details>

⚠️ _Quality Warnings:_

- ⚠️HARNESS:long_context
- Repetitive output (-)
- Output degeneration (character_loop: ' -' repeated)
- Text sanity issue (gibberish(token_noise))
- Generation loop (repetitive_tail)
- Metadata alignment issue (low_metadata_alignment)
- Missing sections (title, description, keywords)
- Low draft improvement
- Likely capped by max token budget
- long_context_repetition(16922tok)

---

<a id="model-mlx-community-molmopoint-8b-fp16"></a>

### ✅ mlx-community/MolmoPoint-8B-fp16

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/MolmoPoint-8B-fp16</summary>

```text
Title:
Lifeboat Station, Southend-on-Sea, England, UK

Description:
A lifeboat station with a large white building featuring a curved roof and green garage doors. A ferris wheel is visible in the background, and a flag with a red cross on a white background is flying from a flagpole. The scene is set on a cloudy day with an overcast sky.

Keywords:
Asphalt, British seaside, Cloudy, Essex, Fence, Flag, Hotel, Lifeboat Station, Modern Architecture, Objects, Overcast, Overcast Sky, Pier, RNLI, Sign, Southend-on-Sea, Thames Estuary, Waterfront, amusement ride, antenna
```

</details>

⚠️ _Quality Warnings:_

- Description sentence violation (3; expected 1-2)
- Keyword count violation (20; expected 10-18)
- Low draft improvement

---

<a id="model-mlx-community-paligemma2-3b-ft-docci-448-bf16"></a>

### ❌ mlx-community/paligemma2-3b-ft-docci-448-bf16

- _Verdict:_ cutoff_degraded | user=avoid
- _Why:_ text repetition | missing required sections | token cap truncation |
  hit token cap (500)
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/paligemma2-3b-ft-docci-448-bf16</summary>

```text
- Do not copy-prompt instructions into the title, description, or keywords fields.
- Do not infer identity, location, event, brand, or style.
- Do not infer the text of the image.
- Do not use the text of the image to create a new image.
- Do not use the text of the image to create a new description.
- Do not use the text of the image to create a new brand.
- Do not use the text of the image to create a new logo.
- Do not use the text of the image to create a new brand.
- Do not use the text of the image to create a new logo.
- Do not use the text of the image to create a new brand.
- Do not use the text of the image to create a new logo.
- Do not use the text of the image to create a new brand.
- Do not use the text of the image to create a new logo.
- Do not use the text of the image to create a new brand.
- Do not use the text of the image to create a new logo.
- Do not use the text of the image to create a new brand.
- Do not use the text of the image to create a new logo.
- Do not use the text of the image to create a new brand.
- Do not use the text of the image to create a new brand.
- Do not use the text of the image to create a new logo.
- Do not use the text of the image to create a new brand.
- Do not use the text of the image to create a new logo.
- Do not use the text of the image to create a new brand.
- Do not use the text of the image to create a new logo.
- Do not use the text of the image to create a new brand.
- Do not use the text of the image to create a new logo.
- Do not use the text of the image to create a new brand.
- Do not use the text of the image to create a new logo.
- Do not use the text of the image to create a new brand.
- Do not use the text of the image to create a new brand.
- Do not use the text of the image to create a new logo.
- Do not use the text of
```

</details>

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "the text of the...")
- Excessive bullet points (32)
- No overlap with supplied context indicators
- Output degeneration (incomplete_sentence: ends with 'of')
- Generation loop (repetitive_tail)
- Metadata alignment issue (low_metadata_alignment)
- Missing sections (title, description, keywords)
- Low draft improvement
- Likely capped by max token budget

---

<a id="model-mlx-community-ornith-10-35b-bf16"></a>

### ✅ mlx-community/Ornith-1.0-35B-bf16

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/Ornith-1.0-35B-bf16</summary>

```text
Title:
- RNLI Lifeboat Station with Ferris Wheel and Hotel

Description:
- A white RNLI lifeboat station with green roller doors stands in the foreground under an overcast sky, with a large white Ferris wheel visible to the left and a multi-storey hotel building to the right.

Keywords:
- Lifeboat Station, RNLI, Ferris Wheel, Hotel, Southend-on-Sea, England, UK, Overcast, Cloudy Sky, Green Doors, Sign, Flag, Antenna, Parking, Asphalt, Waterfront, Architecture, Seaside
```

</details>

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-qwen35-35b-a3b-4bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-4bit

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/Qwen3.5-35B-A3B-4bit</summary>

```text
Title: RNLI Lifeboat Station Southend-on-Sea

Description: A lifeboat station with closed green roller shutters and an RNLI sign stands in Southend-on-Sea, England, with a large Ferris wheel and hotel buildings visible in the background under an overcast sky.

Keywords: RNLI, Lifeboat Station, Southend-on-Sea, England, UK, Ferris Wheel, Hotel, Green Shutters, Overcast Sky, Cloudy, Waterfront, Amusement Ride, Architecture, Sign, Flag, Antenna, Parking, Asphalt
```

</details>

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-qwen35-35b-a3b-6bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-6bit

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/Qwen3.5-35B-A3B-6bit</summary>

```text
Title:
RNLI Lifeboat Station Southend-on-Sea England

Description:
A white RNLI lifeboat station with large green roller doors stands in the foreground under an overcast sky, with a large Ferris wheel visible to the left and a multi-storey hotel building in the background.

Keywords:
RNLI, Lifeboat Station, Southend-on-Sea, England, UK, Europe, Ferris Wheel, Hotel, Overcast Sky, Green Doors, White Building, Asphalt, Parking, Barbed Wire, Flag, Antenna, Seaside, Architecture
```

</details>

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-qwen35-9b-mlx-4bit"></a>

### ✅ mlx-community/Qwen3.5-9B-MLX-4bit

- _Verdict:_ context_budget | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/Qwen3.5-9B-MLX-4bit</summary>

```text
Title:
RNLI Lifeboat Station, Southend-on-Sea, UK

Description:
A modern RNLI lifeboat station with green roller shutter doors and a curved white roof, situated in a paved forecourt under an overcast sky, with a large Ferris wheel and a multi-storey hotel visible in the background.

Keywords:
RNLI, Lifeboat Station, Southend-on-Sea, UK, Roller Shutter, Green Doors, White Roof, Curved Roof, Overcast Sky, Cloudy, Ferris Wheel, Amusement Ride, Hotel, Palace, Brick Wall, Fence, Barbed Wire, Antenna, Satellite Dish, Flag, Paved Forecourt, Parking Markings, Modern Architecture, Coastguard, Emergency Services, Waterfront, England, Europe, GBR, 2026-07-18
```

</details>

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context_budget
- Metadata alignment issue (low_metadata_alignment)
- Keyword count violation (30; expected 10-18)
- Unverified context copy

---

<a id="model-mlx-community-qwen35-35b-a3b-bf16"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-bf16

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/Qwen3.5-35B-A3B-bf16</summary>

```text
Title: RNLI Lifeboat Station Southend-on-Sea England
Description: A white RNLI lifeboat station with closed green roller shutters stands in the foreground under an overcast sky, with a large Ferris wheel and the Palace Hotel visible in the background.
Keywords: RNLI, Lifeboats, Southend-on-Sea, England, UK, Europe, Ferris wheel, Palace Hotel, Amusement ride, Green shutters, Overcast sky, Cloudy, Waterfront, Seaside, Asphalt, Parking, Barbed wire, Antenna
```

</details>

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-qwen2-vl-2b-instruct-4bit"></a>

### ❌ mlx-community/Qwen2-VL-2B-Instruct-4bit

- _Verdict:_ cutoff_degraded | user=avoid
- _Why:_ At mixed burden (16933 tokens), output became repetitive. | harness
  contract | text repetition | missing required sections
- _Maintainer:_ issue_ready | owner=mlx | confidence=high
- _Next action:_ File the retained reproduction evidence with the suspected
  owner.

<details>
<summary>Complete generated output: mlx-community/Qwen2-VL-2B-Instruct-4bit</summary>

```text
-001.jpg
- 17:57:06 BST
- 50.81744°N, 0.13454°W
- 17:57:06 BST: The image is taken at 17:57:06 BST. The time is in the 17:57:06 format, which is the time in the BST (British Standard Time) zone. The time is in the 17:57:06 format, which is the time in the BST (British Standard Time) zone. The time is in the 17:57:06 format, which is the time in the BST (British Standard Time) zone. The time is in the 17:57:06 format, which is the time in the BST (British Standard Time) zone. The time is in the 17:57:06 format, which is the time in the BST (British Standard Time) zone. The time is in the 17:57:06 format, which is the time in the BST (British Standard Time) zone. The time is in the 17:57:06 format, which is the time in the BST (British Standard Time) zone. The time is in the 17:57:06 format, which is the time in the BST (British Standard Time) zone. The time is in the 17:57:06 format, which is the time in the BST (British Standard Time) zone. The time is in the 17:57:06 format, which is the time in the BST (British Standard Time) zone. The time is in the 17:57:06 format, which is the time in the BST (British Standard Time) zone. The time is in the 17:57:06 format, which is the time in the BST (British Standard Time) zone. The time is in the 17:57:06 format, which is the time in the BST (British Standard Time) zone. The time is in the 17:57:06 format, which is the time in the BST (British Standard Time) zone. The time is in the 17:5
```

</details>

⚠️ _Quality Warnings:_

- ⚠️HARNESS:long_context
- Repetitive output (phrase: "the time is in...")
- No overlap with supplied context indicators
- Generation loop (repetitive_tail)
- Metadata alignment issue (low_metadata_alignment)
- Missing sections (title, description, keywords)
- Unverified context copy
- Low draft improvement
- Likely capped by max token budget
- suspicious_precision: 2 overly precise numbers
- long_context_repetition(16933tok)

---

<a id="model-mlx-community-gemma-4-31b-bf16"></a>

### ❌ mlx-community/gemma-4-31b-bf16

- _Verdict:_ cutoff_degraded | user=avoid
- _Why:_ text repetition | missing required sections | token cap truncation |
  hit token cap (500)
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/gemma-4-31b-bf16</summary>

```text


Image:
- A lifeboat station with a ferris wheel in the background.
- The image shows a lifeboat station with a ferris wheel in the background. The lifeboat station is a modern building with a curved roof and large green doors. The ferris wheel is a large, colorful structure with many cabins. The sky is overcast, and the ground is covered with asphalt. There is a fence and a flag in the foreground. The image is taken from a low angle, and the perspective is slightly distorted. The colors are muted, and the lighting is flat. The composition is balanced, with the lifeboat station and ferris wheel on either side of the frame. The image is a photograph, and the style is realistic. The image is a good example of a modern lifeboat station and a ferris wheel in a seaside town. The image is a good example of a modern lifeboat station and a ferris wheel in a seaside town. The image is a good example of a modern lifeboat station and a ferris wheel in a seaside town. The image is a good example of a modern lifeboat station and a ferris wheel in a seaside town. The image is a good example of a modern lifeboat station and a ferris wheel in a seaside town. The image is a good example of a modern lifeboat station and a ferris wheel in a seaside town. The image is a good example of a modern lifeboat station and a ferris wheel in a seaside town. The image is a good example of a modern lifeboat station and a ferris wheel in a seaside town. The image is a good example of a modern lifeboat station and a ferris wheel in a seaside town. The image is a good example of a modern lifeboat station and a ferris wheel in a seaside town. The image is a good example of a modern lifeboat station and a ferris wheel in a seaside town. The image is a good example of a modern lifeboat station and a ferris wheel in a seaside town. The image is a good example of a modern lifeboat station and a ferris wheel in a seaside town. The image is a good example of a modern lifeboat station and a ferris wheel in a seaside town. The image is a good example of a modern lifeboat station and a ferris wheel in a seaside town. The image is a good example of a modern lifeboat station and a ferris wheel in a seaside town. The image is a good example of a modern lifeboat station and a ferris wheel in
```

</details>

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "a ferris wheel in...")
- Output degeneration (incomplete_sentence: ends with 'in')
- Generation loop (repetitive_tail)
- Missing sections (title, description, keywords)
- Low draft improvement
- Likely capped by max token budget

---

<a id="model-mlx-community-qwen35-27b-mxfp8"></a>

### ✅ mlx-community/Qwen3.5-27B-mxfp8

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/Qwen3.5-27B-mxfp8</summary>

```text
Title:
RNLI Lifeboat Station, Southend-on-Sea, England

Description:
A modern white lifeboat station building with two large green roller doors stands under an overcast sky, featuring a blue "Lifeboats" sign with the RNLI logo. In the background, a large white Ferris wheel is visible to the left, while a multi-story cream building labeled "Palace" stands to the right, with barbed wire fencing lining the perimeter walls.

Keywords:
RNLI, Lifeboat Station, Southend-on-Sea, England, Roller Doors, Ferris Wheel, Overcast Sky, Palace Hotel, Barbed Wire, Coastal Architecture, Green Shutters, Cloudy, UK, Seaside, Modern Building, Flag, Antenna, Asphalt
```

</details>

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-qwen36-27b-mxfp8"></a>

### ✅ mlx-community/Qwen3.6-27B-mxfp8

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/Qwen3.6-27B-mxfp8</summary>

```text
Title:
RNLI Lifeboat Station, Southend-on-Sea, England

Description:
A white RNLI lifeboat station building with green roller doors stands in the foreground, featuring a blue sign and a flag on the roof. In the background, a large Ferris wheel and a multi-story hotel building are visible under an overcast sky.

Keywords:
Lifeboat Station, RNLI, Southend-on-Sea, England, UK, Europe, Ferris wheel, Hotel, Overcast, Cloudy, Green, White, Architecture, Seaside, Flag, Roller door, Parking, Fence, Brick wall
```

</details>

⚠️ _Quality Warnings:_

- Keyword count violation (19; expected 10-18)

---

<a id="model-mlx-community-qwen35-27b-4bit"></a>

### ✅ mlx-community/Qwen3.5-27B-4bit

- _Verdict:_ clean | user=recommended
- _Why:_ no flagged signals
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/Qwen3.5-27B-4bit</summary>

```text
Title:
RNLI Lifeboat Station, Southend-on-Sea, Essex, England

Description:
A modern white RNLI lifeboat station with two large green roller shutter doors stands in the foreground, featuring a blue sign with the organisation's logo. In the background, a large white Ferris wheel and a multi-story hotel building are visible against a cloudy, overcast sky.

Keywords:
RNLI, Lifeboat Station, Southend-on-Sea, Ferris wheel, Roller shutter, Hotel, Overcast sky, Modern architecture, Flag, Antenna, Parking bay, Brick wall, Barbed wire, Coastal, England, Europe, Amusement ride, Cloudy
```

</details>

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-2506-bf16"></a>

### ❌ mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

- _Verdict:_ cutoff_degraded | user=avoid
- _Why:_ thinking trace | reasoning budget exhausted | text repetition | hit
  token cap (500)
- _Maintainer:_ not_applicable | owner=unknown | confidence=low
- _Next action:_ No maintainer issue action is indicated by this run.

<details>
<summary>Complete generated output: mlx-community/Kimi-VL-A3B-Thinking-2506-bf16</summary>

```text
◁think▷Okay, let's tackle this. First, I need to analyze the image and create a catalog record with the specified sections: Title, Description, and Keywords. The user wants it in British English, so I should use terms like "Lifeboats" instead of "Lifeboat" and "Pier" instead of "Pier" if needed.

Looking at the image, the main subject is a lifeboat station building. The Ferris wheel is visible in the background, which is a key detail. The building has green doors, a "Lifeboats" sign, and there's a flag. The location is Southend-on-Sea, England, UK, as per the context. The sky is overcast. The parking lot in the foreground has yellow lines. The Ferris wheel is part of an amusement park, so maybe "amusement ride" is a keyword. The building has modern architecture with a curved roof. The flag is the English flag. The building is near the Thames Estuary, as per the context. The building is part of the RNLI, so "RNLI" should be included. The flag is the Union Jack. The Ferris wheel is a prominent feature. The parking area is asphalt. The building has a sign with "Lifeboats". The Ferris wheel is colorful with red accents. The building has a satellite dish on top. The Ferris wheel is large and white. The building has a modern design with large windows. The Ferris wheel is part of a pier or amusement park. The sky is cloudy. The parking lot is marked with yellow lines. The building is near a waterfront. The Ferris wheel is a prominent feature in the background. The building is a lifeboat station. The Ferrland is in Southend-on-Sea. The Ferris wheel is part of an amusement park. The building has a sign with "Lifeboats". The Ferris wheel is a large, white structure. The building has a curved roof. The Ferris wheel is visible behind the building. The building is part of the RNLI. The Ferris wheel is colorful. The parking lot is empty. The building has a satellite dish. The Ferris wheel is part of a pier. The building is near the Thames Estuary. The Ferris wheel is a prominent feature. The building has a sign with "Lifeboats". The Ferris wheel is a large structure. The building has a modern design
```

</details>

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "the ferris wheel is...")
- Generation loop (repetitive_tail)
- Missing sections (title, description, keywords)
- Thinking trace incomplete (expected model protocol; opened with ◁think▷)
- Low draft improvement
- Likely capped by max token budget

---

<!-- markdownlint-enable MD033 MD034 -->
