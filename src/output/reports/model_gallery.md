# Model Output Gallery

Generated on: 2026-07-17 23:16:01 BST

Complete per-model evidence artifact with image metadata, the source prompt,
summary tables, diagnostics, and full generated output for every attempted
model.

_Action Snapshot: see [results.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/results.md) for the full summary._

## Run Stamps

- `mlx-vlm`: `0.6.5`
- `mlx`: `0.32.1.dev20260717+b7c3dd6d`
- `mlx-lm`: `0.31.3`
- `transformers`: `5.14.1`
- `tokenizers`: `0.22.2`
- `huggingface-hub`: `1.24.0`
- _Python Version:_ 3.13.13
- _OS:_ Darwin 25.5.0
- _macOS Version:_ 26.5.2
- _GPU/Chip:_ Apple M5 Max

## Image Metadata

- _Title:_ , Seafront, Deal, England, UK, GBR, Europe
- _Description:_ A coastal view of Deal, Kent, UK, showing the shingle beach,
  sea, and seafront buildings on a partly cloudy day.
- _Keywords:_ Beach, Buildings, Cars, Coastline, Deal, England, Europe,
  Horizon, Kent, Landscape, People, Promenade, Seaside, Shore, Sitting, Sky,
  Swimming, Town, UK, Walking, Water, Waterfront, Waves, architecture, bay,
  british, chalk cliffs, cityscape, clouds, coast, coastal, coastal town,
  destination, english, holiday, houses, marine, maritime, nature, ocean,
  outdoors, overcast, pebble beach, relaxing, resort, sand, scenic, sea, sea
  wall, seafront, seaside town, shingle beach, shoreline, street, summer,
  tourism, traditional, travel, vacation, view, waves crashing
- _Date:_ 2026-07-11 19:04:26 BST
- _Time:_ 19:04:26
- _GPS:_ 51.226814°N, 1.401142°E

## Prompt

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Analyze this image for cataloguing metadata, using British English.
>
> Use only details that are clearly and definitely visible in the image. If a
> detail is uncertain, ambiguous, partially obscured, too small to verify, or
> not directly visible, leave it out. Do not guess.
>
> Treat the metadata hints below as a draft catalog record. Keep only details
> that are clearly confirmed by the image, correct anything contradicted by
> the image, and add important visible details that are definitely present.
>
> &#8203;Return exactly these three sections, and nothing else:
>
> &#8203;Title:
> &#45; 5-10 words, concrete and factual, limited to clearly visible content.
> &#45; Output only the title text after the label.
> &#45; Do not repeat or paraphrase these instructions in the title.
>
> &#8203;Description:
> &#45; 1-2 factual sentences describing the main visible subject, setting,
> lighting, action, and other distinctive visible details. Omit anything
> uncertain or inferred.
> &#45; Output only the description text after the label.
>
> &#8203;Keywords:
> &#45; 10-18 unique comma-separated terms based only on clearly visible subjects,
> setting, colors, composition, and style. Omit uncertain tags rather than
> guessing.
> &#45; Output only the keyword list after the label.
>
> &#8203;Rules:
> &#45; Include only details that are definitely visible in the image.
> &#45; Reuse metadata terms only when they are clearly supported by the image.
> &#45; If metadata and image disagree, follow the image.
> &#45; Prefer omission to speculation.
> &#45; Do not copy prompt instructions into the Title, Description, or Keywords
> fields.
> &#45; Do not infer identity, location, event, brand, species, time period, or
> intent unless visually obvious.
> &#45; Do not output reasoning, notes, hedging, or extra sections.
>
> Context: Authoritative context:
> &#45; Location terms: England, Europe, Town, UK
> &#45; Capture date/time: 2026-07-11 19:04:26 BST 19:04:26
> &#45; GPS: 51.226814°N, 1.401142°E
> &#45; Use this factual context where it improves the catalogue record; do not
> claim that contextual facts are visually observable.
>
> &#8203;Draft descriptive metadata:
> &#45; Existing title: Seafront, Deal, England, UK, GBR, Europe
> &#45; Existing description: A coastal view of Deal, Kent, UK, showing the
> shingle beach, sea, and seafront buildings on a partly cloudy day.
> &#45; Existing keywords: Beach, Buildings, Cars, Coastline, Deal, Horizon, Kent,
> Landscape, People, Promenade, Seaside, Shore, Sitting, Sky, Swimming,
> Walking, Water, Waterfront, Waves, architecture
> &#45; Treat this draft as fallible. Retain supported details, correct errors,
> and add important visible information.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

## Model Quality Summary

Skim-first view of what each model returned, or the strongest diagnostic when it did not produce usable output.

<!-- markdownlint-disable MD034 MD049 -->

| Model                                                                                                                   | Result                        | Quality / diagnostic                                                    | Response / diagnostic                                                                                                                                                                                                                                                                                                                        |
|-------------------------------------------------------------------------------------------------------------------------|-------------------------------|-------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `caveat` / `clean`            | low-draft-improvement                                                   | Title: Deal Seafront, England, UK, GBR, Europe Description: A coastal view of Deal, Kent, UK, showing the shingle beach, sea, and seafront buildings on a partly cloudy day. Keywords: Beach, Buildings, Cars, C ... [tail] , Promenade, Seaside, Shore, Sitting, Sky, Swimming, Walking, Water, Waterfront, Waves, architecture             |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                            | `avoid` / `model shortcoming` | low-draft-improvement; missing-sections                                 | Title: Seafront, Deal, England, UK, GBR, Europe Description: A coastal view of Deal, Kent, UK, showing the shingle beach, sea, and seafront buildings on a partly cloudy day.                                                                                                                                                                |
| [`mlx-community/LFM2-VL-1.6B-8bit`](#model-mlx-community-lfm2-vl-16b-8bit)                                              | `recommended` / `clean`       | low-draft-improvement                                                   | Title: Deal Beach, Deal, Kent, UK, GBR, Europe Description: A coastal view of Deal, Kent, UK, showing the shingle beach, sea, and seafront buildings on a partly cloudy day. Keywords: Beach, Buildings, Coastline, Deal, Horizon, Kent, Landscape, People, Promenade, Seaside, Shore, Sitting, Sky, Waves, Architecture                     |
| [`HuggingFaceTB/SmolVLM-Instruct`](#model-huggingfacetb-smolvlm-instruct)                                               | `avoid` / `model shortcoming` | low-draft-improvement; context-ignored; missing-sections                | Seafront, Deal, England, UK, GBR, Europe.                                                                                                                                                                                                                                                                                                    |
| [`mlx-community/SmolVLM-Instruct-bf16`](#model-mlx-community-smolvlm-instruct-bf16)                                     | `avoid` / `model shortcoming` | low-draft-improvement; context-ignored; missing-sections                | Seafront, Deal, England, UK, GBR, Europe.                                                                                                                                                                                                                                                                                                    |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `caveat` / `clean`            | low-draft-improvement                                                   | Title: Deal, Kent, England, UK, GBR, Europe, Seaside, Beach, Shingle, Buildings, Promenade, Horizon, Landscape, People, Seaside, Shore, Sitting, Sky, Waves, Architecture Description: A coastal view of Deal, K ... [tail] Coastline, Deal, Horizon, Kent, Landscape, People, Seaside, Shore, Sitting, Sky, Waves, Architecture             |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `avoid` / `harness`           | harness:stop-token; low-draft-improvement; reasoning-leak               | &lt;think&gt; &lt;/think&gt; Title: Beachfront with buildings and sea Description: The image shows a coastal beach with sandy areas and seaside buildings under a partly cloudy sky. Waves are visible near the shore, and p ... [tail] ine. Keywords: Beach, Buildings, Sea, Sand, Coastal, Buildings, Water, People, Sky, Waves, Shoreline |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `recommended` / `clean`       | low-draft-improvement                                                   | Title: Seafront buildings and shingle beach with waves Description: A coastal scene shows a row of multi-storey buildings overlooking a shingle beach and the sea under a cloudy sky. White waves break on the s ... [tail] ildings, coastline, horizon, landscape, ocean, people, sea, seaside, shingle, sky, waves, waterfront             |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `avoid` / `model shortcoming` | low-draft-improvement; missing-sections                                 | Title: Seaside View of Deal, Kent, UK Description: A coastal view of Deal, Kent, UK, showcasing the shingle beach, sea, and seafront buildings on a partly cloudy day. The image captures the essence of the sea ... [tail] nditions. The metadata also includes a draft version of the description, which is not yet finalized.             |
| [`mlx-community/paligemma2-10b-ft-docci-448-6bit`](#model-mlx-community-paligemma2-10b-ft-docci-448-6bit)               | `avoid` / `model shortcoming` | low-draft-improvement; context-ignored; missing-sections                | - Do not copy prompt instructions into the Title, Description, or Keywords fields.                                                                                                                                                                                                                                                           |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `avoid` / `semantic mismatch` | missing terms: Cars, Coastline, Deal, Horizon, Kent                     | Title: Coastal Row of Residential Buildings Beside Shingle Beach Description: A coastal scene shows a row of multi-storey residential buildings adjacent to a pebbly shingle beach, with gentle waves rolling on ... [tail] gle shore, UK, waves, waterfront, promenade, pebbles, seaside setting, urban seaside, light activity             |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `avoid` / `model shortcoming` | low-draft-improvement; missing-sections; formatting                     | <\|channel>thought <channel\|>Title: - Shingle beach and seafront buildings in Deal, England Description: - A coastal view of a shingle beach with choppy waves lapping a row of multi-story buildings under a clo ... [tail] zon, Kent, landscape, people, promenade, seaside, shore, sky, water, waterfront, waves, architecture           |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `avoid` / `model shortcoming` | low-draft-improvement; missing-sections; formatting                     | <\|channel>thought <channel\|>Title: - Seafront buildings and shingle beach in Deal England Description: - A coastal view of a shingle beach with waves and multi-story seafront buildings under a cloudy sky. Sev ... [tail] l, England, Kent, landscape, people, promenade, seaside, sea, shingle, sky, water, waterfront, waves           |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `avoid` / `model shortcoming` | unverified-context-copy; metadata-borrowing                             | Title: Seafront, Deal, England, UK, GBR, Europe Description: A coastal view of Deal, Kent, UK, showing the shingle beach, sea, and seafront buildings on a partly cloudy day. Keywords: Beach, Buildings, Cars, ... [tail] draft as fallible. Retain supported details, correct errors, and add important visible information.               |
| [`microsoft/Phi-3.5-vision-instruct`](#model-microsoft-phi-35-vision-instruct)                                          | `caveat` / `clean`            | low-draft-improvement                                                   | Title: Coastal Scene at Deal Beach Description: A coastal view of Deal, Kent, UK, featuring a shingle beach, sea, and seafront buildings on a partly cloudy day. The beach is populated with a few individuals, ... [tail] pe, People, Promenade, Seaside, Shore, Waves, Architecture, Walking, Sitting, Waterfront, Water, Sky              |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `caveat` / `clean`            | low-draft-improvement                                                   | Title: Coastal Scene at Deal Beach Description: A coastal view of Deal, Kent, UK, featuring a shingle beach, sea, and seafront buildings on a partly cloudy day. The beach is populated with a few individuals, ... [tail] pe, People, Promenade, Seaside, Shore, Waves, Architecture, Walking, Sitting, Waterfront, Water, Sky              |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `avoid` / `semantic mismatch` | low-draft-improvement                                                   | Title: Beachfront with Sea and Buildings Description: A coastal scene featuring a pebble beach, sea with waves, and a row of buildings along the promenade on a partly cloudy day. Keywords: Beach, Buildings, C ... [tail] , Landscape, People, Promenade, Seaside, Shore, Sky, Walking, Water, Waterfront, Waves, architecture             |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `avoid` / `semantic mismatch` | missing terms: Coastline, Deal, Kent, Landscape, Promenade              | **Title:** *Seafront buildings along sandy beach with waves and distant horizon* **Description:** A coastal scene featuring a row of traditional townhouses along a sandy shingle beach, with waves crashing ont ... [tail] people sitting, deck chairs, partly cloudy sky, waterfront, distant horizon, coastal scene, England              |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `caveat` / `clean`            | low-draft-improvement                                                   | Title: Seafront Buildings and Beach Description: A coastal scene featuring a row of colorful seafront buildings along a sandy beach with gentle waves lapping the shore under a partly cloudy sky. Keywords: Bea ... [tail] Shingle, Sky, Waves, architecture, cloudy, gentle, partly cloudy, row, sandy, shore, sky, waterfront             |
| [`mlx-community/gemma-4-31b-bf16`](#model-mlx-community-gemma-4-31b-bf16)                                               | `avoid` / `harness`           | harness:prompt-template; low-draft-improvement; context-ignored         | Image:                                                                                                                                                                                                                                                                                                                                       |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `avoid` / `semantic mismatch` | missing terms: Cars, Coastline, Deal, Kent, Landscape                   | Title: Coastal Town Promenade with Shingle Beach and Waves Description: The image shows a row of multi-storey buildings along a pebble beachfront, with the sea waves crashing against the shore. A few people a ... [tail] water, multi-storey architecture, shoreline, urban seaside, muted lighting, horizon view, beachfront             |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                            | `caveat` / `clean`            | low-draft-improvement                                                   | Title: Seafront, Deal, England, UK, GBR, Europe Description: A coastal view of Deal, Kent, UK, showing the shingle beach, sea, and seafront buildings on a partly cloudy day. Keywords: Beach, Buildings, Cars, ... [tail] , Promenade, Seaside, Shore, Sitting, Sky, Swimming, Walking, Water, Waterfront, Waves, architecture              |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `caveat` / `clean`            | context-echo; low-draft-improvement; formatting                         | Title: Seafront, Deal, England, UK, GBR, Europe. Description: A coastal view of Deal, Kent, UK, showing the shingle beach, sea, and seafront buildings on a partly cloudy day. Keywords: Beach, Buildings, Cars, ... [tail] e, Shore, Sitting, Sky, Swimming, Walking, Water, Waterfront, Waves, architecture.<end_of_utterance>             |
| [`mlx-community/gemma-3n-E2B-4bit`](#model-mlx-community-gemma-3n-e2b-4bit)                                             | `avoid` / `cutoff degraded`   | repetitive; low-draft-improvement; cutoff                               | - Do not go to the. - Do not go to the. - Do not go to the. - Do not go to the. - Do not go to the. - Do not go to the. - Do not go to the. - Do not go to the. - Do not go to the. - Do not go to the. - Do not go to the. - Do not go to the. - Do not go to the. - Do not go to the. - Do not go to the. - Do not go to th...             |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `caveat` / `clean`            | low-draft-improvement                                                   | Title: Seafront buildings, shingle beach, waves, Deal Description: A coastal view showing buildings along a promenade, a shingle beach, and waves breaking on the shore. Several people are visible on the beach ... [tail] e, Shore, Sky, Waves, Water, Waterfront, architecture, shingle, terrace, row houses, cloudy sky, sea             |
| [`qnguyen3/nanoLLaVA`](#model-qnguyen3-nanollava)                                                                       | `avoid` / `cutoff degraded`   | repetitive; context-echo; low-draft-improvement                         | Title: Seafront Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, D...             |
| [`mlx-community/InternVL3-14B-8bit`](#model-mlx-community-internvl3-14b-8bit)                                           | `caveat` / `clean`            | low-draft-improvement                                                   | Title: Coastal Town Beachfront Description: A coastal scene featuring a shingle beach with gentle waves, a row of seafront buildings, and a few people enjoying the beach. The sky is partly cloudy, and the wat ... [tail] e, People, Promenade, Seaside, Shore, Sky, Swimming, Walking, Water, Waterfront, Waves, architecture             |
| [`mlx-community/paligemma2-10b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-10b-ft-docci-448-bf16)               | `avoid` / `model shortcoming` | low-draft-improvement; context-ignored; missing-sections                | - Do not copy prompt instructions into the Title, Description, or Keywords fields.                                                                                                                                                                                                                                                           |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `recommended` / `clean`       | low-draft-improvement                                                   | Title: Seafront buildings and shingle beach in Deal, England Description: A coastal view shows a row of multi-storey buildings facing a shingle beach and the sea under a cloudy sky. Several people are scatter ... [tail] zon, Kent, landscape, people, promenade, seaside, shore, sky, water, waterfront, waves, architecture             |
| [`mlx-community/pixtral-12b-bf16`](#model-mlx-community-pixtral-12b-bf16)                                               | `avoid` / `semantic mismatch` | low-draft-improvement                                                   | Title: Seafront Buildings and Beach Description: A row of colorful seafront buildings lines a sandy beach with gentle waves lapping at the shore under a partly cloudy sky. Keywords: Beach, Buildings, Coastlin ... [tail] le, Sky, Waves, architecture, cloudy, gentle, partly cloudy, row, sandy, seafront, shore, sky, waves             |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `recommended` / `clean`       | low-draft-improvement                                                   | Title: Seafront buildings and shingle beach Description: A coastal scene featuring a row of seafront buildings, a shingle beach, and the sea with gentle waves. The sky is partly cloudy, and a few people are v ... [tail] andscape, People, Promenade, Seaside, Shore, Sky, Water, Waves, architecture, shingle, partly cloudy             |
| [`mlx-community/GLM-4.6V-Flash-6bit`](#model-mlx-community-glm-46v-flash-6bit)                                          | `caveat` / `context budget`   | low-draft-improvement; context-budget; missing-sections                 | <\|begin_of_box\|>Title: Seafront with shingle beach and sea Description: A coastal scene showing a shingle beach, sea with waves, and a row of seafront buildings. People are visible on the beach. The sky is ov ... [tail] eople, Promenade, Seaside, Shore, Sitting, Sky, Water, Waterfront, Waves, architecture<\|end_of_box\|>         |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)            | `avoid` / `model shortcoming` | low-draft-improvement; missing-sections                                 | **Title** Deal Seafront, England, UK **Description** A coastal view of Deal, Kent, UK, showing the shingle beach, sea, and seafront buildings on a partly cloudy day. The beach is empty, with a few people walk ... [tail] , Promenade, Seaside, Shore, Sitting, Sky, Swimming, Walking, Water, Waterfront, Waves, Architecture             |
| [`mlx-community/gemma-3-27b-it-qat-8bit`](#model-mlx-community-gemma-3-27b-it-qat-8bit)                                 | `caveat` / `clean`            | low-draft-improvement                                                   | Title: Seafront buildings, shingle beach, and waves Description: A coastal view shows a row of white and pastel-coloured buildings along a shingle beach, with waves breaking on the shore. Several people are v ... [tail] Shore, Sky, Waves, Water, Waterfront, architecture, clouds, daylight, shingle, town, white buildings             |
| [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](#model-jqlive-kimi-vl-a3b-thinking-2506-6bit)                                 | `avoid` / `cutoff degraded`   | cutoff; thinking-incomplete; missing-sections                           | ◁think▷Got it, let's analyze the image step by step. First, the title: The image shows a coastal scene with buildings along the shore, a beach, and the sea. The location is Deal, Kent, UK. So the title should ... [tail] ont, Waves, architecture. But some of these might be inferred. Let's stick to what's clearly visible             |
| [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](#model-mlx-community-kimi-vl-a3b-thinking-8bit)                             | `avoid` / `cutoff degraded`   | cutoff; thinking-trace; missing-sections                                | ◁think▷Okay, let's analyze this image for metadata cataloging based on the user's instructions. First, I need to focus only on what's clearly visible and avoid any assumptions or inferences. The image shows a ... [tail] analyzed the image and the metadata draft provided. The image clearly shows a coastal scene in Deal,             |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `avoid` / `model shortcoming` | missing-sections                                                        | ## Image Description: This is a wide shot of the seafront at Seafront, Deal, in Kent, UK, captured on a partly cloudy day. The scene is dominated by a long, sloping shingle beach that stretches from the foreg ... [tail] eye towards the buildings and the sea. The scene is clearly identifiable as a British coastal town.              |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `avoid` / `cutoff degraded`   | cutoff                                                                  | Title: Deal, Kent, UK Description: A coastal view showing a shingle beach, sea, and seafront buildings on a partly cloudy day. People are visible on the beach. Keywords: Deal, Kent, UK, England, Europe, Beach ... [tail] ance, Scan, View, Look, Gaze, Stare, Glance, Scan, View, Look, Gaze, Stare, Glance, Scan, View, Look             |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                             | `avoid` / `cutoff degraded`   | repetitive; low-draft-improvement; cutoff                               | - Output only the description text after the label. - Do not output the title text after the label. - Do not output the description text after the label. - Output only the description text after the label. - Do not output the description text after the label. - Output only the description text after the label. - Do...              |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `caveat` / `context budget`   | harness:long-context; low-draft-improvement; context-budget             | -                                                                                                                                                                                                                                                                                                                                            |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `avoid` / `cutoff degraded`   | unverified-context-copy; metadata-borrowing; cutoff                     | Here are my reasoning steps: We need to produce a catalog entry with Title, Description, Keywords. Use only details that are clearly visible in the image. The image shows a seafront with a shingle beach, a ro ... [tail] something like "Deal Seafront, Shingle Beach, Buildings, Cars, Waves". That's 7 words. Or "Deal seaf             |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `avoid` / `cutoff degraded`   | repetitive; low-draft-improvement; cutoff                               | Let's tackle this step by step. First, the user provided a prompt with a lot of repetitive and unclear terms, but the key is to focus on what's actually visible in the image. Wait, but wait—the user mentioned ... [tail] tade stade stade stade stade stade stade stade stade stade stade stade stade stade stade stade stade             |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `caveat` / `context budget`   | low-draft-improvement; context-budget; missing-sections                 | <\|begin_of_box\|>Title: Seafront buildings, shingle beach, and waves at Deal Description: A coastal scene with shingle beach, turbulent sea, and a row of multi-story buildings along the shore, under a grey sky ... [tail] eople, Promenade, Seaside, Shore, Sitting, Sky, Water, Waterfront, Waves, architecture<\|end_of_box\|>         |
| [`meta-llama/Llama-3.2-11B-Vision-Instruct`](#model-meta-llama-llama-32-11b-vision-instruct)                            | `avoid` / `model shortcoming` | low-draft-improvement; missing-sections                                 | Seafront, Deal, England, UK The image depicts a seaside town with a long stretch of sandy beach and a row of buildings along the shore. The beach is empty, with a few people scattered about, and the sea is choppy with waves crashing against the shore. The sky is overcast, with a few birds flying overhead.                           |
| [`mlx-community/Molmo-7B-D-0924-8bit`](#model-mlx-community-molmo-7b-d-0924-8bit)                                       | `caveat` / `clean`            | low-draft-improvement                                                   | Remove uncertain tags. Title: Shingle beach at Deal, Kent, UK Description: A partly cloudy day at Deal's shingle beach, with the town's buildings visible along the horizon. The sea is visible on the right, wi ... [tail] , Promenade, Seaside, Shore, Sitting, Sky, Swimming, Walking, Water, Waterfront, Waves, architecture             |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `caveat` / `context budget`   | harness:long-context; low-draft-improvement; context-budget             | <\|im_start\|> addCriterion                                                                                                                                                                                                                                                                                                                  |
| [`mlx-community/Molmo-7B-D-0924-bf16`](#model-mlx-community-molmo-7b-d-0924-bf16)                                       | `caveat` / `clean`            | low-draft-improvement                                                   | Remove uncertain tags. Title: Seafront, Deal, Kent, UK, GBR, Europe Description: A coastal view of Deal, Kent, UK, showing the shingle beach, sea, and seafront buildings on a partly cloudy day. The beach is m ... [tail] tting, Sky, Swimming, Walking, Water, Waterfront, Waves, architecture, English, Seafront, UK, Europe             |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                           | `avoid` / `semantic mismatch` | low-draft-improvement                                                   | Title: Shingle Beach with Seafront Buildings and People Description: A coastal scene featuring a shingle beach with people sitting on blankets, a seafront lined with buildings, and a cloudy sky. The image cap ... [tail] s, Architecture, England, UK, Town, UK, Landscape, Seaside, Shore, Sitting, Waterfront, Water, Waves             |
| [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16)                             | `caveat` / `context budget`   | harness:long-context; low-draft-improvement; context-budget             | -                                                                                                                                                                                                                                                                                                                                            |
| [`mlx-community/paligemma2-3b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-3b-ft-docci-448-bf16)                 | `avoid` / `cutoff degraded`   | low-draft-improvement; cutoff; context-ignored                          | - Do not copy-prompt instructions into the Title, Description, or Keywords fields. - Do not infer identity, location, event, brand, species, time period, or intent. - Use this factual context where it improve ... [tail] oves the catalogue record; do not claim that contextual facts are visually observable. - Do not copy             |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `avoid` / `cutoff degraded`   | harness:long-context; repetitive; low-draft-improvement                 | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -...             |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `caveat` / `clean`            | low-draft-improvement                                                   | Title: Seafront, Deal, England, UK, GBR, Europe Description: A coastal view of Deal, Kent, UK, showing the shingle beach, sea, and seafront buildings on a partly cloudy day. Keywords: Beach, Buildings, Cars, ... [tail] e, People, Promenade, Seaside, Shore, Sitting, Sky, Swimming, Water, Waterfront, Waves, architecture              |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `caveat` / `clean`            | missing terms: Deal, Horizon, Kent, Swimming, Walking                   | Title: Shingle beach and seafront buildings under cloudy sky Description: A wide shingle beach stretches along the coast with small groups of people sitting on the shore. A row of multi-story buildings lines ... [tail] e, waves, cloudy sky, coastline, shore, water, architecture, England, UK, Europe, landscape, seaside              |
| [`mlx-community/Qwen3.5-35B-A3B-6bit`](#model-mlx-community-qwen35-35b-a3b-6bit)                                        | `recommended` / `clean`       | clean                                                                   | Title: Shingle beach and seafront buildings at Deal Description: A wide shingle beach stretches along a row of multi-story seafront buildings under a cloudy sky. Small groups of people are visible on the beac ... [tail] , People, Promenade, Cars, Architecture, England, UK, Shore, Seaside, Landscape, Horizon, Waterfront             |
| [`mlx-community/Qwen3.5-35B-A3B-bf16`](#model-mlx-community-qwen35-35b-a3b-bf16)                                        | `recommended` / `clean`       | clean                                                                   | Title: Shingle beach, seafront buildings, and sea at Deal Description: A wide shingle beach stretches along a row of multi-storey seafront buildings under a cloudy sky. Small groups of people are visible on t ... [tail] , People, Sky, Clouds, Promenade, Cars, Architecture, Seaside, Shore, England, UK, Europe, Landscape             |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `caveat` / `clean`            | missing terms: Kent, Promenade, Seaside, Swimming, Walking; keywords=19 | Title: Shingle beach and seafront buildings in Deal, England Description: A wide expanse of brown shingle beach stretches along the coastline, bordered by a row of multi-story residential buildings and parked ... [tail] s, Ocean, Cars, People, Sky, Horizon, Residential, Water, Landscape, UK, Europe, Shore, Architecture             |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `caveat` / `clean`            | missing terms: Cars, Deal, Swimming, architecture, GBR; keywords=20     | Title: Seafront buildings and shingle beach at low tide Description: A row of multi-storey seafront buildings lines a wide shingle beach where a small group of people are sitting and walking. The choppy, brow ... [tail] Landscape, People, Promenade, Sea, Seaside, Shingle, Shore, Sky, Town, UK, Water, Waterfront, Waves              |
| [`mlx-community/Qwen3.5-27B-mxfp8`](#model-mlx-community-qwen35-27b-mxfp8)                                              | `caveat` / `clean`            | low-draft-improvement                                                   | Title: Shingle Beach and Seafront Buildings, Deal, England Description: A coastal scene showing a wide expanse of brown shingle beach meeting choppy, brownish sea water with white foam. A row of multi-story r ... [tail] ople, Promenade, Sea, Seaside, Shore, Shingle, Sky, Swimming, Water, Waterfront, Waves, architecture             |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                             | `avoid` / `cutoff degraded`   | harness:long-context; repetitive; unverified-context-copy               | -001.jpg - 19:04:26 BST - 51.226814°N, 1.401142°E - 2026-07-11 19:04:26 BST Title: Deal, Kent, UK, 2026-07-11 19:04:26 BST Keywords: - 19:04:26 BST - Deal, Kent, UK, 2026-07-11 19:04:26 BST - 51.226814°N, 1.4 ... [tail] K, 2026-07-11 19:04:26 BST - Deal, Kent, UK, 2026-07-11 19:04:26 BST - Deal, Kent, UK, 2026-07-11 19             |
| [`mlx-community/Qwen3.5-27B-4bit`](#model-mlx-community-qwen35-27b-4bit)                                                | `recommended` / `clean`       | clean                                                                   | Title: Shingle Beach and Seafront Buildings in Deal, Kent Description: A wide coastal view shows a shingle beach meeting the sea, with a row of multi-coloured buildings and parked cars along the promenade in ... [tail] orizon, Kent, Landscape, People, Promenade, Sea, Shingle, Sky, Swimming, Town, UK, Waterfront, Waves              |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `avoid` / `cutoff degraded`   | cutoff; thinking-incomplete; missing-sections                           | ◁think▷Got it, let's analyze the image step by step. First, the title: The image shows a coastal scene with buildings along a beach. The location is Deal, Kent, UK. So the title should be concise. Let's check ... [tail] ne? No, "Partly cloudy sky" is more specific. So keep both. Let's list them as comma-separated. Now,             |
<!-- markdownlint-enable MD034 MD049 -->

## All Model Output and Cost Summary

Every model in this run, with its skim-first output or diagnostic beside the main runtime and peak-memory signals.

<!-- markdownlint-disable MD034 MD049 -->

| Model                                                                                                                   | Result                        | Output / diagnostic                                                                                                                                                                                                                      |   Gen tok | Total   |   Gen TPS |   Peak GB | Quality signal                                                          |
|-------------------------------------------------------------------------------------------------------------------------|-------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|---------|-----------|-----------|-------------------------------------------------------------------------|
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `caveat` / `clean`            | Title: Deal Seafront, England, UK, GBR, Europe Description: A coastal view of Deal, Kent, UK, showing the shingle beach, sea, and seafront buil ... [tail] g, Sky, Swimming, Walking, Water, Waterfront, Waves, architecture             |       100 | 1.25s   |    510    |       1.5 | low-draft-improvement                                                   |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                            | `avoid` / `model shortcoming` | Title: Seafront, Deal, England, UK, GBR, Europe Description: A coastal view of Deal, Kent, UK, showing the shingle beach, sea, and seafront buildings on a partly cloudy day.                                                            |        47 | 1.68s   |    291    |       2.9 | low-draft-improvement; missing-sections                                 |
| [`mlx-community/LFM2-VL-1.6B-8bit`](#model-mlx-community-lfm2-vl-16b-8bit)                                              | `recommended` / `clean`       | Title: Deal Beach, Deal, Kent, UK, GBR, Europe Description: A coastal view of Deal, Kent, UK, showing the shingle beach, sea, and seafront buil ... [tail] ple, Promenade, Seaside, Shore, Sitting, Sky, Waves, Architecture             |        88 | 1.85s   |    308    |       3   | low-draft-improvement                                                   |
| [`HuggingFaceTB/SmolVLM-Instruct`](#model-huggingfacetb-smolvlm-instruct)                                               | `avoid` / `model shortcoming` | Seafront, Deal, England, UK, GBR, Europe.                                                                                                                                                                                                |        15 | 2.18s   |    130    |       5.7 | low-draft-improvement; context-ignored; missing-sections                |
| [`mlx-community/SmolVLM-Instruct-bf16`](#model-mlx-community-smolvlm-instruct-bf16)                                     | `avoid` / `model shortcoming` | Seafront, Deal, England, UK, GBR, Europe.                                                                                                                                                                                                |        15 | 2.20s   |    110    |       5.7 | low-draft-improvement; context-ignored; missing-sections                |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `caveat` / `clean`            | Title: Deal, Kent, England, UK, GBR, Europe, Seaside, Beach, Shingle, Buildings, Promenade, Horizon, Landscape, People, Seaside, Shore, Sitting ... [tail] dscape, People, Seaside, Shore, Sitting, Sky, Waves, Architecture             |       125 | 2.15s   |    185    |       4.1 | low-draft-improvement                                                   |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `avoid` / `harness`           | &lt;think&gt; &lt;/think&gt; Title: Beachfront with buildings and sea Description: The image shows a coastal beach with sandy areas and seaside buildings u ... [tail] a, Sand, Coastal, Buildings, Water, People, Sky, Waves, Shoreline |        79 | 2.87s   |    209    |       4   | harness:stop-token; low-draft-improvement; reasoning-leak               |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `recommended` / `clean`       | Title: Seafront buildings and shingle beach with waves Description: A coastal scene shows a row of multi-storey buildings overlooking a shingle ... [tail] ape, ocean, people, sea, seaside, shingle, sky, waves, waterfront             |        89 | 5.06s   |    104    |      17   | low-draft-improvement                                                   |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `avoid` / `model shortcoming` | Title: Seaside View of Deal, Kent, UK Description: A coastal view of Deal, Kent, UK, showcasing the shingle beach, sea, and seafront buildings ... [tail] s a draft version of the description, which is not yet finalized.              |       214 | 3.08s   |    307    |       2.2 | low-draft-improvement; missing-sections                                 |
| [`mlx-community/paligemma2-10b-ft-docci-448-6bit`](#model-mlx-community-paligemma2-10b-ft-docci-448-6bit)               | `avoid` / `model shortcoming` | - Do not copy prompt instructions into the Title, Description, or Keywords fields.                                                                                                                                                       |        17 | 4.33s   |     29.6  |      12   | low-draft-improvement; context-ignored; missing-sections                |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `avoid` / `semantic mismatch` | Title: Coastal Row of Residential Buildings Beside Shingle Beach Description: A coastal scene shows a row of multi-storey residential buildings ... [tail] romenade, pebbles, seaside setting, urban seaside, light activity             |       118 | 3.73s   |    119    |       6.4 | missing terms: Cars, Coastline, Deal, Horizon, Kent                     |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `avoid` / `model shortcoming` | <\|channel>thought <channel\|>Title: - Shingle beach and seafront buildings in Deal, England Description: - A coastal view of a shingle beach wit ... [tail] nade, seaside, shore, sky, water, waterfront, waves, architecture           |        87 | 6.17s   |     58.9  |      28   | low-draft-improvement; missing-sections; formatting                     |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `avoid` / `model shortcoming` | <\|channel>thought <channel\|>Title: - Seafront buildings and shingle beach in Deal England Description: - A coastal view of a shingle beach with ... [tail] , promenade, seaside, sea, shingle, sky, water, waterfront, waves           |       100 | 6.40s   |     59.7  |      29   | low-draft-improvement; missing-sections; formatting                     |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `avoid` / `model shortcoming` | Title: Seafront, Deal, England, UK, GBR, Europe Description: A coastal view of Deal, Kent, UK, showing the shingle beach, sea, and seafront bui ... [tail] d details, correct errors, and add important visible information.             |       308 | 4.41s   |    130    |       5.5 | unverified-context-copy; metadata-borrowing                             |
| [`microsoft/Phi-3.5-vision-instruct`](#model-microsoft-phi-35-vision-instruct)                                          | `caveat` / `clean`            | Title: Coastal Scene at Deal Beach Description: A coastal view of Deal, Kent, UK, featuring a shingle beach, sea, and seafront buildings on a p ... [tail] re, Waves, Architecture, Walking, Sitting, Waterfront, Water, Sky             |       162 | 5.07s   |     55.6  |       9.6 | low-draft-improvement                                                   |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `caveat` / `clean`            | Title: Coastal Scene at Deal Beach Description: A coastal view of Deal, Kent, UK, featuring a shingle beach, sea, and seafront buildings on a p ... [tail] re, Waves, Architecture, Walking, Sitting, Waterfront, Water, Sky             |       162 | 5.48s   |     52.8  |       9.6 | low-draft-improvement                                                   |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `avoid` / `semantic mismatch` | Title: Beachfront with Sea and Buildings Description: A coastal scene featuring a pebble beach, sea with waves, and a row of buildings along th ... [tail] side, Shore, Sky, Walking, Water, Waterfront, Waves, architecture             |        79 | 5.94s   |     34.3  |      18   | low-draft-improvement                                                   |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `avoid` / `semantic mismatch` | **Title:** *Seafront buildings along sandy beach with waves and distant horizon* **Description:** A coastal scene featuring a row of traditiona ... [tail] y cloudy sky, waterfront, distant horizon, coastal scene, England             |       118 | 6.16s   |     62.8  |      12   | missing terms: Coastline, Deal, Kent, Landscape, Promenade              |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `caveat` / `clean`            | Title: Seafront Buildings and Beach Description: A coastal scene featuring a row of colorful seafront buildings along a sandy beach with gentle ... [tail] cloudy, gentle, partly cloudy, row, sandy, shore, sky, waterfront             |        93 | 6.60s   |     38.9  |      15   | low-draft-improvement                                                   |
| [`mlx-community/gemma-4-31b-bf16`](#model-mlx-community-gemma-4-31b-bf16)                                               | `avoid` / `harness`           | Image:                                                                                                                                                                                                                                   |         4 | 11.57s  |      9.1  |      65   | harness:prompt-template; low-draft-improvement; context-ignored         |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `avoid` / `semantic mismatch` | Title: Coastal Town Promenade with Shingle Beach and Waves Description: The image shows a row of multi-storey buildings along a pebble beachfro ... [tail] horeline, urban seaside, muted lighting, horizon view, beachfront             |       113 | 6.44s   |     58.6  |      12   | missing terms: Cars, Coastline, Deal, Kent, Landscape                   |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                            | `caveat` / `clean`            | Title: Seafront, Deal, England, UK, GBR, Europe Description: A coastal view of Deal, Kent, UK, showing the shingle beach, sea, and seafront bui ... [tail] g, Sky, Swimming, Walking, Water, Waterfront, Waves, architecture             |       109 | 6.04s   |     58.9  |       9.7 | low-draft-improvement                                                   |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `caveat` / `clean`            | Title: Seafront, Deal, England, UK, GBR, Europe. Description: A coastal view of Deal, Kent, UK, showing the shingle beach, sea, and seafront bu ... [tail] alking, Water, Waterfront, Waves, architecture.<end_of_utterance>             |        94 | 7.27s   |     32.2  |      19   | context-echo; low-draft-improvement; formatting                         |
| [`mlx-community/gemma-3n-E2B-4bit`](#model-mlx-community-gemma-3n-e2b-4bit)                                             | `avoid` / `cutoff degraded`   | - Do not go to the. - Do not go to the. - Do not go to the. - Do not go to the. - Do not go to the. - Do not go to the. - Do not go to the. - Do not go to the. - Do not go to the. - Do not go to the. - Do not go to th...             |       500 | 6.77s   |    121    |       6.1 | repetitive; low-draft-improvement; cutoff                               |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `caveat` / `clean`            | Title: Seafront buildings, shingle beach, waves, Deal Description: A coastal view showing buildings along a promenade, a shingle beach, and wav ... [tail] ront, architecture, shingle, terrace, row houses, cloudy sky, sea             |        95 | 7.70s   |     30.8  |      18   | low-draft-improvement                                                   |
| [`qnguyen3/nanoLLaVA`](#model-qnguyen3-nanollava)                                                                       | `avoid` / `cutoff degraded`   | Title: Seafront Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Dea...             |       500 | 6.06s   |    111    |       4.9 | repetitive; context-echo; low-draft-improvement                         |
| [`mlx-community/InternVL3-14B-8bit`](#model-mlx-community-internvl3-14b-8bit)                                           | `caveat` / `clean`            | Title: Coastal Town Beachfront Description: A coastal scene featuring a shingle beach with gentle waves, a row of seafront buildings, and a few ... [tail] e, Sky, Swimming, Walking, Water, Waterfront, Waves, architecture             |        93 | 7.74s   |     31.5  |      19   | low-draft-improvement                                                   |
| [`mlx-community/paligemma2-10b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-10b-ft-docci-448-bf16)               | `avoid` / `model shortcoming` | - Do not copy prompt instructions into the Title, Description, or Keywords fields.                                                                                                                                                       |        17 | 8.81s   |      5.09 |      27   | low-draft-improvement; context-ignored; missing-sections                |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `recommended` / `clean`       | Title: Seafront buildings and shingle beach in Deal, England Description: A coastal view shows a row of multi-storey buildings facing a shingle ... [tail] nade, seaside, shore, sky, water, waterfront, waves, architecture             |        95 | 9.21s   |     23.9  |      20   | low-draft-improvement                                                   |
| [`mlx-community/pixtral-12b-bf16`](#model-mlx-community-pixtral-12b-bf16)                                               | `avoid` / `semantic mismatch` | Title: Seafront Buildings and Beach Description: A row of colorful seafront buildings lines a sandy beach with gentle waves lapping at the shor ... [tail] y, gentle, partly cloudy, row, sandy, seafront, shore, sky, waves             |        89 | 9.46s   |     20.4  |      27   | low-draft-improvement                                                   |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `recommended` / `clean`       | Title: Seafront buildings and shingle beach Description: A coastal scene featuring a row of seafront buildings, a shingle beach, and the sea wi ... [tail] e, Shore, Sky, Water, Waves, architecture, shingle, partly cloudy             |        97 | 9.89s   |     29.2  |      22   | low-draft-improvement                                                   |
| [`mlx-community/GLM-4.6V-Flash-6bit`](#model-mlx-community-glm-46v-flash-6bit)                                          | `caveat` / `context budget`   | <\|begin_of_box\|>Title: Seafront with shingle beach and sea Description: A coastal scene showing a shingle beach, sea with waves, and a row of s ... [tail] itting, Sky, Water, Waterfront, Waves, architecture<\|end_of_box\|>         |        88 | 9.33s   |     56.1  |      11   | low-draft-improvement; context-budget; missing-sections                 |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)            | `avoid` / `model shortcoming` | **Title** Deal Seafront, England, UK **Description** A coastal view of Deal, Kent, UK, showing the shingle beach, sea, and seafront buildings o ... [tail] g, Sky, Swimming, Walking, Water, Waterfront, Waves, Architecture             |       102 | 9.60s   |     19.4  |      15   | low-draft-improvement; missing-sections                                 |
| [`mlx-community/gemma-3-27b-it-qat-8bit`](#model-mlx-community-gemma-3-27b-it-qat-8bit)                                 | `caveat` / `clean`            | Title: Seafront buildings, shingle beach, and waves Description: A coastal view shows a row of white and pastel-coloured buildings along a shin ... [tail] t, architecture, clouds, daylight, shingle, town, white buildings             |       105 | 11.67s  |     17.4  |      32   | low-draft-improvement                                                   |
| [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](#model-jqlive-kimi-vl-a3b-thinking-2506-6bit)                                 | `avoid` / `cutoff degraded`   | ◁think▷Got it, let's analyze the image step by step. First, the title: The image shows a coastal scene with buildings along the shore, a beach, ... [tail] of these might be inferred. Let's stick to what's clearly visible             |       500 | 11.24s  |     71    |      18   | cutoff; thinking-incomplete; missing-sections                           |
| [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](#model-mlx-community-kimi-vl-a3b-thinking-8bit)                             | `avoid` / `cutoff degraded`   | ◁think▷Okay, let's analyze this image for metadata cataloging based on the user's instructions. First, I need to focus only on what's clearly v ... [tail] draft provided. The image clearly shows a coastal scene in Deal,              |       500 | 12.80s  |     60.5  |      22   | cutoff; thinking-trace; missing-sections                                |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `avoid` / `model shortcoming` | ## Image Description: This is a wide shot of the seafront at Seafront, Deal, in Kent, UK, captured on a partly cloudy day. The scene is dominat ... [tail] sea. The scene is clearly identifiable as a British coastal town.             |       401 | 13.25s  |     41.4  |      17   | missing-sections                                                        |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `avoid` / `cutoff degraded`   | Title: Deal, Kent, UK Description: A coastal view showing a shingle beach, sea, and seafront buildings on a partly cloudy day. People are visib ... [tail] , Glance, Scan, View, Look, Gaze, Stare, Glance, Scan, View, Look             |       500 | 14.25s  |     77.1  |       8.4 | cutoff                                                                  |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                             | `avoid` / `cutoff degraded`   | - Output only the description text after the label. - Do not output the title text after the label. - Do not output the description text after ... [tail] e description text after the label. - Output only the description              |       500 | 14.48s  |     47.1  |       4.7 | repetitive; low-draft-improvement; cutoff                               |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `caveat` / `context budget`   | -                                                                                                                                                                                                                                        |         2 | 15.41s  |    179    |       8.6 | harness:long-context; low-draft-improvement; context-budget             |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `avoid` / `cutoff degraded`   | Here are my reasoning steps: We need to produce a catalog entry with Title, Description, Keywords. Use only details that are clearly visible in ... [tail] gle Beach, Buildings, Cars, Waves". That's 7 words. Or "Deal seaf             |       500 | 16.72s  |     40.7  |      15   | unverified-context-copy; metadata-borrowing; cutoff                     |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `avoid` / `cutoff degraded`   | Let's tackle this step by step. First, the user provided a prompt with a lot of repetitive and unclear terms, but the key is to focus on what's ... [tail] stade stade stade stade stade stade stade stade stade stade stade             |       500 | 21.83s  |     44.9  |      60   | repetitive; low-draft-improvement; cutoff                               |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `caveat` / `context budget`   | <\|begin_of_box\|>Title: Seafront buildings, shingle beach, and waves at Deal Description: A coastal scene with shingle beach, turbulent sea, and ... [tail] itting, Sky, Water, Waterfront, Waves, architecture<\|end_of_box\|>         |        93 | 23.21s  |     40.8  |      78   | low-draft-improvement; context-budget; missing-sections                 |
| [`meta-llama/Llama-3.2-11B-Vision-Instruct`](#model-meta-llama-llama-32-11b-vision-instruct)                            | `avoid` / `model shortcoming` | Seafront, Deal, England, UK The image depicts a seaside town with a long stretch of sandy beach and a row of buildings along the shore. The bea ... [tail] the shore. The sky is overcast, with a few birds flying overhead.             |        70 | 18.89s  |      5.09 |      25   | low-draft-improvement; missing-sections                                 |
| [`mlx-community/Molmo-7B-D-0924-8bit`](#model-mlx-community-molmo-7b-d-0924-8bit)                                       | `caveat` / `clean`            | Remove uncertain tags. Title: Shingle beach at Deal, Kent, UK Description: A partly cloudy day at Deal's shingle beach, with the town's buildin ... [tail] g, Sky, Swimming, Walking, Water, Waterfront, Waves, architecture             |       114 | 18.30s  |     46.5  |      41   | low-draft-improvement                                                   |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `caveat` / `context budget`   | <\|im_start\|> addCriterion                                                                                                                                                                                                              |         5 | 20.28s  |     68.5  |      14   | harness:long-context; low-draft-improvement; context-budget             |
| [`mlx-community/Molmo-7B-D-0924-bf16`](#model-mlx-community-molmo-7b-d-0924-bf16)                                       | `caveat` / `clean`            | Remove uncertain tags. Title: Seafront, Deal, Kent, UK, GBR, Europe Description: A coastal view of Deal, Kent, UK, showing the shingle beach, s ... [tail] r, Waterfront, Waves, architecture, English, Seafront, UK, Europe             |       173 | 23.69s  |     29.2  |      48   | low-draft-improvement                                                   |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                           | `avoid` / `semantic mismatch` | Title: Shingle Beach with Seafront Buildings and People Description: A coastal scene featuring a shingle beach with people sitting on blankets, ... [tail] UK, Landscape, Seaside, Shore, Sitting, Waterfront, Water, Waves              |       119 | 26.98s  |      6.04 |      27   | low-draft-improvement                                                   |
| [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16)                             | `caveat` / `context budget`   | -                                                                                                                                                                                                                                        |         2 | 26.13s  |    139    |       8.6 | harness:long-context; low-draft-improvement; context-budget             |
| [`mlx-community/paligemma2-3b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-3b-ft-docci-448-bf16)                 | `avoid` / `cutoff degraded`   | - Do not copy-prompt instructions into the Title, Description, or Keywords fields. - Do not infer identity, location, event, brand, species, ti ... [tail] laim that contextual facts are visually observable. - Do not copy             |       500 | 29.88s  |     18.6  |      11   | low-draft-improvement; cutoff; context-ignored                          |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `avoid` / `cutoff degraded`   | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -...             |       500 | 35.15s  |     73.6  |       8.6 | harness:long-context; repetitive; low-draft-improvement                 |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `caveat` / `clean`            | Title: Seafront, Deal, England, UK, GBR, Europe Description: A coastal view of Deal, Kent, UK, showing the shingle beach, sea, and seafront bui ... [tail] e, Sitting, Sky, Swimming, Water, Waterfront, Waves, architecture             |        95 | 58.35s  |     93.1  |      11   | low-draft-improvement                                                   |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `caveat` / `clean`            | Title: Shingle beach and seafront buildings under cloudy sky Description: A wide shingle beach stretches along the coast with small groups of p ... [tail] ore, water, architecture, England, UK, Europe, landscape, seaside             |       121 | 61.40s  |     94.5  |      26   | missing terms: Deal, Horizon, Kent, Swimming, Walking                   |
| [`mlx-community/Qwen3.5-35B-A3B-6bit`](#model-mlx-community-qwen35-35b-a3b-6bit)                                        | `recommended` / `clean`       | Title: Shingle beach and seafront buildings at Deal Description: A wide shingle beach stretches along a row of multi-story seafront buildings u ... [tail] ture, England, UK, Shore, Seaside, Landscape, Horizon, Waterfront             |       105 | 64.70s  |     89.3  |      35   | clean                                                                   |
| [`mlx-community/Qwen3.5-35B-A3B-bf16`](#model-mlx-community-qwen35-35b-a3b-bf16)                                        | `recommended` / `clean`       | Title: Shingle beach, seafront buildings, and sea at Deal Description: A wide shingle beach stretches along a row of multi-storey seafront buil ... [tail] ars, Architecture, Seaside, Shore, England, UK, Europe, Landscape             |       110 | 82.87s  |     65.7  |      76   | clean                                                                   |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `caveat` / `clean`            | Title: Shingle beach and seafront buildings in Deal, England Description: A wide expanse of brown shingle beach stretches along the coastline, ... [tail] n, Residential, Water, Landscape, UK, Europe, Shore, Architecture              |       124 | 86.20s  |     17.5  |      38   | missing terms: Kent, Promenade, Seaside, Swimming, Walking; keywords=19 |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `caveat` / `clean`            | Title: Seafront buildings and shingle beach at low tide Description: A row of multi-storey seafront buildings lines a wide shingle beach where ... [tail] Seaside, Shingle, Shore, Sky, Town, UK, Water, Waterfront, Waves               |       120 | 97.38s  |     51.4  |      76   | missing terms: Cars, Deal, Swimming, architecture, GBR; keywords=20     |
| [`mlx-community/Qwen3.5-27B-mxfp8`](#model-mlx-community-qwen35-27b-mxfp8)                                              | `caveat` / `clean`            | Title: Shingle Beach and Seafront Buildings, Deal, England Description: A coastal scene showing a wide expanse of brown shingle beach meeting c ... [tail] e, Shingle, Sky, Swimming, Water, Waterfront, Waves, architecture             |       116 | 93.21s  |     17.5  |      38   | low-draft-improvement                                                   |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                             | `avoid` / `cutoff degraded`   | -001.jpg - 19:04:26 BST - 51.226814°N, 1.401142°E - 2026-07-11 19:04:26 BST Title: Deal, Kent, UK, 2026-07-11 19:04:26 BST Keywords: - 19:04:26 ... [tail] Kent, UK, 2026-07-11 19:04:26 BST - Deal, Kent, UK, 2026-07-11 19             |       500 | 88.97s  |    110    |       5.1 | harness:long-context; repetitive; unverified-context-copy               |
| [`mlx-community/Qwen3.5-27B-4bit`](#model-mlx-community-qwen35-27b-4bit)                                                | `recommended` / `clean`       | Title: Shingle Beach and Seafront Buildings in Deal, Kent Description: A wide coastal view shows a shingle beach meeting the sea, with a row of ... [tail] omenade, Sea, Shingle, Sky, Swimming, Town, UK, Waterfront, Waves             |       112 | 103.19s |     29    |      26   | clean                                                                   |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `avoid` / `cutoff degraded`   | ◁think▷Got it, let's analyze the image step by step. First, the title: The image shows a coastal scene with buildings along a beach. The locati ... [tail] specific. So keep both. Let's list them as comma-separated. Now,              |       500 | 123.24s |      4.26 |      40   | cutoff; thinking-incomplete; missing-sections                           |
<!-- markdownlint-enable MD034 MD049 -->

## Quick Navigation

- _Successful outputs:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16),
  [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit),
  [`mlx-community/LFM2-VL-1.6B-8bit`](#model-mlx-community-lfm2-vl-16b-8bit),
  [`HuggingFaceTB/SmolVLM-Instruct`](#model-huggingfacetb-smolvlm-instruct),
  [`mlx-community/SmolVLM-Instruct-bf16`](#model-mlx-community-smolvlm-instruct-bf16),
  [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16),
  [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit),
  [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit),
  +53 more
- _Flagged outputs:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16),
  [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit),
  [`mlx-community/LFM2-VL-1.6B-8bit`](#model-mlx-community-lfm2-vl-16b-8bit),
  [`HuggingFaceTB/SmolVLM-Instruct`](#model-huggingfacetb-smolvlm-instruct),
  [`mlx-community/SmolVLM-Instruct-bf16`](#model-mlx-community-smolvlm-instruct-bf16),
  [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16),
  [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit),
  [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit),
  +47 more

## Model Gallery

Full generated output by model:

<!-- markdownlint-disable MD033 MD034 -->

<a id="model-liquidai-lfm25-vl-450m-mlx-bf16"></a>

### ✅ LiquidAI/LFM2.5-VL-450M-MLX-bf16

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ keywords=20; low-draft-improvement
- _Score:_ 🟠 D (41/100); Δ-33; Lacks visual description of image
- _Review focus:_ watchlist (worse than metadata baseline)
- _Timing:_ Load 0.35s; Gen 0.59s; Total 1.25s
- _Throughput:_ Prompt 22,396 TPS (657 tok); Gen 510 TPS (100 tok)
- _Tokens:_ prompt 657 tok; estimated text 488 tok; estimated non-text 169
  tok; generated 100 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Deal Seafront, England, UK, GBR, Europe
>
> Description: A coastal view of Deal, Kent, UK, showing the shingle beach,
> sea, and seafront buildings on a partly cloudy day.
>
> Keywords: Beach, Buildings, Cars, Coastline, Deal, Horizon, Kent, Landscape,
> People, Promenade, Seaside, Shore, Sitting, Sky, Swimming, Walking, Water,
> Waterfront, Waves, architecture
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Keyword count violation (20; expected 10-18)
- Low draft improvement

---

<a id="model-mlx-community-nanollava-15-4bit"></a>

### ✅ mlx-community/nanoLLaVA-1.5-4bit

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; missing terms: Cars, Coastline,
  Horizon, Landscape, People; low-draft-improvement
- _Score:_ ❌ F (31/100); Δ-43; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (missing sections)
- _Timing:_ Load 0.57s; Gen 0.78s; Total 1.68s
- _Throughput:_ Prompt 6,846 TPS (585 tok); Gen 291 TPS (47 tok)
- _Tokens:_ prompt 585 tok; estimated text 488 tok; estimated non-text 97 tok;
  generated 47 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Seafront, Deal, England, UK, GBR, Europe
> Description: A coastal view of Deal, Kent, UK, showing the shingle beach,
> sea, and seafront buildings on a partly cloudy day.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (keywords)
- Low draft improvement

---

<a id="model-mlx-community-lfm2-vl-16b-8bit"></a>

### ✅ mlx-community/LFM2-VL-1.6B-8bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Swimming, Walking, Water, Waterfront;
  low-draft-improvement
- _Score:_ 🟠 D (44/100); Δ-29; Lacks visual description of image
- _Review focus:_ watchlist (worse than metadata baseline)
- _Timing:_ Load 0.61s; Gen 0.89s; Total 1.85s
- _Throughput:_ Prompt 7,579 TPS (843 tok); Gen 308 TPS (88 tok)
- _Tokens:_ prompt 843 tok; estimated text 488 tok; estimated non-text 355
  tok; generated 88 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Deal Beach, Deal, Kent, UK, GBR, Europe
>
> Description: A coastal view of Deal, Kent, UK, showing the shingle beach,
> sea, and seafront buildings on a partly cloudy day.
>
> Keywords: Beach, Buildings, Coastline, Deal, Horizon, Kent, Landscape,
> People, Promenade, Seaside, Shore, Sitting, Sky, Waves, Architecture
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Low draft improvement

---

<a id="model-huggingfacetb-smolvlm-instruct"></a>

### ✅ HuggingFaceTB/SmolVLM-Instruct

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Beach, Buildings, Cars, Coastline, Horizon; low-draft-improvement
- _Score:_ ❌ F (7/100); Δ-66; Output lacks detail
- _Review focus:_ watchlist (context ignored, missing sections)
- _Timing:_ Load 0.76s; Gen 1.10s; Total 2.18s
- _Throughput:_ Prompt 4,351 TPS (1,800 tok); Gen 130 TPS (15 tok)
- _Tokens:_ prompt 1800 tok; estimated text 488 tok; estimated non-text 1312
  tok; generated 15 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Seafront, Deal, England, UK, GBR, Europe.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Beach, Buildings, Cars, Coastline, Horizon)
- Missing sections (title, description, keywords)
- Low draft improvement
- Ignores trusted hints

---

<a id="model-mlx-community-smolvlm-instruct-bf16"></a>

### ✅ mlx-community/SmolVLM-Instruct-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Beach, Buildings, Cars, Coastline, Horizon; low-draft-improvement
- _Score:_ ❌ F (7/100); Δ-66; Output lacks detail
- _Review focus:_ watchlist (context ignored, missing sections)
- _Timing:_ Load 0.67s; Gen 1.19s; Total 2.20s
- _Throughput:_ Prompt 4,265 TPS (1,800 tok); Gen 110 TPS (15 tok)
- _Tokens:_ prompt 1800 tok; estimated text 488 tok; estimated non-text 1312
  tok; generated 15 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Seafront, Deal, England, UK, GBR, Europe.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Beach, Buildings, Cars, Coastline, Horizon)
- Missing sections (title, description, keywords)
- Low draft improvement
- Ignores trusted hints

---

<a id="model-mlx-community-lfm25-vl-16b-bf16"></a>

### ✅ mlx-community/LFM2.5-VL-1.6B-bf16

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Swimming, Walking, Water, Waterfront;
  low-draft-improvement
- _Score:_ 🟡 C (62/100); Δ-11; Limited novel information
- _Review focus:_ watchlist (worse than metadata baseline)
- _Timing:_ Load 0.58s; Gen 1.21s; Total 2.15s
- _Throughput:_ Prompt 7,717 TPS (843 tok); Gen 185 TPS (125 tok)
- _Tokens:_ prompt 843 tok; estimated text 488 tok; estimated non-text 355
  tok; generated 125 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Deal, Kent, England, UK, GBR, Europe, Seaside, Beach, Shingle, Buildings,
> Promenade, Horizon, Landscape, People, Seaside, Shore, Sitting, Sky, Waves,
> Architecture
>
> &#8203;Description:
> A coastal view of Deal, Kent, UK, showing the shingle beach, sea, and
> seafront buildings on a partly cloudy day.
>
> &#8203;Keywords:
> Beach, Buildings, Cars, Coastline, Deal, Horizon, Kent, Landscape, People,
> Seaside, Shore, Sitting, Sky, Waves, Architecture
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Title length violation (20 words; expected 5-10)
- Low draft improvement

---

<a id="model-mlx-community-minicpm-v-46-8bit"></a>

### ✅ mlx-community/MiniCPM-V-4.6-8bit

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;/think&gt; appeared in generated
  text.; missing terms: Cars, Coastline, Deal, Horizon, Kent; reasoning leak;
  text-sanity=gibberish(token_noise)
- _Score:_ ✅ B (68/100); Δ-6; None identified
- _Review focus:_ watchlist (harness, reasoning leak, text sanity)
- _Timing:_ Load 1.07s; Gen 1.36s; Total 2.87s
- _Throughput:_ Prompt 4,320 TPS (1,205 tok); Gen 209 TPS (79 tok)
- _Tokens:_ prompt 1205 tok; estimated text 488 tok; estimated non-text 717
  tok; generated 79 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &lt;think&gt;
>
> &lt;/think&gt;
>
> &#8203;Title:
> Beachfront with buildings and sea
>
> &#8203;Description:
> The image shows a coastal beach with sandy areas and seaside buildings under
> a partly cloudy sky. Waves are visible near the shore, and people are
> present near the shoreline.
>
> &#8203;Keywords:
> Beach, Buildings, Sea, Sand, Coastal, Buildings, Water, People, Sky, Waves,
> Shoreline
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:stop_token
- Text sanity issue (gibberish(token_noise))
- Reasoning leak (&lt;think&gt;)
- Low draft improvement
- Unknown tags: &lt;think&gt;
- token_leak:&lt;/think&gt;

---

<a id="model-mlx-community-gemma-4-26b-a4b-it-4bit"></a>

### ✅ mlx-community/gemma-4-26b-a4b-it-4bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Deal, Kent, Promenade, Sitting;
  low-draft-improvement
- _Score:_ 🟡 C (60/100); Δ-13; Lacks visual description of image
- _Review focus:_ watchlist (worse than metadata baseline)
- _Timing:_ Load 2.67s; Gen 1.97s; Total 5.06s
- _Throughput:_ Prompt 1,564 TPS (865 tok); Gen 104 TPS (89 tok)
- _Tokens:_ prompt 865 tok; estimated text 488 tok; estimated non-text 377
  tok; generated 89 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Seafront buildings and shingle beach with waves
>
> &#8203;Description:
> A coastal scene shows a row of multi-storey buildings overlooking a shingle
> beach and the sea under a cloudy sky. White waves break on the shore as a
> few people are scattered across the beach.
>
> &#8203;Keywords:
> architecture, beach, buildings, coastline, horizon, landscape, ocean,
> people, sea, seaside, shingle, sky, waves, waterfront
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Low draft improvement

---

<a id="model-mlx-community-fastvlm-05b-bf16"></a>

### ✅ mlx-community/FastVLM-0.5B-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; missing terms: Horizon,
  Landscape, Promenade, Sitting, Swimming; low-draft-improvement
- _Score:_ 🟡 C (50/100); Δ-23; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (missing sections)
- _Timing:_ Load 0.75s; Gen 2.00s; Total 3.08s
- _Throughput:_ Prompt 5,089 TPS (589 tok); Gen 307 TPS (214 tok)
- _Tokens:_ prompt 589 tok; estimated text 488 tok; estimated non-text 101
  tok; generated 214 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Seaside View of Deal, Kent, UK
> Description: A coastal view of Deal, Kent, UK, showcasing the shingle beach,
> sea, and seafront buildings on a partly cloudy day. The image captures the
> essence of the seaside with a mix of natural and man-made elements,
> including the buildings that line the shore and the cars parked along the
> road. The beach is populated with people enjoying the sun and the sea, while
> the seafront buildings add a touch of urban charm to the scene. The colors
> are muted, with the sea and sky blending into a soft palette of blues and
> grays. The image is taken from a distance, providing a panoramic view of the
> coastline and the surrounding area. The metadata suggests that the image is
> a catalog record, with the title and description providing a clear and
> factual description of the scene. The keywords listed are supported by the
> image, and the description includes important details such as the location,
> time, and weather conditions. The metadata also includes a draft version of
> the description, which is not yet finalized.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (keywords)
- Description sentence violation (8; expected 1-2)
- Low draft improvement

---

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-6bit"></a>

### ✅ mlx-community/paligemma2-10b-ft-docci-448-6bit

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Beach, Buildings, Cars, Coastline, Deal; low-draft-improvement
- _Score:_ ❌ F (14/100); Δ-59; Output lacks detail
- _Review focus:_ watchlist (context ignored, missing sections)
- _Timing:_ Load 1.71s; Gen 2.28s; Total 4.33s
- _Throughput:_ Prompt 1,357 TPS (1,612 tok); Gen 29.6 TPS (17 tok)
- _Tokens:_ prompt 1612 tok; estimated text 488 tok; estimated non-text 1124
  tok; generated 17 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#45; Do not copy prompt instructions into the Title, Description, or Keywords
> fields.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Beach, Buildings, Cars, Coastline, Deal)
- Missing sections (title, description, keywords)
- Low draft improvement
- Ignores trusted hints

---

<a id="model-mlx-community-ministral-3-3b-instruct-2512-4bit"></a>

### ✅ mlx-community/Ministral-3-3B-Instruct-2512-4bit

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Coastline, Deal, Horizon, Kent
- _Score:_ 🏆 A (84/100); Δ+10; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 1.00s; Gen 2.39s; Total 3.73s
- _Throughput:_ Prompt 3,183 TPS (2,845 tok); Gen 119 TPS (118 tok)
- _Tokens:_ prompt 2845 tok; estimated text 488 tok; estimated non-text 2357
  tok; generated 118 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Coastal Row of Residential Buildings Beside Shingle Beach
>
> &#8203;Description:
> A coastal scene shows a row of multi-storey residential buildings adjacent
> to a pebbly shingle beach, with gentle waves rolling onto the shore. The sky
> is overcast with light diffused clouds, and a few people are scattered
> along the beach and promenade.
>
> &#8203;Keywords:
> beach, coastal buildings, cloudy sky, England, residential architecture,
> shingle shore, UK, waves, waterfront, promenade, pebbles, seaside setting,
> urban seaside, light activity
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

---

<a id="model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8"></a>

### ✅ mlx-community/diffusiongemma-26B-A4B-it-mxfp8

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title; missing terms: Sitting, Swimming,
  Walking, GBR, showing; formatting=Unknown tags: &lt;channel|&gt;;
  low-draft-improvement
- _Score:_ 🟠 D (47/100); Δ-27; Lacks visual description of image
- _Review focus:_ watchlist (missing sections)
- _Timing:_ Load 3.27s; Gen 2.56s; Total 6.17s
- _Throughput:_ Prompt 2,146 TPS (861 tok); Gen 58.9 TPS (87 tok)
- _Tokens:_ prompt 861 tok; estimated text 488 tok; estimated non-text 373
  tok; generated 87 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> <|channel>thought
> <channel|>Title:
> &#45; Shingle beach and seafront buildings in Deal, England
>
> &#8203;Description:
> &#45; A coastal view of a shingle beach with choppy waves lapping a row of
> multi-story buildings under a cloudy sky.
>
> &#8203;Keywords:
> &#45; Beach, buildings, cars, coastline, Deal, horizon, Kent, landscape, people,
> promenade, seaside, shore, sky, water, waterfront, waves, architecture
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title)
- Low draft improvement
- Unknown tags: <channel\|>

---

<a id="model-mlx-community-diffusiongemma-26b-a4b-it-8bit"></a>

### ✅ mlx-community/diffusiongemma-26B-A4B-it-8bit

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title; missing terms: Coastline, Horizon,
  Shore, Sitting, Swimming; formatting=Unknown tags: &lt;channel|&gt;;
  low-draft-improvement
- _Score:_ 🟠 D (48/100); Δ-26; Lacks visual description of image
- _Review focus:_ watchlist (missing sections)
- _Timing:_ Load 3.30s; Gen 2.74s; Total 6.40s
- _Throughput:_ Prompt 2,146 TPS (861 tok); Gen 59.7 TPS (100 tok)
- _Tokens:_ prompt 861 tok; estimated text 488 tok; estimated non-text 373
  tok; generated 100 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> <|channel>thought
> <channel|>Title:
> &#45; Seafront buildings and shingle beach in Deal England
>
> &#8203;Description:
> &#45; A coastal view of a shingle beach with waves and multi-story seafront
> buildings under a cloudy sky. Several people are visible on the beach and
> cars are parked along the promenade.
>
> &#8203;Keywords:
> &#45; architecture, beach, buildings, cars, coastal Deal, England, Kent,
> landscape, people, promenade, seaside, sea, shingle, sky, water, waterfront,
> waves
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title)
- Low draft improvement
- Unknown tags: <channel\|>

---

<a id="model-mlx-community-smolvlm2-22b-instruct-mlx"></a>

### ✅ mlx-community/SmolVLM2-2.2B-Instruct-mlx

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ keyword duplication=40%; unverified-context-copy
- _Score:_ ✅ B (68/100); Δ-6; None identified
- _Review focus:_ watchlist (unverified context copy)
- _Timing:_ Load 0.68s; Gen 3.38s; Total 4.41s
- _Throughput:_ Prompt 1,921 TPS (701 tok); Gen 130 TPS (308 tok)
- _Tokens:_ prompt 701 tok; estimated text 488 tok; estimated non-text 213
  tok; generated 308 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Seafront, Deal, England, UK, GBR, Europe
> Description: A coastal view of Deal, Kent, UK, showing the shingle beach,
> sea, and seafront buildings on a partly cloudy day.
> Keywords: Beach, Buildings, Cars, Coastline, Deal, Horizon, Kent, Landscape,
> People, Promenade, Seaside, Shore, Sitting, Sky, Swimming, Walking, Water,
> Waterfront, Waves, architecture
>
> Context: Location terms: England, UK, Town, Deal, GBR, Europe
> Capture date/time: 2026-07-11 19:04:26 BST 19:04:26
> GPS: 51.226814°N, 1.401142°E
>
> &#8203;Draft descriptive metadata:
> &#45; Existing title: Seafront, Deal, England, UK, GBR, Europe
> &#45; Existing description: A coastal view of Deal, Kent, UK, showing the
> shingle beach, sea, and seafront buildings on a partly cloudy day.
> &#45; Existing keywords: Beach, Buildings, Cars, Coastline, Deal, Horizon, Kent,
> Landscape, People, Promenade, Seaside, Shore, Sitting, Sky, Swimming,
> Walking, Water, Waterfront, Waves, architecture
> &#45; Treat this draft as fallible. Retain supported details, correct errors,
> and add important visible information.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Keyword count violation (65; expected 10-18)
- Keyword duplication (40% duplicated terms)
- Unverified context copy
- suspicious_precision: 2 overly precise numbers

---

<a id="model-microsoft-phi-35-vision-instruct"></a>

### ✅ microsoft/Phi-3.5-vision-instruct

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Swimming, GBR, showing; keywords=22;
  low-draft-improvement
- _Score:_ 🟡 C (64/100); Δ-10; None identified
- _Review focus:_ watchlist (worse than metadata baseline)
- _Timing:_ Load 1.01s; Gen 3.73s; Total 5.07s
- _Throughput:_ Prompt 3,853 TPS (1,424 tok); Gen 55.6 TPS (162 tok)
- _Tokens:_ prompt 1424 tok; estimated text 488 tok; estimated non-text 936
  tok; generated 162 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Coastal Scene at Deal Beach
>
> Description: A coastal view of Deal, Kent, UK, featuring a shingle beach,
> sea, and seafront buildings on a partly cloudy day. The beach is populated
> with a few individuals, some are sitting and others are walking along the
> shore. The sea is moderately rough with visible waves crashing onto the
> shore. The sky is partly cloudy with no signs of precipitation.
>
> Keywords: Deal, Beach, Shingle, Sea, Seafront, Buildings, Cars, Coastline,
> Horizon, Kent, Landscape, People, Promenade, Seaside, Shore, Waves,
> Architecture, Walking, Sitting, Waterfront, Water, Sky
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Description sentence violation (4; expected 1-2)
- Keyword count violation (22; expected 10-18)
- Low draft improvement

---

<a id="model-mlx-community-phi-35-vision-instruct-bf16"></a>

### ✅ mlx-community/Phi-3.5-vision-instruct-bf16

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Swimming, GBR, showing; keywords=22;
  low-draft-improvement
- _Score:_ 🟡 C (64/100); Δ-10; None identified
- _Review focus:_ watchlist (worse than metadata baseline)
- _Timing:_ Load 0.99s; Gen 3.94s; Total 5.48s
- _Throughput:_ Prompt 3,677 TPS (1,424 tok); Gen 52.8 TPS (162 tok)
- _Tokens:_ prompt 1424 tok; estimated text 488 tok; estimated non-text 936
  tok; generated 162 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Coastal Scene at Deal Beach
>
> Description: A coastal view of Deal, Kent, UK, featuring a shingle beach,
> sea, and seafront buildings on a partly cloudy day. The beach is populated
> with a few individuals, some are sitting and others are walking along the
> shore. The sea is moderately rough with visible waves crashing onto the
> shore. The sky is partly cloudy with no signs of precipitation.
>
> Keywords: Deal, Beach, Shingle, Sea, Seafront, Buildings, Cars, Coastline,
> Horizon, Kent, Landscape, People, Promenade, Seaside, Shore, Waves,
> Architecture, Walking, Sitting, Waterfront, Water, Sky
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Description sentence violation (4; expected 1-2)
- Keyword count violation (22; expected 10-18)
- Low draft improvement

---

<a id="model-mlx-community-internvl3-8b-bf16"></a>

### ✅ mlx-community/InternVL3-8B-bf16

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Deal, Kent, Sitting, Swimming, Seafront;
  low-draft-improvement
- _Score:_ 🟠 D (49/100); Δ-24; Lacks visual description of image
- _Review focus:_ watchlist (worse than metadata baseline)
- _Timing:_ Load 1.66s; Gen 3.95s; Total 5.94s
- _Throughput:_ Prompt 2,698 TPS (2,880 tok); Gen 34.3 TPS (79 tok)
- _Tokens:_ prompt 2880 tok; estimated text 488 tok; estimated non-text 2392
  tok; generated 79 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Beachfront with Sea and Buildings
>
> &#8203;Description:
> A coastal scene featuring a pebble beach, sea with waves, and a row of
> buildings along the promenade on a partly cloudy day.
>
> &#8203;Keywords:
> Beach, Buildings, Cars, Coastline, Horizon, Landscape, People, Promenade,
> Seaside, Shore, Sky, Walking, Water, Waterfront, Waves, architecture
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Low draft improvement

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-nvfp4"></a>

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-nvfp4

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Coastline, Deal, Kent, Landscape, Promenade
- _Score:_ 🏆 A (88/100); Δ+14; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 1.43s; Gen 4.39s; Total 6.16s
- _Throughput:_ Prompt 1,411 TPS (2,846 tok); Gen 62.8 TPS (118 tok)
- _Tokens:_ prompt 2846 tok; estimated text 488 tok; estimated non-text 2358
  tok; generated 118 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;&#42;&#42;Title:&#42;&#42;
> &#42;Seafront buildings along sandy beach with waves and distant horizon&#42;
>
> &#8203;&#42;&#42;Description:&#42;&#42;
> A coastal scene featuring a row of traditional townhouses along a sandy
> shingle beach, with waves crashing onto shore. People sit on deck chairs
> near the waterfront, and parked cars line the road behind the buildings
> under a partly cloudy sky.
>
> &#8203;&#42;&#42;Keywords:&#42;&#42;
> Seafront, sandy beach, shingle, waves, townhouses, row of buildings, parked
> cars, road, people sitting, deck chairs, partly cloudy sky, waterfront,
> distant horizon, coastal scene, England
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

---

<a id="model-mlx-community-pixtral-12b-8bit"></a>

### ✅ mlx-community/pixtral-12b-8bit

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Deal, Horizon, Kent, Landscape;
  keywords=19; low-draft-improvement
- _Score:_ 🟠 D (44/100); Δ-30; Lacks visual description of image
- _Review focus:_ watchlist (worse than metadata baseline)
- _Timing:_ Load 1.79s; Gen 4.46s; Total 6.60s
- _Throughput:_ Prompt 1,842 TPS (2,917 tok); Gen 38.9 TPS (93 tok)
- _Tokens:_ prompt 2917 tok; estimated text 488 tok; estimated non-text 2429
  tok; generated 93 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Seafront Buildings and Beach
>
> &#8203;Description:
> A coastal scene featuring a row of colorful seafront buildings along a sandy
> beach with gentle waves lapping the shore under a partly cloudy sky.
>
> &#8203;Keywords:
> Beach, Buildings, Coastline, Promenade, Sand, Sea, Seafront, Shingle, Sky,
> Waves, architecture, cloudy, gentle, partly cloudy, row, sandy, shore, sky,
> waterfront
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Title length violation (4 words; expected 5-10)
- Keyword count violation (19; expected 10-18)
- Low draft improvement

---

<a id="model-mlx-community-gemma-4-31b-bf16"></a>

### ✅ mlx-community/gemma-4-31b-bf16

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output appears truncated to about 4 tokens.; missing terms:
  Beach, Buildings, Cars, Coastline, Deal; low-draft-improvement
- _Score:_ ❌ F (0/100); Δ-74; Empty or minimal output
- _Review focus:_ watchlist (context ignored, harness)
- _Timing:_ Load 6.63s; Gen 4.58s; Total 11.57s
- _Throughput:_ Prompt 238 TPS (853 tok); Gen 9.1 TPS (4 tok)
- _Tokens:_ prompt 853 tok; estimated text 488 tok; estimated non-text 365
  tok; generated 4 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
>
>
> &#8203;Image:
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:prompt_template
- Context ignored (missing: Beach, Buildings, Cars, Coastline, Deal)
- Low draft improvement
- Ignores trusted hints
- output:truncated(4tok)

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-mxfp4"></a>

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-mxfp4

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Coastline, Deal, Kent, Landscape
- _Score:_ ✅ B (75/100); Δ+2; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 1.42s; Gen 4.64s; Total 6.44s
- _Throughput:_ Prompt 1,288 TPS (2,846 tok); Gen 58.6 TPS (113 tok)
- _Tokens:_ prompt 2846 tok; estimated text 488 tok; estimated non-text 2358
  tok; generated 113 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Coastal Town Promenade with Shingle Beach and Waves
>
> &#8203;Description:
> The image shows a row of multi-storey buildings along a pebble beachfront,
> with the sea waves crashing against the shore. A few people are visible near
> the water, and the sky appears overcast with muted lighting.
>
> &#8203;Keywords:
> Coastal buildings, pebble beach, sea waves, promenade, overcast sky, people
> near water, multi-storey architecture, shoreline, urban seaside, muted
> lighting, horizon view, beachfront
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

---

<a id="model-mlx-community-llava-v16-mistral-7b-8bit"></a>

### ✅ mlx-community/llava-v1.6-mistral-7b-8bit

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ keywords=20; low-draft-improvement
- _Score:_ 🟠 D (41/100); Δ-33; Lacks visual description of image
- _Review focus:_ watchlist (worse than metadata baseline)
- _Timing:_ Load 0.96s; Gen 4.75s; Total 6.04s
- _Throughput:_ Prompt 1,157 TPS (2,608 tok); Gen 58.9 TPS (109 tok)
- _Tokens:_ prompt 2608 tok; estimated text 488 tok; estimated non-text 2120
  tok; generated 109 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Seafront, Deal, England, UK, GBR, Europe
>
> Description: A coastal view of Deal, Kent, UK, showing the shingle beach,
> sea, and seafront buildings on a partly cloudy day.
>
> Keywords: Beach, Buildings, Cars, Coastline, Deal, Horizon, Kent, Landscape,
> People, Promenade, Seaside, Shore, Sitting, Sky, Swimming, Walking, Water,
> Waterfront, Waves, architecture
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Keyword count violation (20; expected 10-18)
- Low draft improvement

---

<a id="model-mlx-community-idefics3-8b-llama3-bf16"></a>

### ✅ mlx-community/Idefics3-8B-Llama3-bf16

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ keywords=20; context echo=60%; formatting=Unknown tags:
  &lt;end_of_utterance&gt;; low-draft-improvement
- _Score:_ 🟠 D (41/100); Δ-33; Lacks visual description of image
- _Review focus:_ watchlist (context echo)
- _Timing:_ Load 2.01s; Gen 4.88s; Total 7.27s
- _Throughput:_ Prompt 2,194 TPS (2,866 tok); Gen 32.2 TPS (94 tok)
- _Tokens:_ prompt 2866 tok; estimated text 488 tok; estimated non-text 2378
  tok; generated 94 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Seafront, Deal, England, UK, GBR, Europe.
>
> Description: A coastal view of Deal, Kent, UK, showing the shingle beach,
> sea, and seafront buildings on a partly cloudy day.
>
> Keywords: Beach, Buildings, Cars, Coastline, Deal, Horizon, Kent, Landscape,
> People, Promenade, Seaside, Shore, Sitting, Sky, Swimming, Walking, Water,
> Waterfront, Waves, architecture.<end_of_utterance>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Keyword count violation (20; expected 10-18)
- Context echo (60% overlap)
- Low draft improvement
- Unknown tags: <end_of_utterance>

---

<a id="model-mlx-community-gemma-3n-e2b-4bit"></a>

### ✅ mlx-community/gemma-3n-E2B-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Beach, Buildings, Cars, Coastline, Deal; excessive
  bullets=63
- _Score:_ 🟠 D (42/100); Δ-31; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, generation loop, missing
  sections, repetitive)
- _Timing:_ Load 1.51s; Gen 4.93s; Total 6.77s
- _Throughput:_ Prompt 2,680 TPS (847 tok); Gen 121 TPS (500 tok)
- _Tokens:_ prompt 847 tok; estimated text 488 tok; estimated non-text 359
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
>
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not go to the.
> &#45; Do not
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "- do not go...")
- Excessive bullet points (63)
- Context ignored (missing: Beach, Buildings, Cars, Coastline, Deal)
- Generation loop (repetitive_tail)
- Missing sections (title, description, keywords)
- Low draft improvement
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-gemma-3-27b-it-qat-4bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-4bit

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Kent, Sitting, Swimming, Walking;
  keywords=20; low-draft-improvement
- _Score:_ 🟠 D (46/100); Δ-28; Lacks visual description of image
- _Review focus:_ watchlist (worse than metadata baseline)
- _Timing:_ Load 2.32s; Gen 5.04s; Total 7.70s
- _Throughput:_ Prompt 655 TPS (856 tok); Gen 30.8 TPS (95 tok)
- _Tokens:_ prompt 856 tok; estimated text 488 tok; estimated non-text 368
  tok; generated 95 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Seafront buildings, shingle beach, waves, Deal
>
> &#8203;Description:
> A coastal view showing buildings along a promenade, a shingle beach, and
> waves breaking on the shore. Several people are visible on the beach.
>
> &#8203;Keywords:
> Beach, Buildings, Coastline, Deal, Horizon, Landscape, People, Promenade,
> Seaside, Shore, Sky, Waves, Water, Waterfront, architecture, shingle,
> terrace, row houses, cloudy sky, sea
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Keyword count violation (20; expected 10-18)
- Low draft improvement

---

<a id="model-qnguyen3-nanollava"></a>

### ✅ qnguyen3/nanoLLaVA

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: description, keywords;
  missing terms: Beach, Buildings, Cars, Coastline, Horizon; context echo=100%
- _Score:_ ❌ F (25/100); Δ-49; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context echo, context ignored, cutoff, generation
  loop, missing sections, repetitive)
- _Timing:_ Load 0.58s; Gen 5.15s; Total 6.06s
- _Throughput:_ Prompt 5,723 TPS (585 tok); Gen 111 TPS (500 tok)
- _Tokens:_ prompt 585 tok; estimated text 488 tok; estimated non-text 97 tok;
  generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Seafront Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal,
> Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal,
> Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal,
> Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal,
> Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal,
> Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal,
> Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal,
> Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal,
> Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal,
> Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal,
> Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal,
> Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal,
> Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal,
> Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal,
> Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal,
> Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal,
> Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal,
> Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal,
> Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal,
> Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal,
> Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal, Deal,
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (Deal,)
- Context ignored (missing: Beach, Buildings, Cars, Coastline, Horizon)
- Generation loop (repetitive_tail)
- Missing sections (description, keywords)
- Title length violation (249 words; expected 5-10)
- Context echo (100% overlap)
- Low draft improvement
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-internvl3-14b-8bit"></a>

### ✅ mlx-community/InternVL3-14B-8bit

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Deal, Kent, Sitting, GBR, view;
  low-draft-improvement
- _Score:_ 🟡 C (58/100); Δ-15; None identified
- _Review focus:_ watchlist (worse than metadata baseline)
- _Timing:_ Load 1.71s; Gen 5.69s; Total 7.74s
- _Throughput:_ Prompt 1,336 TPS (2,880 tok); Gen 31.5 TPS (93 tok)
- _Tokens:_ prompt 2880 tok; estimated text 488 tok; estimated non-text 2392
  tok; generated 93 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Coastal Town Beachfront
>
> &#8203;Description:
> A coastal scene featuring a shingle beach with gentle waves, a row of
> seafront buildings, and a few people enjoying the beach. The sky is partly
> cloudy, and the water appears slightly choppy.
>
> &#8203;Keywords:
> Beach, Buildings, Cars, Coastline, Horizon, Landscape, People, Promenade,
> Seaside, Shore, Sky, Swimming, Walking, Water, Waterfront, Waves,
> architecture
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Title length violation (3 words; expected 5-10)
- Low draft improvement

---

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-bf16"></a>

### ✅ mlx-community/paligemma2-10b-ft-docci-448-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Beach, Buildings, Cars, Coastline, Deal; low-draft-improvement
- _Score:_ ❌ F (14/100); Δ-59; Output lacks detail
- _Review focus:_ watchlist (context ignored, missing sections)
- _Timing:_ Load 2.71s; Gen 5.76s; Total 8.81s
- _Throughput:_ Prompt 863 TPS (1,612 tok); Gen 5.09 TPS (17 tok)
- _Tokens:_ prompt 1612 tok; estimated text 488 tok; estimated non-text 1124
  tok; generated 17 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#45; Do not copy prompt instructions into the Title, Description, or Keywords
> fields.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Beach, Buildings, Cars, Coastline, Deal)
- Missing sections (title, description, keywords)
- Low draft improvement
- Ignores trusted hints

---

<a id="model-mlx-community-gemma-4-31b-it-4bit"></a>

### ✅ mlx-community/gemma-4-31b-it-4bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Sitting, Swimming, Walking, GBR, showing;
  low-draft-improvement
- _Score:_ 🟡 C (53/100); Δ-21; Lacks visual description of image
- _Review focus:_ watchlist (worse than metadata baseline)
- _Timing:_ Load 2.74s; Gen 6.12s; Total 9.21s
- _Throughput:_ Prompt 532 TPS (865 tok); Gen 23.9 TPS (95 tok)
- _Tokens:_ prompt 865 tok; estimated text 488 tok; estimated non-text 377
  tok; generated 95 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Seafront buildings and shingle beach in Deal, England
>
> Description: A coastal view shows a row of multi-storey buildings facing a
> shingle beach and the sea under a cloudy sky. Several people are scattered
> across the beach, and cars are parked along the promenade.
>
> Keywords: beach, buildings, cars, coastline, Deal, England, horizon, Kent,
> landscape, people, promenade, seaside, shore, sky, water, waterfront, waves,
> architecture
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Low draft improvement

---

<a id="model-mlx-community-pixtral-12b-bf16"></a>

### ✅ mlx-community/pixtral-12b-bf16

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Deal, Horizon, Kent, Landscape;
  keywords=19; low-draft-improvement
- _Score:_ 🟠 D (43/100); Δ-31; Lacks visual description of image
- _Review focus:_ watchlist (worse than metadata baseline)
- _Timing:_ Load 2.56s; Gen 6.56s; Total 9.46s
- _Throughput:_ Prompt 1,684 TPS (2,917 tok); Gen 20.4 TPS (89 tok)
- _Tokens:_ prompt 2917 tok; estimated text 488 tok; estimated non-text 2429
  tok; generated 89 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Seafront Buildings and Beach
>
> &#8203;Description:
> A row of colorful seafront buildings lines a sandy beach with gentle waves
> lapping at the shore under a partly cloudy sky.
>
> &#8203;Keywords:
> Beach, Buildings, Coastline, Promenade, Sand, Sea, Shingle, Sky, Waves,
> architecture, cloudy, gentle, partly cloudy, row, sandy, seafront, shore,
> sky, waves
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Title length violation (4 words; expected 5-10)
- Keyword count violation (19; expected 10-18)
- Low draft improvement

---

<a id="model-mlx-community-devstral-small-2-24b-instruct-2512-5bit"></a>

### ✅ mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Deal, Kent, Sitting, Swimming, Walking;
  low-draft-improvement
- _Score:_ 🟡 C (51/100); Δ-23; Lacks visual description of image
- _Review focus:_ watchlist (worse than metadata baseline)
- _Timing:_ Load 2.26s; Gen 7.22s; Total 9.89s
- _Throughput:_ Prompt 696 TPS (2,313 tok); Gen 29.2 TPS (97 tok)
- _Tokens:_ prompt 2313 tok; estimated text 488 tok; estimated non-text 1825
  tok; generated 97 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Seafront buildings and shingle beach
>
> &#8203;Description:
> A coastal scene featuring a row of seafront buildings, a shingle beach, and
> the sea with gentle waves. The sky is partly cloudy, and a few people are
> visible on the beach.
>
> &#8203;Keywords:
> Beach, Buildings, Cars, Coastline, Horizon, Landscape, People, Promenade,
> Seaside, Shore, Sky, Water, Waves, architecture, shingle, partly cloudy
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Low draft improvement

---

<a id="model-mlx-community-glm-46v-flash-6bit"></a>

### ✅ mlx-community/GLM-4.6V-Flash-6bit

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a visual input burden issue first; reduce that
  input load or inspect long-context handling before judging output quality.
- _Key signals:_ output/prompt=1.33%; visual input burden=93%; missing
  sections: title; missing terms: Cars, Horizon, Swimming, Walking, GBR
- _Score:_ 🟡 C (51/100); Δ-22; Limited novel information
- _Review focus:_ watchlist (hallucination, missing sections)
- _Timing:_ Load 1.62s; Gen 7.37s; Total 9.33s
- _Throughput:_ Prompt 1,232 TPS (6,593 tok); Gen 56.1 TPS (88 tok)
- _Tokens:_ prompt 6593 tok; estimated text 488 tok; estimated non-text 6105
  tok; generated 88 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> <|begin_of_box|>Title:
> Seafront with shingle beach and sea
>
> &#8203;Description:
> A coastal scene showing a shingle beach, sea with waves, and a row of
> seafront buildings. People are visible on the beach. The sky is overcast.
>
> &#8203;Keywords:
> Beach, Buildings, Coastline, Deal, Kent, Landscape, People, Promenade,
> Seaside, Shore, Sitting, Sky, Water, Waterfront, Waves,
> architecture<|end_of_box|>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context_budget
- Missing sections (title)
- Description sentence violation (3; expected 1-2)
- Low draft improvement
- Contains unexpected table

---

<a id="model-mlx-community-llama-32-11b-vision-instruct-8bit"></a>

### ✅ mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: GBR; low-draft-improvement
- _Score:_ 🟠 D (49/100); Δ-24; Missing requested structure
- _Review focus:_ watchlist (missing sections)
- _Timing:_ Load 1.63s; Gen 7.61s; Total 9.60s
- _Throughput:_ Prompt 320 TPS (555 tok); Gen 19.4 TPS (102 tok)
- _Tokens:_ prompt 555 tok; estimated text 488 tok; estimated non-text 67 tok;
  generated 102 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#42;&#42;Title&#42;&#42;
> Deal Seafront, England, UK
>
> &#42;&#42;Description&#42;&#42;
> A coastal view of Deal, Kent, UK, showing the shingle beach, sea, and
> seafront buildings on a partly cloudy day. The beach is empty, with a few
> people walking along the shoreline.
>
> &#42;&#42;Keywords&#42;&#42;
> Beach, Buildings, Cars, Coastline, Deal, Horizon, Landscape, People,
> Promenade, Seaside, Shore, Sitting, Sky, Swimming, Walking, Water,
> Waterfront, Waves, Architecture
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)
- Low draft improvement

---

<a id="model-mlx-community-gemma-3-27b-it-qat-8bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-8bit

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Deal, Kent, Sitting, Swimming;
  keywords=19; low-draft-improvement
- _Score:_ 🟡 C (60/100); Δ-14; None identified
- _Review focus:_ watchlist (worse than metadata baseline)
- _Timing:_ Load 3.35s; Gen 7.99s; Total 11.67s
- _Throughput:_ Prompt 581 TPS (856 tok); Gen 17.4 TPS (105 tok)
- _Tokens:_ prompt 856 tok; estimated text 488 tok; estimated non-text 368
  tok; generated 105 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Seafront buildings, shingle beach, and waves
>
> &#8203;Description:
> A coastal view shows a row of white and pastel-coloured buildings along a
> shingle beach, with waves breaking on the shore. Several people are visible
> on the beach, and a road runs parallel to the buildings.
>
> &#8203;Keywords:
> Beach, Buildings, Coastline, Horizon, Landscape, People, Promenade, Seaside,
> Shore, Sky, Waves, Water, Waterfront, architecture, clouds, daylight,
> shingle, town, white buildings
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Keyword count violation (19; expected 10-18)
- Low draft improvement

---

<a id="model-jqlive-kimi-vl-a3b-thinking-2506-6bit"></a>

### ✅ jqlive/Kimi-VL-A3B-Thinking-2506-6bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title; missing terms:
  showing, day; keywords=61
- _Score:_ ✅ B (76/100); Δ+2; Missing requested structure
- _Review focus:_ watchlist (cutoff, missing sections, thinking incomplete)
- _Timing:_ Load 2.07s; Gen 8.82s; Total 11.24s
- _Throughput:_ Prompt 1,332 TPS (1,571 tok); Gen 71.0 TPS (500 tok)
- _Tokens:_ prompt 1571 tok; estimated text 488 tok; estimated non-text 1083
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ◁think▷Got it, let's analyze the image step by step.
>
> First, the title: The image shows a coastal scene with buildings along the
> shore, a beach, and the sea. The location is Deal, Kent, UK. So the title
> should be concise. Let's check the existing draft title: "Seafront, Deal,
> England, UK, GBR, Europe" – that's good. But maybe make it more specific.
> The image has a shingle beach, buildings, sea, and partly cloudy sky. So the
> title could be "Shingle beach and seafront buildings at Deal, Kent, UK".
>
> Description: The main subject is the coastal scene. Visible elements:
> shingle beach, sea with waves, buildings (hotels or residential) along the
> shore, people on the beach (some sitting, some walking), partly cloudy sky.
> So the description should be factual. Let's structure it: "A coastal scene
> featuring a shingle beach, turbulent sea with waves, and a row of buildings
> along the shore under a partly cloudy sky, with people walking and sitting
> on the beach."
>
> Keywords: Need 10-18 unique terms. Visible elements: shingle beach, seafront
> buildings, Deal (location), sea (with waves), people (walking, sitting),
> partly cloudy sky, buildings (architecture), cars (parked along the shore),
> horizon (distant view). Let's list them: Shingle beach, Seafront buildings,
> Deal, Sea (with waves), People (walking, sitting), Partly cloudy sky,
> Buildings (architecture), Cars (parked), Horizon (distant view), Turbulent
> sea, Shoreline.
>
> Wait, check if all are visible. The image has parked cars along the shore,
> people on the beach (some walking, some sitting), buildings (hotels or
> &#8203;residential), shingle beach, sea with waves, partly cloudy sky. So keywords:
> Shingle beach, Seafront buildings, Deal, Sea (with waves), People (walking,
> sitting), Partly cloudy sky, Buildings (architecture), Cars (parked),
> Horizon (distant view), Turbulent sea, Shoreline.
>
> Now, check against the draft metadata. The existing draft keywords include
> Beach, Buildings, Cars, Coastline, Deal, Horizon, Kent, Landscape, People,
> Promenade, Seaside, Shore, Sitting, Sky, Swimming, Walking, Water,
> Waterfront, Waves, architecture. But some of these might be inferred. Let's
> stick to what's clearly visible
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title)
- Description sentence violation (4; expected 1-2)
- Keyword count violation (61; expected 10-18)
- Thinking trace incomplete (expected model protocol; opened with ◁think▷)
- Likely capped by max token budget

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-8bit"></a>

### ✅ mlx-community/Kimi-VL-A3B-Thinking-8bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title; missing terms:
  Water, showing; thinking trace present
- _Score:_ 🏆 A (81/100); Δ+7; None identified
- _Review focus:_ watchlist (cutoff, missing sections)
- _Timing:_ Load 2.51s; Gen 9.93s; Total 12.80s
- _Throughput:_ Prompt 1,582 TPS (1,571 tok); Gen 60.5 TPS (500 tok)
- _Tokens:_ prompt 1571 tok; estimated text 488 tok; estimated non-text 1083
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ◁think▷Okay, let's analyze this image for metadata cataloging based on the
> user's instructions. First, I need to focus only on what's clearly visible
> and avoid any assumptions or inferences. The image shows a coastal scene
> with buildings along a beach. The location is Deal, Kent, UK, as per the
> context provided. The main elements I can confirm are the shingle beach,
> sea, seafront buildings, and people walking along the promenade. The sky is
> partly cloudy, and there are waves crashing onto the shore. The image
> includes cars parked along the seafront and some people sitting on the
> beach. The keywords provided in the draft mention Deal, Kent, UK, Beach,
> Buildings, Coastline, Horizon, Landscape, People, Promenade, Seaside, Shore,
> Sky, Waterfront, Waves, architecture. I should check if all these are
> visible. The existing draft title is "Seafront, Deal, England, UK, GBR,
> Europe," which seems accurate. The description should focus on the visible
> elements: coastal view of Deal, Kent, UK, with shingle beach, seafront
> buildings, partly cloudy sky, people walking and sitting, waves, and the
> promenade. Keywords should include Deal, Kent, UK, Beach, Buildings,
> Coastline, Horizon, Landscape, People, Promenade, Seaside, Shore, Sky,
> Waterfront, Waves, architecture. I need to exclude any uncertain elements
> like "Swimming" since there's no clear evidence of swimming. Also, the
> existing draft's title and description might need slight adjustments based
> on the image details. The GPS coordinates are provided but not visible in
> the image, so I shouldn't include them in the metadata. The date and time
> are also not visible, so omit those. Make sure to correct any discrepancies
> between the draft and the image, like ensuring Deal is included in the title
> and description.◁/think▷Title: Seafront, Deal, Kent, UK
> Description: A coastal view of Deal, Kent, UK, featuring a shingle beach,
> seafront buildings, and people walking along the promenade on a partly
> cloudy day.
> Keywords: Deal, Kent, UK, Beach, Buildings, Coastline, Horizon, Landscape,
> People, Promenade, Seaside, Shore, Sky, Waterfront, Waves,
> architecture<|im_assistant|><|im_user|>,<|im_assistant|><|im_user|>=◁/think▷I
> analyzed the image and the metadata draft provided. The image clearly shows
> a coastal scene in Deal,
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Excessive verbosity
- Missing sections (title)
- Thinking trace present (expected model protocol; ◁think▷)
- Likely capped by max token budget

---

<a id="model-mlx-community-gemma-3n-e4b-it-bf16"></a>

### ✅ mlx-community/gemma-3n-E4B-it-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Landscape, Swimming, Waterfront, architecture, GBR
- _Score:_ ✅ B (76/100); Δ+2; Missing requested structure
- _Review focus:_ watchlist (missing sections)
- _Timing:_ Load 2.30s; Gen 10.61s; Total 13.25s
- _Throughput:_ Prompt 1,907 TPS (855 tok); Gen 41.4 TPS (401 tok)
- _Tokens:_ prompt 855 tok; estimated text 488 tok; estimated non-text 367
  tok; generated 401 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#35;# Image Description:
>
> This is a wide shot of the seafront at Seafront, Deal, in Kent, UK, captured
> on a partly cloudy day. The scene is dominated by a long, sloping shingle
> beach that stretches from the foreground towards a row of buildings lining
> the coastline. The sea occupies the right side of the frame, with gentle
> waves washing onto the shore.
>
> &#42;&#42;Foreground:&#42;&#42; The shingle beach is the most prominent feature in the
> foreground. The pebbles are a mix of brown and grey tones, and the texture
> is clearly visible. Several people are scattered along the beach, some
> walking, some sitting, and one person appears to be playing with a small
> object. A few small, dark objects, possibly beach toys, are also visible.
>
> &#42;&#42;Midground:&#42;&#42; A row of buildings lines the top of the shingle beach. These
> are predominantly two to three stories high, with a mix of architectural
> styles. Many have pitched roofs and are painted in light colours like white
> and cream, with some darker brickwork. Some buildings have scaffolding
> erected on them, suggesting ongoing maintenance or construction. Several
> parked cars are visible along the road that runs parallel to the buildings.
> A few people are walking along the promenade in front of the buildings.
>
> &#42;&#42;Background:&#42;&#42; The sea stretches out to the horizon, with gentle waves
> breaking near the shore. The water is a muted grey-green colour. The sky is
> overcast with a mix of light grey and white clouds, suggesting a diffused
> light. In the far distance, a faint outline of land or possibly another
> coastline can be seen under the hazy sky. A few birds are flying in the sky.
>
> &#42;&#42;Overall Impression:&#42;&#42; The image conveys a typical coastal scene on a calm
> day. The atmosphere is relaxed and peaceful, with people enjoying the
> seaside. The colours are muted and natural, reflecting the overcast weather.
> The composition is balanced, with the beach leading the eye towards the
> buildings and the sea. The scene is clearly identifiable as a British
> coastal town.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)

---

<a id="model-mlx-community-glm-46v-flash-mxfp4"></a>

### ✅ mlx-community/GLM-4.6V-Flash-mxfp4

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ hit token cap (500); missing terms: GBR; keywords=196
- _Score:_ 🏆 A (81/100); Δ+8; None identified
- _Review focus:_ watchlist (cutoff)
- _Timing:_ Load 1.37s; Gen 12.54s; Total 14.25s
- _Throughput:_ Prompt 1,178 TPS (6,593 tok); Gen 77.1 TPS (500 tok)
- _Tokens:_ prompt 6593 tok; estimated text 488 tok; estimated non-text 6105
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Deal, Kent, UK
>
> &#8203;Description:
> A coastal view showing a shingle beach, sea, and seafront buildings on a
> partly cloudy day. People are visible on the beach.
>
> &#8203;Keywords:
> Deal, Kent, UK, England, Europe, Beach, Sea, Seafront, Buildings, Shingle
> Beach, Partly Cloudy, People, Beachgoers, Waves, Water, Shore, Promenade,
> Coastline, Horizon, Sky, Clouds, Cloudy, Overcast, Coastal, Seaside,
> Shoreline, Beachfront, Waterfront, Architecture, Urban, Town, City,
> Residential, Commercial, Mixed-Use, Multi-Storey, Low-Storey, Single-Storey,
> Chimneys, Roofs, Windows, Doors, Parking, Cars, Vehicles, Street, Road,
> Path, Walkway, Promenade, Boardwalk, Beach Access, Beachgoer, Visitor,
> Tourist, Leisure, Recreation, Relaxation, Sunbathing, Swimming, Walking,
> Sitting, Standing, Playing, Enjoying, Enjoyment, Pleasure, Fun,
> Entertainment, Activity, Event, Festival, Holiday, Season, Weather, Climate,
> Temperature, Humidity, Wind, Rain, Snow, Cloud, Sky, Atmosphere,
> Environment, Nature, Landscape, Topography, Geography, Map, Coordinates,
> Latitude, Longitude, Altitude, Elevation, Terrain, Land, Soil, Rock, Stone,
> Sand, Gravel, Pebble, Shell, Coral, Rock, Cliff, Hill, Mountain, Valley,
> Plain, Basin, Delta, Estuary, River, Stream, Creek, Lake, Pond, Water,
> Ocean, Sea, Coast, Shore, Beach, Sand, Pebble, Rock, Stone, Water, Wave,
> Surf, Tide, Current, Flow, Motion, Movement, Action, Scene, View, Image,
> Photograph, Picture, Visual, Media, Content, Data, Information, Knowledge,
> Fact, Truth, Reality, Existence, Being, Entity, Object, Subject, Topic,
> Theme, Concept, Idea, Thought, Feeling, Emotion, Sentiment, Attitude,
> Opinion, Viewpoint, Perspective, Angle, View, Look, Gaze, Stare, Glance,
> Scan, View, Look, Gaze, Stare, Glance, Scan, View, Look, Gaze, Stare,
> Glance, Scan, View, Look, Gaze, Stare, Glance, Scan, View, Look
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Title length violation (3 words; expected 5-10)
- Keyword count violation (196; expected 10-18)
- Likely capped by max token budget

---

<a id="model-mlx-community-paligemma2-3b-pt-896-4bit"></a>

### ✅ mlx-community/paligemma2-3b-pt-896-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Beach, Buildings, Cars, Coastline, Deal; excessive
  bullets=44
- _Score:_ 🟠 D (46/100); Δ-27; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, generation loop, missing
  sections, repetitive)
- _Timing:_ Load 1.42s; Gen 12.72s; Total 14.48s
- _Throughput:_ Prompt 3,484 TPS (4,684 tok); Gen 47.1 TPS (500 tok)
- _Tokens:_ prompt 4684 tok; estimated text 488 tok; estimated non-text 4196
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#45; Output only the description text after the label.
> &#45; Do not output the title text after the label.
> &#45; Do not output the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Do not output the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Do not output the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Do not output the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Do not output the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Do not output the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Do not output the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Do not output the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Do not output the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Do not output the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Do not output the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Do not output the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Do not output the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Do not output the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Do not output the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Do not output the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Do not output the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Do not output the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Do not output the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Do not output the description text after the label.
> &#45; Output only the description text after the label.
> &#45; Do not output the description text after the label.
> &#45; Output only the description
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "text after the label....")
- Excessive bullet points (44)
- Context ignored (missing: Beach, Buildings, Cars, Coastline, Deal)
- Generation loop (repetitive_tail)
- Missing sections (title, description, keywords)
- Low draft improvement
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-qwen-qwen3-vl-2b-instruct"></a>

### ✅ Qwen/Qwen3-VL-2B-Instruct

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Treat this as a mixed burden issue first; reduce that input
  load or inspect long-context handling before judging output quality.
- _Key signals:_ Output appears truncated to about 2 tokens.; At mixed burden
  (16898 tokens), output stayed unusually short (2 tokens; ratio 0.0%; weak
  text signal truncated).; output/prompt=0.01%; mixed burden=97%
- _Score:_ ❌ F (0/100); Δ-74; Empty or minimal output
- _Review focus:_ watchlist (context ignored, harness, long context)
- _Timing:_ Load 0.78s; Gen 14.31s; Total 15.41s
- _Throughput:_ Prompt 1,248 TPS (16,898 tok); Gen 179 TPS (2 tok)
- _Tokens:_ prompt 16898 tok; estimated text 488 tok; estimated non-text 16410
  tok; generated 2 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#45;
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context_budget
- ⚠️HARNESS:long_context
- Context ignored (missing: Beach, Buildings, Cars, Coastline, Deal)
- Low draft improvement
- Ignores trusted hints
- output:truncated(2tok)
- long_context_low_ratio(0.0%;16898->2;truncated)

---

<a id="model-mlx-community-apriel-15-15b-thinker-6bit-mlx"></a>

### ✅ mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Coastline, Horizon, Landscape, Seaside, Shore;
  reasoning leak
- _Score:_ ✅ B (69/100); Δ-4; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, missing sections, reasoning leak,
  unverified context copy)
- _Timing:_ Load 1.62s; Gen 14.75s; Total 16.72s
- _Throughput:_ Prompt 1,534 TPS (3,008 tok); Gen 40.7 TPS (500 tok)
- _Tokens:_ prompt 3008 tok; estimated text 488 tok; estimated non-text 2520
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Here are my reasoning steps:
> We need to produce a catalog entry with Title, Description, Keywords. Use
> only details that are clearly visible in the image. The image shows a
> seafront with a shingle beach, a row of buildings (seafront buildings), a
> promenade, some cars parked, a few people on the beach, waves, partly cloudy
> sky, etc. The location is Deal, Kent, England, UK. The capture date/time is
> given as 2026-07-11 19:04:26 BST. The GPS coordinates are 51.226814°N,
> 1.401142°E. The context says we can use location terms like England, Europe,
> Town, UK. The draft title: "Seafront, Deal, England, UK, GBR, Europe". The
> draft description: "A coastal view of Deal, Kent, UK, showing the shingle
> beach, sea, and seafront buildings on a partly cloudy day." The draft
> keywords: many.
>
> We need to keep only details that are definitely visible. The image shows a
> shingle beach (brownish). The sea is visible with waves. The sky is partly
> cloudy. The seafront buildings are visible, with a mix of white and colored
> facades. There are cars parked along the promenade. There are a few people
> on the beach (two people near the left side, maybe a child and adult). There
> are some birds flying in the sky. There is a pier? Not clearly visible.
> There is a breakwater? Not clearly visible. There is a promenade with a
> railing. There is a small boat? Not visible. There is a small lifebuoy? Not
> visible. There is a sign? Not visible. There is a small building with a
> tower? Possibly a hotel. There is a building with a sign "M" maybe? Not
> clear. There is a small building with a tower near the center. There is a
> row of houses. There is a small car parked near the water. There is a small
> boat? Not visible.
>
> We need to produce a title of 5-10 words, concrete and factual, limited to
> clearly visible content. So something like "Deal Seafront, Shingle Beach,
> Buildings, Cars, Waves". That's 7 words. Or "Deal seaf
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)
- Reasoning leak (here are my reasoning steps)
- Unverified context copy
- Likely capped by max token budget
- suspicious_precision: 2 overly precise numbers

---

<a id="model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16"></a>

### ✅ mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Cars, Coastline, Horizon, Landscape, People;
  repetitive token=phrase: "stade stade stade stade..."
- _Score:_ 🟠 D (38/100); Δ-35; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, generation loop, missing sections,
  repetitive)
- _Timing:_ Load 6.60s; Gen 14.87s; Total 21.83s
- _Throughput:_ Prompt 622 TPS (1,887 tok); Gen 44.9 TPS (500 tok)
- _Tokens:_ prompt 1887 tok; estimated text 488 tok; estimated non-text 1399
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Let's tackle this step by step. First, the user provided a prompt with a lot
> of repetitive and unclear terms, but the key is to focus on what's actually
> visible in the image. Wait, but wait—the user mentioned the image is a
> "panoramic view of the seaside town of Deal in Kent, England, UK, with a
> shingle beach, seafront buildings, and a partly cloudy sky." Wait no, wait
> the user's prompt has a lot of repeated words like "stade" (probably typo
> for "stade" no, wait "stade" is a unit, but no—wait the user's prompt has a
> lot of gibberish: "stade stade stade stade stade stade stade stade stade
> stade stade stade stade stade stade stade stade stade stade stade stade
> stade stade stade stade stade stade stade stade stade stade stade stade
> stade stade stade stade stade stade stade stade stade stade stade stade
> stade stade stade stade stade stade stade stade stade stade stade stade
> stade stade stade stade stade stade stade stade stade stade stade stade
> stade stade stade stade stade stade stade stade stade stade stade stade
> stade stade stade stade stade stade stade stade stade stade stade stade
> stade stade stade stade stade stade stade stade stade stade stade stade
> stade stade stade stade stade stade stade stade stade stade stade stade
> stade stade stade stade stade stade stade stade stade stade stade stade
> stade stade stade stade stade stade stade stade stade stade stade stade
> stade stade stade stade stade stade stade stade stade stade stade stade
> stade stade stade stade stade stade stade stade stade stade stade stade
> stade stade stade stade stade stade stade stade stade stade stade stade
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "stade stade stade stade...")
- Generation loop (repetitive_tail)
- Missing sections (title, description, keywords)
- Low draft improvement
- Likely capped by max token budget

---

<a id="model-mlx-community-glm-46v-nvfp4"></a>

### ✅ mlx-community/GLM-4.6V-nvfp4

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a visual input burden issue first; reduce that
  input load or inspect long-context handling before judging output quality.
- _Key signals:_ output/prompt=1.41%; visual input burden=93%; missing
  sections: title; missing terms: Kent, Swimming, Walking, GBR, view
- _Score:_ 🟡 C (63/100); Δ-10; Limited novel information
- _Review focus:_ watchlist (hallucination, missing sections)
- _Timing:_ Load 6.73s; Gen 16.14s; Total 23.21s
- _Throughput:_ Prompt 493 TPS (6,593 tok); Gen 40.8 TPS (93 tok)
- _Tokens:_ prompt 6593 tok; estimated text 488 tok; estimated non-text 6105
  tok; generated 93 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> <|begin_of_box|>Title:
> Seafront buildings, shingle beach, and waves at Deal
>
> &#8203;Description:
> A coastal scene with shingle beach, turbulent sea, and a row of multi-story
> buildings along the shore, under a grey sky with people relaxing on the
> beach.
>
> &#8203;Keywords:
> Beach, Buildings, Cars, Coastline, Deal, Horizon, Landscape, People,
> Promenade, Seaside, Shore, Sitting, Sky, Water, Waterfront, Waves,
> architecture<|end_of_box|>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context_budget
- Missing sections (title)
- Low draft improvement
- Contains unexpected table

---

<a id="model-meta-llama-llama-32-11b-vision-instruct"></a>

### ✅ meta-llama/Llama-3.2-11B-Vision-Instruct

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Cars, Coastline, Horizon, Kent, Landscape; low-draft-improvement
- _Score:_ 🟠 D (41/100); Δ-33; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (missing sections, trusted hint degraded)
- _Timing:_ Load 2.31s; Gen 16.25s; Total 18.89s
- _Throughput:_ Prompt 288 TPS (556 tok); Gen 5.09 TPS (70 tok)
- _Tokens:_ prompt 556 tok; estimated text 488 tok; estimated non-text 68 tok;
  generated 70 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Seafront, Deal, England, UK
>
> The image depicts a seaside town with a long stretch of sandy beach and a
> row of buildings along the shore. The beach is empty, with a few people
> scattered about, and the sea is choppy with waves crashing against the
> shore. The sky is overcast, with a few birds flying overhead.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)
- Low draft improvement
- Degrades trusted hints

---

<a id="model-mlx-community-molmo-7b-d-0924-8bit"></a>

### ✅ mlx-community/Molmo-7B-D-0924-8bit

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Seafront, GBR, view, showing, seafront;
  keywords=19; low-draft-improvement
- _Score:_ 🟡 C (63/100); Δ-10; None identified
- _Review focus:_ watchlist (worse than metadata baseline)
- _Timing:_ Load 1.25s; Gen 16.69s; Total 18.30s
- _Throughput:_ Prompt 133 TPS (1,764 tok); Gen 46.5 TPS (114 tok)
- _Tokens:_ prompt 1764 tok; estimated text 488 tok; estimated non-text 1276
  tok; generated 114 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Remove uncertain tags.
>
> Title: Shingle beach at Deal, Kent, UK
>
> Description: A partly cloudy day at Deal's shingle beach, with the town's
> buildings visible along the horizon. The sea is visible on the right, with
> waves breaking on the shore. A few people can be seen sitting on the beach,
> enjoying the coastal scenery.
>
> Keywords: Beach, Buildings, Cars, Coastline, Deal, Kent, Landscape, People,
> Promenade, Seaside, Shore, Sitting, Sky, Swimming, Walking, Water,
> Waterfront, Waves, architecture
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Description sentence violation (3; expected 1-2)
- Keyword count violation (19; expected 10-18)
- Low draft improvement

---

<a id="model-mlx-community-x-reasoner-7b-8bit"></a>

### ✅ mlx-community/X-Reasoner-7B-8bit

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Treat this as a mixed burden issue first; reduce that input
  load or inspect long-context handling before judging output quality.
- _Key signals:_ Output appears truncated to about 5 tokens.; At mixed burden
  (16909 tokens), output stayed unusually short (5 tokens; ratio 0.0%; weak
  text signal truncated).; output/prompt=0.03%; mixed burden=97%
- _Score:_ ❌ F (6/100); Δ-68; Output too short to be useful
- _Review focus:_ watchlist (context ignored, harness, long context)
- _Timing:_ Load 1.22s; Gen 18.73s; Total 20.28s
- _Throughput:_ Prompt 944 TPS (16,909 tok); Gen 68.5 TPS (5 tok)
- _Tokens:_ prompt 16909 tok; estimated text 488 tok; estimated non-text 16421
  tok; generated 5 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> <|im_start|>
> addCriterion
>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context_budget
- ⚠️HARNESS:long_context
- Context ignored (missing: Beach, Buildings, Cars, Coastline, Deal)
- Low draft improvement
- Ignores trusted hints
- output:truncated(5tok)
- long_context_low_ratio(0.0%;16909->5;truncated)

---

<a id="model-mlx-community-molmo-7b-d-0924-bf16"></a>

### ✅ mlx-community/Molmo-7B-D-0924-bf16

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Horizon; keywords=23; low-draft-improvement
- _Score:_ ✅ B (77/100); Δ+4; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 1.91s; Gen 21.29s; Total 23.69s
- _Throughput:_ Prompt 124 TPS (1,764 tok); Gen 29.2 TPS (173 tok)
- _Tokens:_ prompt 1764 tok; estimated text 488 tok; estimated non-text 1276
  tok; generated 173 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Remove uncertain tags.
>
> Title: Seafront, Deal, Kent, UK, GBR, Europe
>
> Description: A coastal view of Deal, Kent, UK, showing the shingle beach,
> sea, and seafront buildings on a partly cloudy day. The beach is mostly
> empty except for a few people, and the sea appears murky. The sky is light
> blue with some clouds. The buildings along the seafront are mostly white,
> with one prominent brown building. Cars are parked along the road behind the
> beach. The image captures the essence of a typical English seaside town
> with its characteristic architecture and landscape.
>
> Keywords: Beach, Buildings, Cars, Coastline, Deal, Kent, Landscape, People,
> Promenade, Seaside, Shore, Sitting, Sky, Swimming, Walking, Water,
> Waterfront, Waves, architecture, English, Seafront, UK, Europe
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Description sentence violation (6; expected 1-2)
- Keyword count violation (23; expected 10-18)
- Low draft improvement

---

<a id="model-mlx-community-molmopoint-8b-fp16"></a>

### ✅ mlx-community/MolmoPoint-8B-fp16

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Coastline, Deal, Horizon, Kent;
  keywords=22; low-draft-improvement
- _Score:_ ✅ B (75/100); Δ+1; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 3.13s; Gen 23.45s; Total 26.98s
- _Throughput:_ Prompt 1,244 TPS (3,381 tok); Gen 6.04 TPS (119 tok)
- _Tokens:_ prompt 3381 tok; estimated text 488 tok; estimated non-text 2893
  tok; generated 119 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Shingle Beach with Seafront Buildings and People
>
> &#8203;Description:
> A coastal scene featuring a shingle beach with people sitting on blankets, a
> seafront lined with buildings, and a cloudy sky. The image captures the
> interaction between the beachgoers, the urban waterfront, and the natural
> coastal environment.
>
> &#8203;Keywords:
> Shingle beach, Seafront buildings, People, Blankets, Urban waterfront,
> Coastal landscape, Cloudy sky, Seaside, Water, Waves, Architecture, England,
> UK, Town, UK, Landscape, Seaside, Shore, Sitting, Waterfront, Water, Waves
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Keyword count violation (22; expected 10-18)
- Low draft improvement

---

<a id="model-mlx-community-qwen3-vl-2b-instruct-bf16"></a>

### ✅ mlx-community/Qwen3-VL-2B-Instruct-bf16

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Treat this as a mixed burden issue first; reduce that input
  load or inspect long-context handling before judging output quality.
- _Key signals:_ Output appears truncated to about 2 tokens.; At mixed burden
  (16898 tokens), output stayed unusually short (2 tokens; ratio 0.0%; weak
  text signal truncated).; output/prompt=0.01%; mixed burden=97%
- _Score:_ ❌ F (0/100); Δ-74; Empty or minimal output
- _Review focus:_ watchlist (context ignored, harness, long context)
- _Timing:_ Load 0.94s; Gen 24.84s; Total 26.13s
- _Throughput:_ Prompt 706 TPS (16,898 tok); Gen 139 TPS (2 tok)
- _Tokens:_ prompt 16898 tok; estimated text 488 tok; estimated non-text 16410
  tok; generated 2 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#45;
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context_budget
- ⚠️HARNESS:long_context
- Context ignored (missing: Beach, Buildings, Cars, Coastline, Deal)
- Low draft improvement
- Ignores trusted hints
- output:truncated(2tok)
- long_context_low_ratio(0.0%;16898->2;truncated)

---

<a id="model-mlx-community-paligemma2-3b-ft-docci-448-bf16"></a>

### ✅ mlx-community/paligemma2-3b-ft-docci-448-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Beach, Buildings, Cars, Coastline, Deal;
  low-draft-improvement
- _Score:_ 🟡 C (52/100); Δ-22; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, missing sections)
- _Timing:_ Load 1.66s; Gen 27.87s; Total 29.88s
- _Throughput:_ Prompt 3,267 TPS (1,612 tok); Gen 18.6 TPS (500 tok)
- _Tokens:_ prompt 1612 tok; estimated text 488 tok; estimated non-text 1124
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#45; Do not copy-prompt instructions into the Title, Description, or Keywords
> fields.
> &#45; Do not infer identity, location, event, brand, species, time period, or
> intent.
> &#45; Use this factual context where it improves the catalogue record; do not
> claim that contextual facts are visually observable.
> &#45; Do not copy-prompt instructions into the Title, Description, or Keywords
> fields.
> &#45; Do not infer identity, location, event, brand, species, time period, or
> intent.
> &#45; Use this factual context where it improves the catalogue record; do not
> claim that contextual facts are visually observable.
> &#45; Do not copy-prompt instructions into the Title, Description, or Keywords
> fields.
> &#45; Do not infer identity, location, event, brand, species, time period, or
> intent.
> &#45; Use this factual context where it improves the catalogue record; do not
> claim that contextual facts are visually observable.
> &#45; Do not copy-prompt instructions into the Title, Description, or Keywords
> fields.
> &#45; Do not infer identity, location, event, brand, species, time period, or
> intent.
> &#45; Use this factual context where it improves the catalogue record; do not
> claim that contextual facts are visually observable.
> &#45; Do not copy-prompt instructions into the Title, Description, or Keywords
> fields.
> &#45; Do not infer identity, location, event, brand, species, time period, or
> intent.
> &#45; Use this factual context where it improves the catalogue record; do not
> claim that contextual facts are visually observable.
> &#45; Do not copy-prompt instructions into the Title, Description, or Keywords
> fields.
> &#45; Do not infer identity, location, event, brand, species, time period, or
> intent.
> &#45; Use this factual context where it improves the catalogue record; do not
> claim that contextual facts are visually observable.
> &#45; Do not copy-prompt instructions into the Title, Description, or Keywords
> fields.
> &#45; Do not infer identity, location, event, brand, species, time period, or
> intent.
> &#45; Use this factual context where it improves the catalogue record; do not
> claim that contextual facts are visually observable.
> &#45; Do not copy-prompt instructions into the Title, Description, or Keywords
> fields.
> &#45; Do not infer identity, location, event, brand, species, time period, or
> intent.
> &#45; Use this factual context where it improves the catalogue record; do not
> claim that contextual facts are visually observable.
> &#45; Do not copy
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Beach, Buildings, Cars, Coastline, Deal)
- Missing sections (title, description, keywords)
- Low draft improvement
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-qwen3-vl-2b-thinking-bf16"></a>

### ✅ mlx-community/Qwen3-VL-2B-Thinking-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect KV/cache behavior, memory pressure, and long-context
  execution.
- _Key signals:_ At mixed burden (16900 tokens), output became repetitive.;
  hit token cap (500); missing sections: title, description, keywords; missing
  terms: Beach, Buildings, Cars, Coastline, Deal
- _Score:_ ❌ F (20/100); Δ-53; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, generation
  loop, harness, long context, missing sections, repetitive, text sanity)
- _Timing:_ Load 0.87s; Gen 33.85s; Total 35.15s
- _Throughput:_ Prompt 650 TPS (16,900 tok); Gen 73.6 TPS (500 tok)
- _Tokens:_ prompt 16900 tok; estimated text 488 tok; estimated non-text 16412
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#45; - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
> &#45; - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
> &#45; - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
> &#45; - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
> &#45; - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
> &#45; - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
> &#45; - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
> &#45; - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
> &#45; - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
> &#45; - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
> &#45; - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
> &#45; - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
> &#45; - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
> &#45; - - - - -
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:long_context
- Repetitive output (-)
- Context ignored (missing: Beach, Buildings, Cars, Coastline, Deal)
- Output degeneration (character_loop: ' -' repeated)
- Text sanity issue (gibberish(token_noise))
- Generation loop (repetitive_tail)
- Missing sections (title, description, keywords)
- Low draft improvement
- Likely capped by max token budget
- Ignores trusted hints
- long_context_repetition(16900tok)

---

<a id="model-mlx-community-qwen35-9b-mlx-4bit"></a>

### ✅ mlx-community/Qwen3.5-9B-MLX-4bit

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Walking; keywords=19; low-draft-improvement
- _Score:_ 🟠 D (41/100); Δ-33; Lacks visual description of image
- _Review focus:_ watchlist (worse than metadata baseline)
- _Timing:_ Load 1.44s; Gen 56.58s; Total 58.35s
- _Throughput:_ Prompt 309 TPS (16,928 tok); Gen 93.1 TPS (95 tok)
- _Tokens:_ prompt 16928 tok; estimated text 488 tok; estimated non-text 16440
  tok; generated 95 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Seafront, Deal, England, UK, GBR, Europe
>
> &#8203;Description:
> A coastal view of Deal, Kent, UK, showing the shingle beach, sea, and
> seafront buildings on a partly cloudy day.
>
> &#8203;Keywords:
> Beach, Buildings, Cars, Coastline, Deal, Horizon, Kent, Landscape, People,
> Promenade, Seaside, Shore, Sitting, Sky, Swimming, Water, Waterfront, Waves,
> architecture
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Keyword count violation (19; expected 10-18)
- Low draft improvement

---

<a id="model-mlx-community-qwen35-35b-a3b-4bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-4bit

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Deal, Horizon, Kent, Swimming, Walking
- _Score:_ 🏆 A (82/100); Δ+9; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 2.57s; Gen 58.47s; Total 61.40s
- _Throughput:_ Prompt 300 TPS (16,928 tok); Gen 94.5 TPS (121 tok)
- _Tokens:_ prompt 16928 tok; estimated text 488 tok; estimated non-text 16440
  tok; generated 121 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Shingle beach and seafront buildings under cloudy sky
>
> Description: A wide shingle beach stretches along the coast with small
> groups of people sitting on the shore. A row of multi-story buildings lines
> the promenade behind the beach, with parked cars visible along the street.
> The sea is choppy with white-crested waves rolling onto the shore under an
> overcast sky.
>
> Keywords: shingle beach, sea, seafront, buildings, promenade, cars, people,
> waves, cloudy sky, coastline, shore, water, architecture, England, UK,
> Europe, landscape, seaside
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Description sentence violation (3; expected 1-2)

---

<a id="model-mlx-community-qwen35-35b-a3b-6bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-6bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Kent, Sitting, Swimming, Walking, GBR
- _Score:_ 🟡 C (64/100); Δ-9; None identified
- _Review focus:_ watchlist (worse than metadata baseline)
- _Timing:_ Load 3.17s; Gen 61.18s; Total 64.70s
- _Throughput:_ Prompt 286 TPS (16,928 tok); Gen 89.3 TPS (105 tok)
- _Tokens:_ prompt 16928 tok; estimated text 488 tok; estimated non-text 16440
  tok; generated 105 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Shingle beach and seafront buildings at Deal
>
> &#8203;Description:
> A wide shingle beach stretches along a row of multi-story seafront buildings
> under a cloudy sky. Small groups of people are visible on the beach near
> the water's edge, with waves breaking on the shore.
>
> &#8203;Keywords:
> Shingle beach, Seafront, Buildings, Coastline, Waves, Water, Sky, People,
> Promenade, Cars, Architecture, England, UK, Shore, Seaside, Landscape,
> Horizon, Waterfront
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-qwen35-35b-a3b-bf16"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-bf16

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Horizon, Kent, Sitting, Swimming, Walking
- _Score:_ ✅ B (65/100); Δ-8; None identified
- _Review focus:_ watchlist (worse than metadata baseline)
- _Timing:_ Load 12.78s; Gen 69.75s; Total 82.87s
- _Throughput:_ Prompt 254 TPS (16,928 tok); Gen 65.7 TPS (110 tok)
- _Tokens:_ prompt 16928 tok; estimated text 488 tok; estimated non-text 16440
  tok; generated 110 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Shingle beach, seafront buildings, and sea at Deal
>
> &#8203;Description:
> A wide shingle beach stretches along a row of multi-storey seafront
> buildings under a cloudy sky. Small groups of people are visible on the
> beach near the water's edge, with gentle waves breaking on the shore.
>
> &#8203;Keywords:
> Shingle beach, Seafront, Buildings, Coastline, Water, Waves, People, Sky,
> Clouds, Promenade, Cars, Architecture, Seaside, Shore, England, UK, Europe,
> Landscape
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-qwen36-27b-mxfp8"></a>

### ✅ mlx-community/Qwen3.6-27B-mxfp8

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Kent, Promenade, Seaside, Swimming, Walking;
  keywords=19
- _Score:_ ✅ B (72/100); Δ-1; None identified
- _Timing:_ Load 3.08s; Gen 82.78s; Total 86.20s
- _Throughput:_ Prompt 226 TPS (16,928 tok); Gen 17.5 TPS (124 tok)
- _Tokens:_ prompt 16928 tok; estimated text 488 tok; estimated non-text 16440
  tok; generated 124 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Shingle beach and seafront buildings in Deal, England
>
> &#8203;Description:
> A wide expanse of brown shingle beach stretches along the coastline,
> bordered by a row of multi-story residential buildings and parked cars. The
> sea is choppy with white-capped waves rolling onto the shore, while a few
> people are visible sitting on the beach under a bright, hazy sky.
>
> &#8203;Keywords:
> Shingle beach, Seafront, Buildings, Deal, England, Coastline, Waves, Ocean,
> Cars, People, Sky, Horizon, Residential, Water, Landscape, UK, Europe,
> Shore, Architecture
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Keyword count violation (19; expected 10-18)

---

<a id="model-mlx-community-ornith-10-35b-bf16"></a>

### ✅ mlx-community/Ornith-1.0-35B-bf16

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Cars, Deal, Swimming, architecture, GBR;
  keywords=20
- _Score:_ 🏆 A (83/100); Δ+10; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 13.55s; Gen 83.29s; Total 97.38s
- _Throughput:_ Prompt 216 TPS (16,928 tok); Gen 51.4 TPS (120 tok)
- _Tokens:_ prompt 16928 tok; estimated text 488 tok; estimated non-text 16440
  tok; generated 120 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Seafront buildings and shingle beach at low tide
>
> Description: A row of multi-storey seafront buildings lines a wide shingle
> beach where a small group of people are sitting and walking. The choppy,
> brownish-green sea washes onto the shore under a pale, overcast sky, with
> white cliffs visible on the distant horizon.
>
> Keywords: Beach, Buildings, Boats, Coastline, England, Horizon, Kent,
> Landscape, People, Promenade, Sea, Seaside, Shingle, Shore, Sky, Town, UK,
> Water, Waterfront, Waves
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Keyword count violation (20; expected 10-18)

---

<a id="model-mlx-community-qwen35-27b-mxfp8"></a>

### ✅ mlx-community/Qwen3.5-27B-mxfp8

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Kent, Sitting, Walking, GBR, view;
  keywords=20; low-draft-improvement
- _Score:_ 🟡 C (64/100); Δ-10; None identified
- _Review focus:_ watchlist (worse than metadata baseline)
- _Timing:_ Load 5.03s; Gen 87.83s; Total 93.21s
- _Throughput:_ Prompt 211 TPS (16,928 tok); Gen 17.5 TPS (116 tok)
- _Tokens:_ prompt 16928 tok; estimated text 488 tok; estimated non-text 16440
  tok; generated 116 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Shingle Beach and Seafront Buildings, Deal, England
>
> &#8203;Description:
> A coastal scene showing a wide expanse of brown shingle beach meeting
> choppy, brownish sea water with white foam. A row of multi-story residential
> buildings lines the promenade in the background, with parked cars visible
> along the road.
>
> &#8203;Keywords:
> Beach, Buildings, Cars, Coastline, Deal, England, Horizon, Landscape,
> People, Promenade, Sea, Seaside, Shore, Shingle, Sky, Swimming, Water,
> Waterfront, Waves, architecture
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Keyword count violation (20; expected 10-18)
- Low draft improvement

---

<a id="model-mlx-community-qwen2-vl-2b-instruct-4bit"></a>

### ✅ mlx-community/Qwen2-VL-2B-Instruct-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect KV/cache behavior, memory pressure, and long-context
  execution.
- _Key signals:_ At mixed burden (16909 tokens), output became repetitive.;
  hit token cap (500); missing sections: description; missing terms: Beach,
  Buildings, Cars, Coastline, Horizon
- _Score:_ 🟠 D (43/100); Δ-31; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, generation loop,
  harness, long context, missing sections, repetitive, unverified context
  copy)
- _Timing:_ Load 0.68s; Gen 87.90s; Total 88.97s
- _Throughput:_ Prompt 205 TPS (16,909 tok); Gen 110 TPS (500 tok)
- _Tokens:_ prompt 16909 tok; estimated text 488 tok; estimated non-text 16421
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> -001.jpg
> &#45; 19:04:26 BST
> &#45; 51.226814°N, 1.401142°E
> &#45; 2026-07-11 19:04:26 BST
>
> Title: Deal, Kent, UK, 2026-07-11 19:04:26 BST
>
> &#8203;Keywords:
> &#45; 19:04:26 BST
> &#45; Deal, Kent, UK, 2026-07-11 19:04:26 BST
> &#45; 51.226814°N, 1.401142°E
> &#45; 2026-07-11 19:04:26 BST
> &#45; Deal, Kent, UK, 2026-07-11 19:04:26 BST
> &#45; Deal, Kent, UK, 2026-07-11 19:04:26 BST
> &#45; Deal, Kent, UK, 2026-07-11 19:04:26 BST
> &#45; Deal, Kent, UK, 2026-07-11 19:04:26 BST
> &#45; Deal, Kent, UK, 2026-07-11 19:04:26 BST
> &#45; Deal, Kent, UK, 2026-07-11 19:04:26 BST
> &#45; Deal, Kent, UK, 2026-07-11 19:04:26 BST
> &#45; Deal, Kent, UK, 2026-07-11 19:04:26 BST
> &#45; Deal, Kent, UK, 2026-07-11 19:04:26 BST
> &#45; Deal, Kent, UK, 2026-07-11 19:04:26 BST
> &#45; Deal, Kent, UK, 2026-07-11 19
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:long_context
- Repetitive output (phrase: "deal, kent, uk, 2026-07-11...")
- Context ignored (missing: Beach, Buildings, Cars, Coastline, Horizon)
- Generation loop (repetitive_tail)
- Missing sections (description)
- Keyword count violation (52; expected 10-18)
- Keyword duplication (85% duplicated terms)
- Unverified context copy
- Low draft improvement
- Likely capped by max token budget
- Ignores trusted hints
- suspicious_precision: 4 overly precise numbers
- long_context_repetition(16909tok)

---

<a id="model-mlx-community-qwen35-27b-4bit"></a>

### ✅ mlx-community/Qwen3.5-27B-4bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Seaside, Shore, Walking, Water, architecture
- _Score:_ ✅ B (72/100); Δ-1; None identified
- _Timing:_ Load 2.56s; Gen 100.06s; Total 103.19s
- _Throughput:_ Prompt 178 TPS (16,928 tok); Gen 29.0 TPS (112 tok)
- _Tokens:_ prompt 16928 tok; estimated text 488 tok; estimated non-text 16440
  tok; generated 112 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Shingle Beach and Seafront Buildings in Deal, Kent
>
> &#8203;Description:
> A wide coastal view shows a shingle beach meeting the sea, with a row of
> multi-coloured buildings and parked cars along the promenade in the
> background. A few people are visible sitting on the beach under a bright,
> hazy sky.
>
> &#8203;Keywords:
> Beach, Buildings, Cars, Coastline, Deal, Horizon, Kent, Landscape, People,
> Promenade, Sea, Shingle, Sky, Swimming, Town, UK, Waterfront, Waves
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-2506-bf16"></a>

### ✅ mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title; missing terms:
  Cars, Coastline, Landscape, Promenade, Seaside; keyword duplication=52%
- _Score:_ ✅ B (78/100); Δ+4; Missing requested structure
- _Review focus:_ watchlist (cutoff, missing sections, thinking incomplete)
- _Timing:_ Load 3.28s; Gen 119.63s; Total 123.24s
- _Throughput:_ Prompt 1,084 TPS (1,571 tok); Gen 4.26 TPS (500 tok)
- _Tokens:_ prompt 1571 tok; estimated text 488 tok; estimated non-text 1083
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ◁think▷Got it, let's analyze the image step by step.
>
> First, the title: The image shows a coastal scene with buildings along a
> beach. The location is Deal, Kent, UK. So the title should be concise. Let's
> check the existing draft: "Seafront, Deal, England, UK, GBR, Europe" –
> that's good. But maybe "Seafront" is a bit generic. Let's make it more
> specific. The image has a shingle beach, buildings, sea, and a partly cloudy
> sky. So the title could be "Shingle beach and seafront buildings at Deal,
> Kent, UK".
>
> Description: The main subject is the coastal scene. Visible elements:
> shingle beach, sea with waves, buildings (hotels or residential) along the
> shore, people on the beach (some sitting, some walking), partly cloudy sky.
> So the description should be factual. Let's structure it: "A shingle beach
> with waves along the seafront, buildings lining the shore, and people
> relaxing or walking under a partly cloudy sky in Deal, Kent, UK."
>
> Keywords: Need 10-18 unique terms. Visible elements: shingle beach, seafront
> buildings, people (relaxing, walking), waves, partly cloudy sky, Deal,
> Kent, UK, England, Europe, coastal scene, shoreline, buildings, sea, waves,
> beach, sky, horizon. Let's list them: Shingle beach, Seafront buildings,
> People, Waves, Partly cloudy sky, Deal, Kent, UK, England, Europe, Coastal
> scene, Shoreline, Buildings, Sea, Waves, Beach, Sky, Horizon.
>
> Wait, check if all are visible. The image has buildings along the shore,
> people (some sitting, some walking), shingle beach, sea with waves, partly
> cloudy sky. So keywords: Shingle beach, Seafront buildings, People, Waves,
> Partly cloudy sky, Deal, Kent, UK, England, Europe, Coastal scene,
> Shoreline, Buildings, Sea, Waves, Beach, Sky, Horizon. That's 16 keywords.
> Need to check if any are redundant. "Shoreline" and "Beach" are related but
> distinct. "Buildings" is there. "Sea" and "Waves" are separate. "Sky" and
> "Partly cloudy sky" – maybe combine? No, "Partly cloudy sky" is more
> specific. So keep both. Let's list them as comma-separated.
>
> Now,
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title)
- Description sentence violation (4; expected 1-2)
- Keyword count violation (62; expected 10-18)
- Keyword duplication (52% duplicated terms)
- Thinking trace incomplete (expected model protocol; opened with ◁think▷)
- Likely capped by max token budget

---

<!-- markdownlint-enable MD033 MD034 -->
