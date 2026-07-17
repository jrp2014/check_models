# Model Output Gallery

Generated on: 2026-07-13 05:57:30 BST

Complete per-model evidence artifact with image metadata, the source prompt,
summary tables, diagnostics, and full generated output for every attempted
model.

_Action Snapshot: see [results.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/results.md) for the full summary._

## Run Stamps

- `mlx-vlm`: `0.6.5`
- `mlx`: `0.32.1.dev20260712+4367c73b`
- `mlx-lm`: `0.31.3`
- `transformers`: `5.12.1`
- `tokenizers`: `0.22.2`
- `huggingface-hub`: `1.23.0`
- _Python Version:_ 3.13.13
- _OS:_ Darwin 25.5.0
- _macOS Version:_ 26.5.2
- _GPU/Chip:_ Apple M5 Max

## Image Metadata

- _Title:_ , Deben Estuary, Woodbridge, England, UK, GBR, Europe
- _Description:_ Two sailing boats moored on a river with trees behind on the
  bank
- _Keywords:_ Bird, Boat, Boating, Buoy, Bushes, Coast, Deben Estuary,
  England, Estuary, Europe, Foliage, Forest, Landscape, Mast, Moored, Mudflat,
  Nature, Outdoors, Peaceful, Rigging, River, Riverbank, Sailboat, Sailing,
  Suffolk, Trees, UK, Water, Watercraft, Woodbridge, Woodland, anchored, bank,
  calm, classic boat, flora, getaway, green, holiday, lake, leisure, low tide,
  marine, mooring, mudbank, nautical, recreation, reflection, sail, sailing
  boat, scenic, serene, shore, summer, tide, tranquil, travel, vacation,
  vessel, vintage boat, wooden boat, yacht
- _Date:_ 2026-07-04 19:10:04 BST
- _Time:_ 19:10:04

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
> &#45; Location terms: Deben Estuary, England, Europe, UK, Woodbridge
> &#45; Capture date/time: 2026-07-04 19:10:04 BST 19:10:04
> &#45; Use this factual context where it improves the catalogue record; do not
> claim that contextual facts are visually observable.
>
> &#8203;Draft descriptive metadata:
> &#45; Existing title: Deben Estuary, Woodbridge, England, UK, GBR, Europe
> &#45; Existing description: Two sailing boats moored on a river with trees
> behind on the bank
> &#45; Existing keywords: Bird, Boat, Boating, Buoy, Bushes, Coast, Estuary,
> Foliage, Forest, Landscape, Mast, Moored, Mudflat, Nature, Outdoors,
> Peaceful, Rigging, River, Riverbank, Sailboat
> &#45; Treat this draft as fallible. Retain supported details, correct errors,
> and add important visible information.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

## Model Quality Summary

Skim-first view of what each model returned, or the strongest diagnostic when it did not produce usable output.

<!-- markdownlint-disable MD034 -->

| Model                                                                                                                   | Result                        | Quality / diagnostic                                               | Response / diagnostic                                                                                                                                                                                                                                                                                                                        |
|-------------------------------------------------------------------------------------------------------------------------|-------------------------------|--------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`mlx-community/gemma-4-31b-bf16`](#model-mlx-community-gemma-4-31b-bf16)                                               | `avoid` / `semantic mismatch` | mlx; model-error                                                   | [mlx; model-error] Error: Model Error - Model runtime error during generation for mlx-community/gemma-4-31b-bf16: [METAL] Command buffer execution failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory).                                                                                                       |
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `caveat` / `clean`            | missing terms: Bird, Buoy, Mudflat, behind, bank                   | Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe, Boating, Sailboat, River, Riverbank, Nature, Outdoors, Peaceful, Rigging, Boats, Muddy, Coast, Estuary, Foliage, Forest, Landscape, Mast, Moored, Sa ... [tail] ary, Sailing, Muddy, Coast, Nature, Outdoors, Peaceful, Rigging, Boats, Mushrooms, Trees, Water, Sky             |
| [`mlx-community/LFM2-VL-1.6B-8bit`](#model-mlx-community-lfm2-vl-16b-8bit)                                              | `avoid` / `model shortcoming` | unverified-context-copy; low-draft-improvement; metadata-borrowing | Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe, 2026-07-04 19:10:04 BST 19:10:04 Description: Two sailing boats moored on a river with trees behind on the bank. The boats are white with orange mas ... [tail] t, Landscape, Mast, Moored, Mudflat, Nature, Outdoors, Peaceful, Rigging, River, Riverbank, Sailboat             |
| [`qnguyen3/nanoLLaVA`](#model-qnguyen3-nanollava)                                                                       | `avoid` / `model shortcoming` | low-draft-improvement; missing-sections                            | Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe Description: Two sailing boats moored on a river with trees behind on the bank.                                                                                                                                                                                                   |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                            | `caveat` / `clean`            | low-draft-improvement                                              | Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe Description: Two sailing boats moored on a river with trees behind on the bank Keywords: Bird, Boat, Boating, Buoy, Bushes, Coast, Estuary, Foliage, ... [tail] dflat, Nature, Outdoors, Peaceful, Rigging, River, Riverbank, Sailboat Do you need more information?              |
| [`HuggingFaceTB/SmolVLM-Instruct`](#model-huggingfacetb-smolvlm-instruct)                                               | `avoid` / `model shortcoming` | low-draft-improvement; context-ignored; missing-sections           | Deben Estuary, Woodbridge, England, UK, GBR, Europe.                                                                                                                                                                                                                                                                                         |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `avoid` / `harness`           | harness:stop-token; reasoning-leak; formatting                     | &lt;think&gt; &lt;/think&gt; Title: Sailing boats moored near dense foliage on water Description: The image shows two sailing boats floating on calm water with lush green trees in the background. The scene appears peacef ... [tail] ling boats, water, trees, foliage, moored, calm, green, river, nature, peaceful, mast, rigging, buoy |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `avoid` / `model shortcoming` | unverified-context-copy; low-draft-improvement; metadata-borrowing | Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe Description: Two sailing boats moored on a river with trees behind on the bank Keywords: Bird, Boat, Boating, Buoy, Bushes, Coast, Estuary, Foliage, ... [tail] where it improves the catalogue record; do not claim that contextual facts are visually observable.               |
| [`mlx-community/paligemma2-10b-ft-docci-448-6bit`](#model-mlx-community-paligemma2-10b-ft-docci-448-6bit)               | `avoid` / `model shortcoming` | low-draft-improvement; context-ignored; missing-sections           | - Do not copy prompt instructions into the Title, Description, or Keywords fields.                                                                                                                                                                                                                                                           |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `avoid` / `semantic mismatch` | missing terms: Bird, Boating, Buoy, Bushes, Coast                  | Title: Two small sailboats moored on the Deben Estuary Description: Two small sailboats with wooden masts are moored on the water against a backdrop of dense green trees. The scene is captured in natural ligh ... [tail] trees, foliage, forest, greenery, nature, outdoors, landscape, mooring, estuary, vertical, daylight              |
| [`microsoft/Phi-3.5-vision-instruct`](#model-microsoft-phi-35-vision-instruct)                                          | `avoid` / `semantic mismatch` | low-draft-improvement                                              | Title: Sailboats Moored in Deben Estuary Description: Two sailboats are moored on a river with lush greenery in the background. The boats are equipped with masts and rigging, indicating they are ready for sai ... [tail] bank, Nature, Lush, Greenery, Mudflat, Peaceful, Outdoors, Coastal, Sailing, Mudflats, Forest, Birds             |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `avoid` / `cutoff degraded`   | repetitive; low-draft-improvement; cutoff                          | Title: - Sailboats, River, Woodbridge, England, Estuary, Trees, Boating, Boats, Moored, Buoy, Landscape, Nature, Riverbank, Foliage, Coast, Landscape, Estuary, River, Nature, Boats, Trees, Landscape, Boating, ... [tail] iverbank, Foliage, Coast, Landscape, Boating, Boats, Moored, Buoy, Landscape, Nature, Estuary, River             |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `avoid` / `semantic mismatch` | low-draft-improvement                                              | Title: Two sailboats moored on a river Description: Two sailboats with orange masts are moored on a calm river, surrounded by dense green foliage on the riverbank. Keywords: Boat, Boating, Buoy, Bushes, Coast, Estuary, Foliage, Forest, Mast, Moored, Nature, Outdoors, River, Riverbank, Sailboat                                       |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `avoid` / `semantic mismatch` | missing terms: Bird, Boating, Bushes, Coast, Estuary               | Title: Sailboat Anchored Near Wooded Riverbank Description: A traditional wooden sailboat with two tall masts and red sails is moored on calm water, surrounded by dense green foliage and trees along the river ... [tail] n hull, two masts, red sails, river, dense foliage, green trees, calm water, moored, buoy, riverbank             |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `avoid` / `semantic mismatch` | low-draft-improvement                                              | Title: Sailboats Moored on River with Trees Description: Two sailboats are moored on a calm river, with dense green trees lining the riverbank behind them. Keywords: Sailboat, River, Trees, Moored, Masts, Sails, Buoy, Calm Water, Green, Nature, Outdoors, Riverbank, Woodland, Boat, Sailing, Peaceful, Estuary, Foliage                |
| [`mlx-community/paligemma2-10b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-10b-ft-docci-448-bf16)               | `avoid` / `model shortcoming` | low-draft-improvement; context-ignored; missing-sections           | - Do not copy prompt instructions into the Title, Description, or Keywords fields.                                                                                                                                                                                                                                                           |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `caveat` / `clean`            | low-draft-improvement; formatting                                  | Title: Sailing Boats on the Deben Estuary, Woodbridge, England, UK, GBR, Europe Description: Two sailing boats are moored on a river with trees behind on the bank. The boats have masts and rigging. The water ... [tail] UK, GBR, Europe, River, Trees, Bank, Masts, Rigging, Calm Water, Sky Not Visible.<end_of_utterance>               |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `recommended` / `clean`       | low-draft-improvement                                              | Title: - Two sailing boats moored on the Deben Estuary Description: - Two wooden sailboats with tall masts are anchored on a river next to a dense green forest bank. A small buoy floats near the vessels in th ... [tail] foliage, forest, landscape, mast, moored, mud, nature, outdoors, rigging, river, riverbank, sailboat             |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `avoid` / `semantic mismatch` | missing terms: Bird, Boating, Bushes, Coast, Landscape             | Title: Traditional Wooden Sailboat in Forested Riverbank Setting Description: A classic wooden sailboat with tall masts and visible rigging lines is moored on calm water beside a dense, green forest. The boat ... [tail] en foliage, tall mast, rigging, buoy, moored vessel, nature scene, tranquil waters, maritime setting             |
| [`mlx-community/SmolVLM-Instruct-bf16`](#model-mlx-community-smolvlm-instruct-bf16)                                     | `avoid` / `model shortcoming` | low-draft-improvement; context-ignored; missing-sections           | Deben Estuary, Woodbridge, England, UK, GBR, Europe.                                                                                                                                                                                                                                                                                         |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `avoid` / `model shortcoming` | unverified-context-copy; low-draft-improvement; metadata-borrowing | Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe Description: Two sailing boats moored on a river with trees behind on the bank Keywords: Bird, Boat, Boating, Buoy, Bushes, Coast, Estuary, Foliage, ... [tail] draft as fallible. Retain supported details, correct errors, and add important visible information.               |
| [`mlx-community/pixtral-12b-bf16`](#model-mlx-community-pixtral-12b-bf16)                                               | `avoid` / `semantic mismatch` | low-draft-improvement                                              | Title: Sailboats Moored on River with Trees Description: Two sailboats are moored on a calm river, with dense green trees lining the riverbank behind them. Keywords: Sailboat, River, Trees, Moored, Masts, Sails, Buoy, Calm Water, Green, Nature, Outdoors, Riverbank, Woodland, Boat, Sailing, Peaceful, Estuary                         |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `avoid` / `semantic mismatch` | missing terms: Bird, Boating, Bushes, Coast, Forest                | **Title:** Sailboat Moored on Deben Estuary **Description:** A classic wooden sailboat with two masts is anchored near a riverbank, surrounded by dense green foliage. Visible details include red sails, wooden ... [tail] loating buoy, Classic boat, Deben Estuary, Moored vessel, Green landscape, Woodbridge, Rigging lines             |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `recommended` / `clean`       | low-draft-improvement                                              | Title: Sailing boats moored on the Deben Estuary, England Description: Two sailing boats with wooden masts are moored on a river in front of a dense green forest. A small buoy floats in the foreground on the ... [tail] est, trees, foliage, water, buoy, moored, Deben Estuary, England, nature, outdoors, landscape, green              |
| [`mlx-community/InternVL3-14B-8bit`](#model-mlx-community-internvl3-14b-8bit)                                           | `avoid` / `semantic mismatch` | low-draft-improvement                                              | Title: Sailing Boats on Deben Estuary Description: Two sailboats are moored on the Deben Estuary, with lush green trees lining the riverbank in the background. The boats have masts and rigging, and one has a ... [tail] ndscape, Mast, Moored, Nature, Outdoors, Peaceful, River, Riverbank, Sailboat, Trees, UK, Woodbridge              |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `avoid` / `semantic mismatch` | low-draft-improvement                                              | Title: Sailboat on River Description: A wooden sailboat with two masts is moored on a calm river, surrounded by dense green trees on the riverbank. The boat has red sails and a small duck swimming nearby. Key ... [tail] ts, moored, duck, calm water, green foliage, nature, outdoors, peaceful, rigging, riverbank, estuary             |
| [`mlx-community/gemma-3n-E2B-4bit`](#model-mlx-community-gemma-3n-e2b-4bit)                                             | `avoid` / `cutoff degraded`   | repetitive; cutoff; context-ignored                                | - Do not take it out of the. - Do not take it out of the. - Do not take it out of the. - Do not take it out of the. - Do not take it out of the. - Do not take it out of the. - Do not take it out of the. - Do not take it out of the. - Do not take it out of the. - Do not take it out of the. - Do not take it out of the...             |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `avoid` / `semantic mismatch` | low-draft-improvement                                              | Title: - Two sailing boats moored on the Deben Estuary Description: - Two small wooden sailboats are moored on a river against a backdrop of dense green forest trees. One boat features a red sail cover while ... [tail] stuary, Foliage, Forest, Green,, Mast, Moored, Nature, Outdoors, Rigging, River, Riverbank, Sailboat              |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `avoid` / `semantic mismatch` | low-draft-improvement                                              | Title: Sailboats Moored in Deben Estuary Description: Two sailboats are moored on a river with lush greenery in the background. The boats are equipped with masts and rigging, indicating they are ready for sai ... [tail] bank, Nature, Lush, Greenery, Mudflat, Peaceful, Outdoors, Coastal, Sailing, Mudflats, Forest, Birds             |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `avoid` / `model shortcoming` | unverified-context-copy; metadata-borrowing                        | ## Image Description for Cataloguing **Title:** Two Sailboats Moored on the Deben Estuary **Description:** Two small sailboats are moored side-by-side on a calm body of water, likely the Deben Estuary in Engl ... [tail] ar and well-defined, with all elements readily visible. There are no obscured or indistinct details.             |
| [`mlx-community/gemma-3-27b-it-qat-8bit`](#model-mlx-community-gemma-3-27b-it-qat-8bit)                                 | `caveat` / `clean`            | low-draft-improvement                                              | Title: Two sailing boats moored on the river Deben Description: Two sailing boats are moored on calm water, with dense green trees lining the riverbank behind them. The boats have masts and rigging visible ag ... [tail] ast, Moored, Nature, Outdoors, River, Riverbank, Sailboat, Trees, Water, Calm, Green, Masts, Rigging             |
| [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](#model-jqlive-kimi-vl-a3b-thinking-2506-6bit)                                 | `avoid` / `cutoff degraded`   | low-draft-improvement; cutoff; reasoning-leak                      | ◁think▷Got it, let's analyze the image step by step. First, the title: The main visible elements are two sailboats on a body of water with trees in the background. So the title should be concise. Let's check ... [tail] it location unless visible. So keywords from image: Sailboats, river, trees, water, foliage, buoy, m              |
| [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](#model-mlx-community-kimi-vl-a3b-thinking-8bit)                             | `avoid` / `cutoff degraded`   | low-draft-improvement; cutoff; reasoning-leak                      | ◁think▷Okay, let's tackle this metadata cataloging task. First, I need to analyze the image based on the given rules. The user wants me to extract metadata using British English, focusing only on clearly visi ... [tail] unique terms. So keywords: Sailboats, Mast, Rigging, Moored, River, Riverbank, Trees, Foliage, Buoy              |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)            | `caveat` / `clean`            | low-draft-improvement                                              | Title: Two Sailing Boats Moored on a River with Trees Behind Description: The image depicts two sailing boats moored on a river, with a dense forest of trees in the background. The boats are positioned in the ... [tail] t, Landscape, Mast, Moored, Mudflat, Nature, Outdoors, Peaceful, Rigging, River, Riverbank, Sailboat             |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `caveat` / `clean`            | low-draft-improvement                                              | Title: Two sailing boats moored on a riverbank Description: Two sailing boats are moored in calm water, with dense green trees forming the backdrop. A single orange buoy is visible in the foreground, and the ... [tail] Mudflat, Nature, Outdoors, River, Riverbank, Sailboat, Trees, Water, Reflection, Calm, Green, Orange              |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `avoid` / `cutoff degraded`   | low-draft-improvement; cutoff; missing-sections                    | Let's tackle this step by step. First, the title needs to be 5-10 words, concrete, and factual. The image has a sailboat in a wooded area by a river. Wait, the existing title is "Deben Estuary, Woodbridge, En ... [tail] on the bank" but let's correct: Are there two boats? The context says "a sailboat and a wooden boat              |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                            | `caveat` / `token cap`        | unverified-context-copy; low-draft-improvement; metadata-borrowing | Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe Description: Two sailing boats moored on a river with trees behind on the bank Keywords: Bird, Boat, Boating, Buoy, Bushes, Coast, Estuary, Foliage, ... [tail] ure, Outdoors, Peaceful, Rigging, River, Riverbank, Sailboat Metadata hints: - Title: Deben Estuary,              |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                             | `avoid` / `cutoff degraded`   | repetitive; cutoff; context-ignored                                | - Output only the description text after the label. - Output only the keyword list after the label. - Output only the description text after the label. - Output only the keyword list after the label. - Output only the description text after the label. - Output only the keyword list after the label. - Output only the...             |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `avoid` / `cutoff degraded`   | repetitive; cutoff                                                 | Title: Two sailing boats moored on a river Description: Two sailing boats are moored on a river. The boats have orange masts and red covers. The riverbank is covered with green foliage and trees. A buoy is vi ... [tail] over, Black hull, White superstructure, Red sail cover, Orange mast, White rigging, Red cover, Black             |
| [`mlx-community/GLM-4.6V-Flash-6bit`](#model-mlx-community-glm-46v-flash-6bit)                                          | `caveat` / `context budget`   | context-budget; missing-sections                                   | <\|begin_of_box\|>Title: Two sailboats moored on water with trees in background Description: Two sailboats with orange masts and red covers are moored on calm water. Behind them is a dense area of green trees a ... [tail] Trees, Green foliage, Bank, Calm water, Mast, Rigging, Boat, Estuary, Nature, Outdoors<\|end_of_box\|>         |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `avoid` / `cutoff degraded`   | cutoff; reasoning-leak; missing-sections                           | Here are my reasoning steps: We need to produce a catalog entry with Title, Description, Keywords. Use only details that are clearly visible. The image shows a river (likely Deben Estuary) with two sailing bo ... [tail] ting, lighting, action, and other distinctive visible details. So something like: "Two sailing boats             |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `caveat` / `context budget`   | low-draft-improvement; context-budget; missing-sections            | <\|begin_of_box\|>Title: Two sailing boats moored on a river with trees behind Description: Two sailing boats are moored on a river, with dense green trees and foliage visible on the riverbank behind them. The ... [tail] oored, Mudflat, Nature, Outdoors, Peaceful, Rigging, River, Riverbank, Sailboat, Trees<\|end_of_box\|>          |
| [`mlx-community/Molmo-7B-D-0924-bf16`](#model-mlx-community-molmo-7b-d-0924-bf16)                                       | `caveat` / `clean`            | low-draft-improvement                                              | Remove uncertain tags. Title: Sailing boats on the River Deben near Woodbridge, England Description: Two sailboats are moored on the River Deben with trees and foliage visible on the riverbank behind them. Th ... [tail] t, Landscape, Mast, Moored, Mudflat, Nature, Outdoors, Peaceful, Rigging, River, Riverbank, Sailboat             |
| [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16)                             | `avoid` / `cutoff degraded`   | harness:long-context; repetitive; low-draft-improvement            | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -...             |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                           | `caveat` / `clean`            | low-draft-improvement                                              | Title: Two moored sailboats on a river with forested bank Description: Two sailboats are anchored on a calm river, with a dense forest visible on the opposite bank. The boats have masts and rigging, but no sa ... [tail] t, Landscape, Mast, Moored, Mudflat, Nature, Outdoors, Peaceful, Rigging, River, Riverbank, Sailboat             |
| [`mlx-community/Molmo-7B-D-0924-8bit`](#model-mlx-community-molmo-7b-d-0924-8bit)                                       | `caveat` / `clean`            | low-draft-improvement                                              | Remove uncertain tags. Title: Sailing boats on the River Deben near Woodbridge, England Description: Two sailboats are moored on the River Deben with trees and foliage visible on the riverbank behind them. Th ... [tail] t, Landscape, Mast, Moored, Mudflat, Nature, Outdoors, Peaceful, Rigging, River, Riverbank, Sailboat             |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `avoid` / `cutoff degraded`   | harness:long-context; repetitive; low-draft-improvement            | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -...             |
| [`mlx-community/paligemma2-3b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-3b-ft-docci-448-bf16)                 | `avoid` / `cutoff degraded`   | repetitive; cutoff; context-ignored                                | - Do not copy the text after the label. - Do not infer the intent of the image. - Use this factual context where it improves the catalogue record; do not claim that contextual facts are visually observable. - ... [tail] the catalogue record; do not claim that contextual facts are visually observable. - Do not copy the              |
| [`meta-llama/Llama-3.2-11B-Vision-Instruct`](#model-meta-llama-llama-32-11b-vision-instruct)                            | `avoid` / `model shortcoming` | low-draft-improvement; missing-sections                            | Sailboats Moored in the Deben Estuary The image depicts two sailboats moored in the Deben Estuary, with a serene and peaceful atmosphere. The boats are situated in the water, with a dense forest of trees in t ... [tail] tranquility and serenity, with the sailboats and the forest creating a sense of harmony and balance.             |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `caveat` / `context budget`   | harness:long-context; low-draft-improvement; context-budget        | <\|im_start\|> addCriterion("图片描述").get("sailboat")                                                                                                                                                                                                                                                                                      |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `avoid` / `semantic mismatch` | missing terms: Bird, Boating, Bushes, Coast, Mudflat               | Title: Two sailboats moored on the Deben Estuary with a wooded bank Description: Two sailboats with orange masts are moored on the water of the Deben Estuary, featuring a dark-hulled vessel in the foreground ... [tail] ank, Mast, Rigging, Buoy, Foliage, Trees, Water, Nature, Outdoors, Landscape, Estuary, Boat, Foliage              |
| [`mlx-community/Qwen3.5-35B-A3B-6bit`](#model-mlx-community-qwen35-35b-a3b-6bit)                                        | `avoid` / `semantic mismatch` | missing terms: Bird, Boat, Boating, Bushes, Coast                  | Title: Two moored sailboats on the Deben Estuary with red covers Description: Two sailing yachts with tall wooden masts are moored on the water, featuring red canvas covers over their booms. A dense green tre ... [tail] rbank, trees, foliage, water, buoys, rigging, estuary, nature, outdoors, calm, green, UK, Woodbridge             |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `avoid` / `cutoff degraded`   | harness:long-context; repetitive; low-draft-improvement            | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -...             |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `avoid` / `semantic mismatch` | missing terms: Bird, Boating, Bushes, Coast, Estuary               | Title: Two moored sailboats on a riverbank Description: Two small sailing boats are moored side by side on calm water, with their sails furled and masts visible. The background consists of a densely wooded ri ... [tail] , River, Riverbank, Trees, Forest, Greenery, Nature, Outdoors, Mast, Rigging, Buoy, Water, Landscape             |
| [`mlx-community/Qwen3.5-35B-A3B-bf16`](#model-mlx-community-qwen35-35b-a3b-bf16)                                        | `avoid` / `semantic mismatch` | low-draft-improvement                                              | Title: Two wooden sailboats moored on the Deben Estuary Description: Two wooden sailing yachts with tall masts are moored on calm water against a backdrop of dense green trees and a muddy riverbank. Keywords: ... [tail] moored, riverbank, trees, foliage, green, water, buoy, estuary, UK, nature, outdoors, calm, sailing              |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `avoid` / `semantic mismatch` | missing terms: Bird, Boating, Bushes, Coast, Landscape             | Title: Two wooden sailing boats moored on the Deben Estuary Description: Two wooden sailing boats with tall masts are moored in the water, backed by a dense green riverbank. A buoy floats in the foreground wh ... [tail] pe, Sailboat, Boat, Moored, River, Water, Trees, Foliage, Forest, Green, Wooden, Mast, Rigging, Buoy             |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                             | `avoid` / `cutoff degraded`   | harness:long-context; repetitive; low-draft-improvement            | -001.jpg - 10-18 unique comma-separated terms based only on clearly visible subjects, setting, lighting, action, and other distinctive visible details. Omit anything uncertain or inferred. - Output only the k ... [tail] s, Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats,             |
| [`mlx-community/Qwen3.5-27B-mxfp8`](#model-mlx-community-qwen35-27b-mxfp8)                                              | `avoid` / `semantic mismatch` | missing terms: Bird, Boating, Bushes, Coast, Foliage               | Title: Two Moored Sailboats on River with Forested Bank Description: Two sailboats with wooden masts are moored on calm water, positioned in front of a densely vegetated riverbank. The foreground boat feature ... [tail] , Rigging, Hull, Water, Nature, Landscape, Vegetation, Buoy, Deck, Estuary, Wood, Greenery, Outdoors             |
| [`mlx-community/Qwen3.5-27B-4bit`](#model-mlx-community-qwen35-27b-4bit)                                                | `avoid` / `semantic mismatch` | missing terms: Bird, Boating, Bushes, Coast, Mudflat               | Title: Two sailboats moored on river with forested bank Description: Two sailboats with furled sails are moored on calm water in front of a densely vegetated riverbank. The foreground vessel has a dark hull w ... [tail] y, Mast, Rigging, Hull, Water, Vegetation, Trees, Nature, Landscape, Sailing, Boat, Estuary, Foliage             |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `avoid` / `cutoff degraded`   | low-draft-improvement; cutoff; reasoning-leak                      | ◁think▷Got it, let's analyze the image step by step. First, the title: The image shows two sailboats on a body of water with dense green trees in the background. The location is Deben Estuary, England, UK, Eu ... [tail] isible as a keyword. Let's stick to what's visible. So final keywords: Sailboats, River, Trees, Foli             |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `recommended` / `clean`       | clean                                                              | Title: Two sailing boats moored on a river Description: Two sailing boats are moored on calm water, with the foreground vessel featuring a black hull, wooden transom, and tall orange masts. A dense backdrop o ... [tail] Foliage, Nature, Landscape, Mast, Rigging, Hull, Buoy, Outdoors, Green, Wood, Vessel, Estuary, Bank              |
<!-- markdownlint-enable MD034 -->

## All Model Output and Cost Summary

Every model in this run, with its skim-first output or diagnostic beside the main runtime and peak-memory signals.

<!-- markdownlint-disable MD034 -->

| Model                                                                                                                   | Result                        | Output / diagnostic                                                                                                                                                                                                                     | Gen tok   | Total   | Gen TPS   | Peak GB   | Quality signal                                                     |
|-------------------------------------------------------------------------------------------------------------------------|-------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|---------|-----------|-----------|--------------------------------------------------------------------|
| [`mlx-community/gemma-4-31b-bf16`](#model-mlx-community-gemma-4-31b-bf16)                                               | `avoid` / `semantic mismatch` | [mlx; model-error] Error: Model Error - Model runtime error during generation for mlx-community/gemma-4-31b-bf16: [METAL] Command buffer execution fa ... [tail] ory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory).            | -         | 135.96s | -         | -         | mlx; model-error                                                   |
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `caveat` / `clean`            | Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe, Boating, Sailboat, River, Riverbank, Nature, Outdoors, Peaceful, Rigging, Boats, Mu ... [tail] Outdoors, Peaceful, Rigging, Boats, Mushrooms, Trees, Water, Sky             | 181       | 1.55s   | 504       | 1.4       | missing terms: Bird, Buoy, Mudflat, behind, bank                   |
| [`mlx-community/LFM2-VL-1.6B-8bit`](#model-mlx-community-lfm2-vl-16b-8bit)                                              | `avoid` / `model shortcoming` | Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe, 2026-07-04 19:10:04 BST 19:10:04 Description: Two sailing boats moored on a river w ... [tail] , Nature, Outdoors, Peaceful, Rigging, River, Riverbank, Sailboat            | 144       | 1.91s   | 331       | 3.0       | unverified-context-copy; low-draft-improvement; metadata-borrowing |
| [`qnguyen3/nanoLLaVA`](#model-qnguyen3-nanollava)                                                                       | `avoid` / `model shortcoming` | Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe Description: Two sailing boats moored on a river with trees behind on the bank.                                                                                              | 37        | 2.06s   | 113       | 5.1       | low-draft-improvement; missing-sections                            |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                            | `caveat` / `clean`            | Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe Description: Two sailing boats moored on a river with trees behind on the bank Keywo ... [tail] Rigging, River, Riverbank, Sailboat Do you need more information?            | 98        | 2.18s   | 192       | 2.8       | low-draft-improvement                                              |
| [`HuggingFaceTB/SmolVLM-Instruct`](#model-huggingfacetb-smolvlm-instruct)                                               | `avoid` / `model shortcoming` | Deben Estuary, Woodbridge, England, UK, GBR, Europe.                                                                                                                                                                                    | 18        | 2.27s   | 127       | 5.6       | low-draft-improvement; context-ignored; missing-sections           |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `avoid` / `harness`           | &lt;think&gt; &lt;/think&gt; Title: Sailing boats moored near dense foliage on water Description: The image shows two sailing boats floating on calm water ... [tail] moored, calm, green, river, nature, peaceful, mast, rigging, buoy | 81        | 2.69s   | 181       | 4.0       | harness:stop-token; reasoning-leak; formatting                     |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `avoid` / `model shortcoming` | Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe Description: Two sailing boats moored on a river with trees behind on the bank Keywo ... [tail] cord; do not claim that contextual facts are visually observable.            | 171       | 2.95s   | 292       | 2.2       | unverified-context-copy; low-draft-improvement; metadata-borrowing |
| [`mlx-community/paligemma2-10b-ft-docci-448-6bit`](#model-mlx-community-paligemma2-10b-ft-docci-448-6bit)               | `avoid` / `model shortcoming` | - Do not copy prompt instructions into the Title, Description, or Keywords fields.                                                                                                                                                      | 17        | 4.25s   | 30.5      | 12        | low-draft-improvement; context-ignored; missing-sections           |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `avoid` / `semantic mismatch` | Title: Two small sailboats moored on the Deben Estuary Description: Two small sailboats with wooden masts are moored on the water against a bac ... [tail] nature, outdoors, landscape, mooring, estuary, vertical, daylight            | 86        | 5.57s   | 56.7      | 18        | missing terms: Bird, Boating, Buoy, Bushes, Coast                  |
| [`microsoft/Phi-3.5-vision-instruct`](#model-microsoft-phi-35-vision-instruct)                                          | `avoid` / `semantic mismatch` | Title: Sailboats Moored in Deben Estuary Description: Two sailboats are moored on a river with lush greenery in the background. The boats are e ... [tail] at, Peaceful, Outdoors, Coastal, Sailing, Mudflats, Forest, Birds            | 130       | 4.65s   | 57.6      | 9.5       | low-draft-improvement                                              |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `avoid` / `cutoff degraded`   | Title: - Sailboats, River, Woodbridge, England, Estuary, Trees, Boating, Boats, Moored, Buoy, Landscape, Nature, Riverbank, Foliage, Coast, Lan ... [tail] , Boating, Boats, Moored, Buoy, Landscape, Nature, Estuary, River            | 500       | 4.37s   | 175       | 4.1       | repetitive; low-draft-improvement; cutoff                          |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `avoid` / `semantic mismatch` | Title: Two sailboats moored on a river Description: Two sailboats with orange masts are moored on a calm river, surrounded by dense green folia ... [tail] orest, Mast, Moored, Nature, Outdoors, River, Riverbank, Sailboat            | 81        | 5.82s   | 35.3      | 18        | low-draft-improvement                                              |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `avoid` / `semantic mismatch` | Title: Sailboat Anchored Near Wooded Riverbank Description: A traditional wooden sailboat with two tall masts and red sails is moored on calm w ... [tail] , dense foliage, green trees, calm water, moored, buoy, riverbank            | 105       | 6.26s   | 66.1      | 13        | missing terms: Bird, Boating, Bushes, Coast, Estuary               |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `avoid` / `semantic mismatch` | Title: Sailboats Moored on River with Trees Description: Two sailboats are moored on a calm river, with dense green trees lining the riverbank ... [tail] s, Riverbank, Woodland, Boat, Sailing, Peaceful, Estuary, Foliage             | 92        | 7.02s   | 39.4      | 16        | low-draft-improvement                                              |
| [`mlx-community/paligemma2-10b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-10b-ft-docci-448-bf16)               | `avoid` / `model shortcoming` | - Do not copy prompt instructions into the Title, Description, or Keywords fields.                                                                                                                                                      | 17        | 7.83s   | 5.63      | 26        | low-draft-improvement; context-ignored; missing-sections           |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `caveat` / `clean`            | Title: Sailing Boats on the Deben Estuary, Woodbridge, England, UK, GBR, Europe Description: Two sailing boats are moored on a river with trees ... [tail] k, Masts, Rigging, Calm Water, Sky Not Visible.<end_of_utterance>            | 109       | 7.64s   | 33.3      | 19        | low-draft-improvement; formatting                                  |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `recommended` / `clean`       | Title: - Two sailing boats moored on the Deben Estuary Description: - Two wooden sailboats with tall masts are anchored on a river next to a de ... [tail] oored, mud, nature, outdoors, rigging, river, riverbank, sailboat            | 89        | 9.35s   | 20.6      | 29        | low-draft-improvement                                              |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `avoid` / `semantic mismatch` | Title: Traditional Wooden Sailboat in Forested Riverbank Setting Description: A classic wooden sailboat with tall masts and visible rigging lin ... [tail] y, moored vessel, nature scene, tranquil waters, maritime setting            | 125       | 7.66s   | 64.9      | 7.8       | missing terms: Bird, Boating, Bushes, Coast, Landscape             |
| [`mlx-community/SmolVLM-Instruct-bf16`](#model-mlx-community-smolvlm-instruct-bf16)                                     | `avoid` / `model shortcoming` | Deben Estuary, Woodbridge, England, UK, GBR, Europe.                                                                                                                                                                                    | 18        | 10.10s  | 38.6      | 5.7       | low-draft-improvement; context-ignored; missing-sections           |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `avoid` / `model shortcoming` | Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe Description: Two sailing boats moored on a river with trees behind on the bank Keywo ... [tail] d details, correct errors, and add important visible information.            | 275       | 9.98s   | 120       | 5.5       | unverified-context-copy; low-draft-improvement; metadata-borrowing |
| [`mlx-community/pixtral-12b-bf16`](#model-mlx-community-pixtral-12b-bf16)                                               | `avoid` / `semantic mismatch` | Title: Sailboats Moored on River with Trees Description: Two sailboats are moored on a calm river, with dense green trees lining the riverbank ... [tail] , Outdoors, Riverbank, Woodland, Boat, Sailing, Peaceful, Estuary             | 88        | 9.67s   | 20.7      | 28        | low-draft-improvement                                              |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `avoid` / `semantic mismatch` | **Title:** Sailboat Moored on Deben Estuary **Description:** A classic wooden sailboat with two masts is anchored near a riverbank, surrounded ... [tail] stuary, Moored vessel, Green landscape, Woodbridge, Rigging lines             | 108       | 8.86s   | 26.5      | 13        | missing terms: Bird, Boating, Bushes, Coast, Forest                |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `recommended` / `clean`       | Title: Sailing boats moored on the Deben Estuary, England Description: Two sailing boats with wooden masts are moored on a river in front of a ... [tail] oored, Deben Estuary, England, nature, outdoors, landscape, green             | 90        | 9.98s   | 19.6      | 22        | low-draft-improvement                                              |
| [`mlx-community/InternVL3-14B-8bit`](#model-mlx-community-internvl3-14b-8bit)                                           | `avoid` / `semantic mismatch` | Title: Sailing Boats on Deben Estuary Description: Two sailboats are moored on the Deben Estuary, with lush green trees lining the riverbank in ... [tail] oors, Peaceful, River, Riverbank, Sailboat, Trees, UK, Woodbridge            | 109       | 9.05s   | 23.6      | 18        | low-draft-improvement                                              |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `avoid` / `semantic mismatch` | Title: Sailboat on River Description: A wooden sailboat with two masts is moored on a calm river, surrounded by dense green trees on the riverb ... [tail] foliage, nature, outdoors, peaceful, rigging, riverbank, estuary             | 96        | 9.70s   | 30.4      | 23        | low-draft-improvement                                              |
| [`mlx-community/gemma-3n-E2B-4bit`](#model-mlx-community-gemma-3n-e2b-4bit)                                             | `avoid` / `cutoff degraded`   | - Do not take it out of the. - Do not take it out of the. - Do not take it out of the. - Do not take it out of the. - Do not take it out of the. - Do not take it out of the. - Do not take it out of the. - Do not take...             | 500       | 9.13s   | 77.6      | 6.0       | repetitive; cutoff; context-ignored                                |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `avoid` / `semantic mismatch` | Title: - Two sailing boats moored on the Deben Estuary Description: - Two small wooden sailboats are moored on a river against a backdrop of de ... [tail] st, Moored, Nature, Outdoors, Rigging, River, Riverbank, Sailboat            | 96        | 11.37s  | 15.0      | 28        | low-draft-improvement                                              |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `avoid` / `semantic mismatch` | Title: Sailboats Moored in Deben Estuary Description: Two sailboats are moored on a river with lush greenery in the background. The boats are e ... [tail] at, Peaceful, Outdoors, Coastal, Sailing, Mudflats, Forest, Birds            | 130       | 9.22s   | 19.1      | 9.5       | low-draft-improvement                                              |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `avoid` / `model shortcoming` | ## Image Description for Cataloguing **Title:** Two Sailboats Moored on the Deben Estuary **Description:** Two small sailboats are moored side- ... [tail] nts readily visible. There are no obscured or indistinct details.            | 332       | 10.81s  | 46.8      | 17        | unverified-context-copy; metadata-borrowing                        |
| [`mlx-community/gemma-3-27b-it-qat-8bit`](#model-mlx-community-gemma-3-27b-it-qat-8bit)                                 | `caveat` / `clean`            | Title: Two sailing boats moored on the river Deben Description: Two sailing boats are moored on calm water, with dense green trees lining the r ... [tail] r, Riverbank, Sailboat, Trees, Water, Calm, Green, Masts, Rigging            | 108       | 11.81s  | 17.7      | 32        | low-draft-improvement                                              |
| [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](#model-jqlive-kimi-vl-a3b-thinking-2506-6bit)                                 | `avoid` / `cutoff degraded`   | ◁think▷Got it, let's analyze the image step by step. First, the title: The main visible elements are two sailboats on a body of water with tree ... [tail] ords from image: Sailboats, river, trees, water, foliage, buoy, m            | 500       | 11.69s  | 66.7      | 18        | low-draft-improvement; cutoff; reasoning-leak                      |
| [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](#model-mlx-community-kimi-vl-a3b-thinking-8bit)                             | `avoid` / `cutoff degraded`   | ◁think▷Okay, let's tackle this metadata cataloging task. First, I need to analyze the image based on the given rules. The user wants me to extr ... [tail] ts, Mast, Rigging, Moored, River, Riverbank, Trees, Foliage, Buoy            | 500       | 12.63s  | 61.5      | 22        | low-draft-improvement; cutoff; reasoning-leak                      |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)            | `caveat` / `clean`            | Title: Two Sailing Boats Moored on a River with Trees Behind Description: The image depicts two sailing boats moored on a river, with a dense f ... [tail] , Nature, Outdoors, Peaceful, Rigging, River, Riverbank, Sailboat            | 148       | 12.03s  | 19.0      | 15        | low-draft-improvement                                              |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `caveat` / `clean`            | Title: Two sailing boats moored on a riverbank Description: Two sailing boats are moored in calm water, with dense green trees forming the back ... [tail] iverbank, Sailboat, Trees, Water, Reflection, Calm, Green, Orange            | 105       | 13.13s  | 18.9      | 18        | low-draft-improvement                                              |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `avoid` / `cutoff degraded`   | Let's tackle this step by step. First, the title needs to be 5-10 words, concrete, and factual. The image has a sailboat in a wooded area by a ... [tail] e there two boats? The context says "a sailboat and a wooden boat             | 500       | 15.93s  | 57.4      | 60        | low-draft-improvement; cutoff; missing-sections                    |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                            | `caveat` / `token cap`        | Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe Description: Two sailing boats moored on a river with trees behind on the bank Keywo ... [tail] iver, Riverbank, Sailboat Metadata hints: - Title: Deben Estuary,            | 500       | 12.43s  | 59.9      | 9.7       | unverified-context-copy; low-draft-improvement; metadata-borrowing |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                             | `avoid` / `cutoff degraded`   | - Output only the description text after the label. - Output only the keyword list after the label. - Output only the description text after the label. - Output only the keyword list after the label. - Output only the...            | 500       | 13.61s  | 50.0      | 4.6       | repetitive; cutoff; context-ignored                                |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `avoid` / `cutoff degraded`   | Title: Two sailing boats moored on a river Description: Two sailing boats are moored on a river. The boats have orange masts and red covers. Th ... [tail] ure, Red sail cover, Orange mast, White rigging, Red cover, Black            | 500       | 21.21s  | 80.0      | 8.4       | repetitive; cutoff                                                 |
| [`mlx-community/GLM-4.6V-Flash-6bit`](#model-mlx-community-glm-46v-flash-6bit)                                          | `caveat` / `context budget`   | <\|begin_of_box\|>Title: Two sailboats moored on water with trees in background Description: Two sailboats with orange masts and red covers are m ... [tail] ter, Mast, Rigging, Boat, Estuary, Nature, Outdoors<\|end_of_box\|>        | 96        | 17.55s  | 15.2      | 11        | context-budget; missing-sections                                   |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `avoid` / `cutoff degraded`   | Here are my reasoning steps: We need to produce a catalog entry with Title, Description, Keywords. Use only details that are clearly visible. T ... [tail] istinctive visible details. So something like: "Two sailing boats            | 500       | 17.93s  | 38.8      | 15        | cutoff; reasoning-leak; missing-sections                           |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `caveat` / `context budget`   | <\|begin_of_box\|>Title: Two sailing boats moored on a river with trees behind Description: Two sailing boats are moored on a river, with dense g ... [tail] eaceful, Rigging, River, Riverbank, Sailboat, Trees<\|end_of_box\|>        | 112       | 21.86s  | 32.7      | 78        | low-draft-improvement; context-budget; missing-sections            |
| [`mlx-community/Molmo-7B-D-0924-bf16`](#model-mlx-community-molmo-7b-d-0924-bf16)                                       | `caveat` / `clean`            | Remove uncertain tags. Title: Sailing boats on the River Deben near Woodbridge, England Description: Two sailboats are moored on the River Debe ... [tail] , Nature, Outdoors, Peaceful, Rigging, River, Riverbank, Sailboat            | 114       | 21.87s  | 30.3      | 48        | low-draft-improvement                                              |
| [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16)                             | `avoid` / `cutoff degraded`   | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -...            | 500       | 22.25s  | 91.9      | 8.6       | harness:long-context; repetitive; low-draft-improvement            |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                           | `caveat` / `clean`            | Title: Two moored sailboats on a river with forested bank Description: Two sailboats are anchored on a calm river, with a dense forest visible ... [tail] , Nature, Outdoors, Peaceful, Rigging, River, Riverbank, Sailboat             | 106       | 23.94s  | 5.95      | 27        | low-draft-improvement                                              |
| [`mlx-community/Molmo-7B-D-0924-8bit`](#model-mlx-community-molmo-7b-d-0924-8bit)                                       | `caveat` / `clean`            | Remove uncertain tags. Title: Sailing boats on the River Deben near Woodbridge, England Description: Two sailboats are moored on the River Debe ... [tail] , Nature, Outdoors, Peaceful, Rigging, River, Riverbank, Sailboat            | 114       | 28.52s  | 50.9      | 41        | low-draft-improvement                                              |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `avoid` / `cutoff degraded`   | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -...            | 500       | 22.84s  | 92.5      | 8.6       | harness:long-context; repetitive; low-draft-improvement            |
| [`mlx-community/paligemma2-3b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-3b-ft-docci-448-bf16)                 | `avoid` / `cutoff degraded`   | - Do not copy the text after the label. - Do not infer the intent of the image. - Use this factual context where it improves the catalogue reco ... [tail] that contextual facts are visually observable. - Do not copy the             | 500       | 29.49s  | 18.9      | 11        | repetitive; cutoff; context-ignored                                |
| [`meta-llama/Llama-3.2-11B-Vision-Instruct`](#model-meta-llama-llama-32-11b-vision-instruct)                            | `avoid` / `model shortcoming` | Sailboats Moored in the Deben Estuary The image depicts two sailboats moored in the Deben Estuary, with a serene and peaceful atmosphere. The b ... [tail] sailboats and the forest creating a sense of harmony and balance.            | 134       | 45.18s  | 3.35      | 25        | low-draft-improvement; missing-sections                            |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `caveat` / `context budget`   | <\|im_start\|> addCriterion("图片描述").get("sailboat")                                                                                                                                                                                 | 14        | 60.31s  | 46.4      | 13        | harness:long-context; low-draft-improvement; context-budget        |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `avoid` / `semantic mismatch` | Title: Two sailboats moored on the Deben Estuary with a wooded bank Description: Two sailboats with orange masts are moored on the water of the ... [tail] Trees, Water, Nature, Outdoors, Landscape, Estuary, Boat, Foliage            | 137       | 64.68s  | 56.1      | 76        | missing terms: Bird, Boating, Bushes, Coast, Mudflat               |
| [`mlx-community/Qwen3.5-35B-A3B-6bit`](#model-mlx-community-qwen35-35b-a3b-6bit)                                        | `avoid` / `semantic mismatch` | Title: Two moored sailboats on the Deben Estuary with red covers Description: Two sailing yachts with tall wooden masts are moored on the water ... [tail] , rigging, estuary, nature, outdoors, calm, green, UK, Woodbridge            | 113       | 62.95s  | 68.7      | 35        | missing terms: Bird, Boat, Boating, Bushes, Coast                  |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `avoid` / `cutoff degraded`   | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -...            | 500       | 68.35s  | 27.9      | 8.6       | harness:long-context; repetitive; low-draft-improvement            |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `avoid` / `semantic mismatch` | Title: Two moored sailboats on a riverbank Description: Two small sailing boats are moored side by side on calm water, with their sails furled ... [tail] Greenery, Nature, Outdoors, Mast, Rigging, Buoy, Water, Landscape             | 99        | 69.00s  | 91.4      | 11        | missing terms: Bird, Boating, Bushes, Coast, Estuary               |
| [`mlx-community/Qwen3.5-35B-A3B-bf16`](#model-mlx-community-qwen35-35b-a3b-bf16)                                        | `avoid` / `semantic mismatch` | Title: Two wooden sailboats moored on the Deben Estuary Description: Two wooden sailing yachts with tall masts are moored on calm water against ... [tail] green, water, buoy, estuary, UK, nature, outdoors, calm, sailing             | 93        | 75.08s  | 59.7      | 76        | low-draft-improvement                                              |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `avoid` / `semantic mismatch` | Title: Two wooden sailing boats moored on the Deben Estuary Description: Two wooden sailing boats with tall masts are moored in the water, back ... [tail] Water, Trees, Foliage, Forest, Green, Wooden, Mast, Rigging, Buoy            | 105       | 76.73s  | 68.2      | 26        | missing terms: Bird, Boating, Bushes, Coast, Landscape             |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                             | `avoid` / `cutoff degraded`   | -001.jpg - 10-18 unique comma-separated terms based only on clearly visible subjects, setting, lighting, action, and other distinctive visible ... [tail] s, Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats,             | 500       | 82.62s  | 207       | 5.1       | harness:long-context; repetitive; low-draft-improvement            |
| [`mlx-community/Qwen3.5-27B-mxfp8`](#model-mlx-community-qwen35-27b-mxfp8)                                              | `avoid` / `semantic mismatch` | Title: Two Moored Sailboats on River with Forested Bank Description: Two sailboats with wooden masts are moored on calm water, positioned in fr ... [tail] dscape, Vegetation, Buoy, Deck, Estuary, Wood, Greenery, Outdoors            | 131       | 103.01s | 18.3      | 38        | missing terms: Bird, Boating, Bushes, Coast, Foliage               |
| [`mlx-community/Qwen3.5-27B-4bit`](#model-mlx-community-qwen35-27b-4bit)                                                | `avoid` / `semantic mismatch` | Title: Two sailboats moored on river with forested bank Description: Two sailboats with furled sails are moored on calm water in front of a den ... [tail] tation, Trees, Nature, Landscape, Sailing, Boat, Estuary, Foliage            | 121       | 106.12s | 23.9      | 26        | missing terms: Bird, Boating, Bushes, Coast, Mudflat               |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `avoid` / `cutoff degraded`   | ◁think▷Got it, let's analyze the image step by step. First, the title: The image shows two sailboats on a body of water with dense green trees ... [tail] what's visible. So final keywords: Sailboats, River, Trees, Foli              | 500       | 122.12s | 4.37      | 39        | low-draft-improvement; cutoff; reasoning-leak                      |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `recommended` / `clean`       | Title: Two sailing boats moored on a river Description: Two sailing boats are moored on calm water, with the foreground vessel featuring a blac ... [tail] Rigging, Hull, Buoy, Outdoors, Green, Wood, Vessel, Estuary, Bank            | 109       | 142.77s | 6.35      | 38        | clean                                                              |
<!-- markdownlint-enable MD034 -->

## Quick Navigation

- _Successful outputs:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16),
  [`mlx-community/LFM2-VL-1.6B-8bit`](#model-mlx-community-lfm2-vl-16b-8bit),
  [`qnguyen3/nanoLLaVA`](#model-qnguyen3-nanollava),
  [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit),
  [`HuggingFaceTB/SmolVLM-Instruct`](#model-huggingfacetb-smolvlm-instruct),
  [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit),
  [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16),
  [`mlx-community/paligemma2-10b-ft-docci-448-6bit`](#model-mlx-community-paligemma2-10b-ft-docci-448-6bit),
  +52 more
- _Flagged outputs:_ [`mlx-community/gemma-4-31b-bf16`](#model-mlx-community-gemma-4-31b-bf16),
  [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16),
  [`mlx-community/LFM2-VL-1.6B-8bit`](#model-mlx-community-lfm2-vl-16b-8bit),
  [`qnguyen3/nanoLLaVA`](#model-qnguyen3-nanollava),
  [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit),
  [`HuggingFaceTB/SmolVLM-Instruct`](#model-huggingfacetb-smolvlm-instruct),
  [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit),
  [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16),
  +44 more
- _Failed outputs:_ [`mlx-community/gemma-4-31b-bf16`](#model-mlx-community-gemma-4-31b-bf16)

## Model Gallery

Full generated output by model:

<!-- markdownlint-disable MD033 MD034 -->

<a id="model-mlx-community-gemma-4-31b-bf16"></a>

### ❌ mlx-community/gemma-4-31b-bf16

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Status:_ Failed (Model Error)
- _Owner:_ likely owner `model`; reported package `mlx`; failure stage `Model
  Error`; diagnostic code `MLX_DECODE_MODEL`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Error summary:_ see error details below
- _Key signals:_ keywords=66; reasoning leak;
  text-sanity=gibberish(char_noise)
- _Failure context:_ type `ValueError`; phase `decode`; code
  `MLX_DECODE_MODEL`; package `mlx`
- _Tokens:_ prompt n/a; estimated text 478 tok; estimated non-text n/a;
  generated n/a; requested max 500 tok; stop reason exception

_Error details:_

> Model runtime error during generation for mlx-community/gemma-4-31b-bf16:
> [METAL] Command buffer execution failed: Insufficient Memory
> (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory).

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/common.py", line 141, in wired_limit
    yield
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/dispatch.py", line 1075, in stream_generate
    detokenizer.add_token(token, skip_special_token_ids=skip_special_token_ids)
    ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/tokenizer_utils.py", line 171, in add_token
    if self.is_byte_token[token]:
       ~~~~~~~~~~~~~~~~~~^^^^^^^
IndexError: list index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/.worktrees/report-output-redesign/src/check_models.py", line 21051, in _run_generation_with_retry_workaround
    return generate_once()
  File "~/Documents/AI/mlx/check_models/.worktrees/report-output-redesign/src/check_models.py", line 21427, in _generate_once
    return strict_generate(
        model=model,
    ...<21 lines>...
        **extra_kwargs,
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/dispatch.py", line 1240, in generate
    for response in stream_generate(
                    ~~~~~~~~~~~~~~~^
        model, processor, prompt, image, audio, video, verbose=verbose, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ):
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/dispatch.py", line 1009, in stream_generate
    with wired_limit(model, [generation_stream]):
         ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.13/contextlib.py", line 162, in __exit__
    self.gen.throw(value)
    ~~~~~~~~~~~~~~^^^^^^^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/common.py", line 145, in wired_limit
    mx.synchronize(stream)
    ~~~~~~~~~~~~~~^^^^^^^^
RuntimeError: [METAL] Command buffer execution failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory).

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/.worktrees/report-output-redesign/src/check_models.py", line 21619, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
        phase_callback=_update_phase,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        phase_timer=phase_timer,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/.worktrees/report-output-redesign/src/check_models.py", line 21458, in _run_model_generation
    output = _run_generation_with_retry_workaround(
        params=params,
        generate_once=_generate_once,
    )
  File "~/Documents/AI/mlx/check_models/.worktrees/report-output-redesign/src/check_models.py", line 21085, in _run_generation_with_retry_workaround
    raise _tag_exception_failure_phase(ValueError(msg), "decode") from gen_err
ValueError: Model runtime error during generation for mlx-community/gemma-4-31b-bf16: [METAL] Command buffer execution failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory).
```

</details>

---

<a id="model-liquidai-lfm25-vl-450m-mlx-bf16"></a>

### ✅ LiquidAI/LFM2.5-VL-450M-MLX-bf16

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Buoy, Mudflat, behind, bank
- _Score:_ ✅ B (76/100); Δ+27; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 0.38s; Gen 0.81s; Total 1.55s
- _Throughput:_ Prompt 17,590 TPS (639 tok); Gen 504 TPS (181 tok)
- _Tokens:_ prompt 639 tok; estimated text 478 tok; estimated non-text 161
  tok; generated 181 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe, Boating,
> Sailboat, River, Riverbank, Nature, Outdoors, Peaceful, Rigging, Boats,
> Muddy, Coast, Estuary, Foliage, Forest, Landscape, Mast, Moored, Sailing,
> Boats, Mushrooms, Trees, Water, Sky
>
> Description: Two sailboats moored on a river with dense green foliage in the
> background. The scene is serene, with calm water and a clear sky. The boats
> are positioned near the riverbank, with trees and bushes lining the shore.
> The overall atmosphere is peaceful and natural.
>
> Keywords: Boat, River, Estuary, Sailing, Muddy, Coast, Nature, Outdoors,
> Peaceful, Rigging, Boats, Mushrooms, Trees, Water, Sky
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Title length violation (30 words; expected 5-10)
- Description sentence violation (4; expected 1-2)

---

<a id="model-mlx-community-lfm2-vl-16b-8bit"></a>

### ✅ mlx-community/LFM2-VL-1.6B-8bit

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ keywords=20; unverified-context-copy; low-draft-improvement
- _Score:_ 🟡 C (59/100); Δ+10; Missing requested structure
- _Review focus:_ watchlist (unverified context copy)
- _Timing:_ Load 0.54s; Gen 0.99s; Total 1.91s
- _Throughput:_ Prompt 8,078 TPS (829 tok); Gen 331 TPS (144 tok)
- _Tokens:_ prompt 829 tok; estimated text 478 tok; estimated non-text 351
  tok; generated 144 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe, 2026-07-04
> 19:10:04 BST 19:10:04
>
> Description: Two sailing boats moored on a river with trees behind on the
> bank. The boats are white with orange masts and rigging. The riverbank is
> lined with green bushes and trees. The water is calm and blue.
>
> Keywords: Bird, Boat, Boating, Buoy, Bushes, Coast, Estuary, Foliage,
> Forest, Landscape, Mast, Moored, Mudflat, Nature, Outdoors, Peaceful,
> Rigging, River, Riverbank, Sailboat
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Title length violation (17 words; expected 5-10)
- Description sentence violation (4; expected 1-2)
- Keyword count violation (20; expected 10-18)
- Unverified context copy
- Low draft improvement

---

<a id="model-qnguyen3-nanollava"></a>

### ✅ qnguyen3/nanoLLaVA

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; missing terms: Bird, Boating,
  Buoy, Bushes, Coast; low-draft-improvement
- _Score:_ 🟠 D (46/100); Δ-3; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (missing sections)
- _Timing:_ Load 0.70s; Gen 1.00s; Total 2.06s
- _Throughput:_ Prompt 5,194 TPS (562 tok); Gen 113 TPS (37 tok)
- _Tokens:_ prompt 562 tok; estimated text 478 tok; estimated non-text 84 tok;
  generated 37 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe
> Description: Two sailing boats moored on a river with trees behind on the
> bank.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (keywords)
- Low draft improvement

---

<a id="model-mlx-community-nanollava-15-4bit"></a>

### ✅ mlx-community/nanoLLaVA-1.5-4bit

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ keywords=21; low-draft-improvement
- _Score:_ 🟡 C (51/100); Δ+2; Lacks visual description of image
- _Timing:_ Load 0.67s; Gen 1.15s; Total 2.18s
- _Throughput:_ Prompt 5,914 TPS (562 tok); Gen 192 TPS (98 tok)
- _Tokens:_ prompt 562 tok; estimated text 478 tok; estimated non-text 84 tok;
  generated 98 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe
> Description: Two sailing boats moored on a river with trees behind on the
> bank
> Keywords: Bird, Boat, Boating, Buoy, Bushes, Coast, Estuary, Foliage,
> Forest, Landscape, Mast, Moored, Mudflat, Nature, Outdoors, Peaceful,
> Rigging, River, Riverbank, Sailboat
>
> Do you need more information?
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Keyword count violation (21; expected 10-18)
- Low draft improvement

---

<a id="model-huggingfacetb-smolvlm-instruct"></a>

### ✅ HuggingFaceTB/SmolVLM-Instruct

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Bird, Boat, Boating, Buoy, Bushes; low-draft-improvement
- _Score:_ ❌ F (11/100); Δ-38; Output lacks detail
- _Review focus:_ watchlist (context ignored, missing sections)
- _Timing:_ Load 0.68s; Gen 1.22s; Total 2.27s
- _Throughput:_ Prompt 4,303 TPS (1,776 tok); Gen 127 TPS (18 tok)
- _Tokens:_ prompt 1776 tok; estimated text 478 tok; estimated non-text 1298
  tok; generated 18 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Deben Estuary, Woodbridge, England, UK, GBR, Europe.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Bird, Boat, Boating, Buoy, Bushes)
- Missing sections (title, description, keywords)
- Low draft improvement
- Ignores trusted hints

---

<a id="model-mlx-community-minicpm-v-46-8bit"></a>

### ✅ mlx-community/MiniCPM-V-4.6-8bit

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;/think&gt; appeared in generated
  text.; missing terms: Bird, Boating, Bushes, Coast, Estuary; reasoning leak;
  text-sanity=gibberish(token_noise)
- _Score:_ 🏆 A (80/100); Δ+31; None identified
- _Review focus:_ watchlist (harness, reasoning leak, text sanity)
- _Timing:_ Load 0.94s; Gen 1.36s; Total 2.69s
- _Throughput:_ Prompt 4,387 TPS (1,179 tok); Gen 181 TPS (81 tok)
- _Tokens:_ prompt 1179 tok; estimated text 478 tok; estimated non-text 701
  tok; generated 81 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &lt;think&gt;
>
> &lt;/think&gt;
>
> &#8203;Title:
> Sailing boats moored near dense foliage on water
>
> &#8203;Description:
> The image shows two sailing boats floating on calm water with lush green
> trees in the background. The scene appears peaceful with natural
> surroundings.
>
> &#8203;Keywords:
> sailing boats, water, trees, foliage, moored, calm, green, river, nature,
> peaceful, mast, rigging, buoy
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:stop_token
- Text sanity issue (gibberish(token_noise))
- Reasoning leak (&lt;think&gt;)
- Unknown tags: &lt;think&gt;
- token_leak:&lt;/think&gt;

---

<a id="model-mlx-community-fastvlm-05b-bf16"></a>

### ✅ mlx-community/FastVLM-0.5B-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ keywords=28; unverified-context-copy; low-draft-improvement
- _Score:_ 🟡 C (55/100); Δ+6; Lacks visual description of image
- _Review focus:_ watchlist (unverified context copy)
- _Timing:_ Load 0.87s; Gen 1.70s; Total 2.95s
- _Throughput:_ Prompt 5,427 TPS (566 tok); Gen 292 TPS (171 tok)
- _Tokens:_ prompt 566 tok; estimated text 478 tok; estimated non-text 88 tok;
  generated 171 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe
> Description: Two sailing boats moored on a river with trees behind on the
> bank
> Keywords: Bird, Boat, Boating, Buoy, Bushes, Coast, Estuary, Foliage,
> Forest, Landscape, Mast, Moored, Mudflat, Nature, Outdoors, Peaceful,
> Rigging, River, Riverbank, Sailboat
> Context: Authoritative context: Deben Estuary, Woodbridge, England, UK, GBR,
> Europe
> Capture date/time: 2026-07-04 19:10:04 BST 19:10:04
> Use this factual context where it improves the catalogue record; do not
> claim that contextual facts are visually observable.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Keyword count violation (28; expected 10-18)
- Unverified context copy
- Low draft improvement

---

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-6bit"></a>

### ✅ mlx-community/paligemma2-10b-ft-docci-448-6bit

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Bird, Boat, Boating, Buoy, Bushes; low-draft-improvement
- _Score:_ ❌ F (14/100); Δ-35; Output lacks detail
- _Review focus:_ watchlist (context ignored, missing sections)
- _Timing:_ Load 1.62s; Gen 2.27s; Total 4.25s
- _Throughput:_ Prompt 1,360 TPS (1,582 tok); Gen 30.5 TPS (17 tok)
- _Tokens:_ prompt 1582 tok; estimated text 478 tok; estimated non-text 1104
  tok; generated 17 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#45; Do not copy prompt instructions into the Title, Description, or Keywords
> fields.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Bird, Boat, Boating, Buoy, Bushes)
- Missing sections (title, description, keywords)
- Low draft improvement
- Ignores trusted hints

---

<a id="model-mlx-community-gemma-4-26b-a4b-it-4bit"></a>

### ✅ mlx-community/gemma-4-26b-a4b-it-4bit

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Buoy, Bushes, Coast
- _Score:_ ✅ B (79/100); Δ+30; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 2.51s; Gen 2.69s; Total 5.57s
- _Throughput:_ Prompt 1,367 TPS (835 tok); Gen 56.7 TPS (86 tok)
- _Tokens:_ prompt 835 tok; estimated text 478 tok; estimated non-text 357
  tok; generated 86 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Two small sailboats moored on the Deben Estuary
>
> &#8203;Description:
> Two small sailboats with wooden masts are moored on the water against a
> backdrop of dense green trees. The scene is captured in natural light with a
> vertical composition.
>
> &#8203;Keywords:
> boat, sailboat, mast, rigging, water, river, trees, foliage, forest,
> greenery, nature, outdoors, landscape, mooring, estuary, vertical, daylight
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

---

<a id="model-microsoft-phi-35-vision-instruct"></a>

### ✅ microsoft/Phi-3.5-vision-instruct

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Boating, Buoy, Bushes, Coast, Foliage;
  keywords=21; low-draft-improvement
- _Score:_ 🟡 C (58/100); Δ+9; Lacks visual description of image
- _Timing:_ Load 1.21s; Gen 3.08s; Total 4.65s
- _Throughput:_ Prompt 3,981 TPS (1,394 tok); Gen 57.6 TPS (130 tok)
- _Tokens:_ prompt 1394 tok; estimated text 478 tok; estimated non-text 916
  tok; generated 130 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Sailboats Moored in Deben Estuary
>
> Description: Two sailboats are moored on a river with lush greenery in the
> background. The boats are equipped with masts and rigging, indicating they
> are ready for sailing.
>
> Keywords: Sailboats, Deben Estuary, Woodbridge, England, UK, Europe, Mast,
> Rigging, River, Riverbank, Nature, Lush, Greenery, Mudflat, Peaceful,
> Outdoors, Coastal, Sailing, Mudflats, Forest, Birds
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Keyword count violation (21; expected 10-18)
- Low draft improvement

---

<a id="model-mlx-community-lfm25-vl-16b-bf16"></a>

### ✅ mlx-community/LFM2.5-VL-1.6B-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: description, keywords;
  missing terms: Bird, Bushes, Forest, Mast, Mudflat; repetitive token=phrase:
  "boating, boats, moored, buoy,..."
- _Score:_ ❌ F (34/100); Δ-14; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, generation loop, missing sections,
  repetitive)
- _Timing:_ Load 0.59s; Gen 3.41s; Total 4.37s
- _Throughput:_ Prompt 7,934 TPS (829 tok); Gen 175 TPS (500 tok)
- _Tokens:_ prompt 829 tok; estimated text 478 tok; estimated non-text 351
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> &#45; Sailboats, River, Woodbridge, England, Estuary, Trees, Boating, Boats,
> Moored, Buoy, Landscape, Nature, Riverbank, Foliage, Coast, Landscape,
> Estuary, River, Nature, Boats, Trees, Landscape, Boating, Boats, Moored,
> Buoy, Landscape, Nature, Estuary, Riverbank, Foliage, Coast, Landscape,
> Boating, Boats, Moored, Buoy, Landscape, Nature, Estuary, Riverbank,
> Foliage, Coast, Landscape, Boating, Boats, Moored, Buoy, Landscape, Nature,
> Estuary, Riverbank, Foliage, Coast, Landscape, Boating, Boats, Moored, Buoy,
> Landscape, Nature, Estuary, Riverbank, Foliage, Coast, Landscape, Boating,
> Boats, Moored, Buoy, Landscape, Nature, Estuary, Riverbank, Foliage, Coast,
> Landscape, Boating, Boats, Moored, Buoy, Landscape, Nature, Estuary,
> Riverbank, Foliage, Coast, Landscape, Boating, Boats, Moored, Buoy,
> Landscape, Nature, Estuary, Riverbank, Foliage, Coast, Landscape, Boating,
> Boats, Moored, Buoy, Landscape, Nature, Estuary, Riverbank, Foliage, Coast,
> Landscape, Boating, Boats, Moored, Buoy, Landscape, Nature, Estuary,
> Riverbank, Foliage, Coast, Landscape, Boating, Boats, Moored, Buoy,
> Landscape, Nature, Estuary, Riverbank, Foliage, Coast, Landscape, Boating,
> Boats, Moored, Buoy, Landscape, Nature, Estuary, Riverbank, Foliage, Coast,
> Landscape, Boating, Boats, Moored, Buoy, Landscape, Nature, Estuary,
> Riverbank, Foliage, Coast, Landscape, Boating, Boats, Moored, Buoy,
> Landscape, Nature, Estuary, Riverbank, Foliage, Coast, Landscape, Boating,
> Boats, Moored, Buoy, Landscape, Nature, Estuary, Riverbank, Foliage, Coast,
> Landscape, Boating, Boats, Moored, Buoy, Landscape, Nature, Estuary, River
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "boating, boats, moored, buoy,...")
- Generation loop (repetitive_tail)
- Missing sections (description, keywords)
- Title length violation (184 words; expected 5-10)
- Low draft improvement
- Likely capped by max token budget

---

<a id="model-mlx-community-internvl3-8b-bf16"></a>

### ✅ mlx-community/InternVL3-8B-bf16

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Landscape, Mudflat, Peaceful, Rigging;
  low-draft-improvement
- _Score:_ 🟡 C (64/100); Δ+15; Limited novel information
- _Timing:_ Load 1.66s; Gen 3.79s; Total 5.82s
- _Throughput:_ Prompt 2,720 TPS (2,345 tok); Gen 35.3 TPS (81 tok)
- _Tokens:_ prompt 2345 tok; estimated text 478 tok; estimated non-text 1867
  tok; generated 81 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Two sailboats moored on a river
>
> &#8203;Description:
> Two sailboats with orange masts are moored on a calm river, surrounded by
> dense green foliage on the riverbank.
>
> &#8203;Keywords:
> Boat, Boating, Buoy, Bushes, Coast, Estuary, Foliage, Forest, Mast, Moored,
> Nature, Outdoors, River, Riverbank, Sailboat
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Low draft improvement

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-mxfp4"></a>

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-mxfp4

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Bushes, Coast, Estuary
- _Score:_ 🏆 A (83/100); Δ+34; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 1.51s; Gen 4.38s; Total 6.26s
- _Throughput:_ Prompt 1,414 TPS (3,173 tok); Gen 66.1 TPS (105 tok)
- _Tokens:_ prompt 3173 tok; estimated text 478 tok; estimated non-text 2695
  tok; generated 105 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Sailboat Anchored Near Wooded Riverbank
>
> &#8203;Description:
> A traditional wooden sailboat with two tall masts and red sails is moored on
> calm water, surrounded by dense green foliage and trees along the
> riverbank. The boat’s hull and rigging are clearly visible, and a small buoy
> floats nearby.
>
> &#8203;Keywords:
> Sailboat, wooden hull, two masts, red sails, river, dense foliage, green
> trees, calm water, moored, buoy, riverbank
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

---

<a id="model-mlx-community-pixtral-12b-8bit"></a>

### ✅ mlx-community/pixtral-12b-8bit

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Bushes, Coast, Forest;
  low-draft-improvement
- _Score:_ ✅ B (72/100); Δ+23; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 1.99s; Gen 4.67s; Total 7.02s
- _Throughput:_ Prompt 1,822 TPS (3,366 tok); Gen 39.4 TPS (92 tok)
- _Tokens:_ prompt 3366 tok; estimated text 478 tok; estimated non-text 2888
  tok; generated 92 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Sailboats Moored on River with Trees
>
> &#8203;Description:
> Two sailboats are moored on a calm river, with dense green trees lining the
> riverbank behind them.
>
> &#8203;Keywords:
> Sailboat, River, Trees, Moored, Masts, Sails, Buoy, Calm Water, Green,
> Nature, Outdoors, Riverbank, Woodland, Boat, Sailing, Peaceful, Estuary,
> Foliage
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Low draft improvement

---

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-bf16"></a>

### ✅ mlx-community/paligemma2-10b-ft-docci-448-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Bird, Boat, Boating, Buoy, Bushes; low-draft-improvement
- _Score:_ ❌ F (14/100); Δ-35; Output lacks detail
- _Review focus:_ watchlist (context ignored, missing sections)
- _Timing:_ Load 2.47s; Gen 4.98s; Total 7.83s
- _Throughput:_ Prompt 1,128 TPS (1,582 tok); Gen 5.63 TPS (17 tok)
- _Tokens:_ prompt 1582 tok; estimated text 478 tok; estimated non-text 1104
  tok; generated 17 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#45; Do not copy prompt instructions into the Title, Description, or Keywords
> fields.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Bird, Boat, Boating, Buoy, Bushes)
- Missing sections (title, description, keywords)
- Low draft improvement
- Ignores trusted hints

---

<a id="model-mlx-community-idefics3-8b-llama3-bf16"></a>

### ✅ mlx-community/Idefics3-8B-Llama3-bf16

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Buoy, Bushes, Coast;
  formatting=Unknown tags: &lt;end_of_utterance&gt;; low-draft-improvement
- _Score:_ ✅ B (70/100); Δ+21; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 2.14s; Gen 5.12s; Total 7.64s
- _Throughput:_ Prompt 2,483 TPS (2,855 tok); Gen 33.3 TPS (109 tok)
- _Tokens:_ prompt 2855 tok; estimated text 478 tok; estimated non-text 2377
  tok; generated 109 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Sailing Boats on the Deben Estuary, Woodbridge, England, UK, GBR,
> Europe
>
> Description: Two sailing boats are moored on a river with trees behind on
> the bank. The boats have masts and rigging. The water is calm. The sky is
> not visible.
>
> Keywords: Sailing Boats, Deben Estuary, Woodbridge, England, UK, GBR,
> Europe, River, Trees, Bank, Masts, Rigging, Calm Water, Sky Not
> Visible.<end_of_utterance>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Title length violation (11 words; expected 5-10)
- Description sentence violation (4; expected 1-2)
- Low draft improvement
- Unknown tags: <end_of_utterance>

---

<a id="model-mlx-community-diffusiongemma-26b-a4b-it-8bit"></a>

### ✅ mlx-community/diffusiongemma-26B-A4B-it-8bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Coast, Mudflat, Peaceful, GBR;
  low-draft-improvement
- _Score:_ ✅ B (65/100); Δ+16; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 3.46s; Gen 5.45s; Total 9.35s
- _Throughput:_ Prompt 1,852 TPS (835 tok); Gen 20.6 TPS (89 tok)
- _Tokens:_ prompt 835 tok; estimated text 478 tok; estimated non-text 357
  tok; generated 89 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> &#45; Two sailing boats moored on the Deben Estuary
>
> &#8203;Description:
> &#45; Two wooden sailboats with tall masts are anchored on a river next to a
> dense green forest bank. A small buoy floats near the vessels in the calm
> water.
>
> &#8203;Keywords:
> Boat, boating, buoy, bushes, estuary, foliage, forest, landscape, mast,
> moored, mud, nature, outdoors, rigging, river, riverbank, sailboat
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Low draft improvement

---

<a id="model-mlx-community-ministral-3-3b-instruct-2512-4bit"></a>

### ✅ mlx-community/Ministral-3-3B-Instruct-2512-4bit

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Bushes, Coast, Landscape
- _Score:_ 🏆 A (86/100); Δ+37; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 1.36s; Gen 5.74s; Total 7.66s
- _Throughput:_ Prompt 984 TPS (3,172 tok); Gen 64.9 TPS (125 tok)
- _Tokens:_ prompt 3172 tok; estimated text 478 tok; estimated non-text 2694
  tok; generated 125 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Traditional Wooden Sailboat in Forested Riverbank Setting
>
> &#8203;Description:
> A classic wooden sailboat with tall masts and visible rigging lines is
> moored on calm water beside a dense, green forest. The boat features a light
> green hull, a red sail, and a small buoy nearby. The scene is bathed in
> natural daylight, with lush vegetation dominating the riverbank.
>
> &#8203;Keywords:
> boat, sailboat, wooden vessel, sail, forest, riverbank, estuary, green
> foliage, tall mast, rigging, buoy, moored vessel, nature scene, tranquil
> waters, maritime setting
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Description sentence violation (3; expected 1-2)

---

<a id="model-mlx-community-smolvlm-instruct-bf16"></a>

### ✅ mlx-community/SmolVLM-Instruct-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Bird, Boat, Boating, Buoy, Bushes; low-draft-improvement
- _Score:_ ❌ F (11/100); Δ-38; Output lacks detail
- _Review focus:_ watchlist (context ignored, missing sections)
- _Timing:_ Load 1.54s; Gen 6.32s; Total 10.10s
- _Throughput:_ Prompt 964 TPS (1,776 tok); Gen 38.6 TPS (18 tok)
- _Tokens:_ prompt 1776 tok; estimated text 478 tok; estimated non-text 1298
  tok; generated 18 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Deben Estuary, Woodbridge, England, UK, GBR, Europe.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Bird, Boat, Boating, Buoy, Bushes)
- Missing sections (title, description, keywords)
- Low draft improvement
- Ignores trusted hints

---

<a id="model-mlx-community-smolvlm2-22b-instruct-mlx"></a>

### ✅ mlx-community/SmolVLM2-2.2B-Instruct-mlx

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Mast; keyword duplication=41%;
  unverified-context-copy; low-draft-improvement
- _Score:_ 🟡 C (56/100); Δ+7; Lacks visual description of image
- _Review focus:_ watchlist (unverified context copy)
- _Timing:_ Load 1.47s; Gen 6.33s; Total 9.98s
- _Throughput:_ Prompt 1,902 TPS (676 tok); Gen 120 TPS (275 tok)
- _Tokens:_ prompt 676 tok; estimated text 478 tok; estimated non-text 198
  tok; generated 275 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe
> Description: Two sailing boats moored on a river with trees behind on the
> bank
> Keywords: Bird, Boat, Boating, Buoy, Bushes, Coast, Estuary, Foliage,
> Forest, Landscape, Moored, Mudflat, Nature, Outdoors, Peaceful, Rigging,
> River, Riverbank, Sailboat
>
> Context: Location terms: Deben Estuary, Woodbridge, England, UK, GBR, Europe
> Capture date/time: 2026-07-04 19:10:04 BST 19:10:04
>
> &#8203;Draft descriptive metadata:
> &#45; Existing title: Deben Estuary, Woodbridge, England, UK, GBR, Europe
> &#45; Existing description: Two sailing boats moored on a river with trees
> behind on the bank
> &#45; Existing keywords: Bird, Boat, Boating, Buoy, Bushes, Coast, Estuary,
> Foliage, Forest, Landscape, Moored, Mudflat, Nature, Outdoors, Peaceful,
> Rigging, River, Riverbank, Sailboat
> &#45; Treat this draft as fallible. Retain supported details, correct errors,
> and add important visible information.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Keyword count violation (56; expected 10-18)
- Keyword duplication (41% duplicated terms)
- Unverified context copy
- Low draft improvement

---

<a id="model-mlx-community-pixtral-12b-bf16"></a>

### ✅ mlx-community/pixtral-12b-bf16

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Bushes, Coast, Foliage;
  low-draft-improvement
- _Score:_ ✅ B (73/100); Δ+24; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 2.79s; Gen 6.52s; Total 9.67s
- _Throughput:_ Prompt 1,903 TPS (3,366 tok); Gen 20.7 TPS (88 tok)
- _Tokens:_ prompt 3366 tok; estimated text 478 tok; estimated non-text 2888
  tok; generated 88 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Sailboats Moored on River with Trees
>
> &#8203;Description:
> Two sailboats are moored on a calm river, with dense green trees lining the
> riverbank behind them.
>
> &#8203;Keywords:
> Sailboat, River, Trees, Moored, Masts, Sails, Buoy, Calm Water, Green,
> Nature, Outdoors, Riverbank, Woodland, Boat, Sailing, Peaceful, Estuary
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
- _Key signals:_ missing terms: Bird, Boating, Bushes, Coast, Forest
- _Score:_ 🏆 A (85/100); Δ+36; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 1.58s; Gen 6.90s; Total 8.86s
- _Throughput:_ Prompt 1,392 TPS (3,173 tok); Gen 26.5 TPS (108 tok)
- _Tokens:_ prompt 3173 tok; estimated text 478 tok; estimated non-text 2695
  tok; generated 108 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;&#42;&#42;Title:&#42;&#42;
> Sailboat Moored on Deben Estuary
>
> &#8203;&#42;&#42;Description:&#42;&#42;
> A classic wooden sailboat with two masts is anchored near a riverbank,
> surrounded by dense green foliage. Visible details include red sails, wooden
> hull, and a nearby floating buoy.
>
> &#8203;&#42;&#42;Keywords:&#42;&#42;
> Sailboat, Wooden hull, Two masts, Red sails, Riverbank, Dense foliage,
> Floating buoy, Classic boat, Deben Estuary, Moored vessel, Green landscape,
> Woodbridge, Rigging lines
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

---

<a id="model-mlx-community-gemma-4-31b-it-4bit"></a>

### ✅ mlx-community/gemma-4-31b-it-4bit

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Bushes, Coast, Mudflat;
  low-draft-improvement
- _Score:_ ✅ B (74/100); Δ+25; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 2.70s; Gen 6.91s; Total 9.98s
- _Throughput:_ Prompt 478 TPS (835 tok); Gen 19.6 TPS (90 tok)
- _Tokens:_ prompt 835 tok; estimated text 478 tok; estimated non-text 357
  tok; generated 90 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Sailing boats moored on the Deben Estuary, England
>
> Description: Two sailing boats with wooden masts are moored on a river in
> front of a dense green forest. A small buoy floats in the foreground on the
> water.
>
> Keywords: boat, sailing boat, mast, rigging, river, riverbank, forest,
> trees, foliage, water, buoy, moored, Deben Estuary, England, nature,
> outdoors, landscape, green
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Low draft improvement

---

<a id="model-mlx-community-internvl3-14b-8bit"></a>

### ✅ mlx-community/InternVL3-14B-8bit

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Bushes, Coast, Mudflat;
  low-draft-improvement
- _Score:_ ✅ B (71/100); Δ+22; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 1.76s; Gen 6.91s; Total 9.05s
- _Throughput:_ Prompt 1,410 TPS (2,345 tok); Gen 23.6 TPS (109 tok)
- _Tokens:_ prompt 2345 tok; estimated text 478 tok; estimated non-text 1867
  tok; generated 109 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Sailing Boats on Deben Estuary
>
> &#8203;Description:
> Two sailboats are moored on the Deben Estuary, with lush green trees lining
> the riverbank in the background. The boats have masts and rigging, and one
> has a red cover over part of its deck.
>
> &#8203;Keywords:
> Boat, Buoy, Deben Estuary, Foliage, Forest, Landscape, Mast, Moored, Nature,
> Outdoors, Peaceful, River, Riverbank, Sailboat, Trees, UK, Woodbridge
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Low draft improvement

---

<a id="model-mlx-community-devstral-small-2-24b-instruct-2512-5bit"></a>

### ✅ mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Buoy, Bushes, Coast;
  low-draft-improvement
- _Score:_ ✅ B (72/100); Δ+23; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 2.31s; Gen 7.00s; Total 9.70s
- _Throughput:_ Prompt 798 TPS (2,640 tok); Gen 30.4 TPS (96 tok)
- _Tokens:_ prompt 2640 tok; estimated text 478 tok; estimated non-text 2162
  tok; generated 96 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Sailboat on River
>
> &#8203;Description:
> A wooden sailboat with two masts is moored on a calm river, surrounded by
> dense green trees on the riverbank. The boat has red sails and a small duck
> swimming nearby.
>
> &#8203;Keywords:
> sailboat, river, trees, wooden boat, red sails, masts, moored, duck, calm
> water, green foliage, nature, outdoors, peaceful, rigging, riverbank,
> estuary
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Title length violation (3 words; expected 5-10)
- Low draft improvement

---

<a id="model-mlx-community-gemma-3n-e2b-4bit"></a>

### ✅ mlx-community/gemma-3n-E2B-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Bird, Boat, Boating, Buoy, Bushes; excessive
  bullets=50
- _Score:_ 🟠 D (48/100); Δ-0; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, generation loop, missing
  sections, repetitive)
- _Timing:_ Load 1.50s; Gen 7.26s; Total 9.13s
- _Throughput:_ Prompt 2,617 TPS (821 tok); Gen 77.6 TPS (500 tok)
- _Tokens:_ prompt 821 tok; estimated text 478 tok; estimated non-text 343
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
>
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
> &#45; Do not take it out of the.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "- do not take...")
- Excessive bullet points (50)
- Context ignored (missing: Bird, Boat, Boating, Buoy, Bushes)
- Generation loop (repetitive_tail)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8"></a>

### ✅ mlx-community/diffusiongemma-26B-A4B-it-mxfp8

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Bushes, Coast, Landscape, Mudflat;
  low-draft-improvement
- _Score:_ ✅ B (71/100); Δ+22; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 3.41s; Gen 7.55s; Total 11.37s
- _Throughput:_ Prompt 1,851 TPS (835 tok); Gen 15.0 TPS (96 tok)
- _Tokens:_ prompt 835 tok; estimated text 478 tok; estimated non-text 357
  tok; generated 96 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> &#45; Two sailing boats moored on the Deben Estuary
>
> &#8203;Description:
> &#45; Two small wooden sailboats are moored on a river against a backdrop of
> dense green forest trees. One boat features a red sail cover while the other
> has a wooden hull.
>
> &#8203;Keywords:
> &#45; Boat, Boating, Buoy, Estuary, Foliage, Forest, Green,, Mast, Moored,
> Nature, Outdoors, Rigging, River, Riverbank, Sailboat
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Low draft improvement

---

<a id="model-mlx-community-phi-35-vision-instruct-bf16"></a>

### ✅ mlx-community/Phi-3.5-vision-instruct-bf16

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Boating, Buoy, Bushes, Coast, Foliage;
  keywords=21; low-draft-improvement
- _Score:_ 🟡 C (58/100); Δ+9; Lacks visual description of image
- _Timing:_ Load 0.98s; Gen 7.84s; Total 9.22s
- _Throughput:_ Prompt 3,290 TPS (1,394 tok); Gen 19.1 TPS (130 tok)
- _Tokens:_ prompt 1394 tok; estimated text 478 tok; estimated non-text 916
  tok; generated 130 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Sailboats Moored in Deben Estuary
>
> Description: Two sailboats are moored on a river with lush greenery in the
> background. The boats are equipped with masts and rigging, indicating they
> are ready for sailing.
>
> Keywords: Sailboats, Deben Estuary, Woodbridge, England, UK, Europe, Mast,
> Rigging, River, Riverbank, Nature, Lush, Greenery, Mudflat, Peaceful,
> Outdoors, Coastal, Sailing, Mudflats, Forest, Birds
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Keyword count violation (21; expected 10-18)
- Low draft improvement

---

<a id="model-mlx-community-gemma-3n-e4b-it-bf16"></a>

### ✅ mlx-community/gemma-3n-E4B-it-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Buoy, Bushes, Coast, Foliage;
  keywords=27; unverified-context-copy
- _Score:_ 🏆 A (83/100); Δ+34; None identified
- _Review focus:_ watchlist (unverified context copy)
- _Timing:_ Load 2.39s; Gen 8.05s; Total 10.81s
- _Throughput:_ Prompt 1,914 TPS (829 tok); Gen 46.8 TPS (332 tok)
- _Tokens:_ prompt 829 tok; estimated text 478 tok; estimated non-text 351
  tok; generated 332 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#35;# Image Description for Cataloguing
>
> &#42;&#42;Title:&#42;&#42; Two Sailboats Moored on the Deben Estuary
>
> &#42;&#42;Description:&#42;&#42; Two small sailboats are moored side-by-side on a calm body
> of water, likely the Deben Estuary in England. The sailboats are the primary
> focus, with the one on the left being a darker hull with a white deck and a
> red and white sail partially furled. The sailboat on the right has a
> lighter hull, a white deck, and a red sail also partially furled. Both boats
> have masts and rigging visible.
>
> A small, brown duck is swimming near the stern of the lighter sailboat. The
> water is a muted grey-blue, reflecting the overcast sky. The background is
> dominated by lush green trees lining the bank of the estuary, creating a
> natural and peaceful setting. The trees are dense and varied, suggesting a
> healthy riparian environment.
>
> The overall scene is tranquil and depicts a typical day on the water. The
> lighting is soft and diffused, consistent with an overcast day.
>
> &#42;&#42;Keywords:&#42;&#42; Boat, Sailboat, Moored, Estuary, Water, Duck, Trees, Green,
> England, UK, Boating, Nature, Calm, Overcast, Riverbank, Mast, Rigging.
>
> &#42;&#42;Location:&#42;&#42; Deben Estuary, near Woodbridge, Suffolk, England, UK.
>
> &#42;&#42;Date/Time:&#42;&#42; July 4th, 2026, 19:10:04 BST.
>
> &#42;&#42;Note:&#42;&#42; The image is clear and well-defined, with all elements readily
> visible. There are no obscured or indistinct details.
>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Description sentence violation (10; expected 1-2)
- Keyword count violation (27; expected 10-18)
- Unverified context copy

---

<a id="model-mlx-community-gemma-3-27b-it-qat-8bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-8bit

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Bushes, Coast, Mudflat, Peaceful;
  keywords=19; low-draft-improvement
- _Score:_ ✅ B (72/100); Δ+23; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 3.39s; Gen 8.05s; Total 11.81s
- _Throughput:_ Prompt 590 TPS (830 tok); Gen 17.7 TPS (108 tok)
- _Tokens:_ prompt 830 tok; estimated text 478 tok; estimated non-text 352
  tok; generated 108 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Two sailing boats moored on the river Deben
>
> &#8203;Description:
> Two sailing boats are moored on calm water, with dense green trees lining
> the riverbank behind them. The boats have masts and rigging visible against
> the foliage, and a single orange buoy is visible in the foreground.
>
> &#8203;Keywords:
> Boat, Boating, Estuary, Foliage, Forest, Landscape, Mast, Moored, Nature,
> Outdoors, River, Riverbank, Sailboat, Trees, Water, Calm, Green, Masts,
> Rigging
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
  GBR; keywords=59
- _Score:_ 🟡 C (57/100); Δ+8; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, degeneration, generation loop, missing
  sections, reasoning leak)
- _Timing:_ Load 2.23s; Gen 9.07s; Total 11.69s
- _Throughput:_ Prompt 1,627 TPS (1,564 tok); Gen 66.7 TPS (500 tok)
- _Tokens:_ prompt 1564 tok; estimated text 478 tok; estimated non-text 1086
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ◁think▷Got it, let's analyze the image step by step.
>
> First, the title: The main visible elements are two sailboats on a body of
> water with trees in the background. So the title should be concise. Let's
> check the existing draft: "Two sailing boats moored on a river with trees
> behind on the bank" – that's good, but maybe more precise. Wait, the image
> shows two sailboats, so "Two sailboats moored on a river with dense green
> foliage in the background" – but need to be factual. Let's check the image:
> two sailboats, water, trees. So title could be "Two sailboats on a river
> with dense green trees in the background".
>
> Description: The main subject is two sailboats on a river, with dense green
> trees in the background. The setting is a river, likely Deben Estuary (from
> context), but the image doesn't show the location explicitly, so maybe omit
> location unless confirmed. The lighting is natural daylight, calm water. The
> action is the boats being moored. So description: "Two sailboats moored on
> a river with dense green trees lining the banks in the background, under
> bright daylight."
>
> Keywords: Let's list visible elements: Sailboats, river, trees, water,
> foliage, boats, masts, rigging, buoy, natural landscape. Let's check the
> existing draft keywords: Bird, Boat, Boating, Buoy, Bushes, Coast, Estuary,
> Foliage, Forest, Landscape, Mast, Moored, Mudflat, Nature, Outdoors,
> Peaceful, Rigging, River, Riverbank, Sailboat. But in the image, there's no
> bird, mudflat, coast, forest (maybe bushes), peaceful (not explicitly
> shown), etc. So update keywords to: Sailboats, river, trees, water, foliage,
> buoy, masts, rigging, natural landscape, outdoors, daylight, Deben Estuary
> (but wait, the context says Deben Estuary is authoritative, but is it
> visible? The image doesn't show "Deben Estuary" text. So maybe omit unless
> confirmed. Wait, the context says "Authoritative context: Location terms:
> Deben Estuary, England, Europe, UK, Woodbridge". But the image doesn't have
> visible text or landmarks indicating Deben Estuary. So better to omit
> location unless visible. So keywords from image: Sailboats, river, trees,
> water, foliage, buoy, m
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Output degeneration (incomplete_sentence: ends with 'm')
- Generation loop (degeneration)
- Missing sections (title)
- Description sentence violation (5; expected 1-2)
- Keyword count violation (59; expected 10-18)
- Reasoning leak (◁think▷, let's analyze the image)
- Low draft improvement
- Likely capped by max token budget

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-8bit"></a>

### ✅ mlx-community/Kimi-VL-A3B-Thinking-8bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description;
  missing terms: Bird, Boating, Coast, Landscape, Mudflat; keywords=24
- _Score:_ 🟠 D (49/100); Δ-0; Lacks visual description of image
- _Review focus:_ watchlist (cutoff, missing sections, reasoning leak)
- _Timing:_ Load 2.55s; Gen 9.71s; Total 12.63s
- _Throughput:_ Prompt 1,646 TPS (1,564 tok); Gen 61.5 TPS (500 tok)
- _Tokens:_ prompt 1564 tok; estimated text 478 tok; estimated non-text 1086
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ◁think▷Okay, let's tackle this metadata cataloging task. First, I need to
> analyze the image based on the given rules. The user wants me to extract
> metadata using British English, focusing only on clearly visible details.
> Let me start with the title.
>
> The image shows boats on a river with trees in the background. The draft
> metadata mentions "Deben Estuary, Woodbridge, England, UK, GBR, Europe" as
> the location. Since the image shows boats on water with trees behind them,
> "Deben Estuary" is definitely visible. The boats are moored, so "Moored" is
> a keyword. The location is Woodbridge, which is part of the draft. The date
> and time are provided, but the user said to use the context where it
> improves the record, so maybe include the date. The title should be concise,
> so "Two moored sailboats on Deben Estuary at Woodbridge, England" seems
> accurate.
>
> For the description, the main subject is the sailboats, with trees in the
> background. The lighting is natural, possibly overcast since the image looks
> a bit dark. The boats have masts and rigging, which are visible. The
> keywords should include sailboats, mast, rigging, moored, river, riverbank,
> trees, foliage, nature, peaceful, outdoor, buoy. I need to avoid uncertain
> terms like "forest" since it's just bushes and trees. The draft keywords
> include "Forest," but the image shows bushes, so I'll replace that. Also,
> "peaceful" is inferred from the scene, but maybe "tranquil" is better. Wait,
> &#8203;the user said to omit inferred terms. So stick to what's visible:
> "tranquil" might be inferred, so better to leave it out. Include "outdoors"
> and "riverbank."
>
> Keywords: Sailboats, Mast, Rigging, Moored, River, Riverbank, Trees,
> Foliage, Buoy, Nature, Outdoor, England, UK. Wait, the location is already
> in the title, so maybe not repeat "England" in keywords. The draft had "GBR"
> but the user says to use the context where it improves, so maybe include
> "UK" and "England" in the title but not in keywords. Keywords should be
> unique terms. So keywords: Sailboats, Mast, Rigging, Moored, River,
> Riverbank, Trees, Foliage, Buoy
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description)
- Keyword count violation (24; expected 10-18)
- Reasoning leak (◁think▷)
- Low draft improvement
- Likely capped by max token budget

---

<a id="model-mlx-community-llama-32-11b-vision-instruct-8bit"></a>

### ✅ mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Deben, GBR, bank; keywords=19;
  low-draft-improvement
- _Score:_ 🟡 C (60/100); Δ+11; None identified
- _Timing:_ Load 1.74s; Gen 9.93s; Total 12.03s
- _Throughput:_ Prompt 332 TPS (543 tok); Gen 19.0 TPS (148 tok)
- _Tokens:_ prompt 543 tok; estimated text 478 tok; estimated non-text 65 tok;
  generated 148 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Two Sailing Boats Moored on a River with Trees Behind
>
> Description: The image depicts two sailing boats moored on a river, with a
> dense forest of trees in the background. The boats are positioned in the
> center of the image, with their masts and rigging visible. The river is calm
> and peaceful, with a few ripples on its surface. The forest behind the
> boats is lush and green, with a variety of trees and foliage visible.
>
> Keywords: Boat, Boating, Buoy, Bushes, Coast, Estuary, Foliage, Forest,
> Landscape, Mast, Moored, Mudflat, Nature, Outdoors, Peaceful, Rigging,
> River, Riverbank, Sailboat
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Description sentence violation (4; expected 1-2)
- Keyword count violation (19; expected 10-18)
- Low draft improvement

---

<a id="model-mlx-community-gemma-3-27b-it-qat-4bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-4bit

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Bushes, Forest, Peaceful, Rigging;
  keywords=21; low-draft-improvement
- _Score:_ ✅ B (79/100); Δ+30; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 2.54s; Gen 10.17s; Total 13.13s
- _Throughput:_ Prompt 206 TPS (830 tok); Gen 18.9 TPS (105 tok)
- _Tokens:_ prompt 830 tok; estimated text 478 tok; estimated non-text 352
  tok; generated 105 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Two sailing boats moored on a riverbank
>
> &#8203;Description:
> Two sailing boats are moored in calm water, with dense green trees forming
> the backdrop. A single orange buoy is visible in the foreground, and the
> boats are reflected in the water.
>
> &#8203;Keywords:
> Boat, Boating, Buoy, Coast, Estuary, Foliage, Landscape, Mast, Moored,
> Mudflat, Nature, Outdoors, River, Riverbank, Sailboat, Trees, Water,
> Reflection, Calm, Green, Orange
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Keyword count violation (21; expected 10-18)
- Low draft improvement

---

<a id="model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16"></a>

### ✅ mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Bird, Boating, Bushes, Coast, Foliage;
  low-draft-improvement
- _Score:_ 🟡 C (53/100); Δ+4; Lacks visual description of image
- _Review focus:_ watchlist (cutoff, missing sections)
- _Timing:_ Load 4.86s; Gen 10.69s; Total 15.93s
- _Throughput:_ Prompt 1,407 TPS (1,883 tok); Gen 57.4 TPS (500 tok)
- _Tokens:_ prompt 1883 tok; estimated text 478 tok; estimated non-text 1405
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Let's tackle this step by step. First, the title needs to be 5-10 words,
> concrete, and factual. The image has a sailboat in a wooded area by a river.
> Wait, the existing title is "Deben Estuary, Woodbridge, England, UK, GBR,
> Europe" but let's check if that's visible. Wait no, the user said to use
> only details clearly visible. Wait the context says Deben Estuary,
> Woodbridge, England, UK, GBR, Europe is authoritative context. But wait the
> image description mentions "a wooded area with a river, a sailboat with a
> red sail, a wooden boat, and a buoy". Wait let's go back. The user's prompt
> &#8203;says to use only details clearly visible. Let's list what's visible:
>
> &#45; A sailboat (sailing vessel) with a red sail (since the prompt mentions
> "red sail" in the image description? Wait no, wait the user's image prompt
> says "a red sail" but wait the original image (wait no, the user provided a
> text description? Wait no, wait the user's input is a bit messy, but wait
> the context says "Authoritative context: Location terms: Deben Estuary,
> England, Europe, UK, Woodbridge" so that's a fact. Then the existing draft
> title is "Deben Estuary, Woodbridge, England, UK, GBR, Europe" but let's
> check the rules: Title should be 5-10 words, concrete, factual. Wait "Deben
> Estuary, Woodbridge, England, UK" is 6 words, that's good. But wait does the
> image show Deben Estuary? The context says yes, so we can use that.
>
> Next, Description: Need 1-2 factual sentences. Main subject: A sailboat with
> &#8203;a red sail and a wooden boat moored in a wooded riverbank area. Setting:
> Wooded riverbank with trees, river. Lighting: Probably daytime (since it's a
> clear image). Distinctive details: Red sail, wooden boat, buoy. Wait the
> existing draft description is "Two sailing boats moored on a river with
> trees behind on the bank" but let's correct: Are there two boats? The
> context says "a sailboat and a wooden boat
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)
- Low draft improvement
- Likely capped by max token budget

---

<a id="model-mlx-community-llava-v16-mistral-7b-8bit"></a>

### ✅ mlx-community/llava-v1.6-mistral-7b-8bit

- _Recommendation:_ use with caveats; review verdict: token cap
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (500); keywords=27; low-draft-improvement
- _Score:_ 🟡 C (53/100); Δ+4; Lacks visual description of image
- _Review focus:_ watchlist (unverified context copy)
- _Timing:_ Load 0.95s; Gen 11.12s; Total 12.43s
- _Throughput:_ Prompt 1,346 TPS (2,791 tok); Gen 59.9 TPS (500 tok)
- _Tokens:_ prompt 2791 tok; estimated text 478 tok; estimated non-text 2313
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe
>
> Description: Two sailing boats moored on a river with trees behind on the
> bank
>
> Keywords: Bird, Boat, Boating, Buoy, Bushes, Coast, Estuary, Foliage,
> Forest, Landscape, Mast, Moored, Mudflat, Nature, Outdoors, Peaceful,
> Rigging, River, Riverbank, Sailboat
>
> Context: Authoritative context:
> &#45; Location terms: Deben Estuary, England, UK, Woodbridge
> &#45; Capture date/time: 2026-07-04 19:10:04 BST 19:10:04
>
> &#8203;Metadata hints:
> &#45; Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe
> &#45; Description: Two sailing boats moored on a river with trees behind on the
> bank
> &#45; Keywords: Bird, Boat, Boating, Buoy, Bushes, Coast, Estuary, Foliage,
> Forest, Landscape, Mast, Moored, Mudflat, Nature, Outdoors, Peaceful,
> Rigging, River, Riverbank, Sailboat
>
> &#8203;Metadata:
> &#45; Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe
> &#45; Description: Two sailing boats moored on a river with trees behind on the
> bank
> &#45; Keywords: Bird, Boat, Boating, Buoy, Bushes, Coast, Estuary, Foliage,
> Forest, Landscape, Mast, Moored, Mudflat, Nature, Outdoors, Peaceful,
> Rigging, River, Riverbank, Sailboat
>
> &#8203;Metadata hints:
> &#45; Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe
> &#45; Description: Two sailing boats moored on a river with trees behind on the
> bank
> &#45; Keywords: Bird, Boat, Boating, Buoy, Bushes, Coast, Estuary, Foliage,
> Forest, Landscape, Mast, Moored, Mudflat, Nature, Outdoors, Peaceful,
> Rigging, River, Riverbank, Sailboat
>
> &#8203;Metadata hints:
> &#45; Title: Deben Estuary,
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Keyword count violation (27; expected 10-18)
- Unverified context copy
- Low draft improvement
- Likely capped by max token budget

---

<a id="model-mlx-community-paligemma2-3b-pt-896-4bit"></a>

### ✅ mlx-community/paligemma2-3b-pt-896-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Bird, Boat, Boating, Buoy, Bushes; excessive
  bullets=46
- _Score:_ 🟠 D (49/100); Δ+0; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, generation loop, missing
  sections, repetitive)
- _Timing:_ Load 1.28s; Gen 11.95s; Total 13.61s
- _Throughput:_ Prompt 3,912 TPS (4,654 tok); Gen 50.0 TPS (500 tok)
- _Tokens:_ prompt 4654 tok; estimated text 478 tok; estimated non-text 4176
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#45; Output only the description text after the label.
> &#45; Output only the keyword list after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the keyword list after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the keyword list after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the keyword list after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the keyword list after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the keyword list after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the keyword list after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the keyword list after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the keyword list after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the keyword list after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the keyword list after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the keyword list after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the keyword list after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the keyword list after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the keyword list after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the keyword list after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the keyword list after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the keyword list after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the keyword list after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the keyword list after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the keyword list after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the keyword list after the label.
> &#45; Output only the description text after the label.
> &#45; Output only the keyword
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "- output only the...")
- Excessive bullet points (46)
- Context ignored (missing: Bird, Boat, Boating, Buoy, Bushes)
- Generation loop (repetitive_tail)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-glm-46v-flash-mxfp4"></a>

### ✅ mlx-community/GLM-4.6V-Flash-mxfp4

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ hit token cap (500); missing terms: Bird, Boating, Bushes,
  Coast, Forest; keyword duplication=80%; repetitive token=phrase: "black
  hull, white superstructu..."
- _Score:_ ✅ B (79/100); Δ+30; None identified
- _Review focus:_ watchlist (cutoff, generation loop, repetitive)
- _Timing:_ Load 5.15s; Gen 13.72s; Total 21.21s
- _Throughput:_ Prompt 947 TPS (6,592 tok); Gen 80.0 TPS (500 tok)
- _Tokens:_ prompt 6592 tok; estimated text 478 tok; estimated non-text 6114
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Two sailing boats moored on a river
>
> &#8203;Description:
> Two sailing boats are moored on a river. The boats have orange masts and red
> covers. The riverbank is covered with green foliage and trees. A buoy is
> visible in the water near the boats.
>
> &#8203;Keywords:
> Boat, Moored, River, Sailing boat, Mast, Rigging, Foliage, Riverbank,
> Estuary, Water, Buoy, Red cover, Orange mast, Green foliage, Trees,
> Riverbank vegetation, Mooring buoy, Sailing rig, Mast head, Sail cover,
> Wooden hull, Black hull, White superstructure, Red sail cover, Orange mast,
> White rigging, Red cover, Black hull, White superstructure, Red sail cover,
> Orange mast, White rigging, Red cover, Black hull, White superstructure, Red
> sail cover, Orange mast, White rigging, Red cover, Black hull, White
> superstructure, Red sail cover, Orange mast, White rigging, Red cover, Black
> hull, White superstructure, Red sail cover, Orange mast, White rigging, Red
> cover, Black hull, White superstructure, Red sail cover, Orange mast, White
> rigging, Red cover, Black hull, White superstructure, Red sail cover,
> Orange mast, White rigging, Red cover, Black hull, White superstructure, Red
> sail cover, Orange mast, White rigging, Red cover, Black hull, White
> superstructure, Red sail cover, Orange mast, White rigging, Red cover, Black
> hull, White superstructure, Red sail cover, Orange mast, White rigging, Red
> cover, Black hull, White superstructure, Red sail cover, Orange mast, White
> rigging, Red cover, Black hull, White superstructure, Red sail cover,
> Orange mast, White rigging, Red cover, Black hull, White superstructure, Red
> sail cover, Orange mast, White rigging, Red cover, Black hull, White
> superstructure, Red sail cover, Orange mast, White rigging, Red cover, Black
> hull, White superstructure, Red sail cover, Orange mast, White rigging, Red
> cover, Black hull, White superstructure, Red sail cover, Orange mast, White
> rigging, Red cover, Black hull, White superstructure, Red sail cover,
> Orange mast, White rigging, Red cover, Black hull, White superstructure, Red
> sail cover, Orange mast, White rigging, Red cover, Black
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "black hull, white superstructu...")
- Generation loop (repetitive_tail)
- Description sentence violation (4; expected 1-2)
- Keyword count violation (130; expected 10-18)
- Keyword duplication (80% duplicated terms)
- Likely capped by max token budget

---

<a id="model-mlx-community-glm-46v-flash-6bit"></a>

### ✅ mlx-community/GLM-4.6V-Flash-6bit

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a visual input burden issue first; reduce that
  input load or inspect long-context handling before judging output quality.
- _Key signals:_ output/prompt=1.46%; visual input burden=93%; missing
  sections: title; missing terms: Bird, Boating, Buoy, Bushes, Coast
- _Score:_ 🏆 A (82/100); Δ+33; None identified
- _Review focus:_ watchlist (hallucination, missing sections)
- _Timing:_ Load 1.71s; Gen 15.44s; Total 17.55s
- _Throughput:_ Prompt 769 TPS (6,592 tok); Gen 15.2 TPS (96 tok)
- _Tokens:_ prompt 6592 tok; estimated text 478 tok; estimated non-text 6114
  tok; generated 96 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> <|begin_of_box|>Title:
> Two sailboats moored on water with trees in background
>
> &#8203;Description:
> Two sailboats with orange masts and red covers are moored on calm water.
> Behind them is a dense area of green trees and foliage along the bank.
>
> &#8203;Keywords:
> Sailboat, Sailboats, Orange mast, Red cover, Moored, Water, Trees, Green
> foliage, Bank, Calm water, Mast, Rigging, Boat, Estuary, Nature,
> Outdoors<|end_of_box|>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context_budget
- Missing sections (title)
- Contains unexpected table

---

<a id="model-mlx-community-apriel-15-15b-thinker-6bit-mlx"></a>

### ✅ mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, keywords;
  missing terms: Bird, Boating, Bushes, Coast, Forest; reasoning leak
- _Score:_ ✅ B (78/100); Δ+29; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, generation loop, missing sections,
  reasoning leak)
- _Timing:_ Load 1.98s; Gen 15.59s; Total 17.93s
- _Throughput:_ Prompt 1,553 TPS (3,457 tok); Gen 38.8 TPS (500 tok)
- _Tokens:_ prompt 3457 tok; estimated text 478 tok; estimated non-text 2979
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Here are my reasoning steps:
> We need to produce a catalog entry with Title, Description, Keywords. Use
> only details that are clearly visible. The image shows a river (likely Deben
> Estuary) with two sailing boats moored. The larger boat is a sailboat with
> multiple masts (looks like a ketch or maybe a schooner). The hull is white
> with orange stripe and black bottom. The name "Mona Lisa" is visible on the
> side? Actually the text on the hull appears to be "Mona Lisa"? Let's check:
> The hull has "Mona Lisa" written in orange? The image shows "Mona Lisa" in
> orange letters on the side of the boat. The second smaller boat is a small
> motorboat? Actually it's a small white boat with a red canopy. It appears to
> be a small motorboat or a small sailing boat? It has a red canvas cover. It
> is moored next to the larger boat. There's a yellow buoy floating near the
> small boat. The background is dense green foliage, trees, shrubs, a
> riverbank. The water is calm, reflecting the trees. The lighting appears to
> be late afternoon/evening, with warm light, maybe golden hour. The sky is
> not visible. The masts are tall, with rigging lines. The sails are not
> hoisted. The larger boat has multiple masts with rigging. The small boat has
> a red canopy. There's a yellow buoy. The water is a muted blue-gray. The
> overall scene is peaceful.
>
> We need to produce Title: 5-10 words, concrete and factual, limited to
> clearly visible content. So something like "Two moored sailboats on a calm
> river with green banks". That's 10 words? Let's count: "Two" (1) "moored"
> (2) "sailboats" (3) "on" (4) "a" (5) "calm" (6) "river" (7) "with" (8)
> "green" (9) "banks" (10). That's 10 words. That's okay. Or "Two moored
> sailboats on a calm river with green banks". That's fine.
>
> Description: 1-2 factual sentences describing main visible subject, setting,
> &#8203;lighting, action, and other distinctive visible details. So something like:
> "Two sailing boats
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Text sanity issue (numeric_loop)
- Generation loop (numeric_loop)
- Missing sections (title, keywords)
- Reasoning leak (here are my reasoning steps)
- Likely capped by max token budget

---

<a id="model-mlx-community-glm-46v-nvfp4"></a>

### ✅ mlx-community/GLM-4.6V-nvfp4

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a visual input burden issue first; reduce that
  input load or inspect long-context handling before judging output quality.
- _Key signals:_ output/prompt=1.70%; visual input burden=93%; missing
  sections: title; missing terms: Bird, Coast, Deben, GBR, bank
- _Score:_ 🟡 C (56/100); Δ+7; Limited novel information
- _Review focus:_ watchlist (hallucination, missing sections)
- _Timing:_ Load 5.85s; Gen 15.63s; Total 21.86s
- _Throughput:_ Prompt 563 TPS (6,592 tok); Gen 32.7 TPS (112 tok)
- _Tokens:_ prompt 6592 tok; estimated text 478 tok; estimated non-text 6114
  tok; generated 112 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> <|begin_of_box|>Title:
> Two sailing boats moored on a river with trees behind
>
> &#8203;Description:
> Two sailing boats are moored on a river, with dense green trees and foliage
> visible on the riverbank behind them. The water appears calm, and the boats
> have visible masts and rigging.
>
> &#8203;Keywords:
> Boat, Boating, Buoy, Bushes, Estuary, Foliage, Forest, Landscape, Mast,
> Moored, Mudflat, Nature, Outdoors, Peaceful, Rigging, River, Riverbank,
> Sailboat, Trees<|end_of_box|>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context_budget
- Missing sections (title)
- Keyword count violation (19; expected 10-18)
- Low draft improvement
- Contains unexpected table

---

<a id="model-mlx-community-molmo-7b-d-0924-bf16"></a>

### ✅ mlx-community/Molmo-7B-D-0924-bf16

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: GBR, bank; keywords=20; low-draft-improvement
- _Score:_ 🟡 C (59/100); Δ+10; None identified
- _Timing:_ Load 1.88s; Gen 19.55s; Total 21.87s
- _Throughput:_ Prompt 118 TPS (1,749 tok); Gen 30.3 TPS (114 tok)
- _Tokens:_ prompt 1749 tok; estimated text 478 tok; estimated non-text 1271
  tok; generated 114 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Remove uncertain tags.
>
> Title: Sailing boats on the River Deben near Woodbridge, England
>
> Description: Two sailboats are moored on the River Deben with trees and
> foliage visible on the riverbank behind them. The boats are positioned in
> calm waters, creating a serene coastal scene.
>
> Keywords: Bird, Boat, Boating, Buoy, Bushes, Coast, Estuary, Foliage,
> Forest, Landscape, Mast, Moored, Mudflat, Nature, Outdoors, Peaceful,
> Rigging, River, Riverbank, Sailboat
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Keyword count violation (20; expected 10-18)
- Low draft improvement

---

<a id="model-mlx-community-qwen3-vl-2b-instruct-bf16"></a>

### ✅ mlx-community/Qwen3-VL-2B-Instruct-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect KV/cache behavior, memory pressure, and long-context
  execution.
- _Key signals:_ At mixed burden (16779 tokens), output became repetitive.;
  hit token cap (500); missing sections: title, description, keywords; missing
  terms: Bird, Boat, Boating, Buoy, Bushes
- _Score:_ ❌ F (20/100); Δ-29; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, generation
  loop, harness, long context, missing sections, repetitive, text sanity)
- _Timing:_ Load 0.86s; Gen 21.00s; Total 22.25s
- _Throughput:_ Prompt 1,147 TPS (16,779 tok); Gen 91.9 TPS (500 tok)
- _Tokens:_ prompt 16779 tok; estimated text 478 tok; estimated non-text 16301
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
- Context ignored (missing: Bird, Boat, Boating, Buoy, Bushes)
- Output degeneration (character_loop: ' -' repeated)
- Text sanity issue (gibberish(token_noise))
- Generation loop (repetitive_tail)
- Missing sections (title, description, keywords)
- Low draft improvement
- Likely capped by max token budget
- Ignores trusted hints
- long_context_repetition(16779tok)

---

<a id="model-mlx-community-molmopoint-8b-fp16"></a>

### ✅ mlx-community/MolmoPoint-8B-fp16

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Deben, GBR, sailing, trees; keywords=19;
  low-draft-improvement
- _Score:_ 🟠 D (48/100); Δ-1; Lacks visual description of image
- _Review focus:_ watchlist (Lacks visual description of image)
- _Timing:_ Load 2.36s; Gen 21.20s; Total 23.94s
- _Throughput:_ Prompt 1,318 TPS (3,377 tok); Gen 5.95 TPS (106 tok)
- _Tokens:_ prompt 3377 tok; estimated text 478 tok; estimated non-text 2899
  tok; generated 106 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Two moored sailboats on a river with forested bank
>
> &#8203;Description:
> Two sailboats are anchored on a calm river, with a dense forest visible on
> the opposite bank. The boats have masts and rigging, but no sails are
> visible.
>
> &#8203;Keywords:
> Boat, Boating, Buoy, Bushes, Coast, Estuary, Foliage, Forest, Landscape,
> Mast, Moored, Mudflat, Nature, Outdoors, Peaceful, Rigging, River,
> Riverbank, Sailboat
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Keyword count violation (19; expected 10-18)
- Low draft improvement

---

<a id="model-mlx-community-molmo-7b-d-0924-8bit"></a>

### ✅ mlx-community/Molmo-7B-D-0924-8bit

- _Recommendation:_ use with caveats; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: GBR, bank; keywords=20; low-draft-improvement
- _Score:_ 🟡 C (59/100); Δ+10; None identified
- _Timing:_ Load 4.42s; Gen 21.69s; Total 28.52s
- _Throughput:_ Prompt 119 TPS (1,749 tok); Gen 50.9 TPS (114 tok)
- _Tokens:_ prompt 1749 tok; estimated text 478 tok; estimated non-text 1271
  tok; generated 114 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Remove uncertain tags.
>
> Title: Sailing boats on the River Deben near Woodbridge, England
>
> Description: Two sailboats are moored on the River Deben with trees and
> foliage visible on the riverbank behind them. The boats are positioned in
> calm waters, creating a serene coastal scene.
>
> Keywords: Bird, Boat, Boating, Buoy, Bushes, Coast, Estuary, Foliage,
> Forest, Landscape, Mast, Moored, Mudflat, Nature, Outdoors, Peaceful,
> Rigging, River, Riverbank, Sailboat
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Keyword count violation (20; expected 10-18)
- Low draft improvement

---

<a id="model-qwen-qwen3-vl-2b-instruct"></a>

### ✅ Qwen/Qwen3-VL-2B-Instruct

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect KV/cache behavior, memory pressure, and long-context
  execution.
- _Key signals:_ At mixed burden (16779 tokens), output became repetitive.;
  hit token cap (500); missing sections: title, description, keywords; missing
  terms: Bird, Boat, Boating, Buoy, Bushes
- _Score:_ ❌ F (20/100); Δ-29; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, generation
  loop, harness, long context, missing sections, repetitive, text sanity)
- _Timing:_ Load 0.69s; Gen 21.79s; Total 22.84s
- _Throughput:_ Prompt 1,078 TPS (16,779 tok); Gen 92.5 TPS (500 tok)
- _Tokens:_ prompt 16779 tok; estimated text 478 tok; estimated non-text 16301
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
- Context ignored (missing: Bird, Boat, Boating, Buoy, Bushes)
- Output degeneration (character_loop: ' -' repeated)
- Text sanity issue (gibberish(token_noise))
- Generation loop (repetitive_tail)
- Missing sections (title, description, keywords)
- Low draft improvement
- Likely capped by max token budget
- Ignores trusted hints
- long_context_repetition(16779tok)

---

<a id="model-mlx-community-paligemma2-3b-ft-docci-448-bf16"></a>

### ✅ mlx-community/paligemma2-3b-ft-docci-448-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Bird, Boat, Boating, Buoy, Bushes; excessive
  bullets=34
- _Score:_ 🟡 C (54/100); Δ+5; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, generation loop, missing
  sections, repetitive)
- _Timing:_ Load 1.61s; Gen 27.52s; Total 29.49s
- _Throughput:_ Prompt 3,427 TPS (1,582 tok); Gen 18.9 TPS (500 tok)
- _Tokens:_ prompt 1582 tok; estimated text 478 tok; estimated non-text 1104
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#45; Do not copy the text after the label.
> &#45; Do not infer the intent of the image.
> &#45; Use this factual context where it improves the catalogue record; do not
> claim that contextual facts are visually observable.
> &#45; Do not copy the text after the label.
> &#45; Do not infer the intent of the image.
> &#45; Use this factual context where it improves the catalogue record; do not
> claim that contextual facts are visually observable.
> &#45; Do not copy the text after the label.
> &#45; Do not infer the intent of the image.
> &#45; Use this factual context where it improves the catalogue record; do not
> claim that contextual facts are visually observable.
> &#45; Do not copy the text after the label.
> &#45; Do not infer the intent of the image.
> &#45; Use this factual context where it improves the catalogue record; do not
> claim that contextual facts are visually observable.
> &#45; Do not copy the text after the label.
> &#45; Do not infer the intent of the image.
> &#45; Use this factual context where it improves the catalogue record; do not
> claim that contextual facts are visually observable.
> &#45; Do not copy the text after the label.
> &#45; Do not infer the intent of the image.
> &#45; Use this factual context where it improves the catalogue record; do not
> claim that contextual facts are visually observable.
> &#45; Do not copy the text after the label.
> &#45; Do not infer the intent of the image.
> &#45; Use this factual context where it improves the catalogue record; do not
> claim that contextual facts are visually observable.
> &#45; Do not copy the text after the label.
> &#45; Do not infer the intent of the image.
> &#45; Use this factual context where it improves the catalogue record; do not
> claim that contextual facts are visually observable.
> &#45; Do not copy the text after the label.
> &#45; Do not infer the intent of the image.
> &#45; Use this factual context where it improves the catalogue record; do not
> claim that contextual facts are visually observable.
> &#45; Do not copy the text after the label.
> &#45; Do not infer the intent of the image.
> &#45; Use this factual context where it improves the catalogue record; do not
> claim that contextual facts are visually observable.
> &#45; Do not copy the text after the label.
> &#45; Do not infer the intent of the image.
> &#45; Use this factual context where it improves the catalogue record; do not
> claim that contextual facts are visually observable.
> &#45; Do not copy the
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Repetitive output (phrase: "- do not copy...")
- Excessive bullet points (34)
- Context ignored (missing: Bird, Boat, Boating, Buoy, Bushes)
- Generation loop (repetitive_tail)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-meta-llama-llama-32-11b-vision-instruct"></a>

### ✅ meta-llama/Llama-3.2-11B-Vision-Instruct

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Bird, Boating, Buoy, Bushes, Coast; low-draft-improvement
- _Score:_ 🟠 D (47/100); Δ-2; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (missing sections, trusted hint degraded)
- _Timing:_ Load 2.36s; Gen 42.44s; Total 45.18s
- _Throughput:_ Prompt 288 TPS (544 tok); Gen 3.35 TPS (134 tok)
- _Tokens:_ prompt 544 tok; estimated text 478 tok; estimated non-text 66 tok;
  generated 134 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Sailboats Moored in the Deben Estuary
>
> The image depicts two sailboats moored in the Deben Estuary, with a serene
> and peaceful atmosphere. The boats are situated in the water, with a dense
> forest of trees in the background. The scene is characterized by the
> calmness of the water and the lush greenery of the surrounding environment.
> The sailboats are moored to the water, with their masts and rigging visible,
> and the forest provides a natural backdrop to the scene. The overall mood
> of the image is one of tranquility and serenity, with the sailboats and the
> forest creating a sense of harmony and balance.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)
- Low draft improvement
- Degrades trusted hints

---

<a id="model-mlx-community-x-reasoner-7b-8bit"></a>

### ✅ mlx-community/X-Reasoner-7B-8bit

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Treat this as a mixed burden issue first; reduce that input
  load or inspect long-context handling before judging output quality.
- _Key signals:_ Output is very short relative to prompt size (0.1%) with weak
  text signal 'truncated', suggesting possible early-stop or prompt-handling
  issues.; At mixed burden (16790 tokens), output stayed unusually short (14
  tokens; ratio 0.1%; weak text signal truncated).; output/prompt=0.08%; mixed
  burden=97%
- _Score:_ ❌ F (4/100); Δ-45; Output too short to be useful
- _Review focus:_ watchlist (context ignored, harness, long context, missing
  sections)
- _Timing:_ Load 3.14s; Gen 55.00s; Total 60.31s
- _Throughput:_ Prompt 336 TPS (16,790 tok); Gen 46.4 TPS (14 tok)
- _Tokens:_ prompt 16790 tok; estimated text 478 tok; estimated non-text 16312
  tok; generated 14 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> <|im_start|>
> addCriterion("图片描述").get("sailboat")
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context_budget
- ⚠️HARNESS:long_context
- Context ignored (missing: Bird, Boat, Boating, Buoy, Bushes)
- Missing sections (title, description, keywords)
- Low draft improvement
- Ignores trusted hints
- output:output_ratio(0.1%;truncated)
- long_context_low_ratio(0.1%;16790->14;truncated)

---

<a id="model-mlx-community-ornith-10-35b-bf16"></a>

### ✅ mlx-community/Ornith-1.0-35B-bf16

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Bushes, Coast, Mudflat
- _Score:_ 🏆 A (82/100); Δ+33; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 6.07s; Gen 58.21s; Total 64.68s
- _Throughput:_ Prompt 307 TPS (16,808 tok); Gen 56.1 TPS (137 tok)
- _Tokens:_ prompt 16808 tok; estimated text 478 tok; estimated non-text 16330
  tok; generated 137 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Two sailboats moored on the Deben Estuary with a wooded bank
>
> &#8203;Description:
> Two sailboats with orange masts are moored on the water of the Deben
> Estuary, featuring a dark-hulled vessel in the foreground and a light-hulled
> boat behind it. A dense green forest covers the riverbank in the
> background, while a buoy floats in the foreground water.
>
> &#8203;Keywords:
> Sailboat, Deben Estuary, Woodbridge, England, Moored, Riverbank, Mast,
> Rigging, Buoy, Foliage, Trees, Water, Nature, Outdoors, Landscape, Estuary,
> Boat, Foliage
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Title length violation (11 words; expected 5-10)

---

<a id="model-mlx-community-qwen35-35b-a3b-6bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-6bit

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boat, Boating, Bushes, Coast
- _Score:_ 🏆 A (92/100); Δ+42; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 3.35s; Gen 59.23s; Total 62.95s
- _Throughput:_ Prompt 296 TPS (16,808 tok); Gen 68.7 TPS (113 tok)
- _Tokens:_ prompt 16808 tok; estimated text 478 tok; estimated non-text 16330
  tok; generated 113 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Two moored sailboats on the Deben Estuary with red covers
>
> Description: Two sailing yachts with tall wooden masts are moored on the
> water, featuring red canvas covers over their booms. A dense green tree line
> and muddy riverbank form the background under natural daylight.
>
> Keywords: Deben Estuary, sailboats, moored, wooden masts, red covers,
> riverbank, trees, foliage, water, buoys, rigging, estuary, nature, outdoors,
> calm, green, UK, Woodbridge
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

---

<a id="model-mlx-community-qwen3-vl-2b-thinking-bf16"></a>

### ✅ mlx-community/Qwen3-VL-2B-Thinking-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect KV/cache behavior, memory pressure, and long-context
  execution.
- _Key signals:_ At mixed burden (16781 tokens), output became repetitive.;
  hit token cap (500); missing sections: title, description, keywords; missing
  terms: Bird, Boat, Boating, Buoy, Bushes
- _Score:_ ❌ F (20/100); Δ-29; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, generation
  loop, harness, long context, missing sections, repetitive, text sanity)
- _Timing:_ Load 0.97s; Gen 66.78s; Total 68.35s
- _Throughput:_ Prompt 354 TPS (16,781 tok); Gen 27.9 TPS (500 tok)
- _Tokens:_ prompt 16781 tok; estimated text 478 tok; estimated non-text 16303
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
- Context ignored (missing: Bird, Boat, Boating, Buoy, Bushes)
- Output degeneration (character_loop: ' -' repeated)
- Text sanity issue (gibberish(token_noise))
- Generation loop (repetitive_tail)
- Missing sections (title, description, keywords)
- Low draft improvement
- Likely capped by max token budget
- Ignores trusted hints
- long_context_repetition(16781tok)

---

<a id="model-mlx-community-qwen35-9b-mlx-4bit"></a>

### ✅ mlx-community/Qwen3.5-9B-MLX-4bit

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Bushes, Coast, Estuary
- _Score:_ ✅ B (78/100); Δ+30; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 1.56s; Gen 67.05s; Total 69.00s
- _Throughput:_ Prompt 258 TPS (16,808 tok); Gen 91.4 TPS (99 tok)
- _Tokens:_ prompt 16808 tok; estimated text 478 tok; estimated non-text 16330
  tok; generated 99 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Two moored sailboats on a riverbank
>
> &#8203;Description:
> Two small sailing boats are moored side by side on calm water, with their
> sails furled and masts visible. The background consists of a densely wooded
> riverbank with lush green trees and undergrowth.
>
> &#8203;Keywords:
> Boat, Boats, Moored, River, Riverbank, Trees, Forest, Greenery, Nature,
> Outdoors, Mast, Rigging, Buoy, Water, Landscape
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

---

<a id="model-mlx-community-qwen35-35b-a3b-bf16"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-bf16

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Bushes, Coast, Forest;
  low-draft-improvement
- _Score:_ ✅ B (67/100); Δ+18; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 6.62s; Gen 68.07s; Total 75.08s
- _Throughput:_ Prompt 256 TPS (16,808 tok); Gen 59.7 TPS (93 tok)
- _Tokens:_ prompt 16808 tok; estimated text 478 tok; estimated non-text 16330
  tok; generated 93 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Two wooden sailboats moored on the Deben Estuary
> Description: Two wooden sailing yachts with tall masts are moored on calm
> water against a backdrop of dense green trees and a muddy riverbank.
> Keywords: Deben Estuary, sailboat, wooden boat, mast, rigging, moored,
> riverbank, trees, foliage, green, water, buoy, estuary, UK, nature,
> outdoors, calm, sailing
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Low draft improvement

---

<a id="model-mlx-community-qwen35-35b-a3b-4bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-4bit

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Bushes, Coast, Landscape
- _Score:_ ✅ B (72/100); Δ+23; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 2.75s; Gen 73.62s; Total 76.73s
- _Throughput:_ Prompt 236 TPS (16,808 tok); Gen 68.2 TPS (105 tok)
- _Tokens:_ prompt 16808 tok; estimated text 478 tok; estimated non-text 16330
  tok; generated 105 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Two wooden sailing boats moored on the Deben Estuary
>
> Description: Two wooden sailing boats with tall masts are moored in the
> water, backed by a dense green riverbank. A buoy floats in the foreground
> while the boats display names on their sterns.
>
> Keywords: Deben Estuary, Woodbridge, England, UK, Europe, Sailboat, Boat,
> Moored, River, Water, Trees, Foliage, Forest, Green, Wooden, Mast, Rigging,
> Buoy
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

---

<a id="model-mlx-community-qwen2-vl-2b-instruct-4bit"></a>

### ✅ mlx-community/Qwen2-VL-2B-Instruct-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect KV/cache behavior, memory pressure, and long-context
  execution.
- _Key signals:_ At mixed burden (16790 tokens), output became repetitive.;
  hit token cap (500); missing sections: description, keywords; missing terms:
  Bird, Buoy, Bushes, Coast, Estuary
- _Score:_ 🟠 D (50/100); Δ+0; Lacks visual description of image
- _Review focus:_ watchlist (cutoff, generation loop, harness, long context,
  missing sections, repetitive, trusted hint degraded)
- _Timing:_ Load 2.45s; Gen 79.52s; Total 82.62s
- _Throughput:_ Prompt 222 TPS (16,790 tok); Gen 207 TPS (500 tok)
- _Tokens:_ prompt 16790 tok; estimated text 478 tok; estimated non-text 16312
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> -001.jpg
> &#45; 10-18 unique comma-separated terms based only on clearly visible subjects,
> setting, lighting, action, and other distinctive visible details. Omit
> anything uncertain or inferred.
> &#45; Output only the keyword list after the label.
>
> Title: Boating, Boats, River, Trees, Mudflat, Boats, Boats, Boats, Boats,
> Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats,
> Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats,
> Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats,
> Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats,
> Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats,
> Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats,
> Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats,
> Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats,
> Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats,
> Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats,
> Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats,
> Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats,
> Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats, Boats,
> Boats, Boats, Boats,
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:long_context
- Repetitive output (phrase: "boats, boats, boats, boats,...")
- Generation loop (repetitive_tail)
- Missing sections (description, keywords)
- Title length violation (149 words; expected 5-10)
- Low draft improvement
- Likely capped by max token budget
- Degrades trusted hints
- long_context_repetition(16790tok)

---

<a id="model-mlx-community-qwen35-27b-mxfp8"></a>

### ✅ mlx-community/Qwen3.5-27B-mxfp8

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Bushes, Coast, Foliage
- _Score:_ 🏆 A (90/100); Δ+41; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 3.38s; Gen 99.24s; Total 103.01s
- _Throughput:_ Prompt 184 TPS (16,808 tok); Gen 18.3 TPS (131 tok)
- _Tokens:_ prompt 16808 tok; estimated text 478 tok; estimated non-text 16330
  tok; generated 131 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Two Moored Sailboats on River with Forested Bank
>
> &#8203;Description:
> Two sailboats with wooden masts are moored on calm water, positioned in
> front of a densely vegetated riverbank. The foreground boat features a dark
> hull with a wooden transom displaying the name "SIOLA" and "ESSC", while the
> boat behind has a light green hull and red sail covers.
>
> &#8203;Keywords:
> Sailboat, Moored, River, Forest, Trees, Masts, Rigging, Hull, Water, Nature,
> Landscape, Vegetation, Buoy, Deck, Estuary, Wood, Greenery, Outdoors
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

---

<a id="model-mlx-community-qwen35-27b-4bit"></a>

### ✅ mlx-community/Qwen3.5-27B-4bit

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Bushes, Coast, Mudflat
- _Score:_ 🏆 A (87/100); Δ+38; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 2.85s; Gen 100.84s; Total 106.12s
- _Throughput:_ Prompt 177 TPS (16,808 tok); Gen 23.9 TPS (121 tok)
- _Tokens:_ prompt 16808 tok; estimated text 478 tok; estimated non-text 16330
  tok; generated 121 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Two sailboats moored on river with forested bank
>
> &#8203;Description:
> Two sailboats with furled sails are moored on calm water in front of a
> densely vegetated riverbank. The foreground vessel has a dark hull with a
> wooden stern displaying the name 'SIOLA', while a second boat with a light
> hull is positioned slightly behind it.
>
> &#8203;Keywords:
> Sailboat, River, Forest, Moored, Buoy, Mast, Rigging, Hull, Water,
> Vegetation, Trees, Nature, Landscape, Sailing, Boat, Estuary, Foliage
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-2506-bf16"></a>

### ✅ mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title; missing terms:
  Mudflat, Peaceful, Riverbank, GBR, sailing; keyword duplication=50%
- _Score:_ 🟡 C (57/100); Δ+8; Lacks visual description of image
- _Review focus:_ watchlist (cutoff, missing sections, reasoning leak)
- _Timing:_ Load 3.57s; Gen 118.17s; Total 122.12s
- _Throughput:_ Prompt 539 TPS (1,564 tok); Gen 4.37 TPS (500 tok)
- _Tokens:_ prompt 1564 tok; estimated text 478 tok; estimated non-text 1086
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ◁think▷Got it, let's analyze the image step by step.
>
> First, the title: The image shows two sailboats on a body of water with
> dense green trees in the background. The location is Deben Estuary, England,
> UK, Europe. So the title should be concise. Let's pick "Two sailboats
> moored on Deben Estuary" as it's factual and visible.
>
> Description: The main subject is two sailboats on water, with dense green
> foliage and trees in the background. The lighting is natural daylight, and
> the scene is outdoors. So the description would be: "Two sailboats moored on
> a river with dense green trees and foliage in the background."
>
> Keywords: Let's list visible elements: Sailboats, River, Trees, Foliage,
> Water, Moored, Buoy, Mast, Rigging, Nature, Outdoors, Landscape. Need 10-18
> terms. Let's check each: Sailboats (2), River (Deben Estuary), Trees (dense
> green), Foliage, Water, Moored, Buoy (small orange), Mast (two masts),
> Rigging (visible on boats), Nature, Outdoors, Landscape. That's 11 terms.
> Maybe add "Boats" instead of "Sailboats"? Wait, "Sailboats" is better. Let's
> check again. The existing draft had "Boating" but maybe "Sailboats" is more
> specific. Let's list: Sailboats, River, Trees, Foliage, Water, Moored,
> Buoy, Mast, Rigging, Nature, Outdoors, Landscape. That's 10. Maybe add
> "Boats" but "Sailboats" is better. Let's confirm. The image has two
> sailboats, so "Sailboats" is accurate. So keywords: Sailboats, River, Trees,
> Foliage, Water, Moored, Buoy, Mast, Rigging, Nature, Outdoors, Landscape.
>
> Wait, the existing draft had "Bird" but there's no bird in the image. So
> remove "Bird". Also, "Coast" wasn't visible. "Bushes" was in the draft but
> the image shows dense trees, not bushes. So remove "Bushes". "Forest" might
> be too broad; it's trees. "Estuary" is part of the location but maybe not
> &#8203;visible as a keyword. Let's stick to what's visible. So final keywords:
> Sailboats, River, Trees, Foli
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title)
- Description sentence violation (3; expected 1-2)
- Keyword count violation (54; expected 10-18)
- Keyword duplication (50% duplicated terms)
- Reasoning leak (◁think▷, let's analyze the image)
- Low draft improvement
- Likely capped by max token budget

---

<a id="model-mlx-community-qwen36-27b-mxfp8"></a>

### ✅ mlx-community/Qwen3.6-27B-mxfp8

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Bushes, Coast, Forest
- _Score:_ 🏆 A (85/100); Δ+36; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 4.46s; Gen 137.78s; Total 142.77s
- _Throughput:_ Prompt 141 TPS (16,808 tok); Gen 6.35 TPS (109 tok)
- _Tokens:_ prompt 16808 tok; estimated text 478 tok; estimated non-text 16330
  tok; generated 109 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Two sailing boats moored on a river
>
> &#8203;Description:
> Two sailing boats are moored on calm water, with the foreground vessel
> featuring a black hull, wooden transom, and tall orange masts. A dense
> backdrop of green trees and foliage lines the riverbank behind them.
>
> &#8203;Keywords:
> Sailing boat, Moored, River, Water, Trees, Foliage, Nature, Landscape, Mast,
> Rigging, Hull, Buoy, Outdoors, Green, Wood, Vessel, Estuary, Bank
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<!-- markdownlint-enable MD033 MD034 -->
