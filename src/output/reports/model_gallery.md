# Model Output Gallery

Generated on: 2026-07-10 23:37:01 BST

Complete per-model evidence artifact with image metadata, the source prompt,
summary tables, diagnostics, and full generated output for every attempted
model.

_Action Snapshot: see [results.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/results.md) for the full summary._

## Run Stamps

- `mlx-vlm`: `0.6.5`
- `mlx`: `0.32.1.dev20260710+4367c73b`
- `mlx-lm`: `0.31.3`
- `transformers`: `5.13.0`
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
> Context: Existing metadata hints (high confidence; use only when visually
> &#8203;confirmed):
> &#45; Title hint: , Deben Estuary, Woodbridge, England, UK, GBR, Europe
> &#45; Description hint: Two sailing boats moored on a river with trees behind on
> the bank
> &#45; Keyword hints: Bird, Boat, Boating, Buoy, Bushes, Coast, Deben Estuary,
> England, Estuary, Europe, Foliage, Forest, Landscape, Mast, Moored, Mudflat,
> Nature, Outdoors, Peaceful, Rigging
> &#45; Capture metadata: Taken on 2026-07-04 19:10:04 BST (at 19:10:04 local
> time).
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

## Model Quality Summary

Skim-first view of what each model returned, or the strongest diagnostic when it did not produce usable output.

<!-- markdownlint-disable MD034 -->

| Model                                                                                                                   | Result                        | Quality / diagnostic                                         | Response / diagnostic                                                                                                                                                                                                                                                                                                                        |
|-------------------------------------------------------------------------------------------------------------------------|-------------------------------|--------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`mlx-community/gemma-4-31b-bf16`](#model-mlx-community-gemma-4-31b-bf16)                                               | `avoid` / `semantic mismatch` | mlx-vlm; model-error                                         | [mlx-vlm; model-error] Error: Model Error - Model runtime error during generation for mlx-community/gemma-4-31b-bf16: [METAL] Command buffer execution failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory).                                                                                                   |
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `avoid` / `model shortcoming` | metadata-borrowing                                           | Title: Deben Estuary Boats on a River Description: - Two sailboats moored on a calm river with dense green foliage in the background. - The boats are positioned in the foreground, with the water reflecting th ... [tail] l, Foliage, Forest, Landscape, Mast, Moored, Mudflat, Coast, Deben Estuary, England, Estuary, Europe             |
| [`mlx-community/LFM2-VL-1.6B-8bit`](#model-mlx-community-lfm2-vl-16b-8bit)                                              | `avoid` / `model shortcoming` | metadata-borrowing                                           | Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe Description: Two sailing boats moored on a river with trees behind on the bank Keywords: Boat, Deben Estuary, England, Estuary, Europe, Foliage, Landscape, Nature, Outdoors, Peaceful, Rigging                                                                                   |
| [`qnguyen3/nanoLLaVA`](#model-qnguyen3-nanollava)                                                                       | `avoid` / `model shortcoming` | metadata-borrowing; missing-sections                         | Title: 5-10 boats moored on a river with trees behind Description: Two boats tied to a bank with trees behind them.                                                                                                                                                                                                                          |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                            | `avoid` / `model shortcoming` | metadata-borrowing                                           | Title: 5-10 words, concrete and factual, limited to clearly visible content. Description: Two sailing boats moored on a river with trees behind on the bank. Keywords: Bird, Boat, Boating, Buoy, Bushes, Coast, ... [tail] ary, Europe, Foliage, Forest, Landscape, Mast, Moored, Mudflat, Nature, Outdoors, Peaceful, Rigging.             |
| [`mlx-community/paligemma2-3b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-3b-ft-docci-448-bf16)                 | `avoid` / `harness`           | harness:prompt-template; context-ignored                     | - Daytime.                                                                                                                                                                                                                                                                                                                                   |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `avoid` / `harness`           | harness:stop-token; reasoning-leak; formatting               | &lt;think&gt; &lt;/think&gt; Title: Sailing boats moored near dense green foliage on water Description: The image shows two sailing boats floating on water with lush trees in the background. The scene appears calm with n ... [tail] ds: sailing boats, water, trees, moored, calm, green, nature, river, boats, foliage, peace, outdoors |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `avoid` / `model shortcoming` | metadata-borrowing                                           | Title: - Deben Estuary, Woodbridge, England, UK, GBR, Europe, River, Boating, Sailing, Boats, Mast, Rigging, Buoy, Trees, Landscape, Nature, Estuary, Coast, Foliage, Outdoors, Peaceful Description: - Two sail ... [tail] dscape, Nature, Estuary, River, Coast, Foliage, Outdoors, Peaceful, Woodbridge, England, GBR, Europe             |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `avoid` / `model shortcoming` | metadata-borrowing; missing-sections                         | Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe Description: Two sailing boats moored on a river with trees behind on the bank. The boats are moored close to each other, with one boat slightly ahea ... [tail] reen and brown, with some blue from the water. The overall mood of the image is peaceful and serene.             |
| [`HuggingFaceTB/SmolVLM-Instruct`](#model-huggingfacetb-smolvlm-instruct)                                               | `avoid` / `model shortcoming` | metadata-borrowing                                           | Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe Description: Two sailing boats moored on a river with trees behind on the bank Keywords: Bird, Boat, Boating, Buoy, Bushes, Coast, Deben Estuary, England, Estuary, Europe, Foliage, Forest, Landscape, Moored, Mudflat, Nature, Outdoors, Peaceful, Rigging                      |
| [`mlx-community/SmolVLM-Instruct-bf16`](#model-mlx-community-smolvlm-instruct-bf16)                                     | `avoid` / `model shortcoming` | metadata-borrowing                                           | Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe Description: Two sailing boats moored on a river with trees behind on the bank Keywords: Bird, Boat, Boating, Buoy, Bushes, Coast, Deben Estuary, England, Estuary, Europe, Foliage, Forest, Landscape, Moored, Mudflat, Nature, Outdoors, Peaceful, Rigging                      |
| [`mlx-community/paligemma2-10b-ft-docci-448-6bit`](#model-mlx-community-paligemma2-10b-ft-docci-448-6bit)               | `avoid` / `semantic mismatch` | context-ignored; missing-sections                            | - Location: 52° 0' 0" N, 1° 0' 0" E.                                                                                                                                                                                                                                                                                                         |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `avoid` / `semantic mismatch` | missing terms: Bird, Boating, Buoy, Bushes, Coast            | Title: Two small sailing boats moored on a river Description: Two small wooden sailing boats with tall masts are moored on the water against a backdrop of dense green trees. The scene is captured in natural l ... [tail] g, river, water, trees, foliage, forest, landscape, nature, outdoors, greenery, vertical composition             |
| [`microsoft/Phi-3.5-vision-instruct`](#model-microsoft-phi-35-vision-instruct)                                          | `recommended` / `clean`       | clean                                                        | Title: Sailing Boats at Deben Estuary Description: Two sailing boats are moored on a river, surrounded by lush greenery and trees. The boats are docked on the riverbank, with one boat having a prominent mast ... [tail] ords: Sailing Boats, Deben Estuary, River, Trees, Greenery, Mast, Rigging, Mudflat, Nature, Outdoors              |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `avoid` / `model shortcoming` | metadata-borrowing                                           | Title: - Two sailing boats moored on the Deben Estuary Description: - Two wooden sailboats with tall masts are moored on a river next to a dense green forest bank. A small buoy floats in the water near the ve ... [tail] ary, England, estuary, foliage, forest, landscape, mast, moored, nature, outdoors, peaceful, rigging             |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `avoid` / `semantic mismatch` | missing terms: Bird, Boating, Buoy, Bushes, Coast            | Title: Traditional sailboat moored by riverbank with dense foliage Description: A classic wooden sailboat with tall masts and red sails is moored on calm water beside a grassy shoreline thick with lush green ... [tail] red sails, green foliage, dense trees, nature scene, waterfront, maritime setting, coastal landscape              |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `recommended` / `clean`       | clean                                                        | Title: Sailing Boats at Deben Estuary Description: Two sailing boats are moored on a river, surrounded by lush greenery and trees. The boats are docked on the riverbank, with one boat having a prominent mast ... [tail] ords: Sailing Boats, Deben Estuary, River, Trees, Greenery, Mast, Rigging, Mudflat, Nature, Outdoors              |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `avoid` / `semantic mismatch` | context-echo; instruction-echo; metadata-borrowing           | Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe Description: Two sailing boats moored on a river with trees behind on the bank Keywords: Bird, Boat, Boating, Buoy, Bushes, Coast, Deben Estuary, Eng ... [tail] rs, Peaceful, Rigging - Capture metadata: Taken on 2026-07-04 19:10:04 BST (at 19:10:04 local time).             |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                            | `avoid` / `model shortcoming` | context-ignored; missing-sections                            | The image shows a boat on a river.                                                                                                                                                                                                                                                                                                           |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `avoid` / `model shortcoming` | metadata-borrowing                                           | Title: - Two sailing boats moored on the Deben Estuary Description: - Two small sailing boats are moored on a river in front of a dense green forest on the bank. Keywords: - Boat, Boating, Estuary, Mast, Rigging, River, Forest, Trees, Foliage, Nature, Outdoors, Landscape, Peaceful, England, Europe                                   |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `avoid` / `semantic mismatch` | missing terms: Bird, Boating, Coast, Deben Estuary, Estuary  | Title: Two sailing boats moored on river Description: Two sailing boats with orange masts are moored on a calm river, surrounded by dense green foliage on the bank. Keywords: Boats, Sailing, River, Moored, Foliage, Trees, Mast, Nature, Outdoors, Peaceful, Buoy, Bushes                                                                 |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `avoid` / `semantic mismatch` | missing terms: Bird, Bushes, Coast, Deben Estuary, Estuary   | Title: Two sailing boats moored on a waterway Description: Two sailing boats are moored on calm water, with dense green foliage lining the banks. A single orange buoy is visible in the foreground. Keywords: Boat, Boating, Buoy, Foliage, Green, Landscape, Mast, Moored, Outdoors, River, Sailing, Trees, Water, Waterway                |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `avoid` / `semantic mismatch` | missing terms: Bird, Boating, Bushes, Coast, Deben Estuary   | Title: Sailing Boat on Calm Water with Forest Background Description: A single wooden sailboat with two tall masts and red sails is moored on calm water, surrounded by dense green foliage and trees along the ... [tail] ooden hull, two masts, red sails, calm water, dense forest, green foliage, shoreline, anchored, buoy              |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `caveat` / `context budget`   | metadata-borrowing; context-budget                           | Title: Two Boats Moored on River Description: Two sailing boats are moored on a river with dense green foliage on the bank behind them. Keywords: Boat, Boating, Moored, River, Sailing, Trees, Foliage, Green, Mast, Rigging, Nature, Outdoors, Peaceful, Water, Riverbank, Forest, Europe                                                  |
| [`mlx-community/InternVL3-14B-8bit`](#model-mlx-community-internvl3-14b-8bit)                                           | `avoid` / `semantic mismatch` | missing terms: Bird, Bushes, Coast, Deben Estuary, Estuary   | Title: Sailing Boats on River Description: Two sailing boats are moored on a calm river with dense green foliage on the bank behind them. Keywords: Boat, Boating, Buoy, Foliage, Forest, Landscape, Mast, Moored, Nature, Outdoors, Peaceful, River, Rigging, Trees, Water                                                                  |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `caveat` / `context budget`   | metadata-borrowing; context-budget                           | **Title:** Sailing Boat Moored on Deben Estuary **Description:** A single wooden sailboat with two masts is anchored near a buoy on calm waters, surrounded by dense green foliage along the riverbank. Visible ... [tail] , Calm waters, Buoy, Red sails, Rigging, Moored, Deben Estuary, Natural setting, Woodbridge, England              |
| [`mlx-community/gemma-3n-E2B-4bit`](#model-mlx-community-gemma-3n-e2b-4bit)                                             | `avoid` / `cutoff degraded`   | metadata-borrowing; cutoff; degeneration                     | - 19:10:18:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:...             |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `avoid` / `model shortcoming` | metadata-borrowing; formatting                               | Title: Sailing Boats on the Deben Estuary, Woodbridge, England, UK. Description: Two sailing boats are moored on a river with trees behind on the bank. The boats have their sails down and are anchored in the ... [tail] River, Trees, Moored, Sails, Anchored, Green, Lush, Healthy, Environment, Daytime.<end_of_utterance>              |
| [`mlx-community/pixtral-12b-bf16`](#model-mlx-community-pixtral-12b-bf16)                                               | `avoid` / `semantic mismatch` | missing terms: Bird, Bushes, Coast, Deben Estuary, Estuary   | Title: Two Boats Moored on River Description: Two sailing boats are moored on a river with dense green foliage on the bank behind them. Keywords: Boat, Boating, Moored, River, Sailing, Trees, Foliage, Green, Mast, Rigging, Nature, Outdoors, Peaceful, Water, Riverbank, Wooden, Buoy                                                    |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `avoid` / `semantic mismatch` | missing terms: Bird, Boating, Bushes, Coast, Deben Estuary   | Title: Sailing boats moored on a river with trees Description: Two sailing boats with tall wooden masts are moored on a river in front of a dense green forest. A small buoy floats in the water in the foregrou ... [tail] forest, foliage, mast, rigging, moored, buoy, water, nature, outdoors, landscape, green, wooden mast             |
| [`mlx-community/paligemma2-10b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-10b-ft-docci-448-bf16)               | `avoid` / `semantic mismatch` | context-ignored; missing-sections                            | - Location: 51.9786° N, 1.4786° E.                                                                                                                                                                                                                                                                                                           |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `avoid` / `model shortcoming` | metadata-borrowing                                           | ## Image Description for Cataloguing **Title:** Two Sailboats Moored on the Deben Estuary **Description:** Two small sailing boats are moored on a calm body of water, likely the Deben Estuary in England. The ... [tail] ation:** Deben Estuary, Woodbury, England, UK, GBR, Europe. **Date Taken:** 2026-07-04 19:10:04 BST.              |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `avoid` / `semantic mismatch` | missing terms: Bird, Boating, Bushes, Coast, Deben Estuary   | Title: Sailing Boat on River Description: A wooden sailing boat with two masts is moored on a calm river, surrounded by dense green trees on the bank. A red buoy floats nearby in the water. Keywords: Boat, Sailing boat, River, Trees, Green foliage, Wooden mast, Red buoy, Calm water, Moored, Outdoors, Nature, Landscape              |
| [`mlx-community/gemma-3-27b-it-qat-8bit`](#model-mlx-community-gemma-3-27b-it-qat-8bit)                                 | `avoid` / `semantic mismatch` | missing terms: Bird, Buoy, Bushes, Deben Estuary, Forest     | Title: Two Sailing Boats Moored on River Description: Two sailing boats are moored on a grey river, with dense green trees lining the bank behind them. The lighting is soft and even, illuminating the boats an ... [tail] ape, Mast, Moored, Nature, Outdoors, River, Sailing, Trees, Water, Grey, Green, Sunlight, Reflection             |
| [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](#model-jqlive-kimi-vl-a3b-thinking-2506-6bit)                                 | `avoid` / `model shortcoming` | metadata-borrowing; reasoning-leak; missing-sections         | ◁think▷Got it, let's analyze the image step by step. First, the title: The image shows two sailing boats on a body of water with trees in the background. The location hint mentions Deben Estuary, Woodbridge, ... [tail] , Buoy, Foliage, Landscape, Nature, Outdoors, Peaceful, Mast, Rigging, Coast, Forest<\|im_assistant\|>            |
| [`mlx-community/GLM-4.6V-Flash-6bit`](#model-mlx-community-glm-46v-flash-6bit)                                          | `caveat` / `context budget`   | context-budget; missing-sections                             | <\|begin_of_box\|>Title: Two sailing boats moored on water with trees behind Description: Two sailing boats are moored on a body of water. Behind the boats, there is a dense area of green trees and foliage. The ... [tail] liage, Estuary, Nature, Outdoors, Peaceful, Mast, Rigging, Buoy, Riverbank, Vegetation<\|end_of_box\|>         |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)            | `avoid` / `model shortcoming` | metadata-borrowing; missing-sections                         | Title Deben Estuary, England, UK Description Two sailing boats are moored on a river with trees behind on the bank. The river is surrounded by lush greenery and has a peaceful atmosphere. The boats are in goo ... [tail] and, Estuary, Foliage, Forest, Landscape, Mast, Moored, Mudflat, Nature, Outdoors, Peaceful, Rigging             |
| [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](#model-mlx-community-kimi-vl-a3b-thinking-8bit)                             | `avoid` / `cutoff degraded`   | metadata-borrowing; cutoff; reasoning-leak                   | ◁think▷Okay, let's tackle this metadata analysis. First, I need to look at the image and extract the metadata based on what's clearly visible. The user provided some hints, but I should only use what's actual ... [tail] boats, boats, masts, buoy, foliage, trees, Deben Estuary, England, Europe, mudflat, nature, outdoors             |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                             | `avoid` / `cutoff degraded`   | repetitive; cutoff; context-ignored                          | - Output only the description text after the label. - Output only the keyword list after the label. - Output only the description text after the label. - Output only the keyword list after the label. - Output only the description text after the label. - Output only the keyword list after the label. - Output only the...             |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `avoid` / `cutoff degraded`   | metadata-borrowing; cutoff                                   | Title: Two sailing boats moored on a river Description: Two sailing boats are moored on a river. The boats have orange masts and red covers. The background features dense green foliage along the riverbank. Ke ... [tail] l, Red sail cover, Orange mast, Sailing rig, Moored, River, Sailing, Mast, Rigging, Foliage, Estuary             |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `avoid` / `cutoff degraded`   | metadata-borrowing; cutoff; degeneration                     | Let's tackle this step by step. First, the title needs to be 5-10 words, concrete and factual. The image has a sailboat in a wooded area by water. Wait, the context mentions Deben Estuary, Woodbridge, England ... [tail] tart with the title. The title needs to be 5-10 words, concrete and factual. Let's see: "Sailboat in             |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `caveat` / `context budget`   | harness:long-context; context-budget; context-ignored        | -                                                                                                                                                                                                                                                                                                                                            |
| [`mlx-community/Molmo-7B-D-0924-8bit`](#model-mlx-community-molmo-7b-d-0924-8bit)                                       | `avoid` / `model shortcoming` | metadata-borrowing; missing-sections                         | Photographer: John Doe. Location: Suffolk, England, UK. Event: Summer sailing day. Subject: Sailing boats in Deben Estuary.                                                                                                                                                                                                                  |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `caveat` / `context budget`   | context-budget; missing-sections                             | <\|begin_of_box\|>Title: Two sailing boats moored on a river with trees behind Description: Two sailing boats are moored on a river, with dense green trees and foliage visible on the bank behind them. The water ... [tail] ging, Black hull, White hull, Red cover, Wooden boat, Natural setting, Water, Outdoors<\|end_of_box\|>         |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `avoid` / `cutoff degraded`   | metadata-borrowing; cutoff; reasoning-leak                   | Here are my reasoning steps: We need to produce three sections: Title, Description, Keywords. Use only details that are clearly visible. The image shows a river or estuary with two sailing boats moored, one l ... [tail] range‑striped masts, while a yellow buoy floats nearby." That's two sentences. Must be 1-2 sentences             |
| [`mlx-community/Molmo-7B-D-0924-bf16`](#model-mlx-community-molmo-7b-d-0924-bf16)                                       | `avoid` / `model shortcoming` | metadata-borrowing; missing-sections                         | Photographer: John Doe. Location: Suffolk, England, UK. Event: Summer day at the coast. Subject: Sailing boats in Deben Estuary. Style: Photographic representationalism realism.                                                                                                                                                            |
| [`meta-llama/Llama-3.2-11B-Vision-Instruct`](#model-meta-llama-llama-32-11b-vision-instruct)                            | `avoid` / `model shortcoming` | metadata-borrowing; missing-sections                         | Deben Estuary, England, UK The image depicts two sailboats moored in a river, with a dense forest in the background. The boats are positioned in the center of the image, with the forest serving as a natural b ... [tail] nd serene atmosphere, with the boats and the surrounding environment blending harmoniously together.             |
| [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16)                             | `caveat` / `context budget`   | harness:long-context; context-budget; context-ignored        | -                                                                                                                                                                                                                                                                                                                                            |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `caveat` / `context budget`   | harness:long-context; context-budget; degeneration           | <\|im_start\|> addCriterion 要求：生成一个与图片内容相关的标题                                                                                                                                                                                                                                                                               |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                           | `avoid` / `semantic mismatch` | missing terms: Bird, Deben Estuary, Mudflat, Peaceful, Deben | Title: Two moored sailing boats on a river with trees Description: Two small sailing boats are anchored on a calm river, with a dense line of trees visible on the opposite bank. The boats have their sails dow ... [tail] g, Buoy, Bushes, Coast, Estuary, Foliage, Forest, Landscape, Mast, Moored, Nature, Outdoors, Rigging             |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `avoid` / `model shortcoming` | metadata-borrowing; reasoning-leak; missing-sections         | ◁think▷◁/think▷Title: Two sailing boats moored on a river with trees in the background Description: Two sailing boats are moored on a calm river with dense green foliage and trees lining the riverbank in the ... [tail] ter, Boat, Mast, Buoy, Nature, Outdoors, Peaceful, England, Europe, Landscape, Coast<\|im_assistant\|>            |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `avoid` / `cutoff degraded`   | harness:long-context; repetitive; cutoff                     | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -...             |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `caveat` / `context budget`   | metadata-borrowing; context-budget                           | Title: Two moored sailboats in a river estuary Description: Two small sailing boats are moored side by side in calm water, with their sails furled and masts standing upright. The background consists of a dens ... [tail] , Forest, Landscape, Mast, Moored, Mudflat, Nature, Outdoors, Peaceful, Rigging, River, Trees, Water             |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `avoid` / `semantic mismatch` | missing terms: Bird, Boating, Bushes, Coast, Deben Estuary   | Title: Two moored sailing boats on a river Description: Two sailing boats with tall wooden masts are moored on a river, backed by a dense bank of green trees and vegetation. A small buoy floats in the foregro ... [tail] ater, wooden hull, red canvas, rigging, nature, outdoors, estuary, bank, vegetation, calm, landscape             |
| [`mlx-community/Qwen3.5-35B-A3B-6bit`](#model-mlx-community-qwen35-35b-a3b-6bit)                                        | `caveat` / `context budget`   | metadata-borrowing; context-budget                           | Title: Two sailing boats moored on a river with trees behind Description: Two sailing boats with tall masts are moored on a river, backed by a dense bank of green trees. A buoy floats in the foreground water. ... [tail] Europe, Foliage, Forest, Landscape, Mast, Moored, Mudflat, Nature, Outdoors, Rigging, River, Sailing             |
| [`mlx-community/Qwen3.5-35B-A3B-bf16`](#model-mlx-community-qwen35-35b-a3b-bf16)                                        | `caveat` / `context budget`   | metadata-borrowing; context-budget                           | Title: Two sailing boats moored on a river with trees behind Description: Two sailing boats with tall masts are moored on a calm river, backed by a dense bank of green trees and foliage. A buoy floats in the ... [tail] Europe, Foliage, Forest, Landscape, Mast, Moored, Mudflat, Nature, Outdoors, River, Rigging, Sailing              |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `recommended` / `clean`       | clean                                                        | Title: Two sailboats moored on a river with a wooded bank Description: Two sailing boats with orange masts are moored on calm water in front of a steep, densely wooded riverbank. A small orange buoy floats in ... [tail] st, Rigging, Buoy, Trees, Foliage, Bank, Water, Wooden, Green, Nature, Outdoors, Landscape, Peaceful             |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                             | `caveat` / `context budget`   | metadata-borrowing; context-budget; missing-sections         | -001.jpg - Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe - Description: Two sailing boats moored on a river with trees behind on the bank                                                                                                                                                                                       |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `recommended` / `clean`       | clean                                                        | Title: Two sailing boats moored on a river Description: Two sailing boats are moored on calm water, with the foreground vessel featuring a black hull, wooden transom, and tall orange masts. A dense backdrop o ... [tail] Trees, Foliage, Nature, Landscape, Mast, Rigging, Hull, Buoy, Outdoors, Green, Wood, Vessel, Forest              |
| [`mlx-community/Qwen3.5-27B-4bit`](#model-mlx-community-qwen35-27b-4bit)                                                | `avoid` / `semantic mismatch` | missing terms: Bird, Boating, Bushes, Coast, Deben Estuary   | Title: Two sailing boats moored on river with forested bank Description: Two sailing boats with wooden masts and furled sails are moored on calm water in front of a densely vegetated riverbank. The foreground ... [tail] ng, Foliage, Water, Buoy, Trees, Nature, Landscape, Outdoors, Hull, Sails, Vegetation, Estuary, Wood             |
| [`mlx-community/Qwen3.5-27B-mxfp8`](#model-mlx-community-qwen35-27b-mxfp8)                                              | `avoid` / `semantic mismatch` | missing terms: Bird, Boating, Bushes, Coast, Deben Estuary   | Title: Two Moored Sailboats on a River with Forested Bank Description: Two sailboats with wooden masts are moored on calm water, positioned in front of a densely vegetated riverbank. The foreground boat featu ... [tail] , Rigging, Hull, Water, Nature, Landscape, Greenery, Buoy, Outdoors, Vessels, Foliage, Estuary, Wood             |
<!-- markdownlint-enable MD034 -->

## All Model Output and Cost Summary

Every model in this run, with its skim-first output or diagnostic beside the main runtime and peak-memory signals.

<!-- markdownlint-disable MD034 -->

| Model                                                                                                                   | Result                        | Output / diagnostic                                                                                                                                                                                                                      | Gen tok   | Total   | Gen TPS   | Peak GB   | Quality signal                                               |
|-------------------------------------------------------------------------------------------------------------------------|-------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|---------|-----------|-----------|--------------------------------------------------------------|
| [`mlx-community/gemma-4-31b-bf16`](#model-mlx-community-gemma-4-31b-bf16)                                               | `avoid` / `semantic mismatch` | [mlx-vlm; model-error] Error: Model Error - Model runtime error during generation for mlx-community/gemma-4-31b-bf16: [METAL] Command buffer execution ... [tail] y (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory).              | -         | 224.31s | -         | -         | mlx-vlm; model-error                                         |
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `avoid` / `model shortcoming` | Title: Deben Estuary Boats on a River Description: - Two sailboats moored on a calm river with dense green foliage in the background. - The boa ... [tail] , Moored, Mudflat, Coast, Deben Estuary, England, Estuary, Europe             | 121       | 1.36s   | 508       | 1.4       | metadata-borrowing                                           |
| [`mlx-community/LFM2-VL-1.6B-8bit`](#model-mlx-community-lfm2-vl-16b-8bit)                                              | `avoid` / `model shortcoming` | Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe Description: Two sailing boats moored on a river with trees behind on the bank Keywo ... [tail] , Europe, Foliage, Landscape, Nature, Outdoors, Peaceful, Rigging             | 69        | 1.72s   | 325       | 3.0       | metadata-borrowing                                           |
| [`qnguyen3/nanoLLaVA`](#model-qnguyen3-nanollava)                                                                       | `avoid` / `model shortcoming` | Title: 5-10 boats moored on a river with trees behind Description: Two boats tied to a bank with trees behind them.                                                                                                                      | 31        | 1.88s   | 112       | 4.6       | metadata-borrowing; missing-sections                         |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                            | `avoid` / `model shortcoming` | Title: 5-10 words, concrete and factual, limited to clearly visible content. Description: Two sailing boats moored on a river with trees behind ... [tail] cape, Mast, Moored, Mudflat, Nature, Outdoors, Peaceful, Rigging.             | 93        | 1.89s   | 335       | 2.7       | metadata-borrowing                                           |
| [`mlx-community/paligemma2-3b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-3b-ft-docci-448-bf16)                 | `avoid` / `harness`           | - Daytime.                                                                                                                                                                                                                               | 5         | 3.06s   | 24.1      | 11        | harness:prompt-template; context-ignored                     |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `avoid` / `harness`           | &lt;think&gt; &lt;/think&gt; Title: Sailing boats moored near dense green foliage on water Description: The image shows two sailing boats floating on water ... [tail] ored, calm, green, nature, river, boats, foliage, peace, outdoors | 77        | 2.65s   | 225       | 4.0       | harness:stop-token; reasoning-leak; formatting               |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `avoid` / `model shortcoming` | Title: - Deben Estuary, Woodbridge, England, UK, GBR, Europe, River, Boating, Sailing, Boats, Mast, Rigging, Buoy, Trees, Landscape, Nature, Es ... [tail] st, Foliage, Outdoors, Peaceful, Woodbridge, England, GBR, Europe             | 171       | 2.40s   | 188       | 4.1       | metadata-borrowing                                           |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `avoid` / `model shortcoming` | Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe Description: Two sailing boats moored on a river with trees behind on the bank. The ... [tail] the water. The overall mood of the image is peaceful and serene.               | 191       | 2.73s   | 348       | 2.2       | metadata-borrowing; missing-sections                         |
| [`HuggingFaceTB/SmolVLM-Instruct`](#model-huggingfacetb-smolvlm-instruct)                                               | `avoid` / `model shortcoming` | Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe Description: Two sailing boats moored on a river with trees behind on the bank Keywo ... [tail] , Landscape, Moored, Mudflat, Nature, Outdoors, Peaceful, Rigging             | 90        | 2.83s   | 129       | 5.6       | metadata-borrowing                                           |
| [`mlx-community/SmolVLM-Instruct-bf16`](#model-mlx-community-smolvlm-instruct-bf16)                                     | `avoid` / `model shortcoming` | Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe Description: Two sailing boats moored on a river with trees behind on the bank Keywo ... [tail] , Landscape, Moored, Mudflat, Nature, Outdoors, Peaceful, Rigging             | 90        | 2.84s   | 127       | 5.6       | metadata-borrowing                                           |
| [`mlx-community/paligemma2-10b-ft-docci-448-6bit`](#model-mlx-community-paligemma2-10b-ft-docci-448-6bit)               | `avoid` / `semantic mismatch` | - Location: 52° 0' 0" N, 1° 0' 0" E.                                                                                                                                                                                                     | 27        | 4.46s   | 32.0      | 12        | context-ignored; missing-sections                            |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `avoid` / `semantic mismatch` | Title: Two small sailing boats moored on a river Description: Two small wooden sailing boats with tall masts are moored on the water against a ... [tail] rest, landscape, nature, outdoors, greenery, vertical composition              | 88        | 5.31s   | 60.4      | 18        | missing terms: Bird, Boating, Buoy, Bushes, Coast            |
| [`microsoft/Phi-3.5-vision-instruct`](#model-microsoft-phi-35-vision-instruct)                                          | `recommended` / `clean`       | Title: Sailing Boats at Deben Estuary Description: Two sailing boats are moored on a river, surrounded by lush greenery and trees. The boats ar ... [tail] River, Trees, Greenery, Mast, Rigging, Mudflat, Nature, Outdoors              | 101       | 3.91s   | 57.7      | 9.5       | clean                                                        |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `avoid` / `model shortcoming` | Title: - Two sailing boats moored on the Deben Estuary Description: - Two wooden sailboats with tall masts are moored on a river next to a dens ... [tail] est, landscape, mast, moored, nature, outdoors, peaceful, rigging             | 89        | 6.30s   | 58.9      | 29        | metadata-borrowing                                           |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `avoid` / `semantic mismatch` | Title: Traditional sailboat moored by riverbank with dense foliage Description: A classic wooden sailboat with tall masts and red sails is moor ... [tail] es, nature scene, waterfront, maritime setting, coastal landscape             | 107       | 5.20s   | 168       | 7.8       | missing terms: Bird, Boating, Buoy, Bushes, Coast            |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `recommended` / `clean`       | Title: Sailing Boats at Deben Estuary Description: Two sailing boats are moored on a river, surrounded by lush greenery and trees. The boats ar ... [tail] River, Trees, Greenery, Mast, Rigging, Mudflat, Nature, Outdoors              | 101       | 4.05s   | 55.5      | 9.5       | clean                                                        |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `avoid` / `semantic mismatch` | Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe Description: Two sailing boats moored on a river with trees behind on the bank Keywo ... [tail] adata: Taken on 2026-07-04 19:10:04 BST (at 19:10:04 local time).             | 248       | 3.91s   | 134       | 5.5       | context-echo; instruction-echo; metadata-borrowing           |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                            | `avoid` / `model shortcoming` | The image shows a boat on a river.                                                                                                                                                                                                       | 11        | 4.26s   | 68.1      | 9.7       | context-ignored; missing-sections                            |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `avoid` / `model shortcoming` | Title: - Two sailing boats moored on the Deben Estuary Description: - Two small sailing boats are moored on a river in front of a dense green f ... [tail] , Foliage, Nature, Outdoors, Landscape, Peaceful, England, Europe             | 78        | 7.01s   | 33.0      | 28        | metadata-borrowing                                           |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `avoid` / `semantic mismatch` | Title: Two sailing boats moored on river Description: Two sailing boats with orange masts are moored on a calm river, surrounded by dense green ... [tail] d, Foliage, Trees, Mast, Nature, Outdoors, Peaceful, Buoy, Bushes             | 71        | 5.56s   | 34.8      | 18        | missing terms: Bird, Boating, Coast, Deben Estuary, Estuary  |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `avoid` / `semantic mismatch` | Title: Two sailing boats moored on a waterway Description: Two sailing boats are moored on calm water, with dense green foliage lining the bank ... [tail] e, Mast, Moored, Outdoors, River, Sailing, Trees, Water, Waterway             | 78        | 6.91s   | 31.5      | 18        | missing terms: Bird, Bushes, Coast, Deben Estuary, Estuary   |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `avoid` / `semantic mismatch` | Title: Sailing Boat on Calm Water with Forest Background Description: A single wooden sailboat with two tall masts and red sails is moored on c ... [tail] alm water, dense forest, green foliage, shoreline, anchored, buoy             | 95        | 5.99s   | 65.0      | 13        | missing terms: Bird, Boating, Bushes, Coast, Deben Estuary   |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `caveat` / `context budget`   | Title: Two Boats Moored on River Description: Two sailing boats are moored on a river with dense green foliage on the bank behind them. Keyword ... [tail] ing, Nature, Outdoors, Peaceful, Water, Riverbank, Forest, Europe             | 78        | 6.52s   | 39.0      | 16        | metadata-borrowing; context-budget                           |
| [`mlx-community/InternVL3-14B-8bit`](#model-mlx-community-internvl3-14b-8bit)                                           | `avoid` / `semantic mismatch` | Title: Sailing Boats on River Description: Two sailing boats are moored on a calm river with dense green foliage on the bank behind them. Keywo ... [tail] Moored, Nature, Outdoors, Peaceful, River, Rigging, Trees, Water              | 72        | 6.63s   | 31.4      | 18        | missing terms: Bird, Bushes, Coast, Deben Estuary, Estuary   |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `caveat` / `context budget`   | **Title:** Sailing Boat Moored on Deben Estuary **Description:** A single wooden sailboat with two masts is anchored near a buoy on calm waters ... [tail] ging, Moored, Deben Estuary, Natural setting, Woodbridge, England             | 120       | 6.56s   | 61.8      | 13        | metadata-borrowing; context-budget                           |
| [`mlx-community/gemma-3n-E2B-4bit`](#model-mlx-community-gemma-3n-e2b-4bit)                                             | `avoid` / `cutoff degraded`   | - 19:10:18:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00...             | 500       | 6.71s   | 124       | 6.0       | metadata-borrowing; cutoff; degeneration                     |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `avoid` / `model shortcoming` | Title: Sailing Boats on the Deben Estuary, Woodbridge, England, UK. Description: Two sailing boats are moored on a river with trees behind on t ... [tail] ed, Green, Lush, Healthy, Environment, Daytime.<end_of_utterance>             | 126       | 8.03s   | 32.1      | 19        | metadata-borrowing; formatting                               |
| [`mlx-community/pixtral-12b-bf16`](#model-mlx-community-pixtral-12b-bf16)                                               | `avoid` / `semantic mismatch` | Title: Two Boats Moored on River Description: Two sailing boats are moored on a river with dense green foliage on the bank behind them. Keyword ... [tail] gging, Nature, Outdoors, Peaceful, Water, Riverbank, Wooden, Buoy             | 80        | 9.17s   | 20.1      | 28        | missing terms: Bird, Bushes, Coast, Deben Estuary, Estuary   |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `avoid` / `semantic mismatch` | Title: Sailing boats moored on a river with trees Description: Two sailing boats with tall wooden masts are moored on a river in front of a den ... [tail] red, buoy, water, nature, outdoors, landscape, green, wooden mast             | 81        | 9.51s   | 19.6      | 22        | missing terms: Bird, Boating, Bushes, Coast, Deben Estuary   |
| [`mlx-community/paligemma2-10b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-10b-ft-docci-448-bf16)               | `avoid` / `semantic mismatch` | - Location: 51.9786° N, 1.4786° E.                                                                                                                                                                                                       | 25        | 9.34s   | 5.49      | 26        | context-ignored; missing-sections                            |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `avoid` / `model shortcoming` | ## Image Description for Cataloguing **Title:** Two Sailboats Moored on the Deben Estuary **Description:** Two small sailing boats are moored o ... [tail] ngland, UK, GBR, Europe. **Date Taken:** 2026-07-04 19:10:04 BST.             | 272       | 9.25s   | 48.5      | 17        | metadata-borrowing                                           |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `avoid` / `semantic mismatch` | Title: Sailing Boat on River Description: A wooden sailing boat with two masts is moored on a calm river, surrounded by dense green trees on th ... [tail] n mast, Red buoy, Calm water, Moored, Outdoors, Nature, Landscape             | 82        | 9.59s   | 29.9      | 23        | missing terms: Bird, Boating, Bushes, Coast, Deben Estuary   |
| [`mlx-community/gemma-3-27b-it-qat-8bit`](#model-mlx-community-gemma-3-27b-it-qat-8bit)                                 | `avoid` / `semantic mismatch` | Title: Two Sailing Boats Moored on River Description: Two sailing boats are moored on a grey river, with dense green trees lining the bank behi ... [tail] , River, Sailing, Trees, Water, Grey, Green, Sunlight, Reflection             | 91        | 10.75s  | 17.7      | 32        | missing terms: Bird, Buoy, Bushes, Deben Estuary, Forest     |
| [`jqlive/Kimi-VL-A3B-Thinking-2506-6bit`](#model-jqlive-kimi-vl-a3b-thinking-2506-6bit)                                 | `avoid` / `model shortcoming` | ◁think▷Got it, let's analyze the image step by step. First, the title: The image shows two sailing boats on a body of water with trees in the b ... [tail] Outdoors, Peaceful, Mast, Rigging, Coast, Forest<\|im_assistant\|>            | 393       | 9.59s   | 70.0      | 18        | metadata-borrowing; reasoning-leak; missing-sections         |
| [`mlx-community/GLM-4.6V-Flash-6bit`](#model-mlx-community-glm-46v-flash-6bit)                                          | `caveat` / `context budget`   | <\|begin_of_box\|>Title: Two sailing boats moored on water with trees behind Description: Two sailing boats are moored on a body of water. Behind ... [tail] eaceful, Mast, Rigging, Buoy, Riverbank, Vegetation<\|end_of_box\|>         | 92        | 9.49s   | 58.4      | 11        | context-budget; missing-sections                             |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)            | `avoid` / `model shortcoming` | Title Deben Estuary, England, UK Description Two sailing boats are moored on a river with trees behind on the bank. The river is surrounded by ... [tail] scape, Mast, Moored, Mudflat, Nature, Outdoors, Peaceful, Rigging              | 129       | 10.41s  | 20.5      | 15        | metadata-borrowing; missing-sections                         |
| [`mlx-community/Kimi-VL-A3B-Thinking-8bit`](#model-mlx-community-kimi-vl-a3b-thinking-8bit)                             | `avoid` / `cutoff degraded`   | ◁think▷Okay, let's tackle this metadata analysis. First, I need to look at the image and extract the metadata based on what's clearly visible. ... [tail] trees, Deben Estuary, England, Europe, mudflat, nature, outdoors               | 500       | 11.99s  | 63.9      | 22        | metadata-borrowing; cutoff; reasoning-leak                   |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                             | `avoid` / `cutoff degraded`   | - Output only the description text after the label. - Output only the keyword list after the label. - Output only the description text after the label. - Output only the keyword list after the label. - Output only the...             | 500       | 13.35s  | 50.6      | 4.6       | repetitive; cutoff; context-ignored                          |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `avoid` / `cutoff degraded`   | Title: Two sailing boats moored on a river Description: Two sailing boats are moored on a river. The boats have orange masts and red covers. Th ... [tail] ling rig, Moored, River, Sailing, Mast, Rigging, Foliage, Estuary             | 500       | 13.95s  | 79.9      | 8.4       | metadata-borrowing; cutoff                                   |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `avoid` / `cutoff degraded`   | Let's tackle this step by step. First, the title needs to be 5-10 words, concrete and factual. The image has a sailboat in a wooded area by wat ... [tail] s to be 5-10 words, concrete and factual. Let's see: "Sailboat in             | 500       | 18.55s  | 45.8      | 60        | metadata-borrowing; cutoff; degeneration                     |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `caveat` / `context budget`   | -                                                                                                                                                                                                                                        | 2         | 15.46s  | 181       | 8.6       | harness:long-context; context-budget; context-ignored        |
| [`mlx-community/Molmo-7B-D-0924-8bit`](#model-mlx-community-molmo-7b-d-0924-8bit)                                       | `avoid` / `model shortcoming` | Photographer: John Doe. Location: Suffolk, England, UK. Event: Summer sailing day. Subject: Sailing boats in Deben Estuary.                                                                                                              | 31        | 16.91s  | 46.7      | 41        | metadata-borrowing; missing-sections                         |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `caveat` / `context budget`   | <\|begin_of_box\|>Title: Two sailing boats moored on a river with trees behind Description: Two sailing boats are moored on a river, with dense g ... [tail] over, Wooden boat, Natural setting, Water, Outdoors<\|end_of_box\|>         | 103       | 20.90s  | 40.7      | 78        | context-budget; missing-sections                             |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `avoid` / `cutoff degraded`   | Here are my reasoning steps: We need to produce three sections: Title, Description, Keywords. Use only details that are clearly visible. The im ... [tail] buoy floats nearby." That's two sentences. Must be 1-2 sentences              | 500       | 16.84s  | 41.5      | 15        | metadata-borrowing; cutoff; reasoning-leak                   |
| [`mlx-community/Molmo-7B-D-0924-bf16`](#model-mlx-community-molmo-7b-d-0924-bf16)                                       | `avoid` / `model shortcoming` | Photographer: John Doe. Location: Suffolk, England, UK. Event: Summer day at the coast. Subject: Sailing boats in Deben Estuary. Style: Photographic representationalism realism.                                                        | 42        | 18.56s  | 27.0      | 48        | metadata-borrowing; missing-sections                         |
| [`meta-llama/Llama-3.2-11B-Vision-Instruct`](#model-meta-llama-llama-32-11b-vision-instruct)                            | `avoid` / `model shortcoming` | Deben Estuary, England, UK The image depicts two sailboats moored in a river, with a dense forest in the background. The boats are positioned i ... [tail] s and the surrounding environment blending harmoniously together.             | 73        | 19.20s  | 5.13      | 25        | metadata-borrowing; missing-sections                         |
| [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16)                             | `caveat` / `context budget`   | -                                                                                                                                                                                                                                        | 2         | 18.83s  | 180       | 8.6       | harness:long-context; context-budget; context-ignored        |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `caveat` / `context budget`   | <\|im_start\|> addCriterion 要求：生成一个与图片内容相关的标题                                                                                                                                                                           | 14        | 19.80s  | 60.6      | 13        | harness:long-context; context-budget; degeneration           |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                           | `avoid` / `semantic mismatch` | Title: Two moored sailing boats on a river with trees Description: Two small sailing boats are anchored on a calm river, with a dense line of t ... [tail] liage, Forest, Landscape, Mast, Moored, Nature, Outdoors, Rigging             | 94        | 23.31s  | 5.94      | 27        | missing terms: Bird, Deben Estuary, Mudflat, Peaceful, Deben |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `avoid` / `model shortcoming` | ◁think▷◁/think▷Title: Two sailing boats moored on a river with trees in the background Description: Two sailing boats are moored on a calm rive ... [tail] oors, Peaceful, England, Europe, Landscape, Coast<\|im_assistant\|>           | 96        | 26.96s  | 4.65      | 39        | metadata-borrowing; reasoning-leak; missing-sections         |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `avoid` / `cutoff degraded`   | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -...             | 500       | 26.64s  | 88.4      | 8.6       | harness:long-context; repetitive; cutoff                     |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `caveat` / `context budget`   | Title: Two moored sailboats in a river estuary Description: Two small sailing boats are moored side by side in calm water, with their sails fur ... [tail] Mudflat, Nature, Outdoors, Peaceful, Rigging, River, Trees, Water             | 126       | 62.13s  | 92.4      | 11        | metadata-borrowing; context-budget                           |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `avoid` / `semantic mismatch` | Title: Two moored sailing boats on a river Description: Two sailing boats with tall wooden masts are moored on a river, backed by a dense bank ... [tail] ing, nature, outdoors, estuary, bank, vegetation, calm, landscape              | 106       | 66.69s  | 104       | 26        | missing terms: Bird, Boating, Bushes, Coast, Deben Estuary   |
| [`mlx-community/Qwen3.5-35B-A3B-6bit`](#model-mlx-community-qwen35-35b-a3b-6bit)                                        | `caveat` / `context budget`   | Title: Two sailing boats moored on a river with trees behind Description: Two sailing boats with tall masts are moored on a river, backed by a ... [tail] Mast, Moored, Mudflat, Nature, Outdoors, Rigging, River, Sailing               | 99        | 67.39s  | 92.2      | 35        | metadata-borrowing; context-budget                           |
| [`mlx-community/Qwen3.5-35B-A3B-bf16`](#model-mlx-community-qwen35-35b-a3b-bf16)                                        | `caveat` / `context budget`   | Title: Two sailing boats moored on a river with trees behind Description: Two sailing boats with tall masts are moored on a calm river, backed ... [tail] Mast, Moored, Mudflat, Nature, Outdoors, River, Rigging, Sailing               | 102       | 74.82s  | 64.6      | 76        | metadata-borrowing; context-budget                           |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `recommended` / `clean`       | Title: Two sailboats moored on a river with a wooded bank Description: Two sailing boats with orange masts are moored on calm water in front of ... [tail] Bank, Water, Wooden, Green, Nature, Outdoors, Landscape, Peaceful             | 105       | 75.10s  | 60.9      | 76        | clean                                                        |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                             | `caveat` / `context budget`   | -001.jpg - Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe - Description: Two sailing boats moored on a river with trees behind on the bank                                                                                   | 44        | 75.30s  | 221       | 5.1       | metadata-borrowing; context-budget; missing-sections         |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `recommended` / `clean`       | Title: Two sailing boats moored on a river Description: Two sailing boats are moored on calm water, with the foreground vessel featuring a blac ... [tail] Mast, Rigging, Hull, Buoy, Outdoors, Green, Wood, Vessel, Forest              | 114       | 88.30s  | 17.6      | 38        | clean                                                        |
| [`mlx-community/Qwen3.5-27B-4bit`](#model-mlx-community-qwen35-27b-4bit)                                                | `avoid` / `semantic mismatch` | Title: Two sailing boats moored on river with forested bank Description: Two sailing boats with wooden masts and furled sails are moored on cal ... [tail] ture, Landscape, Outdoors, Hull, Sails, Vegetation, Estuary, Wood             | 130       | 90.29s  | 29.6      | 26        | missing terms: Bird, Boating, Bushes, Coast, Deben Estuary   |
| [`mlx-community/Qwen3.5-27B-mxfp8`](#model-mlx-community-qwen35-27b-mxfp8)                                              | `avoid` / `semantic mismatch` | Title: Two Moored Sailboats on a River with Forested Bank Description: Two sailboats with wooden masts are moored on calm water, positioned in ... [tail] dscape, Greenery, Buoy, Outdoors, Vessels, Foliage, Estuary, Wood              | 135       | 97.00s  | 16.7      | 38        | missing terms: Bird, Boating, Bushes, Coast, Deben Estuary   |
<!-- markdownlint-enable MD034 -->

## Quick Navigation

- _Successful outputs:_ [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16),
  [`mlx-community/LFM2-VL-1.6B-8bit`](#model-mlx-community-lfm2-vl-16b-8bit),
  [`qnguyen3/nanoLLaVA`](#model-qnguyen3-nanollava),
  [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit),
  [`mlx-community/paligemma2-3b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-3b-ft-docci-448-bf16),
  [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit),
  [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16),
  [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16),
  +52 more
- _Flagged outputs:_ [`mlx-community/gemma-4-31b-bf16`](#model-mlx-community-gemma-4-31b-bf16),
  [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16),
  [`mlx-community/LFM2-VL-1.6B-8bit`](#model-mlx-community-lfm2-vl-16b-8bit),
  [`qnguyen3/nanoLLaVA`](#model-qnguyen3-nanollava),
  [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit),
  [`mlx-community/paligemma2-3b-ft-docci-448-bf16`](#model-mlx-community-paligemma2-3b-ft-docci-448-bf16),
  [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit),
  [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16),
  +37 more
- _Failed outputs:_ [`mlx-community/gemma-4-31b-bf16`](#model-mlx-community-gemma-4-31b-bf16)

## Model Gallery

Full generated output by model:

<!-- markdownlint-disable MD033 MD034 -->

<a id="model-mlx-community-gemma-4-31b-bf16"></a>

### ❌ mlx-community/gemma-4-31b-bf16

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Status:_ Failed (Model Error)
- _Owner:_ likely owner `model`; reported package `mlx-vlm`; failure stage
  `Model Error`; diagnostic code `MLX_VLM_DECODE_MODEL`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Error summary:_ see error details below
- _Key signals:_ keywords=57; context echo=100%; nonvisual metadata reused;
  reasoning leak
- _Failure context:_ type `ValueError`; phase `decode`; code
  `MLX_VLM_DECODE_MODEL`; package `mlx-vlm`
- _Tokens:_ prompt n/a; estimated text 435 tok; estimated non-text n/a;
  generated n/a; requested max 500 tok; stop reason exception

_Error details:_

> Model runtime error during generation for mlx-community/gemma-4-31b-bf16:
> [METAL] Command buffer execution failed: Insufficient Memory
> (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory).

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/common.py", line 141, in wired_limit
    yield
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/dispatch.py", line 1075, in stream_generate
    detokenizer.add_token(token, skip_special_token_ids=skip_special_token_ids)
    ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/tokenizer_utils.py", line 171, in add_token
    if self.is_byte_token[token]:
       ~~~~~~~~~~~~~~~~~~^^^^^^^
IndexError: list index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 20210, in _run_generation_with_retry_workaround
    yield

  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 20586, in _generate_once
    strict_generate = cast("StrictGenerateCallable", generate)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<23 lines>...
            presence_penalty=params.presence_penalty,

  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/dispatch.py", line 1242, in generate
    ):

        if response.is_draft:
        ^^^^^^^^^^^^^^^^^^^^^
            diffusion_output.handle_draft(response)

  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/dispatch.py", line 1009, in stream_generate
    with wired_limit(model, [generation_stream]):
         ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/contextlib.py", line 162, in __exit__
    self.gen.throw(value)
    ~~~~~~~~~~~~~~^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/common.py", line 145, in wired_limit
    mx.synchronize(stream)
    ~~~~~~~~~~~~~~^^^^^^^^
RuntimeError: [METAL] Command buffer execution failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory).

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 20770, in process_image_with_model
    )

    ...<3 lines>...
            params.model_identifier,

  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 20617, in _run_model_generation
    kv_quant_scheme=params.kv_quant_scheme,
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
    max_tokens=params.max_tokens,

  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 20244, in _run_generation_with_retry_workaround
    msg = f"Model generation failed for {params.model_identifier}: {retry_known_err}"
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: Model runtime error during generation for mlx-community/gemma-4-31b-bf16: [METAL] Command buffer execution failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory).
```

</details>

---

<a id="model-liquidai-lfm25-vl-450m-mlx-bf16"></a>

### ✅ LiquidAI/LFM2.5-VL-450M-MLX-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Buoy, Bushes, Rigging, Woodbridge;
  keywords=19; nonvisual metadata reused
- _Score:_ ✅ B (67/100); Δ+18; None identified
- _Review focus:_ watchlist (metadata borrowing)
- _Timing:_ Load 0.34s; Gen 0.67s; Total 1.36s
- _Throughput:_ Prompt 19,915 TPS (585 tok); Gen 508 TPS (121 tok)
- _Tokens:_ prompt 585 tok; estimated text 435 tok; estimated non-text 150
  tok; generated 121 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Deben Estuary Boats on a River
>
> &#8203;Description:
> &#45; Two sailboats moored on a calm river with dense green foliage in the
> background.
> &#45; The boats are positioned in the foreground, with the water reflecting the
> greenery.
> &#45; The scene is serene, with no visible activity or people.
>
> &#8203;Keywords:
> &#45; Boat, Boating, River, Estuary, Deben Estuary, Nature, Outdoors, Peaceful,
> Foliage, Forest, Landscape, Mast, Moored, Mudflat, Coast, Deben Estuary,
> England, Estuary, Europe
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Description sentence violation (3; expected 1-2)
- Keyword count violation (19; expected 10-18)
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-lfm2-vl-16b-8bit"></a>

### ✅ mlx-community/LFM2-VL-1.6B-8bit

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Buoy, Bushes, Coast; nonvisual
  metadata reused
- _Score:_ 🟡 C (51/100); Δ+2; Lacks visual description of image
- _Review focus:_ watchlist (metadata borrowing)
- _Timing:_ Load 0.55s; Gen 0.79s; Total 1.72s
- _Throughput:_ Prompt 7,311 TPS (775 tok); Gen 325 TPS (69 tok)
- _Tokens:_ prompt 775 tok; estimated text 435 tok; estimated non-text 340
  tok; generated 69 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe
> Description: Two sailing boats moored on a river with trees behind on the
> bank
> Keywords: Boat, Deben Estuary, England, Estuary, Europe, Foliage, Landscape,
> Nature, Outdoors, Peaceful, Rigging
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Nonvisual metadata borrowing

---

<a id="model-qnguyen3-nanollava"></a>

### ✅ qnguyen3/nanoLLaVA

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; missing terms: Bird, Boating,
  Buoy, Bushes, Coast; nonvisual metadata reused
- _Score:_ 🟡 C (59/100); Δ+10; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (metadata borrowing, missing sections, trusted
  hint degraded)
- _Timing:_ Load 0.57s; Gen 0.95s; Total 1.88s
- _Throughput:_ Prompt 4,858 TPS (512 tok); Gen 112 TPS (31 tok)
- _Tokens:_ prompt 512 tok; estimated text 435 tok; estimated non-text 77 tok;
  generated 31 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: 5-10 boats moored on a river with trees behind
> Description: Two boats tied to a bank with trees behind them.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (keywords)
- Nonvisual metadata borrowing
- Degrades trusted hints

---

<a id="model-mlx-community-nanollava-15-4bit"></a>

### ✅ mlx-community/nanoLLaVA-1.5-4bit

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Woodbridge, GBR; keywords=20; nonvisual
  metadata reused
- _Score:_ 🟠 D (46/100); Δ-3; Lacks visual description of image
- _Review focus:_ watchlist (metadata borrowing)
- _Timing:_ Load 0.56s; Gen 0.98s; Total 1.89s
- _Throughput:_ Prompt 3,794 TPS (512 tok); Gen 335 TPS (93 tok)
- _Tokens:_ prompt 512 tok; estimated text 435 tok; estimated non-text 77 tok;
  generated 93 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: 5-10 words, concrete and factual, limited to clearly visible content.
> Description: Two sailing boats moored on a river with trees behind on the
> bank.
> Keywords: Bird, Boat, Boating, Buoy, Bushes, Coast, Deben Estuary, England,
> Estuary, Europe, Foliage, Forest, Landscape, Mast, Moored, Mudflat, Nature,
> Outdoors, Peaceful, Rigging.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Title length violation (11 words; expected 5-10)
- Keyword count violation (20; expected 10-18)
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-paligemma2-3b-ft-docci-448-bf16"></a>

### ✅ mlx-community/paligemma2-3b-ft-docci-448-bf16

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `model-config`; harness signal `prompt_template`
- _Next step:_ Inspect model repo config, chat template, and EOS settings.
- _Key signals:_ Output appears truncated to about 5 tokens.; missing terms:
  Bird, Boat, Boating, Buoy, Bushes
- _Score:_ ❌ F (6/100); Δ-43; Output too short to be useful
- _Review focus:_ watchlist (context ignored, harness)
- _Timing:_ Load 1.47s; Gen 1.22s; Total 3.06s
- _Throughput:_ Prompt 3,412 TPS (1,530 tok); Gen 24.1 TPS (5 tok)
- _Tokens:_ prompt 1530 tok; estimated text 435 tok; estimated non-text 1095
  tok; generated 5 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#45; Daytime.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:prompt_template
- Context ignored (missing: Bird, Boat, Boating, Buoy, Bushes)
- Ignores trusted hints
- output:truncated(5tok)

---

<a id="model-mlx-community-minicpm-v-46-8bit"></a>

### ✅ mlx-community/MiniCPM-V-4.6-8bit

- _Recommendation:_ avoid for now; review verdict: harness
- _Owner:_ likely owner `mlx-vlm`; harness signal `stop_token`
- _Next step:_ Inspect EOS/stop-token stripping; control tokens are leaking
  into user-facing text.
- _Key signals:_ Special control token &lt;/think&gt; appeared in generated
  text.; missing terms: Bird, Boating, Buoy, Bushes, Coast; reasoning leak;
  text-sanity=gibberish(token_noise)
- _Score:_ 🏆 A (82/100); Δ+33; None identified
- _Review focus:_ watchlist (harness, reasoning leak, text sanity, trusted
  hint degraded)
- _Timing:_ Load 0.99s; Gen 1.27s; Total 2.65s
- _Throughput:_ Prompt 3,996 TPS (1,127 tok); Gen 225 TPS (77 tok)
- _Tokens:_ prompt 1127 tok; estimated text 435 tok; estimated non-text 692
  tok; generated 77 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &lt;think&gt;
>
> &lt;/think&gt;
>
> &#8203;Title:
> Sailing boats moored near dense green foliage on water
>
> &#8203;Description:
> The image shows two sailing boats floating on water with lush trees in the
> background. The scene appears calm with natural lighting.
>
> &#8203;Keywords:
> sailing boats, water, trees, moored, calm, green, nature, river, boats,
> foliage, peace, outdoors
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:stop_token
- Text sanity issue (gibberish(token_noise))
- Reasoning leak (&lt;think&gt;)
- Degrades trusted hints
- Unknown tags: &lt;think&gt;
- token_leak:&lt;/think&gt;

---

<a id="model-mlx-community-lfm25-vl-16b-bf16"></a>

### ✅ mlx-community/LFM2.5-VL-1.6B-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Bushes, Forest, Mudflat; keywords=19;
  nonvisual metadata reused
- _Score:_ 🟠 D (44/100); Δ-6; Lacks visual description of image
- _Review focus:_ watchlist (metadata borrowing)
- _Timing:_ Load 0.57s; Gen 1.46s; Total 2.40s
- _Throughput:_ Prompt 7,420 TPS (775 tok); Gen 188 TPS (171 tok)
- _Tokens:_ prompt 775 tok; estimated text 435 tok; estimated non-text 340
  tok; generated 171 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> &#45; Deben Estuary, Woodbridge, England, UK, GBR, Europe, River, Boating,
> Sailing, Boats, Mast, Rigging, Buoy, Trees, Landscape, Nature, Estuary,
> Coast, Foliage, Outdoors, Peaceful
>
> &#8203;Description:
> &#45; Two sailing boats moored on a river with trees behind on the bank. The
> boats have tall masts and are anchored in calm water. The surrounding area
> features dense foliage and a forested bank. The scene is tranquil and
> natural, with no visible human activity.
>
> &#8203;Keywords:
> &#45; Boat, Sailing, Boats, Mast, Rigging, Buoy, Trees, Landscape, Nature,
> Estuary, River, Coast, Foliage, Outdoors, Peaceful, Woodbridge, England,
> GBR, Europe
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Title length violation (22 words; expected 5-10)
- Description sentence violation (4; expected 1-2)
- Keyword count violation (19; expected 10-18)
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-fastvlm-05b-bf16"></a>

### ✅ mlx-community/FastVLM-0.5B-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: keywords; missing terms: Bird, Boating,
  Buoy, Bushes, Coast; nonvisual metadata reused
- _Score:_ ✅ B (78/100); Δ+29; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (metadata borrowing, missing sections)
- _Timing:_ Load 0.63s; Gen 1.67s; Total 2.73s
- _Throughput:_ Prompt 4,653 TPS (516 tok); Gen 348 TPS (191 tok)
- _Tokens:_ prompt 516 tok; estimated text 435 tok; estimated non-text 81 tok;
  generated 191 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe
>
> Description: Two sailing boats moored on a river with trees behind on the
> bank. The boats are moored close to each other, with one boat slightly ahead
> of the other. The boats have their sails down, and the masts are tall and
> slender. The water is calm, and there are no other boats or people visible
> in the image. The trees in the background are lush and green, indicating a
> healthy forest. The sky is overcast, suggesting it might be a cool or cloudy
> day. The image is taken from a distance, and there are no people or other
> objects visible in the foreground. The boats are the main focus of the
> image, and they are well-lit by natural light. The colors in the image are
> mostly green and brown, with some blue from the water. The overall mood of
> the image is peaceful and serene.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (keywords)
- Description sentence violation (10; expected 1-2)
- Nonvisual metadata borrowing

---

<a id="model-huggingfacetb-smolvlm-instruct"></a>

### ✅ HuggingFaceTB/SmolVLM-Instruct

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Mast; keywords=19; nonvisual metadata reused
- _Score:_ 🟠 D (46/100); Δ-3; Lacks visual description of image
- _Review focus:_ watchlist (metadata borrowing)
- _Timing:_ Load 0.72s; Gen 1.75s; Total 2.83s
- _Throughput:_ Prompt 4,208 TPS (1,722 tok); Gen 129 TPS (90 tok)
- _Tokens:_ prompt 1722 tok; estimated text 435 tok; estimated non-text 1287
  tok; generated 90 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe
> Description: Two sailing boats moored on a river with trees behind on the
> bank
> Keywords: Bird, Boat, Boating, Buoy, Bushes, Coast, Deben Estuary, England,
> Estuary, Europe, Foliage, Forest, Landscape, Moored, Mudflat, Nature,
> Outdoors, Peaceful, Rigging
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Keyword count violation (19; expected 10-18)
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-smolvlm-instruct-bf16"></a>

### ✅ mlx-community/SmolVLM-Instruct-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Mast; keywords=19; nonvisual metadata reused
- _Score:_ 🟠 D (46/100); Δ-3; Lacks visual description of image
- _Review focus:_ watchlist (metadata borrowing)
- _Timing:_ Load 0.65s; Gen 1.82s; Total 2.84s
- _Throughput:_ Prompt 4,272 TPS (1,722 tok); Gen 127 TPS (90 tok)
- _Tokens:_ prompt 1722 tok; estimated text 435 tok; estimated non-text 1287
  tok; generated 90 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe
> Description: Two sailing boats moored on a river with trees behind on the
> bank
> Keywords: Bird, Boat, Boating, Buoy, Bushes, Coast, Deben Estuary, England,
> Estuary, Europe, Foliage, Forest, Landscape, Moored, Mudflat, Nature,
> Outdoors, Peaceful, Rigging
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Keyword count violation (19; expected 10-18)
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-6bit"></a>

### ✅ mlx-community/paligemma2-10b-ft-docci-448-6bit

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Bird, Boat, Boating, Buoy, Bushes; text-sanity=numeric_loop
- _Score:_ ❌ F (20/100); Δ-30; Output lacks detail
- _Review focus:_ watchlist (context ignored, missing sections)
- _Timing:_ Load 1.56s; Gen 2.53s; Total 4.46s
- _Throughput:_ Prompt 1,366 TPS (1,530 tok); Gen 32.0 TPS (27 tok)
- _Tokens:_ prompt 1530 tok; estimated text 435 tok; estimated non-text 1095
  tok; generated 27 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#45; Location: 52° 0' 0" N, 1° 0' 0" E.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Bird, Boat, Boating, Buoy, Bushes)
- Text sanity issue (numeric_loop)
- Missing sections (title, description, keywords)
- Ignores trusted hints

---

<a id="model-mlx-community-gemma-4-26b-a4b-it-4bit"></a>

### ✅ mlx-community/gemma-4-26b-a4b-it-4bit

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Buoy, Bushes, Coast
- _Score:_ 🏆 A (81/100); Δ+32; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 2.34s; Gen 2.60s; Total 5.31s
- _Throughput:_ Prompt 1,336 TPS (783 tok); Gen 60.4 TPS (88 tok)
- _Tokens:_ prompt 783 tok; estimated text 435 tok; estimated non-text 348
  tok; generated 88 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Two small sailing boats moored on a river
>
> &#8203;Description:
> Two small wooden sailing boats with tall masts are moored on the water
> against a backdrop of dense green trees. The scene is captured in natural
> light with a lush forest bank in the background.
>
> &#8203;Keywords:
> boat, sailing boat, mast, rigging, mooring, river, water, trees, foliage,
> forest, landscape, nature, outdoors, greenery, vertical composition
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

---

<a id="model-microsoft-phi-35-vision-instruct"></a>

### ✅ microsoft/Phi-3.5-vision-instruct

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Buoy, Bushes, Coast
- _Score:_ 🟡 C (54/100); Δ+4; Lacks visual description of image
- _Timing:_ Load 0.93s; Gen 2.61s; Total 3.91s
- _Throughput:_ Prompt 3,700 TPS (1,330 tok); Gen 57.7 TPS (101 tok)
- _Tokens:_ prompt 1330 tok; estimated text 435 tok; estimated non-text 895
  tok; generated 101 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Sailing Boats at Deben Estuary
>
> Description: Two sailing boats are moored on a river, surrounded by lush
> greenery and trees. The boats are docked on the riverbank, with one boat
> having a prominent mast and rigging.
>
> Keywords: Sailing Boats, Deben Estuary, River, Trees, Greenery, Mast,
> Rigging, Mudflat, Nature, Outdoors
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-diffusiongemma-26b-a4b-it-8bit"></a>

### ✅ mlx-community/diffusiongemma-26B-A4B-it-8bit

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Coast, Mudflat, Woodbridge, GBR;
  nonvisual metadata reused
- _Score:_ 🟡 C (64/100); Δ+15; None identified
- _Review focus:_ watchlist (metadata borrowing)
- _Timing:_ Load 3.28s; Gen 2.62s; Total 6.30s
- _Throughput:_ Prompt 2,028 TPS (783 tok); Gen 58.9 TPS (89 tok)
- _Tokens:_ prompt 783 tok; estimated text 435 tok; estimated non-text 348
  tok; generated 89 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> &#45; Two sailing boats moored on the Deben Estuary
>
> &#8203;Description:
> &#45; Two wooden sailboats with tall masts are moored on a river next to a dense
> green forest bank. A small buoy floats in the water near the vessels.
>
> &#8203;Keywords:
> &#45; boat, boating, buoy, bushes, Deben Estuary, England, estuary, foliage,
> forest, landscape, mast, moored, nature, outdoors, peaceful, rigging
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Nonvisual metadata borrowing

---

<a id="model-mlx-community-ministral-3-3b-instruct-2512-4bit"></a>

### ✅ mlx-community/Ministral-3-3B-Instruct-2512-4bit

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Buoy, Bushes, Coast
- _Score:_ 🏆 A (86/100); Δ+37; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 2.12s; Gen 2.67s; Total 5.20s
- _Throughput:_ Prompt 2,148 TPS (3,123 tok); Gen 168 TPS (107 tok)
- _Tokens:_ prompt 3123 tok; estimated text 435 tok; estimated non-text 2688
  tok; generated 107 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Traditional sailboat moored by riverbank with dense foliage
>
> &#8203;Description:
> A classic wooden sailboat with tall masts and red sails is moored on calm
> water beside a grassy shoreline thick with lush green trees and shrubs. The
> lighting suggests overcast or soft natural daylight.
>
> &#8203;Keywords:
> boat, sailboat, wooden vessel, riverbank, moored, sail masts, red sails,
> green foliage, dense trees, nature scene, waterfront, maritime setting,
> coastal landscape
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

---

<a id="model-mlx-community-phi-35-vision-instruct-bf16"></a>

### ✅ mlx-community/Phi-3.5-vision-instruct-bf16

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Buoy, Bushes, Coast
- _Score:_ 🟡 C (54/100); Δ+4; Lacks visual description of image
- _Timing:_ Load 0.98s; Gen 2.69s; Total 4.05s
- _Throughput:_ Prompt 3,762 TPS (1,330 tok); Gen 55.5 TPS (101 tok)
- _Tokens:_ prompt 1330 tok; estimated text 435 tok; estimated non-text 895
  tok; generated 101 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Sailing Boats at Deben Estuary
>
> Description: Two sailing boats are moored on a river, surrounded by lush
> greenery and trees. The boats are docked on the riverbank, with one boat
> having a prominent mast and rigging.
>
> Keywords: Sailing Boats, Deben Estuary, River, Trees, Greenery, Mast,
> Rigging, Mudflat, Nature, Outdoors
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-smolvlm2-22b-instruct-mlx"></a>

### ✅ mlx-community/SmolVLM2-2.2B-Instruct-mlx

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Mast; keyword duplication=44%; context
  echo=100%; nonvisual metadata reused
- _Score:_ 🟡 C (55/100); Δ+6; Lacks visual description of image
- _Review focus:_ watchlist (context echo, instruction echo, metadata
  borrowing, reasoning leak)
- _Timing:_ Load 0.66s; Gen 2.89s; Total 3.91s
- _Throughput:_ Prompt 1,861 TPS (622 tok); Gen 134 TPS (248 tok)
- _Tokens:_ prompt 622 tok; estimated text 435 tok; estimated non-text 187
  tok; generated 248 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe
> Description: Two sailing boats moored on a river with trees behind on the
> bank
> Keywords: Bird, Boat, Boating, Buoy, Bushes, Coast, Deben Estuary, England,
> Estuary, Europe, Foliage, Forest, Landscape, Moored, Mudflat, Nature,
> Outdoors, Peaceful, Rigging
>
> Context: Existing metadata hints (high confidence; use only when visually
> &#8203;confirmed):
> &#45; Title hint: , Deben Estuary, Woodbridge, England, UK, GBR, Europe
> &#45; Description hint: Two sailing boats moored on a river with trees behind on
> the bank
> &#45; Keyword hints: Bird, Boat, Boating, Buoy, Bushes, Coast, Deben Estuary,
> England, Estuary, Europe, Foliage, Forest, Landscape, Moored, Mudflat,
> Nature, Outdoors, Peaceful, Rigging
> &#45; Capture metadata: Taken on 2026-07-04 19:10:04 BST (at 19:10:04 local
> time).
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Text sanity issue (numeric_loop)
- Keyword count violation (48; expected 10-18)
- Keyword duplication (44% duplicated terms)
- Reasoning leak (context: existing metadata hints, title hint:)
- Context echo (100% overlap)
- Instruction echo
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-llava-v16-mistral-7b-8bit"></a>

### ✅ mlx-community/llava-v1.6-mistral-7b-8bit

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Bird, Boating, Buoy, Bushes, Coast
- _Score:_ ❌ F (7/100); Δ-42; Output lacks detail
- _Review focus:_ watchlist (context ignored, missing sections)
- _Timing:_ Load 0.96s; Gen 2.93s; Total 4.26s
- _Throughput:_ Prompt 1,326 TPS (2,730 tok); Gen 68.1 TPS (11 tok)
- _Tokens:_ prompt 2730 tok; estimated text 435 tok; estimated non-text 2295
  tok; generated 11 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image shows a boat on a river.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Bird, Boating, Buoy, Bushes, Coast)
- Missing sections (title, description, keywords)
- Ignores trusted hints

---

<a id="model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8"></a>

### ✅ mlx-community/diffusiongemma-26B-A4B-it-mxfp8

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Buoy, Bushes, Coast, Mudflat; nonvisual
  metadata reused
- _Score:_ 🟡 C (63/100); Δ+14; Limited novel information
- _Review focus:_ watchlist (metadata borrowing)
- _Timing:_ Load 3.19s; Gen 3.44s; Total 7.01s
- _Throughput:_ Prompt 2,072 TPS (783 tok); Gen 33.0 TPS (78 tok)
- _Tokens:_ prompt 783 tok; estimated text 435 tok; estimated non-text 348
  tok; generated 78 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> &#45; Two sailing boats moored on the Deben Estuary
>
> &#8203;Description:
> &#45; Two small sailing boats are moored on a river in front of a dense green
> forest on the bank.
>
> &#8203;Keywords:
> &#45; Boat, Boating, Estuary, Mast, Rigging, River, Forest, Trees, Foliage,
> Nature, Outdoors, Landscape, Peaceful, England, Europe
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Nonvisual metadata borrowing

---

<a id="model-mlx-community-internvl3-8b-bf16"></a>

### ✅ mlx-community/InternVL3-8B-bf16

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Coast, Deben Estuary, Estuary
- _Score:_ 🟡 C (64/100); Δ+14; Limited novel information
- _Timing:_ Load 1.65s; Gen 3.52s; Total 5.56s
- _Throughput:_ Prompt 2,767 TPS (2,295 tok); Gen 34.8 TPS (71 tok)
- _Tokens:_ prompt 2295 tok; estimated text 435 tok; estimated non-text 1860
  tok; generated 71 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Two sailing boats moored on river
>
> Description: Two sailing boats with orange masts are moored on a calm river,
> surrounded by dense green foliage on the bank.
>
> Keywords: Boats, Sailing, River, Moored, Foliage, Trees, Mast, Nature,
> Outdoors, Peaceful, Buoy, Bushes
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

---

<a id="model-mlx-community-gemma-3-27b-it-qat-4bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-4bit

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Bushes, Coast, Deben Estuary, Estuary
- _Score:_ ✅ B (79/100); Δ+30; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 2.28s; Gen 4.25s; Total 6.91s
- _Throughput:_ Prompt 637 TPS (778 tok); Gen 31.5 TPS (78 tok)
- _Tokens:_ prompt 778 tok; estimated text 435 tok; estimated non-text 343
  tok; generated 78 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Two sailing boats moored on a waterway
>
> &#8203;Description:
> Two sailing boats are moored on calm water, with dense green foliage lining
> the banks. A single orange buoy is visible in the foreground.
>
> &#8203;Keywords:
> Boat, Boating, Buoy, Foliage, Green, Landscape, Mast, Moored, Outdoors,
> River, Sailing, Trees, Water, Waterway
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-mxfp4"></a>

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-mxfp4

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Bushes, Coast, Deben Estuary
- _Score:_ 🏆 A (89/100); Δ+40; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 1.33s; Gen 4.28s; Total 5.99s
- _Throughput:_ Prompt 1,374 TPS (3,124 tok); Gen 65.0 TPS (95 tok)
- _Tokens:_ prompt 3124 tok; estimated text 435 tok; estimated non-text 2689
  tok; generated 95 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Sailing Boat on Calm Water with Forest Background
>
> &#8203;Description:
> A single wooden sailboat with two tall masts and red sails is moored on calm
> water, surrounded by dense green foliage and trees along the shoreline. The
> boat appears to be anchored near a buoy.
>
> &#8203;Keywords:
> Sailing boat, wooden hull, two masts, red sails, calm water, dense forest,
> green foliage, shoreline, anchored, buoy
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

---

<a id="model-mlx-community-pixtral-12b-8bit"></a>

### ✅ mlx-community/pixtral-12b-8bit

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 87% and the output stays weak under that load.
- _Key signals:_ output/prompt=2.35%; nontext prompt burden=87%; missing
  terms: Bird, Buoy, Bushes, Coast, Deben Estuary; nonvisual metadata reused
- _Score:_ ✅ B (70/100); Δ+21; Limited novel information
- _Review focus:_ watchlist (metadata borrowing)
- _Timing:_ Load 1.73s; Gen 4.43s; Total 6.52s
- _Throughput:_ Prompt 1,720 TPS (3,317 tok); Gen 39.0 TPS (78 tok)
- _Tokens:_ prompt 3317 tok; estimated text 435 tok; estimated non-text 2882
  tok; generated 78 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Two Boats Moored on River
>
> &#8203;Description:
> Two sailing boats are moored on a river with dense green foliage on the bank
> behind them.
>
> &#8203;Keywords:
> Boat, Boating, Moored, River, Sailing, Trees, Foliage, Green, Mast, Rigging,
> Nature, Outdoors, Peaceful, Water, Riverbank, Forest, Europe
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context_budget
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-internvl3-14b-8bit"></a>

### ✅ mlx-community/InternVL3-14B-8bit

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Bushes, Coast, Deben Estuary, Estuary
- _Score:_ 🟡 C (57/100); Δ+8; Limited novel information
- _Timing:_ Load 1.71s; Gen 4.54s; Total 6.63s
- _Throughput:_ Prompt 1,414 TPS (2,295 tok); Gen 31.4 TPS (72 tok)
- _Tokens:_ prompt 2295 tok; estimated text 435 tok; estimated non-text 1860
  tok; generated 72 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Sailing Boats on River
>
> &#8203;Description:
> Two sailing boats are moored on a calm river with dense green foliage on the
> bank behind them.
>
> &#8203;Keywords:
> Boat, Boating, Buoy, Foliage, Forest, Landscape, Mast, Moored, Nature,
> Outdoors, Peaceful, River, Rigging, Trees, Water
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Title length violation (4 words; expected 5-10)

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-nvfp4"></a>

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-nvfp4

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 86% and the output stays weak under that load.
- _Key signals:_ output/prompt=3.84%; nontext prompt burden=86%; missing
  terms: Bird, Boating, Bushes, Coast, Forest; nonvisual metadata reused
- _Score:_ 🏆 A (83/100); Δ+34; None identified
- _Review focus:_ watchlist (metadata borrowing)
- _Timing:_ Load 1.44s; Gen 4.75s; Total 6.56s
- _Throughput:_ Prompt 1,380 TPS (3,124 tok); Gen 61.8 TPS (120 tok)
- _Tokens:_ prompt 3124 tok; estimated text 435 tok; estimated non-text 2689
  tok; generated 120 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;&#42;&#42;Title:&#42;&#42;
> Sailing Boat Moored on Deben Estuary
>
> &#8203;&#42;&#42;Description:&#42;&#42;
> A single wooden sailboat with two masts is anchored near a buoy on calm
> waters, surrounded by dense green foliage along the riverbank. Visible
> rigging and sails, including red covers, contrast against the natural
> setting.
>
> &#8203;&#42;&#42;Keywords:&#42;&#42;
> Sailboat, Wooden hull, Two masts, Riverbank, Dense foliage, Green foliage,
> Calm waters, Buoy, Red sails, Rigging, Moored, Deben Estuary, Natural
> setting, Woodbridge, England
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context_budget
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-gemma-3n-e2b-4bit"></a>

### ✅ mlx-community/gemma-3n-E2B-4bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Bird, Boat, Boating, Buoy, Bushes; nonvisual
  metadata reused
- _Score:_ ❌ F (2/100); Δ-47; Output too short to be useful
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, generation
  loop, metadata borrowing, missing sections)
- _Timing:_ Load 1.46s; Gen 4.86s; Total 6.71s
- _Throughput:_ Prompt 2,459 TPS (769 tok); Gen 124 TPS (500 tok)
- _Tokens:_ prompt 769 tok; estimated text 435 tok; estimated non-text 334
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
>
> &#45;
> 19:10:18:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Bird, Boat, Boating, Buoy, Bushes)
- Output degeneration (character_loop: ':00' repeated)
- Generation loop (degeneration)
- Missing sections (title, description, keywords)
- Nonvisual metadata borrowing
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-idefics3-8b-llama3-bf16"></a>

### ✅ mlx-community/Idefics3-8B-Llama3-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Buoy, Bushes, Coast; nonvisual
  metadata reused; formatting=Unknown tags: &lt;end_of_utterance&gt;
- _Score:_ ✅ B (73/100); Δ+24; None identified
- _Review focus:_ watchlist (metadata borrowing)
- _Timing:_ Load 1.93s; Gen 5.73s; Total 8.03s
- _Throughput:_ Prompt 2,506 TPS (2,805 tok); Gen 32.1 TPS (126 tok)
- _Tokens:_ prompt 2805 tok; estimated text 435 tok; estimated non-text 2370
  tok; generated 126 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Sailing Boats on the Deben Estuary, Woodbridge, England, UK.
>
> Description: Two sailing boats are moored on a river with trees behind on
> the bank. The boats have their sails down and are anchored in the water. The
> trees are green and lush, indicating a healthy environment. The sky is not
> visible in the image, but the lighting suggests it is daytime.
>
> Keywords: Sailing Boats, Deben Estuary, Woodbridge, England, UK, River,
> Trees, Moored, Sails, Anchored, Green, Lush, Healthy, Environment,
> Daytime.<end_of_utterance>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Description sentence violation (4; expected 1-2)
- Nonvisual metadata borrowing
- Unknown tags: <end_of_utterance>

---

<a id="model-mlx-community-pixtral-12b-bf16"></a>

### ✅ mlx-community/pixtral-12b-bf16

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Bushes, Coast, Deben Estuary, Estuary
- _Score:_ ✅ B (70/100); Δ+21; Limited novel information
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 2.58s; Gen 6.22s; Total 9.17s
- _Throughput:_ Prompt 1,911 TPS (3,317 tok); Gen 20.1 TPS (80 tok)
- _Tokens:_ prompt 3317 tok; estimated text 435 tok; estimated non-text 2882
  tok; generated 80 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Two Boats Moored on River
>
> &#8203;Description:
> Two sailing boats are moored on a river with dense green foliage on the bank
> behind them.
>
> &#8203;Keywords:
> Boat, Boating, Moored, River, Sailing, Trees, Foliage, Green, Mast, Rigging,
> Nature, Outdoors, Peaceful, Water, Riverbank, Wooden, Buoy
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

---

<a id="model-mlx-community-gemma-4-31b-it-4bit"></a>

### ✅ mlx-community/gemma-4-31b-it-4bit

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Bushes, Coast, Deben Estuary
- _Score:_ ✅ B (74/100); Δ+25; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 2.77s; Gen 6.33s; Total 9.51s
- _Throughput:_ Prompt 474 TPS (783 tok); Gen 19.6 TPS (81 tok)
- _Tokens:_ prompt 783 tok; estimated text 435 tok; estimated non-text 348
  tok; generated 81 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Sailing boats moored on a river with trees
>
> Description: Two sailing boats with tall wooden masts are moored on a river
> in front of a dense green forest. A small buoy floats in the water in the
> foreground.
>
> Keywords: boat, sailing boat, river, trees, forest, foliage, mast, rigging,
> moored, buoy, water, nature, outdoors, landscape, green, wooden mast
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

---

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-bf16"></a>

### ✅ mlx-community/paligemma2-10b-ft-docci-448-bf16

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Bird, Boat, Boating, Buoy, Bushes; text-sanity=gibberish(token_noise)
- _Score:_ ❌ F (19/100); Δ-30; Output lacks detail
- _Review focus:_ watchlist (context ignored, missing sections, text sanity)
- _Timing:_ Load 2.50s; Gen 6.48s; Total 9.34s
- _Throughput:_ Prompt 1,132 TPS (1,530 tok); Gen 5.49 TPS (25 tok)
- _Tokens:_ prompt 1530 tok; estimated text 435 tok; estimated non-text 1095
  tok; generated 25 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#45; Location: 51.9786° N, 1.4786° E.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: Bird, Boat, Boating, Buoy, Bushes)
- Text sanity issue (gibberish(token_noise))
- Missing sections (title, description, keywords)
- Ignores trusted hints
- suspicious_precision: 2 overly precise numbers

---

<a id="model-mlx-community-gemma-3n-e4b-it-bf16"></a>

### ✅ mlx-community/gemma-3n-E4B-it-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Bushes, Coast, Forest, Landscape;
  keywords=22; nonvisual metadata reused
- _Score:_ 🏆 A (84/100); Δ+34; None identified
- _Review focus:_ watchlist (metadata borrowing)
- _Timing:_ Load 2.30s; Gen 6.57s; Total 9.25s
- _Throughput:_ Prompt 1,787 TPS (777 tok); Gen 48.5 TPS (272 tok)
- _Tokens:_ prompt 777 tok; estimated text 435 tok; estimated non-text 342
  tok; generated 272 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#35;# Image Description for Cataloguing
>
> &#42;&#42;Title:&#42;&#42; Two Sailboats Moored on the Deben Estuary
>
> &#42;&#42;Description:&#42;&#42; Two small sailing boats are moored on a calm body of water,
> likely the Deben Estuary in England. The boats are positioned side-by-side,
> with their masts reaching upwards. The boat on the left is predominantly
> dark with a white deck and orange and white striped masts. The boat on the
> right is lighter in colour, possibly white or light green, with red and
> white striped masts. Both boats have their sails furled.
>
> The background is dominated by lush green trees lining the banks of the
> estuary. The foliage is dense and creates a vibrant backdrop. A small, brown
> duck is swimming near the stern of the lighter boat. The water is a muted
> grey-blue, reflecting the overcast sky. The overall scene is peaceful and
> depicts a typical day on the estuary.
>
> &#42;&#42;Keywords:&#42;&#42; Boat, Boating, Sailboat, Moored, Estuary, Water, Trees, Duck,
> England, Nature, Calm, Peaceful, Mast, Rigging, Buoy.
>
> &#42;&#42;Location:&#42;&#42; Deben Estuary, Woodbury, England, UK, GBR, Europe.
>
> &#42;&#42;Date Taken:&#42;&#42; 2026-07-04 19:10:04 BST.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Description sentence violation (10; expected 1-2)
- Keyword count violation (22; expected 10-18)
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-devstral-small-2-24b-instruct-2512-5bit"></a>

### ✅ mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Bushes, Coast, Deben Estuary
- _Score:_ ✅ B (71/100); Δ+22; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 2.14s; Gen 7.02s; Total 9.59s
- _Throughput:_ Prompt 696 TPS (2,591 tok); Gen 29.9 TPS (82 tok)
- _Tokens:_ prompt 2591 tok; estimated text 435 tok; estimated non-text 2156
  tok; generated 82 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Sailing Boat on River
>
> &#8203;Description:
> A wooden sailing boat with two masts is moored on a calm river, surrounded
> by dense green trees on the bank. A red buoy floats nearby in the water.
>
> &#8203;Keywords:
> Boat, Sailing boat, River, Trees, Green foliage, Wooden mast, Red buoy, Calm
> water, Moored, Outdoors, Nature, Landscape
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Title length violation (4 words; expected 5-10)

---

<a id="model-mlx-community-gemma-3-27b-it-qat-8bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-8bit

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Buoy, Bushes, Deben Estuary, Forest
- _Score:_ 🏆 A (85/100); Δ+36; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 3.31s; Gen 7.06s; Total 10.75s
- _Throughput:_ Prompt 558 TPS (778 tok); Gen 17.7 TPS (91 tok)
- _Tokens:_ prompt 778 tok; estimated text 435 tok; estimated non-text 343
  tok; generated 91 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Two Sailing Boats Moored on River
>
> &#8203;Description:
> Two sailing boats are moored on a grey river, with dense green trees lining
> the bank behind them. The lighting is soft and even, illuminating the boats
> and foliage.
>
> &#8203;Keywords:
> Boat, Boating, Coast, Estuary, Foliage, Landscape, Mast, Moored, Nature,
> Outdoors, River, Sailing, Trees, Water, Grey, Green, Sunlight, Reflection
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

---

<a id="model-jqlive-kimi-vl-a3b-thinking-2506-6bit"></a>

### ✅ jqlive/Kimi-VL-A3B-Thinking-2506-6bit

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title; missing terms: Bird, Boating,
  Bushes, Mudflat, river; keyword duplication=39%; nonvisual metadata reused
- _Score:_ 🟡 C (58/100); Δ+8; Missing requested structure
- _Review focus:_ watchlist (metadata borrowing, missing sections, reasoning
  leak)
- _Timing:_ Load 2.00s; Gen 7.19s; Total 9.59s
- _Throughput:_ Prompt 1,584 TPS (1,514 tok); Gen 70.0 TPS (393 tok)
- _Tokens:_ prompt 1514 tok; estimated text 435 tok; estimated non-text 1079
  tok; generated 393 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ◁think▷Got it, let's analyze the image step by step.
>
> First, the title: The image shows two sailing boats on a body of water with
> trees in the background. The location hint mentions Deben Estuary,
> Woodbridge, England, UK, GBR, Europe. So the title should be concise and
> factual. Let's check the visible elements: two boats, water, trees, buoy. So
> "Two sailing boats moored on Deben Estuary" fits.
>
> Description: The main subject is two sailing boats on water, moored near a
> tree-lined shore. The setting is Deben Estuary, with lush green foliage,
> calm water, and a buoy visible. Lighting is natural daylight, peaceful
> outdoor scene.
>
> Keywords: Sailing boats, Deben Estuary, England, Europe, Trees, Water,
> Moored, Buoy, Foliage, Landscape, Nature, Outdoors, Peaceful, Mast, Rigging,
> Coast, Forest.
>
> Now, check against the rules: only visible details. The title is two sailing
> boats moored on Deben Estuary. Description: Two sailing boats moored on
> Deben Estuary with lush green trees in the background, calm water, and a
> buoy visible. Keywords: Sailing boats, Deben Estuary, England, Europe,
> Trees, Water, Moored, Buoy, Foliage, Landscape, Nature, Outdoors, Peaceful,
> Mast, Rigging, Coast, Forest.◁/think▷Title:
> Two sailing boats moored on Deben Estuary
>
> &#8203;Description:
> Two sailing boats moored on Deben Estuary with lush green trees in the
> background, calm water, and a buoy visible.
>
> &#8203;Keywords:
> Sailing boats, Deben Estuary, England, Europe, Trees, Water, Moored, Buoy,
> Foliage, Landscape, Nature, Outdoors, Peaceful, Mast, Rigging, Coast,
> Forest<|im_assistant|>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title)
- Description sentence violation (3; expected 1-2)
- Keyword count violation (38; expected 10-18)
- Keyword duplication (39% duplicated terms)
- Reasoning leak (◁think▷, ◁/think▷)
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-glm-46v-flash-6bit"></a>

### ✅ mlx-community/GLM-4.6V-Flash-6bit

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 93% and the output stays weak under that load.
- _Key signals:_ output/prompt=1.41%; nontext prompt burden=93%; missing
  sections: title; missing terms: Bird, Boating, Bushes, Coast, Deben Estuary
- _Score:_ ✅ B (66/100); Δ+16; None identified
- _Review focus:_ watchlist (hallucination, missing sections)
- _Timing:_ Load 1.49s; Gen 7.63s; Total 9.49s
- _Throughput:_ Prompt 1,179 TPS (6,542 tok); Gen 58.4 TPS (92 tok)
- _Tokens:_ prompt 6542 tok; estimated text 435 tok; estimated non-text 6107
  tok; generated 92 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> <|begin_of_box|>Title:
> Two sailing boats moored on water with trees behind
>
> &#8203;Description:
> Two sailing boats are moored on a body of water. Behind the boats, there is
> a dense area of green trees and foliage. The water appears calm.
>
> &#8203;Keywords:
> Sailing boat, Boat, Moored, Water, Trees, Green foliage, Estuary, Nature,
> Outdoors, Peaceful, Mast, Rigging, Buoy, Riverbank, Vegetation<|end_of_box|>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context_budget
- Missing sections (title)
- Description sentence violation (3; expected 1-2)
- Contains unexpected table

---

<a id="model-mlx-community-llama-32-11b-vision-instruct-8bit"></a>

### ✅ mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Bird, Woodbridge, GBR; nonvisual metadata reused
- _Score:_ 🟠 D (39/100); Δ-10; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (metadata borrowing, missing sections)
- _Timing:_ Load 1.51s; Gen 8.51s; Total 10.41s
- _Throughput:_ Prompt 296 TPS (493 tok); Gen 20.5 TPS (129 tok)
- _Tokens:_ prompt 493 tok; estimated text 435 tok; estimated non-text 58 tok;
  generated 129 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title
> Deben Estuary, England, UK
>
> Description
> Two sailing boats are moored on a river with trees behind on the bank. The
> river is surrounded by lush greenery and has a peaceful atmosphere. The
> boats are in good condition, with their masts and rigging intact. The
> riverbank is covered with dense foliage, and the water is calm and clear.
>
> Keywords
> Boat, Boating, Buoy, Bushes, Coast, Deben Estuary, England, Estuary,
> Foliage, Forest, Landscape, Mast, Moored, Mudflat, Nature, Outdoors,
> Peaceful, Rigging
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-8bit"></a>

### ✅ mlx-community/Kimi-VL-A3B-Thinking-8bit

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title; missing terms:
  Bird, Boating, Bushes, Coast, Forest; nonvisual metadata reused
- _Score:_ ✅ B (66/100); Δ+17; Lacks visual description of image
- _Review focus:_ watchlist (cutoff, metadata borrowing, missing sections,
  reasoning leak)
- _Timing:_ Load 2.18s; Gen 9.43s; Total 11.99s
- _Throughput:_ Prompt 1,565 TPS (1,514 tok); Gen 63.9 TPS (500 tok)
- _Tokens:_ prompt 1514 tok; estimated text 435 tok; estimated non-text 1079
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ◁think▷Okay, let's tackle this metadata analysis. First, I need to look at
> the image and extract the metadata based on what's clearly visible. The user
> provided some hints, but I should only use what's actually in the image.
>
> Starting with the title. The hints mention Deben Estuary, Woodbridge,
> England, UK, GBR, Europe. The image shows boats on water with trees in the
> background. The location is definitely the Deben Estuary in Woodbridge,
> England. So the title should include that. Maybe "Two Sailboats Moored on
> Deben Estuary, Woodbridge, England".
>
> For the description, the main subject is two sailboats anchored on the
> water. The setting is the estuary with dense green foliage and trees along
> the banks. The lighting seems natural, possibly overcast since the colors
> are a bit muted. The boats have masts and rigging, and there's a buoy
> visible. The overall atmosphere is peaceful and natural. So the description
> should mention the boats, the estuary, the trees, and the peaceful setting.
>
> Keywords need to be from what's visible. The boats are sailboats, so
> "sailboats" and "masts" are definite. The location is Deben Estuary,
> England, Europe. The environment includes foliage, trees, water, buoy,
> mudflat (since the boats are anchored near the shore which might have
> mudflat areas). The image is outdoors, peaceful, natural, and the style is
> landscape. The time is given in the capture metadata as 2026-07-04 19:10:04
> BST, but since the user said not to infer time period unless obvious, maybe
> omit that. So keywords would be: sailboats, boats, masts, buoy, foliage,
> trees, Deben Estuary, England, Europe, landscape, mudflat, nature, outdoors,
> peaceful, rigging, water.◁/think▷Title: Two Sailboats Moored on Deben
> Estuary, Woodbridge, England
>
> Description: Two sailboats with orange masts and rigging are anchored on
> calm water, surrounded by dense green foliage and trees lining the banks of
> the Deben Estuary in Woodbridge, England. The scene captures a peaceful
> natural landscape with a visible buoy and mudflat near the shoreline.
>
> Keywords: sailboats, boats, masts, buoy, foliage, trees, Deben Estuary,
> England, Europe, mudflat, nature, outdoors
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title)
- Reasoning leak (◁think▷, ◁/think▷)
- Nonvisual metadata borrowing
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
- _Score:_ 🟠 D (49/100); Δ-0; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, generation loop, missing
  sections, repetitive)
- _Timing:_ Load 1.15s; Gen 11.83s; Total 13.35s
- _Throughput:_ Prompt 3,945 TPS (4,602 tok); Gen 50.6 TPS (500 tok)
- _Tokens:_ prompt 4602 tok; estimated text 435 tok; estimated non-text 4167
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
- _Key signals:_ hit token cap (500); missing terms: Bird, Boating, Buoy,
  Bushes, Coast; keyword duplication=84%; nonvisual metadata reused
- _Score:_ ✅ B (79/100); Δ+29; None identified
- _Review focus:_ watchlist (cutoff, metadata borrowing)
- _Timing:_ Load 1.32s; Gen 12.25s; Total 13.95s
- _Throughput:_ Prompt 1,192 TPS (6,542 tok); Gen 79.9 TPS (500 tok)
- _Tokens:_ prompt 6542 tok; estimated text 435 tok; estimated non-text 6107
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Two sailing boats moored on a river
>
> &#8203;Description:
> Two sailing boats are moored on a river. The boats have orange masts and red
> covers. The background features dense green foliage along the riverbank.
>
> &#8203;Keywords:
> Boat, Moored, River, Sailing, Mast, Rigging, Foliage, Estuary, England,
> Nature, Outdoors, Peaceful, Water, Wooden boat, Red cover, Orange mast,
> Dense green foliage, Riverbank, Sailing boat, Mooring, River water, Wooden
> hull, Red sail cover, Orange mast, Sailing rig, Moored, River, Sailing,
> Mast, Rigging, Foliage, Estuary, England, Nature, Outdoors, Peaceful, Water,
> Wooden boat, Red cover, Orange mast, Dense green foliage, Riverbank,
> Sailing boat, Mooring, River water, Wooden hull, Red sail cover, Orange
> mast, Sailing rig, Moored, River, Sailing, Mast, Rigging, Foliage, Estuary,
> England, Nature, Outdoors, Peaceful, Water, Wooden boat, Red cover, Orange
> mast, Dense green foliage, Riverbank, Sailing boat, Mooring, River water,
> Wooden hull, Red sail cover, Orange mast, Sailing rig, Moored, River,
> Sailing, Mast, Rigging, Foliage, Estuary, England, Nature, Outdoors,
> Peaceful, Water, Wooden boat, Red cover, Orange mast, Dense green foliage,
> Riverbank, Sailing boat, Mooring, River water, Wooden hull, Red sail cover,
> Orange mast, Sailing rig, Moored, River, Sailing, Mast, Rigging, Foliage,
> Estuary, England, Nature, Outdoors, Peaceful, Water, Wooden boat, Red cover,
> Orange mast, Dense green foliage, Riverbank, Sailing boat, Mooring, River
> water, Wooden hull, Red sail cover, Orange mast, Sailing rig, Moored, River,
> Sailing, Mast, Rigging, Foliage, Estuary, England, Nature, Outdoors,
> Peaceful, Water, Wooden boat, Red cover, Orange mast, Dense green foliage,
> Riverbank, Sailing boat, Mooring, River water, Wooden hull, Red sail cover,
> Orange mast, Sailing rig, Moored, River, Sailing, Mast, Rigging, Foliage,
> Estuary
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Description sentence violation (3; expected 1-2)
- Keyword count violation (152; expected 10-18)
- Keyword duplication (84% duplicated terms)
- Nonvisual metadata borrowing
- Likely capped by max token budget

---

<a id="model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16"></a>

### ✅ mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ hit token cap (500); missing sections: title, description,
  keywords; missing terms: Bird, Boat, Boating, Buoy, Bushes; nonvisual
  metadata reused
- _Score:_ 🟠 D (46/100); Δ-4; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, degeneration, generation loop, metadata
  borrowing, missing sections, trusted hint degraded)
- _Timing:_ Load 4.93s; Gen 13.24s; Total 18.55s
- _Throughput:_ Prompt 1,075 TPS (1,826 tok); Gen 45.8 TPS (500 tok)
- _Tokens:_ prompt 1826 tok; estimated text 435 tok; estimated non-text 1391
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Let's tackle this step by step. First, the title needs to be 5-10 words,
> concrete and factual. The image has a sailboat in a wooded area by water.
> Wait, the context mentions Deben Estuary, Woodbridge, England, UK, GBR,
> Europe. But wait, the user said to use only details clearly visible. Wait,
> the prompt's context says "Existing metadata hints (high confidence; use
> only when visually confirmed)". Wait, but let's check the image description
> again. The user provided a prompt with a lot of repeated words, but the
> actual image (even though the text is messy) probably has a sailboat in a
> wooded area by water. Wait, no, wait the user's prompt has a lot of typos,
> but the context says the title hint is Deben Estuary, Woodbridge, England,
> UK, GBR, Europe. But wait, the rules say to use only details clearly
> visible. Wait, but the user's context says "Existing metadata hints (high
> confidence; use only when visually confirmed)". Wait, but let's go back. The
> user's instruction says: "Analyze this image for cataloguing metadata...
> Use only details that are clearly and definitely visible in the image. If a
> detail is uncertain, ambiguous, partially obscured, too small to verify, or
> not directly visible, leave it out. Treat the metadata hints below as a
> draft catalog record. Keep only details that are clearly confirmed by the
> image, correct anything contradicted by the image, and add important visible
> details that are definitely present." Oh, right, the metadata hints are a
> draft, but we have to use the image. Wait, but the user's prompt has a lot
> of repeated words, but maybe the actual image (since the user provided a
> text description with typos) is a sailboat in a wooded area by water. Wait,
> no, wait the user's prompt has a lot of typos, but the context says the
> title hint is Deben Estuary, Woodbridge, England, UK, GBR, Europe. But wait,
> the rules say to use only details clearly visible. Wait, but maybe the
> image is a sailboat in a wooded area by water. Let's start with the title.
> The title needs to be 5-10 words, concrete and factual. Let's see: "Sailboat
> in
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Output degeneration (incomplete_sentence: ends with 'in')
- Generation loop (degeneration)
- Missing sections (title, description, keywords)
- Nonvisual metadata borrowing
- Likely capped by max token budget
- Degrades trusted hints

---

<a id="model-qwen-qwen3-vl-2b-instruct"></a>

### ✅ Qwen/Qwen3-VL-2B-Instruct

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 97% and the output stays weak under that load.
- _Key signals:_ Output appears truncated to about 2 tokens.; At long prompt
  length (16729 tokens), output stayed unusually short (2 tokens; ratio 0.0%;
  weak text signal truncated).; output/prompt=0.01%; nontext prompt burden=97%
- _Score:_ ❌ F (0/100); Δ-49; Empty or minimal output
- _Review focus:_ watchlist (context ignored, harness, long context)
- _Timing:_ Load 0.67s; Gen 14.44s; Total 15.46s
- _Throughput:_ Prompt 1,228 TPS (16,729 tok); Gen 181 TPS (2 tok)
- _Tokens:_ prompt 16729 tok; estimated text 435 tok; estimated non-text 16294
  tok; generated 2 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#45;
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context_budget
- ⚠️HARNESS:long_context
- Context ignored (missing: Bird, Boat, Boating, Buoy, Bushes)
- Ignores trusted hints
- output:truncated(2tok)
- long_context_low_ratio(0.0%;16729->2;truncated)

---

<a id="model-mlx-community-molmo-7b-d-0924-8bit"></a>

### ✅ mlx-community/Molmo-7B-D-0924-8bit

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Bird, Boating, Buoy, Bushes, Coast; nonvisual metadata reused
- _Score:_ ❌ F (23/100); Δ-26; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (metadata borrowing, missing sections, trusted
  hint degraded)
- _Timing:_ Load 1.76s; Gen 14.77s; Total 16.91s
- _Throughput:_ Prompt 129 TPS (1,699 tok); Gen 46.7 TPS (31 tok)
- _Tokens:_ prompt 1699 tok; estimated text 435 tok; estimated non-text 1264
  tok; generated 31 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Photographer: John Doe. Location: Suffolk, England, UK. Event: Summer
> sailing day. Subject: Sailing boats in Deben Estuary.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)
- Nonvisual metadata borrowing
- Degrades trusted hints

---

<a id="model-mlx-community-glm-46v-nvfp4"></a>

### ✅ mlx-community/GLM-4.6V-nvfp4

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 93% and the output stays weak under that load.
- _Key signals:_ output/prompt=1.57%; nontext prompt burden=93%; missing
  sections: title; missing terms: Bird, Boating, Bushes, Coast, Deben Estuary
- _Score:_ 🏆 A (82/100); Δ+33; None identified
- _Review focus:_ watchlist (hallucination, missing sections)
- _Timing:_ Load 5.67s; Gen 14.84s; Total 20.90s
- _Throughput:_ Prompt 556 TPS (6,542 tok); Gen 40.7 TPS (103 tok)
- _Tokens:_ prompt 6542 tok; estimated text 435 tok; estimated non-text 6107
  tok; generated 103 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> <|begin_of_box|>Title:
> Two sailing boats moored on a river with trees behind
>
> &#8203;Description:
> Two sailing boats are moored on a river, with dense green trees and foliage
> visible on the bank behind them. The water appears calm, and a buoy is
> visible in the foreground.
>
> &#8203;Keywords:
> Boat, Sailing boat, Moored, River, Trees, Foliage, Buoy, Mast, Rigging,
> Black hull, White hull, Red cover, Wooden boat, Natural setting, Water,
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
  missing terms: Boating, Bushes, Deben Estuary, Landscape, Nature; nonvisual
  metadata reused
- _Score:_ ✅ B (76/100); Δ+26; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (cutoff, metadata borrowing, missing sections,
  reasoning leak)
- _Timing:_ Load 1.57s; Gen 14.89s; Total 16.84s
- _Throughput:_ Prompt 1,461 TPS (3,408 tok); Gen 41.5 TPS (500 tok)
- _Tokens:_ prompt 3408 tok; estimated text 435 tok; estimated non-text 2973
  tok; generated 500 tok; requested max 500 tok; stop reason max_tokens

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Here are my reasoning steps:
> We need to produce three sections: Title, Description, Keywords. Use only
> details that are clearly visible. The image shows a river or estuary with
> two sailing boats moored, one larger with multiple masts (looks like a ketch
> or maybe a schooner) and a smaller boat. The background is dense green
> trees/forest. There's a buoy (yellow) in the water near the smaller boat.
> The water is calm. The lighting appears to be late afternoon/evening
> (sunlight from side, shadows). The larger boat has a red canvas (cover) on
> the deck. The masts have orange and white stripes at the top. The hull of
> the larger boat is black with orange stripe and the name "MIRAGE" visible on
> the side. The smaller boat is white with a green stripe. The water is a
> muted blue-gray. The trees are lush green. There's a bird perched on a
> branch? Not clearly visible. The image does not show a bird. There's no
> visible bird. There's no sign of a mudflat. There's no visible sign of a
> coast. There's no sign of a buoy? Actually there is a yellow buoy near the
> smaller boat. So buoy is visible. There's a red cover on the larger boat.
> There's a rope? The rigging is visible. The masts are tall. The background
> is dense foliage. The water is calm. The lighting is soft, maybe golden
> hour. The image is taken from a distance, showing the boats and trees.
>
> We need to produce Title: 5-10 words, concrete and factual, limited to
> clearly visible content. So something like "Two moored sailing boats on a
> green riverbank". That's 7 words. Or "Two sailing boats moored on a
> tree-lined river". That's 9 words. Must be 5-10 words. Let's count: "Two
> sailing boats moored on a tree-lined river" = 9 words. Good.
>
> Description: 1-2 factual sentences describing main visible subject, setting,
> &#8203;lighting, action, and other distinctive visible details. So something like:
> "Two sailing boats are moored on a calm river with dense green trees lining
> the bank. The larger boat has a red canvas and orange‑striped masts, while
> a yellow buoy floats nearby." That's two sentences. Must be 1-2 sentences
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title, keywords)
- Description sentence violation (4; expected 1-2)
- Reasoning leak (here are my reasoning steps)
- Nonvisual metadata borrowing
- Likely capped by max token budget

---

<a id="model-mlx-community-molmo-7b-d-0924-bf16"></a>

### ✅ mlx-community/Molmo-7B-D-0924-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Bird, Boating, Buoy, Bushes, Foliage; nonvisual metadata reused
- _Score:_ ❌ F (30/100); Δ-20; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (metadata borrowing, missing sections, trusted
  hint degraded)
- _Timing:_ Load 1.80s; Gen 16.38s; Total 18.56s
- _Throughput:_ Prompt 123 TPS (1,699 tok); Gen 27.0 TPS (42 tok)
- _Tokens:_ prompt 1699 tok; estimated text 435 tok; estimated non-text 1264
  tok; generated 42 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Photographer: John Doe. Location: Suffolk, England, UK. Event: Summer day
> at the coast. Subject: Sailing boats in Deben Estuary. Style: Photographic
> representationalism realism.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)
- Nonvisual metadata borrowing
- Degrades trusted hints

---

<a id="model-meta-llama-llama-32-11b-vision-instruct"></a>

### ✅ meta-llama/Llama-3.2-11B-Vision-Instruct

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title, description, keywords; missing
  terms: Bird, Boating, Buoy, Bushes, Coast; nonvisual metadata reused
- _Score:_ 🟡 C (52/100); Δ+3; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (metadata borrowing, missing sections, trusted
  hint degraded)
- _Timing:_ Load 2.20s; Gen 16.62s; Total 19.20s
- _Throughput:_ Prompt 270 TPS (494 tok); Gen 5.13 TPS (73 tok)
- _Tokens:_ prompt 494 tok; estimated text 435 tok; estimated non-text 59 tok;
  generated 73 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Deben Estuary, England, UK
>
> The image depicts two sailboats moored in a river, with a dense forest in
> the background. The boats are positioned in the center of the image, with
> the forest serving as a natural backdrop. The scene is characterized by a
> peaceful and serene atmosphere, with the boats and the surrounding
> environment blending harmoniously together.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)
- Nonvisual metadata borrowing
- Degrades trusted hints

---

<a id="model-mlx-community-qwen3-vl-2b-instruct-bf16"></a>

### ✅ mlx-community/Qwen3-VL-2B-Instruct-bf16

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 97% and the output stays weak under that load.
- _Key signals:_ Output appears truncated to about 2 tokens.; At long prompt
  length (16729 tokens), output stayed unusually short (2 tokens; ratio 0.0%;
  weak text signal truncated).; output/prompt=0.01%; nontext prompt burden=97%
- _Score:_ ❌ F (0/100); Δ-49; Empty or minimal output
- _Review focus:_ watchlist (context ignored, harness, long context)
- _Timing:_ Load 1.00s; Gen 17.46s; Total 18.83s
- _Throughput:_ Prompt 1,009 TPS (16,729 tok); Gen 180 TPS (2 tok)
- _Tokens:_ prompt 16729 tok; estimated text 435 tok; estimated non-text 16294
  tok; generated 2 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#45;
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context_budget
- ⚠️HARNESS:long_context
- Context ignored (missing: Bird, Boat, Boating, Buoy, Bushes)
- Ignores trusted hints
- output:truncated(2tok)
- long_context_low_ratio(0.0%;16729->2;truncated)

---

<a id="model-mlx-community-x-reasoner-7b-8bit"></a>

### ✅ mlx-community/X-Reasoner-7B-8bit

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 97% and the output stays weak under that load.
- _Key signals:_ Output is very short relative to prompt size (0.1%) with weak
  text signal 'truncated', suggesting possible early-stop or prompt-handling
  issues.; At long prompt length (16740 tokens), output stayed unusually short
  (14 tokens; ratio 0.1%; weak text signal truncated).; output/prompt=0.08%;
  nontext prompt burden=97%
- _Score:_ ❌ F (6/100); Δ-43; Output too short to be useful
- _Review focus:_ watchlist (context ignored, degeneration, harness, long
  context, missing sections)
- _Timing:_ Load 1.11s; Gen 18.31s; Total 19.80s
- _Throughput:_ Prompt 971 TPS (16,740 tok); Gen 60.6 TPS (14 tok)
- _Tokens:_ prompt 16740 tok; estimated text 435 tok; estimated non-text 16305
  tok; generated 14 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> <|im_start|>
> addCriterion
> 要求：生成一个与图片内容相关的标题
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context_budget
- ⚠️HARNESS:long_context
- Context ignored (missing: Bird, Boat, Boating, Buoy, Bushes)
- Output degeneration (encoding_shift)
- Missing sections (title, description, keywords)
- Ignores trusted hints
- output:output_ratio(0.1%;truncated)
- long_context_low_ratio(0.1%;16740->14;truncated)

---

<a id="model-mlx-community-molmopoint-8b-fp16"></a>

### ✅ mlx-community/MolmoPoint-8B-fp16

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Deben Estuary, Mudflat, Peaceful, Deben
- _Score:_ 🟡 C (55/100); Δ+6; Lacks visual description of image
- _Timing:_ Load 3.28s; Gen 19.59s; Total 23.31s
- _Throughput:_ Prompt 1,298 TPS (3,327 tok); Gen 5.94 TPS (94 tok)
- _Tokens:_ prompt 3327 tok; estimated text 435 tok; estimated non-text 2892
  tok; generated 94 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Two moored sailing boats on a river with trees
>
> &#8203;Description:
> Two small sailing boats are anchored on a calm river, with a dense line of
> trees visible on the opposite bank. The boats have their sails down, and a
> buoy floats nearby in the water.
>
> &#8203;Keywords:
> Boat, Boating, Buoy, Bushes, Coast, Estuary, Foliage, Forest, Landscape,
> Mast, Moored, Nature, Outdoors, Rigging
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-2506-bf16"></a>

### ✅ mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

- _Recommendation:_ avoid for now; review verdict: model shortcoming
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; the requested
  output contract is not being met.
- _Key signals:_ missing sections: title; missing terms: Bird, Boating,
  Bushes, Deben Estuary, Estuary; nonvisual metadata reused; reasoning leak
- _Score:_ ✅ B (66/100); Δ+17; None identified
- _Review focus:_ watchlist (metadata borrowing, missing sections, reasoning
  leak)
- _Timing:_ Load 3.50s; Gen 23.07s; Total 26.96s
- _Throughput:_ Prompt 927 TPS (1,514 tok); Gen 4.65 TPS (96 tok)
- _Tokens:_ prompt 1514 tok; estimated text 435 tok; estimated non-text 1079
  tok; generated 96 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ◁think▷◁/think▷Title:
> Two sailing boats moored on a river with trees in the background
>
> &#8203;Description:
> Two sailing boats are moored on a calm river with dense green foliage and
> trees lining the riverbank in the background.
>
> &#8203;Keywords:
> Sailing boat, River, Trees, Moored, Foliage, Water, Boat, Mast, Buoy,
> Nature, Outdoors, Peaceful, England, Europe, Landscape,
> Coast<|im_assistant|>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Missing sections (title)
- Reasoning leak (◁think▷, ◁/think▷)
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-qwen3-vl-2b-thinking-bf16"></a>

### ✅ mlx-community/Qwen3-VL-2B-Thinking-bf16

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `mlx`; harness signal `long_context`
- _Next step:_ Inspect long-context cache behavior under heavy image-token
  burden.
- _Key signals:_ At long prompt length (16731 tokens), output became
  repetitive.; hit token cap (500); missing sections: title, description,
  keywords; missing terms: Bird, Boat, Boating, Buoy, Bushes
- _Score:_ ❌ F (20/100); Δ-29; Keywords are not specific or diverse enough
- _Review focus:_ watchlist (context ignored, cutoff, degeneration, generation
  loop, harness, long context, missing sections, repetitive, text sanity)
- _Timing:_ Load 0.95s; Gen 25.29s; Total 26.64s
- _Throughput:_ Prompt 896 TPS (16,731 tok); Gen 88.4 TPS (500 tok)
- _Tokens:_ prompt 16731 tok; estimated text 435 tok; estimated non-text 16296
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
- Likely capped by max token budget
- Ignores trusted hints
- long_context_repetition(16731tok)

---

<a id="model-mlx-community-qwen35-9b-mlx-4bit"></a>

### ✅ mlx-community/Qwen3.5-9B-MLX-4bit

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 97% and the output stays weak under that load.
- _Key signals:_ output/prompt=0.75%; nontext prompt burden=97%; missing
  terms: Bird, Bushes, Woodbridge, GBR, behind; keywords=20
- _Score:_ ✅ B (77/100); Δ+28; None identified
- _Review focus:_ watchlist (metadata borrowing)
- _Timing:_ Load 1.39s; Gen 60.37s; Total 62.13s
- _Throughput:_ Prompt 288 TPS (16,756 tok); Gen 92.4 TPS (126 tok)
- _Tokens:_ prompt 16756 tok; estimated text 435 tok; estimated non-text 16321
  tok; generated 126 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Two moored sailboats in a river estuary
>
> &#8203;Description:
> Two small sailing boats are moored side by side in calm water, with their
> sails furled and masts standing upright. The background consists of a
> densely wooded riverbank with lush green trees and undergrowth, and a
> visible mudflat along the shore.
>
> &#8203;Keywords:
> Boat, Boating, Buoy, Coast, Deben Estuary, England, Estuary, Foliage,
> Forest, Landscape, Mast, Moored, Mudflat, Nature, Outdoors, Peaceful,
> Rigging, River, Trees, Water
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context_budget
- Keyword count violation (20; expected 10-18)
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-qwen35-35b-a3b-4bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-4bit

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Bushes, Coast, Deben Estuary
- _Score:_ ✅ B (79/100); Δ+30; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 2.55s; Gen 63.71s; Total 66.69s
- _Throughput:_ Prompt 271 TPS (16,756 tok); Gen 104 TPS (106 tok)
- _Tokens:_ prompt 16756 tok; estimated text 435 tok; estimated non-text 16321
  tok; generated 106 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Two moored sailing boats on a river
> Description: Two sailing boats with tall wooden masts are moored on a river,
> backed by a dense bank of green trees and vegetation. A small buoy floats
> in the foreground water while the boats display red canvas covers and wooden
> hulls.
> Keywords: sailing boats, masts, river, moored, trees, green foliage, buoy,
> water, wooden hull, red canvas, rigging, nature, outdoors, estuary, bank,
> vegetation, calm, landscape
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

---

<a id="model-mlx-community-qwen35-35b-a3b-6bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-6bit

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 97% and the output stays weak under that load.
- _Key signals:_ output/prompt=0.59%; nontext prompt burden=97%; missing
  terms: Bird, Bushes, Coast, Peaceful, Woodbridge; nonvisual metadata reused
- _Score:_ ✅ B (68/100); Δ+19; None identified
- _Review focus:_ watchlist (metadata borrowing)
- _Timing:_ Load 3.21s; Gen 63.81s; Total 67.39s
- _Throughput:_ Prompt 271 TPS (16,756 tok); Gen 92.2 TPS (99 tok)
- _Tokens:_ prompt 16756 tok; estimated text 435 tok; estimated non-text 16321
  tok; generated 99 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Two sailing boats moored on a river with trees behind
> Description: Two sailing boats with tall masts are moored on a river, backed
> by a dense bank of green trees. A buoy floats in the foreground water.
> Keywords: Boat, Boating, Buoy, Deben Estuary, England, Estuary, Europe,
> Foliage, Forest, Landscape, Mast, Moored, Mudflat, Nature, Outdoors,
> Rigging, River, Sailing
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context_budget
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-qwen35-35b-a3b-bf16"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-bf16

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 97% and the output stays weak under that load.
- _Key signals:_ output/prompt=0.61%; nontext prompt burden=97%; missing
  terms: Bird, Bushes, Coast, Peaceful, Woodbridge; nonvisual metadata reused
- _Score:_ ✅ B (68/100); Δ+19; None identified
- _Review focus:_ watchlist (metadata borrowing)
- _Timing:_ Load 9.07s; Gen 65.38s; Total 74.82s
- _Throughput:_ Prompt 267 TPS (16,756 tok); Gen 64.6 TPS (102 tok)
- _Tokens:_ prompt 16756 tok; estimated text 435 tok; estimated non-text 16321
  tok; generated 102 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Two sailing boats moored on a river with trees behind
>
> Description: Two sailing boats with tall masts are moored on a calm river,
> backed by a dense bank of green trees and foliage. A buoy floats in the
> foreground water.
>
> Keywords: Boat, Boating, Buoy, Deben Estuary, England, Estuary, Europe,
> Foliage, Forest, Landscape, Mast, Moored, Mudflat, Nature, Outdoors, River,
> Rigging, Sailing
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context_budget
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-ornith-10-35b-bf16"></a>

### ✅ mlx-community/Ornith-1.0-35B-bf16

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Bushes, Coast, Deben Estuary
- _Score:_ 🏆 A (85/100); Δ+36; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 8.32s; Gen 66.40s; Total 75.10s
- _Throughput:_ Prompt 264 TPS (16,756 tok); Gen 60.9 TPS (105 tok)
- _Tokens:_ prompt 16756 tok; estimated text 435 tok; estimated non-text 16321
  tok; generated 105 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Title: Two sailboats moored on a river with a wooded bank
>
> Description: Two sailing boats with orange masts are moored on calm water in
> front of a steep, densely wooded riverbank. A small orange buoy floats in
> the foreground near the stern of the nearest vessel.
>
> Keywords: Sailboat, Boat, River, Estuary, Moored, Mast, Rigging, Buoy,
> Trees, Foliage, Bank, Water, Wooden, Green, Nature, Outdoors, Landscape,
> Peaceful
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-qwen2-vl-2b-instruct-4bit"></a>

### ✅ mlx-community/Qwen2-VL-2B-Instruct-4bit

- _Recommendation:_ use with caveats; review verdict: context budget
- _Owner:_ likely owner `model`
- _Next step:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 97% and the output stays weak under that load.
- _Key signals:_ output/prompt=0.26%; nontext prompt burden=97%; missing
  sections: keywords; missing terms: Bird, Boating, Buoy, Bushes, Coast
- _Score:_ 🟠 D (48/100); Δ-1; Limited novel information
- _Review focus:_ watchlist (metadata borrowing, missing sections)
- _Timing:_ Load 0.58s; Gen 74.34s; Total 75.30s
- _Throughput:_ Prompt 229 TPS (16,740 tok); Gen 221 TPS (44 tok)
- _Tokens:_ prompt 16740 tok; estimated text 435 tok; estimated non-text 16305
  tok; generated 44 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> -001.jpg
> &#45; Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe
> &#45; Description: Two sailing boats moored on a river with trees behind on the
> bank
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context_budget
- Missing sections (keywords)
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-qwen36-27b-mxfp8"></a>

### ✅ mlx-community/Qwen3.6-27B-mxfp8

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Bushes, Coast, Deben Estuary
- _Score:_ 🏆 A (84/100); Δ+35; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 3.07s; Gen 84.84s; Total 88.30s
- _Throughput:_ Prompt 216 TPS (16,756 tok); Gen 17.6 TPS (114 tok)
- _Tokens:_ prompt 16756 tok; estimated text 435 tok; estimated non-text 16321
  tok; generated 114 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Two sailing boats moored on a river
>
> &#8203;Description:
> Two sailing boats are moored on calm water, with the foreground vessel
> featuring a black hull, wooden transom, and tall orange masts. A dense
> backdrop of green trees and foliage lines the riverbank, while a buoy floats
> in the foreground.
>
> &#8203;Keywords:
> Boat, Sailing, Moored, River, Water, Trees, Foliage, Nature, Landscape,
> Mast, Rigging, Hull, Buoy, Outdoors, Green, Wood, Vessel, Forest
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-qwen35-27b-4bit"></a>

### ✅ mlx-community/Qwen3.5-27B-4bit

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Bushes, Coast, Deben Estuary
- _Score:_ 🏆 A (85/100); Δ+36; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 2.24s; Gen 87.54s; Total 90.29s
- _Throughput:_ Prompt 204 TPS (16,756 tok); Gen 29.6 TPS (130 tok)
- _Tokens:_ prompt 16756 tok; estimated text 435 tok; estimated non-text 16321
  tok; generated 130 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Two sailing boats moored on river with forested bank
>
> &#8203;Description:
> Two sailing boats with wooden masts and furled sails are moored on calm
> water in front of a densely vegetated riverbank. The foreground boat has a
> dark hull with a polished wooden stern displaying the name 'SIOLA', while a
> second boat with a light hull is positioned slightly behind it.
>
> &#8203;Keywords:
> Sailing boats, River, Forest, Moored, Masts, Rigging, Foliage, Water, Buoy,
> Trees, Nature, Landscape, Outdoors, Hull, Sails, Vegetation, Estuary, Wood
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

---

<a id="model-mlx-community-qwen35-27b-mxfp8"></a>

### ✅ mlx-community/Qwen3.5-27B-mxfp8

- _Recommendation:_ avoid for now; review verdict: semantic mismatch
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model limitation for this prompt; trusted hint
  coverage is still weak.
- _Key signals:_ missing terms: Bird, Boating, Bushes, Coast, Deben Estuary
- _Score:_ 🏆 A (90/100); Δ+41; None identified
- _Review focus:_ strong candidate for first-pass review
- _Timing:_ Load 3.20s; Gen 93.38s; Total 97.00s
- _Throughput:_ Prompt 198 TPS (16,756 tok); Gen 16.7 TPS (135 tok)
- _Tokens:_ prompt 16756 tok; estimated text 435 tok; estimated non-text 16321
  tok; generated 135 tok; requested max 500 tok; stop reason completed

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Title:
> Two Moored Sailboats on a River with Forested Bank
>
> &#8203;Description:
> Two sailboats with wooden masts are moored on calm water, positioned in
> front of a densely vegetated riverbank. The foreground boat features a dark
> hull with a wooden transom displaying the name "SIOLA" and "ESSC", while the
> boat behind has a light green hull and red sail covers.
>
> &#8203;Keywords:
> Sailboats, Moored, River, Forest, Trees, Masts, Rigging, Hull, Water,
> Nature, Landscape, Greenery, Buoy, Outdoors, Vessels, Foliage, Estuary, Wood
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

---

<!-- markdownlint-enable MD033 MD034 -->
