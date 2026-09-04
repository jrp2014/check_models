# Model Output Gallery

Generated on: 2026-09-04 19:33:10 BST

These models serve many purposes; this run probes exactly one narrow task:
producing catalogue metadata for a single photograph from the assisted-lane
prompt and whatever context it supplies — here, camera-recorded capture
context plus draft descriptive hints previously produced by a more capable
model. Results say nothing about a model's fitness for other uses.

Complete per-model evidence artifact with image metadata, the source prompt, a
facts-only chooser, and full generated or crash output for every attempted
model.

## Reference Image

![Reference image](assets/source-image.jpg)

## Current-run Chooser

Current-run usability and captured resource facts only. Total time is end-to-end; throughput covers generation only and requires at least 16 generated tokens. Prefill/first is first-token latency when captured; Prompt tok is the full rendered prompt including image tokens, which drives prefill cost. For cross-attention architectures the token count reflects the tokenised text burden only, not total vision prefill compute.

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                                   | Usability             | Total s | Gen TPS    | Prefill/first s | Peak GB | Prompt tok | Gen tok | Observations                     |
|-------------------------------------------------------------------------------------------------------------------------|-----------------------|---------|------------|-----------------|---------|------------|---------|----------------------------------|
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-4bit)            | `usable`              | 6.66s   | 134 tok/s  | 0.84            | 19      | 1,605      | 406     | none                             |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `usable`              | 9.40s   | 32.3 tok/s | 1.02            | 18      | 2,597      | 162     | none                             |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `usable`              | 6.17s   | 34.4 tok/s | 0.76            | 17      | 2,086      | 88      | none                             |
| [`mlx-community/LFM2.5-VL-3B-OptiQ-4bit`](#model-mlx-community-lfm25-vl-3b-optiq-4bit)                                  | `usable`              | 2.91s   | 213 tok/s  | 0.46            | 4.0     | 2,086      | 83      | none                             |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `usable`              | 7.13s   | 67.4 tok/s | 1.97            | 13      | 2,918      | 180     | none                             |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `usable`              | 3.79s   | 190 tok/s  | 0.96            | 7.8     | 2,917      | 139     | none                             |
| [`mlx-community/Muse-Glimmer-30B-OptiQ-4bit`](#model-mlx-community-muse-glimmer-30b-optiq-4bit)                         | `usable`              | 50.46s  | 25.2 tok/s | 6.91            | 25      | 4,388      | 992     | none                             |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `usable`              | 29.95s  | 89.8 tok/s | 17.67           | 8.4     | 16,522     | 907     | none                             |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `usable`              | 59.95s  | 116 tok/s  | 54.70           | 24      | 16,536     | 143     | none                             |
| [`mlx-community/Qwen3.8-27B-4bit`](#model-mlx-community-qwen38-27b-4bit)                                                | `usable`              | 79.21s  | 30.1 tok/s | 71.23           | 21      | 16,536     | 131     | none                             |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `usable`              | 4.67s   | 128 tok/s  | 0.34            | 16      | 574        | 105     | none                             |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `usable`              | 8.05s   | 26.5 tok/s | 0.95            | 19      | 574        | 90      | none                             |
| [`mlx-community/granite-4.0-3b-vision-4bit`](#model-mlx-community-granite-40-3b-vision-4bit)                            | `usable`              | 2.81s   | 178 tok/s  | 0.40            | 4.7     | 1,371      | 88      | none                             |
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `usable with caveats` | 1.69s   | 484 tok/s  | 0.12            | 1.9     | 2,094      | 76      | title/keyword constraints failed |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `usable with caveats` | 9.90s   | 30.8 tok/s | 2.83            | 23      | 2,385      | 119     | title/keyword constraints failed |
| [`mlx-community/GLM-4.6V-Flash-4bit`](#model-mlx-community-glm-46v-flash-4bit)                                          | `usable with caveats` | 8.57s   | 78.6 tok/s | 5.12            | 8.7     | 6,430      | 92      | control tokens visible           |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `usable with caveats` | 20.53s  | 44.2 tok/s | 10.54           | 78      | 6,430      | 108     | control tokens visible           |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-8bit`](#model-mlx-community-kimi-vl-a3b-thinking-2506-8bit)                   | `usable with caveats` | 18.49s  | 67.2 tok/s | 0.70            | 20      | 1,307      | 971     | role tokens visible              |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `usable with caveats` | 2.72s   | 189 tok/s  | 0.34            | 4.0     | 2,094      | 119     | title/keyword constraints failed |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `usable with caveats` | 7.30s   | 64.8 tok/s | 2.00            | 13      | 2,918      | 180     | title/keyword constraints failed |
| [`mlx-community/Molmo2-8B-4bit`](#model-mlx-community-molmo2-8b-4bit)                                                   | `usable with caveats` | 4.99s   | 73.6 tok/s | 0.74            | 8.5     | 1,516      | 132     | title/keyword constraints failed |
| [`mlx-community/North-Micro-Vision-Instruct-4bit`](#model-mlx-community-north-micro-vision-instruct-4bit)               | `usable with caveats` | 6.31s   | 229 tok/s  | 3.34            | 3.9     | 4,062      | 152     | title/keyword constraints failed |
| [`mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit`](#model-mlx-community-ornith-15-35b-a3b-optiq-4bit)                      | `usable with caveats` | 5.61s   | 112 tok/s  | 0.62            | 24      | 1,262      | 135     | title/keyword constraints failed |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `usable with caveats` | 4.45s   | 58.3 tok/s | 0.29            | 9.5     | 1,103      | 128     | title/keyword constraints failed |
| [`mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit`](#model-mlx-community-qwen3-vl-30b-a3b-instruct-4bit)                   | `usable with caveats` | 58.82s  | 88.4 tok/s | 53.94           | 22      | 16,520     | 133     | title/keyword constraints failed |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `usable with caveats` | 60.70s  | 91.5 tok/s | 56.39           | 10.0    | 16,536     | 132     | title/keyword constraints failed |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `usable with caveats` | 3.00s   | 127 tok/s  | 0.40            | 5.4     | 1,401      | 84      | title/keyword constraints failed |
| [`mlx-community/Step-3.7-Flash-oQ2e`](#model-mlx-community-step-37-flash-oq2e)                                          | `usable with caveats` | 22.92s  | 46.7 tok/s | 12.92           | 70      | 3,464      | 114     | title/keyword constraints failed |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `usable with caveats` | 22.04s  | 59.1 tok/s | 17.12           | 13      | 16,531     | 136     | title/keyword constraints failed |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `usable with caveats` | 6.24s   | 66.4 tok/s | 0.32            | 28      | 570        | 95      | title/keyword constraints failed |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `usable with caveats` | 8.55s   | 29.9 tok/s | 0.95            | 17      | 569        | 128     | title/keyword constraints failed |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `usable with caveats` | 7.34s   | 39.4 tok/s | 1.82            | 16      | 3,111      | 110     | title/keyword constraints failed |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Resource Highlights

Fastest clean completion: `mlx-community/LFM2.5-VL-3B-OptiQ-4bit` at 213 tok/s

Average clean-completion throughput: 97.3 tok/s (indicative only: tokenizers and architectures differ across models)

Lowest peak memory among clean completions: `mlx-community/LFM2.5-VL-3B-OptiQ-4bit` at 4.0 GB

## Avoid for This Run

No unusable or not-evaluated models in this run.

## Output at a Glance

The first 280 characters of each model's actual output (or failure evidence for crashes), in chooser order. Complete exact evidence for every model follows below.

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                                   | Usability             | Output preview                                                                                                                                                                                                                                                                                                |
|-------------------------------------------------------------------------------------------------------------------------|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-4bit)            | `usable`              | Alright, here's what I'm thinking. I need to generate some metadata for this image, and it's got to be spot-on. First things first, I'll break down the request. It's all about creating a title, a description, and some keywords. The image itself is a classic English scene – a s...                      |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `usable`              | Title: Winchester City Mill and River Itchen in Hampshire, England.<br>Description: The historic Winchester City Mill, situated over the River Itchen in Winchester, Hampshire, England, is a popular tourist attraction. The mill features a distinctive red brick exterior with a stee...                   |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `usable`              | Title: Historic Mill by River Itchen<br>Description: Pedestrians walk and relax near the historic Winchester City Mill, featuring a charming stone bridge and clay-tiled roof under a blue sky.<br>Keywords: Winchester City Mill, historic watermill, River Itchen, Hampshire, England, Na...                |
| [`mlx-community/LFM2.5-VL-3B-OptiQ-4bit`](#model-mlx-community-lfm25-vl-3b-optiq-4bit)                                  | `usable`              | Title: Historic Mill Over River Itchen<br><br>Description: Visitors walk along a pathway beside the historic Winchester City Mill, an ancient watermill with distinctive clay tiles and chimneys, set against a blue sky with clouds.<br><br>Keywords: National Trust, Hampshire, England, Mill,...           |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `usable`              | Title:<br>**Winchester City Mill: Historic Watermill by the River Itchen**<br><br>Description:<br>A group of visitors, including two women walking along the riverside path, explore the historic **Winchester City Mill**, a National Trust-listed watermill with a distinctive **stone and bric...          |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `usable`              | Title:<br>Historic Winchester Watermill by the River Itchen<br><br>Description:<br>A picturesque scene captures the Winchester City Mill, a traditional brick-and-stone watermill with clay-tiled roofs, flowing over the River Itchen under bright blue skies. Visitors stroll along the riversi...          |
| [`mlx-community/Muse-Glimmer-30B-OptiQ-4bit`](#model-mlx-community-muse-glimmer-30b-optiq-4bit)                         | `usable`              | Title:<br>Historic Winchester City Mill Over River Itchen<br><br>Description:<br>Captured on 31 August 2026, the historic Winchester City Mill, a brick and flint watermill with a clay-tile roof and chimney, stands over the River Itchen in Winchester, Hampshire, England. Two girls walk alo...          |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `usable`              | Got it, let's tackle this step by step. First, the title needs to be 5-10 words. The image shows a historic watermill, so "Winchester City Mill, River Itchen, Hampshire" is key. Let's make it concise: "Winchester City Mill, River Itchen, Hampshire, England"<br><br>Next, the descript...                |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `usable`              | Title: Historic Watermill Over River Itchen<br><br>Description: Captured on August 31, 2026, this image depicts the historic Winchester City Mill, an ancient structure featuring a weathered clay-tiled roof and flint-and-brick walls, situated over the River Itchen in Hampshire. Under...                |
| [`mlx-community/Qwen3.8-27B-4bit`](#model-mlx-community-qwen38-27b-4bit)                                                | `usable`              | Title:<br>Winchester City Mill Spanning the River Itchen<br><br>Description:<br>The historic Winchester City Mill, a brick and stone structure with a clay-tiled roof, stands over the River Itchen in Hampshire, England. Pedestrians walk along the bridge pathway in front of the building whi...          |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `usable`              | Title: People walking by historic Winchester City Mill<br>Description: Two women walk along a stone pathway in front of the historic brick and flint Winchester City Mill over the River Itchen. The scene captures the ancient watermill's tiled roof and stone architecture under a br...                   |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `usable`              | Title: Visitors at the historic Winchester City Mill, Hampshire<br>Description: Visitors walk along a pathway and relax in front of the ancient Winchester City Mill, situated over the River Itchen in Hampshire, England, under a bright blue sky.<br>Keywords: Winchester City Mill, Riv...                |
| [`mlx-community/granite-4.0-3b-vision-4bit`](#model-mlx-community-granite-40-3b-vision-4bit)                            | `usable`              | Title: "Historic Mill by the River Itchen"<br>Description: "Visitors stroll along the pathway in front of the historic Winchester City Mill, enjoying the serene setting by the River Itchen."<br>Keywords: Historic Mill, Winchester City Mill, River Itchen, Pedestrians, Elderly woman,...                 |
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `usable_with_caveats` | Title:<br>Historic Winchester City Mill<br><br>Description:<br>A historic stone-built watermill stands over the River Itchen in Winchester, England, with visitors walking along a stone bridge and enjoying the scenic view.<br><br>Keywords:<br>Historic, watermill, Winchester, England, River Itchen,...  |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `usable_with_caveats` | Title:<br>Visitors at Winchester City Mill<br><br>Description:<br>Visitors walk along the pathway and relax in front of the historic Winchester City Mill, an ancient watermill situated over the River Itchen in Winchester, Hampshire, England, under a blue sky.<br><br>Keywords:<br>Winchester City Mi... |
| [`mlx-community/GLM-4.6V-Flash-4bit`](#model-mlx-community-glm-46v-flash-4bit)                                          | `usable_with_caveats` | <\|begin_of_box\|>Title: Winchester City Mill by River Itchen<br>Description: A historic brick and stone watermill with a tiled roof and chimney sits over the River Itchen, with people walking along the riverbank pathway under a blue sky with clouds.<br>Keywords: Winchester City Mill,...              |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `usable_with_caveats` | <\|begin_of_box\|>Title: Historic Winchester City Mill by River Itchen<br><br>Description: Two girls walk along the pathway in front of the historic Winchester City Mill, a brick and stone watermill with clay tiles, situated over the River Itchen in Hampshire, England, under a blue sk...              |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-8bit`](#model-mlx-community-kimi-vl-a3b-thinking-2506-8bit)                   | `usable_with_caveats` | ◁think▷Okay, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. The capture date is 2026-08-31 15:02:20 UTC+01:00, but I shouldn't claim it's visible. The descriptive hints mention Winchester City Mill,...                       |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `usable_with_caveats` | Title: Historic Winchester City Mill<br><br>Description: Visitors stroll along the pathway and relax in front of the ancient Winchester City Mill, an iconic watermill situated over the River Itchen in Winchester, Hampshire, England. The mill's red brick structure and clay tile roof...                 |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `usable_with_caveats` | **Title:**<br>*Winchester City Mill: Riverside Pedestrians &amp; Historic Charm*<br><br>**Description:**<br>A vibrant scene captures visitors strolling and relaxing beside the iconic **Winchester City Mill** on **August 31, 2026 (UTC+01:00)**, its ancient stone arches framing a rushing **Rive...      |
| [`mlx-community/Molmo2-8B-4bit`](#model-mlx-community-molmo2-8b-4bit)                                                   | `usable_with_caveats` | Title: Historic Mill and River Scene with Pedestrians<br><br>Description: The image captures a historic watermill in Winchester, Hampshire, England, with a stone wall and clay-tiled roof. Two women walk along a pathway above the River Itchen, while an elderly woman sits nearby. The...                 |
| [`mlx-community/North-Micro-Vision-Instruct-4bit`](#model-mlx-community-north-micro-vision-instruct-4bit)               | `usable_with_caveats` | Title: Historic Winchester City Mill<br><br>Description: Visitors walk along the pathway and relax in front of the historic Winchester City Mill, an ancient watermill situated over the River Itchen in Winchester, Hampshire, England. The mill features a distinctive brick and stone fa...                |
| [`mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit`](#model-mlx-community-ornith-15-35b-a3b-optiq-4bit)                      | `usable_with_caveats` | Title: Visitors at Winchester City Mill Over River Itchen<br><br>Description: People stroll along the stone bridge and relax in front of the historic Winchester City Mill, an ancient watermill built over the River Itchen. The brick and flake-stone structure with its weathered clay-t...                |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `usable_with_caveats` | Title: Historic Winchester City Mill Overlooking River Itchen<br><br>Description: The historic Winchester City Mill, a National Trust property, stands majestically over the River Itchen in Hampshire, England. Visitors are seen enjoying the scenic view and the historic architecture o...                |
| [`mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit`](#model-mlx-community-qwen3-vl-30b-a3b-instruct-4bit)                   | `usable_with_caveats` | Title: Historic mill by the River Itchen with pedestrians<br><br>Description: On a bright day with a blue sky and scattered clouds, pedestrians walk along a stone bridge over the River Itchen in Winchester, Hampshire, England. The scene features the historic Winchester City Mill, a...                 |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `usable_with_caveats` | Title:<br>Visitors walk across bridge beside historic Winchester City Mill on River Itchen.<br><br>Description:<br>On a bright day in late August 2026, pedestrians stroll along a stone bridge adjacent to the ancient Winchester City Mill, a National Trust-listed watermill perched over the...           |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `usable_with_caveats` | Title: Winchester City Mill<br>Description: A historic watermill situated over the River Itchen in Winchester, Hampshire, England.<br>Keywords: Adobe Stock, Any Vision, Arch, Blue sky, Chimney, Clay tiles, Elderly woman, England, Girls, Hampshire, Mill, National Trust, Pedestrians,...                 |
| [`mlx-community/Step-3.7-Flash-oQ2e`](#model-mlx-community-step-37-flash-oq2e)                                          | `usable_with_caveats` | Title:<br>Winchester City Mill Over the River Itchen<br><br>Description:<br>Visitors walk along the pathway and relax in front of the historic Winchester City Mill, an ancient watermill situated over the River Itchen in Winchester, Hampshire, England, under a bright blue sky with white cl...          |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `usable_with_caveats` | Title:<br>Winchester City Mill: A Historic Watermill<br><br>Description:<br>Visitors stroll along the pathway beneath a historic brick mill over the River Itchen, with an elderly woman relaxing on the stone wall, under a bright blue sky. The mill's weathered clay-tiled roof and stone chim...          |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `usable_with_caveats` | Title: Historic Winchester City Mill on the River Itchen<br>Description: Visitors walk along a stone pathway in front of the ancient brick and stone watermill over flowing rapids in Hampshire. The historic building features a clay tiled roof and arches over the water.<br><br>Keywords:...              |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `usable_with_caveats` | Title: Winchester City Mill, Hampshire: People on the Pathway<br><br>Description: Captured on 31st August 2026, this image shows Winchester City Mill, a historic brick and stone building spanning the River Itchen; pedestrians walk along the pathway in bright daylight, with rapids vi...                |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `usable_with_caveats` | Title:<br>Historic Winchester Mill on River Itchen, Hampshire<br><br>Description:<br>Two individuals walk along the pathway in front of the historic Winchester City Mill, an ancient watermill situated over the River Itchen in Winchester, Hampshire, England, under a bright blue sky.<br><br>Keywo...    |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Run Stamps

- `mlx-vlm`: `0.7.0rc0`
- `mlx`: `0.32.3.dev20260904+b6368984b`
- `mlx-lm`: `0.32.0`
- `transformers`: `5.16.1`
- `tokenizers`: `0.23.2`
- `huggingface-hub`: `1.30.0`
- *Python Version:* 3.14.7
- *OS:* Darwin 25.6.0
- *macOS Version:* 26.6.2
- *GPU/Chip:* Apple M5 Max
- *MLX Device:* Apple M5 Max
- *GPU Architecture:* applegpu_g17s
- *RAM:* 128.0 GB
- *Recommended Working Set:* 108 GB
- *Fused Attention:* Available

## Image Metadata

- *Description:* Visitors walk along the pathway and relax in front of the
  historic Winchester City Mill, an ancient watermill situated over the River
  Itchen in Winchester, Hampshire, England.
- *Keywords:* Adobe Stock, Any Vision, Arch, Blue sky, Chimney, Clay tiles,
  Elderly woman, England, Girls, Hampshire, Mill, National Trust, Pedestrians,
  People, Rapids, River Itchen, Riverbank, Scenery, Sitting, Stone wall,
  Strolling, Tiled Roof, Trees, United Kingdom, Water, Waterway, Wheelchair,
  Winchester, Winchester City Mill, architecture, brick architecture, brick
  building, brick wall, bridge, british, checkerboard wall, culture, cumulus
  clouds, day, english heritage, exterior, facade, flint, flowing water,
  foliage, footbridge, great britain, heritage, historic building, history,
  landmark, lantern, medieval, old, outdoors, quaint, railing, river, scenic,
  senior, sluice, stone bridge, stream, summer, tourism, tourist attraction,
  traditional, travel destination, vintage, walking, watermill
- *Date:* 2026-08-31 15:02:20 UTC+01:00
- *Time:* 15:02:20

## Prompt

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Create British-English catalogue metadata from the image and supplied
> context.
>
> Treat any capture date/time and GPS as authoritative facts, but do not claim
> they are visible. Descriptive hints may be incomplete or wrong: retain
> details supported by the image, correct conflicts, and add important visible
> details. Prefer image evidence when a hint conflicts, and omit uncertain
> details.
>
> Context: Authoritative context:
> &#45; Capture date/time: 2026-08-31 15:02:20 UTC+01:00
>
> &#8203;Descriptive hints:
> &#45; Description hint: Visitors walk along the pathway and relax in front of
> the historic Winchester City Mill, an ancient watermill situated over the
> River Itchen in Winchester, Hampshire, England.
> &#45; Keyword hints: Adobe Stock, Any Vision, Arch, Blue sky, Chimney, Clay
> tiles, Elderly woman, England, Girls, Hampshire, Mill, National Trust,
> Pedestrians, People, Rapids, River Itchen, Riverbank, Scenery, Sitting,
> Stone wall
>
> &#8203;Write:
> &#45; a concrete 5-10-word title;
> &#45; a 1-2-sentence factual description combining relevant context with the
> main visible subject, setting, action, lighting, and distinctive details;
> &#45; 10-18 unique, comma-separated keywords covering relevant context and
> visible details.
>
> &#8203;Return exactly these three sections and nothing else:
> &#8203;Title:
> &#8203;Description:
> &#8203;Keywords:
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

## Complete Per-model Evidence

Complete generated or crash evidence for every attempted model.

<a id="model-mlx-community-ernie-45-vl-28b-a3b-thinking-4bit"></a>

### mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit

<details>
<summary>Complete evidence: mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type ernie4_5_moe_vl)
- *Model load time:* 1.66s
- *Generation time:* 4.56s
- *Total time:* 6.66s
- *Input validation time:* 0.413
- *Prompt preparation time:* 0.0254
- *First-token latency:* 0.841
- *Cleanup time:* 0.0576
- *Prompt tokens:* 1,605
- *Generation tokens:* 406
- *Total tokens:* 2,011
- *Prompt throughput (raw):* 1,908 tok/s
- *Generation throughput (raw):* 134 tok/s
- *Peak memory:* 19
- *Active memory:* 17
- *Cache memory:* 0.14
- *Model-load active memory:* 17.4
- *Post-cleanup active memory:* 17.4
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1481
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl.Ernie4_5_VLProcessor
- *Tokenizer:* mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl.Ernie4_5_VLTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 846ea5576854468f25af3767d769b0805b1b08b6
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--ERNIE-4.5-VL-28B-A3B-Thinking-4bit/snapshots/846ea5576854468f25af3767d769b0805b1b08b6
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 800,
  "thinking_end_token": "&lt;/think&gt;", "thinking_start_token":
  "&lt;think&gt;", "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Alright, here's what I'm thinking. I need to generate some metadata for this image, and it's got to be spot-on. First things first, I'll break down the request. It's all about creating a title, a description, and some keywords. The image itself is a classic English scene – a stone house with a tiled roof, a river, and some people. The hints are a bit of a mess, but I'll use them as a starting point.

I'm seeing a stone house with a tiled roof, a river, and some people. The hints mention a watermill, but the image doesn't show one. I'll stick to what I see. The house is old, with a tiled roof and a stone wall. The river is flowing, and there are people walking along a path. The sky is bright.

The title needs to be short and sweet. I'll go with "Stone House by the River." The description needs to be factual and concise. I'll mention the stone house, the river, the people, and the sunny day. The keywords are crucial. I'll include things like "stone house," "river," "people," "England," and "tiled roof." I'll make sure to include "stone wall" and "tiled roof" as well.

I'll double-check everything. The title is good. The description is accurate. The keywords are relevant. I'm ready to go.
&lt;/think&gt;

Title: Stone House by the River

Description: A sunny day scene showing a stone house with a tiled roof by a flowing river, with people walking along a path nearby.

Keywords: stone house, river, people, England, tiled roof, stone wall, riverbank, scenery, walking, sunny day, Hampshire, ancient, old, building, water, flowing water
</pre>

<details>
<summary>Exact raw output</summary>

```text
Alright, here's what I'm thinking. I need to generate some metadata for this image, and it's got to be spot-on. First things first, I'll break down the request. It's all about creating a title, a description, and some keywords. The image itself is a classic English scene – a stone house with a tiled roof, a river, and some people. The hints are a bit of a mess, but I'll use them as a starting point.

I'm seeing a stone house with a tiled roof, a river, and some people. The hints mention a watermill, but the image doesn't show one. I'll stick to what I see. The house is old, with a tiled roof and a stone wall. The river is flowing, and there are people walking along a path. The sky is bright.

The title needs to be short and sweet. I'll go with "Stone House by the River." The description needs to be factual and concise. I'll mention the stone house, the river, the people, and the sunny day. The keywords are crucial. I'll include things like "stone house," "river," "people," "England," and "tiled roof." I'll make sure to include "stone wall" and "tiled roof" as well.

I'll double-check everything. The title is good. The description is accurate. The keywords are relevant. I'm ready to go.
</think>

Title: Stone House by the River

Description: A sunny day scene showing a stone house with a tiled roof by a flowing river, with people walking along a path nearby.

Keywords: stone house, river, people, England, tiled roof, stone wall, riverbank, scenery, walking, sunny day, Hampshire, ancient, old, building, water, flowing water
```

</details>

</details>

---

<a id="model-mlx-community-idefics3-8b-llama3-bf16"></a>

### mlx-community/Idefics3-8B-Llama3-bf16

<details>
<summary>Complete evidence: mlx-community/Idefics3-8B-Llama3-bf16</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type idefics3)
- *Model load time:* 2.05s
- *Generation time:* 6.88s
- *Total time:* 9.40s
- *Input validation time:* 0.439
- *Prompt preparation time:* 0.0269
- *First-token latency:* 1.02
- *Cleanup time:* 0.064
- *Prompt tokens:* 2,597
- *Generation tokens:* 162
- *Total tokens:* 2,759
- *Prompt throughput (raw):* 2,544 tok/s
- *Generation throughput (raw):* 32.3 tok/s
- *Peak memory:* 18
- *Active memory:* 17
- *Cache memory:* 0.43
- *Model-load active memory:* 16.9
- *Post-cleanup active memory:* 16.9
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1328
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.idefics3.processing_idefics3.Idefics3Processor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 8c2a30c48864f3251701b7bde40f601d25535098
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Idefics3-8B-Llama3-bf16/snapshots/8c2a30c48864f3251701b7bde40f601d25535098
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|end_of_text\|>

*Readable output:*

<pre class="model-output-readable">
 Title: Winchester City Mill and River Itchen in Hampshire, England.
Description: The historic Winchester City Mill, situated over the River Itchen in Winchester, Hampshire, England, is a popular tourist attraction. The mill features a distinctive red brick exterior with a steeply pitched roof and a prominent chimney. In the foreground, two girls are walking along a pathway that crosses over the river, with a small waterfall below. The mill is surrounded by lush greenery and offers a picturesque view of the river and the surrounding countryside. The sky above is partly cloudy, with patches of blue sky visible.
Keywords: Winchester City Mill, Hampshire, England, River Itchen, historic mill, red brick, steeply pitched roof, prominent chimney, pathway, waterfall, lush greenery, picturesque view, partly cloudy sky.
</pre>

</details>

---

<a id="model-mlx-community-internvl3-8b-bf16"></a>

### mlx-community/InternVL3-8B-bf16

<details>
<summary>Complete evidence: mlx-community/InternVL3-8B-bf16</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type internvl_chat)
- *Model load time:* 1.63s
- *Generation time:* 4.08s
- *Total time:* 6.17s
- *Input validation time:* 0.44
- *Prompt preparation time:* 0.0202
- *First-token latency:* 0.761
- *Cleanup time:* 0.0649
- *Prompt tokens:* 2,086
- *Generation tokens:* 88
- *Total tokens:* 2,174
- *Prompt throughput (raw):* 2,739 tok/s
- *Generation throughput (raw):* 34.4 tok/s
- *Peak memory:* 17
- *Active memory:* 16
- *Cache memory:* 0.17
- *Model-load active memory:* 15.9
- *Post-cleanup active memory:* 15.9
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1296
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.internvl_chat.processor.InternVLChatProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* e0df3dd79263467173214b67ef6d6a0cc5a475fd
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--InternVL3-8B-bf16/snapshots/e0df3dd79263467173214b67ef6d6a0cc5a475fd
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Historic Mill by River Itchen
Description: Pedestrians walk and relax near the historic Winchester City Mill, featuring a charming stone bridge and clay-tiled roof under a blue sky.
Keywords: Winchester City Mill, historic watermill, River Itchen, Hampshire, England, National Trust, clay tiles, stone bridge, blue sky, pedestrians, relaxation, stone wall, riverbank, scenic, chimney, elderly woman, girls
</pre>

</details>

---

<a id="model-mlx-community-lfm25-vl-3b-optiq-4bit"></a>

### mlx-community/LFM2.5-VL-3B-OptiQ-4bit

<details>
<summary>Complete evidence: mlx-community/LFM2.5-VL-3B-OptiQ-4bit</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type lfm2_vl)
- *Model load time:* 0.99s
- *Generation time:* 1.45s
- *Total time:* 2.91s
- *Input validation time:* 0.437
- *Prompt preparation time:* 0.0265
- *First-token latency:* 0.465
- *Cleanup time:* 0.0621
- *Prompt tokens:* 2,086
- *Generation tokens:* 83
- *Total tokens:* 2,169
- *Prompt throughput (raw):* 4,489 tok/s
- *Generation throughput (raw):* 213 tok/s
- *Peak memory:* 4.0
- *Active memory:* 2.8
- *Cache memory:* 0.08
- *Model-load active memory:* 2.81
- *Post-cleanup active memory:* 2.81
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1342
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* transformers.models.lfm2_vl.processing_lfm2_vl.Lfm2VlProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 12c5ae49304158b0a133fcea9ba4486a6d6c8cad
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--LFM2.5-VL-3B-OptiQ-4bit/snapshots/12c5ae49304158b0a133fcea9ba4486a6d6c8cad
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Historic Mill Over River Itchen

Description: Visitors walk along a pathway beside the historic Winchester City Mill, an ancient watermill with distinctive clay tiles and chimneys, set against a blue sky with clouds.

Keywords: National Trust, Hampshire, England, Mill, River Itchen, Pedestrians, People, Rapids, Riverbank, Scenery, Sitting, Adobe Stock, Any Vision
</pre>

</details>

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-mxfp4"></a>

### mlx-community/Ministral-3-14B-Instruct-2512-mxfp4

<details>
<summary>Complete evidence: mlx-community/Ministral-3-14B-Instruct-2512-mxfp4</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type mistral3)
- *Model load time:* 1.39s
- *Generation time:* 5.27s
- *Total time:* 7.13s
- *Input validation time:* 0.44
- *Prompt preparation time:* 0.0263
- *First-token latency:* 1.97
- *Cleanup time:* 0.063
- *Prompt tokens:* 2,918
- *Generation tokens:* 180
- *Total tokens:* 3,098
- *Prompt throughput (raw):* 1,480 tok/s
- *Generation throughput (raw):* 67.4 tok/s
- *Peak memory:* 13
- *Active memory:* 8.0
- *Cache memory:* 0.60
- *Model-load active memory:* 8
- *Post-cleanup active memory:* 8
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 3729
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.mistral3.processing_mistral3.Mistral3Processor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 7c992876448fc5426841a85c6eb951b056fe38d2
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Ministral-3-14B-Instruct-2512-mxfp4/snapshots/7c992876448fc5426841a85c6eb951b056fe38d2
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Title:
**Winchester City Mill: Historic Watermill by the River Itchen**

Description:
A group of visitors, including two women walking along the riverside path, explore the historic **Winchester City Mill**, a National Trust-listed watermill with a distinctive **stone and brick facade** and **clay-tiled roof**, set against the flowing **River Itchen** under a bright **blue-sky backdrop** with scattered clouds. The mill’s **stone arch bridge** and **rapid water flow** enhance the scenic charm, captured on a clear summer afternoon.

Keywords:
Winchester City Mill, River Itchen, National Trust, historic watermill, stone bridge, clay-tiled roof, brick and stone facade, riverside path, pedestrians, scenic riverbank, blue sky, England, Hampshire, summer landscape, elderly visitors, leisure walk, architectural heritage
</pre>

</details>

---

<a id="model-mlx-community-ministral-3-3b-instruct-2512-4bit"></a>

### mlx-community/Ministral-3-3B-Instruct-2512-4bit

<details>
<summary>Complete evidence: mlx-community/Ministral-3-3B-Instruct-2512-4bit</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type mistral3)
- *Model load time:* 1.02s
- *Generation time:* 2.30s
- *Total time:* 3.79s
- *Input validation time:* 0.437
- *Prompt preparation time:* 0.0249
- *First-token latency:* 0.965
- *Cleanup time:* 0.0629
- *Prompt tokens:* 2,917
- *Generation tokens:* 139
- *Total tokens:* 3,056
- *Prompt throughput (raw):* 3,024 tok/s
- *Generation throughput (raw):* 190 tok/s
- *Peak memory:* 7.8
- *Active memory:* 2.7
- *Cache memory:* 0.38
- *Model-load active memory:* 2.75
- *Post-cleanup active memory:* 2.75
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 3728
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.mistral3.processing_mistral3.Mistral3Processor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* a962dcb09eee4169c890e544c9eb938f1113fdee
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Ministral-3-3B-Instruct-2512-4bit/snapshots/a962dcb09eee4169c890e544c9eb938f1113fdee
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Title:
Historic Winchester Watermill by the River Itchen

Description:
A picturesque scene captures the Winchester City Mill, a traditional brick-and-stone watermill with clay-tiled roofs, flowing over the River Itchen under bright blue skies. Visitors stroll along the riverside path, with a stone bridge and waterfall adding charm to the quaint, well-preserved setting.

Keywords:
Winchester City Mill, brick-and-stone structure, clay tiles, River Itchen, stone bridge, flowing water, blue sky, historic watermill, pathway, traditional architecture, Hampshire, England, National Trust heritage, scenic riverside, visitors walking, waterfall rapids
</pre>

</details>

---

<a id="model-mlx-community-muse-glimmer-30b-optiq-4bit"></a>

### mlx-community/Muse-Glimmer-30B-OptiQ-4bit

<details>
<summary>Complete evidence: mlx-community/Muse-Glimmer-30B-OptiQ-4bit</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type muse_glimmer)
- *Model load time:* 2.94s
- *Generation time:* 47.04s
- *Total time:* 50.46s
- *Input validation time:* 0.437
- *Prompt preparation time:* 0.0371
- *First-token latency:* 6.91
- *Cleanup time:* 0.0648
- *Prompt tokens:* 4,388
- *Generation tokens:* 992
- *Total tokens:* 5,380
- *Prompt throughput (raw):* 635 tok/s
- *Generation throughput (raw):* 25.2 tok/s
- *Peak memory:* 25
- *Active memory:* 24
- *Cache memory:* 0.30
- *Model-load active memory:* 23.9
- *Post-cleanup active memory:* 23.9
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1529
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.muse_glimmer.processing_muse_glimmer.MuseGlimmerProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* b4a74fa6001f1eca3b23eeeb702ffad2773a218f
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Muse-Glimmer-30B-OptiQ-4bit/snapshots/b4a74fa6001f1eca3b23eeeb702ffad2773a218f
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|end_of_text\|>

*Readable output:*

<pre class="model-output-readable">
Title:
Historic Winchester City Mill Over River Itchen

Description:
Captured on 31 August 2026, the historic Winchester City Mill, a brick and flint watermill with a clay-tile roof and chimney, stands over the River Itchen in Winchester, Hampshire, England. Two girls walk along the riverside walkway beside the mill’s arched water outlets and stone lower storey, while an elderly woman sits on the patio under a partly cloudy blue sky.

Keywords:
Winchester City Mill, River Itchen, Hampshire, England, historic watermill, brick and flint facade, clay tile roof, chimney, arched water outlet, rapids, riverbank, pedestrians, girls walking, elderly woman sitting, partly cloudy sky, National Trust
</pre>

</details>

---

<a id="model-mlx-community-qwen3-vl-2b-thinking-bf16"></a>

### mlx-community/Qwen3-VL-2B-Thinking-bf16

<details>
<summary>Complete evidence: mlx-community/Qwen3-VL-2B-Thinking-bf16</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Model load time:* 0.78s
- *Generation time:* 28.70s
- *Total time:* 29.95s
- *Input validation time:* 0.437
- *Prompt preparation time:* 0.0295
- *First-token latency:* 17.7
- *Cleanup time:* 0.0664
- *Prompt tokens:* 16,522
- *Generation tokens:* 907
- *Total tokens:* 17,429
- *Prompt throughput (raw):* 935 tok/s
- *Generation throughput (raw):* 89.8 tok/s
- *Peak memory:* 8.4
- *Active memory:* 4.3
- *Cache memory:* 2.7
- *Model-load active memory:* 4.26
- *Post-cleanup active memory:* 4.26
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1371
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* c325e5ea14c215bb08fa0d668c81fa2581f9050b
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3-VL-2B-Thinking-bf16/snapshots/c325e5ea14c215bb08fa0d668c81fa2581f9050b
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 800,
  "thinking_end_token": "&lt;/think&gt;", "thinking_start_token":
  "&lt;think&gt;", "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Got it, let's tackle this step by step. First, the title needs to be 5-10 words. The image shows a historic watermill, so "Winchester City Mill, River Itchen, Hampshire" is key. Let's make it concise: "Winchester City Mill, River Itchen, Hampshire, England"

Next, the description. It has to combine context with visible subject, setting, action, lighting, details. The context is capture date, but we don't need to include that. The main subject is the mill, people walking, the river, the historic building. The lighting is bright, blue sky. Distinctive details: brick and stone walls, chimney, people (elderly woman, girls), river rapids. So: "Visitors walk along the historic Winchester City Mill's pathway, with blue sky, brick and stone walls, and river rapids flowing beneath the mill."

Wait, check the context: the description hint says "Visitors walk along the pathway and relax in front of the historic Winchester City Mill..." So the description should include that. Let's adjust: "Visitors walk along the historic Winchester City Mill's pathway, with blue sky, brick and stone walls, and river rapids flowing beneath the mill, where an elderly woman sits and girls stroll."

Now keywords. Need 10-18 unique, comma-separated. From context: England, Hampshire, Mill, National Trust (since it's a historic site, likely National Trust), River Itchen, Winchester City Mill, Pedestrians, People, Stone wall, Brick wall, Chimney, Blue sky, Rapids, Riverbank, Scenery, Elderly woman, Girls. Let's list them: England, Hampshire, Winchester City Mill, National Trust, River Itchen, Pedestrians, People, Stone wall, Brick wall, Chimney, Blue sky, Rapids, Riverbank, Scenery, Elderly woman, Girls. Wait, check if all are visible. The image has a chimney (yes, the red brick chimney), brick and stone walls (yes), blue sky (yes), river rapids (yes), riverbank (yes), scenery (yes), elderly woman (yes), girls (yes), National Trust (since it's a historic site, likely managed by National Trust), Winchester City Mill (yes). Let's make sure they are unique. Let's count: England, Hampshire, Winchester City Mill, National Trust, River Itchen, Pedestrians, People, Stone wall, Brick wall, Chimney, Blue sky, Rapids, Riverbank, Scenery, Elderly woman, Girls. That's 15. Let's check if any are redundant. "People" is included, "Pedestrians" is a subset. "Scenery" is okay. "Riverbank" is visible. "Chimney" is there. "Brick wall" and "Stone wall" are both present. "Blue sky" is visible. "Rapids" are the river's flow. "National Trust" is a key context. "Winchester City Mill" is the main subject. "Elderly woman" and "Girls" are visible. Let's check if all are in the image. The image has the mill, people walking, elderly woman sitting, girls walking, chimney, brick and stone walls, blue sky, river rapids. So the keywords should include those. Let's list them: England, Hampshire, Winchester City Mill, National Trust, River Itchen, Pedestrians, People, Stone wall, Brick wall, Chimney, Blue sky, Rapids, Riverbank, Scenery, Elderly woman, Girls. That's 15. Let's check if any are missing. The context says "National Trust" is a keyword hint, so we should include that. The description mentions "historic Winchester City Mill", so "Winchester City Mill" is a keyword
&lt;/think&gt;

Title: Winchester City Mill, River Itchen, Hampshire, England

Description: Visitors walk along the historic Winchester City Mill's pathway, with blue sky, brick and stone walls, and river rapids flowing beneath the mill, where an elderly woman sits and girls stroll.

Keywords: England, Hampshire, Winchester City Mill, National Trust, River Itchen, Pedestrians, People, Stone wall, Brick wall, Chimney, Blue sky, Rapids, Riverbank, Scenery, Elderly woman, Girls
</pre>

<details>
<summary>Exact raw output</summary>

```text
Got it, let's tackle this step by step. First, the title needs to be 5-10 words. The image shows a historic watermill, so "Winchester City Mill, River Itchen, Hampshire" is key. Let's make it concise: "Winchester City Mill, River Itchen, Hampshire, England"

Next, the description. It has to combine context with visible subject, setting, action, lighting, details. The context is capture date, but we don't need to include that. The main subject is the mill, people walking, the river, the historic building. The lighting is bright, blue sky. Distinctive details: brick and stone walls, chimney, people (elderly woman, girls), river rapids. So: "Visitors walk along the historic Winchester City Mill's pathway, with blue sky, brick and stone walls, and river rapids flowing beneath the mill."

Wait, check the context: the description hint says "Visitors walk along the pathway and relax in front of the historic Winchester City Mill..." So the description should include that. Let's adjust: "Visitors walk along the historic Winchester City Mill's pathway, with blue sky, brick and stone walls, and river rapids flowing beneath the mill, where an elderly woman sits and girls stroll."

Now keywords. Need 10-18 unique, comma-separated. From context: England, Hampshire, Mill, National Trust (since it's a historic site, likely National Trust), River Itchen, Winchester City Mill, Pedestrians, People, Stone wall, Brick wall, Chimney, Blue sky, Rapids, Riverbank, Scenery, Elderly woman, Girls. Let's list them: England, Hampshire, Winchester City Mill, National Trust, River Itchen, Pedestrians, People, Stone wall, Brick wall, Chimney, Blue sky, Rapids, Riverbank, Scenery, Elderly woman, Girls. Wait, check if all are visible. The image has a chimney (yes, the red brick chimney), brick and stone walls (yes), blue sky (yes), river rapids (yes), riverbank (yes), scenery (yes), elderly woman (yes), girls (yes), National Trust (since it's a historic site, likely managed by National Trust), Winchester City Mill (yes). Let's make sure they are unique. Let's count: England, Hampshire, Winchester City Mill, National Trust, River Itchen, Pedestrians, People, Stone wall, Brick wall, Chimney, Blue sky, Rapids, Riverbank, Scenery, Elderly woman, Girls. That's 15. Let's check if any are redundant. "People" is included, "Pedestrians" is a subset. "Scenery" is okay. "Riverbank" is visible. "Chimney" is there. "Brick wall" and "Stone wall" are both present. "Blue sky" is visible. "Rapids" are the river's flow. "National Trust" is a key context. "Winchester City Mill" is the main subject. "Elderly woman" and "Girls" are visible. Let's check if all are in the image. The image has the mill, people walking, elderly woman sitting, girls walking, chimney, brick and stone walls, blue sky, river rapids. So the keywords should include those. Let's list them: England, Hampshire, Winchester City Mill, National Trust, River Itchen, Pedestrians, People, Stone wall, Brick wall, Chimney, Blue sky, Rapids, Riverbank, Scenery, Elderly woman, Girls. That's 15. Let's check if any are missing. The context says "National Trust" is a keyword hint, so we should include that. The description mentions "historic Winchester City Mill", so "Winchester City Mill" is a keyword
</think>

Title: Winchester City Mill, River Itchen, Hampshire, England

Description: Visitors walk along the historic Winchester City Mill's pathway, with blue sky, brick and stone walls, and river rapids flowing beneath the mill, where an elderly woman sits and girls stroll.

Keywords: England, Hampshire, Winchester City Mill, National Trust, River Itchen, Pedestrians, People, Stone wall, Brick wall, Chimney, Blue sky, Rapids, Riverbank, Scenery, Elderly woman, Girls
```

</details>

</details>

---

<a id="model-mlx-community-qwen35-35b-a3b-4bit"></a>

### mlx-community/Qwen3.5-35B-A3B-4bit

<details>
<summary>Complete evidence: mlx-community/Qwen3.5-35B-A3B-4bit</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5_moe)
- *Model load time:* 2.52s
- *Generation time:* 56.96s
- *Total time:* 59.95s
- *Input validation time:* 0.434
- *Prompt preparation time:* 0.0344
- *First-token latency:* 54.7
- *Cleanup time:* 0.0645
- *Prompt tokens:* 16,536
- *Generation tokens:* 143
- *Total tokens:* 16,679
- *Prompt throughput (raw):* 302 tok/s
- *Generation throughput (raw):* 116 tok/s
- *Peak memory:* 24
- *Active memory:* 20
- *Cache memory:* 0.89
- *Model-load active memory:* 20.4
- *Post-cleanup active memory:* 20.4
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1382
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 1e20fd8d42056f870933bf98ca6211024744f7ec
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Historic Watermill Over River Itchen

Description: Captured on August 31, 2026, this image depicts the historic Winchester City Mill, an ancient structure featuring a weathered clay-tiled roof and flint-and-brick walls, situated over the River Itchen in Hampshire. Under a bright blue sky with scattered clouds, pedestrians cross a stone bridge in the foreground while an elderly woman sits nearby.

Keywords: Winchester City Mill, River Itchen, Hampshire, England, Clay tiles, Flint wall, Stone bridge, Blue sky, Clouds, Pedestrians, Elderly woman, Watermill, National Trust, Scenery, Architecture, Riverbank, Rapids, People
</pre>

</details>

---

<a id="model-mlx-community-qwen38-27b-4bit"></a>

### mlx-community/Qwen3.8-27B-4bit

<details>
<summary>Complete evidence: mlx-community/Qwen3.8-27B-4bit</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5)
- *Model load time:* 2.14s
- *Generation time:* 76.59s
- *Total time:* 79.21s
- *Input validation time:* 0.436
- *Prompt preparation time:* 0.0369
- *First-token latency:* 71.2
- *Cleanup time:* 0.0685
- *Prompt tokens:* 16,536
- *Generation tokens:* 131
- *Total tokens:* 16,667
- *Prompt throughput (raw):* 232 tok/s
- *Generation throughput (raw):* 30.1 tok/s
- *Peak memory:* 21
- *Active memory:* 16
- *Cache memory:* 1.9
- *Model-load active memory:* 16.1
- *Post-cleanup active memory:* 16.1
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1382
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 3e6447f082e89cc7f0bc6e5441afd38dfce760ff
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/3e6447f082e89cc7f0bc6e5441afd38dfce760ff
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
Winchester City Mill Spanning the River Itchen

Description:
The historic Winchester City Mill, a brick and stone structure with a clay-tiled roof, stands over the River Itchen in Hampshire, England. Pedestrians walk along the bridge pathway in front of the building while an elderly woman sits nearby, set against a backdrop of blue sky and white clouds.

Keywords:
Winchester City Mill, River Itchen, Hampshire, England, Watermill, Brick building, Clay tiles, Stone bridge, Pedestrians, Elderly woman, Blue sky, Clouds, Historic architecture, Water, Scenery, National Trust
</pre>

</details>

---

<a id="model-mlx-community-gemma-4-26b-a4b-it-4bit"></a>

### mlx-community/gemma-4-26b-a4b-it-4bit

<details>
<summary>Complete evidence: mlx-community/gemma-4-26b-a4b-it-4bit</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type gemma4)
- *Model load time:* 2.36s
- *Generation time:* 1.84s
- *Total time:* 4.67s
- *Input validation time:* 0.43
- *Prompt preparation time:* 0.0386
- *First-token latency:* 0.34
- *Cleanup time:* 0.0594
- *Prompt tokens:* 574
- *Generation tokens:* 105
- *Total tokens:* 679
- *Prompt throughput (raw):* 1,691 tok/s
- *Generation throughput (raw):* 128 tok/s
- *Peak memory:* 16
- *Active memory:* 15
- *Cache memory:* 0.20
- *Model-load active memory:* 15.3
- *Post-cleanup active memory:* 15.3
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1345
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.gemma4.processing_gemma4.Gemma4Processor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 0d77464eeb233a2da68ebf9d7dc4edaac7db956d
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--gemma-4-26b-a4b-it-4bit/snapshots/0d77464eeb233a2da68ebf9d7dc4edaac7db956d
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
Title: People walking by historic Winchester City Mill
Description: Two women walk along a stone pathway in front of the historic brick and flint Winchester City Mill over the River Itchen. The scene captures the ancient watermill's tiled roof and stone architecture under a bright, cloudy sky.
Keywords: Winchester, River Itchen, Hampshire, England, historic mill, watermill, stone wall, clay tiles, pedestrians, riverbank, architecture, brickwork, blue sky, scenic, National Trust, river, stone bridge
</pre>

</details>

---

<a id="model-mlx-community-gemma-4-31b-it-4bit"></a>

### mlx-community/gemma-4-31b-it-4bit

<details>
<summary>Complete evidence: mlx-community/gemma-4-31b-it-4bit</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type gemma4)
- *Model load time:* 2.54s
- *Generation time:* 5.04s
- *Total time:* 8.05s
- *Input validation time:* 0.429
- *Prompt preparation time:* 0.0388
- *First-token latency:* 0.952
- *Cleanup time:* 0.062
- *Prompt tokens:* 574
- *Generation tokens:* 90
- *Total tokens:* 664
- *Prompt throughput (raw):* 603 tok/s
- *Generation throughput (raw):* 26.5 tok/s
- *Peak memory:* 19
- *Active memory:* 18
- *Cache memory:* 0.76
- *Model-load active memory:* 18.4
- *Post-cleanup active memory:* 18.4
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1345
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.gemma4.processing_gemma4.Gemma4Processor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 696d436c404745a59f30e4939a658162b0a9e57f
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--gemma-4-31b-it-4bit/snapshots/696d436c404745a59f30e4939a658162b0a9e57f
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
Title: Visitors at the historic Winchester City Mill, Hampshire
Description: Visitors walk along a pathway and relax in front of the ancient Winchester City Mill, situated over the River Itchen in Hampshire, England, under a bright blue sky.
Keywords: Winchester City Mill, River Itchen, Hampshire, England, watermill, National Trust, historic building, riverbank, pedestrians, clay tiles, stone wall, arch, river, scenery, summer
</pre>

</details>

---

<a id="model-mlx-community-granite-40-3b-vision-4bit"></a>

### mlx-community/granite-4.0-3b-vision-4bit

<details>
<summary>Complete evidence: mlx-community/granite-4.0-3b-vision-4bit</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type granite4_vision)
- *Model load time:* 0.62s
- *Generation time:* 1.72s
- *Total time:* 2.81s
- *Input validation time:* 0.43
- *Prompt preparation time:* 0.027
- *First-token latency:* 0.404
- *Cleanup time:* 0.0619
- *Prompt tokens:* 1,371
- *Generation tokens:* 88
- *Total tokens:* 1,459
- *Prompt throughput (raw):* 3,394 tok/s
- *Generation throughput (raw):* 178 tok/s
- *Peak memory:* 4.7
- *Active memory:* 3.0
- *Cache memory:* 0.22
- *Model-load active memory:* 3.02
- *Post-cleanup active memory:* 3.02
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1515
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.granite4_vision.processing_granite4_vision.Granite4VisionProcessor
- *Tokenizer:* transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 70fe1d89f42c71b5635cd945ed304bed438fa1e3
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--granite-4.0-3b-vision-4bit/snapshots/70fe1d89f42c71b5635cd945ed304bed438fa1e3
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|end_of_text\|>

*Readable output:*

<pre class="model-output-readable">
Title: "Historic Mill by the River Itchen"
Description: "Visitors stroll along the pathway in front of the historic Winchester City Mill, enjoying the serene setting by the River Itchen."
Keywords: Historic Mill, Winchester City Mill, River Itchen, Pedestrians, Elderly woman, Girls, Stone wall, Chimney, Blue sky, Rapids, Scenery, National Trust, Adobe Stock, Any Vision
</pre>

</details>

---

<a id="model-liquidai-lfm25-vl-450m-mlx-bf16"></a>

### LiquidAI/LFM2.5-VL-450M-MLX-bf16

<details>
<summary>Complete evidence: LiquidAI/LFM2.5-VL-450M-MLX-bf16</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Title has 4 words (requested 5-10); Duplicate keywords:
  bridge
- *Arch supported by installed mlx-vlm:* yes (model_type lfm2-vl via lfm2_vl)
- *Model load time:* 0.41s
- *Generation time:* 0.84s
- *Total time:* 1.69s
- *Input validation time:* 0.42
- *Prompt preparation time:* 0.0222
- *First-token latency:* 0.118
- *Cleanup time:* 0.0579
- *Prompt tokens:* 2,094
- *Generation tokens:* 76
- *Total tokens:* 2,170
- *Prompt throughput (raw):* 17,780 tok/s
- *Generation throughput (raw):* 484 tok/s
- *Peak memory:* 1.9
- *Active memory:* 0.90
- *Cache memory:* 0.08
- *Model-load active memory:* 0.897
- *Post-cleanup active memory:* 0.898
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1342
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* transformers.models.lfm2_vl.processing_lfm2_vl.Lfm2VlProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* ed71acdae0799c98cb450787d049d1343b65cd52
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--LiquidAI--LFM2.5-VL-450M-MLX-bf16/snapshots/ed71acdae0799c98cb450787d049d1343b65cd52
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
Historic Winchester City Mill

Description:
A historic stone-built watermill stands over the River Itchen in Winchester, England, with visitors walking along a stone bridge and enjoying the scenic view.

Keywords:
Historic, watermill, Winchester, England, River Itchen, bridge, visitors, stone, bridge, people, scenic, architecture, natural beauty
</pre>

</details>

---

<a id="model-mlx-community-devstral-small-2-24b-instruct-2512-5bit"></a>

### mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

<details>
<summary>Complete evidence: mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Keyword list has 23 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type mistral3)
- *Model load time:* 2.13s
- *Generation time:* 7.33s
- *Total time:* 9.90s
- *Input validation time:* 0.412
- *Prompt preparation time:* 0.0246
- *First-token latency:* 2.83
- *Cleanup time:* 0.061
- *Prompt tokens:* 2,385
- *Generation tokens:* 119
- *Total tokens:* 2,504
- *Prompt throughput (raw):* 842 tok/s
- *Generation throughput (raw):* 30.8 tok/s
- *Peak memory:* 23
- *Active memory:* 18
- *Cache memory:* 0.47
- *Model-load active memory:* 18
- *Post-cleanup active memory:* 18
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1291
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.mistral3.processing_mistral3.Mistral3Processor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 0a970d20ad7da57b0e2accc35c5b8628f5d02063
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Devstral-Small-2-24B-Instruct-2512-5bit/snapshots/0a970d20ad7da57b0e2accc35c5b8628f5d02063
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Title:
Visitors at Winchester City Mill

Description:
Visitors walk along the pathway and relax in front of the historic Winchester City Mill, an ancient watermill situated over the River Itchen in Winchester, Hampshire, England, under a blue sky.

Keywords:
Winchester City Mill, River Itchen, Hampshire, England, watermill, historic building, brick and stone architecture, blue sky, visitors, pathway, riverbank, rapids, chimney, clay tiles, stone wall, National Trust, scenic view, pedestrians, people, girls, elderly woman, sitting, relaxing
</pre>

</details>

---

<a id="model-mlx-community-glm-46v-flash-4bit"></a>

### mlx-community/GLM-4.6V-Flash-4bit

<details>
<summary>Complete evidence: mlx-community/GLM-4.6V-Flash-4bit</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Unrecognised model control tokens remain visible
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v)
- *Model load time:* 1.25s
- *Generation time:* 6.87s
- *Total time:* 8.57s
- *Input validation time:* 0.417
- *Prompt preparation time:* 0.0249
- *First-token latency:* 5.12
- *Cleanup time:* 0.0634
- *Prompt tokens:* 6,430
- *Generation tokens:* 92
- *Total tokens:* 6,522
- *Prompt throughput (raw):* 1,257 tok/s
- *Generation throughput (raw):* 78.6 tok/s
- *Peak memory:* 8.7
- *Active memory:* 7.1
- *Cache memory:* 0.46
- *Model-load active memory:* 7.07
- *Post-cleanup active memory:* 7.07
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1372
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.glm4v.processing.Glm46VProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* bd7b20686e8cbf22ad221a3de8399cdc5a2a7e0e
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--GLM-4.6V-Flash-4bit/snapshots/bd7b20686e8cbf22ad221a3de8399cdc5a2a7e0e
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|endoftext\|>

*Readable output:*

<pre class="model-output-readable">
&lt;|begin_of_box|&gt;Title: Winchester City Mill by River Itchen
Description: A historic brick and stone watermill with a tiled roof and chimney sits over the River Itchen, with people walking along the riverbank pathway under a blue sky with clouds.
Keywords: Winchester City Mill, River Itchen, historic watermill, brick building, stone wall, riverbank, pedestrians, blue sky, clouds, chimney, tiled roof, Hampshire, England, National Trust.&lt;|end_of_box|&gt;
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|begin_of_box|>Title: Winchester City Mill by River Itchen
Description: A historic brick and stone watermill with a tiled roof and chimney sits over the River Itchen, with people walking along the riverbank pathway under a blue sky with clouds.
Keywords: Winchester City Mill, River Itchen, historic watermill, brick building, stone wall, riverbank, pedestrians, blue sky, clouds, chimney, tiled roof, Hampshire, England, National Trust.<|end_of_box|>
```

</details>

</details>

---

<a id="model-mlx-community-glm-46v-nvfp4"></a>

### mlx-community/GLM-4.6V-nvfp4

<details>
<summary>Complete evidence: mlx-community/GLM-4.6V-nvfp4</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Unrecognised model control tokens remain visible
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v_moe)
- *Model load time:* 5.78s
- *Generation time:* 14.27s
- *Total time:* 20.53s
- *Input validation time:* 0.44
- *Prompt preparation time:* 0.0318
- *First-token latency:* 10.5
- *Cleanup time:* 0.0659
- *Prompt tokens:* 6,430
- *Generation tokens:* 108
- *Total tokens:* 6,538
- *Prompt throughput (raw):* 610 tok/s
- *Generation throughput (raw):* 44.2 tok/s
- *Peak memory:* 78
- *Active memory:* 62
- *Cache memory:* 1.4
- *Model-load active memory:* 61.9
- *Post-cleanup active memory:* 61.9
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1372
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.glm4v_moe.processing.Glm46VMoEProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 2da6855d4e28a0e61c84543262074bc17ac27d6e
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--GLM-4.6V-nvfp4/snapshots/2da6855d4e28a0e61c84543262074bc17ac27d6e
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|endoftext\|>

*Readable output:*

<pre class="model-output-readable">
&lt;|begin_of_box|&gt;Title: Historic Winchester City Mill by River Itchen

Description: Two girls walk along the pathway in front of the historic Winchester City Mill, a brick and stone watermill with clay tiles, situated over the River Itchen in Hampshire, England, under a blue sky with white clouds.

Keywords: Winchester City Mill, River Itchen, Hampshire, England, watermill, brick, stone, clay tiles, chimney, blue sky, girls, pedestrians, elderly woman, National Trust, riverbank, scenery, sitting, rapids&lt;|end_of_box|&gt;
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|begin_of_box|>Title: Historic Winchester City Mill by River Itchen

Description: Two girls walk along the pathway in front of the historic Winchester City Mill, a brick and stone watermill with clay tiles, situated over the River Itchen in Hampshire, England, under a blue sky with white clouds.

Keywords: Winchester City Mill, River Itchen, Hampshire, England, watermill, brick, stone, clay tiles, chimney, blue sky, girls, pedestrians, elderly woman, National Trust, riverbank, scenery, sitting, rapids<|end_of_box|>
```

</details>

</details>

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-2506-8bit"></a>

### mlx-community/Kimi-VL-A3B-Thinking-2506-8bit

<details>
<summary>Complete evidence: mlx-community/Kimi-VL-A3B-Thinking-2506-8bit</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Conversation-role control tokens remain visible
- *Arch supported by installed mlx-vlm:* yes (model_type kimi_vl)
- *Model load time:* 2.15s
- *Generation time:* 15.88s
- *Total time:* 18.49s
- *Input validation time:* 0.436
- *Prompt preparation time:* 0.0209
- *First-token latency:* 0.702
- *Cleanup time:* 0.0646
- *Prompt tokens:* 1,307
- *Generation tokens:* 971
- *Total tokens:* 2,278
- *Prompt throughput (raw):* 1,862 tok/s
- *Generation throughput (raw):* 67.2 tok/s
- *Peak memory:* 20
- *Active memory:* 18
- *Cache memory:* 1.5
- *Model-load active memory:* 17.8
- *Post-cleanup active memory:* 17.8
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1478
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.kimi_vl.processing_kimi_vl.KimiVLProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* e5abbe34cbfabd829fafd0362856e5b468d19f85
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Kimi-VL-A3B-Thinking-2506-8bit/snapshots/e5abbe34cbfabd829fafd0362856e5b468d19f85
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* [EOS]

*Readable output:*

<pre class="model-output-readable">
◁think▷Okay, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. The capture date is 2026-08-31 15:02:20 UTC+01:00, but I shouldn't claim it's visible. The descriptive hints mention Winchester City Mill, River Itchen, Hampshire, England, people walking, elderly woman, girls, Adobe Stock, Any Vision, Arch, Blue sky, Chimney, Clay tiles, Stone wall, Pedestrians, Rapids, Scenery, Sitting, National Trust.

Looking at the image: there's a historic building (Winchester City Mill) with a brick and stone facade, a tiled roof, a chimney, a stone bridge over a river with rapids, people walking on the bridge, an elderly woman sitting, and a blue sky. The keywords from the hints are relevant. Need to make sure the title is 5-10 words, description 1-2 sentences, and 10-18 keywords.

Title: "Historic Winchester City Mill with River Itchen and Pedestrians" – that's 8 words.

Description: "A historic brick and stone building, the Winchester City Mill, stands beside the River Itchen in Hampshire, England, with pedestrians walking on a stone bridge and an elderly woman sitting nearby under a blue sky." That's 2 sentences, covers the main elements.

Keywords: Winchester City Mill, River Itchen, Hampshire, England, Pedestrians, Elderly woman, Blue sky, Adobe Stock, Any Vision, Arch, Chimney, Stone wall, Clay tiles, Rapids, Scenery, Sitting, National Trust. Let me count: 15 keywords. Check if all hints are covered. The hints include National Trust, which is part of the context but not in the image. Wait, the image might not show National Trust explicitly. Hmm, the context says it's a National Trust property. Should I include it? The user said to retain details supported by the image. Since the image doesn't show the National Trust sign, maybe omit. So remove National Trust. Also, check if all visible elements are covered. The image has a stone wall, clay tiles on the roof, chimney, arch (the bridge arch), blue sky, rapids in the river, sitting (elderly woman), pedestrians (two people walking), and the building itself. So keywords: Winchester City Mill, River Itchen, Hampshire, England, Pedestrians, Elderly woman, Blue sky, Adobe Stock, Any Vision, Arch, Chimney, Stone wall, Clay tiles, Rapids, Scenery. That's 13. Maybe add "Stone bridge" since the bridge is visible. So adjust: Winchester City Mill, River Itchen, Hampshire, England, Pedestrians, Elderly woman, Blue sky, Adobe Stock, Any Vision, Arch, Chimney, Stone wall, Clay tiles, Rapids, Scenery, Stone bridge. Now 14. Maybe "Stone bridge" is better than "Arch" as it's more descriptive. Alternatively, keep Arch. Wait, the bridge has an arch, so maybe Arch is okay. Let me check the image: the bridge has an arch structure. So maybe keep Arch. Let me recount: Winchester City Mill, River Itchen, Hampshire, England, Pedestrians, Elderly woman, Blue sky, Adobe Stock, Any Vision, Arch, Chimney, Stone wall, Clay tiles, Rapids, Scenery. That's 14. Maybe add "National Trust" even if not visible? The context says it's a National Trust property. The user said to retain details supported by the image. Since the image doesn't show the National Trust sign, maybe omit. So final keywords: Winchester City Mill, River Itchen, Hampshire, England, Pedestrians, Elderly woman, Blue sky, Adobe Stock, Any Vision, Arch, Chimney, Stone wall, Clay tiles, Rapids, Scenery. That's 13. Wait, the context mentions "National Trust" as a keyword hint. Should I include it? The user said to retain details supported by the image. Since the image doesn't show the National Trust, maybe omit. So proceed with the visible elements.◁/think▷Title: Historic Winchester City Mill with River Itchen and Pedestrians
Description: A historic brick and stone building, the Winchester City Mill, stands beside the River Itchen in Hampshire, England, with pedestrians walking on a stone bridge and an elderly woman sitting nearby under a blue sky.
Keywords: Winchester City Mill, River Itchen, Hampshire, England, Pedestrians, Elderly woman, Blue sky, Adobe Stock, Any Vision, Arch, Chimney, Stone wall, Clay tiles, Rapids, Scenery&lt;|im_assistant|&gt;
</pre>

<details>
<summary>Exact raw output</summary>

```text
◁think▷Okay, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. The capture date is 2026-08-31 15:02:20 UTC+01:00, but I shouldn't claim it's visible. The descriptive hints mention Winchester City Mill, River Itchen, Hampshire, England, people walking, elderly woman, girls, Adobe Stock, Any Vision, Arch, Blue sky, Chimney, Clay tiles, Stone wall, Pedestrians, Rapids, Scenery, Sitting, National Trust.

Looking at the image: there's a historic building (Winchester City Mill) with a brick and stone facade, a tiled roof, a chimney, a stone bridge over a river with rapids, people walking on the bridge, an elderly woman sitting, and a blue sky. The keywords from the hints are relevant. Need to make sure the title is 5-10 words, description 1-2 sentences, and 10-18 keywords.

Title: "Historic Winchester City Mill with River Itchen and Pedestrians" – that's 8 words. 

Description: "A historic brick and stone building, the Winchester City Mill, stands beside the River Itchen in Hampshire, England, with pedestrians walking on a stone bridge and an elderly woman sitting nearby under a blue sky." That's 2 sentences, covers the main elements.

Keywords: Winchester City Mill, River Itchen, Hampshire, England, Pedestrians, Elderly woman, Blue sky, Adobe Stock, Any Vision, Arch, Chimney, Stone wall, Clay tiles, Rapids, Scenery, Sitting, National Trust. Let me count: 15 keywords. Check if all hints are covered. The hints include National Trust, which is part of the context but not in the image. Wait, the image might not show National Trust explicitly. Hmm, the context says it's a National Trust property. Should I include it? The user said to retain details supported by the image. Since the image doesn't show the National Trust sign, maybe omit. So remove National Trust. Also, check if all visible elements are covered. The image has a stone wall, clay tiles on the roof, chimney, arch (the bridge arch), blue sky, rapids in the river, sitting (elderly woman), pedestrians (two people walking), and the building itself. So keywords: Winchester City Mill, River Itchen, Hampshire, England, Pedestrians, Elderly woman, Blue sky, Adobe Stock, Any Vision, Arch, Chimney, Stone wall, Clay tiles, Rapids, Scenery. That's 13. Maybe add "Stone bridge" since the bridge is visible. So adjust: Winchester City Mill, River Itchen, Hampshire, England, Pedestrians, Elderly woman, Blue sky, Adobe Stock, Any Vision, Arch, Chimney, Stone wall, Clay tiles, Rapids, Scenery, Stone bridge. Now 14. Maybe "Stone bridge" is better than "Arch" as it's more descriptive. Alternatively, keep Arch. Wait, the bridge has an arch, so maybe Arch is okay. Let me check the image: the bridge has an arch structure. So maybe keep Arch. Let me recount: Winchester City Mill, River Itchen, Hampshire, England, Pedestrians, Elderly woman, Blue sky, Adobe Stock, Any Vision, Arch, Chimney, Stone wall, Clay tiles, Rapids, Scenery. That's 14. Maybe add "National Trust" even if not visible? The context says it's a National Trust property. The user said to retain details supported by the image. Since the image doesn't show the National Trust sign, maybe omit. So final keywords: Winchester City Mill, River Itchen, Hampshire, England, Pedestrians, Elderly woman, Blue sky, Adobe Stock, Any Vision, Arch, Chimney, Stone wall, Clay tiles, Rapids, Scenery. That's 13. Wait, the context mentions "National Trust" as a keyword hint. Should I include it? The user said to retain details supported by the image. Since the image doesn't show the National Trust, maybe omit. So proceed with the visible elements.◁/think▷Title: Historic Winchester City Mill with River Itchen and Pedestrians  
Description: A historic brick and stone building, the Winchester City Mill, stands beside the River Itchen in Hampshire, England, with pedestrians walking on a stone bridge and an elderly woman sitting nearby under a blue sky.  
Keywords: Winchester City Mill, River Itchen, Hampshire, England, Pedestrians, Elderly woman, Blue sky, Adobe Stock, Any Vision, Arch, Chimney, Stone wall, Clay tiles, Rapids, Scenery<|im_assistant|>
```

</details>

</details>

---

<a id="model-mlx-community-lfm25-vl-16b-bf16"></a>

### mlx-community/LFM2.5-VL-1.6B-bf16

<details>
<summary>Complete evidence: mlx-community/LFM2.5-VL-1.6B-bf16</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Title has 4 words (requested 5-10)
- *Arch supported by installed mlx-vlm:* yes (model_type lfm2_vl)
- *Model load time:* 0.70s
- *Generation time:* 1.57s
- *Total time:* 2.72s
- *Input validation time:* 0.436
- *Prompt preparation time:* 0.0212
- *First-token latency:* 0.341
- *Cleanup time:* 0.0611
- *Prompt tokens:* 2,094
- *Generation tokens:* 119
- *Total tokens:* 2,213
- *Prompt throughput (raw):* 6,146 tok/s
- *Generation throughput (raw):* 189 tok/s
- *Peak memory:* 4.0
- *Active memory:* 3.2
- *Cache memory:* 0.06
- *Model-load active memory:* 3.19
- *Post-cleanup active memory:* 3.19
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1342
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* transformers.models.lfm2_vl.processing_lfm2_vl.Lfm2VlProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 16a710cf8afca206ff16a95a4ad6fe657f876ce1
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--LFM2.5-VL-1.6B-bf16/snapshots/16a710cf8afca206ff16a95a4ad6fe657f876ce1
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Historic Winchester City Mill

Description: Visitors stroll along the pathway and relax in front of the ancient Winchester City Mill, an iconic watermill situated over the River Itchen in Winchester, Hampshire, England. The mill's red brick structure and clay tile roof stand out against the blue sky, while the stone wall and water rapids add to the picturesque scenery.

Keywords: Winchester City Mill, historic watermill, River Itchen, Hampshire, England, stone wall, clay tiles, blue sky, visitors, mill, water rapids, scenery, pedestrians
</pre>

</details>

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-nvfp4"></a>

### mlx-community/Ministral-3-14B-Instruct-2512-nvfp4

<details>
<summary>Complete evidence: mlx-community/Ministral-3-14B-Instruct-2512-nvfp4</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Keyword list has 19 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type mistral3)
- *Model load time:* 1.44s
- *Generation time:* 5.39s
- *Total time:* 7.30s
- *Input validation time:* 0.437
- *Prompt preparation time:* 0.0263
- *First-token latency:* 2
- *Cleanup time:* 0.0647
- *Prompt tokens:* 2,918
- *Generation tokens:* 180
- *Total tokens:* 3,098
- *Prompt throughput (raw):* 1,456 tok/s
- *Generation throughput (raw):* 64.8 tok/s
- *Peak memory:* 13
- *Active memory:* 8.4
- *Cache memory:* 0.60
- *Model-load active memory:* 8.42
- *Post-cleanup active memory:* 8.43
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 3729
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.mistral3.processing_mistral3.Mistral3Processor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 28777b889d841a86369c736175cb77258c8134b2
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Ministral-3-14B-Instruct-2512-nvfp4/snapshots/28777b889d841a86369c736175cb77258c8134b2
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
**Title:**
*Winchester City Mill: Riverside Pedestrians &amp; Historic Charm*

**Description:**
A vibrant scene captures visitors strolling and relaxing beside the iconic **Winchester City Mill** on **August 31, 2026 (UTC+01:00)**, its ancient stone arches framing a rushing **River Itchen** under a bright blue sky. Distinctive features include clay-tiled roofs, a chimney, elderly pedestrians, and scenic riverbank details.

**Keywords:**
Winchester City Mill, River Itchen, Hampshire, England, National Trust, historic watermill, pedestrians, elderly visitors, riverbank, rushing rapids, clay-tiled roof, stone arches, blue sky, scenic pathway, 2026-08-31, riverside charm, traditional architecture, pedestrian bridge, scenic relaxation
</pre>

<details>
<summary>Exact raw output</summary>

```text
**Title:**
*Winchester City Mill: Riverside Pedestrians & Historic Charm*

**Description:**
A vibrant scene captures visitors strolling and relaxing beside the iconic **Winchester City Mill** on **August 31, 2026 (UTC+01:00)**, its ancient stone arches framing a rushing **River Itchen** under a bright blue sky. Distinctive features include clay-tiled roofs, a chimney, elderly pedestrians, and scenic riverbank details.

**Keywords:**
Winchester City Mill, River Itchen, Hampshire, England, National Trust, historic watermill, pedestrians, elderly visitors, riverbank, rushing rapids, clay-tiled roof, stone arches, blue sky, scenic pathway, 2026-08-31, riverside charm, traditional architecture, pedestrian bridge, scenic relaxation
```

</details>

</details>

---

<a id="model-mlx-community-molmo2-8b-4bit"></a>

### mlx-community/Molmo2-8B-4bit

<details>
<summary>Complete evidence: mlx-community/Molmo2-8B-4bit</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Keyword list has 19 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type molmo2)
- *Model load time:* 1.28s
- *Generation time:* 3.24s
- *Total time:* 4.99s
- *Input validation time:* 0.436
- *Prompt preparation time:* 0.0232
- *First-token latency:* 0.736
- *Cleanup time:* 0.0648
- *Prompt tokens:* 1,516
- *Generation tokens:* 132
- *Total tokens:* 1,648
- *Prompt throughput (raw):* 2,060 tok/s
- *Generation throughput (raw):* 73.6 tok/s
- *Peak memory:* 8.5
- *Active memory:* 6.4
- *Cache memory:* 0.31
- *Model-load active memory:* 6.44
- *Post-cleanup active memory:* 6.45
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1329
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.molmo2.processing.Molmo2Processor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 4fcbe926577641af12aa5086cf5c292db9dd9730
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Molmo2-8B-4bit/snapshots/4fcbe926577641af12aa5086cf5c292db9dd9730
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Historic Mill and River Scene with Pedestrians

Description: The image captures a historic watermill in Winchester, Hampshire, England, with a stone wall and clay-tiled roof. Two women walk along a pathway above the River Itchen, while an elderly woman sits nearby. The scene is set against a blue sky with white clouds, and the National Trust building is visible in the background.

Keywords: Winchester, Hampshire, England, watermill, stone wall, clay tiles, river, Itchen, pedestrians, elderly woman, women, National Trust, blue sky, clouds, historic building, scenic view, riverbank, sitting area, historic architecture
</pre>

</details>

---

<a id="model-mlx-community-north-micro-vision-instruct-4bit"></a>

### mlx-community/North-Micro-Vision-Instruct-4bit

<details>
<summary>Complete evidence: mlx-community/North-Micro-Vision-Instruct-4bit</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Title has 4 words (requested 5-10); Keyword list has 20
  terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type cohere_compass)
- *Model load time:* 1.03s
- *Generation time:* 4.82s
- *Total time:* 6.31s
- *Input validation time:* 0.436
- *Prompt preparation time:* 0.0228
- *First-token latency:* 3.34
- *Cleanup time:* 0.0648
- *Prompt tokens:* 4,062
- *Generation tokens:* 152
- *Total tokens:* 4,214
- *Prompt throughput (raw):* 1,215 tok/s
- *Generation throughput (raw):* 229 tok/s
- *Peak memory:* 3.9
- *Active memory:* 2.2
- *Cache memory:* 0.64
- *Model-load active memory:* 2.17
- *Post-cleanup active memory:* 2.17
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1422
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.cohere_compass.processing_cohere_compass.CohereCompassProcessor
- *Tokenizer:* transformers.models.cohere.tokenization_cohere.CohereTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 87466363e6c5f57adf91c18c3a62c3c74765f8df
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--North-Micro-Vision-Instruct-4bit/snapshots/87466363e6c5f57adf91c18c3a62c3c74765f8df
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|END_OF_TURN_TOKEN\|>

*Readable output:*

<pre class="model-output-readable">
Title: Historic Winchester City Mill

Description: Visitors walk along the pathway and relax in front of the historic Winchester City Mill, an ancient watermill situated over the River Itchen in Winchester, Hampshire, England. The mill features a distinctive brick and stone facade with a prominent chimney and clay-tiled roof. A concrete bridge spans the River Itchen, connecting the mill to the surrounding area. Pedestrians enjoy the scenic riverside setting, with lush greenery visible in the background.

Keywords: Adobe Stock, Any Vision, Arch, Blue sky, Chimney, Clay tiles, Elderly woman, England, Girls, Hampshire, Mill, National Trust, Pedestrians, People, Rapids, River Itchen, Riverbank, Scenery, Sitting, Stone wall
</pre>

</details>

---

<a id="model-mlx-community-ornith-15-35b-a3b-optiq-4bit"></a>

### mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit

<details>
<summary>Complete evidence: mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Keyword list has 20 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5_moe)
- *Model load time:* 2.56s
- *Generation time:* 2.57s
- *Total time:* 5.61s
- *Input validation time:* 0.439
- *Prompt preparation time:* 0.0337
- *First-token latency:* 0.617
- *Cleanup time:* 0.0654
- *Prompt tokens:* 1,262
- *Generation tokens:* 135
- *Total tokens:* 1,397
- *Prompt throughput (raw):* 2,046 tok/s
- *Generation throughput (raw):* 112 tok/s
- *Peak memory:* 24
- *Active memory:* 23
- *Cache memory:* 0.14
- *Model-load active memory:* 23.1
- *Post-cleanup active memory:* 23.1
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1382
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 5f31fcd089ce6f47f1073d3ae43dcb3a0bd1869f
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Ornith-1.5-35B-A3B-OptiQ-4bit/snapshots/5f31fcd089ce6f47f1073d3ae43dcb3a0bd1869f
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Visitors at Winchester City Mill Over River Itchen

Description: People stroll along the stone bridge and relax in front of the historic Winchester City Mill, an ancient watermill built over the River Itchen. The brick and flake-stone structure with its weathered clay-tiled roof stands under a bright blue sky with scattered clouds, while water rushes through the arches beneath the walkway.

Keywords: Winchester City Mill, River Itchen, Hampshire, England, watermill, clay tiles, chimney, stone bridge, arch, pedestrians, people, elderly woman, girls, sitting, Rapids, riverbank, blue sky, scenery, National Trust, historic building
</pre>

</details>

---

<a id="model-mlx-community-phi-35-vision-instruct-bf16"></a>

### mlx-community/Phi-3.5-vision-instruct-bf16

<details>
<summary>Complete evidence: mlx-community/Phi-3.5-vision-instruct-bf16</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Keyword list has 19 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type phi3_v)
- *Model load time:* 0.92s
- *Generation time:* 3.07s
- *Total time:* 4.45s
- *Input validation time:* 0.436
- *Prompt preparation time:* 0.0188
- *First-token latency:* 0.287
- *Cleanup time:* 0.0604
- *Prompt tokens:* 1,103
- *Generation tokens:* 128
- *Total tokens:* 1,231
- *Prompt throughput (raw):* 3,841 tok/s
- *Generation throughput (raw):* 58.3 tok/s
- *Peak memory:* 9.5
- *Active memory:* 8.3
- *Cache memory:* 0.52
- *Model-load active memory:* 8.29
- *Post-cleanup active memory:* 8.29
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1312
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.phi3_v.processing_phi3_v.Phi3VProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* d8da684308c275a86659e2b36a9189b2f4aec8ea
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Phi-3.5-vision-instruct-bf16/snapshots/d8da684308c275a86659e2b36a9189b2f4aec8ea
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|endoftext\|>

*Readable output:*

<pre class="model-output-readable">
Title: Historic Winchester City Mill Overlooking River Itchen

Description: The historic Winchester City Mill, a National Trust property, stands majestically over the River Itchen in Hampshire, England. Visitors are seen enjoying the scenic view and the historic architecture of the mill.

Keywords: Historic, Winchester, City Mill, National Trust, River Itchen, Hampshire, England, Elderly woman, Girls, Pedestrians, Stone wall, Chimney, Clay tiles, Blue sky, Scenery, Rapids, Riverbank, Sitting, Ancient
</pre>

</details>

---

<a id="model-mlx-community-qwen3-vl-30b-a3b-instruct-4bit"></a>

### mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit

<details>
<summary>Complete evidence: mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Keyword list has 19 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl_moe)
- *Model load time:* 1.90s
- *Generation time:* 56.45s
- *Total time:* 58.82s
- *Input validation time:* 0.435
- *Prompt preparation time:* 0.0285
- *First-token latency:* 53.9
- *Cleanup time:* 0.0659
- *Prompt tokens:* 16,520
- *Generation tokens:* 133
- *Total tokens:* 16,653
- *Prompt throughput (raw):* 306 tok/s
- *Generation throughput (raw):* 88.4 tok/s
- *Peak memory:* 22
- *Active memory:* 18
- *Cache memory:* 2.3
- *Model-load active memory:* 18.3
- *Post-cleanup active memory:* 18.3
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1363
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 0555d34cb1ed80c0e61a5635194c70027b4c2ff3
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3-VL-30B-A3B-Instruct-4bit/snapshots/0555d34cb1ed80c0e61a5635194c70027b4c2ff3
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Historic mill by the River Itchen with pedestrians

Description: On a bright day with a blue sky and scattered clouds, pedestrians walk along a stone bridge over the River Itchen in Winchester, Hampshire, England. The scene features the historic Winchester City Mill, a brick and stone building with a tiled roof and distinctive chimneys, where an elderly woman sits on a bench.

Keywords: Winchester City Mill, River Itchen, Hampshire, England, historic mill, brick building, tiled roof, stone bridge, pedestrians, elderly woman, walking, blue sky, clouds, water, rapids, chimney, diamond-paned windows, stone wall, National Trust
</pre>

</details>

---

<a id="model-mlx-community-qwen35-9b-mlx-4bit"></a>

### mlx-community/Qwen3.5-9B-MLX-4bit

<details>
<summary>Complete evidence: mlx-community/Qwen3.5-9B-MLX-4bit</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Title has 12 words (requested 5-10)
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5)
- *Model load time:* 1.45s
- *Generation time:* 58.77s
- *Total time:* 60.70s
- *Input validation time:* 0.446
- *Prompt preparation time:* 0.0339
- *First-token latency:* 56.4
- *Cleanup time:* 0.0649
- *Prompt tokens:* 16,536
- *Generation tokens:* 132
- *Total tokens:* 16,668
- *Prompt throughput (raw):* 293 tok/s
- *Generation throughput (raw):* 91.5 tok/s
- *Peak memory:* 10.0
- *Active memory:* 6.0
- *Cache memory:* 1.2
- *Model-load active memory:* 5.95
- *Post-cleanup active memory:* 5.95
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1382
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 938d8919941c6e7efd3c7150eff7fe9d12afa631
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-MLX-4bit/snapshots/938d8919941c6e7efd3c7150eff7fe9d12afa631
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
Visitors walk across bridge beside historic Winchester City Mill on River Itchen.

Description:
On a bright day in late August 2026, pedestrians stroll along a stone bridge adjacent to the ancient Winchester City Mill, a National Trust-listed watermill perched over the River Itchen in Hampshire, England, with rushing rapids visible below and a partly cloudy blue sky above.

Keywords:
Winchester City Mill, River Itchen, Hampshire, England, National Trust, historic mill, stone bridge, pedestrians, elderly woman, girls, rapids, clay tiles, brickwork, chimney, stone wall, blue sky
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title:  
Visitors walk across bridge beside historic Winchester City Mill on River Itchen.

Description:  
On a bright day in late August 2026, pedestrians stroll along a stone bridge adjacent to the ancient Winchester City Mill, a National Trust-listed watermill perched over the River Itchen in Hampshire, England, with rushing rapids visible below and a partly cloudy blue sky above.

Keywords:  
Winchester City Mill, River Itchen, Hampshire, England, National Trust, historic mill, stone bridge, pedestrians, elderly woman, girls, rapids, clay tiles, brickwork, chimney, stone wall, blue sky
```

</details>

</details>

---

<a id="model-mlx-community-smolvlm2-22b-instruct-mlx"></a>

### mlx-community/SmolVLM2-2.2B-Instruct-mlx

<details>
<summary>Complete evidence: mlx-community/SmolVLM2-2.2B-Instruct-mlx</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Title has 3 words (requested 5-10); Keyword list has 20
  terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type smolvlm)
- *Model load time:* 0.68s
- *Generation time:* 1.86s
- *Total time:* 3.00s
- *Input validation time:* 0.437
- *Prompt preparation time:* 0.0194
- *First-token latency:* 0.405
- *Cleanup time:* 0.06
- *Prompt tokens:* 1,401
- *Generation tokens:* 84
- *Total tokens:* 1,485
- *Prompt throughput (raw):* 3,462 tok/s
- *Generation throughput (raw):* 127 tok/s
- *Peak memory:* 5.4
- *Active memory:* 4.5
- *Cache memory:* 0.35
- *Model-load active memory:* 4.49
- *Post-cleanup active memory:* 4.49
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1323
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.smolvlm.processing_smolvlm.SmolVLMProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 844516024a1c4400d34489b89ee067d794e432ed
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--SmolVLM2-2.2B-Instruct-mlx/snapshots/844516024a1c4400d34489b89ee067d794e432ed
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <end_of_utterance>

*Readable output:*

<pre class="model-output-readable">
 Title: Winchester City Mill
Description: A historic watermill situated over the River Itchen in Winchester, Hampshire, England.
Keywords: Adobe Stock, Any Vision, Arch, Blue sky, Chimney, Clay tiles, Elderly woman, England, Girls, Hampshire, Mill, National Trust, Pedestrians, People, Rapids, River Itchen, Riverbank, Scenery, Sitting, Stone wall
</pre>

</details>

---

<a id="model-mlx-community-step-37-flash-oq2e"></a>

### mlx-community/Step-3.7-Flash-oQ2e

<details>
<summary>Complete evidence: mlx-community/Step-3.7-Flash-oQ2e</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Keyword list has 20 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type step3p7)
- *Model load time:* 5.48s
- *Generation time:* 16.97s
- *Total time:* 22.92s
- *Input validation time:* 0.435
- *Prompt preparation time:* 0.0399
- *First-token latency:* 12.9
- *Cleanup time:* 0.0638
- *Prompt tokens:* 3,464
- *Generation tokens:* 114
- *Total tokens:* 3,578
- *Prompt throughput (raw):* 268 tok/s
- *Generation throughput (raw):* 46.7 tok/s
- *Peak memory:* 70
- *Active memory:* 63
- *Cache memory:* 0.39
- *Model-load active memory:* 63
- *Post-cleanup active memory:* 63
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1369
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.step3p7.processing_step3p7.Step3VLProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 3dacb46f724ac89725bcd922fb779c7ed1499fe7
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Step-3.7-Flash-oQ2e/snapshots/3dacb46f724ac89725bcd922fb779c7ed1499fe7
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
Winchester City Mill Over the River Itchen

Description:
Visitors walk along the pathway and relax in front of the historic Winchester City Mill, an ancient watermill situated over the River Itchen in Winchester, Hampshire, England, under a bright blue sky with white clouds.

Keywords:
Adobe Stock, Any Vision, Arch, Blue sky, Chimney, Clay tiles, Elderly woman, England, Girls, Hampshire, Mill, National Trust, Pedestrians, People, Rapids, River Itchen, Riverbank, Scenery, Sitting, Stone wall
</pre>

</details>

---

<a id="model-mlx-community-x-reasoner-7b-8bit"></a>

### mlx-community/X-Reasoner-7B-8bit

<details>
<summary>Complete evidence: mlx-community/X-Reasoner-7B-8bit</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Keyword list has 23 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type qwen2_5_vl)
- *Model load time:* 1.25s
- *Generation time:* 20.33s
- *Total time:* 22.04s
- *Input validation time:* 0.438
- *Prompt preparation time:* 0.029
- *First-token latency:* 17.1
- *Cleanup time:* 0.0677
- *Prompt tokens:* 16,531
- *Generation tokens:* 136
- *Total tokens:* 16,667
- *Prompt throughput (raw):* 966 tok/s
- *Generation throughput (raw):* 59.1 tok/s
- *Peak memory:* 13
- *Active memory:* 9.4
- *Cache memory:* 1.4
- *Model-load active memory:* 9.45
- *Post-cleanup active memory:* 9.45
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1421
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen2_5_vl.processing_qwen2_5_vl.Qwen2_5_VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 21732e74613b465bc98e9d5ec210aba5c7adbcc1
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--X-Reasoner-7B-8bit/snapshots/21732e74613b465bc98e9d5ec210aba5c7adbcc1
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
Winchester City Mill: A Historic Watermill

Description:
Visitors stroll along the pathway beneath a historic brick mill over the River Itchen, with an elderly woman relaxing on the stone wall, under a bright blue sky. The mill's weathered clay-tiled roof and stone chimney stand out against the vibrant surroundings.

Keywords: Winchester City Mill, Hampshire, England, River Itchen, Watermill, Historic, Brick, Clay tiles, Chimney, Stone wall, Elderly woman, Girls, Pedestrians, Rapids, Blue sky, National Trust, Scenery, Riverbank, Pathway, Mill building, Traditional architecture, Historical site, Riverside setting.
</pre>

</details>

---

<a id="model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8"></a>

### mlx-community/diffusiongemma-26B-A4B-it-mxfp8

<details>
<summary>Complete evidence: mlx-community/diffusiongemma-26B-A4B-it-mxfp8</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Duplicate keywords: england
- *Arch supported by installed mlx-vlm:* yes (model_type diffusion_gemma)
- *Model load time:* 3.23s
- *Generation time:* 2.53s
- *Total time:* 6.24s
- *Input validation time:* 0.435
- *Prompt preparation time:* 0.0392
- *First-token latency:* 0.324
- *Cleanup time:* 0.0612
- *Prompt tokens:* 570
- *Generation tokens:* 95
- *Total tokens:* 665
- *Prompt throughput (raw):* 1,762 tok/s
- *Generation throughput (raw):* 66.4 tok/s
- *Peak memory:* 28
- *Active memory:* 27
- *Cache memory:* 0.01
- *Model-load active memory:* 27.2
- *Post-cleanup active memory:* 27.2
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1317
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.diffusion_gemma.processing_diffusion_gemma.DiffusionGemma4Processor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* ded389e478f86d498ad9e7f47666e83b166a28f1
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--diffusiongemma-26B-A4B-it-mxfp8/snapshots/ded389e478f86d498ad9e7f47666e83b166a28f1
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
Title: Historic Winchester City Mill on the River Itchen
Description: Visitors walk along a stone pathway in front of the ancient brick and stone watermill over flowing rapids in Hampshire. The historic building features a clay tiled roof and arches over the water.

Keywords: Winchester, City Mill, River Itchen, Hampshire, National Trust, Brick, Stone, Clay tiles, Pedestrians, River, Rapids, Arch, Chimney, England, England
</pre>

</details>

---

<a id="model-mlx-community-gemma-3-27b-it-qat-4bit"></a>

### mlx-community/gemma-3-27b-it-qat-4bit

<details>
<summary>Complete evidence: mlx-community/gemma-3-27b-it-qat-4bit</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Keyword list has 19 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type gemma3)
- *Model load time:* 2.21s
- *Generation time:* 5.88s
- *Total time:* 8.55s
- *Input validation time:* 0.431
- *Prompt preparation time:* 0.0229
- *First-token latency:* 0.955
- *Cleanup time:* 0.0664
- *Prompt tokens:* 569
- *Generation tokens:* 128
- *Total tokens:* 697
- *Prompt throughput (raw):* 596 tok/s
- *Generation throughput (raw):* 29.9 tok/s
- *Peak memory:* 17
- *Active memory:* 16
- *Cache memory:* 0.43
- *Model-load active memory:* 16.1
- *Post-cleanup active memory:* 16.1
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1346
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.gemma3.processing_gemma3.Gemma3Processor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* fc4e000f32af1b7b6779294e490a7d2a80bac611
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--gemma-3-27b-it-qat-4bit/snapshots/fc4e000f32af1b7b6779294e490a7d2a80bac611
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
Title: Winchester City Mill, Hampshire: People on the Pathway

Description: Captured on 31st August 2026, this image shows Winchester City Mill, a historic brick and stone building spanning the River Itchen; pedestrians walk along the pathway in bright daylight, with rapids visible below. The mill features a distinctive red clay tiled roof and chimney.

Keywords: Winchester City Mill, Hampshire, England, River Itchen, Watermill, Historic building, Brickwork, Stone wall, Arch, Pathway, Pedestrians, People, Rapids, Riverbank, Clay tiles, Chimney, Blue sky, Exterior, August
</pre>

</details>

---

<a id="model-mlx-community-pixtral-12b-8bit"></a>

### mlx-community/pixtral-12b-8bit

<details>
<summary>Complete evidence: mlx-community/pixtral-12b-8bit</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Keyword list has 19 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type pixtral)
- *Model load time:* 1.66s
- *Generation time:* 5.23s
- *Total time:* 7.34s
- *Input validation time:* 0.425
- *Prompt preparation time:* 0.0209
- *First-token latency:* 1.82
- *Cleanup time:* 0.062
- *Prompt tokens:* 3,111
- *Generation tokens:* 110
- *Total tokens:* 3,221
- *Prompt throughput (raw):* 1,706 tok/s
- *Generation throughput (raw):* 39.4 tok/s
- *Peak memory:* 16
- *Active memory:* 13
- *Cache memory:* 0.59
- *Model-load active memory:* 13.5
- *Post-cleanup active memory:* 13.5
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1291
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.pixtral.processing_pixtral.PixtralProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 79e24b66302d4ac5e75491767f8914b2aa6ac1f8
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--pixtral-12b-8bit/snapshots/79e24b66302d4ac5e75491767f8914b2aa6ac1f8
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Title:
Historic Winchester Mill on River Itchen, Hampshire

Description:
Two individuals walk along the pathway in front of the historic Winchester City Mill, an ancient watermill situated over the River Itchen in Winchester, Hampshire, England, under a bright blue sky.

Keywords:
Adobe Stock, Any Vision, Arch, Blue sky, Chimney, Clay tiles, Elderly woman, England, Hampshire, Mill, National Trust, Pedestrians, People, Rapids, River Itchen, Riverbank, Scenery, Sitting, Stone wall
</pre>

</details>

---
