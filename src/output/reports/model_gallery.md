# Model Output Gallery

Generated on: 2026-09-04 15:03:15 BST

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

| Model                                                                                                                   | Usability             | Total s | Gen TPS    | Prefill/first s | Peak GB | Prompt tok | Gen tok | Observations                                                                                                                      |
|-------------------------------------------------------------------------------------------------------------------------|-----------------------|---------|------------|-----------------|---------|------------|---------|-----------------------------------------------------------------------------------------------------------------------------------|
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `usable`              | 8.87s   | 32.3 tok/s | 1.09            | 18      | 2,597      | 162     | none                                                                                                                              |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `usable`              | 5.74s   | 34.3 tok/s | 0.81            | 17      | 2,086      | 88      | none                                                                                                                              |
| [`mlx-community/LFM2.5-VL-3B-OptiQ-4bit`](#model-mlx-community-lfm25-vl-3b-optiq-4bit)                                  | `usable`              | 2.39s   | 210 tok/s  | 0.46            | 4.0     | 2,086      | 83      | none                                                                                                                              |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `usable`              | 6.99s   | 67.1 tok/s | 2.27            | 13      | 2,918      | 180     | none                                                                                                                              |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `usable`              | 3.33s   | 190 tok/s  | 0.97            | 7.8     | 2,917      | 139     | none                                                                                                                              |
| [`mlx-community/Muse-Glimmer-30B-OptiQ-4bit`](#model-mlx-community-muse-glimmer-30b-optiq-4bit)                         | `usable`              | 50.63s  | 25.4 tok/s | 7.85            | 25      | 4,388      | 992     | none                                                                                                                              |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `usable`              | 28.07s  | 91.5 tok/s | 16.39           | 8.4     | 16,522     | 907     | none                                                                                                                              |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `usable`              | 57.81s  | 110 tok/s  | 52.93           | 24      | 16,536     | 143     | none                                                                                                                              |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `usable`              | 86.51s  | 17.8 tok/s | 74.45           | 33      | 16,536     | 139     | none                                                                                                                              |
| [`mlx-community/Qwen3.8-27B-4bit`](#model-mlx-community-qwen38-27b-4bit)                                                | `usable`              | 74.62s  | 30.5 tok/s | 67.22           | 21      | 16,536     | 131     | none                                                                                                                              |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `usable`              | 5.60s   | 81.9 tok/s | 0.33            | 29      | 570        | 84      | none                                                                                                                              |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `usable`              | 5.95s   | 56.2 tok/s | 0.32            | 28      | 570        | 85      | none                                                                                                                              |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `usable`              | 4.34s   | 129 tok/s  | 0.40            | 16      | 574        | 105     | none                                                                                                                              |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `usable`              | 7.98s   | 25.1 tok/s | 1.13            | 20      | 574        | 90      | none                                                                                                                              |
| [`mlx-community/granite-4.0-3b-vision-4bit`](#model-mlx-community-granite-40-3b-vision-4bit)                            | `usable`              | 2.47s   | 176 tok/s  | 0.40            | 4.7     | 1,371      | 88      | none                                                                                                                              |
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `usable with caveats` | 1.62s   | 483 tok/s  | 0.11            | 1.9     | 2,094      | 76      | title/keyword constraints failed                                                                                                  |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `usable with caveats` | 9.91s   | 30.2 tok/s | 3.15            | 23      | 2,385      | 119     | title/keyword constraints failed                                                                                                  |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `usable with caveats` | 19.05s  | 63.1 tok/s | 2.65            | 60      | 1,605      | 583     | title/keyword constraints failed                                                                                                  |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `usable with caveats` | 25.67s  | 39.9 tok/s | 15.17           | 78      | 6,430      | 108     | control tokens visible                                                                                                            |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `usable with caveats` | 157.63s | 4.67 tok/s | 1.49            | 40      | 1,307      | 710     | role tokens visible                                                                                                               |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `usable with caveats` | 2.23s   | 183 tok/s  | 0.34            | 4.0     | 2,094      | 119     | title/keyword constraints failed                                                                                                  |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `usable with caveats` | 6.86s   | 64.7 tok/s | 2.02            | 13      | 2,918      | 180     | title/keyword constraints failed                                                                                                  |
| [`mlx-community/Molmo2-8B-4bit`](#model-mlx-community-molmo2-8b-4bit)                                                   | `usable with caveats` | 4.40s   | 72.7 tok/s | 0.74            | 8.1     | 1,516      | 132     | title/keyword constraints failed                                                                                                  |
| [`mlx-community/North-Micro-Vision-Instruct-4bit`](#model-mlx-community-north-micro-vision-instruct-4bit)               | `usable with caveats` | 5.62s   | 229 tok/s  | 3.09            | 3.9     | 4,062      | 152     | title/keyword constraints failed                                                                                                  |
| [`mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit`](#model-mlx-community-ornith-15-35b-a3b-optiq-4bit)                      | `usable with caveats` | 5.41s   | 107 tok/s  | 0.72            | 24      | 1,262      | 135     | title/keyword constraints failed                                                                                                  |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `usable with caveats` | 4.08s   | 56.8 tok/s | 0.30            | 9.5     | 1,103      | 128     | title/keyword constraints failed                                                                                                  |
| [`mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit`](#model-mlx-community-qwen3-vl-30b-a3b-instruct-4bit)                   | `usable with caveats` | 58.72s  | 87.1 tok/s | 54.20           | 23      | 16,520     | 133     | title/keyword constraints failed                                                                                                  |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `usable with caveats` | 56.12s  | 92.1 tok/s | 52.23           | 10.0    | 16,536     | 132     | title/keyword constraints failed                                                                                                  |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `usable with caveats` | 2.73s   | 125 tok/s  | 0.41            | 5.4     | 1,401      | 84      | title/keyword constraints failed                                                                                                  |
| [`mlx-community/Step-3.7-Flash-oQ2e`](#model-mlx-community-step-37-flash-oq2e)                                          | `usable with caveats` | 25.61s  | 46.6 tok/s | 14.62           | 70      | 3,464      | 114     | title/keyword constraints failed                                                                                                  |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `usable with caveats` | 20.47s  | 57.8 tok/s | 15.85           | 13      | 16,531     | 136     | title/keyword constraints failed                                                                                                  |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `usable with caveats` | 8.05s   | 31.2 tok/s | 0.99            | 17      | 569        | 128     | title/keyword constraints failed                                                                                                  |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `usable with caveats` | 6.80s   | 38.9 tok/s | 1.70            | 16      | 3,111      | 110     | title/keyword constraints failed                                                                                                  |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `unusable`            | 25.88s  | 92.2 tok/s | 13.44           | 8.4     | 16,520     | 1,000   | repeated text; cut off at token limit; title/keyword constraints failed                                                           |
| [`jinaai/jina-vlm-mlx`](#model-jinaai-jina-vlm-mlx)                                                                     | `unusable`            | 4.61s   | 139 tok/s  | 0.95            | 3.7     | 2,050      | 294     | missing required fields                                                                                                           |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `unusable`            | 27.99s  | 42.0 tok/s | 2.00            | 15      | 3,202      | 1,000   | missing required fields; echoes instructions; extra text before Title; cut off at token limit                                     |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `unusable`            | 2.43s   | 359 tok/s  | 0.11            | 2.2     | 307        | 155     | missing required fields                                                                                                           |
| [`mlx-community/GLM-4.1V-9B-Thinking-8bit`](#model-mlx-community-glm-41v-9b-thinking-8bit)                              | `unusable`            | 29.38s  | 46.4 tok/s | 5.49            | 13      | 6,426      | 1,000   | echoes instructions; extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `unusable`            | 20.79s  | 76.4 tok/s | 5.71            | 8.4     | 6,430      | 1,000   | repeated text; cut off at token limit; title/keyword constraints failed                                                           |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `unusable`            | 2.21s   | 288 tok/s  | 0.25            | 3.3     | 907        | 70      | missing required fields; extra text before Title                                                                                  |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `unusable`            | 6.81s   | 48.8 tok/s | 0.36            | 17      | 568        | 167     | missing required fields; extra text before Title                                                                                  |
| [`tencent/Youtu-VL-4B-Instruct`](#model-tencent-youtu-vl-4b-instruct)                                                   | `not evaluated`       | 1.06s   | -          | -               | -       | -          | -       | none                                                                                                                              |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Resource Highlights

Fastest clean completion: `mlx-community/LFM2.5-VL-3B-OptiQ-4bit` at 210 tok/s

Average clean-completion throughput: 85.2 tok/s (indicative only: tokenizers and architectures differ across models)

Lowest peak memory among clean completions: `mlx-community/LFM2.5-VL-3B-OptiQ-4bit` at 4.0 GB

## Avoid for This Run

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                  | Usability       | Observations                                                                                                                      |
|--------------------------------------------------------------------------------------------------------|-----------------|-----------------------------------------------------------------------------------------------------------------------------------|
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                        | `unusable`      | repeated text; cut off at token limit; title/keyword constraints failed                                                           |
| [`jinaai/jina-vlm-mlx`](#model-jinaai-jina-vlm-mlx)                                                    | `unusable`      | missing required fields                                                                                                           |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx) | `unusable`      | missing required fields; echoes instructions; extra text before Title; cut off at token limit                                     |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                             | `unusable`      | missing required fields                                                                                                           |
| [`mlx-community/GLM-4.1V-9B-Thinking-8bit`](#model-mlx-community-glm-41v-9b-thinking-8bit)             | `unusable`      | echoes instructions; extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                       | `unusable`      | repeated text; cut off at token limit; title/keyword constraints failed                                                           |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                           | `unusable`      | missing required fields; extra text before Title                                                                                  |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                      | `unusable`      | missing required fields; extra text before Title                                                                                  |
| [`tencent/Youtu-VL-4B-Instruct`](#model-tencent-youtu-vl-4b-instruct)                                  | `not evaluated` | none                                                                                                                              |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Output at a Glance

The first 280 characters of each model's actual output (or failure evidence for crashes), in chooser order. Complete exact evidence for every model follows below.

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                                   | Usability             | Output preview                                                                                                                                                                                                                                                                                                |
|-------------------------------------------------------------------------------------------------------------------------|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `usable`              | Title: Winchester City Mill and River Itchen in Hampshire, England.<br>Description: The historic Winchester City Mill, situated over the River Itchen in Winchester, Hampshire, England, is a popular tourist attraction. The mill features a distinctive red brick exterior with a stee...                   |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `usable`              | Title: Historic Mill by River Itchen<br>Description: Pedestrians walk and relax near the historic Winchester City Mill, featuring a charming stone bridge and clay-tiled roof under a blue sky.<br>Keywords: Winchester City Mill, historic watermill, River Itchen, Hampshire, England, Na...                |
| [`mlx-community/LFM2.5-VL-3B-OptiQ-4bit`](#model-mlx-community-lfm25-vl-3b-optiq-4bit)                                  | `usable`              | Title: Historic Mill Over River Itchen<br><br>Description: Visitors walk along a pathway beside the historic Winchester City Mill, an ancient watermill with distinctive clay tiles and chimneys, set against a blue sky with clouds.<br><br>Keywords: National Trust, Hampshire, England, Mill,...           |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `usable`              | Title:<br>**Winchester City Mill: Historic Watermill by the River Itchen**<br><br>Description:<br>A group of visitors, including two women walking along the riverside path, explore the historic **Winchester City Mill**, a National Trust-listed watermill with a distinctive **stone and bric...          |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `usable`              | Title:<br>Historic Winchester Watermill by the River Itchen<br><br>Description:<br>A picturesque scene captures the Winchester City Mill, a traditional brick-and-stone watermill with clay-tiled roofs, flowing over the River Itchen under bright blue skies. Visitors stroll along the riversi...          |
| [`mlx-community/Muse-Glimmer-30B-OptiQ-4bit`](#model-mlx-community-muse-glimmer-30b-optiq-4bit)                         | `usable`              | Title:<br>Historic Winchester City Mill Over River Itchen<br><br>Description:<br>Captured on 31 August 2026, the historic Winchester City Mill, a brick and flint watermill with a clay-tile roof and chimney, stands over the River Itchen in Winchester, Hampshire, England. Two girls walk alo...          |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `usable`              | Got it, let's tackle this step by step. First, the title needs to be 5-10 words. The image shows a historic watermill, so "Winchester City Mill, River Itchen, Hampshire" is key. Let's make it concise: "Winchester City Mill, River Itchen, Hampshire, England"<br><br>Next, the descript...                |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `usable`              | Title: Historic Watermill Over River Itchen<br><br>Description: Captured on August 31, 2026, this image depicts the historic Winchester City Mill, an ancient structure featuring a weathered clay-tiled roof and flint-and-brick walls, situated over the River Itchen in Hampshire. Under...                |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `usable`              | Title:<br>Winchester City Mill on the River Itchen<br><br>Description:<br>Visitors walk along the bridge and an elderly woman sits with a walker in front of the historic Winchester City Mill, a brick and stone watermill situated over the River Itchen in Hampshire, England. The scene is ca...          |
| [`mlx-community/Qwen3.8-27B-4bit`](#model-mlx-community-qwen38-27b-4bit)                                                | `usable`              | Title:<br>Winchester City Mill Spanning the River Itchen<br><br>Description:<br>The historic Winchester City Mill, a brick and stone structure with a clay-tiled roof, stands over the River Itchen in Hampshire, England. Pedestrians walk along the bridge pathway in front of the building whi...          |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `usable`              | Title: Historic Winchester City Mill on River Itchen<br>Description: Visitors walk along a stone pathway in front of the ancient brick and stone watermill situated over the rushing River Itchen in Hampshire.<br>Keywords: Winchester, Hampshire, River Itchen, England, watermill, histo...                |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `usable`              | Title: Historic Winchester City Mill on the River Itchen<br>Description: Visitors walk along a stone pathway in front of the ancient brick and stone watermill situated over the rushing River Itchen in Hampshire.<br>Keywords: Winchester, Hampshire, River Itchen, England, watermill, N...                |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `usable`              | Title: People walking by historic Winchester City Mill<br>Description: Two women walk along a stone pathway in front of the historic brick and flint Winchester City Mill over the River Itchen. The scene captures the ancient watermill's tiled roof and stone architecture under a br...                   |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `usable`              | Title: Visitors at the historic Winchester City Mill, Hampshire<br>Description: Visitors walk along a pathway and relax in front of the ancient Winchester City Mill, situated over the River Itchen in Hampshire, England, under a bright blue sky.<br>Keywords: Winchester City Mill, Riv...                |
| [`mlx-community/granite-4.0-3b-vision-4bit`](#model-mlx-community-granite-40-3b-vision-4bit)                            | `usable`              | Title: "Historic Mill by the River Itchen"<br>Description: "Visitors stroll along the pathway in front of the historic Winchester City Mill, enjoying the serene setting by the River Itchen."<br>Keywords: Historic Mill, Winchester City Mill, River Itchen, Pedestrians, Elderly woman,...                 |
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `usable_with_caveats` | Title:<br>Historic Winchester City Mill<br><br>Description:<br>A historic stone-built watermill stands over the River Itchen in Winchester, England, with visitors walking along a stone bridge and enjoying the scenic view.<br><br>Keywords:<br>Historic, watermill, Winchester, England, River Itchen,...  |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `usable_with_caveats` | Title:<br>Visitors at Winchester City Mill<br><br>Description:<br>Visitors walk along the pathway and relax in front of the historic Winchester City Mill, an ancient watermill situated over the River Itchen in Winchester, Hampshire, England, under a blue sky.<br><br>Keywords:<br>Winchester City Mi... |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `usable_with_caveats` | Alright, let's get to work. I've got the image and the hints, and I need to craft some metadata. First, the title. It needs to be concise and descriptive. "Hampshire Riverbank with Strollers" is a good start, but I can do better. "Strollers on the Riverbank near Winchester" is...                      |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `usable_with_caveats` | <\|begin_of_box\|>Title: Historic Winchester City Mill by River Itchen<br><br>Description: Two girls walk along the pathway in front of the historic Winchester City Mill, a brick and stone watermill with clay tiles, situated over the River Itchen in Hampshire, England, under a blue sk...              |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `usable_with_caveats` | ◁think▷Okay, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. The capture date/time is 2026-08-31 15:02:20 UTC+01:00, but I shouldn't claim it's visible. The descriptive hints mention Winchester City M...                      |
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
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `usable_with_caveats` | Title: Winchester City Mill, Hampshire: People on the Pathway<br><br>Description: Captured on 31st August 2026, this image shows Winchester City Mill, a historic brick and stone building spanning the River Itchen; pedestrians walk along the pathway in bright daylight, with rapids vi...                |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `usable_with_caveats` | Title:<br>Historic Winchester Mill on River Itchen, Hampshire<br><br>Description:<br>Two individuals walk along the pathway in front of the historic Winchester City Mill, an ancient watermill situated over the River Itchen in Winchester, Hampshire, England, under a bright blue sky.<br><br>Keywo...    |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `unusable`            | Title: River Itchen, Hampshire, England<br><br>Description: A historic watermill in Hampshire, England, with a stone wall and brick building, is situated over the River Itchen. Pedestrians walk along a bridge, and an elderly woman sits on a bench in front of the mill. The scene is s...                |
| [`jinaai/jina-vlm-mlx`](#model-jinaai-jina-vlm-mlx)                                                                     | `unusable`            | Title:<br>Historic Winchester City Mill<br><br>Description:<br>The historic Winchester City Mill, an ancient watermill situated over the River Itchen in Winchester, Hampshire, England, is captured in this image. Pedestrians walk along the pathway, relaxing in front of the mill, which is a...          |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `unusable`            | Here are my reasoning steps:<br>We need to produce British-English catalogue metadata from the image and supplied context. The image shows a historic watermill (Winchester City Mill) over a river (River Itchen). The building is a brick and stone structure with a tiled roof, a chi...                   |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `unusable`            | A tranquil scene captures visitors enjoying a walk along the historic Winchester City Mill, a picturesque watermill nestled by the River Itchen in Winchester, Hampshire, England. The image, taken on 2026-08-31 15:02:20 UTC+01:00, showcases the blend of natural and architectura...                      |
| [`mlx-community/GLM-4.1V-9B-Thinking-8bit`](#model-mlx-community-glm-41v-9b-thinking-8bit)                              | `unusable`            | &lt;think&gt;Got it, let's tackle this task step by step. First, the title needs to be 5-10 words, concrete. The image shows a historic mill by a river with people, so maybe "Winchester City Mill by River Itchen".<br><br>Next, the description: need to combine context (capture date, locati...          |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `unusable`            | Title: Winchester City Mill<br>Description: The historic Winchester City Mill, an ancient watermill over the River Itchen, is shown with visitors walking along the pathway and an elderly woman sitting nearby.<br>Keywords: Winchester City Mill, River Itchen, watermill, historic build...                |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `unusable`            | Visitors walk near historic mill over river under bright sky.<br>Lighting is sunny with clear blue skies above the old building.<br><br>Keywords: historic mill, river Itchen, stone bridge, brick building, blue sky, pedestrians, elderly woman, girls, England, Hampshire, watermill, stone...             |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `unusable`            | A picturesque scene unfolds at the historic Winchester City Mill, a charming watermill nestled along the River Itchen in Winchester, Hampshire, England. The mill's weathered brick and stone facade, topped with a terracotta tiled roof, exudes a sense of timelessness. Two pedest...                      |
| [`tencent/Youtu-VL-4B-Instruct`](#model-tencent-youtu-vl-4b-instruct)                                                   | `not_evaluated`       | Model loading failed: cannot import name 'DefaultFastImageProcessorKwargs' from 'transformers.image_processing_utils_fast' (unknown location)                                                                                                                                                                 |
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

<a id="model-mlx-community-idefics3-8b-llama3-bf16"></a>

### mlx-community/Idefics3-8B-Llama3-bf16

<details>
<summary>Complete evidence: mlx-community/Idefics3-8B-Llama3-bf16</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type idefics3)
- *Model load time:* 1.93s
- *Generation time:* 6.94s
- *Total time:* 8.87s
- *Input validation time:* 6.48e-05
- *Prompt preparation time:* 0.00268
- *First-token latency:* 1.09
- *Cleanup time:* 0.106
- *Prompt tokens:* 2,597
- *Generation tokens:* 162
- *Total tokens:* 2,759
- *Prompt throughput (raw):* 2,389 tok/s
- *Generation throughput (raw):* 32.3 tok/s
- *Peak memory:* 18
- *Active memory:* 17
- *Cache memory:* 0.43
- *Model-load active memory:* 16.9
- *Post-cleanup active memory:* 0.003
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
- *Model load time:* 1.62s
- *Generation time:* 4.11s
- *Total time:* 5.74s
- *Input validation time:* 6.7e-05
- *Prompt preparation time:* 0.00289
- *First-token latency:* 0.814
- *Cleanup time:* 0.0999
- *Prompt tokens:* 2,086
- *Generation tokens:* 88
- *Total tokens:* 2,174
- *Prompt throughput (raw):* 2,564 tok/s
- *Generation throughput (raw):* 34.3 tok/s
- *Peak memory:* 17
- *Active memory:* 16
- *Cache memory:* 0.17
- *Model-load active memory:* 15.9
- *Post-cleanup active memory:* 0.00331
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
- *Model load time:* 0.90s
- *Generation time:* 1.47s
- *Total time:* 2.39s
- *Input validation time:* 4.59e-05
- *Prompt preparation time:* 0.0092
- *First-token latency:* 0.46
- *Cleanup time:* 0.0791
- *Prompt tokens:* 2,086
- *Generation tokens:* 83
- *Total tokens:* 2,169
- *Prompt throughput (raw):* 4,533 tok/s
- *Generation throughput (raw):* 210 tok/s
- *Peak memory:* 4.0
- *Active memory:* 2.8
- *Cache memory:* 0.08
- *Model-load active memory:* 2.82
- *Post-cleanup active memory:* 0.00436
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
- *Model load time:* 1.37s
- *Generation time:* 5.60s
- *Total time:* 6.99s
- *Input validation time:* 4.98e-05
- *Prompt preparation time:* 0.00758
- *First-token latency:* 2.27
- *Cleanup time:* 0.0924
- *Prompt tokens:* 2,918
- *Generation tokens:* 180
- *Total tokens:* 3,098
- *Prompt throughput (raw):* 1,288 tok/s
- *Generation throughput (raw):* 67.1 tok/s
- *Peak memory:* 13
- *Active memory:* 8.0
- *Cache memory:* 0.60
- *Model-load active memory:* 8.01
- *Post-cleanup active memory:* 0.00513
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
- *Model load time:* 0.98s
- *Generation time:* 2.33s
- *Total time:* 3.33s
- *Input validation time:* 5.25e-05
- *Prompt preparation time:* 0.00717
- *First-token latency:* 0.969
- *Cleanup time:* 0.0817
- *Prompt tokens:* 2,917
- *Generation tokens:* 139
- *Total tokens:* 3,056
- *Prompt throughput (raw):* 3,011 tok/s
- *Generation throughput (raw):* 190 tok/s
- *Peak memory:* 7.8
- *Active memory:* 2.8
- *Cache memory:* 0.38
- *Model-load active memory:* 2.75
- *Post-cleanup active memory:* 0.00565
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
- *Model load time:* 3.01s
- *Generation time:* 47.59s
- *Total time:* 50.63s
- *Input validation time:* 4.98e-05
- *Prompt preparation time:* 0.0201
- *First-token latency:* 7.85
- *Cleanup time:* 0.125
- *Prompt tokens:* 4,388
- *Generation tokens:* 992
- *Total tokens:* 5,380
- *Prompt throughput (raw):* 559 tok/s
- *Generation throughput (raw):* 25.4 tok/s
- *Peak memory:* 25
- *Active memory:* 24
- *Cache memory:* 0.30
- *Model-load active memory:* 23.9
- *Post-cleanup active memory:* 0.00637
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
- *Model load time:* 0.72s
- *Generation time:* 27.33s
- *Total time:* 28.07s
- *Input validation time:* 6.2e-05
- *Prompt preparation time:* 0.013
- *First-token latency:* 16.4
- *Cleanup time:* 0.0976
- *Prompt tokens:* 16,522
- *Generation tokens:* 907
- *Total tokens:* 17,429
- *Prompt throughput (raw):* 1,008 tok/s
- *Generation throughput (raw):* 91.5 tok/s
- *Peak memory:* 8.4
- *Active memory:* 4.3
- *Cache memory:* 2.7
- *Model-load active memory:* 4.26
- *Post-cleanup active memory:* 0.00778
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
- *Generation time:* 55.26s
- *Total time:* 57.81s
- *Input validation time:* 6.1e-05
- *Prompt preparation time:* 0.0155
- *First-token latency:* 52.9
- *Cleanup time:* 0.13
- *Prompt tokens:* 16,536
- *Generation tokens:* 143
- *Total tokens:* 16,679
- *Prompt throughput (raw):* 312 tok/s
- *Generation throughput (raw):* 110 tok/s
- *Peak memory:* 24
- *Active memory:* 20
- *Cache memory:* 0.89
- *Model-load active memory:* 20.4
- *Post-cleanup active memory:* 0.0086
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

<a id="model-mlx-community-qwen36-27b-mxfp8"></a>

### mlx-community/Qwen3.6-27B-mxfp8

<details>
<summary>Complete evidence: mlx-community/Qwen3.6-27B-mxfp8</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5)
- *Model load time:* 3.17s
- *Generation time:* 83.32s
- *Total time:* 86.51s
- *Input validation time:* 6.09e-05
- *Prompt preparation time:* 0.0162
- *First-token latency:* 74.5
- *Cleanup time:* 0.145
- *Prompt tokens:* 16,536
- *Generation tokens:* 139
- *Total tokens:* 16,675
- *Prompt throughput (raw):* 222 tok/s
- *Generation throughput (raw):* 17.8 tok/s
- *Peak memory:* 33
- *Active memory:* 29
- *Cache memory:* 1.9
- *Model-load active memory:* 28.7
- *Post-cleanup active memory:* 0.00964
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
- *Resolved model revision:* 5db9fd9c38ce083048ee12217eceed7482ca7ad4
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-27B-mxfp8/snapshots/5db9fd9c38ce083048ee12217eceed7482ca7ad4
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
Winchester City Mill on the River Itchen

Description:
Visitors walk along the bridge and an elderly woman sits with a walker in front of the historic Winchester City Mill, a brick and stone watermill situated over the River Itchen in Hampshire, England. The scene is captured under a blue sky with white clouds, highlighting the building's clay tile roof and the rushing water below.

Keywords:
Winchester City Mill, River Itchen, Hampshire, England, Watermill, Bridge, Pedestrians, Elderly woman, Brick building, Stone wall, Rapids, Blue sky, Chimney, Clay tiles, Scenery, National Trust, Walking, Riverbank
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
- *Model load time:* 2.05s
- *Generation time:* 72.54s
- *Total time:* 74.62s
- *Input validation time:* 6.73e-05
- *Prompt preparation time:* 0.0184
- *First-token latency:* 67.2
- *Cleanup time:* 0.131
- *Prompt tokens:* 16,536
- *Generation tokens:* 131
- *Total tokens:* 16,667
- *Prompt throughput (raw):* 246 tok/s
- *Generation throughput (raw):* 30.5 tok/s
- *Peak memory:* 21
- *Active memory:* 16
- *Cache memory:* 1.9
- *Model-load active memory:* 16.1
- *Post-cleanup active memory:* 0.0101
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

<a id="model-mlx-community-diffusiongemma-26b-a4b-it-8bit"></a>

### mlx-community/diffusiongemma-26B-A4B-it-8bit

<details>
<summary>Complete evidence: mlx-community/diffusiongemma-26B-A4B-it-8bit</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type diffusion_gemma)
- *Model load time:* 3.34s
- *Generation time:* 2.23s
- *Total time:* 5.60s
- *Input validation time:* 5.98e-05
- *Prompt preparation time:* 0.0248
- *First-token latency:* 0.329
- *Cleanup time:* 0.156
- *Prompt tokens:* 570
- *Generation tokens:* 84
- *Total tokens:* 654
- *Prompt throughput (raw):* 1,733 tok/s
- *Generation throughput (raw):* 81.9 tok/s
- *Peak memory:* 29
- *Active memory:* 28
- *Cache memory:* 0.01
- *Model-load active memory:* 28
- *Post-cleanup active memory:* 0.0108
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
- *Resolved model revision:* 7b95e3887078ba56283c24f2578d6e5a06b9d7e8
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--diffusiongemma-26B-A4B-it-8bit/snapshots/7b95e3887078ba56283c24f2578d6e5a06b9d7e8
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
Title: Historic Winchester City Mill on River Itchen
Description: Visitors walk along a stone pathway in front of the ancient brick and stone watermill situated over the rushing River Itchen in Hampshire.
Keywords: Winchester, Hampshire, River Itchen, England, watermill, historic, brick, stone, clay tiles, chimney, pedestrians, rapids, riverbank, National Trust, architecture, scenery
</pre>

</details>

---

<a id="model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8"></a>

### mlx-community/diffusiongemma-26B-A4B-it-mxfp8

<details>
<summary>Complete evidence: mlx-community/diffusiongemma-26B-A4B-it-mxfp8</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type diffusion_gemma)
- *Model load time:* 3.23s
- *Generation time:* 2.71s
- *Total time:* 5.95s
- *Input validation time:* 6.29e-05
- *Prompt preparation time:* 0.000949
- *First-token latency:* 0.324
- *Cleanup time:* 0.119
- *Prompt tokens:* 570
- *Generation tokens:* 85
- *Total tokens:* 655
- *Prompt throughput (raw):* 1,761 tok/s
- *Generation throughput (raw):* 56.2 tok/s
- *Peak memory:* 28
- *Active memory:* 27
- *Cache memory:* 0.01
- *Model-load active memory:* 27.2
- *Post-cleanup active memory:* 0.0108
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
Description: Visitors walk along a stone pathway in front of the ancient brick and stone watermill situated over the rushing River Itchen in Hampshire.
Keywords: Winchester, Hampshire, River Itchen, England, watermill, National Trust, historic, pedestrians, stone wall, brick, clay tiles, chimney, arch, river, rapids, scenery
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
- *Model load time:* 2.39s
- *Generation time:* 1.92s
- *Total time:* 4.34s
- *Input validation time:* 4.85e-05
- *Prompt preparation time:* 0.0226
- *First-token latency:* 0.405
- *Cleanup time:* 0.123
- *Prompt tokens:* 574
- *Generation tokens:* 105
- *Total tokens:* 679
- *Prompt throughput (raw):* 1,418 tok/s
- *Generation throughput (raw):* 129 tok/s
- *Peak memory:* 16
- *Active memory:* 15
- *Cache memory:* 0.20
- *Model-load active memory:* 15.4
- *Post-cleanup active memory:* 0.0124
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
- *Model load time:* 2.55s
- *Generation time:* 5.42s
- *Total time:* 7.98s
- *Input validation time:* 5.02e-05
- *Prompt preparation time:* 0.00161
- *First-token latency:* 1.13
- *Cleanup time:* 0.138
- *Prompt tokens:* 574
- *Generation tokens:* 90
- *Total tokens:* 664
- *Prompt throughput (raw):* 507 tok/s
- *Generation throughput (raw):* 25.1 tok/s
- *Peak memory:* 20
- *Active memory:* 18
- *Cache memory:* 0.76
- *Model-load active memory:* 18.4
- *Post-cleanup active memory:* 0.013
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
- *Model load time:* 0.66s
- *Generation time:* 1.80s
- *Total time:* 2.47s
- *Input validation time:* 6.11e-05
- *Prompt preparation time:* 0.0106
- *First-token latency:* 0.395
- *Cleanup time:* 0.0908
- *Prompt tokens:* 1,371
- *Generation tokens:* 88
- *Total tokens:* 1,459
- *Prompt throughput (raw):* 3,469 tok/s
- *Generation throughput (raw):* 176 tok/s
- *Peak memory:* 4.7
- *Active memory:* 3.0
- *Cache memory:* 0.22
- *Model-load active memory:* 3.03
- *Post-cleanup active memory:* 0.0132
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
- *Model load time:* 0.38s
- *Generation time:* 0.82s
- *Total time:* 1.62s
- *Input validation time:* 0.407
- *Prompt preparation time:* 0.00671
- *First-token latency:* 0.108
- *Cleanup time:* 0.0609
- *Prompt tokens:* 2,094
- *Generation tokens:* 76
- *Total tokens:* 2,170
- *Prompt throughput (raw):* 19,429 tok/s
- *Generation throughput (raw):* 483 tok/s
- *Peak memory:* 1.9
- *Active memory:* 0.90
- *Cache memory:* 0.08
- *Model-load active memory:* 0.897
- *Post-cleanup active memory:* 0.000132
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
- *Model load time:* 2.14s
- *Generation time:* 7.76s
- *Total time:* 9.91s
- *Input validation time:* 6.02e-05
- *Prompt preparation time:* 0.00874
- *First-token latency:* 3.15
- *Cleanup time:* 0.106
- *Prompt tokens:* 2,385
- *Generation tokens:* 119
- *Total tokens:* 2,504
- *Prompt throughput (raw):* 756 tok/s
- *Generation throughput (raw):* 30.2 tok/s
- *Peak memory:* 23
- *Active memory:* 18
- *Cache memory:* 0.47
- *Model-load active memory:* 18
- *Post-cleanup active memory:* 0.00128
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

<a id="model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16"></a>

### mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

<details>
<summary>Complete evidence: mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Keyword list has 24 terms (requested 10-18); Duplicate
  keywords: riverbank, national trust
- *Arch supported by installed mlx-vlm:* yes (model_type ernie4_5_moe_vl)
- *Model load time:* 5.96s
- *Generation time:* 13.08s
- *Total time:* 19.05s
- *Input validation time:* 8.66e-05
- *Prompt preparation time:* 0.0109
- *First-token latency:* 2.65
- *Cleanup time:* 0.221
- *Prompt tokens:* 1,605
- *Generation tokens:* 583
- *Total tokens:* 2,188
- *Prompt throughput (raw):* 605 tok/s
- *Generation throughput (raw):* 63.1 tok/s
- *Peak memory:* 60
- *Active memory:* 59
- *Cache memory:* 0.15
- *Model-load active memory:* 58.8
- *Post-cleanup active memory:* 0.00149
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
- *Resolved model revision:* 32dae5c38006e20ac158bc94cd1d5967d19b2652
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--ERNIE-4.5-VL-28B-A3B-Thinking-bf16/snapshots/32dae5c38006e20ac158bc94cd1d5967d19b2652
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "temperature": 0.0, "thinking_budget": 800,
  "thinking_end_token": "&lt;/think&gt;", "thinking_start_token":
  "&lt;think&gt;", "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Alright, let's get to work. I've got the image and the hints, and I need to craft some metadata. First, the title. It needs to be concise and descriptive. "Hampshire Riverbank with Strollers" is a good start, but I can do better. "Strollers on the Riverbank near Winchester" is more specific, and I'll go with that.

Now, the description. I need to weave together the context and the visual elements. The image shows a historic stone house, a river, and people walking. The context mentions the Winchester City Mill and the River Itchen. I'll start with the setting: "A historic stone house and riverbank scene..." Then, I'll add the people and the mill. "A historic stone house and riverbank scene with people walking along a path near the River Itchen and the Winchester City Mill." That's a solid, factual description.

Finally, the keywords. I need to cover the key elements. I'll start with the obvious: Hampshire, England, River Itchen, Winchester, stone house, riverbank, strollers, elderly woman, girls, and the mill. I'll add some more general terms like "scenery," "walkway," "water," and "bridge" to broaden the searchability. I'll also include "National Trust" and "Adobe Stock" as they are in the hints. I'll make sure to avoid anything that isn't clearly visible. I'll refine the list to make sure it's between 10 and 18 keywords.

Okay, here's the final output:

Title: Strollers on the Riverbank near Winchester

Description: A historic stone house and riverbank scene with people walking along a path near the River Itchen and the Winchester City Mill.

Keywords: Hampshire, England, River Itchen, Winchester, stone house, riverbank, strollers, elderly woman, girls, mill, National Trust, scenery, walkway, water, bridge, National Trust, Adobe Stock, blue sky, chimney, clay tiles, rapids, riverbank, stone wall, people.
&lt;/think&gt;

Title: Strollers on the Riverbank near Winchester

Description: A historic stone house and riverbank scene with people walking along a path near the River Itchen and the Winchester City Mill.

Keywords: Hampshire, England, River Itchen, Winchester, stone house, riverbank, strollers, elderly woman, girls, mill, National Trust, scenery, walkway, water, bridge, National Trust, Adobe Stock, blue sky, chimney, clay tiles, rapids, riverbank, stone wall, people.
</pre>

<details>
<summary>Exact raw output</summary>

```text
Alright, let's get to work. I've got the image and the hints, and I need to craft some metadata. First, the title. It needs to be concise and descriptive. "Hampshire Riverbank with Strollers" is a good start, but I can do better. "Strollers on the Riverbank near Winchester" is more specific, and I'll go with that.

Now, the description. I need to weave together the context and the visual elements. The image shows a historic stone house, a river, and people walking. The context mentions the Winchester City Mill and the River Itchen. I'll start with the setting: "A historic stone house and riverbank scene..." Then, I'll add the people and the mill. "A historic stone house and riverbank scene with people walking along a path near the River Itchen and the Winchester City Mill." That's a solid, factual description.

Finally, the keywords. I need to cover the key elements. I'll start with the obvious: Hampshire, England, River Itchen, Winchester, stone house, riverbank, strollers, elderly woman, girls, and the mill. I'll add some more general terms like "scenery," "walkway," "water," and "bridge" to broaden the searchability. I'll also include "National Trust" and "Adobe Stock" as they are in the hints. I'll make sure to avoid anything that isn't clearly visible. I'll refine the list to make sure it's between 10 and 18 keywords.

Okay, here's the final output:

Title: Strollers on the Riverbank near Winchester

Description: A historic stone house and riverbank scene with people walking along a path near the River Itchen and the Winchester City Mill.

Keywords: Hampshire, England, River Itchen, Winchester, stone house, riverbank, strollers, elderly woman, girls, mill, National Trust, scenery, walkway, water, bridge, National Trust, Adobe Stock, blue sky, chimney, clay tiles, rapids, riverbank, stone wall, people.
</think>

Title: Strollers on the Riverbank near Winchester

Description: A historic stone house and riverbank scene with people walking along a path near the River Itchen and the Winchester City Mill.

Keywords: Hampshire, England, River Itchen, Winchester, stone house, riverbank, strollers, elderly woman, girls, mill, National Trust, scenery, walkway, water, bridge, National Trust, Adobe Stock, blue sky, chimney, clay tiles, rapids, riverbank, stone wall, people.
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
- *Model load time:* 7.11s
- *Generation time:* 18.54s
- *Total time:* 25.67s
- *Input validation time:* 6.37e-05
- *Prompt preparation time:* 0.0116
- *First-token latency:* 15.2
- *Cleanup time:* 0.283
- *Prompt tokens:* 6,430
- *Generation tokens:* 108
- *Total tokens:* 6,538
- *Prompt throughput (raw):* 424 tok/s
- *Generation throughput (raw):* 39.9 tok/s
- *Peak memory:* 78
- *Active memory:* 62
- *Cache memory:* 1.4
- *Model-load active memory:* 61.9
- *Post-cleanup active memory:* 0.00274
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

<a id="model-mlx-community-kimi-vl-a3b-thinking-2506-bf16"></a>

### mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

<details>
<summary>Complete evidence: mlx-community/Kimi-VL-A3B-Thinking-2506-bf16</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Conversation-role control tokens remain visible
- *Arch supported by installed mlx-vlm:* yes (model_type kimi_vl)
- *Model load time:* 3.25s
- *Generation time:* 154.36s
- *Total time:* 157.63s
- *Input validation time:* 4.61e-05
- *Prompt preparation time:* 0.00405
- *First-token latency:* 1.49
- *Cleanup time:* 0.145
- *Prompt tokens:* 1,307
- *Generation tokens:* 710
- *Total tokens:* 2,017
- *Prompt throughput (raw):* 879 tok/s
- *Generation throughput (raw):* 4.67 tok/s
- *Peak memory:* 40
- *Active memory:* 33
- *Cache memory:* 5.6
- *Model-load active memory:* 32.8
- *Post-cleanup active memory:* 0.00397
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
- *Resolved model revision:* fb254434d4026bee7aa840dea1c5d59feea8fd48
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Kimi-VL-A3B-Thinking-2506-bf16/snapshots/fb254434d4026bee7aa840dea1c5d59feea8fd48
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* [EOS]

*Readable output:*

<pre class="model-output-readable">
◁think▷Okay, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. The capture date/time is 2026-08-31 15:02:20 UTC+01:00, but I shouldn't claim it's visible. The descriptive hints mention Winchester City Mill, River Itchen, Hampshire, England, people walking, stone wall, chimney, etc. The image shows a historic brick building with a tiled roof, a bridge over a river, people walking, and the river flowing.

For the title, something concise like "Winchester City Mill and River Itchen Scene" works. The description should combine the context and visible elements: "Historic Winchester City Mill by River Itchen with people walking on a stone bridge, brick building, and flowing water." Keywords need to be 10-18 unique terms. Let's list what's visible: Winchester City Mill, River Itchen, England, Hampshire, stone bridge, brick building, tiled roof, chimney, pedestrians, elderly woman, girls, blue sky, Adobe Stock, National Trust, Rapids, Scenery. Wait, the hints mention "Elderly woman" and "Girls" but in the image, there are two women, one might be elderly. Also, "Adobe Stock" and "National Trust" are hints but maybe not visible. Check the image: the building is a National Trust property, so include that. "Adobe Stock" might be a watermark, but the user said not to claim visible details. So maybe omit that. So keywords: Winchester City Mill, River Itchen, England, Hampshire, Stone bridge, Brick building, Tiled roof, Chimney, Pedestrians, Riverbank, Scenery, National Trust. That's 10 keywords. Let me check the hints again. The hints include "Clay tiles" but the roof is tiled, so "Tiled roof" is better. "Rapids" might not be visible, the water is flowing but not necessarily rapids. Maybe remove that. "Elderly woman" and "Girls" – the image has two people, one might be elderly. So include "Elderly woman" and "Girls". So keywords: Winchester City Mill, River Itchen, England, Hampshire, Stone bridge, Brick building, Tiled roof, Chimney, Pedestrians, Riverbank, Scenery, National Trust, Elderly woman, Girls. That's 12. Maybe too many? Wait, the user said 10-18, so 12 is okay. Let me adjust. Maybe "Riverbank" instead of "Scenery". Wait, the hints have "Scenery" as a keyword. So include that. Let me finalize: Winchester City Mill, River Itchen, England, Hampshire, Stone bridge, Brick building, Tiled roof, Chimney, Pedestrians, Riverbank, Scenery, National Trust, Elderly woman, Girls. That's 12. Okay.◁/think▷Title: Winchester City Mill and River Itchen Scene
Description: Historic Winchester City Mill by River Itchen with people walking on a stone bridge, featuring a brick building, tiled roof, and flowing water under a blue sky.
Keywords: Winchester City Mill, River Itchen, England, Hampshire, Stone bridge, Brick building, Tiled roof, Chimney, Pedestrians, Riverbank, Scenery, National Trust, Elderly woman, Girls&lt;|im_assistant|&gt;
</pre>

<details>
<summary>Exact raw output</summary>

```text
◁think▷Okay, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. The capture date/time is 2026-08-31 15:02:20 UTC+01:00, but I shouldn't claim it's visible. The descriptive hints mention Winchester City Mill, River Itchen, Hampshire, England, people walking, stone wall, chimney, etc. The image shows a historic brick building with a tiled roof, a bridge over a river, people walking, and the river flowing. 

For the title, something concise like "Winchester City Mill and River Itchen Scene" works. The description should combine the context and visible elements: "Historic Winchester City Mill by River Itchen with people walking on a stone bridge, brick building, and flowing water." Keywords need to be 10-18 unique terms. Let's list what's visible: Winchester City Mill, River Itchen, England, Hampshire, stone bridge, brick building, tiled roof, chimney, pedestrians, elderly woman, girls, blue sky, Adobe Stock, National Trust, Rapids, Scenery. Wait, the hints mention "Elderly woman" and "Girls" but in the image, there are two women, one might be elderly. Also, "Adobe Stock" and "National Trust" are hints but maybe not visible. Check the image: the building is a National Trust property, so include that. "Adobe Stock" might be a watermark, but the user said not to claim visible details. So maybe omit that. So keywords: Winchester City Mill, River Itchen, England, Hampshire, Stone bridge, Brick building, Tiled roof, Chimney, Pedestrians, Riverbank, Scenery, National Trust. That's 10 keywords. Let me check the hints again. The hints include "Clay tiles" but the roof is tiled, so "Tiled roof" is better. "Rapids" might not be visible, the water is flowing but not necessarily rapids. Maybe remove that. "Elderly woman" and "Girls" – the image has two people, one might be elderly. So include "Elderly woman" and "Girls". So keywords: Winchester City Mill, River Itchen, England, Hampshire, Stone bridge, Brick building, Tiled roof, Chimney, Pedestrians, Riverbank, Scenery, National Trust, Elderly woman, Girls. That's 12. Maybe too many? Wait, the user said 10-18, so 12 is okay. Let me adjust. Maybe "Riverbank" instead of "Scenery". Wait, the hints have "Scenery" as a keyword. So include that. Let me finalize: Winchester City Mill, River Itchen, England, Hampshire, Stone bridge, Brick building, Tiled roof, Chimney, Pedestrians, Riverbank, Scenery, National Trust, Elderly woman, Girls. That's 12. Okay.◁/think▷Title: Winchester City Mill and River Itchen Scene  
Description: Historic Winchester City Mill by River Itchen with people walking on a stone bridge, featuring a brick building, tiled roof, and flowing water under a blue sky.  
Keywords: Winchester City Mill, River Itchen, England, Hampshire, Stone bridge, Brick building, Tiled roof, Chimney, Pedestrians, Riverbank, Scenery, National Trust, Elderly woman, Girls<|im_assistant|>
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
- *Model load time:* 0.63s
- *Generation time:* 1.60s
- *Total time:* 2.23s
- *Input validation time:* 6.83e-05
- *Prompt preparation time:* 0.00399
- *First-token latency:* 0.336
- *Cleanup time:* 0.0777
- *Prompt tokens:* 2,094
- *Generation tokens:* 119
- *Total tokens:* 2,213
- *Prompt throughput (raw):* 6,231 tok/s
- *Generation throughput (raw):* 183 tok/s
- *Peak memory:* 4.0
- *Active memory:* 3.2
- *Cache memory:* 0.06
- *Model-load active memory:* 3.2
- *Post-cleanup active memory:* 0.0041
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
- *Model load time:* 1.43s
- *Generation time:* 5.43s
- *Total time:* 6.86s
- *Input validation time:* 5.12e-05
- *Prompt preparation time:* 0.0011
- *First-token latency:* 2.02
- *Cleanup time:* 0.0932
- *Prompt tokens:* 2,918
- *Generation tokens:* 180
- *Total tokens:* 3,098
- *Prompt throughput (raw):* 1,447 tok/s
- *Generation throughput (raw):* 64.7 tok/s
- *Peak memory:* 13
- *Active memory:* 8.4
- *Cache memory:* 0.60
- *Model-load active memory:* 8.43
- *Post-cleanup active memory:* 0.00539
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
- *Model load time:* 1.06s
- *Generation time:* 3.32s
- *Total time:* 4.40s
- *Input validation time:* 4.85e-05
- *Prompt preparation time:* 0.00565
- *First-token latency:* 0.742
- *Cleanup time:* 0.0877
- *Prompt tokens:* 1,516
- *Generation tokens:* 132
- *Total tokens:* 1,648
- *Prompt throughput (raw):* 2,044 tok/s
- *Generation throughput (raw):* 72.7 tok/s
- *Peak memory:* 8.1
- *Active memory:* 6.5
- *Cache memory:* 0.31
- *Model-load active memory:* 6.45
- *Post-cleanup active memory:* 0.00596
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
- *Model load time:* 0.99s
- *Generation time:* 4.62s
- *Total time:* 5.62s
- *Input validation time:* 5.63e-05
- *Prompt preparation time:* 0.00486
- *First-token latency:* 3.09
- *Cleanup time:* 0.1
- *Prompt tokens:* 4,062
- *Generation tokens:* 152
- *Total tokens:* 4,214
- *Prompt throughput (raw):* 1,315 tok/s
- *Generation throughput (raw):* 229 tok/s
- *Peak memory:* 3.9
- *Active memory:* 2.2
- *Cache memory:* 0.64
- *Model-load active memory:* 2.18
- *Post-cleanup active memory:* 0.0069
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
- *Model load time:* 2.68s
- *Generation time:* 2.70s
- *Total time:* 5.41s
- *Input validation time:* 5.11e-05
- *Prompt preparation time:* 0.019
- *First-token latency:* 0.722
- *Cleanup time:* 0.129
- *Prompt tokens:* 1,262
- *Generation tokens:* 135
- *Total tokens:* 1,397
- *Prompt throughput (raw):* 1,747 tok/s
- *Generation throughput (raw):* 107 tok/s
- *Peak memory:* 24
- *Active memory:* 23
- *Cache memory:* 0.14
- *Model-load active memory:* 23.1
- *Post-cleanup active memory:* 0.00741
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
- *Model load time:* 0.93s
- *Generation time:* 3.15s
- *Total time:* 4.08s
- *Input validation time:* 6.45e-05
- *Prompt preparation time:* 0.00158
- *First-token latency:* 0.298
- *Cleanup time:* 0.0887
- *Prompt tokens:* 1,103
- *Generation tokens:* 128
- *Total tokens:* 1,231
- *Prompt throughput (raw):* 3,702 tok/s
- *Generation throughput (raw):* 56.8 tok/s
- *Peak memory:* 9.5
- *Active memory:* 8.3
- *Cache memory:* 0.52
- *Model-load active memory:* 8.3
- *Post-cleanup active memory:* 0.00747
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
- *Model load time:* 1.93s
- *Generation time:* 56.78s
- *Total time:* 58.72s
- *Input validation time:* 5.59e-05
- *Prompt preparation time:* 0.00215
- *First-token latency:* 54.2
- *Cleanup time:* 0.13
- *Prompt tokens:* 16,520
- *Generation tokens:* 133
- *Total tokens:* 16,653
- *Prompt throughput (raw):* 305 tok/s
- *Generation throughput (raw):* 87.1 tok/s
- *Peak memory:* 23
- *Active memory:* 18
- *Cache memory:* 2.3
- *Model-load active memory:* 18.3
- *Post-cleanup active memory:* 0.00809
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
- *Model load time:* 1.41s
- *Generation time:* 54.69s
- *Total time:* 56.12s
- *Input validation time:* 5.4e-05
- *Prompt preparation time:* 0.00169
- *First-token latency:* 52.2
- *Cleanup time:* 0.098
- *Prompt tokens:* 16,536
- *Generation tokens:* 132
- *Total tokens:* 16,668
- *Prompt throughput (raw):* 317 tok/s
- *Generation throughput (raw):* 92.1 tok/s
- *Peak memory:* 10.0
- *Active memory:* 6.0
- *Cache memory:* 1.2
- *Model-load active memory:* 5.96
- *Post-cleanup active memory:* 0.00913
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
- *Model load time:* 0.71s
- *Generation time:* 2.02s
- *Total time:* 2.73s
- *Input validation time:* 7.7e-05
- *Prompt preparation time:* 0.00278
- *First-token latency:* 0.407
- *Cleanup time:* 0.0856
- *Prompt tokens:* 1,401
- *Generation tokens:* 84
- *Total tokens:* 1,485
- *Prompt throughput (raw):* 3,440 tok/s
- *Generation throughput (raw):* 125 tok/s
- *Peak memory:* 5.4
- *Active memory:* 4.5
- *Cache memory:* 0.35
- *Model-load active memory:* 4.5
- *Post-cleanup active memory:* 0.0103
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
- *Model load time:* 7.55s
- *Generation time:* 18.02s
- *Total time:* 25.61s
- *Input validation time:* 6.28e-05
- *Prompt preparation time:* 0.0218
- *First-token latency:* 14.6
- *Cleanup time:* 0.271
- *Prompt tokens:* 3,464
- *Generation tokens:* 114
- *Total tokens:* 3,578
- *Prompt throughput (raw):* 237 tok/s
- *Generation throughput (raw):* 46.6 tok/s
- *Peak memory:* 70
- *Active memory:* 63
- *Cache memory:* 0.39
- *Model-load active memory:* 63
- *Post-cleanup active memory:* 0.0105
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
- *Model load time:* 1.18s
- *Generation time:* 19.27s
- *Total time:* 20.47s
- *Input validation time:* 7.43e-05
- *Prompt preparation time:* 0.00466
- *First-token latency:* 15.9
- *Cleanup time:* 0.104
- *Prompt tokens:* 16,531
- *Generation tokens:* 136
- *Total tokens:* 16,667
- *Prompt throughput (raw):* 1,043 tok/s
- *Generation throughput (raw):* 57.8 tok/s
- *Peak memory:* 13
- *Active memory:* 9.5
- *Cache memory:* 1.4
- *Model-load active memory:* 9.46
- *Post-cleanup active memory:* 0.0108
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

<a id="model-mlx-community-gemma-3-27b-it-qat-4bit"></a>

### mlx-community/gemma-3-27b-it-qat-4bit

<details>
<summary>Complete evidence: mlx-community/gemma-3-27b-it-qat-4bit</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Keyword list has 19 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type gemma3)
- *Model load time:* 2.28s
- *Generation time:* 5.75s
- *Total time:* 8.05s
- *Input validation time:* 5.12e-05
- *Prompt preparation time:* 0.00459
- *First-token latency:* 0.988
- *Cleanup time:* 0.131
- *Prompt tokens:* 569
- *Generation tokens:* 128
- *Total tokens:* 697
- *Prompt throughput (raw):* 576 tok/s
- *Generation throughput (raw):* 31.2 tok/s
- *Peak memory:* 17
- *Active memory:* 16
- *Cache memory:* 0.43
- *Model-load active memory:* 16.1
- *Post-cleanup active memory:* 0.0114
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
- *Generation time:* 5.13s
- *Total time:* 6.80s
- *Input validation time:* 6.08e-05
- *Prompt preparation time:* 0.00419
- *First-token latency:* 1.7
- *Cleanup time:* 0.107
- *Prompt tokens:* 3,111
- *Generation tokens:* 110
- *Total tokens:* 3,221
- *Prompt throughput (raw):* 1,829 tok/s
- *Generation throughput (raw):* 38.9 tok/s
- *Peak memory:* 16
- *Active memory:* 13
- *Cache memory:* 0.59
- *Model-load active memory:* 13.5
- *Post-cleanup active memory:* 0.0134
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

<a id="model-qwen-qwen3-vl-2b-instruct"></a>

### Qwen/Qwen3-VL-2B-Instruct

<details>
<summary>Complete evidence: Qwen/Qwen3-VL-2B-Instruct</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Response repeats the same text; Response appears cut off at
  the token limit; Title has 4 words (requested 5-10); Keyword list has 327
  terms (requested 10-18); Duplicate keywords: national trust, pedestrians,
  riverbank, stone wall, brick building, blue sky, chimney, clay tiles,
  elderly woman, girls, rapids, scenery, sitting, people, river
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Model load time:* 0.67s
- *Generation time:* 25.20s
- *Total time:* 25.88s
- *Input validation time:* 3.67e-05
- *Prompt preparation time:* 0.00964
- *First-token latency:* 13.4
- *Cleanup time:* 0.0839
- *Prompt tokens:* 16,520
- *Generation tokens:* 1,000
- *Total tokens:* 17,520
- *Prompt throughput (raw):* 1,229 tok/s
- *Generation throughput (raw):* 92.2 tok/s
- *Peak memory:* 8.4
- *Active memory:* 4.3
- *Cache memory:* 2.4
- *Model-load active memory:* 4.26
- *Post-cleanup active memory:* 0.000443
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1363
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 89644892e4d85e24eaac8bacfd4f463576704203
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: River Itchen, Hampshire, England

Description: A historic watermill in Hampshire, England, with a stone wall and brick building, is situated over the River Itchen. Pedestrians walk along a bridge, and an elderly woman sits on a bench in front of the mill. The scene is set under a blue sky with white clouds, and the river flows with rapids near the bridge.

Keywords: River Itchen, Hampshire, England, Mill, National Trust, Pedestrians, Riverbank, Stone wall, Brick building, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Rapids, Scenery, Sitting, People, River, National Trust, Pedestrians, Riverbank, Stone wall, Brick building, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Rapids, Scenery, Sitting, People, River, National Trust, Pedestrians, Riverbank, Stone wall, Brick building, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Rapids, Scenery, Sitting, People, River, National Trust, Pedestrians, Riverbank, Stone wall, Brick building, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Rapids, Scenery, Sitting, People, River, National Trust, Pedestrians, Riverbank, Stone wall, Brick building, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Rapids, Scenery, Sitting, People, River, National Trust, Pedestrians, Riverbank, Stone wall, Brick building, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Rapids, Scenery, Sitting, People, River, National Trust, Pedestrians, Riverbank, Stone wall, Brick building, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Rapids, Scenery, Sitting, People, River, National Trust, Pedestrians, Riverbank, Stone wall, Brick building, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Rapids, Scenery, Sitting, People, River, National Trust, Pedestrians, Riverbank, Stone wall, Brick building, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Rapids, Scenery, Sitting, People, River, National Trust, Pedestrians, Riverbank, Stone wall, Brick building, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Rapids, Scenery, Sitting, People, River, National Trust, Pedestrians, Riverbank, Stone wall, Brick building, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Rapids, Scenery, Sitting, People, River, National Trust, Pedestrians, Riverbank, Stone wall, Brick building, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Rapids, Scenery, Sitting, People, River, National Trust, Pedestrians, Riverbank, Stone wall, Brick building, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Rapids, Scenery, Sitting, People, River, National Trust, Pedestrians, Riverbank, Stone wall, Brick building, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Rapids, Scenery, Sitting, People, River, National Trust, Pedestrians, Riverbank, Stone wall, Brick building, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Rapids, Scenery, Sitting, People, River, National Trust, Pedestrians, Riverbank, Stone wall, Brick building, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Rapids, Scenery, Sitting, People, River, National Trust, Pedestrians, Riverbank, Stone wall, Brick building, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Rapids, Scenery, Sitting, People, River, National Trust, Pedestrians, Riverbank, Stone wall, Brick building, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Rapids, Scenery, Sitting, People, River, National Trust, Pedestrians, Riverbank, Stone wall, Brick building, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Rapids, Scenery, Sitting, People, River, National Trust, Pedestrians, Riverbank, Stone wall, Brick building, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Rapids, Scenery, Sitting, People, River, National Trust, Pedestrians, Riverbank, Stone wall, Brick building, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Rapids, Scenery, Sitting, People, River, National Trust, Pedestrians, Riverbank, Stone wall, Brick building, Blue sky, Chimney, Clay tiles
</pre>

</details>

---

<a id="model-jinaai-jina-vlm-mlx"></a>

### jinaai/jina-vlm-mlx

<details>
<summary>Complete evidence: jinaai/jina-vlm-mlx</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* none
- *Observations:* Missing or empty fields: Keywords
- *Arch supported by installed mlx-vlm:* yes (model_type jvlm via jina_vlm)
- *Model load time:* 0.58s
- *Generation time:* 4.02s
- *Total time:* 4.61s
- *Input validation time:* 5.02e-05
- *Prompt preparation time:* 0.00211
- *First-token latency:* 0.946
- *Cleanup time:* 0.0751
- *Prompt tokens:* 2,050
- *Generation tokens:* 294
- *Total tokens:* 2,344
- *Prompt throughput (raw):* 2,168 tok/s
- *Generation throughput (raw):* 139 tok/s
- *Peak memory:* 3.7
- *Active memory:* 2.1
- *Cache memory:* 0.33
- *Model-load active memory:* 2.11
- *Post-cleanup active memory:* 0.000755
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1303
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.jina_vlm.processing_jinavlm.JinaVLMProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* a987631a01dc554a787d87a45fb01fb48f8aaca4
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--jinaai--jina-vlm-mlx/snapshots/a987631a01dc554a787d87a45fb01fb48f8aaca4
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|endoftext\|>

*Readable output:*

<pre class="model-output-readable">
Title:
Historic Winchester City Mill

Description:
The historic Winchester City Mill, an ancient watermill situated over the River Itchen in Winchester, Hampshire, England, is captured in this image. Pedestrians walk along the pathway, relaxing in front of the mill, which is adorned with Adobe Stock, Any Vision, Arch, Blue sky, Chimney, Clay tiles, Elderly woman, England, Girls, Hampshire, Mill, National Trust, Pedestrians, People, Rapids, River Itchen, Riverbank, Scenery, Sitting, Stone wall. The mill is constructed with Adobe Stock, Any Vision, Arch, Blue sky, Chimney, Clay tiles, Elderly woman, England, Girls, Hampshire, Mill, National Trust, Pedestrians, People, Rapids, River Itchen, Riverbank, Scenery, Sitting, Stone wall. The scene is set under a Blue sky, with the mill's chimney and clay tiles adding to its historic charm. Pedestrians walk along the pathway, relaxing in front of the mill, which is adorned with Adobe Stock, Any Vision, Arch, Blue sky, Chimney, Clay tiles, Elderly woman, England, Girls, Hampshire, Mill, National Trust, Pedestrians, People, Rapids, River Itchen, Riverbank, Scenery, Sitting, Stone wall. The scene is set under a Blue sky, with the mill's chimney and clay tiles adding to its historic charm.
</pre>

</details>

---

<a id="model-mlx-community-apriel-15-15b-thinker-6bit-mlx"></a>

### mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX

<details>
<summary>Complete evidence: mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* none
- *Observations:* Missing or empty fields: Description, Keywords, Title;
  Response repeats the task instructions instead of only returning the
  requested fields; Extra text appears before the Title field; Response
  appears cut off at the token limit
- *Arch supported by installed mlx-vlm:* yes (model_type pixtral)
- *Model load time:* 1.58s
- *Generation time:* 26.40s
- *Total time:* 27.99s
- *Input validation time:* 4.85e-05
- *Prompt preparation time:* 0.00801
- *First-token latency:* 2
- *Cleanup time:* 0.102
- *Prompt tokens:* 3,202
- *Generation tokens:* 1,000
- *Total tokens:* 4,202
- *Prompt throughput (raw):* 1,599 tok/s
- *Generation throughput (raw):* 42.0 tok/s
- *Peak memory:* 15
- *Active memory:* 13
- *Cache memory:* 0.87
- *Model-load active memory:* 12.6
- *Post-cleanup active memory:* 0.00102
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1665
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.pixtral.processing_pixtral.PixtralProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 24cb8fef6052e8d6e0dd7d467cf2d3db2dec19b2
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Apriel-1.5-15b-Thinker-6bit-MLX/snapshots/24cb8fef6052e8d6e0dd7d467cf2d3db2dec19b2
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Here are my reasoning steps:
We need to produce British-English catalogue metadata from the image and supplied context. The image shows a historic watermill (Winchester City Mill) over a river (River Itchen). The building is a brick and stone structure with a tiled roof, a chimney, windows with lattice, a stone wall, a bridge over the water, a pathway with people walking, a woman in a blue dress, a man with a camera, some people sitting on a bench. The river has rapids. The sky is partly cloudy with blue sky. The location is Winchester, Hampshire, England. The building is likely a National Trust property. The date/time is 2026-08-31 15:02:20 UTC+01:00. The description should combine relevant context with main visible subject, setting, action, lighting, distinctive details. Title: 5-10 words. Description: 1-2 sentences. Keywords: 10-18 unique, comma-separated.

We need to ensure British English. Use "mill", "river", "chimney", "clay tiles", "stone wall", "pedestrians", "elderly woman", "girls", "rapids", "scenery", "sitting", "National Trust", "Winchester City Mill", "River Itchen", "Hampshire", "England", "Adobe Stock", "Any Vision". Also "arch"? There's an arch of the bridge. "Blue sky". "Stone wall". "Clay tiles". "Chimney". "Pedestrians". "People". "Riverbank". "Scenery". "Sitting". "National Trust". "Winchester". "Hampshire". "England". "River Itchen". "Rapids". "Bridge". "Pathway". "Elderly woman". "Girls". "Adobe Stock". "Any Vision". "Arch". "Blue sky". "Chimney". "Clay tiles". "Elderly woman". "England". "Girls". "Hampshire". "Mill". "National Trust". "Pedestrians". "People". "Rapids". "River Itchen". "Riverbank". "Scenery". "Sitting". "Stone wall". That's many. Need 10-18 unique keywords. We'll pick 14.

Now the description: "Visitors stroll along a stone pathway beside the historic Winchester City Mill, a 16th‑century watermill with a red‑clay tiled roof and a brick‑and‑stone façade, while an elderly woman in a blue dress walks past and a couple sit on a bench under a bright, partly‑cloudy sky." That's one sentence. Could add second sentence about the river rapids. "The River Itchen flows beneath a stone bridge, its rapids visible in the foreground, framing the mill’s historic setting." That's two sentences.

Title: "Winchester City Mill by River Itchen". That's 5 words. Or "Historic Winchester City Mill on River Itchen". That's 7 words. Title must be 5-10 words. We'll do "Historic Winchester City Mill on River Itchen". That's 7 words.

Now keywords: we need 10-18 unique, comma-separated. Let's list: Winchester City Mill, River Itchen, Hampshire, England, National Trust, historic watermill, red clay tiles, chimney, stone wall, stone bridge, rapids, pedestrians, elderly woman, blue sky, pathway, sitting, Adobe Stock, Any Vision. That's 18. Ensure each is unique. Let's count: 1 Winchester City Mill, 2 River Itchen, 3 Hampshire, 4 England, 5 National Trust, 6 historic watermill, 7 red clay tiles, 8 chimney, 9 stone wall, 10 stone bridge, 11 rapids, 12 pedestrians, 13 elderly woman, 14 blue sky, 15 pathway, 16 sitting, 17 Adobe Stock, 18 Any Vision. That's 18, within range.

Make sure we follow the format: Title: line, Description: line(s), Keywords: line. No extra text.

Now check for any conflicts: The hint says "Visitors walk along the pathway and relax in front of the historic Winchester City Mill, an ancient watermill situated over the River Itchen in Winchester, Hampshire, England." The image shows a stone wall and a bridge, a pathway, people walking, a woman in blue, a couple sitting. So description matches.

The hint includes "Girls". The image does not show girls explicitly, but there is
</pre>

</details>

---

<a id="model-mlx-community-fastvlm-05b-bf16"></a>

### mlx-community/FastVLM-0.5B-bf16

<details>
<summary>Complete evidence: mlx-community/FastVLM-0.5B-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* none
- *Observations:* Missing or empty fields: Title, Description, Keywords
- *Arch supported by installed mlx-vlm:* yes (model_type llava_qwen2 via
  fastvlm)
- *Model load time:* 0.64s
- *Generation time:* 1.78s
- *Total time:* 2.43s
- *Input validation time:* 5.5e-05
- *Prompt preparation time:* 0.00217
- *First-token latency:* 0.106
- *Cleanup time:* 0.0794
- *Prompt tokens:* 307
- *Generation tokens:* 155
- *Total tokens:* 462
- *Prompt throughput (raw):* 2,892 tok/s
- *Generation throughput (raw):* 359 tok/s
- *Peak memory:* 2.2
- *Active memory:* 1.2
- *Cache memory:* 0.03
- *Model-load active memory:* 1.25
- *Post-cleanup active memory:* 0.0018
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1386
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.fastvlm.processing.FastVLMProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 81ffe929046666c43de53691147b1669ba0f3a4c
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--FastVLM-0.5B-bf16/snapshots/81ffe929046666c43de53691147b1669ba0f3a4c
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
A tranquil scene captures visitors enjoying a walk along the historic Winchester City Mill, a picturesque watermill nestled by the River Itchen in Winchester, Hampshire, England. The image, taken on 2026-08-31 15:02:20 UTC+01:00, showcases the blend of natural and architectural beauty, with the mill's distinctive red clay tiles and the surrounding lush greenery. The river's gentle flow and the clear blue sky above add to the serene atmosphere, while the presence of visitors highlights the mill's role as a popular tourist attraction. The image, sourced from Adobe Stock, captures the essence of this iconic landmark, offering a glimpse into the rich history and cultural significance of the Winchester City Mill.
</pre>

</details>

---

<a id="model-mlx-community-glm-41v-9b-thinking-8bit"></a>

### mlx-community/GLM-4.1V-9B-Thinking-8bit

<details>
<summary>Complete evidence: mlx-community/GLM-4.1V-9B-Thinking-8bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Response repeats the task instructions instead of only
  returning the requested fields; Extra text appears before the Title field;
  Response appears cut off at the token limit; Internal reasoning block
  appears incomplete; Keyword list has 42 terms (requested 10-18); Duplicate
  keywords: blue sky, chimney, clay tiles, elderly woman, girls, hampshire,
  mill, national trust, pedestrians, people, river itchen, riverbank, scenery,
  sitting
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v)
- *Model load time:* 1.68s
- *Generation time:* 27.70s
- *Total time:* 29.38s
- *Input validation time:* 6.72e-05
- *Prompt preparation time:* 0.00274
- *First-token latency:* 5.49
- *Cleanup time:* 0.103
- *Prompt tokens:* 6,426
- *Generation tokens:* 1,000
- *Total tokens:* 7,426
- *Prompt throughput (raw):* 1,169 tok/s
- *Generation throughput (raw):* 46.4 tok/s
- *Peak memory:* 13
- *Active memory:* 12
- *Cache memory:* 0.44
- *Model-load active memory:* 11.8
- *Post-cleanup active memory:* 0.00211
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1348
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.glm4v.processing.Glm46VProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 9677807f106500eb7690391c27645d59f6855cfb
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--GLM-4.1V-9B-Thinking-8bit/snapshots/9677807f106500eb7690391c27645d59f6855cfb
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|endoftext\|>

*Readable output:*

<pre class="model-output-readable">
&lt;think&gt;Got it, let's tackle this task step by step. First, the title needs to be 5-10 words, concrete. The image shows a historic mill by a river with people, so maybe "Winchester City Mill by River Itchen".

Next, the description: need to combine context (capture date, location) with main subject (mill), setting (river, historic), action (visitors walk, relax), lighting (blue sky), distinctive details (brick, stone, water). So something like "Historic Winchester City Mill by River Itchen with visitors on a pathway, under a blue sky, showcasing brick and stone architecture." Wait, need to make sure it's 1-2 sentences. Let's check the context: capture date is 2026-08-31, location is Winchester, Hampshire, England, mill, people walking, sitting, river. So description: "Visitors walk and relax near the historic Winchester City Mill, a brick and stone structure over the River Itchen, under a bright blue sky."

Then keywords: need 10-18 unique, comma-separated. From hints: Adobe Stock, Any Vision (but maybe not visible, but context says National Trust? Wait, the hints include National Trust, so maybe include. Also, the elements: arch, blue sky, chimney, clay tiles, elderly woman, girls, Hampshire, mill, National Trust, pedestrians, people, river Itchen, riverbank, scenery, sitting, stone wall. Let's list them: "Arch, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Hampshire, Mill, National Trust, Pedestrians, People, River Itchen, Riverbank, Scenery, Sitting, Stone wall" – that's 16, which is within 10-18.

Wait, let's check each part. Title: "Winchester City Mill by River Itchen" (7 words, concrete). Description: "Visitors walk and relax near the historic Winchester City Mill, a brick and stone structure over the River Itchen, under a bright blue sky, with distinctive architectural features like a chimney and stone arches." Wait, need to make sure it's 1-2 sentences. Let's make it concise. "Visitors walk and relax near the historic Winchester City Mill, a brick and stone structure over the River Itchen, under a bright blue sky." That's one sentence. Then keywords: combine all relevant hints. Let's list all hints: Adobe Stock, Any Vision, Arch, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, England, Hampshire, Mill, National Trust, Pedestrians, People, Rapids, River Itchen, Riverbank, Scenery, Sitting, Stone wall. Remove duplicates: England is already in Hampshire, so keep Hampshire. Remove Adobe Stock/Any Vision if not visible, but context says "National Trust" is a hint, so include. So keywords: Arch, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Hampshire, Mill, National Trust, Pedestrians, People, River Itchen, Riverbank, Scenery, Sitting, Stone wall. That's 16, which is good.

Now, let's finalize each section:

Title: "Winchester City Mill by River Itchen" (5 words, concrete)

Description: "Visitors walk and relax near the historic Winchester City Mill, a brick and stone structure over the River Itchen, under a bright blue sky." (1 sentence, combines context and visible details)

Keywords: Arch, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Hampshire, Mill, National Trust, Pedestrians, People, River Itchen, Riverbank, Scenery, Sitting, Stone wall (comma-separated, unique, 16 words)

Wait, check if "Rapids" is visible? The water has some movement, maybe rapids? The description hint says "Rapids" but the image might have water flow. But the description hint says "Rapids" as a hint, but maybe the image shows water flow. But the user says "retain details supported by the image, correct conflicts, and add important visible details." The image has water flowing under the mill, so maybe "Rapids" is a detail. But let's check the hints again. The hints include "Rapids" as a keyword. So maybe include "Rapids" if visible. The water under the mill has some movement, so maybe "Rapids" is a detail. So add "Rapids" to keywords. Then keywords would be Arch, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Hampshire, Mill, National Trust, Pedestrians, People, River Itchen, Riverbank, Rapids, Scenery, Sitting, Stone wall. That's
</pre>

<details>
<summary>Exact raw output</summary>

```text
<think>Got it, let's tackle this task step by step. First, the title needs to be 5-10 words, concrete. The image shows a historic mill by a river with people, so maybe "Winchester City Mill by River Itchen". 

Next, the description: need to combine context (capture date, location) with main subject (mill), setting (river, historic), action (visitors walk, relax), lighting (blue sky), distinctive details (brick, stone, water). So something like "Historic Winchester City Mill by River Itchen with visitors on a pathway, under a blue sky, showcasing brick and stone architecture." Wait, need to make sure it's 1-2 sentences. Let's check the context: capture date is 2026-08-31, location is Winchester, Hampshire, England, mill, people walking, sitting, river. So description: "Visitors walk and relax near the historic Winchester City Mill, a brick and stone structure over the River Itchen, under a bright blue sky." 

Then keywords: need 10-18 unique, comma-separated. From hints: Adobe Stock, Any Vision (but maybe not visible, but context says National Trust? Wait, the hints include National Trust, so maybe include. Also, the elements: arch, blue sky, chimney, clay tiles, elderly woman, girls, Hampshire, mill, National Trust, pedestrians, people, river Itchen, riverbank, scenery, sitting, stone wall. Let's list them: "Arch, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Hampshire, Mill, National Trust, Pedestrians, People, River Itchen, Riverbank, Scenery, Sitting, Stone wall" – that's 16, which is within 10-18. 

Wait, let's check each part. Title: "Winchester City Mill by River Itchen" (7 words, concrete). Description: "Visitors walk and relax near the historic Winchester City Mill, a brick and stone structure over the River Itchen, under a bright blue sky, with distinctive architectural features like a chimney and stone arches." Wait, need to make sure it's 1-2 sentences. Let's make it concise. "Visitors walk and relax near the historic Winchester City Mill, a brick and stone structure over the River Itchen, under a bright blue sky." That's one sentence. Then keywords: combine all relevant hints. Let's list all hints: Adobe Stock, Any Vision, Arch, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, England, Hampshire, Mill, National Trust, Pedestrians, People, Rapids, River Itchen, Riverbank, Scenery, Sitting, Stone wall. Remove duplicates: England is already in Hampshire, so keep Hampshire. Remove Adobe Stock/Any Vision if not visible, but context says "National Trust" is a hint, so include. So keywords: Arch, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Hampshire, Mill, National Trust, Pedestrians, People, River Itchen, Riverbank, Scenery, Sitting, Stone wall. That's 16, which is good. 

Now, let's finalize each section:

Title: "Winchester City Mill by River Itchen" (5 words, concrete)

Description: "Visitors walk and relax near the historic Winchester City Mill, a brick and stone structure over the River Itchen, under a bright blue sky." (1 sentence, combines context and visible details)

Keywords: Arch, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Hampshire, Mill, National Trust, Pedestrians, People, River Itchen, Riverbank, Scenery, Sitting, Stone wall (comma-separated, unique, 16 words)

Wait, check if "Rapids" is visible? The water has some movement, maybe rapids? The description hint says "Rapids" but the image might have water flow. But the description hint says "Rapids" as a hint, but maybe the image shows water flow. But the user says "retain details supported by the image, correct conflicts, and add important visible details." The image has water flowing under the mill, so maybe "Rapids" is a detail. But let's check the hints again. The hints include "Rapids" as a keyword. So maybe include "Rapids" if visible. The water under the mill has some movement, so maybe "Rapids" is a detail. So add "Rapids" to keywords. Then keywords would be Arch, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Hampshire, Mill, National Trust, Pedestrians, People, River Itchen, Riverbank, Rapids, Scenery, Sitting, Stone wall. That's 
```

</details>

</details>

---

<a id="model-mlx-community-glm-46v-flash-mxfp4"></a>

### mlx-community/GLM-4.6V-Flash-mxfp4

<details>
<summary>Complete evidence: mlx-community/GLM-4.6V-Flash-mxfp4</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Response repeats the same text; Response appears cut off at
  the token limit; Title has 3 words (requested 5-10); Keyword list has 230
  terms (requested 10-18); Duplicate keywords: hampshire, england, rapids,
  arched doorway, tourist attraction, historic watermill, diamond paned
  windows, weathered roof tiles, red brick chimney, stone archway over river,
  mossy stone wall, green ivy, traditional english mill architecture, historic
  watermill on river itchen, winchester, national trust property, scenic
  riverfront, historic mill with water flow, traditional english architecture,
  stone and brick building
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v)
- *Model load time:* 1.34s
- *Generation time:* 19.44s
- *Total time:* 20.79s
- *Input validation time:* 6.13e-05
- *Prompt preparation time:* 0.00843
- *First-token latency:* 5.71
- *Cleanup time:* 0.0888
- *Prompt tokens:* 6,430
- *Generation tokens:* 1,000
- *Total tokens:* 7,430
- *Prompt throughput (raw):* 1,126 tok/s
- *Generation throughput (raw):* 76.4 tok/s
- *Peak memory:* 8.4
- *Active memory:* 6.8
- *Cache memory:* 0.45
- *Model-load active memory:* 6.78
- *Post-cleanup active memory:* 0.00243
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1372
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.glm4v.processing.Glm46VProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 773591fa7388b5f0db2f5ec11ed9dc3a23779f1b
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--GLM-4.6V-Flash-mxfp4/snapshots/773591fa7388b5f0db2f5ec11ed9dc3a23779f1b
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|endoftext\|>

*Readable output:*

<pre class="model-output-readable">
Title: Winchester City Mill
Description: The historic Winchester City Mill, an ancient watermill over the River Itchen, is shown with visitors walking along the pathway and an elderly woman sitting nearby.
Keywords: Winchester City Mill, River Itchen, watermill, historic building, pedestrians, elderly woman, stone wall, clay tiles, blue sky, chimney, Hampshire, England, National Trust, riverbank, scenery, rapids, sitting, walking, pathway, water, brick building, arched doorway, window with diamond panes, outdoor lamp, greenery, moss, stone bridge, flowing water, reflection, cloudy sky, historic architecture, traditional building, riverfront, tourist attraction, English countryside, historic mill, water-powered mill, river crossing, stone archway, brick chimney, tiled roof, weathered stone, mossy stone, water flow, rapids, riverbank scenery, historic watermill, traditional English mill, riverfront historic building, stone and brick architecture, water-powered historic mill, river Itchen watermill, Winchester historic mill, National Trust managed historic site, traditional English watermill, historic riverfront architecture, stone bridge over river, water flow over weir, historic mill building with arched doorway, diamond-paned windows, brick and stone construction, weathered roof tiles, red brick chimney, stone archway over river, mossy stone wall, green ivy, traditional English mill architecture, historic watermill on River Itchen, Winchester, Hampshire, England, National Trust property, tourist attraction, scenic riverfront, historic mill with water flow, traditional English architecture, stone and brick building, arched doorway, diamond-paned windows, red brick chimney, weathered roof tiles, stone archway over river, mossy stone wall, green ivy, traditional English mill architecture, historic watermill on River Itchen, Winchester, Hampshire, England, National Trust property, tourist attraction, scenic riverfront, historic mill with water flow, traditional English architecture, stone and brick building, arched doorway, diamond-paned windows, red brick chimney, weathered roof tiles, stone archway over river, mossy stone wall, green ivy, traditional English mill architecture, historic watermill on River Itchen, Winchester, Hampshire, England, National Trust property, tourist attraction, scenic riverfront, historic mill with water flow, traditional English architecture, stone and brick building, arched doorway, diamond-paned windows, red brick chimney, weathered roof tiles, stone archway over river, mossy stone wall, green ivy, traditional English mill architecture, historic watermill on River Itchen, Winchester, Hampshire, England, National Trust property, tourist attraction, scenic riverfront, historic mill with water flow, traditional English architecture, stone and brick building, arched doorway, diamond-paned windows, red brick chimney, weathered roof tiles, stone archway over river, mossy stone wall, green ivy, traditional English mill architecture, historic watermill on River Itchen, Winchester, Hampshire, England, National Trust property, tourist attraction, scenic riverfront, historic mill with water flow, traditional English architecture, stone and brick building, arched doorway, diamond-paned windows, red brick chimney, weathered roof tiles, stone archway over river, mossy stone wall, green ivy, traditional English mill architecture, historic watermill on River Itchen, Winchester, Hampshire, England, National Trust property, tourist attraction, scenic riverfront, historic mill with water flow, traditional English architecture, stone and brick building, arched doorway, diamond-paned windows, red brick chimney, weathered roof tiles, stone archway over river, mossy stone wall, green ivy, traditional English mill architecture, historic watermill on River Itchen, Winchester, Hampshire, England, National Trust property, tourist attraction, scenic riverfront, historic mill with water flow, traditional English architecture, stone and brick building, arched doorway, diamond-paned windows, red brick chimney, weathered roof tiles, stone archway over river, mossy stone wall, green ivy, traditional English mill architecture, historic watermill on River Itchen, Winchester, Hampshire, England, National Trust property, tourist attraction, scenic riverfront, historic mill with water flow, traditional English architecture, stone and brick building, arched doorway, diamond-paned windows, red brick chimney, weathered roof tiles, stone archway over river, mossy stone wall, green ivy, traditional English mill architecture, historic watermill on River Itchen, Winchester, Hampshire, England, National Trust property, tourist attraction, scenic riverfront, historic mill with water flow, traditional English architecture, stone and brick building, arched doorway, diamond-paned windows, red brick chimney, weathered roof tiles, stone archway over river, mossy stone wall, green ivy, traditional English mill architecture, historic watermill
</pre>

</details>

---

<a id="model-mlx-community-minicpm-v-46-8bit"></a>

### mlx-community/MiniCPM-V-4.6-8bit

<details>
<summary>Complete evidence: mlx-community/MiniCPM-V-4.6-8bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* none
- *Observations:* Missing or empty fields: Title, Description; Extra text
  appears before the Title field
- *Arch supported by installed mlx-vlm:* yes (model_type minicpmv4_6)
- *Model load time:* 0.94s
- *Generation time:* 1.26s
- *Total time:* 2.21s
- *Input validation time:* 3.55e-05
- *Prompt preparation time:* 0.00955
- *First-token latency:* 0.249
- *Cleanup time:* 0.0898
- *Prompt tokens:* 907
- *Generation tokens:* 70
- *Total tokens:* 977
- *Prompt throughput (raw):* 3,641 tok/s
- *Generation throughput (raw):* 288 tok/s
- *Peak memory:* 3.3
- *Active memory:* 2.3
- *Cache memory:* 0.05
- *Model-load active memory:* 2.32
- *Post-cleanup active memory:* 0.00487
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1359
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.minicpmv4_6.processing_minicpmv4_6.MiniCPMVProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 03721395f6b82cd000cc74cde28fcff8abd9a04c
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--MiniCPM-V-4.6-8bit/snapshots/03721395f6b82cd000cc74cde28fcff8abd9a04c
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Visitors walk near historic mill over river under bright sky.
Lighting is sunny with clear blue skies above the old building.

Keywords: historic mill, river Itchen, stone bridge, brick building, blue sky, pedestrians, elderly woman, girls, England, Hampshire, watermill, stone wall, Rapids, Scenery, National Trust
</pre>

<details>
<summary>Exact raw output</summary>

```text
Visitors walk near historic mill over river under bright sky.  
Lighting is sunny with clear blue skies above the old building.

Keywords: historic mill, river Itchen, stone bridge, brick building, blue sky, pedestrians, elderly woman, girls, England, Hampshire, watermill, stone wall, Rapids, Scenery, National Trust
```

</details>

</details>

---

<a id="model-mlx-community-gemma-3n-e4b-it-bf16"></a>

### mlx-community/gemma-3n-E4B-it-bf16

<details>
<summary>Complete evidence: mlx-community/gemma-3n-E4B-it-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* none
- *Observations:* Missing or empty fields: Title, Description; Extra text
  appears before the Title field
- *Arch supported by installed mlx-vlm:* yes (model_type gemma3n)
- *Model load time:* 2.29s
- *Generation time:* 4.43s
- *Total time:* 6.81s
- *Input validation time:* 6.43e-05
- *Prompt preparation time:* 0.00468
- *First-token latency:* 0.361
- *Cleanup time:* 0.126
- *Prompt tokens:* 568
- *Generation tokens:* 167
- *Total tokens:* 735
- *Prompt throughput (raw):* 1,572 tok/s
- *Generation throughput (raw):* 48.8 tok/s
- *Peak memory:* 17
- *Active memory:* 16
- *Cache memory:* 0.05
- *Model-load active memory:* 15.7
- *Post-cleanup active memory:* 0.0119
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1348
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.gemma3n.processing_gemma3n.Gemma3nProcessor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* d9c02d0b2fa8cf26c1cb5dd9e756db59cdbe8a4a
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--gemma-3n-E4B-it-bf16/snapshots/d9c02d0b2fa8cf26c1cb5dd9e756db59cdbe8a4a
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
A picturesque scene unfolds at the historic Winchester City Mill, a charming watermill nestled along the River Itchen in Winchester, Hampshire, England. The mill's weathered brick and stone facade, topped with a terracotta tiled roof, exudes a sense of timelessness. Two pedestrians stroll along a stone walkway in front of the mill, adding a touch of life to the tranquil setting. The river flows gently beneath the walkway, its waters reflecting the clear blue sky dotted with fluffy white clouds. The scene is bathed in natural daylight, highlighting the mill's architectural details and the surrounding greenery.

Keywords: Adobe Stock, Aerial view, Architecture, Blue sky, Chimney, Clay tiles, Elderly woman, England, Mill, National Trust, Pedestrians, People, River Itchen, Riverbank, Scenery, Sitting, Stone wall
</pre>

</details>

---

<a id="model-tencent-youtu-vl-4b-instruct"></a>

### tencent/Youtu-VL-4B-Instruct

<details>
<summary>Complete evidence: tencent/Youtu-VL-4B-Instruct</summary>

- *Execution:* crashed
- *Usability:* not_evaluated
- *Maintainer status:* actionable_failure
- *Observations:* none
- *Failure phase:* model_load
- *Error stage:* Lib Version
- *Error code:* MODEL_REPO_CODE_MODEL_LOAD_LIB_VERSION
- *Error type:* ValueError
- *Error package:* model-repo-code
- *Error message:* Model loading failed: cannot import name
  'DefaultFastImageProcessorKwargs' from
  'transformers.image_processing_utils_fast' (unknown location)
- *Root exception type:* ImportError
- *Root exception module:* builtins
- *Root exception message:* cannot import name
  'DefaultFastImageProcessorKwargs' from
  'transformers.image_processing_utils_fast' (unknown location)
- *Arch supported by installed mlx-vlm:* yes (model_type youtu_vl)
- *Model load time:* 1.06s
- *Generation time:* -
- *Total time:* 1.06s
- *Input validation time:* 6.6e-05
- *Prompt preparation time:* -
- *First-token latency:* -
- *Cleanup time:* 0.146
- *Prompt tokens:* -
- *Generation tokens:* -
- *Total tokens:* -
- *Prompt throughput (raw):* -
- *Generation throughput (raw):* -
- *Peak memory:* -
- *Active memory:* -
- *Cache memory:* -
- *Model-load active memory:* -
- *Post-cleanup active memory:* 0.0134
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* exception
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* not captured
- *Image placeholders:* not captured
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* not captured
- *Tokenizer:* not captured
- *Requested model revision:* not requested
- *Resolved model revision:* 8d30a0e49662a1d628a472b12df264dbcd768753
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--tencent--Youtu-VL-4B-Instruct/snapshots/8d30a0e49662a1d628a472b12df264dbcd768753
- *Generation settings:* not captured
- *EOS token:* not captured

#### Complete traceback

```python
Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13601, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12726, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 817, in _typed_mlx_vlm_load
    loaded: tuple[nn.Module, ProcessorMixin] = _mlx_vlm_load(
                                               ~~~~~~~~~~~~~^
        path_or_hf_repo=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1298, in load
    processor = load_processor(model_path, True, eos_token_ids=eos_token_id, **kwargs)
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1443, in load_processor
    processor = AutoProcessor.from_pretrained(model_path, **kwargs)
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/base.py", line 652, in _patched_auto_processor_from_pretrained
    return previous_from_pretrained.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/base.py", line 652, in _patched_auto_processor_from_pretrained
    return previous_from_pretrained.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/base.py", line 652, in _patched_auto_processor_from_pretrained
    return previous_from_pretrained.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  [Previous line repeated 10 more times]
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/molmo2/processing.py", line 768, in _patched_auto_processor_from_pretrained_molmo2
    return _original_auto_processor_from_pretrained_molmo2.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/base.py", line 652, in _patched_auto_processor_from_pretrained
    return previous_from_pretrained.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/base.py", line 652, in _patched_auto_processor_from_pretrained
    return previous_from_pretrained.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/base.py", line 652, in _patched_auto_processor_from_pretrained
    return previous_from_pretrained.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  [Previous line repeated 11 more times]
  File "~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages/transformers/models/auto/processing_auto.py", line 326, in from_pretrained
    return processor_class.from_pretrained(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages/transformers/processing_utils.py", line 1735, in from_pretrained
    args = cls._get_arguments_from_pretrained(pretrained_model_name_or_path, processor_dict, **kwargs)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages/transformers/processing_utils.py", line 1875, in _get_arguments_from_pretrained
    sub_processor = auto_processor_class.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, **kwargs
    )
  File "~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages/transformers/models/auto/image_processing_auto.py", line 672, in from_pretrained
    image_processor_class = get_class_from_dynamic_module(class_ref, pretrained_model_name_or_path, **kwargs)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages/transformers/dynamic_module_utils.py", line 623, in get_class_from_dynamic_module
    return get_class_in_module(class_name, final_module, force_reload=force_download)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages/transformers/dynamic_module_utils.py", line 309, in get_class_in_module
    module_spec.loader.exec_module(module)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^
  File "<frozen importlib._bootstrap_external>", line 759, in exec_module
  File "<frozen importlib._bootstrap>", line 491, in _call_with_frames_removed
  File "~/.cache/huggingface/modules/transformers_modules/_8d30a0e49662a1d628a472b12df264dbcd768753/71940f1d5c3bbe2f/image_processing_siglip2_fast.py", line 7, in <module>
    from transformers.image_processing_utils_fast import (
    ...<3 lines>...
    )
ImportError: cannot import name 'DefaultFastImageProcessorKwargs' from 'transformers.image_processing_utils_fast' (unknown location)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14614, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13616, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: cannot import name 'DefaultFastImageProcessorKwargs' from 'transformers.image_processing_utils_fast' (unknown location)

```

#### Captured upstream output

```text
=== STDERR ===
Downloading bytes:           |  0.00B
Reconstructing (incomplete total...): |          |  0.00B /  0.00B
Fetching 19 files:   0%|          | 0/19 [00:00<?, ?it/s]
Fetching 19 files: 100%|##########| 19/19 [00:00<00:00, 3199.06it/s]
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
[15:03:13] DEBUG    HF Cache Info for tencent/Youtu-VL-4B-Instruct: size=10229.6 MB, files=27
```

</details>

---
