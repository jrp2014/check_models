# Model Output Gallery

Generated on: 2026-08-10 20:22:08 BST

Complete per-model evidence artifact with image metadata, the source prompt, a
facts-only chooser, and full generated or crash output for every attempted
model.

## Reference Image

![Reference image](assets/source-image.jpg)

## Current-run Chooser

Current-run usability and captured resource facts only. Total time is end-to-end; throughput covers generation only and requires at least 16 generated tokens. Prefill/first is first-token latency when captured; Prompt tok is the full rendered prompt including image tokens, which drives prefill cost. For cross-attention architectures the token count reflects the tokenised text burden only, not total vision prefill compute.

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                                   | Usability             | Total s | Gen TPS    | Prefill/first s | Peak GB | Prompt tok | Gen tok | Observations                                                                                                             |
|-------------------------------------------------------------------------------------------------------------------------|-----------------------|---------|------------|-----------------|---------|------------|---------|--------------------------------------------------------------------------------------------------------------------------|
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `usable`              | 9.40s   | 30.3 tok/s | 3.26            | 23      | 2,426      | 104     | none                                                                                                                     |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `usable`              | 6.50s   | 31.4 tok/s | 1.01            | 17      | 2,147      | 101     | none                                                                                                                     |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `usable`              | 7.08s   | 64.1 tok/s | 2.11            | 13      | 2,959      | 198     | none                                                                                                                     |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `usable`              | 7.24s   | 60.9 tok/s | 2.19            | 13      | 2,959      | 188     | none                                                                                                                     |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `usable`              | 3.47s   | 178 tok/s  | 1.05            | 7.8     | 2,958      | 161     | none                                                                                                                     |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `usable`              | 90.43s  | 55.9 tok/s | 78.95           | 74      | 16,596     | 102     | none                                                                                                                     |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `usable`              | 3.93s   | 54.8 tok/s | 0.31            | 9.3     | 1,164      | 119     | none                                                                                                                     |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `usable`              | 61.26s  | 106 tok/s  | 56.75           | 24      | 16,596     | 102     | none                                                                                                                     |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `usable`              | 65.26s  | 92.0 tok/s | 61.99           | 10.0    | 16,596     | 88      | none                                                                                                                     |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `usable`              | 89.30s  | 17.0 tok/s | 79.20           | 35      | 16,596     | 103     | none                                                                                                                     |
| [`mlx-community/Step-3.7-Flash-oQ2e`](#model-mlx-community-step-37-flash-oq2e)                                          | `usable`              | 25.36s  | 44.8 tok/s | 16.30           | 70      | 3,524      | 111     | none                                                                                                                     |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `usable`              | 7.96s   | 31.0 tok/s | 1.04            | 18      | 618        | 126     | none                                                                                                                     |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `usable`              | 4.56s   | 110 tok/s  | 0.43            | 16      | 623        | 101     | none                                                                                                                     |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `usable`              | 8.69s   | 23.4 tok/s | 1.53            | 20      | 623        | 85      | none                                                                                                                     |
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `usable_with_caveats` | 1.42s   | 510 tok/s  | 0.04            | 1.2     | 425        | 119     | title/keyword constraints failed                                                                                         |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `usable_with_caveats` | 16.57s  | 94.2 tok/s | 13.76           | 8.4     | 16,581     | 123     | title/keyword constraints failed                                                                                         |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `usable_with_caveats` | 9.16s   | 80.7 tok/s | 6.26            | 8.4     | 6,483      | 85      | title/keyword constraints failed                                                                                         |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `usable_with_caveats` | 21.91s  | 41.8 tok/s | 12.94           | 78      | 6,483      | 100     | control tokens visible                                                                                                   |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `usable_with_caveats` | 8.24s   | 32.1 tok/s | 1.11            | 18      | 2,644      | 141     | role tokens visible; title/keyword constraints failed                                                                    |
| [`mlx-community/Molmo-7B-D-0924-8bit`](#model-mlx-community-molmo-7b-d-0924-8bit)                                       | `usable_with_caveats` | 5.73s   | 49.7 tok/s | 0.71            | 11      | 1,547      | 135     | title/keyword constraints failed                                                                                         |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                             | `usable_with_caveats` | 80.82s  | 219 tok/s  | 78.62           | 5.1     | 16,592     | 141     | title/keyword constraints failed                                                                                         |
| [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16)                             | `usable_with_caveats` | 22.13s  | 75.1 tok/s | 18.87           | 8.4     | 16,581     | 123     | title/keyword constraints failed                                                                                         |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `usable_with_caveats` | 2.65s   | 129 tok/s  | 0.34            | 5.5     | 462        | 112     | title/keyword constraints failed                                                                                         |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `usable_with_caveats` | 6.35s   | 44.0 tok/s | 0.36            | 29      | 619        | 84      | control tokens visible; title/keyword constraints failed                                                                 |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `usable_with_caveats` | 5.89s   | 53.6 tok/s | 0.35            | 28      | 619        | 81      | control tokens visible                                                                                                   |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `usable_with_caveats` | 7.04s   | 37.7 tok/s | 1.98            | 16      | 3,149      | 106     | title/keyword constraints failed                                                                                         |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `unusable`            | 16.00s  | 42.5 tok/s | 2.06            | 15      | 3,240      | 500     | missing required fields; echoes instructions; extra text before Title; cut off at token limit                            |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `unusable`            | 16.21s  | 59.5 tok/s | 1.88            | 60      | 1,665      | 500     | missing required fields; cut off at token limit; incomplete thinking block                                               |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `unusable`            | 2.16s   | 351 tok/s  | 0.10            | 2.1     | 368        | 131     | missing required fields; extra text before Title                                                                         |
| [`mlx-community/GLM-4.1V-9B-Thinking-8bit`](#model-mlx-community-glm-41v-9b-thinking-8bit)                              | `unusable`            | 18.30s  | 47.4 tok/s | 5.59            | 13      | 6,479      | 500     | extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed             |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `unusable`            | 129.36s | 4.04 tok/s | 1.66            | 40      | 1,355      | 500     | missing required fields; echoes instructions; extra text before Title; cut off at token limit; incomplete thinking block |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `unusable`            | 4.17s   | 166 tok/s  | 0.09            | 4.1     | 615        | 500     | repeated text; cut off at token limit; title/keyword constraints failed                                                  |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)            | `unusable`            | 32.51s  | 17.7 tok/s | 2.16            | 15      | 333        | 500     | repeated text; cut off at token limit; title/keyword constraints failed                                                  |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `unusable`            | 2.01s   | 256 tok/s  | 0.26            | 3.8     | 965        | 40      | missing required fields; extra text before Title                                                                         |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                           | `unusable`            | 28.10s  | 5.98 tok/s | 2.55            | 24      | 3,160      | 134     | missing required fields                                                                                                  |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `unusable`            | 26.29s  | 81.1 tok/s | 18.42           | 8.4     | 16,583     | 500     | missing required fields; cut off at token limit; incomplete thinking block                                               |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `unusable`            | 32.02s  | 51.8 tok/s | 20.34           | 13      | 16,592     | 500     | repeated text; cut off at token limit; title/keyword constraints failed                                                  |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `unusable`            | 16.26s  | 45.8 tok/s | 0.40            | 17      | 617        | 500     | missing required fields                                                                                                  |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                            | `unusable`            | 13.61s  | 55.6 tok/s | 2.84            | 9.7     | 2,552      | 500     | repeated text; missing required fields; extra text before Title; cut off at token limit                                  |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                            | `unusable`            | 1.81s   | 224 tok/s  | 0.09            | 2.2     | 364        | 77      | missing required fields                                                                                                  |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                             | `unusable`            | 15.78s  | 40.9 tok/s | 1.16            | 4.4     | 4,444      | 500     | repeated text; missing required fields; echoes instructions; cut off at token limit                                      |
| [`mlx-community/Inkling-Small-mlx-4bit`](#model-mlx-community-inkling-small-mlx-4bit)                                   | `not_evaluated`       | 0.20s   | -          | -               | -       | -          | -       | none                                                                                                                     |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Resource Highlights

Fastest clean completion: `mlx-community/Ministral-3-3B-Instruct-2512-4bit` at 178 tok/s

Average clean-completion throughput: 64.2 tok/s (indicative only: tokenizers and architectures differ across models)

Lowest peak memory among clean completions: `mlx-community/Ministral-3-3B-Instruct-2512-4bit` at 7.8 GB

## Avoid for This Run

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                        | Usability       | Observations                                                                                                             |
|--------------------------------------------------------------------------------------------------------------|-----------------|--------------------------------------------------------------------------------------------------------------------------|
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)       | `unusable`      | missing required fields; echoes instructions; extra text before Title; cut off at token limit                            |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16) | `unusable`      | missing required fields; cut off at token limit; incomplete thinking block                                               |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                   | `unusable`      | missing required fields; extra text before Title                                                                         |
| [`mlx-community/GLM-4.1V-9B-Thinking-8bit`](#model-mlx-community-glm-41v-9b-thinking-8bit)                   | `unusable`      | extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed             |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)        | `unusable`      | missing required fields; echoes instructions; extra text before Title; cut off at token limit; incomplete thinking block |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                | `unusable`      | repeated text; cut off at token limit; title/keyword constraints failed                                                  |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit) | `unusable`      | repeated text; cut off at token limit; title/keyword constraints failed                                                  |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                 | `unusable`      | missing required fields; extra text before Title                                                                         |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                | `unusable`      | missing required fields                                                                                                  |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                  | `unusable`      | missing required fields; cut off at token limit; incomplete thinking block                                               |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                | `unusable`      | repeated text; cut off at token limit; title/keyword constraints failed                                                  |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                            | `unusable`      | missing required fields                                                                                                  |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                 | `unusable`      | repeated text; missing required fields; extra text before Title; cut off at token limit                                  |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                 | `unusable`      | missing required fields                                                                                                  |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                  | `unusable`      | repeated text; missing required fields; echoes instructions; cut off at token limit                                      |
| [`mlx-community/Inkling-Small-mlx-4bit`](#model-mlx-community-inkling-small-mlx-4bit)                        | `not_evaluated` | none                                                                                                                     |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Output at a Glance

The first 280 characters of each model's actual output (or failure evidence for crashes), in chooser order. Complete exact evidence for every model follows below.

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                                   | Usability             | Output preview                                                                                                                                                                                                                                                                                                        |
|-------------------------------------------------------------------------------------------------------------------------|-----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `usable`              | Title:<br>Arundel Cathedral and Residential Street<br><br>Description:<br>A view of Arundel Cathedral of Our Lady &amp; St. Philip Howard in Arundel, England, with a residential street in the foreground featuring a red car parked in a garage and a cottage with a stone wall and garden.<br><br>Keywor...        |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `usable`              | Title: Arundel Cathedral and Neighboring Cottage<br><br>Description: The image captures a serene view of Arundel Cathedral with a red car parked in the foreground, set against a clear blue sky on August 6, 2026.<br><br>Keywords: Arundel, Arundel Cathedral of Our Lady &amp; St. Philip Howard,...               |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `usable`              | Title:<br>**Arundel Cathedral &amp; Quaint Cottage, Sunset Light**<br><br>Description:<br>A photograph taken on **6 August 2026** near **Arundel, West Sussex (50.806659°N, 0.551382°W)**, captures the **Arundel Cathedral of Our Lady &amp; St. Philip Howard** in golden evening light, juxtaposed wit...          |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `usable`              | **Title:**<br>Arundel Cathedral &amp; Cottage – Sunset View, UK<br><br>**Description:**<br>A photograph captured on **August 6, 2026, at 18:26 UTC+1, near 50.806659°N, 0.551382°W** showcases **Arundel Cathedral of Our Lady &amp; St. Philip Howard** towering over a quaint cottage with a red car pa...          |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `usable`              | Title:<br>Arundel’s Gothic Masterpiece with Quaint Residences<br><br>Description:<br>A bright, clear evening in August 2026 captures the grandeur of Arundel Cathedral of Our Lady &amp; St. Philip Howard against a vivid blue sky, framed by traditional English cottages with stone walls and red...               |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `usable`              | Title: Red car parked outside cottage with cathedral spire<br>Description: A red Toyota Yaris is parked in a driveway in front of a cream-coloured cottage with a flint wall, set against the backdrop of Arundel Cathedral's Gothic spire under a clear blue sky.<br>Keywords: Arundel, Ca...                        |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `usable`              | Title: Arundel Cathedral View from the Street<br><br>Description: Arundel Cathedral of Our Lady &amp; St. Philip Howard stands majestically over the quaint English town, captured from a street perspective on a clear day. The cathedral's French-Gothic architecture is highlighted by the b...                    |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `usable`              | Title: Red Car Parked Before Arundel Cathedral<br><br>Description: A red hatchback is parked in a driveway beneath a tiled roof, framed by a stone wall and brick buildings, with the Gothic spires of Arundel Cathedral rising against a clear blue sky.<br><br>Keywords: Arundel Cathedral, Got...                  |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `usable`              | Title: Arundel Cathedral and Residential Street Scene<br><br>Description: A view of Arundel Cathedral of Our Lady &amp; St. Philip Howard, with a red car parked in a residential street below, under a clear blue sky.<br><br>Keywords: Arundel Cathedral, Residential Street, Red Car, Blue Sky, Go...              |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `usable`              | Title:<br>Arundel Cathedral and Residential Street Scene<br><br>Description:<br>A view of the Gothic Arundel Cathedral rising behind a white cottage with a tiled roof and a parked red car. The scene is captured on a clear, sunny day with bright blue skies and distinct shadows.<br><br>Keywords:...             |
| [`mlx-community/Step-3.7-Flash-oQ2e`](#model-mlx-community-step-37-flash-oq2e)                                          | `usable`              | Title:<br>Arundel Cathedral and Residential Street View<br><br>Description:<br>A view of Arundel Cathedral of Our Lady and St Philip Howard rising above a residential street in Arundel, England, featuring a red car parked in a driveway, a brick cottage, and a clear blue sky on a bright da...                  |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `usable`              | Title: Arundel Cathedral &amp; Neighbourhood, West Sussex, August 2026<br>Description: Captured on 6th August 2026, Arundel Cathedral of Our Lady &amp; St Philip Howard dominates the skyline above residential buildings and a stone wall. A red car is parked on the street in the foreground...                   |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `usable`              | Title:<br>Arundel Cathedral overlooking residential houses in England<br><br>Description:<br>A view of the French-Gothic Arundel Cathedral rising above traditional brick and stone houses under a clear blue sky. A red car is parked in a driveway next to a white garage door in the foregroun...                  |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `usable`              | Title: Arundel Cathedral of Our Lady and St Philip Howard<br>Description: A view of the French-Gothic Arundel Cathedral towering over residential cottages and a red car parked in a driveway under a clear blue sky.<br>Keywords: Arundel, Arundel Cathedral, England, West Sussex, cathed...                        |
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `usable_with_caveats` | Title:<br>Arundel Cathedral of Our Lady &amp; St. Philip Howard, Arundel, England, UK, GBR, Europe<br><br>Description:<br>Arundel Cathedral of Our Lady &amp; St. Philip Howard, Arundel, England, UK, GBR, Europe, viewed from a street corner with a red car parked in front.<br><br>Keywords:<br>Arundel Cathed... |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `usable_with_caveats` | Title: Arundel Cathedral View<br><br>Description: A red car is parked in a narrow street in Arundel, England, with the Arundel Cathedral of Our Lady &amp; St. Philip Howard visible in the background under a clear blue sky. The scene features traditional stone buildings, a white cottage,...                    |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `usable_with_caveats` | Title: Arundel Cathedral View<br>Description: The image shows a view of Arundel Cathedral of Our Lady &amp; St. Philip Howard, with a red car parked near a white garage door, set against a clear blue sky.<br>Keywords: Arundel Cathedral, red car, white garage, blue sky, England, UK, Goth...                    |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `usable_with_caveats` | <\|begin_of_box\|>Title: Arundel Cathedral and Red Car in English Village<br>Description: A red car is parked in a cobblestone driveway beside a white cottage, with Arundel Cathedral’s Gothic spire and rose window visible under a clear blue sky in Arundel, England.<br>Keywords: Arunde...                      |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `usable_with_caveats` | Title: Arundel Cathedral of Our Lady &amp; St. Philip Howard, Arundel, England, UK, GBR, Europe.<br>Description: The Arundel Cathedral of Our Lady and St. Philip Howard is a French-Gothic style cathedral located in Arundel, England, UK. The cathedral is surrounded by a neighborhood o...                       |
| [`mlx-community/Molmo-7B-D-0924-8bit`](#model-mlx-community-molmo-7b-d-0924-8bit)                                       | `usable_with_caveats` | Title: Arundel Cathedral of Our Lady &amp; St. Philip Howard, Arundel, England, UK, GBR, Europe<br><br>Description: A red car is parked in front of a white cottage with a stone wall, set against the backdrop of the impressive Arundel Cathedral under a clear blue sky. The scene captures...                     |
| [`mlx-community/Qwen2-VL-2B-Instruct-4bit`](#model-mlx-community-qwen2-vl-2b-instruct-4bit)                             | `usable_with_caveats` | Title:<br>Arundel Cathedral of Our Lady &amp; St. Philip Howard, Arundel, England, UK, GBR, Europe<br><br>Description:<br>Arundel, UK - October 31, 2021: View of Arundel Cathedral of Our Lady and St. Philip Howard, with a red car parked in front of a house, under a clear blue sky.<br><br>Keywords:...         |
| [`mlx-community/Qwen3-VL-2B-Instruct-bf16`](#model-mlx-community-qwen3-vl-2b-instruct-bf16)                             | `usable_with_caveats` | Title: Arundel Cathedral View<br><br>Description: A red car is parked in a narrow street in Arundel, England, with the Arundel Cathedral of Our Lady &amp; St. Philip Howard visible in the background under a clear blue sky. The scene features traditional stone buildings, a white cottage,...                    |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `usable_with_caveats` | Title: Arundel Cathedral of Our Lady &amp; St. Philip Howard<br>Description: Arundel, UK - October 31, 2021: View of Arundel Cathedral of Our Lady and St Philip Howard<br>Keywords: Adobe Stock, Any Vision, Arundel, Arundel Cathedral of Our Lady &amp; St. Philip Howard, Blue sky, Bush, Car,...                 |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `usable_with_caveats` | <\|channel>thought<br><channel\|>Title: Arundel Cathedral over residential houses in England<br>Description: A view of the French-Gothic Arundel Cathedral of Our Lady &amp; St. Philip Howard rising above local stone cottages and a red car under a clear sky.<br>Keywords: Arundel, Cathedral, E...               |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `usable_with_caveats` | <\|channel>thought<br><channel\|>Title: Arundel Cathedral of Our Lady &amp; St. Philip Howard<br>Description: A view of the French-Gothic Arundel Cathedral rising over neighborhood stone walls and a red parked car under a clear blue sky.<br>Keywords: Arundel, Cathedral, Church, England, Goth...               |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `usable_with_caveats` | Title:<br>Arundel Cathedral under Clear Blue Sky<br><br>Description:<br>Arundel Cathedral of Our Lady and St. Philip Howard stands prominently against a clear blue sky, with a red car parked nearby in a quaint neighborhood.<br><br>Keywords:<br>Arundel Cathedral, Our Lady &amp; St. Philip Howard, Blue...      |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `unusable`            | Here are my reasoning steps:<br>We need to produce British-English catalogue metadata from the image and supplied context. The context includes capture date/time and GPS. The descriptive hints are hints, not authoritative. We need to treat capture date/time and GPS as authoritati...                           |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `unusable`            | Alright, let's get to work. I've got the image and the hints, and I need to craft some metadata. First, the title. The hints are a bit of a mess, but the image is clear. It's a view of Arundel Cathedral, a Gothic church, with a red car and a stone wall. I'll go with "Arundel C...                              |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `unusable`            | A concrete 5-10-word title: Arundel Cathedral of Our Lady &amp; St. Philip Howard, Arundel, England, UK, GBR, Europe<br>Description: A photograph of the Arundel Cathedral of Our Lady and St. Philip Howard in Arundel, UK, taken on 2021-08-06 at 18:26:35 UTC+01:00, with a blue sky and...                        |
| [`mlx-community/GLM-4.1V-9B-Thinking-8bit`](#model-mlx-community-glm-41v-9b-thinking-8bit)                              | `unusable`            | &lt;think&gt;Got it, let's tackle this. First, the title needs to be 5-10 words. The main subject is Arundel Cathedral with surrounding buildings. So maybe "Arundel Cathedral and Neighborhood View".<br><br>Then the description: need to combine context (date/time, location) with visible el...                  |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `unusable`            | ◁think▷Okay, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. The user wants a title, a descriptive paragraph, and 10-18 keywords.<br><br>Starting with the title: The main subject is Arundel Cathedral, so th...                        |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `unusable`            | Title: Arundel Cathedral of Our Lady &amp; St. Philip Howard, Arundel, England, UK, GBR, Europe<br><br>Description: A stunning view of Arundel Cathedral of Our Lady and St Philip Howard, showcasing its French-Gothic architecture against a clear blue sky. The cathedral's intricate spires...                    |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)            | `unusable`            | Title: Arundel Cathedral of Our Lady and St. Philip Howard in the Afternoon Sun<br><br>Description: The Arundel Cathedral of Our Lady and St. Philip Howard is captured in the late afternoon sun, showcasing its intricate architecture and surrounding neighborhood. The image features a...                        |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `unusable`            | Stunning cathedral view in quaint British neighborhood.<br><br>Keywords: Arundel, cathedral, church, England, Europe, blue sky, red car, cottage, flower, brick, roof, neighborhood                                                                                                                                   |
| [`mlx-community/MolmoPoint-8B-fp16`](#model-mlx-community-molmopoint-8b-fp16)                                           | `unusable`            | Arundel Cathedral: Gothic Beauty Amidst English Suburb<br><br>Captured on August 6, 2026, this image showcases Arundel Cathedral of Our Lady &amp; St. Philip Howard in Arundel, England. The grand Gothic structure stands prominently against a clear blue sky, surrounded by a mix of reside...                    |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `unusable`            | Got it, let's tackle this step by step. First, the title. The context says it's Arundel Cathedral of Our Lady &amp; St. Philip Howard, so the title should be concise. The hint says "Arundel Cathedral of Our Lady &amp; St. Philip Howard, Arundel, England, UK, GBR, Europe" but we need a...                      |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `unusable`            | Title: Arundel Cathedral and Cottage at Dusk<br><br>Description: A serene view of Arundel Cathedral of Our Lady &amp; St. Philip Howard, framed by a quaint cottage and vibrant red car, under a clear blue sky at dusk.<br><br>Keywords: Arundel, Arundel Cathedral, England, UK, Europe, Gothic arc...              |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `unusable`            | <start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><star...                              |
| [`mlx-community/llava-v1.6-mistral-7b-8bit`](#model-mlx-community-llava-v16-mistral-7b-8bit)                            | `unusable`            | Arundel Cathedral of Our Lady &amp; St. Philip Howard, Arundel, UK - October 31, 2021: View of Arundel Cathedral of Our Lady and St Philip Howard<br><br>Title:<br>Description:<br>Keywords: [/] Arundel Cathedral of Our Lady &amp; St. Philip Howard, Arundel, UK - October 31, 2021: View of Arundel C...          |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                            | `unusable`            | Title: Arundel Cathedral of Our Lady &amp; St. Philip Howard, Arundel, UK - October 31, 2021: View of Arundel Cathedral of Our Lady and St. Philip Howard<br>Description: A 10-minute description of the cathedral, including its location, the date and time of the view, and the surroundi...                       |
| [`mlx-community/paligemma2-3b-pt-896-4bit`](#model-mlx-community-paligemma2-3b-pt-896-4bit)                             | `unusable`            | - Title hint:<br>- Description hint:<br>- 10-18 unique, comma-separated keywords covering relevant context and visible details.<br>- 10-18 unique, comma-separated keywords covering relevant context and visible details.<br>- 10-18 unique, comma-separated keywords covering relevant context...                   |
| [`mlx-community/Inkling-Small-mlx-4bit`](#model-mlx-community-inkling-small-mlx-4bit)                                   | `not_evaluated`       | Model loading failed: Received 362 parameters not in model:<br>audio_tower.encoder.biases,<br>audio_tower.encoder.scales,<br>language_model.model.layers.10.mlp.experts.down_proj.biases,<br>language_model.model.layers.10.mlp.experts.down_proj.scales,<br>language_model.model.layers.10.mlp.expe...               |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Run Stamps

- `mlx-vlm`: `0.6.11`
- `mlx`: `0.32.1.dev20260810+e78d894c8`
- `mlx-lm`: `0.31.3`
- `transformers`: `5.15.0`
- `tokenizers`: `0.22.2`
- `huggingface-hub`: `1.27.0`
- *Python Version:* 3.13.13
- *OS:* Darwin 25.6.0
- *macOS Version:* 26.6.1
- *GPU/Chip:* Apple M5 Max
- *MLX Device:* Apple M5 Max
- *GPU Architecture:* applegpu_g17s
- *RAM:* 128.0 GB
- *Recommended Working Set:* 108 GB
- *Fused Attention:* Available

## Image Metadata

- *Title:* , Arundel Cathedral of Our Lady & St. Philip Howard, Arundel,
  England, UK, GBR, Europe
- *Description:* Arundel, UK - October 31, 2021: View of Arundel Cathedral of
  Our Lady and St Philip Howard
- *Keywords:* Adobe Stock, Any Vision, Arundel, Arundel Cathedral of Our Lady
  & St. Philip Howard, Blue sky, Bush, Car, Cathedral, Church, Cottage,
  England, Europe, Flower, French-Gothic, Neighborhood, Objects, Parking, Red
  Car, Roof, Sky, Stone, Stone wall, Town, Tree, UK, United Kingdom, Wall,
  West Sussex, architecture, brick, british, building, charming, christianity,
  cityscape, day, destination, dormer window, driveway, english, european,
  flint wall, gothic, heritage, historic, home, house, landmark, old, outdoor,
  outdoors, picturesque, plant, quaint, religion, residential, scenic, spire,
  street, summer, sunny, tourism, traditional, travel, urban, village, wheelie
  bin, window
- *Date:* 2026-08-06 18:26:35 UTC+01:00
- *Time:* 18:26:35
- *GPS:* 50.806659°N, 0.551382°W

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
> &#45; Capture date/time: 2026-08-06 18:26:35 UTC+01:00
> &#45; GPS: 50.806659°N, 0.551382°W
>
> &#8203;Descriptive hints:
> &#45; Title hint: Arundel Cathedral of Our Lady &amp; St. Philip Howard, Arundel,
> England, UK, GBR, Europe
> &#45; Description hint: Arundel, UK - October 31, 2021: View of Arundel
> Cathedral of Our Lady and St Philip Howard
> &#45; Keyword hints: Adobe Stock, Any Vision, Arundel, Arundel Cathedral of Our
> Lady &amp; St. Philip Howard, Blue sky, Bush, Car, Cathedral, Church, Cottage,
> England, Europe, Flower, French-Gothic, Neighborhood, Objects, Parking, Red
> Car, Roof, Sky
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

<a id="model-mlx-community-devstral-small-2-24b-instruct-2512-5bit"></a>

### mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

<details>
<summary>Complete evidence: mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type mistral3)
- *Model load time:* 2.14s
- *Generation time:* 7.25s
- *Total time:* 9.40s
- *Input validation time:* 5.28e-05
- *Prompt preparation time:* 0.00715
- *First-token latency:* 3.26
- *Cleanup time:* 0.094
- *Prompt tokens:* 2,426
- *Generation tokens:* 104
- *Total tokens:* 2,530
- *Prompt throughput (raw):* 744 tok/s
- *Generation throughput (raw):* 30.3 tok/s
- *Peak memory:* 23
- *Active memory:* 18
- *Cache memory:* 0.47
- *Model-load active memory:* 18
- *Post-cleanup active memory:* 0.000968
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1352
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.mistral3.processing_mistral3.Mistral3Processor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 0a970d20ad7da57b0e2accc35c5b8628f5d02063
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Devstral-Small-2-24B-Instruct-2512-5bit/snapshots/0a970d20ad7da57b0e2accc35c5b8628f5d02063
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Title:
Arundel Cathedral and Residential Street

Description:
A view of Arundel Cathedral of Our Lady &amp; St. Philip Howard in Arundel, England, with a residential street in the foreground featuring a red car parked in a garage and a cottage with a stone wall and garden.

Keywords:
Arundel Cathedral, Residential Street, Red Car, Cottage, Stone Wall, Garden, Church, Gothic Architecture, Blue Sky, Parking, England, UK, Europe, Historic Building, Urban Scene
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title:
Arundel Cathedral and Residential Street

Description:
A view of Arundel Cathedral of Our Lady & St. Philip Howard in Arundel, England, with a residential street in the foreground featuring a red car parked in a garage and a cottage with a stone wall and garden.

Keywords:
Arundel Cathedral, Residential Street, Red Car, Cottage, Stone Wall, Garden, Church, Gothic Architecture, Blue Sky, Parking, England, UK, Europe, Historic Building, Urban Scene
```

</details>

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
- *Generation time:* 4.87s
- *Total time:* 6.50s
- *Input validation time:* 4.56e-05
- *Prompt preparation time:* 0.00186
- *First-token latency:* 1.01
- *Cleanup time:* 0.109
- *Prompt tokens:* 2,147
- *Generation tokens:* 101
- *Total tokens:* 2,248
- *Prompt throughput (raw):* 2,136 tok/s
- *Generation throughput (raw):* 31.4 tok/s
- *Peak memory:* 17
- *Active memory:* 16
- *Cache memory:* 0.17
- *Model-load active memory:* 15.9
- *Post-cleanup active memory:* 0.00302
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1357
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.internvl_chat.processor.InternVLChatProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* e0df3dd79263467173214b67ef6d6a0cc5a475fd
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--InternVL3-8B-bf16/snapshots/e0df3dd79263467173214b67ef6d6a0cc5a475fd
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Arundel Cathedral and Neighboring Cottage

Description: The image captures a serene view of Arundel Cathedral with a red car parked in the foreground, set against a clear blue sky on August 6, 2026.

Keywords: Arundel, Arundel Cathedral of Our Lady &amp; St. Philip Howard, Blue sky, Car, Cathedral, Cottage, England, Europe, Flower, French-Gothic, Neighborhood, Parking, Red Car, Roof, Sky
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title: Arundel Cathedral and Neighboring Cottage

Description: The image captures a serene view of Arundel Cathedral with a red car parked in the foreground, set against a clear blue sky on August 6, 2026.

Keywords: Arundel, Arundel Cathedral of Our Lady & St. Philip Howard, Blue sky, Car, Cathedral, Cottage, England, Europe, Flower, French-Gothic, Neighborhood, Parking, Red Car, Roof, Sky
```

</details>

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
- *Model load time:* 1.33s
- *Generation time:* 5.74s
- *Total time:* 7.08s
- *Input validation time:* 4.74e-05
- *Prompt preparation time:* 0.00668
- *First-token latency:* 2.11
- *Cleanup time:* 0.0898
- *Prompt tokens:* 2,959
- *Generation tokens:* 198
- *Total tokens:* 3,157
- *Prompt throughput (raw):* 1,404 tok/s
- *Generation throughput (raw):* 64.1 tok/s
- *Peak memory:* 13
- *Active memory:* 8.0
- *Cache memory:* 0.60
- *Model-load active memory:* 8.01
- *Post-cleanup active memory:* 0.00511
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 3790
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.mistral3.processing_mistral3.Mistral3Processor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 7c992876448fc5426841a85c6eb951b056fe38d2
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Ministral-3-14B-Instruct-2512-mxfp4/snapshots/7c992876448fc5426841a85c6eb951b056fe38d2
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Title:
**Arundel Cathedral &amp; Quaint Cottage, Sunset Light**

Description:
A photograph taken on **6 August 2026** near **Arundel, West Sussex (50.806659°N, 0.551382°W)**, captures the **Arundel Cathedral of Our Lady &amp; St. Philip Howard** in golden evening light, juxtaposed with a traditional stone cottage and a red car parked in its garage. The scene features clear blue skies, lush greenery, and a blend of Gothic architectural details with rustic residential charm.

Keywords:
Arundel Cathedral, Our Lady &amp; St. Philip Howard, Gothic architecture, stone cottage, red car, garage entrance, West Sussex, England, UK, European heritage, sunset lighting, clear blue sky, garden plants, traditional British village, historical church, French-Gothic influence, residential neighborhood, 2026-08-06
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title:
**Arundel Cathedral & Quaint Cottage, Sunset Light**

Description:
A photograph taken on **6 August 2026** near **Arundel, West Sussex (50.806659°N, 0.551382°W)**, captures the **Arundel Cathedral of Our Lady & St. Philip Howard** in golden evening light, juxtaposed with a traditional stone cottage and a red car parked in its garage. The scene features clear blue skies, lush greenery, and a blend of Gothic architectural details with rustic residential charm.

Keywords:
Arundel Cathedral, Our Lady & St. Philip Howard, Gothic architecture, stone cottage, red car, garage entrance, West Sussex, England, UK, European heritage, sunset lighting, clear blue sky, garden plants, traditional British village, historical church, French-Gothic influence, residential neighborhood, 2026-08-06
```

</details>

</details>

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-nvfp4"></a>

### mlx-community/Ministral-3-14B-Instruct-2512-nvfp4

<details>
<summary>Complete evidence: mlx-community/Ministral-3-14B-Instruct-2512-nvfp4</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type mistral3)
- *Model load time:* 1.42s
- *Generation time:* 5.81s
- *Total time:* 7.24s
- *Input validation time:* 4.99e-05
- *Prompt preparation time:* 0.000243
- *First-token latency:* 2.19
- *Cleanup time:* 0.0942
- *Prompt tokens:* 2,959
- *Generation tokens:* 188
- *Total tokens:* 3,147
- *Prompt throughput (raw):* 1,353 tok/s
- *Generation throughput (raw):* 60.9 tok/s
- *Peak memory:* 13
- *Active memory:* 8.4
- *Cache memory:* 0.60
- *Model-load active memory:* 8.43
- *Post-cleanup active memory:* 0.00538
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 3790
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.mistral3.processing_mistral3.Mistral3Processor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 28777b889d841a86369c736175cb77258c8134b2
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Ministral-3-14B-Instruct-2512-nvfp4/snapshots/28777b889d841a86369c736175cb77258c8134b2
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
**Title:**
Arundel Cathedral &amp; Cottage – Sunset View, UK

**Description:**
A photograph captured on **August 6, 2026, at 18:26 UTC+1, near 50.806659°N, 0.551382°W** showcases **Arundel Cathedral of Our Lady &amp; St. Philip Howard** towering over a quaint cottage with a red car parked outside. The scene features **French-Gothic architecture, clear blue skies, and well-maintained gardens** against a serene evening backdrop.

**Keywords:**
Arundel, Arundel Cathedral, Cathedral, Cottage, England, UK, Europe, French-Gothic, Sunset, Red Car, Blue Sky, Church, Neighborhood, Parking, Roof, Gardens, Historical Architecture, 2026-08-06
</pre>

<details>
<summary>Exact raw output</summary>

```text
**Title:**
Arundel Cathedral & Cottage – Sunset View, UK

**Description:**
A photograph captured on **August 6, 2026, at 18:26 UTC+1, near 50.806659°N, 0.551382°W** showcases **Arundel Cathedral of Our Lady & St. Philip Howard** towering over a quaint cottage with a red car parked outside. The scene features **French-Gothic architecture, clear blue skies, and well-maintained gardens** against a serene evening backdrop.

**Keywords:**
Arundel, Arundel Cathedral, Cathedral, Cottage, England, UK, Europe, French-Gothic, Sunset, Red Car, Blue Sky, Church, Neighborhood, Parking, Roof, Gardens, Historical Architecture, 2026-08-06
```

</details>

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
- *Model load time:* 0.96s
- *Generation time:* 2.50s
- *Total time:* 3.47s
- *Input validation time:* 5.49e-05
- *Prompt preparation time:* 0.00681
- *First-token latency:* 1.05
- *Cleanup time:* 0.0904
- *Prompt tokens:* 2,958
- *Generation tokens:* 161
- *Total tokens:* 3,119
- *Prompt throughput (raw):* 2,807 tok/s
- *Generation throughput (raw):* 178 tok/s
- *Peak memory:* 7.8
- *Active memory:* 2.8
- *Cache memory:* 0.41
- *Model-load active memory:* 2.75
- *Post-cleanup active memory:* 0.00564
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 3789
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.mistral3.processing_mistral3.Mistral3Processor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* a962dcb09eee4169c890e544c9eb938f1113fdee
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Ministral-3-3B-Instruct-2512-4bit/snapshots/a962dcb09eee4169c890e544c9eb938f1113fdee
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Title:
Arundel’s Gothic Masterpiece with Quaint Residences

Description:
A bright, clear evening in August 2026 captures the grandeur of Arundel Cathedral of Our Lady &amp; St. Philip Howard against a vivid blue sky, framed by traditional English cottages with stone walls and red parked cars. The cathedral’s intricate Gothic architecture, including pointed spires and a prominent rose window, dominates the skyline, while lush greenery and blooming flowers add charm to the peaceful neighborhood.

Keywords:
Arundel Cathedral, Gothic architecture, stone cottages, red car, blue sky, August 2026, English countryside, rose window, traditional English buildings, urban park setting, historic stone walls, floral decorations, cathedral spires, vintage neighborhood.
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title:
Arundel’s Gothic Masterpiece with Quaint Residences

Description:
A bright, clear evening in August 2026 captures the grandeur of Arundel Cathedral of Our Lady & St. Philip Howard against a vivid blue sky, framed by traditional English cottages with stone walls and red parked cars. The cathedral’s intricate Gothic architecture, including pointed spires and a prominent rose window, dominates the skyline, while lush greenery and blooming flowers add charm to the peaceful neighborhood.

Keywords:
Arundel Cathedral, Gothic architecture, stone cottages, red car, blue sky, August 2026, English countryside, rose window, traditional English buildings, urban park setting, historic stone walls, floral decorations, cathedral spires, vintage neighborhood.
```

</details>

</details>

---

<a id="model-mlx-community-ornith-10-35b-bf16"></a>

### mlx-community/Ornith-1.0-35B-bf16

<details>
<summary>Complete evidence: mlx-community/Ornith-1.0-35B-bf16</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5_moe)
- *Model load time:* 8.12s
- *Generation time:* 82.29s
- *Total time:* 90.43s
- *Input validation time:* 6.25e-05
- *Prompt preparation time:* 0.0141
- *First-token latency:* 79
- *Cleanup time:* 0.27
- *Prompt tokens:* 16,596
- *Generation tokens:* 102
- *Total tokens:* 16,698
- *Prompt throughput (raw):* 210 tok/s
- *Generation throughput (raw):* 55.9 tok/s
- *Peak memory:* 74
- *Active memory:* 70
- *Cache memory:* 0.88
- *Model-load active memory:* 70.2
- *Post-cleanup active memory:* 0.00708
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1443
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 9ef631ad2d0c4c26783d4f94d0a0de9516e41a4b
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Ornith-1.0-35B-bf16/snapshots/9ef631ad2d0c4c26783d4f94d0a0de9516e41a4b
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Red car parked outside cottage with cathedral spire
Description: A red Toyota Yaris is parked in a driveway in front of a cream-coloured cottage with a flint wall, set against the backdrop of Arundel Cathedral's Gothic spire under a clear blue sky.
Keywords: Arundel, Cathedral, England, UK, Red Car, Cottage, Roof, Spire, Blue Sky, Flints, Garden, Flower, Architecture, Church, Neighborhood, Parking, Toyota, Gothic
</pre>

</details>

---

<a id="model-mlx-community-phi-35-vision-instruct-bf16"></a>

### mlx-community/Phi-3.5-vision-instruct-bf16

<details>
<summary>Complete evidence: mlx-community/Phi-3.5-vision-instruct-bf16</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type phi3_v)
- *Model load time:* 0.91s
- *Generation time:* 3.02s
- *Total time:* 3.93s
- *Input validation time:* 4.59e-05
- *Prompt preparation time:* 0.000976
- *First-token latency:* 0.308
- *Cleanup time:* 0.0993
- *Prompt tokens:* 1,164
- *Generation tokens:* 119
- *Total tokens:* 1,283
- *Prompt throughput (raw):* 3,775 tok/s
- *Generation throughput (raw):* 54.8 tok/s
- *Peak memory:* 9.3
- *Active memory:* 8.3
- *Cache memory:* 0.62
- *Model-load active memory:* 8.3
- *Post-cleanup active memory:* 0.00714
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1373
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.phi3_v.processing_phi3_v.Phi3VProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* d8da684308c275a86659e2b36a9189b2f4aec8ea
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Phi-3.5-vision-instruct-bf16/snapshots/d8da684308c275a86659e2b36a9189b2f4aec8ea
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|endoftext\|>

*Readable output:*

<pre class="model-output-readable">
Title: Arundel Cathedral View from the Street

Description: Arundel Cathedral of Our Lady &amp; St. Philip Howard stands majestically over the quaint English town, captured from a street perspective on a clear day. The cathedral's French-Gothic architecture is highlighted by the blue sky and the surrounding greenery.

Keywords: Arundel Cathedral, Our Lady &amp; St. Philip Howard, Arundel, England, French-Gothic, blue sky, greenery, street view, quaint town, architecture
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title: Arundel Cathedral View from the Street

Description: Arundel Cathedral of Our Lady & St. Philip Howard stands majestically over the quaint English town, captured from a street perspective on a clear day. The cathedral's French-Gothic architecture is highlighted by the blue sky and the surrounding greenery.

Keywords: Arundel Cathedral, Our Lady & St. Philip Howard, Arundel, England, French-Gothic, blue sky, greenery, street view, quaint town, architecture
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
- *Model load time:* 2.60s
- *Generation time:* 58.63s
- *Total time:* 61.26s
- *Input validation time:* 5.7e-05
- *Prompt preparation time:* 0.0152
- *First-token latency:* 56.7
- *Cleanup time:* 0.107
- *Prompt tokens:* 16,596
- *Generation tokens:* 102
- *Total tokens:* 16,698
- *Prompt throughput (raw):* 292 tok/s
- *Generation throughput (raw):* 106 tok/s
- *Peak memory:* 24
- *Active memory:* 21
- *Cache memory:* 0.89
- *Model-load active memory:* 20.4
- *Post-cleanup active memory:* 0.00859
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1443
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 1e20fd8d42056f870933bf98ca6211024744f7ec
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Red Car Parked Before Arundel Cathedral

Description: A red hatchback is parked in a driveway beneath a tiled roof, framed by a stone wall and brick buildings, with the Gothic spires of Arundel Cathedral rising against a clear blue sky.

Keywords: Arundel Cathedral, Gothic architecture, red car, stone wall, brick building, blue sky, roof, driveway, spire, England, UK, neighborhood, flowers, windows, church, sky, architecture, car
</pre>

</details>

---

<a id="model-mlx-community-qwen35-9b-mlx-4bit"></a>

### mlx-community/Qwen3.5-9B-MLX-4bit

<details>
<summary>Complete evidence: mlx-community/Qwen3.5-9B-MLX-4bit</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5)
- *Model load time:* 1.38s
- *Generation time:* 63.87s
- *Total time:* 65.26s
- *Input validation time:* 5.78e-05
- *Prompt preparation time:* 0.000329
- *First-token latency:* 62
- *Cleanup time:* 0.0953
- *Prompt tokens:* 16,596
- *Generation tokens:* 88
- *Total tokens:* 16,684
- *Prompt throughput (raw):* 268 tok/s
- *Generation throughput (raw):* 92.0 tok/s
- *Peak memory:* 10.0
- *Active memory:* 6.6
- *Cache memory:* 1.2
- *Model-load active memory:* 5.96
- *Post-cleanup active memory:* 0.00911
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1443
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 938d8919941c6e7efd3c7150eff7fe9d12afa631
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-MLX-4bit/snapshots/938d8919941c6e7efd3c7150eff7fe9d12afa631
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Arundel Cathedral and Residential Street Scene

Description: A view of Arundel Cathedral of Our Lady &amp; St. Philip Howard, with a red car parked in a residential street below, under a clear blue sky.

Keywords: Arundel Cathedral, Residential Street, Red Car, Blue Sky, Gothic Architecture, Stone Walls, Flower Boxes, Cobblestone Path, White House, Cathedral Spire, England, Europe
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title: Arundel Cathedral and Residential Street Scene

Description: A view of Arundel Cathedral of Our Lady & St. Philip Howard, with a red car parked in a residential street below, under a clear blue sky.

Keywords: Arundel Cathedral, Residential Street, Red Car, Blue Sky, Gothic Architecture, Stone Walls, Flower Boxes, Cobblestone Path, White House, Cathedral Spire, England, Europe
```

</details>

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
- *Model load time:* 3.08s
- *Generation time:* 86.19s
- *Total time:* 89.30s
- *Input validation time:* 5.09e-05
- *Prompt preparation time:* 0.0148
- *First-token latency:* 79.2
- *Cleanup time:* 0.161
- *Prompt tokens:* 16,596
- *Generation tokens:* 103
- *Total tokens:* 16,699
- *Prompt throughput (raw):* 210 tok/s
- *Generation throughput (raw):* 17.0 tok/s
- *Peak memory:* 35
- *Active memory:* 29
- *Cache memory:* 1.9
- *Model-load active memory:* 28.7
- *Post-cleanup active memory:* 0.00962
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1443
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 5db9fd9c38ce083048ee12217eceed7482ca7ad4
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-27B-mxfp8/snapshots/5db9fd9c38ce083048ee12217eceed7482ca7ad4
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
Arundel Cathedral and Residential Street Scene

Description:
A view of the Gothic Arundel Cathedral rising behind a white cottage with a tiled roof and a parked red car. The scene is captured on a clear, sunny day with bright blue skies and distinct shadows.

Keywords:
Arundel, Cathedral, Church, Gothic, England, UK, Red Car, Cottage, Residential, Street, Blue Sky, Architecture, Roof, Garden, Stone Wall, Daylight, Landscape, Europe
</pre>

</details>

---

<a id="model-mlx-community-step-37-flash-oq2e"></a>

### mlx-community/Step-3.7-Flash-oQ2e

<details>
<summary>Complete evidence: mlx-community/Step-3.7-Flash-oQ2e</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type step3p7)
- *Model load time:* 5.72s
- *Generation time:* 19.63s
- *Total time:* 25.36s
- *Input validation time:* 4.82e-05
- *Prompt preparation time:* 0.014
- *First-token latency:* 16.3
- *Cleanup time:* 0.205
- *Prompt tokens:* 3,524
- *Generation tokens:* 111
- *Total tokens:* 3,635
- *Prompt throughput (raw):* 216 tok/s
- *Generation throughput (raw):* 44.8 tok/s
- *Peak memory:* 70
- *Active memory:* 63
- *Cache memory:* 0.41
- *Model-load active memory:* 63
- *Post-cleanup active memory:* 0.01
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1430
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.step3p7.processing_step3p7.Step3VLProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 3dacb46f724ac89725bcd922fb779c7ed1499fe7
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Step-3.7-Flash-oQ2e/snapshots/3dacb46f724ac89725bcd922fb779c7ed1499fe7
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
Arundel Cathedral and Residential Street View

Description:
A view of Arundel Cathedral of Our Lady and St Philip Howard rising above a residential street in Arundel, England, featuring a red car parked in a driveway, a brick cottage, and a clear blue sky on a bright day.

Keywords:
Arundel, Arundel Cathedral, Cathedral, Church, Gothic architecture, Red car, Parking, Residential street, Brick building, Blue sky, England, UK, Europe, Roof, Window, Flower, Bush, Stone wall
</pre>

</details>

---

<a id="model-mlx-community-gemma-3-27b-it-qat-4bit"></a>

### mlx-community/gemma-3-27b-it-qat-4bit

<details>
<summary>Complete evidence: mlx-community/gemma-3-27b-it-qat-4bit</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type gemma3)
- *Model load time:* 2.30s
- *Generation time:* 5.65s
- *Total time:* 7.96s
- *Input validation time:* 5.03e-05
- *Prompt preparation time:* 0.00263
- *First-token latency:* 1.04
- *Cleanup time:* 0.151
- *Prompt tokens:* 618
- *Generation tokens:* 126
- *Total tokens:* 744
- *Prompt throughput (raw):* 596 tok/s
- *Generation throughput (raw):* 31.0 tok/s
- *Peak memory:* 18
- *Active memory:* 16
- *Cache memory:* 0.46
- *Model-load active memory:* 16.1
- *Post-cleanup active memory:* 0.0108
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1407
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.gemma3.processing_gemma3.Gemma3Processor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* fc4e000f32af1b7b6779294e490a7d2a80bac611
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--gemma-3-27b-it-qat-4bit/snapshots/fc4e000f32af1b7b6779294e490a7d2a80bac611
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
Title: Arundel Cathedral &amp; Neighbourhood, West Sussex, August 2026
Description: Captured on 6th August 2026, Arundel Cathedral of Our Lady &amp; St Philip Howard dominates the skyline above residential buildings and a stone wall. A red car is parked on the street in the foreground under a clear blue sky.
Keywords: Arundel, Arundel Cathedral of Our Lady &amp; St Philip Howard, Church, Cathedral, West Sussex, England, UK, Europe, Architecture, French-Gothic, Roof, Stone wall, Red car, Residential buildings, Blue sky, Street, Parking, Cobblestone street
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title: Arundel Cathedral & Neighbourhood, West Sussex, August 2026
Description: Captured on 6th August 2026, Arundel Cathedral of Our Lady & St Philip Howard dominates the skyline above residential buildings and a stone wall. A red car is parked on the street in the foreground under a clear blue sky.
Keywords: Arundel, Arundel Cathedral of Our Lady & St Philip Howard, Church, Cathedral, West Sussex, England, UK, Europe, Architecture, French-Gothic, Roof, Stone wall, Red car, Residential buildings, Blue sky, Street, Parking, Cobblestone street
```

</details>

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
- *Model load time:* 2.56s
- *Generation time:* 1.98s
- *Total time:* 4.56s
- *Input validation time:* 3.86e-05
- *Prompt preparation time:* 0.0226
- *First-token latency:* 0.428
- *Cleanup time:* 0.14
- *Prompt tokens:* 623
- *Generation tokens:* 101
- *Total tokens:* 724
- *Prompt throughput (raw):* 1,456 tok/s
- *Generation throughput (raw):* 110 tok/s
- *Peak memory:* 16
- *Active memory:* 15
- *Cache memory:* 0.21
- *Model-load active memory:* 15.4
- *Post-cleanup active memory:* 0.0119
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1406
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.gemma4.processing_gemma4.Gemma4Processor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 0d77464eeb233a2da68ebf9d7dc4edaac7db956d
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--gemma-4-26b-a4b-it-4bit/snapshots/0d77464eeb233a2da68ebf9d7dc4edaac7db956d
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
Title:
Arundel Cathedral overlooking residential houses in England

Description:
A view of the French-Gothic Arundel Cathedral rising above traditional brick and stone houses under a clear blue sky. A red car is parked in a driveway next to a white garage door in the foreground.

Keywords:
Arundel, Cathedral, England, Europe, architecture, blue sky, red car, house, roof, church, Gothic, residential, street, building, stone wall, brick house, landscape
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
- *Model load time:* 2.85s
- *Generation time:* 5.82s
- *Total time:* 8.69s
- *Input validation time:* 3.48e-05
- *Prompt preparation time:* 0.000294
- *First-token latency:* 1.53
- *Cleanup time:* 0.203
- *Prompt tokens:* 623
- *Generation tokens:* 85
- *Total tokens:* 708
- *Prompt throughput (raw):* 408 tok/s
- *Generation throughput (raw):* 23.4 tok/s
- *Peak memory:* 20
- *Active memory:* 18
- *Cache memory:* 0.80
- *Model-load active memory:* 18.4
- *Post-cleanup active memory:* 0.0124
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1406
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.gemma4.processing_gemma4.Gemma4Processor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 696d436c404745a59f30e4939a658162b0a9e57f
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--gemma-4-31b-it-4bit/snapshots/696d436c404745a59f30e4939a658162b0a9e57f
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
Title: Arundel Cathedral of Our Lady and St Philip Howard
Description: A view of the French-Gothic Arundel Cathedral towering over residential cottages and a red car parked in a driveway under a clear blue sky.
Keywords: Arundel, Arundel Cathedral, England, West Sussex, cathedral, church, French-Gothic, architecture, cottage, red car, residential, blue sky, Europe, UK, neighborhood, stone wall
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
- *Observations:* Title has 13 words (requested 5-10); Duplicate keywords:
  arundel, england, europe
- *Arch supported by installed mlx-vlm:* yes (model_type lfm2-vl via lfm2_vl)
- *Model load time:* 0.41s
- *Generation time:* 0.67s
- *Total time:* 1.42s
- *Input validation time:* 0.333
- *Prompt preparation time:* 0.00635
- *First-token latency:* 0.0363
- *Cleanup time:* 0.0591
- *Prompt tokens:* 425
- *Generation tokens:* 119
- *Total tokens:* 544
- *Prompt throughput (raw):* 11,699 tok/s
- *Generation throughput (raw):* 510 tok/s
- *Peak memory:* 1.2
- *Active memory:* 0.90
- *Cache memory:* 0.02
- *Model-load active memory:* 0.897
- *Post-cleanup active memory:* 0.000132
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1403
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* transformers.models.lfm2_vl.processing_lfm2_vl.Lfm2VlProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 6c33f49ebc0b50b75385f49ad3beddcb720d0c75
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--LiquidAI--LFM2.5-VL-450M-MLX-bf16/snapshots/6c33f49ebc0b50b75385f49ad3beddcb720d0c75
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
Arundel Cathedral of Our Lady &amp; St. Philip Howard, Arundel, England, UK, GBR, Europe

Description:
Arundel Cathedral of Our Lady &amp; St. Philip Howard, Arundel, England, UK, GBR, Europe, viewed from a street corner with a red car parked in front.

Keywords:
Arundel Cathedral, Arundel, England, GBR, Europe, red car, street corner, Arundel, UK, Arundel, England, Europe, cathedral, church, street, parking, sky
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title:
Arundel Cathedral of Our Lady & St. Philip Howard, Arundel, England, UK, GBR, Europe

Description:
Arundel Cathedral of Our Lady & St. Philip Howard, Arundel, England, UK, GBR, Europe, viewed from a street corner with a red car parked in front.

Keywords:
Arundel Cathedral, Arundel, England, GBR, Europe, red car, street corner, Arundel, UK, Arundel, England, Europe, cathedral, church, street, parking, sky
```

</details>

</details>

---

<a id="model-qwen-qwen3-vl-2b-instruct"></a>

### Qwen/Qwen3-VL-2B-Instruct

<details>
<summary>Complete evidence: Qwen/Qwen3-VL-2B-Instruct</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Title has 3 words (requested 5-10)
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Model load time:* 0.65s
- *Generation time:* 15.90s
- *Total time:* 16.57s
- *Input validation time:* 4.55e-05
- *Prompt preparation time:* 0.00915
- *First-token latency:* 13.8
- *Cleanup time:* 0.0835
- *Prompt tokens:* 16,581
- *Generation tokens:* 123
- *Total tokens:* 16,704
- *Prompt throughput (raw):* 1,205 tok/s
- *Generation throughput (raw):* 94.2 tok/s
- *Peak memory:* 8.4
- *Active memory:* 4.3
- *Cache memory:* 2.6
- *Model-load active memory:* 4.26
- *Post-cleanup active memory:* 0.000443
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1424
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 89644892e4d85e24eaac8bacfd4f463576704203
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Arundel Cathedral View

Description: A red car is parked in a narrow street in Arundel, England, with the Arundel Cathedral of Our Lady &amp; St. Philip Howard visible in the background under a clear blue sky. The scene features traditional stone buildings, a white cottage, and lush greenery, with the cathedral's Gothic architecture and spire standing prominently.

Keywords: Arundel, Cathedral, Church, England, Europe, Blue sky, Car, Red Car, Roof, Sky, Neighborhood, Objects, Parking, Cottage, Flower, French-Gothic, Bush, Street
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title: Arundel Cathedral View

Description: A red car is parked in a narrow street in Arundel, England, with the Arundel Cathedral of Our Lady & St. Philip Howard visible in the background under a clear blue sky. The scene features traditional stone buildings, a white cottage, and lush greenery, with the cathedral's Gothic architecture and spire standing prominently.

Keywords: Arundel, Cathedral, Church, England, Europe, Blue sky, Car, Red Car, Roof, Sky, Neighborhood, Objects, Parking, Cottage, Flower, French-Gothic, Bush, Street
```

</details>

</details>

---

<a id="model-mlx-community-glm-46v-flash-mxfp4"></a>

### mlx-community/GLM-4.6V-Flash-mxfp4

<details>
<summary>Complete evidence: mlx-community/GLM-4.6V-Flash-mxfp4</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Title has 3 words (requested 5-10)
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v)
- *Model load time:* 1.34s
- *Generation time:* 7.81s
- *Total time:* 9.16s
- *Input validation time:* 4.77e-05
- *Prompt preparation time:* 0.00684
- *First-token latency:* 6.26
- *Cleanup time:* 0.0774
- *Prompt tokens:* 6,483
- *Generation tokens:* 85
- *Total tokens:* 6,568
- *Prompt throughput (raw):* 1,035 tok/s
- *Generation throughput (raw):* 80.7 tok/s
- *Peak memory:* 8.4
- *Active memory:* 6.8
- *Cache memory:* 0.46
- *Model-load active memory:* 6.78
- *Post-cleanup active memory:* 0.00213
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1433
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.glm4v.processing.Glm46VProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 773591fa7388b5f0db2f5ec11ed9dc3a23779f1b
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--GLM-4.6V-Flash-mxfp4/snapshots/773591fa7388b5f0db2f5ec11ed9dc3a23779f1b
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|endoftext\|>

*Readable output:*

<pre class="model-output-readable">
Title: Arundel Cathedral View
Description: The image shows a view of Arundel Cathedral of Our Lady &amp; St. Philip Howard, with a red car parked near a white garage door, set against a clear blue sky.
Keywords: Arundel Cathedral, red car, white garage, blue sky, England, UK, Gothic architecture, neighborhood street, brick wall, flower pots, cobblestone path.
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title: Arundel Cathedral View
Description: The image shows a view of Arundel Cathedral of Our Lady & St. Philip Howard, with a red car parked near a white garage door, set against a clear blue sky.
Keywords: Arundel Cathedral, red car, white garage, blue sky, England, UK, Gothic architecture, neighborhood street, brick wall, flower pots, cobblestone path.
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
- *Model load time:* 6.05s
- *Generation time:* 15.84s
- *Total time:* 21.91s
- *Input validation time:* 4.87e-05
- *Prompt preparation time:* 0.00673
- *First-token latency:* 12.9
- *Cleanup time:* 0.205
- *Prompt tokens:* 6,483
- *Generation tokens:* 100
- *Total tokens:* 6,583
- *Prompt throughput (raw):* 501 tok/s
- *Generation throughput (raw):* 41.8 tok/s
- *Peak memory:* 78
- *Active memory:* 62
- *Cache memory:* 1.4
- *Model-load active memory:* 61.9
- *Post-cleanup active memory:* 0.00244
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1433
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.glm4v_moe.processing.Glm46VMoEProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 2da6855d4e28a0e61c84543262074bc17ac27d6e
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--GLM-4.6V-nvfp4/snapshots/2da6855d4e28a0e61c84543262074bc17ac27d6e
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|endoftext\|>

*Readable output:*

<pre class="model-output-readable">
&lt;|begin_of_box|&gt;Title: Arundel Cathedral and Red Car in English Village
Description: A red car is parked in a cobblestone driveway beside a white cottage, with Arundel Cathedral’s Gothic spire and rose window visible under a clear blue sky in Arundel, England.
Keywords: Arundel Cathedral, Arundel, England, UK, Cathedral, Church, Red Car, Cottage, Blue Sky, Flower, Roof, Neighborhood, Gothic, Parking, Bush, Stone Wall&lt;|end_of_box|&gt;
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|begin_of_box|>Title: Arundel Cathedral and Red Car in English Village
Description: A red car is parked in a cobblestone driveway beside a white cottage, with Arundel Cathedral’s Gothic spire and rose window visible under a clear blue sky in Arundel, England.
Keywords: Arundel Cathedral, Arundel, England, UK, Cathedral, Church, Red Car, Cottage, Blue Sky, Flower, Roof, Neighborhood, Gothic, Parking, Bush, Stone Wall<|end_of_box|>
```

</details>

</details>

---

<a id="model-mlx-community-idefics3-8b-llama3-bf16"></a>

### mlx-community/Idefics3-8B-Llama3-bf16

<details>
<summary>Complete evidence: mlx-community/Idefics3-8B-Llama3-bf16</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Conversation-role control tokens remain visible; Title has
  13 words (requested 5-10)
- *Arch supported by installed mlx-vlm:* yes (model_type idefics3)
- *Model load time:* 1.99s
- *Generation time:* 6.24s
- *Total time:* 8.24s
- *Input validation time:* 6.84e-05
- *Prompt preparation time:* 0.00125
- *First-token latency:* 1.11
- *Cleanup time:* 0.118
- *Prompt tokens:* 2,644
- *Generation tokens:* 141
- *Total tokens:* 2,785
- *Prompt throughput (raw):* 2,372 tok/s
- *Generation throughput (raw):* 32.1 tok/s
- *Peak memory:* 18
- *Active memory:* 17
- *Cache memory:* 0.43
- *Model-load active memory:* 16.9
- *Post-cleanup active memory:* 0.0027
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1389
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.idefics3.processing_idefics3.Idefics3Processor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 8c2a30c48864f3251701b7bde40f601d25535098
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Idefics3-8B-Llama3-bf16/snapshots/8c2a30c48864f3251701b7bde40f601d25535098
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|end_of_text\|>

*Readable output:*

<pre class="model-output-readable">
 Title: Arundel Cathedral of Our Lady &amp; St. Philip Howard, Arundel, England, UK, GBR, Europe.
Description: The Arundel Cathedral of Our Lady and St. Philip Howard is a French-Gothic style cathedral located in Arundel, England, UK. The cathedral is surrounded by a neighborhood of brick cottages and has a red car parked in front of it.
Keywords: Arundel, Arundel Cathedral of Our Lady &amp; St. Philip Howard, Blue sky, Bush, Car, Cathedral, Church, Cottage, England, Europe, Flower, French-Gothic, Neighborhood, Objects, Parking, Red Car, Roof, Sky.&lt;end_of_utterance&gt;
</pre>

<details>
<summary>Exact raw output</summary>

```text
 Title: Arundel Cathedral of Our Lady & St. Philip Howard, Arundel, England, UK, GBR, Europe.
Description: The Arundel Cathedral of Our Lady and St. Philip Howard is a French-Gothic style cathedral located in Arundel, England, UK. The cathedral is surrounded by a neighborhood of brick cottages and has a red car parked in front of it.
Keywords: Arundel, Arundel Cathedral of Our Lady & St. Philip Howard, Blue sky, Bush, Car, Cathedral, Church, Cottage, England, Europe, Flower, French-Gothic, Neighborhood, Objects, Parking, Red Car, Roof, Sky.<end_of_utterance>
```

</details>

</details>

---

<a id="model-mlx-community-molmo-7b-d-0924-8bit"></a>

### mlx-community/Molmo-7B-D-0924-8bit

<details>
<summary>Complete evidence: mlx-community/Molmo-7B-D-0924-8bit</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Title has 13 words (requested 5-10)
- *Arch supported by installed mlx-vlm:* yes (model_type molmo)
- *Model load time:* 1.26s
- *Generation time:* 4.46s
- *Total time:* 5.73s
- *Input validation time:* 5.13e-05
- *Prompt preparation time:* 0.00144
- *First-token latency:* 0.707
- *Cleanup time:* 0.1
- *Prompt tokens:* 1,547
- *Generation tokens:* 135
- *Total tokens:* 1,682
- *Prompt throughput (raw):* 2,187 tok/s
- *Generation throughput (raw):* 49.7 tok/s
- *Peak memory:* 11
- *Active memory:* 9.0
- *Cache memory:* 0.14
- *Model-load active memory:* 9.04
- *Post-cleanup active memory:* 0.00595
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1348
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.molmo.processing_molmo.MolmoProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 90a14ed7a230088904c7556fbe6d67b295c33f5f
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Molmo-7B-D-0924-8bit/snapshots/90a14ed7a230088904c7556fbe6d67b295c33f5f
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|endoftext\|>

*Readable output:*

<pre class="model-output-readable">
 Title: Arundel Cathedral of Our Lady &amp; St. Philip Howard, Arundel, England, UK, GBR, Europe

Description: A red car is parked in front of a white cottage with a stone wall, set against the backdrop of the impressive Arundel Cathedral under a clear blue sky. The scene captures a picturesque view of this historic English landmark and its surrounding neighborhood.

Keywords: Arundel, Arundel Cathedral of Our Lady &amp; St. Philip Howard, Blue sky, Bush, Car, Cathedral, Church, Cottage, England, Europe, Flower, French-Gothic, Neighborhood, Objects, Parking, Red Car, Roof, Sky
</pre>

<details>
<summary>Exact raw output</summary>

```text
 Title: Arundel Cathedral of Our Lady & St. Philip Howard, Arundel, England, UK, GBR, Europe

Description: A red car is parked in front of a white cottage with a stone wall, set against the backdrop of the impressive Arundel Cathedral under a clear blue sky. The scene captures a picturesque view of this historic English landmark and its surrounding neighborhood.

Keywords: Arundel, Arundel Cathedral of Our Lady & St. Philip Howard, Blue sky, Bush, Car, Cathedral, Church, Cottage, England, Europe, Flower, French-Gothic, Neighborhood, Objects, Parking, Red Car, Roof, Sky
```

</details>

</details>

---

<a id="model-mlx-community-qwen2-vl-2b-instruct-4bit"></a>

### mlx-community/Qwen2-VL-2B-Instruct-4bit

<details>
<summary>Complete evidence: mlx-community/Qwen2-VL-2B-Instruct-4bit</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Title has 13 words (requested 5-10); Keyword list has 20
  terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type qwen2_vl)
- *Model load time:* 0.57s
- *Generation time:* 80.23s
- *Total time:* 80.82s
- *Input validation time:* 4.15e-05
- *Prompt preparation time:* 0.00357
- *First-token latency:* 78.6
- *Cleanup time:* 0.0792
- *Prompt tokens:* 16,592
- *Generation tokens:* 141
- *Total tokens:* 16,733
- *Prompt throughput (raw):* 211 tok/s
- *Generation throughput (raw):* 219 tok/s
- *Peak memory:* 5.1
- *Active memory:* 1.3
- *Cache memory:* 0.86
- *Model-load active memory:* 1.26
- *Post-cleanup active memory:* 0.00746
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1482
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen2_vl.processing_qwen2_vl.Qwen2VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 01af461cdb9574acc09084a0ef94e216e142b085
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen2-VL-2B-Instruct-4bit/snapshots/01af461cdb9574acc09084a0ef94e216e142b085
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
Arundel Cathedral of Our Lady &amp; St. Philip Howard, Arundel, England, UK, GBR, Europe

Description:
Arundel, UK - October 31, 2021: View of Arundel Cathedral of Our Lady and St. Philip Howard, with a red car parked in front of a house, under a clear blue sky.

Keywords:
Adobe Stock, Any Vision, Arundel, Arundel Cathedral of Our Lady &amp; St. Philip Howard, Blue sky, Bush, Car, Cathedral, Church, Cottage, England, Europe, Flower, French-Gothic, Neighborhood, Objects, Parking, Red Car, Roof, Sky
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title:
Arundel Cathedral of Our Lady & St. Philip Howard, Arundel, England, UK, GBR, Europe

Description:
Arundel, UK - October 31, 2021: View of Arundel Cathedral of Our Lady and St. Philip Howard, with a red car parked in front of a house, under a clear blue sky.

Keywords:
Adobe Stock, Any Vision, Arundel, Arundel Cathedral of Our Lady & St. Philip Howard, Blue sky, Bush, Car, Cathedral, Church, Cottage, England, Europe, Flower, French-Gothic, Neighborhood, Objects, Parking, Red Car, Roof, Sky
```

</details>

</details>

---

<a id="model-mlx-community-qwen3-vl-2b-instruct-bf16"></a>

### mlx-community/Qwen3-VL-2B-Instruct-bf16

<details>
<summary>Complete evidence: mlx-community/Qwen3-VL-2B-Instruct-bf16</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Title has 3 words (requested 5-10)
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Model load time:* 0.75s
- *Generation time:* 21.38s
- *Total time:* 22.13s
- *Input validation time:* 4.08e-05
- *Prompt preparation time:* 0.000222
- *First-token latency:* 18.9
- *Cleanup time:* 0.103
- *Prompt tokens:* 16,581
- *Generation tokens:* 123
- *Total tokens:* 16,704
- *Prompt throughput (raw):* 879 tok/s
- *Generation throughput (raw):* 75.1 tok/s
- *Peak memory:* 8.4
- *Active memory:* 4.3
- *Cache memory:* 2.6
- *Model-load active memory:* 4.26
- *Post-cleanup active memory:* 0.00777
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1424
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* c8a67a84327484ba87f5ec4f8fb927cdafd791aa
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3-VL-2B-Instruct-bf16/snapshots/c8a67a84327484ba87f5ec4f8fb927cdafd791aa
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Arundel Cathedral View

Description: A red car is parked in a narrow street in Arundel, England, with the Arundel Cathedral of Our Lady &amp; St. Philip Howard visible in the background under a clear blue sky. The scene features traditional stone buildings, a white cottage, and lush greenery, with the cathedral's Gothic architecture and spire standing prominently.

Keywords: Arundel, Cathedral, Church, England, Europe, Blue sky, Car, Red Car, Roof, Sky, Neighborhood, Objects, Parking, Cottage, Flower, French-Gothic, Bush, Street
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title: Arundel Cathedral View

Description: A red car is parked in a narrow street in Arundel, England, with the Arundel Cathedral of Our Lady & St. Philip Howard visible in the background under a clear blue sky. The scene features traditional stone buildings, a white cottage, and lush greenery, with the cathedral's Gothic architecture and spire standing prominently.

Keywords: Arundel, Cathedral, Church, England, Europe, Blue sky, Car, Red Car, Roof, Sky, Neighborhood, Objects, Parking, Cottage, Flower, French-Gothic, Bush, Street
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
- *Observations:* Keyword list has 20 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type smolvlm)
- *Model load time:* 0.73s
- *Generation time:* 1.92s
- *Total time:* 2.65s
- *Input validation time:* 5.23e-05
- *Prompt preparation time:* 0.00203
- *First-token latency:* 0.34
- *Cleanup time:* 0.0774
- *Prompt tokens:* 462
- *Generation tokens:* 112
- *Total tokens:* 574
- *Prompt throughput (raw):* 1,359 tok/s
- *Generation throughput (raw):* 129 tok/s
- *Peak memory:* 5.5
- *Active memory:* 4.5
- *Cache memory:* 0.19
- *Model-load active memory:* 4.5
- *Post-cleanup active memory:* 0.00973
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1384
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.smolvlm.processing_smolvlm.SmolVLMProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 844516024a1c4400d34489b89ee067d794e432ed
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--SmolVLM2-2.2B-Instruct-mlx/snapshots/844516024a1c4400d34489b89ee067d794e432ed
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <end_of_utterance>

*Readable output:*

<pre class="model-output-readable">
 Title: Arundel Cathedral of Our Lady &amp; St. Philip Howard
Description: Arundel, UK - October 31, 2021: View of Arundel Cathedral of Our Lady and St Philip Howard
Keywords: Adobe Stock, Any Vision, Arundel, Arundel Cathedral of Our Lady &amp; St. Philip Howard, Blue sky, Bush, Car, Cathedral, Church, Cottage, England, Europe, Flower, French-Gothic, Neighborhood, Objects, Parking, Red Car, Roof, Sky
</pre>

<details>
<summary>Exact raw output</summary>

```text
 Title: Arundel Cathedral of Our Lady & St. Philip Howard
Description: Arundel, UK - October 31, 2021: View of Arundel Cathedral of Our Lady and St Philip Howard
Keywords: Adobe Stock, Any Vision, Arundel, Arundel Cathedral of Our Lady & St. Philip Howard, Blue sky, Bush, Car, Cathedral, Church, Cottage, England, Europe, Flower, French-Gothic, Neighborhood, Objects, Parking, Red Car, Roof, Sky
```

</details>

</details>

---

<a id="model-mlx-community-diffusiongemma-26b-a4b-it-8bit"></a>

### mlx-community/diffusiongemma-26B-A4B-it-8bit

<details>
<summary>Complete evidence: mlx-community/diffusiongemma-26B-A4B-it-8bit</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Unrecognised model control tokens remain visible; Duplicate
  keywords: historic, urban
- *Arch supported by installed mlx-vlm:* yes (model_type diffusion_gemma)
- *Model load time:* 3.33s
- *Generation time:* 2.99s
- *Total time:* 6.35s
- *Input validation time:* 8.02e-05
- *Prompt preparation time:* 0.0208
- *First-token latency:* 0.36
- *Cleanup time:* 0.141
- *Prompt tokens:* 619
- *Generation tokens:* 84
- *Total tokens:* 703
- *Prompt throughput (raw):* 1,720 tok/s
- *Generation throughput (raw):* 44.0 tok/s
- *Peak memory:* 29
- *Active memory:* 28
- *Cache memory:* 0.01
- *Model-load active memory:* 28
- *Post-cleanup active memory:* 0.0103
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1378
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.diffusion_gemma.processing_diffusion_gemma.DiffusionGemma4Processor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 7b95e3887078ba56283c24f2578d6e5a06b9d7e8
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--diffusiongemma-26B-A4B-it-8bit/snapshots/7b95e3887078ba56283c24f2578d6e5a06b9d7e8
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
&lt;|channel&gt;thought
&lt;channel|&gt;Title: Arundel Cathedral over residential houses in England
Description: A view of the French-Gothic Arundel Cathedral of Our Lady &amp; St. Philip Howard rising above local stone cottages and a red car under a clear sky.
Keywords: Arundel, Cathedral, England, Gothic, architecture, church, stone, house, red car, roof, sky, neighborhood, historic, historic, urban, urban
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|channel>thought
<channel|>Title: Arundel Cathedral over residential houses in England
Description: A view of the French-Gothic Arundel Cathedral of Our Lady & St. Philip Howard rising above local stone cottages and a red car under a clear sky.
Keywords: Arundel, Cathedral, England, Gothic, architecture, church, stone, house, red car, roof, sky, neighborhood, historic, historic, urban, urban
```

</details>

</details>

---

<a id="model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8"></a>

### mlx-community/diffusiongemma-26B-A4B-it-mxfp8

<details>
<summary>Complete evidence: mlx-community/diffusiongemma-26B-A4B-it-mxfp8</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Unrecognised model control tokens remain visible
- *Arch supported by installed mlx-vlm:* yes (model_type diffusion_gemma)
- *Model load time:* 3.28s
- *Generation time:* 2.60s
- *Total time:* 5.89s
- *Input validation time:* 3.76e-05
- *Prompt preparation time:* 0.000325
- *First-token latency:* 0.354
- *Cleanup time:* 0.145
- *Prompt tokens:* 619
- *Generation tokens:* 81
- *Total tokens:* 700
- *Prompt throughput (raw):* 1,747 tok/s
- *Generation throughput (raw):* 53.6 tok/s
- *Peak memory:* 28
- *Active memory:* 27
- *Cache memory:* 0.01
- *Model-load active memory:* 27.2
- *Post-cleanup active memory:* 0.0103
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1378
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.diffusion_gemma.processing_diffusion_gemma.DiffusionGemma4Processor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* ded389e478f86d498ad9e7f47666e83b166a28f1
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--diffusiongemma-26B-A4B-it-mxfp8/snapshots/ded389e478f86d498ad9e7f47666e83b166a28f1
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
&lt;|channel&gt;thought
&lt;channel|&gt;Title: Arundel Cathedral of Our Lady &amp; St. Philip Howard
Description: A view of the French-Gothic Arundel Cathedral rising over neighborhood stone walls and a red parked car under a clear blue sky.
Keywords: Arundel, Cathedral, Church, England, Gothic, Architecture, Stone, Cottage, Red Car, Roof, Sky, Neighborhood, Parking, Flower, Historic, Europe
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|channel>thought
<channel|>Title: Arundel Cathedral of Our Lady & St. Philip Howard
Description: A view of the French-Gothic Arundel Cathedral rising over neighborhood stone walls and a red parked car under a clear blue sky.
Keywords: Arundel, Cathedral, Church, England, Gothic, Architecture, Stone, Cottage, Red Car, Roof, Sky, Neighborhood, Parking, Flower, Historic, Europe
```

</details>

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
- *Model load time:* 1.71s
- *Generation time:* 5.32s
- *Total time:* 7.04s
- *Input validation time:* 5.87e-05
- *Prompt preparation time:* 0.00236
- *First-token latency:* 1.98
- *Cleanup time:* 0.13
- *Prompt tokens:* 3,149
- *Generation tokens:* 106
- *Total tokens:* 3,255
- *Prompt throughput (raw):* 1,592 tok/s
- *Generation throughput (raw):* 37.7 tok/s
- *Peak memory:* 16
- *Active memory:* 13
- *Cache memory:* 0.59
- *Model-load active memory:* 13.5
- *Post-cleanup active memory:* 0.0144
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1352
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.pixtral.processing_pixtral.PixtralProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 79e24b66302d4ac5e75491767f8914b2aa6ac1f8
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--pixtral-12b-8bit/snapshots/79e24b66302d4ac5e75491767f8914b2aa6ac1f8
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Title:
Arundel Cathedral under Clear Blue Sky

Description:
Arundel Cathedral of Our Lady and St. Philip Howard stands prominently against a clear blue sky, with a red car parked nearby in a quaint neighborhood.

Keywords:
Arundel Cathedral, Our Lady &amp; St. Philip Howard, Blue Sky, Red Car, Church, Cathedral, England, Europe, Neighborhood, Roof, Stone Building, Brick House, Garden, Flower, Sky, Gothic Architecture, Parking, Street, 2026
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title:
Arundel Cathedral under Clear Blue Sky

Description:
Arundel Cathedral of Our Lady and St. Philip Howard stands prominently against a clear blue sky, with a red car parked nearby in a quaint neighborhood.

Keywords:
Arundel Cathedral, Our Lady & St. Philip Howard, Blue Sky, Red Car, Church, Cathedral, England, Europe, Neighborhood, Roof, Stone Building, Brick House, Garden, Flower, Sky, Gothic Architecture, Parking, Street, 2026
```

</details>

</details>

---

<a id="model-mlx-community-apriel-15-15b-thinker-6bit-mlx"></a>

### mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX

<details>
<summary>Complete evidence: mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* none
- *Observations:* Missing or empty fields: Title, Keywords; Response repeats
  the task instructions instead of only returning the requested fields; Extra
  text appears before the Title field; Response appears cut off at the token
  limit
- *Arch supported by installed mlx-vlm:* yes (model_type pixtral)
- *Model load time:* 1.63s
- *Generation time:* 14.35s
- *Total time:* 16.00s
- *Input validation time:* 5.18e-05
- *Prompt preparation time:* 0.00753
- *First-token latency:* 2.06
- *Cleanup time:* 0.0863
- *Prompt tokens:* 3,240
- *Generation tokens:* 500
- *Total tokens:* 3,740
- *Prompt throughput (raw):* 1,570 tok/s
- *Generation throughput (raw):* 42.5 tok/s
- *Peak memory:* 15
- *Active memory:* 13
- *Cache memory:* 0.77
- *Model-load active memory:* 12.6
- *Post-cleanup active memory:* 0.000706
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1726
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.pixtral.processing_pixtral.PixtralProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 24cb8fef6052e8d6e0dd7d467cf2d3db2dec19b2
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Apriel-1.5-15b-Thinker-6bit-MLX/snapshots/24cb8fef6052e8d6e0dd7d467cf2d3db2dec19b2
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Here are my reasoning steps:
We need to produce British-English catalogue metadata from the image and supplied context. The context includes capture date/time and GPS. The descriptive hints are hints, not authoritative. We need to treat capture date/time and GPS as authoritative facts, but not claim they are visible. So we can include them in description if needed? The description should combine relevant context with main visible subject, setting, action, lighting, distinctive details. The title should be concrete 5-10 words. Keywords: 10-18 unique, comma-separated.

We need to ensure we use British English (spelling). The image shows a view of Arundel Cathedral (Our Lady and St Philip Howard) from a narrow street with a red car parked in a driveway, stone walls, brick houses, a garden with flowers, a blue sky, etc. The cathedral is Gothic, with spires, a tall central spire, and a large rose window. The lighting appears to be late afternoon/evening (sun low, long shadows). The capture date/time is 2026-08-06 18:26:35 UTC+01:00 (so 6:26 PM BST). The GPS is 50.806659°N, 0.551382°W (Arundel). So we can incorporate that.

We need to produce Title: 5-10 words. Something like "Arundel Cathedral from a quiet lane". That's 6 words. Or "Red car beside Arundel Cathedral". That's 5 words. Or "Arundel Cathedral under blue sky". That's 5 words. Title should be concrete, 5-10 words. Let's choose "Red car parked near Arundel Cathedral". That's 6 words? Let's count: Red(1) car(2) parked(3) near(4) Arundel(5) Cathedral(6). That's 6 words. Good.

Description: 1-2 sentences factual, combine relevant context with main visible subject, setting, action, lighting, distinctive details. Should mention capture date/time? Possibly mention "captured on 6 August 2026 at 18:26 UTC+01". But we need to combine context with visible subject. So something like: "A
</pre>

</details>

---

<a id="model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16"></a>

### mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

<details>
<summary>Complete evidence: mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Missing or empty fields: Title, Description, Keywords;
  Response appears cut off at the token limit; Internal reasoning block
  appears incomplete
- *Arch supported by installed mlx-vlm:* yes (model_type ernie4_5_moe_vl)
- *Model load time:* 5.24s
- *Generation time:* 10.96s
- *Total time:* 16.21s
- *Input validation time:* 4.57e-05
- *Prompt preparation time:* 0.00958
- *First-token latency:* 1.88
- *Cleanup time:* 0.189
- *Prompt tokens:* 1,665
- *Generation tokens:* 500
- *Total tokens:* 2,165
- *Prompt throughput (raw):* 885 tok/s
- *Generation throughput (raw):* 59.5 tok/s
- *Peak memory:* 60
- *Active memory:* 59
- *Cache memory:* 0.14
- *Model-load active memory:* 58.8
- *Post-cleanup active memory:* 0.0012
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1542
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl.Ernie4_5_VLProcessor
- *Tokenizer:* mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl.Ernie4_5_VLTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 32dae5c38006e20ac158bc94cd1d5967d19b2652
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--ERNIE-4.5-VL-28B-A3B-Thinking-bf16/snapshots/32dae5c38006e20ac158bc94cd1d5967d19b2652
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Alright, let's get to work. I've got the image and the hints, and I need to craft some metadata. First, the title. The hints are a bit of a mess, but the image is clear. It's a view of Arundel Cathedral, a Gothic church, with a red car and a stone wall. I'll go with "Arundel Cathedral and Red Car at Sunset." That's descriptive and accurate.

Now for the description. I need to weave in the context – the date, time, and location – with the visual details. The hints are a bit off, but the image is what matters. I'll start with the location and time, then describe the scene: the cathedral, the red car, the stone wall, the sunset. I'll make sure to mention the Gothic architecture and the red car. I'll also note the stone wall and the red car. The final description will be: "A panoramic view of Arundel Cathedral in England, captured at sunset on August 6, 2026. The Gothic church is silhouetted against a bright blue sky, with a red car parked near a stone wall and a stone archway in the foreground."

Finally, the keywords. I'll pull from the hints, but I'll prioritize what's actually in the image. I'll include the location, the cathedral, the car, the architecture, the sky, and the time of day. I'll also add some general terms like "architecture" and "road." I'll make sure to include the date and time. I'll also add "stone wall" and "red car" to the list. I'll make sure to include "sunset" and "blue sky". I'll also add "Gothic" and "England". I'll make sure to include "road" and "stone archway". I'll also add "sunset" and "blue sky". I'll also add "Gothic" and "England". I'll also add "road" and "stone archway".

Here's the final list: Arundel, Arundel Cathedral, England, UK, Gothic architecture, red car, stone wall, blue sky, sunset, panoramic view, stone
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
- *Observations:* Missing or empty fields: Title, Keywords; Extra text appears
  before the Title field
- *Arch supported by installed mlx-vlm:* yes (model_type llava_qwen2 via
  fastvlm)
- *Model load time:* 0.60s
- *Generation time:* 1.55s
- *Total time:* 2.16s
- *Input validation time:* 5.77e-05
- *Prompt preparation time:* 0.00157
- *First-token latency:* 0.0964
- *Cleanup time:* 0.0721
- *Prompt tokens:* 368
- *Generation tokens:* 131
- *Total tokens:* 499
- *Prompt throughput (raw):* 3,819 tok/s
- *Generation throughput (raw):* 351 tok/s
- *Peak memory:* 2.1
- *Active memory:* 1.2
- *Cache memory:* 0.03
- *Model-load active memory:* 1.25
- *Post-cleanup active memory:* 0.00151
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1447
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.fastvlm.processing.FastVLMProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 81ffe929046666c43de53691147b1669ba0f3a4c
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--FastVLM-0.5B-bf16/snapshots/81ffe929046666c43de53691147b1669ba0f3a4c
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
A concrete 5-10-word title: Arundel Cathedral of Our Lady &amp; St. Philip Howard, Arundel, England, UK, GBR, Europe
Description: A photograph of the Arundel Cathedral of Our Lady and St. Philip Howard in Arundel, UK, taken on 2021-08-06 at 18:26:35 UTC+01:00, with a blue sky and a red car parked in front of the cathedral. The image captures the historic architecture of the cathedral, including its spires and Gothic elements, and the surrounding neighborhood.
</pre>

<details>
<summary>Exact raw output</summary>

```text
A concrete 5-10-word title: Arundel Cathedral of Our Lady & St. Philip Howard, Arundel, England, UK, GBR, Europe
Description: A photograph of the Arundel Cathedral of Our Lady and St. Philip Howard in Arundel, UK, taken on 2021-08-06 at 18:26:35 UTC+01:00, with a blue sky and a red car parked in front of the cathedral. The image captures the historic architecture of the cathedral, including its spires and Gothic elements, and the surrounding neighborhood.
```

</details>

</details>

---

<a id="model-mlx-community-glm-41v-9b-thinking-8bit"></a>

### mlx-community/GLM-4.1V-9B-Thinking-8bit

<details>
<summary>Complete evidence: mlx-community/GLM-4.1V-9B-Thinking-8bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Extra text appears before the Title field; Response appears
  cut off at the token limit; Internal reasoning block appears incomplete;
  Keyword list has 39 terms (requested 10-18); Duplicate keywords: england,
  red car, blue sky, neighborhood, church, cottage, stone wall, flower,
  parking, car, cathedral
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v)
- *Model load time:* 1.67s
- *Generation time:* 16.63s
- *Total time:* 18.30s
- *Input validation time:* 3.92e-05
- *Prompt preparation time:* 0.00164
- *First-token latency:* 5.59
- *Cleanup time:* 0.0914
- *Prompt tokens:* 6,479
- *Generation tokens:* 500
- *Total tokens:* 6,979
- *Prompt throughput (raw):* 1,160 tok/s
- *Generation throughput (raw):* 47.4 tok/s
- *Peak memory:* 13
- *Active memory:* 12
- *Cache memory:* 0.42
- *Model-load active memory:* 11.8
- *Post-cleanup active memory:* 0.00182
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1409
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.glm4v.processing.Glm46VProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 9677807f106500eb7690391c27645d59f6855cfb
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--GLM-4.1V-9B-Thinking-8bit/snapshots/9677807f106500eb7690391c27645d59f6855cfb
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|endoftext\|>

*Readable output:*

<pre class="model-output-readable">
&lt;think&gt;Got it, let's tackle this. First, the title needs to be 5-10 words. The main subject is Arundel Cathedral with surrounding buildings. So maybe "Arundel Cathedral and Neighborhood View".

Then the description: need to combine context (date/time, location) with visible elements. The image shows the cathedral, houses, a red car, blue sky, etc. So something like "Arundel Cathedral of Our Lady &amp; St. Philip Howard in Arundel, England, viewed from a residential neighborhood with a red car parked near the cathedral under a clear blue sky, captured on 2026-08-06."

Wait, but the description hint had October 31, 2021, but the authoritative context says 2026-08-06. So need to use the authoritative date. Also, the description should be factual. Let's make it concise.

Keywords need to be 10-18 unique. Let's list: Arundel Cathedral, England, UK, red car, blue sky, neighborhood, church, cottage, stone wall, flower, Gothic architecture, parking, roof, sky, brick building, residential area.

Now, let's structure each part.

Title: "Arundel Cathedral and Residential View"

Description: "Arundel Cathedral of Our Lady &amp; St. Philip Howard in Arundel, England, viewed from a residential neighborhood with a red car parked near the cathedral under a clear blue sky, captured on 2026-08-06."

Wait, need to make sure it's 1-2 sentences. Let's adjust. "Arundel Cathedral of Our Lady &amp; St. Philip Howard in Arundel, England, viewed from a residential area with a red car near the cathedral, under a clear blue sky, captured on 2026-08-06 UTC+01:00."

Keywords: Arundel Cathedral, England, UK, red car, blue sky, neighborhood, church, cottage, stone wall, flower, Gothic architecture, parking, roof, sky, brick building, residential area, stone wall, car, cathedral, England.

Wait, let's check the hints. The keyword hints include: Adobe Stock, Any Vision, Arundel, Arundel Cathedral of Our Lady &amp; St. Philip Howard, Blue sky, Bush, Car, Cathedral, Church, Cottage, England, Europe, Flower, French-Gothic, Neighborhood, Objects, Parking, Red Car
</pre>

<details>
<summary>Exact raw output</summary>

```text
<think>Got it, let's tackle this. First, the title needs to be 5-10 words. The main subject is Arundel Cathedral with surrounding buildings. So maybe "Arundel Cathedral and Neighborhood View". 

Then the description: need to combine context (date/time, location) with visible elements. The image shows the cathedral, houses, a red car, blue sky, etc. So something like "Arundel Cathedral of Our Lady & St. Philip Howard in Arundel, England, viewed from a residential neighborhood with a red car parked near the cathedral under a clear blue sky, captured on 2026-08-06."

Wait, but the description hint had October 31, 2021, but the authoritative context says 2026-08-06. So need to use the authoritative date. Also, the description should be factual. Let's make it concise.

Keywords need to be 10-18 unique. Let's list: Arundel Cathedral, England, UK, red car, blue sky, neighborhood, church, cottage, stone wall, flower, Gothic architecture, parking, roof, sky, brick building, residential area.

Now, let's structure each part.

Title: "Arundel Cathedral and Residential View"

Description: "Arundel Cathedral of Our Lady & St. Philip Howard in Arundel, England, viewed from a residential neighborhood with a red car parked near the cathedral under a clear blue sky, captured on 2026-08-06."

Wait, need to make sure it's 1-2 sentences. Let's adjust. "Arundel Cathedral of Our Lady & St. Philip Howard in Arundel, England, viewed from a residential area with a red car near the cathedral, under a clear blue sky, captured on 2026-08-06 UTC+01:00."

Keywords: Arundel Cathedral, England, UK, red car, blue sky, neighborhood, church, cottage, stone wall, flower, Gothic architecture, parking, roof, sky, brick building, residential area, stone wall, car, cathedral, England.

Wait, let's check the hints. The keyword hints include: Adobe Stock, Any Vision, Arundel, Arundel Cathedral of Our Lady & St. Philip Howard, Blue sky, Bush, Car, Cathedral, Church, Cottage, England, Europe, Flower, French-Gothic, Neighborhood, Objects, Parking, Red Car
```

</details>

</details>

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-2506-bf16"></a>

### mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

<details>
<summary>Complete evidence: mlx-community/Kimi-VL-A3B-Thinking-2506-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Missing or empty fields: Title, Description; Response
  repeats the task instructions instead of only returning the requested
  fields; Extra text appears before the Title field; Response appears cut off
  at the token limit; Internal reasoning block appears incomplete
- *Arch supported by installed mlx-vlm:* yes (model_type kimi_vl)
- *Model load time:* 3.30s
- *Generation time:* 126.05s
- *Total time:* 129.36s
- *Input validation time:* 6.68e-05
- *Prompt preparation time:* 0.00187
- *First-token latency:* 1.66
- *Cleanup time:* 0.172
- *Prompt tokens:* 1,355
- *Generation tokens:* 500
- *Total tokens:* 1,855
- *Prompt throughput (raw):* 818 tok/s
- *Generation throughput (raw):* 4.04 tok/s
- *Peak memory:* 40
- *Active memory:* 33
- *Cache memory:* 5.6
- *Model-load active memory:* 32.8
- *Post-cleanup active memory:* 0.00367
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1539
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.kimi_vl.processing_kimi_vl.KimiVLProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* fb254434d4026bee7aa840dea1c5d59feea8fd48
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Kimi-VL-A3B-Thinking-2506-bf16/snapshots/fb254434d4026bee7aa840dea1c5d59feea8fd48
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* [EOS]

*Readable output:*

<pre class="model-output-readable">
◁think▷Okay, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. The user wants a title, a descriptive paragraph, and 10-18 keywords.

Starting with the title: The main subject is Arundel Cathedral, so the title should be concise. Maybe "Arundel Cathedral of Our Lady &amp; St Philip Howard, Arundel, UK" but needs to be 5-10 words. Let's check the hint: "Arundel Cathedral of Our Lady &amp; St Philip Howard, Arundel, England, UK, GBR, Europe" is a bit long. Maybe shorten to "Arundel Cathedral of Our Lady &amp; St Philip Howard, Arundel, UK" (9 words).

Next, the description. The context says October 31, 2021, but the image is from 2026? Wait, the capture date is 2026, but the description hint is from 2021. Hmm, need to use the image's date? Wait, the user says to treat capture date/time as authoritative but not claim it's visible. So maybe use the image's date? Wait, the image's context says capture date is 2026-08-06, but the description hint is October 31, 2021. There's a conflict. The user says to retain details supported by the image, correct conflicts, and add important visible details. The image shows a clear view of the cathedral, houses, a red car, blue sky, etc. So maybe the description should be based on the image, not the hint's date. So the description should mention the cathedral, the red car, the houses, the blue sky, maybe the architectural details. So something like "View of Arundel Cathedral of Our Lady and St Philip Howard in Arundel, UK, with residential buildings, a red car, and a clear blue sky in the background." That's a 1-2 sentence.

Keywords: Need 10-18 unique keywords. From the image: Arundel Cathedral, Our Lady &amp; St Philip Howard, Arundel, UK, England, Europe, Red car, Blue sky, Brick buildings, Stone walls, Dormer windows, Church spires, Residential area, Flowers, Garage door, Street, Streetlamp, House, etc. Let's pick the most distinctive ones: Arundel Cathedral, Our Lady
</pre>

<details>
<summary>Exact raw output</summary>

```text
◁think▷Okay, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. The user wants a title, a descriptive paragraph, and 10-18 keywords. 

Starting with the title: The main subject is Arundel Cathedral, so the title should be concise. Maybe "Arundel Cathedral of Our Lady & St Philip Howard, Arundel, UK" but needs to be 5-10 words. Let's check the hint: "Arundel Cathedral of Our Lady & St Philip Howard, Arundel, England, UK, GBR, Europe" is a bit long. Maybe shorten to "Arundel Cathedral of Our Lady & St Philip Howard, Arundel, UK" (9 words).

Next, the description. The context says October 31, 2021, but the image is from 2026? Wait, the capture date is 2026, but the description hint is from 2021. Hmm, need to use the image's date? Wait, the user says to treat capture date/time as authoritative but not claim it's visible. So maybe use the image's date? Wait, the image's context says capture date is 2026-08-06, but the description hint is October 31, 2021. There's a conflict. The user says to retain details supported by the image, correct conflicts, and add important visible details. The image shows a clear view of the cathedral, houses, a red car, blue sky, etc. So maybe the description should be based on the image, not the hint's date. So the description should mention the cathedral, the red car, the houses, the blue sky, maybe the architectural details. So something like "View of Arundel Cathedral of Our Lady and St Philip Howard in Arundel, UK, with residential buildings, a red car, and a clear blue sky in the background." That's a 1-2 sentence.

Keywords: Need 10-18 unique keywords. From the image: Arundel Cathedral, Our Lady & St Philip Howard, Arundel, UK, England, Europe, Red car, Blue sky, Brick buildings, Stone walls, Dormer windows, Church spires, Residential area, Flowers, Garage door, Street, Streetlamp, House, etc. Let's pick the most distinctive ones: Arundel Cathedral, Our Lady
```

</details>

</details>

---

<a id="model-mlx-community-lfm25-vl-16b-bf16"></a>

### mlx-community/LFM2.5-VL-1.6B-bf16

<details>
<summary>Complete evidence: mlx-community/LFM2.5-VL-1.6B-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Response repeats the same text; Response appears cut off at
  the token limit; Title has 13 words (requested 5-10); Keyword list has 149
  terms (requested 10-18); Duplicate keywords: sunlight, historic, modern,
  parking, stonework, spires
- *Arch supported by installed mlx-vlm:* yes (model_type lfm2_vl)
- *Model load time:* 0.59s
- *Generation time:* 3.57s
- *Total time:* 4.17s
- *Input validation time:* 5.12e-05
- *Prompt preparation time:* 0.00393
- *First-token latency:* 0.0923
- *Cleanup time:* 0.082
- *Prompt tokens:* 615
- *Generation tokens:* 500
- *Total tokens:* 1,115
- *Prompt throughput (raw):* 6,660 tok/s
- *Generation throughput (raw):* 166 tok/s
- *Peak memory:* 4.1
- *Active memory:* 3.2
- *Cache memory:* 0.02
- *Model-load active memory:* 3.2
- *Post-cleanup active memory:* 0.0038
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1403
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* transformers.models.lfm2_vl.processing_lfm2_vl.Lfm2VlProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 16a710cf8afca206ff16a95a4ad6fe657f876ce1
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--LFM2.5-VL-1.6B-bf16/snapshots/16a710cf8afca206ff16a95a4ad6fe657f876ce1
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Arundel Cathedral of Our Lady &amp; St. Philip Howard, Arundel, England, UK, GBR, Europe

Description: A stunning view of Arundel Cathedral of Our Lady and St Philip Howard, showcasing its French-Gothic architecture against a clear blue sky. The cathedral's intricate spires and detailed stonework are highlighted by the warm sunlight, while a red car is parked nearby, adding a touch of modernity to the historic scene.

Keywords: Arundel Cathedral, French-Gothic architecture, Arundel, England, UK, GBR, Europe, Cathedral, Church, Red car, Blue sky, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Warm, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stone
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title: Arundel Cathedral of Our Lady & St. Philip Howard, Arundel, England, UK, GBR, Europe

Description: A stunning view of Arundel Cathedral of Our Lady and St Philip Howard, showcasing its French-Gothic architecture against a clear blue sky. The cathedral's intricate spires and detailed stonework are highlighted by the warm sunlight, while a red car is parked nearby, adding a touch of modernity to the historic scene.

Keywords: Arundel Cathedral, French-Gothic architecture, Arundel, England, UK, GBR, Europe, Cathedral, Church, Red car, Blue sky, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Warm, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stone
```

</details>

</details>

---

<a id="model-mlx-community-llama-32-11b-vision-instruct-8bit"></a>

### mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

<details>
<summary>Complete evidence: mlx-community/Llama-3.2-11B-Vision-Instruct-8bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Response repeats the same text; Response appears cut off at
  the token limit; Title has 13 words (requested 5-10); Keyword list has 59
  terms (requested 10-18); Duplicate keywords: st philip s church of our lady
  and st philip s church of our lady and st philip s
- *Arch supported by installed mlx-vlm:* yes (model_type mllama)
- *Model load time:* 1.52s
- *Generation time:* 30.98s
- *Total time:* 32.51s
- *Input validation time:* 4.7e-05
- *Prompt preparation time:* 0.00132
- *First-token latency:* 2.16
- *Cleanup time:* 0.0969
- *Prompt tokens:* 333
- *Generation tokens:* 500
- *Total tokens:* 833
- *Prompt throughput (raw):* 154 tok/s
- *Generation throughput (raw):* 17.7 tok/s
- *Peak memory:* 15
- *Active memory:* 11
- *Cache memory:* 0.69
- *Model-load active memory:* 11.4
- *Post-cleanup active memory:* 0.00433
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1456
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.mllama.processing_mllama.MllamaProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 8451adc50203b50b8f4199e75e753fb9c06e2af6
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Llama-3.2-11B-Vision-Instruct-8bit/snapshots/8451adc50203b50b8f4199e75e753fb9c06e2af6
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|eot_id\|>

*Readable output:*

<pre class="model-output-readable">
Title: Arundel Cathedral of Our Lady and St. Philip Howard in the Afternoon Sun

Description: The Arundel Cathedral of Our Lady and St. Philip Howard is captured in the late afternoon sun, showcasing its intricate architecture and surrounding neighborhood. The image features a red car parked in a garage, surrounded by a mix of old and new buildings, with the cathedral's spires and towers rising above the rooftops.

Keywords: Arundel, Cathedral, Church, Church Architecture, Church Building, Church Exterior, Church Interior, Church Landscaping, Church Lighting, Church Location, Church Neighborhood, Church Parking, Church Roof, Church Spires, Church Towers, Church Windows, English Architecture, European Architecture, Gothic Architecture, Historic Architecture, Historic Buildings, Historic Landmarks, Historic Landscapes, Historic Neighborhoods, Historic Places, Historic Sites, Historic Towns, Historic Villages, Old Buildings, Old Churches, Old Landmarks, Old Neighborhoods, Old Places, Old Towns, Old Villages, Religious Architecture, Religious Buildings, Religious Landmarks, Religious Landscapes, Religious Neighborhoods, Religious Places, Religious Sites, Religious Towns, Religious Villages, St. Philip Howard, St. Philip's Church, St. Philip's Cathedral, St. Philip's Church of Our Lady, St. Philip's Church of Our Lady and St. Philip, St. Philip's Church of Our Lady and St. Philip's, St. Philip's Church of Our Lady and St. Philip's Cathedral, St. Philip's Church of Our Lady and St. Philip's Church, St. Philip's Church of Our Lady and St. Philip's Church of Our Lady, St. Philip's Church of Our Lady and St. Philip's Church of Our Lady and St. Philip's, St. Philip's Church of Our Lady and St. Philip's Church of Our Lady and St. Philip's Church, St. Philip's Church of Our Lady and St. Philip's Church of Our Lady and St. Philip's Church of Our Lady, St. Philip's Church of Our Lady and St. Philip's Church of Our Lady and St. Philip's Church of Our Lady and St. Philip's, St. Philip's Church of Our Lady and St. Philip's Church of Our Lady and St. Philip's Church of Our Lady and St. Philip's Church, St. Philip's Church of Our Lady and St. Philip's Church of Our Lady and St. Philip's
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
- *Model load time:* 0.91s
- *Generation time:* 1.08s
- *Total time:* 2.01s
- *Input validation time:* 4.88e-05
- *Prompt preparation time:* 0.00795
- *First-token latency:* 0.256
- *Cleanup time:* 0.0896
- *Prompt tokens:* 965
- *Generation tokens:* 40
- *Total tokens:* 1,005
- *Prompt throughput (raw):* 3,768 tok/s
- *Generation throughput (raw):* 256 tok/s
- *Peak memory:* 3.8
- *Active memory:* 2.3
- *Cache memory:* 0.05
- *Model-load active memory:* 2.32
- *Post-cleanup active memory:* 0.00485
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1420
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.minicpmv4_6.processing_minicpmv4_6.MiniCPMVProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 03721395f6b82cd000cc74cde28fcff8abd9a04c
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--MiniCPM-V-4.6-8bit/snapshots/03721395f6b82cd000cc74cde28fcff8abd9a04c
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Stunning cathedral view in quaint British neighborhood.

Keywords: Arundel, cathedral, church, England, Europe, blue sky, red car, cottage, flower, brick, roof, neighborhood
</pre>

</details>

---

<a id="model-mlx-community-molmopoint-8b-fp16"></a>

### mlx-community/MolmoPoint-8B-fp16

<details>
<summary>Complete evidence: mlx-community/MolmoPoint-8B-fp16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* none
- *Observations:* Missing or empty fields: Title, Description, Keywords
- *Arch supported by installed mlx-vlm:* yes (model_type molmo_point)
- *Model load time:* 2.24s
- *Generation time:* 25.85s
- *Total time:* 28.10s
- *Input validation time:* 4.26e-05
- *Prompt preparation time:* 0.00485
- *First-token latency:* 2.55
- *Cleanup time:* 0.121
- *Prompt tokens:* 3,160
- *Generation tokens:* 134
- *Total tokens:* 3,294
- *Prompt throughput (raw):* 1,242 tok/s
- *Generation throughput (raw):* 5.98 tok/s
- *Peak memory:* 24
- *Active memory:* 19
- *Cache memory:* 5.0
- *Model-load active memory:* 17.4
- *Post-cleanup active memory:* 0.00657
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1390
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.molmo_point.processing_molmo_point.MolmoPointProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 0a60033b4e4813fb53df4c7523857d2ec972c7d9
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--MolmoPoint-8B-fp16/snapshots/0a60033b4e4813fb53df4c7523857d2ec972c7d9
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Arundel Cathedral: Gothic Beauty Amidst English Suburb

Captured on August 6, 2026, this image showcases Arundel Cathedral of Our Lady &amp; St. Philip Howard in Arundel, England. The grand Gothic structure stands prominently against a clear blue sky, surrounded by a mix of residential buildings and lush greenery. A red car is visible in the foreground, adding a modern touch to the historic scene.

Arundel Cathedral, Arundel, England, Blue sky, Bush, Car, Cathedral, Church, Cottage, Flower, French-Gothic, Neighborhood, Objects, Parking, Red Car, Roof, Sky
</pre>

<details>
<summary>Exact raw output</summary>

```text
Arundel Cathedral: Gothic Beauty Amidst English Suburb

Captured on August 6, 2026, this image showcases Arundel Cathedral of Our Lady & St. Philip Howard in Arundel, England. The grand Gothic structure stands prominently against a clear blue sky, surrounded by a mix of residential buildings and lush greenery. A red car is visible in the foreground, adding a modern touch to the historic scene.

Arundel Cathedral, Arundel, England, Blue sky, Bush, Car, Cathedral, Church, Cottage, Flower, French-Gothic, Neighborhood, Objects, Parking, Red Car, Roof, Sky
```

</details>

</details>

---

<a id="model-mlx-community-qwen3-vl-2b-thinking-bf16"></a>

### mlx-community/Qwen3-VL-2B-Thinking-bf16

<details>
<summary>Complete evidence: mlx-community/Qwen3-VL-2B-Thinking-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Missing or empty fields: Title, Description, Keywords;
  Response appears cut off at the token limit; Internal reasoning block
  appears incomplete
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Model load time:* 0.77s
- *Generation time:* 25.50s
- *Total time:* 26.29s
- *Input validation time:* 5.12e-05
- *Prompt preparation time:* 0.0122
- *First-token latency:* 18.4
- *Cleanup time:* 0.118
- *Prompt tokens:* 16,583
- *Generation tokens:* 500
- *Total tokens:* 17,083
- *Prompt throughput (raw):* 900 tok/s
- *Generation throughput (raw):* 81.1 tok/s
- *Peak memory:* 8.4
- *Active memory:* 4.3
- *Cache memory:* 2.4
- *Model-load active memory:* 4.26
- *Post-cleanup active memory:* 0.00808
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1432
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* c325e5ea14c215bb08fa0d668c81fa2581f9050b
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3-VL-2B-Thinking-bf16/snapshots/c325e5ea14c215bb08fa0d668c81fa2581f9050b
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Got it, let's tackle this step by step. First, the title. The context says it's Arundel Cathedral of Our Lady &amp; St. Philip Howard, so the title should be concise. The hint says "Arundel Cathedral of Our Lady &amp; St. Philip Howard, Arundel, England, UK, GBR, Europe" but we need a 5-10 word title. Let's check the image: it's a view of the cathedral, so maybe "Arundel Cathedral, England, UK, Gothic Architecture, Blue Sky, Residential Neighborhood" but need to be precise. Wait, the title hint is the main subject, so "Arundel Cathedral of Our Lady &amp; St. Philip Howard, Arundel, England" but maybe shorter. Let's go with "Arundel Cathedral, England, UK, Gothic, Blue Sky, Residential Area" but need to fit 5-10 words. Maybe "Arundel Cathedral of Our Lady &amp; St. Philip Howard, Arundel, England, UK" but that's 10 words. Wait, the title should be a concrete title. Let's see the image: the main subject is the cathedral, so "Arundel Cathedral, England, UK, Gothic, Blue Sky, Residential Neighborhood" but maybe "Arundel Cathedral, England, UK, Gothic Architecture, Blue Sky, Residential Area" – but let's check the context. The context says "Arundel Cathedral of Our Lady &amp; St. Philip Howard, Arundel, England, UK, GBR, Europe" so the title should include that. Wait, the title hint is "Arundel Cathedral of Our Lady &amp; St. Philip Howard, Arundel, England, UK, GBR, Europe" but we need a title. Let's make it "Arundel Cathedral of Our Lady &amp; St. Philip Howard, Arundel, England, UK" – that's 10 words. Wait, maybe "Arundel Cathedral, England, UK, Gothic, Blue Sky, Residential Neighborhood" – but let's check the image. The image shows the cathedral with a residential area in front, so the title should capture that. Let's go with "Arundel Cathedral of Our Lady &amp; St. Philip Howard, Arundel, England, UK" – that's 10 words. Wait, maybe "Arundel Cathedral, England, UK, Gothic
</pre>

<details>
<summary>Exact raw output</summary>

```text
Got it, let's tackle this step by step. First, the title. The context says it's Arundel Cathedral of Our Lady & St. Philip Howard, so the title should be concise. The hint says "Arundel Cathedral of Our Lady & St. Philip Howard, Arundel, England, UK, GBR, Europe" but we need a 5-10 word title. Let's check the image: it's a view of the cathedral, so maybe "Arundel Cathedral, England, UK, Gothic Architecture, Blue Sky, Residential Neighborhood" but need to be precise. Wait, the title hint is the main subject, so "Arundel Cathedral of Our Lady & St. Philip Howard, Arundel, England" but maybe shorter. Let's go with "Arundel Cathedral, England, UK, Gothic, Blue Sky, Residential Area" but need to fit 5-10 words. Maybe "Arundel Cathedral of Our Lady & St. Philip Howard, Arundel, England, UK" but that's 10 words. Wait, the title should be a concrete title. Let's see the image: the main subject is the cathedral, so "Arundel Cathedral, England, UK, Gothic, Blue Sky, Residential Neighborhood" but maybe "Arundel Cathedral, England, UK, Gothic Architecture, Blue Sky, Residential Area" – but let's check the context. The context says "Arundel Cathedral of Our Lady & St. Philip Howard, Arundel, England, UK, GBR, Europe" so the title should include that. Wait, the title hint is "Arundel Cathedral of Our Lady & St. Philip Howard, Arundel, England, UK, GBR, Europe" but we need a title. Let's make it "Arundel Cathedral of Our Lady & St. Philip Howard, Arundel, England, UK" – that's 10 words. Wait, maybe "Arundel Cathedral, England, UK, Gothic, Blue Sky, Residential Neighborhood" – but let's check the image. The image shows the cathedral with a residential area in front, so the title should capture that. Let's go with "Arundel Cathedral of Our Lady & St. Philip Howard, Arundel, England, UK" – that's 10 words. Wait, maybe "Arundel Cathedral, England, UK, Gothic
```

</details>

</details>

---

<a id="model-mlx-community-x-reasoner-7b-8bit"></a>

### mlx-community/X-Reasoner-7B-8bit

<details>
<summary>Complete evidence: mlx-community/X-Reasoner-7B-8bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Response repeats the same text; Response appears cut off at
  the token limit; Keyword list has 184 terms (requested 10-18); Duplicate
  keywords: stone wall, dusk, peaceful, historic, traditional, residential,
  urban, architectural, landmark, cultural, scenic, picturesque, tranquil,
  serene, calm, quiet
- *Arch supported by installed mlx-vlm:* yes (model_type qwen2_5_vl)
- *Model load time:* 1.16s
- *Generation time:* 30.85s
- *Total time:* 32.02s
- *Input validation time:* 5.57e-05
- *Prompt preparation time:* 0.000263
- *First-token latency:* 20.3
- *Cleanup time:* 0.168
- *Prompt tokens:* 16,592
- *Generation tokens:* 500
- *Total tokens:* 17,092
- *Prompt throughput (raw):* 816 tok/s
- *Generation throughput (raw):* 51.8 tok/s
- *Peak memory:* 13
- *Active memory:* 9.5
- *Cache memory:* 1.3
- *Model-load active memory:* 9.46
- *Post-cleanup active memory:* 0.0103
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1482
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen2_5_vl.processing_qwen2_5_vl.Qwen2_5_VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 21732e74613b465bc98e9d5ec210aba5c7adbcc1
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--X-Reasoner-7B-8bit/snapshots/21732e74613b465bc98e9d5ec210aba5c7adbcc1
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Arundel Cathedral and Cottage at Dusk

Description: A serene view of Arundel Cathedral of Our Lady &amp; St. Philip Howard, framed by a quaint cottage and vibrant red car, under a clear blue sky at dusk.

Keywords: Arundel, Arundel Cathedral, England, UK, Europe, Gothic architecture, French-Gothic, blue sky, cottage, red car, parking, roof, flower, bush, neighborhood, streetlamp, stone wall, window, cross, steeple, garage, 20 mph sign, brick building, white house, stone pathway, lamp post, 2026-08-06, 18:26:35 UTC+01:00, 50.806659°N, 0.551382°W, flower box, potted flowers, stone fence, white garage door, satellite dish, chimney, streetlight, lamp, window box, flower pot, stone wall, brick, white, red, blue, green, yellow, dusk, sunlight, shadow, clear sky, peaceful, historic, traditional, residential, urban, architectural, landmark, cultural, scenic, picturesque, tranquil, serene, calm, quiet, evening, sunset, dusk, twilight, peaceful, calm, quiet, serene, tranquil, historic, traditional, residential, urban, architectural, landmark, cultural, scenic, picturesque, tranquil, calm, quiet, serene, peaceful, historic, traditional, residential, urban, architectural, landmark, cultural, scenic, picturesque, tranquil, calm, quiet, serene, peaceful, historic, traditional, residential, urban, architectural, landmark, cultural, scenic, picturesque, tranquil, calm, quiet, serene, peaceful, historic, traditional, residential, urban, architectural, landmark, cultural, scenic, picturesque, tranquil, calm, quiet, serene, peaceful, historic, traditional, residential, urban, architectural, landmark, cultural, scenic, picturesque, tranquil, calm, quiet, serene, peaceful, historic, traditional, residential, urban, architectural, landmark, cultural, scenic, picturesque, tranquil, calm, quiet, serene, peaceful, historic, traditional, residential, urban, architectural, landmark, cultural, scenic, picturesque, tranquil, calm, quiet, serene, peaceful, historic, traditional, residential, urban, architectural, landmark, cultural, scenic, picturesque, tranquil, calm, quiet,
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title: Arundel Cathedral and Cottage at Dusk

Description: A serene view of Arundel Cathedral of Our Lady & St. Philip Howard, framed by a quaint cottage and vibrant red car, under a clear blue sky at dusk.

Keywords: Arundel, Arundel Cathedral, England, UK, Europe, Gothic architecture, French-Gothic, blue sky, cottage, red car, parking, roof, flower, bush, neighborhood, streetlamp, stone wall, window, cross, steeple, garage, 20 mph sign, brick building, white house, stone pathway, lamp post, 2026-08-06, 18:26:35 UTC+01:00, 50.806659°N, 0.551382°W, flower box, potted flowers, stone fence, white garage door, satellite dish, chimney, streetlight, lamp, window box, flower pot, stone wall, brick, white, red, blue, green, yellow, dusk, sunlight, shadow, clear sky, peaceful, historic, traditional, residential, urban, architectural, landmark, cultural, scenic, picturesque, tranquil, serene, calm, quiet, evening, sunset, dusk, twilight, peaceful, calm, quiet, serene, tranquil, historic, traditional, residential, urban, architectural, landmark, cultural, scenic, picturesque, tranquil, calm, quiet, serene, peaceful, historic, traditional, residential, urban, architectural, landmark, cultural, scenic, picturesque, tranquil, calm, quiet, serene, peaceful, historic, traditional, residential, urban, architectural, landmark, cultural, scenic, picturesque, tranquil, calm, quiet, serene, peaceful, historic, traditional, residential, urban, architectural, landmark, cultural, scenic, picturesque, tranquil, calm, quiet, serene, peaceful, historic, traditional, residential, urban, architectural, landmark, cultural, scenic, picturesque, tranquil, calm, quiet, serene, peaceful, historic, traditional, residential, urban, architectural, landmark, cultural, scenic, picturesque, tranquil, calm, quiet, serene, peaceful, historic, traditional, residential, urban, architectural, landmark, cultural, scenic, picturesque, tranquil, calm, quiet, serene, peaceful, historic, traditional, residential, urban, architectural, landmark, cultural, scenic, picturesque, tranquil, calm, quiet,
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
- *Observations:* Missing or empty fields: Title, Description, Keywords
- *Arch supported by installed mlx-vlm:* yes (model_type gemma3n)
- *Model load time:* 4.39s
- *Generation time:* 11.86s
- *Total time:* 16.26s
- *Input validation time:* 0.000116
- *Prompt preparation time:* 0.0027
- *First-token latency:* 0.405
- *Cleanup time:* 0.142
- *Prompt tokens:* 617
- *Generation tokens:* 500
- *Total tokens:* 1,117
- *Prompt throughput (raw):* 1,524 tok/s
- *Generation throughput (raw):* 45.8 tok/s
- *Peak memory:* 17
- *Active memory:* 16
- *Cache memory:* 0.04
- *Model-load active memory:* 15.7
- *Post-cleanup active memory:* 0.0114
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1409
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.gemma3n.processing_gemma3n.Gemma3nProcessor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* d9c02d0b2fa8cf26c1cb5dd9e756db59cdbe8a4a
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--gemma-3n-E4B-it-bf16/snapshots/d9c02d0b2fa8cf26c1cb5dd9e756db59cdbe8a4a
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;&lt;start_of_image&gt;
</pre>

<details>
<summary>Exact raw output</summary>

```text
<start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image><start_of_image>
```

</details>

</details>

---

<a id="model-mlx-community-llava-v16-mistral-7b-8bit"></a>

### mlx-community/llava-v1.6-mistral-7b-8bit

<details>
<summary>Complete evidence: mlx-community/llava-v1.6-mistral-7b-8bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Response repeats the same text; Missing or empty fields:
  Title, Description; Extra text appears before the Title field; Response
  appears cut off at the token limit
- *Arch supported by installed mlx-vlm:* yes (model_type llava_next)
- *Model load time:* 0.94s
- *Generation time:* 12.66s
- *Total time:* 13.61s
- *Input validation time:* 5.03e-05
- *Prompt preparation time:* 0.00223
- *First-token latency:* 2.84
- *Cleanup time:* 0.161
- *Prompt tokens:* 2,552
- *Generation tokens:* 500
- *Total tokens:* 3,052
- *Prompt throughput (raw):* 898 tok/s
- *Generation throughput (raw):* 55.6 tok/s
- *Peak memory:* 9.7
- *Active memory:* 8.1
- *Cache memory:* 0.92
- *Model-load active memory:* 8.05
- *Post-cleanup active memory:* 0.0125
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1354
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.llava_next.processing_llava_next.LlavaNextProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* b8df5f329d95a7abe6429ed46093f9b84e8e6396
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--llava-v1.6-mistral-7b-8bit/snapshots/b8df5f329d95a7abe6429ed46093f9b84e8e6396
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Arundel Cathedral of Our Lady &amp; St. Philip Howard, Arundel, UK - October 31, 2021: View of Arundel Cathedral of Our Lady and St Philip Howard

Title:
Description:
Keywords: [/] Arundel Cathedral of Our Lady &amp; St. Philip Howard, Arundel, UK - October 31, 2021: View of Arundel Cathedral of Our Lady and St Philip Howard

Title:
Description:
Keywords: [/] Arundel Cathedral of Our Lady &amp; St. Philip Howard, Arundel, UK - October 31, 2021: View of Arundel Cathedral of Our Lady and St Philip Howard

Title:
Description:
Keywords: [/] Arundel Cathedral of Our Lady &amp; St. Philip Howard, Arundel, UK - October 31, 2021: View of Arundel Cathedral of Our Lady and St Philip Howard

Title:
Description:
Keywords: [/] Arundel Cathedral of Our Lady &amp; St. Philip Howard, Arundel, UK - October 31, 2021: View of Arundel Cathedral of Our Lady and St Philip Howard

Title:
Description:
Keywords: [/] Arundel Cathedral of Our Lady &amp; St. Philip Howard, Arundel, UK - October 31, 2021: View of Arundel Cathedral of Our Lady and St Philip Howard

Title:
Description:
Keywords: [/] Arundel Cathedral of Our Lady &amp; St. Philip Howard, Arundel, UK - October 31, 2021: View of Arundel Cathedral of Our Lady and St Philip Howard

Title:
Description:
Keywords: [/] Arundel Cathedral of Our Lady &amp; St. Philip Howard, Arundel, UK - October 31, 2021: View of Arundel Cathedral of Our Lady and St Philip Howard

Title:
Description:
Keywords: [/car, church, cathedral, arundel, uk, europe, sky, roof, building, architecture, gothic, windows,
</pre>

<details>
<summary>Exact raw output</summary>

```text
Arundel Cathedral of Our Lady & St. Philip Howard, Arundel, UK - October 31, 2021: View of Arundel Cathedral of Our Lady and St Philip Howard

Title:
Description:
Keywords: [/] Arundel Cathedral of Our Lady & St. Philip Howard, Arundel, UK - October 31, 2021: View of Arundel Cathedral of Our Lady and St Philip Howard

Title:
Description:
Keywords: [/] Arundel Cathedral of Our Lady & St. Philip Howard, Arundel, UK - October 31, 2021: View of Arundel Cathedral of Our Lady and St Philip Howard

Title:
Description:
Keywords: [/] Arundel Cathedral of Our Lady & St. Philip Howard, Arundel, UK - October 31, 2021: View of Arundel Cathedral of Our Lady and St Philip Howard

Title:
Description:
Keywords: [/] Arundel Cathedral of Our Lady & St. Philip Howard, Arundel, UK - October 31, 2021: View of Arundel Cathedral of Our Lady and St Philip Howard

Title:
Description:
Keywords: [/] Arundel Cathedral of Our Lady & St. Philip Howard, Arundel, UK - October 31, 2021: View of Arundel Cathedral of Our Lady and St Philip Howard

Title:
Description:
Keywords: [/] Arundel Cathedral of Our Lady & St. Philip Howard, Arundel, UK - October 31, 2021: View of Arundel Cathedral of Our Lady and St Philip Howard

Title:
Description:
Keywords: [/] Arundel Cathedral of Our Lady & St. Philip Howard, Arundel, UK - October 31, 2021: View of Arundel Cathedral of Our Lady and St Philip Howard

Title:
Description:
Keywords: [/car, church, cathedral, arundel, uk, europe, sky, roof, building, architecture, gothic, windows,
```

</details>

</details>

---

<a id="model-mlx-community-nanollava-15-4bit"></a>

### mlx-community/nanoLLaVA-1.5-4bit

<details>
<summary>Complete evidence: mlx-community/nanoLLaVA-1.5-4bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* none
- *Observations:* Missing or empty fields: Keywords
- *Arch supported by installed mlx-vlm:* yes (model_type llava-qwen2 via
  llava_bunny)
- *Model load time:* 0.66s
- *Generation time:* 1.15s
- *Total time:* 1.81s
- *Input validation time:* 5.03e-05
- *Prompt preparation time:* 0.00187
- *First-token latency:* 0.0863
- *Cleanup time:* 0.103
- *Prompt tokens:* 364
- *Generation tokens:* 77
- *Total tokens:* 441
- *Prompt throughput (raw):* 4,218 tok/s
- *Generation throughput (raw):* 224 tok/s
- *Peak memory:* 2.2
- *Active memory:* 0.61
- *Cache memory:* 0.28
- *Model-load active memory:* 0.611
- *Post-cleanup active memory:* 0.0131
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1438
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 5240204744963d72823e5de933c528c4aa82dfca
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--nanoLLaVA-1.5-4bit/snapshots/5240204744963d72823e5de933c528c4aa82dfca
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Arundel Cathedral of Our Lady &amp; St. Philip Howard, Arundel, UK - October 31, 2021: View of Arundel Cathedral of Our Lady and St. Philip Howard
Description: A 10-minute description of the cathedral, including its location, the date and time of the view, and the surrounding area.
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title: Arundel Cathedral of Our Lady & St. Philip Howard, Arundel, UK - October 31, 2021: View of Arundel Cathedral of Our Lady and St. Philip Howard
Description: A 10-minute description of the cathedral, including its location, the date and time of the view, and the surrounding area.
```

</details>

</details>

---

<a id="model-mlx-community-paligemma2-3b-pt-896-4bit"></a>

### mlx-community/paligemma2-3b-pt-896-4bit

<details>
<summary>Complete evidence: mlx-community/paligemma2-3b-pt-896-4bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Response repeats the same text; Missing or empty fields:
  Title, Description, Keywords; Response repeats the task instructions instead
  of only returning the requested fields; Response appears cut off at the
  token limit
- *Arch supported by installed mlx-vlm:* yes (model_type paligemma)
- *Model load time:* 1.45s
- *Generation time:* 14.32s
- *Total time:* 15.78s
- *Input validation time:* 4.55e-05
- *Prompt preparation time:* 0.000198
- *First-token latency:* 1.16
- *Cleanup time:* 0.155
- *Prompt tokens:* 4,444
- *Generation tokens:* 500
- *Total tokens:* 4,944
- *Prompt throughput (raw):* 3,847 tok/s
- *Generation throughput (raw):* 40.9 tok/s
- *Peak memory:* 4.4
- *Active memory:* 1.7
- *Cache memory:* 1.8
- *Model-load active memory:* 1.73
- *Post-cleanup active memory:* 0.0142
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* 1338
- *Image placeholders:* 1
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.paligemma.processing_paligemma.PaliGemmaProcessor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* a26bac48c7a661dfdafe1799c90177f818e79925
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--paligemma2-3b-pt-896-4bit/snapshots/a26bac48c7a661dfdafe1799c90177f818e79925
- *Generation settings:* {"max_tokens": 500, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
- Title hint:
- Description hint:
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18
</pre>

</details>

---

<a id="model-mlx-community-inkling-small-mlx-4bit"></a>

### mlx-community/Inkling-Small-mlx-4bit

<details>
<summary>Complete evidence: mlx-community/Inkling-Small-mlx-4bit</summary>

- *Execution:* crashed
- *Usability:* not_evaluated
- *Maintainer status:* actionable_failure
- *Observations:* none
- *Failure phase:* model_load
- *Error stage:* Model Error
- *Error code:* MLX_VLM_MODEL_LOAD_MODEL
- *Error type:* ValueError
- *Error package:* mlx-vlm
- *Error message:* Model loading failed: Received 362 parameters not in model:
  <br>audio_tower.encoder.biases,<br>audio_tower.encoder.scales,<br>language_model.model.layers.10.mlp.experts.down_proj.biases,<br>language_model.model.layers.10.mlp.experts.down_proj.scales,<br>language_model.model.layers.10.mlp.experts.down_proj.weight,<br>language_model.model.layers.10.mlp.experts.gate_proj.biases,<br>language_model.model.layers.10.mlp.experts.gate_proj.scales,<br>language_model.model.layers.10.mlp.experts.gate_proj.weight,<br>language_model.model.layers.10.mlp.experts.up_proj.biases,<br>language_model.model.layers.10.mlp.experts.up_proj.scales,<br>language_model.model.layers.10.mlp.experts.up_proj.weight,<br>language_model.model.layers.11.mlp.experts.down_proj.biases,<br>language_model.model.layers.11.mlp.experts.down_proj.scales,<br>language_model.model.layers.11.mlp.experts.down_proj.weight,<br>language_model.model.layers.11.mlp.experts.gate_proj.biases,<br>language_model.model.layers.11.mlp.experts.gate_proj.scales,<br>language_model.model.layers.11.mlp.experts.gate_proj.weight,<br>language_model.model.layers.11.mlp.experts.up_proj.biases,<br>language_model.model.layers.11.mlp.experts.up_proj.scales,<br>language_model.model.layers.11.mlp.experts.up_proj.weight,<br>language_model.model.layers.12.mlp.experts.down_proj.biases,<br>language_model.model.layers.12.mlp.experts.down_proj.scales,<br>language_model.model.layers.12.mlp.experts.down_proj.weight,<br>language_model.model.layers.12.mlp.experts.gate_proj.biases,<br>language_model.model.layers.12.mlp.experts.gate_proj.scales,<br>language_model.model.layers.12.mlp.experts.gate_proj.weight,<br>language_model.model.layers.12.mlp.experts.up_proj.biases,<br>language_model.model.layers.12.mlp.experts.up_proj.scales,<br>language_model.model.layers.12.mlp.experts.up_proj.weight,<br>language_model.model.layers.13.mlp.experts.down_proj.biases,<br>language_model.model.layers.13.mlp.experts.down_proj.scales,<br>language_model.model.layers.13.mlp.experts.down_proj.weight,<br>language_model.model.layers.13.mlp.experts.gate_proj.biases,<br>language_model.model.layers.13.mlp.experts.gate_proj.scales,<br>language_model.model.layers.13.mlp.experts.gate_proj.weight,<br>language_model.model.layers.13.mlp.experts.up_proj.biases,<br>language_model.model.layers.13.mlp.experts.up_proj.scales,<br>language_model.model.layers.13.mlp.experts.up_proj.weight,<br>language_model.model.layers.14.mlp.experts.down_proj.biases,<br>language_model.model.layers.14.mlp.experts.down_proj.scales,<br>language_model.model.layers.14.mlp.experts.down_proj.weight,<br>language_model.model.layers.14.mlp.experts.gate_proj.biases,<br>language_model.model.layers.14.mlp.experts.gate_proj.scales,<br>language_model.model.layers.14.mlp.experts.gate_proj.weight,<br>language_model.model.layers.14.mlp.experts.up_proj.biases,<br>language_model.model.layers.14.mlp.experts.up_proj.scales,<br>language_model.model.layers.14.mlp.experts.up_proj.weight,<br>language_model.model.layers.15.mlp.experts.down_proj.biases,<br>language_model.model.layers.15.mlp.experts.down_proj.scales,<br>language_model.model.layers.15.mlp.experts.down_proj.weight,<br>language_model.model.layers.15.mlp.experts.gate_proj.biases,<br>language_model.model.layers.15.mlp.experts.gate_proj.scales,<br>language_model.model.layers.15.mlp.experts.gate_proj.weight,<br>language_model.model.layers.15.mlp.experts.up_proj.biases,<br>language_model.model.layers.15.mlp.experts.up_proj.scales,<br>language_model.model.layers.15.mlp.experts.up_proj.weight,<br>language_model.model.layers.16.mlp.experts.down_proj.biases,<br>language_model.model.layers.16.mlp.experts.down_proj.scales,<br>language_model.model.layers.16.mlp.experts.down_proj.weight,<br>language_model.model.layers.16.mlp.experts.gate_proj.biases,<br>language_model.model.layers.16.mlp.experts.gate_proj.scales,<br>language_model.model.layers.16.mlp.experts.gate_proj.weight,<br>language_model.model.layers.16.mlp.experts.up_proj.biases,<br>language_model.model.layers.16.mlp.experts.up_proj.scales,<br>language_model.model.layers.16.mlp.experts.up_proj.weight,<br>language_model.model.layers.17.mlp.experts.down_proj.biases,<br>language_model.model.layers.17.mlp.experts.down_proj.scales,<br>language_model.model.layers.17.mlp.experts.down_proj.weight,<br>language_model.model.layers.17.mlp.experts.gate_proj.biases,<br>language_model.model.layers.17.mlp.experts.gate_proj.scales,<br>language_model.model.layers.17.mlp.experts.gate_proj.weight,<br>language_model.model.layers.17.mlp.experts.up_proj.biases,<br>language_model.model.layers.17.mlp.experts.up_proj.scales,<br>language_model.model.layers.17.mlp.experts.up_proj.weight,<br>language_model.model.layers.18.mlp.experts.down_proj.biases,<br>language_model.model.layers.18.mlp.experts.down_proj.scales,<br>language_model.model.layers.18.mlp.experts.down_proj.weight,<br>language_model.model.layers.18.mlp.experts.gate_proj.biases,<br>language_model.model.layers.18.mlp.experts.gate_proj.scales,<br>language_model.model.layers.18.mlp.experts.gate_proj.weight,<br>language_model.model.layers.18.mlp.experts.up_proj.biases,<br>language_model.model.layers.18.mlp.experts.up_proj.scales,<br>language_model.model.layers.18.mlp.experts.up_proj.weight,<br>language_model.model.layers.19.mlp.experts.down_proj.biases,<br>language_model.model.layers.19.mlp.experts.down_proj.scales,<br>language_model.model.layers.19.mlp.experts.down_proj.weight,<br>language_model.model.layers.19.mlp.experts.gate_proj.biases,<br>language_model.model.layers.19.mlp.experts.gate_proj.scales,<br>language_model.model.layers.19.mlp.experts.gate_proj.weight,<br>language_model.model.layers.19.mlp.experts.up_proj.biases,<br>language_model.model.layers.19.mlp.experts.up_proj.scales,<br>language_model.model.layers.19.mlp.experts.up_proj.weight,<br>language_model.model.layers.2.mlp.experts.down_proj.biases,<br>language_model.model.layers.2.mlp.experts.down_proj.scales,<br>language_model.model.layers.2.mlp.experts.down_proj.weight,<br>language_model.model.layers.2.mlp.experts.gate_proj.biases,<br>language_model.model.layers.2.mlp.experts.gate_proj.scales,<br>language_model.model.layers.2.mlp.experts.gate_proj.weight,<br>language_model.model.layers.2.mlp.experts.up_proj.biases,<br>language_model.model.layers.2.mlp.experts.up_proj.scales,<br>language_model.model.layers.2.mlp.experts.up_proj.weight,<br>language_model.model.layers.20.mlp.experts.down_proj.biases,<br>language_model.model.layers.20.mlp.experts.down_proj.scales,<br>language_model.model.layers.20.mlp.experts.down_proj.weight,<br>language_model.model.layers.20.mlp.experts.gate_proj.biases,<br>language_model.model.layers.20.mlp.experts.gate_proj.scales,<br>language_model.model.layers.20.mlp.experts.gate_proj.weight,<br>language_model.model.layers.20.mlp.experts.up_proj.biases,<br>language_model.model.layers.20.mlp.experts.up_proj.scales,<br>language_model.model.layers.20.mlp.experts.up_proj.weight,<br>language_model.model.layers.21.mlp.experts.down_proj.biases,<br>language_model.model.layers.21.mlp.experts.down_proj.scales,<br>language_model.model.layers.21.mlp.experts.down_proj.weight,<br>language_model.model.layers.21.mlp.experts.gate_proj.biases,<br>language_model.model.layers.21.mlp.experts.gate_proj.scales,<br>language_model.model.layers.21.mlp.experts.gate_proj.weight,<br>language_model.model.layers.21.mlp.experts.up_proj.biases,<br>language_model.model.layers.21.mlp.experts.up_proj.scales,<br>language_model.model.layers.21.mlp.experts.up_proj.weight,<br>language_model.model.layers.22.mlp.experts.down_proj.biases,<br>language_model.model.layers.22.mlp.experts.down_proj.scales,<br>language_model.model.layers.22.mlp.experts.down_proj.weight,<br>language_model.model.layers.22.mlp.experts.gate_proj.biases,<br>language_model.model.layers.22.mlp.experts.gate_proj.scales,<br>language_model.model.layers.22.mlp.experts.gate_proj.weight,<br>language_model.model.layers.22.mlp.experts.up_proj.biases,<br>language_model.model.layers.22.mlp.experts.up_proj.scales,<br>language_model.model.layers.22.mlp.experts.up_proj.weight,<br>language_model.model.layers.23.mlp.experts.down_proj.biases,<br>language_model.model.layers.23.mlp.experts.down_proj.scales,<br>language_model.model.layers.23.mlp.experts.down_proj.weight,<br>language_model.model.layers.23.mlp.experts.gate_proj.biases,<br>language_model.model.layers.23.mlp.experts.gate_proj.scales,<br>language_model.model.layers.23.mlp.experts.gate_proj.weight,<br>language_model.model.layers.23.mlp.experts.up_proj.biases,<br>language_model.model.layers.23.mlp.experts.up_proj.scales,<br>language_model.model.layers.23.mlp.experts.up_proj.weight,<br>language_model.model.layers.24.mlp.experts.down_proj.biases,<br>language_model.model.layers.24.mlp.experts.down_proj.scales,<br>language_model.model.layers.24.mlp.experts.down_proj.weight,<br>language_model.model.layers.24.mlp.experts.gate_proj.biases,<br>language_model.model.layers.24.mlp.experts.gate_proj.scales,<br>language_model.model.layers.24.mlp.experts.gate_proj.weight,<br>language_model.model.layers.24.mlp.experts.up_proj.biases,<br>language_model.model.layers.24.mlp.experts.up_proj.scales,<br>language_model.model.layers.24.mlp.experts.up_proj.weight,<br>language_model.model.layers.25.mlp.experts.down_proj.biases,<br>language_model.model.layers.25.mlp.experts.down_proj.scales,<br>language_model.model.layers.25.mlp.experts.down_proj.weight,<br>language_model.model.layers.25.mlp.experts.gate_proj.biases,<br>language_model.model.layers.25.mlp.experts.gate_proj.scales,<br>language_model.model.layers.25.mlp.experts.gate_proj.weight,<br>language_model.model.layers.25.mlp.experts.up_proj.biases,<br>language_model.model.layers.25.mlp.experts.up_proj.scales,<br>language_model.model.layers.25.mlp.experts.up_proj.weight,<br>language_model.model.layers.26.mlp.experts.down_proj.biases,<br>language_model.model.layers.26.mlp.experts.down_proj.scales,<br>language_model.model.layers.26.mlp.experts.down_proj.weight,<br>language_model.model.layers.26.mlp.experts.gate_proj.biases,<br>language_model.model.layers.26.mlp.experts.gate_proj.scales,<br>language_model.model.layers.26.mlp.experts.gate_proj.weight,<br>language_model.model.layers.26.mlp.experts.up_proj.biases,<br>language_model.model.layers.26.mlp.experts.up_proj.scales,<br>language_model.model.layers.26.mlp.experts.up_proj.weight,<br>language_model.model.layers.27.mlp.experts.down_proj.biases,<br>language_model.model.layers.27.mlp.experts.down_proj.scales,<br>language_model.model.layers.27.mlp.experts.down_proj.weight,<br>language_model.model.layers.27.mlp.experts.gate_proj.biases,<br>language_model.model.layers.27.mlp.experts.gate_proj.scales,<br>language_model.model.layers.27.mlp.experts.gate_proj.weight,<br>language_model.model.layers.27.mlp.experts.up_proj.biases,<br>language_model.model.layers.27.mlp.experts.up_proj.scales,<br>language_model.model.layers.27.mlp.experts.up_proj.weight,<br>language_model.model.layers.28.mlp.experts.down_proj.biases,<br>language_model.model.layers.28.mlp.experts.down_proj.scales,<br>language_model.model.layers.28.mlp.experts.down_proj.weight,<br>language_model.model.layers.28.mlp.experts.gate_proj.biases,<br>language_model.model.layers.28.mlp.experts.gate_proj.scales,<br>language_model.model.layers.28.mlp.experts.gate_proj.weight,<br>language_model.model.layers.28.mlp.experts.up_proj.biases,<br>language_model.model.layers.28.mlp.experts.up_proj.scales,<br>language_model.model.layers.28.mlp.experts.up_proj.weight,<br>language_model.model.layers.29.mlp.experts.down_proj.biases,<br>language_model.model.layers.29.mlp.experts.down_proj.scales,<br>language_model.model.layers.29.mlp.experts.down_proj.weight,<br>language_model.model.layers.29.mlp.experts.gate_proj.biases,<br>language_model.model.layers.29.mlp.experts.gate_proj.scales,<br>language_model.model.layers.29.mlp.experts.gate_proj.weight,<br>language_model.model.layers.29.mlp.experts.up_proj.biases,<br>language_model.model.layers.29.mlp.experts.up_proj.scales,<br>language_model.model.layers.29.mlp.experts.up_proj.weight,<br>language_model.model.layers.3.mlp.experts.down_proj.biases,<br>language_model.model.layers.3.mlp.experts.down_proj.scales,<br>language_model.model.layers.3.mlp.experts.down_proj.weight,<br>language_model.model.layers.3.mlp.experts.gate_proj.biases,<br>language_model.model.layers.3.mlp.experts.gate_proj.scales,<br>language_model.model.layers.3.mlp.experts.gate_proj.weight,<br>language_model.model.layers.3.mlp.experts.up_proj.biases,<br>language_model.model.layers.3.mlp.experts.up_proj.scales,<br>language_model.model.layers.3.mlp.experts.up_proj.weight,<br>language_model.model.layers.30.mlp.experts.down_proj.biases,<br>language_model.model.layers.30.mlp.experts.down_proj.scales,<br>language_model.model.layers.30.mlp.experts.down_proj.weight,<br>language_model.model.layers.30.mlp.experts.gate_proj.biases,<br>language_model.model.layers.30.mlp.experts.gate_proj.scales,<br>language_model.model.layers.30.mlp.experts.gate_proj.weight,<br>language_model.model.layers.30.mlp.experts.up_proj.biases,<br>language_model.model.layers.30.mlp.experts.up_proj.scales,<br>language_model.model.layers.30.mlp.experts.up_proj.weight,<br>language_model.model.layers.31.mlp.experts.down_proj.biases,<br>language_model.model.layers.31.mlp.experts.down_proj.scales,<br>language_model.model.layers.31.mlp.experts.down_proj.weight,<br>language_model.model.layers.31.mlp.experts.gate_proj.biases,<br>language_model.model.layers.31.mlp.experts.gate_proj.scales,<br>language_model.model.layers.31.mlp.experts.gate_proj.weight,<br>language_model.model.layers.31.mlp.experts.up_proj.biases,<br>language_model.model.layers.31.mlp.experts.up_proj.scales,<br>language_model.model.layers.31.mlp.experts.up_proj.weight,<br>language_model.model.layers.32.mlp.experts.down_proj.biases,<br>language_model.model.layers.32.mlp.experts.down_proj.scales,<br>language_model.model.layers.32.mlp.experts.down_proj.weight,<br>language_model.model.layers.32.mlp.experts.gate_proj.biases,<br>language_model.model.layers.32.mlp.experts.gate_proj.scales,<br>language_model.model.layers.32.mlp.experts.gate_proj.weight,<br>language_model.model.layers.32.mlp.experts.up_proj.biases,<br>language_model.model.layers.32.mlp.experts.up_proj.scales,<br>language_model.model.layers.32.mlp.experts.up_proj.weight,<br>language_model.model.layers.33.mlp.experts.down_proj.biases,<br>language_model.model.layers.33.mlp.experts.down_proj.scales,<br>language_model.model.layers.33.mlp.experts.down_proj.weight,<br>language_model.model.layers.33.mlp.experts.gate_proj.biases,<br>language_model.model.layers.33.mlp.experts.gate_proj.scales,<br>language_model.model.layers.33.mlp.experts.gate_proj.weight,<br>language_model.model.layers.33.mlp.experts.up_proj.biases,<br>language_model.model.layers.33.mlp.experts.up_proj.scales,<br>language_model.model.layers.33.mlp.experts.up_proj.weight,<br>language_model.model.layers.34.mlp.experts.down_proj.biases,<br>language_model.model.layers.34.mlp.experts.down_proj.scales,<br>language_model.model.layers.34.mlp.experts.down_proj.weight,<br>language_model.model.layers.34.mlp.experts.gate_proj.biases,<br>language_model.model.layers.34.mlp.experts.gate_proj.scales,<br>language_model.model.layers.34.mlp.experts.gate_proj.weight,<br>language_model.model.layers.34.mlp.experts.up_proj.biases,<br>language_model.model.layers.34.mlp.experts.up_proj.scales,<br>language_model.model.layers.34.mlp.experts.up_proj.weight,<br>language_model.model.layers.35.mlp.experts.down_proj.biases,<br>language_model.model.layers.35.mlp.experts.down_proj.scales,<br>language_model.model.layers.35.mlp.experts.down_proj.weight,<br>language_model.model.layers.35.mlp.experts.gate_proj.biases,<br>language_model.model.layers.35.mlp.experts.gate_proj.scales,<br>language_model.model.layers.35.mlp.experts.gate_proj.weight,<br>language_model.model.layers.35.mlp.experts.up_proj.biases,<br>language_model.model.layers.35.mlp.experts.up_proj.scales,<br>language_model.model.layers.35.mlp.experts.up_proj.weight,<br>language_model.model.layers.36.mlp.experts.down_proj.biases,<br>language_model.model.layers.36.mlp.experts.down_proj.scales,<br>language_model.model.layers.36.mlp.experts.down_proj.weight,<br>language_model.model.layers.36.mlp.experts.gate_proj.biases,<br>language_model.model.layers.36.mlp.experts.gate_proj.scales,<br>language_model.model.layers.36.mlp.experts.gate_proj.weight,<br>language_model.model.layers.36.mlp.experts.up_proj.biases,<br>language_model.model.layers.36.mlp.experts.up_proj.scales,<br>language_model.model.layers.36.mlp.experts.up_proj.weight,<br>language_model.model.layers.37.mlp.experts.down_proj.biases,<br>language_model.model.layers.37.mlp.experts.down_proj.scales,<br>language_model.model.layers.37.mlp.experts.down_proj.weight,<br>language_model.model.layers.37.mlp.experts.gate_proj.biases,<br>language_model.model.layers.37.mlp.experts.gate_proj.scales,<br>language_model.model.layers.37.mlp.experts.gate_proj.weight,<br>language_model.model.layers.37.mlp.experts.up_proj.biases,<br>language_model.model.layers.37.mlp.experts.up_proj.scales,<br>language_model.model.layers.37.mlp.experts.up_proj.weight,<br>language_model.model.layers.38.mlp.experts.down_proj.biases,<br>language_model.model.layers.38.mlp.experts.down_proj.scales,<br>language_model.model.layers.38.mlp.experts.down_proj.weight,<br>language_model.model.layers.38.mlp.experts.gate_proj.biases,<br>language_model.model.layers.38.mlp.experts.gate_proj.scales,<br>language_model.model.layers.38.mlp.experts.gate_proj.weight,<br>language_model.model.layers.38.mlp.experts.up_proj.biases,<br>language_model.model.layers.38.mlp.experts.up_proj.scales,<br>language_model.model.layers.38.mlp.experts.up_proj.weight,<br>language_model.model.layers.39.mlp.experts.down_proj.biases,<br>language_model.model.layers.39.mlp.experts.down_proj.scales,<br>language_model.model.layers.39.mlp.experts.down_proj.weight,<br>language_model.model.layers.39.mlp.experts.gate_proj.biases,<br>language_model.model.layers.39.mlp.experts.gate_proj.scales,<br>language_model.model.layers.39.mlp.experts.gate_proj.weight,<br>language_model.model.layers.39.mlp.experts.up_proj.biases,<br>language_model.model.layers.39.mlp.experts.up_proj.scales,<br>language_model.model.layers.39.mlp.experts.up_proj.weight,<br>language_model.model.layers.4.mlp.experts.down_proj.biases,<br>language_model.model.layers.4.mlp.experts.down_proj.scales,<br>language_model.model.layers.4.mlp.experts.down_proj.weight,<br>language_model.model.layers.4.mlp.experts.gate_proj.biases,<br>language_model.model.layers.4.mlp.experts.gate_proj.scales,<br>language_model.model.layers.4.mlp.experts.gate_proj.weight,<br>language_model.model.layers.4.mlp.experts.up_proj.biases,<br>language_model.model.layers.4.mlp.experts.up_proj.scales,<br>language_model.model.layers.4.mlp.experts.up_proj.weight,<br>language_model.model.layers.40.mlp.experts.down_proj.biases,<br>language_model.model.layers.40.mlp.experts.down_proj.scales,<br>language_model.model.layers.40.mlp.experts.down_proj.weight,<br>language_model.model.layers.40.mlp.experts.gate_proj.biases,<br>language_model.model.layers.40.mlp.experts.gate_proj.scales,<br>language_model.model.layers.40.mlp.experts.gate_proj.weight,<br>language_model.model.layers.40.mlp.experts.up_proj.biases,<br>language_model.model.layers.40.mlp.experts.up_proj.scales,<br>language_model.model.layers.40.mlp.experts.up_proj.weight,<br>language_model.model.layers.41.mlp.experts.down_proj.biases,<br>language_model.model.layers.41.mlp.experts.down_proj.scales,<br>language_model.model.layers.41.mlp.experts.down_proj.weight,<br>language_model.model.layers.41.mlp.experts.gate_proj.biases,<br>language_model.model.layers.41.mlp.experts.gate_proj.scales,<br>language_model.model.layers.41.mlp.experts.gate_proj.weight,<br>language_model.model.layers.41.mlp.experts.up_proj.biases,<br>language_model.model.layers.41.mlp.experts.up_proj.scales,<br>language_model.model.layers.41.mlp.experts.up_proj.weight,<br>language_model.model.layers.5.mlp.experts.down_proj.biases,<br>language_model.model.layers.5.mlp.experts.down_proj.scales,<br>language_model.model.layers.5.mlp.experts.down_proj.weight,<br>language_model.model.layers.5.mlp.experts.gate_proj.biases,<br>language_model.model.layers.5.mlp.experts.gate_proj.scales,<br>language_model.model.layers.5.mlp.experts.gate_proj.weight,<br>language_model.model.layers.5.mlp.experts.up_proj.biases,<br>language_model.model.layers.5.mlp.experts.up_proj.scales,<br>language_model.model.layers.5.mlp.experts.up_proj.weight,<br>language_model.model.layers.6.mlp.experts.down_proj.biases,<br>language_model.model.layers.6.mlp.experts.down_proj.scales,<br>language_model.model.layers.6.mlp.experts.down_proj.weight,<br>language_model.model.layers.6.mlp.experts.gate_proj.biases,<br>language_model.model.layers.6.mlp.experts.gate_proj.scales,<br>language_model.model.layers.6.mlp.experts.gate_proj.weight,<br>language_model.model.layers.6.mlp.experts.up_proj.biases,<br>language_model.model.layers.6.mlp.experts.up_proj.scales,<br>language_model.model.layers.6.mlp.experts.up_proj.weight,<br>language_model.model.layers.7.mlp.experts.down_proj.biases,<br>language_model.model.layers.7.mlp.experts.down_proj.scales,<br>language_model.model.layers.7.mlp.experts.down_proj.weight,<br>language_model.model.layers.7.mlp.experts.gate_proj.biases,<br>language_model.model.layers.7.mlp.experts.gate_proj.scales,<br>language_model.model.layers.7.mlp.experts.gate_proj.weight,<br>language_model.model.layers.7.mlp.experts.up_proj.biases,<br>language_model.model.layers.7.mlp.experts.up_proj.scales,<br>language_model.model.layers.7.mlp.experts.up_proj.weight,<br>language_model.model.layers.8.mlp.experts.down_proj.biases,<br>language_model.model.layers.8.mlp.experts.down_proj.scales,<br>language_model.model.layers.8.mlp.experts.down_proj.weight,<br>language_model.model.layers.8.mlp.experts.gate_proj.biases,<br>language_model.model.layers.8.mlp.experts.gate_proj.scales,<br>language_model.model.layers.8.mlp.experts.gate_proj.weight,<br>language_model.model.layers.8.mlp.experts.up_proj.biases,<br>language_model.model.layers.8.mlp.experts.up_proj.scales,<br>language_model.model.layers.8.mlp.experts.up_proj.weight,<br>language_model.model.layers.9.mlp.experts.down_proj.biases,<br>language_model.model.layers.9.mlp.experts.down_proj.scales,<br>language_model.model.layers.9.mlp.experts.down_proj.weight,<br>language_model.model.layers.9.mlp.experts.gate_proj.biases,<br>language_model.model.layers.9.mlp.experts.gate_proj.scales,<br>language_model.model.layers.9.mlp.experts.gate_proj.weight,<br>language_model.model.layers.9.mlp.experts.up_proj.biases,<br>language_model.model.layers.9.mlp.experts.up_proj.scales,<br>language_model.model.layers.9.mlp.experts.up_proj.weight.
- *Root exception type:* ValueError
- *Root exception module:* builtins
- *Root exception message:* Received 362 parameters not in model:
  <br>audio_tower.encoder.biases,<br>audio_tower.encoder.scales,<br>language_model.model.layers.10.mlp.experts.down_proj.biases,<br>language_model.model.layers.10.mlp.experts.down_proj.scales,<br>language_model.model.layers.10.mlp.experts.down_proj.weight,<br>language_model.model.layers.10.mlp.experts.gate_proj.biases,<br>language_model.model.layers.10.mlp.experts.gate_proj.scales,<br>language_model.model.layers.10.mlp.experts.gate_proj.weight,<br>language_model.model.layers.10.mlp.experts.up_proj.biases,<br>language_model.model.layers.10.mlp.experts.up_proj.scales,<br>language_model.model.layers.10.mlp.experts.up_proj.weight,<br>language_model.model.layers.11.mlp.experts.down_proj.biases,<br>language_model.model.layers.11.mlp.experts.down_proj.scales,<br>language_model.model.layers.11.mlp.experts.down_proj.weight,<br>language_model.model.layers.11.mlp.experts.gate_proj.biases,<br>language_model.model.layers.11.mlp.experts.gate_proj.scales,<br>language_model.model.layers.11.mlp.experts.gate_proj.weight,<br>language_model.model.layers.11.mlp.experts.up_proj.biases,<br>language_model.model.layers.11.mlp.experts.up_proj.scales,<br>language_model.model.layers.11.mlp.experts.up_proj.weight,<br>language_model.model.layers.12.mlp.experts.down_proj.biases,<br>language_model.model.layers.12.mlp.experts.down_proj.scales,<br>language_model.model.layers.12.mlp.experts.down_proj.weight,<br>language_model.model.layers.12.mlp.experts.gate_proj.biases,<br>language_model.model.layers.12.mlp.experts.gate_proj.scales,<br>language_model.model.layers.12.mlp.experts.gate_proj.weight,<br>language_model.model.layers.12.mlp.experts.up_proj.biases,<br>language_model.model.layers.12.mlp.experts.up_proj.scales,<br>language_model.model.layers.12.mlp.experts.up_proj.weight,<br>language_model.model.layers.13.mlp.experts.down_proj.biases,<br>language_model.model.layers.13.mlp.experts.down_proj.scales,<br>language_model.model.layers.13.mlp.experts.down_proj.weight,<br>language_model.model.layers.13.mlp.experts.gate_proj.biases,<br>language_model.model.layers.13.mlp.experts.gate_proj.scales,<br>language_model.model.layers.13.mlp.experts.gate_proj.weight,<br>language_model.model.layers.13.mlp.experts.up_proj.biases,<br>language_model.model.layers.13.mlp.experts.up_proj.scales,<br>language_model.model.layers.13.mlp.experts.up_proj.weight,<br>language_model.model.layers.14.mlp.experts.down_proj.biases,<br>language_model.model.layers.14.mlp.experts.down_proj.scales,<br>language_model.model.layers.14.mlp.experts.down_proj.weight,<br>language_model.model.layers.14.mlp.experts.gate_proj.biases,<br>language_model.model.layers.14.mlp.experts.gate_proj.scales,<br>language_model.model.layers.14.mlp.experts.gate_proj.weight,<br>language_model.model.layers.14.mlp.experts.up_proj.biases,<br>language_model.model.layers.14.mlp.experts.up_proj.scales,<br>language_model.model.layers.14.mlp.experts.up_proj.weight,<br>language_model.model.layers.15.mlp.experts.down_proj.biases,<br>language_model.model.layers.15.mlp.experts.down_proj.scales,<br>language_model.model.layers.15.mlp.experts.down_proj.weight,<br>language_model.model.layers.15.mlp.experts.gate_proj.biases,<br>language_model.model.layers.15.mlp.experts.gate_proj.scales,<br>language_model.model.layers.15.mlp.experts.gate_proj.weight,<br>language_model.model.layers.15.mlp.experts.up_proj.biases,<br>language_model.model.layers.15.mlp.experts.up_proj.scales,<br>language_model.model.layers.15.mlp.experts.up_proj.weight,<br>language_model.model.layers.16.mlp.experts.down_proj.biases,<br>language_model.model.layers.16.mlp.experts.down_proj.scales,<br>language_model.model.layers.16.mlp.experts.down_proj.weight,<br>language_model.model.layers.16.mlp.experts.gate_proj.biases,<br>language_model.model.layers.16.mlp.experts.gate_proj.scales,<br>language_model.model.layers.16.mlp.experts.gate_proj.weight,<br>language_model.model.layers.16.mlp.experts.up_proj.biases,<br>language_model.model.layers.16.mlp.experts.up_proj.scales,<br>language_model.model.layers.16.mlp.experts.up_proj.weight,<br>language_model.model.layers.17.mlp.experts.down_proj.biases,<br>language_model.model.layers.17.mlp.experts.down_proj.scales,<br>language_model.model.layers.17.mlp.experts.down_proj.weight,<br>language_model.model.layers.17.mlp.experts.gate_proj.biases,<br>language_model.model.layers.17.mlp.experts.gate_proj.scales,<br>language_model.model.layers.17.mlp.experts.gate_proj.weight,<br>language_model.model.layers.17.mlp.experts.up_proj.biases,<br>language_model.model.layers.17.mlp.experts.up_proj.scales,<br>language_model.model.layers.17.mlp.experts.up_proj.weight,<br>language_model.model.layers.18.mlp.experts.down_proj.biases,<br>language_model.model.layers.18.mlp.experts.down_proj.scales,<br>language_model.model.layers.18.mlp.experts.down_proj.weight,<br>language_model.model.layers.18.mlp.experts.gate_proj.biases,<br>language_model.model.layers.18.mlp.experts.gate_proj.scales,<br>language_model.model.layers.18.mlp.experts.gate_proj.weight,<br>language_model.model.layers.18.mlp.experts.up_proj.biases,<br>language_model.model.layers.18.mlp.experts.up_proj.scales,<br>language_model.model.layers.18.mlp.experts.up_proj.weight,<br>language_model.model.layers.19.mlp.experts.down_proj.biases,<br>language_model.model.layers.19.mlp.experts.down_proj.scales,<br>language_model.model.layers.19.mlp.experts.down_proj.weight,<br>language_model.model.layers.19.mlp.experts.gate_proj.biases,<br>language_model.model.layers.19.mlp.experts.gate_proj.scales,<br>language_model.model.layers.19.mlp.experts.gate_proj.weight,<br>language_model.model.layers.19.mlp.experts.up_proj.biases,<br>language_model.model.layers.19.mlp.experts.up_proj.scales,<br>language_model.model.layers.19.mlp.experts.up_proj.weight,<br>language_model.model.layers.2.mlp.experts.down_proj.biases,<br>language_model.model.layers.2.mlp.experts.down_proj.scales,<br>language_model.model.layers.2.mlp.experts.down_proj.weight,<br>language_model.model.layers.2.mlp.experts.gate_proj.biases,<br>language_model.model.layers.2.mlp.experts.gate_proj.scales,<br>language_model.model.layers.2.mlp.experts.gate_proj.weight,<br>language_model.model.layers.2.mlp.experts.up_proj.biases,<br>language_model.model.layers.2.mlp.experts.up_proj.scales,<br>language_model.model.layers.2.mlp.experts.up_proj.weight,<br>language_model.model.layers.20.mlp.experts.down_proj.biases,<br>language_model.model.layers.20.mlp.experts.down_proj.scales,<br>language_model.model.layers.20.mlp.experts.down_proj.weight,<br>language_model.model.layers.20.mlp.experts.gate_proj.biases,<br>language_model.model.layers.20.mlp.experts.gate_proj.scales,<br>language_model.model.layers.20.mlp.experts.gate_proj.weight,<br>language_model.model.layers.20.mlp.experts.up_proj.biases,<br>language_model.model.layers.20.mlp.experts.up_proj.scales,<br>language_model.model.layers.20.mlp.experts.up_proj.weight,<br>language_model.model.layers.21.mlp.experts.down_proj.biases,<br>language_model.model.layers.21.mlp.experts.down_proj.scales,<br>language_model.model.layers.21.mlp.experts.down_proj.weight,<br>language_model.model.layers.21.mlp.experts.gate_proj.biases,<br>language_model.model.layers.21.mlp.experts.gate_proj.scales,<br>language_model.model.layers.21.mlp.experts.gate_proj.weight,<br>language_model.model.layers.21.mlp.experts.up_proj.biases,<br>language_model.model.layers.21.mlp.experts.up_proj.scales,<br>language_model.model.layers.21.mlp.experts.up_proj.weight,<br>language_model.model.layers.22.mlp.experts.down_proj.biases,<br>language_model.model.layers.22.mlp.experts.down_proj.scales,<br>language_model.model.layers.22.mlp.experts.down_proj.weight,<br>language_model.model.layers.22.mlp.experts.gate_proj.biases,<br>language_model.model.layers.22.mlp.experts.gate_proj.scales,<br>language_model.model.layers.22.mlp.experts.gate_proj.weight,<br>language_model.model.layers.22.mlp.experts.up_proj.biases,<br>language_model.model.layers.22.mlp.experts.up_proj.scales,<br>language_model.model.layers.22.mlp.experts.up_proj.weight,<br>language_model.model.layers.23.mlp.experts.down_proj.biases,<br>language_model.model.layers.23.mlp.experts.down_proj.scales,<br>language_model.model.layers.23.mlp.experts.down_proj.weight,<br>language_model.model.layers.23.mlp.experts.gate_proj.biases,<br>language_model.model.layers.23.mlp.experts.gate_proj.scales,<br>language_model.model.layers.23.mlp.experts.gate_proj.weight,<br>language_model.model.layers.23.mlp.experts.up_proj.biases,<br>language_model.model.layers.23.mlp.experts.up_proj.scales,<br>language_model.model.layers.23.mlp.experts.up_proj.weight,<br>language_model.model.layers.24.mlp.experts.down_proj.biases,<br>language_model.model.layers.24.mlp.experts.down_proj.scales,<br>language_model.model.layers.24.mlp.experts.down_proj.weight,<br>language_model.model.layers.24.mlp.experts.gate_proj.biases,<br>language_model.model.layers.24.mlp.experts.gate_proj.scales,<br>language_model.model.layers.24.mlp.experts.gate_proj.weight,<br>language_model.model.layers.24.mlp.experts.up_proj.biases,<br>language_model.model.layers.24.mlp.experts.up_proj.scales,<br>language_model.model.layers.24.mlp.experts.up_proj.weight,<br>language_model.model.layers.25.mlp.experts.down_proj.biases,<br>language_model.model.layers.25.mlp.experts.down_proj.scales,<br>language_model.model.layers.25.mlp.experts.down_proj.weight,<br>language_model.model.layers.25.mlp.experts.gate_proj.biases,<br>language_model.model.layers.25.mlp.experts.gate_proj.scales,<br>language_model.model.layers.25.mlp.experts.gate_proj.weight,<br>language_model.model.layers.25.mlp.experts.up_proj.biases,<br>language_model.model.layers.25.mlp.experts.up_proj.scales,<br>language_model.model.layers.25.mlp.experts.up_proj.weight,<br>language_model.model.layers.26.mlp.experts.down_proj.biases,<br>language_model.model.layers.26.mlp.experts.down_proj.scales,<br>language_model.model.layers.26.mlp.experts.down_proj.weight,<br>language_model.model.layers.26.mlp.experts.gate_proj.biases,<br>language_model.model.layers.26.mlp.experts.gate_proj.scales,<br>language_model.model.layers.26.mlp.experts.gate_proj.weight,<br>language_model.model.layers.26.mlp.experts.up_proj.biases,<br>language_model.model.layers.26.mlp.experts.up_proj.scales,<br>language_model.model.layers.26.mlp.experts.up_proj.weight,<br>language_model.model.layers.27.mlp.experts.down_proj.biases,<br>language_model.model.layers.27.mlp.experts.down_proj.scales,<br>language_model.model.layers.27.mlp.experts.down_proj.weight,<br>language_model.model.layers.27.mlp.experts.gate_proj.biases,<br>language_model.model.layers.27.mlp.experts.gate_proj.scales,<br>language_model.model.layers.27.mlp.experts.gate_proj.weight,<br>language_model.model.layers.27.mlp.experts.up_proj.biases,<br>language_model.model.layers.27.mlp.experts.up_proj.scales,<br>language_model.model.layers.27.mlp.experts.up_proj.weight,<br>language_model.model.layers.28.mlp.experts.down_proj.biases,<br>language_model.model.layers.28.mlp.experts.down_proj.scales,<br>language_model.model.layers.28.mlp.experts.down_proj.weight,<br>language_model.model.layers.28.mlp.experts.gate_proj.biases,<br>language_model.model.layers.28.mlp.experts.gate_proj.scales,<br>language_model.model.layers.28.mlp.experts.gate_proj.weight,<br>language_model.model.layers.28.mlp.experts.up_proj.biases,<br>language_model.model.layers.28.mlp.experts.up_proj.scales,<br>language_model.model.layers.28.mlp.experts.up_proj.weight,<br>language_model.model.layers.29.mlp.experts.down_proj.biases,<br>language_model.model.layers.29.mlp.experts.down_proj.scales,<br>language_model.model.layers.29.mlp.experts.down_proj.weight,<br>language_model.model.layers.29.mlp.experts.gate_proj.biases,<br>language_model.model.layers.29.mlp.experts.gate_proj.scales,<br>language_model.model.layers.29.mlp.experts.gate_proj.weight,<br>language_model.model.layers.29.mlp.experts.up_proj.biases,<br>language_model.model.layers.29.mlp.experts.up_proj.scales,<br>language_model.model.layers.29.mlp.experts.up_proj.weight,<br>language_model.model.layers.3.mlp.experts.down_proj.biases,<br>language_model.model.layers.3.mlp.experts.down_proj.scales,<br>language_model.model.layers.3.mlp.experts.down_proj.weight,<br>language_model.model.layers.3.mlp.experts.gate_proj.biases,<br>language_model.model.layers.3.mlp.experts.gate_proj.scales,<br>language_model.model.layers.3.mlp.experts.gate_proj.weight,<br>language_model.model.layers.3.mlp.experts.up_proj.biases,<br>language_model.model.layers.3.mlp.experts.up_proj.scales,<br>language_model.model.layers.3.mlp.experts.up_proj.weight,<br>language_model.model.layers.30.mlp.experts.down_proj.biases,<br>language_model.model.layers.30.mlp.experts.down_proj.scales,<br>language_model.model.layers.30.mlp.experts.down_proj.weight,<br>language_model.model.layers.30.mlp.experts.gate_proj.biases,<br>language_model.model.layers.30.mlp.experts.gate_proj.scales,<br>language_model.model.layers.30.mlp.experts.gate_proj.weight,<br>language_model.model.layers.30.mlp.experts.up_proj.biases,<br>language_model.model.layers.30.mlp.experts.up_proj.scales,<br>language_model.model.layers.30.mlp.experts.up_proj.weight,<br>language_model.model.layers.31.mlp.experts.down_proj.biases,<br>language_model.model.layers.31.mlp.experts.down_proj.scales,<br>language_model.model.layers.31.mlp.experts.down_proj.weight,<br>language_model.model.layers.31.mlp.experts.gate_proj.biases,<br>language_model.model.layers.31.mlp.experts.gate_proj.scales,<br>language_model.model.layers.31.mlp.experts.gate_proj.weight,<br>language_model.model.layers.31.mlp.experts.up_proj.biases,<br>language_model.model.layers.31.mlp.experts.up_proj.scales,<br>language_model.model.layers.31.mlp.experts.up_proj.weight,<br>language_model.model.layers.32.mlp.experts.down_proj.biases,<br>language_model.model.layers.32.mlp.experts.down_proj.scales,<br>language_model.model.layers.32.mlp.experts.down_proj.weight,<br>language_model.model.layers.32.mlp.experts.gate_proj.biases,<br>language_model.model.layers.32.mlp.experts.gate_proj.scales,<br>language_model.model.layers.32.mlp.experts.gate_proj.weight,<br>language_model.model.layers.32.mlp.experts.up_proj.biases,<br>language_model.model.layers.32.mlp.experts.up_proj.scales,<br>language_model.model.layers.32.mlp.experts.up_proj.weight,<br>language_model.model.layers.33.mlp.experts.down_proj.biases,<br>language_model.model.layers.33.mlp.experts.down_proj.scales,<br>language_model.model.layers.33.mlp.experts.down_proj.weight,<br>language_model.model.layers.33.mlp.experts.gate_proj.biases,<br>language_model.model.layers.33.mlp.experts.gate_proj.scales,<br>language_model.model.layers.33.mlp.experts.gate_proj.weight,<br>language_model.model.layers.33.mlp.experts.up_proj.biases,<br>language_model.model.layers.33.mlp.experts.up_proj.scales,<br>language_model.model.layers.33.mlp.experts.up_proj.weight,<br>language_model.model.layers.34.mlp.experts.down_proj.biases,<br>language_model.model.layers.34.mlp.experts.down_proj.scales,<br>language_model.model.layers.34.mlp.experts.down_proj.weight,<br>language_model.model.layers.34.mlp.experts.gate_proj.biases,<br>language_model.model.layers.34.mlp.experts.gate_proj.scales,<br>language_model.model.layers.34.mlp.experts.gate_proj.weight,<br>language_model.model.layers.34.mlp.experts.up_proj.biases,<br>language_model.model.layers.34.mlp.experts.up_proj.scales,<br>language_model.model.layers.34.mlp.experts.up_proj.weight,<br>language_model.model.layers.35.mlp.experts.down_proj.biases,<br>language_model.model.layers.35.mlp.experts.down_proj.scales,<br>language_model.model.layers.35.mlp.experts.down_proj.weight,<br>language_model.model.layers.35.mlp.experts.gate_proj.biases,<br>language_model.model.layers.35.mlp.experts.gate_proj.scales,<br>language_model.model.layers.35.mlp.experts.gate_proj.weight,<br>language_model.model.layers.35.mlp.experts.up_proj.biases,<br>language_model.model.layers.35.mlp.experts.up_proj.scales,<br>language_model.model.layers.35.mlp.experts.up_proj.weight,<br>language_model.model.layers.36.mlp.experts.down_proj.biases,<br>language_model.model.layers.36.mlp.experts.down_proj.scales,<br>language_model.model.layers.36.mlp.experts.down_proj.weight,<br>language_model.model.layers.36.mlp.experts.gate_proj.biases,<br>language_model.model.layers.36.mlp.experts.gate_proj.scales,<br>language_model.model.layers.36.mlp.experts.gate_proj.weight,<br>language_model.model.layers.36.mlp.experts.up_proj.biases,<br>language_model.model.layers.36.mlp.experts.up_proj.scales,<br>language_model.model.layers.36.mlp.experts.up_proj.weight,<br>language_model.model.layers.37.mlp.experts.down_proj.biases,<br>language_model.model.layers.37.mlp.experts.down_proj.scales,<br>language_model.model.layers.37.mlp.experts.down_proj.weight,<br>language_model.model.layers.37.mlp.experts.gate_proj.biases,<br>language_model.model.layers.37.mlp.experts.gate_proj.scales,<br>language_model.model.layers.37.mlp.experts.gate_proj.weight,<br>language_model.model.layers.37.mlp.experts.up_proj.biases,<br>language_model.model.layers.37.mlp.experts.up_proj.scales,<br>language_model.model.layers.37.mlp.experts.up_proj.weight,<br>language_model.model.layers.38.mlp.experts.down_proj.biases,<br>language_model.model.layers.38.mlp.experts.down_proj.scales,<br>language_model.model.layers.38.mlp.experts.down_proj.weight,<br>language_model.model.layers.38.mlp.experts.gate_proj.biases,<br>language_model.model.layers.38.mlp.experts.gate_proj.scales,<br>language_model.model.layers.38.mlp.experts.gate_proj.weight,<br>language_model.model.layers.38.mlp.experts.up_proj.biases,<br>language_model.model.layers.38.mlp.experts.up_proj.scales,<br>language_model.model.layers.38.mlp.experts.up_proj.weight,<br>language_model.model.layers.39.mlp.experts.down_proj.biases,<br>language_model.model.layers.39.mlp.experts.down_proj.scales,<br>language_model.model.layers.39.mlp.experts.down_proj.weight,<br>language_model.model.layers.39.mlp.experts.gate_proj.biases,<br>language_model.model.layers.39.mlp.experts.gate_proj.scales,<br>language_model.model.layers.39.mlp.experts.gate_proj.weight,<br>language_model.model.layers.39.mlp.experts.up_proj.biases,<br>language_model.model.layers.39.mlp.experts.up_proj.scales,<br>language_model.model.layers.39.mlp.experts.up_proj.weight,<br>language_model.model.layers.4.mlp.experts.down_proj.biases,<br>language_model.model.layers.4.mlp.experts.down_proj.scales,<br>language_model.model.layers.4.mlp.experts.down_proj.weight,<br>language_model.model.layers.4.mlp.experts.gate_proj.biases,<br>language_model.model.layers.4.mlp.experts.gate_proj.scales,<br>language_model.model.layers.4.mlp.experts.gate_proj.weight,<br>language_model.model.layers.4.mlp.experts.up_proj.biases,<br>language_model.model.layers.4.mlp.experts.up_proj.scales,<br>language_model.model.layers.4.mlp.experts.up_proj.weight,<br>language_model.model.layers.40.mlp.experts.down_proj.biases,<br>language_model.model.layers.40.mlp.experts.down_proj.scales,<br>language_model.model.layers.40.mlp.experts.down_proj.weight,<br>language_model.model.layers.40.mlp.experts.gate_proj.biases,<br>language_model.model.layers.40.mlp.experts.gate_proj.scales,<br>language_model.model.layers.40.mlp.experts.gate_proj.weight,<br>language_model.model.layers.40.mlp.experts.up_proj.biases,<br>language_model.model.layers.40.mlp.experts.up_proj.scales,<br>language_model.model.layers.40.mlp.experts.up_proj.weight,<br>language_model.model.layers.41.mlp.experts.down_proj.biases,<br>language_model.model.layers.41.mlp.experts.down_proj.scales,<br>language_model.model.layers.41.mlp.experts.down_proj.weight,<br>language_model.model.layers.41.mlp.experts.gate_proj.biases,<br>language_model.model.layers.41.mlp.experts.gate_proj.scales,<br>language_model.model.layers.41.mlp.experts.gate_proj.weight,<br>language_model.model.layers.41.mlp.experts.up_proj.biases,<br>language_model.model.layers.41.mlp.experts.up_proj.scales,<br>language_model.model.layers.41.mlp.experts.up_proj.weight,<br>language_model.model.layers.5.mlp.experts.down_proj.biases,<br>language_model.model.layers.5.mlp.experts.down_proj.scales,<br>language_model.model.layers.5.mlp.experts.down_proj.weight,<br>language_model.model.layers.5.mlp.experts.gate_proj.biases,<br>language_model.model.layers.5.mlp.experts.gate_proj.scales,<br>language_model.model.layers.5.mlp.experts.gate_proj.weight,<br>language_model.model.layers.5.mlp.experts.up_proj.biases,<br>language_model.model.layers.5.mlp.experts.up_proj.scales,<br>language_model.model.layers.5.mlp.experts.up_proj.weight,<br>language_model.model.layers.6.mlp.experts.down_proj.biases,<br>language_model.model.layers.6.mlp.experts.down_proj.scales,<br>language_model.model.layers.6.mlp.experts.down_proj.weight,<br>language_model.model.layers.6.mlp.experts.gate_proj.biases,<br>language_model.model.layers.6.mlp.experts.gate_proj.scales,<br>language_model.model.layers.6.mlp.experts.gate_proj.weight,<br>language_model.model.layers.6.mlp.experts.up_proj.biases,<br>language_model.model.layers.6.mlp.experts.up_proj.scales,<br>language_model.model.layers.6.mlp.experts.up_proj.weight,<br>language_model.model.layers.7.mlp.experts.down_proj.biases,<br>language_model.model.layers.7.mlp.experts.down_proj.scales,<br>language_model.model.layers.7.mlp.experts.down_proj.weight,<br>language_model.model.layers.7.mlp.experts.gate_proj.biases,<br>language_model.model.layers.7.mlp.experts.gate_proj.scales,<br>language_model.model.layers.7.mlp.experts.gate_proj.weight,<br>language_model.model.layers.7.mlp.experts.up_proj.biases,<br>language_model.model.layers.7.mlp.experts.up_proj.scales,<br>language_model.model.layers.7.mlp.experts.up_proj.weight,<br>language_model.model.layers.8.mlp.experts.down_proj.biases,<br>language_model.model.layers.8.mlp.experts.down_proj.scales,<br>language_model.model.layers.8.mlp.experts.down_proj.weight,<br>language_model.model.layers.8.mlp.experts.gate_proj.biases,<br>language_model.model.layers.8.mlp.experts.gate_proj.scales,<br>language_model.model.layers.8.mlp.experts.gate_proj.weight,<br>language_model.model.layers.8.mlp.experts.up_proj.biases,<br>language_model.model.layers.8.mlp.experts.up_proj.scales,<br>language_model.model.layers.8.mlp.experts.up_proj.weight,<br>language_model.model.layers.9.mlp.experts.down_proj.biases,<br>language_model.model.layers.9.mlp.experts.down_proj.scales,<br>language_model.model.layers.9.mlp.experts.down_proj.weight,<br>language_model.model.layers.9.mlp.experts.gate_proj.biases,<br>language_model.model.layers.9.mlp.experts.gate_proj.scales,<br>language_model.model.layers.9.mlp.experts.gate_proj.weight,<br>language_model.model.layers.9.mlp.experts.up_proj.biases,<br>language_model.model.layers.9.mlp.experts.up_proj.scales,<br>language_model.model.layers.9.mlp.experts.up_proj.weight.
- *Arch supported by installed mlx-vlm:* yes (model_type inkling_mm_model via
  inkling)
- *Model load time:* 0.20s
- *Generation time:* -
- *Total time:* 0.20s
- *Input validation time:* 5.81e-05
- *Prompt preparation time:* -
- *First-token latency:* -
- *Cleanup time:* 0.0762
- *Prompt tokens:* -
- *Generation tokens:* -
- *Total tokens:* -
- *Prompt throughput (raw):* -
- *Generation throughput (raw):* -
- *Peak memory:* -
- *Active memory:* -
- *Cache memory:* -
- *Model-load active memory:* -
- *Post-cleanup active memory:* 0.0027
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* exception
- *Requested maximum tokens:* 500
- *Rendered prompt characters:* not captured
- *Image placeholders:* not captured
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* not captured
- *Tokenizer:* not captured
- *Requested model revision:* not requested
- *Resolved model revision:* f0cafad5b1a3e54be06ba03fe07b4cd4e8bcc612
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Inkling-Small-mlx-4bit/snapshots/f0cafad5b1a3e54be06ba03fe07b4cd4e8bcc612
- *Generation settings:* not captured
- *EOS token:* not captured

#### Complete traceback

```python
Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11997, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11474, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 756, in _typed_mlx_vlm_load
    loaded: tuple[nn.Module, ProcessorMixin] = _mlx_vlm_load(
                                               ~~~~~~~~~~~~~^
        path_or_hf_repo=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 964, in load
    model = load_model(model_path, lazy, strict=strict, **kwargs)
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 840, in load_model
    model.load_weights(list(weights.items()), strict=strict)
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "~/Documents/AI/mlx/mlx/python/mlx/nn/layers/base.py", line 185, in load_weights
    raise ValueError(
        f"Received {num_extra} parameters not in model: \n{extras}."
    )
ValueError: Received 362 parameters not in model: 
audio_tower.encoder.biases,
audio_tower.encoder.scales,
language_model.model.layers.10.mlp.experts.down_proj.biases,
language_model.model.layers.10.mlp.experts.down_proj.scales,
language_model.model.layers.10.mlp.experts.down_proj.weight,
language_model.model.layers.10.mlp.experts.gate_proj.biases,
language_model.model.layers.10.mlp.experts.gate_proj.scales,
language_model.model.layers.10.mlp.experts.gate_proj.weight,
language_model.model.layers.10.mlp.experts.up_proj.biases,
language_model.model.layers.10.mlp.experts.up_proj.scales,
language_model.model.layers.10.mlp.experts.up_proj.weight,
language_model.model.layers.11.mlp.experts.down_proj.biases,
language_model.model.layers.11.mlp.experts.down_proj.scales,
language_model.model.layers.11.mlp.experts.down_proj.weight,
language_model.model.layers.11.mlp.experts.gate_proj.biases,
language_model.model.layers.11.mlp.experts.gate_proj.scales,
language_model.model.layers.11.mlp.experts.gate_proj.weight,
language_model.model.layers.11.mlp.experts.up_proj.biases,
language_model.model.layers.11.mlp.experts.up_proj.scales,
language_model.model.layers.11.mlp.experts.up_proj.weight,
language_model.model.layers.12.mlp.experts.down_proj.biases,
language_model.model.layers.12.mlp.experts.down_proj.scales,
language_model.model.layers.12.mlp.experts.down_proj.weight,
language_model.model.layers.12.mlp.experts.gate_proj.biases,
language_model.model.layers.12.mlp.experts.gate_proj.scales,
language_model.model.layers.12.mlp.experts.gate_proj.weight,
language_model.model.layers.12.mlp.experts.up_proj.biases,
language_model.model.layers.12.mlp.experts.up_proj.scales,
language_model.model.layers.12.mlp.experts.up_proj.weight,
language_model.model.layers.13.mlp.experts.down_proj.biases,
language_model.model.layers.13.mlp.experts.down_proj.scales,
language_model.model.layers.13.mlp.experts.down_proj.weight,
language_model.model.layers.13.mlp.experts.gate_proj.biases,
language_model.model.layers.13.mlp.experts.gate_proj.scales,
language_model.model.layers.13.mlp.experts.gate_proj.weight,
language_model.model.layers.13.mlp.experts.up_proj.biases,
language_model.model.layers.13.mlp.experts.up_proj.scales,
language_model.model.layers.13.mlp.experts.up_proj.weight,
language_model.model.layers.14.mlp.experts.down_proj.biases,
language_model.model.layers.14.mlp.experts.down_proj.scales,
language_model.model.layers.14.mlp.experts.down_proj.weight,
language_model.model.layers.14.mlp.experts.gate_proj.biases,
language_model.model.layers.14.mlp.experts.gate_proj.scales,
language_model.model.layers.14.mlp.experts.gate_proj.weight,
language_model.model.layers.14.mlp.experts.up_proj.biases,
language_model.model.layers.14.mlp.experts.up_proj.scales,
language_model.model.layers.14.mlp.experts.up_proj.weight,
language_model.model.layers.15.mlp.experts.down_proj.biases,
language_model.model.layers.15.mlp.experts.down_proj.scales,
language_model.model.layers.15.mlp.experts.down_proj.weight,
language_model.model.layers.15.mlp.experts.gate_proj.biases,
language_model.model.layers.15.mlp.experts.gate_proj.scales,
language_model.model.layers.15.mlp.experts.gate_proj.weight,
language_model.model.layers.15.mlp.experts.up_proj.biases,
language_model.model.layers.15.mlp.experts.up_proj.scales,
language_model.model.layers.15.mlp.experts.up_proj.weight,
language_model.model.layers.16.mlp.experts.down_proj.biases,
language_model.model.layers.16.mlp.experts.down_proj.scales,
language_model.model.layers.16.mlp.experts.down_proj.weight,
language_model.model.layers.16.mlp.experts.gate_proj.biases,
language_model.model.layers.16.mlp.experts.gate_proj.scales,
language_model.model.layers.16.mlp.experts.gate_proj.weight,
language_model.model.layers.16.mlp.experts.up_proj.biases,
language_model.model.layers.16.mlp.experts.up_proj.scales,
language_model.model.layers.16.mlp.experts.up_proj.weight,
language_model.model.layers.17.mlp.experts.down_proj.biases,
language_model.model.layers.17.mlp.experts.down_proj.scales,
language_model.model.layers.17.mlp.experts.down_proj.weight,
language_model.model.layers.17.mlp.experts.gate_proj.biases,
language_model.model.layers.17.mlp.experts.gate_proj.scales,
language_model.model.layers.17.mlp.experts.gate_proj.weight,
language_model.model.layers.17.mlp.experts.up_proj.biases,
language_model.model.layers.17.mlp.experts.up_proj.scales,
language_model.model.layers.17.mlp.experts.up_proj.weight,
language_model.model.layers.18.mlp.experts.down_proj.biases,
language_model.model.layers.18.mlp.experts.down_proj.scales,
language_model.model.layers.18.mlp.experts.down_proj.weight,
language_model.model.layers.18.mlp.experts.gate_proj.biases,
language_model.model.layers.18.mlp.experts.gate_proj.scales,
language_model.model.layers.18.mlp.experts.gate_proj.weight,
language_model.model.layers.18.mlp.experts.up_proj.biases,
language_model.model.layers.18.mlp.experts.up_proj.scales,
language_model.model.layers.18.mlp.experts.up_proj.weight,
language_model.model.layers.19.mlp.experts.down_proj.biases,
language_model.model.layers.19.mlp.experts.down_proj.scales,
language_model.model.layers.19.mlp.experts.down_proj.weight,
language_model.model.layers.19.mlp.experts.gate_proj.biases,
language_model.model.layers.19.mlp.experts.gate_proj.scales,
language_model.model.layers.19.mlp.experts.gate_proj.weight,
language_model.model.layers.19.mlp.experts.up_proj.biases,
language_model.model.layers.19.mlp.experts.up_proj.scales,
language_model.model.layers.19.mlp.experts.up_proj.weight,
language_model.model.layers.2.mlp.experts.down_proj.biases,
language_model.model.layers.2.mlp.experts.down_proj.scales,
language_model.model.layers.2.mlp.experts.down_proj.weight,
language_model.model.layers.2.mlp.experts.gate_proj.biases,
language_model.model.layers.2.mlp.experts.gate_proj.scales,
language_model.model.layers.2.mlp.experts.gate_proj.weight,
language_model.model.layers.2.mlp.experts.up_proj.biases,
language_model.model.layers.2.mlp.experts.up_proj.scales,
language_model.model.layers.2.mlp.experts.up_proj.weight,
language_model.model.layers.20.mlp.experts.down_proj.biases,
language_model.model.layers.20.mlp.experts.down_proj.scales,
language_model.model.layers.20.mlp.experts.down_proj.weight,
language_model.model.layers.20.mlp.experts.gate_proj.biases,
language_model.model.layers.20.mlp.experts.gate_proj.scales,
language_model.model.layers.20.mlp.experts.gate_proj.weight,
language_model.model.layers.20.mlp.experts.up_proj.biases,
language_model.model.layers.20.mlp.experts.up_proj.scales,
language_model.model.layers.20.mlp.experts.up_proj.weight,
language_model.model.layers.21.mlp.experts.down_proj.biases,
language_model.model.layers.21.mlp.experts.down_proj.scales,
language_model.model.layers.21.mlp.experts.down_proj.weight,
language_model.model.layers.21.mlp.experts.gate_proj.biases,
language_model.model.layers.21.mlp.experts.gate_proj.scales,
language_model.model.layers.21.mlp.experts.gate_proj.weight,
language_model.model.layers.21.mlp.experts.up_proj.biases,
language_model.model.layers.21.mlp.experts.up_proj.scales,
language_model.model.layers.21.mlp.experts.up_proj.weight,
language_model.model.layers.22.mlp.experts.down_proj.biases,
language_model.model.layers.22.mlp.experts.down_proj.scales,
language_model.model.layers.22.mlp.experts.down_proj.weight,
language_model.model.layers.22.mlp.experts.gate_proj.biases,
language_model.model.layers.22.mlp.experts.gate_proj.scales,
language_model.model.layers.22.mlp.experts.gate_proj.weight,
language_model.model.layers.22.mlp.experts.up_proj.biases,
language_model.model.layers.22.mlp.experts.up_proj.scales,
language_model.model.layers.22.mlp.experts.up_proj.weight,
language_model.model.layers.23.mlp.experts.down_proj.biases,
language_model.model.layers.23.mlp.experts.down_proj.scales,
language_model.model.layers.23.mlp.experts.down_proj.weight,
language_model.model.layers.23.mlp.experts.gate_proj.biases,
language_model.model.layers.23.mlp.experts.gate_proj.scales,
language_model.model.layers.23.mlp.experts.gate_proj.weight,
language_model.model.layers.23.mlp.experts.up_proj.biases,
language_model.model.layers.23.mlp.experts.up_proj.scales,
language_model.model.layers.23.mlp.experts.up_proj.weight,
language_model.model.layers.24.mlp.experts.down_proj.biases,
language_model.model.layers.24.mlp.experts.down_proj.scales,
language_model.model.layers.24.mlp.experts.down_proj.weight,
language_model.model.layers.24.mlp.experts.gate_proj.biases,
language_model.model.layers.24.mlp.experts.gate_proj.scales,
language_model.model.layers.24.mlp.experts.gate_proj.weight,
language_model.model.layers.24.mlp.experts.up_proj.biases,
language_model.model.layers.24.mlp.experts.up_proj.scales,
language_model.model.layers.24.mlp.experts.up_proj.weight,
language_model.model.layers.25.mlp.experts.down_proj.biases,
language_model.model.layers.25.mlp.experts.down_proj.scales,
language_model.model.layers.25.mlp.experts.down_proj.weight,
language_model.model.layers.25.mlp.experts.gate_proj.biases,
language_model.model.layers.25.mlp.experts.gate_proj.scales,
language_model.model.layers.25.mlp.experts.gate_proj.weight,
language_model.model.layers.25.mlp.experts.up_proj.biases,
language_model.model.layers.25.mlp.experts.up_proj.scales,
language_model.model.layers.25.mlp.experts.up_proj.weight,
language_model.model.layers.26.mlp.experts.down_proj.biases,
language_model.model.layers.26.mlp.experts.down_proj.scales,
language_model.model.layers.26.mlp.experts.down_proj.weight,
language_model.model.layers.26.mlp.experts.gate_proj.biases,
language_model.model.layers.26.mlp.experts.gate_proj.scales,
language_model.model.layers.26.mlp.experts.gate_proj.weight,
language_model.model.layers.26.mlp.experts.up_proj.biases,
language_model.model.layers.26.mlp.experts.up_proj.scales,
language_model.model.layers.26.mlp.experts.up_proj.weight,
language_model.model.layers.27.mlp.experts.down_proj.biases,
language_model.model.layers.27.mlp.experts.down_proj.scales,
language_model.model.layers.27.mlp.experts.down_proj.weight,
language_model.model.layers.27.mlp.experts.gate_proj.biases,
language_model.model.layers.27.mlp.experts.gate_proj.scales,
language_model.model.layers.27.mlp.experts.gate_proj.weight,
language_model.model.layers.27.mlp.experts.up_proj.biases,
language_model.model.layers.27.mlp.experts.up_proj.scales,
language_model.model.layers.27.mlp.experts.up_proj.weight,
language_model.model.layers.28.mlp.experts.down_proj.biases,
language_model.model.layers.28.mlp.experts.down_proj.scales,
language_model.model.layers.28.mlp.experts.down_proj.weight,
language_model.model.layers.28.mlp.experts.gate_proj.biases,
language_model.model.layers.28.mlp.experts.gate_proj.scales,
language_model.model.layers.28.mlp.experts.gate_proj.weight,
language_model.model.layers.28.mlp.experts.up_proj.biases,
language_model.model.layers.28.mlp.experts.up_proj.scales,
language_model.model.layers.28.mlp.experts.up_proj.weight,
language_model.model.layers.29.mlp.experts.down_proj.biases,
language_model.model.layers.29.mlp.experts.down_proj.scales,
language_model.model.layers.29.mlp.experts.down_proj.weight,
language_model.model.layers.29.mlp.experts.gate_proj.biases,
language_model.model.layers.29.mlp.experts.gate_proj.scales,
language_model.model.layers.29.mlp.experts.gate_proj.weight,
language_model.model.layers.29.mlp.experts.up_proj.biases,
language_model.model.layers.29.mlp.experts.up_proj.scales,
language_model.model.layers.29.mlp.experts.up_proj.weight,
language_model.model.layers.3.mlp.experts.down_proj.biases,
language_model.model.layers.3.mlp.experts.down_proj.scales,
language_model.model.layers.3.mlp.experts.down_proj.weight,
language_model.model.layers.3.mlp.experts.gate_proj.biases,
language_model.model.layers.3.mlp.experts.gate_proj.scales,
language_model.model.layers.3.mlp.experts.gate_proj.weight,
language_model.model.layers.3.mlp.experts.up_proj.biases,
language_model.model.layers.3.mlp.experts.up_proj.scales,
language_model.model.layers.3.mlp.experts.up_proj.weight,
language_model.model.layers.30.mlp.experts.down_proj.biases,
language_model.model.layers.30.mlp.experts.down_proj.scales,
language_model.model.layers.30.mlp.experts.down_proj.weight,
language_model.model.layers.30.mlp.experts.gate_proj.biases,
language_model.model.layers.30.mlp.experts.gate_proj.scales,
language_model.model.layers.30.mlp.experts.gate_proj.weight,
language_model.model.layers.30.mlp.experts.up_proj.biases,
language_model.model.layers.30.mlp.experts.up_proj.scales,
language_model.model.layers.30.mlp.experts.up_proj.weight,
language_model.model.layers.31.mlp.experts.down_proj.biases,
language_model.model.layers.31.mlp.experts.down_proj.scales,
language_model.model.layers.31.mlp.experts.down_proj.weight,
language_model.model.layers.31.mlp.experts.gate_proj.biases,
language_model.model.layers.31.mlp.experts.gate_proj.scales,
language_model.model.layers.31.mlp.experts.gate_proj.weight,
language_model.model.layers.31.mlp.experts.up_proj.biases,
language_model.model.layers.31.mlp.experts.up_proj.scales,
language_model.model.layers.31.mlp.experts.up_proj.weight,
language_model.model.layers.32.mlp.experts.down_proj.biases,
language_model.model.layers.32.mlp.experts.down_proj.scales,
language_model.model.layers.32.mlp.experts.down_proj.weight,
language_model.model.layers.32.mlp.experts.gate_proj.biases,
language_model.model.layers.32.mlp.experts.gate_proj.scales,
language_model.model.layers.32.mlp.experts.gate_proj.weight,
language_model.model.layers.32.mlp.experts.up_proj.biases,
language_model.model.layers.32.mlp.experts.up_proj.scales,
language_model.model.layers.32.mlp.experts.up_proj.weight,
language_model.model.layers.33.mlp.experts.down_proj.biases,
language_model.model.layers.33.mlp.experts.down_proj.scales,
language_model.model.layers.33.mlp.experts.down_proj.weight,
language_model.model.layers.33.mlp.experts.gate_proj.biases,
language_model.model.layers.33.mlp.experts.gate_proj.scales,
language_model.model.layers.33.mlp.experts.gate_proj.weight,
language_model.model.layers.33.mlp.experts.up_proj.biases,
language_model.model.layers.33.mlp.experts.up_proj.scales,
language_model.model.layers.33.mlp.experts.up_proj.weight,
language_model.model.layers.34.mlp.experts.down_proj.biases,
language_model.model.layers.34.mlp.experts.down_proj.scales,
language_model.model.layers.34.mlp.experts.down_proj.weight,
language_model.model.layers.34.mlp.experts.gate_proj.biases,
language_model.model.layers.34.mlp.experts.gate_proj.scales,
language_model.model.layers.34.mlp.experts.gate_proj.weight,
language_model.model.layers.34.mlp.experts.up_proj.biases,
language_model.model.layers.34.mlp.experts.up_proj.scales,
language_model.model.layers.34.mlp.experts.up_proj.weight,
language_model.model.layers.35.mlp.experts.down_proj.biases,
language_model.model.layers.35.mlp.experts.down_proj.scales,
language_model.model.layers.35.mlp.experts.down_proj.weight,
language_model.model.layers.35.mlp.experts.gate_proj.biases,
language_model.model.layers.35.mlp.experts.gate_proj.scales,
language_model.model.layers.35.mlp.experts.gate_proj.weight,
language_model.model.layers.35.mlp.experts.up_proj.biases,
language_model.model.layers.35.mlp.experts.up_proj.scales,
language_model.model.layers.35.mlp.experts.up_proj.weight,
language_model.model.layers.36.mlp.experts.down_proj.biases,
language_model.model.layers.36.mlp.experts.down_proj.scales,
language_model.model.layers.36.mlp.experts.down_proj.weight,
language_model.model.layers.36.mlp.experts.gate_proj.biases,
language_model.model.layers.36.mlp.experts.gate_proj.scales,
language_model.model.layers.36.mlp.experts.gate_proj.weight,
language_model.model.layers.36.mlp.experts.up_proj.biases,
language_model.model.layers.36.mlp.experts.up_proj.scales,
language_model.model.layers.36.mlp.experts.up_proj.weight,
language_model.model.layers.37.mlp.experts.down_proj.biases,
language_model.model.layers.37.mlp.experts.down_proj.scales,
language_model.model.layers.37.mlp.experts.down_proj.weight,
language_model.model.layers.37.mlp.experts.gate_proj.biases,
language_model.model.layers.37.mlp.experts.gate_proj.scales,
language_model.model.layers.37.mlp.experts.gate_proj.weight,
language_model.model.layers.37.mlp.experts.up_proj.biases,
language_model.model.layers.37.mlp.experts.up_proj.scales,
language_model.model.layers.37.mlp.experts.up_proj.weight,
language_model.model.layers.38.mlp.experts.down_proj.biases,
language_model.model.layers.38.mlp.experts.down_proj.scales,
language_model.model.layers.38.mlp.experts.down_proj.weight,
language_model.model.layers.38.mlp.experts.gate_proj.biases,
language_model.model.layers.38.mlp.experts.gate_proj.scales,
language_model.model.layers.38.mlp.experts.gate_proj.weight,
language_model.model.layers.38.mlp.experts.up_proj.biases,
language_model.model.layers.38.mlp.experts.up_proj.scales,
language_model.model.layers.38.mlp.experts.up_proj.weight,
language_model.model.layers.39.mlp.experts.down_proj.biases,
language_model.model.layers.39.mlp.experts.down_proj.scales,
language_model.model.layers.39.mlp.experts.down_proj.weight,
language_model.model.layers.39.mlp.experts.gate_proj.biases,
language_model.model.layers.39.mlp.experts.gate_proj.scales,
language_model.model.layers.39.mlp.experts.gate_proj.weight,
language_model.model.layers.39.mlp.experts.up_proj.biases,
language_model.model.layers.39.mlp.experts.up_proj.scales,
language_model.model.layers.39.mlp.experts.up_proj.weight,
language_model.model.layers.4.mlp.experts.down_proj.biases,
language_model.model.layers.4.mlp.experts.down_proj.scales,
language_model.model.layers.4.mlp.experts.down_proj.weight,
language_model.model.layers.4.mlp.experts.gate_proj.biases,
language_model.model.layers.4.mlp.experts.gate_proj.scales,
language_model.model.layers.4.mlp.experts.gate_proj.weight,
language_model.model.layers.4.mlp.experts.up_proj.biases,
language_model.model.layers.4.mlp.experts.up_proj.scales,
language_model.model.layers.4.mlp.experts.up_proj.weight,
language_model.model.layers.40.mlp.experts.down_proj.biases,
language_model.model.layers.40.mlp.experts.down_proj.scales,
language_model.model.layers.40.mlp.experts.down_proj.weight,
language_model.model.layers.40.mlp.experts.gate_proj.biases,
language_model.model.layers.40.mlp.experts.gate_proj.scales,
language_model.model.layers.40.mlp.experts.gate_proj.weight,
language_model.model.layers.40.mlp.experts.up_proj.biases,
language_model.model.layers.40.mlp.experts.up_proj.scales,
language_model.model.layers.40.mlp.experts.up_proj.weight,
language_model.model.layers.41.mlp.experts.down_proj.biases,
language_model.model.layers.41.mlp.experts.down_proj.scales,
language_model.model.layers.41.mlp.experts.down_proj.weight,
language_model.model.layers.41.mlp.experts.gate_proj.biases,
language_model.model.layers.41.mlp.experts.gate_proj.scales,
language_model.model.layers.41.mlp.experts.gate_proj.weight,
language_model.model.layers.41.mlp.experts.up_proj.biases,
language_model.model.layers.41.mlp.experts.up_proj.scales,
language_model.model.layers.41.mlp.experts.up_proj.weight,
language_model.model.layers.5.mlp.experts.down_proj.biases,
language_model.model.layers.5.mlp.experts.down_proj.scales,
language_model.model.layers.5.mlp.experts.down_proj.weight,
language_model.model.layers.5.mlp.experts.gate_proj.biases,
language_model.model.layers.5.mlp.experts.gate_proj.scales,
language_model.model.layers.5.mlp.experts.gate_proj.weight,
language_model.model.layers.5.mlp.experts.up_proj.biases,
language_model.model.layers.5.mlp.experts.up_proj.scales,
language_model.model.layers.5.mlp.experts.up_proj.weight,
language_model.model.layers.6.mlp.experts.down_proj.biases,
language_model.model.layers.6.mlp.experts.down_proj.scales,
language_model.model.layers.6.mlp.experts.down_proj.weight,
language_model.model.layers.6.mlp.experts.gate_proj.biases,
language_model.model.layers.6.mlp.experts.gate_proj.scales,
language_model.model.layers.6.mlp.experts.gate_proj.weight,
language_model.model.layers.6.mlp.experts.up_proj.biases,
language_model.model.layers.6.mlp.experts.up_proj.scales,
language_model.model.layers.6.mlp.experts.up_proj.weight,
language_model.model.layers.7.mlp.experts.down_proj.biases,
language_model.model.layers.7.mlp.experts.down_proj.scales,
language_model.model.layers.7.mlp.experts.down_proj.weight,
language_model.model.layers.7.mlp.experts.gate_proj.biases,
language_model.model.layers.7.mlp.experts.gate_proj.scales,
language_model.model.layers.7.mlp.experts.gate_proj.weight,
language_model.model.layers.7.mlp.experts.up_proj.biases,
language_model.model.layers.7.mlp.experts.up_proj.scales,
language_model.model.layers.7.mlp.experts.up_proj.weight,
language_model.model.layers.8.mlp.experts.down_proj.biases,
language_model.model.layers.8.mlp.experts.down_proj.scales,
language_model.model.layers.8.mlp.experts.down_proj.weight,
language_model.model.layers.8.mlp.experts.gate_proj.biases,
language_model.model.layers.8.mlp.experts.gate_proj.scales,
language_model.model.layers.8.mlp.experts.gate_proj.weight,
language_model.model.layers.8.mlp.experts.up_proj.biases,
language_model.model.layers.8.mlp.experts.up_proj.scales,
language_model.model.layers.8.mlp.experts.up_proj.weight,
language_model.model.layers.9.mlp.experts.down_proj.biases,
language_model.model.layers.9.mlp.experts.down_proj.scales,
language_model.model.layers.9.mlp.experts.down_proj.weight,
language_model.model.layers.9.mlp.experts.gate_proj.biases,
language_model.model.layers.9.mlp.experts.gate_proj.scales,
language_model.model.layers.9.mlp.experts.gate_proj.weight,
language_model.model.layers.9.mlp.experts.up_proj.biases,
language_model.model.layers.9.mlp.experts.up_proj.scales,
language_model.model.layers.9.mlp.experts.up_proj.weight.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12427, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12012, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: Received 362 parameters not in model: 
audio_tower.encoder.biases,
audio_tower.encoder.scales,
language_model.model.layers.10.mlp.experts.down_proj.biases,
language_model.model.layers.10.mlp.experts.down_proj.scales,
language_model.model.layers.10.mlp.experts.down_proj.weight,
language_model.model.layers.10.mlp.experts.gate_proj.biases,
language_model.model.layers.10.mlp.experts.gate_proj.scales,
language_model.model.layers.10.mlp.experts.gate_proj.weight,
language_model.model.layers.10.mlp.experts.up_proj.biases,
language_model.model.layers.10.mlp.experts.up_proj.scales,
language_model.model.layers.10.mlp.experts.up_proj.weight,
language_model.model.layers.11.mlp.experts.down_proj.biases,
language_model.model.layers.11.mlp.experts.down_proj.scales,
language_model.model.layers.11.mlp.experts.down_proj.weight,
language_model.model.layers.11.mlp.experts.gate_proj.biases,
language_model.model.layers.11.mlp.experts.gate_proj.scales,
language_model.model.layers.11.mlp.experts.gate_proj.weight,
language_model.model.layers.11.mlp.experts.up_proj.biases,
language_model.model.layers.11.mlp.experts.up_proj.scales,
language_model.model.layers.11.mlp.experts.up_proj.weight,
language_model.model.layers.12.mlp.experts.down_proj.biases,
language_model.model.layers.12.mlp.experts.down_proj.scales,
language_model.model.layers.12.mlp.experts.down_proj.weight,
language_model.model.layers.12.mlp.experts.gate_proj.biases,
language_model.model.layers.12.mlp.experts.gate_proj.scales,
language_model.model.layers.12.mlp.experts.gate_proj.weight,
language_model.model.layers.12.mlp.experts.up_proj.biases,
language_model.model.layers.12.mlp.experts.up_proj.scales,
language_model.model.layers.12.mlp.experts.up_proj.weight,
language_model.model.layers.13.mlp.experts.down_proj.biases,
language_model.model.layers.13.mlp.experts.down_proj.scales,
language_model.model.layers.13.mlp.experts.down_proj.weight,
language_model.model.layers.13.mlp.experts.gate_proj.biases,
language_model.model.layers.13.mlp.experts.gate_proj.scales,
language_model.model.layers.13.mlp.experts.gate_proj.weight,
language_model.model.layers.13.mlp.experts.up_proj.biases,
language_model.model.layers.13.mlp.experts.up_proj.scales,
language_model.model.layers.13.mlp.experts.up_proj.weight,
language_model.model.layers.14.mlp.experts.down_proj.biases,
language_model.model.layers.14.mlp.experts.down_proj.scales,
language_model.model.layers.14.mlp.experts.down_proj.weight,
language_model.model.layers.14.mlp.experts.gate_proj.biases,
language_model.model.layers.14.mlp.experts.gate_proj.scales,
language_model.model.layers.14.mlp.experts.gate_proj.weight,
language_model.model.layers.14.mlp.experts.up_proj.biases,
language_model.model.layers.14.mlp.experts.up_proj.scales,
language_model.model.layers.14.mlp.experts.up_proj.weight,
language_model.model.layers.15.mlp.experts.down_proj.biases,
language_model.model.layers.15.mlp.experts.down_proj.scales,
language_model.model.layers.15.mlp.experts.down_proj.weight,
language_model.model.layers.15.mlp.experts.gate_proj.biases,
language_model.model.layers.15.mlp.experts.gate_proj.scales,
language_model.model.layers.15.mlp.experts.gate_proj.weight,
language_model.model.layers.15.mlp.experts.up_proj.biases,
language_model.model.layers.15.mlp.experts.up_proj.scales,
language_model.model.layers.15.mlp.experts.up_proj.weight,
language_model.model.layers.16.mlp.experts.down_proj.biases,
language_model.model.layers.16.mlp.experts.down_proj.scales,
language_model.model.layers.16.mlp.experts.down_proj.weight,
language_model.model.layers.16.mlp.experts.gate_proj.biases,
language_model.model.layers.16.mlp.experts.gate_proj.scales,
language_model.model.layers.16.mlp.experts.gate_proj.weight,
language_model.model.layers.16.mlp.experts.up_proj.biases,
language_model.model.layers.16.mlp.experts.up_proj.scales,
language_model.model.layers.16.mlp.experts.up_proj.weight,
language_model.model.layers.17.mlp.experts.down_proj.biases,
language_model.model.layers.17.mlp.experts.down_proj.scales,
language_model.model.layers.17.mlp.experts.down_proj.weight,
language_model.model.layers.17.mlp.experts.gate_proj.biases,
language_model.model.layers.17.mlp.experts.gate_proj.scales,
language_model.model.layers.17.mlp.experts.gate_proj.weight,
language_model.model.layers.17.mlp.experts.up_proj.biases,
language_model.model.layers.17.mlp.experts.up_proj.scales,
language_model.model.layers.17.mlp.experts.up_proj.weight,
language_model.model.layers.18.mlp.experts.down_proj.biases,
language_model.model.layers.18.mlp.experts.down_proj.scales,
language_model.model.layers.18.mlp.experts.down_proj.weight,
language_model.model.layers.18.mlp.experts.gate_proj.biases,
language_model.model.layers.18.mlp.experts.gate_proj.scales,
language_model.model.layers.18.mlp.experts.gate_proj.weight,
language_model.model.layers.18.mlp.experts.up_proj.biases,
language_model.model.layers.18.mlp.experts.up_proj.scales,
language_model.model.layers.18.mlp.experts.up_proj.weight,
language_model.model.layers.19.mlp.experts.down_proj.biases,
language_model.model.layers.19.mlp.experts.down_proj.scales,
language_model.model.layers.19.mlp.experts.down_proj.weight,
language_model.model.layers.19.mlp.experts.gate_proj.biases,
language_model.model.layers.19.mlp.experts.gate_proj.scales,
language_model.model.layers.19.mlp.experts.gate_proj.weight,
language_model.model.layers.19.mlp.experts.up_proj.biases,
language_model.model.layers.19.mlp.experts.up_proj.scales,
language_model.model.layers.19.mlp.experts.up_proj.weight,
language_model.model.layers.2.mlp.experts.down_proj.biases,
language_model.model.layers.2.mlp.experts.down_proj.scales,
language_model.model.layers.2.mlp.experts.down_proj.weight,
language_model.model.layers.2.mlp.experts.gate_proj.biases,
language_model.model.layers.2.mlp.experts.gate_proj.scales,
language_model.model.layers.2.mlp.experts.gate_proj.weight,
language_model.model.layers.2.mlp.experts.up_proj.biases,
language_model.model.layers.2.mlp.experts.up_proj.scales,
language_model.model.layers.2.mlp.experts.up_proj.weight,
language_model.model.layers.20.mlp.experts.down_proj.biases,
language_model.model.layers.20.mlp.experts.down_proj.scales,
language_model.model.layers.20.mlp.experts.down_proj.weight,
language_model.model.layers.20.mlp.experts.gate_proj.biases,
language_model.model.layers.20.mlp.experts.gate_proj.scales,
language_model.model.layers.20.mlp.experts.gate_proj.weight,
language_model.model.layers.20.mlp.experts.up_proj.biases,
language_model.model.layers.20.mlp.experts.up_proj.scales,
language_model.model.layers.20.mlp.experts.up_proj.weight,
language_model.model.layers.21.mlp.experts.down_proj.biases,
language_model.model.layers.21.mlp.experts.down_proj.scales,
language_model.model.layers.21.mlp.experts.down_proj.weight,
language_model.model.layers.21.mlp.experts.gate_proj.biases,
language_model.model.layers.21.mlp.experts.gate_proj.scales,
language_model.model.layers.21.mlp.experts.gate_proj.weight,
language_model.model.layers.21.mlp.experts.up_proj.biases,
language_model.model.layers.21.mlp.experts.up_proj.scales,
language_model.model.layers.21.mlp.experts.up_proj.weight,
language_model.model.layers.22.mlp.experts.down_proj.biases,
language_model.model.layers.22.mlp.experts.down_proj.scales,
language_model.model.layers.22.mlp.experts.down_proj.weight,
language_model.model.layers.22.mlp.experts.gate_proj.biases,
language_model.model.layers.22.mlp.experts.gate_proj.scales,
language_model.model.layers.22.mlp.experts.gate_proj.weight,
language_model.model.layers.22.mlp.experts.up_proj.biases,
language_model.model.layers.22.mlp.experts.up_proj.scales,
language_model.model.layers.22.mlp.experts.up_proj.weight,
language_model.model.layers.23.mlp.experts.down_proj.biases,
language_model.model.layers.23.mlp.experts.down_proj.scales,
language_model.model.layers.23.mlp.experts.down_proj.weight,
language_model.model.layers.23.mlp.experts.gate_proj.biases,
language_model.model.layers.23.mlp.experts.gate_proj.scales,
language_model.model.layers.23.mlp.experts.gate_proj.weight,
language_model.model.layers.23.mlp.experts.up_proj.biases,
language_model.model.layers.23.mlp.experts.up_proj.scales,
language_model.model.layers.23.mlp.experts.up_proj.weight,
language_model.model.layers.24.mlp.experts.down_proj.biases,
language_model.model.layers.24.mlp.experts.down_proj.scales,
language_model.model.layers.24.mlp.experts.down_proj.weight,
language_model.model.layers.24.mlp.experts.gate_proj.biases,
language_model.model.layers.24.mlp.experts.gate_proj.scales,
language_model.model.layers.24.mlp.experts.gate_proj.weight,
language_model.model.layers.24.mlp.experts.up_proj.biases,
language_model.model.layers.24.mlp.experts.up_proj.scales,
language_model.model.layers.24.mlp.experts.up_proj.weight,
language_model.model.layers.25.mlp.experts.down_proj.biases,
language_model.model.layers.25.mlp.experts.down_proj.scales,
language_model.model.layers.25.mlp.experts.down_proj.weight,
language_model.model.layers.25.mlp.experts.gate_proj.biases,
language_model.model.layers.25.mlp.experts.gate_proj.scales,
language_model.model.layers.25.mlp.experts.gate_proj.weight,
language_model.model.layers.25.mlp.experts.up_proj.biases,
language_model.model.layers.25.mlp.experts.up_proj.scales,
language_model.model.layers.25.mlp.experts.up_proj.weight,
language_model.model.layers.26.mlp.experts.down_proj.biases,
language_model.model.layers.26.mlp.experts.down_proj.scales,
language_model.model.layers.26.mlp.experts.down_proj.weight,
language_model.model.layers.26.mlp.experts.gate_proj.biases,
language_model.model.layers.26.mlp.experts.gate_proj.scales,
language_model.model.layers.26.mlp.experts.gate_proj.weight,
language_model.model.layers.26.mlp.experts.up_proj.biases,
language_model.model.layers.26.mlp.experts.up_proj.scales,
language_model.model.layers.26.mlp.experts.up_proj.weight,
language_model.model.layers.27.mlp.experts.down_proj.biases,
language_model.model.layers.27.mlp.experts.down_proj.scales,
language_model.model.layers.27.mlp.experts.down_proj.weight,
language_model.model.layers.27.mlp.experts.gate_proj.biases,
language_model.model.layers.27.mlp.experts.gate_proj.scales,
language_model.model.layers.27.mlp.experts.gate_proj.weight,
language_model.model.layers.27.mlp.experts.up_proj.biases,
language_model.model.layers.27.mlp.experts.up_proj.scales,
language_model.model.layers.27.mlp.experts.up_proj.weight,
language_model.model.layers.28.mlp.experts.down_proj.biases,
language_model.model.layers.28.mlp.experts.down_proj.scales,
language_model.model.layers.28.mlp.experts.down_proj.weight,
language_model.model.layers.28.mlp.experts.gate_proj.biases,
language_model.model.layers.28.mlp.experts.gate_proj.scales,
language_model.model.layers.28.mlp.experts.gate_proj.weight,
language_model.model.layers.28.mlp.experts.up_proj.biases,
language_model.model.layers.28.mlp.experts.up_proj.scales,
language_model.model.layers.28.mlp.experts.up_proj.weight,
language_model.model.layers.29.mlp.experts.down_proj.biases,
language_model.model.layers.29.mlp.experts.down_proj.scales,
language_model.model.layers.29.mlp.experts.down_proj.weight,
language_model.model.layers.29.mlp.experts.gate_proj.biases,
language_model.model.layers.29.mlp.experts.gate_proj.scales,
language_model.model.layers.29.mlp.experts.gate_proj.weight,
language_model.model.layers.29.mlp.experts.up_proj.biases,
language_model.model.layers.29.mlp.experts.up_proj.scales,
language_model.model.layers.29.mlp.experts.up_proj.weight,
language_model.model.layers.3.mlp.experts.down_proj.biases,
language_model.model.layers.3.mlp.experts.down_proj.scales,
language_model.model.layers.3.mlp.experts.down_proj.weight,
language_model.model.layers.3.mlp.experts.gate_proj.biases,
language_model.model.layers.3.mlp.experts.gate_proj.scales,
language_model.model.layers.3.mlp.experts.gate_proj.weight,
language_model.model.layers.3.mlp.experts.up_proj.biases,
language_model.model.layers.3.mlp.experts.up_proj.scales,
language_model.model.layers.3.mlp.experts.up_proj.weight,
language_model.model.layers.30.mlp.experts.down_proj.biases,
language_model.model.layers.30.mlp.experts.down_proj.scales,
language_model.model.layers.30.mlp.experts.down_proj.weight,
language_model.model.layers.30.mlp.experts.gate_proj.biases,
language_model.model.layers.30.mlp.experts.gate_proj.scales,
language_model.model.layers.30.mlp.experts.gate_proj.weight,
language_model.model.layers.30.mlp.experts.up_proj.biases,
language_model.model.layers.30.mlp.experts.up_proj.scales,
language_model.model.layers.30.mlp.experts.up_proj.weight,
language_model.model.layers.31.mlp.experts.down_proj.biases,
language_model.model.layers.31.mlp.experts.down_proj.scales,
language_model.model.layers.31.mlp.experts.down_proj.weight,
language_model.model.layers.31.mlp.experts.gate_proj.biases,
language_model.model.layers.31.mlp.experts.gate_proj.scales,
language_model.model.layers.31.mlp.experts.gate_proj.weight,
language_model.model.layers.31.mlp.experts.up_proj.biases,
language_model.model.layers.31.mlp.experts.up_proj.scales,
language_model.model.layers.31.mlp.experts.up_proj.weight,
language_model.model.layers.32.mlp.experts.down_proj.biases,
language_model.model.layers.32.mlp.experts.down_proj.scales,
language_model.model.layers.32.mlp.experts.down_proj.weight,
language_model.model.layers.32.mlp.experts.gate_proj.biases,
language_model.model.layers.32.mlp.experts.gate_proj.scales,
language_model.model.layers.32.mlp.experts.gate_proj.weight,
language_model.model.layers.32.mlp.experts.up_proj.biases,
language_model.model.layers.32.mlp.experts.up_proj.scales,
language_model.model.layers.32.mlp.experts.up_proj.weight,
language_model.model.layers.33.mlp.experts.down_proj.biases,
language_model.model.layers.33.mlp.experts.down_proj.scales,
language_model.model.layers.33.mlp.experts.down_proj.weight,
language_model.model.layers.33.mlp.experts.gate_proj.biases,
language_model.model.layers.33.mlp.experts.gate_proj.scales,
language_model.model.layers.33.mlp.experts.gate_proj.weight,
language_model.model.layers.33.mlp.experts.up_proj.biases,
language_model.model.layers.33.mlp.experts.up_proj.scales,
language_model.model.layers.33.mlp.experts.up_proj.weight,
language_model.model.layers.34.mlp.experts.down_proj.biases,
language_model.model.layers.34.mlp.experts.down_proj.scales,
language_model.model.layers.34.mlp.experts.down_proj.weight,
language_model.model.layers.34.mlp.experts.gate_proj.biases,
language_model.model.layers.34.mlp.experts.gate_proj.scales,
language_model.model.layers.34.mlp.experts.gate_proj.weight,
language_model.model.layers.34.mlp.experts.up_proj.biases,
language_model.model.layers.34.mlp.experts.up_proj.scales,
language_model.model.layers.34.mlp.experts.up_proj.weight,
language_model.model.layers.35.mlp.experts.down_proj.biases,
language_model.model.layers.35.mlp.experts.down_proj.scales,
language_model.model.layers.35.mlp.experts.down_proj.weight,
language_model.model.layers.35.mlp.experts.gate_proj.biases,
language_model.model.layers.35.mlp.experts.gate_proj.scales,
language_model.model.layers.35.mlp.experts.gate_proj.weight,
language_model.model.layers.35.mlp.experts.up_proj.biases,
language_model.model.layers.35.mlp.experts.up_proj.scales,
language_model.model.layers.35.mlp.experts.up_proj.weight,
language_model.model.layers.36.mlp.experts.down_proj.biases,
language_model.model.layers.36.mlp.experts.down_proj.scales,
language_model.model.layers.36.mlp.experts.down_proj.weight,
language_model.model.layers.36.mlp.experts.gate_proj.biases,
language_model.model.layers.36.mlp.experts.gate_proj.scales,
language_model.model.layers.36.mlp.experts.gate_proj.weight,
language_model.model.layers.36.mlp.experts.up_proj.biases,
language_model.model.layers.36.mlp.experts.up_proj.scales,
language_model.model.layers.36.mlp.experts.up_proj.weight,
language_model.model.layers.37.mlp.experts.down_proj.biases,
language_model.model.layers.37.mlp.experts.down_proj.scales,
language_model.model.layers.37.mlp.experts.down_proj.weight,
language_model.model.layers.37.mlp.experts.gate_proj.biases,
language_model.model.layers.37.mlp.experts.gate_proj.scales,
language_model.model.layers.37.mlp.experts.gate_proj.weight,
language_model.model.layers.37.mlp.experts.up_proj.biases,
language_model.model.layers.37.mlp.experts.up_proj.scales,
language_model.model.layers.37.mlp.experts.up_proj.weight,
language_model.model.layers.38.mlp.experts.down_proj.biases,
language_model.model.layers.38.mlp.experts.down_proj.scales,
language_model.model.layers.38.mlp.experts.down_proj.weight,
language_model.model.layers.38.mlp.experts.gate_proj.biases,
language_model.model.layers.38.mlp.experts.gate_proj.scales,
language_model.model.layers.38.mlp.experts.gate_proj.weight,
language_model.model.layers.38.mlp.experts.up_proj.biases,
language_model.model.layers.38.mlp.experts.up_proj.scales,
language_model.model.layers.38.mlp.experts.up_proj.weight,
language_model.model.layers.39.mlp.experts.down_proj.biases,
language_model.model.layers.39.mlp.experts.down_proj.scales,
language_model.model.layers.39.mlp.experts.down_proj.weight,
language_model.model.layers.39.mlp.experts.gate_proj.biases,
language_model.model.layers.39.mlp.experts.gate_proj.scales,
language_model.model.layers.39.mlp.experts.gate_proj.weight,
language_model.model.layers.39.mlp.experts.up_proj.biases,
language_model.model.layers.39.mlp.experts.up_proj.scales,
language_model.model.layers.39.mlp.experts.up_proj.weight,
language_model.model.layers.4.mlp.experts.down_proj.biases,
language_model.model.layers.4.mlp.experts.down_proj.scales,
language_model.model.layers.4.mlp.experts.down_proj.weight,
language_model.model.layers.4.mlp.experts.gate_proj.biases,
language_model.model.layers.4.mlp.experts.gate_proj.scales,
language_model.model.layers.4.mlp.experts.gate_proj.weight,
language_model.model.layers.4.mlp.experts.up_proj.biases,
language_model.model.layers.4.mlp.experts.up_proj.scales,
language_model.model.layers.4.mlp.experts.up_proj.weight,
language_model.model.layers.40.mlp.experts.down_proj.biases,
language_model.model.layers.40.mlp.experts.down_proj.scales,
language_model.model.layers.40.mlp.experts.down_proj.weight,
language_model.model.layers.40.mlp.experts.gate_proj.biases,
language_model.model.layers.40.mlp.experts.gate_proj.scales,
language_model.model.layers.40.mlp.experts.gate_proj.weight,
language_model.model.layers.40.mlp.experts.up_proj.biases,
language_model.model.layers.40.mlp.experts.up_proj.scales,
language_model.model.layers.40.mlp.experts.up_proj.weight,
language_model.model.layers.41.mlp.experts.down_proj.biases,
language_model.model.layers.41.mlp.experts.down_proj.scales,
language_model.model.layers.41.mlp.experts.down_proj.weight,
language_model.model.layers.41.mlp.experts.gate_proj.biases,
language_model.model.layers.41.mlp.experts.gate_proj.scales,
language_model.model.layers.41.mlp.experts.gate_proj.weight,
language_model.model.layers.41.mlp.experts.up_proj.biases,
language_model.model.layers.41.mlp.experts.up_proj.scales,
language_model.model.layers.41.mlp.experts.up_proj.weight,
language_model.model.layers.5.mlp.experts.down_proj.biases,
language_model.model.layers.5.mlp.experts.down_proj.scales,
language_model.model.layers.5.mlp.experts.down_proj.weight,
language_model.model.layers.5.mlp.experts.gate_proj.biases,
language_model.model.layers.5.mlp.experts.gate_proj.scales,
language_model.model.layers.5.mlp.experts.gate_proj.weight,
language_model.model.layers.5.mlp.experts.up_proj.biases,
language_model.model.layers.5.mlp.experts.up_proj.scales,
language_model.model.layers.5.mlp.experts.up_proj.weight,
language_model.model.layers.6.mlp.experts.down_proj.biases,
language_model.model.layers.6.mlp.experts.down_proj.scales,
language_model.model.layers.6.mlp.experts.down_proj.weight,
language_model.model.layers.6.mlp.experts.gate_proj.biases,
language_model.model.layers.6.mlp.experts.gate_proj.scales,
language_model.model.layers.6.mlp.experts.gate_proj.weight,
language_model.model.layers.6.mlp.experts.up_proj.biases,
language_model.model.layers.6.mlp.experts.up_proj.scales,
language_model.model.layers.6.mlp.experts.up_proj.weight,
language_model.model.layers.7.mlp.experts.down_proj.biases,
language_model.model.layers.7.mlp.experts.down_proj.scales,
language_model.model.layers.7.mlp.experts.down_proj.weight,
language_model.model.layers.7.mlp.experts.gate_proj.biases,
language_model.model.layers.7.mlp.experts.gate_proj.scales,
language_model.model.layers.7.mlp.experts.gate_proj.weight,
language_model.model.layers.7.mlp.experts.up_proj.biases,
language_model.model.layers.7.mlp.experts.up_proj.scales,
language_model.model.layers.7.mlp.experts.up_proj.weight,
language_model.model.layers.8.mlp.experts.down_proj.biases,
language_model.model.layers.8.mlp.experts.down_proj.scales,
language_model.model.layers.8.mlp.experts.down_proj.weight,
language_model.model.layers.8.mlp.experts.gate_proj.biases,
language_model.model.layers.8.mlp.experts.gate_proj.scales,
language_model.model.layers.8.mlp.experts.gate_proj.weight,
language_model.model.layers.8.mlp.experts.up_proj.biases,
language_model.model.layers.8.mlp.experts.up_proj.scales,
language_model.model.layers.8.mlp.experts.up_proj.weight,
language_model.model.layers.9.mlp.experts.down_proj.biases,
language_model.model.layers.9.mlp.experts.down_proj.scales,
language_model.model.layers.9.mlp.experts.down_proj.weight,
language_model.model.layers.9.mlp.experts.gate_proj.biases,
language_model.model.layers.9.mlp.experts.gate_proj.scales,
language_model.model.layers.9.mlp.experts.gate_proj.weight,
language_model.model.layers.9.mlp.experts.up_proj.biases,
language_model.model.layers.9.mlp.experts.up_proj.scales,
language_model.model.layers.9.mlp.experts.up_proj.weight.

```

#### Captured upstream output

```text
=== STDERR ===
Downloading bytes:           |  0.00B
Reconstructing (incomplete total...): |          |  0.00B /  0.00B
Fetching 54 files:   0%|          | 0/54 [00:00<?, ?it/s]
Fetching 54 files: 100%|##########| 54/54 [00:00<00:00, 5373.61it/s]
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
[20:08:29] DEBUG    HF Cache Info for mlx-community/Inkling-Small-mlx-4bit: size=146358.0 MB, files=58
```

</details>

---
