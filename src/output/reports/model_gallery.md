# Model Output Gallery

Generated on: 2026-08-28 20:07:02 BST

Complete per-model evidence artifact with image metadata, the source prompt, a
facts-only chooser, and full generated or crash output for every attempted
model.

## Reference Image

![Reference image](assets/source-image.jpg)

## Current-run Chooser

Current-run usability and captured resource facts only. Total time is end-to-end; throughput covers generation only and requires at least 16 generated tokens. Prefill/first is first-token latency when captured; Prompt tok is the full rendered prompt including image tokens, which drives prefill cost. For cross-attention architectures the token count reflects the tokenised text burden only, not total vision prefill compute.

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                                   | Usability             | Total s | Gen TPS    | Prefill/first s | Peak GB | Prompt tok | Gen tok | Observations                                                                                                                |
|-------------------------------------------------------------------------------------------------------------------------|-----------------------|---------|------------|-----------------|---------|------------|---------|-----------------------------------------------------------------------------------------------------------------------------|
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `usable`              | 1.60s   | 481 tok/s  | 0.11            | 1.9     | 2,117      | 118     | none                                                                                                                        |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `usable`              | 10.95s  | 29.4 tok/s | 3.58            | 23      | 2,401      | 121     | none                                                                                                                        |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `usable`              | 8.59s   | 34.0 tok/s | 1.08            | 18      | 2,618      | 164     | none                                                                                                                        |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `usable`              | 5.82s   | 67.6 tok/s | 1.99            | 13      | 2,934      | 123     | none                                                                                                                        |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `usable`              | 3.21s   | 190 tok/s  | 0.98            | 7.8     | 2,933      | 134     | none                                                                                                                        |
| [`mlx-community/Molmo2-8B-4bit`](#model-mlx-community-molmo2-8b-4bit)                                                   | `usable`              | 5.50s   | 72.5 tok/s | 1.93            | 8.8     | 1,529      | 133     | none                                                                                                                        |
| [`mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit`](#model-mlx-community-qwen3-vl-30b-a3b-instruct-4bit)                   | `usable`              | 59.53s  | 87.1 tok/s | 55.14           | 23      | 16,552     | 129     | none                                                                                                                        |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `usable`              | 56.24s  | 110 tok/s  | 51.54           | 24      | 16,568     | 111     | none                                                                                                                        |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `usable`              | 8.77s   | 31.3 tok/s | 1.03            | 17      | 593        | 153     | none                                                                                                                        |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `usable`              | 4.18s   | 129 tok/s  | 0.41            | 16      | 598        | 100     | none                                                                                                                        |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `usable`              | 7.88s   | 26.4 tok/s | 1.13            | 20      | 598        | 95      | none                                                                                                                        |
| [`mlx-community/granite-4.0-3b-vision-4bit`](#model-mlx-community-granite-40-3b-vision-4bit)                            | `usable`              | 2.88s   | 171 tok/s  | 0.92            | 4.6     | 1,383      | 85      | none                                                                                                                        |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `usable with caveats` | 15.97s  | 93.4 tok/s | 13.25           | 8.4     | 16,552     | 112     | title/keyword constraints failed                                                                                            |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `usable with caveats` | 9.72s   | 79.6 tok/s | 6.23            | 8.4     | 6,392      | 126     | title/keyword constraints failed                                                                                            |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `usable with caveats` | 29.24s  | 41.5 tok/s | 18.02           | 78      | 6,392      | 95      | control tokens visible                                                                                                      |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `usable with caveats` | 6.30s   | 36.5 tok/s | 1.30            | 17      | 2,118      | 99      | title/keyword constraints failed                                                                                            |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `usable with caveats` | 150.24s | 4.73 tok/s | 1.84            | 40      | 1,329      | 682     | role tokens visible                                                                                                         |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `usable with caveats` | 6.45s   | 64.8 tok/s | 2.01            | 13      | 2,934      | 161     | title/keyword constraints failed                                                                                            |
| [`mlx-community/North-Micro-Vision-Instruct-4bit`](#model-mlx-community-north-micro-vision-instruct-4bit)               | `usable with caveats` | 5.61s   | 230 tok/s  | 3.10            | 3.9     | 4,085      | 161     | title/keyword constraints failed                                                                                            |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `usable with caveats` | 66.45s  | 64.2 tok/s | 53.62           | 74      | 16,568     | 110     | title/keyword constraints failed; draft hints copied unchanged                                                              |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `usable with caveats` | 4.40s   | 56.3 tok/s | 0.43            | 9.4     | 1,138      | 135     | title/keyword constraints failed                                                                                            |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `usable with caveats` | 27.70s  | 90.3 tok/s | 15.88           | 8.4     | 16,554     | 913     | title/keyword constraints failed                                                                                            |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `usable with caveats` | 65.40s  | 89.0 tok/s | 61.47           | 10.0    | 16,568     | 139     | title/keyword constraints failed                                                                                            |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `usable with caveats` | 98.20s  | 17.2 tok/s | 85.84           | 33      | 16,568     | 135     | title/keyword constraints failed                                                                                            |
| [`mlx-community/Qwen3.8-27B-4bit`](#model-mlx-community-qwen38-27b-4bit)                                                | `usable with caveats` | 83.60s  | 29.5 tok/s | 75.98           | 21      | 16,568     | 133     | title/keyword constraints failed                                                                                            |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `usable with caveats` | 2.67s   | 123 tok/s  | 0.39            | 5.5     | 1,430      | 104     | title/keyword constraints failed; draft hints copied unchanged                                                              |
| [`mlx-community/Step-3.7-Flash-oQ2e`](#model-mlx-community-step-37-flash-oq2e)                                          | `usable with caveats` | 38.45s  | 18.8 tok/s | 22.33           | 70      | 3,496      | 108     | title/keyword constraints failed; draft hints copied unchanged                                                              |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `usable with caveats` | 6.48s   | 53.4 tok/s | 0.65            | 29      | 594        | 88      | control tokens visible                                                                                                      |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `usable with caveats` | 6.00s   | 60.4 tok/s | 0.42            | 28      | 594        | 84      | control tokens visible                                                                                                      |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `usable with caveats` | 6.96s   | 39.4 tok/s | 1.90            | 16      | 3,124      | 112     | title/keyword constraints failed                                                                                            |
| [`jinaai/jina-vlm-mlx`](#model-jinaai-jina-vlm-mlx)                                                                     | `unusable`            | 10.15s  | 133 tok/s  | 1.34            | 3.7     | 2,072      | 1,000   | repeated text; cut off at token limit; title/keyword constraints failed                                                     |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `unusable`            | 28.43s  | 41.3 tok/s | 2.03            | 15      | 3,215      | 1,000   | missing required fields; echoes instructions; cut off at token limit                                                        |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `unusable`            | 33.20s  | 50.9 tok/s | 3.95            | 60      | 1,641      | 1,000   | repeated text; cut off at token limit; title/keyword constraints failed                                                     |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `unusable`            | 2.18s   | 344 tok/s  | 0.22            | 2.1     | 339        | 54      | missing required fields                                                                                                     |
| [`mlx-community/GLM-4.1V-9B-Thinking-8bit`](#model-mlx-community-glm-41v-9b-thinking-8bit)                              | `unusable`            | 29.14s  | 47.0 tok/s | 5.57            | 13      | 6,388      | 1,000   | repeated text; extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `unusable`            | 6.86s   | 186 tok/s  | 0.34            | 4.0     | 2,117      | 1,000   | repeated text; cut off at token limit; title/keyword constraints failed                                                     |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `unusable`            | 2.03s   | 275 tok/s  | 0.25            | 3.8     | 937        | 76      | missing required fields; extra text before Title                                                                            |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `unusable`            | 38.23s  | 55.8 tok/s | 18.09           | 13      | 16,563     | 1,000   | repeated text; cut off at token limit; title/keyword constraints failed                                                     |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `unusable`            | 7.10s   | 48.7 tok/s | 0.87            | 17      | 592        | 164     | missing required fields                                                                                                     |
| [`mlx-community/LFM2.5-VL-3B-OptiQ-4bit`](#model-mlx-community-lfm25-vl-3b-optiq-4bit)                                  | `not evaluated`       | 0.21s   | -          | -               | -       | -          | -       | none                                                                                                                        |
| [`tencent/Youtu-VL-4B-Instruct`](#model-tencent-youtu-vl-4b-instruct)                                                   | `not evaluated`       | 1.01s   | -          | -               | -       | -          | -       | none                                                                                                                        |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Resource Highlights

Fastest clean completion: `LiquidAI/LFM2.5-VL-450M-MLX-bf16` at 481 tok/s

Average clean-completion throughput: 119 tok/s (indicative only: tokenizers and architectures differ across models)

Lowest peak memory among clean completions: `LiquidAI/LFM2.5-VL-450M-MLX-bf16` at 1.9 GB

## Avoid for This Run

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                        | Usability       | Observations                                                                                                                |
|--------------------------------------------------------------------------------------------------------------|-----------------|-----------------------------------------------------------------------------------------------------------------------------|
| [`jinaai/jina-vlm-mlx`](#model-jinaai-jina-vlm-mlx)                                                          | `unusable`      | repeated text; cut off at token limit; title/keyword constraints failed                                                     |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)       | `unusable`      | missing required fields; echoes instructions; cut off at token limit                                                        |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16) | `unusable`      | repeated text; cut off at token limit; title/keyword constraints failed                                                     |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                   | `unusable`      | missing required fields                                                                                                     |
| [`mlx-community/GLM-4.1V-9B-Thinking-8bit`](#model-mlx-community-glm-41v-9b-thinking-8bit)                   | `unusable`      | repeated text; extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                | `unusable`      | repeated text; cut off at token limit; title/keyword constraints failed                                                     |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                 | `unusable`      | missing required fields; extra text before Title                                                                            |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                | `unusable`      | repeated text; cut off at token limit; title/keyword constraints failed                                                     |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                            | `unusable`      | missing required fields                                                                                                     |
| [`mlx-community/LFM2.5-VL-3B-OptiQ-4bit`](#model-mlx-community-lfm25-vl-3b-optiq-4bit)                       | `not evaluated` | none                                                                                                                        |
| [`tencent/Youtu-VL-4B-Instruct`](#model-tencent-youtu-vl-4b-instruct)                                        | `not evaluated` | none                                                                                                                        |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Output at a Glance

The first 280 characters of each model's actual output (or failure evidence for crashes), in chooser order. Complete exact evidence for every model follows below.

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                                   | Usability             | Output preview                                                                                                                                                                                                                                                                                                   |
|-------------------------------------------------------------------------------------------------------------------------|-----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `usable`              | Title:<br>Boathouse on stilts in a wetland<br><br>Description:<br>A weathered wooden boathouse stands on stilts over a serene pond, framed by lush overhanging tree branches and surrounding wetland foliage. The scene is bathed in soft, diffused light under a cloudy sky, with reflections of...             |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `usable`              | Title:<br>Weathered Boathouse Over Serene Pond<br><br>Description:<br>A weathered wooden boathouse on stilts overlooks a tranquil pond, framed by lush overhanging tree branches and surrounded by wetland foliage, captured on a cloudy afternoon in August 2026.<br><br>Keywords:<br>Boathouse, Pond, We...    |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `usable`              | Title: Weathered Wooden Boathouse on Stilts Over a Serene Pond.<br><br>Description: A weathered wooden boathouse built on stilts stands over the edge of a serene pond, framed by lush overhanging tree branches, wooden decking, and surrounding wetland foliage. The boathouse's roof is...                    |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `usable`              | Title:<br>**"Stilted Boathouse Over a Wetland Pond"**<br><br>Description:<br>A weathered wooden boathouse on stilts extends over a calm pond, surrounded by lush foliage and overhanging tree branches, captured on a cloudy August afternoon in a wetland landscape.<br><br>Keywords:<br>Boathouse, wetla...    |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `usable`              | Title:<br>Weathered Pondside Wooden Shed in Overgrown Wetland<br><br>Description:<br>A rustic, elevated wooden boathouse or bird hide, constructed on a wooden deck over a shallow pond, is bathed in soft cloudy light amid dense foliage, moss, and reeds. The structure’s weathered planks and...             |
| [`mlx-community/Molmo2-8B-4bit`](#model-mlx-community-molmo2-8b-4bit)                                                   | `usable`              | Title: Stilted Wooden Hide in Marshy Wetland<br><br>Description: A weathered wooden structure on stilts overlooks a serene pond, surrounded by lush foliage and reeds. Built for birdwatching, it features a wooden deck and is partially obscured by overhanging branches, with a cloudy s...                   |
| [`mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit`](#model-mlx-community-qwen3-vl-30b-a3b-instruct-4bit)                   | `usable`              | Title: Weathered wooden boathouse on a calm pond<br>Description: A weathered wooden boathouse stands on stilts over a calm pond, its dark timber walls and moss-covered roof contrasting with the lush green foliage and overhanging tree branches. The scene, captured on a cloudy day,...                      |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `usable`              | Title: Weathered Wooden Boathouse Over Pond<br><br>Description: Captured on 21 August 2026, this weathered wooden boathouse stands on stilts over a calm pond, framed by lush overhanging tree branches and surrounded by wetland foliage under a cloudy sky.<br><br>Keywords: boathouse, pond, w...             |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `usable`              | Title: Boathouse on Stilts, Wetland Landscape<br><br>Description: Captured on 21st August 2026 at 14:34 UTC+01, this weathered wooden boathouse stands on stilts above a pond at 51.441113°N, 0.565406°W, with a cloudy sky and reflected light on the water. The structure is framed by lu...                   |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `usable`              | Title: Weathered wooden boathouse over a serene pond<br>Description: A rustic wooden boathouse built on stilts stands over a calm pond, framed by lush overhanging tree branches and wetland foliage under a cloudy sky. The structure features a wooden deck and reflects the surroundi...                      |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `usable`              | Title: Weathered Wooden Bird Hide Over a Serene Pond<br>Description: A weathered wooden bird hide on stilts extends over a pond, framed by lush overhanging tree branches under a cloudy sky. The structure is accessed via a wooden boardwalk and surrounded by wetland foliage.<br>Keywor...                   |
| [`mlx-community/granite-4.0-3b-vision-4bit`](#model-mlx-community-granite-40-3b-vision-4bit)                            | `usable`              | Title: "Weathered Wooden Boathouse by a Serene Pond"<br>Description: A weathered wooden boathouse stands on stilts over a calm pond, surrounded by lush foliage and a serene wetland landscape.<br>Keywords: Weathered Boathouse, Pond, Wetland, Lush Foliage, Marshland, Bird Hide, Birdwa...                   |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `usable_with_caveats` | Title: Wooden boathouse by pond<br><br>Description: A weathered wooden boathouse on stilts stands over a calm pond, surrounded by lush green foliage and reeds under a cloudy sky, with a wooden deck and moss-covered roof.<br><br>Keywords: boathouse, pond, wooden shed, stilts, foliage, reed...             |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `usable_with_caveats` | Title: Weathered Boathouse by Serene Pond<br>Description: A weathered wooden boathouse built on stilts stands over the edge of a serene pond, framed by lush overhanging tree branches, wooden decking, and surrounding wetland foliage.<br>Keywords: weathered boathouse, serene pond, ove...                   |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `usable_with_caveats` | <\|begin_of_box\|>Title: Weathered Wooden Boathouse Over Pond<br>Description: A weathered wooden boathouse on stilts stands over a serene pond, framed by lush tree branches, wooden decking, and surrounding wetland foliage, under a cloudy sky.<br>Keywords: boathouse, wooden, stilts, po...                 |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `usable_with_caveats` | Title: Stilted Boathouse Over Pond<br>Description: A weathered wooden boathouse on stilts overlooks a tranquil pond, surrounded by lush foliage and under a cloudy sky. The scene captures serene outdoor architecture amidst a wetland setting.<br>Keywords: Wooden boathouse, Stilts, Pon...                   |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `usable_with_caveats` | ◁think▷Okay, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. The user wants a 5-10 word title, a 1-2 sentence description, and 10-18 keywords.<br><br>Starting with the title: The image shows a weathered woo...                   |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `usable_with_caveats` | **Title:**<br>*"Weathered Boathouse Over Serene Wetland Pond"*<br><br>**Description:**<br>A weathered wooden boathouse, built on stilts, stands over a calm pond surrounded by lush foliage, reeds, and overhanging branches. Captured on **21 August 2026** under a cloudy sky, its wooden decki...             |
| [`mlx-community/North-Micro-Vision-Instruct-4bit`](#model-mlx-community-north-micro-vision-instruct-4bit)               | `usable_with_caveats` | Title: Boathouse in the Wetland Landscape<br><br>Description: A weathered wooden boathouse stands over the edge of a serene pond, framed by lush overhanging tree branches. The structure features wooden decking and a rustic wooden shed, with a wooden railing leading up to it. The sur...                   |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `usable_with_caveats` | Title: Wooden boathouse on stilts over a pond<br>Description: A weathered wooden boathouse built on stilts stands over the edge of a serene pond, framed by lush overhanging tree branches, wooden decking, and surrounding wetland foliage.<br>Keywords: Cloudy Sky, Foliage, Forest, Gras...                   |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `usable_with_caveats` | Title: Serene Wooden Boathouse Over Pond<br><br>Description: A weathered wooden boathouse stands majestically over a tranquil pond, surrounded by lush greenery and a cloudy sky, offering a peaceful retreat for nature enthusiasts.<br><br>Keywords: wooden boathouse, pond, serene, lush, gree...             |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `usable_with_caveats` | Got it, let's tackle this step by step. First, the title needs to be 5-10 words. The image shows a weathered wooden boathouse on stilts over a pond, with trees and wetland. So maybe "Weathered Stilt Boathouse by Pond with Mossy Roof" – that's 9 words, fits.<br><br>Next, the descript...                   |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `usable_with_caveats` | Title:<br>Weathered Wooden Bird Hide Over Serene Pond in Wetland<br><br>Description:<br>A weathered wooden bird hide stands on stilts over a calm pond, surrounded by lush greenery and overhanging trees under a cloudy sky. The structure features a wooden deck with railings and vertical pla...             |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `usable_with_caveats` | Title:<br>Wooden Boathouse on Stilts Over a Pond<br><br>Description:<br>A weathered wooden boathouse built on stilts stands over the edge of a serene pond, framed by lush overhanging tree branches, wooden decking, and surrounding wetland foliage. The scene is set under a cloudy sky, with...              |
| [`mlx-community/Qwen3.8-27B-4bit`](#model-mlx-community-qwen38-27b-4bit)                                                | `usable_with_caveats` | Title:<br>Weathered wooden boathouse on a serene wetland pond<br><br>Description:<br>A weathered wooden boathouse with a mossy roof stands on stilts over a calm pond, featuring a wooden boardwalk with cross-braced railings. The structure is framed by overhanging green tree branches and su...             |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `usable_with_caveats` | Title: Wooden Boathouse<br>Description: A weathered wooden boathouse built on stilts stands over the edge of a serene pond, framed by lush overhanging tree branches, wooden decking, and surrounding wetland foliage.<br>Keywords: Cloudy Sky, Foliage, Forest, Grass, Lake, Landscape, Le...                   |
| [`mlx-community/Step-3.7-Flash-oQ2e`](#model-mlx-community-step-37-flash-oq2e)                                          | `usable_with_caveats` | Title:<br>Weathered wooden boathouse over a pond<br><br>Description:<br>A weathered wooden boathouse built on stilts stands over the edge of a serene pond, framed by lush overhanging tree branches, wooden decking, and surrounding wetland foliage.<br><br>Keywords:<br>Cloudy Sky, Foliage, Forest, Gr...    |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `usable_with_caveats` | <\|channel>thought<br><channel\|>Title: Weathered Wooden Boathouse Over Serene Wetland Pond<br>Description: A weathered wooden boathouse built on stilts stands over a serene pond, framed by lush green trees and wetland foliage under a cloudy sky.<br>Keywords: architecture, birdwatching,...               |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `usable_with_caveats` | <\|channel>thought<br><channel\|>Title: Weathered Wooden Boathouse over a Serene Pond<br>Description: A weathered wooden boathouse built on stilts stands over a pond, framed by lush green foliage under a cloudy sky.<br>Keywords: Boathouse, Wooden, Pond, Stilts, Wetland, Foliage, Trees, B...              |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `usable_with_caveats` | Title:<br>Weathered Boathouse Overlooking Serene Wetland Pond<br><br>Description:<br>A weathered wooden boathouse stands on stilts over a tranquil pond, surrounded by lush foliage and framed by overhanging tree branches under a cloudy sky.<br><br>Keywords:<br>Cloudy Sky, Foliage, Forest, Grass, La...    |
| [`jinaai/jina-vlm-mlx`](#model-jinaai-jina-vlm-mlx)                                                                     | `unusable`            | Title:<br>A Serene Pondside Hideout<br>Description:<br>A rustic wooden boathouse built on stilts overlooks a tranquil pond, surrounded by lush foliage and framed by the overhanging branches of a nearby tree. The wooden deck and railing add to the charm of this peaceful outdoor retreat,...                |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `unusable`            | Here are my reasoning steps:<br>We need to produce British-English catalogue metadata from the image and supplied context. The context includes capture date/time and GPS. The descriptive hints are given. We need to treat capture date/time and GPS as authoritative facts, but not c...                      |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `unusable`            | Alright, let's get to work. I've got an image and some hints to work with, and my task is to create some metadata for it. First things first, I need to break down the request. It's a straightforward task: title, description, and keywords. The image is of a boathouse, and I hav...                         |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `unusable`            | A weathered wooden boathouse stands over a serene pond, surrounded by lush foliage and a cloudy sky, with a concrete 5-10-word title: "A tranquil lakeside scene with a weathered wooden boathouse and overhanging trees."                                                                                       |
| [`mlx-community/GLM-4.1V-9B-Thinking-8bit`](#model-mlx-community-glm-41v-9b-thinking-8bit)                              | `unusable`            | &lt;think&gt;Got it, let's tackle this task step by step. First, the title needs to be 5-10 words, concrete. Looking at the image: weathered wooden boathouse, stilts, pond, trees, etc. Maybe "Weathered Wooden Boathouse by the Pond".<br><br>Next, the description: combine context (date/time...             |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `unusable`            | Title: "Moss-Covered Wooden Boathouse Over Wetland Pond"<br><br>Description: A weathered wooden boathouse stands elevated on stilts over a serene pond, surrounded by lush foliage and marshland. The overhanging tree branches frame the structure, while the cloudy sky casts a soft, dif...                   |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `unusable`            | Weathered wooden boathouse by calm pond under cloudy skies.<br>It stands elevated over the water surrounded by lush greenery and dense foliage.<br><br>Keywords: wooden boathouse, pond, cloudy sky, foliage, marshland, reeds, wetland, trees, water reflection, moss, outdoors, landscape, d...                |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `unusable`            | Title:<br>Wooden Boathouse Over Serene Pond<br><br>Description:<br>A weathered wooden boathouse, elevated on stilts, stands over a tranquil pond, framed by lush greenery and overhanging tree branches, with a cloudy sky and reflections on the water.<br><br>Keywords:<br>Wooden boathouse, stilted, po...    |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `unusable`            | ## Image Description<br><br>A weathered wooden boathouse stands on stilts over a calm pond. The boathouse is constructed from vertical wooden planks and has a simple, pitched roof covered in moss. A wooden walkway with railings leads to the boathouse. Lush green foliage overhangs th...                   |
| [`mlx-community/LFM2.5-VL-3B-OptiQ-4bit`](#model-mlx-community-lfm25-vl-3b-optiq-4bit)                                  | `not_evaluated`       | Model loading failed: Received 600 parameters not in model:<br>model.embed_tokens.biases,<br>model.embed_tokens.scales,<br>model.embed_tokens.weight,<br>model.embedding_norm.weight,<br>model.layers.0.conv.conv.weight,<br>model.layers.0.conv.in_proj.biases,<br>model.layers.0.conv.in_proj.scales,<br>mo... |
| [`tencent/Youtu-VL-4B-Instruct`](#model-tencent-youtu-vl-4b-instruct)                                                   | `not_evaluated`       | Model loading failed: cannot import name 'DefaultFastImageProcessorKwargs' from 'transformers.image_processing_utils_fast' (unknown location)                                                                                                                                                                    |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Run Stamps

- `mlx-vlm`: `0.7.0rc0`
- `mlx`: `0.32.3.dev20260828+99e45f71d`
- `mlx-lm`: `0.32.0`
- `transformers`: `5.16.1`
- `tokenizers`: `0.23.1`
- `huggingface-hub`: `1.29.0`
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

- *Description:* A weathered wooden boathouse built on stilts stands over the
  edge of a serene pond, framed by lush overhanging tree branches, wooden
  decking, and surrounding wetland foliage.
- *Keywords:* Cloudy Sky, Foliage, Forest, Grass, Lake, Landscape, Leaves,
  Marshland, Moss, Outdoors, Pond, Reeds, Trees, Water reflection, Wetland,
  Wooden shed, architecture, bird hide, birdwatching, boardwalk, boathouse,
  cabin, conservation, countryside, ecology, environment, habitat, jetty,
  marsh, mossy roof, nature, nature reserve, observation hide, outdoor,
  overcast, peaceful, pier, platform, reflection, rustic, scenic, secluded,
  serene, stilt house, stilts, swamp, timber, tranquility, water, waterfront,
  weathered wood, wildlife observation, wooden cabin, wooden deck, wooden hut,
  wooden railing, wooden structure
- *Date:* 2026-08-21 14:34:53 UTC+01:00
- *Time:* 14:34:53
- *GPS:* 51.441113°N, 0.565406°W

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
> &#45; Capture date/time: 2026-08-21 14:34:53 UTC+01:00
> &#45; GPS: 51.441113°N, 0.565406°W
>
> &#8203;Descriptive hints:
> &#45; Description hint: A weathered wooden boathouse built on stilts stands over
> the edge of a serene pond, framed by lush overhanging tree branches, wooden
> decking, and surrounding wetland foliage.
> &#45; Keyword hints: Cloudy Sky, Foliage, Forest, Grass, Lake, Landscape,
> Leaves, Marshland, Moss, Outdoors, Pond, Reeds, Trees, Water reflection,
> Wetland, Wooden shed, architecture, bird hide, birdwatching, boardwalk
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

<a id="model-liquidai-lfm25-vl-450m-mlx-bf16"></a>

### LiquidAI/LFM2.5-VL-450M-MLX-bf16

<details>
<summary>Complete evidence: LiquidAI/LFM2.5-VL-450M-MLX-bf16</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type lfm2-vl via lfm2_vl)
- *Model load time:* 0.39s
- *Generation time:* 0.85s
- *Total time:* 1.60s
- *Input validation time:* 0.349
- *Prompt preparation time:* 0.00663
- *First-token latency:* 0.11
- *Cleanup time:* 0.0635
- *Prompt tokens:* 2,117
- *Generation tokens:* 118
- *Total tokens:* 2,235
- *Prompt throughput (raw):* 19,319 tok/s
- *Generation throughput (raw):* 481 tok/s
- *Peak memory:* 1.9
- *Active memory:* 0.90
- *Cache memory:* 0.08
- *Model-load active memory:* 0.897
- *Post-cleanup active memory:* 0.000132
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1360
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
Boathouse on stilts in a wetland

Description:
A weathered wooden boathouse stands on stilts over a serene pond, framed by lush overhanging tree branches and surrounding wetland foliage. The scene is bathed in soft, diffused light under a cloudy sky, with reflections of trees and reeds visible in the still water.

Keywords:
Boathouse, wetland, wooden, stilts, cloudy, overhanging, trees, reeds, pond, reflections, moss, landscape, nature, birdwatching, boardwalk
</pre>

</details>

---

<a id="model-mlx-community-devstral-small-2-24b-instruct-2512-5bit"></a>

### mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

<details>
<summary>Complete evidence: mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type mistral3)
- *Model load time:* 2.44s
- *Generation time:* 8.49s
- *Total time:* 10.95s
- *Input validation time:* 7.25e-05
- *Prompt preparation time:* 0.0131
- *First-token latency:* 3.58
- *Cleanup time:* 0.125
- *Prompt tokens:* 2,401
- *Generation tokens:* 121
- *Total tokens:* 2,522
- *Prompt throughput (raw):* 671 tok/s
- *Generation throughput (raw):* 29.4 tok/s
- *Peak memory:* 23
- *Active memory:* 18
- *Cache memory:* 0.47
- *Model-load active memory:* 18
- *Post-cleanup active memory:* 0.00128
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1309
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
Weathered Boathouse Over Serene Pond

Description:
A weathered wooden boathouse on stilts overlooks a tranquil pond, framed by lush overhanging tree branches and surrounded by wetland foliage, captured on a cloudy afternoon in August 2026.

Keywords:
Boathouse, Pond, Wetland, Wooden Decking, Overhanging Trees, Cloudy Sky, Moss, Reeds, Foliage, Reflection, Boardwalk, Architecture, Bird Hide, Birdwatching, Landscape, Outdoors, Water, Stilted Structure
</pre>

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
- *Model load time:* 1.95s
- *Generation time:* 6.63s
- *Total time:* 8.59s
- *Input validation time:* 5.52e-05
- *Prompt preparation time:* 0.00226
- *First-token latency:* 1.08
- *Cleanup time:* 0.122
- *Prompt tokens:* 2,618
- *Generation tokens:* 164
- *Total tokens:* 2,782
- *Prompt throughput (raw):* 2,414 tok/s
- *Generation throughput (raw):* 34.0 tok/s
- *Peak memory:* 18
- *Active memory:* 17
- *Cache memory:* 0.43
- *Model-load active memory:* 16.9
- *Post-cleanup active memory:* 0.003
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1346
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
 Title: Weathered Wooden Boathouse on Stilts Over a Serene Pond.

Description: A weathered wooden boathouse built on stilts stands over the edge of a serene pond, framed by lush overhanging tree branches, wooden decking, and surrounding wetland foliage. The boathouse's roof is covered in moss, and the wooden structure shows signs of age and wear. The pond's calm surface reflects the cloudy sky above, and the surrounding wetland is filled with reeds and other vegetation. The scene is a tranquil and picturesque example of a natural landscape.

Keywords: weathered wooden boathouse, stilts, serene pond, lush overhanging tree branches, wooden decking, wetland foliage, moss, cloudy sky, reflection, reeds, wetland, natural landscape.
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
- *Model load time:* 1.45s
- *Generation time:* 4.36s
- *Total time:* 5.82s
- *Input validation time:* 5.98e-05
- *Prompt preparation time:* 0.00749
- *First-token latency:* 1.99
- *Cleanup time:* 0.0941
- *Prompt tokens:* 2,934
- *Generation tokens:* 123
- *Total tokens:* 3,057
- *Prompt throughput (raw):* 1,475 tok/s
- *Generation throughput (raw):* 67.6 tok/s
- *Peak memory:* 13
- *Active memory:* 8.0
- *Cache memory:* 0.56
- *Model-load active memory:* 8.01
- *Post-cleanup active memory:* 0.00488
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 3747
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
**"Stilted Boathouse Over a Wetland Pond"**

Description:
A weathered wooden boathouse on stilts extends over a calm pond, surrounded by lush foliage and overhanging tree branches, captured on a cloudy August afternoon in a wetland landscape.

Keywords:
Boathouse, wetland, pond, wooden structure, stilts, foliage, overhanging branches, cloudy sky, marshland, landscape, boardwalk, moss, reeds, water reflection, birdwatching hide, serene nature, August 2026, UK countryside
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
- *Model load time:* 0.95s
- *Generation time:* 2.24s
- *Total time:* 3.21s
- *Input validation time:* 4.7e-05
- *Prompt preparation time:* 0.00756
- *First-token latency:* 0.982
- *Cleanup time:* 0.0828
- *Prompt tokens:* 2,933
- *Generation tokens:* 134
- *Total tokens:* 3,067
- *Prompt throughput (raw):* 2,987 tok/s
- *Generation throughput (raw):* 190 tok/s
- *Peak memory:* 7.8
- *Active memory:* 2.8
- *Cache memory:* 0.38
- *Model-load active memory:* 2.75
- *Post-cleanup active memory:* 0.00541
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 3746
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
Weathered Pondside Wooden Shed in Overgrown Wetland

Description:
A rustic, elevated wooden boathouse or bird hide, constructed on a wooden deck over a shallow pond, is bathed in soft cloudy light amid dense foliage, moss, and reeds. The structure’s weathered planks and cross supports contrast with the tranquil water and lush greenery beyond.

Keywords:
boardwalk, cloudy sky, moss-covered deck, pond edge, reedy wetland, rustic wooden shed, stilts, tree canopy, water reflection, woodland setting, bird hide, architecture, serene lighting, wetland foliage
</pre>

</details>

---

<a id="model-mlx-community-molmo2-8b-4bit"></a>

### mlx-community/Molmo2-8B-4bit

<details>
<summary>Complete evidence: mlx-community/Molmo2-8B-4bit</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type molmo2)
- *Model load time:* 1.07s
- *Generation time:* 4.42s
- *Total time:* 5.50s
- *Input validation time:* 4.23e-05
- *Prompt preparation time:* 0.0056
- *First-token latency:* 1.93
- *Cleanup time:* 0.087
- *Prompt tokens:* 1,529
- *Generation tokens:* 133
- *Total tokens:* 1,662
- *Prompt throughput (raw):* 793 tok/s
- *Generation throughput (raw):* 72.5 tok/s
- *Peak memory:* 8.8
- *Active memory:* 6.5
- *Cache memory:* 0.31
- *Model-load active memory:* 6.45
- *Post-cleanup active memory:* 0.00572
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1347
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
Title: Stilted Wooden Hide in Marshy Wetland

Description: A weathered wooden structure on stilts overlooks a serene pond, surrounded by lush foliage and reeds. Built for birdwatching, it features a wooden deck and is partially obscured by overhanging branches, with a cloudy sky visible through the trees.

Keywords: Stilted building, Marshland, Wetland, Pond, Reeds, Wooden deck, Bird hide, Birdwatching, Cloudy sky, Foliage, Trees, Water reflection, Wetland architecture, Outdoors, Marshy landscape, Stilted structure, Marshy pond, Stilted hide
</pre>

</details>

---

<a id="model-mlx-community-qwen3-vl-30b-a3b-instruct-4bit"></a>

### mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit

<details>
<summary>Complete evidence: mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl_moe)
- *Model load time:* 1.90s
- *Generation time:* 57.62s
- *Total time:* 59.53s
- *Input validation time:* 5.47e-05
- *Prompt preparation time:* 0.00149
- *First-token latency:* 55.1
- *Cleanup time:* 0.129
- *Prompt tokens:* 16,552
- *Generation tokens:* 129
- *Total tokens:* 16,681
- *Prompt throughput (raw):* 300 tok/s
- *Generation throughput (raw):* 87.1 tok/s
- *Peak memory:* 23
- *Active memory:* 18
- *Cache memory:* 2.3
- *Model-load active memory:* 18.3
- *Post-cleanup active memory:* 0.00744
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1381
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
Title: Weathered wooden boathouse on a calm pond
Description: A weathered wooden boathouse stands on stilts over a calm pond, its dark timber walls and moss-covered roof contrasting with the lush green foliage and overhanging tree branches. The scene, captured on a cloudy day, features a wooden deck and railings, with reeds and water reflections visible in the foreground.
Keywords: wooden boathouse, pond, wetland, reeds, trees, foliage, moss, wooden deck, stilts, cloudy sky, water reflection, landscape, nature, outdoors, architecture, bird hide, marshland, grass
</pre>

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
- *Model load time:* 2.71s
- *Generation time:* 53.50s
- *Total time:* 56.24s
- *Input validation time:* 7.1e-05
- *Prompt preparation time:* 0.0159
- *First-token latency:* 51.5
- *Cleanup time:* 0.126
- *Prompt tokens:* 16,568
- *Generation tokens:* 111
- *Total tokens:* 16,679
- *Prompt throughput (raw):* 321 tok/s
- *Generation throughput (raw):* 110 tok/s
- *Peak memory:* 24
- *Active memory:* 20
- *Cache memory:* 0.89
- *Model-load active memory:* 20.4
- *Post-cleanup active memory:* 0.00795
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1400
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
Title: Weathered Wooden Boathouse Over Pond

Description: Captured on 21 August 2026, this weathered wooden boathouse stands on stilts over a calm pond, framed by lush overhanging tree branches and surrounded by wetland foliage under a cloudy sky.

Keywords: boathouse, pond, wooden structure, wetland, foliage, cloudy sky, trees, reeds, water reflection, boardwalk, moss, nature, architecture, bird hide, landscape, outdoors, greenery, stillness
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
- *Model load time:* 2.28s
- *Generation time:* 6.48s
- *Total time:* 8.77s
- *Input validation time:* 6.99e-05
- *Prompt preparation time:* 0.00425
- *First-token latency:* 1.03
- *Cleanup time:* 0.132
- *Prompt tokens:* 593
- *Generation tokens:* 153
- *Total tokens:* 746
- *Prompt throughput (raw):* 577 tok/s
- *Generation throughput (raw):* 31.3 tok/s
- *Peak memory:* 17
- *Active memory:* 16
- *Cache memory:* 0.45
- *Model-load active memory:* 16.1
- *Post-cleanup active memory:* 0.0107
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1364
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
Title: Boathouse on Stilts, Wetland Landscape

Description: Captured on 21st August 2026 at 14:34 UTC+01, this weathered wooden boathouse stands on stilts above a pond at 51.441113°N, 0.565406°W, with a cloudy sky and reflected light on the water. The structure is framed by lush foliage and accessed via a wooden deck and boardwalk.

Keywords: Boathouse, Wooden structure, Stilts, Pond, Wetland, Marshland, Trees, Foliage, Water reflection, Cloudy Sky, Wooden decking, Boardwalk, Moss, Architecture, Exterior, Landscape, Reeds, Grass
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
- *Model load time:* 2.38s
- *Generation time:* 1.77s
- *Total time:* 4.18s
- *Input validation time:* 5.57e-05
- *Prompt preparation time:* 0.0229
- *First-token latency:* 0.409
- *Cleanup time:* 0.123
- *Prompt tokens:* 598
- *Generation tokens:* 100
- *Total tokens:* 698
- *Prompt throughput (raw):* 1,462 tok/s
- *Generation throughput (raw):* 129 tok/s
- *Peak memory:* 16
- *Active memory:* 15
- *Cache memory:* 0.20
- *Model-load active memory:* 15.4
- *Post-cleanup active memory:* 0.0118
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1363
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
Title: Weathered wooden boathouse over a serene pond
Description: A rustic wooden boathouse built on stilts stands over a calm pond, framed by lush overhanging tree branches and wetland foliage under a cloudy sky. The structure features a wooden deck and reflects the surrounding greenery in the still water.
Keywords: architecture, bird hide, birdwatching, boardwalk, foliage, forest, lake, marshland, moss, pond, reeds, trees, water reflection, wetland, wooden shed
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
- *Generation time:* 5.33s
- *Total time:* 7.88s
- *Input validation time:* 5.55e-05
- *Prompt preparation time:* 0.00185
- *First-token latency:* 1.13
- *Cleanup time:* 0.142
- *Prompt tokens:* 598
- *Generation tokens:* 95
- *Total tokens:* 693
- *Prompt throughput (raw):* 529 tok/s
- *Generation throughput (raw):* 26.4 tok/s
- *Peak memory:* 20
- *Active memory:* 18
- *Cache memory:* 0.78
- *Model-load active memory:* 18.4
- *Post-cleanup active memory:* 0.0123
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1363
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
Title: Weathered Wooden Bird Hide Over a Serene Pond
Description: A weathered wooden bird hide on stilts extends over a pond, framed by lush overhanging tree branches under a cloudy sky. The structure is accessed via a wooden boardwalk and surrounded by wetland foliage.
Keywords: bird hide, birdwatching, wooden shed, pond, wetland, boardwalk, foliage, trees, water reflection, marshland, reeds, cloudy sky, architecture, landscape, outdoors, nature
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
- *Generation time:* 2.20s
- *Total time:* 2.88s
- *Input validation time:* 7.33e-05
- *Prompt preparation time:* 0.0101
- *First-token latency:* 0.916
- *Cleanup time:* 0.0886
- *Prompt tokens:* 1,383
- *Generation tokens:* 85
- *Total tokens:* 1,468
- *Prompt throughput (raw):* 1,511 tok/s
- *Generation throughput (raw):* 171 tok/s
- *Peak memory:* 4.6
- *Active memory:* 3.0
- *Cache memory:* 0.23
- *Model-load active memory:* 3.03
- *Post-cleanup active memory:* 0.0125
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1533
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
Title: "Weathered Wooden Boathouse by a Serene Pond"
Description: A weathered wooden boathouse stands on stilts over a calm pond, surrounded by lush foliage and a serene wetland landscape.
Keywords: Weathered Boathouse, Pond, Wetland, Lush Foliage, Marshland, Bird Hide, Birdwatching, Boardwalk, Moss, Reeds, Architecture
</pre>

</details>

---

<a id="model-qwen-qwen3-vl-2b-instruct"></a>

### Qwen/Qwen3-VL-2B-Instruct

<details>
<summary>Complete evidence: Qwen/Qwen3-VL-2B-Instruct</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Title has 4 words (requested 5-10); Keyword list has 24
  terms (requested 10-18); Duplicate keywords: pond, trees
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Model load time:* 0.66s
- *Generation time:* 15.29s
- *Total time:* 15.97s
- *Input validation time:* 4.6e-05
- *Prompt preparation time:* 0.00979
- *First-token latency:* 13.2
- *Cleanup time:* 0.0841
- *Prompt tokens:* 16,552
- *Generation tokens:* 112
- *Total tokens:* 16,664
- *Prompt throughput (raw):* 1,249 tok/s
- *Generation throughput (raw):* 93.4 tok/s
- *Peak memory:* 8.4
- *Active memory:* 4.3
- *Cache memory:* 2.6
- *Model-load active memory:* 4.26
- *Post-cleanup active memory:* 0.000443
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1381
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
Title: Wooden boathouse by pond

Description: A weathered wooden boathouse on stilts stands over a calm pond, surrounded by lush green foliage and reeds under a cloudy sky, with a wooden deck and moss-covered roof.

Keywords: boathouse, pond, wooden shed, stilts, foliage, reeds, moss, cloudy sky, wetland, lake, trees, water reflection, outdoor, birdwatching, architecture, bird hide, landscape, grass, marshland, wooden decking, leaves, pond, trees, water
</pre>

</details>

---

<a id="model-mlx-community-glm-46v-flash-mxfp4"></a>

### mlx-community/GLM-4.6V-Flash-mxfp4

<details>
<summary>Complete evidence: mlx-community/GLM-4.6V-Flash-mxfp4</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Keyword list has 25 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v)
- *Model load time:* 1.34s
- *Generation time:* 8.37s
- *Total time:* 9.72s
- *Input validation time:* 6.9e-05
- *Prompt preparation time:* 0.00825
- *First-token latency:* 6.23
- *Cleanup time:* 0.1
- *Prompt tokens:* 6,392
- *Generation tokens:* 126
- *Total tokens:* 6,518
- *Prompt throughput (raw):* 1,026 tok/s
- *Generation throughput (raw):* 79.6 tok/s
- *Peak memory:* 8.4
- *Active memory:* 6.8
- *Cache memory:* 0.46
- *Model-load active memory:* 6.78
- *Post-cleanup active memory:* 0.00243
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1390
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
Title: Weathered Boathouse by Serene Pond
Description: A weathered wooden boathouse built on stilts stands over the edge of a serene pond, framed by lush overhanging tree branches, wooden decking, and surrounding wetland foliage.
Keywords: weathered boathouse, serene pond, overhanging tree branches, wooden decking, wetland foliage, cloudy sky, foliage, forest, grass, lake, landscape, leaves, marshland, moss, outdoors, pond, reeds, trees, water reflection, wetland, wooden shed, architecture, bird hide, birdwatching, boardwalk
</pre>

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
- *Model load time:* 8.25s
- *Generation time:* 20.97s
- *Total time:* 29.24s
- *Input validation time:* 7.58e-05
- *Prompt preparation time:* 0.00882
- *First-token latency:* 18
- *Cleanup time:* 0.266
- *Prompt tokens:* 6,392
- *Generation tokens:* 95
- *Total tokens:* 6,487
- *Prompt throughput (raw):* 355 tok/s
- *Generation throughput (raw):* 41.5 tok/s
- *Peak memory:* 78
- *Active memory:* 62
- *Cache memory:* 1.4
- *Model-load active memory:* 61.9
- *Post-cleanup active memory:* 0.00274
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1390
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
&lt;|begin_of_box|&gt;Title: Weathered Wooden Boathouse Over Pond
Description: A weathered wooden boathouse on stilts stands over a serene pond, framed by lush tree branches, wooden decking, and surrounding wetland foliage, under a cloudy sky.
Keywords: boathouse, wooden, stilts, pond, cloudy sky, foliage, trees, wetland, marshland, reeds, water reflection, boardwalk, bird hide, architecture, outdoors, landscape&lt;|end_of_box|&gt;
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|begin_of_box|>Title: Weathered Wooden Boathouse Over Pond
Description: A weathered wooden boathouse on stilts stands over a serene pond, framed by lush tree branches, wooden decking, and surrounding wetland foliage, under a cloudy sky.
Keywords: boathouse, wooden, stilts, pond, cloudy sky, foliage, trees, wetland, marshland, reeds, water reflection, boardwalk, bird hide, architecture, outdoors, landscape<|end_of_box|>
```

</details>

</details>

---

<a id="model-mlx-community-internvl3-8b-bf16"></a>

### mlx-community/InternVL3-8B-bf16

<details>
<summary>Complete evidence: mlx-community/InternVL3-8B-bf16</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Title has 4 words (requested 5-10)
- *Arch supported by installed mlx-vlm:* yes (model_type internvl_chat)
- *Model load time:* 1.64s
- *Generation time:* 4.65s
- *Total time:* 6.30s
- *Input validation time:* 7.1e-05
- *Prompt preparation time:* 0.00302
- *First-token latency:* 1.3
- *Cleanup time:* 0.106
- *Prompt tokens:* 2,118
- *Generation tokens:* 99
- *Total tokens:* 2,217
- *Prompt throughput (raw):* 1,625 tok/s
- *Generation throughput (raw):* 36.5 tok/s
- *Peak memory:* 17
- *Active memory:* 16
- *Cache memory:* 0.17
- *Model-load active memory:* 15.9
- *Post-cleanup active memory:* 0.00331
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1314
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
Title: Stilted Boathouse Over Pond
Description: A weathered wooden boathouse on stilts overlooks a tranquil pond, surrounded by lush foliage and under a cloudy sky. The scene captures serene outdoor architecture amidst a wetland setting.
Keywords: Wooden boathouse, Stilts, Pond, Lush foliage, Cloudy sky, Wetland, Reeds, Water reflection, Architecture, Outdoors, Boardwalk, Trees, Leaves, Grass, Lake, Landscape
</pre>

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
- *Model load time:* 3.29s
- *Generation time:* 146.94s
- *Total time:* 150.24s
- *Input validation time:* 6.91e-05
- *Prompt preparation time:* 0.0037
- *First-token latency:* 1.84
- *Cleanup time:* 0.187
- *Prompt tokens:* 1,329
- *Generation tokens:* 682
- *Total tokens:* 2,011
- *Prompt throughput (raw):* 724 tok/s
- *Generation throughput (raw):* 4.73 tok/s
- *Peak memory:* 40
- *Active memory:* 33
- *Cache memory:* 5.6
- *Model-load active memory:* 32.8
- *Post-cleanup active memory:* 0.00397
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1496
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
◁think▷Okay, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. The user wants a 5-10 word title, a 1-2 sentence description, and 10-18 keywords.

Starting with the title: The image shows a weathered wooden boathouse on stilts over a pond, surrounded by trees and foliage. The context mentions a serene pond, so maybe something like "Weathered Wooden Boathouse Over Serene Pond" but that's a bit long. Maybe shorten to "Weathered wooden boathouse over serene pond". That's 7 words.

Next, the description: Need to combine context and visible details. The boathouse is weathered, on stilts, over a pond, with lush foliage, trees, wooden decking, wetland. So something like "A weathered wooden boathouse on stilts overlooks a serene pond, surrounded by lush foliage and wetland vegetation, with a wooden deck and overhanging trees framing the scene under a cloudy sky." That's a bit long, maybe trim to "A weathered wooden boathouse on stilts overlooks a serene pond, surrounded by lush foliage and wetland vegetation, with a wooden deck and overhanging trees under a cloudy sky." That's 2 sentences.

Keywords: Need to pick 10-18 unique ones. From the hints: Cloudy Sky, Foliage, Forest, Grass, Lake, Landscape, Leaves, Marshland, Moss, Outdoors, Pond, Water reflection, Wetland, Wooden shed, architecture, bird hide, birdwatching, boardwalk. Let's pick the most relevant: Cloudy Sky, Foliage, Forest, Grass, Lake, Landscape, Leaves, Marshland, Pond, Wetland, Wooden shed, Boardwalk, Water reflection, Bird hide, Birdwatching, Landscape. Wait, maybe check the image: the boathouse is a bird hide, so include that. Also, the decking is visible. So maybe: Cloudy Sky, Foliage, Forest, Grass, Lake, Landscape, Leaves, Marshland, Pond, Wetland, Wooden shed, Boardwalk, Water reflection, Bird hide, Birdwatching, Landscape. That's 14 keywords. Let me check if they are all visible or inferred. The image has a cloudy sky, foliage, forest, grass, lake, landscape, leaves, marshland, pond, wetland, wooden shed, boardwalk, water reflection, bird hide, birdwatching, and landscape again. Maybe remove one "Landscape" to avoid repetition. So: Cloudy Sky, Foliage, Forest, Grass, Lake, Landscape, Leaves, Marshland, Pond, Wetland, Wooden shed, Boardwalk, Water reflection, Bird hide, Birdwatching. That's 14.◁/think▷Title: Weathered wooden boathouse over serene pond
Description: A weathered wooden boathouse on stilts overlooks a serene pond, surrounded by lush foliage and wetland vegetation, with a wooden deck and overhanging trees under a cloudy sky.
Keywords: Cloudy Sky, Foliage, Forest, Grass, Lake, Landscape, Leaves, Marshland, Pond, Wetland, Wooden shed, Boardwalk, Water reflection, Bird hide, Birdwatching&lt;|im_assistant|&gt;
</pre>

<details>
<summary>Exact raw output</summary>

```text
◁think▷Okay, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. The user wants a 5-10 word title, a 1-2 sentence description, and 10-18 keywords. 

Starting with the title: The image shows a weathered wooden boathouse on stilts over a pond, surrounded by trees and foliage. The context mentions a serene pond, so maybe something like "Weathered Wooden Boathouse Over Serene Pond" but that's a bit long. Maybe shorten to "Weathered wooden boathouse over serene pond". That's 7 words.

Next, the description: Need to combine context and visible details. The boathouse is weathered, on stilts, over a pond, with lush foliage, trees, wooden decking, wetland. So something like "A weathered wooden boathouse on stilts overlooks a serene pond, surrounded by lush foliage and wetland vegetation, with a wooden deck and overhanging trees framing the scene under a cloudy sky." That's a bit long, maybe trim to "A weathered wooden boathouse on stilts overlooks a serene pond, surrounded by lush foliage and wetland vegetation, with a wooden deck and overhanging trees under a cloudy sky." That's 2 sentences.

Keywords: Need to pick 10-18 unique ones. From the hints: Cloudy Sky, Foliage, Forest, Grass, Lake, Landscape, Leaves, Marshland, Moss, Outdoors, Pond, Water reflection, Wetland, Wooden shed, architecture, bird hide, birdwatching, boardwalk. Let's pick the most relevant: Cloudy Sky, Foliage, Forest, Grass, Lake, Landscape, Leaves, Marshland, Pond, Wetland, Wooden shed, Boardwalk, Water reflection, Bird hide, Birdwatching, Landscape. Wait, maybe check the image: the boathouse is a bird hide, so include that. Also, the decking is visible. So maybe: Cloudy Sky, Foliage, Forest, Grass, Lake, Landscape, Leaves, Marshland, Pond, Wetland, Wooden shed, Boardwalk, Water reflection, Bird hide, Birdwatching, Landscape. That's 14 keywords. Let me check if they are all visible or inferred. The image has a cloudy sky, foliage, forest, grass, lake, landscape, leaves, marshland, pond, wetland, wooden shed, boardwalk, water reflection, bird hide, birdwatching, and landscape again. Maybe remove one "Landscape" to avoid repetition. So: Cloudy Sky, Foliage, Forest, Grass, Lake, Landscape, Leaves, Marshland, Pond, Wetland, Wooden shed, Boardwalk, Water reflection, Bird hide, Birdwatching. That's 14.◁/think▷Title: Weathered wooden boathouse over serene pond  
Description: A weathered wooden boathouse on stilts overlooks a serene pond, surrounded by lush foliage and wetland vegetation, with a wooden deck and overhanging trees under a cloudy sky.  
Keywords: Cloudy Sky, Foliage, Forest, Grass, Lake, Landscape, Leaves, Marshland, Pond, Wetland, Wooden shed, Boardwalk, Water reflection, Bird hide, Birdwatching<|im_assistant|>
```

</details>

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
- *Model load time:* 1.40s
- *Generation time:* 5.04s
- *Total time:* 6.45s
- *Input validation time:* 5.52e-05
- *Prompt preparation time:* 0.00095
- *First-token latency:* 2.01
- *Cleanup time:* 0.0956
- *Prompt tokens:* 2,934
- *Generation tokens:* 161
- *Total tokens:* 3,095
- *Prompt throughput (raw):* 1,458 tok/s
- *Generation throughput (raw):* 64.8 tok/s
- *Peak memory:* 13
- *Active memory:* 8.4
- *Cache memory:* 0.60
- *Model-load active memory:* 8.43
- *Post-cleanup active memory:* 0.00515
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 3747
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
*"Weathered Boathouse Over Serene Wetland Pond"*

**Description:**
A weathered wooden boathouse, built on stilts, stands over a calm pond surrounded by lush foliage, reeds, and overhanging branches. Captured on **21 August 2026** under a cloudy sky, its wooden decking and reflections highlight its rustic charm and natural setting.

**Keywords:**
Boathouse, Stilts, Wetland, Pond, Lush Foliage, Reeds, Overhanging Branches, Wooden Decking, Cloudy Sky, Rustic Architecture, Reflection, Forest Landscape, Marshland, Moss, Birdwatching Hide, Stilts Structure, Natural Lighting, Serene Waterscape, UK Landscape
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
- *Observations:* Keyword list has 24 terms (requested 10-18); Duplicate
  keywords: boathouse
- *Arch supported by installed mlx-vlm:* yes (model_type cohere_compass)
- *Model load time:* 1.00s
- *Generation time:* 4.60s
- *Total time:* 5.61s
- *Input validation time:* 9.96e-05
- *Prompt preparation time:* 0.00514
- *First-token latency:* 3.1
- *Cleanup time:* 0.0857
- *Prompt tokens:* 4,085
- *Generation tokens:* 161
- *Total tokens:* 4,246
- *Prompt throughput (raw):* 1,318 tok/s
- *Generation throughput (raw):* 230 tok/s
- *Peak memory:* 3.9
- *Active memory:* 2.2
- *Cache memory:* 0.64
- *Model-load active memory:* 2.18
- *Post-cleanup active memory:* 0.00624
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1440
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
Title: Boathouse in the Wetland Landscape

Description: A weathered wooden boathouse stands over the edge of a serene pond, framed by lush overhanging tree branches. The structure features wooden decking and a rustic wooden shed, with a wooden railing leading up to it. The surrounding area is filled with wetland foliage, reeds, and a calm body of water reflecting the cloudy sky. The scene is tranquil, with the boathouse serving as a focal point in the natural landscape.

Keywords: Boathouse, Wetland Landscape, Foliage, Reeds, Water, Cloudy Sky, Outdoor, Pond, Architecture, Birdwatching, Forest, Moss, Leaves, Grass, Reflection, Nature, Wooden structure, Boathouse, Landscape, Wildlife, Serenity, Natural setting, Outdoor scene, Wetland environment.
</pre>

</details>

---

<a id="model-mlx-community-ornith-10-35b-bf16"></a>

### mlx-community/Ornith-1.0-35B-bf16

<details>
<summary>Complete evidence: mlx-community/Ornith-1.0-35B-bf16</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Keyword list has 20 terms (requested 10-18); Title,
  Description and Keywords copy all supplied hints unchanged
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5_moe)
- *Model load time:* 9.86s
- *Generation time:* 56.56s
- *Total time:* 66.45s
- *Input validation time:* 6.7e-05
- *Prompt preparation time:* 0.0193
- *First-token latency:* 53.6
- *Cleanup time:* 0.272
- *Prompt tokens:* 16,568
- *Generation tokens:* 110
- *Total tokens:* 16,678
- *Prompt throughput (raw):* 309 tok/s
- *Generation throughput (raw):* 64.2 tok/s
- *Peak memory:* 74
- *Active memory:* 70
- *Cache memory:* 0.89
- *Model-load active memory:* 70.2
- *Post-cleanup active memory:* 0.00675
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1400
- *Image placeholders:* 0
- *Processed image:* not captured
- *Image patch count:* not captured
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 9ef631ad2d0c4c26783d4f94d0a0de9516e41a4b
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Ornith-1.0-35B-bf16/snapshots/9ef631ad2d0c4c26783d4f94d0a0de9516e41a4b
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Wooden boathouse on stilts over a pond
Description: A weathered wooden boathouse built on stilts stands over the edge of a serene pond, framed by lush overhanging tree branches, wooden decking, and surrounding wetland foliage.
Keywords: Cloudy Sky, Foliage, Forest, Grass, Lake, Landscape, Leaves, Marshland, Moss, Outdoors, Pond, Reeds, Trees, Water reflection, Wetland, Wooden shed, architecture, bird hide, birdwatching, boardwalk
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
- *Observations:* Keyword list has 20 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type phi3_v)
- *Model load time:* 1.05s
- *Generation time:* 3.35s
- *Total time:* 4.40s
- *Input validation time:* 7.61e-05
- *Prompt preparation time:* 0.00172
- *First-token latency:* 0.425
- *Cleanup time:* 0.0887
- *Prompt tokens:* 1,138
- *Generation tokens:* 135
- *Total tokens:* 1,273
- *Prompt throughput (raw):* 2,676 tok/s
- *Generation throughput (raw):* 56.3 tok/s
- *Peak memory:* 9.4
- *Active memory:* 8.3
- *Cache memory:* 0.52
- *Model-load active memory:* 8.3
- *Post-cleanup active memory:* 0.00682
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1330
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
Title: Serene Wooden Boathouse Over Pond

Description: A weathered wooden boathouse stands majestically over a tranquil pond, surrounded by lush greenery and a cloudy sky, offering a peaceful retreat for nature enthusiasts.

Keywords: wooden boathouse, pond, serene, lush, greenery, cloudy sky, tranquil, nature enthusiasts, weathered, architecture, wetland, foliage, overhanging branches, wooden decking, water reflection, reeds, marshland, moss, outdoors, bird hide
</pre>

</details>

---

<a id="model-mlx-community-qwen3-vl-2b-thinking-bf16"></a>

### mlx-community/Qwen3-VL-2B-Thinking-bf16

<details>
<summary>Complete evidence: mlx-community/Qwen3-VL-2B-Thinking-bf16</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Keyword list has 20 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Model load time:* 0.71s
- *Generation time:* 26.97s
- *Total time:* 27.70s
- *Input validation time:* 6.27e-05
- *Prompt preparation time:* 0.012
- *First-token latency:* 15.9
- *Cleanup time:* 0.0953
- *Prompt tokens:* 16,554
- *Generation tokens:* 913
- *Total tokens:* 17,467
- *Prompt throughput (raw):* 1,042 tok/s
- *Generation throughput (raw):* 90.3 tok/s
- *Peak memory:* 8.4
- *Active memory:* 4.3
- *Cache memory:* 2.7
- *Model-load active memory:* 4.26
- *Post-cleanup active memory:* 0.00713
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1389
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
Got it, let's tackle this step by step. First, the title needs to be 5-10 words. The image shows a weathered wooden boathouse on stilts over a pond, with trees and wetland. So maybe "Weathered Stilt Boathouse by Pond with Mossy Roof" – that's 9 words, fits.

Next, the description. It has to combine context and visible details. Context is capture date/time and GPS, but we don't claim them visible. The main subject is the boathouse. Setting: over pond, lush trees, wetland. Action: standing, serene. Lighting: cloudy sky. Distinctive details: wooden decking, moss on roof, reeds, water reflection. So: "Weathered wooden boathouse on stilts over calm pond, framed by mossy roof and lush trees, with wooden deck and reeds lining the water, under cloudy sky."

Then keywords. Need 10-18 unique, comma-separated. From context: Cloudy Sky, Foliage, Forest, Grass, Lake, Landscape, Leaves, Marshland, Moss, Outdoors, Pond, Reeds, Trees, Water reflection, Wetland, Wooden shed, architecture, bird hide, birdwatching, boardwalk. Wait, the descriptive hints have some, but we need to pick relevant ones. Let's check the image: boathouse (wooden shed), pond, moss on roof, reeds, trees, water reflection, wetland, cloudy sky, foliage. So list: Cloudy Sky, Foliage, Forest, Grass, Lake, Landscape, Leaves, Marshland, Moss, Outdoors, Pond, Reeds, Trees, Water reflection, Wetland, Wooden shed, architecture, bird hide, birdwatching, boardwalk. Wait, but the descriptive hints include "Cloudy Sky, Foliage, Forest, Grass, Lake, Landscape, Leaves, Marshland, Moss, Outdoors, Pond, Reeds, Trees, Water reflection, Wetland, Wooden shed, architecture, bird hide, birdwatching, boardwalk". But we need to pick 10-18 unique. Let's check the image: the boathouse is wooden, so "Wooden shed" is there. The pond is a lake? Wait, the context says "Lake" but the image is a pond. Wait, the descriptive hints have "Lake" but the image is a pond. Wait, the problem says "Descriptive hints" include "Lake" but the image is a pond. Wait, the image shows a pond, so maybe "Pond" is correct. Let's list the keywords:

Cloudy Sky, Foliage, Forest, Grass, Lake, Landscape, Leaves, Marshland, Moss, Outdoors, Pond, Reeds, Trees, Water reflection, Wetland, Wooden shed, architecture, bird hide, birdwatching, boardwalk. Wait, but "boardwalk" is in the descriptive hints? The image has a wooden deck, which is a boardwalk. So "boardwalk" is a keyword. Let's check the descriptive hints: "boardwalk" is listed. So the keywords should include those. Let's count: 1. Cloudy Sky, 2. Foliage, 3. Forest, 4. Grass, 5. Lake, 6. Landscape, 7. Leaves, 8. Marshland, 9. Moss, 10. Outdoors, 11. Pond, 12. Reeds, 13. Trees, 14. Water reflection, 15. Wetland, 16. Wooden shed, 17. architecture, 18. bird hide, 19. birdwatching, 20. boardwalk. Wait, but the problem says 10-1
&lt;/think&gt;

Title: Weathered Stilt Boathouse by Pond with Mossy Roof
Description: Weathered wooden boathouse on stilts over calm pond, framed by mossy roof and lush trees, with wooden deck and reeds lining water, under cloudy sky.
Keywords: Cloudy Sky, Foliage, Forest, Grass, Lake, Landscape, Leaves, Marshland, Moss, Outdoors, Pond, Reeds, Trees, Water reflection, Wetland, Wooden shed, architecture, bird hide, birdwatching, boardwalk
</pre>

<details>
<summary>Exact raw output</summary>

```text
Got it, let's tackle this step by step. First, the title needs to be 5-10 words. The image shows a weathered wooden boathouse on stilts over a pond, with trees and wetland. So maybe "Weathered Stilt Boathouse by Pond with Mossy Roof" – that's 9 words, fits.

Next, the description. It has to combine context and visible details. Context is capture date/time and GPS, but we don't claim them visible. The main subject is the boathouse. Setting: over pond, lush trees, wetland. Action: standing, serene. Lighting: cloudy sky. Distinctive details: wooden decking, moss on roof, reeds, water reflection. So: "Weathered wooden boathouse on stilts over calm pond, framed by mossy roof and lush trees, with wooden deck and reeds lining the water, under cloudy sky."

Then keywords. Need 10-18 unique, comma-separated. From context: Cloudy Sky, Foliage, Forest, Grass, Lake, Landscape, Leaves, Marshland, Moss, Outdoors, Pond, Reeds, Trees, Water reflection, Wetland, Wooden shed, architecture, bird hide, birdwatching, boardwalk. Wait, the descriptive hints have some, but we need to pick relevant ones. Let's check the image: boathouse (wooden shed), pond, moss on roof, reeds, trees, water reflection, wetland, cloudy sky, foliage. So list: Cloudy Sky, Foliage, Forest, Grass, Lake, Landscape, Leaves, Marshland, Moss, Outdoors, Pond, Reeds, Trees, Water reflection, Wetland, Wooden shed, architecture, bird hide, birdwatching, boardwalk. Wait, but the descriptive hints include "Cloudy Sky, Foliage, Forest, Grass, Lake, Landscape, Leaves, Marshland, Moss, Outdoors, Pond, Reeds, Trees, Water reflection, Wetland, Wooden shed, architecture, bird hide, birdwatching, boardwalk". But we need to pick 10-18 unique. Let's check the image: the boathouse is wooden, so "Wooden shed" is there. The pond is a lake? Wait, the context says "Lake" but the image is a pond. Wait, the descriptive hints have "Lake" but the image is a pond. Wait, the problem says "Descriptive hints" include "Lake" but the image is a pond. Wait, the image shows a pond, so maybe "Pond" is correct. Let's list the keywords:

Cloudy Sky, Foliage, Forest, Grass, Lake, Landscape, Leaves, Marshland, Moss, Outdoors, Pond, Reeds, Trees, Water reflection, Wetland, Wooden shed, architecture, bird hide, birdwatching, boardwalk. Wait, but "boardwalk" is in the descriptive hints? The image has a wooden deck, which is a boardwalk. So "boardwalk" is a keyword. Let's check the descriptive hints: "boardwalk" is listed. So the keywords should include those. Let's count: 1. Cloudy Sky, 2. Foliage, 3. Forest, 4. Grass, 5. Lake, 6. Landscape, 7. Leaves, 8. Marshland, 9. Moss, 10. Outdoors, 11. Pond, 12. Reeds, 13. Trees, 14. Water reflection, 15. Wetland, 16. Wooden shed, 17. architecture, 18. bird hide, 19. birdwatching, 20. boardwalk. Wait, but the problem says 10-1
</think>

Title: Weathered Stilt Boathouse by Pond with Mossy Roof
Description: Weathered wooden boathouse on stilts over calm pond, framed by mossy roof and lush trees, with wooden deck and reeds lining water, under cloudy sky.
Keywords: Cloudy Sky, Foliage, Forest, Grass, Lake, Landscape, Leaves, Marshland, Moss, Outdoors, Pond, Reeds, Trees, Water reflection, Wetland, Wooden shed, architecture, bird hide, birdwatching, boardwalk
```

</details>

</details>

---

<a id="model-mlx-community-qwen35-9b-mlx-4bit"></a>

### mlx-community/Qwen3.5-9B-MLX-4bit

<details>
<summary>Complete evidence: mlx-community/Qwen3.5-9B-MLX-4bit</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Keyword list has 20 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5)
- *Model load time:* 1.38s
- *Generation time:* 64.01s
- *Total time:* 65.40s
- *Input validation time:* 6.43e-05
- *Prompt preparation time:* 0.00113
- *First-token latency:* 61.5
- *Cleanup time:* 0.106
- *Prompt tokens:* 16,568
- *Generation tokens:* 139
- *Total tokens:* 16,707
- *Prompt throughput (raw):* 270 tok/s
- *Generation throughput (raw):* 89.0 tok/s
- *Peak memory:* 10.0
- *Active memory:* 6.0
- *Cache memory:* 1.2
- *Model-load active memory:* 5.96
- *Post-cleanup active memory:* 0.00847
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1400
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
Weathered Wooden Bird Hide Over Serene Pond in Wetland

Description:
A weathered wooden bird hide stands on stilts over a calm pond, surrounded by lush greenery and overhanging trees under a cloudy sky. The structure features a wooden deck with railings and vertical planks, with visible moss on the roof and faint engraved text on the side.

Keywords:
Bird hide, wooden structure, pond, wetland, stilts, overhanging trees, cloudy sky, wooden deck, railings, moss, reeds, foliage, water reflection, forest, landscape, birdwatching, architecture, outdoors, UK, 2026
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title:  
Weathered Wooden Bird Hide Over Serene Pond in Wetland

Description:  
A weathered wooden bird hide stands on stilts over a calm pond, surrounded by lush greenery and overhanging trees under a cloudy sky. The structure features a wooden deck with railings and vertical planks, with visible moss on the roof and faint engraved text on the side.

Keywords:  
Bird hide, wooden structure, pond, wetland, stilts, overhanging trees, cloudy sky, wooden deck, railings, moss, reeds, foliage, water reflection, forest, landscape, birdwatching, architecture, outdoors, UK, 2026
```

</details>

</details>

---

<a id="model-mlx-community-qwen36-27b-mxfp8"></a>

### mlx-community/Qwen3.6-27B-mxfp8

<details>
<summary>Complete evidence: mlx-community/Qwen3.6-27B-mxfp8</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Keyword list has 20 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5)
- *Model load time:* 3.34s
- *Generation time:* 94.84s
- *Total time:* 98.20s
- *Input validation time:* 8.95e-05
- *Prompt preparation time:* 0.0174
- *First-token latency:* 85.8
- *Cleanup time:* 0.16
- *Prompt tokens:* 16,568
- *Generation tokens:* 135
- *Total tokens:* 16,703
- *Prompt throughput (raw):* 193 tok/s
- *Generation throughput (raw):* 17.2 tok/s
- *Peak memory:* 33
- *Active memory:* 29
- *Cache memory:* 1.9
- *Model-load active memory:* 28.7
- *Post-cleanup active memory:* 0.00898
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1400
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
Wooden Boathouse on Stilts Over a Pond

Description:
A weathered wooden boathouse built on stilts stands over the edge of a serene pond, framed by lush overhanging tree branches, wooden decking, and surrounding wetland foliage. The scene is set under a cloudy sky, with water reflections and reeds adding to the tranquil atmosphere.

Keywords:
Cloudy Sky, Foliage, Forest, Grass, Lake, Landscape, Leaves, Marshland, Moss, Outdoors, Pond, Reeds, Trees, Water reflection, Wetland, Wooden shed, architecture, bird hide, birdwatching, boardwalk
</pre>

</details>

---

<a id="model-mlx-community-qwen38-27b-4bit"></a>

### mlx-community/Qwen3.8-27B-4bit

<details>
<summary>Complete evidence: mlx-community/Qwen3.8-27B-4bit</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Keyword list has 20 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5)
- *Model load time:* 2.11s
- *Generation time:* 81.46s
- *Total time:* 83.60s
- *Input validation time:* 7.82e-05
- *Prompt preparation time:* 0.0177
- *First-token latency:* 76
- *Cleanup time:* 0.133
- *Prompt tokens:* 16,568
- *Generation tokens:* 133
- *Total tokens:* 16,701
- *Prompt throughput (raw):* 218 tok/s
- *Generation throughput (raw):* 29.5 tok/s
- *Peak memory:* 21
- *Active memory:* 16
- *Cache memory:* 1.9
- *Model-load active memory:* 16.1
- *Post-cleanup active memory:* 0.00949
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1400
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
Weathered wooden boathouse on a serene wetland pond

Description:
A weathered wooden boathouse with a mossy roof stands on stilts over a calm pond, featuring a wooden boardwalk with cross-braced railings. The structure is framed by overhanging green tree branches and surrounded by lush wetland foliage under a cloudy sky.

Keywords:
Cloudy Sky, Foliage, Forest, Grass, Lake, Landscape, Leaves, Marshland, Moss, Outdoors, Pond, Reeds, Trees, Water reflection, Wetland, Wooden shed, architecture, bird hide, birdwatching, boardwalk
</pre>

</details>

---

<a id="model-mlx-community-smolvlm2-22b-instruct-mlx"></a>

### mlx-community/SmolVLM2-2.2B-Instruct-mlx

<details>
<summary>Complete evidence: mlx-community/SmolVLM2-2.2B-Instruct-mlx</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Title has 2 words (requested 5-10); Keyword list has 20
  terms (requested 10-18); Title, Description and Keywords copy all supplied
  hints unchanged
- *Arch supported by installed mlx-vlm:* yes (model_type smolvlm)
- *Model load time:* 0.65s
- *Generation time:* 2.01s
- *Total time:* 2.67s
- *Input validation time:* 7.97e-05
- *Prompt preparation time:* 0.0032
- *First-token latency:* 0.394
- *Cleanup time:* 0.0888
- *Prompt tokens:* 1,430
- *Generation tokens:* 104
- *Total tokens:* 1,534
- *Prompt throughput (raw):* 3,626 tok/s
- *Generation throughput (raw):* 123 tok/s
- *Peak memory:* 5.5
- *Active memory:* 4.5
- *Cache memory:* 0.35
- *Model-load active memory:* 4.5
- *Post-cleanup active memory:* 0.0096
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1341
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
 Title: Wooden Boathouse
Description: A weathered wooden boathouse built on stilts stands over the edge of a serene pond, framed by lush overhanging tree branches, wooden decking, and surrounding wetland foliage.
Keywords: Cloudy Sky, Foliage, Forest, Grass, Lake, Landscape, Leaves, Marshland, Moss, Outdoors, Pond, Reeds, Trees, Water reflection, Wetland, Wooden shed, architecture, bird hide, birdwatching, boardwalk
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
- *Observations:* Keyword list has 20 terms (requested 10-18); Title,
  Description and Keywords copy all supplied hints unchanged
- *Arch supported by installed mlx-vlm:* yes (model_type step3p7)
- *Model load time:* 9.46s
- *Generation time:* 28.97s
- *Total time:* 38.45s
- *Input validation time:* 5.06e-05
- *Prompt preparation time:* 0.0198
- *First-token latency:* 22.3
- *Cleanup time:* 0.365
- *Prompt tokens:* 3,496
- *Generation tokens:* 108
- *Total tokens:* 3,604
- *Prompt throughput (raw):* 157 tok/s
- *Generation throughput (raw):* 18.8 tok/s
- *Peak memory:* 70
- *Active memory:* 63
- *Cache memory:* 0.41
- *Model-load active memory:* 63
- *Post-cleanup active memory:* 0.00986
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1387
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
Weathered wooden boathouse over a pond

Description:
A weathered wooden boathouse built on stilts stands over the edge of a serene pond, framed by lush overhanging tree branches, wooden decking, and surrounding wetland foliage.

Keywords:
Cloudy Sky, Foliage, Forest, Grass, Lake, Landscape, Leaves, Marshland, Moss, Outdoors, Pond, Reeds, Trees, Water reflection, Wetland, Wooden shed, architecture, bird hide, birdwatching, boardwalk
</pre>

</details>

---

<a id="model-mlx-community-diffusiongemma-26b-a4b-it-8bit"></a>

### mlx-community/diffusiongemma-26B-A4B-it-8bit

<details>
<summary>Complete evidence: mlx-community/diffusiongemma-26B-A4B-it-8bit</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Unrecognised model control tokens remain visible
- *Arch supported by installed mlx-vlm:* yes (model_type diffusion_gemma)
- *Model load time:* 3.41s
- *Generation time:* 3.05s
- *Total time:* 6.48s
- *Input validation time:* 6.31e-05
- *Prompt preparation time:* 0.0232
- *First-token latency:* 0.65
- *Cleanup time:* 0.143
- *Prompt tokens:* 594
- *Generation tokens:* 88
- *Total tokens:* 682
- *Prompt throughput (raw):* 914 tok/s
- *Generation throughput (raw):* 53.4 tok/s
- *Peak memory:* 29
- *Active memory:* 28
- *Cache memory:* 0.01
- *Model-load active memory:* 28
- *Post-cleanup active memory:* 0.0102
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1335
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
&lt;|channel&gt;thought
&lt;channel|&gt;Title: Weathered Wooden Boathouse Over Serene Wetland Pond
Description: A weathered wooden boathouse built on stilts stands over a serene pond, framed by lush green trees and wetland foliage under a cloudy sky.
Keywords: architecture, birdwatching, hide, boardwalk, foliage, forest, landscape, marshland, moss, outdoors, pond, reeds, trees, water reflection, wetland, wooden shed
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|channel>thought
<channel|>Title: Weathered Wooden Boathouse Over Serene Wetland Pond
Description: A weathered wooden boathouse built on stilts stands over a serene pond, framed by lush green trees and wetland foliage under a cloudy sky.
Keywords: architecture, birdwatching, hide, boardwalk, foliage, forest, landscape, marshland, moss, outdoors, pond, reeds, trees, water reflection, wetland, wooden shed
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
- *Model load time:* 3.38s
- *Generation time:* 2.60s
- *Total time:* 6.00s
- *Input validation time:* 8.07e-05
- *Prompt preparation time:* 0.00196
- *First-token latency:* 0.422
- *Cleanup time:* 0.135
- *Prompt tokens:* 594
- *Generation tokens:* 84
- *Total tokens:* 678
- *Prompt throughput (raw):* 1,407 tok/s
- *Generation throughput (raw):* 60.4 tok/s
- *Peak memory:* 28
- *Active memory:* 27
- *Cache memory:* 0.01
- *Model-load active memory:* 27.2
- *Post-cleanup active memory:* 0.0102
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1335
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
&lt;|channel&gt;thought
&lt;channel|&gt;Title: Weathered Wooden Boathouse over a Serene Pond
Description: A weathered wooden boathouse built on stilts stands over a pond, framed by lush green foliage under a cloudy sky.
Keywords: Boathouse, Wooden, Pond, Stilts, Wetland, Foliage, Trees, Boardwalk, Re, Marshland, Reflection, Cloudy Sky, Architecture, Nature, Outdoors
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|channel>thought
<channel|>Title: Weathered Wooden Boathouse over a Serene Pond
Description: A weathered wooden boathouse built on stilts stands over a pond, framed by lush green foliage under a cloudy sky.
Keywords: Boathouse, Wooden, Pond, Stilts, Wetland, Foliage, Trees, Boardwalk, Re, Marshland, Reflection, Cloudy Sky, Architecture, Nature, Outdoors
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
- *Observations:* Keyword list has 20 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type pixtral)
- *Model load time:* 1.67s
- *Generation time:* 5.28s
- *Total time:* 6.96s
- *Input validation time:* 4.61e-05
- *Prompt preparation time:* 0.00372
- *First-token latency:* 1.9
- *Cleanup time:* 0.107
- *Prompt tokens:* 3,124
- *Generation tokens:* 112
- *Total tokens:* 3,236
- *Prompt throughput (raw):* 1,646 tok/s
- *Generation throughput (raw):* 39.4 tok/s
- *Peak memory:* 16
- *Active memory:* 13
- *Cache memory:* 0.59
- *Model-load active memory:* 13.5
- *Post-cleanup active memory:* 0.0128
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1309
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
Weathered Boathouse Overlooking Serene Wetland Pond

Description:
A weathered wooden boathouse stands on stilts over a tranquil pond, surrounded by lush foliage and framed by overhanging tree branches under a cloudy sky.

Keywords:
Cloudy Sky, Foliage, Forest, Grass, Lake, Landscape, Leaves, Marshland, Moss, Outdoors, Pond, Reeds, Trees, Water reflection, Wetland, Wooden shed, architecture, bird hide, birdwatching, boardwalk
</pre>

</details>

---

<a id="model-jinaai-jina-vlm-mlx"></a>

### jinaai/jina-vlm-mlx

<details>
<summary>Complete evidence: jinaai/jina-vlm-mlx</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Response repeats the same text; Response appears cut off at
  the token limit; Title has 4 words (requested 5-10); Keyword list has 380
  terms (requested 10-18); Duplicate keywords: boathouse, pond, foliage, tree,
  deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside
- *Arch supported by installed mlx-vlm:* yes (model_type jvlm via jina_vlm)
- *Model load time:* 0.46s
- *Generation time:* 9.68s
- *Total time:* 10.15s
- *Input validation time:* 5.12e-05
- *Prompt preparation time:* 0.00207
- *First-token latency:* 1.34
- *Cleanup time:* 0.0768
- *Prompt tokens:* 2,072
- *Generation tokens:* 1,000
- *Total tokens:* 3,072
- *Prompt throughput (raw):* 1,546 tok/s
- *Generation throughput (raw):* 133 tok/s
- *Peak memory:* 3.7
- *Active memory:* 2.1
- *Cache memory:* 0.38
- *Model-load active memory:* 2.11
- *Post-cleanup active memory:* 0.000755
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1321
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
A Serene Pondside Hideout
Description:
A rustic wooden boathouse built on stilts overlooks a tranquil pond, surrounded by lush foliage and framed by the overhanging branches of a nearby tree. The wooden deck and railing add to the charm of this peaceful outdoor retreat, perfect for birdwatching or simply enjoying the serenity of nature.
Keywords:
boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond,
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
- *Observations:* Missing or empty fields: Title, Description, Keywords;
  Response repeats the task instructions instead of only returning the
  requested fields; Response appears cut off at the token limit
- *Arch supported by installed mlx-vlm:* yes (model_type pixtral)
- *Model load time:* 1.62s
- *Generation time:* 26.80s
- *Total time:* 28.43s
- *Input validation time:* 5.88e-05
- *Prompt preparation time:* 0.00786
- *First-token latency:* 2.03
- *Cleanup time:* 0.117
- *Prompt tokens:* 3,215
- *Generation tokens:* 1,000
- *Total tokens:* 4,215
- *Prompt throughput (raw):* 1,583 tok/s
- *Generation throughput (raw):* 41.3 tok/s
- *Peak memory:* 15
- *Active memory:* 13
- *Cache memory:* 0.87
- *Model-load active memory:* 12.6
- *Post-cleanup active memory:* 0.00102
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1683
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
We need to produce British-English catalogue metadata from the image and supplied context. The context includes capture date/time and GPS. The descriptive hints are given. We need to treat capture date/time and GPS as authoritative facts, but not claim they are visible. So we can include them in description? The description should combine relevant context with main visible subject, setting, action, lighting, and distinctive details. So we can mention that the image was captured on 21 August 2026 at 14:34:53 (UTC+01:00) at coordinates 51.441113°N, 0.565406°W. But we need to be careful: The description should be 1-2 sentences factual, combining relevant context with main visible subject, setting, action, lighting, and distinctive details. So we can say something like: "A weathered wooden boathouse on stilts overlooks a tranquil pond, surrounded by lush overhanging foliage and wetland reeds, captured on 21 August 2026 at 14:34:53 UTC+01:00 at 51.441113°N, 0.565406°W under a cloudy sky." That would be a single sentence. Or we can break into two sentences: "A weathered wooden boathouse on stilts overlooks a tranquil pond, surrounded by lush overhanging foliage and wetland reeds. The scene was captured on 21 August 2026 at 14:34:53 UTC+01:00 at 51.441113°N, 0.565406°W under a cloudy sky." That is two sentences.

Now the title: 5-10 words, concrete. Something like "Stilted Boathouse Over Serene Pond". That's 5 words. Or "Weathered Boathouse on Stilts by Pond". That's 7 words. Let's pick "Weathered Boathouse on Stilts by Pond". That's 7 words.

Now keywords: 10-18 unique, comma-separated. Should cover relevant context and visible details. Use British English. Keywords: Cloudy Sky, Foliage, Forest, Grass, Lake, Landscape, Leaves, Marshland, Moss, Outdoors, Pond, Reeds, Trees, Water reflection, Wetland, Wooden shed, architecture, bird hide, birdwatching, boardwalk. That's 20 items, but we need 10-18. So we need to select 10-18 unique keywords. Let's pick 14 maybe. Let's list: Cloudy Sky, Foliage, Forest, Grass, Pond, Reeds, Wetland, Wooden shed, Architecture, Bird hide, Birdwatching, Boardwalk, Water reflection, Moss. That's 14. Ensure they are unique and separated by commas. Use British spelling: "Grass" is fine. "Lake" maybe not needed because it's a pond. "Landscape" maybe. But we have enough.

Now we need to output exactly three sections: Title, Description, Keywords. No extra text. Ensure each line starts with the label and colon. Title: "Weathered Boathouse on Stilts by Pond". Description: two sentences. Keywords: list.

Check for any conflict: The descriptive hints mention "bird hide, birdwatching". The image shows a wooden structure that could be a bird hide. So we can include those keywords. The context includes capture date/time and GPS. We can mention them in description. The description should combine relevant context with main visible subject, setting, action, lighting, and distinctive details. So we need to mention the cloudy sky (lighting). The image shows overcast, diffused light. So we can mention "overcast lighting". The description can mention "soft, diffused light from an overcast sky". That is good.

Now we need to ensure the description is 1-2 sentences. Let's craft:

"Weathered wooden boathouse on stilts overlooks a tranquil pond, framed by lush overhanging foliage, reeds, and wetland vegetation under a cloudy sky. Captured on 21 August 2026 at 14:34:53 UTC+01:00 at 51.441113°N, 0.565406°W, the scene shows soft, diffused lighting and a calm water reflection."

That's two sentences. Good.

Now keywords: we
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
- *Observations:* Response repeats the same text; Response appears cut off at
  the token limit; Keyword list has 158 terms (requested 10-18); Duplicate
  keywords: boardwalk
- *Arch supported by installed mlx-vlm:* yes (model_type ernie4_5_moe_vl)
- *Model load time:* 7.74s
- *Generation time:* 25.44s
- *Total time:* 33.20s
- *Input validation time:* 6.88e-05
- *Prompt preparation time:* 0.0108
- *First-token latency:* 3.95
- *Cleanup time:* 0.253
- *Prompt tokens:* 1,641
- *Generation tokens:* 1,000
- *Total tokens:* 2,641
- *Prompt throughput (raw):* 415 tok/s
- *Generation throughput (raw):* 50.9 tok/s
- *Peak memory:* 60
- *Active memory:* 59
- *Cache memory:* 0.17
- *Model-load active memory:* 58.8
- *Post-cleanup active memory:* 0.00149
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1499
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
Alright, let's get to work. I've got an image and some hints to work with, and my task is to create some metadata for it. First things first, I need to break down the request. It's a straightforward task: title, description, and keywords. The image is of a boathouse, and I have the context to work with.

The image itself is a bit of a mess, but I can see the core elements. It's a wooden boathouse, likely a traditional one, built on stilts over water. The setting is a wetland or marsh, with reeds, willows, and a pond. The lighting is overcast, which is important for the mood.

Now, the hints. The description is a bit repetitive, but I can extract the key elements: weathered wooden boathouse, stilts, pond, tree branches, decking, wetland foliage. The keywords are more useful, but some are redundant. I need to be careful about what to include.

The authoritative context is crucial. The date and time are useless for the image itself, but the GPS coordinates are gold. That's the key to the location. I'll use that to pinpoint the location.

Okay, time to synthesize. The title needs to be concise and descriptive. I'll go with "Wooden Boathouse Over a Wetland Pond." The description needs to be factual and incorporate the context. I'll start with the location, then describe the boathouse, the setting, and the weather. The keywords need to be unique and relevant. I'll use the authoritative GPS coordinates, the key elements from the hints, and the location name.

Let's refine the description. I'll make sure it's accurate and flows well. I'll also make sure to include the location name.

Final check: Title, description, keywords. All good. I'm ready to output the metadata.
&lt;/think&gt;

Title: Wooden Boathouse Over a Wetland Pond

Description: A weathered wooden boathouse built on stilts stands over the edge of a serene pond in a wetland, framed by lush overhanging tree branches and surrounding reeds. The scene is set in a tranquil, overcast landscape with a damp, green atmosphere.

Keywords: 51.441113°N, 0.565406°W, boathouse, wetland, pond, reeds, willow, overcast sky, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk,
</pre>

<details>
<summary>Exact raw output</summary>

```text
Alright, let's get to work. I've got an image and some hints to work with, and my task is to create some metadata for it. First things first, I need to break down the request. It's a straightforward task: title, description, and keywords. The image is of a boathouse, and I have the context to work with.

The image itself is a bit of a mess, but I can see the core elements. It's a wooden boathouse, likely a traditional one, built on stilts over water. The setting is a wetland or marsh, with reeds, willows, and a pond. The lighting is overcast, which is important for the mood.

Now, the hints. The description is a bit repetitive, but I can extract the key elements: weathered wooden boathouse, stilts, pond, tree branches, decking, wetland foliage. The keywords are more useful, but some are redundant. I need to be careful about what to include.

The authoritative context is crucial. The date and time are useless for the image itself, but the GPS coordinates are gold. That's the key to the location. I'll use that to pinpoint the location.

Okay, time to synthesize. The title needs to be concise and descriptive. I'll go with "Wooden Boathouse Over a Wetland Pond." The description needs to be factual and incorporate the context. I'll start with the location, then describe the boathouse, the setting, and the weather. The keywords need to be unique and relevant. I'll use the authoritative GPS coordinates, the key elements from the hints, and the location name.

Let's refine the description. I'll make sure it's accurate and flows well. I'll also make sure to include the location name.

Final check: Title, description, keywords. All good. I'm ready to output the metadata.
</think>

Title: Wooden Boathouse Over a Wetland Pond

Description: A weathered wooden boathouse built on stilts stands over the edge of a serene pond in a wetland, framed by lush overhanging tree branches and surrounding reeds. The scene is set in a tranquil, overcast landscape with a damp, green atmosphere.

Keywords: 51.441113°N, 0.565406°W, boathouse, wetland, pond, reeds, willow, overcast sky, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk,
```

</details>

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
- *Model load time:* 0.63s
- *Generation time:* 1.54s
- *Total time:* 2.18s
- *Input validation time:* 6.4e-05
- *Prompt preparation time:* 0.00244
- *First-token latency:* 0.217
- *Cleanup time:* 0.0797
- *Prompt tokens:* 339
- *Generation tokens:* 54
- *Total tokens:* 393
- *Prompt throughput (raw):* 1,559 tok/s
- *Generation throughput (raw):* 344 tok/s
- *Peak memory:* 2.1
- *Active memory:* 1.2
- *Cache memory:* 0.03
- *Model-load active memory:* 1.25
- *Post-cleanup active memory:* 0.0018
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1404
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
A weathered wooden boathouse stands over a serene pond, surrounded by lush foliage and a cloudy sky, with a concrete 5-10-word title: "A tranquil lakeside scene with a weathered wooden boathouse and overhanging trees."
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
- *Observations:* Response repeats the same text; Extra text appears before
  the Title field; Response appears cut off at the token limit; Internal
  reasoning block appears incomplete; Keyword list has 80 terms (requested
  10-18); Duplicate keywords: foliage, pond, wetland, wooden shed, trees,
  water reflection, marshland, reeds, leaves, outdoors, landscape, moss,
  boardwalk, architecture, grass, bird hide, birdwatching
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v)
- *Model load time:* 1.71s
- *Generation time:* 27.43s
- *Total time:* 29.14s
- *Input validation time:* 6.22e-05
- *Prompt preparation time:* 0.00256
- *First-token latency:* 5.57
- *Cleanup time:* 0.106
- *Prompt tokens:* 6,388
- *Generation tokens:* 1,000
- *Total tokens:* 7,388
- *Prompt throughput (raw):* 1,146 tok/s
- *Generation throughput (raw):* 47.0 tok/s
- *Peak memory:* 13
- *Active memory:* 12
- *Cache memory:* 0.43
- *Model-load active memory:* 11.8
- *Post-cleanup active memory:* 0.00211
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1366
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
&lt;think&gt;Got it, let's tackle this task step by step. First, the title needs to be 5-10 words, concrete. Looking at the image: weathered wooden boathouse, stilts, pond, trees, etc. Maybe "Weathered Wooden Boathouse by the Pond".

Next, the description: combine context (date/time, GPS) with main subject, setting, action, lighting, distinctive details. The image has a weathered wooden structure on stilts over water, lush trees, cloudy sky, wetland. So something like "A weathered wooden boathouse on stilts overlooks a serene pond, surrounded by lush foliage and under a cloudy sky, captured on 2026-08-21." Wait, need to make sure it's 1-2 sentences. Let's check the hints and image. The boathouse is on stilts, over pond, framed by trees, wooden decking. So description: "A weathered wooden boathouse on stilts stands over a serene pond, framed by lush overhanging trees and wetland vegetation, under a cloudy sky, captured on 2026-08-21."

Then keywords: need 10-18 unique, comma-separated. From hints and image: Cloudy Sky, Foliage, Forest, Grass, Lake, Landscape, Leaves, Marshland, Moss, Outdoors, Pond, Reeds, Trees, Water reflection, Wetland, Wooden shed, architecture, bird hide, birdwatching, boardwalk. Let's pick the most relevant ones. Let's list: Cloudy Sky, Foliage, Pond, Wetland, Wooden shed, Trees, Water reflection, Marshland, Reeds, Leaves, Outdoors, Landscape, Moss, Boardwalk, Architecture, Grass. Wait, need to make sure they are unique. Let's check each hint.

Now, let's structure each part:

Title: "Weathered Wooden Boathouse by the Pond" (about 8 words, concrete)

Description: Combine context (date/time, GPS) with subject, setting, etc. So: "A weathered wooden boathouse on stilts overlooks a serene pond, surrounded by lush foliage and under a cloudy sky, captured on 2026-08-21 at 14:34:53 UTC+01:00 near 51.441113°N, 0.565406°W." Wait, but need to make it 1-2 sentences. Maybe shorter: "A weathered wooden boathouse on stilts stands over a serene pond, framed by lush trees and wetland vegetation, under a cloudy sky, captured on 2026-08-21." That's two sentences? Wait, no, 1-2 sentences. Let's make it one: "A weathered wooden boathouse on stilts overlooks a serene pond, surrounded by lush foliage and under a cloudy sky, captured on 2026-08-21."

Keywords: Let's pick the most relevant ones from hints and image. Let's list: Cloudy Sky, Foliage, Pond, Wetland, Wooden shed, Trees, Water reflection, Marshland, Reeds, Leaves, Outdoors, Landscape, Moss, Boardwalk, Architecture, Grass. Wait, let's check the hints again. The hints include: Cloudy Sky, Foliage, Forest, Grass, Lake, Landscape, Leaves, Marshland, Moss, Outdoors, Pond, Reeds, Trees, Water reflection, Wetland, Wooden shed, architecture, bird hide, birdwatching, boardwalk. So unique ones: Cloudy Sky, Foliage, Forest (or Trees), Grass, Lake (or Pond), Landscape, Leaves, Marshland, Moss, Outdoors, Pond, Reeds, Trees, Water reflection, Wetland, Wooden shed, Architecture, Bird hide, Birdwatching, Boardwalk. Let's pick 12-15. Let's do: Cloudy Sky, Foliage, Pond, Wetland, Wooden shed, Trees, Water reflection, Marshland, Reeds, Leaves, Outdoors, Landscape, Moss, Boardwalk, Architecture. That's 15, unique.

Wait, let's make sure the description is factual, combining context with main subject. So: "A weathered wooden boathouse on stilts stands over a serene pond, surrounded by lush foliage and under a cloudy sky, captured on 2026-08-21 at 14:34:53 UTC+01:00 near 51.441113°N, 0.565406°W." That's two sentences, but maybe can make it one. "A weathered wooden boathouse on stilts overlooks a serene pond, framed by lush trees and wetland vegetation, under a cloudy sky, captured
</pre>

<details>
<summary>Exact raw output</summary>

```text
<think>Got it, let's tackle this task step by step. First, the title needs to be 5-10 words, concrete. Looking at the image: weathered wooden boathouse, stilts, pond, trees, etc. Maybe "Weathered Wooden Boathouse by the Pond". 

Next, the description: combine context (date/time, GPS) with main subject, setting, action, lighting, distinctive details. The image has a weathered wooden structure on stilts over water, lush trees, cloudy sky, wetland. So something like "A weathered wooden boathouse on stilts overlooks a serene pond, surrounded by lush foliage and under a cloudy sky, captured on 2026-08-21." Wait, need to make sure it's 1-2 sentences. Let's check the hints and image. The boathouse is on stilts, over pond, framed by trees, wooden decking. So description: "A weathered wooden boathouse on stilts stands over a serene pond, framed by lush overhanging trees and wetland vegetation, under a cloudy sky, captured on 2026-08-21." 

Then keywords: need 10-18 unique, comma-separated. From hints and image: Cloudy Sky, Foliage, Forest, Grass, Lake, Landscape, Leaves, Marshland, Moss, Outdoors, Pond, Reeds, Trees, Water reflection, Wetland, Wooden shed, architecture, bird hide, birdwatching, boardwalk. Let's pick the most relevant ones. Let's list: Cloudy Sky, Foliage, Pond, Wetland, Wooden shed, Trees, Water reflection, Marshland, Reeds, Leaves, Outdoors, Landscape, Moss, Boardwalk, Architecture, Grass. Wait, need to make sure they are unique. Let's check each hint. 

Now, let's structure each part:

Title: "Weathered Wooden Boathouse by the Pond" (about 8 words, concrete)

Description: Combine context (date/time, GPS) with subject, setting, etc. So: "A weathered wooden boathouse on stilts overlooks a serene pond, surrounded by lush foliage and under a cloudy sky, captured on 2026-08-21 at 14:34:53 UTC+01:00 near 51.441113°N, 0.565406°W." Wait, but need to make it 1-2 sentences. Maybe shorter: "A weathered wooden boathouse on stilts stands over a serene pond, framed by lush trees and wetland vegetation, under a cloudy sky, captured on 2026-08-21." That's two sentences? Wait, no, 1-2 sentences. Let's make it one: "A weathered wooden boathouse on stilts overlooks a serene pond, surrounded by lush foliage and under a cloudy sky, captured on 2026-08-21." 

Keywords: Let's pick the most relevant ones from hints and image. Let's list: Cloudy Sky, Foliage, Pond, Wetland, Wooden shed, Trees, Water reflection, Marshland, Reeds, Leaves, Outdoors, Landscape, Moss, Boardwalk, Architecture, Grass. Wait, let's check the hints again. The hints include: Cloudy Sky, Foliage, Forest, Grass, Lake, Landscape, Leaves, Marshland, Moss, Outdoors, Pond, Reeds, Trees, Water reflection, Wetland, Wooden shed, architecture, bird hide, birdwatching, boardwalk. So unique ones: Cloudy Sky, Foliage, Forest (or Trees), Grass, Lake (or Pond), Landscape, Leaves, Marshland, Moss, Outdoors, Pond, Reeds, Trees, Water reflection, Wetland, Wooden shed, Architecture, Bird hide, Birdwatching, Boardwalk. Let's pick 12-15. Let's do: Cloudy Sky, Foliage, Pond, Wetland, Wooden shed, Trees, Water reflection, Marshland, Reeds, Leaves, Outdoors, Landscape, Moss, Boardwalk, Architecture. That's 15, unique. 

Wait, let's make sure the description is factual, combining context with main subject. So: "A weathered wooden boathouse on stilts stands over a serene pond, surrounded by lush foliage and under a cloudy sky, captured on 2026-08-21 at 14:34:53 UTC+01:00 near 51.441113°N, 0.565406°W." That's two sentences, but maybe can make it one. "A weathered wooden boathouse on stilts overlooks a serene pond, framed by lush trees and wetland vegetation, under a cloudy sky, captured
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
  the token limit; Keyword list has 395 terms (requested 10-18); Duplicate
  keywords: elevated, wetland, pond, outdoor, railings, reflection, natural,
  environment, architecture, birdwatching, peaceful, serene
- *Arch supported by installed mlx-vlm:* yes (model_type lfm2_vl)
- *Model load time:* 0.61s
- *Generation time:* 6.25s
- *Total time:* 6.86s
- *Input validation time:* 6.67e-05
- *Prompt preparation time:* 0.00417
- *First-token latency:* 0.336
- *Cleanup time:* 0.0807
- *Prompt tokens:* 2,117
- *Generation tokens:* 1,000
- *Total tokens:* 3,117
- *Prompt throughput (raw):* 6,303 tok/s
- *Generation throughput (raw):* 186 tok/s
- *Peak memory:* 4.0
- *Active memory:* 3.2
- *Cache memory:* 0.07
- *Model-load active memory:* 3.2
- *Post-cleanup active memory:* 0.0041
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1360
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
Title: "Moss-Covered Wooden Boathouse Over Wetland Pond"

Description: A weathered wooden boathouse stands elevated on stilts over a serene pond, surrounded by lush foliage and marshland. The overhanging tree branches frame the structure, while the cloudy sky casts a soft, diffused light over the scene. The wooden decking and railings add to the rustic charm of this tranquil outdoor setting.

Keywords: boathouse, wooden, moss, elevated, wetland, pond, overhanging, foliage, marshland, cloudy, diffused, light, rustic, tranquil, outdoor, setting, structure, railings, decking, railings, trees, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated
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
- *Model load time:* 0.83s
- *Generation time:* 1.18s
- *Total time:* 2.03s
- *Input validation time:* 5.43e-05
- *Prompt preparation time:* 0.00962
- *First-token latency:* 0.247
- *Cleanup time:* 0.0918
- *Prompt tokens:* 937
- *Generation tokens:* 76
- *Total tokens:* 1,013
- *Prompt throughput (raw):* 3,791 tok/s
- *Generation throughput (raw):* 275 tok/s
- *Peak memory:* 3.8
- *Active memory:* 2.3
- *Cache memory:* 0.05
- *Model-load active memory:* 2.32
- *Post-cleanup active memory:* 0.00462
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1377
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
Weathered wooden boathouse by calm pond under cloudy skies.
It stands elevated over the water surrounded by lush greenery and dense foliage.

Keywords: wooden boathouse, pond, cloudy sky, foliage, marshland, reeds, wetland, trees, water reflection, moss, outdoors, landscape, deck, railing, greenery, overhanging branches
</pre>

<details>
<summary>Exact raw output</summary>

```text
Weathered wooden boathouse by calm pond under cloudy skies.  
It stands elevated over the water surrounded by lush greenery and dense foliage.

Keywords: wooden boathouse, pond, cloudy sky, foliage, marshland, reeds, wetland, trees, water reflection, moss, outdoors, landscape, deck, railing, greenery, overhanging branches
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
  the token limit; Keyword list has 436 terms (requested 10-18); Duplicate
  keywords: pond, wetland, grass, reeds, trees, architecture, moss, weathered,
  landscape, outdoor, nature, structure, environment, ecosystem, habitat,
  accurate, precise, detailed, comprehensive, thorough
- *Arch supported by installed mlx-vlm:* yes (model_type qwen2_5_vl)
- *Model load time:* 1.27s
- *Generation time:* 36.95s
- *Total time:* 38.23s
- *Input validation time:* 8.15e-05
- *Prompt preparation time:* 0.00526
- *First-token latency:* 18.1
- *Cleanup time:* 0.0988
- *Prompt tokens:* 16,563
- *Generation tokens:* 1,000
- *Total tokens:* 17,563
- *Prompt throughput (raw):* 916 tok/s
- *Generation throughput (raw):* 55.8 tok/s
- *Peak memory:* 13
- *Active memory:* 9.5
- *Cache memory:* 1.3
- *Model-load active memory:* 9.46
- *Post-cleanup active memory:* 0.0102
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1439
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
Wooden Boathouse Over Serene Pond

Description:
A weathered wooden boathouse, elevated on stilts, stands over a tranquil pond, framed by lush greenery and overhanging tree branches, with a cloudy sky and reflections on the water.

Keywords:
Wooden boathouse, stilted, pond, serene, cloudy sky, lush foliage, overhanging branches, water reflection, wetland, grass, reeds, trees, architecture, bird hide, birdwatching, boardwalk, moss, weathered, landscape, outdoor, nature, countryside, 2026-08-21, 51.441113°N, 0.565406°W, architecture, boathouse, shed, structure, stillness, calm, natural setting, rural, environment, ecosystem, wildlife, habitat, peaceful, quiet, still, calmness, tranquility, natural beauty, greenery, foliage, leaves, moss, water, reflection, pond, lake, wetland, marshland, forest, grass, reeds, trees, sky, clouds, overcast, weathered, aged, rustic, traditional, historical, cultural, heritage, preservation, conservation, nature reserve, wildlife sanctuary, ecological, biodiversity, ecosystem, habitat, natural, environment, outdoor, outdoors, nature, landscape, scenery, view, perspective, angle, composition, balance, symmetry, harmony, contrast, texture, material, wood, wooden, structure, building, shelter, protection, weather, climate, season, summer, August, 2026, date, time, location, coordinates, GPS, authoritative, factual, descriptive, evidence-based, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed,
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title:
Wooden Boathouse Over Serene Pond

Description:
A weathered wooden boathouse, elevated on stilts, stands over a tranquil pond, framed by lush greenery and overhanging tree branches, with a cloudy sky and reflections on the water.

Keywords: 
Wooden boathouse, stilted, pond, serene, cloudy sky, lush foliage, overhanging branches, water reflection, wetland, grass, reeds, trees, architecture, bird hide, birdwatching, boardwalk, moss, weathered, landscape, outdoor, nature, countryside, 2026-08-21, 51.441113°N, 0.565406°W, architecture, boathouse, shed, structure, stillness, calm, natural setting, rural, environment, ecosystem, wildlife, habitat, peaceful, quiet, still, calmness, tranquility, natural beauty, greenery, foliage, leaves, moss, water, reflection, pond, lake, wetland, marshland, forest, grass, reeds, trees, sky, clouds, overcast, weathered, aged, rustic, traditional, historical, cultural, heritage, preservation, conservation, nature reserve, wildlife sanctuary, ecological, biodiversity, ecosystem, habitat, natural, environment, outdoor, outdoors, nature, landscape, scenery, view, perspective, angle, composition, balance, symmetry, harmony, contrast, texture, material, wood, wooden, structure, building, shelter, protection, weather, climate, season, summer, August, 2026, date, time, location, coordinates, GPS, authoritative, factual, descriptive, evidence-based, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed,
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
- *Model load time:* 2.28s
- *Generation time:* 4.80s
- *Total time:* 7.10s
- *Input validation time:* 6.04e-05
- *Prompt preparation time:* 0.00505
- *First-token latency:* 0.871
- *Cleanup time:* 0.126
- *Prompt tokens:* 592
- *Generation tokens:* 164
- *Total tokens:* 756
- *Prompt throughput (raw):* 680 tok/s
- *Generation throughput (raw):* 48.7 tok/s
- *Peak memory:* 17
- *Active memory:* 16
- *Cache memory:* 0.05
- *Model-load active memory:* 15.7
- *Post-cleanup active memory:* 0.0113
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1366
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
## Image Description

A weathered wooden boathouse stands on stilts over a calm pond. The boathouse is constructed from vertical wooden planks and has a simple, pitched roof covered in moss. A wooden walkway with railings leads to the boathouse. Lush green foliage overhangs the structure, creating dappled shade. The pond reflects the surrounding greenery and the overcast sky. Reeds and other aquatic plants grow along the water's edge. The overall scene evokes a sense of tranquility and rustic charm.

## Keywords

boathouse, wooden, pond, water, reflection, foliage, trees, green, stilts, walkway, railings, rustic, tranquil, overcast, nature, landscape, reeds, aquatic plants, countryside, rural, peaceful, summer, outdoors, architecture, building, boat storage.
</pre>

</details>

---

<a id="model-mlx-community-lfm25-vl-3b-optiq-4bit"></a>

### mlx-community/LFM2.5-VL-3B-OptiQ-4bit

<details>
<summary>Complete evidence: mlx-community/LFM2.5-VL-3B-OptiQ-4bit</summary>

- *Execution:* crashed
- *Usability:* not_evaluated
- *Maintainer status:* actionable_failure
- *Observations:* none
- *Failure phase:* model_load
- *Error stage:* Model Error
- *Error code:* MLX_VLM_MODEL_LOAD_MODEL
- *Error type:* ValueError
- *Error package:* mlx-vlm
- *Error message:* Model loading failed: Received 600 parameters not in model:
  <br>model.embed_tokens.biases,<br>model.embed_tokens.scales,<br>model.embed_tokens.weight,<br>model.embedding_norm.weight,<br>model.layers.0.conv.conv.weight,<br>model.layers.0.conv.in_proj.biases,<br>model.layers.0.conv.in_proj.scales,<br>model.layers.0.conv.in_proj.weight,<br>model.layers.0.conv.out_proj.biases,<br>model.layers.0.conv.out_proj.scales,<br>model.layers.0.conv.out_proj.weight,<br>model.layers.0.feed_forward.w1.biases,<br>model.layers.0.feed_forward.w1.scales,<br>model.layers.0.feed_forward.w1.weight,<br>model.layers.0.feed_forward.w2.biases,<br>model.layers.0.feed_forward.w2.scales,<br>model.layers.0.feed_forward.w2.weight,<br>model.layers.0.feed_forward.w3.biases,<br>model.layers.0.feed_forward.w3.scales,<br>model.layers.0.feed_forward.w3.weight,<br>model.layers.0.ffn_norm.weight,<br>model.layers.0.operator_norm.weight,<br>model.layers.1.conv.conv.weight,<br>model.layers.1.conv.in_proj.biases,<br>model.layers.1.conv.in_proj.scales,<br>model.layers.1.conv.in_proj.weight,<br>model.layers.1.conv.out_proj.biases,<br>model.layers.1.conv.out_proj.scales,<br>model.layers.1.conv.out_proj.weight,<br>model.layers.1.feed_forward.w1.biases,<br>model.layers.1.feed_forward.w1.scales,<br>model.layers.1.feed_forward.w1.weight,<br>model.layers.1.feed_forward.w2.biases,<br>model.layers.1.feed_forward.w2.scales,<br>model.layers.1.feed_forward.w2.weight,<br>model.layers.1.feed_forward.w3.biases,<br>model.layers.1.feed_forward.w3.scales,<br>model.layers.1.feed_forward.w3.weight,<br>model.layers.1.ffn_norm.weight,<br>model.layers.1.operator_norm.weight,<br>model.layers.10.conv.conv.weight,<br>model.layers.10.conv.in_proj.biases,<br>model.layers.10.conv.in_proj.scales,<br>model.layers.10.conv.in_proj.weight,<br>model.layers.10.conv.out_proj.biases,<br>model.layers.10.conv.out_proj.scales,<br>model.layers.10.conv.out_proj.weight,<br>model.layers.10.feed_forward.w1.biases,<br>model.layers.10.feed_forward.w1.scales,<br>model.layers.10.feed_forward.w1.weight,<br>model.layers.10.feed_forward.w2.biases,<br>model.layers.10.feed_forward.w2.scales,<br>model.layers.10.feed_forward.w2.weight,<br>model.layers.10.feed_forward.w3.biases,<br>model.layers.10.feed_forward.w3.scales,<br>model.layers.10.feed_forward.w3.weight,<br>model.layers.10.ffn_norm.weight,<br>model.layers.10.operator_norm.weight,<br>model.layers.11.conv.conv.weight,<br>model.layers.11.conv.in_proj.biases,<br>model.layers.11.conv.in_proj.scales,<br>model.layers.11.conv.in_proj.weight,<br>model.layers.11.conv.out_proj.biases,<br>model.layers.11.conv.out_proj.scales,<br>model.layers.11.conv.out_proj.weight,<br>model.layers.11.feed_forward.w1.biases,<br>model.layers.11.feed_forward.w1.scales,<br>model.layers.11.feed_forward.w1.weight,<br>model.layers.11.feed_forward.w2.biases,<br>model.layers.11.feed_forward.w2.scales,<br>model.layers.11.feed_forward.w2.weight,<br>model.layers.11.feed_forward.w3.biases,<br>model.layers.11.feed_forward.w3.scales,<br>model.layers.11.feed_forward.w3.weight,<br>model.layers.11.ffn_norm.weight,<br>model.layers.11.operator_norm.weight,<br>model.layers.12.conv.conv.weight,<br>model.layers.12.conv.in_proj.biases,<br>model.layers.12.conv.in_proj.scales,<br>model.layers.12.conv.in_proj.weight,<br>model.layers.12.conv.out_proj.biases,<br>model.layers.12.conv.out_proj.scales,<br>model.layers.12.conv.out_proj.weight,<br>model.layers.12.feed_forward.w1.biases,<br>model.layers.12.feed_forward.w1.scales,<br>model.layers.12.feed_forward.w1.weight,<br>model.layers.12.feed_forward.w2.biases,<br>model.layers.12.feed_forward.w2.scales,<br>model.layers.12.feed_forward.w2.weight,<br>model.layers.12.feed_forward.w3.biases,<br>model.layers.12.feed_forward.w3.scales,<br>model.layers.12.feed_forward.w3.weight,<br>model.layers.12.ffn_norm.weight,<br>model.layers.12.operator_norm.weight,<br>model.layers.13.feed_forward.w1.biases,<br>model.layers.13.feed_forward.w1.scales,<br>model.layers.13.feed_forward.w1.weight,<br>model.layers.13.feed_forward.w2.biases,<br>model.layers.13.feed_forward.w2.scales,<br>model.layers.13.feed_forward.w2.weight,<br>model.layers.13.feed_forward.w3.biases,<br>model.layers.13.feed_forward.w3.scales,<br>model.layers.13.feed_forward.w3.weight,<br>model.layers.13.ffn_norm.weight,<br>model.layers.13.operator_norm.weight,<br>model.layers.13.self_attn.k_layernorm.weight,<br>model.layers.13.self_attn.k_proj.biases,<br>model.layers.13.self_attn.k_proj.scales,<br>model.layers.13.self_attn.k_proj.weight,<br>model.layers.13.self_attn.out_proj.biases,<br>model.layers.13.self_attn.out_proj.scales,<br>model.layers.13.self_attn.out_proj.weight,<br>model.layers.13.self_attn.q_layernorm.weight,<br>model.layers.13.self_attn.q_proj.biases,<br>model.layers.13.self_attn.q_proj.scales,<br>model.layers.13.self_attn.q_proj.weight,<br>model.layers.13.self_attn.v_proj.biases,<br>model.layers.13.self_attn.v_proj.scales,<br>model.layers.13.self_attn.v_proj.weight,<br>model.layers.14.conv.conv.weight,<br>model.layers.14.conv.in_proj.biases,<br>model.layers.14.conv.in_proj.scales,<br>model.layers.14.conv.in_proj.weight,<br>model.layers.14.conv.out_proj.biases,<br>model.layers.14.conv.out_proj.scales,<br>model.layers.14.conv.out_proj.weight,<br>model.layers.14.feed_forward.w1.biases,<br>model.layers.14.feed_forward.w1.scales,<br>model.layers.14.feed_forward.w1.weight,<br>model.layers.14.feed_forward.w2.biases,<br>model.layers.14.feed_forward.w2.scales,<br>model.layers.14.feed_forward.w2.weight,<br>model.layers.14.feed_forward.w3.biases,<br>model.layers.14.feed_forward.w3.scales,<br>model.layers.14.feed_forward.w3.weight,<br>model.layers.14.ffn_norm.weight,<br>model.layers.14.operator_norm.weight,<br>model.layers.15.conv.conv.weight,<br>model.layers.15.conv.in_proj.biases,<br>model.layers.15.conv.in_proj.scales,<br>model.layers.15.conv.in_proj.weight,<br>model.layers.15.conv.out_proj.biases,<br>model.layers.15.conv.out_proj.scales,<br>model.layers.15.conv.out_proj.weight,<br>model.layers.15.feed_forward.w1.biases,<br>model.layers.15.feed_forward.w1.scales,<br>model.layers.15.feed_forward.w1.weight,<br>model.layers.15.feed_forward.w2.biases,<br>model.layers.15.feed_forward.w2.scales,<br>model.layers.15.feed_forward.w2.weight,<br>model.layers.15.feed_forward.w3.biases,<br>model.layers.15.feed_forward.w3.scales,<br>model.layers.15.feed_forward.w3.weight,<br>model.layers.15.ffn_norm.weight,<br>model.layers.15.operator_norm.weight,<br>model.layers.16.conv.conv.weight,<br>model.layers.16.conv.in_proj.biases,<br>model.layers.16.conv.in_proj.scales,<br>model.layers.16.conv.in_proj.weight,<br>model.layers.16.conv.out_proj.biases,<br>model.layers.16.conv.out_proj.scales,<br>model.layers.16.conv.out_proj.weight,<br>model.layers.16.feed_forward.w1.biases,<br>model.layers.16.feed_forward.w1.scales,<br>model.layers.16.feed_forward.w1.weight,<br>model.layers.16.feed_forward.w2.biases,<br>model.layers.16.feed_forward.w2.scales,<br>model.layers.16.feed_forward.w2.weight,<br>model.layers.16.feed_forward.w3.biases,<br>model.layers.16.feed_forward.w3.scales,<br>model.layers.16.feed_forward.w3.weight,<br>model.layers.16.ffn_norm.weight,<br>model.layers.16.operator_norm.weight,<br>model.layers.17.feed_forward.w1.biases,<br>model.layers.17.feed_forward.w1.scales,<br>model.layers.17.feed_forward.w1.weight,<br>model.layers.17.feed_forward.w2.biases,<br>model.layers.17.feed_forward.w2.scales,<br>model.layers.17.feed_forward.w2.weight,<br>model.layers.17.feed_forward.w3.biases,<br>model.layers.17.feed_forward.w3.scales,<br>model.layers.17.feed_forward.w3.weight,<br>model.layers.17.ffn_norm.weight,<br>model.layers.17.operator_norm.weight,<br>model.layers.17.self_attn.k_layernorm.weight,<br>model.layers.17.self_attn.k_proj.biases,<br>model.layers.17.self_attn.k_proj.scales,<br>model.layers.17.self_attn.k_proj.weight,<br>model.layers.17.self_attn.out_proj.biases,<br>model.layers.17.self_attn.out_proj.scales,<br>model.layers.17.self_attn.out_proj.weight,<br>model.layers.17.self_attn.q_layernorm.weight,<br>model.layers.17.self_attn.q_proj.biases,<br>model.layers.17.self_attn.q_proj.scales,<br>model.layers.17.self_attn.q_proj.weight,<br>model.layers.17.self_attn.v_proj.biases,<br>model.layers.17.self_attn.v_proj.scales,<br>model.layers.17.self_attn.v_proj.weight,<br>model.layers.18.conv.conv.weight,<br>model.layers.18.conv.in_proj.biases,<br>model.layers.18.conv.in_proj.scales,<br>model.layers.18.conv.in_proj.weight,<br>model.layers.18.conv.out_proj.biases,<br>model.layers.18.conv.out_proj.scales,<br>model.layers.18.conv.out_proj.weight,<br>model.layers.18.feed_forward.w1.biases,<br>model.layers.18.feed_forward.w1.scales,<br>model.layers.18.feed_forward.w1.weight,<br>model.layers.18.feed_forward.w2.biases,<br>model.layers.18.feed_forward.w2.scales,<br>model.layers.18.feed_forward.w2.weight,<br>model.layers.18.feed_forward.w3.biases,<br>model.layers.18.feed_forward.w3.scales,<br>model.layers.18.feed_forward.w3.weight,<br>model.layers.18.ffn_norm.weight,<br>model.layers.18.operator_norm.weight,<br>model.layers.19.conv.conv.weight,<br>model.layers.19.conv.in_proj.biases,<br>model.layers.19.conv.in_proj.scales,<br>model.layers.19.conv.in_proj.weight,<br>model.layers.19.conv.out_proj.biases,<br>model.layers.19.conv.out_proj.scales,<br>model.layers.19.conv.out_proj.weight,<br>model.layers.19.feed_forward.w1.biases,<br>model.layers.19.feed_forward.w1.scales,<br>model.layers.19.feed_forward.w1.weight,<br>model.layers.19.feed_forward.w2.biases,<br>model.layers.19.feed_forward.w2.scales,<br>model.layers.19.feed_forward.w2.weight,<br>model.layers.19.feed_forward.w3.biases,<br>model.layers.19.feed_forward.w3.scales,<br>model.layers.19.feed_forward.w3.weight,<br>model.layers.19.ffn_norm.weight,<br>model.layers.19.operator_norm.weight,<br>model.layers.2.feed_forward.w1.biases,<br>model.layers.2.feed_forward.w1.scales,<br>model.layers.2.feed_forward.w1.weight,<br>model.layers.2.feed_forward.w2.biases,<br>model.layers.2.feed_forward.w2.scales,<br>model.layers.2.feed_forward.w2.weight,<br>model.layers.2.feed_forward.w3.biases,<br>model.layers.2.feed_forward.w3.scales,<br>model.layers.2.feed_forward.w3.weight,<br>model.layers.2.ffn_norm.weight,<br>model.layers.2.operator_norm.weight,<br>model.layers.2.self_attn.k_layernorm.weight,<br>model.layers.2.self_attn.k_proj.biases,<br>model.layers.2.self_attn.k_proj.scales,<br>model.layers.2.self_attn.k_proj.weight,<br>model.layers.2.self_attn.out_proj.biases,<br>model.layers.2.self_attn.out_proj.scales,<br>model.layers.2.self_attn.out_proj.weight,<br>model.layers.2.self_attn.q_layernorm.weight,<br>model.layers.2.self_attn.q_proj.biases,<br>model.layers.2.self_attn.q_proj.scales,<br>model.layers.2.self_attn.q_proj.weight,<br>model.layers.2.self_attn.v_proj.biases,<br>model.layers.2.self_attn.v_proj.scales,<br>model.layers.2.self_attn.v_proj.weight,<br>model.layers.20.conv.conv.weight,<br>model.layers.20.conv.in_proj.biases,<br>model.layers.20.conv.in_proj.scales,<br>model.layers.20.conv.in_proj.weight,<br>model.layers.20.conv.out_proj.biases,<br>model.layers.20.conv.out_proj.scales,<br>model.layers.20.conv.out_proj.weight,<br>model.layers.20.feed_forward.w1.biases,<br>model.layers.20.feed_forward.w1.scales,<br>model.layers.20.feed_forward.w1.weight,<br>model.layers.20.feed_forward.w2.biases,<br>model.layers.20.feed_forward.w2.scales,<br>model.layers.20.feed_forward.w2.weight,<br>model.layers.20.feed_forward.w3.biases,<br>model.layers.20.feed_forward.w3.scales,<br>model.layers.20.feed_forward.w3.weight,<br>model.layers.20.ffn_norm.weight,<br>model.layers.20.operator_norm.weight,<br>model.layers.21.feed_forward.w1.biases,<br>model.layers.21.feed_forward.w1.scales,<br>model.layers.21.feed_forward.w1.weight,<br>model.layers.21.feed_forward.w2.biases,<br>model.layers.21.feed_forward.w2.scales,<br>model.layers.21.feed_forward.w2.weight,<br>model.layers.21.feed_forward.w3.biases,<br>model.layers.21.feed_forward.w3.scales,<br>model.layers.21.feed_forward.w3.weight,<br>model.layers.21.ffn_norm.weight,<br>model.layers.21.operator_norm.weight,<br>model.layers.21.self_attn.k_layernorm.weight,<br>model.layers.21.self_attn.k_proj.biases,<br>model.layers.21.self_attn.k_proj.scales,<br>model.layers.21.self_attn.k_proj.weight,<br>model.layers.21.self_attn.out_proj.biases,<br>model.layers.21.self_attn.out_proj.scales,<br>model.layers.21.self_attn.out_proj.weight,<br>model.layers.21.self_attn.q_layernorm.weight,<br>model.layers.21.self_attn.q_proj.biases,<br>model.layers.21.self_attn.q_proj.scales,<br>model.layers.21.self_attn.q_proj.weight,<br>model.layers.21.self_attn.v_proj.biases,<br>model.layers.21.self_attn.v_proj.scales,<br>model.layers.21.self_attn.v_proj.weight,<br>model.layers.22.conv.conv.weight,<br>model.layers.22.conv.in_proj.biases,<br>model.layers.22.conv.in_proj.scales,<br>model.layers.22.conv.in_proj.weight,<br>model.layers.22.conv.out_proj.biases,<br>model.layers.22.conv.out_proj.scales,<br>model.layers.22.conv.out_proj.weight,<br>model.layers.22.feed_forward.w1.biases,<br>model.layers.22.feed_forward.w1.scales,<br>model.layers.22.feed_forward.w1.weight,<br>model.layers.22.feed_forward.w2.biases,<br>model.layers.22.feed_forward.w2.scales,<br>model.layers.22.feed_forward.w2.weight,<br>model.layers.22.feed_forward.w3.biases,<br>model.layers.22.feed_forward.w3.scales,<br>model.layers.22.feed_forward.w3.weight,<br>model.layers.22.ffn_norm.weight,<br>model.layers.22.operator_norm.weight,<br>model.layers.23.conv.conv.weight,<br>model.layers.23.conv.in_proj.biases,<br>model.layers.23.conv.in_proj.scales,<br>model.layers.23.conv.in_proj.weight,<br>model.layers.23.conv.out_proj.biases,<br>model.layers.23.conv.out_proj.scales,<br>model.layers.23.conv.out_proj.weight,<br>model.layers.23.feed_forward.w1.biases,<br>model.layers.23.feed_forward.w1.scales,<br>model.layers.23.feed_forward.w1.weight,<br>model.layers.23.feed_forward.w2.biases,<br>model.layers.23.feed_forward.w2.scales,<br>model.layers.23.feed_forward.w2.weight,<br>model.layers.23.feed_forward.w3.biases,<br>model.layers.23.feed_forward.w3.scales,<br>model.layers.23.feed_forward.w3.weight,<br>model.layers.23.ffn_norm.weight,<br>model.layers.23.operator_norm.weight,<br>model.layers.24.feed_forward.w1.biases,<br>model.layers.24.feed_forward.w1.scales,<br>model.layers.24.feed_forward.w1.weight,<br>model.layers.24.feed_forward.w2.biases,<br>model.layers.24.feed_forward.w2.scales,<br>model.layers.24.feed_forward.w2.weight,<br>model.layers.24.feed_forward.w3.biases,<br>model.layers.24.feed_forward.w3.scales,<br>model.layers.24.feed_forward.w3.weight,<br>model.layers.24.ffn_norm.weight,<br>model.layers.24.operator_norm.weight,<br>model.layers.24.self_attn.k_layernorm.weight,<br>model.layers.24.self_attn.k_proj.biases,<br>model.layers.24.self_attn.k_proj.scales,<br>model.layers.24.self_attn.k_proj.weight,<br>model.layers.24.self_attn.out_proj.biases,<br>model.layers.24.self_attn.out_proj.scales,<br>model.layers.24.self_attn.out_proj.weight,<br>model.layers.24.self_attn.q_layernorm.weight,<br>model.layers.24.self_attn.q_proj.biases,<br>model.layers.24.self_attn.q_proj.scales,<br>model.layers.24.self_attn.q_proj.weight,<br>model.layers.24.self_attn.v_proj.biases,<br>model.layers.24.self_attn.v_proj.scales,<br>model.layers.24.self_attn.v_proj.weight,<br>model.layers.25.conv.conv.weight,<br>model.layers.25.conv.in_proj.biases,<br>model.layers.25.conv.in_proj.scales,<br>model.layers.25.conv.in_proj.weight,<br>model.layers.25.conv.out_proj.biases,<br>model.layers.25.conv.out_proj.scales,<br>model.layers.25.conv.out_proj.weight,<br>model.layers.25.feed_forward.w1.biases,<br>model.layers.25.feed_forward.w1.scales,<br>model.layers.25.feed_forward.w1.weight,<br>model.layers.25.feed_forward.w2.biases,<br>model.layers.25.feed_forward.w2.scales,<br>model.layers.25.feed_forward.w2.weight,<br>model.layers.25.feed_forward.w3.biases,<br>model.layers.25.feed_forward.w3.scales,<br>model.layers.25.feed_forward.w3.weight,<br>model.layers.25.ffn_norm.weight,<br>model.layers.25.operator_norm.weight,<br>model.layers.26.conv.conv.weight,<br>model.layers.26.conv.in_proj.biases,<br>model.layers.26.conv.in_proj.scales,<br>model.layers.26.conv.in_proj.weight,<br>model.layers.26.conv.out_proj.biases,<br>model.layers.26.conv.out_proj.scales,<br>model.layers.26.conv.out_proj.weight,<br>model.layers.26.feed_forward.w1.biases,<br>model.layers.26.feed_forward.w1.scales,<br>model.layers.26.feed_forward.w1.weight,<br>model.layers.26.feed_forward.w2.biases,<br>model.layers.26.feed_forward.w2.scales,<br>model.layers.26.feed_forward.w2.weight,<br>model.layers.26.feed_forward.w3.biases,<br>model.layers.26.feed_forward.w3.scales,<br>model.layers.26.feed_forward.w3.weight,<br>model.layers.26.ffn_norm.weight,<br>model.layers.26.operator_norm.weight,<br>model.layers.27.feed_forward.w1.biases,<br>model.layers.27.feed_forward.w1.scales,<br>model.layers.27.feed_forward.w1.weight,<br>model.layers.27.feed_forward.w2.biases,<br>model.layers.27.feed_forward.w2.scales,<br>model.layers.27.feed_forward.w2.weight,<br>model.layers.27.feed_forward.w3.biases,<br>model.layers.27.feed_forward.w3.scales,<br>model.layers.27.feed_forward.w3.weight,<br>model.layers.27.ffn_norm.weight,<br>model.layers.27.operator_norm.weight,<br>model.layers.27.self_attn.k_layernorm.weight,<br>model.layers.27.self_attn.k_proj.biases,<br>model.layers.27.self_attn.k_proj.scales,<br>model.layers.27.self_attn.k_proj.weight,<br>model.layers.27.self_attn.out_proj.biases,<br>model.layers.27.self_attn.out_proj.scales,<br>model.layers.27.self_attn.out_proj.weight,<br>model.layers.27.self_attn.q_layernorm.weight,<br>model.layers.27.self_attn.q_proj.biases,<br>model.layers.27.self_attn.q_proj.scales,<br>model.layers.27.self_attn.q_proj.weight,<br>model.layers.27.self_attn.v_proj.biases,<br>model.layers.27.self_attn.v_proj.scales,<br>model.layers.27.self_attn.v_proj.weight,<br>model.layers.28.conv.conv.weight,<br>model.layers.28.conv.in_proj.biases,<br>model.layers.28.conv.in_proj.scales,<br>model.layers.28.conv.in_proj.weight,<br>model.layers.28.conv.out_proj.biases,<br>model.layers.28.conv.out_proj.scales,<br>model.layers.28.conv.out_proj.weight,<br>model.layers.28.feed_forward.w1.biases,<br>model.layers.28.feed_forward.w1.scales,<br>model.layers.28.feed_forward.w1.weight,<br>model.layers.28.feed_forward.w2.biases,<br>model.layers.28.feed_forward.w2.scales,<br>model.layers.28.feed_forward.w2.weight,<br>model.layers.28.feed_forward.w3.biases,<br>model.layers.28.feed_forward.w3.scales,<br>model.layers.28.feed_forward.w3.weight,<br>model.layers.28.ffn_norm.weight,<br>model.layers.28.operator_norm.weight,<br>model.layers.29.conv.conv.weight,<br>model.layers.29.conv.in_proj.biases,<br>model.layers.29.conv.in_proj.scales,<br>model.layers.29.conv.in_proj.weight,<br>model.layers.29.conv.out_proj.biases,<br>model.layers.29.conv.out_proj.scales,<br>model.layers.29.conv.out_proj.weight,<br>model.layers.29.feed_forward.w1.biases,<br>model.layers.29.feed_forward.w1.scales,<br>model.layers.29.feed_forward.w1.weight,<br>model.layers.29.feed_forward.w2.biases,<br>model.layers.29.feed_forward.w2.scales,<br>model.layers.29.feed_forward.w2.weight,<br>model.layers.29.feed_forward.w3.biases,<br>model.layers.29.feed_forward.w3.scales,<br>model.layers.29.feed_forward.w3.weight,<br>model.layers.29.ffn_norm.weight,<br>model.layers.29.operator_norm.weight,<br>model.layers.3.conv.conv.weight,<br>model.layers.3.conv.in_proj.biases,<br>model.layers.3.conv.in_proj.scales,<br>model.layers.3.conv.in_proj.weight,<br>model.layers.3.conv.out_proj.biases,<br>model.layers.3.conv.out_proj.scales,<br>model.layers.3.conv.out_proj.weight,<br>model.layers.3.feed_forward.w1.biases,<br>model.layers.3.feed_forward.w1.scales,<br>model.layers.3.feed_forward.w1.weight,<br>model.layers.3.feed_forward.w2.biases,<br>model.layers.3.feed_forward.w2.scales,<br>model.layers.3.feed_forward.w2.weight,<br>model.layers.3.feed_forward.w3.biases,<br>model.layers.3.feed_forward.w3.scales,<br>model.layers.3.feed_forward.w3.weight,<br>model.layers.3.ffn_norm.weight,<br>model.layers.3.operator_norm.weight,<br>model.layers.4.conv.conv.weight,<br>model.layers.4.conv.in_proj.biases,<br>model.layers.4.conv.in_proj.scales,<br>model.layers.4.conv.in_proj.weight,<br>model.layers.4.conv.out_proj.biases,<br>model.layers.4.conv.out_proj.scales,<br>model.layers.4.conv.out_proj.weight,<br>model.layers.4.feed_forward.w1.biases,<br>model.layers.4.feed_forward.w1.scales,<br>model.layers.4.feed_forward.w1.weight,<br>model.layers.4.feed_forward.w2.biases,<br>model.layers.4.feed_forward.w2.scales,<br>model.layers.4.feed_forward.w2.weight,<br>model.layers.4.feed_forward.w3.biases,<br>model.layers.4.feed_forward.w3.scales,<br>model.layers.4.feed_forward.w3.weight,<br>model.layers.4.ffn_norm.weight,<br>model.layers.4.operator_norm.weight,<br>model.layers.5.feed_forward.w1.biases,<br>model.layers.5.feed_forward.w1.scales,<br>model.layers.5.feed_forward.w1.weight,<br>model.layers.5.feed_forward.w2.biases,<br>model.layers.5.feed_forward.w2.scales,<br>model.layers.5.feed_forward.w2.weight,<br>model.layers.5.feed_forward.w3.biases,<br>model.layers.5.feed_forward.w3.scales,<br>model.layers.5.feed_forward.w3.weight,<br>model.layers.5.ffn_norm.weight,<br>model.layers.5.operator_norm.weight,<br>model.layers.5.self_attn.k_layernorm.weight,<br>model.layers.5.self_attn.k_proj.biases,<br>model.layers.5.self_attn.k_proj.scales,<br>model.layers.5.self_attn.k_proj.weight,<br>model.layers.5.self_attn.out_proj.biases,<br>model.layers.5.self_attn.out_proj.scales,<br>model.layers.5.self_attn.out_proj.weight,<br>model.layers.5.self_attn.q_layernorm.weight,<br>model.layers.5.self_attn.q_proj.biases,<br>model.layers.5.self_attn.q_proj.scales,<br>model.layers.5.self_attn.q_proj.weight,<br>model.layers.5.self_attn.v_proj.biases,<br>model.layers.5.self_attn.v_proj.scales,<br>model.layers.5.self_attn.v_proj.weight,<br>model.layers.6.conv.conv.weight,<br>model.layers.6.conv.in_proj.biases,<br>model.layers.6.conv.in_proj.scales,<br>model.layers.6.conv.in_proj.weight,<br>model.layers.6.conv.out_proj.biases,<br>model.layers.6.conv.out_proj.scales,<br>model.layers.6.conv.out_proj.weight,<br>model.layers.6.feed_forward.w1.biases,<br>model.layers.6.feed_forward.w1.scales,<br>model.layers.6.feed_forward.w1.weight,<br>model.layers.6.feed_forward.w2.biases,<br>model.layers.6.feed_forward.w2.scales,<br>model.layers.6.feed_forward.w2.weight,<br>model.layers.6.feed_forward.w3.biases,<br>model.layers.6.feed_forward.w3.scales,<br>model.layers.6.feed_forward.w3.weight,<br>model.layers.6.ffn_norm.weight,<br>model.layers.6.operator_norm.weight,<br>model.layers.7.conv.conv.weight,<br>model.layers.7.conv.in_proj.biases,<br>model.layers.7.conv.in_proj.scales,<br>model.layers.7.conv.in_proj.weight,<br>model.layers.7.conv.out_proj.biases,<br>model.layers.7.conv.out_proj.scales,<br>model.layers.7.conv.out_proj.weight,<br>model.layers.7.feed_forward.w1.biases,<br>model.layers.7.feed_forward.w1.scales,<br>model.layers.7.feed_forward.w1.weight,<br>model.layers.7.feed_forward.w2.biases,<br>model.layers.7.feed_forward.w2.scales,<br>model.layers.7.feed_forward.w2.weight,<br>model.layers.7.feed_forward.w3.biases,<br>model.layers.7.feed_forward.w3.scales,<br>model.layers.7.feed_forward.w3.weight,<br>model.layers.7.ffn_norm.weight,<br>model.layers.7.operator_norm.weight,<br>model.layers.8.conv.conv.weight,<br>model.layers.8.conv.in_proj.biases,<br>model.layers.8.conv.in_proj.scales,<br>model.layers.8.conv.in_proj.weight,<br>model.layers.8.conv.out_proj.biases,<br>model.layers.8.conv.out_proj.scales,<br>model.layers.8.conv.out_proj.weight,<br>model.layers.8.feed_forward.w1.biases,<br>model.layers.8.feed_forward.w1.scales,<br>model.layers.8.feed_forward.w1.weight,<br>model.layers.8.feed_forward.w2.biases,<br>model.layers.8.feed_forward.w2.scales,<br>model.layers.8.feed_forward.w2.weight,<br>model.layers.8.feed_forward.w3.biases,<br>model.layers.8.feed_forward.w3.scales,<br>model.layers.8.feed_forward.w3.weight,<br>model.layers.8.ffn_norm.weight,<br>model.layers.8.operator_norm.weight,<br>model.layers.9.feed_forward.w1.biases,<br>model.layers.9.feed_forward.w1.scales,<br>model.layers.9.feed_forward.w1.weight,<br>model.layers.9.feed_forward.w2.biases,<br>model.layers.9.feed_forward.w2.scales,<br>model.layers.9.feed_forward.w2.weight,<br>model.layers.9.feed_forward.w3.biases,<br>model.layers.9.feed_forward.w3.scales,<br>model.layers.9.feed_forward.w3.weight,<br>model.layers.9.ffn_norm.weight,<br>model.layers.9.operator_norm.weight,<br>model.layers.9.self_attn.k_layernorm.weight,<br>model.layers.9.self_attn.k_proj.biases,<br>model.layers.9.self_attn.k_proj.scales,<br>model.layers.9.self_attn.k_proj.weight,<br>model.layers.9.self_attn.out_proj.biases,<br>model.layers.9.self_attn.out_proj.scales,<br>model.layers.9.self_attn.out_proj.weight,<br>model.layers.9.self_attn.q_layernorm.weight,<br>model.layers.9.self_attn.q_proj.biases,<br>model.layers.9.self_attn.q_proj.scales,<br>model.layers.9.self_attn.q_proj.weight,<br>model.layers.9.self_attn.v_proj.biases,<br>model.layers.9.self_attn.v_proj.scales,<br>model.layers.9.self_attn.v_proj.weight.
- *Root exception type:* ValueError
- *Root exception module:* builtins
- *Root exception message:* Received 600 parameters not in model:
  <br>model.embed_tokens.biases,<br>model.embed_tokens.scales,<br>model.embed_tokens.weight,<br>model.embedding_norm.weight,<br>model.layers.0.conv.conv.weight,<br>model.layers.0.conv.in_proj.biases,<br>model.layers.0.conv.in_proj.scales,<br>model.layers.0.conv.in_proj.weight,<br>model.layers.0.conv.out_proj.biases,<br>model.layers.0.conv.out_proj.scales,<br>model.layers.0.conv.out_proj.weight,<br>model.layers.0.feed_forward.w1.biases,<br>model.layers.0.feed_forward.w1.scales,<br>model.layers.0.feed_forward.w1.weight,<br>model.layers.0.feed_forward.w2.biases,<br>model.layers.0.feed_forward.w2.scales,<br>model.layers.0.feed_forward.w2.weight,<br>model.layers.0.feed_forward.w3.biases,<br>model.layers.0.feed_forward.w3.scales,<br>model.layers.0.feed_forward.w3.weight,<br>model.layers.0.ffn_norm.weight,<br>model.layers.0.operator_norm.weight,<br>model.layers.1.conv.conv.weight,<br>model.layers.1.conv.in_proj.biases,<br>model.layers.1.conv.in_proj.scales,<br>model.layers.1.conv.in_proj.weight,<br>model.layers.1.conv.out_proj.biases,<br>model.layers.1.conv.out_proj.scales,<br>model.layers.1.conv.out_proj.weight,<br>model.layers.1.feed_forward.w1.biases,<br>model.layers.1.feed_forward.w1.scales,<br>model.layers.1.feed_forward.w1.weight,<br>model.layers.1.feed_forward.w2.biases,<br>model.layers.1.feed_forward.w2.scales,<br>model.layers.1.feed_forward.w2.weight,<br>model.layers.1.feed_forward.w3.biases,<br>model.layers.1.feed_forward.w3.scales,<br>model.layers.1.feed_forward.w3.weight,<br>model.layers.1.ffn_norm.weight,<br>model.layers.1.operator_norm.weight,<br>model.layers.10.conv.conv.weight,<br>model.layers.10.conv.in_proj.biases,<br>model.layers.10.conv.in_proj.scales,<br>model.layers.10.conv.in_proj.weight,<br>model.layers.10.conv.out_proj.biases,<br>model.layers.10.conv.out_proj.scales,<br>model.layers.10.conv.out_proj.weight,<br>model.layers.10.feed_forward.w1.biases,<br>model.layers.10.feed_forward.w1.scales,<br>model.layers.10.feed_forward.w1.weight,<br>model.layers.10.feed_forward.w2.biases,<br>model.layers.10.feed_forward.w2.scales,<br>model.layers.10.feed_forward.w2.weight,<br>model.layers.10.feed_forward.w3.biases,<br>model.layers.10.feed_forward.w3.scales,<br>model.layers.10.feed_forward.w3.weight,<br>model.layers.10.ffn_norm.weight,<br>model.layers.10.operator_norm.weight,<br>model.layers.11.conv.conv.weight,<br>model.layers.11.conv.in_proj.biases,<br>model.layers.11.conv.in_proj.scales,<br>model.layers.11.conv.in_proj.weight,<br>model.layers.11.conv.out_proj.biases,<br>model.layers.11.conv.out_proj.scales,<br>model.layers.11.conv.out_proj.weight,<br>model.layers.11.feed_forward.w1.biases,<br>model.layers.11.feed_forward.w1.scales,<br>model.layers.11.feed_forward.w1.weight,<br>model.layers.11.feed_forward.w2.biases,<br>model.layers.11.feed_forward.w2.scales,<br>model.layers.11.feed_forward.w2.weight,<br>model.layers.11.feed_forward.w3.biases,<br>model.layers.11.feed_forward.w3.scales,<br>model.layers.11.feed_forward.w3.weight,<br>model.layers.11.ffn_norm.weight,<br>model.layers.11.operator_norm.weight,<br>model.layers.12.conv.conv.weight,<br>model.layers.12.conv.in_proj.biases,<br>model.layers.12.conv.in_proj.scales,<br>model.layers.12.conv.in_proj.weight,<br>model.layers.12.conv.out_proj.biases,<br>model.layers.12.conv.out_proj.scales,<br>model.layers.12.conv.out_proj.weight,<br>model.layers.12.feed_forward.w1.biases,<br>model.layers.12.feed_forward.w1.scales,<br>model.layers.12.feed_forward.w1.weight,<br>model.layers.12.feed_forward.w2.biases,<br>model.layers.12.feed_forward.w2.scales,<br>model.layers.12.feed_forward.w2.weight,<br>model.layers.12.feed_forward.w3.biases,<br>model.layers.12.feed_forward.w3.scales,<br>model.layers.12.feed_forward.w3.weight,<br>model.layers.12.ffn_norm.weight,<br>model.layers.12.operator_norm.weight,<br>model.layers.13.feed_forward.w1.biases,<br>model.layers.13.feed_forward.w1.scales,<br>model.layers.13.feed_forward.w1.weight,<br>model.layers.13.feed_forward.w2.biases,<br>model.layers.13.feed_forward.w2.scales,<br>model.layers.13.feed_forward.w2.weight,<br>model.layers.13.feed_forward.w3.biases,<br>model.layers.13.feed_forward.w3.scales,<br>model.layers.13.feed_forward.w3.weight,<br>model.layers.13.ffn_norm.weight,<br>model.layers.13.operator_norm.weight,<br>model.layers.13.self_attn.k_layernorm.weight,<br>model.layers.13.self_attn.k_proj.biases,<br>model.layers.13.self_attn.k_proj.scales,<br>model.layers.13.self_attn.k_proj.weight,<br>model.layers.13.self_attn.out_proj.biases,<br>model.layers.13.self_attn.out_proj.scales,<br>model.layers.13.self_attn.out_proj.weight,<br>model.layers.13.self_attn.q_layernorm.weight,<br>model.layers.13.self_attn.q_proj.biases,<br>model.layers.13.self_attn.q_proj.scales,<br>model.layers.13.self_attn.q_proj.weight,<br>model.layers.13.self_attn.v_proj.biases,<br>model.layers.13.self_attn.v_proj.scales,<br>model.layers.13.self_attn.v_proj.weight,<br>model.layers.14.conv.conv.weight,<br>model.layers.14.conv.in_proj.biases,<br>model.layers.14.conv.in_proj.scales,<br>model.layers.14.conv.in_proj.weight,<br>model.layers.14.conv.out_proj.biases,<br>model.layers.14.conv.out_proj.scales,<br>model.layers.14.conv.out_proj.weight,<br>model.layers.14.feed_forward.w1.biases,<br>model.layers.14.feed_forward.w1.scales,<br>model.layers.14.feed_forward.w1.weight,<br>model.layers.14.feed_forward.w2.biases,<br>model.layers.14.feed_forward.w2.scales,<br>model.layers.14.feed_forward.w2.weight,<br>model.layers.14.feed_forward.w3.biases,<br>model.layers.14.feed_forward.w3.scales,<br>model.layers.14.feed_forward.w3.weight,<br>model.layers.14.ffn_norm.weight,<br>model.layers.14.operator_norm.weight,<br>model.layers.15.conv.conv.weight,<br>model.layers.15.conv.in_proj.biases,<br>model.layers.15.conv.in_proj.scales,<br>model.layers.15.conv.in_proj.weight,<br>model.layers.15.conv.out_proj.biases,<br>model.layers.15.conv.out_proj.scales,<br>model.layers.15.conv.out_proj.weight,<br>model.layers.15.feed_forward.w1.biases,<br>model.layers.15.feed_forward.w1.scales,<br>model.layers.15.feed_forward.w1.weight,<br>model.layers.15.feed_forward.w2.biases,<br>model.layers.15.feed_forward.w2.scales,<br>model.layers.15.feed_forward.w2.weight,<br>model.layers.15.feed_forward.w3.biases,<br>model.layers.15.feed_forward.w3.scales,<br>model.layers.15.feed_forward.w3.weight,<br>model.layers.15.ffn_norm.weight,<br>model.layers.15.operator_norm.weight,<br>model.layers.16.conv.conv.weight,<br>model.layers.16.conv.in_proj.biases,<br>model.layers.16.conv.in_proj.scales,<br>model.layers.16.conv.in_proj.weight,<br>model.layers.16.conv.out_proj.biases,<br>model.layers.16.conv.out_proj.scales,<br>model.layers.16.conv.out_proj.weight,<br>model.layers.16.feed_forward.w1.biases,<br>model.layers.16.feed_forward.w1.scales,<br>model.layers.16.feed_forward.w1.weight,<br>model.layers.16.feed_forward.w2.biases,<br>model.layers.16.feed_forward.w2.scales,<br>model.layers.16.feed_forward.w2.weight,<br>model.layers.16.feed_forward.w3.biases,<br>model.layers.16.feed_forward.w3.scales,<br>model.layers.16.feed_forward.w3.weight,<br>model.layers.16.ffn_norm.weight,<br>model.layers.16.operator_norm.weight,<br>model.layers.17.feed_forward.w1.biases,<br>model.layers.17.feed_forward.w1.scales,<br>model.layers.17.feed_forward.w1.weight,<br>model.layers.17.feed_forward.w2.biases,<br>model.layers.17.feed_forward.w2.scales,<br>model.layers.17.feed_forward.w2.weight,<br>model.layers.17.feed_forward.w3.biases,<br>model.layers.17.feed_forward.w3.scales,<br>model.layers.17.feed_forward.w3.weight,<br>model.layers.17.ffn_norm.weight,<br>model.layers.17.operator_norm.weight,<br>model.layers.17.self_attn.k_layernorm.weight,<br>model.layers.17.self_attn.k_proj.biases,<br>model.layers.17.self_attn.k_proj.scales,<br>model.layers.17.self_attn.k_proj.weight,<br>model.layers.17.self_attn.out_proj.biases,<br>model.layers.17.self_attn.out_proj.scales,<br>model.layers.17.self_attn.out_proj.weight,<br>model.layers.17.self_attn.q_layernorm.weight,<br>model.layers.17.self_attn.q_proj.biases,<br>model.layers.17.self_attn.q_proj.scales,<br>model.layers.17.self_attn.q_proj.weight,<br>model.layers.17.self_attn.v_proj.biases,<br>model.layers.17.self_attn.v_proj.scales,<br>model.layers.17.self_attn.v_proj.weight,<br>model.layers.18.conv.conv.weight,<br>model.layers.18.conv.in_proj.biases,<br>model.layers.18.conv.in_proj.scales,<br>model.layers.18.conv.in_proj.weight,<br>model.layers.18.conv.out_proj.biases,<br>model.layers.18.conv.out_proj.scales,<br>model.layers.18.conv.out_proj.weight,<br>model.layers.18.feed_forward.w1.biases,<br>model.layers.18.feed_forward.w1.scales,<br>model.layers.18.feed_forward.w1.weight,<br>model.layers.18.feed_forward.w2.biases,<br>model.layers.18.feed_forward.w2.scales,<br>model.layers.18.feed_forward.w2.weight,<br>model.layers.18.feed_forward.w3.biases,<br>model.layers.18.feed_forward.w3.scales,<br>model.layers.18.feed_forward.w3.weight,<br>model.layers.18.ffn_norm.weight,<br>model.layers.18.operator_norm.weight,<br>model.layers.19.conv.conv.weight,<br>model.layers.19.conv.in_proj.biases,<br>model.layers.19.conv.in_proj.scales,<br>model.layers.19.conv.in_proj.weight,<br>model.layers.19.conv.out_proj.biases,<br>model.layers.19.conv.out_proj.scales,<br>model.layers.19.conv.out_proj.weight,<br>model.layers.19.feed_forward.w1.biases,<br>model.layers.19.feed_forward.w1.scales,<br>model.layers.19.feed_forward.w1.weight,<br>model.layers.19.feed_forward.w2.biases,<br>model.layers.19.feed_forward.w2.scales,<br>model.layers.19.feed_forward.w2.weight,<br>model.layers.19.feed_forward.w3.biases,<br>model.layers.19.feed_forward.w3.scales,<br>model.layers.19.feed_forward.w3.weight,<br>model.layers.19.ffn_norm.weight,<br>model.layers.19.operator_norm.weight,<br>model.layers.2.feed_forward.w1.biases,<br>model.layers.2.feed_forward.w1.scales,<br>model.layers.2.feed_forward.w1.weight,<br>model.layers.2.feed_forward.w2.biases,<br>model.layers.2.feed_forward.w2.scales,<br>model.layers.2.feed_forward.w2.weight,<br>model.layers.2.feed_forward.w3.biases,<br>model.layers.2.feed_forward.w3.scales,<br>model.layers.2.feed_forward.w3.weight,<br>model.layers.2.ffn_norm.weight,<br>model.layers.2.operator_norm.weight,<br>model.layers.2.self_attn.k_layernorm.weight,<br>model.layers.2.self_attn.k_proj.biases,<br>model.layers.2.self_attn.k_proj.scales,<br>model.layers.2.self_attn.k_proj.weight,<br>model.layers.2.self_attn.out_proj.biases,<br>model.layers.2.self_attn.out_proj.scales,<br>model.layers.2.self_attn.out_proj.weight,<br>model.layers.2.self_attn.q_layernorm.weight,<br>model.layers.2.self_attn.q_proj.biases,<br>model.layers.2.self_attn.q_proj.scales,<br>model.layers.2.self_attn.q_proj.weight,<br>model.layers.2.self_attn.v_proj.biases,<br>model.layers.2.self_attn.v_proj.scales,<br>model.layers.2.self_attn.v_proj.weight,<br>model.layers.20.conv.conv.weight,<br>model.layers.20.conv.in_proj.biases,<br>model.layers.20.conv.in_proj.scales,<br>model.layers.20.conv.in_proj.weight,<br>model.layers.20.conv.out_proj.biases,<br>model.layers.20.conv.out_proj.scales,<br>model.layers.20.conv.out_proj.weight,<br>model.layers.20.feed_forward.w1.biases,<br>model.layers.20.feed_forward.w1.scales,<br>model.layers.20.feed_forward.w1.weight,<br>model.layers.20.feed_forward.w2.biases,<br>model.layers.20.feed_forward.w2.scales,<br>model.layers.20.feed_forward.w2.weight,<br>model.layers.20.feed_forward.w3.biases,<br>model.layers.20.feed_forward.w3.scales,<br>model.layers.20.feed_forward.w3.weight,<br>model.layers.20.ffn_norm.weight,<br>model.layers.20.operator_norm.weight,<br>model.layers.21.feed_forward.w1.biases,<br>model.layers.21.feed_forward.w1.scales,<br>model.layers.21.feed_forward.w1.weight,<br>model.layers.21.feed_forward.w2.biases,<br>model.layers.21.feed_forward.w2.scales,<br>model.layers.21.feed_forward.w2.weight,<br>model.layers.21.feed_forward.w3.biases,<br>model.layers.21.feed_forward.w3.scales,<br>model.layers.21.feed_forward.w3.weight,<br>model.layers.21.ffn_norm.weight,<br>model.layers.21.operator_norm.weight,<br>model.layers.21.self_attn.k_layernorm.weight,<br>model.layers.21.self_attn.k_proj.biases,<br>model.layers.21.self_attn.k_proj.scales,<br>model.layers.21.self_attn.k_proj.weight,<br>model.layers.21.self_attn.out_proj.biases,<br>model.layers.21.self_attn.out_proj.scales,<br>model.layers.21.self_attn.out_proj.weight,<br>model.layers.21.self_attn.q_layernorm.weight,<br>model.layers.21.self_attn.q_proj.biases,<br>model.layers.21.self_attn.q_proj.scales,<br>model.layers.21.self_attn.q_proj.weight,<br>model.layers.21.self_attn.v_proj.biases,<br>model.layers.21.self_attn.v_proj.scales,<br>model.layers.21.self_attn.v_proj.weight,<br>model.layers.22.conv.conv.weight,<br>model.layers.22.conv.in_proj.biases,<br>model.layers.22.conv.in_proj.scales,<br>model.layers.22.conv.in_proj.weight,<br>model.layers.22.conv.out_proj.biases,<br>model.layers.22.conv.out_proj.scales,<br>model.layers.22.conv.out_proj.weight,<br>model.layers.22.feed_forward.w1.biases,<br>model.layers.22.feed_forward.w1.scales,<br>model.layers.22.feed_forward.w1.weight,<br>model.layers.22.feed_forward.w2.biases,<br>model.layers.22.feed_forward.w2.scales,<br>model.layers.22.feed_forward.w2.weight,<br>model.layers.22.feed_forward.w3.biases,<br>model.layers.22.feed_forward.w3.scales,<br>model.layers.22.feed_forward.w3.weight,<br>model.layers.22.ffn_norm.weight,<br>model.layers.22.operator_norm.weight,<br>model.layers.23.conv.conv.weight,<br>model.layers.23.conv.in_proj.biases,<br>model.layers.23.conv.in_proj.scales,<br>model.layers.23.conv.in_proj.weight,<br>model.layers.23.conv.out_proj.biases,<br>model.layers.23.conv.out_proj.scales,<br>model.layers.23.conv.out_proj.weight,<br>model.layers.23.feed_forward.w1.biases,<br>model.layers.23.feed_forward.w1.scales,<br>model.layers.23.feed_forward.w1.weight,<br>model.layers.23.feed_forward.w2.biases,<br>model.layers.23.feed_forward.w2.scales,<br>model.layers.23.feed_forward.w2.weight,<br>model.layers.23.feed_forward.w3.biases,<br>model.layers.23.feed_forward.w3.scales,<br>model.layers.23.feed_forward.w3.weight,<br>model.layers.23.ffn_norm.weight,<br>model.layers.23.operator_norm.weight,<br>model.layers.24.feed_forward.w1.biases,<br>model.layers.24.feed_forward.w1.scales,<br>model.layers.24.feed_forward.w1.weight,<br>model.layers.24.feed_forward.w2.biases,<br>model.layers.24.feed_forward.w2.scales,<br>model.layers.24.feed_forward.w2.weight,<br>model.layers.24.feed_forward.w3.biases,<br>model.layers.24.feed_forward.w3.scales,<br>model.layers.24.feed_forward.w3.weight,<br>model.layers.24.ffn_norm.weight,<br>model.layers.24.operator_norm.weight,<br>model.layers.24.self_attn.k_layernorm.weight,<br>model.layers.24.self_attn.k_proj.biases,<br>model.layers.24.self_attn.k_proj.scales,<br>model.layers.24.self_attn.k_proj.weight,<br>model.layers.24.self_attn.out_proj.biases,<br>model.layers.24.self_attn.out_proj.scales,<br>model.layers.24.self_attn.out_proj.weight,<br>model.layers.24.self_attn.q_layernorm.weight,<br>model.layers.24.self_attn.q_proj.biases,<br>model.layers.24.self_attn.q_proj.scales,<br>model.layers.24.self_attn.q_proj.weight,<br>model.layers.24.self_attn.v_proj.biases,<br>model.layers.24.self_attn.v_proj.scales,<br>model.layers.24.self_attn.v_proj.weight,<br>model.layers.25.conv.conv.weight,<br>model.layers.25.conv.in_proj.biases,<br>model.layers.25.conv.in_proj.scales,<br>model.layers.25.conv.in_proj.weight,<br>model.layers.25.conv.out_proj.biases,<br>model.layers.25.conv.out_proj.scales,<br>model.layers.25.conv.out_proj.weight,<br>model.layers.25.feed_forward.w1.biases,<br>model.layers.25.feed_forward.w1.scales,<br>model.layers.25.feed_forward.w1.weight,<br>model.layers.25.feed_forward.w2.biases,<br>model.layers.25.feed_forward.w2.scales,<br>model.layers.25.feed_forward.w2.weight,<br>model.layers.25.feed_forward.w3.biases,<br>model.layers.25.feed_forward.w3.scales,<br>model.layers.25.feed_forward.w3.weight,<br>model.layers.25.ffn_norm.weight,<br>model.layers.25.operator_norm.weight,<br>model.layers.26.conv.conv.weight,<br>model.layers.26.conv.in_proj.biases,<br>model.layers.26.conv.in_proj.scales,<br>model.layers.26.conv.in_proj.weight,<br>model.layers.26.conv.out_proj.biases,<br>model.layers.26.conv.out_proj.scales,<br>model.layers.26.conv.out_proj.weight,<br>model.layers.26.feed_forward.w1.biases,<br>model.layers.26.feed_forward.w1.scales,<br>model.layers.26.feed_forward.w1.weight,<br>model.layers.26.feed_forward.w2.biases,<br>model.layers.26.feed_forward.w2.scales,<br>model.layers.26.feed_forward.w2.weight,<br>model.layers.26.feed_forward.w3.biases,<br>model.layers.26.feed_forward.w3.scales,<br>model.layers.26.feed_forward.w3.weight,<br>model.layers.26.ffn_norm.weight,<br>model.layers.26.operator_norm.weight,<br>model.layers.27.feed_forward.w1.biases,<br>model.layers.27.feed_forward.w1.scales,<br>model.layers.27.feed_forward.w1.weight,<br>model.layers.27.feed_forward.w2.biases,<br>model.layers.27.feed_forward.w2.scales,<br>model.layers.27.feed_forward.w2.weight,<br>model.layers.27.feed_forward.w3.biases,<br>model.layers.27.feed_forward.w3.scales,<br>model.layers.27.feed_forward.w3.weight,<br>model.layers.27.ffn_norm.weight,<br>model.layers.27.operator_norm.weight,<br>model.layers.27.self_attn.k_layernorm.weight,<br>model.layers.27.self_attn.k_proj.biases,<br>model.layers.27.self_attn.k_proj.scales,<br>model.layers.27.self_attn.k_proj.weight,<br>model.layers.27.self_attn.out_proj.biases,<br>model.layers.27.self_attn.out_proj.scales,<br>model.layers.27.self_attn.out_proj.weight,<br>model.layers.27.self_attn.q_layernorm.weight,<br>model.layers.27.self_attn.q_proj.biases,<br>model.layers.27.self_attn.q_proj.scales,<br>model.layers.27.self_attn.q_proj.weight,<br>model.layers.27.self_attn.v_proj.biases,<br>model.layers.27.self_attn.v_proj.scales,<br>model.layers.27.self_attn.v_proj.weight,<br>model.layers.28.conv.conv.weight,<br>model.layers.28.conv.in_proj.biases,<br>model.layers.28.conv.in_proj.scales,<br>model.layers.28.conv.in_proj.weight,<br>model.layers.28.conv.out_proj.biases,<br>model.layers.28.conv.out_proj.scales,<br>model.layers.28.conv.out_proj.weight,<br>model.layers.28.feed_forward.w1.biases,<br>model.layers.28.feed_forward.w1.scales,<br>model.layers.28.feed_forward.w1.weight,<br>model.layers.28.feed_forward.w2.biases,<br>model.layers.28.feed_forward.w2.scales,<br>model.layers.28.feed_forward.w2.weight,<br>model.layers.28.feed_forward.w3.biases,<br>model.layers.28.feed_forward.w3.scales,<br>model.layers.28.feed_forward.w3.weight,<br>model.layers.28.ffn_norm.weight,<br>model.layers.28.operator_norm.weight,<br>model.layers.29.conv.conv.weight,<br>model.layers.29.conv.in_proj.biases,<br>model.layers.29.conv.in_proj.scales,<br>model.layers.29.conv.in_proj.weight,<br>model.layers.29.conv.out_proj.biases,<br>model.layers.29.conv.out_proj.scales,<br>model.layers.29.conv.out_proj.weight,<br>model.layers.29.feed_forward.w1.biases,<br>model.layers.29.feed_forward.w1.scales,<br>model.layers.29.feed_forward.w1.weight,<br>model.layers.29.feed_forward.w2.biases,<br>model.layers.29.feed_forward.w2.scales,<br>model.layers.29.feed_forward.w2.weight,<br>model.layers.29.feed_forward.w3.biases,<br>model.layers.29.feed_forward.w3.scales,<br>model.layers.29.feed_forward.w3.weight,<br>model.layers.29.ffn_norm.weight,<br>model.layers.29.operator_norm.weight,<br>model.layers.3.conv.conv.weight,<br>model.layers.3.conv.in_proj.biases,<br>model.layers.3.conv.in_proj.scales,<br>model.layers.3.conv.in_proj.weight,<br>model.layers.3.conv.out_proj.biases,<br>model.layers.3.conv.out_proj.scales,<br>model.layers.3.conv.out_proj.weight,<br>model.layers.3.feed_forward.w1.biases,<br>model.layers.3.feed_forward.w1.scales,<br>model.layers.3.feed_forward.w1.weight,<br>model.layers.3.feed_forward.w2.biases,<br>model.layers.3.feed_forward.w2.scales,<br>model.layers.3.feed_forward.w2.weight,<br>model.layers.3.feed_forward.w3.biases,<br>model.layers.3.feed_forward.w3.scales,<br>model.layers.3.feed_forward.w3.weight,<br>model.layers.3.ffn_norm.weight,<br>model.layers.3.operator_norm.weight,<br>model.layers.4.conv.conv.weight,<br>model.layers.4.conv.in_proj.biases,<br>model.layers.4.conv.in_proj.scales,<br>model.layers.4.conv.in_proj.weight,<br>model.layers.4.conv.out_proj.biases,<br>model.layers.4.conv.out_proj.scales,<br>model.layers.4.conv.out_proj.weight,<br>model.layers.4.feed_forward.w1.biases,<br>model.layers.4.feed_forward.w1.scales,<br>model.layers.4.feed_forward.w1.weight,<br>model.layers.4.feed_forward.w2.biases,<br>model.layers.4.feed_forward.w2.scales,<br>model.layers.4.feed_forward.w2.weight,<br>model.layers.4.feed_forward.w3.biases,<br>model.layers.4.feed_forward.w3.scales,<br>model.layers.4.feed_forward.w3.weight,<br>model.layers.4.ffn_norm.weight,<br>model.layers.4.operator_norm.weight,<br>model.layers.5.feed_forward.w1.biases,<br>model.layers.5.feed_forward.w1.scales,<br>model.layers.5.feed_forward.w1.weight,<br>model.layers.5.feed_forward.w2.biases,<br>model.layers.5.feed_forward.w2.scales,<br>model.layers.5.feed_forward.w2.weight,<br>model.layers.5.feed_forward.w3.biases,<br>model.layers.5.feed_forward.w3.scales,<br>model.layers.5.feed_forward.w3.weight,<br>model.layers.5.ffn_norm.weight,<br>model.layers.5.operator_norm.weight,<br>model.layers.5.self_attn.k_layernorm.weight,<br>model.layers.5.self_attn.k_proj.biases,<br>model.layers.5.self_attn.k_proj.scales,<br>model.layers.5.self_attn.k_proj.weight,<br>model.layers.5.self_attn.out_proj.biases,<br>model.layers.5.self_attn.out_proj.scales,<br>model.layers.5.self_attn.out_proj.weight,<br>model.layers.5.self_attn.q_layernorm.weight,<br>model.layers.5.self_attn.q_proj.biases,<br>model.layers.5.self_attn.q_proj.scales,<br>model.layers.5.self_attn.q_proj.weight,<br>model.layers.5.self_attn.v_proj.biases,<br>model.layers.5.self_attn.v_proj.scales,<br>model.layers.5.self_attn.v_proj.weight,<br>model.layers.6.conv.conv.weight,<br>model.layers.6.conv.in_proj.biases,<br>model.layers.6.conv.in_proj.scales,<br>model.layers.6.conv.in_proj.weight,<br>model.layers.6.conv.out_proj.biases,<br>model.layers.6.conv.out_proj.scales,<br>model.layers.6.conv.out_proj.weight,<br>model.layers.6.feed_forward.w1.biases,<br>model.layers.6.feed_forward.w1.scales,<br>model.layers.6.feed_forward.w1.weight,<br>model.layers.6.feed_forward.w2.biases,<br>model.layers.6.feed_forward.w2.scales,<br>model.layers.6.feed_forward.w2.weight,<br>model.layers.6.feed_forward.w3.biases,<br>model.layers.6.feed_forward.w3.scales,<br>model.layers.6.feed_forward.w3.weight,<br>model.layers.6.ffn_norm.weight,<br>model.layers.6.operator_norm.weight,<br>model.layers.7.conv.conv.weight,<br>model.layers.7.conv.in_proj.biases,<br>model.layers.7.conv.in_proj.scales,<br>model.layers.7.conv.in_proj.weight,<br>model.layers.7.conv.out_proj.biases,<br>model.layers.7.conv.out_proj.scales,<br>model.layers.7.conv.out_proj.weight,<br>model.layers.7.feed_forward.w1.biases,<br>model.layers.7.feed_forward.w1.scales,<br>model.layers.7.feed_forward.w1.weight,<br>model.layers.7.feed_forward.w2.biases,<br>model.layers.7.feed_forward.w2.scales,<br>model.layers.7.feed_forward.w2.weight,<br>model.layers.7.feed_forward.w3.biases,<br>model.layers.7.feed_forward.w3.scales,<br>model.layers.7.feed_forward.w3.weight,<br>model.layers.7.ffn_norm.weight,<br>model.layers.7.operator_norm.weight,<br>model.layers.8.conv.conv.weight,<br>model.layers.8.conv.in_proj.biases,<br>model.layers.8.conv.in_proj.scales,<br>model.layers.8.conv.in_proj.weight,<br>model.layers.8.conv.out_proj.biases,<br>model.layers.8.conv.out_proj.scales,<br>model.layers.8.conv.out_proj.weight,<br>model.layers.8.feed_forward.w1.biases,<br>model.layers.8.feed_forward.w1.scales,<br>model.layers.8.feed_forward.w1.weight,<br>model.layers.8.feed_forward.w2.biases,<br>model.layers.8.feed_forward.w2.scales,<br>model.layers.8.feed_forward.w2.weight,<br>model.layers.8.feed_forward.w3.biases,<br>model.layers.8.feed_forward.w3.scales,<br>model.layers.8.feed_forward.w3.weight,<br>model.layers.8.ffn_norm.weight,<br>model.layers.8.operator_norm.weight,<br>model.layers.9.feed_forward.w1.biases,<br>model.layers.9.feed_forward.w1.scales,<br>model.layers.9.feed_forward.w1.weight,<br>model.layers.9.feed_forward.w2.biases,<br>model.layers.9.feed_forward.w2.scales,<br>model.layers.9.feed_forward.w2.weight,<br>model.layers.9.feed_forward.w3.biases,<br>model.layers.9.feed_forward.w3.scales,<br>model.layers.9.feed_forward.w3.weight,<br>model.layers.9.ffn_norm.weight,<br>model.layers.9.operator_norm.weight,<br>model.layers.9.self_attn.k_layernorm.weight,<br>model.layers.9.self_attn.k_proj.biases,<br>model.layers.9.self_attn.k_proj.scales,<br>model.layers.9.self_attn.k_proj.weight,<br>model.layers.9.self_attn.out_proj.biases,<br>model.layers.9.self_attn.out_proj.scales,<br>model.layers.9.self_attn.out_proj.weight,<br>model.layers.9.self_attn.q_layernorm.weight,<br>model.layers.9.self_attn.q_proj.biases,<br>model.layers.9.self_attn.q_proj.scales,<br>model.layers.9.self_attn.q_proj.weight,<br>model.layers.9.self_attn.v_proj.biases,<br>model.layers.9.self_attn.v_proj.scales,<br>model.layers.9.self_attn.v_proj.weight.
- *Arch supported by installed mlx-vlm:* yes (model_type lfm2_vl)
- *Model load time:* 0.21s
- *Generation time:* -
- *Total time:* 0.21s
- *Input validation time:* 6.73e-05
- *Prompt preparation time:* -
- *First-token latency:* -
- *Cleanup time:* 0.0778
- *Prompt tokens:* -
- *Generation tokens:* -
- *Total tokens:* -
- *Prompt throughput (raw):* -
- *Generation throughput (raw):* -
- *Peak memory:* -
- *Active memory:* -
- *Cache memory:* -
- *Model-load active memory:* -
- *Post-cleanup active memory:* 0.0041
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
- *Resolved model revision:* 12c5ae49304158b0a133fcea9ba4486a6d6c8cad
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--LFM2.5-VL-3B-OptiQ-4bit/snapshots/12c5ae49304158b0a133fcea9ba4486a6d6c8cad
- *Generation settings:* not captured
- *EOS token:* not captured

#### Complete traceback

```python
Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13505, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12735, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 822, in _typed_mlx_vlm_load
    loaded: tuple[nn.Module, ProcessorMixin] = _mlx_vlm_load(
                                               ~~~~~~~~~~~~~^
        path_or_hf_repo=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1202, in load
    model = load_model(model_path, lazy, strict=strict, **kwargs)
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1067, in load_model
    model.load_weights(list(weights.items()), strict=strict)
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "~/Documents/AI/mlx/mlx/python/mlx/nn/layers/base.py", line 185, in load_weights
    raise ValueError(
        f"Received {num_extra} parameters not in model: \n{extras}."
    )
ValueError: Received 600 parameters not in model: 
model.embed_tokens.biases,
model.embed_tokens.scales,
model.embed_tokens.weight,
model.embedding_norm.weight,
model.layers.0.conv.conv.weight,
model.layers.0.conv.in_proj.biases,
model.layers.0.conv.in_proj.scales,
model.layers.0.conv.in_proj.weight,
model.layers.0.conv.out_proj.biases,
model.layers.0.conv.out_proj.scales,
model.layers.0.conv.out_proj.weight,
model.layers.0.feed_forward.w1.biases,
model.layers.0.feed_forward.w1.scales,
model.layers.0.feed_forward.w1.weight,
model.layers.0.feed_forward.w2.biases,
model.layers.0.feed_forward.w2.scales,
model.layers.0.feed_forward.w2.weight,
model.layers.0.feed_forward.w3.biases,
model.layers.0.feed_forward.w3.scales,
model.layers.0.feed_forward.w3.weight,
model.layers.0.ffn_norm.weight,
model.layers.0.operator_norm.weight,
model.layers.1.conv.conv.weight,
model.layers.1.conv.in_proj.biases,
model.layers.1.conv.in_proj.scales,
model.layers.1.conv.in_proj.weight,
model.layers.1.conv.out_proj.biases,
model.layers.1.conv.out_proj.scales,
model.layers.1.conv.out_proj.weight,
model.layers.1.feed_forward.w1.biases,
model.layers.1.feed_forward.w1.scales,
model.layers.1.feed_forward.w1.weight,
model.layers.1.feed_forward.w2.biases,
model.layers.1.feed_forward.w2.scales,
model.layers.1.feed_forward.w2.weight,
model.layers.1.feed_forward.w3.biases,
model.layers.1.feed_forward.w3.scales,
model.layers.1.feed_forward.w3.weight,
model.layers.1.ffn_norm.weight,
model.layers.1.operator_norm.weight,
model.layers.10.conv.conv.weight,
model.layers.10.conv.in_proj.biases,
model.layers.10.conv.in_proj.scales,
model.layers.10.conv.in_proj.weight,
model.layers.10.conv.out_proj.biases,
model.layers.10.conv.out_proj.scales,
model.layers.10.conv.out_proj.weight,
model.layers.10.feed_forward.w1.biases,
model.layers.10.feed_forward.w1.scales,
model.layers.10.feed_forward.w1.weight,
model.layers.10.feed_forward.w2.biases,
model.layers.10.feed_forward.w2.scales,
model.layers.10.feed_forward.w2.weight,
model.layers.10.feed_forward.w3.biases,
model.layers.10.feed_forward.w3.scales,
model.layers.10.feed_forward.w3.weight,
model.layers.10.ffn_norm.weight,
model.layers.10.operator_norm.weight,
model.layers.11.conv.conv.weight,
model.layers.11.conv.in_proj.biases,
model.layers.11.conv.in_proj.scales,
model.layers.11.conv.in_proj.weight,
model.layers.11.conv.out_proj.biases,
model.layers.11.conv.out_proj.scales,
model.layers.11.conv.out_proj.weight,
model.layers.11.feed_forward.w1.biases,
model.layers.11.feed_forward.w1.scales,
model.layers.11.feed_forward.w1.weight,
model.layers.11.feed_forward.w2.biases,
model.layers.11.feed_forward.w2.scales,
model.layers.11.feed_forward.w2.weight,
model.layers.11.feed_forward.w3.biases,
model.layers.11.feed_forward.w3.scales,
model.layers.11.feed_forward.w3.weight,
model.layers.11.ffn_norm.weight,
model.layers.11.operator_norm.weight,
model.layers.12.conv.conv.weight,
model.layers.12.conv.in_proj.biases,
model.layers.12.conv.in_proj.scales,
model.layers.12.conv.in_proj.weight,
model.layers.12.conv.out_proj.biases,
model.layers.12.conv.out_proj.scales,
model.layers.12.conv.out_proj.weight,
model.layers.12.feed_forward.w1.biases,
model.layers.12.feed_forward.w1.scales,
model.layers.12.feed_forward.w1.weight,
model.layers.12.feed_forward.w2.biases,
model.layers.12.feed_forward.w2.scales,
model.layers.12.feed_forward.w2.weight,
model.layers.12.feed_forward.w3.biases,
model.layers.12.feed_forward.w3.scales,
model.layers.12.feed_forward.w3.weight,
model.layers.12.ffn_norm.weight,
model.layers.12.operator_norm.weight,
model.layers.13.feed_forward.w1.biases,
model.layers.13.feed_forward.w1.scales,
model.layers.13.feed_forward.w1.weight,
model.layers.13.feed_forward.w2.biases,
model.layers.13.feed_forward.w2.scales,
model.layers.13.feed_forward.w2.weight,
model.layers.13.feed_forward.w3.biases,
model.layers.13.feed_forward.w3.scales,
model.layers.13.feed_forward.w3.weight,
model.layers.13.ffn_norm.weight,
model.layers.13.operator_norm.weight,
model.layers.13.self_attn.k_layernorm.weight,
model.layers.13.self_attn.k_proj.biases,
model.layers.13.self_attn.k_proj.scales,
model.layers.13.self_attn.k_proj.weight,
model.layers.13.self_attn.out_proj.biases,
model.layers.13.self_attn.out_proj.scales,
model.layers.13.self_attn.out_proj.weight,
model.layers.13.self_attn.q_layernorm.weight,
model.layers.13.self_attn.q_proj.biases,
model.layers.13.self_attn.q_proj.scales,
model.layers.13.self_attn.q_proj.weight,
model.layers.13.self_attn.v_proj.biases,
model.layers.13.self_attn.v_proj.scales,
model.layers.13.self_attn.v_proj.weight,
model.layers.14.conv.conv.weight,
model.layers.14.conv.in_proj.biases,
model.layers.14.conv.in_proj.scales,
model.layers.14.conv.in_proj.weight,
model.layers.14.conv.out_proj.biases,
model.layers.14.conv.out_proj.scales,
model.layers.14.conv.out_proj.weight,
model.layers.14.feed_forward.w1.biases,
model.layers.14.feed_forward.w1.scales,
model.layers.14.feed_forward.w1.weight,
model.layers.14.feed_forward.w2.biases,
model.layers.14.feed_forward.w2.scales,
model.layers.14.feed_forward.w2.weight,
model.layers.14.feed_forward.w3.biases,
model.layers.14.feed_forward.w3.scales,
model.layers.14.feed_forward.w3.weight,
model.layers.14.ffn_norm.weight,
model.layers.14.operator_norm.weight,
model.layers.15.conv.conv.weight,
model.layers.15.conv.in_proj.biases,
model.layers.15.conv.in_proj.scales,
model.layers.15.conv.in_proj.weight,
model.layers.15.conv.out_proj.biases,
model.layers.15.conv.out_proj.scales,
model.layers.15.conv.out_proj.weight,
model.layers.15.feed_forward.w1.biases,
model.layers.15.feed_forward.w1.scales,
model.layers.15.feed_forward.w1.weight,
model.layers.15.feed_forward.w2.biases,
model.layers.15.feed_forward.w2.scales,
model.layers.15.feed_forward.w2.weight,
model.layers.15.feed_forward.w3.biases,
model.layers.15.feed_forward.w3.scales,
model.layers.15.feed_forward.w3.weight,
model.layers.15.ffn_norm.weight,
model.layers.15.operator_norm.weight,
model.layers.16.conv.conv.weight,
model.layers.16.conv.in_proj.biases,
model.layers.16.conv.in_proj.scales,
model.layers.16.conv.in_proj.weight,
model.layers.16.conv.out_proj.biases,
model.layers.16.conv.out_proj.scales,
model.layers.16.conv.out_proj.weight,
model.layers.16.feed_forward.w1.biases,
model.layers.16.feed_forward.w1.scales,
model.layers.16.feed_forward.w1.weight,
model.layers.16.feed_forward.w2.biases,
model.layers.16.feed_forward.w2.scales,
model.layers.16.feed_forward.w2.weight,
model.layers.16.feed_forward.w3.biases,
model.layers.16.feed_forward.w3.scales,
model.layers.16.feed_forward.w3.weight,
model.layers.16.ffn_norm.weight,
model.layers.16.operator_norm.weight,
model.layers.17.feed_forward.w1.biases,
model.layers.17.feed_forward.w1.scales,
model.layers.17.feed_forward.w1.weight,
model.layers.17.feed_forward.w2.biases,
model.layers.17.feed_forward.w2.scales,
model.layers.17.feed_forward.w2.weight,
model.layers.17.feed_forward.w3.biases,
model.layers.17.feed_forward.w3.scales,
model.layers.17.feed_forward.w3.weight,
model.layers.17.ffn_norm.weight,
model.layers.17.operator_norm.weight,
model.layers.17.self_attn.k_layernorm.weight,
model.layers.17.self_attn.k_proj.biases,
model.layers.17.self_attn.k_proj.scales,
model.layers.17.self_attn.k_proj.weight,
model.layers.17.self_attn.out_proj.biases,
model.layers.17.self_attn.out_proj.scales,
model.layers.17.self_attn.out_proj.weight,
model.layers.17.self_attn.q_layernorm.weight,
model.layers.17.self_attn.q_proj.biases,
model.layers.17.self_attn.q_proj.scales,
model.layers.17.self_attn.q_proj.weight,
model.layers.17.self_attn.v_proj.biases,
model.layers.17.self_attn.v_proj.scales,
model.layers.17.self_attn.v_proj.weight,
model.layers.18.conv.conv.weight,
model.layers.18.conv.in_proj.biases,
model.layers.18.conv.in_proj.scales,
model.layers.18.conv.in_proj.weight,
model.layers.18.conv.out_proj.biases,
model.layers.18.conv.out_proj.scales,
model.layers.18.conv.out_proj.weight,
model.layers.18.feed_forward.w1.biases,
model.layers.18.feed_forward.w1.scales,
model.layers.18.feed_forward.w1.weight,
model.layers.18.feed_forward.w2.biases,
model.layers.18.feed_forward.w2.scales,
model.layers.18.feed_forward.w2.weight,
model.layers.18.feed_forward.w3.biases,
model.layers.18.feed_forward.w3.scales,
model.layers.18.feed_forward.w3.weight,
model.layers.18.ffn_norm.weight,
model.layers.18.operator_norm.weight,
model.layers.19.conv.conv.weight,
model.layers.19.conv.in_proj.biases,
model.layers.19.conv.in_proj.scales,
model.layers.19.conv.in_proj.weight,
model.layers.19.conv.out_proj.biases,
model.layers.19.conv.out_proj.scales,
model.layers.19.conv.out_proj.weight,
model.layers.19.feed_forward.w1.biases,
model.layers.19.feed_forward.w1.scales,
model.layers.19.feed_forward.w1.weight,
model.layers.19.feed_forward.w2.biases,
model.layers.19.feed_forward.w2.scales,
model.layers.19.feed_forward.w2.weight,
model.layers.19.feed_forward.w3.biases,
model.layers.19.feed_forward.w3.scales,
model.layers.19.feed_forward.w3.weight,
model.layers.19.ffn_norm.weight,
model.layers.19.operator_norm.weight,
model.layers.2.feed_forward.w1.biases,
model.layers.2.feed_forward.w1.scales,
model.layers.2.feed_forward.w1.weight,
model.layers.2.feed_forward.w2.biases,
model.layers.2.feed_forward.w2.scales,
model.layers.2.feed_forward.w2.weight,
model.layers.2.feed_forward.w3.biases,
model.layers.2.feed_forward.w3.scales,
model.layers.2.feed_forward.w3.weight,
model.layers.2.ffn_norm.weight,
model.layers.2.operator_norm.weight,
model.layers.2.self_attn.k_layernorm.weight,
model.layers.2.self_attn.k_proj.biases,
model.layers.2.self_attn.k_proj.scales,
model.layers.2.self_attn.k_proj.weight,
model.layers.2.self_attn.out_proj.biases,
model.layers.2.self_attn.out_proj.scales,
model.layers.2.self_attn.out_proj.weight,
model.layers.2.self_attn.q_layernorm.weight,
model.layers.2.self_attn.q_proj.biases,
model.layers.2.self_attn.q_proj.scales,
model.layers.2.self_attn.q_proj.weight,
model.layers.2.self_attn.v_proj.biases,
model.layers.2.self_attn.v_proj.scales,
model.layers.2.self_attn.v_proj.weight,
model.layers.20.conv.conv.weight,
model.layers.20.conv.in_proj.biases,
model.layers.20.conv.in_proj.scales,
model.layers.20.conv.in_proj.weight,
model.layers.20.conv.out_proj.biases,
model.layers.20.conv.out_proj.scales,
model.layers.20.conv.out_proj.weight,
model.layers.20.feed_forward.w1.biases,
model.layers.20.feed_forward.w1.scales,
model.layers.20.feed_forward.w1.weight,
model.layers.20.feed_forward.w2.biases,
model.layers.20.feed_forward.w2.scales,
model.layers.20.feed_forward.w2.weight,
model.layers.20.feed_forward.w3.biases,
model.layers.20.feed_forward.w3.scales,
model.layers.20.feed_forward.w3.weight,
model.layers.20.ffn_norm.weight,
model.layers.20.operator_norm.weight,
model.layers.21.feed_forward.w1.biases,
model.layers.21.feed_forward.w1.scales,
model.layers.21.feed_forward.w1.weight,
model.layers.21.feed_forward.w2.biases,
model.layers.21.feed_forward.w2.scales,
model.layers.21.feed_forward.w2.weight,
model.layers.21.feed_forward.w3.biases,
model.layers.21.feed_forward.w3.scales,
model.layers.21.feed_forward.w3.weight,
model.layers.21.ffn_norm.weight,
model.layers.21.operator_norm.weight,
model.layers.21.self_attn.k_layernorm.weight,
model.layers.21.self_attn.k_proj.biases,
model.layers.21.self_attn.k_proj.scales,
model.layers.21.self_attn.k_proj.weight,
model.layers.21.self_attn.out_proj.biases,
model.layers.21.self_attn.out_proj.scales,
model.layers.21.self_attn.out_proj.weight,
model.layers.21.self_attn.q_layernorm.weight,
model.layers.21.self_attn.q_proj.biases,
model.layers.21.self_attn.q_proj.scales,
model.layers.21.self_attn.q_proj.weight,
model.layers.21.self_attn.v_proj.biases,
model.layers.21.self_attn.v_proj.scales,
model.layers.21.self_attn.v_proj.weight,
model.layers.22.conv.conv.weight,
model.layers.22.conv.in_proj.biases,
model.layers.22.conv.in_proj.scales,
model.layers.22.conv.in_proj.weight,
model.layers.22.conv.out_proj.biases,
model.layers.22.conv.out_proj.scales,
model.layers.22.conv.out_proj.weight,
model.layers.22.feed_forward.w1.biases,
model.layers.22.feed_forward.w1.scales,
model.layers.22.feed_forward.w1.weight,
model.layers.22.feed_forward.w2.biases,
model.layers.22.feed_forward.w2.scales,
model.layers.22.feed_forward.w2.weight,
model.layers.22.feed_forward.w3.biases,
model.layers.22.feed_forward.w3.scales,
model.layers.22.feed_forward.w3.weight,
model.layers.22.ffn_norm.weight,
model.layers.22.operator_norm.weight,
model.layers.23.conv.conv.weight,
model.layers.23.conv.in_proj.biases,
model.layers.23.conv.in_proj.scales,
model.layers.23.conv.in_proj.weight,
model.layers.23.conv.out_proj.biases,
model.layers.23.conv.out_proj.scales,
model.layers.23.conv.out_proj.weight,
model.layers.23.feed_forward.w1.biases,
model.layers.23.feed_forward.w1.scales,
model.layers.23.feed_forward.w1.weight,
model.layers.23.feed_forward.w2.biases,
model.layers.23.feed_forward.w2.scales,
model.layers.23.feed_forward.w2.weight,
model.layers.23.feed_forward.w3.biases,
model.layers.23.feed_forward.w3.scales,
model.layers.23.feed_forward.w3.weight,
model.layers.23.ffn_norm.weight,
model.layers.23.operator_norm.weight,
model.layers.24.feed_forward.w1.biases,
model.layers.24.feed_forward.w1.scales,
model.layers.24.feed_forward.w1.weight,
model.layers.24.feed_forward.w2.biases,
model.layers.24.feed_forward.w2.scales,
model.layers.24.feed_forward.w2.weight,
model.layers.24.feed_forward.w3.biases,
model.layers.24.feed_forward.w3.scales,
model.layers.24.feed_forward.w3.weight,
model.layers.24.ffn_norm.weight,
model.layers.24.operator_norm.weight,
model.layers.24.self_attn.k_layernorm.weight,
model.layers.24.self_attn.k_proj.biases,
model.layers.24.self_attn.k_proj.scales,
model.layers.24.self_attn.k_proj.weight,
model.layers.24.self_attn.out_proj.biases,
model.layers.24.self_attn.out_proj.scales,
model.layers.24.self_attn.out_proj.weight,
model.layers.24.self_attn.q_layernorm.weight,
model.layers.24.self_attn.q_proj.biases,
model.layers.24.self_attn.q_proj.scales,
model.layers.24.self_attn.q_proj.weight,
model.layers.24.self_attn.v_proj.biases,
model.layers.24.self_attn.v_proj.scales,
model.layers.24.self_attn.v_proj.weight,
model.layers.25.conv.conv.weight,
model.layers.25.conv.in_proj.biases,
model.layers.25.conv.in_proj.scales,
model.layers.25.conv.in_proj.weight,
model.layers.25.conv.out_proj.biases,
model.layers.25.conv.out_proj.scales,
model.layers.25.conv.out_proj.weight,
model.layers.25.feed_forward.w1.biases,
model.layers.25.feed_forward.w1.scales,
model.layers.25.feed_forward.w1.weight,
model.layers.25.feed_forward.w2.biases,
model.layers.25.feed_forward.w2.scales,
model.layers.25.feed_forward.w2.weight,
model.layers.25.feed_forward.w3.biases,
model.layers.25.feed_forward.w3.scales,
model.layers.25.feed_forward.w3.weight,
model.layers.25.ffn_norm.weight,
model.layers.25.operator_norm.weight,
model.layers.26.conv.conv.weight,
model.layers.26.conv.in_proj.biases,
model.layers.26.conv.in_proj.scales,
model.layers.26.conv.in_proj.weight,
model.layers.26.conv.out_proj.biases,
model.layers.26.conv.out_proj.scales,
model.layers.26.conv.out_proj.weight,
model.layers.26.feed_forward.w1.biases,
model.layers.26.feed_forward.w1.scales,
model.layers.26.feed_forward.w1.weight,
model.layers.26.feed_forward.w2.biases,
model.layers.26.feed_forward.w2.scales,
model.layers.26.feed_forward.w2.weight,
model.layers.26.feed_forward.w3.biases,
model.layers.26.feed_forward.w3.scales,
model.layers.26.feed_forward.w3.weight,
model.layers.26.ffn_norm.weight,
model.layers.26.operator_norm.weight,
model.layers.27.feed_forward.w1.biases,
model.layers.27.feed_forward.w1.scales,
model.layers.27.feed_forward.w1.weight,
model.layers.27.feed_forward.w2.biases,
model.layers.27.feed_forward.w2.scales,
model.layers.27.feed_forward.w2.weight,
model.layers.27.feed_forward.w3.biases,
model.layers.27.feed_forward.w3.scales,
model.layers.27.feed_forward.w3.weight,
model.layers.27.ffn_norm.weight,
model.layers.27.operator_norm.weight,
model.layers.27.self_attn.k_layernorm.weight,
model.layers.27.self_attn.k_proj.biases,
model.layers.27.self_attn.k_proj.scales,
model.layers.27.self_attn.k_proj.weight,
model.layers.27.self_attn.out_proj.biases,
model.layers.27.self_attn.out_proj.scales,
model.layers.27.self_attn.out_proj.weight,
model.layers.27.self_attn.q_layernorm.weight,
model.layers.27.self_attn.q_proj.biases,
model.layers.27.self_attn.q_proj.scales,
model.layers.27.self_attn.q_proj.weight,
model.layers.27.self_attn.v_proj.biases,
model.layers.27.self_attn.v_proj.scales,
model.layers.27.self_attn.v_proj.weight,
model.layers.28.conv.conv.weight,
model.layers.28.conv.in_proj.biases,
model.layers.28.conv.in_proj.scales,
model.layers.28.conv.in_proj.weight,
model.layers.28.conv.out_proj.biases,
model.layers.28.conv.out_proj.scales,
model.layers.28.conv.out_proj.weight,
model.layers.28.feed_forward.w1.biases,
model.layers.28.feed_forward.w1.scales,
model.layers.28.feed_forward.w1.weight,
model.layers.28.feed_forward.w2.biases,
model.layers.28.feed_forward.w2.scales,
model.layers.28.feed_forward.w2.weight,
model.layers.28.feed_forward.w3.biases,
model.layers.28.feed_forward.w3.scales,
model.layers.28.feed_forward.w3.weight,
model.layers.28.ffn_norm.weight,
model.layers.28.operator_norm.weight,
model.layers.29.conv.conv.weight,
model.layers.29.conv.in_proj.biases,
model.layers.29.conv.in_proj.scales,
model.layers.29.conv.in_proj.weight,
model.layers.29.conv.out_proj.biases,
model.layers.29.conv.out_proj.scales,
model.layers.29.conv.out_proj.weight,
model.layers.29.feed_forward.w1.biases,
model.layers.29.feed_forward.w1.scales,
model.layers.29.feed_forward.w1.weight,
model.layers.29.feed_forward.w2.biases,
model.layers.29.feed_forward.w2.scales,
model.layers.29.feed_forward.w2.weight,
model.layers.29.feed_forward.w3.biases,
model.layers.29.feed_forward.w3.scales,
model.layers.29.feed_forward.w3.weight,
model.layers.29.ffn_norm.weight,
model.layers.29.operator_norm.weight,
model.layers.3.conv.conv.weight,
model.layers.3.conv.in_proj.biases,
model.layers.3.conv.in_proj.scales,
model.layers.3.conv.in_proj.weight,
model.layers.3.conv.out_proj.biases,
model.layers.3.conv.out_proj.scales,
model.layers.3.conv.out_proj.weight,
model.layers.3.feed_forward.w1.biases,
model.layers.3.feed_forward.w1.scales,
model.layers.3.feed_forward.w1.weight,
model.layers.3.feed_forward.w2.biases,
model.layers.3.feed_forward.w2.scales,
model.layers.3.feed_forward.w2.weight,
model.layers.3.feed_forward.w3.biases,
model.layers.3.feed_forward.w3.scales,
model.layers.3.feed_forward.w3.weight,
model.layers.3.ffn_norm.weight,
model.layers.3.operator_norm.weight,
model.layers.4.conv.conv.weight,
model.layers.4.conv.in_proj.biases,
model.layers.4.conv.in_proj.scales,
model.layers.4.conv.in_proj.weight,
model.layers.4.conv.out_proj.biases,
model.layers.4.conv.out_proj.scales,
model.layers.4.conv.out_proj.weight,
model.layers.4.feed_forward.w1.biases,
model.layers.4.feed_forward.w1.scales,
model.layers.4.feed_forward.w1.weight,
model.layers.4.feed_forward.w2.biases,
model.layers.4.feed_forward.w2.scales,
model.layers.4.feed_forward.w2.weight,
model.layers.4.feed_forward.w3.biases,
model.layers.4.feed_forward.w3.scales,
model.layers.4.feed_forward.w3.weight,
model.layers.4.ffn_norm.weight,
model.layers.4.operator_norm.weight,
model.layers.5.feed_forward.w1.biases,
model.layers.5.feed_forward.w1.scales,
model.layers.5.feed_forward.w1.weight,
model.layers.5.feed_forward.w2.biases,
model.layers.5.feed_forward.w2.scales,
model.layers.5.feed_forward.w2.weight,
model.layers.5.feed_forward.w3.biases,
model.layers.5.feed_forward.w3.scales,
model.layers.5.feed_forward.w3.weight,
model.layers.5.ffn_norm.weight,
model.layers.5.operator_norm.weight,
model.layers.5.self_attn.k_layernorm.weight,
model.layers.5.self_attn.k_proj.biases,
model.layers.5.self_attn.k_proj.scales,
model.layers.5.self_attn.k_proj.weight,
model.layers.5.self_attn.out_proj.biases,
model.layers.5.self_attn.out_proj.scales,
model.layers.5.self_attn.out_proj.weight,
model.layers.5.self_attn.q_layernorm.weight,
model.layers.5.self_attn.q_proj.biases,
model.layers.5.self_attn.q_proj.scales,
model.layers.5.self_attn.q_proj.weight,
model.layers.5.self_attn.v_proj.biases,
model.layers.5.self_attn.v_proj.scales,
model.layers.5.self_attn.v_proj.weight,
model.layers.6.conv.conv.weight,
model.layers.6.conv.in_proj.biases,
model.layers.6.conv.in_proj.scales,
model.layers.6.conv.in_proj.weight,
model.layers.6.conv.out_proj.biases,
model.layers.6.conv.out_proj.scales,
model.layers.6.conv.out_proj.weight,
model.layers.6.feed_forward.w1.biases,
model.layers.6.feed_forward.w1.scales,
model.layers.6.feed_forward.w1.weight,
model.layers.6.feed_forward.w2.biases,
model.layers.6.feed_forward.w2.scales,
model.layers.6.feed_forward.w2.weight,
model.layers.6.feed_forward.w3.biases,
model.layers.6.feed_forward.w3.scales,
model.layers.6.feed_forward.w3.weight,
model.layers.6.ffn_norm.weight,
model.layers.6.operator_norm.weight,
model.layers.7.conv.conv.weight,
model.layers.7.conv.in_proj.biases,
model.layers.7.conv.in_proj.scales,
model.layers.7.conv.in_proj.weight,
model.layers.7.conv.out_proj.biases,
model.layers.7.conv.out_proj.scales,
model.layers.7.conv.out_proj.weight,
model.layers.7.feed_forward.w1.biases,
model.layers.7.feed_forward.w1.scales,
model.layers.7.feed_forward.w1.weight,
model.layers.7.feed_forward.w2.biases,
model.layers.7.feed_forward.w2.scales,
model.layers.7.feed_forward.w2.weight,
model.layers.7.feed_forward.w3.biases,
model.layers.7.feed_forward.w3.scales,
model.layers.7.feed_forward.w3.weight,
model.layers.7.ffn_norm.weight,
model.layers.7.operator_norm.weight,
model.layers.8.conv.conv.weight,
model.layers.8.conv.in_proj.biases,
model.layers.8.conv.in_proj.scales,
model.layers.8.conv.in_proj.weight,
model.layers.8.conv.out_proj.biases,
model.layers.8.conv.out_proj.scales,
model.layers.8.conv.out_proj.weight,
model.layers.8.feed_forward.w1.biases,
model.layers.8.feed_forward.w1.scales,
model.layers.8.feed_forward.w1.weight,
model.layers.8.feed_forward.w2.biases,
model.layers.8.feed_forward.w2.scales,
model.layers.8.feed_forward.w2.weight,
model.layers.8.feed_forward.w3.biases,
model.layers.8.feed_forward.w3.scales,
model.layers.8.feed_forward.w3.weight,
model.layers.8.ffn_norm.weight,
model.layers.8.operator_norm.weight,
model.layers.9.feed_forward.w1.biases,
model.layers.9.feed_forward.w1.scales,
model.layers.9.feed_forward.w1.weight,
model.layers.9.feed_forward.w2.biases,
model.layers.9.feed_forward.w2.scales,
model.layers.9.feed_forward.w2.weight,
model.layers.9.feed_forward.w3.biases,
model.layers.9.feed_forward.w3.scales,
model.layers.9.feed_forward.w3.weight,
model.layers.9.ffn_norm.weight,
model.layers.9.operator_norm.weight,
model.layers.9.self_attn.k_layernorm.weight,
model.layers.9.self_attn.k_proj.biases,
model.layers.9.self_attn.k_proj.scales,
model.layers.9.self_attn.k_proj.weight,
model.layers.9.self_attn.out_proj.biases,
model.layers.9.self_attn.out_proj.scales,
model.layers.9.self_attn.out_proj.weight,
model.layers.9.self_attn.q_layernorm.weight,
model.layers.9.self_attn.q_proj.biases,
model.layers.9.self_attn.q_proj.scales,
model.layers.9.self_attn.q_proj.weight,
model.layers.9.self_attn.v_proj.biases,
model.layers.9.self_attn.v_proj.scales,
model.layers.9.self_attn.v_proj.weight.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14520, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13520, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: Received 600 parameters not in model: 
model.embed_tokens.biases,
model.embed_tokens.scales,
model.embed_tokens.weight,
model.embedding_norm.weight,
model.layers.0.conv.conv.weight,
model.layers.0.conv.in_proj.biases,
model.layers.0.conv.in_proj.scales,
model.layers.0.conv.in_proj.weight,
model.layers.0.conv.out_proj.biases,
model.layers.0.conv.out_proj.scales,
model.layers.0.conv.out_proj.weight,
model.layers.0.feed_forward.w1.biases,
model.layers.0.feed_forward.w1.scales,
model.layers.0.feed_forward.w1.weight,
model.layers.0.feed_forward.w2.biases,
model.layers.0.feed_forward.w2.scales,
model.layers.0.feed_forward.w2.weight,
model.layers.0.feed_forward.w3.biases,
model.layers.0.feed_forward.w3.scales,
model.layers.0.feed_forward.w3.weight,
model.layers.0.ffn_norm.weight,
model.layers.0.operator_norm.weight,
model.layers.1.conv.conv.weight,
model.layers.1.conv.in_proj.biases,
model.layers.1.conv.in_proj.scales,
model.layers.1.conv.in_proj.weight,
model.layers.1.conv.out_proj.biases,
model.layers.1.conv.out_proj.scales,
model.layers.1.conv.out_proj.weight,
model.layers.1.feed_forward.w1.biases,
model.layers.1.feed_forward.w1.scales,
model.layers.1.feed_forward.w1.weight,
model.layers.1.feed_forward.w2.biases,
model.layers.1.feed_forward.w2.scales,
model.layers.1.feed_forward.w2.weight,
model.layers.1.feed_forward.w3.biases,
model.layers.1.feed_forward.w3.scales,
model.layers.1.feed_forward.w3.weight,
model.layers.1.ffn_norm.weight,
model.layers.1.operator_norm.weight,
model.layers.10.conv.conv.weight,
model.layers.10.conv.in_proj.biases,
model.layers.10.conv.in_proj.scales,
model.layers.10.conv.in_proj.weight,
model.layers.10.conv.out_proj.biases,
model.layers.10.conv.out_proj.scales,
model.layers.10.conv.out_proj.weight,
model.layers.10.feed_forward.w1.biases,
model.layers.10.feed_forward.w1.scales,
model.layers.10.feed_forward.w1.weight,
model.layers.10.feed_forward.w2.biases,
model.layers.10.feed_forward.w2.scales,
model.layers.10.feed_forward.w2.weight,
model.layers.10.feed_forward.w3.biases,
model.layers.10.feed_forward.w3.scales,
model.layers.10.feed_forward.w3.weight,
model.layers.10.ffn_norm.weight,
model.layers.10.operator_norm.weight,
model.layers.11.conv.conv.weight,
model.layers.11.conv.in_proj.biases,
model.layers.11.conv.in_proj.scales,
model.layers.11.conv.in_proj.weight,
model.layers.11.conv.out_proj.biases,
model.layers.11.conv.out_proj.scales,
model.layers.11.conv.out_proj.weight,
model.layers.11.feed_forward.w1.biases,
model.layers.11.feed_forward.w1.scales,
model.layers.11.feed_forward.w1.weight,
model.layers.11.feed_forward.w2.biases,
model.layers.11.feed_forward.w2.scales,
model.layers.11.feed_forward.w2.weight,
model.layers.11.feed_forward.w3.biases,
model.layers.11.feed_forward.w3.scales,
model.layers.11.feed_forward.w3.weight,
model.layers.11.ffn_norm.weight,
model.layers.11.operator_norm.weight,
model.layers.12.conv.conv.weight,
model.layers.12.conv.in_proj.biases,
model.layers.12.conv.in_proj.scales,
model.layers.12.conv.in_proj.weight,
model.layers.12.conv.out_proj.biases,
model.layers.12.conv.out_proj.scales,
model.layers.12.conv.out_proj.weight,
model.layers.12.feed_forward.w1.biases,
model.layers.12.feed_forward.w1.scales,
model.layers.12.feed_forward.w1.weight,
model.layers.12.feed_forward.w2.biases,
model.layers.12.feed_forward.w2.scales,
model.layers.12.feed_forward.w2.weight,
model.layers.12.feed_forward.w3.biases,
model.layers.12.feed_forward.w3.scales,
model.layers.12.feed_forward.w3.weight,
model.layers.12.ffn_norm.weight,
model.layers.12.operator_norm.weight,
model.layers.13.feed_forward.w1.biases,
model.layers.13.feed_forward.w1.scales,
model.layers.13.feed_forward.w1.weight,
model.layers.13.feed_forward.w2.biases,
model.layers.13.feed_forward.w2.scales,
model.layers.13.feed_forward.w2.weight,
model.layers.13.feed_forward.w3.biases,
model.layers.13.feed_forward.w3.scales,
model.layers.13.feed_forward.w3.weight,
model.layers.13.ffn_norm.weight,
model.layers.13.operator_norm.weight,
model.layers.13.self_attn.k_layernorm.weight,
model.layers.13.self_attn.k_proj.biases,
model.layers.13.self_attn.k_proj.scales,
model.layers.13.self_attn.k_proj.weight,
model.layers.13.self_attn.out_proj.biases,
model.layers.13.self_attn.out_proj.scales,
model.layers.13.self_attn.out_proj.weight,
model.layers.13.self_attn.q_layernorm.weight,
model.layers.13.self_attn.q_proj.biases,
model.layers.13.self_attn.q_proj.scales,
model.layers.13.self_attn.q_proj.weight,
model.layers.13.self_attn.v_proj.biases,
model.layers.13.self_attn.v_proj.scales,
model.layers.13.self_attn.v_proj.weight,
model.layers.14.conv.conv.weight,
model.layers.14.conv.in_proj.biases,
model.layers.14.conv.in_proj.scales,
model.layers.14.conv.in_proj.weight,
model.layers.14.conv.out_proj.biases,
model.layers.14.conv.out_proj.scales,
model.layers.14.conv.out_proj.weight,
model.layers.14.feed_forward.w1.biases,
model.layers.14.feed_forward.w1.scales,
model.layers.14.feed_forward.w1.weight,
model.layers.14.feed_forward.w2.biases,
model.layers.14.feed_forward.w2.scales,
model.layers.14.feed_forward.w2.weight,
model.layers.14.feed_forward.w3.biases,
model.layers.14.feed_forward.w3.scales,
model.layers.14.feed_forward.w3.weight,
model.layers.14.ffn_norm.weight,
model.layers.14.operator_norm.weight,
model.layers.15.conv.conv.weight,
model.layers.15.conv.in_proj.biases,
model.layers.15.conv.in_proj.scales,
model.layers.15.conv.in_proj.weight,
model.layers.15.conv.out_proj.biases,
model.layers.15.conv.out_proj.scales,
model.layers.15.conv.out_proj.weight,
model.layers.15.feed_forward.w1.biases,
model.layers.15.feed_forward.w1.scales,
model.layers.15.feed_forward.w1.weight,
model.layers.15.feed_forward.w2.biases,
model.layers.15.feed_forward.w2.scales,
model.layers.15.feed_forward.w2.weight,
model.layers.15.feed_forward.w3.biases,
model.layers.15.feed_forward.w3.scales,
model.layers.15.feed_forward.w3.weight,
model.layers.15.ffn_norm.weight,
model.layers.15.operator_norm.weight,
model.layers.16.conv.conv.weight,
model.layers.16.conv.in_proj.biases,
model.layers.16.conv.in_proj.scales,
model.layers.16.conv.in_proj.weight,
model.layers.16.conv.out_proj.biases,
model.layers.16.conv.out_proj.scales,
model.layers.16.conv.out_proj.weight,
model.layers.16.feed_forward.w1.biases,
model.layers.16.feed_forward.w1.scales,
model.layers.16.feed_forward.w1.weight,
model.layers.16.feed_forward.w2.biases,
model.layers.16.feed_forward.w2.scales,
model.layers.16.feed_forward.w2.weight,
model.layers.16.feed_forward.w3.biases,
model.layers.16.feed_forward.w3.scales,
model.layers.16.feed_forward.w3.weight,
model.layers.16.ffn_norm.weight,
model.layers.16.operator_norm.weight,
model.layers.17.feed_forward.w1.biases,
model.layers.17.feed_forward.w1.scales,
model.layers.17.feed_forward.w1.weight,
model.layers.17.feed_forward.w2.biases,
model.layers.17.feed_forward.w2.scales,
model.layers.17.feed_forward.w2.weight,
model.layers.17.feed_forward.w3.biases,
model.layers.17.feed_forward.w3.scales,
model.layers.17.feed_forward.w3.weight,
model.layers.17.ffn_norm.weight,
model.layers.17.operator_norm.weight,
model.layers.17.self_attn.k_layernorm.weight,
model.layers.17.self_attn.k_proj.biases,
model.layers.17.self_attn.k_proj.scales,
model.layers.17.self_attn.k_proj.weight,
model.layers.17.self_attn.out_proj.biases,
model.layers.17.self_attn.out_proj.scales,
model.layers.17.self_attn.out_proj.weight,
model.layers.17.self_attn.q_layernorm.weight,
model.layers.17.self_attn.q_proj.biases,
model.layers.17.self_attn.q_proj.scales,
model.layers.17.self_attn.q_proj.weight,
model.layers.17.self_attn.v_proj.biases,
model.layers.17.self_attn.v_proj.scales,
model.layers.17.self_attn.v_proj.weight,
model.layers.18.conv.conv.weight,
model.layers.18.conv.in_proj.biases,
model.layers.18.conv.in_proj.scales,
model.layers.18.conv.in_proj.weight,
model.layers.18.conv.out_proj.biases,
model.layers.18.conv.out_proj.scales,
model.layers.18.conv.out_proj.weight,
model.layers.18.feed_forward.w1.biases,
model.layers.18.feed_forward.w1.scales,
model.layers.18.feed_forward.w1.weight,
model.layers.18.feed_forward.w2.biases,
model.layers.18.feed_forward.w2.scales,
model.layers.18.feed_forward.w2.weight,
model.layers.18.feed_forward.w3.biases,
model.layers.18.feed_forward.w3.scales,
model.layers.18.feed_forward.w3.weight,
model.layers.18.ffn_norm.weight,
model.layers.18.operator_norm.weight,
model.layers.19.conv.conv.weight,
model.layers.19.conv.in_proj.biases,
model.layers.19.conv.in_proj.scales,
model.layers.19.conv.in_proj.weight,
model.layers.19.conv.out_proj.biases,
model.layers.19.conv.out_proj.scales,
model.layers.19.conv.out_proj.weight,
model.layers.19.feed_forward.w1.biases,
model.layers.19.feed_forward.w1.scales,
model.layers.19.feed_forward.w1.weight,
model.layers.19.feed_forward.w2.biases,
model.layers.19.feed_forward.w2.scales,
model.layers.19.feed_forward.w2.weight,
model.layers.19.feed_forward.w3.biases,
model.layers.19.feed_forward.w3.scales,
model.layers.19.feed_forward.w3.weight,
model.layers.19.ffn_norm.weight,
model.layers.19.operator_norm.weight,
model.layers.2.feed_forward.w1.biases,
model.layers.2.feed_forward.w1.scales,
model.layers.2.feed_forward.w1.weight,
model.layers.2.feed_forward.w2.biases,
model.layers.2.feed_forward.w2.scales,
model.layers.2.feed_forward.w2.weight,
model.layers.2.feed_forward.w3.biases,
model.layers.2.feed_forward.w3.scales,
model.layers.2.feed_forward.w3.weight,
model.layers.2.ffn_norm.weight,
model.layers.2.operator_norm.weight,
model.layers.2.self_attn.k_layernorm.weight,
model.layers.2.self_attn.k_proj.biases,
model.layers.2.self_attn.k_proj.scales,
model.layers.2.self_attn.k_proj.weight,
model.layers.2.self_attn.out_proj.biases,
model.layers.2.self_attn.out_proj.scales,
model.layers.2.self_attn.out_proj.weight,
model.layers.2.self_attn.q_layernorm.weight,
model.layers.2.self_attn.q_proj.biases,
model.layers.2.self_attn.q_proj.scales,
model.layers.2.self_attn.q_proj.weight,
model.layers.2.self_attn.v_proj.biases,
model.layers.2.self_attn.v_proj.scales,
model.layers.2.self_attn.v_proj.weight,
model.layers.20.conv.conv.weight,
model.layers.20.conv.in_proj.biases,
model.layers.20.conv.in_proj.scales,
model.layers.20.conv.in_proj.weight,
model.layers.20.conv.out_proj.biases,
model.layers.20.conv.out_proj.scales,
model.layers.20.conv.out_proj.weight,
model.layers.20.feed_forward.w1.biases,
model.layers.20.feed_forward.w1.scales,
model.layers.20.feed_forward.w1.weight,
model.layers.20.feed_forward.w2.biases,
model.layers.20.feed_forward.w2.scales,
model.layers.20.feed_forward.w2.weight,
model.layers.20.feed_forward.w3.biases,
model.layers.20.feed_forward.w3.scales,
model.layers.20.feed_forward.w3.weight,
model.layers.20.ffn_norm.weight,
model.layers.20.operator_norm.weight,
model.layers.21.feed_forward.w1.biases,
model.layers.21.feed_forward.w1.scales,
model.layers.21.feed_forward.w1.weight,
model.layers.21.feed_forward.w2.biases,
model.layers.21.feed_forward.w2.scales,
model.layers.21.feed_forward.w2.weight,
model.layers.21.feed_forward.w3.biases,
model.layers.21.feed_forward.w3.scales,
model.layers.21.feed_forward.w3.weight,
model.layers.21.ffn_norm.weight,
model.layers.21.operator_norm.weight,
model.layers.21.self_attn.k_layernorm.weight,
model.layers.21.self_attn.k_proj.biases,
model.layers.21.self_attn.k_proj.scales,
model.layers.21.self_attn.k_proj.weight,
model.layers.21.self_attn.out_proj.biases,
model.layers.21.self_attn.out_proj.scales,
model.layers.21.self_attn.out_proj.weight,
model.layers.21.self_attn.q_layernorm.weight,
model.layers.21.self_attn.q_proj.biases,
model.layers.21.self_attn.q_proj.scales,
model.layers.21.self_attn.q_proj.weight,
model.layers.21.self_attn.v_proj.biases,
model.layers.21.self_attn.v_proj.scales,
model.layers.21.self_attn.v_proj.weight,
model.layers.22.conv.conv.weight,
model.layers.22.conv.in_proj.biases,
model.layers.22.conv.in_proj.scales,
model.layers.22.conv.in_proj.weight,
model.layers.22.conv.out_proj.biases,
model.layers.22.conv.out_proj.scales,
model.layers.22.conv.out_proj.weight,
model.layers.22.feed_forward.w1.biases,
model.layers.22.feed_forward.w1.scales,
model.layers.22.feed_forward.w1.weight,
model.layers.22.feed_forward.w2.biases,
model.layers.22.feed_forward.w2.scales,
model.layers.22.feed_forward.w2.weight,
model.layers.22.feed_forward.w3.biases,
model.layers.22.feed_forward.w3.scales,
model.layers.22.feed_forward.w3.weight,
model.layers.22.ffn_norm.weight,
model.layers.22.operator_norm.weight,
model.layers.23.conv.conv.weight,
model.layers.23.conv.in_proj.biases,
model.layers.23.conv.in_proj.scales,
model.layers.23.conv.in_proj.weight,
model.layers.23.conv.out_proj.biases,
model.layers.23.conv.out_proj.scales,
model.layers.23.conv.out_proj.weight,
model.layers.23.feed_forward.w1.biases,
model.layers.23.feed_forward.w1.scales,
model.layers.23.feed_forward.w1.weight,
model.layers.23.feed_forward.w2.biases,
model.layers.23.feed_forward.w2.scales,
model.layers.23.feed_forward.w2.weight,
model.layers.23.feed_forward.w3.biases,
model.layers.23.feed_forward.w3.scales,
model.layers.23.feed_forward.w3.weight,
model.layers.23.ffn_norm.weight,
model.layers.23.operator_norm.weight,
model.layers.24.feed_forward.w1.biases,
model.layers.24.feed_forward.w1.scales,
model.layers.24.feed_forward.w1.weight,
model.layers.24.feed_forward.w2.biases,
model.layers.24.feed_forward.w2.scales,
model.layers.24.feed_forward.w2.weight,
model.layers.24.feed_forward.w3.biases,
model.layers.24.feed_forward.w3.scales,
model.layers.24.feed_forward.w3.weight,
model.layers.24.ffn_norm.weight,
model.layers.24.operator_norm.weight,
model.layers.24.self_attn.k_layernorm.weight,
model.layers.24.self_attn.k_proj.biases,
model.layers.24.self_attn.k_proj.scales,
model.layers.24.self_attn.k_proj.weight,
model.layers.24.self_attn.out_proj.biases,
model.layers.24.self_attn.out_proj.scales,
model.layers.24.self_attn.out_proj.weight,
model.layers.24.self_attn.q_layernorm.weight,
model.layers.24.self_attn.q_proj.biases,
model.layers.24.self_attn.q_proj.scales,
model.layers.24.self_attn.q_proj.weight,
model.layers.24.self_attn.v_proj.biases,
model.layers.24.self_attn.v_proj.scales,
model.layers.24.self_attn.v_proj.weight,
model.layers.25.conv.conv.weight,
model.layers.25.conv.in_proj.biases,
model.layers.25.conv.in_proj.scales,
model.layers.25.conv.in_proj.weight,
model.layers.25.conv.out_proj.biases,
model.layers.25.conv.out_proj.scales,
model.layers.25.conv.out_proj.weight,
model.layers.25.feed_forward.w1.biases,
model.layers.25.feed_forward.w1.scales,
model.layers.25.feed_forward.w1.weight,
model.layers.25.feed_forward.w2.biases,
model.layers.25.feed_forward.w2.scales,
model.layers.25.feed_forward.w2.weight,
model.layers.25.feed_forward.w3.biases,
model.layers.25.feed_forward.w3.scales,
model.layers.25.feed_forward.w3.weight,
model.layers.25.ffn_norm.weight,
model.layers.25.operator_norm.weight,
model.layers.26.conv.conv.weight,
model.layers.26.conv.in_proj.biases,
model.layers.26.conv.in_proj.scales,
model.layers.26.conv.in_proj.weight,
model.layers.26.conv.out_proj.biases,
model.layers.26.conv.out_proj.scales,
model.layers.26.conv.out_proj.weight,
model.layers.26.feed_forward.w1.biases,
model.layers.26.feed_forward.w1.scales,
model.layers.26.feed_forward.w1.weight,
model.layers.26.feed_forward.w2.biases,
model.layers.26.feed_forward.w2.scales,
model.layers.26.feed_forward.w2.weight,
model.layers.26.feed_forward.w3.biases,
model.layers.26.feed_forward.w3.scales,
model.layers.26.feed_forward.w3.weight,
model.layers.26.ffn_norm.weight,
model.layers.26.operator_norm.weight,
model.layers.27.feed_forward.w1.biases,
model.layers.27.feed_forward.w1.scales,
model.layers.27.feed_forward.w1.weight,
model.layers.27.feed_forward.w2.biases,
model.layers.27.feed_forward.w2.scales,
model.layers.27.feed_forward.w2.weight,
model.layers.27.feed_forward.w3.biases,
model.layers.27.feed_forward.w3.scales,
model.layers.27.feed_forward.w3.weight,
model.layers.27.ffn_norm.weight,
model.layers.27.operator_norm.weight,
model.layers.27.self_attn.k_layernorm.weight,
model.layers.27.self_attn.k_proj.biases,
model.layers.27.self_attn.k_proj.scales,
model.layers.27.self_attn.k_proj.weight,
model.layers.27.self_attn.out_proj.biases,
model.layers.27.self_attn.out_proj.scales,
model.layers.27.self_attn.out_proj.weight,
model.layers.27.self_attn.q_layernorm.weight,
model.layers.27.self_attn.q_proj.biases,
model.layers.27.self_attn.q_proj.scales,
model.layers.27.self_attn.q_proj.weight,
model.layers.27.self_attn.v_proj.biases,
model.layers.27.self_attn.v_proj.scales,
model.layers.27.self_attn.v_proj.weight,
model.layers.28.conv.conv.weight,
model.layers.28.conv.in_proj.biases,
model.layers.28.conv.in_proj.scales,
model.layers.28.conv.in_proj.weight,
model.layers.28.conv.out_proj.biases,
model.layers.28.conv.out_proj.scales,
model.layers.28.conv.out_proj.weight,
model.layers.28.feed_forward.w1.biases,
model.layers.28.feed_forward.w1.scales,
model.layers.28.feed_forward.w1.weight,
model.layers.28.feed_forward.w2.biases,
model.layers.28.feed_forward.w2.scales,
model.layers.28.feed_forward.w2.weight,
model.layers.28.feed_forward.w3.biases,
model.layers.28.feed_forward.w3.scales,
model.layers.28.feed_forward.w3.weight,
model.layers.28.ffn_norm.weight,
model.layers.28.operator_norm.weight,
model.layers.29.conv.conv.weight,
model.layers.29.conv.in_proj.biases,
model.layers.29.conv.in_proj.scales,
model.layers.29.conv.in_proj.weight,
model.layers.29.conv.out_proj.biases,
model.layers.29.conv.out_proj.scales,
model.layers.29.conv.out_proj.weight,
model.layers.29.feed_forward.w1.biases,
model.layers.29.feed_forward.w1.scales,
model.layers.29.feed_forward.w1.weight,
model.layers.29.feed_forward.w2.biases,
model.layers.29.feed_forward.w2.scales,
model.layers.29.feed_forward.w2.weight,
model.layers.29.feed_forward.w3.biases,
model.layers.29.feed_forward.w3.scales,
model.layers.29.feed_forward.w3.weight,
model.layers.29.ffn_norm.weight,
model.layers.29.operator_norm.weight,
model.layers.3.conv.conv.weight,
model.layers.3.conv.in_proj.biases,
model.layers.3.conv.in_proj.scales,
model.layers.3.conv.in_proj.weight,
model.layers.3.conv.out_proj.biases,
model.layers.3.conv.out_proj.scales,
model.layers.3.conv.out_proj.weight,
model.layers.3.feed_forward.w1.biases,
model.layers.3.feed_forward.w1.scales,
model.layers.3.feed_forward.w1.weight,
model.layers.3.feed_forward.w2.biases,
model.layers.3.feed_forward.w2.scales,
model.layers.3.feed_forward.w2.weight,
model.layers.3.feed_forward.w3.biases,
model.layers.3.feed_forward.w3.scales,
model.layers.3.feed_forward.w3.weight,
model.layers.3.ffn_norm.weight,
model.layers.3.operator_norm.weight,
model.layers.4.conv.conv.weight,
model.layers.4.conv.in_proj.biases,
model.layers.4.conv.in_proj.scales,
model.layers.4.conv.in_proj.weight,
model.layers.4.conv.out_proj.biases,
model.layers.4.conv.out_proj.scales,
model.layers.4.conv.out_proj.weight,
model.layers.4.feed_forward.w1.biases,
model.layers.4.feed_forward.w1.scales,
model.layers.4.feed_forward.w1.weight,
model.layers.4.feed_forward.w2.biases,
model.layers.4.feed_forward.w2.scales,
model.layers.4.feed_forward.w2.weight,
model.layers.4.feed_forward.w3.biases,
model.layers.4.feed_forward.w3.scales,
model.layers.4.feed_forward.w3.weight,
model.layers.4.ffn_norm.weight,
model.layers.4.operator_norm.weight,
model.layers.5.feed_forward.w1.biases,
model.layers.5.feed_forward.w1.scales,
model.layers.5.feed_forward.w1.weight,
model.layers.5.feed_forward.w2.biases,
model.layers.5.feed_forward.w2.scales,
model.layers.5.feed_forward.w2.weight,
model.layers.5.feed_forward.w3.biases,
model.layers.5.feed_forward.w3.scales,
model.layers.5.feed_forward.w3.weight,
model.layers.5.ffn_norm.weight,
model.layers.5.operator_norm.weight,
model.layers.5.self_attn.k_layernorm.weight,
model.layers.5.self_attn.k_proj.biases,
model.layers.5.self_attn.k_proj.scales,
model.layers.5.self_attn.k_proj.weight,
model.layers.5.self_attn.out_proj.biases,
model.layers.5.self_attn.out_proj.scales,
model.layers.5.self_attn.out_proj.weight,
model.layers.5.self_attn.q_layernorm.weight,
model.layers.5.self_attn.q_proj.biases,
model.layers.5.self_attn.q_proj.scales,
model.layers.5.self_attn.q_proj.weight,
model.layers.5.self_attn.v_proj.biases,
model.layers.5.self_attn.v_proj.scales,
model.layers.5.self_attn.v_proj.weight,
model.layers.6.conv.conv.weight,
model.layers.6.conv.in_proj.biases,
model.layers.6.conv.in_proj.scales,
model.layers.6.conv.in_proj.weight,
model.layers.6.conv.out_proj.biases,
model.layers.6.conv.out_proj.scales,
model.layers.6.conv.out_proj.weight,
model.layers.6.feed_forward.w1.biases,
model.layers.6.feed_forward.w1.scales,
model.layers.6.feed_forward.w1.weight,
model.layers.6.feed_forward.w2.biases,
model.layers.6.feed_forward.w2.scales,
model.layers.6.feed_forward.w2.weight,
model.layers.6.feed_forward.w3.biases,
model.layers.6.feed_forward.w3.scales,
model.layers.6.feed_forward.w3.weight,
model.layers.6.ffn_norm.weight,
model.layers.6.operator_norm.weight,
model.layers.7.conv.conv.weight,
model.layers.7.conv.in_proj.biases,
model.layers.7.conv.in_proj.scales,
model.layers.7.conv.in_proj.weight,
model.layers.7.conv.out_proj.biases,
model.layers.7.conv.out_proj.scales,
model.layers.7.conv.out_proj.weight,
model.layers.7.feed_forward.w1.biases,
model.layers.7.feed_forward.w1.scales,
model.layers.7.feed_forward.w1.weight,
model.layers.7.feed_forward.w2.biases,
model.layers.7.feed_forward.w2.scales,
model.layers.7.feed_forward.w2.weight,
model.layers.7.feed_forward.w3.biases,
model.layers.7.feed_forward.w3.scales,
model.layers.7.feed_forward.w3.weight,
model.layers.7.ffn_norm.weight,
model.layers.7.operator_norm.weight,
model.layers.8.conv.conv.weight,
model.layers.8.conv.in_proj.biases,
model.layers.8.conv.in_proj.scales,
model.layers.8.conv.in_proj.weight,
model.layers.8.conv.out_proj.biases,
model.layers.8.conv.out_proj.scales,
model.layers.8.conv.out_proj.weight,
model.layers.8.feed_forward.w1.biases,
model.layers.8.feed_forward.w1.scales,
model.layers.8.feed_forward.w1.weight,
model.layers.8.feed_forward.w2.biases,
model.layers.8.feed_forward.w2.scales,
model.layers.8.feed_forward.w2.weight,
model.layers.8.feed_forward.w3.biases,
model.layers.8.feed_forward.w3.scales,
model.layers.8.feed_forward.w3.weight,
model.layers.8.ffn_norm.weight,
model.layers.8.operator_norm.weight,
model.layers.9.feed_forward.w1.biases,
model.layers.9.feed_forward.w1.scales,
model.layers.9.feed_forward.w1.weight,
model.layers.9.feed_forward.w2.biases,
model.layers.9.feed_forward.w2.scales,
model.layers.9.feed_forward.w2.weight,
model.layers.9.feed_forward.w3.biases,
model.layers.9.feed_forward.w3.scales,
model.layers.9.feed_forward.w3.weight,
model.layers.9.ffn_norm.weight,
model.layers.9.operator_norm.weight,
model.layers.9.self_attn.k_layernorm.weight,
model.layers.9.self_attn.k_proj.biases,
model.layers.9.self_attn.k_proj.scales,
model.layers.9.self_attn.k_proj.weight,
model.layers.9.self_attn.out_proj.biases,
model.layers.9.self_attn.out_proj.scales,
model.layers.9.self_attn.out_proj.weight,
model.layers.9.self_attn.q_layernorm.weight,
model.layers.9.self_attn.q_proj.biases,
model.layers.9.self_attn.q_proj.scales,
model.layers.9.self_attn.q_proj.weight,
model.layers.9.self_attn.v_proj.biases,
model.layers.9.self_attn.v_proj.scales,
model.layers.9.self_attn.v_proj.weight.

```

#### Captured upstream output

```text
=== STDERR ===
Downloading bytes:           |  0.00B
Reconstructing (incomplete total...): |          |  0.00B /  0.00B
Fetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]
Fetching 10 files: 100%|##########| 10/10 [00:00<00:00, 2621.77it/s]
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
[19:56:34] DEBUG    HF Cache Info for mlx-community/LFM2.5-VL-3B-OptiQ-4bit: size=2698.2 MB, files=12
```

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
- *Error code:* TRANSFORMERS_MODEL_LOAD_LIB_VERSION
- *Error type:* ValueError
- *Error package:* transformers
- *Error message:* Model loading failed: cannot import name
  'DefaultFastImageProcessorKwargs' from
  'transformers.image_processing_utils_fast' (unknown location)
- *Root exception type:* ImportError
- *Root exception module:* builtins
- *Root exception message:* cannot import name
  'DefaultFastImageProcessorKwargs' from
  'transformers.image_processing_utils_fast' (unknown location)
- *Arch supported by installed mlx-vlm:* yes (model_type youtu_vl)
- *Model load time:* 1.01s
- *Generation time:* -
- *Total time:* 1.01s
- *Input validation time:* 6.31e-05
- *Prompt preparation time:* -
- *First-token latency:* -
- *Cleanup time:* 0.153
- *Prompt tokens:* -
- *Generation tokens:* -
- *Total tokens:* -
- *Prompt throughput (raw):* -
- *Generation throughput (raw):* -
- *Peak memory:* -
- *Active memory:* -
- *Cache memory:* -
- *Model-load active memory:* -
- *Post-cleanup active memory:* 0.0128
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13505, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12735, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 822, in _typed_mlx_vlm_load
    loaded: tuple[nn.Module, ProcessorMixin] = _mlx_vlm_load(
                                               ~~~~~~~~~~~~~^
        path_or_hf_repo=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1212, in load
    processor = load_processor(model_path, True, eos_token_ids=eos_token_id, **kwargs)
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1357, in load_processor
    processor = AutoProcessor.from_pretrained(model_path, **kwargs)
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/base.py", line 644, in _patched_auto_processor_from_pretrained
    return previous_from_pretrained.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/base.py", line 644, in _patched_auto_processor_from_pretrained
    return previous_from_pretrained.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/base.py", line 644, in _patched_auto_processor_from_pretrained
    return previous_from_pretrained.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  [Previous line repeated 9 more times]
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/molmo2/processing.py", line 768, in _patched_auto_processor_from_pretrained_molmo2
    return _original_auto_processor_from_pretrained_molmo2.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/base.py", line 644, in _patched_auto_processor_from_pretrained
    return previous_from_pretrained.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/base.py", line 644, in _patched_auto_processor_from_pretrained
    return previous_from_pretrained.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/base.py", line 644, in _patched_auto_processor_from_pretrained
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14520, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13520, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: cannot import name 'DefaultFastImageProcessorKwargs' from 'transformers.image_processing_utils_fast' (unknown location)

```

#### Captured upstream output

```text
=== STDERR ===
Downloading bytes:           |  0.00B
Reconstructing (incomplete total...): |          |  0.00B /  0.00B
Fetching 19 files:   0%|          | 0/19 [00:00<?, ?it/s]
Fetching 19 files: 100%|##########| 19/19 [00:00<00:00, 3082.50it/s]
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
[20:07:00] DEBUG    HF Cache Info for tencent/Youtu-VL-4B-Instruct: size=10229.6 MB, files=27
```

</details>

---
