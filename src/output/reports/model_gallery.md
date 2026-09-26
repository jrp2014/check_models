# Model Output Gallery

Generated on: 2026-09-26 23:32:46 BST

- *Evaluation lane:* assisted
- *Prompt hints:* the image's description and keyword hints were included in the prompt, so field content may be copied from them rather than seen
- *Assessment:* General checks + metadata fields and duplicate keywords; length limits and factual accuracy not assessed
- *Input image:* JPEG, 9,984 x 6,656 pixels (66.5 MP), 49.4 MB

This run records model responses to one shared image and prompt (evaluation
lane: assisted). Mechanical checks are not factual-accuracy judgments; inspect
the image, prompt and final answers before choosing a model. Results do not
establish fitness for other tasks.

Complete per-model evidence artifact with image metadata, the source prompt, a
facts-only chooser, and full generated or crash output for every attempted
model.

## Reference Image

![Reference image](assets/source-image-b32a199300908fe5.jpg)

## Current-run Chooser

Mechanical observations and captured resource facts for this run only. No concerns detected does not mean the response fulfilled an arbitrary prompt or described the image accurately. Consult the assessment scope above. Total time is end-to-end; throughput covers generation only and requires at least 16 generated tokens. Prefill/first is the measured time to first token (input preparation, prefill and the first decode step) when captured, else upstream's model-loop first-token time; Prompt tok is the full rendered prompt including image tokens, which drives prefill cost. For cross-attention architectures the token count reflects the tokenised text burden only, not total vision prefill compute.

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                                               | Mechanical checks      | Total s | Gen TPS    | Prefill/first s | Peak GB | Prompt tok | Gen tok | Observations                                                                                      |
|-------------------------------------------------------------------------------------------------------------------------------------|------------------------|---------|------------|-----------------|---------|------------|---------|---------------------------------------------------------------------------------------------------|
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                                        | `no concerns detected` | 2.39s   | 475 tok/s  | 0.78            | 1.9     | 2,119      | 141     | none                                                                                              |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit)             | `no concerns detected` | 15.65s  | 29.3 tok/s | 5.88            | 23      | 2,394      | 123     | none                                                                                              |
| [`mlx-community/GLM-4.6V-Flash-4bit`](#model-mlx-community-glm-46v-flash-4bit)                                                      | `no concerns detected` | 10.58s  | 73.6 tok/s | 6.63            | 8.7     | 6,454      | 139     | none                                                                                              |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                                | `no concerns detected` | 35.98s  | 40.2 tok/s | 18.58           | 78      | 6,454      | 142     | none                                                                                              |
| [`mlx-community/InternVL3-14B-4bit`](#model-mlx-community-internvl3-14b-4bit)                                                       | `no concerns detected` | 7.52s   | 56.3 tok/s | 3.61            | 10      | 2,115      | 116     | none                                                                                              |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                                         | `no concerns detected` | 6.81s   | 36.6 tok/s | 1.71            | 17      | 2,115      | 103     | none                                                                                              |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-8bit`](#model-mlx-community-kimi-vl-a3b-thinking-2506-8bit)                               | `no concerns detected` | 18.59s  | 60.7 tok/s | 3.60            | 20      | 1,331      | 729     | none                                                                                              |
| [`mlx-community/LFM2.5-VL-3B-OptiQ-4bit`](#model-mlx-community-lfm25-vl-3b-optiq-4bit)                                              | `no concerns detected` | 3.83s   | 206 tok/s  | 1.41            | 4.0     | 2,111      | 110     | none                                                                                              |
| [`mlx-community/MiniCPM-o-4_5-4bit`](#model-mlx-community-minicpm-o-45-4bit)                                                        | `no concerns detected` | 3.62s   | 104 tok/s  | 0.95            | 7.0     | 393        | 114     | none                                                                                              |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)                     | `no concerns detected` | 7.30s   | 65.5 tok/s | 2.74            | 13      | 2,927      | 160     | none                                                                                              |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)                         | `no concerns detected` | 4.09s   | 188 tok/s  | 1.58            | 7.8     | 2,926      | 152     | none                                                                                              |
| [`mlx-community/North-Micro-Vision-Instruct-4bit`](#model-mlx-community-north-micro-vision-instruct-4bit)                           | `no concerns detected` | 5.30s   | 165 tok/s  | 2.77            | 3.9     | 4,085      | 112     | none                                                                                              |
| [`mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit`](#model-mlx-community-ornith-15-35b-a3b-optiq-4bit)                                  | `no concerns detected` | 7.59s   | 60.2 tok/s | 1.74            | 24      | 1,291      | 151     | none                                                                                              |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                                    | `no concerns detected` | 6.37s   | 37.1 tok/s | 1.20            | 9.3     | 1,141      | 136     | none                                                                                              |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                                         | `no concerns detected` | 28.55s  | 87.3 tok/s | 16.13           | 8.4     | 16,551     | 918     | none                                                                                              |
| [`mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit`](#model-mlx-community-qwen3-vl-30b-a3b-instruct-4bit)                               | `no concerns detected` | 38.61s  | 75.2 tok/s | 33.96           | 23      | 16,549     | 128     | none                                                                                              |
| [`mlx-community/Qwen3-VL-32B-Instruct-4bit`](#model-mlx-community-qwen3-vl-32b-instruct-4bit)                                       | `no concerns detected` | 73.38s  | 16.7 tok/s | 59.54           | 26      | 16,549     | 181     | none                                                                                              |
| [`mlx-community/Qwen3-VL-8B-Instruct-4bit`](#model-mlx-community-qwen3-vl-8b-instruct-4bit)                                         | `no concerns detected` | 43.38s  | 67.6 tok/s | 39.77           | 11      | 16,549     | 104     | none                                                                                              |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                                    | `no concerns detected` | 40.68s  | 73.5 tok/s | 35.34           | 25      | 16,565     | 111     | none                                                                                              |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                                      | `no concerns detected` | 39.83s  | 84.5 tok/s | 35.39           | 11      | 16,565     | 142     | none                                                                                              |
| [`mlx-community/Qwen3.8-27B-nvfp4`](#model-mlx-community-qwen38-27b-nvfp4)                                                          | `no concerns detected` | 66.31s  | 28.2 tok/s | 57.92           | 21      | 16,565     | 144     | none                                                                                              |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                                        | `no concerns detected` | 3.41s   | 125 tok/s  | 1.35            | 5.6     | 1,433      | 78      | none                                                                                              |
| [`mlx-community/Step-3.7-Flash-oQ3e`](#model-mlx-community-step-37-flash-oq3e)                                                      | `no concerns detected` | 106.26s | 44.0 tok/s | 81.04           | 92      | 3,494      | 147     | none                                                                                              |
| [`mlx-community/aya-vision-8b-4bit`](#model-mlx-community-aya-vision-8b-4bit)                                                       | `no concerns detected` | 5.87s   | 91.7 tok/s | 2.13            | 6.5     | 2,089      | 112     | none                                                                                              |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                             | `no concerns detected` | 8.55s   | 39.2 tok/s | 4.19            | 28      | 592        | 85      | none                                                                                              |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                             | `no concerns detected` | 11.33s  | 29.4 tok/s | 1.69            | 17      | 591        | 175     | none                                                                                              |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                             | `no concerns detected` | 6.53s   | 76.1 tok/s | 1.25            | 16      | 596        | 130     | none                                                                                              |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                                     | `no concerns detected` | 8.31s   | 26.8 tok/s | 1.68            | 20      | 596        | 89      | none                                                                                              |
| [`mlx-community/gemma-4-e4b-it-4bit`](#model-mlx-community-gemma-4-e4b-it-4bit)                                                     | `no concerns detected` | 4.82s   | 96.4 tok/s | 1.16            | 5.9     | 592        | 94      | none                                                                                              |
| [`mlx-community/granite-4.0-3b-vision-4bit`](#model-mlx-community-granite-40-3b-vision-4bit)                                        | `no concerns detected` | 4.57s   | 129 tok/s  | 2.09            | 4.7     | 1,383      | 99      | none                                                                                              |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                                           | `no concerns detected` | 7.85s   | 37.2 tok/s | 2.39            | 16      | 3,117      | 114     | none                                                                                              |
| [`nativ-community/Mage-VL-OptiQ-4bit`](#model-nativ-community-mage-vl-optiq-4bit)                                                   | `no concerns detected` | 4.88s   | 124 tok/s  | 2.23            | 5.4     | 4,212      | 130     | none                                                                                              |
| [`nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit`](#model-nativ-community-mistral-small-32-24b-instruct-2506-4bit)        | `no concerns detected` | 8.66s   | 35.0 tok/s | 2.23            | 18      | 1,273      | 134     | none                                                                                              |
| [`nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit`](#model-nativ-community-nemotron-3-nano-omni-30b-a3b-reasoning-4bit) | `no concerns detected` | 10.71s  | 91.5 tok/s | 6.07            | 23      | 3,628      | 144     | none                                                                                              |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                             | `concerns detected`    | 10.30s  | 34.4 tok/s | 2.53            | 18      | 2,619      | 164     | duplicate keywords                                                                                |
| [`mlx-community/gemma-4-12B-it-4bit`](#model-mlx-community-gemma-4-12b-it-4bit)                                                     | `concerns detected`    | 6.20s   | 58.1 tok/s | 1.32            | 7.6     | 596        | 108     | duplicate keywords                                                                                |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-4bit)                        | `major concerns`       | 9.03s   | 95.2 tok/s | 2.22            | 19      | 1,639      | 425     | stopped early: repeating; labelled fields not detected; incomplete thinking block                 |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                                          | `major concerns`       | 4.13s   | 311 tok/s  | 2.11            | 1.8     | 336        | 51      | labelled fields not detected                                                                      |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)                        | `major concerns`       | 58.73s  | 18.7 tok/s | 2.82            | 15      | 308        | 1,000   | repeated text; cut off at token limit; duplicate keywords                                         |
| [`mlx-community/MiniCPM-V-4.6-4bit`](#model-mlx-community-minicpm-v-46-4bit)                                                        | `major concerns`       | 4.73s   | 247 tok/s  | 2.71            | 3.3     | 934        | 84      | incomplete thinking block                                                                         |
| [`mlx-community/Molmo2-8B-4bit`](#model-mlx-community-molmo2-8b-4bit)                                                               | `major concerns`       | 8.66s   | 70.3 tok/s | 3.57            | 8.1     | 1,526      | 225     | stopped early: repeating; duplicate keywords                                                      |
| [`mlx-community/Muse-Glimmer-30B-OptiQ-4bit`](#model-mlx-community-muse-glimmer-30b-optiq-4bit)                                     | `major concerns`       | 63.41s  | 20.7 tok/s | 11.04           | 25      | 4,409      | 1,000   | control tokens visible; labelled fields not detected; cut off at token limit; role tokens visible |
| [`mlx-community/Qwen2-VL-7B-Instruct-4bit`](#model-mlx-community-qwen2-vl-7b-instruct-4bit)                                         | `major concerns`       | 43.45s  | 89.5 tok/s | 39.15           | 9.3     | 16,560     | 225     | repeated text; stopped early: repeating; duplicate keywords                                       |
| [`mlx-community/SmolVLM-256M-Instruct-4bit`](#model-mlx-community-smolvlm-256m-instruct-4bit)                                       | `major concerns`       | 2.63s   | 322 tok/s  | 1.05            | 1.1     | 1,212      | 38      | labelled fields not detected                                                                      |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                                       | `major concerns`       | 20.22s  | 56.2 tok/s | 13.21           | 14      | 16,560     | 250     | stopped early: repeating; duplicate keywords                                                      |
| [`mlx-community/gemma-3n-E4B-it-4bit`](#model-mlx-community-gemma-3n-e4b-it-4bit)                                                   | `major concerns`       | 8.06s   | 60.6 tok/s | 2.11            | 6.9     | 590        | 196     | labelled fields not detected                                                                      |
| [`mlx-community/granite-vision-3.2-2b-nvfp4`](#model-mlx-community-granite-vision-32-2b-nvfp4)                                      | `major concerns`       | 4.78s   | 141 tok/s  | 2.59            | 4.2     | 5,581      | 102     | labelled fields not detected                                                                      |
| [`mlx-community/llm-jp-4-vl-9b-mlx-4bit`](#model-mlx-community-llm-jp-4-vl-9b-mlx-4bit)                                             | `major concerns`       | 3.45s   | 111 tok/s  | 1.49            | 6.7     | 2,197      | 16      | control tokens visible; labelled fields not detected                                              |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                                        | `major concerns`       | 2.41s   | 164 tok/s  | 0.89            | 1.4     | 332        | 21      | labelled fields not detected                                                                      |
| [`mlx-community/InternVL3_5-1B-4bit`](#model-mlx-community-internvl35-1b-4bit)                                                      | `not assessed`         | 0.13s   | -          | -               | -       | -          | -       | none                                                                                              |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Resource Highlights

Quickest completion without detected concerns (end-to-end, including model load): `LiquidAI/LFM2.5-VL-450M-MLX-bf16` at 2.39s

Lowest peak memory among completions without detected concerns: `LiquidAI/LFM2.5-VL-450M-MLX-bf16` at 1.9 GB

Decode tok/s stays per model in the chooser and is not averaged across models: tokenizers, image-token expansion and reasoning lengths differ too much for a cross-model mean to guide a choice.

## Avoid for This Run

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                        | Mechanical checks | Observations                                                                                      |
|--------------------------------------------------------------------------------------------------------------|-------------------|---------------------------------------------------------------------------------------------------|
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-4bit) | `major concerns`  | stopped early: repeating; labelled fields not detected; incomplete thinking block                 |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                   | `major concerns`  | labelled fields not detected                                                                      |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit) | `major concerns`  | repeated text; cut off at token limit; duplicate keywords                                         |
| [`mlx-community/MiniCPM-V-4.6-4bit`](#model-mlx-community-minicpm-v-46-4bit)                                 | `major concerns`  | incomplete thinking block                                                                         |
| [`mlx-community/Molmo2-8B-4bit`](#model-mlx-community-molmo2-8b-4bit)                                        | `major concerns`  | stopped early: repeating; duplicate keywords                                                      |
| [`mlx-community/Muse-Glimmer-30B-OptiQ-4bit`](#model-mlx-community-muse-glimmer-30b-optiq-4bit)              | `major concerns`  | control tokens visible; labelled fields not detected; cut off at token limit; role tokens visible |
| [`mlx-community/Qwen2-VL-7B-Instruct-4bit`](#model-mlx-community-qwen2-vl-7b-instruct-4bit)                  | `major concerns`  | repeated text; stopped early: repeating; duplicate keywords                                       |
| [`mlx-community/SmolVLM-256M-Instruct-4bit`](#model-mlx-community-smolvlm-256m-instruct-4bit)                | `major concerns`  | labelled fields not detected                                                                      |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                | `major concerns`  | stopped early: repeating; duplicate keywords                                                      |
| [`mlx-community/gemma-3n-E4B-it-4bit`](#model-mlx-community-gemma-3n-e4b-it-4bit)                            | `major concerns`  | labelled fields not detected                                                                      |
| [`mlx-community/granite-vision-3.2-2b-nvfp4`](#model-mlx-community-granite-vision-32-2b-nvfp4)               | `major concerns`  | labelled fields not detected                                                                      |
| [`mlx-community/llm-jp-4-vl-9b-mlx-4bit`](#model-mlx-community-llm-jp-4-vl-9b-mlx-4bit)                      | `major concerns`  | control tokens visible; labelled fields not detected                                              |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                 | `major concerns`  | labelled fields not detected                                                                      |
| [`mlx-community/InternVL3_5-1B-4bit`](#model-mlx-community-internvl35-1b-4bit)                               | `not assessed`    | none                                                                                              |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Output at a Glance

A compact preview of each model's final answer (or failure evidence for crashes), in chooser order. Where the requested catalogue fields were detected, the preview shows a little of each: the title, the start of the description, and the first keywords with their count, so the weakest field is not hidden behind a long description. Otherwise it is the first 280 characters. A closed reasoning trace is left out of the preview and reported as an omitted-character count; the complete output, trace included, is in the model's evidence block below.

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                                               | Mechanical checks      | Output preview                                                                                                                                                                                                                                                                                                                                                                            |
|-------------------------------------------------------------------------------------------------------------------------------------|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                                        | `no concerns detected` | Title: Sailboats on a River \| Description: Two sailboats glide across a calm river, surrounded by dense green forest and a partly cloudy sky. The sailboat on t... \| Keywords (15): Boat, Boating, Catamaran, Sailboat, Sailing, Life jacket, Man, Sail, Sails, Water, Forest, Sky, Trees, River, ...                                                                                   |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit)             | `no concerns detected` | Title: Two sailors on a catamaran and dinghy \| Description: Two sailors navigate a catamaran (sail number 1067) and a Laser dinghy (sail number GBR 188572) on calm waters,... \| Keywords (20): Boat, Boating, Catamaran, Clouds, Dinghy, Estuary, Forest, Laser dinghy, Life jacket, Man, Mast, ...                                                                                    |
| [`mlx-community/GLM-4.6V-Flash-4bit`](#model-mlx-community-glm-46v-flash-4bit)                                                      | `no concerns detected` | Title: Two Sailors on Dinghies \| Description: Two sailors steer small dinghies—a Vortex catamaran (sail number 1067) on the left and a Laser dinghy (sail number... \| Keywords (19): Boat, Boating, Catamaran, Dinghy, Estuary, Forest, Laser dinghy, Life jacket, Man, Mast, Outdoor recreation, ...                                                                                   |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                                | `no concerns detected` | Title: Two Sailors in Vortex and Laser Dinghies on Calm Waters \| Description: Two sailors navigate a Vortex catamaran (sail number 1067) and a Laser dinghy (sail... \| Keywords (19): Boat, Boating, Catamaran, Clouds, Dinghy, Forest, Laser dinghy, Life jacket, Man, Mast, Outdoor recreation, ...                                                                                   |
| [`mlx-community/InternVL3-14B-4bit`](#model-mlx-community-internvl3-14b-4bit)                                                       | `no concerns detected` | Title: Sailing Dinghies on Calm Waters \| Description: Two sailors navigate small dinghies, a Vortex catamaran (1067) and a Laser dinghy (GBR 188572), on calm waters with a... \| Keywords (20): Boat, Boating, Catamaran, Clouds, Dinghy, Estuary, Forest, Laser dinghy, Life jacket, Man, Mast, ...                                                                                    |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                                         | `no concerns detected` | Title: Sailing on Calm Waters with Catamaran and Laser Dinghy \| Description: Two sailors navigate a Vortex catamaran and a Laser dinghy on calm waters near a fo... \| Keywords (17): Sailing, Catamaran, Laser dinghy, Vortex, GBR, 188572, 1067, Dinghy, Life jacket, Outdoor recreation, River, ...                                                                                   |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-8bit`](#model-mlx-community-kimi-vl-a3b-thinking-2506-8bit)                               | `no concerns detected` | Title: Two Sailors Navigate dinghies in Calm Waters Near Forested Shoreline \| Description: Two sailors steer a Vortex catamaran (sail 1067) and a Laser dinghy (... \| Keywords (19): Boat, Boating, Catamaran, Dinghy, Estuary, Forest, Laser dinghy, Life jacket, Man, Mast, Outdoor recreation, ...[2,349 characters of reasoning omitted; complete output in the evidence block]     |
| [`mlx-community/LFM2.5-VL-3B-OptiQ-4bit`](#model-mlx-community-lfm25-vl-3b-optiq-4bit)                                              | `no concerns detected` | Title: Sailors race dinghies across calm waters. \| Description: Two sailors compete in a Vortex catamaran and Laser dinghy against a backdrop of dense green woodland. The s... \| Keywords (19): Boat, Boating, Catamaran, Clouds, Dinghy, Estuary, Forest, Laser dinghy, Life jacket, Man, Mast, ...                                                                                   |
| [`mlx-community/MiniCPM-o-4_5-4bit`](#model-mlx-community-minicpm-o-45-4bit)                                                        | `no concerns detected` | Title: Sailboats on Calm Water near Green Woodland \| Description: Two sailors navigate a Vortex catamaran and Laser dinghy across tranquil waters, with dense fo... \| Keywords (17): Sailboat, Dinghy, Sailor, Vortex catamaran, Laser dinghy, Sail number 1067, Sail number 188572, Life jacket, ...                                                                                   |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)                     | `no concerns detected` | Title: **Sailing Dinghies in Coastal Waters – Vortex and Laser** \| Description: Two sailors navigate a Vortex catamaran (sail number 1067) and a Laser dinghy... \| Keywords (16): Boating, coastal waters, dinghy sailing, Laser dinghy, life jackets, manoeuvring, outdoor recreation, sailing, ...                                                                                    |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)                         | `no concerns detected` | Title: Coastal Sailing Adventure with Catamaran and Laser Dinghy \| Description: Two sailors navigate small boats—one a Vortex catamaran (sail number 1067) and... \| Keywords (19): Catamaran, Coastal waters, Dinghy, Estuary, Forest, Laser dinghy, Life jacket, Man, Mast, Outdoor recreation, ...                                                                                    |
| [`mlx-community/North-Micro-Vision-Instruct-4bit`](#model-mlx-community-north-micro-vision-instruct-4bit)                           | `no concerns detected` | Title: Sailboats on Calm Waters \| Description: Two sailors navigate small dinghies across tranquil waters, one steering a Vortex catamaran (sail number 1067) and the other... \| Keywords (20): Boat, Boating, Catamaran, Clouds, Dinghy, Estuary, Forest, Laser dinghy, Life jacket, Man, Mast, ...                                                                                    |
| [`mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit`](#model-mlx-community-ornith-15-35b-a3b-optiq-4bit)                                  | `no concerns detected` | Title: Two Sailboats Racing on Calm Waters \| Description: On 19 September 2026, an orange Vortex catamaran (sail number 1067) helmed by a sailor in a blue jacket a... \| Keywords (19): Sailing, Sailboat, Catamaran, Dinghy, Laser, Man, Sailor, Life jacket, Mast, Boat, Water, River, Estuary, ...                                                                                   |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                                    | `no concerns detected` | Title: Sailors on Dinghies in Coastal Waters \| Description: On September 19, 2026, two sailors navigate their respective dinghies, a Vortex catamaran and a Laser dinghy,... \| Keywords (18): Sailors, Dinghies, Vortex, Laser, Coastal Waters, Forest, Sailing, Trees, Water, Clouds, Man, Mast, ...                                                                                   |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                                         | `no concerns detected` | Title: Sailors Steering Vortex &amp; Laser Dinghies on Calm Water \| Description: Two sailors steer Vortex (1067) and Laser dinghy (GBR 188572) across calm river wat... \| Keywords (19): Boat, Boating, Catamaran, Clouds, Dinghy, Estuary, Laser dinghy, Life jacket, Man, Mast, Outdoor recreation, ...[2,719 characters of reasoning omitted; complete output in the evidence block] |
| [`mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit`](#model-mlx-community-qwen3-vl-30b-a3b-instruct-4bit)                               | `no concerns detected` | Title: Two sailboats racing on calm water \| Description: Two sailors compete in a race on small dinghies—a Vortex catamaran (number 1067) and a Laser dinghy (num... \| Keywords (19): Boat, Boating, Catamaran, Clouds, Dinghy, Forest, Laser dinghy, Life jacket, Man, Mast, Outdoor recreation, ...                                                                                   |
| [`mlx-community/Qwen3-VL-32B-Instruct-4bit`](#model-mlx-community-qwen3-vl-32b-instruct-4bit)                                       | `no concerns detected` | Title: Sailors in Vortex Catamaran and Laser Dinghy on Calm Water \| Description: On 2026-09-19, two sailors navigate small dinghies on calm waters: a Vortex catam... \| Keywords (24): Sailboat, Sailing, Dinghy, Catamaran, Vortex, Laser, Sail number, GBR, 1067, 188572, Sailor, Water, River, ...                                                                                   |
| [`mlx-community/Qwen3-VL-8B-Instruct-4bit`](#model-mlx-community-qwen3-vl-8b-instruct-4bit)                                         | `no concerns detected` | Title: Sailors race dinghies on calm water \| Description: Two sailors navigate a Vortex catamaran and Laser dinghy across tranquil waters, framed by dense green fore... \| Keywords (18): Sailboat, Dinghy, Catamaran, Laser, Sailing, Water, Forest, Trees, Shoreline, Sky, Clouds, Man, Sailor, ...                                                                                   |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                                    | `no concerns detected` | Title: Vortex Catamaran and Laser Sailing on Water \| Description: Two sailors navigate a Vortex catamaran and a Laser dinghy on calm river waters on 19 Septe... \| Keywords (16): Vortex catamaran, Laser dinghy, sailors, calm water, river, shoreline, dense green woodland, partly cloudy sky, ...                                                                                   |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                                      | `no concerns detected` | Title: Two Sailors Navigate Vortex Catamaran and Laser Dinghy on Calm Waters \| Description: Two sailors steer a Vortex catamaran (sail 1067) and a Laser dingh... \| Keywords (21): Sailing, Dinghy, Catamaran, Vortex, Laser, Sailboat, Sailor, Life jacket, Mast, River, Estuary, Forest, Trees, ...                                                                                   |
| [`mlx-community/Qwen3.8-27B-nvfp4`](#model-mlx-community-qwen38-27b-nvfp4)                                                          | `no concerns detected` | Title: Two sailors racing Vortex and Laser dinghies on calm waters \| Description: A Vortex catamaran with sail number 1067 on the left and a Laser dinghy with sail n... \| Keywords (15): Vortex, Laser, catamaran, dinghy, sailing, sailors, estuary, woodland, boats, water, masts, GBR 188572, ...                                                                                   |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                                        | `no concerns detected` | Title: Sailing on the River \| Description: Two sailors are sailing their boats across the river. \| Keywords (20): Boat, Boating, Catamaran, Clouds, Dinghy, Estuary, Forest, Laser dinghy, Life jacket, Man, Mast, ...                                                                                                                                                                  |
| [`mlx-community/Step-3.7-Flash-oQ3e`](#model-mlx-community-step-37-flash-oq3e)                                                      | `no concerns detected` | Title: Two sailors on dinghies across calm waters \| Description: On 19 September 2026 at 17:12 UTC+1, two sailors steer small dinghies across calm coastal or river... \| Keywords (18): Sailboat, Sailing, Sailor, Dinghy, Catamaran, Laser dinghy, Vortex, Boat, Boating, Water, River, Estuary, ...                                                                                   |
| [`mlx-community/aya-vision-8b-4bit`](#model-mlx-community-aya-vision-8b-4bit)                                                       | `no concerns detected` | Title: Sailing Adventure on the River \| Description: Two sailors navigate their boats across a serene river, with one in a Vortex catamaran and the other in a Laser dinghy, bot... \| Keywords (16): Catamaran, Dinghy, River, Sailing, Trees, Water, Sky, Boat, Boating, Mast, Life jacket, Man, ...                                                                                   |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                             | `no concerns detected` | Title: Two Sailors Sailing Dinghies on Calm Water \| Description: Two sailors steer a grey Vortex catamaran and a white Laser dinghy across calm waters agains... \| Keywords (15): Sailing, Sailboat, Catamaran, Dinghy, Sailor, Mast, Water, Forest, River, Outdoor Recreation, Boating, Estuary, ...                                                                                   |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                             | `no concerns detected` | Title: Sailing Dinghies on Calm Water, September 2026 \| Description: Captured on 19th September 2026, this image shows a Vortex catamaran (sail number 1067) and a Laser din... \| Keywords (27): Boat, Boating, Catamaran, Clouds, Dinghy, Estuary, Forest, Laser dinghy, Life jacket, Man, Mast, ...                                                                                   |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                             | `no concerns detected` | Title: Two sailors steering small boats across calm water \| Description: Two sailors navigate small boats across calm waters against a backdrop of dense green wo... \| Keywords (18): Boat, Boating, Catamaran, Clouds, Dinghy, Forest, Laser dinghy, Life jacket, Man, Mast, Outdoor recreation, ...                                                                                   |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                                     | `no concerns detected` | Title: Two Sailors Steering Dinghies on Calm Water \| Description: A Vortex catamaran and a Laser dinghy sail across calm waters against a backdrop of dense green woodland u... \| Keywords (19): Boat, Boating, Catamaran, Clouds, Dinghy, Estuary, Forest, Laser dinghy, Life jacket, Man, Mast, ...                                                                                   |
| [`mlx-community/gemma-4-e4b-it-4bit`](#model-mlx-community-gemma-4-e4b-it-4bit)                                                     | `no concerns detected` | Title: Two Dinghies Sail on Calm Woodland Water \| Description: Two small dinghies navigate placid waters beneath a backdrop of dense green woodland under partly... \| Keywords (15): Catamaran, Dinghy, Laser, Sailing, Boating, Woodland, Estuary, Sailboat, Water, Outdoor, Recreation, Trees, ...                                                                                    |
| [`mlx-community/granite-4.0-3b-vision-4bit`](#model-mlx-community-granite-40-3b-vision-4bit)                                        | `no concerns detected` | Title: "Sailors in Dinghies on a Calm Day" \| Description: Two sailors navigate their dinghies, a Vortex catamaran and a Laser dinghy, across a serene body of water wi... \| Keywords (13): Sailors, Dinghies, Vortex catamaran, Laser dinghy, Water, Trees, Coastline, Sailing, Life jacket, Man, ...                                                                                   |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                                           | `no concerns detected` | Title: Sailors Navigate Calm Waters in Dinghies \| Description: Two sailors steer small dinghies—a Vortex catamaran and a Laser dinghy—across calm waters with dense green wo... \| Keywords (21): Boat, Boating, Catamaran, Clouds, Dinghy, Estuary, Forest, Laser dinghy, Life jacket, Man, Mast, ...                                                                                   |
| [`nativ-community/Mage-VL-OptiQ-4bit`](#model-nativ-community-mage-vl-optiq-4bit)                                                   | `no concerns detected` | Title: Two Sailboats Glide Across Calm Waters Amidst Lush Forest \| Description: Two sailors navigate a Vortex catamaran (sail number 1067) and a Laser dinghy (sail number G... \| Keywords (20): Boat, Boating, Catamaran, Clouds, Dinghy, Estuary, Forest, Laser dinghy, Life jacket, Man, Mast, ...                                                                                   |
| [`nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit`](#model-nativ-community-mistral-small-32-24b-instruct-2506-4bit)        | `no concerns detected` | Title: Sailors Navigate Calm Waters in Dinghies \| Description: Two sailors steer small dinghies—a Vortex catamaran (sail number 1067) and a Laser dinghy (sai... \| Keywords (18): Boat, Boating, Catamaran, Clouds, Dinghy, Estuary, Forest, Laser dinghy, Life jacket, Mast, Outdoor recreation, ...                                                                                   |
| [`nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit`](#model-nativ-community-nemotron-3-nano-omni-30b-a3b-reasoning-4bit) | `no concerns detected` | Title: Two Sailors Compete in Dinghy Race \| Description: On a calm body of water under a partly cloudy sky, a sailor in a blue life jacket steers a Vortex catamaran with sa... \| Keywords (20): Boat, Boating, Catamaran, Clouds, Dinghy, Estuary, Forest, Laser dinghy, Life jacket, Man, Mast, ...                                                                                   |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                             | `concerns detected`    | Title: Laser and Vortex catamaran sailboats on a river with trees. \| Description: Two sailboats, a Laser dinghy with sail number GBR 188572 and a Vortex catam... \| Keywords (12): Laser dinghy, Vortex catamaran, sailboats, river, woodland, sail number GBR 188572, sail number 1067, sailors, ...                                                                                   |
| [`mlx-community/gemma-4-12B-it-4bit`](#model-mlx-community-gemma-4-12b-it-4bit)                                                     | `concerns detected`    | Title: Two Sailors Steering Small Dinghies on Calm Water \| Description: Two sailors navigate a Vortex catamaran and a Laser dinghy across calm water against a back... \| Keywords (19): Boat, Boating, Catamaran, Sailing, Sailor, Laser dinghy, Water, River, Estuary, Forest, Trees, Shoreline, ...                                                                                   |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-4bit)                        | `major concerns`       | Alright, let's get this done. I need to create some metadata for this image, and it's my job to be precise.<br><br>First, I need to figure out a good title. "Sailboats on the Water" is too generic. "Two Sailors on Dinghies" is better, but I want something more specific. "Sailors on...                                                                                             |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                                          | `major concerns`       | A serene scene of two sailors navigating calm waters in a Vortex catamaran and Laser dinghy, set against a backdrop of dense green woodland, under a clear sky, with a sailboat and a man in a life jacket in the distance.                                                                                                                                                               |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)                        | `major concerns`       | Title: Two Sailors Navigate Calm Waters in a Forested Estuary \| Description: On a sunny day in September 2026, two sailors, one in a Vortex catamaran and the o... \| Keywords (343): Sailboat, Catamaran, Estuary, Forest, Sailing, Sailors, Vortex, Laser, GBR, 1067, 188572, Sail, Boat, Water, ...                                                                                   |
| [`mlx-community/MiniCPM-V-4.6-4bit`](#model-mlx-community-minicpm-v-46-4bit)                                                        | `major concerns`       | Title: Sailing vessels on calm waters \| Description: Two sailors are sailing small dinghies, a Vortex catamaran and a Laser dinghy, across calm water with green forest... \| Keywords (12): boats, sailing, dinghy, catamaran, water, nature, forest, sail, person, life jacket, outdoor, recreation                                                                                    |
| [`mlx-community/Molmo2-8B-4bit`](#model-mlx-community-molmo2-8b-4bit)                                                               | `major concerns`       | Title: Catamaran and Laser Dinghy Sail on Calm River \| Description: Two sailors navigate small dinghies across a tranquil river, with a Vortex catamaran on the... \| Keywords (52): Boat, Boating, Catamaran, Dinghy, Estuary, Forest, Laser dinghy, Life jacket, Man, Mast, Outdoor recreation, ...                                                                                    |
| [`mlx-community/Muse-Glimmer-30B-OptiQ-4bit`](#model-mlx-community-muse-glimmer-30b-optiq-4bit)                                     | `major concerns`       | Title: (not detected) \| Description: (not detected) \| Keywords (1): Need British English.                                                                                                                                                                                                                                                                                               |
| [`mlx-community/Qwen2-VL-7B-Instruct-4bit`](#model-mlx-community-qwen2-vl-7b-instruct-4bit)                                         | `major concerns`       | Title: Sailing Adventure \| Description: Two sailors navigate a Vortex catamaran (sail number 1067) and a Laser dinghy (sail number GBR 188572) across calm waters,... \| Keywords (67): Sailing, Catamaran, Laser dinghy, Vortex, Sail number, Water, Trees, Forest, Woodland, Adventure, Outdoor, ...                                                                                   |
| [`mlx-community/SmolVLM-256M-Instruct-4bit`](#model-mlx-community-smolvlm-256m-instruct-4bit)                                       | `major concerns`       | A 5-10-word, 1-2-sentence, factual description combining relevant context with the main visible subject, setting, action, lighting, and distinctive details.                                                                                                                                                                                                                              |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                                       | `major concerns`       | Title: Sailing Catamaran and Laser Dinghy on Calm Waters \| Description: Two sailors navigate a Vortex catamaran (sail number 1067) and a Laser dinghy (sail number GBR 18857... \| Keywords (39): Sailing, Catamaran, Laser dinghy, Vortex, Sail number 1067, Sail number GBR 188572, Calm waters, ...                                                                                   |
| [`mlx-community/gemma-3n-E4B-it-4bit`](#model-mlx-community-gemma-3n-e4b-it-4bit)                                                   | `major concerns`       | Two sailors are engaged in a sailing competition on a calm body of water, likely an estuary or a sheltered bay, surrounded by lush green vegetation. On the left, a vibrant orange and white catamaran, identified by the sail number 1067 and the name "VORTEX," is being steered by...                                                                                                  |
| [`mlx-community/granite-vision-3.2-2b-nvfp4`](#model-mlx-community-granite-vision-32-2b-nvfp4)                                      | `major concerns`       | Title: "Harmony on the Water" \| Description: Two sailors navigate their small dinghies, a Vortex catamaran and a Laser dinghy, across calm waters, with a dense green woodland backdrop. The scene is set in an estuary, with a clear sky overhead. The sailors are equ... \| Keywords: (not detected)                                                                                   |
| [`mlx-community/llm-jp-4-vl-9b-mlx-4bit`](#model-mlx-community-llm-jp-4-vl-9b-mlx-4bit)                                             | `major concerns`       | <\|channel\|> analysis<\|message\|> The image shows two small sailboats racing on a river.                                                                                                                                                                                                                                                                                                |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                                        | `major concerns`       | "Boating in the Countryside: A Glimpse of Sailboats and Forests"                                                                                                                                                                                                                                                                                                                          |
| [`mlx-community/InternVL3_5-1B-4bit`](#model-mlx-community-internvl35-1b-4bit)                                                      | `not assessed`         | Model loading failed: Model type internvl not supported. Error: No module named 'mlx_vlm.speculative.drafters.internvl'                                                                                                                                                                                                                                                                   |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Run Stamps

- `mlx-vlm`: `0.7.3`
- `mlx`: `0.32.3.dev20260926+a2a09fd56`
- `transformers`: `5.17.0`
- `tokenizers`: `0.23.2`
- `huggingface-hub`: `1.33.0`
- *Python Version:* 3.14.7
- *OS:* Darwin 27.0.0
- *macOS Version:* 27.0
- *GPU/Chip:* Apple M5 Max
- *MLX Device:* Apple M5 Max
- *GPU Architecture:* applegpu_g17s
- *RAM:* 128.0 GB
- *Recommended Working Set:* 108 GB
- *Fused Attention:* Available

## Image Metadata

- *Description:* Two sailors steer small dinghies—a Vortex catamaran (sail
  number 1067) on the left and a Laser dinghy (sail number GBR 188572) on the
  right—across calm coastal or river waters against a backdrop of dense green
  woodland.
- *Keywords:* Boat, Boating, Catamaran, Clouds, Dinghy, Estuary, Forest, Laser
  dinghy, Life jacket, Man, Mast, Outdoor recreation, River, Sailboat,
  Sailing, Sailor, Shoreline, Sky, Trees, Water, Water sports, Yacht, active
  lifestyle, adventure, aquatic sports, boat race, buoyancy aid, calm water,
  coastal, competition, daytime, hobby, lake, leisure, maritime, nature,
  navigation, outdoor, recreation, regatta, rigging, sail, sailing boat,
  scenic, single-handed, skiff, sport, summer, vessel, water sport,
  watercraft, watersport, yachting
- *Date:* 2026-09-19 17:12:46 UTC+01:00
- *Time:* 17:12:46

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
> &#45; Capture date/time: 2026-09-19 17:12:46 UTC+01:00
>
> &#8203;Descriptive hints:
> &#45; Description hint: Two sailors steer small dinghies—a Vortex catamaran
> (sail number 1067) on the left and a Laser dinghy (sail number GBR 188572)
> on the right—across calm coastal or river waters against a backdrop of dense
> green woodland.
> &#45; Keyword hints: Boat, Boating, Catamaran, Clouds, Dinghy, Estuary, Forest,
> Laser dinghy, Life jacket, Man, Mast, Outdoor recreation, River, Sailboat,
> Sailing, Sailor, Shoreline, Sky, Trees, Water
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
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type lfm2-vl via lfm2_vl)
- *Model load time:* 0.41s
- *Generation time:* 1.08s
- *Total time:* 2.39s
- *Prompt preparation time:* 0.528
- *First-token latency:* 0.279
- *Cleanup time:* 0.0884
- *Prompt tokens:* 2,119
- *Generation tokens:* 141
- *Total tokens:* 2,260
- *Prompt throughput (raw):* 7,601 tok/s
- *Generation throughput (raw):* 475 tok/s
- *Peak memory (GB):* 1.9
- *Active memory (GB):* 0.90
- *Cache memory (GB):* 0.08
- *Model-load active memory (GB):* 0.897
- *Post-cleanup active memory (GB):* 0.00015
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1357
- *Image placeholders:* 1
- *Processor:* transformers.models.lfm2_vl.processing_lfm2_vl.Lfm2VlProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* ed71acdae0799c98cb450787d049d1343b65cd52
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--LiquidAI--LFM2.5-VL-450M-MLX-bf16/snapshots/ed71acdae0799c98cb450787d049d1343b65cd52
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
Sailboats on a River

Description:
Two sailboats glide across a calm river, surrounded by dense green forest and a partly cloudy sky. The sailboat on the left, a Vortex catamaran, has a prominent sail marked with the number 1067 and a life jacket on the man steering it. The sailboat on the right, a Laser dinghy, features a white sail with the number 188572 and the letters GBR.

Keywords:
Boat, Boating, Catamaran, Sailboat, Sailing, Life jacket, Man, Sail, Sails, Water, Forest, Sky, Trees, River, Clouds
</pre>

</details>

---

<a id="model-mlx-community-devstral-small-2-24b-instruct-2512-5bit"></a>

### mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

<details>
<summary>Complete evidence: mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type mistral3)
- *Model load time:* 4.99s
- *Generation time:* 10.09s
- *Total time:* 15.65s
- *Prompt preparation time:* 0.564
- *First-token latency:* 4.31
- *Cleanup time:* 0.14
- *Prompt tokens:* 2,394
- *Generation tokens:* 123
- *Total tokens:* 2,517
- *Prompt throughput (raw):* 556 tok/s
- *Generation throughput (raw):* 29.3 tok/s
- *Peak memory (GB):* 23
- *Active memory (GB):* 18
- *Cache memory (GB):* 0.47
- *Model-load active memory (GB):* 18
- *Post-cleanup active memory (GB):* 0.000412
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1306
- *Processor:* mlx_vlm.models.mistral3.processing_mistral3.Mistral3Processor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 0a970d20ad7da57b0e2accc35c5b8628f5d02063
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Devstral-Small-2-24B-Instruct-2512-5bit/snapshots/0a970d20ad7da57b0e2accc35c5b8628f5d02063
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.15, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Title:
Two sailors on a catamaran and dinghy

Description:
Two sailors navigate a catamaran (sail number 1067) and a Laser dinghy (sail number GBR 188572) on calm waters, surrounded by dense green woodland under a partly cloudy sky.

Keywords:
Boat, Boating, Catamaran, Clouds, Dinghy, Estuary, Forest, Laser dinghy, Life jacket, Man, Mast, Outdoor recreation, River, Sailboat, Sailing, Sailor, Shoreline, Sky, Trees, Water
</pre>

</details>

---

<a id="model-mlx-community-glm-46v-flash-4bit"></a>

### mlx-community/GLM-4.6V-Flash-4bit

<details>
<summary>Complete evidence: mlx-community/GLM-4.6V-Flash-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v)
- *Model load time:* 1.45s
- *Generation time:* 8.52s
- *Total time:* 10.58s
- *Prompt preparation time:* 0.605
- *First-token latency:* 6.08
- *Cleanup time:* 0.108
- *Prompt tokens:* 6,454
- *Generation tokens:* 139
- *Total tokens:* 6,593
- *Prompt throughput (raw):* 1,062 tok/s
- *Generation throughput (raw):* 73.6 tok/s
- *Peak memory (GB):* 8.7
- *Active memory (GB):* 7.1
- *Cache memory (GB):* 0.47
- *Model-load active memory (GB):* 7.07
- *Post-cleanup active memory (GB):* 0.00125
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1387
- *Image placeholders:* 1
- *Processor:* mlx_vlm.models.glm4v.processing.Glm46VProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* bd7b20686e8cbf22ad221a3de8399cdc5a2a7e0e
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--GLM-4.6V-Flash-4bit/snapshots/bd7b20686e8cbf22ad221a3de8399cdc5a2a7e0e
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.8, "top_k": 2, "top_p": 0.6}
- *EOS token:* <\|endoftext\|>

*Readable output:*

<pre class="model-output-readable">
Title: Two Sailors on Dinghies
Description: Two sailors steer small dinghies—a Vortex catamaran (sail number 1067) on the left and a Laser dinghy (sail number GBR 188572) on the right—across calm waters against a backdrop of dense green woodland, captured on 2026-09-19 at 17:12:46.
Keywords: Boat, Boating, Catamaran, Dinghy, Estuary, Forest, Laser dinghy, Life jacket, Man, Mast, Outdoor recreation, River, Sailboat, Sailing, Sailor, Shoreline, Sky, Trees, Water
</pre>

</details>

---

<a id="model-mlx-community-glm-46v-nvfp4"></a>

### mlx-community/GLM-4.6V-nvfp4

<details>
<summary>Complete evidence: mlx-community/GLM-4.6V-nvfp4</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v_moe)
- *Model load time:* 13.04s
- *Generation time:* 22.11s
- *Total time:* 35.98s
- *Prompt preparation time:* 0.815
- *First-token latency:* 16.3
- *Cleanup time:* 0.294
- *Prompt tokens:* 6,454
- *Generation tokens:* 142
- *Total tokens:* 6,596
- *Prompt throughput (raw):* 395 tok/s
- *Generation throughput (raw):* 40.2 tok/s
- *Peak memory (GB):* 78
- *Active memory (GB):* 62
- *Cache memory (GB):* 1.4
- *Model-load active memory (GB):* 61.9
- *Post-cleanup active memory (GB):* 0.00156
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1387
- *Image placeholders:* 1
- *Processor:* mlx_vlm.models.glm4v_moe.processing.Glm46VMoEProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 2da6855d4e28a0e61c84543262074bc17ac27d6e
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--GLM-4.6V-nvfp4/snapshots/2da6855d4e28a0e61c84543262074bc17ac27d6e
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.8, "top_k": 2, "top_p": 0.6}
- *EOS token:* <\|endoftext\|>

*Readable output:*

<pre class="model-output-readable">
Title: Two Sailors in Vortex and Laser Dinghies on Calm Waters

Description: Two sailors navigate a Vortex catamaran (sail number 1067) and a Laser dinghy (sail number GBR 188572) across calm river waters, with dense green woodland and a partly cloudy sky in the background; the catamaran sailor wears a life jacket, and both vessels are under sail.

Keywords: Boat, Boating, Catamaran, Clouds, Dinghy, Forest, Laser dinghy, Life jacket, Man, Mast, Outdoor recreation, River, Sailboat, Sailing, Sailor, Shoreline, Sky, Trees, Water
</pre>

</details>

---

<a id="model-mlx-community-internvl3-14b-4bit"></a>

### mlx-community/InternVL3-14B-4bit

<details>
<summary>Complete evidence: mlx-community/InternVL3-14B-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type internvl_chat)
- *Model load time:* 1.16s
- *Generation time:* 5.68s
- *Total time:* 7.52s
- *Prompt preparation time:* 0.678
- *First-token latency:* 2.96
- *Cleanup time:* 0.116
- *Prompt tokens:* 2,115
- *Generation tokens:* 116
- *Total tokens:* 2,231
- *Prompt throughput (raw):* 714 tok/s
- *Generation throughput (raw):* 56.3 tok/s
- *Peak memory (GB):* 10
- *Active memory (GB):* 8.9
- *Cache memory (GB):* 0.51
- *Model-load active memory (GB):* 8.94
- *Post-cleanup active memory (GB):* 0.00213
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1311
- *Image placeholders:* 1
- *Processor:* mlx_vlm.models.internvl_chat.processor.InternVLChatProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 26328eaab82cdc642fa3b73d8fa0b73db398710d
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--InternVL3-14B-4bit/snapshots/26328eaab82cdc642fa3b73d8fa0b73db398710d
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Sailing Dinghies on Calm Waters

Description: Two sailors navigate small dinghies, a Vortex catamaran (1067) and a Laser dinghy (GBR 188572), on calm waters with a backdrop of dense green woodland.

Keywords: Boat, Boating, Catamaran, Clouds, Dinghy, Estuary, Forest, Laser dinghy, Life jacket, Man, Mast, Outdoor recreation, River, Sailboat, Sailing, Sailor, Shoreline, Sky, Trees, Water
</pre>

</details>

---

<a id="model-mlx-community-internvl3-8b-bf16"></a>

### mlx-community/InternVL3-8B-bf16

<details>
<summary>Complete evidence: mlx-community/InternVL3-8B-bf16</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type internvl_chat)
- *Model load time:* 1.61s
- *Generation time:* 4.52s
- *Total time:* 6.81s
- *Prompt preparation time:* 0.676
- *First-token latency:* 1.06
- *Cleanup time:* 0.113
- *Prompt tokens:* 2,115
- *Generation tokens:* 103
- *Total tokens:* 2,218
- *Prompt throughput (raw):* 2,002 tok/s
- *Generation throughput (raw):* 36.6 tok/s
- *Peak memory (GB):* 17
- *Active memory (GB):* 16
- *Cache memory (GB):* 0.17
- *Model-load active memory (GB):* 15.9
- *Post-cleanup active memory (GB):* 0.00244
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1311
- *Image placeholders:* 1
- *Processor:* mlx_vlm.models.internvl_chat.processor.InternVLChatProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* e0df3dd79263467173214b67ef6d6a0cc5a475fd
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--InternVL3-8B-bf16/snapshots/e0df3dd79263467173214b67ef6d6a0cc5a475fd
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Sailing on Calm Waters with Catamaran and Laser Dinghy

Description: Two sailors navigate a Vortex catamaran and a Laser dinghy on calm waters near a forested shoreline during the late afternoon.

Keywords: Sailing, Catamaran, Laser dinghy, Vortex, GBR, 188572, 1067, Dinghy, Life jacket, Outdoor recreation, River, Trees, Water, Forest, Mast, Sails, Man
</pre>

</details>

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-2506-8bit"></a>

### mlx-community/Kimi-VL-A3B-Thinking-2506-8bit

<details>
<summary>Complete evidence: mlx-community/Kimi-VL-A3B-Thinking-2506-8bit</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type kimi_vl)
- *Model load time:* 2.36s
- *Generation time:* 15.60s
- *Total time:* 18.59s
- *Prompt preparation time:* 0.614
- *First-token latency:* 2.91
- *Cleanup time:* 0.146
- *Prompt tokens:* 1,331
- *Generation tokens:* 729
- *Total tokens:* 2,060
- *Prompt throughput (raw):* 457 tok/s
- *Generation throughput (raw):* 60.7 tok/s
- *Peak memory (GB):* 20
- *Active memory (GB):* 18
- *Cache memory (GB):* 1.5
- *Model-load active memory (GB):* 17.8
- *Post-cleanup active memory (GB):* 0.0031
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1493
- *Processor:* mlx_vlm.models.kimi_vl.processing_kimi_vl.KimiVLProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* e5abbe34cbfabd829fafd0362856e5b468d19f85
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Kimi-VL-A3B-Thinking-2506-8bit/snapshots/e5abbe34cbfabd829fafd0362856e5b468d19f85
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.6, "top_p": 1.0}
- *EOS token:* [EOS]

*Readable output:*

<pre class="model-output-readable">
◁think▷Okay, let's tackle this task step by step. First, I need to create a British-English catalogue metadata based on the provided context and image. The key points from the context are:

1. **Authoritative context**:
   - Capture date/time: 2026-09-19 17:12:46 UTC+01:00
   - Descriptive hints: Two sailors steer dinghies (Vortex catamaran with sail number 1067 on the left, Laser dinghy GBR 188572 on the right) across calm waters with dense green woodland in the background.
   - Keywords: Boat, Boating, Catamaran, Clouds, Dinghy, Estuary, Forest, Laser dinghy, Life jacket, Man, Mast, Outdoor recreation, River, Sailboat, Sailor, Shoreline, Sky, Trees, Water

2. **Image details**:
   - Two sailboats: Vortex catamaran (sail 1067) and Laser dinghy (GBR 188572)
   - Sailors on each boat
   - Calm waters, likely an estuary or river
   - Dense green woodland in the background
   - Sky with clouds
   - Other boats in the distance

3. **Constraints**:
   - Treat capture date/time and GPS as facts but not visible
   - Descriptive hints may be incomplete or wrong: retain image details, correct conflicts, add important visible details
   - Prefer image evidence over hints, omit uncertain details

Now, let's process the image:

**Title**: Need a 5-10 word title. Focus on the main elements: two dinghies, sailors, setting. Maybe "Two Sailors Navigate Dinghies in Calm Waters Near Forested Shoreline"

**Description**: A concise 1-2 sentence description combining context and visible details. Mention the two dinghies (catamaran and Laser), sailors, calm waters, forested shoreline, and the time (though not visible but authoritative context). So: "Two sailors navigate a Vortex catamaran (sail 1067) and a Laser dinghy (GBR 188572) across calm waters near a forested shoreline under a partly cloudy sky."

**Keywords**: Need 10-18 unique keywords from the context and image. Let's list them:
- Boat
- Boating
- Catamaran
- Dinghy
- Estuary
- Forest
- Laser dinghy
- Life jacket
- Man
- Mast
- Outdoor recreation
- River
- Sailboat
- Sailor
- Shoreline
- Sky
- Trees
- Water
- Vortex catamaran

Now, check if all important elements are covered and no conflicting info. The title and description are clear. Keywords are all from the context and image, no conflicts. Make sure to retain all key elements without adding uncertain info.◁/think▷Title: Two Sailors Navigate dinghies in Calm Waters Near Forested Shoreline
Description: Two sailors steer a Vortex catamaran (sail 1067) and a Laser dinghy (GBR 188572) across calm waters near a forested shoreline under a partly cloudy sky.
Keywords: Boat, Boating, Catamaran, Dinghy, Estuary, Forest, Laser dinghy, Life jacket, Man, Mast, Outdoor recreation, River, Sailboat, Sailor, Shoreline, Sky, Trees, Water, Vortex catamaran
</pre>

<details>
<summary>Exact raw output</summary>

```text
◁think▷Okay, let's tackle this task step by step. First, I need to create a British-English catalogue metadata based on the provided context and image. The key points from the context are:

1. **Authoritative context**: 
   - Capture date/time: 2026-09-19 17:12:46 UTC+01:00
   - Descriptive hints: Two sailors steer dinghies (Vortex catamaran with sail number 1067 on the left, Laser dinghy GBR 188572 on the right) across calm waters with dense green woodland in the background.
   - Keywords: Boat, Boating, Catamaran, Clouds, Dinghy, Estuary, Forest, Laser dinghy, Life jacket, Man, Mast, Outdoor recreation, River, Sailboat, Sailor, Shoreline, Sky, Trees, Water

2. **Image details**: 
   - Two sailboats: Vortex catamaran (sail 1067) and Laser dinghy (GBR 188572)
   - Sailors on each boat
   - Calm waters, likely an estuary or river
   - Dense green woodland in the background
   - Sky with clouds
   - Other boats in the distance

3. **Constraints**:
   - Treat capture date/time and GPS as facts but not visible
   - Descriptive hints may be incomplete or wrong: retain image details, correct conflicts, add important visible details
   - Prefer image evidence over hints, omit uncertain details

Now, let's process the image:

**Title**: Need a 5-10 word title. Focus on the main elements: two dinghies, sailors, setting. Maybe "Two Sailors Navigate Dinghies in Calm Waters Near Forested Shoreline"

**Description**: A concise 1-2 sentence description combining context and visible details. Mention the two dinghies (catamaran and Laser), sailors, calm waters, forested shoreline, and the time (though not visible but authoritative context). So: "Two sailors navigate a Vortex catamaran (sail 1067) and a Laser dinghy (GBR 188572) across calm waters near a forested shoreline under a partly cloudy sky."

**Keywords**: Need 10-18 unique keywords from the context and image. Let's list them:
- Boat
- Boating
- Catamaran
- Dinghy
- Estuary
- Forest
- Laser dinghy
- Life jacket
- Man
- Mast
- Outdoor recreation
- River
- Sailboat
- Sailor
- Shoreline
- Sky
- Trees
- Water
- Vortex catamaran

Now, check if all important elements are covered and no conflicting info. The title and description are clear. Keywords are all from the context and image, no conflicts. Make sure to retain all key elements without adding uncertain info.◁/think▷Title: Two Sailors Navigate dinghies in Calm Waters Near Forested Shoreline  
Description: Two sailors steer a Vortex catamaran (sail 1067) and a Laser dinghy (GBR 188572) across calm waters near a forested shoreline under a partly cloudy sky.  
Keywords: Boat, Boating, Catamaran, Dinghy, Estuary, Forest, Laser dinghy, Life jacket, Man, Mast, Outdoor recreation, River, Sailboat, Sailor, Shoreline, Sky, Trees, Water, Vortex catamaran
```

</details>

</details>

---

<a id="model-mlx-community-lfm25-vl-3b-optiq-4bit"></a>

### mlx-community/LFM2.5-VL-3B-OptiQ-4bit

<details>
<summary>Complete evidence: mlx-community/LFM2.5-VL-3B-OptiQ-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type lfm2_vl)
- *Model load time:* 1.16s
- *Generation time:* 1.95s
- *Total time:* 3.83s
- *Prompt preparation time:* 0.711
- *First-token latency:* 0.726
- *Cleanup time:* 0.0948
- *Prompt tokens:* 2,111
- *Generation tokens:* 110
- *Total tokens:* 2,221
- *Prompt throughput (raw):* 2,909 tok/s
- *Generation throughput (raw):* 206 tok/s
- *Peak memory (GB):* 4.0
- *Active memory (GB):* 2.8
- *Cache memory (GB):* 0.08
- *Model-load active memory (GB):* 2.81
- *Post-cleanup active memory (GB):* 0.00336
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1357
- *Image placeholders:* 1
- *Processor:* transformers.models.lfm2_vl.processing_lfm2_vl.Lfm2VlProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 7886c0b4a4b529265c226a45aa8763ea9ac46ebd
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--LFM2.5-VL-3B-OptiQ-4bit/snapshots/7886c0b4a4b529265c226a45aa8763ea9ac46ebd
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Sailors race dinghies across calm waters.
Description: Two sailors compete in a Vortex catamaran and Laser dinghy against a backdrop of dense green woodland. The scene is set on a calm day with a blue sky and scattered clouds.
Keywords: Boat, Boating, Catamaran, Clouds, Dinghy, Estuary, Forest, Laser dinghy, Life jacket, Man, Mast, Outdoor recreation, River, Sailboat, Sailing, Sailor, Shoreline, Sky, Trees
</pre>

</details>

---

<a id="model-mlx-community-minicpm-o-45-4bit"></a>

### mlx-community/MiniCPM-o-4_5-4bit

<details>
<summary>Complete evidence: mlx-community/MiniCPM-o-4_5-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type minicpmo)
- *Model load time:* 1.03s
- *Generation time:* 2.05s
- *Total time:* 3.62s
- *Prompt preparation time:* 0.522
- *First-token latency:* 0.457
- *Cleanup time:* 0.112
- *Prompt tokens:* 393
- *Generation tokens:* 114
- *Total tokens:* 507
- *Prompt throughput (raw):* 860 tok/s
- *Generation throughput (raw):* 104 tok/s
- *Peak memory (GB):* 7.0
- *Active memory (GB):* 6.1
- *Cache memory (GB):* 0.09
- *Model-load active memory (GB):* 6.15
- *Post-cleanup active memory (GB):* 0.0047
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1361
- *Image placeholders:* 1
- *Processor:* mlx_vlm.models.minicpmo.processing_minicpmo.MiniCPMOProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 592c09d85e7bffbc8f8e45a417aa8e9a96119814
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--MiniCPM-o-4_5-4bit/snapshots/592c09d85e7bffbc8f8e45a417aa8e9a96119814
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.6, "top_k": 20, "top_p": 0.95}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Sailboats on Calm Water near Green Woodland

Description: Two sailors navigate a Vortex catamaran and Laser dinghy across tranquil waters, with dense forest and scattered clouds under daylight conditions, showcasing outdoor sailing activity near a shoreline.

Keywords: Sailboat, Dinghy, Sailor, Vortex catamaran, Laser dinghy, Sail number 1067, Sail number 188572, Life jacket, Mast, Water, Forest, Shoreline, Estuary, Boating, Outdoor recreation, Sky, Clouds
</pre>

</details>

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-mxfp4"></a>

### mlx-community/Ministral-3-14B-Instruct-2512-mxfp4

<details>
<summary>Complete evidence: mlx-community/Ministral-3-14B-Instruct-2512-mxfp4</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type mistral3)
- *Model load time:* 1.54s
- *Generation time:* 5.18s
- *Total time:* 7.30s
- *Prompt preparation time:* 0.566
- *First-token latency:* 2.2
- *Cleanup time:* 0.115
- *Prompt tokens:* 2,927
- *Generation tokens:* 160
- *Total tokens:* 3,087
- *Prompt throughput (raw):* 1,328 tok/s
- *Generation throughput (raw):* 65.5 tok/s
- *Peak memory (GB):* 13
- *Active memory (GB):* 8.0
- *Cache memory (GB):* 0.60
- *Model-load active memory (GB):* 8.01
- *Post-cleanup active memory (GB):* 0.00497
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 3744
- *Processor:* mlx_vlm.models.mistral3.processing_mistral3.Mistral3Processor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 7c992876448fc5426841a85c6eb951b056fe38d2
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Ministral-3-14B-Instruct-2512-mxfp4/snapshots/7c992876448fc5426841a85c6eb951b056fe38d2
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Title:
**Sailing Dinghies in Coastal Waters – Vortex and Laser**

Description:
Two sailors navigate a Vortex catamaran (sail number 1067) and a Laser dinghy (GBR 188572) on calm waters near a dense woodland shoreline, captured on 19 September 2026 under overcast daylight conditions.

Keywords:
Boating, coastal waters, dinghy sailing, Laser dinghy, life jackets, manoeuvring, outdoor recreation, sailing, shoreline woodland, small catamaran (Vortex), two-person dinghies, UK sail number (GBR), calm water, overcast sky, recreational sailing, 2026-09-19
</pre>

</details>

---

<a id="model-mlx-community-ministral-3-3b-instruct-2512-4bit"></a>

### mlx-community/Ministral-3-3B-Instruct-2512-4bit

<details>
<summary>Complete evidence: mlx-community/Ministral-3-3B-Instruct-2512-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type mistral3)
- *Model load time:* 1.13s
- *Generation time:* 2.39s
- *Total time:* 4.09s
- *Prompt preparation time:* 0.566
- *First-token latency:* 1.04
- *Cleanup time:* 0.0987
- *Prompt tokens:* 2,926
- *Generation tokens:* 152
- *Total tokens:* 3,078
- *Prompt throughput (raw):* 2,811 tok/s
- *Generation throughput (raw):* 188 tok/s
- *Peak memory (GB):* 7.8
- *Active memory (GB):* 2.8
- *Cache memory (GB):* 0.41
- *Model-load active memory (GB):* 2.75
- *Post-cleanup active memory (GB):* 0.00523
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 3743
- *Processor:* mlx_vlm.models.mistral3.processing_mistral3.Mistral3Processor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* a962dcb09eee4169c890e544c9eb938f1113fdee
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Ministral-3-3B-Instruct-2512-4bit/snapshots/a962dcb09eee4169c890e544c9eb938f1113fdee
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Title:
Coastal Sailing Adventure with Catamaran and Laser Dinghy

Description:
Two sailors navigate small boats—one a Vortex catamaran (sail number 1067) and the other a Laser dinghy (GBR 188572)—on calm waters under a partly cloudy sky, surrounded by dense green woodland. Both wear life jackets, with the catamaran equipped with a small outboard motor.

Keywords:
Catamaran, Coastal waters, Dinghy, Estuary, Forest, Laser dinghy, Life jacket, Man, Mast, Outdoor recreation, River, Sailboat, Sailing, Shoreline, Sky, Trees, Vortex boat, Watercraft, Wind conditions
</pre>

</details>

---

<a id="model-mlx-community-north-micro-vision-instruct-4bit"></a>

### mlx-community/North-Micro-Vision-Instruct-4bit

<details>
<summary>Complete evidence: mlx-community/North-Micro-Vision-Instruct-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type cohere_compass)
- *Model load time:* 1.06s
- *Generation time:* 3.45s
- *Total time:* 5.30s
- *Prompt preparation time:* 0.785
- *First-token latency:* 2.02
- *Cleanup time:* 0.105
- *Prompt tokens:* 4,085
- *Generation tokens:* 112
- *Total tokens:* 4,197
- *Prompt throughput (raw):* 2,025 tok/s
- *Generation throughput (raw):* 165 tok/s
- *Peak memory (GB):* 3.9
- *Active memory (GB):* 2.2
- *Cache memory (GB):* 0.65
- *Model-load active memory (GB):* 2.18
- *Post-cleanup active memory (GB):* 0.00647
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1437
- *Processor:* mlx_vlm.models.cohere_compass.processing_cohere_compass.CohereCompassProcessor
- *Tokenizer:* transformers.models.cohere.tokenization_cohere.CohereTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 87466363e6c5f57adf91c18c3a62c3c74765f8df
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--North-Micro-Vision-Instruct-4bit/snapshots/87466363e6c5f57adf91c18c3a62c3c74765f8df
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.7, "top_k": 20, "top_p": 0.8}
- *EOS token:* <\|END_OF_TURN_TOKEN\|>

*Readable output:*

<pre class="model-output-readable">
Title: Sailboats on Calm Waters

Description: Two sailors navigate small dinghies across tranquil waters, one steering a Vortex catamaran (sail number 1067) and the other a Laser dinghy (sail number GBR 1888572) amidst a backdrop of dense green woodland.

Keywords: Boat, Boating, Catamaran, Clouds, Dinghy, Estuary, Forest, Laser dinghy, Life jacket, Man, Mast, Outdoor recreation, River, Sailboat, Sailing, Sailor, Shoreline, Sky, Trees, Water
</pre>

</details>

---

<a id="model-mlx-community-ornith-15-35b-a3b-optiq-4bit"></a>

### mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit

<details>
<summary>Complete evidence: mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5_moe)
- *Model load time:* 2.65s
- *Generation time:* 4.25s
- *Total time:* 7.59s
- *Prompt preparation time:* 0.68
- *First-token latency:* 1.1
- *Cleanup time:* 0.175
- *Prompt tokens:* 1,291
- *Generation tokens:* 151
- *Total tokens:* 1,442
- *Prompt throughput (raw):* 1,173 tok/s
- *Generation throughput (raw):* 60.2 tok/s
- *Peak memory (GB):* 24
- *Active memory (GB):* 23
- *Cache memory (GB):* 0.14
- *Model-load active memory (GB):* 23.1
- *Post-cleanup active memory (GB):* 0.00698
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1397
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 4620fdbbd1e7a1f14f936d49f1aa012abcda4569
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Ornith-1.5-35B-A3B-OptiQ-4bit/snapshots/4620fdbbd1e7a1f14f936d49f1aa012abcda4569
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 1.0, "top_k": 20, "top_p": 0.95}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
Two Sailboats Racing on Calm Waters

Description:
On 19 September 2026, an orange Vortex catamaran (sail number 1067) helmed by a sailor in a blue jacket and life jacket sails alongside a white Laser dinghy (GBR 188572) helmed by a crew member in dark clothing, both crossing calm waters with dense green woodland and a moored sailing vessel beyond.

Keywords:
Sailing, Sailboat, Catamaran, Dinghy, Laser, Man, Sailor, Life jacket, Mast, Boat, Water, River, Estuary, Shoreline, Trees, Forest, Sky, Clouds, Outdoor recreation
</pre>

</details>

---

<a id="model-mlx-community-phi-35-vision-instruct-bf16"></a>

### mlx-community/Phi-3.5-vision-instruct-bf16

<details>
<summary>Complete evidence: mlx-community/Phi-3.5-vision-instruct-bf16</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type phi3_v)
- *Model load time:* 0.94s
- *Generation time:* 4.87s
- *Total time:* 6.37s
- *Prompt preparation time:* 0.55
- *First-token latency:* 0.686
- *Cleanup time:* 0.103
- *Prompt tokens:* 1,141
- *Generation tokens:* 136
- *Total tokens:* 1,277
- *Prompt throughput (raw):* 1,664 tok/s
- *Generation throughput (raw):* 37.1 tok/s
- *Peak memory (GB):* 9.3
- *Active memory (GB):* 8.3
- *Cache memory (GB):* 0.52
- *Model-load active memory (GB):* 8.3
- *Post-cleanup active memory (GB):* 0.00705
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1327
- *Image placeholders:* 1
- *Processor:* mlx_vlm.models.phi3_v.processing_phi3_v.Phi3VProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* d8da684308c275a86659e2b36a9189b2f4aec8ea
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Phi-3.5-vision-instruct-bf16/snapshots/d8da684308c275a86659e2b36a9189b2f4aec8ea
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|endoftext\|>

*Readable output:*

<pre class="model-output-readable">
Title: Sailors on Dinghies in Coastal Waters

Description: On September 19, 2026, two sailors navigate their respective dinghies, a Vortex catamaran and a Laser dinghy, across calm waters near a forested shoreline.

Keywords: Sailors, Dinghies, Vortex, Laser, Coastal Waters, Forest, Sailing, Trees, Water, Clouds, Man, Mast, Outdoor Recreation, Sailboat, Shoreline, Sky, Estuary, Life Jacket
</pre>

</details>

---

<a id="model-mlx-community-qwen3-vl-2b-thinking-bf16"></a>

### mlx-community/Qwen3-VL-2B-Thinking-bf16

<details>
<summary>Complete evidence: mlx-community/Qwen3-VL-2B-Thinking-bf16</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Model load time:* 0.81s
- *Generation time:* 26.65s
- *Total time:* 28.55s
- *Prompt preparation time:* 1.08
- *First-token latency:* 15.2
- *Cleanup time:* 0.11
- *Prompt tokens:* 16,551
- *Generation tokens:* 918
- *Total tokens:* 17,469
- *Prompt throughput (raw):* 1,088 tok/s
- *Generation throughput (raw):* 87.3 tok/s
- *Peak memory (GB):* 8.4
- *Active memory (GB):* 4.3
- *Cache memory (GB):* 2.7
- *Model-load active memory (GB):* 4.26
- *Post-cleanup active memory (GB):* 0.00767
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1386
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* c325e5ea14c215bb08fa0d668c81fa2581f9050b
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3-VL-2B-Thinking-bf16/snapshots/c325e5ea14c215bb08fa0d668c81fa2581f9050b
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "repetition_context_size": 20,
  "repetition_penalty": 1.0, "seed": 0, "temperature": 1.0, "thinking_budget":
  800, "thinking_end_token": "&lt;/think&gt;", "thinking_start_token":
  "&lt;think&gt;", "top_k": 20, "top_p": 0.95}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Got it, let's tackle this. First, the title needs to be concrete and 5-10 words. Looking at the image: two sailors on dinghies, Vortex catamaran (1067) and Laser (GBR 188572). Title could be "Sailors Steering Vortex &amp; Laser Dinghies on Calm Water".

Then description: combine context and visible details. Context is capture date/time but we don't need to include that in the description, but the main subject is the two boats. The description should mention the boats, action (steering), setting (calm waters, woodland), lighting (sunny, maybe), and distinctive details like the sail numbers and the life jackets. So: "Two sailors on Vortex catamaran (1067) and Laser dinghy (GBR 188572) steer small boats across calm water, with green woodland and blue sky."

Wait, check the context: capture date is given but not needed in description. The descriptive hints have Vortex catamaran, Laser dinghy, sail numbers, life jackets, etc. So the description should include those. Let's make sure to include key details: Vortex (with 1067), Laser (GBR 188572), sailors, calm water, woodland, sky.

Then keywords: need 10-18 unique, comma-separated. From context and visible: Boat, Boating, Catamaran, Clouds (there are clouds in the sky), Dinghy, Estuary (wait, the image says "coastal or river waters"—maybe estuary? But let's check: the background is dense trees, so maybe river or estuary. But "Estuary" is a keyword hint, so include that. Laser dinghy, Life jacket (on the Vortex sailor, the other is a sailor on Laser), Man, Mast (sails have masts), Outdoor recreation, River, Sailboat, Sailing, Sailor, Shoreline, Sky, Trees, Water. Let's list them:

Boat, Boating, Catamaran, Clouds, Dinghy, Estuary, Laser dinghy, Life jacket, Man, Mast, Outdoor recreation, River, Sailboat, Sailing, Sailor, Shoreline, Sky, Trees, Water.

Wait, need to check if all these are visible. Life jacket: yes, the Vortex sailor has one. Mast: both have masts. Sailor: visible. Sky: yes, blue sky with clouds. Trees: background. Water: yes. Estuary: the water looks like a river or estuary, but the hints say Estuary, so include. River: also possible. Let's make sure it's 10-18. Let's count: 1. Boat, 2. Boating, 3. Catamaran, 4. Clouds, 5. Dinghy, 6. Estuary, 7. Laser dinghy, 8. Life jacket, 9. Man, 10. Mast, 11. Outdoor recreation, 12. River, 13. Sailboat, 14. Sailing, 15. Sailor, 16. Shoreline, 17. Sky, 18. Trees, 19. Water. Wait, but need 10-18, so maybe cut some. Let's pick the essential ones:

Boat, Boating, Catamaran, Clouds, Dinghy, Estuary, Laser dinghy, Life jacket, Man, Mast, Outdoor recreation, River, Sailboat, Sailing, Sailor, Shoreline, Sky, Trees, Water. That's 19. Let's adjust: 10-18.

Wait, the context has "
&lt;/think&gt;

Title: Sailors Steering Vortex &amp; Laser Dinghies on Calm Water

Description: Two sailors steer Vortex (1067) and Laser dinghy (GBR 188572) across calm river waters, with dense woodland backdrop and blue sky under sunny conditions.

Keywords: Boat, Boating, Catamaran, Clouds, Dinghy, Estuary, Laser dinghy, Life jacket, Man, Mast, Outdoor recreation, River, Sailboat, Sailing, Sailor, Shoreline, Sky, Trees, Water
</pre>

<details>
<summary>Exact raw output</summary>

```text
Got it, let's tackle this. First, the title needs to be concrete and 5-10 words. Looking at the image: two sailors on dinghies, Vortex catamaran (1067) and Laser (GBR 188572). Title could be "Sailors Steering Vortex & Laser Dinghies on Calm Water".

Then description: combine context and visible details. Context is capture date/time but we don't need to include that in the description, but the main subject is the two boats. The description should mention the boats, action (steering), setting (calm waters, woodland), lighting (sunny, maybe), and distinctive details like the sail numbers and the life jackets. So: "Two sailors on Vortex catamaran (1067) and Laser dinghy (GBR 188572) steer small boats across calm water, with green woodland and blue sky."

Wait, check the context: capture date is given but not needed in description. The descriptive hints have Vortex catamaran, Laser dinghy, sail numbers, life jackets, etc. So the description should include those. Let's make sure to include key details: Vortex (with 1067), Laser (GBR 188572), sailors, calm water, woodland, sky.

Then keywords: need 10-18 unique, comma-separated. From context and visible: Boat, Boating, Catamaran, Clouds (there are clouds in the sky), Dinghy, Estuary (wait, the image says "coastal or river waters"—maybe estuary? But let's check: the background is dense trees, so maybe river or estuary. But "Estuary" is a keyword hint, so include that. Laser dinghy, Life jacket (on the Vortex sailor, the other is a sailor on Laser), Man, Mast (sails have masts), Outdoor recreation, River, Sailboat, Sailing, Sailor, Shoreline, Sky, Trees, Water. Let's list them:

Boat, Boating, Catamaran, Clouds, Dinghy, Estuary, Laser dinghy, Life jacket, Man, Mast, Outdoor recreation, River, Sailboat, Sailing, Sailor, Shoreline, Sky, Trees, Water.

Wait, need to check if all these are visible. Life jacket: yes, the Vortex sailor has one. Mast: both have masts. Sailor: visible. Sky: yes, blue sky with clouds. Trees: background. Water: yes. Estuary: the water looks like a river or estuary, but the hints say Estuary, so include. River: also possible. Let's make sure it's 10-18. Let's count: 1. Boat, 2. Boating, 3. Catamaran, 4. Clouds, 5. Dinghy, 6. Estuary, 7. Laser dinghy, 8. Life jacket, 9. Man, 10. Mast, 11. Outdoor recreation, 12. River, 13. Sailboat, 14. Sailing, 15. Sailor, 16. Shoreline, 17. Sky, 18. Trees, 19. Water. Wait, but need 10-18, so maybe cut some. Let's pick the essential ones:

Boat, Boating, Catamaran, Clouds, Dinghy, Estuary, Laser dinghy, Life jacket, Man, Mast, Outdoor recreation, River, Sailboat, Sailing, Sailor, Shoreline, Sky, Trees, Water. That's 19. Let's adjust: 10-18.

Wait, the context has "
</think>

Title: Sailors Steering Vortex & Laser Dinghies on Calm Water

Description: Two sailors steer Vortex (1067) and Laser dinghy (GBR 188572) across calm river waters, with dense woodland backdrop and blue sky under sunny conditions.

Keywords: Boat, Boating, Catamaran, Clouds, Dinghy, Estuary, Laser dinghy, Life jacket, Man, Mast, Outdoor recreation, River, Sailboat, Sailing, Sailor, Shoreline, Sky, Trees, Water
```

</details>

</details>

---

<a id="model-mlx-community-qwen3-vl-30b-a3b-instruct-4bit"></a>

### mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit

<details>
<summary>Complete evidence: mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl_moe)
- *Model load time:* 1.91s
- *Generation time:* 35.67s
- *Total time:* 38.61s
- *Prompt preparation time:* 1.01
- *First-token latency:* 33.1
- *Cleanup time:* 0.133
- *Prompt tokens:* 16,549
- *Generation tokens:* 128
- *Total tokens:* 16,677
- *Prompt throughput (raw):* 501 tok/s
- *Generation throughput (raw):* 75.2 tok/s
- *Peak memory (GB):* 23
- *Active memory (GB):* 18
- *Cache memory (GB):* 2.3
- *Model-load active memory (GB):* 18.3
- *Post-cleanup active memory (GB):* 0.00798
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1378
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 0555d34cb1ed80c0e61a5635194c70027b4c2ff3
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3-VL-30B-A3B-Instruct-4bit/snapshots/0555d34cb1ed80c0e61a5635194c70027b4c2ff3
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "repetition_context_size": 20, "repetition_penalty": 1.0, "seed": 0,
  "temperature": 0.7, "top_k": 20, "top_p": 0.8}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Two sailboats racing on calm water
Description: Two sailors compete in a race on small dinghies—a Vortex catamaran (number 1067) and a Laser dinghy (number GBR 188572)—on calm water. The scene is set against a backdrop of a dense green forest under a partly cloudy sky.
Keywords: Boat, Boating, Catamaran, Clouds, Dinghy, Forest, Laser dinghy, Life jacket, Man, Mast, Outdoor recreation, River, Sailboat, Sailing, Sailor, Shoreline, Sky, Trees, Water
</pre>

</details>

---

<a id="model-mlx-community-qwen3-vl-32b-instruct-4bit"></a>

### mlx-community/Qwen3-VL-32B-Instruct-4bit

<details>
<summary>Complete evidence: mlx-community/Qwen3-VL-32B-Instruct-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Model load time:* 1.95s
- *Generation time:* 70.42s
- *Total time:* 73.38s
- *Prompt preparation time:* 0.999
- *First-token latency:* 58.6
- *Cleanup time:* 0.186
- *Prompt tokens:* 16,549
- *Generation tokens:* 181
- *Total tokens:* 16,730
- *Prompt throughput (raw):* 282 tok/s
- *Generation throughput (raw):* 16.7 tok/s
- *Peak memory (GB):* 26
- *Active memory (GB):* 20
- *Cache memory (GB):* 5.5
- *Model-load active memory (GB):* 19.6
- *Post-cleanup active memory (GB):* 0.00829
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1378
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 6e5644d3ea4b953b5221ffd02339bf897041038a
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3-VL-32B-Instruct-4bit/snapshots/6e5644d3ea4b953b5221ffd02339bf897041038a
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "repetition_context_size": 20, "repetition_penalty": 1.0, "seed": 0,
  "temperature": 0.7, "top_k": 20, "top_p": 0.8}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Sailors in Vortex Catamaran and Laser Dinghy on Calm Water

Description: On 2026-09-19, two sailors navigate small dinghies on calm waters: a Vortex catamaran (sail number 1067) with an orange and grey sail, and a white Laser dinghy (sail number GBR 188572) with a red sun emblem, against a backdrop of dense green woodland under a partly cloudy sky.

Keywords: Sailboat, Sailing, Dinghy, Catamaran, Vortex, Laser, Sail number, GBR, 1067, 188572, Sailor, Water, River, Estuary, Forest, Trees, Shoreline, Sky, Clouds, Life jacket, Outdoor recreation, Mast, Boating, Boat
</pre>

</details>

---

<a id="model-mlx-community-qwen3-vl-8b-instruct-4bit"></a>

### mlx-community/Qwen3-VL-8B-Instruct-4bit

<details>
<summary>Complete evidence: mlx-community/Qwen3-VL-8B-Instruct-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Model load time:* 1.00s
- *Generation time:* 41.31s
- *Total time:* 43.38s
- *Prompt preparation time:* 1.07
- *First-token latency:* 38.7
- *Cleanup time:* 0.119
- *Prompt tokens:* 16,549
- *Generation tokens:* 104
- *Total tokens:* 16,653
- *Prompt throughput (raw):* 427 tok/s
- *Generation throughput (raw):* 67.6 tok/s
- *Peak memory (GB):* 11
- *Active memory (GB):* 5.8
- *Cache memory (GB):* 3.4
- *Model-load active memory (GB):* 5.77
- *Post-cleanup active memory (GB):* 0.0086
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1378
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* defcdea7cc7a4b0858fea563cbbce171d328e457
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3-VL-8B-Instruct-4bit/snapshots/defcdea7cc7a4b0858fea563cbbce171d328e457
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "repetition_context_size": 20, "repetition_penalty": 1.0, "seed": 0,
  "temperature": 0.7, "top_k": 20, "top_p": 0.8}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Sailors race dinghies on calm water

Description: Two sailors navigate a Vortex catamaran and Laser dinghy across tranquil waters, framed by dense green forest under a partly cloudy sky. The scene captures active sailing with visible sail numbers and life jackets.

Keywords: Sailboat, Dinghy, Catamaran, Laser, Sailing, Water, Forest, Trees, Shoreline, Sky, Clouds, Man, Sailor, Life jacket, Outdoor recreation, Estuary, Boat, Boating
</pre>

</details>

---

<a id="model-mlx-community-qwen35-35b-a3b-4bit"></a>

### mlx-community/Qwen3.5-35B-A3B-4bit

<details>
<summary>Complete evidence: mlx-community/Qwen3.5-35B-A3B-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5_moe)
- *Model load time:* 2.76s
- *Generation time:* 36.85s
- *Total time:* 40.68s
- *Prompt preparation time:* 1.06
- *First-token latency:* 34.4
- *Cleanup time:* 0.155
- *Prompt tokens:* 16,565
- *Generation tokens:* 111
- *Total tokens:* 16,676
- *Prompt throughput (raw):* 482 tok/s
- *Generation throughput (raw):* 73.5 tok/s
- *Peak memory (GB):* 25
- *Active memory (GB):* 20
- *Cache memory (GB):* 0.90
- *Model-load active memory (GB):* 20.4
- *Post-cleanup active memory (GB):* 0.00911
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1397
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 1e20fd8d42056f870933bf98ca6211024744f7ec
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 1.0, "top_k": 20, "top_p": 0.95}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Vortex Catamaran and Laser Sailing on Water

Description: Two sailors navigate a Vortex catamaran and a Laser dinghy on calm river waters on 19 September 2026. The scene is framed by dense green woodland along the shoreline under a partly cloudy sky.

Keywords: Vortex catamaran, Laser dinghy, sailors, calm water, river, shoreline, dense green woodland, partly cloudy sky, outdoor recreation, sailing, man, mast, forest, Boating, sailboat, Water
</pre>

</details>

---

<a id="model-mlx-community-qwen35-9b-mlx-4bit"></a>

### mlx-community/Qwen3.5-9B-MLX-4bit

<details>
<summary>Complete evidence: mlx-community/Qwen3.5-9B-MLX-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5)
- *Model load time:* 1.72s
- *Generation time:* 37.08s
- *Total time:* 39.83s
- *Prompt preparation time:* 1.03
- *First-token latency:* 34.5
- *Cleanup time:* 0.16
- *Prompt tokens:* 16,565
- *Generation tokens:* 142
- *Total tokens:* 16,707
- *Prompt throughput (raw):* 480 tok/s
- *Generation throughput (raw):* 84.5 tok/s
- *Peak memory (GB):* 11
- *Active memory (GB):* 6.0
- *Cache memory (GB):* 1.2
- *Model-load active memory (GB):* 5.96
- *Post-cleanup active memory (GB):* 0.00964
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1397
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 938d8919941c6e7efd3c7150eff7fe9d12afa631
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-MLX-4bit/snapshots/938d8919941c6e7efd3c7150eff7fe9d12afa631
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
Two Sailors Navigate Vortex Catamaran and Laser Dinghy on Calm Waters

Description:
Two sailors steer a Vortex catamaran (sail 1067) and a Laser dinghy (GBR 188572) across calm waters, with dense woodland lining the shore under an overcast sky.

Keywords:
Sailing, Dinghy, Catamaran, Vortex, Laser, Sailboat, Sailor, Life jacket, Mast, River, Estuary, Forest, Trees, Sky, Clouds, Outdoor recreation, Shoreline, Water, GBR, 1067, 188572
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title:  
Two Sailors Navigate Vortex Catamaran and Laser Dinghy on Calm Waters

Description:  
Two sailors steer a Vortex catamaran (sail 1067) and a Laser dinghy (GBR 188572) across calm waters, with dense woodland lining the shore under an overcast sky.

Keywords:  
Sailing, Dinghy, Catamaran, Vortex, Laser, Sailboat, Sailor, Life jacket, Mast, River, Estuary, Forest, Trees, Sky, Clouds, Outdoor recreation, Shoreline, Water, GBR, 1067, 188572
```

</details>

</details>

---

<a id="model-mlx-community-qwen38-27b-nvfp4"></a>

### mlx-community/Qwen3.8-27B-nvfp4

<details>
<summary>Complete evidence: mlx-community/Qwen3.8-27B-nvfp4</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5)
- *Model load time:* 2.11s
- *Generation time:* 63.04s
- *Total time:* 66.31s
- *Prompt preparation time:* 1.15
- *First-token latency:* 56.9
- *Cleanup time:* 0.159
- *Prompt tokens:* 16,565
- *Generation tokens:* 144
- *Total tokens:* 16,709
- *Prompt throughput (raw):* 291 tok/s
- *Generation throughput (raw):* 28.2 tok/s
- *Peak memory (GB):* 21
- *Active memory (GB):* 16
- *Cache memory (GB):* 1.9
- *Model-load active memory (GB):* 16.1
- *Post-cleanup active memory (GB):* 0.0101
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1397
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-nvfp4/snapshots/5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 1.0, "top_k": 20, "top_p": 0.95}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
Two sailors racing Vortex and Laser dinghies on calm waters

Description:
A Vortex catamaran with sail number 1067 on the left and a Laser dinghy with sail number GBR 188572 on the right are sailed by individuals across a calm estuary. Both boats are set against a shoreline of dense green woodland under a bright, cloudy sky, with other moored vessels visible in the background.

Keywords:
Vortex, Laser, catamaran, dinghy, sailing, sailors, estuary, woodland, boats, water, masts, GBR 188572, sail numbers, outdoor, sky
</pre>

</details>

---

<a id="model-mlx-community-smolvlm2-22b-instruct-mlx"></a>

### mlx-community/SmolVLM2-2.2B-Instruct-mlx

<details>
<summary>Complete evidence: mlx-community/SmolVLM2-2.2B-Instruct-mlx</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type smolvlm)
- *Model load time:* 0.65s
- *Generation time:* 1.97s
- *Total time:* 3.41s
- *Prompt preparation time:* 0.779
- *First-token latency:* 0.616
- *Cleanup time:* 0.103
- *Prompt tokens:* 1,433
- *Generation tokens:* 78
- *Total tokens:* 1,511
- *Prompt throughput (raw):* 2,327 tok/s
- *Generation throughput (raw):* 125 tok/s
- *Peak memory (GB):* 5.6
- *Active memory (GB):* 4.5
- *Cache memory (GB):* 0.35
- *Model-load active memory (GB):* 4.5
- *Post-cleanup active memory (GB):* 0.0104
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1338
- *Image placeholders:* 1
- *Processor:* mlx_vlm.models.smolvlm.processing_smolvlm.SmolVLMProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 844516024a1c4400d34489b89ee067d794e432ed
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--SmolVLM2-2.2B-Instruct-mlx/snapshots/844516024a1c4400d34489b89ee067d794e432ed
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <end_of_utterance>

*Readable output:*

<pre class="model-output-readable">
 Title: Sailing on the River
Description: Two sailors are sailing their boats across the river.
Keywords: Boat, Boating, Catamaran, Clouds, Dinghy, Estuary, Forest, Laser dinghy, Life jacket, Man, Mast, Outdoor recreation, River, Sailboat, Sailing, Sailor, Shoreline, Sky, Trees, Water
</pre>

</details>

---

<a id="model-mlx-community-step-37-flash-oq3e"></a>

### mlx-community/Step-3.7-Flash-oQ3e

<details>
<summary>Complete evidence: mlx-community/Step-3.7-Flash-oQ3e</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type step3p7)
- *Model load time:* 18.38s
- *Generation time:* 84.39s
- *Total time:* 106.26s
- *Prompt preparation time:* 3.48
- *First-token latency:* 75.5
- *Cleanup time:* 0.479
- *Prompt tokens:* 3,494
- *Generation tokens:* 147
- *Total tokens:* 3,641
- *Prompt throughput (raw):* 46.3 tok/s
- *Generation throughput (raw):* 44.0 tok/s
- *Peak memory (GB):* 92
- *Active memory (GB):* 85
- *Cache memory (GB):* 0.41
- *Model-load active memory (GB):* 85
- *Post-cleanup active memory (GB):* 0.0106
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1384
- *Processor:* mlx_vlm.models.step3p7.processing_step3p7.Step3VLProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 41d17ee00e168a2918bb839e4a7b6e445c6f03f1
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Step-3.7-Flash-oQ3e/snapshots/41d17ee00e168a2918bb839e4a7b6e445c6f03f1
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
Two sailors on dinghies across calm waters

Description:
On 19 September 2026 at 17:12 UTC+1, two sailors steer small dinghies across calm coastal or river waters, with a Vortex catamaran (sail number 1067) on the left and a Laser dinghy (sail number GBR 188572) on the right, set against a backdrop of dense green woodland under a partly cloudy sky.

Keywords:
Sailboat, Sailing, Sailor, Dinghy, Catamaran, Laser dinghy, Vortex, Boat, Boating, Water, River, Estuary, Shoreline, Forest, Trees, Mast, Life jacket, Outdoor recreation
</pre>

</details>

---

<a id="model-mlx-community-aya-vision-8b-4bit"></a>

### mlx-community/aya-vision-8b-4bit

<details>
<summary>Complete evidence: mlx-community/aya-vision-8b-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type aya_vision)
- *Model load time:* 1.25s
- *Generation time:* 3.36s
- *Total time:* 5.87s
- *Prompt preparation time:* 1.25
- *First-token latency:* 0.964
- *Cleanup time:* 0.194
- *Prompt tokens:* 2,089
- *Generation tokens:* 112
- *Total tokens:* 2,201
- *Prompt throughput (raw):* 2,167 tok/s
- *Generation throughput (raw):* 91.7 tok/s
- *Peak memory (GB):* 6.5
- *Active memory (GB):* 5.5
- *Cache memory (GB):* 0.34
- *Model-load active memory (GB):* 5.46
- *Post-cleanup active memory (GB):* 0.0115
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 4244
- *Image placeholders:* 1
- *Processor:* mlx_vlm.models.aya_vision.processing_aya_vision.AyaVisionOutputProcessor
- *Tokenizer:* transformers.models.cohere.tokenization_cohere.CohereTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 3e679b3e08f04d741c8965ebf0cb258295e8acf7
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--aya-vision-8b-4bit/snapshots/3e679b3e08f04d741c8965ebf0cb258295e8acf7
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|END_OF_TURN_TOKEN\|>

*Readable output:*

<pre class="model-output-readable">
Title: Sailing Adventure on the River

Description: Two sailors navigate their boats across a serene river, with one in a Vortex catamaran and the other in a Laser dinghy, both wearing life jackets and skillfully steering their vessels against the backdrop of dense green woodland. The scene captures a peaceful moment of outdoor recreation under a partly cloudy sky.

Keywords: Catamaran, Dinghy, River, Sailing, Trees, Water, Sky, Boat, Boating, Mast, Life jacket, Man, Outdoor recreation, Forest, Clouds, Estuary
</pre>

</details>

---

<a id="model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8"></a>

### mlx-community/diffusiongemma-26B-A4B-it-mxfp8

<details>
<summary>Complete evidence: mlx-community/diffusiongemma-26B-A4B-it-mxfp8</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type diffusion_gemma)
- *Model load time:* 3.69s
- *Generation time:* 4.20s
- *Total time:* 8.55s
- *Prompt preparation time:* 0.651
- *First-token latency:* 1.2
- *Cleanup time:* 0.204
- *Prompt tokens:* 592
- *Generation tokens:* 85
- *Total tokens:* 677
- *Prompt throughput (raw):* 493 tok/s
- *Generation throughput (raw):* 39.2 tok/s
- *Peak memory (GB):* 28
- *Active memory (GB):* 27
- *Cache memory (GB):* 0.01
- *Model-load active memory (GB):* 27.2
- *Post-cleanup active memory (GB):* 0.0115
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1332
- *Image placeholders:* 1
- *Processor:* mlx_vlm.models.diffusion_gemma.processing_diffusion_gemma.DiffusionGemma4Processor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* ded389e478f86d498ad9e7f47666e83b166a28f1
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--diffusiongemma-26B-A4B-it-mxfp8/snapshots/ded389e478f86d498ad9e7f47666e83b166a28f1
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
Title: Two Sailors Sailing Dinghies on Calm Water
Description: Two sailors steer a grey Vortex catamaran and a white Laser dinghy across calm waters against a dense woodland backdrop under a cloudy sky.
Keywords: Sailing, Sailboat, Catamaran, Dinghy, Sailor, Mast, Water, Forest, River, Outdoor Recreation, Boating, Estuary, Sky, Clouds, Life jacket,
</pre>

</details>

---

<a id="model-mlx-community-gemma-3-27b-it-qat-4bit"></a>

### mlx-community/gemma-3-27b-it-qat-4bit

<details>
<summary>Complete evidence: mlx-community/gemma-3-27b-it-qat-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type gemma3)
- *Model load time:* 2.93s
- *Generation time:* 7.65s
- *Total time:* 11.33s
- *Prompt preparation time:* 0.726
- *First-token latency:* 1.09
- *Cleanup time:* 0.252
- *Prompt tokens:* 591
- *Generation tokens:* 175
- *Total tokens:* 766
- *Prompt throughput (raw):* 543 tok/s
- *Generation throughput (raw):* 29.4 tok/s
- *Peak memory (GB):* 17
- *Active memory (GB):* 16
- *Cache memory (GB):* 0.44
- *Model-load active memory (GB):* 16.1
- *Post-cleanup active memory (GB):* 0.012
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1361
- *Image placeholders:* 1
- *Processor:* mlx_vlm.models.gemma3.processing_gemma3.Gemma3Processor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* fc4e000f32af1b7b6779294e490a7d2a80bac611
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--gemma-3-27b-it-qat-4bit/snapshots/fc4e000f32af1b7b6779294e490a7d2a80bac611
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.0, "top_k": 64, "top_p": 0.95}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
Title: Sailing Dinghies on Calm Water, September 2026

Description: Captured on 19th September 2026, this image shows a Vortex catamaran (sail number 1067) and a Laser dinghy (sail number GBR 188572) sailing on calm water, with dense woodland forming the backdrop. The scene is lit by bright daylight, revealing details of the boats, sailors, and shoreline.

Keywords: Boat, Boating, Catamaran, Clouds, Dinghy, Estuary, Forest, Laser dinghy, Life jacket, Man, Mast, Outdoor recreation, River, Sailboat, Sailing, Sailor, Shoreline, Sky, Trees, Water, Vortex, GBR 188572, 1067, Daylight, September, Woodland, Reflection
</pre>

</details>

---

<a id="model-mlx-community-gemma-4-26b-a4b-it-4bit"></a>

### mlx-community/gemma-4-26b-a4b-it-4bit

<details>
<summary>Complete evidence: mlx-community/gemma-4-26b-a4b-it-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type gemma4)
- *Model load time:* 2.81s
- *Generation time:* 2.96s
- *Total time:* 6.53s
- *Prompt preparation time:* 0.754
- *First-token latency:* 0.533
- *Cleanup time:* 0.169
- *Prompt tokens:* 596
- *Generation tokens:* 130
- *Total tokens:* 726
- *Prompt throughput (raw):* 1,117 tok/s
- *Generation throughput (raw):* 76.1 tok/s
- *Peak memory (GB):* 16
- *Active memory (GB):* 15
- *Cache memory (GB):* 0.22
- *Model-load active memory (GB):* 15.4
- *Post-cleanup active memory (GB):* 0.0136
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1360
- *Image placeholders:* 1
- *Processor:* mlx_vlm.models.gemma4.processing_gemma4.Gemma4Processor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 0d77464eeb233a2da68ebf9d7dc4edaac7db956d
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--gemma-4-26b-a4b-it-4bit/snapshots/0d77464eeb233a2da68ebf9d7dc4edaac7db956d
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 1.0, "top_k": 64, "top_p": 0.95}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
Title: Two sailors steering small boats across calm water

Description: Two sailors navigate small boats across calm waters against a backdrop of dense green woodland under a cloudy sky. On the left, a person in a blue life jacket steers a Vortex catamaran with sail number 1067, while on the right, another person steers a white Laser dinghy with sail number GBR 188572.

Keywords: Boat, Boating, Catamaran, Clouds, Dinghy, Forest, Laser dinghy, Life jacket, Man, Mast, Outdoor recreation, River, Sailboat, Sailing, Sailor, Sky, Trees, Water
</pre>

</details>

---

<a id="model-mlx-community-gemma-4-31b-it-4bit"></a>

### mlx-community/gemma-4-31b-it-4bit

<details>
<summary>Complete evidence: mlx-community/gemma-4-31b-it-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type gemma4)
- *Model load time:* 2.67s
- *Generation time:* 5.01s
- *Total time:* 8.31s
- *Prompt preparation time:* 0.627
- *First-token latency:* 1.09
- *Cleanup time:* 0.177
- *Prompt tokens:* 596
- *Generation tokens:* 89
- *Total tokens:* 685
- *Prompt throughput (raw):* 548 tok/s
- *Generation throughput (raw):* 26.8 tok/s
- *Peak memory (GB):* 20
- *Active memory (GB):* 18
- *Cache memory (GB):* 0.79
- *Model-load active memory (GB):* 18.4
- *Post-cleanup active memory (GB):* 0.0141
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1360
- *Image placeholders:* 1
- *Processor:* mlx_vlm.models.gemma4.processing_gemma4.Gemma4Processor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 696d436c404745a59f30e4939a658162b0a9e57f
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--gemma-4-31b-it-4bit/snapshots/696d436c404745a59f30e4939a658162b0a9e57f
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 1.0, "top_k": 64, "top_p": 0.95}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
Title: Two Sailors Steering Dinghies on Calm Water
Description: A Vortex catamaran and a Laser dinghy sail across calm waters against a backdrop of dense green woodland under a cloudy sky.
Keywords: Boat, Boating, Catamaran, Clouds, Dinghy, Estuary, Forest, Laser dinghy, Life jacket, Man, Mast, Outdoor recreation, River, Sailboat, Sailing, Sailor, Shoreline, Trees, Water
</pre>

</details>

---

<a id="model-mlx-community-gemma-4-e4b-it-4bit"></a>

### mlx-community/gemma-4-e4b-it-4bit

<details>
<summary>Complete evidence: mlx-community/gemma-4-e4b-it-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type gemma4)
- *Model load time:* 1.93s
- *Generation time:* 2.14s
- *Total time:* 4.82s
- *Prompt preparation time:* 0.728
- *First-token latency:* 0.47
- *Cleanup time:* 0.168
- *Prompt tokens:* 592
- *Generation tokens:* 94
- *Total tokens:* 686
- *Prompt throughput (raw):* 1,261 tok/s
- *Generation throughput (raw):* 96.4 tok/s
- *Peak memory (GB):* 5.9
- *Active memory (GB):* 5.2
- *Cache memory (GB):* 0.08
- *Model-load active memory (GB):* 5.16
- *Post-cleanup active memory (GB):* 0.0147
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1332
- *Image placeholders:* 1
- *Processor:* mlx_vlm.models.gemma4.processing_gemma4.Gemma4Processor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 475b9088d29754a3379866cf5aeb6b41acd313c2
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--gemma-4-e4b-it-4bit/snapshots/475b9088d29754a3379866cf5aeb6b41acd313c2
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 1.0, "top_k": 64, "top_p": 0.95}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
Title: Two Dinghies Sail on Calm Woodland Water
Description: Two small dinghies navigate placid waters beneath a backdrop of dense green woodland under partly cloudy skies. The catamaran on the left and the Laser dinghy on the right feature distinctive sail numbers.
Keywords: Catamaran, Dinghy, Laser, Sailing, Boating, Woodland, Estuary, Sailboat, Water, Outdoor, Recreation, Trees, Small Craft, Sky, Dinghy Racing
</pre>

</details>

---

<a id="model-mlx-community-granite-40-3b-vision-4bit"></a>

### mlx-community/granite-4.0-3b-vision-4bit

<details>
<summary>Complete evidence: mlx-community/granite-4.0-3b-vision-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type granite4_vision)
- *Model load time:* 0.71s
- *Generation time:* 2.86s
- *Total time:* 4.57s
- *Prompt preparation time:* 0.998
- *First-token latency:* 1.15
- *Cleanup time:* 0.138
- *Prompt tokens:* 1,383
- *Generation tokens:* 99
- *Total tokens:* 1,482
- *Prompt throughput (raw):* 1,199 tok/s
- *Generation throughput (raw):* 129 tok/s
- *Peak memory (GB):* 4.7
- *Active memory (GB):* 3.0
- *Cache memory (GB):* 0.22
- *Model-load active memory (GB):* 3.03
- *Post-cleanup active memory (GB):* 0.0149
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1530
- *Image placeholders:* 1
- *Processor:* mlx_vlm.models.granite4_vision.processing_granite4_vision.Granite4VisionProcessor
- *Tokenizer:* transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 70fe1d89f42c71b5635cd945ed304bed438fa1e3
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--granite-4.0-3b-vision-4bit/snapshots/70fe1d89f42c71b5635cd945ed304bed438fa1e3
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|end_of_text\|>

*Readable output:*

<pre class="model-output-readable">
Title: "Sailors in Dinghies on a Calm Day"
Description: Two sailors navigate their dinghies, a Vortex catamaran and a Laser dinghy, across a serene body of water with dense green woodland in the background.
Keywords: Sailors, Dinghies, Vortex catamaran, Laser dinghy, Water, Trees, Coastline, Sailing, Life jacket, Man, Sail number, 1067, 188572
</pre>

</details>

---

<a id="model-mlx-community-pixtral-12b-8bit"></a>

### mlx-community/pixtral-12b-8bit

<details>
<summary>Complete evidence: mlx-community/pixtral-12b-8bit</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type pixtral)
- *Model load time:* 1.83s
- *Generation time:* 5.46s
- *Total time:* 7.85s
- *Prompt preparation time:* 0.557
- *First-token latency:* 1.87
- *Cleanup time:* 0.135
- *Prompt tokens:* 3,117
- *Generation tokens:* 114
- *Total tokens:* 3,231
- *Prompt throughput (raw):* 1,670 tok/s
- *Generation throughput (raw):* 37.2 tok/s
- *Peak memory (GB):* 16
- *Active memory (GB):* 13
- *Cache memory (GB):* 0.59
- *Model-load active memory (GB):* 13.5
- *Post-cleanup active memory (GB):* 0.0163
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1306
- *Processor:* mlx_vlm.models.pixtral.processing_pixtral.PixtralProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 79e24b66302d4ac5e75491767f8914b2aa6ac1f8
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--pixtral-12b-8bit/snapshots/79e24b66302d4ac5e75491767f8914b2aa6ac1f8
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Title:
Sailors Navigate Calm Waters in Dinghies

Description:
Two sailors steer small dinghies—a Vortex catamaran and a Laser dinghy—across calm waters with dense green woodland in the background under a partly cloudy sky.

Keywords:
Boat, Boating, Catamaran, Clouds, Dinghy, Estuary, Forest, Laser dinghy, Life jacket, Man, Mast, Outdoor recreation, River, Sailboat, Sailing, Sailor, Shoreline, Sky, Trees, Water, Woodland
</pre>

</details>

---

<a id="model-nativ-community-mage-vl-optiq-4bit"></a>

### nativ-community/Mage-VL-OptiQ-4bit

<details>
<summary>Complete evidence: nativ-community/Mage-VL-OptiQ-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type mage_vl)
- *Model load time:* 0.83s
- *Generation time:* 3.28s
- *Total time:* 4.88s
- *Prompt preparation time:* 0.763
- *First-token latency:* 1.52
- *Cleanup time:* 0.113
- *Prompt tokens:* 4,212
- *Generation tokens:* 130
- *Total tokens:* 4,342
- *Prompt throughput (raw):* 2,764 tok/s
- *Generation throughput (raw):* 124 tok/s
- *Peak memory (GB):* 5.4
- *Active memory (GB):* 3.9
- *Cache memory (GB):* 0.72
- *Model-load active memory (GB):* 3.93
- *Post-cleanup active memory (GB):* 0.0166
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1436
- *Processor:* mlx_vlm.models.mage_vl.processing_mage_vl.MageVLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 4f0a424370e54984ed783db8f8633a219c72db78
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--nativ-community--Mage-VL-OptiQ-4bit/snapshots/4f0a424370e54984ed783db8f8633a219c72db78
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Two Sailboats Glide Across Calm Waters Amidst Lush Forest
Description: Two sailors navigate a Vortex catamaran (sail number 1067) and a Laser dinghy (sail number GBR 188572) on a serene estuary, with dense green woodland in the background under a partly cloudy sky.
Keywords: Boat, Boating, Catamaran, Clouds, Dinghy, Estuary, Forest, Laser dinghy, Life jacket, Man, Mast, Outdoor recreation, River, Sailboat, Sailing, Sailor, Shoreline, Sky, Trees, Water
</pre>

</details>

---

<a id="model-nativ-community-mistral-small-32-24b-instruct-2506-4bit"></a>

### nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit

<details>
<summary>Complete evidence: nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type mistral3)
- *Model load time:* 2.01s
- *Generation time:* 6.06s
- *Total time:* 8.66s
- *Prompt preparation time:* 0.589
- *First-token latency:* 1.7
- *Cleanup time:* 0.121
- *Prompt tokens:* 1,273
- *Generation tokens:* 134
- *Total tokens:* 1,407
- *Prompt throughput (raw):* 749 tok/s
- *Generation throughput (raw):* 35.0 tok/s
- *Peak memory (GB):* 18
- *Active memory (GB):* 15
- *Cache memory (GB):* 0.28
- *Model-load active memory (GB):* 15.1
- *Post-cleanup active memory (GB):* 0.0168
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1306
- *Processor:* mlx_vlm.models.mistral3.processing_mistral3.Mistral3Processor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* bdbeb0d8c89eb01efd49b01139eaa9f5fa3fe19b
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--nativ-community--Mistral-Small-3.2-24B-Instruct-2506-4bit/snapshots/bdbeb0d8c89eb01efd49b01139eaa9f5fa3fe19b
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.15, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Title:
Sailors Navigate Calm Waters in Dinghies

Description:
Two sailors steer small dinghies—a Vortex catamaran (sail number 1067) and a Laser dinghy (sail number GBR 188572)—across calm waters on 19 September 2026, with dense green woodland in the background.

Keywords:
Boat, Boating, Catamaran, Clouds, Dinghy, Estuary, Forest, Laser dinghy, Life jacket, Mast, Outdoor recreation, River, Sailboat, Sailing, Sailor, Shoreline, Trees, Water
</pre>

</details>

---

<a id="model-nativ-community-nemotron-3-nano-omni-30b-a3b-reasoning-4bit"></a>

### nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit

<details>
<summary>Complete evidence: nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type
  nemotronh_nano_omni_reasoning_v3 via nemotron_h_nano_omni)
- *Model load time:* 2.45s
- *Generation time:* 7.65s
- *Total time:* 10.71s
- *Prompt preparation time:* 0.611
- *First-token latency:* 5.5
- *Cleanup time:* 0.159
- *Prompt tokens:* 3,628
- *Generation tokens:* 144
- *Total tokens:* 3,772
- *Prompt throughput (raw):* 660 tok/s
- *Generation throughput (raw):* 91.5 tok/s
- *Peak memory (GB):* 23
- *Active memory (GB):* 20
- *Cache memory (GB):* 0.14
- *Model-load active memory (GB):* 19.7
- *Post-cleanup active memory (GB):* 0.0171
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1388
- *Image placeholders:* 1
- *Processor:* mlx_vlm.models.nemotron_h_nano_omni.processing_nemotron_h_nano_omni.NemotronHNanoOmniProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 75c89904e1c2a045098d9111e7cb250a4d8adcec
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--nativ-community--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit/snapshots/75c89904e1c2a045098d9111e7cb250a4d8adcec
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "repetition_context_size": 20, "repetition_penalty": 1.0, "seed": 0,
  "temperature": 0.6, "top_p": 0.95}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">

Title: Two Sailors Compete in Dinghy Race

Description: On a calm body of water under a partly cloudy sky, a sailor in a blue life jacket steers a Vortex catamaran with sail number 1067, while another sailor in a black jacket operates a Laser dinghy with sail number GBR 188572, both navigating near a forested shoreline.

Keywords: Boat, Boating, Catamaran, Clouds, Dinghy, Estuary, Forest, Laser dinghy, Life jacket, Man, Mast, Outdoor recreation, River, Sailboat, Sailing, Sailor, Shoreline, Sky, Trees, Water
</pre>

</details>

---

<a id="model-mlx-community-idefics3-8b-llama3-bf16"></a>

### mlx-community/Idefics3-8B-Llama3-bf16

<details>
<summary>Complete evidence: mlx-community/Idefics3-8B-Llama3-bf16</summary>

- *Execution:* completed
- *Mechanical checks:* concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* Duplicate keywords: river
- *Arch supported by installed mlx-vlm:* yes (model_type idefics3)
- *Model load time:* 2.20s
- *Generation time:* 7.31s
- *Total time:* 10.30s
- *Prompt preparation time:* 0.773
- *First-token latency:* 1.79
- *Cleanup time:* 0.114
- *Prompt tokens:* 2,619
- *Generation tokens:* 164
- *Total tokens:* 2,783
- *Prompt throughput (raw):* 1,466 tok/s
- *Generation throughput (raw):* 34.4 tok/s
- *Peak memory (GB):* 18
- *Active memory (GB):* 17
- *Cache memory (GB):* 0.43
- *Model-load active memory (GB):* 16.9
- *Post-cleanup active memory (GB):* 0.00182
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1343
- *Image placeholders:* 1
- *Processor:* mlx_vlm.models.idefics3.processing_idefics3.Idefics3Processor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 8c2a30c48864f3251701b7bde40f601d25535098
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Idefics3-8B-Llama3-bf16/snapshots/8c2a30c48864f3251701b7bde40f601d25535098
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|end_of_text\|>

*Readable output:*

<pre class="model-output-readable">
 Title: Laser and Vortex catamaran sailboats on a river with trees.

Description: Two sailboats, a Laser dinghy with sail number GBR 188572 and a Vortex catamaran with sail number 1067, are sailing on a calm river. The Laser dinghy is on the right, and the Vortex catamaran is on the left. Both boats have their sails up and are being steered by their respective sailors. In the background, there is a dense green woodland. The sky is partly cloudy, and the lighting is natural daylight.

Keywords: Laser dinghy, Vortex catamaran, sailboats, river, woodland, sail number GBR 188572, sail number 1067, sailors, sail, river, natural daylight, partly cloudy sky.
</pre>

</details>

---

<a id="model-mlx-community-gemma-4-12b-it-4bit"></a>

### mlx-community/gemma-4-12B-it-4bit

<details>
<summary>Complete evidence: mlx-community/gemma-4-12B-it-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* Duplicate keywords: sailing
- *Arch supported by installed mlx-vlm:* yes (model_type gemma4_unified)
- *Model load time:* 2.18s
- *Generation time:* 3.19s
- *Total time:* 6.20s
- *Prompt preparation time:* 0.823
- *First-token latency:* 0.586
- *Cleanup time:* 0.132
- *Prompt tokens:* 596
- *Generation tokens:* 108
- *Total tokens:* 704
- *Prompt throughput (raw):* 1,016 tok/s
- *Generation throughput (raw):* 58.1 tok/s
- *Peak memory (GB):* 7.6
- *Active memory (GB):* 6.8
- *Cache memory (GB):* 0.32
- *Model-load active memory (GB):* 6.76
- *Post-cleanup active memory (GB):* 0.0131
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1360
- *Image placeholders:* 1
- *Processor:* mlx_vlm.models.gemma4_unified.processing_gemma4_unified.Gemma4UnifiedProcessor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 73bcf09092aa277861d5a191b989b666f7f32e8f
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--gemma-4-12B-it-4bit/snapshots/73bcf09092aa277861d5a191b989b666f7f32e8f
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 1.0, "top_k": 64, "top_p": 0.95}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
Title: Two Sailors Steering Small Dinghies on Calm Water
Description: Two sailors navigate a Vortex catamaran and a Laser dinghy across calm water against a backdrop of dense green woodland. The scene captures a moment of outdoor sailing with the boats positioned in front of a forested shoreline.
Keywords: Boat, Boating, Catamaran, Sailing, Sailor, Laser dinghy, Water, River, Estuary, Forest, Trees, Shoreline, Sailboat, Life jacket, Outdoor recreation, Mast, Clouds, Sailing, Sailing_sport
</pre>

</details>

---

<a id="model-mlx-community-ernie-45-vl-28b-a3b-thinking-4bit"></a>

### mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit

<details>
<summary>Complete evidence: mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Generation was stopped early after sustained repeated
  output; Required labelled fields not detected: title, description, keywords;
  Internal reasoning block appears incomplete
- *Arch supported by installed mlx-vlm:* yes (model_type ernie4_5_moe_vl)
- *Model load time:* 1.60s
- *Generation time:* 6.69s
- *Total time:* 9.03s
- *Prompt preparation time:* 0.732
- *First-token latency:* 1.57
- *Cleanup time:* 0.116
- *Prompt tokens:* 1,639
- *Generation tokens:* 425
- *Total tokens:* 2,064
- *Prompt throughput (raw):* 1,045 tok/s
- *Generation throughput (raw):* 95.2 tok/s
- *Peak memory (GB):* 19
- *Active memory (GB):* 17
- *Cache memory (GB):* 0.29
- *Model-load active memory (GB):* 17.4
- *Post-cleanup active memory (GB):* 0.000625
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* repetition_abort
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1496
- *Processor:* mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl.Ernie4_5_VLProcessor
- *Tokenizer:* mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl.Ernie4_5_VLTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 846ea5576854468f25af3767d769b0805b1b08b6
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--ERNIE-4.5-VL-28B-A3B-Thinking-4bit/snapshots/846ea5576854468f25af3767d769b0805b1b08b6
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "repetition_context_size": 20,
  "repetition_penalty": 1.0, "seed": 0, "temperature": 0.6, "thinking_budget":
  800, "thinking_end_token": "&lt;/think&gt;", "thinking_start_token":
  "&lt;think&gt;", "top_p": 0.95}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Alright, let's get this done. I need to create some metadata for this image, and it's my job to be precise.

First, I need to figure out a good title. "Sailboats on the Water" is too generic. "Two Sailors on Dinghies" is better, but I want something more specific. "Sailors on Catamaran and Laser Dinghy" - that's it! Concise and accurate.

Next, the description. I'll start with the basics: date, time, and location. Then I'll describe the scene. I'll mention the two boats, the sailors, and the setting. The hint about the boats is great. Let me work that in. Okay, I'm going to use the hint about the catamaran and laser. I'll refine that. "On a calm day, two sailors navigate a Vortex catamaran and a Laser dinghy on a lake in the UK." That's good, it's accurate, and it's a good length.

Finally, the keywords. I'll go through the hints and add in some keywords that are relevant. Let's see, "sailboat," "dinghy," "catamaran," "laser," "sailor," "boat," "water," "lake," "sky," "trees," "shoreline," "recreation," "sport," "vessel," "wind," "man," "life jacket," "hull," "rigging," "sail number," "1067," "GBR 188572," "green," "calm," "daytime," "September," "UK," "England," "solent," "estuary," "marina," "hull number," "hull number," "hull number," "hull number," "hull number," "hull number," "hull number," "hull number," "hull
</pre>

</details>

---

<a id="model-mlx-community-fastvlm-05b-bf16"></a>

### mlx-community/FastVLM-0.5B-bf16

<details>
<summary>Complete evidence: mlx-community/FastVLM-0.5B-bf16</summary>

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* Required labelled fields not detected: title, description,
  keywords
- *Arch supported by installed mlx-vlm:* yes (model_type llava_qwen2 via
  fastvlm)
- *Model load time:* 0.65s
- *Generation time:* 2.27s
- *Total time:* 4.13s
- *Prompt preparation time:* 1.2
- *First-token latency:* 0.978
- *Cleanup time:* 0.0915
- *Prompt tokens:* 336
- *Generation tokens:* 51
- *Total tokens:* 387
- *Prompt throughput (raw):* 344 tok/s
- *Generation throughput (raw):* 311 tok/s
- *Peak memory (GB):* 1.8
- *Active memory (GB):* 1.2
- *Cache memory (GB):* 0.02
- *Model-load active memory (GB):* 1.25
- *Post-cleanup active memory (GB):* 0.000936
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1401
- *Image placeholders:* 1
- *Processor:* mlx_vlm.models.fastvlm.processing.FastVLMProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 81ffe929046666c43de53691147b1669ba0f3a4c
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--FastVLM-0.5B-bf16/snapshots/81ffe929046666c43de53691147b1669ba0f3a4c
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
A serene scene of two sailors navigating calm waters in a Vortex catamaran and Laser dinghy, set against a backdrop of dense green woodland, under a clear sky, with a sailboat and a man in a life jacket in the distance.
</pre>

</details>

---

<a id="model-mlx-community-llama-32-11b-vision-instruct-8bit"></a>

### mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

<details>
<summary>Complete evidence: mlx-community/Llama-3.2-11B-Vision-Instruct-8bit</summary>

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Response repeats the same text; Response appears cut off at
  the token limit; Duplicate keywords: adventure, learning, education,
  training, practice, improvement, progress, success, achievement,
  accomplishment, pride, satisfaction, happiness, joy, laughter, smiles,
  gratitude, appreciation, wonder, awe, amazement, enthusiasm, excitement,
  exploration, discovery
- *Arch supported by installed mlx-vlm:* yes (model_type mllama)
- *Model load time:* 1.73s
- *Generation time:* 56.40s
- *Total time:* 58.73s
- *Prompt preparation time:* 0.603
- *First-token latency:* 2.27
- *Cleanup time:* 0.112
- *Prompt tokens:* 308
- *Generation tokens:* 1,000
- *Total tokens:* 1,308
- *Prompt throughput (raw):* 136 tok/s
- *Generation throughput (raw):* 18.7 tok/s
- *Peak memory (GB):* 15
- *Active memory (GB):* 11
- *Cache memory (GB):* 0.40
- *Model-load active memory (GB):* 11.3
- *Post-cleanup active memory (GB):* 0.00389
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1410
- *Image placeholders:* 1
- *Processor:* mlx_vlm.models.mllama.processing_mllama.MllamaProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 8451adc50203b50b8f4199e75e753fb9c06e2af6
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Llama-3.2-11B-Vision-Instruct-8bit/snapshots/8451adc50203b50b8f4199e75e753fb9c06e2af6
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|eot_id\|>

*Readable output:*

<pre class="model-output-readable">
Title: Two Sailors Navigate Calm Waters in a Forested Estuary

Description: On a sunny day in September 2026, two sailors, one in a Vortex catamaran and the other in a Laser dinghy, navigate their boats across the calm waters of an estuary, surrounded by dense green woodland and a blue sky with clouds. The Vortex catamaran, with its distinctive orange and grey sail, is sail number 1067, while the Laser dinghy, with its white sail bearing the GBR 188572 number, is a British vessel.

Keywords: Sailboat, Catamaran, Estuary, Forest, Sailing, Sailors, Vortex, Laser, GBR, 1067, 188572, Sail, Boat, Water, Trees, Clouds, Sky, Blue, Green, Woodland, Nature, Outdoor, Recreation, Sport, Activity, Adventure, Fun, Relaxation, Peaceful, Calm, Serene, Scenic, Breathtaking, Beautiful, Picturesque, Idyllic, Tranquil, Soothing, Uplifting, Inspiring, Motivational, Encouraging, Empowering, Confidence-Boosting, Self-Improvement, Personal-Growth, Development, Learning, Education, Training, Practice, Improvement, Progress, Success, Achievement, Accomplishment, Pride, Satisfaction, Happiness, Joy, Laughter, Smiles, Gratitude, Appreciation, Wonder, Awe, Amazement, Enthusiasm, Excitement, Adventure, Exploration, Discovery, Learning, Education, Training, Practice, Improvement, Progress, Success, Achievement, Accomplishment, Pride, Satisfaction, Happiness, Joy, Laughter, Smiles, Gratitude, Appreciation, Wonder, Awe, Amazement, Enthusiasm, Excitement, Adventure, Exploration, Discovery, Learning, Education, Training, Practice, Improvement, Progress, Success, Achievement, Accomplishment, Pride, Satisfaction, Happiness, Joy, Laughter, Smiles, Gratitude, Appreciation, Wonder, Awe, Amazement, Enthusiasm, Excitement, Adventure, Exploration, Discovery, Learning, Education, Training, Practice, Improvement, Progress, Success, Achievement, Accomplishment, Pride, Satisfaction, Happiness, Joy, Laughter, Smiles, Gratitude, Appreciation, Wonder, Awe, Amazement, Enthusiasm, Excitement, Adventure, Exploration, Discovery, Learning, Education, Training, Practice, Improvement, Progress, Success, Achievement, Accomplishment, Pride, Satisfaction, Happiness, Joy, Laughter, Smiles, Gratitude, Appreciation, Wonder, Awe, Amazement, Enthusiasm, Excitement, Adventure, Exploration, Discovery, Learning, Education, Training, Practice, Improvement, Progress, Success, Achievement, Accomplishment, Pride, Satisfaction, Happiness, Joy, Laughter, Smiles, Gratitude, Appreciation, Wonder, Awe, Amazement, Enthusiasm, Excitement, Adventure, Exploration, Discovery, Learning, Education, Training, Practice, Improvement, Progress, Success, Achievement, Accomplishment, Pride, Satisfaction, Happiness, Joy, Laughter, Smiles, Gratitude, Appreciation, Wonder, Awe, Amazement, Enthusiasm, Excitement, Adventure, Exploration, Discovery, Learning, Education, Training, Practice, Improvement, Progress, Success, Achievement, Accomplishment, Pride, Satisfaction, Happiness, Joy, Laughter, Smiles, Gratitude, Appreciation, Wonder, Awe, Amazement, Enthusiasm, Excitement, Adventure, Exploration, Discovery, Learning, Education, Training, Practice, Improvement, Progress, Success, Achievement, Accomplishment, Pride, Satisfaction, Happiness, Joy, Laughter, Smiles, Gratitude, Appreciation, Wonder, Awe, Amazement, Enthusiasm, Excitement, Adventure, Exploration, Discovery, Learning, Education, Training, Practice, Improvement, Progress, Success, Achievement, Accomplishment, Pride, Satisfaction, Happiness, Joy, Laughter, Smiles, Gratitude, Appreciation, Wonder, Awe, Amazement, Enthusiasm, Excitement, Adventure, Exploration, Discovery, Learning, Education, Training, Practice, Improvement, Progress, Success, Achievement, Accomplishment, Pride, Satisfaction, Happiness, Joy, Laughter, Smiles, Gratitude, Appreciation, Wonder, Awe, Amazement, Enthusiasm, Excitement, Adventure, Exploration, Discovery, Learning, Education, Training, Practice, Improvement, Progress, Success, Achievement, Accomplishment, Pride, Satisfaction, Happiness, Joy, Laughter, Smiles, Gratitude, Appreciation, Wonder, Awe, Amazement, En
</pre>

</details>

---

<a id="model-mlx-community-minicpm-v-46-4bit"></a>

### mlx-community/MiniCPM-V-4.6-4bit

<details>
<summary>Complete evidence: mlx-community/MiniCPM-V-4.6-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Internal reasoning block appears incomplete
- *Arch supported by installed mlx-vlm:* yes (model_type minicpmv4_6)
- *Model load time:* 0.95s
- *Generation time:* 3.06s
- *Total time:* 4.73s
- *Prompt preparation time:* 0.709
- *First-token latency:* 2.04
- *Cleanup time:* 0.106
- *Prompt tokens:* 934
- *Generation tokens:* 84
- *Total tokens:* 1,018
- *Prompt throughput (raw):* 458 tok/s
- *Generation throughput (raw):* 247 tok/s
- *Peak memory (GB):* 3.3
- *Active memory (GB):* 2.2
- *Cache memory (GB):* 0.05
- *Model-load active memory (GB):* 2.16
- *Post-cleanup active memory (GB):* 0.00439
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1374
- *Image placeholders:* 1
- *Processor:* mlx_vlm.models.minicpmv4_6.processing_minicpmv4_6.MiniCPMVProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 86cd463d33a946e4481b77e3c10fc63121b60a19
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--MiniCPM-V-4.6-4bit/snapshots/86cd463d33a946e4481b77e3c10fc63121b60a19
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "repetition_context_size": 20, "repetition_penalty": 1.0, "seed": 0,
  "temperature": 0.7, "top_k": 0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
&lt;think&gt;
Here's the response following the instructions:

Title: Sailing vessels on calm waters

Description: Two sailors are sailing small dinghies, a Vortex catamaran and a Laser dinghy, across calm water with green forest in the background.

Keywords: boats, sailing, dinghy, catamaran, water, nature, forest, sail, person, life jacket, outdoor, recreation
</pre>

<details>
<summary>Exact raw output</summary>

```text
<think>
Here's the response following the instructions:

Title: Sailing vessels on calm waters

Description: Two sailors are sailing small dinghies, a Vortex catamaran and a Laser dinghy, across calm water with green forest in the background.

Keywords: boats, sailing, dinghy, catamaran, water, nature, forest, sail, person, life jacket, outdoor, recreation
```

</details>

</details>

---

<a id="model-mlx-community-molmo2-8b-4bit"></a>

### mlx-community/Molmo2-8B-4bit

<details>
<summary>Complete evidence: mlx-community/Molmo2-8B-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Generation was stopped early after sustained repeated
  output; Duplicate keywords: blue stripes, white boat, blue canopy, white
  hull
- *Arch supported by installed mlx-vlm:* yes (model_type molmo2)
- *Model load time:* 1.17s
- *Generation time:* 6.77s
- *Total time:* 8.66s
- *Prompt preparation time:* 0.707
- *First-token latency:* 2.91
- *Cleanup time:* 0.103
- *Prompt tokens:* 1,526
- *Generation tokens:* 225
- *Total tokens:* 1,751
- *Prompt throughput (raw):* 524 tok/s
- *Generation throughput (raw):* 70.3 tok/s
- *Peak memory (GB):* 8.1
- *Active memory (GB):* 6.5
- *Cache memory (GB):* 1.8
- *Model-load active memory (GB):* 6.45
- *Post-cleanup active memory (GB):* 0.00554
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* repetition_abort
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1344
- *Image placeholders:* 1
- *Processor:* mlx_vlm.models.molmo2.processing.Molmo2Processor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 4fcbe926577641af12aa5086cf5c292db9dd9730
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Molmo2-8B-4bit/snapshots/4fcbe926577641af12aa5086cf5c292db9dd9730
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Catamaran and Laser Dinghy Sail on Calm River

Description: Two sailors navigate small dinghies across a tranquil river, with a Vortex catamaran on the left and a Laser dinghy on the right. The scene features calm waters, dense green woodland, and a clear blue sky with scattered clouds.

Keywords: Boat, Boating, Catamaran, Dinghy, Estuary, Forest, Laser dinghy, Life jacket, Man, Mast, Outdoor recreation, River, Sailboat, Sailing, Sailor, Shoreline, Sky, Trees, Water, Sail number, Vortex, Calm waters, Dense woodland, Clear sky, Scattered clouds, Small dinghies, Two sailors, White hulls, Blue life jackets, White sails, Orange stripes, White text, Blue stripes, White boat, Blue canopy, White hull, Blue stripes, White boat, Blue canopy, White hull, Blue stripes, White boat, Blue canopy, White hull, Blue stripes, White boat, Blue canopy, White hull, Blue stripes, White boat, Blue canopy, White hull,
</pre>

</details>

---

<a id="model-mlx-community-muse-glimmer-30b-optiq-4bit"></a>

### mlx-community/Muse-Glimmer-30B-OptiQ-4bit

<details>
<summary>Complete evidence: mlx-community/Muse-Glimmer-30B-OptiQ-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Unrecognised model control tokens remain visible; Required
  labelled fields not detected: title, description; Response appears cut off
  at the token limit; Conversation-role control tokens remain visible
- *Arch supported by installed mlx-vlm:* yes (model_type muse_glimmer)
- *Model load time:* 3.29s
- *Generation time:* 59.42s
- *Total time:* 63.41s
- *Prompt preparation time:* 0.697
- *First-token latency:* 10.4
- *Cleanup time:* 0.145
- *Prompt tokens:* 4,409
- *Generation tokens:* 1,000
- *Total tokens:* 5,409
- *Prompt throughput (raw):* 424 tok/s
- *Generation throughput (raw):* 20.7 tok/s
- *Peak memory (GB):* 25
- *Active memory (GB):* 24
- *Cache memory (GB):* 0.23
- *Model-load active memory (GB):* 23.9
- *Post-cleanup active memory (GB):* 0.00595
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1544
- *Processor:* mlx_vlm.models.muse_glimmer.processing_muse_glimmer.MuseGlimmerProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 98377360cbc84f982e90336f956b08adb46cad88
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Muse-Glimmer-30B-OptiQ-4bit/snapshots/98377360cbc84f982e90336f956b08adb46cad88
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 1.0, "top_k": 64, "top_p": 0.95}
- *EOS token:* <\|end_of_text\|>

*Readable output:*

<pre class="model-output-readable">
 to=self&lt;|message|&gt;Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-09-19 17:12:46 UTC+01:00

Descriptive hints:
- Description hint: Two sailors steer small dinghies—a Vortex catamaran (sail number 1067) on the left and a Laser dinghy (sail number GBR 188572) on the right—across calm coastal or river waters against a backdrop of dense green woodland.
- Keyword hints: Boat, Boating, Catamaran, Clouds, Dinghy, Estuary, Forest, Laser dinghy, Life jacket, Man, Mast, Outdoor recreation, River, Sailboat, Sailing, Sailor, Shoreline, Sky, Trees, Water

Write:
- a concrete 5-10-word title;
- a 1-2-sentence factual description combining relevant context with the main visible subject, setting, action, lighting, and distinctive details;
- 10-18 unique, comma-separated keywords covering relevant context and visible details.

Return exactly these three sections and nothing else:
Title:
Description:
Keywords:

Need British English.

Title: 5-10 words. Concrete. Maybe "Vortex Catamaran and Laser Dinghy Sailing on Water". That's 8 words? Vortex(1) Catamaran2 and3 Laser4 Dinghy5 Sailing6 on7 Water8. Good.

Description: 1-2 sentence factual description combining relevant context with main visible subject, setting, action, lighting, distinctive details. Need mention capture date/time? It says treat capture date/time as authoritative facts, but do not claim they are visible. So can include date/time in description? Probably yes as context. It says combining relevant context with main visible subject... Setting, action, lighting, distinctive details. Might include date. But not claim visible. Could say Captured on 2026-09-19... That's a fact. Probably acceptable.

The image shows two sailors: left Vortex catamaran sail number 1067 with grey sail with orange stripes, orange hull, sailor in blue jacket red life jacket standing. Right Laser dinghy sail number 188572 GBR white sail with blue corners red sun logo, sailor in black life jacket sitting. Background dense green woodland, shoreline grass, calm water, partly cloudy sky. In background a white motorboat with blue cover and a small white boat with blue trim and buoy.

Need correct conflicts: description hint says Vortex catamaran. Actually Vortex is a single hull? Looks like a small catamaran? Might be a Vortex dinghy? Actually Vortex is a 1-person cat? The boat looks like a small catamaran? The hull looks like a single hull with maybe a small platform? Could be a Vortex 1-person dinghy? Might be a Vortex? Let's assume image evidence: the boat on left has a hull with "VORTEX" and 1067. It appears to be a small trimaran? Or a catamaran? Hints say catamaran. Might keep as small sailing dinghy. Could say Vortex dinghy. Might be safe to say Vortex sailing dinghy. But hint says catamaran. Image shows what looks like a small boat with maybe two hulls? Actually it's a single hull with a small outrigger? Hmm looks like a Vortex class dinghy which is a monohull? Let's keep description as Vortex sailing boat. Might follow hint but image evidence shows it's a single hull with maybe a cat? I think Vortex is a dinghy, not catamaran. Could note as Vortex dinghy. Might be conflict. Prefer image evidence. I'd say Vortex dinghy (sail number 1067). Might keep catamaran? Safer to say small sailing dinghy.

Laser dinghy sail number 188572 GBR. The sail number on sail is 188572, GBR. Good.

Lighting: daylight, partly cloudy.

Provide keywords 10-18 unique comma-separated.

Let's craft description: On 2026-09-19 at 17:12:46 BST, two sailors are on the water under a partly cloudy sky. On the left a white and orange Vortex dinghy numbered 1067 with a grey and orange sail is underway with a sailor standing in a blue jacket and life jacket; on the right a white Laser dinghy numbered GBR 188572 with a white and blue sail is underway with a sailor seated in a black life jacket. Both are on calm water with wooded shoreline
</pre>

<details>
<summary>Exact raw output</summary>

```text
 to=self<|message|>Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-09-19 17:12:46 UTC+01:00

Descriptive hints:
- Description hint: Two sailors steer small dinghies—a Vortex catamaran (sail number 1067) on the left and a Laser dinghy (sail number GBR 188572) on the right—across calm coastal or river waters against a backdrop of dense green woodland.
- Keyword hints: Boat, Boating, Catamaran, Clouds, Dinghy, Estuary, Forest, Laser dinghy, Life jacket, Man, Mast, Outdoor recreation, River, Sailboat, Sailing, Sailor, Shoreline, Sky, Trees, Water

Write:
- a concrete 5-10-word title;
- a 1-2-sentence factual description combining relevant context with the main visible subject, setting, action, lighting, and distinctive details;
- 10-18 unique, comma-separated keywords covering relevant context and visible details.

Return exactly these three sections and nothing else:
Title:
Description:
Keywords:

Need British English.

Title: 5-10 words. Concrete. Maybe "Vortex Catamaran and Laser Dinghy Sailing on Water". That's 8 words? Vortex(1) Catamaran2 and3 Laser4 Dinghy5 Sailing6 on7 Water8. Good.

Description: 1-2 sentence factual description combining relevant context with main visible subject, setting, action, lighting, distinctive details. Need mention capture date/time? It says treat capture date/time as authoritative facts, but do not claim they are visible. So can include date/time in description? Probably yes as context. It says combining relevant context with main visible subject... Setting, action, lighting, distinctive details. Might include date. But not claim visible. Could say Captured on 2026-09-19... That's a fact. Probably acceptable.

The image shows two sailors: left Vortex catamaran sail number 1067 with grey sail with orange stripes, orange hull, sailor in blue jacket red life jacket standing. Right Laser dinghy sail number 188572 GBR white sail with blue corners red sun logo, sailor in black life jacket sitting. Background dense green woodland, shoreline grass, calm water, partly cloudy sky. In background a white motorboat with blue cover and a small white boat with blue trim and buoy.

Need correct conflicts: description hint says Vortex catamaran. Actually Vortex is a single hull? Looks like a small catamaran? Might be a Vortex dinghy? Actually Vortex is a 1-person cat? The boat looks like a small catamaran? The hull looks like a single hull with maybe a small platform? Could be a Vortex 1-person dinghy? Might be a Vortex? Let's assume image evidence: the boat on left has a hull with "VORTEX" and 1067. It appears to be a small trimaran? Or a catamaran? Hints say catamaran. Might keep as small sailing dinghy. Could say Vortex dinghy. Might be safe to say Vortex sailing dinghy. But hint says catamaran. Image shows what looks like a small boat with maybe two hulls? Actually it's a single hull with a small outrigger? Hmm looks like a Vortex class dinghy which is a monohull? Let's keep description as Vortex sailing boat. Might follow hint but image evidence shows it's a single hull with maybe a cat? I think Vortex is a dinghy, not catamaran. Could note as Vortex dinghy. Might be conflict. Prefer image evidence. I'd say Vortex dinghy (sail number 1067). Might keep catamaran? Safer to say small sailing dinghy.

Laser dinghy sail number 188572 GBR. The sail number on sail is 188572, GBR. Good.

Lighting: daylight, partly cloudy.

Provide keywords 10-18 unique comma-separated.

Let's craft description: On 2026-09-19 at 17:12:46 BST, two sailors are on the water under a partly cloudy sky. On the left a white and orange Vortex dinghy numbered 1067 with a grey and orange sail is underway with a sailor standing in a blue jacket and life jacket; on the right a white Laser dinghy numbered GBR 188572 with a white and blue sail is underway with a sailor seated in a black life jacket. Both are on calm water with wooded shoreline
```

</details>

</details>

---

<a id="model-mlx-community-qwen2-vl-7b-instruct-4bit"></a>

### mlx-community/Qwen2-VL-7B-Instruct-4bit

<details>
<summary>Complete evidence: mlx-community/Qwen2-VL-7B-Instruct-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Response repeats the same text; Generation was stopped early
  after sustained repeated output; Duplicate keywords: trees, forest, mast,
  life jacket, river, estuary, sky, shoreline, sail, boat, boating
- *Arch supported by installed mlx-vlm:* yes (model_type qwen2_vl)
- *Model load time:* 0.81s
- *Generation time:* 41.67s
- *Total time:* 43.45s
- *Prompt preparation time:* 0.962
- *First-token latency:* 38.3
- *Cleanup time:* 0.116
- *Prompt tokens:* 16,560
- *Generation tokens:* 225
- *Total tokens:* 16,785
- *Prompt throughput (raw):* 432 tok/s
- *Generation throughput (raw):* 89.5 tok/s
- *Peak memory (GB):* 9.3
- *Active memory (GB):* 4.7
- *Cache memory (GB):* 2.4
- *Model-load active memory (GB):* 4.68
- *Post-cleanup active memory (GB):* 0.00736
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* repetition_abort
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1436
- *Processor:* mlx_vlm.models.qwen2_vl.processing_qwen2_vl.Qwen2VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 1c638e970be36948e087e77d103b9147e1a9a1dd
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen2-VL-7B-Instruct-4bit/snapshots/1c638e970be36948e087e77d103b9147e1a9a1dd
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Sailing Adventure

Description: Two sailors navigate a Vortex catamaran (sail number 1067) and a Laser dinghy (sail number GBR 188572) across calm waters, with dense green woodland in the background.

Keywords: Sailing, Catamaran, Laser dinghy, Vortex, Sail number, Water, Trees, Forest, Woodland, Adventure, Outdoor, Action, Sailboat, Mast, Life jacket, River, Estuary, Sky, Clouds, Trees, Shoreline, Sail, Boat, Boating, Estuary, Forest, Life jacket, Mast, River, Sky, Trees, Shoreline, Sail, Boat, Boating, Estuary, Forest, Life jacket, Mast, River, Sky, Trees, Shoreline, Sail, Boat, Boating, Estuary, Forest, Life jacket, Mast, River, Sky, Trees, Shoreline, Sail, Boat, Boating, Estuary, Forest, Life jacket, Mast, River, Sky, Trees, Shoreline, Sail, Boat,
</pre>

</details>

---

<a id="model-mlx-community-smolvlm-256m-instruct-4bit"></a>

### mlx-community/SmolVLM-256M-Instruct-4bit

<details>
<summary>Complete evidence: mlx-community/SmolVLM-256M-Instruct-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* Required labelled fields not detected: title, description,
  keywords
- *Arch supported by installed mlx-vlm:* yes (model_type idefics3)
- *Model load time:* 0.57s
- *Generation time:* 1.17s
- *Total time:* 2.63s
- *Prompt preparation time:* 0.883
- *First-token latency:* 0.208
- *Cleanup time:* 0.0895
- *Prompt tokens:* 1,212
- *Generation tokens:* 38
- *Total tokens:* 1,250
- *Prompt throughput (raw):* 5,818 tok/s
- *Generation throughput (raw):* 322 tok/s
- *Peak memory (GB):* 1.1
- *Active memory (GB):* 0.16
- *Cache memory (GB):* 0.10
- *Model-load active memory (GB):* 0.16
- *Post-cleanup active memory (GB):* 0.0103
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1338
- *Image placeholders:* 1
- *Processor:* mlx_vlm.models.idefics3.processing_idefics3.Idefics3Processor
- *Tokenizer:* transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 69cb5195f414ceb6398c5581254673d2c6c8d0d8
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--SmolVLM-256M-Instruct-4bit/snapshots/69cb5195f414ceb6398c5581254673d2c6c8d0d8
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
 A 5-10-word, 1-2-sentence, factual description combining relevant context with the main visible subject, setting, action, lighting, and distinctive details.
</pre>

</details>

---

<a id="model-mlx-community-x-reasoner-7b-8bit"></a>

### mlx-community/X-Reasoner-7B-8bit

<details>
<summary>Complete evidence: mlx-community/X-Reasoner-7B-8bit</summary>

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Generation was stopped early after sustained repeated
  output; Duplicate keywords: blue and white forest, blue and white sky, blue
  and white water, blue and white sail
- *Arch supported by installed mlx-vlm:* yes (model_type qwen2_5_vl)
- *Model load time:* 1.21s
- *Generation time:* 17.66s
- *Total time:* 20.22s
- *Prompt preparation time:* 1.34
- *First-token latency:* 12.4
- *Cleanup time:* 0.186
- *Prompt tokens:* 16,560
- *Generation tokens:* 250
- *Total tokens:* 16,810
- *Prompt throughput (raw):* 1,340 tok/s
- *Generation throughput (raw):* 56.2 tok/s
- *Peak memory (GB):* 14
- *Active memory (GB):* 9.5
- *Cache memory (GB):* 2.4
- *Model-load active memory (GB):* 9.46
- *Post-cleanup active memory (GB):* 0.0109
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* repetition_abort
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1436
- *Processor:* mlx_vlm.models.qwen2_5_vl.processing_qwen2_5_vl.Qwen2_5_VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 21732e74613b465bc98e9d5ec210aba5c7adbcc1
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--X-Reasoner-7B-8bit/snapshots/21732e74613b465bc98e9d5ec210aba5c7adbcc1
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
Sailing Catamaran and Laser Dinghy on Calm Waters

Description:
Two sailors navigate a Vortex catamaran (sail number 1067) and a Laser dinghy (sail number GBR 188572) across tranquil waters, set against a backdrop of dense green woodland under a partly cloudy sky.

Keywords: Sailing, Catamaran, Laser dinghy, Vortex, Sail number 1067, Sail number GBR 188572, Calm waters, Distant shoreline, Forest, Cloudy sky, Life jackets, Outdoor recreation, River, Shoreline, Trees, Water, Man, Mast, Sunburst logo, Blue and white sails, Orange and black sail, Blue and white catamaran, White Laser dinghy, Blue and white forest, Blue and white sky, Blue and white water, Blue and white sail, Blue and white forest, Blue and white sky, Blue and white water, Blue and white sail, Blue and white forest, Blue and white sky, Blue and white water, Blue and white sail, Blue and white forest, Blue and white sky, Blue and white water, Blue and white sail,
</pre>

</details>

---

<a id="model-mlx-community-gemma-3n-e4b-it-4bit"></a>

### mlx-community/gemma-3n-E4B-it-4bit

<details>
<summary>Complete evidence: mlx-community/gemma-3n-E4B-it-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* Required labelled fields not detected: title, description,
  keywords
- *Arch supported by installed mlx-vlm:* yes (model_type gemma3n)
- *Model load time:* 2.03s
- *Generation time:* 5.35s
- *Total time:* 8.06s
- *Prompt preparation time:* 0.651
- *First-token latency:* 1.52
- *Cleanup time:* 0.137
- *Prompt tokens:* 590
- *Generation tokens:* 196
- *Total tokens:* 786
- *Prompt throughput (raw):* 388 tok/s
- *Generation throughput (raw):* 60.6 tok/s
- *Peak memory (GB):* 6.9
- *Active memory (GB):* 5.8
- *Cache memory (GB):* 0.05
- *Model-load active memory (GB):* 5.83
- *Post-cleanup active memory (GB):* 0.0126
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1363
- *Image placeholders:* 1
- *Processor:* mlx_vlm.models.gemma3n.processing_gemma3n.Gemma3nProcessor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 505468a22e5703ff090e222aae9beedec49b383f
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--gemma-3n-E4B-it-4bit/snapshots/505468a22e5703ff090e222aae9beedec49b383f
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.0, "top_k": 64, "top_p": 0.95}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
Two sailors are engaged in a sailing competition on a calm body of water, likely an estuary or a sheltered bay, surrounded by lush green vegetation. On the left, a vibrant orange and white catamaran, identified by the sail number 1067 and the name "VORTEX," is being steered by a person wearing a blue jacket and dark pants. The catamaran has a grey and orange striped sail with the sail number prominently displayed. On the right, a white and blue Laser dinghy, bearing the sail number GBR 188572, is being sailed by a person in a dark jacket and dark pants. The Laser has a white sail with a red logo and the British flag. Both boats are relatively close to each other, suggesting a close race. The water is calm, reflecting the boats and the surrounding greenery. The sky is partly cloudy, providing ample daylight for the sailing activity. The overall scene depicts a competitive yet serene sporting event.
</pre>

</details>

---

<a id="model-mlx-community-granite-vision-32-2b-nvfp4"></a>

### mlx-community/granite-vision-3.2-2b-nvfp4

<details>
<summary>Complete evidence: mlx-community/granite-vision-3.2-2b-nvfp4</summary>

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* Required labelled fields not detected: keywords
- *Arch supported by installed mlx-vlm:* yes (model_type granite_vision)
- *Model load time:* 0.45s
- *Generation time:* 3.31s
- *Total time:* 4.78s
- *Prompt preparation time:* 1.01
- *First-token latency:* 1.67
- *Cleanup time:* 0.0998
- *Prompt tokens:* 5,581
- *Generation tokens:* 102
- *Total tokens:* 5,683
- *Prompt throughput (raw):* 3,339 tok/s
- *Generation throughput (raw):* 141 tok/s
- *Peak memory (GB):* 4.2
- *Active memory (GB):* 2.4
- *Cache memory (GB):* 0.94
- *Model-load active memory (GB):* 2.36
- *Post-cleanup active memory (GB):* 0.015
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1483
- *Image placeholders:* 1
- *Processor:* mlx_vlm.models.granite_vision.processing_granite_vision.GraniteVisionProcessor
- *Tokenizer:* transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* f0c70965cdae87ebf38e84ff958e841ad943e8e6
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--granite-vision-3.2-2b-nvfp4/snapshots/f0c70965cdae87ebf38e84ff958e841ad943e8e6
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|end_of_text\|>

*Readable output:*

<pre class="model-output-readable">
Title: "Harmony on the Water"
Description: Two sailors navigate their small dinghies, a Vortex catamaran and a Laser dinghy, across calm waters, with a dense green woodland backdrop. The scene is set in an estuary, with a clear sky overhead. The sailors are equipped with life jackets, and the boats are moored in a serene, outdoor recreational setting.
</pre>

</details>

---

<a id="model-mlx-community-llm-jp-4-vl-9b-mlx-4bit"></a>

### mlx-community/llm-jp-4-vl-9b-mlx-4bit

<details>
<summary>Complete evidence: mlx-community/llm-jp-4-vl-9b-mlx-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Unrecognised model control tokens remain visible; Required
  labelled fields not detected: title, description, keywords
- *Arch supported by installed mlx-vlm:* yes (model_type llmjpvl)
- *Model load time:* 1.13s
- *Generation time:* 1.64s
- *Total time:* 3.45s
- *Prompt preparation time:* 0.688
- *First-token latency:* 0.832
- *Cleanup time:* 0.17
- *Prompt tokens:* 2,197
- *Generation tokens:* 16
- *Total tokens:* 2,213
- *Prompt throughput (raw):* 2,641 tok/s
- *Generation throughput (raw):* 111 tok/s
- *Peak memory (GB):* 6.7
- *Active memory (GB):* 5.7
- *Cache memory (GB):* 0.35
- *Model-load active memory (GB):* 5.7
- *Post-cleanup active memory (GB):* 0.0154
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1565
- *Image placeholders:* 1
- *Processor:* transformers_modules._9c056d48b1e611dc586139a5deb927ae363cfe6f.0e62407644efd7c3.processing_llmjpvl.LLMjpVLProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 9c056d48b1e611dc586139a5deb927ae363cfe6f
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--llm-jp-4-vl-9b-mlx-4bit/snapshots/9c056d48b1e611dc586139a5deb927ae363cfe6f
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|return\|>

*Readable output:*

<pre class="model-output-readable">
&lt;|channel|&gt; analysis&lt;|message|&gt; The image shows two small sailboats racing on a river.
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|channel|> analysis<|message|> The image shows two small sailboats racing on a river.
```

</details>

</details>

---

<a id="model-mlx-community-nanollava-15-4bit"></a>

### mlx-community/nanoLLaVA-1.5-4bit

<details>
<summary>Complete evidence: mlx-community/nanoLLaVA-1.5-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* Required labelled fields not detected: title, description,
  keywords
- *Arch supported by installed mlx-vlm:* yes (model_type llava-qwen2 via
  llava_bunny)
- *Model load time:* 0.72s
- *Generation time:* 1.02s
- *Total time:* 2.41s
- *Prompt preparation time:* 0.667
- *First-token latency:* 0.208
- *Cleanup time:* 0.116
- *Prompt tokens:* 332
- *Generation tokens:* 21
- *Total tokens:* 353
- *Prompt throughput (raw):* 1,599 tok/s
- *Generation throughput (raw):* 164 tok/s
- *Peak memory (GB):* 1.4
- *Active memory (GB):* 0.61
- *Cache memory (GB):* 0.28
- *Model-load active memory (GB):* 0.614
- *Post-cleanup active memory (GB):* 0.016
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1392
- *Image placeholders:* 1
- *Processor:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 5240204744963d72823e5de933c528c4aa82dfca
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--nanoLLaVA-1.5-4bit/snapshots/5240204744963d72823e5de933c528c4aa82dfca
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
"Boating in the Countryside: A Glimpse of Sailboats and Forests"
</pre>

</details>

---

<a id="model-mlx-community-internvl35-1b-4bit"></a>

### mlx-community/InternVL3_5-1B-4bit

<details>
<summary>Complete evidence: mlx-community/InternVL3_5-1B-4bit</summary>

- *Execution:* crashed
- *Mechanical checks:* not assessed
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* actionable_failure
- *Observations:* none
- *Failure phase:* model_load
- *Error stage:* Unsupported Arch
- *Error code:* MLX_VLM_MODEL_LOAD_UNSUPPORTED_ARCH
- *Error type:* ValueError
- *Error package:* mlx-vlm
- *Error message:* Model loading failed: Model type internvl not supported.
  Error: No module named 'mlx_vlm.speculative.drafters.internvl'
- *Root exception type:* ValueError
- *Root exception module:* builtins
- *Root exception message:* Model type internvl not supported. Error: No
  module named 'mlx_vlm.speculative.drafters.internvl'
- *Arch supported by installed mlx-vlm:* no (model_type internvl)
- *Model load time:* 0.13s
- *Generation time:* -
- *Total time:* 0.13s
- *Prompt preparation time:* -
- *First-token latency:* -
- *Cleanup time:* 0.0824
- *Prompt tokens:* -
- *Generation tokens:* -
- *Total tokens:* -
- *Prompt throughput (raw):* -
- *Generation throughput (raw):* -
- *Peak memory (GB):* -
- *Active memory (GB):* -
- *Cache memory (GB):* -
- *Model-load active memory (GB):* -
- *Post-cleanup active memory (GB):* 0.00244
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* exception
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* not captured
- *Processor:* not captured
- *Tokenizer:* not captured
- *Requested model revision:* not requested
- *Resolved model revision:* f9d179a8be8ac53e96c6ee5cce8493856d4b8f09
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--InternVL3_5-1B-4bit/snapshots/f9d179a8be8ac53e96c6ee5cce8493856d4b8f09
- *Generation settings:* not captured
- *EOS token:* not captured

#### Complete traceback

```python
Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14636, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13511, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 841, in _typed_mlx_vlm_load
    loaded: tuple[nn.Module, ProcessorMixin] = _mlx_vlm_load(
                                               ~~~~~~~~~~~~~^
        path_or_hf_repo=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1307, in load
    model = load_model(model_path, lazy, strict=strict, **kwargs)
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 965, in load_model
    model_class, _ = get_model_and_args(config=config, model_path=model_path)
                     ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 785, in get_model_and_args
    raise ValueError(msg)
ValueError: Model type internvl not supported. Error: No module named 'mlx_vlm.speculative.drafters.internvl'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 15758, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14651, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: Model type internvl not supported. Error: No module named 'mlx_vlm.speculative.drafters.internvl'

```

#### Captured upstream output

```text
=== STDERR ===
[23:19:03] INFO     Loading model weights and processor...
Fetching 14 files:   0%|          | 0/14 [00:00<?, ?it/s]
Fetching 14 files: 100%|██████████| 14/14 [00:00<00:00, 4677.79it/s]
ERROR:root:Model type internvl not supported. Error: No module named 'mlx_vlm.speculative.drafters.internvl'
[23:19:03] DEBUG    HF Cache Info for mlx-community/InternVL3_5-1B-4bit: size=1046.5 MB, files=16
```

</details>

---
